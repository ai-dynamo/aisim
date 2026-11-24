# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``aisim data`` (fetch, status, clear, validate), ``aisim fetch`` and ``aisim init``: a thin command line
over ``aisim_core.data`` and the Hugging Face model-metadata cache (``aisim_core.sdk.hf_model_files``).

Selection, revision pinning, caching and verification all live in the library;
this module only maps flags to library calls and library results to JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from aisim_core.data.config import (
    ConfigError,
    InitConfig,
    config_path,
    load_config,
    resolve_source,
    save_config,
    shipped_pin,
)
from aisim_core.data.fetch import (
    Cache,
    DataFetchError,
    HfTransport,
    RevisionError,
    Selection,
    SelectionError,
    Transport,
    fetch,
    plan,
)
from aisim_core.sdk import hf_model_files

from .data_parser import (  # noqa: F401
    _PREDICTION_PURPOSE,
    MODEL_CONFIG_CATEGORY,
    PURPOSES,
    SUPPORTED_CATEGORIES,
    add_data_parser,
)

EXIT_FAILURE = 1
EXIT_USAGE = 2


class UnsupportedSelectionError(SelectionError):
    code = "unsupported"


def make_transport() -> Transport:
    """Transport factory; tests patch this to inject a fake."""
    return HfTransport()


def build_selection(args: argparse.Namespace) -> Selection:
    if args.category not in SUPPORTED_CATEGORIES:
        raise UnsupportedSelectionError(
            f"category {args.category!r} is not supported by 'data fetch' yet; "
            f"supported categories: {', '.join(SUPPORTED_CATEGORIES)}"
        )
    if args.purpose in ("validation", "results"):
        raise UnsupportedSelectionError(
            f"purpose {args.purpose!r} is not supported by 'data fetch' yet; use 'prediction' or 'full-evidence'"
        )
    if args.all and args.category != "op-latency":
        raise UnsupportedSelectionError(
            f"--all is only available for op-latency; {args.category} downloads are always selective "
            "(--configuration or --hardware, plus --model)"
        )
    purpose = None
    if args.purpose == "prediction":
        purpose = _PREDICTION_PURPOSE.get(args.category)
        if purpose is None:
            raise UnsupportedSelectionError(
                f"purpose 'prediction' has no data for category {args.category!r}; use 'full-evidence'"
            )
    configurations = tuple(c.strip() for raw in args.configuration for c in raw.split(",") if c.strip())
    if args.configuration and not configurations:  # e.g. --configuration ',': never widen it to the whole category
        raise UsageError("--configuration needs at least one configuration ID")
    # No selector at all means the whole op-latency database; other categories stay selective.
    whole = args.category == "op-latency" and (
        args.all
        or not (
            configurations
            or args.hardware
            or args.model
            or args.framework
            or args.framework_version
            or args.operation_family
        )
    )
    selection = Selection(
        category=args.category,
        configuration_ids=configurations,
        model=args.model,
        hardware=args.hardware,
        framework=args.framework,
        framework_version=args.framework_version,
        operation_family=args.operation_family,
        purpose=purpose,
        select_all=whole,
    )
    selection.validate()
    return selection


LARGE_DOWNLOAD_BYTES = 256 * 1024 * 1024
_NAME = "aisim data"


def _sha_dirs(cache: Cache) -> list[tuple[str, str, Path]]:
    """``(repo, sha, path)`` of every cached revision."""
    found = []
    if not cache.root.is_dir():
        return found
    for repo_dir in sorted(p for p in cache.root.iterdir() if p.is_dir() and "__" in p.name):
        for sha_dir in sorted(p for p in repo_dir.iterdir() if p.is_dir() and re.fullmatch(r"[0-9a-f]{40}", p.name)):
            found.append((repo_dir.name.replace("__", "/", 1), sha_dir.name, sha_dir))
    return found


def _dir_size(path: Path) -> int:
    total = 0
    for dirpath, _, names in os.walk(path):
        for name in names:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                pass
    return total


def _mib(n: int) -> str:
    return f"{n / (1 << 20):.1f} MiB"


def _max_bytes(args: argparse.Namespace) -> int | None:
    if getattr(args, "no_limit", False) or not os.environ.get("AISIM_DATA_MAX_BYTES"):
        return None
    return int(os.environ["AISIM_DATA_MAX_BYTES"])


def _download(selection: Selection, args: argparse.Namespace, repo: str, revision: str) -> dict[str, Any]:
    """Plan, announce a large download, then download. No byte cap unless AISIM_DATA_MAX_BYTES is set."""
    cache = Cache(Path(args.cache_dir) if args.cache_dir else None)
    transport = None if args.offline else make_transport()
    common = {"repo": repo, "transport": transport, "cache": cache, "offline": args.offline}
    cap = _max_bytes(args)
    planned = plan(selection, revision=revision, max_bytes=cap, **common)  # planning downloads metadata: bound it too
    if getattr(args, "dry_run", False):
        return {"dry_run": True, **planned.to_dict()}
    if not args.offline and planned.uncached_bytes >= LARGE_DOWNLOAD_BYTES:
        sys.stderr.write(
            f"{_NAME}: downloading {_mib(planned.uncached_bytes)} in {len(planned.uncached)} files "
            f"from {repo}@{planned.revision[:12]}\n"
        )
    # The plan resolved the revision; reuse the SHA so the download needs no second lookup. When the plan followed a
    # retired repository to its successor, the SHA belongs to the successor and fetch() would take it verbatim from
    # the retired name, without the alias and its path mapping. Keep the requested revision for the source and hand
    # the planned commit over separately, so the download is the commit that was planned and budgeted.
    # The metadata the plan downloaded is spent: the rest of the allowance bounds the artifacts.
    remaining = None if cap is None else max(cap - planned.metadata_bytes, 0)
    if planned.repo == repo:
        result = fetch(selection, revision=planned.revision, max_bytes=remaining, **common)
    else:
        result = fetch(selection, revision=revision, pinned_sha=planned.revision, max_bytes=remaining, **common)
    result.plan.requested_revision = revision
    result.plan.metadata_bytes += planned.metadata_bytes
    return {"dry_run": False, **result.to_dict()}


def run_fetch(args: argparse.Namespace) -> int:
    if args.category == MODEL_CONFIG_CATEGORY:
        return run_fetch_model_configs(args)
    if args.refresh:
        raise UsageError(f"--refresh only applies to --category {MODEL_CONFIG_CATEGORY}")
    saved = load_config(config_path(cache_dir=args.cache_dir))
    repo, revision = resolve_source(args.repo, args.revision, config=saved)
    selection = build_selection(args)
    payload = _download(selection, args, repo, revision)
    _emit(payload)
    return 0


def run_init(args: argparse.Namespace) -> int:
    mode = "all" if args.all else "hardware" if args.hardware else "on-demand" if args.on_demand else None
    if mode is None:
        if not sys.stdin.isatty():
            raise UsageError(
                "init needs a terminal to ask which op data to install; without one pass --all (whole op "
                "database now), --on-demand (download when a run needs it) or --hardware H (one platform now)"
            )
        mode = _prompt_mode()
    if not args.offline:
        _hf_token_step()
    path = config_path(cache_dir=args.cache_dir)
    saved = load_config(path)
    repo, revision = args.repo, args.revision
    if saved:  # an omitted flag keeps the saved source; a revision belongs to its repository
        if repo is None and revision is None:
            repo, revision = saved.repo, saved.revision
        elif repo is not None and revision is None and repo == saved.repo:
            revision = saved.revision
        elif repo is None and revision is not None:
            repo = saved.repo
    config = InitConfig(mode=mode, hardware=args.hardware, repo=repo, revision=revision)
    payload: dict[str, Any] = {"mode": mode, "config": str(path)}
    if mode != "on-demand":
        repo, revision = resolve_source(args.repo, args.revision, config=saved)
        selection = Selection(
            category="op-latency", hardware=args.hardware, purpose="operation-tables", select_all=mode == "all"
        )
        ns = argparse.Namespace(cache_dir=args.cache_dir, offline=args.offline, no_limit=args.no_limit, dry_run=False)
        payload["fetch"] = _download(selection, ns, repo, revision)
        # Save the source the download used, with the revision resolved to its commit, so later runs select
        # exactly that data (and an offline fetch needs no lookup) even with no flag or saved source.
        config = replace(config, repo=repo, revision=payload["fetch"]["plan"]["revision"])
    save_config(config, path)  # saved only after the download succeeded
    if mode != "on-demand" and not args.no_models:
        # Best effort: a model that is gated, denied or unreachable is reported, never a failed init; the loader
        # fetches it through the same cache the next time a run needs it.
        payload["model_configs"] = prefetch_model_configs(hf_model_files.load_manifest(), args=args)
        _warn_about_models(payload["model_configs"])
    _emit(payload)
    return 0


def _model_config_cache(args: argparse.Namespace) -> hf_model_files.ModelFileCache:
    """The model-metadata cache: ``model-configs`` inside the data cache (``--cache-dir`` or ``AISIM_DATA_CACHE``)."""
    return hf_model_files.ModelFileCache(Cache(Path(args.cache_dir) if args.cache_dir else None).root / "model-configs")


def prefetch_model_configs(
    models: Sequence[str],
    *,
    args: argparse.Namespace,
    revision: str | None = None,
    dry_run: bool = False,
    refresh: bool = False,
) -> dict[str, Any]:
    """Cache the Hugging Face metadata files of ``models`` and report each model's outcome.

    A model AISim bundles needs nothing. A gated, denied or missing model is ``skipped`` and an unreachable one
    ``failed``, both with a reason; neither stops the others. ``--offline`` and ``--dry-run`` open no connection.
    """
    from aisim_core.sdk.common import DefaultHFModels
    from aisim_core.sdk.utils import model_files_to_cache

    cache = _model_config_cache(args)
    offline = bool(args.offline) or hf_model_files.offline_requested() or dry_run
    transport = None if offline else hf_model_files.make_transport()
    if not offline and models:
        sys.stderr.write(f"{_NAME}: fetching {len(models)} model config(s) from Hugging Face into {cache.root}\n")
    entries = []
    for model in models:
        entry = hf_model_files.prefetch_model(
            model,
            revision=revision,
            bundled=DefaultHFModels,
            extra_files=model_files_to_cache,
            transport=transport,
            cache=cache,
            offline=offline,
            refresh=refresh,
        )
        if dry_run and entry.status == "missing":
            entry.status, entry.reason = "to-fetch", None
        entries.append(entry)
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.status] = counts.get(entry.status, 0) + 1
    return {
        "cache": str(cache.root),
        "dry_run": dry_run,
        "offline": offline,
        "counts": counts,
        "models": [entry.to_dict() for entry in entries],
    }


def _warn_about_models(result: dict[str, Any]) -> None:
    for entry in result["models"]:
        if entry["status"] in ("skipped", "failed", "missing"):
            sys.stderr.write(f"{_NAME}: warning: model config {entry['model']} {entry['status']}: {entry['reason']}\n")
    counts = result["counts"]
    if counts.get("failed") or counts.get("missing"):
        sys.stderr.write(
            f"{_NAME}: warning: {counts.get('failed', 0) + counts.get('missing', 0)} model config(s) are not cached; "
            f"a run downloads them when it needs them, or retry with "
            f"`aisim data fetch --category {MODEL_CONFIG_CATEGORY}`\n"
        )


def run_fetch_model_configs(args: argparse.Namespace) -> int:
    """``aisim data fetch --category model-config``: the shipped manifest, or the models named by ``--model``."""
    for flag in ("repo", "hardware", "framework", "framework_version", "operation_family"):
        if getattr(args, flag):
            raise UsageError(
                f"--{flag.replace('_', '-')} does not apply to --category {MODEL_CONFIG_CATEGORY}; "
                "select models with --model (comma-separated) and --revision"
            )
    if args.all or args.configuration:
        raise UsageError(f"--all and --configuration do not apply to --category {MODEL_CONFIG_CATEGORY}")
    models = tuple(m.strip() for m in (args.model or "").split(",") if m.strip())
    if args.model is not None and not models:
        raise UsageError("--model needs at least one Hugging Face model ID")
    if args.revision and len(models) != 1:
        raise UsageError("--revision names one model's commit: pass exactly one --model with it")
    result = prefetch_model_configs(
        models or hf_model_files.load_manifest(),
        args=args,
        revision=args.revision,
        dry_run=args.dry_run,
        refresh=args.refresh,
    )
    _emit({"category": MODEL_CONFIG_CATEGORY, **result})
    _warn_about_models(result)
    broken = result["counts"].get("failed", 0) + (0 if args.dry_run else result["counts"].get("missing", 0))
    return EXIT_FAILURE if broken else 0


_TOKEN_URL = "https://huggingface.co/settings/tokens"


def _hf_token() -> str | None:
    """The credential ``huggingface_hub`` would send: ``HF_TOKEN`` or a saved login."""
    from huggingface_hub import get_token

    return get_token()


def _read_token() -> str:
    import getpass

    return getpass.getpass("Paste a token to save it (input is hidden), or press Enter to continue without one: ")


def _save_token(token: str) -> None:
    """Save through Hugging Face's own login, so every Hugging Face tool and later AISim downloads use it."""
    import contextlib

    from huggingface_hub import login

    with contextlib.redirect_stdout(sys.stderr):  # login narrates on stdout, which carries only our JSON
        login(token=token, add_to_git_credential=False)


def _hf_token_step() -> None:
    """Before any download: use an existing Hugging Face token, or explain the rate limit and offer to save one.

    The token never enters AISim's config, logs or output; it goes only to Hugging Face's own token store.
    """
    out = sys.stderr
    if _hf_token():
        out.write("Using your Hugging Face token: authenticated downloads have much higher rate limits.\n")
        return
    out.write(
        "No Hugging Face token found. Anonymous downloads are rate limited (HTTP 429): the op database is\n"
        "thousands of files and can stall for minutes. A free read-only token avoids this:\n"
        f"  {_TOKEN_URL}\n"
    )
    if not sys.stdin.isatty():
        out.write("Set HF_TOKEN or run `hf auth login` to use one; continuing without it.\n")
        return
    try:
        token = _read_token().strip()
    except EOFError:  # input closed: same as pressing Enter
        return
    if not token:
        return
    try:
        _save_token(token)
    except Exception as exc:  # a rejected token or no network: the download still works anonymously
        out.write(f"The token was not saved ({type(exc).__name__}); continuing without it.\n")
        return
    out.write("Token saved to your Hugging Face login; AISim and other Hugging Face tools will use it.\n")


def _prompt_mode() -> str:
    sys.stderr.write("How should AISim get op-latency data?\n  1) op database now\n  2) op data on demand\n")
    while True:
        sys.stderr.write("Choose 1 or 2: ")
        sys.stderr.flush()
        answer = sys.stdin.readline()
        if not answer:  # EOF: never loop forever
            raise UsageError("no answer; pass --all, --on-demand or --hardware H")
        if answer.strip() in ("1", "2"):
            return "all" if answer.strip() == "1" else "on-demand"


def run_status(args: argparse.Namespace) -> int:
    cache = Cache(Path(args.cache_dir) if args.cache_dir else None)
    path = config_path(cache_dir=args.cache_dir)
    saved = load_config(path)
    repo, revision = resolve_source(args.repo, args.revision, config=saved)
    shipped_repo, shipped_revision = shipped_pin()
    revisions = []
    selection = Selection(category="op-latency", purpose="operation-tables", select_all=True)
    for cached_repo, sha, location in _sha_dirs(cache):
        entry: dict[str, Any] = {"repo": cached_repo, "revision": sha, "size_bytes": _dir_size(location)}
        entry["pinned"] = (cached_repo, sha) == (repo, revision)
        try:
            offline_plan = plan(selection, repo=cached_repo, revision=sha, cache=cache, offline=True)
            entry["files_cached"] = sum(1 for f in offline_plan.files if f.cached)
            entry["files_missing"] = len(offline_plan.missing)
            entry["missing"] = offline_plan.missing[:20]
        except DataFetchError as exc:
            entry["error"] = str(exc)
        revisions.append(entry)
    pinned_cached = any(r["pinned"] for r in revisions)
    models = _model_config_cache(args)
    payload = {
        "pin": {
            "repo": repo,
            "revision": revision,
            "shipped": {"repo": shipped_repo, "revision": shipped_revision},
            "cached": pinned_cached,
            "note": None if re.fullmatch(r"[0-9a-f]{40}", revision) else "a branch or tag resolves online only",
        },
        "cache": {"path": str(cache.root), "size_bytes": _dir_size(cache.root) if cache.root.is_dir() else 0},
        "init": {
            "mode": saved.mode if saved else "on-demand",
            "hardware": saved.hardware if saved else None,
            "saved": saved is not None,
            "config": str(path),
        },
        "model_configs": {
            "path": str(models.root),
            "models": len(models.repos()),
            "size_bytes": _dir_size(models.root) if models.root.is_dir() else 0,
        },
        "revisions": revisions,
    }
    _emit(payload)
    return 0


def run_clear(args: argparse.Namespace) -> int:
    cache = Cache(Path(args.cache_dir) if args.cache_dir else None)
    path = config_path(cache_dir=args.cache_dir)
    targets = [p for _, _, p in _sha_dirs(cache)]
    extras = [p for p in (cache.root / "materialized", _model_config_cache(args).root) if p.exists()]
    sizes = sum(_dir_size(p) for p in (*targets, *extras))
    if not args.yes:
        if not sys.stdin.isatty():
            raise UsageError("clear asks for confirmation; pass --yes to remove without a terminal")
        sys.stderr.write(
            f"Remove {len(targets)} cached revision(s) ({_mib(sizes)}) under {cache.root} and the init config "
            f"{path}? [y/N] "
        )
        sys.stderr.flush()
        if sys.stdin.readline().strip().lower() not in ("y", "yes"):
            sys.stderr.write("aborted\n")
            return EXIT_FAILURE
    failures: list[str] = []
    removed = freed = 0
    for target in (*targets, *extras):
        size = _dir_size(target)
        try:
            shutil.rmtree(target)
        except OSError as exc:
            failures.append(f"{target}: {exc}")
            continue
        if target in targets:
            removed += 1
        freed += size
    for repo_dir in sorted(p for p in cache.root.glob("*__*") if p.is_dir()) if cache.root.is_dir() else []:
        try:
            repo_dir.rmdir()  # only when empty
        except OSError:
            pass
    if failures:
        for failure in failures:
            sys.stderr.write(f"{_NAME}: could not remove {failure}\n")
        _emit({"removed_revisions": removed, "freed_bytes": freed, "config_removed": False, "failed": failures})
        return EXIT_FAILURE  # keep the init config: the cache is only partly cleared
    config_removed = path.exists()
    path.unlink(missing_ok=True)
    _emit({"removed_revisions": removed, "freed_bytes": freed, "config_removed": config_removed})
    return 0


def run_validate(args: argparse.Namespace) -> int:
    from .data_validate import dumps, validate_local, validate_repo

    if bool(args.path) == bool(args.repo):
        raise UsageError("validate needs exactly one of PATH or --repo")
    if args.path:
        report = validate_local(args.path)
    else:
        saved = load_config(config_path(cache_dir=args.cache_dir))
        repo, revision = resolve_source(args.repo, args.revision, config=saved)  # the same precedence as fetch
        cache = Cache(Path(args.cache_dir) if args.cache_dir else None)
        report = validate_repo(repo, revision, transport=make_transport(), cache=cache)
    sys.stdout.write(dumps(report))
    return 0 if report["ok"] else EXIT_FAILURE


class UsageError(SelectionError):
    code = "usage"


def run_data_command(args: argparse.Namespace) -> int:
    handlers = {
        "fetch": run_fetch,
        "init": run_init,
        "status": run_status,
        "clear": run_clear,
        "validate": run_validate,
    }
    try:
        return handlers[args.data_command](args)
    except FileNotFoundError as exc:
        return _fail(UsageError(str(exc)), EXIT_USAGE, args)
    except SelectionError as exc:  # includes UnsupportedSelectionError and UsageError
        return _fail(exc, EXIT_USAGE, args)
    except (DataFetchError, ConfigError) as exc:
        return _fail(exc if isinstance(exc, DataFetchError) else DataFetchError(str(exc)), EXIT_FAILURE, args)
    except ValueError as exc:  # malformed environment or flag value
        return _fail(UsageError(str(exc)), EXIT_USAGE, args)
    except Exception as exc:  # transport errors (network, auth, unknown repo) are not library errors
        return _fail(RevisionError(f"{type(exc).__name__}: {exc}"), EXIT_FAILURE, args)


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _fail(exc: DataFetchError, status: int, args: argparse.Namespace) -> int:
    label = f"aisim data {args.data_command}" if args.command == "data" else f"aisim {args.command}"
    sys.stderr.write(f"{label}: {exc}\n")
    sys.stderr.write(json.dumps(exc.to_dict(), allow_nan=False) + "\n")
    return status
