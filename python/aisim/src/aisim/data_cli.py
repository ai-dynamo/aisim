# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``aisim data`` (fetch, status, clear, validate), ``aisim fetch`` and ``aisim init``: a thin command line
over ``aisim_core.data``.

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

from .data_parser import _PREDICTION_PURPOSE, PURPOSES, SUPPORTED_CATEGORIES, add_data_parser  # noqa: F401

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
    planned = plan(selection, revision=revision, **common)
    if getattr(args, "dry_run", False):
        return {"dry_run": True, **planned.to_dict()}
    if not args.offline and planned.uncached_bytes >= LARGE_DOWNLOAD_BYTES:
        sys.stderr.write(
            f"{_NAME}: downloading {_mib(planned.uncached_bytes)} in {len(planned.uncached)} files "
            f"from {repo}@{planned.revision[:12]}\n"
        )
    # The plan resolved the revision; reuse the SHA so the download needs no second lookup.
    result = fetch(selection, revision=planned.revision, max_bytes=_max_bytes(args), **common)
    result.plan.requested_revision = revision
    return {"dry_run": False, **result.to_dict()}


def run_fetch(args: argparse.Namespace) -> int:
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
    save_config(config, path)  # saved only after the download succeeded
    _emit(payload)
    return 0


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
        "revisions": revisions,
    }
    _emit(payload)
    return 0


def run_clear(args: argparse.Namespace) -> int:
    cache = Cache(Path(args.cache_dir) if args.cache_dir else None)
    path = config_path(cache_dir=args.cache_dir)
    targets = [p for _, _, p in _sha_dirs(cache)]
    extras = [p for p in (cache.root / "materialized",) if p.exists()]
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
        repo, revision = resolve_source(args.repo, args.revision, env={})
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
