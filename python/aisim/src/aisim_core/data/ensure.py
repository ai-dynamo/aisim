# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make the dataset a resolved configuration needs available before a predictor is built.

:func:`ensure_data` is the hidden loading step of DEP #435 ("Loading is hidden
behind predict/estimate"). For a resolved ``best_available`` configuration it
derives the catalog selection (``system``, ``backend``, ``backend_version``,
``model``), then resolves, plans, fetches, verifies and loads the data through
the category loader, and returns the configuration with the loaded inputs wired
in plus the dataset revision that was used.

Source order, per category named by the :class:`DataPin`:

1. local override (``DataPin.local_root``, a dataset-layout directory; no network),
2. cached pinned Hugging Face revision (a warm cache makes zero network calls;
   a cold cache downloads when the uncached size is within ``max_bytes``),
3. packaged data: only for categories the pin does not name. A category the pin
   names is never served from packaged data; absent or invalid data is an error.

``op-latency`` is composed, not selected: every operation family published for the configuration's
system/backend/version is verified and written, with the packaged system specs, into one ``systems/`` root
(:func:`~aisim_core.data.op_tree.load_op_latency_tree`) that replaces ``systems_paths``. Two entries for the same family
are a conflict resolved with ``AISIM_DATA_CONFIGURATION``; a family the dataset does not publish is absent, never
filled from packaged tables by the pin. The versions a selected entry's ``reuse.yaml`` declares as donors are part of
the root too (the native database reads them beside the requested version and skips a donor that is absent), fetched
by the same pin; a donor the dataset does not publish is skipped, as it is there.

With no pin the whole step is a no-op and behavior is unchanged. Credentials are
the standard Hugging Face ones (``HF_TOKEN`` / ``huggingface-cli login``); this
module accepts, stores and prints no token.

Environment (read by :func:`pin_from_environment`): ``AISIM_DATASET=<repo>@<revision>``
(use a full commit SHA for offline reuse), ``AISIM_DATA_LOCAL``, ``AISIM_DATA_CATEGORIES``
(default ``op-latency``), ``AISIM_DATA_CONFIGURATION``, ``AISIM_DATA_MAX_BYTES``,
``AISIM_DATA_OFFLINE`` and ``AISIM_DATA_CACHE``.
"""

from __future__ import annotations

import contextvars
import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import tempfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from aisim_core.data import CategoryIndex, DatasetIndex, Entry, _file, _verify
from aisim_core.data.config import config_path, load_config, resolve_source
from aisim_core.data.fetch import (
    DATASET_INDEX_PATH,
    BudgetExceededError,
    Cache,
    DataFetchError,
    FetchPlan,
    FetchResult,
    IntegrityError,
    OfflineMissingError,
    RevisionError,
    Selection,
    SelectionError,
    Transport,
    _resolve_source,
    fetch,
    fetch_support,
    resolve_revision,
)
from aisim_core.data.fpm_library import PREDICTION_ROLES, STAGED_MARKER, _stageable_roots, staged_inputs_digest
from aisim_core.data.loaders import LOADERS, FpmData, LoaderError
from aisim_core.data.op_schema import DATA_PREFIX, QUERY_VERSIONS_NAME
from aisim_core.data.op_tree import (
    OpTreeLoad,
    _op_latency_index,
    composed_root_matches,
    load_op_latency_tree,
    lookup_yaml,
    op_latency_inputs_digest,
    read_entry_manifest,
    verify_op_latency_support,
)
from aisim_core.data.systems import PACKAGED_SYSTEMS_DIR, packaged_system_specs

DEFAULT_MAX_BYTES = 256 * 1024 * 1024
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_ensuring: contextvars.ContextVar[bool] = contextvars.ContextVar("aisim_ensuring_data", default=False)


# --------------------------------------------------------------------------- errors


class MissingDataError(DataFetchError):
    """The pinned source has no usable data for this configuration."""

    code = "missing-data"

    def __init__(self, message: str, *, selection: Selection | None = None, source: str | None = None):
        super().__init__(message)
        self.selection = selection
        self.source = source

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "source": self.source,
            "selection": None if self.selection is None else {k: v for k, v in vars(self.selection).items() if v},
        }


class DataTooLargeError(DataFetchError):
    """Auto-download is refused above the size threshold."""

    code = "data-too-large"

    def __init__(self, uncached_bytes: int, max_bytes: int, command: str, already_downloaded: int = 0):
        done = (
            f"{already_downloaded} bytes of this pin were already downloaded and cached; this selection is the rest. "
            if already_downloaded
            else "Nothing was downloaded. "
        )
        super().__init__(
            f"dataset needs {uncached_bytes} uncached bytes, above the auto-download threshold of {max_bytes} "
            f"bytes (AISIM_DATA_MAX_BYTES). {done}To proceed, run: {command}"
        )
        self.uncached_bytes, self.max_bytes, self.command = uncached_bytes, max_bytes, command

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "uncached_bytes": self.uncached_bytes,
            "max_bytes": self.max_bytes,
            "command": self.command,
        }


class DataConflictError(DataFetchError):
    """Dataset content contradicts an explicit user setting or is ambiguous."""

    code = "data-conflict"


# --------------------------------------------------------------------------- pin and result


@dataclass(frozen=True)
class DataPin:
    """Where the data comes from. No token field: auth is the standard HF environment."""

    repo: str | None = None
    revision: str | None = None  # full commit SHA recommended: it resolves with no network
    local_root: Path | None = None
    categories: tuple[str, ...] = ("op-latency",)
    configuration_id: str | None = None  # disambiguates several catalog matches
    max_bytes: int = DEFAULT_MAX_BYTES
    offline: bool = False
    cache_dir: Path | None = None
    # Built from the source ``aisim init`` saved, not named by the caller or the environment. Such an ambient pin
    # selects op-latency data only for runs that read operation tables (see ``rust_engine_step._ensure_dataset``).
    saved_source: bool = False

    def __post_init__(self) -> None:
        if self.local_root is None and not (self.repo and self.revision):
            raise ValueError("a data pin needs local_root or both repo and revision")
        if not self.categories:
            raise ValueError("a data pin needs at least one category; AISIM_DATA_CATEGORIES must name one or more")
        unknown = [c for c in self.categories if c not in LOADERS or c == "e2e"]
        if unknown:
            raise ValueError(f"categories not loadable into a predictor: {unknown}")


@dataclass
class EnsuredData:
    """The configuration with dataset inputs wired in, and where they came from."""

    config: dict[str, Any]
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)  # category -> provenance
    roots: dict[str, list[str]] = field(default_factory=dict)  # category -> systems roots it contributed
    downloaded_bytes: int = 0  # everything this call fetched, charged against the pin's one size allowance

    def to_dict(self) -> dict[str, Any]:
        return {"datasets": self.datasets}

    @property
    def revisions(self) -> dict[str, str]:
        return {c: d["revision"] for c, d in self.datasets.items()}


def pin_from_environment(env: Mapping[str, str] | None = None) -> DataPin | None:
    """Build a pin from ``AISIM_DATA*`` variables, else the source saved by ``aisim init``.

    ``None`` when neither names a source (the shipped default applies).
    """
    env = os.environ if env is None else env
    spec, local = env.get("AISIM_DATASET"), env.get("AISIM_DATA_LOCAL")
    from_saved = False
    if not spec and not local:
        saved = load_config(config_path(env))  # a corrupt saved config raises ConfigError
        if not saved or not (saved.repo or saved.revision):
            return None
        spec = "{}@{}".format(*resolve_source(env=env, config=saved))  # same rules as ``data fetch``
        from_saved = True
    repo = revision = None
    if spec:
        repo, sep, revision = spec.partition("@")
        if not sep or not repo or not revision:
            raise ValueError("AISIM_DATASET must be <owner>/<name>@<revision>")
        # Keep the requested name until transport setup: the successor's path mapping must survive resolution.
    kwargs: dict[str, Any] = {}
    if env.get("AISIM_DATA_CATEGORIES"):
        kwargs["categories"] = tuple(c.strip() for c in env["AISIM_DATA_CATEGORIES"].split(",") if c.strip())
    if env.get("AISIM_DATA_MAX_BYTES"):
        kwargs["max_bytes"] = int(env["AISIM_DATA_MAX_BYTES"])
    if env.get("AISIM_DATA_CACHE"):
        kwargs["cache_dir"] = Path(env["AISIM_DATA_CACHE"])
    return DataPin(
        repo=repo,
        revision=revision,
        local_root=Path(local) if local else None,
        configuration_id=env.get("AISIM_DATA_CONFIGURATION") or None,
        offline=env.get("AISIM_DATA_OFFLINE", "").lower() in {"1", "true", "yes"},
        saved_source=from_saved,
        **kwargs,
    )


@contextmanager
def _guard():
    token = _ensuring.set(True)
    try:
        yield
    finally:
        _ensuring.reset(token)


def ensuring() -> bool:
    """True while :func:`ensure_data` is running (its own predictor checks must not recurse)."""
    return _ensuring.get()


# --------------------------------------------------------------------------- catalog lookup


def _matching_entries(root: Path, selection: Selection) -> list[Entry]:
    """Entries of the category index under ``root`` that match ``selection`` (files already verified)."""
    try:
        dataset = DatasetIndex.model_validate_json(_file(root, DATASET_INDEX_PATH).read_bytes())
        reference = dataset.categories.get(selection.category)
        if reference is None:
            return []
        index = CategoryIndex.model_validate_json(_verify(root, reference.index).read_bytes())
    except (ValueError, ValidationError) as exc:
        raise MissingDataError(f"dataset catalog under {root} is unusable: {exc}", selection=selection) from exc
    return [e for e in index.entries if selection.matches(e)]


def _describe(selection: Selection) -> str:
    parts = {
        "system": selection.hardware,
        "backend": selection.framework,
        "backend_version": selection.framework_version,
        "model": selection.model,
    }
    return ", ".join(f"{k}={v}" for k, v in parts.items() if v is not None)


def _command(pin: DataPin, selection: Selection) -> str:
    """A ``aisim data fetch`` command that is valid for ``selection`` (it needs a selector)."""
    flags = ["--category", selection.category]
    for flag, value in (
        ("--model", selection.model),
        ("--hardware", selection.hardware),
        ("--framework", selection.framework),
        ("--framework-version", selection.framework_version),
        ("--operation-family", selection.operation_family),
        ("--repo", pin.repo),
        ("--revision", pin.revision),
        ("--cache-dir", pin.cache_dir),
    ):
        if value:
            flags += [flag, str(value)]
    # One flag per ID: without them the command would download every matching entry, not the one the pin names.
    for configuration_id in selection.configuration_ids:
        flags += ["--configuration", configuration_id]
    has_selector = selection.configuration_ids or selection.hardware
    if selection.select_all or (selection.category == "op-latency" and not has_selector):
        flags.append("--all")
    # The command line does not apply the auto-download threshold to an explicit fetch unless
    # AISIM_DATA_MAX_BYTES is set; --no-limit makes the command valid in both cases.
    flags.append("--no-limit")
    return shlex.join(["aisim", "data", "fetch", *flags])


# --------------------------------------------------------------------------- ensure


def ensure_data(
    config: Mapping[str, Any],
    pin: DataPin | None,
    *,
    transport: Transport | None = None,
) -> EnsuredData:
    """Return ``config`` with verified dataset inputs wired in; no-op without a pin.

    Raises :class:`MissingDataError`, :class:`DataTooLargeError`,
    :class:`DataConflictError` or another :class:`DataFetchError`; never falls
    back to other data for a category the pin names.
    """
    out = EnsuredData(config=copy.deepcopy(dict(config)))
    if pin is None:
        return out
    if transport is None and pin.local_root is None and not pin.offline:
        from aisim_core.data.fetch import HfTransport

        transport = HfTransport()
    if pin.local_root is None:
        repo, revision, transport = _resolve_source(pin.repo, pin.revision, transport)
        pin = replace(pin, repo=repo, revision=revision)
        # One commit for the whole call: policy, primary and dependency fetches must read the same revision even if
        # a branch moves meanwhile. A full SHA needs no network.
        try:
            pin = replace(pin, revision=resolve_revision(transport, pin.repo, pin.revision, offline=pin.offline))
        except RevisionError as exc:
            raise MissingDataError(f"cannot resolve {pin.repo}@{pin.revision} to a commit: {exc}") from exc
    with _guard():
        if "op-latency" in pin.categories:
            _resolve_backend_version(out, pin, transport)
        owner = _configuration_owner(pin, transport, out)
        # Operation data first: an FPM library staged after it must take operation tables from the pinned root alone.
        for category in sorted(pin.categories, key=lambda name: name != "op-latency"):
            # A configuration ID names one entry of one category: it constrains only that category.
            category_pin = pin if owner in (None, category) else replace(pin, configuration_id=None)
            _ensure_category(out, category, category_pin, transport)
        if "op-latency" in out.roots and "fpm" in out.roots:
            _compose_pinned_roots(out)
    return out


def _table_digests(root: Path) -> dict[str, str]:
    """``{relative path: sha256}`` of the operation tables under ``root/data``."""
    base = Path(root) / "data"
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(base.rglob("*.parquet"))
        if p.is_file()
    }


def _compose_pinned_roots(out: EnsuredData) -> None:
    """Order the roots of an FPM and an op-latency pin the same way whatever order the pin names them.

    The roots are the FPM bundle's retained inputs, then the op-latency root. A table both provide with different
    content has no category-order-free answer, so it is a conflict naming both entries.
    """
    fpm_roots, op_root = [Path(p) for p in out.roots["fpm"]], Path(out.roots["op-latency"][0])
    op_tables = _table_digests(op_root)
    for root in fpm_roots:
        clashes = sorted(path for path, digest in _table_digests(root).items() if op_tables.get(path, digest) != digest)
        if clashes:
            fpm = out.datasets["fpm"].get("configuration_id")
            op = ", ".join(out.datasets["op-latency"].get("configuration_ids", ()))
            raise DataConflictError(
                f"fpm entry {fpm} and op-latency entries {op} provide different content for the same operation "
                f"table(s) {clashes[:3]}; pin one of them (AISIM_DATA_CATEGORIES) or republish the bundle"
            )
    out.config["systems_paths"] = list(dict.fromkeys([*map(str, fpm_roots), str(op_root)]))


def _configuration_owner(pin: DataPin, transport: Transport | None, out: EnsuredData) -> str | None:
    """The category whose index lists ``pin.configuration_id``; ``None`` when no choice has to be made.

    A single-category pin keeps its plain behavior (the category reports a miss itself). With several categories
    the ID must belong to exactly one of them, or the pin cannot complete and says which categories it searched.
    """
    if pin.configuration_id is None or len(pin.categories) < 2:
        return None
    owners = [c for c in pin.categories if pin.configuration_id in _category_configuration_ids(pin, transport, c, out)]
    if not owners:
        raise MissingDataError(
            f"configuration {pin.configuration_id} is not in any category of the pin {list(pin.categories)}"
        )
    return owners[0]


def _category_configuration_ids(
    pin: DataPin, transport: Transport | None, category: str, out: EnsuredData | None = None
) -> set[str]:
    """Configuration IDs listed by one category index of the pinned dataset (empty when it has no such category)."""
    try:
        if pin.local_root is not None:
            root = Path(pin.local_root).absolute()
            dataset = DatasetIndex.model_validate_json(_file(root, DATASET_INDEX_PATH).read_bytes())
            reference = dataset.categories.get(category)
            if reference is None:
                return set()
            index = CategoryIndex.model_validate_json(_verify(root, reference.index).read_bytes())
        else:
            index = _remote_category_index(pin, transport, category, out)
    except SelectionError:
        return set()
    except (ValueError, ValidationError) as exc:
        raise MissingDataError(f"the {category} catalog is unusable: {exc}") from exc
    return {e.configuration_id for e in index.entries}


def _remote_category_index(
    pin: DataPin, transport: Transport | None, category: str, out: EnsuredData | None
) -> CategoryIndex:
    """The pinned remote dataset's index of ``category``: the cache first, then a read charged to the pin."""
    from aisim_core.data.fetch import load_category_index

    common = {"repo": pin.repo, "revision": pin.revision, "cache": Cache(pin.cache_dir)}
    try:
        return load_category_index(category, transport=None, offline=True, **common)[1]
    except (OfflineMissingError, RevisionError, IntegrityError) as exc:
        if pin.offline:
            raise MissingDataError(f"offline: the {category} catalog is not usable: {exc}") from exc
    return _fetch_preselection(load_category_index, category, pin, transport, out)[1]


_SLOT_ALIASES = ("current", "previous", "next")


def _resolve_backend_version(out: EnsuredData, pin: DataPin, transport: Transport | None) -> None:
    """Turn an omitted backend version or a slot alias into the literal version of the pinned dataset's policy.

    Selection and identity checks compare literal versions, but a configuration may omit the version (its public
    default) or name ``current``/``previous``/``next``. The pinned dataset's own ``query_versions`` lookup
    decides them, as the packaged one does for unpinned runs; an explicit version is left alone.
    """
    requested = out.config.get("backend_version")
    if requested is not None and requested not in _SLOT_ALIASES:
        return
    system, backend = out.config.get("system"), out.config.get("backend")
    if not system or not backend:
        return  # nothing to select; the category selection reports the missing field
    index, query_versions = _version_policy(pin, transport, out)
    versions = {(e.identity.hardware, e.identity.framework, e.identity.framework_version) for e in index.entries}
    out.config["backend_version"] = _slot_version(system, backend, requested, versions, query_versions)


def _fetch_preselection(fetcher, category, pin, transport, out):
    """Charge policy and ownership reads to the same allowance as primary and dependency fetches."""
    spent = out.downloaded_bytes if out is not None else 0
    accounting = FetchPlan(repo=pin.repo, revision=pin.revision, requested_revision=pin.revision, offline=False)
    try:
        return fetcher(
            category,
            repo=pin.repo,
            revision=pin.revision,
            transport=transport,
            cache=Cache(pin.cache_dir),
            max_bytes=max(pin.max_bytes - spent, 0),
            _accounting=accounting,
        )
    except BudgetExceededError as exc:
        selection = Selection(
            category=category,
            configuration_ids=(pin.configuration_id,) if pin.configuration_id else (),
            hardware=(out.config.get("system") if out is not None else None),
            model=(out.config.get("model") if out is not None else None),
        )
        raise DataTooLargeError(
            spent + exc.plan.uncached_bytes + exc.plan.metadata_bytes,
            pin.max_bytes,
            _command(pin, selection),
            already_downloaded=spent + accounting.metadata_bytes,
        ) from exc
    finally:
        if out is not None:
            out.downloaded_bytes += accounting.metadata_bytes


def _version_policy(pin: DataPin, transport: Transport | None, out: EnsuredData):
    """The op-latency category index of the pinned dataset and its parsed ``query_versions`` document."""
    name = f"{DATA_PREFIX}/{QUERY_VERSIONS_NAME}.parquet"
    where = str(pin.local_root) if pin.local_root is not None else f"{pin.repo}@{pin.revision}"
    try:
        if pin.local_root is not None:
            root = Path(pin.local_root).absolute()
            index = _op_latency_index(root)
            support = {a.path: a for a in index.support}
            if name not in support:
                # An unindexed file has no pinned digest; remote pins resolve the policy only through the index, so
                # a local root must too, or a stray file could pick the backend version.
                raise MissingDataError(
                    f"{where} does not index {name}; the backend version cannot be resolved (pass an explicit "
                    "backend_version, or republish the dataset with the current publisher)"
                )
            path = _verify(root, support[name])
        else:
            cache = Cache(pin.cache_dir)
            common = {"repo": pin.repo, "revision": pin.revision, "cache": cache}
            try:
                _, index, paths = fetch_support("op-latency", transport=None, offline=True, **common)
            except (OfflineMissingError, RevisionError, IntegrityError) as exc:
                if pin.offline:
                    raise MissingDataError(f"offline: the version policy of {where} is not usable: {exc}") from exc
                _, index, paths = _fetch_preselection(fetch_support, "op-latency", pin, transport, out)
            path = paths.get(name)
        if path is None or not Path(path).is_file():
            raise MissingDataError(f"{where} publishes no {name}; the backend version cannot be resolved")
        document = yaml.safe_load(lookup_yaml(Path(path)))
        if not isinstance(document, dict):
            raise ValueError("version policy must be a mapping")
    except (LoaderError, ValueError, SelectionError) as exc:
        raise MissingDataError(f"cannot read the version policy of {where}: {exc}") from exc
    try:
        _validate_version_policy(document)
    except ValueError as exc:
        raise MissingDataError(f"cannot read the version policy of {where}: {exc}") from exc
    return index, document


def _validate_version_policy(document: Any) -> None:
    """Reject malformed policies while permitting a valid mapping with no applicable entry."""
    if not isinstance(document, dict):
        raise ValueError("version policy must be a mapping")
    if document.keys() - {"schema_version", "defaults", "overrides"}:
        raise ValueError("unknown version policy fields")
    version = document.get("schema_version", 1)
    if type(version) is not int or version != 1:
        raise ValueError("unsupported version policy schema_version")

    def entries(mapping):
        if not isinstance(mapping, dict):
            raise ValueError("version policy entries must be mappings")
        for backend, slots in mapping.items():
            if not isinstance(backend, str) or not isinstance(slots, dict):
                raise ValueError("version policy backend slots must be mappings")
            if any(
                name not in ("current", "previous") or (v is not None and (not isinstance(v, str) or not v))
                for name, v in slots.items()
            ):
                raise ValueError("version policy slots must be version strings or null")

    entries(document.get("defaults", {}))
    overrides = document.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("version policy overrides must be a mapping")
    for system, backends in overrides.items():
        if not isinstance(system, str):
            raise ValueError("version policy system names must be strings")
        entries(backends)


def _slot_version(system: str, backend: str, requested: str | None, versions, document: Mapping[str, Any]) -> str:
    from aisim_core.sdk.perf_database import _version_sort_tuple

    held = {v for hw, fw, v in versions if hw == system and fw == backend}
    if not held:
        raise MissingDataError(f"the pinned dataset holds no {backend} operation data for {system}")
    overrides = (document.get("overrides") or {}).get(system, {}).get(backend)
    entry = overrides if overrides is not None else (document.get("defaults") or {}).get(backend)
    if not entry or not entry.get("current"):
        if requested is None:  # no authored policy: the latest version the dataset holds
            return max(held, key=_version_sort_tuple)
        raise MissingDataError(
            f"no queryable versions are defined for {backend} on {system}; cannot resolve {requested!r}"
        )
    slots = {"current": str(entry["current"])}
    if entry.get("previous"):
        slots["previous"] = str(entry["previous"])
    if overrides is None:
        # next is the highest version newer than current that any defaults-governed system holds, else current.
        frozen = {
            hw for hw, per_backend in (document.get("overrides") or {}).items() if (per_backend or {}).get(backend)
        }
        current_key = _version_sort_tuple(slots["current"])
        newer = {
            v for hw, fw, v in versions if fw == backend and hw not in frozen and _version_sort_tuple(v) > current_key
        }
        slots["next"] = max(newer, key=_version_sort_tuple) if newer else slots["current"]
    resolved = slots.get(requested or "current")
    if resolved is None:
        raise MissingDataError(f"{backend} on {system} has no {requested!r} version; available slots: {slots}")
    return resolved


def _ensure_category(out: EnsuredData, category: str, pin: DataPin, transport: Transport | None) -> None:
    loader = LOADERS[category]()
    try:
        selection = loader.select(out.config)
    except SelectionError as exc:
        raise MissingDataError(f"cannot derive a {category} selection from the configuration: {exc}") from exc
    if category == "host-cost":
        _ensure_host_cost(out, loader, selection, pin, transport)
        return
    # A configuration ID names one entry of a category. Operation families are complementary inputs of one root,
    # so there it chooses only among the entries of its own family and every other family still arrives.
    if pin.configuration_id and category != "op-latency":
        selection = replace(selection, configuration_ids=(pin.configuration_id,))

    if pin.local_root is not None:
        root = Path(pin.local_root).absolute()
        source, revision, downloaded = "local", f"local:{root}", 0
        source_key = "local-" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    else:
        if category == "op-latency" and pin.configuration_id:
            # Choose from the catalog before any manifest is planned, so a rejected alternative is never downloaded.
            where = f"{pin.repo}@{pin.revision}"
            try:
                index = _remote_category_index(pin, transport, category, out)
                chosen = _one_entry_per_family(
                    [e for e in index.entries if selection.matches(e)], selection, where, pin.configuration_id
                )
            except SelectionError as exc:
                raise MissingDataError(f"{exc} ({_describe(selection)} in {where})", selection=selection) from exc
            except (ValueError, ValidationError) as exc:
                raise MissingDataError(f"the {category} catalog is unusable: {exc}", selection=selection) from exc
            selection = replace(selection, configuration_ids=tuple(e.configuration_id for e in chosen))
        result = _fetch_pinned(selection, pin, transport, out)
        root = result.root
        source = "hf-download" if result.downloaded_bytes else "hf-cache"
        revision, downloaded = result.plan.revision, result.downloaded_bytes
        source_key = f"{pin.repo.replace('/', '__')}-{revision}"

    entries = _matching_entries(root, selection)
    where = f"{pin.repo}@{revision}" if pin.local_root is None else str(root)
    if not entries:
        raise MissingDataError(
            f"no {category} data for {_describe(selection)} in {where}; there is no fallback to other data"
            + _categories_hint(pin),
            selection=selection,
            source=where,
        )
    if category == "op-latency":
        entries = _one_entry_per_family(entries, selection, where, pin.configuration_id)
        loaded, entry_ids = _load_op_latency(
            root,
            entries,
            out.config,
            Cache(pin.cache_dir),
            source_key,
            selection,
            where,
            fetch_donors=None
            if pin.local_root is not None
            else lambda wanted: _fetch_pinned(wanted, pin, transport, out),
        )
        _wire(out.config, category, loaded, None)
        out.roots[category] = [str(loaded.root)]
        out.datasets[category] = {
            "repo": pin.repo,
            "revision": revision,
            "source": source,
            "configuration_ids": entry_ids,
            "downloaded_bytes": downloaded,
        }
        return
    if len(entries) > 1:
        ids = ", ".join(sorted(e.configuration_id for e in entries))
        raise DataConflictError(
            f"{len(entries)} {category} entries match {_describe(selection)} in {where} ({ids}); "
            "set AISIM_DATA_CONFIGURATION (DataPin.configuration_id) to choose one"
        )
    entry = entries[0]
    if out.config.get("backend_version") is None and entry.identity.framework_version is not None:
        # The one match fixes an omitted version, so the identity check and the returned configuration use it. An
        # explicit version never reaches here with another one (the selection filters it) and several matches
        # were refused above.
        out.config["backend_version"] = entry.identity.framework_version
    cache = Cache(pin.cache_dir)
    output_dir = cache.root / "materialized" / source_key / entry.configuration_id
    try:
        loaded = _load(
            loader, category, root, entry, out.config, output_dir, isolate_roots="op-latency" in out.datasets
        )
    except (LoaderError, ValueError) as exc:
        raise MissingDataError(
            f"{category} data {entry.configuration_id} in {where} failed verification: {exc}",
            selection=selection,
            source=where,
        ) from exc
    _wire(out.config, category, loaded, entry)
    out.roots[category] = [str(p) for p in loaded.config.get("systems_paths") or ()]
    out.datasets[category] = {
        "repo": pin.repo,
        "revision": revision,
        "source": source,
        "configuration_id": entry.configuration_id,
        "downloaded_bytes": downloaded,
    }


def _ensure_host_cost(
    out: EnsuredData, loader, selection: Selection, pin: DataPin, transport: Transport | None
) -> None:
    """Wire ``config["host_cost"]``: the measured tokenizer and unqueued overhead, or ``None`` when unmeasured.

    The configuration names ``system``, ``backend``, ``backend_version``, ``model`` and ``endpoint``. Entries are
    chosen from the catalog first (closest collected version, tokenizer of the model, the system's host CPU), so only
    the chosen ones are fetched. An unmeasured deployment is not an error: it has no default, so it wires ``None``.
    """
    from aisim_core.host_cost.lookup import choose, overhead, tokenizer_key

    config = out.config
    missing = [name for name in ("backend_version", "model", "endpoint") if not config.get(name)]
    if missing:
        raise MissingDataError(f"a host-cost selection needs {', '.join(missing)} in the configuration")
    if pin.local_root is not None:
        root = Path(pin.local_root).absolute()
        where, revision, source = str(root), f"local:{root}", "local"
        catalog = _matching_entries(root, selection)
    else:
        where, revision = f"{pin.repo}@{pin.revision}", pin.revision
        try:
            index = _remote_category_index(pin, transport, "host-cost", out)
        except SelectionError:
            index = None  # this revision publishes no host-cost category
        catalog = [e for e in index.entries if selection.matches(e)] if index is not None else []
        source = "hf-cache"
    try:
        key = tokenizer_key(config["model"], catalog)
        tokenizer, unqueued = (
            choose(
                catalog,
                framework=config["backend"],
                version=config["backend_version"],
                endpoint=config["endpoint"],
                system=config["system"],
                tokenizer=key,
            )
            if key is not None
            else (None, None)
        )
    except ValueError as exc:
        raise DataConflictError(f"host-cost data in {where}: {exc}") from exc
    chosen = [c for c in (tokenizer, unqueued) if c is not None]
    downloaded = 0
    if chosen and pin.local_root is None:
        ids = tuple(c.entry.configuration_id for c in chosen)
        result = _fetch_pinned(replace(selection, configuration_ids=ids), pin, transport, out)
        root, downloaded = result.root, result.downloaded_bytes
        source = "hf-download" if downloaded else "hf-cache"
    manifests = []
    for choice in chosen:
        try:
            manifest = loader.load(root, choice.entry.manifest.path)
        except (LoaderError, ValueError) as exc:
            raise MissingDataError(
                f"host-cost data {choice.entry.configuration_id} in {where} failed verification: {exc}",
                selection=selection,
                source=where,
            ) from exc
        if (manifest.configuration_id, manifest.purpose, manifest.identity) != (
            choice.entry.configuration_id,
            choice.entry.purpose,
            choice.entry.identity,
        ):
            raise MissingDataError(f"host-cost catalog and manifest identity differ in {where}", source=where)
        manifests.append(manifest)
    config["host_cost"] = (
        overhead(
            config["backend_version"],
            (tokenizer, manifests[0]),
            (unqueued, manifests[1]) if unqueued is not None else None,
        ).to_dict()
        if tokenizer is not None
        else None
    )
    out.datasets["host-cost"] = {
        "repo": pin.repo,
        "revision": revision,
        "source": source,
        "configuration_ids": [c.entry.configuration_id for c in chosen],
        "downloaded_bytes": downloaded,
    }


def _categories_hint(pin: DataPin) -> str:
    return (
        f"; this pin loads {', '.join(pin.categories)} (set AISIM_DATA_CATEGORIES to load other categories, e.g. fpm)"
    )


def _fetch_pinned(
    selection: Selection, pin: DataPin, transport: Transport | None, out: EnsuredData | None = None
) -> FetchResult:
    """Fetch one selection of the pin; ``out`` carries what earlier fetches of this call already downloaded."""
    spent = 0 if out is None else out.downloaded_bytes
    cache = Cache(pin.cache_dir)
    common = {"repo": pin.repo, "revision": pin.revision, "cache": cache}
    where = f"{pin.repo}@{pin.revision}"
    # Warm cache first: a full-SHA pin needs no network at all.
    try:
        return fetch(selection, transport=None, offline=True, **common)
    except OfflineMissingError as exc:
        if pin.offline:
            raise MissingDataError(
                f"offline: {where} is not fully cached for {_describe(selection)}; missing: " + ", ".join(exc.missing),
                selection=selection,
                source=where,
            ) from exc
    except IntegrityError as exc:
        # A corrupt cached catalog: an online pin repairs it by fetching again, an offline one cannot.
        if pin.offline:
            raise MissingDataError(
                f"offline: the cached catalog of {where} is corrupt: {exc}", selection=selection, source=where
            ) from exc
    except RevisionError as exc:
        if pin.offline:
            raise MissingDataError(
                f"offline: {where} is not a full commit SHA, so it cannot be resolved without network: {exc}",
                selection=selection,
                source=where,
            ) from exc
    except SelectionError as exc:
        # Cached catalog exists but matches nothing: the answer is final for this revision.
        raise MissingDataError(
            f"{exc} ({_describe(selection)} in {where}){_categories_hint(pin)}", selection=selection, source=where
        ) from exc
    try:
        # The auto-download allowance belongs to the whole pin: a donor fetched after the primary selection gets only
        # what the earlier fetches left.
        result = fetch(selection, transport=transport, max_bytes=max(pin.max_bytes - spent, 0), **common)
        if out is not None:
            out.downloaded_bytes += result.downloaded_bytes + result.plan.metadata_bytes
        return result
    except BudgetExceededError as exc:
        raise DataTooLargeError(
            spent + exc.plan.uncached_bytes + exc.plan.metadata_bytes,
            pin.max_bytes,
            _command(pin, selection),
            already_downloaded=spent,
        ) from exc
    except SelectionError as exc:
        raise MissingDataError(
            f"{exc} ({_describe(selection)} in {where}){_categories_hint(pin)}", selection=selection, source=where
        ) from exc


def _materialization_is_intact(systems: Path, manifest) -> bool:
    """True when every runtime artifact of ``manifest`` is present under ``systems`` with its declared bytes."""
    try:
        for artifact in manifest.artifacts:
            if artifact.runtime_path is not None:
                _verify(systems, artifact.model_copy(update={"path": artifact.runtime_path}))
    except (ValueError, OSError):
        return False
    return True


def _cached_pin_is_valid(output_dir: Path, manifest, cached: Mapping[str, Any]) -> bool:
    """True when the cached config pins the manifest's profile at the path this materialization wrote."""
    try:
        profile = next(a for a in manifest.artifacts if a.role == "execution-profile")
        pin = cached["estimator_config"]["op_level"]["execution_profile"]
        if pin.get("sha256") != profile.sha256 or Path(pin.get("path", "")) != output_dir / "profile.json":
            return False
        _verify(output_dir, profile.model_copy(update={"path": "profile.json"}))
    except (StopIteration, KeyError, TypeError, AttributeError, ValueError, OSError):
        return False
    return True


def _drop_generated_library_path(config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of ``config`` without an ``fpm_parquet_path`` that an earlier ensure call generated.

    Such a path names a staged library of another materialization, in this cache or another one; keeping it would
    pair a new dataset revision with the old library. A path the user set elsewhere, even one inside a cache
    directory, is a genuine override and stays: only a root AISim staged counts as generated.
    """
    out = dict(config)
    estimator = out.get("estimator_config")
    interpolation = estimator.get("fpm_interpolation") if isinstance(estimator, Mapping) else None
    path = interpolation.get("fpm_parquet_path") if isinstance(interpolation, Mapping) else None
    # The generated file itself is a symlink to the dataset source. Resolve its
    # parent to identify the materialization without following that final link.
    if path is not None:
        parent = Path(path).expanduser().parent.resolve()
        if any(_is_generated_root(ancestor) for ancestor in (parent, *parent.parents)):
            interpolation = {k: v for k, v in interpolation.items() if k != "fpm_parquet_path"}
            out["estimator_config"] = {**out["estimator_config"], "fpm_interpolation": interpolation}
    return out


_GENERATED_ROOT_NAME = re.compile(r".+-[0-9a-f]{64}")


def _is_generated_root(path: str | Path) -> bool:
    """True for a staged library root that an earlier ensure call wrote, in any cache directory.

    A root has the shape ``<cache>/materialized/<source key>/<configuration>-<roots digest>`` and carries the staging
    marker that records AISim owns it (a cache directory that no longer exists leaves nothing to own, so the stale
    name still counts). A configuration saved with another cache directory names such a root too, and it must not
    outlive its pin as an explicit root; a manual file that only sits under ``materialized`` is not one.
    """
    resolved = Path(path).expanduser().resolve()
    if resolved.parent.parent.name != "materialized" or _GENERATED_ROOT_NAME.fullmatch(resolved.name) is None:
        return False
    return (resolved / STAGED_MARKER).is_file() or not resolved.exists()


def _drop_generated_roots(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """``config`` without the staged roots of earlier ensure calls in ``systems_paths``.

    A returned configuration lists its own staging root. Keeping it as an explicit root when the configuration is
    repinned would let that stale copy of the system definition and linked data shadow the roots a fresh load uses.
    """
    paths = config.get("systems_paths")
    if not paths:
        return config
    kept = [p for p in paths if not _is_generated_root(p)]
    return config if len(kept) == len(paths) else {**config, "systems_paths": kept}


def _load(
    loader,
    category: str,
    root: Path,
    entry: Entry,
    config: Mapping[str, Any],
    output_dir: Path,
    *,
    isolate_roots: bool = False,
):
    manifest_path = entry.manifest.path
    # Preparation saves absolute paths, including for a relative cache; compare and key on the same spelling.
    output_dir = output_dir.absolute()
    if category == "fpm":
        # The catalog's pinned digest, identity and purpose first (a partly refreshed local dataset must be refused);
        # the peek hashes no artifact, so a library entry is not hashed twice.
        peek = read_entry_manifest(root, entry, config, loader, roles=())
        if any(a.role == "fpm-library" for a in peek.artifacts):
            # Roots change the copied system definition and linked op data. Keep each ordered effective root
            # set in its own materialization so later calls cannot replace an earlier configuration's inputs.
            # A configuration returned by an earlier call lists that call's own staging root; it is derived from
            # the same inputs, not a different user root, so it must not key this materialization or shadow it.
            # With an op-latency pin the pinned root is the only source: no active root may fill a missing family.
            config = _drop_generated_roots(_drop_generated_library_path(config))
            roots = tuple(
                str(p)
                for p in _stageable_roots(config.get("systems_paths"), output_dir, include_active=not isolate_roots)
                if not _is_generated_root(p)
            )
            # The key covers the ordered roots and the copies and links the build would make, so a definition edited in
            # place, or a table that now resolves to another root, gets its own directory rather than replacing the
            # one an earlier configuration still points at. An unchanged plan reuses its directory.
            # It also covers the verified library pair, so a corrected pair published under the same configuration ID
            # (even under new file names) is staged beside the earlier one rather than over it.
            pair = sorted((a.role, a.path, a.sha256, a.bytes) for a in peek.artifacts if a.role in PREDICTION_ROLES)
            inputs = {
                "roots": roots,
                "staged": staged_inputs_digest(
                    entry.identity.hardware, entry.identity.framework, entry.identity.framework_version, roots
                ),
                "pair": pair,
            }
            roots_key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
            output_dir = output_dir.with_name(f"{output_dir.name}-{roots_key}")
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            # The tuple above is the complete staging input: staging must not expand the active roots again.
            return loader.load(
                root,
                manifest_path,
                output_dir=output_dir,
                config=config,
                systems_paths=roots,
                isolate_systems_roots=True,
            )
        manifest = read_entry_manifest(root, entry, config, loader)  # then verifies every declared artifact
        # Keyed by content: a configuration ID names an identity, so a dataset republished in place with corrected
        # measurements must not be served from the preparation of the old ones.
        content = hashlib.sha256()
        for artifact in sorted(manifest.artifacts, key=lambda a: a.path):
            content.update(
                f"{artifact.path}\0{artifact.runtime_path}\0{artifact.role}\0{artifact.sha256}\0{artifact.bytes}\n".encode()
            )
        output_dir = output_dir.with_name(f"{output_dir.name}-{content.hexdigest()[:16]}")
        output_dir = output_dir.absolute()
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        # The lock covers the check, the rebuild of a stale directory and the build itself: a second caller must not
        # remove a directory the first is still writing (config.json is written last).
        with _cache_lock(output_dir):
            cached = output_dir / "config.json"
            if cached.is_file():  # materialized by an earlier load; config.json is written last
                try:
                    cached_config = json.loads(cached.read_text())
                    if _materialization_is_intact(output_dir / "systems", manifest) and _cached_pin_is_valid(
                        output_dir, manifest, cached_config
                    ):
                        cached_config["systems_paths"] = [str(output_dir / "systems")]
                        return FpmData(manifest, config=cached_config)
                except ValueError:
                    pass
            if output_dir.exists():  # partial, stale or edited: rebuild, never serve or trip over it
                shutil.rmtree(output_dir)
            data = loader.load(root, manifest_path, output_dir=output_dir, config=config)
            if data.config is None:
                raise LoaderError("an fpm-profile artifact cannot be built into a best_available configuration")
            return data
    raise ValueError(f"no loader for category {category!r}")


def _one_entry_per_family(
    entries: list[Entry], selection: Selection, where: str, configuration_id: str | None = None
) -> list[Entry]:
    """Operation families are complementary inputs of one root; two entries for one family are a real conflict.

    ``configuration_id`` settles the conflict of its own family only: the other families are kept as they are, and
    one that still has several entries stays a conflict.
    """
    if configuration_id is not None:
        chosen = next((e for e in entries if e.configuration_id == configuration_id), None)
        if chosen is None:
            raise MissingDataError(
                f"no op-latency data for configuration {configuration_id} in {where}", selection=selection, source=where
            )
        entries = [e for e in entries if e.identity.operation_family != chosen.identity.operation_family or e == chosen]
    by_family: dict[str | None, list[Entry]] = {}
    for entry in entries:
        by_family.setdefault(entry.identity.operation_family, []).append(entry)
    clashes = {
        family: sorted(e.configuration_id for e in group) for family, group in by_family.items() if len(group) > 1
    }
    if clashes:
        detail = "; ".join(f"{family}: {', '.join(ids)}" for family, ids in sorted(clashes.items(), key=str))
        raise DataConflictError(
            f"several op-latency entries match one operation family for {_describe(selection)} in {where} ({detail}); "
            "set AISIM_DATA_CONFIGURATION (DataPin.configuration_id) to choose one"
        )
    return sorted(entries, key=lambda e: e.configuration_id)


def _declared_donors(
    root: Path,
    entries: list[Entry],
    manifests: list[Any],
    selection: Selection,
    fetch_donors: Callable[[Selection], Any] | None,
) -> list[Entry]:
    """Catalog entries of the versions that ``reuse.yaml`` declares as donors of the selected entries.

    The native database resolves a donor one hop from the requested version, only from files under the same
    root, and skips a donor whose file is absent, so a root without the donor version silently loses the measured
    coverage the dataset holds. A donor the catalog does not publish is skipped the same way. ``fetch_donors``
    downloads a donor selection (``None`` for a local root, which already holds everything).
    """
    chosen = {e.configuration_id for e in entries}
    donors: list[Entry] = []
    for entry, manifest in zip(entries, manifests, strict=True):
        declaration = next((a for a in manifest.artifacts if a.role == "reuse-declaration"), None)
        if declaration is None:
            continue
        for version in _donor_versions(yaml.safe_load(_verify(root, declaration).read_text(encoding="utf-8"))):
            wanted = replace(
                selection,
                configuration_ids=(),
                framework_version=version,
                operation_family=entry.identity.operation_family,
            )
            found = _matching_entries(root, wanted)
            if len(found) > 1:
                ids = ", ".join(sorted(e.configuration_id for e in found))
                raise DataConflictError(
                    f"{len(found)} op-latency entries match donor version {version} of {entry.configuration_id} "
                    f"({ids}); a donor must be one entry"
                )
            if not found or found[0].configuration_id in chosen:
                continue
            if fetch_donors is not None:
                fetch_donors(wanted)
            chosen.add(found[0].configuration_id)
            donors.append(found[0])
    return donors


def _collective_dependencies(
    root: Path,
    entries: list[Entry],
    config: Mapping[str, Any],
    selection: Selection,
    fetch_entries: Callable[[Selection], Any] | None,
) -> list[Entry]:
    """Entries of the collective libraries (``misc.nccl_version``, ``misc.oneccl_version``) the system spec names.

    The native communication readers resolve ``comm/nccl/<version>`` and ``comm/oneccl/<version>`` from the system
    spec, independently of the serving backend, so a root without them cannot answer collective queries even
    though the dataset publishes the tables. A library or version the dataset does not publish is skipped; the
    native query then reports the absent table.
    """
    spec_path = packaged_system_specs(PACKAGED_SYSTEMS_DIR).get(str(config.get("system")))
    if spec_path is None:
        return []
    misc = (yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}).get("misc") or {}
    chosen = {e.configuration_id for e in entries}
    found_entries: list[Entry] = []
    for library in ("nccl", "oneccl"):
        version = misc.get(f"{library}_version")
        if version is None:
            continue
        wanted = replace(
            selection,
            configuration_ids=(),
            framework=library,
            framework_version=str(version),
            operation_family=None,
        )
        found = [e for e in _matching_entries(root, wanted) if e.configuration_id not in chosen]
        if len(found) > 1:
            ids = ", ".join(sorted(e.configuration_id for e in found))
            raise DataConflictError(f"{len(found)} op-latency entries match {library} {version} ({ids})")
        if found:
            if fetch_entries is not None:
                fetch_entries(wanted)
            chosen.add(found[0].configuration_id)
            found_entries.append(found[0])
    return found_entries


def _shared_layer_sources(
    root: Path,
    entries: list[Entry],
    config: Mapping[str, Any],
    selection: Selection,
    fetch_entries: Callable[[Selection], Any] | None,
) -> list[Entry]:
    """Entries the native shared-layer resolver may read besides the requested version.

    With shared-layer reuse on (anything but an explicit ``enable_shared_layer: false``) the resolver fills
    missing tables from the same backend's strictly earlier versions (every family, including ones the requested
    version lacks) and from other backends the pinned ``perf_data_reuse_manifest`` allows this backend to consume.
    A root without them silently loses measured coverage the dataset holds, so they join it, fetched by the
    same pin. This is a superset of what one query reads; ``enable_shared_layer: false`` opts out.
    """
    version, system, backend = selection.framework_version, selection.hardware, selection.framework
    if config.get("enable_shared_layer") is False or not (version and system and backend):
        return []
    from aisim_core.sdk.perf_database import _version_sort_tuple

    catalog = _op_latency_index(root).entries
    allowed = _cross_backend_frameworks(root, system, backend)
    chosen = {e.configuration_id for e in entries}
    wanted: dict[tuple[str, str], list[Entry]] = {}
    for entry in catalog:
        identity = entry.identity
        if entry.configuration_id in chosen or identity.hardware != system:
            continue
        earlier = identity.framework == backend and _version_sort_tuple(
            identity.framework_version
        ) < _version_sort_tuple(version)
        cross = identity.framework in allowed and identity.operation_family != "comm"
        if earlier or cross:
            wanted.setdefault((identity.framework, identity.framework_version), []).append(entry)
    found: list[Entry] = []
    for (framework, source_version), group in sorted(wanted.items()):
        if fetch_entries is not None:
            fetch_entries(
                replace(
                    selection,
                    configuration_ids=(),
                    framework=framework,
                    framework_version=source_version,
                    operation_family=None,
                )
            )
        found.extend(group)
    return found


def _cross_backend_frameworks(root: Path, system: str, backend: str) -> set[str]:
    """Frameworks whose rows the pinned reuse manifest lets ``backend`` inherit on ``system``."""
    from aisim_core.data.op_schema import REUSE_MANIFEST_NAME

    path = Path(root) / DATA_PREFIX / f"{REUSE_MANIFEST_NAME}.parquet"
    if not path.is_file():
        return set()
    document = yaml.safe_load(lookup_yaml(path)) or {}
    allowed: set[str] = set()
    for group in document.get("groups") or []:
        if group.get("tier") in ("shared", "shared_fallback") and system in (group.get("systems") or ()):
            frameworks = set(group.get("frameworks") or ())
            if backend in frameworks:
                allowed |= frameworks
    return allowed - {backend, "nccl", "oneccl"}


def _donor_versions(document: Any) -> list[str]:
    """Sorted distinct ``from_version`` values of a parsed ``reuse.yaml``."""
    reuse = document.get("reuse") if isinstance(document, dict) else None
    if not isinstance(reuse, list) or not all(
        isinstance(e, dict) and isinstance(e.get("from_version"), str) and e["from_version"] for e in reuse
    ):
        raise LoaderError("reuse.yaml must hold a 'reuse' list whose entries name a from_version string")
    return sorted({e["from_version"] for e in reuse})


@contextmanager
def _cache_lock(target: Path):
    """An exclusive advisory lock for one cache key (``<target>.lock``), shared by threads and processes."""
    lock = target.with_name(f"{target.name}.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a+b") as handle:
        try:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        except ImportError:  # no flock (Windows): private staging still protects the build itself
            pass
        yield


def _load_op_latency(
    root: Path,
    entries: list[Entry],
    config: Mapping[str, Any],
    cache: Cache,
    source_key: str,
    selection: Selection,
    where: str,
    fetch_donors: Callable[[Selection], Any] | None = None,
) -> tuple[OpTreeLoad, list[str]]:
    """Compose every selected family, its declared donor versions and the system specs into one ``systems/`` root."""
    ids = [e.configuration_id for e in entries]
    try:
        # Verifies every artifact in the fetched tree, also when the composed root is reused.
        verify_op_latency_support(root)  # a reused root must pass the same checks as a cold build
        manifests = [read_entry_manifest(root, entry, config) for entry in entries]
        donors = _declared_donors(root, entries, manifests, selection, fetch_donors)
        donors += _collective_dependencies(root, entries + donors, config, selection, fetch_donors)
        donors += _shared_layer_sources(root, entries + donors, config, selection, fetch_donors)
        # Donors and collective libraries are other versions or libraries of the same system, so the version check
        # of ``config`` is not theirs.
        manifests += [read_entry_manifest(root, donor) for donor in donors]
        entries = [*entries, *donors]
        ids = [e.configuration_id for e in entries]
        # Keyed by content, not identity: a dataset republished in place (or a newer package) gets a new root.
        digest = op_latency_inputs_digest(root, manifests, PACKAGED_SYSTEMS_DIR)[:16]
        output_dir = cache.root / "materialized" / source_key / f"op-latency-{digest}"
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        # Validation, repair and build of one cache key are serialized: two callers that both found a root damaged must
        # not let the slower one delete what the faster one just rebuilt. The decision is made under the lock.
        with _cache_lock(output_dir):
            if (output_dir / "systems").is_dir():
                if composed_root_matches(output_dir / "systems", root, manifests, PACKAGED_SYSTEMS_DIR):
                    return OpTreeLoad(output_dir / "systems", len(entries), 0, 0, (), frozenset(), 0.0), ids
                shutil.rmtree(output_dir, ignore_errors=True)  # edited or damaged in place: rebuild from the dataset
            # Build in a directory private to this attempt and rename it, so a concurrent reader never sees a
            # half-written root.
            staging = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.", suffix=".tmp", dir=output_dir.parent))
            try:
                load_op_latency_tree(root, staging / "systems", entries=entries, packaged_systems=PACKAGED_SYSTEMS_DIR)
                try:
                    staging.rename(output_dir)
                except OSError:
                    if not (output_dir / "systems").is_dir():  # another attempt won the race: its root is equivalent
                        raise
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        return OpTreeLoad(output_dir / "systems", len(entries), 0, 0, (), frozenset(), 0.0), ids
    except (LoaderError, ValueError) as exc:
        raise MissingDataError(
            f"op-latency data {', '.join(ids)} in {where} failed verification: {exc}", selection=selection, source=where
        ) from exc


# Scalar settings an installed execution profile requires; the value that means "caller did not choose" per field.
_PROFILE_SCALARS = (
    ("estimation_mode", ("auto", None)),
    ("enable_shared_layer", (None,)),
    ("fallback_policy", (None,)),
    ("database_mode", (None,)),
)


def _adopt_prepared_settings(config: dict[str, Any], prepared: Mapping[str, Any], entry: Entry | None) -> None:
    """Carry the settings the validated prepared configuration requires into the caller's unspecified fields.

    Native construction rejects an execution profile under ``auto`` estimation, shared-layer reuse or the default
    correction. The prepared configuration passed those checks, so its choices fill what the caller left at the
    default, and an explicit incompatible choice is reported here, naming the entry, instead of failing later.
    """
    where = f"dataset entry {entry.configuration_id}" if entry is not None else "the dataset entry"
    for name, unspecified in _PROFILE_SCALARS:
        wanted = prepared.get(name)
        if wanted is None:
            continue
        have = config.get(name)
        if have in unspecified:
            config[name] = wanted
        elif have != wanted:
            raise DataConflictError(f"configuration sets {name}={have!r} but {where} requires {wanted!r}")
    wanted_correction = (prepared.get("estimator_config") or {}).get("correction")
    if wanted_correction is not None:
        estimator = config.setdefault("estimator_config", {})
        have_correction = estimator.get("correction")
        if have_correction is None:
            estimator["correction"] = copy.deepcopy(wanted_correction)
        elif bool(have_correction.get("enabled")) != bool(wanted_correction.get("enabled")):
            raise DataConflictError(f"configuration enables output correction but {where} requires it disabled")


def _wire(config: dict[str, Any], category: str, loaded: Any, entry: Entry | None) -> None:
    """Merge loaded inputs into ``config``, preserving explicit user choices."""
    if category == "op-latency":
        # Native construction tries later roots when required tables are missing.
        # A category pin must fail on that absence, never select unpinned tables.
        config["systems_paths"] = [str(loaded.root)]
        return
    loaded_config = loaded.config
    if loaded.kind == "library":
        # The loader built this from the user's configuration, so every user choice is kept.
        config.clear()
        config.update(loaded_config)
        return
    paths = [*(loaded_config.get("systems_paths") or ()), *(config.get("systems_paths") or ())]
    config["systems_paths"] = list(dict.fromkeys(str(p) for p in paths))
    wanted = loaded_config["estimator_config"]["op_level"]["execution_profile"]
    _adopt_prepared_settings(config, loaded_config, entry)
    estimator = config.setdefault("estimator_config", {})
    op_level = estimator.setdefault("op_level", {})
    user = op_level.get("execution_profile")
    if user is None:
        op_level["execution_profile"] = wanted
    elif user.get("sha256") not in (None, wanted.get("sha256")):
        raise DataConflictError(
            f"configuration pins execution profile {user.get('sha256')} but dataset entry "
            f"{entry.configuration_id} provides {wanted.get('sha256')}"
        )
    else:
        op_level["execution_profile"] = {**user, **wanted}
