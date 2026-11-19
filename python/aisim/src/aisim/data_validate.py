# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``aisim data validate``: name, manifest and catalog checks for a local dataset or a Hugging Face revision.

Local checks need no network. Every finding is ``{"check", "path", "message", ...}``; name findings add
``kind``, ``location``, ``value`` and ``canonical``. Category-specific entry checks (for example the FPM
profile checks) register through :func:`register_entry_check` and run for each manifest of their category.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args

from pydantic import ValidationError

from aisim_core.data import Category, CategoryIndex, Manifest, _verify, check_dataset, check_manifest
from aisim_core.data.fetch import Cache, Selection, SelectionError, Transport, fetch, resolve_revision
from aisim_core.data.loaders import LoaderError
from aisim_core.data.names_check import check_names, discover_manifests

#: Categories with no offline manifest checker; their manifests get name checks and digest checks only.
UNCHECKED_CATEGORIES = ("e2e", "eval", "systems")
_ERRORS = (ValueError, ValidationError, LoaderError, OSError)

#: ``fn(root, manifest_path, manifest) -> list of finding dicts``; keyed by manifest category.
EntryCheck = Callable[[Path, str, Manifest], list[dict[str, Any]]]
ENTRY_CHECKS: dict[str, list[EntryCheck]] = {}


def register_entry_check(category: str, check: EntryCheck) -> None:
    """Add a category-specific check that runs for every manifest of ``category``."""
    ENTRY_CHECKS.setdefault(category, []).append(check)


def _finding(check: str, path: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"check": check, "path": path, "message": message, **extra}


def validate_local(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path}: no such file or directory")
    findings: list[dict[str, Any]] = []
    skipped: set[str] = set()
    manifests, discovery_errors = discover_manifests(path)
    for message in discovery_errors:
        manifest_path, _, detail = message.partition(": ")
        findings.append(_finding("manifest", manifest_path, detail))
    root = _dataset_root(path, manifests)
    dataset_checked = False
    cataloged: set[str] = set()  # manifests check_dataset already checked

    if path.is_dir() and path == root and (root / "catalog" / "dataset.json").is_file():
        dataset_checked = True
        try:
            index = check_dataset(root, unchecked_categories=UNCHECKED_CATEGORIES)
            for reference in index.categories.values():
                entries = CategoryIndex.model_validate_json(_verify(root, reference.index).read_bytes())
                cataloged.update(entry.manifest.path for entry in entries.entries)
        except _ERRORS as exc:
            findings.append(_finding("dataset", "catalog/dataset.json", f"{type(exc).__name__}: {exc}"))
    for manifest_file in manifests:
        relative = manifest_file.relative_to(root).as_posix()
        try:
            manifest = Manifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))
        except _ERRORS as exc:
            findings.append(_finding("manifest", relative, f"{type(exc).__name__}: {exc}"))
            continue
        if manifest.category in UNCHECKED_CATEGORIES:
            skipped.add(manifest.category)
        elif relative not in cataloged:  # check_dataset already ran check_manifest for catalog entries
            try:
                check_manifest(root, relative)
            except _ERRORS as exc:
                findings.append(_finding("manifest", relative, f"{type(exc).__name__}: {exc}"))
        for check in ENTRY_CHECKS.get(manifest.category, ()):
            findings.extend(check(root, relative, manifest))

    name_findings, name_errors = check_names(path, root)
    for item in name_findings:
        findings.append(_finding("names", item.path, f"{item.finding.value!r} is not canonical", **_name_fields(item)))
    for message in name_errors:
        manifest_path, _, detail = message.partition(": ")
        findings.append(_finding("manifest", manifest_path, detail))
    return {
        "ok": not findings,
        "source": {"path": str(path)},
        "manifests": len(manifests),
        "dataset_catalog_checked": dataset_checked,
        "categories_without_offline_checker": sorted(skipped),
        "findings": findings,
    }


def _dataset_root(path: Path, manifests: list[Path]) -> Path:
    """The directory artifact paths are relative to: the nearest ancestor with a catalog, or one that holds
    the first manifest's first artifact; otherwise ``path`` itself (its folder for a file)."""
    start = path if path.is_dir() else path.parent
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (candidate / "catalog" / "dataset.json").is_file():
            return candidate
    for manifest_file in manifests[:1]:
        try:
            document = json.loads(manifest_file.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                break  # the manifest check reports it
            first = (document.get("artifacts") or [])[0]["path"]
        except (OSError, ValueError, LookupError, TypeError):
            break
        for candidate in candidates:
            if (candidate / first).is_file():
                return candidate
    return start


def _name_fields(item: Any) -> dict[str, Any]:
    f = item.finding
    return {"kind": f.kind, "location": f.location, "value": f.value, "canonical": f.canonical}


def validate_repo(repo: str, revision: str, *, transport: Transport, cache: Cache | None = None) -> dict[str, Any]:
    """Download every category of ``repo@revision`` into the cache, then validate the commit root locally."""
    cache = cache or Cache()
    sha = resolve_revision(transport, repo, revision)
    for category in get_args(Category):
        try:
            fetch(
                Selection(category=category, select_all=True), repo=repo, revision=sha, transport=transport, cache=cache
            )
        except SelectionError:  # the dataset has no such category (or it is empty)
            continue
    report = validate_local(cache.commit_root(repo, sha))
    report["source"] = {"repo": repo, "revision": sha, "requested_revision": revision}
    return report


def dumps(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, allow_nan=False) + "\n"
