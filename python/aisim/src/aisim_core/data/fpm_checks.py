# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FPM entry checks for dataset validation (a later ``aisim data validate`` calls these).

Each function returns findings and never raises for a bad entry, so a validator can report them all:

* :func:`check_fpm_entry`: the required roles are present exactly once, and the library and its sidecar match the
  entry manifest (size, SHA-256, row count, sidecar ``parquet_sha256`` and ``row_count``, optional ``resources``).
* :func:`check_fpm_identities`: one identity (model, hardware, framework, version, parallelism, precision) maps to
  at most one entry.
* :func:`check_fpm_dataset`: both checks over every entry of a local dataset root.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from pydantic import ValidationError

from aisim_core.data import Manifest, _file, _verify, configuration_id
from aisim_core.data.fpm_library import (
    EVALUATION_ROLES,
    LIBRARY_SCHEMA_ID,
    NATIVE_REQUIRED_COLUMNS,
    ROLE_LIBRARY,
    ROLE_METADATA,
    FpmContractError,
    read_library_metadata,
)


@dataclass(frozen=True)
class FpmFinding:
    code: str  # stable identifier, e.g. "missing-role"
    path: str  # manifest path the finding belongs to
    message: str

    def __str__(self) -> str:
        return f"{self.path}: [{self.code}] {self.message}"


def check_fpm_entry(root: Path, manifest_path: str, *, include_measurements: bool = False) -> list[FpmFinding]:
    """Check one entry manifest against the files under ``root``.

    Prediction-tier files (library and sidecar) are always verified. Evaluation-tier files are verified only with
    ``include_measurements``; a default fetch does not download them.
    """
    root = Path(root)
    found: list[FpmFinding] = []

    def add(code: str, message: str) -> None:
        found.append(FpmFinding(code, manifest_path, message))

    try:
        manifest = Manifest.model_validate_json(_file(root, manifest_path).read_bytes())
    except (OSError, ValueError, ValidationError) as exc:
        return [FpmFinding("invalid-manifest", manifest_path, str(exc).splitlines()[0])]
    if manifest.category != "fpm":
        add("wrong-category", f"expected category fpm, got {manifest.category!r}")
    if manifest.configuration_id != configuration_id(manifest.category, manifest.purpose, manifest.identity):
        add("configuration-identity", "configuration_id does not match the manifest identity")
    by_role = defaultdict(list)
    for artifact in manifest.artifacts:
        by_role[artifact.role].append(artifact)
    for role in (ROLE_LIBRARY, ROLE_METADATA):
        count = len(by_role.get(role, ()))
        if count != 1:
            add(
                "missing-role" if count == 0 else "duplicate-role",
                f"role {role!r} must appear exactly once, found {count}",
            )
    if by_role.get("measurement-data") and not by_role.get("measurement-manifest"):
        add("missing-role", "role 'measurement-manifest' is required when 'measurement-data' is present")
    library = next(iter(by_role.get(ROLE_LIBRARY, [])), None)
    sidecar = next(iter(by_role.get(ROLE_METADATA, [])), None)
    if len(by_role.get(ROLE_LIBRARY, [])) != 1 or len(by_role.get(ROLE_METADATA, [])) != 1:
        return found

    if library.artifact_schema != LIBRARY_SCHEMA_ID:
        add("library-schema", f"library schema {library.artifact_schema!r}; supported: {LIBRARY_SCHEMA_ID!r}")
    if library.format != "parquet" or library.rows is None:
        add("library-declaration", "fpm-library must declare format='parquet' and a row count")
        return found
    checked = [library, sidecar]
    if include_measurements:
        checked += [a for role in EVALUATION_ROLES for a in by_role.get(role, [])]
    for artifact in checked:
        try:
            _verify(root, artifact)
        except (OSError, ValueError) as exc:
            add("hash-mismatch", f"{artifact.role}: {exc}")
    if any(f.code == "hash-mismatch" for f in found):
        return found
    try:
        columns = set(pq.ParquetFile(_file(root, library.path)).schema_arrow.names)
    except (OSError, ValueError) as exc:
        add("library-columns", f"cannot read the library columns: {exc}")
        columns = None
    if columns is not None and (absent := [c for c in NATIVE_REQUIRED_COLUMNS if c not in columns]):
        add("library-columns", f"library lacks columns the native loader requires: {', '.join(absent)}")
    try:
        metadata = read_library_metadata(_file(root, sidecar.path))
    except (FpmContractError, ValueError) as exc:
        add("invalid-sidecar", str(exc))
        return found
    if metadata.get("parquet_sha256") not in (None, library.sha256):
        add("sidecar-hash", "sidecar parquet_sha256 differs from the library in the manifest")
    if library.rows is not None and metadata.get("row_count") != library.rows:
        add("sidecar-rows", f"sidecar row_count {metadata.get('row_count')} differs from library rows {library.rows}")
    return found


def identity_key(manifest: Manifest) -> str:
    """Stable key for the entry identity (the hash input without the settings that do not identify data)."""
    identity = manifest.identity
    return json.dumps(
        [
            identity.model,
            identity.hardware,
            identity.framework,
            identity.framework_version,
            identity.parallelism,
            identity.precision,
            identity.dp_attention,
            identity.multinode,
            identity.offload_mode,
        ],
        sort_keys=True,
        separators=(",", ":"),
    )


def check_fpm_identities(entries: Iterable[tuple[str, Manifest]]) -> list[FpmFinding]:
    """Reject two entries with one identity: callers could not choose between their libraries."""
    seen: dict[str, str] = {}
    found = []
    for path, manifest in entries:
        key = identity_key(manifest)
        if key in seen:
            found.append(
                FpmFinding("duplicate-identity", path, f"same identity as {seen[key]}; merge or remove one entry")
            )
        else:
            seen[key] = path
    return found


def check_fpm_dataset(root: Path, *, include_measurements: bool = False) -> list[FpmFinding]:
    """Check every FPM entry listed by ``catalog/fpm/index.json`` (or ``catalog/dataset.json``) of a local root."""
    from aisim_core.data import CategoryIndex, DatasetIndex

    root = Path(root)
    try:
        dataset = DatasetIndex.model_validate_json(_file(root, "catalog/dataset.json").read_bytes())
        reference = dataset.categories["fpm"]
        index = CategoryIndex.model_validate_json(_file(root, reference.index.path).read_bytes())
    except (KeyError, OSError, ValueError, ValidationError) as exc:
        return [FpmFinding("invalid-catalog", "catalog/dataset.json", f"cannot read the fpm category: {exc}")]
    found: list[FpmFinding] = []
    manifests = []
    for entry in index.entries:
        found += check_fpm_entry(root, entry.manifest.path, include_measurements=include_measurements)
        try:
            manifests.append(
                (entry.manifest.path, Manifest.model_validate_json(_file(root, entry.manifest.path).read_bytes()))
            )
        except (OSError, ValueError, ValidationError):
            continue  # already reported as invalid-manifest
    return found + check_fpm_identities(manifests)
