# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AISim FPM Gym (scripts/fpm_accuracy/models/aic_fpm_database.py); the Gym migrates onto this module later.

"""The FPM library contract: sidecar schema, staging, schema-6 recovery and resource precedence.

An FPM entry has two tiers (see :mod:`aisim_core.data.fpm_checks`):

* **prediction** (also the training input): the library ``fpm/fpm_forward_perf.parquet`` (schema
  ``aic_fpm_forward_perf/v6``) and its sidecar ``fpm/fpm_forward_perf.metadata.json``;
* **evaluation**: everything under ``measurements/``. Default fetching and loading never need it.

The sidecar may carry an optional ``resources`` block (memory and cache geometry, scheduler limits and
provenance) validated by :class:`aisim_core.fpm_profile.FpmResourceProfile`. At load, resources resolve as
user configuration, then sidecar ``resources``, then *pending*. Pending means missing: the error names the fields
to supply. Entries published before ``resources`` existed stay valid.

This module is offline and never changes performance-model behavior.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from aisim_core.fpm_profile import FpmDeploymentProfile, FpmResourceProfile

LIBRARY_SCHEMA_NAME = "aic_fpm_forward_perf"
LIBRARY_SCHEMA_ID = "aic_fpm_forward_perf/v6"
SUPPORTED_LIBRARY_VERSIONS = (6,)
LIBRARY_FILE = "fpm_forward_perf.parquet"
METADATA_FILE = "fpm_forward_perf.metadata.json"

ROLE_LIBRARY = "fpm-library"
ROLE_METADATA = "fpm-library-metadata"
ROLE_LEGACY_MANIFEST = "legacy-configuration-manifest"
PREDICTION_ROLES = (ROLE_LIBRARY, ROLE_METADATA)
EVALUATION_ROLES = ("measurement-data", "measurement-manifest", "measurement-provenance")

MEASUREMENT_MANIFEST_VERSION = 5
_TOPOLOGY = ("tp", "pp", "dp", "moe_tp", "moe_ep", "cp")


class FpmContractError(ValueError):
    """An FPM library, sidecar or resource declaration does not satisfy the contract."""


class FpmResourcesPendingError(FpmContractError):
    """Resources are required but neither the configuration nor the sidecar supplies them."""

    def __init__(self, message: str, fields: Sequence[str]):
        super().__init__(message)
        self.fields = tuple(fields)


# --------------------------------------------------------------------------- schema models (snapshotted)


class FpmLibraryMetadata(BaseModel):
    """``fpm/fpm_forward_perf.metadata.json`` for ``aic_fpm_forward_perf/v6``.

    Collectors add provenance keys over time, so unknown keys are allowed; the fields below are the ones
    AISim reads. ``resources`` is optional: user configuration overrides it, and an entry without it needs the
    user to supply resources before cache sizing or replay.
    """

    model_config = ConfigDict(extra="allow")

    schema_name: Literal["aic_fpm_forward_perf"]
    schema_version: Literal[6] = Field(description="Library schema version; only 6 is supported.")
    row_count: int = Field(gt=0, strict=True)
    parquet_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    system: str | None = None
    backend: str | None = None
    backend_version: str | None = None
    coordinate_system: str | None = None
    measurement_policy: str | None = None
    measurement_repeats: int | None = Field(default=None, strict=True)
    warmup_repeats: int | None = Field(default=None, strict=True)
    metadata_provenance: str | None = None
    configuration_selector: dict[str, JsonValue] | None = Field(
        default=None,
        json_schema_extra={"properties": {name: {"type": "integer", "minimum": 1} for name in _TOPOLOGY}},
    )
    resources: FpmResourceProfile | None = None

    @model_validator(mode="after")
    def _selector_topology(self):
        for name in _TOPOLOGY:
            value = (self.configuration_selector or {}).get(name)
            if name in (self.configuration_selector or {}) and (type(value) is not int or value < 1):
                raise ValueError(f"configuration_selector.{name} must be a positive integer")
        return self


class FpmLibraryRow(BaseModel):
    """Notes for one row of ``fpm_forward_perf.parquet`` (``aic_fpm_forward_perf/v6``).

    The table is read natively; this model records the columns AISim relies on. A table may carry more. The
    required fields are exactly the columns the native loader looks up unconditionally (see
    ``NATIVE_REQUIRED_COLUMNS``); single-sample legacy tables may omit the per-row measurement-policy columns.
    """

    model_config = ConfigDict(extra="allow")

    cell_id: str
    model_path: str
    system: str
    backend: str
    backend_version: str
    weight_quantization: str
    tp: int
    pp: int
    dp: int
    moe_tp: int
    moe_ep: int
    cp: int
    workload_kind: str
    batch_size: int = Field(ge=1)
    total_prefill_tokens: int = Field(ge=0)
    total_kv_read_tokens: int = Field(ge=0)
    partition_policy: str
    latency_ms: float
    gemm_quant_mode: str
    moe_quant_mode: str
    fmha_quant_mode: str
    comm_quant_mode: str
    kv_cache_dtype: str
    moe_backend: str
    attention_backend: str
    enable_wideep: bool
    enable_eplb: bool
    measurement_policy: str | None = None
    measurement_repeats: int | None = None
    warmup_repeats: int | None = None


NATIVE_REQUIRED_COLUMNS: tuple[str, ...] = tuple(
    name for name, field_info in FpmLibraryRow.model_fields.items() if field_info.is_required()
)


class FpmMeasurementFile(BaseModel):
    """One file listed by a measurement manifest."""

    model_config = ConfigDict(extra="allow")

    measurement_file_id: str
    role: Literal["truth", "window", "configuration"] | str
    path: str
    format: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0, strict=True)


class FpmMeasurementManifest(BaseModel):
    """``measurements/manifest.json`` (``measurement-manifest/v5``): the evaluation tier."""

    model_config = ConfigDict(extra="allow")

    manifest_version: Literal[5]
    measurement_artifact_id: str
    measurement_protocol_id: str
    model_id: str
    model_revision: str | None = None
    framework: str
    framework_version: str
    system: str | None = None
    parallelism: str | None = None
    files: list[FpmMeasurementFile] = Field(min_length=1)
    collection_runs: list[dict[str, JsonValue]] = Field(default_factory=list)


# --------------------------------------------------------------------------- reading


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if duplicated := sorted({key for key in keys if keys.count(key) > 1}):
        raise ValueError(f"duplicate JSON key {duplicated[0]!r}")
    return dict(pairs)


def read_library_metadata(path: Path) -> dict[str, Any]:
    """Read a sidecar strictly and check the library schema name, version and ``resources``."""
    try:
        raw = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, ValueError) as exc:
        raise FpmContractError(f"cannot read FPM sidecar {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise FpmContractError(f"FPM sidecar {path} must contain a JSON object")
    version = raw.get("schema_version")
    if raw.get("schema_name") != LIBRARY_SCHEMA_NAME:
        raise FpmContractError(f"FPM sidecar {path} is not an {LIBRARY_SCHEMA_NAME} sidecar")
    if type(version) is int and version >= 7:
        raise FpmContractError(
            f"FPM library schema v{version} is not supported yet (supported: v6); "
            "v7 and v8 library support is a planned follow-up"
        )
    if version not in SUPPORTED_LIBRARY_VERSIONS:
        raise FpmContractError(f"unsupported FPM library schema version {version!r}; supported: v6")
    try:
        FpmLibraryMetadata.model_validate(raw)
    except ValidationError as exc:
        raise FpmContractError(f"invalid FPM sidecar {path}: {_first_error(exc)}") from exc
    return raw


def _first_error(exc: ValidationError) -> str:
    error = exc.errors()[0]
    where = ".".join(str(part) for part in error["loc"])
    return f"{where}: {error['msg']}"


# --------------------------------------------------------------------------- schema-6 recovery


def recover_reconstructed_metadata(
    metadata: dict[str, Any], parquet_path: Path, *, parquet_sha256: str, metadata_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Complete historical schema-6 sidecars using explicit, verified row facts.

    The schema-6 collector's row writer and sidecar publisher declare balanced iteration coordinates and the
    single-sample policy (collector/fpm_forward/database.py at e8828036e3d32d8361e2abb819e0ee60e569f518). Only that
    contract is accepted, not inferred producer history. Provenance, unknown revisions and every existing
    declaration stay unchanged. Cost is one digest pass and a bounded scan of contract and selector columns; no
    measured latency is loaded or rewritten. Returns ``(metadata, recovered_fields)``.
    """
    if metadata.get("schema_version") != 6 or metadata.get("metadata_provenance") != "reconstructed_from_parquet":
        return metadata, {}
    declarations = ("coordinate_system", "measurement_policy", "system", "backend", "backend_version")
    if all(name in metadata for name in declarations):
        # Complete sidecars need no reconstruction proof; the native loader validates their declared contract.
        return metadata, {}
    if type(metadata["schema_version"]) is not int or metadata.get("schema_name") != LIBRARY_SCHEMA_NAME:
        raise FpmContractError("Reconstructed FPM sidecar requires the explicit schema-6 contract")
    if sha256_file(parquet_path) != parquet_sha256 or metadata.get("parquet_sha256") != parquet_sha256:
        raise FpmContractError("Reconstructed FPM sources differ from their pinned hashes")
    selector = metadata.get("configuration_selector")
    identity_fields = ("model_path", "system", "backend", "backend_version")
    if not isinstance(selector, dict) or not set(identity_fields) <= selector.keys():
        raise FpmContractError("Reconstructed FPM sidecar requires an explicit configuration selector")
    row_count = metadata.get("row_count")
    if type(row_count) is not int or row_count <= 0:
        raise FpmContractError("Reconstructed FPM sidecar requires a positive integer row_count")
    coordinate_fields = ("batch_size", "total_prefill_tokens", "total_kv_read_tokens")
    columns = list(
        dict.fromkeys(
            (
                *selector,
                *coordinate_fields,
                "partition_policy",
                "measurement_policy",
                "measurement_repeats",
                "warmup_repeats",
            )
        )
    )
    import pyarrow.parquet as pq

    try:
        parquet = pq.ParquetFile(parquet_path)
        if parquet.metadata.num_rows != row_count:
            raise FpmContractError("Reconstructed FPM sidecar row_count differs from parquet")
        if not set(columns) <= set(parquet.schema_arrow.names):
            raise FpmContractError("Reconstructed FPM sidecar requires explicit schema-6 row contract columns")
        for batch in parquet.iter_batches(batch_size=4096, columns=columns):
            for row in batch.to_pylist():
                for name, expected in selector.items():
                    if type(row[name]) is not type(expected) or row[name] != expected:
                        raise FpmContractError(f"Reconstructed FPM sidecar selector conflicts with row {name}")
                if any(not isinstance(row[name], str) or not row[name] for name in identity_fields):
                    raise FpmContractError("Reconstructed FPM rows require explicit nonempty deployment identity")
                if row["partition_policy"] != "balanced_v1":
                    raise FpmContractError("Reconstructed FPM rows require balanced_v1 partition_policy")
                if (
                    any(type(row[name]) is not int or row[name] < 0 for name in coordinate_fields)
                    or row["batch_size"] == 0
                ):
                    raise FpmContractError("Reconstructed FPM rows have invalid iteration-total coordinates")
                if (
                    row["measurement_policy"] != "dynamo_native_single_sample_v1"
                    or type(row["measurement_repeats"]) is not int
                    or row["measurement_repeats"] != 1
                    or type(row["warmup_repeats"]) is not int
                    or row["warmup_repeats"] != 0
                ):
                    raise FpmContractError("Reconstructed FPM rows require the explicit single-sample policy")
    except FpmContractError:
        raise
    except Exception as exc:
        raise FpmContractError("Cannot validate reconstructed FPM row contracts") from exc
    facts = {
        "coordinate_system": "iteration_totals_balanced_v1",
        "measurement_policy": "dynamo_native_single_sample_v1",
        "measurement_repeats": 1,
        "warmup_repeats": 0,
        **{name: selector[name] for name in identity_fields[1:]},
    }
    for name, value in {**selector, **facts}.items():
        if name in metadata and (type(metadata[name]) is not type(value) or metadata[name] != value):
            raise FpmContractError(f"Reconstructed FPM sidecar conflicts with row-derived {name}")
    recovered = {name: value for name, value in facts.items() if name not in metadata}
    return {**metadata, **recovered}, recovered


# --------------------------------------------------------------------------- staging


@dataclass
class StagedFpmLibrary:
    """A ``systems/`` root holding one verified library and its (recovered) sidecar."""

    systems_root: Path
    parquet_path: Path  # staged, resolves to the verified source file
    metadata_path: Path
    metadata: dict[str, Any]  # effective sidecar, including recovered fields and ``resources`` when declared
    diagnostics: dict[str, Any]
    _temporary: TemporaryDirectory | None = field(default=None, repr=False)

    @property
    def resources(self) -> dict[str, Any] | None:
        return self.metadata.get("resources")

    def close(self) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None


def stage_fpm_library(
    parquet_path: str | Path,
    metadata_path: str | Path,
    *,
    system: str,
    backend: str,
    backend_version: str,
    output_dir: str | Path | None = None,
    systems_paths: Sequence[str | Path] | None = None,
    parquet_sha256: str | None = None,
    metadata_sha256: str | None = None,
    artifact_id: str | None = None,
) -> StagedFpmLibrary:
    """Stage a library and its sidecar into a ``systems/`` root for ``best_available``.

    The system definition and the other data of the selected backend version are linked from ``systems_paths``
    (the active systems roots follow as a fallback), so operation data stays reachable while the FPM
    directory is isolated.
    Sources outside the destination are linked, never modified; a schema-6 sidecar that needs recovery is
    rewritten in the staged copy. A recovered sidecar inside the destination is copied before replacement;
    the Parquet input must resolve outside the destination. ``output_dir`` is created (a previous staging there
    is replaced); without it a temporary root
    is used and removed by :meth:`StagedFpmLibrary.close`. Native validation still decides whether the staged
    table supports the requested estimator and workload.
    """
    system = _safe_component(system, "system")
    parquet, sidecar = Path(parquet_path).resolve(), Path(metadata_path).resolve()
    destination = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    if destination is not None and parquet.is_relative_to(destination):
        raise FpmContractError("FPM parquet input must resolve outside output_dir")
    systems_paths = _stageable_roots(systems_paths, output_dir)
    if not parquet.is_file() or not sidecar.is_file():
        raise FpmContractError(f"FPM library {artifact_id or parquet} is not materialized locally")
    metadata = read_library_metadata(sidecar)
    actual_parquet_sha256, actual_metadata_sha256 = sha256_file(parquet), sha256_file(sidecar)
    if parquet_sha256 not in (None, actual_parquet_sha256) or metadata_sha256 not in (None, actual_metadata_sha256):
        raise FpmContractError(f"FPM library {artifact_id or parquet} differs from its supplied hash")
    parquet_sha256, metadata_sha256 = actual_parquet_sha256, actual_metadata_sha256
    _check_sidecar_describes_library(metadata, parquet, parquet_sha256)
    metadata, recovered = recover_reconstructed_metadata(
        metadata, parquet, parquet_sha256=parquet_sha256, metadata_sha256=metadata_sha256
    )
    text = None
    recovery = None
    if recovered:
        text = json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
        recovery = {
            "policy": "schema6_reconstructed_row_contract_v1",
            "source_metadata_sha256": metadata_sha256,
            "staged_metadata_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "recovered_fields": recovered,
        }
    source_yaml = _system_definition(system, systems_paths)
    marker = {
        "system": system,
        "backend": backend,
        "backend_version": backend_version,
        "parquet_sha256": parquet_sha256,
        "metadata_sha256": metadata_sha256,
        "systems_paths": [str(p) for p in systems_paths] if systems_paths is not None else None,
        "parquet_path": str(parquet),
        "metadata_path": str(sidecar),
        "system_definition": {"path": str(source_yaml), "sha256": sha256_file(source_yaml)},
    }
    temporary = None
    if output_dir is None:
        temporary = TemporaryDirectory(prefix="aisim-fpm-")
        root = Path(temporary.name)
        build = root
    else:
        root = Path(output_dir).expanduser().absolute()
        _check_staging_destination(root)
        root.parent.mkdir(parents=True, exist_ok=True)
        # Unique per call: concurrent stagings of one destination never share or delete a build.
        build = Path(mkdtemp(prefix=f".{root.name}.staging-", dir=root.parent))
    try:
        target = _prepare_system_overlay(build, source_yaml, backend, backend_version, systems_paths)
        (target / LIBRARY_FILE).symlink_to(parquet)
        if text is not None:
            (target / METADATA_FILE).write_text(text, encoding="utf-8")
        elif destination is not None and sidecar.is_relative_to(destination):
            # A recovered sidecar is a regular file in the old staging, which replacement removes.
            shutil.copyfile(sidecar, target / METADATA_FILE)
        else:
            (target / METADATA_FILE).symlink_to(sidecar)
        if output_dir is not None:
            # The selected overlay links, not only the inputs, identify a reusable staging: a root added to or
            # reordered in front of the fallbacks can change which file a link names.
            marker["overlay"] = _overlay_links(build)
            (build / STAGED_MARKER).write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")
            with _INSTALL_LOCK:  # one installer or reader of a destination at a time within the process
                if _check_staging_destination(root) == marker:  # an earlier verified staging: reuse it
                    shutil.rmtree(build, ignore_errors=True)
                else:
                    _replace_staged_root(build, root, marker)
                return _staged(root, metadata, recovery, parquet_sha256, artifact_id, None)
    except BaseException:
        if temporary is not None:
            temporary.cleanup()
        else:
            shutil.rmtree(build, ignore_errors=True)
        raise
    return _staged(root, metadata, recovery, parquet_sha256, artifact_id, temporary)


STAGED_MARKER = ".staged.json"


def _stageable_roots(systems_paths: Sequence[str | Path] | None, output_dir: str | Path | None) -> list[Path]:
    """Ordered effective roots, including active defaults, canonicalized and excluding the destination."""
    from aisim_core.sdk.perf_database import _normalize_systems_paths, get_systems_paths

    destination = Path(output_dir).expanduser().absolute().resolve() if output_dir is not None else None
    roots = [
        Path(r).expanduser().resolve()
        for p in [*(systems_paths or ()), *get_systems_paths()]
        for r in _normalize_systems_paths([os.fspath(p)])
    ]
    return list(dict.fromkeys(r for r in roots if r != destination))


class _StagingMarker(BaseModel):
    """Recognize directories owned by this staging implementation before allowing replacement."""

    model_config = ConfigDict(extra="forbid", strict=True)

    system: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    backend_version: str = Field(min_length=1)
    parquet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    systems_paths: list[str] | None
    parquet_path: str = Field(min_length=1)
    metadata_path: str = Field(min_length=1)
    system_definition: dict[str, str] | None = None
    overlay: list[list[str]] | None = None


def _overlay_links(build: Path) -> list[list[str]]:
    """Every symlink of a staged root as sorted ``[relative path, target]`` pairs."""
    links = []
    for directory, names, files in os.walk(build):
        for name in (*names, *files):
            path = Path(directory) / name
            if path.is_symlink():
                links.append([path.relative_to(build).as_posix(), os.readlink(path)])
    return sorted(links)


def _check_sidecar_describes_library(metadata: Mapping[str, Any], parquet: Path, parquet_sha256: str) -> None:
    """A declared digest and the row count must describe the staged table, not another one."""
    import pyarrow.parquet as pq

    if metadata.get("parquet_sha256") not in (None, parquet_sha256):
        raise FpmContractError("FPM sidecar parquet_sha256 differs from the library")
    try:
        rows = pq.ParquetFile(parquet).metadata.num_rows
    except Exception as exc:
        raise FpmContractError(f"cannot read the FPM library {parquet}: {exc}") from exc
    if metadata.get("row_count") != rows:
        raise FpmContractError(f"FPM sidecar row_count {metadata.get('row_count')} differs from the library ({rows})")


def _staged_marker(root: Path) -> dict[str, Any] | None:
    try:
        return _StagingMarker.model_validate_json((root / STAGED_MARKER).read_bytes()).model_dump()
    except (OSError, ValueError):
        return None


def _check_staging_destination(root: Path) -> dict[str, Any] | None:
    marker = _staged_marker(root)
    if root.is_symlink() or (root.exists() and (not root.is_dir() or (any(root.iterdir()) and marker is None))):
        raise FpmContractError(f"refusing to replace {root}: nonempty output_dir requires a valid staging marker")
    return marker


_INSTALL_LOCK = threading.Lock()


def _replace_staged_root(build: Path, root: Path, marker: Mapping[str, Any]) -> None:
    """Retain the previous root until installation succeeds, restoring it on failure.

    The caller serializes installation within a process (``_INSTALL_LOCK``). Across processes, a concurrent
    winner that installed the same staging is accepted; any other filesystem failure is a
    :class:`FpmContractError`.
    """
    _check_staging_destination(root)
    backup_dir = Path(mkdtemp(prefix=f".{root.name}.backup-", dir=root.parent))
    backup = backup_dir / "previous"
    try:
        if root.exists():
            os.replace(root, backup)
        os.replace(build, root)
    except BaseException as exc:
        if backup.exists() and not root.exists():
            # If restoration also fails, retain the backup for recovery.
            os.replace(backup, root)
        shutil.rmtree(backup_dir, ignore_errors=True)
        if isinstance(exc, OSError):
            if _staged_marker(root) == marker:
                shutil.rmtree(build, ignore_errors=True)
                return
            raise FpmContractError(f"cannot install the staged FPM root {root}: {exc}") from exc
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)


def _staged(root, metadata, recovery, parquet_sha256, artifact_id, temporary) -> StagedFpmLibrary:
    parquets = list(root.glob(f"**/{LIBRARY_FILE}"))
    if len(parquets) != 1:
        raise FpmContractError("the staged FPM root must contain exactly one library")
    return StagedFpmLibrary(
        systems_root=root,
        parquet_path=parquets[0],
        metadata_path=parquets[0].with_name(METADATA_FILE),
        metadata=metadata,
        diagnostics={
            "fpm_artifact_id": artifact_id,
            "fpm_parquet_sha256": parquet_sha256,
            "fpm_row_count": metadata.get("row_count"),
            **({"fpm_metadata_recovery": recovery} if recovery is not None else {}),
        },
        _temporary=temporary,
    )


def _system_definition(system: str, systems_paths: Sequence[str | Path]) -> Path:
    source_yaml = next(
        (Path(root) / f"{system}.yaml" for root in systems_paths if (Path(root) / f"{system}.yaml").is_file()),
        None,
    )
    if source_yaml is None:
        raise FpmContractError(f"AISim has no system definition for {system!r}")
    return source_yaml


def _prepare_system_overlay(
    systems_root: Path, source_yaml: Path, backend: str, backend_version: str, systems_paths: Sequence[str | Path]
) -> Path:
    import yaml

    target_yaml = systems_root / source_yaml.name
    shutil.copy2(source_yaml, target_yaml)
    spec = yaml.safe_load(target_yaml.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not isinstance(spec.get("data_dir"), str):
        raise FpmContractError(f"AISim system definition {source_yaml} has no data_dir")
    data_dir = _safe_relative_path(spec["data_dir"], "system data_dir")
    target_data_root = systems_root / data_dir
    backend_name = _safe_component(backend, "backend")
    version = _safe_component(backend_version, "backend version")
    target_version = target_data_root / backend_name / version
    target_version.mkdir(parents=True)
    for root in systems_paths:
        source_data_root = (Path(root) / data_dir).resolve()
        if source_data_root.is_dir():
            _link_overlay_entries(source_data_root, target_data_root, backend_name, version)
    return target_version


def _link_overlay_entries(source_root: Path, target_root: Path, backend: str, version: str) -> None:
    """Expose op/SOL data while keeping the selected FPM directory isolated."""
    for entry in source_root.iterdir():
        if entry.name != backend:
            _link_if_missing(entry, target_root / entry.name)
    source_backend, target_backend = source_root / backend, target_root / backend
    if not source_backend.is_dir():
        return
    for entry in source_backend.iterdir():
        if entry.name != version:
            _link_if_missing(entry, target_backend / entry.name)
    source_version, target_version = source_backend / version, target_backend / version
    if not source_version.is_dir():
        return
    for entry in source_version.iterdir():
        if entry.name not in {LIBRARY_FILE, METADATA_FILE}:
            _link_if_missing(entry, target_version / entry.name)


def _link_if_missing(source: Path, target: Path) -> None:
    if not target.exists() and not target.is_symlink():
        try:
            resolved = source.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise FpmContractError(f"cannot resolve FPM overlay source {source}: {exc}") from exc
        target.symlink_to(resolved, target_is_directory=resolved.is_dir())


def _safe_relative_path(value: Any, field_name: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise FpmContractError(f"AISim {field_name} must be a safe relative path")
    return path


def _safe_component(value: Any, field_name: str) -> str:
    component = str(value)
    if component in {"", ".", ".."} or Path(component).name != component:
        raise FpmContractError(f"AISim {field_name} must be one safe path component")
    return component


# --------------------------------------------------------------------------- resources

_RESOURCE_REQUIRED = ("cache_layout", "max_num_tokens", "max_batch_size", "provenance")
_MEMORY_FIELDS = ("weights_bytes", "activations_bytes", "runtime_overhead_bytes", "comm_overhead_bytes")


def missing_resource_fields(resources: Mapping[str, Any] | None) -> list[str]:
    """Names of the fields still to supply for a valid ``resources`` block (``None`` means all)."""
    given = resources or {}
    missing = [name for name in _RESOURCE_REQUIRED if given.get(name) is None]
    layout = given.get("cache_layout")
    if layout == "linear" and given.get("kv_bytes_per_token") is None:
        missing.append("kv_bytes_per_token")
    elif layout == "grouped" and not given.get("cache_groups"):
        missing.append("cache_groups")
    elif layout is None:
        missing.append("kv_bytes_per_token (linear) or cache_groups (grouped)")
    return missing


def missing_memory_fields(resources: FpmResourceProfile | None) -> list[str]:
    """What timing-only resources still lack before cache sizing or replay (empty when ready)."""
    if resources is None or resources.memory_ready:
        return []
    if resources.cache_layout == "unmodeled":
        return ["cache_layout (a supported layout; unmodeled supports timing queries only)"]
    return ["runtime_memory, or all of " + ", ".join(_MEMORY_FIELDS)]


def pending_resources_error(
    resources: Mapping[str, Any] | None, *, where: str, sidecar: str = METADATA_FILE
) -> FpmResourcesPendingError:
    fields = missing_resource_fields(resources)
    return FpmResourcesPendingError(
        f"FPM resources are pending for {where}; supply these fields in the deployment `resources` of the "
        f"fpm_profile, or publish them in a `resources` block of {sidecar}: {', '.join(fields)}. "
        "These fields are needed for cache sizing and replay, along with runtime_memory (or all four declared "
        "non-KV byte fields). Plain autoregressive timing queries do not require resources",
        fields,
    )


@dataclass(frozen=True)
class ResolvedResources:
    resources: FpmResourceProfile | None
    source: Literal["user", "metadata", "pending"]


def resolve_resources(
    user: Mapping[str, Any] | FpmResourceProfile | None,
    metadata: Mapping[str, Any] | None,
    *,
    where: str = "this deployment",
    required: bool = False,
) -> ResolvedResources:
    """Resolve one deployment's resources: user configuration > sidecar ``resources`` > pending.

    The first available block is used whole (never merged field by field), so the source stays unambiguous.
    Pending resolves to ``None`` unless ``required``, in which case the error names the fields to supply.
    """
    if user is not None:
        return ResolvedResources(_validated(user, where, "configuration"), "user")
    declared = (metadata or {}).get("resources")
    if declared is not None:
        return ResolvedResources(_validated(declared, where, METADATA_FILE), "metadata")
    if required:
        raise pending_resources_error(None, where=where)
    return ResolvedResources(None, "pending")


def _validated(value: Mapping[str, Any] | FpmResourceProfile, where: str, source: str) -> FpmResourceProfile:
    if isinstance(value, FpmResourceProfile):
        return value
    try:
        return FpmResourceProfile.model_validate(value)
    except ValidationError as exc:
        missing = missing_resource_fields(value) if isinstance(value, Mapping) else []
        hint = f"; missing: {', '.join(missing)}" if missing else ""
        raise FpmContractError(f"invalid resources from {source} for {where}: {_first_error(exc)}{hint}") from exc


_IDENTITY_KEYS = ("system", "backend", "backend_version", "worker_type")
# A selector that pins precision matches only a deployment that establishes the same precision.
_PRECISION_KEYS = (
    "weight_quantization",
    "gemm_quant_mode",
    "moe_quant_mode",
    "fmha_quant_mode",
    "comm_quant_mode",
    "kv_cache_dtype",
)


def resource_metadata_for_deployment(
    profile: Mapping[str, Any],
    deployment: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
    identity: Mapping[str, str] | None = None,
) -> Mapping[str, Any] | None:
    """Return sidecar resources only for the selected model and deployment."""
    selector = (metadata or {}).get("configuration_selector") or {}
    if not selector and _distinct_deployments(profile) > 1:
        # Without a selector the sidecar does not say which deployment it describes; a profile with several
        # distinct deployments keeps each of them pending rather than guessing.
        return None
    applies = all(deployment.get(name, 1) == selector[name] for name in _TOPOLOGY if name in selector)
    wanted = {**{k: selector[k] for k in _IDENTITY_KEYS if k in selector}, **(identity or {})}
    applies = applies and all(
        deployment.get(name) in (None, value)
        for name, value in wanted.items()
        if name in _IDENTITY_KEYS and name != "worker_type"
    )
    # A shared deployment can serve other roles, so it must not inherit role-specific evidence.
    if "worker_type" in wanted:
        applies = applies and deployment.get("worker_type") == wanted["worker_type"]
    # Selectors also pin precision, implementation and model identity. Use the profile's own defaults for
    # omitted deployment fields; never attach resources from a differently selected configuration.
    selected_identity = {
        **{name: value for name, value in profile.items() if name != "deployments"},
        "model_path": profile.get("model"),
        **{
            name: info.get_default()
            for name, info in FpmDeploymentProfile.model_fields.items()
            if not info.is_required() and name not in {"resources", "execution_profile"}
        },
        **deployment,
    }
    applies = applies and all(
        selected_identity.get(name) == value
        for name, value in selector.items()
        if name not in _TOPOLOGY
        and name not in _IDENTITY_KEYS
        and (name in selected_identity or name in _PRECISION_KEYS)
    )
    return metadata if applies else None


def _distinct_deployments(profile: Mapping[str, Any]) -> int:
    deployments = profile.get("deployments")
    if not isinstance(deployments, list):
        return 1
    keys = {
        json.dumps([d.get(name, 1) for name in _TOPOLOGY] + [d.get("worker_type")], default=str)
        for d in deployments
        if isinstance(d, Mapping)
    }
    return len(keys)


def complete_fpm_profile(
    profile: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
    *,
    identity: Mapping[str, str] | None = None,
    required: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """Fill deployments that lack ``resources`` from the sidecar; the user's block always wins.

    ``profile`` is the user's ``fpm_profile`` dictionary (deployments may omit ``resources``). A sidecar block
    applies to the deployment its ``configuration_selector`` names (every incomplete deployment when the sidecar
    has no selector) and only to deployments of the library's own ``identity`` (system, backend, backend_version);
    a deployment that names another hardware, version or role is left to its own resources. A selector that names
    a role requires that explicit deployment role; an omitted or null role cannot inherit role-specific resources.
    Returns the completed dictionary and the source of each deployment's resources.
    A deployment with neither raises :class:`FpmResourcesPendingError` when ``required``; otherwise its
    resources stay absent and its source is ``pending``.
    """
    out = json.loads(json.dumps(profile, allow_nan=False))
    deployments = out.get("deployments")
    if not isinstance(deployments, list) or not deployments:
        raise FpmContractError("fpm_profile requires at least one deployment")
    sources: list[str] = []
    for index, deployment in enumerate(deployments):
        if not isinstance(deployment, dict):
            raise FpmContractError("fpm_profile deployments must be objects")
        where = (
            f"deployment {index} (tp={deployment.get('tp')}, dp={deployment.get('dp')}, ep={deployment.get('moe_ep')})"
        )
        resolved = resolve_resources(
            deployment.get("resources"),
            resource_metadata_for_deployment(out, deployment, metadata, identity),
            where=where,
            required=required,
        )
        if resolved.resources is not None:
            deployment["resources"] = resolved.resources.model_dump(mode="json")
        sources.append(resolved.source)
    return out, sources


def with_estimator_path(config: dict[str, Any], parquet_path: Path) -> None:
    """Point the interpolation estimator at the staged library unless the user chose a path."""
    estimator = config.get("estimator_config")
    if estimator is None:
        estimator = {}
    elif not isinstance(estimator, Mapping):
        raise FpmContractError("estimator_config must be a mapping")
    estimator = dict(estimator)
    interpolation = estimator.get("fpm_interpolation")
    if interpolation is None:
        interpolation = {}
    elif not isinstance(interpolation, Mapping):
        raise FpmContractError("estimator_config.fpm_interpolation must be a mapping")
    interpolation = dict(interpolation)
    interpolation.setdefault("fpm_parquet_path", str(parquet_path))
    estimator["fpm_interpolation"] = interpolation
    config["estimator_config"] = estimator
