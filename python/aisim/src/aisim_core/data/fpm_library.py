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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from aisim_core.fpm_profile import FpmResourceProfile

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
    schema_version: int = Field(description="Library schema version; 6 is supported, 7 and 8 are follow-ups.")
    row_count: int = Field(gt=0)
    parquet_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    system: str | None = None
    backend: str | None = None
    backend_version: str | None = None
    coordinate_system: str | None = None
    measurement_policy: str | None = None
    measurement_repeats: int | None = None
    warmup_repeats: int | None = None
    metadata_provenance: str | None = None
    configuration_selector: dict[str, JsonValue] | None = None
    resources: FpmResourceProfile | None = None


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
    batch_size: int
    total_prefill_tokens: int
    total_kv_read_tokens: int
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
    bytes: int = Field(ge=0)


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


def read_library_metadata(path: Path) -> dict[str, Any]:
    """Read a sidecar strictly and check the library schema name, version and ``resources``."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)
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
    Original files are linked, never modified; only a schema-6 sidecar that needs recovery is rewritten, in the
    staged copy. ``output_dir`` is created (a previous staging there is replaced); without it a temporary root
    is used and removed by :meth:`StagedFpmLibrary.close`. Native validation still decides whether the staged
    table supports the requested estimator and workload.
    """
    parquet, sidecar = Path(parquet_path).resolve(), Path(metadata_path).resolve()
    systems_paths = _stageable_roots(systems_paths, output_dir)
    if not parquet.is_file() or not sidecar.is_file():
        raise FpmContractError(f"FPM library {artifact_id or parquet} is not materialized locally")
    metadata = read_library_metadata(sidecar)
    parquet_sha256 = parquet_sha256 or sha256_file(parquet)
    metadata_sha256 = metadata_sha256 or sha256_file(sidecar)
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
    marker = {
        "system": system,
        "backend": backend,
        "backend_version": backend_version,
        "parquet_sha256": parquet_sha256,
        "metadata_sha256": metadata_sha256,
        "systems_paths": [str(p) for p in systems_paths] if systems_paths is not None else None,
        "parquet_path": str(parquet),
        "metadata_path": str(sidecar),
    }
    temporary = None
    if output_dir is None:
        temporary = TemporaryDirectory(prefix="aisim-fpm-")
        root = Path(temporary.name)
        build = root
    else:
        root = Path(output_dir).absolute()
        build = root.with_name(f".{root.name}.{os.getpid()}.staging")
        if _staged_marker(root) == marker:  # an earlier verified staging of the same files: reuse it
            return _staged(root, metadata, recovery, parquet_sha256, artifact_id, None)
        shutil.rmtree(build, ignore_errors=True)
        build.mkdir(parents=True)
    try:
        target = _prepare_system_overlay(build, system, backend, backend_version, systems_paths)
        (target / LIBRARY_FILE).symlink_to(parquet)
        if text is not None:
            (target / METADATA_FILE).write_text(text, encoding="utf-8")
        else:
            (target / METADATA_FILE).symlink_to(sidecar)
        if output_dir is not None:
            (build / STAGED_MARKER).write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")
            if root.exists():
                shutil.rmtree(root)
            os.replace(build, root)
    except BaseException:
        if temporary is not None:
            temporary.cleanup()
        else:
            shutil.rmtree(build, ignore_errors=True)
        raise
    return _staged(root, metadata, recovery, parquet_sha256, artifact_id, temporary)


STAGED_MARKER = ".staged.json"


def _stageable_roots(systems_paths: Sequence[str | Path] | None, output_dir: str | Path | None) -> list[Path] | None:
    """Systems roots as the canonical resolver reads them (``default``, ``~``), minus the root being replaced."""
    if systems_paths is None:
        return None
    from aisim_core.sdk.perf_database import _normalize_systems_paths

    destination = Path(output_dir).expanduser().absolute().resolve() if output_dir is not None else None
    roots = [Path(r).expanduser() for p in systems_paths for r in _normalize_systems_paths([os.fspath(p)])]
    return [r for r in roots if destination is None or r.absolute().resolve() != destination]


def _staged_marker(root: Path) -> Any:
    try:
        return json.loads((root / STAGED_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


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


def _prepare_system_overlay(
    systems_root: Path, system: str, backend: str, backend_version: str, systems_paths: Sequence[str | Path] | None
) -> Path:
    import yaml

    from aisim_core.sdk import perf_database

    # The caller's roots first; the active defaults still supply a system definition they do not carry.
    systems_paths = [*(systems_paths or ()), *perf_database.get_systems_paths()]
    source_yaml = next(
        (Path(root) / f"{system}.yaml" for root in systems_paths if (Path(root) / f"{system}.yaml").is_file()),
        None,
    )
    if source_yaml is None:
        raise FpmContractError(f"AISim has no system definition for {system!r}")
    target_yaml = systems_root / source_yaml.name
    shutil.copy2(source_yaml, target_yaml)
    spec = yaml.safe_load(target_yaml.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not isinstance(spec.get("data_dir"), str):
        raise FpmContractError(f"AISim system definition {source_yaml} has no data_dir")
    data_dir = _safe_relative_path(spec["data_dir"], "system data_dir")
    source_data_root = (source_yaml.parent / data_dir).resolve()
    target_data_root = systems_root / data_dir
    backend_name = _safe_component(backend, "backend")
    version = _safe_component(backend_version, "backend version")
    target_version = target_data_root / backend_name / version
    target_version.mkdir(parents=True)
    if source_data_root.is_dir():
        _link_overlay_entries(source_data_root, target_data_root, backend_name, version)
    return target_version


def _link_overlay_entries(source_root: Path, target_root: Path, backend: str, version: str) -> None:
    """Expose op/SOL data while keeping the selected FPM directory isolated."""
    for entry in source_root.iterdir():
        if entry.name != backend:
            (target_root / entry.name).symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
    source_backend, target_backend = source_root / backend, target_root / backend
    if not source_backend.is_dir():
        return
    for entry in source_backend.iterdir():
        if entry.name != version:
            (target_backend / entry.name).symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
    source_version, target_version = source_backend / version, target_backend / version
    if not source_version.is_dir():
        return
    for entry in source_version.iterdir():
        if entry.name not in {LIBRARY_FILE, METADATA_FILE}:
            (target_version / entry.name).symlink_to(entry.resolve(), target_is_directory=entry.is_dir())


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
        "Timing queries need only these; cache sizing and replay also need runtime_memory (or all four declared "
        "non-KV byte fields)",
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
        missing = missing_resource_fields(value)
        hint = f"; missing: {', '.join(missing)}" if missing else ""
        raise FpmContractError(f"invalid resources from {source} for {where}: {_first_error(exc)}{hint}") from exc


_TOPOLOGY = ("tp", "pp", "dp", "moe_tp", "moe_ep", "cp")


_IDENTITY_KEYS = ("system", "backend", "backend_version", "worker_type")


def complete_fpm_profile(
    profile: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
    *,
    identity: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Fill deployments that lack ``resources`` from the sidecar; the user's block always wins.

    ``profile`` is the user's ``fpm_profile`` dictionary (deployments may omit ``resources``). A sidecar block
    applies to the deployment its ``configuration_selector`` names (every incomplete deployment when the sidecar
    has no selector) and only to deployments of the library's own ``identity`` (system, backend, backend_version);
    a deployment that names another hardware, version or role is left to its own resources.
    Returns the completed dictionary and the source of each deployment's resources.
    A deployment with neither raises :class:`FpmResourcesPendingError`.
    """
    selector = (metadata or {}).get("configuration_selector") or {}
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
        applies = all(deployment.get(name, 1) == selector[name] for name in _TOPOLOGY if name in selector)
        wanted = {**{k: selector[k] for k in _IDENTITY_KEYS if k in selector}, **(identity or {})}
        applies = applies and all(
            deployment.get(name) in (None, value) for name, value in wanted.items() if name in _IDENTITY_KEYS
        )
        resolved = resolve_resources(
            deployment.get("resources"),
            metadata if applies else None,
            where=where,
            required=True,
        )
        deployment["resources"] = resolved.resources.model_dump(mode="json")
        sources.append(resolved.source)
    return out, sources


def with_estimator_path(config: dict[str, Any], parquet_path: Path) -> None:
    """Point the interpolation estimator at the staged library unless the user chose a path."""
    estimator = dict(config.get("estimator_config") or {})
    interpolation = dict(estimator.get("fpm_interpolation") or {})
    interpolation.setdefault("fpm_parquet_path", str(parquet_path))
    estimator["fpm_interpolation"] = interpolation
    config["estimator_config"] = estimator
