# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-category loaders that turn verified local dataset files into AISim inputs.

A loader declares ``category``, accepted manifest schema versions and purposes,
``select(config) -> Selection`` (the catalog lookup for a resolved
configuration, ready for :func:`aisim_core.data.fetch.fetch`) and
``load(root, manifest_path, ...)`` (verified local files -> the structure AISim
already consumes). ``root`` is a dataset root such as ``FetchResult.root``.

Loaders never touch the network, never change performance-model behavior and are
not wired into any caller yet. They reject hash mismatches, unsupported schemas,
an inconsistent category/purpose/identity and invalid paths.

* :class:`OpLatencyLoader` materializes a ``systems`` root (``data/<system>/...``
  plus hardware specs) that can be appended to ``systems_paths``.
* :class:`FpmLoader` stages the published prediction tier (library plus sidecar, never ``measurements/``) into a
  ``systems`` root and resolves resources (:mod:`aisim_core.data.fpm_library`). Entries that carry the older
  ``execution-profile`` and ``predictor-config`` roles still load through :func:`prepare_execution_profile`.
* :class:`E2eLoader` returns pyarrow table views.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa

from aisim_core.data import (
    Artifact,
    Category,
    Identity,
    Manifest,
    _file,
    _verified_bytes,
    _verify,
    configuration_id,
    prepare_execution_profile,
)
from aisim_core.data.fetch import Selection
from aisim_core.data.fpm_library import (
    EVALUATION_ROLES,
    PREDICTION_ROLES,
    ROLE_LEGACY_MANIFEST,
    ROLE_LIBRARY,
    ROLE_METADATA,
    FpmContractError,
    ResolvedResources,
    complete_fpm_profile,
    missing_memory_fields,
    missing_resource_fields,
    read_library_metadata,
    resolve_resources,
    resource_metadata_for_deployment,
    stage_fpm_library,
    with_estimator_path,
)
from aisim_core.data.parquet import read_parquet

MANIFEST_SCHEMA = "aisim-artifact-manifest/v1"


class LoaderError(ValueError):
    """A manifest or artifact is not acceptable to this loader."""


class Loader(ABC):
    """Contract every category loader implements."""

    category: ClassVar[Category]
    manifest_schemas: ClassVar[tuple[str, ...]] = (MANIFEST_SCHEMA,)
    # Manifest categories this loader accepts; defaults to ``(category,)``.
    manifest_categories: ClassVar[tuple[str, ...] | None] = None
    purposes: ClassVar[tuple[str, ...]]

    @abstractmethod
    def select(self, config: Mapping[str, Any]) -> Selection:
        """Catalog selection for a resolved configuration (no I/O)."""

    @abstractmethod
    def load(self, root: Path, manifest_path: str, **kwargs: Any) -> Any:
        """Verify the manifest and its files under ``root`` and build the result."""

    # -- shared helpers ------------------------------------------------------

    def _selection(self, config: Mapping[str, Any], **extra: Any) -> Selection:
        selection = Selection(
            category=self.category,
            hardware=config.get("system"),
            framework=config.get("backend"),
            framework_version=config.get("backend_version"),
            purpose=self.purposes[0] if len(self.purposes) == 1 else None,
            **extra,
        )
        selection.validate()
        return selection

    def read_manifest(
        self,
        root: Path,
        manifest_path: str,
        config: Mapping[str, Any] | None = None,
        *,
        roles: tuple[str, ...] | None = None,
    ) -> Manifest:
        """Parse and verify a manifest of this category (hashes, identity, paths).

        ``roles`` limits the hash check to artifacts of those roles; files of other roles may be absent.
        """
        root = Path(root)
        manifest = Manifest.model_validate_json(_file(root, manifest_path).read_bytes())
        if manifest.schema_version not in self.manifest_schemas:
            raise LoaderError(f"unsupported manifest schema: {manifest.schema_version}")
        categories = self.manifest_categories or (self.category,)
        if manifest.category not in categories:
            raise LoaderError(f"{type(self).__name__} cannot load a {manifest.category!r} manifest")
        if manifest.purpose not in self.purposes:
            raise LoaderError(f"{type(self).__name__} cannot load purpose {manifest.purpose!r}")
        if manifest.configuration_id != configuration_id(manifest.category, manifest.purpose, manifest.identity):
            raise LoaderError("configuration identity hash mismatch")
        for name in ("artifact_id", "path", "runtime_path"):
            values = [getattr(a, name) for a in manifest.artifacts if getattr(a, name) is not None]
            if len(values) != len(set(values)):
                raise LoaderError(f"duplicate artifact {name}")
        for artifact in manifest.artifacts:
            if roles is None or artifact.role in roles:
                _verify(root, artifact)
        if config is not None:
            _check_identity(manifest.identity, config)
        return manifest


def _check_identity(identity: Identity, config: Mapping[str, Any]) -> None:
    declared = (identity.hardware, identity.framework, identity.framework_version)
    wanted = (config.get("system"), config.get("backend"), config.get("backend_version"))
    if declared != wanted:
        raise LoaderError(f"manifest identity {declared} differs from configuration {wanted}")
    model = config.get("model")
    if identity.model is not None and model is not None and identity.model != model:
        raise LoaderError(f"manifest model {identity.model!r} differs from configuration {model!r}")


# --------------------------------------------------------------------------- op-latency


@dataclass(frozen=True)
class OpLatencyData:
    """A materialized systems root, ready to append to ``systems_paths``."""

    systems_root: Path
    manifest: Manifest
    files: tuple[str, ...]  # runtime paths relative to ``systems_root``


class OpLatencyLoader(Loader):
    """Materialize verified operation tables as a ``systems/`` root.

    Only artifacts with a ``runtime_path`` are copied; the layout matches
    ``systems/data/<system>/<family>/<backend>/<version>/`` plus system specs,
    exactly what :func:`prepare_execution_profile` writes under ``systems``.
    """

    category: ClassVar[Category] = "op-latency"
    purposes: ClassVar[tuple[str, ...]] = ("operation-tables",)

    def select(self, config: Mapping[str, Any]) -> Selection:
        # Operation tables are model-independent: the model never selects them.
        return self._selection(config, operation_family=config.get("operation_family"))

    def load(
        self,
        root: Path,
        manifest_path: str,
        *,
        output_dir: Path,
        config: Mapping[str, Any] | None = None,
    ) -> OpLatencyData:
        """Copy verified runtime inputs to ``output_dir/systems`` (``output_dir`` must not exist)."""
        root, output_dir = Path(root), Path(output_dir).absolute()
        manifest = self.read_manifest(root, manifest_path, config)
        if manifest.identity.model is not None:
            raise LoaderError("operation table storage must remain model-independent")
        inputs = [a for a in manifest.artifacts if a.runtime_path is not None]
        if not inputs:
            raise LoaderError("manifest declares no runtime inputs")
        systems = output_dir / "systems"
        output_dir.mkdir(parents=True, exist_ok=False)
        try:
            for artifact in inputs:
                destination = systems / artifact.runtime_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(_verify(root, artifact), destination)
                _verify(systems, artifact.model_copy(update={"path": artifact.runtime_path}))
        except BaseException:
            shutil.rmtree(output_dir)
            raise
        return OpLatencyData(systems, manifest, tuple(sorted(a.runtime_path for a in inputs)))


# --------------------------------------------------------------------------- fpm


@dataclass(frozen=True)
class FpmData:
    """What an FPM entry provides for ``best_available``.

    ``config`` is a complete ``best_available`` configuration: the user's configuration with the staged
    ``systems_paths`` and interpolation library path, and ``fpm_profile`` completed with resolved resources when
    the user supplied one. ``kind`` is ``"library"`` for the published library plus sidecar and
    ``"execution-profile"`` for the older profile bundles (``config`` from :func:`prepare_execution_profile`).

    ``resources`` and ``memory_pending`` describe deployment zero, not the deployment selected by a worker's
    configuration. Use ``resources_by_deployment`` and ``memory_pending_by_deployment`` in profile order to
    inspect other deployments; this loader does not select a worker or role.
    """

    manifest: Manifest
    config: dict | None = None
    profile: Any | None = None  # FpmModelProfile from an ``fpm-profile`` artifact or a completed user profile
    kind: str = "execution-profile"
    systems_root: Path | None = None
    resources: ResolvedResources | None = None  # first deployment, library kind
    resource_sources: tuple[str, ...] = ()  # per deployment of ``fpm_profile``
    diagnostics: Mapping[str, Any] | None = None
    resources_by_deployment: tuple[ResolvedResources, ...] = ()
    # Owner of the staged root (a temporary directory when no ``output_dir`` was given); keep this object alive
    # while ``config`` is in use and call :meth:`close` to remove a temporary root.
    _staged: Any = field(default=None, repr=False, compare=False)

    def close(self) -> None:
        if self._staged is not None:
            self._staged.close()

    @property
    def memory_pending(self) -> tuple[str, ...]:
        """Missing cache sizing or replay fields for deployment zero; plain AR timing can omit them."""
        if self.resources is None:
            return ()
        if self.resources.resources is None:
            return tuple(missing_resource_fields(None))
        return tuple(missing_memory_fields(self.resources.resources))

    @property
    def memory_pending_by_deployment(self) -> tuple[tuple[str, ...], ...]:
        """Missing fields in profile order (one entry for the requested configuration without a profile)."""
        resolved = self.resources_by_deployment or ((self.resources,) if self.resources is not None else ())
        return tuple(
            tuple(missing_resource_fields(None) if r.resources is None else missing_memory_fields(r.resources))
            for r in resolved
        )


class FpmLoader(Loader):
    """FPM data from the published library, or from an older execution profile.

    * **Library** (roles ``fpm-library`` + ``fpm-library-metadata``): only those two files are verified and
      staged; ``measurements/`` and ``legacy-configuration-manifest`` files may be absent. The user's
      configuration supplies model, system, backend, estimator choice and the optional ``fpm_profile``; resources
      resolve as user configuration, then sidecar ``resources``, then pending.
    * **Execution profile** manifests go through :func:`prepare_execution_profile`; an ``fpm-profile`` JSON
      artifact is read with :func:`load_fpm_profile`.
    """

    category: ClassVar[Category] = "fpm"
    manifest_categories: ClassVar[tuple[str, ...]] = ("fpm", "op-latency")
    purposes: ClassVar[tuple[str, ...]] = ("execution-profile",)
    # Library prediction needs neither evaluation-tier files nor the legacy configuration manifest.
    # Legacy execution-profile bundles still require every declared artifact for verification.
    excluded_roles: ClassVar[tuple[str, ...]] = (*EVALUATION_ROLES, ROLE_LEGACY_MANIFEST)

    def select(self, config: Mapping[str, Any]) -> Selection:
        return self._selection(
            config,
            model=config.get("model"),
            parallelism=_parallelism(config),
            library_exclude_roles=self.excluded_roles,
        )

    def load(
        self,
        root: Path,
        manifest_path: str,
        *,
        output_dir: Path | None = None,
        config: Mapping[str, Any] | None = None,
        require_resources: bool = False,
        systems_paths: tuple[str, ...] | None = None,
    ) -> FpmData:
        """Verify and stage an entry; ``output_dir`` receives the staged ``systems`` root (library kind).

        ``require_resources`` raises :class:`FpmResourcesPendingError` when neither the user's ``fpm_profile`` nor the
        sidecar supplies resources, naming the fields to supply. Without it, a library without resources still loads
        for timing queries.
        """
        root = Path(root)
        peek = Manifest.model_validate_json(_file(root, manifest_path).read_bytes())
        if any(a.role == ROLE_LIBRARY for a in peek.artifacts):
            return self._load_library(root, manifest_path, output_dir, config, require_resources, systems_paths)
        manifest = self.read_manifest(root, manifest_path, config)
        profiles = [a for a in manifest.artifacts if a.role == "fpm-profile"]
        if profiles:
            if len(profiles) != 1:
                raise LoaderError("exactly one fpm-profile artifact is required")
            return FpmData(manifest, profile=_load_fpm_profile_artifact(root, profiles[0]))
        if output_dir is None:
            raise LoaderError("output_dir is required to prepare an execution profile")
        return FpmData(manifest, config=prepare_execution_profile(root, manifest_path, Path(output_dir)))

    def _load_library(
        self,
        root: Path,
        manifest_path: str,
        output_dir: Path | None,
        config: Mapping[str, Any] | None,
        require_resources: bool,
        systems_paths: tuple[str, ...] | None,
    ) -> FpmData:
        user_profile = (config or {}).get("fpm_profile")
        if user_profile is not None and not isinstance(user_profile, Mapping):
            raise LoaderError("fpm_profile must be a mapping for an FPM library load")
        manifest = self.read_manifest(root, manifest_path, config, roles=PREDICTION_ROLES)
        if config is not None:
            _check_parallelism(manifest.identity, config)
        libraries = [a for a in manifest.artifacts if a.role == ROLE_LIBRARY]
        sidecars = [a for a in manifest.artifacts if a.role == ROLE_METADATA]
        if len(libraries) != 1 or len(sidecars) != 1:
            raise LoaderError(f"exactly one {ROLE_LIBRARY} and one {ROLE_METADATA} artifact are required")
        library, sidecar = libraries[0], sidecars[0]
        identity = manifest.identity
        user_paths = tuple(systems_paths or (config or {}).get("systems_paths") or ()) or None
        library_file, sidecar_file = _verify(root, library), _verify(root, sidecar)
        estimator = (config or {}).get("estimator_config")
        interpolation = estimator.get("fpm_interpolation") if isinstance(estimator, Mapping) else None
        explicit = interpolation.get("fpm_parquet_path") if isinstance(interpolation, Mapping) else None
        # The sidecar describes only the library it ships with. A user path to another library keeps its
        # resources and selector out of reach of this sidecar, so they stay pending unless the user supplies them.
        foreign = explicit is not None and Path(explicit).expanduser().resolve() != library_file.resolve()
        base = dict(config or {})
        with_estimator_path(base, library_file)
        if output_dir is not None:
            # Staging replaces a previous root at output_dir, so reject an unusable resource contract first.
            try:
                declared = read_library_metadata(sidecar_file)
            except FpmContractError as exc:
                raise LoaderError(str(exc)) from exc
            _resolve_library_resources(
                dict(base),
                user_profile,
                identity,
                declared,
                require_resources,
                config is not None,
                foreign,
            )
        try:
            staged = stage_fpm_library(
                library_file,
                sidecar_file,
                system=identity.hardware,
                backend=identity.framework,
                backend_version=identity.framework_version,
                output_dir=output_dir,
                systems_paths=user_paths,
                parquet_sha256=library.sha256,
                metadata_sha256=sidecar.sha256,
                artifact_id=library.artifact_id,
            )
        except FpmContractError as exc:
            raise LoaderError(str(exc)) from exc
        out = dict(config or {})
        out["systems_paths"] = list(
            dict.fromkeys(str(p) for p in (staged.systems_root, *(out.get("systems_paths") or ())))
        )
        try:
            with_estimator_path(out, staged.parquet_path)
            resources, sources, profile, resources_by_deployment = _resolve_library_resources(
                out, user_profile, identity, staged.metadata, require_resources, config is not None, foreign
            )
        except FpmContractError:
            staged.close()
            raise
        return FpmData(
            manifest,
            config=out,
            profile=profile,
            kind="library",
            systems_root=staged.systems_root,
            resources=resources,
            resource_sources=tuple(sources),
            diagnostics=staged.diagnostics,
            resources_by_deployment=resources_by_deployment,
            _staged=staged,
        )


def _resolve_library_resources(
    out: dict[str, Any],
    user_profile: Mapping[str, Any] | None,
    identity: Identity,
    metadata: Mapping[str, Any],
    require_resources: bool,
    has_config: bool,
    foreign: bool,
):
    """Resolve resources for a library load; sets ``out["fpm_profile"]`` when the user supplied one."""
    if foreign:
        metadata = {}
    sources: tuple[str, ...] | list[str] = ()
    profile = None
    if user_profile is not None:
        completed, sources = complete_fpm_profile(
            dict(user_profile),
            metadata,
            identity={
                "system": identity.hardware,
                "backend": identity.framework,
                "backend_version": identity.framework_version,
            },
            required=require_resources,
        )
        out["fpm_profile"] = completed
        from aisim_core.fpm_profile import load_fpm_profile

        # A draft stays a dictionary until every deployment has resources. Do not invent a typed
        # resource block merely to validate a timing-only load.
        if "pending" not in sources:
            profile = load_fpm_profile(completed)
            blocks = [deployment.resources for deployment in profile.deployments]
        else:
            blocks = [resolve_resources(d.get("resources"), None).resources for d in completed["deployments"]]
        by_deployment = tuple(ResolvedResources(block, source) for block, source in zip(blocks, sources, strict=True))
        return by_deployment[0], tuple(sources), profile, by_deployment
    if metadata.get("resources") is not None and metadata.get("configuration_selector"):
        if not has_config:
            metadata = None  # No requested deployment to match against the selector.
        else:
            from aisim_core.sdk import RustForwardPassPerfModel

            # Canonical normalization owns defaults and aliases; do not assume omitted topology.
            normalized = RustForwardPassPerfModel.normalize_config(out)
            deployment = {
                **normalized,
                "dp": normalized["attention_dp"],
                "moe_tp": normalized.get("moe_tp_size"),
                "moe_ep": normalized.get("moe_ep_size"),
                "cp": normalized.get("cp_size"),
                "kv_cache_dtype": normalized.get("kvcache_quant_mode"),
            }
            deployment = {name: value for name, value in deployment.items() if value is not None}
            metadata = resource_metadata_for_deployment(normalized, deployment, metadata)
    resources = resolve_resources(None, metadata, required=require_resources)
    return resources, (), None, (resources,)


def _parallelism(config: Mapping[str, Any]) -> tuple[tuple[str, int], ...] | None:
    """Catalog parallelism dimensions of a configuration (degree 1 dropped), or None when it names none."""
    keys = {
        "tp": "tp",
        "attention_dp": "dp",
        "moe_tp_size": "moe_tp",
        "moe_ep_size": "ep",
        "pp": "pp",
        "cp_size": "cp",
        "dcp": "dcp",
    }
    # The same aliases the Rust configuration accepts.
    aliases = {"tp": "tp_size", "pp": "pp_size", "attention_dp": "attention_dp_size"}

    def value(key: str) -> Any:
        found = config.get(key)
        return found if found is not None else config.get(aliases[key]) if key in aliases else None

    if not any(value(k) is not None for k in keys):
        return None
    dims = {dim: value(key) for key, dim in keys.items() if value(key) not in (None, 1)}
    return tuple(sorted(dims.items()))


def _check_parallelism(identity: Identity, config: Mapping[str, Any]) -> None:
    wanted = _parallelism(config)
    if wanted is None or identity.parallelism is None:
        return
    declared = [tuple(sorted(dims.items())) for dims in identity.parallelism.values()]
    if wanted not in declared:
        raise LoaderError(f"manifest parallelism {identity.parallelism} differs from configuration {dict(wanted)}")


def _load_fpm_profile_artifact(root: Path, artifact: Artifact):
    from aisim_core.fpm_profile import load_fpm_profile

    if artifact.format != "json":
        raise LoaderError("fpm-profile artifact must be JSON")
    return load_fpm_profile(_verify(root, artifact).read_text())


# --------------------------------------------------------------------------- e2e


@dataclass(frozen=True)
class E2eTables:
    """Verified Parquet artifacts as pyarrow tables keyed by ``artifact_id``."""

    manifest: Manifest
    tables: dict[str, pa.Table] = field(default_factory=dict)

    def __getitem__(self, artifact_id: str) -> pa.Table:
        return self.tables[artifact_id]


class E2eLoader(Loader):
    """E2E tables as pyarrow views; nothing is converted or materialized."""

    category: ClassVar[Category] = "e2e"
    purposes: ClassVar[tuple[str, ...]] = ("e2e-configuration",)

    def __init__(self, accepted_schemas: tuple[str, ...] | None = None):
        # None accepts exactly the schema each artifact declares in its manifest.
        self.accepted_schemas = accepted_schemas

    def select(self, config: Mapping[str, Any]) -> Selection:
        return self._selection(config, model=config.get("model"))

    def load(
        self,
        root: Path,
        manifest_path: str,
        *,
        config: Mapping[str, Any] | None = None,
        columns: Mapping[str, list[str]] | None = None,
    ) -> E2eTables:
        """Read every Parquet artifact; the stamped schema ID must match what is accepted."""
        root = Path(root)
        manifest = self.read_manifest(root, manifest_path, config)
        tables = {}
        for artifact in manifest.artifacts:
            if artifact.format != "parquet":
                continue
            if self.accepted_schemas is not None and artifact.artifact_schema not in self.accepted_schemas:
                raise LoaderError(f"unsupported table schema: {artifact.artifact_schema}")
            # Parse the verified bytes, so a file replaced after the check cannot reach the caller unverified.
            tables[artifact.artifact_id] = read_parquet(
                pa.BufferReader(_verified_bytes(root, artifact)),
                artifact.artifact_schema,
                columns=(columns or {}).get(artifact.artifact_id),
            )
        if not tables:
            raise LoaderError("manifest declares no Parquet tables")
        return E2eTables(manifest, tables)


LOADERS: dict[str, type[Loader]] = {cls.category: cls for cls in (OpLatencyLoader, FpmLoader, E2eLoader)}
