# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``host-cost`` category: measured serving-path host time per request, and the fit over it.

The category holds two purposes:

- ``host-cost-profile``: one (framework, framework version, system, tokenizer) tokenizer measurement on that
  system's host CPU. It stores one Parquet row per timed request (``aisim-host-cost/v1``) plus an envelope
  manifest that carries the host CPU and software provenance, the regression, and the GPU time of the
  allocation. It feeds the queued ``engine.tokenization``.
- ``host-cost-unqueued``: the rest of the serving path's host time before the engine for one (framework,
  framework version, endpoint, system, tokenizer): client TTFT minus engine time at one lane. It stores one
  Parquet row per measured run (``aisim-host-cost-unqueued/v1``); the manifest keeps the mean over the runs of
  the fixed and the total per-token cost. It feeds the unqueued ``engine.extra_cpu_overhead``, whose per-token
  term is the total less the tokenizer slope replay pairs it with.

The manifest is public. It names the tokenizer by its Hugging Face repository and keeps only the host facts
in :data:`PUBLIC_HOST_FACTS`; host names, scheduler job IDs, node load and local paths go to the internal
record that :func:`internal_record` returns, which belongs under ``evidence/`` in ``aisim-data-internal``.

Layout: ``data/host-cost/<framework>/<version>/<system>/<tokenizer>/{requests.parquet,aisim-manifest.json}``
and ``.../<tokenizer>/unqueued-<endpoint>/{runs.parquet,aisim-manifest.json}``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from aisim_core.data import Artifact, Identity, Manifest, _verify, configuration_id
from aisim_core.data.gpu_time import GpuTime
from aisim_core.data.parquet import read_parquet, write_parquet
from aisim_core.host_cost.fit import QUEUED_COMPONENTS, Fit, fit_rows

HOST_COST_SCHEMA = "aisim-host-cost/v1"
UNQUEUED_SCHEMA = "aisim-host-cost-unqueued/v1"
CATEGORY = "host-cost"
PURPOSE = "host-cost-profile"
UNQUEUED_PURPOSE = "host-cost-unqueued"
MANIFEST_NAME = "aisim-manifest.json"
# Host facts a fit depends on and that identify no machine or job.
PUBLIC_HOST_FACTS = ("cpu_model", "cpu_sockets", "cpu_threads", "affinity_cpus", "python", "tokenizers_parallelism")
_HF_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


class HostCostRow(BaseModel):
    """One timed request. Times are milliseconds, the median over the repeats of that request."""

    model_config = ConfigDict(extra="forbid", strict=True)

    request: int = Field(ge=0)
    tokens: int = Field(gt=0)
    body_kb: float = Field(ge=0)
    json_ms: float = Field(ge=0)
    validate_ms: float = Field(ge=0)
    template_ms: float = Field(ge=0)
    encode_ms: float = Field(ge=0)
    hash_ms: float | None = Field(default=None, ge=0)  # only where the framework hashes the prompt before scheduling


_ARROW = pa.schema(
    [
        ("request", pa.int64()),
        ("tokens", pa.int64()),
        ("body_kb", pa.float64()),
        ("json_ms", pa.float64()),
        ("validate_ms", pa.float64()),
        ("template_ms", pa.float64()),
        ("encode_ms", pa.float64()),
        ("hash_ms", pa.float64()),
    ]
)


class UnqueuedRunRow(BaseModel):
    """One measured run of the unqueued overhead: ``outside engine = base_ms + total_per_input_token_us * tokens``.

    ``outside engine`` is client TTFT minus the engine's prefill span, per request, at one lane. ``requests`` and
    ``r2`` are null when the run was summarized without its per-request fit statistics.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    run: str = Field(min_length=1)
    requests: int | None = Field(default=None, gt=0)
    base_ms: float = Field(allow_inf_nan=False)
    total_per_input_token_us: float = Field(ge=0, allow_inf_nan=False)
    r2: float | None = Field(default=None, le=1, allow_inf_nan=False)


_UNQUEUED_ARROW = pa.schema(
    [
        ("run", pa.string()),
        ("requests", pa.int64()),
        ("base_ms", pa.float64()),
        ("total_per_input_token_us", pa.float64()),
        ("r2", pa.float64()),
    ]
)


@dataclass(frozen=True)
class UnqueuedFit:
    """The mean over runs of an unqueued entry, and the tokenizer slope it was split with when written."""

    base_ms: float
    total_per_input_token_us: float
    runs: int
    tokenizer_per_input_token_us: float
    tokenizer_configuration_id: str

    @property
    def per_input_token_us(self) -> float:
        """The unqueued per-token term against the tokenizer slope recorded at write time."""
        return self.total_per_input_token_us - self.tokenizer_per_input_token_us

    def to_json(self) -> dict[str, Any]:
        return {
            "base_ms": self.base_ms,
            "total_per_input_token_us": self.total_per_input_token_us,
            "runs": self.runs,
            "tokenizer_per_input_token_us": self.tokenizer_per_input_token_us,
            "tokenizer_configuration_id": self.tokenizer_configuration_id,
            "per_input_token_us": self.per_input_token_us,
        }


def _unqueued_mean(rows: Sequence[UnqueuedRunRow]) -> tuple[float, float]:
    if not rows:
        raise ValueError("an unqueued entry needs at least one run")
    return (
        math.fsum(r.base_ms for r in rows) / len(rows),
        math.fsum(r.total_per_input_token_us for r in rows) / len(rows),
    )


def entry_directory(framework: str, version: str, system: str, tokenizer: str, *, prefix: str = "data") -> str:
    return "/".join(part for part in (prefix.strip("/"), CATEGORY, framework, version, system, tokenizer) if part)


def _identity(environment: Mapping[str, Any], system: str, model: str) -> Identity:
    host = environment["host"]
    settings = {
        "cpu_model": host["cpu_model"],
        "tokenizer_key": environment["tokenizer_key"],
        "tokenizer_class": environment["tokenizer_class"],
        "tokenizer_is_fast": environment["tokenizer_is_fast"],
        "tokenizer_workers": environment["tokenizer_workers"],
        "hash_algorithm": environment.get("hash_algorithm"),
        "block_size": environment.get("block_size"),
        "endpoint": "chat",
    }
    return Identity(
        hardware=system,
        framework=environment["framework"],
        framework_version=environment["framework_version"],
        operation_family=None,
        model=model,
        effective_settings=settings,
        unresolved_identity=None,
    )


def write_entry(
    root: str | Path,
    run_dir: str | Path,
    *,
    system: str,
    model: str,
    gpu_time: GpuTime | None,
    prefix: str = "data",
    extra_provenance: Mapping[str, Any] | None = None,
) -> str:
    """Write one harness run directory (rows.jsonl + environment.json) as a configuration; return the manifest path.

    ``model`` is the Hugging Face repository the tokenizer files came from; the run's own ``model`` may be a local
    path and is not published. ``gpu_time`` must not carry scheduler job IDs (see :func:`internal_record`).
    """
    if not _HF_REPO.fullmatch(model):
        raise ValueError(f"model must be a Hugging Face repository ID such as 'org/name', not {model!r}")
    if gpu_time is not None and "job_ids" in gpu_time.source:
        raise ValueError("gpu_time.source carries job IDs; keep them in the internal record")
    root, run_dir = Path(root), Path(run_dir)
    environment = json.loads((run_dir / "environment.json").read_text())
    raw_rows = [json.loads(line) for line in (run_dir / "rows.jsonl").read_text().splitlines() if line.strip()]
    rows = [HostCostRow(**{k: v for k, v in row.items() if k != "hash_ms" or v is not None}) for row in raw_rows]
    fit = fit_rows([row.model_dump() for row in rows])
    identity = _identity(environment, system, model)
    directory = entry_directory(
        environment["framework"], environment["framework_version"], system, environment["tokenizer_key"], prefix=prefix
    )
    relative = f"{directory}/requests.parquet"
    table = pa.Table.from_pylist([row.model_dump() for row in rows], schema=_ARROW)
    write_parquet(table, root / relative, HOST_COST_SCHEMA)
    data = (root / relative).read_bytes()
    artifact = Artifact(
        artifact_id="requests",
        path=relative,
        role="host-cost-requests",
        format="parquet",
        schema=HOST_COST_SCHEMA,
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        rows=len(rows),
    )
    provenance: dict[str, Any] = {
        "host_cost": {
            "environment": {
                **{k: v for k, v in environment.items() if k not in {"host", "schema", "model"}},
                "model": model,
            },
            "host": {k: environment["host"].get(k) for k in PUBLIC_HOST_FACTS},
            "fit": {"components": list(QUEUED_COMPONENTS), **fit.to_json()},
            "gpu_time": gpu_time.model_dump(mode="json", by_alias=True) if gpu_time is not None else None,
            **dict(extra_provenance or {}),
        }
    }
    manifest = Manifest(
        schema_version="aisim-artifact-manifest/v1",
        category=CATEGORY,
        purpose=PURPOSE,
        configuration_id=configuration_id(CATEGORY, PURPOSE, identity),
        identity=identity,
        requested_settings=None,
        artifacts=[artifact],
        provenance=json.loads(json.dumps(provenance)),
        evidence_gaps=[],
    )
    path = f"{directory}/{MANIFEST_NAME}"
    (root / path).write_text(
        json.dumps(manifest.model_dump(mode="json", by_alias=True), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return path


def unqueued_directory(
    framework: str, version: str, system: str, tokenizer: str, endpoint: str, *, prefix: str = "data"
) -> str:
    return f"{entry_directory(framework, version, system, tokenizer, prefix=prefix)}/unqueued-{endpoint}"


def write_unqueued_entry(
    root: str | Path,
    *,
    framework: str,
    version: str,
    system: str,
    endpoint: str,
    model: str,
    runs: Sequence[Mapping[str, Any]],
    tokenizer_manifest: Manifest,
    method: Mapping[str, Any],
    prefix: str = "data",
) -> str:
    """Write one unqueued-overhead configuration; return the manifest path.

    ``version`` is the framework version the runs were served with; it may differ from the tokenizer entry's
    (the closest collected one). ``tokenizer_manifest`` is the ``host-cost-profile`` entry whose slope the total is
    split with; its host CPU and tokenizer become this entry's. ``model`` is the Hugging Face repository that was
    served. ``method`` describes how the runs were measured; it must not name hosts or jobs.
    """
    if not _HF_REPO.fullmatch(model):
        raise ValueError(f"model must be a Hugging Face repository ID such as 'org/name', not {model!r}")
    if tokenizer_manifest.purpose != PURPOSE:
        raise ValueError("the tokenizer manifest must be a host-cost-profile entry")
    tokenizer_settings = tokenizer_manifest.identity.effective_settings
    if (tokenizer_manifest.identity.framework, tokenizer_manifest.identity.hardware) != (framework, system):
        raise ValueError("the tokenizer entry measures another framework or system")
    root = Path(root)
    rows = [UnqueuedRunRow(**dict(run)) for run in runs]
    base_ms, total_us = _unqueued_mean(rows)
    tokenizer_us = read_fit(tokenizer_manifest).per_input_token_us
    fit = UnqueuedFit(base_ms, total_us, len(rows), tokenizer_us, tokenizer_manifest.configuration_id)
    if base_ms < 0 or fit.per_input_token_us < 0:
        raise ValueError(f"unqueued overhead must be non-negative, got {fit.to_json()}")
    identity = Identity(
        hardware=system,
        framework=framework,
        framework_version=version,
        operation_family=None,
        model=model,
        effective_settings={
            "cpu_model": tokenizer_settings["cpu_model"],
            "tokenizer_key": tokenizer_settings["tokenizer_key"],
            "endpoint": endpoint,
        },
        unresolved_identity=None,
    )
    directory = unqueued_directory(
        framework, version, system, str(tokenizer_settings["tokenizer_key"]), endpoint, prefix=prefix
    )
    relative = f"{directory}/runs.parquet"
    table = pa.Table.from_pylist([row.model_dump() for row in rows], schema=_UNQUEUED_ARROW)
    write_parquet(table, root / relative, UNQUEUED_SCHEMA)
    data = (root / relative).read_bytes()
    artifact = Artifact(
        artifact_id="runs",
        path=relative,
        role="host-cost-unqueued-runs",
        format="parquet",
        schema=UNQUEUED_SCHEMA,
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        rows=len(rows),
    )
    provenance = {"host_cost": {"method": dict(method), "fit": fit.to_json()}}
    manifest = Manifest(
        schema_version="aisim-artifact-manifest/v1",
        category=CATEGORY,
        purpose=UNQUEUED_PURPOSE,
        configuration_id=configuration_id(CATEGORY, UNQUEUED_PURPOSE, identity),
        identity=identity,
        requested_settings=None,
        artifacts=[artifact],
        provenance=json.loads(json.dumps(provenance, allow_nan=False)),
        evidence_gaps=[],
    )
    path = f"{directory}/{MANIFEST_NAME}"
    (root / path).write_text(
        json.dumps(manifest.model_dump(mode="json", by_alias=True), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return path


def read_unqueued_fit(manifest: Manifest) -> UnqueuedFit:
    """The fit stored in an unqueued manifest."""
    if manifest.purpose != UNQUEUED_PURPOSE:
        raise ValueError(f"not an unqueued host-cost manifest: {manifest.purpose!r}")
    stored = manifest.provenance["host_cost"]["fit"]
    return UnqueuedFit(
        base_ms=float(stored["base_ms"]),
        total_per_input_token_us=float(stored["total_per_input_token_us"]),
        runs=int(stored["runs"]),
        tokenizer_per_input_token_us=float(stored["tokenizer_per_input_token_us"]),
        tokenizer_configuration_id=str(stored["tokenizer_configuration_id"]),
    )


def internal_record(run_dir: str | Path, *, job_ids: list[str]) -> dict[str, Any]:
    """The cluster facts of one entry that the public manifest leaves out: host, jobs, load and tokenizer path."""
    environment = json.loads((Path(run_dir) / "environment.json").read_text())
    return {
        "schema": "aisim-host-cost-internal/v1",
        "tokenizer_path": environment["model"],
        "host": environment["host"],
        "job_ids": list(job_ids),
    }


def read_fit(manifest: Manifest) -> Fit:
    """The fit stored in a manifest."""
    stored = dict(manifest.provenance["host_cost"]["fit"])
    stored.pop("components", None)
    stored["per_input_token_us_ci95"] = tuple(stored["per_input_token_us_ci95"])
    return Fit(**stored)


def check_entry(root: str | Path, path: str) -> Manifest:
    """Verify the envelope, the Parquet rows, and that the stored fit is the fit (or run mean) of those rows."""
    root = Path(root)
    manifest = Manifest.model_validate_json((root / path).read_bytes())
    if manifest.category != CATEGORY or manifest.purpose not in (PURPOSE, UNQUEUED_PURPOSE):
        raise ValueError("not a host-cost manifest")
    if manifest.configuration_id != configuration_id(manifest.category, manifest.purpose, manifest.identity):
        raise ValueError("configuration identity hash mismatch")
    (artifact,) = manifest.artifacts
    if manifest.purpose == UNQUEUED_PURPOSE:
        table = read_parquet(_verify(root, artifact), UNQUEUED_SCHEMA)
        runs = [UnqueuedRunRow(**row) for row in table.to_pylist()]
        base_ms, total_us = _unqueued_mean(runs)
        fit = read_unqueued_fit(manifest)
        if (
            fit.runs != len(runs)
            or not math.isclose(fit.base_ms, base_ms, rel_tol=1e-9, abs_tol=1e-9)
            or not math.isclose(fit.total_per_input_token_us, total_us, rel_tol=1e-9, abs_tol=1e-9)
        ):
            raise ValueError("stored unqueued fit differs from the mean of the runs")
        if fit.base_ms < 0 or fit.per_input_token_us < 0:
            raise ValueError("stored unqueued fit is negative")
        return manifest
    table = read_parquet(_verify(root, artifact), HOST_COST_SCHEMA)
    rows = [HostCostRow(**row) for row in table.to_pylist()]
    recomputed = fit_rows([row.model_dump() for row in rows])
    stored = read_fit(manifest)
    for name in ("base_ms", "per_input_token_us"):
        want, have = getattr(recomputed, name), getattr(stored, name)
        if abs(want - have) > 1e-9 * max(1.0, abs(have)):
            raise ValueError(f"stored fit {name} differs from the fit of the rows")
    return manifest


def register_entries(root: str | Path, manifest_paths: list[str]) -> str:
    """Register entries in the ``host-cost`` category index and ``catalog/dataset.json``; return the index path.

    ``catalog/dataset.json`` must exist (a fresh dataset has one); other categories are left untouched.
    """
    from aisim_core.data import CategoryIndex, CategoryReference, DatasetIndex, Entry

    root = Path(root)

    def ref(relative: str, schema: str, kind: str) -> Artifact:
        data = (root / relative).read_bytes()
        return Artifact(
            artifact_id=kind,
            path=relative,
            role=kind,
            format="json",
            schema=schema,
            sha256=hashlib.sha256(data).hexdigest(),
            bytes=len(data),
        )

    def dump(model: BaseModel) -> bytes:
        text = json.dumps(model.model_dump(mode="json", by_alias=True), indent=2, sort_keys=True, allow_nan=False)
        return (text + "\n").encode()

    index_path = f"catalog/{CATEGORY}/index.json"
    existing: dict[str, Entry] = {}
    if (root / index_path).is_file():
        existing = {
            e.configuration_id: e for e in CategoryIndex.model_validate_json((root / index_path).read_bytes()).entries
        }
    for path in manifest_paths:
        manifest = Manifest.model_validate_json((root / path).read_bytes())
        existing[manifest.configuration_id] = Entry(
            configuration_id=manifest.configuration_id,
            purpose=manifest.purpose,
            identity=manifest.identity,
            manifest=ref(path, "aisim-artifact-manifest/v1", "manifest"),
        )
    index = CategoryIndex(
        schema_version="aisim-category-index/v1",
        category=CATEGORY,
        entries=[existing[key] for key in sorted(existing)],
    )
    (root / index_path).parent.mkdir(parents=True, exist_ok=True)
    (root / index_path).write_bytes(dump(index))
    dataset = DatasetIndex.model_validate_json((root / "catalog/dataset.json").read_bytes())
    categories = dict(dataset.categories)
    categories[CATEGORY] = CategoryReference(
        root=f"data/{CATEGORY}", layout="category", index=ref(index_path, "aisim-category-index/v1", "index")
    )
    (root / "catalog/dataset.json").write_bytes(dump(dataset.model_copy(update={"categories": categories})))
    return index_path
