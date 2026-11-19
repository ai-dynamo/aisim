# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Redaction rules and the publication content check (DEP #435, "Accidental disclosure").

What may move from ``aisim-data-internal`` to the public dataset is decided by a
:class:`RedactionPolicy`. The defaults are the most conservative reading of the
disclosure decision (AISIM-2312):

* Visibility is a label per path prefix in ``mirror/visibility.json``
  (:class:`VisibilityLedger`); an unlabeled path is ``internal``.
* ``producer.run_url`` (``producer_id``), image names and digests, patch
  references, logs and raw tool output are withheld. They must be absent or the
  literal ``[REDACTED]``; their original hash is kept as lineage in the private
  promotion ledger, never as a public field.
* Data derived from InferenceX is published as aggregates only. Per-point truth,
  per-request rows, raw output and logs are never public, so an InferenceX-derived
  configuration or evaluation record is not promotable; a page may publish rows that
  pass :func:`check_disclosure`.
* Promotion is a path-identical copy, so redaction happens before a directory is
  labeled ``public``. This module checks the result; it never rewrites bytes.

Every check is pure and offline. ``check_publishable`` is also what the ``public``
profile of :mod:`aisim_core.data.mirror` runs, so the public repository CI
re-checks anything that was promoted.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aisim_core.data import Manifest, _verify
from aisim_core.data.catalog import catalog_files
from aisim_core.data.e2e import (
    OBSERVATIONS_SCHEMA,
    REQUESTS_SCHEMA,
    E2eManifestBody,
    body_from_manifest,
    read_snapshot,
)
from aisim_core.data.eval import load_record, record_paths
from aisim_core.data.hygiene import Finding, checked_path, yaml_sidecar_value
from aisim_core.data.parquet import read_parquet

Label = Literal["public", "internal", "restricted"]
LEDGER_PATH = "mirror/visibility.json"
LEDGER_SCHEMA = "aisim-visibility/v1"
REDACTED = "[REDACTED]"
INFX_MARKERS = ("inferencex", "infx")

_RUN_URL = re.compile(r"https?://[^\s\"']+/(?:actions/runs|-/pipelines|-/jobs|pipelines|builds)/\d+", re.I)
_DIGEST = re.compile(r"@sha256:[0-9a-f]{16,}", re.I)
_REGISTRY = re.compile(r"(?:nvcr\.io|gitlab-master|ghcr\.io|docker\.io|gcr\.io|\.dkr\.ecr\.|urm\.|artifactory)", re.I)
_INTERNAL_HOST = re.compile(r"\b[\w-]+(?:\.[\w-]+)*\.(?:nvidia\.com|nvidia\.net)\b|gitlab-master", re.I)
_PATCH_VALUE = re.compile(r"(?m)^(?:diff --git |--- a/|\+\+\+ b/)|\.(?:patch|diff)\b", re.I)
_SECRET = re.compile(
    r"(hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_\-]{20,}|bearer\s+[A-Za-z0-9._\-]{16,})", re.I
)
_IMAGE_KEY = re.compile(r"(^|_)(image|images|container|docker_image)(_|$)", re.I)
_PATCH_KEY = re.compile(r"(^|_)patch(es)?($|_)", re.I)
_RUN_KEY = re.compile(r"(^|_)run_url$|^producer_id$", re.I)


@dataclass(frozen=True)
class RedactionPolicy:
    """Conservative defaults; widening any field is a recorded maintainer decision."""

    allow_provenance_artifacts: bool = False  # logs and raw tool output (tier 3)
    allow_request_rows: bool = False  # per-request tables, for every configuration
    # Per-request tables of self-collected configurations (body ``origin == "self-collected"``).
    # InferenceX-derived and official (upstream-licensed) configurations never use this.
    allow_self_collected_request_rows: bool = True
    allowed_run_url_prefixes: tuple[str, ...] = ()
    allowed_image_prefixes: tuple[str, ...] = ()
    # "aggregates-only": InferenceX-derived configurations and records are not promotable.
    # "point-level": they may be promoted once every other rule passes.
    inferencex_disclosure: Literal["aggregates-only", "point-level"] = "aggregates-only"
    min_scored: int = 5  # smallest cell an aggregate row may describe


DEFAULT_POLICY = RedactionPolicy()

# Fields a published aggregate row may carry. Anything else (case ids, per-point
# truth, request rows, URLs, images) is refused rather than filtered.
AGGREGATE_FIELDS = frozenset(
    {
        "hardware",
        "model",
        "framework",
        "framework_version",
        "layer",
        "n_total",
        "n_scored",
        "n_configs",
        "mape_pct",
        "residual_quantiles_pct",
        "coverage",
        "status",
        "created_at",
        "snapshot_id",
        "dataset_sha",
    }
)


# --------------------------------------------------------------------------- visibility


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


_LABEL_RANK = ("public", "restricted", "internal")  # least to most restrictive; unknown labels fail validation


class VisibilityEntry(_Model):
    path: str = Field(min_length=1)  # a file or a directory prefix, repository-relative
    label: Label
    origin: str | None = None  # e.g. "inferencex"; marks derived data for the disclosure rule
    # path -> SHA-256 of the unredacted original, when the labeled bytes are a redaction of it.
    original_sha256: dict[str, str] = Field(default_factory=dict)


class VisibilityLedger(_Model):
    schema_version: Literal["aisim-visibility/v1"] = LEDGER_SCHEMA
    entries: list[VisibilityEntry] = Field(default_factory=list)

    def entry_for(self, path: str) -> VisibilityEntry | None:
        """Longest matching prefix wins, so a file can be narrower than its directory."""
        hits = [e for e in self.entries if path == e.path.rstrip("/") or path.startswith(e.path.rstrip("/") + "/")]
        # Longest normalized prefix wins; equal prefixes resolve to the most restrictive label.
        return max(hits, key=lambda e: (len(e.path.rstrip("/")), _LABEL_RANK.index(e.label)), default=None)

    def label_for(self, path: str) -> Label:
        entry = self.entry_for(path)
        return entry.label if entry else "internal"  # unlabeled fails closed


def load_ledger(root: str | Path) -> VisibilityLedger:
    path = Path(root) / LEDGER_PATH
    if not path.is_file():
        return VisibilityLedger()
    return VisibilityLedger.model_validate_json(path.read_bytes())


def write_ledger(root: str | Path, ledger: VisibilityLedger) -> Path:
    path = Path(root) / LEDGER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(ledger.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- text scanning


def _walk(value: Any, key: str = "", seen: set[tuple[int, str]] | None = None) -> Iterable[tuple[str, str]]:
    """Yield (key, string) for every string in a JSON-like value, with the nearest key name."""
    if seen is None:
        seen = set()
    if isinstance(value, (Mapping, list, tuple)):
        marker = (id(value), key)
        if marker in seen:
            return  # YAML aliases are scanned once per field context, without exponential expansion
        seen.add(marker)
    if isinstance(value, Mapping):
        for k, v in value.items():
            yield "", str(k)  # content rules apply to keys; field-name rules apply only to their values
            context = key if _IMAGE_KEY.search(key) or _PATCH_KEY.search(key) or _RUN_KEY.search(key) else str(k)
            yield from _walk(v, context, seen)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk(v, key, seen)
    elif isinstance(value, str):
        yield key, value
    elif isinstance(value, bytes):
        # Binary extension columns and Arrow metadata must not bypass disclosure
        # checks. Invalid UTF-8 is refused rather than decoded lossily.
        yield key, value.decode("utf-8")


def _json_text(text: str) -> Any:
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return text


def scan_value(value: Any, policy: RedactionPolicy, path: str, where: str, *, _key: str = "") -> list[Finding]:
    """Findings for run URLs, images, digests, patch references, secrets and internal hosts."""
    out: list[Finding] = []

    def flag(code: str, detail: str) -> None:
        out.append(Finding(code, f"{where}: {detail}", path))

    for key, text in _walk(value, _key):
        if text == REDACTED or not text:
            continue
        nested = _json_text(text) if text.lstrip(" \t\r\n")[:1] in ("{", "[") else None
        if isinstance(nested, (dict, list)):
            out.extend(scan_value(nested, policy, path, where, _key=key))
            continue
        if (_RUN_KEY.search(key) or _RUN_URL.search(text)) and not any(
            text.startswith(p) for p in policy.allowed_run_url_prefixes
        ):
            flag("run-url", "producer run URL must be removed or [REDACTED]")
        if _DIGEST.search(text):
            flag("image-digest", "image digest must be removed")
        elif (_IMAGE_KEY.search(key) or _REGISTRY.search(text)) and not any(
            text.startswith(p) for p in policy.allowed_image_prefixes
        ):
            flag("image-name", "image name must be removed or [REDACTED]")
        if _PATCH_KEY.search(key) or _PATCH_VALUE.search(text):
            flag("patch-reference", "patch reference or diff text must be removed")
        if _SECRET.search(text):
            flag("secret", "credential-like text")
        if _INTERNAL_HOST.search(text):
            flag("internal-host", "internal host name")
    return out


# --------------------------------------------------------------------------- content check


def _is_infx(source_kind: str | None) -> bool:
    return bool(source_kind) and any(m in source_kind.lower() for m in INFX_MARKERS)


def _request_rows_allowed(body: E2eManifestBody | None, policy: RedactionPolicy) -> bool:
    if policy.allow_request_rows:
        return True
    return (
        policy.allow_self_collected_request_rows
        and body is not None
        and body.origin == "self-collected"
        and not _is_infx(body.source_kind)
    )


def _check_manifest(root: Path, rel: str, manifest: Manifest, policy: RedactionPolicy) -> list[Finding]:
    out: list[Finding] = []
    body: E2eManifestBody | None = None
    if manifest.category == "e2e":
        body = body_from_manifest(manifest)
        if _is_infx(body.source_kind) and policy.inferencex_disclosure == "aggregates-only":
            out.append(
                Finding(
                    "infx-aggregates-only",
                    "InferenceX-derived configurations are not published; publish aggregates (check_disclosure)",
                    rel,
                )
            )
        for info in body.provenance_artifacts:
            if not policy.allow_provenance_artifacts:
                out.append(
                    Finding("provenance-withheld", f"{info.kind} artifacts are withheld ({info.artifact_id})", rel)
                )
            elif not info.redacted:
                out.append(Finding("unredacted-provenance", f"{info.artifact_id} is not redacted", rel))
    for artifact in manifest.artifacts:
        suffixes = {"parquet": ".parquet", "json": ".json", "yaml": ".yaml", "gzip": ".gz"}
        if artifact.format not in suffixes or Path(artifact.path).suffix.lower() != suffixes[artifact.format]:
            out.append(
                Finding(
                    "artifact-format", f"unsupported or mismatched artifact format: {artifact.format}", artifact.path
                )
            )
            continue
        if artifact.format == "gzip" and not policy.allow_provenance_artifacts:
            out.append(Finding("provenance-withheld", "logs and raw output are withheld", artifact.path))
        if artifact.artifact_schema == REQUESTS_SCHEMA and not _request_rows_allowed(body, policy):
            out.append(Finding("request-rows", "per-request rows are not published", artifact.path))
        if artifact.artifact_schema == OBSERVATIONS_SCHEMA:
            out.extend(_check_observations(root / artifact.path, artifact.path, policy))
        elif artifact.artifact_schema == REQUESTS_SCHEMA and _request_rows_allowed(body, policy):
            out.extend(_check_requests(root / artifact.path, artifact.path, policy))
        elif artifact.format == "parquet":
            # Metrics, series, op tables, snapshot members/acquisition and eval
            # truth all have free-text columns. Scan every column, including
            # extension schemas, rather than exempting unmatched schema IDs.
            out.extend(
                _check_table_text(
                    read_parquet(root / artifact.path, artifact.artifact_schema), artifact.path, policy, "column"
                )
            )
        elif artifact.format == "json":
            out.extend(
                scan_value(
                    json.loads(checked_path(root, artifact.path).read_bytes()), policy, artifact.path, "JSON sidecar"
                )
            )
        elif artifact.format == "yaml":
            data = checked_path(root, artifact.path).read_bytes()
            out.extend(scan_value(yaml_sidecar_value(data), policy, artifact.path, "YAML sidecar"))
            out.extend(scan_value(data.decode("utf-8"), policy, artifact.path, "YAML source"))
    out.extend(scan_value(manifest.model_dump(mode="json", by_alias=True), policy, rel, "manifest"))
    return out


def _check_observations(path: Path, rel: str, policy: RedactionPolicy) -> list[Finding]:
    table: pa.Table = read_parquet(path, OBSERVATIONS_SCHEMA)
    out: list[Finding] = []
    for name in table.column_names:
        values = [v for v in table.column(name).to_pylist() if v not in (None, "", REDACTED)]
        if not values:
            continue
        if name in ("raw_artifact_id", "log_artifact_id") and not policy.allow_provenance_artifacts:
            out.append(Finding("provenance-withheld", f"{name} links withheld evidence", rel))
        elif name == "evidence_refs" and any(values) and not policy.allow_provenance_artifacts:
            out.append(Finding("provenance-withheld", "evidence_refs link withheld evidence", rel))
    out.extend(_check_table_text(table, rel, policy, "column"))
    return out


def _check_requests(path: Path, rel: str, policy: RedactionPolicy) -> list[Finding]:
    """Scan every string column of an accepted request table; one finding per column stops the copy."""
    return _check_table_text(read_parquet(path, REQUESTS_SCHEMA), rel, policy, "request column")


def _check_table_text(table: pa.Table, rel: str, policy: RedactionPolicy, where: str) -> list[Finding]:
    """Scan metadata and text, including UTF-8 binary cells; undecodable bytes are invalid."""
    out: list[Finding] = []
    metadata = [table.schema.metadata or {}]

    def field_metadata(field: pa.Field) -> None:
        metadata.append(field.metadata or {})
        if pa.types.is_struct(field.type):
            for child in field.type:
                field_metadata(child)
        elif (
            pa.types.is_list(field.type)
            or pa.types.is_large_list(field.type)
            or pa.types.is_fixed_size_list(field.type)
        ):
            field_metadata(field.type.value_field)
        elif pa.types.is_map(field.type):
            field_metadata(field.type.key_field)
            field_metadata(field.type.item_field)

    for field in table.schema:
        field_metadata(field)
    for entry in metadata:
        decoded = {key.decode("utf-8"): value.decode("utf-8") for key, value in entry.items()}
        out.extend(scan_value(decoded, policy, rel, "Parquet metadata"))
    for name in table.column_names:
        values = table.column(name).to_pylist()
        if values:
            out.extend(scan_value({name: values}, policy, rel, f"{where} {name}")[:1])
    return out


def _check_snapshot(root: Path, rel: str, policy: RedactionPolicy) -> list[Finding]:
    snapshot, _ = read_snapshot(root, rel)
    out = scan_value(snapshot.model_dump(mode="json", by_alias=True), policy, rel, "snapshot metadata")
    for artifact in (snapshot.members, snapshot.acquisition):
        if artifact is not None:
            path = _verify(root, artifact)
            out.extend(
                _check_table_text(
                    read_parquet(path, artifact.artifact_schema), artifact.path, policy, "snapshot column"
                )
            )
    return out


def _check_eval_record(
    root: Path, rel: str, policy: RedactionPolicy, infx_ids: set[str], classified_ids: set[str]
) -> list[Finding]:
    record = load_record(root / rel)
    out = scan_value(record.model_dump(mode="json", by_alias=True), policy, rel, "eval record")
    if missing := set(record.truth.configuration_ids) - classified_ids:
        out.append(
            Finding(
                "eval-configuration-unresolved", f"cannot classify evaluated configurations: {sorted(missing)}", rel
            )
        )
    for artifact in record.truth.artifacts:
        path = _verify((root / rel).parent, artifact)
        artifact_rel = path.relative_to(root).as_posix()
        if artifact.format == "gzip" and not policy.allow_provenance_artifacts:
            out.append(Finding("provenance-withheld", "logs and raw output are withheld", artifact.path))
        if artifact.format == "parquet":
            out.extend(
                _check_table_text(
                    read_parquet(path, artifact.artifact_schema), artifact_rel, policy, "eval truth column"
                )
            )
        elif artifact.format == "json":
            out.extend(scan_value(json.loads(path.read_bytes()), policy, artifact_rel, "eval truth JSON"))
        elif artifact.format != "gzip":
            out.append(Finding("artifact-format", f"unsupported eval truth format: {artifact.format}", artifact_rel))
    if record.findings and not policy.allow_provenance_artifacts:
        out.append(Finding("provenance-withheld", "linked finding artifacts may be logs", rel))
    if policy.inferencex_disclosure == "aggregates-only" and infx_ids & set(record.truth.configuration_ids):
        out.append(Finding("infx-aggregates-only", "record scores InferenceX-derived truth; publish aggregates", rel))
    return out


def check_publishable(
    root: str | Path,
    paths: Iterable[str],
    *,
    manifests: Mapping[str, Manifest] | None = None,
    policy: RedactionPolicy = DEFAULT_POLICY,
) -> list[Finding]:
    """Content-check ``paths`` (repository-relative) for publication. Empty list means publishable.

    ``manifests`` maps manifest path to a manifest already loaded by the caller; manifests
    among ``paths`` that are missing from it are read here.
    """
    root = Path(root)
    chosen = sorted(set(paths))
    loaded = dict(manifests or {})
    findings: list[Finding] = []
    from aisim_core.data.mirror import list_tree, snapshot_documents

    for f in list_tree(root):
        if f.is_symlink:
            findings.append(Finding("symlink", "symbolic links are not published", f.path))
    try:
        for rel in chosen:
            checked_path(root, rel)
    except ValueError as exc:
        findings.append(Finding("path-unsafe", str(exc)))
    if findings:
        return findings
    eval_records = {p.relative_to(root).as_posix() for p in record_paths(root / "eval", recursive=True)}
    for rel in chosen:
        if rel.endswith("/manifest.json") and rel not in loaded:
            try:
                loaded[rel] = Manifest.model_validate_json((root / rel).read_bytes())
            except (OSError, ValidationError, ValueError, RecursionError) as exc:
                findings.append(Finding("document-invalid", str(exc).splitlines()[0], rel))
    infx_ids = set()
    invalid = set()
    for rel, manifest in loaded.items():
        try:
            for artifact in manifest.artifacts:
                checked_path(root, artifact.path)
            if manifest.category == "e2e" and _is_infx(body_from_manifest(manifest).source_kind):
                infx_ids.add(manifest.configuration_id)
        except (ValueError, OSError) as exc:
            findings.append(Finding("document-invalid", str(exc).splitlines()[0], rel))
            invalid.add(rel)
    chosen_set = set(chosen)
    catalog_shards = {p.relative_to(root).as_posix() for p in catalog_files(root)}
    snapshots = snapshot_documents(root, chosen)
    for rel, manifest in sorted(loaded.items()):
        if rel in chosen_set and rel not in invalid:
            try:
                findings.extend(_check_manifest(root, rel, manifest, policy))
            except (OSError, ValidationError, ValueError, RecursionError) as exc:
                findings.append(Finding("document-invalid", str(exc).splitlines()[0], rel))
    for rel in chosen:
        if rel in catalog_shards:
            try:
                parquet = pq.ParquetFile(checked_path(root, rel))
                findings.extend(_check_table_text(parquet.read(), rel, policy, "catalog column"))
                findings.extend(scan_value(parquet.metadata.created_by, policy, rel, "Parquet producer"))
            except (OSError, ValueError, RecursionError) as exc:
                findings.append(Finding("document-invalid", str(exc).splitlines()[0], rel))
        if rel.startswith("systems/") and rel.endswith(".yaml"):
            try:
                data = checked_path(root, rel).read_bytes()
                findings.extend(scan_value(yaml_sidecar_value(data), policy, rel, "system spec"))
                findings.extend(scan_value(data.decode("utf-8"), policy, rel, "system spec source"))
            except (OSError, ValueError, RecursionError) as exc:
                findings.append(Finding("document-invalid", str(exc).splitlines()[0], rel))
        # Match the snapshot documents declared by mirror._check_snapshots.
        if rel in snapshots:
            try:
                findings.extend(_check_snapshot(root, rel, policy))
            except (OSError, ValidationError, ValueError, RecursionError) as exc:
                findings.append(Finding("document-invalid", str(exc).splitlines()[0], rel))
        if rel in eval_records:
            try:
                findings.extend(
                    _check_eval_record(root, rel, policy, infx_ids, {m.configuration_id for m in loaded.values()})
                )
            except (OSError, ValidationError, ValueError, RecursionError) as exc:
                findings.append(Finding("document-invalid", str(exc).splitlines()[0], rel))
    return findings


def _finite_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def check_disclosure(rows: Iterable[Mapping[str, Any]], policy: RedactionPolicy = DEFAULT_POLICY) -> list[Finding]:
    """Check aggregate rows for a public page or catalog (InferenceX-derived or not).

    Only :data:`AGGREGATE_FIELDS` may appear; each row must cover at least ``policy.min_scored``
    scored points; no value may carry a URL, image, patch reference or credential.
    """
    out: list[Finding] = []
    for i, row in enumerate(rows):
        where = f"row {i}"
        for key in sorted(set(row) - AGGREGATE_FIELDS):
            out.append(Finding("not-aggregate", f"field {key!r} is not an allowed aggregate field", where))
        scored = row.get("n_scored")
        if type(scored) is not int or scored < policy.min_scored:
            out.append(Finding("small-cell", f"n_scored must be an integer >= {policy.min_scored}", where))
        total = row.get("n_total")
        if type(scored) is int and type(total) is int and scored >= 0 and total >= 0 and scored > total:
            out.append(Finding("not-aggregate", "n_scored exceeds n_total", where))
        for key, value in row.items():
            if key not in AGGREGATE_FIELDS:
                continue
            if key in {"n_total", "n_scored", "n_configs"}:
                valid = type(value) is int and value >= 0
            elif key in {"coverage", "mape_pct"}:
                valid = value is None or (_finite_number(value) and value >= 0)
                if key == "coverage" and valid:
                    valid = value is None or value <= 1
            elif key == "residual_quantiles_pct":
                valid = value is None or (
                    isinstance(value, Mapping)
                    and set(value) <= {"p0", "p25", "p50", "p75", "p90", "p95", "p99", "p100"}
                    and all(_finite_number(v) for v in value.values())
                )
            else:
                valid = isinstance(value, str)
            if not valid:
                out.append(Finding("not-aggregate", f"invalid aggregate shape for {key!r}", where))
        out.extend(scan_value({k: v for k, v in row.items() if k in AGGREGATE_FIELDS}, policy, where, where))
    return out
