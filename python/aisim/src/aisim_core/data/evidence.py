# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal-only evidence root: permanent raw evidence for configurations that are public.

``data/`` in the private mirror holds unreleased-hardware data that is promoted by a
path-identical copy and then deleted. ``evidence/`` is different: it keeps the raw lane
bundles, logs and collection scripts behind a configuration whose data is public, and it is
never promoted (:data:`aisim_core.data.promote.PROMOTABLE_ROOTS` excludes it) and never
exists in the public dataset (the ``public`` profile rejects the root).

Each configuration has one ``evidence/<config path>/evidence-manifest.json`` listing every
file next to it with its hash. Files are opaque blobs that are stored and hash-checked but
never loaded, so suffix rules do not apply; undeclared files and hash mismatches are errors.
"""

from __future__ import annotations

import hashlib
import posixpath
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aisim_core.data import Category, Digest
from aisim_core.data.hygiene import Finding

EVIDENCE_ROOT = "evidence/"
EVIDENCE_MANIFEST = "evidence-manifest.json"
EVIDENCE_SCHEMA = "aisim-evidence/v1"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class EvidenceArtifact(_Model):
    path: str = Field(min_length=1)  # relative to the manifest's directory
    sha256: Digest
    size: int = Field(ge=0)
    kind: str = Field(min_length=1)  # e.g. raw-bundle, capture-script, log


class EvidenceManifest(_Model):
    schema_version: Literal["aisim-evidence/v1"] = EVIDENCE_SCHEMA
    configuration_id: str = Field(min_length=1)
    category: Category
    config_path: str = Field(min_length=1)  # directory of the public configuration, under ``data/``
    artifacts: list[EvidenceArtifact] = Field(min_length=1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_evidence(root: str | Path, paths: Iterable[str]) -> tuple[list[Finding], set[str]]:
    """Validate every ``evidence/`` file in ``paths``; return findings and the declared paths."""
    root = Path(root)
    files = sorted(p for p in set(paths) if p.startswith(EVIDENCE_ROOT))
    findings: list[Finding] = []
    declared: set[str] = set()
    for rel in files:
        parts = rel.split("/")
        if ".." in parts or "." in parts or "" in parts or posixpath.normpath(rel) != rel or "\\" in rel:
            findings.append(Finding("path-traversal", "path is not normalized", rel))
    manifests = [p for p in files if p.endswith("/" + EVIDENCE_MANIFEST)]
    for rel in manifests:
        declared.add(rel)
        directory = posixpath.dirname(rel)
        try:
            manifest = EvidenceManifest.model_validate_json((root / rel).read_bytes())
        except (OSError, ValidationError, ValueError) as exc:
            findings.append(Finding("document-invalid", f"EvidenceManifest: {str(exc).splitlines()[0]}", rel))
            continue
        if not manifest.config_path.startswith(f"data/{manifest.category}/"):
            findings.append(Finding("evidence-config-path", "config_path must be under data/<category>/", rel))
        if posixpath.join(EVIDENCE_ROOT, manifest.config_path) != directory:
            findings.append(Finding("evidence-config-path", "manifest must sit at evidence/<config_path>/", rel))
        seen: set[str] = set()
        for artifact in manifest.artifacts:
            target = posixpath.normpath(posixpath.join(directory, artifact.path))
            if artifact.path in seen or not target.startswith(directory + "/") or target == rel:
                findings.append(Finding("evidence-artifact", f"duplicate or escaping path {artifact.path!r}", rel))
                continue
            seen.add(artifact.path)
            declared.add(target)
            actual = root / target
            if not actual.is_file() or actual.is_symlink():
                findings.append(Finding("evidence-missing", "declared evidence file is missing", target))
            elif actual.stat().st_size != artifact.size or _sha256(actual) != artifact.sha256:
                findings.append(Finding("evidence-digest", "size or sha256 differs from the manifest", target))
    for rel in files:
        if rel not in declared:
            findings.append(Finding("evidence-undeclared", "file is not listed in an evidence manifest", rel))
    return findings, declared
