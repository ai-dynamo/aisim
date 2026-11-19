# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Promotion from ``aisim-data-internal`` to the public dataset (DEP #435, Phase 7).

Promotion is a path-identical copy of selected configuration and evaluation
directories. It never rewrites bytes: redaction happens first and is checked
(:mod:`aisim_core.data.redaction`), so what is copied is exactly what was
reviewed. The steps, all enforced by :func:`plan_promotion` and
:func:`apply_promotion`:

1. the source is a mirror, never a :data:`~aisim_core.data.mirror.NEVER_PROMOTED`
   cache (``aisim-infx-cache``), and its tree passes the shared validator;
2. every selected file is labeled ``public`` in ``mirror/visibility.json``;
3. the selection is closed: each manifest's artifacts, each evaluation record's truth
   inputs and each snapshot's members are inside the selection, so a snapshot never
   spans private and public membership;
4. the content check finds nothing to redact;
5. named reviewers, from a trusted allow-list and not the requester, approve the exact
   content digest; any change to a selected byte invalidates the approvals;
6. files are copied verbatim, the category indexes and the catalog in the target are
   regenerated, and the target must then pass the ``public`` profile of the validator;
   otherwise the target is restored.

The private promotion ledger (``mirror/promotions/<digest>.json``) keeps the lineage:
the source commit, every promoted file hash and the hash of each unredacted original.
It never reaches the public dataset.

CLI: ``python -m aisim_core.data.promote plan|apply``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aisim_core.data import Artifact, CategoryIndex, DatasetIndex, Entry, Manifest
from aisim_core.data.catalog import CATALOG_DIR, build_catalog
from aisim_core.data.e2e import read_snapshot
from aisim_core.data.eval import load_record, record_paths
from aisim_core.data.hygiene import Finding
from aisim_core.data.mirror import (
    INTERNAL_REPO,
    MIRROR_CONTROL_DIR,
    NEVER_PROMOTED,
    list_tree,
    snapshot_documents,
    validate_tree,
)
from aisim_core.data.redaction import (
    DEFAULT_POLICY,
    RedactionPolicy,
    check_publishable,
    load_ledger,
)

REQUEST_SCHEMA = "aisim-promotion/v1"
PROMOTABLE_ROOTS = ("data/", "eval/")
_SHA40 = r"^[0-9a-f]{40}$"
_SHA64 = r"^[0-9a-f]{64}$"


class PromotionError(Exception):
    """Promotion refused or rolled back."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Approval(_Model):
    reviewer: str = Field(min_length=1)
    approved_at: str = Field(min_length=1)
    content_digest: str = Field(pattern=_SHA64)


class PromotionRequest(_Model):
    schema_version: Literal["aisim-promotion/v1"] = REQUEST_SCHEMA
    source_repo: str = INTERNAL_REPO
    target_repo: str = Field(pattern=r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
    source_commit: str = Field(pattern=_SHA40)  # internal commit the reviewers looked at
    directories: list[str] = Field(min_length=1)
    requested_by: str = Field(min_length=1)
    approvals: list[Approval] = Field(default_factory=list)


@dataclass(frozen=True)
class PromotionPolicy:
    """Trusted configuration. Load it from the default branch, never from the tree being promoted."""

    reviewers: frozenset[str] = frozenset()  # empty means nobody may approve (fail closed)
    min_approvals: int = 1
    redaction: RedactionPolicy = DEFAULT_POLICY

    @classmethod
    def from_file(cls, path: str | Path) -> PromotionPolicy:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("reviewers"), list):
            raise ValueError("reviewers must be a list of non-empty strings")
        if unknown := set(raw) - {"reviewers", "min_approvals"}:
            raise ValueError(f"unknown approval-policy keys: {sorted(unknown)}")
        reviewers = raw["reviewers"]
        if any(not isinstance(name, str) or not name.strip() for name in reviewers):
            raise ValueError("reviewers must be a list of non-empty strings")
        threshold = raw.get("min_approvals", 1)
        if type(threshold) is not int or threshold < 1:
            raise ValueError("min_approvals must be a positive integer")
        return cls(reviewers=frozenset(reviewers), min_approvals=threshold)


@dataclass
class PromotionPlan:
    request: PromotionRequest
    files: dict[str, str] = field(default_factory=dict)  # path -> sha256 of the bytes to copy
    manifests: list[str] = field(default_factory=list)
    lineage: dict[str, str] = field(default_factory=dict)  # promoted path -> unredacted original sha256
    findings: list[Finding] = field(default_factory=list)
    approved_by: list[str] = field(default_factory=list)
    approved_digest: str | None = None

    @property
    def digest(self) -> str:
        """What reviewers approve: target, source commit and every file path and hash."""
        payload = {
            "target_repo": self.request.target_repo,
            "source_commit": self.request.source_commit,
            "files": sorted(self.files.items()),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def content_ok(self) -> bool:
        return not [f for f in self.findings if f.severity == "error" and not f.code.startswith("approval-")]

    @property
    def ok(self) -> bool:
        return not [f for f in self.findings if f.severity == "error"]


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _check_source_checkout(source: Path, commit: str) -> None:
    """Bind provenance to a clean checkout at the exact reviewed revision."""

    def git(*args: str) -> str:
        env = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "GIT_DIR",
                "GIT_WORK_TREE",
                "GIT_COMMON_DIR",
                "GIT_INDEX_FILE",
                "GIT_OBJECT_DIRECTORY",
                "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                "GIT_PREFIX",
                "GIT_SHALLOW_FILE",
                "GIT_GRAFT_FILE",
                "GIT_IMPLICIT_WORK_TREE",
                "GIT_REPLACE_REF_BASE",
                "GIT_NO_REPLACE_OBJECTS",
                "GIT_CONFIG",
                "GIT_CONFIG_PARAMETERS",
                "GIT_CONFIG_COUNT",
            }
            and not key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
        }
        result = subprocess.run(["git", "-C", str(source), *args], capture_output=True, text=True, check=True, env=env)
        return result.stdout.strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != source.resolve():
            raise PromotionError("source root must be the Git checkout root")
        if git("rev-parse", "HEAD") != commit:
            raise PromotionError("source HEAD differs from request.source_commit")
        if git("status", "--porcelain", "--untracked-files=all", "--ignored"):
            raise PromotionError("source changed: checkout has uncommitted or ignored files")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PromotionError("source must be a readable Git checkout") from exc


def _select(source: Path, directories: list[str], plan: PromotionPlan) -> None:
    for raw in directories:
        d = raw.rstrip("/")
        parts = PurePosixPath(d).parts
        if not d or d.startswith("/") or ".." in parts or "." in parts or "\\" in d:
            plan.findings.append(Finding("selection-path", "directory must be a normalized relative path", raw))
        elif not (d + "/").startswith(PROMOTABLE_ROOTS):
            plan.findings.append(
                Finding("selection-root", f"only {list(PROMOTABLE_ROOTS)} directories are promoted", raw)
            )
        elif not (source / d).is_dir() or (source / d).is_symlink():
            plan.findings.append(Finding("selection-missing", "directory does not exist in the source", raw))
        else:
            for f in list_tree(source / d):
                rel = f"{d}/{f.path}"
                if f.is_symlink:
                    plan.findings.append(Finding("symlink", "symbolic links are not promoted", rel))
                else:
                    plan.files[rel] = _sha(source / rel)


def _check_closure(source: Path, plan: PromotionPlan) -> None:
    chosen = set(plan.files)
    snapshots = snapshot_documents(source, chosen)
    eval_records = {p.relative_to(source).as_posix() for p in record_paths(source / "eval", recursive=True)}
    configs: set[str] = set()
    for rel in sorted(chosen):
        if rel.endswith("/manifest.json"):
            try:
                manifest = Manifest.model_validate_json((source / rel).read_bytes())
            except (ValidationError, ValueError, OSError) as exc:
                plan.findings.append(Finding("document-invalid", str(exc).splitlines()[0], rel))
                continue
            plan.manifests.append(rel)
            configs.add(manifest.configuration_id)
            for a in manifest.artifacts:
                if a.path not in chosen:
                    plan.findings.append(Finding("closure", f"artifact {a.path} is outside the selection", rel))
    for rel in sorted(chosen):
        if rel in snapshots:
            try:
                snapshot, members = read_snapshot(source, rel)
            except (ValueError, ValidationError, OSError) as exc:
                plan.findings.append(Finding("snapshot-invalid", str(exc).splitlines()[0], rel))
                continue
            for artifact in (snapshot.members, snapshot.acquisition):
                if artifact is not None and artifact.path not in chosen:
                    plan.findings.append(Finding("closure", f"artifact {artifact.path} is outside the selection", rel))
            outside = sorted({m.manifest_path for m in members} - chosen)
            if outside:
                plan.findings.append(
                    Finding("snapshot-spans", f"snapshot has members outside the selection: {outside[:3]}", rel)
                )
        if rel in eval_records:
            try:
                record = load_record(source / rel)
            except (ValueError, ValidationError, OSError) as exc:
                plan.findings.append(Finding("document-invalid", str(exc).splitlines()[0], rel))
                continue
            base = rel.rsplit("/", 1)[0]
            for a in [*record.truth.artifacts, *record.findings]:
                if f"{base}/{a.path}" not in chosen:
                    plan.findings.append(Finding("closure", f"artifact {a.path} is outside the selection", rel))
    eval_needs = set()
    for rel in chosen:
        if rel in eval_records:
            try:
                eval_needs |= set(load_record(source / rel).truth.configuration_ids)
            except (ValueError, ValidationError, OSError):
                continue
    missing = sorted(eval_needs - configs)
    if missing:
        plan.findings.append(
            Finding(
                "eval-truth-not-selected",
                f"evaluated configurations are not promoted with the record: {missing[:3]}",
                None,
            )
        )


def plan_promotion(
    source_root: str | Path, request: PromotionRequest, policy: PromotionPolicy | None = None
) -> PromotionPlan:
    """Check everything that does not change the target. Approvals are checked last."""
    policy = policy or PromotionPolicy()
    source = Path(source_root)
    plan = PromotionPlan(request)
    try:
        _check_source_checkout(source, request.source_commit)
    except PromotionError as exc:
        plan.findings.append(Finding("source-checkout", str(exc)))
    if policy.redaction != DEFAULT_POLICY:
        plan.findings.append(
            Finding("redaction-policy", "public promotion requires the authoritative default disclosure policy")
        )
    if request.source_repo in NEVER_PROMOTED:
        plan.findings.append(Finding("never-promoted", f"{request.source_repo} is never a promotion source"))
        return plan
    if request.source_repo == request.target_repo:
        plan.findings.append(Finding("same-repo", "source and target must differ"))
        return plan
    shared = validate_tree(source, repo=request.source_repo, profile="mirror")
    plan.findings.extend(Finding(f"source-{f.code}", f.message, f.path, f.severity) for f in shared.errors)
    if any(f.code == "source-symlink" for f in plan.findings):
        return plan
    _select(source, request.directories, plan)
    if not plan.files:
        plan.findings.append(Finding("selection-empty", "selection contains no files"))

    ledger = load_ledger(source)
    for rel in sorted(plan.files):
        entry = ledger.entry_for(rel)
        label = entry.label if entry else "internal"
        if label != "public":
            plan.findings.append(Finding("not-public", f"labeled {label!r}; only 'public' is promoted", rel))
        if entry and rel in entry.original_sha256:
            plan.lineage[rel] = entry.original_sha256[rel]
    _check_closure(source, plan)
    for f in shared.findings:  # unreferenced files would be copied without any content check
        if f.code == "orphan-file" and f.path in plan.files:
            plan.findings.append(Finding("orphan-file", f.message, f.path))
    plan.findings.extend(check_publishable(source, plan.files, policy=policy.redaction))
    _check_approvals(plan, policy)
    return plan


def _check_approvals(plan: PromotionPlan, policy: PromotionPolicy) -> None:
    allowed = {r.casefold() for r in policy.reviewers}  # approvals count by normalized identity
    requester = plan.request.requested_by.casefold()
    names: set[str] = set()
    for approval in plan.request.approvals:
        who = approval.reviewer.casefold()
        if who not in allowed:
            plan.findings.append(Finding("approval-reviewer", f"{approval.reviewer!r} is not a named reviewer"))
        elif who == requester:
            plan.findings.append(Finding("approval-self", "the requester cannot approve their own promotion"))
        elif approval.content_digest != plan.digest:
            plan.findings.append(Finding("approval-stale", f"{approval.reviewer!r} approved different content"))
        else:
            names.add(who)
    plan.approved_by = sorted(names)
    if len(names) < max(1, policy.min_approvals):
        plan.findings.append(
            Finding("approval-missing", f"{len(names)} valid approval(s); {max(1, policy.min_approvals)} required")
        )
    elif plan.ok:
        plan.approved_digest = plan.digest


# --------------------------------------------------------------------------- apply


def _artifact(root: Path, rel: str, schema: str) -> Artifact:
    path = root / rel
    return Artifact(
        artifact_id=rel, path=rel, role="manifest" if schema.startswith("aisim-artifact") else "index",
        format="json", schema=schema, sha256=_sha(path), bytes=path.stat().st_size,
    )  # fmt: skip


def _dump(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json", by_alias=True), indent=2, sort_keys=True, allow_nan=False) + "\n"


def register_manifests(target: Path, manifest_paths: list[str], dataset_id: str) -> None:
    """Add manifests to ``catalog/dataset.json`` and their category indexes (idempotent)."""
    index_path = target / CATALOG_DIR / "dataset.json"
    dataset = (
        DatasetIndex.model_validate_json(index_path.read_bytes())
        if index_path.is_file()
        else DatasetIndex(schema_version="aisim-dataset-index/v1", dataset_id=dataset_id, categories={})
    )
    by_category: dict[str, list[tuple[str, Manifest]]] = {}
    for rel in manifest_paths:
        manifest = Manifest.model_validate_json((target / rel).read_bytes())
        by_category.setdefault(manifest.category, []).append((rel, manifest))
    categories = dict(dataset.categories)
    for category, items in sorted(by_category.items()):
        index_rel = categories[category].index.path if category in categories else f"{CATALOG_DIR}/{category}.json"
        existing = (
            {
                e.configuration_id: e
                for e in CategoryIndex.model_validate_json((target / index_rel).read_bytes()).entries
            }
            if (target / index_rel).is_file()
            else {}
        )
        for rel, manifest in items:
            existing[manifest.configuration_id] = Entry(
                configuration_id=manifest.configuration_id,
                purpose=manifest.purpose,
                identity=manifest.identity,
                manifest=_artifact(target, rel, "aisim-artifact-manifest/v1"),
            )
        index = CategoryIndex(
            schema_version="aisim-category-index/v1",
            category=category,  # type: ignore[arg-type]
            entries=[existing[k] for k in sorted(existing)],
        )
        (target / index_rel).parent.mkdir(parents=True, exist_ok=True)
        (target / index_rel).write_text(_dump(index), encoding="utf-8")
        ref = _artifact(target, index_rel, "aisim-category-index/v1")
        root = categories[category].root if category in categories else f"data/{category}"  # type: ignore[index]
        categories[category] = {"root": root, "index": ref, "layout": "category"}  # type: ignore[index,assignment]
    updated = DatasetIndex.model_validate(
        {**dataset.model_dump(mode="json", by_alias=True), "categories": {
            k: (v if isinstance(v, dict) else v.model_dump(mode="json", by_alias=True)) for k, v in categories.items()
        }}
    )  # fmt: skip
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(_dump(updated), encoding="utf-8")


def _index_paths(target: Path, index_paths: Iterable[str] = ()) -> set[Path]:
    base = target / CATALOG_DIR
    paths = {p for p in base.rglob("*") if p.is_file()} if base.is_dir() else set()
    paths.update(target / rel for rel in index_paths if (target / rel).is_file())
    return paths


def _snapshot(target: Path, index_paths: Iterable[str] = ()) -> dict[str, bytes]:
    return {p.relative_to(target).as_posix(): p.read_bytes() for p in _index_paths(target, index_paths)}


def apply_promotion(
    plan: PromotionPlan, source_root: str | Path, target_root: str | Path, *, write_ledger: bool = True
) -> Path | None:
    """Apply with one writer per target, including validation and rollback.

    A competing writer fails before reading the target snapshot. The adjacent
    lock is removed on exit; after a process crash an operator must verify no
    writer remains before removing its lock.
    """
    target = Path(target_root).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / f".{target.name}.promotion.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise PromotionError(f"another promotion holds the target lock: {lock}") from exc
    try:
        return _apply_promotion(plan, source_root, target_root, write_ledger=write_ledger)
    finally:
        os.close(fd)
        lock.unlink()


def _apply_promotion(
    plan: PromotionPlan, source_root: str | Path, target_root: str | Path, *, write_ledger: bool = True
) -> Path | None:
    """Copy the approved files, regenerate indexes and catalog, and prove the target is valid.

    Raises :class:`PromotionError` (target restored) when anything fails. Returns the ledger path.
    """
    source, target = Path(source_root), Path(target_root)
    if not plan.ok:
        raise PromotionError(
            "plan is not approved: " + "; ".join(f"{f.code} {f.path or ''}" for f in plan.findings[:5])
        )
    if plan.approved_digest is None or plan.digest != plan.approved_digest:
        raise PromotionError("plan content differs from the approved digest")
    ledger_now = load_ledger(source)
    for rel in plan.files:
        if ledger_now.label_for(rel) != "public":
            raise PromotionError(f"source visibility is no longer public: {rel}")
    _check_source_checkout(source, plan.request.source_commit)
    for rel, digest in plan.files.items():  # the reviewed bytes must still be the bytes on disk
        if not (source / rel).is_file() or _sha(source / rel) != digest:
            raise PromotionError(f"source changed since planning: {rel}")
        if (target / rel).exists() and _sha(target / rel) != digest:
            raise PromotionError(f"target already has different content at {rel}; promotion never overwrites")
    index_path = target / CATALOG_DIR / "dataset.json"
    _check_contained(target, list(plan.files) + [f"{CATALOG_DIR}/dataset.json"])
    index_paths = (
        {ref.index.path for ref in DatasetIndex.model_validate_json(index_path.read_bytes()).categories.values()}
        if index_path.is_file()
        else set()
    )
    _check_contained(target, list(index_paths))
    before = _snapshot(target, index_paths)
    added: list[Path] = []
    ledger: Path | None = None
    try:
        for rel in sorted(plan.files):
            dest = target / rel
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                added.append(dest)  # registered first so a partial copy is rolled back too
                shutil.copyfile(source / rel, dest)
        for rel, digest in plan.files.items():
            if _sha(target / rel) != digest:
                raise PromotionError(f"copied bytes differ from the reviewed bytes: {rel}")
        register_manifests(target, plan.manifests, plan.request.target_repo)
        build_catalog(target)
        report = validate_tree(target, repo=plan.request.target_repo, profile="public")
        if not report.ok:
            first = report.errors[0]
            raise PromotionError(
                f"public validator rejects the result: {first.code} {first.path or ''}: {first.message}"
            )
        if write_ledger:  # lineage is part of the promotion: a target without its record is rolled back
            ledger = source / MIRROR_CONTROL_DIR / "promotions" / f"{plan.digest}.json"
            ledger.parent.mkdir(parents=True, exist_ok=True)
            ledger.write_text(_ledger_text(plan, report.fingerprint), encoding="utf-8")
    except BaseException:
        if ledger is not None:
            ledger.unlink(missing_ok=True)
        for path in added:
            path.unlink(missing_ok=True)
            for parent in path.parents:  # drop directories this promotion created
                if parent == target or any(parent.iterdir()):
                    break
                parent.rmdir()
        for rel in {p.relative_to(target).as_posix() for p in _index_paths(target, index_paths)} - set(before):
            (target / rel).unlink(missing_ok=True)
        for directory in sorted((target / CATALOG_DIR).rglob("*"), reverse=True):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        if (target / CATALOG_DIR).is_dir() and not any((target / CATALOG_DIR).iterdir()):
            (target / CATALOG_DIR).rmdir()
        for rel, data in before.items():
            restored = target / rel
            restored.parent.mkdir(parents=True, exist_ok=True)
            restored.write_bytes(data)
        raise
    return ledger


def _ledger_text(plan: PromotionPlan, fingerprint: str) -> str:
    return (
        json.dumps(
            {
                "schema_version": "aisim-promotion-ledger/v1",
                "digest": plan.digest,
                "source_repo": plan.request.source_repo,
                "source_commit": plan.request.source_commit,
                "target_repo": plan.request.target_repo,
                "requested_by": plan.request.requested_by,
                "approved_by": plan.approved_by,
                "files": plan.files,
                "original_sha256": plan.lineage,
                "validator_fingerprint": fingerprint,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _check_contained(target: Path, rels: list[str]) -> None:
    """Refuse a destination whose existing ancestors are symlinks or resolve outside ``target``."""
    root = target.resolve()
    for rel in rels:
        current = target
        for part in Path(rel).parts[:-1]:
            current = current / part
            if current.is_symlink() or (current.exists() and not current.resolve().is_relative_to(root)):
                raise PromotionError(f"target path escapes the target root through a symlink: {rel}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "apply"):
        p = sub.add_parser(name)
        p.add_argument("--source-root", type=Path, required=True)
        p.add_argument("--request", type=Path, required=True)
        p.add_argument("--policy", type=Path, required=True, help="trusted reviewers file (not from the source tree)")
        if name == "apply":
            p.add_argument("--target-root", type=Path, required=True)
    args = parser.parse_args(argv)
    request = PromotionRequest.model_validate_json(args.request.read_bytes())
    plan = plan_promotion(args.source_root, request, PromotionPolicy.from_file(args.policy))
    print(f"content digest {plan.digest}")
    for f in plan.findings:
        print(f"{f.severity.upper()} {f.code} {f.path or ''}: {f.message}", file=sys.stderr)
    if args.command == "plan":
        return 0 if plan.ok else 1
    try:
        apply_promotion(plan, args.source_root, args.target_root)
    except PromotionError as exc:
        print(f"ERROR promotion refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
