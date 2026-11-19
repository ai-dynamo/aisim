# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mirror validator, redaction rules and promotion procedure, on synthetic data only."""

import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pytest

from aisim_core.data import Artifact, Identity, Manifest, configuration_id
from aisim_core.data.catalog import build_catalog
from aisim_core.data.e2e import (
    E2eManifestBody,
    SeriesRow,
    Setting,
    SnapshotAcquisition,
    SnapshotMember,
    aiperf_to_configuration,
    inferencex_to_configurations,
    load_e2e,
    write_configuration,
    write_snapshot,
)
from aisim_core.data.eval import (
    EvalRecord,
    PredictionIdentity,
    compute_results,
    record_dir,
    write_record,
    write_truth_table,
)
from aisim_core.data.mirror import (
    INFX_CACHE_REPO,
    INTERNAL_REPO,
    validate_tree,
    validator_fingerprint,
)
from aisim_core.data.mirror import (
    main as mirror_main,
)
from aisim_core.data.op_schema import OP_TABLE_SCHEMA_ID
from aisim_core.data.parquet import write_parquet
from aisim_core.data.promote import (
    Approval,
    PromotionError,
    PromotionPolicy,
    PromotionRequest,
    apply_promotion,
    plan_promotion,
    register_manifests,
)
from aisim_core.data.redaction import (
    RedactionPolicy,
    VisibilityEntry,
    VisibilityLedger,
    check_disclosure,
    check_publishable,
    write_ledger,
)
from aisim_core.data.schemas import write_snapshots
from tests.unit.data.test_op_publish import _build as build_op_fixture
from tests.unit.data.test_op_publish import tree as op_source  # noqa: F401

pytestmark = pytest.mark.unit

PUBLIC = "nvidia/aisim-data-test"
CLEAN = {"producer_id": None, "image": None, "raw_artifact_id": None, "log_artifact_id": None, "evidence_refs": []}


def _body(model="synthetic/model-a", source="developer"):
    return E2eManifestBody(
        origin="self-collected",
        model_id=model,
        deployment="aggregated",
        hardware="h100_sxm",
        framework="vllm",
        framework_version="0.11.0",
        parallelism={"aggregated": {"tp": 8}},
        parallelism_complete=True,
        precision={"weights": "fp8"},
        settings={"max_num_seqs": Setting(value=256, authority="observed")},
        source_kind=source,
    )


def _summary():
    stat = {"unit": "ms", "avg": 100.0, "p50": 90.0, "p90": 150.0, "p99": 300.0}
    return {
        "is_complete": True,
        "was_cancelled": False,
        "request_count": {"avg": 2},
        "error_request_count": {"avg": 0},
        "time_to_first_token": stat,
        "inter_token_latency": {"unit": "ms", "avg": 10.0, "p50": 9.0, "p90": 14.0},
        "request_latency": {"unit": "seconds", "avg": 1.5, "p50": 1.4, "p90": 2.0},
        "output_token_throughput": {"unit": "tokens/second", "avg": 9600.0},
    }


def _records():
    metrics = {
        "input_sequence_length": {"value": 1024, "unit": "tokens"},
        "output_sequence_length": {"value": 128, "unit": "tokens"},
        "time_to_first_token": {"value": 0.1, "unit": "seconds"},
        "inter_token_latency": {"value": 10, "unit": "ms"},
    }
    return [{"request_id": i, "error": None, "metrics": metrics} for i in range(2)]


def _config(model="synthetic/model-a", *, clean=True, **kw):
    cfg = aiperf_to_configuration(
        _summary(), _records(), _body(model), observation_id=f"dev:{model}", device_count=8, **kw
    )
    if clean:  # the redacted, aggregate-only form that may be public
        cfg = dataclasses.replace(
            cfg,
            observations=[o.model_copy(update=CLEAN) for o in cfg.observations],
            requests=[],
            provenance=[],
        )
    return cfg


def _eval(root, configuration_ids, *, case_ids=5, inputs=None):
    rows = [(f"s{i}", {"tokens": 10 * i, **(inputs or {})}, 10.0 * i) for i in range(1, case_ids + 1)]
    art = write_truth_table(root / "truth.parquet", rows)
    ident = Identity(
        hardware="h100_sxm", framework="vllm", framework_version="0.11.0", operation_family=None,
        model="synthetic/model-a", effective_settings={}, unresolved_identity=None,
        parallelism={"aggregated": {"tp": 8}}, parallelism_complete=True, deployment="aggregated",
    )  # fmt: skip
    results = compute_results([(c, t, t * 1.1) for c, _, t in rows])
    record = EvalRecord.model_validate(
        {
            "schema_version": "aisim-eval/v1",
            "prediction": PredictionIdentity.build("fpm_interpolation", {"p": "b" * 64}, "a" * 64).model_dump(),
            "truth": {
                "layer": "e2e",
                "configuration_ids": configuration_ids,
                "identity": ident.model_dump(),
                "artifacts": [art.model_dump(by_alias=True)],
                "truth_table_artifact_id": "truth-table",
            },
            "results": results.model_dump(),
            "assumptions": [],
            "simulation_cost": {},
            "producer": {"kind": "agent", "name": "t", "evaluator_revision": "r1"},
            "tier": "submitted",
            "created_at": "2026-11-07T00:00:00Z",
        }
    )
    # The truth table sits next to the record, under the record's identity directory.
    directory = record_dir(root, ident)
    directory.mkdir(parents=True, exist_ok=True)
    shutil.move(root / "truth.parquet", directory / "truth.parquet")
    return write_record(root, record)


def _mirror(tmp_path, *, repo=INTERNAL_REPO, dev=("synthetic/model-a",)):
    root = tmp_path / "mirror"
    root.mkdir()
    manifests = [write_configuration(root, _config(m), prefix="data") for m in dev]
    register_manifests(root, manifests, repo)
    build_catalog(root)
    return root, manifests


def _op_mirror(tmp_path, column, value):
    root = tmp_path / "mirror"
    directory = "data/op-latency/h100_sxm/gemm/vllm/0.11.0"
    relative = f"{directory}/gemm_perf.parquet"
    columns = {"kernel_source": ["synthetic-kernel"] * 2, "latency_ms": [1.0, 2.0]}
    columns[column] = ["synthetic-kernel" if column == "kernel_source" else None, value]
    table = pa.table(columns)
    path = write_parquet(table, root / relative, OP_TABLE_SCHEMA_ID)
    identity = Identity(
        hardware="h100_sxm", framework="vllm", framework_version="0.11.0", operation_family="gemm",
        model=None, effective_settings={}, unresolved_identity=None,
    )  # fmt: skip
    manifest = Manifest(
        schema_version="aisim-artifact-manifest/v1",
        category="op-latency",
        purpose="operation-tables",
        configuration_id=configuration_id("op-latency", "operation-tables", identity),
        identity=identity,
        requested_settings=None,
        artifacts=[
            Artifact(
                artifact_id="gemm_perf.parquet",
                path=relative,
                role="operation-table",
                format="parquet",
                schema=OP_TABLE_SCHEMA_ID,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size,
                rows=table.num_rows,
                runtime_path="data/h100_sxm/gemm/vllm/0.11.0/gemm_perf.parquet",
            )
        ],
        provenance={},
        evidence_gaps=[],
    )
    manifest_path = f"{directory}/manifest.json"
    (root / manifest_path).write_text(manifest.model_dump_json(by_alias=True))
    register_manifests(root, [manifest_path], INTERNAL_REPO)
    build_catalog(root)
    _public(root, directory)
    return root, directory, relative


def _public(root, *prefixes, **entry):
    write_ledger(root, VisibilityLedger(entries=[VisibilityEntry(path=p, label="public", **entry) for p in prefixes]))


def _reviewers():
    return PromotionPolicy(reviewers=frozenset({"alice", "bob"}))


def _commit_source(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    if not (root / ".git").exists():
        git("init", "-q")
    git("add", "-A")
    git(
        "-c",
        "user.name=Synthetic Fixture",
        "-c",
        "user.email=fixture@example.test",
        "commit",
        "--no-gpg-sign",
        "-qm",
        "synthetic dataset",
    )
    return git("rev-parse", "HEAD")


def _request(root, directories, plan=None, **kw):
    if not (root / ".git").exists():
        _commit_source(root)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    approvals = [Approval(reviewer="alice", approved_at="2026-11-09", content_digest=plan.digest)] if plan else []
    return PromotionRequest(
        target_repo=PUBLIC,
        source_commit=kw.pop("source_commit", commit),
        directories=directories,
        requested_by="carol",
        approvals=approvals,
        **kw,
    )


# ---------------------------------------------------------------- 2343: mirror validator


def test_clean_mirror_passes_with_repo_id_and_fingerprint(tmp_path):
    root, _ = _mirror(tmp_path)
    report = validate_tree(root, repo=INTERNAL_REPO)
    assert report.ok, report.errors
    assert report.configurations == 1 and report.fingerprint == validator_fingerprint()
    # The expected repository is configurable; a different id is refused.
    assert "dataset-id-mismatch" in {f.code for f in validate_tree(root, repo="ai-dynamo/other").errors}
    assert mirror_main(["validate", "--root", str(root)]) == 0


@pytest.mark.parametrize("profile", ["mirror", "public"])
def test_installed_validation_workflow_is_repository_infrastructure(tmp_path, profile):
    repo = INTERNAL_REPO if profile == "mirror" else PUBLIC
    root, _ = _mirror(tmp_path, repo=repo)
    payload_files = validate_tree(root, repo=repo, profile=profile).files
    workflow = root / ".github/workflows/validate.yml"
    workflow.parent.mkdir(parents=True)
    template = Path(__file__).resolve().parents[5] / ".github/mirror/aisim-data-internal-validate.yml"
    shutil.copyfile(template, workflow)

    report_path = tmp_path / "report.json"
    assert (
        mirror_main(
            ["validate", "--root", str(root), "--repo", repo, "--profile", profile, "--output", str(report_path)]
        )
        == 0
    )
    report = json.loads(report_path.read_text())
    assert report["ok"] and report["configurations"] == 1 and report["files"] == payload_files
    assert not report["findings"]


@pytest.mark.parametrize("profile", ["mirror", "public"])
def test_root_dataset_metadata_is_not_a_payload(tmp_path, profile):
    repo = INTERNAL_REPO if profile == "mirror" else PUBLIC
    root, _ = _mirror(tmp_path, repo=repo)
    payload_files = validate_tree(root, repo=repo, profile=profile).files
    (root / "README.md").write_text("---\nlicense: apache-2.0\n---\n# Synthetic dataset\n")
    (root / ".gitattributes").write_text("*.parquet filter=lfs diff=lfs merge=lfs -text\n")
    report = validate_tree(root, repo=repo, profile=profile)
    assert report.ok, report.errors
    assert report.files == payload_files and report.configurations == 1


@pytest.mark.parametrize("profile", ["mirror", "public"])
@pytest.mark.parametrize("name", ["README.md", ".gitattributes"])
def test_root_dataset_metadata_symlinks_are_rejected(tmp_path, profile, name):
    root, _ = _mirror(tmp_path)
    (root / name).symlink_to(tmp_path / "missing")
    report = validate_tree(root, profile=profile)
    assert any(f.code == "symlink" and f.path == name for f in report.errors)


@pytest.mark.parametrize("name", ["README.md", ".gitattributes"])
@pytest.mark.parametrize(
    "content,code",
    [
        (b"\x7fELFexecutable", "forbidden-content"),
        (b"version https://git-lfs.github.com/spec/v1\n", "lfs-pointer"),
        (b"\xff", "metadata-invalid"),
        (b"Bearer " + b"a" * 24, "secret"),
    ],
)
def test_public_root_metadata_receives_content_checks(tmp_path, name, content, code):
    root, _ = _mirror(tmp_path, repo=PUBLIC)
    (root / name).write_bytes(content)
    report = validate_tree(root, repo=PUBLIC, profile="public")
    assert any(f.code == code and f.path == name for f in report.errors)


def test_root_metadata_does_not_bypass_bootstrap_or_disclosure_rules(tmp_path):
    (tmp_path / "README.md").write_text("Bearer " + "a" * 24)
    assert not validate_tree(tmp_path).ok  # metadata does not replace the dataset index
    assert validate_tree(tmp_path, allow_empty=True).ok  # internal text remains private
    public = validate_tree(tmp_path, profile="public", allow_empty=True)
    assert any(f.code == "secret" and f.path == "README.md" for f in public.errors)


def test_root_metadata_size_is_bounded_before_reading(tmp_path, monkeypatch):
    from aisim_core.data.mirror import MAX_METADATA_BYTES

    path = tmp_path / "README.md"
    with path.open("wb") as handle:
        handle.truncate(MAX_METADATA_BYTES + 1)

    def refuse_read(path):
        pytest.fail("oversized repository metadata was read")

    monkeypatch.setattr(Path, "read_bytes", refuse_read)
    report = validate_tree(tmp_path, allow_empty=True)
    assert any(f.code == "metadata-too-large" and f.path == "README.md" for f in report.errors)


@pytest.mark.parametrize("profile", ["mirror", "public"])
def test_repository_workflow_symlinks_are_rejected(tmp_path, profile):
    root, _ = _mirror(tmp_path)
    workflow = root / ".github/workflows/validate.yml"
    workflow.parent.mkdir(parents=True)
    workflow.symlink_to(tmp_path / "missing.yml")
    report = validate_tree(root, profile=profile)
    assert not report.ok
    assert any(f.code == "symlink" and f.path == ".github/workflows/validate.yml" for f in report.errors)


@pytest.mark.parametrize("profile", ["mirror", "public"])
@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/other.yml",
        ".github/raw.json",
        "data/validate.yml",
        "data/README.md",
        "readme.md",
        ".gitmodules",
    ],
)
def test_repository_infrastructure_exemption_is_narrow(tmp_path, profile, path):
    root, _ = _mirror(tmp_path)
    unexpected = root / path
    unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_text("{}\n")
    report = validate_tree(root, profile=profile)
    assert not report.ok
    assert any(f.path == path and f.code in {"path-root", "suffix"} for f in report.errors)


def test_infx_cache_is_validated_with_its_own_repo_id(tmp_path):
    root, _ = _mirror(tmp_path, repo=INFX_CACHE_REPO)
    assert validate_tree(root, repo=INFX_CACHE_REPO).ok


def test_empty_mirror_needs_explicit_bootstrap(tmp_path):
    (tmp_path / "e").mkdir()
    assert not validate_tree(tmp_path / "e").ok
    assert validate_tree(tmp_path / "e", allow_empty=True).ok
    workflow = tmp_path / "e/.github/workflows/validate.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("name: Validate mirror\n")
    assert not validate_tree(tmp_path / "e").ok
    assert validate_tree(tmp_path / "e", allow_empty=True).ok
    workflow.unlink()
    workflow.symlink_to(tmp_path / "missing.yml")
    assert "symlink" in {f.code for f in validate_tree(tmp_path / "e", allow_empty=True).errors}


@pytest.mark.parametrize("profile", ["mirror", "public"])
def test_internal_fails_on_everything_the_public_validator_rejects(tmp_path, profile):
    """Exit gate: one function judges both repositories, so every defect below fails both."""
    root, manifests = _mirror(tmp_path, repo=PUBLIC if profile == "public" else INTERNAL_REPO)
    repo = PUBLIC if profile == "public" else INTERNAL_REPO
    assert validate_tree(root, repo=repo, profile=profile).ok
    directory = (root / manifests[0]).parent
    metrics = directory / "metrics.parquet"
    original = metrics.read_bytes()

    metrics.write_bytes(original[:-9] + b"corrupt!!")  # tampered table
    assert "manifest-invalid" in {f.code for f in validate_tree(root, repo=repo, profile=profile).errors}
    metrics.write_bytes(original)

    (root / "eval").mkdir()
    (root / "eval" / "bad.json").write_text("{not json")  # malformed record
    assert "eval-record-invalid" in {f.code for f in validate_tree(root, repo=repo, profile=profile).errors}
    (root / "eval" / "bad.json").unlink()

    (root / "data" / "tool.py").write_text("print(1)")  # executable content
    codes = {f.code for f in validate_tree(root, repo=repo, profile=profile).errors}
    assert {"suffix"} <= codes
    (root / "data" / "tool.py").unlink()

    (root / "data" / "link.json").symlink_to(metrics)
    assert "symlink" in {f.code for f in validate_tree(root, repo=repo, profile=profile).errors}
    (root / "data" / "link.json").unlink()
    assert validate_tree(root, repo=repo, profile=profile).ok


def test_catalog_drift_and_schema_drift_fail(tmp_path):
    root, _ = _mirror(tmp_path)
    shard = next((root / "catalog").rglob("*.parquet"))
    shard.write_bytes(shard.read_bytes()[:-5] + b"xxxxx")
    assert "catalog-drift" in {f.code for f in validate_tree(root, repo=INTERNAL_REPO).errors}
    build_catalog(root)
    schemas = root / "catalog" / "schemas"
    schemas.mkdir()
    (schemas / "aisim-eval.v1.json").write_text("{}\n")
    assert "schema-drift" in {f.code for f in validate_tree(root, repo=INTERNAL_REPO).errors}


def test_mirror_control_files_are_private_only(tmp_path):
    root, _ = _mirror(tmp_path)
    _public(root, "data/")
    assert validate_tree(root, repo=INTERNAL_REPO).ok
    report = validate_tree(root, repo=INTERNAL_REPO, profile="public")
    assert "control-file" in {f.code for f in report.errors}


# ---------------------------------------------------------------- 2344: redaction rules


def test_unredacted_run_url_image_digest_patch_and_logs_are_refused(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    dirty = _config(clean=False, log=b"server log\n" * 4, log_redacted=True)
    obs = [
        o.model_copy(
            update={"producer_id": "https://github.com/o/r/actions/runs/123", "image": "nvcr.io/x/y@sha256:" + "d" * 64}
        )
        for o in dirty.observations
    ]
    dirty = dataclasses.replace(dirty, observations=obs)
    path = write_configuration(root, dirty, prefix="data")
    files = [p.relative_to(root).as_posix() for p in (root / path).parent.iterdir()]
    codes = {f.code for f in check_publishable(root, files)}
    assert {"run-url", "provenance-withheld"} <= codes
    assert "request-rows" not in codes  # self-collected per-request rows are publishable
    assert "image-digest" in codes or "image-name" in codes


def _request_row_codes(tmp_path, *, origin="self-collected", source="developer", policy=None):
    root = tmp_path / "r"
    root.mkdir(parents=True)
    body = _body(source=source).model_copy(update={"origin": origin})
    cfg = aiperf_to_configuration(_summary(), _records(), body, observation_id="dev:m", device_count=8)
    cfg = dataclasses.replace(cfg, observations=[o.model_copy(update=CLEAN) for o in cfg.observations], provenance=[])
    assert cfg.requests
    path = write_configuration(root, cfg, prefix="data")
    files = [p.relative_to(root).as_posix() for p in (root / path).parent.iterdir()]
    kwargs = {} if policy is None else {"policy": policy}
    return {f.code for f in check_publishable(root, files, **kwargs)}


def test_request_rows_published_only_for_self_collected_non_infx(tmp_path):
    assert "request-rows" not in _request_row_codes(tmp_path / "a")
    assert "request-rows" in _request_row_codes(tmp_path / "b", origin="official")
    assert "request-rows" in _request_row_codes(tmp_path / "c", source="inferencex")
    strict = RedactionPolicy(allow_self_collected_request_rows=False)
    assert "request-rows" in _request_row_codes(tmp_path / "d", policy=strict)


def test_text_rules_cover_patch_image_and_secret_values():
    from aisim_core.data.redaction import DEFAULT_POLICY, scan_value

    hits = scan_value(
        {"patch_ref": "fix.patch", "container_image": "nvcr.io/nvidia/vllm:25.01", "note": "token hf_" + "a" * 30,
         "u": "https://gitlab-master.nvidia.com/a/b/-/jobs/42", "dispatch_mode": "fast", "ok": "[REDACTED]"},
        DEFAULT_POLICY, "p", "t",
    )  # fmt: skip
    codes = {f.code for f in hits}
    assert {"patch-reference", "image-name", "secret", "run-url", "internal-host"} <= codes
    assert not [
        f for f in scan_value({"dispatch_mode": "fast"}, DEFAULT_POLICY, "p", "t") if f.code == "patch-reference"
    ]


def test_inferencex_derived_is_aggregates_only_by_default(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    (infx,) = inferencex_to_configurations([_infx()])
    infx = dataclasses.replace(
        infx, observations=[o.model_copy(update=CLEAN) for o in infx.observations], provenance=[], requests=[]
    )
    path = write_configuration(root, infx, prefix="data")
    files = [p.relative_to(root).as_posix() for p in (root / path).parent.iterdir()]
    assert "infx-aggregates-only" in {f.code for f in check_publishable(root, files)}
    from aisim_core.data.redaction import RedactionPolicy

    relaxed = RedactionPolicy(inferencex_disclosure="point-level")
    assert "infx-aggregates-only" not in {f.code for f in check_publishable(root, files, policy=relaxed)}


def _infx():
    return dict(
        model="synthetic/model-a", hardware="h100_sxm", framework="vllm", precision="fp8", spec_method="none",
        disagg=False, is_multinode=False, prefill_tp=1, prefill_ep=1, prefill_dp_attention=False,
        prefill_num_workers=0, decode_tp=8, decode_ep=1, decode_dp_attention=False, decode_num_workers=1,
        num_prefill_gpu=0, num_decode_gpu=8, id=1, date="2026-10-17", isl=1024, osl=128, conc=32,
        metrics={"mean_ttft": 0.1, "mean_tpot": 0.01, "tput_per_gpu": 1200.0},
        image="img", benchmark_type="single_turn", run_url="https://github.com/o/r/actions/runs/9",
    )  # fmt: skip


def test_aggregate_disclosure_allows_only_aggregate_fields_and_large_cells():
    ok = {"hardware": "h100", "model": "m", "framework": "vllm", "n_scored": 12, "mape_pct": 4.2, "coverage": 0.9}
    assert check_disclosure([ok]) == []
    leaky = {**ok, "case_id": "s1", "truth": 1.0, "run_url": "https://x/actions/runs/1"}
    assert {"not-aggregate"} <= {f.code for f in check_disclosure([leaky])}
    assert "small-cell" in {f.code for f in check_disclosure([{**ok, "n_scored": 2}])}


# ---------------------------------------------------------------- 2344: promotion


def _promotable(tmp_path):
    root, manifests = _mirror(tmp_path)
    directory = str(Path(manifests[0]).parent)
    _public(root, directory)
    return root, directory


def _target(tmp_path):
    target = tmp_path / "public"
    target.mkdir()
    return target


def test_sample_configuration_is_promoted_by_copy_only_and_passes_public_validator(tmp_path):
    """Exit gate for AISIM-2344."""
    root, directory = _promotable(tmp_path)
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert draft.content_ok and not draft.ok  # needs approval
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    assert plan.ok, plan.findings

    target = _target(tmp_path)
    ledger = apply_promotion(plan, root, target)
    for rel, digest in plan.files.items():  # path-identical and byte-identical
        assert (target / rel).read_bytes() == (root / rel).read_bytes()
    assert not (target / "mirror").exists()
    report = validate_tree(target, repo=PUBLIC, profile="public")
    assert report.ok, report.errors
    assert report.configurations == 1
    entry = json.loads(ledger.read_text())
    assert entry["approved_by"] == ["alice"] and entry["source_commit"] == plan.request.source_commit
    # The source tree still validates as a mirror with its new ledger entry.
    assert validate_tree(root, repo=INTERNAL_REPO).ok


def test_promotion_preserves_target_root_metadata(tmp_path):
    root, directory = _promotable(tmp_path)
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    assert plan.ok, plan.findings
    target = _target(tmp_path)
    metadata = {
        "README.md": b"# Public dataset\n",
        ".gitattributes": b"*.parquet filter=lfs diff=lfs merge=lfs -text\n",
    }
    for name, content in metadata.items():
        (target / name).write_bytes(content)
    apply_promotion(plan, root, target)
    assert validate_tree(target, repo=PUBLIC, profile="public").ok
    assert {name: (target / name).read_bytes() for name in metadata} == metadata
    assert all((target / path).read_bytes() == (root / path).read_bytes() for path in plan.files)


def test_unlabeled_and_restricted_paths_are_not_promoted(tmp_path):
    root, manifests = _mirror(tmp_path)
    directory = str(Path(manifests[0]).parent)
    plan = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert "not-public" in {f.code for f in plan.findings}
    write_ledger(root, VisibilityLedger(entries=[VisibilityEntry(path=directory, label="restricted")]))
    assert "not-public" in {f.code for f in plan_promotion(root, _request(root, [directory]), _reviewers()).findings}


def test_approval_rules(tmp_path):
    root, directory = _promotable(tmp_path)
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    stale = _request(root, [directory], draft)
    stale = stale.model_copy(
        update={"approvals": [Approval(reviewer="alice", approved_at="d", content_digest="0" * 64)]}
    )
    assert "approval-stale" in {f.code for f in plan_promotion(root, stale, _reviewers()).findings}
    self_approved = _request(root, [directory], draft).model_copy(update={"requested_by": "alice"})
    assert "approval-self" in {f.code for f in plan_promotion(root, self_approved, _reviewers()).findings}
    stranger = _request(root, [directory], draft).model_copy(
        update={"approvals": [Approval(reviewer="mallory", approved_at="d", content_digest=draft.digest)]}
    )
    assert "approval-reviewer" in {f.code for f in plan_promotion(root, stranger, _reviewers()).findings}
    # No reviewers configured: fail closed.
    assert not plan_promotion(root, _request(root, [directory], draft), PromotionPolicy()).ok


def test_changed_bytes_invalidate_approval_and_apply(tmp_path, monkeypatch):
    from aisim_core.data import promote

    root, directory = _promotable(tmp_path)
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    real_check = promote._check_source_checkout

    def change_after_checkout_check(*args):
        real_check(*args)
        (root / directory / "metrics.parquet").write_bytes(b"changed")

    monkeypatch.setattr(promote, "_check_source_checkout", change_after_checkout_check)
    target = _target(tmp_path)
    with pytest.raises(PromotionError, match="source changed since planning"):
        apply_promotion(plan, root, target)
    assert list(target.iterdir()) == []


def test_dirty_content_blocks_promotion_even_if_labeled_public(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    dirty = _config(clean=False)
    path = write_configuration(root, dirty, prefix="data")
    register_manifests(root, [path], INTERNAL_REPO)
    build_catalog(root)
    directory = str(Path(path).parent)
    _public(root, directory)
    plan = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert "provenance-withheld" in {f.code for f in plan.findings}
    assert not plan.content_ok


def test_infx_cache_and_same_repo_are_never_sources(tmp_path):
    root, directory = _promotable(tmp_path)
    cache = _request(root, [directory], source_repo=INFX_CACHE_REPO)
    assert "never-promoted" in {f.code for f in plan_promotion(root, cache, _reviewers()).findings}
    same = _request(root, [directory], source_repo=PUBLIC)
    assert "same-repo" in {f.code for f in plan_promotion(root, same, _reviewers()).findings}


def test_selection_must_be_inside_data_or_eval_and_exist(tmp_path):
    root, directory = _promotable(tmp_path)
    codes = {
        f.code for f in plan_promotion(root, _request(root, ["catalog", "../x", "data/nope"]), _reviewers()).findings
    }
    assert {"selection-root", "selection-path", "selection-missing"} <= codes


def test_target_is_restored_when_public_validator_rejects(tmp_path):
    root, directory = _promotable(tmp_path)
    target = _target(tmp_path)
    (target / "data").mkdir()
    (target / "data" / "stray.py").write_text("x")  # public tree already invalid: the result must be refused
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    with pytest.raises(PromotionError, match="public validator rejects"):
        apply_promotion(plan, root, target)
    assert [p.name for p in (target / "data").iterdir()] == ["stray.py"] and not (target / "catalog").exists()


@pytest.mark.parametrize("document_name", ["snapshot.json", "input-snapshot.json", "frozen-input.json"])
def test_snapshot_never_spans_private_and_public(tmp_path, document_name):
    from aisim_core.data.e2e import SnapshotMember, write_snapshot

    root, manifests = _mirror(tmp_path, dev=("synthetic/model-a", "synthetic/model-b"))
    public_dir, private_dir = (str(Path(m).parent) for m in manifests)
    members = [
        SnapshotMember(
            configuration_id=observation.configuration_id,
            observation_id=observation.observation_id, record_sha256=observation.record_sha256, manifest_path=m,
        )
        for m in manifests
        for observation in load_e2e(root, m).observations
    ]  # fmt: skip
    snap_dir = root / "data" / "e2e-snapshots" / "s1"
    snap_dir.mkdir(parents=True)
    write_snapshot(
        root, f"data/e2e-snapshots/s1/{document_name}", members,
        source_scope={"kind": "synthetic"}, selection_policy="p", completeness="complete",
    )  # fmt: skip
    _public(root, public_dir, "data/e2e-snapshots/s1")
    report = validate_tree(root)
    assert report.ok, report.errors
    plan = plan_promotion(root, _request(root, [public_dir, "data/e2e-snapshots/s1"]), _reviewers())
    assert "snapshot-spans" in {f.code for f in plan.findings}
    _ = private_dir


def _snapshot_mirror(tmp_path, *, source_scope=None, selection_policy="synthetic", member=None, acquisition=None):
    root, manifests = _mirror(tmp_path)
    manifest = manifests[0]
    observation = load_e2e(root, manifest).observations[0]
    snapshot_dir = "data/e2e-snapshots/s1"
    (root / snapshot_dir).mkdir(parents=True)
    write_snapshot(
        root,
        f"{snapshot_dir}/snapshot.json",
        [
            SnapshotMember(
                configuration_id=observation.configuration_id,
                observation_id=observation.observation_id,
                record_sha256=observation.record_sha256,
                manifest_path=manifest,
                **(member or {}),
            )
        ],
        source_scope=source_scope or {"kind": "synthetic"},
        selection_policy=selection_policy,
        completeness="complete",
        acquisition=[SnapshotAcquisition(observation_id=observation.observation_id, **acquisition)]
        if acquisition is not None
        else [],
    )
    directories = [str(Path(manifest).parent), snapshot_dir]
    _public(root, *directories)
    return root, directories


@pytest.mark.parametrize(
    ("location", "column"),
    [
        ("metadata", "source_scope"),
        ("metadata", "selection_policy"),
        ("members", "future_text"),
        ("acquisition", "source_uri"),
        ("acquisition", "future_text"),
    ],
)
@pytest.mark.parametrize(
    ("value", "code"),
    [
        ("https://node7.corp.nvidia.com/v1", "internal-host"),
        (json.dumps({"authorization": "Bearer " + "a" * 24}), "secret"),
    ],
)
def test_snapshot_text_blocks_promotion_and_public_validation(tmp_path, location, column, value, code):
    if location == "metadata":
        updates = {column: {"endpoint": value} if column == "source_scope" else value}
    else:
        updates = {"member" if location == "members" else "acquisition": {column: value}}
    root, directories = _snapshot_mirror(tmp_path, **updates)
    private = validate_tree(root, repo=INTERNAL_REPO)
    assert private.ok, private.errors  # valid, hash-linked private data; all referenced configurations are selected
    draft = plan_promotion(root, _request(root, directories), _reviewers())
    plan = plan_promotion(root, _request(root, directories, draft), _reviewers())
    assert not plan.content_ok and not plan.ok  # valid approvals cannot override content checks
    filename = "snapshot.json" if location == "metadata" else f"snapshot-{location}.parquet"
    path = f"{directories[-1]}/{filename}"
    assert any(f.code == code and f.path == path for f in plan.findings)
    assert any(f.code == code and f.path == path for f in validate_tree(root, profile="public").errors)
    target = _target(tmp_path)
    with pytest.raises(PromotionError):
        apply_promotion(plan, root, target)
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("value", [None, "", "[REDACTED]", '{"endpoint": "https://benchmark.example/data"}'])
@pytest.mark.parametrize("with_acquisition", [False, True])
def test_clean_snapshot_is_promoted_by_copy_and_passes_public_validation(tmp_path, value, with_acquisition):
    root, directories = _snapshot_mirror(
        tmp_path,
        source_scope={"kind": "synthetic", "endpoint": value},
        member={"future_text": value},
        acquisition={"source_uri": value, "future_text": value} if with_acquisition else None,
    )
    draft = plan_promotion(root, _request(root, directories), _reviewers())
    plan = plan_promotion(root, _request(root, directories, draft), _reviewers())
    assert plan.ok, plan.findings
    target = _target(tmp_path)
    apply_promotion(plan, root, target)
    for path in (root / directories[-1]).iterdir():
        assert (target / path.relative_to(root)).read_bytes() == path.read_bytes()
    report = validate_tree(target, repo=PUBLIC, profile="public")
    assert report.ok, report.errors


def test_eval_record_is_promoted_with_its_configuration(tmp_path):
    root, manifests = _mirror(tmp_path)
    from aisim_core.data import Manifest

    cid = Manifest.model_validate_json((root / manifests[0]).read_bytes()).configuration_id
    record = _eval(root, [cid])
    build_catalog(root)
    eval_dir = str(record.parent.relative_to(root))
    cfg_dir = str(Path(manifests[0]).parent)
    _public(root, cfg_dir, eval_dir)
    alone = plan_promotion(root, _request(root, [eval_dir]), _reviewers())
    assert "eval-truth-not-selected" in {f.code for f in alone.findings}
    draft = plan_promotion(root, _request(root, [cfg_dir, eval_dir]), _reviewers())
    plan = plan_promotion(root, _request(root, [cfg_dir, eval_dir], draft), _reviewers())
    assert plan.ok, plan.findings
    target = _target(tmp_path)
    apply_promotion(plan, root, target)
    report = validate_tree(target, repo=PUBLIC, profile="public")
    assert report.ok and report.eval_records == 1, report.errors


def test_lineage_of_redacted_originals_is_recorded_privately(tmp_path):
    root, directory = _promotable(tmp_path)
    original = "e" * 64
    path = f"{directory}/metrics.parquet"
    _public(root, directory, original_sha256={path: original})
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    ledger = apply_promotion(plan, root, _target(tmp_path))
    assert json.loads(ledger.read_text())["original_sha256"] == {path: original}


# ---------------------------------------------------------------- evidence root


def _evidence(root, files=None):
    config = "data/e2e/m/h100_sxm/vllm/0.11.0/tp8/cfg-" + "a" * 64
    directory = root / "evidence" / config
    directory.mkdir(parents=True)
    files = files or {"bundle.tar.gz": b"\x1f\x8bbundle", "capture.sh": b"#!/bin/sh\n"}
    artifacts = []
    for name, data in files.items():
        (directory / name).write_bytes(data)
        artifacts.append(
            {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "kind": "raw-bundle"}
        )
    manifest = {"configuration_id": "cfg-" + "a" * 64, "category": "e2e", "config_path": config, "artifacts": artifacts}
    (directory / "evidence-manifest.json").write_text(json.dumps(manifest))
    return directory


def test_evidence_root_is_internal_only_and_hash_checked(tmp_path):
    root, _ = _mirror(tmp_path)
    directory = _evidence(root)
    assert validate_tree(root, repo=INTERNAL_REPO).ok
    public = validate_tree(root, repo=INTERNAL_REPO, profile="public")
    assert "path-root" in {f.code for f in public.errors}
    (directory / "stray.log").write_text("x")
    assert "evidence-undeclared" in {f.code for f in validate_tree(root, repo=INTERNAL_REPO).errors}
    (directory / "stray.log").unlink()
    (directory / "capture.sh").write_text("#!/bin/sh\necho changed\n")
    assert "evidence-digest" in {f.code for f in validate_tree(root, repo=INTERNAL_REPO).errors}
    (directory / "capture.sh").unlink()
    assert "evidence-missing" in {f.code for f in validate_tree(root, repo=INTERNAL_REPO).errors}


def test_evidence_is_never_promoted(tmp_path):
    root, _ = _mirror(tmp_path)
    _evidence(root)
    plan = plan_promotion(root, _request(root, ["evidence/data"]), _reviewers())
    assert "selection-root" in {f.code for f in plan.findings}


def test_eval_record_scoring_infx_truth_is_aggregates_only(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    (infx,) = inferencex_to_configurations([_infx()])
    infx = dataclasses.replace(
        infx, observations=[o.model_copy(update=CLEAN) for o in infx.observations], provenance=[], requests=[]
    )
    manifest = write_configuration(root, infx, prefix="data")
    register_manifests(root, [manifest], INTERNAL_REPO)
    from aisim_core.data import Manifest

    cid = Manifest.model_validate_json((root / manifest).read_bytes()).configuration_id
    record = _eval(root, [cid])
    build_catalog(root)
    eval_dir = str(record.parent.relative_to(root))
    cfg_dir = str(Path(manifest).parent)
    _public(root, cfg_dir, eval_dir)
    files = [p.relative_to(root).as_posix() for d in (cfg_dir, eval_dir) for p in (root / d).rglob("*") if p.is_file()]
    codes = {f.code for f in check_publishable(root, files)}
    assert "infx-aggregates-only" in codes
    plan = plan_promotion(root, _request(root, [cfg_dir, eval_dir]), _reviewers())
    assert "infx-aggregates-only" in {f.code for f in plan.findings}
    relaxed = RedactionPolicy(inferencex_disclosure="point-level")
    assert "infx-aggregates-only" not in {f.code for f in check_publishable(root, files, policy=relaxed)}


def test_request_table_text_is_scanned(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    records = _records()
    records[0]["error"] = "connect failed to node7.corp.nvidia.com with Bearer " + "a" * 24
    body = _body()
    cfg = aiperf_to_configuration(_summary(), records, body, observation_id="dev:m", device_count=8)
    cfg = dataclasses.replace(cfg, observations=[o.model_copy(update=CLEAN) for o in cfg.observations], provenance=[])
    path = write_configuration(root, cfg, prefix="data")
    files = [p.relative_to(root).as_posix() for p in (root / path).parent.iterdir()]
    found = check_publishable(root, files)
    assert {"internal-host", "secret"} & {f.code for f in found}
    assert any("request column error" in f.message for f in found)


@pytest.mark.parametrize("column", ["attrs", "qualification", "future_text"])
@pytest.mark.parametrize(
    ("value", "code"),
    [
        (json.dumps({"benchmark_url": "https://node7.corp.nvidia.com/report"}), "internal-host"),
        (json.dumps({"authorization": "Bearer " + "a" * 24}), "secret"),
    ],
)
def test_observation_text_blocks_promotion_and_public_validation(tmp_path, column, value, code):
    root = tmp_path / "mirror"
    cfg = _config()
    cfg = dataclasses.replace(cfg, observations=[o.model_copy(update={column: value}) for o in cfg.observations])
    manifest = write_configuration(root, cfg, prefix="data")
    register_manifests(root, [manifest], INTERNAL_REPO)
    build_catalog(root)
    directory = str(Path(manifest).parent)
    _public(root, directory)
    assert validate_tree(root, repo=INTERNAL_REPO).ok  # valid private data, including extension columns
    plan = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert not plan.content_ok
    assert any(f.code == code and f"column {column}" in f.message for f in plan.findings)
    assert code in {f.code for f in validate_tree(root, profile="public").errors}


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("metrics", "attrs"),
        ("metrics", "source_name"),
        ("metrics", "future_text"),
        ("series", "source"),
        ("series", "future_text"),
    ],
)
@pytest.mark.parametrize(
    ("value", "code"),
    [
        (json.dumps({"endpoint": "https://node7.corp.nvidia.com/v1"}), "internal-host"),
        (json.dumps({"authorization": "Bearer " + "a" * 24}), "secret"),
    ],
)
def test_metric_and_series_text_blocks_promotion_and_public_validation(tmp_path, table, column, value, code):
    root = tmp_path / "mirror"
    cfg = _config()
    if table == "series":
        cfg = dataclasses.replace(
            cfg,
            series=[
                SeriesRow(
                    observation_id=cfg.observations[0].observation_id,
                    metric="running_requests",
                    start_ms=0.0,
                    end_ms=100.0,
                    value=1.0,
                )
            ],
        )
    rows = list(getattr(cfg, table))
    rows[0] = rows[0].model_copy(update={column: value})
    cfg = dataclasses.replace(cfg, **{table: rows})
    manifest = write_configuration(root, cfg, prefix="data")
    register_manifests(root, [manifest], INTERNAL_REPO)
    build_catalog(root)
    directory = str(Path(manifest).parent)
    _public(root, directory)
    assert validate_tree(root, repo=INTERNAL_REPO).ok  # valid private data, not an orphan or corrupt table
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    assert not plan.content_ok and not plan.ok  # approval does not override disclosure checks
    artifact_path = f"{directory}/{table}.parquet"
    assert any(f.code == code and f.path == artifact_path and f"column {column}" in f.message for f in plan.findings)
    assert any(f.code == code and f.path == artifact_path for f in validate_tree(root, profile="public").errors)
    with pytest.raises(PromotionError):
        apply_promotion(plan, root, _target(tmp_path))


@pytest.mark.parametrize("value", [None, "", "[REDACTED]", '{"source": "public-benchmark"}'])
def test_clean_metric_and_series_text_can_be_promoted(tmp_path, value):
    root = tmp_path / "mirror"
    cfg = _config()
    cfg = dataclasses.replace(
        cfg,
        metrics=[row.model_copy(update={"attrs": value}) for row in cfg.metrics],
        series=[
            SeriesRow(
                observation_id=cfg.observations[0].observation_id,
                metric="running_requests",
                start_ms=0.0,
                end_ms=100.0,
                value=1.0,
                source=value,
            )
        ],
    )
    manifest = write_configuration(root, cfg, prefix="data")
    register_manifests(root, [manifest], INTERNAL_REPO)
    build_catalog(root)
    directory = str(Path(manifest).parent)
    _public(root, directory)
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    assert plan.ok, plan.findings
    target = _target(tmp_path)
    apply_promotion(plan, root, target)
    for table in ("metrics", "series"):
        relative = f"{directory}/{table}.parquet"
        assert (target / relative).read_bytes() == (root / relative).read_bytes()
    report = validate_tree(target, repo=PUBLIC, profile="public")
    assert report.ok, report.errors


@pytest.mark.parametrize(
    ("column", "value", "code"),
    [
        ("kernel_source", "https://node7.corp.nvidia.com/v1", "internal-host"),
        ("kernel_source", "Bearer " + "a" * 24, "secret"),
        ("future_text", "https://node7.corp.nvidia.com/v1", "internal-host"),
        ("future_text", "Bearer " + "a" * 24, "secret"),
        ("future_text", json.dumps({"endpoint": "https://node7.corp.nvidia.com/v1"}), "internal-host"),
        ("future_text", {"authorization": "Bearer " + "a" * 24}, "secret"),
        ("future_text", ["https://node7.corp.nvidia.com/v1"], "internal-host"),
    ],
)
def test_operation_table_text_blocks_promotion_and_public_validation(tmp_path, column, value, code):
    root, directory, relative = _op_mirror(tmp_path, column, value)
    report = validate_tree(root, repo=INTERNAL_REPO)
    assert report.ok, report.errors  # accepted private table with valid schema, hashes and catalog
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    assert not plan.content_ok and not plan.ok  # approval cannot override disclosure checks
    assert any(f.code == code and f.path == relative and f"column {column}" in f.message for f in plan.findings)
    assert any(f.code == code and f.path == relative for f in validate_tree(root, profile="public").errors)
    target = _target(tmp_path)
    with pytest.raises(PromotionError):
        apply_promotion(plan, root, target)
    assert not (target / relative).exists()


@pytest.mark.parametrize("value", [None, "", "[REDACTED]", '{"source": "public-benchmark"}'])
def test_clean_operation_table_text_can_be_promoted(tmp_path, value):
    root, directory, relative = _op_mirror(tmp_path, "future_text", value)
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    assert plan.ok, plan.findings
    target = _target(tmp_path)
    apply_promotion(plan, root, target)
    assert (target / relative).read_bytes() == (root / relative).read_bytes()
    report = validate_tree(target, repo=PUBLIC, profile="public")
    assert report.ok, report.errors


@pytest.mark.parametrize(
    ("inputs", "code"),
    [
        ({"endpoint": "https://node7.corp.nvidia.com/v1"}, "internal-host"),
        ({"headers": {"authorization": "Bearer " + "a" * 24}}, "secret"),
    ],
)
def test_eval_truth_text_blocks_promotion_and_public_validation(tmp_path, inputs, code):
    root, manifests = _mirror(tmp_path)
    cfg = _config()
    record = _eval(root, [cfg.body.configuration_id()], inputs=inputs)
    build_catalog(root)
    cfg_dir = str(Path(manifests[0]).parent)
    eval_dir = record.parent.relative_to(root).as_posix()
    _public(root, cfg_dir, eval_dir)
    assert validate_tree(root, repo=INTERNAL_REPO).ok  # hashes and schemas are valid
    plan = plan_promotion(root, _request(root, [cfg_dir, eval_dir]), _reviewers())
    assert not plan.content_ok
    assert any(f.code == code and f.path == f"{eval_dir}/truth.parquet" for f in plan.findings)
    assert code in {f.code for f in validate_tree(root, profile="public").errors}


def test_conflicting_ledger_entries_resolve_to_the_most_restrictive_label():
    for entries in (
        [("data/a", "public"), ("data/a", "restricted")],
        [("data/a", "restricted"), ("data/a/", "public")],
    ):
        ledger = VisibilityLedger(entries=[VisibilityEntry(path=p, label=lab) for p, lab in entries])
        assert ledger.label_for("data/a/x") == "restricted"
    narrow = VisibilityLedger(
        entries=[VisibilityEntry(path="data/a", label="restricted"), VisibilityEntry(path="data/a/b", label="public")]
    )
    assert narrow.label_for("data/a/b/c") == "public"  # a narrower prefix still wins


def test_unreferenced_file_blocks_promotion_and_public_validation(tmp_path):
    root, directory = _promotable(tmp_path)
    (root / directory / "notes.json").write_text('{"token": "private"}')
    plan = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert "orphan-file" in {f.code for f in plan.findings}
    assert any(
        f.code == "orphan-file" and f.severity == "error" for f in validate_tree(root, profile="public").findings
    )
    assert any(
        f.severity == "warning" and f.code == "orphan-file" and f.path == f"{directory}/notes.json"
        for f in validate_tree(root).findings
    )


def test_one_reviewer_in_two_spellings_counts_once(tmp_path):
    root, directory = _promotable(tmp_path)
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    two = _request(root, [directory], draft).model_copy(
        update={
            "approvals": [
                Approval(reviewer="alice", approved_at="d", content_digest=draft.digest),
                Approval(reviewer="Alice", approved_at="d", content_digest=draft.digest),
            ]
        }
    )
    policy = PromotionPolicy(reviewers=frozenset({"alice", "bob"}), min_approvals=2)
    assert "approval-missing" in {f.code for f in plan_promotion(root, two, policy).findings}
    distinct = two.model_copy(
        update={
            "approvals": [Approval(reviewer=r, approved_at="d", content_digest=draft.digest) for r in ("alice", "bob")]
        }
    )
    assert plan_promotion(root, distinct, policy).ok


def _approved(tmp_path):
    root, directory = _promotable(tmp_path)
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    return root, directory, plan_promotion(root, _request(root, [directory], draft), _reviewers())


def test_symlinked_target_directory_is_refused_without_writing_outside(tmp_path):
    root, directory, plan = _approved(tmp_path)
    target = _target(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (target / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PromotionError, match="symlink"):
        apply_promotion(plan, root, target)
    assert list(outside.iterdir()) == []


def test_failed_copy_and_failed_ledger_restore_the_target(tmp_path, monkeypatch):
    root, directory, plan = _approved(tmp_path)
    target = _target(tmp_path)
    real = shutil.copyfile

    def partial(src, dst, *a, **k):
        Path(dst).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr("aisim_core.data.promote.shutil.copyfile", partial)
    with pytest.raises(OSError, match="disk full"):
        apply_promotion(plan, root, target)
    assert not any(p.is_file() for p in target.rglob("*"))
    monkeypatch.setattr("aisim_core.data.promote.shutil.copyfile", real)

    real_write = Path.write_text
    partial_ledgers = []

    def deny(self, *a, **k):
        if "promotions" in self.parts:
            self.write_bytes(b'{"partial":')
            partial_ledgers.append(self)
            raise PermissionError("read-only")
        return real_write(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", deny)
    with pytest.raises(PermissionError):
        apply_promotion(plan, root, target)
    assert not any(p.is_file() for p in target.rglob("*"))
    assert len(partial_ledgers) == 1 and not partial_ledgers[0].exists()


@pytest.mark.parametrize("case", ["alias", "configuration"])
def test_evidence_rejects_aliases_and_wrong_configuration(tmp_path, case):
    root, _ = _mirror(tmp_path)
    directory = _evidence(root, {"raw.log": b"fixture"})
    path = directory / "evidence-manifest.json"
    doc = json.loads(path.read_bytes())
    if case == "alias":
        doc["artifacts"].append({**doc["artifacts"][0], "path": "./raw.log", "kind": "another-kind"})
        code = "evidence-artifact"
    else:
        doc["configuration_id"] = "cfg-" + "b" * 64
        code = "evidence-configuration"
    path.write_text(json.dumps(doc))
    assert code in {f.code for f in validate_tree(root).errors}


@pytest.mark.parametrize("extension", [["Bearer " + "a" * 24], {"nested": "node.corp.nvidia.com"}])
def test_nested_table_extension_text_blocks_publication(tmp_path, extension):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisim_core.data import Manifest

    root, manifests = _mirror(tmp_path)
    path = root / manifests[0]
    manifest = Manifest.model_validate_json(path.read_bytes())
    metrics = next(a for a in manifest.artifacts if a.artifact_schema == "aisim-e2e-metrics/v1")
    table = pq.read_table(root / metrics.path)
    table = table.append_column("extension", pa.array([extension] * table.num_rows))
    pq.write_table(table, root / metrics.path)
    artifacts = [
        a.model_copy(
            update={
                "sha256": hashlib.sha256((root / a.path).read_bytes()).hexdigest(),
                "bytes": (root / a.path).stat().st_size,
            }
        )
        if a == metrics
        else a
        for a in manifest.artifacts
    ]
    path.write_text(manifest.model_copy(update={"artifacts": artifacts}).model_dump_json(by_alias=True))
    code = "secret" if isinstance(extension, list) else "internal-host"
    assert code in {f.code for f in check_publishable(root, manifests)}
    register_manifests(root, manifests, INTERNAL_REPO)
    build_catalog(root)
    assert code in {f.code for f in validate_tree(root, profile="public").errors}


def test_declared_json_sidecar_is_scanned(tmp_path):
    from aisim_core.data import Artifact, Manifest

    root, manifests = _mirror(tmp_path)
    path = root / manifests[0]
    manifest = Manifest.model_validate_json(path.read_bytes())
    rel = path.parent.relative_to(root).as_posix() + "/errors.json"
    payload = b'{"errors":["Bearer aaaaaaaaaaaaaaaaaaaaaaaa node.corp.nvidia.com"]}'
    (root / rel).write_bytes(payload)
    artifact = Artifact(
        artifact_id="errors",
        path=rel,
        role="errors",
        format="json",
        schema="collector-errors/unversioned",
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
    )
    path.write_text(
        manifest.model_copy(update={"artifacts": [*manifest.artifacts, artifact]}).model_dump_json(by_alias=True)
    )
    register_manifests(root, manifests, INTERNAL_REPO)
    build_catalog(root)
    directory = path.parent.relative_to(root).as_posix()
    _public(root, directory)
    assert validate_tree(root).ok
    assert {"secret", "internal-host"} <= {f.code for f in check_publishable(root, manifests)}
    assert "secret" in {f.code for f in plan_promotion(root, _request(root, [directory]), _reviewers()).findings}
    assert "secret" in {f.code for f in validate_tree(root, profile="public").errors}


def test_malformed_e2e_body_returns_findings(tmp_path):
    root, manifests = _mirror(tmp_path)
    path = root / manifests[0]
    doc = json.loads(path.read_bytes())
    doc["provenance"] = {}
    path.write_text(json.dumps(doc))
    findings = check_publishable(root, manifests)
    assert any(f.code == "document-invalid" and f.path == manifests[0] for f in findings)
    assert not validate_tree(root, profile="public").ok


@pytest.mark.parametrize("value", [[0.1, 0.2], {"case-a": 0.1}, True, float("nan")])
def test_aggregate_shapes_cannot_carry_point_level_measurements(value):
    assert "not-aggregate" in {f.code for f in check_disclosure([{"n_scored": 5, "coverage": value}])}
    assert (
        check_disclosure([{"n_scored": 5, "coverage": 0.9, "residual_quantiles_pct": {"p50": -1.0, "p99": 2.0}}]) == []
    )
    assert "not-aggregate" in {
        f.code for f in check_disclosure([{"n_scored": 5, "residual_quantiles_pct": {"p50": [1.0, 2.0]}}])
    }


def test_text_keys_and_json_encoded_field_context_are_scanned():
    from aisim_core.data.redaction import DEFAULT_POLICY, scan_value

    for value in (True, None):
        assert "secret" in {f.code for f in scan_value({"Bearer " + "a" * 24: value}, DEFAULT_POLICY, "p", "t")}
    for value in ("private-image:tag", json.dumps(["private-image:tag"]), json.dumps({"name": "private-image:tag"})):
        assert "image-name" in {f.code for f in scan_value({"image": value}, DEFAULT_POLICY, "p", "t")}
    assert scan_value({"image": "[REDACTED]", "dispatch_mode": "fast"}, DEFAULT_POLICY, "p", "t") == []


@pytest.mark.parametrize("profile", ["mirror", "public"])
def test_symlinked_documents_are_rejected_before_any_read(tmp_path, monkeypatch, profile):
    root, _ = _mirror(tmp_path)
    index = root / "catalog/dataset.json"
    outside = tmp_path / "external.json"
    index.replace(outside)
    index.symlink_to(outside)
    original = Path.open
    reads = []

    def observe(path, *args, **kwargs):
        if path.resolve() == outside.resolve():
            reads.append(path)
            pytest.fail("validator followed a document symlink")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observe)
    report = validate_tree(root, profile=profile)
    assert "symlink" in {f.code for f in report.errors} and not reads
    assert "symlink" in {f.code for f in check_publishable(root, ["catalog/dataset.json"])}
    assert not reads


def test_category_index_io_error_is_reported(tmp_path, monkeypatch):
    from aisim_core.data import mirror

    root, _ = _mirror(tmp_path)
    real = mirror._verify

    def deny(root, artifact):
        if artifact.path == "catalog/e2e.json":
            raise PermissionError("denied fixture index")
        return real(root, artifact)

    monkeypatch.setattr(mirror, "_verify", deny)
    output = tmp_path / "report.json"
    assert mirror_main(["validate", "--root", str(root), "--output", str(output)]) == 1
    assert "index-integrity" in {f["code"] for f in json.loads(output.read_bytes())["findings"]}


@pytest.mark.parametrize("change", ["missing", "configuration", "observation", "hash", "none"])
def test_standalone_snapshot_validation_checks_member_contents(tmp_path, change):
    root, directories = _snapshot_mirror(tmp_path)
    path = directories[0] + "/manifest.json"
    snapshot_dir = directories[1]
    obs = load_e2e(root, path).observations[0]
    member = SnapshotMember(
        configuration_id=obs.configuration_id,
        observation_id=obs.observation_id,
        record_sha256=obs.record_sha256,
        manifest_path=path,
    )
    updates = {
        "missing": {"manifest_path": "data/e2e/missing/manifest.json"},
        "configuration": {"configuration_id": "cfg-" + "f" * 64},
        "observation": {"observation_id": "absent"},
        "hash": {"record_sha256": "f" * 64},
        "none": {},
    }
    write_snapshot(
        root,
        snapshot_dir + "/snapshot.json",
        [member.model_copy(update=updates[change])],
        source_scope={"kind": "synthetic"},
        selection_policy="synthetic",
        completeness="complete",
    )
    shutil.rmtree(root / "mirror")  # a standalone public tree has no private visibility ledger
    report = validate_tree(root, profile="public")
    if change == "none":
        assert report.ok, report.errors
    else:
        assert "snapshot-invalid" in {f.code for f in report.errors}


def test_nondefault_disclosure_policy_is_rejected_during_planning(tmp_path):
    root, directory = _promotable(tmp_path)
    policy = PromotionPolicy(
        reviewers=frozenset({"alice"}), redaction=RedactionPolicy(inferencex_disclosure="point-level")
    )
    plan = plan_promotion(root, _request(root, [directory]), policy)
    assert not plan.content_ok and "redaction-policy" in {f.code for f in plan.findings}


def test_rollback_restores_a_removed_catalog_directory(tmp_path, monkeypatch):
    from aisim_core.data import promote as promotion
    from aisim_core.data.hygiene import Finding
    from aisim_core.data.mirror import TreeReport

    root, _, plan = _approved(tmp_path)
    target = _target(tmp_path)
    stale = target / "catalog/fpm/obsolete.parquet"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale fixture shard")
    monkeypatch.setattr(
        promotion,
        "validate_tree",
        lambda *a, **k: TreeReport(repo=PUBLIC, profile="public", findings=[Finding("late", "rejected")]),
    )
    with pytest.raises(PromotionError, match="late"):
        apply_promotion(plan, root, target)
    assert stale.read_bytes() == b"stale fixture shard"
    assert [p for p in target.rglob("*") if p.is_file()] == [stale]


def test_promotion_rechecks_visibility_without_selected_byte_changes(tmp_path):
    root, directory, plan = _approved(tmp_path)
    selected_before = {rel: (root / rel).read_bytes() for rel in plan.files}
    write_ledger(root, VisibilityLedger(entries=[VisibilityEntry(path=directory, label="restricted")]))
    target = _target(tmp_path)
    with pytest.raises(PromotionError, match="visibility"):
        apply_promotion(plan, root, target)
    assert not list(target.iterdir())
    assert selected_before == {rel: (root / rel).read_bytes() for rel in plan.files}


def test_promotion_source_commit_and_clean_checkout_are_enforced(tmp_path):
    root, directory, plan = _approved(tmp_path)
    wrong = _request(root, [directory], source_commit="a" * 40)
    assert "source-checkout" in {f.code for f in plan_promotion(root, wrong, _reviewers()).findings}
    dirty = root / "mirror/extra.json"
    dirty.write_text("{}")
    assert "source-checkout" in {f.code for f in plan_promotion(root, plan.request, _reviewers()).findings}
    with pytest.raises(PromotionError, match="uncommitted"):
        apply_promotion(plan, root, _target(tmp_path))
    _commit_source(root)
    with pytest.raises(PromotionError, match="HEAD"):
        apply_promotion(plan, root, tmp_path / "another-target")


def _extra_truth_artifact(root, manifests, name, payload, artifact_format):
    from aisim_core.data import Artifact
    from aisim_core.data.eval import load_record

    cid = load_e2e(root, manifests[0]).manifest.configuration_id
    original = _eval(root, [cid])
    record = load_record(original)
    path = original.parent / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    artifact = Artifact(
        artifact_id="extra-truth",
        path=name,
        role="truth",
        format=artifact_format,
        schema="extra-truth/v1",
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        rows=1 if artifact_format == "parquet" else None,
    )
    truth = record.truth.model_copy(update={"artifacts": [*record.truth.artifacts, artifact]})
    record = record.model_copy(update={"truth": truth, "supersedes": record.record_id})
    new_path = write_record(root, record)
    build_catalog(root)
    return new_path.relative_to(root).as_posix()


@pytest.mark.parametrize(
    "payload,code",
    [(b"\x7fELFmalicious", "forbidden-content"), (b"version https://git-lfs.github.com/spec/v1\n", "lfs-pointer")],
)
def test_declared_gzip_still_receives_content_checks(tmp_path, payload, code):
    root, manifests = _mirror(tmp_path)
    _extra_truth_artifact(root, manifests, "raw.gz", payload, "gzip")
    report = validate_tree(root)
    assert code in {f.code for f in report.errors}


def test_gzip_truth_is_withheld_from_publication(tmp_path):
    import gzip

    root, manifests = _mirror(tmp_path)
    rel = _extra_truth_artifact(root, manifests, "raw.gz", gzip.compress(b"private original output"), "gzip")
    assert validate_tree(root).ok
    assert "provenance-withheld" in {f.code for f in check_publishable(root, [rel])}
    assert "provenance-withheld" in {f.code for f in validate_tree(root, profile="public").errors}


def test_uppercase_large_parquet_uses_its_footer(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root, manifests = _mirror(tmp_path)
    path = tmp_path / "large.parquet"
    table = pa.table({"payload": ["x" * (5 * 1024 * 1024)]}).replace_schema_metadata(
        {b"aisim.schema": b"extra-truth/v1"}
    )
    pq.write_table(table, path, compression=None, use_dictionary=False)
    assert path.stat().st_size > 4 * 1024 * 1024
    _extra_truth_artifact(root, manifests, "large.PARQUET", path.read_bytes(), "parquet")
    report = validate_tree(root)
    assert report.ok, report.errors


def test_operation_publisher_yaml_sidecars_can_be_promoted(tmp_path, op_source):  # noqa: F811
    out = tmp_path / "published"
    build_op_fixture(op_source, out)
    root = tmp_path / "mirror"
    directory = "data/op-latency/h200_sxm/gemm/vllm"
    shutil.copytree(out / directory, root / directory)
    manifests = [p.relative_to(root).as_posix() for p in (root / directory).rglob("manifest.json")]
    register_manifests(root, manifests, INTERNAL_REPO)
    build_catalog(root)
    _public(root, directory)
    assert validate_tree(root).ok
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert draft.content_ok, draft.findings
    approved = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    target = _target(tmp_path)
    apply_promotion(approved, root, target)
    assert validate_tree(target, repo=PUBLIC, profile="public").ok
    for path in (root / directory).rglob("*.yaml"):
        assert (target / path.relative_to(root)).read_bytes() == path.read_bytes()


def _manifest_sidecar(root, manifest_rel, name, payload, fmt, role, schema):
    manifest_path = root / manifest_rel
    manifest = Manifest.model_validate_json(manifest_path.read_bytes())
    rel = (manifest_path.parent / name).relative_to(root).as_posix()
    (root / rel).write_bytes(payload)
    artifact = Artifact(
        artifact_id=name,
        path=rel,
        role=role,
        format=fmt,
        schema=schema,
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
    )
    manifest_path.write_text(
        manifest.model_copy(update={"artifacts": [*manifest.artifacts, artifact]}).model_dump_json(by_alias=True)
    )
    register_manifests(root, [manifest_rel], INTERNAL_REPO)
    build_catalog(root)
    return rel


@pytest.mark.parametrize(
    "name,role,schema",
    [
        ("collection_meta.yaml", "collection-meta", "aisim-collection-meta/v1"),
        ("reuse.yaml", "reuse-declaration", "aisim-reuse-declaration/v1"),
    ],
)
@pytest.mark.parametrize(
    "payload,code",
    [
        (b'authorization: "Bearer aaaaaaaaaaaaaaaaaaaaaaaa"\n', "secret"),
        (b'endpoint: "https://node.corp.nvidia.com"\n', "internal-host"),
        (b'authorization: "\\u0068f_aaaaaaaaaaaaaaaaaaaaaaaa"\n', "secret"),
        (b"bad: [\n", "yaml-invalid"),
        (b'!!python/object/apply:os.system ["true"]\n', "yaml-invalid"),
    ],
)
def test_yaml_sidecars_are_content_checked_before_promotion(tmp_path, name, role, schema, payload, code):
    root, directory, _ = _op_mirror(tmp_path, "kernel_source", "clean")
    rel = _manifest_sidecar(root, f"{directory}/manifest.json", name, payload, "yaml", role, schema)
    plan = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert not plan.content_ok
    assert any(f.code.removeprefix("source-") == code and f.path == rel for f in plan.findings)
    assert code in {f.code for f in validate_tree(root, profile="public").errors}


@pytest.mark.parametrize("dirty", [False, True])
def test_yaml_aliases_remain_bounded_and_keep_field_context(tmp_path, dirty):
    root, directory, _ = _op_mirror(tmp_path, "kernel_source", "clean")
    payload = "abi: &a0 {name: clean}\n" + "".join(f"a{i}: &a{i} [*a{i - 1}, *a{i - 1}]\n" for i in range(1, 25))
    if dirty:
        payload += "images: *a0\n"
    _manifest_sidecar(
        root,
        f"{directory}/manifest.json",
        "collection_meta.yaml",
        payload.encode(),
        "yaml",
        "collection-meta",
        "aisim-collection-meta/v1",
    )
    shutil.rmtree(root / "mirror")
    report = validate_tree(root, profile="public")
    assert ("image-name" in {f.code for f in report.errors}) == dirty
    assert report.ok == (not dirty)


@pytest.mark.parametrize("field", ["effective_settings", "unresolved_identity"])
@pytest.mark.parametrize(
    "payload,code",
    [
        ({"endpoint": "https://node.corp.nvidia.com"}, "internal-host"),
        ({"authorization": "Bearer " + "a" * 24}, "secret"),
    ],
)
def test_entire_manifest_identity_is_scanned(tmp_path, field, payload, code):
    root, directory, _ = _op_mirror(tmp_path, "kernel_source", "clean")
    rel = f"{directory}/manifest.json"
    path = root / rel
    manifest = Manifest.model_validate_json(path.read_bytes())
    identity = manifest.identity.model_copy(update={field: payload})
    manifest = manifest.model_copy(
        update={
            "identity": identity,
            "configuration_id": configuration_id(manifest.category, manifest.purpose, identity),
        }
    )
    path.write_text(manifest.model_dump_json(by_alias=True))
    # Rebuild the indexes without the old configuration ID.
    shutil.rmtree(root / "catalog")
    register_manifests(root, [rel], INTERNAL_REPO)
    build_catalog(root)
    assert validate_tree(root).ok
    assert code in {f.code for f in validate_tree(root, profile="public").errors}
    assert code in {f.code for f in plan_promotion(root, _request(root, [directory]), _reviewers()).findings}


def test_manifest_artifact_ids_are_scanned(tmp_path):
    root, manifests = _mirror(tmp_path)
    path = root / manifests[0]
    manifest = Manifest.model_validate_json(path.read_bytes())
    artifacts = [
        a.model_copy(update={"artifact_id": "hf_" + "a" * 24}) if i == 0 else a
        for i, a in enumerate(manifest.artifacts)
    ]
    path.write_text(manifest.model_copy(update={"artifacts": artifacts}).model_dump_json(by_alias=True))
    register_manifests(root, manifests, INTERNAL_REPO)
    build_catalog(root)
    directory = path.parent.relative_to(root).as_posix()
    _public(root, directory)
    assert validate_tree(root).ok
    assert "secret" in {f.code for f in validate_tree(root, profile="public").errors}
    assert "secret" in {f.code for f in plan_promotion(root, _request(root, [directory]), _reviewers()).findings}


@pytest.mark.parametrize("dirty", [False, True])
def test_whitespace_json_table_cells_are_decoded(tmp_path, dirty):
    value = "hf_" + "a" * 24 if dirty else "clean"
    encoded = " \t\r\n" + json.dumps({"authorization": value}).replace("hf_", "\\u0068f_")
    root, directory, _ = _op_mirror(tmp_path, "attrs", encoded)
    report = validate_tree(root, profile="public")
    plan = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert ("secret" in {f.code for f in report.errors}) == dirty
    assert plan.content_ok == (not dirty)


@pytest.mark.parametrize("name", ["errors.json", "errors.JSON", "nested/errors.json"])
@pytest.mark.parametrize("dirty", [False, True])
def test_json_truth_sidecars_validate_and_are_scanned(tmp_path, name, dirty):
    root, manifests = _mirror(tmp_path)
    payload = json.dumps({"authorization": "hf_" + "a" * 24 if dirty else "clean"}).encode()
    rel = _extra_truth_artifact(root, manifests, name, payload, "json")
    report = validate_tree(root)
    assert report.ok, report.errors
    public = validate_tree(root, profile="public")
    assert ("secret" in {f.code for f in public.errors}) == dirty
    assert public.ok == (not dirty)
    directory = str(Path(manifests[0]).parent)
    eval_dir = str(Path(rel).parent)
    _public(root, directory, eval_dir)
    draft = plan_promotion(root, _request(root, [directory, eval_dir]), _reviewers())
    assert ("secret" in {f.code for f in draft.findings}) == dirty
    assert draft.content_ok == (not dirty), draft.findings
    if not dirty:
        approved = plan_promotion(root, _request(root, [directory, eval_dir], draft), _reviewers())
        target = _target(tmp_path)
        apply_promotion(approved, root, target)
        assert validate_tree(target, repo=PUBLIC, profile="public").ok


def test_malformed_eval_record_is_still_rejected(tmp_path):
    root, manifests = _mirror(tmp_path)
    cid = load_e2e(root, manifests[0]).manifest.configuration_id
    record_path = _eval(root, [cid])
    build_catalog(root)
    bad = record_path.parent / ("a" * 64 + ".json")
    bad.write_text('{"schema_version":"aisim-eval/v1"}')
    assert "eval-record-invalid" in {f.code for f in validate_tree(root).errors}


def test_unsupported_truth_artifact_format_blocks_publication(tmp_path):
    root, manifests = _mirror(tmp_path)
    rel = _extra_truth_artifact(root, manifests, "opaque.JSON", b"{}", "opaque")
    assert validate_tree(root).ok
    assert "artifact-format" in {f.code for f in validate_tree(root, profile="public").errors}
    directory = str(Path(manifests[0]).parent)
    eval_dir = str(Path(rel).parent)
    _public(root, directory, eval_dir)
    plan = plan_promotion(root, _request(root, [directory, eval_dir]), _reviewers())
    assert "artifact-format" in {f.code for f in plan.findings}


@pytest.mark.parametrize(
    "schema,column",
    [
        ("aisim-e2e-snapshot-members/v1", "observation_id"),
        ("aisim-e2e-snapshot-acquisition/v1", "source_uri"),
        ("aisim-eval-truth/v1", "inputs_json"),
        ("synthetic-extension/v1", "attrs"),
    ],
)
@pytest.mark.parametrize("dirty", [False, True])
def test_every_manifest_parquet_artifact_is_scanned(tmp_path, schema, column, dirty):
    root, directory, _ = _op_mirror(tmp_path, "kernel_source", "clean")
    rel = f"{directory}/extension.parquet"
    value = "Bearer " + "a" * 24 if dirty else "synthetic"
    # Snapshot IDs/URIs and eval/extension JSON columns can all carry free text.
    table = pa.table({column: [json.dumps({"authorization": value}) if column in ("inputs_json", "attrs") else value]})
    artifact_path = write_parquet(table, root / rel, schema)
    manifest_rel = f"{directory}/manifest.json"
    manifest_path = root / manifest_rel
    manifest = Manifest.model_validate_json(manifest_path.read_bytes())
    artifact = Artifact(
        artifact_id="extension",
        path=rel,
        role="metadata",
        format="parquet",
        schema=schema,
        sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        bytes=artifact_path.stat().st_size,
        rows=1,
    )
    manifest_path.write_text(
        manifest.model_copy(update={"artifacts": [*manifest.artifacts, artifact]}).model_dump_json(by_alias=True)
    )
    register_manifests(root, [manifest_rel], INTERNAL_REPO)
    build_catalog(root)
    internal = validate_tree(root)
    assert internal.ok, internal.errors
    public_root = tmp_path / "public"
    shutil.copytree(root, public_root)
    shutil.rmtree(public_root / "mirror")
    public = validate_tree(public_root, profile="public")
    plan = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert any(f.code == "secret" and f.path == rel for f in public.errors) == dirty
    assert any(f.code == "secret" and f.path == rel for f in plan.findings) == dirty
    assert public.ok == plan.content_ok == (not dirty)


@pytest.mark.parametrize("kind", ["manifest", "eval"])
def test_deep_json_sidecars_report_findings_without_aborting(tmp_path, kind):
    depth = sys.getrecursionlimit() + 100
    payload = b"[" * depth + b"0" + b"]" * depth
    if kind == "eval":
        root, manifests = _mirror(tmp_path)
        record_rel = _extra_truth_artifact(root, manifests, "deep.json", payload, "json")
        document_rel = record_rel
        directories = [str(Path(manifests[0]).parent), str(Path(record_rel).parent)]
        _public(root, *directories)
    else:
        root, directory, _ = _op_mirror(tmp_path, "kernel_source", "clean")
        directories = [directory]
        rel = f"{directory}/deep.json"
        (root / rel).write_bytes(payload)
        manifest_rel = f"{directory}/manifest.json"
        document_rel = manifest_rel
        path = root / manifest_rel
        manifest = Manifest.model_validate_json(path.read_bytes())
        artifact = Artifact(
            artifact_id="deep",
            path=rel,
            role="metadata",
            format="json",
            schema="synthetic/v1",
            sha256=hashlib.sha256(payload).hexdigest(),
            bytes=len(payload),
        )
        path.write_text(
            manifest.model_copy(update={"artifacts": [*manifest.artifacts, artifact]}).model_dump_json(by_alias=True)
        )
        register_manifests(root, [manifest_rel], INTERNAL_REPO)
        build_catalog(root)
    output = tmp_path / "report.json"
    public_root = tmp_path / "public"
    shutil.copytree(root, public_root)
    shutil.rmtree(public_root / "mirror")
    assert mirror_main(["validate", "--root", str(public_root), "--profile", "public", "--output", str(output)]) == 1
    report = json.loads(output.read_bytes())
    assert not report["ok"]
    # JSON decoders differ in their nesting limit; the document scan must always
    # turn either decoding or walking recursion failures into a finding.
    assert any(f["code"] == "document-invalid" and f["path"] == document_rel for f in report["findings"])
    plan = plan_promotion(root, _request(root, directories), _reviewers())
    assert not plan.content_ok and "document-invalid" in {f.code for f in plan.findings}


@pytest.mark.parametrize("failure", ["encoding", "io"])
def test_unreadable_schema_snapshot_writes_cli_report(tmp_path, monkeypatch, failure):
    root, _ = _mirror(tmp_path)
    path = write_snapshots(root / "catalog/schemas")[0]
    if failure == "encoding":
        path.write_bytes(b"\xff\xfe")
    else:
        read_text = Path.read_text
        read_bytes = Path.read_bytes

        def denied(self, *a, **kw):
            if self == path:
                raise OSError("synthetic read failure")
            return read_text(self, *a, **kw)

        monkeypatch.setattr(Path, "read_text", denied)

        def denied_bytes(self):
            if self == path:
                raise OSError("synthetic read failure")
            return read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", denied_bytes)
    report_path = tmp_path / "report.json"
    assert mirror_main(["validate", "--root", str(root), "--output", str(report_path)]) == 1
    report = json.loads(report_path.read_text())
    assert not report["ok"] and "schema-unreadable" in {f["code"] for f in report["findings"]}


@pytest.mark.parametrize(
    "reviewers,threshold",
    [
        (["alice"], 1.9),
        ("alice", 1),
        ([1], 1),
        (["alice"], True),
        (["alice"], "2"),
        (["alice"], 0),
        (["alice"], -1),
        (None, 1),
        ([""], 1),
    ],
)
def test_policy_rejects_malformed_reviewers_and_thresholds(tmp_path, reviewers, threshold):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"reviewers": reviewers, "min_approvals": threshold}))
    with pytest.raises(ValueError):
        PromotionPolicy.from_file(path)


def test_policy_preserves_valid_threshold_and_default(tmp_path):
    path = tmp_path / "policy.json"
    for options, expected in (({}, 1), ({"min_approvals": 2}, 2)):
        path.write_text(json.dumps({"reviewers": ["alice", "bob"], **options}))
        policy = PromotionPolicy.from_file(path)
        assert policy.reviewers == frozenset({"alice", "bob"}) and policy.min_approvals == expected


# Regression cases for the scoped publication and transaction findings.
def _refresh_artifact(root, directory, relative, **updates):
    path = root / directory / "manifest.json"
    manifest = Manifest.model_validate_json(path.read_bytes())
    artifacts = [
        a.model_copy(
            update={
                "sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest(),
                "bytes": (root / relative).stat().st_size,
                **updates,
            }
        )
        if a.path == relative
        else a
        for a in manifest.artifacts
    ]
    path.write_text(manifest.model_copy(update={"artifacts": artifacts}).model_dump_json(by_alias=True))
    register_manifests(root, [str(path.relative_to(root))], INTERNAL_REPO)
    build_catalog(root)


@pytest.mark.parametrize("location", ["file", "field", "binary"])
@pytest.mark.parametrize("value,blocked", [("https://node7.corp.nvidia.com/v1", True), ("public-benchmark", False)])
def test_parquet_metadata_and_binary_values_reach_both_publication_gates(tmp_path, location, value, blocked):
    import pyarrow.parquet as pq

    root, directory, relative = _op_mirror(tmp_path, "future_text", None)
    table = pq.read_table(root / relative)
    metadata = None
    if location == "binary":
        table = table.append_column("diagnostic", pa.array([None, value.encode()], type=pa.binary()))
    elif location == "field":
        table = table.set_column(0, table.schema.field(0).with_metadata({b"origin": value.encode()}), table.column(0))
    else:
        metadata = {"origin": value}
    write_parquet(table, root / relative, OP_TABLE_SCHEMA_ID, extra_metadata=metadata)
    _refresh_artifact(root, directory, relative)
    assert validate_tree(root).ok
    plan = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert ("internal-host" in {f.code for f in plan.findings}) is blocked
    public = tmp_path / "public"
    shutil.copytree(root, public, ignore=shutil.ignore_patterns(".git", "mirror"))
    report = validate_tree(public, profile="public")
    assert ("internal-host" in {f.code for f in report.errors}) is blocked
    assert report.ok is not blocked


@pytest.mark.parametrize(
    "declared_format,name", [("opaque", "errors.json"), ("json", "errors.json"), ("yaml", "errors.json")]
)
def test_manifest_sidecars_must_have_supported_consistent_formats(tmp_path, declared_format, name):
    root, directory, _ = _op_mirror(tmp_path, "future_text", None)
    relative = f"{directory}/{name}"
    (root / relative).write_text('{"endpoint":"https://node7.corp.nvidia.com/v1"}')
    path = root / directory / "manifest.json"
    manifest = Manifest.model_validate_json(path.read_bytes())
    artifact = Artifact(
        artifact_id="errors",
        path=relative,
        role="metadata",
        format=declared_format,
        schema="extension/v1",
        sha256=hashlib.sha256((root / relative).read_bytes()).hexdigest(),
        bytes=(root / relative).stat().st_size,
    )
    path.write_text(
        manifest.model_copy(update={"artifacts": [*manifest.artifacts, artifact]}).model_dump_json(by_alias=True)
    )
    register_manifests(root, [str(path.relative_to(root))], INTERNAL_REPO)
    build_catalog(root)
    for findings in (
        plan_promotion(root, _request(root, [directory]), _reviewers()).findings,
        validate_tree(root, profile="public").errors,
    ):
        expected = "internal-host" if declared_format == "json" else "artifact-format"
        assert any(f.code == expected and f.path == relative for f in findings)


def test_unresolved_eval_configuration_is_not_public(tmp_path):
    root, _ = _mirror(tmp_path)
    record = _eval(root, ["cfg-" + "f" * 64])
    build_catalog(root)
    directory = str(record.parent.relative_to(root))
    _public(root, directory)
    assert validate_tree(root).ok
    assert "eval-configuration-unresolved" in {f.code for f in validate_tree(root, profile="public").errors}
    plan = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert {"eval-truth-not-selected", "eval-configuration-unresolved"} <= {f.code for f in plan.findings}


@pytest.mark.parametrize(
    "row,valid",
    [
        ({"n_scored": 5, "n_total": 1}, False),
        ({"n_scored": 5}, True),
        ({"n_scored": 5, "n_total": 5, "coverage": 1.0, "residual_quantiles_pct": {"p50": -2.5}}, True),
        ({"n_scored": 5, "coverage": 10**400}, False),
        ({"n_scored": 5, "residual_quantiles_pct": {"p50": 10**400}}, False),
    ],
)
def test_aggregate_counts_and_large_numbers(row, valid):
    assert (not check_disclosure([row])) is valid


def test_unknown_python_validation_profile_is_refused(tmp_path):
    root, _ = _mirror(tmp_path)
    (root / "README.md").write_text("https://node7.corp.nvidia.com")
    with pytest.raises(ValueError, match="unsupported validation profile"):
        validate_tree(root, profile="publc")


def test_operation_evidence_version_directory_and_strict_identity(tmp_path):
    from pydantic import ValidationError

    from aisim_core.data.evidence import EvidenceManifest, check_evidence

    root, directory, _ = _op_mirror(tmp_path, "future_text", None)
    manifest = Manifest.model_validate_json((root / directory / "manifest.json").read_bytes())
    destination = root / "evidence" / directory
    destination.mkdir(parents=True)
    (destination / "capture.log").write_bytes(b"fixture")
    doc = {
        "configuration_id": manifest.configuration_id,
        "category": "op-latency",
        "config_path": directory,
        "artifacts": [
            {"path": "capture.log", "sha256": hashlib.sha256(b"fixture").hexdigest(), "size": 7, "kind": "log"}
        ],
    }
    (destination / "evidence-manifest.json").write_text(json.dumps(doc))
    assert validate_tree(root).ok
    wrong = {**doc, "configuration_id": "cfg-" + "d" * 64}
    (destination / "evidence-manifest.json").write_text(json.dumps(wrong))
    assert "evidence-configuration" in {f.code for f in validate_tree(root).errors}
    # Evidence survives removal of the public configuration from the private mirror.
    evidence_paths = [p.relative_to(root).as_posix() for p in destination.iterdir()]
    assert check_evidence(root, evidence_paths, manifests={})[0] == []
    (destination / "evidence-manifest.json").write_text(json.dumps(doc))
    for bad in ["anything", "cfg-" + "A" * 64]:
        with pytest.raises(ValidationError):
            EvidenceManifest.model_validate({**doc, "configuration_id": bad})
    e2e = _evidence(root)
    wrong = json.loads((e2e / "evidence-manifest.json").read_text())
    wrong["configuration_id"] = "cfg-" + "b" * 64
    (e2e / "evidence-manifest.json").write_text(json.dumps(wrong))
    assert "evidence-configuration" in {
        f.code for f in check_evidence(root, [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()])[0]
    }


@pytest.mark.parametrize(
    "kind,code",
    [("metadata", "metadata-unreadable"), ("evidence", "evidence-unreadable"), ("catalog", "catalog-unreadable")],
)
def test_unreadable_files_produce_cli_reports(tmp_path, monkeypatch, kind, code):
    from aisim_core.data import mirror

    root, _ = _mirror(tmp_path)
    if kind == "metadata":
        bad = root / "README.md"
        bad.write_text("dataset card")
        original = Path.read_bytes

        def read(path):
            if path == bad:
                raise PermissionError("synthetic unreadable file")
            return original(path)

        monkeypatch.setattr(Path, "read_bytes", read)
    elif kind == "evidence":
        bad = _evidence(root) / "capture.sh"
        original = Path.open

        def open_file(path, *args, **kw):
            if path == bad:
                raise PermissionError("synthetic unreadable file")
            return original(path, *args, **kw)

        monkeypatch.setattr(Path, "open", open_file)
    else:
        bad = root / "catalog/systems.parquet"
        original = mirror.pq.read_table

        def read_table(path, *args, **kw):
            if Path(path) == bad:
                raise PermissionError("synthetic unreadable file")
            return original(path, *args, **kw)

        monkeypatch.setattr(mirror.pq, "read_table", read_table)
    output = tmp_path / "report.json"
    assert mirror_main(["validate", "--root", str(root), "--output", str(output)]) == 1
    assert any(
        f["code"] == code and f["path"] == str(bad.relative_to(root))
        for f in json.loads(output.read_text())["findings"]
    )


def test_declared_snapshot_named_sidecars_are_not_snapshot_documents(tmp_path):
    root, directory, _ = _op_mirror(tmp_path, "future_text", None)
    relative = f"{directory}/snapshot-run/snapshot-settings.json"
    path = root / relative
    path.parent.mkdir()
    path.write_text('{"mode":"public"}')
    manifest_path = root / directory / "manifest.json"
    manifest = Manifest.model_validate_json(manifest_path.read_bytes())
    artifact = Artifact(
        artifact_id="settings",
        path=relative,
        role="metadata",
        format="json",
        schema="settings/v1",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        bytes=path.stat().st_size,
    )
    manifest_path.write_text(
        manifest.model_copy(update={"artifacts": [*manifest.artifacts, artifact]}).model_dump_json(by_alias=True)
    )
    register_manifests(root, [str(manifest_path.relative_to(root))], INTERNAL_REPO)
    build_catalog(root)
    public = tmp_path / "public"
    shutil.copytree(root, public, ignore=shutil.ignore_patterns(".git", "mirror"))
    assert validate_tree(public, profile="public").ok
    assert plan_promotion(root, _request(root, [directory]), _reviewers()).content_ok


def test_system_publisher_bundle_is_admitted_narrowly(tmp_path):
    from aisim_core.data.systems import PACKAGED_SYSTEMS_DIR, publish_system_spec

    root, _ = _mirror(tmp_path)
    publish_system_spec(PACKAGED_SYSTEMS_DIR / "h100_sxm.yaml", root)
    for profile in ["mirror", "public"]:
        report = validate_tree(root, profile=profile)
        assert report.ok, report.errors
    (root / "systems/h100_sxm/extra.yaml").write_text("unexpected: true")
    assert "yaml-sidecar" in {f.code for f in validate_tree(root).errors}


@pytest.mark.parametrize("change", ["writer", "row", "schema", "missing", "stale"])
def test_catalog_drift_compares_ordered_data_and_schema(tmp_path, change):
    import pyarrow.parquet as pq

    from aisim_core.data.mirror import TreeReport, _check_catalog_drift

    root, _ = _mirror(tmp_path)
    shard = root / "catalog/systems.parquet"
    table = pq.read_table(shard)
    if change == "writer":
        data = shard.read_bytes()
        created_by = pq.ParquetFile(shard).metadata.created_by.encode()
        assert created_by in data
        shard.write_bytes(data.replace(created_by, b"x" * len(created_by)))
        assert pq.ParquetFile(shard).metadata.created_by != created_by.decode()
    elif change == "row":
        rows = table.to_pylist()
        rows[0]["spec_sha256"] = "f" * 64
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), shard)
    elif change == "schema":
        pq.write_table(table.append_column("unexpected", pa.array([1] * table.num_rows)), shard)
    elif change == "missing":
        shard.unlink()
    else:
        stale = root / "catalog/e2e/stale.parquet"
        stale.parent.mkdir(exist_ok=True)
        stale.write_bytes(shard.read_bytes())
    report = TreeReport(None, "mirror")
    _check_catalog_drift(root, report, set())
    assert bool(report.errors) is (change != "writer"), report.errors


@pytest.mark.parametrize("location", ["producer", "schema-metadata", "field-metadata"])
@pytest.mark.parametrize("dirty", [False, True])
def test_catalog_footer_disclosure_is_independent_of_logical_drift(tmp_path, location, dirty):
    import pyarrow.parquet as pq

    root, _ = _mirror(tmp_path, repo=PUBLIC)
    shard = root / "catalog/systems.parquet"
    table = pq.read_table(shard)
    value = b"node7.corp.nvidia.com" if dirty else b"public-writer"
    if location == "producer":
        created_by = pq.ParquetFile(shard).metadata.created_by.encode()
        replacement = value.ljust(len(created_by), b" ")
        assert len(replacement) == len(created_by)
        shard.write_bytes(shard.read_bytes().replace(created_by, replacement))
    elif location == "schema-metadata":
        pq.write_table(table.replace_schema_metadata({**(table.schema.metadata or {}), b"origin": value}), shard)
    else:
        field = table.schema.field(0).with_metadata({b"origin": value})
        pq.write_table(table.set_column(0, field, table.column(0)), shard)
    mirror = validate_tree(root)
    assert mirror.ok is (location == "producer")
    assert any(f.code == "catalog-drift" for f in mirror.errors) is (location != "producer")
    rel = shard.relative_to(root).as_posix()
    findings = check_publishable(root, [rel])
    assert any(f.code == "internal-host" and f.path == rel for f in findings) is dirty
    report = validate_tree(root, profile="public")
    assert report.ok is (location == "producer" and not dirty)
    assert any(f.code == "internal-host" and f.path == rel for f in report.errors) is dirty


def test_snapshot_discovery_bounds_sidecar_reads_and_keeps_custom_filenames(tmp_path, monkeypatch):
    import io

    from aisim_core.data.hygiene import CandidatePolicy
    from aisim_core.data.mirror import snapshot_documents

    root, directories = _snapshot_mirror(tmp_path)
    snapshot = next((root / directories[1]).glob("snapshot.json"))
    custom = snapshot.with_name("frozen-view.json")
    snapshot.rename(custom)
    limit = CandidatePolicy().max_json_bytes
    oversized = root / "data/large-sidecar.json"
    payload = b'{"padding":"' + b"x" * limit + b'"}'
    oversized.write_bytes(payload)
    original = Path.open
    reads = []

    class Bounded(io.BytesIO):
        def read(self, size=-1):
            assert 0 <= size <= limit + 1
            reads.append(size)
            return super().read(size)

    def opened(path, *args, **kwargs):
        if path == oversized:
            return Bounded(payload)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    paths = [p.relative_to(root).as_posix() for p in (custom, oversized)]
    assert snapshot_documents(root, paths) == {paths[0]}
    report = validate_tree(root)
    assert any(f.code == "json-too-large" and f.path == paths[1] for f in report.errors)
    assert not any(f.code == "snapshot-invalid" for f in report.errors)
    assert reads and max(reads) == limit + 1


def test_promotion_lock_covers_snapshot_and_rollback(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from aisim_core.data import promote as module

    root, _, plan = _approved(tmp_path)
    target = _target(tmp_path)
    original = module._snapshot
    entered, release = threading.Event(), threading.Event()
    snapshots = []

    def pause(path, index_paths):
        result = original(path, index_paths)
        snapshots.append(result)
        entered.set()
        assert release.wait(10)
        return result

    monkeypatch.setattr(module, "_snapshot", pause)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(apply_promotion, plan, root, target, write_ledger=False)
        try:
            assert entered.wait(10)
            with pytest.raises(PromotionError, match="target lock"):
                apply_promotion(plan, root, target, write_ledger=False)
            assert len(snapshots) == 1  # competing writer cannot capture a stale rollback snapshot
        finally:
            release.set()
        first.result()
    before = {str(p.relative_to(target)): p.read_bytes() for p in target.rglob("*") if p.is_file()}

    def fail(*args, **kw):
        raise OSError("second transaction failed")

    monkeypatch.setattr(module, "build_catalog", fail)
    with pytest.raises(OSError, match="second transaction failed"):
        apply_promotion(plan, root, target, write_ledger=False)
    assert before == {str(p.relative_to(target)): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    assert validate_tree(target, profile="public").ok


@pytest.mark.parametrize("rollback", [False, True])
def test_promotion_preserves_noncanonical_category_index(tmp_path, monkeypatch, rollback):
    from aisim_core.data import DatasetIndex

    root, _, plan = _approved(tmp_path)
    container = tmp_path / "prior"
    container.mkdir()
    target, manifests = _mirror(container, repo=PUBLIC, dev=("synthetic/prior-model",))
    old = target / "catalog/e2e.json"
    custom = target / "data/indexes/existing.json"
    custom.parent.mkdir()
    old.rename(custom)
    path = target / "catalog/dataset.json"
    dataset = DatasetIndex.model_validate_json(path.read_bytes())
    ref = dataset.categories["e2e"]
    artifact = ref.index.model_copy(update={"path": "data/indexes/existing.json"})
    path.write_text(
        dataset.model_copy(update={"categories": {"e2e": ref.model_copy(update={"index": artifact})}}).model_dump_json(
            by_alias=True
        )
    )
    assert validate_tree(target).ok
    if rollback:
        from aisim_core.data import promote as module
        from aisim_core.data.mirror import TreeReport

        before = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}

        def reject(*args, **kwargs):
            assert custom.read_bytes() != before[custom.relative_to(target)]  # rewrite already happened
            report = TreeReport(None, "public")
            report.add("fixture-rejected", "exercise rollback after category index registration")
            return report

        monkeypatch.setattr(module, "validate_tree", reject)
        with pytest.raises(PromotionError, match="fixture-rejected"):
            apply_promotion(plan, root, target, write_ledger=False)
        assert {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()} == before
        assert validate_tree(target, profile="public").ok
        return
    apply_promotion(plan, root, target, write_ledger=False)
    updated = DatasetIndex.model_validate_json(path.read_bytes())
    assert updated.categories["e2e"].index.path == "data/indexes/existing.json"
    assert validate_tree(target, profile="public").configurations == 2
    assert (target / manifests[0]).is_file()
