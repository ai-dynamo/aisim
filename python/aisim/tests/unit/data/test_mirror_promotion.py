# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mirror validator, redaction rules and promotion procedure, on synthetic data only."""

import dataclasses
import hashlib
import json
import shutil
import subprocess
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
def test_repository_workflow_symlinks_are_rejected(tmp_path, profile):
    root, _ = _mirror(tmp_path)
    workflow = root / ".github/workflows/validate.yml"
    workflow.parent.mkdir(parents=True)
    workflow.symlink_to(tmp_path / "missing.yml")
    report = validate_tree(root, profile=profile)
    assert not report.ok
    assert any(f.code == "symlink" and f.path == ".github/workflows/validate.yml" for f in report.errors)


@pytest.mark.parametrize("profile", ["mirror", "public"])
@pytest.mark.parametrize("path", [".github/workflows/other.yml", ".github/raw.json", "data/validate.yml"])
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


def test_changed_bytes_invalidate_approval_and_apply(tmp_path):
    root, directory = _promotable(tmp_path)
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    (root / directory / "metrics.parquet").write_bytes(b"changed")
    with pytest.raises(PromotionError, match="source changed"):
        apply_promotion(plan, root, _target(tmp_path))


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


def test_snapshot_never_spans_private_and_public(tmp_path):
    from aisim_core.data.e2e import SnapshotMember, write_snapshot

    root, manifests = _mirror(tmp_path, dev=("synthetic/model-a", "synthetic/model-b"))
    public_dir, private_dir = (str(Path(m).parent) for m in manifests)
    from aisim_core.data import Manifest

    members = [
        SnapshotMember(
            configuration_id=Manifest.model_validate_json((root / m).read_bytes()).configuration_id,
            observation_id=f"o{i}", record_sha256="b" * 64, manifest_path=m,
        )
        for i, m in enumerate(manifests)
    ]  # fmt: skip
    snap_dir = root / "data" / "e2e-snapshots" / "s1"
    snap_dir.mkdir(parents=True)
    write_snapshot(
        root, "data/e2e-snapshots/s1/snapshot.json", members,
        source_scope={"kind": "synthetic"}, selection_policy="p", completeness="complete",
    )  # fmt: skip
    _public(root, public_dir, "data/e2e-snapshots/s1")
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

    def deny(self, *a, **k):
        if "promotions" in self.parts:
            raise PermissionError("read-only")
        return real_write(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", deny)
    with pytest.raises(PermissionError):
        apply_promotion(plan, root, target)
    assert not any(p.is_file() for p in target.rglob("*"))


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
    original = Path.read_bytes
    reads = []

    def observe(path):
        reads.append(path)
        assert path.resolve() != outside.resolve(), "validator followed a document symlink"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", observe)
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
