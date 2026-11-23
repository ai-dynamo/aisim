# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mirror validator, redaction rules and promotion procedure, on synthetic data only."""

import contextlib
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisim_core.data import Artifact, Identity, Manifest, configuration_id, redaction
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


def _eval(root, configuration_ids, *, case_ids=5, inputs=None, snapshot_id=None):
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
                "snapshot_id": snapshot_id,
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
    columns = {"kernel_source": ["synthetic-kernel"] * 2, "latency_ms": [1.0, 2.0]}
    columns[column] = ["synthetic-kernel" if column == "kernel_source" else None, value]
    return _op_mirror_table(tmp_path, pa.table(columns))


def _op_mirror_table(
    tmp_path, table, *, store_schema=True, row_group_size=None, footer=None, provenance=None, **writer_options
):
    """A valid private op-latency mirror holding exactly ``table`` (any Arrow layout the writer accepts).

    ``store_schema=False`` writes the file as another producer might: the Arrow schema is not embedded, so a
    reader sees plain string columns although the stored dictionary pages keep every value their writer put there.
    ``row_group_size`` splits the rows into row groups, which read back as several chunks with their own dictionaries.
    ``footer`` is ``(old, new)``: the same-length bytes ``new`` replace ``old`` inside the file's footer only, so the
    visible cells and the data pages stay as written while the footer's statistics say something else.
    """
    root = tmp_path / "mirror"
    directory = "data/op-latency/h100_sxm/gemm/vllm/0.11.0"
    relative = f"{directory}/gemm_perf.parquet"
    if store_schema and row_group_size is None and not writer_options:
        path = write_parquet(table, root / relative, OP_TABLE_SCHEMA_ID)
    else:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with pq.ParquetWriter(
            path, table.schema, compression="zstd", store_schema=store_schema, **writer_options
        ) as writer:
            writer.add_key_value_metadata({"aisim.schema": OP_TABLE_SCHEMA_ID})
            writer.write_table(table, row_group_size=row_group_size)
    if footer is not None:
        old, new = footer
        assert len(old) == len(new)
        data = path.read_bytes()
        length = int.from_bytes(data[-8:-4], "little")
        body, tail = data[: -8 - length], data[-8 - length :]
        assert tail.count(old) >= 1  # the statistics hold the value
        path.write_bytes(body + tail.replace(old, new))
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
        provenance=provenance or {},
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


def _git(root, *args):
    local_keys = {
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
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in local_keys and not key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
    }
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, env=env).strip()


def _commit_source(root):
    def git(*args):
        return _git(root, *args)

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
    commit = _git(root, "rev-parse", "HEAD")
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


@pytest.mark.parametrize("dirty", [False, True])
def test_promotion_follows_indexed_noncanonical_manifest_names(tmp_path, dirty):
    root, manifests = _mirror(tmp_path)
    old = root / manifests[0]
    renamed = old.with_name("configuration.json")
    old.rename(renamed)
    rel = renamed.relative_to(root).as_posix()
    if dirty:
        manifest = Manifest.model_validate_json(renamed.read_bytes())
        renamed.write_text(
            manifest.model_copy(update={"provenance": {**manifest.provenance, "password": "withheld"}}).model_dump_json(
                by_alias=True
            )
        )
    register_manifests(root, [rel], INTERNAL_REPO)
    build_catalog(root)
    directory = renamed.parent.relative_to(root).as_posix()
    _public(root, directory)
    assert validate_tree(root).ok
    plan = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert plan.manifests == [rel]
    if dirty:
        assert "secret" in {f.code for f in plan.findings}
    else:
        assert {f.code for f in plan.findings} == {"approval-missing"}
        approved = plan_promotion(root, _request(root, [directory], plan), _reviewers())
        assert approved.ok, approved.findings
        target = tmp_path / "public"
        apply_promotion(approved, root, target)
        assert load_e2e(target, rel).observations == load_e2e(root, rel).observations
        assert validate_tree(target, repo=PUBLIC, profile="public").ok


@pytest.mark.parametrize("origin", ["inferencex", "InferenceX-derived", "self-collected"])
def test_visibility_origin_enforces_inferencex_disclosure(tmp_path, origin):
    root, manifests = _mirror(tmp_path)
    directory = str(Path(manifests[0]).parent)
    _public(root, directory, origin=origin)
    plan = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert ("infx-aggregates-only" in {f.code for f in plan.findings}) is (origin != "self-collected")
    if origin == "self-collected":
        approved = plan_promotion(root, _request(root, [directory], plan), _reviewers())
        assert approved.ok, approved.findings


@pytest.mark.parametrize(
    "value,blocked",
    [
        ('{"note":"host.nvidia.com","note":"clean"}', True),
        ('{"nested":[{"note":"hidden","note":"clean"}]}', True),
        ('{"note":"hidden","note":"clean","number":1e309}', True),
        ('{"note":"clean"}', False),
        ("ordinary text", False),
    ],
)
def test_duplicate_json_in_operation_metadata_reaches_both_publication_gates(tmp_path, value, blocked):
    root, directory, _ = _op_mirror(tmp_path, "attrs", value)
    assert validate_tree(root).ok
    public = validate_tree(root, profile="public")
    plan = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert ("duplicate-json-key" in {f.code for f in public.errors}) is blocked
    assert ("duplicate-json-key" in {f.code for f in plan.findings}) is blocked


def test_encoded_metadata_with_large_numbers_keeps_credential_context(tmp_path):
    root, directory, _ = _op_mirror(tmp_path, "attrs", r'{"\u0070assword":"sensitive-value","number":1e309}')
    assert validate_tree(root).ok
    assert "secret" in {f.code for f in validate_tree(root, profile="public").errors}
    plan = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert "secret" in {f.code for f in plan.findings}


@pytest.mark.parametrize(
    "value",
    [
        "{",
        "[]",
        '{"schema_version":"unknown","entries":[]}',
        '{"entries":[null]}',
        '{"entries":[{"path":"data","label":"internal"}],"entries":[{"path":"data","label":"public"}]}',
        '{"entries":[{"path":"data","label":"internal","label":"public"}]}',
    ],
)
def test_malformed_visibility_is_a_validation_finding_not_a_planning_exception(tmp_path, monkeypatch, value):
    from aisim_core.data import promote as module

    root, manifests = _mirror(tmp_path)
    path = root / "mirror/visibility.json"
    path.parent.mkdir()
    path.write_text(value)
    report = validate_tree(root)
    assert any(f.code == "document-invalid" and f.path == "mirror/visibility.json" for f in report.errors)

    def no_second_parse(*args):
        pytest.fail("planning parsed a ledger already rejected by mirror validation")

    monkeypatch.setattr(module, "load_ledger", no_second_parse)
    plan = plan_promotion(root, _request(root, [str(Path(manifests[0]).parent)]), _reviewers())
    assert any(f.code == "source-document-invalid" and f.path == "mirror/visibility.json" for f in plan.findings)


@pytest.mark.parametrize("location", ["manifest", "copy"])
def test_oversized_system_yaml_is_rejected_before_system_loaders(tmp_path, monkeypatch, location):
    from aisim_core.data import systems
    from aisim_core.data.hygiene import CandidatePolicy

    root, _ = _mirror(tmp_path)
    systems.publish_system_spec(systems.PACKAGED_SYSTEMS_DIR / "h100_sxm.yaml", root)
    path = root / "systems/h100_sxm/manifest.yaml"
    if location == "copy":
        path = next(p for p in path.parent.glob("*.yaml") if p.name != "manifest.yaml")
    path.write_bytes(b" " * (CandidatePolicy().max_json_bytes + 1))

    def must_not_read(*args, **kwargs):
        pytest.fail("system loader read an oversized structured document")

    monkeypatch.setattr(systems, "read_manifest", must_not_read)
    monkeypatch.setattr(systems, "check_published_system", must_not_read)
    report = validate_tree(root)
    assert [(f.code, f.path) for f in report.errors] == [("yaml-too-large", path.relative_to(root).as_posix())]


@pytest.mark.parametrize("profile", ["mirror", "public"])
def test_explicit_catalog_coverage_is_retained_for_rebuild_and_validation(tmp_path, profile):
    from aisim_core.data.catalog import Catalog

    root, _ = _mirror(tmp_path, repo=PUBLIC if profile == "public" else INTERNAL_REPO)
    build_catalog(root, expected_models=["synthetic/missing-model"], expected_frameworks=["sglang"])
    assert Catalog(root).rows(model="synthetic/missing-model", framework="sglang", status="missing")
    before = {str(p.relative_to(root)): p.read_bytes() for p in (root / "catalog").rglob("*") if p.is_file()}
    report = validate_tree(root, profile=profile)
    assert report.ok, report.errors
    build_catalog(root)
    assert before == {str(p.relative_to(root)): p.read_bytes() for p in (root / "catalog").rglob("*") if p.is_file()}
    build_catalog(root, expected_models=[], expected_frameworks=[])
    assert not Catalog(root).rows(model="synthetic/missing-model")
    assert validate_tree(root, profile=profile).ok


@pytest.mark.parametrize(
    "value", ["[]", '{"expected_models":42}', '{"expected_models":[],"expected_models":[]}', "[" * 2000 + "]" * 2000]
)
def test_invalid_catalog_coverage_returns_a_validation_report(tmp_path, value):
    root, _ = _mirror(tmp_path)
    (root / "catalog/coverage.json").write_text(value)
    assert "catalog-unbuildable" in {f.code for f in validate_tree(root).errors}


def test_oversized_catalog_grid_returns_a_cli_validation_report(tmp_path):
    root, _ = _mirror(tmp_path)
    (root / "catalog/coverage.json").write_text(
        json.dumps(
            {
                "schema_version": "aisim-catalog-coverage/v1",
                "expected_models": [f"model{i}" for i in range(1000)],
                "expected_frameworks": [f"framework{i}" for i in range(1000)],
            }
        )
    )
    output = tmp_path / "report.json"
    assert mirror_main(["validate", "--root", str(root), "--output", str(output)]) == 1
    findings = json.loads(output.read_bytes())["findings"]
    assert any(f["code"] == "catalog-unbuildable" and "projected catalog grid" in f["message"] for f in findings)


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
    metrics = next(directory.rglob("metrics.parquet"))  # tables live under an immutable generation directory
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
    """Exit gate: a promoted sample is a copy and passes the public validator."""
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


def test_empty_promotion_source_cannot_bypass_cache_repository_identity(tmp_path):
    root, manifests = _mirror(tmp_path, repo=INFX_CACHE_REPO)
    assert validate_tree(root, repo=INFX_CACHE_REPO).ok
    with pytest.raises(ValueError, match="source_repo"):
        _request(root, [str(Path(manifests[0]).parent)], source_repo="")
    assert "dataset-id-mismatch" in {f.code for f in validate_tree(root, repo="").errors}
    assert _request(root, [str(Path(manifests[0]).parent)]).source_repo == INTERNAL_REPO


def test_promotion_rejects_selected_noncanonical_generated_index_before_writing(tmp_path):
    from aisim_core.data import DatasetIndex

    root, directory = _promotable(tmp_path)
    relative = "data/indexes/e2e.json"
    custom = root / relative
    custom.parent.mkdir()
    custom.write_text(json.dumps(json.loads((root / "catalog/e2e.json").read_bytes()), separators=(",", ":")))
    (root / "catalog/e2e.json").unlink()
    index_path = root / "catalog/dataset.json"
    dataset = DatasetIndex.model_validate_json(index_path.read_bytes())
    ref = dataset.categories["e2e"]
    artifact = ref.index.model_copy(
        update={
            "path": relative,
            "sha256": hashlib.sha256(custom.read_bytes()).hexdigest(),
            "bytes": custom.stat().st_size,
        }
    )
    index_path.write_text(
        dataset.model_copy(update={"categories": {"e2e": ref.model_copy(update={"index": artifact})}}).model_dump_json(
            by_alias=True
        )
    )
    selection = [directory, str(custom.parent.relative_to(root))]
    _public(root, *selection)
    assert validate_tree(root).ok
    target = _target(tmp_path)
    shutil.copytree(root, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git", "mirror"))
    target_index = target / "catalog/dataset.json"
    target_index.write_text(
        DatasetIndex.model_validate_json(target_index.read_bytes())
        .model_copy(update={"dataset_id": PUBLIC})
        .model_dump_json(by_alias=True)
    )
    assert validate_tree(target, profile="public").ok
    draft = plan_promotion(root, _request(root, selection), _reviewers())
    plan = plan_promotion(root, _request(root, selection, draft), _reviewers())
    assert plan.ok, plan.findings
    before = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    with pytest.raises(PromotionError, match="generated indexes overlap approved files"):
        apply_promotion(plan, root, target)
    assert {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("dirty", [False, True])
def test_artifact_only_publication_scans_through_supplied_owner(tmp_path, dirty):
    cfg = _config()
    metrics = list(cfg.metrics)
    metrics[0] = metrics[0].model_copy(
        update={"attrs": json.dumps({"endpoint": "node7.corp.nvidia.com" if dirty else "public"})}
    )
    root = tmp_path / "mirror"
    relative = write_configuration(root, dataclasses.replace(cfg, metrics=metrics), prefix="data")
    configuration = load_e2e(root, relative)
    artifact = next(a for a in configuration.manifest.artifacts if a.path.endswith("/metrics.parquet"))
    findings = check_publishable(root, [artifact.path], manifests={relative: configuration.manifest})
    assert any(f.code == "internal-host" and f.path == artifact.path for f in findings) is dirty
    if not dirty:
        assert findings == []


@pytest.mark.parametrize("dirty", [False, True])
def test_eval_findings_json_scanned_when_provenance_is_allowed(tmp_path, dirty):
    from aisim_core.data.eval import load_record

    root, manifests = _mirror(tmp_path)
    path = _eval(root, [load_e2e(root, manifests[0]).manifest.configuration_id])
    original = load_record(path)
    payload = json.dumps({"endpoint": "node7.corp.nvidia.com" if dirty else "public"}).encode()
    sidecar = path.parent / "findings.json"
    sidecar.write_bytes(payload)
    artifact = Artifact(
        artifact_id="findings",
        path=sidecar.name,
        role="findings",
        format="json",
        schema="findings/v1",
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
    )
    record = original.model_copy(update={"findings": [artifact], "supersedes": original.record_id})
    record_path = write_record(root, record).relative_to(root).as_posix()
    owners = {rel: load_e2e(root, rel).manifest for rel in manifests}
    policy = RedactionPolicy(allow_provenance_artifacts=True)
    findings = check_publishable(root, [record_path], manifests=owners, policy=policy)
    assert any(f.code == "internal-host" and f.path.endswith("/findings.json") for f in findings) is dirty
    if not dirty:
        assert findings == []
    assert "provenance-withheld" in {f.code for f in check_publishable(root, [record_path], manifests=owners)}
    sidecar.write_bytes(b"{}")
    assert "document-invalid" in {
        f.code for f in check_publishable(root, [record_path], manifests=owners, policy=policy)
    }


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
        next((root / directory).rglob("metrics.parquet")).write_bytes(b"changed")

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


@contextlib.contextmanager
def _unbound_membership(root, manifest):
    """Hide a configuration while a snapshot is written, as for external membership.

    The writer binds members whose manifest it can see to the full version of their observation, rebuilding the
    member row. A member it cannot see keeps exactly the fields it was given, including a future column or a
    deliberately wrong pin, which is what these tests exercise.
    """
    visible = root / Path(manifest).parent
    hidden = root / "hidden-configuration"
    visible.rename(hidden)
    try:
        yield
    finally:
        hidden.rename(visible)


def _snapshot_mirror(tmp_path, *, source_scope=None, selection_policy="synthetic", member=None, acquisition=None):
    root, manifests = _mirror(tmp_path)
    manifest = manifests[0]
    observation = load_e2e(root, manifest).observations[0]
    snapshot_dir = "data/e2e-snapshots/s1"
    (root / snapshot_dir).mkdir(parents=True)
    with _unbound_membership(root, manifest) if member else contextlib.nullcontext():
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
    path = next((root / directories[-1]).rglob(filename)).relative_to(root).as_posix()
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
    for path in (p for p in (root / directories[-1]).rglob("*") if p.is_file()):
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
    path = next((root / directory).rglob("metrics.parquet")).relative_to(root).as_posix()
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


@pytest.mark.parametrize("snapshot", ["infx", "self-collected", "unknown"])
def test_eval_snapshot_reference_is_classified_for_infx_disclosure(tmp_path, snapshot):
    from aisim_core.data import Manifest
    from aisim_core.data.e2e import SnapshotMember, write_snapshot

    root = tmp_path / "m"
    root.mkdir()
    (infx,) = inferencex_to_configurations([_infx()])
    infx = dataclasses.replace(
        infx, observations=[o.model_copy(update=CLEAN) for o in infx.observations], provenance=[], requests=[]
    )
    infx_manifest = write_configuration(root, infx, prefix="data")
    own_manifest = write_configuration(root, _config("synthetic/model-a"), prefix="data")
    register_manifests(root, [infx_manifest, own_manifest], INTERNAL_REPO)
    build_catalog(root)
    own_cid = Manifest.model_validate_json((root / own_manifest).read_bytes()).configuration_id
    manifest = {"infx": infx_manifest, "self-collected": own_manifest}.get(snapshot, own_manifest)
    members = [
        SnapshotMember(
            configuration_id=o.configuration_id, observation_id=o.observation_id,
            record_sha256=o.record_sha256, manifest_path=manifest,
        )
        for o in load_e2e(root, manifest).observations
    ]  # fmt: skip
    (root / "data/e2e-snapshots/s1").mkdir(parents=True)
    named = write_snapshot(
        root, "data/e2e-snapshots/s1/snapshot.json", members,
        source_scope={"kind": "synthetic"}, selection_policy="p", completeness="complete",
    ).snapshot_id  # fmt: skip
    # The record lists only the self-collected configuration but names the InferenceX snapshot as its truth input.
    record = _eval(root, [own_cid], snapshot_id="f" * 64 if snapshot == "unknown" else named)
    build_catalog(root)
    # Publish only the self-collected configuration, the snapshot and the record; the InferenceX shard is not selected.
    selected = [str(Path(own_manifest).parent), "data/e2e-snapshots", "eval"]
    files = [p.relative_to(root).as_posix() for d in selected for p in (root / d).rglob("*") if p.is_file()]
    manifests = {m: Manifest.model_validate_json((root / m).read_bytes()) for m in (infx_manifest, own_manifest)}
    codes = {f.code for f in check_publishable(root, files, manifests=manifests)}
    assert ("infx-aggregates-only" in codes) is (snapshot == "infx")
    assert ("eval-configuration-unresolved" in codes) is (snapshot == "unknown")
    _ = record


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
    artifact_path = next((root / directory).rglob(f"{table}.parquet")).relative_to(root).as_posix()
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
        relative = next((root / directory).rglob(f"{table}.parquet")).relative_to(root).as_posix()
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
        # A password in URL userinfo under a neutral field, on a public host and in no known token format.
        ("kernel_source", "https://synthetic-user:plain-synthetic-password@example.org/repo", "secret"),
        ("future_text", "https://synthetic-user:plain-synthetic-password@example.org/repo", "secret"),
        ("future_text", "https://:plain-synthetic-password@example.org/repo", "secret"),
        ("future_text", {"endpoint": "git+ssh://synthetic-user:plain-synthetic-password@example.org/repo"}, "secret"),
        ("future_text", json.dumps({"source_url": "https://u:p@example.org/repo"}), "secret"),
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


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "[REDACTED]",
        '{"source": "public-benchmark"}',
        # Ordinary public URLs, including userinfo without a password and an @ outside the authority.
        "https://example.org/repo",
        "ssh://git@example.org/repo.git",
        "https://example.org/path?next=a:b@c",
    ],
)
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


def _dictionary_table(layout, unused):
    """An op-latency table whose ``diagnostic`` column keeps ``unused`` only as an unreferenced dictionary value."""
    dictionary = pa.array([unused, "public-benchmark", "public-benchmark"]).dictionary_encode()
    if layout == "list":
        column = pa.ListArray.from_arrays(pa.array([0, 1, 2], pa.int32()), dictionary.slice(1))
    elif layout == "struct":
        column = pa.StructArray.from_arrays([dictionary.slice(1)], names=["label"])
    elif layout == "map":
        column = pa.MapArray.from_arrays(pa.array([0, 1, 2], pa.int32()), pa.array(["a", "b"]), dictionary.slice(1))
    else:  # an ordinary filter or slice, as in the review reproduction
        column = dictionary.slice(1)
    table = pa.table({"kernel_source": ["synthetic-kernel"] * 2, "latency_ms": [1.0, 2.0], "diagnostic": column})
    assert unused not in json.dumps(table["diagnostic"].to_pylist())  # no visible row carries it
    return table


DICTIONARY_LAYOUTS = [
    pytest.param("sliced", {}, id="sliced"),
    pytest.param("sliced", {"row_group_size": 1}, id="chunked-row-groups"),
    pytest.param("list", {}, id="list"),
    pytest.param("struct", {}, id="struct"),
    pytest.param("map", {}, id="map"),
    # The embedded Arrow schema is absent: the file reads back as plain strings.
    pytest.param("sliced", {"store_schema": False}, id="plain-strings"),
]


@pytest.mark.parametrize(("layout", "options"), DICTIONARY_LAYOUTS)
@pytest.mark.parametrize(
    ("unused", "code"),
    [
        ("https://node7.corp.nvidia.com/v1", "internal-host"),
        ("hf_" + "a" * 34, "secret"),
    ],
    ids=["internal-url", "hf-token"],
)
def test_unused_dictionary_values_block_promotion_and_public_validation(tmp_path, layout, options, unused, code):
    root, directory, relative = _op_mirror_table(tmp_path, _dictionary_table(layout, unused), **options)
    # The file really stores the unreferenced value, although no visible row carries it.
    schema = pq.ParquetFile(root / relative).schema
    text_columns = [schema.column(i).path for i in range(len(schema)) if schema.column(i).physical_type == "BYTE_ARRAY"]
    stored = pq.ParquetFile(root / relative, read_dictionary=text_columns).read()
    stored_values = [d.to_pylist() for d in redaction._stored_dictionaries(stored["diagnostic"])]
    assert unused in json.dumps(stored_values)
    assert validate_tree(root, repo=INTERNAL_REPO).ok  # valid private data with a normal, readable table
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    assert not plan.content_ok and not plan.ok  # approval cannot override disclosure checks
    found = [f for f in plan.findings if f.code == code and f.path == relative]
    assert found and "column diagnostic (stored dictionary)" in found[0].message
    assert any(f.code == code and f.path == relative for f in validate_tree(root, profile="public").errors)
    target = _target(tmp_path)
    with pytest.raises(PromotionError):
        apply_promotion(plan, root, target)
    assert not (target / relative).exists()


@pytest.mark.parametrize("options", [{}, {"store_schema": False}], ids=["arrow-schema", "plain-strings"])
@pytest.mark.parametrize("field", ["password", "api_key", "authorization"])
def test_an_unused_nested_credential_value_blocks_promotion_and_public_validation(tmp_path, options, field):
    """The field name of a stored dictionary is key context: a plain password trips no token pattern by itself."""
    unused = "synthetic-plain-password-123"
    dictionary = pa.array([unused, "[REDACTED]", "[REDACTED]"]).dictionary_encode().slice(1)
    column = pa.StructArray.from_arrays([dictionary], names=[field])
    table = pa.table({"kernel_source": ["synthetic-kernel"] * 2, "latency_ms": [1.0, 2.0], "diagnostic": column})
    assert unused not in json.dumps(table["diagnostic"].to_pylist())  # both visible rows only carry [REDACTED]
    root, directory, relative = _op_mirror_table(tmp_path, table, **options)
    assert validate_tree(root, repo=INTERNAL_REPO).ok
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    assert not plan.content_ok and not plan.ok
    found = [f for f in plan.findings if f.code == "secret" and f.path == relative]
    assert found and "column diagnostic (stored dictionary)" in found[0].message
    assert any(f.code == "secret" and f.path == relative for f in validate_tree(root, profile="public").errors)
    target = _target(tmp_path)
    with pytest.raises(PromotionError):
        apply_promotion(plan, root, target)
    assert not (target / relative).exists()


@pytest.mark.parametrize("options", [{}, {"store_schema": False}], ids=["arrow-schema", "plain-strings"])
@pytest.mark.parametrize("dictionary_keys", [False, True], ids=["plain-keys", "dictionary-keys"])
def test_an_unused_map_credential_value_blocks_promotion_and_public_validation(tmp_path, options, dictionary_keys):
    """A map item dictionary keeps no key for its unused values, so the keys' names are applied conservatively."""
    unused = "synthetic-plain-password-123"
    items = pa.array([unused, "[REDACTED]", "[REDACTED]"]).dictionary_encode().slice(1)
    keys = pa.array(["password", "password"])
    if dictionary_keys:
        keys = pa.array(["password", "ordinary", "ordinary"]).dictionary_encode().slice(1)  # "password" only stored
    column = pa.MapArray.from_arrays(pa.array([0, 1, 2], pa.int32()), keys, items)
    table = pa.table({"kernel_source": ["synthetic-kernel"] * 2, "latency_ms": [1.0, 2.0], "diagnostic": column})
    assert unused not in json.dumps(table["diagnostic"].to_pylist())  # every visible value is redacted
    root, directory, relative = _op_mirror_table(tmp_path, table, **options)
    assert validate_tree(root, repo=INTERNAL_REPO).ok
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    assert not plan.content_ok and not plan.ok
    found = [f for f in plan.findings if f.code == "secret" and f.path == relative]
    assert found and "column diagnostic (stored dictionary)" in found[0].message
    assert any(f.code == "secret" and f.path == relative for f in validate_tree(root, profile="public").errors)
    target = _target(tmp_path)
    with pytest.raises(PromotionError):
        apply_promotion(plan, root, target)
    assert not (target / relative).exists()


def test_list_view_detection_survives_a_pyarrow_without_the_predicates(monkeypatch):
    """PyArrow 15 has no list-view predicates; a clean scalar column must still be traversed."""
    monkeypatch.delattr(pa.types, "is_list_view", raising=False)
    monkeypatch.delattr(pa.types, "is_large_list_view", raising=False)
    assert redaction._is_list_view(pa.string()) is False
    assert [d.to_pylist() for _, d in redaction._stored_dictionaries_at(pa.array(["a"]))] == []
    stored = pa.array(["a", "b"]).dictionary_encode()
    assert [d.to_pylist() for d in redaction._stored_dictionaries(stored)] == [["a", "b"]]


def test_map_key_contexts_cover_stored_dictionary_keys_and_ignore_plain_names():
    keys = pa.array(["password", "ordinary", "ordinary"]).dictionary_encode().slice(1)
    assert redaction._map_key_contexts(keys) == ["password"]  # only stored in the dictionary, still a context
    assert redaction._map_key_contexts(pa.array([b"api_key", b"label"])) == ["api_key"]


def test_stored_dictionaries_keep_their_struct_field_names():
    inner = pa.array(["a", "b"]).dictionary_encode()
    outer = pa.StructArray.from_arrays([pa.StructArray.from_arrays([inner], names=["password"])], names=["auth"])
    seen = [(fields, d.to_pylist()) for fields, d in redaction._stored_dictionaries_at(pa.chunked_array([outer]))]
    assert seen == [(("auth", "password"), ["a", "b"])]


@pytest.mark.parametrize(("layout", "options"), DICTIONARY_LAYOUTS)
def test_clean_dictionary_columns_are_promoted_with_their_unused_values(tmp_path, layout, options):
    root, directory, relative = _op_mirror_table(tmp_path, _dictionary_table(layout, "retired-benchmark"), **options)
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    assert plan.ok, plan.findings
    target = _target(tmp_path)
    apply_promotion(plan, root, target)
    assert (target / relative).read_bytes() == (root / relative).read_bytes()
    report = validate_tree(target, repo=PUBLIC, profile="public")
    assert report.ok, report.errors


def _extension_table(kind, secret):
    """An op-latency table whose ``diagnostic`` column has an Arrow extension type that carries ``secret``."""
    if kind == "tensor-dim-names":
        extension = pa.fixed_shape_tensor(pa.int32(), [2], dim_names=[secret])
        storage = pa.array([[1, 2], [3, 4]], pa.list_(pa.int32(), 2))
    else:
        parameters = ("synthetic", secret) if kind == "opaque-vendor-name" else (secret, "synthetic")
        extension = pa.opaque(pa.string(), *parameters)
        storage = pa.array(["public"] * 2)
    column = pa.ExtensionArray.from_storage(extension, storage)
    if kind == "nested":  # an extension inside a list column
        column = pa.ListArray.from_arrays(pa.array([0, 1, 2], pa.int32()), column)
    return pa.table({"kernel_source": ["synthetic-kernel"] * 2, "latency_ms": [1.0, 2.0], "diagnostic": column})


EXTENSION_KINDS = ["opaque-type-name", "opaque-vendor-name", "tensor-dim-names", "nested"]


def _need_extension_types(kind):
    needed = "fixed_shape_tensor" if kind == "tensor-dim-names" else "opaque"
    if not hasattr(pa, needed):
        pytest.skip(f"this PyArrow has no built-in {needed} extension type")


@pytest.mark.parametrize("kind", EXTENSION_KINDS)
@pytest.mark.parametrize(
    ("secret", "code"),
    [("https://node7.corp.nvidia.com/v1", "internal-host"), ("hf_" + "a" * 34, "secret")],
    ids=["internal-url", "hf-token"],
)
def test_built_in_extension_parameters_block_promotion_and_public_validation(tmp_path, kind, secret, code):
    _need_extension_types(kind)
    root, directory, relative = _op_mirror_table(tmp_path, _extension_table(kind, secret))
    # The reader restores the extension into type attributes; field metadata stays empty, so the token is recoverable
    # from the published file's type alone.
    schema_type = str(pq.read_table(root / relative).schema.field("diagnostic").type)
    assert secret in schema_type
    assert validate_tree(root, repo=INTERNAL_REPO).ok  # valid private data with a readable table
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    assert not plan.content_ok and not plan.ok  # approval cannot override disclosure checks
    found = [f for f in plan.findings if f.code == code and f.path == relative]
    assert found and "Arrow extension type" in found[0].message
    assert any(f.code == code and f.path == relative for f in validate_tree(root, profile="public").errors)
    target = _target(tmp_path)
    with pytest.raises(PromotionError):
        apply_promotion(plan, root, target)
    assert not (target / relative).exists()


@pytest.mark.parametrize("kind", EXTENSION_KINDS)
def test_clean_built_in_extension_types_are_promoted(tmp_path, kind):
    _need_extension_types(kind)
    root, directory, relative = _op_mirror_table(tmp_path, _extension_table(kind, "public-label"))
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    assert plan.ok, plan.findings
    target = _target(tmp_path)
    apply_promotion(plan, root, target)
    assert (target / relative).read_bytes() == (root / relative).read_bytes()
    report = validate_tree(target, repo=PUBLIC, profile="public")
    assert report.ok, report.errors


class _LabeledType(pa.ExtensionType):
    """A Python extension type whose serialized parameters are whatever bytes it was given."""

    def __init__(self, parameters: bytes):
        self._parameters = parameters
        super().__init__(pa.string(), "aisim.test.labeled")

    def __arrow_ext_serialize__(self):
        return self._parameters

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls(serialized)


@pytest.fixture
def labeled_extension():
    pa.register_extension_type(_LabeledType(b""))
    yield
    pa.unregister_extension_type("aisim.test.labeled")


@pytest.mark.parametrize(
    ("parameters", "code"),
    [
        (b'{"label": "https://node7.corp.nvidia.com/v1"}', "internal-host"),
        (b'{"label": "' + b"hf_" + b"a" * 34 + b'"}', "secret"),
        (b"\xff\xfe\x00 not text", "extension-type"),  # parameters that cannot be inspected are refused
    ],
)
def test_serialized_extension_parameters_are_scanned_or_refused(tmp_path, labeled_extension, parameters, code):
    column = pa.ExtensionArray.from_storage(_LabeledType(parameters), pa.array(["public"] * 2))
    table = pa.table({"kernel_source": ["synthetic-kernel"] * 2, "latency_ms": [1.0, 2.0], "diagnostic": column})
    root, directory, relative = _op_mirror_table(tmp_path, table)
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    assert not plan.content_ok and any(f.code == code and f.path == relative for f in plan.findings)
    assert any(f.code == code and f.path == relative for f in validate_tree(root, profile="public").errors)


def test_a_plain_text_extension_parameter_is_accepted(tmp_path, labeled_extension):
    column = pa.ExtensionArray.from_storage(_LabeledType(b'{"label": "public-benchmark"}'), pa.array(["a", "b"]))
    table = pa.table({"kernel_source": ["synthetic-kernel"] * 2, "latency_ms": [1.0, 2.0], "diagnostic": column})
    root, directory, _ = _op_mirror_table(tmp_path, table)
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert plan_promotion(root, _request(root, [directory], draft), _reviewers()).ok


def _statistics_table(secret, visible=None):
    """A table whose visible cells are clean: the largest one is as long as ``secret``, so a footer can swap them."""
    low = ("public-benchmark-" + "a" * len(secret))[: len(secret)]
    high = visible or ("public-benchmark-" + "z" * len(secret))[: len(secret)]
    return pa.table({"kernel_source": ["synthetic-kernel"] * 2, "latency_ms": [1.0, 2.0], "diagnostic": [low, high]})


@pytest.mark.parametrize(
    ("secret", "code"),
    [("https://node7.corp.nvidia.com/v1", "internal-host"), ("hf_" + "a" * 34, "secret")],
    ids=["internal-url", "hf-token"],
)
@pytest.mark.parametrize("bound", ["max", "min"])
def test_footer_statistics_block_promotion_and_public_validation(tmp_path, secret, code, bound):
    table = _statistics_table(secret)
    cells = table["diagnostic"].to_pylist()
    old = cells[1] if bound == "max" else cells[0]
    root, directory, relative = _op_mirror_table(tmp_path, table, footer=(old.encode(), secret.encode()))
    # The visible cells are clean and the dictionary pages hold only them; the footer alone carries the text.
    assert secret not in json.dumps(pq.read_table(root / relative).to_pydict())
    footer_statistics = pq.ParquetFile(root / relative).metadata.row_group(0).column(2).statistics
    assert secret in (footer_statistics.min, footer_statistics.max)
    assert validate_tree(root, repo=INTERNAL_REPO).ok  # valid private data: digests were taken after the edit
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    assert not plan.content_ok and not plan.ok  # approval cannot override disclosure checks
    found = [f for f in plan.findings if f.code == code and f.path == relative]
    assert found and "column diagnostic (Parquet statistics)" in found[0].message
    assert any(f.code == code and f.path == relative for f in validate_tree(root, profile="public").errors)
    target = _target(tmp_path)
    with pytest.raises(PromotionError):
        apply_promotion(plan, root, target)
    assert not (target / relative).exists()


def test_footer_statistics_are_judged_under_their_column_names(tmp_path):
    visible = "[REDACTED]"  # a redacted credential cell passes; the footer must not carry another value
    column = pa.StructArray.from_arrays([pa.array([visible, visible])], names=["password"])
    table = pa.table({"kernel_source": ["synthetic-kernel"] * 2, "latency_ms": [1.0, 2.0], "diagnostic": column})
    clean_root, clean_directory, _ = _op_mirror_table(tmp_path / "clean", table)
    draft = plan_promotion(clean_root, _request(clean_root, [clean_directory]), _reviewers())
    assert plan_promotion(clean_root, _request(clean_root, [clean_directory], draft), _reviewers()).ok
    root, directory, relative = _op_mirror_table(tmp_path / "edited", table, footer=(visible.encode(), b"hunter2-xy"))
    plan = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert any(
        f.code == "secret" and f.path == relative and "diagnostic.password (Parquet statistics)" in f.message
        for f in plan.findings
    )


def test_undecodable_footer_statistics_are_refused(tmp_path):
    table = _statistics_table("public-benchmark-zzzz")
    old = table["diagnostic"].to_pylist()[1].encode()
    root, directory, relative = _op_mirror_table(tmp_path, table, footer=(old, b"\xff" * len(old)))
    plan = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert not plan.content_ok
    assert any(f.code == "parquet-statistics" and f.path == relative for f in plan.findings)
    assert any(f.code == "parquet-statistics" for f in validate_tree(root, profile="public").errors)


def test_a_page_index_over_text_columns_is_refused_and_clean_statistics_are_promoted(tmp_path):
    table = _statistics_table("public-benchmark-zzzz")
    root, directory, relative = _op_mirror_table(tmp_path / "indexed", table, write_page_index=True)
    plan = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert any(f.code == "parquet-page-index" and f.path == relative for f in plan.findings)
    # The same table written normally has clean min/max values in its footer and promotes as a copy.
    root, directory, relative = _op_mirror_table(tmp_path / "plain", table)
    statistics = pq.ParquetFile(root / relative).metadata.row_group(0).column(2).statistics
    assert statistics.has_min_max and statistics.min.startswith("public-benchmark-")
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    assert plan.ok, plan.findings
    target = _target(tmp_path)
    apply_promotion(plan, root, target)
    assert (target / relative).read_bytes() == (root / relative).read_bytes()
    assert validate_tree(target, repo=PUBLIC, profile="public").ok


@pytest.mark.parametrize(
    ("provenance", "expected"),
    [
        ({"image": {"password": 12345}}, {"secret"}),  # a populated numeric credential inside an image configuration
        ({"image": {"password": "hunter2"}}, {"secret", "image-name"}),  # both rules apply to a string
        ({"image": {"password": "[REDACTED]"}}, set()),  # a redacted control stays exempt
        ({"image": ["ghcr.io/acme/benchmark:1"]}, {"image-name"}),  # a plain image list keeps its own rule
    ],
    ids=["numeric-credential", "string-credential", "redacted-control", "plain-image-list"],
)
def test_a_credential_nested_in_an_image_configuration_keeps_its_own_rule(tmp_path, provenance, expected):
    table = pa.table({"kernel_source": ["synthetic-kernel"] * 2, "latency_ms": [1.0, 2.0]})
    root, directory, relative = _op_mirror_table(tmp_path, table, provenance=provenance)
    manifest = f"{directory}/manifest.json"
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    assert {f.code for f in plan.findings if f.path == manifest} == expected
    public = validate_tree(root, profile="public")
    assert {f.code for f in public.errors if f.path == manifest} == expected
    if not expected:
        assert plan.ok
        target = _target(tmp_path)
        apply_promotion(plan, root, target)
        assert validate_tree(target, repo=PUBLIC, profile="public").ok


def test_stored_dictionaries_cover_chunks_and_nested_dictionaries():
    private = pa.array(["unused-a", "kept"]).dictionary_encode()
    other = pa.array(["unused-b", "kept"]).dictionary_encode()
    nested = pa.DictionaryArray.from_arrays(pa.array([0, 0], pa.int8()), pa.array([["unused-c"], ["kept"]]))
    columns = {
        "chunked": pa.chunked_array([private.slice(1), other.slice(1)]),
        "list": pa.ListArray.from_arrays(pa.array([0, 1], pa.int32()), private.slice(1)),
        "fixed": pa.FixedSizeListArray.from_arrays(private.slice(1), 1),
        "struct": pa.StructArray.from_arrays([private.slice(1)], names=["v"]),
        "map": pa.MapArray.from_arrays(pa.array([0, 1], pa.int32()), pa.array(["k"]), private.slice(1)),
        "dictionary of lists": nested,
    }
    seen = {name: [d.to_pylist() for d in redaction._stored_dictionaries(array)] for name, array in columns.items()}
    flat = {name: json.dumps(values) for name, values in seen.items()}
    assert "unused-a" in flat["chunked"] and "unused-b" in flat["chunked"]
    for name in ("list", "fixed", "struct", "map"):
        assert "unused-a" in flat[name], name
    assert "unused-c" in flat["dictionary of lists"]


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


@pytest.mark.parametrize("payload", ["manifest", "artifact"])
def test_dangling_payload_symlink_is_refused_before_any_write(tmp_path, payload):
    root, _, plan = _approved(tmp_path)
    assert plan.ok, plan.findings
    target = _target(tmp_path)
    rel = plan.manifests[0] if payload == "manifest" else next(p for p in plan.files if p not in plan.manifests)
    dest = target / rel
    dest.parent.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    dest.symlink_to(outside)
    before = set(target.rglob("*"))

    with pytest.raises(PromotionError, match="symlink"):
        apply_promotion(plan, root, target)

    assert not outside.exists()
    assert dest.is_symlink() and dest.readlink() == outside
    assert set(target.rglob("*")) == before


def test_dangling_custom_category_index_symlink_is_refused_before_any_write(tmp_path):
    from aisim_core.data import DatasetIndex

    root, _, plan = _approved(tmp_path)
    assert plan.ok, plan.findings
    prior = tmp_path / "prior"
    prior.mkdir()
    target, _ = _mirror(prior, repo=PUBLIC, dev=("synthetic/prior-model",))
    dest = target / "data/indexes/existing.json"
    dest.parent.mkdir()
    (target / "catalog/e2e.json").rename(dest)
    path = target / "catalog/dataset.json"
    dataset = DatasetIndex.model_validate_json(path.read_bytes())
    ref = dataset.categories["e2e"]
    artifact = ref.index.model_copy(update={"path": dest.relative_to(target).as_posix()})
    path.write_text(
        dataset.model_copy(update={"categories": {"e2e": ref.model_copy(update={"index": artifact})}}).model_dump_json(
            by_alias=True
        )
    )
    assert validate_tree(target, profile="public").ok
    dest.unlink()
    outside = tmp_path / "outside-index.json"
    dest.symlink_to(outside)
    before_paths = set(target.rglob("*"))
    before_bytes = {p: p.read_bytes() for p in before_paths if p.is_file()}

    with pytest.raises(PromotionError, match="symlink"):
        apply_promotion(plan, root, target)

    assert not outside.exists()
    assert dest.is_symlink() and dest.readlink() == outside
    assert set(target.rglob("*")) == before_paths
    assert {p: p.read_bytes() for p in before_paths if p.is_file()} == before_bytes


@pytest.mark.parametrize("linked", ["catalog/e2e", "catalog/e2e/h100_sxm.parquet"])
def test_symlinked_catalog_destination_is_refused_before_any_write(tmp_path, linked):
    root, directory, plan = _approved(tmp_path)
    target = _target(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "h100_sxm.parquet"
    sentinel.write_bytes(b"external bytes")
    (target / "catalog").mkdir()
    if linked == "catalog/e2e":
        (target / "catalog/e2e").symlink_to(outside, target_is_directory=True)
    else:
        (target / "catalog/e2e").mkdir()
        (target / linked).symlink_to(sentinel)
    with pytest.raises(PromotionError, match="symlink"):
        apply_promotion(plan, root, target)
    assert sentinel.read_bytes() == b"external bytes" and [p.name for p in outside.iterdir()] == [sentinel.name]
    assert not (target / "data").exists()  # nothing was copied either


@pytest.mark.parametrize("edited", [False, True])
def test_selected_files_are_bound_to_the_recorded_source_commit(tmp_path, edited):
    root, directory = _promotable(tmp_path)
    commit = _commit_source(root)
    if edited:
        # assume-unchanged hides the edit from `git status`; the recorded commit never contained these bytes.
        _git(root, "update-index", "--assume-unchanged", *_git(root, "ls-files").splitlines())
        manifest = root / directory / "manifest.json"
        document = Manifest.model_validate_json(manifest.read_bytes())
        manifest.write_text(
            document.model_copy(update={"evidence_gaps": [{"field": "new"}]}).model_dump_json(by_alias=True)
        )
        register_manifests(root, [manifest.relative_to(root).as_posix()], INTERNAL_REPO)
        build_catalog(root)
        assert _git(root, "status", "--porcelain", "--untracked-files=all", "--ignored") == ""
    plan = plan_promotion(root, _request(root, [directory], source_commit=commit), _reviewers())
    findings = [f for f in plan.findings if f.code == "source-checkout"]
    assert bool(findings) is edited
    if edited:
        assert "differs from the reviewed revision" in findings[0].message and not plan.content_ok


@pytest.mark.parametrize("failure", ["read-only-file", "replace-fails"])
def test_retrying_an_applied_plan_keeps_the_earlier_ledger_on_failure(tmp_path, monkeypatch, failure):
    from aisim_core.data import promote as promotion

    root, directory, plan = _approved(tmp_path)
    target = _target(tmp_path)
    ledger = apply_promotion(plan, root, target)
    original = ledger.read_bytes()
    if failure == "read-only-file":
        ledger.chmod(0o444)  # the directory stays writable, as in a retry of the same plan
        assert apply_promotion(plan, root, target) == ledger  # replaced atomically, never half-written
        assert json.loads(ledger.read_text())["digest"] == plan.digest
        return
    real = os.replace

    def failing(src, dst, *args, **kwargs):
        if Path(dst) == ledger:
            raise PermissionError("ledger directory is not writable")
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(promotion.os, "replace", failing)
    with pytest.raises(PermissionError):
        apply_promotion(plan, root, target)
    assert ledger.read_bytes() == original  # the earlier provenance record survives the failed retry
    assert [p.name for p in ledger.parent.iterdir()] == [ledger.name]  # and no temporary file is left behind
    assert (target / directory / "manifest.json").is_file()


def _two_step_promotion(tmp_path):
    """A target already holding one configuration, and an approved plan that adds a second to its catalog."""
    root, manifests = _mirror(tmp_path, dev=("synthetic/model-a", "synthetic/model-b"))
    directories = [str(Path(m).parent) for m in manifests]
    _public(root, *directories)
    plans = []
    for directory in directories:
        draft = plan_promotion(root, _request(root, [directory]), _reviewers())
        plans.append(plan_promotion(root, _request(root, [directory], draft), _reviewers()))
    target = _target(tmp_path)
    apply_promotion(plans[0], root, target)
    snapshot = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    return root, directories, plans[1], target, snapshot


def test_rollback_restores_only_changed_catalog_files_when_dataset_json_is_read_only(tmp_path, monkeypatch):
    root, directories, plan, target, before = _two_step_promotion(tmp_path)
    dataset = target / "catalog/dataset.json"
    dataset.chmod(0o444)  # readable, so planning works; the category index is rewritten, then this write fails
    restored = []
    real = Path.write_bytes

    def record(path, data):
        restored.append(path)
        return real(path, data)

    monkeypatch.setattr(Path, "write_bytes", record)
    try:
        with pytest.raises(PermissionError):
            apply_promotion(plan, root, target)
    finally:
        dataset.chmod(0o644)
    assert target / "catalog/e2e.json" in restored and dataset not in restored  # the untouched file is left alone
    assert not (target / directories[1]).exists()  # the copied payload is gone
    assert before == {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}


def test_rollback_attempts_every_restore_and_names_those_it_could_not_do(tmp_path, monkeypatch):
    root, directories, plan, target, before = _two_step_promotion(tmp_path)
    index = target / "catalog/e2e.json"
    dataset = target / "catalog/dataset.json"
    real = Path.write_bytes

    def refuse(path, data):
        if path == index:
            raise PermissionError("index is not writable")
        return real(path, data)

    real_write_text = Path.write_text

    def fail_dataset(path, *args, **kwargs):  # the registration that precedes the rollback
        if path == dataset:
            raise OSError("cannot write the dataset index")
        return real_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_dataset)
    monkeypatch.setattr(Path, "write_bytes", refuse)
    with pytest.raises(PromotionError, match="could not restore: catalog/e2e.json") as raised:
        apply_promotion(plan, root, target)
    assert isinstance(raised.value.__cause__, OSError)  # the original failure stays attached
    monkeypatch.undo()
    assert not (target / directories[1]).exists()  # the other cleanup still ran


@pytest.mark.parametrize(
    "blocked_kind,restore_fails", [("payload", False), ("catalog", False), ("directory", False), ("catalog", True)]
)
def test_rollback_continues_after_cleanup_failure(tmp_path, monkeypatch, blocked_kind, restore_fails):
    from aisim_core.data import promote as promotion
    from aisim_core.data.hygiene import Finding
    from aisim_core.data.mirror import TreeReport

    root, directories, plan, target, before = _two_step_promotion(tmp_path)
    index = target / "catalog/e2e.json"
    shard = target / "catalog/e2e/new-shard.parquet"
    blocked = {
        "payload": target / sorted(plan.files)[0],
        "catalog": shard,
        "directory": target / directories[1],
    }[blocked_kind]
    cleanup_attempts = []
    real_unlink, real_rmdir, real_write = Path.unlink, Path.rmdir, Path.write_bytes

    def remove(path, *args, **kwargs):
        cleanup_attempts.append(path)
        if path == blocked:
            raise PermissionError("fixture cleanup denied")
        return real_unlink(path, *args, **kwargs)

    def rmdir(path, *args, **kwargs):
        cleanup_attempts.append(path)
        if path == blocked:
            raise PermissionError("fixture cleanup denied")
        return real_rmdir(path, *args, **kwargs)

    def write(path, data):
        if restore_fails and path == index:
            raise PermissionError("fixture restore denied")
        return real_write(path, data)

    def reject(*args, **kwargs):
        assert index.read_bytes() != before[index.relative_to(target)]  # registration changed the live index
        shard.write_bytes(b"new catalog shard")
        return TreeReport(PUBLIC, "public", findings=[Finding("late-rejection", "fixture")])

    monkeypatch.setattr(promotion, "validate_tree", reject)
    monkeypatch.setattr(Path, "unlink", remove)
    monkeypatch.setattr(Path, "rmdir", rmdir)
    monkeypatch.setattr(Path, "write_bytes", write)
    with pytest.raises(PromotionError, match="rollback could not") as raised:
        apply_promotion(plan, root, target)
    assert blocked in cleanup_attempts and blocked.exists()
    assert blocked.relative_to(target).as_posix() in str(raised.value)
    assert isinstance(raised.value.__cause__, PromotionError)
    assert "late-rejection" in str(raised.value.__cause__)
    if restore_fails:
        assert "could not restore: catalog/e2e.json" in str(raised.value)
        assert index.read_bytes() != before[index.relative_to(target)]
    for rel, data in before.items():
        if not (restore_fails and target / rel == index):
            assert (target / rel).read_bytes() == data
    assert all(not (target / rel).exists() for rel in plan.files if target / rel != blocked)
    if blocked != shard:
        assert not shard.exists()
    assert not (target.parent / f".{target.name}.promotion.lock").exists()


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

    real_replace = os.replace
    refused = []

    def deny(src, dst, *a, **k):
        if "promotions" in Path(dst).parts:  # the ledger is written to a temporary file and moved into place
            refused.append(Path(dst))
            raise PermissionError("read-only")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr("aisim_core.data.promote.os.replace", deny)
    with pytest.raises(PermissionError):
        apply_promotion(plan, root, target)
    assert not any(p.is_file() for p in target.rglob("*"))
    ledger_dir = root.parent / f".{root.name}.promotion-ledger"
    assert len(refused) == 1 and not any(p.is_file() for p in ledger_dir.rglob("*"))  # no partial ledger remains


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
    shutil.rmtree(root / snapshot_dir / "snapshot-generations")  # the replaced snapshot leaves no orphan tables
    # A deliberately wrong pin cannot be written by a writer that sees the configuration, so hide it for the write.
    with _unbound_membership(root, path):
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


def test_rollback_keeps_a_preexisting_empty_target_directory(tmp_path, monkeypatch):
    from aisim_core.data import promote as promotion
    from aisim_core.data.hygiene import Finding
    from aisim_core.data.mirror import TreeReport

    root, _, plan = _approved(tmp_path)
    target = _target(tmp_path)
    (target / "data").mkdir()
    monkeypatch.setattr(
        promotion,
        "validate_tree",
        lambda *a, **k: TreeReport(repo=PUBLIC, profile="public", findings=[Finding("late", "rejected")]),
    )
    with pytest.raises(PromotionError, match="late"):
        apply_promotion(plan, root, target)
    assert (target / "data").is_dir() and not any((target / "data").iterdir())


@pytest.mark.parametrize("value", ["{}", '{"schema_version":"aisim-catalog-coverage/v2"}'])
def test_catalog_coverage_requires_the_supported_version(tmp_path, value):
    root, _ = _mirror(tmp_path)
    (root / "catalog/coverage.json").write_text(value)
    assert "catalog-unbuildable" in {f.code for f in validate_tree(root).errors}


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


def _publisher_tree_as_mirror(tmp_path, op_source):  # noqa: F811
    """The canonical publisher output, owned by the internal mirror (a hardware note it does not accept removed)."""
    root = tmp_path / "mirror"
    build_op_fixture(op_source, root)
    (root / "data/op-latency/h200_sxm/README.md").unlink()
    dataset = root / "catalog/dataset.json"
    dataset.write_text(json.dumps({**json.loads(dataset.read_bytes()), "dataset_id": INTERNAL_REPO}))
    return root


def test_category_support_travels_with_a_promoted_operation_configuration(tmp_path, op_source):  # noqa: F811
    from aisim_core.data import CategoryIndex
    from aisim_core.data.op_tree import load_op_latency_tree

    root = _publisher_tree_as_mirror(tmp_path, op_source)
    build_catalog(root)
    support = {"data/op-latency/perf_data_reuse_manifest.parquet", "data/op-latency/query_versions.parquet"}
    assert {
        a.path for a in CategoryIndex.model_validate_json((root / "catalog/op-latency.json").read_bytes()).support
    } == support
    directory = "data/op-latency/h200_sxm/gemm/vllm/1.0"  # the lookup tables sit above the selected directory
    _public(root, "data/op-latency")
    assert validate_tree(root).ok
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert draft.content_ok, draft.findings
    approved = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    assert approved.ok and support <= set(approved.files), approved.findings
    target = _target(tmp_path)
    apply_promotion(approved, root, target)
    index = CategoryIndex.model_validate_json((target / "catalog/op-latency.json").read_bytes())
    assert {a.path for a in index.support} == support
    assert validate_tree(target, repo=PUBLIC, profile="public").ok
    load_op_latency_tree(target, tmp_path / "composed")  # fails without the lookups the pinned fetch needs


@pytest.mark.parametrize("embedded", ["plain-synthetic-value", "[REDACTED]"])
def test_embedded_arrow_schema_metadata_is_scanned_in_the_promoted_file(tmp_path, embedded):
    import pyarrow.parquet as pq

    root, directory, relative = _op_mirror(tmp_path, "future_text", None)
    path = root / relative
    table = pq.read_table(path)
    schema = table.schema.with_metadata({**(table.schema.metadata or {}), b"password": embedded.encode()})
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        writer.write_table(table.cast(schema))
        # The reader exposes this override, while the encoded ARROW:schema keeps the original value.
        writer.add_key_value_metadata({"password": "[REDACTED]"})
    _refresh_artifact(root, directory, relative)
    leaked = embedded != "[REDACTED]"
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert ("secret" in {f.code for f in draft.findings}) is leaked
    scanned = tmp_path / "scanned"
    shutil.copytree(root, scanned, ignore=shutil.ignore_patterns(".git", "mirror"))
    assert ("secret" in {f.code for f in validate_tree(scanned, profile="public").errors}) is leaked
    approved = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    target = _target(tmp_path)
    if leaked:
        with pytest.raises(PromotionError, match="not approved"):
            apply_promotion(approved, root, target)
        return
    apply_promotion(approved, root, target)
    assert b"plain-synthetic-value" not in (target / relative).read_bytes()  # the copied bytes carry no credential


def test_eval_discovery_never_reads_an_oversized_json_whole(tmp_path, monkeypatch):
    from aisim_core.data.eval import record_paths
    from aisim_core.data.hygiene import CandidatePolicy

    root, manifests = _mirror(tmp_path)
    record = _eval(root, [load_e2e(root, manifests[0]).manifest.configuration_id])
    big = record.parent / "unselected.json"
    big.write_bytes(b" " * (CandidatePolicy().max_json_bytes + 1024) + b"{}")

    def refusing(real):
        def refuse(path, *args, **kwargs):
            if path == big:
                pytest.fail("eval discovery read an oversized JSON file whole")
            return real(path, *args, **kwargs)

        return refuse

    for name in ("read_text", "read_bytes"):  # an unbounded read of the oversized file would exhaust memory
        monkeypatch.setattr(Path, name, refusing(getattr(Path, name)))
    assert big in record_paths(root / "eval", recursive=True)  # still a candidate, never a silent sidecar
    rels = [p.relative_to(root).as_posix() for p in (record, big)]
    findings = check_publishable(root, rels)
    assert any(f.code == "document-invalid" and "limit" in f.message and f.path == rels[1] for f in findings)


def test_one_promotion_registers_two_new_categories(tmp_path, op_source):  # noqa: F811
    from aisim_core.data import DatasetIndex

    root = _publisher_tree_as_mirror(tmp_path, op_source)
    e2e_manifest = write_configuration(root, _config("synthetic/model-a"), prefix="data")
    register_manifests(root, [e2e_manifest], INTERNAL_REPO)
    build_catalog(root)
    directories = ["data/op-latency/h200_sxm/gemm/vllm/1.0", str(Path(e2e_manifest).parent)]
    _public(root, "data/op-latency", directories[1])
    draft = plan_promotion(root, _request(root, directories), _reviewers())
    assert draft.content_ok, draft.findings
    approved = plan_promotion(root, _request(root, directories, draft), _reviewers())
    assert approved.ok, approved.findings
    target = _target(tmp_path)
    apply_promotion(approved, root, target)  # a fresh target gains both categories in one apply
    dataset = DatasetIndex.model_validate_json((target / "catalog/dataset.json").read_bytes())
    assert set(dataset.categories) == {"op-latency", "e2e"}
    assert validate_tree(target, repo=PUBLIC, profile="public").ok


def test_category_support_is_scanned_like_any_published_file(tmp_path, op_source):  # noqa: F811
    import pyarrow.parquet as pq

    root = _publisher_tree_as_mirror(tmp_path, op_source)
    path = root / "data/op-latency/query_versions.parquet"
    table = pq.read_table(path)
    metadata = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items() if k != b"aisim.schema"}
    write_parquet(
        table,
        path,
        "aisim-op-query-versions/v1",
        extra_metadata={**metadata, "origin": "https://node7.corp.nvidia.com/v1"},
    )
    index = root / "catalog/op-latency.json"
    document = json.loads(index.read_bytes())
    for artifact in document["support"]:
        if artifact["path"] == "data/op-latency/query_versions.parquet":
            artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            artifact["bytes"] = path.stat().st_size
    index.write_text(json.dumps(document))
    dataset = root / "catalog/dataset.json"
    dataset_doc = json.loads(dataset.read_bytes())
    dataset_doc["categories"]["op-latency"]["index"].update(
        sha256=hashlib.sha256(index.read_bytes()).hexdigest(), bytes=index.stat().st_size
    )
    dataset.write_text(json.dumps(dataset_doc))
    build_catalog(root)
    _public(root, "data/op-latency")
    directory = "data/op-latency/h200_sxm/gemm/vllm/1.0"
    plan = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert "internal-host" in {f.code for f in plan.findings}


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
        (b"setting: !!bool unknown\n", "yaml-invalid"),
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
    value = "hf_" + "a" * 24 if dirty else "[REDACTED]"
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
    payload = json.dumps({"authorization": "hf_" + "a" * 24 if dirty else "[REDACTED]"}).encode()
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
    table = pa.table(
        {
            column: [
                json.dumps({"authorization": value if dirty else "[REDACTED]"})
                if column in ("inputs_json", "attrs")
                else value
            ]
        }
    )
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


def _set_column_chunk_file_path(path, value):
    """Rewrite the footer so every column chunk carries ``file_path`` (the data pages are untouched)."""
    import pyarrow.parquet as pq

    raw = path.read_bytes()
    footer_size = int.from_bytes(raw[-8:-4], "little")
    footer = pq.read_metadata(path)
    footer.set_file_path(value)
    sink = pa.BufferOutputStream()
    footer.write_metadata_file(sink)
    written = sink.getvalue().to_pybytes()[4:-8]
    path.write_bytes(raw[: len(raw) - 8 - footer_size] + written + len(written).to_bytes(4, "little") + b"PAR1")


@pytest.mark.parametrize("location", ["file", "field", "binary", "chunk-path"])
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
    elif location == "chunk-path":
        pass
    else:
        metadata = {"origin": value}
    write_parquet(table, root / relative, OP_TABLE_SCHEMA_ID, extra_metadata=metadata)
    if location == "chunk-path":
        _set_column_chunk_file_path(root / relative, value)
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


@pytest.mark.parametrize("manifest_name", ["manifest.json", "configuration.json"])
def test_operation_evidence_version_directory_and_strict_identity(tmp_path, manifest_name):
    from pydantic import ValidationError

    from aisim_core.data.evidence import EvidenceManifest, check_evidence

    root, directory, _ = _op_mirror(tmp_path, "future_text", None)
    original = root / directory / "manifest.json"
    if manifest_name != original.name:
        original.rename(original.with_name(manifest_name))
        register_manifests(root, [f"{directory}/{manifest_name}"], INTERNAL_REPO)
        build_catalog(root)
    manifest = Manifest.model_validate_json((root / directory / manifest_name).read_bytes())
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


def test_evidence_duplicate_keys_are_rejected(tmp_path):
    root, _ = _mirror(tmp_path)
    directory = _evidence(root)
    path = directory / "evidence-manifest.json"
    text = path.read_text()
    assert validate_tree(root).ok
    path.write_text(text.replace('"kind": "raw-bundle"', '"kind": "log", "kind": "raw-bundle"', 1))
    report = validate_tree(root)
    assert any(f.code == "document-invalid" and "duplicate JSON key" in f.message for f in report.errors)


@pytest.mark.parametrize("location", ["evidence", "dataset-index", "category-index", "snapshot"])
@pytest.mark.parametrize("profile", ["mirror", "public"])
def test_structured_documents_are_bounded_before_loading(tmp_path, monkeypatch, location, profile):
    import io

    from aisim_core.data.hygiene import CandidatePolicy

    root, directories = _snapshot_mirror(tmp_path)
    if location == "evidence":
        path = _evidence(root) / "evidence-manifest.json"
    elif location == "snapshot":
        path = root / directories[1] / "snapshot.json"
    elif location == "dataset-index":
        path = root / "catalog/dataset.json"
    else:
        dataset = json.loads((root / "catalog/dataset.json").read_bytes())
        path = root / dataset["categories"]["e2e"]["index"]["path"]
    limit = CandidatePolicy().max_json_bytes
    payload = b" " * (limit + 1)
    path.write_bytes(payload)
    original = Path.open

    class Bounded(io.BytesIO):
        def read(self, size=-1):
            assert 0 <= size <= limit + 1
            return super().read(size)

    def opened(candidate, *args, **kwargs):
        if candidate == path:
            return Bounded(payload)
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    report = validate_tree(root, profile=profile)
    relative = path.relative_to(root).as_posix()
    assert any(
        f.path == relative and f.code in {"json-too-large", "document-invalid", "path-root"} for f in report.errors
    )


@pytest.mark.parametrize("declared", [False, True])
def test_uppercase_gzip_requires_a_declaration(tmp_path, declared):
    root, directory, _ = _op_mirror(tmp_path, "future_text", None)
    path = root / directory / "capture.GZ"
    import gzip

    path.write_bytes(gzip.compress(b"synthetic evidence"))
    if declared:
        manifest_path = root / directory / "manifest.json"
        manifest = Manifest.model_validate_json(manifest_path.read_bytes())
        artifact = Artifact(
            artifact_id="uppercase-capture",
            path=path.relative_to(root).as_posix(),
            role="raw-output",
            format="gzip",
            schema="aisim-provenance/v1",
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            bytes=path.stat().st_size,
        )
        manifest_path.write_text(
            manifest.model_copy(update={"artifacts": [*manifest.artifacts, artifact]}).model_dump_json(by_alias=True)
        )
        register_manifests(root, [manifest_path.relative_to(root).as_posix()], INTERNAL_REPO)
        build_catalog(root)
    report = validate_tree(root)
    assert ("undeclared-archive" in {f.code for f in report.errors}) is not declared
    assert report.ok is declared, report.errors


def test_policy_rejects_misspelled_approval_threshold(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"reviewers": ["alice", "bob"], "min_approval": 2}))
    with pytest.raises(ValueError, match="unknown approval-policy keys"):
        PromotionPolicy.from_file(path)


def test_empty_directory_cannot_be_approved_for_promotion(tmp_path):
    root, _ = _promotable(tmp_path)
    (root / "data/empty").mkdir()
    plan = plan_promotion(root, _request(root, ["data/empty"]), _reviewers())
    assert not plan.content_ok and "selection-empty" in {f.code for f in plan.findings}


def test_fixture_and_promotion_git_ignore_foreign_repository_environment(tmp_path, monkeypatch):
    from aisim_core.data.promote import _check_source_checkout

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "untouched").write_text("foreign checkout")
    _commit_source(foreign)
    before = {p.relative_to(foreign): p.read_bytes() for p in foreign.rglob("*") if p.is_file()}
    for name, value in {
        "GIT_DIR": str(foreign / ".git"),
        "GIT_WORK_TREE": str(foreign),
        "GIT_INDEX_FILE": str(foreign / ".git/index"),
        "GIT_COMMON_DIR": str(foreign / ".git"),
        "GIT_OBJECT_DIRECTORY": str(foreign / ".git/objects"),
        "GIT_PREFIX": "foreign/",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("AISIM_FIXTURE_UNRELATED", "preserved")
    original_run = subprocess.run

    def run(*args, **kwargs):
        assert kwargs["env"]["AISIM_FIXTURE_UNRELATED"] == "preserved"
        assert not {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"} & kwargs["env"].keys()
        return original_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    root, directory = _promotable(tmp_path)
    commit = _commit_source(root)
    request = _request(root, [directory])
    assert request.source_commit == commit
    _check_source_checkout(root, commit)
    assert os.environ["AISIM_FIXTURE_UNRELATED"] == "preserved"
    assert before == {p.relative_to(foreign): p.read_bytes() for p in foreign.rglob("*") if p.is_file()}


def test_rollback_does_not_read_unreadable_generated_catalog(tmp_path, monkeypatch):
    from aisim_core.data import promote as promotion
    from aisim_core.data.hygiene import Finding
    from aisim_core.data.mirror import TreeReport

    root, _, plan = _approved(tmp_path)
    (tmp_path / "existing").mkdir()
    target, _ = _mirror(tmp_path / "existing", repo=PUBLIC, dev=("synthetic/model-b",))
    before = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    unreadable = target / "catalog/e2e/unreadable.parquet"
    original = Path.read_bytes

    def read_bytes(path):
        if path == unreadable:
            raise OSError("generated shard stays unreadable")
        return original(path)

    def reject(*args, **kwargs):
        unreadable.write_bytes(b"unreadable shard")
        return TreeReport(PUBLIC, "public", findings=[Finding("catalog-unreadable", "fixture")])

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(promotion, "validate_tree", reject)
    with pytest.raises(PromotionError, match="catalog-unreadable"):
        apply_promotion(plan, root, target)
    assert not unreadable.exists()
    assert before == {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}


@pytest.mark.parametrize("redacted", [False, True])
def test_public_validation_rejects_populated_password_fields(tmp_path, redacted):
    root, manifests = _mirror(tmp_path, repo=PUBLIC)
    manifest_path = root / manifests[0]
    manifest = Manifest.model_validate_json(manifest_path.read_bytes())
    value = "[REDACTED]" if redacted else "***REDACTED_SECRET_ASSIGNMENT***"
    manifest = manifest.model_copy(update={"provenance": {**manifest.provenance, "password": value}})
    manifest_path.write_text(manifest.model_dump_json(by_alias=True))
    register_manifests(root, [manifest_path.relative_to(root).as_posix()], PUBLIC)
    build_catalog(root)
    report = validate_tree(root, profile="public")
    assert ("secret" in {f.code for f in report.errors}) is not redacted
    assert report.ok is redacted, report.errors


@pytest.mark.parametrize("redacted", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("binary_keys", [False, True])
def test_public_validation_preserves_arrow_map_disclosure_context(tmp_path, redacted, nested, binary_keys):
    import pyarrow.parquet as pq

    root, directory, _ = _op_mirror(tmp_path, "future_text", None)
    shard = root / directory / "gemm_perf.parquet"
    table = pq.read_table(shard)
    value = "[REDACTED]" if redacted else "private-worker:v1"
    key = b"image" if binary_keys else "image"
    dtype = pa.map_(pa.binary() if binary_keys else pa.string(), pa.string())
    cell = [(key, value), (key, "[REDACTED]")]
    if nested:
        dtype = pa.struct([pa.field("nested", pa.list_(dtype))])
        cell = {"nested": [cell]}
    table = table.append_column("attrs", pa.array([cell] * table.num_rows, type=dtype))
    write_parquet(table, shard, OP_TABLE_SCHEMA_ID)
    manifest_path = root / directory / "manifest.json"
    manifest = Manifest.model_validate_json(manifest_path.read_bytes())
    artifacts = [
        artifact.model_copy(
            update={"sha256": hashlib.sha256(shard.read_bytes()).hexdigest(), "bytes": shard.stat().st_size}
        )
        if artifact.path == shard.relative_to(root).as_posix()
        else artifact
        for artifact in manifest.artifacts
    ]
    manifest_path.write_text(manifest.model_copy(update={"artifacts": artifacts}).model_dump_json(by_alias=True))
    register_manifests(root, [manifest_path.relative_to(root).as_posix()], INTERNAL_REPO)
    build_catalog(root)
    shutil.rmtree(root / "mirror")  # public checkouts omit the internal visibility ledger
    report = validate_tree(root, profile="public")
    assert ("image-name" in {f.code for f in report.errors}) is not redacted
    assert report.ok is redacted, report.errors


@pytest.mark.parametrize("value", ["plain-synthetic-value", "[REDACTED]"])
@pytest.mark.parametrize("key", [b"password", "password"])
def test_binary_map_keys_keep_their_field_name_context(key, value):
    from aisim_core.data.redaction import DEFAULT_POLICY, scan_value

    # Arrow returns a binary map key as bytes; "b'password'" must not hide the credential-field rule.
    findings = scan_value([{key: value}], DEFAULT_POLICY, "p", "t")
    assert ("secret" in {f.code for f in findings}) is (value != "[REDACTED]")


def test_binary_and_text_map_keys_stay_separate_entries():
    from aisim_core.data.redaction import DEFAULT_POLICY, scan_value

    findings = scan_value([{b"password": "[REDACTED]", "password": "plain-synthetic-value"}], DEFAULT_POLICY, "p", "t")
    assert "secret" in {f.code for f in findings}


def test_invalid_utf8_map_key_is_refused():
    from aisim_core.data.redaction import DEFAULT_POLICY, scan_value

    with pytest.raises(UnicodeDecodeError):
        scan_value([{b"\xff": "[REDACTED]"}], DEFAULT_POLICY, "p", "t")


@pytest.mark.parametrize("name", ["snapshot.JSON", "frozen-view.JSON"])
def test_uppercase_snapshot_documents_reach_publication_checks(tmp_path, name):
    root, directories = _snapshot_mirror(tmp_path)
    original = root / directories[1] / "snapshot.json"
    renamed = original.with_name(name)
    original.rename(renamed)
    report = validate_tree(root)
    assert report.ok, report.errors
    assert not any(f.code == "orphan-file" for f in report.findings)
    request = _request(root, directories)
    draft = plan_promotion(root, request, _reviewers())
    plan = plan_promotion(root, _request(root, directories, draft), _reviewers())
    assert plan.ok, plan.findings
    target = _target(tmp_path)
    apply_promotion(plan, root, target)
    assert (target / renamed.relative_to(root)).read_bytes() == renamed.read_bytes()
    assert validate_tree(target, profile="public").ok


@pytest.mark.parametrize("which", ["members", "acquisition"])
def test_snapshot_artifacts_must_be_in_the_approved_selection(tmp_path, which):
    from aisim_core.data.e2e import read_snapshot

    root, directories = _snapshot_mirror(
        tmp_path, acquisition={"acquired_at": "2026-11-07T00:00:00Z", "source_uri": "https://example.test"}
    )
    document = root / directories[1] / "snapshot.json"
    snapshot, _ = read_snapshot(root, str(document.relative_to(root)))
    artifact = getattr(snapshot, which)
    outside = root / "data/elsewhere" / Path(artifact.path).name
    outside.parent.mkdir(parents=True)
    (root / artifact.path).rename(outside)
    moved = artifact.model_copy(update={"path": str(outside.relative_to(root))})
    document.write_text(snapshot.model_copy(update={which: moved}).model_dump_json(by_alias=True))
    assert validate_tree(root).ok
    plan = plan_promotion(root, _request(root, directories), _reviewers())
    assert any(f.code == "closure" and moved.path in f.message for f in plan.findings)
    assert not plan.content_ok


def test_lineage_uses_the_governing_visibility_entry(tmp_path):
    root, directory = _promotable(tmp_path)
    relative = next((root / directory).rglob("metrics.parquet")).relative_to(root).as_posix()
    write_ledger(
        root,
        VisibilityLedger(
            entries=[
                VisibilityEntry(path=directory, label="public", original_sha256={relative: "a" * 64}),
                VisibilityEntry(path=relative, label="public", original_sha256={relative: "b" * 64}),
            ]
        ),
    )
    draft = plan_promotion(root, _request(root, [directory]), _reviewers())
    plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
    ledger = apply_promotion(plan, root, _target(tmp_path))
    assert json.loads(ledger.read_text())["original_sha256"] == {relative: "b" * 64}


def test_observation_provenance_classifies_configuration_and_eval(tmp_path):
    root = tmp_path / "mirror"
    root.mkdir()
    config = _config()
    config = dataclasses.replace(
        config, observations=[o.model_copy(update={"source_kind": "inferencex"}) for o in config.observations]
    )
    manifest = write_configuration(root, config, prefix="data")
    register_manifests(root, [manifest], INTERNAL_REPO)
    record = _eval(root, [config.body.configuration_id()])
    build_catalog(root)
    directories = [str(Path(manifest).parent), str(record.parent.relative_to(root))]
    _public(root, *directories)
    assert validate_tree(root).ok
    plan = plan_promotion(root, _request(root, directories), _reviewers())
    blocked = {f.path for f in plan.findings if f.code == "infx-aggregates-only"}
    assert manifest in blocked and str(record.relative_to(root)) in blocked
    report = validate_tree(root, profile="public")
    blocked = {f.path for f in report.errors if f.code == "infx-aggregates-only"}
    assert manifest in blocked and str(record.relative_to(root)) in blocked


@pytest.mark.parametrize("dirty", [False, True])
def test_operation_parquet_producer_reaches_both_publication_gates(tmp_path, dirty):
    import pyarrow.parquet as pq

    root, directory, relative = _op_mirror(tmp_path, "kernel_source", "synthetic-kernel")
    path = root / relative
    created_by = pq.ParquetFile(path).metadata.created_by.encode()
    replacement = (b"node7.corp.nvidia.com" if dirty else b"public-writer").ljust(len(created_by), b" ")
    assert len(replacement) == len(created_by)
    path.write_bytes(path.read_bytes().replace(created_by, replacement))
    _refresh_artifact(root, directory, relative)
    assert validate_tree(root).ok
    plan = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert any(f.code == "internal-host" and f.path == relative for f in plan.findings) is dirty
    shutil.rmtree(root / "mirror")  # the private visibility ledger is not part of the public dataset
    report = validate_tree(root, profile="public")
    assert report.ok is (not dirty)
    assert any(f.code == "internal-host" and f.path == relative for f in report.errors) is dirty


@pytest.mark.parametrize("location", ["dataset", "category", "noncanonical-category"])
@pytest.mark.parametrize("dirty", [False, True])
def test_dataset_indexes_receive_public_disclosure_checks(tmp_path, location, dirty):
    from aisim_core.data import DatasetIndex

    root, _ = _mirror(tmp_path, repo=PUBLIC)
    dataset_path = root / "catalog/dataset.json"
    dataset = json.loads(dataset_path.read_bytes())
    ref = dataset["categories"]["e2e"]["index"]
    category_path = root / ref["path"]
    value = "node7.corp.nvidia.com" if dirty else "public-index"
    if location == "dataset":
        ref["artifact_id"] = value
    else:
        category = json.loads(category_path.read_bytes())
        category["entries"][0]["manifest"]["artifact_id"] = value
        if location == "noncanonical-category":
            old = category_path
            category_path = root / "data/indexes/custom.JSON"
            category_path.parent.mkdir(parents=True)
            old.unlink()
            ref["path"] = str(category_path.relative_to(root))
        category_path.write_text(json.dumps(category))
        ref.update(sha256=hashlib.sha256(category_path.read_bytes()).hexdigest(), bytes=category_path.stat().st_size)
    dataset_path.write_text(DatasetIndex.model_validate(dataset).model_dump_json(by_alias=True))
    assert validate_tree(root).ok
    report = validate_tree(root, profile="public")
    assert report.ok is (not dirty), report.errors
    relative = str((dataset_path if location == "dataset" else category_path).relative_to(root))
    assert any(f.code == "internal-host" and f.path == relative for f in report.errors) is dirty


@pytest.mark.parametrize("mutation", ["remove-file", "replace-hash", "clear-findings"])
def test_apply_rechecks_the_approved_digest_before_writing(tmp_path, mutation):
    root, _, plan = _approved(tmp_path)
    if mutation == "clear-findings":
        plan = plan_promotion(root, plan.request.model_copy(update={"approvals": []}), _reviewers())
        plan.findings.clear()
    elif mutation == "remove-file":
        plan.files.pop(next(iter(plan.files)))
    else:
        plan.files[next(iter(plan.files))] = "0" * 64
    target = tmp_path / "target"
    with pytest.raises(PromotionError, match="approved digest"):
        apply_promotion(plan, root, target)
    assert not target.exists()


def test_apply_refuses_an_unapproved_configuration_added_to_the_plan(tmp_path):
    root, manifests = _mirror(tmp_path, dev=("synthetic/model-a", "synthetic/model-b"))
    directories = [str(Path(manifest).parent) for manifest in manifests]
    _public(root, *directories)
    draft = plan_promotion(root, _request(root, directories[:1]), _reviewers())
    plan = plan_promotion(root, _request(root, directories[:1], draft), _reviewers())
    assert plan.ok, plan.findings
    for path in (root / directories[1]).rglob("*"):
        if path.is_file():
            plan.files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    plan.manifests.append(manifests[1])
    target = tmp_path / "target"
    with pytest.raises(PromotionError, match="approved digest"):
        apply_promotion(plan, root, target)
    assert not target.exists()


@pytest.mark.parametrize("location", ["directory", "artifact"])
def test_standalone_evidence_preflights_parent_links_before_reading(tmp_path, monkeypatch, location):
    from aisim_core.data.evidence import check_evidence

    root = tmp_path / "root"
    directory = _evidence(root, {"raw.log": b"fixture"})
    if location == "directory":
        moved = tmp_path / "outside"
        directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)
        paths = [str((directory / p.name).relative_to(root)) for p in moved.iterdir()]
        forbidden = {directory / p.name for p in moved.iterdir()}
    else:
        moved = tmp_path / "outside"
        moved.mkdir()
        (directory / "raw.log").rename(moved / "raw.log")
        (directory / "linked").symlink_to(moved, target_is_directory=True)
        manifest = directory / "evidence-manifest.json"
        document = json.loads(manifest.read_text())
        document["artifacts"][0]["path"] = "linked/raw.log"
        manifest.write_text(json.dumps(document))
        paths = [str(p.relative_to(root)) for p in [manifest, directory / "linked/raw.log"]]
        forbidden = {directory / "linked/raw.log"}
    original = Path.open

    def opened(path, *args, **kwargs):
        assert path not in forbidden, "evidence validator opened a file through a linked directory"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    findings, _ = check_evidence(root, paths)
    assert any("symlink" in f.message for f in findings)


def test_visibility_loader_is_strict_bounded_and_defaults_only_for_absence(tmp_path, monkeypatch):
    from aisim_core.data.hygiene import CandidatePolicy
    from aisim_core.data.redaction import load_ledger

    assert load_ledger(tmp_path) == VisibilityLedger()
    path = tmp_path / "mirror/visibility.json"
    path.parent.mkdir()
    for content in [b'{"entries":[],"entries":[]}', b'{"schema_version":"unsupported"}', b"[]", b"{"]:
        path.write_bytes(content)
        with pytest.raises(ValueError):
            load_ledger(tmp_path)
    path.write_bytes(b" " * (CandidatePolicy().max_json_bytes + 1))
    with pytest.raises(ValueError, match="exceeds"):
        load_ledger(tmp_path)
    path.unlink()
    path.mkdir()
    with pytest.raises(OSError):
        load_ledger(tmp_path)


@pytest.mark.parametrize("location", ["manifest", "copy"])
def test_malformed_system_yaml_writes_cli_report(tmp_path, location):
    from aisim_core.data import systems

    root, _ = _mirror(tmp_path)
    systems.publish_system_spec(systems.PACKAGED_SYSTEMS_DIR / "h100_sxm.yaml", root)
    path = root / "systems/h100_sxm/manifest.yaml"
    if location == "copy":
        path = next(p for p in path.parent.glob("*.yaml") if p.name != "manifest.yaml")
    path.write_text("invalid: !!bool not-a-bool\n")
    output = tmp_path / "report.json"
    assert mirror_main(["validate", "--root", str(root), "--output", str(output)]) == 1
    assert any(f["code"] == "system-spec-invalid" for f in json.loads(output.read_text())["findings"])


@pytest.mark.parametrize("oversized", [False, True])
def test_promotion_stops_before_reopening_oversized_snapshot(tmp_path, monkeypatch, oversized):
    from aisim_core.data import promote as module
    from aisim_core.data.hygiene import CandidatePolicy

    root, directories = _snapshot_mirror(tmp_path)
    if oversized:
        (root / directories[1] / "snapshot.json").write_bytes(b" " * (CandidatePolicy().max_json_bytes + 1))
    original = module.read_snapshot
    calls = []

    def read(*args):
        calls.append(args)
        assert not oversized, "planning reopened a snapshot rejected by the size preflight"
        return original(*args)

    monkeypatch.setattr(module, "read_snapshot", read)
    plan = plan_promotion(root, _request(root, directories), _reviewers())
    assert ("source-json-too-large" in {f.code for f in plan.findings}) is oversized
    assert bool(calls) is not oversized
    if not oversized:
        assert plan.content_ok, plan.findings


def test_approval_policy_rejects_duplicate_threshold(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text('{"reviewers":["alice"],"min_approvals":2,"min_approvals":1}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        PromotionPolicy.from_file(path)


def test_narrow_public_label_cannot_clear_inferencex_ancestry(tmp_path):
    root, manifests = _mirror(tmp_path)
    directory = str(Path(manifests[0]).parent)
    write_ledger(
        root,
        VisibilityLedger(
            entries=[
                VisibilityEntry(path="data", label="internal", origin="inferencex"),
                VisibilityEntry(path=directory, label="public"),
            ]
        ),
    )
    plan = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert "not-public" not in {f.code for f in plan.findings}
    assert {f.path for f in plan.findings if f.code == "infx-aggregates-only"} == set(plan.files)


def test_manifest_registration_alone_invalidates_approval(tmp_path):
    root, manifests = _mirror(tmp_path, dev=("synthetic/model-a", "synthetic/model-b"))
    directories = [str(Path(manifest).parent) for manifest in manifests]
    _public(root, *directories)
    draft = plan_promotion(root, _request(root, directories[:1]), _reviewers())
    plan = plan_promotion(root, _request(root, directories[:1], draft), _reviewers())
    assert plan.ok, plan.findings
    target = _target(tmp_path)
    shutil.copytree(root / directories[1], target / directories[1])
    before = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    files = plan.files.copy()
    digest = plan.digest
    plan.manifests.append(manifests[1])
    assert plan.files == files
    assert plan.digest != digest
    with pytest.raises(PromotionError, match="approved digest"):
        apply_promotion(plan, root, target)
    # Even recomputing approvals cannot register a manifest outside the selected files.
    plan.approved_digest = plan.digest
    with pytest.raises(PromotionError, match="manifest.*selection"):
        apply_promotion(plan, root, target)
    assert {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("value", ["[0.1,0.2,0.3,0.4,0.5]", '{"case-a":0.1}', "  [1, 2]", "[1, 2"])
def test_aggregate_status_rejects_serialized_measurements(value):
    assert "not-aggregate" in {f.code for f in check_disclosure([{"n_scored": 5, "status": value}])}


@pytest.mark.parametrize("value", ["complete", "partial", "failed", "have", "missing", "stale"])
def test_aggregate_status_accepts_ordinary_values(value):
    assert check_disclosure([{"n_scored": 5, "status": value}]) == []


@pytest.mark.parametrize("lead", ["", "\n", "\n\n  \n"])  # Hugging Face accepts front matter after whitespace
@pytest.mark.parametrize("value", ["plain-synthetic-value", "[REDACTED]"])
def test_dataset_card_front_matter_credential_fields(tmp_path, value, lead):
    root, _ = _mirror(tmp_path)
    (root / "README.md").write_text(f'{lead}---\npassword: "{value}"\n---\n# Synthetic dataset\n')
    shutil.rmtree(root / "mirror", ignore_errors=True)
    report = validate_tree(root, profile="public")
    secrets = [f for f in report.errors if f.code == "secret" and f.path == "README.md"]
    assert bool(secrets) == (value != "[REDACTED]")
    assert report.ok == (value == "[REDACTED]"), report.errors


@pytest.mark.parametrize("which", ["members", "acquisition"])
@pytest.mark.parametrize("location", [".git/hidden-table.parquet", "data/elsewhere/table.parquet"])
def test_snapshot_tables_must_be_dataset_payloads(tmp_path, monkeypatch, which, location):
    from aisim_core.data.e2e import read_snapshot

    root, directories = _snapshot_mirror(
        tmp_path, acquisition={"acquired_at": "2026-11-07T00:00:00Z", "source_uri": "https://example.test"}
    )
    document = root / directories[1] / "snapshot.json"
    snapshot, _ = read_snapshot(root, document.relative_to(root).as_posix())
    table = getattr(snapshot, which)
    destination = root / location
    destination.parent.mkdir(parents=True, exist_ok=True)
    (root / table.path).rename(destination)
    document.write_text(
        snapshot.model_copy(update={which: table.model_copy(update={"path": location})}).model_dump_json(by_alias=True)
    )
    hidden = location.startswith(".git/")
    if hidden:
        real_open = Path.open

        def refuse_hidden_table(self, *args, **kwargs):
            if self == destination:
                pytest.fail("validation opened a snapshot table outside the dataset payload")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", refuse_hidden_table)
    shutil.rmtree(root / "mirror", ignore_errors=True)
    report = validate_tree(root, profile="public")
    assert ("snapshot-invalid" in {f.code for f in report.errors}) is hidden
    # A noncanonical table path inside the payload inventory stays supported.
    assert hidden or report.ok, report.errors


@pytest.mark.parametrize("value", ["plain-synthetic-value", "[REDACTED]"])
def test_dataset_card_front_matter_ends_at_an_unindented_delimiter(tmp_path, value):
    root, _ = _mirror(tmp_path)
    # An indented --- is block-scalar text in YAML; the Hub only closes the card at an unindented one.
    card = f'\n---\ndescription: |\n  first\n  ---\n  second\npassword: "{value}"\n---\n# Synthetic dataset\n'
    (root / "README.md").write_text(card)
    shutil.rmtree(root / "mirror", ignore_errors=True)
    report = validate_tree(root, profile="public")
    secrets = [f for f in report.errors if f.code == "secret" and f.path == "README.md"]
    assert bool(secrets) == (value != "[REDACTED]")
    assert report.ok == (value == "[REDACTED]"), report.errors


@pytest.mark.parametrize("profile", ["mirror", "public"])
@pytest.mark.parametrize("location", [".git/hidden-index.json", "data/indexes/e2e.json"])
def test_category_indexes_must_be_dataset_payloads(tmp_path, monkeypatch, profile, location):
    from aisim_core.data import DatasetIndex
    from aisim_core.data.promote import _artifact

    root, _ = _mirror(tmp_path)
    index_path = root / "catalog/dataset.json"
    dataset = DatasetIndex.model_validate_json(index_path.read_bytes())
    original = root / dataset.categories["e2e"].index.path
    destination = root / location
    destination.parent.mkdir(parents=True, exist_ok=True)
    original.rename(destination)
    categories = dict(dataset.categories)
    categories["e2e"] = categories["e2e"].model_copy(
        update={"index": _artifact(root, location, "aisim-category-index/v1")}
    )
    index_path.write_text(dataset.model_copy(update={"categories": categories}).model_dump_json(by_alias=True))
    if profile == "public":
        shutil.rmtree(root / "mirror", ignore_errors=True)
    if location.startswith(".git/"):
        real_open = Path.open

        def refuse_hidden_index(self, *args, **kwargs):
            if self == destination:
                pytest.fail("validation opened an index outside the dataset payload")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", refuse_hidden_index)
    report = validate_tree(root, profile=profile)
    assert ("index-integrity" in {f.code for f in report.errors}) == location.startswith(".git/")
    assert report.ok == (not location.startswith(".git/")), report.errors


def test_competing_eval_successors_of_one_record_are_reported(tmp_path):
    from aisim_core.data.eval import current_records, load_record, record_path

    root, manifests = _mirror(tmp_path)
    first_path = _eval(root, [load_e2e(root, manifests[0]).manifest.configuration_id])
    first = load_record(first_path)
    # A linear history of three records is valid: each record has exactly one successor.
    second = first.model_copy(update={"supersedes": first.record_id, "created_at": "2026-11-08T00:00:00Z"})
    write_record(root, second)
    third = second.model_copy(update={"supersedes": second.record_id, "created_at": "2026-11-09T00:00:00Z"})
    third_path = write_record(root, third)
    build_catalog(root)
    assert validate_tree(root).ok, validate_tree(root).errors
    assert [r.record_id for r in current_records(first_path.parent)] == [third.record_id]
    # Another branch published its own successor of the same record. Merging keeps both files: their names differ.
    producer = second.producer.model_copy(update={"name": "another branch"})
    rival = third.model_copy(update={"producer": producer})
    rival_path = record_path(root, rival)
    rival_path.write_text(json.dumps(rival.model_dump(by_alias=True, mode="json"), indent=2, sort_keys=True) + "\n")
    report = validate_tree(root)
    conflicts = [f for f in report.errors if f.code == "eval-competing-successors"]
    assert {f.path for f in conflicts} == {
        rival_path.relative_to(root).as_posix(),
        third_path.relative_to(root).as_posix(),
    }
    for finding in conflicts:
        assert second.record_id in finding.message and third.record_id in finding.message
        assert rival.record_id in finding.message
    assert not report.ok


def test_an_eval_successor_cannot_rank_below_its_predecessor(tmp_path):
    from aisim_core.data.eval import VERIFIER_VERSION, GpuTime, current_records, load_record

    root, manifests = _mirror(tmp_path)
    first_path = _eval(root, [load_e2e(root, manifests[0]).manifest.configuration_id])
    submitted = load_record(first_path)
    measured = GpuTime(authority="measured", gpu_count=4, wall_seconds=1.0)

    def successor(predecessor, tier, day):
        update = {"supersedes": predecessor.record_id, "tier": tier, "created_at": f"2026-11-{day}T00:00:00Z"}
        update["gpu_time"] = measured.model_dump()
        update["verification"] = (
            None
            if tier == "submitted"
            else {
                "verifier_version": VERIFIER_VERSION,
                "recomputed_fingerprint": predecessor.prediction.fingerprint,
                "verified_record_id": predecessor.record_id,
            }
        )
        return EvalRecord.model_validate({**predecessor.model_dump(by_alias=True), **update})

    verified = successor(submitted, "verified", "08")
    retry = successor(verified, "verified", "09")  # a same-tier retry
    qualified = successor(retry, "qualified", "10")  # an upgrade
    for record in (verified, retry, qualified):
        write_record(root, record)
    build_catalog(root)
    assert validate_tree(root).ok, validate_tree(root).errors  # submitted -> verified -> verified -> qualified
    directory = first_path.parent
    # A downgrade names the qualified record as its predecessor and would hide it from the current view.
    demoted = successor(qualified, "submitted", "11")
    demoted_path = write_record(root, demoted)
    report = validate_tree(root)
    found = [f for f in report.errors if f.code == "eval-tier-downgrade"]
    assert [f.path for f in found] == [demoted_path.relative_to(root).as_posix()]
    assert qualified.record_id in found[0].message and "submitted" in found[0].message
    assert not report.ok and [r.tier for r in current_records(directory)] == ["submitted"]


@pytest.mark.parametrize("predecessor", ["valid", "missing", "invalid"])
def test_eval_successor_requires_verified_local_predecessor(tmp_path, predecessor):
    from aisim_core.data.eval import load_record

    root, manifests = _mirror(tmp_path)
    first_path = _eval(root, [load_e2e(root, manifests[0]).manifest.configuration_id])
    first = load_record(first_path)
    successor = first.model_copy(update={"supersedes": first.record_id, "created_at": "2026-11-08T00:00:00Z"})
    second_path = write_record(root, successor)
    build_catalog(root)
    if predecessor == "missing":
        first_path.unlink()
    elif predecessor == "invalid":
        first_path.write_text("{}")
    report = validate_tree(root)
    missing = [f for f in report.errors if f.code == "eval-predecessor-missing"]
    assert bool(missing) == (predecessor != "valid")
    if missing:
        assert missing[0].path == second_path.relative_to(root).as_posix()
    else:
        assert report.ok, report.errors


@pytest.mark.parametrize("gate", ["plan", "apply"])
def test_visibility_cannot_be_changed_through_assume_unchanged(tmp_path, gate):
    root, directory = _promotable(tmp_path)
    if gate == "plan":
        write_ledger(root, VisibilityLedger(entries=[VisibilityEntry(path=directory, label="restricted")]))
        _commit_source(root)
    else:
        draft = plan_promotion(root, _request(root, [directory]), _reviewers())
        plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
        assert plan.ok
    _git(root, "update-index", "--assume-unchanged", "mirror/visibility.json")
    _public(root, directory, origin="altered label provenance")
    assert not _git(root, "status", "--porcelain", "--untracked-files=all", "--ignored")
    if gate == "plan":
        plan = plan_promotion(root, _request(root, [directory]), _reviewers())
        assert any(f.code == "source-checkout" and "visibility" in f.message for f in plan.findings)
    else:
        target = _target(tmp_path)
        with pytest.raises(PromotionError, match="visibility ledger differs"):
            apply_promotion(plan, root, target)
        assert not list(target.iterdir())


def test_two_approved_promotions_share_one_reviewed_source_revision(tmp_path):
    root, manifests = _mirror(tmp_path, dev=("synthetic/model-a", "synthetic/model-b"))
    directories = [str(Path(path).parent) for path in manifests]
    _public(root, *directories)
    plans = []
    for directory in directories:
        draft = plan_promotion(root, _request(root, [directory]), _reviewers())
        plan = plan_promotion(root, _request(root, [directory], draft), _reviewers())
        assert plan.ok, plan.findings
        plans.append(plan)
    before = {
        f.relative_to(root).as_posix(): f.read_bytes() for f in root.rglob("*") if f.is_file() and ".git" not in f.parts
    }
    target = _target(tmp_path)
    ledgers = [apply_promotion(plan, root, target) for plan in plans]
    assert plans[0].request.source_commit == plans[1].request.source_commit == _git(root, "rev-parse", "HEAD")
    assert all(not ledger.is_relative_to(root) and ledger.is_file() for ledger in ledgers)
    assert not _git(root, "status", "--porcelain", "--untracked-files=all", "--ignored")
    assert before == {
        f.relative_to(root).as_posix(): f.read_bytes() for f in root.rglob("*") if f.is_file() and ".git" not in f.parts
    }
    report = validate_tree(target, profile="public")
    assert report.ok and report.configurations == 2, report.errors


@pytest.mark.parametrize("secret", [True, False])
@pytest.mark.parametrize("merge", ["inline", "sequence", "literal-key"])
def test_yaml_overridden_merge_defaults_are_scanned_in_publication_gates(tmp_path, secret, merge):
    root, directory, _ = _op_mirror(tmp_path, "kernel_source", "clean")
    value = "plain-synthetic-value" if secret else "[REDACTED]"
    defaults = f'{{password: "{value}"}}'
    if merge == "sequence":
        defaults = f"[{defaults}, {{option: clean}}]"
    payload = f'options:\n  <<: {defaults}\n  password: "[REDACTED]"\n'
    if merge == "literal-key":
        payload += '  "<<": "[REDACTED]"\n'
    payload = payload.encode()
    rel = _manifest_sidecar(
        root,
        f"{directory}/manifest.json",
        "collection_meta.yaml",
        payload,
        "yaml",
        "collection-meta",
        "aisim-collection-meta/v1",
    )
    plan = plan_promotion(root, _request(root, [directory]), _reviewers())
    assert any(f.code == "secret" and f.path == rel for f in plan.findings) == secret
    assert plan.content_ok == (not secret), plan.findings
    shutil.rmtree(root / "mirror")
    report = validate_tree(root, profile="public")
    assert any(f.code == "secret" and f.path == rel for f in report.errors) == secret
    assert report.ok == (not secret), report.errors


@pytest.mark.parametrize("secret", [True, False])
def test_system_yaml_scans_overridden_source_defaults(tmp_path, secret):
    value = "plain-synthetic-value" if secret else "[REDACTED]"
    rel = "systems/synthetic/spec.yaml"
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    path.write_text(f'options:\n  <<: {{password: "{value}"}}\n  password: "[REDACTED]"\n')
    findings = check_publishable(tmp_path, [rel])
    assert any(f.code == "secret" and f.path == rel for f in findings) == secret
    assert bool(findings) == secret
