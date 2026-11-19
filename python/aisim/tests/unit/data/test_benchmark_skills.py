# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dry-run walkthrough of the benchmark skill helpers on fixtures (no GPUs, no network).

plan (catalog) -> record GPU time (sacct and Kubernetes documents) -> store, verify and
attest an eval record. This exercises the deterministic checks the skills call; it does
not replace a run on a real model and GPU.
"""

import hashlib
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

from aisim_core.data import Artifact, _verify
from aisim_core.data.catalog import build_catalog
from aisim_core.data.eval import (
    PredictionIdentity,
    current_records,
    load_record,
    record_dir,
    verify_record,
    write_record,
)
from tests.unit.data.test_catalog import (
    NEW,
    dataset,  # noqa: F401  (fixture)
)
from tests.unit.data.test_eval import _predict
from tests.unit.data.test_eval import built as built_record  # noqa: F401  (fixture)

pytestmark = pytest.mark.unit

SKILLS = Path(__file__).resolve().parents[5] / ".agents" / "skills"
QWEN = "Qwen/Qwen3-32B"


def _script(skill: str, name: str):
    path = SKILLS / skill / "scripts" / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skill_frontmatter_and_links():
    skills = sorted(SKILLS.glob("aisim-benchmark*"))
    assert [s.name for s in skills] == [
        "aisim-benchmark",
        "aisim-benchmark-collect",
        "aisim-benchmark-evaluate",
        "aisim-benchmark-plan",
        "aisim-benchmark-publish",
    ]
    for skill in skills:
        text = (skill / "SKILL.md").read_text(encoding="utf-8")
        assert re.match(rf"---\nname: {skill.name}\ndescription: .+\n", text)
        assert len(text.splitlines()) < 120
        assert (skill / "agents" / "openai.yaml").is_file()
        assert (SKILLS.parents[1] / ".claude" / "skills" / skill.name).is_symlink()


def test_plan_groups_cells_and_budget(built_catalog, capsys):
    root, systems = built_catalog
    plan = _script("aisim-benchmark-plan", "plan_catalog.py")
    assert (
        plan.main(
            [
                "--dataset-root",
                str(root),
                "--hardware",
                "b200_sxm",
                "--model",
                QWEN,
                "--gpus",
                "8",
                "--wall-hours",
                "2",
                "--prediction-fingerprint",
                PredictionIdentity.build("fpm_interpolation", {"p": "b" * 64}, NEW).fingerprint,
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    cells = out["cells"]
    assert {(c["layer"], c["model"]) for c in cells["reuse"]} == {("fpm", QWEN), ("e2e", QWEN)}
    assert out["budget"]["authority"] == "estimated" and out["budget"]["gpu_seconds"] == 8 * 2 * 3600 * 1.25
    assert out["prior_recorded_gpu_seconds"] == 2400.0
    assert plan.main(["--dataset-root", str(root / "missing"), "--hardware", "b200_sxm"]) == 2


def test_plan_rejects_bad_budget_and_reports_unknown_model_and_zero_gpu_time(built_catalog, capsys):
    root, _ = built_catalog
    plan = _script("aisim-benchmark-plan", "plan_catalog.py")
    base = ["--dataset-root", str(root), "--hardware", "b200_sxm"]
    for bad in (
        ["--gpus", "8"],
        ["--wall-hours", "2"],
        ["--gpus", "0", "--wall-hours", "2"],
        ["--gpus", "8", "--wall-hours", "-1"],
        ["--gpus", "8", "--wall-hours", "nan"],
        ["--gpus", "8", "--wall-hours", "2", "--retry-margin", "-1"],
        ["--gpus", "8", "--wall-hours", "2", "--retry-margin", "inf"],
    ):
        with pytest.raises(SystemExit) as exc:
            plan.main([*base, *bad])
        assert exc.value.code == 2
    capsys.readouterr()
    assert plan.main([*base, "--model", "Brand/New-Model"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert {c["layer"] for c in out["not_in_catalog"]} == {"fpm", "e2e"}


@pytest.fixture
def built_catalog(dataset):  # noqa: F811
    root, systems = dataset
    build_catalog(root, systems_dir=systems)
    return root, systems


SACCT_HEADER = "JobID|JobName|State|Submit|Start|End|ElapsedRaw|Elapsed|NNodes|AllocTRES"
SACCT = "\n".join(
    [
        SACCT_HEADER,
        "1|b|FAILED|2026-01-01T00:00:00|2026-01-01T00:00:00|2026-01-01T00:10:00|600|00:10:00|1|gres/gpu=8",
        "2|b|COMPLETED|2026-01-01T00:20:00|2026-01-01T01:00:00|2026-01-01T02:00:00|3600|01:00:00|1|gres/gpu=8",
        "2.batch|batch|COMPLETED|2026-01-01T01:00:00|2026-01-01T01:00:00|2026-01-01T02:00:00|3600|01:00:00|1|gres/gpu=8",
    ]
)


def test_record_gpu_time_slurm_and_unfinished(tmp_path, capsys):
    tool = _script("aisim-benchmark-collect", "record_gpu_time.py")
    sacct = tmp_path / "sacct.txt"
    sacct.write_text(SACCT + "\n")
    assert tool.main(["slurm", "--sacct", str(sacct), "--gpu-type", "b200"]) == 0
    out = json.loads(capsys.readouterr().out)
    gt = out["gpu_time"]
    assert gt["authority"] == "measured" and gt["gpu_seconds"] == 3600 * 8
    assert gt["failed"]["gpu_seconds"] == 600 * 8 and gt["failed"]["count"] == 1
    assert out["findings"]["qualification"] == []

    sacct.write_text(SACCT_HEADER + "\n3|b|RUNNING|x|x|Unknown|10|00:00:10|1|gres/gpu=8\n")
    assert tool.main(["slurm", "--sacct", str(sacct)]) == 2  # not final: no number invented


def test_requeued_attempt_with_same_job_id_is_a_retry_not_the_result():
    from aisim_core.data.gpu_time import gpu_time_from_sacct

    text = "\n".join(
        [
            SACCT_HEADER,
            "7|b|REQUEUED|2026-01-01T00:00:00|2026-01-01T00:00:00|2026-01-01T00:10:00|600|00:10:00|1|gres/gpu=8",
            "7|b|COMPLETED|2026-01-01T00:20:00|2026-01-01T01:00:00|2026-01-01T02:00:00|3600|01:00:00|1|gres/gpu=8",
        ]
    )
    gt = gpu_time_from_sacct(text)
    assert gt.retried is not None and gt.retried.count == 1 and gt.gpu_seconds == 3600 * 8


@pytest.mark.parametrize("container_times", [True, False])
def test_record_gpu_time_kubernetes(tmp_path, capsys, container_times):
    tool = _script("aisim-benchmark-collect", "record_gpu_time.py")
    job = {
        "metadata": {"name": "j", "namespace": "ns", "uid": "job-a"},
        "status": {"succeeded": 1, "startTime": "2025-12-31T23:00:00Z", "completionTime": "2026-01-01T00:30:00Z"},
        "spec": {"template": {"spec": {"containers": [{"resources": {"limits": {"nvidia.com/gpu": "4"}}}]}}},
    }
    pod = {
        "metadata": {"name": "j-abc", "ownerReferences": [{"kind": "Job", "uid": "job-a"}]},
        "spec": {"containers": [{"resources": {"limits": {"nvidia.com/gpu": "4"}}}]},
        "status": {
            "phase": "Succeeded",
            "containerStatuses": [
                {"state": {"terminated": {"startedAt": "2026-01-01T00:00:00Z", "finishedAt": "2026-01-01T00:30:00Z"}}}
            ],
        },
    }
    if not container_times:
        del pod["status"]["containerStatuses"]
    (tmp_path / "job.json").write_text(json.dumps(job))
    (tmp_path / "pods.json").write_text(json.dumps({"items": [pod]}))
    if not container_times:
        assert tool.main(["k8s", "--job", str(tmp_path / "job.json"), "--pods", str(tmp_path / "pods.json")]) == 2
        captured = capsys.readouterr()
        assert not captured.out and "no terminated container times" in captured.err
        return
    assert tool.main(["k8s", "--job", str(tmp_path / "job.json"), "--pods", str(tmp_path / "pods.json")]) == 0
    gt = json.loads(capsys.readouterr().out)["gpu_time"]
    assert gt["gpu_count"] == 4 and gt["gpu_seconds"] == 1800 * 4 and gt["gpu_type"] is None
    pod["metadata"]["ownerReferences"][0]["uid"] = "job-b"
    (tmp_path / "pods.json").write_text(json.dumps({"items": [pod]}))
    assert tool.main(["k8s", "--job", str(tmp_path / "job.json"), "--pods", str(tmp_path / "pods.json")]) == 2
    assert "owner" in capsys.readouterr().err


@pytest.mark.parametrize(
    "job,pods",
    [
        ('{"metadata":', "[]"),
        ("[]", "[]"),
        ('{"metadata":{"uid":"j"}}', "[1]"),
        ('{"metadata":{"uid":"j"}}', '{"items":null}'),
        (None, "[]"),
        ('{"metadata":{"uid":"j"}}', None),
    ],
)
def test_gpu_time_bad_input_returns_status_two(tmp_path, capsys, job, pods):
    tool = _script("aisim-benchmark-collect", "record_gpu_time.py")
    for name, text in (("job", job), ("pods", pods)):
        if text is not None:
            (tmp_path / f"{name}.json").write_text(text)
    assert tool.main(["k8s", "--job", str(tmp_path / "job.json"), "--pods", str(tmp_path / "pods.json")]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.startswith("error:")
    assert tool.main(["slurm", "--sacct", str(tmp_path / "missing.txt")]) == 2


def test_store_verify_attest_walkthrough(built_record, tmp_path, capsys):  # noqa: F811
    root, record = built_record
    path = write_record(root, record)
    tool = _script("aisim-benchmark-evaluate", "check_eval_record.py")

    assert tool.main([str(path)]) == 0  # load + findings only
    info = json.loads(capsys.readouterr().out)
    assert info["tier"] == "submitted" and info["gpu_time_findings"]["qualification"][0]["severity"] == "error"

    good = tmp_path / "good.py"
    good.write_text(
        "from __future__ import annotations\nfrom dataclasses import dataclass\n"
        "@dataclass\nclass Predictor:\n    scale: float = 1.1\n"
        "def predict(inputs):\n"
        "    return inputs['tokens'] * Predictor().scale + (5.0 if inputs['tokens'] == 40 else 0.0)\n"
        f"def fingerprint():\n    return {record.prediction.fingerprint!r}\n"
    )
    assert tool.main([str(path), "--truth-root", str(root), "--predictor", str(good), "--attest-root", str(root)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verification"]["ok"] and out["verified_record_id"]
    assert [r.tier for r in current_records(path.parent)] == ["verified"]

    bad = tmp_path / "bad.py"
    bad.write_text(good.read_text().replace("1.1", "1.5"))
    assert tool.main([str(path), "--truth-root", str(root), "--predictor", str(bad)]) == 1
    assert _predict({"tokens": 10}) == pytest.approx(11.0)


@pytest.mark.parametrize("status", ["partial", "failed"])
def test_incomplete_attempt_is_verified_but_not_attested(built_record, tmp_path, capsys, status):  # noqa: F811
    root, record = built_record
    record = record.model_copy(update={"status": status, "status_reason": "synthetic interrupted attempt"})
    path = write_record(root, record)
    predictor = tmp_path / "predictor.py"
    predictor.write_text(
        "def predict(inputs):\n    return inputs['tokens'] * 1.1 + (5.0 if inputs['tokens'] == 40 else 0.0)\n"
        f"def fingerprint():\n    return {record.prediction.fingerprint!r}\n"
    )
    destination = tmp_path / "attested"
    tool = _script("aisim-benchmark-evaluate", "check_eval_record.py")
    assert (
        tool.main(
            [str(path), "--truth-root", str(root), "--predictor", str(predictor), "--attest-root", str(destination)]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["verification"]["ok"] and out["status"] == status
    assert "verified_record_id" not in out and "verified_record_path" not in out
    assert not destination.exists() and load_record(path) == record


@pytest.mark.parametrize("has_previous", [False, True, None])
def test_failed_predictor_import_restores_module(tmp_path, monkeypatch, has_previous):
    tool = _script("aisim-benchmark-evaluate", "check_eval_record.py")
    previous = None if has_previous is None else object()
    if has_previous is not False:
        monkeypatch.setitem(sys.modules, "benchmark_predictor", previous)
    else:
        monkeypatch.delitem(sys.modules, "benchmark_predictor", raising=False)
    predictor = tmp_path / "broken.py"
    predictor.write_text("raise RuntimeError('synthetic import failure')\n")
    with pytest.raises(RuntimeError, match="synthetic import failure"):
        tool._load_predictor(predictor)
    assert sys.modules.get("benchmark_predictor") is (previous if has_previous else None)
    assert ("benchmark_predictor" in sys.modules) == (has_previous is not False)


def test_attestation_requires_a_predictor_and_copies_the_predecessor(built_record, tmp_path, capsys):  # noqa: F811
    root, record = built_record
    finding_bytes = b'{"reason":"fixture interruption"}'
    (root / "findings.json").write_bytes(finding_bytes)
    finding = Artifact(
        artifact_id="findings",
        path="findings.json",
        role="findings",
        format="json",
        schema="fixture-findings/v1",
        sha256=hashlib.sha256(finding_bytes).hexdigest(),
        bytes=len(finding_bytes),
    )
    previous = record.model_copy(
        update={"status": "failed", "status_reason": "temporary fixture failure", "findings": [finding]}
    )
    write_record(root, previous)
    record = record.model_copy(update={"supersedes": previous.record_id})
    path = write_record(root, record)
    tool = _script("aisim-benchmark-evaluate", "check_eval_record.py")
    dest = tmp_path / "separate-dataset"
    with pytest.raises(SystemExit) as exc:
        tool.main([str(path), "--attest-root", str(dest)])
    assert exc.value.code == 2 and not dest.exists()
    predictor = tmp_path / "predictor.py"
    predictor.write_text(
        "def predict(inputs):\n    return inputs['tokens'] * 1.1 + (5.0 if inputs['tokens'] == 40 else 0.0)\n"
        f"def fingerprint():\n    return {record.prediction.fingerprint!r}\n"
    )
    assert tool.main([str(path), "--truth-root", str(root), "--predictor", str(predictor)]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["verification"]["ok"] and "verified_record_path" not in info
    assert (
        tool.main([str(path), "--truth-root", str(root), "--predictor", str(predictor), "--attest-root", str(dest)])
        == 0
    )
    verified = load_record(json.loads(capsys.readouterr().out)["verified_record_path"])
    assert verified.supersedes == record.record_id
    predecessor = next(dest.rglob(f"{record.record_id}.json"))
    assert load_record(predecessor) == record and path.is_file()
    assert load_record(predecessor.parent / f"{previous.record_id}.json") == previous
    shutil.rmtree(root / "eval")
    for artifact in record.truth.artifacts:
        (root / artifact.path).unlink()
    (root / finding.path).unlink()
    assert _verify(predecessor.parent, finding).read_bytes() == finding_bytes
    assert verify_record(verified, predecessor.parent, _predict, verified.prediction.fingerprint).ok
    for historical in (previous, record):
        assert verify_record(historical, predecessor.parent, _predict, historical.prediction.fingerprint).ok


def test_attestation_rejects_conflicting_destination_artifact(built_record, tmp_path):  # noqa: F811
    root, record = built_record
    path = write_record(root, record)
    from aisim_core.data.eval import record_dir

    destination = tmp_path / "attested"
    target = record_dir(destination, record.truth.identity) / record.truth.artifacts[0].path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"conflicting bytes")
    predictor = tmp_path / "predict.py"
    predictor.write_text(
        "def predict(inputs):\n    return inputs['tokens'] * 1.1 + (5.0 if inputs['tokens'] == 40 else 0.0)\n"
        f"def fingerprint():\n    return {record.prediction.fingerprint!r}\n"
    )
    with pytest.raises(ValueError, match="integrity mismatch"):
        _script("aisim-benchmark-evaluate", "check_eval_record.py").main(
            [str(path), "--truth-root", str(root), "--predictor", str(predictor), "--attest-root", str(destination)]
        )
    assert target.read_bytes() == b"conflicting bytes" and not list(destination.rglob("*.json"))


@pytest.mark.parametrize("same_content", [False, True])
def test_concurrent_attestations_never_replace_published_artifacts(built_record, tmp_path, monkeypatch, same_content):  # noqa: F811
    root, record = built_record
    tool = _script("aisim-benchmark-evaluate", "check_eval_record.py")
    content = b"writer A evidence"
    (root / "finding.json").write_bytes(content)
    finding = Artifact(
        artifact_id="finding", path="finding.json", role="finding", format="json", schema="finding/v1",
        sha256=hashlib.sha256(content).hexdigest(), bytes=len(content),
    )  # fmt: skip
    record = record.model_copy(update={"findings": [finding]})
    first_path = write_record(root, record)
    second_root = tmp_path / "writer-b"
    second_root.mkdir()
    for artifact in record.truth.artifacts:
        shutil.copyfile(root / artifact.path, second_root / artifact.path)
    other_content = content if same_content else b"writer B evidence"
    (second_root / finding.path).write_bytes(other_content)
    other_finding = finding.model_copy(update={"sha256": hashlib.sha256(other_content).hexdigest()})
    other_record = record.model_copy(update={"findings": [other_finding]})
    second_path = write_record(second_root, other_record)
    predictor = tmp_path / "predict.py"
    predictor.write_text(
        "def predict(inputs):\n    return inputs['tokens'] * 1.1 + (5.0 if inputs['tokens'] == 40 else 0.0)\n"
        f"def fingerprint():\n    return {record.prediction.fingerprint!r}\n"
    )
    destination = tmp_path / "attested"
    target = record_dir(destination, record.truth.identity) / finding.path
    original = tool.os.link
    winners = []

    def interleave(staged, final):
        if Path(final) == target and not winners:
            winners.append(other_record.record_id)
            assert (
                tool.main(
                    [
                        str(second_path),
                        "--truth-root",
                        str(second_root),
                        "--predictor",
                        str(predictor),
                        "--attest-root",
                        str(destination),
                    ]
                )
                == 0
            )
        return original(staged, final)

    monkeypatch.setattr(tool.os, "link", interleave)
    args = [
        str(first_path),
        "--truth-root",
        str(root),
        "--predictor",
        str(predictor),
        "--attest-root",
        str(destination),
    ]
    if same_content:
        assert tool.main(args) == 0
    else:
        with pytest.raises(ValueError, match="integrity mismatch"):
            tool.main(args)
    assert winners and target.read_bytes() == other_content
    assert not list(destination.rglob("*.tmp"))
    for path in destination.rglob("*.json"):
        if path == target:
            continue
        published = load_record(path)
        for artifact in [*published.truth.artifacts, *published.findings]:
            _verify(path.parent, artifact)
        assert verify_record(published, path.parent, _predict, published.prediction.fingerprint).ok


def test_planner_requires_current_fingerprint_and_reads_only_selected_layer(built_catalog, capsys):
    root, _ = built_catalog
    plan = _script("aisim-benchmark-plan", "plan_catalog.py")
    record = next(load_record(p) for p in root.glob("eval/**/*.json") if load_record(p).truth.layer == "fpm")
    base = ["--dataset-root", str(root), "--hardware", "b200_sxm", "--model", QWEN, "--layer", "fpm"]
    for fingerprints, reuse in [([], False), (["0" * 64], False), ([record.prediction.fingerprint], True)]:
        argv = [*base, *[v for f in fingerprints for v in ("--prediction-fingerprint", f)]]
        assert plan.main(argv) == 0
        cells = json.loads(capsys.readouterr().out)["cells"]
        assert bool(cells["reuse"]) is reuse
        assert bool(cells["evaluate_only"]) is not reuse
    op = root / "catalog/op-latency/b200_sxm.parquet"
    op.parent.mkdir(exist_ok=True)
    op.write_bytes(b"incompatible schema")
    assert plan.main(base) == 0
    capsys.readouterr()
    with pytest.raises(ValueError, match="Parquet"):
        plan.main(base[:-2])  # an unfiltered plan still consumes op coverage


def test_evaluation_reuse_requires_complete_status_and_matching_behavior(built_record):  # noqa: F811
    root, record = built_record
    for status in ("failed", "partial"):
        previous = record.model_copy(update={"status": status, "status_reason": "temporary unavailable input"})
        path = write_record(root / status, previous)
        current = current_records(path.parent)
        assert previous in current and not [r for r in current if r.status == "complete"]
        retry = record.model_copy(update={"supersedes": previous.record_id})
        write_record(root / status, retry)
        assert retry in current_records(path.parent) and previous not in current_records(path.parent)
    identity = PredictionIdentity.build("fpm_interpolation", {"p": "b" * 64}, "a" * 64, code_version="behavior-v1")
    assert identity == PredictionIdentity.build(
        "fpm_interpolation", {"p": "b" * 64}, "a" * 64, code_version="behavior-v1"
    )
    assert (
        identity.fingerprint
        != PredictionIdentity.build(
            "fpm_interpolation", {"p": "b" * 64}, "a" * 64, code_version="behavior-v2"
        ).fingerprint
    )


def test_partial_accuracy_is_evaluate_only_and_catalog_download_is_sufficient(built_catalog, tmp_path, capsys):
    from aisim_core.data.catalog import Catalog

    root, systems = built_catalog
    for path in root.glob("eval/**/*.json"):
        record = load_record(path)
        if record.truth.layer == "fpm" and record.truth.identity.model == QWEN:
            path.unlink()
            write_record(root, record.model_copy(update={"status": "partial", "status_reason": "interrupted fixture"}))
    build_catalog(root, systems_dir=systems)
    downloaded = tmp_path / "downloaded"
    shutil.copytree(root / "catalog", downloaded / "catalog")
    fingerprint = PredictionIdentity.build("fpm_interpolation", {"p": "b" * 64}, NEW).fingerprint
    assert (
        _script("aisim-benchmark-plan", "plan_catalog.py").main(
            [
                "--dataset-root",
                str(downloaded),
                "--hardware",
                "b200_sxm",
                "--model",
                QWEN,
                "--prediction-fingerprint",
                fingerprint,
            ]
        )
        == 0
    )
    cells = json.loads(capsys.readouterr().out)["cells"]
    assert [r["layer"] for r in cells["reuse"]] == ["e2e"]
    assert [r["layer"] for r in cells["evaluate_only"]] == ["fpm"]
    assert next(r for r in Catalog(downloaded).rows(model=QWEN) if r.layer == "fpm").acc_mape_pct is None
    changed = PredictionIdentity.build("fpm_interpolation", {"p": "b" * 64}, NEW, code_version="new-behavior")
    assert (
        _script("aisim-benchmark-plan", "plan_catalog.py").main(
            [
                "--dataset-root",
                str(downloaded),
                "--hardware",
                "b200_sxm",
                "--model",
                QWEN,
                "--prediction-fingerprint",
                changed.fingerprint,
            ]
        )
        == 0
    )
    assert not json.loads(capsys.readouterr().out)["cells"]["reuse"]
