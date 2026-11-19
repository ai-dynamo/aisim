# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local consumer contracts over synthetic artifacts, without native prediction."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from aisim.sweeper.config import OptimizationTarget
from aisim.sweeper.score import make_candidate

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "benchmarks/evidence/skills-catalog"
SKILLS = ("aisim-deployment-search", "aisim-gpu-selection-and-sizing")


@pytest.fixture(params=SKILLS)
def summarizer(request):
    path = ROOT / "skills" / request.param / "scripts/summarize_recommendation.py"
    spec = importlib.util.spec_from_file_location("summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def artifact():
    return json.loads((EVIDENCE / "consumer-fixture.json").read_text())


def invoke(module, monkeypatch, tmp_path, data, *args):
    if data is not None:
        (tmp_path / "recommendation.json").write_text(json.dumps(data))
    monkeypatch.setattr(sys, "argv", ["summarize_recommendation.py", str(tmp_path), *args])
    return module.main()


def test_copies_and_release_metric_contract():
    sources = [(ROOT / "skills" / skill / "scripts/summarize_recommendation.py").read_bytes() for skill in SKILLS]
    assert sources[0] == sources[1]
    contract = json.loads((EVIDENCE / "producer-contract.json").read_text())
    # Verify every literal metric read against the inspected, immutable 0.14.0 producer.
    reads = {
        call.args[0].value
        for call in ast.walk(ast.parse(sources[0]))
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "m"
        and call.func.attr == "get"
    }
    assert reads <= set(contract["metric_keys"])


@pytest.mark.parametrize("json_mode", [False, True])
def test_summary_consumes_producer_metrics(summarizer, artifact, monkeypatch, tmp_path, capsys, json_mode):
    ledger = artifact["recommendation"]
    # Exercise the actual checked-out producer, including its metric whitelist.
    produced = make_candidate(
        ledger["candidates"][0]["config"], artifact["prediction_report"], OptimizationTarget.THROUGHPUT_PER_GPU
    )
    ledger["candidates"][0].update(produced.model_dump())
    assert invoke(summarizer, monkeypatch, tmp_path, ledger, *(["--json"] if json_mode else [])) == 0
    out = capsys.readouterr().out
    if json_mode:
        result = json.loads(out)
        row = result["selected"][0]
        for field in ("completed_requests", "sla_compliant_requests", "output_tok_s_per_gpu"):
            assert row[field] == artifact["expected"][field]
        assert result["summary"]["selected_total"] == 1
        assert result["summary"]["not_feasible_total"] == 1
        assert not {"num_requests", "p90_tpot_ms", "p90_ttft_ms", "mean_itl_ms"} & row.keys()
    else:
        assert "completed=6000 sla_ok=5940" in out
        assert "(1000.0/gpu)" in out
        assert "completed=6000/" not in out
        assert "candidate-000002 infeasible sla_constraint" in out


@pytest.mark.parametrize("json_mode", [False, True])
def test_valid_empty_run(summarizer, monkeypatch, tmp_path, capsys, json_mode):
    assert (
        invoke(summarizer, monkeypatch, tmp_path, {"candidates": [], "views": {}}, *(["--json"] if json_mode else []))
        == 0
    )
    if json_mode:
        assert json.loads(capsys.readouterr().out)["selected"] == []


def test_missing_file(summarizer, monkeypatch, tmp_path, capsys):
    assert invoke(summarizer, monkeypatch, tmp_path, None) == 2
    assert "not found" in capsys.readouterr().err


@pytest.mark.parametrize(
    "data",
    [{}, [], {"candidates": [], "views": []}, {"candidates": [{"candidate_id": "candidate-000001"}], "views": {}}],
)
def test_malformed_envelope(summarizer, monkeypatch, tmp_path, capsys, data):
    assert invoke(summarizer, monkeypatch, tmp_path, data, "--json") == 2
    assert "not a valid AISim recommendation" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", None),
        ("status", "unknown"),
        ("status", []),
        ("candidate_id", []),
        ("config", []),
        ("metrics", []),
        ("metrics", {"mean_ttft_ms": "fast"}),
        ("metrics", {"mean_ttft_ms": float("nan")}),
        ("score", "high"),
        ("score", None),
        ("used_gpus", True),
        ("used_gpus", None),
        ("used_gpus", -1),
        ("prediction_config", []),
        ("reason", "unexpected rejection"),
        ("config", {"deployment_mode": []}),
    ],
)
def test_invalid_candidate(summarizer, artifact, monkeypatch, tmp_path, capsys, field, value):
    ledger = artifact["recommendation"]
    ledger["candidates"][0][field] = value
    assert invoke(summarizer, monkeypatch, tmp_path, ledger, "--json", "--best-per-mode") == 2
    assert "not a valid AISim recommendation" in capsys.readouterr().err


@pytest.mark.parametrize(
    "view", [["missing"], ["candidate-000002"], ["candidate-000001", "candidate-000001"], [1], None, "candidate-000001"]
)
def test_invalid_view(summarizer, artifact, monkeypatch, tmp_path, capsys, view):
    ledger = artifact["recommendation"]
    ledger["views"]["top_n"] = view
    assert invoke(summarizer, monkeypatch, tmp_path, ledger, "--json") == 2
    assert "not a valid AISim recommendation" in capsys.readouterr().err


def test_duplicate_ids(summarizer, artifact, monkeypatch, tmp_path):
    ledger = artifact["recommendation"]
    ledger["candidates"].append(copy.deepcopy(ledger["candidates"][0]))
    assert invoke(summarizer, monkeypatch, tmp_path, ledger) == 2


@pytest.mark.parametrize("candidate", ["missing", "candidate-000002"])
def test_ineligible_export(summarizer, artifact, monkeypatch, tmp_path, candidate):
    dest = tmp_path / "export.yaml"
    assert (
        invoke(summarizer, monkeypatch, tmp_path, artifact["recommendation"], "--export", candidate, "--to", str(dest))
        == 2
    )
    assert not dest.exists()


def test_export_requires_destination(summarizer, artifact, monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as exc:
        invoke(summarizer, monkeypatch, tmp_path, artifact["recommendation"], "--export", "candidate-000001")
    assert exc.value.code == 2


def test_export_and_best_per_mode(summarizer, artifact, monkeypatch, tmp_path, capsys):
    ledger = artifact["recommendation"]
    ledger["views"]["top_n"] = []
    assert invoke(summarizer, monkeypatch, tmp_path, ledger, "--json", "--best-per-mode") == 0
    result = json.loads(capsys.readouterr().out)
    assert result["best_per_mode"]["agg"]["candidate_id"] == "candidate-000001"
    assert result["best_per_mode"]["agg"]["config_file"] is None
    dest = tmp_path / "export.yaml"
    assert invoke(summarizer, monkeypatch, tmp_path, ledger, "--export", "candidate-000001", "--to", str(dest)) == 0
    assert yaml.safe_load(dest.read_text()) == ledger["candidates"][0]["prediction_config"]
