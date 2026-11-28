# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw worker evidence must survive the benchmark driver's summary fold."""

import json
import runpy
import statistics
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def test_retains_complete_responses_for_every_rotated_round(tmp_path, monkeypatch):
    cases = [{"case_id": "fixture", "model_id": "qwen3-32b", "database_mode": "SILICON"}]
    module = ModuleType("tools.forward_perf_gate.cases")
    module.expand_cases = lambda: cases
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.delenv("HOST_PERF_MODELS", raising=False)
    monkeypatch.delenv("HOST_PERF_MODES", raising=False)
    responses = []
    requests = []

    def worker(command, **kwargs):
        request = json.loads(Path(command[-1]).read_text())
        requests.append(request)
        samples = [1.0, 3.0, 8.0 + len(responses)]
        response = {
            "protocol_version": 2,
            "revision": request["revision"],
            "extra_envelope_field": "preserved",
            "results": [
                {
                    "case_id": "fixture",
                    "case": cases[0],
                    "case_hash": "fixture-hash",
                    "status": "OK",
                    "predicted_value_ms": 10.0 + len(responses),
                    "environment": {"python": "fixture", "cpu_affinity": [2, 3]},
                    "cold_us": 42.0,
                    "warm_samples_us": samples,
                    "warm": {"call_median_us": statistics.median(samples)},
                }
            ],
        }
        responses.append(response)
        return SimpleNamespace(stdout=json.dumps(response))

    output = tmp_path / "evidence.json"
    driver = Path(__file__).with_name("host_perf.py")
    monkeypatch.setattr(sys, "argv", [str(driver), str(tmp_path), "2", str(output), "base", "head"])
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr("subprocess.run", worker)
    runpy.run_path(str(driver), run_name="__main__")
    result = json.loads(output.read_text())
    assert result["orders"] == [["base", "head"], ["head", "base"]]
    assert [run["response"] for run in result["worker_runs"]] == responses
    assert [run["request"] for run in result["worker_runs"]] == requests
    assert [(run["round"], run["build"]) for run in result["worker_runs"]] == [
        (0, "base"),
        (0, "head"),
        (1, "head"),
        (1, "base"),
    ]
    for build in ("base", "head"):
        saved = [run["response"]["results"][0] for run in result["worker_runs"] if run["build"] == build]
        assert result["results"][build]["fixture"]["warm"] == [statistics.median(c["warm_samples_us"]) for c in saved]
        assert result["results"][build]["fixture"]["cold"] == [c["cold_us"] for c in saved]
    # Every worker succeeded, but their predictions differ. That does not
    # establish the correctness or equivalence of either revision.
    assert result["output_equivalence"]["status"] == "not_checked"


@pytest.mark.parametrize("variable,value", [("HOST_PERF_MODELS", "missing"), ("HOST_PERF_MODES", "missing")])
def test_empty_selection_exits_with_selector_reason_before_workers(tmp_path, monkeypatch, variable, value):
    module = ModuleType("tools.forward_perf_gate.cases")
    module.expand_cases = lambda: [{"model_id": "qwen3-32b", "database_mode": "SILICON"}]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.delenv("HOST_PERF_MODELS", raising=False)
    monkeypatch.delenv("HOST_PERF_MODES", raising=False)
    monkeypatch.setenv(variable, value)
    driver = Path(__file__).with_name("host_perf.py")
    output = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", [str(driver), str(tmp_path), "2", str(output), "base", "head"])
    monkeypatch.setattr(sys, "path", sys.path.copy())

    def worker(*args, **kwargs):
        pytest.fail("no worker may start for an empty selection")

    monkeypatch.setattr("subprocess.run", worker)
    with pytest.raises(SystemExit, match="No cases match HOST_PERF_MODELS=.*HOST_PERF_MODES="):
        runpy.run_path(str(driver), run_name="__main__")
    assert not output.exists()
