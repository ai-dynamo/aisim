# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every round must retain the samples underlying its warm-call median."""

import json
import runpy
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace


def test_retains_warm_samples_for_every_rotated_round(tmp_path, monkeypatch):
    responses = []

    def worker(command, **kwargs):
        samples = [1.0, 3.0 + len(responses), 9.0]
        response = {
            "fixture": {
                "value": 12.0,
                "cold_us": 42.0,
                "warm_us": statistics.median(samples),
                "warm_samples_us": samples,
            }
        }
        responses.append(response)
        return SimpleNamespace(stdout=json.dumps(response))

    output = tmp_path / "evidence.json"
    driver = Path(__file__).with_name("affected_perf.py")
    monkeypatch.setattr(sys, "argv", [str(driver), str(tmp_path), "2", str(output), "base", "head"])
    monkeypatch.setattr("subprocess.run", worker)
    runpy.run_path(str(driver), run_name="__main__")
    result = json.loads(output.read_text())
    assert result["orders"] == [["base", "head"], ["head", "base"]]
    for build, indices in (("base", [0, 3]), ("head", [1, 2])):
        saved = result["results"][build]["fixture"]
        expected = [responses[i]["fixture"] for i in indices]
        assert saved["warm_samples_us"] == [row["warm_samples_us"] for row in expected]
        assert saved["warm"] == [statistics.median(samples) for samples in saved["warm_samples_us"]]
        assert saved["cold"] == [row["cold_us"] for row in expected]
        assert saved["value"] == [row["value"] for row in expected]


def test_shim_calls_old_and_new_native_signatures():
    from affected_perf import qualifying_constructor

    def old(name, scale, kernel, phase):
        return name, scale, kernel, phase

    def new(name, scale, kernel, phase, *, state_dtype="unrecorded", token_width=1):
        return state_dtype, token_width

    qualify = {"on": True}
    arguments = ("scan", 1, "selective_state_update", "generation")
    assert qualifying_constructor(old, qualify)(*arguments) == arguments
    wrapped = qualifying_constructor(new, qualify)
    assert wrapped(*arguments) == ("bfloat16", 1)
    assert wrapped(*arguments, state_dtype="float16", token_width=4) == ("float16", 4)
    qualify["on"] = False
    assert wrapped(*arguments) == ("unrecorded", 1)
