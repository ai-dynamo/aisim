# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate evidence must pin sources and retain each round's timing samples."""

import json
import statistics
import subprocess
import sys
from types import SimpleNamespace

import pytest
import scoped_aggregate_perf as driver
from scoped_aggregate_perf import verify_source


def test_backend_source_matches_pinned_git_blob(tmp_path, monkeypatch):
    source = tmp_path / "python/aisim/src/aisim_core/sdk/backends/base_backend.py"
    source.parent.mkdir(parents=True)
    source.write_text("class BaseBackend: pass\n")
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "fixture"], check=True)
    monkeypatch.chdir(tmp_path)
    revision = verify_source(source, "HEAD")
    assert len(revision) == 40
    source.write_text("class BaseBackend: edited = True\n")
    with pytest.raises(ValueError, match="backend source .* does not match"):
        verify_source(source, revision)
    with pytest.raises(subprocess.CalledProcessError):
        verify_source(source, "missing-revision")


def test_all_rounds_retain_samples_and_recomputable_medians(tmp_path, monkeypatch):
    output = tmp_path / "aggregate.json"
    source = tmp_path / "baseline.py"
    source.write_text(
        "class BaseBackend:\n"
        "    run_agg = None\n"
        "    _run_agg_mixed_step = None\n"
        "    _schedule_qualified_mamba_prefill = None\n"
    )
    monkeypatch.setattr(sys, "argv", [driver.__file__, str(source), "base", "head", str(output)])
    monkeypatch.setattr(driver, "verify_source", lambda path, revision: revision)
    monkeypatch.setattr(driver, "get_model", lambda *args: object())
    monkeypatch.setattr(driver, "get_database_view", lambda *args, **kwargs: object())
    monkeypatch.setattr(driver, "get_backend", lambda *args: SimpleNamespace())
    measurements = []

    def measure(*args):
        index = len(measurements)
        sample = {"miss_us": index % 11 + 1.0, "hit_us": index % 7 + 0.5, "prediction": {"ttft_ms": 42.0}}
        measurements.append(sample)
        return sample

    monkeypatch.setattr(driver, "measure", measure)
    driver.main()
    result = json.loads(output.read_text())
    assert len(result["cases"]) == 4
    cursor = 0
    for case in result["cases"].values():
        cursor += 2  # unrecorded priming calls
        assert len(case["rounds"]) == 6
        for index, row in enumerate(case["rounds"]):
            order = ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
            assert row["order"] == order
            for offset, side in enumerate(order):
                samples = row[side]["samples"]
                assert len(samples) == 30
                assert samples == measurements[cursor + offset : cursor + 60 : 2]
                for metric in ("miss_us", "hit_us"):
                    assert statistics.median(sample[metric] for sample in samples) == row[side][metric]
            cursor += 60
    assert cursor == len(measurements)
