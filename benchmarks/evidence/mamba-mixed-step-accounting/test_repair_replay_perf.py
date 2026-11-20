# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The evidence driver must retain every child process's qualification record."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace


def test_multiple_process_records_remain_linked_to_each_prediction(tmp_path, monkeypatch):
    records = [
        {"calls": 4, "native_sha256": "fixture-binary", "native_sources_sha256": "fixture-source"},
        {"calls": 0, "native_sha256": "fixture-binary", "native_sources_sha256": "fixture-source"},
    ]

    def cli(command, *, env, **kwargs):
        dest = Path(command[command.index("--output-dir") + 1])
        dest.mkdir()
        report = {
            "completed_requests": 1000,
            "total_input_tokens": 2_048_000,
            "total_output_tokens": 32_000,
            "mean_ttft_ms": 2,
            "mean_tpot_ms": 1,
            "duration_ms": 100,
            "wall_time_ms": 10,
        }
        (dest / "prediction.json").write_text(json.dumps(report))
        (dest / "resource-runtime.json").write_text(json.dumps({"peak_observed_rss_bytes": 1024}))
        Path(env["REPAIR_BUILD_RECORD"]).write_text("".join(json.dumps(record) + "\n" for record in records))
        return SimpleNamespace(returncode=0)

    output = tmp_path / "evidence.json"
    driver = Path(__file__).with_name("repair_replay_perf.py")
    monkeypatch.setattr(sys, "argv", [str(driver), sys.executable, str(tmp_path), "2", str(output)])
    monkeypatch.setattr("subprocess.run", cli)
    runpy.run_path(str(driver), run_name="__main__")
    result = json.loads(output.read_text())
    assert len(result["rounds"]) == 2
    for round_result in result["rounds"]:
        for build in ("baseline", "candidate"):
            for case in ("nano", "control"):
                assert round_result["results"][build][case]["build"] == records
