# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit retained replay samples and the identities behind their summaries."""

import gzip
import hashlib
import json
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).parent


def test_canonical_replay_samples_match_report():
    report = json.loads((ROOT / "canonical-mtp-perf.json").read_text())
    companion = ROOT / report["raw_requests"]["path"]
    assert hashlib.sha256(companion.read_bytes()).hexdigest() == report["raw_requests"]["sha256"]
    raw = json.loads(gzip.decompress(companion.read_bytes()))
    assert report["driver_sha256"] == hashlib.sha256((ROOT / "canonical_mtp_perf.py").read_bytes()).hexdigest()
    assert report["host"]["cpu"] and report["host"]["machine"] and report["host"]["python"]
    assert report["env_overrides"] == {"AIC_ALLOW_UNLISTED_VERSIONS": "1"}
    assert len(report["rounds"]) == 6
    used = set()
    for index, record in enumerate(report["rounds"]):
        assert record["order"] == (["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"])
        for side in ("baseline", "candidate"):
            for case in ("nano", "control"):
                result = record["results"][side][case]
                assert result["build"]["revision"] == report["python_revisions"][side]
                assert (
                    result["build"]["native_sha256"] == report["build_verification"]["shared_native"]["native_sha256"]
                )
                assert "--capture-per-request" in result["command"]
                samples = raw[result["requests_sha256"]]
                used.add(result["requests_sha256"])
                assert len(samples) == result["completed_requests"] == 1000
                assert len({sample["uuid"] for sample in samples}) == 1000
                assert all(sample["terminal_status"] == "completed" for sample in samples)
                assert sum(sample["input_length"] for sample in samples) == result["total_input_tokens"] == 2_048_000
                assert sum(sample["output_length"] for sample in samples) == result["total_output_tokens"] == 32_000
                assert statistics.mean(sample["ttft_ms"] for sample in samples) == pytest.approx(
                    result["mean_ttft_ms"], abs=1e-9, rel=0
                )
                assert statistics.mean(sample["itl_ms"] for sample in samples) == pytest.approx(
                    result["mean_tpot_ms"], abs=1e-9, rel=0
                )
        for metric in ("mean_ttft_ms", "mean_tpot_ms", "duration_ms"):
            assert record["results"]["baseline"]["control"][metric] == record["results"]["candidate"]["control"][metric]
    assert used == raw.keys()


def test_qualified_replay_keeps_live_shim_records():
    report = json.loads((ROOT / "repair-replay-4e940e38.json").read_text())
    assert report["driver_sha256"] == hashlib.sha256((ROOT / "repair_replay_perf.py").read_bytes()).hexdigest()
    assert len(report["rounds"]) == 6
    for record in report["rounds"]:
        for side in ("baseline", "candidate"):
            for case in ("nano", "control"):
                result = record["results"][side][case]
                assert result["completed_requests"] == 1000
                assert result["total_input_tokens"] == 2_048_000
                assert result["total_output_tokens"] == 32_000
                identities = result["build"]
                assert identities
                assert all(identity["native_path"] and identity["constructor_path"] for identity in identities)
                assert {identity["native_sha256"] for identity in identities} == {
                    report["build_verification"]["verified_builds"][side]["native_sha256"]
                }
                if case == "nano":
                    assert any(identity["calls"] > 0 for identity in identities)
        assert (
            record["results"]["baseline"]["control"]["duration_ms"]
            == record["results"]["candidate"]["control"]["duration_ms"]
        )
