# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from collector.fpm_forward.native_artifact import _decode_latency_gate, validate_native_collection
from collector.fpm_forward.runtime.trtllm import observer, sitecustomize

from .test_fpm_power import _trtllm_fixture

pytestmark = pytest.mark.unit


def context(phase="decode"):
    return {
        "phase": phase,
        "decode_total_kv_read_tokens": 32768,
        "bundle_sha256": "a" * 64,
        "run_id": "run",
        "grid_digest": "grid",
        "plan_sha256": "plan-sha",
        "collector_attempt_id": "attempt",
        "cell_id": f"fpm-power-{phase}",
    }


class FakeExecutor:
    is_warmup = False
    enable_attention_dp = False
    dist = SimpleNamespace(tp_size=2, pp_size=1)

    def __init__(self, rank):
        self.global_rank = rank
        self.hooks = []

    def _prepare_and_schedule_batch(self):
        self.hooks.append("schedule")
        return "batch", None

    def _collect_scheduled_batch_stats(self, stats):
        self.hooks.append("snapshot")
        return stats

    def _handle_responses(self):
        self.hooks.append("complete")
        return ["response"]

    def shutdown(self):
        self.hooks.append("shutdown")
        return "shutdown complete"


def run(
    tmp_path,
    monkeypatch,
    *,
    bad_rank=None,
    shape_change=False,
    energy_failure=None,
    count=16,
    omit_peer=False,
    phase="decode",
):
    current = [0.0]
    readings = iter((1000, 621_000))
    settings = {**observer.SETTINGS, "max_batch_size": 4}

    def energy(uuids):
        def read():
            if energy_failure in {"nvml_permission", "nvml_error"}:
                raise observer.EnergyError(energy_failure)
            value = next(readings)
            if energy_failure == "counter_wrap":
                value = -value
            elif energy_failure == "counter_not_advancing":
                value = 1000
            return dict.fromkeys(uuids, value)

        return read

    class Executor(FakeExecutor):
        pass

    real_observer = observer.Observer
    monkeypatch.setattr(
        observer, "Observer", lambda *args: real_observer(*args, clock=lambda: current[0], energy=energy)
    )
    monkeypatch.setattr(observer, "time", SimpleNamespace(monotonic=lambda: current[0]))
    monkeypatch.setattr(observer, "device_uuid", lambda executor: f"GPU-{executor.global_rank}")
    monkeypatch.setattr(observer, "effective_settings", lambda _: settings)
    observer.install_hooks(Executor, context(phase), tmp_path)
    ranks = [Executor(rank) for rank in range(2)]
    for i in range(count):
        for executor in ranks:
            executor.iter_counter = i
            # Rank-local jitter leaves ordering and shape intact while failing the 10% gate.
            jitter = max(0, i - 8) * 0.025 if executor.global_rank == bad_rank else 0.0
            current[0] = 1 + i * 0.21 + jitter
            assert executor._prepare_and_schedule_batch() == ("batch", None)
            if omit_peer and i == 5 and executor.global_rank == 1:
                continue
            stats = SimpleNamespace(
                num_ctx_requests=4 if phase == "prefill" else 0,
                num_ctx_tokens=8192 if phase == "prefill" else 0,
                num_ctx_kv_tokens=0,
                num_gen_requests=0 if phase == "prefill" else (3 if shape_change and i == 5 else 4),
                num_gen_kv_tokens=0 if phase == "prefill" else 32764 + 4 * i,
            )
            assert executor._collect_scheduled_batch_stats(stats) is stats
            current[0] += 0.02
            assert executor._handle_responses() == ["response"]
    for executor in reversed(ranks):
        assert executor.shutdown() == "shutdown complete"
        if not omit_peer:
            assert executor.hooks == ["schedule", "snapshot", "complete"] * count + ["shutdown"]
    return json.loads((tmp_path / "benchmark.json").read_text())["results"][0]["fpms"][0]


@pytest.mark.parametrize("bad_rank", [None, 0, 1])
def test_rank_gate_and_collector_contract(tmp_path, monkeypatch, bad_rank):
    plan, cell, cell_dir = _trtllm_fixture(tmp_path)
    result = run(cell_dir / "raw" / "pod-0", monkeypatch, bad_rank=bad_rank)
    collection = validate_native_collection(
        cell,
        cell_dir / "raw",
        expected_backend="trtllm",
        expected_bundle_sha256="a" * 64,
        expected_plan_sha256=plan.sha256,
        expected_attempt_id="attempt",
    )
    (point,) = collection.points
    assert point.rank_wall_times[0][1] == pytest.approx(0.21)
    if bad_rank is None:
        assert result["power"]["window_iterations"] == 10
        assert point.power_w == pytest.approx(620 / 2.1)
    else:
        assert result["power"] == observer.unavailable("latency_drift")
        assert point.power_w is None


def test_prefill_observer_reports_zero_cached_kv(tmp_path, monkeypatch):
    result = run(tmp_path, monkeypatch, phase="prefill")
    assert result["scheduled_requests"] == {
        "num_prefill_requests": 4,
        "sum_prefill_tokens": 8192,
        "sum_prefill_kv_tokens": 0,
        "num_decode_requests": 0,
        "sum_decode_kv_tokens": 0,
    }
    assert result["power"]["status"] == "measured"
    assert result["power"]["kv_drift_tokens"] == 0


@pytest.mark.parametrize("reason", ["nvml_permission", "nvml_error", "counter_wrap", "counter_not_advancing"])
def test_unavailable_energy_keeps_latency(tmp_path, monkeypatch, reason):
    result = run(tmp_path, monkeypatch, energy_failure=reason)
    assert result["wall_time"] == pytest.approx(0.21)
    assert result["power"] == observer.unavailable(reason)


def test_shape_change_keeps_latency(tmp_path, monkeypatch):
    result = run(tmp_path, monkeypatch, shape_change=True)
    assert result["wall_time"] == pytest.approx(0.21)
    assert result["power"] == observer.unavailable("window_shape_mismatch")


@pytest.mark.parametrize(
    ("options", "reason"),
    [({"count": 6}, "window_below_minimum"), ({"omit_peer": True}, "rank_boundary_mismatch")],
)
def test_incomplete_window_keeps_latency(tmp_path, monkeypatch, options, reason):
    result = run(tmp_path, monkeypatch, **options)
    assert result["wall_time"] == pytest.approx(0.21)
    assert result["power"] == observer.unavailable(reason)


@pytest.mark.parametrize(
    ("walls", "repeats"),
    [
        ([1.0] * 10, 10),
        ([1.0] * 5 + [1.1] * 5, 10),
        ([1.0] * 5 + [0.89] * 5, 10),
        ([1.0] * 9, 9),
        ([1.0] * 9, 10),
        ([1.0] * 9 + [float("nan")], 10),
    ],
)
def test_observer_matches_collector_latency_gate(walls, repeats):
    assert observer.latency_gate(walls, repeats) == _decode_latency_gate(walls, repeats)


def test_source_hash_mismatch_fails_before_executor_runs(tmp_path, monkeypatch):
    root = Path(sitecustomize.__file__).parent
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in manifest["files"]}
    bound = hashlib.sha256(observer.canonical({"manifest": manifest, "files": files}).encode()).hexdigest()
    source = tmp_path / "py_executor.py"
    source.write_text("raise AssertionError('executor ran before verification')")
    monkeypatch.setattr(sitecustomize.importlib.metadata, "version", lambda _: "1.3.0rc29")
    with pytest.raises(RuntimeError, match="pinned executor source mismatch"):
        sitecustomize._verify({"bundle_sha256": bound}, manifest_path, source)
