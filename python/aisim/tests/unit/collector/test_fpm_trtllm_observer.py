# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from collector.fpm_forward.native_artifact import _decode_latency_gate, validate_native_collection
from collector.fpm_forward.runtime.trtllm import observer, sitecustomize
from collector.fpm_forward.runtime_observations import _validate_import_binding

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
        self.model_engine = SimpleNamespace(_warmup_timer=SimpleNamespace(purpose="final_executor"))

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


def test_profiling_executor_is_not_observed(tmp_path, monkeypatch):
    class Executor(FakeExecutor):
        pass

    observer.install_hooks(Executor, context(), tmp_path)
    executor = Executor(0)
    executor.model_engine._warmup_timer.purpose = "memory_profiling"
    monkeypatch.setattr(observer, "device_uuid", lambda _: pytest.fail("profiling used the observer"))
    assert executor._prepare_and_schedule_batch() == ("batch", None)
    assert not hasattr(executor, "_aisim_trtllm_observer")


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
    setup_cost=0.0,
    read_cost=0.0,
    setups=None,
    shutdown=True,
    extra=None,
):
    current = [0.0]
    readings = iter((1000, 621_000))
    settings = {**observer.SETTINGS, "max_batch_size": 4}

    def energy(uuids):
        # Model NVML init and handle lookup as wall time on the observer's clock.
        current[0] += setup_cost
        if setups is not None:
            setups.append(("setup", observer_steps()))

        def read():
            current[0] += read_cost
            if setups is not None:
                setups.append(("read", observer_steps()))
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
    observer.install_hooks(Executor, {**context(phase), **(extra or {})}, tmp_path)
    ranks = [Executor(rank) for rank in range(2)]

    def observer_steps():
        leader = ranks[0]._aisim_trtllm_observer
        return ("anchor" if leader.anchor else "sample" if leader.sample else "none", len(leader.steps))

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
    if shutdown:
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


def test_nvml_setup_stays_outside_the_energy_window(tmp_path, monkeypatch):
    setups = []
    result = run(tmp_path, monkeypatch, setup_cost=0.5, setups=setups)
    (leader,) = (json.loads(path.read_text()) for path in tmp_path.glob("trtllm-*/rank-0.json"))
    # NVML opens once, in the sample pass; the anchor and closing passes only read counters.
    assert setups == [("setup", ("sample", 0)), ("read", ("anchor", 0)), ("read", ("anchor", 10))]
    for reading in (leader["energy_start"], leader["energy_end"]):
        assert reading["after_s"] == reading["before_s"]
    assert result["power"]["status"] == "measured"
    assert result["power"]["window_iterations"] == 10


def test_nvml_reader_initializes_once(monkeypatch):
    calls = []

    class NVMLError(Exception):
        pass

    fake = SimpleNamespace(
        NVMLError=NVMLError,
        NVMLError_NoPermission=type("NVMLError_NoPermission", (NVMLError,), {}),
        nvmlInit=lambda: calls.append("init"),
        nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlDeviceGetHandleByUUID=lambda uuid: calls.append("handle") or uuid,
        nvmlDeviceGetTotalEnergyConsumption=lambda handle: calls.append("read") or len(handle),
    )
    monkeypatch.setitem(__import__("sys").modules, "pynvml", fake)
    read = observer.nvml_reader(["GPU-a", "GPU-bb"])
    assert read() == read() == {"GPU-a": 5, "GPU-bb": 6}
    read.close()
    assert calls == ["init", "handle", "handle", "read", "read", "read", "read", "shutdown"]


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
    [
        ({"count": 6}, "window_below_minimum"),
        ({"count": 6, "shutdown": False}, "window_below_minimum"),
        ({"omit_peer": True}, "rank_boundary_mismatch"),
    ],
)
def test_incomplete_window_keeps_latency(tmp_path, monkeypatch, options, reason):
    result = run(tmp_path, monkeypatch, **options)
    assert result["wall_time"] == pytest.approx(0.21)
    assert result["power"] == observer.unavailable(reason)


def test_valid_window_publishes_power_without_shutdown(tmp_path, monkeypatch):
    # The leader publishes once the peer's record exists, with no shutdown callback.
    result = run(tmp_path, monkeypatch, shutdown=False)
    assert result["wall_time"] == pytest.approx(0.21)
    assert result["power"]["status"] == "measured"
    assert result["power"]["window_iterations"] == 10


@pytest.mark.parametrize(("window", "iterations"), [(None, 10), (2.5, 12)])
def test_power_window_sets_and_records_the_window(tmp_path, monkeypatch, window, iterations):
    extra = {} if window is None else {"power_window_s": window}
    result = run(tmp_path, monkeypatch, extra=extra)
    assert result["power"]["status"] == "measured"
    assert result["power"]["window_iterations"] == iterations
    config = json.loads((tmp_path / "benchmark.json").read_text())["config"]
    assert config == {"mode": "decode", "power_window_s": 2.0 if window is None else window}


def first_batch(tmp_path, monkeypatch, *, extra=None, settings=None, dist=(2, 1), attention_dp=False):
    class Executor(FakeExecutor):
        enable_attention_dp = attention_dp

    Executor.dist = SimpleNamespace(tp_size=dist[0], pp_size=dist[1])
    monkeypatch.setattr(observer, "device_uuid", lambda _: "GPU-0")
    monkeypatch.setattr(
        observer, "effective_settings", lambda _: {**observer.SETTINGS, "max_batch_size": 4, **(settings or {})}
    )
    observer.install_hooks(Executor, {**context(), **(extra or {})}, tmp_path)
    executor = Executor(0)
    executor.iter_counter = 0
    return executor._prepare_and_schedule_batch()


@pytest.mark.parametrize("window", [float("nan"), float("inf"), 1.99])
def test_invalid_power_window_raises_at_the_first_batch(tmp_path, monkeypatch, window):
    with pytest.raises(ValueError, match="rank or window"):
        first_batch(tmp_path, monkeypatch, extra={"power_window_s": window})


@pytest.mark.parametrize("tolerance", [0, 128])
def test_decode_kv_tolerance_accepts_its_bounds(tmp_path, monkeypatch, tolerance):
    assert first_batch(tmp_path, monkeypatch, extra={"decode_kv_tolerance": tolerance}) == ("batch", None)


@pytest.mark.parametrize("tolerance", [129, -1, 2.0, True])
def test_decode_kv_tolerance_rejects_other_values(tmp_path, monkeypatch, tolerance):
    with pytest.raises(ValueError, match="decode KV tolerance"):
        first_batch(tmp_path, monkeypatch, extra={"decode_kv_tolerance": tolerance})


def test_off_target_decode_coordinate_is_recorded(tmp_path, monkeypatch):
    plan, cell, cell_dir = _trtllm_fixture(tmp_path)
    # The first contiguous decode step reads 32,768 KV tokens, 32 below the target.
    extra = {"decode_total_kv_read_tokens": 32800, "decode_kv_tolerance": 128}
    run(cell_dir / "raw" / "pod-0", monkeypatch, extra=extra)
    collection = validate_native_collection(
        cell,
        cell_dir / "raw",
        expected_backend="trtllm",
        expected_bundle_sha256="a" * 64,
        expected_plan_sha256=plan.sha256,
        expected_attempt_id="attempt",
    )
    (point,) = collection.points
    assert point.point["total_kv_read_tokens"] == 32768
    assert point.power_w == pytest.approx(620 / 2.1)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"dist": (4, 1)}, "TP2/DP1"),
        ({"dist": (2, 2)}, "TP2/DP1"),
        ({"attention_dp": True}, "TP2/DP1"),
        ({"settings": {"max_batch_size": 3}}, "max_batch_size"),
        ({"settings": {"max_batch_size": "4"}}, "max_batch_size"),
    ],
)
def test_serving_topology_and_batch_gates(tmp_path, monkeypatch, options, message):
    with pytest.raises(ValueError, match=message):
        first_batch(tmp_path, monkeypatch, **options)


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


def load_executor(tmp_path, monkeypatch, verify):
    """Run the bundle's import loader on a stub executor; return the import receipt."""
    root = Path(sitecustomize.__file__).parent
    source = tmp_path / "py_executor.py"
    source.write_text(
        "class PyExecutor:\n"
        "    def _prepare_and_schedule_batch(self): pass\n"
        "    def _collect_scheduled_batch_stats(self, stats): return stats\n"
        "    def _handle_responses(self): pass\n"
        "    def shutdown(self): pass\n"
    )
    monkeypatch.setattr(sitecustomize, "_verify", verify)
    session = {**context(), "attempt_id": "attempt", "configuration": "tp2"}
    (tmp_path / "context.json").write_text(json.dumps(session))
    monkeypatch.setenv("AISIMULATE_RUNTIME_CONTEXT", str(tmp_path / "context.json"))
    monkeypatch.setenv("AISIMULATE_RUNTIME_INSTRUMENTATION", str(root / "manifest.json"))
    monkeypatch.setenv("AISIMULATE_RUNTIME_OBSERVATION_DIR", str(tmp_path / "out"))
    monkeypatch.delitem(sys.modules, "_aisim_trtllm_fpm_observer", raising=False)
    module = ModuleType(sitecustomize._TARGET)
    module.__spec__ = SimpleNamespace(origin=str(source))
    try:
        sitecustomize._ExecutorLoader().exec_module(module)
    finally:
        receipts = list((tmp_path / "out").glob("runtime-instrumentation-imports-*.json"))
        assert len(receipts) == 1
    return json.loads(receipts[0].read_text()), module


def test_import_receipt_is_not_the_collector_binding(tmp_path, monkeypatch):
    root = Path(sitecustomize.__file__).parent
    files = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in json.loads((root / "manifest.json").read_text())["files"]
    }
    receipt, module = load_executor(
        tmp_path, monkeypatch, lambda context, manifest, path: (path.read_bytes(), files, "e" * 64)
    )
    assert receipt["status"] == "verified"
    assert module.PyExecutor._prepare_and_schedule_batch.__name__ == "schedule_hook"
    assert receipt["schema_version"] == sitecustomize.RECEIPT_SCHEMA
    assert receipt["imports"][sitecustomize._TARGET]["sha256"] == "e" * 64
    collector_binding = {
        "schema_version": "aisimulate-runtime-instrumentation-imports/v1",
        "loader_sha256": receipt["loader_sha256"],
    }
    with pytest.raises(ValueError, match="instrumentation import schema_version mismatch"):
        _validate_import_binding({"instrumentation_binding": receipt}, collector_binding, None)


def test_failed_verification_writes_a_failed_receipt_and_raises(tmp_path, monkeypatch):
    def verify(context, manifest, path):
        raise RuntimeError("TRT-LLM pinned executor source mismatch: expected a, got b")

    with pytest.raises(RuntimeError, match="pinned executor source mismatch"):
        load_executor(tmp_path, monkeypatch, verify)
    (path,) = (tmp_path / "out").glob("runtime-instrumentation-imports-*.json")
    receipt = json.loads(path.read_text())
    assert receipt["status"] == "failed"
    assert receipt["error"] == "RuntimeError: TRT-LLM pinned executor source mismatch: expected a, got b"
    assert receipt["schema_version"] == sitecustomize.RECEIPT_SCHEMA
    assert receipt["imports"] == {}
    assert "_aisim_trtllm_fpm_observer" not in sys.modules


def test_missing_executor_source_raises_import_error(monkeypatch):
    monkeypatch.setattr(sitecustomize.importlib.machinery.PathFinder, "find_spec", lambda *args: None)
    finder = sitecustomize._ExecutorFinder()
    assert finder.find_spec("json") is None
    with pytest.raises(ImportError, match="executor source is missing"):
        finder.find_spec(sitecustomize._TARGET)
