# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Passive TP2 observer for the pinned non-overlap TRT-LLM PyExecutor."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass

VERSION = "1.3.0rc29"
BOUNDARY = "trtllm_pyexecutor_completion_interval"
SCHEDULED_KEYS = (
    "num_prefill_requests",
    "sum_prefill_tokens",
    "sum_prefill_kv_tokens",
    "num_decode_requests",
    "sum_decode_kv_tokens",
)
SETTINGS = {
    "enable_iter_perf_stats": True,
    "disable_overlap_scheduler": True,
    "cuda_graph_config": None,
    "enable_block_reuse": False,
    "enable_chunked_prefill": False,
    "max_num_tokens": 8192,
    "beam_width": 1,
    "speculation_enabled": False,
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path, value):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(canonical(value) + "\n")
    temporary.replace(path)


def latency_gate(walls, repeats):
    # Same contract as the collector's _decode_latency_gate. This module also runs
    # in the serving image, where the collector and its dependencies are absent.
    if len(walls) != repeats or any(
        type(wall) not in (float, int) or not math.isfinite(wall) or wall <= 0 for wall in walls
    ):
        return "latency_evidence_invalid"
    if repeats < 10:
        return "latency_evidence_insufficient"
    first, last = statistics.median(walls[:5]), statistics.median(walls[-5:])
    change, bound = abs(last - first), 0.10 * first
    if change > bound and not math.isclose(change, bound, rel_tol=1e-9):
        return "latency_drift"
    return None


def unavailable(reason):
    return dict(
        status="unavailable",
        unavailable_reason=reason,
        gpu_energy=[],
        window_s=None,
        window_iterations=None,
        kv_drift_tokens=None,
        step_wall_times_s=None,
    )


class EnergyError(Exception):
    pass


def device_uuid(executor):
    import torch

    value = getattr(torch.cuda.get_device_properties(executor.device_id), "uuid", None)
    if value is None:
        return None
    value = str(value)
    return value if value.startswith("GPU-") else f"GPU-{value}"


def nvml_reader(uuids):
    """Open NVML once for the two rank-reported UUIDs; each read only samples the counters.

    Initialization and handle lookup stay outside the energy window, so neither
    their idle time nor their latency enters the measured interval.
    """
    try:
        import pynvml as nvml
    except ImportError:
        raise EnergyError("nvml_error") from None

    def translate(error):
        return EnergyError("nvml_permission" if isinstance(error, nvml.NVMLError_NoPermission) else "nvml_error")

    try:
        nvml.nvmlInit()
    except nvml.NVMLError as error:
        raise translate(error) from None
    try:
        handles = {uuid: nvml.nvmlDeviceGetHandleByUUID(uuid) for uuid in uuids}
    except nvml.NVMLError as error:
        nvml.nvmlShutdown()
        raise translate(error) from None

    def read():
        try:
            return {uuid: nvml.nvmlDeviceGetTotalEnergyConsumption(handle) for uuid, handle in handles.items()}
        except nvml.NVMLError as error:
            raise translate(error) from None

    def close():
        try:
            nvml.nvmlShutdown()
        except nvml.NVMLError:
            pass

    read.close = close
    return read


def effective_settings(executor):
    args = executor.llm_args
    return {
        "enable_iter_perf_stats": executor.enable_iter_perf_stats,
        "disable_overlap_scheduler": executor.disable_overlap_scheduler,
        "cuda_graph_config": args.cuda_graph_config,
        "enable_block_reuse": args.kv_cache_config.enable_block_reuse,
        "enable_chunked_prefill": executor.model_engine.attn_runtime_features.chunked_prefill,
        "max_num_tokens": executor.max_num_tokens,
        "max_batch_size": executor.max_batch_size,
        "beam_width": args.max_beam_width,
        "speculation_enabled": executor.model_engine.spec_config is not None,
    }


@dataclass
class Step:
    iteration: int
    completed_s: float
    wall_s: float
    outside_s: float
    scheduled: tuple[int, ...]


class Observer:
    def __init__(self, context, directory, rank, uuid, settings, *, clock=time.monotonic, energy=nvml_reader):
        self.context, self.directory, self.rank, self.clock = context, directory, rank, clock
        self.energy = energy
        self.settings = settings
        if any(type(settings[k]) is not type(v) or settings[k] != v for k, v in SETTINGS.items()):
            raise ValueError("TRT-LLM FPM pinned serving settings mismatch")
        if type(settings["max_batch_size"]) is not int or settings["max_batch_size"] < 4:
            raise ValueError("TRT-LLM FPM requires max_batch_size >= 4")
        self.phase = context["phase"]
        self.target_kv = context["decode_total_kv_read_tokens"] if self.phase == "decode" else 0
        self.window_s = context.get("power_window_s", 2.0)
        self.target_kv_tolerance = context.get("decode_kv_tolerance", 0)
        if type(self.target_kv_tolerance) is not int or not 0 <= self.target_kv_tolerance <= 128:
            raise ValueError("invalid TRT-LLM FPM decode KV tolerance")
        if (
            self.phase not in ("prefill", "decode")
            or rank not in (0, 1)
            or not math.isfinite(self.window_s)
            or self.window_s < 2
        ):
            raise ValueError("invalid TRT-LLM FPM phase, rank or window")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.scope = self.directory / f"trtllm-{hashlib.sha256(canonical(context).encode()).hexdigest()}"
        self.scope.mkdir(exist_ok=True)
        write_json(self.scope / f"device-{rank}.json", {"gpu_uuid": uuid, "serving_config": settings})
        self.steps = []
        self.anchor = self.sample = self.previous = self.snapshot = None
        self.reading = self.finish_reading = None
        self.reader = None
        self.reason = None
        self.finished = False
        self.published = False

    def schedule(self, iteration, stamp):
        self.scheduled = (iteration, stamp)
        self.snapshot = None

    def snapshot_stats(self, stats):
        self.snapshot = (
            stats.num_ctx_requests,
            stats.num_ctx_tokens,
            stats.num_ctx_kv_tokens,
            stats.num_gen_requests,
            stats.num_gen_kv_tokens,
        )

    def matches(self, shape):
        if self.phase == "prefill":
            return shape == (4, 8192, 0, 0, 0)
        return shape is not None and shape[:4] == (0, 0, 0, 4)

    def complete(self, stamp):
        if self.finished:
            self.publish()
            return
        if self.snapshot is None:
            return
        iteration, scheduled = self.scheduled
        previous = self.previous
        self.previous = (iteration, stamp, self.snapshot)
        shape = self.snapshot
        contiguous = (
            previous is not None
            and iteration == previous[0] + 1
            and self.matches(shape)
            and self.matches(previous[2])
            and (self.phase == "prefill" or shape[-1] == previous[2][-1] + 4)
        )
        if not contiguous:
            if self.sample is not None:
                self.finish("window_shape_mismatch")
            return
        step = Step(iteration, stamp, stamp - previous[1], scheduled - previous[1], shape)
        if self.sample is None:
            if self.phase == "decode" and abs(shape[-1] - self.target_kv) > self.target_kv_tolerance:
                return
            if math.isfinite(step.wall_s) and step.wall_s > 0:
                self.sample = step
                if self.rank == 0:
                    # Open NVML in the pass before the anchor: its stall stays out
                    # of the latency sample, the energy window and its step times.
                    self.prepare_energy()
            return
        if self.anchor is None:
            self.anchor = step
            if self.rank == 0:
                self.reading = self.read_energy()
            return
        self.steps.append(step)
        if self.rank == 0:
            start = self.reading["after_s"] if self.reading else self.anchor.completed_s
            if stamp - start >= self.window_s:
                self.finish_reading = self.read_energy()
                self.finish(self.reason)
        elif (self.scope / "rank-0.json").exists():
            self.finish(None)
        elif len(self.steps) >= 4096:
            self.finish("window_below_minimum")

    def prepare_energy(self):
        if self.reader is not None or self.reason:
            return
        try:
            devices = [json.loads((self.scope / f"device-{rank}.json").read_text()) for rank in (0, 1)]
            uuids = [device["gpu_uuid"] for device in devices]
            if len(set(uuids)) != 2 or any(not isinstance(uuid, str) or not uuid.startswith("GPU-") for uuid in uuids):
                raise EnergyError("gpu_mapping_unavailable")
            if any(device["serving_config"] != self.settings for device in devices):
                raise EnergyError("rank_settings_mismatch")
            self.reader = self.energy(uuids)
        except EnergyError as error:
            self.reason = str(error)
        except FileNotFoundError:
            self.reason = "gpu_mapping_unavailable"

    def read_energy(self):
        self.prepare_energy()
        if self.reason or self.reader is None:
            return None
        try:
            # The bracket covers only the counter reads.
            before = self.clock()
            values = self.reader()
            return {"before_s": before, "after_s": self.clock(), "energy_mj": values}
        except EnergyError as error:
            self.reason = str(error)
        return None

    def close_energy(self):
        reader, self.reader = self.reader, None
        if close := getattr(reader, "close", None):
            close()

    def finish(self, reason):
        if self.finished:
            return
        self.finished = True
        self.close_energy()
        write_json(
            self.scope / f"rank-{self.rank}.json",
            {
                "rank": self.rank,
                "sample": asdict(self.sample) if self.sample else None,
                "anchor": asdict(self.anchor) if self.anchor else None,
                "steps": [asdict(step) for step in self.steps],
                "window_out_of_span_s": sum(step.outside_s for step in self.steps),
                "reason": reason,
                "energy_start": self.reading,
                "energy_end": self.finish_reading,
            },
        )
        self.publish()

    def publish(self, *, force=False):
        if self.rank != 0 or not self.finished or self.published:
            return
        lead_path, peer_path = (self.scope / f"rank-{rank}.json" for rank in (0, 1))
        if not lead_path.exists() or (not peer_path.exists() and not force):
            return
        leader = json.loads(lead_path.read_text())
        peer = json.loads(peer_path.read_text()) if peer_path.exists() else None
        if leader["sample"] is None:
            return
        power = qualify(self.phase, leader, peer)
        write_json(self.directory / "benchmark.json", artifact(self.context, self.settings, leader, power))
        write_json(
            self.directory / "collector-provenance.json",
            {
                "schema_name": "aic_fpm_collector_provenance",
                "schema_version": 1,
                "cell_id": self.context["cell_id"],
                "plan_sha256": self.context["plan_sha256"],
                "attempt_id": self.context["collector_attempt_id"],
                "runtime": {"backend": "trtllm", "backend_version": VERSION},
            },
        )
        write_json(self.scope / "published.json", {"power": power})
        self.published = True


def qualify(phase, leader, peer):
    if leader["reason"]:
        return unavailable(leader["reason"])
    steps = leader["steps"]
    anchor = leader["anchor"]
    if not anchor or not steps or not leader["energy_start"] or not leader["energy_end"]:
        return unavailable("window_below_minimum")
    if (
        not peer
        or not peer["anchor"]
        or peer["anchor"]["iteration"] != anchor["iteration"]
        or peer["anchor"]["scheduled"] != anchor["scheduled"]
        or not peer["sample"]
        or peer["sample"]["iteration"] != leader["sample"]["iteration"]
        or peer["sample"]["scheduled"] != leader["sample"]["scheduled"]
    ):
        return unavailable("rank_boundary_mismatch")
    repeats = len(steps)
    peer_steps = peer["steps"][:repeats]
    if len(peer_steps) != repeats or any(
        a["iteration"] != b["iteration"] or a["scheduled"] != b["scheduled"]
        for a, b in zip(steps, peer_steps, strict=True)
    ):
        return unavailable("rank_boundary_mismatch")
    if phase == "decode":
        for rank_steps in (steps, peer_steps):
            if reason := latency_gate([step["wall_s"] for step in rank_steps], repeats):
                return unavailable(reason)
    start, end = leader["energy_start"], leader["energy_end"]
    duration = end["before_s"] - start["before_s"]
    if end["before_s"] - start["after_s"] < 2:
        return unavailable("window_below_minimum")
    if len(start["energy_mj"]) != 2 or start["energy_mj"].keys() != end["energy_mj"].keys():
        return unavailable("gpu_mapping_unavailable")
    energy = []
    for uuid, begin in start["energy_mj"].items():
        delta = end["energy_mj"][uuid] - begin
        if delta <= 0:
            return unavailable("counter_wrap" if delta < 0 else "counter_not_advancing")
        energy.append({"gpu_uuid": uuid, "energy_wms": delta})
    return dict(
        status="measured",
        unavailable_reason=None,
        gpu_energy=energy,
        window_s=duration,
        window_iterations=repeats,
        kv_drift_tokens=steps[-1]["scheduled"][-1] - anchor["scheduled"][-1],
        step_wall_times_s=[step["wall_s"] for step in steps],
    )


def artifact(context, settings, leader, power):
    phase, sample = context["phase"], leader["sample"]
    scheduled = dict(zip(SCHEDULED_KEYS, sample["scheduled"], strict=True))
    point = dict(
        benchmark_id=1,
        point_type=phase,
        batch_size=4,
        total_prefill_tokens=scheduled["sum_prefill_tokens"],
        total_kv_read_tokens=scheduled["sum_decode_kv_tokens"] + scheduled["sum_prefill_kv_tokens"],
        sample_reasons=[],
    )
    wall = sample["wall_s"]
    fpm = dict(counter_id=1, dp_rank=0, wall_time=wall, scheduled_requests=scheduled, power=power)
    return dict(
        schema_version=3,
        artifact_type="rank",
        status="complete",
        valid=True,
        usable=True,
        timing_valid=True,
        stop_reason=None,
        error=None,
        run_id=context["run_id"],
        grid_digest=context["grid_digest"],
        timing_boundary=BOUNDARY,
        producer={"trtllm_package_version": VERSION, "bundle_sha256": context["bundle_sha256"]},
        serving_config=settings,
        config={"mode": phase, "power_window_s": context.get("power_window_s", 2.0)},
        coverage={"expected_points": 1, "completed_points": 1, "skipped_points": 0},
        dp={"rank": 0, "size": 1},
        results=[{"point": point, "fpms": [fpm]}],
        iteration_groups=[
            dict(
                benchmark_id=1,
                point=point,
                expected_dp_ranks=[0],
                complete=True,
                wall_time=wall,
                rank_results=[{"dp_rank": 0, "fpms": [fpm]}],
            )
        ],
        skipped_points=[],
        missing_phases=[],
        timing={
            "benchmark_elapsed_seconds": leader["steps"][-1]["completed_s"] - sample["completed_s"] + wall
            if leader["steps"]
            else wall,
            "measured_iteration_seconds": wall,
        },
    )


def install_hooks(executor_class, context, directory):
    schedule = executor_class._prepare_and_schedule_batch
    snapshot = executor_class._collect_scheduled_batch_stats
    complete = executor_class._handle_responses
    shutdown = executor_class.shutdown

    def schedule_hook(self, *args, **kwargs):
        result = schedule(self, *args, **kwargs)
        if not self.is_warmup:
            observer = getattr(self, "_aisim_trtllm_observer", None)
            if observer is None:
                if (self.dist.tp_size, self.dist.pp_size, self.enable_attention_dp) != (2, 1, False):
                    raise ValueError("TRT-LLM FPM requires TP2/DP1")
                observer = self._aisim_trtllm_observer = Observer(
                    context, directory, self.global_rank, device_uuid(self), effective_settings(self)
                )
            observer.schedule(self.iter_counter, time.monotonic())
        return result

    def snapshot_hook(self, *args, **kwargs):
        result = snapshot(self, *args, **kwargs)
        if observer := getattr(self, "_aisim_trtllm_observer", None):
            observer.snapshot_stats(result)
        return result

    def complete_hook(self, *args, **kwargs):
        result = complete(self, *args, **kwargs)
        if observer := getattr(self, "_aisim_trtllm_observer", None):
            observer.complete(time.monotonic())
        return result

    def shutdown_hook(self, *args, **kwargs):
        result = shutdown(self, *args, **kwargs)
        if observer := getattr(self, "_aisim_trtllm_observer", None):
            observer.finish("window_below_minimum")
            observer.publish(force=observer.rank == 0)
        return result

    executor_class._prepare_and_schedule_batch = schedule_hook
    executor_class._collect_scheduled_batch_stats = snapshot_hook
    executor_class._handle_responses = complete_hook
    executor_class.shutdown = shutdown_hook
