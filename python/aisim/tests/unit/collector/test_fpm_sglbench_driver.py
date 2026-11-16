# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# The fake engine and the batch-sequence checks follow SGLang's scheduler at v0.5.22 and v0.5.23; see
# THIRD_PARTY_NOTICES.md.
# Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d
# Source: https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90
# The run lifecycle, the point procedures and five failure reason strings are re-implemented from ai-dynamo/dynamo
# (Apache-2.0, https://github.com/ai-dynamo/dynamo), components/src/dynamo/vllm/instrumented_scheduler.py at
# b20fe6fe94021b854597965c7305931175e8f482. See THIRD_PARTY_NOTICES.md.

"""CPU-only test of ``runtime/sglbench_driver.py`` against a FAKE engine and AISim's two real checkers.

No SGLang and no GPU are involved. Everything that stands in for an engine is FAKE and is named so:

- ``_FakeEngine`` answers ``generate`` with cached-token counts from ``PageCacheModel`` (the model of SGLang's
  prefix cache written in ``test_fpm_sglbench_content.py``) and appends the batch lines of each call to FAKE
  recorder files. It writes those files itself, to record format 1; ``sglbench_recorder.py`` is not used here.
  Its batches follow a FAKE scheduler: extend batches filled up to the chunk budget, then one decode batch for
  every further output token.
- Every time is FAKE. ``_FakeClock`` counts whole ticks of 2**-20 s and moves only when the fake engine runs a
  batch or the driver sleeps. The durations come from ``FAKE_*`` constants. They were typed in to exercise the
  arithmetic and were never measured. The artifacts say so in ``producer``, among the facts of the launcher;
  the few tests that make the driver leave those facts out hold no other claim about the times either.
- A fault is something the fake engine does on purpose to one call: a wrong batch line, a wrong response, a
  dropped cache.

The driver, the grid, the content plan and the artifact module are the real ones, loaded flat from the runtime
directory as the pod loads them. The two checkers are used as shipped, through the helpers of
``test_fpm_sglbench_artifact.py``: the validator (``validate_native_collection``) and the gate (the Python text
inside ``check_result_files()`` of ``runtime/fpm_exec.sh``).
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import inspect
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest

from .test_fpm_runner import _cell
from .test_fpm_sglbench_artifact import GATE_ACCEPT, GATE_INVALID, _gate, _pod, _validate
from .test_fpm_sglbench_content import PageCacheModel

RUNTIME = Path(__file__).resolve().parents[3] / "collector/fpm_forward/runtime"
# In the pod the runtime files lie flat in one directory and import each other by bare name. The directory goes
# to the END of sys.path, so that it shadows nothing.
if str(RUNTIME) not in sys.path:
    sys.path.append(str(RUNTIME))

import sglbench_artifact as artifact
import sglbench_content as content
import sglbench_driver as driver
import sglbench_grid as grid

pytestmark = pytest.mark.unit

# FAKE run inputs.
TOKEN_LOW, TOKEN_HIGH = 1000, 51_000
FAKE_RUN_ID = "fa4e0123456789abcdef0123456789ab"
FAKE_RUN_TAG = "fa4e0123"  # the first eight characters of the run id
FAKE_WORKER_ID = "fake-cell-node0"
FAKE_FIRST_PID = 4100
LEADER_LOG = "sched-4100.jsonl"
FOLLOWER_LOG = "sched-4101.jsonl"
FAKE_FOREIGN_RID = "fake-foreign-request"
FAKE_TIMEOUT_SECONDS = 10_800.0

# FAKE timings, in clock ticks of 2**-20 s. An extend batch takes a base and one tick per computed token. Decode
# batch i of a call takes (base + requests) * tenths[i]: a short first step, a short last step and unsorted kept
# steps. Decode steps 3 to 5 are the items 2, 3 and 4, and their upper median is the item 4, ten tenths.
TICKS_PER_SECOND = 1 << 20
FAKE_EXTEND_BASE_TICKS = 4096
FAKE_DECODE_BASE_TICKS = 300
FAKE_DECODE_STEP_TENTHS = (6, 15, 12, 9, 10, 3)
# FAKE overlap: every batch of a call but its first is launched this long before the batch ahead of it has
# delivered its output, as overlap scheduling does. The time from the launch of such a batch to its output is
# then longer than the step between two outputs.
FAKE_OVERLAP_LEAD_TICKS = 250
# FAKE: the extend batch of a warm shot takes this much longer than that of the measured shot of the same shape,
# so a sample taken from the wrong shot would show.
FAKE_WARM_SHOT_EXTRA_TICKS = 500
# The driver never sleeps this often in a test; a wait without end fails the test instead of hanging it.
FAKE_SLEEP_LIMIT = 10_000

# The run-invariant settings a launcher would hand over, and its own facts for the producer block.
FAKE_CONFIG = {"warmup_iterations": 5, "timeout": 10_800, "points_source": "generated", "content_seed": "0"}
FAKE_PRODUCER = {
    "timing_source": "FAKE: typed into test_fpm_sglbench_driver.py, never measured",
    "engine": "FAKE: a page-cache model and hand-written recorder lines, no SGLang",
}

# Few values on every axis: most tests need a run, not a dense grid.
SMALL_CAPS = grid.SampleCaps(
    prefill_max_new_token_samples=3,
    prefill_max_kv_read_token_samples=3,
    decode_max_kv_read_token_samples=3,
    decode_max_batch_size_samples=3,
    prefix_max_batch_size_samples=2,
)

# The order of the calls outside any point, which the request ids carry in place of the attempt.
FREE_BATCH_SHAPE, FREE_ALIGNED_SEED, FREE_ALIGNED_SHOTS, FREE_UNALIGNED_SEED, FREE_UNALIGNED_AGAIN = range(5)
FREE_SEVEN_TOKEN_RUN, FREE_WARMUP = 5, 6


def _limits(page_size: int, **overrides) -> grid.Limits:
    """Small FAKE limits that grow with the page size and admit the 256-token warm-up prompt.

    Page 1: I = 428, S = 1184, R = 9, M = 80, C = 96. Page 64: I = 812, S = 2304, R = 9, M = 320, C = 384.
    """
    unit = grid.cached_unit(page_size)
    record = {
        "page_size": page_size,
        "max_req_input_len": 8 * unit + 300,
        "max_total_num_tokens": grid.floor_pages(24 * unit + 800, page_size),
        "max_running_requests": 9,
        "max_prefill_tokens": 5 * unit,
        "chunked_prefill_size": 6 * unit,
        "decode_graph_sizes": (1, 2, 4, 8),
        "prefill_graph_sizes": (4, 8, 16, 32, 64),
    }
    record.update(overrides)
    return grid.Limits(**record)


# ---------------------------------------------------------------------------------------------------------------
# Fake clock, fake recorder files, fake engine
# ---------------------------------------------------------------------------------------------------------------


class _FakeClock:
    """A FAKE monotonic clock in whole ticks of 2**-20 s, so the difference of two readings is exact."""

    def __init__(self) -> None:
        self.ticks = 5000 * TICKS_PER_SECOND
        self.sleeps = 0
        self.on_sleep: list = []

    def read(self) -> float:
        return self.ticks / TICKS_PER_SECOND

    def advance(self, ticks: int) -> None:
        self.ticks += ticks

    def sleep(self, seconds: float) -> None:
        """FAKE sleep: nothing waits, the fake clock moves and late writers get their turn."""
        self.sleeps += 1
        assert self.sleeps <= FAKE_SLEEP_LIMIT, "the driver waits without end"
        self.ticks += max(1, round(seconds * TICKS_PER_SECOND))
        for hook in list(self.on_sleep):
            hook()


def _fake_header(pid: int, tp_rank: int, limits: grid.Limits, processes: int) -> dict:
    """A FAKE recorder header of format 1, typed here key by key."""
    record = limits.as_dict()
    return {
        "type": "header",
        "format": 1,
        "pid": pid,
        "class": "SglbenchRecorderScheduler",
        "mro": ["SglbenchRecorderScheduler", "Scheduler", "object"],
        "sglang_version": "0.5.23",
        "leader": tp_rank == 0,
        "ranks": {
            "tp_size": processes,
            "pp_size": 1,
            "attn_dp_size": 1,
            "attn_dp_enabled": False,
            "attn_tp_rank": tp_rank,
            "attn_cp_rank": 0,
            "attn_dp_rank": 0,
            "pp_rank": 0,
        },
        "limits": {
            "page_size": record["page_size"],
            "max_req_input_len": record["max_req_input_len"],
            "max_total_num_tokens": record["max_total_num_tokens"],
            "max_running_requests": record["max_running_requests"],
            "max_prefill_tokens": record["max_prefill_tokens"],
            "chunked_prefill_size": record["chunked_prefill_size"],
        },
        "decode_graph_sizes": record["decode_graph_sizes"],
        "prefill_graph_sizes": record["prefill_graph_sizes"],
        "enable_overlap": True,
        "checks": {
            "disable_radix_cache": False,
            "is_hybrid_swa": False,
            "is_hybrid_ssm": False,
            "max_queued_requests": None,
            "max_new_tokens_limit": None,
            "schedule_policy": "fcfs",
            "enable_priority_scheduling": False,
            "is_mixed_chunk": False,
            "prefill_decode_interval": 0,
            "prefill_delayer": False,
            "min_free_slots_delayer": False,
            "recv_skipper": False,
            "input_blocker": False,
            "pp_max_micro_batch_size": record["max_running_requests"],
            "is_hip": False,
        },
    }


def _edit(path: str, value, *, process: int | None = None):
    """A header edit: set ``header[a][b] = value`` for ``path`` "a.b", in every process or in one."""

    def edit(index: int, header: dict) -> dict:
        if process is None or process == index:
            *parents, last = path.split(".")
            target = header
            for key in parents:
                target = target[key]
            target[last] = value
        return header

    return edit


class _FakeLog:
    """One FAKE recorder file of one FAKE scheduler process."""

    def __init__(self, path: Path, clock_offset_ticks: int, slower_ticks: int) -> None:
        self.path = path
        self.clock_offset_ticks = clock_offset_ticks
        self.slower_ticks = slower_ticks
        self.seq = 0
        self.forward_iter = 0

    def write(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def batch(self, batch: dict, start_ticks: int) -> None:
        """Append the batch line of one FAKE batch that started at ``start_ticks`` of the fake clock."""
        self.seq += 1
        self.forward_iter += 1
        launch = start_ticks + self.clock_offset_ticks - batch.get("lead_ticks", 0)
        out = start_ticks + self.clock_offset_ticks + batch["ticks"] + self.slower_ticks
        extend = batch["mode"] == "EXTEND"
        self.write(
            {
                "type": "batch",
                "seq": self.seq,
                "forward_iter": self.forward_iter,
                "mode": batch["mode"],
                "launch_ts": batch.get("launch_ts", launch / TICKS_PER_SECOND),
                "run_return_ts": (launch + 1) / TICKS_PER_SECOND,
                "out_ts": batch.get("out_ts", out / TICKS_PER_SECOND),
                "copy_done": batch.get("copy_done", True),
                "rids": batch["rids"],
                "extend_num_tokens": batch["extend_num_tokens"] if extend else None,
                "prefix_lens": batch["prefix_lens"] if extend else None,
                "extend_lens": batch["extend_lens"] if extend else None,
            }
        )


@dataclasses.dataclass
class _FakeCall:
    """One call the driver made to the fake engine."""

    prompts: list[list[int]]
    rids: list[str]
    max_new_tokens: int
    extra: tuple  # positional and keyword arguments beyond the three of the call; the driver sends none
    modes: list[str]  # the modes of the FAKE batches written for the call, faults included

    @property
    def key(self) -> tuple[int, str, int]:
        parts = content.parse_request_id(self.rids[0])
        return parts["benchmark_id"], parts["stage"], parts["attempt"]


def _extend_batch(rids=(), prefix_lens=(), extend_lens=()) -> dict:
    return {"mode": "EXTEND", "rids": list(rids), "prefix_lens": list(prefix_lens), "extend_lens": list(extend_lens)}


class _FakeEngine:
    """Stands in for an SGLang engine and its scheduler processes. Everything it reports is FAKE.

    ``faults`` maps ``(benchmark_id, stage, attempt)`` of a call to the name of what goes wrong in it.
    ``evictions`` maps a benchmark id to a number of attempts: the cache is dropped just before the measured
    call (``meas`` or ``run``) of that many attempts of the point.
    """

    def __init__(
        self,
        recorder_dir: Path,
        limits: grid.Limits,
        clock: _FakeClock,
        *,
        processes: int = 1,
        faults: dict | None = None,
        evictions: dict | None = None,
        edit_header=None,
        headers: bool = True,
    ) -> None:
        self.limits = limits
        self.clock = clock
        self.processes = processes
        self.faults = dict(faults or {})
        self.evictions = dict(evictions or {})
        self.edit_header = edit_header
        self.cache = PageCacheModel(limits.page_size)
        self.calls: list[_FakeCall] = []
        self.late: list = []  # writes of a late follower, done after a few FAKE sleeps
        self.pending: list = []  # FAKE batches of a call that are written when the next call starts
        # Every FAKE process has its own clock origin and the followers are a little slower, as the monotonic
        # clocks of real processes are unrelated. A driver that mixed the logs up would show it in its samples.
        self.logs = [
            _FakeLog(recorder_dir / f"sched-{FAKE_FIRST_PID + index}.jsonl", index * 1000 * TICKS_PER_SECOND, index * 7)
            for index in range(processes)
        ]
        if headers:
            self.write_headers()

    def write_headers(self) -> None:
        for index, log in enumerate(self.logs):
            header = _fake_header(FAKE_FIRST_PID + index, index, self.limits, self.processes)
            if self.edit_header is not None:
                header = self.edit_header(index, header)
            if header is not None:
                log.write(header)

    def fake_eviction(self) -> None:
        """FAKE fault: the engine needed room and dropped every unlocked cache entry."""
        self.cache = PageCacheModel(self.limits.page_size)

    # -- the FAKE scheduler

    def _fake_batches(self, rids: list[str], prompts: list[list[int]], cached: list[int], output_tokens: int) -> list:
        """Extend batches filled up to the chunk budget, then one decode batch for every further output token.

        A request is charged its uncached tokens rounded up to whole pages. One that does not fit what is left
        of the budget is cut: the part that fits goes into this batch and the rest into the next ones.
        """
        page, chunk = self.limits.page_size, self.limits.chunked_prefill_size
        batches = []
        current = _extend_batch()
        left_in_batch = chunk
        for rid, prompt, hit in zip(rids, prompts, cached, strict=True):
            done = hit
            while True:
                remaining = len(prompt) - done
                if chunk is None or grid.ceil_pages(remaining, page) <= left_in_batch:
                    part = remaining
                else:
                    part = grid.floor_pages(left_in_batch, page)
                if part:
                    current["rids"].append(rid)
                    current["prefix_lens"].append(done)
                    current["extend_lens"].append(part)
                    done += part
                    if chunk is not None:
                        left_in_batch -= grid.ceil_pages(part, page)
                if done == len(prompt):
                    break
                batches.append(current)
                current = _extend_batch()
                left_in_batch = chunk
        if current["rids"]:
            batches.append(current)
        batches.extend({"mode": "DECODE", "rids": list(rids)} for _step in range(output_tokens - 1))
        return batches

    @staticmethod
    def _fake_fault_in_batches(fault: str | None, batches: list, rids: list[str], prompts: list[list[int]]) -> None:
        """Apply a FAKE fault that changes the batch lines of a call."""
        decodes = [index for index, batch in enumerate(batches) if batch["mode"] == "DECODE"]
        first = batches[0]
        if fault == "foreign_id":
            first["rids"].append(FAKE_FOREIGN_RID)
            first["prefix_lens"].append(0)
            first["extend_lens"].append(1)
        elif fault == "split_extend":
            half = len(first["rids"]) // 2
            assert half, "a split needs two requests"
            batches[0:1] = [
                _extend_batch(first["rids"][:half], first["prefix_lens"][:half], first["extend_lens"][:half]),
                _extend_batch(first["rids"][half:], first["prefix_lens"][half:], first["extend_lens"][half:]),
            ]
        elif fault == "chunked":
            # The last token of the first request is computed in a second extend batch.
            first["extend_lens"][0] -= 1
            assert first["extend_lens"][0] > 0, "a chunk needs two tokens"
            batches.insert(1, _extend_batch(rids[:1], [len(prompts[0]) - 1], [1]))
        elif fault == "decode_follows":
            batches.append({"mode": "DECODE", "rids": list(rids)})
        elif fault == "decode_missing_id":
            assert len(rids) > 1, "a missing id needs two requests"
            batches[decodes[3]]["rids"] = list(rids[:-1])
        elif fault == "decode_foreign_id":
            batches[decodes[3]]["rids"] = [*rids, FAKE_FOREIGN_RID]
        elif fault == "five_decode":
            del batches[decodes[-1]]
        elif fault == "seven_decode":
            batches.append({"mode": "DECODE", "rids": list(rids)})
        elif fault == "foreign_between":
            batches.insert(decodes[2], _extend_batch([FAKE_FOREIGN_RID], [0], [1]))
        elif fault == "idle_between":
            batches.insert(decodes[2], {"mode": "IDLE", "rids": []})
        elif fault == "idle_after":
            batches.append({"mode": "IDLE", "rids": []})
        elif fault == "extend_after_decode":
            batches.insert(1, batches.pop(0))
        elif fault == "second_extend_of_one_request":
            batches.insert(1, _extend_batch(rids[:1], first["prefix_lens"][:1], first["extend_lens"][:1]))
        elif fault == "extend_lens_one_less":
            first["extend_lens"] = [tokens - 1 for tokens in first["extend_lens"]]
        elif fault == "wrong_extend_lens":
            first["extend_lens"][0] += 1
        elif fault == "wrong_prefix_lens":
            first["prefix_lens"][0] += 1
        elif fault == "line_missed_the_cache":
            first["extend_lens"] = [len(prompt) for prompt in prompts]
            first["prefix_lens"] = [0] * len(prompts)
        elif fault == "no_lens":
            first["prefix_lens"] = None
        elif fault == "copy_done_false":
            first["copy_done"] = False
        elif fault == "no_out_ts":
            first["out_ts"] = None
        elif fault == "whole_number_launch_ts":
            first["launch_ts"] = 5000
        elif fault == "decode_copy_done_false":
            batches[decodes[3]]["copy_done"] = False
        elif fault == "short_lens":
            first["prefix_lens"] = first["prefix_lens"][:-1]
        elif fault == "idle_with_ids_follows":
            batches.append({"mode": "IDLE", "rids": list(rids)})

    def _write_batches(self, fault: str | None, batches: list, stage: str) -> None:
        if fault == "silent":
            return
        for step, batch in enumerate(batches):
            if batch["mode"] == "EXTEND":
                computed = sum(batch["extend_lens"])
                batch.setdefault("extend_num_tokens", computed + (1 if fault == "wrong_extend_tokens" else 0))
                batch["ticks"] = FAKE_EXTEND_BASE_TICKS + computed
                if stage == "warm":
                    batch["ticks"] += FAKE_WARM_SHOT_EXTRA_TICKS
            elif batch["mode"] == "DECODE":
                decode_step = sum(1 for earlier in batches[:step] if earlier["mode"] == "DECODE")
                tenths = FAKE_DECODE_STEP_TENTHS[decode_step % len(FAKE_DECODE_STEP_TENTHS)]
                batch["ticks"] = (FAKE_DECODE_BASE_TICKS + len(batch["rids"])) * tenths
            else:
                batch["ticks"] = 0  # a FAKE idle batch takes no time
            if fault == "zero_duration":
                batch["ticks"] = 0
            if step:
                batch["lead_ticks"] = FAKE_OVERLAP_LEAD_TICKS
            start = self.clock.ticks
            self.clock.advance(batch["ticks"])
            for index, log in enumerate(self.logs):
                if index and fault == "follower_missing":
                    continue
                if index and fault == "follower_other_ids":
                    log.batch({**batch, "rids": [*batch["rids"][:-1], FAKE_FOREIGN_RID]}, start)
                elif index and fault == "follower_other_mode":
                    log.batch({**batch, "mode": "MIXED"}, start)
                elif index and fault == "follower_late":
                    self.late.append((log, batch, start))
                else:
                    log.batch(batch, start)
        if fault == "follower_late":
            sleeps_then = self.clock.sleeps

            def write_late() -> None:
                if self.clock.sleeps >= sleeps_then + 3:
                    for log, batch, start in self.late:
                        log.batch(batch, start)
                    self.late.clear()
                    self.clock.on_sleep.remove(write_late)

            self.clock.on_sleep.append(write_late)

    # -- the call

    def generate(self, prompts, rids, max_new_tokens, *args, **kwargs) -> list[dict]:
        """Answer one call as the engine object of the launcher would: one dict per request, in input order."""
        call = _FakeCall([list(prompt) for prompt in prompts], list(rids), max_new_tokens, (args, kwargs), [])
        self.calls.append(call)
        benchmark_id, stage, attempt = call.key
        fault = self.faults.get(call.key)
        if stage in ("meas", "run") and self.evictions.get(benchmark_id, 0) > attempt:
            self.fake_eviction()
        if fault == "evict":
            self.fake_eviction()
        # FAKE: a batch the scheduler launched for the call before reaches the log only now.
        for batch in self.pending:
            for log in self.logs:
                log.batch(batch, self.clock.ticks)
        self.pending.clear()
        if fault == "raise":
            raise RuntimeError("FAKE engine failure")
        if fault == "raise_value_error":
            raise ValueError("FAKE engine failure")
        if fault == "system_exit":
            raise SystemExit(0)
        if fault == "keyboard_interrupt":
            raise KeyboardInterrupt

        # A call is one batch of arrivals: every request is matched against the cache before any is stored.
        cached = [self.cache.cached(prompt) for prompt in call.prompts]
        batches = self._fake_batches(call.rids, call.prompts, cached, max_new_tokens)
        self._fake_fault_in_batches(fault, batches, call.rids, call.prompts)
        call.modes.extend(batch["mode"] for batch in batches)
        self._write_batches(fault, batches, stage)
        if fault == "fault_line":
            self.logs[0].write({"type": "fault", "where": "process_batch_result", "error": "FAKE recorder fault"})
        if fault == "follower_fault_line":
            self.logs[-1].write({"type": "fault", "where": "process_batch_result", "error": "FAKE recorder fault"})
        if fault == "late_decode":
            self.pending.append({"mode": "DECODE", "rids": list(call.rids), "ticks": FAKE_DECODE_BASE_TICKS})
        for prompt in call.prompts:
            # A finished request stores its prompt and the generated tokens that have KV: all but the last.
            # The FAKE output tokens lie below the plan's range, so they equal no planned token.
            self.cache.finish_prefill([*prompt, *range(max_new_tokens - 1)])
        if fault == "deadline":
            self.clock.advance(round(FAKE_TIMEOUT_SECONDS * TICKS_PER_SECOND))

        responses = [
            {
                "rid": rid,
                "prompt_tokens": len(prompt),
                "cached_tokens": hit,
                "completion_tokens": max_new_tokens,
                "finish_type": "length",
                "num_retractions": 0,
            }
            for rid, prompt, hit in zip(call.rids, call.prompts, cached, strict=True)
        ]
        if fault == "prompt_count":
            responses[0]["prompt_tokens"] += 1
        elif fault == "retraction":
            responses[-1]["num_retractions"] = 1
        elif fault == "finish_stop":
            responses[0]["finish_type"] = "stop"
        elif fault == "short_completion":
            responses[0]["completion_tokens"] -= 1
        elif fault == "cached_off":
            responses[0]["cached_tokens"] += 1
        elif fault == "missing_key":
            del responses[0]["cached_tokens"]
        elif fault == "short_list":
            responses.pop()
        elif fault == "wrong_rid":
            responses[0]["rid"] = FAKE_FOREIGN_RID
        return responses


# ---------------------------------------------------------------------------------------------------------------
# Running the driver against the fakes
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class _FakeRun:
    """One driver run against the fake engine, with what the test planned itself."""

    run: driver.BenchmarkRun
    engine: _FakeEngine
    clock: _FakeClock
    cell: object
    root: Path
    pod: Path
    recorder_dir: Path
    limits: grid.Limits
    settings: driver.RunSettings
    tokens: tuple[int, int]

    @property
    def built(self) -> grid.Grid:
        """The grid, planned here from the same limits and settings."""
        return grid.build_grid(self.settings.mode, self.limits, self.settings.caps, manifest=self.settings.manifest)

    def execute(self) -> None:
        self.run.execute(self.engine, token_low=self.tokens[0], token_high=self.tokens[1])

    def payload(self) -> dict:
        (payload,) = self.run.payloads()
        return payload

    def sent(self) -> list[dict]:
        """The parts of every request id the fake engine received, in order."""
        return [content.parse_request_id(rid) for call in self.engine.calls for rid in call.rids]

    def sent_benchmark_ids(self) -> set[int]:
        return {parts["benchmark_id"] for parts in self.sent()}

    def calls_of(self, benchmark_id: int, stage: str | None = None) -> list[_FakeCall]:
        return [
            call
            for call in self.engine.calls
            if call.key[0] == benchmark_id and (stage is None or call.key[1] == stage)
        ]


def _start(
    tmp_path: Path,
    mode: str = "prefill",
    page_size: int = 1,
    *,
    limits: grid.Limits | None = None,
    processes: int = 1,
    faults: dict | None = None,
    evictions: dict | None = None,
    edit_header=None,
    headers: bool = True,
    tokens: tuple[int, int] = (TOKEN_LOW, TOKEN_HIGH),
    execute: bool = True,
    started_before_seconds: float = 0.0,
    **overrides,
) -> _FakeRun:
    """Lay out a pod, start a fake engine, build a run and, unless told not to, execute it.

    ``started_before_seconds`` is the FAKE time between the start of the launcher and the run: what an engine
    build would take.
    """
    limits = _limits(page_size) if limits is None else limits
    cell = _cell(phase="prefill" if mode == "agg" else mode, dp=1, strategy="single")
    pod = _pod(tmp_path, cell)
    # Where the launcher puts the recorder files: under the directory of the output path. The validator sweeps
    # that tree, so the files of the FAKE recorder are in its way on purpose.
    recorder_dir = pod / "sglbench-recorder" / FAKE_RUN_ID
    recorder_dir.mkdir(parents=True)
    clock = _FakeClock()
    engine = _FakeEngine(
        recorder_dir,
        limits,
        clock,
        processes=processes,
        faults=faults,
        evictions=evictions,
        edit_header=edit_header,
        headers=headers,
    )
    values = {
        "mode": mode,
        "caps": SMALL_CAPS,
        "manifest": None,
        "warmup_iterations": 5,
        "timeout_seconds": FAKE_TIMEOUT_SECONDS,
        "output_path": str(pod / "benchmark.json"),
        "run_id": FAKE_RUN_ID,
        "content_seed": "0",
        "worker_id": FAKE_WORKER_ID,
        "expected_scheduler_processes": processes,
        "config": dict(FAKE_CONFIG),
        "producer": dict(FAKE_PRODUCER),
        "prefill_max_requests": None,
        "max_queued_requests": None,
        "log_wait_seconds": 2.0,
        "header_wait_seconds": 2.0,
    }
    values.update(overrides)
    settings = driver.RunSettings(**values)
    started_at = clock.read() - started_before_seconds
    run = driver.BenchmarkRun(settings, recorder_dir, started_at=started_at, clock=clock.read, sleep=clock.sleep)
    fake = _FakeRun(run, engine, clock, cell, tmp_path, pod, recorder_dir, limits, settings, tokens)
    if execute:
        fake.execute()
    return fake


def _expected_sample(point: grid.Point) -> float:
    """The FAKE sample a point must carry, worked out here from the FAKE timings."""
    if point.phase == grid.PREFILL:
        return (FAKE_EXTEND_BASE_TICKS + point.batch_size * point.new_tokens) / TICKS_PER_SECOND
    return (FAKE_DECODE_BASE_TICKS + point.batch_size) * 10 / TICKS_PER_SECOND


def _expected_steps(point: grid.Point) -> list[float]:
    return [
        (FAKE_DECODE_BASE_TICKS + point.batch_size) * tenths / TICKS_PER_SECOND for tenths in FAKE_DECODE_STEP_TENTHS
    ]


def _expected_regime(point: grid.Point) -> str:
    if point.phase == grid.DECODE:
        return "real_kv"
    return "real_prefix" if point.cached_tokens else "not_applicable"


def _assert_accepted(fake: _FakeRun) -> dict:
    """A complete run: write it, then the validator returns every planned point and the gate exits 0."""
    built = fake.built
    assert (fake.run.status, fake.run.reason) == ("complete", None)
    written = fake.run.write()
    rank_path = str(fake.pod / "benchmark.json")
    assert written == {"status": "complete", "paths": [rank_path, str(fake.pod / "benchmark_merged.json")]}
    on_disk = json.loads(Path(rank_path).read_text(encoding="utf-8"))
    assert on_disk == fake.payload()
    assert (on_disk["status"], on_disk["valid"], on_disk["usable"]) == ("complete", True, True)
    assert on_disk["coverage"] == {
        "expected_points": len(built.points),
        "completed_points": len(built.points),
        "skipped_points": 0,
    }
    assert (on_disk["run_id"], on_disk["grid_digest"]) == (FAKE_RUN_ID, built.digest)
    # Every row carries the FAKE sample of the leader, the worker id and, for decode, the six FAKE steps.
    for point, row in zip(built.points, on_disk["results"], strict=True):
        (fpm,) = row["fpms"]
        assert (row["point"], row["kv_seed_regime"]) == (point.as_dict(), _expected_regime(point))
        assert (fpm["dp_rank"], fpm["wall_time"], fpm["worker_id"]) == (0, _expected_sample(point), FAKE_WORKER_ID)
        if point.phase == grid.DECODE:
            assert fpm[artifact.DECODE_STEPS_KEY] == _expected_steps(point)
        else:
            assert artifact.DECODE_STEPS_KEY not in fpm
    if fake.settings.mode != grid.AGG:
        collection = _validate(fake.cell, fake.root)
        assert [measurement.point for measurement in collection.points] == built.point_dicts()
        assert [measurement.kv_seed_regime for measurement in collection.points] == [
            _expected_regime(point) for point in built.points
        ]
        assert [measurement.rank_wall_times for measurement in collection.points] == [
            ((0, _expected_sample(point)),) for point in built.points
        ]
        assert (collection.runtime_run_id, collection.runtime_grid_digest) == (FAKE_RUN_ID, built.digest)
    gate = _gate([rank_path], fake.settings.mode)
    assert gate.returncode == GATE_ACCEPT, gate.stderr
    return on_disk


def _assert_rejected(fake: _FakeRun, status: str) -> dict:
    """A run that is not complete: write it, then the validator raises and the gate fails the pod."""
    written = fake.run.write()
    assert fake.run.status == status
    rank_path = str(fake.pod / "benchmark.json")
    merged_path = fake.pod / "benchmark_merged.json"
    if status == "failed":
        # A failed run has no merged file.
        assert written == {"status": "failed", "paths": [rank_path]}
        assert not merged_path.exists()
    else:
        assert written == {"status": "partial", "paths": [rank_path, str(merged_path)]}
    on_disk = json.loads(Path(rank_path).read_text(encoding="utf-8"))
    assert on_disk == fake.payload()
    assert (on_disk["status"], on_disk["valid"]) == (status, False)
    with pytest.raises(ValueError, match="invalid native terminal result envelope"):
        _validate(fake.cell, fake.root)
    gate = _gate([rank_path], fake.cell.workload_kind)
    assert gate.returncode == GATE_INVALID
    assert f"status={status!r} valid=False" in gate.stderr
    return on_disk


def _assert_point_failed(fake: _FakeRun, victim: grid.Point, reason: str) -> dict:
    """The run stopped at ``victim`` with ``reason``: nothing after it was sent and the artifact names it."""
    message = f"benchmark_id={victim.benchmark_id}: benchmark point failed: {reason}"
    assert (fake.run.status, fake.run.reason) == ("failed", message)
    assert max(fake.sent_benchmark_ids()) == victim.benchmark_id
    on_disk = _assert_rejected(fake, "failed")
    assert (on_disk["error"], on_disk["errors"], on_disk["stop_reason"]) == (message, [message], None)
    assert on_disk["skipped_points"] == [
        {"point": victim.as_dict(), "kv_seed_regime": _expected_regime(victim), "reason": reason}
    ]
    assert on_disk["coverage"] == {
        "expected_points": len(fake.built.points),
        "completed_points": victim.benchmark_id - 1,
        "skipped_points": 1,
    }
    # The rows of the points before it are kept.
    assert [row["point"] for row in on_disk["results"]] == fake.built.point_dicts()[: victim.benchmark_id - 1]
    return on_disk


def _assert_failed_before_the_grid(fake: _FakeRun, reason: str) -> dict:
    """The run failed before it had a grid: no request was sent and the artifact has no expected point."""
    assert (fake.run.status, fake.run.reason) == ("failed", reason)
    assert fake.engine.calls == []
    on_disk = _assert_rejected(fake, "failed")
    assert (on_disk["error"], on_disk["grid_digest"]) == (reason, grid.grid_digest([]))
    assert on_disk["coverage"] == {"expected_points": 0, "completed_points": 0, "skipped_points": 0}
    return on_disk


def _victim(fake: _FakeRun, *, batch: int = 2, cached: bool = True) -> grid.Point:
    """A planned point after the first two with at least ``batch`` requests and, if asked, cached tokens."""
    for point in fake.built.points[2:]:
        expected = grid.expected_cached_tokens(point.phase, point.cached_tokens, fake.limits.page_size)
        if point.batch_size >= batch and bool(expected) == cached:
            return point
    raise AssertionError("the planned grid has no such point")


def _plan_only(tmp_path: Path, mode: str, page_size: int = 1, **options) -> _FakeRun:
    """A run that is laid out but not executed, to pick a victim from its planned grid."""
    return _start(tmp_path / "plan", mode, page_size, execute=False, **options)


# ---------------------------------------------------------------------------------------------------------------
# Complete runs: both checkers accept
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize("mode", ["prefill", "decode", "agg"])
def test_complete_run_of_the_default_grid_passes_the_validator_and_the_gate(tmp_path, mode, page_size):
    fake = _start(tmp_path, mode, page_size, caps=grid.SampleCaps())

    built = fake.built
    assert len(built.points) > 50 and built.missing_phases == () and built.source == "generated"
    on_disk = _assert_accepted(fake)

    assert on_disk["config"] == {"mode": mode, **FAKE_CONFIG}
    stats = on_disk["producer"]["content"]["issued"]
    need = on_disk["producer"]["first_token_need"]
    assert on_disk["producer"] == {
        "name": "aisim_sglbench",
        "artifact_format": 1,
        **FAKE_PRODUCER,
        "timing_metric": "scheduler_wall_time",
        "timing_clock": "process_local_monotonic",
        "staging": "one_batch_call",
        "recorder_format": 1,
        "scheduler_processes": 1,
        "limits": fake.limits.as_dict(),
        "page_size": page_size,
        "enable_overlap": True,
        "content": {
            "source": "seeded_token_ids",
            "seed": "0",
            "token_range": [TOKEN_LOW, TOKEN_HIGH],
            "issued": stats,
        },
        "restaged_points": [],
        "self_test": ["batch_shape", "aligned_prefix", "unaligned_prompt", "seven_token_run"],
        "first_token_need": need,
    }
    assert ("kvwarm" in on_disk) == (mode != "prefill")
    # The first tokens the run was planned to take are the ones the content plan issued: no re-stage happened.
    zero_cached_batches = [
        point.batch_size for point in built.points if point.phase == "prefill" and not point.cached_tokens
    ]
    zero_cached = sum(zero_cached_batches)
    slot_batches = [point.batch_size for point in built.points if point.phase == "decode" or point.cached_tokens]
    assert need == {
        "slots": max(slot_batches),
        "zero_cached_prefill_prompts": 2 * zero_cached,
        "warmup_prompts": 1,
        "self_test_free_prompts": 4,
        "restage_reserve": 2 * max(zero_cached_batches, default=0),
        "total": need["total"],
    }
    assert need["total"] == sum(value for key, value in need.items() if key != "total")
    assert (stats["slots"], stats["zero_cached_prompts"], stats["free_prompts"]) == (
        need["slots"],
        need["zero_cached_prefill_prompts"],
        need["self_test_free_prompts"] + need["warmup_prompts"],
    )
    # A decode prompt is issued for the decode points only: the self-test sends none on a plan slot.
    assert stats["decode_prompts"] == sum(point.batch_size for point in built.points if point.phase == "decode")
    # The elapsed time is the fake clock's, from the start of execute to its end.
    assert on_disk["timing"]["benchmark_elapsed_seconds"] == fake.clock.read() - 5000.0
    assert on_disk["timing"]["measured_iteration_seconds"] == sum(_expected_sample(point) for point in built.points)
    if mode == "agg":
        # AISim launches one phase per cell: a gate started for one phase refuses the agg file.
        assert _gate([str(fake.pod / "benchmark.json")], "prefill").returncode == GATE_INVALID


def test_the_stages_of_each_point_are_sent_in_order_with_the_planned_prompts(tmp_path):
    fake = _start(tmp_path, "agg")
    built = fake.built
    plan = content.ContentPlan("0", TOKEN_LOW, TOKEN_HIGH)

    for point in built.points:
        calls = fake.calls_of(point.benchmark_id)
        batch = point.batch_size
        if point.phase == "prefill":
            stages = ["seed", "warm", "meas"] if point.cached_tokens else ["warm", "meas"]
            assert [call.key for call in calls] == [(point.benchmark_id, stage, 0) for stage in stages]
            # Every call of a prefill point asks for one output token and ran as one extend batch, the seed
            # aside: a seed above the chunk budget is cut into several, which is allowed.
            assert [call.max_new_tokens for call in calls] == [1] * len(stages)
            warm, meas = calls[-2:]
            assert warm.modes == meas.modes == ["EXTEND"]
            assert all(set(call.modes) == {"EXTEND"} for call in calls)
            prompt_tokens = point.cached_tokens + point.new_tokens
            for call in (warm, meas):
                assert [len(prompt) for prompt in call.prompts] == [prompt_tokens] * batch
            if point.cached_tokens:
                # The seed is the slot's first k tokens, and both shots start with it and differ after it.
                assert calls[0].prompts == [plan.slot_prefix(0, slot, point.cached_tokens) for slot in range(batch)]
                for slot in range(batch):
                    assert warm.prompts[slot][: point.cached_tokens] == calls[0].prompts[slot]
                    assert meas.prompts[slot][: point.cached_tokens] == calls[0].prompts[slot]
                    assert warm.prompts[slot][point.cached_tokens] != meas.prompts[slot][point.cached_tokens]
        else:
            seed, run = calls
            assert [call.key for call in calls] == [(point.benchmark_id, "seed", 0), (point.benchmark_id, "run", 0)]
            assert (seed.max_new_tokens, run.max_new_tokens) == (1, 7)
            # The seed and the run send the same prompts: the slot's first k - 3 tokens.
            assert (
                seed.prompts
                == run.prompts
                == [plan.slot_prefix(0, slot, point.cached_tokens - 3) for slot in range(batch)]
            )
            assert run.modes[-6:] == ["DECODE"] * 6 and set(run.modes[:-6]) == {"EXTEND"}
        for call in calls:
            assert [content.parse_request_id(rid)["slot"] for rid in call.rids] == list(range(batch))
    # Points run in grid order and nothing else was sent for a point.
    order = [call.key[0] for call in fake.engine.calls if call.key[0]]
    assert order == sorted(order) and set(order) == {point.benchmark_id for point in built.points}


def test_request_fields(tmp_path):
    fake = _start(tmp_path, "agg")
    assert fake.run.status == "complete"

    rids = [rid for call in fake.engine.calls for rid in call.rids]
    for call in fake.engine.calls:
        # Three arguments and nothing else: no sampling option, priority, salt or logprob request.
        assert call.extra == ((), {})
        assert len(call.rids) == len(call.prompts) >= 1
        assert all(
            type(token) is int and TOKEN_LOW <= token < TOKEN_HIGH for prompt in call.prompts for token in prompt
        )
        benchmark_id, stage, attempt = call.key
        for slot, rid in enumerate(call.rids):
            # The id is the content module's, with the first eight characters of the run id, rank 0 and the slot.
            assert rid == f"sglb-{FAKE_RUN_TAG}-p{benchmark_id}-{stage}{attempt}-r0-s{slot}-e"
            assert rid == content.request_id(FAKE_RUN_ID[:8], benchmark_id, stage, attempt, 0, slot)
        assert call.max_new_tokens == {"seed": 1, "warm": 1, "meas": 1, "run": 7}.get(stage, call.max_new_tokens)
    # One id per request, never repeated, and none a prefix of another (SGLang aborts by id prefix).
    assert len(set(rids)) == len(rids)
    assert not any(longer.startswith(shorter) for shorter, longer in itertools.pairwise(sorted(rids)))
    # The calls outside any point come first: benchmark id 0, stage "free", numbered in the attempt field.
    free = [call for call in fake.engine.calls if call.key[0] == 0]
    assert fake.engine.calls[: len(free)] == free
    assert [call.key for call in free] == [(0, "free", number) for number in range(7)]
    assert [len(call.prompts) for call in free] == [2, 2, 2, 2, 2, 2, 1]
    assert [call.max_new_tokens for call in free] == [1, 1, 1, 1, 1, 7, 5]
    # The self-test: G = 16 tokens at page size 1. The warm-up: one prompt of 256 tokens.
    assert [len(call.prompts[0]) for call in free] == [16, 16, 32, 39, 39, 16, 256]
    assert free[FREE_UNALIGNED_SEED].prompts == free[FREE_UNALIGNED_AGAIN].prompts
    assert free[FREE_SEVEN_TOKEN_RUN].modes == ["EXTEND"] + ["DECODE"] * 6
    # No decode prompt was sent on a plan slot before the prefill points: the seven-token run of the self-test
    # starts with first tokens that no slot sequence owns.
    plan = content.ContentPlan("0", TOKEN_LOW, TOKEN_HIGH)
    slot_first_tokens = {plan.slot_prefix(0, slot, 1)[0] for slot in range(fake.limits.max_running_requests)}
    assert not {prompt[0] for prompt in free[FREE_SEVEN_TOKEN_RUN].prompts} & slot_first_tokens
    assert {prompt[0] for prompt in free[FREE_ALIGNED_SEED].prompts} <= slot_first_tokens


def test_a_points_manifest_is_measured_in_file_order(tmp_path):
    manifest = grid.parse_points_manifest(
        json.dumps(
            {
                "schema_version": 3,
                "prefill": [
                    {"batch_size": 2, "total_prefill_tokens": 64, "total_kv_read_tokens": 64},
                    {"batch_size": 1, "total_prefill_tokens": 5, "total_kv_read_tokens": 0},
                    {"batch_size": 4, "total_prefill_tokens": 16, "total_kv_read_tokens": 384, "rows": [[4, 96]] * 4},
                ],
                "decode": [{"batch_size": 3, "total_kv_read_tokens": 60}],
            }
        )
    )
    config = {**FAKE_CONFIG, "points_source": "manifest"}
    fake = _start(tmp_path, "prefill", 16, limits=_limits(16), manifest=manifest, config=config)

    built = fake.built
    assert built.source == "manifest"
    assert [(point.batch_size, point.new_tokens, point.cached_tokens) for point in built.points] == [
        (2, 32, 32),
        (1, 5, 0),
        (4, 4, 96),
    ]
    on_disk = _assert_accepted(fake)
    assert [row["point"]["sample_reasons"] for row in on_disk["results"]] == [
        ["explicit", "prefill_real_seed"],
        ["explicit"],
        ["explicit", "prefill_real_seed"],
    ]
    assert on_disk["config"]["points_source"] == "manifest"
    # A manifest point that does not fit is refused when the grid is built, before any request.
    bad = grid.parse_points_manifest(
        json.dumps(
            {
                "schema_version": 1,
                "prefill": [{"batch_size": 1, "total_prefill_tokens": 8, "total_kv_read_tokens": 8}],
                "decode": [],
            }
        )
    )
    refused = _start(tmp_path / "refused", "prefill", 16, limits=_limits(16), manifest=bad)
    assert refused.run.reason.startswith(
        "rejected: prefill[0]: explicit benchmark point is infeasible: reason=cached_not_page_aligned "
    )
    assert refused.engine.calls == []
    assert refused.payload()["coverage"]["expected_points"] == 0


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_two_scheduler_processes_are_read_as_a_leader_and_a_follower(tmp_path, mode):
    fake = _start(tmp_path, mode, processes=2)

    # Both FAKE processes wrote every batch; the follower's clock is 1000 s ahead and each of its batches is
    # seven ticks slower. The samples are the leader's.
    lines = {
        name: [json.loads(line) for line in (fake.recorder_dir / name).read_text(encoding="utf-8").splitlines()]
        for name in (LEADER_LOG, FOLLOWER_LOG)
    }
    assert (lines[LEADER_LOG][0]["leader"], lines[FOLLOWER_LOG][0]["leader"]) == (True, False)
    assert len(lines[LEADER_LOG]) == len(lines[FOLLOWER_LOG]) > 20
    assert lines[FOLLOWER_LOG][1]["launch_ts"] - lines[LEADER_LOG][1]["launch_ts"] == 1000.0
    on_disk = _assert_accepted(fake)
    assert on_disk["producer"]["scheduler_processes"] == 2


def test_a_decode_run_above_the_chunk_budget_takes_two_extend_batches(tmp_path):
    # Page size 64 with a chunk budget of 384 tokens: six requests fill one extend batch, whatever they compute.
    fake = _start(tmp_path, "decode", 64)

    split = [point for point in fake.built.points if point.batch_size > 6]
    assert len(split) >= 6
    for point in split:
        (run,) = fake.calls_of(point.benchmark_id, "run")
        assert run.modes == ["EXTEND", "EXTEND"] + ["DECODE"] * 6
    whole = [point for point in fake.built.points if point.batch_size <= 6]
    assert whole and all(
        fake.calls_of(point.benchmark_id, "run")[0].modes == ["EXTEND"] + ["DECODE"] * 6 for point in whole
    )
    # The reader holds the lines of the last call only, however long the run was: seven for a run of one request.
    assert fake.built.points[-1].batch_size == 1
    assert [line["mode"] for line in fake.run.logs.batches(LEADER_LOG)] == ["EXTEND"] + ["DECODE"] * 6
    _assert_accepted(fake)


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_an_idle_batch_without_requests_between_two_decode_steps_is_ignored(tmp_path, mode):
    plan = _plan_only(tmp_path, mode)
    victim = _victim(plan)
    stage = "meas" if mode == "prefill" else "run"
    # In a prefill shot the idle line follows the extend batch; in a decode run it lies between two steps.
    fault = "idle_between" if mode == "decode" else "idle_after"
    fake = _start(tmp_path, mode, faults={(victim.benchmark_id, stage, 0): fault})
    if mode == "decode":
        assert "IDLE" in fake.calls_of(victim.benchmark_id, "run")[0].modes
    _assert_accepted(fake)


# ---------------------------------------------------------------------------------------------------------------
# Re-stage after a wrong cached count
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_one_eviction_is_recovered_by_the_one_restage(tmp_path, mode):
    victim = _victim(_plan_only(tmp_path, mode))
    fake = _start(tmp_path, mode, evictions={victim.benchmark_id: 1})

    stages = ["seed", "warm", "meas"] if mode == "prefill" else ["seed", "run"]
    assert [call.key for call in fake.calls_of(victim.benchmark_id)] == [
        (victim.benchmark_id, stage, attempt) for attempt in (0, 1) for stage in stages
    ]
    # No other point was staged twice.
    assert {parts["benchmark_id"] for parts in fake.sent() if parts["attempt"] and parts["benchmark_id"]} == {
        victim.benchmark_id
    }
    on_disk = _assert_accepted(fake)
    assert on_disk["producer"]["restaged_points"] == [victim.benchmark_id]


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_two_evictions_fail_the_point(tmp_path, mode):
    victim = _victim(_plan_only(tmp_path, mode))
    fake = _start(tmp_path, mode, evictions={victim.benchmark_id: 2})

    expected = grid.expected_cached_tokens(mode, victim.cached_tokens, 1)
    stage = "meas" if mode == "prefill" else "run"
    first_rid = f"sglb-{FAKE_RUN_TAG}-p{victim.benchmark_id}-{stage}1-r0-s0-e"
    on_disk = _assert_point_failed(
        fake,
        victim,
        f"real_seed_cache_validation_failed: {victim.batch_size} of {victim.batch_size} request(s) report another "
        f"cached count than the planned {expected}: {first_rid} reports 0",
    )
    # The point was tried exactly twice.
    attempts = collections.Counter(call.key[2] for call in fake.calls_of(victim.benchmark_id, stage))
    assert attempts == {0: 1, 1: 1}
    assert on_disk["producer"]["restaged_points"] == [victim.benchmark_id]


def test_a_wrong_cached_count_in_the_warm_shot_restages_at_once(tmp_path):
    victim = _victim(_plan_only(tmp_path, "prefill"))
    fake = _start(tmp_path, "prefill", faults={(victim.benchmark_id, "warm", 0): "cached_off"})

    # The measured shot of the first attempt was not sent.
    assert [call.key[1:] for call in fake.calls_of(victim.benchmark_id)] == [
        ("seed", 0),
        ("warm", 0),
        ("seed", 1),
        ("warm", 1),
        ("meas", 1),
    ]
    assert _assert_accepted(fake)["producer"]["restaged_points"] == [victim.benchmark_id]


def test_cached_tokens_of_the_measured_batch_that_differ_from_the_plan_restage_the_point(tmp_path):
    # The responses report the planned count and the recorder line of the kept batch does not.
    victim = _victim(_plan_only(tmp_path, "prefill"))
    once = _start(tmp_path / "once", "prefill", faults={(victim.benchmark_id, "meas", 0): "wrong_prefix_lens"})
    assert _assert_accepted(once)["producer"]["restaged_points"] == [victim.benchmark_id]

    faults = {(victim.benchmark_id, "meas", attempt): "wrong_prefix_lens" for attempt in (0, 1)}
    twice = _start(tmp_path / "twice", "prefill", faults=faults)
    planned = victim.batch_size * victim.cached_tokens
    _assert_point_failed(
        twice,
        victim,
        f"real_seed_cache_validation_failed: sum_prefill_kv_tokens is {planned + 1} in the kept batch, "
        f"the point plans {planned}",
    )


def test_a_kept_batch_that_missed_the_cache_is_restaged_and_not_blamed_on_its_token_count(tmp_path):
    # The recorder line of the kept batch says that nothing was cached and every prompt was computed whole,
    # while the responses report the planned counts. Two counters then differ from the plan. The cached tokens
    # are compared first, so the point is staged again as after an eviction, and fails as one on the second miss.
    victim = _victim(_plan_only(tmp_path, "prefill"))
    faults = {(victim.benchmark_id, "meas", attempt): "line_missed_the_cache" for attempt in (0, 1)}
    fake = _start(tmp_path, "prefill", faults=faults)

    planned = victim.batch_size * victim.cached_tokens
    _assert_point_failed(
        fake,
        victim,
        f"real_seed_cache_validation_failed: sum_prefill_kv_tokens is 0 in the kept batch, the point plans {planned}",
    )
    assert fake.payload()["producer"]["restaged_points"] == [victim.benchmark_id]


def test_a_seed_is_not_held_to_a_cached_count(tmp_path):
    # A seed finds what earlier points left: its count is not planned and a surprising one fails nothing.
    victim = _victim(_plan_only(tmp_path, "decode"))
    fake = _start(tmp_path, "decode", faults={(victim.benchmark_id, "seed", 0): "cached_off"})
    assert _assert_accepted(fake)["producer"]["restaged_points"] == []


# ---------------------------------------------------------------------------------------------------------------
# Validity checks: each one fails the point at once, with its reason
# ---------------------------------------------------------------------------------------------------------------

_RID = "sglb-fa4e0123-p{id}-{stage}0-r0-s{slot}-e"
_SIZE = "measured_batch_size_mismatch"
_TOKENS = "measured_prefill_tokens_mismatch"
_CONTEXT = "measured_decode_context_mismatch"
_INCOMPLETE = "request_not_completed"
_RECORDER = "recorder_fault: sched-4100.jsonl"

# The reasons are filled in from the victim and from the FAKE leader log. "{seq}" is the seq of the first batch
# of the faulty call, "{b}" the batch size, "{bn}" the planned new tokens of the batch, "{p}" the prompt tokens
# of a request, "{m}" the output tokens asked for and "{e}" the tokens a request computes in the extend batch
# of a decode run. A digit after a name is added, and a zero subtracts one: "{b1}" is b + 1, "{b0}" is b - 1.
_FOREIGN_IN_EXTEND = (
    _SIZE + ": the extend batch (seq {seq}) holds {b1} request(s), 1 of them foreign, "
    "expected exactly the {b} of the call"
)
_SPLIT = _SIZE + ": the call ran as 2 extend batches of [1, 1] requests, expected one of 2"
_CHUNKED = _TOKENS + ": {rid0} is in 2 extend batches: the request was chunked"
_DECODE_AFTER = _TOKENS + ": a DECODE batch (seq {seq1}) holds requests of a one-token call"
_TOKENS_REPORTED = _TOKENS + ": the extend batch (seq {seq}) reports {bn1} computed tokens and its requests sum to {bn}"
_TOKENS_COUNTED = _TOKENS + ": sum_prefill_tokens is {bn1} in the kept batch, the point plans {bn}"
_PROMPT = _TOKENS + ": {rid0}: the engine reports {p1} prompt tokens, the prompt has {p}"
_RETRACTED = _INCOMPLETE + ": {rid_last}: finish_type='length', completion_tokens={m} of {m}, num_retractions=1"
_STOPPED = _INCOMPLETE + ": {rid0}: finish_type='stop', completion_tokens={m} of {m}, num_retractions=0"
_SHORT = _INCOMPLETE + ": {rid0}: finish_type='length', completion_tokens={m0} of {m}, num_retractions=0"
_RAISED = "engine_call_failed: RuntimeError: FAKE engine failure"
_NO_COPY_EVENT = _RECORDER + ": batch seq {seq} was stamped without a copy event (copy_done=False)"
_NO_LENS = _RECORDER + ": extend batch seq {seq} has no usable prefix_lens, extend_lens and extend_num_tokens"
_NO_TIME = _RECORDER + ": {what} took 0.0 s on the scheduler's clock"
_FAULT_LINE = _RECORDER + ": process_batch_result: FAKE recorder fault"
_EXTEND_TOKENS = (
    _CONTEXT + ": extend batch seq {seq} computes {lens} tokens ({total} in all), expected {e} for each request"
)
_SEQUENCE = _CONTEXT + ": the batches of the run are {found}, expected one or more EXTEND and then 6 DECODE"
_BETWEEN = _CONTEXT + ": a batch of other requests lies between the batches of the run (seq {seqs})"
_DECODE_IDS = _SIZE + ": decode batch 4 (seq {seq4}) holds {held} request(s), expected exactly the {b} of the run"

PREFILL_FAULTS = [
    # (stage, FAKE fault, reason). None stands for a reason that holds the repr of what the engine returned.
    ("meas", "foreign_id", _FOREIGN_IN_EXTEND),
    ("warm", "foreign_id", _FOREIGN_IN_EXTEND),
    ("meas", "split_extend", _SPLIT),
    ("warm", "split_extend", _SPLIT),
    ("meas", "chunked", _CHUNKED),
    ("warm", "chunked", _CHUNKED),
    ("meas", "decode_follows", _DECODE_AFTER),
    ("warm", "decode_follows", _DECODE_AFTER),
    ("meas", "wrong_extend_tokens", _TOKENS_REPORTED),
    ("meas", "wrong_extend_lens", _TOKENS_COUNTED),
    ("meas", "prompt_count", _PROMPT),
    ("seed", "prompt_count", _PROMPT),
    ("meas", "retraction", _RETRACTED),
    ("seed", "retraction", _RETRACTED),
    ("meas", "finish_stop", _STOPPED),
    ("warm", "short_completion", _SHORT),
    ("meas", "raise", _RAISED),
    ("seed", "raise", _RAISED),
    ("meas", "missing_key", "engine_call_failed: {rid0}: cached_tokens is None"),
    ("meas", "wrong_rid", None),
    ("meas", "short_list", None),
    ("meas", "copy_done_false", _NO_COPY_EVENT),
    ("meas", "no_lens", _NO_LENS),
    ("meas", "short_lens", _NO_LENS),
    ("meas", "idle_with_ids_follows", _TOKENS + ": a IDLE batch (seq {seq1}) holds requests of a one-token call"),
    ("meas", "raise_value_error", "engine_call_failed: ValueError: FAKE engine failure"),
    ("meas", "zero_duration", _NO_TIME.replace("{what}", "extend batch seq {seq}")),
    # A time that is not a JSON float was not read off the scheduler's clock.
    ("meas", "no_out_ts", _RECORDER + ": batch seq {seq} has out_ts=None"),
    ("meas", "whole_number_launch_ts", _RECORDER + ": batch seq {seq} has launch_ts=5000"),
    ("meas", "fault_line", _FAULT_LINE),
    ("seed", "fault_line", _FAULT_LINE),
]

DECODE_FAULTS = [
    ("run", "foreign_id", _CONTEXT + ": extend batch seq {seq} holds a foreign request"),
    ("run", "wrong_extend_lens", _EXTEND_TOKENS),
    ("run", "wrong_extend_tokens", _EXTEND_TOKENS),
    ("run", "second_extend_of_one_request", _CONTEXT + ": {rid0} is in 2 extend batches of the run"),
    ("run", "seven_decode", _SEQUENCE.replace("{found}", "1 EXTEND, 7 DECODE")),
    ("run", "extend_after_decode", _SEQUENCE.replace("{found}", "1 DECODE, 1 EXTEND, 5 DECODE")),
    ("run", "foreign_between", _BETWEEN),
    ("run", "decode_missing_id", _DECODE_IDS.replace("{held}", "{b0}")),
    ("run", "decode_foreign_id", _DECODE_IDS.replace("{held}", "{b1}")),
    ("run", "five_decode", "no_fpm_before_deadline: sched-4100.jsonl holds 5 of the 6 decode batch(es) after 2.0 s"),
    ("run", "prompt_count", _PROMPT),
    ("seed", "prompt_count", _PROMPT),
    ("run", "retraction", _RETRACTED),
    ("seed", "retraction", _RETRACTED),
    ("run", "short_completion", _SHORT),
    ("run", "finish_stop", _STOPPED),
    ("run", "raise", _RAISED),
    ("seed", "raise", _RAISED),
    ("run", "copy_done_false", _NO_COPY_EVENT),
    ("run", "decode_copy_done_false", _NO_COPY_EVENT.replace("{seq}", "{seq4}")),
    ("run", "zero_duration", _NO_TIME.replace("{what}", "decode step 1 (seq {seq1})")),
    ("run", "no_out_ts", _RECORDER + ": batch seq {seq} has out_ts=None"),
    ("run", "fault_line", _FAULT_LINE),
]


def _fill(reason: str | None, fake: _FakeRun, victim: grid.Point, stage: str) -> str | None:
    """Fill the numbers of an expected reason in from the victim and from the FAKE leader log."""
    if reason is None:
        return None
    (call,) = fake.calls_of(victim.benchmark_id, stage)
    lines = [json.loads(line) for line in (fake.recorder_dir / LEADER_LOG).read_text(encoding="utf-8").splitlines()]
    mine = [line for line in lines if line["type"] == "batch" and set(line["rids"]) & set(call.rids)]
    seqs = [line["seq"] for line in mine]
    seq = seqs[0] if seqs else 0
    first = mine[0] if mine else {}
    batch = victim.batch_size
    planned_new = batch * victim.new_tokens
    extend = victim.new_tokens
    if victim.phase == grid.DECODE:
        extend = grid.expected_extend_tokens(grid.DECODE, 0, victim.cached_tokens, fake.limits.page_size)
    return reason.format(
        seq=seq,
        seq1=seq + 1,
        seq4=seq + 4,
        seqs=seqs,
        b=batch,
        b0=batch - 1,
        b1=batch + 1,
        bn=planned_new,
        bn1=planned_new + 1,
        p=len(call.prompts[0]),
        p1=len(call.prompts[0]) + 1,
        m=call.max_new_tokens,
        m0=call.max_new_tokens - 1,
        e=extend,
        lens=first.get("extend_lens"),
        total=first.get("extend_num_tokens"),
        rid0=_RID.format(id=victim.benchmark_id, stage=stage, slot=0),
        rid_last=_RID.format(id=victim.benchmark_id, stage=stage, slot=batch - 1),
    )


def _assert_stopped_at_a_check(fake: _FakeRun, victim: grid.Point, stage: str, reason: str | None) -> None:
    """The faulty call failed its point with ``reason``; nothing was staged again and nothing followed."""
    expected = _fill(reason, fake, victim, stage)
    if expected is None:
        start = f"benchmark_id={victim.benchmark_id}: benchmark point failed: engine_call_failed: "
        assert fake.run.reason.startswith(start), fake.run.reason
        expected = fake.run.reason.removeprefix(f"benchmark_id={victim.benchmark_id}: benchmark point failed: ")
    _assert_point_failed(fake, victim, expected)
    # No re-stage: only a wrong cached count is staged again.
    assert fake.payload()["producer"]["restaged_points"] == []
    assert {parts["attempt"] for parts in fake.sent() if parts["benchmark_id"]} == {0}
    # The calls of the point after the faulty one were not sent.
    stages = ["seed", "warm", "meas"] if victim.phase == grid.PREFILL else ["seed", "run"]
    sent = [call.key[1] for call in fake.calls_of(victim.benchmark_id)]
    assert sent == stages[: stages.index(stage) + 1]


@pytest.mark.parametrize(
    ("stage", "fault", "reason"), PREFILL_FAULTS, ids=[f"{row[0]}-{row[1]}" for row in PREFILL_FAULTS]
)
def test_a_prefill_point_that_fails_a_check_ends_the_run_with_its_reason(tmp_path, stage, fault, reason):
    victim = _victim(_plan_only(tmp_path, "prefill"))
    assert (victim.batch_size, victim.new_tokens) == (2, 4) and victim.cached_tokens > 0
    fake = _start(tmp_path, "prefill", faults={(victim.benchmark_id, stage, 0): fault})
    _assert_stopped_at_a_check(fake, victim, stage, reason)


@pytest.mark.parametrize(
    ("stage", "fault", "reason"), DECODE_FAULTS, ids=[f"{row[0]}-{row[1]}" for row in DECODE_FAULTS]
)
def test_a_decode_point_that_fails_a_check_ends_the_run_with_its_reason(tmp_path, stage, fault, reason):
    victim = _victim(_plan_only(tmp_path, "decode"))
    assert victim.batch_size >= 8 and grid.expected_cached_tokens("decode", victim.cached_tokens, 1) > 0
    fake = _start(tmp_path, "decode", faults={(victim.benchmark_id, stage, 0): fault})
    _assert_stopped_at_a_check(fake, victim, stage, reason)


@pytest.mark.parametrize(
    ("mode", "stage", "fault", "reason"),
    [
        ("prefill", "meas", "follower_missing", "sched-4101.jsonl lacks 1 of the 1 batch(es) of the call"),
        ("prefill", "warm", "follower_missing", "sched-4101.jsonl lacks 1 of the 1 batch(es) of the call"),
        ("prefill", "meas", "follower_other_ids", "sched-4101.jsonl lacks 1 of the 1 batch(es) of the call"),
        ("prefill", "meas", "follower_other_mode", "sched-4101.jsonl lacks 1 of the 1 batch(es) of the call"),
        ("decode", "run", "follower_missing", "sched-4101.jsonl lacks 7 of the 7 batch(es) of the call"),
    ],
)
def test_a_follower_log_without_the_batches_of_a_call_fails_the_point_at_the_deadline(
    tmp_path, mode, stage, fault, reason
):
    victim = _victim(_plan_only(tmp_path, mode, processes=2))
    fake = _start(tmp_path, mode, processes=2, faults={(victim.benchmark_id, stage, 0): fault})

    _assert_point_failed(fake, victim, f"no_fpm_before_deadline: {reason} after 2.0 s")
    # The driver waited on the FAKE clock for the whole of log_wait_seconds, in steps of the poll interval.
    assert fake.clock.sleeps == 2.0 / driver.POLL_SECONDS


def test_a_follower_that_writes_late_is_waited_for(tmp_path):
    victim = _victim(_plan_only(tmp_path, "decode", processes=2))
    fake = _start(tmp_path, "decode", processes=2, faults={(victim.benchmark_id, "run", 0): "follower_late"})

    assert fake.clock.sleeps == 3 and fake.engine.late == []
    _assert_accepted(fake)


def test_a_follower_is_not_waited_for_in_a_seed(tmp_path):
    # Nothing of a seed is kept, so only the leader's log is read for it.
    victim = _victim(_plan_only(tmp_path, "decode", processes=2))
    fake = _start(tmp_path, "decode", processes=2, faults={(victim.benchmark_id, "seed", 0): "follower_missing"})

    assert fake.clock.sleeps == 0
    _assert_accepted(fake)


def test_a_log_without_any_batch_of_a_call_fails_at_the_deadline(tmp_path):
    victim = _victim(_plan_only(tmp_path, "prefill"))
    # FAKE: the recorder writes nothing at all for the measured shot.
    fake = _start(tmp_path, "prefill", faults={(victim.benchmark_id, "meas", 0): "silent"})

    _assert_point_failed(
        fake,
        victim,
        "no_fpm_before_deadline: sched-4100.jsonl holds no extend batch for 2 of 2 request(s) after 2.0 s",
    )


def test_a_decode_context_that_is_not_the_planned_one_fails_the_counter_check(tmp_path, monkeypatch):
    # A FAKE planning error: every decode prompt is one token longer than the point's context less three. The
    # engine does what it is asked, every response matches its prompt, and the FAKE recorder line hides the
    # extra token of the extend batch. Only the counters of the kept decode batch, taken from the prompt
    # tokens the engine reported, show that the middle kept step read one token more than planned.
    real = content.ContentPlan.decode_prompt
    monkeypatch.setattr(content.ContentPlan, "decode_prompt", lambda self, rank, slot, k: real(self, rank, slot, k + 1))
    manifest = grid.parse_points_manifest(
        json.dumps({"schema_version": 1, "prefill": [], "decode": [{"batch_size": 2, "total_kv_read_tokens": 80}]})
    )
    fake = _start(tmp_path, "decode", 64, manifest=manifest, faults={(1, "run", 0): "extend_lens_one_less"})

    (run,) = fake.calls_of(1, "run")
    assert [len(prompt) for prompt in run.prompts] == [38, 38]  # planned: 40 - 3 = 37
    assert fake.run.reason == (
        "benchmark_id=1: benchmark point failed: measured_decode_context_mismatch: sum_decode_kv_tokens is 82 "
        "in the kept batch, the point plans 80"
    )
    assert fake.payload()["producer"]["restaged_points"] == []


def test_a_fault_line_between_two_points_fails_the_run_without_naming_a_point(tmp_path, monkeypatch):
    fake = _start(tmp_path, "prefill", execute=False)
    real = artifact.build_measurement
    victim = 3

    def build_then_fault(record, rank_fpms):
        # FAKE: the recorder faults after the batches of the point were read and before the next point starts.
        if record["benchmark_id"] == victim:
            fake.engine.logs[0].write({"type": "fault", "where": "run_batch", "error": "FAKE late fault"})
        return real(record, rank_fpms)

    monkeypatch.setattr(artifact, "build_measurement", build_then_fault)
    fake.execute()

    reason = "recorder_fault: sched-4100.jsonl: run_batch: FAKE late fault"
    assert (fake.run.status, fake.run.reason) == ("failed", reason)
    assert max(fake.sent_benchmark_ids()) == victim
    on_disk = _assert_rejected(fake, "failed")
    assert (on_disk["error"], on_disk["skipped_points"]) == (reason, [])
    assert on_disk["coverage"]["completed_points"] == victim


def test_a_fault_line_after_the_last_point_fails_a_run_that_has_every_row(tmp_path, monkeypatch):
    fake = _start(tmp_path, "decode", execute=False)
    real = artifact.build_measurement
    last = len(fake.built.points)

    def build_then_fault(record, rank_fpms):
        if record["benchmark_id"] == last:
            fake.engine.logs[0].write({"type": "fault", "where": "header", "error": "FAKE late fault"})
            fake.engine.logs[0].write({"type": "fault", "where": "header", "error": "FAKE second fault"})
        return real(record, rank_fpms)

    monkeypatch.setattr(artifact, "build_measurement", build_then_fault)
    fake.execute()

    assert fake.run.reason == "recorder_fault: sched-4100.jsonl: header: FAKE late fault (and 1 more)"
    on_disk = _assert_rejected(fake, "failed")
    assert on_disk["coverage"] == {"expected_points": last, "completed_points": last, "skipped_points": 0}


# ---------------------------------------------------------------------------------------------------------------
# Recorder headers
# ---------------------------------------------------------------------------------------------------------------


def test_a_missing_header_fails_the_run_after_the_header_wait(tmp_path):
    fake = _start(tmp_path, headers=False)

    _assert_failed_before_the_grid(
        fake, f"recorder_header_invalid: 0 of 1 recorder header(s) in {fake.recorder_dir} after 2.0 s"
    )
    assert fake.clock.sleeps == 2.0 / driver.POLL_SECONDS


def test_a_header_that_arrives_late_is_waited_for(tmp_path):
    fake = _start(tmp_path, "decode", headers=False, execute=False)

    def write_headers_late() -> None:
        if fake.clock.sleeps == 5:
            fake.engine.write_headers()

    fake.clock.on_sleep.append(write_headers_late)
    fake.execute()

    assert fake.clock.sleeps == 5
    _assert_accepted(fake)


def test_one_header_of_two_fails_the_run(tmp_path):
    fake = _start(tmp_path, processes=2, edit_header=lambda index, header: header if index == 0 else None)
    _assert_failed_before_the_grid(
        fake, f"recorder_header_invalid: 1 of 2 recorder header(s) in {fake.recorder_dir} after 2.0 s"
    )


def test_more_headers_than_scheduler_processes_fail_the_run(tmp_path):
    fake = _start(tmp_path, processes=2, expected_scheduler_processes=1)
    _assert_failed_before_the_grid(
        fake, "recorder_header_invalid: 2 recorder headers, expected 1: ['sched-4100.jsonl', 'sched-4101.jsonl']"
    )


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        # The stock scheduler was built: the class replacement did not apply.
        (_edit("class", "Scheduler"), "sched-4100.jsonl: class 'Scheduler', expected 'SglbenchRecorderScheduler'"),
        (_edit("format", 2), "sched-4100.jsonl: format 2, expected 1"),
        (_edit("format", True), "sched-4100.jsonl: format True, expected 1"),
        (_edit("leader", False), "0 leader header(s), expected exactly one: []"),
        (_edit("leader", 1), "sched-4100.jsonl: leader is 1, expected true or false"),
        (_edit("checks", {}), "sched-4100.jsonl: the header has no checks.disable_radix_cache"),
        (_edit("limits", {"page_size": 1}), "sched-4100.jsonl: the header has no limits.max_req_input_len"),
        (
            _edit("limits.max_total_num_tokens", 1185),
            "sched-4100.jsonl: limits: max_total_num_tokens 1185 is not a multiple of page_size 16",
        ),
        (
            _edit("limits.max_running_requests", 9.0),
            "sched-4100.jsonl: limits: max_running_requests must be an int, got float: 9.0",
        ),
    ],
)
def test_an_invalid_header_fails_the_run(tmp_path, edit, reason):
    limits = _limits(16)
    # A float cap equals the micro-batch cap, so only the limits record refuses it.
    fake = _start(tmp_path, limits=limits, edit_header=edit)
    _assert_failed_before_the_grid(fake, f"recorder_header_invalid: {reason}")


def test_two_leaders_fail_the_run(tmp_path):
    fake = _start(tmp_path, processes=2, edit_header=_edit("leader", True))
    _assert_failed_before_the_grid(
        fake,
        "recorder_header_invalid: 2 leader header(s), expected exactly one: ['sched-4100.jsonl', 'sched-4101.jsonl']",
    )


UNREADABLE = {"unreadable": "RuntimeError('FAKE: not published')"}


def _unreadable(path: str) -> tuple:
    """A header value the recorder could not read, and the rejection it must give."""
    return path, UNREADABLE, f"the recorder could not read {path} in sched-4100.jsonl: {UNREADABLE['unreadable']}"


_RADIX_OFF = "the radix cache is off (disable_radix_cache={v}); SGLang may have disabled it for this model"
_HYBRID = "hybrid SWA or state-space models are not supported ({name}=true)"
_POLICY = "schedule policy must be fcfs without priority scheduling ({name})"
_AMD = "only NVIDIA CUDA is supported (is_hip={v}); SGLang has AMD-only admission rules"
_INTERVAL = "prefill_decode_interval must be 0 (got {v}); SGLang sets 22 for some Qwen3-VL setups"
_OUTPUT_CAP = "SGLANG_MAX_NEW_TOKENS_LIMIT={v} is below the 7 output tokens of a decode run"

STAGE_H_REJECTIONS = [
    (
        "enable_overlap",
        False,
        "overlap scheduling is off (enable_overlap=false); the output time of a batch would not follow its GPU work",
    ),
    ("checks.disable_radix_cache", True, _RADIX_OFF.format(v="true")),
    ("checks.is_hybrid_swa", True, _HYBRID.format(name="is_hybrid_swa")),
    ("checks.is_hybrid_ssm", True, _HYBRID.format(name="is_hybrid_ssm")),
    ("checks.is_mixed_chunk", True, "--enable-mixed-chunk is not supported (is_mixed_chunk=true)"),
    ("checks.enable_priority_scheduling", True, _POLICY.format(name="enable_priority_scheduling=true")),
    (
        "checks.prefill_delayer",
        True,
        "the prefill delayer is not supported (also check SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE)",
    ),
    ("checks.min_free_slots_delayer", True, "min_free_slots_delay must not be set (got min_free_slots_delayer=true)"),
    ("checks.recv_skipper", True, "scheduler_recv_interval must be 1 (got recv_skipper=true)"),
    (
        "checks.input_blocker",
        True,
        "SGLANG_ENABLE_COLOCATED_BATCH_GEN must be unset without DP attention (input_blocker=true)",
    ),
    ("checks.is_hip", True, _AMD.format(v="true")),
    # Only the boolean false passes: a recorder that wrote another falsy value did not write format 1.
    ("checks.is_hip", None, _AMD.format(v="null")),
    ("checks.disable_radix_cache", 0, _RADIX_OFF.format(v="0")),
    ("checks.schedule_policy", "lpm", _POLICY.format(name="schedule_policy=lpm")),
    ("checks.prefill_decode_interval", 22, _INTERVAL.format(v="22")),
    ("checks.prefill_decode_interval", False, _INTERVAL.format(v="false")),
    ("checks.max_new_tokens_limit", 6, _OUTPUT_CAP.format(v="6")),
    ("checks.max_new_tokens_limit", 1, _OUTPUT_CAP.format(v="1")),
    ("checks.max_new_tokens_limit", True, _OUTPUT_CAP.format(v="true")),
    ("checks.pp_max_micro_batch_size", 8, "pp_max_micro_batch_size 8 differs from the running cap 9"),
    ("checks.pp_max_micro_batch_size", None, "pp_max_micro_batch_size null differs from the running cap 9"),
    ("ranks.attn_dp_enabled", True, "DP attention is not supported yet (attn_dp_size=1)"),
    ("ranks.attn_dp_size", 2, "DP attention is not supported yet (attn_dp_size=2)"),
    ("ranks.pp_size", 2, "pipeline parallelism is not supported (pp_size=2)"),
    ("sglang_version", "0.5.24", "SGLang 0.5.24 is not a tested version (tested: 0.5.22, 0.5.23)"),
    ("sglang_version", "0.5.2", "SGLang 0.5.2 is not a tested version (tested: 0.5.22, 0.5.23)"),
    # A value the recorder could not read is never taken for an allowed one.
    _unreadable("leader"),
    _unreadable("enable_overlap"),
    _unreadable("sglang_version"),
    _unreadable("decode_graph_sizes"),
    _unreadable("prefill_graph_sizes"),
    _unreadable("ranks.tp_size"),
    _unreadable("ranks.pp_size"),
    _unreadable("ranks.attn_dp_size"),
    _unreadable("ranks.attn_dp_enabled"),
    *(_unreadable(f"limits.{name}") for name in _fake_header(0, 0, _limits(1), 1)["limits"]),
    *(_unreadable(f"checks.{name}") for name in _fake_header(0, 0, _limits(1), 1)["checks"]),
]


@pytest.mark.parametrize(
    ("path", "value", "reason"), STAGE_H_REJECTIONS, ids=[f"{row[0]}={row[1]!r:.12}" for row in STAGE_H_REJECTIONS]
)
def test_a_scheduler_setting_the_rules_do_not_cover_is_rejected(tmp_path, path, value, reason):
    fake = _start(tmp_path, edit_header=_edit(path, value))
    on_disk = _assert_failed_before_the_grid(fake, f"rejected: {reason}")
    assert on_disk["producer"]["scheduler_processes"] == 1


def test_overlap_off_is_recorded_in_the_rejected_artifact(tmp_path):
    fake = _start(tmp_path, edit_header=_edit("enable_overlap", False))
    assert fake.payload()["producer"]["enable_overlap"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("checks.prefill_decode_interval", None),
        ("checks.max_new_tokens_limit", 0),
        ("checks.max_new_tokens_limit", -1),
        ("checks.max_new_tokens_limit", 7),
        ("checks.max_new_tokens_limit", 4096),
        ("checks.max_queued_requests", 9),
        ("sglang_version", "0.5.22"),
        ("limits.chunked_prefill_size", None),
        ("decode_graph_sizes", []),
        ("prefill_graph_sizes", []),
        # Ranks the checks do not use may be unreadable.
        ("ranks.attn_cp_rank", UNREADABLE),
    ],
)
def test_allowed_header_values_do_not_reject(tmp_path, path, value):
    fake = _start(tmp_path, "decode", edit_header=_edit(path, value), limits=_limits(1))
    assert (fake.run.status, fake.run.reason) == ("complete", None)


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        ("limits.page_size", 16, "page_size: [1, 16]"),
        ("limits.max_req_input_len", 427, "max_req_input_len: [428, 427]"),
        ("limits.max_total_num_tokens", 1200, "max_total_num_tokens: [1184, 1200]"),
        ("limits.max_prefill_tokens", 96, "max_prefill_tokens: [80, 96]"),
        ("limits.chunked_prefill_size", None, "chunked_prefill_size: [96, null]"),
        ("decode_graph_sizes", [1, 2, 4], "decode_graph_sizes: [[1, 2, 4, 8], [1, 2, 4]]"),
        ("prefill_graph_sizes", [], "prefill_graph_sizes: [[4, 8, 16, 32, 64], []]"),
        ("ranks.tp_size", 1, "tp_size: [2, 1]"),
        ("sglang_version", "0.5.22", 'sglang_version: ["0.5.23", "0.5.22"]'),
    ],
)
def test_headers_that_disagree_are_rejected(tmp_path, path, value, reason):
    fake = _start(tmp_path, processes=2, edit_header=_edit(path, value, process=1))
    _assert_failed_before_the_grid(fake, f"rejected: scheduler processes disagree on {reason}")


def test_a_running_cap_that_differs_between_processes_is_rejected(tmp_path):
    def edit(index: int, header: dict) -> dict:
        if index == 1:
            header["limits"]["max_running_requests"] = 8
            header["checks"]["pp_max_micro_batch_size"] = 8
        return header

    fake = _start(tmp_path, processes=2, edit_header=edit)
    _assert_failed_before_the_grid(fake, "rejected: scheduler processes disagree on max_running_requests: [9, 8]")


def test_a_bad_setting_in_the_follower_alone_is_rejected(tmp_path):
    fake = _start(tmp_path, processes=2, edit_header=_edit("checks.is_mixed_chunk", True, process=1))
    _assert_failed_before_the_grid(fake, "rejected: --enable-mixed-chunk is not supported (is_mixed_chunk=true)")


# ---------------------------------------------------------------------------------------------------------------
# Grid, request caps and token range
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_a_phase_without_a_feasible_point_fails_the_run(tmp_path, mode):
    # An input limit of one token admits no prompt at all.
    fake = _start(tmp_path, mode, limits=_limits(16, max_req_input_len=1))

    reason = f"no feasible benchmark points for phase(s): {mode}"
    assert (fake.run.status, fake.run.reason) == ("failed", reason)
    assert fake.engine.calls == []
    on_disk = _assert_rejected(fake, "failed")
    assert (on_disk["error"], on_disk["missing_phases"]) == (reason, [mode])
    assert on_disk["grid_digest"] == fake.built.digest


@pytest.mark.parametrize(
    ("overrides", "edit", "reason"),
    [
        ({"prefill_max_requests": 8}, None, "prefill_max_requests=8 is below the largest batch 9"),
        ({"max_queued_requests": 8}, None, "max_queued_requests=8 is below the largest batch 9"),
        ({"max_queued_requests": 0}, None, "max_queued_requests=0 is below the largest batch 9"),
        ({}, _edit("checks.max_queued_requests", 8), "max_queued_requests=8 is below the largest batch 9"),
        ({"prefill_max_requests": 9.0}, None, "prefill_max_requests=9.0 is below the largest batch 9"),
    ],
)
def test_a_request_cap_below_the_largest_batch_is_rejected(tmp_path, overrides, edit, reason):
    fake = _start(tmp_path, "decode", edit_header=edit, **overrides)

    assert max(point.batch_size for point in fake.built.points) == 9
    assert (fake.run.status, fake.run.reason) == ("failed", f"rejected: {reason}")
    assert fake.engine.calls == []
    # The grid exists by then: the artifact carries its digest and its size.
    on_disk = _assert_rejected(fake, "failed")
    assert on_disk["grid_digest"] == fake.built.digest
    assert on_disk["coverage"]["expected_points"] == len(fake.built.points)


def test_request_caps_at_the_largest_batch_are_accepted(tmp_path):
    fake = _start(
        tmp_path,
        "decode",
        prefill_max_requests=9,
        max_queued_requests=9,
        edit_header=_edit("checks.max_queued_requests", 9),
    )
    assert fake.run.status == "complete"
    # In a prefill run the largest batch is 2, and a cap of 2 is enough.
    prefill = _start(tmp_path / "prefill", "prefill", prefill_max_requests=2, max_queued_requests=2)
    assert max(point.batch_size for point in prefill.built.points) == 2
    assert prefill.run.status == "complete"


def test_a_token_range_below_the_content_minimum_fails_the_run(tmp_path):
    fake = _start(tmp_path, tokens=(1000, 5095))

    reason = "token_range_too_small: the token range [1000, 5095) holds 4095 ids, the content plan needs at least 4096"
    assert (fake.run.status, fake.run.reason) == ("failed", reason)
    assert fake.engine.calls == []
    _assert_rejected(fake, "failed")


def test_a_run_that_needs_more_first_tokens_than_the_range_holds_fails_before_any_request(tmp_path):
    # A running cap of 5000 plans decode batches of 5000 requests: 5000 slots, four free prompts of the
    # self-test and the warm-up prompt.
    limits = _limits(1, max_running_requests=5000, max_total_num_tokens=1_000_000)
    fake = _start(tmp_path, "decode", limits=limits, tokens=(1000, 6004))

    assert max(point.batch_size for point in fake.built.points) == 5000
    need = {
        "slots": 5000,
        "zero_cached_prefill_prompts": 0,
        "warmup_prompts": 1,
        "self_test_free_prompts": 4,
        "restage_reserve": 0,
        "total": 5005,
    }
    reason = (
        f"token_range_too_small: the run needs 5005 first tokens ({need}) and the token range [1000, 6004) "
        "holds 5004 ids"
    )
    assert (fake.run.status, fake.run.reason) == ("failed", reason)
    assert fake.engine.calls == []
    _assert_rejected(fake, "failed")


def test_a_token_range_of_exactly_the_planned_need_is_enough(tmp_path, monkeypatch):
    # The content plan accepts a small range here, so that a small grid can fill it to the last id.
    monkeypatch.setattr(content, "MIN_VOCAB_SIZE", 16)
    need = _start(tmp_path / "need", "prefill").payload()["producer"]["first_token_need"]
    assert need["restage_reserve"] == 4 and need["warmup_prompts"] == 1

    exact = _start(tmp_path / "exact", "prefill", tokens=(1000, 1000 + need["total"]))
    assert exact.run.status == "complete"
    short = _start(tmp_path / "short", "prefill", tokens=(1000, 1000 + need["total"] - 1))
    assert short.run.reason.startswith(f"token_range_too_small: the run needs {need['total']} first tokens (")
    assert short.engine.calls == []
    # Without the warm-up the run needs one first token less.
    no_warmup = _start(tmp_path / "skip", "prefill", tokens=(1000, 1000 + need["total"] - 1), warmup_iterations=0)
    assert no_warmup.run.status == "complete"
    assert no_warmup.payload()["producer"]["first_token_need"]["total"] == need["total"] - 1


def test_a_grid_without_any_slot_still_plans_the_two_slots_of_the_self_test(tmp_path):
    manifest = grid.parse_points_manifest(
        json.dumps(
            {
                "schema_version": 1,
                "prefill": [{"batch_size": 3, "total_prefill_tokens": 12, "total_kv_read_tokens": 0}],
                "decode": [],
            }
        )
    )
    fake = _start(tmp_path, "prefill", manifest=manifest)

    assert fake.run.status == "complete"
    producer = fake.payload()["producer"]
    assert producer["first_token_need"] == {
        "slots": 2,
        "zero_cached_prefill_prompts": 6,
        "warmup_prompts": 1,
        "self_test_free_prompts": 4,
        "restage_reserve": 6,
        "total": 19,
    }
    issued = producer["content"]["issued"]
    assert (issued["slots"], issued["zero_cached_prompts"], issued["free_prompts"]) == (2, 6, 5)


def test_a_second_restaged_zero_cached_point_runs_out_of_first_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(content, "MIN_VOCAB_SIZE", 16)
    plan = _plan_only(tmp_path, "prefill")
    zero_cached = [point for point in plan.built.points if not point.cached_tokens and point.batch_size == 2]
    first, second = zero_cached[:2]
    need = _start(tmp_path / "need", "prefill").payload()["producer"]["first_token_need"]
    faults = {(first.benchmark_id, "warm", 0): "cached_off", (second.benchmark_id, "warm", 0): "cached_off"}

    # With room to spare both points recover on their re-stage.
    roomy = _start(tmp_path / "roomy", "prefill", faults=faults)
    assert roomy.run.status == "complete"
    assert roomy.payload()["producer"]["restaged_points"] == [first.benchmark_id, second.benchmark_id]

    # In a range of exactly the planned need the reserve covers one re-stage of two requests.
    exact = _start(tmp_path / "exact", "prefill", faults=faults, tokens=(1000, 1000 + need["total"]))
    assert exact.run.reason == (
        f"benchmark_id={second.benchmark_id}: benchmark point failed: token_range_too_small: the re-stage needs 4 "
        f"more first tokens and 0 are left in [1000, {1000 + need['total']})"
    )
    assert exact.payload()["producer"]["restaged_points"] == [first.benchmark_id, second.benchmark_id]
    # Nothing was sent for the second attempt of the second point.
    assert [call.key[2] for call in exact.calls_of(second.benchmark_id)] == [0]


# ---------------------------------------------------------------------------------------------------------------
# Self-test and warm-up
# ---------------------------------------------------------------------------------------------------------------

_FREE_RID = "sglb-fa4e0123-p0-free{call}-r0-s0-e"
_OTHER_COUNT = "{wrong} of 2 request(s) report another cached count than the planned {planned}: {rid} reports {got}"
_ALL_BUT_LAST = ["batch_shape", "aligned_prefix", "unaligned_prompt"]

SELF_TEST_FAULTS = [
    # (call, FAKE fault, reason after "self_test_failed: ", the checks that passed before)
    (FREE_BATCH_SHAPE, "split_extend", "batch_shape: " + _SPLIT, []),
    (FREE_BATCH_SHAPE, "decode_follows", "batch_shape: " + _DECODE_AFTER.format(seq1=2), []),
    (
        FREE_BATCH_SHAPE,
        "cached_off",
        "batch_shape: " + _OTHER_COUNT.format(wrong=1, planned=0, rid=_FREE_RID.format(call=0), got=1),
        [],
    ),
    (
        FREE_ALIGNED_SHOTS,
        "evict",
        "aligned_prefix: " + _OTHER_COUNT.format(wrong=2, planned=16, rid=_FREE_RID.format(call=2), got=0),
        ["batch_shape"],
    ),
    (FREE_ALIGNED_SHOTS, "split_extend", "aligned_prefix: " + _SPLIT, ["batch_shape"]),
    (
        FREE_UNALIGNED_AGAIN,
        "evict",
        "unaligned_prompt: " + _OTHER_COUNT.format(wrong=2, planned=38, rid=_FREE_RID.format(call=4), got=0),
        ["batch_shape", "aligned_prefix"],
    ),
    (
        # An engine that counted the whole prompt as cached, with no token left to compute.
        FREE_UNALIGNED_AGAIN,
        "cached_off",
        "unaligned_prompt: " + _OTHER_COUNT.format(wrong=1, planned=38, rid=_FREE_RID.format(call=4), got=39),
        ["batch_shape", "aligned_prefix"],
    ),
    (
        FREE_SEVEN_TOKEN_RUN,
        "seven_decode",
        "seven_token_run: " + _SEQUENCE.replace("{found}", "1 EXTEND, 7 DECODE"),
        _ALL_BUT_LAST,
    ),
    (
        FREE_SEVEN_TOKEN_RUN,
        "decode_missing_id",
        "seven_token_run: " + _DECODE_IDS.format(seq4=10, held=1, b=2),
        _ALL_BUT_LAST,
    ),
    (
        FREE_SEVEN_TOKEN_RUN,
        "split_extend",
        "seven_token_run: measured_decode_context_mismatch: the run has 2 extend batches, expected one",
        _ALL_BUT_LAST,
    ),
    (
        FREE_SEVEN_TOKEN_RUN,
        "cached_off",
        "seven_token_run: " + _OTHER_COUNT.format(wrong=1, planned=0, rid=_FREE_RID.format(call=5), got=1),
        _ALL_BUT_LAST,
    ),
    (FREE_ALIGNED_SEED, "raise", "aligned_prefix: " + _RAISED, ["batch_shape"]),
    (FREE_UNALIGNED_SEED, "retraction", None, ["batch_shape", "aligned_prefix"]),
]


@pytest.mark.parametrize(
    ("call", "fault", "reason", "passed"), SELF_TEST_FAULTS, ids=[f"{row[0]}-{row[1]}" for row in SELF_TEST_FAULTS]
)
def test_a_failed_self_test_check_ends_the_run_before_the_sweep(tmp_path, call, fault, reason, passed):
    fake = _start(tmp_path, "prefill", faults={(0, "free", call): fault})

    if reason is None:
        reason = f"{driver.SELF_TEST_CHECKS[len(passed)]}: request_not_completed: " + (
            f"sglb-fa4e0123-p0-free{call}-r0-s1-e: finish_type='length', completion_tokens=1 of 1, num_retractions=1"
        )
    assert (fake.run.status, fake.run.reason) == ("failed", f"self_test_failed: {reason}")
    # No point was staged and no later check ran.
    assert fake.sent_benchmark_ids() == {0}
    assert [made.key[2] for made in fake.engine.calls] == list(range(call + 1))
    on_disk = _assert_rejected(fake, "failed")
    assert on_disk["producer"]["self_test"] == passed
    assert (on_disk["skipped_points"], on_disk["coverage"]["completed_points"]) == ([], 0)
    assert on_disk["coverage"]["expected_points"] == len(fake.built.points)


@pytest.mark.parametrize(
    ("call", "check", "batches"),
    [
        (FREE_BATCH_SHAPE, "batch_shape", 1),
        (FREE_ALIGNED_SHOTS, "aligned_prefix", 1),
        (FREE_SEVEN_TOKEN_RUN, "seven_token_run", 7),
    ],
)
def test_the_checked_calls_of_the_self_test_wait_for_the_follower_log(tmp_path, call, check, batches):
    fake = _start(tmp_path, processes=2, faults={(0, "free", call): "follower_missing"})
    assert fake.run.reason == (
        f"self_test_failed: {check}: no_fpm_before_deadline: sched-4101.jsonl lacks {batches} of the {batches} "
        "batch(es) of the call after 2.0 s"
    )
    assert fake.sent_benchmark_ids() == {0}


@pytest.mark.parametrize("page_size", [1, 3, 16, 24, 64])
def test_the_self_test_expects_the_cached_counts_of_the_page_size(tmp_path, page_size):
    fake = _start(tmp_path, "decode", limits=_limits(page_size))
    assert fake.run.status == "complete"

    unit = -(-16 // page_size) * page_size  # the smallest page multiple of at least 16 tokens
    free = [call for call in fake.engine.calls if call.key[0] == 0]
    assert [len(call.prompts[0]) for call in free[:6]] == [unit, unit, 2 * unit, 2 * unit + 7, 2 * unit + 7, unit]
    # What the fake engine's cache model reported for the two prefix checks: the whole aligned prefix, and the
    # unaligned prompt less its last token, rounded down to whole pages.
    lines = [json.loads(line) for line in (fake.recorder_dir / LEADER_LOG).read_text(encoding="utf-8").splitlines()[1:]]
    by_first_rid = {line["rids"][0]: line for line in lines}
    assert by_first_rid[_FREE_RID.format(call=FREE_ALIGNED_SHOTS)]["prefix_lens"] == [unit, unit]
    rounded = (2 * unit + 6) // page_size * page_size
    assert by_first_rid[_FREE_RID.format(call=FREE_UNALIGNED_AGAIN)]["prefix_lens"] == [rounded, rounded]
    assert rounded > unit


def test_limits_that_do_not_admit_a_self_test_call_fail_the_self_test(tmp_path):
    # A chunk budget of one cached unit holds one request of 16 tokens, not two.
    fake = _start(tmp_path, limits=_limits(1, chunked_prefill_size=16))
    assert fake.run.reason == (
        "self_test_failed: batch_shape: the limits of this engine do not admit two requests of 16 tokens in one "
        "extend batch: chunk_budget"
    )
    assert fake.engine.calls == []

    # An input limit of 32 tokens admits a prompt of 16 tokens and not one of 16 cached and 16 new tokens.
    aligned = _start(tmp_path / "aligned", limits=_limits(1, max_req_input_len=32))
    assert aligned.run.reason == (
        "self_test_failed: aligned_prefix: the limits of this engine do not admit two requests of 16 cached and "
        "16 new tokens in one extend batch: input_too_long"
    )
    assert [call.key[2] for call in aligned.engine.calls] == [FREE_BATCH_SHAPE]

    # An input limit of 39 tokens admits the first two checks and not the unaligned prompt of 39 tokens.
    short = _start(tmp_path / "short", limits=_limits(1, max_req_input_len=39), warmup_iterations=0)
    assert short.run.reason == (
        "self_test_failed: unaligned_prompt: the limits of this engine do not admit two requests of 39 tokens: "
        "input_too_long"
    )
    assert [call.key[2] for call in short.engine.calls] == [FREE_BATCH_SHAPE, FREE_ALIGNED_SEED, FREE_ALIGNED_SHOTS]


def test_an_unaligned_self_test_prompt_above_the_chunk_budget_may_be_chunked(tmp_path):
    # Two prompts of 39 tokens do not fit a chunk budget of 32 tokens in one extend batch. The first send is a
    # seed and may be chunked, so the check runs: 16 tokens of each prompt are cached, the first request
    # computes its 23 tokens, the second 9 of its 23 and the rest in a second extend batch.
    fake = _start(tmp_path, limits=_limits(1, chunked_prefill_size=32))
    free = [call for call in fake.engine.calls if call.key[0] == 0]
    assert free[FREE_UNALIGNED_SEED].modes == ["EXTEND", "EXTEND"]
    assert free[FREE_UNALIGNED_AGAIN].modes == ["EXTEND"]
    assert fake.run.status == "complete"


def test_the_warm_up_is_one_prompt_of_256_tokens_and_is_skipped_at_zero(tmp_path):
    with_warmup = _start(tmp_path / "five", "decode", warmup_iterations=5)
    free = [call for call in with_warmup.engine.calls if call.key[0] == 0]
    assert len(free) == 7
    warmup = free[FREE_WARMUP]
    assert ([len(prompt) for prompt in warmup.prompts], warmup.max_new_tokens) == ([256], 5)
    # 256 tokens are above the chunk budget of 96 here: three extend batches, then four decode batches.
    assert warmup.modes == ["EXTEND"] * 3 + ["DECODE"] * 4
    assert with_warmup.run.status == "complete"

    without = _start(tmp_path / "zero", "decode", warmup_iterations=0)
    free = [call for call in without.engine.calls if call.key[0] == 0]
    assert [call.key[2] for call in free] == list(range(6))
    assert not any(len(prompt) == 256 for call in without.engine.calls for prompt in call.prompts)
    assert without.run.status == "complete"
    assert without.payload()["producer"]["first_token_need"]["warmup_prompts"] == 0

    one = _start(tmp_path / "one", "decode", warmup_iterations=1)
    assert [call for call in one.engine.calls if call.key[0] == 0][FREE_WARMUP].modes == ["EXTEND"] * 3
    assert one.run.status == "complete"


def test_a_warm_up_that_does_not_run_to_its_last_token_fails_the_run(tmp_path):
    fake = _start(tmp_path, faults={(0, "free", FREE_WARMUP): "short_completion"})
    assert fake.run.reason == (
        f"request_not_completed: {_FREE_RID.format(call=6)}: finish_type='length', completion_tokens=4 of 5, "
        "num_retractions=0"
    )
    assert fake.sent_benchmark_ids() == {0}
    assert fake.payload()["producer"]["self_test"] == list(driver.SELF_TEST_CHECKS)


# ---------------------------------------------------------------------------------------------------------------
# Soft deadline
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_a_deadline_that_passes_before_the_first_point_gives_a_partial_run_without_rows(tmp_path, mode):
    # The self-test and the warm-up take FAKE milliseconds, which is more than this deadline.
    fake = _start(tmp_path, mode, timeout_seconds=0.001)

    assert (fake.run.status, fake.run.reason) == ("partial", "timeout")
    assert fake.sent_benchmark_ids() == {0}
    on_disk = _assert_rejected(fake, "partial")
    assert (on_disk["stop_reason"], on_disk["error"], on_disk["usable"]) == ("timeout", None, False)
    assert on_disk["coverage"] == {
        "expected_points": len(fake.built.points),
        "completed_points": 0,
        "skipped_points": 0,
    }


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_a_deadline_that_passes_between_two_points_gives_a_partial_run(tmp_path, mode):
    completed = 4
    stage = "meas" if mode == "prefill" else "run"
    fake = _start(tmp_path, mode, faults={(completed, stage, 0): "deadline"})

    assert (fake.run.status, fake.run.reason) == ("partial", "timeout")
    # Nothing was sent for a point after the deadline.
    assert fake.sent_benchmark_ids() == set(range(completed + 1))
    on_disk = _assert_rejected(fake, "partial")
    assert (on_disk["stop_reason"], on_disk["error"], on_disk["usable"]) == ("timeout", None, True)
    assert on_disk["errors"] == [f"benchmark stopped by timeout after {completed} of {len(fake.built.points)} points"]
    assert [row["point"] for row in on_disk["results"]] == fake.built.point_dicts()[:completed]
    assert on_disk["timing"]["benchmark_elapsed_seconds"] >= FAKE_TIMEOUT_SECONDS
    merged = json.loads((fake.pod / "benchmark_merged.json").read_text(encoding="utf-8"))
    assert (merged["artifact_type"], merged["status"], merged["valid"]) == ("merged", "partial", False)


def test_the_deadline_is_reached_at_exactly_the_timeout(tmp_path):
    # The FAKE jump after point 4 is the timeout itself, less what the run has taken until then.
    probe = _start(tmp_path / "probe", "decode", faults={(4, "run", 0): "deadline"})
    elapsed_then = probe.payload()["timing"]["benchmark_elapsed_seconds"] - FAKE_TIMEOUT_SECONDS

    exact = _start(tmp_path / "exact", "decode", timeout_seconds=elapsed_then)
    assert (exact.run.status, exact.sent_benchmark_ids()) == ("partial", set(range(5)))
    # One tick more and point 5 still starts; the deadline has passed before point 6.
    later = _start(tmp_path / "later", "decode", timeout_seconds=elapsed_then + 1 / TICKS_PER_SECOND)
    assert (later.run.status, later.sent_benchmark_ids()) == ("partial", set(range(6)))


def test_the_deadline_counts_from_the_start_of_the_launcher_and_the_elapsed_time_from_execute(tmp_path):
    # FAKE: the launcher started long before the run, as it does when the engine build takes its time.
    in_time = _start(tmp_path / "in-time", "decode", started_before_seconds=FAKE_TIMEOUT_SECONDS - 1.0)
    assert in_time.run.status == "complete"
    # The elapsed time of the artifact is the benchmark's own: the FAKE engine build is not in it.
    elapsed = in_time.payload()["timing"]["benchmark_elapsed_seconds"]
    assert elapsed == in_time.clock.read() - 5000.0
    assert in_time.payload()["timing"]["measured_iteration_seconds"] < elapsed < 1.0

    # With the whole timeout gone before execute, the deadline has passed before the first point.
    late = _start(tmp_path / "late", "decode", started_before_seconds=FAKE_TIMEOUT_SECONDS)
    assert (late.run.status, late.run.reason, late.sent_benchmark_ids()) == ("partial", "timeout", {0})
    assert 0.0 < late.payload()["timing"]["benchmark_elapsed_seconds"] < 1.0


def test_a_deadline_that_passes_after_the_last_point_leaves_the_run_complete(tmp_path):
    last = len(_plan_only(tmp_path, "decode").built.points)
    fake = _start(tmp_path, "decode", faults={(last, "run", 0): "deadline"})

    on_disk = _assert_accepted(fake)
    assert on_disk["stop_reason"] is None
    assert on_disk["timing"]["benchmark_elapsed_seconds"] >= FAKE_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------------------------------------------
# record_failure, write, and stops
# ---------------------------------------------------------------------------------------------------------------


def test_a_failure_recorded_before_execute_is_written_with_the_digest_of_an_empty_grid(tmp_path):
    fake = _start(tmp_path, execute=False)
    assert (fake.run.status, fake.run.reason) == (None, None)

    fake.run.record_failure("rejected: FAKE refusal before the engine was built")
    fake.run.record_failure("stopped: FAKE later failure")  # the first failure is kept

    reason = "rejected: FAKE refusal before the engine was built"
    assert (fake.run.status, fake.run.reason) == ("failed", reason)
    fake.clock.advance(3 * TICKS_PER_SECOND)  # FAKE: three seconds pass before the files are written
    on_disk = _assert_rejected(fake, "failed")
    # Nothing was executed, so no benchmark time has elapsed.
    assert on_disk["timing"] == {"benchmark_elapsed_seconds": 0.0, "measured_iteration_seconds": 0.0}
    assert (on_disk["error"], on_disk["errors"], on_disk["grid_digest"]) == (reason, [reason], grid.grid_digest([]))
    assert on_disk["coverage"] == {"expected_points": 0, "completed_points": 0, "skipped_points": 0}
    assert on_disk["config"] == {"mode": "prefill", **FAKE_CONFIG}
    assert on_disk["producer"]["timing_source"].startswith("FAKE")
    assert (on_disk["producer"]["scheduler_processes"], on_disk["producer"]["limits"]) == (0, None)
    # A run that already holds a failure sends nothing when it is executed.
    fake.execute()
    assert fake.engine.calls == [] and fake.run.reason == reason
    with pytest.raises(RuntimeError, match="execute was already called"):
        fake.execute()


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_a_failure_recorded_after_execute_fails_a_complete_run_and_keeps_its_rows(tmp_path, mode):
    fake = _start(tmp_path, mode)
    assert fake.run.status == "complete"

    fake.run.record_failure("stopped: FAKE signal after the sweep")

    assert (fake.run.status, fake.run.reason) == ("failed", "stopped: FAKE signal after the sweep")
    on_disk = _assert_rejected(fake, "failed")
    points = len(fake.built.points)
    assert on_disk["coverage"] == {"expected_points": points, "completed_points": points, "skipped_points": 0}
    assert [row["fpms"][0]["wall_time"] for row in on_disk["results"]] == [
        _expected_sample(point) for point in fake.built.points
    ]


def test_a_failure_recorded_after_a_partial_run_keeps_the_stop_reason(tmp_path):
    fake = _start(tmp_path, "decode", faults={(2, "run", 0): "deadline"})
    fake.run.record_failure("stopped: FAKE signal")

    on_disk = _assert_rejected(fake, "failed")
    assert (on_disk["stop_reason"], on_disk["error"]) == ("timeout", "stopped: FAKE signal")


def test_record_failure_refuses_what_is_not_a_reason(tmp_path):
    fake = _start(tmp_path, execute=False)
    with pytest.raises(TypeError, match="reason must be a str"):
        fake.run.record_failure(RuntimeError("FAKE"))
    with pytest.raises(ValueError, match="reason must not be empty"):
        fake.run.record_failure("")
    assert fake.run.status is None


@pytest.mark.parametrize("stop", ["system_exit", "keyboard_interrupt"])
def test_a_stop_raised_by_the_engine_passes_through_execute(tmp_path, stop):
    # SGLang turns SIGTERM into SystemExit inside an engine call once the first request has been sent.
    victim = _victim(_plan_only(tmp_path, "decode"))
    fake = _start(tmp_path, "decode", execute=False, faults={(victim.benchmark_id, "run", 0): stop})

    with pytest.raises(SystemExit if stop == "system_exit" else KeyboardInterrupt):
        fake.execute()

    # The run has no outcome until the caller records the stop.
    assert (fake.run.status, fake.run.reason) == (None, None)
    with pytest.raises(RuntimeError, match="the run has no outcome yet"):
        fake.run.payloads()
    fake.run.record_failure("stopped: FAKE SystemExit from an engine call")
    on_disk = _assert_rejected(fake, "failed")
    # The stop names no point, and the rows of the points before it are kept.
    assert (on_disk["error"], on_disk["skipped_points"]) == ("stopped: FAKE SystemExit from an engine call", [])
    assert on_disk["coverage"]["completed_points"] == victim.benchmark_id - 1
    # The elapsed time ends where execute was left, not where the artifact was built.
    elapsed = on_disk["timing"]["benchmark_elapsed_seconds"]
    fake.clock.advance(TICKS_PER_SECOND)
    assert fake.payload()["timing"]["benchmark_elapsed_seconds"] == elapsed


def test_an_unexpected_exception_is_recorded_as_an_internal_error(tmp_path, monkeypatch):
    fake = _start(tmp_path, "prefill", execute=False)
    real = artifact.build_fpm

    def fake_build_fpm(record, rank, wall_time, extra=None):
        if record["benchmark_id"] == 3:
            raise ZeroDivisionError("FAKE bug")
        return real(record, rank, wall_time, extra=extra)

    monkeypatch.setattr(artifact, "build_fpm", fake_build_fpm)
    fake.execute()

    # The text starts with the reason, so a caller can tell an internal error by it; the point that was being
    # staged is named at the end and is not listed as a failed point.
    reason = "internal_error: ZeroDivisionError: FAKE bug (while staging benchmark_id=3)"
    assert (fake.run.status, fake.run.reason) == ("failed", reason)
    assert max(fake.sent_benchmark_ids()) == 3
    on_disk = _assert_rejected(fake, "failed")
    assert (on_disk["error"], on_disk["skipped_points"]) == (reason, [])
    assert on_disk["coverage"]["completed_points"] == 2


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"warmup_iterations": -1}, "internal_error: ValueError: settings.warmup_iterations must be >= 0, got -1"),
        (
            {"warmup_iterations": True},
            "internal_error: TypeError: settings.warmup_iterations must be an int, got bool: True",
        ),
        (
            {"expected_scheduler_processes": 0},
            "internal_error: ValueError: settings.expected_scheduler_processes must be >= 1, got 0",
        ),
        ({"timeout_seconds": 0}, "internal_error: ValueError: settings.timeout_seconds must be finite and > 0, got 0"),
        (
            {"timeout_seconds": float("inf")},
            "internal_error: ValueError: settings.timeout_seconds must be finite and > 0, got inf",
        ),
        (
            {"log_wait_seconds": -1.0},
            "internal_error: ValueError: settings.log_wait_seconds must be finite and >= 0, got -1.0",
        ),
        (
            {"header_wait_seconds": "60"},
            "internal_error: TypeError: settings.header_wait_seconds must be a number of seconds, got str: '60'",
        ),
        ({"caps": None}, "internal_error: TypeError: settings.caps must be a SampleCaps record, got NoneType"),
        ({"worker_id": None}, "internal_error: TypeError: settings.worker_id must be a str, got None"),
        ({"content_seed": 0}, "internal_error: TypeError: settings.content_seed must be a str, got 0"),
    ],
)
def test_a_bad_setting_is_recorded_as_an_internal_error_before_any_request(tmp_path, overrides, reason):
    fake = _start(tmp_path, **overrides)
    assert (fake.run.status, fake.run.reason) == ("failed", reason)
    assert fake.engine.calls == []


def test_the_constructor_refuses_what_no_artifact_can_carry(tmp_path):
    fake = _start(tmp_path, execute=False)
    good = fake.settings

    def build(**changes):
        return driver.BenchmarkRun(dataclasses.replace(good, **changes), fake.recorder_dir, started_at=0.0)

    with pytest.raises(ValueError, match="settings.mode must be one of"):
        build(mode="both")
    with pytest.raises(ValueError, match="output_path must name a file"):
        build(output_path=str(fake.pod) + "/")
    with pytest.raises(ValueError, match="run_id must be non-empty"):
        build(run_id="")
    with pytest.raises(ValueError, match="run_id must be non-empty and use only"):
        build(run_id="fake run")
    with pytest.raises(TypeError, match="settings.run_id must be a str"):
        build(run_id=7)
    with pytest.raises(TypeError, match="settings must be a RunSettings record"):
        driver.BenchmarkRun({"mode": "prefill"}, fake.recorder_dir, started_at=0.0)
    with pytest.raises(TypeError, match="started_at must be a finite reading"):
        driver.BenchmarkRun(good, fake.recorder_dir, started_at=None)
    # The defaults are the real clock and the real sleep, and the two waits of the settings.
    run = build()
    assert (run._clock, run._sleep) == (driver.time.monotonic, driver.time.sleep)
    fields = {field.name: field.default for field in dataclasses.fields(driver.RunSettings)}
    assert (fields["log_wait_seconds"], fields["header_wait_seconds"]) == (30.0, 60.0)


def test_a_block_that_is_not_json_fails_at_the_dry_build_before_any_request(tmp_path):
    fake = _start(tmp_path, producer={**FAKE_PRODUCER, "handle": object()})

    assert fake.run.status == "failed"
    assert fake.run.reason.startswith("internal_error: TypeError: producer is not JSON data: ")
    assert fake.engine.calls == []
    # The failed artifact is written without the launcher's two blocks and says so.
    on_disk = _assert_rejected(fake, "failed")
    assert on_disk["config"] == {"mode": "prefill"}
    assert on_disk["producer"]["omitted_launcher_blocks"] == fake.run.reason.removeprefix("internal_error: ")
    assert "timing_source" not in on_disk["producer"]
    assert on_disk["producer"]["limits"] == fake.limits.as_dict()


@pytest.mark.parametrize(
    ("overrides", "start"),
    [
        ({"config": {**FAKE_CONFIG, 7: "a key that is not a str"}}, "internal_error: TypeError: config "),
        ({"config": {**FAKE_CONFIG, "mode": "decode"}}, "internal_error: ValueError: config may not contain 'mode'"),
        ({"config": None}, "internal_error: TypeError: settings.config must be a dict, got NoneType"),
        (
            {"producer": {"page_size": 64}},
            "internal_error: ValueError: settings.producer may not contain ['page_size']",
        ),
        ({"producer": {"name": "other"}}, "internal_error: ValueError: producer may not contain ['name']"),
        ({"producer": ["a", "list"]}, "internal_error: TypeError: settings.producer must be a dict, got list"),
        ({"producer": {"nan": float("nan")}}, "internal_error: ValueError: producer is not JSON data: "),
    ],
)
def test_a_block_the_artifact_refuses_fails_at_the_dry_build(tmp_path, overrides, start):
    fake = _start(tmp_path, **overrides)
    assert fake.run.reason.startswith(start), fake.run.reason
    assert fake.engine.calls == []
    assert "omitted_launcher_blocks" in _assert_rejected(fake, "failed")["producer"]


def test_a_fact_the_launcher_adds_to_producer_after_execute_is_written(tmp_path):
    fake = _start(tmp_path, "decode")
    # The launcher keeps the dict it handed over and records an engine shutdown error in it.
    fake.settings.producer["engine_shutdown_error"] = "FAKE: RuntimeError('children not reaped')"

    on_disk = _assert_accepted(fake)
    assert on_disk["producer"]["engine_shutdown_error"] == "FAKE: RuntimeError('children not reaped')"


def test_write_falls_back_to_a_failed_artifact_when_the_complete_one_cannot_be_written(tmp_path, monkeypatch):
    fake = _start(tmp_path, "decode")
    real = artifact.write_artifacts
    statuses = []

    def refuse_complete(output_path, rank_payloads):
        statuses.append(rank_payloads[0]["status"])
        if rank_payloads[0]["status"] == "complete":
            raise ValueError("rank 0: FAKE refusal of a complete artifact")
        return real(output_path, rank_payloads)

    monkeypatch.setattr(artifact, "write_artifacts", refuse_complete)
    assert fake.run.status == "complete"
    written = fake.run.write()

    reason = "internal_error: ValueError: rank 0: FAKE refusal of a complete artifact"
    assert statuses == ["complete", "failed"]
    assert written == {"status": "failed", "paths": [str(fake.pod / "benchmark.json")]}
    assert (fake.run.status, fake.run.reason) == ("failed", reason)
    on_disk = json.loads((fake.pod / "benchmark.json").read_text(encoding="utf-8"))
    # The failed artifact keeps every row and both blocks of the launcher.
    assert (on_disk["status"], on_disk["error"]) == ("failed", reason)
    assert len(on_disk["results"]) == len(fake.built.points)
    assert on_disk["producer"]["timing_source"].startswith("FAKE")
    assert "omitted_launcher_blocks" not in on_disk["producer"]
    assert not (fake.pod / "benchmark_merged.json").exists()


def test_write_leaves_the_launcher_blocks_out_when_they_cannot_be_written(tmp_path):
    fake = _start(tmp_path, "decode")
    assert fake.run.status == "complete"
    fake.settings.producer["engine_shutdown_error"] = RuntimeError("FAKE: not JSON")

    on_disk = _assert_rejected(fake, "failed")
    text = fake.run.reason.removeprefix("internal_error: ")
    assert text.startswith("TypeError: producer is not JSON data: ")
    assert (on_disk["config"], on_disk["producer"]["omitted_launcher_blocks"]) == ({"mode": "decode"}, text)
    assert len(on_disk["results"]) == len(fake.built.points)


def test_write_raises_when_not_even_the_reduced_artifact_can_be_built(tmp_path, monkeypatch):
    fake = _start(tmp_path, "decode")

    def refuse(_output_path, _rank_payloads):
        raise ValueError("FAKE refusal")

    monkeypatch.setattr(artifact, "write_artifacts", refuse)

    with pytest.raises(ValueError, match="FAKE refusal"):
        fake.run.write()
    assert fake.run.reason == "internal_error: ValueError: FAKE refusal"
    assert not (fake.pod / "benchmark.json").exists()
    with pytest.raises(RuntimeError, match="write was already called for this run and did not finish"):
        fake.run.write()


def test_write_writes_once(tmp_path, monkeypatch):
    fake = _start(tmp_path, "prefill")
    first = fake.run.write()
    path = fake.pod / "benchmark.json"
    before = (path.stat().st_ino, path.read_bytes())

    # A failure recorded afterwards, as a late signal would, does not reach the files.
    fake.run.record_failure("stopped: FAKE signal after the write")
    calls = []
    monkeypatch.setattr(artifact, "write_artifacts", lambda *args: calls.append(args))
    second = fake.run.write()

    assert second == first == {"status": "complete", "paths": [str(path), str(fake.pod / "benchmark_merged.json")]}
    assert calls == [] and (path.stat().st_ino, path.read_bytes()) == before
    # The run keeps the outcome that was written.
    assert (fake.run.status, fake.run.reason, fake.payload()["status"]) == ("complete", None, "complete")
    second["paths"].clear()  # the returned dict is the caller's
    assert fake.run.write() == first


def test_write_without_an_outcome_writes_a_failed_artifact(tmp_path):
    fake = _start(tmp_path, execute=False)
    written = fake.run.write()

    reason = "internal_error: the result files were written before the run had an outcome"
    assert written == {"status": "failed", "paths": [str(fake.pod / "benchmark.json")]}
    assert fake.run.reason == reason


def test_write_raises_what_is_not_an_artifact_error(tmp_path):
    fake = _start(tmp_path, "decode", output_path=str(tmp_path / "missing-directory" / "benchmark.json"))
    assert fake.run.status == "complete"
    with pytest.raises(FileNotFoundError):
        fake.run.write()
    assert fake.run.status == "complete"


# ---------------------------------------------------------------------------------------------------------------
# ordinary_token_range
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "excluded", "expected"),
    [
        (10, [], (0, 10)),
        (10, [0], (1, 10)),
        (10, [9], (0, 9)),
        (10, [5], (0, 5)),  # runs of 5 and of 4
        (10, [4], (5, 10)),  # runs of 4 and of 5
        (11, [5], (0, 5)),  # two runs of 5: the lower one
        (10, [0, 1, 2, 7], (3, 7)),
        (10, [7, 2, 0, 1, 2], (3, 7)),  # any order, repeats allowed
        (10, [10, 11, 500, -1], (0, 10)),  # ids outside [0, size) exclude nothing
        (1, [], (0, 1)),
        (151_936, [151_643, 151_644, 151_645, *range(151_646, 151_670)], (0, 151_643)),
        (32_000, [0, 1, 2, 31_999], (3, 31_999)),
    ],
)
def test_ordinary_token_range_is_the_longest_run_without_an_excluded_id(size, excluded, expected):
    low, high = driver.ordinary_token_range(size, excluded)
    assert (low, high) == expected
    # Checked against the definition: no excluded id inside, and no longer run anywhere.
    taken = set(excluded)
    assert not taken & set(range(low, high))
    free = [token not in taken for token in range(size)]
    longest = max(len(list(group)) for value, group in itertools.groupby(free) if value)
    assert high - low == longest
    assert driver.ordinary_token_range(size=size, excluded=iter(excluded)) == expected


def test_ordinary_token_range_refuses_bad_input():
    with pytest.raises(ValueError, match=r"every token id in \[0, 3\) is excluded"):
        driver.ordinary_token_range(3, [0, 1, 2])
    with pytest.raises(ValueError, match="size must be >= 1, got 0"):
        driver.ordinary_token_range(0, [])
    with pytest.raises(TypeError, match="size must be an int, got float"):
        driver.ordinary_token_range(10.0, [])
    with pytest.raises(TypeError, match="size must be an int, got bool"):
        driver.ordinary_token_range(True, [])
    with pytest.raises(TypeError, match="an excluded token id must be an int, got str"):
        driver.ordinary_token_range(10, ["3"])
    with pytest.raises(TypeError, match="an excluded token id must be an int, got bool"):
        driver.ordinary_token_range(10, [True])


# ---------------------------------------------------------------------------------------------------------------
# RecorderLogs
# ---------------------------------------------------------------------------------------------------------------


def _line(record: dict) -> bytes:
    return json.dumps(record, separators=(",", ":")).encode() + b"\n"


def _batch_line(seq: int, mode: str = "EXTEND", rids=("a",)) -> dict:
    return {"type": "batch", "seq": seq, "mode": mode, "rids": list(rids)}


def test_recorder_logs_read_only_what_was_appended_and_keep_a_half_written_line_back(tmp_path):
    logs = driver.RecorderLogs(tmp_path / "not-yet-there")
    logs.poll()  # a directory that does not exist holds no file
    assert (logs.headers, logs.faults, logs.batches("sched-1.jsonl")) == ({}, [], [])

    logs = driver.RecorderLogs(tmp_path)
    path = tmp_path / "sched-1.jsonl"
    header = {"type": "header", "format": 1}
    first = _line(_batch_line(1))
    path.write_bytes(_line(header) + first[:10])
    (tmp_path / "other.jsonl").write_bytes(b"not a recorder file\n")
    (tmp_path / "sched-2.json").write_bytes(b"not a recorder file\n")
    logs.poll()
    assert logs.headers == {"sched-1.jsonl": header}
    assert logs.batches("sched-1.jsonl") == []

    with path.open("ab") as handle:
        handle.write(first[10:] + _line(_batch_line(2, "DECODE", ["a", "b"]))[:-1])
    logs.poll()
    assert [line["seq"] for line in logs.batches("sched-1.jsonl")] == [1]
    with path.open("ab") as handle:
        handle.write(b"\n")
    logs.poll()
    logs.poll()  # nothing new: nothing is read twice
    assert logs.batches("sched-1.jsonl") == [_batch_line(1), _batch_line(2, "DECODE", ["a", "b"])]
    assert logs.faults == []

    # forget drops the lines read so far and keeps counting: the next line must have seq 3.
    logs.forget()
    assert logs.batches("sched-1.jsonl") == [] and logs.headers == {"sched-1.jsonl": header}
    with path.open("ab") as handle:
        handle.write(_line(_batch_line(3)) + _line(_batch_line(5)))
    logs.poll()
    assert logs.batches("sched-1.jsonl") == [_batch_line(3)]
    assert logs.faults == [
        {"file": "sched-1.jsonl", "where": "reader", "error": "line 5: batch line with seq 5, expected 4"}
    ]


def test_recorder_logs_ignore_idle_lines_without_requests_and_keep_the_others(tmp_path):
    logs = driver.RecorderLogs(str(tmp_path))
    lines = [
        {"type": "header", "format": 1},
        _batch_line(1, "IDLE", []),
        _batch_line(2, "IDLE", ["a"]),
        _batch_line(3, "EXTEND", []),
        _batch_line(4),
    ]
    (tmp_path / "sched-9.jsonl").write_bytes(b"".join(_line(line) for line in lines))
    logs.poll()
    # The idle line without requests is left out and still counts for seq.
    assert [line["seq"] for line in logs.batches("sched-9.jsonl")] == [2, 3, 4]
    assert logs.faults == []


def test_recorder_logs_list_fault_lines_and_everything_that_is_not_format_1(tmp_path):
    logs = driver.RecorderLogs(tmp_path)
    good = tmp_path / "sched-1.jsonl"
    good.write_bytes(
        _line({"type": "header", "format": 1})
        + _line(_batch_line(1))
        + _line({"type": "fault", "where": "run_batch", "error": "FAKE"})
        + b"not json\n"
        + b"\xff\xfe\n"
        + b"[1, 2]\n"
        + b"\n"
        + _line({"type": "summary"})
        + _line({"type": "header", "format": 1})
        + _line(_batch_line(3))  # seq 2 was lost
        + _line(_batch_line(4, mode=None))
        + _line({"type": "batch", "seq": 5, "mode": "EXTEND", "rids": ["a", 7]})
        + _line({"type": "batch", "seq": "6", "mode": "EXTEND", "rids": []})
        + _line(_batch_line(7))
    )
    (tmp_path / "sched-2.jsonl").write_bytes(
        _line(_batch_line(1)) + _line({"type": "fault", "where": "header", "error": "FAKE: no header"})
    )
    logs.poll()

    assert list(logs.headers) == ["sched-1.jsonl"]
    # A line that is not format 1 is not kept.
    assert [line["seq"] for line in logs.batches("sched-1.jsonl")] == [1, 7]
    assert logs.batches("sched-2.jsonl") == []
    assert [(fault["file"], fault["where"]) for fault in logs.faults] == [
        ("sched-1.jsonl", "run_batch"),
        *[("sched-1.jsonl", "reader")] * 10,
        ("sched-2.jsonl", "reader"),
        ("sched-2.jsonl", "header"),
    ]
    errors = [fault["error"] for fault in logs.faults]
    assert errors[0] == "FAKE"
    assert errors[1].startswith("line 4: not a JSON line: ")
    assert errors[2].startswith("line 5: not a JSON line: ")
    assert errors[3:] == [
        "line 6: not a JSON object: list",
        "line 7: not a JSON line: Expecting value: line 1 column 1 (char 0)",
        "line 8: unknown line type 'summary'",
        "line 9: a header that is not the first line of its file",
        "line 10: batch line with seq 3, expected 2",
        "line 11: batch line without a mode and a list of request ids",
        "line 12: batch line without a mode and a list of request ids",
        "line 13: batch line with seq '6', expected 6",
        "line 1: a batch line in a file without a header",
        "FAKE: no header",
    ]


def test_a_recorder_line_that_is_not_format_1_fails_the_run_before_the_first_request(tmp_path):
    fake = _start(tmp_path, "decode", execute=False)
    with (fake.recorder_dir / LEADER_LOG).open("a", encoding="utf-8") as handle:
        handle.write("FAKE: not a recorder line\n")
    fake.execute()

    assert fake.run.reason.startswith("recorder_fault: sched-4100.jsonl: reader: line 2: not a JSON line: ")
    assert fake.engine.calls == []
    _assert_rejected(fake, "failed")


# ---------------------------------------------------------------------------------------------------------------
# The module itself
# ---------------------------------------------------------------------------------------------------------------


def test_the_driver_imports_only_the_standard_library_and_its_three_siblings_and_reads_no_environment():
    source = (RUNTIME / "sglbench_driver.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imported.add(node.module.split(".")[0])
    siblings = {"sglbench_grid", "sglbench_content", "sglbench_artifact"}
    assert imported - siblings <= set(sys.stdlib_module_names)
    assert imported & {name for name in imported if name.startswith("sglbench_")} == siblings
    # No environment variable is read: every value is an argument.
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not attributes & {"environ", "getenv", "environb", "putenv"}
    assert source.isascii()
    assert "No upstream function or block of source is copied." in source


_POD_SCRIPT = """
import sys

sys.path.append(sys.argv[1])
import sglbench_driver as driver

forbidden = {"sglang", "torch", "numpy", "dynamo", "aisim", "aisim_core", "collector"}
loaded = sorted(name for name in sys.modules if name.split(".")[0] in forbidden)
assert not loaded, loaded
third_party = sorted(
    name for name, module in sys.modules.items() if "site-packages" in (getattr(module, "__file__", None) or "")
)
assert not third_party, third_party
assert driver.ordinary_token_range(10, [5]) == (0, 5)
print(sorted(name for name in sys.modules if name.startswith("sglbench_")))
"""


def test_the_driver_imports_in_a_bare_interpreter_without_sglang():
    done = subprocess.run(
        [sys.executable, "-I", "-S", "-c", _POD_SCRIPT, str(RUNTIME)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "['sglbench_artifact', 'sglbench_content', 'sglbench_driver', 'sglbench_grid']"


def _module_constants(path: Path) -> dict:
    """The module-level constants of a file that are plain literals, read without importing it."""
    constants = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return constants


def test_the_constants_the_driver_repeats_are_the_ones_of_the_recorder_and_of_compat():
    # Typed here once more, as the recorder log format gives them.
    assert (driver.RECORD_FORMAT, driver.RECORDER_CLASS_NAME) == (1, "SglbenchRecorderScheduler")
    assert (driver.LOG_FILE_PREFIX, driver.LOG_FILE_SUFFIX) == ("sched-", ".jsonl")
    assert driver.TESTED_SGLANG_VERSIONS == ("0.5.22", "0.5.23")
    assert driver.WARMUP_PROMPT_TOKENS == 256
    # The two modules that own the values import SGLang inside functions and are not imported here; their
    # literals are read from the files beside this one.
    constants = _module_constants(RUNTIME / "sglbench_recorder.py")
    assert (constants["RECORD_FORMAT"], constants["RECORDER_CLASS_NAME"]) == (
        driver.RECORD_FORMAT,
        driver.RECORDER_CLASS_NAME,
    )
    assert (constants["LOG_FILE_PREFIX"], constants["LOG_FILE_SUFFIX"]) == (
        driver.LOG_FILE_PREFIX,
        driver.LOG_FILE_SUFFIX,
    )
    assert constants["_LIMIT_NAMES"] == driver._LIMIT_NAMES
    assert _module_constants(RUNTIME / "sglbench_compat.py")["TESTED_VERSIONS"] == driver.TESTED_SGLANG_VERSIONS


def test_the_public_names_have_the_agreed_shape(tmp_path):
    # The launcher is written against these names.
    assert [field.name for field in dataclasses.fields(driver.RunSettings)] == [
        "mode",
        "caps",
        "manifest",
        "warmup_iterations",
        "timeout_seconds",
        "output_path",
        "run_id",
        "content_seed",
        "worker_id",
        "expected_scheduler_processes",
        "config",
        "producer",
        "prefill_max_requests",
        "max_queued_requests",
        "log_wait_seconds",
        "header_wait_seconds",
    ]
    fake = _start(tmp_path, execute=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        fake.settings.mode = "decode"
    # A run with other settings is built from a copy, with the same run id and start.
    changed = dataclasses.replace(fake.settings, expected_scheduler_processes=2, prefill_max_requests=64)
    assert (changed.run_id, changed.expected_scheduler_processes) == (FAKE_RUN_ID, 2)

    def kinds(function) -> list[tuple[str, str]]:
        return [(name, parameter.kind.name) for name, parameter in inspect.signature(function).parameters.items()]

    assert kinds(driver.BenchmarkRun.__init__) == [
        ("self", "POSITIONAL_OR_KEYWORD"),
        ("settings", "POSITIONAL_OR_KEYWORD"),
        ("recorder_dir", "POSITIONAL_OR_KEYWORD"),
        ("started_at", "KEYWORD_ONLY"),
        ("clock", "KEYWORD_ONLY"),
        ("sleep", "KEYWORD_ONLY"),
    ]
    assert kinds(driver.BenchmarkRun.execute) == [
        ("self", "POSITIONAL_OR_KEYWORD"),
        ("engine", "POSITIONAL_OR_KEYWORD"),
        ("token_low", "KEYWORD_ONLY"),
        ("token_high", "KEYWORD_ONLY"),
    ]
    assert kinds(driver.BenchmarkRun.record_failure) == [
        ("self", "POSITIONAL_OR_KEYWORD"),
        ("reason", "POSITIONAL_OR_KEYWORD"),
    ]
    assert (
        kinds(driver.BenchmarkRun.payloads) == kinds(driver.BenchmarkRun.write) == [("self", "POSITIONAL_OR_KEYWORD")]
    )
    assert kinds(driver.ordinary_token_range) == [
        ("size", "POSITIONAL_OR_KEYWORD"),
        ("excluded", "POSITIONAL_OR_KEYWORD"),
    ]
    assert isinstance(driver.BenchmarkRun.status, property) and isinstance(driver.BenchmarkRun.reason, property)
    assert isinstance(driver.RecorderLogs(tmp_path), driver.RecorderLogs)
    assert (fake.run.settings, fake.run.recorder_dir) == (fake.settings, str(fake.recorder_dir))


def test_the_reason_strings_and_the_staging_constants_are_the_shared_ones():
    # The five strings of Dynamo's vLLM self-benchmark and this module's own, as the artifact carries them.
    assert (
        driver.REASON_BATCH_SIZE,
        driver.REASON_CACHE,
        driver.REASON_PREFILL_TOKENS,
        driver.REASON_DECODE_CONTEXT,
        driver.REASON_NO_BATCHES,
    ) == (
        "measured_batch_size_mismatch",
        "real_seed_cache_validation_failed",
        "measured_prefill_tokens_mismatch",
        "measured_decode_context_mismatch",
        "no_fpm_before_deadline",
    )
    assert (
        driver.REASON_NOT_COMPLETED,
        driver.REASON_ENGINE_CALL,
        driver.REASON_RECORDER_FAULT,
        driver.REASON_HEADER,
        driver.REASON_SELF_TEST,
        driver.REASON_REJECTED,
        driver.REASON_TOKEN_RANGE,
        driver.REASON_INTERNAL,
    ) == (
        "request_not_completed",
        "engine_call_failed",
        "recorder_fault",
        "recorder_header_invalid",
        "self_test_failed",
        "rejected",
        "token_range_too_small",
        "internal_error",
    )
    # A decode run has seven output tokens: one from an extend batch and six from six decode batches, of which
    # the middle kept one reads the prompt and three generated tokens.
    assert grid.DECODE_OUTPUT_TOKENS - 1 == artifact.DECODE_RAW_STEPS == 6
    assert artifact.DECODE_KEPT_STEP_INDICES[len(artifact.DECODE_KEPT_STEP_INDICES) // 2] == grid.DECODE_PROMPT_LEAD
    assert {"seed", "warm", "meas", "run", "free"} == set(content.STAGES)
    assert (driver.RANK, driver.DP_SIZE, driver.RUN_TAG_CHARACTERS) == (0, 1, 8)


# ---------------------------------------------------------------------------------------------------------------
# Additions proposed by the review
# ---------------------------------------------------------------------------------------------------------------


def test_the_leader_is_found_by_its_header_and_not_by_the_order_of_the_files(tmp_path):
    # FAKE: the process with the higher pid is the leader. Its batches are seven ticks slower than those of the
    # other process, so a driver that took its times from the first file would show it in every sample.
    def flip(index: int, header: dict) -> dict:
        header["leader"] = index == 1
        header["ranks"]["attn_tp_rank"] = 1 - index
        return header

    fake = _start(tmp_path, "prefill", processes=2, edit_header=flip)

    assert (fake.run.status, fake.run.reason) == ("complete", None)
    assert [row["fpms"][0]["wall_time"] for row in fake.payload()["results"]] == [
        _expected_sample(point) + 7 / TICKS_PER_SECOND for point in fake.built.points
    ]


def test_a_fault_line_in_a_follower_log_fails_the_point(tmp_path):
    victim = _victim(_plan_only(tmp_path, "prefill", processes=2))
    fake = _start(tmp_path, "prefill", processes=2, faults={(victim.benchmark_id, "meas", 0): "follower_fault_line"})
    _assert_point_failed(fake, victim, "recorder_fault: sched-4101.jsonl: process_batch_result: FAKE recorder fault")


def test_the_header_wait_and_the_log_wait_are_two_settings(tmp_path):
    missing = _start(tmp_path / "header", headers=False, header_wait_seconds=3.0, log_wait_seconds=1.0)
    assert missing.run.reason.endswith(" after 3.0 s")
    assert missing.clock.sleeps == 3.0 / driver.POLL_SECONDS

    victim = _victim(_plan_only(tmp_path, "prefill"))
    silent = _start(
        tmp_path / "log",
        "prefill",
        faults={(victim.benchmark_id, "meas", 0): "silent"},
        header_wait_seconds=3.0,
        log_wait_seconds=1.0,
    )
    assert silent.run.reason.endswith(" after 1.0 s")
    assert silent.clock.sleeps == 1.0 / driver.POLL_SECONDS


def test_a_fault_line_is_looked_for_before_the_deadline(tmp_path, monkeypatch):
    fake = _start(tmp_path, "prefill", execute=False)
    real = artifact.build_measurement

    def build_then_fault_and_deadline(record, rank_fpms):
        # FAKE: after point 3 the recorder faults and the soft deadline passes.
        if record["benchmark_id"] == 3:
            fake.engine.logs[0].write({"type": "fault", "where": "run_batch", "error": "FAKE late fault"})
            fake.clock.advance(round(FAKE_TIMEOUT_SECONDS * TICKS_PER_SECOND))
        return real(record, rank_fpms)

    monkeypatch.setattr(artifact, "build_measurement", build_then_fault_and_deadline)
    fake.execute()

    assert (fake.run.status, fake.run.reason) == (
        "failed",
        "recorder_fault: sched-4100.jsonl: run_batch: FAKE late fault",
    )


@pytest.mark.parametrize(
    ("mode", "stage", "reason"),
    [
        ("decode", "run", "measured_decode_context_mismatch"),
        ("prefill", "meas", "measured_prefill_tokens_mismatch"),
        ("prefill", "warm", "measured_prefill_tokens_mismatch"),
    ],
)
def test_a_batch_that_is_written_after_its_call_was_checked_fails_the_run(tmp_path, mode, stage, reason):
    # FAKE: the scheduler ran one decode batch more for the call, and its line reaches the log only when the
    # next call starts, as the line of a batch does that overlap scheduling launched before the last output.
    victim = _victim(_plan_only(tmp_path, mode))
    fake = _start(tmp_path, mode, faults={(victim.benchmark_id, stage, 0): "late_decode"})

    assert fake.run.status == "failed"
    assert f"{reason}: a DECODE batch (seq " in fake.run.reason
    assert fake.run.reason.endswith(") holds requests of a call that had already been checked")
    # At most one more point was staged before the late line was found.
    assert max(fake.sent_benchmark_ids()) <= victim.benchmark_id + 1
    _assert_rejected(fake, "failed")


def test_a_seventh_decode_batch_of_the_self_test_that_is_written_late_fails_the_run(tmp_path):
    fake = _start(tmp_path, "decode", faults={(0, "free", FREE_SEVEN_TOKEN_RUN): "late_decode"})
    assert fake.run.status == "failed"
    assert "a DECODE batch (seq " in fake.run.reason
    assert fake.sent_benchmark_ids() <= {0, 1}


def test_a_recorder_file_that_appears_after_the_header_wait_fails_the_run(tmp_path):
    fake = _start(tmp_path, "prefill", execute=False)
    real = fake.engine.generate

    def generate_after_a_new_process(prompts, rids, max_new_tokens):
        # FAKE: one more scheduler process writes its header once the run has started.
        path = fake.recorder_dir / "sched-9999.jsonl"
        if not path.exists():
            path.write_text(json.dumps(_fake_header(9999, 1, fake.limits, 1)) + "\n", encoding="utf-8")
        return real(prompts, rids, max_new_tokens)

    fake.engine.generate = generate_after_a_new_process
    fake.execute()

    assert fake.run.reason == (
        "self_test_failed: batch_shape: recorder_header_invalid: 2 recorder headers after the run started, "
        "expected 1: ['sched-4100.jsonl', 'sched-9999.jsonl']"
    )
    assert fake.sent_benchmark_ids() == {0}
    _assert_rejected(fake, "failed")


def test_the_request_caps_also_cover_the_pairs_of_the_self_test(tmp_path):
    manifest = grid.parse_points_manifest(
        json.dumps(
            {
                "schema_version": 1,
                "prefill": [{"batch_size": 1, "total_prefill_tokens": 8, "total_kv_read_tokens": 16}],
                "decode": [],
            }
        )
    )
    fake = _start(tmp_path, "prefill", manifest=manifest, prefill_max_requests=1)
    assert fake.run.reason == "rejected: prefill_max_requests=1 is below the largest batch 2"
    assert fake.engine.calls == []
    queued = _start(tmp_path / "queued", "prefill", manifest=manifest, max_queued_requests=1)
    assert queued.run.reason == "rejected: max_queued_requests=1 is below the largest batch 2"
    assert _start(tmp_path / "two", "prefill", manifest=manifest, prefill_max_requests=2).run.status == "complete"
