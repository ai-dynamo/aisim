# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# The fake engine and the batch-sequence checks follow SGLang's scheduler at v0.5.22 and v0.5.23; see
# THIRD_PARTY_NOTICES.md.
# Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d
# Source: https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90
# The run lifecycle, the point procedures and six failure reason strings are re-implemented from ai-dynamo/dynamo
# (Apache-2.0, https://github.com/ai-dynamo/dynamo), components/src/dynamo/vllm/instrumented_scheduler.py at
# b20fe6fe94021b854597965c7305931175e8f482. See THIRD_PARTY_NOTICES.md.

"""CPU-only test of ``runtime/sglbench_driver.py`` against a FAKE engine and AISim's two real checkers.

No SGLang and no GPU are involved. Everything that stands in for an engine is FAKE and is named so:

- ``_FakeEngine`` answers ``generate`` with cached-token counts from ``PageCacheModel`` (the model of SGLang's
  prefix cache written in ``test_fpm_sglbench_content.py``) and appends the batch lines of each call to FAKE
  recorder files. It writes those files itself, to record format 3; ``sglbench_recorder.py`` is not used here.
  Its batches follow a FAKE scheduler: extend batches filled up to the chunk budget, then one decode batch for
  every further output token.
- With ``dp_size`` above 1 the fake engine stands in for an engine with DP attention: one FAKE scheduler
  process, one prefix-cache model and one log per attention-DP rank. A request that names no rank is dealt to
  the ranks in turn by a FAKE counter that never resets, as SGLang's DP controller deals them, and the response
  says where it ran (``dp_rank``). Every rank forms the batches of its own requests, and the ranks then run
  in FAKE lockstep: in each step every rank writes one line, an ``IDLE`` line without requests when it has
  nothing left, all with the same ``forward_iter`` and the same ``global_num_tokens``. A rank that decodes in a
  step in which another rank extends, and no rank idles, writes its line as SGLang writes a decode batch it
  has turned into an extend batch: mode ``EXTEND`` and one computed token per request
  (``_FakeEngine._fake_decode_to_extend``). SGLang does that only when the prefill graphs of every rank can
  replay the step; a test can make the fake an engine whose graphs cannot, which then leaves the decode batch
  as it is. No SGLang input blocker is modelled: the fake releases every call on all ranks in one step unless
  a fault says otherwise.
- Every time is FAKE. ``_FakeClock`` counts whole ticks of 2**-20 s and moves only when the fake engine runs a
  batch or the driver sleeps. The durations come from ``FAKE_*`` constants. They were typed in to exercise the
  arithmetic and were never measured. The artifacts say so in ``producer``, among the facts of the launcher;
  the few tests that make the driver leave those facts out hold no other claim about the times either.
- A fault is something the fake engine does on purpose to one call: a wrong batch line, a wrong response, a
  dropped cache.
- The text of a text run is FAKE. ``_fake_conversations`` types in lists of token ids in the place of tokenized
  conversations, and ``FAKE_DATASET`` is a record of a dataset file that does not exist. No tokenizer and no
  dataset are involved.

The driver, the grid, the content plan and the artifact module are the real ones, loaded flat from the runtime
directory as the pod loads them. The two checkers are used as shipped, through the helpers of
``test_fpm_sglbench_artifact.py``: the validator (``validate_native_collection``) and the gate (the Python text
inside ``check_result_files()`` of ``runtime/fpm_exec.sh``).
"""

from __future__ import annotations

import collections
import dataclasses
import itertools
import json
import sys
from pathlib import Path

import pytest

from .test_fpm_runner import _cell
from .test_fpm_sglbench_artifact import GATE_ACCEPT, GATE_INVALID, POD_SGLANG_VERSION, _gate, _pod, _validate
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
    # The release the launcher checked. The validator compares it with the provenance file of the pod.
    "sglang_version": POD_SGLANG_VERSION,
    # What SGLang resolves for the command of an fp8 cell, which the stand-in cell of these tests is. The
    # validator names it when it refuses a stored dtype.
    "kv_cache_dtype": "fp8_e4m3",
}
# The dtype of the KV pool every FAKE scheduler process reports in its header, as the recorder writes it: the
# text of the torch dtype SGLang stores for ``fp8_e4m3``. The validator holds it against the KV-cache label of
# the cell, and the stand-in cell of these tests is labelled fp8.
FAKE_KV_STORED_DTYPE = "torch.float8_e4m3fn"

# Few values on every axis: most tests need a run, not a dense grid.
SMALL_CAPS = grid.SampleCaps(
    prefill_max_new_token_samples=3,
    prefill_max_kv_read_token_samples=3,
    decode_max_kv_read_token_samples=3,
    decode_max_batch_size_samples=3,
    prefix_max_batch_size_samples=2,
)

# FAKE text of a text run. Every id of a FAKE conversation lies in a narrow band of the token range, so that a
# token that came from the pool can be told from an allocated first token and from a seeded id.
FAKE_TEXT_LOW, FAKE_TEXT_HIGH = 20_000, 20_050
# FAKE record of the dataset file, with the six keys the launcher writes. No such file exists.
FAKE_DATASET = {
    "path": "/fake/datasets/FAKE_conversations.json",
    "sha256": "0123456789abcdef" * 4,
    "collection_pool": "conversation_sha256_even",
    "conversations": 67,
    "url": None,
    "downloaded": False,
}
# The ``kvwarm`` block of a decode or agg artifact of a run with seeded ids, and what a text run adds to it.
SEEDED_KVWARM = {"enabled": True, "warm_eligible": True, "skip_reason": None, "method": "radix_cache_seed"}
FAKE_KVWARM_DATASET = {key: FAKE_DATASET[key] for key in ("path", "sha256", "collection_pool", "conversations")}

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


def _fake_conversations(count: int = 64) -> list[list[int]]:
    """FAKE tokenized conversations, repetitive on purpose: all start with the same five ids of a 50-id band.

    Three ids lie outside the token range of the run and one conversation holds nothing else, as the ids of a
    special token written inside a text would. A pool over the token range drops them and counts them; a
    wider pool, as the launcher builds it over the ids the model counts, keeps them.
    """
    band = FAKE_TEXT_HIGH - FAKE_TEXT_LOW
    rows = []
    for index in range(count):
        length = 24 + (index * 7) % 40
        row = [FAKE_TEXT_LOW + step for step in range(5)]
        row += [FAKE_TEXT_LOW + (index * 13 + step * step) % band for step in range(length - 5)]
        rows.append(row)
    rows[3] = [*rows[3][:9], 5, *rows[3][9:], TOKEN_HIGH]
    rows.insert(7, [TOKEN_HIGH + 7])
    return rows


def _fake_pool(conversations: list[list[int]] | None = None, tokens: tuple[int, int] = (TOKEN_LOW, TOKEN_HIGH)):
    """A real text pool over FAKE conversations."""
    return content.TextPool(_fake_conversations() if conversations is None else conversations, *tokens)


def _is_fake_text(tokens: list[int]) -> bool:
    return all(FAKE_TEXT_LOW <= token < FAKE_TEXT_HIGH for token in tokens)


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


def _fake_header(
    pid: int, tp_rank: int, limits: grid.Limits, processes: int, *, dp_size: int = 1, dp_rank: int = 0
) -> dict:
    """A FAKE recorder header of format 3, typed here key by key.

    Without DP attention the process with TP rank 0 is the leader. With ``dp_size`` above 1 the header is that
    of a process of attention-DP rank ``dp_rank`` in an engine with SGLang's input blocker.
    """
    record = limits.as_dict()
    return {
        "type": "header",
        "format": 3,
        "pid": pid,
        "class": "SglbenchRecorderScheduler",
        "mro": ["SglbenchRecorderScheduler", "Scheduler", "object"],
        "sglang_version": "0.5.23",
        "leader": tp_rank == 0,
        "ranks": {
            "tp_size": processes,
            "pp_size": 1,
            "attn_dp_size": dp_size,
            "attn_dp_enabled": dp_size > 1,
            "attn_tp_rank": tp_rank,
            "attn_cp_rank": 0,
            "attn_dp_rank": dp_rank,
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
        "kv_cache_stored_dtype": FAKE_KV_STORED_DTYPE,
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
            "input_blocker": dp_size > 1,
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
                # The count of the lines of this FAKE process, idle ones included, unless a fault spoils it.
                "forward_iter": batch.get("forward_iter", self.forward_iter),
                "mode": batch["mode"],
                "launch_ts": batch.get("launch_ts", launch / TICKS_PER_SECOND),
                "run_return_ts": (launch + 1) / TICKS_PER_SECOND,
                "out_ts": batch.get("out_ts", out / TICKS_PER_SECOND),
                "copy_done": batch.get("copy_done", True),
                "rids": batch["rids"],
                "extend_num_tokens": batch["extend_num_tokens"] if extend else None,
                "prefix_lens": batch["prefix_lens"] if extend else None,
                "extend_lens": batch["extend_lens"] if extend else None,
                # Null without DP attention, as on a batch SGLang wrote no counts on.
                "global_num_tokens": batch.get("global_num_tokens"),
            }
        )


@dataclasses.dataclass
class _FakeCall:
    """One call the driver made to the fake engine."""

    prompts: list[list[int]]
    rids: list[str]
    max_new_tokens: int
    extra: tuple  # positional and keyword arguments beyond the three of the call; the driver sends none
    modes: list[str]  # the modes of the FAKE batches rank 0 wrote for the call, faults included
    # Under DP attention: the rank each request ran on, and the modes of the lines each rank wrote for the call.
    ranks: list[int] = dataclasses.field(default_factory=list)
    rank_modes: list[list[str]] = dataclasses.field(default_factory=list)

    @property
    def key(self) -> tuple[int, str, int]:
        parts = content.parse_request_id(self.rids[0])
        return parts["benchmark_id"], parts["stage"], parts["attempt"]


def _extend_batch(rids=(), prefix_lens=(), extend_lens=()) -> dict:
    return {"mode": "EXTEND", "rids": list(rids), "prefix_lens": list(prefix_lens), "extend_lens": list(extend_lens)}


def _idle_batch() -> dict:
    """A FAKE idle batch: what a rank with nothing to run writes for a step in which another rank runs."""
    return {"mode": "IDLE", "rids": []}


def _fake_step_tokens(batch: dict) -> int:
    """What SGLang counts for one rank in one step: computed tokens, decoding requests, or 0 for an idle rank."""
    if batch["mode"] == "EXTEND":
        return batch["extend_num_tokens"]
    return len(batch["rids"]) if batch["mode"] == "DECODE" else 0


# The key of a FAKE batch that marks a decode step the FAKE engine ran as an extend batch. It is no field of a
# recorder line.
FAKE_DECODE_STEP = "fake_decode_step"


def _is_fake_decode_step(batch: dict) -> bool:
    """Whether a FAKE batch is a decode step of its requests: a decode batch, or one that was run as an extend."""
    return batch["mode"] == "DECODE" or batch.get(FAKE_DECODE_STEP, False)


class _FakeEngine:
    """Stands in for an SGLang engine and its scheduler processes. Everything it reports is FAKE.

    ``faults`` maps ``(benchmark_id, stage, attempt)`` of a call to the name of what goes wrong in it. Under DP
    attention a fault that is about one rank can name it, ``"split_extend@1"``; without a name it is rank 0.
    ``evictions`` maps a benchmark id to a number of attempts: the cache is dropped just before the measured
    call (``meas`` or ``run``) of that many attempts of the point, on every rank or on ``eviction_ranks``.

    ``dp_size`` above 1 makes it an engine with DP attention; the module docstring says what that fakes.
    ``prefill_graphs_replay`` says whether the FAKE prefill graphs of every rank can replay any step, which
    is what lets SGLang turn a decode batch into an extend batch (``_fake_decode_to_extend``).
    """

    def __init__(
        self,
        recorder_dir: Path,
        limits: grid.Limits,
        clock: _FakeClock,
        *,
        processes: int = 1,
        dp_size: int = 1,
        faults: dict | None = None,
        evictions: dict | None = None,
        eviction_ranks: tuple[int, ...] | None = None,
        edit_header=None,
        headers: bool = True,
        prefill_graphs_replay: bool = True,
    ) -> None:
        assert dp_size == 1 or processes == dp_size, "FAKE DP attention: one scheduler process per rank"
        self.limits = limits
        self.clock = clock
        self.processes = processes
        self.dp_size = dp_size
        self.prefill_graphs_replay = prefill_graphs_replay
        self.faults = dict(faults or {})
        self.evictions = dict(evictions or {})
        self.eviction_ranks = eviction_ranks
        self.edit_header = edit_header
        # One FAKE prefix cache per attention-DP rank.
        self.caches = [PageCacheModel(limits.page_size) for _rank in range(dp_size)]
        # FAKE of the counter of SGLang's DP controller: the rank the next request without a rank goes to.
        self.round_robin = 0
        self.calls: list[_FakeCall] = []
        self.late: list = []  # writes of a late follower, done after a few FAKE sleeps
        self.pending: list = []  # (rank, FAKE batch) of a call, written when the next call starts
        # Every FAKE process has its own clock origin and the later ones are a little slower, as the monotonic
        # clocks of real processes are unrelated. A driver that mixed the logs up would show it in its samples.
        self.logs = [
            _FakeLog(recorder_dir / f"sched-{FAKE_FIRST_PID + index}.jsonl", index * 1000 * TICKS_PER_SECOND, index * 7)
            for index in range(processes)
        ]
        if headers:
            self.write_headers()

    def write_headers(self) -> None:
        for index, log in enumerate(self.logs):
            if self.dp_size > 1:
                # FAKE DP attention: process ``index`` is the one scheduler process, and so the leader, of rank
                # ``index``.
                header = _fake_header(
                    FAKE_FIRST_PID + index, 0, self.limits, self.processes, dp_size=self.dp_size, dp_rank=index
                )
            else:
                header = _fake_header(FAKE_FIRST_PID + index, index, self.limits, self.processes)
            if self.edit_header is not None:
                header = self.edit_header(index, header)
            log.write(header)

    def fake_eviction(self, ranks: tuple[int, ...] | None = None) -> None:
        """FAKE fault: the engine needed room and dropped every unlocked cache entry, on all ranks or on some."""
        for rank in range(self.dp_size) if ranks is None else ranks:
            self.caches[rank] = PageCacheModel(self.limits.page_size)

    def _rank_logs(self, rank: int) -> list[_FakeLog]:
        """The FAKE logs that hold the batches of one rank: every process without DP attention, else its own."""
        return self.logs if self.dp_size == 1 else [self.logs[rank]]

    # -- the FAKE scheduler

    def _fake_batches(self, rids: list[str], prompts: list[list[int]], cached: list[int], output_tokens: int) -> list:
        """Extend batches filled up to the chunk budget, then one decode batch for every further output token.

        A request is charged its uncached tokens rounded up to whole pages. One that does not fit what is left
        of the budget is cut: the part that fits goes into this batch and the rest into the next ones. A rank
        that got no request of a call forms no batch.
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
        if rids:
            batches.extend({"mode": "DECODE", "rids": list(rids)} for _step in range(output_tokens - 1))
        return batches

    @staticmethod
    def _fake_fault_in_batches(fault: str | None, batches: list, rids: list[str], prompts: list[list[int]]) -> None:
        """Apply a FAKE fault that changes the batch lines of one rank of a call."""
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
            batches.insert(decodes[2], _idle_batch())
        elif fault == "extend_after_decode":
            batches.insert(1, batches.pop(0))
        elif fault == "second_extend_of_one_request":
            batches.insert(1, _extend_batch(rids[:1], first["prefix_lens"][:1], first["extend_lens"][:1]))
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
        elif fault == "late":
            # FAKE: the rank was released one step after the others. It idles while they run their first batch.
            batches.insert(0, _idle_batch())
        elif fault == "no_forward_iter":
            first["forward_iter"] = None
        elif fault == "decode_as_extend":
            # FAKE: the first decode step of the rank ran as an extend batch of one token per request, with
            # the fields SGLang gives a decode batch it turns into an extend batch.
            batches[decodes[0]] = {
                "mode": "EXTEND",
                "rids": list(rids),
                "prefix_lens": [len(prompt) for prompt in prompts],
                "extend_lens": [1] * len(rids),
                "extend_num_tokens": len(rids),
                FAKE_DECODE_STEP: True,
            }

    def _fake_decode_to_extend(self, rank_batches: list[list[dict]], prompt_tokens: dict[str, int]) -> None:
        """FAKE of what SGLang does to a rank that decodes in a step in which another rank extends.

        Under DP attention SGLang turns such a decode batch into an extend batch, unless a rank idles in the
        step (srt/managers/scheduler_components/dp_attn.py
        ``SchedulerDPAttnAdapter.maybe_convert_decode_to_extend``). The batch keeps its requests and its token
        count of the step. Its mode becomes ``EXTEND``, every request computes one token after everything it
        holds so far, and ``extend_num_tokens`` is the number of requests (srt/managers/schedule_batch.py
        ``ScheduleBatch.convert_decode_to_extend``). SGLang does it only when every rank reports that its
        prefill graphs can replay the step (the vote in ``prepare_mlp_sync_batch_raw``); FAKE: the ranks
        report what ``prefill_graphs_replay`` says, for every step alike. Read in the source at both tags;
        no engine showed such a batch.

        The batches are changed in place, after the faults of the call, so that the modes a test reads off a
        call are the ones the logs hold. Without DP attention nothing is changed.
        """
        if self.dp_size == 1 or not self.prefill_graphs_replay:
            return
        for step in range(max(len(batches) for batches in rank_batches)):
            row = [batches[step] if step < len(batches) else _idle_batch() for batches in rank_batches]
            modes = {batch["mode"] for batch in row}
            if "EXTEND" not in modes or "IDLE" in modes:
                continue
            for batches, batch in zip(rank_batches, row, strict=True):
                if batch["mode"] != "DECODE":
                    continue
                # The decode step with zero-based index i reads the prompt and the i output tokens before it.
                done = sum(1 for earlier in batches[:step] if _is_fake_decode_step(earlier))
                rids = list(batch["rids"])
                batches[step] = {
                    **batch,
                    "mode": "EXTEND",
                    # FAKE: a request the call does not hold has no prompt here.
                    "prefix_lens": [prompt_tokens.get(rid, 0) + done for rid in rids],
                    "extend_lens": [1] * len(rids),
                    "extend_num_tokens": len(rids),
                    FAKE_DECODE_STEP: True,
                }

    def _fake_global_tokens(self, fault: str | None, target: int, rank: int, tokens: list[int]) -> list | None:
        """What one rank writes as ``global_num_tokens`` for a step in which the ranks ran ``tokens``."""
        if fault == "uneven_tokens":
            # FAKE: every rank holds the same list, and the list says that one rank ran one token more.
            return [count + (1 if index == target else 0) for index, count in enumerate(tokens)]
        if fault == "own_tokens_only":
            # FAKE: every rank holds its own count alone, which is what SGLang writes on a batch when it does
            # not keep the counts of all ranks (srt/managers/scheduler_components/dp_attn.py
            # ``_update_gather_batch`` without ``require_mlp_tp_gather``).
            return [tokens[rank]]
        if rank == target:
            if fault == "unequal_tokens":
                return [*tokens[:-1], tokens[-1] + 1]
            if fault == "no_global_tokens":
                return None
        return list(tokens)

    def _write_step(self, fault: str | None, target: int, row: list[dict], start: int) -> None:
        """Write the FAKE batches of one step, one per rank, to the logs of their ranks.

        Under DP attention every line of the step gets the same ``global_num_tokens``. Every log counts its
        own lines, so the lines of one step also carry the same ``forward_iter``.
        """
        tokens = [_fake_step_tokens(batch) for batch in row] if self.dp_size > 1 else None
        for rank, batch in enumerate(row):
            if tokens is not None:
                batch = {**batch, "global_num_tokens": self._fake_global_tokens(fault, target, rank, tokens)}
            for index, log in enumerate(self._rank_logs(rank)):
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

    def _write_batches(self, fault: str | None, target: int, rank_batches: list[list[dict]], stage: str) -> None:
        """Run the FAKE batches of a call: step by step, every rank its next batch, or an idle one."""
        if fault == "silent":
            return
        for step in range(max(len(batches) for batches in rank_batches)):
            row = []
            for rank, batches in enumerate(rank_batches):
                batch = batches[step] if step < len(batches) else _idle_batch()
                faulty = fault if rank == target else None
                if batch["mode"] == "EXTEND":
                    computed = sum(batch["extend_lens"])
                    batch.setdefault("extend_num_tokens", computed + (1 if faulty == "wrong_extend_tokens" else 0))
                    batch["ticks"] = FAKE_EXTEND_BASE_TICKS + computed
                    if stage == "warm":
                        batch["ticks"] += FAKE_WARM_SHOT_EXTRA_TICKS
                elif batch["mode"] == "DECODE":
                    decode_step = sum(1 for earlier in batches[:step] if _is_fake_decode_step(earlier))
                    tenths = FAKE_DECODE_STEP_TENTHS[decode_step % len(FAKE_DECODE_STEP_TENTHS)]
                    batch["ticks"] = (FAKE_DECODE_BASE_TICKS + len(batch["rids"])) * tenths
                else:
                    batch["ticks"] = 0  # a FAKE idle batch takes no time
                if faulty == "zero_duration":
                    batch["ticks"] = 0
                if step:
                    batch["lead_ticks"] = FAKE_OVERLAP_LEAD_TICKS
                row.append(batch)
            start = self.clock.ticks
            # The ranks run a step together: it takes as long as the slowest of them.
            self.clock.advance(max(batch["ticks"] for batch in row))
            self._write_step(fault, target, row, start)
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
        fault, _, named_rank = self.faults.get(call.key, "").partition("@")
        fault = fault or None
        target = int(named_rank) if named_rank else 0
        if stage in ("meas", "run") and self.evictions.get(benchmark_id, 0) > attempt:
            self.fake_eviction(self.eviction_ranks)
        if fault == "evict":
            self.fake_eviction((target,) if named_rank else None)
        # FAKE: a batch the scheduler launched for the call before reaches the log only now. Under DP attention
        # the other ranks idle in that step.
        for rank, batch in self.pending:
            row = [batch if index == rank else {**_idle_batch(), "ticks": 0} for index in range(self.dp_size)]
            self._write_step(None, 0, row, self.clock.ticks)
        self.pending.clear()
        if fault == "raise":
            raise RuntimeError("FAKE engine failure")
        if fault == "system_exit":
            raise SystemExit(0)
        if fault == "keyboard_interrupt":
            raise KeyboardInterrupt

        # FAKE placement. The call names no rank, so each request goes to the next rank in turn, from where
        # the request before it left off, whatever call that one belonged to. Without DP attention that is
        # rank 0 for all.
        size = self.dp_size
        call.ranks = [(self.round_robin + index) % size for index in range(len(call.rids))]
        self.round_robin = (self.round_robin + len(call.rids)) % size
        if fault == "misplaced":
            # FAKE: the first two requests of the call ran on each other's rank.
            call.ranks[0], call.ranks[1] = call.ranks[1], call.ranks[0]
        # A call is one batch of arrivals: every request is matched against the cache of its rank before any
        # is stored.
        cached = [self.caches[rank].cached(prompt) for rank, prompt in zip(call.ranks, call.prompts, strict=True)]
        rank_batches = []
        mine: list[list[int]] = []
        for rank in range(size):
            mine.append([index for index, ran_on in enumerate(call.ranks) if ran_on == rank])
            rank_rids = [call.rids[index] for index in mine[rank]]
            rank_prompts = [call.prompts[index] for index in mine[rank]]
            batches = self._fake_batches(
                rank_rids, rank_prompts, [cached[index] for index in mine[rank]], max_new_tokens
            )
            if rank == target and batches:
                self._fake_fault_in_batches(fault, batches, rank_rids, rank_prompts)
            rank_batches.append(batches)
        self._fake_decode_to_extend(
            rank_batches, {rid: len(prompt) for rid, prompt in zip(call.rids, call.prompts, strict=True)}
        )
        call.modes.extend(batch["mode"] for batch in rank_batches[0])
        call.rank_modes = [[batch["mode"] for batch in batches] for batches in rank_batches]
        self._write_batches(fault, target, rank_batches, stage)
        if fault == "fault_line":
            record = {"type": "fault", "where": "process_batch_result", "error": "FAKE recorder fault"}
            self._rank_logs(target)[0].write(record)
        if fault == "follower_fault_line":
            self.logs[-1].write({"type": "fault", "where": "process_batch_result", "error": "FAKE recorder fault"})
        if fault == "late_decode":
            late_rids = [call.rids[index] for index in mine[target]]
            self.pending.append((target, {"mode": "DECODE", "rids": late_rids, "ticks": FAKE_DECODE_BASE_TICKS}))
        for rank, prompt in zip(call.ranks, call.prompts, strict=True):
            # A finished request stores its prompt and the generated tokens that have KV: all but the last.
            # The FAKE output tokens are negative numbers, so they equal no token id: not an allocated one, not
            # a seeded one and not an id of a text, which a pool can keep from 0 upwards.
            self.caches[rank].finish_prefill([*prompt, *range(-1, -max_new_tokens, -1)])
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
        if size > 1:
            # Under DP attention a response names the rank its request ran on.
            for response, rank in zip(responses, call.ranks, strict=True):
                response["dp_rank"] = rank
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
        elif fault == "no_dp_rank":
            del responses[0]["dp_rank"]
        elif fault == "dp_rank_none":
            responses[0]["dp_rank"] = None
        elif fault == "dp_rank_true":
            responses[0]["dp_rank"] = True
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
    # The text pool and the dataset record of a text run; both None in a run with seeded ids.
    text_pool: object = None
    dataset: dict | None = None

    @property
    def built(self) -> grid.Grid:
        """The grid, planned here from the same limits and settings."""
        return grid.build_grid(self.settings.mode, self.limits, self.settings.caps, manifest=self.settings.manifest)

    def execute(self) -> None:
        low, high = self.tokens
        if self.text_pool is None:
            # A run with seeded ids is started with the call it always had.
            self.run.execute(self.engine, token_low=low, token_high=high)
        else:
            self.run.execute(
                self.engine, token_low=low, token_high=high, text_pool=self.text_pool, dataset=self.dataset
            )

    def payload(self) -> dict:
        """The one rank artifact of a run without DP attention."""
        (payload,) = self.run.payloads()
        return payload

    @property
    def dp_size(self) -> int:
        return self.settings.dp_size

    def rank_paths(self) -> list[str]:
        """Where the rank files go: rank 0 at the output path, rank ``r`` beside it with ``_dp<r>``."""
        return [
            str(self.pod / ("benchmark.json" if rank == 0 else f"benchmark_dp{rank}.json"))
            for rank in range(self.dp_size)
        ]

    def rank_lines(self, rank: int) -> list[dict]:
        """The batch lines the FAKE leader of an attention-DP rank wrote, in file order."""
        text = (self.recorder_dir / f"sched-{FAKE_FIRST_PID + rank}.jsonl").read_text(encoding="utf-8")
        return [line for line in map(json.loads, text.splitlines()) if line["type"] == "batch"]

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
    processes: int | None = None,
    dp_size: int = 1,
    faults: dict | None = None,
    evictions: dict | None = None,
    eviction_ranks: tuple[int, ...] | None = None,
    edit_header=None,
    headers: bool = True,
    prefill_graphs_replay: bool = True,
    tokens: tuple[int, int] = (TOKEN_LOW, TOKEN_HIGH),
    execute: bool = True,
    started_before_seconds: float = 0.0,
    text_pool: object = None,
    dataset: dict | None = None,
    **overrides,
) -> _FakeRun:
    """Lay out a pod, start a fake engine, build a run and, unless told not to, execute it.

    ``started_before_seconds`` is the FAKE time between the start of the launcher and the run: what an engine
    build would take. ``text_pool`` and ``dataset`` make the run a text run. ``dp_size`` above 1 makes it a run
    with that many attention-DP ranks against a FAKE engine with one scheduler process per rank, for a cell
    of the layout ``dep``. ``prefill_graphs_replay`` set to False makes that engine one that never turns a
    decode batch into an extend batch.
    """
    limits = _limits(page_size) if limits is None else limits
    processes = dp_size if processes is None else processes
    cell = _cell(phase="prefill" if mode == "agg" else mode, dp=dp_size, strategy="single" if dp_size == 1 else "dep")
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
        dp_size=dp_size,
        faults=faults,
        evictions=evictions,
        eviction_ranks=eviction_ranks,
        edit_header=edit_header,
        headers=headers,
        prefill_graphs_replay=prefill_graphs_replay,
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
        "dp_size": dp_size,
    }
    values.update(overrides)
    settings = driver.RunSettings(**values)
    started_at = clock.read() - started_before_seconds
    run = driver.BenchmarkRun(settings, recorder_dir, started_at=started_at, clock=clock.read, sleep=clock.sleep)
    fake = _FakeRun(run, engine, clock, cell, tmp_path, pod, recorder_dir, limits, settings, tokens, text_pool, dataset)
    if execute:
        fake.execute()
    return fake


def _start_text(tmp_path: Path, mode: str = "prefill", page_size: int = 1, **options) -> _FakeRun:
    """``_start`` for a text run: a real pool over the FAKE conversations and the FAKE dataset record."""
    options.setdefault("text_pool", _fake_pool())
    options.setdefault("dataset", dict(FAKE_DATASET))
    return _start(tmp_path, mode, page_size, **options)


def _expected_sample(point: grid.Point, rank: int = 0) -> float:
    """The FAKE sample a point must carry on one attention-DP rank, worked out here from the FAKE timings.

    The batch size and the token counts of a point are those of ONE rank. The FAKE process of rank ``r``
    delivers every output seven ticks per rank later than rank 0, which shows in a prefill sample (launch to
    output) and not in a decode step (output to output).
    """
    if point.phase == grid.PREFILL:
        return (FAKE_EXTEND_BASE_TICKS + point.batch_size * point.new_tokens + 7 * rank) / TICKS_PER_SECOND
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
    """A run that is not complete: write it, then the validator raises and the gate fails the pod.

    Returns the file of rank 0. A run with several attention-DP ranks writes one file per rank, all with the
    same status.
    """
    written = fake.run.write()
    assert fake.run.status == status
    rank_paths = fake.rank_paths()
    merged_path = fake.pod / "benchmark_merged.json"
    if status == "failed":
        # A failed run has no merged file.
        assert written == {"status": "failed", "paths": rank_paths}
        assert not merged_path.exists()
    else:
        assert written == {"status": "partial", "paths": [*rank_paths, str(merged_path)]}
    on_disk = [json.loads(Path(path).read_text(encoding="utf-8")) for path in rank_paths]
    assert on_disk == fake.run.payloads()
    for rank, payload in enumerate(on_disk):
        assert (payload["status"], payload["valid"]) == (status, False)
        assert payload["dp"] == {"size": fake.dp_size, "rank": rank}
    with pytest.raises(ValueError, match="invalid native terminal result envelope"):
        _validate(fake.cell, fake.root)
    gate = _gate(rank_paths, fake.cell.workload_kind)
    assert gate.returncode == GATE_INVALID
    assert f"status={status!r} valid=False" in gate.stderr
    return on_disk[0]


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
        "recorder_format": 3,
        "scheduler_processes": 1,
        "limits": fake.limits.as_dict(),
        "page_size": page_size,
        "enable_overlap": True,
        "kv_cache_stored_dtype": FAKE_KV_STORED_DTYPE,
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
    # A run with seeded ids writes the ``kvwarm`` block it always wrote: four keys and no dataset.
    assert on_disk.get("kvwarm") == (None if mode == "prefill" else SEEDED_KVWARM)
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


def test_a_manifest_point_that_does_not_fit_is_rejected_before_any_request(tmp_path):
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
    ("meas", "chunked", _CHUNKED),
    ("meas", "decode_follows", _DECODE_AFTER),
    ("meas", "wrong_extend_tokens", _TOKENS_REPORTED),
    ("meas", "wrong_extend_lens", _TOKENS_COUNTED),
    ("meas", "prompt_count", _PROMPT),
    ("seed", "prompt_count", _PROMPT),
    ("meas", "retraction", _RETRACTED),
    ("meas", "finish_stop", _STOPPED),
    ("warm", "short_completion", _SHORT),
    ("meas", "raise", _RAISED),
    ("meas", "missing_key", "engine_call_failed: {rid0}: cached_tokens is None"),
    ("meas", "wrong_rid", None),
    ("meas", "short_list", None),
    ("meas", "copy_done_false", _NO_COPY_EVENT),
    ("meas", "no_lens", _NO_LENS),
    ("meas", "zero_duration", _NO_TIME.replace("{what}", "extend batch seq {seq}")),
    # A time that is not a JSON float was not read off the scheduler's clock.
    ("meas", "no_out_ts", _RECORDER + ": batch seq {seq} has out_ts=None"),
    ("meas", "whole_number_launch_ts", _RECORDER + ": batch seq {seq} has launch_ts=5000"),
    ("meas", "fault_line", _FAULT_LINE),
    # FAKE: the recorder writes nothing at all for the measured shot.
    (
        "meas",
        "silent",
        "no_fpm_before_deadline: sched-4100.jsonl holds no extend batch for {b} of {b} request(s) after 2.0 s",
    ),
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
    ("run", "retraction", _RETRACTED),
    ("run", "short_completion", _SHORT),
    ("run", "finish_stop", _STOPPED),
    ("run", "copy_done_false", _NO_COPY_EVENT),
    ("run", "decode_copy_done_false", _NO_COPY_EVENT.replace("{seq}", "{seq4}")),
    ("run", "zero_duration", _NO_TIME.replace("{what}", "decode step 1 (seq {seq1})")),
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
        # A recorder of the format before: its header lacks the dtype of the KV pool.
        (_edit("format", 2), "sched-4100.jsonl: format 2, expected 3"),
        (_edit("leader", False), "0 leader header(s), expected exactly one: []"),
        (_edit("leader", 1), "sched-4100.jsonl: leader is 1, expected true or false"),
        (_edit("checks", {}), "sched-4100.jsonl: the header has no checks.disable_radix_cache"),
        (_edit("limits", {"page_size": 1}), "sched-4100.jsonl: the header has no limits.max_req_input_len"),
        (
            _edit("limits.max_total_num_tokens", 1185),
            "sched-4100.jsonl: limits: max_total_num_tokens 1185 is not a multiple of page_size 16",
        ),
        # A header that holds anything but the text of the dtype of the KV pool.
        (_edit("kv_cache_stored_dtype", None), "sched-4100.jsonl: kv_cache_stored_dtype is None, expected a string"),
        (_edit("kv_cache_stored_dtype", ""), "sched-4100.jsonl: kv_cache_stored_dtype is '', expected a string"),
    ],
)
def test_an_invalid_header_fails_the_run(tmp_path, edit, reason):
    limits = _limits(16)
    fake = _start(tmp_path, limits=limits, edit_header=edit)
    _assert_failed_before_the_grid(fake, f"recorder_header_invalid: {reason}")


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
_OTHER_LAYOUT = (
    "the scheduler reports attn_dp_size={size} and attn_dp_enabled={enabled}, the run was set up for {planned} "
    "attention-DP rank(s)"
)

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
    # Only the boolean false passes: a recorder that wrote another falsy value did not write format 3.
    ("checks.disable_radix_cache", 0, _RADIX_OFF.format(v="0")),
    ("checks.schedule_policy", "lpm", _POLICY.format(name="schedule_policy=lpm")),
    ("checks.prefill_decode_interval", 22, _INTERVAL.format(v="22")),
    ("checks.prefill_decode_interval", False, _INTERVAL.format(v="false")),
    ("checks.max_new_tokens_limit", 6, _OUTPUT_CAP.format(v="6")),
    ("checks.pp_max_micro_batch_size", 8, "pp_max_micro_batch_size 8 differs from the running cap 9"),
    # The run was set up without DP attention: a scheduler that reports another layout is refused.
    ("ranks.attn_dp_enabled", True, _OTHER_LAYOUT.format(size="1", enabled="true", planned=1)),
    ("ranks.attn_dp_size", 2, _OTHER_LAYOUT.format(size="2", enabled="false", planned=1)),
    ("ranks.pp_size", 2, "pipeline parallelism is not supported (pp_size=2)"),
    ("sglang_version", "0.5.24", "SGLang 0.5.24 is not a tested version (tested: 0.5.22, 0.5.23)"),
    # A value the recorder could not read is never taken for an allowed one.
    _unreadable("leader"),
    _unreadable("kv_cache_stored_dtype"),
    _unreadable("limits.page_size"),
    _unreadable("checks.recv_skipper"),
]


@pytest.mark.parametrize(
    ("path", "value", "reason"), STAGE_H_REJECTIONS, ids=[f"{row[0]}={row[1]!r:.12}" for row in STAGE_H_REJECTIONS]
)
def test_a_scheduler_setting_the_rules_do_not_cover_is_rejected(tmp_path, path, value, reason):
    fake = _start(tmp_path, edit_header=_edit(path, value))
    on_disk = _assert_failed_before_the_grid(fake, f"rejected: {reason}")
    assert on_disk["producer"]["scheduler_processes"] == 1


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
        # The driver is not told the KV-cache label of its cell, so every dtype the processes agree on passes
        # here. The host holds it against the cell.
        ("kv_cache_stored_dtype", "torch.bfloat16"),
        ("kv_cache_stored_dtype", "torch.float32"),
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
        # Every scheduler process holds a KV pool of its own: a process that stored another dtype fails the run.
        (
            "kv_cache_stored_dtype",
            "torch.bfloat16",
            'kv_cache_stored_dtype: ["torch.float8_e4m3fn", "torch.bfloat16"]',
        ),
    ],
)
def test_headers_that_disagree_are_rejected(tmp_path, path, value, reason):
    fake = _start(tmp_path, processes=2, edit_header=_edit(path, value, process=1))
    _assert_failed_before_the_grid(fake, f"rejected: scheduler processes disagree on {reason}")


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


def test_a_token_range_below_the_content_minimum_fails_the_run(tmp_path):
    fake = _start(tmp_path, tokens=(1000, 5095))

    reason = "token_range_too_small: the token range [1000, 5095) holds 4095 ids, the content plan needs at least 4096"
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


# ---------------------------------------------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------------------------------------------

_FREE_RID = "sglb-fa4e0123-p0-free{call}-r0-s0-e"
_OTHER_COUNT = "{wrong} of 2 request(s) report another cached count than the planned {planned}: {rid} reports {got}"
_ALL_BUT_LAST = ["batch_shape", "aligned_prefix", "unaligned_prompt"]

SELF_TEST_FAULTS = [
    # (call, FAKE fault, reason after "self_test_failed: ", the checks that passed before)
    (FREE_BATCH_SHAPE, "split_extend", "batch_shape: " + _SPLIT, []),
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
]


@pytest.mark.parametrize(
    ("call", "fault", "reason", "passed"), SELF_TEST_FAULTS, ids=[f"{row[0]}-{row[1]}" for row in SELF_TEST_FAULTS]
)
def test_a_failed_self_test_check_ends_the_run_before_the_sweep(tmp_path, call, fault, reason, passed):
    fake = _start(tmp_path, "prefill", faults={(0, "free", call): fault})

    assert (fake.run.status, fake.run.reason) == ("failed", f"self_test_failed: {reason}")
    # No point was staged and no later check ran.
    assert fake.sent_benchmark_ids() == {0}
    assert [made.key[2] for made in fake.engine.calls] == list(range(call + 1))
    on_disk = _assert_rejected(fake, "failed")
    assert on_disk["producer"]["self_test"] == passed
    assert (on_disk["skipped_points"], on_disk["coverage"]["completed_points"]) == ([], 0)
    assert on_disk["coverage"]["expected_points"] == len(fake.built.points)


def test_the_self_test_expects_the_cached_counts_of_the_page_size(tmp_path):
    fake = _start(tmp_path, "decode", limits=_limits(16))
    assert fake.run.status == "complete"

    unit = 16  # one page of 16 tokens: the smallest page multiple of at least 16 tokens
    free = [call for call in fake.engine.calls if call.key[0] == 0]
    assert [len(call.prompts[0]) for call in free[:6]] == [unit, unit, 2 * unit, 2 * unit + 7, 2 * unit + 7, unit]


def test_limits_that_do_not_admit_a_self_test_call_fail_the_self_test(tmp_path):
    # A chunk budget of one cached unit holds one request of 16 tokens, not two.
    fake = _start(tmp_path, limits=_limits(1, chunked_prefill_size=16))
    assert fake.run.reason == (
        "self_test_failed: batch_shape: the limits of this engine do not admit two requests of 16 tokens in one "
        "extend batch: chunk_budget"
    )
    assert fake.engine.calls == []


# ---------------------------------------------------------------------------------------------------------------
# Soft deadline
# ---------------------------------------------------------------------------------------------------------------


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


@pytest.mark.parametrize(
    ("mode", "reason"),
    [("decode", "measured_decode_context_mismatch"), ("prefill", "measured_prefill_tokens_mismatch")],
)
def test_a_batch_that_is_written_late_for_the_last_call_before_the_deadline_fails_the_run(
    tmp_path, monkeypatch, mode, reason
):
    completed = 4
    fake = _start(tmp_path, mode, execute=False)
    real = artifact.build_measurement

    def build_then_late_batch_and_deadline(record, rank_fpms):
        # FAKE: the scheduler ran one decode batch more for the last call of point 4. Its line reaches the log
        # when that call has been checked, and the soft deadline passes before the next point.
        if record["benchmark_id"] == completed:
            late = {"mode": "DECODE", "rids": fake.engine.calls[-1].rids, "ticks": FAKE_DECODE_BASE_TICKS}
            fake.engine.logs[0].batch(late, fake.clock.ticks)
            fake.clock.advance(round(FAKE_TIMEOUT_SECONDS * TICKS_PER_SECOND))
        return real(record, rank_fpms)

    monkeypatch.setattr(artifact, "build_measurement", build_then_late_batch_and_deadline)
    fake.execute()

    # Not a partial run: the late batch fails it, and it names no point, as a late batch at the end of a sweep.
    assert fake.run.status == "failed"
    assert fake.run.reason.startswith(f"{reason}: a DECODE batch (seq ")
    assert fake.run.reason.endswith(") holds requests of a call that had already been checked")
    # Nothing was sent for a point after the deadline.
    assert fake.sent_benchmark_ids() == set(range(completed + 1))
    on_disk = _assert_rejected(fake, "failed")
    # The file keeps both facts and the rows of the points that completed.
    assert (on_disk["stop_reason"], on_disk["error"], on_disk["usable"]) == ("timeout", fake.run.reason, False)
    assert on_disk["coverage"] == {
        "expected_points": len(fake.built.points),
        "completed_points": completed,
        "skipped_points": 0,
    }
    assert [row["point"] for row in on_disk["results"]] == fake.built.point_dicts()[:completed]


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
        (
            {"expected_scheduler_processes": 0},
            "internal_error: ValueError: settings.expected_scheduler_processes must be >= 1, got 0",
        ),
        ({"timeout_seconds": 0}, "internal_error: ValueError: settings.timeout_seconds must be finite and > 0, got 0"),
    ],
)
def test_a_bad_setting_is_recorded_as_an_internal_error_before_any_request(tmp_path, overrides, reason):
    fake = _start(tmp_path, **overrides)
    assert (fake.run.status, fake.run.reason) == ("failed", reason)
    assert fake.engine.calls == []


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
        ({"config": None}, "internal_error: TypeError: settings.config must be a dict, got NoneType"),
        (
            {"producer": {"page_size": 64}},
            "internal_error: ValueError: settings.producer may not contain ['page_size']",
        ),
        ({"producer": {"nan": float("nan")}}, "internal_error: ValueError: producer is not JSON data: "),
    ],
)
def test_a_block_the_artifact_refuses_fails_at_the_dry_build(tmp_path, overrides, start):
    fake = _start(tmp_path, **overrides)
    assert fake.run.reason.startswith(start), fake.run.reason
    assert fake.engine.calls == []
    assert "omitted_launcher_blocks" in _assert_rejected(fake, "failed")["producer"]


def test_write_leaves_the_launcher_blocks_out_when_they_cannot_be_written(tmp_path):
    fake = _start(tmp_path, "decode")
    assert fake.run.status == "complete"
    fake.settings.producer["engine_shutdown_error"] = RuntimeError("FAKE: not JSON")

    on_disk = _assert_rejected(fake, "failed")
    text = fake.run.reason.removeprefix("internal_error: ")
    assert text.startswith("TypeError: producer is not JSON data: ")
    assert (on_disk["config"], on_disk["producer"]["omitted_launcher_blocks"]) == ({"mode": "decode"}, text)
    assert len(on_disk["results"]) == len(fake.built.points)


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
    header = {"type": "header", "format": 3}
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


def test_recorder_logs_list_fault_lines_and_everything_that_is_not_a_line_of_the_format(tmp_path):
    logs = driver.RecorderLogs(tmp_path)
    good = tmp_path / "sched-1.jsonl"
    good.write_bytes(
        _line({"type": "header", "format": 3})
        + _line(_batch_line(1))
        + _line({"type": "fault", "where": "run_batch", "error": "FAKE"})
        + b"not json\n"
        + _line({"type": "header", "format": 3})
        + _line(_batch_line(3))  # seq 2 was lost
        + _line(_batch_line(4))
    )
    (tmp_path / "sched-2.jsonl").write_bytes(_line(_batch_line(1)))
    logs.poll()

    assert list(logs.headers) == ["sched-1.jsonl"]
    # A line that is not format 3 is not kept.
    assert [line["seq"] for line in logs.batches("sched-1.jsonl")] == [1, 4]
    assert logs.batches("sched-2.jsonl") == []
    # The fault line is passed on as it is; the reader lists every other line with its number.
    assert logs.faults[0] == {"file": "sched-1.jsonl", "where": "run_batch", "error": "FAKE"}
    expected = [
        ("sched-1.jsonl", "line 4: not a JSON line: "),
        ("sched-1.jsonl", "line 5: a header that is not the first line of its file"),
        ("sched-1.jsonl", "line 6: batch line with seq 3, expected 2"),
        ("sched-2.jsonl", "line 1: a batch line in a file without a header"),
    ]
    assert [(fault["file"], fault["where"]) for fault in logs.faults[1:]] == [(name, "reader") for name, _ in expected]
    assert all(fault["error"].startswith(start) for fault, (_, start) in zip(logs.faults[1:], expected, strict=True))


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


# ---------------------------------------------------------------------------------------------------------------
# Text runs: prompts made of the conversations of a text pool
# ---------------------------------------------------------------------------------------------------------------


def _longest_prompt(fake: _FakeRun) -> int:
    """The longest prompt a run sends, worked out here: the points, the self-test and the warm-up."""
    unit = grid.cached_unit(fake.limits.page_size)
    lengths = [2 * unit + 7, 256]
    for point in fake.built.points:
        lengths.append(point.cached_tokens + point.new_tokens if point.phase == "prefill" else point.cached_tokens - 3)
    return max(lengths)


@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize("mode", ["prefill", "decode", "agg"])
def test_a_text_run_of_a_small_grid_passes_the_validator_and_the_gate(tmp_path, mode, page_size):
    fake = _start_text(tmp_path, mode, page_size)

    built = fake.built
    assert len(built.points) >= 8 and built.missing_phases == ()
    # Every cached count the fake engine's cache model reported was the planned one, although every FAKE
    # conversation starts with the same five ids: no point was staged twice.
    on_disk = _assert_accepted(fake)
    producer = on_disk["producer"]
    assert producer["restaged_points"] == []

    # The producer block names the source, the dataset and the pool beside what a seeded run writes.
    pool = fake.text_pool
    assert producer["content"] == {
        "source": "dataset_text_chains",
        "seed": "0",
        "token_range": [TOKEN_LOW, TOKEN_HIGH],
        "issued": producer["content"]["issued"],
        "dataset": FAKE_DATASET,
        "pool": {
            "id_range": [TOKEN_LOW, TOKEN_HIGH],
            "conversations": 64,
            "tokens": pool.tokens,
            "dropped_tokens": 3,
            "dropped_conversations": 1,
        },
    }
    assert pool.tokens == sum(len(row) for row in _fake_conversations()) - 3
    assert (driver.content.CONTENT_TEXT_CHAINS, driver.content.CONTENT_SEEDED_IDS) == (
        "dataset_text_chains",
        "seeded_token_ids",
    )
    seeded_keys = set(_start(tmp_path / "seeded", mode, page_size).payload()["producer"])
    assert set(producer) == seeded_keys
    # The ``kvwarm`` block of a decode or agg artifact names the dataset with Dynamo's four keys; a prefill
    # artifact has no such block.
    if mode == "prefill":
        assert "kvwarm" not in on_disk
    else:
        assert on_disk["kvwarm"] == {**SEEDED_KVWARM, "dataset": FAKE_KVWARM_DATASET}
        merged = json.loads((fake.pod / "benchmark_merged.json").read_text(encoding="utf-8"))
        assert merged["kvwarm"] == on_disk["kvwarm"]
    # A prompt that is one slot prefix or one free prompt is text after its first token.
    free = [call for call in fake.engine.calls if call.key[0] == 0]
    whole = [free[number] for number in (FREE_BATCH_SHAPE, FREE_UNALIGNED_SEED, FREE_SEVEN_TOKEN_RUN, FREE_WARMUP)]
    whole += [call for call in fake.engine.calls if call.key[1] in ("seed", "run")]
    assert len(free) == 7 and len(whole) > 8
    for call in whole:
        assert all(_is_fake_text(prompt[1:]) for prompt in call.prompts)


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_a_text_pool_that_is_not_deeper_than_the_longest_prompt_fails_before_any_request(tmp_path, mode):
    need = _longest_prompt(_plan_only(tmp_path, mode)) + 1
    # Typed here on their own: the longest prefill prompt of this grid has 420 tokens, and the deepest decode
    # context is the input limit of 428 tokens, whose prompt has three tokens less.
    assert need == {"prefill": 421, "decode": 426}[mode]

    def pool_of(tokens: int):
        # Two FAKE conversations that hold ``tokens`` ids in all.
        ids = [FAKE_TEXT_LOW + step % 50 for step in range(tokens)]
        return _fake_pool([ids[:100], ids[100:]])

    fake = _start_text(tmp_path / "shallow", mode, text_pool=pool_of(need))
    reason = f"content_too_shallow: the text pool holds {need} tokens and the longest prompt needs {need}"
    assert (fake.run.status, fake.run.reason) == ("failed", reason)
    assert fake.engine.calls == []
    on_disk = _assert_rejected(fake, "failed")
    # The grid exists by then.
    assert on_disk["coverage"]["expected_points"] == len(fake.built.points)

    # One token more is enough: every chain can then give the longest prompt and the token after it.
    deep = _start_text(tmp_path / "deep", mode, text_pool=pool_of(need + 1))
    assert (deep.run.status, deep.run.reason) == ("complete", None)
    _assert_accepted(deep)


@pytest.mark.parametrize(("mode", "page_size"), [("prefill", 1), ("decode", 64)])
def test_a_text_run_with_a_pool_wider_than_the_token_range_sends_the_ids_outside_it(tmp_path, mode, page_size):
    # The launcher's pool keeps the ids the model counts, so a text can hold ids outside the token range of
    # the run. Here the pool keeps the three FAKE ids that a pool over the token range drops.
    outside = {5, TOKEN_HIGH, TOKEN_HIGH + 7}
    pool = _fake_pool(tokens=(0, TOKEN_HIGH + 8))
    assert (pool.conversations, pool.dropped_tokens, pool.dropped_conversations) == (65, 0, 0)
    fake = _start_text(tmp_path, mode, page_size, text_pool=pool)

    # Every cached count the fake engine's cache model reported was the planned one: no point was staged twice.
    on_disk = _assert_accepted(fake)
    producer = on_disk["producer"]
    assert producer["restaged_points"] == []
    assert producer["content"]["token_range"] == [TOKEN_LOW, TOKEN_HIGH]
    # The pool's own range is written beside the token range, which it holds.
    assert producer["content"]["pool"] == {
        "id_range": [0, TOKEN_HIGH + 8],
        "conversations": 65,
        "tokens": sum(len(row) for row in _fake_conversations()),
        "dropped_tokens": 0,
        "dropped_conversations": 0,
    }
    prompts = [prompt for call in fake.engine.calls for prompt in call.prompts]
    # The first token of every prompt is an allocated one, from the token range; the text after it is not
    # held to the range, and ids outside it reached the engine.
    assert all(TOKEN_LOW <= prompt[0] < TOKEN_HIGH for prompt in prompts)
    sent_outside = {token for prompt in prompts for token in prompt if not TOKEN_LOW <= token < TOKEN_HIGH}
    assert sent_outside and sent_outside <= outside


# ---------------------------------------------------------------------------------------------------------------
# Attention-DP ranks: balanced calls, every check per rank, the ranks held against each other
# ---------------------------------------------------------------------------------------------------------------

# Under FAKE DP attention the second FAKE process is the leader of rank 1.
RANK_1_LOG = "sched-4101.jsonl"
_PLACEMENT = "dp_rank_placement_mismatch"
_STEP = "dp_ranks_out_of_step"
_STEPS_DIFFER = _STEP + ": batch 1 of {of} (EXTEND) ran in the steps [{step}, {step1}], one forward_iter per rank"
_TOKENS_DIFFER = (
    _STEP + ": batch 1 of {of} (EXTEND) in step {step} has the global_num_tokens {tokens}, one list per rank; "
    "expected 2 equal counts, the same on every rank"
)
# A decode run in which the log of a rank holds a decode step that ran as an extend batch.
_DECODE_AS_EXTEND = (
    _STEP + ": rank {rank}: the batches of the run are {found}; the EXTEND batch(es) with seq {seqs} hold only "
    "requests that an earlier batch of the run had extended, which is how SGLang runs a decode step of a rank "
    "while another rank extends"
)


def _lines_of_call(fake: _FakeRun, rank: int, call: _FakeCall) -> list[dict]:
    """The batch lines of a rank's FAKE leader log that hold a request of that rank of a call, in log order."""
    wanted = set(call.rids[rank :: fake.dp_size])
    return [line for line in fake.rank_lines(rank) if wanted & set(line["rids"])]


def _assert_decode_steps_as_extend(lines: list[dict], rank: int, prompts: list[list[int]]) -> None:
    """``lines`` are the first decode steps of one rank of a call, which hold the requests of ``prompts``. The
    FAKE engine must have written each as SGLang writes a decode batch it has turned into an extend batch."""
    assert lines
    for step, line in enumerate(lines):
        assert (line["mode"], line["extend_lens"], line["extend_num_tokens"]) == (
            "EXTEND",
            [1] * len(prompts),
            len(prompts),
        )
        # Every request reads its prompt and the output tokens of the decode steps before this one.
        assert line["prefix_lens"] == [len(prompt) + step for prompt in prompts]
        # The count of the rank in the step is still the one of a decode batch: its requests.
        assert line["global_num_tokens"][rank] == len(prompts)


def _rank_rid(benchmark_id: int, stage: str, attempt: int, rank: int, slot: int) -> str:
    return f"sglb-{FAKE_RUN_TAG}-p{benchmark_id}-{stage}{attempt}-r{rank}-s{slot}-e"


def _line_of(fake: _FakeRun, rank: int, rid: str) -> dict:
    """The first batch line of a rank's FAKE leader log that holds a request."""
    return next(line for line in fake.rank_lines(rank) if rid in line["rids"])


def _assert_ranks_accepted(fake: _FakeRun) -> list[dict]:
    """A complete run with several attention-DP ranks: write it, then both checkers accept every rank file.

    Returns the rank files as they are on disk, in rank order.
    """
    built = fake.built
    size = fake.dp_size
    assert (fake.run.status, fake.run.reason) == ("complete", None)
    written = fake.run.write()
    rank_paths = fake.rank_paths()
    merged_path = fake.pod / "benchmark_merged.json"
    assert written == {"status": "complete", "paths": [*rank_paths, str(merged_path)]}
    assert sorted(path.name for path in fake.pod.glob("benchmark*")) == sorted(
        [*(Path(path).name for path in rank_paths), merged_path.name]
    )
    on_disk = [json.loads(Path(path).read_text(encoding="utf-8")) for path in rank_paths]
    assert on_disk == fake.run.payloads()
    for rank, payload in enumerate(on_disk):
        assert (payload["status"], payload["valid"], payload["usable"]) == ("complete", True, True)
        assert payload["dp"] == {"size": size, "rank": rank}
        assert payload["coverage"] == {
            "expected_points": len(built.points),
            "completed_points": len(built.points),
            "skipped_points": 0,
        }
        assert (payload["run_id"], payload["grid_digest"]) == (FAKE_RUN_ID, built.digest)
        for point, row, group in zip(built.points, payload["results"], payload["iteration_groups"], strict=True):
            # A row holds the record of its own rank: the rank's FAKE sample, and the batch size and the token
            # counts of ONE rank, which are the point's.
            (fpm,) = row["fpms"]
            assert row["point"] == point.as_dict()
            assert (fpm["dp_rank"], fpm["wall_time"], fpm["worker_id"]) == (
                rank,
                _expected_sample(point, rank),
                FAKE_WORKER_ID,
            )
            assert fpm["scheduled_requests"] == artifact.scheduled_requests(point.as_dict())
            if point.phase == grid.DECODE:
                assert fpm[artifact.DECODE_STEPS_KEY] == _expected_steps(point)
            # A group holds the record of every rank, and its time is the slowest rank's.
            assert [entry["dp_rank"] for entry in group["rank_results"]] == list(range(size))
            assert group["rank_results"][rank]["fpms"] == [fpm]
            assert group["wall_time"] == max(_expected_sample(point, other) for other in range(size))
        # Every rank file carries the same groups and the same facts of the run.
        assert payload["iteration_groups"] == on_disk[0]["iteration_groups"]
        assert payload["producer"] == on_disk[0]["producer"]
    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    assert (merged["artifact_type"], merged["rank_files"]) == ("merged", rank_paths)
    assert merged["dp"]["source_ranks"] == list(range(size))
    if fake.settings.mode != grid.AGG:
        collection = _validate(fake.cell, fake.root)
        assert [measurement.point for measurement in collection.points] == built.point_dicts()
        assert [measurement.rank_wall_times for measurement in collection.points] == [
            tuple((rank, _expected_sample(point, rank)) for rank in range(size)) for point in built.points
        ]
        assert (collection.runtime_run_id, collection.runtime_grid_digest) == (FAKE_RUN_ID, built.digest)
    gate = _gate(rank_paths, fake.settings.mode)
    assert gate.returncode == GATE_ACCEPT, gate.stderr
    return on_disk


def _assert_not_staged_again(fake: _FakeRun) -> None:
    assert fake.run.payloads()[0]["producer"]["restaged_points"] == []
    assert {parts["attempt"] for parts in fake.sent() if parts["benchmark_id"]} == {0}


@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize("mode", ["prefill", "decode", "agg"])
def test_a_complete_run_with_two_ranks_passes_the_validator_and_the_gate(tmp_path, mode, page_size):
    fake = _start(tmp_path / "two", mode, page_size, dp_size=2)
    single = _start(tmp_path / "one", mode, page_size)

    built = fake.built
    # The grid is a grid of per-rank points: the one a run without DP attention plans from the same limits.
    assert built.points == single.built.points and len(built.points) >= 8
    on_disk = _assert_ranks_accepted(fake)

    # The producer block says how the stages were sent. Its other keys are those of a run with one rank.
    producer, one_rank = on_disk[0]["producer"], single.payload()["producer"]
    assert producer["staging"] == "balanced_batch_call" and one_rank["staging"] == "one_batch_call"
    assert (producer["scheduler_processes"], producer["recorder_format"]) == (2, 3)
    assert set(producer) == set(one_rank)
    assert (producer["limits"], producer["self_test"]) == (fake.limits.as_dict(), list(driver.SELF_TEST_CHECKS))
    assert producer["restaged_points"] == []
    # The first-token need is the need of one rank, and the plan issued it once for each of the two.
    assert producer["first_token_need"] == one_rank["first_token_need"]
    issued, issued_by_one = producer["content"]["issued"], one_rank["content"]["issued"]
    assert issued == {key: 2 * value for key, value in issued_by_one.items()}

    # Every call is balanced: twice the requests of the same call of a run with one rank, slot by slot and rank
    # inside, and each request ran on the rank its id names.
    assert [call.key for call in fake.engine.calls] == [call.key for call in single.engine.calls]
    for call, one in zip(fake.engine.calls, single.engine.calls, strict=True):
        benchmark_id, stage, attempt = call.key
        slots = len(one.rids)
        assert call.rids == [
            _rank_rid(benchmark_id, stage, attempt, rank, slot) for slot in range(slots) for rank in range(2)
        ]
        assert call.ranks == [0, 1] * slots
        assert (call.max_new_tokens, call.extra) == (one.max_new_tokens, ((), {}))
        # Rank 0 is sent what the run with one rank sends: content depends on the seed, the rank and the slot.
        assert (call.prompts[0::2], call.rids[0::2]) == (one.prompts, one.rids)
        # Rank 1 is sent prompts of the same lengths and of its own content.
        assert [len(prompt) for prompt in call.prompts[1::2]] == [len(prompt) for prompt in one.prompts]
        assert all(own != other for own, other in zip(call.prompts[0::2], call.prompts[1::2], strict=True))
        # Both ranks ran the batches of the call in the same shape.
        assert call.rank_modes == [one.modes, one.modes]


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_a_complete_run_with_three_ranks_passes_the_validator_and_the_gate(tmp_path, mode):
    fake = _start(tmp_path, mode, dp_size=3)

    on_disk = _assert_ranks_accepted(fake)
    assert [Path(path).name for path in fake.rank_paths()] == [
        "benchmark.json",
        "benchmark_dp1.json",
        "benchmark_dp2.json",
    ]
    assert [payload["dp"] for payload in on_disk] == [{"size": 3, "rank": rank} for rank in range(3)]
    assert all(call.ranks == [0, 1, 2] * (len(call.rids) // 3) for call in fake.engine.calls)
    # The warm-up is one prompt of 256 tokens on every rank, and the self-test a pair on every rank.
    free = [call for call in fake.engine.calls if call.key[0] == 0]
    assert [len(call.prompts) for call in free] == [6, 6, 6, 6, 6, 6, 3]
    assert [len(prompt) for prompt in free[FREE_WARMUP].prompts] == [256, 256, 256]


@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_a_text_run_with_two_ranks_passes_the_validator_and_the_gate(tmp_path, mode, page_size):
    fake = _start_text(tmp_path, mode, page_size, dp_size=2)

    on_disk = _assert_ranks_accepted(fake)
    producer = on_disk[0]["producer"]
    assert (producer["staging"], producer["restaged_points"]) == ("balanced_batch_call", [])
    assert producer["content"]["source"] == "dataset_text_chains"
    assert producer["content"]["dataset"] == FAKE_DATASET
    if mode == "decode":
        # Every rank file names the dataset in its ``kvwarm`` block, and the merged file with them.
        assert [payload["kvwarm"] for payload in on_disk] == [{**SEEDED_KVWARM, "dataset": FAKE_KVWARM_DATASET}] * 2
    # A prompt that is one slot prefix or one free prompt is text after its first token, on both ranks, and a
    # rank's prompts are its own.
    free = [call for call in fake.engine.calls if call.key[0] == 0]
    whole = [free[number] for number in (FREE_BATCH_SHAPE, FREE_UNALIGNED_SEED, FREE_SEVEN_TOKEN_RUN, FREE_WARMUP)]
    whole += [call for call in fake.engine.calls if call.key[1] in ("seed", "run")]
    assert len(whole) > 8
    for call in whole:
        assert all(_is_fake_text(prompt[1:]) for prompt in call.prompts)
    for call in fake.engine.calls:
        assert all(own != other for own, other in zip(call.prompts[0::2], call.prompts[1::2], strict=True))


# -- placement


@pytest.mark.parametrize(
    ("mode", "measured", "fault", "found"),
    [
        ("prefill", True, "misplaced", "dp_rank 1"),
        ("decode", True, "misplaced", "dp_rank 1"),
        ("decode", False, "misplaced", "dp_rank 1"),
        ("prefill", True, "no_dp_rank", "no dp_rank"),
        ("prefill", True, "dp_rank_none", "no dp_rank"),
        # Only an int is a rank: the engine object hands over an int or None.
        ("prefill", True, "dp_rank_true", "dp_rank True"),
    ],
)
def test_a_request_that_did_not_run_on_its_planned_rank_fails_the_point(tmp_path, mode, fault, found, measured):
    victim = _victim(_plan_only(tmp_path, mode, dp_size=2))
    stage = "seed" if not measured else "meas" if mode == "prefill" else "run"
    fake = _start(tmp_path, mode, dp_size=2, faults={(victim.benchmark_id, stage, 0): fault})

    rid = _rank_rid(victim.benchmark_id, stage, 0, 0, 0)
    _assert_point_failed(
        fake, victim, f"{_PLACEMENT}: {rid}: the engine reports {found}, the request was planned for rank 0"
    )
    # Placement is looked at on every call, the seed included, and a point that was placed wrongly is not
    # staged again.
    _assert_not_staged_again(fake)
    assert fake.calls_of(victim.benchmark_id)[-1].key[1] == stage


# -- the same step on every rank


@pytest.mark.parametrize("stage", ["meas", "warm"])
def test_a_rank_that_runs_a_prefill_shot_one_step_late_fails_the_point(tmp_path, stage):
    victim = _victim(_plan_only(tmp_path, "prefill", dp_size=2))
    fake = _start(tmp_path, "prefill", dp_size=2, faults={(victim.benchmark_id, stage, 0): "late@1"})

    # Each rank's own log looks right: its one batch holds exactly its requests. Rank 1 idled in the step in
    # which rank 0 ran its batch, and rank 0 in the step of rank 1.
    (call,) = fake.calls_of(victim.benchmark_id, stage)
    assert call.rank_modes == [["EXTEND"], ["IDLE", "EXTEND"]]
    first = _line_of(fake, 0, call.rids[0])
    late = _line_of(fake, 1, call.rids[1])
    assert (first["mode"], late["mode"]) == ("EXTEND", "EXTEND")
    assert late["forward_iter"] == first["forward_iter"] + 1
    assert sorted(first["rids"]) == sorted(call.rids[0::2]) and sorted(late["rids"]) == sorted(call.rids[1::2])
    reason = _STEPS_DIFFER.format(of=1, step=first["forward_iter"], step1=late["forward_iter"])
    _assert_point_failed(fake, victim, reason)
    _assert_not_staged_again(fake)


def test_a_rank_that_runs_a_decode_run_one_step_late_fails_the_point_at_once(tmp_path):
    victim = _victim(_plan_only(tmp_path, "decode", dp_size=2))
    fake = _start(tmp_path, "decode", dp_size=2, faults={(victim.benchmark_id, "run", 0): "late@1"})

    # Rank 1 idled in the first step and extended in the second, in which rank 0 ran its first decode step.
    # The FAKE engine wrote that step of rank 0 as SGLang does: as an extend batch of one token per request.
    # So the log of rank 0 holds two EXTEND and five DECODE lines for the run.
    (run,) = fake.calls_of(victim.benchmark_id, "run")
    assert run.rank_modes == [["EXTEND", "EXTEND"] + ["DECODE"] * 5, ["IDLE", "EXTEND"] + ["DECODE"] * 6]
    on_time, late = _lines_of_call(fake, 0, run), _lines_of_call(fake, 1, run)
    assert (len(on_time), len(late)) == (7, 7)
    assert late[0]["forward_iter"] == on_time[1]["forward_iter"] == on_time[0]["forward_iter"] + 1
    _assert_decode_steps_as_extend(on_time[1:2], 0, run.prompts[0::2])
    # The line holds exactly the requests of rank 0, each of them for the second time in an extend batch.
    assert sorted(on_time[1]["rids"]) == sorted(on_time[0]["rids"]) == sorted(run.rids[0::2])
    reason = _DECODE_AS_EXTEND.format(rank=0, found="2 EXTEND, 5 DECODE", seqs=[on_time[1]["seq"]])
    _assert_point_failed(fake, victim, reason)
    _assert_not_staged_again(fake)
    # The run failed when its batches were looked at: the driver did not wait for a sixth DECODE line of rank 0
    # until the log wait ran out.
    assert fake.clock.sleeps == 0


def test_a_rank_that_runs_a_decode_run_one_step_late_on_an_engine_that_keeps_its_decode_batches(tmp_path):
    # FAKE: the prefill graphs cannot replay the step, so the decode step of rank 0 beside the extend batch of
    # rank 1 stays a decode batch, as SGLang leaves it then.
    victim = _victim(_plan_only(tmp_path, "decode", dp_size=2))
    faults = {(victim.benchmark_id, "run", 0): "late@1"}
    fake = _start(tmp_path, "decode", dp_size=2, faults=faults, prefill_graphs_replay=False)

    # Each rank's own log looks right: one extend batch and six decode batches with exactly its requests.
    # Rank 1 ran each of them one step after rank 0.
    (run,) = fake.calls_of(victim.benchmark_id, "run")
    assert run.rank_modes == [["EXTEND"] + ["DECODE"] * 6, ["IDLE", "EXTEND"] + ["DECODE"] * 6]
    first, late = _lines_of_call(fake, 0, run)[0], _lines_of_call(fake, 1, run)[0]
    assert late["forward_iter"] == first["forward_iter"] + 1
    reason = _STEPS_DIFFER.format(of=7, step=first["forward_iter"], step1=late["forward_iter"])
    _assert_point_failed(fake, victim, reason)
    _assert_not_staged_again(fake)
    assert fake.clock.sleeps == 0


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_a_seed_is_not_held_to_one_step_on_all_ranks(tmp_path, mode):
    # Nothing of a seed is kept: it only has to complete on its rank, in whatever steps.
    victim = _victim(_plan_only(tmp_path, mode, dp_size=2))
    fake = _start(tmp_path, mode, dp_size=2, faults={(victim.benchmark_id, "seed", 0): "late@1"})

    (seed,) = fake.calls_of(victim.benchmark_id, "seed")
    assert seed.rank_modes[1][0] == "IDLE"
    _assert_ranks_accepted(fake)


@pytest.mark.parametrize(
    ("fault", "tokens"),
    [
        # The lists of the two ranks differ.
        ("unequal_tokens@1", "[[8, 8], [8, 9]]"),
        # Both ranks hold the same list, and it says that rank 1 ran another number of tokens than rank 0.
        ("uneven_tokens@1", "[[8, 9], [8, 9]]"),
        # A rank whose batch holds no counts.
        ("no_global_tokens@0", "[null, [8, 8]]"),
        # Every rank holds the same list, and it has one entry where the run has two ranks: the lists are
        # equal and their entries are, so only their length tells.
        ("own_tokens_only", "[[8], [8]]"),
    ],
)
def test_token_counts_that_do_not_show_the_same_step_on_every_rank_fail_the_point(tmp_path, fault, tokens):
    victim = _victim(_plan_only(tmp_path, "prefill", dp_size=2))
    assert (victim.batch_size, victim.new_tokens) == (2, 4)  # eight computed tokens on each rank
    fake = _start(tmp_path, "prefill", dp_size=2, faults={(victim.benchmark_id, "meas", 0): fault})

    # The step number is the same on both ranks; only the token counts tell.
    rid = _rank_rid(victim.benchmark_id, "meas", 0, 0, 0)
    step = _line_of(fake, 0, rid)["forward_iter"]
    assert _line_of(fake, 1, _rank_rid(victim.benchmark_id, "meas", 0, 1, 0))["forward_iter"] == step
    _assert_point_failed(fake, victim, _TOKENS_DIFFER.format(of=1, step=step, tokens=tokens))
    _assert_not_staged_again(fake)


def test_a_rank_that_splits_the_extend_of_a_decode_run_fails_the_point(tmp_path):
    # The sequence of rank 1 is one a decode run may have: two extend batches and six decode batches with
    # exactly its requests. It took one extend batch more than rank 0.
    victim = _victim(_plan_only(tmp_path, "decode", dp_size=2))
    # FAKE: an engine whose prefill graphs cannot replay the step. Rank 0 then keeps its decode batches, and
    # each rank's own sequence is one a decode run may have. The ranks are held against each other.
    kept = _start(
        tmp_path / "kept",
        "decode",
        dp_size=2,
        faults={(victim.benchmark_id, "run", 0): "split_extend@1"},
        prefill_graphs_replay=False,
    )
    (run,) = kept.calls_of(victim.benchmark_id, "run")
    assert run.rank_modes == [["EXTEND"] + ["DECODE"] * 6, ["EXTEND", "EXTEND"] + ["DECODE"] * 6]
    _assert_point_failed(kept, victim, f"{_STEP}: the ranks ran the call as [7, 8] batches, one number per rank")
    _assert_not_staged_again(kept)


def test_a_decode_step_that_ran_as_an_extend_batch_is_never_kept_as_a_batch_of_the_point(tmp_path):
    victim = _victim(_plan_only(tmp_path, "decode", dp_size=2))
    fake = _start(tmp_path, "decode", dp_size=2, faults={(victim.benchmark_id, "run", 0): "decode_as_extend@0"})

    # FAKE: the first decode step of rank 0 ran as an extend batch, and so did that of rank 1, which the FAKE
    # engine turned into one because rank 0 extended in the step. The ranks are then in step with each other:
    # line for line the same mode, the same step number and the same token counts.
    (run,) = fake.calls_of(victim.benchmark_id, "run")
    assert run.rank_modes == [["EXTEND", "EXTEND"] + ["DECODE"] * 5] * 2
    lines = [_lines_of_call(fake, rank, run) for rank in range(2)]
    for own, other in zip(*lines, strict=True):
        assert (own["mode"], own["forward_iter"], own["global_num_tokens"]) == (
            other["mode"],
            other["forward_iter"],
            other["global_num_tokens"],
        )
        assert len(set(own["global_num_tokens"])) == 1
    for rank in range(2):
        _assert_decode_steps_as_extend(lines[rank][1:2], rank, run.prompts[rank::2])
    # At page size 1 a request of the run computes one token in its extend batch, as it does in such a line, so
    # the token counts of the line are those of a planned extend batch. The run still fails: it has five
    # lines of mode DECODE.
    assert grid.expected_extend_tokens(grid.DECODE, 0, victim.cached_tokens, 1) == 1
    reason = _DECODE_AS_EXTEND.format(rank=0, found="2 EXTEND, 5 DECODE", seqs=[lines[0][1]["seq"]])
    _assert_point_failed(fake, victim, reason)
    _assert_not_staged_again(fake)
    assert fake.clock.sleeps == 0


def test_a_step_number_that_is_not_a_count_is_a_recorder_fault(tmp_path):
    victim = _victim(_plan_only(tmp_path, "prefill", dp_size=2))
    fake = _start(tmp_path, "prefill", dp_size=2, faults={(victim.benchmark_id, "meas", 0): "no_forward_iter@1"})

    seq = _line_of(fake, 1, _rank_rid(victim.benchmark_id, "meas", 0, 1, 0))["seq"]
    _assert_point_failed(fake, victim, f"recorder_fault: {RANK_1_LOG}: batch seq {seq} has forward_iter=None")


@pytest.mark.parametrize("rank", [0, 1])
def test_an_idle_batch_inside_the_decode_run_of_a_rank_fails_the_point(tmp_path, rank):
    victim = _victim(_plan_only(tmp_path, "decode", dp_size=2))
    fake = _start(tmp_path, "decode", dp_size=2, faults={(victim.benchmark_id, "run", 0): f"idle_between@{rank}"})

    (run,) = fake.calls_of(victim.benchmark_id, "run")
    assert run.rank_modes[rank] == ["EXTEND", "DECODE", "DECODE", "IDLE", "DECODE", "DECODE", "DECODE", "DECODE"]
    seqs = [line["seq"] for line in fake.rank_lines(rank) if run.rids[rank] in line["rids"]]
    assert len(seqs) == 7 and seqs[-1] - seqs[0] == 7  # the idle line took one number between them
    _assert_point_failed(
        fake,
        victim,
        f"measured_decode_context_mismatch: rank {rank}: an IDLE batch without requests lies between the batches "
        f"of the run (seq {seqs})",
    )
    _assert_not_staged_again(fake)


# -- every check on every rank


# (mode, stage, FAKE fault, the reason a run with one rank gives for it, the rank the fault is on)
RANK_FAULTS = [
    ("prefill", "meas", "split_extend", _SPLIT, 1),
    ("prefill", "warm", "split_extend", _SPLIT, 1),
    ("prefill", "meas", "foreign_id", _FOREIGN_IN_EXTEND, 0),
    ("prefill", "meas", "wrong_extend_tokens", _TOKENS_REPORTED, 1),
    ("prefill", "meas", "wrong_extend_lens", _TOKENS_COUNTED, 1),
    ("decode", "run", "foreign_id", _CONTEXT + ": extend batch seq {seq} holds a foreign request", 1),
    ("decode", "run", "seven_decode", _SEQUENCE.replace("{found}", "1 EXTEND, 7 DECODE"), 1),
    ("decode", "run", "decode_missing_id", _DECODE_IDS.replace("{held}", "{b0}"), 1),
]


@pytest.mark.parametrize(
    ("mode", "stage", "fault", "reason", "rank"),
    RANK_FAULTS,
    ids=lambda value: str(value) if not isinstance(value, str) or " " not in value else None,
)
def test_a_check_that_fails_on_one_rank_fails_the_point_and_names_the_rank(tmp_path, mode, stage, fault, reason, rank):
    victim = _victim(_plan_only(tmp_path, mode, dp_size=2))
    fake = _start(tmp_path, mode, dp_size=2, faults={(victim.benchmark_id, stage, 0): f"{fault}@{rank}"})

    # The reason is the one a run with one rank gives, with the rank in front of its detail and the numbers of
    # that rank's log.
    (call,) = fake.calls_of(victim.benchmark_id, stage)
    mine = [line for line in fake.rank_lines(rank) if set(line["rids"]) & set(call.rids[rank::2])]
    seq = mine[0]["seq"]
    batch = victim.batch_size
    code, _, detail = reason.partition(": ")
    detail = detail.format(
        seq=seq,
        seq4=seq + 4,
        b=batch,
        b0=batch - 1,
        b1=batch + 1,
        bn=batch * victim.new_tokens,
        bn1=batch * victim.new_tokens + 1,
    )
    _assert_point_failed(fake, victim, f"{code}: rank {rank}: {detail}")
    _assert_not_staged_again(fake)


@pytest.mark.parametrize(
    ("mode", "stage", "fault", "reason"),
    [
        ("prefill", "meas", "copy_done_false", "batch seq {seq} was stamped without a copy event (copy_done=False)"),
        (
            "prefill",
            "meas",
            "no_lens",
            "extend batch seq {seq} has no usable prefix_lens, extend_lens and extend_num_tokens",
        ),
        (
            "decode",
            "run",
            "decode_copy_done_false",
            "batch seq {seq4} was stamped without a copy event (copy_done=False)",
        ),
        ("decode", "run", "fault_line", "process_batch_result: FAKE recorder fault"),
    ],
)
def test_a_recorder_fault_on_rank_1_names_the_log_of_rank_1(tmp_path, mode, stage, fault, reason):
    victim = _victim(_plan_only(tmp_path, mode, dp_size=2))
    fake = _start(tmp_path, mode, dp_size=2, faults={(victim.benchmark_id, stage, 0): f"{fault}@1"})

    seq = _line_of(fake, 1, _rank_rid(victim.benchmark_id, stage, 0, 1, 0))["seq"]
    _assert_point_failed(fake, victim, f"recorder_fault: {RANK_1_LOG}: " + reason.format(seq=seq, seq4=seq + 4))


@pytest.mark.parametrize(
    ("mode", "stage", "reason"),
    [("decode", "run", "measured_decode_context_mismatch"), ("prefill", "meas", "measured_prefill_tokens_mismatch")],
)
def test_a_late_batch_on_rank_1_fails_the_run_and_names_the_rank(tmp_path, mode, stage, reason):
    victim = _victim(_plan_only(tmp_path, mode, dp_size=2))
    fake = _start(tmp_path, mode, dp_size=2, faults={(victim.benchmark_id, stage, 0): "late_decode@1"})

    assert fake.run.status == "failed"
    assert f"{reason}: rank 1: a DECODE batch (seq " in fake.run.reason
    assert fake.run.reason.endswith(") holds requests of a call that had already been checked")
    _assert_rejected(fake, "failed")


# -- a wrong cached count on one rank


@pytest.mark.parametrize("ranks", [(1,), (0,), None], ids=["rank-1", "rank-0", "both-ranks"])
@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_a_cache_miss_on_one_rank_stages_the_point_again_on_all_ranks(tmp_path, mode, ranks):
    victim = _victim(_plan_only(tmp_path, mode, dp_size=2))
    fake = _start(tmp_path, mode, dp_size=2, evictions={victim.benchmark_id: 1}, eviction_ranks=ranks)

    # Every stage of the point was sent twice, each time to both ranks, the seed included.
    stages = ["seed", "warm", "meas"] if mode == "prefill" else ["seed", "run"]
    calls = fake.calls_of(victim.benchmark_id)
    assert [call.key for call in calls] == [
        (victim.benchmark_id, stage, attempt) for attempt in (0, 1) for stage in stages
    ]
    assert all(call.ranks == [0, 1] * victim.batch_size for call in calls)
    assert {parts["benchmark_id"] for parts in fake.sent() if parts["attempt"] and parts["benchmark_id"]} == {
        victim.benchmark_id
    }
    on_disk = _assert_ranks_accepted(fake)
    assert on_disk[0]["producer"]["restaged_points"] == [victim.benchmark_id]


def _decode_point_above_the_chunk_budget(plan: _FakeRun) -> grid.Point:
    """A planned decode point whose prompts, once they are computed whole, do not fit one extend batch of a rank."""
    chunk = plan.limits.chunked_prefill_size
    return next(
        point
        for point in plan.built.points[2:]
        if point.batch_size >= 2 and point.batch_size * (point.cached_tokens - grid.DECODE_PROMPT_LEAD) > chunk
    )


def test_a_cache_miss_on_one_rank_in_a_decode_run_above_the_chunk_budget_is_staged_again(tmp_path):
    victim = _decode_point_above_the_chunk_budget(_plan_only(tmp_path, "decode", dp_size=2))
    fake = _start(tmp_path, "decode", dp_size=2, evictions={victim.benchmark_id: 1}, eviction_ranks=(1,))

    # First attempt. The rank that lost its cache computed its prompts again, in several extend batches. The
    # rank that hit its cache extended in the first step only and decoded from the second step on, beside the
    # further extend batches of the other rank. The FAKE engine wrote those decode steps as SGLang does, as
    # extend batches of one token per request, so the log of that rank holds fewer than six DECODE lines.
    first, second = fake.calls_of(victim.benchmark_id, "run")
    assert (first.key[2], second.key[2]) == (0, 1)

    # The driver saw the missed cache in the responses and staged the point again on both ranks, without
    # waiting for DECODE lines that were never written. The second attempt ran as planned and is kept.
    assert [call.key[1:] for call in fake.calls_of(victim.benchmark_id)] == [
        ("seed", 0),
        ("run", 0),
        ("seed", 1),
        ("run", 1),
    ]
    assert second.rank_modes == [["EXTEND"] + ["DECODE"] * 6] * 2
    assert fake.clock.sleeps == 0
    on_disk = _assert_ranks_accepted(fake)
    assert on_disk[0]["producer"]["restaged_points"] == [victim.benchmark_id]


# -- self-test


@pytest.mark.parametrize(
    ("call", "fault", "reason", "passed"),
    [
        (FREE_BATCH_SHAPE, "late@1", "batch_shape: " + _STEPS_DIFFER.format(of=1, step=1, step1=2), []),
        (
            FREE_BATCH_SHAPE,
            "uneven_tokens@1",
            "batch_shape: " + _TOKENS_DIFFER.format(of=1, step=1, tokens="[[32, 33], [32, 33]]"),
            [],
        ),
        (
            # The cache of rank 1 alone lost the seeded prefix.
            FREE_ALIGNED_SHOTS,
            "evict@1",
            "aligned_prefix: 2 of 4 request(s) report another cached count than the planned 16: "
            + _rank_rid(0, "free", 2, 1, 0)
            + " reports 0",
            ["batch_shape"],
        ),
        (
            # Rank 1 was released one step late. Rank 0 then ran its first decode step beside the extend batch
            # of rank 1, and the FAKE engine, as SGLang, wrote that step as an extend batch: line 7 of the log
            # of rank 0, after the five one-token calls and the extend batch of the run.
            FREE_SEVEN_TOKEN_RUN,
            "late@1",
            "seven_token_run: " + _DECODE_AS_EXTEND.format(rank=0, found="2 EXTEND, 5 DECODE", seqs=[7]),
            _ALL_BUT_LAST,
        ),
        (
            FREE_SEVEN_TOKEN_RUN,
            "misplaced",
            "seven_token_run: " + _PLACEMENT + ": " + _rank_rid(0, "free", 5, 0, 0) + ": the engine reports dp_rank 1, "
            "the request was planned for rank 0",
            _ALL_BUT_LAST,
        ),
    ],
    ids=lambda value: value if isinstance(value, str) and " " not in value else None,
)
def test_a_self_test_check_that_fails_on_one_rank_ends_the_run_before_the_sweep(tmp_path, call, fault, reason, passed):
    fake = _start(tmp_path, "prefill", dp_size=2, faults={(0, "free", call): fault})

    assert (fake.run.status, fake.run.reason) == ("failed", f"self_test_failed: {reason}")
    assert fake.sent_benchmark_ids() == {0}
    assert [made.key[2] for made in fake.engine.calls] == list(range(call + 1))
    on_disk = _assert_rejected(fake, "failed")
    assert on_disk["producer"]["self_test"] == passed


# -- headers


def _edit_rank(path: str, value, *, rank: int):
    return _edit(path, value, process=rank)


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        # No leader for rank 1: its one scheduler process says it holds no request socket.
        (
            _edit_rank("leader", False, rank=1),
            "recorder_header_invalid: no leader header for attention-DP rank(s) [1], expected exactly one for each "
            "of [0, 1]: the leaders are ['sched-4100.jsonl']",
        ),
        # Two leaders for rank 0, and so none for rank 1.
        (
            _edit_rank("ranks.attn_dp_rank", 0, rank=1),
            "recorder_header_invalid: 2 leader headers for attention-DP rank 0, expected exactly one: "
            "['sched-4100.jsonl', 'sched-4101.jsonl']",
        ),
        (
            _edit_rank("ranks.attn_dp_rank", 2, rank=1),
            "recorder_header_invalid: sched-4101.jsonl: attn_dp_rank is 2, expected one of [0, 1]",
        ),
        # The layout a scheduler reports is not the one the run was set up for.
        (
            _edit("ranks.attn_dp_enabled", False),
            "rejected: " + _OTHER_LAYOUT.format(size="2", enabled="false", planned=2),
        ),
        # Without SGLang's input blocker the ranks would not be released together.
        (
            _edit("checks.input_blocker", False),
            "rejected: SGLANG_ENABLE_COLOCATED_BATCH_GEN must be set under DP attention (input_blocker=false); "
            "the launcher sets it before it builds the engine",
        ),
        (
            _edit_rank("checks.input_blocker", None, rank=1),
            "rejected: SGLANG_ENABLE_COLOCATED_BATCH_GEN must be set under DP attention (input_blocker=null); "
            "the launcher sets it before it builds the engine",
        ),
    ],
)
def test_a_header_that_does_not_fit_a_run_with_two_ranks_fails_the_run(tmp_path, edit, reason):
    fake = _start(tmp_path, dp_size=2, edit_header=edit)
    _assert_failed_before_the_grid(fake, reason)


def test_a_scheduler_process_without_a_request_socket_is_refused_under_dp_attention(tmp_path):
    # FAKE: a third scheduler process, the second TP rank inside attention-DP rank 1. Attention TP above 1
    # inside a rank is not supported.
    fake = _start(
        tmp_path, dp_size=2, expected_scheduler_processes=3, edit_header=_edit("ranks.tp_size", 3), execute=False
    )
    header = _fake_header(4102, 1, fake.limits, 3, dp_size=2, dp_rank=1)
    (fake.recorder_dir / "sched-4102.jsonl").write_text(json.dumps(header) + "\n", encoding="utf-8")
    fake.execute()

    _assert_failed_before_the_grid(
        fake,
        "rejected: attention TP above 1 inside an attention-DP rank is not supported: ['sched-4102.jsonl'] hold "
        "no request socket",
    )


# -- runs that do not complete


def test_a_run_with_two_ranks_that_is_cut_short_by_the_deadline_is_partial_in_every_rank_file(tmp_path):
    fake = _start(tmp_path, "decode", dp_size=2, faults={(3, "run", 0): "deadline"})

    assert (fake.run.status, fake.run.reason) == ("partial", "timeout")
    on_disk = _assert_rejected(fake, "partial")
    assert on_disk["coverage"]["completed_points"] == 3
    merged = json.loads((fake.pod / "benchmark_merged.json").read_text(encoding="utf-8"))
    assert (merged["status"], merged["dp"]["source_ranks"]) == ("partial", [0, 1])


def test_a_failure_recorded_before_execute_is_written_for_every_rank(tmp_path):
    # The launcher records a refusal before the engine is built, when it already knows the number of ranks.
    fake = _start(tmp_path, dp_size=2, execute=False)
    fake.run.record_failure("rejected: FAKE refusal before the engine was built")

    on_disk = _assert_rejected(fake, "failed")
    assert (on_disk["error"], on_disk["grid_digest"]) == (
        "rejected: FAKE refusal before the engine was built",
        grid.grid_digest([]),
    )
    assert on_disk["producer"]["staging"] == "balanced_batch_call"
    assert sorted(path.name for path in fake.pod.glob("benchmark*")) == ["benchmark.json", "benchmark_dp1.json"]


# ---------------------------------------------------------------------------------------------------------------
# What the KV pool takes from the generated prefill block
# ---------------------------------------------------------------------------------------------------------------


def test_a_run_tells_what_the_kv_pool_takes_from_a_generated_prefill_block_and_nothing_otherwise(tmp_path):
    for mode in ("prefill", "agg"):
        fake = _start(tmp_path / mode, mode, caps=grid.SampleCaps())
        assert fake.run.status == "complete"
        cut = grid.prefill_pool_cut(fake.limits, grid.SampleCaps())
        assert cut is not None and fake.run.prefill_pool_cut() == cut
    # Before the grid exists.
    assert _plan_only(tmp_path, "prefill", caps=grid.SampleCaps()).run.prefill_pool_cut() is None
    # The same limits with caps that plan no batch above two.
    small = _start(tmp_path / "small", "prefill")
    assert small.run.status == "complete" and small.run.prefill_pool_cut() is None
    # The default caps with the smallest pool that holds the whole grid.
    roomy = _start(tmp_path / "roomy", "prefill", limits=_limits(1, max_total_num_tokens=1709), caps=grid.SampleCaps())
    assert roomy.run.status == "complete" and roomy.run.prefill_pool_cut() is None
    # A decode run: its block is not looked at.
    decode = _start(tmp_path / "decode", "decode", caps=grid.SampleCaps())
    assert decode.run.status == "complete" and decode.run.prefill_pool_cut() is None
    # A run with a points manifest, on the limits and caps of the generated runs above.
    points = [{"batch_size": 4, "total_prefill_tokens": 16, "total_kv_read_tokens": 64}]
    manifest = grid.parse_points_manifest(json.dumps({"schema_version": 3, "prefill": points, "decode": []}))
    listed = _start(tmp_path / "manifest", "prefill", manifest=manifest, caps=grid.SampleCaps())
    assert listed.run.status == "complete" and listed.built.source == "manifest"
    assert listed.run.prefill_pool_cut() is None


def _execute_with_hook(fake: _FakeRun, hook) -> None:
    low, high = fake.tokens
    fake.run.execute(fake.engine, token_low=low, token_high=high, on_grid=hook)


def test_the_grid_hook_is_called_once_when_the_grid_is_planned_and_before_any_request(tmp_path):
    fake = _start(tmp_path, "prefill", caps=grid.SampleCaps(), execute=False)
    seen = []

    _execute_with_hook(fake, lambda: seen.append((fake.run.prefill_pool_cut(), len(fake.engine.calls))))

    # The answer was there when the hook ran, and no request had been sent.
    cut = grid.prefill_pool_cut(fake.limits, grid.SampleCaps())
    assert cut is not None and seen == [(cut, 0)]
    assert fake.run.status == "complete" and len(fake.engine.calls) > 0

    # The hook is also called for a grid that misses a phase.
    missing = _start(tmp_path / "missing", "prefill", 16, limits=grid.Limits(16, 100, 32, 8, 512, 256), execute=False)
    seen = []
    _execute_with_hook(missing, lambda: seen.append(missing.run.prefill_pool_cut()))
    assert missing.run.status == "failed" and len(seen) == 1 and seen[0].dropped_pairs == seen[0].pairs
