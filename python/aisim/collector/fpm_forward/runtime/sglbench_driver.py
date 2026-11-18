# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# The batch-sequence checks follow SGLang's scheduler at v0.5.22 and v0.5.23; see THIRD_PARTY_NOTICES.md.
# Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d
# Source: https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90
# The run lifecycle, the point procedures and nine failure reason strings are re-implemented from ai-dynamo/dynamo
# (Apache-2.0, https://github.com/ai-dynamo/dynamo), components/src/dynamo/vllm/instrumented_scheduler.py at
# b20fe6fe94021b854597965c7305931175e8f482. See THIRD_PARTY_NOTICES.md.

"""Driver of the SGLang self-benchmark: one run, from the recorder headers to the result files.

This module runs in the launcher's process inside the SGLang pod, where AISim is not installed. It uses the
standard library and the sibling modules ``sglbench_grid``, ``sglbench_content`` and ``sglbench_artifact``. It
imports no SGLang and reads no environment variable. The engine is an object the caller hands over: anything
with ``generate(prompts, rids, max_new_tokens) -> list[dict]``. Every setting is an argument. For a run whose
prompts are dataset text the caller also hands over the text pool (``sglbench_content.TextPool``) and the
record of the dataset it was made from; the driver reads no file and tokenizes nothing.

A run (``BenchmarkRun.execute``), in this order:

1. wait for one recorder header per scheduler process and check each one;
2. refuse engine settings the feasibility rules do not cover (as each scheduler process reports them);
3. build the limits record from the leader's header, and the grid from it;
4. check the two request caps against the largest batch of the grid;
5. compare the first tokens the run needs with the token range and, in a text run, the tokens of the text
   pool with the longest prompt of the grid; then build the content plan;
6. build one empty artifact, so that a setting that is not JSON fails before any request;
7. run the self-test, then the global warm-up;
8. stage, check and record every point in grid order. Before each point a recorder fault fails the run and
   the soft deadline ends it as partial;
9. look for recorder faults once more.

The first failure is recorded and ends the run; ``execute`` does not raise for it. ``BenchmarkRun.write``
writes the rank file and, unless the run failed, the merged file.

A stage is one engine call: ``b`` prompts, ``b`` request ids and one output-token count. The batches the
scheduler formed for the call are read from the recorder logs by request id. The recorder is a subclass of
SGLang's ``Scheduler`` that writes one line per batch in every scheduler process; this module only reads its
files. The process that receives the requests is the leader, and all times come from its log.

Failure reasons. Five are the strings of Dynamo's vLLM self-benchmark, with the meaning they have here:

    measured_batch_size_mismatch       a timed batch does not hold exactly the planned requests
    real_seed_cache_validation_failed  a request reported another cached count than planned, twice
    measured_prefill_tokens_mismatch   a prompt count differs, a request was chunked, or the computed tokens differ
    measured_decode_context_mismatch   a decode run is not extend batches and then six decode batches
    no_fpm_before_deadline             a log did not hold the batches of a stage in time

Four more are the verdicts of the content probe of Dynamo's vLLM self-benchmark (``_kvwarm_probe_content``). There
a verdict switches the real-text warm-up off and the run goes on with synthetic content; here each one fails
the run, because a text run has no other content to fall back to:

    dataset_unavailable                the dataset file cannot be had: no file, a wrong hash, a failed download
    dataset_empty                      the dataset gives no text, or no token id of its texts is one the model
                                       counts
    tokenizer_unavailable              the served tokenizer could not turn the texts into token ids
    content_too_shallow                the text pool does not hold more tokens than the longest prompt needs

The driver itself gives the last one only. The launcher gives the other three through ``record_failure``,
because it is the module that reads the dataset and calls the tokenizer.

The others are this module's own: ``request_not_completed``, ``engine_call_failed``, ``recorder_fault``,
``recorder_header_invalid``, ``self_test_failed``, ``rejected``, ``token_range_too_small`` and
``internal_error``. The launcher adds ``stopped`` through ``record_failure``. A reason is followed by
``: <detail>`` where there is one.

Checked against SGLang v0.5.22 (baa9befa381511cd4d56f83980951e5a45fffc6d) and
v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90) of ``sgl-project/sglang`` (Apache-2.0), by reading these
files under ``python/sglang``::

    srt/managers/scheduler.py                   Scheduler.run_batch (batch.forward_iter, batch.launch_ts),
                                                Scheduler.process_batch_result, Scheduler.event_loop_overlap,
                                                Scheduler._reqs_finishing_inflight,
                                                Scheduler.get_next_batch_to_run,
                                                Scheduler.handle_batch_generate_request,
                                                Scheduler.init_model_worker, Scheduler.init_chunked_prefill,
                                                Scheduler.init_schedule_policy,
                                                Scheduler.init_watch_dog_memory_saver_input_blocker,
                                                Scheduler._abort_on_queued_limit
    srt/managers/schedule_batch.py              ScheduleBatch.prepare_for_extend (extend_num_tokens,
                                                prefix_lens, extend_lens), ScheduleBatch.prepare_for_decode,
                                                ScheduleBatch.copy, FINISH_LENGTH, Req.retraction_count
    srt/model_executor/forward_batch_info.py    ForwardMode
    srt/managers/tokenizer_manager.py           TokenizerManager._send_batch_request,
                                                TokenizerManager._handle_batch_output (the ``meta_info`` keys)
    srt/managers/scheduler_components/
        request_receiver.py                     SchedulerRequestReceiver.recv_requests
        recv_skipper.py                         SchedulerRecvSkipper.maybe_create
        metrics_reporter.py                     SchedulerMetricsReporter._build_scheduled_request_metrics
    srt/managers/schedule_policy.py             _IS_HIP
    srt/runtime_context.py                      get_parallel

Batch sequences that running engines showed are kept in
``benchmarks/evidence/collector/sglbench-engine-probe-20261113``: one header per scheduler process (check_1),
one call as one extend batch (check_2), the cached counts (check_3 and check_5), a seven-token call as one
extend batch and six decode batches (check_4), and a call above the chunk budget as several extend batches
(check_9). Overlap scheduling was on in every one of those runs.

The lifecycle and the point procedures follow Dynamo's vLLM self-benchmark, in the file and at the revision the
header of this file names: the global warm-up of one 256-token prompt (``_bench_step_warmup``), seed, warm shot
and measured shot of a prefill point with one re-stage after a missed cache hit (``_bench_realseed_stage_point``,
``_bench_realseed_pending_step``), the soft deadline looked at between two points
(``_bench_stop_at_timeout_boundary``), the comparison of the scheduled counters with the point
(``_bench_fpm_validation_failure``) and the test that the dataset text is deep enough for the longest prompt
(``_kvwarm_probe_content``). One difference: Dynamo always runs its first point, because that
function does nothing while no point has completed; this driver also looks at the deadline before the first
point, because its clock starts with the launcher and not with the first benchmark step.

No upstream function or block of source is copied. What follows SGLang and Dynamo is named in the header of this
file and recorded in THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import math
import os
import time
from collections.abc import Callable, Iterable

import sglbench_artifact as artifact
import sglbench_content as content
import sglbench_grid as grid

# ---------------------------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------------------------

# What the recorder writes, repeated here because this module may import only the three sibling modules named
# in its docstring. tests/unit/collector/test_fpm_sglbench_driver.py compares each value with the recorder's
# and the version list with the one of sglbench_compat.py.
RECORD_FORMAT = 1
RECORDER_CLASS_NAME = "SglbenchRecorderScheduler"
LOG_FILE_PREFIX = "sched-"
LOG_FILE_SUFFIX = ".jsonl"
TESTED_SGLANG_VERSIONS = ("0.5.22", "0.5.23")

# The global warm-up is one prompt of 256 tokens, as in Dynamo's vLLM self-benchmark (_bench_step_warmup).
WARMUP_PROMPT_TOKENS = 256

# This version has one attention-DP rank: DP attention is refused before a run starts.
RANK = 0
DP_SIZE = 1

# Request ids carry the first characters of the run id only, which keeps the recorder lines short. The
# artifact carries the whole run id.
RUN_TAG_CHARACTERS = 8

# Seconds between two reads of the recorder logs while a header or a batch line is awaited.
POLL_SECONDS = 0.02

# SGLang: srt/model_executor/forward_batch_info.py ``ForwardMode``; the recorder writes ``forward_mode.name``.
MODE_EXTEND = "EXTEND"
MODE_DECODE = "DECODE"
MODE_IDLE = "IDLE"

# Reason strings of Dynamo's vLLM self-benchmark (see the module docstring for what each means here).
REASON_BATCH_SIZE = "measured_batch_size_mismatch"
REASON_CACHE = "real_seed_cache_validation_failed"
REASON_PREFILL_TOKENS = "measured_prefill_tokens_mismatch"
REASON_DECODE_CONTEXT = "measured_decode_context_mismatch"
REASON_NO_BATCHES = "no_fpm_before_deadline"
# The four verdicts of Dynamo's content probe (_kvwarm_probe_content), which fail a text run here. The driver
# gives the last one; the launcher gives the other three and takes the strings from here.
REASON_DATASET_UNAVAILABLE = "dataset_unavailable"
REASON_DATASET_EMPTY = "dataset_empty"
REASON_TOKENIZER_UNAVAILABLE = "tokenizer_unavailable"
REASON_CONTENT_TOO_SHALLOW = "content_too_shallow"
# This module's own reason strings.
REASON_NOT_COMPLETED = "request_not_completed"
REASON_ENGINE_CALL = "engine_call_failed"
REASON_RECORDER_FAULT = "recorder_fault"
REASON_HEADER = "recorder_header_invalid"
REASON_SELF_TEST = "self_test_failed"
REASON_REJECTED = "rejected"
REASON_TOKEN_RANGE = "token_range_too_small"
REASON_INTERNAL = "internal_error"

# The self-test sends pairs of requests.
SELF_TEST_REQUESTS = 2
SELF_TEST_CHECKS = ("batch_shape", "aligned_prefix", "unaligned_prompt", "seven_token_run")
# Free prompts of the self-test: one pair for the batch-shape check and one for the seven-token run.
_SELF_TEST_FREE_PROMPTS = 2 * SELF_TEST_REQUESTS
# The unaligned prompt of the self-test is this many tokens longer than two cached units, which are whole pages.
# It then ends inside a page at every page size but 1 and 7. At those two it ends on a page boundary, and the
# check still holds the engine to the count it must report: the prompt less its last page (at page size 1 its
# last token).
_UNALIGNED_EXTRA_TOKENS = 7

# Seeds and prefill shots ask for one output token. Each then stores its prompt and nothing after it, which
# the planned cached counts assume (sglbench_content.py, "Conditions on the caller").
_ONE_OUTPUT_TOKEN = grid.PREFILL_OUTPUT_TOKENS

# The keys of the dataset record that go into the ``kvwarm`` block of a decode or agg artifact of a text run.
# They are the four keys Dynamo's vLLM self-benchmark writes there (_kvwarm_load_texts); the launcher's record
# holds two more, which stay in the ``producer`` block.
_KVWARM_DATASET_KEYS = ("path", "sha256", "collection_pool", "conversations")

# How the message of the content plan begins when a text chain has used every conversation of the pool
# (sglbench_content.py). tests/unit/collector/test_fpm_sglbench_driver.py compares it with what the plan raises.
_POOL_EXHAUSTED = "text pool exhausted"

# The response keys of one request, as the engine object returns them.
_RESPONSE_INT_KEYS = ("prompt_tokens", "cached_tokens", "completion_tokens", "num_retractions")

# The six limits of a recorder header. Each is an attribute of SGLang's Scheduler (srt/managers/scheduler.py):
# ``page_size`` and ``chunked_prefill_size`` (``None`` when chunking is off, ``Scheduler.init_chunked_prefill``),
# and the four values ``Scheduler.init_model_worker`` unpacks from ``TpModelWorker.get_worker_info``
# (srt/managers/tp_worker.py). ``max_running_requests`` is the per-rank cap after SGLang's own clamp.
_LIMIT_NAMES = (
    "page_size",
    "max_req_input_len",
    "max_total_num_tokens",
    "max_running_requests",
    "max_prefill_tokens",
    "chunked_prefill_size",
)
# Captured graph sizes: ``ModelRunner.decode_cuda_graph_runner.capture_bs`` and
# ``ModelRunner.prefill_cuda_graph_runner.capture_num_tokens`` (srt/model_executor/model_runner.py), as the
# recorder reads them.
_GRAPH_NAMES = ("decode_graph_sizes", "prefill_graph_sizes")

# Header values that every scheduler process must report alike.
_AGREED_VALUES = (
    ("sglang_version",),
    *(("limits", name) for name in _LIMIT_NAMES),
    *((name,) for name in _GRAPH_NAMES),
    ("ranks", "tp_size"),
)

# Scheduler facts of the header's ``checks`` block that must be the boolean False, with the text of the
# rejection. Each is a plain attribute of SGLang's Scheduler (srt/managers/scheduler.py), read at both tags:
# ``disable_radix_cache``, ``is_hybrid_swa``, ``is_hybrid_ssm`` (set in ``Scheduler.__init__`` from the cache
# build), ``enable_priority_scheduling`` (``Scheduler.__init__``), ``is_mixed_chunk``
# (``Scheduler.init_chunked_prefill``), ``prefill_delayer`` (``Scheduler.init_schedule_policy``),
# ``min_free_slots_delayer`` (``Scheduler.init_model_worker``), ``recv_skipper`` and ``input_blocker``
# (``Scheduler.init_watch_dog_memory_saver_input_blocker``). The recorder writes the last four as
# ``attribute is not None``. ``is_hip`` is srt/managers/schedule_policy.py ``_IS_HIP``.
# Each of them changes how a batch is admitted, what is cached or what the timed step holds, and the
# feasibility rules of sglbench_grid.py assume all of them off.
_CHECKS_FALSE = (
    ("disable_radix_cache", "the radix cache is off ({name}={v}); SGLang may have disabled it for this model"),
    ("is_hybrid_swa", "hybrid SWA or state-space models are not supported ({name}={v})"),
    ("is_hybrid_ssm", "hybrid SWA or state-space models are not supported ({name}={v})"),
    ("is_mixed_chunk", "--enable-mixed-chunk is not supported ({name}={v})"),
    ("enable_priority_scheduling", "schedule policy must be fcfs without priority scheduling ({name}={v})"),
    ("prefill_delayer", "the prefill delayer is not supported (also check SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE)"),
    ("min_free_slots_delayer", "min_free_slots_delay must not be set (got {name}={v})"),
    # A receive skipper exists only when scheduler_recv_interval is above 1
    # (srt/managers/scheduler_components/recv_skipper.py ``SchedulerRecvSkipper.maybe_create``). Requests of
    # one call could then enter the scheduler in different steps.
    ("recv_skipper", "scheduler_recv_interval must be 1 (got {name}={v})"),
    ("input_blocker", "SGLANG_ENABLE_COLOCATED_BATCH_GEN must be unset without DP attention ({name}={v})"),
    ("is_hip", "only NVIDIA CUDA is supported ({name}={v}); SGLang has AMD-only admission rules"),
)

# Which reason a difference in one of the five scheduled counters stands for, in the order they are compared.
# ``None`` is a wrong cached count: the point is staged once more. The reasons are those Dynamo's vLLM
# self-benchmark gives for the same counters (_bench_fpm_validation_failure); it has a string of its own for
# the cached prefill tokens, where this driver re-stages.
_COUNTER_REASONS = (
    ("num_prefill_requests", REASON_BATCH_SIZE),
    ("num_decode_requests", REASON_BATCH_SIZE),
    ("sum_prefill_kv_tokens", None),
    ("sum_prefill_tokens", REASON_PREFILL_TOKENS),
    ("sum_decode_kv_tokens", REASON_DECODE_CONTEXT),
)


class _Failure(Exception):
    """A failure of the run. ``reason`` is the text the artifact carries."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _CacheMiss(Exception):
    """A request found another number of cached tokens than planned. A point is then staged once more."""


def _shown(value: object) -> str:
    """A header value for a message: JSON spelling for true, false and null, plain text otherwise."""
    if value is None or isinstance(value, bool):
        return json.dumps(value)
    return str(value)


def _is_int(value: object) -> bool:
    """True for an ``int`` that is not a ``bool``."""
    return type(value) is int


def _is_int_list(value: object, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(_is_int(item) for item in value)


# ---------------------------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunSettings:
    """What the launcher decides about one run. The driver changes none of it.

    ``config`` and ``producer`` are read when the artifact is built, not when the run is created. A fact the
    launcher adds to its ``producer`` dict after ``execute`` (an error of the engine shutdown) is therefore
    written too.
    """

    # grid.PREFILL, grid.DECODE or grid.AGG.
    mode: str
    # Sample caps of the generated grid; ignored when a manifest is given.
    caps: grid.SampleCaps
    # Result of grid.parse_points_manifest, or None for the generated grid.
    manifest: dict | None
    # Output tokens of the global warm-up; 0 skips it.
    warmup_iterations: int
    # Soft deadline in seconds, counted from ``started_at`` of the run.
    timeout_seconds: float
    # Path of the rank 0 result file. Its directory must exist.
    output_path: str
    # New for every launch. Request ids carry its first RUN_TAG_CHARACTERS characters.
    run_id: str
    # Seed of the content plan, as a string.
    content_seed: str
    # Written into every forward-pass record as ``worker_id``.
    worker_id: str
    # How many scheduler processes must have written a recorder header (the TP size in this version).
    expected_scheduler_processes: int
    # Run-invariant settings, written as ``config``. AISim compares them between two launches of a cell.
    config: dict
    # Facts of the launcher for the ``producer`` block. The driver adds its own keys beside them.
    producer: dict
    # SGLang's resolved ``prefill_max_requests`` and ``max_queued_requests``; None means unset.
    prefill_max_requests: int | None
    max_queued_requests: int | None
    # How long the logs of a stage may lag behind its engine call.
    log_wait_seconds: float = 30.0
    # How long the recorder headers may take to appear.
    header_wait_seconds: float = 60.0
    # Where the prompts of the run come from, as far as the launcher knows it when it makes the run:
    # ``sglbench_content.CONTENT_SEEDED_IDS`` or ``CONTENT_TEXT_CHAINS``. Only an artifact that is written
    # before ``execute`` was called reads it, so that a text run whose dataset could not be had still says
    # what it was started for. ``execute`` itself goes by the text pool it is given.
    content_source: str = content.CONTENT_SEEDED_IDS


def _require_setting_int(name: str, value: object, minimum: int) -> None:
    if not _is_int(value):
        raise TypeError(f"settings.{name} must be an int, got {type(value).__name__}: {value!r}")
    if value < minimum:
        raise ValueError(f"settings.{name} must be >= {minimum}, got {value}")


def _require_setting_seconds(name: str, value: object, *, positive: bool) -> None:
    if type(value) not in (int, float):
        raise TypeError(f"settings.{name} must be a number of seconds, got {type(value).__name__}: {value!r}")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"settings.{name} must be finite and {'> 0' if positive else '>= 0'}, got {value!r}")


# ---------------------------------------------------------------------------------------------------------------
# Token range
# ---------------------------------------------------------------------------------------------------------------


def ordinary_token_range(size: int, excluded: Iterable[int]) -> tuple[int, int]:
    """The longest run of token ids in ``[0, size)`` that holds no excluded id, as ``(low, high)``.

    ``high`` is exclusive, as ``sglbench_content.ContentPlan`` takes it. ``excluded`` holds the special and
    added token ids of the served tokenizer; an id outside ``[0, size)`` excludes nothing. Of two runs of the
    same length the lower one is returned. Nothing is converted: ``size`` and every id must be an ``int`` and
    not a ``bool``. ``ValueError`` when ``size`` is below 1 or every id is excluded.

    Whether the range is large enough for a run is decided by ``BenchmarkRun.execute``, which knows the grid.
    """
    if not _is_int(size):
        raise TypeError(f"size must be an int, got {type(size).__name__}: {size!r}")
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    inside = set()
    for token in excluded:
        if not _is_int(token):
            raise TypeError(f"an excluded token id must be an int, got {type(token).__name__}: {token!r}")
        if 0 <= token < size:
            inside.add(token)
    best_low, best_high = 0, 0
    low = 0
    # ``size`` closes the last run. A strict comparison keeps the first of two runs of equal length.
    for token in [*sorted(inside), size]:
        if token - low > best_high - best_low:
            best_low, best_high = low, token
        low = token + 1
    if best_high == best_low:
        raise ValueError(f"every token id in [0, {size}) is excluded: there is no ordinary token")
    return best_low, best_high


# ---------------------------------------------------------------------------------------------------------------
# Recorder logs
# ---------------------------------------------------------------------------------------------------------------


class RecorderLogs:
    """Incremental reader of one recorder directory (record format 1).

    The directory holds one file per scheduler process, ``sched-<pid>.jsonl``: one JSON object per line, the
    header first, then one line per batch, and a fault line for every recorder step that raised. ``poll``
    reads what was appended since the last call. A line without its newline is kept back until the rest
    arrives.

    After a poll:

    - ``headers`` maps a file name to its header line;
    - ``batches(name)`` is the list of the batch lines of one file in file order, without the lines of mode
      ``IDLE`` that hold no request and without the lines read before the last ``forget()``;
    - ``faults`` lists the fault lines as ``{"file", "where", "error"}``.

    Anything that is not format 1 is added to ``faults`` with ``where`` set to ``"reader"`` and never repaired:
    a line that is not a JSON object, an unknown line type, a header that is not the first line, a batch line
    before the header, a batch line without ``mode`` and ``rids``, and a ``seq`` that does not count the batch
    lines of its file from 1. Such a line is not kept. The caller fails the run on any entry.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = os.fspath(directory)
        self.headers: dict[str, dict] = {}
        self.faults: list[dict] = []
        self._batches: dict[str, list[dict]] = {}
        self._offsets: dict[str, int] = {}
        self._partial: dict[str, bytes] = {}
        self._lines: dict[str, int] = {}
        self._batch_lines: dict[str, int] = {}

    def poll(self) -> None:
        """Read the new complete lines of every log file. A directory that does not exist yet holds no file."""
        try:
            names = sorted(os.listdir(self.directory))
        except FileNotFoundError:
            return
        for name in names:
            if name.startswith(LOG_FILE_PREFIX) and name.endswith(LOG_FILE_SUFFIX):
                self._read(name)

    def batches(self, name: str) -> list[dict]:
        """The kept batch lines of one file, oldest first. The list grows with every poll; do not change it."""
        return self._batches.setdefault(name, [])

    def forget(self) -> None:
        """Drop the batch lines read so far, so that a long run does not hold every line of it in memory.

        The lines stay counted: ``seq`` goes on from where it was. Headers and faults are kept.
        """
        for lines in self._batches.values():
            lines.clear()

    def _read(self, name: str) -> None:
        offset = self._offsets.get(name, 0)
        with open(os.path.join(self.directory, name), "rb") as handle:
            handle.seek(offset)
            data = handle.read()
        if not data:
            return
        self._offsets[name] = offset + len(data)
        *lines, rest = (self._partial.pop(name, b"") + data).split(b"\n")
        if rest:
            self._partial[name] = rest
        for raw in lines:
            number = self._lines.get(name, 0) + 1
            self._lines[name] = number
            self._take(name, number, raw)

    def _reader_fault(self, name: str, number: int, error: str) -> None:
        self.faults.append({"file": name, "where": "reader", "error": f"line {number}: {error}"})

    def _take(self, name: str, number: int, raw: bytes) -> None:
        try:
            record = json.loads(raw.decode("utf-8"))
        except ValueError as error:  # UnicodeDecodeError and json.JSONDecodeError are both ValueErrors
            self._reader_fault(name, number, f"not a JSON line: {error}")
            return
        if not isinstance(record, dict):
            self._reader_fault(name, number, f"not a JSON object: {type(record).__name__}")
            return
        kind = record.get("type")
        if kind == "fault":
            self.faults.append({"file": name, "where": record.get("where"), "error": record.get("error")})
        elif kind == "header":
            if number == 1:
                self.headers[name] = record
            else:
                self._reader_fault(name, number, "a header that is not the first line of its file")
        elif kind == "batch":
            if name in self.headers:
                self._take_batch(name, number, record)
            else:
                self._reader_fault(name, number, "a batch line in a file without a header")
        else:
            self._reader_fault(name, number, f"unknown line type {kind!r}")

    def _take_batch(self, name: str, number: int, record: dict) -> None:
        expected = self._batch_lines.get(name, 0) + 1
        seq, mode, rids = record.get("seq"), record.get("mode"), record.get("rids")
        self._batch_lines[name] = seq if _is_int(seq) else expected
        # ``seq`` counts the batch lines of a file from 1, so a gap is a line that was lost.
        if not _is_int(seq) or seq != expected:
            self._reader_fault(name, number, f"batch line with seq {seq!r}, expected {expected}")
            return
        if type(mode) is not str or not isinstance(rids, list) or any(type(rid) is not str for rid in rids):
            self._reader_fault(name, number, "batch line without a mode and a list of request ids")
            return
        # Under DP attention SGLang gives a rank that has nothing to run an idle batch without requests
        # (srt/managers/scheduler_components/dp_attn.py ``get_idle_batch``, which calls
        # ``ScheduleBatch.prepare_for_idle``). Such a line belongs to no stage.
        if mode == MODE_IDLE and not rids:
            return
        self.batches(name).append(record)


def _header_value(name: str, header: dict, *path: str) -> object:
    """One value of a header. A missing key is an invalid header; a value the recorder could not read rejects."""
    value: object = header
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise _Failure(f"{REASON_HEADER}: {name}: the header has no {'.'.join(path)}")
        value = value[key]
    # The recorder writes a value it could not read as {"unreadable": <repr of the error>}.
    if isinstance(value, dict) and "unreadable" in value:
        raise _Failure(
            f"{REASON_REJECTED}: the recorder could not read {'.'.join(path)} in {name}: {value['unreadable']}"
        )
    return value


def _holding(lines: list[dict], wanted: frozenset[str]) -> list[tuple[int, dict]]:
    """The lines that hold at least one wanted request id, each with its position in ``lines``."""
    return [(position, line) for position, line in enumerate(lines) if not wanted.isdisjoint(line["rids"])]


def _batch_key(line: dict) -> tuple[str, tuple[str, ...]]:
    return line["mode"], tuple(sorted(line["rids"]))


def _compare_counters(expected: dict[str, int], observed: dict[str, int]) -> None:
    """Fail unless the counters observed for the kept batch are the ones the artifact derives from the point."""
    for key, reason in _COUNTER_REASONS:
        if observed[key] != expected[key]:
            detail = f"{key} is {observed[key]} in the kept batch, the point plans {expected[key]}"
            if reason is None:
                raise _CacheMiss(detail)
            raise _Failure(f"{reason}: {detail}")


@dataclasses.dataclass
class _Stage:
    """One engine call and what the leader's log holds for it."""

    rids: list[str]
    prompts: list[list[int]]
    output_tokens: int
    responses: list[dict]
    # (position among the leader's batch lines since the call, batch line) of every line that holds a request
    # of the call.
    batches: list[tuple[int, dict]]


# ---------------------------------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------------------------------


class BenchmarkRun:
    """One benchmark run: its settings, its state and its artifact.

    ``recorder_dir`` is the directory the recorder of this run writes to. ``started_at`` is a reading of
    ``clock`` taken when the launcher started: the soft deadline counts from it, so the engine build is
    inside the deadline. ``clock`` and ``sleep`` are ``time.monotonic`` and ``time.sleep`` unless a test
    replaces them.

    The constructor raises for the three settings without which not even a failed artifact can be written:
    the mode, the output path and the run id. Every other setting is checked by ``execute``, which records a
    bad one as a failure.

    ``status`` is ``None`` until ``execute`` has returned or a failure was recorded, then ``"complete"``,
    ``"partial"`` or ``"failed"``. ``reason`` is ``None``, ``"timeout"`` or the failure text, which is the
    ``error`` of the artifact.
    """

    def __init__(
        self,
        settings: RunSettings,
        recorder_dir: str | os.PathLike[str],
        *,
        started_at: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], object] = time.sleep,
    ) -> None:
        if not isinstance(settings, RunSettings):
            raise TypeError(f"settings must be a RunSettings record, got {type(settings).__name__}")
        if settings.mode not in grid.MODES:
            raise ValueError(f"settings.mode must be one of {grid.MODES}, got {settings.mode!r}")
        if type(settings.run_id) is not str:
            raise TypeError(f"settings.run_id must be a str, got {type(settings.run_id).__name__}")
        if type(started_at) not in (int, float) or not math.isfinite(started_at):
            raise TypeError(f"started_at must be a finite reading of the clock, got {started_at!r}")
        # Both raise for a value that no result file and no request id can carry.
        artifact.merged_output_path(artifact.rank_output_path(settings.output_path, RANK))
        self._run_tag = settings.run_id[:RUN_TAG_CHARACTERS]
        content.request_id(self._run_tag, 0, "free", 0, RANK, 0)

        self.settings = settings
        self.recorder_dir = os.fspath(recorder_dir)
        self.logs = RecorderLogs(self.recorder_dir)
        self._started_at = started_at
        self._clock = clock
        self._sleep = sleep

        self._executed = False
        self._finished = False
        self._execute_started_at: float | None = None
        self._ended_at: float | None = None
        self._failure: dict | None = None
        self._stop_reason: str | None = None
        self._grid: grid.Grid | None = None
        self._limits: grid.Limits | None = None
        self._plan: content.ContentPlan | None = None
        # The text pool and the record of its dataset, both None in a run with seeded token ids.
        self._text_pool: content.TextPool | None = None
        self._dataset: dict | None = None
        self._leader: str | None = None
        self._followers: list[str] = []
        self._enable_overlap: bool | None = None
        self._measurements: list[dict] = []
        self._restaged: list[int] = []
        self._current_point: dict | None = None
        self._free_calls = 0
        # The last call that was checked: its request ids, how many leader lines held them, its point and its
        # output tokens.
        self._checked: tuple[frozenset[str], int, int, int] | None = None
        self._self_test_passed: list[str] = []
        self._first_tokens: dict[str, int] | None = None
        self._spare_first_tokens = 0
        self._omitted_blocks: str | None = None
        self._write_started = False
        self._written: dict | None = None

    # ------------------------------------------------------------------ outcome

    @property
    def status(self) -> str | None:
        if self._failure is not None:
            return artifact.STATUS_FAILED
        if not self._finished:
            return None
        if self._stop_reason is not None and len(self._measurements) < self._expected_points():
            return artifact.STATUS_PARTIAL
        return artifact.STATUS_COMPLETE

    @property
    def reason(self) -> str | None:
        status = self.status
        if status == artifact.STATUS_PARTIAL:
            return self._stop_reason
        if status != artifact.STATUS_FAILED:
            return None
        point, text = self._failure["point"], self._failure["reason"]
        if point is None:
            return text
        # The sentence sglbench_artifact.build_rank_artifact writes as ``error`` for a failed point.
        return f"benchmark_id={point['benchmark_id']}: benchmark point failed: {text}"

    def record_failure(self, reason: str) -> None:
        """Record a failure that belongs to no point. The first failure of a run is kept; later ones change nothing.

        Before ``execute`` the artifact then has the digest of an empty grid and no expected point. After it,
        a complete or partial run becomes failed and keeps the rows of the points that completed.
        """
        if type(reason) is not str:
            raise TypeError(f"reason must be a str, got {type(reason).__name__}: {reason!r}")
        if not reason:
            raise ValueError("reason must not be empty")
        self._fail(reason, None)

    def _fail(self, reason: str, point: dict | None) -> None:
        # Once the files are written the outcome is the one they carry: a later failure changes nothing.
        if self._failure is None and self._written is None:
            self._failure = {"point": point, "reason": reason}

    # ------------------------------------------------------------------ execute

    def execute(
        self,
        engine: object,
        *,
        token_low: int,
        token_high: int,
        text_pool: content.TextPool | None = None,
        dataset: dict | None = None,
    ) -> None:
        """Run the benchmark against ``engine``. The ids the run makes up are in ``[token_low, token_high)``.

        Without ``text_pool`` the prompts are seeded token ids of that range. With it the run is a text run:
        every prompt is made of the pool's conversations, apart from the allocated first tokens, which come
        from that range (sglbench_content.py). The ids of a conversation are the pool's and can lie outside
        the range.
        ``dataset`` is then the record of the file the pool was made from, a dict with at least the keys
        ``path``, ``sha256``, ``collection_pool`` and ``conversations``. It is written into the ``producer``
        block as it is, and those four keys into the ``kvwarm`` block of a decode or agg artifact. The two
        arguments are given together or not at all: ``TypeError`` otherwise, and for a pool or a record of
        another type.

        A failure of the run is recorded, not raised: see ``status`` and ``reason``. An exception that is not
        an ``Exception`` (``SystemExit``, ``KeyboardInterrupt``) passes through, for the caller to record as a
        stop. A second call raises ``RuntimeError``. A run that already holds a failure sends nothing.
        """
        if self._executed:
            raise RuntimeError("execute was already called for this run")
        if (text_pool is None) != (dataset is None):
            given, absent = ("text_pool", "dataset") if dataset is None else ("dataset", "text_pool")
            raise TypeError(f"text_pool and dataset must be given together: {given} was given without {absent}")
        if text_pool is not None:
            if not isinstance(text_pool, content.TextPool):
                raise TypeError(f"text_pool must be a TextPool, got {type(text_pool).__name__}")
            if not isinstance(dataset, dict):
                raise TypeError(f"dataset must be a dict, got {type(dataset).__name__}")
            absent_keys = [key for key in _KVWARM_DATASET_KEYS if key not in dataset]
            if absent_keys:
                raise TypeError(f"dataset must have the keys {_KVWARM_DATASET_KEYS}, it has no {absent_keys}")
        self._executed = True
        self._text_pool, self._dataset = text_pool, dataset
        if self._failure is not None:
            return
        self._execute_started_at = self._clock()
        try:
            self._run(engine, token_low, token_high)
            self._finished = True
        except _Failure as failure:
            self._fail(failure.reason, self._current_point)
        except Exception as error:
            exhausted = isinstance(error, ValueError) and str(error).startswith(_POOL_EXHAUSTED)
            if self._text_pool is not None and exhausted:
                # A text chain used every conversation of the pool before a prompt was complete. That is a
                # finding about the dataset and about the point that asked for the prompt, not a fault of the
                # code. The check before the sweep is meant to make it impossible.
                self._fail(f"{REASON_CONTENT_TOO_SHALLOW}: {error}", self._current_point)
                return
            # An unexpected exception is not a finding about a point, so it names none in the artifact. The
            # point that was being staged is said in the text.
            point = self._current_point
            where = "" if point is None else f" (while staging benchmark_id={point['benchmark_id']})"
            self._fail(f"{REASON_INTERNAL}: {type(error).__name__}: {error}{where}", None)
        finally:
            self._ended_at = self._clock()

    def _run(self, engine: object, token_low: int, token_high: int) -> None:
        self._check_settings()
        headers = self._wait_for_headers()
        self._check_headers(headers)
        self._limits = self._limits_of(headers)
        built = self._build_grid()
        if built.missing_phases:
            # The artifact module words this failure; the driver only asks it for the sentence.
            self._fail(self._payload(blocks=({}, {}), elapsed=0.0)["error"], None)
            return
        self._check_request_caps(headers)
        self._plan = self._content_plan(token_low, token_high)
        self._dry_build()
        self._require_no_fault()
        self._run_self_test(engine)
        self._warm_up(engine)
        for point in built.points:
            self._require_no_fault()
            # The soft deadline is looked at between two points only, and before the first one. A point that
            # has started is finished; a deadline that passes after the last point leaves the run complete.
            if self._clock() - self._started_at >= self.settings.timeout_seconds:
                self._stop_reason = artifact.STOP_REASON_TIMEOUT
                return
            record = point.as_dict()
            self._current_point = record
            self._measurements.append(self._measure(engine, point, record))
            self._current_point = None
        self._require_no_fault()
        self._refuse_late_batches()

    def _check_settings(self) -> None:
        settings = self.settings
        if not isinstance(settings.caps, grid.SampleCaps):
            raise TypeError(f"settings.caps must be a SampleCaps record, got {type(settings.caps).__name__}")
        _require_setting_int("warmup_iterations", settings.warmup_iterations, 0)
        _require_setting_int("expected_scheduler_processes", settings.expected_scheduler_processes, 1)
        _require_setting_seconds("timeout_seconds", settings.timeout_seconds, positive=True)
        _require_setting_seconds("log_wait_seconds", settings.log_wait_seconds, positive=False)
        _require_setting_seconds("header_wait_seconds", settings.header_wait_seconds, positive=False)
        for name in ("content_seed", "worker_id"):
            if type(getattr(settings, name)) is not str:
                raise TypeError(f"settings.{name} must be a str, got {getattr(settings, name)!r}")
        sources = (content.CONTENT_SEEDED_IDS, content.CONTENT_TEXT_CHAINS)
        if settings.content_source not in sources:
            raise ValueError(f"settings.content_source must be one of {sources}, got {settings.content_source!r}")
        if settings.content_source == content.CONTENT_TEXT_CHAINS and self._text_pool is None:
            raise TypeError("settings.content_source names dataset text, and execute was given no text pool")

    # ------------------------------------------------------------------ headers

    def _wait_for_headers(self) -> dict[str, dict]:
        """Wait for one valid header per scheduler process and find the leader among them."""
        expected = self.settings.expected_scheduler_processes
        deadline = self._clock() + self.settings.header_wait_seconds
        while True:
            self.logs.poll()
            if len(self.logs.headers) >= expected:
                break
            if self._clock() >= deadline:
                faults = f"; recorder faults: {self.logs.faults}" if self.logs.faults else ""
                raise _Failure(
                    f"{REASON_HEADER}: {len(self.logs.headers)} of {expected} recorder header(s) in "
                    f"{self.recorder_dir} after {self.settings.header_wait_seconds} s{faults}"
                )
            self._sleep(POLL_SECONDS)
        headers = {name: self.logs.headers[name] for name in sorted(self.logs.headers)}
        if len(headers) != expected:
            raise _Failure(f"{REASON_HEADER}: {len(headers)} recorder headers, expected {expected}: {list(headers)}")
        leaders = []
        for name, header in headers.items():
            if not _is_int(header.get("format")) or header["format"] != RECORD_FORMAT:
                raise _Failure(f"{REASON_HEADER}: {name}: format {header.get('format')!r}, expected {RECORD_FORMAT}")
            # SGLang logs a class replacement that failed and builds the stock scheduler
            # (srt/plugins/hook_registry.py ``HookRegistry.apply_hooks``), and a plugin that registers later
            # displaces the recorder. The class name in the header is the only sign of either.
            if header.get("class") != RECORDER_CLASS_NAME:
                raise _Failure(
                    f"{REASON_HEADER}: {name}: class {header.get('class')!r}, expected {RECORDER_CLASS_NAME!r}"
                )
            # ``leader``: this process reads the tokenizer socket (srt/managers/scheduler.py
            # ``Scheduler.init_ipc_channels`` sets ``recv_from_tokenizer`` there and None elsewhere).
            leader = _header_value(name, header, "leader")
            if type(leader) is not bool:
                raise _Failure(f"{REASON_HEADER}: {name}: leader is {leader!r}, expected true or false")
            if leader:
                leaders.append(name)
        if len(leaders) != 1:
            raise _Failure(f"{REASON_HEADER}: {len(leaders)} leader header(s), expected exactly one: {leaders}")
        self._leader = leaders[0]
        self._followers = [name for name in headers if name != self._leader]
        overlap = headers[self._leader].get("enable_overlap")
        self._enable_overlap = overlap if type(overlap) is bool else None
        return headers

    def _check_headers(self, headers: dict[str, dict]) -> None:
        """Refuse what the feasibility rules and the timing do not cover, as every scheduler process reports it."""
        for name, header in headers.items():
            self._check_header(name, header)
        for path in _AGREED_VALUES:
            values = [_header_value(name, header, *path) for name, header in headers.items()]
            if any(value != values[0] for value in values[1:]):
                raise _Failure(
                    f"{REASON_REJECTED}: scheduler processes disagree on {path[-1]}: {json.dumps(values, default=repr)}"
                )

    def _check_header(self, name: str, header: dict) -> None:
        def read(*path: str) -> object:
            return _header_value(name, header, *path)

        def reject(text: str) -> None:
            raise _Failure(f"{REASON_REJECTED}: {text}")

        # ``sglang.__version__``, as the scheduler process imports it.
        version = read("sglang_version")
        if version not in TESTED_SGLANG_VERSIONS:
            reject(f"SGLang {_shown(version)} is not a tested version (tested: {', '.join(TESTED_SGLANG_VERSIONS)})")
        for limit in _LIMIT_NAMES:
            read("limits", limit)
        for graph in _GRAPH_NAMES:
            read(graph)
        # With overlap scheduling off the result of a batch normally carries no copy event:
        # ``Scheduler.run_batch`` (srt/managers/scheduler.py) records one on its overlap path, and off it only
        # for the auxiliary output of a sampling observer (``Scheduler._copy_auxiliary_output_to_cpu``). The
        # recorder's output time could then lie before the end of the GPU work of the batch.
        overlap = read("enable_overlap")
        if overlap is not True:
            reject(
                f"overlap scheduling is off (enable_overlap={_shown(overlap)}); "
                "the output time of a batch would not follow its GPU work"
            )
        for key, text in _CHECKS_FALSE:
            value = read("checks", key)
            if value is not False:
                reject(text.format(name=key, v=_shown(value)))
        # ``Scheduler.schedule_policy``: every policy but fcfs reorders the waiting queue
        # (srt/managers/schedule_policy.py ``SchedulePolicy.calc_priority``).
        policy = read("checks", "schedule_policy")
        if policy != "fcfs":
            reject(f"schedule policy must be fcfs without priority scheduling (schedule_policy={_shown(policy)})")
        # After an extend batch SGLang admits no prefill for this many rounds (srt/managers/scheduler.py
        # ``Scheduler._should_defer_prefill``), so decode batches would run between the extend batches of one
        # call.
        interval = read("checks", "prefill_decode_interval")
        if interval is not None and not (_is_int(interval) and interval == 0):
            reject(
                f"prefill_decode_interval must be 0 (got {_shown(interval)}); SGLang sets 22 for some Qwen3-VL setups"
            )
        # ``Scheduler.max_new_tokens_limit`` is SGLANG_MAX_NEW_TOKENS_LIMIT; SGLang applies it when it is above
        # 0 (``Scheduler.init_req_max_new_tokens``). Below seven a decode run has fewer than six decode batches.
        limit = read("checks", "max_new_tokens_limit")
        if limit is not None and not (_is_int(limit) and (limit <= 0 or limit >= grid.DECODE_OUTPUT_TOKENS)):
            reject(
                f"SGLANG_MAX_NEW_TOKENS_LIMIT={_shown(limit)} is below the "
                f"{grid.DECODE_OUTPUT_TOKENS} output tokens of a decode run"
            )
        # The cap SGLang applies to running plus new requests is ``pp_max_micro_batch_size``
        # (``Scheduler.get_num_allocatable_reqs``). The grid uses ``max_running_requests``, so the two must be
        # the same number.
        micro_batch, running = read("checks", "pp_max_micro_batch_size"), read("limits", "max_running_requests")
        if not _is_int(micro_batch) or micro_batch != running:
            reject(f"pp_max_micro_batch_size {_shown(micro_batch)} differs from the running cap {_shown(running)}")
        # srt/runtime_context.py ``get_parallel``: one batch call cannot place its requests on several
        # attention-DP ranks, and pipeline parallelism runs another event loop.
        dp_size, pp_size = read("ranks", "attn_dp_size"), read("ranks", "pp_size")
        if read("ranks", "attn_dp_enabled") is not False or not (_is_int(dp_size) and dp_size == 1):
            reject(f"DP attention is not supported yet (attn_dp_size={_shown(dp_size)})")
        if not (_is_int(pp_size) and pp_size == 1):
            reject(f"pipeline parallelism is not supported (pp_size={_shown(pp_size)})")
        read("ranks", "tp_size")
        read("checks", "max_queued_requests")

    def _limits_of(self, headers: dict[str, dict]) -> grid.Limits:
        header = headers[self._leader]
        record = {name: _header_value(self._leader, header, "limits", name) for name in _LIMIT_NAMES}
        record.update({name: _header_value(self._leader, header, name) for name in _GRAPH_NAMES})
        try:
            return grid.Limits.from_mapping(record)
        except (TypeError, ValueError) as error:
            raise _Failure(f"{REASON_HEADER}: {self._leader}: limits: {error}") from error

    def _build_grid(self) -> grid.Grid:
        settings = self.settings
        try:
            self._grid = grid.build_grid(settings.mode, self._limits, settings.caps, manifest=settings.manifest)
        except ValueError as error:
            raise _Failure(f"{REASON_REJECTED}: {error}") from error
        return self._grid

    def _check_request_caps(self, headers: dict[str, dict]) -> None:
        """Both request caps must be unset or at least the largest batch of the grid.

        SGLang puts at most ``prefill_max_requests`` requests in an extend batch
        (srt/managers/schedule_policy.py ``PrefillAdder.add_one_req``) and aborts a request that would make
        the waiting queue longer than ``max_queued_requests`` (srt/managers/scheduler.py
        ``Scheduler._abort_on_queued_limit``).
        """
        # The self-test sends pairs, whatever the grid holds.
        largest = max(SELF_TEST_REQUESTS, *(point.batch_size for point in self._grid.points))
        caps = [
            ("prefill_max_requests", self.settings.prefill_max_requests),
            ("max_queued_requests", self.settings.max_queued_requests),
        ]
        for name, header in headers.items():
            caps.append(("max_queued_requests", _header_value(name, header, "checks", "max_queued_requests")))
        for name, value in caps:
            if value is not None and not (_is_int(value) and value >= largest):
                raise _Failure(f"{REASON_REJECTED}: {name}={_shown(value)} is below the largest batch {largest}")

    # ------------------------------------------------------------------ content

    def _first_token_need(self) -> dict[str, int]:
        """First tokens the whole run takes from the token range, by what takes them.

        The content plan gives every slot sequence and every zero-cached prompt a first token of its own and
        never gives one back (sglbench_content.py, "First-token capacity"). A slot is taken by a prefill
        point with cached tokens, by a decode point and by the two prefix checks of the self-test. A
        zero-cached prompt is taken by each request of the warm and of the measured shot of a prefill point
        without cached tokens, by the warm-up and by each free prompt of the self-test. The reserve covers the
        re-stage of the largest such prefill point: two more prompts for each of its requests.
        """
        points = self._grid.points
        slot_batches = [point.batch_size for point in points if point.phase == grid.DECODE or point.cached_tokens]
        zero_cached = [point.batch_size for point in points if point.phase == grid.PREFILL and not point.cached_tokens]
        need = {
            "slots": max([SELF_TEST_REQUESTS, *slot_batches]),
            "zero_cached_prefill_prompts": 2 * sum(zero_cached),
            "warmup_prompts": 1 if self.settings.warmup_iterations else 0,
            "self_test_free_prompts": _SELF_TEST_FREE_PROMPTS,
            "restage_reserve": 2 * max(zero_cached, default=0),
        }
        need["total"] = sum(need.values())
        return need

    def _longest_prompt(self) -> int:
        """The length of the longest prompt the run sends: over the points, the self-test and the warm-up.

        A prefill shot has its cached and its new tokens, a decode prompt its context less the three tokens
        the run generates before the kept step, and the longest prompt of the self-test is the unaligned one.
        """
        lengths = [2 * grid.cached_unit(self._limits.page_size) + _UNALIGNED_EXTRA_TOKENS]
        if self.settings.warmup_iterations:
            lengths.append(WARMUP_PROMPT_TOKENS)
        for point in self._grid.points:
            if point.phase == grid.PREFILL:
                lengths.append(point.cached_tokens + point.new_tokens)
            else:
                lengths.append(point.cached_tokens - grid.DECODE_PROMPT_LEAD)
        return max(lengths)

    def _require_deep_pool(self) -> None:
        """Fail a text run whose pool does not hold more tokens than the longest prompt needs.

        A prompt part is a chain of whole conversations, each used at most once, so one chain gives at most
        the tokens of the whole pool and never wraps (sglbench_content.py). The need is the longest prompt and
        one token more: a prefill shot makes the plan look at the slot's token after the cached prefix.
        Dynamo's vLLM self-benchmark proves the same before its sweep, one chain at its depth cap
        (``_kvwarm_probe_content``). There a shallow dataset switches the real text off; here it fails the
        run before any request.
        """
        need = self._longest_prompt() + 1
        tokens = self._text_pool.tokens
        if tokens <= need:
            raise _Failure(
                f"{REASON_CONTENT_TOO_SHALLOW}: the text pool holds {tokens} tokens and the longest prompt needs {need}"
            )

    def _content_plan(self, token_low: int, token_high: int) -> content.ContentPlan:
        """Build the one content plan of the run, after the token range was compared with the planned need.

        In a text run the pool is compared with the longest prompt as well, and the plan draws every prompt
        part from it. A run with seeded token ids builds its plan without a pool.
        """
        if not _is_int(token_low) or not _is_int(token_high):
            raise TypeError(f"token_low and token_high must be ints, got {token_low!r} and {token_high!r}")
        need = self._first_token_need()
        size = token_high - token_low
        if size < content.MIN_VOCAB_SIZE:
            raise _Failure(
                f"{REASON_TOKEN_RANGE}: the token range [{token_low}, {token_high}) holds {size} ids, "
                f"the content plan needs at least {content.MIN_VOCAB_SIZE}"
            )
        if need["total"] > size:
            raise _Failure(
                f"{REASON_TOKEN_RANGE}: the run needs {need['total']} first tokens ({need}) and the token range "
                f"[{token_low}, {token_high}) holds {size} ids"
            )
        self._first_tokens = need
        # What is left for re-stages of zero-cached points beyond the one the reserve covers.
        self._spare_first_tokens = size - need["total"] + need["restage_reserve"]
        if self._text_pool is None:
            return content.ContentPlan(self.settings.content_seed, token_low, token_high)
        self._require_deep_pool()
        return content.ContentPlan(self.settings.content_seed, token_low, token_high, pool=self._text_pool)

    def _take_spare_first_tokens(self, point: grid.Point) -> None:
        """Account for the zero-cached prompts a re-staged prefill point without cached tokens issues again."""
        if point.phase != grid.PREFILL or point.cached_tokens:
            return
        needed = 2 * point.batch_size
        if needed > self._spare_first_tokens:
            raise _Failure(
                f"{REASON_TOKEN_RANGE}: the re-stage needs {needed} more first tokens and "
                f"{self._spare_first_tokens} are left in [{self._plan.token_low}, {self._plan.token_high})"
            )
        self._spare_first_tokens -= needed

    def _dry_build(self) -> None:
        """Build one empty artifact with the real blocks: a value that is not JSON fails here, not after the sweep."""
        try:
            self._payload(failure={"point": None, "reason": "dry build"}, elapsed=0.0)
        except (TypeError, ValueError) as error:
            self._omitted_blocks = f"{type(error).__name__}: {error}"
            raise _Failure(f"{REASON_INTERNAL}: {self._omitted_blocks}") from error

    # ------------------------------------------------------------------ stages

    def _require_no_fault(self) -> None:
        """Read the logs and fail the run on any fault line: the recorder then missed or mangled a batch."""
        self.logs.poll()
        self._raise_on_fault()

    def _refuse_late_batches(self) -> None:
        """Fail when the leader wrote a batch for a call after that call had been checked.

        With overlap scheduling SGLang launches the next batch before it processes the result of the batch
        before it (srt/managers/scheduler.py ``Scheduler.event_loop_overlap``). A batch beyond the ones a call
        must have is therefore written after the response of the call, when the call has been checked. It is
        looked for here, among the lines read since the last call, before they are dropped: a line that holds
        requests of the last call beyond the lines that call was checked on, or requests of an earlier call.
        """
        if self._checked is None:
            return
        wanted, count, benchmark_id, output_tokens = self._checked
        lines = self.logs.batches(self._leader)
        late = [(line, benchmark_id, output_tokens > 1) for _position, line in _holding(lines, wanted)[count:]]
        for line in lines:
            for rid in line["rids"]:
                parts = None if rid in wanted else content.parse_request_id(rid)
                if parts is not None and parts["run_id"] == self._run_tag:
                    late.append((line, parts["benchmark_id"], parts["stage"] == "run"))
                    break
        if not late:
            return
        line, owner, decode_run = late[0]
        if self._current_point is not None and self._current_point["benchmark_id"] != owner:
            self._current_point = None  # the batch belongs to a call outside the point that is being staged
        raise _Failure(
            f"{REASON_DECODE_CONTEXT if decode_run else REASON_PREFILL_TOKENS}: a {line['mode']} batch "
            f"(seq {line['seq']}) holds requests of a call that had already been checked"
        )

    def _raise_on_fault(self) -> None:
        faults = self.logs.faults
        if faults:
            first = faults[0]
            more = f" (and {len(faults) - 1} more)" if len(faults) > 1 else ""
            raise _Failure(f"{REASON_RECORDER_FAULT}: {first['file']}: {first['where']}: {first['error']}{more}")
        # A log file that appears after the headers were checked is a scheduler process nothing was checked for.
        expected = self.settings.expected_scheduler_processes
        if self._leader is not None and len(self.logs.headers) != expected:
            raise _Failure(
                f"{REASON_HEADER}: {len(self.logs.headers)} recorder headers after the run started, expected "
                f"{expected}: {sorted(self.logs.headers)}"
            )

    def _stage(
        self,
        engine: object,
        benchmark_id: int,
        stage: str,
        attempt: int,
        prompts: list[list[int]],
        output_tokens: int,
        *,
        followers: bool,
    ) -> _Stage:
        """Send one engine call, check every response and collect the batches of the call from the logs.

        Deviation note (one batch call per stage).
        What differs: serving receives requests one at a time and the scheduler batches what happens to
        wait. Here the ``b`` requests of a stage go out in one call, as one list of prompts. SGLang sends a
        pre-tokenized batch as one message (srt/managers/tokenizer_manager.py
        ``TokenizerManager._send_batch_request``) and queues every request of it in one receive step
        (srt/managers/scheduler.py ``Scheduler.handle_batch_generate_request``), so the next step forms one
        extend batch from them when they fit. The batch itself is formed and run by the stock scheduler.
        Direction of the error: none known for the timed batch. The batch sizes of a table are the planned
        ones and not a serving mix.
        Sources: read at both tags; seen on running engines:
        benchmarks/evidence/collector/sglbench-engine-probe-20261113 (check_2, check_9).
        How to retire it: nothing to retire; it is how a batch of a chosen size is staged.

        The engine object adds the request fields that are the same for every request (temperature 0,
        ``ignore_eos``, streaming). The driver chooses the three that differ: the prompts and the request
        ids, which become ``input_ids`` and ``rid`` of srt/managers/io_struct.py ``GenerateReqInput``, and
        the output-token count, ``max_new_tokens`` of srt/sampling/sampling_params.py ``SamplingParams``.
        """
        rids = [
            content.request_id(self._run_tag, benchmark_id, stage, attempt, RANK, slot) for slot in range(len(prompts))
        ]
        # The lines of earlier calls are read and dropped: the request ids of this call are new, so its
        # batches can only be among the lines that follow.
        self.logs.poll()
        self._refuse_late_batches()
        self.logs.forget()
        try:
            responses = engine.generate(prompts, rids, output_tokens)
        except Exception as error:
            raise _Failure(f"{REASON_ENGINE_CALL}: {type(error).__name__}: {error}") from error
        self._check_responses(rids, prompts, output_tokens, responses)
        wanted = frozenset(rids)
        batches = self._wait_for_batches(wanted, output_tokens, followers)
        self._checked = (wanted, len(batches), benchmark_id, output_tokens)
        return _Stage(rids, prompts, output_tokens, responses, batches)

    def _check_responses(
        self, rids: list[str], prompts: list[list[int]], output_tokens: int, responses: object
    ) -> None:
        """Every request of every stage must have been taken whole and must have run to its last token.

        The keys are those of an output's ``meta_info`` (srt/managers/tokenizer_manager.py
        ``TokenizerManager._handle_batch_output``): ``prompt_tokens`` is the length of the prompt as the
        scheduler holds it, ``completion_tokens`` the output length, ``finish_type`` the type of the finish
        reason, which is ``"length"`` when the request stopped at its ``max_new_tokens``
        (srt/managers/schedule_batch.py ``FINISH_LENGTH``), and ``num_retractions`` counts how often the
        request was pushed out of a batch and run again (``Req.retraction_count``).
        """
        if not isinstance(responses, list) or len(responses) != len(rids):
            raise _Failure(f"{REASON_ENGINE_CALL}: the call returned {responses!r:.200} for {len(rids)} request(s)")
        for rid, response in zip(rids, responses, strict=True):
            if not isinstance(response, dict) or response.get("rid") != rid or "finish_type" not in response:
                raise _Failure(f"{REASON_ENGINE_CALL}: {rid}: the response is {response!r:.200}")
            for key in _RESPONSE_INT_KEYS:
                if not _is_int(response.get(key)):
                    raise _Failure(f"{REASON_ENGINE_CALL}: {rid}: {key} is {response.get(key)!r}")
        for rid, prompt, response in zip(rids, prompts, responses, strict=True):
            if response["prompt_tokens"] != len(prompt):
                raise _Failure(
                    f"{REASON_PREFILL_TOKENS}: {rid}: the engine reports {response['prompt_tokens']} prompt "
                    f"tokens, the prompt has {len(prompt)}"
                )
            done = response["finish_type"] == "length" and response["completion_tokens"] == output_tokens
            if not done or response["num_retractions"] != 0:
                raise _Failure(
                    f"{REASON_NOT_COMPLETED}: {rid}: finish_type={response['finish_type']!r}, "
                    f"completion_tokens={response['completion_tokens']} of {output_tokens}, "
                    f"num_retractions={response['num_retractions']}"
                )

    def _wait_for_batches(self, wanted: frozenset[str], output_tokens: int, followers: bool) -> list[tuple[int, dict]]:
        """Read the logs until they hold the batches of a call; give up after ``log_wait_seconds``.

        The recorder writes a batch line before it lets the stock scheduler process the result
        (sglbench_recorder.py), and the leader sends the output of a batch from there. So the leader's lines
        of a call are in its file when the engine call returns. The other scheduler processes run the same
        batches and write from their own processes in no order with the call, so their files are waited
        for. A fault line that turns up meanwhile fails the run at once.
        """
        deadline = self._clock() + self.settings.log_wait_seconds
        while True:
            self.logs.poll()
            self._raise_on_fault()
            batches = _holding(self.logs.batches(self._leader), wanted)
            missing = self._missing_batches(batches, wanted, output_tokens, followers)
            if missing is None:
                return batches
            if self._clock() >= deadline:
                raise _Failure(f"{REASON_NO_BATCHES}: {missing} after {self.settings.log_wait_seconds} s")
            self._sleep(POLL_SECONDS)

    def _missing_batches(
        self, batches: list[tuple[int, dict]], wanted: frozenset[str], output_tokens: int, followers: bool
    ) -> str | None:
        """What the logs still lack for a call, or ``None`` when they hold every batch the call must have.

        A call of ``m`` output tokens must have an extend batch for every request, which yields the first
        token, and ``m - 1`` decode batches. With overlap scheduling on there is no decode batch beyond
        them: SGLang drops a request from the next decode batch when the result still in the queue will
        finish it (srt/managers/scheduler.py ``Scheduler._reqs_finishing_inflight``, switched on by
        ``Scheduler.event_loop_overlap``).
        """
        extended: set[str] = set()
        decodes = 0
        for _position, line in batches:
            if line["mode"] == MODE_EXTEND:
                extended.update(line["rids"])
            elif line["mode"] == MODE_DECODE:
                decodes += 1
        lacking = len(wanted - extended)
        if lacking:
            return f"{self._leader} holds no extend batch for {lacking} of {len(wanted)} request(s)"
        if decodes < output_tokens - 1:
            return f"{self._leader} holds {decodes} of the {output_tokens - 1} decode batch(es)"
        if followers:
            # With a TP size above 1 every TP rank runs every batch: rank 0 reads the tokenizer socket and
            # broadcasts each request to the other ranks, which schedule it the same way
            # (srt/managers/scheduler_components/request_receiver.py ``SchedulerRequestReceiver.recv_requests``,
            # read in the source at both tags). Seen on running engines on 2026-11-16, in two launches at
            # TP 2, one with SGLang 0.5.22 and one with 0.5.23: both ranks wrote the same batches and every
            # call passed this check. The files of those launches are not kept in this repository; the probe
            # driver kept in benchmarks/evidence/collector/sglbench-engine-probe-20261113 read the leader's
            # log only. Each of the leader's batches must be in every other log with the same mode and the
            # same requests.
            needed = collections.Counter(_batch_key(line) for _position, line in batches)
            for name in self._followers:
                found = collections.Counter(
                    _batch_key(line) for _position, line in _holding(self.logs.batches(name), wanted)
                )
                short = needed - found
                if short:
                    return f"{name} lacks {sum(short.values())} of the {len(batches)} batch(es) of the call"
        return None

    def _require_cached(self, call: _Stage, expected: int) -> None:
        """Every request must report the planned number of cached tokens (``cached_tokens`` of ``meta_info``)."""
        wrong = [response for response in call.responses if response["cached_tokens"] != expected]
        if wrong:
            raise _CacheMiss(
                f"{len(wrong)} of {len(call.responses)} request(s) report another cached count than the planned "
                f"{expected}: {wrong[0]['rid']} reports {wrong[0]['cached_tokens']}"
            )

    def _one_extend_batch(self, call: _Stage) -> dict:
        """The one extend batch of a one-token call, which must hold exactly the requests of the call."""
        lines = [line for _position, line in call.batches]
        others = [line for line in lines if line["mode"] != MODE_EXTEND]
        if others:
            raise _Failure(
                f"{REASON_PREFILL_TOKENS}: a {others[0]['mode']} batch (seq {others[0]['seq']}) holds requests "
                "of a one-token call"
            )
        counts = collections.Counter(rid for line in lines for rid in line["rids"])
        chunked = [rid for rid in call.rids if counts[rid] > 1]
        if chunked:
            raise _Failure(
                f"{REASON_PREFILL_TOKENS}: {chunked[0]} is in {counts[chunked[0]]} extend batches: "
                "the request was chunked"
            )
        if len(lines) != 1:
            raise _Failure(
                f"{REASON_BATCH_SIZE}: the call ran as {len(lines)} extend batches of "
                f"{[len(line['rids']) for line in lines]} requests, expected one of {len(call.rids)}"
            )
        if sorted(lines[0]["rids"]) != sorted(call.rids):
            foreign = sorted(set(lines[0]["rids"]) - set(call.rids))
            raise _Failure(
                f"{REASON_BATCH_SIZE}: the extend batch (seq {lines[0]['seq']}) holds {len(lines[0]['rids'])} "
                f"request(s), {len(foreign)} of them foreign, expected exactly the {len(call.rids)} of the call"
            )
        return lines[0]

    def _extend_counts(self, line: dict) -> tuple[list[int], list[int], int]:
        """``prefix_lens``, ``extend_lens`` and ``extend_num_tokens`` of an extend batch line.

        SGLang sets all three when it forms an extend batch (srt/managers/schedule_batch.py
        ``ScheduleBatch.prepare_for_extend``): per request the cached tokens and the tokens this batch
        computes, in the order of the requests, and the sum of the computed tokens.
        """
        size = len(line["rids"])
        prefix_lens, extend_lens, extend_tokens = (
            line.get(key) for key in ("prefix_lens", "extend_lens", "extend_num_tokens")
        )
        if not _is_int_list(prefix_lens, size) or not _is_int_list(extend_lens, size) or not _is_int(extend_tokens):
            raise _Failure(
                f"{REASON_RECORDER_FAULT}: {self._leader}: extend batch seq {line['seq']} has no usable "
                "prefix_lens, extend_lens and extend_num_tokens"
            )
        return prefix_lens, extend_lens, extend_tokens

    def _output_time(self, line: dict) -> float:
        """``out_ts`` of a batch line: the time its result was on the host, on the leader's monotonic clock.

        The recorder takes it after the copy event of the result (srt/managers/utils.py
        ``GenerationBatchResult.copy_done``) and before the stock ``Scheduler.process_batch_result`` runs. A
        line that says the result had no such event is not a time after the GPU work and is refused.

        All times are the leader's. With a TP size above 1 the other ranks run the same batches on clocks of
        their own; the leader is the process whose loop forms the batches and sends their output.
        """
        if line.get("copy_done") is not True:
            raise _Failure(
                f"{REASON_RECORDER_FAULT}: {self._leader}: batch seq {line['seq']} was stamped without a copy "
                f"event (copy_done={line.get('copy_done')!r})"
            )
        return self._time(line, "out_ts")

    def _time(self, line: dict, key: str) -> float:
        value = line.get(key)
        if type(value) is not float or not math.isfinite(value):
            raise _Failure(f"{REASON_RECORDER_FAULT}: {self._leader}: batch seq {line['seq']} has {key}={value!r}")
        return value

    def _seconds(self, value: float, what: str) -> float:
        if not value > 0:
            raise _Failure(f"{REASON_RECORDER_FAULT}: {self._leader}: {what} took {value!r} s on the scheduler's clock")
        return value

    # ------------------------------------------------------------------ self-test and warm-up

    def _free_stage(self, engine: object, prompts: list[list[int]], output_tokens: int, *, followers: bool) -> _Stage:
        """A stage outside any point: benchmark id 0, stage ``free``, and a call number in place of the attempt."""
        call = self._free_calls
        self._free_calls += 1
        return self._stage(engine, 0, "free", call, prompts, output_tokens, followers=followers)

    def _run_self_test(self, engine: object) -> None:
        """Check, on this engine, the four behaviours the point procedures rely on.

        The checks use the content plan of the run: a second plan would issue the same zero-cached prompts
        again. The batch-shape check and the seven-token run use free prompts. The two prefix checks use
        slots 0 and 1 with one-token requests only, so no generated token follows a slot sequence in the
        cache before the prefill points run.

        Deviation note (requests before the first timed batch).
        What differs: the self-test and the warm-up leave their prompts in the prefix cache, and nothing is
        flushed before or between points. Serving also runs on a cache that earlier requests filled.
        Direction of the error: none known. For a prefill sample SGLang allocates and evicts while it forms
        the batch (srt/managers/scheduler.py ``Scheduler.get_next_batch_to_run``), before ``run_batch``
        starts the timed interval. A decode step is timed from output to output, so an eviction between two
        steps would lie inside it. On running engines the decode sample taken after more tokens of other
        prompts than the pool holds had been sent was within 0.05 percent of the sample after a flush:
        benchmarks/evidence/collector/sglbench-engine-probe-20261113 (check_8; Qwen3-8B, TP 1, both tags).
        How to retire it: nothing to retire.
        """
        unit = grid.cached_unit(self._limits.page_size)
        checks = (
            self._self_test_batch_shape,
            self._self_test_aligned_prefix,
            self._self_test_unaligned_prompt,
            self._self_test_seven_token_run,
        )
        for name, check in zip(SELF_TEST_CHECKS, checks, strict=True):
            try:
                check(engine, unit)
            except _Failure as failure:
                raise _Failure(f"{REASON_SELF_TEST}: {name}: {failure.reason}") from failure
            except _CacheMiss as miss:
                raise _Failure(f"{REASON_SELF_TEST}: {name}: {miss}") from miss
            self._self_test_passed.append(name)

    def _require_shape(self, what: str, reason: str | None, allowed: tuple[str, ...] = ()) -> None:
        """A self-test call must fit the limits of this engine; otherwise the check could not tell anything."""
        if reason is not None and reason not in allowed:
            raise _Failure(f"the limits of this engine do not admit {what}: {reason}")

    def _self_test_batch_shape(self, engine: object, unit: int) -> None:
        """One call of two requests gives one extend batch that holds both, and no other batch."""
        self._require_shape(
            f"two requests of {unit} tokens in one extend batch",
            grid.prefill_unfit_reason(self._limits, SELF_TEST_REQUESTS, unit, 0),
        )
        prompts = [self._plan.free_prompt(RANK, unit) for _slot in range(SELF_TEST_REQUESTS)]
        call = self._free_stage(engine, prompts, _ONE_OUTPUT_TOKEN, followers=True)
        self._require_cached(call, 0)
        self._one_extend_batch(call)

    def _self_test_aligned_prefix(self, engine: object, unit: int) -> None:
        """Two requests that continue a stored page-aligned prefix report exactly that prefix as cached."""
        self._require_shape(
            f"two requests of {unit} cached and {unit} new tokens in one extend batch",
            grid.prefill_unfit_reason(self._limits, SELF_TEST_REQUESTS, unit, unit),
        )
        plan = self._plan
        seeds = [plan.prefill_seed_prompt(RANK, slot, unit) for slot in range(SELF_TEST_REQUESTS)]
        self._free_stage(engine, seeds, _ONE_OUTPUT_TOKEN, followers=False)
        shots = [plan.prefill_prompt(RANK, slot, unit, unit) for slot in range(SELF_TEST_REQUESTS)]
        call = self._free_stage(engine, shots, _ONE_OUTPUT_TOKEN, followers=True)
        self._require_cached(call, grid.expected_cached_tokens(grid.PREFILL, unit, self._limits.page_size))
        self._one_extend_batch(call)

    def _self_test_unaligned_prompt(self, engine: object, unit: int) -> None:
        """A prompt that ends inside a page, sent a second time, reports the count rounded down to whole pages.

        This is the cached count a decode run relies on: its prompt is the one its seed stored, SGLang
        recomputes at least the last token and matches whole pages only. The expected count is the one the
        grid module derives for a decode point with this prompt.
        """
        length = 2 * unit + _UNALIGNED_EXTRA_TOKENS
        self._require_shape(
            f"two requests of {length} tokens",
            grid.prefill_unfit_reason(self._limits, SELF_TEST_REQUESTS, length, 0),
            allowed=(grid.REASON_CHUNK_BUDGET, grid.REASON_PREFILL_TOKEN_BUDGET),
        )
        prompts = [self._plan.prefill_seed_prompt(RANK, slot, length) for slot in range(SELF_TEST_REQUESTS)]
        self._free_stage(engine, prompts, _ONE_OUTPUT_TOKEN, followers=False)
        call = self._free_stage(engine, prompts, _ONE_OUTPUT_TOKEN, followers=False)
        context = length + grid.DECODE_PROMPT_LEAD
        self._require_cached(call, grid.expected_cached_tokens(grid.DECODE, context, self._limits.page_size))

    def _self_test_seven_token_run(self, engine: object, unit: int) -> None:
        """A seven-token call of two requests gives one extend batch and then six decode batches of both."""
        context = unit + grid.DECODE_PROMPT_LEAD
        self._require_shape(
            f"a seven-token run of two requests of {unit} tokens",
            grid.prefill_unfit_reason(self._limits, SELF_TEST_REQUESTS, unit, 0)
            or grid.decode_unfit_reason(self._limits, SELF_TEST_REQUESTS, context),
        )
        prompts = [self._plan.free_prompt(RANK, unit) for _slot in range(SELF_TEST_REQUESTS)]
        call = self._free_stage(engine, prompts, grid.DECODE_OUTPUT_TOKENS, followers=True)
        self._require_cached(call, 0)
        extends, _decodes = self._run_sequence(call, unit)
        if len(extends) != 1:
            raise _Failure(f"{REASON_DECODE_CONTEXT}: the run has {len(extends)} extend batches, expected one")

    def _warm_up(self, engine: object) -> None:
        """The global warm-up: one free prompt of 256 tokens with ``warmup_iterations`` output tokens.

        Dynamo's vLLM self-benchmark sends the same request before its sweep and skips it at 0
        (``_bench_step_warmup``). Here the request also ignores EOS, as every request of the run does, so
        it always runs its full number of decode steps.
        """
        iterations = self.settings.warmup_iterations
        if iterations == 0:
            return
        prompt = self._plan.free_prompt(RANK, WARMUP_PROMPT_TOKENS)
        self._free_stage(engine, [prompt], iterations, followers=False)

    # ------------------------------------------------------------------ points

    def _measure(self, engine: object, point: grid.Point, record: dict) -> dict:
        """Stage one point and return its measurement; stage it a second time when a cached count was wrong.

        Only a wrong cached count is staged again, once, with attempt 1 in the request ids: an eviction can
        take a seeded prefix away, and the second attempt seeds it again. Every other failure ends the run.
        Dynamo's vLLM self-benchmark re-stages a seeded prefill point once in the same case
        (``_bench_realseed_pending_step``).
        """
        miss: _CacheMiss | None = None
        for attempt in (0, 1):
            if attempt:
                self._restaged.append(point.benchmark_id)
                self._take_spare_first_tokens(point)
            try:
                if point.phase == grid.PREFILL:
                    wall_time, extra = self._prefill_attempt(engine, point, record, attempt)
                else:
                    wall_time, extra = self._decode_attempt(engine, point, record, attempt)
            except _CacheMiss as error:
                miss = error
                continue
            fpm = artifact.build_fpm(record, RANK, wall_time, extra=extra)
            return artifact.build_measurement(record, {RANK: fpm})
        raise _Failure(f"{REASON_CACHE}: {miss}")

    def _prefill_attempt(self, engine: object, point: grid.Point, record: dict, attempt: int) -> tuple[float, dict]:
        """Seed, warm shot and measured shot of a prefill point ``(b, n, k)``; returns the sample and the extras.

        The seed stores the first ``k`` tokens of every slot. It is not timed, may be chunked, and its
        cached count is not checked: it depends on what earlier points left in the cache. Each shot is ``b``
        prompts of the slot's prefix and ``n`` new tokens that no stored prompt continues with, so a shot can
        match the ``k`` seeded tokens and nothing more.

        Deviation note (warm shot).
        What differs: every prefill point runs once, as the warm shot, before the measured shot of the same
        shape is timed. Serving times a shape whether or not it ran before.
        Direction of the error: the sample leaves out what only the first batch of a shape pays, so it can be
        below a cold first batch in serving. Dynamo's vLLM self-benchmark warms a seeded prefill point the
        same way (``_bench_realseed_stage_point``).
        How to retire it: time the warm shot as a second sample if cold-shape cost is to be modelled.
        """
        batch, new, cached = point.batch_size, point.new_tokens, point.cached_tokens
        plan = self._plan
        expected_cached = grid.expected_cached_tokens(grid.PREFILL, cached, self._limits.page_size)
        if cached > 0:
            seeds = [plan.prefill_seed_prompt(RANK, slot, cached) for slot in range(batch)]
            self._stage(engine, point.benchmark_id, "seed", attempt, seeds, _ONE_OUTPUT_TOKEN, followers=False)
        line: dict = {}
        for shot in ("warm", "meas"):
            prompts = [plan.prefill_prompt(RANK, slot, cached, new) for slot in range(batch)]
            call = self._stage(engine, point.benchmark_id, shot, attempt, prompts, _ONE_OUTPUT_TOKEN, followers=True)
            self._require_cached(call, expected_cached)
            line = self._one_extend_batch(call)

        # The measured shot. Its one extend batch is the kept batch; its counters come from the recorder line.
        prefix_lens, extend_lens, extend_tokens = self._extend_counts(line)
        if sum(extend_lens) != extend_tokens:
            raise _Failure(
                f"{REASON_PREFILL_TOKENS}: the extend batch (seq {line['seq']}) reports {extend_tokens} computed "
                f"tokens and its requests sum to {sum(extend_lens)}"
            )
        observed = {
            "num_prefill_requests": len(line["rids"]),
            "sum_prefill_tokens": extend_tokens,
            "sum_prefill_kv_tokens": sum(prefix_lens),
            "num_decode_requests": 0,
            "sum_decode_kv_tokens": 0,
        }
        _compare_counters(artifact.scheduled_requests(record), observed)
        # The sample: from ``batch.launch_ts``, which SGLang sets in ``Scheduler.run_batch`` once the batch is
        # formed (srt/managers/scheduler.py), to the result being on the host. Both are readings of
        # ``time.monotonic()`` in the leader's process. Forming the batch, with its KV allocation and any
        # eviction, is outside the interval. Inside it lies what the scheduler loop does between the return of
        # ``run_batch`` and the result call, as it does in serving.
        wall_time = self._seconds(
            self._output_time(line) - self._time(line, "launch_ts"), f"extend batch seq {line['seq']}"
        )
        return wall_time, {"worker_id": self.settings.worker_id}

    def _decode_attempt(self, engine: object, point: grid.Point, record: dict, attempt: int) -> tuple[float, dict]:
        """Seed and run of a decode point ``(b, k)``; returns the sample and the extras of the record.

        Both calls send the same ``b`` prompts of ``k - 3`` tokens. The seed asks for one output token and
        stores the prompt. The run asks for seven: its extend batches compute only the part of the prompt
        that is not cached, and each of its six decode batches reads real KV. The sample is the upper median
        of decode steps 3 to 5; the deviation note for that choice is at ``DECODE_KEPT_STEP_INDICES`` in
        sglbench_artifact.py. A step is the time between the outputs of two successive batches, so it holds
        whatever the scheduler does between them, as a decode step in serving does.
        """
        batch, context = point.batch_size, point.cached_tokens
        page = self._limits.page_size
        prompts = [self._plan.decode_prompt(RANK, slot, context) for slot in range(batch)]
        self._stage(engine, point.benchmark_id, "seed", attempt, prompts, _ONE_OUTPUT_TOKEN, followers=False)
        call = self._stage(
            engine, point.benchmark_id, "run", attempt, prompts, grid.DECODE_OUTPUT_TOKENS, followers=True
        )
        self._require_cached(call, grid.expected_cached_tokens(grid.DECODE, context, page))
        extends, decodes = self._run_sequence(call, grid.expected_extend_tokens(grid.DECODE, 0, context, page))

        # Raw step i: output of decode batch i minus the output of the batch before it, which is the last
        # extend batch for step 1.
        steps = []
        previous = self._output_time(extends[-1])
        for number, line in enumerate(decodes, 1):
            output = self._output_time(line)
            steps.append(self._seconds(output - previous, f"decode step {number} (seq {line['seq']})"))
            previous = output

        # The counters of the middle kept decode batch. The decode batch with zero-based index i reads the
        # prompt and the i output tokens that already have KV, so the middle kept one reads ``k`` per request.
        # SGLang's own decode counter is not used: it sums ``batch.seq_lens_cpu`` when the result is processed
        # (srt/managers/scheduler_components/metrics_reporter.py
        # ``SchedulerMetricsReporter._build_scheduled_request_metrics``), after the step has added one token
        # to every request, so it is higher by the batch size.
        middle = artifact.DECODE_KEPT_STEP_INDICES[len(artifact.DECODE_KEPT_STEP_INDICES) // 2]
        kept = decodes[middle]
        prompt_tokens = {response["rid"]: response["prompt_tokens"] for response in call.responses}
        observed = {
            "num_prefill_requests": 0,
            "sum_prefill_tokens": 0,
            "sum_prefill_kv_tokens": 0,
            "num_decode_requests": len(kept["rids"]),
            "sum_decode_kv_tokens": sum(prompt_tokens[rid] + middle for rid in kept["rids"]),
        }
        _compare_counters(artifact.scheduled_requests(record), observed)
        extra = {artifact.DECODE_STEPS_KEY: steps, "worker_id": self.settings.worker_id}
        return artifact.decode_wall_time(steps), extra

    def _run_sequence(self, call: _Stage, expected_extend: int) -> tuple[list[dict], list[dict]]:
        """The extend batches and the six decode batches of a seven-token call, after their sequence was checked.

        The batches must be one or more extend batches and then exactly six decode batches, next to each
        other in the leader's log. The extend batches hold only requests of the call, each request in
        exactly one of them and each computing ``expected_extend`` tokens. Every decode batch holds exactly
        the requests of the call. A call above the chunk budget takes several extend batches, and SGLang
        prefers prefill to decode (srt/managers/scheduler.py ``Scheduler.get_next_batch_to_run``), so no
        decode batch runs before the last of them.
        """
        modes = [line["mode"] for _position, line in call.batches]
        extends = sum(1 for mode in modes if mode == MODE_EXTEND)
        if not extends or modes != [MODE_EXTEND] * extends + [MODE_DECODE] * artifact.DECODE_RAW_STEPS:
            found = ", ".join(f"{count} {mode}" for mode, count in _runs(modes))
            raise _Failure(
                f"{REASON_DECODE_CONTEXT}: the batches of the run are {found}, expected one or more EXTEND "
                f"and then {artifact.DECODE_RAW_STEPS} DECODE"
            )
        positions = [position for position, _line in call.batches]
        if positions != list(range(positions[0], positions[0] + len(positions))):
            raise _Failure(
                f"{REASON_DECODE_CONTEXT}: a batch of other requests lies between the batches of the run "
                f"(seq {[line['seq'] for _position, line in call.batches]})"
            )
        lines = [line for _position, line in call.batches]
        extend_lines, decode_lines = lines[:extends], lines[extends:]
        wanted = set(call.rids)
        seen: collections.Counter[str] = collections.Counter()
        for line in extend_lines:
            if not wanted.issuperset(line["rids"]):
                raise _Failure(f"{REASON_DECODE_CONTEXT}: extend batch seq {line['seq']} holds a foreign request")
            seen.update(line["rids"])
            _prefix_lens, extend_lens, extend_tokens = self._extend_counts(line)
            if any(tokens != expected_extend for tokens in extend_lens) or sum(extend_lens) != extend_tokens:
                raise _Failure(
                    f"{REASON_DECODE_CONTEXT}: extend batch seq {line['seq']} computes {extend_lens} tokens "
                    f"({extend_tokens} in all), expected {expected_extend} for each request"
                )
        split = [rid for rid in call.rids if seen[rid] != 1]
        if split:
            raise _Failure(f"{REASON_DECODE_CONTEXT}: {split[0]} is in {seen[split[0]]} extend batches of the run")
        for number, line in enumerate(decode_lines, 1):
            if sorted(line["rids"]) != sorted(call.rids):
                raise _Failure(
                    f"{REASON_BATCH_SIZE}: decode batch {number} (seq {line['seq']}) holds "
                    f"{len(line['rids'])} request(s), expected exactly the {len(call.rids)} of the run"
                )
        return extend_lines, decode_lines

    # ------------------------------------------------------------------ artifact

    def _expected_points(self) -> int:
        return 0 if self._grid is None else len(self._grid.points)

    def _own_producer_facts(self) -> dict:
        """What the driver knows about the run, for the ``producer`` block.

        One of these keys has a reader: for a cell of a plan whose model has expert layers, the host validator
        requires ``content.source`` to be ``CONTENT_TEXT_CHAINS`` (native_artifact.py,
        ``_validate_sglang_moe_producer``). The other keys of the block with a reader are the launcher's:
        ``sglang_version``, which the validator compares with the pod's provenance file
        (``validate_native_collection``), and for such a cell ``tp_size`` and ``moe.ep_size``, which it compares
        with the sizes of the cell. No AISim reader parses the rest.
        """
        limits, plan, pool = self._limits, self._plan, self._text_pool
        if plan is not None:
            source = plan.source
        elif pool is not None:
            # Before a plan exists: the source this run was started for.
            source = content.CONTENT_TEXT_CHAINS
        else:
            # Before ``execute``: what the launcher said when it made the run.
            source = self.settings.content_source
        content_facts = {
            "source": source,
            "seed": self.settings.content_seed,
            "token_range": None if plan is None else [plan.token_low, plan.token_high],
            "issued": None if plan is None else plan.stats(),
        }
        if pool is not None:
            # The record of the dataset is the launcher's. It is left out with the launcher's blocks when the
            # artifact could not be written with them.
            if self._omitted_blocks is None:
                content_facts["dataset"] = dict(self._dataset)
            # ``id_range`` is the range of ids the pool keeps, ``[low, high)`` written as two numbers like
            # ``token_range``. It holds ``token_range`` and can be wider: the ids of a text are not held to the
            # range of the ids the run makes up. ``dropped_tokens`` counts the ids of the texts outside it.
            content_facts["pool"] = {
                "id_range": [pool.token_low, pool.token_high],
                "conversations": pool.conversations,
                "tokens": pool.tokens,
                "dropped_tokens": pool.dropped_tokens,
                "dropped_conversations": pool.dropped_conversations,
            }
        return {
            "timing_metric": "scheduler_wall_time",
            "timing_clock": "process_local_monotonic",
            "staging": "one_batch_call",
            "recorder_format": RECORD_FORMAT,
            "scheduler_processes": len(self.logs.headers),
            "limits": None if limits is None else limits.as_dict(),
            "page_size": None if limits is None else limits.page_size,
            "enable_overlap": self._enable_overlap,
            "content": content_facts,
            "restaged_points": list(self._restaged),
            "self_test": list(self._self_test_passed),
            "first_token_need": None if self._first_tokens is None else dict(self._first_tokens),
        }

    def _blocks(self) -> tuple[dict, dict]:
        """The ``config`` and ``producer`` blocks as they stand now."""
        own = self._own_producer_facts()
        if self._omitted_blocks is not None:
            # The launcher's blocks could not be written. A failed artifact is written without them and
            # says so, because no artifact at all would hide the failure from the gate. The record of the
            # dataset is the launcher's too: ``_own_producer_facts`` and ``_payload`` leave it out then.
            return {}, {**own, "omitted_launcher_blocks": self._omitted_blocks}
        launcher = self.settings.producer
        for name, block in (("config", self.settings.config), ("producer", launcher)):
            if not isinstance(block, dict):
                raise TypeError(f"settings.{name} must be a dict, got {type(block).__name__}")
        clash = sorted(key for key in launcher if key in own)
        if clash:
            raise ValueError(f"settings.producer may not contain {clash}: the driver writes these keys")
        return self.settings.config, {**launcher, **own}

    def _payload(
        self,
        *,
        blocks: tuple[dict, dict] | None = None,
        failure: dict | None = None,
        elapsed: float | None = None,
    ) -> dict:
        """The rank artifact of the state now. The arguments replace single values for a trial build."""
        config, producer = self._blocks() if blocks is None else blocks
        built = self._grid
        text_run: dict = {}
        if self._dataset is not None and self.settings.mode != grid.PREFILL and self._omitted_blocks is None:
            # The ``kvwarm`` block of a decode or agg artifact of a text run names the dataset; a prefill
            # artifact has no such block. A run with seeded ids makes the call it always made.
            text_run["kvwarm_dataset"] = {key: self._dataset[key] for key in _KVWARM_DATASET_KEYS}
        return artifact.build_rank_artifact(
            dp_rank=RANK,
            dp_size=DP_SIZE,
            mode=self.settings.mode,
            run_id=self.settings.run_id,
            grid_digest=grid.grid_digest([]) if built is None else built.digest,
            expected_points=self._expected_points(),
            measurements=self._measurements,
            benchmark_elapsed_seconds=self._elapsed() if elapsed is None else elapsed,
            config=config,
            producer=producer,
            missing_phases=() if built is None else built.missing_phases,
            stop_reason=self._stop_reason,
            failure=self._failure if failure is None else failure,
            **text_run,
        )

    def _elapsed(self) -> float:
        """``benchmark_elapsed_seconds``: from the start of ``execute`` to its end, on the clock of the deadline.

        It is 0.0 for a run that was never executed. The engine build lies before it, as the model load lies
        before the elapsed time of Dynamo's vLLM self-benchmark, which starts its clock at its first benchmark
        step (``_bench_start_timing``). The soft deadline is another matter: it counts from ``started_at``.
        """
        if self._execute_started_at is None:
            return 0.0
        ended = self._clock() if self._ended_at is None else self._ended_at
        return ended - self._execute_started_at

    def payloads(self) -> list[dict]:
        """The rank artifacts of the run as it stands now: one, because this version has one rank.

        Raises ``RuntimeError`` while the run has no outcome, and what
        ``sglbench_artifact.build_rank_artifact`` raises for a state it cannot write.
        """
        if self.status is None:
            raise RuntimeError("the run has no outcome yet: execute has not returned and no failure was recorded")
        return [self._payload()]

    def write(self) -> dict:
        """Write the result files once and return ``{"status": ..., "paths": [...]}``.

        A failed run writes the rank file only. A complete or partial run writes the rank file and then the
        merged file. ``paths`` lists them in that order.

        If the artifact cannot be built or written as it stands (``TypeError`` or ``ValueError`` of the
        artifact module), the run becomes failed with that text and the failed artifact is written instead.
        If that one cannot be built either, it is written without the launcher's ``config`` and ``producer``
        blocks, and in a text run without the launcher's record of the dataset, and says so in ``producer``.
        A run without an outcome is written as failed. Other errors, such as a missing directory, are raised.

        A second call returns the result of the first and touches no file. After a call that raised, a
        second call raises ``RuntimeError``: the files of a run are never written twice.
        """
        if self._written is not None:
            return {"status": self._written["status"], "paths": list(self._written["paths"])}
        if self._write_started:
            raise RuntimeError("write was already called for this run and did not finish")
        self._write_started = True
        if self.status is None:
            self._fail(f"{REASON_INTERNAL}: the result files were written before the run had an outcome", None)
        while True:
            try:
                written = artifact.write_artifacts(self.settings.output_path, self.payloads())
                break
            except (TypeError, ValueError) as error:
                text = f"{type(error).__name__}: {error}"
                if self._failure is None:
                    self._fail(f"{REASON_INTERNAL}: {text}", None)
                elif self._omitted_blocks is None:
                    self._omitted_blocks = text
                else:
                    raise
        paths = list(written["rank_files"])
        if written["merged_output_path"] is not None:
            paths.append(written["merged_output_path"])
        self._written = {"status": self.status, "paths": paths}
        return {"status": self._written["status"], "paths": list(paths)}


def _runs(modes: list[str]) -> list[tuple[str, int]]:
    """``modes`` as runs of equal neighbours: ``[("EXTEND", 1), ("DECODE", 5)]``."""
    runs: list[tuple[str, int]] = []
    for mode in modes:
        if runs and runs[-1][0] == mode:
            runs[-1] = (mode, runs[-1][1] + 1)
        else:
            runs.append((mode, 1))
    return runs
