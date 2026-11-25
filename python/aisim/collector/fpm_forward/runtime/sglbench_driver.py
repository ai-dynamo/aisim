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

1. wait for one recorder header per scheduler process, check each one and find the leader of every
   attention-DP rank;
2. refuse engine settings the feasibility rules do not cover (as each scheduler process reports them);
3. build the limits record from a leader's header, and the grid from it (every process must report the same
   limits, and the same dtype of its KV pool);
4. check the two request caps against the largest batch of the grid;
5. compare the first tokens the run needs with the token range and, in a text run, the tokens of the text
   pool with the longest prompt of the grid; then build the content plan;
6. build one empty artifact, so that a setting that is not JSON fails before any request;
7. run the self-test, then the global warm-up;
8. stage, check and record every point in grid order. Before each point a recorder fault fails the run and
   the soft deadline ends it as partial;
9. look for recorder faults once more, and for a batch that was written late for the last call. A run the
   soft deadline ended is looked at for such a batch as well, and fails on one.

The first failure is recorded and ends the run; ``execute`` does not raise for it. ``BenchmarkRun.write``
writes one rank file for every attention-DP rank and, unless the run failed, the merged file.

A stage is one engine call: ``b`` prompts, ``b`` request ids and one output-token count. The batches the
scheduler formed for the call are read from the recorder logs by request id. The recorder is a subclass of
SGLang's ``Scheduler`` that writes one line per batch in every scheduler process; this module only reads its
files. The process that receives the requests is the leader, and all times come from its log.

Attention-DP ranks. ``settings.dp_size`` is SGLang's resolved ``attn_dp_size``, ``D`` below. With ``D`` above 1
every rank has a leader, a prefix cache and a log of its own, and a point of batch size ``b`` is ``b`` requests
on EVERY rank. A stage is then still one engine call, a balanced one: ``D * b`` prompts and request ids, ordered
slot by slot and rank inside, so that item ``i`` is the prompt of rank ``i mod D`` (the deviation note at
``BenchmarkRun._stage`` says why that places them). Every call of the run is balanced, the warm-up and the
self-test included. Every check of a stage runs once per rank, against that rank's leader log and that rank's
request ids. The kept batches of the ranks are then held against each other by step number and token counts,
never by time: clocks of different scheduler processes are not compared. Each rank gets a forward-pass record
from its own log, and the run writes one rank file per rank. With ``D`` equal to 1 none of this applies: the
calls, the checks and the files are those of a run without DP attention.

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
``internal_error``, and two that only a run with several attention-DP ranks gives:

    dp_rank_placement_mismatch         a request ran on another rank than planned, or its response names no rank
    dp_ranks_out_of_step               the ranks did not run the kept batches of a stage in the same steps

The launcher adds ``stopped`` through ``record_failure``. A reason is followed by ``: <detail>`` where there
is one.

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
                                                ScheduleBatch.convert_decode_to_extend, ScheduleBatch.copy,
                                                FINISH_LENGTH, Req.retraction_count
    srt/model_executor/forward_batch_info.py    ForwardMode
    srt/managers/tokenizer_manager.py           TokenizerManager._send_batch_request,
                                                TokenizerManager._handle_batch_output (the ``meta_info`` keys),
                                                TokenizerManager._handle_batch_request,
                                                TokenizerManager._should_use_batch_tokenization
    srt/managers/scheduler_components/
        request_receiver.py                     SchedulerRequestReceiver.recv_requests
        recv_skipper.py                         SchedulerRecvSkipper.maybe_create
        metrics_reporter.py                     SchedulerMetricsReporter._build_scheduled_request_metrics
        dp_attn.py                              prepare_mlp_sync_batch_raw, _update_gather_batch,
                                                SchedulerDPAttnAdapter.maybe_convert_decode_to_extend,
                                                SchedulerDPAttnAdapter.get_idle_batch
        output_streamer.py                      SchedulerOutputStreamer._stream_output_generation (``dp_rank``)
    srt/managers/scheduler_input_blocker.py     SchedulerInputBlocker, input_blocker_guard_region
    srt/managers/data_parallel_controller.py    DataParallelController.round_robin_scheduler,
                                                DataParallelController.init_dispatcher
    srt/utils/poll_based_barrier.py             PollBasedBarrier
    srt/arg_groups/parallel_hook.py             handle_data_parallelism
    srt/managers/schedule_policy.py             _IS_HIP
    srt/runtime_context.py                      get_parallel
    srt/model_executor/model_runner.py          ModelRunner.configure_kv_cache_dtype
    srt/mem_cache/kv_cache_dtype.py             configure_kv_cache_dtype

Batch sequences that running engines showed are kept in
``benchmarks/evidence/collector/sglbench-engine-probe-20261113``: one header per scheduler process (check_1),
one call as one extend batch (check_2), the cached counts (check_3 and check_5), a seven-token call as one
extend batch and six decode batches (check_4), and a call above the chunk budget as several extend batches
(check_9). Overlap scheduling was on in every one of those runs. None of them had DP attention.

With DP attention this module ran on engines on 2026-11-18, through sglbench_launcher.py, with SGLang 0.5.22
and 0.5.23 on two H200 GPUs: two-rank DEP cells of one model with expert layers (``Qwen/Qwen1.5-MoE-A2.7B``,
``--tp-size 2 --ep-size 2 --attn-dp-size 2``, dataset text, page size 1). Each release ran a prefill cell of six
points and a decode cell of four points, a smoke grid, through AISim's collector, and all four cells passed with
no point staged again. With SGLang 0.5.23 the two cells also ran with
``--enable-dp-attention-local-control-broadcast``, started by the launcher's own command outside the collector.
In all of them every request ran on its planned rank, every batch line of one rank had a line of the other rank
with the same ``forward_iter``, the same mode, as many requests and the same ``global_num_tokens``, whose two
entries were equal, and no rank wrote an IDLE line. With SGLang 0.5.23 the two cells of the default grid ran
through the collector as well: 2,892 prefill points and 857 decode points, none staged again. There every batch
line of the prefill cell (its seed, warm and measured calls, 8,433 steps) and of the decode cell's runs (5,999
steps) had its line on the other rank as above. The seed calls of the decode cell did not run
alike on the two ranks, which nothing here requires: 534 seed steps held different numbers of requests, and in
281 steps one rank ran a seed batch while the other wrote an IDLE line. No IDLE line lay inside a run. The
docstrings below call all of these "the runs of 2026-11-18".

Not run on an engine under DP attention: more than four ranks, and four ranks in any cell but the prefill cell
of the default grid (the launcher's own command, not validated); any failure path (a request on another rank than
planned, ranks out of step, a cache miss on one rank and its re-stage on all ranks, a decode step that SGLang
runs as an extend batch, an IDLE batch inside a run); a page size other than 1; seeded token ids; local control
broadcast with SGLang 0.5.22; a model other than the one named. These rest on the source at both tags and on
the tests. What this module does about a decode step that runs as an extend batch (``_decode_steps_as_extend``)
was written after the runs. One more smoke of the two cells with SGLang 0.5.23 passed with it; no decode step
ran as an extend batch there, so that part has run in the tests only.

The dtype of the KV pool came with record format 3, after every run named above: the driver reads
``kv_cache_stored_dtype`` from each header, requires every scheduler process to report the same one and
writes it into the ``producer`` block. It ran on engines on 2026-11-25, through sglbench_launcher.py and
AISim's collector, on H200 GPUs with one dense model (``Qwen/Qwen3-0.6B``): three prefill cells of a smoke
grid with SGLang 0.5.23 (one GPU with a bfloat16 and with an FP8 KV cache, and two GPUs at ``--tp-size 2``,
where both headers held the same dtype) and two decode cells of a smoke grid with SGLang 0.5.22 (one GPU,
one for each KV cache). All five passed. Headers that disagree, a header without the key and a value the
recorder marked unreadable have run in the tests only, as has format 3 under DP attention.

A spike of the same day ran stock SGLang at both releases with the same model and layout, without this module:
a script of its own sent SGLang's public batch call with SGLANG_ENABLE_COLOCATED_BATCH_GEN set and read the logs
of an earlier recorder script. The docstrings below say "the spike of 2026-11-18" where a statement rests on it
alone, as the cost of the input blocker in a decode step, a call of one prompt and what a prompt finds cached
on the other rank do. The files of the runs and of the spike are not kept in this repository.

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
RECORD_FORMAT = 3
RECORDER_CLASS_NAME = "SglbenchRecorderScheduler"
LOG_FILE_PREFIX = "sched-"
LOG_FILE_SUFFIX = ".jsonl"
TESTED_SGLANG_VERSIONS = ("0.5.22", "0.5.23")

# The global warm-up is one prompt of 256 tokens, as in Dynamo's vLLM self-benchmark (_bench_step_warmup).
WARMUP_PROMPT_TOKENS = 256

# What the ``producer`` block says about how a stage is sent: one batch call of ``b`` prompts without DP
# attention, one balanced batch call of ``b`` prompts for every attention-DP rank with it.
STAGING_ONE_BATCH_CALL = "one_batch_call"
STAGING_BALANCED_BATCH_CALL = "balanced_batch_call"

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
# Two more of this module's own, which only a run with several attention-DP ranks gives. A request ran on
# another rank than the one it was planned for; the ranks did not run the kept batches of a stage in the same
# scheduler steps. Neither is staged again.
REASON_DP_PLACEMENT = "dp_rank_placement_mismatch"
REASON_DP_STEP = "dp_ranks_out_of_step"

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
# The response key that names the attention-DP rank a request ran on: ``dp_rank`` of the ``meta_info``
# (srt/managers/tokenizer_manager.py ``TokenizerManager._handle_batch_output`` copies it from the output of the
# scheduler that ran the request, srt/managers/scheduler_components/output_streamer.py
# ``SchedulerOutputStreamer._stream_output_generation``). The engine object hands it over as an int, or as
# ``None`` when SGLang reported none. It is read only in a run with several ranks.
_RESPONSE_RANK_KEY = "dp_rank"

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
# The dtype of the KV pool as text: ``str(ModelRunner.kv_cache_dtype)``, which
# ``ModelRunner.configure_kv_cache_dtype`` sets in every scheduler process once it has loaded the model
# (srt/model_executor/model_runner.py; srt/mem_cache/kv_cache_dtype.py ``configure_kv_cache_dtype``). With
# ``--kv-cache-dtype auto`` it is the dtype of the model, or an FP8 dtype for a model whose quantization
# config asks for an FP8 KV cache, so only the scheduler process can tell.
_KV_STORED_DTYPE_NAME = "kv_cache_stored_dtype"

# Header values that every scheduler process must report alike.
_AGREED_VALUES = (
    ("sglang_version",),
    *(("limits", name) for name in _LIMIT_NAMES),
    *((name,) for name in _GRAPH_NAMES),
    ("ranks", "tp_size"),
    (_KV_STORED_DTYPE_NAME,),
)

# Scheduler facts of the header's ``checks`` block that must be the boolean False, with the text of the
# rejection. Each is a plain attribute of SGLang's Scheduler (srt/managers/scheduler.py), read at both tags:
# ``disable_radix_cache``, ``is_hybrid_swa``, ``is_hybrid_ssm`` (set in ``Scheduler.__init__`` from the cache
# build), ``enable_priority_scheduling`` (``Scheduler.__init__``), ``is_mixed_chunk``
# (``Scheduler.init_chunked_prefill``), ``prefill_delayer`` (``Scheduler.init_schedule_policy``),
# ``min_free_slots_delayer`` (``Scheduler.init_model_worker``) and ``recv_skipper``
# (``Scheduler.init_watch_dog_memory_saver_input_blocker``). The recorder writes the last three as
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
    ("is_hip", "only NVIDIA CUDA is supported ({name}={v}); SGLang has AMD-only admission rules"),
)

# ``checks.input_blocker`` is not in the list above: it must be False without DP attention and True with it.
# The recorder writes it as ``Scheduler.input_blocker is not None``, and SGLang builds that object in every
# scheduler process exactly when SGLANG_ENABLE_COLOCATED_BATCH_GEN is set
# (``Scheduler.init_watch_dog_memory_saver_input_blocker``; srt/managers/scheduler_input_blocker.py
# ``SchedulerInputBlocker``). Without DP attention it would hold the requests of a call back for nothing. With
# DP attention it is what releases the requests of a balanced call on every rank in one step.
_BLOCKER_WITHOUT_DP = "SGLANG_ENABLE_COLOCATED_BATCH_GEN must be unset without DP attention ({name}={v})"
_BLOCKER_WITH_DP = (
    "SGLANG_ENABLE_COLOCATED_BATCH_GEN must be set under DP attention ({name}={v}); the launcher sets it "
    "before it builds the engine"
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
    # Path of the rank 0 result file. Its directory must exist. Rank ``r`` above 0 is written beside it, at
    # ``sglbench_artifact.rank_output_path(output_path, r)``.
    output_path: str
    # New for every launch. Request ids carry its first RUN_TAG_CHARACTERS characters.
    run_id: str
    # Seed of the content plan, as a string.
    content_seed: str
    # Written into every forward-pass record as ``worker_id``.
    worker_id: str
    # How many scheduler processes must have written a recorder header: SGLang's resolved ``tp_size``, which
    # under DP attention is the width over all attention-DP ranks.
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
    # The number of attention-DP ranks: SGLang's resolved ``attn_dp_size``, 1 without DP attention. Above 1
    # every stage is a balanced call, every check runs per rank and one rank file is written per rank.
    dp_size: int = 1


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
    """Incremental reader of one recorder directory (record format 3).

    The directory holds one file per scheduler process, ``sched-<pid>.jsonl``: one JSON object per line, the
    header first, then one line per batch, and a fault line for every recorder step that raised. ``poll``
    reads what was appended since the last call. A line without its newline is kept back until the rest
    arrives.

    After a poll:

    - ``headers`` maps a file name to its header line;
    - ``batches(name)`` is the list of the batch lines of one file in file order, without the lines of mode
      ``IDLE`` that hold no request and without the lines read before the last ``forget()``;
    - ``faults`` lists the fault lines as ``{"file", "where", "error"}``.

    Anything that is not a line of the format is added to ``faults`` with ``where`` set to ``"reader"`` and
    never repaired: a line that is not a JSON object, an unknown line type, a header that is not the first
    line, a batch line before the header, a batch line without ``mode`` and ``rids``, and a ``seq`` that does
    not count the batch lines of its file from 1. Such a line is not kept. The caller fails the run on any
    entry. The format number itself is in the header, which the caller checks; the other keys of a batch line
    are checked where they are used.
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
        # ``ScheduleBatch.prepare_for_idle``). Such a line belongs to no stage. It still has its ``seq``, so
        # a caller sees from the ``seq`` of the lines it does get that one was left out between them. Idle
        # lines were seen in the spike of 2026-11-18, after a call of one prompt, and in the runs of that day
        # among the seed steps of the default grid's decode cell; none lay inside a run (module docstring).
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


def _decode_steps_as_extend(lines: list[dict]) -> list[dict]:
    """The lines of mode EXTEND among ``lines`` whose requests had all been extended by an earlier one of them.

    ``lines`` are the batch lines of one attention-DP rank that hold requests of one call, in log order. Only
    a run with several attention-DP ranks asks for this. Under DP attention SGLang turns the decode batch of
    a rank into an extend batch when another rank runs an extend batch in the same step, no rank idles and
    every rank reports that its prefill graphs can replay the step
    (srt/managers/scheduler_components/dp_attn.py ``SchedulerDPAttnAdapter.maybe_convert_decode_to_extend``;
    the report is the vote in ``prepare_mlp_sync_batch_raw``). Whether a given step is converted therefore
    depends on the engine, and the callers take either. A converted batch keeps its requests and its token
    count of the step. Its mode becomes EXTEND, every request computes one token after everything it holds so
    far, and ``extend_num_tokens`` is the number of requests (srt/managers/schedule_batch.py
    ``ScheduleBatch.convert_decode_to_extend``). In the log of the rank that is a line of mode EXTEND whose
    requests an earlier extend batch of the call had already extended: a decode step of those requests under
    another mode.

    A request that is chunked, with no other request beside its later part, gives such a line as well. So a
    caller concludes nothing from one by itself: ``BenchmarkRun._missing_batches`` only ends its wait on it,
    and ``BenchmarkRun._run_sequence`` only words by it a failure that the modes have already caused.

    Read in the source at both tags. No engine showed such a line.
    """
    extended: set[str] = set()
    found = []
    for line in lines:
        if line["mode"] != MODE_EXTEND:
            continue
        if extended.issuperset(line["rids"]):
            found.append(line)
        extended.update(line["rids"])
    return found


def _compare_counters(expected: dict[str, int], observed: dict[str, int], where: str = "") -> None:
    """Fail unless the counters observed for the kept batch are the ones the artifact derives from the point.

    ``where`` names the attention-DP rank of the batch in a run with several ranks; it is empty otherwise.
    """
    for key, reason in _COUNTER_REASONS:
        if observed[key] != expected[key]:
            detail = f"{where}{key} is {observed[key]} in the kept batch, the point plans {expected[key]}"
            if reason is None:
                raise _CacheMiss(detail)
            raise _Failure(f"{reason}: {detail}")


@dataclasses.dataclass
class _RankCall:
    """The requests of one engine call that belong to one attention-DP rank, and what that rank's log holds."""

    rank: int
    # The file name of the rank's leader log.
    log: str
    rids: list[str]
    prompts: list[list[int]]
    responses: list[dict]
    # (position among the batch lines of ``log`` since the call, batch line) of every line that holds a
    # request of the rank.
    batches: list[tuple[int, dict]]


@dataclasses.dataclass
class _Stage:
    """One engine call: its output-token count and its part on every attention-DP rank, in rank order."""

    output_tokens: int
    ranks: list[_RankCall]


# ---------------------------------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------------------------------


class BenchmarkRun:
    """One benchmark run: its settings, its state and its artifact.

    ``recorder_dir`` is the directory the recorder of this run writes to. ``started_at`` is a reading of
    ``clock`` taken when the launcher started: the soft deadline counts from it, so the engine build is
    inside the deadline. ``clock`` and ``sleep`` are ``time.monotonic`` and ``time.sleep`` unless a test
    replaces them.

    The constructor raises for the four settings without which not even a failed artifact can be written:
    the mode, the output path, the run id and the number of attention-DP ranks. Every other setting is checked
    by ``execute``, which records a bad one as a failure.

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
        # The number of rank files, and of the ranks every stage is sent to.
        if not _is_int(settings.dp_size):
            raise TypeError(
                f"settings.dp_size must be an int, got {type(settings.dp_size).__name__}: {settings.dp_size!r}"
            )
        if settings.dp_size < 1:
            raise ValueError(f"settings.dp_size must be >= 1, got {settings.dp_size}")
        self._dp_size = settings.dp_size
        # These raise for a value that no result file and no request id can carry, for every rank.
        for rank in range(self._dp_size):
            artifact.rank_output_path(settings.output_path, rank)
        artifact.merged_output_path(artifact.rank_output_path(settings.output_path, 0))
        self._run_tag = settings.run_id[:RUN_TAG_CHARACTERS]
        content.request_id(self._run_tag, 0, "free", 0, self._dp_size - 1, 0)

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
        self._on_grid: Callable[[], object] | None = None
        # The leader log of every attention-DP rank, in rank order; empty until the headers were read. The
        # followers are the logs of the other scheduler processes, which exist only without DP attention.
        self._leaders: list[str] = []
        self._followers: list[str] = []
        self._enable_overlap: bool | None = None
        # The dtype of the KV pool that every scheduler process reported, None until the headers were checked.
        self._kv_cache_stored_dtype: str | None = None
        self._measurements: list[dict] = []
        self._restaged: list[int] = []
        self._current_point: dict | None = None
        self._free_calls = 0
        # The last call that was checked: its point, its output tokens and, for every rank, its request ids
        # and how many lines of the rank's leader log held them.
        self._checked: tuple[int, int, list[tuple[frozenset[str], int]]] | None = None
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

    def prefill_pool_cut(self) -> grid.PoolCut | None:
        """What the KV pool takes from the generated prefill block of this run, or ``None``.

        ``None`` also before the grid exists, for a decode run and for a run with a points manifest: a manifest
        point that does not fit the pool is refused by name, and the decode block is not looked at
        (sglbench_grid.py, above ``PoolCut``, says why). Nothing of this goes into the artifact; the launcher
        prints it.
        """
        if self._limits is None or self._grid is None or self._grid.source != grid.SOURCE_GENERATED:
            return None
        if self.settings.mode not in (grid.PREFILL, grid.AGG):
            return None
        return grid.prefill_pool_cut(self._limits, self.settings.caps)

    # ------------------------------------------------------------------ execute

    def execute(
        self,
        engine: object,
        *,
        token_low: int,
        token_high: int,
        text_pool: content.TextPool | None = None,
        dataset: dict | None = None,
        on_grid: Callable[[], object] | None = None,
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

        ``on_grid`` is called once, without arguments, when the grid is planned and before any request is
        sent, also when the grid misses a phase and the run then fails. It is the caller's chance to say
        something about the grid while the run still has all its time before it (the launcher's lines about
        ``prefill_pool_cut``). It is not called when no grid could be planned. An exception it raises is a
        failure of the run like any other.

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
        self._on_grid = on_grid
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
        if self._on_grid is not None:
            self._on_grid()
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
                # No call follows the last one, so the look for a batch that was written late for it is made
                # here, as at the end of the sweep. The stop reason is set first: a late batch then fails the
                # run, and the failed artifact still names the deadline and keeps the rows that completed.
                self._refuse_late_batches()
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
        """Wait for one valid header per scheduler process and find the leader of every attention-DP rank."""
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
        self._leaders = self._leaders_by_rank(headers, leaders)
        self._followers = [name for name in headers if name not in self._leaders]
        overlap = headers[self._leaders[0]].get("enable_overlap")
        self._enable_overlap = overlap if type(overlap) is bool else None
        return headers

    def _leaders_by_rank(self, headers: dict[str, dict], leaders: list[str]) -> list[str]:
        """The leader log of every attention-DP rank, in rank order, from the headers that say ``leader``.

        Without DP attention exactly one process is a leader. With it every rank must have exactly one: SGLang
        gives a request socket to every process whose ``attn_tp_rank`` is 0 (srt/managers/scheduler.py
        ``Scheduler.init_ipc_channels``), one per attention-DP rank, and the DP controller sends that process
        the requests of its rank. The rank is ``ranks.attn_dp_rank`` of the header (srt/runtime_context.py
        ``get_parallel``). In the runs of 2026-11-18 (module docstring) both scheduler processes of every
        engine were leaders, with ranks 0 and 1, and the requests whose responses named a rank were found in
        the log of that rank's leader. A rank without a leader, or with two, was not run on an engine.
        """
        size = self._dp_size
        if size == 1:
            if len(leaders) != 1:
                raise _Failure(f"{REASON_HEADER}: {len(leaders)} leader header(s), expected exactly one: {leaders}")
            return leaders
        by_rank: dict[int, list[str]] = {}
        for name in leaders:
            rank = _header_value(name, headers[name], "ranks", "attn_dp_rank")
            if not _is_int(rank) or not 0 <= rank < size:
                raise _Failure(
                    f"{REASON_HEADER}: {name}: attn_dp_rank is {_shown(rank)}, expected one of {list(range(size))}"
                )
            by_rank.setdefault(rank, []).append(name)
        for rank in sorted(by_rank):
            if len(by_rank[rank]) != 1:
                raise _Failure(
                    f"{REASON_HEADER}: {len(by_rank[rank])} leader headers for attention-DP rank {rank}, "
                    f"expected exactly one: {by_rank[rank]}"
                )
        missing = [rank for rank in range(size) if rank not in by_rank]
        if missing:
            raise _Failure(
                f"{REASON_HEADER}: no leader header for attention-DP rank(s) {missing}, expected exactly one "
                f"for each of {list(range(size))}: the leaders are {leaders}"
            )
        return [by_rank[rank][0] for rank in range(size)]

    def _check_headers(self, headers: dict[str, dict]) -> None:
        """Refuse what the feasibility rules and the timing do not cover, as every scheduler process reports it."""
        for name, header in headers.items():
            self._check_header(name, header)
        if self._dp_size > 1 and self._followers:
            # A scheduler process without a request socket is, under DP attention, a TP rank inside an
            # attention-DP rank (``attn_tp_size`` above 1). It would run the batches of its rank's leader. The
            # comparison of a follower with its leader (``_missing_batches``) was only seen on engines
            # without DP attention, so the layout is refused here as the launcher refuses it before the
            # engine is built.
            raise _Failure(
                f"{REASON_REJECTED}: attention TP above 1 inside an attention-DP rank is not supported: "
                f"{self._followers} hold no request socket"
            )
        for path in _AGREED_VALUES:
            values = [_header_value(name, header, *path) for name, header in headers.items()]
            if any(value != values[0] for value in values[1:]):
                raise _Failure(
                    f"{REASON_REJECTED}: scheduler processes disagree on {path[-1]}: {json.dumps(values, default=repr)}"
                )
        # Every scheduler process holds a KV pool of its own, and all of them have now reported the same
        # dtype, so the one of the first leader is the dtype of the run. The driver is not told the cell's
        # KV-cache label: the host holds this value against it (native_artifact.py,
        # ``_validate_sglang_kv_cache_dtype``).
        leader = self._leaders[0]
        self._kv_cache_stored_dtype = _header_value(leader, headers[leader], _KV_STORED_DTYPE_NAME)

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
        # The recorder writes the text of a torch dtype, for example "torch.bfloat16", and marks a value that
        # is no torch dtype as one it could not read. Which dtypes a cell admits is the host's rule.
        stored = read(_KV_STORED_DTYPE_NAME)
        if type(stored) is not str or not stored:
            raise _Failure(f"{REASON_HEADER}: {name}: {_KV_STORED_DTYPE_NAME} is {stored!r}, expected a string")
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
        # srt/runtime_context.py ``get_parallel``. The attention-DP layout must be the one the run was set up
        # for: ``settings.dp_size`` decides how every stage is sent, how it is checked and how many rank files
        # are written. ``attn_dp_enabled`` is derived from the size and must go with it. Pipeline parallelism
        # runs another event loop.
        planned = self._dp_size
        dp_size, pp_size = read("ranks", "attn_dp_size"), read("ranks", "pp_size")
        enabled = read("ranks", "attn_dp_enabled")
        if not (_is_int(dp_size) and dp_size == planned) or enabled is not (planned > 1):
            reject(
                f"the scheduler reports attn_dp_size={_shown(dp_size)} and attn_dp_enabled={_shown(enabled)}, "
                f"the run was set up for {planned} attention-DP rank(s)"
            )
        if not (_is_int(pp_size) and pp_size == 1):
            reject(f"pipeline parallelism is not supported (pp_size={_shown(pp_size)})")
        read("ranks", "tp_size")
        read("checks", "max_queued_requests")
        # The input blocker: off without DP attention, on with it (see the comment at ``_BLOCKER_WITH_DP``).
        # It is looked at after the layout, so that a header of another layout is refused for its layout.
        blocker = read("checks", "input_blocker")
        if blocker is not (planned > 1):
            text = _BLOCKER_WITH_DP if planned > 1 else _BLOCKER_WITHOUT_DP
            reject(text.format(name="input_blocker", v=_shown(blocker)))

    def _limits_of(self, headers: dict[str, dict]) -> grid.Limits:
        """The limits record of the run, read from the header of the first leader.

        Every scheduler process must have reported the same limits and graph sizes by now
        (``_AGREED_VALUES``), so the record holds for every attention-DP rank: each rank has its own KV pool
        and admission, and a point is sent to all of them. What a scheduler reports is already its own share:
        SGLang divides the chunk budget by the number of ranks when it resolves its arguments
        (srt/arg_groups/parallel_hook.py ``handle_data_parallelism``; 4,096 of the default 8,192 on both
        ranks in the runs of 2026-11-18). Dynamo's vLLM self-benchmark takes the smallest value of each
        limit over the ranks instead (``_BenchmarkSynchronizer.negotiate_capacity``). Here ranks that differ
        reject the run, because a point is checked on every rank against one record.
        """
        leader = self._leaders[0]
        header = headers[leader]
        record = {name: _header_value(leader, header, "limits", name) for name in _LIMIT_NAMES}
        record.update({name: _header_value(leader, header, name) for name in _GRAPH_NAMES})
        try:
            return grid.Limits.from_mapping(record)
        except (TypeError, ValueError) as error:
            raise _Failure(f"{REASON_HEADER}: {leader}: limits: {error}") from error

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
        ``Scheduler._abort_on_queued_limit``). Both are caps of one scheduler, and the batch of a point is the
        batch of one attention-DP rank, so the comparison is the same with any number of ranks.
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

        Every number is the need of ONE attention-DP rank. The plan allocates the first tokens of each rank on
        their own from the whole token range, and every rank is sent the same stages, so the need of a rank
        is the need of the run whatever the number of ranks.
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
        """Fail when a leader wrote a batch for a call after that call had been checked.

        With overlap scheduling SGLang launches the next batch before it processes the result of the batch
        before it (srt/managers/scheduler.py ``Scheduler.event_loop_overlap``). A batch beyond the ones a call
        must have is therefore written after the response of the call, when the call has been checked. It is
        looked for here, among the lines read since the last call, before they are dropped: a line that holds
        requests of the last call beyond the lines that call was checked on, or requests of an earlier call.
        Every rank's leader log is looked at with the requests of that rank; a request of the run in the log
        of another rank than its own is found the same way.

        It runs before every call, and once after the last call of a run: at the end of the sweep, and at the
        soft deadline when that ends the run.
        """
        if self._checked is None:
            return
        benchmark_id, output_tokens, held = self._checked
        late = []
        for rank, (wanted, count) in enumerate(held):
            lines = self.logs.batches(self._leaders[rank])
            late.extend(
                (rank, line, benchmark_id, output_tokens > 1) for _position, line in _holding(lines, wanted)[count:]
            )
            for line in lines:
                for rid in line["rids"]:
                    parts = None if rid in wanted else content.parse_request_id(rid)
                    if parts is not None and parts["run_id"] == self._run_tag:
                        late.append((rank, line, parts["benchmark_id"], parts["stage"] == "run"))
                        break
        if not late:
            return
        rank, line, owner, decode_run = late[0]
        if self._current_point is not None and self._current_point["benchmark_id"] != owner:
            self._current_point = None  # the batch belongs to a call outside the point that is being staged
        raise _Failure(
            f"{REASON_DECODE_CONTEXT if decode_run else REASON_PREFILL_TOKENS}: {self._where(rank)}a "
            f"{line['mode']} batch (seq {line['seq']}) holds requests of a call that had already been checked"
        )

    def _where(self, rank: int) -> str:
        """The start of a detail that is about one attention-DP rank: empty in a run with one rank."""
        return "" if self._dp_size == 1 else f"rank {rank}: "

    def _raise_on_fault(self) -> None:
        faults = self.logs.faults
        if faults:
            first = faults[0]
            more = f" (and {len(faults) - 1} more)" if len(faults) > 1 else ""
            raise _Failure(f"{REASON_RECORDER_FAULT}: {first['file']}: {first['where']}: {first['error']}{more}")
        # A log file that appears after the headers were checked is a scheduler process nothing was checked for.
        expected = self.settings.expected_scheduler_processes
        if self._leaders and len(self.logs.headers) != expected:
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
        prompts: list[list[list[int]]],
        output_tokens: int,
        *,
        followers: bool,
        expected_cached: int | None = None,
    ) -> _Stage:
        """Send one engine call, check every response and collect the batches of the call from the logs.

        ``prompts`` holds one list of prompts for every attention-DP rank, in rank order, all of one length.

        ``expected_cached`` is given for the run of a decode point: the cached count every request of it must
        report. With one rank it changes nothing. With several ranks the responses are held against it before
        the logs are waited for. A response with another count means that a rank missed its cache and extends
        for longer than the others, so the ranks do not run the planned steps together: SGLang can run decode
        steps of the other ranks as extend batches meanwhile (``_decode_steps_as_extend``), and the log of a
        rank need not hold six lines of mode DECODE for the call. Nothing of such a call is kept, so only its
        extend batches are waited for. The caller's ``_require_cached`` finds the wrong count next, and the
        point is staged again on all ranks. A miss on one rank only was not run on an engine.

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

        Deviation note (balanced batch call under DP attention).
        What differs: with ``D`` attention-DP ranks a stage is ``b`` requests on every rank, and they go out
        as one call of ``D * b`` prompts, ordered slot by slot and rank inside. Under DP attention SGLang does
        not send a list call as one message: it sends the items one by one (srt/managers/tokenizer_manager.py
        ``TokenizerManager._handle_batch_request``; ``TokenizerManager._should_use_batch_tokenization`` is
        false there unless the server field ``enable_tokenizer_batch_encode`` is on, which the launcher
        refuses for such a run). The DP controller deals requests that name no rank to the ranks in turn
        (srt/managers/data_parallel_controller.py ``DataParallelController.round_robin_scheduler``), so item
        ``i`` runs on rank ``i mod D`` as long as every call that reaches the engine holds a multiple of ``D``
        requests. That is why every call of the run is balanced, the warm-up and the self-test too. With
        SGLANG_ENABLE_COLOCATED_BATCH_GEN set, which the launcher does for such a run, SGLang puts a block
        and an unblock message of its own around those sends (srt/managers/scheduler_input_blocker.py
        ``input_blocker_guard_region``); the controller passes both to every rank
        (``DataParallelController.init_dispatcher``). Each rank's leader holds the requests back until the
        unblock has arrived on all ranks (``SchedulerInputBlocker``, srt/utils/poll_based_barrier.py
        ``PollBasedBarrier``), so every rank forms its batch from all its ``b`` requests in the same scheduler
        step. The driver sends no block message, names no rank in a request and uses no SGLang name beyond
        the batch call. Serving has no such blocker: requests arrive one at a time and every rank batches
        what happens to wait.
        Dynamo's vLLM self-benchmark has one driver inside each rank's scheduler and aligns the ranks with a
        barrier of its own (``_BenchmarkSynchronizer``), whose wait ends before the timed step starts. Here
        one driver stages all ranks and SGLang's blocker is the barrier.
        Direction of the error: for a prefill batch none known. For decode, the blocker asks all ranks
        whether a release is due in every scheduler step, for the whole life of the engine
        (``SchedulerInputBlocker.handle``), and serving does not. That lies inside every timed decode step
        and can only make it longer. Measured in the spike of 2026-11-18 on the model named in the module
        docstring, with free-running requests of 48 output tokens at context 1,024, the variable on and off:
        a steady decode step of 32 requests per rank took 6.45 to 6.60 ms in all eight launches, with no
        difference; one of 4 requests per rank took 3.32 to 3.50 ms, and 3.72 ms in one launch with the
        variable on.
        Sources: read at both tags. In the runs of 2026-11-18 (module docstring) every balanced call of this
        module placed item ``i`` on rank ``i mod 2``, up to 8,192 requests in one call, and both shots of
        every prefill point ran as one extend batch with exactly its ``b`` requests on each rank; either
        check would have failed the run otherwise. Seen in the spike only, without this module: calls of one
        prompt ran on the ranks in turn, and the cost above.
        How to retire it: the staging has nothing to retire. The cost in a decode step goes when SGLang
        releases a batch on all ranks without a question in every step.

        The engine object adds the request fields that are the same for every request (temperature 0,
        ``ignore_eos``, streaming). The driver chooses the three that differ: the prompts and the request
        ids, which become ``input_ids`` and ``rid`` of srt/managers/io_struct.py ``GenerateReqInput``, and
        the output-token count, ``max_new_tokens`` of srt/sampling/sampling_params.py ``SamplingParams``.
        """
        size = self._dp_size
        batch = len(prompts[0]) if prompts else 0
        if len(prompts) != size or any(len(of_rank) != batch for of_rank in prompts):
            raise ValueError(
                f"a stage needs the same number of prompts for each of the {size} attention-DP rank(s), "
                f"got {[len(of_rank) for of_rank in prompts]}"
            )
        parts = [
            _RankCall(
                rank,
                self._leaders[rank],
                [content.request_id(self._run_tag, benchmark_id, stage, attempt, rank, slot) for slot in range(batch)],
                prompts[rank],
                [],
                [],
            )
            for rank in range(size)
        ]
        # The call: slot by slot and rank inside, so that item i is a request of rank i % size. With one rank
        # these are the prompts and the request ids of that rank, in slot order.
        call_rids = [parts[rank].rids[slot] for slot in range(batch) for rank in range(size)]
        call_prompts = [parts[rank].prompts[slot] for slot in range(batch) for rank in range(size)]
        # The lines of earlier calls are read and dropped: the request ids of this call are new, so its
        # batches can only be among the lines that follow.
        self.logs.poll()
        self._refuse_late_batches()
        self.logs.forget()
        try:
            responses = engine.generate(call_prompts, call_rids, output_tokens)
        except Exception as error:
            raise _Failure(f"{REASON_ENGINE_CALL}: {type(error).__name__}: {error}") from error
        self._check_responses(call_rids, call_prompts, output_tokens, responses)
        for part in parts:
            part.responses = responses[part.rank :: size]
        # With several ranks, and only for a call that was given its planned cached count: a rank missed its
        # cache. The extend batches of the call are then waited for and no decode batch.
        missed = (
            size > 1
            and expected_cached is not None
            and any(response["cached_tokens"] != expected_cached for response in responses)
        )
        self._wait_for_batches(parts, _ONE_OUTPUT_TOKEN if missed else output_tokens, followers)
        self._checked = (benchmark_id, output_tokens, [(frozenset(part.rids), len(part.batches)) for part in parts])
        return _Stage(output_tokens, parts)

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

        In a run with several attention-DP ranks every request must also have run on its planned rank
        (``_check_placement``). That is looked at before the counts, which mean nothing on another rank.
        """
        if not isinstance(responses, list) or len(responses) != len(rids):
            raise _Failure(f"{REASON_ENGINE_CALL}: the call returned {responses!r:.200} for {len(rids)} request(s)")
        for rid, response in zip(rids, responses, strict=True):
            if not isinstance(response, dict) or response.get("rid") != rid or "finish_type" not in response:
                raise _Failure(f"{REASON_ENGINE_CALL}: {rid}: the response is {response!r:.200}")
            for key in _RESPONSE_INT_KEYS:
                if not _is_int(response.get(key)):
                    raise _Failure(f"{REASON_ENGINE_CALL}: {rid}: {key} is {response.get(key)!r}")
        if self._dp_size > 1:
            self._check_placement(rids, responses)
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

    def _check_placement(self, rids: list[str], responses: list[dict]) -> None:
        """In a run with several attention-DP ranks: item ``i`` of a call must have run on rank ``i mod D``.

        The call names no rank, so where a request runs rests on the order in which SGLang's DP controller
        deals requests out (see the deviation note at ``_stage``). The response says where it did run:
        ``dp_rank`` of the ``meta_info`` is the rank of the scheduler that sent the output. A request on
        another rank than its seed finds nothing cached there, and a point with another number of requests on
        a rank than planned is not the planned point, so a response that names another rank, or none, fails
        the run. The point is not staged again: the order of the controller does not come back by itself.
        In the runs of 2026-11-18 (module docstring) every response named its planned rank, so this module
        never saw a request on another rank. That a prompt finds nothing cached on the other rank was seen
        in the spike of 2026-11-18, which sent prompts to the rank that had not stored their prefix and got
        0 cached tokens.
        """
        size = self._dp_size
        for position, (rid, response) in enumerate(zip(rids, responses, strict=True)):
            planned = position % size
            ran_on = response.get(_RESPONSE_RANK_KEY)
            if not _is_int(ran_on) or ran_on != planned:
                found = "no dp_rank" if ran_on is None else f"dp_rank {ran_on!r}"
                raise _Failure(
                    f"{REASON_DP_PLACEMENT}: {rid}: the engine reports {found}, the request was planned for "
                    f"rank {planned}"
                )

    def _wait_for_batches(self, parts: list[_RankCall], output_tokens: int, followers: bool) -> None:
        """Read the logs until they hold the batches of a call on every rank; give up after ``log_wait_seconds``.

        The recorder writes a batch line before it lets the stock scheduler process the result
        (sglbench_recorder.py), and a leader sends the output of a batch from there. So the lines a leader
        wrote for a call are in its file when the engine call returns, on every rank. The other scheduler
        processes run the same batches and write from their own processes in no order with the call, so their
        files are waited for. A fault line that turns up meanwhile fails the run at once.

        The lines of each rank are left in ``batches`` of its part.
        """
        deadline = self._clock() + self.settings.log_wait_seconds
        while True:
            self.logs.poll()
            self._raise_on_fault()
            missing = None
            for part in parts:
                part.batches = _holding(self.logs.batches(part.log), frozenset(part.rids))
                missing = self._missing_batches(part, output_tokens, followers)
                if missing is not None:
                    break
            if missing is None:
                return
            if self._clock() >= deadline:
                raise _Failure(f"{REASON_NO_BATCHES}: {missing} after {self.settings.log_wait_seconds} s")
            self._sleep(POLL_SECONDS)

    def _missing_batches(self, part: _RankCall, output_tokens: int, followers: bool) -> str | None:
        """What the logs still lack for one rank of a call, or ``None`` when they hold every batch it must have.

        A call of ``m`` output tokens must have an extend batch for every request, which yields the first
        token, and ``m - 1`` decode batches. With overlap scheduling on there is no decode batch beyond
        them: SGLang drops a request from the next decode batch when the result still in the queue will
        finish it (srt/managers/scheduler.py ``Scheduler._reqs_finishing_inflight``, switched on by
        ``Scheduler.event_loop_overlap``).

        With several attention-DP ranks a decode step that SGLang ran as an extend batch counts among the
        decode batches here (``_decode_steps_as_extend``). It is still a step of the call, so the logs lack
        nothing and the wait ends; what the ranks ran is then judged by the checks of the stage, which keep
        no such line: ``_one_extend_batch`` and ``_run_sequence`` take a request in two extend batches for a
        failure, and ``_run_sequence`` wants six lines of mode DECODE. A wait that went on to its deadline
        would report a run whose ranks were out of step as a log without its batches. With one rank the
        count is that of the DECODE lines alone, as it always was.
        """
        batches, wanted = part.batches, frozenset(part.rids)
        extended: set[str] = set()
        decodes = 0
        for _position, line in batches:
            if line["mode"] == MODE_EXTEND:
                extended.update(line["rids"])
            elif line["mode"] == MODE_DECODE:
                decodes += 1
        if self._dp_size > 1:
            decodes += len(_decode_steps_as_extend([line for _position, line in batches]))
        lacking = len(wanted - extended)
        if lacking:
            return f"{part.log} holds no extend batch for {lacking} of {len(wanted)} request(s)"
        if decodes < output_tokens - 1:
            return f"{part.log} holds {decodes} of the {output_tokens - 1} decode batch(es)"
        if followers:
            # With a TP size above 1 every TP rank runs every batch: rank 0 reads the tokenizer socket and
            # broadcasts each request to the other ranks, which schedule it the same way
            # (srt/managers/scheduler_components/request_receiver.py ``SchedulerRequestReceiver.recv_requests``,
            # read in the source at both tags). Seen on running engines on 2026-11-16, in two launches at
            # TP 2, one with SGLang 0.5.22 and one with 0.5.23: both ranks wrote the same batches and every
            # call passed this check. The files of those launches are not kept in this repository; the probe
            # driver kept in benchmarks/evidence/collector/sglbench-engine-probe-20261113 read the leader's
            # log only. Each of the leader's batches must be in every other log with the same mode and the
            # same requests. A run with several attention-DP ranks has no such log: ``_check_headers``
            # refuses one.
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
        """Every request must report the planned number of cached tokens (``cached_tokens`` of ``meta_info``).

        The requests of every attention-DP rank are looked at here, before any batch of the call is checked,
        so that a wrong count on any rank is found as such and the point is staged again on all ranks. A rank
        that missed its cache computes more than planned, and what the other ranks then log for the shared
        steps would otherwise be read as their own fault.

        The caller comes here from ``_stage``, which has by then waited for the batch lines of the call. For
        a one-token call those are its extend batches, which every request has whatever it found cached. For
        the run of a decode point with several ranks ``_stage`` is given the planned count and holds the
        responses against it before it waits: after a miss it waits for the extend batches only, so that
        the miss reaches this method and not the deadline of the log wait.
        """
        responses = [response for part in call.ranks for response in part.responses]
        wrong = [response for response in responses if response["cached_tokens"] != expected]
        if wrong:
            raise _CacheMiss(
                f"{len(wrong)} of {len(responses)} request(s) report another cached count than the planned "
                f"{expected}: {wrong[0]['rid']} reports {wrong[0]['cached_tokens']}"
            )

    def _one_extend_batch(self, part: _RankCall) -> dict:
        """The one extend batch of one rank of a one-token call, which must hold exactly the rank's requests."""
        where = self._where(part.rank)
        lines = [line for _position, line in part.batches]
        others = [line for line in lines if line["mode"] != MODE_EXTEND]
        if others:
            raise _Failure(
                f"{REASON_PREFILL_TOKENS}: {where}a {others[0]['mode']} batch (seq {others[0]['seq']}) holds "
                "requests of a one-token call"
            )
        counts = collections.Counter(rid for line in lines for rid in line["rids"])
        chunked = [rid for rid in part.rids if counts[rid] > 1]
        if chunked:
            raise _Failure(
                f"{REASON_PREFILL_TOKENS}: {where}{chunked[0]} is in {counts[chunked[0]]} extend batches: "
                "the request was chunked"
            )
        if len(lines) != 1:
            raise _Failure(
                f"{REASON_BATCH_SIZE}: {where}the call ran as {len(lines)} extend batches of "
                f"{[len(line['rids']) for line in lines]} requests, expected one of {len(part.rids)}"
            )
        if sorted(lines[0]["rids"]) != sorted(part.rids):
            foreign = sorted(set(lines[0]["rids"]) - set(part.rids))
            raise _Failure(
                f"{REASON_BATCH_SIZE}: {where}the extend batch (seq {lines[0]['seq']}) holds "
                f"{len(lines[0]['rids'])} request(s), {len(foreign)} of them foreign, expected exactly the "
                f"{len(part.rids)} of the call"
            )
        return lines[0]

    def _extend_counts(self, log: str, line: dict) -> tuple[list[int], list[int], int]:
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
                f"{REASON_RECORDER_FAULT}: {log}: extend batch seq {line['seq']} has no usable "
                "prefix_lens, extend_lens and extend_num_tokens"
            )
        return prefix_lens, extend_lens, extend_tokens

    def _output_time(self, log: str, line: dict) -> float:
        """``out_ts`` of a batch line of the leader log ``log``: the time its result was on the host.

        The recorder takes it after the copy event of the result (srt/managers/utils.py
        ``GenerationBatchResult.copy_done``) and before the stock ``Scheduler.process_batch_result`` runs. A
        line that says the result had no such event is not a time after the GPU work and is refused.

        All times are a leader's, on that process's monotonic clock. Without DP attention and with a TP size
        above 1 the other TP ranks run the same batches on clocks of their own; the leader is the process
        whose loop forms the batches and sends their output. With several attention-DP ranks each rank's
        times come from its own leader, and a time of one rank is never set against a time of another.
        """
        if line.get("copy_done") is not True:
            raise _Failure(
                f"{REASON_RECORDER_FAULT}: {log}: batch seq {line['seq']} was stamped without a copy "
                f"event (copy_done={line.get('copy_done')!r})"
            )
        return self._time(log, line, "out_ts")

    def _time(self, log: str, line: dict, key: str) -> float:
        value = line.get(key)
        if type(value) is not float or not math.isfinite(value):
            raise _Failure(f"{REASON_RECORDER_FAULT}: {log}: batch seq {line['seq']} has {key}={value!r}")
        return value

    def _seconds(self, log: str, value: float, what: str) -> float:
        if not value > 0:
            raise _Failure(f"{REASON_RECORDER_FAULT}: {log}: {what} took {value!r} s on the scheduler's clock")
        return value

    def _require_same_step(self, kept: list[list[dict]]) -> None:
        """In a run with several attention-DP ranks: the kept batches of a stage ran in the same steps on all.

        ``kept`` holds, for every rank in rank order, the batch lines of that rank that the stage is judged
        by: the one extend batch of a prefill shot, or every extend batch and the six decode batches of a
        decode run. Each of them has passed the checks of its own rank. They are compared line by line:

        - the ranks must hold the same number of them;
        - ``forward_iter`` must be equal on all ranks. It counts the batches a scheduler process ran, idle
          ones included, and under DP attention every rank runs one batch in every step in which any rank
          has tokens (srt/managers/scheduler_components/dp_attn.py ``prepare_mlp_sync_batch_raw``), so one
          step has one number on all ranks;
        - ``global_num_tokens`` must be the same list on all ranks, with one entry per rank and the same
          number in every entry. SGLang writes into it what each rank runs in the step
          (``_update_gather_batch``): the computed tokens of an extend batch, the requests of a decode batch,
          0 for a rank that idles. Every rank is sent the same shapes, so the entries are equal when every
          rank runs the same part of the point in the step. Unequal entries mean that a rank ran something
          else in it, or nothing.

        Without this a rank that was released one step late would still show one batch with exactly its
        requests in its own log, and the timed step of the other ranks would have held only their own tokens
        in the layers that all ranks share. The same holds for a rank that still extends while the others
        decode: its batches of the run are then one step behind theirs.

        Deviation note (alignment is checked after the fact).
        What differs: Dynamo's vLLM self-benchmark lets a rank start its timed step only when all ranks have
        reported the same point (``_BenchmarkSynchronizer.synchronize``). Here SGLang's input blocker
        releases the ranks, and the driver reads from the logs afterwards whether they ran together.
        Direction of the error: none; a stage that was not aligned is not kept. A difference fails the point
        and it is not staged again. Times of different scheduler processes are not compared.
        Sources: read at both tags. In the runs of 2026-11-18 (module docstring) every stage passed this
        check: the kept batches of the two ranks carried the same ``forward_iter`` and the same
        ``global_num_tokens``, whose two entries were equal. A stage that fails it was not run on an engine.
        How to retire it: nothing to retire.
        """
        size = self._dp_size
        if size == 1:
            return
        counts = [len(lines) for lines in kept]
        if any(count != counts[0] for count in counts):
            raise _Failure(f"{REASON_DP_STEP}: the ranks ran the call as {counts} batches, one number per rank")
        for number in range(counts[0]):
            lines = [kept[rank][number] for rank in range(size)]
            steps = [line.get("forward_iter") for line in lines]
            for rank, step in enumerate(steps):
                if not _is_int(step):
                    raise _Failure(
                        f"{REASON_RECORDER_FAULT}: {self._leaders[rank]}: batch seq {lines[rank]['seq']} has "
                        f"forward_iter={step!r}"
                    )
            what = f"batch {number + 1} of {counts[0]} ({lines[0]['mode']})"
            if any(step != steps[0] for step in steps):
                raise _Failure(f"{REASON_DP_STEP}: {what} ran in the steps {steps}, one forward_iter per rank")
            tokens = [line.get("global_num_tokens") for line in lines]
            first = tokens[0]
            if not _is_int_list(first, size) or len(set(first)) != 1 or any(other != first for other in tokens):
                raise _Failure(
                    f"{REASON_DP_STEP}: {what} in step {steps[0]} has the global_num_tokens "
                    f"{json.dumps(tokens, default=repr)}, one list per rank; expected {size} equal counts, the "
                    "same on every rank"
                )

    # ------------------------------------------------------------------ self-test and warm-up

    def _for_every_rank(self, count: int, make: Callable[[int, int], list[int]]) -> list[list[list[int]]]:
        """``count`` prompts for every attention-DP rank: ``make(rank, slot)`` for each slot, rank by rank.

        The content plan keeps its counters per rank, so the order of the ranks here changes no prompt.
        """
        return [[make(rank, slot) for slot in range(count)] for rank in range(self._dp_size)]

    def _free_stage(
        self, engine: object, prompts: list[list[list[int]]], output_tokens: int, *, followers: bool
    ) -> _Stage:
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

        With several attention-DP ranks every check sends its pair to every rank in one balanced call and
        holds on every rank: each rank has a prefix cache of its own, and a rank the self-test never reached
        would be measured without any of this known about it. The three checks that look at batches also
        hold the ranks against each other (``_require_same_step``), so an engine that does not release a
        balanced call on all ranks in one step fails here, before the sweep.

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
        prompts = self._for_every_rank(SELF_TEST_REQUESTS, lambda rank, _slot: self._plan.free_prompt(rank, unit))
        call = self._free_stage(engine, prompts, _ONE_OUTPUT_TOKEN, followers=True)
        self._require_cached(call, 0)
        self._require_same_step([[self._one_extend_batch(part)] for part in call.ranks])

    def _self_test_aligned_prefix(self, engine: object, unit: int) -> None:
        """Two requests that continue a stored page-aligned prefix report exactly that prefix as cached."""
        self._require_shape(
            f"two requests of {unit} cached and {unit} new tokens in one extend batch",
            grid.prefill_unfit_reason(self._limits, SELF_TEST_REQUESTS, unit, unit),
        )
        plan = self._plan
        seeds = self._for_every_rank(SELF_TEST_REQUESTS, lambda rank, slot: plan.prefill_seed_prompt(rank, slot, unit))
        self._free_stage(engine, seeds, _ONE_OUTPUT_TOKEN, followers=False)
        shots = self._for_every_rank(SELF_TEST_REQUESTS, lambda rank, slot: plan.prefill_prompt(rank, slot, unit, unit))
        call = self._free_stage(engine, shots, _ONE_OUTPUT_TOKEN, followers=True)
        self._require_cached(call, grid.expected_cached_tokens(grid.PREFILL, unit, self._limits.page_size))
        self._require_same_step([[self._one_extend_batch(part)] for part in call.ranks])

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
        prompts = self._for_every_rank(
            SELF_TEST_REQUESTS, lambda rank, slot: self._plan.prefill_seed_prompt(rank, slot, length)
        )
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
        prompts = self._for_every_rank(SELF_TEST_REQUESTS, lambda rank, _slot: self._plan.free_prompt(rank, unit))
        call = self._free_stage(engine, prompts, grid.DECODE_OUTPUT_TOKENS, followers=True)
        self._require_cached(call, 0)
        kept = []
        for part in call.ranks:
            extends, decodes = self._run_sequence(part, unit)
            if len(extends) != 1:
                raise _Failure(
                    f"{REASON_DECODE_CONTEXT}: {self._where(part.rank)}the run has {len(extends)} extend "
                    "batches, expected one"
                )
            kept.append([*extends, *decodes])
        self._require_same_step(kept)

    def _warm_up(self, engine: object) -> None:
        """The global warm-up: one free prompt of 256 tokens with ``warmup_iterations`` output tokens.

        Dynamo's vLLM self-benchmark sends the same request before its sweep and skips it at 0
        (``_bench_step_warmup``). Here the request also ignores EOS, as every request of the run does, so
        it always runs its full number of decode steps.

        With several attention-DP ranks the call holds one such prompt for every rank. In Dynamo's benchmark
        every rank has its own driver and so its own warm-up request; here one call reaches all ranks, and
        being balanced it leaves the placement of the calls after it as it was (see ``_stage``). A rank that
        got no warm-up would run its first points cold, and the published time of a point is the slowest
        rank's.
        """
        iterations = self.settings.warmup_iterations
        if iterations == 0:
            return
        prompts = self._for_every_rank(1, lambda rank, _slot: self._plan.free_prompt(rank, WARMUP_PROMPT_TOKENS))
        self._free_stage(engine, prompts, iterations, followers=False)

    # ------------------------------------------------------------------ points

    def _measure(self, engine: object, point: grid.Point, record: dict) -> dict:
        """Stage one point and return its measurement; stage it a second time when a cached count was wrong.

        Only a wrong cached count is staged again, once, with attempt 1 in the request ids: an eviction can
        take a seeded prefix away, and the second attempt seeds it again. Every other failure ends the run.
        Dynamo's vLLM self-benchmark re-stages a seeded prefill point once in the same case
        (``_bench_realseed_pending_step``).

        With several attention-DP ranks a wrong cached count on ANY rank stages the point again on ALL
        ranks, seed included: the ranks share every step, so one rank cannot be staged alone. Any other
        failure on any rank ends the run. The measurement then holds one forward-pass record per rank, each
        from that rank's own leader log; the artifact module takes the largest of their times as the time of
        the point, as the validator requires.

        Deviation note (a re-stage under DP attention).
        What differs: Dynamo's vLLM self-benchmark has one driver per rank and does not coordinate a
        re-stage between them; its default switches real-seed prefill off under DP attention for that reason
        (``_bench_realseed_on``). Here one driver stages all ranks, so the re-stage is one more balanced
        attempt.
        Direction of the error: none known; nothing of the first attempt is kept.
        Sources: this module and its tests. No point was staged again in the runs of 2026-11-18 (module
        docstring), so a re-stage under DP attention was not run on an engine.
        How to retire it: nothing to retire.
        """
        miss: _CacheMiss | None = None
        for attempt in (0, 1):
            if attempt:
                self._restaged.append(point.benchmark_id)
                self._take_spare_first_tokens(point)
            try:
                if point.phase == grid.PREFILL:
                    samples = self._prefill_attempt(engine, point, record, attempt)
                else:
                    samples = self._decode_attempt(engine, point, record, attempt)
            except _CacheMiss as error:
                miss = error
                continue
            rank_fpms = {
                rank: artifact.build_fpm(record, rank, wall_time, extra=extra)
                for rank, (wall_time, extra) in enumerate(samples)
            }
            return artifact.build_measurement(record, rank_fpms)
        raise _Failure(f"{REASON_CACHE}: {miss}")

    def _prefill_attempt(
        self, engine: object, point: grid.Point, record: dict, attempt: int
    ) -> list[tuple[float, dict]]:
        """Seed, warm shot and measured shot of a prefill point ``(b, n, k)``.

        Returns the sample and the extras of the record of every attention-DP rank, in rank order. Every rank
        is sent the same three calls with its own prompts: ``b`` requests on each rank in each call.

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
            seeds = self._for_every_rank(batch, lambda rank, slot: plan.prefill_seed_prompt(rank, slot, cached))
            self._stage(engine, point.benchmark_id, "seed", attempt, seeds, _ONE_OUTPUT_TOKEN, followers=False)
        parts: list[_RankCall] = []
        lines: list[dict] = []
        for shot in ("warm", "meas"):
            prompts = self._for_every_rank(batch, lambda rank, slot: plan.prefill_prompt(rank, slot, cached, new))
            call = self._stage(engine, point.benchmark_id, shot, attempt, prompts, _ONE_OUTPUT_TOKEN, followers=True)
            self._require_cached(call, expected_cached)
            parts = call.ranks
            lines = [self._one_extend_batch(part) for part in parts]
            if shot == "warm":
                # Nothing of the warm shot is kept, but it is only the rehearsal of the measured shot when
                # the ranks ran it together as well.
                self._require_same_step([[line] for line in lines])

        # The measured shot. On every rank its one extend batch is the kept batch; its counters come from the
        # recorder line. They are compared before the ranks are held against each other, so that a kept batch
        # that missed the cache on one rank stages the point again and is not taken for ranks out of step.
        expected = artifact.scheduled_requests(record)
        for part, line in zip(parts, lines, strict=True):
            where = self._where(part.rank)
            prefix_lens, extend_lens, extend_tokens = self._extend_counts(part.log, line)
            if sum(extend_lens) != extend_tokens:
                raise _Failure(
                    f"{REASON_PREFILL_TOKENS}: {where}the extend batch (seq {line['seq']}) reports "
                    f"{extend_tokens} computed tokens and its requests sum to {sum(extend_lens)}"
                )
            observed = {
                "num_prefill_requests": len(line["rids"]),
                "sum_prefill_tokens": extend_tokens,
                "sum_prefill_kv_tokens": sum(prefix_lens),
                "num_decode_requests": 0,
                "sum_decode_kv_tokens": 0,
            }
            _compare_counters(expected, observed, where)
        self._require_same_step([[line] for line in lines])
        # The sample of a rank: from ``batch.launch_ts``, which SGLang sets in ``Scheduler.run_batch`` once the
        # batch is formed (srt/managers/scheduler.py), to the result being on the host. Both are readings of
        # ``time.monotonic()`` in the process of the rank's leader. Forming the batch, with its KV allocation
        # and any eviction, is outside the interval. Inside it lies what the scheduler loop does between the
        # return of ``run_batch`` and the result call, as it does in serving.
        samples = []
        for part, line in zip(parts, lines, strict=True):
            wall_time = self._seconds(
                part.log,
                self._output_time(part.log, line) - self._time(part.log, line, "launch_ts"),
                f"extend batch seq {line['seq']}",
            )
            samples.append((wall_time, {"worker_id": self.settings.worker_id}))
        return samples

    def _decode_attempt(
        self, engine: object, point: grid.Point, record: dict, attempt: int
    ) -> list[tuple[float, dict]]:
        """Seed and run of a decode point ``(b, k)``.

        Returns the sample and the extras of the record of every attention-DP rank, in rank order. Each rank
        has its own six step times, from its own leader log, and its own sample of them. The time of the
        point is the largest of the rank samples (sglbench_artifact.py), as in Dynamo's vLLM self-benchmark,
        which reduces per rank and then takes the largest rank.

        Both calls send the same ``b`` prompts of ``k - 3`` tokens. The seed asks for one output token and
        stores the prompt. The run asks for seven: its extend batches compute only the part of the prompt
        that is not cached, and each of its six decode batches reads real KV. The sample is the upper median
        of decode steps 3 to 5; the deviation note for that choice is at ``DECODE_KEPT_STEP_INDICES`` in
        sglbench_artifact.py. A step is the time between the outputs of two successive batches, so it holds
        whatever the scheduler does between them, as a decode step in serving does.
        """
        batch, context = point.batch_size, point.cached_tokens
        page = self._limits.page_size
        prompts = self._for_every_rank(batch, lambda rank, slot: self._plan.decode_prompt(rank, slot, context))
        self._stage(engine, point.benchmark_id, "seed", attempt, prompts, _ONE_OUTPUT_TOKEN, followers=False)
        expected_cached = grid.expected_cached_tokens(grid.DECODE, context, page)
        # The run is given the planned cached count, so that a miss on one rank of several is staged again
        # whatever the other ranks logged for the steps they ran meanwhile (``_stage``).
        call = self._stage(
            engine,
            point.benchmark_id,
            "run",
            attempt,
            prompts,
            grid.DECODE_OUTPUT_TOKENS,
            followers=True,
            expected_cached=expected_cached,
        )
        self._require_cached(call, expected_cached)
        expected_extend = grid.expected_extend_tokens(grid.DECODE, 0, context, page)
        sequences = [self._run_sequence(part, expected_extend) for part in call.ranks]

        middle = artifact.DECODE_KEPT_STEP_INDICES[len(artifact.DECODE_KEPT_STEP_INDICES) // 2]
        expected = artifact.scheduled_requests(record)
        samples = []
        for part, (extends, decodes) in zip(call.ranks, sequences, strict=True):
            # Raw step i: output of decode batch i minus the output of the batch before it, which is the last
            # extend batch for step 1. Both are readings of the clock of the rank's leader.
            steps = []
            previous = self._output_time(part.log, extends[-1])
            for number, line in enumerate(decodes, 1):
                output = self._output_time(part.log, line)
                steps.append(self._seconds(part.log, output - previous, f"decode step {number} (seq {line['seq']})"))
                previous = output

            # The counters of the middle kept decode batch. The decode batch with zero-based index i reads
            # the prompt and the i output tokens that already have KV, so the middle kept one reads ``k`` per
            # request. SGLang's own decode counter is not used: it sums ``batch.seq_lens_cpu`` when the result
            # is processed (srt/managers/scheduler_components/metrics_reporter.py
            # ``SchedulerMetricsReporter._build_scheduled_request_metrics``), after the step has added one
            # token to every request, so it is higher by the batch size.
            kept = decodes[middle]
            prompt_tokens = {response["rid"]: response["prompt_tokens"] for response in part.responses}
            observed = {
                "num_prefill_requests": 0,
                "sum_prefill_tokens": 0,
                "sum_prefill_kv_tokens": 0,
                "num_decode_requests": len(kept["rids"]),
                "sum_decode_kv_tokens": sum(prompt_tokens[rid] + middle for rid in kept["rids"]),
            }
            _compare_counters(expected, observed, self._where(part.rank))
            extra = {artifact.DECODE_STEPS_KEY: steps, "worker_id": self.settings.worker_id}
            samples.append((artifact.decode_wall_time(steps), extra))
        # Every extend batch and the six decode batches of the run, rank against rank.
        self._require_same_step([[*extends, *decodes] for extends, decodes in sequences])
        return samples

    def _run_sequence(self, part: _RankCall, expected_extend: int) -> tuple[list[dict], list[dict]]:
        """The extend batches and the six decode batches of one rank of a seven-token call, after a check.

        The batches must be one or more extend batches and then exactly six decode batches, next to each
        other in the log of the rank's leader. The extend batches hold only requests of the rank, each
        request in exactly one of them and each computing ``expected_extend`` tokens. Every decode batch
        holds exactly the requests of the rank. A call above the chunk budget takes several extend batches,
        and SGLang prefers prefill to decode (srt/managers/scheduler.py ``Scheduler.get_next_batch_to_run``),
        so no decode batch runs before the last of them.

        With several attention-DP ranks no idle batch may lie between them either. A rank is given an idle
        batch for a step in which it has nothing to run and another rank has (see ``RecorderLogs``); inside
        a run that means the rank dropped out of the steps the other ranks were timed in. The reader leaves
        idle lines out, so one shows as a gap in ``seq``.

        Under DP attention a rank that decodes while another one still extends can also have its decode
        steps run as extend batches of one token per request (``_decode_steps_as_extend``). That happens
        when the ranks are out of step: one was released a step late, or took one extend batch more. The
        wait of the stage ends on such a run (``_missing_batches``), and the run fails here, at the first
        rank whose log shows it. It then has fewer than six lines of mode DECODE, so the mode check below
        fails it, with the reason ``dp_ranks_out_of_step`` when it holds an extend batch of requests that
        were already extended. Such a line is never kept or counted as a decode batch of a point: the six
        kept decode batches must be of mode DECODE, and every request must be in exactly one extend batch,
        which also fails a run whose modes pass. Neither an idle batch inside a run nor a decode step run as
        an extend batch was seen on an engine.
        """
        where = self._where(part.rank)
        modes = [line["mode"] for _position, line in part.batches]
        extends = sum(1 for mode in modes if mode == MODE_EXTEND)
        if not extends or modes != [MODE_EXTEND] * extends + [MODE_DECODE] * artifact.DECODE_RAW_STEPS:
            found = ", ".join(f"{count} {mode}" for mode, count in _runs(modes))
            lines_of_run = [line for _position, line in part.batches]
            converted = _decode_steps_as_extend(lines_of_run) if self._dp_size > 1 else []
            if converted:
                raise _Failure(
                    f"{REASON_DP_STEP}: {where}the batches of the run are {found}; the EXTEND batch(es) with "
                    f"seq {[line['seq'] for line in converted]} hold only requests that an earlier batch of "
                    "the run had extended, which is how SGLang runs a decode step of a rank while another "
                    "rank extends"
                )
            raise _Failure(
                f"{REASON_DECODE_CONTEXT}: {where}the batches of the run are {found}, expected one or more "
                f"EXTEND and then {artifact.DECODE_RAW_STEPS} DECODE"
            )
        positions = [position for position, _line in part.batches]
        lines = [line for _position, line in part.batches]
        seqs = [line["seq"] for line in lines]
        if positions != list(range(positions[0], positions[0] + len(positions))):
            raise _Failure(
                f"{REASON_DECODE_CONTEXT}: {where}a batch of other requests lies between the batches of the "
                f"run (seq {seqs})"
            )
        if self._dp_size > 1 and seqs != list(range(seqs[0], seqs[0] + len(seqs))):
            raise _Failure(
                f"{REASON_DECODE_CONTEXT}: {where}an IDLE batch without requests lies between the batches of "
                f"the run (seq {seqs})"
            )
        extend_lines, decode_lines = lines[:extends], lines[extends:]
        wanted = set(part.rids)
        seen: collections.Counter[str] = collections.Counter()
        for line in extend_lines:
            if not wanted.issuperset(line["rids"]):
                raise _Failure(
                    f"{REASON_DECODE_CONTEXT}: {where}extend batch seq {line['seq']} holds a foreign request"
                )
            seen.update(line["rids"])
            _prefix_lens, extend_lens, extend_tokens = self._extend_counts(part.log, line)
            if any(tokens != expected_extend for tokens in extend_lens) or sum(extend_lens) != extend_tokens:
                raise _Failure(
                    f"{REASON_DECODE_CONTEXT}: {where}extend batch seq {line['seq']} computes {extend_lens} "
                    f"tokens ({extend_tokens} in all), expected {expected_extend} for each request"
                )
        split = [rid for rid in part.rids if seen[rid] != 1]
        if split:
            raise _Failure(
                f"{REASON_DECODE_CONTEXT}: {where}{split[0]} is in {seen[split[0]]} extend batches of the run"
            )
        for number, line in enumerate(decode_lines, 1):
            if sorted(line["rids"]) != sorted(part.rids):
                raise _Failure(
                    f"{REASON_BATCH_SIZE}: {where}decode batch {number} (seq {line['seq']}) holds "
                    f"{len(line['rids'])} request(s), expected exactly the {len(part.rids)} of the run"
                )
        return extend_lines, decode_lines

    # ------------------------------------------------------------------ artifact

    def _expected_points(self) -> int:
        return 0 if self._grid is None else len(self._grid.points)

    def _own_producer_facts(self) -> dict:
        """What the driver knows about the run, for the ``producer`` block.

        Two of these keys have a reader, the host validator (native_artifact.py). For every cell of an SGLang
        plan it requires ``kv_cache_stored_dtype`` to be a dtype the KV-cache label of the cell stands for
        (``_validate_sglang_kv_cache_dtype``). For a cell of a plan whose model has expert layers it requires
        ``content.source`` to be ``CONTENT_TEXT_CHAINS`` (``_validate_sglang_moe_producer``). The other keys of
        the block with a reader are the launcher's: ``sglang_version``, which the validator compares with the
        pod's provenance file (``validate_native_collection``); ``kv_cache_dtype``, which it names when it
        refuses a stored dtype and requires to be ``nvfp4`` for a cell with that KV-cache label; and
        ``tp_size``, ``attn_dp_size`` and the ``moe`` block, which it compares with the sizes of the cell
        (``_validate_sglang_moe_producer``, ``_validate_sglang_dense_producer``). No AISim reader parses the
        rest.

        The facts are those of the run, not of a rank: every rank file carries the same block. ``limits`` is
        the one record every rank reported, and so is ``kv_cache_stored_dtype``, the text of the torch dtype
        of the KV pools (None in the failed artifact of a run whose headers were not accepted).
        ``first_token_need`` is the need of one rank, and ``content.issued`` counts the prompts of all ranks.
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
            "staging": STAGING_ONE_BATCH_CALL if self._dp_size == 1 else STAGING_BALANCED_BATCH_CALL,
            "recorder_format": RECORD_FORMAT,
            "scheduler_processes": len(self.logs.headers),
            "limits": None if limits is None else limits.as_dict(),
            "page_size": None if limits is None else limits.page_size,
            "enable_overlap": self._enable_overlap,
            "kv_cache_stored_dtype": self._kv_cache_stored_dtype,
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
        rank: int = 0,
        blocks: tuple[dict, dict] | None = None,
        failure: dict | None = None,
        elapsed: float | None = None,
    ) -> dict:
        """The artifact of one attention-DP rank for the state now.

        The ranks differ in ``dp.rank`` and in the record each row holds; everything else is the same call.
        The other arguments replace single values for a trial build.
        """
        config, producer = self._blocks() if blocks is None else blocks
        built = self._grid
        text_run: dict = {}
        if self._dataset is not None and self.settings.mode != grid.PREFILL and self._omitted_blocks is None:
            # The ``kvwarm`` block of a decode or agg artifact of a text run names the dataset; a prefill
            # artifact has no such block. A run with seeded ids makes the call it always made.
            text_run["kvwarm_dataset"] = {key: self._dataset[key] for key in _KVWARM_DATASET_KEYS}
        return artifact.build_rank_artifact(
            dp_rank=rank,
            dp_size=self._dp_size,
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
        """The rank artifacts of the run as it stands now: one per attention-DP rank, in rank order.

        Raises ``RuntimeError`` while the run has no outcome, and what
        ``sglbench_artifact.build_rank_artifact`` raises for a state it cannot write.
        """
        if self.status is None:
            raise RuntimeError("the run has no outcome yet: execute has not returned and no failure was recorded")
        return [self._payload(rank=rank) for rank in range(self._dp_size)]

    def write(self) -> dict:
        """Write the result files once and return ``{"status": ..., "paths": [...]}``.

        A failed run writes the rank files only. A complete or partial run writes the rank files and then the
        merged file. ``paths`` lists the rank files in rank order and then the merged file. Rank 0 is at
        ``settings.output_path`` and rank ``r`` beside it (``sglbench_artifact.rank_output_path``).

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
