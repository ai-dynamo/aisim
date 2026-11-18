# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# The start-up checks read SGLang's resolved arguments at v0.5.22 and v0.5.23; see THIRD_PARTY_NOTICES.md.
# Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d
# Source: https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90
# The benchmark flags, their defaults, their environment variables and the dataset of a text run are re-implemented from
# ai-dynamo/dynamo (Apache-2.0, https://github.com/ai-dynamo/dynamo), components/src/dynamo/vllm/backend_args.py and
# instrumented_scheduler.py at b20fe6fe94021b854597965c7305931175e8f482. See THIRD_PARTY_NOTICES.md.

"""Launcher of AISim's SGLang self-benchmark: flags, start-up checks, engine, run, result files, exit code.

Where it runs
    As the main program of an SGLang pod, started by path: ``python3 <dir>/sglbench_launcher.py <flags>``. AISim
    is not installed there. The module imports the standard library and its sibling ``sglbench_*`` modules,
    which lie flat in the same directory. They are found because Python puts the directory of a script started
    by path on ``sys.path``. With ``-I``, ``-P``, ``PYTHONSAFEPATH`` or ``-m`` it does not, and the start fails at
    the first sibling import, before any artifact can be written. SGLang is reached only through
    ``sglbench_compat``.

    SGLang starts its scheduler processes with ``spawn``, and each of them imports this file again as
    ``__mp_main__``. So nothing runs at import: everything is inside ``main``, behind the ``__main__`` guard.

What ``main`` does, in this order
    1. Reads the clock. The soft deadline counts from here, so the engine build is inside it.
    2. Reads its own flags and variables. Every other flag goes to SGLang's own parser unchanged.
    3. Makes the run id and the run object, so that every later failure can be written as a failed artifact.
    4. Installs its handlers for SIGTERM and SIGINT. Deletes its own old result files, the rank file and the
       merged file, and nothing else. Creates the recorder directory and names it to the scheduler processes.
    5. Refuses an SGLang release that was not tested. Loads SGLang's plugins, parses and resolves SGLang's
       arguments, and refuses every setup the feasibility rules do not cover (``_RESOLVED_CHECKS``,
       ``_check_sglang_environment``, the platform and the expert-parallel layout). The expert keys of the
       model config decide where the prompts come from: dataset text for a model with expert layers, seeded
       token ids for any other, unless ``DYN_BENCH_PREFILL_CONTENT`` says which (``ENV_PREFILL_CONTENT``).
    6. Writes the dump of the resolved arguments when ``--dump-config-to`` is given.
    7. For a text run: finds the dataset file (a local file, the cached file or a download), hashes it, parses
       it and takes its texts. This comes before the engine build, so a dataset that cannot be had costs
       seconds and no model load.
    8. Builds the engine with the recorder's scheduler entry.
    9. Takes the token range from the tokenizer. For a text run: reads how many token ids the model counts
       and turns texts into token ids with the served tokenizer until the text pool is large enough.
    10. Runs the benchmark (``sglbench_driver.BenchmarkRun.execute``).
    11. Always, also after a failure or a stop: shuts the engine down, writes the result files, prints the
        outcome, stays alive for ``FINAL_WAIT_SECONDS`` and returns the exit code.

Exit codes
    0 complete; 64 rejected; 65 failed run; 66 soft deadline (partial artifact); 70 internal error. ``argparse``
    exits 2 for the launcher's own flags. From step 3 on every failure leaves a failed rank artifact at the
    output path and one line ``[sglbench] FAILED: <reason>`` on stderr, unless the artifact cannot be written:
    an output path that names no file, or a directory that cannot be made or written to. The write error is
    then printed after the reason. An unknown value of ``DYN_BENCHMARK_MODE`` is refused with 64 and no
    artifact, because an artifact needs the mode.

    A text run has no other content to fall back to. When its dataset cannot be had, gives no text, cannot be
    turned into token ids or is shorter than the longest prompt, the run fails with 65 and the reason
    ``dataset_unavailable``, ``dataset_empty``, ``tokenizer_unavailable`` or ``content_too_shallow``.

    Not covered: a scheduler process that dies. SGLang then kills the launcher's process with SIGKILL
    (srt/utils/watchdog.py ``SubprocessWatchdog``; srt/entrypoints/engine.py ``launch_phase_sigquit_handler``;
    srt/managers/tokenizer_manager.py ``SignalHandler``; srt/utils/common.py ``kill_process_tree``). No
    launcher code runs after that and no artifact is written; the shell sees status 137.

Stops
    A stop is recorded as the failure ``stopped: <what>``, but only until the write of the result files
    starts. From then on SIGTERM and SIGINT are ignored where the launcher still owns the handler, because
    the gate of AISim's collector sends SIGTERM to the process group after it has accepted the files
    (collector/fpm_forward/runtime/fpm_exec.sh ``terminate_engine``). The comment at ``_StopRequested`` says
    how a stop arrives before and after SGLang takes SIGTERM over.

Checked against SGLang v0.5.22 (baa9befa381511cd4d56f83980951e5a45fffc6d) and
v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90) of ``sgl-project/sglang`` (Apache-2.0), by reading these
files under ``python/sglang`` at both tags::

    launch_server.py                         the order load_plugins, prepare_server_args, resolve_once
    srt/server_args.py                       prepare_server_args, ServerArgs.resolve_once
    srt/arg_groups/model_override_base.py    ResolvingConfig.__getattr__, ep_scale_joiner_of
    srt/arg_groups/arg_utils.py              a bool field is a store_true flag
    srt/arg_groups/fields/parallel.py        Parallel
    srt/arg_groups/fields/schedule.py        Schedule
    srt/arg_groups/fields/memory.py          Memory
    srt/arg_groups/fields/serving.py         Serving
    srt/arg_groups/fields/exec_.py           ExecMoe, ExecDllm, ExecKernel
    srt/arg_groups/fields/spec.py            Spec
    srt/arg_groups/fields/disagg.py          Disagg
    srt/arg_groups/fields/device.py          Device
    srt/arg_groups/parallel_hook.py          handle_context_parallelism, handle_deprecated_dp_attention,
                                             handle_dwdp, handle_elastic_ep, handle_eplb_and_dispatch
    srt/arg_groups/overrides.py              _a2a_ep_size
    srt/arg_groups/moe_hook.py               handle_a2a_moe
    srt/arg_groups/validation_hook.py        check_server_args
    srt/arg_groups/kv_cache_hook.py          the RuntimeError and NotImplementedError a resolution step raises
    srt/runtime_context.py                   attn_dp_enabled_of, PlatformContext, derive_parallel_widths
    srt/eplb/expert_location.py              ExpertLocationMetadata._init_common
    srt/layers/moe/fused_moe_triton/layer.py FusedMoE.__init__
    srt/environ.py                           EnvField.get, EnvBool.parse, EnvInt.parse and six variables of Envs
    srt/utils/common.py                      get_bool_env_var, kill_process_tree, require_attn_tp_gather,
                                             require_mlp_sync
    srt/mem_cache/common.py                  match_kv_cache
    srt/utils/watchdog.py                    SubprocessWatchdog
    srt/entrypoints/engine.py                Engine.__init__, Engine._launch_scheduler_processes,
                                             Engine._launch_subprocesses, Engine.shutdown,
                                             _set_envs_and_config, _wait_for_scheduler_ready
    srt/managers/tokenizer_manager.py        TokenizerManager.__init__, auto_create_handle_loop,
                                             sigterm_watchdog, _handle_batch_request,
                                             _coalesce_streaming_chunks, SignalHandler
    srt/managers/scheduler.py                Scheduler.__init__, init_chunked_prefill, init_schedule_policy,
                                             init_req_max_new_tokens, init_watch_dog_memory_saver_input_blocker,
                                             maybe_init_scripted_scheduler_hook, update_running_batch,
                                             get_next_batch_to_run, get_num_allocatable_reqs,
                                             _should_defer_prefill, _dispatch_event_loop_once
    srt/managers/schedule_policy.py          SchedulePolicy.calc_priority, SchedulePolicy._compute_prefix_matches,
                                             PrefillAdder._kv_shard_reserve_scratch
    srt/managers/scheduler_components/request_receiver.py   recv_limit_reached
    srt/managers/min_free_slots_delayer.py   resolve_min_free_slots
    srt/managers/scheduler_components/recv_skipper.py   SchedulerRecvSkipper.maybe_create
    srt/managers/scheduler_components/output_streamer.py   _GenerationStreamAccumulator.accept
    srt/managers/data_parallel_controller.py DataParallelController.dispatch_batch_generate
    srt/mem_cache/kv_cache_builder.py        build_kv_cache
    srt/mem_cache/registry.py                create_tree_cache

Seen on running engines: benchmarks/evidence/collector/sglbench-engine-probe-20261113 (check_1). The probe
driver kept in that folder parsed with ``prepare_server_args`` and built its engine with another scheduler
entry at SGLang 0.5.22 and 0.5.23, at TP 1 and TP 2. It called neither ``resolve_once`` before the build nor
any check of this module.

This launcher itself ran on engines on 2026-11-16, with SGLang 0.5.22 and 0.5.23 on H200 GPUs, in eight
launches. Four ran to a complete result (TP 1 and TP 2, page sizes 1 and 64, prefill and decode), and AISim's
validator and the gate's checker accepted their files. The other four: a forced cache mismatch (failed
artifact, exit code 65), a soft deadline (partial artifact, exit code 66), a killed scheduler process (SGLang
killed the launcher, status 137, no file) and a SIGTERM in the sweep (failed artifact, exit code 65). The files
of those launches are not kept in this repository. Every one of them served a model without expert layers and
sent seeded token ids.

Text runs ran on engines on 2026-11-17, on H200 GPUs. A model with expert layers (``Qwen/Qwen1.5-MoE-A2.7B``)
ran on one GPU with SGLang 0.5.22 and with ``--tp-size 2 --ep-size 2`` with SGLang 0.5.23, each as a prefill
cell and a decode cell of AISim's default grid. A model without expert layers ran with
``DYN_BENCH_PREFILL_CONTENT=sharegpt_chain`` and SGLang 0.5.23 in a smoke run. Every one of them ran to a
complete result that AISim's validator accepted, with the default dataset found in the cache folder. The files
are not kept in this repository. The texts of those runs gave no token id outside the token range: the pool
of every one of them dropped nothing while it still left such ids out, and held the same tokens in a smoke
run with SGLang 0.5.23 after it began to keep them. A prompt that holds such an id, a download of the
dataset, and every way in which a text run or the check of the expert-parallel layout fails, rest on the
source and on the tests.

The flags, their defaults and their ``DYN_*`` variables are those of Dynamo's vLLM worker, in the files and at
the revision the header of this file names: ``backend_args.py`` (the flag declarations and their value rules)
and ``instrumented_scheduler.py`` (the variables the benchmark reads by itself, ``uuid4().hex`` as the run id).
The dataset of a text run follows the same file: how the file is found, cached, downloaded and verified
(``_kvwarm_resolve_dataset``), the default URL, its pinned SHA-256 and the download timeout, and the names of
the four verdicts of its content probe (``_kvwarm_probe_content``).

No upstream function or block of source is copied. What follows SGLang and Dynamo is named in the header of this
file and recorded in THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import http.client
import json
import math
import os
import secrets
import signal
import sys
import time
import traceback
import urllib.request
import uuid
from collections.abc import Callable, Mapping

import sglbench_artifact as artifact
import sglbench_compat as compat
import sglbench_content as content
import sglbench_driver as driver
import sglbench_grid as grid
import sglbench_recorder as recorder

# ---------------------------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------------------------

# Exit codes. The gate of AISim's collector uses 1, 2, 124, 130 and 143 itself
# (collector/fpm_forward/runtime/fpm_exec.sh), and ``argparse`` exits 2, so these stay apart from them. The
# gate reads the launcher's code only when the launcher ends before its files validate; the reason is
# therefore also printed.
EXIT_COMPLETE = 0
EXIT_REJECTED = 64
EXIT_FAILED = 65
EXIT_PARTIAL = 66
EXIT_INTERNAL = 70

# The launcher's own failure reason. The other reasons are the driver's.
REASON_STOPPED = "stopped"

# Time limit of one engine call. An engine request has no limit of its own inside SGLang
# (sglbench_compat.py, ``generate_batch``).
ENGINE_CALL_TIMEOUT_SECONDS = 900.0

# How long the launcher stays alive after its last result file is in place. The gate polls every two seconds:
# one poll checks the files and then tests whether the launcher still lives. A launcher that ends between the
# two is reported as "Engine exited before writing all FPM benchmark outputs" although its files are complete
# (collector/fpm_forward/runtime/fpm_exec.sh, the wait loop). Five seconds cover one full poll with a wide
# margin, so the poll that accepts the files starts while the launcher lives.
FINAL_WAIT_SECONDS = 5.0

# The recorder logs go to <directory of the output path>/sglbench-recorder/<run id>/. No file in that tree may
# match benchmark*.json, resolved-config*.json or fpm-*.json: AISim's collector reads every such file under the
# results directory (sglbench_recorder.py names its files sched-<pid>.jsonl).
RECORDER_DIR_NAME = "sglbench-recorder"

DEFAULT_OUTPUT_PATH = "/tmp/benchmark_results.json"
DEFAULT_WARMUP_ITERATIONS = 5
DEFAULT_TIMEOUT_SECONDS = 900
DEFAULT_CONTENT_SEED = "0"

# SGLang's flag for the field ``incremental_streaming_output`` (srt/arg_groups/fields/serving.py ``Serving``).
# A bool field is a store_true flag (srt/arg_groups/arg_utils.py), so the flag can only switch it on.
#
# Deviation note (incremental streaming output is forced on).
# What differs: ``sglang serve`` runs with the field off unless the operator passes the flag. The benchmark
# always passes it. A stream chunk then holds only the tokens since the chunk before it, and chunks that queue
# up are joined; with the field off every chunk holds all tokens so far and queued chunks but the last are
# dropped (srt/managers/tokenizer_manager.py ``TokenizerManager._coalesce_streaming_chunks``).
# Direction of the error: none known for a timed batch. The field is read by the tokenizer manager, in the
# launcher's process (``TokenizerManager.__init__``). The scheduler process, whose batches are timed, does not
# read it.
# How to retire it: stop adding the flag when cumulative streaming is the path to measure.
STREAMING_FLAG = "--incremental-streaming-output"

# The DYN_* variable that gives a flag its default when the flag is absent, by argparse destination. Flag
# names, types, defaults and variables are those of Dynamo's vLLM worker (backend_args.py at the revision in
# the header of this file). How a variable is read is this module's own rule: a variable that is set counts as
# given, also when it is empty, and the flag decides when both are there.
# ``--dump-config-to`` has no variable here: the files of that revision do not declare the flag.
_FLAG_VARIABLES = {
    "benchmark_mode": "DYN_BENCHMARK_MODE",
    "benchmark_points_file": "DYN_BENCHMARK_POINTS_FILE",
    "prefill_max_new_token_samples": "DYN_PREFILL_MAX_NEW_TOKEN_SAMPLES",
    "prefill_max_kv_read_token_samples": "DYN_PREFILL_MAX_KV_READ_TOKEN_SAMPLES",
    "decode_max_kv_read_token_samples": "DYN_DECODE_MAX_KV_READ_TOKEN_SAMPLES",
    "decode_max_batch_size_samples": "DYN_DECODE_MAX_BATCH_SIZE_SAMPLES",
    "prefix_max_batch_size_samples": "DYN_PREFIX_MAX_BATCH_SIZE_SAMPLES",
    "benchmark_max_batch_size": "DYN_BENCHMARK_MAX_BATCH_SIZE",
    "benchmark_warmup_iterations": "DYN_BENCHMARK_WARMUP_ITERATIONS",
    "benchmark_output_path": "DYN_BENCHMARK_OUTPUT_PATH",
    "benchmark_timeout": "DYN_BENCHMARK_TIMEOUT",
}

# The six sample caps, by the field name of ``sglbench_grid.SampleCaps``, which is also the argparse
# destination of the flag. The defaults are the ones of that record.
_CAP_DEFAULTS = {field.name: field.default for field in dataclasses.fields(grid.SampleCaps)}
# The name of each cap in the ``config`` block of the artifact: Dynamo's name for the same setting.
_CAP_CONFIG_KEYS = {name: name.removeprefix("benchmark_") for name in _CAP_DEFAULTS}

# Flags of the vLLM benchmark that ask for something this benchmark does not have. Each is refused when it is
# switched on, by its flag or by its variable; the ``--no-`` form means off and is accepted.
# (argparse destination, flag, DYN_* variable, why it is refused)
_REFUSED_SWITCHES = (
    (
        "benchmark_collect_imbalanced",
        "--benchmark-collect-imbalanced",
        "DYN_BENCHMARK_COLLECT_IMBALANCED",
        "every request of a point is the same here",
    ),
    (
        "benchmark_randomize_kda_state",
        "--benchmark-randomize-kda-state",
        "DYN_BENCHMARK_RANDOMIZE_KDA_STATE",
        "every decode context here is computed by real requests",
    ),
    (
        "benchmark_hybrid_live_state",
        "--benchmark-hybrid-live-state",
        "DYN_BENCHMARK_HYBRID_LIVE_STATE",
        "models with a recurrent state are refused",
    ),
)
# Dynamo's vLLM benchmark lets this variable decide over the flag and the variable above it
# (instrumented_scheduler.py, ``_bench_init``), and refuses a value that is not a boolean word.
ENV_COLLECT_IMBALANCED_OVERRIDE = "DYN_FPM_BENCH_COLLECT_IMBALANCED"
_TRUE_WORDS = ("1", "true", "yes", "on")
_FALSE_WORDS = ("0", "false", "no", "off")

# The five deprecated granularity flags of the vLLM benchmark. Each is refused when it is given at all, by its
# flag or by its variable. (argparse destination, flag, DYN_* variable, the flag to use)
_LEGACY_FLAGS = (
    (
        "benchmark_prefill_granularity",
        "--benchmark-prefill-granularity",
        "DYN_BENCHMARK_PREFILL_GRANULARITY",
        "--prefill-max-new-token-samples",
    ),
    (
        "benchmark_prefill_kv_read_granularity",
        "--benchmark-prefill-kv-read-granularity",
        "DYN_BENCHMARK_PREFILL_KV_READ_GRANULARITY",
        "--prefill-max-kv-read-token-samples",
    ),
    (
        "benchmark_prefill_batch_granularity",
        "--benchmark-prefill-batch-granularity",
        "DYN_BENCHMARK_PREFILL_BATCH_GRANULARITY",
        "--prefix-max-batch-size-samples",
    ),
    (
        "benchmark_decode_length_granularity",
        "--benchmark-decode-length-granularity",
        "DYN_BENCHMARK_DECODE_LENGTH_GRANULARITY",
        "--decode-max-kv-read-token-samples",
    ),
    (
        "benchmark_decode_batch_granularity",
        "--benchmark-decode-batch-granularity",
        "DYN_BENCHMARK_DECODE_BATCH_GRANULARITY",
        "--decode-max-batch-size-samples",
    ),
)

# Variables Dynamo's vLLM benchmark reads by itself (instrumented_scheduler.py), and what they mean here.
ENV_CONTENT_SEED = "DYN_BENCH_CONTENT_SEED"  # seed of the prompt content, kept as text
ENV_WORKER_ID = "DYN_FPM_WORKER_ID"  # written into every forward-pass record
ENV_OUTPUT_PATH = "DYN_FPM_BENCHMARK_OUTPUT_PATH"  # replaces --benchmark-output-path when set
# The two switches of real cached state (instrumented_scheduler.py ``_kvwarm_flag_on`` and
# ``_bench_realseed_on``). This benchmark has real state only, so each must be unset or say on. In Dynamo,
# ``auto`` of the second one means on for some models and off for others; here it is accepted.
_REAL_STATE_SWITCHES = (
    ("DYN_BENCH_KV_WARMUP", ("on", "1", "true")),
    ("DYN_BENCH_PREFILL_REAL_SEED", ("on", "1", "true", "auto")),
)
# Where the prompts of a run come from. Dynamo reads the variable in ``_bench_synthetic_token_ids``; here it has
# three accepted values:
# - unset or empty: dataset text when the model has expert layers, because the routing of a token to its
#   experts depends on the content, and seeded token ids for every other model;
# - ``random``: seeded token ids. A model with expert layers is then refused;
# - ``sharegpt_chain``: dataset text for any model, one without expert layers included.
# In a text run every prompt is text: seeds, shots, prompts without cached tokens, the self-test and the
# warm-up. In Dynamo the variable changes the prefill prompts only, and the decode contexts are text whenever
# its warm-up passes its gate. Dynamo's third value, ``sharegpt``, cuts windows out of one flat pool that
# wraps around; that construction is not built here, so the word is refused like any other.
ENV_PREFILL_CONTENT = "DYN_BENCH_PREFILL_CONTENT"
SEEDED_CONTENT = "random"
TEXT_CONTENT = "sharegpt_chain"
# The dataset of a text run. The three variables mean what they mean in Dynamo (``_kvwarm_resolve_dataset``),
# with two differences: an empty value counts as unset, and a local file is hashed and compared with the hash
# variable too, where Dynamo never compares a local file.
# - The dataset: a URL (http or https), downloaded into the cache directory unless the file is there already,
#   or the path of a local file. Unset: DEFAULT_DATASET_URL.
# - The cache directory. Unset: ``fpm_datasets`` beside the directory HF_HOME names, or
#   DEFAULT_DATASET_CACHE_DIR when HF_HOME is unset as well. AISim's run script sets the variable to a scratch
#   directory when the directory beside HF_HOME cannot be written (src/aisim/generator/builders/fpm_builder.py).
# - The SHA-256 the file must have. Unset: DEFAULT_DATASET_SHA256 for the default URL and no check for
#   any other dataset.
# A run with seeded token ids reads none of them.
ENV_KVWARM_DATASET = "DYN_BENCH_KV_WARMUP_DATASET"
ENV_KVWARM_CACHE_DIR = "DYN_BENCH_KV_WARMUP_CACHE_DIR"
ENV_KVWARM_SHA256 = "DYN_BENCH_KV_WARMUP_SHA256"
ENV_HF_HOME = "HF_HOME"
_DATASET_VARIABLES = (ENV_KVWARM_DATASET, ENV_KVWARM_CACHE_DIR, ENV_KVWARM_SHA256, ENV_HF_HOME)
# The default dataset, its pinned digest and the socket timeout of a download, with the values of Dynamo's
# ``_KVWARM_DEFAULT_DATASET_URL``, ``_KVWARM_DEFAULT_DATASET_SHA256`` and ``_KVWARM_DOWNLOAD_TIMEOUT_S``. The
# directory is the one Dynamo's ``_kvwarm_resolve_dataset`` uses without HF_HOME.
DEFAULT_DATASET_URL = (
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/"
    "ShareGPT_V3_unfiltered_cleaned_split.json"
)
DEFAULT_DATASET_SHA256 = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
DATASET_DOWNLOAD_TIMEOUT_SECONDS = 60
DEFAULT_DATASET_CACHE_DIR = "/tmp/fpm_datasets"
# Collector rule "rows depend only on the pin and the plan, never on environment variables". The default
# dataset and its digest are constants of this file, so they belong to the pin, as does what AISim's run script
# sets by itself. Any other dataset, cache directory, digest or content word must be given through
# ``K8sConfig.extra_env`` of the collection, which AISim hashes into the plan with the other generator
# overrides (collector/fpm_forward/planner.py). A value that reaches the pod in any other way is outside the
# plan; the launcher cannot tell where a variable came from, so this is a rule for whoever sets one.
#
# Dynamo's ``DYN_BENCH_POOL_TAG`` draws the windows of its flat text pool again, and its two giant-KV variables
# repeat a decode step. Neither construction exists here, so each is refused when it is set.
_REFUSED_VARIABLES = ("DYN_BENCH_POOL_TAG", "DYN_BENCH_GIANT_KV_THRESHOLD", "DYN_BENCH_GIANT_KV_REPEATS")
# Not read at all: every FPM_* variable. They belong to the gate of AISim's collector.

# The text pool of a text run. Texts are turned into token ids in dataset order, POOL_ENCODE_BATCH texts in one
# tokenizer call, until the pool holds POOL_TOKEN_TARGET tokens or the texts run out. A prompt part is a chain
# of whole conversations, each used once, so it can give at most the tokens of the whole pool; the driver
# compares the pool with the longest prompt once the grid exists. The target is this module's own choice. It
# is about two hundred times an input limit of 40954 tokens, the limit in every recorder header of
# benchmarks/evidence/collector/sglbench-engine-probe-20261113, and at four bytes a token the pool then takes
# about 32 MB.
#
# Deviation note (the text of a run is the head of the dataset).
# What differs: serving sees whatever text its clients send. A text run takes its text from the first
# conversations of the dataset's collection half, in dataset order, as many as fill the pool; the rest of the
# file is parsed and not used. Dynamo's vLLM self-benchmark shuffles all kept conversations for every chain.
# Direction of the error: not measured. The pool decides which text a run can show to the expert layers, and a
# larger pool would show more of the dataset.
# How to retire it: raise POOL_TOKEN_TARGET, or draw the conversations of the pool from the whole file.
POOL_TOKEN_TARGET = 8_000_000
POOL_ENCODE_BATCH = 256
# A file is hashed and downloaded in pieces of this size.
_FILE_PIECE_BYTES = 1 << 20

# Exceptions with which SGLang refuses a setup while it loads plugins, parses, resolves and reads the model
# config: its hooks raise all four (srt/arg_groups, for example validation_hook.py and kv_cache_hook.py).
_SGLANG_REFUSALS = (ValueError, AssertionError, NotImplementedError, RuntimeError)
# During the engine build the same holds for three of them: srt/arg_groups/validation_hook.py
# ``check_server_args`` asserts and raises ValueError, and srt/entrypoints/engine.py ``Engine.__init__`` raises
# ValueError. A RuntimeError of the build is a scheduler process that died or did not report ready
# (``_wait_for_scheduler_ready``): the engine failed, the setup was not refused.
_BUILD_REFUSALS = (ValueError, AssertionError, NotImplementedError)

_STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)


# ---------------------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------------------


class _Failed(Exception):
    """A step of the launcher failed. ``reason`` is the whole failure text of the artifact."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _rejected(text: str) -> _Failed:
    return _Failed(f"{driver.REASON_REJECTED}: {text}")


def _say(text: str) -> None:
    """Print one line of the launcher. SGLang resets the logging setup, so this does not go through logging."""
    print(f"[sglbench] {text}", file=sys.stderr, flush=True)


def _shown(value: object) -> str:
    """A value for a message: JSON spelling for true, false and null, plain text otherwise."""
    if value is None or isinstance(value, bool):
        return json.dumps(value)
    return str(value)


def _one_line(error: BaseException) -> str:
    """The text of an exception on one line, so that a reason stays one line of a log."""
    return " ".join(str(error).split())


def _described(error: BaseException) -> str:
    return f"{type(error).__name__}: {_one_line(error)}"


# ---------------------------------------------------------------------------------------------------------------
# Stops
# ---------------------------------------------------------------------------------------------------------------


class _StopRequested(SystemExit):
    """Raised in the main thread by the launcher's handler of SIGTERM and SIGINT.

    It subclasses ``SystemExit`` on purpose. A signal can arrive while the engine's event loop runs a callback
    or a task step, and asyncio lets only ``SystemExit`` and ``KeyboardInterrupt`` leave the loop from there;
    any other exception is handed to the loop's exception handler or stored on the task, and the run would go
    on (CPython 3.12, asyncio/events.py ``Handle._run`` and asyncio/tasks.py ``Task.__step``). The driver and
    ``sglbench_compat.generate_batch`` let both pass. SGLang makes uvloop the event loop policy when its engine
    module is imported (srt/entrypoints/engine.py); uvloop's source was not read. The launch of 2026-11-16 with
    a SIGTERM in the sweep recorded the stop, but not whether the signal arrived inside an engine call, so that
    uvloop's loop treats the exception as asyncio's does is still not established.

    How a stop arrives:

    - before the first request, and for SIGINT always: this exception, from the handler below;
    - after the first request, for SIGTERM: SGLang puts a handler of its own on the event loop
      (srt/managers/tokenizer_manager.py ``TokenizerManager.auto_create_handle_loop``). It only sets a flag,
      and ``sigterm_watchdog`` acts on it only when it finds no request in flight; it runs only while the
      engine's loop runs, which here is inside an engine call. So the launcher takes the signal back after
      every engine call (``_SignalStops.reclaim``), and the stop arrives as this exception again. If SGLang
      does reach its ``sys.exit(0)``, that leaves the engine call as a plain ``SystemExit``. A SIGTERM during
      the very first engine call stays SGLang's. Run once on an engine, on 2026-11-16 with SGLang 0.5.22: a
      SIGTERM sent in the sweep, after the first request, ended the run as ``stopped: signal SIGTERM`` with
      exit code 65, and SGLang's own handler logged nothing. The first-call case and SGLang's ``sys.exit(0)``
      were read and not run.

    Both are recorded as ``stopped: <what>``.
    """

    def __init__(self, what: str) -> None:
        super().__init__(what)
        self.what = what


class _SignalStops:
    """The launcher's handling of SIGTERM and SIGINT.

    While armed, the first signal raises ``_StopRequested`` once; the handler disarms itself before it raises,
    so the code that handles the stop is not interrupted a second time. While disarmed, a signal is only
    noted in ``late``. ``ignore`` ends both: from the start of the final write a signal changes nothing.
    """

    def __init__(self) -> None:
        self._armed = False
        # The signal the handler raised a stop for, or None. It is kept because a stop can be lost on its way:
        # Python drops what a destructor raises, and library code may catch every exception.
        self.raised: str | None = None
        self.late: list[str] = []
        # One bound method object, kept so that ``signal.getsignal`` can be compared with it.
        self._handler = self._on_signal

    def install(self) -> None:
        self._armed = True
        for signum in _STOP_SIGNALS:
            signal.signal(signum, self._handler)

    def _on_signal(self, signum: int, _frame: object) -> None:
        name = signal.Signals(signum).name
        if self._armed:
            self._armed = False
            self.raised = name
            raise _StopRequested(f"signal {name}")
        self.late.append(name)

    def disarm(self) -> None:
        self._armed = False

    def reclaim(self) -> None:
        """Put the launcher's handler back on a signal that SGLang has taken over, while the handler is armed.

        SGLang's own SIGTERM handling does not stop a benchmark run: ``TokenizerManager.sigterm_watchdog``
        (srt/managers/tokenizer_manager.py) waits until no request is in flight, and it runs only while the
        engine's event loop runs, which here is inside an engine call, after ``generate_request`` has
        registered the requests of that call (``_init_req_state``). That was read in the source. What this
        method does was run once on an engine; see ``_StopRequested``.
        """
        if not self._armed:
            return
        for signum in _STOP_SIGNALS:
            if signal.getsignal(signum) is not self._handler:
                signal.signal(signum, self._handler)

    def ignore(self) -> None:
        """Ignore SIGTERM and SIGINT from now on, where the handler is still the launcher's.

        A handler that SGLang has put in the place of the launcher's is left alone: it does nothing while its
        event loop does not run.
        """
        for signum in _STOP_SIGNALS:
            if signal.getsignal(signum) is self._handler:
                signal.signal(signum, signal.SIG_IGN)


def _stop_text(error: BaseException) -> str:
    """What stopped the run, for an exception that is not an ``Exception``."""
    if isinstance(error, SystemExit):
        return f"SystemExit({error.code!r})"
    text = _one_line(error)
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


# ---------------------------------------------------------------------------------------------------------------
# Flags and variables of the launcher
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Options:
    """What the launcher's own flags and variables say. Nothing here comes from SGLang."""

    mode: str
    points_file: str | None
    points_sha256: str | None
    # Result of sglbench_grid.parse_points_manifest, or None for the generated grid.
    manifest: dict | None
    # The six sample caps as they were given, by field name of sglbench_grid.SampleCaps.
    cap_values: dict
    # The caps record, or None when the given values are refused.
    caps: grid.SampleCaps | None
    warmup_iterations: int
    timeout: int
    output_path: str
    dump_config_to: str | None
    content_seed: str
    # ``DYN_BENCH_PREFILL_CONTENT``: SEEDED_CONTENT, TEXT_CONTENT, or None when the model decides.
    prefill_content: str | None
    # The dataset variables and HF_HOME that are set and not empty, as text. Only a text run reads them.
    dataset_environ: dict
    worker_id: str
    # What goes to SGLang's own parser.
    sglang_argv: list[str]
    # The first bad value among the flags and variables, or None.
    rejection: str | None
    # Set when the output path names no file. No artifact can be written then.
    output_problem: str | None


def _build_parser() -> argparse.ArgumentParser:
    """The parser of the launcher's own flags. Abbreviations are off.

    With abbreviations on, a prefix of a launcher flag would be taken for it, and what was meant for SGLang
    would be swallowed here. Every flag has the default ``None``, which means "not given": the default then
    comes from the flag's variable, and after that from the constants of this module.
    """
    parser = argparse.ArgumentParser(
        prog="sglbench_launcher.py",
        allow_abbrev=False,
        description=(
            "Run AISim's SGLang self-benchmark: build SGLang's engine, measure the benchmark points and write "
            "the result files. Every flag that is not listed here goes to SGLang's own argument parser."
        ),
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=grid.MODES,
        default=None,
        help="required: prefill, decode or agg (the prefill points, then the decode points)",
    )
    parser.add_argument("--benchmark-points-file", default=None, help="points manifest; replaces the generated grid")
    for name, default in _CAP_DEFAULTS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=int, default=None, help=f"sample cap, default {default}"
        )
    parser.add_argument(
        "--benchmark-warmup-iterations",
        type=int,
        default=None,
        help=f"output tokens of the global warm-up, default {DEFAULT_WARMUP_ITERATIONS}; 0 skips it",
    )
    parser.add_argument("--benchmark-output-path", default=None, help=f"rank file, default {DEFAULT_OUTPUT_PATH}")
    parser.add_argument(
        "--benchmark-timeout",
        type=int,
        default=None,
        help=f"soft deadline in seconds from the start of the launcher, default {DEFAULT_TIMEOUT_SECONDS}",
    )
    parser.add_argument("--dump-config-to", default=None, help="write SGLang's resolved arguments to this file")
    for dest, flag, _variable, _why in _REFUSED_SWITCHES:
        parser.add_argument(flag, dest=dest, action=argparse.BooleanOptionalAction, default=None, help="not supported")
    for dest, flag, _variable, replacement in _LEGACY_FLAGS:
        parser.add_argument(flag, dest=dest, default=None, help=f"not supported; use {replacement}")
    return parser


def _result_paths(output_path: str) -> tuple[str, str]:
    """The two files the launcher owns: the rank file and the merged file. Raises for a path that names none."""
    rank_path = artifact.rank_output_path(output_path, driver.RANK)
    return rank_path, artifact.merged_output_path(rank_path)


def _refused_switches(flags: argparse.Namespace, environ: Mapping[str, str]) -> list[str]:
    """The refusal of every switch of the vLLM benchmark that is on, by its flag or by its variable."""
    problems = []
    for dest, flag, variable, why in _REFUSED_SWITCHES:
        switched_on, source = getattr(flags, dest), flag
        if switched_on is None:
            # The variable switches the flag on for the four words Dynamo's benchmark takes as true for its
            # own boolean variable, after strip and lower (instrumented_scheduler.py, ``_bench_init``).
            switched_on, source = environ.get(variable, "").strip().lower() in _TRUE_WORDS, variable
        if dest == "benchmark_collect_imbalanced" and ENV_COLLECT_IMBALANCED_OVERRIDE in environ:
            override = environ[ENV_COLLECT_IMBALANCED_OVERRIDE]
            word = override.strip().lower()
            if word not in _TRUE_WORDS and word not in _FALSE_WORDS:
                problems.append(f"{ENV_COLLECT_IMBALANCED_OVERRIDE}={override!r} is not a boolean")
                continue
            switched_on, source = word in _TRUE_WORDS, ENV_COLLECT_IMBALANCED_OVERRIDE
        if switched_on:
            problems.append(f"{source} is not supported: {why}")
    return problems


def _refused_variables(environ: Mapping[str, str]) -> list[str]:
    """The refusal of every variable of the vLLM benchmark that asks for something this benchmark lacks."""
    problems = []
    for variable, accepted in _REAL_STATE_SWITCHES:
        if variable in environ and environ[variable].strip().lower() not in accepted:
            problems.append(
                f"{variable}={environ[variable]!r} is not supported: the benchmark stages real cached state only "
                f"(accepted: {', '.join(accepted)})"
            )
    word = environ.get(ENV_PREFILL_CONTENT, "")
    if word and word not in (SEEDED_CONTENT, TEXT_CONTENT):
        problems.append(f"{ENV_PREFILL_CONTENT}={word!r} is not supported (accepted: {SEEDED_CONTENT}, {TEXT_CONTENT})")
    for variable in _REFUSED_VARIABLES:
        if environ.get(variable):
            problems.append(f"{variable} is not supported (it is set to {environ[variable]!r})")
    return problems


def _read_points_file(path: str) -> tuple[dict, str]:
    """Read and parse the points manifest. The file is only read, never changed.

    AISim's collector hashes the staged file before and after the run
    (collector/fpm_forward/runner.py, the point manifest receipt).
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    return grid.parse_points_manifest(raw), hashlib.sha256(raw).hexdigest()


def _read_options(argv: list[str], environ: Mapping[str, str]) -> _Options:
    """Read the launcher's flags and variables. ``argparse`` errors exit 2; other bad values are kept as text.

    ``environ`` is the process environment; it is an argument so that the reading can be tested. This module
    is the only one of the benchmark that reads variables, apart from the recorder's directory.
    """
    parser = _build_parser()
    flags, rest = parser.parse_known_args(argv)
    problems: list[str] = []

    def text_value(dest: str, default: str | None) -> str | None:
        value = getattr(flags, dest)
        if value is None:
            value = environ.get(_FLAG_VARIABLES[dest])
        return default if value is None else value

    def int_value(dest: str, default: int | None) -> int | None:
        value = getattr(flags, dest)
        variable = _FLAG_VARIABLES[dest]
        if value is None and variable in environ:
            try:
                value = int(environ[variable])
            except ValueError:
                problems.append(f"{variable}={environ[variable]!r} is not an integer")
        return default if value is None else value

    # The mode is required here; in Dynamo's vLLM worker it is optional, because that worker also serves.
    # Without a mode no artifact can be written. A missing mode ends as an argparse error does; an unknown
    # value of the variable, which argparse never sees, is refused with the launcher's own exit code.
    mode = text_value("benchmark_mode", None)
    if mode is None:
        parser.error(f"--benchmark-mode is required (or {_FLAG_VARIABLES['benchmark_mode']}): {', '.join(grid.MODES)}")
    if mode not in grid.MODES:
        _say(
            f"FAILED: {driver.REASON_REJECTED}: {_FLAG_VARIABLES['benchmark_mode']}={mode!r}: "
            f"choose from {', '.join(grid.MODES)}"
        )
        _say("no result file was written: an artifact needs the mode")
        raise SystemExit(EXIT_REJECTED)

    for dest, flag, variable, replacement in _LEGACY_FLAGS:
        if getattr(flags, dest) is not None:
            problems.append(f"{flag} is not supported; use {replacement}")
        elif variable in environ:
            problems.append(f"{variable} is not supported; use {replacement}")
    problems.extend(_refused_switches(flags, environ))

    warmup_iterations = int_value("benchmark_warmup_iterations", DEFAULT_WARMUP_ITERATIONS)
    if warmup_iterations < 0:
        problems.append(f"--benchmark-warmup-iterations must be at least 0, got {warmup_iterations}")
    timeout = int_value("benchmark_timeout", DEFAULT_TIMEOUT_SECONDS)
    if timeout <= 0:
        problems.append(f"--benchmark-timeout must be above 0, got {timeout}")
    cap_values = {name: int_value(name, default) for name, default in _CAP_DEFAULTS.items()}
    caps = None
    try:
        # The bounds of the caps have one home, the record itself.
        caps = grid.SampleCaps(**cap_values)
    except (TypeError, ValueError) as error:
        problems.append(f"sample caps: {error}")

    # DYN_FPM_BENCHMARK_OUTPUT_PATH wins over the flag and its own variable, as in Dynamo's vLLM benchmark.
    if ENV_OUTPUT_PATH in environ:
        output_path, output_source = environ[ENV_OUTPUT_PATH], ENV_OUTPUT_PATH
    else:
        output_path = text_value("benchmark_output_path", DEFAULT_OUTPUT_PATH)
        output_source = "--benchmark-output-path"
    output_problem = None
    result_paths: tuple[str, ...] = ()
    try:
        result_paths = _result_paths(output_path)
    except (TypeError, ValueError) as error:
        output_problem = f"{output_source} {output_path!r}: {error}"

    dump_config_to = flags.dump_config_to
    if dump_config_to is not None:
        if not dump_config_to:
            problems.append("--dump-config-to must name a file")
        elif os.path.abspath(dump_config_to) in {os.path.abspath(path) for path in result_paths}:
            problems.append(f"--dump-config-to {dump_config_to!r} is one of the result files of the run")

    # Dynamo reads the seed as text, "0" when unset. The content plan needs a non-empty UTF-8 string.
    content_seed = environ.get(ENV_CONTENT_SEED, DEFAULT_CONTENT_SEED)
    try:
        usable_seed = bool(content_seed.encode("utf-8"))
    except UnicodeEncodeError:
        usable_seed = False
    if not usable_seed:
        problems.append(f"{ENV_CONTENT_SEED}={content_seed!r} must be a non-empty UTF-8 string")
    problems.extend(_refused_variables(environ))
    # Unset and empty leave the choice to the model. A word that was refused above is not kept.
    prefill_content = environ.get(ENV_PREFILL_CONTENT) or None
    if prefill_content not in (SEEDED_CONTENT, TEXT_CONTENT):
        prefill_content = None
    # The dataset variables are kept as text and looked at only when the run turns out to be a text run, which
    # is known once the model config was read. An empty value counts as unset. Whoever sets one of them must
    # set it through ``K8sConfig.extra_env``, so that it is part of the plan; see the comment at the constants.
    dataset_environ = {name: environ[name] for name in _DATASET_VARIABLES if environ.get(name)}

    points_file = text_value("benchmark_points_file", None)
    manifest, points_sha256 = None, None
    if points_file is not None:
        try:
            manifest, points_sha256 = _read_points_file(points_file)
        except (OSError, ValueError) as error:
            problems.append(f"--benchmark-points-file {points_file!r}: {_one_line(error)}")

    # SGLang gets every other flag as it was typed, and the streaming flag when it is not among them.
    sglang_argv = list(rest)
    if STREAMING_FLAG not in sglang_argv:
        sglang_argv.append(STREAMING_FLAG)

    return _Options(
        mode=mode,
        points_file=points_file,
        points_sha256=points_sha256,
        manifest=manifest,
        cap_values=cap_values,
        caps=caps,
        warmup_iterations=warmup_iterations,
        timeout=timeout,
        output_path=output_path,
        dump_config_to=dump_config_to,
        content_seed=content_seed,
        prefill_content=prefill_content,
        dataset_environ=dataset_environ,
        worker_id=environ.get(ENV_WORKER_ID, ""),
        sglang_argv=sglang_argv,
        rejection=problems[0] if problems else None,
        output_problem=output_problem,
    )


def _config_block(options: _Options) -> dict:
    """The ``config`` block of the artifact: only values that repeat when the same cell is launched again.

    AISim copies the block, without ``output_path``, into its measurement evidence and compares it between
    two launches of one cell (collector/fpm_forward/measurement_evidence.py ``extract_measurement_evidence``
    and ``compare_measurements``). So it holds no run id, time, process id, host name or measured capacity.
    The key names are those of Dynamo's vLLM benchmark where it has the setting.
    """
    config = {
        "warmup_iterations": options.warmup_iterations,
        "timeout": options.timeout,
    }
    config.update({_CAP_CONFIG_KEYS[name]: value for name, value in options.cap_values.items()})
    config["points_source"] = grid.SOURCE_GENERATED if options.points_file is None else grid.SOURCE_MANIFEST
    config["content_seed"] = options.content_seed
    config["output_path"] = options.output_path
    return config


# ---------------------------------------------------------------------------------------------------------------
# Start-up checks on SGLang's environment variables
# ---------------------------------------------------------------------------------------------------------------


def _sglang_env_flag(environ: Mapping[str, str], name: str) -> bool:
    """Whether SGLang reads the variable ``name`` as true.

    srt/environ.py ``EnvBool.parse``: true for true, 1, yes and y in any letter case. Any other value is
    false: either it is one of SGLang's false words, or ``EnvField.get`` warns and takes the default, which
    is false for every variable tested with this function.
    """
    return environ.get(name, "").lower() in ("true", "1", "yes", "y")


def _sglang_env_int(environ: Mapping[str, str], name: str) -> int | None:
    """The integer SGLang reads from the variable ``name``, or None when it takes its default.

    srt/environ.py ``EnvInt.parse`` is ``int(value)``; for a value that is not an integer ``EnvField.get``
    warns and takes the default.
    """
    if name not in environ:
        return None
    try:
        return int(environ[name])
    except ValueError:
        return None


def _check_sglang_environment(environ: Mapping[str, str]) -> None:
    """Refuse variables of SGLang that change admission, caching or the steps of a run.

    Each is read in the scheduler processes, which inherit the launcher's environment, so the launcher can
    test them before it builds anything. The scheduler side of two of them is tested again from the recorder
    headers (``max_new_tokens_limit`` and ``input_blocker``, sglbench_driver.py).

    It must be given the live environment after SGLang was imported: the import rewrites variables under the
    legacy ``SGL_`` prefix (srt/environ.py ``_handle_deprecated_envs``).
    """
    # srt/managers/scheduler.py ``Scheduler.__init__`` reads it and ``Scheduler.init_req_max_new_tokens``
    # applies it when it is above 0, with a warning and no error. A decode run asks for seven output tokens.
    limit = _sglang_env_int(environ, "SGLANG_MAX_NEW_TOKENS_LIMIT")
    if limit is not None and 0 < limit < grid.DECODE_OUTPUT_TOKENS:
        raise _rejected(
            f"SGLANG_MAX_NEW_TOKENS_LIMIT={limit} is below the {grid.DECODE_OUTPUT_TOKENS} output tokens of a "
            "decode run"
        )
    # srt/mem_cache/common.py ``match_kv_cache`` and srt/managers/schedule_policy.py
    # ``SchedulePolicy._compute_prefix_matches`` turn every cache match into a miss.
    if _sglang_env_flag(environ, "SGLANG_RADIX_FORCE_MISS"):
        raise _rejected("SGLANG_RADIX_FORCE_MISS disables every cache hit")
    # srt/managers/scheduler.py: the first forces retractions (``TEST_RETRACT``, used in
    # ``Scheduler.update_running_batch``), the second installs a scripted test hook
    # (``Scheduler.maybe_init_scripted_scheduler_hook``).
    for name in ("SGLANG_TEST_RETRACT", "SGLANG_TEST_SCRIPTED_RUNTIME"):
        if _sglang_env_flag(environ, name):
            raise _rejected(f"{name} is a test switch and must be unset")
    # srt/managers/scheduler.py ``Scheduler.__init__`` reads it (``max_recv_per_poll``). At 0 or above it caps
    # the messages taken in one poll (srt/managers/scheduler_components/request_receiver.py
    # ``recv_limit_reached``); a negative value, SGLang's default, is no cap.
    per_poll = _sglang_env_int(environ, "SGLANG_SCHEDULER_MAX_RECV_PER_POLL")
    if per_poll is not None and per_poll >= 0:
        raise _rejected(f"SGLANG_SCHEDULER_MAX_RECV_PER_POLL must be unset (got {per_poll})")
    # srt/managers/scheduler.py ``Scheduler.init_watch_dog_memory_saver_input_blocker`` and
    # srt/managers/tokenizer_manager.py ``TokenizerManager._handle_batch_request`` read this one with
    # srt/utils/common.py ``get_bool_env_var``, which takes only true and 1 as true. It creates the input
    # blocker, with one poll over all ranks in every scheduler step.
    if environ.get("SGLANG_ENABLE_COLOCATED_BATCH_GEN", "").lower() in ("true", "1"):
        raise _rejected("SGLANG_ENABLE_COLOCATED_BATCH_GEN must be unset without DP attention")


# ---------------------------------------------------------------------------------------------------------------
# Start-up checks on SGLang's resolved arguments
# ---------------------------------------------------------------------------------------------------------------


def _only(*allowed: object) -> Callable[[object], bool]:
    """A test that accepts exactly the given values, each with its own type (``True`` is not ``1``)."""

    def accepts(value: object) -> bool:
        return any(type(value) is type(item) and value == item for item in allowed)

    return accepts


def _int_at_most(limit: int) -> Callable[[object], bool]:
    def accepts(value: object) -> bool:
        return type(value) is int and value <= limit

    return accepts


# The resolved fields of the expert layers that are written into the ``producer`` block of a model with expert
# layers. ``moe_runner_backend`` is recorded and refuses nothing: which kernels run the expert layers is a
# serving choice (srt/arg_groups/fields/exec_.py ``ExecMoe``).
_MOE_RECORDED_FIELDS = ("ep_size", "moe_a2a_backend", "moe_runner_backend", "moe_dense_tp_size", "enable_eplb")

_CACHE_TIER_TEXT = "only the built-in in-GPU radix cache is supported ({name}={v})"
_ELASTIC_EP_TEXT = "elastic EP is not supported ({name}={v})"
_DISAGGREGATION_TEXT = "disaggregated serving is not supported ({name}={v})"
_POLICY_TEXT = "schedule policy must be fcfs without priority scheduling ({name}={v})"

# What the launcher refuses before it builds the engine: (field, the test a value must pass, the refusal).
# Every field is read from the resolved view of SGLang's arguments, which holds what SGLang decided and not
# only what the operator typed (sglbench_compat.py, ``resolve``). All fields exist under these names, with the
# defaults that pass, at both tags; a field that is missing is an error, never a pass. The feasibility rules
# of sglbench_grid.py and the batch checks of sglbench_driver.py assume every one of these settings.
_RESOLVED_CHECKS: tuple[tuple[str, Callable[[object], bool], str], ...] = (
    # srt/arg_groups/fields/parallel.py ``Parallel``. On a node rank above 0 the engine has no tokenizer
    # manager (srt/entrypoints/engine.py ``Engine._launch_subprocesses``).
    ("nnodes", _only(1), "more than one node is not supported ({name}={v})"),
    ("node_rank", _only(0), "more than one node is not supported ({name}={v})"),
    # ``Parallel``. The deprecated --enable-dp-attention is resolved into this field
    # (srt/arg_groups/parallel_hook.py ``handle_deprecated_dp_attention``). SGLang's own test,
    # srt/runtime_context.py ``attn_dp_enabled_of``, is ``attn_dp_size > 1 or ep_join_mode == "scale"``; the
    # second half is refused with elastic EP below. One batch call cannot place its requests on several
    # attention-DP ranks.
    ("attn_dp_size", _only(1), "DP attention is not supported yet ({name}={v})"),
    # ``Parallel``. Replicas run behind a controller that hands out the requests of a batch one by one
    # (srt/managers/data_parallel_controller.py ``DataParallelController.dispatch_batch_generate``).
    ("dp_size", _only(1), "data-parallel replicas are not supported ({name}={v})"),
    # ``Parallel``. Pipeline parallelism runs another event loop (srt/managers/scheduler.py
    # ``_dispatch_event_loop_once``).
    ("pp_size", _only(1), "pipeline parallelism is not supported ({name}={v})"),
    # ``Parallel``. A prompt is then split over ranks, which the feasibility rules do not model.
    ("attn_cp_size", _only(1), "attention context parallelism is not supported ({name}={v})"),
    ("enable_prefill_cp", _only(False), "attention context parallelism is not supported ({name}={v})"),
    # ``Parallel``. The prefix cache then works in pages of page_size * dcp_size while the scheduler reports
    # the base page size (srt/mem_cache/kv_cache_builder.py ``build_kv_cache``).
    (
        "dcp_size",
        _only(1),
        "decode context parallelism is not supported ({name}={v}); it changes the cache page size",
    ),
    # ``Parallel``. DWDP forces DP attention on and CUDA graphs off (srt/arg_groups/parallel_hook.py
    # ``handle_dwdp``).
    ("dwdp_size", _int_at_most(1), "DWDP is not supported ({name}={v})"),
    # srt/arg_groups/fields/exec_.py ``ExecMoe`` and ``Parallel`` (``max_ep_size``). SGLang tests the backend
    # with ``is not None`` (srt/arg_groups/parallel_hook.py ``handle_elastic_ep``), so the text "none" counts
    # as set there and is refused here.
    ("elastic_ep_backend", _only(None), _ELASTIC_EP_TEXT),
    ("ep_join_mode", _only(None), _ELASTIC_EP_TEXT),
    ("max_ep_size", _only(None), _ELASTIC_EP_TEXT),
    ("enable_elastic_expert_backup", _only(False), _ELASTIC_EP_TEXT),
    # ``ExecMoe``. With any other backend SGLang sends a token to the one rank that holds its expert, and
    # resolution makes ``ep_size`` the TP size by itself (srt/arg_groups/overrides.py ``_a2a_ep_size``); three
    # of those backends also need DP attention (srt/arg_groups/moe_hook.py ``handle_a2a_moe``). The one
    # exception is "ascend_tp", which ``handle_a2a_moe`` resolves to "none" and which so passes here. With "none"
    # every EP rank runs the expert layers over the same tokens and the partial outputs are summed (the
    # comment in srt/arg_groups/parallel_hook.py ``handle_eplb_and_dispatch``).
    ("moe_a2a_backend", _only("none"), "only the MoE all-to-all backend none is supported ({name}={v})"),
    # ``Parallel``. Set, it gives the dense MLP layers a TP size of their own, and the scheduler then gathers
    # and synchronises across ranks (srt/utils/common.py ``require_attn_tp_gather`` and ``require_mlp_sync``).
    ("moe_dense_tp_size", _only(None), "moe_dense_tp_size must not be set (got {v})"),
    # ``ExecMoe``. EPLB places the experts on the ranks anew every ``eplb_rebalance_num_iterations``
    # iterations (the help text of that field), and redundant experts add copies of experts to the ranks
    # (srt/eplb/expert_location.py ``ExpertLocationMetadata._init_common``). Where an expert runs would then
    # depend on the requests that came before.
    ("enable_eplb", _only(False), "EPLB is not supported ({name}={v})"),
    ("ep_num_redundant_experts", _only(0), "redundant experts are not supported ({name}={v})"),
    # srt/arg_groups/fields/spec.py ``Spec`` and srt/arg_groups/fields/disagg.py ``Disagg``. A speculative
    # step commits several tokens and runs draft batches.
    ("speculative_algorithm", _only(None), "speculative decoding is not supported ({name}={v})"),
    ("decoupled_spec_role", _only("null"), "speculative decoding is not supported ({name}={v})"),
    # ``Disagg``. Each of the three runs another event loop (srt/managers/scheduler.py
    # ``_dispatch_event_loop_once``).
    ("disaggregation_mode", _only("null"), _DISAGGREGATION_TEXT),
    ("enable_pdmux", _only(False), _DISAGGREGATION_TEXT),
    ("enable_pd_role_switch", _only(False), _DISAGGREGATION_TEXT),
    # srt/arg_groups/fields/exec_.py ``ExecDllm``. Batches are then formed by another function
    # (srt/managers/scheduler.py ``Scheduler.get_next_batch_to_run``).
    ("dllm_algorithm", _only(None), "diffusion models are not supported ({name}={v})"),
    # srt/arg_groups/fields/serving.py ``Serving``. With more workers the engine talks to a router and not
    # to the tokenizer manager (srt/entrypoints/engine.py ``Engine.__init__``).
    ("tokenizer_worker_num", _only(1), "more than one tokenizer worker is not supported ({name}={v})"),
    # ``Serving``. The token range of the prompts comes from the tokenizer.
    ("skip_tokenizer_init", _only(False), "--skip-tokenizer-init is not supported; the benchmark needs the tokenizer"),
    # srt/arg_groups/fields/memory.py ``Memory``. SGLang switches the cache off by itself for some models.
    # The scheduler's own value is tested again from the recorder headers.
    (
        "disable_radix_cache",
        _only(False),
        "the radix cache is off ({name}={v}); SGLang may have disabled it for this model",
    ),
    # ``Memory``. enable_hierarchical_cache, enable_session_radix_cache and radix_cache_backend pick the
    # cache tree in srt/mem_cache/registry.py ``create_tree_cache``; enable_lmcache, enable_flexkv and
    # enable_unified_cache_external_linker pick it in ``default_radix_cache_factory`` of the same file.
    # hicache_storage_backend adds a storage tier outside the GPU, and enable_hisparse changes how a request
    # passes from prefill to decode (srt/managers/scheduler.py ``Scheduler.get_next_batch_to_run``). Each
    # changes admission or the cached counts.
    ("enable_hierarchical_cache", _only(False), _CACHE_TIER_TEXT),
    ("enable_lmcache", _only(False), _CACHE_TIER_TEXT),
    ("enable_flexkv", _only(False), _CACHE_TIER_TEXT),
    ("enable_unified_cache_external_linker", _only(False), _CACHE_TIER_TEXT),
    ("enable_session_radix_cache", _only(False), _CACHE_TIER_TEXT),
    ("enable_hisparse", _only(False), _CACHE_TIER_TEXT),
    ("radix_cache_backend", _only(None), _CACHE_TIER_TEXT),
    ("hicache_storage_backend", _only(None), _CACHE_TIER_TEXT),
    # srt/arg_groups/fields/schedule.py ``Schedule``. Running decode rows would be merged into the extend
    # batch (srt/managers/scheduler.py ``Scheduler.init_chunked_prefill``, ``is_mixed_chunk``).
    ("enable_mixed_chunk", _only(False), "--enable-mixed-chunk is not supported"),
    # ``Schedule``. Above 1 the scheduler skips receive steps
    # (srt/managers/scheduler_components/recv_skipper.py ``SchedulerRecvSkipper.maybe_create``).
    ("scheduler_recv_interval", _int_at_most(1), "scheduler_recv_interval must be 1 (got {v})"),
    # ``Schedule``. Every other policy reorders the waiting queue (srt/managers/schedule_policy.py
    # ``SchedulePolicy.calc_priority``), and priority scheduling sorts it and opens the preemption path.
    ("schedule_policy", _only("fcfs"), _POLICY_TEXT),
    ("enable_priority_scheduling", _only(False), _POLICY_TEXT),
    # ``Schedule``. Resolution turns None into 0. Above 0 no prefill is admitted for that many rounds after
    # an extend batch (srt/managers/scheduler.py ``Scheduler._should_defer_prefill``).
    (
        "prefill_decode_interval",
        _only(None, 0),
        "prefill_decode_interval must be 0 (got {v}); SGLang sets 22 for some Qwen3-VL setups",
    ),
    # ``Parallel``. It caps running plus new requests per step (srt/managers/scheduler.py
    # ``Scheduler.get_num_allocatable_reqs``). Unset, the scheduler fills in the running cap, which is what
    # the grid uses.
    ("pp_max_micro_batch_size", _only(None), "pp_max_micro_batch_size must not be set (got {v})"),
    # ``Schedule``. Prefill is held back while requests run (srt/managers/min_free_slots_delayer.py
    # ``resolve_min_free_slots``). The scheduler's own value is tested again from the recorder headers.
    ("min_free_slots_delay", _only(None), "min_free_slots_delay must not be set (got {v})"),
    # ``Schedule``. SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE switches it on in resolution; the scheduler then
    # builds a delayer that can veto admission (srt/managers/scheduler.py ``Scheduler.init_schedule_policy``).
    (
        "enable_prefill_delayer",
        _only(False),
        "the prefill delayer is not supported (also check SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE)",
    ),
    # ``Parallel``. Sharding reserves scratch pages for a request in admission
    # (srt/managers/schedule_policy.py ``PrefillAdder._kv_shard_reserve_scratch``).
    ("enable_kv_cache_sharding", _only(False), "KV cache sharding is not supported"),
    # ``Serving``. The first is forced by STREAMING_FLAG; this checks that SGLang took it. With a stream
    # interval of 1 the scheduler sends the output of a request after every step
    # (srt/managers/scheduler_components/output_streamer.py ``_GenerationStreamAccumulator.accept``), which is
    # the step that is timed.
    (
        "incremental_streaming_output",
        _only(True),
        "incremental_streaming_output must be on (got {v}); the launcher passes " + STREAMING_FLAG,
    ),
    ("stream_interval", _only(1), "stream_interval must be 1 (got {v})"),
    # ``Serving``. The tokenizer manager puts these settings under the sampling settings of every request
    # (srt/managers/tokenizer_manager.py ``TokenizerManager._create_tokenized_object``), and a request's own
    # ``stream_interval`` decides over the server's (``_GenerationStreamAccumulator.accept``).
    (
        "preferred_sampling_params",
        _only(None),
        "--preferred-sampling-params is not supported ({name}={v}); it adds sampling settings to every request",
    ),
)

# A field that only one of the two tested releases has, and the value written when the release has none. It
# is recorded in the ``producer`` block and refuses nothing. ``enable_nvidia_megakernel``: v0.5.23 only
# (srt/arg_groups/fields/exec_.py ``ExecKernel``); it changes the kernels of a decode step, which is a serving
# choice. The resolved view raises ``AttributeError`` for a name that is not a field
# (srt/arg_groups/model_override_base.py ``ResolvingConfig.__getattr__``), so the default is taken at v0.5.22.
_ONE_RELEASE_FIELDS = (("enable_nvidia_megakernel", None),)


def _resolved(view: object, name: str) -> object:
    """One resolved field. A field SGLang does not have is an error that names it."""
    try:
        return getattr(view, name)
    except AttributeError as error:
        raise AttributeError(
            f"SGLang's resolved arguments have no field {name!r}; the start-up check that reads it cannot run"
        ) from error


def _check_resolved(view: object) -> None:
    """Refuse the first setting of ``_RESOLVED_CHECKS`` that does not pass."""
    for name, accepts, text in _RESOLVED_CHECKS:
        value = _resolved(view, name)
        if not accepts(value):
            raise _rejected(text.format(name=name, v=_shown(value)))


def _sglang_call(doing: str, call: Callable, *args: object):
    """Run one SGLang step; what SGLang raises to refuse a setup becomes a rejection with its text."""
    try:
        return call(*args)
    except _SGLANG_REFUSALS as error:
        raise _rejected(f"SGLang raised {type(error).__name__} while {doing}: {_one_line(error)}") from error


def _check_expert_parallel(view: object, experts: dict) -> None:
    """Refuse an expert-parallel size the benchmark does not cover.

    srt/arg_groups/fields/parallel.py ``Parallel``: ``ep_size`` is 1 unless the operator sets it. SGLang
    derives the widths of the expert layers from it, ``moe_tp_size = tp_size // ep_size // moe_dp_size``
    (srt/runtime_context.py ``derive_parallel_widths``). At 1 every TP rank holds a slice of every expert; at
    the TP size every rank holds whole experts, its share of them
    (srt/layers/moe/fused_moe_triton/layer.py ``FusedMoE.__init__``). Both keep one scheduler process for each
    TP rank (srt/entrypoints/engine.py ``Engine._launch_scheduler_processes``), which is the number the driver
    waits for. A size between the two mixes the layouts, and AISim plans no cell for it. A model without
    expert layers has nothing the flag could spread, so any size but 1 is a mistake in its command.

    ``moe_dp_size`` (``Parallel``) is not tested here. Both statements above need it to be 1, and SGLang's
    resolution sees to that: it asserts ``moe_dp_size == 1`` whenever the value differs from ``attn_cp_size``
    (srt/arg_groups/parallel_hook.py ``handle_context_parallelism``), and ``attn_cp_size`` must be 1
    (``_RESOLVED_CHECKS``). The assert is SGLang's and is gone when Python runs with ``-O``.
    """
    ep_size, tp_size = _resolved(view, "ep_size"), _resolved(view, "tp_size")
    if _only(1)(ep_size):
        return
    if not experts:
        raise _rejected(f"expert parallelism needs a model with expert layers (ep_size={_shown(ep_size)})")
    if type(ep_size) is not int or type(tp_size) is not int or ep_size != tp_size:
        raise _rejected(f"ep_size must be 1 or the TP size (ep_size={_shown(ep_size)}, tp_size={_shown(tp_size)})")


def _is_text_run(prefill_content: str | None, experts: dict) -> bool:
    """Whether the prompts of this run are dataset text. ``ENV_PREFILL_CONTENT`` says what each value means."""
    if prefill_content is None:
        return bool(experts)
    if prefill_content == SEEDED_CONTENT and experts:
        key, value = next(iter(experts.items()))
        raise _rejected(
            f"{ENV_PREFILL_CONTENT}={SEEDED_CONTENT} is not supported for a model with expert layers "
            f"({key}={_shown(value)}): the routing of a token to its experts depends on the content"
        )
    return prefill_content == TEXT_CONTENT


def _start_up_checks(sglang_argv: list[str], environ: Mapping[str, str], producer: dict) -> tuple[object, object, dict]:
    """Parse and resolve SGLang's arguments and refuse what the benchmark does not cover.

    Returns ``(server_args, view, experts)``: SGLang's record, the view of its resolved values, and the
    expert keys of the model config. ``main`` decides from the keys where the prompts come from
    (``_is_text_run``) and then tests the expert-parallel layout (``_check_expert_parallel``). Nothing is
    built here, so a refused setup costs no model load. Resolution does probe the device and reads the model
    config.
    """
    # The release is tested first, so that an untested one is refused by name and not by a field it lacks.
    version = compat.sglang_version()
    producer["sglang_version"] = version
    if version not in compat.TESTED_VERSIONS:
        raise _rejected(f"SGLang {version} is not a tested version (tested: {', '.join(compat.TESTED_VERSIONS)})")
    # The order of SGLang's own launcher (launch_server.py): plugins first, because a plugin may replace a
    # resolution step, then parse, then resolve before anything is started.
    _sglang_call("loading its plugins", compat.load_plugins)
    try:
        server_args = _sglang_call("parsing its arguments", compat.parse_server_args, sglang_argv)
    except _StopRequested:
        raise
    except SystemExit as error:
        # SGLang's parser ends the process for an unknown flag or a missing model path, after it has printed
        # its usage. The run exists by now, so this is written as a failed artifact like every other refusal.
        raise _rejected(
            f"SGLang's argument parser exited with status {error.code!r}; its message is on stderr"
        ) from error
    _check_sglang_environment(environ)
    view = _sglang_call("resolving its arguments", compat.resolve, server_args)
    _check_resolved(view)

    # srt/runtime_context.py ``PlatformContext`` and srt/arg_groups/fields/device.py ``Device``. A ROCm build
    # also names its device "cuda", and SGLang has admission rules for AMD only
    # (srt/managers/schedule_policy.py), so all three are tested.
    platform = compat.platform_facts()
    device = _resolved(view, "device")
    if platform["is_cuda"] is not True or platform["is_hip"] is not False or device != "cuda":
        raise _rejected(
            f"only NVIDIA CUDA is supported (device={_shown(device)}, is_cuda={_shown(platform['is_cuda'])}, "
            f"is_hip={_shown(platform['is_hip'])}); SGLang has AMD-only admission rules"
        )
    # Whether the model has expert layers decides where the prompts come from: in such a model the routing of
    # a token to its experts depends on the content, so rows measured with seeded ids would not stand for
    # served text. Only the model config is read for this, never the weights.
    experts = _sglang_call("reading the model config", compat.model_expert_keys, server_args)
    producer["expert_keys"] = _json_ready(experts)
    producer["moe"] = _json_ready({name: _resolved(view, name) for name in _MOE_RECORDED_FIELDS}) if experts else None

    # How the output of a request is streamed: every request is sent with ``stream=True``
    # (sglbench_compat.py, ``generate_batch``), and the two server settings are the resolved ones.
    producer["streaming"] = {
        "stream": True,
        "incremental_streaming_output": _resolved(view, "incremental_streaming_output"),
        "stream_interval": _resolved(view, "stream_interval"),
    }
    for name, absent in _ONE_RELEASE_FIELDS:
        producer[name] = getattr(view, name, absent)
    return server_args, view, experts


def _resolved_settings(view: object, producer: dict) -> dict:
    """The run settings that only SGLang's resolved arguments give.

    srt/entrypoints/engine.py ``Engine._launch_scheduler_processes`` starts one scheduler process for each
    pipeline rank and each TP rank. Pipeline parallelism and every data-parallel layout are refused above, so
    the number of scheduler processes is ``tp_size``. The two request caps stay None in SGLang unless the
    operator sets them (srt/arg_groups/fields/schedule.py ``Schedule``); the driver compares them with the
    largest batch once the grid exists.
    """
    tp_size = _resolved(view, "tp_size")
    if type(tp_size) is not int or tp_size < 1:
        raise TypeError(f"SGLang's resolved tp_size is {tp_size!r}, not a positive int")
    producer["tp_size"] = tp_size
    return {
        "expected_scheduler_processes": tp_size,
        "prefill_max_requests": _resolved(view, "prefill_max_requests"),
        "max_queued_requests": _resolved(view, "max_queued_requests"),
    }


# ---------------------------------------------------------------------------------------------------------------
# The dump of the resolved arguments
# ---------------------------------------------------------------------------------------------------------------


# Fields of SGLang's resolved arguments that hold a secret. SGLang itself hides the three watermark fields
# before it logs the record or serves it (srt/sampling/watermarking/core.py ``redact_watermark_secrets``, used
# by srt/entrypoints/engine.py ``_server_args_for_logging``); the two keys are in
# srt/arg_groups/fields/serving.py ``Serving``. The dump is copied to the collector host.
_SECRET_FIELDS = ("api_key", "admin_api_key", "watermark_key", "watermark_key_b", "watermark_config")


def _json_ready(value: object) -> object:
    """``value`` as strict JSON data. What JSON cannot hold is written as its ``repr``.

    SGLang's resolved arguments are not all JSON: the record can hold callables and other objects
    (srt/server_args.py ``ServerArgs.resolved_dict``).
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {
            key if isinstance(key, str) else f"<{type(key).__name__}> {key!r}": _json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return repr(value)


def _write_dump(path: str, payload: dict) -> None:
    """Write the dump so that the file appears complete or not at all.

    AISim's collector parses every ``resolved-config*.json`` under the results directory as one JSON object
    (collector/fpm_forward/database.py ``_validate_backend_markers``). The temporary file starts with a dot
    and ends in ``.tmp``, so it matches neither that pattern nor ``benchmark*.json``.
    """
    text = json.dumps(_json_ready(payload), sort_keys=True, allow_nan=False)
    directory, name = os.path.split(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = os.path.join(directory, f".{name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _dump_resolved_arguments(options: _Options, run_id: str, version: str, server_args: object) -> None:
    """Write SGLang's resolved arguments and the launcher's own settings to ``--dump-config-to``.

    The dump is written after the arguments are resolved and before the engine is built, so a run that fails
    in the build still leaves it. It never holds the environment: the file is copied to the collector host.
    """
    benchmark = {
        "mode": options.mode,
        # The name only: the directory is a place of the machine the launcher ran on.
        "points_file": None if options.points_file is None else os.path.basename(options.points_file),
        "points_sha256": options.points_sha256,
        **options.cap_values,
        "warmup_iterations": options.warmup_iterations,
        "timeout": options.timeout,
        "output_path": options.output_path,
        "content_seed": options.content_seed,
        "worker_id": options.worker_id,
        "run_id": run_id,
    }
    resolved = dict(compat.resolved_dict(server_args))
    for name in _SECRET_FIELDS:
        if resolved.get(name) is not None:
            resolved[name] = "<redacted>"
    payload = {
        "producer": artifact.PRODUCER_NAME,
        "sglang_version": version,
        "config": {"server_args": resolved, "benchmark": benchmark},
    }
    try:
        _write_dump(options.dump_config_to, payload)
    except OSError as error:
        raise _rejected(f"--dump-config-to {options.dump_config_to!r}: {_described(error)}") from error


# ---------------------------------------------------------------------------------------------------------------
# The dataset and the text pool of a text run
# ---------------------------------------------------------------------------------------------------------------


def _unavailable(what: str) -> _Failed:
    return _Failed(f"{driver.REASON_DATASET_UNAVAILABLE}: {what}")


def _sha256_of_file(path: str) -> str:
    """The SHA-256 of a file as hex digits, read piece by piece."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            piece = handle.read(_FILE_PIECE_BYTES)
            if not piece:
                return digest.hexdigest()
            digest.update(piece)


def _require_sha256(path: str, wanted: str | None, what: str) -> str:
    """Hash ``path`` and return the digest; fail the run when a digest is wanted and it is another one."""
    digest = _sha256_of_file(path)
    if wanted and digest != wanted:
        raise _unavailable(f"{what} {path} has the SHA-256 {digest}, expected {wanted}")
    return digest


def _download(url: str, target: str) -> None:
    """Write what ``url`` serves into the file ``target``.

    ``urllib.request.urlopen`` with a socket timeout, as Dynamo's ``_kvwarm_resolve_dataset`` calls it: an
    endpoint that stalls fails the download and does not hold the launcher. Nothing is tried a second time.
    ``urllib`` takes proxy settings from the process environment by itself. ``target`` must not exist: the file
    is created, never opened over another one.

    A body that ends before the length the server announced is an ``OSError`` here. ``http.client`` itself
    reports none for a read with a size: ``HTTPResponse.read(amt)`` returns an empty piece, as at the end of
    a whole body (CPython 3.12, http/client.py).
    """
    with urllib.request.urlopen(url, timeout=DATASET_DOWNLOAD_TIMEOUT_SECONDS) as response, open(target, "xb") as out:
        written = 0
        while True:
            piece = response.read(_FILE_PIECE_BYTES)
            if not piece:
                break
            out.write(piece)
            written += len(piece)
        headers = getattr(response, "headers", None)
        announced = None if headers is None else headers.get("Content-Length")
        if announced is not None and announced.strip().isdigit() and int(announced) != written:
            raise OSError(f"the transfer ended after {written} of the {announced.strip()} bytes the server announced")


def _resolve_dataset(environ: Mapping[str, str]) -> dict:
    """Find the dataset file of a text run and hash it: ``{"path", "sha256", "url", "downloaded"}``.

    ``environ`` holds the three dataset variables and HF_HOME; the comment at ``ENV_KVWARM_DATASET`` says what
    each means. The steps are those of Dynamo's ``_kvwarm_resolve_dataset``:

    - a value that does not start with ``http://`` or ``https://`` is a local file. It must exist. It is hashed
      and, when the hash variable is set, compared with it; Dynamo does not compare a local file;
    - a URL is looked for in the cache directory under the last part of its path. A file that is there is
      hashed and compared with the expected digest. A file with another digest fails the run; it is neither
      deleted nor fetched again, as in Dynamo;
    - otherwise the URL is downloaded into ``<file>.<pid>.<random>.part`` beside its place, hashed, and renamed
      into place. Several launchers may fetch the same file into one directory at once, and the rename is
      atomic. Dynamo's name carries the process id only, which tells the ranks of one host apart; launchers
      in two pods that share a volume can have the same process id, so the name here has a random part too.
      A part with another digest is deleted, and so is the part of a download that broke off or ended short.

    Whatever goes wrong fails the run with ``dataset_unavailable: <what>``: a missing file, another digest, a
    directory that cannot be made or written, and an ``OSError`` (``urllib.error.URLError`` and a socket
    timeout are of that kind), a ``ValueError`` or an ``http.client`` exception of the download.
    """
    spec = environ.get(ENV_KVWARM_DATASET) or DEFAULT_DATASET_URL
    given = environ.get(ENV_KVWARM_SHA256) or None
    try:
        if not spec.startswith(("http://", "https://")):
            if not os.path.isfile(spec):
                raise _unavailable(f"the dataset path {spec} is not an existing file")
            digest = _require_sha256(spec, given, "the dataset file")
            return {"path": spec, "sha256": digest, "url": None, "downloaded": False}
        cache_root = environ.get(ENV_KVWARM_CACHE_DIR)
        if not cache_root:
            hf_home = environ.get(ENV_HF_HOME)
            cache_root = os.path.join(hf_home, os.pardir, "fpm_datasets") if hf_home else DEFAULT_DATASET_CACHE_DIR
        cache_root = os.path.abspath(cache_root)
        os.makedirs(cache_root, exist_ok=True)
        cached = os.path.join(cache_root, os.path.basename(spec.split("?")[0]) or "kvwarm_dataset.json")
        wanted = given or (DEFAULT_DATASET_SHA256 if spec == DEFAULT_DATASET_URL else None)
        if os.path.exists(cached):
            digest = _require_sha256(cached, wanted, "the cached dataset file")
            return {"path": cached, "sha256": digest, "url": spec, "downloaded": False}
        part = f"{cached}.{os.getpid()}.{secrets.token_hex(8)}.part"
        _say(f"downloading the dataset {spec} to {cached}")
        try:
            _download(spec, part)
            digest = _require_sha256(part, wanted, "the downloaded dataset file")
            os.replace(part, cached)
        except BaseException:
            try:
                os.unlink(part)
            except FileNotFoundError:
                pass
            raise
        return {"path": cached, "sha256": digest, "url": spec, "downloaded": True}
    except (OSError, ValueError, http.client.HTTPException) as error:
        raise _unavailable(_described(error)) from error


def _load_dataset(environ: Mapping[str, str]) -> tuple[dict, list[str]]:
    """The record of the dataset of a text run and its texts. It runs before the engine is built.

    The file is found and hashed (``_resolve_dataset``), read whole and parsed as JSON, and its texts are
    taken by ``sglbench_content.conversation_texts``, which restates the recipe of Dynamo's
    ``_kvwarm_load_texts``. The record has the four keys Dynamo writes into its ``kvwarm`` block (``path``,
    ``sha256``, ``collection_pool``, ``conversations``, the number of kept texts) and two of this module:
    ``url``, the URL the dataset variable named or None for a local file, and ``downloaded``, whether this run
    fetched it. A file this run did not fetch is whatever lay in the cache directory under that name; its
    digest is recorded, and compared when a digest is expected. A file that cannot be read or parsed fails
    the run with ``dataset_unavailable: <what>``, and one that gives no text with ``dataset_empty``.
    """
    resolved = _resolve_dataset(environ)
    path = resolved["path"]
    try:
        with open(path, encoding="utf-8") as handle:
            records = json.loads(handle.read())
        texts = content.conversation_texts(records)
    except (OSError, ValueError) as error:
        raise _unavailable(f"{path}: {_described(error)}") from error
    if not texts:
        _say(f"the dataset {path} gives no text: no record has a text that the recipe keeps")
        raise _Failed(driver.REASON_DATASET_EMPTY)
    record = {
        "path": path,
        "sha256": resolved["sha256"],
        "collection_pool": content.COLLECTION_POOL,
        "conversations": len(texts),
        "url": resolved["url"],
        "downloaded": resolved["downloaded"],
    }
    return record, texts


def _build_text_pool(engine: object, texts: list[str], id_limit: int) -> content.TextPool:
    """Turn texts into token ids with the served tokenizer until the text pool is large enough.

    The texts are taken in order, ``POOL_ENCODE_BATCH`` in one tokenizer call, until the pool holds
    ``POOL_TOKEN_TARGET`` tokens or no text is left. The pool keeps the ids of a text as the tokenizer returns
    them, the id of a special token included, which a tokenizer gives for the written form of that token
    inside a text. Dynamo's vLLM self-benchmark sends them in the same way (``_kvwarm_chain_token_ids``).
    The pool leaves out, and counts, only an id at or above ``id_limit``, the number of ids the served model
    counts: SGLang does not check input ids, and what the model does with such an id was not read. The
    tokenizer is the engine's own, in the launcher's process; the scheduler processes are idle meanwhile,
    because no request has been sent yet.

    Whatever the tokenizer raises fails the run with ``tokenizer_unavailable: <what>``, and a pool that kept no
    conversation with ``dataset_empty``. An ``id_limit`` the pool refuses is ``token_range_too_small``, the
    reason the driver gives for a token range it refuses.
    """
    started = time.monotonic()
    try:
        pool = content.TextPool([], 0, id_limit)
    except ValueError as error:
        # The pool words its refusal for a token range. The range it was given here is the model's ids.
        raise _Failed(
            f"{driver.REASON_TOKEN_RANGE}: the model counts {id_limit} token ids, which a text pool refuses: "
            f"{_one_line(error)}"
        ) from error
    encoded = 0
    try:
        tokenizer = compat.tokenizer_class(engine)
        while encoded < len(texts) and pool.tokens < POOL_TOKEN_TARGET:
            batch = texts[encoded : encoded + POOL_ENCODE_BATCH]
            pool.add(compat.encode_texts(engine, batch))
            encoded += len(batch)
    except Exception as error:
        raise _Failed(f"{driver.REASON_TOKENIZER_UNAVAILABLE}: {_described(error)}") from error
    _say(
        f"the text pool holds {pool.tokens} tokens in {pool.conversations} conversations: {encoded} of "
        f"{len(texts)} texts went through {tokenizer} in {time.monotonic() - started:.1f} s, and "
        f"{pool.dropped_tokens} token ids at or above {id_limit}, the number of ids the model counts, and "
        f"{pool.dropped_conversations} conversations without an id below it were dropped"
    )
    if pool.conversations == 0:
        raise _Failed(driver.REASON_DATASET_EMPTY)
    return pool


# ---------------------------------------------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------------------------------------------


class _EngineAdapter:
    """The engine as the driver sees it: ``generate(prompts, rids, max_new_tokens)``, one batch call."""

    def __init__(self, engine: object, stops: _SignalStops) -> None:
        self._engine = engine
        self._stops = stops

    def generate(self, prompts: list[list[int]], rids: list[str], max_new_tokens: int) -> list[dict]:
        try:
            return compat.generate_batch(self._engine, prompts, rids, max_new_tokens, ENGINE_CALL_TIMEOUT_SECONDS)
        finally:
            # SGLang puts a handler of its own on SIGTERM in its first request; see ``_SignalStops.reclaim``.
            self._stops.reclaim()


def _build_engine(server_args: object) -> object:
    """Build SGLang's engine with the recorder's entry for its scheduler processes.

    The engine comes from SGLang's own parser and its own ``Engine`` class, so the model, the kernels and the
    scheduler are the ones ``sglang serve`` would run with these arguments.

    Deviation note (engine in the launcher's process, recorder in the scheduler processes).
    What differs: serving puts an HTTP server in front of the tokenizer manager; here the launcher calls the
    engine in its own process. Every scheduler process is started through ``sglbench_recorder.recorder_entry``
    in the place of SGLang's entry, which makes the scheduler a subclass that writes one line per batch.
    Direction of the error: none known from the missing HTTP server, which runs in another process than the
    scheduler in serving too. The recorder's own cost is described at its code (sglbench_recorder.py).
    How to retire it: nothing to retire; it is how the batches are observed.
    """
    try:
        return compat.build_engine(server_args, recorder.recorder_entry)
    except _BUILD_REFUSALS as error:
        raise _rejected(
            f"SGLang raised {type(error).__name__} while building the engine: {_one_line(error)}"
        ) from error
    except Exception as error:
        raise _Failed(f"{driver.REASON_ENGINE_CALL}: the engine build raised {_described(error)}") from error


def _token_range(engine: object) -> tuple[int, int]:
    """The ordinary token ids of the served tokenizer, as ``(low, high)`` with ``high`` exclusive.

    SGLang does not check input ids against the vocabulary, so the ids the benchmark makes up itself, the
    seeded ids and the allocated first tokens, are neither special nor added tokens. A tokenizer that cannot
    say which those are is refused. The ids of dataset text are not held to this range (``_text_id_limit``).
    """
    try:
        return driver.ordinary_token_range(**compat.vocabulary(engine))
    except Exception as error:
        raise _rejected(f"the token range cannot be taken from the tokenizer: {_described(error)}") from error


def _text_id_limit(engine: object) -> int:
    """The number of token ids the served model counts: the ids of a text must lie below it.

    A text keeps every id its tokenizer returned, as in Dynamo's vLLM self-benchmark. This is the one bound
    left: a tokenizer can hold more ids than the model counts.
    """
    try:
        return compat.model_vocabulary_size(engine)
    except Exception as error:
        raise _rejected(f"the vocabulary size cannot be taken from the model config: {_described(error)}") from error


# ---------------------------------------------------------------------------------------------------------------
# Result files, the final wait and the exit code
# ---------------------------------------------------------------------------------------------------------------


def _remove_old_outputs(output_path: str) -> None:
    """Delete the rank file and the merged file of an earlier run, and nothing else.

    The results directory also holds files of the collector and of its gate (the provenance file, the points
    receipt, logs), which are not the launcher's. Under the gate nothing is found here: it refuses to start
    when a result file exists (collector/fpm_forward/runtime/fpm_exec.sh).
    """
    for path in _result_paths(output_path):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _stay_alive(seconds: float) -> None:
    """Wait for ``seconds`` on the monotonic clock. A sleep that returns early is taken up again."""
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(remaining)


def _exit_code(status: str | None, reason: str | None) -> int:
    """The exit code of a run with this outcome. A failure is told apart by the first word of its reason."""
    if status == artifact.STATUS_COMPLETE:
        return EXIT_COMPLETE
    if status == artifact.STATUS_PARTIAL:
        return EXIT_PARTIAL
    if reason is not None and reason.startswith(f"{driver.REASON_REJECTED}: "):
        return EXIT_REJECTED
    if reason is not None and reason.startswith(f"{driver.REASON_INTERNAL}: "):
        return EXIT_INTERNAL
    return EXIT_FAILED


def _finish(run: driver.BenchmarkRun, engine: object | None, stops: _SignalStops, producer: dict) -> int:
    """Shut the engine down, write the result files, print the outcome, wait, and return the exit code.

    The order matters. The shutdown comes first: it ends the scheduler processes, so the recorder logs no
    longer change when the gate accepts the result files and the collector copies the results directory. The
    write does not depend on the shutdown having worked.
    """
    if engine is not None:
        try:
            compat.shutdown(engine)
        except BaseException as error:
            # srt/entrypoints/engine.py ``Engine.shutdown`` raises RuntimeError when its child processes are
            # not gone after 60 s. That is recorded and printed; a complete run stays complete.
            producer["engine_shutdown_error"] = _described(error)
            _say(f"the engine shutdown failed: {_described(error)}")

    # From here on a signal changes nothing: the gate sends SIGTERM once it has accepted the files, and a
    # second write would replace files it has already read.
    stops.ignore()
    if stops.raised is not None:
        # Recorded already when the stop reached ``main``; this is for a stop that was lost on its way.
        run.record_failure(f"{REASON_STOPPED}: signal {stops.raised}")
    if stops.late:
        run.record_failure(f"{REASON_STOPPED}: signal {stops.late[0]} before the result files were written")

    write_error = None
    paths: list[str] = []
    try:
        paths = run.write()["paths"]
    except Exception as error:
        write_error = _described(error)
    status, reason = run.status, run.reason

    if write_error is not None:
        if status == artifact.STATUS_FAILED:
            _say(f"FAILED: {reason}")
            _say(f"the failed artifact could not be written: {write_error}")
        else:
            status = artifact.STATUS_FAILED
            reason = f"{driver.REASON_INTERNAL}: the result files could not be written: {write_error}"
            _say(f"FAILED: {reason}")
    elif status == artifact.STATUS_COMPLETE:
        _say(f"complete: wrote {', '.join(paths)}")
    elif status == artifact.STATUS_PARTIAL:
        _say(f"PARTIAL: the soft deadline ended the run ({reason}); wrote {', '.join(paths)}")
    else:
        _say(f"FAILED: {reason}")
        _say(f"wrote the failed artifact {', '.join(paths)}")

    # Nothing is written during the wait and nothing can end the process: see FINAL_WAIT_SECONDS.
    _stay_alive(FINAL_WAIT_SECONDS)
    return _exit_code(status, reason)


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run one benchmark and return the exit code. The module docstring gives the order of the steps."""
    # The clock is read before anything else, and before SGLang is imported: the gate's own deadline counts
    # from the start of the process, so the soft deadline must count the engine build as well.
    started_at = time.monotonic()
    options = _read_options(sys.argv[1:] if argv is None else list(argv), os.environ)
    if options.output_problem is not None:
        _say(f"FAILED: {driver.REASON_REJECTED}: {options.output_problem}")
        _say("no result file was written: the output path names no file")
        return EXIT_REJECTED

    # New for every launch, as in Dynamo's vLLM benchmark: AISim refuses a run id it has seen before.
    run_id = uuid.uuid4().hex
    # Facts of the launcher for the ``producer`` block. The dict is filled in as the run goes on; the driver
    # reads it when the artifact is built.
    producer: dict = {
        "engine_call_timeout_seconds": ENGINE_CALL_TIMEOUT_SECONDS,
        "points_file": (
            None
            if options.points_file is None
            else {"name": os.path.basename(options.points_file), "sha256": options.points_sha256}
        ),
    }
    settings = driver.RunSettings(
        mode=options.mode,
        # A run with refused caps is never executed, so the record of defaults that stands in here is not read.
        caps=grid.SampleCaps() if options.caps is None else options.caps,
        manifest=options.manifest,
        warmup_iterations=options.warmup_iterations,
        timeout_seconds=float(options.timeout),
        output_path=options.output_path,
        run_id=run_id,
        content_seed=options.content_seed,
        worker_id=options.worker_id,
        # Provisional: these three come from SGLang's resolved arguments and are replaced after the checks.
        expected_scheduler_processes=1,
        prefill_max_requests=None,
        max_queued_requests=None,
        config=_config_block(options),
        producer=producer,
        # What the operator asked for by name. Without that word the model decides, once its config was read.
        content_source=(
            content.CONTENT_TEXT_CHAINS if options.prefill_content == TEXT_CONTENT else content.CONTENT_SEEDED_IDS
        ),
    )
    # The recorder directory is new for every run and absolute: the scheduler processes read its name from
    # the environment.
    recorder_dir = os.path.join(os.path.dirname(os.path.abspath(options.output_path)), RECORDER_DIR_NAME, run_id)

    def new_run(run_settings: driver.RunSettings) -> driver.BenchmarkRun:
        return driver.BenchmarkRun(
            run_settings, recorder_dir, started_at=started_at, clock=time.monotonic, sleep=time.sleep
        )

    run = new_run(settings)
    stops = _SignalStops()
    engine = None
    try:
        try:
            stops.install()
            # The failed artifact of a refused flag needs the directory too; the gate makes it, a shell does not.
            os.makedirs(os.path.dirname(os.path.abspath(options.output_path)), exist_ok=True)
            _remove_old_outputs(options.output_path)
            if options.rejection is not None:
                raise _rejected(options.rejection)
            os.makedirs(recorder_dir)
            os.environ[recorder.ENV_RECORDER_DIR] = recorder_dir
            _say(f"run {run_id}: mode {options.mode}, output {options.output_path}, recorder logs in {recorder_dir}")

            server_args, view, experts = _start_up_checks(options.sglang_argv, os.environ, producer)
            text_run = _is_text_run(options.prefill_content, experts)
            # From here on the run knows its content source, so that a text run that is refused for its layout
            # or fails on its dataset, before any prompt exists, is still written as a text run.
            source = content.CONTENT_TEXT_CHAINS if text_run else content.CONTENT_SEEDED_IDS
            settings = dataclasses.replace(settings, content_source=source)
            run = new_run(settings)
            _check_expert_parallel(view, experts)
            run = new_run(dataclasses.replace(settings, **_resolved_settings(view, producer)))
            if options.dump_config_to is not None:
                _dump_resolved_arguments(options, run_id, producer["sglang_version"], server_args)

            dataset, texts, pool = None, None, None
            if text_run:
                # Before the engine is built: a dataset that cannot be had then costs seconds and no model load.
                dataset, texts = _load_dataset(options.dataset_environ)
                _say(
                    f"text run: the dataset {dataset['path']} (SHA-256 {dataset['sha256']}) gives "
                    f"{dataset['conversations']} texts"
                )
            _say(f"SGLang {producer['sglang_version']}: the start-up checks passed; building the engine")
            engine = _build_engine(server_args)
            token_low, token_high = _token_range(engine)
            if text_run:
                # Only the allocated first tokens come from the token range: a text keeps its own ids.
                _say(
                    "the engine is built; the first token of each prompt part comes from the token ids "
                    f"[{token_low}, {token_high}), the other tokens are dataset text"
                )
                pool = _build_text_pool(engine, texts, _text_id_limit(engine))
                # The pool holds the token ids; the texts are not needed again.
                texts = None
            else:
                _say(f"the engine is built; prompts use the token ids [{token_low}, {token_high})")
            run.execute(
                _EngineAdapter(engine, stops),
                token_low=token_low,
                token_high=token_high,
                text_pool=pool,
                dataset=dataset,
            )
        finally:
            # From here on a signal is noted and no longer raised, so that what follows cannot be interrupted.
            stops.disarm()
    except _Failed as failure:
        run.record_failure(failure.reason)
    except _StopRequested as stop:
        run.record_failure(f"{REASON_STOPPED}: {stop.what}")
    except Exception as error:
        # Not a failure the launcher knows: the traceback goes to stderr for whoever has to find the cause.
        traceback.print_exc(file=sys.stderr)
        run.record_failure(f"{driver.REASON_INTERNAL}: {_described(error)}")
    except BaseException as error:
        # SystemExit and KeyboardInterrupt that the launcher did not raise itself: see ``_StopRequested``.
        run.record_failure(f"{REASON_STOPPED}: {_stop_text(error)}")
    return _finish(run, engine, stops, producer)


if __name__ == "__main__":
    sys.exit(main())
