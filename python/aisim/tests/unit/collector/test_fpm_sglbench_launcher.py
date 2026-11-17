# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# The fake resolved arguments, scheduler, batch and hook registry follow SGLang at v0.5.22 and v0.5.23; see
# THIRD_PARTY_NOTICES.md.
# Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d
# Source: https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90
# The expected benchmark flags, their defaults and their environment variables follow ai-dynamo/dynamo
# (Apache-2.0, https://github.com/ai-dynamo/dynamo), components/src/dynamo/vllm/backend_args.py and
# instrumented_scheduler.py at b20fe6fe94021b854597965c7305931175e8f482. See THIRD_PARTY_NOTICES.md.

"""CPU-only test of ``runtime/sglbench_launcher.py``. No SGLang, no GPU and no engine are involved.

The launcher, the driver, the recorder, the grid, the content plan and the artifact module are the real ones,
loaded flat from the runtime directory as the pod loads them. Everything that stands in for SGLang is FAKE and
is named so:

- A FAKE ``sglang`` package tree (``_fake_sglang_modules``) holds the names the benchmark imports from SGLang:
  a FAKE parser, a FAKE record of server arguments with a FAKE resolving view, FAKE platform facts, a FAKE
  model config, a FAKE hook registry, a FAKE scheduler entry and a FAKE engine class. A fixture puts the tree
  in ``sys.modules`` and pytest takes it out again. ``_FakeWorld`` holds what the fakes return and notes the
  order of the calls.
- The end-to-end tests replace no module of the benchmark. ``main`` runs with the real functions of
  ``sglbench_compat``, which reach the FAKE package: the engine is the subclass the real ``build_engine``
  makes of the FAKE engine class, and every call goes through the real ``generate_batch`` on an event loop.
- The other tests replace the functions of ``sglbench_compat`` by FAKE ones (``_install_fakes``), which note
  their calls and can be made to raise. They show what the launcher asks of that module and in which order.
- In both kinds ``_FakeEngine`` starts its FAKE scheduler processes as SGLang does: it calls the scheduler
  entry it was given with seven positional arguments. That entry is the launcher's choice, the REAL
  ``sglbench_recorder.recorder_entry``. It registers the REAL recorder class with the FAKE hook registry and
  calls the FAKE ``run_scheduler_process``, which builds the class on ``FakeBaseScheduler``. Each engine call
  then makes the fake engine call ``run_batch`` and ``process_batch_result`` of that object in the order of
  SGLang's overlap loop. So every line of the recorder log is written by the real recorder, and the real
  driver reads it.
- ``FakeBaseScheduler``, ``FakeBatch``, ``FakeRequest``, ``FakeResult`` and ``FakeCopyDone`` carry the
  attributes the recorder reads from SGLang's objects.
- The cached-token counts come from ``PageCacheModel``, the model of SGLang's prefix cache written in
  ``test_fpm_sglbench_content.py``. They are model results, not measurements.
- Every time is FAKE. ``_FakeClock`` counts ticks of 2**-20 s and moves only when a FAKE batch runs or
  something sleeps on it. The launcher, the driver and the recorder all read it in the place of the real
  clock; each FAKE scheduler process reads it with an offset of its own. The batch durations are the
  ``FAKE_*`` constants. They were typed in to exercise the arithmetic and were never measured. Every artifact
  these tests write carries the worker id ``FAKE-cell-node0`` in each of its records and stays under pytest's
  temporary directory.
- Signals are real, sent to the test process with ``signal.raise_signal``, and only while a handler of the
  launcher or ``SIG_IGN`` is in place. A fixture restores the handlers after every test.

The two checkers are used as shipped, through the helpers of ``test_fpm_sglbench_artifact.py``: the validator
(``validate_native_collection``, called as the host calls it for a cell of an SGLang plan) and the gate (the
Python text inside ``check_result_files()`` of ``runtime/fpm_exec.sh``). ``_pod`` writes the collected layout
with the provenance file of an SGLang cell: ``"backend": "sglang"`` and the release the FAKE ``sglang`` package
reports, which the validator compares with the release the launcher wrote into its artifact.

The names and the texts the tests expect are typed here on their own: the flags, their defaults and variables
(Dynamo's vLLM worker), the fields of SGLang's resolved arguments with their defaults (the field files of
SGLang at both tags) and the refusal of each.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import dataclasses
import enum
import fnmatch
import hashlib
import inspect
import json
import os
import shutil
import signal
import subprocess
import sys
import types
from pathlib import Path

import pytest

from .test_fpm_runner import _cell
from .test_fpm_sglbench_artifact import GATE_ACCEPT, GATE_INVALID, _gate, _pod, _validate
from .test_fpm_sglbench_content import PageCacheModel

RUNTIME = Path(__file__).resolve().parents[3] / "collector/fpm_forward/runtime"
MODULE_PATH = RUNTIME / "sglbench_launcher.py"
# In the pod the runtime files lie flat in one directory and import each other by bare name. The directory goes
# to the END of sys.path, so that it shadows nothing.
if str(RUNTIME) not in sys.path:
    sys.path.append(str(RUNTIME))

import sglbench_artifact as artifact
import sglbench_compat as compat
import sglbench_content as content
import sglbench_driver as driver
import sglbench_grid as grid
import sglbench_launcher as launcher
import sglbench_recorder as recorder

pytestmark = pytest.mark.unit

# The real writer of the result files. The tests put a wrapper around it that notes when it ran.
REAL_WRITE_ARTIFACTS = artifact.write_artifacts
# The functions of sglbench_compat the launcher calls, as they are before a test replaces them.
REAL_COMPAT = {
    name: getattr(compat, name)
    for name in (
        "sglang_version",
        "load_plugins",
        "parse_server_args",
        "resolve",
        "resolved_dict",
        "platform_facts",
        "model_expert_keys",
        "build_engine",
        "generate_batch",
        "vocabulary",
        "shutdown",
    )
}

RECORDER_ENV = "AISIM_SGLBENCH_RECORDER_DIR"
STREAMING_FLAG = "--incremental-streaming-output"
FORBIDDEN_FILE_PATTERNS = ("benchmark*.json", "resolved-config*.json", "fpm-*.json", "collector-provenance.json")

# FAKE run inputs.
FAKE_SGLANG_VERSION = "0.5.23"
FAKE_SGLANG_ARGV = ["--model-path", "fake/dense-model", "--page-size", "1"]
FAKE_WORKER_ID = "FAKE-cell-node0"
FAKE_PORT_ARGS = "FAKE port args"
FAKE_FIRST_PID = 4100
FAKE_VOCABULARY = {"size": 60_000, "excluded": [0, 1, 2, 50_000, 50_001, 59_999]}
# FAKE: the model's own bound is a little above the tokenizer's, so the smaller of the two is the tokenizer's.
FAKE_MODEL_VOCAB_SIZE = 60_010
# The longest run of ids in [0, 60000) without an excluded id.
FAKE_TOKEN_RANGE = [3, 50_000]
FAKE_TIMEOUT_SECONDS = 10_800

# FAKE timings, in clock ticks of 2**-20 s. An extend batch takes a base and one tick per computed token; the
# extend batch of a warm shot takes a little longer, so that a sample taken from the wrong shot would show.
# Decode batch i of a call takes (base + requests) * tenths[i]. Decode steps 3 to 5 are the items 2, 3 and 4,
# and their upper median is ten tenths.
TICKS_PER_SECOND = 1 << 20
FAKE_EXTEND_BASE_TICKS = 4096
FAKE_WARM_SHOT_EXTRA_TICKS = 500
FAKE_DECODE_BASE_TICKS = 300
FAKE_DECODE_STEP_TENTHS = (6, 15, 12, 9, 10, 3)
# FAKE host time the scheduler spends in one ``run_batch`` call.
FAKE_HOST_TICKS = 1
# Nothing sleeps this often on the fake clock in a test; a wait without end fails the test instead of hanging it.
FAKE_SLEEP_LIMIT = 20_000

# Few values on every axis: most tests need a run, not a dense grid.
SMALL_CAPS = grid.SampleCaps(
    prefill_max_new_token_samples=3,
    prefill_max_kv_read_token_samples=3,
    decode_max_kv_read_token_samples=3,
    decode_max_batch_size_samples=3,
    prefix_max_batch_size_samples=2,
)
SMALL_CAP_FLAGS = [
    "--prefill-max-new-token-samples=3",
    "--prefill-max-kv-read-token-samples=3",
    "--decode-max-kv-read-token-samples=3",
    "--decode-max-batch-size-samples=3",
    "--prefix-max-batch-size-samples=2",
]

# The keys of the ``producer`` block: the launcher's own in a run without a shutdown error, and the driver's.
LAUNCHER_PRODUCER_KEYS = (
    "engine_call_timeout_seconds",
    "points_file",
    "sglang_version",
    "streaming",
    "enable_nvidia_megakernel",
    "tp_size",
)
DRIVER_PRODUCER_KEYS = (
    "timing_metric",
    "timing_clock",
    "staging",
    "recorder_format",
    "scheduler_processes",
    "limits",
    "page_size",
    "enable_overlap",
    "content",
    "restaged_points",
    "self_test",
    "first_token_need",
)

# The fields of SGLang's resolved arguments that the launcher reads, with the value a default single-GPU dense
# setup resolves to. Typed from the field files of SGLang at both tags (srt/arg_groups/fields: parallel.py,
# schedule.py, memory.py, serving.py, exec_.py, spec.py, disagg.py, device.py); ``device`` and
# ``prefill_decode_interval`` are None there and resolution fills them in. ``enable_nvidia_megakernel`` exists
# at v0.5.23 only.
FAKE_RESOLVED_ARGUMENTS = {
    "nnodes": 1,
    "node_rank": 0,
    "tp_size": 1,
    "attn_dp_size": 1,
    "dp_size": 1,
    "pp_size": 1,
    "attn_cp_size": 1,
    "enable_prefill_cp": False,
    "dcp_size": 1,
    "dwdp_size": 1,
    "elastic_ep_backend": None,
    "ep_join_mode": None,
    "max_ep_size": None,
    "enable_elastic_expert_backup": False,
    "speculative_algorithm": None,
    "decoupled_spec_role": "null",
    "disaggregation_mode": "null",
    "enable_pdmux": False,
    "enable_pd_role_switch": False,
    "dllm_algorithm": None,
    "tokenizer_worker_num": 1,
    "skip_tokenizer_init": False,
    "disable_radix_cache": False,
    "enable_hierarchical_cache": False,
    "enable_lmcache": False,
    "enable_flexkv": False,
    "enable_unified_cache_external_linker": False,
    "enable_session_radix_cache": False,
    "enable_hisparse": False,
    "radix_cache_backend": None,
    "hicache_storage_backend": None,
    "enable_mixed_chunk": False,
    "scheduler_recv_interval": 1,
    "schedule_policy": "fcfs",
    "enable_priority_scheduling": False,
    "prefill_decode_interval": 0,
    "pp_max_micro_batch_size": None,
    "min_free_slots_delay": None,
    "enable_prefill_delayer": False,
    "enable_kv_cache_sharding": False,
    "incremental_streaming_output": True,
    "stream_interval": 1,
    "preferred_sampling_params": None,
    "prefill_max_requests": None,
    "max_queued_requests": None,
    "device": "cuda",
    "enable_nvidia_megakernel": False,
}

_TIER = "only the built-in in-GPU radix cache is supported"
_ELASTIC = "elastic EP is not supported"
_DISAGG = "disaggregated serving is not supported"
_POLICY = "schedule policy must be fcfs without priority scheduling"
_SPEC = "speculative decoding is not supported"
_CP = "attention context parallelism is not supported"

# One refused value for every field the launcher tests on the resolved arguments, with the whole refusal.
STAGE_S_REFUSALS = [
    ("nnodes", 2, "more than one node is not supported (nnodes=2)"),
    ("node_rank", 1, "more than one node is not supported (node_rank=1)"),
    ("attn_dp_size", 2, "DP attention is not supported yet (attn_dp_size=2)"),
    ("dp_size", 2, "data-parallel replicas are not supported (dp_size=2)"),
    ("pp_size", 2, "pipeline parallelism is not supported (pp_size=2)"),
    ("attn_cp_size", 2, f"{_CP} (attn_cp_size=2)"),
    ("enable_prefill_cp", True, f"{_CP} (enable_prefill_cp=true)"),
    ("dcp_size", 2, "decode context parallelism is not supported (dcp_size=2); it changes the cache page size"),
    ("dwdp_size", 2, "DWDP is not supported (dwdp_size=2)"),
    ("elastic_ep_backend", "mooncake", f"{_ELASTIC} (elastic_ep_backend=mooncake)"),
    ("ep_join_mode", "scale", f"{_ELASTIC} (ep_join_mode=scale)"),
    ("max_ep_size", 8, f"{_ELASTIC} (max_ep_size=8)"),
    ("enable_elastic_expert_backup", True, f"{_ELASTIC} (enable_elastic_expert_backup=true)"),
    ("speculative_algorithm", "EAGLE", f"{_SPEC} (speculative_algorithm=EAGLE)"),
    ("decoupled_spec_role", "draft", f"{_SPEC} (decoupled_spec_role=draft)"),
    ("disaggregation_mode", "prefill", f"{_DISAGG} (disaggregation_mode=prefill)"),
    ("enable_pdmux", True, f"{_DISAGG} (enable_pdmux=true)"),
    ("enable_pd_role_switch", True, f"{_DISAGG} (enable_pd_role_switch=true)"),
    ("dllm_algorithm", "LowConfidence", "diffusion models are not supported (dllm_algorithm=LowConfidence)"),
    ("tokenizer_worker_num", 2, "more than one tokenizer worker is not supported (tokenizer_worker_num=2)"),
    ("skip_tokenizer_init", True, "--skip-tokenizer-init is not supported; the benchmark needs the tokenizer"),
    (
        "disable_radix_cache",
        True,
        "the radix cache is off (disable_radix_cache=true); SGLang may have disabled it for this model",
    ),
    ("enable_hierarchical_cache", True, f"{_TIER} (enable_hierarchical_cache=true)"),
    ("enable_lmcache", True, f"{_TIER} (enable_lmcache=true)"),
    ("enable_flexkv", True, f"{_TIER} (enable_flexkv=true)"),
    ("enable_unified_cache_external_linker", True, f"{_TIER} (enable_unified_cache_external_linker=true)"),
    ("enable_session_radix_cache", True, f"{_TIER} (enable_session_radix_cache=true)"),
    ("enable_hisparse", True, f"{_TIER} (enable_hisparse=true)"),
    ("radix_cache_backend", "flexkv", f"{_TIER} (radix_cache_backend=flexkv)"),
    ("hicache_storage_backend", "file", f"{_TIER} (hicache_storage_backend=file)"),
    ("enable_mixed_chunk", True, "--enable-mixed-chunk is not supported"),
    ("scheduler_recv_interval", 2, "scheduler_recv_interval must be 1 (got 2)"),
    ("schedule_policy", "lpm", f"{_POLICY} (schedule_policy=lpm)"),
    ("enable_priority_scheduling", True, f"{_POLICY} (enable_priority_scheduling=true)"),
    (
        "prefill_decode_interval",
        22,
        "prefill_decode_interval must be 0 (got 22); SGLang sets 22 for some Qwen3-VL setups",
    ),
    ("pp_max_micro_batch_size", 8, "pp_max_micro_batch_size must not be set (got 8)"),
    ("min_free_slots_delay", 4, "min_free_slots_delay must not be set (got 4)"),
    (
        "enable_prefill_delayer",
        True,
        "the prefill delayer is not supported (also check SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE)",
    ),
    ("enable_kv_cache_sharding", True, "KV cache sharding is not supported"),
    (
        "incremental_streaming_output",
        False,
        "incremental_streaming_output must be on (got false); the launcher passes --incremental-streaming-output",
    ),
    ("stream_interval", 2, "stream_interval must be 1 (got 2)"),
    (
        "preferred_sampling_params",
        {"stream_interval": 8},
        "--preferred-sampling-params is not supported (preferred_sampling_params={'stream_interval': 8}); "
        "it adds sampling settings to every request",
    ),
]

# More values that must be refused: a value of another type is never taken for the allowed one.
STAGE_S_MORE_REFUSALS = [
    ("nnodes", True, "more than one node is not supported (nnodes=true)"),
    ("nnodes", None, "more than one node is not supported (nnodes=null)"),
    ("node_rank", False, "more than one node is not supported (node_rank=false)"),
    ("attn_dp_size", 0, "DP attention is not supported yet (attn_dp_size=0)"),
    ("dp_size", None, "data-parallel replicas are not supported (dp_size=null)"),
    ("dwdp_size", None, "DWDP is not supported (dwdp_size=null)"),
    ("dwdp_size", True, "DWDP is not supported (dwdp_size=true)"),
    # SGLang takes the text "none" for a backend that is set (it tests ``is not None``).
    ("elastic_ep_backend", "none", f"{_ELASTIC} (elastic_ep_backend=none)"),
    ("elastic_ep_backend", False, f"{_ELASTIC} (elastic_ep_backend=false)"),
    ("enable_elastic_expert_backup", None, f"{_ELASTIC} (enable_elastic_expert_backup=null)"),
    ("speculative_algorithm", "", f"{_SPEC} (speculative_algorithm=)"),
    ("decoupled_spec_role", None, f"{_SPEC} (decoupled_spec_role=null)"),
    ("disaggregation_mode", None, f"{_DISAGG} (disaggregation_mode=null)"),
    ("enable_mixed_chunk", None, "--enable-mixed-chunk is not supported"),
    ("enable_mixed_chunk", 0, "--enable-mixed-chunk is not supported"),
    ("scheduler_recv_interval", None, "scheduler_recv_interval must be 1 (got null)"),
    ("schedule_policy", "FCFS", f"{_POLICY} (schedule_policy=FCFS)"),
    (
        "prefill_decode_interval",
        False,
        "prefill_decode_interval must be 0 (got false); SGLang sets 22 for some Qwen3-VL setups",
    ),
    ("pp_max_micro_batch_size", 0, "pp_max_micro_batch_size must not be set (got 0)"),
    ("min_free_slots_delay", 0, "min_free_slots_delay must not be set (got 0)"),
    (
        "incremental_streaming_output",
        1,
        "incremental_streaming_output must be on (got 1); the launcher passes --incremental-streaming-output",
    ),
    ("stream_interval", True, "stream_interval must be 1 (got true)"),
    ("stream_interval", "1", "stream_interval must be 1 (got 1)"),
    ("stream_interval", 0, "stream_interval must be 1 (got 0)"),
]
# Values other than the default that pass.
STAGE_S_ACCEPTED = [
    ("dwdp_size", 0),
    ("scheduler_recv_interval", 0),
    ("scheduler_recv_interval", -1),
    ("prefill_decode_interval", None),
]

# SGLang's own variables: what is refused, with the whole refusal, and what passes. SGLang reads a boolean
# variable as true for true, 1, yes and y in any case (srt/environ.py ``EnvBool``), takes its default for a
# value it cannot read, and reads SGLANG_ENABLE_COLOCATED_BATCH_GEN with another function that knows only true
# and 1 (srt/utils/common.py ``get_bool_env_var``).
_TOKEN_LIMIT = "SGLANG_MAX_NEW_TOKENS_LIMIT={} is below the 7 output tokens of a decode run"
_PER_POLL = "SGLANG_SCHEDULER_MAX_RECV_PER_POLL must be unset (got {})"
_BLOCKER = "SGLANG_ENABLE_COLOCATED_BATCH_GEN must be unset without DP attention"
SGLANG_ENV_REFUSALS = [
    ("SGLANG_MAX_NEW_TOKENS_LIMIT", "6", _TOKEN_LIMIT.format(6)),
    ("SGLANG_MAX_NEW_TOKENS_LIMIT", "1", _TOKEN_LIMIT.format(1)),
    ("SGLANG_MAX_NEW_TOKENS_LIMIT", " 3 ", _TOKEN_LIMIT.format(3)),
    ("SGLANG_RADIX_FORCE_MISS", "1", "SGLANG_RADIX_FORCE_MISS disables every cache hit"),
    ("SGLANG_RADIX_FORCE_MISS", "true", "SGLANG_RADIX_FORCE_MISS disables every cache hit"),
    ("SGLANG_RADIX_FORCE_MISS", "YES", "SGLANG_RADIX_FORCE_MISS disables every cache hit"),
    ("SGLANG_RADIX_FORCE_MISS", "y", "SGLANG_RADIX_FORCE_MISS disables every cache hit"),
    ("SGLANG_TEST_RETRACT", "True", "SGLANG_TEST_RETRACT is a test switch and must be unset"),
    ("SGLANG_TEST_SCRIPTED_RUNTIME", "1", "SGLANG_TEST_SCRIPTED_RUNTIME is a test switch and must be unset"),
    ("SGLANG_SCHEDULER_MAX_RECV_PER_POLL", "0", _PER_POLL.format(0)),
    ("SGLANG_SCHEDULER_MAX_RECV_PER_POLL", "4", _PER_POLL.format(4)),
    ("SGLANG_ENABLE_COLOCATED_BATCH_GEN", "1", _BLOCKER),
    ("SGLANG_ENABLE_COLOCATED_BATCH_GEN", "TRUE", _BLOCKER),
]
SGLANG_ENV_ACCEPTED = [
    ("SGLANG_MAX_NEW_TOKENS_LIMIT", "7"),
    ("SGLANG_MAX_NEW_TOKENS_LIMIT", "4096"),
    ("SGLANG_MAX_NEW_TOKENS_LIMIT", "0"),
    ("SGLANG_MAX_NEW_TOKENS_LIMIT", "-1"),
    ("SGLANG_MAX_NEW_TOKENS_LIMIT", "many"),
    ("SGLANG_MAX_NEW_TOKENS_LIMIT", ""),
    ("SGLANG_RADIX_FORCE_MISS", "0"),
    ("SGLANG_RADIX_FORCE_MISS", "false"),
    ("SGLANG_RADIX_FORCE_MISS", "No"),
    ("SGLANG_RADIX_FORCE_MISS", "maybe"),
    ("SGLANG_RADIX_FORCE_MISS", ""),
    ("SGLANG_RADIX_FORCE_MISS", " 1"),
    ("SGLANG_TEST_RETRACT", "0"),
    ("SGLANG_TEST_SCRIPTED_RUNTIME", "false"),
    ("SGLANG_SCHEDULER_MAX_RECV_PER_POLL", "-1"),
    ("SGLANG_SCHEDULER_MAX_RECV_PER_POLL", "all"),
    ("SGLANG_ENABLE_COLOCATED_BATCH_GEN", "0"),
    ("SGLANG_ENABLE_COLOCATED_BATCH_GEN", "false"),
    ("SGLANG_ENABLE_COLOCATED_BATCH_GEN", "yes"),
    ("SGLANG_ENABLE_COLOCATED_BATCH_GEN", ""),
]

_IMBALANCED = "is not supported: every request of a point is the same here"
_KDA = "is not supported: every decode context here is computed by real requests"
_HYBRID = "is not supported: models with a recurrent state are refused"
_REAL_STATE = "is not supported: the benchmark stages real cached state only"
_LEGACY = [
    ("--benchmark-prefill-granularity", "DYN_BENCHMARK_PREFILL_GRANULARITY", "--prefill-max-new-token-samples"),
    (
        "--benchmark-prefill-kv-read-granularity",
        "DYN_BENCHMARK_PREFILL_KV_READ_GRANULARITY",
        "--prefill-max-kv-read-token-samples",
    ),
    (
        "--benchmark-prefill-batch-granularity",
        "DYN_BENCHMARK_PREFILL_BATCH_GRANULARITY",
        "--prefix-max-batch-size-samples",
    ),
    (
        "--benchmark-decode-length-granularity",
        "DYN_BENCHMARK_DECODE_LENGTH_GRANULARITY",
        "--decode-max-kv-read-token-samples",
    ),
    (
        "--benchmark-decode-batch-granularity",
        "DYN_BENCHMARK_DECODE_BATCH_GRANULARITY",
        "--decode-max-batch-size-samples",
    ),
]

# The launcher's own flags and variables: (flags, variables, the refusal).
OWN_REFUSALS = [
    (["--benchmark-collect-imbalanced"], {}, f"--benchmark-collect-imbalanced {_IMBALANCED}"),
    ([], {"DYN_BENCHMARK_COLLECT_IMBALANCED": "true"}, f"DYN_BENCHMARK_COLLECT_IMBALANCED {_IMBALANCED}"),
    ([], {"DYN_BENCHMARK_COLLECT_IMBALANCED": " On "}, f"DYN_BENCHMARK_COLLECT_IMBALANCED {_IMBALANCED}"),
    ([], {"DYN_FPM_BENCH_COLLECT_IMBALANCED": "1"}, f"DYN_FPM_BENCH_COLLECT_IMBALANCED {_IMBALANCED}"),
    # Dynamo's second variable decides over the flag.
    (
        ["--no-benchmark-collect-imbalanced"],
        {"DYN_FPM_BENCH_COLLECT_IMBALANCED": "Yes"},
        f"DYN_FPM_BENCH_COLLECT_IMBALANCED {_IMBALANCED}",
    ),
    ([], {"DYN_FPM_BENCH_COLLECT_IMBALANCED": "maybe"}, "DYN_FPM_BENCH_COLLECT_IMBALANCED='maybe' is not a boolean"),
    ([], {"DYN_FPM_BENCH_COLLECT_IMBALANCED": ""}, "DYN_FPM_BENCH_COLLECT_IMBALANCED='' is not a boolean"),
    (["--benchmark-randomize-kda-state"], {}, f"--benchmark-randomize-kda-state {_KDA}"),
    ([], {"DYN_BENCHMARK_RANDOMIZE_KDA_STATE": "1"}, f"DYN_BENCHMARK_RANDOMIZE_KDA_STATE {_KDA}"),
    (["--benchmark-hybrid-live-state"], {}, f"--benchmark-hybrid-live-state {_HYBRID}"),
    ([], {"DYN_BENCHMARK_HYBRID_LIVE_STATE": "yes"}, f"DYN_BENCHMARK_HYBRID_LIVE_STATE {_HYBRID}"),
    *(([flag, "8"], {}, f"{flag} is not supported; use {use}") for flag, _variable, use in _LEGACY),
    *(([], {variable: "8"}, f"{variable} is not supported; use {use}") for _flag, variable, use in _LEGACY),
    ([], {"DYN_BENCHMARK_DECODE_BATCH_GRANULARITY": ""}, _LEGACY[4][1] + " is not supported; use " + _LEGACY[4][2]),
    (["--benchmark-prefill-granularity", "many"], {}, f"{_LEGACY[0][0]} is not supported; use {_LEGACY[0][2]}"),
    (["--benchmark-warmup-iterations", "-1"], {}, "--benchmark-warmup-iterations must be at least 0, got -1"),
    ([], {"DYN_BENCHMARK_WARMUP_ITERATIONS": "-2"}, "--benchmark-warmup-iterations must be at least 0, got -2"),
    (["--benchmark-timeout", "0"], {}, "--benchmark-timeout must be above 0, got 0"),
    ([], {"DYN_BENCHMARK_TIMEOUT": "-5"}, "--benchmark-timeout must be above 0, got -5"),
    ([], {"DYN_BENCHMARK_TIMEOUT": "soon"}, "DYN_BENCHMARK_TIMEOUT='soon' is not an integer"),
    ([], {"DYN_PREFILL_MAX_NEW_TOKEN_SAMPLES": ""}, "DYN_PREFILL_MAX_NEW_TOKEN_SAMPLES='' is not an integer"),
    (
        ["--prefill-max-new-token-samples", "1"],
        {},
        "sample caps: prefill_max_new_token_samples must be >= 2, got 1",
    ),
    (
        ["--prefill-max-kv-read-token-samples", "1"],
        {},
        "sample caps: prefill_max_kv_read_token_samples must be >= 2, got 1",
    ),
    (
        ["--decode-max-kv-read-token-samples", "0"],
        {},
        "sample caps: decode_max_kv_read_token_samples must be >= 2, got 0",
    ),
    (
        [],
        {"DYN_DECODE_MAX_BATCH_SIZE_SAMPLES": "1"},
        "sample caps: decode_max_batch_size_samples must be >= 2, got 1",
    ),
    (
        ["--prefix-max-batch-size-samples", "0"],
        {},
        "sample caps: prefix_max_batch_size_samples must be >= 1, got 0",
    ),
    (["--benchmark-max-batch-size", "0"], {}, "sample caps: benchmark_max_batch_size must be >= 1, got 0"),
    (["--dump-config-to", ""], {}, "--dump-config-to must name a file"),
    ([], {"DYN_BENCH_CONTENT_SEED": ""}, "DYN_BENCH_CONTENT_SEED='' must be a non-empty UTF-8 string"),
    ([], {"DYN_BENCH_KV_WARMUP": "off"}, f"DYN_BENCH_KV_WARMUP='off' {_REAL_STATE} (accepted: on, 1, true)"),
    ([], {"DYN_BENCH_KV_WARMUP": "0"}, f"DYN_BENCH_KV_WARMUP='0' {_REAL_STATE} (accepted: on, 1, true)"),
    ([], {"DYN_BENCH_KV_WARMUP": "False"}, f"DYN_BENCH_KV_WARMUP='False' {_REAL_STATE} (accepted: on, 1, true)"),
    ([], {"DYN_BENCH_KV_WARMUP": ""}, f"DYN_BENCH_KV_WARMUP='' {_REAL_STATE} (accepted: on, 1, true)"),
    ([], {"DYN_BENCH_KV_WARMUP": "auto"}, f"DYN_BENCH_KV_WARMUP='auto' {_REAL_STATE} (accepted: on, 1, true)"),
    (
        [],
        {"DYN_BENCH_PREFILL_REAL_SEED": "off"},
        f"DYN_BENCH_PREFILL_REAL_SEED='off' {_REAL_STATE} (accepted: on, 1, true, auto)",
    ),
    (
        [],
        {"DYN_BENCH_PREFILL_REAL_SEED": "yes"},
        f"DYN_BENCH_PREFILL_REAL_SEED='yes' {_REAL_STATE} (accepted: on, 1, true, auto)",
    ),
    (
        [],
        {"DYN_BENCH_PREFILL_CONTENT": "sharegpt"},
        "DYN_BENCH_PREFILL_CONTENT='sharegpt' is not supported: prompts are seeded token ids (random)",
    ),
    (
        [],
        {"DYN_BENCH_PREFILL_CONTENT": "Random"},
        "DYN_BENCH_PREFILL_CONTENT='Random' is not supported: prompts are seeded token ids (random)",
    ),
    ([], {"DYN_BENCH_POOL_TAG": "a"}, "DYN_BENCH_POOL_TAG is not supported (it is set to 'a')"),
    (
        [],
        {"DYN_BENCH_GIANT_KV_THRESHOLD": "1000"},
        "DYN_BENCH_GIANT_KV_THRESHOLD is not supported (it is set to '1000')",
    ),
    ([], {"DYN_BENCH_GIANT_KV_REPEATS": "3"}, "DYN_BENCH_GIANT_KV_REPEATS is not supported (it is set to '3')"),
]

# Flags and variables that are accepted and change nothing.
OWN_ACCEPTED = [
    (["--no-benchmark-collect-imbalanced"], {}),
    (["--no-benchmark-randomize-kda-state"], {}),
    (["--no-benchmark-hybrid-live-state"], {}),
    # The flag decides over its own variable.
    (["--no-benchmark-collect-imbalanced"], {"DYN_BENCHMARK_COLLECT_IMBALANCED": "true"}),
    (["--no-benchmark-hybrid-live-state"], {"DYN_BENCHMARK_HYBRID_LIVE_STATE": "1"}),
    # Dynamo's second variable decides over the flag, also to switch it off.
    (["--benchmark-collect-imbalanced"], {"DYN_FPM_BENCH_COLLECT_IMBALANCED": "off"}),
    ([], {"DYN_BENCHMARK_COLLECT_IMBALANCED": "true", "DYN_FPM_BENCH_COLLECT_IMBALANCED": " No "}),
    ([], {"DYN_BENCHMARK_COLLECT_IMBALANCED": "false"}),
    ([], {"DYN_BENCHMARK_COLLECT_IMBALANCED": "maybe"}),
    ([], {"DYN_BENCHMARK_RANDOMIZE_KDA_STATE": ""}),
    ([], {"DYN_BENCH_KV_WARMUP": "on"}),
    ([], {"DYN_BENCH_KV_WARMUP": " ON "}),
    ([], {"DYN_BENCH_KV_WARMUP": "1"}),
    ([], {"DYN_BENCH_KV_WARMUP": "True"}),
    ([], {"DYN_BENCH_PREFILL_REAL_SEED": "on"}),
    ([], {"DYN_BENCH_PREFILL_REAL_SEED": "auto"}),
    ([], {"DYN_BENCH_PREFILL_REAL_SEED": "TRUE"}),
    ([], {"DYN_BENCH_PREFILL_CONTENT": "random"}),
    ([], {"DYN_BENCH_PREFILL_CONTENT": ""}),
    ([], {"DYN_BENCH_POOL_TAG": "", "DYN_BENCH_GIANT_KV_THRESHOLD": "", "DYN_BENCH_GIANT_KV_REPEATS": ""}),
    # Ignored in this version, and the FPM_* variables of the gate are never read.
    (
        [],
        {
            "DYN_BENCH_KV_WARMUP_DATASET": "/data/x",
            "DYN_BENCH_KV_WARMUP_CACHE_DIR": "/cache",
            "DYN_BENCH_KV_WARMUP_SHA256": "abc",
            "DYN_DUMP_CONFIG_TO": "/results/resolved-config-node0.json",
            "FPM_BENCHMARK_MODE": "decode",
            "FPM_BENCHMARK_OUTPUT_PATH": "/elsewhere/benchmark.json",
            "FPM_RUN_ID": "cell",
        },
    ),
]


# ---------------------------------------------------------------------------------------------------------------
# FAKE clock
# ---------------------------------------------------------------------------------------------------------------


class _FakeClock:
    """A FAKE monotonic clock in whole ticks of 2**-20 s, so the difference of two readings is exact."""

    def __init__(self) -> None:
        self.ticks = 7000 * TICKS_PER_SECOND
        self.sleeps: list[float] = []
        self.on_sleep: list = []
        # FAKE: the part of a requested sleep that passes before it returns.
        self.sleep_fraction = 1.0

    def monotonic(self) -> float:
        return self.ticks / TICKS_PER_SECOND

    def advance(self, ticks: int) -> None:
        self.ticks += ticks

    def sleep(self, seconds: float) -> None:
        """FAKE sleep: nothing waits, the fake clock moves."""
        self.sleeps.append(seconds)
        assert len(self.sleeps) <= FAKE_SLEEP_LIMIT, "something waits without end"
        for hook in list(self.on_sleep):
            hook(seconds)
        self.ticks += max(1, round(seconds * self.sleep_fraction * TICKS_PER_SECOND))


# ---------------------------------------------------------------------------------------------------------------
# FAKE SGLang objects of a scheduler process
# ---------------------------------------------------------------------------------------------------------------


class FakeForwardMode(enum.IntEnum):
    """FAKE of SGLang's ForwardMode: an IntEnum whose member names the recorder logs."""

    EXTEND = 1
    DECODE = 2
    MIXED = 3
    IDLE = 4


class FakeRequest:
    """FAKE of SGLang's Req: only ``rid``."""

    def __init__(self, rid: str) -> None:
        self.rid = rid


@dataclasses.dataclass
class FakeBatch:
    """FAKE of SGLang's ScheduleBatch with the fields the recorder reads, and with its ``copy``."""

    reqs: list
    forward_mode: FakeForwardMode
    forward_iter: int | None = None
    launch_ts: float | None = None
    extend_num_tokens: int | None = None
    prefix_lens: list | None = None
    extend_lens: list | None = None

    def copy(self) -> FakeBatch:
        return dataclasses.replace(self, reqs=self.reqs[:])


class FakeCopyDone:
    """FAKE of the copy event of a result: waiting on it moves the fake clock to the end of the FAKE GPU work."""

    def __init__(self, world: _FakeWorld, done_ticks: int) -> None:
        self._world = world
        self._done_ticks = done_ticks

    def synchronize(self) -> None:
        self._world.clock.ticks = max(self._world.clock.ticks, self._done_ticks)


class FakeResult:
    """FAKE of SGLang's GenerationBatchResult: only ``copy_done``."""

    def __init__(self, copy_done: FakeCopyDone) -> None:
        self.copy_done = copy_done


@dataclasses.dataclass
class _FakeProcess:
    """One FAKE scheduler process: its TP rank, its FAKE process id and the origin of its own FAKE clock."""

    tp_rank: int
    pid: int
    clock_offset_ticks: int
    parallel: types.SimpleNamespace


class FakeBaseScheduler:
    """FAKE of SGLang's Scheduler: the attributes the recorder reads and the two methods it wraps.

    ``run_batch`` stamps ``forward_iter`` and ``launch_ts`` on the batch as the stock method does and returns a
    result with a copy event.
    """

    def __init__(self, server_args: _FakeServerArgs, port_args: object) -> None:
        assert port_args == FAKE_PORT_ARGS
        world = server_args.world
        self.fake_world = world
        self.fake_process = world.current_process
        limits = world.limits
        self.page_size = limits.page_size
        self.max_req_input_len = limits.max_req_input_len
        self.max_total_num_tokens = limits.max_total_num_tokens
        self.max_running_requests = limits.max_running_requests
        self.max_prefill_tokens = limits.max_prefill_tokens
        self.chunked_prefill_size = limits.chunked_prefill_size
        self.tp_worker = types.SimpleNamespace(
            model_runner=types.SimpleNamespace(
                decode_cuda_graph_runner=types.SimpleNamespace(capture_bs=list(limits.decode_graph_sizes)),
                prefill_cuda_graph_runner=types.SimpleNamespace(capture_num_tokens=list(limits.prefill_graph_sizes)),
            )
        )
        self.enable_overlap = True
        # Only the process with TP rank 0 reads the request socket.
        self.recv_from_tokenizer = object() if self.fake_process.tp_rank == 0 else None
        self.disable_radix_cache = False
        self.is_hybrid_swa = False
        self.is_hybrid_ssm = False
        self.max_queued_requests = None
        self.max_new_tokens_limit = None
        self.schedule_policy = "fcfs"
        self.enable_priority_scheduling = False
        self.is_mixed_chunk = False
        self.prefill_decode_interval = 0
        self.prefill_delayer = None
        self.min_free_slots_delayer = None
        self.recv_skipper = None
        self.input_blocker = None
        for name, value in world.scheduler_attributes.items():
            setattr(self, name, value)
        self.forward_ct = 0

    def run_batch(self, batch: FakeBatch) -> FakeResult:
        world = self.fake_world
        self.forward_ct += 1
        batch.forward_iter = self.forward_ct
        batch.launch_ts = world.scheduler_seconds()
        world.clock.advance(FAKE_HOST_TICKS)
        return FakeResult(FakeCopyDone(world, world.step_done_ticks))

    def process_batch_result(self, batch: FakeBatch, result: FakeResult) -> None:
        return None


class _FakeHookType(enum.Enum):
    REPLACE = "replace"


def _fake_sglang_modules(world: _FakeWorld) -> dict[str, types.ModuleType]:
    """A FAKE ``sglang`` package tree with only the names ``sglbench_recorder`` and ``sglbench_compat`` import.

    The recorder's names are used in every launch. The names of ``sglbench_compat`` are reached only when a
    test leaves the real compat functions in place.
    """

    def module(name: str, **attributes: object) -> types.ModuleType:
        made = types.ModuleType(name)
        made.__path__ = []  # a package, so that its submodules can be imported
        for key, value in attributes.items():
            setattr(made, key, value)
        return made

    registered: list[tuple] = []

    class FakeHookRegistry:
        """FAKE of SGLang's HookRegistry: it keeps what was registered."""

        @staticmethod
        def register(target: str, hook: object, hook_type: object) -> None:
            registered.append((target, hook, hook_type))

    scheduler_module = module("sglang.srt.managers.scheduler", Scheduler=FakeBaseScheduler)

    def run_scheduler_process(server_args, port_args, gpu_id, tp_rank, pp_rank, dp_rank, pipe_writer):
        """FAKE of SGLang's scheduler entry: apply the class replacement, build the scheduler, report it.

        The real entry then runs the event loop for the life of the process. Here the fake engine keeps the
        scheduler object and calls its two batch methods itself.
        """
        world.entry_calls.append((gpu_id, tp_rank, pp_rank, dp_rank))
        stock = scheduler_module.Scheduler
        for target, hook, hook_type in registered:
            assert (target, hook_type) == ("sglang.srt.managers.scheduler.Scheduler", _FakeHookType.REPLACE)
            scheduler_module.Scheduler = hook
        try:
            scheduler = scheduler_module.Scheduler(server_args, port_args)
        finally:
            # The next FAKE process starts with the stock class and an empty registry, as a spawned one does.
            scheduler_module.Scheduler = stock
            registered.clear()
        pipe_writer.send(scheduler)

    scheduler_module.run_scheduler_process = run_scheduler_process

    # -- what sglbench_compat reaches

    def load_plugins() -> None:
        world.note("load_plugins")

    def prepare_server_args(argv):
        """FAKE of the parser of ``sglang serve``: a handful of its flags. An unknown flag exits 2."""
        world.note("prepare_server_args")
        parser = argparse.ArgumentParser(prog="sglang serve (FAKE)")
        parser.add_argument("--model-path", required=True)
        parser.add_argument("--tp-size", type=int, default=1)
        parser.add_argument("--page-size", type=int, default=1)
        parser.add_argument("--stream-interval", type=int, default=1)
        parser.add_argument(STREAMING_FLAG, action="store_true")
        parsed = parser.parse_args(argv)
        world.sglang_argv = list(argv)
        # The FAKE engine follows the flags, so a flag that does not reach this parser shows in the run.
        world.use(parsed.page_size, parsed.tp_size)
        world.resolved["stream_interval"] = parsed.stream_interval
        world.resolved["incremental_streaming_output"] = parsed.incremental_streaming_output
        world.server_args = FakeSglangServerArgs(world, list(argv))
        return world.server_args

    def get_platform():
        world.note("get_platform")
        return types.SimpleNamespace(**world.platform)

    class FakeModelConfig:
        """FAKE of SGLang's ModelConfig: the two Hugging Face configs, here one object."""

        @classmethod
        def from_server_args(cls, server_args):
            assert server_args is world.server_args and server_args.resolved
            world.note("from_server_args")
            config = types.SimpleNamespace(vocab_size=FAKE_MODEL_VOCAB_SIZE, **world.experts)
            return types.SimpleNamespace(hf_config=config, hf_text_config=config, vocab_size=FAKE_MODEL_VOCAB_SIZE)

    class FakeSglangEngine:
        """FAKE of ``sglang.srt.entrypoints.engine.Engine``, the class sglbench_compat subclasses.

        It has the seam (``run_scheduler_process_func``, read through ``self``), an event loop of its own,
        ``async_generate`` in its batch form with a stream of chunks, a tokenizer manager, and ``shutdown``.
        """

        run_scheduler_process_func = staticmethod(run_scheduler_process)

        def __init__(self, **kwargs) -> None:
            server_args = kwargs["server_args"]
            assert server_args is world.server_args and server_args.resolved
            world.note("Engine")
            # SGLang takes a loop of its own when none is running.
            self.loop = asyncio.new_event_loop()
            self.fake = _FakeEngine(world, server_args, self.run_scheduler_process_func)
            self.tokenizer_manager = types.SimpleNamespace(
                tokenizer=_FakeTokenizer(), model_config=types.SimpleNamespace(vocab_size=FAKE_MODEL_VOCAB_SIZE)
            )

        async def async_generate(
            self, prompt=None, sampling_params=None, input_ids=None, stream=False, rid=None, **more
        ):
            world.engine_requests.append(
                {"prompt": prompt, "sampling_params": dict(sampling_params), "stream": stream, "more": more}
            )
            assert stream is True
            return self._fake_stream(input_ids, rid, sampling_params["max_new_tokens"])

        async def _fake_stream(self, prompts, rids, max_new_tokens):
            """FAKE stream of a batch call: one chunk per request and output token, requests interleaved."""
            fault = world.call_faults.get(_FakeCall(prompts, rids, max_new_tokens, None, []).key)
            if fault == "sglang_exit":
                # FAKE of SGLang after SIGTERM: a task of its own on the loop ends the process with sys.exit(0).
                async def fake_sigterm_watchdog() -> None:
                    sys.exit(0)

                world.tasks.append(self.loop.create_task(fake_sigterm_watchdog()))
                await asyncio.sleep(0)
            responses = self.fake.fake_generate(prompts, rids, max_new_tokens, None)
            for step in range(1, max_new_tokens + 1):
                for index, response in enumerate(responses):
                    finish = None
                    if step == max_new_tokens:
                        finish = {"type": "length", "length": max_new_tokens}
                        if fault == "abort" and index == 0:
                            finish = {"type": "abort", "message": "FAKE abort", "status_code": 400, "err_type": None}
                    yield {
                        "index": index,
                        "text": "",
                        "output_ids": [step - 1],
                        "meta_info": {
                            "id": response["rid"],
                            "finish_reason": finish,
                            "prompt_tokens": response["prompt_tokens"],
                            "cached_tokens": response["cached_tokens"],
                            "completion_tokens": step,
                            "num_retractions": 0,
                        },
                    }

        def get_all_child_pids(self) -> list[int]:
            return [process.pid for process in world.processes]

        def shutdown(self) -> None:
            world.shutdowns += 1
            world.note("Engine.shutdown")
            # Not SGLang's doing, only tidiness of the fake: what a stop left on the loop is cancelled, and the
            # loop is closed.
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            if pending:
                self.loop.run_until_complete(asyncio.wait(pending))
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()

    hook_registry = module("sglang.srt.plugins.hook_registry", HookRegistry=FakeHookRegistry, HookType=_FakeHookType)
    plugins = module("sglang.srt.plugins", hook_registry=hook_registry, load_plugins=load_plugins)
    schedule_policy = module("sglang.srt.managers.schedule_policy", _IS_HIP=False)
    managers = module("sglang.srt.managers", scheduler=scheduler_module, schedule_policy=schedule_policy)
    runtime_context = module(
        "sglang.srt.runtime_context", get_parallel=lambda: world.current_process.parallel, get_platform=get_platform
    )
    server_args_module = module("sglang.srt.server_args", prepare_server_args=prepare_server_args)
    overrides = module("sglang.srt.arg_groups.overrides", resolving_view=_FakeResolvingView)
    arg_groups = module("sglang.srt.arg_groups", overrides=overrides)
    model_config = module("sglang.srt.configs.model_config", ModelConfig=FakeModelConfig)
    configs = module("sglang.srt.configs", model_config=model_config)
    engine_module = module("sglang.srt.entrypoints.engine", Engine=FakeSglangEngine)
    entrypoints = module("sglang.srt.entrypoints", engine=engine_module)
    srt = module(
        "sglang.srt",
        managers=managers,
        plugins=plugins,
        runtime_context=runtime_context,
        server_args=server_args_module,
        arg_groups=arg_groups,
        configs=configs,
        entrypoints=entrypoints,
    )
    top = module("sglang", srt=srt, __version__=world.version)
    made_modules = (
        top,
        srt,
        managers,
        scheduler_module,
        schedule_policy,
        runtime_context,
        plugins,
        hook_registry,
        server_args_module,
        arg_groups,
        overrides,
        configs,
        model_config,
        entrypoints,
        engine_module,
    )
    return {made.__name__: made for made in made_modules}


class FakeSglangServerArgs:
    """FAKE of SGLang's ServerArgs: the operator's input as fields, and what resolution declared beside them.

    As in SGLang, resolution writes no field. ``device`` and ``prefill_decode_interval`` stay None, as the
    operator left them, and their resolved values are only in ``declared``.
    """

    def __init__(self, world: _FakeWorld, argv: list[str]) -> None:
        self.world = world
        self.argv = argv
        self.resolved = False
        self.declared: dict = {}
        self.fields = {**world.resolved, "device": None, "prefill_decode_interval": None}

    def __getattr__(self, name: str) -> object:
        fields = self.__dict__.get("fields", {})
        if name not in fields:
            raise AttributeError(f"'FakeSglangServerArgs' object has no attribute {name!r}")
        return fields[name]

    def resolve_once(self) -> None:
        self.world.note("resolve_once")
        if not self.resolved:
            self.declared = {name: self.world.resolved[name] for name in ("device", "prefill_decode_interval")}
            self.resolved = True

    def resolved_dict(self) -> dict:
        self.world.note("resolved_dict")
        return {**self.fields, **self.declared, "custom_sigquit_handler": _fake_limits}


class _FakeResolvingView:
    """FAKE of SGLang's resolving view: a declared value first, the field of the record otherwise."""

    def __init__(self, server_args: FakeSglangServerArgs) -> None:
        object.__setattr__(self, "_server_args", server_args)

    def __getattr__(self, name: str) -> object:
        server_args = object.__getattribute__(self, "_server_args")
        if name in server_args.declared:
            return server_args.declared[name]
        return getattr(server_args, name)


class _FakeTokenizer:
    """FAKE of the tokenizer SGLang serves with: the four attributes sglbench_compat reads."""

    vocab_size = FAKE_VOCABULARY["size"]
    all_special_ids = (0, 1, 2)
    additional_stop_token_ids = frozenset({59_999})

    def get_added_vocab(self) -> dict:
        return {"<FAKE added a>": 50_000, "<FAKE added b>": 50_001, "<FAKE special>": 2}


# ---------------------------------------------------------------------------------------------------------------
# FAKE compat side: server arguments, resolved view, engine
# ---------------------------------------------------------------------------------------------------------------


class _FakeServerArgs:
    """FAKE of SGLang's ServerArgs record: what the fake parser was given."""

    def __init__(self, world: _FakeWorld, argv: list[str]) -> None:
        self.world = world
        self.argv = argv


class _FakeView:
    """FAKE of SGLang's view of the resolved arguments: a field is an attribute, any other name raises."""

    def __init__(self, fields: dict) -> None:
        object.__setattr__(self, "_fields", fields)

    def __getattr__(self, name: str) -> object:
        fields = object.__getattribute__(self, "_fields")
        if name not in fields:
            raise AttributeError(f"'FakeServerArgs' object has no attribute {name!r}")
        return fields[name]


@dataclasses.dataclass
class _FakeCall:
    """One call the launcher's engine adapter made through ``generate_batch``."""

    prompts: list[list[int]]
    rids: list[str]
    max_new_tokens: int
    timeout_seconds: object
    modes: list[str]

    @property
    def key(self) -> tuple[int, str, int]:
        parts = content.parse_request_id(self.rids[0])
        return parts["benchmark_id"], parts["stage"], parts["attempt"]


def _fake_limits(page_size: int) -> grid.Limits:
    """Small FAKE limits that grow with the page size and admit the 256-token warm-up prompt."""
    unit = grid.cached_unit(page_size)
    return grid.Limits(
        page_size=page_size,
        max_req_input_len=8 * unit + 300,
        max_total_num_tokens=grid.floor_pages(24 * unit + 800, page_size),
        max_running_requests=9,
        max_prefill_tokens=5 * unit,
        chunked_prefill_size=6 * unit,
        decode_graph_sizes=(1, 2, 4, 8),
        prefill_graph_sizes=(4, 8, 16, 32, 64),
    )


class _FakeWorld:
    """Everything FAKE of one launch, and a note of what the launcher did with it."""

    def __init__(self, page_size: int = 1, tp_size: int = 1) -> None:
        self.clock = _FakeClock()
        self.limits = _fake_limits(page_size)
        self.tp_size = tp_size
        self.version = FAKE_SGLANG_VERSION
        self.resolved = {**FAKE_RESOLVED_ARGUMENTS, "tp_size": tp_size}
        self.platform = {"is_cuda": True, "is_hip": False}
        self.experts: dict = {}
        self.vocabulary = {"size": FAKE_VOCABULARY["size"], "excluded": list(FAKE_VOCABULARY["excluded"])}
        # FAKE resolved dict with values JSON cannot hold, as SGLang's can have.
        self.resolved_dict = {
            "model_path": "fake/dense-model",
            "page_size": page_size,
            "tp_size": tp_size,
            "custom_sigquit_handler": _fake_limits,
            "ranks": (0, 1),
            "nested": {"ratio": float("inf"), 3: "FAKE value under a key that is not text"},
        }
        # The names of the compat functions in the order they were called, with "write" and "sleep" between.
        self.events: list[str] = []
        # Compat function name -> exception it raises, or callable it runs first.
        self.raises: dict[str, BaseException] = {}
        self.hooks: dict[str, object] = {}
        # (benchmark id, stage, attempt) of an engine call -> what the FAKE engine does in it.
        self.call_faults: dict[tuple, str] = {}
        self.scheduler_attributes: dict[str, object] = {}
        self.sglang_takes_sigterm = False
        self.sglang_sigterm_notes: list[int] = []

        self.sglang_argv: list[str] | None = None
        self.server_args: object = None
        # What reached the FAKE engine class of the FAKE sglang package, call by call.
        self.engine_requests: list[dict] = []
        self.tasks: list = []
        self.entry: object = None
        self.entry_calls: list[tuple] = []
        self.processes: list[_FakeProcess] = []
        self.current_process: _FakeProcess | None = None
        self.schedulers: list = []
        self.calls: list[_FakeCall] = []
        self.batches_run = 0
        self.cache = PageCacheModel(page_size)
        self.gpu_free_ticks = 0
        self.step_done_ticks = 0
        self.shutdowns = 0

    def note(self, name: str) -> None:
        """Note a call among the events, run the hook a test put on it, raise the error a test put on it."""
        self.events.append(name)
        hook = self.hooks.get(name)
        if hook is not None:
            hook()
        if name in self.raises:
            raise self.raises[name]

    def use(self, page_size: int, tp_size: int) -> None:
        """Make the FAKE engine one with this page size and this many scheduler processes."""
        self.limits = _fake_limits(page_size)
        self.cache = PageCacheModel(page_size)
        self.tp_size = tp_size
        self.resolved["tp_size"] = tp_size

    # -- FAKE clocks

    def scheduler_seconds(self) -> float:
        """The FAKE monotonic clock of the FAKE scheduler process that is running now."""
        return (self.clock.ticks + self.current_process.clock_offset_ticks) / TICKS_PER_SECOND

    def fake_sglang_sigterm_handler(self, signum: int, _frame: object) -> None:
        """FAKE of the handler SGLang puts on SIGTERM after its first request: it only notes the signal."""
        self.sglang_sigterm_notes.append(signum)


class _FakePipeWriter:
    def __init__(self) -> None:
        self.sent: list = []

    def send(self, message: object) -> None:
        self.sent.append(message)


class _FakeEngine:
    """Stands in for SGLang's engine and its scheduler processes. Everything it reports is FAKE."""

    def __init__(self, world: _FakeWorld, server_args: _FakeServerArgs, scheduler_entry) -> None:
        self.world = world
        world.entry = scheduler_entry
        writer = _FakePipeWriter()
        for tp_rank in range(world.tp_size):
            parallel = types.SimpleNamespace(
                tp_size=world.tp_size,
                pp_size=1,
                attn_dp_size=1,
                attn_dp_enabled=False,
                attn_tp_rank=tp_rank,
                attn_cp_rank=0,
                attn_dp_rank=0,
                pp_rank=0,
                pp_max_micro_batch_size=world.limits.max_running_requests,
            )
            # Every FAKE process has a clock origin of its own, as the monotonic clocks of real processes do.
            process = _FakeProcess(tp_rank, FAKE_FIRST_PID + tp_rank, (tp_rank + 1) * 1000 * TICKS_PER_SECOND, parallel)
            world.processes.append(process)
            world.current_process = process
            # SGLang's engine passes seven positional arguments when no DP controller is in the way:
            # server_args, port_args, gpu_id, tp_rank, pp_rank, dp_rank (None) and the pipe writer.
            scheduler_entry(server_args, FAKE_PORT_ARGS, tp_rank, tp_rank, 0, None, writer)
        world.schedulers = list(writer.sent)

    # -- the FAKE scheduler loop

    def _fake_batches(self, rids: list[str], prompts: list[list[int]], cached: list[int], output_tokens: int) -> list:
        """Extend batches filled up to the chunk budget, then one decode batch for every further output token.

        A request is charged its uncached tokens rounded up to whole pages. A request that does not fit what is
        left of the budget is cut: the part that fits goes into this batch and the rest into the next ones.
        """
        page, chunk = self.world.limits.page_size, self.world.limits.chunked_prefill_size
        batches: list[dict] = []
        current = {"mode": "EXTEND", "rids": [], "prefix_lens": [], "extend_lens": []}
        room = chunk
        for rid, prompt, hit in zip(rids, prompts, cached, strict=True):
            done = hit
            while done < len(prompt):
                left = len(prompt) - done
                part = left if grid.ceil_pages(left, page) <= room else grid.floor_pages(room, page)
                if part:
                    current["rids"].append(rid)
                    current["prefix_lens"].append(done)
                    current["extend_lens"].append(part)
                    done += part
                    room -= grid.ceil_pages(part, page)
                if done < len(prompt):
                    assert current["rids"], "FAKE limits with a chunk budget below one page"
                    batches.append(current)
                    current = {"mode": "EXTEND", "rids": [], "prefix_lens": [], "extend_lens": []}
                    room = chunk
        if current["rids"]:
            batches.append(current)
        batches.extend({"mode": "DECODE", "rids": list(rids)} for _step in range(output_tokens - 1))
        return batches

    def _run_batches(self, batches: list[dict], stage: str) -> None:
        """Run the FAKE batches of one call on every FAKE scheduler process, in the order of the overlap loop.

        In one pass SGLang's overlap loop runs the next batch first and processes the result of the batch
        before it second. The result is processed on a copy of the batch.
        """
        world = self.world
        waiting: list = [None] * len(world.schedulers)
        last_extend: dict = {}
        decode_step = 0
        for batch in batches:
            if batch["mode"] == "EXTEND":
                computed = sum(batch["extend_lens"])
                ticks = FAKE_EXTEND_BASE_TICKS + computed + (FAKE_WARM_SHOT_EXTRA_TICKS if stage == "warm" else 0)
                last_extend = {
                    "extend_num_tokens": computed,
                    "prefix_lens": list(batch["prefix_lens"]),
                    "extend_lens": list(batch["extend_lens"]),
                }
            else:
                tenths = FAKE_DECODE_STEP_TENTHS[decode_step % len(FAKE_DECODE_STEP_TENTHS)]
                ticks = (FAKE_DECODE_BASE_TICKS + len(batch["rids"])) * tenths
                decode_step += 1
            # FAKE GPU: it runs one batch after the other.
            world.step_done_ticks = world.gpu_free_ticks = max(world.gpu_free_ticks, world.clock.ticks) + ticks
            world.batches_run += 1
            queued = []
            for scheduler in world.schedulers:
                world.current_process = scheduler.fake_process
                # A decode batch still carries the extend fields of the extend batch its object came from.
                fake_batch = FakeBatch(
                    reqs=[FakeRequest(rid) for rid in batch["rids"]],
                    forward_mode=FakeForwardMode[batch["mode"]],
                    **last_extend,
                )
                # As in SGLang's loop: the batch is run first, and the copy is taken after that.
                result = scheduler.run_batch(fake_batch)
                queued.append((fake_batch.copy(), result))
            for index, scheduler in enumerate(world.schedulers):
                if waiting[index] is not None:
                    world.current_process = scheduler.fake_process
                    scheduler.process_batch_result(*waiting[index])
            waiting = queued
        for index, scheduler in enumerate(world.schedulers):
            if waiting[index] is not None:
                world.current_process = scheduler.fake_process
                scheduler.process_batch_result(*waiting[index])

    # -- the call

    def fake_generate(self, prompts, rids, max_new_tokens, timeout_seconds) -> list[dict]:
        """Answer one batch call: one dict per request, in input order, as ``generate_batch`` returns them."""
        world = self.world
        call = _FakeCall([list(prompt) for prompt in prompts], list(rids), max_new_tokens, timeout_seconds, [])
        world.calls.append(call)
        if world.sglang_takes_sigterm and len(world.calls) == 1:
            # FAKE of what SGLang does at its first request: it puts a handler of its own on SIGTERM.
            signal.signal(signal.SIGTERM, world.fake_sglang_sigterm_handler)
        fault = world.call_faults.get(call.key)
        if fault == "system_exit":
            raise SystemExit(0)
        if fault == "keyboard_interrupt":
            raise KeyboardInterrupt
        if fault == "sigterm":
            _deliver(signal.SIGTERM)
        if fault == "engine_error":
            raise compat.EngineCallError("FAKE: the engine call timed out")
        if fault == "evict":
            world.cache = PageCacheModel(world.limits.page_size)

        # A call is one batch of arrivals: every request is matched against the cache before any is stored.
        cached = [world.cache.cached(prompt) for prompt in call.prompts]
        batches = self._fake_batches(call.rids, call.prompts, cached, max_new_tokens)
        call.modes.extend(batch["mode"] for batch in batches)
        self._run_batches(batches, call.key[1])
        for prompt in call.prompts:
            # A finished request stores its prompt and the generated tokens that have KV: all but the last.
            # The FAKE output tokens lie below the token range of the prompts, so they equal no planned token.
            world.cache.finish_prefill([*prompt, *range(max_new_tokens - 1)])
        if fault == "deadline":
            world.clock.advance(FAKE_TIMEOUT_SECONDS * TICKS_PER_SECOND)
        return [
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


def _deliver(signum: int) -> None:
    """Send a real signal to the test process, but never one that would end it or interrupt pytest."""
    handler = signal.getsignal(signum)
    harmless = handler is signal.SIG_IGN or (callable(handler) and handler is not signal.default_int_handler)
    assert harmless, f"{signal.Signals(signum).name} has no handler of the launcher: {handler!r}"
    signal.raise_signal(signum)


def _install_fakes(monkeypatch, world: _FakeWorld, *, real_compat: bool = False, fake_pids: bool = False) -> None:
    """Put the FAKE clock and the FAKE ``sglang`` package in place and, unless told not to, FAKE compat functions.

    With ``real_compat`` the functions of ``sglbench_compat`` stay the real ones and reach the FAKE ``sglang``
    package; no module of the benchmark is replaced then.
    """
    step = world.note

    def sglang_version() -> str:
        step("sglang_version")
        return world.version

    def load_plugins() -> None:
        step("load_plugins")

    def parse_server_args(argv):
        step("parse_server_args")
        world.sglang_argv = list(argv)
        # FAKE parser: the one flag the launcher adds decides the field it stands for.
        world.resolved["incremental_streaming_output"] = STREAMING_FLAG in argv
        world.server_args = _FakeServerArgs(world, list(argv))
        return world.server_args

    def resolve(server_args):
        assert server_args is world.server_args
        step("resolve")
        return _FakeView(world.resolved)

    def resolved_dict(server_args):
        assert server_args is world.server_args
        step("resolved_dict")
        return world.resolved_dict

    def platform_facts() -> dict:
        step("platform_facts")
        return dict(world.platform)

    def model_expert_keys(server_args) -> dict:
        assert server_args is world.server_args
        step("model_expert_keys")
        return dict(world.experts)

    def build_engine(server_args, scheduler_entry):
        assert server_args is world.server_args
        step("build_engine")
        return _FakeEngine(world, server_args, scheduler_entry)

    def generate_batch(engine, prompts, rids, max_new_tokens, timeout_seconds):
        assert isinstance(engine, _FakeEngine)
        return engine.fake_generate(prompts, rids, max_new_tokens, timeout_seconds)

    def vocabulary(engine) -> dict:
        assert isinstance(engine, _FakeEngine)
        step("vocabulary")
        return {key: (list(value) if isinstance(value, list) else value) for key, value in world.vocabulary.items()}

    def shutdown(engine) -> None:
        assert isinstance(engine, _FakeEngine)
        world.shutdowns += 1
        step("shutdown")

    fakes = (
        sglang_version,
        load_plugins,
        parse_server_args,
        resolve,
        resolved_dict,
        platform_facts,
        model_expert_keys,
        build_engine,
        generate_batch,
        vocabulary,
        shutdown,
    )
    assert [fake.__name__ for fake in fakes] == list(REAL_COMPAT)
    if not real_compat:
        for fake in fakes:
            # A fake takes the arguments of the function it stands for, by the same names.
            real = REAL_COMPAT[fake.__name__]
            assert list(inspect.signature(fake).parameters) == list(inspect.signature(real).parameters)
            monkeypatch.setattr(compat, fake.__name__, fake)

    def sleep(seconds: float) -> None:
        world.events.append("sleep")
        world.clock.sleep(seconds)

    def monotonic() -> float:
        if "clock" not in world.events:
            # The first reading of the launcher's clock is noted among the events.
            world.events.append("clock")
        return world.clock.monotonic()

    # The launcher and the driver read the fake clock in the place of time.monotonic and time.sleep. The
    # recorder reads the clock of the FAKE scheduler process that is running.
    monkeypatch.setattr(launcher, "time", types.SimpleNamespace(monotonic=monotonic, sleep=sleep))
    monkeypatch.setattr(recorder, "time", types.SimpleNamespace(monotonic=world.scheduler_seconds))
    if fake_pids or world.tp_size > 1:
        # Two recorder objects in one process would share a process id and so a log file. Each FAKE scheduler
        # process gets a FAKE id; the recorder sees it through a stand-in for the ``os`` module.
        fake_os = types.SimpleNamespace(environ=os.environ, path=os.path, getpid=lambda: world.current_process.pid)
        monkeypatch.setattr(recorder, "os", fake_os)
    for name, made in _fake_sglang_modules(world).items():
        monkeypatch.setitem(sys.modules, name, made)

    if real_compat:
        return

    # Not a fake: the real writer of the result files, with a note of when it ran.
    def noted_write(output_path, rank_payloads):
        world.events.append("write")
        return REAL_WRITE_ARTIFACTS(output_path, rank_payloads)

    monkeypatch.setattr(artifact, "write_artifacts", noted_write)


# ---------------------------------------------------------------------------------------------------------------
# Launching
# ---------------------------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _process_state(monkeypatch):
    """Every test starts without the variables the launcher reads and gets its signal handlers back."""
    for name in list(os.environ):
        if name.startswith(("DYN_", "SGLANG_", "FPM_", "AISIM_SGLBENCH_")):
            monkeypatch.delenv(name)
    # The launcher sets this variable for the life of its process. Setting and deleting it here makes pytest
    # take the launcher's value out again after the test.
    monkeypatch.setenv(RECORDER_ENV, "")
    monkeypatch.delenv(RECORDER_ENV)
    saved = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    yield
    for signum, handler in saved.items():
        if handler is not None:
            signal.signal(signum, handler)


@dataclasses.dataclass
class _Launch:
    """One call of ``launcher.main`` against the fakes, with what the test planned itself."""

    code: int
    world: _FakeWorld
    mode: str
    cell: object
    root: Path
    pod: Path
    caps: grid.SampleCaps
    manifest: dict | None = None

    @property
    def rank_path(self) -> Path:
        return self.pod / "benchmark.json"

    @property
    def merged_path(self) -> Path:
        return self.pod / "benchmark_merged.json"

    def artifact(self) -> dict:
        return json.loads(self.rank_path.read_text(encoding="utf-8"))

    def planned(self) -> grid.Grid:
        """The grid, planned here from the FAKE limits and the caps the flags carry."""
        return grid.build_grid(self.mode, self.world.limits, self.caps, manifest=self.manifest)

    def recorder_dirs(self) -> list[Path]:
        return sorted((self.pod / "sglbench-recorder").iterdir())

    def calls_of(self, benchmark_id: int) -> list[_FakeCall]:
        return [call for call in self.world.calls if call.key[0] == benchmark_id]


def _new_pod(tmp_path: Path, mode: str, sglang_version: str = FAKE_SGLANG_VERSION):
    cell = _cell(phase="prefill" if mode == "agg" else mode, dp=1, strategy="single")
    return cell, _pod(tmp_path, cell, sglang_version=sglang_version)


def _launch(
    tmp_path: Path,
    monkeypatch,
    mode: str = "prefill",
    *,
    page_size: int = 1,
    tp_size: int = 1,
    flags: tuple | list = (),
    env: dict | None = None,
    prepare=None,
    pod: tuple | None = None,
    caps: grid.SampleCaps = SMALL_CAPS,
    cap_flags: list | None = None,
    manifest: dict | None = None,
    real_compat: bool = False,
) -> _Launch:
    """Lay out a pod, put the fakes in place and call ``launcher.main`` with the flags of a small run.

    With ``real_compat`` the FAKE engine gets its page size and its TP size from the flags alone, through the
    FAKE parser of the FAKE ``sglang`` package.
    """
    world = _FakeWorld() if real_compat else _FakeWorld(page_size, tp_size)
    cell, pod_dir = _new_pod(tmp_path, mode) if pod is None else pod
    if prepare is not None:
        prepare(world, pod_dir)
    _install_fakes(monkeypatch, world, real_compat=real_compat, fake_pids=tp_size > 1)
    monkeypatch.setenv("DYN_FPM_WORKER_ID", FAKE_WORKER_ID)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    argv = [
        "--benchmark-mode",
        mode,
        "--benchmark-output-path",
        str(pod_dir / "benchmark.json"),
        "--benchmark-timeout",
        str(FAKE_TIMEOUT_SECONDS),
        *(SMALL_CAP_FLAGS if cap_flags is None else cap_flags),
        *flags,
        *_fake_sglang_argv(page_size, tp_size),
    ]
    code = launcher.main(argv)
    return _Launch(code, world, mode, cell, tmp_path, pod_dir, caps, manifest)


def _fake_sglang_argv(page_size: int = 1, tp_size: int = 1) -> list[str]:
    """Flags of SGLang's own parser, which the launcher has to hand on."""
    return ["--model-path", "fake/dense-model", "--page-size", str(page_size), "--tp-size", str(tp_size)]


def _planned(mode: str, page_size: int = 1) -> grid.Grid:
    """The grid of a small run, planned here from the FAKE limits."""
    return grid.build_grid(mode, _fake_limits(page_size), SMALL_CAPS)


def _fake_expected_sample(point: grid.Point) -> float:
    """The FAKE sample a point must carry, worked out here from the FAKE timings."""
    if point.phase == grid.PREFILL:
        return (FAKE_EXTEND_BASE_TICKS + point.batch_size * point.new_tokens) / TICKS_PER_SECOND
    return (FAKE_DECODE_BASE_TICKS + point.batch_size) * 10 / TICKS_PER_SECOND


def _fake_expected_steps(point: grid.Point) -> list[float]:
    return [
        (FAKE_DECODE_BASE_TICKS + point.batch_size) * tenths / TICKS_PER_SECOND for tenths in FAKE_DECODE_STEP_TENTHS
    ]


def _expected_regime(point: grid.Point) -> str:
    if point.phase == grid.DECODE:
        return "real_kv"
    return "real_prefix" if point.cached_tokens else "not_applicable"


def _assert_accepted(launch: _Launch) -> dict:
    """A complete run: exit code 0, every planned point in the file, the validator and the gate accept."""
    built = launch.planned()
    assert launch.code == 0
    on_disk = launch.artifact()
    assert (on_disk["status"], on_disk["valid"], on_disk["usable"], on_disk["error"]) == ("complete", True, True, None)
    assert on_disk["coverage"] == {
        "expected_points": len(built.points),
        "completed_points": len(built.points),
        "skipped_points": 0,
    }
    assert on_disk["grid_digest"] == built.digest
    # Every row carries the FAKE sample of the leader, the FAKE worker id and, for decode, the six FAKE steps.
    for point, row in zip(built.points, on_disk["results"], strict=True):
        (fpm,) = row["fpms"]
        assert (row["point"], row["kv_seed_regime"]) == (point.as_dict(), _expected_regime(point))
        assert (fpm["dp_rank"], fpm["wall_time"], fpm["worker_id"]) == (0, _fake_expected_sample(point), FAKE_WORKER_ID)
        if point.phase == grid.DECODE:
            assert fpm[artifact.DECODE_STEPS_KEY] == _fake_expected_steps(point)
    assert launch.merged_path.exists()
    if launch.mode != grid.AGG:
        collection = _validate(launch.cell, launch.root)
        assert [measurement.point for measurement in collection.points] == built.point_dicts()
        assert [measurement.rank_wall_times for measurement in collection.points] == [
            ((0, _fake_expected_sample(point)),) for point in built.points
        ]
        assert (collection.runtime_run_id, collection.runtime_grid_digest) == (on_disk["run_id"], built.digest)
    gate = _gate([str(launch.rank_path)], launch.mode)
    assert gate.returncode == GATE_ACCEPT, gate.stderr
    return on_disk


def _assert_failed(launch: _Launch, code: int, reason: str) -> dict:
    """A failed run: the exit code, a failed rank artifact with the reason, no merged file, both checkers refuse."""
    assert launch.code == code
    on_disk = launch.artifact()
    assert (on_disk["status"], on_disk["valid"], on_disk["usable"]) == ("failed", False, False)
    assert (on_disk["error"], on_disk["errors"]) == (reason, [reason])
    assert not launch.merged_path.exists()
    with pytest.raises(ValueError, match="invalid native terminal result envelope"):
        _validate(launch.cell, launch.root)
    gate = _gate([str(launch.rank_path)], launch.cell.workload_kind)
    assert gate.returncode == GATE_INVALID
    assert "status='failed' valid=False" in gate.stderr
    return on_disk


def _assert_failed_before_the_grid(launch: _Launch, code: int, reason: str) -> dict:
    on_disk = _assert_failed(launch, code, reason)
    assert launch.world.calls == []
    assert on_disk["grid_digest"] == grid.grid_digest([])
    assert on_disk["coverage"] == {"expected_points": 0, "completed_points": 0, "skipped_points": 0}
    return on_disk


def _stderr_lines(capsys) -> list[str]:
    """The launcher's own lines. All of them go to stderr; it prints nothing on stdout."""
    captured = capsys.readouterr()
    assert captured.out == ""
    return [line for line in captured.err.splitlines() if line.startswith("[sglbench]")]


def _tree(directory: Path) -> dict[str, bytes]:
    """Every file under ``directory`` with its content."""
    return {
        str(path.relative_to(directory)): path.read_bytes() for path in sorted(directory.rglob("*")) if path.is_file()
    }


def _options(argv: list[str], env: dict | None = None):
    return launcher._read_options(["--benchmark-mode", "prefill", *argv], env or {})


# ---------------------------------------------------------------------------------------------------------------
# End to end: no module of the benchmark is replaced. The launcher, the real compat functions, the real
# recorder, the real driver and the real artifact module run against the FAKE sglang package alone.
# ---------------------------------------------------------------------------------------------------------------


def _assert_no_module_of_the_benchmark_is_replaced() -> None:
    assert all(getattr(compat, name) is real for name, real in REAL_COMPAT.items())
    assert artifact.write_artifacts is REAL_WRITE_ARTIFACTS
    assert launcher.compat is compat and launcher.driver is driver and launcher.recorder is recorder
    assert launcher.artifact is artifact and launcher.grid is grid and driver.content is content


@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_end_to_end_run_passes_the_validator_and_the_gate(tmp_path, monkeypatch, capsys, mode, page_size):
    dump = tmp_path / "raw" / "pod-0" / "resolved-config-node0.json"
    launch = _launch(
        tmp_path, monkeypatch, mode, page_size=page_size, real_compat=True, flags=["--dump-config-to", str(dump)]
    )

    _assert_no_module_of_the_benchmark_is_replaced()
    built = launch.planned()
    assert len(built.points) >= 8 and built.missing_phases == ()
    on_disk = _assert_accepted(launch)
    world = launch.world
    # The page size of the FAKE engine came from the flag the launcher handed on to the FAKE parser.
    assert world.limits.page_size == page_size
    assert world.sglang_argv == [*_fake_sglang_argv(page_size), STREAMING_FLAG]

    # The entry of every scheduler process was the recorder's, called as SGLang calls it.
    assert world.entry is recorder.recorder_entry
    assert world.entry_calls == [(0, 0, 0, None)]
    assert type(world.schedulers[0]).__name__ == "SglbenchRecorderScheduler"
    assert type(world.schedulers[0]).__mro__[1] is FakeBaseScheduler

    # The log of the run: one directory, one file, a header and one line for every FAKE batch, no fault.
    (recorder_dir,) = launch.recorder_dirs()
    assert recorder_dir.name == on_disk["run_id"]
    (log,) = sorted(recorder_dir.iterdir())
    assert log.name == f"sched-{os.getpid()}.jsonl"
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["type"] == "header" and lines[0]["class"] == "SglbenchRecorderScheduler"
    assert (lines[0]["sglang_version"], lines[0]["leader"], lines[0]["enable_overlap"]) == ("0.5.23", True, True)
    assert lines[0]["limits"]["page_size"] == page_size
    assert [line["type"] for line in lines[1:]] == ["batch"] * world.batches_run
    assert [line["seq"] for line in lines[1:]] == list(range(1, world.batches_run + 1))
    assert all(line["copy_done"] is True for line in lines[1:])
    # The lines carry the FAKE clock of the FAKE scheduler process, not the launcher's.
    assert all(8000.0 <= line["launch_ts"] < line["out_ts"] < 9000.0 for line in lines[1:])
    assert world.clock.monotonic() < 8000.0
    # A decode batch carries no extend field although the FAKE batch object held stale ones.
    decode_lines = [line for line in lines[1:] if line["mode"] == "DECODE"]
    assert all(line["extend_num_tokens"] is line["prefix_lens"] is line["extend_lens"] is None for line in decode_lines)
    assert bool(decode_lines)

    # What reached the FAKE engine class: one streamed batch call per stage, with three sampling settings and
    # no other request field.
    assert len(world.engine_requests) == len(world.calls)
    for request, call in zip(world.engine_requests, world.calls, strict=True):
        assert request == {
            "prompt": None,
            "sampling_params": {"max_new_tokens": call.max_new_tokens, "temperature": 0.0, "ignore_eos": True},
            "stream": True,
            "more": {},
        }
    # The points ran in grid order, after the self-test and the warm-up.
    order = [call.key[0] for call in world.calls if call.key[0]]
    assert order == sorted(order) and set(order) == {point.benchmark_id for point in built.points}
    assert [call.key for call in world.calls[:7]] == [(0, "free", number) for number in range(7)]
    assert [call.max_new_tokens for call in world.calls[:7]] == [1, 1, 1, 1, 1, 7, 5]
    assert all(
        FAKE_TOKEN_RANGE[0] <= token < FAKE_TOKEN_RANGE[1] for call in world.calls for p in call.prompts for token in p
    )
    # Request ids carry the first eight characters of the run id.
    assert world.calls[0].rids[0] == f"sglb-{on_disk['run_id'][:8]}-p0-free0-r0-s0-e"

    assert on_disk["config"] == {
        "mode": mode,
        "warmup_iterations": 5,
        "timeout": FAKE_TIMEOUT_SECONDS,
        "prefill_max_new_token_samples": 3,
        "prefill_max_kv_read_token_samples": 3,
        "decode_max_kv_read_token_samples": 3,
        "decode_max_batch_size_samples": 3,
        "prefix_max_batch_size_samples": 2,
        "max_batch_size": None,
        "points_source": "generated",
        "content_seed": "0",
        "output_path": str(launch.rank_path),
    }
    producer = on_disk["producer"]
    assert {key: producer[key] for key in LAUNCHER_PRODUCER_KEYS} == {
        "engine_call_timeout_seconds": 900.0,
        "streaming": {"stream": True, "incremental_streaming_output": True, "stream_interval": 1},
        "points_file": None,
        "sglang_version": "0.5.23",
        "enable_nvidia_megakernel": False,
        "tp_size": 1,
    }
    assert set(producer) == {*LAUNCHER_PRODUCER_KEYS, *DRIVER_PRODUCER_KEYS, "name", "artifact_format"}
    assert (producer["name"], producer["scheduler_processes"], producer["recorder_format"]) == ("aisim_sglbench", 1, 1)
    assert (producer["limits"], producer["page_size"]) == (world.limits.as_dict(), page_size)
    assert (producer["enable_overlap"], producer["restaged_points"]) == (True, [])
    assert producer["content"]["token_range"] == FAKE_TOKEN_RANGE
    assert producer["self_test"] == ["batch_shape", "aligned_prefix", "unaligned_prompt", "seven_token_run"]

    # What the launcher asked of the FAKE sglang package, in order, and then the final wait.
    assert world.events == [
        "clock",
        "load_plugins",
        "prepare_server_args",
        "resolve_once",
        "get_platform",
        "from_server_args",
        "resolved_dict",
        "Engine",
        "Engine.shutdown",
        "sleep",
    ]
    assert _stderr_lines(capsys)[-1] == f"[sglbench] complete: wrote {launch.rank_path}, {launch.merged_path}"
    # The dump holds what SGLang resolved, not what the operator left unset, and lies beside the result files.
    dumped = json.loads(dump.read_text(encoding="utf-8"))
    assert (dumped["producer"], dumped["sglang_version"]) == ("aisim_sglbench", "0.5.23")
    server_args = dumped["config"]["server_args"]
    assert (server_args["device"], server_args["prefill_decode_interval"]) == ("cuda", 0)
    assert server_args["custom_sigquit_handler"] == repr(_fake_limits)
    assert dumped["config"]["benchmark"]["run_id"] == on_disk["run_id"]
    # No file of the recorder tree has a name the collector or the gate sweeps for.
    for path in (launch.pod / "sglbench-recorder").rglob("*"):
        assert not any(fnmatch.fnmatch(path.name, pattern) for pattern in FORBIDDEN_FILE_PATTERNS)


def test_end_to_end_agg_run_is_accepted_by_the_gate_in_agg_mode(tmp_path, monkeypatch):
    launch = _launch(tmp_path, monkeypatch, "agg", page_size=64, real_compat=True)

    _assert_no_module_of_the_benchmark_is_replaced()
    on_disk = _assert_accepted(launch)
    built = launch.planned()
    assert {point.phase for point in built.points} == {"prefill", "decode"}
    assert on_disk["config"]["mode"] == "agg" and "kvwarm" in on_disk
    # AISim launches one phase per cell: a gate started for one phase refuses the agg file.
    assert _gate([str(launch.rank_path)], "prefill").returncode == GATE_INVALID


def test_end_to_end_with_two_scheduler_processes(tmp_path, monkeypatch):
    launch = _launch(tmp_path, monkeypatch, "decode", tp_size=2, real_compat=True)

    on_disk = _assert_accepted(launch)
    world = launch.world
    # ``--tp-size 2`` reached the FAKE parser; the resolved TP size is the number of headers the driver waits for.
    assert (on_disk["producer"]["tp_size"], on_disk["producer"]["scheduler_processes"]) == (2, 2)
    assert world.entry_calls == [(0, 0, 0, None), (1, 1, 0, None)]
    (recorder_dir,) = launch.recorder_dirs()
    logs = {path.name: path.read_text(encoding="utf-8").splitlines() for path in sorted(recorder_dir.iterdir())}
    assert list(logs) == ["sched-4100.jsonl", "sched-4101.jsonl"]
    headers = [json.loads(lines[0]) for lines in logs.values()]
    assert [(header["pid"], header["leader"], header["ranks"]["attn_tp_rank"]) for header in headers] == [
        (4100, True, 0),
        (4101, False, 1),
    ]
    # Both processes ran every batch, each on a FAKE clock of its own.
    assert [len(lines) for lines in logs.values()] == [1 + world.batches_run] * 2
    follower = [json.loads(line) for line in logs["sched-4101.jsonl"][1:]]
    assert all(9000.0 <= line["launch_ts"] < line["out_ts"] < 10000.0 for line in follower)


def test_end_to_end_with_a_points_manifest(tmp_path, monkeypatch):
    document = {
        "schema_version": 3,
        "prefill": [
            {"batch_size": 2, "total_prefill_tokens": 64, "total_kv_read_tokens": 64},
            {"batch_size": 1, "total_prefill_tokens": 5, "total_kv_read_tokens": 0},
            {"batch_size": 4, "total_prefill_tokens": 16, "total_kv_read_tokens": 384, "rows": [[4, 96]] * 4},
        ],
        "decode": [{"batch_size": 3, "total_kv_read_tokens": 60}],
    }
    raw = json.dumps(document, indent=1).encode()
    points_file = tmp_path / "staged" / "benchmark-points.json"
    points_file.parent.mkdir()
    points_file.write_bytes(raw)
    manifest = grid.parse_points_manifest(raw)

    launch = _launch(
        tmp_path,
        monkeypatch,
        "prefill",
        page_size=16,
        flags=["--benchmark-points-file", str(points_file)],
        manifest=manifest,
        real_compat=True,
    )

    _assert_no_module_of_the_benchmark_is_replaced()
    on_disk = _assert_accepted(launch)
    built = launch.planned()
    # The file's points in the file's order, each stamped explicit; the caps on the command line are ignored.
    assert [(point.batch_size, point.new_tokens, point.cached_tokens) for point in built.points] == [
        (2, 32, 32),
        (1, 5, 0),
        (4, 4, 96),
    ]
    assert all("explicit" in row["point"]["sample_reasons"] for row in on_disk["results"])
    assert on_disk["config"]["points_source"] == "manifest"
    # The name and the digest, not the directory the file was staged in.
    assert on_disk["producer"]["points_file"] == {"name": points_file.name, "sha256": hashlib.sha256(raw).hexdigest()}
    assert str(points_file.parent) not in json.dumps(on_disk)
    # The staged file was read and not changed, and it was not copied into the results.
    assert points_file.read_bytes() == raw
    assert not list(launch.pod.rglob("benchmark-points*"))


def test_end_to_end_a_signal_inside_the_engines_event_loop_is_recorded_as_a_stop(tmp_path, monkeypatch):
    victim = _planned("decode").points[1]

    def prepare(world, _pod_dir):
        # The signal arrives while the engine's event loop runs a step of the engine call.
        world.call_faults[(victim.benchmark_id, "run", 0)] = "sigterm"

    launch = _launch(tmp_path, monkeypatch, "decode", prepare=prepare, real_compat=True)

    _assert_no_module_of_the_benchmark_is_replaced()
    on_disk = _assert_failed(launch, 65, "stopped: signal SIGTERM")
    assert on_disk["coverage"]["completed_points"] == victim.benchmark_id - 1
    assert launch.world.events[-2:] == ["Engine.shutdown", "sleep"]


def test_end_to_end_sglangs_own_exit_after_sigterm_is_recorded_as_a_stop(tmp_path, monkeypatch):
    victim = _planned("prefill").points[2]

    def prepare(world, _pod_dir):
        # FAKE of SGLang once it owns SIGTERM: a task on the engine's loop calls sys.exit(0) during a call.
        world.call_faults[(victim.benchmark_id, "warm", 0)] = "sglang_exit"

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare, real_compat=True)

    on_disk = _assert_failed(launch, 65, "stopped: SystemExit(0)")
    assert on_disk["coverage"]["completed_points"] == victim.benchmark_id - 1
    assert launch.world.events[-2:] == ["Engine.shutdown", "sleep"]
    # The warm shot never reached the FAKE scheduler: the exit came first.
    assert not [call for call in launch.calls_of(victim.benchmark_id) if call.key[1] == "warm"]
    assert launch.world.shutdowns == 1


def test_end_to_end_an_aborted_request_fails_its_point(tmp_path, monkeypatch):
    victim = _planned("prefill").points[1]

    def prepare(world, _pod_dir):
        world.call_faults[(victim.benchmark_id, "meas", 0)] = "abort"

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare, real_compat=True)

    rid = launch.calls_of(victim.benchmark_id)[-1].rids[0]
    reason = (
        f"benchmark_id={victim.benchmark_id}: benchmark point failed: engine_call_failed: EngineCallError: "
        f"request {rid} was aborted: 'FAKE abort' (status_code=400)"
    )
    _assert_failed(launch, 65, reason)
    assert launch.world.events[-2:] == ["Engine.shutdown", "sleep"]


@pytest.mark.parametrize(
    ("flags", "reason"),
    [
        (["--no-such-flag"], "rejected: SGLang's argument parser exited with status 2; its message is on stderr"),
        (["--stream-interval", "4"], "rejected: stream_interval must be 1 (got 4)"),
    ],
)
def test_end_to_end_a_flag_sglang_or_the_benchmark_refuses(tmp_path, monkeypatch, capfd, flags, reason):
    launch = _launch(tmp_path, monkeypatch, "prefill", flags=flags, real_compat=True)

    _assert_failed_before_the_grid(launch, 64, reason)
    assert "Engine" not in launch.world.events
    if flags == ["--no-such-flag"]:
        assert "unrecognized arguments: --no-such-flag" in capfd.readouterr().err


def test_a_point_that_misses_the_cache_once_is_staged_again(tmp_path, monkeypatch):
    # A decode point whose run must find cached tokens.
    victim = next(point for point in _planned("decode").points[1:] if point.cached_tokens > 8)

    def prepare(world, _pod_dir):
        world.call_faults[(victim.benchmark_id, "run", 0)] = "evict"

    launch = _launch(tmp_path, monkeypatch, "decode", prepare=prepare)

    on_disk = _assert_accepted(launch)
    assert on_disk["producer"]["restaged_points"] == [victim.benchmark_id]
    assert [call.key[1:] for call in launch.calls_of(victim.benchmark_id)] == [
        ("seed", 0),
        ("run", 0),
        ("seed", 1),
        ("run", 1),
    ]


# ---------------------------------------------------------------------------------------------------------------
# Exit codes, and a failed artifact after each kind of failure
# ---------------------------------------------------------------------------------------------------------------


def test_exit_code_65_and_a_failed_artifact_when_a_point_fails(tmp_path, monkeypatch, capsys):
    victim = _planned("prefill").points[3]

    def prepare(world, _pod_dir):
        world.call_faults[(victim.benchmark_id, "meas", 0)] = "engine_error"

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    reason = (
        f"benchmark_id={victim.benchmark_id}: benchmark point failed: engine_call_failed: EngineCallError: "
        "FAKE: the engine call timed out"
    )
    on_disk = _assert_failed(launch, 65, reason)
    assert on_disk["coverage"]["completed_points"] == victim.benchmark_id - 1
    assert on_disk["skipped_points"][0]["point"] == victim.as_dict()
    # The run stopped at the point: nothing after it was sent, and the engine was still shut down.
    assert max(call.key[0] for call in launch.world.calls) == victim.benchmark_id
    assert launch.world.events[-3:] == ["shutdown", "write", "sleep"]
    lines = _stderr_lines(capsys)
    assert lines[-2:] == [f"[sglbench] FAILED: {reason}", f"[sglbench] wrote the failed artifact {launch.rank_path}"]
    assert sum(line.startswith("[sglbench] FAILED: ") for line in lines) == 1


def test_exit_code_66_and_a_partial_artifact_at_the_soft_deadline(tmp_path, monkeypatch, capsys):
    planned = _planned("decode")
    last_done = planned.points[1]

    def prepare(world, _pod_dir):
        world.call_faults[(last_done.benchmark_id, "run", 0)] = "deadline"

    launch = _launch(tmp_path, monkeypatch, "decode", prepare=prepare)

    assert launch.code == 66
    on_disk = launch.artifact()
    assert (on_disk["status"], on_disk["valid"], on_disk["usable"]) == ("partial", False, True)
    assert (on_disk["stop_reason"], on_disk["error"]) == ("timeout", None)
    assert on_disk["coverage"] == {
        "expected_points": len(planned.points),
        "completed_points": last_done.benchmark_id,
        "skipped_points": 0,
    }
    # A partial run has a merged file; both checkers refuse the run.
    assert launch.merged_path.exists()
    with pytest.raises(ValueError, match="invalid native terminal result envelope"):
        _validate(launch.cell, launch.root)
    gate = _gate([str(launch.rank_path)], "decode")
    assert gate.returncode == GATE_INVALID and "status='partial' valid=False" in gate.stderr
    assert _stderr_lines(capsys)[-1] == (
        f"[sglbench] PARTIAL: the soft deadline ended the run (timeout); wrote {launch.rank_path}, {launch.merged_path}"
    )
    assert launch.world.events[-3:] == ["shutdown", "write", "sleep"]


def test_the_soft_deadline_counts_from_the_start_of_the_launcher(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        # FAKE: resolving the arguments and building the engine take longer than the whole timeout.
        world.hooks["resolve"] = lambda: world.clock.advance(4000 * TICKS_PER_SECOND)
        world.hooks["build_engine"] = lambda: world.clock.advance(7000 * TICKS_PER_SECOND)

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    # The self-test and the warm-up still ran, and no point did.
    assert launch.code == 66
    on_disk = launch.artifact()
    assert (on_disk["status"], on_disk["stop_reason"], on_disk["usable"]) == ("partial", "timeout", False)
    assert on_disk["coverage"]["completed_points"] == 0
    assert {call.key[0] for call in launch.world.calls} == {0}


def test_the_clock_is_read_before_the_flags_and_before_sglang(tmp_path, monkeypatch):
    real_read_options = launcher._read_options

    def prepare(world, _pod_dir):
        def slow_read_options(argv, environ):
            # FAKE: reading the flags takes longer than the whole timeout.
            world.clock.advance(100 * TICKS_PER_SECOND)
            return real_read_options(argv, environ)

        monkeypatch.setattr(launcher, "_read_options", slow_read_options)

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare, flags=["--benchmark-timeout", "50"])

    # The soft deadline had passed before the first point, so the clock was read before the flags.
    assert launch.code == 66 and launch.artifact()["coverage"]["completed_points"] == 0
    assert launch.world.events[:2] == ["clock", "sglang_version"]


def test_the_flags_and_variables_reach_the_run(tmp_path, monkeypatch):
    caps = dataclasses.replace(SMALL_CAPS, benchmark_max_batch_size=2)

    launch = _launch(
        tmp_path,
        monkeypatch,
        "decode",
        flags=["--benchmark-max-batch-size", "2", "--benchmark-warmup-iterations", "0"],
        env={"DYN_BENCH_CONTENT_SEED": "007"},
        caps=caps,
    )

    on_disk = _assert_accepted(launch)
    assert max(point.batch_size for point in launch.planned().points) == 2
    assert max(point.batch_size for point in _planned("decode").points) > 2
    assert (on_disk["config"]["max_batch_size"], on_disk["config"]["warmup_iterations"]) == (2, 0)
    # Every engine call went through the adapter with the per-call time limit.
    assert {call.timeout_seconds for call in launch.world.calls} == {900.0}
    assert type(launch.world.calls[0].timeout_seconds) is float
    # The warm-up is skipped at 0: six calls of the self-test come before the first point.
    assert [call.key[0] for call in launch.world.calls[:7]] == [0] * 6 + [1]
    # The seed is text and reaches the content plan: the first prompt is the plan's first free prompt.
    assert (on_disk["config"]["content_seed"], on_disk["producer"]["content"]["seed"]) == ("007", "007")
    plan = content.ContentPlan("007", *FAKE_TOKEN_RANGE)
    assert launch.world.calls[0].prompts[0] == plan.free_prompt(0, 16)
    assert launch.world.calls[0].prompts[0] != content.ContentPlan("7", *FAKE_TOKEN_RANGE).free_prompt(0, 16)


def test_an_old_output_that_cannot_be_deleted_is_a_failure_and_not_a_crash(tmp_path, monkeypatch, capsys):
    cell, pod_dir = _new_pod(tmp_path, "prefill")
    (pod_dir / "benchmark.json").mkdir()

    launch = _launch(tmp_path, monkeypatch, "prefill", pod=(cell, pod_dir))

    assert launch.code == 70
    assert launch.world.events == ["clock", "write", "sleep"]
    lines = _stderr_lines(capsys)
    assert lines[0].startswith("[sglbench] FAILED: internal_error: ")
    assert lines[1].startswith("[sglbench] the failed artifact could not be written: ")
    assert (pod_dir / "benchmark.json").is_dir()


def test_exit_code_64_and_a_failed_artifact_for_a_bad_flag_before_anything_is_started(tmp_path, monkeypatch, capsys):
    cell, pod_dir = _new_pod(tmp_path, "prefill")
    stale = {"benchmark.json": b'{"stale": "rank"}', "benchmark_merged.json": b'{"stale": "merged"}'}
    for name, data in stale.items():
        (pod_dir / name).write_bytes(data)

    launch = _launch(tmp_path, monkeypatch, "prefill", flags=["--benchmark-timeout", "0"], pod=(cell, pod_dir))

    reason = "rejected: --benchmark-timeout must be above 0, got 0"
    on_disk = _assert_failed_before_the_grid(launch, 64, reason)
    # SGLang was not touched, no recorder directory was made and the variable was not set.
    assert launch.world.events == ["clock", "write", "sleep"]
    assert not (pod_dir / "sglbench-recorder").exists()
    assert RECORDER_ENV not in os.environ
    # The config block holds the values as they were given.
    assert on_disk["config"]["timeout"] == 0
    assert _stderr_lines(capsys) == [
        f"[sglbench] FAILED: {reason}",
        f"[sglbench] wrote the failed artifact {launch.rank_path}",
    ]


def test_exit_code_70_and_a_failed_artifact_for_an_unexpected_error(tmp_path, monkeypatch, capsys):
    def prepare(world, _pod_dir):
        world.raises["platform_facts"] = TypeError("FAKE: get_platform().is_cuda is None, not a bool")

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    reason = "internal_error: TypeError: FAKE: get_platform().is_cuda is None, not a bool"
    _assert_failed_before_the_grid(launch, 70, reason)
    assert "build_engine" not in launch.world.events and "shutdown" not in launch.world.events
    assert _stderr_lines(capsys)[-2] == f"[sglbench] FAILED: {reason}"


def test_exit_code_64_without_an_artifact_when_the_output_path_names_no_file(tmp_path, monkeypatch, capsys):
    world = _FakeWorld()
    _install_fakes(monkeypatch, world)

    code = launcher.main(["--benchmark-mode", "prefill", "--benchmark-output-path", f"{tmp_path}/results/"])

    assert code == 64
    # The clock was read, and nothing else happened.
    assert world.events == ["clock"] and not (tmp_path / "results").exists()
    assert _stderr_lines(capsys) == [
        f"[sglbench] FAILED: rejected: --benchmark-output-path '{tmp_path}/results/': output_path must name a "
        f"file, got '{tmp_path}/results/'",
        "[sglbench] no result file was written: the output path names no file",
    ]


def test_exit_codes_follow_the_first_word_of_the_reason():
    assert launcher._exit_code("complete", None) == 0
    assert launcher._exit_code("partial", "timeout") == 66
    assert launcher._exit_code("failed", "rejected: x") == 64
    assert launcher._exit_code("failed", "internal_error: x") == 70
    for reason in (
        "stopped: signal SIGTERM",
        "recorder_header_invalid: x",
        "self_test_failed: x",
        "token_range_too_small: x",
        "recorder_fault: x",
        "engine_call_failed: x",
        "benchmark_id=3: benchmark point failed: rejected: x",
        "benchmark_id=3: benchmark point failed: measured_batch_size_mismatch: x",
        "no feasible benchmark points for phase(s): decode",
        "rejected",
        "rejectedly: x",
    ):
        assert launcher._exit_code("failed", reason) == 65
    assert (launcher.EXIT_COMPLETE, launcher.EXIT_REJECTED, launcher.EXIT_FAILED) == (0, 64, 65)
    assert (launcher.EXIT_PARTIAL, launcher.EXIT_INTERNAL) == (66, 70)
    # None of them is a code the gate or argparse uses itself.
    assert not {64, 65, 66, 70} & {1, 2, 124, 130, 143}


# ---------------------------------------------------------------------------------------------------------------
# Order at the end: shutdown, write, wait
# ---------------------------------------------------------------------------------------------------------------


def test_shutdown_comes_before_the_write_and_the_wait_after_it(tmp_path, monkeypatch, capsys):
    seen = {}

    def prepare(world, pod_dir):
        def at_shutdown():
            seen["files_at_shutdown"] = sorted(path.name for path in pod_dir.glob("benchmark*"))

        def at_sleep(seconds):
            seen.setdefault("sleeps", []).append(seconds)
            seen["files_at_wait"] = sorted(path.name for path in pod_dir.glob("benchmark*"))
            seen["tree_at_wait"] = _tree(pod_dir)
            seen["handlers"] = [signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)]
            seen.setdefault("stderr_at_wait", capsys.readouterr().err)

        world.hooks["shutdown"] = at_shutdown
        world.clock.on_sleep.append(at_sleep)

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    assert launch.code == 0
    assert launch.world.events[-3:] == ["shutdown", "write", "sleep"]
    assert launch.world.shutdowns == 1
    # The outcome is printed before the wait begins.
    assert "[sglbench] complete: wrote" in seen["stderr_at_wait"]
    # No result file existed when the engine was shut down; both existed when the wait began.
    assert seen["files_at_shutdown"] == []
    assert seen["files_at_wait"] == ["benchmark.json", "benchmark_merged.json"]
    # The wait is one sleep of five seconds on the fake clock, and nothing is written during or after it.
    assert seen["sleeps"] == [5.0]
    assert _tree(launch.pod) == seen["tree_at_wait"]
    # SIGTERM and SIGINT were ignored from the start of the write on.
    assert seen["handlers"] == [signal.SIG_IGN, signal.SIG_IGN]


def test_a_sleep_that_returns_early_is_taken_up_again_until_five_seconds_have_passed(monkeypatch):
    clock = _FakeClock()
    clock.sleep_fraction = 0.3
    monkeypatch.setattr(launcher, "time", types.SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep))
    before = clock.monotonic()

    launcher._stay_alive(launcher.FINAL_WAIT_SECONDS)

    assert launcher.FINAL_WAIT_SECONDS == 5.0
    assert clock.monotonic() - before >= 5.0
    assert len(clock.sleeps) > 3 and clock.sleeps[0] == 5.0
    # Each further sleep asks for what is left.
    assert all(later < earlier for earlier, later in zip(clock.sleeps, clock.sleeps[1:], strict=False))


def test_a_shutdown_error_does_not_spoil_a_complete_run(tmp_path, monkeypatch, capsys):
    def prepare(world, _pod_dir):
        world.raises["shutdown"] = RuntimeError("FAKE: 2 child processes\nwere not reaped in 60 s")

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    on_disk = _assert_accepted(launch)
    text = "RuntimeError: FAKE: 2 child processes were not reaped in 60 s"
    assert on_disk["producer"]["engine_shutdown_error"] == text
    assert launch.world.events[-3:] == ["shutdown", "write", "sleep"]
    lines = _stderr_lines(capsys)
    assert f"[sglbench] the engine shutdown failed: {text}" in lines
    assert not any(line.startswith("[sglbench] FAILED") for line in lines)


def test_a_failed_write_is_printed_and_ends_with_exit_code_70(tmp_path, monkeypatch, capsys):
    def prepare(world, pod_dir):
        # The results directory is taken away while the engine shuts down.
        world.hooks["shutdown"] = lambda: shutil.rmtree(pod_dir)

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    assert launch.code == 70 and not launch.pod.exists()
    (line,) = [line for line in _stderr_lines(capsys) if line.startswith("[sglbench] FAILED: ")]
    assert line.startswith("[sglbench] FAILED: internal_error: the result files could not be written: ")
    assert "FileNotFoundError" in line
    assert launch.world.events[-3:] == ["shutdown", "write", "sleep"]


def test_a_failed_write_does_not_hide_the_first_reason(tmp_path, monkeypatch, capsys):
    def prepare(world, pod_dir):
        world.experts = {"num_experts": 8}
        world.hooks["model_expert_keys"] = lambda: shutil.rmtree(pod_dir)

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    assert launch.code == 64
    lines = _stderr_lines(capsys)
    assert lines[-2] == "[sglbench] FAILED: rejected: MoE models are not supported yet (num_experts=8)"
    assert lines[-1].startswith("[sglbench] the failed artifact could not be written: FileNotFoundError")


# ---------------------------------------------------------------------------------------------------------------
# Stops
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("signum", "name"), [(signal.SIGTERM, "SIGTERM"), (signal.SIGINT, "SIGINT")])
@pytest.mark.parametrize("where", ["resolve", "build_engine"])
def test_a_signal_before_the_first_request_is_recorded_as_a_stop(tmp_path, monkeypatch, where, signum, name):
    def prepare(world, _pod_dir):
        world.hooks[where] = lambda: _deliver(signum)

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    _assert_failed_before_the_grid(launch, 65, f"stopped: signal {name}")
    # The engine object never came back from the build, so there is nothing to shut down.
    assert launch.world.events[-3:] == [where, "write", "sleep"]
    assert launch.world.shutdowns == 0


def test_a_signal_during_an_engine_call_is_recorded_as_a_stop_and_the_engine_is_shut_down(tmp_path, monkeypatch):
    victim = _planned("prefill").points[2]

    def prepare(world, _pod_dir):
        world.call_faults[(victim.benchmark_id, "warm", 0)] = "sigterm"

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    on_disk = _assert_failed(launch, 65, "stopped: signal SIGTERM")
    # The rows of the points before the stop are kept, and the stop names no point.
    assert on_disk["coverage"]["completed_points"] == victim.benchmark_id - 1
    assert on_disk["skipped_points"] == []
    assert launch.world.events[-3:] == ["shutdown", "write", "sleep"]


@pytest.mark.parametrize(
    ("fault", "what"),
    [("system_exit", "SystemExit(0)"), ("keyboard_interrupt", "KeyboardInterrupt")],
)
def test_a_system_exit_from_an_engine_call_is_recorded_as_a_stop(tmp_path, monkeypatch, capsys, fault, what):
    victim = _planned("decode").points[1]

    def prepare(world, _pod_dir):
        # FAKE of SGLang after its first request: it owns SIGTERM and ends an engine call with sys.exit(0).
        world.sglang_takes_sigterm = True
        world.call_faults[(victim.benchmark_id, "run", 0)] = fault

    launch = _launch(tmp_path, monkeypatch, "decode", prepare=prepare)

    on_disk = _assert_failed(launch, 65, f"stopped: {what}")
    assert on_disk["coverage"]["completed_points"] == victim.benchmark_id - 1
    assert launch.world.events[-3:] == ["shutdown", "write", "sleep"]
    assert launch.world.shutdowns == 1
    assert f"[sglbench] FAILED: stopped: {what}" in _stderr_lines(capsys)


@pytest.mark.parametrize(("signum", "name"), [(signal.SIGTERM, "SIGTERM"), (signal.SIGINT, "SIGINT")])
def test_a_signal_during_the_shutdown_fails_the_run_because_nothing_is_written_yet(tmp_path, monkeypatch, signum, name):
    def prepare(world, _pod_dir):
        world.hooks["shutdown"] = lambda: _deliver(signum)

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    # Every point completed, and the run is still failed: the signal came before the write.
    on_disk = _assert_failed(launch, 65, f"stopped: signal {name} before the result files were written")
    assert on_disk["coverage"]["completed_points"] == on_disk["coverage"]["expected_points"] > 0
    assert launch.world.events[-3:] == ["shutdown", "write", "sleep"]


def test_no_signal_changes_anything_once_the_final_write_has_started(tmp_path, monkeypatch):
    seen = {}

    def prepare(world, pod_dir):
        def at_wait(_seconds):
            seen["tree"] = _tree(pod_dir)
            seen["handlers"] = [signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)]
            # The gate sends SIGTERM to the process group once it has accepted the files.
            _deliver(signal.SIGTERM)
            _deliver(signal.SIGINT)
            _deliver(signal.SIGTERM)

        world.clock.on_sleep.append(at_wait)

    launch = _launch(tmp_path, monkeypatch, "decode", prepare=prepare)

    assert seen["handlers"] == [signal.SIG_IGN, signal.SIG_IGN]
    assert launch.code == 0
    assert _tree(launch.pod) == seen["tree"]
    assert launch.world.events.count("write") == 1 and launch.world.events[-1] == "sleep"
    _assert_accepted(launch)


def test_a_handler_sglang_has_taken_over_is_taken_back_and_ignored_at_the_end(tmp_path, monkeypatch):
    seen = {}

    def prepare(world, pod_dir):
        world.sglang_takes_sigterm = True

        def at_wait(_seconds):
            seen["handlers"] = [signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)]
            seen["tree"] = _tree(pod_dir)
            _deliver(signal.SIGTERM)
            _deliver(signal.SIGINT)

        world.clock.on_sleep.append(at_wait)

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    # The launcher took SIGTERM back after the first engine call, so both signals are ignored at the end and
    # SGLang's FAKE handler never saw one.
    assert seen["handlers"] == [signal.SIG_IGN, signal.SIG_IGN]
    assert launch.world.sglang_sigterm_notes == []
    assert launch.code == 0 and _tree(launch.pod) == seen["tree"]


def test_the_stop_exception_leaves_an_event_loop_and_is_raised_once():
    # asyncio lets only SystemExit and KeyboardInterrupt leave a callback or a task step.
    assert issubclass(launcher._StopRequested, SystemExit)
    assert not issubclass(launcher._StopRequested, Exception)
    stops = launcher._SignalStops()
    stops.install()
    assert signal.getsignal(signal.SIGTERM) == signal.getsignal(signal.SIGINT)

    with pytest.raises(launcher._StopRequested) as raised:
        _deliver(signal.SIGINT)
    assert raised.value.what == "signal SIGINT"
    # The handler disarmed itself: a second signal is noted and does not interrupt the handling of the first.
    _deliver(signal.SIGTERM)
    assert stops.late == ["SIGTERM"]

    stops.ignore()
    assert [signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)] == [signal.SIG_IGN, signal.SIG_IGN]
    _deliver(signal.SIGTERM)
    assert stops.late == ["SIGTERM"]


def test_a_stop_that_arrives_inside_a_callback_of_an_event_loop_leaves_the_loop():
    # After its first request SGLang runs its engine calls on an event loop. A signal can arrive while the loop
    # runs a callback; an exception that is not a SystemExit would be logged by the loop and the call would go on.
    stops = launcher._SignalStops()
    stops.install()
    loop = asyncio.new_event_loop()

    async def engine_call() -> str:
        loop.call_soon(_deliver, signal.SIGTERM)
        for _turn in range(3):
            await asyncio.sleep(0)
        return "the loop went on"

    try:
        with pytest.raises(launcher._StopRequested) as raised:
            loop.run_until_complete(engine_call())
        assert raised.value.what == "signal SIGTERM"
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


def test_a_disarmed_handler_notes_a_signal_without_raising():
    stops = launcher._SignalStops()
    stops.install()
    stops.disarm()

    _deliver(signal.SIGTERM)
    _deliver(signal.SIGINT)

    assert stops.late == ["SIGTERM", "SIGINT"]


# ---------------------------------------------------------------------------------------------------------------
# Own flags and variables
# ---------------------------------------------------------------------------------------------------------------


def test_flag_defaults():
    options = _options(FAKE_SGLANG_ARGV)

    assert options.mode == "prefill"
    assert options.cap_values == {
        "prefill_max_new_token_samples": 64,
        "prefill_max_kv_read_token_samples": 16,
        "decode_max_kv_read_token_samples": 128,
        "decode_max_batch_size_samples": 128,
        "prefix_max_batch_size_samples": 3,
        "benchmark_max_batch_size": None,
    }
    assert options.caps == grid.SampleCaps()
    assert (options.warmup_iterations, options.timeout) == (5, 900)
    assert options.output_path == "/tmp/benchmark_results.json"
    assert (options.points_file, options.manifest, options.points_sha256) == (None, None, None)
    assert options.dump_config_to is None
    assert (options.content_seed, options.worker_id) == ("0", "")
    assert (options.rejection, options.output_problem) == (None, None)
    assert launcher._config_block(options) == {
        "warmup_iterations": 5,
        "timeout": 900,
        "prefill_max_new_token_samples": 64,
        "prefill_max_kv_read_token_samples": 16,
        "decode_max_kv_read_token_samples": 128,
        "decode_max_batch_size_samples": 128,
        "prefix_max_batch_size_samples": 3,
        "max_batch_size": None,
        "points_source": "generated",
        "content_seed": "0",
        "output_path": "/tmp/benchmark_results.json",
    }


DYN_DEFAULTS = {
    "DYN_BENCHMARK_MODE": "decode",
    "DYN_PREFILL_MAX_NEW_TOKEN_SAMPLES": "7",
    "DYN_PREFILL_MAX_KV_READ_TOKEN_SAMPLES": "8",
    "DYN_DECODE_MAX_KV_READ_TOKEN_SAMPLES": "9",
    "DYN_DECODE_MAX_BATCH_SIZE_SAMPLES": "10",
    "DYN_PREFIX_MAX_BATCH_SIZE_SAMPLES": "11",
    "DYN_BENCHMARK_MAX_BATCH_SIZE": "12",
    "DYN_BENCHMARK_WARMUP_ITERATIONS": "0",
    "DYN_BENCHMARK_OUTPUT_PATH": "/results/from-variable.json",
    "DYN_BENCHMARK_TIMEOUT": "10800",
    "DYN_BENCH_CONTENT_SEED": "007",
    "DYN_FPM_WORKER_ID": "cell-node0",
}


def test_every_flag_takes_its_default_from_its_dyn_variable():
    options = launcher._read_options([], DYN_DEFAULTS)

    assert options.mode == "decode"
    assert options.cap_values == {
        "prefill_max_new_token_samples": 7,
        "prefill_max_kv_read_token_samples": 8,
        "decode_max_kv_read_token_samples": 9,
        "decode_max_batch_size_samples": 10,
        "prefix_max_batch_size_samples": 11,
        "benchmark_max_batch_size": 12,
    }
    assert options.caps == grid.SampleCaps(7, 8, 9, 10, 11, 12)
    assert (options.warmup_iterations, options.timeout) == (0, 10800)
    assert options.output_path == "/results/from-variable.json"
    # The seed stays text, leading zeros included.
    assert (options.content_seed, options.worker_id) == ("007", "cell-node0")
    assert options.rejection is None


def test_a_flag_wins_over_its_variable_and_the_fpm_output_path_wins_over_both(tmp_path):
    points_file = tmp_path / "points.json"
    points_file.write_text('{"schema_version": 1, "prefill": [], "decode": []}', encoding="utf-8")
    argv = [
        "--benchmark-mode=agg",
        "--prefill-max-new-token-samples=20",
        "--prefill-max-kv-read-token-samples=21",
        "--decode-max-kv-read-token-samples=22",
        "--decode-max-batch-size-samples=23",
        "--prefix-max-batch-size-samples=24",
        "--benchmark-max-batch-size=25",
        "--benchmark-warmup-iterations=3",
        "--benchmark-output-path=/results/from-flag.json",
        "--benchmark-timeout=60",
    ]
    variables = {**DYN_DEFAULTS, "DYN_BENCHMARK_POINTS_FILE": str(points_file)}

    options = launcher._read_options(argv, variables)

    assert options.mode == "agg"
    assert list(options.cap_values.values()) == [20, 21, 22, 23, 24, 25]
    assert (options.warmup_iterations, options.timeout) == (3, 60)
    assert options.output_path == "/results/from-flag.json"
    # The points file has no flag here, so its variable gives it.
    assert options.points_file == str(points_file) and options.manifest["schema_version"] == 1
    assert launcher._config_block(options)["points_source"] == "manifest"

    override = launcher._read_options(argv, {**variables, "DYN_FPM_BENCHMARK_OUTPUT_PATH": "/results/benchmark.json"})
    assert override.output_path == "/results/benchmark.json"
    only_variable = launcher._read_options(
        ["--benchmark-mode", "prefill"], {"DYN_FPM_BENCHMARK_OUTPUT_PATH": "/results/b.json"}
    )
    assert only_variable.output_path == "/results/b.json"


@pytest.mark.parametrize(
    ("argv", "variables", "message"),
    [
        ([], {}, "--benchmark-mode is required (or DYN_BENCHMARK_MODE): prefill, decode, agg"),
        (FAKE_SGLANG_ARGV, {"FPM_BENCHMARK_MODE": "prefill"}, "--benchmark-mode is required"),
        (["--benchmark-mode", "both"], {}, "argument --benchmark-mode: invalid choice: 'both'"),
        (["--benchmark-mode", "prefill", "--benchmark-timeout", "soon"], {}, "invalid int value: 'soon'"),
        (["--benchmark-mode", "prefill", "--prefill-max-new-token-samples", "2.5"], {}, "invalid int value: '2.5'"),
        # An abbreviation of the mode flag is not the mode flag.
        (["--benchmark-mod", "prefill"], {}, "--benchmark-mode is required"),
    ],
)
def test_argparse_errors_exit_2(capsys, argv, variables, message):
    with pytest.raises(SystemExit) as raised:
        launcher._read_options(argv, variables)

    assert raised.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("value", ["both", ""])
def test_an_unknown_mode_in_the_variable_is_refused_with_64_and_no_artifact(capsys, value):
    with pytest.raises(SystemExit) as raised:
        launcher._read_options([], {"DYN_BENCHMARK_MODE": value})

    assert raised.value.code == 64
    assert _stderr_lines(capsys) == [
        f"[sglbench] FAILED: rejected: DYN_BENCHMARK_MODE={value!r}: choose from prefill, decode, agg",
        "[sglbench] no result file was written: an artifact needs the mode",
    ]


def test_abbreviations_are_refused_and_other_flags_pass_through_unchanged():
    argv = [
        "--model-path",
        "fake/model",
        "--benchmark-time",
        "5",
        "--benchmark-warmup-iterations",
        "2",
        "--tp",
        "2",
        "--prefill-max",
        "9",
        "--enable-metrics",
        "--dump-config",
        "x",
        "--benchmark-output",
        "y",
    ]

    options = _options(argv)

    # ``--benchmark-time``, ``--prefill-max``, ``--dump-config`` and ``--benchmark-output`` are not taken for the
    # launcher's flags of which they are a prefix. They go to SGLang's parser as they were typed, in order.
    assert (options.timeout, options.dump_config_to, options.output_path) == (900, None, "/tmp/benchmark_results.json")
    assert options.cap_values["prefill_max_new_token_samples"] == 64
    assert options.warmup_iterations == 2
    assert options.sglang_argv == [
        "--model-path",
        "fake/model",
        "--benchmark-time",
        "5",
        "--tp",
        "2",
        "--prefill-max",
        "9",
        "--enable-metrics",
        "--dump-config",
        "x",
        "--benchmark-output",
        "y",
        STREAMING_FLAG,
    ]


def test_the_streaming_flag_is_added_once():
    assert _options([]).sglang_argv == [STREAMING_FLAG]
    assert _options(["--model-path", "m"]).sglang_argv == ["--model-path", "m", STREAMING_FLAG]
    given = ["--model-path", "m", STREAMING_FLAG, "--tp-size", "2"]
    assert _options(given).sglang_argv == given


@pytest.mark.parametrize(("argv", "variables", "refusal"), OWN_REFUSALS)
def test_refused_flags_and_variables(argv, variables, refusal):
    options = _options(argv, variables)

    assert options.rejection == refusal
    assert options.output_problem is None


@pytest.mark.parametrize(("argv", "variables"), OWN_ACCEPTED)
def test_accepted_flags_and_variables(argv, variables):
    options = _options(argv, variables)

    assert options.rejection is None
    assert (options.mode, options.output_path) == ("prefill", "/tmp/benchmark_results.json")
    assert options.dump_config_to is None
    assert options.sglang_argv == [STREAMING_FLAG]


def test_every_refused_flag_and_variable_of_the_module_has_a_test():
    flags = {argv[0] for argv, _variables, _refusal in OWN_REFUSALS if argv}
    variables = {name for _argv, names, _refusal in OWN_REFUSALS for name in names}
    assert {flag for _dest, flag, _variable, _why in launcher._REFUSED_SWITCHES} <= flags
    assert {variable for _dest, _flag, variable, _why in launcher._REFUSED_SWITCHES} <= variables
    assert {flag for _dest, flag, _variable, _use in launcher._LEGACY_FLAGS} == {flag for flag, _v, _u in _LEGACY}
    assert {variable for _dest, _flag, variable, _use in launcher._LEGACY_FLAGS} <= variables
    assert set(launcher._REFUSED_VARIABLES) <= variables
    assert {variable for variable, _accepted in launcher._REAL_STATE_SWITCHES} <= variables
    assert launcher.ENV_COLLECT_IMBALANCED_OVERRIDE in variables
    # The variable of each accepted flag is the one Dynamo's vLLM worker reads for it.
    assert launcher._FLAG_VARIABLES == {
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


def test_the_first_bad_value_is_the_reason():
    options = _options(["--benchmark-hybrid-live-state", "--benchmark-timeout", "0"], {"DYN_BENCH_POOL_TAG": "x"})

    assert options.rejection == f"--benchmark-hybrid-live-state {_HYBRID}"


def test_a_seed_that_is_not_utf_8_is_refused():
    options = _options([], {"DYN_BENCH_CONTENT_SEED": "seed-\udcff"})

    assert options.rejection == "DYN_BENCH_CONTENT_SEED='seed-\\udcff' must be a non-empty UTF-8 string"


def test_the_points_file_is_read_and_parsed_before_anything_else_happens(tmp_path):
    raw = b'{"schema_version": 2, "prefill": [], "decode": [{"batch_size": 2, "total_kv_read_tokens": 16}]}'
    points_file = tmp_path / "benchmark-points.json"
    points_file.write_bytes(raw)
    points_file.chmod(0o444)

    options = _options(["--benchmark-points-file", str(points_file)])

    assert options.rejection is None
    assert options.manifest == {"schema_version": 2, "prefill": [], "decode": [(2, 16)]}
    assert options.points_sha256 == hashlib.sha256(raw).hexdigest()
    assert points_file.read_bytes() == raw

    missing = _options(["--benchmark-points-file", str(tmp_path / "none.json")])
    assert missing.rejection.startswith(f"--benchmark-points-file '{tmp_path}/none.json': [Errno 2]")
    points_file.chmod(0o644)
    points_file.write_bytes(b'{"schema_version": 9, "prefill": [], "decode": []}')
    refused = _options(["--benchmark-points-file", str(points_file)])
    assert refused.rejection == (
        f"--benchmark-points-file '{points_file}': benchmark points manifest schema_version must be 1, 2 or 3, got 9"
    )
    assert refused.manifest is None


@pytest.mark.parametrize("name", ["benchmark.json", "benchmark_merged.json", "./benchmark.json"])
def test_a_dump_path_that_is_a_result_file_is_refused(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)

    options = _options(["--benchmark-output-path", f"{tmp_path}/benchmark.json", "--dump-config-to", name])

    assert options.rejection == f"--dump-config-to {name!r} is one of the result files of the run"


@pytest.mark.parametrize("path", ["", "/results/", "results/"])
def test_an_output_path_that_names_no_file(path):
    options = _options(["--benchmark-output-path", path])

    assert options.output_problem.startswith(f"--benchmark-output-path {path!r}: output_path must ")
    # The variable wins over the flag, and the problem names the one the path came from.
    by_variable = _options(["--benchmark-output-path", "/results/b.json"], {"DYN_FPM_BENCHMARK_OUTPUT_PATH": path})
    assert by_variable.output_problem.startswith(f"DYN_FPM_BENCHMARK_OUTPUT_PATH {path!r}: output_path must ")


# ---------------------------------------------------------------------------------------------------------------
# Start-up checks on SGLang's arguments and variables
# ---------------------------------------------------------------------------------------------------------------


def test_the_default_resolved_arguments_pass():
    launcher._check_resolved(_FakeView(dict(FAKE_RESOLVED_ARGUMENTS)))


@pytest.mark.parametrize(("field", "value", "refusal"), [*STAGE_S_REFUSALS, *STAGE_S_MORE_REFUSALS])
def test_every_refused_setting_of_the_resolved_arguments(field, value, refusal):
    view = _FakeView({**FAKE_RESOLVED_ARGUMENTS, field: value})

    with pytest.raises(launcher._Failed) as raised:
        launcher._check_resolved(view)

    assert raised.value.reason == f"rejected: {refusal}"


@pytest.mark.parametrize(("field", "value"), STAGE_S_ACCEPTED)
def test_other_values_of_the_resolved_arguments_that_pass(field, value):
    launcher._check_resolved(_FakeView({**FAKE_RESOLVED_ARGUMENTS, field: value}))


def test_every_field_the_launcher_tests_has_a_refused_value_here():
    tested = [name for name, _accepts, _text in launcher._RESOLVED_CHECKS]
    assert tested == [field for field, _value, _refusal in STAGE_S_REFUSALS]
    assert len(set(tested)) == len(tested) == 42
    # The fields the launcher reads besides: the platform, the three run settings and the recorded field.
    assert set(FAKE_RESOLVED_ARGUMENTS) - set(tested) == {
        "device",
        "tp_size",
        "prefill_max_requests",
        "max_queued_requests",
        "enable_nvidia_megakernel",
    }


@pytest.mark.parametrize("field", [name for name, _value, _refusal in STAGE_S_REFUSALS])
def test_a_field_sglang_does_not_have_is_an_error_and_not_a_pass(field):
    fields = dict(FAKE_RESOLVED_ARGUMENTS)
    del fields[field]

    with pytest.raises(AttributeError, match=f"SGLang's resolved arguments have no field '{field}'"):
        launcher._check_resolved(_FakeView(fields))


@pytest.mark.parametrize(
    ("field", "value", "refusal"), [STAGE_S_REFUSALS[2], STAGE_S_REFUSALS[21], STAGE_S_REFUSALS[39]]
)
def test_a_refused_setting_ends_the_launch_before_the_engine_is_built(
    tmp_path, monkeypatch, capsys, field, value, refusal
):
    def prepare(world, _pod_dir):
        if field == "incremental_streaming_output":
            # FAKE: an SGLang that does not take the flag the launcher adds.
            world.hooks["resolve"] = lambda: world.resolved.update({field: value})
        else:
            world.resolved[field] = value

    launch = _launch(
        tmp_path,
        monkeypatch,
        "prefill",
        prepare=prepare,
        flags=["--dump-config-to", str(tmp_path / "resolved-config-node0.json")],
    )

    _assert_failed_before_the_grid(launch, 64, f"rejected: {refusal}")
    assert launch.world.events == [
        "clock",
        "sglang_version",
        "load_plugins",
        "parse_server_args",
        "resolve",
        "write",
        "sleep",
    ]
    # No dump is written for a setup that was refused.
    assert not (tmp_path / "resolved-config-node0.json").exists()
    assert f"[sglbench] FAILED: rejected: {refusal}" in _stderr_lines(capsys)


def test_a_missing_field_ends_the_launch_as_an_internal_error(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        del world.resolved["dcp_size"]

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    reason = (
        "internal_error: AttributeError: SGLang's resolved arguments have no field 'dcp_size'; the start-up check "
        "that reads it cannot run"
    )
    _assert_failed_before_the_grid(launch, 70, reason)
    assert "build_engine" not in launch.world.events


@pytest.mark.parametrize("missing", ["tp_size", "prefill_max_requests", "max_queued_requests", "device"])
def test_the_fields_read_after_the_checks_must_exist_too(tmp_path, monkeypatch, missing):
    def prepare(world, _pod_dir):
        del world.resolved[missing]

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    assert launch.code == 70
    assert launch.artifact()["error"].startswith(
        f"internal_error: AttributeError: SGLang's resolved arguments have no field '{missing}'"
    )
    assert "build_engine" not in launch.world.events


@pytest.mark.parametrize("megakernel", [True, False, "absent"])
def test_a_field_that_only_one_release_has_is_recorded_with_a_default(tmp_path, monkeypatch, megakernel):
    def prepare(world, _pod_dir):
        if megakernel == "absent":
            # v0.5.22 has no such field.
            del world.resolved["enable_nvidia_megakernel"]
            world.version = "0.5.22"
        else:
            world.resolved["enable_nvidia_megakernel"] = megakernel

    # The provenance file of the pod names the release that is installed there.
    pod = _new_pod(tmp_path, "prefill", "0.5.22" if megakernel == "absent" else FAKE_SGLANG_VERSION)
    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare, pod=pod)

    on_disk = _assert_accepted(launch)
    assert on_disk["producer"]["enable_nvidia_megakernel"] == (None if megakernel == "absent" else megakernel)
    assert on_disk["producer"]["sglang_version"] == ("0.5.22" if megakernel == "absent" else "0.5.23")
    assert launcher._ONE_RELEASE_FIELDS == (("enable_nvidia_megakernel", None),)


@pytest.mark.parametrize("tp_size", [0, None, True, "2", 1.0])
def test_a_tp_size_that_is_not_a_positive_int_is_an_internal_error(tmp_path, monkeypatch, tp_size):
    def prepare(world, _pod_dir):
        world.resolved["tp_size"] = tp_size

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    reason = f"internal_error: TypeError: SGLang's resolved tp_size is {tp_size!r}, not a positive int"
    _assert_failed_before_the_grid(launch, 70, reason)


@pytest.mark.parametrize(
    ("setting", "value", "detail"),
    [("prefill_max_requests", 1, "prefill_max_requests=1"), ("max_queued_requests", 1, "max_queued_requests=1")],
)
def test_the_request_caps_of_the_resolved_arguments_reach_the_driver(tmp_path, monkeypatch, setting, value, detail):
    def prepare(world, _pod_dir):
        world.resolved[setting] = value

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    largest = max(point.batch_size for point in launch.planned().points)
    on_disk = _assert_failed(launch, 64, f"rejected: {detail} is below the largest batch {largest}")
    # The grid existed when the cap was compared, and no request was sent.
    assert on_disk["coverage"]["expected_points"] == len(launch.planned().points)
    assert launch.world.calls == [] and launch.world.events[-3:] == ["shutdown", "write", "sleep"]


def test_request_caps_at_or_above_the_largest_batch_are_accepted(tmp_path, monkeypatch):
    largest = max(point.batch_size for point in _planned("prefill").points)

    def prepare(world, _pod_dir):
        world.resolved.update({"prefill_max_requests": largest, "max_queued_requests": largest})

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    assert largest > 1
    _assert_accepted(launch)


@pytest.mark.parametrize(("name", "value", "refusal"), SGLANG_ENV_REFUSALS)
def test_refused_variables_of_sglang(name, value, refusal):
    with pytest.raises(launcher._Failed) as raised:
        launcher._check_sglang_environment({name: value})

    assert raised.value.reason == f"rejected: {refusal}"


@pytest.mark.parametrize(("name", "value"), SGLANG_ENV_ACCEPTED)
def test_accepted_variables_of_sglang(name, value):
    launcher._check_sglang_environment({name: value})
    launcher._check_sglang_environment({})


def test_a_variable_of_sglang_is_refused_before_the_arguments_are_resolved(tmp_path, monkeypatch):
    launch = _launch(tmp_path, monkeypatch, "decode", env={"SGLANG_MAX_NEW_TOKENS_LIMIT": "4"})

    _assert_failed_before_the_grid(
        launch, 64, "rejected: SGLANG_MAX_NEW_TOKENS_LIMIT=4 is below the 7 output tokens of a decode run"
    )
    assert launch.world.events == ["clock", "sglang_version", "load_plugins", "parse_server_args", "write", "sleep"]


@pytest.mark.parametrize("version", ["0.5.21", "0.5.24", "0.5.23.post1", "0.0.0.dev0", "0.5.230"])
def test_an_untested_release_is_refused_first(tmp_path, monkeypatch, version):
    def prepare(world, _pod_dir):
        world.version = version
        # A field the release lacks does not hide the reason.
        del world.resolved["dcp_size"]

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    on_disk = _assert_failed_before_the_grid(
        launch, 64, f"rejected: SGLang {version} is not a tested version (tested: 0.5.22, 0.5.23)"
    )
    assert launch.world.events == ["clock", "sglang_version", "write", "sleep"]
    assert on_disk["producer"]["sglang_version"] == version


@pytest.mark.parametrize(
    ("platform", "device", "shown"),
    [
        ({"is_cuda": False, "is_hip": False}, "cpu", "device=cpu, is_cuda=false, is_hip=false"),
        ({"is_cuda": True, "is_hip": True}, "cuda", "device=cuda, is_cuda=true, is_hip=true"),
        ({"is_cuda": False, "is_hip": True}, "cuda", "device=cuda, is_cuda=false, is_hip=true"),
        ({"is_cuda": True, "is_hip": False}, "xpu", "device=xpu, is_cuda=true, is_hip=false"),
        ({"is_cuda": True, "is_hip": False}, None, "device=null, is_cuda=true, is_hip=false"),
        ({"is_cuda": 1, "is_hip": False}, "cuda", "device=cuda, is_cuda=1, is_hip=false"),
        ({"is_cuda": True, "is_hip": None}, "cuda", "device=cuda, is_cuda=true, is_hip=null"),
    ],
)
def test_a_platform_that_is_not_nvidia_cuda_is_refused(tmp_path, monkeypatch, platform, device, shown):
    def prepare(world, _pod_dir):
        world.platform = platform
        world.resolved["device"] = device

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    reason = f"rejected: only NVIDIA CUDA is supported ({shown}); SGLang has AMD-only admission rules"
    _assert_failed_before_the_grid(launch, 64, reason)
    assert launch.world.events[-3:] == ["platform_facts", "write", "sleep"]


@pytest.mark.parametrize(
    ("experts", "shown"),
    [
        ({"num_local_experts": 8}, "num_local_experts=8"),
        ({"n_routed_experts": 256, "num_experts_per_tok": 8}, "n_routed_experts=256"),
        ({"moe_intermediate_size": 1408}, "moe_intermediate_size=1408"),
    ],
)
def test_a_model_with_expert_layers_is_refused(tmp_path, monkeypatch, experts, shown):
    def prepare(world, _pod_dir):
        world.experts = experts

    launch = _launch(tmp_path, monkeypatch, "decode", prepare=prepare)

    _assert_failed_before_the_grid(launch, 64, f"rejected: MoE models are not supported yet ({shown})")
    assert launch.world.events[-3:] == ["model_expert_keys", "write", "sleep"]


SGLANG_STEPS = {
    "load_plugins": "loading its plugins",
    "parse_server_args": "parsing its arguments",
    "resolve": "resolving its arguments",
    "model_expert_keys": "reading the model config",
}


@pytest.mark.parametrize("error_type", [ValueError, AssertionError, NotImplementedError, RuntimeError])
@pytest.mark.parametrize("step", list(SGLANG_STEPS))
def test_what_sglang_raises_to_refuse_a_setup_is_a_rejection_with_its_text(tmp_path, monkeypatch, step, error_type):
    def prepare(world, _pod_dir):
        world.raises[step] = error_type("FAKE: --page-size must divide\n    the chunked prefill size")

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    reason = (
        f"rejected: SGLang raised {error_type.__name__} while {SGLANG_STEPS[step]}: FAKE: --page-size must divide "
        "the chunked prefill size"
    )
    _assert_failed_before_the_grid(launch, 64, reason)
    assert launch.world.events[-3:] == [step, "write", "sleep"]


@pytest.mark.parametrize("step", ["sglang_version", *SGLANG_STEPS, "platform_facts", "resolved_dict"])
def test_any_other_exception_of_an_sglang_step_is_an_internal_error(tmp_path, monkeypatch, step):
    def prepare(world, _pod_dir):
        world.raises[step] = OSError("FAKE: fake/dense-model is not a local folder")

    launch = _launch(
        tmp_path, monkeypatch, "prefill", prepare=prepare, flags=["--dump-config-to", str(tmp_path / "dump.json")]
    )

    _assert_failed_before_the_grid(launch, 70, "internal_error: OSError: FAKE: fake/dense-model is not a local folder")
    assert "build_engine" not in launch.world.events


@pytest.mark.parametrize("status", [2, 0])
def test_an_exit_of_sglangs_parser_is_a_rejection(tmp_path, monkeypatch, status):
    def prepare(world, _pod_dir):
        world.raises["parse_server_args"] = SystemExit(status)

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    reason = f"rejected: SGLang's argument parser exited with status {status}; its message is on stderr"
    _assert_failed_before_the_grid(launch, 64, reason)
    assert launch.world.events == ["clock", "sglang_version", "load_plugins", "parse_server_args", "write", "sleep"]


def test_sglang_gets_the_other_flags_and_the_streaming_flag(tmp_path, monkeypatch):
    launch = _launch(tmp_path, monkeypatch, "prefill", flags=["--enable-metrics", "--benchmark-tim", "3"])

    assert launch.world.sglang_argv == [
        "--enable-metrics",
        "--benchmark-tim",
        "3",
        *_fake_sglang_argv(),
        STREAMING_FLAG,
    ]
    assert launch.code == 0


# ---------------------------------------------------------------------------------------------------------------
# Engine build, token range
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("error_type", [ValueError, AssertionError, NotImplementedError])
def test_a_setup_sglang_refuses_while_the_engine_is_built_is_a_rejection(tmp_path, monkeypatch, error_type):
    def prepare(world, _pod_dir):
        world.raises["build_engine"] = error_type("FAKE: SGLANG_RUST_SERVER is not supported")

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    reason = (
        f"rejected: SGLang raised {error_type.__name__} while building the engine: "
        "FAKE: SGLANG_RUST_SERVER is not supported"
    )
    _assert_failed_before_the_grid(launch, 64, reason)
    # No engine object exists, so nothing is shut down.
    assert launch.world.events[-3:] == ["build_engine", "write", "sleep"]


@pytest.mark.parametrize("error_type", [RuntimeError, TimeoutError, TypeError, AttributeError, OSError])
def test_an_engine_that_does_not_come_up_is_a_failed_run(tmp_path, monkeypatch, error_type):
    def prepare(world, _pod_dir):
        world.raises["build_engine"] = error_type("FAKE: Rank 0 scheduler died during initialization")

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    reason = (
        f"engine_call_failed: the engine build raised {error_type.__name__}: "
        "FAKE: Rank 0 scheduler died during initialization"
    )
    _assert_failed_before_the_grid(launch, 65, reason)


@pytest.mark.parametrize(
    ("vocabulary", "error", "detail"),
    [
        (None, AttributeError("FAKE: the tokenizer has no all_special_ids"), "AttributeError: FAKE: the tokenizer"),
        (None, ValueError("FAKE: engine.tokenizer_manager.tokenizer is None"), "ValueError: FAKE: engine.tokenizer"),
        (None, KeyError("size"), "KeyError: 'size'"),
        ({"size": 3, "excluded": [0, 1, 2]}, None, "ValueError: every token id in [0, 3) is excluded"),
        ({"size": 10}, None, "TypeError: ordinary_token_range() missing 1 required positional argument"),
        ({"size": "big", "excluded": []}, None, "TypeError: size must be an int"),
    ],
)
def test_a_token_range_that_cannot_be_taken_is_a_rejection(tmp_path, monkeypatch, vocabulary, error, detail):
    def prepare(world, _pod_dir):
        if error is not None:
            world.raises["vocabulary"] = error
        else:
            world.vocabulary = vocabulary

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    assert launch.code == 64
    on_disk = _assert_failed_before_the_grid(launch, 64, launch.artifact()["error"])
    assert on_disk["error"].startswith(f"rejected: the token range cannot be taken from the tokenizer: {detail}")
    # The engine was built, so it is shut down.
    assert launch.world.events[-4:] == ["vocabulary", "shutdown", "write", "sleep"]


def test_a_token_range_that_is_too_small_for_the_run_is_the_drivers_failure(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        world.vocabulary = {"size": 5000, "excluded": [10, 4000]}

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    # The range exists, so the launcher accepts it; the driver finds it too small for a content plan.
    _assert_failed(launch, 65, launch.artifact()["error"])
    assert launch.artifact()["error"].startswith(
        "token_range_too_small: the token range [11, 4000) holds 3989 ids, the content plan needs at least 4096"
    )
    assert launch.world.calls == [] and launch.world.events[-3:] == ["shutdown", "write", "sleep"]


def test_a_recorder_that_did_not_take_the_place_of_the_scheduler_fails_the_run(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        # FAKE of scheduler processes that do not see the variable: the stock class is built and no header is
        # written.
        world.hooks["build_engine"] = lambda: os.environ.pop(RECORDER_ENV)

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    assert launch.code == 65
    assert launch.artifact()["error"].startswith("recorder_header_invalid: 0 of 1 recorder header(s) in ")
    assert type(launch.world.schedulers[0]) is FakeBaseScheduler
    assert launch.world.calls == []


# ---------------------------------------------------------------------------------------------------------------
# Old files, recorder directory, config block
# ---------------------------------------------------------------------------------------------------------------


OTHER_FILES = {
    "collector-provenance.json": None,
    "point_manifest_receipt.json": b'{"receipt": 1}',
    "etcd.log": b"etcd\n",
    "runtime-preflight.json": b"{}",
    "engine.stdout.log": b"out\n",
    "engine.stderr.log": b"err\n",
    "benchmark_dp1.json": b'{"not": "ours"}',
    "benchmark_merged_worker_probe.json": b'{"not": "ours"}',
    "resolved-config-node7.json": b'{"not": "ours"}',
    "sglbench-recorder/older-run/sched-1.jsonl": b'{"type":"header"}\n',
}


@pytest.mark.parametrize("failing", [False, True])
def test_only_the_launchers_own_old_files_are_deleted(tmp_path, monkeypatch, failing):
    cell, pod_dir = _new_pod(tmp_path, "prefill")
    for name, data in OTHER_FILES.items():
        if data is not None:
            (pod_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (pod_dir / name).write_bytes(data)
    (pod_dir / "benchmark.json").write_bytes(b'{"stale": "rank"}')
    (pod_dir / "benchmark_merged.json").write_bytes(b'{"stale": "merged"}')
    before = {name: (pod_dir / name).read_bytes() for name in OTHER_FILES}
    seen = {}

    def prepare(world, _pod_dir):
        world.hooks["sglang_version"] = lambda: seen.update(at_start=sorted(p.name for p in pod_dir.glob("benchmark*")))
        if failing:
            world.experts = {"num_experts": 4}

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare, pod=(cell, pod_dir))

    # Both old result files were gone before SGLang was touched; nothing else was.
    assert seen["at_start"] == ["benchmark_dp1.json", "benchmark_merged_worker_probe.json"]
    assert {name: (pod_dir / name).read_bytes() for name in OTHER_FILES} == before
    assert launch.code == (64 if failing else 0)
    assert launch.artifact()["status"] == ("failed" if failing else "complete")
    # A failed run has no merged file, also not the stale one.
    assert launch.merged_path.exists() is not failing


def test_the_recorder_directory_is_new_absolute_and_named_to_the_scheduler_processes(tmp_path, monkeypatch):
    seen = {}

    def prepare(world, pod_dir):
        def at_build():
            seen["variable"] = os.environ[RECORDER_ENV]
            seen["entries"] = sorted(os.listdir(os.environ[RECORDER_ENV]))

        world.hooks["build_engine"] = at_build
        seen["before"] = (pod_dir / "sglbench-recorder").exists()

    monkeypatch.chdir(tmp_path)
    cell, pod_dir = _new_pod(tmp_path, "prefill")
    world = _FakeWorld()
    _install_fakes(monkeypatch, world)
    prepare(world, pod_dir)
    # The output path is given relative to the working directory.
    code = launcher.main(
        ["--benchmark-mode", "prefill", "--benchmark-output-path", "raw/pod-0/benchmark.json", *SMALL_CAP_FLAGS]
    )

    assert code == 0 and seen["before"] is False
    run_id = json.loads((pod_dir / "benchmark.json").read_text(encoding="utf-8"))["run_id"]
    assert seen["variable"] == str(pod_dir / "sglbench-recorder" / run_id)
    assert os.path.isabs(seen["variable"])
    # It existed and was empty when the engine build began.
    assert seen["entries"] == []
    assert len(run_id) == 32 and int(run_id, 16) >= 0


def test_config_is_equal_across_two_launches_and_the_run_id_differs(tmp_path, monkeypatch):
    cell, pod_dir = _new_pod(tmp_path, "decode")
    dump = pod_dir / "resolved-config-node0.json"
    flags = ["--dump-config-to", str(dump)]

    first = _launch(tmp_path, monkeypatch, "decode", pod=(cell, pod_dir), flags=flags)
    first_file, first_dump = first.artifact(), json.loads(dump.read_text(encoding="utf-8"))
    second = _launch(tmp_path, monkeypatch, "decode", pod=(cell, pod_dir), flags=flags)
    second_file, second_dump = second.artifact(), json.loads(dump.read_text(encoding="utf-8"))

    assert (first.code, second.code) == (0, 0)
    assert first_file["run_id"] != second_file["run_id"]
    assert first_file["config"] == second_file["config"]
    assert first_file["grid_digest"] == second_file["grid_digest"]
    # The config block holds exactly the settings that repeat, and no value that names a run.
    assert list(first_file["config"]) == [
        "mode",
        "warmup_iterations",
        "timeout",
        "prefill_max_new_token_samples",
        "prefill_max_kv_read_token_samples",
        "decode_max_kv_read_token_samples",
        "decode_max_batch_size_samples",
        "prefix_max_batch_size_samples",
        "max_batch_size",
        "points_source",
        "content_seed",
        "output_path",
    ]
    text = json.dumps(first_file["config"])
    assert first_file["run_id"] not in text and "sglbench-recorder" not in text
    assert all(value is None or type(value) in (int, str) for value in first_file["config"].values())
    # Each launch has a recorder directory of its own; the older one is kept.
    assert [path.name for path in second.recorder_dirs()] == sorted([first_file["run_id"], second_file["run_id"]])
    assert (first_dump["config"]["benchmark"]["run_id"], second_dump["config"]["benchmark"]["run_id"]) == (
        first_file["run_id"],
        second_file["run_id"],
    )
    _assert_accepted(second)


# ---------------------------------------------------------------------------------------------------------------
# The dump of the resolved arguments
# ---------------------------------------------------------------------------------------------------------------


def test_the_dump_is_written_atomically_before_the_engine_is_built(tmp_path, monkeypatch):
    cell, pod_dir = _new_pod(tmp_path, "prefill")
    dump = pod_dir / "resolved-config-node0.json"
    seen = {}
    replaced = []
    real_replace = os.replace

    def noted_replace(source, target):
        replaced.append((str(source), str(target), Path(source).exists(), Path(target).exists()))
        return real_replace(source, target)

    def prepare(world, _pod_dir):
        def at_build():
            seen["dump_at_build"] = json.loads(dump.read_text(encoding="utf-8"))
            seen["names_at_build"] = sorted(path.name for path in pod_dir.iterdir())

        world.hooks["build_engine"] = at_build
        world.raises["build_engine"] = RuntimeError("FAKE: the engine did not come up")
        monkeypatch.setattr(launcher.os, "replace", noted_replace)

    launch = _launch(
        tmp_path, monkeypatch, "prefill", prepare=prepare, pod=(cell, pod_dir), flags=["--dump-config-to", str(dump)]
    )

    # A run that fails in the build still leaves the dump.
    assert launch.code == 65
    content_now = json.loads(dump.read_text(encoding="utf-8"))
    assert content_now == seen["dump_at_build"]
    run_id = launch.artifact()["run_id"]
    assert content_now == {
        "producer": "aisim_sglbench",
        "sglang_version": "0.5.23",
        "config": {
            "server_args": {
                "model_path": "fake/dense-model",
                "page_size": 1,
                "tp_size": 1,
                # What JSON cannot hold is written as its repr.
                "custom_sigquit_handler": repr(_fake_limits),
                "ranks": [0, 1],
                "nested": {"ratio": "inf", "<int> 3": "FAKE value under a key that is not text"},
            },
            "benchmark": {
                "mode": "prefill",
                "points_file": None,
                "points_sha256": None,
                "prefill_max_new_token_samples": 3,
                "prefill_max_kv_read_token_samples": 3,
                "decode_max_kv_read_token_samples": 3,
                "decode_max_batch_size_samples": 3,
                "prefix_max_batch_size_samples": 2,
                "benchmark_max_batch_size": None,
                "warmup_iterations": 5,
                "timeout": FAKE_TIMEOUT_SECONDS,
                "output_path": str(launch.rank_path),
                "content_seed": "0",
                "worker_id": FAKE_WORKER_ID,
                "run_id": run_id,
            },
        },
    }
    # It is one JSON object without NaN or Infinity, and the environment is not in it.
    json.loads(dump.read_text(encoding="utf-8"), parse_constant=pytest.fail)
    assert "environ" not in dump.read_text(encoding="utf-8") and "DYN_" not in dump.read_text(encoding="utf-8")
    # It came into place by one rename of a hidden temporary in the same directory, and nothing was left behind.
    (dump_rename,) = [entry for entry in replaced if entry[1] == str(dump)]
    source = Path(dump_rename[0])
    assert source.parent == pod_dir and source.name.startswith(".resolved-config-node0.json.")
    assert source.name.endswith(".tmp") and dump_rename[2:] == (True, False)
    assert not any(fnmatch.fnmatch(source.name, pattern) for pattern in FORBIDDEN_FILE_PATTERNS)
    assert seen["names_at_build"] == ["collector-provenance.json", "resolved-config-node0.json", "sglbench-recorder"]
    assert launch.world.events.index("resolved_dict") < launch.world.events.index("build_engine")


def test_no_dump_without_the_flag_and_a_dump_directory_is_made(tmp_path, monkeypatch):
    plain = _launch(tmp_path / "plain", monkeypatch, "prefill")
    assert "resolved_dict" not in plain.world.events
    assert not list(plain.pod.glob("resolved-config*"))

    dump = tmp_path / "made" / "deeper" / "resolved-config-node0.json"
    launch = _launch(tmp_path / "made-run", monkeypatch, "prefill", flags=[f"--dump-config-to={dump}"])
    assert launch.code == 0 and json.loads(dump.read_text(encoding="utf-8"))["producer"] == "aisim_sglbench"
    assert [path.name for path in dump.parent.iterdir()] == ["resolved-config-node0.json"]


def test_a_dump_that_cannot_be_written_is_a_rejection(tmp_path, monkeypatch):
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory", encoding="utf-8")
    dump = blocker / "resolved-config-node0.json"

    launch = _launch(tmp_path, monkeypatch, "prefill", flags=["--dump-config-to", str(dump)])

    assert launch.code == 64
    error = launch.artifact()["error"]
    assert error.startswith(f"rejected: --dump-config-to '{dump}': ")
    assert "build_engine" not in launch.world.events
    assert blocker.read_text(encoding="utf-8") == "not a directory"


def test_values_json_cannot_hold_are_written_as_their_repr():
    class Unknown:
        def __repr__(self) -> str:
            return "<FAKE object>"

    ready = launcher._json_ready(
        {
            "text": "x",
            "flag": True,
            "none": None,
            "number": 3,
            "ratio": 0.5,
            "nan": float("nan"),
            "minus": float("-inf"),
            "tuple": (1, (2, Unknown())),
            "set": {7},
            (1, 2): "tuple key",
            None: "none key",
            "object": Unknown(),
            "bytes": b"x",
        }
    )

    assert ready == {
        "text": "x",
        "flag": True,
        "none": None,
        "number": 3,
        "ratio": 0.5,
        "nan": "nan",
        "minus": "-inf",
        "tuple": [1, [2, "<FAKE object>"]],
        "set": "{7}",
        "<tuple> (1, 2)": "tuple key",
        "<NoneType> None": "none key",
        "object": "<FAKE object>",
        "bytes": "b'x'",
    }
    assert json.loads(json.dumps(ready, allow_nan=False)) == ready


# ---------------------------------------------------------------------------------------------------------------
# The module itself
# ---------------------------------------------------------------------------------------------------------------


def test_the_engine_adapter_makes_one_batch_call_with_the_time_limit(monkeypatch):
    seen = []

    def generate_batch(*arguments, **keywords):
        seen.append((arguments, keywords))
        return ["FAKE answer"]

    monkeypatch.setattr(compat, "generate_batch", generate_batch)
    engine = object()

    stops = launcher._SignalStops()
    answer = launcher._EngineAdapter(engine, stops).generate([[1, 2]], ["rid"], 7)

    assert answer == ["FAKE answer"]
    assert seen == [((engine, [[1, 2]], ["rid"], 7, 900.0), {})]
    assert launcher.ENGINE_CALL_TIMEOUT_SECONDS == 900.0


def test_nothing_runs_at_import():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    allowed_calls = {"_only", "_int_at_most", "fields", "removeprefix"}
    guards = []
    for position, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Expr):
            # Only the module docstring.
            assert position == 0 and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            calls = [inner for inner in ast.walk(node) if isinstance(inner, ast.Call)]
            names = {call.func.id if isinstance(call.func, ast.Name) else call.func.attr for call in calls}
            assert names <= allowed_calls, f"line {node.lineno} calls {sorted(names - allowed_calls)} at import"
            continue
        assert isinstance(node, ast.If), f"line {node.lineno}: a {type(node).__name__} statement at module level"
        guards.append(node)
    # ``main`` is called in one place, behind the guard, and its value is the exit code.
    (guard,) = guards
    assert ast.unparse(guard.test) == "__name__ == '__main__'"
    assert [ast.unparse(statement) for statement in guard.body] == ["sys.exit(main())"] and not guard.orelse
    assert tree.body[-1] is guard
    # The module reads the environment in one place only: the mapping ``main`` hands to its readers.
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert source.count("os.environ") == 3 and "os.getenv" not in source
    assert "FPM_" not in source.replace("DYN_FPM_", "").replace("every FPM_* variable", "")


def test_the_module_imports_without_sglang_torch_or_numpy_and_changes_nothing(tmp_path):
    script = f"""
import importlib.abc, os, signal, sys

class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {{"sglang", "torch", "numpy", "aisim", "aisim_core", "collector", "dynamo"}}:
            raise ImportError(f"refused in this test: {{name}}")
        return None

sys.meta_path.insert(0, Refuse())
sys.path.append({str(RUNTIME)!r})

def state():
    handlers = [signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)]
    return dict(os.environ), handlers, sorted(os.listdir("."))

before = state()
import sglbench_launcher
assert state() == before
refused = {{"sglang", "torch", "numpy", "aisim", "aisim_core", "collector", "dynamo"}}
assert not [name for name in sys.modules if name.split(".")[0] in refused]
assert callable(sglbench_launcher.main)
print("imported")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert (result.stdout, result.stderr) == ("imported\n", "")
    assert list(tmp_path.iterdir()) == []


def test_started_by_path_the_script_reads_its_flags_and_exits_2_without_a_mode(tmp_path):
    # Started by path, Python puts the script's directory on sys.path; that is how the sibling modules are found.
    command = [sys.executable, "-S", "-B", str(MODULE_PATH)]
    options = {
        "cwd": tmp_path,
        "env": {"PATH": os.environ.get("PATH", "")},
        "capture_output": True,
        "text": True,
        "check": False,
        "timeout": 120,
    }

    missing = subprocess.run([*command, "--model-path", "fake/model"], **options)
    assert missing.returncode == 2
    assert "--benchmark-mode is required (or DYN_BENCHMARK_MODE): prefill, decode, agg" in missing.stderr

    usage = subprocess.run([*command, "--help"], **options)
    assert usage.returncode == 0
    for flag in (
        "--benchmark-mode",
        "--benchmark-points-file",
        "--prefill-max-new-token-samples",
        "--prefill-max-kv-read-token-samples",
        "--decode-max-kv-read-token-samples",
        "--decode-max-batch-size-samples",
        "--prefix-max-batch-size-samples",
        "--benchmark-max-batch-size",
        "--benchmark-warmup-iterations",
        "--benchmark-output-path",
        "--benchmark-timeout",
        "--dump-config-to",
    ):
        assert flag in usage.stdout
    assert list(tmp_path.iterdir()) == []


def test_the_constants_the_modules_share():
    assert launcher.RECORDER_DIR_NAME == "sglbench-recorder"
    assert recorder.ENV_RECORDER_DIR == RECORDER_ENV
    assert launcher.STREAMING_FLAG == STREAMING_FLAG
    assert launcher.REASON_STOPPED == "stopped"
    assert (driver.REASON_REJECTED, driver.REASON_INTERNAL, driver.REASON_ENGINE_CALL) == (
        "rejected",
        "internal_error",
        "engine_call_failed",
    )
    assert compat.TESTED_VERSIONS == ("0.5.22", "0.5.23")
    # No key of the launcher's producer facts is one the driver or the artifact module writes.
    reserved = {*DRIVER_PRODUCER_KEYS, "name", "artifact_format", "omitted_launcher_blocks"}
    assert not reserved & {*LAUNCHER_PRODUCER_KEYS, "engine_shutdown_error"}


def test_the_header_and_the_docstring_of_the_module():
    lines = MODULE_PATH.read_text(encoding="utf-8").splitlines()
    assert [line.removeprefix("# ") for line in lines[:9]] == [
        "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team",
        "SPDX-License-Identifier: Apache-2.0",
        "The start-up checks read SGLang's resolved arguments at v0.5.22 and v0.5.23; see THIRD_PARTY_NOTICES.md.",
        "Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d",
        "Source: https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90",
        "The benchmark flags, their defaults and their environment variables are re-implemented from ai-dynamo/dynamo",
        "(Apache-2.0, https://github.com/ai-dynamo/dynamo), components/src/dynamo/vllm/backend_args.py and",
        "instrumented_scheduler.py at b20fe6fe94021b854597965c7305931175e8f482. See THIRD_PARTY_NOTICES.md.",
    ]
    assert all(line.startswith("# ") for line in lines[:9])
    assert lines[9] == "" and lines[10].startswith('"""')
    docstring = " ".join(launcher.__doc__.split())
    assert (
        "No upstream function or block of source is copied. What follows SGLang and Dynamo is named in the header "
        "of this file and recorded in THIRD_PARTY_NOTICES.md."
    ) in docstring
    assert "baa9befa381511cd4d56f83980951e5a45fffc6d" in docstring
    assert "297e54fd105e974cdf401e5b4d20253a22c16f90" in docstring
    # Plain ASCII, and no line above the length limit.
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert source.isascii() and max(len(line) for line in lines) <= 120
    # The module imports SGLang nowhere: it reaches it through sglbench_compat only.
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert sorted(imported) == [
        "__future__",
        "argparse",
        "collections.abc",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "os",
        "secrets",
        "sglbench_artifact",
        "sglbench_compat",
        "sglbench_driver",
        "sglbench_grid",
        "sglbench_recorder",
        "signal",
        "sys",
        "time",
        "traceback",
        "uuid",
    ]
    # Comments and docstrings name sources the repository can resolve, never the notes of the build.
    lowered = source.lower()
    for banned in ("phase 0", "phase 1", "phase 2", "the design", "the contract", "spec.md", "rejections.md"):
        assert banned not in lowered, banned


# ---------------------------------------------------------------------------------------------------------------
# Tests proposed by the review
# ---------------------------------------------------------------------------------------------------------------


def test_a_sigterm_after_sglang_took_the_handler_over_is_still_recorded_as_a_stop(tmp_path, monkeypatch):
    victim = _planned("decode").points[1]

    def prepare(world, _pod_dir):
        # FAKE of SGLang in a benchmark run: its handler notes the signal and its watchdog never gets to exit.
        world.sglang_takes_sigterm = True
        world.call_faults[(victim.benchmark_id, "run", 0)] = "sigterm"

    launch = _launch(tmp_path, monkeypatch, "decode", prepare=prepare)

    on_disk = _assert_failed(launch, 65, "stopped: signal SIGTERM")
    assert on_disk["coverage"]["completed_points"] == victim.benchmark_id - 1
    assert launch.world.events[-3:] == ["shutdown", "write", "sleep"]
    assert launch.world.sglang_sigterm_notes == []


def test_a_stop_that_is_lost_on_its_way_is_still_recorded(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        def at_build():
            try:
                _deliver(signal.SIGTERM)
            except BaseException:
                # FAKE of code under the signal that catches everything.
                pass

        world.hooks["build_engine"] = at_build

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    _assert_failed(launch, 65, "stopped: signal SIGTERM")


def test_a_refused_flag_leaves_a_failed_artifact_in_an_output_directory_that_did_not_exist(tmp_path, monkeypatch):
    world = _FakeWorld()
    _install_fakes(monkeypatch, world)
    output = tmp_path / "new" / "results" / "benchmark.json"

    code = launcher.main(
        ["--benchmark-mode", "prefill", "--benchmark-output-path", str(output), "--benchmark-timeout", "0"]
    )

    assert code == 64
    on_disk = json.loads(output.read_text(encoding="utf-8"))
    assert (on_disk["status"], on_disk["error"]) == ("failed", "rejected: --benchmark-timeout must be above 0, got 0")
    assert world.events == ["clock", "write", "sleep"]
    assert sorted(path.name for path in output.parent.iterdir()) == ["benchmark.json"]


def test_secrets_of_the_resolved_arguments_are_not_written_to_the_dump(tmp_path, monkeypatch):
    cell, pod_dir = _new_pod(tmp_path, "prefill")
    dump = pod_dir / "resolved-config-node0.json"
    secrets_of_sglang = {
        "api_key": "sk-FAKE-secret",
        "admin_api_key": "adm-FAKE-secret",
        "watermark_key": 123456789,
        "watermark_key_b": 987654321,
        "watermark_config": "/FAKE/watermark.json",
    }

    def prepare(world, _pod_dir):
        world.resolved_dict.update(secrets_of_sglang)
        # FAKE: a secret field the operator left unset.
        world.resolved_dict["watermark_key_b"] = None

    launch = _launch(
        tmp_path, monkeypatch, "prefill", prepare=prepare, pod=(cell, pod_dir), flags=["--dump-config-to", str(dump)]
    )

    assert launch.code == 0
    text = dump.read_text(encoding="utf-8")
    server_args = json.loads(text)["config"]["server_args"]
    given = {name: value for name, value in secrets_of_sglang.items() if name != "watermark_key_b"}
    assert {name: server_args[name] for name in given} == dict.fromkeys(given, "<redacted>")
    assert not any(str(value) in text for value in secrets_of_sglang.values())
    # A field that is unset stays null, and the record the fake handed out is not changed.
    assert server_args["watermark_key_b"] is None
    assert launch.world.resolved_dict["api_key"] == "sk-FAKE-secret"


def test_a_signal_while_the_token_range_is_taken_is_recorded_as_a_stop(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        world.hooks["vocabulary"] = lambda: _deliver(signal.SIGTERM)

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    _assert_failed_before_the_grid(launch, 65, "stopped: signal SIGTERM")
    # The engine was built, so it is shut down before the write.
    assert launch.world.events[-4:] == ["vocabulary", "shutdown", "write", "sleep"]


def test_an_exit_raised_by_the_shutdown_does_not_stop_the_write(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        world.raises["shutdown"] = SystemExit(0)

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    on_disk = _assert_accepted(launch)
    assert on_disk["producer"]["engine_shutdown_error"] == "SystemExit: 0"
    assert launch.world.events[-3:] == ["shutdown", "write", "sleep"]


@pytest.mark.parametrize("word", ["0", "false", "FALSE", " no ", "off"])
def test_every_false_word_of_the_override_switches_the_flag_off(word):
    options = _options(["--benchmark-collect-imbalanced"], {"DYN_FPM_BENCH_COLLECT_IMBALANCED": word})
    assert options.rejection is None


def test_a_flag_decides_and_its_variable_is_not_even_read_then():
    options = _options(
        ["--benchmark-timeout", "60", "--prefill-max-new-token-samples", "4", "--benchmark-warmup-iterations", "1"],
        {
            "DYN_BENCHMARK_TIMEOUT": "soon",
            "DYN_PREFILL_MAX_NEW_TOKEN_SAMPLES": "",
            "DYN_BENCHMARK_WARMUP_ITERATIONS": "-3",
        },
    )
    assert options.rejection is None
    assert (options.timeout, options.warmup_iterations) == (60, 1)
    assert options.cap_values["prefill_max_new_token_samples"] == 4


def test_a_dump_whose_rename_fails_leaves_no_temporary_file(tmp_path, monkeypatch):
    cell, pod_dir = _new_pod(tmp_path, "prefill")
    dump = pod_dir / "resolved-config-node0.json"
    dump.mkdir()  # a directory in the place of the file: the temporary is written, the rename fails

    launch = _launch(tmp_path, monkeypatch, "prefill", pod=(cell, pod_dir), flags=["--dump-config-to", str(dump)])

    assert launch.code == 64
    assert launch.artifact()["error"].startswith(f"rejected: --dump-config-to '{dump}': ")
    assert sorted(path.name for path in pod_dir.iterdir()) == [
        "benchmark.json",
        "collector-provenance.json",
        "resolved-config-node0.json",
        "sglbench-recorder",
    ]
    assert "build_engine" not in launch.world.events


def test_an_unexpected_error_prints_its_traceback(tmp_path, monkeypatch, capsys):
    def prepare(world, _pod_dir):
        world.raises["platform_facts"] = TypeError("FAKE: is_cuda is None")

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    assert launch.code == 70
    err = capsys.readouterr().err
    assert "Traceback (most recent call last)" in err and "TypeError: FAKE: is_cuda is None" in err


def test_a_failed_write_of_a_partial_run_ends_with_exit_code_70(tmp_path, monkeypatch, capsys):
    last_done = _planned("decode").points[1]

    def prepare(world, pod_dir):
        world.call_faults[(last_done.benchmark_id, "run", 0)] = "deadline"
        world.hooks["shutdown"] = lambda: shutil.rmtree(pod_dir)

    launch = _launch(tmp_path, monkeypatch, "decode", prepare=prepare)

    assert launch.code == 70
    lines = _stderr_lines(capsys)
    assert [line for line in lines if line.startswith("[sglbench] FAILED: ")][0].startswith(
        "[sglbench] FAILED: internal_error: the result files could not be written: "
    )


def test_sglangs_variables_are_tested_as_they_stand_after_sglang_was_imported(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        # FAKE of what importing sglang does with a variable under its legacy prefix (srt/environ.py
        # ``_handle_deprecated_envs``, called at the import of sglang): SGL_X becomes SGLANG_X.
        world.hooks["sglang_version"] = lambda: monkeypatch.setenv(
            "SGLANG_RADIX_FORCE_MISS", os.environ["SGL_RADIX_FORCE_MISS"]
        )

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare, env={"SGL_RADIX_FORCE_MISS": "1"})

    _assert_failed_before_the_grid(launch, 64, "rejected: SGLANG_RADIX_FORCE_MISS disables every cache hit")
