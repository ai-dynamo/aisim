# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# The fake resolved arguments, scheduler, batch and hook registry follow SGLang at v0.5.22 and v0.5.23; see
# THIRD_PARTY_NOTICES.md.
# Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d
# Source: https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90
# The expected benchmark flags, their defaults, their environment variables and the dataset of a text run follow
# ai-dynamo/dynamo (Apache-2.0, https://github.com/ai-dynamo/dynamo), components/src/dynamo/vllm/backend_args.py and
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
- The dataset of a text run is FAKE: ``_fake_dataset_records`` types in records in the layout of the default
  dataset, and a test writes them to a file under pytest's temporary directory. ``_FakeTokenizer`` turns
  every word into one FAKE id; no tokenizer is involved. No test opens a connection: a fixture puts a
  refusal in the place of ``urllib.request.urlopen``, and the tests of a download put a FAKE response there.
  The same fixture points the default cache directory into the temporary directory.

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
import http.client
import inspect
import json
import os
import shutil
import signal
import subprocess
import sys
import types
import urllib.error
from pathlib import Path

import pytest
from collector.fpm_forward.native_artifact import validate_native_collection
from collector.fpm_forward.types import ParallelTopology

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
        "model_vocabulary_size",
        "encode_texts",
        "tokenizer_class",
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
    "expert_keys",
    "moe",
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
# at v0.5.23 only. The six fields of the expert layers have these defaults for every model (``ep_size`` and
# ``moe_dense_tp_size`` in parallel.py, the other four in exec_.py ``ExecMoe``).
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
    "moe_a2a_backend": "none",
    "moe_dense_tp_size": None,
    "enable_eplb": False,
    "ep_num_redundant_experts": 0,
    "ep_size": 1,
    "moe_runner_backend": "auto",
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
_A2A = "only the MoE all-to-all backend none is supported"
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
    ("moe_a2a_backend", "deepep", f"{_A2A} (moe_a2a_backend=deepep)"),
    ("moe_dense_tp_size", 1, "moe_dense_tp_size must not be set (got 1)"),
    ("enable_eplb", True, "EPLB is not supported (enable_eplb=true)"),
    ("ep_num_redundant_experts", 2, "redundant experts are not supported (ep_num_redundant_experts=2)"),
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
    ("moe_a2a_backend", None, f"{_A2A} (moe_a2a_backend=null)"),
    ("moe_a2a_backend", "None", f"{_A2A} (moe_a2a_backend=None)"),
    ("moe_a2a_backend", "flashinfer", f"{_A2A} (moe_a2a_backend=flashinfer)"),
    # SGLang itself allows 1 and the TP size; either gives the dense layers a TP size of their own.
    ("moe_dense_tp_size", 2, "moe_dense_tp_size must not be set (got 2)"),
    ("moe_dense_tp_size", 0, "moe_dense_tp_size must not be set (got 0)"),
    ("enable_eplb", None, "EPLB is not supported (enable_eplb=null)"),
    ("enable_eplb", 0, "EPLB is not supported (enable_eplb=0)"),
    ("ep_num_redundant_experts", None, "redundant experts are not supported (ep_num_redundant_experts=null)"),
    ("ep_num_redundant_experts", False, "redundant experts are not supported (ep_num_redundant_experts=false)"),
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
_CONTENT_WORDS = "is not supported (accepted: random, sharegpt_chain)"
# The refusal of seeded ids for a model with expert layers; the braces take the first expert key and its value.
_SEEDED_WITH_EXPERTS = (
    "DYN_BENCH_PREFILL_CONTENT=random is not supported for a model with expert layers ({}): the routing of a "
    "token to its experts depends on the content"
)
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
        f"DYN_BENCH_PREFILL_CONTENT='sharegpt' {_CONTENT_WORDS}",
    ),
    # The word is compared as it stands: no other letter case and no blank around it.
    ([], {"DYN_BENCH_PREFILL_CONTENT": "Random"}, f"DYN_BENCH_PREFILL_CONTENT='Random' {_CONTENT_WORDS}"),
    (
        [],
        {"DYN_BENCH_PREFILL_CONTENT": "SHAREGPT_CHAIN"},
        f"DYN_BENCH_PREFILL_CONTENT='SHAREGPT_CHAIN' {_CONTENT_WORDS}",
    ),
    ([], {"DYN_BENCH_PREFILL_CONTENT": " random"}, f"DYN_BENCH_PREFILL_CONTENT=' random' {_CONTENT_WORDS}"),
    ([], {"DYN_BENCH_PREFILL_CONTENT": "zeros"}, f"DYN_BENCH_PREFILL_CONTENT='zeros' {_CONTENT_WORDS}"),
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
    ([], {"DYN_BENCH_PREFILL_CONTENT": "sharegpt_chain"}),
    ([], {"DYN_BENCH_PREFILL_CONTENT": ""}),
    ([], {"DYN_BENCH_POOL_TAG": "", "DYN_BENCH_GIANT_KV_THRESHOLD": "", "DYN_BENCH_GIANT_KV_REPEATS": ""}),
    # The dataset variables are kept as text for a text run and refuse nothing by themselves; the FPM_*
    # variables of the gate are never read.
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
        parser.add_argument("--ep-size", type=int, default=1)
        parser.add_argument("--page-size", type=int, default=1)
        parser.add_argument("--stream-interval", type=int, default=1)
        parser.add_argument(STREAMING_FLAG, action="store_true")
        parsed = parser.parse_args(argv)
        world.sglang_argv = list(argv)
        # The FAKE engine follows the flags, so a flag that does not reach this parser shows in the run.
        world.use(parsed.page_size, parsed.tp_size)
        world.resolved["ep_size"] = parsed.ep_size
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
                tokenizer=_FakeTokenizer(world), model_config=types.SimpleNamespace(vocab_size=FAKE_MODEL_VOCAB_SIZE)
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
    """FAKE of the tokenizer SGLang serves with: the four attributes sglbench_compat reads and the batch call.

    The batch call turns every word of a text into one FAKE id inside the token range of the prompts. Three
    written forms stand for tokens outside it, as a real tokenizer gives the id of a special token for its
    written form inside a text: ``<FAKE-special>`` and ``<FAKE-added>``, whose ids the FAKE model counts, and
    ``<FAKE-beyond>``, whose id is the first one the FAKE model does not count.
    """

    vocab_size = FAKE_VOCABULARY["size"]
    all_special_ids = (0, 1, 2)
    additional_stop_token_ids = frozenset({59_999})

    def __init__(self, world: _FakeWorld | None = None) -> None:
        self.world = world

    def get_added_vocab(self) -> dict:
        return {"<FAKE added a>": 50_000, "<FAKE added b>": 50_001, "<FAKE special>": 2}

    @staticmethod
    def fake_id(word: str) -> int:
        if word == "<FAKE-special>":
            return 2
        if word == "<FAKE-added>":
            return 50_000
        if word == "<FAKE-beyond>":
            return FAKE_MODEL_VOCAB_SIZE
        return 100 + int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % 49_000

    def __call__(self, texts, add_special_tokens=True) -> dict:
        assert add_special_tokens is False, "the benchmark asks for no BOS or EOS around a text"
        world = self.world
        if world is not None:
            world.tokenizer_batches.append(list(texts))
            world.note("tokenizer")
            if world.tokenizer_ids_outside:
                # FAKE: a tokenizer whose every id lies above the ids the model counts.
                return {"input_ids": [[FAKE_MODEL_VOCAB_SIZE + 5 for _word in text.split()] for text in texts]}
        return {"input_ids": [[self.fake_id(word) for word in text.split()] for text in texts]}


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
        # The texts of every batch call of the FAKE tokenizer, a switch that puts all its ids above the ids the
        # model counts, and that count.
        self.tokenizer_batches: list[list[str]] = []
        self.tokenizer_ids_outside = False
        self.model_vocab_size = FAKE_MODEL_VOCAB_SIZE

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
            # The FAKE output tokens are negative numbers, so they equal no token id: not an allocated one, not
            # a seeded one and not an id of a text, which the pool keeps from 0 upwards.
            world.cache.finish_prefill([*prompt, *range(-1, -max_new_tokens, -1)])
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

    def model_vocabulary_size(engine) -> int:
        assert isinstance(engine, _FakeEngine)
        step("model_vocabulary_size")
        return world.model_vocab_size

    def encode_texts(engine, texts):
        assert isinstance(engine, _FakeEngine)
        step("encode_texts")
        return _FakeTokenizer(world)(texts, add_special_tokens=False)["input_ids"]

    def tokenizer_class(engine) -> str:
        assert isinstance(engine, _FakeEngine)
        step("tokenizer_class")
        return "fake_tokenizers.FakeTokenizer"

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
        model_vocabulary_size,
        encode_texts,
        tokenizer_class,
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


def _refuse_the_network(*args, **kwargs):
    raise AssertionError(f"a test tried to open a connection: urlopen{args!r}")


@pytest.fixture(autouse=True)
def _process_state(monkeypatch, tmp_path):
    """Every test starts without the variables the launcher reads and gets its signal handlers back.

    No test may reach the network or write outside pytest's temporary directory. ``urlopen`` is a refusal
    unless a test puts a FAKE response in its place, HF_HOME is unset, and the default cache directory of the
    dataset lies in the temporary directory.
    """
    for name in list(os.environ):
        if name.startswith(("DYN_", "SGLANG_", "FPM_", "AISIM_SGLBENCH_")) or name == "HF_HOME":
            monkeypatch.delenv(name)
    monkeypatch.setattr(launcher.urllib.request, "urlopen", _refuse_the_network)
    monkeypatch.setattr(launcher, "DEFAULT_DATASET_CACHE_DIR", str(tmp_path / "FAKE-default-cache"))
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
    # The record of the FAKE dataset a text run must write, as the test worked it out; None otherwise.
    dataset: dict | None = None

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


# ---------------------------------------------------------------------------------------------------------------
# FAKE dataset of a text run
# ---------------------------------------------------------------------------------------------------------------

# The variables of a text run, typed here on their own.
ENV_CONTENT = "DYN_BENCH_PREFILL_CONTENT"
ENV_DATASET = "DYN_BENCH_KV_WARMUP_DATASET"
ENV_CACHE_DIR = "DYN_BENCH_KV_WARMUP_CACHE_DIR"
ENV_SHA256 = "DYN_BENCH_KV_WARMUP_SHA256"
# A FAKE address. The top-level name ``invalid`` is reserved and resolves nowhere; no test connects to it.
FAKE_DATASET_URL = "https://datasets.fake.invalid/FAKE/resolve/main/FAKE_conversations.json?download=true"
FAKE_DATASET_NAME = "FAKE_conversations.json"


def _fake_dataset_records(count: int = 160) -> list[dict]:
    """FAKE records in the layout of the default dataset: an id and a list of turns with ``from`` and ``value``.

    Every record has two turns of numbered FAKE words. Some hold the written form of a FAKE special token, and
    some that of a FAKE token the model does not count.
    Three more records are ones the recipe drops or shortens: a text below 64 characters, a record without
    turns and a turn without a value.
    """
    records = []
    for index in range(count):
        words = [f"fake{(index * 31 + step * 7) % 400}" for step in range(40 + index % 30)]
        if index % 9 == 0:
            words.insert(5, "<FAKE-special>")
        if index % 13 == 4:
            words.insert(7, "<FAKE-beyond>")
        half = len(words) // 2
        turns = [
            {"from": "human", "value": " ".join(words[:half])},
            {"from": "gpt", "value": " ".join(words[half:])},
        ]
        if index % 11 == 0:
            turns.insert(1, {"from": "system"})
        records.append({"id": f"FAKE-{index}", "conversations": turns})
    records.append({"id": "FAKE-short", "conversations": [{"from": "human", "value": "FAKE: too short"}]})
    records.append({"id": "FAKE-no-turns"})
    return records


def _fake_kept_texts(records: list[dict]) -> list[str]:
    """The texts the recipe keeps of FAKE records, worked out here on its own.

    The values of a record joined by a newline; a text of at least 64 characters whose SHA-256 starts with an
    even byte is kept (Dynamo's ``_kvwarm_load_texts``).
    """
    texts = []
    for record in records:
        text = "\n".join(turn["value"] for turn in record.get("conversations", []) if turn.get("value"))
        if len(text) >= 64 and hashlib.sha256(text.encode()).digest()[0] % 2 == 0:
            texts.append(text)
    return texts


def _write_fake_dataset(directory: Path, records: object = None, name: str = FAKE_DATASET_NAME) -> tuple[Path, str]:
    """Write FAKE records as a JSON file under ``directory``; returns its path and its SHA-256."""
    directory.mkdir(parents=True, exist_ok=True)
    data = json.dumps(_fake_dataset_records() if records is None else records).encode()
    path = directory / name
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


class _FakeResponse:
    """FAKE of what ``urllib.request.urlopen`` returns: a context manager with ``read``."""

    def __init__(self, data: bytes, fail_after: BaseException | None = None, on_read=None) -> None:
        self.data = data
        self.fail_after = fail_after
        self.on_read = on_read
        self.offset = 0

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_error: object) -> bool:
        return False

    def read(self, size: int) -> bytes:
        if self.on_read is not None:
            self.on_read()
        piece = self.data[self.offset : self.offset + size]
        self.offset += len(piece)
        if not piece and self.fail_after is not None:
            # FAKE: the transfer breaks off after its last piece.
            raise self.fail_after
        return piece


class _FakeUrlopen:
    """FAKE ``urlopen``: it answers with FAKE bytes or raises, and notes how it was called. No connection."""

    def __init__(self, data: bytes = b"", error: BaseException | None = None, fail_after=None, on_read=None) -> None:
        self.data = data
        self.error = error
        self.fail_after = fail_after
        self.on_read = on_read
        self.calls: list[tuple] = []

    def __call__(self, *args, **kwargs) -> _FakeResponse:
        self.calls.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return _FakeResponse(self.data, self.fail_after, self.on_read)


def _launch_text(
    tmp_path: Path,
    monkeypatch,
    mode: str = "decode",
    *,
    records: object = None,
    word: str | None = "sharegpt_chain",
    env: dict | None = None,
    **options,
) -> _Launch:
    """``_launch`` for a text run: a FAKE dataset file as a local path, and the content word ``word``.

    ``word`` None leaves the content variable unset. ``env`` is put over both variables, so a test can point
    the dataset elsewhere. The launch carries the dataset record the artifact must hold.
    """
    records = _fake_dataset_records() if records is None else records
    path, digest = _write_fake_dataset(tmp_path / "FAKE-datasets", records)
    variables = {ENV_DATASET: str(path), **({} if word is None else {ENV_CONTENT: word}), **(env or {})}
    launch = _launch(tmp_path, monkeypatch, mode, env=variables, **options)
    launch.dataset = {
        "path": str(path),
        "sha256": digest,
        "collection_pool": "conversation_sha256_even",
        "conversations": len(_fake_kept_texts(records)) if isinstance(records, list) else 0,
        "url": None,
        "downloaded": False,
    }
    return launch


def _fake_pool_counts(texts: list[str]) -> dict:
    """The record of a pool over these texts as the result file holds it: its id range and its four counts.

    Worked out here from the FAKE tokenizer and the FAKE model's ids. A pool keeps every id the model counts,
    also one outside the token range of the prompts.
    """
    low, high = 0, FAKE_MODEL_VOCAB_SIZE
    counts = {
        "id_range": [low, high],
        "conversations": 0,
        "tokens": 0,
        "dropped_tokens": 0,
        "dropped_conversations": 0,
    }
    for text in texts:
        ids = [_FakeTokenizer.fake_id(word) for word in text.split()]
        kept = [token for token in ids if low <= token < high]
        counts["dropped_tokens"] += len(ids) - len(kept)
        counts["tokens"] += len(kept)
        counts["conversations" if kept else "dropped_conversations"] += 1
    return counts


def _tep_cell(mode: str, tp_size: int = 2):
    """A cell of the tensor- and expert-parallel layout: ``tp_size`` GPUs, one expert group per GPU."""
    cell = _cell(phase=mode, dp=1, strategy="tep")
    topology = ParallelTopology(tp=tp_size, pp=1, dp=1, moe_tp=1, moe_ep=tp_size, cp=1)
    return dataclasses.replace(cell, topology=topology)


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
    assert launcher.content is content


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
        # A model without expert layers: no expert key, and no layout of expert layers to record.
        "expert_keys": {},
        "moe": None,
    }
    assert set(producer) == {*LAUNCHER_PRODUCER_KEYS, *DRIVER_PRODUCER_KEYS, "name", "artifact_format"}
    # A run with seeded ids writes the content block it always wrote.
    assert set(producer["content"]) == {"source", "seed", "token_range", "issued"}
    assert (producer["content"]["source"], producer["content"]["seed"]) == ("seeded_token_ids", "0")
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
        # A model with expert layers is refused when seeded ids are asked for, right after its config was read.
        world.experts = {"num_experts": 8}
        world.hooks["model_expert_keys"] = lambda: shutil.rmtree(pod_dir)

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare, env={"DYN_BENCH_PREFILL_CONTENT": "random"})

    assert launch.code == 64
    lines = _stderr_lines(capsys)
    assert lines[-2] == f"[sglbench] FAILED: rejected: {_SEEDED_WITH_EXPERTS.format('num_experts=8')}"
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
    # No content word: the model decides. No dataset variable: the defaults of a text run apply.
    assert (options.prefill_content, options.dataset_environ) == (None, {})
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
    assert len(set(tested)) == len(tested) == 46
    # The fields the launcher reads besides: the platform, the three run settings, the recorded field, the
    # expert-parallel size, which is tested against the TP size and the model, and the runner of the expert
    # layers, which is recorded for a model that has them.
    assert set(FAKE_RESOLVED_ARGUMENTS) - set(tested) == {
        "device",
        "tp_size",
        "prefill_max_requests",
        "max_queued_requests",
        "enable_nvidia_megakernel",
        "ep_size",
        "moe_runner_backend",
    }


@pytest.mark.parametrize("field", [name for name, _value, _refusal in STAGE_S_REFUSALS])
def test_a_field_sglang_does_not_have_is_an_error_and_not_a_pass(field):
    fields = dict(FAKE_RESOLVED_ARGUMENTS)
    del fields[field]

    with pytest.raises(AttributeError, match=f"SGLang's resolved arguments have no field '{field}'"):
        launcher._check_resolved(_FakeView(fields))


_REFUSED_AT_LAUNCH = (
    "attn_dp_size",
    "moe_a2a_backend",
    "enable_eplb",
    "disable_radix_cache",
    "incremental_streaming_output",
)


@pytest.mark.parametrize(
    ("field", "value", "refusal"), [row for row in STAGE_S_REFUSALS if row[0] in _REFUSED_AT_LAUNCH]
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


@pytest.mark.parametrize("missing", ["tp_size", "ep_size", "prefill_max_requests", "max_queued_requests", "device"])
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


EXPERT_KEY_SETS = [
    ({"num_local_experts": 8}, "num_local_experts=8"),
    ({"n_routed_experts": 256, "num_experts_per_tok": 8}, "n_routed_experts=256"),
    ({"moe_intermediate_size": 1408}, "moe_intermediate_size=1408"),
]


@pytest.mark.parametrize(("experts", "shown"), EXPERT_KEY_SETS)
def test_seeded_ids_are_refused_for_a_model_with_expert_layers(tmp_path, monkeypatch, experts, shown):
    def prepare(world, _pod_dir):
        world.experts = experts

    launch = _launch(tmp_path, monkeypatch, "decode", prepare=prepare, env={"DYN_BENCH_PREFILL_CONTENT": "random"})

    on_disk = _assert_failed_before_the_grid(launch, 64, f"rejected: {_SEEDED_WITH_EXPERTS.format(shown)}")
    assert launch.world.events[-3:] == ["model_expert_keys", "write", "sleep"]
    # The keys were recorded before the refusal.
    assert on_disk["producer"]["expert_keys"] == experts


@pytest.mark.parametrize(("experts", "_shown"), EXPERT_KEY_SETS)
def test_a_model_with_expert_layers_runs_with_dataset_text(tmp_path, monkeypatch, experts, _shown):
    def prepare(world, _pod_dir):
        world.experts = experts

    # The variable is unset: the expert keys alone make the run a text run.
    launch = _launch_text(tmp_path, monkeypatch, "decode", prepare=prepare, word=None)

    on_disk = _assert_accepted(launch)
    producer = on_disk["producer"]
    assert producer["content"]["source"] == "dataset_text_chains"
    assert producer["content"]["dataset"] == launch.dataset
    assert producer["expert_keys"] == experts
    # One GPU: every rank holds every expert, and SGLang's defaults for the expert layers are recorded.
    assert producer["moe"] == {
        "ep_size": 1,
        "moe_a2a_backend": "none",
        "moe_runner_backend": "auto",
        "moe_dense_tp_size": None,
        "enable_eplb": False,
    }
    events = launch.world.events
    assert events.index("model_expert_keys") < events.index("build_engine") < events.index("encode_texts")


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

    # The failing launch asks for seeded ids with a model that has expert layers, which is refused.
    env = {"DYN_BENCH_PREFILL_CONTENT": "random"} if failing else None
    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare, pod=(cell, pod_dir), env=env)

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
        "The benchmark flags, their defaults, their environment variables and the dataset of a text run are "
        "re-implemented from",
        "ai-dynamo/dynamo (Apache-2.0, https://github.com/ai-dynamo/dynamo), "
        "components/src/dynamo/vllm/backend_args.py and",
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
        "http.client",
        "json",
        "math",
        "os",
        "secrets",
        "sglbench_artifact",
        "sglbench_compat",
        "sglbench_content",
        "sglbench_driver",
        "sglbench_grid",
        "sglbench_recorder",
        "signal",
        "sys",
        "time",
        "traceback",
        "urllib.request",
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


# ---------------------------------------------------------------------------------------------------------------
# Text runs: the content word
# ---------------------------------------------------------------------------------------------------------------

# Whether a run is a text run: (the content variable, the expert keys of the model, text run).
CONTENT_TABLE = [
    (None, {}, False),
    ("", {}, False),
    ("random", {}, False),
    ("sharegpt_chain", {}, True),
    (None, {"num_experts": 8}, True),
    ("", {"num_experts": 8}, True),
    ("sharegpt_chain", {"num_experts": 8}, True),
]


@pytest.mark.parametrize(("word", "experts", "text"), CONTENT_TABLE)
def test_the_content_word_and_the_expert_keys_decide_whether_a_run_is_a_text_run(word, experts, text):
    variables = {} if word is None else {ENV_CONTENT: word}
    options = _options([], variables)
    assert options.rejection is None
    # Unset and empty are the same: the model decides.
    assert options.prefill_content == (word or None)
    assert launcher._is_text_run(options.prefill_content, experts) is text


def test_seeded_ids_with_expert_keys_are_refused_and_an_unknown_word_is_not_kept():
    with pytest.raises(launcher._Failed) as raised:
        launcher._is_text_run("random", {"num_experts_per_tok": 4, "num_experts": 60})
    assert raised.value.reason == f"rejected: {_SEEDED_WITH_EXPERTS.format('num_experts_per_tok=4')}"
    # A word that is refused is not kept as the choice, so nothing later takes it for one.
    refused = _options([], {ENV_CONTENT: "sharegpt"})
    assert (refused.rejection, refused.prefill_content) == (
        f"DYN_BENCH_PREFILL_CONTENT='sharegpt' {_CONTENT_WORDS}",
        None,
    )
    assert (launcher.SEEDED_CONTENT, launcher.TEXT_CONTENT) == ("random", "sharegpt_chain")


@pytest.mark.parametrize(("word", "experts", "text"), CONTENT_TABLE)
def test_a_launch_sends_what_the_content_word_and_the_model_ask_for(tmp_path, monkeypatch, word, experts, text):
    def prepare(world, _pod_dir):
        world.experts = experts

    if text:
        launch = _launch_text(tmp_path, monkeypatch, "prefill", prepare=prepare, word=word)
    else:
        # A run with seeded ids does not look at the dataset variables: these name nothing that exists.
        env = {
            ENV_DATASET: str(tmp_path / "FAKE-no-such-file.json"),
            ENV_SHA256: "0" * 64,
            ENV_CACHE_DIR: "/FAKE/no/cache",
        }
        launch = _launch(
            tmp_path,
            monkeypatch,
            "prefill",
            prepare=prepare,
            env={**env, **({} if word is None else {ENV_CONTENT: word})},
        )

    on_disk = _assert_accepted(launch)
    content_block = on_disk["producer"]["content"]
    assert content_block["source"] == ("dataset_text_chains" if text else "seeded_token_ids")
    assert set(content_block) == {"source", "seed", "token_range", "issued", *(("dataset", "pool") if text else ())}
    assert ("encode_texts" in launch.world.events) is text and ("tokenizer_class" in launch.world.events) is text
    assert on_disk["producer"]["expert_keys"] == experts
    assert (on_disk["producer"]["moe"] is None) is (not experts)
    # The config block, which AISim compares between two launches of a cell, is the same for both sources.
    assert "content_seed" in on_disk["config"] and "dataset" not in json.dumps(on_disk["config"])


def test_the_options_keep_the_dataset_variables_as_text_and_take_an_empty_one_for_unset():
    variables = {
        ENV_DATASET: "/FAKE/datasets/conversations.json",
        ENV_CACHE_DIR: "",
        ENV_SHA256: "FAKE-digest",
        "HF_HOME": "/FAKE/models/hf-home",
        ENV_CONTENT: "sharegpt_chain",
    }
    options = _options([], variables)

    assert options.rejection is None
    assert options.dataset_environ == {
        ENV_DATASET: "/FAKE/datasets/conversations.json",
        ENV_SHA256: "FAKE-digest",
        "HF_HOME": "/FAKE/models/hf-home",
    }
    assert _options([], {ENV_DATASET: "", ENV_CACHE_DIR: "", ENV_SHA256: "", "HF_HOME": ""}).dataset_environ == {}
    assert (launcher.ENV_KVWARM_DATASET, launcher.ENV_KVWARM_CACHE_DIR, launcher.ENV_KVWARM_SHA256) == (
        ENV_DATASET,
        ENV_CACHE_DIR,
        ENV_SHA256,
    )
    assert launcher.ENV_PREFILL_CONTENT == ENV_CONTENT and launcher.ENV_HF_HOME == "HF_HOME"


# ---------------------------------------------------------------------------------------------------------------
# Text runs: finding the dataset file
# ---------------------------------------------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unavailable(environ: dict) -> str:
    """The dataset cannot be had: the text after ``dataset_unavailable: ``, which is one line."""
    with pytest.raises(launcher._Failed) as raised:
        launcher._resolve_dataset(environ)
    reason = raised.value.reason
    assert reason.startswith("dataset_unavailable: ") and "\n" not in reason
    return reason.removeprefix("dataset_unavailable: ")


def _fake_download(monkeypatch, data: bytes = b"[]", **options) -> _FakeUrlopen:
    """Put a FAKE ``urlopen`` in place that serves ``data``. Nothing is fetched."""
    fake = _FakeUrlopen(data, **options)
    monkeypatch.setattr(launcher.urllib.request, "urlopen", fake)
    return fake


# The random part in the name of a part file, made FAKE and fixed where a test has to name the file.
FAKE_PART_TOKEN = "FAKE0123456789ab"


def _fake_part_name(monkeypatch, cached: object) -> str:
    """Fix the random part of the next part files and return the name of the part file of ``cached``."""
    monkeypatch.setattr(launcher.secrets, "token_hex", lambda _bytes: FAKE_PART_TOKEN)
    return f"{cached}.{os.getpid()}.{FAKE_PART_TOKEN}.part"


def test_a_local_dataset_file_is_hashed_and_compared_with_the_hash_variable(tmp_path):
    path, digest = _write_fake_dataset(tmp_path)
    expected = {"path": str(path), "sha256": digest, "url": None, "downloaded": False}

    # No pinned digest applies to a local file: without the hash variable any content is taken.
    assert launcher._resolve_dataset({ENV_DATASET: str(path)}) == expected
    assert launcher._resolve_dataset({ENV_DATASET: str(path), ENV_SHA256: digest}) == expected
    # The cache directory and HF_HOME mean nothing for a local file, and neither directory is made.
    elsewhere = {ENV_CACHE_DIR: str(tmp_path / "cache"), "HF_HOME": str(tmp_path / "models" / "hf-home")}
    assert launcher._resolve_dataset({ENV_DATASET: str(path), **elsewhere}) == expected
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [FAKE_DATASET_NAME]
    # Dynamo does not compare a local file; here another digest fails the run, and the file is left alone.
    wrong = "0" * 64
    reason = _unavailable({ENV_DATASET: str(path), ENV_SHA256: wrong})
    assert reason == f"the dataset file {path} has the SHA-256 {digest}, expected {wrong}"
    assert _sha256(path.read_bytes()) == digest


def test_a_local_path_that_is_not_an_existing_file_is_unavailable(tmp_path):
    missing = tmp_path / "FAKE-none.json"
    assert _unavailable({ENV_DATASET: str(missing)}) == f"the dataset path {missing} is not an existing file"
    assert _unavailable({ENV_DATASET: str(tmp_path)}) == f"the dataset path {tmp_path} is not an existing file"
    # Only http and https are fetched: any other word is a path.
    other = "ftp://datasets.fake.invalid/FAKE.json"
    assert _unavailable({ENV_DATASET: other}) == f"the dataset path {other} is not an existing file"


def test_a_url_is_downloaded_once_into_the_cache_directory(tmp_path, monkeypatch):
    data = json.dumps(_fake_dataset_records(12)).encode()
    cache = tmp_path / "cache" / "deeper"
    seen = {}
    fake = _fake_download(monkeypatch, data, on_read=lambda: seen.setdefault("during", sorted(os.listdir(cache))))
    environ = {ENV_DATASET: FAKE_DATASET_URL, ENV_CACHE_DIR: str(cache)}
    part = _fake_part_name(monkeypatch, FAKE_DATASET_NAME)

    first = launcher._resolve_dataset(environ)

    cached = cache / FAKE_DATASET_NAME
    assert first == {"path": str(cached), "sha256": _sha256(data), "url": FAKE_DATASET_URL, "downloaded": True}
    assert cached.read_bytes() == data
    # One call with the address and the socket timeout, and nothing else.
    assert fake.calls == [((FAKE_DATASET_URL,), {"timeout": 60})]
    assert launcher.DATASET_DOWNLOAD_TIMEOUT_SECONDS == 60
    # The bytes went into a part file named after this process and a random part, and were renamed into place.
    assert seen["during"] == [part]
    assert sorted(os.listdir(cache)) == [FAKE_DATASET_NAME]

    # The file is there now: it is hashed again and not fetched again.
    second = launcher._resolve_dataset(environ)
    assert second == {**first, "downloaded": False} and len(fake.calls) == 1
    # A dataset of one's own has no pinned digest: without the hash variable the cached file is taken as it is.
    cached.write_bytes(b"[]")
    assert launcher._resolve_dataset(environ) == {**second, "sha256": _sha256(b"[]")}
    assert len(fake.calls) == 1


def test_the_hash_variable_is_compared_with_a_download_and_with_a_cached_file(tmp_path, monkeypatch):
    data = json.dumps(_fake_dataset_records(12)).encode()
    digest, wrong = _sha256(data), "f" * 64
    cache = tmp_path / "cache"
    fake = _fake_download(monkeypatch, data)
    environ = {ENV_DATASET: FAKE_DATASET_URL, ENV_CACHE_DIR: str(cache)}
    cached = cache / FAKE_DATASET_NAME
    part = _fake_part_name(monkeypatch, cached)

    # A download with another digest is deleted and nothing is cached.
    reason = _unavailable({**environ, ENV_SHA256: wrong})
    assert reason == f"the downloaded dataset file {part} has the SHA-256 {digest}, expected {wrong}"
    assert os.listdir(cache) == [] and len(fake.calls) == 1
    # With its own digest the same download is kept.
    assert launcher._resolve_dataset({**environ, ENV_SHA256: digest})["downloaded"] is True
    assert os.listdir(cache) == [FAKE_DATASET_NAME] and len(fake.calls) == 2
    # A cached file with another digest fails the run. It is neither deleted nor fetched again.
    reason = _unavailable({**environ, ENV_SHA256: wrong})
    assert reason == f"the cached dataset file {cached} has the SHA-256 {digest}, expected {wrong}"
    assert cached.read_bytes() == data and len(fake.calls) == 2


def test_the_default_dataset_is_dynamos_address_with_its_pinned_digest(tmp_path, monkeypatch):
    # The two constants of Dynamo's vLLM self-benchmark, typed here on their own.
    url = (
        "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/"
        "resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    )
    pinned = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
    name = "ShareGPT_V3_unfiltered_cleaned_split.json"
    assert (url, pinned) == (launcher.DEFAULT_DATASET_URL, launcher.DEFAULT_DATASET_SHA256)

    data = b'[{"FAKE": "these bytes are not the default dataset"}]'
    cache = tmp_path / "cache"
    fake = _fake_download(monkeypatch, data)
    part = _fake_part_name(monkeypatch, cache / name)
    # No dataset variable, or an empty one: the default address. The FAKE bytes do not have the pinned digest,
    # so the download is refused and deleted.
    for environ in ({ENV_CACHE_DIR: str(cache)}, {ENV_DATASET: "", ENV_SHA256: "", ENV_CACHE_DIR: str(cache)}):
        reason = _unavailable(environ)
        assert reason == f"the downloaded dataset file {part} has the SHA-256 {_sha256(data)}, expected {pinned}"
    assert [call[0] for call in fake.calls] == [(url,), (url,)] and os.listdir(cache) == []

    # FAKE: with the digest of the FAKE bytes in the place of the pinned one, the default is fetched and kept.
    monkeypatch.setattr(launcher, "DEFAULT_DATASET_SHA256", _sha256(data))
    record = launcher._resolve_dataset({ENV_CACHE_DIR: str(cache)})
    assert record == {"path": str(cache / name), "sha256": _sha256(data), "url": url, "downloaded": True}
    # A cached default with other content fails the run, and the hash variable decides over the pinned digest.
    (cache / name).write_bytes(b"[]")
    reason = _unavailable({ENV_CACHE_DIR: str(cache)})
    assert (
        reason == f"the cached dataset file {cache / name} has the SHA-256 {_sha256(b'[]')}, expected {_sha256(data)}"
    )
    assert (cache / name).read_bytes() == b"[]" and len(fake.calls) == 3
    assert launcher._resolve_dataset({ENV_CACHE_DIR: str(cache), ENV_SHA256: _sha256(b"[]")})["downloaded"] is False


@pytest.mark.parametrize(
    ("url", "name"),
    [
        (FAKE_DATASET_URL, FAKE_DATASET_NAME),
        ("http://datasets.fake.invalid/a/b/FAKE.json", "FAKE.json"),
        ("https://datasets.fake.invalid/FAKE.json?revision=main&download=true", "FAKE.json"),
        # An address that ends in a slash names no file: Dynamo's stand-in name.
        ("https://datasets.fake.invalid/FAKE/", "kvwarm_dataset.json"),
        ("https://datasets.fake.invalid/FAKE/?download=true", "kvwarm_dataset.json"),
    ],
)
def test_the_cached_file_is_named_after_the_last_part_of_the_address(tmp_path, monkeypatch, url, name):
    fake = _fake_download(monkeypatch)

    record = launcher._resolve_dataset({ENV_DATASET: url, ENV_CACHE_DIR: str(tmp_path)})

    assert record == {"path": str(tmp_path / name), "sha256": _sha256(b"[]"), "url": url, "downloaded": True}
    assert fake.calls == [((url,), {"timeout": 60})]


def test_the_cache_directory_is_the_variable_then_beside_hf_home_then_the_default(tmp_path, monkeypatch):
    _fake_download(monkeypatch)
    url = {ENV_DATASET: FAKE_DATASET_URL}
    monkeypatch.chdir(tmp_path)

    # The variable decides, also over HF_HOME. It is made absolute and created with its parents.
    hf_home = tmp_path / "models" / "hf-home"
    record = launcher._resolve_dataset({**url, ENV_CACHE_DIR: "relative/cache", "HF_HOME": str(hf_home)})
    assert record["path"] == str(tmp_path / "relative" / "cache" / FAKE_DATASET_NAME)
    assert not (tmp_path / "models").exists()
    # Without it: ``fpm_datasets`` beside the directory HF_HOME names, which itself is not made.
    record = launcher._resolve_dataset({**url, ENV_CACHE_DIR: "", "HF_HOME": str(hf_home)})
    assert record["path"] == str(tmp_path / "models" / "fpm_datasets" / FAKE_DATASET_NAME)
    assert sorted(entry.name for entry in (tmp_path / "models").iterdir()) == ["fpm_datasets"]
    # With neither: the default directory, which a fixture of this file points into the temporary directory.
    record = launcher._resolve_dataset(url)
    assert record["path"] == str(tmp_path / "FAKE-default-cache" / FAKE_DATASET_NAME)
    assert _resolve_through_options(url)["path"] == record["path"]
    # The default of the module itself is Dynamo's, read here from the source because the fixture replaced it.
    assert '\nDEFAULT_DATASET_CACHE_DIR = "/tmp/fpm_datasets"\n' in MODULE_PATH.read_text(encoding="utf-8")


def _resolve_through_options(variables: dict) -> dict:
    """Resolve the dataset from the variables as ``main`` does: through the options."""
    return launcher._resolve_dataset(_options([], variables).dataset_environ)


FAKE_DOWNLOAD_ERRORS = [
    (urllib.error.URLError("FAKE: no route to host"), "URLError: <urlopen error FAKE: no route to host>"),
    (TimeoutError("FAKE: the read timed out"), "TimeoutError: FAKE: the read timed out"),
    (ConnectionResetError("FAKE: reset by peer"), "ConnectionResetError: FAKE: reset by peer"),
    (
        PermissionError("FAKE: the part file\ncannot be written"),
        "PermissionError: FAKE: the part file cannot be written",
    ),
    (ValueError("FAKE: unknown url type"), "ValueError: FAKE: unknown url type"),
    (
        http.client.RemoteDisconnected("FAKE: closed without response"),
        "RemoteDisconnected: FAKE: closed without response",
    ),
    (http.client.IncompleteRead(b"FAKE"), "IncompleteRead: IncompleteRead(4 bytes read)"),
]


@pytest.mark.parametrize(
    ("error", "shown"), FAKE_DOWNLOAD_ERRORS, ids=[row[1].split(":")[0] for row in FAKE_DOWNLOAD_ERRORS]
)
@pytest.mark.parametrize("when", ["at the start", "in the transfer"])
def test_a_download_that_fails_is_unavailable_and_leaves_no_part_file(tmp_path, monkeypatch, error, shown, when):
    data = json.dumps(_fake_dataset_records(12)).encode()
    if when == "at the start":
        fake = _fake_download(monkeypatch, data, error=error)
    else:
        # FAKE: every byte arrives, and then the transfer breaks off.
        fake = _fake_download(monkeypatch, data, fail_after=error)

    reason = _unavailable({ENV_DATASET: FAKE_DATASET_URL, ENV_CACHE_DIR: str(tmp_path)})

    assert reason == shown
    # Nothing is tried a second time, and neither a part file nor a cached file is left.
    assert len(fake.calls) == 1 and os.listdir(tmp_path) == []


def test_an_http_error_of_the_download_is_unavailable(tmp_path, monkeypatch):
    error = urllib.error.HTTPError(FAKE_DATASET_URL, 404, "FAKE: Not Found", None, None)
    _fake_download(monkeypatch, error=error)

    reason = _unavailable({ENV_DATASET: FAKE_DATASET_URL, ENV_CACHE_DIR: str(tmp_path)})

    assert reason == "HTTPError: HTTP Error 404: FAKE: Not Found" and os.listdir(tmp_path) == []


def test_a_stop_during_a_download_passes_and_leaves_no_part_file(tmp_path, monkeypatch):
    _fake_download(monkeypatch, b"FAKE bytes", fail_after=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        launcher._resolve_dataset({ENV_DATASET: FAKE_DATASET_URL, ENV_CACHE_DIR: str(tmp_path)})

    assert os.listdir(tmp_path) == []


def test_two_launchers_with_one_process_id_do_not_share_a_part_file(tmp_path, monkeypatch):
    # FAKE: launchers in two pods that share the cache directory, both running as process 7. The second one
    # starts and finishes while the first is in its third piece.
    data = json.dumps(_fake_dataset_records(40)).encode()
    monkeypatch.setattr(launcher, "_FILE_PIECE_BYTES", 512)
    monkeypatch.setattr(launcher.os, "getpid", lambda: 7)
    environ = {ENV_DATASET: FAKE_DATASET_URL, ENV_CACHE_DIR: str(tmp_path), ENV_SHA256: _sha256(data)}
    seen: dict = {"reads": 0}

    def on_read():
        seen["reads"] += 1
        if seen["reads"] == 3:
            seen["parts"] = sorted(os.listdir(tmp_path))
            monkeypatch.setattr(launcher.urllib.request, "urlopen", _FakeUrlopen(data))
            seen["second"] = launcher._resolve_dataset(environ)

    monkeypatch.setattr(launcher.urllib.request, "urlopen", _FakeUrlopen(data, on_read=on_read))
    first = launcher._resolve_dataset(environ)

    cached = tmp_path / FAKE_DATASET_NAME
    expected = {"path": str(cached), "sha256": _sha256(data), "url": FAKE_DATASET_URL, "downloaded": True}
    assert (first, seen["second"]) == (expected, expected)
    assert os.listdir(tmp_path) == [FAKE_DATASET_NAME] and cached.read_bytes() == data
    # The part file of the first launcher: the process id and sixteen random hex digits.
    (part,) = seen["parts"]
    assert fnmatch.fnmatch(part, f"{FAKE_DATASET_NAME}.7.{'[0-9a-f]' * 16}.part")


def test_a_download_that_ends_before_the_announced_length_is_unavailable(tmp_path, monkeypatch):
    data = json.dumps(_fake_dataset_records(12)).encode()
    sent = len(data)

    def answer_announcing(length: str):
        # FAKE response with a ``Content-Length`` header; it serves every byte of ``data`` and then ends
        # without an error, as ``http.client`` ends a read with a size.
        response = _FakeResponse(data)
        response.headers = {"Content-Length": length}
        monkeypatch.setattr(launcher.urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    environ = {ENV_DATASET: FAKE_DATASET_URL, ENV_CACHE_DIR: str(tmp_path)}
    answer_announcing(f" {sent + 100} ")
    reason = _unavailable(environ)
    assert reason == f"OSError: the transfer ended after {sent} of the {sent + 100} bytes the server announced"
    # Nothing is cached, so a later run fetches the file again and does not find a part of it.
    assert os.listdir(tmp_path) == []

    answer_announcing(str(sent))
    assert launcher._resolve_dataset(environ)["downloaded"] is True
    assert (tmp_path / FAKE_DATASET_NAME).read_bytes() == data


def test_a_cache_directory_or_a_cached_file_that_cannot_be_used_is_unavailable(tmp_path, monkeypatch):
    fake = _fake_download(monkeypatch)
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory", encoding="utf-8")

    # A file in the place of the cache directory, or of one of its parents.
    for cache in (blocker, blocker / "cache"):
        reason = _unavailable({ENV_DATASET: FAKE_DATASET_URL, ENV_CACHE_DIR: str(cache)})
        assert reason.split(":")[0] in ("FileExistsError", "NotADirectoryError")
    # A directory in the place of the cached file.
    (tmp_path / "cache" / FAKE_DATASET_NAME).mkdir(parents=True)
    reason = _unavailable({ENV_DATASET: FAKE_DATASET_URL, ENV_CACHE_DIR: str(tmp_path / "cache")})
    assert reason.split(":")[0] in ("IsADirectoryError", "PermissionError")
    assert fake.calls == [] and blocker.read_text(encoding="utf-8") == "not a directory"


# ---------------------------------------------------------------------------------------------------------------
# Text runs: the texts of the dataset
# ---------------------------------------------------------------------------------------------------------------


def test_the_dataset_record_and_its_texts(tmp_path):
    records = _fake_dataset_records()
    path, digest = _write_fake_dataset(tmp_path, records)

    record, texts = launcher._load_dataset({ENV_DATASET: str(path)})

    kept = _fake_kept_texts(records)
    # The recipe keeps about half of the long texts, in the order of the records, and drops the three others.
    assert texts == kept and 50 < len(kept) < 110 and len(records) == 162
    assert record == {
        "path": str(path),
        "sha256": digest,
        "collection_pool": "conversation_sha256_even",
        "conversations": len(kept),
        "url": None,
        "downloaded": False,
    }
    # Dynamo's four keys first, in its order, then the two of this producer.
    assert list(record) == ["path", "sha256", "collection_pool", "conversations", "url", "downloaded"]
    assert content.COLLECTION_POOL == "conversation_sha256_even"


@pytest.mark.parametrize(
    ("data", "shown"),
    [
        (b"FAKE: this is not JSON", "JSONDecodeError: Expecting value: line 1 column 1 (char 0)"),
        (b"", "JSONDecodeError: Expecting value: line 1 column 1 (char 0)"),
        (b"\xff\xfe[]", "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte"),
        (b'{"records": []}', "ValueError: the dataset must be a list of records, got dict"),
        (b'[{"conversations": []}, 7]', "ValueError: record 1 of the dataset must be a mapping, got int"),
        (
            b'[{"conversations": ["a turn"]}]',
            "ValueError: record 0 of the dataset: turn 0 must be a mapping, got str",
        ),
    ],
)
def test_a_dataset_file_that_cannot_be_parsed_is_unavailable(tmp_path, data, shown):
    path = tmp_path / FAKE_DATASET_NAME
    path.write_bytes(data)

    with pytest.raises(launcher._Failed) as raised:
        launcher._load_dataset({ENV_DATASET: str(path)})

    assert raised.value.reason == f"dataset_unavailable: {path}: {shown}"


@pytest.mark.parametrize(
    "records",
    [
        [],
        [{"id": "FAKE", "conversations": [{"from": "human", "value": "FAKE: too short"}]}] * 5,
        [{"id": "FAKE"}, {"id": "FAKE", "conversations": None}, {"id": "FAKE", "conversations": [{"from": "gpt"}]}],
    ],
)
def test_a_dataset_without_a_kept_text_is_empty(tmp_path, capsys, records):
    path, _digest = _write_fake_dataset(tmp_path, records)

    with pytest.raises(launcher._Failed) as raised:
        launcher._load_dataset({ENV_DATASET: str(path)})

    # The reason is the bare word, as in Dynamo; the line before it says which file gave nothing.
    assert raised.value.reason == "dataset_empty"
    assert _stderr_lines(capsys) == [
        f"[sglbench] the dataset {path} gives no text: no record has a text that the recipe keeps"
    ]


# ---------------------------------------------------------------------------------------------------------------
# Text runs: the text pool and the run
# ---------------------------------------------------------------------------------------------------------------


def _assert_free_calls_are_text(world: _FakeWorld, text_ids: set[int]) -> None:
    """The seven calls before the first point are text after their allocated tokens.

    Every prompt starts with an allocated token. The shots of the aligned-prefix check, the third call, also
    have an allocated token after their cached unit.
    """
    unit = grid.cached_unit(world.limits.page_size)
    for number, call in enumerate(world.calls[:7]):
        assert call.key == (0, "free", number)
        allocated = {0, unit} if number == 2 else {0}
        for prompt in call.prompts:
            assert all(token in text_ids for position, token in enumerate(prompt) if position not in allocated)


def _longest_prompt(built: grid.Grid, page_size: int = 1) -> int:
    """The longest prompt of a run with a warm-up, worked out here: the points, the self-test, the warm-up."""
    lengths = [2 * grid.cached_unit(page_size) + 7, 256]
    for point in built.points:
        lengths.append(point.cached_tokens + point.new_tokens if point.phase == "prefill" else point.cached_tokens - 3)
    return max(lengths)


@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_a_text_run_of_a_model_without_expert_layers_passes_the_validator_and_the_gate(
    tmp_path, monkeypatch, capsys, mode, page_size
):
    launch = _launch_text(tmp_path, monkeypatch, mode, page_size=page_size)

    on_disk = _assert_accepted(launch)
    world = launch.world
    kept = _fake_kept_texts(_fake_dataset_records())
    # The dataset was read before the engine was built, and the pool was made after the token range was taken:
    # one tokenizer call, because the FAKE dataset has fewer texts than one batch holds.
    assert world.events == [
        "clock",
        "sglang_version",
        "load_plugins",
        "parse_server_args",
        "resolve",
        "platform_facts",
        "model_expert_keys",
        "build_engine",
        "vocabulary",
        "model_vocabulary_size",
        "tokenizer_class",
        "encode_texts",
        "tokenizer",
        "shutdown",
        "write",
        "sleep",
    ]
    assert world.tokenizer_batches == [kept] and len(kept) < launcher.POOL_ENCODE_BATCH == 256
    assert launcher.POOL_TOKEN_TARGET == 8_000_000

    # The producer block: the source, the record of the dataset and the record of the pool, which is its id
    # range and its four counts.
    producer = on_disk["producer"]
    counts = _fake_pool_counts(kept)
    assert producer["content"] == {
        "source": "dataset_text_chains",
        "seed": "0",
        "token_range": FAKE_TOKEN_RANGE,
        "issued": producer["content"]["issued"],
        "dataset": launch.dataset,
        "pool": counts,
    }
    # Some FAKE texts hold the written form of a FAKE token the model does not count. Those ids are dropped
    # and no other: the ids of the FAKE special token, which lie outside the token range, are text.
    beyond = sum(text.split().count("<FAKE-beyond>") for text in kept)
    special = sum(text.split().count("<FAKE-special>") for text in kept)
    assert counts["dropped_tokens"] == beyond > 0 and special > 0 and counts["dropped_conversations"] == 0
    assert counts["tokens"] == sum(len(text.split()) for text in kept) - beyond
    assert (producer["expert_keys"], producer["moe"]) == ({}, None)
    assert set(producer) == {*LAUNCHER_PRODUCER_KEYS, *DRIVER_PRODUCER_KEYS, "name", "artifact_format"}
    # The ``kvwarm`` block of a decode artifact names the dataset with Dynamo's four keys.
    if mode == "decode":
        assert on_disk["kvwarm"] == {
            "enabled": True,
            "warm_eligible": True,
            "skip_reason": None,
            "method": "radix_cache_seed",
            "dataset": {key: launch.dataset[key] for key in ("path", "sha256", "collection_pool", "conversations")},
        }
    else:
        assert "kvwarm" not in on_disk
    # The config block is the one of a run with seeded ids: it holds nothing of the dataset.
    assert list(on_disk["config"]) == [
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

    # Every prompt is made of ids the FAKE tokenizer gave for words of the dataset, apart from the allocated
    # tokens: the first token of a prompt and the first new token of a shot with cached tokens.
    text_ids = {_FakeTokenizer.fake_id(word) for text in kept for word in text.split()} - {FAKE_MODEL_VOCAB_SIZE}
    assert 2 in text_ids and not FAKE_TOKEN_RANGE[0] <= 2 < FAKE_TOKEN_RANGE[1]
    built = launch.planned()
    _assert_free_calls_are_text(world, text_ids)
    for point in built.points:
        for call in launch.calls_of(point.benchmark_id):
            skipped = {0, point.cached_tokens} if call.key[1] in ("warm", "meas") else {0}
            for prompt in call.prompts:
                assert all(token in text_ids for position, token in enumerate(prompt) if position not in skipped)
                # The allocated tokens come from the token range. A token of the text can lie outside it.
                assert all(FAKE_TOKEN_RANGE[0] <= prompt[position] < FAKE_TOKEN_RANGE[1] for position in skipped)
    # No id the model does not count reached the engine, and the id of the FAKE special token did.
    sent = {token for call in world.calls for prompt in call.prompts for token in prompt}
    assert max(sent) < FAKE_MODEL_VOCAB_SIZE and 2 in sent

    lines = _stderr_lines(capsys)
    dataset = launch.dataset
    assert (
        f"[sglbench] text run: the dataset {dataset['path']} (SHA-256 {dataset['sha256']}) gives {len(kept)} texts"
    ) in lines
    (pool_line,) = [line for line in lines if line.startswith("[sglbench] the text pool holds ")]
    assert pool_line.startswith(
        f"[sglbench] the text pool holds {counts['tokens']} tokens in {counts['conversations']} conversations: "
        f"{len(kept)} of {len(kept)} texts went through fake_tokenizers.FakeTokenizer in "
    )
    assert pool_line.endswith(
        f" s, and {counts['dropped_tokens']} token ids at or above {FAKE_MODEL_VOCAB_SIZE}, the number of ids the "
        "model counts, and 0 conversations without an id below it were dropped"
    )
    built_line = (
        "[sglbench] the engine is built; the first token of each prompt part comes from the token ids [3, 50000), "
        "the other tokens are dataset text"
    )
    assert lines.index(pool_line) > lines.index(built_line)
    # Only a run with seeded ids says that its prompts use the token range.
    assert not any("prompts use the token ids" in line for line in lines)


def test_the_pool_is_filled_batch_by_batch_in_dataset_order_until_it_holds_its_target(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "POOL_ENCODE_BATCH", 8)
    monkeypatch.setattr(launcher, "POOL_TOKEN_TARGET", 1500)

    launch = _launch_text(tmp_path, monkeypatch, "decode")

    on_disk = _assert_accepted(launch)
    kept = _fake_kept_texts(_fake_dataset_records())
    batches = launch.world.tokenizer_batches
    # Batches of eight texts in the order of the dataset, and no text after the batch that reached the target.
    assert all(len(batch) == 8 for batch in batches) and 2 < len(batches) < len(kept) // 8
    encoded = [text for batch in batches for text in batch]
    assert encoded == kept[: len(encoded)]
    counts = _fake_pool_counts(encoded)
    assert on_disk["producer"]["content"]["pool"] == counts
    assert _fake_pool_counts(encoded[:-8])["tokens"] < 1500 <= counts["tokens"]
    # The record of the dataset still counts every kept text of the file, not only those in the pool.
    assert on_disk["producer"]["content"]["dataset"]["conversations"] == len(kept) > len(encoded)


def test_the_pool_stops_with_the_batch_that_brings_it_to_its_target_exactly(tmp_path, monkeypatch):
    kept = _fake_kept_texts(_fake_dataset_records())
    # FAKE target: exactly the tokens of the first twenty texts, taken one text in a call.
    target = _fake_pool_counts(kept[:20])["tokens"]
    monkeypatch.setattr(launcher, "POOL_ENCODE_BATCH", 1)
    monkeypatch.setattr(launcher, "POOL_TOKEN_TARGET", target)

    launch = _launch_text(tmp_path, monkeypatch, "decode")

    on_disk = _assert_accepted(launch)
    assert launch.world.tokenizer_batches == [[text] for text in kept[:20]]
    assert on_disk["producer"]["content"]["pool"]["tokens"] == target


def test_a_text_run_takes_a_downloaded_dataset_and_then_the_cached_file(tmp_path, monkeypatch):
    records = _fake_dataset_records()
    data = json.dumps(records).encode()
    cache = tmp_path / "FAKE-cache"
    fake = _FakeUrlopen(data)
    env = {
        ENV_CONTENT: "sharegpt_chain",
        ENV_DATASET: FAKE_DATASET_URL,
        ENV_CACHE_DIR: str(cache),
        ENV_SHA256: _sha256(data),
    }

    def prepare(_world, _pod_dir):
        monkeypatch.setattr(launcher.urllib.request, "urlopen", fake)

    first = _launch(tmp_path / "first", monkeypatch, "prefill", env=env, prepare=prepare)
    second = _launch(tmp_path / "second", monkeypatch, "prefill", env=env, prepare=prepare)

    record = {
        "path": str(cache / FAKE_DATASET_NAME),
        "sha256": _sha256(data),
        "collection_pool": "conversation_sha256_even",
        "conversations": len(_fake_kept_texts(records)),
        "url": FAKE_DATASET_URL,
        "downloaded": True,
    }
    assert _assert_accepted(first)["producer"]["content"]["dataset"] == record
    assert _assert_accepted(second)["producer"]["content"]["dataset"] == {**record, "downloaded": False}
    assert fake.calls == [((FAKE_DATASET_URL,), {"timeout": 60})]
    # Both launches sent the same prompts: the content is a function of the seed and of the pool.
    assert [call.prompts for call in first.world.calls] == [call.prompts for call in second.world.calls]


# The four reasons with which a text run fails. Each ends with exit code 65 and a failed artifact that carries
# the reason. The launcher gives the first three and the driver the fourth.


def test_a_dataset_that_cannot_be_had_fails_the_run_before_the_engine_is_built(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "FAKE-no-such-dataset.json"
    reason = f"dataset_unavailable: the dataset path {missing} is not an existing file"
    dump = tmp_path / "resolved-config-node0.json"

    launch = _launch_text(
        tmp_path, monkeypatch, "decode", flags=["--dump-config-to", str(dump)], env={ENV_DATASET: str(missing)}
    )

    on_disk = _assert_failed_before_the_grid(launch, 65, reason)
    # SGLang's arguments were resolved and dumped; nothing was built, so nothing is shut down.
    assert launch.world.events[-4:] == ["model_expert_keys", "resolved_dict", "write", "sleep"]
    assert dump.exists() and launch.world.shutdowns == 0
    # The artifact says what the run was started for, although no prompt was ever made.
    assert on_disk["producer"]["content"] == {
        "source": "dataset_text_chains",
        "seed": "0",
        "token_range": None,
        "issued": None,
    }
    assert _stderr_lines(capsys)[-2:] == [
        f"[sglbench] FAILED: {reason}",
        f"[sglbench] wrote the failed artifact {launch.rank_path}",
    ]


@pytest.mark.parametrize(
    ("error", "shown"),
    [
        (urllib.error.URLError("FAKE: the pod has no route to the internet"), "URLError: <urlopen error FAKE: the pod"),
        (TimeoutError("FAKE: timed out"), "TimeoutError: FAKE: timed out"),
    ],
)
def test_a_download_that_fails_fails_the_run_as_dataset_unavailable(tmp_path, monkeypatch, error, shown):
    def prepare(_world, _pod_dir):
        monkeypatch.setattr(launcher.urllib.request, "urlopen", _FakeUrlopen(error=error))

    # No dataset variable: the default address, into a cache directory beside the directory HF_HOME names.
    env = {ENV_CONTENT: "sharegpt_chain", "HF_HOME": str(tmp_path / "FAKE-models" / "hf-home")}
    launch = _launch(tmp_path, monkeypatch, "prefill", env=env, prepare=prepare)

    assert launch.code == 65
    on_disk = _assert_failed_before_the_grid(launch, 65, launch.artifact()["error"])
    assert on_disk["error"].startswith(f"dataset_unavailable: {shown}")
    assert "build_engine" not in launch.world.events
    assert os.listdir(tmp_path / "FAKE-models" / "fpm_datasets") == []


def test_a_dataset_with_another_digest_fails_the_run_as_dataset_unavailable(tmp_path, monkeypatch):
    wrong = "0" * 64

    launch = _launch_text(tmp_path, monkeypatch, "prefill", env={ENV_SHA256: wrong})

    path, digest = launch.dataset["path"], launch.dataset["sha256"]
    reason = f"dataset_unavailable: the dataset file {path} has the SHA-256 {digest}, expected {wrong}"
    _assert_failed_before_the_grid(launch, 65, reason)
    assert "build_engine" not in launch.world.events


def test_a_dataset_that_is_not_a_list_of_records_fails_the_run_as_dataset_unavailable(tmp_path, monkeypatch):
    launch = _launch_text(tmp_path, monkeypatch, "prefill", records={"FAKE": "not a list"})

    reason = (
        f"dataset_unavailable: {launch.dataset['path']}: ValueError: the dataset must be a list of records, got dict"
    )
    _assert_failed_before_the_grid(launch, 65, reason)
    assert "build_engine" not in launch.world.events


def test_a_dataset_without_a_kept_text_fails_the_run_as_dataset_empty(tmp_path, monkeypatch, capsys):
    records = [{"id": "FAKE", "conversations": [{"from": "human", "value": "FAKE: too short"}]}] * 9

    launch = _launch_text(tmp_path, monkeypatch, "decode", records=records)

    _assert_failed_before_the_grid(launch, 65, "dataset_empty")
    assert launch.world.events[-3:] == ["model_expert_keys", "write", "sleep"]
    assert f"[sglbench] the dataset {launch.dataset['path']} gives no text: " in "\n".join(_stderr_lines(capsys))


def test_texts_without_an_id_the_model_counts_fail_the_run_as_dataset_empty(tmp_path, monkeypatch, capsys):
    def prepare(world, _pod_dir):
        # FAKE: a tokenizer that gives an id above the ids the model counts for every word.
        world.tokenizer_ids_outside = True

    launch = _launch_text(tmp_path, monkeypatch, "decode", prepare=prepare)

    on_disk = _assert_failed_before_the_grid(launch, 65, "dataset_empty")
    # The engine was built by then, so it is shut down.
    assert launch.world.events[-5:] == ["encode_texts", "tokenizer", "shutdown", "write", "sleep"]
    assert on_disk["producer"]["content"]["source"] == "dataset_text_chains"
    # The line before the failure says what the pool dropped: every id and so every conversation.
    kept = _fake_kept_texts(_fake_dataset_records())
    words = sum(len(text.split()) for text in kept)
    (pool_line,) = [line for line in _stderr_lines(capsys) if line.startswith("[sglbench] the text pool holds ")]
    assert pool_line.startswith(
        f"[sglbench] the text pool holds 0 tokens in 0 conversations: {len(kept)} of {len(kept)} "
    )
    assert pool_line.endswith(
        f"and {words} token ids at or above {FAKE_MODEL_VOCAB_SIZE}, the number of ids the model counts, and "
        f"{len(kept)} conversations without an id below it were dropped"
    )


@pytest.mark.parametrize(
    ("step", "error", "shown"),
    [
        (
            "encode_texts",
            RuntimeError("FAKE: the tokenizer has\n no batch call"),
            "RuntimeError: FAKE: the tokenizer has no batch call",
        ),
        (
            "encode_texts",
            TypeError("FAKE: the tokenizer returned str among the ids"),
            "TypeError: FAKE: the tokenizer returned",
        ),
        (
            "tokenizer_class",
            ValueError("FAKE: engine.tokenizer_manager.tokenizer is None"),
            "ValueError: FAKE: engine.",
        ),
    ],
)
def test_a_tokenizer_that_cannot_encode_the_texts_fails_the_run_as_tokenizer_unavailable(
    tmp_path, monkeypatch, step, error, shown
):
    def prepare(world, _pod_dir):
        world.raises[step] = error

    launch = _launch_text(tmp_path, monkeypatch, "prefill", prepare=prepare)

    assert launch.code == 65
    on_disk = _assert_failed_before_the_grid(launch, 65, launch.artifact()["error"])
    assert on_disk["error"].startswith(f"tokenizer_unavailable: {shown}")
    assert launch.world.events[-4:] == [step, "shutdown", "write", "sleep"]


def test_a_dataset_that_is_shorter_than_the_longest_prompt_fails_the_run_as_content_too_shallow(tmp_path, monkeypatch):
    # FAKE: ten records, of which the recipe keeps about half: a few hundred tokens.
    records = _fake_dataset_records(10)
    kept = _fake_kept_texts(records)

    launch = _launch_text(tmp_path, monkeypatch, "decode", records=records)

    tokens = _fake_pool_counts(kept)["tokens"]
    need = _longest_prompt(launch.planned()) + 1
    assert 0 < tokens <= need == 426
    reason = f"content_too_shallow: the text pool holds {tokens} tokens and the longest prompt needs {need}"
    on_disk = _assert_failed(launch, 65, reason)
    # The driver found it once the grid existed, before the first request.
    assert launch.world.calls == [] and launch.world.events[-3:] == ["shutdown", "write", "sleep"]
    assert on_disk["coverage"]["expected_points"] == len(launch.planned().points)
    assert on_disk["producer"]["content"]["pool"]["tokens"] == tokens


def test_the_four_reasons_of_a_text_run_are_the_drivers_and_end_with_exit_code_65():
    reasons = (
        driver.REASON_DATASET_UNAVAILABLE,
        driver.REASON_DATASET_EMPTY,
        driver.REASON_TOKENIZER_UNAVAILABLE,
        driver.REASON_CONTENT_TOO_SHALLOW,
    )
    assert reasons == ("dataset_unavailable", "dataset_empty", "tokenizer_unavailable", "content_too_shallow")
    for reason in reasons:
        assert launcher._exit_code("failed", reason) == launcher._exit_code("failed", f"{reason}: FAKE") == 65
        assert launcher._exit_code("failed", f"benchmark_id=3: benchmark point failed: {reason}: FAKE") == 65


def test_a_token_range_that_is_too_small_fails_a_text_run_as_it_fails_a_run_with_seeded_ids(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        world.vocabulary = {"size": 5000, "excluded": [10, 4000]}

    launch = _launch_text(tmp_path, monkeypatch, "prefill", prepare=prepare)

    # The pool does not depend on the token range, so it is built; the driver then finds the range too small
    # for a content plan, with the words it has for a run with seeded ids.
    _assert_failed(launch, 65, launch.artifact()["error"])
    assert launch.artifact()["error"].startswith(
        "token_range_too_small: the token range [11, 4000) holds 3989 ids, the content plan needs at least 4096"
    )
    assert launch.world.calls == [] and launch.world.events[-3:] == ["shutdown", "write", "sleep"]
    assert "encode_texts" in launch.world.events


def test_a_model_that_counts_too_few_ids_for_a_pool_fails_the_run_before_a_text_is_encoded(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        world.model_vocab_size = 100

    launch = _launch_text(tmp_path, monkeypatch, "prefill", prepare=prepare)

    reason = (
        "token_range_too_small: the model counts 100 token ids, which a text pool refuses: "
        "the token range [0, 100) must hold at least 4096 ids, got 100"
    )
    _assert_failed_before_the_grid(launch, 65, reason)
    assert launch.world.events[-4:] == ["model_vocabulary_size", "shutdown", "write", "sleep"]


def test_a_model_config_without_a_vocabulary_size_is_rejected_in_a_text_run(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        world.raises["model_vocabulary_size"] = TypeError("FAKE: model_config.vocab_size is None, not an int")

    launch = _launch_text(tmp_path, monkeypatch, "decode", prepare=prepare)

    reason = (
        "rejected: the vocabulary size cannot be taken from the model config: "
        "TypeError: FAKE: model_config.vocab_size is None, not an int"
    )
    _assert_failed_before_the_grid(launch, 64, reason)
    assert launch.world.events[-4:] == ["model_vocabulary_size", "shutdown", "write", "sleep"]


def test_a_text_keeps_the_last_id_the_model_counts_and_loses_the_first_it_does_not(tmp_path, monkeypatch):
    # FAKE: the written form of the FAKE special token gets the last id the FAKE model counts. That id lies
    # above the size of the FAKE tokenizer, as the id of an added token does, and so above the token range.
    last = FAKE_MODEL_VOCAB_SIZE - 1
    assert FAKE_VOCABULARY["size"] <= last and FAKE_TOKEN_RANGE[1] <= last
    fake_id = _FakeTokenizer.fake_id
    monkeypatch.setattr(
        _FakeTokenizer,
        "fake_id",
        staticmethod(lambda word: last if word == "<FAKE-special>" else fake_id(word)),
    )

    launch = _launch_text(tmp_path, monkeypatch, "prefill")

    on_disk = _assert_accepted(launch)
    kept = _fake_kept_texts(_fake_dataset_records())
    counts = on_disk["producer"]["content"]["pool"]
    assert counts == _fake_pool_counts(kept)
    # Only the ids at the model's bound were dropped; the ids just below it are text and reached the engine.
    assert counts["dropped_tokens"] == sum(text.split().count("<FAKE-beyond>") for text in kept) > 0
    assert sum(text.split().count("<FAKE-special>") for text in kept) > 0
    sent = {token for call in launch.world.calls for prompt in call.prompts for token in prompt}
    assert max(sent) == last
    assert on_disk["producer"]["content"]["token_range"] == FAKE_TOKEN_RANGE


def test_the_id_range_of_the_pool_ends_at_the_number_of_ids_the_model_counts(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        # FAKE: a model that counts one id more than the FAKE model of the other tests.
        world.model_vocab_size = FAKE_MODEL_VOCAB_SIZE + 1

    launch = _launch_text(tmp_path, monkeypatch, "prefill", prepare=prepare)

    recorded = _assert_accepted(launch)["producer"]["content"]
    # The upper end is the number this model counts. With it the id of ``<FAKE-beyond>`` is one of the
    # model's, so no id of a text is dropped. The token range of the prompts is what it was.
    assert recorded["pool"]["id_range"] == [0, FAKE_MODEL_VOCAB_SIZE + 1]
    assert (recorded["pool"]["dropped_tokens"], recorded["token_range"]) == (0, FAKE_TOKEN_RANGE)


def test_a_run_with_seeded_ids_does_not_read_the_number_of_ids_the_model_counts(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        # FAKE: a model config whose size cannot be read. Only a text run holds the ids of a text against it.
        world.raises["model_vocabulary_size"] = TypeError("FAKE: model_config.vocab_size is None, not an int")

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    _assert_accepted(launch)
    assert "vocabulary" in launch.world.events and "model_vocabulary_size" not in launch.world.events


def test_a_signal_while_the_texts_are_encoded_is_recorded_as_a_stop(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        world.hooks["encode_texts"] = lambda: _deliver(signal.SIGTERM)

    launch = _launch_text(tmp_path, monkeypatch, "prefill", prepare=prepare)

    _assert_failed_before_the_grid(launch, 65, "stopped: signal SIGTERM")
    assert launch.world.events[-4:] == ["encode_texts", "shutdown", "write", "sleep"]


# ---------------------------------------------------------------------------------------------------------------
# Models with expert layers: the expert-parallel layout
# ---------------------------------------------------------------------------------------------------------------

FAKE_EXPERTS = {"num_experts": 60, "num_experts_per_tok": 4, "moe_intermediate_size": 1408}


@pytest.mark.parametrize(
    ("ep_size", "tp_size", "experts"),
    [(1, 1, {}), (1, 4, {}), (1, 1, FAKE_EXPERTS), (1, 2, FAKE_EXPERTS), (2, 2, FAKE_EXPERTS), (8, 8, FAKE_EXPERTS)],
)
def test_an_expert_parallel_size_of_one_or_of_the_tp_size_passes(ep_size, tp_size, experts):
    view = _FakeView({**FAKE_RESOLVED_ARGUMENTS, "ep_size": ep_size, "tp_size": tp_size})
    launcher._check_expert_parallel(view, experts)


_EP_WITHOUT_EXPERTS = "expert parallelism needs a model with expert layers (ep_size={})"
_EP_NOT_TP = "ep_size must be 1 or the TP size (ep_size={}, tp_size={})"


@pytest.mark.parametrize(
    ("ep_size", "tp_size", "experts", "refusal"),
    [
        # A model without expert layers has nothing the flag could spread.
        (2, 2, {}, _EP_WITHOUT_EXPERTS.format(2)),
        (4, 2, {}, _EP_WITHOUT_EXPERTS.format(4)),
        (None, 1, {}, _EP_WITHOUT_EXPERTS.format("null")),
        # A size between 1 and the TP size mixes two layouts, and a size above it is none.
        (2, 4, FAKE_EXPERTS, _EP_NOT_TP.format(2, 4)),
        (4, 2, FAKE_EXPERTS, _EP_NOT_TP.format(4, 2)),
        (2, 1, FAKE_EXPERTS, _EP_NOT_TP.format(2, 1)),
        (0, 2, FAKE_EXPERTS, _EP_NOT_TP.format(0, 2)),
        # A value of another type is never taken for an allowed one.
        (True, 2, FAKE_EXPERTS, _EP_NOT_TP.format("true", 2)),
        (2.0, 2, FAKE_EXPERTS, _EP_NOT_TP.format(2.0, 2)),
        ("2", 2, FAKE_EXPERTS, _EP_NOT_TP.format(2, 2)),
        (None, 2, FAKE_EXPERTS, _EP_NOT_TP.format("null", 2)),
        (2, "2", FAKE_EXPERTS, _EP_NOT_TP.format(2, 2)),
    ],
)
def test_any_other_expert_parallel_size_is_refused(ep_size, tp_size, experts, refusal):
    view = _FakeView({**FAKE_RESOLVED_ARGUMENTS, "ep_size": ep_size, "tp_size": tp_size})

    with pytest.raises(launcher._Failed) as raised:
        launcher._check_expert_parallel(view, experts)

    assert raised.value.reason == f"rejected: {refusal}"


@pytest.mark.parametrize(
    ("experts", "resolved", "refusal"),
    [
        ({}, {"ep_size": 2, "tp_size": 2}, _EP_WITHOUT_EXPERTS.format(2)),
        (FAKE_EXPERTS, {"ep_size": 2, "tp_size": 4}, _EP_NOT_TP.format(2, 4)),
        (FAKE_EXPERTS, {"moe_a2a_backend": "deepep"}, f"{_A2A} (moe_a2a_backend=deepep)"),
        (FAKE_EXPERTS, {"moe_dense_tp_size": 1}, "moe_dense_tp_size must not be set (got 1)"),
        (
            FAKE_EXPERTS,
            {"ep_num_redundant_experts": 4},
            "redundant experts are not supported (ep_num_redundant_experts=4)",
        ),
    ],
)
def test_a_layout_of_expert_layers_the_benchmark_does_not_cover_ends_the_launch(
    tmp_path, monkeypatch, experts, resolved, refusal
):
    def prepare(world, _pod_dir):
        world.experts = experts
        world.resolved.update(resolved)

    launch = _launch_text(tmp_path, monkeypatch, "decode", prepare=prepare, word=None)

    on_disk = _assert_failed_before_the_grid(launch, 64, f"rejected: {refusal}")
    assert "build_engine" not in launch.world.events
    if launch.world.events[-3] == "model_expert_keys":
        # Refused after the model config was read: the keys and the layout are in the artifact.
        assert on_disk["producer"]["expert_keys"] == experts
        assert (on_disk["producer"]["moe"] is None) is (not experts)


@pytest.mark.parametrize(
    ("word", "experts", "resolved", "source"),
    [
        # The model has expert layers and no word was given: a text run, refused for its layout.
        (None, FAKE_EXPERTS, {"ep_size": 2, "tp_size": 4}, "dataset_text_chains"),
        # Text was asked for by name, and the setup is refused before the model config is read.
        ("sharegpt_chain", {}, {"stream_interval": 2}, "dataset_text_chains"),
        ("sharegpt_chain", {}, {"ep_size": 2, "tp_size": 2}, "dataset_text_chains"),
        # Seeded ids were asked for by name.
        ("random", FAKE_EXPERTS, {}, "seeded_token_ids"),
        ("random", {}, {"ep_size": 2, "tp_size": 2}, "seeded_token_ids"),
    ],
)
def test_a_refused_run_names_the_source_it_was_started_for(tmp_path, monkeypatch, word, experts, resolved, source):
    def prepare(world, _pod_dir):
        world.experts = experts
        world.resolved.update(resolved)

    launch = _launch_text(tmp_path, monkeypatch, "decode", prepare=prepare, word=word)

    assert launch.code == 64 and "build_engine" not in launch.world.events
    on_disk = launch.artifact()
    assert on_disk["status"] == "failed" and on_disk["error"].startswith("rejected: ")
    assert on_disk["producer"]["content"] == {"source": source, "seed": "0", "token_range": None, "issued": None}


def test_the_layout_of_the_expert_layers_is_recorded(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        world.experts = FAKE_EXPERTS
        world.resolved.update({"ep_size": 2, "moe_runner_backend": "FAKE-runner"})

    launch = _launch_text(tmp_path, monkeypatch, "prefill", prepare=prepare, word=None, tp_size=2)

    producer = _assert_accepted(launch)["producer"]
    assert producer["expert_keys"] == FAKE_EXPERTS
    assert producer["moe"] == {
        "ep_size": 2,
        "moe_a2a_backend": "none",
        "moe_runner_backend": "FAKE-runner",
        "moe_dense_tp_size": None,
        "enable_eplb": False,
    }
    # Expert parallelism does not change the number of scheduler processes: one for each TP rank.
    assert (producer["tp_size"], producer["scheduler_processes"]) == (2, 2)


def test_the_recorded_fields_of_the_expert_layers_must_exist_for_a_model_that_has_them(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        world.experts = FAKE_EXPERTS
        del world.resolved["moe_runner_backend"]

    launch = _launch_text(tmp_path, monkeypatch, "prefill", prepare=prepare, word=None)

    reason = (
        "internal_error: AttributeError: SGLang's resolved arguments have no field 'moe_runner_backend'; the "
        "start-up check that reads it cannot run"
    )
    _assert_failed_before_the_grid(launch, 70, reason)
    assert "build_engine" not in launch.world.events


def test_a_model_without_expert_layers_does_not_need_the_recorded_fields_of_the_expert_layers(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        del world.resolved["moe_runner_backend"]

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare)

    producer = _assert_accepted(launch)["producer"]
    assert (producer["expert_keys"], producer["moe"]) == ({}, None)


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_end_to_end_a_model_with_expert_layers_and_expert_parallelism_passes_the_validator_and_the_gate(
    tmp_path, monkeypatch, capsys, mode
):
    # The cell AISim plans for this layout: two GPUs, tensor and expert parallel, one expert group for each.
    cell = _tep_cell(mode)
    pod_dir = _pod(tmp_path, cell, sglang_version=FAKE_SGLANG_VERSION)

    def prepare(world, _pod_dir):
        world.experts = FAKE_EXPERTS

    # No content word: the expert keys of the FAKE model config make this a text run.
    launch = _launch_text(
        tmp_path,
        monkeypatch,
        mode,
        word=None,
        tp_size=2,
        flags=["--ep-size", "2"],
        prepare=prepare,
        pod=(cell, pod_dir),
        real_compat=True,
    )

    _assert_no_module_of_the_benchmark_is_replaced()
    assert (launch.cell.parallel_strategy, launch.cell.topology.moe_ep, launch.cell.topology.tp) == ("tep", 2, 2)
    on_disk = _assert_accepted(launch)
    world = launch.world
    # ``--ep-size 2`` and ``--tp-size 2`` reached the FAKE parser as they were typed.
    assert world.sglang_argv == ["--ep-size", "2", *_fake_sglang_argv(1, 2), STREAMING_FLAG]
    assert world.entry_calls == [(0, 0, 0, None), (1, 1, 0, None)]

    producer = on_disk["producer"]
    assert set(producer) == {*LAUNCHER_PRODUCER_KEYS, *DRIVER_PRODUCER_KEYS, "name", "artifact_format"}
    assert producer["expert_keys"] == FAKE_EXPERTS
    assert producer["moe"] == {
        "ep_size": 2,
        "moe_a2a_backend": "none",
        "moe_runner_backend": "auto",
        "moe_dense_tp_size": None,
        "enable_eplb": False,
    }
    assert (producer["tp_size"], producer["scheduler_processes"]) == (2, 2)
    kept = _fake_kept_texts(_fake_dataset_records())
    assert producer["content"] == {
        "source": "dataset_text_chains",
        "seed": "0",
        "token_range": FAKE_TOKEN_RANGE,
        "issued": producer["content"]["issued"],
        "dataset": launch.dataset,
        "pool": _fake_pool_counts(kept),
    }
    dataset_keys = ("path", "sha256", "collection_pool", "conversations")
    if mode == "decode":
        # The block the validator demands of a decode cell with expert parallelism, with the dataset in it.
        assert on_disk["kvwarm"] == {
            "enabled": True,
            "warm_eligible": True,
            "skip_reason": None,
            "method": "radix_cache_seed",
            "dataset": {key: launch.dataset[key] for key in dataset_keys},
        }
        merged = json.loads(launch.merged_path.read_text(encoding="utf-8"))
        assert merged["kvwarm"] == on_disk["kvwarm"]
        collection = _validate(launch.cell, launch.root)
        assert {measurement.kv_seed_regime for measurement in collection.points} == {"real_kv"}
    else:
        assert "kvwarm" not in on_disk

    # What the launcher asked of the FAKE sglang package, in order: the model config before the engine, the
    # tokenizer after it, one batch call for the texts of the FAKE dataset.
    assert world.events == [
        "clock",
        "load_plugins",
        "prepare_server_args",
        "resolve_once",
        "get_platform",
        "from_server_args",
        "Engine",
        "tokenizer",
        "Engine.shutdown",
        "sleep",
    ]
    assert world.tokenizer_batches == [kept]
    # Every request went out as token ids, as in a run with seeded ids: no text reaches the engine.
    assert all(request["prompt"] is None and request["more"] == {} for request in world.engine_requests)
    text_ids = {_FakeTokenizer.fake_id(word) for text in kept for word in text.split()}
    _assert_free_calls_are_text(world, text_ids)
    lines = _stderr_lines(capsys)
    assert lines[-1] == f"[sglbench] complete: wrote {launch.rank_path}, {launch.merged_path}"
    (pool_line,) = [line for line in lines if line.startswith("[sglbench] the text pool holds ")]
    assert f"texts went through {__name__}._FakeTokenizer in " in pool_line


def test_end_to_end_seeded_ids_with_a_model_that_has_expert_layers_are_refused(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        world.experts = FAKE_EXPERTS

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare, env={ENV_CONTENT: "random"}, real_compat=True)

    _assert_failed_before_the_grid(launch, 64, f"rejected: {_SEEDED_WITH_EXPERTS.format('num_experts=60')}")
    assert launch.world.events[-2:] == ["from_server_args", "sleep"]


_HOST_WANTS_TEXT = (
    "native SGLang rank artifact of a MoE plan did not send dataset text: "
    "expected producer.content.source 'dataset_text_chains', found 'seeded_token_ids': "
)
_HOST_WANTS_THE_LAYOUT = (
    "native SGLang rank artifact of a MoE plan did not run the layout of its cell: "
    "expected producer.tp_size {} and producer.moe.ep_size {}, found {} and {}: "
)
# name: (GPUs of the cell, the size flags that reach the pod, the expert keys the pod finds, the content word,
# what the host says when it holds the result against the cell of a MoE plan; None when it accepts)
_RUNS_OF_A_MOE_CELL = {
    "tep": (2, ["--ep-size", "2"], FAKE_EXPERTS, None, None),
    "one-gpu": (1, [], FAKE_EXPERTS, None, None),
    "one-gpu-with-the-text-word": (1, [], FAKE_EXPERTS, "sharegpt_chain", None),
    # The command reached the pod without --ep-size, and the launcher takes an expert-parallel size of 1.
    "tep-without-ep-size": (2, [], FAKE_EXPERTS, None, _HOST_WANTS_THE_LAYOUT.format(2, 2, 2, 1)),
    # The pod found no expert layers in the model config.
    "no-experts-found": (1, [], {}, None, _HOST_WANTS_TEXT),
    "no-experts-found-with-the-seeded-word": (1, [], {}, "random", _HOST_WANTS_TEXT),
    "no-experts-found-with-the-text-word": (1, [], {}, "sharegpt_chain", _HOST_WANTS_THE_LAYOUT.format(1, 1, 1, None)),
}


@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize("name", list(_RUNS_OF_A_MOE_CELL))
def test_end_to_end_the_host_holds_what_the_launcher_recorded_against_the_cell_of_a_moe_plan(
    tmp_path, monkeypatch, mode, name
):
    gpus, size_flags, experts, word, refusal = _RUNS_OF_A_MOE_CELL[name]
    # The cells AISim plans for a model with expert layers: the one-GPU cell and the TEP cell.
    cell = _tep_cell(mode, gpus) if gpus > 1 else _cell(phase=mode, dp=1, strategy="single")
    assert (cell.topology.tp, cell.topology.moe_ep) == (gpus, gpus)
    pod_dir = _pod(tmp_path, cell, sglang_version=FAKE_SGLANG_VERSION)

    def prepare(world, _pod_dir):
        world.experts = experts

    launch = _launch_text(
        tmp_path,
        monkeypatch,
        mode,
        word=word,
        tp_size=gpus,
        flags=size_flags,
        prepare=prepare,
        pod=(cell, pod_dir),
        real_compat=True,
    )

    # Each run is complete: the launcher, the gate and the validator without the plan's answer accept it.
    _assert_no_module_of_the_benchmark_is_replaced()
    _assert_accepted(launch)

    def held_against_the_cell():
        """The validator as the host calls it for a cell of an SGLang plan whose model has expert layers."""
        return validate_native_collection(
            cell,
            launch.root / "raw",
            expected_plan_sha256="plan-sha",
            expected_attempt_id="attempt",
            expected_backend="sglang",
            expected_moe=True,
        )

    if refusal is None:
        assert len(held_against_the_cell().points) == len(launch.planned().points)
    else:
        with pytest.raises(ValueError) as refused:
            held_against_the_cell()
        assert str(refused.value) == refusal + str(launch.rank_path)


def test_the_constants_of_a_text_run():
    assert (launcher.POOL_TOKEN_TARGET, launcher.POOL_ENCODE_BATCH) == (8_000_000, 256)
    assert launcher.DATASET_DOWNLOAD_TIMEOUT_SECONDS == 60
    assert launcher._MOE_RECORDED_FIELDS == (
        "ep_size",
        "moe_a2a_backend",
        "moe_runner_backend",
        "moe_dense_tp_size",
        "enable_eplb",
    )
    # The sources the launcher names to the driver are the content module's.
    assert (content.CONTENT_SEEDED_IDS, content.CONTENT_TEXT_CHAINS) == ("seeded_token_ids", "dataset_text_chains")
    # The pool is not refused or wrapped when a variable of Dynamo's flat pool is set: that variable is refused.
    assert "DYN_BENCH_POOL_TAG" in launcher._REFUSED_VARIABLES
