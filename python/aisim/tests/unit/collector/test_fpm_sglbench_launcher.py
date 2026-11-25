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
- The dtype of the KV pool is FAKE. The model runner of a FAKE scheduler holds a FAKE torch dtype
  (``_fake_torch_dtype``): no torch is involved. Which one follows the KV-cache dtype the FAKE resolution
  left, by a FAKE of SGLang's rule for an NVIDIA GPU and a FAKE model that loads in bfloat16 without a
  quantization config (``FAKE_STORED_KV_DTYPES``). A launch passes ``--kv-cache-dtype fp8_e4m3``, as the cell
  builder writes it for the stand-in cell of these tests, whose KV-cache label is fp8. A test can pass
  another choice, give the FAKE resolution an override that names one, as SGLang's per-model overrides do, or
  make the FAKE engine one that stores another dtype than the argument says, as SGLang does for a model
  whose quantization config asks for an FP8 KV cache.
- With ``--attn-dp-size`` above 1 the FAKE engine is one with DP attention, modelled on what was read in
  SGLang's source and on what engines showed on 2026-11-18 (the module docstring of ``sglbench_launcher.py``
  says what ran). It starts one FAKE scheduler process for each attention-DP rank through the same entry, with
  the ten arguments SGLang passes there. A FAKE DP controller gives the requests of a call to the ranks in
  turn, from where the request before left off. Each rank has a FAKE prefix cache of its own. The ranks run in
  lockstep: in every step each rank runs one FAKE batch, an idle one when it has nothing left, and all batches
  of a step carry the same ``global_num_tokens``. A rank that decodes in a step in which another rank
  extends, and no rank idles, runs that step as SGLang runs a decode batch it has turned into an extend
  batch: mode ``EXTEND`` and one computed token per request (``_FakeEngine._fake_decode_to_extend``; read in
  the source, never seen on an engine). SGLang does that only when the prefill graphs of every rank can
  replay the step; a test can make the FAKE engine one whose graphs cannot, which leaves the decode batch as
  it is. A step takes as long as its slowest rank. Each answer names the rank its request ran on. A FAKE
  scheduler has an input blocker exactly when SGLANG_ENABLE_COLOCATED_BATCH_GEN is on in the environment at
  the moment it is built; the FAKE engine releases the requests of a call together either way, so the
  blocker shows in the recorder headers only.
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
- A standard error whose reader has gone is FAKE (``_FakeGoneStderr``) in every test but one. That one starts
  the launcher in a process of its own with a real pipe, closes the reading end and reads the exit status;
  the import of SGLang is refused in that process.
- One test looks names up in the SGLang source of both tested releases: the fields the dump hides and what
  the refusal of SGLang's memory check in a busy scheduler rests on. It needs a folder with one checkout per
  release (``AISIM_SGLBENCH_SGLANG_SOURCES``) and is skipped without it. It parses files and imports nothing.
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
import re
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
from .test_fpm_sglbench_preflight import SOURCES_ENV, seam_rows_not_in_source, sglang_source_roots

RUNTIME = Path(__file__).resolve().parents[3] / "collector/fpm_forward/runtime"
MODULE_PATH = RUNTIME / "sglbench_launcher.py"
# The folder with one SGLang checkout per tested release, when a developer names one. It is read here because a
# fixture of this file takes the variable out of the environment of every test.
SGLANG_SOURCES_AT_IMPORT = os.environ.get(SOURCES_ENV)
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
    "attn_dp_size",
    "kv_cache_dtype",
    "expert_keys",
    "moe",
)
# One more key of the launcher, written only in a run with several attention-DP ranks.
DP_PRODUCER_KEY = "dp_attention"
BLOCKER_ENV = "SGLANG_ENABLE_COLOCATED_BATCH_GEN"
DRIVER_PRODUCER_KEYS = (
    "timing_metric",
    "timing_clock",
    "staging",
    "recorder_format",
    "scheduler_processes",
    "limits",
    "page_size",
    "enable_overlap",
    "kv_cache_stored_dtype",
    "content",
    "restaged_points",
    "self_test",
    "first_token_need",
)

# The fields of SGLang's resolved arguments that the launcher reads, with the value a default single-GPU dense
# setup resolves to. Typed from the field files of SGLang at both tags (srt/arg_groups/fields: parallel.py,
# schedule.py, memory.py, serving.py, exec_.py, spec.py, disagg.py, device.py, model.py); ``device`` and
# ``prefill_decode_interval`` are None there and resolution fills them in. ``enable_nvidia_megakernel`` exists
# at v0.5.23 only. ``kv_cache_dtype`` (model.py) is ``auto`` unless the command or an override of SGLang's
# resolution names another choice. The seven fields of the expert layers have these defaults for every model
# (``ep_size``, ``moe_dense_tp_size`` and ``moe_dp_size`` in parallel.py, the other four in exec_.py
# ``ExecMoe``). The last
# four are read only with ``attn_dp_size`` above 1 (three in parallel.py, ``enable_tokenizer_batch_encode`` in
# serving.py); ``load_balance_method`` is ``auto`` there and resolution turns it into ``round_robin`` outside
# disaggregated serving (srt/arg_groups/serving_hook.py).
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
    "moe_dp_size": 1,
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
    "kv_cache_dtype": "auto",
    "load_balance_method": "round_robin",
    "enable_dp_lm_head": False,
    "enable_dp_attention_local_control_broadcast": False,
    "enable_tokenizer_batch_encode": False,
}
# The four fields the launcher reads only in a run with several attention-DP ranks.
DP_ONLY_FIELDS = (
    "load_balance_method",
    "enable_dp_lm_head",
    "enable_dp_attention_local_control_broadcast",
    "enable_tokenizer_batch_encode",
)

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
    ("attn_dp_size", 0, "attn_dp_size must be a positive int (got 0)"),
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
    ("moe_dp_size", 2, "moe_dp_size must be 1 (got 2)"),
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
    ("attn_dp_size", -1, "attn_dp_size must be a positive int (got -1)"),
    ("attn_dp_size", None, "attn_dp_size must be a positive int (got null)"),
    ("attn_dp_size", True, "attn_dp_size must be a positive int (got true)"),
    ("attn_dp_size", "2", "attn_dp_size must be a positive int (got 2)"),
    ("attn_dp_size", 2.0, "attn_dp_size must be a positive int (got 2.0)"),
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
    # SGLang's resolution takes a size above 1 that divides the TP size when the attention context-parallel
    # size is the same.
    ("moe_dp_size", 4, "moe_dp_size must be 1 (got 4)"),
    ("moe_dp_size", 0, "moe_dp_size must be 1 (got 0)"),
    ("moe_dp_size", None, "moe_dp_size must be 1 (got null)"),
    ("moe_dp_size", True, "moe_dp_size must be 1 (got true)"),
    ("moe_dp_size", "1", "moe_dp_size must be 1 (got 1)"),
    ("moe_dp_size", 1.0, "moe_dp_size must be 1 (got 1.0)"),
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
    # Any number of attention-DP ranks passes here; the layout is tested once the model config was read.
    ("attn_dp_size", 2),
    ("attn_dp_size", 8),
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
# SGLang's scheduler loop tests this integer variable for truth, so every integer but 0 switches the check on,
# a negative one too (srt/environ.py ``EnvInt``).
_BUSY_CHECK = (
    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY must be unset or 0 (got {}): SGLang would check its KV pools "
    "inside every timed step"
)
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
    ("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY", "1", _BUSY_CHECK.format(1)),
    ("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY", "2", _BUSY_CHECK.format(2)),
    ("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY", " 1 ", _BUSY_CHECK.format(1)),
    ("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY", "-1", _BUSY_CHECK.format(-1)),
]
# The variables whose meaning depends on the number of attention-DP ranks, which are tested after resolution:
# (ranks, name, value, the whole refusal).
_SKIP_GATHER = (
    "SGLANG_SCHEDULER_SKIP_ALL_GATHER must be unset under DP attention: the ranks would not tell each other "
    "what they run in a step"
)
_SPEC_COORDINATION = (
    "SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION must be unset under DP attention: it changes what a rank runs "
    "in a step in which another rank extends"
)
DP_ENV_REFUSALS = [
    (1, BLOCKER_ENV, "1", _BLOCKER),
    (1, BLOCKER_ENV, "TRUE", _BLOCKER),
    (1, BLOCKER_ENV, "true", _BLOCKER),
    (2, "SGLANG_SCHEDULER_SKIP_ALL_GATHER", "1", _SKIP_GATHER),
    (2, "SGLANG_SCHEDULER_SKIP_ALL_GATHER", "True", _SKIP_GATHER),
    (4, "SGLANG_SCHEDULER_SKIP_ALL_GATHER", "y", _SKIP_GATHER),
    (2, "SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION", "yes", _SPEC_COORDINATION),
    (3, "SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION", "1", _SPEC_COORDINATION),
]
DP_ENV_ACCEPTED = [
    # One rank: the blocker variable off or unreadable, and the two variables of DP attention, which SGLang
    # does not look at with one rank.
    (1, BLOCKER_ENV, "0"),
    (1, BLOCKER_ENV, "false"),
    (1, BLOCKER_ENV, "yes"),
    (1, BLOCKER_ENV, ""),
    (1, "SGLANG_SCHEDULER_SKIP_ALL_GATHER", "1"),
    (1, "SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION", "1"),
    # Several ranks: whatever the blocker variable says, because the launcher sets it itself; and the two
    # variables of DP attention off or unreadable.
    (2, BLOCKER_ENV, "1"),
    (2, BLOCKER_ENV, "TRUE"),
    (2, BLOCKER_ENV, "0"),
    (2, BLOCKER_ENV, "false"),
    (2, BLOCKER_ENV, ""),
    (2, "SGLANG_SCHEDULER_SKIP_ALL_GATHER", "0"),
    (2, "SGLANG_SCHEDULER_SKIP_ALL_GATHER", "false"),
    (2, "SGLANG_SCHEDULER_SKIP_ALL_GATHER", "maybe"),
    (2, "SGLANG_SCHEDULER_SKIP_ALL_GATHER", ""),
    (2, "SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION", "no"),
    (2, "SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION", ""),
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
    ("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY", "0"),
    ("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY", "true"),
    ("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY", ""),
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
    # None without DP attention; with it, what every attention-DP rank runs in the step of this batch.
    global_num_tokens: list | None = None

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
    """One FAKE scheduler process: its TP rank, its FAKE process id and the origin of its own FAKE clock.

    ``dp_rank`` is the attention-DP rank whose batches the process runs, 0 without DP attention.
    """

    tp_rank: int
    pid: int
    clock_offset_ticks: int
    parallel: types.SimpleNamespace
    dp_rank: int = 0


def _fake_torch_dtype(name: str) -> object:
    """A FAKE of a torch dtype: an object of a class ``dtype`` of a module ``torch`` with the text ``torch.<name>``.

    The recorder imports no torch and tells a dtype by the name of its class. No torch is installed for these
    tests, so what a real dtype looks like to the recorder was not run here.
    """
    return type("dtype", (), {"__module__": "torch", "__str__": lambda self: f"torch.{name}"})()


# FAKE of what a scheduler process of SGLang builds its KV pool with, by the resolved ``kv_cache_dtype``, on an
# NVIDIA GPU and for a FAKE model that loads in bfloat16 and has no quantization config: the dtype of the model
# for ``auto``, the FP8 dtype for ``fp8_e4m3``, and one FP4 dtype for both of SGLang's FP4 recipes, ``nvfp4``
# and ``fp4_mx_block16``. Typed from SGLang's srt/mem_cache/kv_cache_dtype.py ``configure_kv_cache_dtype`` at
# both tags; a real torch dtype has these names.
FAKE_STORED_KV_DTYPES = {
    "auto": "bfloat16",
    "bfloat16": "bfloat16",
    "fp8_e4m3": "float8_e4m3fn",
    "nvfp4": "float4_e2m1fn_x2",
    "fp4_mx_block16": "float4_e2m1fn_x2",
}
# The flag of SGLang's KV-cache dtype, and the choice a launch of these tests passes with it unless a test says
# otherwise: what the cell builder writes for a cell whose KV-cache label is fp8 (sglang_fpm_builder.py
# ``_KV_CACHE_DTYPES``), which the stand-in cell is. The third is the text of the dtype the FAKE scheduler
# processes of such a launch store.
KV_CACHE_FLAG = "--kv-cache-dtype"
FAKE_KV_CACHE_DTYPE = "fp8_e4m3"
FAKE_KV_STORED_DTYPE = "torch.float8_e4m3fn"


def _fake_blocker_variable_is_on() -> bool:
    """FAKE of how SGLang reads the variable of its input blocker: only true and 1 count, in any letter case."""
    return os.environ.get(BLOCKER_ENV, "false").lower() in ("true", "1")


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
                # SGLang's runner holds the torch dtype it builds the KV pool with, and beside it a string
                # that the recorder must not read.
                kv_cache_dtype=world.fake_kv_pool_dtype(),
                kv_cache_dtype_str="FAKE: the string SGLang keeps beside the dtype",
            )
        )
        self.enable_overlap = True
        # Only the process with attention-TP rank 0 reads a request socket: one process without DP attention,
        # one for each attention-DP rank with it.
        self.recv_from_tokenizer = object() if self.fake_process.parallel.attn_tp_rank == 0 else None
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
        # SGLang builds the blocker in every scheduler process, and only when the variable is on at that time.
        self.input_blocker = object() if _fake_blocker_variable_is_on() else None
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

    def run_scheduler_process(
        server_args,
        port_args,
        gpu_id,
        tp_rank,
        pp_rank,
        dp_rank,
        pipe_writer,
        display_tp_rank=None,
        display_dp_rank=None,
        display_moe_ep_rank=None,
    ):
        """FAKE of SGLang's scheduler entry: apply the class replacement, build the scheduler, report it.

        It takes the seven arguments SGLang's engine passes without DP attention and the three display ranks
        that follow them when the process was started for an attention-DP rank. The real entry then runs the
        event loop for the life of the process. Here the fake engine keeps the scheduler object and calls its
        two batch methods itself.
        """
        world.entry_calls.append((gpu_id, tp_rank, pp_rank, dp_rank))
        world.entry_displays.append((display_tp_rank, display_dp_rank, display_moe_ep_rank))
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
        parser.add_argument("--moe-dp-size", type=int, default=1)
        parser.add_argument("--attn-dp-size", type=int, default=1)
        parser.add_argument("--enable-dp-attention-local-control-broadcast", action="store_true")
        parser.add_argument("--enable-dp-lm-head", action="store_true")
        parser.add_argument("--enable-tokenizer-batch-encode", action="store_true")
        parser.add_argument("--load-balance-method", default="auto")
        parser.add_argument("--page-size", type=int, default=1)
        parser.add_argument(KV_CACHE_FLAG, default="auto", choices=sorted(FAKE_STORED_KV_DTYPES))
        parser.add_argument("--stream-interval", type=int, default=1)
        parser.add_argument(STREAMING_FLAG, action="store_true")
        parsed = parser.parse_args(argv)
        world.sglang_argv = list(argv)
        # The FAKE engine follows the flags, so a flag that does not reach this parser shows in the run.
        world.use(parsed.page_size, parsed.tp_size, parsed.attn_dp_size)
        world.resolved["ep_size"] = parsed.ep_size
        world.resolved["moe_dp_size"] = parsed.moe_dp_size
        world.resolved["kv_cache_dtype"] = parsed.kv_cache_dtype
        world.resolved["enable_dp_attention_local_control_broadcast"] = (
            parsed.enable_dp_attention_local_control_broadcast
        )
        world.resolved["enable_dp_lm_head"] = parsed.enable_dp_lm_head
        world.resolved["enable_tokenizer_batch_encode"] = parsed.enable_tokenizer_batch_encode
        # FAKE of SGLang's resolution: ``auto`` becomes ``round_robin`` outside disaggregated serving.
        method = parsed.load_balance_method
        world.resolved["load_balance_method"] = "round_robin" if method == "auto" else method
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
        Under DP attention the stream of a batch call has the same form; the FAKE engine behind it deals the
        requests to the ranks.
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
            fault = world.call_faults.get(_FakeCall(prompts, rids, max_new_tokens, None, []).key, "").partition("@")[0]
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
                    meta_info = {
                        "id": response["rid"],
                        "finish_reason": finish,
                        "prompt_tokens": response["prompt_tokens"],
                        "cached_tokens": response["cached_tokens"],
                        "completion_tokens": step,
                        "num_retractions": 0,
                    }
                    # SGLang sets the key in every chunk: the rank under DP attention, None without a DP
                    # controller. A test can make the FAKE an SGLang whose chunks name no rank, or name it
                    # as text.
                    if world.rank_in_chunks == "as SGLang reports it":
                        meta_info["dp_rank"] = response["dp_rank"]
                    elif world.rank_in_chunks == "as text":
                        meta_info["dp_rank"] = str(response["dp_rank"])
                    else:
                        assert world.rank_in_chunks == "no key"
                    yield {"index": index, "text": "", "output_ids": [step - 1], "meta_info": meta_info}

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
            # FAKE of a per-model override: it declares its fields over the operator's input.
            self.world.resolved.update(self.world.overrides)
            declared = ("device", "prefill_decode_interval", *self.world.overrides)
            self.declared = {name: self.world.resolved[name] for name in declared}
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
    # The modes of the FAKE batches attention-DP rank 0 ran for the call; without DP attention that is all.
    modes: list[str]
    # The attention-DP rank each request ran on, and the modes of the FAKE batches each rank ran for the call.
    ranks: list[int] = dataclasses.field(default_factory=list)
    rank_modes: list[list[str]] = dataclasses.field(default_factory=list)

    @property
    def key(self) -> tuple[int, str, int]:
        parts = content.parse_request_id(self.rids[0])
        return parts["benchmark_id"], parts["stage"], parts["attempt"]


def _fake_limits(page_size: int) -> grid.Limits:
    """Small FAKE limits that grow with the page size and admit the 256-token warm-up prompt.

    They are the limits of one scheduler, so under DP attention of one attention-DP rank: every FAKE rank
    reports the same record.
    """
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

    def __init__(self, page_size: int = 1, tp_size: int = 1, dp_size: int = 1) -> None:
        self.clock = _FakeClock()
        self.limits = _fake_limits(page_size)
        # The number of FAKE scheduler processes, and the number of attention-DP ranks they form.
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.version = FAKE_SGLANG_VERSION
        self.resolved = {**FAKE_RESOLVED_ARGUMENTS, "tp_size": tp_size, "attn_dp_size": dp_size}
        self.platform = {"is_cuda": True, "is_hip": False}
        # FAKE of an override of SGLang's resolution: the fields it declares over the operator's input, as
        # SGLang's per-model overrides do. The FAKE model of most tests has none.
        self.overrides: dict = {}
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
        # The dtype every FAKE scheduler process builds its KV pool with. None: the one the FAKE rule gives for
        # the resolved ``kv_cache_dtype``. A test puts a FAKE dtype here for an engine that stores another one.
        self.kv_pool_dtype: object | None = None
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
        # The three display ranks each call of the entry carried after its seven arguments, or three None.
        self.entry_displays: list[tuple] = []
        self.processes: list[_FakeProcess] = []
        self.current_process: _FakeProcess | None = None
        self.schedulers: list = []
        self.calls: list[_FakeCall] = []
        # FAKE scheduler steps so far. Every FAKE scheduler process runs one batch in each of them.
        self.batches_run = 0
        # One FAKE prefix cache for each attention-DP rank.
        self.caches = [PageCacheModel(page_size) for _rank in range(dp_size)]
        # FAKE of the counter of SGLang's DP controller: the rank the next request without a rank goes to.
        self.round_robin = 0
        # How the chunks of the FAKE sglang package name the rank of a request: "as SGLang reports it",
        # "no key" or "as text".
        self.rank_in_chunks = "as SGLang reports it"
        # FAKE: whether the prefill graphs of every rank can replay any step, which is what lets SGLang turn
        # a decode batch into an extend batch under DP attention.
        self.prefill_graphs_replay = True
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

    def use(self, page_size: int, tp_size: int, dp_size: int = 1) -> None:
        """Make the FAKE engine one with this page size, this many scheduler processes and this many ranks."""
        self.limits = _fake_limits(page_size)
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.caches = [PageCacheModel(page_size) for _rank in range(dp_size)]
        self.resolved["tp_size"] = tp_size
        self.resolved["attn_dp_size"] = dp_size

    def fake_kv_pool_dtype(self) -> object:
        """The FAKE dtype a FAKE scheduler process holds for its KV pool (``FAKE_STORED_KV_DTYPES``)."""
        if self.kv_pool_dtype is not None:
            return self.kv_pool_dtype
        return _fake_torch_dtype(FAKE_STORED_KV_DTYPES[self.resolved["kv_cache_dtype"]])

    def fake_eviction(self, ranks: tuple[int, ...] | None = None) -> None:
        """FAKE fault: the engine needed room and dropped every cache entry, on all ranks or on some."""
        for rank in range(self.dp_size) if ranks is None else ranks:
            self.caches[rank] = PageCacheModel(self.limits.page_size)

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


def _fake_step_tokens(batch: dict) -> int:
    """What SGLang counts for one rank in one step: computed tokens, decoding requests, or 0 for an idle rank."""
    if batch["mode"] == "EXTEND":
        return sum(batch["extend_lens"])
    return len(batch["rids"]) if batch["mode"] == "DECODE" else 0


# The key of a FAKE batch that marks a decode step the FAKE engine ran as an extend batch. It is no field of
# SGLang's batch.
FAKE_DECODE_STEP = "fake_decode_step"


class _FakeEngine:
    """Stands in for SGLang's engine and its scheduler processes. Everything it reports is FAKE.

    With ``world.dp_size`` above 1 it is an engine with DP attention; the module docstring says what that
    fakes. A fault of ``world.call_faults`` that is about one attention-DP rank names it, ``"evict@1"``.
    """

    def __init__(self, world: _FakeWorld, server_args: _FakeServerArgs, scheduler_entry) -> None:
        self.world = world
        world.entry = scheduler_entry
        writer = _FakePipeWriter()
        dp_size = world.dp_size
        # SGLang: a rank has tp_size // attn_dp_size scheduler processes, and the TP rank of a process says
        # which rank it belongs to and which place it has inside it.
        assert world.tp_size % dp_size == 0, "FAKE of SGLang's own refusal: tp_size is no multiple of attn_dp_size"
        processes_per_rank = world.tp_size // dp_size
        expert_groups = world.resolved.get("ep_size")
        expert_groups = expert_groups if type(expert_groups) is int and 0 < expert_groups <= world.tp_size else 1
        for tp_rank in range(world.tp_size):
            dp_rank = tp_rank // processes_per_rank
            parallel = types.SimpleNamespace(
                tp_size=world.tp_size,
                pp_size=1,
                attn_dp_size=dp_size,
                attn_dp_enabled=dp_size > 1,
                attn_tp_rank=tp_rank % processes_per_rank,
                attn_cp_rank=0,
                attn_dp_rank=dp_rank,
                pp_rank=0,
                pp_max_micro_batch_size=world.limits.max_running_requests,
            )
            # Every FAKE process has a clock origin of its own, as the monotonic clocks of real processes do.
            offset = (tp_rank + 1) * 1000 * TICKS_PER_SECOND
            process = _FakeProcess(tp_rank, FAKE_FIRST_PID + tp_rank, offset, parallel, dp_rank)
            world.processes.append(process)
            world.current_process = process
            if dp_size > 1:
                # Under DP attention SGLang starts the entry in a scheduler process with ten positional
                # arguments: the DP rank in the place of None, and after the pipe the three ranks it shows in
                # its log lines.
                expert_rank = tp_rank // (world.tp_size // expert_groups)
                scheduler_entry(
                    server_args, FAKE_PORT_ARGS, tp_rank, tp_rank, 0, dp_rank, writer, tp_rank, dp_rank, expert_rank
                )
            else:
                # SGLang's engine passes seven positional arguments when no DP controller is in the way:
                # server_args, port_args, gpu_id, tp_rank, pp_rank, dp_rank (None) and the pipe writer.
                scheduler_entry(server_args, FAKE_PORT_ARGS, tp_rank, tp_rank, 0, None, writer)
        world.schedulers = list(writer.sent)

    # -- the FAKE scheduler loop

    def _fake_batches(self, rids: list[str], prompts: list[list[int]], cached: list[int], output_tokens: int) -> list:
        """Extend batches filled up to the chunk budget, then one decode batch for every further output token.

        A request is charged its uncached tokens rounded up to whole pages. A request that does not fit what is
        left of the budget is cut: the part that fits goes into this batch and the rest into the next ones. A
        rank that got no request of a call forms no batch.
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
        if rids:
            batches.extend({"mode": "DECODE", "rids": list(rids)} for _step in range(output_tokens - 1))
        return batches

    def _fake_decode_to_extend(self, rank_batches: list[list[dict]], prompt_tokens: dict[str, int]) -> None:
        """FAKE of what SGLang does to a rank that decodes in a step in which another rank extends.

        Under DP attention SGLang turns such a decode batch into an extend batch, unless a rank idles in the
        step (srt/managers/scheduler_components/dp_attn.py
        ``SchedulerDPAttnAdapter.maybe_convert_decode_to_extend``). The batch keeps its requests and its token
        count of the step. Its mode becomes ``EXTEND``, every request computes one token after everything it
        holds so far, and ``extend_num_tokens`` is the number of requests (srt/managers/schedule_batch.py
        ``ScheduleBatch.convert_decode_to_extend``). SGLang does it only when every rank reports that its
        prefill graphs can replay the step (the vote in ``prepare_mlp_sync_batch_raw``); FAKE: the ranks
        report what ``world.prefill_graphs_replay`` says, for every step alike. Read in the source at both
        tags; no engine showed such a batch.

        The batches are changed in place, so that the modes a test reads off a call are the ones the FAKE
        schedulers ran. Without DP attention nothing is changed.
        """
        if self.world.dp_size == 1 or not self.world.prefill_graphs_replay:
            return
        for step in range(max(len(batches) for batches in rank_batches)):
            row = [batches[step] if step < len(batches) else {"mode": "IDLE", "rids": []} for batches in rank_batches]
            modes = {batch["mode"] for batch in row}
            if "EXTEND" not in modes or "IDLE" in modes:
                continue
            for batches, batch in zip(rank_batches, row, strict=True):
                if batch["mode"] != "DECODE":
                    continue
                # The decode step with zero-based index i reads the prompt and the i output tokens before it.
                steps_before = [
                    earlier["mode"] == "DECODE" or FAKE_DECODE_STEP in earlier for earlier in batches[:step]
                ]
                done = sum(steps_before)
                rids = list(batch["rids"])
                batches[step] = {
                    "mode": "EXTEND",
                    "rids": rids,
                    "prefix_lens": [prompt_tokens[rid] + done for rid in rids],
                    "extend_lens": [1] * len(rids),
                    FAKE_DECODE_STEP: True,
                }

    def _run_batches(self, rank_batches: list[list[dict]], stage: str) -> None:
        """Run the FAKE batches of one call, step by step, in the order of the overlap loop.

        ``rank_batches`` holds the batches of every attention-DP rank. In one step every rank runs its next
        batch, or an idle one when it has none left, on each of its FAKE scheduler processes. In one pass
        SGLang's overlap loop runs the next batch first and processes the result of the batch before it
        second. The result is processed on a copy of the batch.
        """
        world = self.world
        size = world.dp_size
        waiting: list = [None] * len(world.schedulers)
        last_extend: list[dict] = [{} for _rank in range(size)]
        decode_steps = [0] * size
        for step in range(max(len(batches) for batches in rank_batches)):
            row = []
            ticks = 0
            for rank, batches in enumerate(rank_batches):
                batch = batches[step] if step < len(batches) else {"mode": "IDLE", "rids": []}
                row.append(batch)
                if batch["mode"] == "EXTEND":
                    computed = sum(batch["extend_lens"])
                    rank_ticks = FAKE_EXTEND_BASE_TICKS + computed
                    rank_ticks += FAKE_WARM_SHOT_EXTRA_TICKS if stage == "warm" else 0
                    # A decode step that runs as an extend batch gets its three extend fields anew, as SGLang
                    # binds them anew on the batch object, and the decode batches after it carry those.
                    last_extend[rank] = {
                        "extend_num_tokens": computed,
                        "prefix_lens": list(batch["prefix_lens"]),
                        "extend_lens": list(batch["extend_lens"]),
                    }
                    decode_steps[rank] += 1 if FAKE_DECODE_STEP in batch else 0
                elif batch["mode"] == "DECODE":
                    tenths = FAKE_DECODE_STEP_TENTHS[decode_steps[rank] % len(FAKE_DECODE_STEP_TENTHS)]
                    rank_ticks = (FAKE_DECODE_BASE_TICKS + len(batch["rids"])) * tenths
                    decode_steps[rank] += 1
                else:
                    rank_ticks = 0  # a FAKE idle batch has no work of its own
                # The ranks run a step together: it takes as long as the slowest of them.
                ticks = max(ticks, rank_ticks)
            # Under DP attention every batch of a step carries what every rank runs in it.
            tokens = [_fake_step_tokens(batch) for batch in row] if size > 1 else None
            # FAKE GPU: it runs one step after the other.
            world.step_done_ticks = world.gpu_free_ticks = max(world.gpu_free_ticks, world.clock.ticks) + ticks
            world.batches_run += 1
            queued = []
            for scheduler in world.schedulers:
                world.current_process = scheduler.fake_process
                rank = scheduler.fake_process.dp_rank
                batch = row[rank]
                # A decode batch still carries the extend fields of the extend batch its object came from.
                fake_batch = FakeBatch(
                    reqs=[FakeRequest(rid) for rid in batch["rids"]],
                    forward_mode=FakeForwardMode[batch["mode"]],
                    global_num_tokens=None if tokens is None else list(tokens),
                    **last_extend[rank],
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
        fault, _, named_rank = world.call_faults.get(call.key, "").partition("@")
        if fault == "system_exit":
            raise SystemExit(0)
        if fault == "keyboard_interrupt":
            raise KeyboardInterrupt
        if fault == "sigterm":
            _deliver(signal.SIGTERM)
        if fault == "engine_error":
            raise compat.EngineCallError("FAKE: the engine call timed out")
        if fault == "evict":
            world.fake_eviction((int(named_rank),) if named_rank else None)
        if fault == "stray_request":
            # FAKE: a request that is not the benchmark's reached the DP controller before this call and took
            # the turn of one rank.
            world.round_robin = (world.round_robin + 1) % world.dp_size

        # FAKE placement. The call names no rank, so each request goes to the next rank in turn, from where
        # the request before it left off. Without DP attention that is rank 0 for all.
        size = world.dp_size
        call.ranks = [(world.round_robin + index) % size for index in range(len(call.rids))]
        world.round_robin = (world.round_robin + len(call.rids)) % size
        # A call is one batch of arrivals: every request is matched against the cache of its rank before any
        # is stored.
        cached = [world.caches[rank].cached(prompt) for rank, prompt in zip(call.ranks, call.prompts, strict=True)]
        rank_batches = []
        for rank in range(size):
            mine = [index for index, ran_on in enumerate(call.ranks) if ran_on == rank]
            rank_batches.append(
                self._fake_batches(
                    [call.rids[index] for index in mine],
                    [call.prompts[index] for index in mine],
                    [cached[index] for index in mine],
                    max_new_tokens,
                )
            )
        self._fake_decode_to_extend(
            rank_batches, {rid: len(prompt) for rid, prompt in zip(call.rids, call.prompts, strict=True)}
        )
        call.modes.extend(batch["mode"] for batch in rank_batches[0])
        call.rank_modes = [[batch["mode"] for batch in batches] for batches in rank_batches]
        self._run_batches(rank_batches, call.key[1])
        for rank, prompt in zip(call.ranks, call.prompts, strict=True):
            # A finished request stores its prompt and the generated tokens that have KV: all but the last.
            # The FAKE output tokens are negative numbers, so they equal no token id: not an allocated one, not
            # a seeded one and not an id of a text, which the pool keeps from 0 upwards.
            world.caches[rank].finish_prefill([*prompt, *range(-1, -max_new_tokens, -1)])
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
                # Under DP attention an answer names the rank its request ran on; without it SGLang reports
                # none.
                "dp_rank": rank if size > 1 else None,
            }
            for rid, prompt, hit, rank in zip(call.rids, call.prompts, cached, call.ranks, strict=True)
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
        # FAKE parser: the one flag the launcher adds decides the field it stands for, and so does the flag
        # of the KV-cache dtype, which is ``auto`` without it as in SGLang.
        world.resolved["incremental_streaming_output"] = STREAMING_FLAG in argv
        if KV_CACHE_FLAG in argv:
            world.resolved["kv_cache_dtype"] = argv[argv.index(KV_CACHE_FLAG) + 1]
        world.server_args = _FakeServerArgs(world, list(argv))
        return world.server_args

    def resolve(server_args):
        assert server_args is world.server_args
        step("resolve")
        # FAKE of SGLang's resolution: an override decides over the operator's input.
        world.resolved.update(world.overrides)
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
    # The launcher sets these variables for the life of its process: the first in every launch, the second
    # in a launch with several attention-DP ranks. Setting and deleting them here makes pytest take the
    # launcher's values out again after the test.
    for name in (RECORDER_ENV, BLOCKER_ENV):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
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

    def rank_paths(self, dp_size: int) -> list[Path]:
        """The rank files of a run with ``dp_size`` attention-DP ranks, typed here on their own."""
        return [self.rank_path, *(self.pod / f"benchmark_dp{rank}.json" for rank in range(1, dp_size))]

    def result_files(self) -> list[str]:
        """The names of the result files in the pod directory, as the collector would sweep them."""
        return sorted(path.name for path in self.pod.glob("benchmark*.json"))

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
    dp_size: int = 1,
    flags: tuple | list = (),
    env: dict | None = None,
    prepare=None,
    pod: tuple | None = None,
    caps: grid.SampleCaps = SMALL_CAPS,
    cap_flags: list | None = None,
    manifest: dict | None = None,
    real_compat: bool = False,
    kv_cache: str = FAKE_KV_CACHE_DTYPE,
) -> _Launch:
    """Lay out a pod, put the fakes in place and call ``launcher.main`` with the flags of a small run.

    ``tp_size`` is the number of FAKE scheduler processes and ``dp_size`` the number of attention-DP ranks
    they form. With ``real_compat`` the FAKE engine gets its page size and both sizes from the flags alone,
    through the FAKE parser of the FAKE ``sglang`` package. ``kv_cache`` is the choice the command passes with
    ``--kv-cache-dtype``: ``fp8_e4m3`` for an fp8 cell, ``auto`` for a bfloat16 cell.
    """
    world = _FakeWorld() if real_compat else _FakeWorld(page_size, tp_size, dp_size)
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
        *_fake_sglang_argv(page_size, tp_size, dp_size, kv_cache),
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


def _pure_tp_cell(mode: str, tp_size: int = 2):
    """A cell of the pure tensor-parallel layout: ``tp_size`` GPUs, each with a slice of every expert."""
    cell = _cell(phase=mode, dp=1, strategy="pure_tp")
    topology = ParallelTopology(tp=tp_size, pp=1, dp=1, moe_tp=tp_size, moe_ep=1, cp=1)
    return dataclasses.replace(cell, topology=topology)


def _dep_cell(mode: str, dp_size: int = 2):
    """A cell of the layout AISim plans as ``dep``: ``dp_size`` GPUs, each one data-parallel rank and one
    expert group."""
    cell = _cell(phase=mode, dp=dp_size, strategy="dep")
    topology = cell.topology
    assert (topology.tp, topology.dp, topology.moe_tp, topology.moe_ep) == (1, dp_size, 1, dp_size)
    return cell


def _fake_sglang_argv(
    page_size: int = 1, tp_size: int = 1, dp_size: int = 1, kv_cache: str = FAKE_KV_CACHE_DTYPE
) -> list[str]:
    """Flags of SGLang's own parser, which the launcher has to hand on.

    ``--attn-dp-size`` is among them only above 1, as the cell builder writes it. The KV-cache dtype is
    always among them, as the cell builder writes it: the one of an fp8 cell unless a test names another.
    """
    argv = ["--model-path", "fake/dense-model", "--page-size", str(page_size), "--tp-size", str(tp_size)]
    argv += [KV_CACHE_FLAG, kv_cache]
    return [*argv, "--attn-dp-size", str(dp_size)] if dp_size > 1 else argv


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
    # Without DP attention the FAKE batches hold no token counts of other ranks, and no rank ever idles.
    assert all(line["global_num_tokens"] is None and line["mode"] != "IDLE" for line in lines[1:])
    # The entry was called with SGLang's seven arguments: no display rank followed the pipe.
    assert world.entry_displays == [(None, None, None)]
    # Every answer of the FAKE engine named no rank, and every request ran on the one rank there is.
    assert {rank for call in world.calls for rank in call.ranks} == {0}

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
        # One attention-DP rank: the size is written all the same, and nothing else about DP attention.
        "attn_dp_size": 1,
        # What SGLang resolved for the KV cache: the choice the command named.
        "kv_cache_dtype": "fp8_e4m3",
        # A model without expert layers: no expert key, and no layout of expert layers to record.
        "expert_keys": {},
        "moe": None,
    }
    assert set(producer) == {*LAUNCHER_PRODUCER_KEYS, *DRIVER_PRODUCER_KEYS, "name", "artifact_format"}
    assert DP_PRODUCER_KEY not in producer and producer["staging"] == "one_batch_call"
    # The launcher left SGLang's blocker variable alone, and no FAKE scheduler has a blocker.
    assert BLOCKER_ENV not in os.environ and lines[0]["checks"]["input_blocker"] is False
    # A run with seeded ids writes the content block it always wrote.
    assert set(producer["content"]) == {"source", "seed", "token_range", "issued"}
    assert (producer["content"]["source"], producer["content"]["seed"]) == ("seeded_token_ids", "0")
    assert (producer["name"], producer["scheduler_processes"], producer["recorder_format"]) == ("aisim_sglbench", 1, 3)
    # The dtype the FAKE scheduler process built its KV pool with, as its header reports it.
    assert (producer["kv_cache_stored_dtype"], lines[0]["kv_cache_stored_dtype"]) == (FAKE_KV_STORED_DTYPE,) * 2
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
    # The file's points in the file's order, each stamped explicit; the caps on the command line are ignored,
    # and the generated axes add nothing: no point of 1 or of 2 new tokens per request stands beside them.
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
# A standard error that takes no lines
# ---------------------------------------------------------------------------------------------------------------


class _FakeGoneStderr:
    """FAKE standard error whose reader goes away: it takes ``good_writes`` writes, and from then on every
    write fails as on a pipe without a reader, and every flush after such a write."""

    def __init__(self, good_writes: int = 0) -> None:
        self.good_writes = good_writes
        self.taken: list[str] = []
        self.refused: list[str] = []

    def write(self, text: str) -> int:
        if len(self.taken) < self.good_writes:
            self.taken.append(text)
            return len(text)
        self.refused.append(text)
        raise BrokenPipeError(32, "FAKE broken pipe")

    def flush(self) -> None:
        if self.refused:
            raise BrokenPipeError(32, "FAKE broken pipe")


@pytest.fixture
def lose_stderr(monkeypatch):
    """Returns a function that puts a ``_FakeGoneStderr`` in the place of ``sys.stderr``.

    pytest puts the stream of before back after the test. The sink the launcher opened is closed here.
    """

    def lose(good_writes: int = 0) -> _FakeGoneStderr:
        stream = _FakeGoneStderr(good_writes)
        monkeypatch.setattr(sys, "stderr", stream)
        return stream

    yield lose
    if getattr(sys.stderr, "name", None) == os.devnull:
        sys.stderr.close()


def test_a_line_that_cannot_be_printed_raises_nothing_and_a_sink_takes_the_next_lines(lose_stderr):
    gone = lose_stderr()

    launcher._say("FAKE first line")

    # The line was tried once, and a sink is in the place of the stream that refused it.
    assert gone.refused == ["[sglbench] FAKE first line"]
    sink = sys.stderr
    assert sink is not gone and sink.name == os.devnull and not sink.closed
    # What follows goes to the sink, a traceback too: nothing is tried on the old stream again.
    launcher._say("FAKE second line")
    try:
        raise RuntimeError("FAKE failure")
    except RuntimeError:
        launcher._say_traceback()
    assert gone.refused == ["[sglbench] FAKE first line"] and sys.stderr is sink


def test_a_traceback_that_cannot_be_printed_raises_nothing_either(lose_stderr):
    gone = lose_stderr()

    try:
        raise RuntimeError("FAKE failure")
    except RuntimeError:
        launcher._say_traceback()

    assert len(gone.refused) == 1 and gone.refused[0].startswith("Traceback")
    assert sys.stderr is not gone and sys.stderr.name == os.devnull


def test_a_sink_that_cannot_be_opened_does_not_raise(lose_stderr, monkeypatch, tmp_path):
    gone = lose_stderr()
    # FAKE: the null device of this machine cannot be opened.
    monkeypatch.setattr(os, "devnull", str(tmp_path / "FAKE-no-such-directory" / "null"))

    launcher._say("FAKE first line")
    launcher._say("FAKE second line")

    # The stream stays where it was, and every line is tried on it.
    assert sys.stderr is gone
    assert gone.refused == ["[sglbench] FAKE first line", "[sglbench] FAKE second line"]


def _fake_shutdown_failure(world, _pod_dir):
    world.raises["shutdown"] = RuntimeError("FAKE: 2 child processes were not reaped in 60 s")


def _fake_deadline_after_the_second_point(world, _pod_dir):
    world.call_faults[(_planned("decode").points[1].benchmark_id, "run", 0)] = "deadline"


def _fake_unexpected_error(world, _pod_dir):
    world.raises["platform_facts"] = TypeError("FAKE: get_platform().is_cuda is None, not a bool")


# What happens in the launch, its mode, and the exit code and the status of the artifact it must end with.
GONE_STDERR_LAUNCHES = [
    pytest.param(None, "prefill", 0, "complete", id="a complete run"),
    pytest.param(_fake_shutdown_failure, "prefill", 0, "complete", id="an engine shutdown that fails"),
    pytest.param(_fake_deadline_after_the_second_point, "decode", 66, "partial", id="a soft deadline"),
    pytest.param(_fake_unexpected_error, "prefill", 70, "failed", id="an unexpected error"),
]


# With 0 the reader has gone before the launcher's first line. With 2 it goes right after that line, which is
# two writes, so that in the launch with an unexpected error the traceback is the first thing that fails.
@pytest.mark.parametrize("good_writes", [0, 2])
@pytest.mark.parametrize(("prepare", "mode", "code", "status"), GONE_STDERR_LAUNCHES)
def test_a_launch_whose_stderr_takes_no_lines_ends_with_the_code_and_the_files_of_its_outcome(
    tmp_path, monkeypatch, lose_stderr, prepare, mode, code, status, good_writes
):
    gone = lose_stderr(good_writes)

    launch = _launch(tmp_path, monkeypatch, mode, prepare=prepare)

    assert launch.code == code
    on_disk = launch.artifact()
    assert on_disk["status"] == status
    if status == "complete":
        _assert_accepted(launch)
    else:
        with pytest.raises(ValueError, match="invalid native terminal result envelope"):
            _validate(launch.cell, launch.root)
    if prepare is _fake_shutdown_failure:
        assert on_disk["producer"]["engine_shutdown_error"].startswith("RuntimeError: FAKE: 2 child processes")
    if prepare is _fake_unexpected_error:
        assert on_disk["error"] == "internal_error: TypeError: FAKE: get_platform().is_cuda is None, not a bool"
    # The files were written and the final wait was kept.
    assert launch.world.events[-2:] == ["write", "sleep"]
    # One print failed on the stream and none was tried on it afterwards: the rest went to the sink.
    assert len(gone.taken) == good_writes and len(gone.refused) == 1
    assert sys.stderr is not gone and sys.stderr.name == os.devnull


# A child process that runs the launcher without SGLang. It waits until the test has closed its standard input,
# which the test does once the reader of the child's standard error is gone, or kept.
_LAUNCH_WITHOUT_SGLANG = """
import sys

sys.stdin.read()


class Refuse:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "sglang":
            raise ImportError("refused in this test: " + name)
        return None


sys.meta_path.insert(0, Refuse())
sys.path.append(sys.argv[1])
import sglbench_launcher

# No gate polls the files of this test, so the launcher need not stay alive for it.
sglbench_launcher.FINAL_WAIT_SECONDS = 0.0
sys.exit(sglbench_launcher.main(sys.argv[2:]))
"""


@pytest.mark.parametrize(
    ("flags", "code", "error"),
    [
        (["--benchmark-timeout", "0"], 64, "rejected: --benchmark-timeout must be above 0, got 0"),
        ([], 70, "internal_error: ImportError: refused in this test: sglang"),
    ],
    ids=["a refused flag", "a start without SGLang"],
)
def test_a_process_whose_stderr_reader_has_gone_exits_with_the_code_of_its_outcome(tmp_path, flags, code, error):
    """A real process and a real pipe. Python makes the exit status 120 when it cannot flush standard error at
    the end, whatever ``main`` returned, and only a process of its own shows that."""
    seen = {}
    for reader in ("kept", "gone"):
        output = tmp_path / reader / "benchmark.json"
        command = [sys.executable, "-I", "-S", "-B", "-c", _LAUNCH_WITHOUT_SGLANG, str(RUNTIME)]
        command += ["--benchmark-mode", "prefill", "--benchmark-output-path", str(output), *flags]
        read_end, write_end = os.pipe()
        try:
            child = subprocess.Popen(
                command,
                cwd=tmp_path,
                env={"PATH": os.environ.get("PATH", "")},
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=write_end,
            )
        finally:
            os.close(write_end)
        with os.fdopen(read_end, encoding="utf-8") as stderr:
            if reader == "gone":
                stderr.close()
            # The child starts the launcher now.
            child.stdin.close()
            said = "" if stderr.closed else stderr.read()
        returncode = child.wait(timeout=120)
        on_disk = json.loads(output.read_text(encoding="utf-8"))
        seen[reader] = (returncode, on_disk["status"], on_disk["error"])
        if reader == "kept":
            assert f"[sglbench] FAILED: {error}\n" in said

    # The same exit code and the same failed artifact, with a reader and without one.
    assert seen == {"kept": (code, "failed", error), "gone": (code, "failed", error)}


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
    assert len(set(tested)) == len(tested) == 47
    # The fields the launcher reads besides: the platform, the three run settings, the recorded field, the
    # expert-parallel size, which is tested against the TP size and the model, the runner of the expert
    # layers, which is recorded for a model that has them, and the KV-cache dtype, which is recorded.
    # Four more are read only in a run with several attention-DP ranks.
    assert set(FAKE_RESOLVED_ARGUMENTS) - set(tested) == {
        "device",
        "tp_size",
        "prefill_max_requests",
        "max_queued_requests",
        "enable_nvidia_megakernel",
        "ep_size",
        "moe_runner_backend",
        "kv_cache_dtype",
        *DP_ONLY_FIELDS,
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
    "moe_dp_size",
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
    "benchmark_merged_worker_probe.json": b'{"not": "ours"}',
    "resolved-config-node7.json": b'{"not": "ours"}',
    "sglbench-recorder/older-run/sched-1.jsonl": b'{"type":"header"}\n',
    # Names that look like the file of an attention-DP rank and are none: no digits, more than digits, another
    # stem, another ending, the merged file of another stem, and a rank file in another directory.
    "benchmark_dp.json": b'{"not": "ours"}',
    "benchmark_dpx.json": b'{"not": "ours"}',
    "benchmark_dp1x.json": b'{"not": "ours"}',
    "benchmark_dp-1.json": b'{"not": "ours"}',
    "benchmark_dp1.json.bak": b'{"not": "ours"}',
    "benchmark_dp1_merged.json": b'{"not": "ours"}',
    "other_dp1.json": b'{"not": "ours"}',
    "xbenchmark_dp1.json": b'{"not": "ours"}',
    "benchmark_dp1": b'{"not": "ours"}',
    "older/benchmark_dp1.json": b'{"not": "ours"}',
}
# What an earlier run in the same directory can have left: the file of every attention-DP rank it had, and the
# merged file. The launcher deletes all of them, whatever the number of ranks of the new run is.
STALE_RESULT_FILES = {
    "benchmark.json": b'{"stale": "rank 0"}',
    "benchmark_merged.json": b'{"stale": "merged"}',
    "benchmark_dp1.json": b'{"stale": "rank 1"}',
    "benchmark_dp2.json": b'{"stale": "rank 2"}',
    "benchmark_dp13.json": b'{"stale": "rank 13"}',
    "benchmark_dp007.json": b'{"stale": "digits"}',
}


@pytest.mark.parametrize("failing", [False, True])
def test_only_the_launchers_own_old_files_are_deleted(tmp_path, monkeypatch, failing):
    cell, pod_dir = _new_pod(tmp_path, "prefill")
    for name, data in OTHER_FILES.items():
        if data is not None:
            (pod_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (pod_dir / name).write_bytes(data)
    for name, data in STALE_RESULT_FILES.items():
        (pod_dir / name).write_bytes(data)
    before = {name: (pod_dir / name).read_bytes() for name in OTHER_FILES}
    seen = {}

    def prepare(world, _pod_dir):
        world.hooks["sglang_version"] = lambda: seen.update(at_start=sorted(p.name for p in pod_dir.glob("benchmark*")))
        if failing:
            world.experts = {"num_experts": 4}

    # The failing launch asks for seeded ids with a model that has expert layers, which is refused.
    env = {"DYN_BENCH_PREFILL_CONTENT": "random"} if failing else None
    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare, pod=(cell, pod_dir), env=env)

    # Every old result file was gone before SGLang was touched, the files of the other ranks included, and
    # nothing else was: what is left starts with the same word and is not the launcher's.
    assert seen["at_start"] == sorted(name for name in OTHER_FILES if name.startswith("benchmark"))
    assert {name: (pod_dir / name).read_bytes() for name in OTHER_FILES} == before
    assert launch.code == (64 if failing else 0)
    assert launch.artifact()["status"] == ("failed" if failing else "complete")
    # A failed run has no merged file, also not the stale one. This run has one rank, and no file of another
    # rank is left beside its own.
    assert launch.merged_path.exists() is not failing
    assert not [name for name in STALE_RESULT_FILES if name.startswith("benchmark_dp") and (pod_dir / name).exists()]


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
    # The module reaches the environment in one place only: the mapping ``main`` hands to its readers, to the
    # recorder's directory variable and to the function that sets SGLang's blocker variable.
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert source.count("os.environ") == 4 and "os.getenv" not in source
    main_source = ast.get_source_segment(
        source, next(node for node in tree.body if getattr(node, "name", "") == "main")
    )
    assert main_source.count("os.environ") == 4
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
    assert not reserved & {*LAUNCHER_PRODUCER_KEYS, DP_PRODUCER_KEY, "engine_shutdown_error"}
    # The two words the driver writes for how a stage is sent, and the variable the launcher sets for the
    # second of them.
    assert (driver.STAGING_ONE_BATCH_CALL, driver.STAGING_BALANCED_BATCH_CALL) == (
        "one_batch_call",
        "balanced_batch_call",
    )
    assert launcher.ENV_INPUT_BLOCKER == BLOCKER_ENV == "SGLANG_ENABLE_COLOCATED_BATCH_GEN"
    assert (driver.REASON_DP_PLACEMENT, driver.REASON_DP_STEP) == ("dp_rank_placement_mismatch", "dp_ranks_out_of_step")
    assert (recorder.RECORD_FORMAT, driver.RECORD_FORMAT) == (3, 3)


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
        "ssl_keyfile_password": "FAKE-keyfile-password",
        "watermark_key": 123456789,
        "watermark_key_b": 987654321,
        "watermark_config": "/FAKE/watermark.json",
        # FAKE: the extra config of a storage backend, with the two keys of an object store in it.
        "hicache_storage_backend_extra_config": (
            '{"endpoint": "http://storage.fake.invalid:8333", "access_key": "FAKE-access-key", '
            '"secret_key": "FAKE-secret-key"}'
        ),
    }
    # Every field the launcher hides has a FAKE value here.
    assert set(secrets_of_sglang) == set(launcher._SECRET_FIELDS)

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
    # The extra config is hidden whole, so the keys inside it are gone too. They are looked for by themselves:
    # the dump would write the quotes of that value with a backslash.
    assert "FAKE-access-key" not in text and "FAKE-secret-key" not in text
    # A field that is unset stays null, and the record the fake handed out is not changed.
    assert server_args["watermark_key_b"] is None
    assert launch.world.resolved_dict["api_key"] == "sk-FAKE-secret"


# Where SGLang defines the fields the dump hides, typed here on their own as ``(module, name, kind)``, the form
# of the seam rows in test_fpm_sglbench_compat.py.
SGLANG_SECRET_FIELD_ROWS = (
    ("sglang.srt.arg_groups.fields.serving", "Serving.api_key", "attribute"),
    ("sglang.srt.arg_groups.fields.serving", "Serving.admin_api_key", "attribute"),
    ("sglang.srt.arg_groups.fields.serving", "Serving.ssl_keyfile_password", "attribute"),
    ("sglang.srt.arg_groups.fields.exec_", "ExecFeatures.watermark_key", "attribute"),
    ("sglang.srt.arg_groups.fields.exec_", "ExecFeatures.watermark_key_b", "attribute"),
    ("sglang.srt.arg_groups.fields.exec_", "ExecFeatures.watermark_config", "attribute"),
    ("sglang.srt.arg_groups.fields.memory", "Memory.hicache_storage_backend_extra_config", "attribute"),
)
# The two keys an extra config of a storage backend can carry, and what the launcher's refusal of SGLang's
# memory check in a busy scheduler names: the variable and the check.
SGLANG_STORE_KEY_AND_BUSY_CHECK_ROWS = (
    ("sglang.srt.mem_cache.storage.seaweedfs.seaweedfs_store", "SeaweedFSConfig.access_key", "attribute"),
    ("sglang.srt.mem_cache.storage.seaweedfs.seaweedfs_store", "SeaweedFSConfig.secret_key", "attribute"),
    ("sglang.srt.environ", "Envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY", "attribute"),
    (
        "sglang.srt.managers.scheduler_components.invariant_checker",
        "SchedulerInvariantChecker.self_check_during_busy",
        "attribute",
    ),
)


def _busy_check_ends_a_turn(python_root: Path, method: str) -> bool:
    """Whether the loop of ``Scheduler.<method>`` in an SGLang checkout ends every turn with the busy check.

    The last statement of the ``while`` loop must be an ``if`` that names the variable and calls the check.
    The file is parsed, never imported, and only names are looked for.
    """
    tree = ast.parse((python_root / "sglang/srt/managers/scheduler.py").read_text(encoding="utf-8"))
    scheduler = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
    function = next(node for node in scheduler.body if isinstance(node, ast.FunctionDef) and node.name == method)
    last = next(node for node in function.body if isinstance(node, ast.While)).body[-1]
    named = {node.attr for node in ast.walk(last) if isinstance(node, ast.Attribute)}
    wanted = {"SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY", "self_check_during_busy"}
    return isinstance(last, ast.If) and wanted <= named


def test_the_hidden_fields_and_the_busy_check_are_in_the_sglang_source_at_both_tags(monkeypatch):
    # Every field the launcher hides has a row.
    assert {name.partition(".")[2] for _module, name, _kind in SGLANG_SECRET_FIELD_ROWS} == set(launcher._SECRET_FIELDS)
    # Skipped from here on where no SGLang source is at hand. With it, each row is looked up in the parsed
    # files of both releases; nothing is imported. ``_process_state`` took the variable that names the source
    # out of the environment, so it is put back as it was when this file was imported.
    if SGLANG_SOURCES_AT_IMPORT is not None:
        monkeypatch.setenv(SOURCES_ENV, SGLANG_SOURCES_AT_IMPORT)
    rows = SGLANG_SECRET_FIELD_ROWS + SGLANG_STORE_KEY_AND_BUSY_CHECK_ROWS
    for version, python_root in sglang_source_roots(compat.TESTED_VERSIONS).items():
        assert seam_rows_not_in_source(rows, python_root) == [], version
        # The look-up notices a field the source does not have.
        absent = ("sglang.srt.arg_groups.fields.serving", "Serving.no_such_field", "attribute")
        assert seam_rows_not_in_source((absent,), python_root)
        # The check is the last thing in every turn of both scheduler loops, behind the variable.
        for method in ("event_loop_normal", "event_loop_overlap"):
            assert _busy_check_ends_a_turn(python_root, method), (version, method)


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
# Models with expert layers: the layout of the expert layers
# ---------------------------------------------------------------------------------------------------------------

FAKE_EXPERTS = {"num_experts": 60, "num_experts_per_tok": 4, "moe_intermediate_size": 1408}


@pytest.mark.parametrize(
    ("ep_size", "tp_size", "experts"),
    [
        (1, 1, {}),
        (1, 4, {}),
        (1, 1, FAKE_EXPERTS),
        # The pure TP layout: every TP rank holds a slice of every expert.
        (1, 2, FAKE_EXPERTS),
        (1, 4, FAKE_EXPERTS),
        (1, 8, FAKE_EXPERTS),
        # The TEP layout: every TP rank holds whole experts, its share of them.
        (2, 2, FAKE_EXPERTS),
        (8, 8, FAKE_EXPERTS),
    ],
)
def test_an_expert_parallel_size_of_one_or_of_the_tp_size_passes(ep_size, tp_size, experts):
    view = _FakeView({**FAKE_RESOLVED_ARGUMENTS, "ep_size": ep_size, "tp_size": tp_size})
    launcher._check_expert_parallel(view, experts)


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_the_sizes_of_a_pure_tp_cell_pass_the_layout_checks_with_one_scheduler_process_for_each_tp_rank(tp_size):
    # What SGLang resolves the command of a pure TP cell to: its ``--tp-size`` and the default of every other size.
    view = _FakeView({**FAKE_RESOLVED_ARGUMENTS, "tp_size": tp_size})
    assert (view.ep_size, view.moe_dp_size, view.attn_dp_size) == (1, 1, 1)

    launcher._check_resolved(view)
    launcher._check_expert_parallel(view, FAKE_EXPERTS)
    launcher._check_dp_attention(view, FAKE_EXPERTS)

    # The driver waits for a recorder header from each TP rank, and the run has one attention-DP rank.
    producer: dict = {}
    settings = launcher._resolved_settings(view, producer)
    assert (settings["expected_scheduler_processes"], settings["dp_size"]) == (tp_size, 1)
    assert producer == {"tp_size": tp_size, "attn_dp_size": 1, "kv_cache_dtype": "auto"}


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
        # The sizes of a pure TP cell, two TP ranks and an expert-parallel size of 1, with a MoE data-parallel
        # size of 2 beside them: a TP rank would hold a wider slice of every expert than its TP size says.
        (FAKE_EXPERTS, {"tp_size": 2, "moe_dp_size": 2}, "moe_dp_size must be 1 (got 2)"),
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


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_end_to_end_a_pure_tp_cell_of_a_model_with_expert_layers_passes_the_validator_and_the_gate(
    tmp_path, monkeypatch, capsys, mode, tp_size
):
    # The cell AISim plans for this layout: ``tp_size`` GPUs, tensor parallel, each with a slice of every expert.
    cell = _pure_tp_cell(mode, tp_size)
    pod_dir = _pod(tmp_path, cell, sglang_version=FAKE_SGLANG_VERSION)

    def prepare(world, _pod_dir):
        world.experts = FAKE_EXPERTS

    # The command of the cell holds ``--tp-size`` and no ``--ep-size``. No content word: the expert keys of the
    # FAKE model config make this a text run.
    launch = _launch_text(
        tmp_path,
        monkeypatch,
        mode,
        word=None,
        tp_size=tp_size,
        prepare=prepare,
        pod=(cell, pod_dir),
        real_compat=True,
    )

    _assert_no_module_of_the_benchmark_is_replaced()
    topology = launch.cell.topology
    assert (launch.cell.parallel_strategy, topology.tp, topology.moe_tp) == ("pure_tp", tp_size, tp_size)
    assert (topology.moe_ep, topology.dp) == (1, 1)
    # Exit code 0, and the gate and the validator without the plan's answer accept the files.
    on_disk = _assert_accepted(launch)
    world = launch.world
    # ``--tp-size`` reached the FAKE parser as it was typed, and no ``--ep-size`` beside it.
    assert world.sglang_argv == [*_fake_sglang_argv(1, tp_size), STREAMING_FLAG]
    assert "--ep-size" not in world.sglang_argv
    # One scheduler process for each TP rank, started as SGLang starts them without DP attention.
    assert world.entry_calls == [(rank, rank, 0, None) for rank in range(tp_size)]

    producer = on_disk["producer"]
    assert set(producer) == {*LAUNCHER_PRODUCER_KEYS, *DRIVER_PRODUCER_KEYS, "name", "artifact_format"}
    assert producer["expert_keys"] == FAKE_EXPERTS
    # SGLang's default expert-parallel size of 1 is recorded: it is what tells this layout from the TEP one.
    assert producer["moe"] == {
        "ep_size": 1,
        "moe_a2a_backend": "none",
        "moe_runner_backend": "auto",
        "moe_dense_tp_size": None,
        "enable_eplb": False,
    }
    assert (producer["tp_size"], producer["attn_dp_size"], producer["scheduler_processes"]) == (tp_size, 1, tp_size)
    kept = _fake_kept_texts(_fake_dataset_records())
    assert producer["content"] == {
        "source": "dataset_text_chains",
        "seed": "0",
        "token_range": FAKE_TOKEN_RANGE,
        "issued": producer["content"]["issued"],
        "dataset": launch.dataset,
        "pool": _fake_pool_counts(kept),
    }
    if mode == "decode":
        # The block the validator demands of a decode cell of this layout too, with the dataset in it.
        assert on_disk["kvwarm"] == {
            "enabled": True,
            "warm_eligible": True,
            "skip_reason": None,
            "method": "radix_cache_seed",
            "dataset": {key: launch.dataset[key] for key in ("path", "sha256", "collection_pool", "conversations")},
        }
        merged = json.loads(launch.merged_path.read_text(encoding="utf-8"))
        assert merged["kvwarm"] == on_disk["kvwarm"]
    else:
        assert "kvwarm" not in on_disk

    # The host holds the recorded sizes against the cell of a MoE plan and accepts them: N, 1 and 1.
    collection = _held_against_the_moe_cell(launch)
    assert [measurement.point for measurement in collection.points] == launch.planned().point_dicts()
    if mode == "decode":
        assert {measurement.kv_seed_regime for measurement in collection.points} == {"real_kv"}
    assert launch.result_files() == ["benchmark.json", "benchmark_merged.json"]
    assert _stderr_lines(capsys)[-1] == f"[sglbench] complete: wrote {launch.rank_path}, {launch.merged_path}"


def test_end_to_end_a_moe_dp_size_above_one_beside_the_command_of_a_pure_tp_cell_is_refused(tmp_path, monkeypatch):
    cell = _pure_tp_cell("decode")
    pod_dir = _pod(tmp_path, cell, sglang_version=FAKE_SGLANG_VERSION)

    def prepare(world, _pod_dir):
        world.experts = FAKE_EXPERTS

    # FAKE: the resolution of the FAKE package lets the size through. SGLang refuses it with an assert, which
    # is gone when Python runs with -O.
    launch = _launch_text(
        tmp_path,
        monkeypatch,
        "decode",
        word=None,
        tp_size=2,
        flags=["--moe-dp-size", "2"],
        prepare=prepare,
        pod=(cell, pod_dir),
        real_compat=True,
    )

    _assert_no_module_of_the_benchmark_is_replaced()
    world = launch.world
    # The flag is SGLang's and reached its parser beside ``--tp-size 2``; no ``--ep-size`` did.
    assert world.sglang_argv == ["--moe-dp-size", "2", *_fake_sglang_argv(1, 2), STREAMING_FLAG]
    assert (world.resolved["tp_size"], world.resolved["ep_size"], world.resolved["moe_dp_size"]) == (2, 1, 2)
    # SGLang would give the expert layers a TP size of 2 // 1 // 2, while the sizes the artifact records, a TP
    # size of 2 and an expert-parallel size of 1, are those of the pure TP layout.
    on_disk = _assert_failed_before_the_grid(launch, 64, "rejected: moe_dp_size must be 1 (got 2)")
    # Refused once the arguments were resolved: the model config was not read and no engine was built.
    assert world.events[-2:] == ["resolve_once", "sleep"] and "Engine" not in world.events
    assert "expert_keys" not in on_disk["producer"] and "moe" not in on_disk["producer"]


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
# The braces take the three sizes the host expects of the cell and then the three it found: SGLang's TP size,
# its expert-parallel size and its number of attention-DP ranks.
_HOST_WANTS_THE_LAYOUT = (
    "native SGLang rank artifact of a MoE plan did not run the layout of its cell: "
    "expected producer.tp_size {}, producer.moe.ep_size {} and producer.attn_dp_size {}, found {}, {} and {}: "
)
# name: (the strategy of the cell, its GPUs, the size flags that reach the pod, the expert keys the pod finds,
# the content word, what the host says when it holds the result against the cell of a MoE plan; None when it
# accepts)
_RUNS_OF_A_MOE_CELL = {
    "tep": ("tep", 2, ["--ep-size", "2"], FAKE_EXPERTS, None, None),
    # The command of a pure TP cell has no --ep-size, and SGLang's expert-parallel size of 1 is its layout.
    "pure-tp": ("pure_tp", 2, [], FAKE_EXPERTS, None, None),
    "one-gpu": ("single", 1, [], FAKE_EXPERTS, None, None),
    "one-gpu-with-the-text-word": ("single", 1, [], FAKE_EXPERTS, "sharegpt_chain", None),
    # The command of a TEP cell reached the pod without --ep-size. The launcher is not told which layout its
    # cell is and runs the pure TP layout to the end; the host finds the expert-parallel size of another cell.
    "tep-without-ep-size": ("tep", 2, [], FAKE_EXPERTS, None, _HOST_WANTS_THE_LAYOUT.format(2, 2, 1, 2, 1, 1)),
    # The other way round: the command of a pure TP cell reached the pod with the --ep-size of the TEP cell.
    "pure-tp-with-ep-size": (
        "pure_tp",
        2,
        ["--ep-size", "2"],
        FAKE_EXPERTS,
        None,
        _HOST_WANTS_THE_LAYOUT.format(2, 1, 1, 2, 2, 1),
    ),
    # The pod found no expert layers in the model config.
    "no-experts-found": ("single", 1, [], {}, None, _HOST_WANTS_TEXT),
    "no-experts-found-with-the-seeded-word": ("single", 1, [], {}, "random", _HOST_WANTS_TEXT),
    "no-experts-found-with-the-text-word": (
        "single",
        1,
        [],
        {},
        "sharegpt_chain",
        _HOST_WANTS_THE_LAYOUT.format(1, 1, 1, 1, None, 1),
    ),
}


@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize("name", list(_RUNS_OF_A_MOE_CELL))
def test_end_to_end_the_host_holds_what_the_launcher_recorded_against_the_cell_of_a_moe_plan(
    tmp_path, monkeypatch, mode, name
):
    strategy, gpus, size_flags, experts, word, refusal = _RUNS_OF_A_MOE_CELL[name]
    # The cells AISim plans for a model with expert layers without DP attention: the one-GPU cell, the TEP cell
    # and the pure TP cell. The expert-parallel size is the GPU count in a TEP cell and 1 in the other two.
    if strategy == "single":
        cell = _cell(phase=mode, dp=1, strategy="single")
    else:
        cell = _tep_cell(mode, gpus) if strategy == "tep" else _pure_tp_cell(mode, gpus)
    assert (cell.parallel_strategy, cell.topology.tp) == (strategy, gpus)
    assert cell.topology.moe_ep == (gpus if strategy == "tep" else 1)
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


def _dense_tp_cell(mode: str, tp_size: int = 2):
    """A cell of a model without expert layers under tensor parallelism: ``tp_size`` GPUs, every other size 1."""
    cell = _cell(phase=mode, dp=1, strategy="tp")
    return dataclasses.replace(cell, topology=ParallelTopology(tp=tp_size, pp=1, dp=1, moe_tp=1, moe_ep=1, cp=1))


# The braces take the tensor-parallel size the host expects of the cell and then the two sizes it found: SGLang's
# TP size and its number of attention-DP ranks.
_HOST_WANTS_THE_DENSE_SIZES = (
    "native SGLang rank artifact of a dense plan did not run the layout of its cell: "
    "expected producer.tp_size {} and producer.attn_dp_size 1, found {} and {}: "
)
_HOST_WANTS_NO_EXPERT_PARALLELISM = (
    "native SGLang rank artifact of a dense plan did not run the layout of its cell: "
    "expected producer.moe null or producer.moe.ep_size 1, found producer.moe.ep_size {}: "
)
# Expert keys the launcher looks up and the plan does not count (capabilities.py, ``_is_moe``, reads the three
# counts ``n_routed_experts``, ``num_local_experts`` and ``num_experts``): a model the plan takes for dense and
# the launcher for one with expert layers.
FAKE_EXPERTS_THE_PLAN_DOES_NOT_COUNT = {"moe_num_experts": 8, "num_experts_per_tok": 2}
# name: (the GPUs of the cell, the --tp-size that reaches the pod, the other size flags that reach it, the
# expert keys the pod finds, the expert-parallel size the launcher records or None for no moe block, what the
# host says when it holds the result against the cell of a dense plan; None when it accepts)
_RUNS_OF_A_DENSE_CELL = {
    "one-gpu": (1, 1, [], {}, None, None),
    "tp": (2, 2, [], {}, None, None),
    # The command of a cell reached the pod with the --tp-size of the other cell. The launcher is not told the
    # size of its cell and runs the size SGLang resolved to the end; the host finds the size of another cell.
    "tp-on-one-gpu": (2, 1, [], {}, None, _HOST_WANTS_THE_DENSE_SIZES.format(2, 1, 1)),
    "one-gpu-on-two": (1, 2, [], {}, None, _HOST_WANTS_THE_DENSE_SIZES.format(1, 2, 1)),
    # The pod found expert layers. Without --ep-size every rank holds a slice of every expert, which is the
    # tensor parallelism of the cell.
    "experts-found": (2, 2, [], FAKE_EXPERTS_THE_PLAN_DOES_NOT_COUNT, 1, None),
    "experts-found-on-one-gpu": (1, 1, [], FAKE_EXPERTS_THE_PLAN_DOES_NOT_COUNT, 1, None),
    # With an --ep-size of the tensor-parallel size the launcher runs the TEP layout, which is no cell of a
    # dense plan.
    "experts-found-with-ep-size": (
        2,
        2,
        ["--ep-size", "2"],
        FAKE_EXPERTS_THE_PLAN_DOES_NOT_COUNT,
        2,
        _HOST_WANTS_NO_EXPERT_PARALLELISM.format(2),
    ),
}


@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize("name", list(_RUNS_OF_A_DENSE_CELL))
def test_end_to_end_the_host_holds_what_the_launcher_recorded_against_the_cell_of_a_dense_plan(
    tmp_path, monkeypatch, mode, name
):
    gpus, tp_size, size_flags, experts, ep_size, refusal = _RUNS_OF_A_DENSE_CELL[name]
    # The cells AISim plans for a model without expert layers: the one-GPU cell and the tensor-parallel cells.
    cell = _cell(phase=mode, dp=1, strategy="single") if gpus == 1 else _dense_tp_cell(mode, gpus)
    assert (cell.topology.tp, cell.topology.dp, cell.topology.moe_ep) == (gpus, 1, 1)
    pod_dir = _pod(tmp_path, cell, sglang_version=FAKE_SGLANG_VERSION)

    def prepare(world, _pod_dir):
        world.experts = experts

    # No content word: the launcher sends seeded ids, or dataset text when it finds expert layers.
    launch = _launch_text(
        tmp_path,
        monkeypatch,
        mode,
        word=None,
        tp_size=tp_size,
        flags=size_flags,
        prepare=prepare,
        pod=(cell, pod_dir),
        real_compat=True,
    )

    # Each run is complete: the launcher, the gate and the validator without the plan's answer accept it.
    _assert_no_module_of_the_benchmark_is_replaced()
    producer = _assert_accepted(launch)["producer"]
    # The three keys the host reads are the launcher's own, and the file of each of these runs holds them: the
    # ``moe`` key is there with null when the launcher found no expert layers.
    recorded_ep_size = None if producer["moe"] is None else producer["moe"]["ep_size"]
    assert (producer["tp_size"], producer["attn_dp_size"], recorded_ep_size) == (tp_size, 1, ep_size)

    def held_against_the_cell():
        """The validator as the host calls it for a cell of an SGLang plan whose model has no expert layers."""
        return validate_native_collection(
            cell,
            launch.root / "raw",
            expected_plan_sha256="plan-sha",
            expected_attempt_id="attempt",
            expected_backend="sglang",
            expected_moe=False,
        )

    if refusal is None:
        assert len(held_against_the_cell().points) == len(launch.planned().points)
    else:
        with pytest.raises(ValueError) as refused:
            held_against_the_cell()
        assert str(refused.value) == refusal + str(launch.rank_path)


# ---------------------------------------------------------------------------------------------------------------
# The KV cache: what SGLang resolved, what the engine stored, and the label of the cell
# ---------------------------------------------------------------------------------------------------------------

# The braces take the KV-cache label of the cell, the stored dtypes the label stands for, the stored dtype the
# host found and the argument SGLang resolved.
_HOST_WANTS_THE_KV_CACHE_OF_THE_CELL = (
    "native SGLang rank artifact did not store the KV cache of its cell: the cell's kv_cache_dtype {!r} stands "
    "for producer.kv_cache_stored_dtype {}, found {!r} (producer.kv_cache_dtype {!r}): "
)
# The braces take the label, the stored dtype the host found and the argument SGLang resolved.
_HOST_HAS_NO_RULE_FOR_THE_LABEL = (
    "native SGLang rank artifact cannot be held against the KV cache of its cell: no rule says which "
    "producer.kv_cache_stored_dtype the cell's kv_cache_dtype {!r} stands for, found {!r} "
    "(producer.kv_cache_dtype {!r}): "
)
# The stored dtypes that each label with a rule stands for, as the host words them. For ``nvfp4`` the host
# wants the resolved argument as well, because SGLang stores two FP4 recipes in one dtype.
_STORED_DTYPES_OF_A_LABEL = {
    "bfloat16": "'torch.bfloat16' or 'torch.float16'",
    "fp8": "'torch.float8_e4m3fn'",
    "nvfp4": "'torch.float4_e2m1fn_x2' with producer.kv_cache_dtype 'nvfp4'",
}
# The other label an option can name for a cell. The host has no rule for it.
_LABELS_WITHOUT_A_RULE = ("int8",)
# name: (the choice the command passes with --kv-cache-dtype, what a FAKE override of SGLang's resolution
# declares, the name of the FAKE dtype the FAKE scheduler processes store or None for the one the FAKE rule
# gives, the resolved argument and the stored dtype the result files must record, the labels of a cell the host
# accepts the files for)
_RUNS_OF_THE_KV_CACHE = {
    # The command of an fp8 cell and that of a bfloat16 cell, for a FAKE model that loads in bfloat16.
    "an-fp8-cell": ("fp8_e4m3", {}, None, "fp8_e4m3", "torch.float8_e4m3fn", ("fp8",)),
    "a-bfloat16-cell": ("auto", {}, None, "auto", "torch.bfloat16", ("bfloat16",)),
    # A FAKE engine whose model loads in float16, as SGLang loads a float16 or a float32 checkpoint.
    "a-bfloat16-cell-of-a-float16-model": ("auto", {}, "float16", "auto", "torch.float16", ("bfloat16",)),
    # A FAKE engine that stores FP8 for ``auto``, as SGLang does for a model whose quantization config asks for
    # an FP8 KV cache. The argument still says ``auto``, and the files are those of an fp8 cell: the argument is
    # compared with nothing.
    "a-bfloat16-cell-of-a-model-that-asks-for-fp8": (
        "auto",
        {},
        "float8_e4m3fn",
        "auto",
        "torch.float8_e4m3fn",
        ("fp8",),
    ),
    # A FAKE override that makes ``fp8_e4m3`` of ``auto``, as SGLang's overrides do for some architectures. The
    # launcher records what resolution left, not what the command said.
    "a-bfloat16-cell-of-an-architecture-sglang-overrides": (
        "auto",
        {"kv_cache_dtype": "fp8_e4m3"},
        None,
        "fp8_e4m3",
        "torch.float8_e4m3fn",
        ("fp8",),
    ),
    # The command the cell builder writes for the label nvfp4, which it passes on under its own name. No plan
    # carries that label today.
    "an-nvfp4-cell": ("nvfp4", {}, None, "nvfp4", "torch.float4_e2m1fn_x2", ("nvfp4",)),
    # SGLang's other FP4 recipe stores the same dtype. No label stands for it: the argument tells the two apart.
    "the-other-fp4-recipe": ("fp4_mx_block16", {}, None, "fp4_mx_block16", "torch.float4_e2m1fn_x2", ()),
    # A FAKE engine that stores the FP4 dtype for ``auto``: the dtype of an nvfp4 cell without its argument.
    "a-bfloat16-cell-that-stored-fp4": ("auto", {}, "float4_e2m1fn_x2", "auto", "torch.float4_e2m1fn_x2", ()),
    # Dtypes no label stands for: another FP8 dtype of torch, and a float32 pool.
    "an-fp8-cell-that-stored-another-fp8-dtype": ("fp8_e4m3", {}, "float8_e5m2", "fp8_e4m3", "torch.float8_e5m2", ()),
    "a-bfloat16-cell-that-stored-float32": ("auto", {}, "float32", "auto", "torch.float32", ()),
}


@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize("name", list(_RUNS_OF_THE_KV_CACHE))
def test_end_to_end_the_host_holds_the_dtype_the_engine_stored_against_the_kv_cache_label_of_the_cell(
    tmp_path, monkeypatch, mode, name
):
    choice, overrides, stored_name, resolved, stored, accepted = _RUNS_OF_THE_KV_CACHE[name]
    # Two GPUs under tensor parallelism: two FAKE scheduler processes, each with a KV pool of its own.
    cell = _dense_tp_cell(mode, 2)
    pod_dir = _pod(tmp_path, cell, sglang_version=FAKE_SGLANG_VERSION)

    def prepare(world, _pod_dir):
        world.overrides = dict(overrides)
        if stored_name is not None:
            world.kv_pool_dtype = _fake_torch_dtype(stored_name)

    launch = _launch(
        tmp_path, monkeypatch, mode, tp_size=2, kv_cache=choice, prepare=prepare, pod=(cell, pod_dir), real_compat=True
    )

    # Each run is complete, whatever it stored: neither the launcher nor the driver is told the label of the
    # cell, and the gate in the pod accepts the files.
    _assert_no_module_of_the_benchmark_is_replaced()
    assert launch.code == 0 and launch.world.sglang_argv[-3:-1] == [KV_CACHE_FLAG, choice]
    on_disk = launch.artifact()
    assert (on_disk["status"], on_disk["valid"], on_disk["usable"], on_disk["error"]) == ("complete", True, True, None)
    assert _gate([str(launch.rank_path)], launch.mode).returncode == GATE_ACCEPT
    # The launcher wrote what SGLang resolved, the driver what both headers of the real recorder report.
    producer = on_disk["producer"]
    assert (producer["kv_cache_dtype"], producer["kv_cache_stored_dtype"]) == (resolved, stored)
    assert [lines[0]["kv_cache_stored_dtype"] for lines in _rank_logs(launch)] == [stored, stored]
    merged = json.loads(launch.merged_path.read_text(encoding="utf-8"))["producer"]
    assert (merged["kv_cache_dtype"], merged["kv_cache_stored_dtype"]) == (resolved, stored)

    for label in (*_STORED_DTYPES_OF_A_LABEL, *_LABELS_WITHOUT_A_RULE):
        labelled = dataclasses.replace(cell, kv_cache_dtype=label)
        # The validator as a caller without a plan calls it, and as the host calls it for a cell of an SGLang
        # plan whose model has no expert layers: the rule is the same for both.
        for expected_moe in (None, False):
            keywords = {
                "expected_plan_sha256": "plan-sha",
                "expected_attempt_id": "attempt",
                "expected_backend": "sglang",
                "expected_moe": expected_moe,
            }
            if label in accepted:
                collection = validate_native_collection(labelled, launch.root / "raw", **keywords)
                assert len(collection.points) == len(launch.planned().points)
                continue
            with pytest.raises(ValueError) as refused:
                validate_native_collection(labelled, launch.root / "raw", **keywords)
            if label in _LABELS_WITHOUT_A_RULE:
                text = _HOST_HAS_NO_RULE_FOR_THE_LABEL.format(label, stored, resolved)
            else:
                dtypes = _STORED_DTYPES_OF_A_LABEL[label]
                text = _HOST_WANTS_THE_KV_CACHE_OF_THE_CELL.format(label, dtypes, stored, resolved)
            assert str(refused.value) == text + str(launch.rank_path)


def test_end_to_end_scheduler_processes_that_stored_different_dtypes_fail_the_run(tmp_path, monkeypatch):
    # FAKE: the two scheduler processes of one engine build their KV pools with different dtypes.
    def prepare(world, _pod_dir):
        dtypes = iter([_fake_torch_dtype("float8_e4m3fn"), _fake_torch_dtype("bfloat16")])
        world.fake_kv_pool_dtype = lambda: next(dtypes)

    launch = _launch(tmp_path, monkeypatch, "prefill", tp_size=2, prepare=prepare, real_compat=True)

    reason = (
        'rejected: scheduler processes disagree on kv_cache_stored_dtype: ["torch.float8_e4m3fn", "torch.bfloat16"]'
    )
    on_disk = _assert_failed_before_the_grid(launch, 64, reason)
    # The launcher's fact is there, and the driver wrote no stored dtype for a run without one.
    producer = on_disk["producer"]
    assert (producer["kv_cache_dtype"], producer["kv_cache_stored_dtype"]) == (FAKE_KV_CACHE_DTYPE, None)


@pytest.mark.parametrize(
    ("held", "error"),
    [
        # The string SGLang keeps beside the dtype, the text of a dtype, and a number.
        ("auto", "TypeError('model_runner.kv_cache_dtype is a builtins.str, not a torch.dtype')"),
        ("torch.bfloat16", "TypeError('model_runner.kv_cache_dtype is a builtins.str, not a torch.dtype')"),
        (0, "TypeError('model_runner.kv_cache_dtype is a builtins.int, not a torch.dtype')"),
    ],
)
def test_end_to_end_a_model_runner_that_holds_no_torch_dtype_fails_the_run(tmp_path, monkeypatch, held, error):
    # FAKE: a model runner whose ``kv_cache_dtype`` is no torch dtype, as a changed SGLang could hold it.
    launch = _launch(
        tmp_path, monkeypatch, "prefill", prepare=lambda world, _pod_dir: setattr(world, "kv_pool_dtype", held)
    )

    log = f"sched-{os.getpid()}.jsonl"
    reason = f"rejected: the recorder could not read kv_cache_stored_dtype in {log}: {error}"
    on_disk = _assert_failed_before_the_grid(launch, 64, reason)
    assert on_disk["producer"]["kv_cache_stored_dtype"] is None


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


# ---------------------------------------------------------------------------------------------------------------
# DP attention: the launch of a DEP cell
# ---------------------------------------------------------------------------------------------------------------


def _fake_expected_rank_sample(point: grid.Point, rank: int) -> float:
    """The FAKE sample of one attention-DP rank, worked out here from the FAKE timings.

    The FAKE scheduler processes of a step launch their batches one after the other, each a FAKE host step
    after the one before, and the results of a step are on the host at one FAKE moment. A prefill sample runs
    from the launch of the rank's own batch, so rank ``r`` has ``r`` host steps less than rank 0. A decode
    sample is taken between two outputs, which are the same moments for every rank.
    """
    if point.phase == grid.PREFILL:
        ticks = FAKE_EXTEND_BASE_TICKS + point.batch_size * point.new_tokens - rank * FAKE_HOST_TICKS
        return ticks / TICKS_PER_SECOND
    return _fake_expected_sample(point)


def _launch_dep(
    tmp_path: Path,
    monkeypatch,
    mode: str = "decode",
    *,
    dp_size: int = 2,
    word: str | None = None,
    flags: tuple | list = (),
    prepare=None,
    pod_dir: Path | None = None,
    **options,
) -> _Launch:
    """Launch a DEP cell: a FAKE model with expert layers on ``dp_size`` attention-DP ranks, with dataset text.

    The command is the one of the cell: ``--tp-size D --attn-dp-size D --ep-size D``. With the FAKE compat
    functions no parser reads ``--ep-size``, so the resolved value is put into the FAKE world here.
    ``pod_dir`` is the pod directory of an earlier launch to run in again; a new one is laid out without it.
    """
    cell = _dep_cell(mode, dp_size)
    if pod_dir is None:
        pod_dir = _pod(tmp_path, cell, sglang_version=FAKE_SGLANG_VERSION)

    def prepare_dep(world, pod):
        world.experts = FAKE_EXPERTS
        world.resolved["ep_size"] = dp_size
        if prepare is not None:
            prepare(world, pod)

    return _launch_text(
        tmp_path,
        monkeypatch,
        mode,
        word=word,
        tp_size=dp_size,
        dp_size=dp_size,
        flags=["--ep-size", str(dp_size), *flags],
        prepare=prepare_dep,
        pod=(cell, pod_dir),
        **options,
    )


def _held_against_the_moe_cell(launch: _Launch):
    """The validator as the host calls it for a cell of an SGLang plan whose model has expert layers."""
    return validate_native_collection(
        launch.cell,
        launch.root / "raw",
        expected_plan_sha256="plan-sha",
        expected_attempt_id="attempt",
        expected_backend="sglang",
        expected_moe=True,
    )


def _assert_ranks_accepted(launch: _Launch, dp_size: int = 2) -> list[dict]:
    """A complete run with several attention-DP ranks: exit code 0, one rank file per rank and the merged
    file, and the validator and the gate accept them. Returns the rank files, in rank order."""
    built = launch.planned()
    assert launch.code == 0
    rank_paths = launch.rank_paths(dp_size)
    assert launch.result_files() == sorted([*(path.name for path in rank_paths), launch.merged_path.name])
    on_disk = [json.loads(path.read_text(encoding="utf-8")) for path in rank_paths]
    for rank, payload in enumerate(on_disk):
        assert (payload["status"], payload["valid"], payload["usable"], payload["error"]) == (
            "complete",
            True,
            True,
            None,
        )
        assert payload["dp"] == {"size": dp_size, "rank": rank}
        assert payload["coverage"] == {
            "expected_points": len(built.points),
            "completed_points": len(built.points),
            "skipped_points": 0,
        }
        assert payload["grid_digest"] == built.digest
        for point, row, group in zip(built.points, payload["results"], payload["iteration_groups"], strict=True):
            # A row holds the record of its own rank: the batch size and the token counts of ONE rank, which
            # are the point's, and the FAKE sample of that rank's own FAKE scheduler process.
            (fpm,) = row["fpms"]
            assert (row["point"], row["kv_seed_regime"]) == (point.as_dict(), _expected_regime(point))
            assert (fpm["dp_rank"], fpm["wall_time"], fpm["worker_id"]) == (
                rank,
                _fake_expected_rank_sample(point, rank),
                FAKE_WORKER_ID,
            )
            assert fpm["scheduled_requests"] == artifact.scheduled_requests(point.as_dict())
            if point.phase == grid.DECODE:
                assert fpm[artifact.DECODE_STEPS_KEY] == _fake_expected_steps(point)
            # A group holds the record of every rank, and its time is the slowest rank's.
            assert [entry["dp_rank"] for entry in group["rank_results"]] == list(range(dp_size))
            assert group["wall_time"] == max(_fake_expected_rank_sample(point, other) for other in range(dp_size))
        # Every rank file carries the same groups, the same run and the same facts of the run.
        assert payload["iteration_groups"] == on_disk[0]["iteration_groups"]
        assert (payload["run_id"], payload["producer"]) == (on_disk[0]["run_id"], on_disk[0]["producer"])
    merged = json.loads(launch.merged_path.read_text(encoding="utf-8"))
    assert (merged["artifact_type"], merged["rank_files"]) == ("merged", [str(path) for path in rank_paths])
    assert merged["dp"]["source_ranks"] == list(range(dp_size))
    if launch.mode != grid.AGG:
        # The host holds the three sizes the launcher recorded against the cell: for a DEP cell of D ranks
        # SGLang's TP size D, its expert-parallel size D and D attention-DP ranks.
        collection = _held_against_the_moe_cell(launch)
        assert [measurement.point for measurement in collection.points] == built.point_dicts()
        assert [measurement.rank_wall_times for measurement in collection.points] == [
            tuple((rank, _fake_expected_rank_sample(point, rank)) for rank in range(dp_size)) for point in built.points
        ]
        assert (collection.runtime_run_id, collection.runtime_grid_digest) == (on_disk[0]["run_id"], built.digest)
    gate = _gate([str(path) for path in rank_paths], launch.mode)
    assert gate.returncode == GATE_ACCEPT, gate.stderr
    return on_disk


def _rank_logs(launch: _Launch) -> list[list[dict]]:
    """The lines of every recorder log of the one run of a launch, in the order of the FAKE process ids."""
    (recorder_dir,) = launch.recorder_dirs()
    return [
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for path in sorted(recorder_dir.iterdir())
    ]


DP_ATTENTION_FACTS = {"local_control_broadcast": False, "input_blocker": True, "placement": "round_robin_checked"}


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_end_to_end_a_dep_launch_passes_the_validator_and_the_gate(tmp_path, monkeypatch, capsys, mode):
    seen = {}

    def prepare(world, _pod_dir):
        # What the FAKE scheduler processes inherit: the variable as it stands when the engine is built.
        world.hooks["Engine"] = lambda: seen.update(blocker=os.environ.get(BLOCKER_ENV))

    # No content word: the expert keys of the FAKE model config make this a text run.
    launch = _launch_dep(tmp_path, monkeypatch, mode, prepare=prepare, real_compat=True)

    _assert_no_module_of_the_benchmark_is_replaced()
    topology = launch.cell.topology
    assert (launch.cell.parallel_strategy, topology.tp, topology.dp, topology.moe_ep) == ("dep", 1, 2, 2)
    built = launch.planned()
    assert len(built.points) >= 8 and built.missing_phases == ()
    on_disk = _assert_ranks_accepted(launch)
    world = launch.world

    # The flags of the cell reached the FAKE parser as they were typed, and nothing was added but the
    # streaming flag.
    assert world.sglang_argv == ["--ep-size", "2", *_fake_sglang_argv(1, 2, 2), STREAMING_FLAG]
    assert world.sglang_argv[-3:-1] == ["--attn-dp-size", "2"]
    # The launcher set SGLang's blocker variable before the engine was built, and left it set.
    assert seen == {"blocker": "1"} and os.environ[BLOCKER_ENV] == "1"

    # One FAKE scheduler process for each attention-DP rank, each started through the recorder's entry with
    # the ten arguments SGLang passes there: the DP rank, and after the pipe the three display ranks.
    assert world.entry is recorder.recorder_entry
    assert world.entry_calls == [(0, 0, 0, 0), (1, 1, 0, 1)]
    assert world.entry_displays == [(0, 0, 0), (1, 1, 1)]
    assert [type(scheduler).__name__ for scheduler in world.schedulers] == ["SglbenchRecorderScheduler"] * 2

    # Two logs, written by the real recorder. Both processes are the leader of their rank and have a blocker.
    logs = _rank_logs(launch)
    headers = [lines[0] for lines in logs]
    assert [(header["pid"], header["leader"], header["format"]) for header in headers] == [
        (4100, True, 3),
        (4101, True, 3),
    ]
    assert [header["ranks"] for header in headers] == [
        {
            "tp_size": 2,
            "pp_size": 1,
            "attn_dp_size": 2,
            "attn_dp_enabled": True,
            "attn_tp_rank": 0,
            "attn_cp_rank": 0,
            "attn_dp_rank": rank,
            "pp_rank": 0,
        }
        for rank in range(2)
    ]
    assert [header["checks"]["input_blocker"] for header in headers] == [True, True]
    assert headers[0]["limits"] == headers[1]["limits"]
    # The FAKE ranks ran in lockstep: line for line the same mode, the same step number and the same token
    # counts, one count for each rank and both equal, and no rank ever idled.
    first, second = (lines[1:] for lines in logs)
    assert len(first) == len(second) == world.batches_run > 20
    for own, other in zip(first, second, strict=True):
        assert (own["mode"], own["forward_iter"], own["global_num_tokens"]) == (
            other["mode"],
            other["forward_iter"],
            other["global_num_tokens"],
        )
        assert own["mode"] in ("EXTEND", "DECODE") and len(own["rids"]) == len(other["rids"]) > 0
        assert len(own["global_num_tokens"]) == 2 and len(set(own["global_num_tokens"])) == 1
        # Each log holds the requests of its own rank only.
        assert {content.parse_request_id(rid)["rank"] for rid in own["rids"]} == {0}
        assert {content.parse_request_id(rid)["rank"] for rid in other["rids"]} == {1}
    # Each FAKE process stamped its lines on a FAKE clock of its own.
    assert all(8000.0 <= line["launch_ts"] < line["out_ts"] < 9000.0 for line in first)
    assert all(9000.0 <= line["launch_ts"] < line["out_ts"] < 10000.0 for line in second)

    # Every call is SGLang's public batch call and nothing else: no request names a rank, and no field is
    # set beyond those of a run with one rank.
    assert len(world.engine_requests) == len(world.calls)
    for request, call in zip(world.engine_requests, world.calls, strict=True):
        assert request == {
            "prompt": None,
            "sampling_params": {"max_new_tokens": call.max_new_tokens, "temperature": 0.0, "ignore_eos": True},
            "stream": True,
            "more": {},
        }
    # Every call is balanced: slot by slot and rank inside. The request ids carry the planned rank, and the
    # FAKE DP controller, which deals the requests out in turn, placed each on that rank.
    for call in world.calls:
        slots, odd = divmod(len(call.rids), 2)
        assert odd == 0 and call.ranks == [0, 1] * slots
        parts = [content.parse_request_id(rid) for rid in call.rids]
        assert [(part["slot"], part["rank"]) for part in parts] == [
            (slot, rank) for slot in range(slots) for rank in (0, 1)
        ]
        assert call.rank_modes[0] == call.rank_modes[1] == call.modes
    # The self-test and the warm-up reached both ranks: seven free calls, the warm-up with one prompt per rank.
    assert [call.key for call in world.calls[:7]] == [(0, "free", number) for number in range(7)]
    assert [len(call.rids) for call in world.calls[:7]] == [4, 4, 4, 4, 4, 4, 2]
    assert [call.max_new_tokens for call in world.calls[:7]] == [1, 1, 1, 1, 1, 7, 5]
    order = [call.key[0] for call in world.calls if call.key[0]]
    assert order == sorted(order) and set(order) == {point.benchmark_id for point in built.points}

    producer = on_disk[0]["producer"]
    assert set(producer) == {
        *LAUNCHER_PRODUCER_KEYS,
        DP_PRODUCER_KEY,
        *DRIVER_PRODUCER_KEYS,
        "name",
        "artifact_format",
    }
    # The sizes SGLang resolved for the cell: its TP size counts the processes of both ranks.
    assert (producer["tp_size"], producer["attn_dp_size"], producer["moe"]["ep_size"]) == (2, 2, 2)
    assert producer[DP_PRODUCER_KEY] == DP_ATTENTION_FACTS
    assert (producer["staging"], producer["scheduler_processes"], producer["recorder_format"]) == (
        "balanced_batch_call",
        2,
        3,
    )
    assert producer["expert_keys"] == FAKE_EXPERTS and producer["restaged_points"] == []
    assert producer["limits"] == world.limits.as_dict()
    assert producer["self_test"] == ["batch_shape", "aligned_prefix", "unaligned_prompt", "seven_token_run"]
    kept = _fake_kept_texts(_fake_dataset_records())
    assert producer["content"] == {
        "source": "dataset_text_chains",
        "seed": "0",
        "token_range": FAKE_TOKEN_RANGE,
        "issued": producer["content"]["issued"],
        "dataset": launch.dataset,
        "pool": _fake_pool_counts(kept),
    }
    if mode == "decode":
        dataset_keys = ("path", "sha256", "collection_pool", "conversations")
        assert on_disk[0]["kvwarm"]["dataset"] == {key: launch.dataset[key] for key in dataset_keys}
        assert on_disk[1]["kvwarm"] == on_disk[0]["kvwarm"]
    else:
        assert "kvwarm" not in on_disk[0] and "kvwarm" not in on_disk[1]

    # What the launcher asked of the FAKE sglang package, in order: as for any text run.
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
    text_ids = {_FakeTokenizer.fake_id(word) for text in kept for word in text.split()}
    _assert_free_calls_are_text(world, text_ids)
    lines = _stderr_lines(capsys)
    rank_0, rank_1 = launch.rank_paths(2)
    assert lines[-1] == f"[sglbench] complete: wrote {rank_0}, {rank_1}, {launch.merged_path}"
    assert (
        "[sglbench] DP attention with 2 ranks: SGLANG_ENABLE_COLOCATED_BATCH_GEN is set to 1 for the scheduler "
        "processes, and every call carries the same number of prompts for every rank"
    ) in lines
    # No file of the recorder tree has a name the collector or the gate sweeps for.
    for path in (launch.pod / "sglbench-recorder").rglob("*"):
        assert not any(fnmatch.fnmatch(path.name, pattern) for pattern in FORBIDDEN_FILE_PATTERNS)


def test_end_to_end_local_control_broadcast_is_taken_from_the_command_line_and_recorded(tmp_path, monkeypatch):
    flag = "--enable-dp-attention-local-control-broadcast"

    launch = _launch_dep(tmp_path, monkeypatch, "decode", flags=[flag], real_compat=True)

    _assert_no_module_of_the_benchmark_is_replaced()
    on_disk = _assert_ranks_accepted(launch)
    # The flag is SGLang's: the launcher handed it on as it was typed, and SGLang resolved the field.
    assert launch.world.sglang_argv == ["--ep-size", "2", flag, *_fake_sglang_argv(1, 2, 2), STREAMING_FLAG]
    assert on_disk[0]["producer"][DP_PRODUCER_KEY] == {**DP_ATTENTION_FACTS, "local_control_broadcast": True}
    # The run is the same either way: the same calls, placed and released in the same way.
    assert on_disk[0]["producer"]["staging"] == "balanced_batch_call"
    assert all(call.ranks == [0, 1] * (len(call.rids) // 2) for call in launch.world.calls)


def test_local_control_broadcast_without_dp_attention_is_not_looked_at(tmp_path, monkeypatch):
    launch = _launch(
        tmp_path, monkeypatch, "prefill", flags=["--enable-dp-attention-local-control-broadcast"], real_compat=True
    )

    producer = _assert_accepted(launch)["producer"]
    assert launch.world.resolved["enable_dp_attention_local_control_broadcast"] is True
    assert (producer["attn_dp_size"], DP_PRODUCER_KEY in producer) == (1, False)


@pytest.mark.parametrize("mode", ["prefill", "decode", "agg"])
def test_a_dep_launch_with_three_ranks(tmp_path, monkeypatch, mode):
    launch = _launch_dep(tmp_path, monkeypatch, mode, dp_size=3)

    on_disk = _assert_ranks_accepted(launch, 3)
    world = launch.world
    assert launch.result_files() == [
        "benchmark.json",
        "benchmark_dp1.json",
        "benchmark_dp2.json",
        "benchmark_merged.json",
    ]
    producer = on_disk[0]["producer"]
    assert (producer["tp_size"], producer["attn_dp_size"], producer["scheduler_processes"]) == (3, 3, 3)
    assert producer[DP_PRODUCER_KEY] == DP_ATTENTION_FACTS
    assert world.entry_calls == [(rank, rank, 0, rank) for rank in range(3)]
    assert all(call.ranks == [0, 1, 2] * (len(call.rids) // 3) for call in world.calls)
    # Three logs in lockstep, with one token count for each of the three ranks.
    logs = [lines[1:] for lines in _rank_logs(launch)]
    assert len({len(lines) for lines in logs}) == 1
    for row in zip(*logs, strict=True):
        assert len({(line["mode"], line["forward_iter"], tuple(line["global_num_tokens"])) for line in row}) == 1
        assert len(row[0]["global_num_tokens"]) == 3 and len(set(row[0]["global_num_tokens"])) == 1
    # The grid is a grid of per-rank points: the one a run with one rank plans from the same FAKE limits.
    assert launch.planned().points == _planned(mode).points


def test_a_dep_launch_plans_and_sends_what_a_one_rank_launch_sends_to_its_one_rank(tmp_path, monkeypatch):
    # The launch with one rank comes first: a launcher with several ranks leaves SGLang's blocker variable set
    # in its process, which here is the test's.
    one = _launch_text(
        tmp_path / "one",
        monkeypatch,
        "prefill",
        word=None,
        prepare=lambda world, _pod: setattr(world, "experts", FAKE_EXPERTS),
    )
    dep = _launch_dep(tmp_path / "dep", monkeypatch, "prefill")

    on_disk, single = _assert_ranks_accepted(dep), _assert_accepted(one)
    assert on_disk[0]["grid_digest"] == single["grid_digest"]
    assert [call.key for call in dep.world.calls] == [call.key for call in one.world.calls]
    for call, other in zip(dep.world.calls, one.world.calls, strict=True):
        # Rank 0 gets the prompts of the run with one rank: content depends on the seed, the rank and the slot,
        # and the run id only names the requests. Rank 1 gets prompts of the same lengths and its own content.
        assert call.prompts[0::2] == other.prompts and call.max_new_tokens == other.max_new_tokens
        assert [len(prompt) for prompt in call.prompts[1::2]] == [len(prompt) for prompt in other.prompts]
        assert call.rank_modes == [other.modes, other.modes]
    # The keys of the producer block differ by the one block about DP attention.
    assert set(on_disk[0]["producer"]) - set(single["producer"]) == {DP_PRODUCER_KEY}
    assert set(single["producer"]) <= set(on_disk[0]["producer"])
    assert (single["producer"]["attn_dp_size"], single["producer"]["staging"]) == (1, "one_batch_call")


# ---------------------------------------------------------------------------------------------------------------
# DP attention: what the launcher admits
# ---------------------------------------------------------------------------------------------------------------

_DEP_ONLY = (
    "attn_dp_size={} is supported only as the DEP layout of a model with expert layers "
    "(tp_size == attn_dp_size and ep_size == tp_size): {}"
)
_ROUND_ROBIN = (
    "load_balance_method must be round_robin under DP attention (got {}); the requests of a call are placed "
    "on the ranks in turn"
)
_LM_HEAD = "enable_dp_lm_head is not supported (enable_dp_lm_head={})"
_BATCH_ENCODE = (
    "enable_tokenizer_batch_encode must be off under DP attention (got {}); SGLang then sends a call as one "
    "batch message outside its block region, so the ranks are not released together"
)

# (expert keys of the model, resolved fields beside the defaults)
DP_LAYOUTS_ACCEPTED = [
    # No DP attention: nothing is tested, whatever the other fields say.
    ({}, {}),
    ({}, {"tp_size": 4, "load_balance_method": "total_tokens", "enable_dp_lm_head": True}),
    ({}, {"tp_size": 2, "enable_tokenizer_batch_encode": True}),
    (FAKE_EXPERTS, {"tp_size": 2, "ep_size": 2, "enable_tokenizer_batch_encode": True}),
    (FAKE_EXPERTS, {"tp_size": 2, "ep_size": 2}),
    # The DEP layout: a model with expert layers, one scheduler process and one expert group for each rank.
    (FAKE_EXPERTS, {"attn_dp_size": 2, "tp_size": 2, "ep_size": 2}),
    (FAKE_EXPERTS, {"attn_dp_size": 4, "tp_size": 4, "ep_size": 4}),
    (
        FAKE_EXPERTS,
        {"attn_dp_size": 2, "tp_size": 2, "ep_size": 2, "enable_dp_attention_local_control_broadcast": True},
    ),
    ({"num_local_experts": 8}, {"attn_dp_size": 8, "tp_size": 8, "ep_size": 8}),
]
# The three sizes of a DEP layout with two ranks.
_DEP_SIZES = {"attn_dp_size": 2, "tp_size": 2, "ep_size": 2}
# (expert keys of the model, resolved fields beside the defaults, the whole refusal)
DP_LAYOUT_REFUSALS = [
    # A model without expert layers has no cell with DP attention.
    ({}, {"attn_dp_size": 2, "tp_size": 2, "ep_size": 2}, _DEP_ONLY.format(2, "the model has no expert layers")),
    (
        {},
        {"attn_dp_size": 2, "tp_size": 2},
        _DEP_ONLY.format(2, "the model has no expert layers; ep_size=1"),
    ),
    # Attention TP above 1 inside a DP rank: the TP size is a multiple of the number of ranks.
    (FAKE_EXPERTS, {"attn_dp_size": 2, "tp_size": 4, "ep_size": 4}, _DEP_ONLY.format(2, "tp_size=4")),
    (FAKE_EXPERTS, {"attn_dp_size": 2, "tp_size": 4, "ep_size": 2}, _DEP_ONLY.format(2, "tp_size=4; ep_size=2")),
    # A TP size below the number of ranks, which SGLang itself refuses.
    (FAKE_EXPERTS, {"attn_dp_size": 2}, _DEP_ONLY.format(2, "tp_size=1")),
    # Every rank holds a slice of every expert: no cell is planned for that under DP attention.
    (FAKE_EXPERTS, {"attn_dp_size": 2, "tp_size": 2}, _DEP_ONLY.format(2, "ep_size=1")),
    (FAKE_EXPERTS, {"attn_dp_size": 4, "tp_size": 4, "ep_size": 1}, _DEP_ONLY.format(4, "ep_size=1")),
    (
        {},
        {"attn_dp_size": 3, "tp_size": 6},
        _DEP_ONLY.format(3, "the model has no expert layers; tp_size=6; ep_size=1"),
    ),
    # A value of another type is never taken for the size.
    (FAKE_EXPERTS, {"attn_dp_size": 2, "tp_size": 2.0, "ep_size": 2}, _DEP_ONLY.format(2, "tp_size=2.0; ep_size=2")),
    (FAKE_EXPERTS, {"attn_dp_size": 2, "tp_size": 2, "ep_size": "2"}, _DEP_ONLY.format(2, "ep_size=2")),
    (FAKE_EXPERTS, {"attn_dp_size": 2, "tp_size": None, "ep_size": 2}, _DEP_ONLY.format(2, "tp_size=null; ep_size=2")),
    # Inside the DEP layout: how the DP controller places a request, and the output head.
    (
        FAKE_EXPERTS,
        {"attn_dp_size": 2, "tp_size": 2, "ep_size": 2, "load_balance_method": "total_tokens"},
        _ROUND_ROBIN.format("total_tokens"),
    ),
    (
        FAKE_EXPERTS,
        {"attn_dp_size": 2, "tp_size": 2, "ep_size": 2, "load_balance_method": "auto"},
        _ROUND_ROBIN.format("auto"),
    ),
    (
        FAKE_EXPERTS,
        {"attn_dp_size": 2, "tp_size": 2, "ep_size": 2, "load_balance_method": None},
        _ROUND_ROBIN.format("null"),
    ),
    (FAKE_EXPERTS, {"attn_dp_size": 2, "tp_size": 2, "ep_size": 2, "enable_dp_lm_head": True}, _LM_HEAD.format("true")),
    (FAKE_EXPERTS, {"attn_dp_size": 2, "tp_size": 2, "ep_size": 2, "enable_dp_lm_head": None}, _LM_HEAD.format("null")),
    (FAKE_EXPERTS, {"attn_dp_size": 2, "tp_size": 2, "ep_size": 2, "enable_dp_lm_head": 0}, _LM_HEAD.format(0)),
    # Inside the DEP layout: batch tokenization, with which SGLang sends a call as one message that its
    # block and unblock messages do not enclose.
    (
        FAKE_EXPERTS,
        {"attn_dp_size": 2, "tp_size": 2, "ep_size": 2, "enable_tokenizer_batch_encode": True},
        _BATCH_ENCODE.format("true"),
    ),
    (
        FAKE_EXPERTS,
        {"attn_dp_size": 2, "tp_size": 2, "ep_size": 2, "enable_tokenizer_batch_encode": None},
        _BATCH_ENCODE.format("null"),
    ),
    (
        FAKE_EXPERTS,
        {"attn_dp_size": 4, "tp_size": 4, "ep_size": 4, "enable_tokenizer_batch_encode": 0},
        _BATCH_ENCODE.format(0),
    ),
    # The settings are tested in this order, and the first that fails is named.
    (
        FAKE_EXPERTS,
        {**_DEP_SIZES, "enable_dp_lm_head": True, "enable_tokenizer_batch_encode": True},
        _LM_HEAD.format("true"),
    ),
]


@pytest.mark.parametrize(("experts", "resolved"), DP_LAYOUTS_ACCEPTED)
def test_no_dp_attention_and_the_dep_layout_pass(experts, resolved):
    view = _FakeView({**FAKE_RESOLVED_ARGUMENTS, **resolved})
    launcher._check_resolved(view)
    launcher._check_expert_parallel(view, experts)
    launcher._check_dp_attention(view, experts)


@pytest.mark.parametrize(("experts", "resolved", "refusal"), DP_LAYOUT_REFUSALS)
def test_every_other_layout_with_dp_attention_is_refused(experts, resolved, refusal):
    view = _FakeView({**FAKE_RESOLVED_ARGUMENTS, **resolved})

    with pytest.raises(launcher._Failed) as raised:
        launcher._check_dp_attention(view, experts)

    assert raised.value.reason == f"rejected: {refusal}"


def test_without_dp_attention_the_four_fields_of_dp_attention_are_not_read(tmp_path, monkeypatch):
    fields = {name: value for name, value in FAKE_RESOLVED_ARGUMENTS.items() if name not in DP_ONLY_FIELDS}
    launcher._check_dp_attention(_FakeView(fields), {})
    producer: dict = {}
    assert launcher._resolved_settings(_FakeView(fields), producer)["dp_size"] == 1

    def prepare(world, _pod_dir):
        for name in DP_ONLY_FIELDS:
            del world.resolved[name]

    # A whole launch with one rank on an SGLang without these fields runs as before.
    _assert_accepted(_launch(tmp_path, monkeypatch, "prefill", prepare=prepare))


@pytest.mark.parametrize("missing", ["tp_size", "ep_size", *DP_ONLY_FIELDS])
def test_with_dp_attention_every_field_the_checks_read_must_exist(tmp_path, monkeypatch, missing):
    def prepare(world, _pod_dir):
        del world.resolved[missing]

    launch = _launch_dep(tmp_path, monkeypatch, "prefill", prepare=prepare)

    assert launch.code == 70 and "build_engine" not in launch.world.events
    assert launch.artifact()["error"].startswith(
        f"internal_error: AttributeError: SGLang's resolved arguments have no field '{missing}'"
    )
    assert BLOCKER_ENV not in os.environ


def _reaches_the_check_of_dp_attention(experts: dict, resolved: dict) -> bool:
    """Whether a launch with these fields gets as far as the layout of DP attention.

    The expert-parallel size is tested first and has its own refusals: it must be 1, or the TP size of a
    model with expert layers. Typed here on its own.
    """
    ep_size, tp_size = resolved.get("ep_size", 1), resolved.get("tp_size", 1)
    if type(ep_size) is int and ep_size == 1:
        return True
    return bool(experts) and type(ep_size) is int and type(tp_size) is int and ep_size == tp_size


DP_LAYOUT_REFUSALS_AT_LAUNCH = [row for row in DP_LAYOUT_REFUSALS if _reaches_the_check_of_dp_attention(*row[:2])]


def test_the_layouts_refused_at_launch_are_of_every_kind():
    refusals = [refusal for _experts, _resolved, refusal in DP_LAYOUT_REFUSALS_AT_LAUNCH]
    assert len(refusals) == 16
    for part in (
        "the model has no expert layers",
        "tp_size=",
        "ep_size=1",
        "load_balance_method",
        "enable_dp_lm_head",
        "enable_tokenizer_batch_encode",
    ):
        assert any(part in refusal for refusal in refusals), part


@pytest.mark.parametrize(("experts", "resolved", "refusal"), DP_LAYOUT_REFUSALS_AT_LAUNCH)
def test_a_dp_attention_layout_the_benchmark_does_not_cover_ends_the_launch_with_64(
    tmp_path, monkeypatch, capsys, experts, resolved, refusal
):
    stale = b'{"stale": "rank 1 of an earlier run"}'

    def prepare(world, pod_dir):
        world.experts = experts
        world.resolved.update(resolved)
        (pod_dir / "benchmark_dp1.json").write_bytes(stale)

    dump = tmp_path / "resolved-config-node0.json"
    launch = _launch_text(
        tmp_path, monkeypatch, "decode", prepare=prepare, word="sharegpt_chain", flags=["--dump-config-to", str(dump)]
    )

    on_disk = _assert_failed_before_the_grid(launch, 64, f"rejected: {refusal}")
    # Refused after the model config was read and before anything was built, set or dumped.
    assert launch.world.events[-3:] == ["model_expert_keys", "write", "sleep"]
    assert "build_engine" not in launch.world.events and not dump.exists()
    assert BLOCKER_ENV not in os.environ
    # The number of ranks was never taken over: one failed rank file, at the output path, and the stale file
    # of another rank is gone.
    assert launch.result_files() == ["benchmark.json"]
    assert (on_disk["dp"], "attn_dp_size" in on_disk["producer"]) == ({"size": 1, "rank": 0}, False)
    assert on_disk["producer"]["expert_keys"] == experts
    assert _stderr_lines(capsys)[-2:] == [
        f"[sglbench] FAILED: rejected: {refusal}",
        f"[sglbench] wrote the failed artifact {launch.rank_path}",
    ]


def test_the_expert_parallel_size_is_tested_before_the_layout_of_dp_attention(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        world.experts = FAKE_EXPERTS
        world.resolved.update({"attn_dp_size": 2, "tp_size": 4, "ep_size": 2})

    launch = _launch_text(tmp_path, monkeypatch, "decode", prepare=prepare, word=None)

    _assert_failed_before_the_grid(launch, 64, f"rejected: {_EP_NOT_TP.format(2, 4)}")


@pytest.mark.parametrize(
    ("resolved", "refusal"),
    [
        # Whole copies of the model beside DP attention: refused as replicas are without it.
        ({"attn_dp_size": 2, "dp_size": 2}, "data-parallel replicas are not supported (dp_size=2)"),
        ({"attn_dp_size": 0}, "attn_dp_size must be a positive int (got 0)"),
        # DWDP switches DP attention on by itself and is refused by name.
        ({"attn_dp_size": 2, "dwdp_size": 2}, "DWDP is not supported (dwdp_size=2)"),
        ({"attn_dp_size": 2, "moe_dense_tp_size": 1}, "moe_dense_tp_size must not be set (got 1)"),
        ({"attn_dp_size": 2, "moe_dp_size": 2}, "moe_dp_size must be 1 (got 2)"),
        ({"attn_dp_size": 2, "moe_a2a_backend": "deepep"}, f"{_A2A} (moe_a2a_backend=deepep)"),
        ({"attn_dp_size": 2, "enable_eplb": True}, "EPLB is not supported (enable_eplb=true)"),
        (
            {"attn_dp_size": 2, "ep_num_redundant_experts": 2},
            "redundant experts are not supported (ep_num_redundant_experts=2)",
        ),
    ],
)
def test_the_settings_refused_without_dp_attention_are_refused_with_it(tmp_path, monkeypatch, resolved, refusal):
    def prepare(world, _pod_dir):
        world.resolved.update(resolved)

    launch = _launch_dep(tmp_path, monkeypatch, "decode", prepare=prepare)

    _assert_failed_before_the_grid(launch, 64, f"rejected: {refusal}")
    assert launch.world.events[-3:] == ["resolve", "write", "sleep"]
    assert BLOCKER_ENV not in os.environ


@pytest.mark.parametrize("real_compat", [False, True])
def test_seeded_ids_are_refused_for_a_dep_launch_as_for_any_model_with_expert_layers(
    tmp_path, monkeypatch, real_compat
):
    launch = _launch_dep(tmp_path, monkeypatch, "prefill", word="random", real_compat=real_compat)

    on_disk = _assert_failed_before_the_grid(launch, 64, f"rejected: {_SEEDED_WITH_EXPERTS.format('num_experts=60')}")
    assert launch.world.events[-2:] == (["from_server_args", "sleep"] if real_compat else ["write", "sleep"])
    assert on_disk["producer"]["content"]["source"] == "seeded_token_ids"
    # Refused before the number of ranks was taken over: one failed file, and the variable is not set.
    assert launch.result_files() == ["benchmark.json"] and BLOCKER_ENV not in os.environ


def test_a_model_without_expert_layers_is_refused_for_dp_attention_also_with_the_text_word(tmp_path, monkeypatch):
    launch = _launch(
        tmp_path,
        monkeypatch,
        "decode",
        tp_size=2,
        dp_size=2,
        env={ENV_CONTENT: "sharegpt_chain"},
        real_compat=True,
    )

    reason = "rejected: " + _DEP_ONLY.format(2, "the model has no expert layers; ep_size=1")
    on_disk = _assert_failed_before_the_grid(launch, 64, reason)
    assert launch.world.events[-2:] == ["from_server_args", "sleep"] and "Engine" not in launch.world.events
    assert on_disk["producer"]["content"]["source"] == "dataset_text_chains"


def test_end_to_end_batch_tokenization_is_refused_for_a_dep_launch_and_not_for_one_rank(tmp_path, monkeypatch):
    flag = "--enable-tokenizer-batch-encode"

    # One rank first: a launch with several ranks leaves SGLang's blocker variable set in the test's process.
    one = _launch(tmp_path / "one", monkeypatch, "prefill", flags=[flag], real_compat=True)
    _assert_accepted(one)
    assert one.world.resolved["enable_tokenizer_batch_encode"] is True

    launch = _launch_dep(tmp_path / "dep", monkeypatch, "decode", flags=[flag], real_compat=True)

    _assert_no_module_of_the_benchmark_is_replaced()
    # The flag is SGLang's and reached its parser; the launcher refused the field SGLang resolved from it.
    assert flag in launch.world.sglang_argv
    _assert_failed_before_the_grid(launch, 64, "rejected: " + _BATCH_ENCODE.format("true"))
    # Refused after the model config was read and before the engine was built or the variable was set.
    assert launch.world.events[-2:] == ["from_server_args", "sleep"] and "Engine" not in launch.world.events
    assert launch.result_files() == ["benchmark.json"] and BLOCKER_ENV not in os.environ


def test_end_to_end_chunks_without_a_rank_fail_a_dep_launch_at_its_first_call(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        # FAKE of an SGLang whose output records list no rank: the chunks then have no ``dp_rank``.
        world.rank_in_chunks = "no key"

    launch = _launch_dep(tmp_path, monkeypatch, "decode", prepare=prepare, real_compat=True)

    _assert_no_module_of_the_benchmark_is_replaced()
    (call,) = launch.world.calls
    reason = (
        f"self_test_failed: batch_shape: dp_rank_placement_mismatch: {call.rids[0]}: the engine reports no "
        "dp_rank, the request was planned for rank 0"
    )
    _assert_every_rank_failed(launch, 65, reason)
    assert launch.world.events[-2:] == ["Engine.shutdown", "sleep"]


def test_end_to_end_a_rank_that_is_no_int_fails_the_engine_call(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        world.rank_in_chunks = "as text"

    launch = _launch_dep(tmp_path, monkeypatch, "prefill", prepare=prepare, real_compat=True)

    (call,) = launch.world.calls
    reason = (
        f"self_test_failed: batch_shape: engine_call_failed: EngineCallError: request {call.rids[0]}: dp_rank is "
        "'0', not a rank"
    )
    _assert_every_rank_failed(launch, 65, reason)


def test_end_to_end_chunks_without_a_rank_do_not_matter_to_a_launch_with_one_rank(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        world.rank_in_chunks = "no key"

    launch = _launch(tmp_path, monkeypatch, "decode", tp_size=2, prepare=prepare, real_compat=True)

    _assert_no_module_of_the_benchmark_is_replaced()
    _assert_accepted(launch)


# ---------------------------------------------------------------------------------------------------------------
# DP attention: SGLang's variables
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("ranks", "name", "value", "refusal"), DP_ENV_REFUSALS)
def test_refused_variables_of_sglang_that_depend_on_the_number_of_ranks(ranks, name, value, refusal):
    with pytest.raises(launcher._Failed) as raised:
        launcher._check_dp_environment({name: value}, ranks)

    assert raised.value.reason == f"rejected: {refusal}"
    # The checks that run before resolution do not look at these variables.
    launcher._check_sglang_environment({name: value})


@pytest.mark.parametrize(("ranks", "name", "value"), DP_ENV_ACCEPTED)
def test_accepted_variables_of_sglang_that_depend_on_the_number_of_ranks(ranks, name, value):
    launcher._check_dp_environment({name: value}, ranks)
    launcher._check_dp_environment({}, ranks)


def test_every_variable_the_launcher_refuses_under_dp_attention_has_a_test():
    assert [name for name, _why in launcher._DP_REFUSED_VARIABLES] == [
        "SGLANG_SCHEDULER_SKIP_ALL_GATHER",
        "SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION",
    ]
    assert {name for ranks, name, _value, _refusal in DP_ENV_REFUSALS if ranks > 1} == {
        name for name, _why in launcher._DP_REFUSED_VARIABLES
    }
    assert launcher.ENV_INPUT_BLOCKER == BLOCKER_ENV


def test_without_dp_attention_the_blocker_variable_is_refused_once_the_arguments_are_resolved(tmp_path, monkeypatch):
    launch = _launch(tmp_path, monkeypatch, "decode", env={BLOCKER_ENV: "1"})

    _assert_failed_before_the_grid(launch, 64, f"rejected: {_BLOCKER}")
    # The number of ranks is SGLang's resolved value, so the refusal comes after resolution and before the
    # platform is looked at.
    assert launch.world.events == [
        "clock",
        "sglang_version",
        "load_plugins",
        "parse_server_args",
        "resolve",
        "write",
        "sleep",
    ]
    # The launcher did not touch the variable.
    assert os.environ[BLOCKER_ENV] == "1"


def test_the_variables_that_do_not_depend_on_the_ranks_and_the_resolved_arguments_are_refused_first(
    tmp_path, monkeypatch
):
    def prepare(world, _pod_dir):
        world.resolved["stream_interval"] = 2

    first = _launch(tmp_path / "a", monkeypatch, "decode", env={BLOCKER_ENV: "1"}, prepare=prepare)
    _assert_failed_before_the_grid(first, 64, "rejected: stream_interval must be 1 (got 2)")

    second = _launch(tmp_path / "b", monkeypatch, "decode", env={BLOCKER_ENV: "1", "SGLANG_RADIX_FORCE_MISS": "1"})
    _assert_failed_before_the_grid(second, 64, "rejected: SGLANG_RADIX_FORCE_MISS disables every cache hit")
    assert "resolve" not in second.world.events


@pytest.mark.parametrize("before", [None, "", "0", "false", "no", "1", "TRUE"])
def test_with_dp_attention_the_launcher_sets_the_blocker_variable_whatever_it_was(tmp_path, monkeypatch, before):
    seen = {}

    def prepare(world, _pod_dir):
        world.hooks["resolve"] = lambda: seen.update(at_resolution=os.environ.get(BLOCKER_ENV))
        world.hooks["build_engine"] = lambda: seen.update(at_the_build=os.environ.get(BLOCKER_ENV))

    env = {} if before is None else {BLOCKER_ENV: before}
    launch = _launch_dep(tmp_path, monkeypatch, "prefill", prepare=prepare, env=env)

    # The variable was the operator's until the layout was admitted, and "1" from then on.
    assert seen == {"at_resolution": before, "at_the_build": "1"}
    assert os.environ[BLOCKER_ENV] == "1"
    on_disk = _assert_ranks_accepted(launch)
    # Every FAKE scheduler process found it on when it was built.
    assert [lines[0]["checks"]["input_blocker"] for lines in _rank_logs(launch)] == [True, True]
    assert on_disk[0]["producer"][DP_PRODUCER_KEY]["input_blocker"] is True


@pytest.mark.parametrize(("ranks", "name", "value", "refusal"), [row for row in DP_ENV_REFUSALS if row[0] == 2])
def test_a_variable_refused_under_dp_attention_ends_the_launch_before_the_engine_is_built(
    tmp_path, monkeypatch, ranks, name, value, refusal
):
    launch = _launch_dep(tmp_path, monkeypatch, "decode", dp_size=ranks, env={name: value})

    _assert_failed_before_the_grid(launch, 64, f"rejected: {refusal}")
    assert launch.world.events[-3:] == ["resolve", "write", "sleep"]
    assert BLOCKER_ENV not in os.environ and launch.result_files() == ["benchmark.json"]


def test_a_variable_sglang_sets_while_it_resolves_is_seen_by_the_check_after_resolution(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        # FAKE of a resolution step that switches the exchange between the ranks off by itself, as SGLang's
        # DWDP step does.
        world.hooks["resolve"] = lambda: monkeypatch.setenv("SGLANG_SCHEDULER_SKIP_ALL_GATHER", "True")

    launch = _launch_dep(tmp_path, monkeypatch, "decode", prepare=prepare)

    _assert_failed_before_the_grid(launch, 64, f"rejected: {_SKIP_GATHER}")
    assert "build_engine" not in launch.world.events


def test_the_same_two_variables_do_not_stop_a_launch_with_one_rank(tmp_path, monkeypatch):
    env = {"SGLANG_SCHEDULER_SKIP_ALL_GATHER": "1", "SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION": "1"}

    launch = _launch(tmp_path, monkeypatch, "prefill", env=env)

    _assert_accepted(launch)
    assert BLOCKER_ENV not in os.environ


def test_scheduler_processes_without_the_blocker_are_refused_from_their_headers(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        # FAKE of scheduler processes that do not see the variable the launcher set.
        world.hooks["build_engine"] = lambda: os.environ.pop(BLOCKER_ENV)

    launch = _launch_dep(tmp_path, monkeypatch, "decode", prepare=prepare)

    reason = (
        "rejected: SGLANG_ENABLE_COLOCATED_BATCH_GEN must be set under DP attention (input_blocker=false); the "
        "launcher sets it before it builds the engine"
    )
    on_disk = _assert_failed_before_the_grid(launch, 64, reason)
    assert launch.world.events[-3:] == ["shutdown", "write", "sleep"]
    assert [scheduler.input_blocker for scheduler in launch.world.schedulers] == [None, None]
    # The number of ranks was known by then: a failed rank file for each of them, and no merged file.
    assert launch.result_files() == ["benchmark.json", "benchmark_dp1.json"]
    assert on_disk["producer"][DP_PRODUCER_KEY] == DP_ATTENTION_FACTS


# ---------------------------------------------------------------------------------------------------------------
# DP attention: the run settings, the producer block and the result files
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("tp_size", "dp_size"), [(1, 1), (2, 1), (2, 2), (4, 4)])
def test_the_resolved_sizes_reach_the_driver_and_the_producer_block(tp_size, dp_size):
    fields = {**FAKE_RESOLVED_ARGUMENTS, "tp_size": tp_size, "attn_dp_size": dp_size, "max_queued_requests": 64}
    producer: dict = {"sglang_version": "0.5.23"}

    settings = launcher._resolved_settings(_FakeView(fields), producer)

    # One recorder header is waited for from every scheduler process: SGLang's TP size, which under DP
    # attention counts the processes of all ranks. The number of ranks is SGLang's attention-DP size.
    assert settings == {
        "expected_scheduler_processes": tp_size,
        "dp_size": dp_size,
        "prefill_max_requests": None,
        "max_queued_requests": 64,
    }
    assert producer == {
        "sglang_version": "0.5.23",
        "tp_size": tp_size,
        "attn_dp_size": dp_size,
        "kv_cache_dtype": "auto",
    }
    # They are fields of the driver's settings record, the second with the default of a run with one rank.
    names = {field.name: field.default for field in dataclasses.fields(driver.RunSettings)}
    assert set(settings) <= set(names) and names["dp_size"] == 1


@pytest.mark.parametrize("dp_size", [0, None, True, "2", 2.0])
def test_an_attention_dp_size_that_is_not_a_positive_int_is_never_taken_over(dp_size):
    fields = {**FAKE_RESOLVED_ARGUMENTS, "attn_dp_size": dp_size}
    producer: dict = {}

    with pytest.raises(TypeError, match="SGLang's resolved attn_dp_size is .*, not a positive int"):
        launcher._resolved_settings(_FakeView(fields), producer)

    assert producer == {}


@pytest.mark.parametrize("choice", ["auto", "fp8_e4m3", "bfloat16", "bf16"])
def test_the_resolved_kv_cache_dtype_reaches_the_producer_block_as_sglang_left_it(choice):
    fields = {**FAKE_RESOLVED_ARGUMENTS, "kv_cache_dtype": choice}
    producer: dict = {}

    settings = launcher._resolved_settings(_FakeView(fields), producer)

    # It is recorded and decides nothing: no setting of the run follows from it.
    assert producer == {"tp_size": 1, "attn_dp_size": 1, "kv_cache_dtype": choice}
    assert "kv_cache_dtype" not in settings
    assert "kv_cache_dtype" not in [name for name, _accepts, _text in launcher._RESOLVED_CHECKS]


@pytest.mark.parametrize("value", [None, "", 8, True, ["auto"]])
def test_a_resolved_kv_cache_dtype_that_is_no_name_is_an_error_and_nothing_is_written(value):
    fields = {**FAKE_RESOLVED_ARGUMENTS, "kv_cache_dtype": value}
    producer: dict = {}

    with pytest.raises(TypeError, match="SGLang's resolved kv_cache_dtype is .*, not the name of a dtype"):
        launcher._resolved_settings(_FakeView(fields), producer)

    assert producer == {}


def test_resolved_arguments_without_the_kv_cache_dtype_are_an_error_that_names_the_field():
    fields = {name: value for name, value in FAKE_RESOLVED_ARGUMENTS.items() if name != "kv_cache_dtype"}
    producer: dict = {}

    with pytest.raises(AttributeError, match="SGLang's resolved arguments have no field 'kv_cache_dtype'"):
        launcher._resolved_settings(_FakeView(fields), producer)

    assert producer == {}


@pytest.mark.parametrize("broadcast", [False, True])
def test_the_blocker_is_switched_on_in_the_mapping_it_is_given_and_recorded(broadcast):
    fields = {**FAKE_RESOLVED_ARGUMENTS, "enable_dp_attention_local_control_broadcast": broadcast}
    environ = {BLOCKER_ENV: "0", "FAKE_OTHER": "kept"}
    producer: dict = {"attn_dp_size": 2}

    launcher._switch_on_input_blocker(_FakeView(fields), environ, producer)

    assert environ == {BLOCKER_ENV: "1", "FAKE_OTHER": "kept"}
    assert producer == {
        "attn_dp_size": 2,
        DP_PRODUCER_KEY: {**DP_ATTENTION_FACTS, "local_control_broadcast": broadcast},
    }
    assert type(producer[DP_PRODUCER_KEY]["local_control_broadcast"]) is bool
    # The block is JSON as it stands, and its keys are no keys of the driver or of the artifact module.
    assert json.loads(json.dumps(producer[DP_PRODUCER_KEY])) == producer[DP_PRODUCER_KEY]
    assert launcher.PLACEMENT_ROUND_ROBIN_CHECKED == "round_robin_checked"


@pytest.mark.parametrize("broadcast", [None, 1, 0, "true"])
def test_a_broadcast_field_that_is_not_a_bool_is_an_error_and_nothing_is_set(broadcast):
    fields = {**FAKE_RESOLVED_ARGUMENTS, "enable_dp_attention_local_control_broadcast": broadcast}
    environ: dict = {}
    producer: dict = {}

    with pytest.raises(TypeError, match="enable_dp_attention_local_control_broadcast is .*, not a bool"):
        launcher._switch_on_input_blocker(_FakeView(fields), environ, producer)

    assert environ == {} and producer == {}


def test_a_broadcast_field_that_is_not_a_bool_ends_the_launch_as_an_internal_error(tmp_path, monkeypatch):
    def prepare(world, _pod_dir):
        world.resolved["enable_dp_attention_local_control_broadcast"] = "yes"

    launch = _launch_dep(tmp_path, monkeypatch, "prefill", prepare=prepare)

    reason = (
        "internal_error: TypeError: SGLang's resolved enable_dp_attention_local_control_broadcast is 'yes', not a bool"
    )
    on_disk = _assert_failed_before_the_grid(launch, 70, reason)
    assert "build_engine" not in launch.world.events and BLOCKER_ENV not in os.environ
    # The sizes were taken over before: both ranks have a failed file, and the block about DP attention,
    # which is written with the variable, is not there.
    assert launch.result_files() == ["benchmark.json", "benchmark_dp1.json"]
    assert on_disk["producer"]["attn_dp_size"] == 2 and DP_PRODUCER_KEY not in on_disk["producer"]


def _assert_every_rank_failed(launch: _Launch, code: int, reason: str, dp_size: int = 2) -> list[dict]:
    """A failed run with several ranks: a failed rank file for every rank with the same reason, no merged file,
    and both checkers refuse."""
    _assert_failed(launch, code, reason)
    rank_paths = launch.rank_paths(dp_size)
    assert launch.result_files() == [path.name for path in rank_paths]
    on_disk = [json.loads(path.read_text(encoding="utf-8")) for path in rank_paths]
    for rank, payload in enumerate(on_disk):
        assert (payload["status"], payload["valid"], payload["usable"]) == ("failed", False, False)
        assert (payload["error"], payload["dp"]) == (reason, {"size": dp_size, "rank": rank})
        assert (payload["run_id"], payload["producer"]) == (on_disk[0]["run_id"], on_disk[0]["producer"])
    gate = _gate([str(path) for path in rank_paths], launch.cell.workload_kind)
    assert gate.returncode == GATE_INVALID
    return on_disk


def test_a_point_that_fails_in_a_dep_launch_leaves_a_failed_file_for_every_rank(tmp_path, monkeypatch, capsys):
    victim = _planned("prefill").points[3]

    def prepare(world, _pod_dir):
        world.call_faults[(victim.benchmark_id, "meas", 0)] = "engine_error"

    launch = _launch_dep(tmp_path, monkeypatch, "prefill", prepare=prepare)

    reason = (
        f"benchmark_id={victim.benchmark_id}: benchmark point failed: engine_call_failed: EngineCallError: "
        "FAKE: the engine call timed out"
    )
    on_disk = _assert_every_rank_failed(launch, 65, reason)
    assert [payload["coverage"]["completed_points"] for payload in on_disk] == [victim.benchmark_id - 1] * 2
    assert launch.world.events[-3:] == ["shutdown", "write", "sleep"]
    rank_0, rank_1 = launch.rank_paths(2)
    assert _stderr_lines(capsys)[-2:] == [
        f"[sglbench] FAILED: {reason}",
        f"[sglbench] wrote the failed artifact {rank_0}, {rank_1}",
    ]


@pytest.mark.parametrize("stage", ["self-test", "point"])
def test_a_request_that_ran_on_another_rank_than_planned_fails_a_dep_launch(tmp_path, monkeypatch, stage):
    victim = _planned("decode").points[2]
    key = (0, "free", 1) if stage == "self-test" else (victim.benchmark_id, "run", 0)

    def prepare(world, _pod_dir):
        # FAKE: one more request reached SGLang's DP controller before this call, so the turn of every
        # request of the call moved on by one rank.
        world.call_faults[key] = "stray_request"

    launch = _launch_dep(tmp_path, monkeypatch, "decode", prepare=prepare)

    call = next(call for call in launch.world.calls if call.key == key)
    assert call.ranks == [1, 0] * (len(call.rids) // 2)
    detail = (
        f"dp_rank_placement_mismatch: {call.rids[0]}: the engine reports dp_rank 1, the request was planned for rank 0"
    )
    if stage == "self-test":
        reason = f"self_test_failed: aligned_prefix: {detail}"
    else:
        reason = f"benchmark_id={victim.benchmark_id}: benchmark point failed: {detail}"
    on_disk = _assert_every_rank_failed(launch, 65, reason)
    # The call was the last one: nothing is staged again after a request ran on another rank.
    assert launch.world.calls[-1] is call
    assert on_disk[0]["producer"]["restaged_points"] == []


def test_a_cache_miss_on_one_rank_stages_the_point_again_on_every_rank(tmp_path, monkeypatch):
    victim = next(point for point in _planned("decode").points[1:] if point.cached_tokens > 8)

    def prepare(world, _pod_dir):
        # FAKE: rank 1 alone dropped its cache just before the run of the point.
        world.call_faults[(victim.benchmark_id, "run", 0)] = "evict@1"

    launch = _launch_dep(tmp_path, monkeypatch, "decode", prepare=prepare)

    on_disk = _assert_ranks_accepted(launch)
    assert on_disk[0]["producer"]["restaged_points"] == [victim.benchmark_id]
    calls = launch.calls_of(victim.benchmark_id)
    assert [call.key[1:] for call in calls] == [("seed", 0), ("run", 0), ("seed", 1), ("run", 1)]
    # Each of the four calls went to both ranks, the second attempt too.
    assert all(call.ranks == [0, 1] * victim.batch_size for call in calls)


def _launch_with_a_miss_on_rank_1_above_the_chunk_budget(tmp_path: Path, monkeypatch, replay: bool = True) -> tuple:
    """A DEP launch in which rank 1 alone drops its cache just before the run of a large decode point.

    The point is the decode point with the most tokens to compute once its cache is gone, more than one
    extend batch of a rank holds. ``replay`` says whether the FAKE prefill graphs can replay a step, which
    decides whether the FAKE engine runs a decode step beside an extend batch as an extend batch. Returns the
    launch, the point and the call of its first run.
    """
    victim = max(_planned("decode").points, key=lambda point: point.batch_size * point.cached_tokens)
    assert victim.batch_size * (victim.cached_tokens - grid.DECODE_PROMPT_LEAD) > _fake_limits(1).chunked_prefill_size

    def prepare(world, _pod_dir):
        world.call_faults[(victim.benchmark_id, "run", 0)] = "evict@1"
        world.prefill_graphs_replay = replay

    launch = _launch_dep(tmp_path, monkeypatch, "decode", prepare=prepare)
    (run,) = [call for call in launch.world.calls if call.key == (victim.benchmark_id, "run", 0)]
    return launch, victim, run


def test_a_cache_miss_on_one_rank_whose_neighbour_decodes_meanwhile_is_staged_again(tmp_path, monkeypatch):
    launch, victim, run = _launch_with_a_miss_on_rank_1_above_the_chunk_budget(tmp_path, monkeypatch)

    # Rank 1 computed its whole prompts again, in more extend batches than rank 0 has decode steps. Rank 0 hit
    # its cache: one extend batch, and then its six decode steps, each beside an extend batch of rank 1. The
    # FAKE engine ran those six steps as SGLang does, as extend batches of one token per request.
    batch = victim.batch_size
    assert run.rank_modes[0] == ["EXTEND"] * 7
    assert run.rank_modes[1][:8] == ["EXTEND"] * 8 and run.rank_modes[1][-6:] == ["DECODE"] * 6
    # The real recorder wrote them as what they were: the log of rank 0 holds seven EXTEND lines for the run
    # and no DECODE line. Each of the six holds exactly the requests of rank 0, with one computed token for
    # each after its prompt and the output tokens before, and the count of a decode batch for its rank.
    first = _rank_logs(launch)[0][1:]
    mine = [line for line in first if set(line["rids"]) & set(run.rids)]
    assert [line["mode"] for line in mine] == ["EXTEND"] * 7
    for step, line in enumerate(mine[1:]):
        assert sorted(line["rids"]) == sorted(run.rids[0::2])
        assert (line["extend_lens"], line["extend_num_tokens"]) == ([1] * batch, batch)
        assert line["prefix_lens"] == [len(prompt) + step for prompt in run.prompts[0::2]]
        assert line["global_num_tokens"][0] == batch < line["global_num_tokens"][1]

    # The driver saw the missed cache in the answers of rank 1 before it waited for decode batches, staged
    # the point again on both ranks and kept the second attempt. The log wait never ran to its deadline.
    on_disk = _assert_ranks_accepted(launch)
    assert on_disk[0]["producer"]["restaged_points"] == [victim.benchmark_id]
    calls = launch.calls_of(victim.benchmark_id)
    assert [call.key[1:] for call in calls] == [("seed", 0), ("run", 0), ("seed", 1), ("run", 1)]
    assert calls[-1].rank_modes == [["EXTEND"] + ["DECODE"] * 6] * 2
    assert driver.POLL_SECONDS not in launch.world.clock.sleeps


@pytest.mark.parametrize("replay", [True, False], ids=["decode-steps-as-extend", "decode-batches-kept"])
def test_a_rank_that_idles_while_another_one_works_writes_idle_lines_the_run_does_not_trip_over(
    tmp_path, monkeypatch, replay
):
    launch, victim, run = _launch_with_a_miss_on_rank_1_above_the_chunk_budget(tmp_path, monkeypatch, replay)

    # Rank 1 computed its whole prompts again, in more extend batches than the chunk budget lets rank 0 need,
    # and rank 0 had nothing left to run while rank 1 still worked. On a FAKE engine whose prefill graphs
    # cannot replay a step, rank 0 kept its six decode batches meanwhile.
    assert len(run.rank_modes[1]) > len(run.rank_modes[0]) == 7
    assert run.rank_modes[0] == (["EXTEND"] * 7 if replay else ["EXTEND"] + ["DECODE"] * 6)
    first, second = (lines[1:] for lines in _rank_logs(launch))
    idle = [line for line in first if line["mode"] == "IDLE"]
    assert len(idle) == len(run.rank_modes[1]) - len(run.rank_modes[0]) and not [
        line for line in second if line["mode"] == "IDLE"
    ]
    # The real recorder wrote each idle batch as a line without a request and without a fault: rank 0 ran
    # nothing in that step and rank 1 did.
    for line in idle:
        assert (line["rids"], line["extend_num_tokens"], line["prefix_lens"], line["extend_lens"]) == (
            [],
            None,
            None,
            None,
        )
        assert line["global_num_tokens"][0] == 0 < line["global_num_tokens"][1]
    assert len(first) == len(second) and [line["forward_iter"] for line in first] == [
        line["forward_iter"] for line in second
    ]
    # The driver read past the idle lines, saw the missed cache in the answers of rank 1 and staged the point
    # again on both ranks. The run is complete.
    on_disk = _assert_ranks_accepted(launch)
    assert on_disk[0]["producer"]["restaged_points"] == [victim.benchmark_id]


def test_the_soft_deadline_ends_a_dep_launch_with_a_partial_file_for_every_rank(tmp_path, monkeypatch):
    planned = _planned("decode")
    last_done = planned.points[1]

    def prepare(world, _pod_dir):
        world.call_faults[(last_done.benchmark_id, "run", 0)] = "deadline"

    launch = _launch_dep(tmp_path, monkeypatch, "decode", prepare=prepare)

    assert launch.code == 66
    assert launch.result_files() == ["benchmark.json", "benchmark_dp1.json", "benchmark_merged.json"]
    for rank, path in enumerate(launch.rank_paths(2)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert (payload["status"], payload["stop_reason"], payload["dp"]) == (
            "partial",
            "timeout",
            {"size": 2, "rank": rank},
        )
        assert payload["coverage"]["completed_points"] == last_done.benchmark_id
    assert _gate([str(path) for path in launch.rank_paths(2)], "decode").returncode == GATE_INVALID


# Names the artifact module can be asked to derive rank files from, also ones AISim never renders.
OUTPUT_NAMES = [
    "benchmark.json",
    "benchmark_results.json",
    "benchmark",
    "a.b.json",
    "x_dp1.json",
    "x_dp.json",
    ".hidden.json",
    ".hidden",
    "benchmark_merged.json",
    "1.2",
]


@pytest.mark.parametrize("name", OUTPUT_NAMES)
def test_the_name_of_a_rank_file_is_read_off_the_artifact_module(tmp_path, name):
    output = str(tmp_path / name)
    head, tail = launcher._rank_file_frame(output)
    rank_0, merged = launcher._result_paths(output)
    assert (rank_0, merged) == (output, artifact.merged_output_path(output))
    for rank in (1, 2, 9, 10, 11, 123, 4096):
        # The file the writer would write for this rank is one the launcher takes for a rank file.
        written = artifact.rank_output_path(output, rank)
        assert Path(written).name == f"{head}{rank}{tail}"
        assert launcher._is_rank_file_name(Path(written).name, head, tail)
        assert launcher._is_result_file(output, written)
    # The file of rank 0 and the merged file are result files and have no name of a rank above 0.
    for own in (rank_0, merged):
        assert launcher._is_result_file(output, own) and not launcher._is_rank_file_name(Path(own).name, head, tail)
    # A name with nothing, or with more than digits, in the place of the rank is none. The last one is a digit
    # of another script, which is no digit of a rank.
    for middle in ("", "x", "1x", "x1", "-1", "1.5", " 1", "1 ", chr(0x661)):
        assert chr(0x661).isdigit() and not launcher._is_rank_file_name(f"{head}{middle}{tail}", head, tail), middle
    assert not launcher._is_rank_file_name(f"x{head}1{tail}", head, tail)
    assert not launcher._is_rank_file_name(f"{head}1{tail}x", head, tail)
    # The same name in another directory is another file.
    assert not launcher._is_result_file(output, str(tmp_path / "other" / f"{head}1{tail}"))
    assert not launcher._is_result_file(output, str(tmp_path / "other" / name))


def test_the_old_outputs_are_the_rank_files_of_every_rank_and_the_merged_file(tmp_path):
    output = tmp_path / "pod" / "benchmark.json"
    # A directory that does not exist yet holds nothing to delete beyond the two fixed names.
    assert launcher._old_outputs(str(output)) == [str(output), str(output.with_name("benchmark_merged.json"))]

    output.parent.mkdir()
    for name in (*OTHER_FILES, *STALE_RESULT_FILES):
        (output.parent / name).parent.mkdir(parents=True, exist_ok=True)
        (output.parent / name).write_bytes(b"{}")
    assert launcher._old_outputs(str(output)) == [
        str(output),
        str(output.with_name("benchmark_merged.json")),
        *(str(output.with_name(name)) for name in sorted(STALE_RESULT_FILES) if name.startswith("benchmark_dp")),
    ]
    launcher._remove_old_outputs(str(output))
    left = sorted(str(path.relative_to(output.parent)) for path in output.parent.rglob("*") if path.is_file())
    assert left == sorted(OTHER_FILES)


@pytest.mark.parametrize(
    "name", ["benchmark_dp1.json", "benchmark_dp2.json", "benchmark_dp17.json", "./benchmark_dp3.json"]
)
def test_a_dump_path_that_is_the_file_of_another_rank_is_refused(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)

    options = _options(["--benchmark-output-path", f"{tmp_path}/benchmark.json", "--dump-config-to", name])

    assert options.rejection == f"--dump-config-to {name!r} is one of the result files of the run"


@pytest.mark.parametrize(
    "name",
    [
        "resolved-config-node0.json",
        "benchmark_dp.json",
        "benchmark_dpx.json",
        "benchmark_dp1.json.txt",
        "benchmark_dp1_merged.json",
        "other/benchmark_dp1.json",
        "other/benchmark.json",
    ],
)
def test_a_dump_path_that_only_looks_like_a_result_file_is_accepted(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)

    options = _options(["--benchmark-output-path", f"{tmp_path}/benchmark.json", "--dump-config-to", name])

    assert options.rejection is None and options.dump_config_to == name


def test_the_rank_files_of_an_earlier_launch_with_more_ranks_are_gone_after_the_next_launch(tmp_path, monkeypatch):
    first = _launch_dep(tmp_path, monkeypatch, "decode", dp_size=3)
    _assert_ranks_accepted(first, 3)
    assert first.result_files() == [
        "benchmark.json",
        "benchmark_dp1.json",
        "benchmark_dp2.json",
        "benchmark_merged.json",
    ]
    first_run = first.artifact()["run_id"]

    # A second launcher process starts without the variable the first one set for itself.
    monkeypatch.delenv(BLOCKER_ENV)
    second = _launch_dep(tmp_path / "again", monkeypatch, "decode", dp_size=2, pod_dir=first.pod)

    assert second.code == 0 and second.pod == first.pod
    assert second.result_files() == ["benchmark.json", "benchmark_dp1.json", "benchmark_merged.json"]
    for path in second.rank_paths(2):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["run_id"] != first_run and payload["dp"]["size"] == 2


def test_the_flags_of_dp_attention_go_to_sglang_as_they_were_typed():
    argv = [
        "--tp-size",
        "2",
        "--attn-dp-size",
        "2",
        "--ep-size=2",
        "--enable-dp-attention-local-control-broadcast",
        "--load-balance-method",
        "round_robin",
    ]

    options = _options(argv)

    # None of them is a flag of the launcher: all pass to SGLang's parser, in their order.
    assert (options.rejection, options.sglang_argv) == (None, [*argv, STREAMING_FLAG])
    own = {action.dest for action in launcher._build_parser()._actions}
    assert not {name for name in own if "dp" in name.split("_")}


def test_the_module_says_what_ran_with_dp_attention_and_holds_the_deviation_note_of_the_blocker():
    docstring = " ".join(launcher.__doc__.split())
    assert "With DP attention this launcher ran on engines on 2026-11-18, with SGLang 0.5.22 and 0.5.23" in docstring
    assert "--tp-size 2 --ep-size 2 --attn-dp-size 2" in docstring
    # It says what did not run, and that the spike of the same day was not a run of the launcher.
    assert "Not run on an engine under DP attention: more than four ranks, and four ranks in any cell but" in docstring
    assert "The launcher did not run in it." in docstring
    note = " ".join(launcher._switch_on_input_blocker.__doc__.split())
    for part in (
        "Deviation note (SGLang's input blocker is switched on under DP attention).",
        "What differs:",
        "Direction of the error:",
        "Sources:",
        "How to retire it:",
        "In the runs of 2026-11-18 (module docstring) the launcher set the variable",
    ):
        assert part in note, part
    # The four modules say the same about what ran: each names the day, the two releases and the layout.
    for module in (compat, driver, recorder):
        text = " ".join(module.__doc__.split())
        assert "2026-11-18" in text and "0.5.22 and 0.5.23" in text, module.__name__
        assert "--tp-size 2 --ep-size 2 --attn-dp-size 2" in text, module.__name__
    assert "Run on engines with DP attention on 2026-11-18 through sglbench_launcher.py" in " ".join(
        compat.__doc__.split()
    )
    # No sentence of the old state is left in any of them.
    for module in (launcher, compat, driver, recorder):
        text = " ".join(Path(module.__file__).read_text(encoding="utf-8").split())
        for old in ("has not run on an engine", "Not run through this", "Not seen through this module"):
            assert old not in text, (module.__name__, old)
    assert "No refusal of this function was run on an engine" in " ".join(launcher._check_dp_attention.__doc__.split())


# ---------------------------------------------------------------------------------------------------------------
# What a launch with one attention-DP rank sends and writes. Without the low end of the prefill new-token axis
# it is what the launcher sent and wrote before it knew DP attention; with it, the same and the added points.
# ---------------------------------------------------------------------------------------------------------------

# A FAKE run id that is the same in every launch, so that the request ids and the result files repeat.
FAKE_FIXED_RUN_ID = "5eedfacade5eedfacade5eedfacade00"

# name: (mode, page size, FAKE scheduler processes, content word, expert keys of the FAKE model, SGLang flags)
ONE_RANK_LAUNCHES = {
    "seeded-prefill-tp1": ("prefill", 1, 1, None, {}, []),
    "seeded-agg-tp2": ("agg", 16, 2, None, {}, []),
    "text-prefill-tp1": ("prefill", 64, 1, "sharegpt_chain", {}, []),
    "text-decode-tep2": ("decode", 1, 2, None, FAKE_EXPERTS, ["--ep-size", "2"]),
}

# SHA-256 of what each of those launches sent, wrote as result files and logged (``_one_rank_digests``) when
# its prefill new-token axis is planned as it was before the axis had a low end: 1 in front of the graph axis
# and nothing more (``_plan_without_the_low_end``).
# They were taken once from the launcher and the accessor module as they stood before DP attention, loaded
# beside the driver and recorder of the change that brought it: the two files of the commit before that change,
# with ``driver.RANK``, a constant the driver no longer has, typed as 0. The result files are hashed without
# ``producer.attn_dp_size``, the one key the launcher wrote since then.
# The digests of the calls are still those. The digests of the result files and of the logs were typed in again
# when the recorder's header gained the dtype of the KV pool (record format 3). A result file then holds
# ``producer.kv_cache_dtype``, ``producer.kv_cache_stored_dtype`` and the format number 3, and the header of a
# log holds the dtype and that number. These digests were taken from the launcher, the driver and the recorder
# as they stand. Each launch was first hashed with those values taken out again, the two keys gone and the
# format number 2 in its place, and gave the digests of before; that was done once, by a script that is not
# kept in this repository. A change that is meant to alter what a launch with one rank sends or writes for the
# points it planned then has to retype them, and the digests of ``ONE_RANK_DIGESTS`` with them.
ONE_RANK_DIGESTS_WITHOUT_THE_LOW_END = {
    "seeded-prefill-tp1": {
        "calls": "4481acf436e2e4105a21581e8e6901533e0d4467aafe93e81c6be8f682a3d9fb",
        "results": "98dfd4f34a1baa39e6b88c7a78d321c571c2c8aa4f3bd72ec2bc03ed1c589abd",
        "logs": "e73a5e8cc56413f2829080870cd83ed730451d3b05af8b11358aead6693d20f2",
    },
    "seeded-agg-tp2": {
        "calls": "177a96a7adf7ba38b0baec5d391abcea06451cc37c22e6ec8ddc456729f07450",
        "results": "98e05b99242bdb7865ba725c7a2302d0a2cd1d7454b448ba43e55661683b9215",
        "logs": "523c35e0e3f209158c849afe50b926fbac746a42edb1f8e3196ee9d6694d5df5",
    },
    "text-prefill-tp1": {
        "calls": "f3e97d41680895354ed4481090e801ee83deaca5171e42fb6276ec895bcc52bb",
        "results": "a9dab74e6678cbadb01299daa44662dbb52d1f126b2fe715d8b6f1a9b9f9f852",
        "logs": "2df2fdfefc18fb68db18e351b118deabc97c89aac13e849a24e112b5e9f47f0f",
    },
    "text-decode-tep2": {
        "calls": "bfcb8e156c7e0039f868abd6e0db62d6ac88a378e2de14e841cb5fc8a5860799",
        "results": "64a3fb56548f3b67c27bc16ce9322fc1425f7e0cde7f9ffa7e8e155c8935a9c4",
        "logs": "e8ee4f06dbe6efa12e86bfa3190cc0a3751ffaea043d514401fd38d7e1f30042",
    },
}

# What the low end adds to each of those launches: the batch size and the cached tokens per request of the
# points of 2 new tokens per request, in the order they run. Typed here from the rules of the axes. The FAKE
# prefill graph sizes start at 4, so a batch of 2 tokens is the smallest that runs with a graph, and 2 joins 1
# on the axis. The small caps keep batch 1 and 2 for it, and three cached prefixes for each batch size: the
# candidates are 0, the cached unit doubled up to the longest prefix, and the longest prefix, and the cap
# keeps the first, the middle one and the last. Beside 2 new tokens the longest prefix below the FAKE input
# limit is 416 tokens of 428 at page sizes 1 and 16 (unit 16: 0, 16, 32, 64, 128, 256, 416) and 768 tokens of
# 812 at page size 64 (unit 64: 0, 64, 128, 256, 512, 768, where the middle of six is the fourth). The larger
# batch runs first, and of its prefixes the longer one. A decode run has no prefill block.
ONE_RANK_LOW_END_POINTS = {
    "seeded-prefill-tp1": [(2, 416), (2, 64), (2, 0), (1, 416), (1, 64), (1, 0)],
    "seeded-agg-tp2": [(2, 416), (2, 64), (2, 0), (1, 416), (1, 64), (1, 0)],
    "text-prefill-tp1": [(2, 768), (2, 256), (2, 0), (1, 768), (1, 256), (1, 0)],
    "text-decode-tep2": [],
}

# The same three digests of the launches as they are planned now, with the low end. A digest cannot be typed
# from a rule: these were taken from the launcher as it stands. Three tests show that they are the digests of
# the right launch. Without the low end the launch still gives the digests above. With it the launch is
# accepted row by row against the plan. And the two launches are compared: they differ by the points of
# ``ONE_RANK_LOW_END_POINTS`` and by what follows from them
# (``test_the_low_end_adds_its_points_to_a_launch_with_one_rank_and_leaves_the_other_points_as_they_were``).
# The decode launch plans what it planned before and keeps its digests. A change to the low end has to retype
# these and no digest above.
ONE_RANK_DIGESTS = {
    "seeded-prefill-tp1": {
        "calls": "bd90820b51553944b7d208ba7e22dc7aca457f2586f25ee3437e2d5409c4293a",
        "results": "d1376b085b296ddba38ca1887152b10d2b693b650ede9c497a5c94dcdfcd2143",
        "logs": "1e30c05c700432a357efbb3ff12326e749dd9b382e1a7a56ec39de757a35f644",
    },
    "seeded-agg-tp2": {
        "calls": "0a62c219b7196115f933ef84001fe4a5fcf81b441c9722af0c7d6297a737acf3",
        "results": "4c14a43bfada0c62a01ef63057b0db8f7f6538c37d5fe7855b78154d87e2cd08",
        "logs": "e1aa5403e6d61ae47b6f7c1906fe2469fa06e6854781048f4c34b1c44e041bd2",
    },
    "text-prefill-tp1": {
        "calls": "66e99d04e6f19cbf68b54779f2a40b4ed0654eec1bd89e5eb68cc066a571bc1f",
        "results": "9ac63ab575d8b4adad6b5a3fe67f02ab8ac5b9ead094e93834935dfc98466a15",
        "logs": "4025602a8fcf8ac614c2804500205aa6a9addafc10c63c9e88370fb468a5469f",
    },
    "text-decode-tep2": ONE_RANK_DIGESTS_WITHOUT_THE_LOW_END["text-decode-tep2"],
}


def _plan_without_the_low_end(monkeypatch) -> None:
    """Make the grid module plan the prefill new-token axis as it did before the axis had a low end.

    The axis was 1 in front of the cut graph axis. It is that again when the module finds no first batch that
    runs with a graph. ``first_prefill_graph_tokens`` is replaced in ``sglbench_grid`` itself, the one module
    object the launcher, the driver and this file plan with, so a launch and what a test expects of it change
    together. Nothing else of the benchmark is replaced.
    """
    assert launcher.grid is grid and driver.grid is grid
    # 4, 65 and 96 are what the small cap of three leaves of the graph axis of the FAKE limits at page size 1.
    assert grid.prefill_new_token_axis(_fake_limits(1), SMALL_CAPS) == [1, 2, 4, 65, 96]
    monkeypatch.setattr(grid, "first_prefill_graph_tokens", lambda graph_sizes: None)
    assert grid.prefill_new_token_axis(_fake_limits(1), SMALL_CAPS) == [1, 4, 65, 96]


def _launch_one_rank(tmp_path: Path, monkeypatch, name: str) -> _Launch:
    """One of ``ONE_RANK_LAUNCHES``, end to end against the FAKE sglang package, with the fixed FAKE run id."""
    mode, page_size, tp_size, word, experts, flags = ONE_RANK_LAUNCHES[name]
    fixed = types.SimpleNamespace(uuid4=lambda: types.SimpleNamespace(hex=FAKE_FIXED_RUN_ID))
    monkeypatch.setattr(launcher, "uuid", fixed)
    options = {
        "page_size": page_size,
        "tp_size": tp_size,
        "flags": flags,
        "prepare": lambda world, _pod_dir: setattr(world, "experts", experts),
        "real_compat": True,
    }
    if word is not None or experts:
        return _launch_text(tmp_path, monkeypatch, mode, word=word, **options)
    return _launch(tmp_path, monkeypatch, mode, **options)


def _one_rank_digests(launch: _Launch) -> dict:
    """What a launch sent and wrote, as three SHA-256 digests.

    ``calls``: the prompts, the request ids and the output-token count of every engine call, with the time
    limit. ``results``: every result file, parsed, in the key order it was written in and without
    ``producer.attn_dp_size``. ``logs``: every recorder log. The directory of the test is written as a fixed
    word, because it differs from run to run. So is the process id of the test in a launch with one FAKE
    scheduler process, whose recorder takes that id; with several processes the ids in the log are FAKE ones.
    """
    root = str(launch.root)
    real_pid = os.getpid() if len(launch.world.processes) == 1 else None

    def digest(text: str) -> str:
        return hashlib.sha256(text.replace(root, "<ROOT>").encode()).hexdigest()

    calls = [[call.prompts, call.rids, call.max_new_tokens, call.timeout_seconds] for call in launch.world.calls]
    results = []
    for name in launch.result_files():
        payload = json.loads((launch.pod / name).read_text(encoding="utf-8"))
        if isinstance(payload.get("producer"), dict):
            payload["producer"].pop("attn_dp_size", None)
        results.append([name, payload])
    logs = []
    for recorder_dir in launch.recorder_dirs():
        for path in sorted(recorder_dir.iterdir()):
            name, text = path.name, path.read_text(encoding="utf-8")
            if real_pid is not None:
                name = name.replace(f"-{real_pid}.", "-<PID>.")
                text = text.replace(f'"pid":{real_pid},', '"pid":"<PID>",')
            logs.append([recorder_dir.name, name, text])
    return {
        "calls": digest(json.dumps(calls)),
        "results": digest(json.dumps(results)),
        "logs": digest(json.dumps(logs)),
    }


@pytest.mark.parametrize("name", list(ONE_RANK_LAUNCHES))
def test_a_launch_with_one_rank_and_without_the_low_end_sends_and_writes_what_it_did_before_dp_attention(
    tmp_path, monkeypatch, name
):
    _plan_without_the_low_end(monkeypatch)
    launch = _launch_one_rank(tmp_path, monkeypatch, name)

    # One function of the grid module is replaced, by ``_plan_without_the_low_end``, and nothing else.
    _assert_no_module_of_the_benchmark_is_replaced()
    on_disk = _assert_accepted(launch)
    assert on_disk["run_id"] == FAKE_FIXED_RUN_ID and len(launch.world.calls) > 20
    # What the result files have gained since: the size, with the value of a run without DP attention, and the
    # two dtypes of the KV cache, which are inside the digest of the result files.
    assert on_disk["producer"]["attn_dp_size"] == 1 and DP_PRODUCER_KEY not in on_disk["producer"]
    assert (on_disk["producer"]["kv_cache_dtype"], on_disk["producer"]["kv_cache_stored_dtype"]) == (
        FAKE_KV_CACHE_DTYPE,
        FAKE_KV_STORED_DTYPE,
    )
    assert _one_rank_digests(launch) == ONE_RANK_DIGESTS_WITHOUT_THE_LOW_END[name]


@pytest.mark.parametrize("name", list(ONE_RANK_LAUNCHES))
def test_a_launch_with_one_rank_sends_and_writes_what_it_did_when_the_low_end_was_added(tmp_path, monkeypatch, name):
    launch = _launch_one_rank(tmp_path, monkeypatch, name)

    _assert_no_module_of_the_benchmark_is_replaced()
    on_disk = _assert_accepted(launch)
    assert on_disk["run_id"] == FAKE_FIXED_RUN_ID and len(launch.world.calls) > 20
    assert on_disk["producer"]["attn_dp_size"] == 1 and DP_PRODUCER_KEY not in on_disk["producer"]
    assert _one_rank_digests(launch) == ONE_RANK_DIGESTS[name]


@pytest.mark.parametrize("name", list(ONE_RANK_LAUNCHES))
def test_the_low_end_adds_its_points_to_a_launch_with_one_rank_and_leaves_the_other_points_as_they_were(
    tmp_path, monkeypatch, name
):
    """What separates the two tables of digests, shown on the two launches themselves.

    The launch with the low end runs the points of ``ONE_RANK_LOW_END_POINTS`` and is, in every other point,
    the launch without it: the same points in the same order, the same rows but for the ids, the same calls up
    to the first added point, and after it calls of the same shape. The shots of the prefill points that run
    after the added ones are the only calls that carry other tokens than before: the tokens of a shot follow
    from how many shots of its kind were issued before it, and the added shots are issued before those.
    """
    launch = _launch_one_rank(tmp_path / "with", monkeypatch, name)
    points = launch.planned().points
    monkeypatch.delenv(RECORDER_ENV, raising=False)
    _plan_without_the_low_end(monkeypatch)
    before = _launch_one_rank(tmp_path / "without", monkeypatch, name)
    assert (launch.code, before.code) == (0, 0)

    added = [point for point in points if point.phase == grid.PREFILL and point.new_tokens == 2]
    assert [(point.batch_size, point.cached_tokens) for point in added] == ONE_RANK_LOW_END_POINTS[name]
    added_ids = [point.benchmark_id for point in added]

    def id_before(benchmark_id: int) -> int:
        """The id a point has in the launch without the added points: one less for each that runs before it."""
        return benchmark_id - sum(1 for added_id in added_ids if added_id < benchmark_id)

    # The plan: the other points are the points of the launch without the low end, in its order.
    others = [point for point in points if point not in added]
    assert [dataclasses.replace(point, benchmark_id=id_before(point.benchmark_id)) for point in others] == list(
        before.planned().points
    )

    # The result files, the rank file and the merged file.
    assert launch.result_files() == before.result_files() == ["benchmark.json", "benchmark_merged.json"]
    for file_name in launch.result_files():
        payloads = [json.loads((run.pod / file_name).read_text(encoding="utf-8")) for run in (launch, before)]
        # The rows of the other points are the rows of the launch without the low end, but for the two ids a
        # row carries.
        rows = [row for row in payloads[0]["results"] if row["point"]["benchmark_id"] not in added_ids]
        for row in rows:
            row["point"]["benchmark_id"] = id_before(row["point"]["benchmark_id"])
            for fpm in row["fpms"]:
                fpm["counter_id"] = id_before(fpm["counter_id"])
        assert len(rows) == len(others) and rows == payloads[1]["results"]
        # The rest of the file is its file, but for the blocks that count points, prompts or FAKE time.
        rest = []
        for run, payload in zip((launch, before), payloads, strict=True):
            for key in ("results", "iteration_groups", "coverage", "grid_digest", "timing"):
                del payload[key]
            del payload["producer"]["content"]["issued"], payload["producer"]["first_token_need"]
            rest.append(json.dumps(payload).replace(str(run.root), "<ROOT>"))
        assert rest[0] == rest[1]

    def sent(call: _FakeCall) -> list:
        return [call.prompts, call.rids, call.max_new_tokens, call.timeout_seconds]

    def lengths(call: _FakeCall) -> list[int]:
        return [len(prompt) for prompt in call.prompts]

    # The calls up to the first added point are the calls of the launch without the low end, to the last token
    # and the last request id.
    calls, calls_before = launch.world.calls, before.world.calls
    first = next((index for index, call in enumerate(calls) if call.key[0] in added_ids), len(calls))
    assert first > 20 and [sent(call) for call in calls[:first]] == [sent(call) for call in calls_before[:first]]
    # The calls of the other points are its calls in stage, prompt lengths, output tokens and time limit.
    other_calls = [call for call in calls if call.key[0] not in added_ids]
    assert [
        (id_before(call.key[0]), *call.key[1:], lengths(call), call.max_new_tokens, call.timeout_seconds)
        for call in other_calls
    ] == [(*call.key, lengths(call), call.max_new_tokens, call.timeout_seconds) for call in calls_before]
    # Other tokens went out for the shots of the prefill points after the added ones, and for nothing else:
    # every seed and every decode run sends the prefix of its slot, which no earlier call changes.
    changed = [call.key for call, old in zip(other_calls, calls_before, strict=True) if call.prompts != old.prompts]
    last_added = max(added_ids, default=len(points))
    later = [point for point in others if point.phase == grid.PREFILL and point.benchmark_id > last_added]
    assert changed == [(point.benchmark_id, stage, 0) for point in later for stage in ("warm", "meas")]
    assert len(later) == len(added) and {point.new_tokens for point in later} <= {1}
    # Each call of an added point is one FAKE batch more in the log of every FAKE scheduler process.
    more = len(calls) - len(calls_before)
    assert more == sum(3 if cached else 2 for _batch, cached in ONE_RANK_LOW_END_POINTS[name])
    (log_dir,), (log_dir_before,) = launch.recorder_dirs(), before.recorder_dirs()
    for log, log_before in zip(sorted(log_dir.iterdir()), sorted(log_dir_before.iterdir()), strict=True):
        lines, lines_before = (path.read_text(encoding="utf-8").splitlines() for path in (log, log_before))
        assert len(lines) == len(lines_before) + more


def test_the_digests_of_a_launch_with_one_rank_notice_a_change(tmp_path, monkeypatch):
    launch = _launch_one_rank(tmp_path, monkeypatch, "seeded-prefill-tp1")
    before = _one_rank_digests(launch)

    # One more token in one prompt, one more key in one result file, one more line in one log.
    launch.world.calls[3].prompts[0].append(7)
    payload = launch.artifact()
    payload["producer"]["FAKE_new_key"] = 1
    launch.rank_path.write_text(json.dumps(payload), encoding="utf-8")
    (log,) = sorted(launch.recorder_dirs()[0].iterdir())
    log.write_text(log.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    after = _one_rank_digests(launch)
    assert [after[key] != before[key] for key in ("calls", "results", "logs")] == [True, True, True]
    # The same launch in another directory has the same digests.
    monkeypatch.delenv(RECORDER_ENV, raising=False)
    again = _launch_one_rank(tmp_path / "elsewhere", monkeypatch, "seeded-prefill-tp1")
    assert _one_rank_digests(again) == before == ONE_RANK_DIGESTS["seeded-prefill-tp1"]


# ---------------------------------------------------------------------------------------------------------------
# The two lines about a prefill grid the KV pool cut
# ---------------------------------------------------------------------------------------------------------------

# The second line: where a pool comes from. The same in every case.
POOL_LEVER_LINE = (
    "[sglbench] a larger pool needs a larger memory share (SGLang's --mem-fraction-static, which a plan of AISim's "
    "collector gives as --fpm-gpu-memory-utilization), or a larger --max-total-tokens where the command gives one"
)
# The prefill new-token axis of the FAKE limits with the launcher's default caps, by page size, typed here from
# the rule. The FAKE prefill graph sizes are 4, 8, 16, 32 and 64, and the largest request is the FAKE chunk
# budget: 96 tokens at page size 1 and 384 at page size 64. The axis holds every size c and c + 1, the
# doublings of 64 that the largest request leaves room for, the largest request, and the low end: 1 and 2,
# because a batch of 2 tokens is the smallest that SGLang runs with the graph of 4. The cap of 64 values cuts
# nothing.
FAKE_NEW_TOKEN_AXIS = {
    1: [1, 2, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 96],
    64: [1, 2, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 256, 384],
}

# For the FAKE limits at page 1 and the launcher's default caps, typed here on their own. The 32 pairs are
# batch 1, 2 and 4 for each n of 1, 2, 4, 5, 8, 9, 16 and 17, batch 1, 2 and 3 for 32, batch 1 and 2 for 33 and
# batch 1 for 64, 65 and 96. The pool of 1,184 tokens ends the cached prefix of the 8 pairs at batch 4 and of
# the pair of n = 32 at batch 3, and the longest batch of four, 4 * (425 + 2) = 1,708 tokens, must stay below
# the pool. ``test_the_numbers_of_the_pool_lines_are_the_ones_a_count_from_the_feasibility_rule_gives`` counts
# the numbers of this line and of the three after it.
POOL_CUT_LINES = [
    "[sglbench] the KV pool of one rank, 1184 tokens, cuts the prefill grid: of the 32 pairs of new tokens and "
    "batch size planned without the pool limit, pairs left out: 0, pairs that end at a shorter cached prefix: 9; "
    "a pool of 1709 tokens holds the whole grid",
    POOL_LEVER_LINE,
]
# At page size 64: I = 812, S = 2304 and a cached unit of 64 tokens. The 41 pairs are batch 1, 2 and 4 for each
# of the 11 values of n up to 64, batch 1, 2 and 3 for 65 and for 128, and batch 1 for 256 and 384. The pool
# shortens the 11 pairs at batch 4 and the 2 at batch 3, and the longest batch of four,
# 4 * (768 + 64 + 65) = 3,588 tokens, must stay below a pool of whole pages.
POOL_CUT_LINES_AT_PAGE_64 = [
    "[sglbench] the KV pool of one rank, 2304 tokens, cuts the prefill grid: of the 41 pairs of new tokens and "
    "batch size planned without the pool limit, pairs left out: 0, pairs that end at a shorter cached prefix: 13; "
    "a pool of 3648 tokens holds the whole grid",
    POOL_LEVER_LINE,
]
# A pool of 2 tokens at page 1 holds no request: every one of the 32 pairs is left out.
POOL_CUT_LINES_WITHOUT_A_PAIR = [
    "[sglbench] the KV pool of one rank, 2 tokens, cuts the prefill grid: of the 32 pairs of new tokens and "
    "batch size planned without the pool limit, pairs left out: 32, pairs that end at a shorter cached prefix: 0; "
    "a pool of 1709 tokens holds the whole grid",
    POOL_LEVER_LINE,
]
# A pool of 434 tokens at page 1, the FAKE input limit of 428 and 6: it keeps every pair and shortens the 10 at
# batch 2 (every n up to 33), the 8 at batch 4 and the one at batch 3.
POOL_CUT_LINES_NOT_ABOVE_THE_CONTEXT = [
    "[sglbench] the KV pool of one rank, 434 tokens, cuts the prefill grid: of the 32 pairs of new tokens and "
    "batch size planned without the pool limit, pairs left out: 0, pairs that end at a shorter cached prefix: 19; "
    "with the present input limit a pool of 1709 tokens holds the whole grid; the pool is not above the model's "
    "context: if it is below the context, SGLang took the input limit from the pool, and a larger pool raises "
    "the input limit too, which can lengthen the grid",
    POOL_LEVER_LINE,
]
# No cap flag at all: the launcher's default caps, which plan batch 1, 2 and 4 for prefill, or 3 where four
# requests no longer fit a step.
DEFAULT_CAPS = {"caps": grid.SampleCaps(), "cap_flags": []}


def _pool_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if "KV pool" in line or "memory share" in line]


def _counted_pool_cut(limits: grid.Limits) -> tuple[int, int, int, int]:
    """What the pool takes from the prefill block of the default caps, counted here from the feasibility rule.

    The four numbers of the launcher's line: the pairs of new tokens and batch size planned without the pool
    limit, how many of them the pool leaves out, how many it ends at a shorter cached prefix, and the smallest
    pool that holds the block planned without the pool limit. Of ``sglbench_grid`` only the rule
    (``prefill_unfit_reason``) and the cached unit are used. The new tokens are ``FAKE_NEW_TOKEN_AXIS``, the
    batch sizes of one value are the first three of the powers of two up to the largest batch and that batch,
    and every largest value is found by trying the candidates one by one. The planned block is counted over
    the same new tokens as the block without the pool limit, which is right while the pool leaves the axis
    whole or leaves no request at all; the test that calls this checks it.
    """
    page = limits.page_size
    unit = grid.cached_unit(page)

    def with_pool(tokens: int) -> grid.Limits:
        return dataclasses.replace(limits, max_total_num_tokens=tokens)

    def longest_prefixes(record: grid.Limits) -> dict[tuple[int, int], int]:
        """(new tokens, batch size) of every pair the record plans -> the longest cached prefix of the pair."""

        def fits(batch: int, new: int, cached: int) -> bool:
            return grid.prefill_unfit_reason(record, batch, new, cached) is None

        longest = {}
        for new in FAKE_NEW_TOKEN_AXIS[page]:
            fitting = [batch for batch in range(1, record.max_running_requests + 1) if fits(batch, new, 0)]
            powers = [batch for batch in fitting if batch & (batch - 1) == 0]
            for batch in sorted({*powers, *fitting[-1:]})[:3]:
                prefixes = range(0, record.max_req_input_len, unit)
                longest[(new, batch)] = max(cached for cached in prefixes if fits(batch, new, cached))
        return longest

    # A pool of a million pages is out of the reach of every point: none of the FAKE limits needs as much as
    # 9 * (812 + 2 * 64 + 1) tokens.
    wanted, kept = longest_prefixes(with_pool(1_000_000 * page)), longest_prefixes(limits)
    dropped = [pair for pair in wanted if pair not in kept]
    shortened = [pair for pair, longest in wanted.items() if pair in kept and kept[pair] < longest]

    def holds(tokens: int) -> bool:
        record = with_pool(tokens)
        return all(
            grid.prefill_unfit_reason(record, batch, new, longest) is None for (new, batch), longest in wanted.items()
        )

    # A pool is a whole number of pages. A pair's longest prefix needs the most of it.
    whole = next(tokens for tokens in range(page, 1_000_000 * page, page) if holds(tokens))
    return len(wanted), len(dropped), len(shortened), whole


@pytest.mark.parametrize(
    ("page_size", "pool_tokens", "line"),
    [
        (1, None, POOL_CUT_LINES[0]),
        (64, None, POOL_CUT_LINES_AT_PAGE_64[0]),
        (1, 2, POOL_CUT_LINES_WITHOUT_A_PAIR[0]),
        (1, 434, POOL_CUT_LINES_NOT_ABOVE_THE_CONTEXT[0]),
    ],
)
def test_the_numbers_of_the_pool_lines_are_the_ones_a_count_from_the_feasibility_rule_gives(
    page_size, pool_tokens, line
):
    """``pool_tokens`` None is the pool of the FAKE limits; a number is the pool a test gives the FAKE engine."""
    limits = _fake_limits(page_size)
    if pool_tokens is not None:
        limits = dataclasses.replace(limits, max_total_num_tokens=pool_tokens)
    # The axis typed above is the one the launcher plans, with 2 beside 1. In these cases a pool either leaves
    # it whole or leaves no request at all.
    roomy = dataclasses.replace(limits, max_total_num_tokens=1_000_000 * page_size)
    assert grid.prefill_new_token_axis(roomy, grid.SampleCaps()) == FAKE_NEW_TOKEN_AXIS[page_size]
    assert grid.prefill_new_token_axis(limits, grid.SampleCaps()) in (FAKE_NEW_TOKEN_AXIS[page_size], [])

    counted = _counted_pool_cut(limits)

    numbers = re.fullmatch(
        r"\[sglbench\] the KV pool of one rank, (\d+) tokens, cuts the prefill grid: of the (\d+) pairs of new "
        r"tokens and batch size planned without the pool limit, pairs left out: (\d+), pairs that end at a shorter "
        r"cached prefix: (\d+); (?:with the present input limit )?a pool of (\d+) tokens holds the whole grid(?:; .*)?",
        line,
    )
    assert [int(number) for number in numbers.groups()] == [limits.max_total_num_tokens, *counted]
    # The grid module, which the launcher takes its numbers from, counts the same.
    cut = grid.prefill_pool_cut(limits, grid.SampleCaps())
    assert (cut.pairs, cut.dropped_pairs, cut.shortened_pairs, cut.whole_grid_pool_tokens) == counted


def _assert_pool_lines_said_twice(lines: list[str], pool_lines: list[str], outcome_lines: int) -> None:
    """The lines stand once where the grid is planned and once directly before the outcome, and nowhere else."""
    assert _pool_lines(lines) == [*pool_lines, *pool_lines]
    first = lines.index(pool_lines[0])
    # The grid is planned after the engine is built and, in a text run, after the text pool is made.
    assert lines[first - 1].startswith(("[sglbench] the engine is built", "[sglbench] the text pool holds "))
    assert lines[first + 1] == pool_lines[1]
    end = len(lines) - outcome_lines
    assert lines[end - len(pool_lines) : end] == pool_lines


@pytest.mark.parametrize("mode", ["prefill", "agg"])
def test_a_prefill_grid_the_kv_pool_cut_is_said_when_it_is_planned_and_before_the_outcome(
    tmp_path, monkeypatch, capsys, mode
):
    launch = _launch(tmp_path, monkeypatch, mode, **DEFAULT_CAPS)

    assert launch.code == 0
    lines = _stderr_lines(capsys)
    _assert_pool_lines_said_twice(lines, POOL_CUT_LINES, 1)
    assert lines[-1] == f"[sglbench] complete: wrote {launch.rank_path}, {launch.merged_path}"
    # The lines are on stderr only: no result file holds them.
    for name in launch.result_files():
        assert "KV pool" not in (launch.pod / name).read_text(encoding="utf-8")


def test_the_pool_lines_of_a_dep_launch_are_about_the_pool_of_one_rank(tmp_path, monkeypatch, capsys):
    launch = _launch_dep(tmp_path, monkeypatch, "prefill", **DEFAULT_CAPS)

    assert launch.code == 0 and len(launch.result_files()) == 3
    _assert_pool_lines_said_twice(_stderr_lines(capsys), POOL_CUT_LINES, 1)


def test_the_pool_lines_at_a_page_size_of_64(tmp_path, monkeypatch, capsys):
    launch = _launch(tmp_path, monkeypatch, "prefill", page_size=64, **DEFAULT_CAPS)

    assert launch.code == 0
    _assert_pool_lines_said_twice(_stderr_lines(capsys), POOL_CUT_LINES_AT_PAGE_64, 1)


def test_nothing_is_said_about_the_pool_with_caps_that_plan_no_batch_above_two(tmp_path, monkeypatch, capsys):
    launch = _launch(tmp_path, monkeypatch, "prefill")

    assert launch.code == 0 and _pool_lines(_stderr_lines(capsys)) == []


def test_nothing_is_said_about_the_pool_in_a_decode_run(tmp_path, monkeypatch, capsys):
    launch = _launch(tmp_path, monkeypatch, "decode", **DEFAULT_CAPS)

    assert launch.code == 0 and _pool_lines(_stderr_lines(capsys)) == []


def test_nothing_is_said_about_a_pool_that_holds_the_whole_prefill_grid(tmp_path, monkeypatch, capsys):
    def prepare(world, _pod_dir):
        world.limits = dataclasses.replace(world.limits, max_total_num_tokens=1709)

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare, **DEFAULT_CAPS)

    assert launch.code == 0 and _pool_lines(_stderr_lines(capsys)) == []


def test_nothing_is_said_about_the_pool_in_a_run_that_is_refused_before_a_grid_exists(tmp_path, monkeypatch, capsys):
    launch = _launch(
        tmp_path, monkeypatch, "prefill", flags=["--stream-interval", "4"], real_compat=True, **DEFAULT_CAPS
    )

    assert launch.code == 64
    lines = _stderr_lines(capsys)
    assert "[sglbench] FAILED: rejected: stream_interval must be 1 (got 4)" in lines
    assert _pool_lines(lines) == []


def test_the_pool_lines_come_before_the_lines_of_a_failed_point(tmp_path, monkeypatch, capsys):
    victim = grid.build_grid("prefill", _fake_limits(1), grid.SampleCaps()).points[1]

    def prepare(world, _pod_dir):
        world.call_faults[(victim.benchmark_id, "meas", 0)] = "abort"

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare, real_compat=True, **DEFAULT_CAPS)

    assert launch.code == 65
    lines = _stderr_lines(capsys)
    _assert_pool_lines_said_twice(lines, POOL_CUT_LINES, 2)
    assert lines[-2].startswith(f"[sglbench] FAILED: benchmark_id={victim.benchmark_id}: benchmark point failed: ")
    assert lines[-1].startswith("[sglbench] wrote the failed artifact ")


def test_the_pool_lines_come_before_the_line_of_a_partial_run(tmp_path, monkeypatch, capsys):
    last_done = grid.build_grid("prefill", _fake_limits(1), grid.SampleCaps()).points[1]

    def prepare(world, _pod_dir):
        # A prefill point is measured in its "meas" call.
        world.call_faults[(last_done.benchmark_id, "meas", 0)] = "deadline"

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare, **DEFAULT_CAPS)

    assert launch.code == 66
    lines = _stderr_lines(capsys)
    _assert_pool_lines_said_twice(lines, POOL_CUT_LINES, 1)
    assert lines[-1].startswith("[sglbench] PARTIAL: the soft deadline ended the run (timeout); wrote ")


def test_the_pool_lines_come_before_the_lines_of_a_run_a_signal_stopped(tmp_path, monkeypatch, capsys):
    victim = grid.build_grid("prefill", _fake_limits(1), grid.SampleCaps()).points[1]

    def prepare(world, _pod_dir):
        world.call_faults[(victim.benchmark_id, "meas", 0)] = "sigterm"

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare, real_compat=True, **DEFAULT_CAPS)

    assert launch.code == 65
    lines = _stderr_lines(capsys)
    _assert_pool_lines_said_twice(lines, POOL_CUT_LINES, 2)
    assert lines[-2] == "[sglbench] FAILED: stopped: signal SIGTERM"


def test_the_pool_lines_come_before_the_line_of_a_failed_write(tmp_path, monkeypatch, capsys):
    def prepare(world, pod_dir):
        # The results directory is taken away while the engine shuts down.
        world.hooks["shutdown"] = lambda: shutil.rmtree(pod_dir)

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare, **DEFAULT_CAPS)

    assert launch.code == 70
    lines = _stderr_lines(capsys)
    _assert_pool_lines_said_twice(lines, POOL_CUT_LINES, 1)
    assert lines[-1].startswith("[sglbench] FAILED: internal_error: the result files could not be written: ")


def test_the_pool_lines_when_the_pool_leaves_out_every_pair(tmp_path, monkeypatch, capsys):
    """A pool of 2 tokens holds no request, so no prefill point is planned and the run fails for that."""

    def prepare(world, _pod_dir):
        world.limits = dataclasses.replace(world.limits, max_total_num_tokens=2)

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare, **DEFAULT_CAPS)

    assert launch.code == 65
    lines = _stderr_lines(capsys)
    _assert_pool_lines_said_twice(lines, POOL_CUT_LINES_WITHOUT_A_PAIR, 2)
    assert lines[-2] == "[sglbench] FAILED: no feasible benchmark points for phase(s): prefill"


def test_the_pool_lines_when_the_pool_is_not_above_the_context(tmp_path, monkeypatch, capsys):
    """The FAKE input limit is 428. A pool of 434 tokens is a pool that is not above the model's context.

    The limits do not say whether it is below the context or equal to it, so the line says "if".
    """

    def prepare(world, _pod_dir):
        world.limits = dataclasses.replace(world.limits, max_total_num_tokens=434)

    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare, **DEFAULT_CAPS)

    assert launch.code == 0
    _assert_pool_lines_said_twice(_stderr_lines(capsys), POOL_CUT_LINES_NOT_ABOVE_THE_CONTEXT, 1)


def test_the_pool_lines_when_only_the_input_limit_comes_from_the_pool(tmp_path, monkeypatch, capsys):
    """With batch 1 alone the pool ends no axis itself; the grid can still be shorter, through the input limit."""

    def prepare(world, _pod_dir):
        world.limits = dataclasses.replace(world.limits, max_total_num_tokens=434)

    one_batch = [*SMALL_CAP_FLAGS[:-1], "--prefix-max-batch-size-samples=1"]
    caps = dataclasses.replace(SMALL_CAPS, prefix_max_batch_size_samples=1)
    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare, caps=caps, cap_flags=one_batch)

    assert launch.code == 0
    cut = grid.prefill_pool_cut(launch.world.limits, caps)
    assert (cut.dropped_pairs, cut.shortened_pairs, cut.pool_sets_input_limit) == (0, 0, True)
    expected = [
        "[sglbench] the KV pool of one rank, 434 tokens, cuts nothing from the prefill grid of the present input "
        "limit; the pool is not above the model's context: if it is below the context, SGLang took the input limit "
        "from the pool, and a larger pool raises the input limit, which can lengthen the grid",
        POOL_LEVER_LINE,
    ]
    _assert_pool_lines_said_twice(_stderr_lines(capsys), expected, 1)


def test_a_fault_in_the_pool_note_changes_neither_the_outcome_nor_the_exit_code(tmp_path, monkeypatch, capsys):
    def broken(_run):
        raise RuntimeError("FAKE fault")

    monkeypatch.setattr(driver.BenchmarkRun, "prefill_pool_cut", broken)
    launch = _launch(tmp_path, monkeypatch, "prefill", **DEFAULT_CAPS)

    assert launch.code == 0
    _assert_accepted(launch)
    note = "[sglbench] what the KV pool takes from the prefill grid could not be worked out: RuntimeError: FAKE fault"
    lines = _stderr_lines(capsys)
    assert [line for line in lines if "KV pool" in line] == [note, note]
    assert lines[-2:] == [note, f"[sglbench] complete: wrote {launch.rank_path}, {launch.merged_path}"]


def test_a_fault_that_cannot_be_put_into_words_is_named_by_its_type(monkeypatch):
    class _Mute(RuntimeError):
        def __str__(self):
            raise RuntimeError("FAKE str fault")

    class _Run:
        def prefill_pool_cut(self):
            raise _Mute()

    said = []
    monkeypatch.setattr(launcher, "_say", said.append)
    launcher._say_pool_cut(_Run())
    assert said == ["what the KV pool takes from the prefill grid could not be worked out: _Mute"]


def test_the_pool_lines_are_said_before_the_first_engine_call_and_after_the_last(tmp_path, monkeypatch):
    """The two places themselves. On stderr the four lines are adjacent wherever they were said, because the
    launcher prints nothing between the planned grid and the end of the run, so the other tests of this section
    pin the count, the text and the line before the outcome, and this one pins when each line was said."""
    seen = {}
    calls_at = []
    say = launcher._say

    def prepare(world, _pod_dir):
        seen["world"] = world

    def counting(text):
        if "KV pool" in text or "memory share" in text:
            calls_at.append(len(seen["world"].calls))
        say(text)

    monkeypatch.setattr(launcher, "_say", counting)
    launch = _launch(tmp_path, monkeypatch, "prefill", prepare=prepare, **DEFAULT_CAPS)

    assert launch.code == 0
    total = len(launch.world.calls)
    assert total > 0 and calls_at == [0, 0, total, total]


def test_a_stderr_that_cannot_take_the_pool_lines_does_not_stop_the_note(monkeypatch):
    """``_say_pool_cut`` raises no ``Exception``, also when printing itself fails."""

    class _Run:
        def prefill_pool_cut(self):
            return grid.PoolCut(36, 12, 2, 2, 43, False)

    said = []

    def refuse(text):
        said.append(text)
        raise OSError("FAKE closed stream")

    monkeypatch.setattr(launcher, "_say", refuse)
    launcher._say_pool_cut(_Run())
    # The first line was tried, and the second was not.
    assert len(said) == 1 and said[0].startswith("the KV pool of one rank, 36 tokens, cuts the prefill grid")


def test_the_flags_of_the_pool_lines_are_the_ones_the_collector_and_sglang_use():
    """The launcher only prints the names. The collector owns two of them: its plan flag and its translation row."""
    from collector.fpm_forward import sglang_fpm_builder

    assert (launcher.MEMORY_SHARE_FLAG, launcher.PLAN_MEMORY_SHARE_FLAG, launcher.POOL_TOKENS_FLAG) == (
        "--mem-fraction-static",
        "--fpm-gpu-memory-utilization",
        "--max-total-tokens",
    )
    forward = Path(launcher.__file__).resolve().parents[1]
    # The plan flag is defined by the collector's parser. The runner hands its value to the generator as
    # ``kv_cache_free_gpu_memory_fraction`` (pinned in test_fpm_runtime_probe.py and test_fpm_profile_collection.py),
    # which the generator renders as vLLM's ``--gpu-memory-utilization``. The cell builder writes that one as
    # SGLang's flag: the row is checked here.
    assert f'"{launcher.PLAN_MEMORY_SHARE_FLAG}",\n' in (forward / "config.py").read_text(encoding="utf-8")
    assert sglang_fpm_builder._TRANSLATION["--gpu-memory-utilization"].writes == (launcher.MEMORY_SHARE_FLAG,)
    # A cell of the collector never carries SGLang's cap on the pool: the builder has no row that writes it.
    assert all(launcher.POOL_TOKENS_FLAG not in row.writes for row in sglang_fpm_builder._TRANSLATION.values())


# ---------------------------------------------------------------------------------------------------------------
# The low end of the prefill new-token axis in a launch: a row on each side of the first step of the table
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("caps", "new_tokens", "batch_sizes"),
    [
        # The small caps: 4, 65 and 96 are what a cap of three leaves of the graph axis, and 1 and 2 do not
        # count against it. Two batch sizes.
        ({}, [96, 65, 4, 2, 1], [2, 1]),
        # The launcher's default caps: ``FAKE_NEW_TOKEN_AXIS`` from its upper end, and batch 4, 2 and 1.
        (DEFAULT_CAPS, FAKE_NEW_TOKEN_AXIS[1][::-1], [4, 2, 1]),
    ],
)
def test_a_generated_prefill_block_is_measured_at_one_and_at_two_new_tokens_per_request(
    tmp_path, monkeypatch, caps, new_tokens, batch_sizes
):
    """The FAKE prefill graph sizes start at 4, as SGLang's default sizes do.

    SGLang then runs a batch of one new token without a graph and a batch of two with the graph of 4. The
    launcher measures both, so that the table has a row on each side of that step, also under a cap that
    leaves three values of the graph axis.
    """
    launch = _launch(tmp_path, monkeypatch, "prefill", real_compat=True, **caps)

    # Accepted: every row is the planned point with the FAKE time of a batch of its size and its new tokens.
    on_disk = _assert_accepted(launch)
    points = [row["point"] for row in on_disk["results"]]
    # New tokens per request, in the order their groups ran.
    per_request = [point["total_prefill_tokens"] // point["batch_size"] for point in points]
    assert all(point["total_prefill_tokens"] % point["batch_size"] == 0 for point in points)
    assert list(dict.fromkeys(per_request)) == new_tokens
    two = [point for point, new in zip(points, per_request, strict=True) if new == 2]
    assert list(dict.fromkeys(point["batch_size"] for point in two)) == batch_sizes
    # Every one of them ran as planned: the measured call sent, for each request, the cached prefix and 2
    # tokens more, and the FAKE scheduler computed exactly those 2 beside the prefix it found in its cache.
    (recorder_dir,) = launch.recorder_dirs()
    (log,) = sorted(recorder_dir.iterdir())
    batches = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()[1:]]
    for point in two:
        batch_size = point["batch_size"]
        cached = point["total_kv_read_tokens"] // batch_size
        (measured,) = [call for call in launch.calls_of(point["benchmark_id"]) if call.key[1] == "meas"]
        assert [len(prompt) for prompt in measured.prompts] == [cached + 2] * batch_size
        assert measured.max_new_tokens == 1
        (batch,) = [batch for batch in batches if batch["rids"] == measured.rids]
        assert (batch["mode"], batch["extend_num_tokens"]) == ("EXTEND", 2 * batch_size)
        assert (batch["prefix_lens"], batch["extend_lens"]) == ([cached] * batch_size, [2] * batch_size)
