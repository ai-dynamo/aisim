# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# MiniMax-M3 support landed in SGLang v0.5.16 (models/minimax_m3.py +
# layers/attention/minimax_sparse_backend.py + minimax_sparse_ops/), and its
# `msa` family pin has since moved to 0.5.23. Qwen4-Exp compressed QSA lands in
# v0.5.21 (models/qwen4_exp.py + layers/attention/qsa/ +
# mem_cache/qsa_kv_pool.py) and was collected at the 0.5.21 this campaign
# validated; it then moved with the `msa` family to 0.5.23 per the
# pin-consolidation rule, so one release (the family pin) serves both routes.
# The 0.5.21 Qwen tables stay under their version key.
# One release per file (owner policy 2026-09-20); earlier code is `git log -- <this file>` away.
__compat__ = "sglang==0.5.23"

"""
MSA Module Collector for SGLang — MiniMax-M3 MSA and Qwen4-Exp QSA
sparse-attention benchmarking.

Two routes share one op family because the perf consumer keys both on
``msa_context_module`` / ``msa_generation_module``:

``attention_type: msa`` — MiniMax-M3 (this section).
``attention_type: qsa`` — Qwen3.8-Flash-Next (see the QSA section below).

Profiles the complete MiniMax-M3 sparse-attention module forward pass (the
fused qkv+index projection, grouped Gemma QK norm + partial RoPE, index
attention + top-k block selection, sparse GQA and the output projection) at
the ModelRunner level, through SGLang's own ServerArgs → ModelRunner →
ScheduleBatch → ForwardBatch pipeline with dummy weights.

Construction follows serving for the sparse layers (layers 3+ of the real
checkpoint): ``MiniMaxM3DecoderLayer`` builds
``MiniMaxM3Attention(is_sparse_attention_layer=True, disable_index_value=True)``
for every layer whose ``sparse_attention_freq``/``sparse_disable_index_value``
entries are set (models/minimax_m3.py:1702-1721@v0.5.23); the collector
truncates those per-layer arrays so its benchmark layer takes that path.
Serving fuses qkv_proj and index_qkv_proj into one GEMM at the end of
``load_weights`` (models/minimax_m3.py:2244-2246, 1117-1158); the dummy loader
never calls ``load_weights`` (model_loader/loader.py:1493-1530), so
``load_model_runner`` runs that same step. Attention dispatch is SGLang's own:
``attn_backend_wrapper`` wraps the platform dense backend with
``MiniMaxHybridAttnBackend``/``MiniMaxSparseAttnBackend`` for every
``is_minimax_sparse`` model (layers/attention/attention_registry.py:375-391),
and the benchmarked layer id is verified to route to the sparse backend.

Per SM, SGLang's M3 server-args override
(arg_groups/model_overrides/minimax_m3.py@v0.5.23) resolves backend and page:
  * SM90: fa3 + page 128 (:178-192); the sparse layers run the Triton path.
  * SM100/103: fa4 + page 128 with a BF16 KV cache, trtllm_mha + page 128 with
    fp8_e4m3 (:147-177). The latter turns on the fp8 attention GEMMs: fp8 q,
    an fp8 index cache and fp8 sparse kernels (server_args.py:592-610
    m3_fp8_attn_gemm_enabled). The main sparse step runs fmha_sm100 (MSA)
    when the image ships it (minimax_sparse_ops/msa.py:68-82,
    minimax_sparse_backend.py:196-249), else Triton.
  * SM89, SM120/121: no override branch. The generic default is flashinfer
    (arg_groups/model_override_base.py:277-334) at page 1
    (arg_groups/overrides.py:1341-1362); the sparse layers run Triton.
The collector never pins a kernel: ``kernel_source`` records what ran.

Op names / perf schema are aligned with collector/trtllm/collect_msa_module.py
(msa_context_module / msa_generation_module) so perf_database queries work
across frameworks; the persisted ``architecture`` is the checkpoint's own
(``MiniMaxM3ForCausalLM`` / ``Qwen4ExpForConditionalGeneration``).
Unless noted, SGLang citations are @v0.5.23 on the MSA route and @v0.5.21 on
the QSA route, relative to python/sglang/srt (kernel paths start at
python/sglang/kernels).

QSA route (``attention_type: qsa``, Qwen3.8-Flash-Next @ SGLang v0.5.21):
the benchmark hosts the real checkpoint config with a 4-layer text stack
(three GDN layers then one QSA layer — the checkpoint's own
``full_attention_interval=4`` period), PLE tables dropped, and per-rank head
counts sharded exactly as serving does. The timed unit is
``Qwen4ExpAttentionDecoderLayer.self_attention``: fused qkv+output-gate
projection, fused Gemma QK-norm + MRoPE + gate, the QSA indexer (index-QK
projection, q-norm, fused RoPE+store, compressed-key top-k selection), the
sparse attention call itself, and o_proj. Attention dispatch is SGLang's own:
the hybrid-GDN wrapper's full-attention side becomes ``QwenSparseAttnBackend``
for every ``is_qwen_qsa`` model (layers/attention/attention_registry.py
:458-466@v0.5.21), and that routing is verified per case. Compressed QSA makes
SGLang pin ``page_size=64`` (arg_groups/model_overrides/qwen4_exp.py), and
decode runs under a captured CUDA graph where serving covers the batch size.
The collector never pins a kernel: it records the route the backend actually
took in ``kernel_source``. Context prefixes are aligned to the checkpoint's
``indexer_compress_ratio`` (4): the pool addresses whole compressed groups and
the extend write plan asserts ``prefix % ratio == 0`` on the device
(qwen_sparse_attn_backend.py:527), so a sub-group prefix such as the declared
1 is measured as the smallest legal one (4) and the RoPE-ceiling point floors
to a group boundary instead of aborting the sweep.

Measurement deviation on that route: the M3 history seeding
(``_initialize_msa_history``, which writes the cached prefix's KV and index
keys through the sparse backend's pooled writer) is MSA-only and does not run
for QSA. The collector clears the request and KV pools between points, but
neither clear zeroes the cache buffers, so a prefixed QSA point reads its
prefix out of whatever the recycled pages hold — zeros on a page's first use, a
previous point's K/V afterwards. That is the condition the accepted C3.3 QSA
rows were measured under. The block choice is score-dependent, but the timed
top-k work is the selected block count, which the geometry fixes, so the
recycled content does not change the measured kernels' shape. Seeding QSA
would move a table already in the dataset, so it stays a documented deviation
until those rows are re-collected.

Usage (inside the pinned SGLang image, collector/framework_manifest.yaml):
    python3 -m collector.sglang.collect_msa_module --mode context --quick \
        --batch-size 2 --seq-len 512 --num-heads 64
    python3 -m collector.sglang.collect_msa_module --mode generation --quick \
        --batch-size 4 --seq-len 2048 --num-heads 64
    python3 -m collector.sglang.collect_msa_module --mode context --quick \
        --model Qwen/Qwen3.8-Flash-Next --num-heads 24 --batch-size 1 --seq-len 8192
"""

import argparse
import functools
import gc
import json
import os
import subprocess
import sys
import tempfile
import traceback
from array import array
from importlib.metadata import version as get_version

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from collector.case_generator import (
    get_mla_module_model_aliases,
    get_mla_module_model_specs,
)
from collector.helper import (
    PerfLogError,
    PhaseClock,
    _resolve_local_model_path,
    benchmark_with_power,
    get_sm_version,
    log_perf,
    render_phase_totals,
    run_measurement_subprocess,
)
from collector.op_plans import plan, plan_cases

try:
    from collector.sglang.runtime_compat import resolved_arg
    from collector.sglang.runtime_limits import (
        KERNEL_LIMIT,
        kernel_limit_error,
        kv_pool_capacity_tokens,
        kv_pool_page_size,
        over_chunk,
        over_kv_pool,
        refusal_summary,
        refuse,
        required_kv_alloc_tokens,
        runtime_chunk_size,
        serving_memory_defaults,
        temporarily_chunked_alloc_extend,
    )
except ModuleNotFoundError:  # direct-script import inside the subprocess
    from runtime_compat import resolved_arg
    from runtime_limits import (
        KERNEL_LIMIT,
        kernel_limit_error,
        kv_pool_capacity_tokens,
        kv_pool_page_size,
        over_chunk,
        over_kv_pool,
        refusal_summary,
        refuse,
        required_kv_alloc_tokens,
        runtime_chunk_size,
        serving_memory_defaults,
        temporarily_chunked_alloc_extend,
    )

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

MSA_ARCHITECTURE = "MiniMaxM3ForCausalLM"  # AIC-canonical persisted architecture
# SGLang's serving identity for the M3 text decoder. The official
# MiniMaxAI/MiniMax-M3 checkpoint is a VL repo whose text_config declares
# architectures=["MiniMaxM3SparseForCausalLM"] (HF config.json), and SGLang
# keys every M3 code path on that name: is_minimax_sparse
# (configs/model_config.py:426-431), the model EntryClass
# (models/minimax_m3.py:2274), the server-args override
# (arg_groups/model_overrides/minimax_m3.py:37-41), and the KV-pool /
# attention-backend wiring (mem_cache/kv_cache_configurator.py:1890-1896,
# 2552-2600; layers/attention/attention_registry.py:375-391).
SGLANG_MSA_ARCHITECTURE = "MiniMaxM3SparseForCausalLM"

# Qwen4-Exp compressed QSA (Qwen3.8-Flash-Next, SGLang v0.5.21). The
# checkpoint's own architectures[0] IS SGLang's model EntryClass, so unlike M3
# above no config rewrite is needed — the bundled config is a real multimodal
# config (vision_config present, language_model_only=false), which is what
# keeps is_mrope_enabled=True on the loaded model.
QSA_ARCHITECTURE = "Qwen4ExpForConditionalGeneration"
# Benchmark-layer surgery: the checkpoint alternates three linear (GDN) layers
# with one full-attention (QSA) layer (full_attention_interval=4,
# layer_types[3] == "full_attention"), so a 4-layer stack reproduces the QSA
# layer's serving construction path with one PLE-free period of context.
QSA_BENCH_NUM_LAYERS = 4
QSA_BENCH_LAYER = 3
QSA_BENCH_LAYER_TYPES = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
]
# MoE experts per benchmark layer. The routed MoE block is outside the timed
# module and every loaded layer builds one (Qwen4Exp always uses
# Qwen2MoeSparseMoeBlock), so the count is shrunk from the checkpoint's 512 to
# keep dummy-weight residency small; top-k stays checkpoint-verbatim and must
# remain <= this count.
QSA_BENCH_NUM_EXPERTS = 16

# Perf-database dtype strings → SGLang ServerArgs kv_cache_dtype values
# (same mapping as collect_mla_module.py).
SGLANG_KV_DTYPE = {"bfloat16": "bfloat16", "fp8": "fp8_e4m3"}

# MiniMax-M3 native GQA geometry (bundled MiniMaxAI--MiniMax-M3_config.json,
# identical to the HF checkpoint text_config): 64 query heads / 4 KV heads /
# 4 index heads. Serving TP sharding (models/minimax_m3.py):
#   q  heads: num_heads = total // attn_tp_size    (:749-751)
#   kv heads: max(1, total_kv // attn_tp_size)     (:752-762,
#             replicated when attn_tp_size > total_kv)
#   index heads: idx_head_tp_size = min(attn_tp_size, total_idx);
#             num_idx_heads = total_idx // idx_head_tp_size — i.e.
#             max(1, total_idx // attn_tp_size), replica ranks share one
#             head when attn_tp_size > total_idx (:784-796).
# All three are emulated on one GPU by shrinking the config head counts.

# Tokens of cached history written per store call (_initialize_msa_history):
# 64K tokens of K/V + index K stay below 200 MiB at TP 1.
_HISTORY_CHUNK_TOKENS = 1 << 16

# Per-subprocess sum of the point phase times printed by run_msa_module.
_POINT_PHASE_TOTALS: dict[str, float] = {}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ═══════════════════════════════════════════════════════════════════════
# Attention routes
# ═══════════════════════════════════════════════════════════════════════

_MSA_MODULE_ATTENTION_TYPES = ("msa", "qsa")


# ═══════════════════════════════════════════════════════════════════════
# Test-case generation (collect.py entrypoints)
# ═══════════════════════════════════════════════════════════════════════


@functools.cache
def _model_max_position_embeddings(model_id: str) -> int:
    """Max context (RoPE table size) from the bundled config — the same value
    SGLang uses for the rotary cache (get_rope max_position,
    models/minimax_m3.py:546-552) and default context_length.

    Multimodal checkpoints (Qwen4-Exp) keep the decoder's RoPE bound in
    ``text_config.max_position_embeddings``; the outer config has no such
    field. Both spellings are read here so the sweep ceiling is the decoder's.

    Raises when the artifact config cannot be resolved: a silent ``None``
    would disable every RoPE ceiling check and widen the sweep past the
    rotary table size (unresolvable declarations fail loudly)."""
    config_dir = _resolve_local_model_path(model_id)
    with open(os.path.join(config_dir, "config.json")) as f:
        config = json.load(f)
    text_config = config.get("text_config") or config
    value = text_config.get("max_position_embeddings") or config.get("max_position_embeddings")
    if not value:
        raise RuntimeError(
            f"max_position_embeddings missing/empty in {model_id} config.json; "
            "cannot bound the MSA sweep against the RoPE table size"
        )
    return int(value)


@functools.cache
def _qsa_compress_ratio(model_id: str) -> int:
    """QSA compressed-group width (tokens per sparse-attention block).

    Qwen4-Exp compresses each ``indexer_compress_ratio`` tokens into one
    sparse block: the checkpoint declares 4
    (``text_config.indexer_compress_ratio``), the pool keeps it verbatim as
    ``qsa_compress_ratio`` and derives ``qsa_block_topk = index_topk // ratio``
    (mem_cache/qsa_kv_pool.py:113-119@v0.5.21), and the QSA pool refuses a page
    that is not a ratio multiple (same file, :72-77).

    A context prefix must be a multiple of this ratio: the extend write plan
    asserts exactly that on the device
    (qwen_sparse_attn_backend.py:527@v0.5.21, "prefix sharing is page-granular
    and the page is a ratio multiple"), and a misaligned prefix raises a
    device-side assert that poisons the CUDA context for the rest of the sweep
    instead of failing one shape. Serving can only produce ratio-aligned
    prefixes, so aligning the sweep keeps it inside the serving envelope.

    Raises when the artifact declares no such field (the QSA route must not
    guess a group width)."""
    config_dir = _resolve_local_model_path(model_id)
    with open(os.path.join(config_dir, "config.json")) as f:
        config = json.load(f)
    text_config = config.get("text_config") or config
    value = text_config.get("indexer_compress_ratio") or config.get("indexer_compress_ratio")
    if not value or int(value) <= 1:
        raise RuntimeError(
            f"indexer_compress_ratio missing/empty in {model_id} config.json; "
            "cannot align the QSA sweep to the compressed-block width"
        )
    return int(value)


_MEMORY_BUDGET_SAFETY_FACTOR = 0.9
# The M3 KV pool page on the SM90/SM100 serving paths (fa3/fa4 + page 128,
# arg_groups/overrides.py:502-529@v0.5.16); page-128 rounding upper-bounds
# any smaller resolved page. Used for plan-time footprint estimates only —
# the worker re-validates against the real resolved page and pool.
_PLAN_PAGE_SIZE = 128
# Compressed QSA forces page_size=64 in SGLang's own Qwen4-Exp override
# (arg_groups/model_overrides/qwen4_exp.py@v0.5.21: compressed_slot ==
# full_slot // compress_ratio needs page-aligned full-KV allocation). 64 is
# also the smallest page that is a multiple of the compress ratio, so the
# estimate matches the resolved pool exactly.
_QSA_PLAN_PAGE_SIZE = 64
_M3_HEAD_DIM = 128
_M3_INDEX_HEADS = 4
_M3_INDEX_DIM = 128
_M3_NATIVE_GQA_RATIO = 16  # 64 q / 4 kv heads (bundled config)


def _loaded_num_layers() -> int:
    """Layers the worker loads (SGLANG_TEST_NUM_LAYERS, default 2) — every one of
    them owns a slot in the main pool and the index-K pool, so the plan-time
    footprint is per token x layers (memory_pool.py:5404 MiniMaxSparseKVPool
    builds one MHA pool over start_layer..end_layer plus one index pool per
    sparse layer @0.5.21)."""
    return int(os.environ.get("SGLANG_TEST_NUM_LAYERS", "2"))


def _estimated_kv_bytes_per_token(num_heads: int, kv_cache_dtype: str) -> int:
    """Upper-bound bytes/token of the M3 pools at this per-GPU head count, for
    ALL loaded layers: main paged K+V (kv-head sharded; fp8 stores 1 B/elem, else
    bf16) plus the index-K side cache (model dtype unless fp8 attn-GEMM,
    memory_pool.py:5392 get_minimax_sparse_index_dtype @0.5.21; index_dtype=
    model_dtype @v0.5.16 kv_cache_configurator.py:1246). The 0.5.16-era
    estimate priced one layer while the worker loads two: the three
    largest-batch decode cells (b=256/512/1024 x 131072/65536/32768) passed the
    plan filter and then failed the worker's execute-or-raise capacity check
    (33.6M planned tokens > 24.5M real pool, H20 0.5.21 smoke 2026-10-01)."""
    kv_heads = max(1, num_heads // _M3_NATIVE_GQA_RATIO)
    main_elem = 1 if kv_cache_dtype == "fp8" else 2
    main = 2 * kv_heads * _M3_HEAD_DIM * main_elem
    index_heads = min(_M3_INDEX_HEADS, num_heads)
    index = index_heads * _M3_INDEX_DIM * 2
    return (main + index) * _loaded_num_layers()


def _qsa_text_config(model_path: str) -> dict:
    """Return the Qwen4-Exp decoder config (outer config for a text-only export).

    QSA geometry is per checkpoint (GQA ratio, head_dim, index schema), so it
    is never inferred from the M3 constants above. A checkpoint without the
    indexer schema raises instead of producing M3-shaped rows under a QSA
    label.
    """
    config_dir = _resolve_local_model_path(model_path)
    with open(os.path.join(config_dir, "config.json")) as f:
        config = json.load(f)
    text_config = config.get("text_config") or config
    required = (
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "indexer_n_heads",
        "indexer_kv_heads",
        "indexer_head_dim",
        "indexer_budget",
        "indexer_compress_ratio",
    )
    missing = [name for name in required if text_config.get(name) is None]
    if missing:
        raise ValueError(
            f"model {model_path} is missing the QSA indexer/config fields {missing}; "
            "the msa_module collector only supports Qwen4-Exp checkpoints for "
            "attention_type qsa"
        )
    return text_config


def _qsa_bytes_per_token(num_heads: int, kv_cache_dtype: str, text_config: dict) -> int:
    """Pool bytes/token of the Qwen4-Exp QSA layer at this per-GPU head count.

    Mirrors QSAKVTokenToKVPool (mem_cache/qsa_kv_pool.py@v0.5.21): the paged
    main K/V of the full-attention layer (kv-head sharded; fp8 stores 1 B/elem,
    else bf16) plus the three QSA side buffers, all ``index_state_dtype`` bf16
    (class attribute, :36) except the int64 MRoPE position ring:

    * ``qsa_key_state_buffer_pool`` — one raw index key per compressed slot
      (``qsa_compressed_capacity = ceil(state_size / ratio)``, :124), so
      ``index_kv_heads * index_head_dim * 2 / ratio`` bytes per token;
    * ``qsa_compressed_k_buffer_pool`` — one compressed key per compressed
      page, same per-token cost (``qsa_compressed_page_size = page/ratio``,
      :123);
    * ``qsa_rope_position_buffer`` — three int64 MRoPE axes per token (:163).

    Per-request pending rings are budgeted with the other per-request buffers
    by SGLang, so they are deliberately excluded — this stays a lower bound
    like the M3 estimate.
    """
    ratio = int(text_config["indexer_compress_ratio"])
    index_state = int(text_config["indexer_kv_heads"]) * int(text_config["indexer_head_dim"]) * 2
    side = 2 * index_state // ratio + 3 * 8
    return _qsa_main_kv_bytes_per_token(num_heads, kv_cache_dtype, text_config) + side


def _qsa_main_kv_bytes_per_token(num_heads: int, kv_cache_dtype: str, text_config: dict) -> int:
    """Paged main K+V bytes/token of one QSA layer (kv-head sharded)."""
    native_q = int(text_config["num_attention_heads"])
    native_kv = int(text_config["num_key_value_heads"])
    gqa_ratio = max(1, native_q // native_kv)
    kv_heads = max(1, num_heads // gqa_ratio)
    main_elem = 1 if kv_cache_dtype == "fp8" else 2
    return 2 * kv_heads * int(text_config["head_dim"]) * main_elem


# Transient bytes/token the PREFIXED QSA extend path materialises on top of the
# pool, as a multiple of one layer's main K+V. For any row with a prefix the
# backend gathers that row's whole context out of the pool — one index_select
# per request, then a torch.cat of the per-request parts, and the same again
# for V (layers/attention/qwen_sparse_attn_backend.py:1326-1361@v0.5.21), so the
# peak is TWO full copies of the context K+V per token. It is a per-shape
# workspace, not part of the pool: the pool holds every request's context for
# the whole sweep, and this is allocated on top for each measured shape.
#
# Measured on H200 (C3.3, one shape per fresh process): 6407-6488 B/token total
# at TP1 against a 2200 B/token pool + 2*2048 transient, and 3610 B/token at TP8
# against 1176 + 2*1024 — i.e. pool + 2x, plus a small per-token path overhead.
_QSA_PREFIX_GATHER_COPIES = 2
# Plan-time headroom over that model: the measured totals run 1.03x (TP1) and
# 1.12x (TP8) of pool + transient, which is the rest of the path's per-token
# metadata (req->token tables, positions, topk indices). 1.15 covers the
# measured worst case and still admits every shape the device can run.
_QSA_PLAN_FOOTPRINT_MARGIN = 1.15
# Mirrors the worker's own max_total_tokens sizing (+5%, see run_msa_module):
# the pool it allocates is this much larger than the tokens the sweep needs.
_PLAN_POOL_SLACK = 1.05


def _qsa_prefix_transient_bytes_per_token(num_heads: int, kv_cache_dtype: str, text_config: dict) -> int:
    """Per-shape extend-path workspace bytes/token for a prefixed QSA chunk."""
    return _QSA_PREFIX_GATHER_COPIES * _qsa_main_kv_bytes_per_token(num_heads, kv_cache_dtype, text_config)


def _kv_cost_estimate(
    attention_type: str, num_heads: int, kv_cache_dtype: str, model_path: str
) -> tuple[int, int, int]:
    """(plan page size, pool bytes/token, per-shape workspace bytes/token)."""
    if attention_type == "qsa":
        text_config = _qsa_text_config(model_path)
        return (
            _QSA_PLAN_PAGE_SIZE,
            _qsa_bytes_per_token(num_heads, kv_cache_dtype, text_config),
            _qsa_prefix_transient_bytes_per_token(num_heads, kv_cache_dtype, text_config),
        )
    # MSA gathers nothing of its own: its pool estimate is the whole footprint.
    return _PLAN_PAGE_SIZE, _estimated_kv_bytes_per_token(num_heads, kv_cache_dtype), 0


def _qsa_footprint_refusal(
    shape: tuple,
    *,
    page_size: int,
    bytes_per_token: int,
    workspace_bytes: int,
    total_memory: int,
) -> str | None:
    """Why a QSA point's pool plus its per-shape gather workspace cannot fit, or None.

    The pool is sized from the case's largest runnable point and stays resident
    for the whole sweep, while the prefixed extend path allocates another two
    copies of the context K/V per measured shape on top of it
    (``_qsa_prefix_transient_bytes_per_token``). Bounding only the pool admits
    points that OOM mid-sweep: the C3.3 run aborted 215 of 11888 shapes that
    way, including b=128 prefix=262140 at ~197 GiB against a 104.67 GiB
    measured peer. This is a capacity refusal, so the worker raises it before
    allocating the point (layer_permissions.md: "refuse at run time, before
    allocation").

    The gather is the *prefixed* extend path's workspace: it concatenates each
    request's cached context (already in the pool) with the new chunk. A
    ``prefix_len == 0`` chunk writes its own K/V and attends over it directly,
    so it materialises no gather copy — charging one there refused points the
    device holds and left real gaps in the collected grid.
    """
    batch, seq_len, prefix_len = shape
    workspace = workspace_bytes if prefix_len else 0
    per_token = (bytes_per_token * _PLAN_POOL_SLACK + workspace) * _QSA_PLAN_FOOTPRINT_MARGIN
    needed = required_kv_alloc_tokens(batch, seq_len, prefix_len, page_size, is_prefill=True) * per_token
    allowed = int(total_memory * _MEMORY_BUDGET_SAFETY_FACTOR)
    if needed <= allowed:
        return None
    footprint = "the pool" + (f" plus the {workspace_bytes} B/token prefix gather" if workspace else "")
    return (
        f"{KERNEL_LIMIT}: does not fit: {footprint} "
        f"needs {needed / 2**30:.1f} GiB, the device allows {allowed / 2**30:.1f} GiB "
        f"({_MEMORY_BUDGET_SAFETY_FACTOR:g} of its memory)"
    )


def _require_aligned_qsa_prefixes(shapes, *, model_path: str) -> None:
    """Refuse a QSA prefix the compressed pool cannot address, on the host.

    The plan already raises every prefix to a group boundary; an ad-hoc shape
    (``--quick``/``--prefix-len``) can still be misaligned, and the device-side
    assert it would hit poisons the CUDA context for the rest of the sweep.
    """
    ratio = _qsa_compress_ratio(model_path)
    misaligned = [shape for shape in shapes if shape[2] % ratio]
    if misaligned:
        raise RuntimeError(
            f"QSA manifest carries prefixes outside the compressed-block width {ratio}: "
            f"{misaligned[:3]} (model={model_path}); the write plan asserts prefix % ratio == 0 "
            "on the device and a violation poisons the CUDA context for every remaining shape"
        )


def _resolve_model_spec(model_path: str):
    """The declared mla_module model row for a checkpoint path (aliases included).

    A targeted run may name an artifact alias (for example the FP8 checkpoint)
    of a row whose representative ``model_path`` is the base artifact —
    ``model_aliases`` share one physical collector case. An undeclared path
    raises: guessing would run a MiniMax module under a Qwen4-Exp row or
    vice versa.
    """
    specs = get_mla_module_model_specs(backend="sglang", apply_model_filter=False)
    exact = [spec for spec in specs if spec.model_path == model_path]
    if not exact:
        exact = [
            spec for spec in specs if model_path in get_mla_module_model_aliases(spec.model_path, backend="sglang")
        ]
    if len(exact) != 1:
        declared = sorted({spec.model_path for spec in specs})
        raise ValueError(
            f"model {model_path!r} resolves to {len(exact)} mla_module model row(s); declared paths: {declared}"
        )
    return exact[0]


def _attention_type_for_model(model_path: str) -> str:
    """Resolve a case's attention type from its declared model row.

    The case tuple keeps the trtllm/sglang positional prefix (no
    attention_type field), so the worker looks it up here — the same model
    identity the perf row persists.
    """
    return _resolve_model_spec(model_path).attention_type


def get_msa_context_module_test_cases():
    return plan_cases("msa_context_module", "sglang", get_sm_version())


def get_msa_generation_module_test_cases():
    return plan_cases("msa_generation_module", "sglang", get_sm_version())


# ═══════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════


def _register_minimax_m3_text_config():
    """Make transformers AutoConfig accept model_type="minimax_m3".

    The bundled text-only config keeps the checkpoint's raw schema
    (model_type "minimax_m3"), which neither transformers 5.12 nor SGLang's
    _CONFIG_REGISTRY knows. Serving materialises the M3 text config as a
    generic attribute bag: SGLang's own VL config coerces the checkpoint's
    text_config dict through CONFIG_MAPPING.get(model_type, PretrainedConfig)
    — an unknown model_type "falls back to PretrainedConfig so dict keys
    still become real attributes" (configs/minimax_vl.py:10-23), and
    the M3 modeling code reads plain attributes only. Registering a bare
    PretrainedConfig alias therefore changes HOW the config object is
    constructed, never which kernel runs (M3 dispatch keys on
    architectures[0] + sparse_attention_config, both preserved verbatim).
    Same shim precedent as collector/trtllm/collect_msa_module.py.
    """
    import transformers

    class _MiniMaxM3TextConfig(transformers.PretrainedConfig):
        model_type = "minimax_m3"

    try:
        transformers.AutoConfig.register("minimax_m3", _MiniMaxM3TextConfig)
    except ValueError as e:  # already registered by an earlier case in this proc
        if "already" not in str(e).lower():
            raise


def _resolve_sglang_model_dir(model_id: str) -> str:
    """Resolve the bundled M3 config and present it under SGLang's identity.

    ``collector.helper._resolve_local_model_path`` materialises the bundled
    AIC config (auto_map stripped). That config carries AIC's canonical
    ``architectures=["MiniMaxM3ForCausalLM"]``; SGLang's text-decoder entry
    class is ``MiniMaxM3SparseForCausalLM`` — exactly what the official
    checkpoint's own text_config.architectures declares (see
    SGLANG_MSA_ARCHITECTURE above). Rewrite only that field so SGLang loads
    the same model class serving uses; every other field stays verbatim.
    """
    base_dir = _resolve_local_model_path(model_id)
    with open(os.path.join(base_dir, "config.json")) as f:
        config = json.load(f)

    if config.get("architectures") not in (
        [MSA_ARCHITECTURE],
        [SGLANG_MSA_ARCHITECTURE],
    ):
        raise ValueError(
            f"bundled config for {model_id} declares architectures="
            f"{config.get('architectures')!r}; expected the MiniMax-M3 text decoder"
        )
    if not config.get("sparse_attention_config"):
        raise ValueError(
            f"model {model_id} has no sparse_attention_config; the MSA module "
            "collector only supports MiniMax-M3 sparse-attention checkpoints"
        )
    config["architectures"] = [SGLANG_MSA_ARCHITECTURE]

    tmp_dir = os.path.join(
        tempfile.gettempdir(),
        f"aic_sglang_msa_config_{model_id.replace('/', '_')}",
    )
    os.makedirs(tmp_dir, exist_ok=True)
    target = os.path.join(tmp_dir, "config.json")
    tmp_target = f"{target}.{os.getpid()}.tmp"
    with open(tmp_target, "w") as f:
        json.dump(config, f)
    os.replace(tmp_target, target)
    return tmp_dir


def _build_model_override_args(config: dict, num_heads: int, target_tp_size: int, num_layers: int) -> dict:
    """Single-GPU emulation of one target-TP rank + benchmark-layer surgery.

    Head shrinking mirrors SGLang's own TP sharding math (citations at the
    top of this file: models/minimax_m3.py:749-762 q/kv, :784-796 index
    heads). The per-layer pattern arrays are truncated so every loaded layer
    takes the sparse construction path of serving layers 3+
    (is_sparse_attention_layer=True, disable_index_value=True —
    models/minimax_m3.py:1702-1721 reading get_minimax_sparse_layer_ids /
    get_minimax_sparse_disable_value_layer_ids,
    configs/model_config.py:460-471). The MLP is made dense
    (moe_layer_freq=0 → MiniMaxM3MLP, models/minimax_m3.py:1725-1744,
    1791-1793) — it is outside the timed attention module; same single-layer
    surgery as the trtllm MSA collector. All sparse scalar fields stay
    checkpoint-verbatim.
    """
    native_q = int(config["num_attention_heads"])
    native_kv = int(config["num_key_value_heads"])
    if native_q % target_tp_size != 0 or native_q // target_tp_size != num_heads:
        raise ValueError(f"num_heads={num_heads} does not equal native {native_q} / target_tp {target_tp_size}")
    local_kv = max(1, native_kv // target_tp_size)

    sparse_cfg = dict(config["sparse_attention_config"])
    native_idx = int(sparse_cfg["sparse_num_index_heads"])
    local_idx = max(1, native_idx // target_tp_size)
    sparse_cfg["sparse_attention_freq"] = [1] * num_layers
    # Serving layers 3+ all disable the index value path
    # (sparse_disable_index_value[3:] == 1 in the checkpoint config); the
    # benchmark layer replicates that flavor.
    sparse_cfg["sparse_disable_index_value"] = [1] * num_layers
    sparse_cfg["sparse_num_index_heads"] = local_idx

    return {
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": local_kv,
        "moe_layer_freq": [0] * num_layers,
        "sparse_attention_config": sparse_cfg,
    }


def _expect(actual, expected, what: str):
    if actual != expected:
        raise RuntimeError(f"{what}={actual!r}, expected {expected!r}")


def _validate_msa_module(model_runner, attn, num_heads: int, target_tp_size: int, head_dim: int):
    """Verify the single-GPU module emulates one target-TP rank of serving.

    Shapes per models/minimax_m3.py: qkv_proj packs
    (num_heads + 2*num_kv_heads)*head_dim (:801-810), o_proj input
    num_heads*head_dim (:812-820), index_qkv_proj packs
    (num_idx_heads + 1)*idx_head_dim with v_head_size=0 when
    disable_index_value (:830-841), index_o_proj is None (:843-844). The
    projection serving runs is their concatenation (_FusedQKVIndexProj,
    :245-284, built by maybe_build_fused_qkv_index :1117-1158 for a CUDA sparse
    layer whose two projections share one quant method, as BF16 ones do).
    """
    local_kv = max(1, num_heads // 16)  # native 64q/4kv → GQA ratio 16
    local_idx = max(1, 4 // target_tp_size)
    if not getattr(attn, "is_sparse_attention_layer", False):
        raise RuntimeError("benchmark layer came out dense; sparse_attention_freq surgery did not take")
    if not getattr(attn, "disable_index_value", False):
        raise RuntimeError("benchmark layer keeps the index-value path; serving layers 3+ disable it")
    _expect(int(attn.num_heads), num_heads, "self_attn.num_heads")
    _expect(int(attn.num_kv_heads), local_kv, "self_attn.num_kv_heads")
    _expect(int(attn.num_idx_heads), local_idx, "self_attn.num_idx_heads")
    _expect(int(attn.head_dim), head_dim, "self_attn.head_dim")
    _expect(
        int(attn.qkv_proj.output_size_per_partition),
        (num_heads + 2 * local_kv) * head_dim,
        "qkv_proj.output_size_per_partition",
    )
    _expect(
        int(attn.o_proj.input_size_per_partition),
        num_heads * head_dim,
        "o_proj.input_size_per_partition",
    )
    _expect(
        int(attn.index_qkv_proj.output_size_per_partition),
        (local_idx + 1) * int(attn.idx_head_dim),
        "index_qkv_proj.output_size_per_partition",
    )
    if attn.index_o_proj is not None:
        raise RuntimeError("index_o_proj exists despite disable_index_value=True")
    fused = getattr(attn, "_fused_qkv_index", None)
    if fused is None:
        raise RuntimeError(
            "benchmark layer has no fused qkv+index projection; serving builds it after loading "
            "(models/minimax_m3.py:1117-1158, 2244-2246) and forward_prepare runs it (:1491-1551)"
        )
    _expect(
        int(fused.output_size_per_partition),
        (num_heads + 2 * local_kv) * head_dim + (local_idx + 1) * int(attn.idx_head_dim),
        "fused_qkv_index_proj.output_size_per_partition",
    )

    backend = model_runner.attn_backend
    sparse = getattr(backend, "sparse", None)
    if sparse is None or attn.attn.layer_id not in getattr(backend, "sparse_layer_ids", ()):
        raise RuntimeError(
            "benchmark layer does not route to MiniMaxSparseAttnBackend; "
            "attn_backend_wrapper did not wrap the runner (attention_registry.py:375-391)"
        )


def cleanup_distributed():
    """Clean up SGLang distributed process-group state between subprocesses."""
    import sglang.srt.distributed.parallel_state as parallel_state

    try:
        parallel_state.destroy_model_parallel()
    except Exception:
        pass

    # destroy_model_parallel only tears down the model-parallel groups; the
    # WORLD process group (and, on 0.5.21+, the published runtime placement)
    # outlives it, which would make the next runner in this process see a stale
    # world size and a stale init guard. Tear both down when the runtime has
    # them.
    if hasattr(parallel_state, "destroy_distributed_environment"):
        try:
            parallel_state.destroy_distributed_environment()
        except Exception:
            pass

    # v0.5.21 replaced the caller-supplied ``ParallelState`` with a
    # process-wide runtime context (``sglang.srt.runtime_context``). The
    # context is process state, not process-group state, so ModelRunner
    # teardown leaves it behind and the NEXT runner in this worker would read
    # the previous run's published config. Reset it when the runtime has one;
    # older SGLang has no such module and needs nothing here.
    try:
        from sglang.srt.runtime_context import reset_context
    except ImportError:
        return
    try:
        reset_context()
    except Exception:
        pass


def _resolved_page_size(server_args):
    """The page size the runner will actually use.

    The Qwen4-Exp override declares ``page_size=64`` for the compressed QSA
    pool (arg_groups/model_overrides/qwen4_exp.py@v0.5.21). Through 0.5.20 that
    landed on the ``ServerArgs`` record itself; 0.5.21 resolves declarations
    into the published schedule bag and leaves the record's raw value
    untouched, while ``ModelRunner`` reads ``get_schedule().page_size``. Read
    the same field the runner reads, so "verified after load" means verified.
    """
    try:
        from sglang.srt.runtime_context import get_schedule
    except ImportError:
        return server_args.page_size
    return getattr(get_schedule(), "page_size", None) or server_args.page_size


def _start_model_runner(
    *,
    local_model_path: str,
    override_args: dict,
    sglang_kv_dtype: str,
    gpu_id: int,
    max_total_tokens: int | None,
    chunked_prefill_size: int | None,
    max_running_requests: int | None,
):
    """ServerArgs → ModelConfig → ModelRunner, as serving constructs them.

    Serving-selected defaults are deliberately NOT overridden: attention
    backend, page size and the fp8 attention-GEMM mode resolve exactly as for
    a server launched with only --kv-cache-dtype (the M3 override and the
    generic default, cited at the top of this file). The prefill CUDA graph
    config is left at serving's default too; this collector never captures a
    prefill graph (deviation note at _run_prefill_point).

    max_running_requests / max_total_tokens / mem_fraction_static are
    benchmark-harness resource knobs (user-tunable ServerArgs, not kernel
    dispatch). run_msa_module leaves chunked_prefill_size to SGLang, which
    derives it from device memory as serving does, and refuses a prefill
    beyond it; mem_fraction_static is pinned because SGLang's heuristic
    couples it to chunked_prefill_size (arg_groups/memory_hook.py
    _handle_gpu_memory_settings) and would go negative.
    """
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import suppress_other_loggers

    suppress_other_loggers()

    load_format = os.environ.get("SGLANG_LOAD_FORMAT", "dummy")

    server_args = ServerArgs(
        model_path=local_model_path,
        dtype="auto",
        device="cuda",
        load_format=load_format,
        tp_size=1,
        trust_remote_code=True,
        disable_radix_cache=True,
        kv_cache_dtype=sglang_kv_dtype,
        attention_backend=None,
        max_total_tokens=max_total_tokens,
        chunked_prefill_size=chunked_prefill_size,
        max_prefill_tokens=(chunked_prefill_size or 16384),
        mem_fraction_static=0.80,
        max_running_requests=max_running_requests,
        json_model_override_args=json.dumps(override_args),
    )

    _set_envs_and_config(server_args)

    model_config = ModelConfig.from_server_args(server_args)

    from collector.sglang.runtime_compat import free_local_port, init_runtime_config

    nccl_port = free_local_port()
    _runner_parallel_kwargs = init_runtime_config(server_args, gpu_id, nccl_port=nccl_port, model_config=model_config)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        nccl_port=nccl_port,
        **_runner_parallel_kwargs,
        server_args=server_args,
    )
    # Serving's load_weights ends by fusing each sparse layer's qkv_proj and
    # index_qkv_proj into one GEMM (models/minimax_m3.py:2244-2246), which
    # forward_prepare then runs with the grouped QK norm/RoPE kernel
    # (:1491-1551). The dummy loader never calls load_weights
    # (model_loader/loader.py:1493-1530; M3 has no post_load_weights hook,
    # :315-321), so without this step the module times two GEMMs and two
    # norm/RoPE kernels. The BF16 weights need no post-processing on CUDA
    # (layers/quantization/unquant.py:568-570), so fusing after the dummy
    # loader's pass gives serving's weights.
    from sglang.srt.models.minimax_m3 import build_minimax_fused_qkv_index

    build_minimax_fused_qkv_index(model_runner.model)
    model_runner.alloc_memory_pool()
    model_runner.init_attention_backends()
    return model_runner, server_args, model_config


def load_model_runner(
    model_path: str,
    num_heads: int,
    kv_cache_dtype: str,
    target_tp_size: int,
    max_total_tokens: int | None,
    chunked_prefill_size: int | None,
    max_running_requests: int | None,
    device: str = "cuda:0",
    attention_type: str = "msa",
):
    """Load an SGLang ModelRunner hosting one sparse-attention module.

    ``attention_type`` selects the route: ``msa`` (MiniMax-M3 sparse layer,
    ``_load_msa_model_runner``) or ``qsa`` (Qwen4-Exp compressed-QSA layer,
    ``_load_qsa_model_runner``). Both build dummy weights of the real serving
    layer and validate the shard.
    """
    if attention_type not in _MSA_MODULE_ATTENTION_TYPES:
        raise ValueError(f"unknown attention_type {attention_type!r}; expected one of {_MSA_MODULE_ATTENTION_TYPES}")
    loader = _load_qsa_model_runner if attention_type == "qsa" else _load_msa_model_runner
    return loader(
        model_path=model_path,
        num_heads=num_heads,
        kv_cache_dtype=kv_cache_dtype,
        target_tp_size=target_tp_size,
        max_total_tokens=max_total_tokens,
        chunked_prefill_size=chunked_prefill_size,
        max_running_requests=max_running_requests,
        device=device,
    )


def _load_msa_model_runner(
    *,
    model_path: str,
    num_heads: int,
    kv_cache_dtype: str,
    target_tp_size: int,
    max_total_tokens: int | None,
    chunked_prefill_size: int | None,
    max_running_requests: int | None,
    device: str = "cuda:0",
):
    """MiniMax-M3 route: serving-selected attention backend + page size.

    The M3 server-args handler resolves attention_backend / page_size per
    platform (arg_groups/overrides.py:465-537@v0.5.16 — fa3 + page 128 on
    SM90, fa4 + page 128 on SM100) inside ServerArgs.__post_init__
    (server_args.py:2969 _handle_model_specific_adjustments); nothing here
    overrides that.
    """
    device_str = str(device)
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0

    num_layers = int(os.environ.get("SGLANG_TEST_NUM_LAYERS", "2"))

    sglang_kv_dtype = SGLANG_KV_DTYPE.get(kv_cache_dtype)
    if sglang_kv_dtype is None:
        raise ValueError(f"unsupported kv_cache_dtype {kv_cache_dtype!r} for MSA (SGLANG_KV_DTYPE)")

    _register_minimax_m3_text_config()
    local_model_path = _resolve_sglang_model_dir(model_path)
    with open(os.path.join(local_model_path, "config.json")) as f:
        raw_config = json.load(f)
    override_args = _build_model_override_args(raw_config, num_heads, target_tp_size, num_layers)

    model_runner, server_args, model_config = _start_model_runner(
        local_model_path=local_model_path,
        override_args=override_args,
        sglang_kv_dtype=sglang_kv_dtype,
        gpu_id=gpu_id,
        max_total_tokens=max_total_tokens,
        chunked_prefill_size=chunked_prefill_size,
        max_running_requests=max_running_requests,
    )
    actual_architecture = (model_config.hf_config.architectures or [None])[0]
    if actual_architecture != SGLANG_MSA_ARCHITECTURE:
        raise RuntimeError(
            f"SGLang loaded architecture={actual_architecture!r} for {model_path}, expected {SGLANG_MSA_ARCHITECTURE!r}"
        )

    head_dim = int(raw_config.get("head_dim", raw_config["hidden_size"] // raw_config["num_attention_heads"]))
    attn = model_runner.model.model.layers[0].self_attn
    _validate_msa_module(model_runner, attn, num_heads, target_tp_size, head_dim)

    sparse = model_runner.attn_backend.sparse
    print(
        f"SGLang resolved: dense={type(model_runner.attn_backend.dense).__name__}, "
        f"attention_backend={resolved_arg(server_args, 'attention_backend')}, page_size={sparse.page_size}, "
        f"kv={sparse.kv_pool.main_pool.dtype}, index_kv={getattr(sparse.kv_pool.index_k_pool, 'dtype', None)}, "
        f"fp8_attn_gemm={sparse.fp8_attn_gemm}, use_msa={sparse.use_msa}, msa_decode={sparse._use_msa_decode}, "
        f"chunked_prefill_size={resolved_arg(server_args, 'chunked_prefill_size')}, "
        f"max_total_tokens={max_total_tokens}, max_running_requests={max_running_requests}"
    )
    return model_runner


def _qsa_local_kv_heads(text_config: dict, target_tp_size: int) -> int:
    """K/V heads on one target-TP rank, exactly as serving shards them.

    Qwen3_5AttentionDecoderLayer.__init__: when ``total_kv_heads >= kv_tp_size``
    the heads divide; otherwise they are replicated, i.e.
    ``num_kv_heads = max(1, total_kv // kv_tp_size)`` (and it asserts
    ``kv_tp_size % total_kv == 0``). Qwen3.8-Flash-Next has 2 native K/V heads,
    so TP 2 keeps 1 and TP 4/8 replicate 1.
    """
    native_kv = int(text_config["num_key_value_heads"])
    if native_kv < target_tp_size and target_tp_size % native_kv != 0:
        raise ValueError(
            f"target_tp_size={target_tp_size} cannot shard or replicate "
            f"num_key_value_heads={native_kv} the way serving does"
        )
    return max(1, native_kv // target_tp_size)


def _build_qsa_model_override_args(config: dict, num_heads: int, target_tp_size: int, num_layers: int) -> dict:
    """Single-GPU emulation of one target-TP rank + benchmark-stack surgery.

    The override is nested under ``text_config`` because the checkpoint is a
    real multimodal Qwen4-Exp config: ``get_config`` applies a dict override to
    a PretrainedConfig sub-config with ``.update()`` (utils/hf_transformers/
    config.py:324-335@v0.5.21), which is exactly what serving's config path
    does for these checkpoints. The outer config is left untouched so
    ``language_model_only`` stays false and the loaded model keeps
    ``is_mrope_enabled``/``model_is_mrope`` on (the QSA bench must feed MRoPE
    positions, see ``_module_positions``).

    Inside the text config:

    * ``num_attention_heads = num_heads`` and
      ``num_key_value_heads = max(1, native_kv // tp)`` reproduce serving's
      per-rank shard (Qwen3_5AttentionDecoderLayer.__init__). The indexer is
      replicated, not sharded (``indexer_n_heads`` stays 4), matching
      QSAIndexer@v0.5.21.
    * ``num_hidden_layers = 4`` with ``layer_types`` set to three linear (GDN)
      layers plus one ``full_attention`` layer — the checkpoint's own
      ``full_attention_interval=4`` period. ``layers_block_type`` and
      ``full_attention_layer_ids`` are properties over ``layer_types``
      (qwen4_exp config + qwen3_next config:260-285), so the hybrid backend
      wrapper wires layer 3 to ``QwenSparseAttnBackend`` with no other surgery.
    * ``ple_layer_ids = []`` drops the PLE tables. They are model-global host
      state, not part of the timed attention module, and their real width
      (vocab x 18M-table projections) is out of scope for a module bench.
      With no PLE layer, ``has_ple`` is false and no ``ple_batch`` is needed.
    * ``num_experts`` is shrunk to ``QSA_BENCH_NUM_EXPERTS``. The routed MoE
      block sits outside the timed module; every loaded layer still builds the
      real ``Qwen2MoeSparseMoeBlock`` with checkpoint-verbatim top-k, just
      over fewer dummy experts, so dummy residency stays small.
    """
    if "text_config" not in config:
        raise ValueError(
            "Qwen4-Exp QSA rows require the checkpoint's multimodal config "
            "(nested text_config); a text-only export would turn off "
            "is_mrope_enabled and change the measured attention path"
        )
    text_config = config["text_config"]
    native_q = int(text_config["num_attention_heads"])
    if native_q % target_tp_size != 0 or native_q // target_tp_size != num_heads:
        raise ValueError(f"num_heads={num_heads} does not equal native {native_q} / target_tp {target_tp_size}")
    if len(QSA_BENCH_LAYER_TYPES) != num_layers:
        raise ValueError(f"QSA benchmark stacks {len(QSA_BENCH_LAYER_TYPES)} layer types, got num_layers={num_layers}")
    return {
        "text_config": {
            "num_hidden_layers": num_layers,
            "layer_types": list(QSA_BENCH_LAYER_TYPES),
            "ple_layer_ids": [],
            "num_attention_heads": num_heads,
            "num_key_value_heads": _qsa_local_kv_heads(text_config, target_tp_size),
            "num_experts": QSA_BENCH_NUM_EXPERTS,
        }
    }


def _validate_qsa_module(model_runner, layer, num_heads: int, target_tp_size: int, text_config: dict):
    """Verify the single-GPU module emulates one target-TP QSA rank of serving.

    Shapes per Qwen3_5AttentionDecoderLayer + Qwen4Exp override
    (@v0.5.21): qkv_proj packs
    ``(num_heads * (1 + attn_output_gate) + 2 * num_kv_heads) * head_dim``
    (Qwen4-Exp forces ``attn_output_gate = True``), o_proj input
    ``num_heads * head_dim``, and the indexer holds the checkpoint's index
    schema verbatim (replicated across TP). Dispatch is proven, not assumed:
    the layer must be the QSA flavor, the hybrid wrapper's full-attention side
    must be ``QwenSparseAttnBackend``, and the loaded pool must be the QSA
    hybrid pool at the compressed page size SGLang pins.
    """
    head_dim = int(text_config["head_dim"])
    local_kv = _qsa_local_kv_heads(text_config, target_tp_size)
    # Qwen3_5AttentionDecoderLayer keeps qkv_proj / o_proj / num_heads on the
    # LAYER (models/qwen3_5.py:1184-1247@v0.5.21) and Qwen4-Exp adds
    # ``is_qsa`` / ``indexer`` there too (models/qwen4_exp.py:1535-1543); the
    # M3 route's ``layer.self_attn`` attribute does not exist on this model.
    if not getattr(layer, "is_qsa", False):
        raise RuntimeError(
            "benchmark layer did not come out QSA; the layer_types surgery or "
            "the checkpoint's indexer schema did not take"
        )
    _expect(int(layer.num_heads), num_heads, "layer.num_heads")
    _expect(int(layer.num_kv_heads), local_kv, "layer.num_kv_heads")
    _expect(int(layer.head_dim), head_dim, "layer.head_dim")
    gate_heads = 2 if getattr(layer, "attn_output_gate", False) else 1
    _expect(
        int(layer.qkv_proj.output_size_per_partition),
        (gate_heads * num_heads + 2 * local_kv) * head_dim,
        "qkv_proj.output_size_per_partition",
    )
    _expect(
        int(layer.o_proj.input_size_per_partition),
        num_heads * head_dim,
        "o_proj.input_size_per_partition",
    )

    indexer = getattr(layer, "indexer", None)
    if indexer is None:
        raise RuntimeError("QSA layer has no indexer")
    _expect(int(indexer.index_n_heads), int(text_config["indexer_n_heads"]), "indexer.index_n_heads")
    _expect(int(indexer.index_kv_heads), int(text_config["indexer_kv_heads"]), "indexer.index_kv_heads")
    _expect(int(indexer.index_head_dim), int(text_config["indexer_head_dim"]), "indexer.index_head_dim")
    _expect(int(indexer.token_topk), int(text_config["indexer_budget"]), "indexer.token_topk")
    _expect(
        int(indexer.compress_ratio),
        int(text_config["indexer_compress_ratio"]),
        "indexer.compress_ratio",
    )
    _expect(
        int(indexer.index_qk_proj.output_size),
        (int(indexer.index_n_heads) + int(indexer.index_kv_heads)) * int(indexer.index_head_dim),
        "indexer.index_qk_proj.output_size",
    )

    if not model_runner.model_config.model_is_mrope:
        raise RuntimeError(
            "loaded Qwen4-Exp model is not MRoPE; the QSA bench feeds "
            "[3, tokens] positions and a non-mrope load would measure a "
            "different attention path"
        )

    backend = model_runner.attn_backend
    full_attn_backend = getattr(backend, "full_attn_backend", None)
    if type(full_attn_backend).__name__ != "QwenSparseAttnBackend":
        raise RuntimeError(
            "hybrid attention wrapper's full-attention backend is "
            f"{type(full_attn_backend).__name__}, expected QwenSparseAttnBackend; "
            "attn_backend_wrapper did not take the is_qwen_qsa branch "
            "(attention_registry.py:458-466@v0.5.21)"
        )
    if int(layer.layer_id) not in getattr(backend, "full_attn_layers", ()):
        raise RuntimeError(
            f"benchmark layer {layer.layer_id} is not in the hybrid wrapper's "
            f"full_attn_layers {getattr(backend, 'full_attn_layers', None)}; the "
            "layer_types surgery did not produce a QSA layer"
        )
    if getattr(full_attn_backend, "qsa_profile", None) is None:
        raise RuntimeError("QwenSparseAttnBackend parsed no QSA profile from the loaded config")

    pool = model_runner.token_to_kv_pool
    ratio = int(text_config["indexer_compress_ratio"])
    page_size = kv_pool_page_size(model_runner)
    _expect(int(getattr(pool, "qsa_compress_ratio", -1)), ratio, "kv_pool.qsa_compress_ratio")
    _expect(
        int(getattr(pool, "qsa_compressed_page_size", -1)),
        page_size // ratio,
        "kv_pool.qsa_compressed_page_size",
    )
    if page_size % ratio != 0 or page_size == 1:
        raise RuntimeError(
            f"QSA requires a page-aligned pool (page={page_size}, ratio={ratio}); "
            "SGLang's qwen4_exp server-args override pins page_size=64"
        )
    main_dtype = pool.get_key_buffer(int(layer.layer_id)).dtype
    if main_dtype != torch.bfloat16:
        raise RuntimeError(
            f"QSA main KV cache is {main_dtype}, expected bfloat16: both "
            "Qwen3.8-Flash-Next checkpoints keep the QSA layer and its KV in BF16"
        )
    index_dtype = pool.get_qsa_key_state_buffer(int(layer.layer_id)).dtype
    if index_dtype != torch.bfloat16:
        raise RuntimeError(
            f"QSA index-state buffer is {index_dtype}, expected bfloat16 (QSAKVTokenToKVPool.index_state_dtype@v0.5.21)"
        )


def _qsa_module(model_runner):
    """The timed unit: the QSA layer's ``self_attention`` bound method."""
    layer = model_runner.model.model.layers[QSA_BENCH_LAYER]
    if not getattr(layer, "is_qsa", False):
        raise RuntimeError(f"layer {QSA_BENCH_LAYER} of the loaded benchmark stack is not a QSA layer")
    return layer.self_attention


def _module_hidden_size(model_runner, attention_type: str) -> int:
    """Hidden width the benchmarked module consumes.

    ``self_attention`` is fed the hyper-connection-mixed hidden state, i.e.
    ``config.hidden_size`` (the hc widening happens in
    ``_prepare_qwen4_exp_attn`` and is undone by ``attn_hyper_connection.mix``
    before the attention call). Read it from the layer so the multimodal outer
    config (which has no ``hidden_size``) is never consulted.

    The QSA route times layer ``QSA_BENCH_LAYER``; the MSA route times layer 0
    of a stack that may be shorter than the QSA benchmark layer.
    """
    layers = model_runner.model.model.layers
    for layer_id in (QSA_BENCH_LAYER, 0) if attention_type == "qsa" else (0,):
        if layer_id >= len(layers):
            continue
        hidden_size = getattr(layers[layer_id], "hidden_size", None)
        if hidden_size:
            return int(hidden_size)
    return int(model_runner.model.config.hidden_size)


def _load_qsa_model_runner(
    *,
    model_path: str,
    num_heads: int,
    kv_cache_dtype: str,
    target_tp_size: int,
    max_total_tokens: int | None,
    chunked_prefill_size: int | None,
    max_running_requests: int | None,
    device: str = "cuda:0",
):
    """Qwen4-Exp QSA route: real checkpoint config, 4-layer serving-shaped stack.

    No config rewrite is needed (unlike M3): the bundled checkpoint config
    already declares ``architectures=["Qwen4ExpForConditionalGeneration"]``,
    which is SGLang's EntryClass, and the multimodal config keeps
    ``is_mrope_enabled``. SGLang's own qwen4_exp server-args override pins
    ``page_size=64`` for the compressed pool
    (arg_groups/model_overrides/qwen4_exp.py@v0.5.21); the resolved value is
    verified after load rather than assumed.
    """
    device_str = str(device)
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0

    sglang_kv_dtype = SGLANG_KV_DTYPE.get(kv_cache_dtype)
    if sglang_kv_dtype is None:
        raise ValueError(f"unsupported kv_cache_dtype {kv_cache_dtype!r} for QSA (SGLANG_KV_DTYPE)")

    local_model_path = _resolve_local_model_path(model_path)
    with open(os.path.join(local_model_path, "config.json")) as f:
        raw_config = json.load(f)
    text_config = _qsa_text_config(model_path)
    override_args = _build_qsa_model_override_args(raw_config, num_heads, target_tp_size, QSA_BENCH_NUM_LAYERS)

    model_runner, server_args, model_config = _start_model_runner(
        local_model_path=local_model_path,
        override_args=override_args,
        sglang_kv_dtype=sglang_kv_dtype,
        gpu_id=gpu_id,
        max_total_tokens=max_total_tokens,
        chunked_prefill_size=chunked_prefill_size,
        max_running_requests=max_running_requests,
    )
    actual_architecture = (model_config.hf_config.architectures or [None])[0]
    if actual_architecture != QSA_ARCHITECTURE:
        raise RuntimeError(
            f"SGLang loaded architecture={actual_architecture!r} for {model_path}, expected {QSA_ARCHITECTURE!r}"
        )
    _expect(int(_resolved_page_size(server_args)), _QSA_PLAN_PAGE_SIZE, "schedule.page_size")

    _validate_qsa_module(
        model_runner,
        model_runner.model.model.layers[QSA_BENCH_LAYER],
        num_heads,
        target_tp_size,
        text_config,
    )
    return model_runner


# ═══════════════════════════════════════════════════════════════════════
# ForwardBatch construction + benchmarking
# ═══════════════════════════════════════════════════════════════════════


class PerfLogWriteError(RuntimeError):
    """Fail the subprocess when a measured row was not durably persisted."""


def _forward_ctx(model_runner, forward_batch):
    """Publish SGLang's per-forward control context, exactly as serving does
    around every model forward (model_executor/model_runner.py:1973-1981).
    RadixAttention resolves the attention backend through it
    (layers/radix_attention.py:183, model_executor/forward_context.py:73-74)."""
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

    return forward_context(ForwardContext(attn_backend=model_runner.attn_backend), forward_batch=forward_batch)


def _make_reqs(batch_size: int, full_length: int):
    """Requests with a full_length-token prompt, built as SGLang's offline batch
    builder builds them (benchmark/one_batch.py:450-476): fill ids are the
    prompt and this round extends [prefix_len=0, extend_end) (managers/
    schedule_batch.py:1103-1106, 1218-1220). The token ids are never embedded."""
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    reqs = []
    for i in range(batch_size):
        # A bulk copy: per-token Python ints cost seconds at 1M-token prompts.
        prompt = array("q")
        prompt.frombytes(torch.randint(0, 10000, (full_length,), dtype=torch.int64).numpy().tobytes())
        req = Req(
            rid=str(i),
            origin_input_text="",
            origin_input_ids=prompt,
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.full_untruncated_fill_ids = array("q", prompt)
        req.logprob_start_len = 0
        req.extend_end = full_length
        reqs.append(req)
    return reqs


def _radix_disabled_tree_cache(model_runner):
    """The prefix cache serving runs with --disable-radix-cache (load_model_runner sets it).

    v0.5.23 deleted ChunkCache (sglang #43201). For a full-attention-only model a
    disabled cache is a UnifiedRadixCache with disable=True and the FULL component
    (mem_cache/registry.py:83-98 -> create_unified_radix_cache :149-191;
    CacheInitParams as mem_cache/kv_cache_builder.py:389-402 builds it, page_size
    the allocator's without DCP). It matches, inserts and evicts nothing. Same
    construction as collect_mla_module._radix_disabled_tree_cache.
    """
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    return UnifiedRadixCache(
        CacheInitParams(
            disable=True,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
            page_size=model_runner.token_to_kv_pool_allocator.page_size,
            tree_components=(ComponentType.FULL,),
        )
    )


def _make_schedule_batch(model_runner, reqs, tree_cache):
    """A non-overlap, non-speculative batch, as the scheduler builds one (ScheduleBatch.init_new)."""
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    return ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )


def _sglang_capture_context():
    """SGLang's own CUDA-graph capture scope (a no-op on runtimes without one).

    Model code branches on ``get_is_capture_mode()``: the QSA indexer skips its
    cos/sin cache resize (qsa_indexer.py:175,401@v0.5.21) and the layer overlaps
    the indexer onto ``alt_stream`` (qwen4_exp.py:1593) only while a graph is
    being captured, and ``project_qk``'s ``positions.max().item()`` is illegal
    on a capturing stream. Serving enters this scope from its graph runners; a
    raw ``torch.cuda.CUDAGraph`` in the benchmark leaves the flag clear, so the
    module would take its eager branch inside the capture and fail. Reuse the
    framework's scope rather than poking the module global: it also disables
    tensor disposal, which would otherwise record ``data_ptr() == 0`` frees
    into the graph.
    """
    from contextlib import nullcontext

    try:
        from sglang.srt.model_executor.runner import model_capture_mode
    except ImportError:
        return nullcontext()
    return model_capture_mode()


def _extend_batch(model_runner, reqs, prefix_len: int):
    """An extend batch over [prefix_len, extend_end) whose cached prefix is an earlier chunk.

    With the radix cache disabled serving matches nothing, so a prefix lives only in
    the request's own row from its previous chunk (mem_cache/allocation.py:363-371).
    The row is built as serving builds it: an extend over [0, prefix_len) allocates it
    (schedule_batch.py:2847-2919, allocation.py:374+), the scheduler's checkpoint
    advances prefix_len to its end (mem_cache/common.py:271-280), and the next round
    extends [prefix_len, extend_end). The KV pool's page size is whatever serving
    resolved: at page 1 both rounds take the token allocator, as serving does. The
    prefill ids reach the forward stream the way SGLang's offline builder hands them
    over (benchmark/one_batch.py:516-523), so the ForwardBatch counts its real tokens
    (model_executor/forward_batch_info.py:1364-1382).
    """
    from sglang.srt.mem_cache.common import checkpoint_kv_cache

    batch_size = len(reqs)
    full_length = reqs[0].extend_end
    tree_cache = _radix_disabled_tree_cache(model_runner)
    if prefix_len:
        for req in reqs:
            req.extend_end = prefix_len
        prefix_batch = _make_schedule_batch(model_runner, reqs, tree_cache)
        with temporarily_chunked_alloc_extend(model_runner, batch_size * prefix_len):
            prefix_batch.prepare_for_extend()
        for req in reqs:
            checkpoint_kv_cache(req, tree_cache)
            req.extend_end = full_length
    batch = _make_schedule_batch(model_runner, reqs, tree_cache)
    with temporarily_chunked_alloc_extend(model_runner, batch_size * (full_length - prefix_len)):
        batch.prepare_for_extend()
    batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
    batch.prefill_input_ids_cpu = None
    return batch


def _initialize_msa_history(model_runner, attention_module, forward_batch, history_length: int) -> None:
    """Write the cached history the benchmark layer reads, through serving's own KV writer.

    The allocator only assigns slots. The pools start as zeros
    (mem_cache/memory_pool.py:2580-2600 MHATokenToKVPool._create_buffers_normal) and
    nothing here runs the forwards that would fill them, so every history block
    would score the same and the decode radix top-k would select among ties. Serving
    stores post-norm/RoPE K/V and index K with set_fused_kv_index_buffer
    (layers/attention/minimax_sparse_backend.py:1483-1494, 1724-1735; the pool's writer
    mem_cache/memory_pool.py:6211-6280); seeded N(0, 1) activations go through the same
    writer outside the measurement. They stand for no particular checkpoint distribution.
    Same treatment as the DSA modules (collect_mla_module._initialize_dsa_history).
    """
    if history_length <= 0:
        return
    layer = attention_module.attn
    pool = model_runner.token_to_kv_pool
    locations = (
        model_runner.req_to_token_pool.req_to_token[forward_batch.req_pool_indices.long(), :history_length]
        .flatten()
        .to(torch.int64)
        .contiguous()
    )
    expected = int(forward_batch.batch_size) * history_length
    if locations.numel() != expected:
        raise ValueError(f"MSA history has {locations.numel()} locations; expected {expected}")
    num_kv_heads, head_dim = int(attention_module.num_kv_heads), int(attention_module.head_dim)
    idx_head_dim = int(attention_module.idx_head_dim)
    generator = torch.Generator(device=locations.device)
    generator.manual_seed(0)

    def activations(n: int, heads: int, dim: int):
        return torch.randn((n, heads, dim), generator=generator, device=locations.device, dtype=torch.bfloat16)

    for start in range(0, expected, _HISTORY_CHUNK_TOKENS):
        slots = locations[start : start + _HISTORY_CHUNK_TOKENS]
        n = slots.numel()
        pool.set_fused_kv_index_buffer(
            layer,
            slots,
            activations(n, num_kv_heads, head_dim),
            activations(n, num_kv_heads, head_dim),
            activations(n, 1, idx_head_dim),
            None if attention_module.disable_index_value else activations(n, 1, idx_head_dim),
            layer.k_scale_float,
            layer.v_scale_float,
            layer.idx_k_scale_float,
            layer.idx_v_scale_float,
        )


def _serving_decode_graph_batch(model_runner, batch_size: int, target_tp_size: int) -> int | None:
    """The captured batch size serving replays this decode batch on, or None when it runs eagerly.

    DecodeCudaGraphRunner replays a batch no larger than its max_bs, padded up to
    the next captured size (model_executor/runner/decode_cuda_graph_runner.py:845-905,
    1615-1631). The sizes come from generate_decode_cuda_graph_batch_sizes over the
    decode max_bs (arg_groups/cuda_graph_hook.py:500-531), which
    _handle_gpu_memory_settings derives from device memory and the server's TP size
    (arg_groups/memory_hook.py:89-183). This runner resolved them for TP 1; the row
    stands for one rank of a TP-``target_tp_size`` server, which below 160 GiB
    captures up to 512 at TP >= 4 instead of 256 (runtime_limits.serving_memory_defaults).
    """
    decode_cfg = resolved_arg(model_runner.server_args, "cuda_graph_config").decode
    if decode_cfg.backend == "disabled":
        return None
    from sglang.srt.arg_groups.cuda_graph_hook import generate_decode_cuda_graph_batch_sizes

    capture_bs = generate_decode_cuda_graph_batch_sizes(
        model_runner.server_args, serving_memory_defaults(target_tp_size)[1]
    )
    return next((int(bs) for bs in sorted(capture_bs) if int(bs) >= batch_size), None)


def _minimax_sparse_ops():
    from sglang.srt.layers.attention.minimax_sparse_ops import minimax_sparse

    return minimax_sparse


def _ran_msa(use_msa: bool) -> bool:
    """Whether the main sparse step ran fmha_sm100 since _run_*_point cleared the flag.

    minimax_sparse_prefill/decode fall back to the Triton kernel when MSA raises
    MSAUnavailableError and record it in _msa_fallback_warned
    (layers/attention/minimax_sparse_ops/minimax_sparse.py:28-40, 217-257, 446-447).
    """
    return use_msa and not _minimax_sparse_ops()._msa_fallback_warned


def _decode_topk_variant(sparse_backend, max_seqlen_k: int) -> str:
    """The decode top-k kernel the sparse indexer selects for this KV bound — the
    gate of flash_decode_with_topk_idx (kernels/ops/attention/minimax_sparse/decode/
    flash_with_topk_idx.py:25, 1040-1055): SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX,
    score blocks of min(max_seqlen, req_to_token width) <= _JIT_TOPK_MAX_NUM_BLOCKS
    (16384 on CUDA; 4096 at v0.5.21) and topk <= 32 select the JIT radix select,
    otherwise the 2-stage Triton top-k."""
    from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import _JIT_TOPK_MAX_NUM_BLOCKS
    from sglang.srt.environ import envs

    score_kv_len = min(int(max_seqlen_k), int(sparse_backend.req_to_token.shape[1]))
    score_blocks = _ceil_div(score_kv_len, sparse_backend.block_size_k)
    use_radix = (
        envs.SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX.get()
        and score_blocks <= _JIT_TOPK_MAX_NUM_BLOCKS
        and sparse_backend.topk_blocks <= 32
    )
    return "topk_radix" if use_radix else "topk_split"


def _prefill_kernel_source(model_runner) -> str:
    """use_msa selects fmha_sm100 for the main step (minimax_sparse_backend.py:196-234)."""
    main = "msa_fmha_sm100" if _ran_msa(model_runner.attn_backend.sparse.use_msa) else "triton_sparse"
    return f"sglang_minimax_prefill_{main}"


def _decode_kernel_source(model_runner, max_seqlen_k: int) -> str:
    """``max_seqlen_k`` is the bound the decode kernels were launched with (see _run_decode_point)."""
    sparse = model_runner.attn_backend.sparse
    if not hasattr(sparse, "_use_msa_decode"):
        # kernel_source records ground truth; a silent False default would
        # mislabel every decode row as triton_sparse if SGLang renames the
        # flag (minimax_sparse_backend.py:289-291 sets it in __init__).
        raise RuntimeError(
            "MiniMaxSparseAttnBackend has no _use_msa_decode attribute "
            "(renamed upstream?); refusing to guess the decode kernel identity"
        )
    main = "msa_fmha_sm100" if _ran_msa(sparse._use_msa_decode) else "triton_sparse"
    return f"sglang_minimax_decode_{main}_{_decode_topk_variant(sparse, max_seqlen_k)}"


def _qsa_full_attn_backend(model_runner):
    """The QSA backend inside Qwen4-Exp's hybrid attention wrapper.

    ``attn_backend_wrapper`` builds
    ``HybridLinearAttnBackend(full_attn_backend=QwenSparseAttnBackend,
    linear_attn_backend=GDNAttnBackend, cfg.full_attention_layer_ids)`` for
    every ``is_qwen_qsa`` model (attention_registry.py:458-466@v0.5.21), so the
    QSA backend is the wrapper's full-attention child.
    """
    full_attn_backend = getattr(model_runner.attn_backend, "full_attn_backend", None)
    if type(full_attn_backend).__name__ != "QwenSparseAttnBackend":
        raise RuntimeError(
            "QSA kernel_source resolution needs the hybrid wrapper's "
            f"QwenSparseAttnBackend, got {type(full_attn_backend).__name__}"
        )
    return full_attn_backend


def _qsa_backend_module():
    """The SGLang module that owns QSA route dispatch.

    The route predicates below live on the module, not the backend instance
    (``_resolve_trtllm_sparse_decode`` / ``_resolve_flash_attn_varlen_func``), so
    the label is resolved from the same code that picks the kernel. Importing it
    here also proves the backend class came from this module.
    """
    import sglang.srt.layers.attention.qwen_sparse_attn_backend as qsa_module

    return qsa_module


def _require_qsa_symbol(module, name: str):
    """Fail loud when an upstream rename would make the label a guess."""
    symbol = getattr(module, name, None)
    if symbol is None:
        raise RuntimeError(
            f"QSA module {module.__name__} has no {name!r} (renamed upstream?); "
            "refusing to guess the kernel_source label"
        )
    return symbol


def _qsa_prefill_kernel_source(model_runner, prefix_len: int) -> str:
    """Record the QSA prefill route serving executes for this point.

    ``QwenSparseAttnBackend.forward_extend`` has exactly two CUDA routes: the
    no-prefix Triton sparse-GQA kernel over the current chunk's K/V, and — when
    any request carries a prefix — the chunk-prefill Triton kernel over the
    packed full-context K/V (qwen_sparse_attn_backend.py:1318,1354@v0.5.21,
    selected by ``any(prefix_lens)``). Both symbols are checked to exist so a
    rename cannot silently mislabel rows.
    """
    module = _qsa_backend_module()
    _require_qsa_symbol(module, "sparse_gqa_fwd_interface_triton")
    _qsa_full_attn_backend(model_runner)
    if prefix_len > 0:
        _require_qsa_symbol(module, "sparse_gqa_fwd_interface_triton_ck")
        return "sglang_qwen4exp_qsa_prefill_triton_ck"
    return "sglang_qwen4exp_qsa_prefill_triton"


def _fa4_cute_varlen_target(varlen):
    """The FA4-cute implementation an SGLang varlen wrapper delegates to, if any.

    When ``flash_attn`` (FA2) is absent, ``_resolve_flash_attn_varlen_func``
    returns a module-local wrapper over
    ``flash_attn.cute.interface.flash_attn_varlen_func``
    (qwen_sparse_attn_backend.py:83-91@v0.5.21). That wrapper carries the same
    name as the FA2 entry point, so it cannot be told apart by name or module;
    identify it by the single closure it captures and only when that closure is
    genuinely the cute interface. Everything else stays unidentified and the
    caller keeps raising.
    """
    if getattr(varlen, "__name__", "") != "flash_attn_varlen_func":
        return None
    freevars = getattr(getattr(varlen, "__code__", None), "co_freevars", ())
    cells = getattr(varlen, "__closure__", None) or ()
    for freevar, cell in zip(freevars, cells, strict=False):
        if freevar != "cute_varlen_func":
            continue
        try:
            target = cell.cell_contents
        except ValueError:
            return None
        if getattr(target, "__module__", "").startswith("flash_attn"):
            return target
    return None


def _qsa_decode_kernel_source(model_runner) -> str:
    """Record the QSA decode route serving executes on this device.

    ``QwenSparseAttnBackend._forward_paged_attention`` resolves, in order
    (qwen_sparse_attn_backend.py:1539,1608,1620@v0.5.21):

    1. the FlashInfer trtllm-gen paged sparse decode — SM100/SM120 only
       (``_resolve_trtllm_sparse_decode``), so it is absent on H200;
    2. on HIP, the packed Triton sparse decode;
    3. otherwise the packed varlen fallback, which is FA2 ``flash_attn``, the
       SM121-only ``qwen38_qsa_sm121_varlen`` replacement, or — when FA2 is not
       installed — the FA4-cute wrapper built over
       ``flash_attn.cute.interface`` via ``_resolve_flash_attn_varlen_func``.

    Each branch is labelled separately because they are different kernels;
    a resolver that returns something this collector cannot name raises rather
    than mislabelling the row.
    """
    module = _qsa_backend_module()
    _qsa_full_attn_backend(model_runner)
    resolve_trtllm = _require_qsa_symbol(module, "_resolve_trtllm_sparse_decode")
    if resolve_trtllm() is not None:
        return "sglang_qwen4exp_qsa_decode_trtllm_sparse"

    from sglang.srt.utils import is_hip

    if is_hip():
        _require_qsa_symbol(module, "sparse_gqa_packed_decode_triton")
        return "sglang_qwen4exp_qsa_decode_triton_packed"

    resolve_varlen = _require_qsa_symbol(module, "_resolve_flash_attn_varlen_func")
    varlen = resolve_varlen()
    name = getattr(varlen, "__name__", "")
    origin = getattr(varlen, "__module__", "")
    if name == "qwen38_qsa_sm121_varlen":
        return "sglang_qwen4exp_qsa_decode_sm121_varlen"
    if origin.startswith("flash_attn"):
        return "sglang_qwen4exp_qsa_decode_fa2_varlen"
    if _fa4_cute_varlen_target(varlen) is not None:
        return "sglang_qwen4exp_qsa_decode_fa4_cute_varlen"
    raise RuntimeError(
        "QSA packed decode resolved "
        f"{origin}.{name}, which has no registered kernel_source label "
        "(neither FA2, the FA4-cute wrapper, nor the SM121 replacement); "
        "add a label before collecting"
    )


def _log_msa_row(
    *,
    perf_filename: str,
    model_path: str,
    architecture: str,
    compute_dtype: str,
    kv_cache_dtype: str,
    gemm_type: str,
    num_heads: int,
    batch_size: int,
    isl: int,
    step: int,
    target_tp_size: int,
    latency_ms: float,
    op_name: str,
    kernel_source: str,
    device_name: str,
    power_stats,
):
    # ``log_perf`` raises PerfLogError when the log file itself is unusable —
    # e.g. a resumed run finds the perf txt and its parked ``.retained`` twin
    # both present, or the inode is unowned. That cannot be fixed by the next
    # point, so convert it to the fatal write error here: the sweep aborts on the
    # first point with the real cause in the traceback instead of counting one
    # identical failure per remaining shape and then discarding the whole case
    # ("persisted no rows"). C3.3 heal 2026-11-01 lost 15 minutes of H200 time
    # per case to exactly this before it was diagnosed.
    try:
        persisted = log_perf(
            item_list=[
                {
                    "model": model_path,
                    "architecture": architecture,
                    "mla_dtype": compute_dtype,
                    "kv_cache_dtype": kv_cache_dtype,
                    "gemm_type": gemm_type,
                    "num_heads": num_heads,
                    "batch_size": batch_size,
                    "isl": isl,
                    "tp_size": target_tp_size,
                    "step": step,
                    "latency": f"{latency_ms:.4f}",
                }
            ],
            framework="SGLang",
            version=get_version("sglang"),
            device_name=device_name,
            op_name=op_name,
            kernel_source=kernel_source,
            perf_filename=perf_filename,
            power_stats=power_stats,
        )
    except PerfLogError as error:
        raise PerfLogWriteError(f"failed to persist MSA row to {perf_filename}: {error}") from error
    if not persisted:
        raise PerfLogWriteError(f"failed to persist MSA row to {perf_filename}")


def _module_positions(attention_type: str, forward_batch, fallback):
    """Positions to feed the timed module.

    ``qsa`` layers are MRoPE: ``Qwen4ExpAttentionDecoderLayer.self_attention``
    routes positions into the fused QK-norm+RoPE kernel with
    ``mrope_axis_map`` whenever ``positions.dim() == 2``, and serving feeds the
    ``[3, tokens]`` MRoPE tensor the multimodal model wrapper swaps in
    (``forward_batch.mrope_positions``, computed by ``ForwardBatch.init_new``
    for extend and decode text-only batches). A 1-D fallback would silently
    change the rotary path, so it raises instead.
    """
    if attention_type != "qsa":
        return fallback
    mrope = getattr(forward_batch, "mrope_positions", None)
    if mrope is None:
        raise RuntimeError(
            "Qwen4-Exp QSA module forward needs MRoPE positions ([3, tokens]); ForwardBatch.mrope_positions is unset"
        )
    return mrope


def _run_prefill_point(
    model_runner,
    attention_module,
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    *,
    num_warmup: int,
    num_iterations: int,
    device: str,
    hidden_size: int,
    attention_type: str = "msa",
):
    """Benchmark one context (prefill) point; returns (latency_ms, power_stats).

    The whole module is timed eagerly. Deviation (every CUDA SM): v0.5.23 serves
    M3 prefill under the breakable prefill CUDA graph (default_prefill_backend,
    model_executor/cuda_graph_config.py:119-123; M3 is on the BCG allow-list,
    configs/model_config.py:2501-2516, and no rule in
    arg_groups/cuda_graph_hook.py:170-245 turns it off). The sparse attention
    runs eagerly between the graph segments there too
    (RadixAttention._eager_attention, layers/radix_attention.py:133-185, 187-250),
    on the same backend code and metadata, so its kernels match. What differs:
    serving replays the fused qkv+index GEMM, the grouped QK norm/RoPE and o_proj
    from captured segments at the token count padded to the capture bucket, and
    copies the attention output into the segment's buffer; this collector
    launches them eagerly at the real token count. Launch overhead reads slower
    here for short prefills, bucket padding makes serving's GEMMs larger than
    timed here; neither is measured. The DSA prefill tables carry the same
    deviation (collect_mla_module.load_model_runner). Retire by timing prefill
    under the breakable runner once the measurement method for prefill graphs is
    decided.
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    point_clock = PhaseClock()
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    batch = _extend_batch(model_runner, _make_reqs(batch_size, prefix_len + seq_len), prefix_len)
    point_clock.mark("batch")
    # return_hidden_states_before_norm=False mirrors the serving worker
    # (managers/tp_worker.py:274).
    forward_batch = ForwardBatch.init_new(batch, model_runner, return_hidden_states_before_norm=False)
    # Serving metadata init for the eager forward (base
    # AttentionBackend.init_forward_metadata → out_graph + in_graph;
    # MiniMaxHybrid fans out to sparse+dense — minimax_sparse_backend.py:1878-1881).
    with _forward_ctx(model_runner, forward_batch):
        model_runner.attn_backend.init_forward_metadata(forward_batch)
    point_clock.mark("metadata")
    # MSA-only: the helper writes M3's fused KV+index history through
    # MiniMaxSparseAttnBackend's pool writer and reads `.attn`/`.num_kv_heads`/
    # `.idx_head_dim`, which a QSA layer does not have. A prefixed QSA point
    # therefore reads its prefix from recycled pages (see the module docstring's
    # measurement deviation); that is the condition the accepted C3.3 rows were
    # measured under.
    if attention_type != "qsa":
        _initialize_msa_history(model_runner, attention_module, forward_batch, prefix_len)
    point_clock.mark("history")

    hidden_states = torch.randn(
        batch_size * seq_len,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    positions = (
        torch.arange(prefix_len, prefix_len + seq_len, device="cuda")
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
        .flatten()
    )
    positions = _module_positions(attention_type, forward_batch, positions)
    point_clock.mark("inputs")

    def kernel_func():
        with _forward_ctx(model_runner, forward_batch):
            return attention_module(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

    _minimax_sparse_ops()._msa_fallback_warned = False
    try:
        with benchmark_with_power(
            device=torch.device(device),
            kernel_func=kernel_func,
            num_warmups=num_warmup,
            num_runs=num_iterations,
            repeat_n=1,
            allow_graph_fail=False,
            use_cuda_graph=False,
        ) as results:
            pass
        point_clock.mark("bench")
        point_clock.absorb(results.get("phase_seconds"), prefix="bench_")
        return results["latency_ms"], results["power_stats"]
    finally:
        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()
        torch.cuda.empty_cache()
        point_clock.mark("cleanup")
        point_clock.add_to(_POINT_PHASE_TOTALS)
        print(f"  [MSA point timing] context b={batch_size}, s={seq_len}, prefix={prefix_len}: {point_clock.render()}")


def _run_decode_point(
    model_runner,
    attention_module,
    batch_size: int,
    kv_len: int,
    target_tp_size: int,
    *,
    num_warmup: int,
    num_iterations: int,
    device: str,
    hidden_size: int,
    attention_type: str = "msa",
):
    """Benchmark one generation (decode) point; returns (latency_ms, power_stats, kernel_source).

    Decode is timed under a captured CUDA graph exactly where serving replays one
    (_serving_decode_graph_batch). Serving's DecodeCudaGraphRunner allocates the
    backend's graph state (init_cuda_graph_state,
    model_executor/runner/decode_cuda_graph_runner.py:429), builds metadata at
    capture with init_forward_metadata_out_graph(in_capture=True) (:1400) and
    refills it before every replay with init_forward_metadata_out_graph (:1812).
    For the sparse backend the benchmark layer routes to
    (minimax_sparse_backend.py:1985-1988), that sequence reduces to the capture
    call: its init_cuda_graph_state is a no-op (:614-615); in_capture fixes the
    host-side _max_seqlen_k at max_context_len (:452-461), which sizes the score
    tensor and picks the top-k kernel baked into the graph; the refill only resets
    that host value to the live length (:463), which the captured kernels never
    read again, so replaying it before this capture would bake the live length in
    instead. Its decode kernels read req_to_token at full width in graph and eager
    mode alike (:1811-1852), so no page-table width differs from serving (unlike
    DSA's). Serving's MSA decode plan is built only when MSA owns decode, which a
    graph-enabled server never does (:289-291, 305-307, 469-470). The dense half
    of the hybrid backend is never read by the benchmark layer.

    QSA route: the same parity is taken through the QSA backend's own graph
    state (``init_cuda_graph_state`` then ``init_forward_metadata_out_graph
    (..., in_capture=True)``, qwen_sparse_attn_backend.py:762,731-749
    @v0.5.21), which sizes the static FA2/compressed-page buffers serving's
    decode graph reads. Only the full-attention child is initialized: the
    timed layer never executes the GDN side, whose graph state belongs to
    serving's graph runner (same rationale as the MSA collector).
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    point_clock = PhaseClock()
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    graph_batch = _serving_decode_graph_batch(model_runner, batch_size, target_tp_size)
    if graph_batch is not None and graph_batch != batch_size:
        # The plan's decode batches are capture sizes; an ad-hoc --quick batch may not be.
        raise RuntimeError(
            f"decode batch {batch_size} replays on serving's padded graph batch {graph_batch}; "
            "the collector times captured batch sizes only"
        )
    use_graph = graph_batch is not None

    batch = _extend_batch(model_runner, _make_reqs(batch_size, kv_len), 0)
    for req in batch.reqs:
        req.output_ids.append(0)
    # The sampled token of the previous step (benchmark/one_batch.py:536-538);
    # prepare_for_decode allocates its slot and advances seq_lens
    # (managers/schedule_batch.py:3780-3789).
    batch.input_ids = torch.zeros(batch_size, dtype=torch.int64, device=batch.device)
    batch.prepare_for_decode()
    forward_batch = ForwardBatch.init_new(batch, model_runner, return_hidden_states_before_norm=False)
    point_clock.mark("batch")

    # MiniMax's sparse child owns both the capture-time metadata and
    # `_max_seqlen_k`. The QSA route's hybrid wrapper exposes no `sparse`
    # attribute at all (its full-attention child is `full_attn_backend`), so
    # reading it here would raise before either QSA timing path starts.
    sparse = None if attention_type == "qsa" else model_runner.attn_backend.sparse
    with _forward_ctx(model_runner, forward_batch):
        model_runner.attn_backend.init_forward_metadata(forward_batch)
        if use_graph:
            if attention_type == "qsa":
                qsa_backend = _qsa_full_attn_backend(model_runner)
                qsa_backend.init_cuda_graph_state(max_bs=batch_size, max_num_tokens=batch_size)
                qsa_backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)
            else:
                # Capture parity for the SPARSE backend only: the benchmarked
                # layer routes exclusively to MiniMaxSparseAttnBackend
                # (hybrid dispatch by layer id,
                # minimax_sparse_backend.py:585-597), so only its capture-time
                # metadata is consumed. The dense fa3 side of the hybrid
                # wrapper needs its full graph-runner state
                # (init_cuda_graph_state buffers) for an in_capture call,
                # which the serving decode graph runner owns — the module
                # bench never executes a dense layer, so its eager metadata is
                # kept.
                model_runner.attn_backend.sparse.init_forward_metadata_out_graph(forward_batch, in_capture=True)
    # The KV bound every MSA decode kernel of this point is launched with
    # (minimax_sparse_backend.py:1823 passes it to minimax_sparse_decode). The
    # QSA route labels its own kernel source from the backend it routed to.
    max_seqlen_k = int(sparse._max_seqlen_k) if sparse is not None else 0
    point_clock.mark("metadata")
    # MSA-only, same reason as the context path: a QSA decode point starts on
    # the pages the allocator hands it, not on a seeded history.
    if attention_type != "qsa":
        _initialize_msa_history(model_runner, attention_module, forward_batch, kv_len)
    point_clock.mark("history")

    hidden_states = torch.randn(
        batch_size,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    # The decoded token's position (seq_lens - 1 after prepare_for_decode).
    positions = torch.full((batch_size,), kv_len, device="cuda", dtype=torch.int64)
    positions = _module_positions(attention_type, forward_batch, positions)
    point_clock.mark("inputs")

    def kernel_func():
        with _forward_ctx(model_runner, forward_batch):
            return attention_module(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

    _minimax_sparse_ops()._msa_fallback_warned = False
    try:
        with benchmark_with_power(
            device=torch.device(device),
            kernel_func=kernel_func,
            num_warmups=num_warmup,
            num_runs=num_iterations,
            repeat_n=1,
            allow_graph_fail=False,
            use_cuda_graph=use_graph,
            capture_context=_sglang_capture_context if attention_type == "qsa" else None,
        ) as results:
            pass
        point_clock.mark("bench")
        point_clock.absorb(results.get("phase_seconds"), prefix="bench_")
        if attention_type == "qsa":
            kernel_source = _qsa_decode_kernel_source(model_runner)
        else:
            kernel_source = _decode_kernel_source(model_runner, max_seqlen_k)
        return results["latency_ms"], results["power_stats"], kernel_source
    finally:
        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()
        torch.cuda.empty_cache()
        point_clock.mark("cleanup")
        point_clock.add_to(_POINT_PHASE_TOTALS)
        print(f"  [MSA point timing] generation b={batch_size}, kv={kv_len}: {point_clock.render()}")


# ═══════════════════════════════════════════════════════════════════════
# Subprocess orchestration
# ═══════════════════════════════════════════════════════════════════════


def run_msa_module(
    num_heads: int,
    model_path: str,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    is_prefill: bool,
    gpu_id: int,
    output_path: str | None = None,
    batch_size_filter: int | None = None,
    target_tp_size: int = 1,
    quick_shape: tuple | None = None,
    inner_manifest=(),
    attention_type: str | None = None,
):
    """Run one sparse-attention module benchmark sweep — inside a subprocess.

    ``attention_type`` selects the module route: ``msa`` (MiniMax-M3 sparse
    layer) or ``qsa`` (Qwen4-Exp compressed QSA layer). When omitted it is
    resolved from the declared model row (``_attention_type_for_model``), so the
    registry's positional case tuple keeps its existing layout.

    ``quick_shape``: optional single (batch, seq, prefix) point for CLI
    smoke runs; replaces the sweep but keeps every construction path.
    """
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    if attention_type is None:
        attention_type = _attention_type_for_model(model_path)
    if attention_type not in _MSA_MODULE_ATTENTION_TYPES:
        raise ValueError(f"unknown attention_type {attention_type!r}")

    if compute_dtype != "bfloat16":
        # SGLang derives the attention GEMM precision itself: bf16, or fp8 q with
        # an fp8_e4m3 KV cache on trtllm_mha + SM100 (server_args.py:592-610).
        raise ValueError(
            f"sparse-attention modules compute in bf16 at this pin "
            f"(M3 kernels assert q.dtype in bf16/fp16, decode/flash_with_topk_idx.py:778-782; "
            f"QSA kernels are bf16-only, qsa/kernel.py@v0.5.21); got compute={compute_dtype}"
        )
    if gemm_type != "bfloat16":
        if attention_type == "qsa":
            raise ValueError(
                f"Qwen3.8-Flash-Next QSA projections are bf16 in both checkpoints "
                f"(the FP8 checkpoint quantizes routed experts only); got gemm={gemm_type!r} "
                f"(mla_module_plan declares the QSA combos)"
            )
        raise ValueError(
            f"MiniMax-M3 is a BF16 artifact; SGLang's quantized-M3 flows are MXFP8 and the "
            f"M3.1 NVFP4 stack (arg_groups/model_overrides/minimax_m3.py:18-34), "
            f"not {gemm_type!r} (skips.py leaves these cases out)"
        )
    if kv_cache_dtype not in SGLANG_KV_DTYPE:
        raise ValueError(f"unsupported kv_cache_dtype {kv_cache_dtype!r}")

    phase = "context" if is_prefill else "generation"

    if quick_shape is not None:
        b, s, prefix = quick_shape
        shapes = [(b, s, prefix)] if is_prefill else [(b, s, 0)]
    else:
        # The worker never re-derives the grid: it runs the points the plan's row lists.
        shapes = [tuple(shape) for shape in inner_manifest]

    if not shapes:
        raise RuntimeError(
            f"MSA module {phase} has no points: the plan queues no empty row; model={model_path}, "
            f"heads={num_heads}, batch_filter={batch_size_filter} (ad-hoc runs: --quick)"
        )
    if attention_type == "qsa":
        # The plan aligns its prefixes; an ad-hoc shape can still be off a
        # compressed group, and the device assert it would hit is unrecoverable.
        _require_aligned_qsa_prefixes(shapes, model_path=model_path)

    # A point this device cannot run is refused with its reason before anything is
    # allocated for it, and the ModelRunner is sized for the points that remain.
    refused: dict[tuple, str] = {}
    if is_prefill:
        # SGLang derives its prefill chunk from device memory alone
        # (_handle_gpu_memory_settings, arg_groups/memory_hook.py:89-176); serving
        # chunks a longer prefill.
        chunk = int(serving_memory_defaults(tp_size=1)[0])
        refuse(shapes, refused, lambda shape: over_chunk(shape, chunk))
        # The QSA extend path allocates a per-shape gather workspace on top of
        # the resident pool, so bounding the pool alone is not enough. Refuse
        # the points whose combined footprint the device cannot hold before the
        # runner is sized for the ones that remain.
        if attention_type == "qsa":
            page_size, bytes_per_token, workspace_bytes = _kv_cost_estimate(
                "qsa", num_heads, kv_cache_dtype, model_path
            )
            total_memory = torch.cuda.get_device_properties(device).total_memory
            refuse(
                shapes,
                refused,
                lambda shape: _qsa_footprint_refusal(
                    shape,
                    page_size=page_size,
                    bytes_per_token=bytes_per_token,
                    workspace_bytes=workspace_bytes,
                    total_memory=total_memory,
                ),
            )
    runnable = [shape for shape in shapes if shape not in refused]
    if not runnable:
        raise RuntimeError(refusal_summary(refused, len(shapes)))

    # 128 is the M3 serving page on SM90/SM100 (arg_groups/model_overrides/
    # minimax_m3.py:163-168, 181-187), 1 elsewhere; page-128 rounding is an upper
    # bound for any smaller resolved page, including QSA's pinned 64
    # (arg_groups/model_overrides/qwen4_exp.py@v0.5.21). The pool SGLang builds
    # is checked against the real page below.
    page_size_guess = 128
    max_total_tokens = max(
        required_kv_alloc_tokens(b, s, p, page_size_guess, is_prefill=is_prefill) for (b, s, p) in runnable
    )
    max_total_tokens += max(1024, max_total_tokens // 20)
    max_running_requests = max(b for (b, _s, _p) in runnable)

    route_label = "QSA" if attention_type == "qsa" else "MSA"
    print(
        f"\n{'=' * 60}\n{route_label} Module {phase}: model={model_path}, heads={num_heads}, "
        f"target_tp={target_tp_size}, kv={kv_cache_dtype}, gemm={gemm_type}, "
        f"shapes={len(shapes)}, GPU={gpu_id}\n{'=' * 60}"
    )

    cleanup_distributed()
    torch.cuda.empty_cache()

    ok = 0
    failures: list[str] = []
    try:
        model_runner = load_model_runner(
            model_path=model_path,
            num_heads=num_heads,
            kv_cache_dtype=kv_cache_dtype,
            target_tp_size=target_tp_size,
            max_total_tokens=max_total_tokens,
            chunked_prefill_size=None,
            max_running_requests=max_running_requests,
            device=device,
            attention_type=attention_type,
        )

        # Refuse what the built runner cannot run: a prefill beyond the chunk it
        # resolved, or a point beyond its KV pool.
        if is_prefill:
            runtime_chunk = runtime_chunk_size(model_runner)
            print(f"[MSA] SGLang runtime chunked_prefill_size={runtime_chunk}")
            refuse(shapes, refused, lambda shape: over_chunk(shape, runtime_chunk))
        capacity = kv_pool_capacity_tokens(model_runner)
        if capacity is not None:
            page_size = kv_pool_page_size(model_runner)
            refuse(shapes, refused, lambda shape: over_kv_pool(shape, page_size, capacity, is_prefill))
        runnable = [shape for shape in shapes if shape not in refused]
        if not runnable:
            raise RuntimeError(refusal_summary(refused, len(shapes)))

        if attention_type == "qsa":
            attention_module = _qsa_module(model_runner)
            architecture = QSA_ARCHITECTURE
        else:
            attention_module = model_runner.model.model.layers[0].self_attn
            architecture = MSA_ARCHITECTURE
        hidden_size = _module_hidden_size(model_runner, attention_type)
        device_name = torch.cuda.get_device_name(device)
        perf_filename = _resolve_perf_path(output_path, f"msa_{phase}_module_perf.txt")
        op_name = f"msa_{phase}_module"

        for i, (b, s, p) in enumerate(runnable):
            label = f"b={b}, s={s}, prefix={p}" if is_prefill else f"b={b}, kv={s}"
            print(f"[{i + 1}/{len(runnable)}] {phase} {label}, heads={num_heads}")
            try:
                if is_prefill:
                    latency, power_stats = _run_prefill_point(
                        model_runner,
                        attention_module,
                        b,
                        s,
                        p,
                        num_warmup=8,
                        num_iterations=10,
                        device=device,
                        hidden_size=hidden_size,
                        attention_type=attention_type,
                    )
                    if attention_type == "qsa":
                        kernel_source = _qsa_prefill_kernel_source(model_runner, p)
                    else:
                        kernel_source = _prefill_kernel_source(model_runner)
                    isl, step = s, p
                else:
                    latency, power_stats, kernel_source = _run_decode_point(
                        model_runner,
                        attention_module,
                        b,
                        s,
                        target_tp_size,
                        num_warmup=8,
                        num_iterations=10,
                        device=device,
                        hidden_size=hidden_size,
                        attention_type=attention_type,
                    )
                    if attention_type == "qsa":
                        kernel_source = _qsa_decode_kernel_source(model_runner)
                    isl, step = 1, s
                _log_msa_row(
                    perf_filename=perf_filename,
                    model_path=model_path,
                    architecture=architecture,
                    compute_dtype=compute_dtype,
                    kv_cache_dtype=kv_cache_dtype,
                    gemm_type=gemm_type,
                    num_heads=num_heads,
                    batch_size=b,
                    isl=isl,
                    step=step,
                    target_tp_size=target_tp_size,
                    latency_ms=latency,
                    op_name=op_name,
                    kernel_source=kernel_source,
                    device_name=device_name,
                    power_stats=power_stats,
                )
                print(f"  {label}: {latency:.4f} ms [{kernel_source}]")
                ok += 1
            except PerfLogWriteError:
                raise
            except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
                print(f"  OOM at {label}: {e}")
                failures.append(f"{label}: OOM: {e}")
                torch.cuda.empty_cache()
            except Exception as e:
                traceback.print_exc()
                error_str = str(e).lower()
                if "cuda" in error_str and "illegal" in error_str:
                    # CUDA context is poisoned; stop to preserve prior rows.
                    failures.append(f"{label}: CUDA illegal access: {e}")
                    raise RuntimeError(
                        f"CUDA illegal access at {label}; aborting subprocess. "
                        f"progress ok={ok} failed={len(failures)} total={len(runnable)}"
                    ) from e
                failures.append(f"{label}: {type(e).__name__}: {e}")

        summary = f"ok={ok} error={len(failures)} refused={len(refused)} total={len(shapes)}"
        print(f"[MSA] {phase} {summary}")
        print(f"[MSA timing] {phase}: {render_phase_totals(_POINT_PHASE_TOTALS)}")
        if ok == 0:
            raise RuntimeError(f"MSA module {phase} persisted no rows; {summary}")
        if failures:
            details = "\n- ".join(failures)
            raise RuntimeError(f"MSA module {phase} failed strict completeness: {summary}; failures:\n- {details}")
        if refused:
            raise RuntimeError(refusal_summary(refused, len(shapes)))
    finally:
        cleanup_distributed()
        torch.cuda.empty_cache()
        gc.collect()


def _resolve_perf_path(output_path: str | None, filename: str) -> str:
    if output_path is not None:
        return os.path.join(output_path, filename)
    collector_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(collector_dir, filename)


def _run_msa_subprocess(
    num_heads: int,
    model_path: str,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    is_prefill: bool,
    gpu_id: int,
    output_path: str | None,
    batch_size_filter: int | None,
    target_tp_size: int,
    inner_manifest=(),
    attention_type: str | None = None,
):
    """Run one MSA/QSA sweep in a subprocess with CUDA_VISIBLE_DEVICES
    isolation (same pattern as collect_mla_module._run_mla_subprocess:
    SGLang's ModelRunner/NCCL state cannot be re-initialized in-process)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    phase = "context" if is_prefill else "generation"
    output_repr = f'"{output_path}"' if output_path else "None"
    batch_repr = "None" if batch_size_filter is None else str(batch_size_filter)
    attention_repr = "None" if attention_type is None else f'"{attention_type}"'
    code = (
        f'import sys; sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")\n'
        f"from collect_msa_module import run_msa_module\n"
        f'run_msa_module({num_heads}, "{model_path}", "{kv_cache_dtype}", '
        f'"{compute_dtype}", "{gemm_type}", {is_prefill}, 0, {output_repr}, '
        f"{batch_repr}, {target_tp_size}, None, {inner_manifest!r}, {attention_repr})\n"
    )

    _timeout_env = os.environ.get("AIC_MLA_MODULE_SUBPROCESS_TIMEOUT_SEC")
    subprocess_timeout = int(_timeout_env) if _timeout_env else None
    try:
        # beats the stall watchdog while the child prints progress; the child dies with the worker
        returncode, stdout = run_measurement_subprocess(
            [sys.executable, "-c", code],
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=subprocess_timeout,
        )
        if stdout:
            print(stdout.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired as exc:
        if exc.output:
            print(exc.output.decode("utf-8", errors="replace"))
        raise RuntimeError(
            f"MSA module {phase} subprocess timed out after {subprocess_timeout}s "
            f"(heads={num_heads}, model={model_path}, kv={kv_cache_dtype})"
        ) from exc

    if returncode != 0:
        tail = ""
        lines = []
        if stdout:
            lines = stdout.decode("utf-8", errors="replace").strip().splitlines()
            tail = "\n".join(lines[-30:])
        # Keep the kernel-limit classification of points the device refused.
        error = kernel_limit_error(lines)
        if error is not None:
            raise RuntimeError(error)
        raise RuntimeError(
            f"MSA module {phase} subprocess failed (exit code {returncode})\n"
            f"--- subprocess output (last 30 lines) ---\n{tail}"
        )


def run_msa_module_worker(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    target_tp_size: int = 1,
    inner_manifest=(),
    *,
    perf_filename: str,
    device: str = "cuda:0",
    attention_type: str | None = None,
):
    """Worker-compatible positional wrapper used by collector/collect.py.

    Positional prefix matches the trtllm MSA worker; seq_len is a 0
    placeholder. ``inner_manifest`` is the [batch, seq_len, prefix_len] points
    the plan's row lists (op_plans._sglang_msa_rows); the subprocess runs
    them, or refuses one with its reason, and never re-derives the grid.
    ``attention_type`` is resolved from the model row inside the subprocess
    when omitted.
    """
    device_str = str(device) if not isinstance(device, str) else device
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0
    is_prefill = "context" in perf_filename
    batch_size_filter = batch_size if is_prefill and batch_size > 0 else None
    route_label = "QSA" if attention_type == "qsa" else "MSA"

    print(f"\n{'=' * 60}")
    print(
        f"{route_label} Module {'Context' if is_prefill else 'Generation'}: model={model_path}, "
        f"heads={num_heads}, target_tp={target_tp_size}, kv={kv_cache_dtype}, "
        f"compute={compute_dtype}, gemm={gemm_type}, "
        f"batch_filter={batch_size_filter or 'all'}, GPU={gpu_id}"
    )
    print(f"{'=' * 60}")

    output_path = os.path.dirname(perf_filename) or os.getcwd()

    _run_msa_subprocess(
        num_heads=num_heads,
        model_path=model_path,
        kv_cache_dtype=kv_cache_dtype,
        compute_dtype=compute_dtype,
        gemm_type=gemm_type,
        is_prefill=is_prefill,
        gpu_id=gpu_id,
        output_path=output_path,
        batch_size_filter=batch_size_filter,
        target_tp_size=target_tp_size,
        inner_manifest=tuple(tuple(shape) for shape in inner_manifest),
        attention_type=attention_type,
    )


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="MiniMax-M3 MSA / Qwen4-Exp QSA module-level collector for SGLang")
    parser.add_argument("--mode", choices=["context", "generation"], required=True)
    parser.add_argument("--model", type=str, default="MiniMaxAI/MiniMax-M3")
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument(
        "--target-tp-size",
        type=int,
        default=None,
        help="defaults to the model row's native // num-heads",
    )
    parser.add_argument(
        "--attention-type",
        choices=["msa", "qsa"],
        default=None,
        help="defaults to the model row's declared attention type",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--prefix-len", type=int, default=0)
    parser.add_argument("--kv-cache-dtype", choices=["bfloat16", "fp8"], default="bfloat16")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quick", action="store_true", help="run a single (batch, seq[, prefix]) point in-process")
    args = parser.parse_args()

    gpu_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
    model_spec = _resolve_model_spec(args.model)
    attention_type = args.attention_type or model_spec.attention_type
    if attention_type != model_spec.attention_type:
        raise SystemExit(
            f"--attention-type {attention_type} contradicts the declared mla_module "
            f"row for {model_spec.model_path} ({model_spec.attention_type})"
        )
    # Native query heads of the declared row: 64 for MiniMax-M3, 24 for
    # Qwen3.8-Flash-Next. The default shard is the row's native count, so
    # --num-heads defaults per model instead of assuming M3's 64.
    native_heads = int(model_spec.native_num_heads)
    num_heads = args.num_heads or native_heads
    if native_heads % num_heads:
        raise SystemExit(f"--num-heads {num_heads} is not a TP shard of the row's {native_heads} native heads")
    implied_tp = native_heads // num_heads
    if args.target_tp_size is not None and args.target_tp_size != implied_tp:
        raise SystemExit(
            f"--target-tp-size {args.target_tp_size} contradicts --num-heads {num_heads}: "
            f"{native_heads} native heads shard to {num_heads} at TP {implied_tp}"
        )
    target_tp = args.target_tp_size or max(1, implied_tp)
    is_prefill = args.mode == "context"

    if args.quick:
        run_msa_module(
            num_heads=num_heads,
            model_path=args.model,
            kv_cache_dtype=args.kv_cache_dtype,
            compute_dtype="bfloat16",
            gemm_type="bfloat16",
            is_prefill=is_prefill,
            gpu_id=gpu_id,
            output_path=args.output_path,
            batch_size_filter=None,
            target_tp_size=target_tp,
            quick_shape=(
                args.batch_size or (2 if is_prefill else 4),
                args.seq_len or (512 if is_prefill else 2048),
                args.prefix_len,
            ),
            attention_type=attention_type,
        )
        return

    # Full sweeps run the plan's rows, so a CLI reproduction runs the exact planned points.
    # The filter compares the row's local head count against the RESOLVED
    # ``num_heads``: ``args.num_heads`` is None when the flag is omitted, and
    # comparing against it selected no M3 or QSA row at all. Each row carries
    # its own TP, which is the plan's authority on the shard it was measured at.
    for _0, batch, heads, kv_dtype, compute, gemm, model, row_tp, points in plan(
        f"msa_{args.mode}_module", "sglang", get_sm_version()
    ).rows:
        # Compare against the row the CLI resolved, not the raw spelling: an
        # artifact alias (``...-FP8``) resolves to its canonical row, and the
        # plan only ever names the canonical path.
        if model != model_spec.model_path or heads != num_heads or kv_dtype != args.kv_cache_dtype:
            continue
        if args.batch_size is not None and batch != args.batch_size:
            continue
        run_msa_module(
            num_heads=heads,
            model_path=model,
            kv_cache_dtype=kv_dtype,
            compute_dtype=compute,
            gemm_type=gemm,
            is_prefill=is_prefill,
            gpu_id=gpu_id,
            output_path=args.output_path,
            batch_size_filter=batch or None,
            target_tp_size=row_tp,
            inner_manifest=points,
        )


if __name__ == "__main__":
    main()
