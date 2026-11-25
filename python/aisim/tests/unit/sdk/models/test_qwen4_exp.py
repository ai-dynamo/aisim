# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8-Flash-Next (Qwen4-Exp) modeling contracts.

The checkpoint mixes two attention families and adds three branches Qwen3.5 has
no equivalent for, so these tests pin the op graph against tensor shapes read
from the published safetensors headers rather than against the Python builder:

* GDN is 16 key / 48 value heads at 128 dims (``in_proj_qkv`` 10240).
* QSA is 24 q / 2 kv heads at 256 dims with a doubled ``q_proj`` (12288).
* The indexer is 4 query / 1 KV head at 128 dims (``index_qk_proj`` 640).
* The gated residual mist is 4 branches x 2560 wide at low-rank 320.
* The n-gram table is 20e6 x 2560 = 51.2e9 parameters (102.4 GB BF16).
"""

import dataclasses
import json
import types

import pytest

from aisim.sdk import common, config, models
from aisim.sdk.models.qwen4_exp import Qwen4ExpModel
from aisim.sdk.operations.embedding import Embedding
from aisim.sdk.operations.msa import ContextMSAModule, GenerationMSAModule

pytestmark = pytest.mark.unit

BF16 = "Qwen/Qwen3.8-Flash-Next"
FP8 = "Qwen/Qwen3.8-Flash-Next-FP8"

# Byte sizes of the two published repositories (sum of every tensor).
REPO_BYTES = {BF16: 360.0e9, FP8: 185.5e9}


def _model_config(tp_size=8, *, nextn=1):
    return config.ModelConfig(tp_size=tp_size, moe_tp_size=1, moe_ep_size=tp_size, nextn=nextn)


def _build(model_path=BF16, **kwargs):
    return models.get_model(model_path, _model_config(**kwargs), "vllm")


def _by_name(phase_ops):
    """Flatten OverlapOp groups into one name->op map (later duplicates win)."""
    out = {}
    stack = list(phase_ops)
    while stack:
        op = stack.pop()
        if hasattr(op, "_group_a"):
            stack.extend(op._group_a)
            stack.extend(op._group_b)
        else:
            out[op._name] = op
    return out


def _with_prefix(ops_by_name, prefix):
    return {name: op for name, op in ops_by_name.items() if name.startswith(prefix)}


def _walk(phase_ops):
    """Every leaf op, descending into the overlap composites."""
    for op in phase_ops:
        group_a = getattr(op, "_group_a", None)
        if group_a is not None:
            yield from _walk(group_a)
            yield from _walk(getattr(op, "_group_b", ()))
        else:
            yield op


def _replicated_weights_bytes(model):
    """Per-rank bytes of weights that are *not* TP-sharded.

    Both runtimes keep the gated-residual projections and the PLE merged
    ``kv_proj`` whole on every rank (``nn.Linear`` and
    ``ReplicatedLinear``/``disable_tp=True``), so the per-rank inventory already
    contains the full copy of each. Reconstructing the checkpoint total by
    multiplying the inventory by TP would count these TP times.
    """
    total = 0.0
    for phase_ops, only in ((model.context_ops, None), (model.generation_ops, "draft_")):
        for op in _walk(phase_ops):
            name = getattr(op, "_name", "")
            if only is None and name.startswith("draft_"):
                continue  # draft weights only appear in the generation inventory
            if only is not None and not name.startswith(only):
                continue
            if "_hc_" in name or name.endswith(("_key_proj_gemm", "_value_proj_gemm")):
                total += op.get_weights()
    return total


@pytest.mark.parametrize("model_path", [BF16, FP8])
def test_builds_for_both_published_checkpoints(model_path):
    model = _build(model_path)

    assert type(model).__name__ == "Qwen4ExpModel"
    assert model.model_family == "QWEN4EXP"
    assert model.context_ops and model.generation_ops
    # A multimodal checkpoint expands its vision tower through the shared builder.
    assert model.encoder_ops, "vision tower must be modeled, not skipped"


def test_qsa_maps_onto_msa_with_checkpoint_geometry():
    """MSA prices the block-sparse core; every dimension comes from the config."""
    ctx = _by_name(_build().context_ops)
    msa = ctx["context_qsa_attention"]

    assert isinstance(msa, ContextMSAModule)
    assert msa._scale_factor == 12.0  # 12 full_attention layers
    # tp=8: 24 q heads -> 3 local, 2 kv heads -> 1 local (ceil).
    assert (msa._num_heads, msa._num_kv_heads) == (3, 1)
    assert (msa._hidden_size, msa._head_dim) == (2560, 256)
    # block_topk (512 blocks) x compress_ratio (4 tokens/block) = 2048 tokens.
    assert (msa._block_size, msa._index_topk) == (4, 2048)
    assert (msa._index_n_heads, msa._index_head_dim) == (4, 128)


def test_qsa_supplements_the_projections_msa_cannot_price():
    """QSA's fused q+gate and indexer-K slices are outside MSA's SOL formula."""
    ctx = _by_name(_build().context_ops)

    gate = ctx["context_qsa_output_gate_gemm"]
    indexer_k = ctx["context_qsa_indexer_k_gemm"]
    # q_proj is 12288 = 2 x 24 x 256, so the gate slice is one 6144 chunk.
    assert (gate._n, gate._k) == (24 * 256 // 8, 2560)
    # index_qk_proj is 640 = (4 + 1) x 128; MSA prices only the query half.
    assert (indexer_k._n, indexer_k._k) == (128, 2560)


def test_hyper_connection_blocks_replace_the_absent_layer_norms():
    """Two 4-branch residual mixers per layer: attention side and MLP side."""
    ctx = _by_name(_build().context_ops)

    down = _with_prefix(ctx, "context_gdn_attn_hc_down_gemm")
    assert len(down) == 1
    down = down["context_gdn_attn_hc_down_gemm"]
    # hc_count * hidden = 4 * 2560 = 10240, unsharded: both runtimes build the
    # gated-residual projections as replicated linears (nn.Linear in SGLang,
    # ReplicatedLinear / disable_tp=True in vLLM), so TP does not divide them.
    assert (down._n, down._k) == (320, 10240)
    up = ctx["context_gdn_attn_hc_up_gemm"]
    assert (up._n, up._k) == (10240, 320)
    inject = ctx["context_gdn_attn_hc_inject_gemm"]
    assert (inject._n, inject._k) == (4, 10240)

    # 48 decoder layers split 36 GDN / 12 QSA, each with an attn and an MLP block.
    mixers = [name for name in ctx if name.endswith("_hc_down_gemm")]
    assert len(mixers) == 2 * 2, "one attn + one mlp mixer per layer family"
    assert ctx["context_gdn_attn_hc_down_gemm"]._scale_factor == 36.0
    assert ctx["context_qsa_mlp_hc_down_gemm"]._scale_factor == 12.0


def test_ple_branch_is_priced_once_with_the_checkpoint_dtype():
    ctx = _by_name(_build().context_ops)

    tables = [op for op in ctx.values() if isinstance(op, Embedding) and "ngram" in op._name]
    assert len(tables) == 1
    table = tables[0]
    # 20e6 rows TP-sharded, 2560-wide rows.
    assert (table._row_size, table._column_size) == (20_000_000 // 8, 2560)
    assert table._scale_factor == 1.0  # one PLE layer (1-based id 2)

    model = _build(FP8)
    # FP8 releases store the table as F8_E4M3: 172 GB of FP8 weights is
    # 120.8 GB routed experts + 51.2 GB of table. BF16 pricing would add 51 GB.
    assert model._ple_table_gemm_quant_mode == common.GEMMQuantMode.fp8_block
    assert _build(BF16)._ple_table_gemm_quant_mode == common.GEMMQuantMode.bfloat16


@pytest.mark.parametrize("model_path", [BF16, FP8])
def test_resident_weights_match_the_published_checkpoint(model_path):
    """The inventory must reproduce each repository's real byte size."""
    model = _build(model_path)
    per_rank = model.get_resident_weights_bytes()

    # The inventory is per rank. TP-sharded weights recover by multiplying by
    # TP, but the replicated share (gated residual + the PLE merged projection)
    # is already whole on every rank and must not be counted TP times.
    replicated = _replicated_weights_bytes(model)
    total = (per_rank - replicated) * model.config.tp_size + replicated

    # What is left over is the vision tower (carried by encoder_ops, not the
    # decoder inventory) and the packed-vs-inventory rounding.
    assert abs(total - REPO_BYTES[model_path]) / REPO_BYTES[model_path] < 0.02


def test_ple_branch_prices_every_checkpoint_module():
    """All nine ``layers.1.ple.*`` tensors must reach the op graph.

    Read off the published safetensors headers: the table (128 storage shards
    behind a 16-head gather) plus ``key_proj``, ``value_proj``, ``conv1d`` and
    the ``norm_key``/``norm_query``/``norm_conv`` group norms. A missing
    ``value_proj`` under-reported the PLE branch by 6.6M parameters per layer,
    so this pins the whole set rather than a representative member.
    """
    ctx = _by_name(_build().context_ops)

    expected = {
        "context_ple_ngram_embedding",
        "context_ple_key_proj_gemm",
        "context_ple_value_proj_gemm",
        "context_ple_norm_key",
        "context_ple_norm_query",
        "context_ple_conv1d",
        "context_ple_norm_conv",
    }
    missing = expected - set(ctx)
    assert not missing, f"PLE modules absent from the op graph: {sorted(missing)}"

    # kv_proj is the merged [hc_count * hidden, hidden] projection from the
    # gathered table width, replicated in both runtimes.
    key, value = ctx["context_ple_key_proj_gemm"], ctx["context_ple_value_proj_gemm"]
    assert (key._n, key._k) == (10240, 2560)
    assert (value._n, value._k) == (2560, 2560)
    # One PLE layer, so every branch op carries the same 1.0 layer count.
    assert {ctx[name]._scale_factor for name in expected if name != "context_ple_ngram_embedding"} == {1.0}


def test_kv_cache_prices_the_qsa_indexer_block_keys():
    model = _build()
    # 12 QSA layers x (2 kv heads ceil-sharded to 1 x 256 dims x 2) = 6144,
    # plus indexer keys: 1 kv head x 128 dims per 4-token block = 384.
    assert model.get_kvcache_elements_per_token() == 6528


@pytest.mark.parametrize(
    ("model_path", "moe_mode"),
    [(BF16, common.MoEQuantMode.bfloat16), (FP8, common.MoEQuantMode.fp8_block)],
)
def test_quant_modes_follow_checkpoint_exclusions(model_path, moe_mode):
    """Only the routed experts are quantized in the FP8 release."""
    model = _build(model_path)

    assert model.config.moe_quant_mode == moe_mode
    # q/k/v/o, the GDN block, shared experts, hyper connections, PLE projections
    # and the MTP FC pair are all in modules_to_not_convert.
    assert model._projection_gemm_quant_mode == common.GEMMQuantMode.bfloat16
    assert model._shared_expert_gemm_quant_mode == common.GEMMQuantMode.bfloat16
    assert model._hc_gemm_quant_mode == common.GEMMQuantMode.bfloat16
    assert model._ple_gemm_quant_mode == common.GEMMQuantMode.bfloat16
    assert model._mtp_fc_gemm_quant_mode == common.GEMMQuantMode.bfloat16


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_fp8_qsa_module_keys_on_the_bf16_kernels_the_runtime_runs(backend):
    """The FP8 release quantizes routed experts only, so QSA and KV stay BF16.

    The checkpoint carries no ``kv_cache_scheme`` and lists every attention
    projection in ``modules_to_not_convert``; both runtimes run the QSA module
    at the projection dtype. Inferring an FP8 KV pool from the weight precision
    alone produced a DsaKey (``kv=fp8``, and ``fmha=fp8`` on SGLang) that no
    collected row can satisfy — the FP8 checkpoint was unpriceable in SILICON.
    """
    for model_path in (BF16, FP8):
        model = models.get_model(model_path, _model_config(), backend)
        assert model.config.kvcache_quant_mode == common.KVCacheQuantMode.bfloat16
        assert model.config.fmha_quant_mode == common.FMHAQuantMode.bfloat16

        module = _by_name(model.context_ops)["context_qsa_attention"]
        spec = json.loads(module._spec_json())["MsaContext"]
        assert spec["kv_cache_dtype"] == "bfloat16"
        assert spec["fmha_quant_mode"] == "bfloat16"


@pytest.mark.parametrize(
    ("explicit_kv", "expected_kv", "expected_fmha"),
    [
        (None, common.KVCacheQuantMode.bfloat16, common.FMHAQuantMode.bfloat16),
        ("fp8", common.KVCacheQuantMode.fp8, common.FMHAQuantMode.fp8),
        ("none", common.KVCacheQuantMode.bfloat16, common.FMHAQuantMode.bfloat16),
    ],
)
def test_checkpoint_exclusions_only_default_the_kv_pool(explicit_kv, expected_kv, expected_fmha):
    """The checkpoint-scoped BF16 default must not clobber an explicit algorithm.

    ``test_fp8_qsa_module_keys_on_the_bf16_kernels_the_runtime_runs`` builds a
    ``ModelConfig`` directly, so it never reaches
    ``_infer_quant_modes_from_raw_config`` where the guard lives. This walks
    the raw config path instead: an absent key gets the BF16 default, while an
    explicit ``fp8`` (or an explicit unquantized value) keeps the caller's pool.
    """
    from aisim_core.sdk import utils
    from aisim_core.sdk.models import helpers

    raw = dict(utils._load_model_config_from_model_path(FP8))
    if explicit_kv is not None:
        raw["kv_cache_quant_algo"] = explicit_kv
    modes = helpers._infer_quant_modes_from_raw_config(raw, raw["architectures"][0])
    assert modes["kvcache_quant_mode"] == expected_kv
    assert modes["fmha_quant_mode"] == expected_fmha


def test_explicit_kv_quant_mode_still_wins_over_the_checkpoint_default():
    """A caller that asks for an FP8 KV pool must keep it (and fail loudly)."""
    model = models.get_model(
        FP8,
        dataclasses.replace(_model_config(), kvcache_quant_mode=common.KVCacheQuantMode.fp8),
        "vllm",
    )
    module = _by_name(model.context_ops)["context_qsa_attention"]
    assert json.loads(module._spec_json())["MsaContext"]["kv_cache_dtype"] == "fp8"


def test_qsa_indexer_projection_reads_as_unquantized():
    """Regression: the indexer is excluded under a name DeepSeek never uses.

    QSA names its fused indexer projection ``self_attn.indexer.index_qk_proj``.
    The projection markers only knew wq_b/wk/weights_proj, so the indexer read
    as quantized even though the checkpoint keeps it BF16, which under-predicts
    indexer GEMM cost.
    """
    from aisim_core.sdk.models.helpers import attention_projection_exclusions
    from aisim_core.sdk.utils import get_model_config_from_model_path

    raw = get_model_config_from_model_path(FP8)["raw_config"]
    assert "indexer" in attention_projection_exclusions(raw)
    assert attention_projection_exclusions(get_model_config_from_model_path(BF16)["raw_config"]) == frozenset()


def test_generation_prices_the_mtp_draft_outside_the_mixed_stack():
    """MTP's draft layer is QSA, so scaling the GDN/QSA mix would mis-assign it."""
    gen = _by_name(_build().generation_ops)

    # The decoder keeps its real counts; ``mtp_scale_factor`` must not leak in.
    assert gen["generation_qsa_attention"]._scale_factor == 12.0
    assert gen["generation_gdn_in_proj_gemm"]._scale_factor == 36.0
    assert gen["generation_qsa_attention"]._scale_factor != 12.0 * _build()._mtp_scale_factor

    draft = _with_prefix(gen, "draft_")
    assert isinstance(draft["draft_attention"], GenerationMSAModule)
    assert draft["draft_attention"]._scale_factor == 1.0
    # No GDN in the draft: its layer_types are ["full_attention"].
    assert not [name for name in draft if "gdn" in name]
    # fc_embedding / fc_hidden are ``ColumnParallelLinear(h, h,
    # gather_output=True)``: at TP8 each rank owns n=320 output columns at the
    # full k=2560 reduction width.
    assert (draft["draft_fc_embedding_gemm"]._n, draft["draft_fc_embedding_gemm"]._k) == (320, 2560)
    assert (draft["draft_fc_hidden_gemm"]._n, draft["draft_fc_hidden_gemm"]._k) == (320, 2560)
    # pre_fc_norm_hidden normalizes the whole hc_count*h manifold on every rank
    # (no TP shard): 2 bytes in + 2 bytes out per element, hc_count=4, h=2560.
    assert json.loads(draft["draft_pre_fc_norm_hidden"]._spec_json())["Elementwise"]["bytes_per_token"] == 81920.0


def test_nextn_zero_drops_the_draft_stack_and_matches_scaling():
    gen = _by_name(_build(nextn=0).generation_ops)

    assert not [name for name in gen if name.startswith("draft_")]
    assert gen["generation_qsa_attention"]._scale_factor == 12.0


def test_ple_table_shard_width_must_divide_tp():
    """The n-gram table is a ``VocabParallelEmbedding``: rows // tp must be exact.

    ``_ple_ops`` prices ``ngram_vocab_size_base // tp`` rows, so an indivisible
    table would silently under-report resident weights on every rank.
    """
    model = _build()
    validator = Qwen4ExpModel._validate_qwen4exp_geometry

    def probe(tp_size, vocab_base):
        fake = types.SimpleNamespace(
            _hidden_size=2560,
            config=config.ModelConfig(tp_size=tp_size),
            extra_params=dataclasses.replace(model.extra_params, ngram_vocab_size_base=vocab_base),
        )
        validator(fake)

    probe(8, 20000000)
    with pytest.raises(ValueError, match="must be divisible by tp_size"):
        probe(3, 20000002)


def test_draft_weights_are_part_of_the_resident_inventory():
    """The draft layer shares no weights with the GDN decoder stack."""
    with_mtp = _build(nextn=1)
    without = _build(nextn=0)

    draft = sum(op.get_weights() for op in with_mtp.generation_ops if op._name.startswith("draft_"))
    assert draft > 0
    assert with_mtp.get_resident_weights_bytes() > without.get_resident_weights_bytes()


@pytest.mark.parametrize("model_path", [BF16, FP8])
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_sol_prices_both_phases_through_the_rust_engine(model_path, backend):
    """The whole graph must be priceable, not merely constructible.

    SOL needs no collected rows, so this is the only end-to-end guard that runs
    without SILICON data for Qwen4-Exp shapes (MQSA rows are not collected yet).
    """
    from aisim_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    model = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model=model_path,
            system="h200_sxm",
            backend=backend,
            worker_type="aggregated",
            tp=8,
            moe_tp_size=1,
            moe_ep_size=8,
            nextn=1,
            database_mode="SOL",
            estimation_mode="op_level",
            fallback_policy="deny",
        )
    )
    try:
        prefill = model.static_phase_latency(batch_size=1, input_tokens=8192, output_tokens=2, prefill=True)
        decode = model.static_phase_latency(batch_size=1, input_tokens=8192, output_tokens=2, prefill=False)
    finally:
        model.close()

    assert prefill > 0
    assert decode > 0
    # A single decode step is orders of magnitude cheaper than prefill; a
    # collapsed ratio means the phase graphs were swapped or mis-scaled.
    assert prefill / decode > 10


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_qsa_module_ops_read_the_checkpoint_architecture(backend):
    """MSA tables are keyed by architecture, so QSA must not read the M3 slice.

    The accepted C3.3 rows are keyed by
    ``Qwen4ExpForConditionalGeneration``; an op that keeps the
    ``MiniMaxM3ForCausalLM`` default would miss them (and, if the tables shared a
    slice, interpolate M3 head geometry). The rows reach a default install once
    an ``ai-dynamo/aisim-data`` revision carries them; this test pins the key,
    not the publication state.
    """
    model = models.get_model(BF16, _model_config(), backend)
    qsa = [
        op
        for op in _walk(model.context_ops + model.generation_ops)
        if isinstance(op, (ContextMSAModule, GenerationMSAModule))
    ]
    assert qsa, "Qwen3.8-Flash-Next must build QSA module ops"
    assert {op._architecture for op in qsa} == {"Qwen4ExpForConditionalGeneration"}


def test_geometry_guards_reject_unpricable_configs():
    model = _build()
    cfg = model.extra_params

    for broken, message in (
        (dataclasses.replace(cfg, hc_count=1), "hyper-connection"),
        (dataclasses.replace(cfg, indexer_budget=2), "block_topk"),
        (dataclasses.replace(cfg, indexer_n_heads=0), "indexer geometry"),
    ):
        model.extra_params = broken
        with pytest.raises(ValueError, match=message):
            model._validate_qwen4exp_geometry()
    model.extra_params = cfg
    model._validate_qwen4exp_geometry()
