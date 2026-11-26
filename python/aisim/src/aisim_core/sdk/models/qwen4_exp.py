# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8-Flash-Next (Qwen4-Exp): GDN + QSA hybrid MoE with PLE and gated residual.

Shares the Qwen3.5 GDN contract (``linear_*``) but differs in four structural
ways, each of which is priced explicitly below:

1. **QSA replaces GQA on full-attention layers.** Qwen Sparse Attention is a
   block-sparse GQA: a compressed indexer scores KV blocks of ``block_size``
   tokens and full attention runs over the top-``block_topk`` blocks. It is
   modeled with the MSA module op (``index_topk = block_topk * block_size``).
2. **Gated residual ("hyper connection").** Every attention and MLP block is
   preceded by a 4-way residual mixer: an RMS norm over the concatenated
   ``hc_count * hidden_size`` manifold, a low-rank down/up pair
   (``hidden*hc_count -> hc_lowrank -> hidden*hc_count``) and a tiny
   ``-> hc_count`` injection GEMM. These replace the usual
   ``input_layernorm``/``post_attention_layernorm`` weights, which the
   checkpoint does not contain. The manifold is **replicated**, not TP-sharded:
   both runtimes build these as replicated linears even though the rest of the
   decoder shards, so the full ``hc_count * hidden_size`` width is the per-rank
   cost.
3. **N-gram PLE.** One decoder layer carries a 20M-row x 2560-wide n-gram
   embedding table (51.2e9 parameters, 102.4 GB BF16) plus the eight modules
   that consume it: a merged ``kv_proj`` (``ple_embed_dim -> [hc_count *
   hidden_size, hidden_size]``), a depthwise ``conv1d`` over the manifold and
   the ``norm_key``/``norm_query``/``norm_conv`` group norms. Only the table is
   TP-sharded; the projections and norms are replicated like the mixers.
4. **MTP draft layer is QSA, not GDN.** The draft stack is a single
   full-attention (QSA) MoE layer with its own hyper-connection blocks and
   ``fc_embedding``/``fc_hidden`` projections, so the mixed decoder stack cannot
   be scaled by ``mtp_scale_factor`` without assigning the draft to the wrong op
   class (12 QSA layers would become 12.25 instead of 13).

Geometry is taken from the checkpoint, not inferred: GDN is 16 key / 48 value
heads at 128 dims (``in_proj_qkv`` 10240 = 2*16*128 + 48*128), QSA is 24 q / 2 kv
heads at 256 dims with a doubled ``q_proj`` (12288 = 2*24*256, sigmoid output
gate), and the indexer is 4 query / 1 KV head at 128 dims (``index_qk_proj``
640 = (4+1)*128).

Modeling caveats that gate accuracy qualification (see
``model_configs/QWEN38_FLASH_NEXT_PROVENANCE.md``):

* The MSA module op prices its own projections from a formula
  (Q, K+V, O, indexer-Q). Qwen4-Exp fuses two more halves into those
  projections: the sigmoid output gate that shares q_proj's weight, and the
  indexer K half of ``index_qk_proj``. The op carries their widths
  (``output_gate_dim`` / ``indexer_k_dim``) so the formula prices them inside
  the module that is measured, rather than as GEMMs outside it. That matters
  because SILICON/HYBRID return the whole module's measured latency: an
  outside GEMM would be charged twice once the QSA rows resolve.
* The MSA analytic path assumes an FP8 indexer. This checkpoint keeps the
  indexer in BF16 (it is listed in ``modules_to_not_convert``), so the analytic
  graph under-predicts index time; the measured QSA module rows are the anchor
  when the lookup resolves.
* Without ``msa_context_module_perf`` / ``msa_generation_module_perf`` rows at
  Qwen4-Exp shapes the op would otherwise fall back to a DSA cross-op transfer,
  which the MiniMax-M3 work measured 15-28x below silicon. The module op now
  enforces that: a non-M3 ``architecture`` with no rows raises the terminal
  empirical miss instead of borrowing (``operators/msa.rs``). The Qwen-geometry
  rows are staged in ``benchmarks/evidence/collector/qwen38-c3.5`` and reach
  SILICON only once the ``ai-dynamo/aisim-data`` revision that carries them is
  published, the pin is raised, and the collected version is queryable (SGLang
  0.5.21 is; vLLM 0.31.0 needs a slot promotion or
  ``AIC_ALLOW_UNLISTED_VERSIONS=1``). Until then SILICON and HYBRID raise, and
  SOL answers analytically. Serving accuracy (8-GPU serve vs replay) is not
  yet qualified.
* The n-gram table is priced as TP-sharded HBM residency, which is the
  ``EngramConfig.cpu_offload=False`` case. vLLM 0.31 defaults that flag to
  ``True``, keeping the table in pinned host memory and looking it up through
  UVA; leaving the default would make this inventory over-report GPU memory by
  the table's share. HBM residency is the only model AISimulate implements for
  n-gram tables, and the DeepSeek-V4.1 FPM identity pins the same flag
  explicitly (``sdk/fpm_identity.py`` raises unless it is ``False``), so the
  launch must pin it here too: ``--engram-config '{"cpu_offload": false}'`` on
  vLLM (``EngramConfig`` is only exposed as that grouped option) and
  ``--no-ple-offload-embedding`` on SGLang. Per rank the table is
  ``num_embeddings / etp_size``, where the ETP group is TP x DP only when
  ``EngramConfig.embedding_across_dp`` is on.
* Sharding is deliberately uneven and was read off the pinned runtime source
  rather than assumed. The n-gram table is a ``VocabParallelEmbedding`` (each
  rank holds ``vocab / tp`` rows), while the gated-residual projections, the
  PLE merged ``kv_proj``, its conv and its norms are replicated on every rank
  (``ReplicatedLinear`` / ``MergedColumnParallelLinear(disable_tp=True)`` in
  vLLM, plain ``nn.Linear`` in SGLang). Treating the replicated share as
  TP-sharded under-reports per-rank memory by ~1.2 GB and under-predicts the
  gated-residual and PLE compute by the TP factor.
* The FP8 release stores the table with a single per-tensor ``weight_scale``,
  while the model reports the model-wide GEMM mode (``fp8_block``). The
  embedding op is priced by bytes, and both modes are one byte per element, so
  this is a label mismatch with no cost effect.
"""

from __future__ import annotations

import aisim_core.sdk.operations as ops
from aisim_core.sdk import common
from aisim_core.sdk.models.base import BaseModel, register_model
from aisim_core.sdk.models.helpers import quant_exclude_patterns
from aisim_core.sdk.models.qwen35 import Qwen35Model

# Substrings that identify a module family in the checkpoint's exclusion globs.
_HC_EXCLUSION_MARKER = "hyper_connection"
_PLE_EXCLUSION_MARKER = ".ple."
_MTP_FC_EXCLUSION_MARKERS = ("fc_embedding", "fc_hidden")
# The n-gram table follows the checkpoint dtype: only its shard-projection
# siblings are listed in ``modules_to_not_convert``.
_PLE_TABLE_QUANT_MARKER = "ngram_embedding"

# Prefix on every MTP draft op. ``BaseModel.get_resident_weights_bytes`` skips
# draft-prefixed ops in the phase inventory; this class adds them back through
# ``get_resident_weights_bytes`` because a QSA draft layer shares no weights
# with the GDN decoder.
_DRAFT_PREFIX = "draft_"


def _excluded_gemm_mode(raw_config: dict, markers: tuple[str, ...], default: common.GEMMQuantMode):
    """``default`` unless a checkpoint exclusion glob names one of ``markers``.

    Mirrors ``_qwen35_mixed_precision_gemm_modes`` for the module families Qwen4-Exp
    adds. A checkpoint that excludes a family keeps it in BF16 regardless of the
    model-wide mode, which is what both serving engines do.
    """
    patterns = (str(p).lower() for p in quant_exclude_patterns(raw_config))
    if any(marker in pattern for pattern in patterns for marker in markers):
        return common.GEMMQuantMode.bfloat16
    return default


@register_model("QWEN4EXP")
class Qwen4ExpModel(Qwen35Model):
    """Qwen3.8-Flash-Next decoder: GDN + QSA hybrid MoE with PLE and gated residual."""

    @classmethod
    def create(cls, model_info: dict, model_config, backend_name: str) -> BaseModel:
        return cls(
            model_info["model_path"],
            model_info["model_family"],
            model_info["architecture"],
            model_info["layers"],
            model_info["n"],
            model_info["n_kv"],
            model_info["d"],
            model_info["hidden_size"],
            model_info["inter_size"],
            model_info["vocab"],
            model_info["context"],
            model_config,
            model_info["extra_params"],
            raw_config=model_info["raw_config"],
            allow_checkpoint_split=not model_info["gemm_quant_mode_is_explicit"],
            backend_name=backend_name,
        )

    def __init__(
        self,
        *args,
        raw_config: dict | None = None,
        allow_checkpoint_split: bool = True,
        backend_name: str,
    ) -> None:
        # Qwen35Model.__init__ builds the op graphs, so the raw config the extra
        # quant families resolve against must be reachable before ``super()``.
        # Nothing later in the chain resets it.
        self._raw_config = raw_config or {}
        # Qwen35Model.__init__ validates the shared GDN contract, resolves the
        # SSM state dtype, the projection/shared-expert/FFN quant modes and the
        # vision tower, then dispatches to the overridden builders below.
        super().__init__(
            *args,
            raw_config=raw_config,
            allow_checkpoint_split=allow_checkpoint_split,
            backend_name=backend_name,
        )
        assert isinstance(self.extra_params, common.Qwen4ExpConfig), (
            "Qwen4ExpModel requires Qwen4ExpConfig extra_params"
        )

    # ------------------------------------------------------------------
    # Quantization: the module families Qwen4-Exp adds over Qwen3.5
    # ------------------------------------------------------------------

    def _extra_gemm_mode(self, markers: tuple[str, ...]) -> common.GEMMQuantMode:
        """Checkpoint-resolved GEMM mode for one added module family (cached).

        Resolution is deferred to first use because ``self.config`` only exists
        once ``BaseModel.__init__`` has run, while the op graphs are built later
        in the same constructor.
        """
        cache = self.__dict__.setdefault("_extra_mode_cache", {})
        if markers not in cache:
            cache[markers] = _excluded_gemm_mode(self._raw_config, markers, self.config.gemm_quant_mode)
        return cache[markers]

    @property
    def _hc_gemm_quant_mode(self) -> common.GEMMQuantMode:
        return self._extra_gemm_mode((_HC_EXCLUSION_MARKER,))

    @property
    def _ple_gemm_quant_mode(self) -> common.GEMMQuantMode:
        return self._extra_gemm_mode((_PLE_EXCLUSION_MARKER,))

    @property
    def _mtp_fc_gemm_quant_mode(self) -> common.GEMMQuantMode:
        return self._extra_gemm_mode(_MTP_FC_EXCLUSION_MARKERS)

    @property
    def _ple_table_gemm_quant_mode(self) -> common.GEMMQuantMode:
        """Dtype of the 51.2e9-parameter n-gram table.

        FP8 releases quantize the table: its shards are absent from
        ``modules_to_not_convert`` while ``ple.conv1d``/``key_proj``/``value_proj``
        are listed, and the FP8 repository stores it as F8_E4M3 (172 GB of FP8
        weights = 120.8 GB routed experts + 51.2 GB table). Pricing it BF16
        over-reports resident weights by 51 GB per model.
        """
        return self._extra_gemm_mode((_PLE_TABLE_QUANT_MARKER,))

    def _validate_qwen4exp_geometry(self) -> None:
        """Reject geometries the op graph cannot price, before any op is built."""
        cfg: common.Qwen4ExpConfig = self.extra_params
        if cfg.indexer_head_dim <= 0 or cfg.indexer_n_heads <= 0 or cfg.indexer_kv_heads <= 0:
            raise ValueError("Qwen3.8-Flash-Next QSA indexer geometry must be positive")
        if cfg.qsa_block_topk <= 0:
            raise ValueError(
                "Qwen3.8-Flash-Next QSA block_topk must be positive "
                f"(indexer_budget={cfg.indexer_budget}, compress_ratio={cfg.indexer_compress_ratio})"
            )
        if cfg.hc_count <= 1:
            raise ValueError("Qwen3.8-Flash-Next hyper-connection needs hc_count > 1")
        # The HC blocks are replicated, so ``hc_count * hidden`` is never divided
        # by TP (see ``_hc_block_ops``); there is no divisibility precondition.
        # The PLE table, however, is a ``VocabParallelEmbedding`` and IS sharded.
        if cfg.ple_layer_ids and cfg.ngram_vocab_size_base % self.config.tp_size != 0:
            raise ValueError(
                f"ngram_vocab_size_base ({cfg.ngram_vocab_size_base}) must be divisible by "
                f"tp_size ({self.config.tp_size}) for the PLE table shard"
            )

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def get_resident_weights_bytes(self) -> float:
        """Phase inventory plus the MTP draft layer and the dtype-correct table.

        The QSA draft layer shares no weights with the GDN decoder stack, so the
        context-op inventory cannot imply it. Draft ops are ``draft_``-prefixed
        and therefore excluded from the base sum; they are added back here at
        their own (nextn-scaled) weight.

        The generic ``Embedding`` op prices every table at BF16, but the n-gram
        table follows the checkpoint dtype (FP8 releases store it as F8_E4M3),
        so its share is re-priced here.
        """
        cfg: common.Qwen4ExpConfig = self.extra_params
        elements = len(cfg.ple_layer_indices) * (cfg.ngram_vocab_size_base // self.config.tp_size) * cfg.ple_embed_dim
        table_delta = elements * (self._ple_table_gemm_quant_mode.value.memory - 2.0)
        draft = sum(op.get_weights() for op in self.generation_ops if op._name.startswith(_DRAFT_PREFIX))
        return float(super().get_resident_weights_bytes() + draft + table_delta)

    def get_kvcache_elements_per_token(self) -> int:
        """Full-attention KV plus the indexer's per-block key cache.

        QSA does not cache indexer queries or values: the indexer keeps one
        ``index_kv_heads * index_head_dim`` key vector per ``block_size`` tokens.
        It is priced at the paged-cache element dtype, which matches the
        checkpoint (the indexer is BF16 even in the FP8 release).
        """
        cfg: common.Qwen4ExpConfig = self.extra_params
        n_qsa = cfg.layer_types.count("full_attention")
        n_kv_per_tp = (self._num_kv_heads + self.config.tp_size - 1) // self.config.tp_size
        per_token = 2 * n_kv_per_tp * self._head_size
        block_size = max(1, cfg.indexer_compress_ratio)
        indexer_per_token = (cfg.indexer_kv_heads * cfg.indexer_head_dim + block_size - 1) // block_size
        return n_qsa * (per_token + indexer_per_token)

    # ------------------------------------------------------------------
    # Op builders shared by both phases
    # ------------------------------------------------------------------

    def _hc_block_ops(self, phase: str, prefix: str, count: float, cfg: common.Qwen4ExpConfig) -> list:
        """One gated-residual ("hyper connection") block.

        Norm over the ``hc_count * hidden`` manifold, down to ``hc_lowrank``, back
        up, and a ``-> hc_count`` injection GEMV that weights the branches.

        Every width here is the full, unsharded one. Both runtimes build these as
        ``ReplicatedLinear``, and the down+inject pair as a
        ``MergedColumnParallelLinear`` with ``disable_tp=True``: the manifold
        state is replicated on every rank, so neither the activations nor the
        weights divide by TP. Dividing them would under-report both compute and
        resident memory.
        """
        hc_w = cfg.hc_count * self._hidden_size
        return [
            ops.ElementWise(f"{prefix}_hc_norm", count, 2 * hc_w, 2 * hc_w, 0.8),
            ops.GEMM(f"{prefix}_hc_down_gemm", count, cfg.hc_lowrank, hc_w, self._hc_gemm_quant_mode),
            ops.GEMM(
                f"{prefix}_hc_up_gemm",
                count,
                hc_w,
                cfg.hc_lowrank,
                self._hc_gemm_quant_mode,
                low_precision_input=True,
            ),
            # hc_count-wide output sits far below the collected GEMM n-grid.
            ops.GEMM(
                f"{prefix}_hc_inject_gemm",
                count,
                cfg.hc_count,
                hc_w,
                common.GEMMQuantMode.bfloat16,
                below_grid_sol=True,
            ),
        ]

    def _qsa_ops(self, phase: str, prefix: str, count: float, cfg: common.Qwen4ExpConfig) -> list:
        """QSA block: the MSA module op, priced with the checkpoint's fused halves.

        The MSA module op prices Q, K+V, O, indexer-Q and — through
        ``output_gate_dim`` / ``indexer_k_dim`` — the two halves Qwen4-Exp fuses
        into those projections: the sigmoid output gate that shares q_proj's
        weight (``2 * heads * head_dim`` wide) and the indexer K half of the
        fused ``index_qk_proj`` (``indexer_kv_heads * indexer_head_dim``). Both
        stay inside the op because the module is what the collector measures:
        SILICON and HYBRID return the whole module's latency, so a separate
        GEMM outside it would be charged twice. See the module docstring.
        """
        tp = self.config.tp_size
        local_heads = max(1, self._num_heads // tp)
        local_kv_heads = max(1, self._num_kv_heads // tp)
        module_cls = ops.ContextMSAModule if phase == "context" else ops.GenerationMSAModule
        module = module_cls(
            f"{prefix}_attention",
            count,
            local_heads,
            local_kv_heads,
            self._hidden_size,
            self._head_size,
            self._head_size,
            cfg.indexer_n_heads,
            cfg.indexer_head_dim,
            cfg.qsa_block_topk * cfg.indexer_compress_ratio,
            cfg.indexer_compress_ratio,
            self.config.kvcache_quant_mode,
            self.config.fmha_quant_mode,
            self._projection_gemm_quant_mode,
            # QSA rows live under the checkpoint's own architecture in the msa
            # tables; the op must not read the M3 slice.
            architecture="Qwen4ExpForConditionalGeneration",
            # Fused projection halves the M3 formula has no slot for.
            output_gate_dim=local_heads * self._head_size,
            indexer_k_dim=cfg.indexer_kv_heads * cfg.indexer_head_dim,
        )
        return [
            # q/k RMSNorm plus the fused residual add.
            ops.ElementWise(f"{prefix}_norm", count, 2 * self._hidden_size, 2 * self._hidden_size, 0.8),
            module,
        ]

    def _ple_ops(self, phase: str, prefix: str, count: float, cfg: common.Qwen4ExpConfig) -> list:
        """n-gram PLE branch, mirroring the checkpoint's nine modules.

        Checked against the runtime rather than inferred: the table is gathered
        per n-gram head (16 heads x 160 dims = 2560), a single merged
        ``kv_proj`` maps that to ``[hc_count * hidden, hidden]`` = ``[10240,
        2560]``, and ``ple_gate`` consumes its ``key``/``value`` halves together
        with the residual hidden state under ``norm_key``/``norm_query``.
        ``conv1d`` is depthwise over the manifold (kernel 4, dilation
        ``ngram_size``) and ``norm_conv`` follows it.

        The checkpoint stores ``key_proj`` and ``value_proj`` as the two halves
        of that merged projection, so the pair is emitted as two GEMMs whose
        combined FLOPs are exact.

        Sharding is deliberately uneven, and both halves were read off the
        runtime source:

        * the table is ``VocabParallelEmbedding`` over the ETP group, so each
          rank holds ``vocab // tp`` rows (``etp_size`` is TP, and TP x DP only
          when ``EngramConfig.embedding_across_dp`` is on, which it is not by
          default);
        * ``kv_proj``, the three grouped norms and ``conv1d`` are all replicated
          (``MergedColumnParallelLinear(disable_tp=True)`` and plain ``nn``
          modules), so their widths do not divide by TP.

        Placement caveat: vLLM's ``EngramConfig.cpu_offload`` defaults to
        ``True``, which keeps the table in pinned host memory and looks it up
        through UVA. The residency priced here (HBM, TP-sharded) is the
        ``cpu_offload=False`` case, which is what this project requires for the
        FPM execution identity; the collection launch must pin that flag. See
        ``model_configs/QWEN38_FLASH_NEXT_PROVENANCE.md``.
        """
        h = self._hidden_size
        hc_w = cfg.hc_count * h
        return [
            ops.Embedding(
                f"{prefix}_ngram_embedding",
                count,
                cfg.ngram_vocab_size_base // self.config.tp_size,
                cfg.ple_embed_dim,
                0.3,
            ),
            # Merged kv_proj, first half: 2560 -> hc_count * hidden.
            ops.GEMM(f"{prefix}_key_proj_gemm", count, hc_w, cfg.ple_embed_dim, self._ple_gemm_quant_mode),
            # Merged kv_proj, second half: 2560 -> 2560.
            ops.GEMM(f"{prefix}_value_proj_gemm", count, h, cfg.ple_embed_dim, self._ple_gemm_quant_mode),
            ops.ElementWise(f"{prefix}_norm_key", count, 2 * hc_w, 2 * hc_w, 0.8),
            # The query side reuses the residual hidden state under the same
            # grouped norm, so its width is counted at the manifold width.
            ops.ElementWise(f"{prefix}_norm_query", count, 2 * hc_w, 2 * hc_w, 0.8),
            # Depthwise conv over the manifold: d_conv taps per element.
            ops.ElementWise(
                f"{prefix}_conv1d",
                count,
                cfg.ple_conv_kernel_size * hc_w,
                hc_w,
                0.8,
            ),
            ops.ElementWise(f"{prefix}_norm_conv", count, 2 * hc_w, 2 * hc_w, 0.8),
        ]

    # ------------------------------------------------------------------
    # Context (prefill)
    # ------------------------------------------------------------------

    def _build_context_ops(self) -> None:
        self._validate_qwen4exp_geometry()
        cfg: common.Qwen4ExpConfig = self.extra_params
        h = self._hidden_size
        tp = self.config.tp_size
        pp = self.config.pp_size
        moe_tp = self.config.moe_tp_size
        moe_ep = self.config.moe_ep_size
        attn_dp = self.config.attention_dp_size
        attn_ar_folded = self._sglang_folds_attn_ar()
        projection_gemm_q = self._projection_gemm_quant_mode
        dense_ffn_gemm_q = self._dense_ffn_gemm_quant_mode
        shared_expert_gemm_q = self._shared_expert_gemm_quant_mode
        moe_q = self.config.moe_quant_mode
        workload_dist = (
            self.config.workload_distribution + "_1.2"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )
        counts = self._count_layer_types()

        nk = cfg.linear_num_key_heads
        hk = cfg.linear_key_head_dim
        nv = cfg.linear_num_value_heads
        hv = cfg.linear_value_head_dim
        d_conv = cfg.linear_conv_kernel_dim

        gdn_nk_per_tp = nk // tp
        gdn_nv_per_tp = nv // tp

        self.context_ops = [
            ops.Embedding("context_embedding", 1, self._vocab_size // tp, h, 0.3),
            ops.CustomAllReduce("context_embedding_ar", 1, h, tp),
        ]

        # --- linear_attention (GDN) layers ---
        if counts["linear"] > 0:
            c = counts["linear"]
            gdn_in_proj_out = (nk * hk + nk * hk + nv * hv + nv * hv) // tp
            gdn_ba_out = 2 * nv // tp
            gdn_out_proj_in = nv * hv // tp
            self.context_ops.extend(self._hc_block_ops("context", "context_gdn_attn", c, cfg))
            self.context_ops.extend(
                [
                    ops.ElementWise("context_gdn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("context_gdn_in_proj_gemm", c, gdn_in_proj_out, h, projection_gemm_q),
                    # 2*nv/tp drops below the collected GEMM n-grid at high TP.
                    ops.GEMM("context_gdn_in_proj_ba_gemm", c, gdn_ba_out, h, projection_gemm_q, below_grid_sol=True),
                    ops.GDNKernel(
                        "context_gdn_conv1d",
                        c,
                        "causal_conv1d_fn",
                        "context",
                        h,
                        gdn_nk_per_tp,
                        hk,
                        gdn_nv_per_tp,
                        hv,
                        d_conv,
                        mamba_ssm_dtype=self._mamba_ssm_dtype,
                    ),
                    ops.GDNKernel(
                        "context_gdn_scan",
                        c,
                        "chunk_gated_delta_rule",
                        "context",
                        h,
                        gdn_nk_per_tp,
                        hk,
                        gdn_nv_per_tp,
                        hv,
                        d_conv,
                        mamba_ssm_dtype=self._mamba_ssm_dtype,
                    ),
                    ops.GEMM(
                        "context_gdn_out_proj_gemm",
                        c,
                        h,
                        gdn_out_proj_in,
                        projection_gemm_q,
                        low_precision_input=True,
                    ),
                    *([] if attn_ar_folded else [ops.CustomAllReduce("context_gdn_ar", c, h, tp)]),
                ]
            )
            self.context_ops.extend(self._hc_block_ops("context", "context_gdn_mlp", c, cfg))
            self.context_ops.extend(
                self._ffn_context_ops(
                    "context_gdn",
                    c,
                    h,
                    tp,
                    moe_tp,
                    moe_ep,
                    attn_dp,
                    dense_ffn_gemm_q,
                    shared_expert_gemm_q,
                    moe_q,
                    workload_dist,
                    cfg,
                )
            )

        # --- full_attention (QSA) layers ---
        if counts["full"] > 0:
            c = counts["full"]
            self.context_ops.extend(self._hc_block_ops("context", "context_qsa_attn", c, cfg))
            self.context_ops.extend(self._qsa_ops("context", "context_qsa", c, cfg))
            self.context_ops.extend(
                [
                    *([] if attn_ar_folded else [ops.CustomAllReduce("context_qsa_ar", c, h, tp)]),
                ]
            )
            self.context_ops.extend(self._hc_block_ops("context", "context_qsa_mlp", c, cfg))
            self.context_ops.extend(
                self._ffn_context_ops(
                    "context_qsa",
                    c,
                    h,
                    tp,
                    moe_tp,
                    moe_ep,
                    attn_dp,
                    dense_ffn_gemm_q,
                    shared_expert_gemm_q,
                    moe_q,
                    workload_dist,
                    cfg,
                )
            )

        # --- n-gram PLE branches (1-based ple_layer_ids; decoder layers only) ---
        n_ple = len(cfg.ple_layer_indices)
        if n_ple:
            self.context_ops.extend(self._ple_ops("context", "context_ple", n_ple, cfg))

        self.context_ops.extend(
            [
                ops.GEMM("context_logits_gemm", 1, self._vocab_size // tp, h, common.GEMMQuantMode.bfloat16),
                ops.P2P("context_p2p", pp - 1, h, pp),
            ]
        )

    # ------------------------------------------------------------------
    # Generation (decode)
    # ------------------------------------------------------------------

    def _build_generation_ops(self) -> None:
        self._validate_qwen4exp_geometry()
        cfg: common.Qwen4ExpConfig = self.extra_params
        h = self._hidden_size
        tp = self.config.tp_size
        pp = self.config.pp_size
        moe_tp = self.config.moe_tp_size
        moe_ep = self.config.moe_ep_size
        attn_dp = self.config.attention_dp_size
        attn_ar_folded = self._sglang_folds_attn_ar()
        projection_gemm_q = self._projection_gemm_quant_mode
        dense_ffn_gemm_q = self._dense_ffn_gemm_quant_mode
        shared_expert_gemm_q = self._shared_expert_gemm_quant_mode
        moe_q = self.config.moe_quant_mode
        workload_dist = (
            self.config.workload_distribution + "_1.2"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )
        counts = self._count_layer_types()

        nk = cfg.linear_num_key_heads
        hk = cfg.linear_key_head_dim
        nv = cfg.linear_num_value_heads
        hv = cfg.linear_value_head_dim
        d_conv = cfg.linear_conv_kernel_dim

        gdn_nk_per_tp = nk // tp
        gdn_nv_per_tp = nv // tp

        # The decoder stack is priced at its real layer counts; the MTP draft
        # layer is a separate group below (see the module docstring). Only the
        # per-iteration token I/O keeps the ``mtp_scale_factor`` convention.
        sf = self._mtp_scale_factor

        self.generation_ops = [
            ops.Embedding("generation_embedding", 1 * sf, self._vocab_size // tp, h, 0.3),
            ops.CustomAllReduce("generation_embedding_ar", 1 * sf, h, tp),
        ]

        # --- linear_attention (GDN) layers ---
        if counts["linear"] > 0:
            c = counts["linear"]
            gdn_in_proj_out = (nk * hk + nk * hk + nv * hv + nv * hv) // tp
            gdn_ba_out = 2 * nv // tp
            gdn_out_proj_in = nv * hv // tp
            self.generation_ops.extend(self._hc_block_ops("generation", "generation_gdn_attn", c, cfg))
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_gdn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("generation_gdn_in_proj_gemm", c, gdn_in_proj_out, h, projection_gemm_q),
                    ops.GEMM(
                        "generation_gdn_in_proj_ba_gemm", c, gdn_ba_out, h, projection_gemm_q, below_grid_sol=True
                    ),
                    ops.GDNKernel(
                        "generation_gdn_conv1d",
                        c,
                        "causal_conv1d_update",
                        "generation",
                        h,
                        gdn_nk_per_tp,
                        hk,
                        gdn_nv_per_tp,
                        hv,
                        d_conv,
                        mamba_ssm_dtype=self._mamba_ssm_dtype,
                    ),
                    ops.GDNKernel(
                        "generation_gdn_recurrence",
                        c,
                        "fused_sigmoid_gating_delta_rule_update",
                        "generation",
                        h,
                        gdn_nk_per_tp,
                        hk,
                        gdn_nv_per_tp,
                        hv,
                        d_conv,
                        mamba_ssm_dtype=self._mamba_ssm_dtype,
                    ),
                    ops.GEMM(
                        "generation_gdn_out_proj_gemm",
                        c,
                        h,
                        gdn_out_proj_in,
                        projection_gemm_q,
                        low_precision_input=True,
                    ),
                    *([] if attn_ar_folded else [ops.CustomAllReduce("generation_gdn_ar", c, h, tp)]),
                ]
            )
            self.generation_ops.extend(self._hc_block_ops("generation", "generation_gdn_mlp", c, cfg))
            self.generation_ops.extend(
                self._ffn_generation_ops(
                    "generation_gdn",
                    c,
                    h,
                    tp,
                    moe_tp,
                    moe_ep,
                    attn_dp,
                    dense_ffn_gemm_q,
                    shared_expert_gemm_q,
                    moe_q,
                    workload_dist,
                    cfg,
                )
            )

        # --- full_attention (QSA) layers ---
        if counts["full"] > 0:
            c = counts["full"]
            self.generation_ops.extend(self._hc_block_ops("generation", "generation_qsa_attn", c, cfg))
            self.generation_ops.extend(self._qsa_ops("generation", "generation_qsa", c, cfg))
            self.generation_ops.extend(
                [*([] if attn_ar_folded else [ops.CustomAllReduce("generation_qsa_ar", c, h, tp)])]
            )
            self.generation_ops.extend(self._hc_block_ops("generation", "generation_qsa_mlp", c, cfg))
            self.generation_ops.extend(
                self._ffn_generation_ops(
                    "generation_qsa",
                    c,
                    h,
                    tp,
                    moe_tp,
                    moe_ep,
                    attn_dp,
                    dense_ffn_gemm_q,
                    shared_expert_gemm_q,
                    moe_q,
                    workload_dist,
                    cfg,
                )
            )

        # --- n-gram PLE branches (draft stack carries none) ---
        n_ple = len(cfg.ple_layer_indices)
        if n_ple:
            self.generation_ops.extend(self._ple_ops("generation", "generation_ple", n_ple, cfg))

        # --- MTP draft stack: nextn x (QSA layer + MoE + 2 HC blocks + mixer) ---
        if self._nextn > 0:
            self.generation_ops.extend(self._draft_ops(cfg, workload_dist, h, moe_tp, moe_ep, attn_dp))

        self.generation_ops.extend(
            [
                ops.GEMM("generation_logits_gemm", 1 * sf, self._vocab_size // tp, h, common.GEMMQuantMode.bfloat16),
                ops.P2P("generation_p2p", (pp - 1) * sf, h, pp),
            ]
        )

    def _draft_ops(
        self,
        cfg: common.Qwen4ExpConfig,
        workload_dist: str,
        h: int,
        moe_tp: int,
        moe_ep: int,
        attn_dp: int,
    ) -> list:
        """MTP draft stack at ``nextn`` layers: one QSA+MoE layer and its inputs.

        The draft layer is full-attention (from ``text_config.mtp.layer_types``),
        so pricing it by scaling the mixed GDN/QSA decoder would move 1/12 of the
        QSA work onto GDN. ``fc_embedding``/``fc_hidden`` project the draft token
        embedding and the final manifold state into the draft hidden width; the
        model-level ``hyper_connection_mixer`` appears once per draft stack.

        Widths follow ``vllm/models/qwen4_exp/nvidia/mtp.py`` @v0.31.0:

        * both projections are ``ColumnParallelLinear(hidden, hidden,
          gather_output=True)``, so each rank owns ``hidden / tp`` output
          columns and the full ``hidden`` reduction width;
        * ``pre_fc_norm_hidden`` is ``GemmaRMSNorm(hidden * hc_count)`` on every
          rank — full width, no TP shard (mtp.py:229);
        * ``pre_fc_norm_embedding`` is ``GemmaRMSNorm(hidden)``.

        The ``gather_output=True`` all-gather the ``fc_*`` pair performs is not
        priced by any op; it is a constant per draft token and is left out
        rather than approximated.
        """
        c = float(self._nextn)
        tp = self.config.tp_size
        ops_list = [
            ops.GEMM(f"{_DRAFT_PREFIX}fc_embedding_gemm", c, h // tp, h, self._mtp_fc_gemm_quant_mode),
            ops.ElementWise(
                f"{_DRAFT_PREFIX}pre_fc_norm_hidden",
                c,
                2 * cfg.hc_count * h,
                2 * cfg.hc_count * h,
                0.8,
            ),
            ops.GEMM(f"{_DRAFT_PREFIX}fc_hidden_gemm", c, h // tp, h, self._mtp_fc_gemm_quant_mode),
            ops.ElementWise(f"{_DRAFT_PREFIX}pre_fc_norm_embedding", c, 2 * h, 2 * h, 0.8),
        ]
        # The draft stack's own residual mixer (weights live under ``mtp.``).
        ops_list.extend(self._hc_block_ops("generation", f"{_DRAFT_PREFIX}mixer", c, cfg))
        ops_list.extend(self._hc_block_ops("generation", f"{_DRAFT_PREFIX}attn", c, cfg))
        ops_list.extend(self._qsa_ops("generation", _DRAFT_PREFIX.rstrip("_"), c, cfg))
        ops_list.extend(self._hc_block_ops("generation", f"{_DRAFT_PREFIX}mlp", c, cfg))
        ops_list.extend(
            self._ffn_generation_ops(
                _DRAFT_PREFIX.rstrip("_"),
                c,
                h,
                tp,
                moe_tp,
                moe_ep,
                attn_dp,
                self._dense_ffn_gemm_quant_mode,
                self._shared_expert_gemm_quant_mode,
                self.config.moe_quant_mode,
                workload_dist,
                cfg,
            )
        )
        return ops_list
