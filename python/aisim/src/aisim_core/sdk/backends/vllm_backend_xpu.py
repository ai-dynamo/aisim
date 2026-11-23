# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""XPU-calibrated vLLM backend, with two overrides vs the base:
TTFT: prefill time x a fitted burst factor when the batch fits the running-batch limit, else
      plus a Little's-law wait for the excess -- two regimes vs the base's single factor.
TPOT: mixed prefill/decode steps counted per-request, instead of averaged in the base.
"""

import copy
import logging
import math
from dataclasses import replace

from aisim_core.sdk.backends.base_backend import _has_qualified_mamba
from aisim_core.sdk.backends.vllm_backend import VLLMBackend
from aisim_core.sdk.config import RuntimeConfig
from aisim_core.sdk.models import BaseModel
from aisim_core.sdk.perf_database import PerfDatabase
from aisim_core.sdk.step_estimate import MixedStepInput, StepEstimate

logger = logging.getLogger(__name__)

# TTFT burst factor (M==bs): 1 + A * log2(bs)^P * ((ctx/isl) * gen(bs)/gen(1))^R. BS and ctx/isl
# get separate exponents (the scheduler's prefill co-batch scales with BS more than ctx/isl);
# gen(bs)/gen(1) is the decode-growth (decode-steal) signal. Fit to measured benchmarks.
BURST_A = 0.45
BURST_BS_EXP = 1.10
BURST_CTX_EXP = 0.50

# gpt-oss MoE: the collected perf data over-counts active experts ~2x vs the real router,
# so replace it with a measured saturating law A*(1-(1-topk/num_experts)^(nt^rho)).
GPT_OSS_ARCH = "GptOssForCausalLM"
GPT_OSS_MOE_A = 34.09
GPT_OSS_MOE_RHO = 0.65


class VLLMXPUBackend(VLLMBackend):
    """XPU-calibrated vLLM backend: TTFT + mixed-step TPOT."""

    # ====================== TTFT (XPU-calibrated) ======================

    def _compute_ttft(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        b: int,
        isl: int,
        osl: int,
        ctx_tokens: int,
        prefix: int,
        *,
        prefill_step_ms: float,
        genonly_step_latency_ms: float,
        encoder_latency_ms: float,
        steps_to_finish_ctx: float,
        decode_iterations: float,
        prefill_total_ms: float | None = None,
    ) -> float:
        """encoder + own_prefill, x burst factor (eff_bs==b) or + admission wait (eff_bs<b); raw for b<=1."""
        if b == 1 and prefill_total_ms is not None:
            # run_agg already summed the actual sequential chunk prices.
            return encoder_latency_ms + prefill_total_ms + self._prefill_dispatch_overhead_ms(model)
        d = 0 if b <= 1 else self._mix_step_gen_tokens(b, ctx_tokens, isl, decode_iterations)
        # Native budget coordinates: op-level includes the cached prefix, while
        # whole-model FPM uses new tokens. Neither recomputes the cached prefix.
        request_tokens = isl
        is_fpm = getattr(model, "forward_model", "op_level") == "fpm"
        if is_fpm:
            request_tokens = max(1, isl - prefix)
        # Cap demand after run_mixed reserves decode slots from the total
        # scheduler budget. The burst fit still reads the capped demand below.
        prefill_ctx = min(ctx_tokens, max(1, b) * request_tokens)
        step = self.run_mixed(
            model,
            database,
            runtime_config,
            MixedStepInput(context_tokens=ctx_tokens if b > 1 else prefill_ctx, num_decode_requests=d),
        )
        exact_text_prefill = (
            b > 1 and not is_fpm and isl == runtime_config.isl and (prefix > 0 or _has_qualified_mamba(model))
        )
        if exact_text_prefill:
            # A packed prompt budget is not a multiplier for already-priced
            # uncached work, especially Mamba's per-chunk recurrent-state cost.
            own_prefill_ms = self._own_cached_prefill_ms(step, model, database, runtime_config, isl, prefix)
        else:
            own_prefill_ms = self._own_prefill_ms(step, model, request_tokens)
        if b <= 1:
            logger.debug(f"ttft(xpu): own_prefill={own_prefill_ms:.2f}ms b<=1 (no queuing)")
            return encoder_latency_ms + own_prefill_ms

        eff_bs = self._effective_decode_bs(model, database, runtime_config, b, ctx_tokens)
        if eff_bs < b:
            # admission-limited: b-eff_bs requests queue for a slot (Little's law).
            tpot = self._decode_tpot_ms(
                step.latency_ms,
                genonly_step_latency_ms,
                eff_bs,
                isl,
                osl,
                ctx_tokens,
                steps_to_finish_ctx,
                all_mix=steps_to_finish_ctx >= decode_iterations,
            )
            ttft = own_prefill_ms + (b - eff_bs) / eff_bs * (own_prefill_ms + osl * tpot)
        else:
            # burst: all b arrive together; decode-growth gen(bs)/gen(1) captures decode-steal.
            gen_bs = self._get_genonly_step_latency(model, database, runtime_config, b, isl, osl)[0]
            gen_1 = self._get_genonly_step_latency(model, database, runtime_config, 1, isl, osl)[0]
            growth = gen_bs / gen_1 if gen_1 > 0 else 1.0

            factor = 1.0 + BURST_A * math.log2(b) ** BURST_BS_EXP * ((prefill_ctx / isl) * growth) ** BURST_CTX_EXP
            ttft = own_prefill_ms * min(factor, float(b))
        regime = "admission" if eff_bs < b else "burst"
        logger.debug(f"ttft(xpu): own_prefill={own_prefill_ms:.2f}ms eff_bs={eff_bs} {regime} ttft={ttft:.2f}ms")
        return encoder_latency_ms + ttft

    def _own_cached_prefill_ms(
        self,
        step: StepEstimate,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        isl: int,
        prefix: int,
    ) -> float:
        """Price one text request's uncached chunks, retaining the mixed step's decoders.

        The native op-level budget includes each chunk's already processed
        prefix. Rust removes that prefix and prices the actual new tokens and
        one recurrent-state charge per chunk, including a short final chunk.
        """
        if step.context_tokens == isl:
            return step.latency_ms + self._prefill_dispatch_overhead_ms(model)
        chunk_runtime = copy.copy(runtime_config)
        chunk_runtime.batch_size = 1
        total_ms = 0.0
        for past in range(prefix, isl, step.context_tokens):
            chunk_runtime.prefix = past
            chunk_runtime.isl = min(isl, past + step.context_tokens)
            # These shapes are already scheduled: bypass the aggregate XPU
            # budget conversion so decoders do not shrink the chunk again.
            chunk = super().run_mixed(
                model,
                database,
                chunk_runtime,
                MixedStepInput(chunk_runtime.isl, step.num_decode_requests),
            )
            # Other requests' decode KV does not change with this request's
            # prefill progress. Reuse its original native decode component.
            total_ms += (
                chunk.component_latency_ms["shared_non_attention"]
                + chunk.component_latency_ms["context_attention"]
                + step.component_latency_ms["decode_attention"]
            )
        return total_ms + self._prefill_dispatch_overhead_ms(model)

    def _own_prefill_ms(self, step: StepEstimate, model: BaseModel, isl: int) -> float:
        """One request's chunked prefill from the mixed step: full attention (undo run_mixed's
        /ceil) + gemm/decode scaled by isl/ctx_eff, plus per-request dispatch. The empirical burst
        factor absorbs the scheduler's real co-batch (load/BS/CPU-dependent)."""
        ctx_eff = step.context_tokens
        ctx_attn = step.per_op_latency_ms.get("context_attention (scaled)", 0.0)
        full_attn = ctx_attn * math.ceil(isl / ctx_eff)
        scale = isl / ctx_eff
        if getattr(model, "forward_model", "op_level") == "fpm" and ctx_eff < isl:
            # A whole-model FPM prices the mean of one request's chunks: count every chunk.
            scale = math.ceil(scale)
        return full_attn + (step.latency_ms - ctx_attn) * scale + self._prefill_dispatch_overhead_ms(model)

    @staticmethod
    def _decode_tpot_ms(mix_lat, genonly_lat, b, isl, osl, ctx_tokens, steps_to_finish_ctx, all_mix=False) -> float:
        """Per-token decode time: the mix+genonly blend a request sees (mirrors _compute_tpot)."""
        if osl <= 0:
            return 0.0
        if all_mix:
            return mix_lat
        prefillers = max(1.0, ctx_tokens / isl)
        nmix = min(steps_to_finish_ctx * max(0.0, b - prefillers) / b, float(osl))
        return (mix_lat * nmix + genonly_lat * (osl - nmix)) / osl

    @staticmethod
    def _prefill_dispatch_overhead_ms(model: BaseModel) -> float:
        """Per-request dispatch overhead: 0.3/layer (base uses 0.8)."""
        return model._num_layers * 0.3

    # =============== EFFECTIVE DECODE BS (KV + prefill limited) ===============

    def _resolve_agg_kwargs(self, kwargs, isl, osl, backend_version=None):
        """Stash util + ctx_tokens for _max_kv_slots (runs before the step methods)."""
        extra = super()._resolve_agg_kwargs(kwargs, isl=isl, osl=osl, backend_version=backend_version)
        self._agg_free_gpu_frac = extra.get("free_gpu_memory_fraction")
        self._agg_ctx_tokens = kwargs.get("ctx_tokens")
        return extra

    def run_agg(self, model, database, runtime_config, **kwargs):
        # Scope the budget/occupancy modeling and its stash to this call, so direct
        # run_mixed callers keep the step they scheduled.
        self._in_agg = True
        try:
            return super().run_agg(model, database, runtime_config, **kwargs)
        finally:
            self._in_agg = False
            self._agg_free_gpu_frac = self._agg_ctx_tokens = None

    def _max_kv_slots(
        self, model: BaseModel, database: PerfDatabase, isl: int, osl: int, ctx_tokens: int | None = None
    ) -> int:
        """KV-cache capacity cap on concurrent decoders: how many sequences' KV fit in the memory
        left after the non-KV footprint (weights + activation + nccl + others). Binds only on
        KV-tight configs (small GPU / long isl+osl / high bs); else Little's law caps first."""
        util = (
            getattr(self, "_agg_free_gpu_frac", None)
            or self.get_default_free_gpu_memory_fraction(database.version)
            or 0.9
        )
        mem_cap = database.system_spec["gpu"]["mem_capacity"]
        ctx = int(getattr(self, "_agg_ctx_tokens", None) or ctx_tokens or 1)

        mem = self._get_memory_usage(model, database, batch_size=1, beam_width=1, isl=1, osl=1, num_tokens=ctx)
        non_kv = (mem["weights"] + mem["activations"] + mem["nccl"] + mem["others"]) * (1 << 30)

        kv_budget = mem_cap * util - non_kv
        if kv_budget <= 0:
            return 1
        return max(1, int(kv_budget / model.get_kvcache_bytes_per_sequence(isl + osl)))

    def _effective_decode_bs(
        self, model: BaseModel, database: PerfDatabase, runtime_config: RuntimeConfig, b: int, ctx_tokens: int
    ) -> int:
        """How many requests actually sit in the decode phase at once, set by whichever of
        three limits binds first:
          * b         -- can't decode more sequences than were requested;
          * KV slots  -- the KV cache physically holds only so many sequences;
          * Little's law ctx*osl/(isl+osl) -- prefill throughput caps steady-state occupancy:
            a new request enters decode only as one finishes prefill, so occupancy settles at
            admission_rate x decode_residency. This is what usually binds on KV-roomy configs.
        b<=1 or degenerate isl/osl: no contention, so the batch decodes as requested."""
        isl = int(runtime_config.isl or 0) + self._visual_context_tokens(model, runtime_config)
        osl = int(runtime_config.osl or 0)
        if b <= 1 or isl <= 0 or osl <= 0:
            return b
        little = max(1, round(ctx_tokens * osl / (isl + osl)))
        return max(1, min(b, self._max_kv_slots(model, database, isl, osl), little))

    # ====================== TPOT (XPU-calibrated) ======================

    def _compute_tpot(
        self,
        *,
        b,
        isl,
        osl,
        ctx_tokens,
        num_mix_steps,
        num_genonly_steps,
        num_mix_steps_for_tpot_calc,
        mix_step_latency_ms,
        genonly_step_latency_ms,
        model=None,
        database=None,
        runtime_config=None,
    ):
        """Per-request TPOT: a request sees (eff_bs - prefillers)/eff_bs of the mix steps; base for b<=1."""
        if osl <= 1 or b <= 1:
            return super()._compute_tpot(
                b=b,
                isl=isl,
                osl=osl,
                ctx_tokens=ctx_tokens,
                num_mix_steps=num_mix_steps,
                num_genonly_steps=num_genonly_steps,
                num_mix_steps_for_tpot_calc=num_mix_steps_for_tpot_calc,
                mix_step_latency_ms=mix_step_latency_ms,
                genonly_step_latency_ms=genonly_step_latency_ms,
            )

        # prefill spans every decode iteration: base has no decode-only steps, so every token is in a mix step
        if num_genonly_steps == 0:
            return mix_step_latency_ms
        # recomputed here (not handed via instance state) so a reused backend can't leak a stale value
        eff_bs = self._effective_decode_bs(model, database, runtime_config, b, ctx_tokens)
        prefillers_per_step = max(1.0, ctx_tokens / isl)
        # cap at osl: a request decodes in at most osl steps, so ngen_eff stays >= 0
        nmix_eff = min(num_mix_steps * max(0.0, eff_bs - prefillers_per_step) / eff_bs, float(osl))
        ngen_eff = osl - nmix_eff
        return (mix_step_latency_ms * nmix_eff + genonly_step_latency_ms * ngen_eff) / osl

    # ==================== GEN-ONLY MoE (gpt-oss) =======================

    @staticmethod
    def _gpt_oss_moe_ms(gen_tokens, topk, num_experts):
        # gpt-oss decode MoE (ms/step): saturating active-experts law, measured-calibrated.
        return GPT_OSS_MOE_A * (1.0 - (1.0 - topk / num_experts) ** (gen_tokens**GPT_OSS_MOE_RHO))

    def _gpt_oss_moe_correction(self, model, per_op_latency_ms, gen_tokens):
        """(delta_ms, corrected generation_moe ms) replacing the over-projected generation_moe
        with the measured law, or None when it does not apply: non-gpt-oss, bs<=1, or MoE split
        across GPUs (moe_tp/ep > 1) where the collected per-GPU slice is already correct."""
        if model.architecture != GPT_OSS_ARCH or gen_tokens <= 1 or "generation_moe" not in per_op_latency_ms:
            return None
        moe_op = next((o for o in model.generation_ops if o._name == "generation_moe"), None)
        if moe_op is None or moe_op._moe_tp_size * moe_op._moe_ep_size > 1:
            return None
        target = self._gpt_oss_moe_ms(gen_tokens, moe_op._topk, moe_op._num_experts)
        return target - per_op_latency_ms["generation_moe"], target

    def _get_genonly_step_latency(self, model, database, runtime_config, gen_tokens, isl, osl):
        # pure decode: at most kv_slots requests reside in KV (no prefill competes).
        if gen_tokens > 1:
            gen_tokens = min(gen_tokens, self._max_kv_slots(model, database, isl, osl))
        lat, energy, per_ops, per_src, moe_fallbacks = super()._get_genonly_step_latency(
            model, database, runtime_config, gen_tokens, isl, osl
        )
        corr = self._gpt_oss_moe_correction(model, per_ops, gen_tokens)
        if corr is None:
            return lat, energy, per_ops, per_src, moe_fallbacks
        delta, target = corr
        # Python-fitted value, not the collected silicon measurement: mark provenance.
        return (
            lat + delta,
            energy,
            {**per_ops, "generation_moe": target},
            {**per_src, "generation_moe": "estimated"},
            moe_fallbacks,
        )

    def _get_genonly_step_estimate(self, model, database, runtime_config, gen_tokens, isl, osl):
        # run_agg's decode/TPOT seam (StepEstimate); mirror the _get_genonly_step_latency correction.
        if gen_tokens > 1:
            gen_tokens = min(gen_tokens, self._max_kv_slots(model, database, isl, osl))
        est = super()._get_genonly_step_estimate(model, database, runtime_config, gen_tokens, isl, osl)
        corr = self._gpt_oss_moe_correction(model, est.per_op_latency_ms, gen_tokens)
        if corr is None:
            return est
        delta, target = corr
        return replace(
            est,
            latency_ms=est.latency_ms + delta,
            per_op_latency_ms={**est.per_op_latency_ms, "generation_moe": target},
            per_op_source={**est.per_op_source, "generation_moe": "estimated"},
        )

    # ==================== MIX-STEP LATENCY (shared) ====================

    def _run_agg_mixed_step(self, model, database, runtime_config, step: MixedStepInput) -> StepEstimate:
        # XPU consumes a total scheduler budget: reserve decode slots before
        # translating the remaining prompt budget to FPM fresh tokens.
        return self.run_mixed(model, database, runtime_config, step)

    def run_mixed(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        step: MixedStepInput,
    ) -> StepEstimate:
        """ctx-semantics change vs the base backend: our step.context_tokens (the `--ctx-tokens`
        arg) is vLLM's --max-num-batched-tokens -- the TOTAL per-step budget shared by prefill AND
        decode. The base bills it as PREFILL tokens with decode stacked on top, so here we reserve
        the decode slots out of the budget and hand the base only the leftover prefill tokens.

        Agg-sweep only: a direct caller's step is already scheduled, so it is estimated as given."""
        if not getattr(self, "_in_agg", False):
            return super().run_mixed(model, database, runtime_config, step)
        # Concurrent decoders = min(requested, Little's-law cap, KV slots).
        isl = int(runtime_config.isl or 0) + self._visual_context_tokens(model, runtime_config)
        osl = int(runtime_config.osl or 0)
        if isl > 0 and osl > 0:
            steady_running = max(1, round(step.context_tokens * osl / (isl + osl)))
            kv_slots = self._max_kv_slots(model, database, isl, osl, ctx_tokens=step.context_tokens)
            cap = min(steady_running, kv_slots)
            if step.num_decode_requests > cap:
                step = replace(step, num_decode_requests=cap)

        # total budget -> prefill tokens: subtract the reserved decode slots.
        request_tokens = isl
        if getattr(model, "forward_model", "op_level") == "fpm":
            request_tokens = max(1, isl - int(runtime_config.prefix or 0))
        demand = max(1, int(getattr(runtime_config, "batch_size", 1) or 1)) * request_tokens
        budget = max(1, step.context_tokens - step.num_decode_requests * (model._nextn + 1))
        if getattr(model, "forward_model", "op_level") == "fpm" and runtime_config.batch_size > 1:
            budget = self._fpm_aggregate_prefill_tokens(
                budget,
                isl,
                int(runtime_config.prefix or 0),
                max(1, runtime_config.batch_size - step.num_decode_requests),
            )
        step = replace(step, context_tokens=max(1, min(demand, budget)))
        step = self._schedule_qualified_mamba_prefill(model, runtime_config, step)
        return super().run_mixed(model, database, runtime_config, step)
