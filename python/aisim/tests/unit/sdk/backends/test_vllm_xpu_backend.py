# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XPU backend selection and XPU-calibrated overrides (VLLMXPUBackend)."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aisim_core.sdk.backends.factory import get_backend
from aisim_core.sdk.backends.vllm_backend import VLLMBackend
from aisim_core.sdk.backends.vllm_backend_xpu import (
    BURST_A,
    BURST_BS_EXP,
    BURST_CTX_EXP,
    GPT_OSS_ARCH,
    VLLMXPUBackend,
)
from aisim_core.sdk.perf_database import is_xpu_system, is_xpu_system_spec
from aisim_core.sdk.step_estimate import MixedStepInput, StepEstimate

pytestmark = pytest.mark.unit

# Arbitrary base generation_moe latency for fixtures; tests assert self-consistency
# (delta == law - base), never a specific measured value.
_FAKE_MOE_MS = 10.0


# ---------------------------------------------------------------------------
# System classification + backend routing
# ---------------------------------------------------------------------------


def test_is_xpu_system_b60_true() -> None:
    # b60's system spec has no sm_version → XPU.
    assert is_xpu_system("b60") is True


def test_is_xpu_system_nvidia_false() -> None:
    # NVIDIA specs carry sm_version → not XPU.
    assert is_xpu_system("h200_sxm") is False


def test_is_xpu_system_none_false() -> None:
    assert is_xpu_system(None) is False


def test_is_xpu_system_unknown_raises() -> None:
    # An unresolvable spec must fail loud, not silently classify as XPU.
    with pytest.raises(ValueError):
        is_xpu_system("nonexistent_typo_system")


def test_is_xpu_system_spec_classifies_loaded_spec() -> None:
    # Classify the already-loaded spec (same roots the database used), no re-load.
    assert is_xpu_system_spec({"gpu": {"mem_capacity": 1}}) is True  # no sm_version
    assert is_xpu_system_spec({"gpu": {"sm_version": 90}}) is False
    assert is_xpu_system_spec(None) is False


def test_get_backend_prefers_passed_system_spec_over_name() -> None:
    # When a loaded spec is passed, it drives classification (not a name re-load).
    assert isinstance(get_backend("vllm", "ignored", system_spec={"gpu": {"mem_capacity": 1}}), VLLMXPUBackend)
    nv = get_backend("vllm", "b60", system_spec={"gpu": {"sm_version": 90}})
    assert type(nv) is VLLMBackend


def test_get_backend_vllm_on_xpu_routes_to_xpu_backend() -> None:
    assert isinstance(get_backend("vllm", "b60"), VLLMXPUBackend)


def test_get_backend_vllm_on_nvidia_uses_base_backend() -> None:
    backend = get_backend("vllm", "h200_sxm")
    assert isinstance(backend, VLLMBackend)
    assert not isinstance(backend, VLLMXPUBackend)


def test_get_backend_vllm_without_system_uses_base_backend() -> None:
    # No system name (disagg/AFD/encoder paths) → generic backend, never XPU.
    assert type(get_backend("vllm")) is VLLMBackend


@pytest.mark.parametrize("system_name", [None, "h200_sxm"])
def test_get_backend_vllm_no_xpu_system_is_exact_base(system_name) -> None:
    # No system or an NVIDIA system, type is base VLLMBackend
    assert type(get_backend("vllm", system_name)) is VLLMBackend


def test_vllm_xpu_backend_is_a_vllm_backend() -> None:
    # XPU backend inherits every non-overridden behavior from the base vLLM backend.
    assert issubclass(VLLMXPUBackend, VLLMBackend)


# ---------------------------------------------------------------------------
# gpt-oss decode-MoE correction: gating + wiring (not the empirical values)
# ---------------------------------------------------------------------------


def _fake_gpt_oss_model(*, moe_tp=1, moe_ep=1, topk=4, num_experts=32, arch=GPT_OSS_ARCH):
    moe_op = MagicMock()
    moe_op._name = "generation_moe"
    moe_op._moe_tp_size = moe_tp
    moe_op._moe_ep_size = moe_ep
    moe_op._topk = topk
    moe_op._num_experts = num_experts
    model = MagicMock()
    model.architecture = arch
    model.generation_ops = [moe_op]
    return model


def test_gpt_oss_moe_correction_applies_when_unsharded() -> None:
    backend = VLLMXPUBackend()
    model = _fake_gpt_oss_model(moe_tp=1, moe_ep=1)
    corr = backend._gpt_oss_moe_correction(model, {"generation_moe": _FAKE_MOE_MS}, gen_tokens=4)
    assert corr is not None
    delta, target = corr
    assert target == pytest.approx(backend._gpt_oss_moe_ms(4, 4, 32))
    assert delta == pytest.approx(target - _FAKE_MOE_MS)


def test_gpt_oss_moe_correction_skips_when_sharded() -> None:
    # moe_tp * moe_ep > 1: the collected per-GPU slice is already correct.
    backend = VLLMXPUBackend()
    for moe_tp, moe_ep in [(2, 1), (1, 4), (2, 2)]:
        model = _fake_gpt_oss_model(moe_tp=moe_tp, moe_ep=moe_ep)
        assert backend._gpt_oss_moe_correction(model, {"generation_moe": _FAKE_MOE_MS}, gen_tokens=4) is None


def test_gpt_oss_moe_correction_skips_non_gpt_oss() -> None:
    backend = VLLMXPUBackend()
    model = _fake_gpt_oss_model(arch="LlamaForCausalLM")
    assert backend._gpt_oss_moe_correction(model, {"generation_moe": _FAKE_MOE_MS}, gen_tokens=4) is None


def test_gpt_oss_moe_correction_skips_single_token() -> None:
    # bs<=1 decode: the law is a no-op (matches the collected router).
    backend = VLLMXPUBackend()
    model = _fake_gpt_oss_model()
    assert backend._gpt_oss_moe_correction(model, {"generation_moe": _FAKE_MOE_MS}, gen_tokens=1) is None


def test_gpt_oss_moe_correction_skips_without_moe_op_in_per_ops() -> None:
    backend = VLLMXPUBackend()
    model = _fake_gpt_oss_model()
    assert backend._gpt_oss_moe_correction(model, {"generation_attention": 1.0}, gen_tokens=4) is None


# ---------------------------------------------------------------------------
# Decode-step seam applies the correction (run_agg's TPOT path)
# ---------------------------------------------------------------------------


def test_get_genonly_step_estimate_applies_gpt_oss_correction(monkeypatch) -> None:
    backend = VLLMXPUBackend()
    model = _fake_gpt_oss_model(moe_tp=1, moe_ep=1)

    def fake_super(self, model_, database_, runtime_config_, gen_tokens, isl, osl):
        return StepEstimate(latency_ms=100.0, energy_wms=0.0, per_op_latency_ms={"generation_moe": _FAKE_MOE_MS})

    monkeypatch.setattr(VLLMBackend, "_get_genonly_step_estimate", fake_super)
    monkeypatch.setattr(VLLMXPUBackend, "_max_kv_slots", lambda self, *a, **k: 10_000)

    est = backend._get_genonly_step_estimate(model, MagicMock(), MagicMock(), 4, 1024, 1024)
    target = backend._gpt_oss_moe_ms(4, 4, 32)
    assert est.per_op_latency_ms["generation_moe"] == pytest.approx(target)
    assert est.latency_ms == pytest.approx(100.0 + (target - _FAKE_MOE_MS))
    # Fitted value, not the collected measurement: provenance must say so.
    assert est.per_op_source["generation_moe"] == "estimated"


def test_get_genonly_step_estimate_untouched_for_sharded(monkeypatch) -> None:
    backend = VLLMXPUBackend()
    model = _fake_gpt_oss_model(moe_tp=2, moe_ep=1)

    def fake_super(self, model_, database_, runtime_config_, gen_tokens, isl, osl):
        return StepEstimate(latency_ms=100.0, energy_wms=0.0, per_op_latency_ms={"generation_moe": _FAKE_MOE_MS})

    monkeypatch.setattr(VLLMBackend, "_get_genonly_step_estimate", fake_super)
    monkeypatch.setattr(VLLMXPUBackend, "_max_kv_slots", lambda self, *a, **k: 10_000)

    est = backend._get_genonly_step_estimate(model, MagicMock(), MagicMock(), 4, 1024, 1024)
    # Sharded: no correction — collected value and step latency unchanged.
    assert est.per_op_latency_ms["generation_moe"] == pytest.approx(_FAKE_MOE_MS)
    assert est.latency_ms == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# TTFT (XPU-calibrated): the three regimes of _compute_ttft
# ---------------------------------------------------------------------------


def _ttft_kwargs(**overrides):
    kwargs = dict(
        prefill_step_ms=0.0,
        genonly_step_latency_ms=1.0,
        encoder_latency_ms=0.0,
        steps_to_finish_ctx=3.0,
        decode_iterations=5.0,
    )
    kwargs.update(overrides)
    return kwargs


def _patch_ttft_deps(monkeypatch, *, own_prefill=10.0, eff_bs=None, gen=2.0, tpot=1.5):
    monkeypatch.setattr(VLLMXPUBackend, "_mix_step_gen_tokens", lambda self, *a, **k: 5, raising=False)
    monkeypatch.setattr(
        VLLMXPUBackend, "run_mixed", lambda self, *a, **k: StepEstimate(latency_ms=40.0, energy_wms=0.0)
    )
    monkeypatch.setattr(VLLMXPUBackend, "_own_prefill_ms", lambda self, *a, **k: own_prefill)
    monkeypatch.setattr(VLLMXPUBackend, "_get_genonly_step_latency", lambda self, *a, **k: (gen, 0, {}, {}, ()))
    monkeypatch.setattr(VLLMXPUBackend, "_decode_tpot_ms", staticmethod(lambda *a, **k: tpot))
    if eff_bs is not None:
        monkeypatch.setattr(VLLMXPUBackend, "_effective_decode_bs", lambda self, *a, **k: eff_bs)


def test_compute_ttft_single_request_is_raw_prefill(monkeypatch) -> None:
    # b<=1: no queuing, just encoder + own prefill (no burst/admission factor).
    backend = VLLMXPUBackend()
    _patch_ttft_deps(monkeypatch, own_prefill=10.0)
    ttft = backend._compute_ttft(
        MagicMock(), MagicMock(), MagicMock(), 1, 100, 10, 100, 0, **_ttft_kwargs(encoder_latency_ms=2.0)
    )
    assert ttft == pytest.approx(12.0)


def test_compute_ttft_burst_applies_capped_factor(monkeypatch) -> None:
    # eff_bs == b: prefill x burst factor, capped at b.
    backend = VLLMXPUBackend()
    _patch_ttft_deps(monkeypatch, own_prefill=10.0, eff_bs=4, gen=2.0)
    b, isl, ctx = 4, 100, 400
    prefill_ctx = min(ctx, b * isl)  # 400
    factor = 1.0 + BURST_A * math.log2(b) ** BURST_BS_EXP * ((prefill_ctx / isl) * 1.0) ** BURST_CTX_EXP
    ttft = backend._compute_ttft(MagicMock(), MagicMock(), MagicMock(), b, isl, 10, ctx, 0, **_ttft_kwargs())
    assert ttft == pytest.approx(10.0 * min(factor, float(b)))


def test_compute_ttft_admission_adds_queue_wait(monkeypatch) -> None:
    # eff_bs < b: own prefill + Little's-law wait for the (b-eff_bs) queued requests.
    backend = VLLMXPUBackend()
    _patch_ttft_deps(monkeypatch, own_prefill=10.0, eff_bs=2, tpot=1.5)
    b, osl = 4, 10
    ttft = backend._compute_ttft(MagicMock(), MagicMock(), MagicMock(), b, 100, osl, 400, 0, **_ttft_kwargs())
    expected = 10.0 + (b - 2) / 2 * (10.0 + osl * 1.5)
    assert ttft == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TPOT (XPU-calibrated): delegation vs the mixed-step blend
# ---------------------------------------------------------------------------


def _tpot_kwargs(**overrides):
    kwargs = dict(
        num_mix_steps=5,
        num_genonly_steps=0,
        num_mix_steps_for_tpot_calc=0,
        mix_step_latency_ms=4.0,
        genonly_step_latency_ms=1.0,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize("b,osl", [(1, 10), (8, 1)])
def test_compute_tpot_delegates_to_base_for_trivial_batch(monkeypatch, b, osl) -> None:
    # b<=1 or osl<=1: no mixed-step accounting, defer to the base backend.
    backend = VLLMXPUBackend()
    monkeypatch.setattr(VLLMBackend, "_compute_tpot", lambda self, **k: 99.0)
    got = backend._compute_tpot(b=b, isl=100, osl=osl, ctx_tokens=200, model=MagicMock(), **_tpot_kwargs())
    assert got == 99.0


def test_compute_tpot_blends_mix_and_genonly_steps(monkeypatch) -> None:
    # A request sees (eff_bs - prefillers)/eff_bs of the mix steps; rest are gen-only.
    backend = VLLMXPUBackend()
    monkeypatch.setattr(VLLMXPUBackend, "_effective_decode_bs", lambda self, *a, **k: 8)
    b, isl, osl, ctx = 8, 100, 10, 200
    got = backend._compute_tpot(
        b=b,
        isl=isl,
        osl=osl,
        ctx_tokens=ctx,
        model=MagicMock(),
        database=MagicMock(),
        runtime_config=MagicMock(),
        # base: osl=10 decode iterations, 5 mix steps -> 5 decode-only steps
        **_tpot_kwargs(num_mix_steps=5, num_genonly_steps=5, mix_step_latency_ms=4.0, genonly_step_latency_ms=1.0),
    )
    prefillers = max(1.0, ctx / isl)  # 2.0
    nmix_eff = min(5 * (8 - prefillers) / 8, float(osl))  # 3.75
    expected = (4.0 * nmix_eff + 1.0 * (osl - nmix_eff)) / osl
    assert got == pytest.approx(expected)


def test_compute_tpot_all_mix_when_base_has_no_decode_only_steps(monkeypatch) -> None:
    # Prefill spans every decode iteration: every token is in a mix step, so TPOT is the mix step.
    backend = VLLMXPUBackend()
    monkeypatch.setattr(VLLMXPUBackend, "_effective_decode_bs", lambda self, *a, **k: 1)
    got = backend._compute_tpot(
        b=2,
        isl=1024,
        osl=2,
        ctx_tokens=512,
        model=MagicMock(),
        database=MagicMock(),
        runtime_config=MagicMock(),
        **_tpot_kwargs(num_mix_steps=4, num_genonly_steps=0, mix_step_latency_ms=15.0, genonly_step_latency_ms=0.0),
    )
    assert got == pytest.approx(15.0)


def test_decode_tpot_all_mix_returns_mix_step() -> None:
    # Same rule for the TTFT admission wait.
    assert VLLMXPUBackend._decode_tpot_ms(15.0, 0.0, 1, 1024, 2, 512, 4, all_mix=True) == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# KV-slot cap and run_mixed budget split
# ---------------------------------------------------------------------------


def test_max_kv_slots_divides_kv_budget_by_per_sequence_bytes(monkeypatch) -> None:
    backend = VLLMXPUBackend()
    backend._agg_free_gpu_frac = 0.5
    backend._agg_ctx_tokens = 1
    gib = 1 << 30
    database = SimpleNamespace(system_spec={"gpu": {"mem_capacity": 20 * gib}}, version="0.28.0")
    monkeypatch.setattr(
        backend, "_get_memory_usage", lambda *a, **k: {"weights": 1.0, "activations": 0.5, "nccl": 0.5, "others": 0.0}
    )
    model = MagicMock()
    model.get_kvcache_bytes_per_sequence.return_value = gib
    # kv_budget = 20*0.5 - 2 = 8 GiB; 8 GiB / 1 GiB-per-seq = 8 slots.
    assert backend._max_kv_slots(model, database, isl=512, osl=512) == 8


def test_max_kv_slots_returns_one_when_budget_exhausted(monkeypatch) -> None:
    backend = VLLMXPUBackend()
    backend._agg_free_gpu_frac = 0.1
    backend._agg_ctx_tokens = 1
    gib = 1 << 30
    database = SimpleNamespace(system_spec={"gpu": {"mem_capacity": 4 * gib}}, version="0.28.0")
    # non-KV footprint exceeds mem*util -> no room for KV -> at least 1.
    monkeypatch.setattr(
        backend, "_get_memory_usage", lambda *a, **k: {"weights": 10.0, "activations": 0.0, "nccl": 0.0, "others": 0.0}
    )
    model = MagicMock()
    model.get_kvcache_bytes_per_sequence.return_value = gib
    assert backend._max_kv_slots(model, database, isl=512, osl=512) == 1


def _capture_mixed_step(monkeypatch, captured):
    """Mirror base run_agg's own `self.run_mixed(...)` call and capture what the base receives."""
    monkeypatch.setattr(VLLMXPUBackend, "_visual_context_tokens", lambda self, *a, **k: 0)
    monkeypatch.setattr(VLLMXPUBackend, "_max_kv_slots", lambda self, *a, **k: 3)
    monkeypatch.setattr(VLLMBackend, "run_mixed", lambda self, m, d, rc, step: captured.setdefault("step", step))
    monkeypatch.setattr(
        VLLMBackend,
        "run_agg",
        lambda self, model, database, runtime_config, **kw: self.run_mixed(model, database, runtime_config, kw["step"]),
    )


def test_agg_run_mixed_reserves_decode_slots_from_the_budget(monkeypatch) -> None:
    # Agg sweep: ctx_tokens is the TOTAL per-step budget, so cap decoders at min(Little, KV)
    # and hand the base only the leftover prefill tokens.
    backend = VLLMXPUBackend()
    captured = {}
    _capture_mixed_step(monkeypatch, captured)
    rc = SimpleNamespace(isl=100, osl=100, batch_size=16)
    step = MixedStepInput(context_tokens=1000, num_decode_requests=50)
    backend.run_agg(MagicMock(), MagicMock(), rc, step=step)
    # steady_running = round(1000*100/200)=500; cap = min(500, 3) = 3.
    assert captured["step"].num_decode_requests == 3
    assert captured["step"].context_tokens == 1000 - 3


def test_direct_run_mixed_estimates_the_scheduled_step_as_given(monkeypatch) -> None:
    # Public API: InferenceSession hands over already-scheduled work, so neither the
    # budget conversion nor the occupancy cap may rewrite the caller's counts.
    from aisim.sdk.inference_session import InferenceSession

    backend = VLLMXPUBackend()
    captured = {}
    _capture_mixed_step(monkeypatch, captured)
    rc = SimpleNamespace(isl=100, osl=100)
    step = MixedStepInput(context_tokens=1000, num_decode_requests=50)
    InferenceSession(model=MagicMock(), database=MagicMock(), backend=backend).run_mixed(rc, step)
    assert captured["step"] == step


def test_prefill_heavy_boundary_prices_decode_steps() -> None:
    # ceil(isl*b/ctx) >= osl: base prices no decode-only steps (0 ms); the XPU blend must still price them.
    from aisim.legacy_cli.api import cli_estimate

    r = cli_estimate(
        model_path="Qwen/Qwen3-0.6B",
        system_name="b60",
        backend_name="vllm",
        backend_version="0.28.0",
        batch_size=2,
        isl=1024,
        osl=2,
        ctx_tokens=512,
    )
    assert r.tpot > 0
    assert r.request_latency > r.ttft  # decode time included


@pytest.mark.parametrize("raises", [False, True])
def test_run_agg_scopes_kv_slot_state_to_the_call(monkeypatch, raises) -> None:
    # The stashed util/ctx must not outlive run_agg (even on error), or later run_mixed calls read stale values.
    backend = VLLMXPUBackend()

    def fake_base_run_agg(self, model, database, runtime_config, **kwargs):
        self._resolve_agg_kwargs(kwargs, isl=1024, osl=128, backend_version="0.28.0")
        assert self._agg_ctx_tokens == 4096  # visible while the call runs
        if raises:
            raise RuntimeError("boom")
        return "summary"

    monkeypatch.setattr(VLLMBackend, "run_agg", fake_base_run_agg)
    if raises:
        with pytest.raises(RuntimeError):
            backend.run_agg(MagicMock(), MagicMock(), MagicMock(), ctx_tokens=4096, free_gpu_memory_fraction=0.5)
    else:
        assert backend.run_agg(MagicMock(), MagicMock(), MagicMock(), ctx_tokens=4096, free_gpu_memory_fraction=0.5)
    assert backend._agg_ctx_tokens is None and backend._agg_free_gpu_frac is None


def test_direct_run_mixed_is_independent_of_call_history(monkeypatch) -> None:
    # A prior agg sweep must not change what a later direct run_mixed estimates.
    backend = VLLMXPUBackend()
    captured_direct = []
    monkeypatch.setattr(VLLMXPUBackend, "_visual_context_tokens", lambda self, *a, **k: 0)
    monkeypatch.setattr(VLLMXPUBackend, "_max_kv_slots", lambda self, *a, **k: 3)
    monkeypatch.setattr(VLLMBackend, "run_mixed", lambda self, m, d, rc, step: captured_direct.append(step) or step)

    def fake_base_run_agg(self, model, database, runtime_config, **kwargs):
        self._resolve_agg_kwargs(kwargs, isl=1024, osl=128, backend_version="0.28.0")
        self.run_mixed(model, database, runtime_config, MixedStepInput(context_tokens=8192, num_decode_requests=99))
        return "summary"

    monkeypatch.setattr(VLLMBackend, "run_agg", fake_base_run_agg)
    rc = SimpleNamespace(isl=1024, osl=128)
    step = MixedStepInput(context_tokens=1000, num_decode_requests=4)

    first = backend.run_mixed(MagicMock(), MagicMock(), rc, step)
    backend.run_agg(MagicMock(), MagicMock(), rc, ctx_tokens=8192)  # agg sweep in between
    second = backend.run_mixed(MagicMock(), MagicMock(), rc, step)
    assert first == second == step
    assert [c for c in captured_direct if c == step] == [step, step]


@pytest.mark.parametrize(("forward_model", "context_tokens"), [("op_level", 4096), ("fpm", 1096)])
def test_compute_ttft_single_request_feeds_the_step_its_model_tokens(monkeypatch, forward_model, context_tokens):
    # One request, isl 4096 with 3000 cached tokens, under run_agg's one-step TTFT
    # budget of isl: op-level pricing reads the prompt, a whole-model FPM reads
    # the 1096 new tokens, and either step is one request's whole prefill.
    backend = VLLMXPUBackend()
    model = MagicMock(forward_model=forward_model, _num_layers=10)
    seen = []

    def run_mixed(self, _model, _database, _runtime, step):
        seen.append(step.context_tokens)
        return StepEstimate(latency_ms=40.0, energy_wms=0.0, context_tokens=step.context_tokens)

    monkeypatch.setattr(VLLMXPUBackend, "run_mixed", run_mixed)
    ttft = backend._compute_ttft(model, MagicMock(), MagicMock(), 1, 4096, 10, 4096, 3000, **_ttft_kwargs())
    assert seen == [context_tokens]
    assert ttft == pytest.approx(40.0 + 10 * 0.3)


@pytest.mark.parametrize(("forward_model", "prefill_ms"), [("op_level", 40.0 * 5000 / 2048), ("fpm", 40.0 * 2)])
def test_compute_ttft_single_request_counts_a_short_final_fpm_chunk(monkeypatch, forward_model, prefill_ms):
    # 5000 prompt tokens with 2000 cached tokens: the 3000-token suffix needs
    # two chunks, whereas using the full prompt would incorrectly count three.
    # The stub isolates the chunk count; native chunk pricing is tested separately.
    backend = VLLMXPUBackend()
    model = MagicMock(forward_model=forward_model, _num_layers=10)
    monkeypatch.setattr(
        VLLMXPUBackend,
        "run_mixed",
        lambda self, _m, _d, _r, step: StepEstimate(
            latency_ms=40.0, energy_wms=0.0, context_tokens=step.context_tokens
        ),
    )
    ttft = backend._compute_ttft(model, MagicMock(), MagicMock(), 1, 5000, 10, 2048, 2000, **_ttft_kwargs())
    assert ttft == pytest.approx(prefill_ms + 10 * 0.3)


@pytest.mark.parametrize(
    "forward_model,prefix,budget,expected",
    [("fpm", 4088, 4096, 8), ("fpm", 3000, 2048, 1096), ("op_level", 0, 100000, 65536)],
)
def test_batched_ttft_caps_prefill_after_reserving_decode(monkeypatch, forward_model, prefix, budget, expected):
    # Fifteen decoders leave one physical prefill slot. Its uncached suffix
    # has 8 or 1096 tokens, even when the token budget could hold more suffixes.
    backend = VLLMXPUBackend()
    backend._in_agg = True
    runtime = SimpleNamespace(isl=4096, osl=128, prefix=prefix, batch_size=16)
    model = SimpleNamespace(forward_model=forward_model, _num_layers=10)
    seen = []

    def native_step(self, _model, _database, _runtime, step):
        seen.append((step.context_tokens, step.num_decode_requests))
        return StepEstimate(latency_ms=10, energy_wms=0, context_tokens=step.context_tokens)

    monkeypatch.setattr(VLLMBackend, "run_mixed", native_step)
    monkeypatch.setattr(backend, "_visual_context_tokens", lambda *_: 0)
    monkeypatch.setattr(backend, "_max_kv_slots", lambda *_, **__: 1000)
    monkeypatch.setattr(backend, "_mix_step_gen_tokens", lambda *_: 15)
    monkeypatch.setattr(backend, "_effective_decode_bs", lambda *_: 16)
    monkeypatch.setattr(backend, "_get_genonly_step_latency", lambda *_: (1, 0, {}, {}, ()))
    monkeypatch.setattr(backend, "_own_cached_prefill_ms", lambda *_: 10)
    backend._compute_ttft(model, None, runtime, 16, 4096, 128, budget, prefix, **_ttft_kwargs())
    assert seen == [(expected, 15)]


@pytest.mark.parametrize("prefix", [0, 3000, 4088])
def test_cached_fpm_burst_keeps_prompt_length_calibration(monkeypatch, prefix):
    # The documented fit is ctx/isl, independent of the suffix used to query
    # FPM. Hold own-prefill at 10 ms and decode growth at one to isolate it.
    backend = VLLMXPUBackend()
    _patch_ttft_deps(monkeypatch, own_prefill=10, eff_bs=16, gen=1)
    runtime = SimpleNamespace(isl=4096, osl=128, prefix=prefix, batch_size=16)
    ttft = backend._compute_ttft(
        SimpleNamespace(forward_model="fpm"), None, runtime, 16, 4096, 128, 2048, prefix, **_ttft_kwargs()
    )
    # 1 + .45 * log2(16)^1.10 * sqrt(min(2048,16*suffix)/4096).
    expected = {0: 24.62054313441224, 3000: 24.62054313441224, 4088: 13.65513578360306}
    assert ttft == pytest.approx(expected[prefix])


def test_batched_cold_unqualified_mamba_keeps_calibrated_scaling(monkeypatch):
    from aisim_core.sdk.operations import Mamba2Kernel

    model = SimpleNamespace(
        forward_model="op_level",
        _num_layers=1,
        context_ops=[
            Mamba2Kernel("context_mamba_ssm", 1, "mamba_chunk_scan_combined", "context", 4096, 64, 64, 128, 4, 8, 128)
        ],
    )
    backend = VLLMXPUBackend()
    calls = []

    def mixed(*_):
        calls.append(1)
        return StepEstimate(latency_ms=40, energy_wms=0, context_tokens=1000)

    monkeypatch.setattr(backend, "run_mixed", mixed)
    monkeypatch.setattr(backend, "_mix_step_gen_tokens", lambda *_: 2)
    monkeypatch.setattr(backend, "_effective_decode_bs", lambda *_: 4)
    monkeypatch.setattr(backend, "_get_genonly_step_latency", lambda *_: (1, 0, {}, {}, ()))
    runtime = SimpleNamespace(isl=4096, osl=64, prefix=0, batch_size=4)
    ttft = backend._compute_ttft(model, None, runtime, 4, 4096, 64, 1002, 0, **_ttft_kwargs())
    # Legacy fit: linearly scale the 40 ms query by 4096/1000, dispatch
    # once, then apply the unchanged burst factor with unit decode growth.
    factor = 1 + BURST_A * math.log2(4) ** BURST_BS_EXP * (1002 / 4096) ** BURST_CTX_EXP
    assert ttft == pytest.approx((40 * 4.096 + 0.3) * factor)
    assert calls == [1]


@pytest.mark.parametrize("native_op", [False, True])
@pytest.mark.parametrize("generation_only", [False, True])
@pytest.mark.parametrize("regime", ["burst", "admission"])
@pytest.mark.parametrize(
    "prefix,budget,chunks",
    [
        (3000, 514, [(3512, 3000), (4024, 3512), (4096, 4024)]),
        (3072, 514, [(3584, 3072), (4096, 3584)]),
        (3584, 514, [(4096, 3584)]),
        (3000, 2050, [(4096, 3000)]),
        # One complete packed request is already the exact native query.
        (3000, 4098, []),
        # Even chunks remain a control; a short cold tail pays a full state.
        (0, 514, [(past + 512, past) for past in range(0, 4096, 512)]),
        (0, 1002, [(1000, 0), (2000, 1000), (3000, 2000), (4000, 3000), (4096, 4000)]),
    ],
)
def test_batched_ttft_prices_native_uncached_chunks(
    monkeypatch, regime, prefix, budget, chunks, native_op, generation_only
):
    from aisim_core import _native
    from aisim_core.sdk import operations as ops
    from aisim_core.sdk.config import ModelConfig, RuntimeConfig
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.perf_database import get_database_view

    # An isolated native accounting fixture, not an XPU calibration claim.
    # Keep scans and decode attention; no measured data or model weights are needed.
    database = get_database_view("b200_sxm", "vllm", "0.24.0", database_mode="SOL", allow_unlisted_version=True)
    assert database is not None
    model = get_model(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1),
        "vllm",
    )
    mamba_type = _native.Mamba2Kernel if native_op else ops.Mamba2Kernel
    model.context_ops = (
        []
        if generation_only
        else [
            mamba_type(
                "context_mamba_ssm",
                1,
                "mamba_chunk_scan_combined",
                "context",
                4096,
                64,
                64,
                128,
                4,
                8,
                128,
                state_dtype="float16",
            )
        ]
    )
    decode_attention = [op for op in model.generation_ops if op._name == "generation_attention"]
    assert decode_attention
    model.generation_ops = decode_attention + [
        mamba_type(
            "generation_mamba_ssm",
            1,
            "selective_state_update",
            "generation",
            4096,
            64,
            64,
            128,
            4,
            8,
            128,
            state_dtype="float16",
        )
    ]
    model._num_layers = 1
    runtime = RuntimeConfig(batch_size=4, isl=4096, osl=64, prefix=prefix)
    backend = VLLMXPUBackend()
    backend._in_agg = True
    monkeypatch.setattr(backend, "_mix_step_gen_tokens", lambda *_: 2)
    monkeypatch.setattr(backend, "_max_kv_slots", lambda *a, **k: 100)
    monkeypatch.setattr(backend, "_effective_decode_bs", lambda *_: 4 if regime == "burst" else 2)
    monkeypatch.setattr(backend, "_get_genonly_step_latency", lambda *_: (1.0, 0.0, {}, {}, ()))
    monkeypatch.setattr(backend, "_decode_tpot_ms", lambda *a, **k: 1.5)
    calls = []
    estimates = []
    native_run_mixed = VLLMBackend.run_mixed

    def capture(self, model, database, runtime, step):
        calls.append((runtime.isl, runtime.prefix, step.context_tokens, step.num_decode_requests))
        estimate = native_run_mixed(self, model, database, runtime, step)
        estimates.append(estimate)
        return estimate

    monkeypatch.setattr(VLLMBackend, "run_mixed", capture)
    ttft = backend._compute_ttft(model, database, runtime, 4, 4096, 64, budget, prefix, **_ttft_kwargs())

    # A scan transfers 20,608 bytes/new token and reads+writes its 1 MiB
    # FP16 state per request. There are two decoders alongside every chunk.
    # For prefix 3000 / budget 514: 1096 new tokens, three prefill states,
    # six decode tokens/states, at B200's 7.7e12 B/s. The final 72 tokens
    # pay a full state charge. Dispatch is 0.3 ms once for the whole request.
    count = max(1, len(chunks))
    activation_bytes = 0 if generation_only else (4096 - prefix) * 20_608
    state_bytes = 0 if generation_only else count * 2 * 1_048_576
    decode_scan_ms = (2 * 20_608 + 2 * 2 * 1_048_576) / 7.7e12 * 1000
    assert estimates[0].per_op_latency_ms["generation_mamba_ssm"] == pytest.approx(decode_scan_ms, rel=1e-9)
    # Concurrent decoders keep the original native KV shape, independently
    # of this request's growing prefix. Reuse that component, not its scan
    # formula, so attention's changing KV footprint is covered too.
    decode_ms = estimates[0].component_latency_ms["decode_attention"]
    own_ms = (activation_bytes + state_bytes) / 7.7e12 * 1000 + count * decode_ms + 0.3
    if regime == "burst":
        factor = 1 + BURST_A * math.log2(4) ** BURST_BS_EXP * (budget / 4096) ** BURST_CTX_EXP
        expected = own_ms * min(factor, 4)
    else:
        expected = own_ms + (4 - 2) / 2 * (own_ms + 64 * 1.5)
    assert ttft == pytest.approx(expected, rel=1e-9)
    assert calls[0] == (4096, prefix, budget - 2, 2)
    assert calls[1:] == [(isl, past, isl, 2) for isl, past in chunks]
    assert (runtime.isl, runtime.prefix, runtime.batch_size) == (4096, prefix, 4)
