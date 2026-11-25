# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aisim.sdk import common
from aisim.sdk.backends.base_backend import BaseBackend
from aisim.sdk.config import ModelConfig, RuntimeConfig
from aisim.sdk.performance_result import MoECommFallback
from aisim.sdk.step_estimate import MixedStepInput, StepEstimate

pytestmark = pytest.mark.unit

_CONTEXT_FALLBACK = MoECommFallback("context", "deepep_ht", 32, 8, 8, 1)
_GENERATION_FALLBACK = MoECommFallback("generation", "deepep_ll", 32, 8, 8, 1)


def _decode_step(latency, energy, per_op_latency, per_op_source, fallbacks):
    return StepEstimate(
        latency_ms=latency,
        energy_wms=energy,
        per_op_latency_ms=per_op_latency,
        per_op_source=per_op_source,
        moe_comm_fallbacks=fallbacks,
    )


class _LatencyResult:
    def __init__(self, latency_ms: float, energy_wms: float) -> None:
        self._latency_ms = latency_ms
        self.energy = energy_wms

    def __float__(self) -> float:
        return self._latency_ms


class _StaticOp:
    def __init__(self, name: str, latency_ms: float, energy_wms: float) -> None:
        self._name = name
        self._latency_ms = latency_ms
        self._energy_wms = energy_wms

    def query(self, *args, **kwargs) -> _LatencyResult:
        return _LatencyResult(self._latency_ms, self._energy_wms)


class _TestBackend(BaseBackend):
    def find_best_agg_result_under_constraints(self, model, database, runtime_config, **kwargs):
        raise NotImplementedError

    def _get_memory_usage(
        self,
        model,
        database,
        batch_size,
        beam_width,
        isl,
        osl,
        num_tokens=0,
        prefix=0,
        encoder_memory=None,
        mtp_scaled_tokens=None,
    ) -> dict[str, float]:
        return {"total": 1.0}


@pytest.fixture
def backend() -> BaseBackend:
    return _TestBackend()


@pytest.fixture
def database():
    return SimpleNamespace(
        backend="test-backend",
        version="test-version",
        system="test-system",
        system_spec={"gpu": {"mem_capacity": 80 * (1 << 30)}},
    )


@pytest.fixture
def model():
    model = MagicMock()
    model.model_path = "test-model"
    model.model_name = "test-model"
    model.forward_model = "op_level"
    model.architecture = "test-arch"  # run_agg keys its result cache by the engine identity
    model._nextn = 0
    model.get_resident_weights_bytes.return_value = 0.0
    model.get_additional_activation_bytes.return_value = 0.0
    model.encoder_ops = []
    model.context_ops = [
        _StaticOp("context_attention", latency_ms=11.0, energy_wms=110.0),
        _StaticOp("logits_gemm", latency_ms=3.0, energy_wms=30.0),
    ]
    model.generation_ops = [
        _StaticOp("generation_attention", latency_ms=2.0, energy_wms=20.0),
        _StaticOp("generation_mlp", latency_ms=1.0, energy_wms=10.0),
    ]
    model.config = ModelConfig(
        tp_size=1,
        pp_size=1,
        attention_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        comm_quant_mode=common.CommQuantMode.half,
    )
    return model


@pytest.fixture
def runtime_config() -> RuntimeConfig:
    return RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2)


class TestMTPActivationMemoryScaling:
    """MTP activation scaling applies only to the decode-token share."""

    @staticmethod
    def _model():
        return SimpleNamespace(
            context_ops=[SimpleNamespace(get_weights=lambda: 0.0)],
            get_resident_weights_bytes=lambda: 0.0,
            get_additional_activation_bytes=lambda _tokens: 0.0,
            config=ModelConfig(
                tp_size=1,
                pp_size=1,
                attention_dp_size=1,
                moe_tp_size=1,
                moe_ep_size=1,
            ),
            _num_heads=32,
            _head_size=128,
            _num_experts=0,
            model_family="test",
            get_kvcache_bytes_per_sequence=lambda _seq_len: 0.0,
            get_kvcache_rank_bytes_per_sequence=lambda _seq_len: 0.0,
        )

    @staticmethod
    def _database():
        return SimpleNamespace(system_spec={"misc": {"nccl_mem": {1: 0}, "other_mem": 0}})

    def _activations(self, *, nextn: int, num_tokens: int, mtp_scaled_tokens: int | None) -> float:
        model = self._model()
        model.config.nextn = nextn
        return BaseBackend()._get_memory_usage(
            model,
            self._database(),
            batch_size=4,
            beam_width=1,
            isl=65_536,
            osl=400,
            num_tokens=num_tokens,
            mtp_scaled_tokens=mtp_scaled_tokens,
        )["activations"]

    def test_mixed_step_scales_only_decode_share(self):
        num_tokens = 65_536 + 3
        base = self._activations(nextn=0, num_tokens=num_tokens, mtp_scaled_tokens=3)
        spec = self._activations(nextn=3, num_tokens=num_tokens, mtp_scaled_tokens=3)
        assert spec / base == pytest.approx((65_536 + 3 * 4) / num_tokens, rel=1e-6)

    def test_decode_only_step_keeps_full_multiplier(self):
        base = self._activations(nextn=0, num_tokens=512, mtp_scaled_tokens=None)
        spec = self._activations(nextn=3, num_tokens=512, mtp_scaled_tokens=None)
        assert spec / base == pytest.approx(4.0, rel=1e-6)

    def test_prefill_only_step_does_not_scale(self):
        base = self._activations(nextn=0, num_tokens=65_536, mtp_scaled_tokens=0)
        spec = self._activations(nextn=3, num_tokens=65_536, mtp_scaled_tokens=0)
        assert spec == pytest.approx(base, rel=1e-9)


class TestAFDPartitionActivationScaling:
    """``get_partition_memory_usage`` must honor the AFD ``num_tokens`` contract.

    ``InferenceSession._estimate_{a,f}_memory_dict`` passes ``num_tokens=0`` for
    ``phase == "prefill"`` -- the sentinel that makes ``_get_memory_usage`` derive
    the count from ``(isl - prefix) * batch_size``, a prefill-only footprint --
    and ``num_tokens=batch_size`` (one token per request) for decode.
    """

    MODEL = "openai/gpt-oss-120b"

    @classmethod
    def _model(cls, *, nextn: int):
        from aisim.sdk.models import get_model

        model = get_model(cls.MODEL, ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1), "vllm")
        model.config.nextn = nextn
        return model

    @staticmethod
    def _database():
        return SimpleNamespace(
            backend="vllm",
            version="test-version",
            system="b200_sxm",
            system_spec={
                "gpu": {"mem_capacity": 180 * (1 << 30)},
                "misc": {"nccl_mem": {1: 0, 2: 0, 4: 0, 8: 0}, "other_mem": 0},
            },
        )

    def _partition_activations(self, *, nextn: int, num_tokens: int) -> float:
        from aisim.sdk.backends.factory import get_backend

        return get_backend("vllm").get_partition_memory_usage(
            self._model(nextn=nextn),
            self._database(),
            partition_ops=[],
            batch_size=4,
            beam_width=1,
            isl=65_536,
            osl=400,
            num_tokens=num_tokens,
            max_seq_len=65_536,
        )["activations"]

    def test_prefill_partition_does_not_scale(self):
        """AFD prefill workers size activations from the derived context tokens.

        Pins the measured shape (gpt-oss-120b, ISL 65,536, batch 4, TP1): the
        context footprint is 44.0 GiB, which the full ``(nextn+1)`` multiplier
        turned into 176.0 GiB -- a 4x over-count that shrinks the A/F worker KV
        budget and over-prunes AFD batch sizes.
        """
        base = self._partition_activations(nextn=0, num_tokens=0)
        spec = self._partition_activations(nextn=3, num_tokens=0)
        assert base == pytest.approx(44.0, abs=0.1)
        assert spec == pytest.approx(base, rel=1e-9)

    def test_decode_partition_keeps_full_multiplier(self):
        """AFD decode workers pass ``num_tokens=batch_size``: every token is verified."""
        base = self._partition_activations(nextn=0, num_tokens=4)
        spec = self._partition_activations(nextn=3, num_tokens=4)
        assert spec / base == pytest.approx(4.0, rel=1e-6)


@pytest.mark.parametrize("mode", ["static", "static_ctx", "static_gen"])
@pytest.mark.parametrize("latency_correction_scale", [1.0, 1.25])
def test_run_static_latency_only_matches_run_static_latency(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
    mode: str,
    latency_correction_scale: float,
) -> None:
    from aisim.sdk.backends import base_backend as base_backend_module

    def _fake_rust_breakdown(model_arg, database_arg, runtime_config_arg, mode_arg, stride_arg, scale_arg):
        # Mode- and scale-aware like the real bridge: the engine applies the
        # flat correction scale itself and only fills the phases the mode ran.
        ctx = {"context_attention": 11.0 * scale_arg, "logits_gemm": 3.0 * scale_arg}
        gen = {"generation_attention": 2.0 * scale_arg, "generation_mlp": 1.0 * scale_arg}
        if mode_arg == "static_ctx":
            gen = {}
        elif mode_arg == "static_gen":
            ctx = {}
        return (
            ctx,
            gen,
            {name: latency * 10.0 for name, latency in ctx.items()},
            {name: latency * 10.0 for name, latency in gen.items()},
            dict.fromkeys(ctx, "silicon"),
            dict.fromkeys(gen, "silicon"),
            (),
        )

    monkeypatch.setattr(base_backend_module, "estimate_static_latency_breakdown_with_rust", _fake_rust_breakdown)
    runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2, engine_step_backend="rust")

    summary = backend.run_static(
        model,
        database,
        runtime_config,
        mode=mode,
        stride=2,
        latency_correction_scale=latency_correction_scale,
    )
    latency_only = backend.run_static_latency_only(
        model,
        database,
        runtime_config,
        mode=mode,
        stride=2,
        latency_correction_scale=latency_correction_scale,
    )

    summary_latency = sum(summary.get_context_latency_dict().values()) + sum(
        summary.get_generation_latency_dict().values()
    )
    request_latency = float(summary.get_summary_df().iloc[0]["request_latency"])

    assert latency_only == pytest.approx(summary_latency)
    assert latency_only == pytest.approx(request_latency, abs=1e-3)


def test_run_static_can_route_to_rust_engine_step_backend(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    from aisim.sdk.backends import base_backend as base_backend_module

    calls = []

    def _fake_rust_breakdown(model_arg, database_arg, runtime_config_arg, mode_arg, stride_arg, scale_arg):
        calls.append((model_arg, database_arg, runtime_config_arg, mode_arg, stride_arg, scale_arg))
        return (
            {"context_qkv_gemm": 4.0, "context_attention": 3.0},
            {"generation_qkv_gemm": 2.0, "generation_attention": 1.0},
            {"context_qkv_gemm": 40.0, "context_attention": 8.0},
            {"generation_qkv_gemm": 12.0, "generation_attention": 5.0},
            {"context_qkv_gemm": "silicon", "context_attention": "empirical"},
            {"generation_qkv_gemm": "silicon", "generation_attention": "mixed"},
            (_CONTEXT_FALLBACK,),
        )

    monkeypatch.setattr(
        base_backend_module,
        "estimate_static_latency_breakdown_with_rust",
        _fake_rust_breakdown,
    )

    summary = backend.run_static(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2, engine_step_backend="rust"),
        mode="static",
        stride=2,
        latency_correction_scale=1.25,
    )

    assert len(calls) == 1
    assert calls[0][3:] == ("static", 2, 1.25)
    # Real op-name keys folded from the compiled engine's per-op results.
    assert summary.get_context_latency_dict() == {"context_qkv_gemm": 4.0, "context_attention": 3.0}
    assert summary.get_generation_latency_dict() == {"generation_qkv_gemm": 2.0, "generation_attention": 1.0}
    # Per-op energies are real and DISTINCT per key — not one phase total
    # smeared onto every key, and not recomputed by a Python double-eval.
    assert summary.get_context_energy_wms_dict() == {"context_qkv_gemm": 40.0, "context_attention": 8.0}
    assert summary.get_generation_energy_wms_dict() == {"generation_qkv_gemm": 12.0, "generation_attention": 5.0}
    assert summary.get_context_source_dict() == {"context_qkv_gemm": "silicon", "context_attention": "empirical"}
    assert summary.get_generation_source_dict() == {"generation_qkv_gemm": "silicon", "generation_attention": "mixed"}
    assert summary.get_moe_comm_fallbacks() == (_CONTEXT_FALLBACK,)


@pytest.mark.parametrize(
    ("mode", "expected_share"),
    [
        # "static" and "static_ctx" both size activations from the DERIVED
        # context token count ((isl - prefix) * batch_size), so no share of that
        # footprint verifies nextn+1 draft tokens.
        ("static", 0),
        ("static_ctx", 0),
        # Decode-only step (num_tokens = batch_size * beam_width): every token is
        # verified, so the full (nextn+1) multiplier stays (no decode-share cap).
        ("static_gen", None),
    ],
)
def test_run_static_declares_mtp_decode_share_per_mode(
    backend: BaseBackend,
    model,
    database,
    monkeypatch,
    mode: str,
    expected_share: int | None,
) -> None:
    """Each static mode must declare its decode-token share to ``_get_memory_usage``.

    The latency side is stubbed at the engine bridge (the compiled engine is the
    only static engine-step executor); memory sizing stays a Python
    ``_get_memory_usage`` call, which is the surface under test here.
    """
    from aisim.sdk.backends import base_backend as base_backend_module

    def _fake_rust_breakdown(model_arg, database_arg, runtime_config_arg, mode_arg, stride_arg, scale_arg):
        ctx = {"context_attention": 1.0}
        gen = {"generation_attention": 1.0}
        if mode_arg == "static_ctx":
            gen = {}
        elif mode_arg == "static_gen":
            ctx = {}
        return (ctx, gen, dict(ctx), dict(gen), dict.fromkeys(ctx, "silicon"), dict.fromkeys(gen, "silicon"), ())

    monkeypatch.setattr(
        base_backend_module,
        "estimate_static_latency_breakdown_with_rust",
        _fake_rust_breakdown,
    )

    captured: dict = {}

    def _record(*args, **kwargs):
        captured.update(kwargs)
        return {"total": 1.0}

    monkeypatch.setattr(backend, "_get_memory_usage", _record)

    backend.run_static(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2, engine_step_backend="rust"),
        mode=mode,
        stride=2,
    )

    assert captured.get("mtp_scaled_tokens") == expected_share


def test_run_agg_declares_mtp_decode_share_at_call_site(
    backend: BaseBackend,
    model,
    database,
    monkeypatch,
) -> None:
    """run_agg must pass the decode-request share of ``num_tokens`` as
    ``mtp_scaled_tokens`` to ``_get_memory_usage``.

    The run_agg counterpart of test_run_static_declares_mtp_decode_share_per_mode
    above: pins the CALL SITE for a mixed step (b > 1: only the num_gen_requests
    decode requests verify nextn+1 draft tokens; the context share is processed
    once). The b == 1 context-only step and its separate decode peak are pinned
    by test_run_agg_b1_uses_scheduled_activation_peak below.
    """
    memory_calls: list[dict] = []

    def _record(*args, **kwargs):
        memory_calls.append(kwargs)
        return {"total": 1.0}

    monkeypatch.setattr(backend, "_get_memory_usage", _record)
    monkeypatch.setattr(
        backend,
        "run_mixed",
        lambda *args, **kwargs: StepEstimate(latency_ms=1.0, energy_wms=1.0),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(1.0, 1.0, {}, {}, ()),
    )

    backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=4, beam_width=1, isl=8, osl=5, prefix=2, engine_step_backend="rust"),
        ctx_tokens=8,
    )

    assert len(memory_calls) == 1
    assert memory_calls[0].get("mtp_scaled_tokens") == 3


def test_trtllm_budget_path_ignores_the_decode_share() -> None:
    """TRT-LLM's agg override intentionally drops ``mtp_scaled_tokens``.

    Its ``num_tokens`` is ``BuildConfig.max_num_tokens`` (a per-iteration budget),
    not the agg-derived mixed-step count, so the decode-share correction is not
    forwarded and the legacy full ``(nextn+1)`` multiplier is retained on that
    path pending its own analysis (tracked in AIC-1755).
    """
    from aisim.sdk.backends.factory import get_backend

    agg_extra = {"max_num_tokens": 8192, "max_seq_len": 4096, "free_gpu_memory_fraction": 0.9}
    kwargs = get_backend("trtllm")._memory_usage_kwargs_for_agg(
        num_tokens=65_539,
        agg_extra=agg_extra,
        mtp_scaled_tokens=3,
    )
    assert kwargs == {"num_tokens": 8192, "max_seq_len": 4096}


def test_run_agg_with_osl_one_does_not_divide_by_zero(
    backend: BaseBackend,
    model,
    database,
    monkeypatch,
) -> None:
    """Regression: osl=1 (no-decode) must not raise and tokens/s/user must be 0.0."""
    monkeypatch.setattr(
        backend,
        "run_mixed",
        lambda *args, **kwargs: StepEstimate(latency_ms=1.0, energy_wms=1.0),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(0.0, 0.0, {}, {}, ()),
    )
    monkeypatch.setattr(
        backend,
        "_get_memory_usage",
        lambda *args, **kwargs: {"total": 1.0},
    )

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=1, prefix=2, engine_step_backend="rust"),
        ctx_tokens=8,
    )

    row = summary.get_summary_df().iloc[0]
    assert row["tpot"] == 0.0  # no generated tokens after the prefill token
    assert row["tokens/s/user"] == 0.0


@pytest.mark.parametrize(
    ("osl", "expected_memory_calls", "expected_activations"),
    [
        (1, [(8, 0)], 0.068359375),
        (5, [(8, 0), (1, None)], 0.2734375),
    ],
)
def test_run_agg_b1_uses_scheduled_activation_peak(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
    osl: int,
    expected_memory_calls: list[tuple[int, int | None]],
    expected_activations: float,
) -> None:
    """A b=1 agg run retains a decode peak only when decode is scheduled."""
    model._nextn = 3
    model.config.nextn = 3
    model.context_ops = [SimpleNamespace(get_weights=lambda: 0.0)]
    model._num_heads = 32
    model._head_size = 128
    model._num_experts = 0
    model.model_family = "test"
    model.get_kvcache_bytes_per_sequence = lambda _seq_len: 0.0
    model.get_kvcache_rank_bytes_per_sequence = lambda _seq_len: 0.0
    database.system_spec["misc"] = {"nccl_mem": {1: 0}, "other_mem": 0}
    monkeypatch.setattr(
        backend,
        "run_mixed",
        lambda *args, **kwargs: StepEstimate(latency_ms=1.0, energy_wms=1.0),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(1.0, 1.0, {"decode": 1.0}, {"decode": "silicon"}, ()),
    )

    memory_calls: list[dict] = []

    def _get_memory_usage(*args, **kwargs):
        memory_calls.append(kwargs)
        return BaseBackend._get_memory_usage(backend, *args, **kwargs)

    monkeypatch.setattr(backend, "_get_memory_usage", _get_memory_usage)

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=1, beam_width=1, isl=8, osl=osl, engine_step_backend="rust"),
        ctx_tokens=8,
    )

    assert [(call["num_tokens"], call["mtp_scaled_tokens"]) for call in memory_calls] == expected_memory_calls
    assert summary.get_memory()["activations"] == pytest.approx(expected_activations)


@pytest.mark.parametrize(
    ("forward_model", "ctx_tokens", "context_tokens", "activation_tokens"),
    [
        # Op-level pricing reads prompt tokens: the whole 8-token prompt fits.
        ("op_level", 8, [8], 6),
        ("op_level", 4, [6, 8], 4),
        # A whole-model FPM reads new tokens: 6 is one request, not two.
        ("fpm", 8, [6], 6),
        ("fpm", 4, [4, 2], 4),
    ],
)
def test_run_agg_b1_schedules_one_request_and_sizes_its_new_tokens(
    monkeypatch, backend, model, database, forward_model, ctx_tokens, context_tokens, activation_tokens
) -> None:
    # isl 8 with 2 cached tokens: the step schedules 6 new tokens, at most the budget.
    model.forward_model = forward_model
    steps: list[MixedStepInput] = []
    memory_calls: list[dict] = []

    def run_mixed(_model, _database, _runtime, step):
        steps.append(step)
        return StepEstimate(latency_ms=1.0, energy_wms=1.0)

    monkeypatch.setattr(backend, "run_mixed", run_mixed)
    monkeypatch.setattr(
        backend, "_get_genonly_step_estimate", lambda *_: _decode_step(1.0, 1.0, {"decode": 1.0}, {"decode": "sol"}, ())
    )
    monkeypatch.setattr(backend, "_get_memory_usage", lambda *_, **kw: memory_calls.append(kw) or {"total": 1.0})
    backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=1, isl=8, osl=1, prefix=2, engine_step_backend="rust"),
        ctx_tokens=ctx_tokens,
    )
    assert [step.context_tokens for step in steps] == context_tokens
    assert [call["num_tokens"] for call in memory_calls] == [activation_tokens]


def test_run_agg_b1_fully_cached_fpm_prompt_reaches_the_prefix_check(monkeypatch, backend, model, database) -> None:
    # No new token to schedule: the step keeps a positive budget, so the engine's
    # prefix check (not MixedStepInput) rejects the point, as for op-level graphs.
    model.forward_model = "fpm"
    steps: list[MixedStepInput] = []

    def run_mixed(_model, _database, _runtime, step):
        steps.append(step)
        raise RuntimeError("isl must be greater than prefix")

    monkeypatch.setattr(backend, "run_mixed", run_mixed)
    with pytest.raises(RuntimeError, match="isl must be greater than prefix"):
        backend.run_agg(
            model,
            database,
            RuntimeConfig(batch_size=1, isl=8, osl=1, prefix=8, engine_step_backend="rust"),
            ctx_tokens=4,
        )
    assert [step.context_tokens for step in steps] == [4]


def test_run_mixed_returns_components_and_counts_speculative_query_tokens(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    from aisim.sdk.backends import base_backend as base_backend_module

    calls: list[dict] = []

    def _fake_mixed_breakdown(model_arg, database_arg, **kwargs):
        calls.append(kwargs)
        return {
            "latency_ms": 8.5,
            "energy_wms": 85.0,
            "component_latency_ms": {"shared_non_attention": 5.0, "context_attention": 2.0, "decode_attention": 1.5},
            "component_energy_wms": {"shared_non_attention": 50.0, "context_attention": 20.0, "decode_attention": 15.0},
            "per_op_latency_ms": {"context_mlp": 5.0},
            "per_op_source": {"context_mlp": "silicon"},
            "moe_comm_fallbacks": (),
        }

    monkeypatch.setattr(base_backend_module, "estimate_mixed_step_breakdown_with_rust", _fake_mixed_breakdown)
    model._nextn = 2

    estimate = backend.run_mixed(
        model,
        database,
        RuntimeConfig(isl=8, osl=5, prefix=2, engine_step_backend="rust"),
        MixedStepInput(
            context_tokens=8,
            num_decode_requests=2,
        ),
    )

    assert estimate.num_decode_requests == 2
    # Every decode request verifies one target token plus the two scheduled
    # drafts (nextn=2); scheduling metadata is computed before the bridge call.
    assert estimate.num_decode_query_tokens == 6
    assert estimate.latency_ms == pytest.approx(sum(estimate.component_latency_ms.values()))
    assert set(estimate.component_latency_ms) == {
        "shared_non_attention",
        "context_attention",
        "decode_attention",
    }
    # The bridge receives the RAW decode-request count; the engine applies the
    # (nextn + 1) scaling internally, exactly like the deleted Python passes.
    assert calls[0]["gen_tokens"] == 2
    assert calls[0]["ctx_tokens"] == 8


def test_run_mixed_rust_path_returns_the_same_structured_contract(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """The rust branch maps the bridge's rich dict 1:1 onto ``StepEstimate``:
    real energy, per-component splits, and the Python branch's per-op keys
    (raw non-attention names plus the two literal attention keys) — the
    synthetic "rust_engine_step_mixed" key no longer exists."""
    from aisim.sdk.backends import base_backend as base_backend_module

    model._nextn = 2
    monkeypatch.setattr(base_backend_module, "should_use_rust_engine_step", lambda *args: True)
    components = {
        "latency_ms": 8.5,
        "energy_wms": 85.0,
        "component_latency_ms": {"shared_non_attention": 5.0, "context_attention": 2.0, "decode_attention": 1.5},
        "component_energy_wms": {"shared_non_attention": 50.0, "context_attention": 20.0, "decode_attention": 15.0},
        "per_op_latency_ms": {"context_mlp": 5.0, "context_attention (scaled)": 2.0, "generation_attention": 1.5},
        "per_op_source": {
            "context_mlp": "silicon",
            "context_attention (scaled)": "empirical",
            "generation_attention": "silicon",
        },
        "moe_comm_fallbacks": (),
    }
    monkeypatch.setattr(
        base_backend_module,
        "estimate_mixed_step_breakdown_with_rust",
        lambda *args, **kwargs: components,
    )

    estimate = backend.run_mixed(
        model,
        database,
        RuntimeConfig(isl=8, osl=5, engine_step_backend="rust"),
        MixedStepInput(context_tokens=8, num_decode_requests=2),
    )

    assert estimate.latency_ms == 8.5
    assert estimate.energy_wms == 85.0  # real energy, not a 0.0 placeholder
    assert estimate.component_latency_ms == components["component_latency_ms"]
    assert estimate.component_energy_wms == components["component_energy_wms"]
    assert estimate.per_op_latency_ms == components["per_op_latency_ms"]
    assert "context_attention (scaled)" in estimate.per_op_latency_ms
    assert estimate.per_op_source == components["per_op_source"]
    assert estimate.context_tokens == 8
    assert estimate.num_decode_requests == 2
    assert estimate.num_decode_query_tokens == 6


def test_get_genonly_step_latency_rust_path_returns_decode_breakdown_verbatim(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """The rust branch returns ``estimate_decode_step_breakdown_with_rust``
    verbatim — real op names, real per-op energy (no synthetic
    "rust_engine_step_generation" key) — without running the Python static
    step."""
    from aisim.sdk.backends import base_backend as base_backend_module

    calls = []
    breakdown = (
        3.0,
        24.0,
        {"generation_attention": 2.0, "generation_mlp": 1.0},
        {"generation_attention": "silicon", "generation_mlp": "empirical"},
        (),
    )

    def _fake_decode_breakdown(model_arg, database_arg, **kwargs):
        calls.append(kwargs)
        return breakdown

    monkeypatch.setattr(base_backend_module, "estimate_decode_step_breakdown_with_rust", _fake_decode_breakdown)

    def _python_step_trap(*args, **kwargs):
        raise AssertionError("the rust path must not run the Python static step")

    monkeypatch.setattr(backend, "run_static", _python_step_trap)

    result = backend._get_genonly_step_latency(
        model,
        database,
        RuntimeConfig(isl=8, osl=6, gen_seq_imbalance_correction_scale=1.25, engine_step_backend="rust"),
        gen_tokens=4,
        isl=8,
        osl=6,
    )

    assert result == breakdown
    assert calls == [{"gen_tokens": 4, "isl": 8, "osl": 6, "gen_seq_imbalance_correction_scale": 1.25}]


def test_run_agg_applies_speculative_progress_in_scheduler(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    model._nextn = 1
    seen_steps: list[MixedStepInput] = []

    def _run_mixed(*args, **kwargs):
        step = args[-1]
        seen_steps.append(step)
        return StepEstimate(
            latency_ms=10.0,
            energy_wms=100.0,
            component_latency_ms={"shared_non_attention": 10.0},
            component_energy_wms={"shared_non_attention": 100.0},
            num_decode_requests=step.num_decode_requests,
            num_decode_query_tokens=step.num_decode_requests * 2,
        )

    monkeypatch.setattr(backend, "run_mixed", _run_mixed)
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(5.0, 50.0, {"decode": 5.0}, {"decode": "silicon"}, ()),
    )

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend="rust"),
        ctx_tokens=8,
        decode_tokens_per_iteration=2.0,
    )
    row = summary.get_summary_df().iloc[0]
    scheduling = summary.get_step_estimates()["scheduling"]

    assert seen_steps[0].num_decode_requests == 1
    assert scheduling["decode_tokens_per_iteration"] == 2.0
    assert scheduling["decode_iterations"] == 3.0
    assert scheduling["num_mix_steps"] == 2.0
    assert scheduling["num_genonly_steps"] == 1.0
    assert row["tpot"] == pytest.approx(4.167)
    assert row["tokens/s"] == pytest.approx(320.0)


def test_run_agg_records_progress_only_when_explicitly_supplied(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """An omitted decode_tokens_per_iteration must leave no scheduler-progress
    marker on the summary: its presence is what tells the upper-layer
    projection to stand down, so recording the 1.0 default would silently
    disable the legacy post-hoc projection flow."""
    monkeypatch.setattr(
        backend,
        "run_mixed",
        lambda *args, **kwargs: StepEstimate(latency_ms=10.0, energy_wms=100.0),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(5.0, 50.0, {"decode": 5.0}, {"decode": "silicon"}, ()),
    )

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend="rust"),
        ctx_tokens=8,
    )
    assert "decode_tokens_per_iteration" not in summary.get_step_estimates()["scheduling"]

    explicit = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend="rust"),
        ctx_tokens=8,
        decode_tokens_per_iteration=1.0,
    )
    assert explicit.get_step_estimates()["scheduling"]["decode_tokens_per_iteration"] == 1.0
    # The two calls schedule identically but carry different projection
    # eligibility; they must not have shared a cache entry.
    assert explicit is not summary


def test_run_agg_preserves_executed_moe_fallbacks_from_mixed_and_decode_steps(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    monkeypatch.setattr(
        backend,
        "run_mixed",
        lambda *args, **kwargs: StepEstimate(
            latency_ms=10.0,
            energy_wms=100.0,
            moe_comm_fallbacks=(_CONTEXT_FALLBACK,),
        ),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(
            5.0,
            50.0,
            {"decode": 5.0},
            {"decode": "estimated"},
            (_CONTEXT_FALLBACK, _GENERATION_FALLBACK),
        ),
    )

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend="rust"),
        ctx_tokens=8,
    )

    assert summary.get_moe_comm_fallbacks() == (_CONTEXT_FALLBACK, _GENERATION_FALLBACK)


def test_run_agg_omits_decode_fallback_when_decode_step_has_zero_weight(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    monkeypatch.setattr(
        backend,
        "run_mixed",
        lambda *args, **kwargs: StepEstimate(
            latency_ms=10.0,
            energy_wms=100.0,
            moe_comm_fallbacks=(_CONTEXT_FALLBACK,),
        ),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(
            5.0,
            50.0,
            {"decode": 5.0},
            {"decode": "estimated"},
            (_GENERATION_FALLBACK,),
        ),
    )

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=1, beam_width=1, isl=8, osl=1, engine_step_backend="rust"),
        ctx_tokens=8,
    )

    assert summary.get_moe_comm_fallbacks() == (_CONTEXT_FALLBACK,)
    assert summary.get_per_ops_data()["scheduling"]["num_genonly_steps"] == 0.0


def test_run_agg_cache_separates_video_workloads(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    model.encoder_config = _vision_encoder_config()
    mixed_calls: list[int] = []

    def _run_mixed(*args, **kwargs):
        mixed_calls.append(args[2].video_frames)
        return StepEstimate(latency_ms=1.0, energy_wms=1.0)

    monkeypatch.setattr(backend, "run_mixed", _run_mixed)
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(1.0, 1.0, {}, {}, ()),
    )

    common_kwargs = dict(
        batch_size=2,
        beam_width=1,
        isl=8,
        osl=5,
        video_height=448,
        video_width=448,
        num_videos_per_request=1,
        engine_step_backend="rust",
    )
    backend.run_agg(model, database, RuntimeConfig(**common_kwargs, video_frames=8), ctx_tokens=8)
    backend.run_agg(model, database, RuntimeConfig(**common_kwargs, video_frames=10), ctx_tokens=8)

    assert mixed_calls == [8, 10]
    assert len(backend._agg_cache) == 2


@pytest.mark.parametrize(
    ("engine_step_backend", "error", "match"),
    [
        ("auto", ValueError, "unknown engine_step_backend 'auto'"),
        (None, TypeError, "compiled engine is the only aggregate engine-step executor"),
    ],
)
def test_run_agg_validates_backend_before_cache_hit(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
    engine_step_backend: str | None,
    error: type[Exception],
    match: str,
) -> None:
    """A cached Rust summary must not bypass request/database validation."""
    monkeypatch.delenv("AICONFIGURATOR_ENGINE_STEP_BACKEND", raising=False)
    monkeypatch.setattr(
        backend,
        "run_mixed",
        lambda *args, **kwargs: StepEstimate(latency_ms=10.0, energy_wms=100.0),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *args, **kwargs: _decode_step(5.0, 50.0, {"decode": 5.0}, {"decode": "silicon"}, ()),
    )

    cached = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend="rust"),
        ctx_tokens=8,
    )
    assert (
        backend.run_agg(
            model,
            database,
            RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend="rust"),
            ctx_tokens=8,
        )
        is cached
    )

    with pytest.raises(error, match=match):
        backend.run_agg(
            model,
            database,
            RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, engine_step_backend=engine_step_backend),
            ctx_tokens=8,
        )


@pytest.mark.parametrize("progress", [0.0, 3.0, float("inf"), float("nan")])
def test_run_agg_rejects_invalid_speculative_progress(
    progress,
    backend: BaseBackend,
    model,
    database,
) -> None:
    model._nextn = 1

    with pytest.raises(ValueError, match="decode_tokens_per_iteration"):
        backend.run_agg(
            model,
            database,
            RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5),
            ctx_tokens=8,
            decode_tokens_per_iteration=progress,
        )


def _vision_encoder_config() -> common.VisionEncoderConfig:
    return common.VisionEncoderConfig(
        depth=1,
        hidden_size=8,
        num_heads=1,
        intermediate_size=8,
        patch_size=14,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=8,
    )


def test_run_mixed_derives_effective_multimodal_isl_for_direct_calls(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """Regression: a direct run_mixed call with a plain (text-only isl) config
    must hand the bridge the image-augmented effective isl, exactly as
    run_static / run_agg model it. Text isl=8 plus 16 visual tokens gives
    effective isl 24 — the visual adjustment happens before the bridge call,
    not inside the engine."""
    from aisim.sdk.backends import base_backend as base_backend_module

    model.encoder_config = _vision_encoder_config()

    calls: list[dict] = []

    def _fake_mixed_breakdown(model_arg, database_arg, **kwargs):
        calls.append(kwargs)
        return {
            "latency_ms": 1.0,
            "energy_wms": 1.0,
            "component_latency_ms": {"shared_non_attention": 1.0},
            "component_energy_wms": {"shared_non_attention": 1.0},
            "per_op_latency_ms": {},
            "per_op_source": {},
            "moe_comm_fallbacks": (),
        }

    monkeypatch.setattr(base_backend_module, "estimate_mixed_step_breakdown_with_rust", _fake_mixed_breakdown)

    backend.run_mixed(
        model,
        database,
        RuntimeConfig(isl=8, osl=6, num_images_per_request=1, num_image_tokens=16, engine_step_backend="rust"),
        MixedStepInput(context_tokens=24, num_decode_requests=1),
    )

    assert calls[-1]["isl"] == 24
    assert calls[-1]["ctx_tokens"] == 24


def test_run_agg_does_not_double_count_visual_tokens_in_run_mixed(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """run_mixed owns the visual-token adjustment, so run_agg must hand it the
    unmodified runtime_config (a pre-adjusted copy would count images twice),
    while the genonly step keeps receiving run_agg's effective isl."""
    model.encoder_config = _vision_encoder_config()

    mixed_configs: list[RuntimeConfig] = []

    def _run_mixed(model_arg, database_arg, runtime_config_arg, step):
        mixed_configs.append(runtime_config_arg)
        return StepEstimate(latency_ms=1.0, energy_wms=1.0)

    genonly_isl: list[int] = []

    def _genonly(model_arg, database_arg, runtime_config_arg, num_tokens, isl, osl):
        genonly_isl.append(isl)
        return _decode_step(1.0, 1.0, {}, {}, ())

    monkeypatch.setattr(backend, "run_mixed", _run_mixed)
    monkeypatch.setattr(backend, "_get_genonly_step_estimate", _genonly)

    runtime_config = RuntimeConfig(
        batch_size=2,
        beam_width=1,
        isl=8,
        osl=5,
        num_images_per_request=1,
        num_image_tokens=16,
        engine_step_backend="rust",
    )
    backend.run_agg(model, database, runtime_config, ctx_tokens=8)

    assert mixed_configs[0] is runtime_config
    assert genonly_isl == [8 + 16]


def test_mixed_step_requires_context_tokens() -> None:
    with pytest.raises(ValueError, match="context_tokens"):
        MixedStepInput(context_tokens=0, num_decode_requests=1)


def test_mix_step_efficiency_base_default_is_one(backend: BaseBackend) -> None:
    assert backend._mix_step_efficiency(ctx_tokens=4096, gen_tokens=16) == 1.0
    assert backend._mix_step_efficiency(ctx_tokens=4096, gen_tokens=0) == 1.0
    assert backend._mix_step_efficiency(ctx_tokens=0, gen_tokens=0) == 1.0


def test_run_static_latency_only_zeroes_energy_with_paired_keys(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    """include_energy=False must zero the energy dicts while keeping their
    key sets identical to the latency dicts (the power coverage gate pairs
    latency and energy by name)."""
    from aisim.sdk.backends import base_backend as base_backend_module

    monkeypatch.setattr(
        base_backend_module,
        "estimate_static_latency_breakdown_with_rust",
        lambda *args, **kwargs: (
            {"context_qkv_gemm": 4.0, "context_attention": 3.0},
            {"generation_qkv_gemm": 2.0, "generation_attention": 1.0},
            {"context_qkv_gemm": 40.0, "context_attention": 8.0},
            {"generation_qkv_gemm": 12.0, "generation_attention": 5.0},
            {"context_qkv_gemm": "silicon", "context_attention": "empirical"},
            {"generation_qkv_gemm": "silicon", "generation_attention": "mixed"},
            (),
        ),
    )

    runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2, engine_step_backend="rust")
    latency = backend.run_static_latency_only(
        model,
        database,
        runtime_config,
        mode="static",
        stride=2,
    )
    assert latency == pytest.approx(10.0)

    (
        context_latency,
        context_energy,
        generation_latency,
        generation_energy,
        _,
        _,
        _,
    ) = backend._run_static_breakdown(model, database, runtime_config, mode="static", stride=2, include_energy=False)
    # Latency-only callers must not observe energy, but the key sets stay
    # paired with the latency dicts (the power coverage gate pairs by name).
    assert context_energy == {"context_qkv_gemm": 0.0, "context_attention": 0.0}
    assert generation_energy == {"generation_qkv_gemm": 0.0, "generation_attention": 0.0}
    assert context_energy.keys() == context_latency.keys()
    assert generation_energy.keys() == generation_latency.keys()


def test_step_requires_a_real_perf_database(
    backend: BaseBackend,
    model,
    database,
) -> None:
    """The compiled engine resolves perf data from disk by identity, so a
    step over a synthetic database is a hard TypeError now that the Python
    step branch is gone (op-level and FPM models alike)."""
    with pytest.raises(TypeError, match="on-disk identity"):
        backend.run_static(
            model,
            database,
            RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2),
            mode="static",
            stride=2,
        )
    with pytest.raises(TypeError, match="on-disk identity"):
        backend.run_mixed(
            model,
            database,
            RuntimeConfig(isl=8, osl=5),
            MixedStepInput(context_tokens=8, num_decode_requests=1),
        )
    with pytest.raises(TypeError, match="on-disk identity"):
        backend._get_genonly_step_latency(
            model,
            database,
            RuntimeConfig(isl=8, osl=5),
            gen_tokens=2,
            isl=8,
            osl=5,
        )


@pytest.mark.parametrize("encoder_energy,expected_coverage,published", [(1000.0, 0.9125, True), (0.0, 0.7875, False)])
def test_run_agg_retains_schedule_weighted_energy_for_publication(
    backend, model, database, monkeypatch, encoder_energy, expected_coverage, published
):
    from aisim.legacy_cli.api import _apply_power_coverage_gate

    monkeypatch.setattr(backend, "_mix_step_efficiency", lambda *_: 0.5)
    monkeypatch.setattr(
        backend,
        "_run_encoder_phase",
        lambda *_: ({"encoder": 10.0}, {"encoder": encoder_energy}, {"encoder": "silicon"}, 0),
    )
    mixed = StepEstimate(
        latency_ms=10.0,
        energy_wms=9000.0,
        context_tokens=8,
        num_decode_requests=1,
        covered_latency_ms=9.0,
        per_op_latency_ms={"gemm": 9.0, "missing": 1.0},
        per_op_energy_wms={"gemm": 9000.0, "missing": 0.0},
    )
    decode = StepEstimate(
        latency_ms=20.0,
        energy_wms=18000.0,
        covered_latency_ms=18.0,
        per_op_latency_ms={"gemm": 18.0, "missing": 2.0},
        per_op_energy_wms={"gemm": 18000.0, "missing": 0.0},
    )
    monkeypatch.setattr(backend, "run_mixed", lambda *_: mixed)
    monkeypatch.setattr(backend, "_get_genonly_step_estimate", lambda *_: decode)
    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, isl=8, osl=5, engine_step_backend="rust"),
        ctx_tokens=8,
    )
    groups = summary.get_aggregate_energy_breakdown()
    assert groups["mix_step"].latency_ms == 10.0  # two mixed steps, each scaled by 0.5
    assert groups["genonly_step"].latency_ms == 60.0  # three decode-only steps
    assert groups["encoder"].latency_ms == 10.0
    assert groups["mix_step"].per_op_energy_wms == {"gemm": 9000.0, "missing": 0.0}
    assert groups["genonly_step"].per_op_energy_wms == {"gemm": 54000.0, "missing": 0.0}
    assert summary.get_power_data_coverage() == pytest.approx(expected_coverage)
    raw = summary.get_result_dict()
    assert raw["power_w"] == pytest.approx((63000.0 + encoder_energy) / 80.0)
    gated = _apply_power_coverage_gate(summary, raw)
    assert (gated["power_w"] is not None) is published
    assert {k: v for k, v in gated.items() if not k.startswith("power_")} == {
        k: v for k, v in raw.items() if not k.startswith("power_")
    }


@pytest.mark.parametrize("covered,expected,published", [(9.0, 0.9, True), (8.9, 0.89, False), (0.0, 0.0, False)])
def test_run_agg_zero_decode_steps_do_not_change_energy_coverage(
    backend, model, database, monkeypatch, covered, expected, published
):
    from aisim.legacy_cli.api import _apply_power_coverage_gate

    mixed = StepEstimate(latency_ms=10.0, energy_wms=1000.0 if covered else 0.0, covered_latency_ms=covered)
    unused_decode = StepEstimate(latency_ms=1000.0, energy_wms=1000000.0, covered_latency_ms=1000.0)
    monkeypatch.setattr(backend, "run_mixed", lambda *_: mixed)
    monkeypatch.setattr(backend, "_get_genonly_step_estimate", lambda *_: unused_decode)
    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=1, isl=8, osl=1, engine_step_backend="rust"),
        ctx_tokens=8,
    )
    assert summary.get_aggregate_energy_breakdown()["genonly_step"].latency_ms == 0
    assert summary.get_power_data_coverage() == pytest.approx(expected)
    assert (_apply_power_coverage_gate(summary, summary.get_result_dict())["power_w"] is not None) is published


@pytest.mark.parametrize(
    "prefix,budget,expected_chunks,qf",
    [
        (2048, 1024, [(3072, 2048), (4096, 3072)], 1.95),
        (1792, 1024, [(2816, 1792), (3840, 2816), (4096, 3840)], 2.00),
        (3000, 2048, [(4096, 3000)], 1.90),
    ],
)
def test_run_agg_single_request_counts_every_prefill_chunk(
    backend, model, database, monkeypatch, prefix, budget, expected_chunks, qf
):
    # A synthetic per-chunk oracle: 1 ms launch + 0.01 ms/new token, with
    # 10 W-ms launch + 0.2 W-ms/new token. Decode is one 3 ms, 90 W-ms step.
    # The last partial chunk pays its own launch and only its remaining tokens.
    calls = []
    raw_steps = []

    def prefill(_model, _database, runtime, step):
        calls.append((runtime.isl, runtime.prefix))
        tokens = runtime.isl - runtime.prefix
        latency, energy = 1 + tokens * 0.01, 10 + tokens * 0.2
        estimate = StepEstimate(
            latency_ms=latency,
            energy_wms=energy,
            covered_latency_ms=latency,
            per_op_latency_ms={"prefill": latency},
            per_op_energy_wms={"prefill": energy},
            per_op_source={"prefill": "silicon"},
            context_tokens=step.context_tokens,
        )
        raw_steps.append(estimate)
        return estimate

    monkeypatch.setattr(backend, "run_mixed", prefill)
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *_: _decode_step(3.0, 90.0, {"decode": 3.0}, {"decode": "silicon"}, ()),
    )
    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=1, isl=4096, prefix=prefix, osl=2, engine_step_backend="rust"),
        ctx_tokens=budget,
    )
    assert calls == expected_chunks
    count = len(expected_chunks)
    prefill_ms, prefill_energy = count + (4096 - prefix) * 0.01, count * 10 + (4096 - prefix) * 0.2
    result = summary.get_result_dict()
    assert result["ttft"] == pytest.approx(
        prefill_ms * qf
    )  # unchanged base queue heuristic, at the uncached step count
    assert result["tokens/s"] == pytest.approx(1000 / (prefill_ms + 3))
    groups = summary.get_aggregate_energy_breakdown()
    assert groups["mix_step"].latency_ms == pytest.approx(prefill_ms)
    assert groups["mix_step"].energy_wms == pytest.approx(prefill_energy)
    assert groups["mix_step"].covered_latency_ms == pytest.approx(prefill_ms)
    evidence = summary.get_step_estimates()
    assert evidence["scheduling"]["mix_context_token_budget"] == budget
    assert evidence["scheduling"]["mix_step_latency_ms"] == pytest.approx(prefill_ms / count)
    assert evidence["scheduling"]["mix_step_energy_wms"] == pytest.approx(prefill_energy / count)
    if count > 1:
        assert "mixed" not in evidence
        assert "mix_context_tokens" not in evidence["scheduling"]
        assert groups["mix_step"].context_tokens == 0
        assert [step.context_tokens for step in raw_steps] == [isl for isl, _ in expected_chunks]
    else:
        assert evidence["mixed"] is raw_steps[0]
        assert evidence["mixed"].context_tokens == 4096
        assert evidence["scheduling"]["mix_context_tokens"] == 4096
    assert result["power_w"] == pytest.approx((prefill_energy + 90) / (prefill_ms + 3))


@pytest.mark.parametrize("sources", [("silicon", "sol"), ("silicon", "sol", "silicon"), ("sol", "sol")])
def test_run_agg_chunk_sources_preserve_mixed_provenance(backend, model, database, monkeypatch, sources):
    labels = iter(sources)

    def prefill(*_):
        return StepEstimate(
            latency_ms=1.0,
            energy_wms=0.0,
            per_op_latency_ms={"context_prefill": 1.0},
            per_op_source={"context_prefill": next(labels)},
        )

    monkeypatch.setattr(backend, "run_mixed", prefill)
    monkeypatch.setattr(backend, "_get_genonly_step_estimate", lambda *_: _decode_step(1, 0, {}, {}, ()))
    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=1, isl=1024 * len(sources), osl=2, engine_step_backend="rust"),
        ctx_tokens=1024,
    )
    assert summary.get_per_ops_source()["mix_step"]["context_prefill"] == ("sol" if len(set(sources)) == 1 else "mixed")


@pytest.mark.parametrize(
    "prefix,budget,expected_steps,qf",
    [
        (2048, 1024, 4, 1.95),
        (1792, 1024, 4, 2.00),  # partial final uncached chunk
        (2816, 512, 8, 2.00),
        (0, 1024, 4, 2.05),
        (3000, 2048, 1, 1.90),  # all uncached tokens fit
        (2048, 4096, 1, 1.90),
    ],
)
def test_run_agg_visual_fallback_preserves_whole_prefill_attention(
    backend, model, database, monkeypatch, prefix, budget, expected_steps, qf
):
    from aisim.sdk.backends import base_backend as base_backend_module

    # Effective ISL is 4096: 3072 text + 1024 image tokens. The unchanged
    # native mean-field contract amortizes a whole-prefill attention price
    # over ceil(effective ISL / budget), even with a cached prefix. Use an
    # independent synthetic price of 12 ms / 1200 W-ms for that whole query,
    # 4 ms / 400 W-ms of visual-mask work, and 2 ms / 200 W-ms shared per step.
    model.encoder_config = _vision_encoder_config()
    model.visual_context_ops = [_StaticOp("visual_attention", 4.0, 400.0)]
    calls = []

    def native_breakdown(_model, _database, **kwargs):
        calls.append(kwargs)
        assert kwargs["isl"] == 4096
        scale = -(-4096 // kwargs["ctx_tokens"])
        latency = {"shared_non_attention": 2.0, "context_attention": 12.0 / scale}
        energy = {"shared_non_attention": 200.0, "context_attention": 1200.0 / scale}
        return {
            "latency_ms": sum(latency.values()),
            "energy_wms": sum(energy.values()),
            "component_latency_ms": latency,
            "component_energy_wms": energy,
            "per_op_latency_ms": {"shared": 2.0, "context_attention (scaled)": 12.0 / scale},
            "per_op_energy_wms": {"shared": 200.0, "context_attention (scaled)": 1200.0 / scale},
            "per_op_source": {"shared": "silicon", "context_attention (scaled)": "silicon"},
            "covered_latency_ms": sum(latency.values()),
            "moe_comm_fallbacks": (),
        }

    monkeypatch.setattr(base_backend_module, "estimate_mixed_step_breakdown_with_rust", native_breakdown)
    monkeypatch.setattr(
        backend,
        "_run_visual_context_phase",
        lambda *_: ({"visual_attention": 4.0}, {"visual_attention": 400.0}, {"visual_attention": "silicon"}),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_estimate",
        lambda *_: _decode_step(3.0, 90.0, {"decode": 3.0}, {"decode": "silicon"}, ()),
    )
    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(
            batch_size=1,
            isl=3072,
            prefix=prefix,
            osl=2,
            num_images_per_request=1,
            num_image_tokens=1024,
            engine_step_backend="rust",
        ),
        ctx_tokens=budget,
    )
    assert len(calls) == 1  # retain the visual mean-field fallback
    prefill_ms = 16.0 + 2.0 * expected_steps
    prefill_energy = 1600.0 + 200.0 * expected_steps
    result = summary.get_result_dict()
    assert result["ttft"] == pytest.approx(prefill_ms * qf)
    assert result["tokens/s"] == pytest.approx(1000 / (prefill_ms + 3))
    group = summary.get_aggregate_energy_breakdown()["mix_step"]
    assert group.per_op_latency_ms["context_attention (scaled)"] == pytest.approx(12.0)
    assert group.per_op_latency_ms["visual_attention"] == pytest.approx(4.0)
    assert group.per_op_energy_wms["context_attention (scaled)"] == pytest.approx(1200.0)
    assert group.per_op_energy_wms["visual_attention"] == pytest.approx(400.0)
    assert group.latency_ms == pytest.approx(prefill_ms)
    assert group.energy_wms == pytest.approx(prefill_energy)
    assert group.covered_latency_ms == pytest.approx(prefill_ms)
    assert result["power_w"] == pytest.approx((prefill_energy + 90) / (prefill_ms + 3))


@pytest.mark.parametrize("prefix,budget", [(2048, 1024), (2816, 512)])
def test_run_agg_visual_fallback_retains_native_attention(backend, prefix, budget, monkeypatch):
    from collections import OrderedDict

    from aisim_core.sdk import rust_engine_step
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.perf_database import get_database_view
    from aisim_core.sdk.rust_engine_step import estimate_static_latency_breakdown_with_rust

    monkeypatch.setattr(rust_engine_step, "_ENGINE_HANDLE_CACHE", OrderedDict())

    # Isolate real native attention from a bundled graph; no weights or
    # measured data are needed. Its static whole-prefill query is the
    # independent, unamortized reference for the aggregate consumer.
    model = get_model("Qwen/Qwen3-32B", ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1), "vllm")
    model.context_ops = [op for op in model.context_ops if op._name == "context_attention"]
    assert model.context_ops
    model.encoder_config = _vision_encoder_config()
    model.encoder_ops = []
    database = get_database_view("b200_sxm", "vllm", "0.24.0", database_mode="SOL", allow_unlisted_version=True)
    assert database is not None
    reference = estimate_static_latency_breakdown_with_rust(
        model, database, RuntimeConfig(batch_size=1, isl=4096, prefix=prefix, osl=1), "static_ctx", 1, 1.0
    )
    attention_ms = sum(reference[0].values())
    assert attention_ms > 0
    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(
            batch_size=1,
            isl=3072,
            prefix=prefix,
            osl=1,
            num_images_per_request=1,
            num_image_tokens=1024,
        ),
        ctx_tokens=budget,
    )
    assert summary.get_aggregate_energy_breakdown()["mix_step"].latency_ms == pytest.approx(attention_ms)


@pytest.mark.parametrize("visual_tokens", [0, 1000])
def test_xpu_aggregate_exact_ttft_total_is_text_only(monkeypatch, visual_tokens):
    from aisim.sdk.backends import base_backend as base_backend_module
    from aisim_core.sdk.backends.vllm_backend_xpu import VLLMXPUBackend
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.perf_database import get_database_view

    backend = VLLMXPUBackend()
    model = get_model("Qwen/Qwen3-0.6B", ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1), "vllm")
    database = get_database_view("b200_sxm", "vllm", "0.24.0", database_mode="SOL", allow_unlisted_version=True)
    assert database is not None
    model._num_layers = 0
    model.encoder_config = _vision_encoder_config()
    model.visual_context_ops = []
    monkeypatch.setattr(backend, "_get_memory_usage", lambda *a, **k: {"total": 1.0})
    monkeypatch.setattr(backend, "_max_kv_slots", lambda *a, **k: 2)
    monkeypatch.setattr(
        backend,
        "_run_encoder_phase",
        lambda *_: ({"encoder": 3.0}, {"encoder": 0.0}, {"encoder": "estimated"}, visual_tokens),
    )
    monkeypatch.setattr(backend, "_get_encoder_component_memory_for_runtime", lambda *_: {})
    monkeypatch.setattr(backend, "_get_genonly_step_estimate", lambda *_: StepEstimate(0.0, 0.0))

    def native_breakdown(_model, _database, **kwargs):
        scale = -(-kwargs["isl"] // kwargs["ctx_tokens"])
        return {
            "latency_ms": 2.0 + 12.0 / scale,
            "energy_wms": 0.0,
            "component_latency_ms": {"shared_non_attention": 2.0, "context_attention": 12.0 / scale},
            "component_energy_wms": {},
            "per_op_latency_ms": {"shared": 2.0, "context_attention (scaled)": 12.0 / scale},
            "per_op_source": {},
            "moe_comm_fallbacks": (),
        }

    monkeypatch.setattr(base_backend_module, "estimate_mixed_step_breakdown_with_rust", native_breakdown)
    totals = []
    compute_ttft = backend._compute_ttft

    def record_ttft(*args, **kwargs):
        totals.append(kwargs.get("prefill_total_ms"))
        return compute_ttft(*args, **kwargs)

    monkeypatch.setattr(backend, "_compute_ttft", record_ttft)
    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(
            batch_size=1,
            isl=3000 - visual_tokens,
            osl=1,
            num_images_per_request=int(visual_tokens > 0),
            num_image_tokens=visual_tokens,
        ),
        ctx_tokens=2048,
    )
    # Invented attention price = 12 ms per query, shared work = 2 ms per
    # step, encoder = 3 ms. Visual mean-field attention undoes the /2 and
    # shared work scales by 3000/2048. Exact text pays two 14 ms queries,
    # including the short final chunk, without applying that scaling.
    prefill_ms = 12.0 + 2.0 * 3000 / 2048 if visual_tokens else 28.0
    assert summary.get_result_dict()["ttft"] == pytest.approx(3.0 + prefill_ms, abs=1e-3)
    assert totals == ([None] if visual_tokens else [prefill_ms])


@pytest.mark.parametrize("prefix,budget,chunks,qf", [(3000, 4096, 1, 1.9), (3000, 512, 3, 2.0), (2048, 1024, 2, 1.95)])
def test_run_agg_visual_fpm_keeps_uncached_step_count(
    backend, model, database, monkeypatch, prefix, budget, chunks, qf
):
    from aisim.sdk.backends import base_backend as base_backend_module

    # FPM's synthetic 5 ms/500 W-ms mean prices the fresh suffix. A nonzero
    # 4 ms/400 W-ms whole-request visual overlay must survive the real
    # run_mixed -> run_agg -> summary path exactly once, including a short tail.
    model.forward_model = "fpm"
    model.encoder_config = _vision_encoder_config()
    model.visual_context_ops = [_StaticOp("visual_attention", 4.0, 400.0)]
    seen = []

    def native_breakdown(_model, _database, **kwargs):
        seen.append(kwargs["ctx_tokens"])
        return {
            "latency_ms": 5.0,
            "energy_wms": 500.0,
            "component_latency_ms": {"shared_non_attention": 5.0},
            "component_energy_wms": {"shared_non_attention": 500.0},
            "per_op_latency_ms": {"fpm": 5.0},
            "per_op_energy_wms": {"fpm": 500.0},
            "per_op_source": {"fpm": "silicon"},
            "covered_latency_ms": 5.0,
            "moe_comm_fallbacks": (),
        }

    monkeypatch.setattr(base_backend_module, "estimate_mixed_step_breakdown_with_rust", native_breakdown)
    monkeypatch.setattr(
        backend,
        "_run_visual_context_phase",
        lambda *_: ({"visual_attention": 4.0}, {"visual_attention": 400.0}, {"visual_attention": "silicon"}),
    )
    monkeypatch.setattr(backend, "_get_genonly_step_estimate", lambda *_: StepEstimate(latency_ms=0.0, energy_wms=0.0))
    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(
            batch_size=1,
            isl=3072,
            prefix=prefix,
            osl=1,
            num_images_per_request=1,
            num_image_tokens=1024,
            engine_step_backend="rust",
        ),
        ctx_tokens=budget,
    )
    assert seen == [min(budget, 4096 - prefix)]
    total_ms = 5.0 * chunks + 4.0
    group = summary.get_aggregate_energy_breakdown()["mix_step"]
    assert summary.get_result_dict()["ttft"] == pytest.approx(total_ms * qf)
    assert group.latency_ms == pytest.approx(total_ms)
    assert group.energy_wms == pytest.approx(total_ms * 100)
    assert group.covered_latency_ms == pytest.approx(total_ms)
    assert group.per_op_latency_ms["visual_attention"] == pytest.approx(4.0)
    assert group.per_op_energy_wms["visual_attention"] == pytest.approx(400.0)


@pytest.mark.parametrize("fresh_tokens", [2000, 2192])
@pytest.mark.parametrize("xpu", [False, True])
def test_batched_fpm_visual_overlay_preserves_fractional_requests(
    backend, model, database, monkeypatch, fresh_tokens, xpu
):
    from aisim.sdk.backends import base_backend as base_backend_module

    if xpu:
        from aisim_core.sdk.backends.vllm_backend_xpu import VLLMXPUBackend

        backend = VLLMXPUBackend()
        monkeypatch.setattr(backend, "_max_kv_slots", lambda *a, **k: 100)
        monkeypatch.setattr(backend, "_get_memory_usage", lambda *a, **k: {"total": 1.0})
        # Isolate aggregate overlay accounting from XPU's calibrated TTFT
        # queries, which add their own independently scheduled overlays.
        monkeypatch.setattr(backend, "_compute_ttft", lambda *a, **k: 1.0)
    model.forward_model = "fpm"
    model.encoder_config = _vision_encoder_config()
    model.visual_context_ops = [_StaticOp("visual_attention", 1.0, 100.0)]
    # Admit two prefills alongside two decoders; isolate the native visual
    # pricing seam with a deliberately nonlinear batch price (batch squared).
    monkeypatch.setattr(backend, "_mix_step_gen_tokens", lambda *_: 2)
    monkeypatch.setattr(backend, "_get_genonly_step_estimate", lambda *_: StepEstimate(0.0, 0.0))
    monkeypatch.setattr(
        base_backend_module,
        "estimate_mixed_step_breakdown_with_rust",
        lambda *a, **kw: {
            "latency_ms": 5.0,
            "energy_wms": 0.0,
            "component_latency_ms": {"shared_non_attention": 5.0},
            "component_energy_wms": {},
            "per_op_latency_ms": {"fpm": 5.0},
            "per_op_source": {"fpm": "silicon"},
            "moe_comm_fallbacks": (),
        },
    )
    batches = []

    def visual_phase(_model, _database, _runtime, batch):
        batches.append(batch)
        return (
            {"visual_attention": float(batch**2)},
            {"visual_attention": float(100 * batch**2)},
            {"visual_attention": "silicon"},
        )

    monkeypatch.setattr(backend, "_run_visual_context_phase", visual_phase)
    # The aggregate accepts prompt capacity: 4096 + 904 -> 2000 fresh
    # tokens, or two full 4096-token prompts -> 2192 fresh tokens.
    budget = 4096 + fresh_tokens - 1096 if fresh_tokens < 2192 else 8192
    if xpu:
        budget += 2  # Reserve the two width-one decoders before translation.
    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(
            batch_size=4,
            isl=3072,
            prefix=3000,
            osl=64,
            num_images_per_request=1,
            num_image_tokens=1024,
            engine_step_backend="rust",
        ),
        ctx_tokens=budget,
    )
    steps = summary.get_step_estimates()["scheduling"]["num_mix_steps"]
    # The representative query holds 1 + 904/1096 requests, or exactly 2.
    # A mixture of batch 1 and 2 prices 1 + 3*904/1096 ms (not 4 ms).
    fraction = (fresh_tokens - 1096) / 1096
    assert steps * ((1 - fraction) + 2 * fraction) == pytest.approx(4)
    expected = steps * (1 + 3 * fraction)
    group = summary.get_aggregate_energy_breakdown()["mix_step"]
    assert group.per_op_latency_ms["visual_attention"] == pytest.approx(expected)
    assert group.per_op_energy_wms["visual_attention"] == pytest.approx(100 * expected)
    assert group.covered_latency_ms == pytest.approx(expected)
    assert batches == ([1, 2] if fresh_tokens == 2000 else [2])
    evidence = summary.get_step_estimates()
    assert ("mixed" in evidence) == (fresh_tokens == 2192)
    assert ("mix_context_tokens" in evidence["scheduling"]) == (fresh_tokens == 2192)
    if fresh_tokens == 2192:
        assert evidence["mixed"].context_tokens == 2192
    # A public scheduled query has ceil(fresh/suffix) real requests. It must
    # retain the native batch-two overlay rather than aggregate weighting.
    direct = backend.run_mixed(
        model,
        database,
        RuntimeConfig(
            batch_size=4,
            isl=3072,
            prefix=3000,
            osl=64,
            num_images_per_request=1,
            num_image_tokens=1024,
            engine_step_backend="rust",
        ),
        MixedStepInput(fresh_tokens, 2),
    )
    assert direct.per_op_latency_ms["visual_attention"] == pytest.approx(4.0)
    assert batches[-1] == 2


@pytest.mark.parametrize("native_op", [False, True])
@pytest.mark.parametrize("phase", ["context", "generation"])
@pytest.mark.parametrize("prefix", [0, 3000])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_run_agg_rejects_chunked_visual_qualified_mamba(
    backend, model, database, monkeypatch, native_op, phase, prefix, batch_size
):
    from aisim_core import _native
    from aisim_core.sdk.operations import Mamba2Kernel

    mamba_type = _native.Mamba2Kernel if native_op else Mamba2Kernel
    scan = mamba_type(
        f"{phase}_mamba_ssm",
        1,
        "mamba_chunk_scan_combined" if phase == "context" else "selective_state_update",
        phase,
        4096,
        64,
        64,
        128,
        4,
        8,
        128,
        state_dtype="float16",
    )
    getattr(model, f"{phase}_ops").append(scan)
    model.encoder_config = _vision_encoder_config()
    # Fail before any native prediction or encoder work: the visual mean-field
    # weight cannot price 512, 512, 72 fresh tokens as three scan state pairs.
    monkeypatch.setattr(backend, "run_mixed", lambda *_: pytest.fail("unsupported graph was priced"))
    with pytest.raises(ValueError, match="qualified Mamba with chunked visual prefill is unsupported"):
        backend.run_agg(
            model,
            database,
            RuntimeConfig(
                batch_size=batch_size, isl=3072, prefix=prefix, osl=1, num_images_per_request=1, num_image_tokens=1024
            ),
            ctx_tokens=512,
        )


@pytest.mark.parametrize("xpu", [False, True])
@pytest.mark.parametrize("budget,scheduled,new_tokens,prefills", [(2048, 4096, 1096, 1), (6000, 8192, 2192, 2)])
def test_qualified_mamba_aggregate_translates_capacity_before_native_validation(
    monkeypatch, xpu, budget, scheduled, new_tokens, prefills
):
    from aisim_core.sdk.backends.factory import get_backend
    from aisim_core.sdk.backends.vllm_backend_xpu import VLLMXPUBackend
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.operations import ElementWise, Mamba2Kernel
    from aisim_core.sdk.perf_database import get_database_view

    database = get_database_view("b200_sxm", "vllm", "0.24.0", database_mode="SOL", allow_unlisted_version=True)
    assert database is not None
    model = get_model("Qwen/Qwen3-0.6B", ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1), "vllm")
    model.context_ops = [
        Mamba2Kernel(
            "context_scan",
            1,
            "mamba_chunk_scan_combined",
            "context",
            4096,
            64,
            64,
            128,
            4,
            4,
            128,
            state_dtype="float16",
        ),
        ElementWise("norm", 1, 1, 0),
    ]
    model.generation_ops = []
    backend = VLLMXPUBackend() if xpu else get_backend("vllm")
    monkeypatch.setattr(backend, "_get_memory_usage", lambda *a, **k: {"total": 1.0})
    if xpu:
        monkeypatch.setattr(backend, "_max_kv_slots", lambda *a, **k: 100)
    runtime = RuntimeConfig(batch_size=4, isl=4096, prefix=3000, osl=128)
    summary = backend.run_agg(model, database, runtime, ctx_tokens=budget)
    mixed = summary.get_step_estimates()["mixed"]
    assert mixed.context_tokens == scheduled
    assert summary.get_step_estimates()["scheduling"]["mix_context_token_budget"] == budget
    # Hand-derived SOL traffic: 18,560 activation bytes per new token,
    # one 1 MiB state read/write per prefill, and 2 bytes per norm token.
    bw = database.system_spec["gpu"]["mem_bw"]
    assert mixed.per_op_latency_ms["context_scan"] == pytest.approx(
        (new_tokens * 18_560 + prefills * 2_097_152) / bw * 1000
    )
    assert mixed.per_op_latency_ms["norm"] == pytest.approx(
        (new_tokens + mixed.num_decode_query_tokens) * 2 / bw * 1000
    )
    # Capacity translation must also reweight the complete aggregate: four
    # uncached suffixes, each with one state read/write, irrespective of how
    # many prefills fit the representative query.
    scheduling = summary.get_step_estimates()["scheduling"]
    assert scheduling["num_mix_steps"] == 4 / prefills
    group = summary.get_aggregate_energy_breakdown()["mix_step"]
    assert group.per_op_latency_ms["context_scan"] == pytest.approx((4 * 1096 * 18_560 + 4 * 2_097_152) / bw * 1000)
    # Public scheduled input keeps the strict boundary instead of silently
    # converting the same invalid shape as if it were scheduler capacity.
    with pytest.raises(ValueError, match="partial prefill extend"):
        backend.run_mixed(model, database, runtime, MixedStepInput(budget, 2))


def test_generation_only_mamba_qualification_uses_bounded_prefill_coordinates(backend):
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.operations import ElementWise, Mamba2Kernel
    from aisim_core.sdk.perf_database import get_database_view

    database = get_database_view("b200_sxm", "vllm", "0.24.0", database_mode="SOL", allow_unlisted_version=True)
    assert database is not None
    model = get_model("Qwen/Qwen3-0.6B", ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1), "vllm")
    model.context_ops = [ElementWise("norm", 1, 1, 0)]
    model.generation_ops = [
        Mamba2Kernel(
            "generation_scan",
            1,
            "selective_state_update",
            "generation",
            4096,
            64,
            64,
            128,
            4,
            4,
            128,
            state_dtype="float16",
        )
    ]
    runtime = RuntimeConfig(batch_size=4, isl=100, prefix=60, osl=128)
    mixed = backend._run_agg_mixed_step(model, database, runtime, MixedStepInput(150, 2))
    # Both phases select the strict split. Aggregate capacity 150 completes
    # two 40-token suffixes, encoded as two whole 100-token prompts, plus
    # two decoders: exactly 82 new positions, never 92 from an overlong chunk.
    assert mixed.context_tokens == 200
    assert mixed.per_op_latency_ms["norm"] == pytest.approx(82 * 2 / database.system_spec["gpu"]["mem_bw"] * 1000)
    with pytest.raises(ValueError, match="partial prefill extend"):
        backend.run_mixed(model, database, runtime, MixedStepInput(150, 2))


def _qualified_scan_model(backend_name):
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.operations import ElementWise, Mamba2Kernel

    model = get_model("Qwen/Qwen3-0.6B", ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1), backend_name)
    # A native micrograph isolates recurrent-state traffic and shared tokens.
    # Its scan reads/writes 18,560 B/token and 2 MiB/state; norm uses 2 B/token.
    model.context_ops = [
        Mamba2Kernel(
            "context_scan",
            1,
            "mamba_chunk_scan_combined",
            "context",
            4096,
            64,
            64,
            128,
            4,
            4,
            128,
            state_dtype="float16",
        ),
        ElementWise("norm", 1, 1, 0),
    ]
    model.generation_ops = [ElementWise("generation_norm", 1, 1, 0)]
    return model


@pytest.mark.parametrize("prefix,first_supported_budget", [(0, 4096), (3000, 1536)])
def test_visual_qualified_sweep_reaches_a_budget_that_fits_the_suffix(monkeypatch, prefix, first_supported_budget):
    from aisim_core.sdk.backends.factory import get_backend
    from aisim_core.sdk.perf_database import get_database_view

    backend = get_backend("trtllm")
    model = _qualified_scan_model("trtllm")
    model.encoder_config = _vision_encoder_config()
    database = get_database_view("b200_sxm", "trtllm", "1.3.0rc20", database_mode="SOL", allow_unlisted_version=True)
    monkeypatch.setattr(backend, "_get_memory_usage", lambda *a, **k: {"total": 1.0})
    runtime = RuntimeConfig(
        batch_size=1,
        isl=3072,
        prefix=prefix,
        osl=2,
        ttft=float("inf"),
        tpot=float("inf"),
        num_images_per_request=1,
        num_image_tokens=1024,
    )
    with pytest.raises(ValueError, match="qualified Mamba with chunked visual prefill is unsupported"):
        backend.run_agg(model, database, runtime, ctx_tokens=512)
    calls = []
    run_agg = backend.run_agg

    def record_candidate(**kwargs):
        calls.append(kwargs["ctx_tokens"])
        return run_agg(**kwargs)

    monkeypatch.setattr(backend, "run_agg", record_candidate)
    summary = backend.find_best_agg_result_under_constraints(
        model, database, runtime, max_batch_size=1, ctx_stride=512, enable_chunked_prefill=True, top_k=0
    )
    assert calls[0] == first_supported_budget
    assert all(budget >= 4096 - prefix for budget in calls)
    assert not summary.get_summary_df().empty
    assert not summary.check_oom()


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_batched_visual_mamba_sweep_continues_after_xpu_decode_reservation(monkeypatch, phase):
    from aisim_core.sdk.backends.vllm_backend import VLLMBackend
    from aisim_core.sdk.backends.vllm_backend_xpu import VLLMXPUBackend
    from aisim_core.sdk.perf_database import get_database_view

    backend = VLLMXPUBackend()
    model = _qualified_scan_model("vllm")
    if phase == "generation":
        # The unsupported visual chunk contract also covers decode-only
        # qualification, including the native prefill-coordinate bound.
        from aisim_core.sdk.operations import Mamba2Kernel

        model.context_ops = []
        model.generation_ops = [
            Mamba2Kernel(
                "generation_scan",
                1,
                "selective_state_update",
                "generation",
                4096,
                64,
                64,
                128,
                4,
                4,
                128,
                state_dtype="float16",
            )
        ]
    model.encoder_config = _vision_encoder_config()
    database = get_database_view("b200_sxm", "vllm", "0.24.0", database_mode="SOL")
    monkeypatch.setattr(backend, "_get_memory_usage", lambda *a, **k: {"total": 1.0})
    monkeypatch.setattr(backend, "_mix_step_gen_tokens", lambda *a: 2)
    monkeypatch.setattr(backend, "_max_kv_slots", lambda *a, **k: 100)
    # The 1096-token total budget fits the suffix before decode reservation.
    # Two admitted decoders leave only 1094 tokens; 1536 is a fitting control.
    monkeypatch.setattr(backend, "_get_ctx_tokens_list_for_agg_sweep", lambda *a: [1096, 1536, 4096])
    runtime = RuntimeConfig(
        batch_size=4,
        isl=3072,
        prefix=3000,
        osl=64,
        ttft=float("inf"),
        tpot=float("inf"),
        num_images_per_request=1,
        num_image_tokens=1024,
    )
    calls, priced = [], []
    run_agg, run_mixed = backend.run_agg, VLLMBackend.run_mixed

    def record_candidate(**kwargs):
        calls.append((kwargs["runtime_config"].batch_size, kwargs["ctx_tokens"]))
        return run_agg(**kwargs)

    def record_native(self, model, database, runtime, step):
        priced.append((step.context_tokens, step.num_decode_requests))
        assert step.context_tokens >= 1096
        return run_mixed(self, model, database, runtime, step)

    monkeypatch.setattr(backend, "run_agg", record_candidate)
    monkeypatch.setattr(VLLMBackend, "run_mixed", record_native)
    summary = backend.find_best_agg_result_under_constraints(
        model, database, runtime, max_batch_size=4, enable_chunked_prefill=True, top_k=0
    )
    assert (4, 1096) in calls and (4, 1536) in calls
    assert calls.index((4, 1536)) > calls.index((4, 1096))
    assert any(decoders == 2 for _, decoders in priced)
    assert not summary.get_summary_df().empty
    assert not summary.check_oom()


@pytest.mark.parametrize("prefix", [0, 3000])
def test_qualified_aggregate_caps_prefill_at_remaining_request_slots(monkeypatch, prefix):
    from aisim_core.sdk.backends.factory import get_backend
    from aisim_core.sdk.perf_database import get_database_view

    backend = get_backend("trtllm")
    model = _qualified_scan_model("trtllm")
    database = get_database_view("b200_sxm", "trtllm", "1.3.0rc20", database_mode="SOL")
    assert database is not None
    monkeypatch.setattr(backend, "_get_memory_usage", lambda *a, **k: {"total": 1.0})
    runtime = RuntimeConfig(batch_size=2, isl=4096, prefix=prefix, osl=128)
    summary = backend.run_agg(model, database, runtime, ctx_tokens=8192)
    mixed = summary.get_step_estimates()["mixed"]
    assert mixed.num_decode_requests == 1
    assert mixed.context_tokens == 4096
    row = summary.get_result_dict()
    assert (row["num_ctx_reqs"], row["num_gen_reqs"], row["gen_tokens"]) == (1, 1, 1)
    bw = database.system_spec["gpu"]["mem_bw"]
    scan_ms = ((4096 - prefix) * 18_560 + 2_097_152) / bw * 1000
    assert mixed.per_op_latency_ms["context_scan"] == pytest.approx(scan_ms)
    assert mixed.per_op_latency_ms["norm"] == pytest.approx((4096 - prefix + 1) * 2 / bw * 1000)
    # Direct mixed-step inputs describe scheduled work, not aggregate capacity:
    # two complete prefills remain valid even beside a decoder.
    direct = backend.run_mixed(model, database, runtime, MixedStepInput(8192, 1))
    assert direct.per_op_latency_ms["context_scan"] == pytest.approx(2 * scan_ms)
    assert direct.per_op_latency_ms["norm"] == pytest.approx((2 * (4096 - prefix) + 1) * 2 / bw * 1000)


@pytest.mark.parametrize("prefix,budget,chunks", [(3072, 512, 2), (3000, 512, 3), (3000, 1096, 1), (0, 512, 8)])
def test_qualified_batched_aggregate_weights_uncached_chunks(monkeypatch, prefix, budget, chunks):
    from aisim_core.sdk.backends.factory import get_backend
    from aisim_core.sdk.perf_database import get_database_view

    backend = get_backend("vllm")
    model = _qualified_scan_model("vllm")
    database = get_database_view("b200_sxm", "vllm", "0.24.0", database_mode="SOL")
    assert database is not None
    monkeypatch.setattr(backend, "_get_memory_usage", lambda *a, **k: {"total": 1.0})
    summary = backend.run_agg(
        model, database, RuntimeConfig(batch_size=4, isl=4096, prefix=prefix, osl=64), ctx_tokens=budget
    )
    scheduling = summary.get_step_estimates()["scheduling"]
    assert scheduling["num_mix_steps"] == 4 * chunks
    assert scheduling["num_genonly_steps"] == 64 - 4 * chunks
    group = summary.get_aggregate_energy_breakdown()["mix_step"]
    bw = database.system_spec["gpu"]["mem_bw"]
    # Four suffixes, one recurrent-state transfer per chunk (including the
    # 72-token tail at prefix=3000), and three decoders in every mixed step.
    scan_ms = (4 * (4096 - prefix) * 18_560 + 4 * chunks * 2_097_152) / bw * 1000
    shared_ms = (4 * (4096 - prefix) + 4 * chunks * 3) * 2 / bw * 1000
    assert group.per_op_latency_ms["context_scan"] == pytest.approx(scan_ms)
    assert group.per_op_latency_ms["norm"] == pytest.approx(shared_ms)
    mixed_total = scan_ms + shared_ms
    decode_ms = 4 * 2 / bw * 1000
    total_ms = mixed_total + (64 - 4 * chunks) * decode_ms
    tpot = total_ms / 64
    # vLLM's existing dispatch and queue calibration, followed by Little's law.
    ttft = (mixed_total / 4 + model._num_layers * 0.8) * 1.25
    expected_rate = min(4 * 63 * 1000 / total_ms, 4 * 63 * 1000 / (ttft + 63 * tpot))
    result = summary.get_result_dict()
    assert result["ttft"] == pytest.approx(ttft)
    assert result["tpot"] == pytest.approx(tpot)
    assert result["tokens/s"] == pytest.approx(expected_rate)
    row = summary.get_summary_df().iloc[0]
    assert row["num_ctx_reqs"] == 1
    assert row["num_gen_reqs"] == 3


def test_aggregate_cache_tracks_memory_only_mamba_config(monkeypatch):
    import dataclasses

    from aisim_core.sdk.backends.factory import get_backend
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.perf_database import get_database_view
    from aisim_core.sdk.rust_engine_step import _cached_engine_handle, engine_identity

    backend = get_backend("vllm")
    model = get_model(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1), "vllm"
    )
    assert model._nextn == 0
    database = get_database_view("b200_sxm", "vllm", "0.24.0", database_mode="SOL")
    assert database is not None
    runtime = RuntimeConfig(batch_size=2, isl=1024, osl=2)
    fp32 = backend.run_agg(model, database, runtime, ctx_tokens=1024)
    assert backend.run_agg(model, database, runtime, ctx_tokens=1024) is fp32
    timing_key = engine_identity(model, database)
    timing_handle = _cached_engine_handle(model, database)
    model.set_hybrid_config(dataclasses.replace(model._hybrid_config, mamba_ssm_cache_dtype="float16"))
    assert engine_identity(model, database) == timing_key
    assert _cached_engine_handle(model, database) is timing_handle
    fp16 = backend.run_agg(model, database, runtime, ctx_tokens=1024)
    cold_fp16 = get_backend("vllm").run_agg(model, database, runtime, ctx_tokens=1024)
    assert fp16 is not fp32
    assert fp16.get_memory() == cold_fp16.get_memory()
    assert fp16.get_memory()["kvcache"] < fp32.get_memory()["kvcache"]
    assert backend.run_agg(model, database, runtime, ctx_tokens=1024) is fp16

    # A capacity between the two real footprints makes the stale verdict
    # observable. Keep each database's capacity fixed throughout its run.
    capacity = int((fp16.get_memory()["total"] + fp32.get_memory()["total"]) * (1 << 29))
    monkeypatch.setitem(database.system_spec["gpu"], "mem_capacity", capacity)
    backend = get_backend("vllm")
    model.set_hybrid_config(dataclasses.replace(model._hybrid_config, mamba_ssm_cache_dtype="float32"))
    assert backend.run_agg(model, database, runtime, ctx_tokens=1024).check_oom()
    model.set_hybrid_config(dataclasses.replace(model._hybrid_config, mamba_ssm_cache_dtype="float16"))
    assert not backend.run_agg(model, database, runtime, ctx_tokens=1024).check_oom()


def test_qualified_chunk_averaging_keeps_other_requests_decode_kv(monkeypatch):
    from aisim_core.sdk.backends.factory import get_backend
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.perf_database import get_database_view

    backend = get_backend("vllm")
    model = _qualified_scan_model("vllm")
    attention_model = get_model("Qwen/Qwen3-0.6B", ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1), "vllm")
    model.generation_ops += [op for op in attention_model.generation_ops if op._name == "generation_attention"]
    database = get_database_view("b200_sxm", "vllm", "0.24.0", database_mode="SOL")
    assert database is not None
    runtime = RuntimeConfig(batch_size=4, isl=4096, prefix=3000, osl=64)
    # All twelve mixed iterations contain the same three decoders. Their KV
    # lengths are independent of the prefilling request's 512/512/72 progress.
    reference = backend.run_mixed(model, database, runtime, MixedStepInput(512, 3))
    assert reference.per_op_latency_ms["generation_attention"] > 0
    monkeypatch.setattr(backend, "_get_memory_usage", lambda *a, **k: {"total": 1.0})
    group = backend.run_agg(model, database, runtime, ctx_tokens=512).get_aggregate_energy_breakdown()["mix_step"]
    assert group.per_op_latency_ms["generation_attention"] == pytest.approx(
        12 * reference.per_op_latency_ms["generation_attention"]
    )


def test_qualified_aggregate_keeps_a_prefill_slot_at_equal_phase_lengths(monkeypatch):
    from aisim_core.sdk.backends.factory import get_backend
    from aisim_core.sdk.perf_database import get_database_view

    backend = get_backend("trtllm")
    model = _qualified_scan_model("trtllm")
    database = get_database_view("b200_sxm", "trtllm", "1.3.0rc20", database_mode="SOL")
    assert database is not None
    monkeypatch.setattr(backend, "_get_memory_usage", lambda *a, **k: {"total": 1.0})
    # Both phases take 128 iterations. The generic throughput heuristic would
    # admit both requests as decoders, leaving no slot for scheduled prefill.
    summary = backend.run_agg(model, database, RuntimeConfig(batch_size=2, isl=4096, osl=128), ctx_tokens=64)
    scheduling = summary.get_step_estimates()["scheduling"]
    assert scheduling["mix_decode_requests"] == 1
    assert scheduling["num_mix_steps"] == 128
    norm_ms = summary.get_aggregate_energy_breakdown()["mix_step"].per_op_latency_ms["norm"]
    assert norm_ms == pytest.approx((2 * 4096 + 128) * 2 / database.system_spec["gpu"]["mem_bw"] * 1000)


def test_fixed_native_decode_pass_keeps_folded_energy_and_coverage(monkeypatch):
    from aisim_core.sdk import rust_engine_step

    # Native pass entries: name, ms, W-ms, source. The prefilling request
    # advances, but a different decode request keeps its original 3 ms / 30
    # W-ms price. A zero-energy 5 ms op stays outside power coverage.
    handle = SimpleNamespace(
        _mixed_step_breakdown_per_op_with_decode_coverage=lambda *a, **k: (
            [("shared", 2.0, 20.0, "silicon"), ("uncovered", 5.0, 0.0, "sol")],
            [("context_scan", 1.0, 10.0, "silicon")],
            [("same_name", 0.25, 2.5, "sol")],
            3.25,
            0.25,
        )
    )
    monkeypatch.setattr(rust_engine_step, "_cached_engine_handle", lambda *_: handle)
    monkeypatch.setattr(rust_engine_step, "_note_rust_provenance", lambda *_: None)
    result = rust_engine_step.estimate_mixed_step_breakdown_with_rust(
        None,
        None,
        ctx_tokens=512,
        gen_tokens=3,
        isl=3512,
        osl=64,
        prefix=3000,
        _decode_ops=([("same_name", 3.0, 30.0, "silicon")], 3.0),
    )
    assert result["latency_ms"] == 11.0
    assert result["energy_wms"] == 60.0
    assert result["covered_latency_ms"] == 6.0
    assert result["per_op_latency_ms"]["same_name"] == 3.0
    assert result["per_op_energy_wms"]["same_name"] == 30.0
    assert result["per_op_source"]["same_name"] == "silicon"


@pytest.mark.parametrize(
    "prefix,budget,old_activation,new_activation,old_total,new_total",
    [
        (3000, 2048, 0.156250, 0.083618, 62.542492, 62.469860),
        (3000, 4096, 0.312500, 0.083618, 62.698742, 62.469860),
        (3000, 8192, 0.312500, 0.083618, 62.698742, 62.469860),
        (0, 2048, 0.156250, 0.156250, 62.542492, 62.542492),
    ],
)
def test_legacy_aggregate_memory_matches_recorded_suffix_evidence(
    prefix, budget, old_activation, new_activation, old_total, new_total
):
    # Historical values from mem-35e632cb.json and mem-1ea0211c.json.
    # This unqualified Nano has 81,920 activation bytes/token; cached work
    # changes from a budget-sized 2048/4096 tokens to the 1096-token suffix.
    from aisim_core.sdk.backends.factory import get_backend
    from aisim_core.sdk.config import ModelConfig, RuntimeConfig
    from aisim_core.sdk.inference_summary import InferenceSummary
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.perf_database import get_database_view

    database = get_database_view("b200_sxm", "vllm", "0.24.0", database_mode="HYBRID", allow_unlisted_version=True)
    model = get_model(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1),
        "vllm",
    )
    backend = get_backend("vllm", "b200_sxm")
    runtime = RuntimeConfig(batch_size=1, isl=4096, osl=64, prefix=prefix)
    summary = backend.run_agg(model, database, runtime, ctx_tokens=budget)
    current = summary.get_memory()
    assert current["activations"] == pytest.approx(new_activation, abs=1e-6)
    assert current["total"] == pytest.approx(new_total, abs=1e-6)
    # Reconstruct only the old memory input, leaving latency prediction alone.
    previous = backend._get_memory_usage(
        model, database, 1, 1, 4096, 64, prefix=prefix, num_tokens=min(4096, budget), mtp_scaled_tokens=0
    )
    assert previous["activations"] == pytest.approx(old_activation, abs=1e-6)
    assert previous["total"] == pytest.approx(old_total, abs=1e-6)
    for key in ("weights", "kvcache", "nccl", "others"):
        assert previous[key] == current[key]
    capacity = database.system_spec["gpu"]["mem_capacity"]
    old_summary = InferenceSummary(runtime)
    old_summary.set_memory_and_check_oom(previous, capacity, **backend._oom_check_kwargs({}))
    assert not old_summary.check_oom() and not summary.check_oom()
    assert not old_summary.check_kv_cache_oom() and not summary.check_kv_cache_oom()


@pytest.mark.parametrize("budget", [1096, 2048, 4096])
def test_completed_qualified_prefill_retains_whole_request_attention(monkeypatch, budget):
    from aisim_core.sdk.backends.factory import get_backend
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.perf_database import get_database_view

    model = _qualified_scan_model("vllm")
    attention = get_model("Qwen/Qwen3-0.6B", ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1), "vllm")
    model.context_ops += [op for op in attention.context_ops if op._name == "context_attention"]
    backend = get_backend("vllm")
    database = get_database_view("b200_sxm", "vllm", "0.24.0", database_mode="SOL")
    monkeypatch.setattr(backend, "_get_memory_usage", lambda *a, **k: {"total": 1.0})
    runtime = RuntimeConfig(batch_size=4, isl=4096, prefix=3000, osl=64)
    reference = backend.run_mixed(model, database, runtime, MixedStepInput(4096, 3))
    summary = backend.run_agg(model, database, runtime, ctx_tokens=budget)
    scheduling = summary.get_step_estimates()["scheduling"]
    assert scheduling["num_mix_steps"] == 4
    assert summary.get_step_estimates()["mixed"].context_tokens == 4096
    expected = 4 * reference.per_op_latency_ms["context_attention (scaled)"]
    assert expected > 0
    group = summary.get_aggregate_energy_breakdown()["mix_step"]
    # Four completed requests each pay the full attention A, not A/ceil(4096/1096).
    assert group.per_op_latency_ms["context_attention (scaled)"] == pytest.approx(expected)
    full = backend.run_agg(model, database, runtime, ctx_tokens=4096)
    assert summary.get_result_dict()["ttft"] == pytest.approx(full.get_result_dict()["ttft"])


@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize("budget", [512, 2048, 4096])
def test_cached_batched_attention_counts_qualified_chunks_and_preserves_legacy(monkeypatch, qualified, budget):
    from aisim_core.sdk.backends.factory import get_backend
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.perf_database import get_database_view

    model = _qualified_scan_model("vllm")
    attention = get_model("Qwen/Qwen3-0.6B", ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1), "vllm")
    if not qualified:
        model.context_ops = []
    model.context_ops += [op for op in attention.context_ops if op._name == "context_attention"]
    backend = get_backend("vllm")
    database = get_database_view("b200_sxm", "vllm", "0.24.0", database_mode="SOL")
    monkeypatch.setattr(backend, "_get_memory_usage", lambda *a, **k: {"total": 1.0})
    runtime = RuntimeConfig(batch_size=4, isl=4096, prefix=3000, osl=64)
    name = "context_attention (scaled)"
    if qualified and budget < 1096:
        # Independent native queries for the 512, 512, 72 suffix walk. Each
        # full chunk coordinate prices that extend's attention without a divisor.
        expected = 0.0
        for past, end in [(3000, 3512), (3512, 4024), (4024, 4096)]:
            chunk = RuntimeConfig(batch_size=1, isl=end, prefix=past, osl=64)
            expected += 4 * backend.run_mixed(model, database, chunk, MixedStepInput(end, 3)).per_op_latency_ms[name]
        steps = 12
    else:
        full = backend.run_mixed(model, database, runtime, MixedStepInput(4096, 3))
        # Qualified one-step suffixes pay four full attention queries. Legacy
        # mean-field attention still cancels ceil(isl/budget) against its full-
        # prompt iteration count: (4*isl/budget)/(isl/budget) = four requests.
        expected = 4 * full.per_op_latency_ms[name]
        steps = 4 if qualified else 4 * 4096 / budget
    summary = backend.run_agg(model, database, runtime, ctx_tokens=budget)
    assert summary.get_step_estimates()["scheduling"]["num_mix_steps"] == steps
    assert summary.get_aggregate_energy_breakdown()["mix_step"].per_op_latency_ms[name] == pytest.approx(expected)


def test_fixed_decode_coverage_uses_native_subtotals_before_name_fold(monkeypatch):
    from aisim_core.sdk import rust_engine_step

    # 10 powered + 5 unpowered ms fold to 15 ms with positive energy.
    # The original decode shape is 10 powered + 15 unpowered ms. Coverage
    # remains 10 ms although the folded latency grows by 10 ms.
    handle = SimpleNamespace(
        _mixed_step_breakdown_per_op_with_decode_coverage=lambda *a, **k: (
            [],
            [],
            [("same_name", 15.0, 100.0, "mixed")],
            10.0,
            10.0,
        )
    )
    monkeypatch.setattr(rust_engine_step, "_cached_engine_handle", lambda *_: handle)
    monkeypatch.setattr(rust_engine_step, "_note_rust_provenance", lambda *_: None)
    captured = rust_engine_step.estimate_mixed_step_breakdown_with_rust(
        None, None, ctx_tokens=512, gen_tokens=1, isl=3512, osl=64, prefix=3000, _capture_decode_ops=True
    )
    assert captured["_decode_ops"][1] == 10.0
    values = rust_engine_step.estimate_mixed_step_breakdown_with_rust(
        None,
        None,
        ctx_tokens=512,
        gen_tokens=1,
        isl=3512,
        osl=64,
        prefix=3000,
        _decode_ops=([("same_name", 25.0, 100.0, "mixed")], 10.0),
    )
    assert values["latency_ms"] == 25.0
    assert values["energy_wms"] == 100.0
    assert values["covered_latency_ms"] == 10.0
