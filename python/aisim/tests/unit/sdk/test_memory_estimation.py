# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pure-Python KV-cache memory estimate in ``sdk.memory``.

Exercises the naive-fallback math (MLA-aware per-token KV, rough weight estimate,
80%-of-post-weight reservation, default constants) and the OfFree/OfTotal native
budget formulas. ``sdk.memory`` imports the compiled ``aisim_core``
extension at module top, so these tests are skipped (``pytest.importorskip``)
when it is not built. The native budget math is tested with a synthetic breakdown
(no perf DB / model build), and the routing in ``estimate_kv_cache`` is driven by
monkeypatching ``KVCacheEstimator.from_request`` (to raise -> fallback, or to
return a ``KVCacheEstimator`` carrying a synthetic breakdown).

The fixture-derived magic numbers (DeepSeek-V3 -> 70_272 bytes/token, Llama-3.1-70B
-> 327_680) match the values the prior Rust naive tests asserted, so this is a
checkable parity assertion for the port.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

# ``sdk.memory`` imports the compiled ``aisim_core`` extension at module
# top, so these pure-Python tests require it to be importable. Skip them when it
# is not built rather than stubbing ``sys.modules`` — a stub would leak to other
# test modules in the same xdist worker and break tests that use the real
# extension (e.g. the FPM pyclass).
pytest.importorskip("aisim_core")

from aisim.sdk import memory
from aisim.sdk.memory import NaiveKVCacheEstimator
from aisim_core.sdk.models.helpers import _apply_model_quant_defaults

_GIB = 1 << 30


# --------------------------------------------------------------------------- #
# Memory-fraction validation (runs before any model build).
# --------------------------------------------------------------------------- #


def test_fraction_validation_rejects_wrong_kind_per_backend():
    with pytest.raises(ValueError, match="incompatible memory fraction"):
        memory._validate_memory_fraction("trtllm", "of_total", 0.9)
    with pytest.raises(ValueError, match="incompatible memory fraction"):
        memory._validate_memory_fraction("vllm", "of_free", 0.9)
    with pytest.raises(ValueError, match="incompatible memory fraction"):
        memory._validate_memory_fraction("sglang", "of_free", 0.9)
    # Correct pairings pass.
    memory._validate_memory_fraction("trtllm", "of_free", 0.9)
    memory._validate_memory_fraction("vllm", "of_total", 0.85)
    memory._validate_memory_fraction("sglang", "of_total", 0.85)


def test_fraction_validation_rejects_out_of_range():
    with pytest.raises(ValueError):
        memory._validate_memory_fraction("trtllm", "of_free", 1.5)
    with pytest.raises(ValueError):
        memory._validate_memory_fraction("vllm", "of_total", -0.1)


def test_naive_reservation_validation_rejects_out_of_range():
    # Boundaries are valid; finite values outside [0, 1] (and non-finite) are not.
    memory._validate_naive_reservation(0.0)
    memory._validate_naive_reservation(1.0)
    memory._validate_naive_reservation(0.8)
    for bad in (-0.1, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="naive_kv_reservation"):
            memory._validate_naive_reservation(bad)


def test_vision_rank_inputs_are_validated_up_front():
    for bad in ("TP", "tensor", ""):
        with pytest.raises(ValueError, match="encoder_parallel"):
            memory._validate_encoder_parallel(bad)
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="reserved_bytes"):
            memory._validate_reserved_bytes(bad)


def test_estimate_num_gpu_blocks_rejects_non_positive_or_non_integer_block_size():
    # Caught up front (before any model build), so no perf DB / fixture is needed.
    # A positive non-integer (e.g. 0.5 -> int() == 0) must be rejected rather than
    # reaching the floor-divide and raising ZeroDivisionError.
    common = dict(
        max_num_tokens=8192,
        max_batch_size=256,
        memory_fraction_kind="of_free",
        memory_fraction_value=0.9,
    )
    for bad in (0, -1, 0.5, 1.5, True):
        with pytest.raises(ValueError, match="scheduler_block_size"):
            memory.estimate_kv_cache("Qwen/Qwen3-32B", "h200_sxm", "trtllm", scheduler_block_size=bad, **common)
        with pytest.raises(ValueError, match="scheduler_block_size"):
            memory.estimate_num_gpu_blocks(
                "Qwen/Qwen3-32B",
                "h200_sxm",
                "trtllm",
                scheduler_block_size=bad,
                **common,
            )


# --------------------------------------------------------------------------- #
# Native budget math: OfFree (TRT-LLM) and OfTotal (vLLM/SGLang).
# --------------------------------------------------------------------------- #


def _breakdown(non_kv: float, kv_per_token: float, capacity: float) -> dict[str, float]:
    return {
        "weights_bytes": non_kv * 0.7,
        "activations_bytes": non_kv * 0.2,
        "runtime_overhead_bytes": non_kv * 0.07,
        "comm_overhead_bytes": non_kv * 0.03,
        "non_kv_bytes": non_kv,
        "kv_size_per_token_bytes": kv_per_token,
        "gpu_memory_capacity_bytes": capacity,
    }


def _native_estimate(breakdown, **kwargs):
    """Run the native budget math over a (synthetic) breakdown via KVCacheEstimator."""
    return memory.KVCacheEstimator(breakdown).estimate(**kwargs)


def _naive_estimate(
    model_path,
    *,
    tp_size=1,
    pp_size=1,
    moe_ep_size=1,
    moe_tp_size=1,
    gpu_memory_capacity_bytes_override,
    naive_kv_reservation,
    allow_hf_config_download=False,
):
    """Build a NaiveKVCacheEstimator and run its estimate (mirrors the old helper)."""
    return memory.NaiveKVCacheEstimator.from_model_path(
        model_path,
        tp_size=tp_size,
        pp_size=pp_size,
        allow_hf_config_download=allow_hf_config_download,
        moe_ep_size=moe_ep_size,
        moe_tp_size=moe_tp_size,
    ).estimate(capacity=gpu_memory_capacity_bytes_override, naive_kv_reservation=naive_kv_reservation)


def test_of_free_budget_formula():
    capacity, non_kv, kv_per_token, f = 141.0 * _GIB, 60.0 * _GIB, 327_680.0, 0.9
    out = _native_estimate(
        _breakdown(non_kv, kv_per_token, capacity),
        is_of_free=True,
        fraction=f,
        gpu_memory_capacity_bytes_override=None,
    )
    ref_kv = (capacity - non_kv) * f
    assert out["total_kv_size_bytes"] == int(ref_kv)
    assert out["total_kv_size_tokens"] == int(ref_kv / kv_per_token)
    assert out["source"] == "native"
    assert out["memory_breakdown"] is not None


def test_of_total_budget_formula_differs_from_of_free():
    capacity, non_kv, kv_per_token, f = 180.0 * _GIB, 70.0 * _GIB, 131_072.0, 0.9
    bd = _breakdown(non_kv, kv_per_token, capacity)
    of_total = _native_estimate(bd, is_of_free=False, fraction=f, gpu_memory_capacity_bytes_override=None)
    of_free = _native_estimate(bd, is_of_free=True, fraction=f, gpu_memory_capacity_bytes_override=None)
    assert of_total["total_kv_size_bytes"] == int(capacity * f - non_kv)
    # OfFree gives a strictly larger budget for f < 1 (the whole point of the split).
    assert of_free["total_kv_size_bytes"] > of_total["total_kv_size_bytes"]


def test_native_no_kv_budget_raises():
    capacity, non_kv = 100.0 * _GIB, 95.0 * _GIB
    with pytest.raises(ValueError, match="no KV budget"):
        _native_estimate(
            _breakdown(non_kv, 100_000.0, capacity),
            is_of_free=False,
            fraction=0.9,
            gpu_memory_capacity_bytes_override=None,
        )


def test_native_zero_per_token_raises():
    with pytest.raises(ValueError, match="kv_size_per_token_bytes"):
        _native_estimate(
            _breakdown(10.0 * _GIB, 0.0, 100.0 * _GIB),
            is_of_free=True,
            fraction=0.9,
            gpu_memory_capacity_bytes_override=None,
        )


@pytest.mark.parametrize(
    "override, resolved, token_capacity",
    [
        (None, "fp8", 2097152),
        ("fp8", "fp8", 2097152),
        ("bfloat16", "bfloat16", 1048576),
        ("nvfp4", "nvfp4", 3728270),
    ],
)
def test_native_capacity_discloses_resolved_kv_precision(monkeypatch, override, resolved, token_capacity):
    def get_model(_path, config, backend, *, database):
        # Exercise real default resolution for an FP8 checkpoint with no KV declaration.
        _apply_model_quant_defaults(config, {"quant_algo": "fp8"}, "Qwen3ForCausalLM", backend)
        # K and V, eight layers, two KV heads, head dimension 64.
        bytes_per_token = 2 * 8 * 2 * 64 * config.kvcache_quant_mode.value.memory
        return SimpleNamespace(
            config=config,
            get_kvcache_bytes_per_sequence=lambda length: length * bytes_per_token,
            get_kvcache_batch_capacity=lambda budget, _batch: int(budget // bytes_per_token),
        )

    monkeypatch.setattr(memory, "get_model", get_model)
    monkeypatch.setattr(
        memory,
        "get_backend",
        lambda _: SimpleNamespace(_get_memory_usage=lambda *_a, **_k: dict(weights=1, activations=1, others=1, nccl=1)),
    )
    monkeypatch.setattr(
        memory.perf_database,
        "get_database",
        lambda *_a, **_k: SimpleNamespace(version="0.19.0", system_spec={"gpu": {"mem_capacity": 8 * _GIB}}),
    )
    diagnostics = {}
    blocks = memory.estimate_num_gpu_blocks(
        "test/fp8-checkpoint",
        "test_gpu",
        "vllm",
        "0.19.0",
        max_num_tokens=128,
        max_batch_size=1,
        memory_fraction_kind="of_total",
        memory_fraction_value=1.0,
        scheduler_block_size=64,
        kvcache_quant_mode=override,
        diagnostics=diagnostics,
    )
    # Identical 4 GiB cache budgets: FP8 uses 2048 bytes/token, BF16 4096, NVFP4 1152.
    assert diagnostics["total_kv_size_bytes"] == 4 * _GIB
    assert diagnostics["total_kv_size_tokens"] == token_capacity
    assert blocks == token_capacity // 64
    assert diagnostics["kvcache_quant_mode"] == resolved


@pytest.mark.parametrize("selector", ["omitted", None, "current", "literal"])
def test_native_capacity_defaults_backend_version_to_current(monkeypatch, selector):
    # Keep real version resolution and database lookup; isolate model weights.
    seen_versions = []
    monkeypatch.setattr(
        memory,
        "get_model",
        lambda *_args, database: SimpleNamespace(
            get_kvcache_bytes_per_sequence=lambda length: 128 * length,
            get_kvcache_batch_capacity=lambda budget, _batch: int(budget // 128),
        ),
    )

    def memory_usage(_model, database, **_kwargs):
        seen_versions.append(database.version)
        return dict(weights=1, activations=1, others=1, nccl=1)

    monkeypatch.setattr(memory, "get_backend", lambda _: SimpleNamespace(_get_memory_usage=memory_usage))
    literal = memory.perf_database.resolve_query_version("h200_sxm", "vllm", "current")
    options = dict(max_num_tokens=8192, max_batch_size=256, memory_fraction_kind="of_total", memory_fraction_value=0.9)
    expected = memory.estimate_kv_cache("test/model", "h200_sxm", "vllm", literal, **options)
    version = {} if selector == "omitted" else {"backend_version": literal if selector == "literal" else selector}
    actual = memory.estimate_kv_cache("test/model", "h200_sxm", "vllm", **version, **options)
    assert actual == expected
    assert actual["source"] == "native"
    assert seen_versions == [literal, literal]


def test_native_capacity_rejects_unavailable_backend_version():
    with pytest.raises(ValueError, match="9.9.9") as error:
        memory.estimate_kv_cache(
            "test/model",
            "h200_sxm",
            "vllm",
            "9.9.9",
            max_num_tokens=8192,
            max_batch_size=256,
            memory_fraction_kind="of_total",
            memory_fraction_value=0.9,
        )
    assert "NoneType" not in str(error.value)


def test_breakdown_applies_nextn_to_model_config(monkeypatch):
    # KVCacheEstimator.from_request must push nextn/MTP onto the ModelConfig before
    # get_model, so _get_memory_usage scales activation memory for speculative
    # decoding instead of treating the request as nextn=0.
    captured = {}

    def _fake_get_model(model_path, model_config, backend, *, database):
        captured["nextn"] = model_config.nextn
        captured["has_nextn_accepted"] = hasattr(model_config, "nextn_accepted")

        class _StubModel:
            def get_kvcache_bytes_per_sequence(self, seq_len):
                return 100.0 * seq_len

            def get_kvcache_max_tokens(self, budget):
                return int(budget // 100)

        return _StubModel()

    class _StubBackend:
        def _get_memory_usage(self, *a, **k):
            return {"weights": 1.0, "activations": 1.0, "others": 1.0, "nccl": 1.0, "kvcache": 0.0}

    class _StubDB:
        def __init__(self):
            self.version = "1.3.0rc20"
            self.system_spec = {"gpu": {"mem_capacity": 100 * _GIB}}

    monkeypatch.setattr(memory, "get_model", _fake_get_model)
    monkeypatch.setattr(memory, "get_backend", lambda backend: _StubBackend())

    def _fake_get_database(*args, **kwargs):
        captured["systems_paths"] = kwargs.get("systems_paths")
        return _StubDB()

    monkeypatch.setattr(memory.perf_database, "get_database", _fake_get_database)

    memory.KVCacheEstimator.from_request(
        "Qwen/Qwen3-32B",
        "h200_sxm",
        "trtllm",
        max_num_tokens=8192,
        max_batch_size=256,
        nextn=2,
        systems_path="/tmp/aic-core-systems",
    )
    assert captured["nextn"] == 2
    assert captured["has_nextn_accepted"] is False
    assert captured["systems_paths"] == "/tmp/aic-core-systems"


def test_breakdown_accepts_nextn_without_acceptance_field(monkeypatch):
    # Capacity math consumes only the core-owned draft depth.
    captured = {}

    def _fake_get_model(model_path, model_config, backend, *, database):
        captured["nextn"] = model_config.nextn
        captured["has_nextn_accepted"] = hasattr(model_config, "nextn_accepted")

        class _StubModel:
            def get_kvcache_bytes_per_sequence(self, seq_len):
                return 100.0 * seq_len

            def get_kvcache_max_tokens(self, budget):
                return int(budget // 100)

        return _StubModel()

    class _StubBackend:
        def _get_memory_usage(self, *a, **k):
            return {"weights": 1.0, "activations": 1.0, "others": 1.0, "nccl": 1.0, "kvcache": 0.0}

    class _StubDB:
        def __init__(self):
            self.version = "1.3.0rc20"
            self.system_spec = {"gpu": {"mem_capacity": 100 * _GIB}}

    monkeypatch.setattr(memory, "get_model", _fake_get_model)
    monkeypatch.setattr(memory, "get_backend", lambda backend: _StubBackend())
    monkeypatch.setattr(memory.perf_database, "get_database", lambda *a, **k: _StubDB())

    memory.KVCacheEstimator.from_request(
        "Qwen/Qwen3-32B",
        "h200_sxm",
        "trtllm",
        max_num_tokens=8192,
        max_batch_size=256,
        nextn=2,
    )
    assert captured["nextn"] == 2
    assert captured["has_nextn_accepted"] is False


@pytest.mark.parametrize("backend", ["sglang", "vllm", "trtllm"])
@pytest.mark.parametrize("extra_graph_gib", [0, 1])
def test_prefill_workspace_respects_backend_kv_pool(monkeypatch, backend, extra_graph_gib):
    """A larger prefill budget must not consume SGLang's static KV pool."""

    class Model:
        def get_kvcache_bytes_per_sequence(self, seq_len):
            return 1024 * seq_len

        def get_kvcache_max_tokens(self, budget):
            return int(budget // 1024)

        def get_kvcache_batch_capacity(self, budget, max_batch_size):
            assert max_batch_size == 128
            return self.get_kvcache_max_tokens(budget)

    class Backend:
        def _get_memory_usage(self, *args, num_tokens, **kwargs):
            return {"weights": 59, "activations": num_tokens / 1024, "others": 2, "nccl": 1}

    class Database:
        def __init__(self):
            self.version = "0.5.12"
            self.system_spec = {"gpu": {"mem_capacity": 100 * _GIB}}

    monkeypatch.setattr(memory, "get_model", lambda *args, database: Model())
    monkeypatch.setattr(memory, "get_backend", lambda *args: Backend())
    monkeypatch.setattr(memory.perf_database, "get_database", lambda *args, **kwargs: Database())
    budgets = []
    for tokens in (1024, 16384):
        result = memory.estimate_kv_cache(
            "test-model",
            "test-system",
            backend,
            max_num_tokens=tokens,
            max_batch_size=128,
            memory_fraction_kind="of_free" if backend == "trtllm" else "of_total",
            memory_fraction_value=0.8,
            cuda_graph_reserved_bytes=extra_graph_gib * _GIB,
        )
        # Diagnostics describe the forward footprint even when it is outside
        # the static pool. Explicit extra graph reservation still consumes KV.
        assert result["memory_breakdown"]["activations_bytes"] == tokens / 1024 * _GIB
        assert result["memory_breakdown"]["cuda_graph_reserved_bytes"] == extra_graph_gib * _GIB
        budgets.append(result["total_kv_size_bytes"])
        assert result["total_kv_size_tokens"] == result["total_kv_size_bytes"] // 1024
    expected_gib = {
        "sglang": [18.6 - extra_graph_gib, 18.6 - extra_graph_gib],
        "vllm": [17 - extra_graph_gib, 2 - extra_graph_gib],
        "trtllm": [(37 - extra_graph_gib) * 0.8, (22 - extra_graph_gib) * 0.8],
    }
    assert budgets == pytest.approx([value * _GIB for value in expected_gib[backend]], abs=1)


def test_native_capacity_override_wins():
    capacity, non_kv, kv_per_token, f = 141.0 * _GIB, 60.0 * _GIB, 327_680.0, 0.9
    override = 200 * _GIB
    out = _native_estimate(
        _breakdown(non_kv, kv_per_token, capacity),
        is_of_free=True,
        fraction=f,
        gpu_memory_capacity_bytes_override=override,
    )
    assert out["total_gpu_capacity_bytes"] == override
    assert out["total_kv_size_bytes"] == int((override - non_kv) * f)


# --------------------------------------------------------------------------- #
# NaiveKVCacheEstimator: per-token KV (standard GQA/MHA vs MLA), weight estimate,
# config->geometry parsing, sliding-window detection, and the byte-budget ->
# token-count inverse with auto LINEAR/HYBRID mode selection.
# --------------------------------------------------------------------------- #


def _estimator(geometry, *, dtype_bytes=2, tp_size=1, pp_size=1, moe_ep_size=1, moe_tp_size=1):
    return memory.NaiveKVCacheEstimator(
        geometry,
        dtype_bytes=dtype_bytes,
        tp_size=tp_size,
        pp_size=pp_size,
        moe_ep_size=moe_ep_size,
        moe_tp_size=moe_tp_size,
    )


def test_naive_kv_per_token_mla():
    # DeepSeek-V3-style: layers=61, kv_lora_rank=512, qk_rope_head_dim=64, bf16:
    #   61 * (512 + 64) * 2 = 70_272 bytes/token. The latent is NOT sharded by TP.
    geom = {"layers": 61, "kv_lora_rank": 512, "qk_rope_head_dim": 64}
    assert _estimator(geom, tp_size=1).kv_bytes_per_token() == 70_272
    assert _estimator(geom, tp_size=8).kv_bytes_per_token() == 70_272


def test_naive_kv_per_token_mla_fails_closed_without_latent_width():
    # MLA detected (kv_lora_rank present) but qk_rope_head_dim missing -> the
    # cached latent width is unclear, so fail closed (None -> caller raises) rather
    # than guessing the latent width.
    assert _estimator({"layers": 61, "kv_lora_rank": 512}).kv_bytes_per_token() is None


def test_naive_kv_per_token_standard():
    # Llama-3.1-70B-style: layers=80, num_kv_heads=8, head_dim=128, bf16:
    #   TP=1: 2 * 8 * 128 * 80 * 2 = 327_680.
    geom = {"layers": 80, "num_kv_heads": 8, "head_dim": 128}
    assert _estimator(geom, tp_size=1).kv_bytes_per_token() == 327_680
    # TP=4: kv_heads_per_rank = ceil(8/4) = 2 -> quarter.
    assert _estimator(geom, tp_size=4).kv_bytes_per_token() == 2 * 2 * 128 * 80 * 2
    # TP=16: kv_heads_per_rank = ceil(8/16) = 1.
    assert _estimator(geom, tp_size=16).kv_bytes_per_token() == 2 * 1 * 128 * 80 * 2


def test_naive_geometry_from_raw_unsupported_arch():
    # Unsupported architecture -> raw-key extraction (parsed=None). head_dim is
    # derived from hidden_size / num_attention_heads when absent.
    hf_config = {
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "vocab_size": 128_000,
        "intermediate_size": 11_008,
    }
    geom = memory.NaiveKVCacheEstimator._geometry(hf_config, None)
    assert geom["head_dim"] == 4096 // 32  # 128
    # Standard branch: 2 * ceil(8/2) * 128 * 32 * 2 (bf16).
    assert _estimator(geom, tp_size=2).kv_bytes_per_token() == 2 * 4 * 128 * 32 * 2
    assert _estimator(geom).weight_bytes() is not None
    # vocab / intermediate_size are required: drop either and the weight is None.
    assert _estimator({**geom, "vocab": None}).weight_bytes() is None
    assert _estimator({**geom, "inter": None}).weight_bytes() is None


@pytest.mark.parametrize(
    ("hf_config", "expected"),
    [
        pytest.param(
            {
                "n_embd": 768,
                "n_layer": 12,
                "n_head": 12,
                "n_inner": None,
                "vocab_size": 50_257,
            },
            (768, 12, 12, 64, 3_072, 50_257),
            id="gpt2",
        ),
        pytest.param(
            {
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": None,
                "ffn_dim": 3_072,
                "vocab_size": 50_272,
            },
            (768, 12, 12, 64, 3_072, 50_272),
            id="opt",
        ),
        pytest.param(
            {"hidden_size": 64, "n_layer": 2, "n_head": 8, "vocab_size": 250_880},
            (64, 2, 8, 8, 256, 250_880),
            id="bloom",
        ),
        pytest.param(
            {
                "n_embd": 4_096,
                "n_layer": 28,
                "n_head": 16,
                "n_inner": None,
                "vocab_size": 50_400,
            },
            (4_096, 28, 16, 256, 16_384, 50_400),
            id="gpt-j",
        ),
        pytest.param(
            {
                "hidden_size": 2_048,
                "num_layers": 24,
                "num_heads": 16,
                "intermediate_size": None,
                "vocab_size": 50_257,
            },
            (2_048, 24, 16, 128, 8_192, 50_257),
            id="gpt-neo",
        ),
        pytest.param(
            {
                "hidden_size": 4_544,
                "num_hidden_layers": 32,
                "num_attention_heads": 71,
                "num_kv_heads": 71,
                "ffn_hidden_size": 18_176,
                "multi_query": True,
                "new_decoder_architecture": False,
                "vocab_size": 65_024,
            },
            (4_544, 32, 1, 64, 18_176, 65_024),
            id="falcon",
        ),
        pytest.param(
            {
                "n_embd": 4_096,
                "n_layer": 28,
                "n_head": 16,
                "n_inner": None,
                "vocab_size": 50_400,
            },
            (4_096, 28, 16, 256, 16_384, 50_400),
            id="codegen",
        ),
        pytest.param(
            {"n_embed": 1_024, "n_layer": 20, "n_head": 16, "n_inner": 0, "vocab_size": 50_400},
            (1_024, 20, 16, 64, 4_096, 50_400),
            id="n-embed-alias",
        ),
    ],
)
def test_naive_geometry_supports_hf_alias_families(hf_config, expected):
    geom = memory.NaiveKVCacheEstimator._geometry(hf_config, None)
    hidden, layers, kv_heads, head_dim, inter, vocab = expected
    assert geom == {
        "layers": layers,
        "num_attention_heads": hidden // head_dim,
        "num_kv_heads": kv_heads,
        "head_dim": head_dim,
        "kv_lora_rank": None,
        "qk_rope_head_dim": None,
        "vocab": vocab,
        "hidden": hidden,
        "inter": inter,
        "num_experts": 0,
        "moe_inter": inter,
        "sliding_window": None,
        "num_swa_layers": None,
        "num_global_layers": None,
    }

    estimator = _estimator(geom)
    expected_weight_params = 2 * vocab * hidden + layers * (4 * hidden**2 + 3 * hidden * inter)
    assert estimator.weight_bytes() == expected_weight_params * 2
    assert estimator.kv_bytes_per_token() == 2 * kv_heads * head_dim * layers * 2


@pytest.mark.parametrize("invalid_preferred", [-1, 0, None, "not-an-int", 1.5, True])
def test_naive_raw_dimension_skips_invalid_preferred_alias(invalid_preferred):
    hf_config = {"hidden_size": invalid_preferred, "n_embd": 768}

    assert memory.NaiveKVCacheEstimator._raw_dimension(hf_config, "hidden_size", "n_embd") == 768


def test_naive_falcon_new_decoder_keeps_explicit_kv_head_count():
    hf_config = {
        "hidden_size": 8_192,
        "num_hidden_layers": 60,
        "num_attention_heads": 64,
        "num_kv_heads": 8,
        "ffn_hidden_size": 32_768,
        "multi_query": True,
        "new_decoder_architecture": True,
        "vocab_size": 65_024,
    }
    geom = memory.NaiveKVCacheEstimator._geometry(hf_config, None)
    assert geom["num_kv_heads"] == 8


@pytest.mark.parametrize("missing_key", ["n_embd", "n_layer", "vocab_size"])
def test_naive_raw_alias_geometry_keeps_required_dimensions_fail_closed(missing_key):
    hf_config = {
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "head_dim": 64,
        "vocab_size": 50_257,
    }
    hf_config.pop(missing_key)
    geom = memory.NaiveKVCacheEstimator._geometry(hf_config, None)
    assert _estimator(geom).weight_bytes() is None


def test_naive_num_experts_alias_does_not_use_dense_ffn_fallback():
    hf_config = {
        "hidden_size": 1_024,
        "num_hidden_layers": 12,
        "num_attention_heads": 16,
        "num_experts": 8,
        "vocab_size": 32_000,
    }

    geom = memory.NaiveKVCacheEstimator._geometry(hf_config, None)

    assert geom["num_experts"] == 8
    assert geom["inter"] is None
    assert _estimator(geom).weight_bytes() is None


def test_naive_kv_per_token_empty_geometry_is_none():
    assert _estimator({}).kv_bytes_per_token() is None
    assert _estimator({}).weight_bytes() is None


def test_naive_weight_bytes_moe_divides_by_ep_and_moe_tp():
    # MoE model: 32 layers, hidden=4096, 64 experts, moe_inter=2048, bf16.
    geom = {
        "hidden": 4096,
        "layers": 32,
        "vocab": 128_000,
        "inter": 11_008,
        "num_experts": 64,
        "moe_inter": 2048,
    }
    # EP=8 should reduce expert weight vs EP=1 (both at tp=8, moe_tp=1).
    w_ep1 = _estimator(geom, tp_size=8, moe_ep_size=1).weight_bytes()
    w_ep8 = _estimator(geom, tp_size=8, moe_ep_size=8).weight_bytes()
    assert w_ep8 < w_ep1

    # moe_tp=2,ep=4 vs moe_tp=1,ep=8: same total MoE sharding (product=8),
    # so expert weight should be identical; non-expert params unchanged.
    w_tp2_ep4 = _estimator(geom, tp_size=8, moe_tp_size=2, moe_ep_size=4).weight_bytes()
    w_tp1_ep8 = _estimator(geom, tp_size=8, moe_tp_size=1, moe_ep_size=8).weight_bytes()
    assert w_tp2_ep4 == w_tp1_ep8

    # Expert weight should scale with the product moe_tp * moe_ep.
    expert_params_per_layer = 64 * 3 * 4096 * 2048
    # EP=1,moe_tp=1 vs EP=8,moe_tp=1: expert diff = (1 - 1/8) of expert params / pp=1.
    expert_diff = w_ep1 - w_ep8
    expected = expert_params_per_layer * 32 * (1 - 1 / 8) * 2  # bf16, pp=1
    assert abs(expert_diff - expected) < 1024


def test_dtype_bytes():
    dtype_bytes = memory.NaiveKVCacheEstimator._dtype_bytes
    assert dtype_bytes({"torch_dtype": "bfloat16"}) == 2
    assert dtype_bytes({"torch_dtype": "float32"}) == 4
    assert dtype_bytes({"torch_dtype": "fp8_e4m3"}) == 1
    assert dtype_bytes({}) == 2  # default bf16


# --------------------------------------------------------------------------- #
# NaiveKVCacheEstimator hybrid sliding-window support: layer-layout detection +
# the piecewise byte-budget -> token-count inverse, so the fallback does not
# assume linear KV growth for SWA models.
# --------------------------------------------------------------------------- #


def test_naive_swa_layout_from_layer_types():
    # Explicit per-layer list (Gemma-3/4 style): count sliding vs the rest.
    hf_config = {"sliding_window": 1024, "layer_types": ["sliding_attention"] * 5 + ["full_attention"]}
    assert memory.NaiveKVCacheEstimator._swa_layout(hf_config) == (1024, 5, 1)


def test_naive_swa_layout_from_pattern():
    # sliding_window + sliding_window_pattern=6: one global layer every 6 layers.
    hf_config = {"sliding_window": 4096, "num_hidden_layers": 30, "sliding_window_pattern": 6}
    assert memory.NaiveKVCacheEstimator._swa_layout(hf_config) == (4096, 25, 5)


@pytest.mark.parametrize("layer_key", ["n_layer", "num_layers"])
def test_naive_swa_layout_from_pattern_supports_layer_aliases(layer_key):
    hf_config = {"sliding_window": 4096, layer_key: 30, "sliding_window_pattern": 6}
    assert memory.NaiveKVCacheEstimator._swa_layout(hf_config) == (4096, 25, 5)


def test_naive_swa_layout_unresolved_returns_none():
    # A bare window with no per-layer signal is too model-specific to split.
    assert memory.NaiveKVCacheEstimator._swa_layout({"sliding_window": 4096}) == (None, None, None)
    assert memory.NaiveKVCacheEstimator._swa_layout({}) == (None, None, None)


def test_naive_hybrid_mode_caps_swa_past_window():
    # 25 SWA + 5 global layers, window 1024, 8 KV heads, head_dim 128, bf16, tp=1.
    geom = {
        "sliding_window": 1024,
        "num_swa_layers": 25,
        "num_global_layers": 5,
        "num_kv_heads": 8,
        "head_dim": 128,
    }
    est = _estimator(geom)
    assert est.mode is memory.NaiveKVCacheMode.HYBRID
    hybrid = memory.NaiveKVCacheMode.HYBRID
    per_layer = 2 * 8 * 128 * 2  # 4096
    rate = (25 + 5) * per_layer  # every layer grows up to the window
    global_const = 5 * per_layer
    assert est.get_kvcache_max_tokens(rate * 512, mode=hybrid) == 512  # within the window: linear
    assert est.get_kvcache_max_tokens(rate * 1024, mode=hybrid) == 1024  # at the boundary
    # Past the window only the 5 global layers grow.
    budget = rate * 1024 + global_const * 3
    assert est.get_kvcache_max_tokens(budget, mode=hybrid) == 1027
    # mode auto-resolves to HYBRID here, so it gives the same answer as forcing it.
    assert est.get_kvcache_max_tokens(budget) == 1027
    # Beats the linear seq_len=1 extrapolation, which over-charges the SWA layers.
    assert est.get_kvcache_max_tokens(budget, mode=hybrid) > budget // rate


def test_naive_mode_linear_for_standard_and_mla():
    # No sliding-window layout -> LINEAR.
    assert _estimator({"num_kv_heads": 8, "head_dim": 128}).mode is memory.NaiveKVCacheMode.LINEAR
    # MLA latent cache is out of scope for the hybrid heuristic -> LINEAR.
    mla_hybrid = {
        "sliding_window": 1024,
        "num_swa_layers": 5,
        "num_global_layers": 1,
        "kv_lora_rank": 512,
        "num_kv_heads": 8,
        "head_dim": 128,
    }
    assert _estimator(mla_hybrid).mode is memory.NaiveKVCacheMode.LINEAR


# --------------------------------------------------------------------------- #
# Naive fallback end-to-end (post-weight reservation + defaults), driven through
# the SDK's existing config loaders + `_parse_hf_config_json` (cached fixtures).
# --------------------------------------------------------------------------- #


# A synthetic config for an architecture AIC does not support, carrying the full
# geometry the naive weight + KV formulas require.
_RAW_UNSUPPORTED = {
    "architectures": ["FooBarForCausalLM"],
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 4096,
    "vocab_size": 128_000,
    "intermediate_size": 14_336,
    "torch_dtype": "bfloat16",
}


def test_naive_fallback_mla_fixture():
    # `deepseek-ai/DeepSeek-V3` is cached in model_configs (DefaultHFModels), so
    # `NaiveKVCacheEstimator._load_config` loads it without a download; the
    # supported arch goes through `_parse_hf_config_json` -> 70_272 bytes/token.
    # moe_ep_size=16 shards the 256 experts across 16 GPUs (16 local each).
    capacity = 180 * _GIB
    out = _naive_estimate(
        "deepseek-ai/DeepSeek-V3",
        tp_size=16,
        pp_size=1,
        moe_ep_size=16,
        gpu_memory_capacity_bytes_override=capacity,
        naive_kv_reservation=memory._DEFAULT_NAIVE_KV_RESERVATION,
        allow_hf_config_download=False,
    )
    assert out["source"] == "naive_fallback"
    assert out["memory_breakdown"] is None
    assert out["kv_size_per_token_bytes"] == 70_272  # config-driven MLA latent


def test_naive_fallback_unsupported_arch_uses_raw_read(monkeypatch):
    # Unsupported architecture: `_parse_hf_config_json` raises, so the estimator
    # falls to the raw-key read. This drives the real parser (FooBar arch is not in
    # ARCHITECTURE_TO_MODEL_FAMILY) through the except branch into
    # `NaiveKVCacheEstimator._geometry(hf_config, None)`.
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "_load_config", lambda *a, **k: dict(_RAW_UNSUPPORTED))
    out = _naive_estimate(
        "foo/bar-unknown-arch",
        tp_size=1,
        pp_size=1,
        gpu_memory_capacity_bytes_override=200 * _GIB,
        naive_kv_reservation=memory._DEFAULT_NAIVE_KV_RESERVATION,
        allow_hf_config_download=False,
    )
    assert out["source"] == "naive_fallback"
    # Raw-read standard branch: layers=48, n_kv=8, head_dim=4096/32=128, bf16.
    assert out["kv_size_per_token_bytes"] == 2 * 8 * 128 * 48 * 2  # 196608


def test_naive_fallback_unsupported_arch_uses_hf_aliases(monkeypatch):
    raw_config = {
        "architectures": ["FooBarForCausalLM"],
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "n_inner": None,
        "vocab_size": 50_257,
        "torch_dtype": "float16",
    }
    monkeypatch.setattr(
        memory.NaiveKVCacheEstimator,
        "_load_config",
        lambda *a, **k: dict(raw_config),
    )

    out = _naive_estimate(
        "foo/bar-aliased-unknown-arch",
        tp_size=1,
        pp_size=1,
        gpu_memory_capacity_bytes_override=20 * _GIB,
        naive_kv_reservation=memory._DEFAULT_NAIVE_KV_RESERVATION,
        allow_hf_config_download=False,
    )

    assert out["source"] == "naive_fallback"
    assert out["kv_size_per_token_bytes"] == 2 * 12 * 64 * 12 * 2


_UNKNOWN_DECODER = {
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "intermediate_size": 1024,
    "vocab_size": 1000,
    "torch_dtype": "bfloat16",
}


# The decoder above: head_dim 256 / 8 = 32, so K and V of 2 heads in 4 layers at 2 bytes hold 2 * 2 * 32 * 4 * 2 =
# 1,024 bytes per token, and 2 * 1,000 * 256 + 4 * (4 * 256**2 + 3 * 256 * 1,024) = 4,706,304 parameters hold
# 9,412,608 bytes.
_UNKNOWN_DECODER_KV_BYTES = 1024
_UNKNOWN_DECODER_WEIGHT_BYTES = 9_412_608
_UNKNOWN_WRAPPER = "UnknownWrapperForConditionalGeneration"
_NO_LAYERS_TEXT = (
    "the config parser refuses the HF config of the supported architecture LlamaForCausalLM: ValueError: "
    "Model config must define 'num_hidden_layers' or a parseable layer pattern."
)


@pytest.mark.parametrize(
    ("config", "text", "cause"),
    [
        ({"architectures": ["LlamaForCausalLM"], "hidden_size": 128}, _NO_LAYERS_TEXT, ValueError),
        (
            {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 2},
            "the config parser refuses the HF config of the supported architecture LlamaForCausalLM: KeyError: "
            "'hidden_size'",
            KeyError,
        ),
        ({"architectures": ["LlamaForCausalLM", "OtherForCausalLM"], "hidden_size": 128}, _NO_LAYERS_TEXT, ValueError),
    ],
    ids=["no-layers", "no-hidden-size", "known-first-of-two-names"],
)
def test_naive_estimate_of_a_known_architecture_fails_with_the_parser_message(config, text, cause):
    with pytest.raises(ValueError) as error:
        NaiveKVCacheEstimator.from_hf_config(config, tp_size=1, pp_size=1)
    assert str(error.value) == text
    assert type(error.value.__cause__) is cause


def test_naive_estimate_names_the_parser_for_a_valid_folder_whose_config_the_parser_refuses(tmp_path):
    from importlib import resources

    from aisim_core.sdk.utils import _load_model_config_from_model_path, _parse_hf_config_json

    # A Llama 4 folder in the form that the hub publishes keeps the image processor config in preprocessor_config.json.
    # The op-level loader attaches that file, so the folder parses. The naive estimate reads config.json alone, and the
    # parser refuses that config.
    name = "meta-llama--Llama-4-Scout-17B-16E-Instruct_config.json"
    document = json.loads(resources.files("aisim_core").joinpath("model_configs", name).read_text())
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(document.pop("image_processor_config")))
    (tmp_path / "config.json").write_text(json.dumps(document))
    parsed = _parse_hf_config_json(_load_model_config_from_model_path(str(tmp_path)))
    assert parsed["architecture"] == "Llama4ForConditionalGeneration"
    with pytest.raises(ValueError) as error:
        NaiveKVCacheEstimator.from_model_path(str(tmp_path), tp_size=1, pp_size=1, allow_hf_config_download=False)
    assert str(error.value) == (
        "the config parser refuses the HF config of the supported architecture Llama4ForConditionalGeneration: "
        "TypeError: Llama 4 config must preserve image_processor_config metadata or provide preprocessor_config.json"
    )


@pytest.mark.parametrize(
    "architectures",
    [["LlamaForCausalLM", 5], [["LlamaForCausalLM"]]],
    ids=["known-name-and-a-number", "nested-list"],
)
def test_naive_estimate_checks_the_architectures_field_before_the_parser(architectures):
    # A known first name with the layer count missing would reach the known-architecture refusal if the parser ran
    # first; a nested list is no name at all.
    with pytest.raises(ValueError) as error:
        NaiveKVCacheEstimator.from_hf_config({"architectures": architectures, "hidden_size": 128}, tp_size=1, pp_size=1)
    assert str(error.value) == f"HF config architectures must be a list of strings, got {architectures!r}"


@pytest.mark.parametrize("key", ["text_config", "llm_config"])
def test_naive_estimate_of_an_unknown_wrapper_reads_its_decoder(key):
    name = _UNKNOWN_WRAPPER
    flat = NaiveKVCacheEstimator.from_hf_config({**_UNKNOWN_DECODER, "architectures": [name]}, tp_size=1, pp_size=1)
    # The wrapper's own fields describe another model. The estimate must not read them, head_dim included, which
    # only the wrapper holds.
    wrapper = {
        "architectures": [name],
        "hidden_size": 4096,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "head_dim": 128,
        "torch_dtype": "float32",
        key: _UNKNOWN_DECODER,
    }
    nested = NaiveKVCacheEstimator.from_hf_config(wrapper, tp_size=1, pp_size=1)
    assert nested.geometry == flat.geometry
    assert nested.kv_bytes_per_token() == flat.kv_bytes_per_token() == _UNKNOWN_DECODER_KV_BYTES
    assert nested.weight_bytes() == flat.weight_bytes() == _UNKNOWN_DECODER_WEIGHT_BYTES
    assert (nested.architecture, nested.dtype_bytes) == (name, flat.dtype_bytes)
    # The decoder names no dtype: the wrapper's applies.
    decoder = {name_: value for name_, value in _UNKNOWN_DECODER.items() if name_ != "torch_dtype"}
    inherited = NaiveKVCacheEstimator.from_hf_config(
        {"architectures": [name], "torch_dtype": "float32", key: decoder}, tp_size=1, pp_size=1
    )
    assert inherited.dtype_bytes == 4
    # A decoder that names its own class supplies the cache architecture as well as the dimensions.
    named = NaiveKVCacheEstimator.from_hf_config(
        {"architectures": [name], key: {**_UNKNOWN_DECODER, "architectures": ["UnknownDecoderForCausalLM"]}},
        tp_size=1,
        pp_size=1,
    )
    assert (named.architecture, named.geometry) == ("UnknownDecoderForCausalLM", flat.geometry)
    # A known optional model_type enables topology checks without adding timing-only fields to this config.
    inferred = NaiveKVCacheEstimator.from_hf_config(
        {**_UNKNOWN_DECODER, "architectures": [name], key: {**_UNKNOWN_DECODER, "model_type": "llama"}},
        tp_size=1,
        pp_size=1,
    )
    assert (inferred.architecture, inferred.kv_bytes_per_token(), inferred.weight_bytes()) == (
        "LlamaForCausalLM",
        _UNKNOWN_DECODER_KV_BYTES,
        _UNKNOWN_DECODER_WEIGHT_BYTES,
    )


@pytest.mark.parametrize("dtype", [{"dtype": {}}, {"torch_dtype": 5, "dtype": []}])
@pytest.mark.parametrize("nested", [False, True], ids=["flat", "decoder"])
def test_naive_estimate_refuses_malformed_decoder_precision(nested, dtype):
    decoder = {key: value for key, value in _UNKNOWN_DECODER.items() if key != "torch_dtype"} | dtype
    config = (
        {"architectures": [_UNKNOWN_WRAPPER], "torch_dtype": "float32", "text_config": decoder} if nested else decoder
    )
    with pytest.raises(ValueError, match="torch_dtype or dtype must be a nonempty string"):
        NaiveKVCacheEstimator.from_hf_config(config, tp_size=1, pp_size=1)


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "decoder"])
@pytest.mark.parametrize(
    ("dtype", "factor"),
    [({"torch_dtype": None}, 1), ({"torch_dtype": 7}, 1), ({"torch_dtype": 7, "dtype": "float32"}, 2)],
)
def test_unregistered_decoder_defaults_an_untyped_torch_dtype(tmp_path, nested, dtype, factor):
    from aisim.capacity import estimate_kv_bytes_per_token

    decoder = {**_UNKNOWN_DECODER, **dtype}
    document = {"architectures": [_UNKNOWN_WRAPPER], "text_config": decoder} if nested else decoder
    model = _write_config(tmp_path, document)
    assert estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1) == factor * _UNKNOWN_DECODER_KV_BYTES


@pytest.mark.parametrize("wrapped", [False, True], ids=["flat", "wrapped"])
def test_gpt_oss_retains_its_sliding_window(tmp_path, wrapped):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    decoder = _load_pre_downloaded_hf_config("openai/gpt-oss-20b")
    config = {"architectures": [_UNKNOWN_WRAPPER], "text_config": decoder} if wrapped else decoder
    checkpoint = _write_config(tmp_path, config)
    estimator = NaiveKVCacheEstimator.from_model_path(checkpoint, tp_size=1, pp_size=1, allow_hf_config_download=False)
    # 12 global layers grow at 2,048 bytes per token each, 12 sliding layers retain only 128 tokens.
    assert estimator.get_kvcache_max_tokens(96 * 1024**2) == (96 * 1024**2) // (12 * 2048) - 128 == 3968


@pytest.mark.parametrize("layer_count", [2, 26])
def test_gpt_oss_rejects_a_window_pattern_that_does_not_cover_its_decoder(layer_count):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    base = _load_pre_downloaded_hf_config("openai/gpt-oss-20b")
    config = {**base, "layer_types": ["sliding_attention", "full_attention"] * (layer_count // 2)}
    with pytest.raises(ValueError, match=rf"layer_types has {layer_count} entries for 24 decoder layers"):
        NaiveKVCacheEstimator.from_hf_config(config, tp_size=1, pp_size=1)


@pytest.mark.parametrize("wrapper", [None, "Step3p7ForConditionalGeneration", _UNKNOWN_WRAPPER])
def test_step3p7_cache_capacity_counts_only_decoder_layers(tmp_path, wrapper):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    base = _load_pre_downloaded_hf_config("stepfun-ai/Step-3.7-Flash")
    decoder = {
        **base,
        "architectures": ["Step3p5ForCausalLM"],
        "layer_types": base["layer_types"] + ["sliding_attention"] * base["num_nextn_predict_layers"],
    }
    config = {"architectures": [wrapper], "text_config": decoder} if wrapper else decoder
    checkpoint = _write_config(tmp_path, config)
    estimator = NaiveKVCacheEstimator.from_model_path(checkpoint, tp_size=1, pp_size=1, allow_hf_config_download=False)

    # Each of the 45 decoder layers stores eight 128-wide bf16 K/V heads. Twelve layers grow past 512 tokens.
    bytes_per_layer = 2 * 8 * 128 * 2
    assert estimator.kv_bytes_per_token() == 45 * bytes_per_layer
    assert estimator.get_kvcache_max_tokens(64 * 1024**2) == 364
    assert estimator.get_kvcache_max_tokens((45 * 512 + 12) * bytes_per_layer) == 513


@pytest.mark.parametrize(
    "window_fields",
    [
        {"sliding_window": -512},
        {"sliding_window": True},
        {"sliding_window": 0.5},
        {"sliding_window": "128"},
        {"sliding_window": 512, "sliding_window_size": 256},
    ],
)
def test_step3p7_checks_the_selected_decoder_window_before_sizing(tmp_path, window_fields):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    decoder = _load_pre_downloaded_hf_config("stepfun-ai/Step-3.7-Flash") | window_fields
    decoder["layer_types"] = decoder["layer_types"] + ["sliding_attention"] * decoder["num_nextn_predict_layers"]
    checkpoint = _write_config(
        tmp_path, {"architectures": ["Step3p7ForConditionalGeneration"], "sliding_window": 512, "text_config": decoder}
    )
    args = (checkpoint, "missing-memory-spec", "vllm", "0.33.0")
    options = dict(
        systems_path=str(tmp_path),
        max_num_tokens=8192,
        max_batch_size=32,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        gpu_memory_capacity_bytes_override=1024 * _GIB,
        allow_naive_fallback=True,
    )
    for estimate, extra in [
        (memory.estimate_kv_cache, {}),
        (memory.estimate_num_gpu_blocks, {"scheduler_block_size": 16}),
    ]:
        with pytest.raises(ValueError, match="sliding_window must|conflicting sliding-window fields"):
            estimate(*args, **options, **extra)


def test_malformed_torch_dtype_keeps_the_explicit_decoder_dtype():
    decoder = {**_UNKNOWN_DECODER, "torch_dtype": 5, "dtype": "float32"}
    estimator = NaiveKVCacheEstimator.from_hf_config(decoder, tp_size=1, pp_size=1)
    assert estimator.kv_bytes_per_token() == 2_048


def test_wrapped_deepseek_v4_uses_the_decoders_compressed_cache(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    model = "deepseek-ai/DeepSeek-V4-Flash"
    config = _load_pre_downloaded_hf_config(model)
    wrapper = {"architectures": [_UNKNOWN_WRAPPER], "text_config": config}
    wrapped = _write_config(tmp_path, wrapper)
    for pp_size in (1, 4):
        assert estimate_kv_bytes_per_token(model, tp_size=1, pp_size=pp_size) == 3_915
    assert estimate_kv_bytes_per_token(wrapped, tp_size=1, pp_size=1) == 3_915
    with pytest.raises(ValueError, match="resident attention layers"):
        estimate_kv_bytes_per_token(wrapped, tp_size=1, pp_size=4)
    for raw in (config, wrapper):
        estimator = NaiveKVCacheEstimator.from_hf_config(raw, tp_size=1, pp_size=1)
        with pytest.raises(ValueError, match="packed cache capacity requires runtime group allocations"):
            estimator.get_kvcache_max_tokens(27 * 1_002_240)
    with pytest.raises(ValueError, match="DeepSeek V4 stores fp8_ds_mla KV"):
        estimate_kv_bytes_per_token(wrapped, tp_size=1, pp_size=1, kvcache_quant_mode="bfloat16")


def test_naive_estimate_reads_the_first_architecture_name_and_the_text_config_of_any_unknown_document():
    # Only the first name decides whether the architecture is known.
    flat = NaiveKVCacheEstimator.from_hf_config(
        {**_UNKNOWN_DECODER, "architectures": [_UNKNOWN_WRAPPER, "LlamaForCausalLM"]}, tp_size=1, pp_size=1
    )
    assert (flat.architecture, flat.kv_bytes_per_token()) == (_UNKNOWN_WRAPPER, _UNKNOWN_DECODER_KV_BYTES)
    # A document that names no architecture is read through its decoder too.
    unnamed = NaiveKVCacheEstimator.from_hf_config(
        {"hidden_size": 4096, "num_hidden_layers": 48, "num_attention_heads": 32, "text_config": _UNKNOWN_DECODER},
        tp_size=1,
        pp_size=1,
    )
    assert (unnamed.architecture, unnamed.kv_bytes_per_token()) == (None, _UNKNOWN_DECODER_KV_BYTES)
    # With both decoder keys side by side, text_config is the decoder.
    both = NaiveKVCacheEstimator.from_hf_config(
        {
            "architectures": [_UNKNOWN_WRAPPER],
            "text_config": _UNKNOWN_DECODER,
            "llm_config": {**_UNKNOWN_DECODER, "num_hidden_layers": 9},
        },
        tp_size=1,
        pp_size=1,
    )
    assert (both.geometry["layers"], both.kv_bytes_per_token()) == (4, _UNKNOWN_DECODER_KV_BYTES)


@pytest.mark.parametrize(
    ("config", "text"),
    [
        (
            {"architectures": [_UNKNOWN_WRAPPER], "hidden_size": 256, "text_config": None},
            "HF config: text_config must be a nonempty object describing one text decoder",
        ),
        (
            {"architectures": [_UNKNOWN_WRAPPER], "hidden_size": 256, "text_config": {}},
            "HF config: text_config must be a nonempty object describing one text decoder",
        ),
        (
            {
                "architectures": [_UNKNOWN_WRAPPER],
                "text_config": {**_UNKNOWN_DECODER, "llm_config": _UNKNOWN_DECODER},
            },
            "HF config: ambiguous nested llm_config in text_config",
        ),
    ],
    ids=["null-text-config", "empty-text-config", "nested-decoder"],
)
def test_naive_estimate_refuses_a_malformed_decoder_of_an_unknown_wrapper(config, text):
    with pytest.raises(ValueError) as error:
        NaiveKVCacheEstimator.from_hf_config(config, tp_size=1, pp_size=1)
    assert str(error.value) == text
    # The cause is the parser's refusal of the unknown architecture.
    assert type(error.value.__cause__) is ValueError
    assert str(error.value.__cause__).startswith(f"The model's architecture {_UNKNOWN_WRAPPER} is not supported.")


def test_naive_estimate_refuses_a_malformed_text_config_of_a_registered_wrapper():
    config = {"architectures": ["Qwen3_5MoeForConditionalGeneration"], "hidden_size": 256, "text_config": None}
    with pytest.raises(ValueError) as error:
        NaiveKVCacheEstimator.from_hf_config(config, tp_size=1, pp_size=1)
    assert str(error.value).startswith(
        "the config parser refuses the HF config of the supported architecture Qwen3_5MoeForConditionalGeneration: "
        "ValueError: "
    )


def test_the_parser_accepts_every_packaged_model_config():
    from importlib import resources

    from aisim_core.sdk.utils import _parse_hf_config_json

    files = sorted(
        path
        for path in resources.files("aisim_core").joinpath("model_configs").iterdir()
        if path.name.endswith("_config.json") and not path.name.endswith("_hf_quant_config.json")
    )
    refused = {}
    for path in files:
        try:
            _parse_hf_config_json(json.loads(path.read_text()))
        except Exception as error:
            refused[path.name] = repr(error)
    assert len(files) >= 97
    assert refused == {}


def _write_config(folder, config):
    (folder / "config.json").write_text(json.dumps(config))
    return str(folder)


def _missing_system_cache_options(systems_path, *, tp_size, capacity):
    return dict(
        systems_path=str(systems_path),
        max_num_tokens=4096,
        max_batch_size=32,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        gpu_memory_capacity_bytes_override=capacity,
        allow_naive_fallback=True,
        naive_kv_reservation=1,
        tp_size=tp_size,
        dcp_size=1,
        moe_tp_size=1,
        moe_ep_size=tp_size,
        scheduler_block_size=16,
    )


_UNKNOWN_WRAPPER_CONFIG = {
    "architectures": [_UNKNOWN_WRAPPER],
    "hidden_size": 4096,
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "text_config": _UNKNOWN_DECODER,
}


def test_replay_capacity_refuses_an_invalid_known_config_and_reads_the_decoder_of_an_unknown_wrapper(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token

    (tmp_path / "known").mkdir()
    known = _write_config(tmp_path / "known", {"architectures": ["LlamaForCausalLM"], "hidden_size": 128})
    with pytest.raises(ValueError) as error:
        estimate_kv_bytes_per_token(known, tp_size=1, pp_size=1)
    assert str(error.value) == _NO_LAYERS_TEXT
    (tmp_path / "wrapper").mkdir()
    wrapper = _write_config(tmp_path / "wrapper", _UNKNOWN_WRAPPER_CONFIG)
    assert estimate_kv_bytes_per_token(wrapper, tp_size=1, pp_size=1) == _UNKNOWN_DECODER_KV_BYTES


def test_memory_preflight_refuses_an_invalid_known_config_and_reads_the_decoder_of_an_unknown_wrapper(tmp_path):
    def preflight(folder):
        return memory.estimate_kv_cache(
            folder,
            "h200_sxm",
            "vllm",
            max_num_tokens=8192,
            max_batch_size=256,
            memory_fraction_kind="of_total",
            memory_fraction_value=0.9,
            gpu_memory_capacity_bytes_override=200 * _GIB,
            allow_naive_fallback=True,
        )

    (tmp_path / "known").mkdir()
    known = _write_config(tmp_path / "known", {"architectures": ["LlamaForCausalLM"], "hidden_size": 128})
    with pytest.raises(ValueError) as error:
        preflight(known)
    assert str(error.value) == _NO_LAYERS_TEXT
    (tmp_path / "wrapper").mkdir()
    out = preflight(_write_config(tmp_path / "wrapper", _UNKNOWN_WRAPPER_CONFIG))
    assert (out["source"], out["kv_size_per_token_bytes"]) == ("naive_fallback", _UNKNOWN_DECODER_KV_BYTES)


def test_naive_fallback_raises_without_config():
    # HF id that is neither local nor pre-cached, with download disabled -> no
    # config -> raise rather than guess with a placeholder constant.
    with pytest.raises(ValueError, match="insufficient model metadata"):
        _naive_estimate(
            "definitely/not-a-real-model-xyz",
            tp_size=1,
            pp_size=1,
            gpu_memory_capacity_bytes_override=200 * _GIB,
            naive_kv_reservation=memory._DEFAULT_NAIVE_KV_RESERVATION,
            allow_hf_config_download=False,
        )


def test_naive_fallback_raises_on_missing_weight_metadata(monkeypatch):
    # Config present but lacking weight-estimate fields (no vocab_size /
    # intermediate_size) -> raise rather than fabricate a placeholder weight.
    raw = {k: v for k, v in _RAW_UNSUPPORTED.items() if k not in ("vocab_size", "intermediate_size")}
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "_load_config", lambda *a, **k: raw)
    with pytest.raises(ValueError, match="weight-estimate fields"):
        _naive_estimate(
            "foo/bar-unknown-arch",
            tp_size=1,
            pp_size=1,
            gpu_memory_capacity_bytes_override=200 * _GIB,
            naive_kv_reservation=memory._DEFAULT_NAIVE_KV_RESERVATION,
            allow_hf_config_download=False,
        )


def test_naive_fallback_honors_reservation_param(monkeypatch):
    # The reservation fraction is caller-adjustable (default 0.80); a 0.5 reserve
    # halves the post-weight KV budget. Uses a full synthetic config so the weight
    # estimate is available (no placeholder fallback any more).
    capacity = 200 * _GIB
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "_load_config", lambda *a, **k: dict(_RAW_UNSUPPORTED))
    weight = float(
        memory.NaiveKVCacheEstimator(
            memory.NaiveKVCacheEstimator._geometry(dict(_RAW_UNSUPPORTED), None),
            dtype_bytes=memory.NaiveKVCacheEstimator._dtype_bytes(_RAW_UNSUPPORTED),
            tp_size=1,
            pp_size=1,
        ).weight_bytes()
    )
    post_weight = capacity - weight
    out = _naive_estimate(
        "foo/bar-unknown-arch",
        tp_size=1,
        pp_size=1,
        gpu_memory_capacity_bytes_override=capacity,
        naive_kv_reservation=0.5,
        allow_hf_config_download=False,
    )
    assert out["total_kv_size_bytes"] == int(post_weight * 0.5)


def test_naive_fallback_requires_capacity_override():
    # Use a cached model so from_model_path succeeds; the capacity check inside
    # .estimate() is what must raise.
    with pytest.raises(ValueError, match="gpu_memory_capacity_bytes_override"):
        _naive_estimate(
            "deepseek-ai/DeepSeek-V3",
            tp_size=1,
            pp_size=1,
            gpu_memory_capacity_bytes_override=None,
            naive_kv_reservation=memory._DEFAULT_NAIVE_KV_RESERVATION,
            allow_hf_config_download=False,
        )


# An unsupported hybrid sliding-window config: 25 SWA + 5 global layers, window
# 1024. The naive fallback should cap the SWA layers at the window rather than
# extrapolate the seq_len=1 per-token rate forever.
_RAW_HYBRID = {
    "architectures": ["FooSwaForCausalLM"],
    "num_hidden_layers": 30,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_size": 2048,
    "vocab_size": 256_000,
    "intermediate_size": 8192,
    "sliding_window": 1024,
    "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 5,
    "torch_dtype": "bfloat16",
}


def test_naive_fallback_hybrid_beats_linear_capacity(monkeypatch):
    # With a large budget the token count runs well past the window, so the
    # window-capped curve fits strictly more tokens than dividing by the full
    # per-token rate would. kv_size_per_token_bytes (the reported seq=1 rate) is
    # unchanged -- only the capacity inversion is curve-aware.
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "_load_config", lambda *a, **k: dict(_RAW_HYBRID))
    out = _naive_estimate(
        "foo/swa-unknown-arch",
        tp_size=1,
        pp_size=1,
        gpu_memory_capacity_bytes_override=200 * _GIB,
        naive_kv_reservation=memory._DEFAULT_NAIVE_KV_RESERVATION,
        allow_hf_config_download=False,
    )
    assert out["source"] == "naive_fallback"
    linear_tokens = out["total_kv_size_bytes"] // out["kv_size_per_token_bytes"]
    assert out["total_kv_size_tokens"] > linear_tokens


@pytest.mark.parametrize(("batch", "chunk"), [(1, 4096), (4, 128), (32, 8192)])
def test_gpt_oss_scheduler_pool_covers_in_flight_prefill(tmp_path, batch, chunk):
    hidden, vocab, layers, experts = 2880, 201088, 24, 32
    weight = 2 * (2 * vocab * hidden + layers * (4 * hidden**2 + experts * 3 * hidden**2))
    args = ("openai/gpt-oss-20b", "missing-memory-spec", "vllm", "0.33.0")
    options = dict(
        systems_path=str(tmp_path),
        max_num_tokens=chunk,
        max_batch_size=batch,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        gpu_memory_capacity_bytes_override=weight + 96 * 1024**2,
        allow_naive_fallback=True,
        naive_kv_reservation=1.0,
    )
    result = memory.estimate_kv_cache(*args, **options, tolerance_fraction=0.5)
    assert result["source"] == "naive_fallback"
    assert result["total_kv_size_bytes"] == 96 * 1024**2
    assert result["kv_size_per_token_bytes"] == 49_152
    # 3,968 tokens with all 24 layers live during a single prefill require 186 MiB.
    # An unknown block size can consume an entire null row even for a single block.
    capacity, adjusted = 1024, 512  # One full-width data row and one full-width null row.
    assert result["total_kv_size_tokens"] == capacity
    assert result["tolerance_adjusted"]["total_kv_size_tokens"] == adjusted
    assert memory.estimate_kv_cache(*args, **options, scheduler_block_size=16)["total_kv_size_tokens"] == 2032
    assert memory.estimate_num_gpu_blocks(*args, **options, scheduler_block_size=16) == 127
    assert memory.estimate_num_gpu_blocks(*args, **options, scheduler_block_size=16, tolerance_fraction=0.5) == 63


@pytest.mark.parametrize("wrapped", [True, False], ids=["registered-wrapper", "flat-text-export"])
@pytest.mark.parametrize(("mode", "width", "retained"), [("bfloat16", 2, 102_297), ("fp8", 1, 207_155)])
def test_gemma4_cache_rates_follow_each_attention_group(tmp_path, wrapped, mode, width, retained):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("nvidia/Gemma-4-26B-A4B-NVFP4")
    checkpoint = _write_config(tmp_path, raw if wrapped else raw["text_config"])
    estimator = NaiveKVCacheEstimator.from_model_path(checkpoint, tp_size=8, pp_size=1, allow_hf_config_download=False)
    rate = (25 * 512 + 5 * 1024) * width
    assert estimate_kv_bytes_per_token(checkpoint, tp_size=8, pp_size=1, kvcache_quant_mode=mode) == rate
    estimator.set_kvcache_quant_mode(mode)
    # With one request the sliding layers retain 1,024 tokens. Global layers continue to grow.
    assert estimator.get_kvcache_max_tokens(_GIB) == (_GIB - 25 * 512 * width * 1024) // (5 * 1024 * width) == retained
    args = (checkpoint, "missing-memory-spec", "vllm", "0.33.0")
    options = dict(
        systems_path=str(tmp_path),
        max_num_tokens=8192,
        max_batch_size=1,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        tp_size=8,
        allow_naive_fallback=True,
        naive_kv_reservation=1.0,
        kvcache_quant_mode=mode,
        gpu_memory_capacity_bytes_override=estimator.weight_bytes() + _GIB,
    )
    result = memory.estimate_kv_cache(*args, **options)
    assert result["kv_size_per_token_bytes"] == rate
    # Each independent scheduler block can open a page in all 30 layers at the larger per-layer size.
    shared_rate = 30 * 1024 * width
    row_rate = 5 * 1024 * width
    assert result["total_kv_size_tokens"] == _GIB // (shared_rate + row_rate)
    expected = (_GIB - 16 * row_rate) // shared_rate
    assert (
        memory.estimate_kv_cache(*args, **options, scheduler_block_size=16)["total_kv_size_tokens"]
        == expected // 16 * 16
    )
    assert memory.estimate_num_gpu_blocks(*args, **options, scheduler_block_size=16) == expected // 16


@pytest.mark.parametrize("other_groups", [4, 8, 16])
@pytest.mark.parametrize("identity", ["named", "step3p5", "step3p7"])
def test_unknown_wrapper_uses_step_decoder_attention_groups(tmp_path, other_groups, identity):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    decoder = _load_pre_downloaded_hf_config("stepfun-ai/Step-3.7-Flash")
    decoder["architectures"] = ["Step3p5ForCausalLM"]
    decoder["num_attention_groups"] = decoder.pop("num_key_value_heads")
    decoder["attention_other_setting"].update(num_attention_groups=other_groups, head_dim=64)
    # Step reads KV heads and explicit head_dim from the decoder for every attention layer.
    decoder["layer_types"].extend(["sliding_attention"] * decoder["num_nextn_predict_layers"])
    selected = dict(decoder)
    if identity != "named":
        selected.pop("architectures")
        selected["model_type"] = identity
    checkpoint = _write_config(tmp_path, {"architectures": [_UNKNOWN_WRAPPER], "text_config": selected})
    assert estimate_kv_bytes_per_token(checkpoint, tp_size=1, pp_size=1) == 45 * 8 * 128 * 2 * 2 == 184_320
    # SGLang's Step model allocates its 33 sliding layers with the separate KV group count.
    assert (
        estimate_kv_bytes_per_token(checkpoint, tp_size=1, pp_size=1, backend="sglang")
        == (33 * other_groups + 12 * 8) * 128 * 2 * 2
    )
    estimator = NaiveKVCacheEstimator.from_hf_config(decoder, tp_size=1, pp_size=1)
    options = dict(
        systems_path=str(tmp_path),
        max_num_tokens=4096,
        max_batch_size=1,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        allow_naive_fallback=True,
        naive_kv_reservation=1.0,
        gpu_memory_capacity_bytes_override=estimator.weight_bytes() + 64 * 1024**2,
    )
    args = (checkpoint, "missing-memory-spec", "vllm", "0.33.0")
    # A merged allocator can reserve an entire 45-layer null row for a single large block.
    assert memory.estimate_kv_cache(*args, **options)["total_kv_size_tokens"] == (64 << 20) // (2 * 45 * 4096) == 182
    assert memory.estimate_num_gpu_blocks(*args, **options, scheduler_block_size=16) == 21
    sharded = NaiveKVCacheEstimator.from_hf_config(decoder, tp_size=8, pp_size=1, moe_tp_size=1, moe_ep_size=8)
    opts = options | {
        "tp_size": 8,
        "dcp_size": 1,
        "moe_tp_size": 1,
        "moe_ep_size": 8,
        "gpu_memory_capacity_bytes_override": sharded.weight_bytes() + (64 << 20),
    }
    assert memory.estimate_kv_cache(*args, **opts)["total_kv_size_tokens"] == (64 << 20) // (2 * 45 * 512) == 1456
    assert memory.estimate_num_gpu_blocks(*args, **opts, scheduler_block_size=16) == 170
    for function, extra in (
        (memory.estimate_kv_cache, {}),
        (memory.estimate_num_gpu_blocks, {"scheduler_block_size": 16}),
    ):
        with pytest.raises(ValueError, match="separate full and sliding pool sizes"):
            function(checkpoint, "missing-memory-spec", "sglang", "0.5.20", **options, **extra)


def test_step_cache_uses_each_query_group_when_the_decoder_omits_its_head_dimension(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("stepfun-ai/Step-3.7-Flash")
    raw["architectures"] = ["Step3p5ForCausalLM"]
    del raw["head_dim"]
    raw["num_sliding_attention_heads"] = 32
    checkpoint = _write_config(tmp_path, {"architectures": [_UNKNOWN_WRAPPER], "text_config": raw})
    # The runtime derives head_dim from 4096 / query heads separately for 33 sliding and 12 full layers.
    assert estimate_kv_bytes_per_token(checkpoint, tp_size=1, pp_size=1) == 2 * 2 * 8 * (33 * 128 + 12 * 64)


def test_nested_fp8_kv_leaves_the_checkpoint_weight_budget_intact(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    decoder = _load_pre_downloaded_hf_config("openai/gpt-oss-20b")
    checkpoint = _write_config(tmp_path, {"architectures": [_UNKNOWN_WRAPPER], "text_config": decoder})
    hidden, layers, vocab, experts = 2880, 24, 201088, 32
    weights = 2 * (2 * vocab * hidden + layers * (4 * hidden**2 + experts * 3 * hidden**2))
    args = (checkpoint, "missing-memory-spec", "vllm", "0.33.0")
    options = dict(
        systems_path=str(tmp_path),
        max_num_tokens=4096,
        max_batch_size=1,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        allow_naive_fallback=True,
        naive_kv_reservation=1.0,
        gpu_memory_capacity_bytes_override=weights + 3 * 1024**2,
    )
    for mode, rate, tokens in [("bfloat16", 49_152, 32), ("fp8", 24_576, 64)]:
        shared = memory.estimate_kv_cache(*args, **options, kvcache_quant_mode=mode)
        assert shared["total_kv_size_bytes"] == 3 * 1024**2
        assert (
            shared["kv_size_per_token_bytes"]
            == estimate_kv_bytes_per_token(checkpoint, tp_size=1, pp_size=1, kvcache_quant_mode=mode)
            == rate
        )
        assert shared["total_kv_size_tokens"] == tokens
        assert memory.estimate_num_gpu_blocks(*args, **options, kvcache_quant_mode=mode, scheduler_block_size=16) == (
            3 * 1024**2 - 16 * rate
        ) // (16 * rate)


@pytest.mark.parametrize("window", [-128, True, False, 1.5, "128"])
def test_gpt_oss_pool_rejects_malformed_sliding_windows(tmp_path, window):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("openai/gpt-oss-20b")
    model = _write_config(tmp_path, {**raw, "sliding_window": window})
    args = (model, "missing-memory-spec", "vllm", "0.33.0")
    options = dict(
        systems_path=str(tmp_path),
        max_num_tokens=8192,
        max_batch_size=32,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        gpu_memory_capacity_bytes_override=64 * _GIB,
        allow_naive_fallback=True,
    )
    with pytest.raises(ValueError, match="sliding_window must be an integer"):
        memory.estimate_kv_cache(*args, **options)
    with pytest.raises(ValueError, match="sliding_window must be an integer"):
        memory.estimate_num_gpu_blocks(*args, **options, scheduler_block_size=16)


@pytest.mark.parametrize("window", [None, 0])
def test_absent_or_disabled_window_preserves_linear_capacity(window):
    raw = {**_RAW_HYBRID, "sliding_window": window}
    estimator = NaiveKVCacheEstimator.from_hf_config(raw, tp_size=1, pp_size=1)
    assert estimator.mode is memory.NaiveKVCacheMode.LINEAR
    assert estimator.get_kvcache_max_tokens(_GIB, max_batch_size=32) == _GIB // estimator.kv_bytes_per_token()


@pytest.mark.parametrize(
    ("model", "tp", "extra"),
    [
        ("deepseek-ai/DeepSeek-V4-Flash", 8, {}),
        ("deepseek-ai/DeepSeek-V4-Flash", 16, {}),
        ("deepseek-ai/DeepSeek-V4-Flash", 3, {}),
        ("deepseek-ai/DeepSeek-V4-Flash", 8, {"dcp_size": 2}),
        ("deepseek-ai/DeepSeek-V4-Pro", 16, {}),
        ("deepseek-ai/DeepSeek-V4-Pro", 32, {}),
        ("nvidia/DeepSeek-V4-Flash-NVFP4", 8, {}),
    ],
)
def test_a_missing_system_cannot_supply_deepseek_v4_packed_capacity(tmp_path, model, tp, extra):
    args = (model, "missing-memory-spec", "vllm", "0.33.0")
    options = dict(
        systems_path=str(tmp_path),
        tp_size=tp,
        moe_tp_size=1,
        moe_ep_size=tp,
        max_num_tokens=256,
        max_batch_size=1,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        gpu_memory_capacity_bytes_override=20_364_048_808 + 27 * 1_002_240,
        allow_naive_fallback=True,
        naive_kv_reservation=1,
        **extra,
    )
    # Raw capacity and both scheduler page sizes must refuse, including through a tolerance-adjusted estimate.
    for function, args_extra in (
        (memory.estimate_kv_cache, {"tolerance_fraction": 0.1}),
        (memory.estimate_num_gpu_blocks, {"scheduler_block_size": 16}),
        (memory.estimate_num_gpu_blocks, {"scheduler_block_size": 256}),
    ):
        with pytest.raises(ValueError, match="packed cache capacity|context parallelism"):
            function(*args, **options, **args_extra)


@pytest.mark.parametrize("source", ["decoder", "sidecar"])
def test_deepseek_v4_marginal_bytes_respect_the_effective_kv_precision(tmp_path, source):
    from pathlib import Path

    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("deepseek-ai/DeepSeek-V4-Flash")
    if source == "decoder":
        raw["quantization_config"]["kv_cache_quant_algo"] = "NVFP4"
    model = _write_config(tmp_path, {"architectures": [_UNKNOWN_WRAPPER], "text_config": raw})
    if source == "sidecar":
        (Path(model) / "hf_quant_config.json").write_text(
            json.dumps({"quantization": {"kv_cache_quant_algo": "NVFP4"}})
        )
    with pytest.raises(ValueError, match="fp8_ds_mla"):
        estimate_kv_bytes_per_token(model, tp_size=8, pp_size=1)
    assert estimate_kv_bytes_per_token(model, tp_size=8, pp_size=1, kvcache_quant_mode="fp8") == 3915


@pytest.mark.parametrize("tp", [1, 4, 8])
@pytest.mark.parametrize("shape", ["registered", "flat-text", "unknown-wrapper"])
def test_gemma_cache_pages_charge_every_short_request(tmp_path, tp, shape):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("nvidia/Gemma-4-26B-A4B-NVFP4")
    if shape == "flat-text":
        raw = raw["text_config"]
    elif shape == "unknown-wrapper":
        raw = {"architectures": [_UNKNOWN_WRAPPER], "text_config": raw["text_config"]}
    model = _write_config(tmp_path, raw)
    estimator = NaiveKVCacheEstimator.from_hf_config(raw, tp_size=tp, pp_size=1, moe_tp_size=1, moe_ep_size=tp)
    sliding = 2 * (-(-8 // tp)) * 256 * 2
    full = 2 * (-(-2 // tp)) * 512 * 2
    # vLLM uses six groups of five layers. Equal per-layer widths at TP4 still require the null row.
    page_bytes = 16 * max(sliding, full)
    data_bytes = 32 * 30 * page_bytes
    row_bytes = 5 * page_bytes
    options = dict(
        systems_path=str(tmp_path),
        max_num_tokens=512,
        max_batch_size=32,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        allow_naive_fallback=True,
        naive_kv_reservation=1,
        tp_size=tp,
        moe_tp_size=1,
        moe_ep_size=tp,
    )
    options["dcp_size"] = 1
    args = (model, "missing-memory-spec", "vllm", "0.33.0")
    merged_row = 16 * (25 * sliding + 5 * full)
    threshold = max(data_bytes + row_bytes, 33 * merged_row)
    cases = [(data_bytes, 31), (threshold - 1, 31), (threshold, 32)]
    if tp == 1:
        assert data_bytes == 120 * 1024**2
        cases.append((110 * 1024**2, 29))
    if tp == 4:
        assert data_bytes == 30 * 1024**2 and threshold == 32_440_320
    for budget, blocks in cases:
        capacity = estimator.weight_bytes() + budget
        result = memory.estimate_kv_cache(
            *args, **options, scheduler_block_size=16, gpu_memory_capacity_bytes_override=capacity
        )
        assert result["kv_size_per_token_bytes"] == (25 * sliding + 5 * full)
        assert result["total_kv_size_tokens"] == 16 * blocks
        assert (
            memory.estimate_num_gpu_blocks(
                *args, **options, scheduler_block_size=16, gpu_memory_capacity_bytes_override=capacity
            )
            == blocks
        )


@pytest.mark.parametrize("window", [0, 1024])
def test_flat_gemma_layout_covers_the_full_decoder_even_without_retention(tmp_path, window):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("nvidia/Gemma-4-26B-A4B-NVFP4")["text_config"]
    checkpoint = _write_config(tmp_path, raw | {"sliding_window": window, "layer_types": ["full_attention"]})
    with pytest.raises(ValueError, match="1 entries for 30 decoder layers"):
        estimate_kv_bytes_per_token(checkpoint, tp_size=8, pp_size=1)


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"moe_ep_size": 32}, "Parallelism width mismatch"),
        ({"dcp_size": 2}, "context parallelism"),
        ({"cp_size": 2}, "context parallelism"),
        ({"gemm_quant_mode": "not_a_mode"}, "unsupported gemm_quant_mode"),
        ({"moe_quant_mode": ""}, "unsupported moe_quant_mode"),
        ({"kvcache_quant_mode": "fp8_typo"}, "unsupported kvcache_quant_mode"),
        ({"fmha_quant_mode": "fp8_typo"}, "unsupported fmha_quant_mode"),
        ({"comm_quant_mode": "fp8_typo"}, "unsupported comm_quant_mode"),
        ({"moe_tp_size": 0.5, "moe_ep_size": 2}, "moe_tp_size must be a positive integer"),
        ({"nextn": 0.5}, "must be an integer draft length"),
        ({"nextn": True}, "must be an integer draft length"),
        ({"max_batch_size": 1.5}, "max_batch_size must be a positive integer"),
        ({"max_batch_size": True}, "max_batch_size must be a positive integer"),
        ({"max_num_tokens": 0}, "max_num_tokens must be a positive integer"),
        ({"max_num_tokens": True}, "max_num_tokens must be a positive integer"),
        ({"max_num_tokens": 1.5}, "max_num_tokens must be a positive integer"),
        ({"max_num_tokens": "8"}, "max_num_tokens must be a positive integer"),
        ({"tp_size": 3, "moe_ep_size": 3, "dcp_size": 1}, "num_attention_heads 64 must be divisible"),
        ({"attention_dp_size": 64, "moe_ep_size": 64}, "32 experts cannot partition"),
    ],
)
def test_wrapped_moe_public_capacity_validates_the_deployment(tmp_path, override, reason):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("openai/gpt-oss-20b")
    model = _write_config(tmp_path, {"architectures": [_UNKNOWN_WRAPPER], "text_config": raw})
    base = dict(
        systems_path=str(tmp_path),
        max_num_tokens=4096,
        max_batch_size=1,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        moe_tp_size=1,
        moe_ep_size=1,
        allow_naive_fallback=True,
        gpu_memory_capacity_bytes_override=60 * _GIB,
    )
    args = (model, "missing-memory-spec", "vllm", "0.33.0")
    for function, extra in (
        (memory.estimate_kv_cache, {}),
        (memory.estimate_num_gpu_blocks, {"scheduler_block_size": 16}),
    ):
        assert function(*args, **base, **extra)
        with pytest.raises(ValueError, match=reason):
            function(*args, **(base | override), **extra)


@pytest.mark.parametrize(
    "model, row_bytes, groups, logical_blocks",
    [
        ("openai/gpt-oss-20b", 393_216, 2, 128),
        ("Qwen/Qwen3-0.6B", 1_835_008, 1, 32),
    ],
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_vllm_public_scheduler_capacity_counts_full_pages_and_a_null_row(
    tmp_path, model, row_bytes, groups, logical_blocks, wrapped
):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config(model)
    if wrapped:
        model = _write_config(tmp_path, {"architectures": [_UNKNOWN_WRAPPER], "text_config": raw})
    weights = NaiveKVCacheEstimator.from_hf_config(raw, tp_size=1, pp_size=1).weight_bytes()
    options = dict(
        systems_path=str(tmp_path),
        max_num_tokens=4096,
        max_batch_size=1 if groups == 2 else 32,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        allow_naive_fallback=True,
        naive_kv_reservation=1,
        scheduler_block_size=16,
    )
    merged_row = groups * row_bytes
    threshold = (logical_blocks + 1) * merged_row
    args = (model, "missing-memory-spec", "vllm", "0.33.0")
    for budget, expected in [
        (threshold - row_bytes, logical_blocks - 1),
        (threshold - 1, logical_blocks - 1),
        (threshold, logical_blocks),
    ]:
        capacity = weights + budget
        result = memory.estimate_kv_cache(*args, **options, gpu_memory_capacity_bytes_override=capacity)
        assert result["total_kv_size_tokens"] == expected * 16
        assert memory.estimate_num_gpu_blocks(*args, **options, gpu_memory_capacity_bytes_override=capacity) == expected

    # A bare scalar still covers its null row when only one 16-token block fits.
    for budget, tokens in [(2 * merged_row - 1, 15), (2 * merged_row, 16)]:
        bare = {k: v for k, v in options.items() if k != "scheduler_block_size"}
        capacity = weights + budget
        result = memory.estimate_kv_cache(*args, **bare, gpu_memory_capacity_bytes_override=capacity)
        assert result["total_kv_size_tokens"] == tokens
        assert (
            memory.estimate_num_gpu_blocks(*args, **options, gpu_memory_capacity_bytes_override=capacity)
            == tokens // 16
        )


def test_nested_scalar_requires_explicit_dcp_only_when_tp_can_partition_the_cache(tmp_path):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("openai/gpt-oss-20b")
    model = _write_config(tmp_path, {"architectures": [_UNKNOWN_WRAPPER], "text_config": raw})
    args = (model, "missing-memory-spec", "vllm", "0.33.0")
    options = dict(
        systems_path=str(tmp_path),
        max_num_tokens=4096,
        max_batch_size=1,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        gpu_memory_capacity_bytes_override=60 * _GIB,
        allow_naive_fallback=True,
    )
    assert memory.estimate_kv_cache(*args, **options) == memory.estimate_kv_cache(*args, **options, dcp_size=1)
    with pytest.raises(ValueError, match="context parallelism"):
        memory.estimate_kv_cache(*args, **options, dcp_size=2)
    opts = options | {"tp_size": 2, "moe_tp_size": 1, "moe_ep_size": 2}
    with pytest.raises(ValueError, match="requires explicit dcp_size"):
        memory.estimate_kv_cache(*args, **opts)
    assert memory.estimate_kv_cache(*args, **opts, dcp_size=1)["source"] == "naive_fallback"
    with pytest.raises(ValueError, match="context parallelism"):
        memory.estimate_kv_cache(*args, **opts, dcp_size=2)
    assert memory.estimate_num_gpu_blocks(*args, **opts, scheduler_block_size=16) > 0


@pytest.mark.parametrize("wrapped", [False, True])
def test_gemma4_default_layer_split_matches_the_explicit_decoder(tmp_path, wrapped):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    text = _load_pre_downloaded_hf_config("nvidia/Gemma-4-26B-A4B-NVFP4")["text_config"]
    del text["layer_types"]
    model = _write_config(tmp_path, {"architectures": [_UNKNOWN_WRAPPER], "text_config": text} if wrapped else text)
    assert estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1) == 225_280
    weights = NaiveKVCacheEstimator.from_hf_config(text, tp_size=1, pp_size=1).weight_bytes()
    opts = dict(
        max_num_tokens=4096,
        max_batch_size=32,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        allow_naive_fallback=True,
        naive_kv_reservation=1,
        systems_path=str(tmp_path),
        gpu_memory_capacity_bytes_override=weights + (110 << 20),
        scheduler_block_size=16,
    )
    args = (model, "missing", "vllm", "0.33.0")
    assert memory.estimate_kv_cache(*args, **opts)["total_kv_size_tokens"] == 29 * 16
    assert memory.estimate_num_gpu_blocks(*args, **opts) == 29


def test_vllm_padding_uses_the_requested_version_and_a_bound_for_unverified_versions(tmp_path):
    text = {
        **_RAW_HYBRID,
        "num_hidden_layers": 26,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "layer_types": ["sliding_attention"] * 11 + ["full_attention"] * 15,
    }
    model = _write_config(tmp_path, {"text_config": text})
    weights = NaiveKVCacheEstimator.from_hf_config(text, tp_size=1, pp_size=1).weight_bytes()
    opts = dict(
        max_num_tokens=4096,
        max_batch_size=32,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        allow_naive_fallback=True,
        naive_kv_reservation=1,
        systems_path=str(tmp_path),
        gpu_memory_capacity_bytes_override=weights + (64 << 20),
        scheduler_block_size=16,
    )
    # 0.14 allocates three 11-layer groups. 0.33 allocates two 15-layer groups.
    for version, count in [("0.14.0", 123), ("0.33.0", 136), ("current", 123)]:
        args = (model, "missing", "vllm", version)
        assert memory.estimate_num_gpu_blocks(*args, **opts) == count
        assert memory.estimate_kv_cache(*args, **opts)["total_kv_size_tokens"] == count * 16


@pytest.mark.parametrize(
    ("name", "rate", "recurrent"),
    [
        ("nvidia/GLM-5.2-NVFP4", 78 * (512 + 64) * 2, False),
        ("nvidia/GLM-5.3-NVFP4", 78 * (512 + 64) * 2, False),
        ("Qwen/Qwen3.8-2.4T-A95B", 376_832, True),
        ("Qwen/Qwen3.8-2.4T-A95B-FP8", 376_832, True),
    ],
)
def test_non_sliding_decoder_types_retain_their_existing_marginal_geometry(tmp_path, name, rate, recurrent):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config(name)
    wrapped = _write_config(tmp_path, {"architectures": [_UNKNOWN_WRAPPER], "text_config": raw})
    opts = dict(
        max_num_tokens=4096,
        max_batch_size=32,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        allow_naive_fallback=True,
        systems_path=str(tmp_path),
        gpu_memory_capacity_bytes_override=10_000 * _GIB,
        scheduler_block_size=16,
    )
    for model in (name, wrapped):
        if model == name:
            assert estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1) == rate
        elif recurrent:
            with pytest.raises(ValueError, match="requires runtime state reservations"):
                estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1)
        else:
            with pytest.raises(ValueError, match="runtime latent and indexer cache formats"):
                estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1)
            assert estimate_kv_bytes_per_token(
                model, tp_size=1, pp_size=1, backend_version="0.22.0", kvcache_quant_mode="bfloat16"
            ) == 78 * (1024 + 128 + 132)
        for function in (memory.estimate_kv_cache, memory.estimate_num_gpu_blocks):
            if recurrent or model == wrapped:
                reason = "requires runtime state reservations" if recurrent else "requires runtime indexer allocations"
                with pytest.raises(ValueError, match=reason):
                    function(model, "missing", "vllm", **opts)
            else:
                assert function(model, "missing", "vllm", **opts)


@pytest.mark.parametrize(
    ("name", "rate"),
    [
        ("deepseek-ai/DeepSeek-V4-Flash", 3915),
        ("sgl-project/DeepSeek-V4-Pro-FP8", 5610),
        ("sgl-project/DeepSeek-V4-Flash-FP8", 3915),
        ("deepseek-ai/DeepSeek-V4.1-Flash", None),
    ],
)
@pytest.mark.parametrize("named", [False, True])
def test_packed_deepseek_model_type_resolves_the_physical_decoder(tmp_path, name, rate, named):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config(name)
    text = dict(raw.get("text_config") or raw)
    if not named:
        text.pop("architectures", None)
    text["intermediate_size"] = text["moe_intermediate_size"]
    model = _write_config(tmp_path, {"architectures": [_UNKNOWN_WRAPPER], "text_config": text})
    reason = "packed cache capacity" if rate else "compressed KV storage varies with position"
    for backend in ("sglang", "vllm"):
        if rate is None:
            with pytest.raises(ValueError, match=reason):
                estimate_kv_bytes_per_token(model, tp_size=8, pp_size=1, backend=backend)
        else:
            assert estimate_kv_bytes_per_token(model, tp_size=8, pp_size=1, backend=backend) == rate
    opts = dict(
        max_num_tokens=4096,
        max_batch_size=2,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        allow_naive_fallback=True,
        systems_path=str(tmp_path),
        gpu_memory_capacity_bytes_override=192 * _GIB,
        tp_size=8,
        dcp_size=1,
        moe_tp_size=1,
        moe_ep_size=8,
    )
    for function, extra in (
        (memory.estimate_kv_cache, {}),
        (memory.estimate_num_gpu_blocks, {"scheduler_block_size": 16}),
    ):
        with pytest.raises(ValueError, match=reason):
            function(model, "missing", "vllm", "0.33.0", **opts, **extra)


def test_packed_deepseek_validates_the_complete_checkpoint_ratio_array(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("sgl-project/DeepSeek-V4-Flash-FP8")
    raw.pop("architectures")
    layers = raw["num_hidden_layers"]
    ratios = raw["compress_ratios"]
    assert len(ratios) == layers + 1 and ratios[-1] == 0
    for label, values, supported in (
        ("backbone", ratios[:layers], True),
        ("backbone-and-mtp", ratios, True),
        ("missing-layer", ratios[: layers - 1], False),
        ("extra-layer", [*ratios, 0], False),
        ("boolean", [False, *ratios[1:]], False),
        ("invalid-ratio", [3, *ratios[1:]], False),
        ("compressed-draft", [*ratios[:-1], 4], False),
    ):
        path = tmp_path / label
        path.mkdir()
        model = _write_config(path, {"text_config": {**raw, "compress_ratios": values}})
        if supported:
            assert estimate_kv_bytes_per_token(model, tp_size=8, pp_size=1) == 3915
        else:
            with pytest.raises(ValueError, match="compress_ratios"):
                estimate_kv_bytes_per_token(model, tp_size=8, pp_size=1)


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "runtime state reservations"),
        ("moonshotai/Kimi-K3", "runtime state reservations"),
        ("Qwen/Qwen3.5-35B-A3B", "runtime state reservations"),
        ("XiaomiMiMo/MiMo-V2-Flash", "runtime cache geometry"),
    ],
)
@pytest.mark.parametrize("named", [False, True])
def test_nested_state_or_per_layer_layout_requires_physical_memory(tmp_path, name, reason, named):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config(name)
    text = dict(raw.get("text_config") or raw)
    if not named:
        text.pop("architectures", None)
    model = _write_config(tmp_path, {"architectures": [_UNKNOWN_WRAPPER], "text_config": text})
    with pytest.raises(ValueError, match=reason):
        estimate_kv_bytes_per_token(model, tp_size=8, pp_size=1)
    opts = _missing_system_cache_options(tmp_path, tp_size=8, capacity=2_000 * _GIB)
    for fn in (memory.estimate_kv_cache, memory.estimate_num_gpu_blocks):
        with pytest.raises(ValueError, match=reason):
            fn(model, "missing", "sglang", "0.5.20", **opts)


def test_selected_qwen_and_gemma_physical_defaults(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    qwen = _load_pre_downloaded_hf_config("Qwen/Qwen3-30B-A3B")
    del qwen["num_experts"], qwen["moe_intermediate_size"]
    model = _write_config(tmp_path, {"text_config": qwen})
    opts = _missing_system_cache_options(tmp_path, tp_size=1, capacity=24 * _GIB)
    with pytest.raises(ValueError, match="no KV budget"):
        memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.20", **opts)

    mixtral = _load_pre_downloaded_hf_config("mistralai/Mixtral-8x7B-v0.1")
    del mixtral["num_local_experts"]
    model = _write_config(tmp_path, {"text_config": mixtral})
    with pytest.raises(ValueError, match="no KV budget"):
        memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.20", **opts)
    for text, fields, count in (
        (mixtral, {"num_experts": 3, "num_local_experts": 2}, 3),
        (qwen, {"num_experts": 3, "num_local_experts": 2}, 2),
    ):
        estimator = memory.NaiveKVCacheEstimator.from_hf_config(
            {"text_config": {**text, **fields}}, tp_size=1, pp_size=1
        )
        assert estimator.geometry["num_experts"] == count

    qwen8b = _load_pre_downloaded_hf_config("Qwen/Qwen3-8B")
    del qwen8b["architectures"], qwen8b["num_key_value_heads"]
    model = _write_config(tmp_path, {"text_config": dict(qwen8b, n_kv_heads=8)})
    assert estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1) == 589_824

    dense = _load_pre_downloaded_hf_config("Qwen/Qwen3-0.6B")
    dense.pop("head_dim")
    dense.update(attention_head_dim=64, use_sliding_window=True, max_window_layers=0)
    dense.pop("layer_types", None)
    dense.pop("sliding_window", None)
    model = _write_config(tmp_path, {"text_config": dense})
    assert estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1) == 114_688
    dense["max_window_layers"] = 5
    _write_config(tmp_path, {"text_config": dense})
    with pytest.raises(ValueError, match="vLLM Qwen3 does not support"):
        estimate_kv_bytes_per_token(model, tp_size=1, pp_size=2)
    dense["sliding_window"] = None
    _write_config(tmp_path, {"text_config": dense})
    assert estimate_kv_bytes_per_token(model, tp_size=1, pp_size=2) == 57_344

    checkpoint = _load_pre_downloaded_hf_config("google/gemma-4-26B-A4B")
    gemma = checkpoint["text_config"]
    for flag, expected in ((True, 225_280), (False, 286_720), (None, 286_720)):
        gemma["attention_k_eq_v"] = flag
        if flag is None:
            del gemma["attention_k_eq_v"]
        for document in (checkpoint, dict(gemma, architectures=["Gemma4ForCausalLM"]), {"text_config": gemma}):
            model = _write_config(tmp_path, document)
            assert estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1) == expected
            assert (
                estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1, backend="sglang", backend_version="0.5.20")
                == 225_280
            )

    gemma.update(attention_k_eq_v=True, attention_head_dim=64)
    del gemma["head_dim"]
    model = _write_config(tmp_path, {"text_config": gemma})
    assert estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1) == 225_280
    gemma["head_dim"] = 256
    gemma["per_layer_config"] = {
        f"{i:02}": {"head_dim": 128, "num_key_value_heads": 16} for i in range(gemma["num_hidden_layers"])
    }
    gemma["per_layer_config"]["00"]["head_dim"] = 256
    for document in (checkpoint, {"text_config": gemma}):
        model = _write_config(tmp_path, document)
        with pytest.raises(ValueError, match="one Gemma 4 cache geometry"):
            estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1, backend_version="0.33.0")
        with pytest.raises(ValueError, match="one Gemma 4 cache geometry"):
            memory.estimate_num_gpu_blocks(model, "missing", "vllm", "0.33.0", **opts)


@pytest.mark.parametrize("registered", [True, False])
def test_sglang_gemma4_uses_the_layer_shapes_of_its_release(tmp_path, registered):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.config_builders import build_model_config
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    document = _load_pre_downloaded_hf_config("google/gemma-4-26B-A4B")
    raw = document["text_config"]
    raw.update(
        layer_types=["full_attention"] * 30,
        head_dim=256,
        num_key_value_heads=8,
        global_head_dim=128,
        num_global_key_value_heads=2,
        attention_k_eq_v=False,
        per_layer_config={str(i): {"head_dim": 128, "num_key_value_heads": 16} for i in range(30)},
    )
    model = _write_config(tmp_path, document if registered else {"text_config": raw})
    opts = _missing_system_cache_options(tmp_path, tp_size=1, capacity=50_531_926_016 + (110 << 20))
    opts["memory_fraction_value"] = 1.0
    for version, width in (("0.5.20", 30_720), ("0.5.21", 30_720), ("0.5.22", 245_760), ("0.5.23", 245_760)):
        assert (
            estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1, backend="sglang", backend_version=version) == width
        )
        if registered:
            native = get_model(
                model,
                build_model_config(tp_size=1, pp_size=1, attention_dp_size=1, moe_tp_size=1, moe_ep_size=1),
                "sglang",
                version,
            )
            assert native.get_kvcache_bytes_per_sequence(1) == width
            assert memory.estimate_num_gpu_blocks(model, "missing", "sglang", version, **opts) == (110 << 20) // (
                16 * width
            )
    raw.update(head_dim=128, num_global_key_value_heads=8)
    for fields in raw["per_layer_config"].values():
        fields["num_key_value_heads"] = 8
    _write_config(tmp_path, document if registered else {"text_config": raw})
    assert (
        estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1, backend="sglang", backend_version="0.5.20") == 122_880
    )
    for value_dim in (128, 256):
        raw["v_head_dim"] = value_dim
        _write_config(tmp_path, document if registered else {"text_config": raw})
        for version in ("0.5.20", "0.5.23"):
            if value_dim == 128:
                assert (
                    estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1, backend="sglang", backend_version=version)
                    == 122_880
                )
            else:
                with pytest.raises(ValueError, match="asymmetric SGLang Gemma 4"):
                    memory.estimate_num_gpu_blocks(model, "missing", "sglang", version, **opts)
    del raw["v_head_dim"]
    raw["per_layer_config"]["0"]["num_key_value_heads"] = 16
    _write_config(tmp_path, document if registered else {"text_config": raw})
    with pytest.raises(ValueError, match="one Gemma 4 cache geometry"):
        estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1, backend="sglang", backend_version="0.5.23")


def test_registered_gemma4_flat_and_per_layer_release_selection(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.config_builders import build_model_config
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    document = _load_pre_downloaded_hf_config("google/gemma-4-26B-A4B")
    text = document["text_config"]
    text["per_layer_config"] = {
        str(i): {"head_dim": 512, "num_key_value_heads": 2}
        for i, kind in enumerate(text["layer_types"])
        if kind == "full_attention"
    }
    path = _write_config(tmp_path, document)
    config = build_model_config(tp_size=1, pp_size=1, attention_dp_size=1, moe_tp_size=1, moe_ep_size=1)
    for backend in ("sglang", "vllm"):
        assert get_model(path, config, backend).get_kvcache_bytes_per_sequence(1) == 225_280

    for override in text["per_layer_config"].values():
        override.update(num_key_value_heads=4, head_dim=128)
    different_path = tmp_path / "overridden"
    different_path.mkdir()
    path = _write_config(different_path, document)
    for backend in ("sglang", "vllm"):
        with pytest.raises(ValueError, match="release"):
            get_model(path, config, backend)
    for version, width in (("0.26.0", 225_280), ("0.32.0", 215_040)):
        assert estimate_kv_bytes_per_token(path, tp_size=1, pp_size=1, backend_version=version) == width
        assert get_model(path, config, "vllm", version).get_kvcache_bytes_per_sequence(1) == width
    with pytest.raises(ValueError, match="runtime profile"):
        get_model(path, config, "vllm", "0.28.0rc1")
    with pytest.raises(ValueError, match="runtime profile"):
        estimate_kv_bytes_per_token(path, tp_size=1, pp_size=1, backend_version="0.28.0rc1")
    with pytest.raises(ValueError, match="cannot read global Gemma 4 heads"):
        get_model(path, config, "vllm", "0.27.1")
    text["allow_global_per_layer_attribute_access"] = True
    legacy_path = tmp_path / "global-access"
    legacy_path.mkdir()
    path = _write_config(legacy_path, document)
    assert get_model(path, config, "vllm", "0.27.1").get_kvcache_bytes_per_sequence(1) == 225_280


@pytest.mark.parametrize("serialized", [False, True])
def test_gemma4_ignores_layer_overrides_equal_to_global_attributes(tmp_path, serialized):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.config import ModelConfig
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    document = _load_pre_downloaded_hf_config("google/gemma-4-26B-A4B")
    text = document["text_config"]
    layers = text["per_layer_config"] = {
        str(i): {"num_key_value_heads": text["num_global_key_value_heads"], "head_dim": text["global_head_dim"]}
        for i, kind in enumerate(text["layer_types"])
        if kind == "full_attention"
    }
    indices = range(text["num_hidden_layers"]) if serialized else (0,)
    epsilon = text.pop("rms_norm_eps")  # Gemma4TextConfig restores the same global default.
    for i in indices:
        layers.setdefault(str(i), {}).update(
            intermediate_size=text["intermediate_size"], rms_norm_eps=epsilon, return_dict=True
        )
    if serialized:
        del text["global_head_dim"], text["num_global_key_value_heads"]
    else:
        layers["0"]["skip"] = []
    path = _write_config(tmp_path, document)
    for backend, version in (("vllm", "0.32.0"), ("sglang", "0.5.23")):
        assert (
            estimate_kv_bytes_per_token(path, tp_size=1, pp_size=1, backend=backend, backend_version=version) == 225_280
        )

    config = ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1)
    if serialized:
        with pytest.raises(ValueError, match="release"):
            get_model(path, config, "vllm")
        for version in ("0.28.0", "0.29.0"):
            with pytest.raises(ValueError, match="runtime profile"):
                estimate_kv_bytes_per_token(path, tp_size=1, pp_size=1, backend_version=version)
            with pytest.raises(ValueError, match="runtime profile"):
                get_model(path, config, "vllm", version)
    else:
        assert get_model(path, config, "vllm").get_kvcache_bytes_per_sequence(1) == 225_280
        for version in ("0.26.0", "0.28.0", "0.29.0"):
            assert estimate_kv_bytes_per_token(path, tp_size=1, pp_size=1, backend_version=version) == 225_280

    layers["0"]["intermediate_size"] *= 2
    other = tmp_path / "heterogeneous"
    other.mkdir()
    with pytest.raises(ValueError, match="per_layer_config"):
        estimate_kv_bytes_per_token(_write_config(other, document), tp_size=1, pp_size=1, backend_version="0.32.0")


@pytest.mark.parametrize(
    ("field", "value"), [("attention_dropout", 0.1), ("rms_norm_eps", 2e-6), ("intermediate_size", 4224)]
)
def test_gemma4_uses_the_declared_serving_reader_for_surviving_overrides(tmp_path, field, value):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    document = _load_pre_downloaded_hf_config("google/gemma-4-26B-A4B")
    text = document["text_config"]
    text["per_layer_config"] = {
        str(i): {"num_key_value_heads": text["num_global_key_value_heads"], "head_dim": text["global_head_dim"]}
        for i, kind in enumerate(text["layer_types"])
        if kind == "full_attention"
    }
    text["per_layer_config"]["0"] = {field: value}
    for allow in (False, True):
        text["allow_global_per_layer_attribute_access"] = allow
        model = _write_config(tmp_path, document)
        for backend, release in (("vllm", "0.33.0"), ("sglang", "0.5.23"), ("sglang", "0.5.21")):
            args = dict(tp_size=1, pp_size=1, backend=backend, backend_version=release)
            if release == "0.5.21" or (backend == "vllm" and (allow or field == "attention_dropout")):
                assert estimate_kv_bytes_per_token(model, **args) == 225_280
            else:
                with pytest.raises(ValueError, match="per_layer_config"):
                    estimate_kv_bytes_per_token(model, **args)


@pytest.mark.parametrize(
    ("missing", "reason"),
    [("vocab_size", "KeyError: 'vocab_size'"), ("num_hidden_layers", "or a parseable layer pattern")],
)
def test_registered_gemma4_parser_error_precedes_runtime_evidence(tmp_path, missing, reason):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    document = _load_pre_downloaded_hf_config("google/gemma-4-26B-A4B")
    text = document["text_config"]
    text["per_layer_config"] = {
        str(i): {"head_dim": 512, "num_key_value_heads": 2}
        for i, kind in enumerate(text["layer_types"])
        if kind == "full_attention"
    }
    del text["global_head_dim"], text["num_global_key_value_heads"], text[missing]
    with pytest.raises(ValueError, match=reason):
        NaiveKVCacheEstimator.from_hf_config(document, tp_size=1, pp_size=1, backend_version="0.28.0")
    with pytest.raises(ValueError, match=reason):
        estimate_kv_bytes_per_token(_write_config(tmp_path, document), tp_size=1, pp_size=1, backend_version="0.28.0")


@pytest.mark.parametrize("backend,release", [("vllm", "0.32.0"), ("sglang", "0.5.23")])
def test_gemma4_per_layer_fields_use_the_initialized_configuration(tmp_path, backend, release):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    document = _load_pre_downloaded_hf_config("google/gemma-4-26B-A4B")
    text = document["text_config"]
    text["use_bidirectional_attention"] = "all"
    text["return_dict"] = False
    text["_output_attentions"] = True
    text["per_layer_config"] = {
        str(i): {"head_dim": text["global_head_dim"], "num_key_value_heads": text["num_global_key_value_heads"]}
        for i, kind in enumerate(text["layer_types"])
        if kind == "full_attention"
    }
    fields = text["per_layer_config"]["0"] = {
        "is_causal": False,
        "num_labels": 2,
        "label2id": {"LABEL_0": 0, "LABEL_1": 1},
        "allow_global_per_layer_attribute_access": False,
        "keys_to_ignore_at_inference": ["past_key_values"],
        "serialize_explicit_per_layer_config": False,
        "per_layer_attributes": None,
        "name_or_path": text.get("_name_or_path", ""),
        "use_return_dict": False,
        "output_attentions": True,
        "attribute_map": {},
        "has_no_defaults_at_init": False,
        "_attn_implementation": None,
    }
    options = dict(tp_size=1, pp_size=1, backend=backend, backend_version=release)
    assert estimate_kv_bytes_per_token(_write_config(tmp_path, document), **options) == 225_280
    fields["default_rope_type"] = "default"
    if backend == "vllm":
        with pytest.raises(ValueError, match="runtime profile"):
            estimate_kv_bytes_per_token(_write_config(tmp_path, document), **options)
        with pytest.raises(ValueError, match="no global default_rope_type"):
            estimate_kv_bytes_per_token(_write_config(tmp_path, document), **(options | {"backend_version": "0.30.0"}))
        text["default_rope_type"] = "default"
    rope = text["rope_scaling"] = {
        **text["rope_parameters"],
        "full_attention": {"rope_type": "default", "rope_theta": 2e5},
    }
    fields["rope_scaling"] = rope
    assert estimate_kv_bytes_per_token(_write_config(tmp_path, document), **options) == 225_280
    fields["is_causal"] = True
    model = _write_config(tmp_path, document)
    if backend == "sglang":
        with pytest.raises(ValueError, match="override for is_causal"):
            estimate_kv_bytes_per_token(model, **options)
    else:
        assert estimate_kv_bytes_per_token(model, **options) == 225_280


def test_vllm_gemma_flat_and_per_layer_configs_need_the_effective_rope_mapping(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    document = _load_pre_downloaded_hf_config("google/gemma-4-26B-A4B")
    text = document["text_config"]
    for per_layer in ({}, {"per_layer_config": None}, {"per_layer_config": {}}):
        child = {k: v for k, v in text.items() if k not in ("per_layer_config", "rope_scaling")}
        child.update(per_layer)
        for version in ("0.31.0", "0.32.0", "0.33.0"):
            options = dict(tp_size=1, pp_size=1, backend="vllm", backend_version=version)
            if not per_layer:
                model = _write_config(tmp_path, document | {"text_config": child})
                width = estimate_kv_bytes_per_token(model, **options)
                assert width > 0 and _compiled_vllm_transfer_widths(model, version) == (width, width)
                flat = {"rope_type": "default", "rope_theta": 10_000.0}
                model = _write_config(tmp_path, document | {"text_config": child | {"rope_scaling": flat}})
                assert _compiled_vllm_transfer_widths(model, version) == (width, width)
            nested = {"default": {"rope_type": "default", "rope_theta": 10000}}
            model = _write_config(tmp_path, document | {"text_config": child | {"rope_scaling": nested}})
            with pytest.raises(ValueError, match="flat rope_parameters"):
                _compiled_vllm_transfer_widths(model, version)
            model = _write_config(tmp_path, document | {"text_config": child | {"rope_scaling": None}})
            with pytest.raises(ValueError, match="rope_parameters to be a mapping"):
                estimate_kv_bytes_per_token(model, **options)
            with pytest.raises(ValueError, match="rope_parameters to be a mapping"):
                _compiled_vllm_transfer_widths(model, version)
        for kind in ("full_attention", "sliding_attention"):
            partial = {k: v for k, v in text["rope_parameters"].items() if k != kind}
            for rope in (partial, partial | {kind: None}):
                model = _write_config(tmp_path, document | {"text_config": child | {"rope_scaling": rope}})
                with pytest.raises(ValueError, match=rf"rope_parameters\.{kind} to be a mapping"):
                    _compiled_vllm_transfer_widths(model, "0.32.0")


def test_gemma4_calibrated_xpu_adapter_uses_the_official_image(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.config import ModelConfig
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.speculation import SpeculationConfig
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    document = _load_pre_downloaded_hf_config("google/gemma-4-26B-A4B")
    text = document["text_config"]
    heads = text.pop("num_global_key_value_heads")
    dim = text.pop("global_head_dim")
    text["per_layer_config"] = {
        str(i): {"head_dim": dim, "num_key_value_heads": heads}
        for i, kind in enumerate(text["layer_types"])
        if kind == "full_attention"
    }
    path = _write_config(tmp_path, document)
    (tmp_path / "test_xpu.yaml").write_text("data_dir: data/test_xpu\ngpu: {name: Intel B60}\n")
    directory = tmp_path / "data/test_xpu/gemm/vllm/0.28.0"
    directory.mkdir(parents=True)
    runtime = {
        "framework": "vllm_xpu",
        "version": "0.28.0",
        "image": "vllm/vllm-openai-xpu:v0.28.0",
        "image_digest": "sha256:4756b66a077627133cee653b551f6f5eaa1b9a981b5eea13edd33fcd3b0d3ca3",
    }
    args = dict(pp_size=1, backend_version="0.28.0", system="test_xpu", systems_paths=[str(tmp_path)])
    with pytest.raises(ValueError, match="runtime profile"):
        estimate_kv_bytes_per_token(path, tp_size=1, **args)
    (directory / "gemm_perf.txt").write_text("m,n,k,latency\n1,1,1,1\n")
    metadata = directory / "collection_meta.yaml"
    table = {"gemm_perf": {"rows": 1, "status": "complete"}}
    index = runtime["image_digest"]
    manifest = "sha256:d3e9c5f146a8251c7489107e6353cf512fcd1bf14a16cab17148125a62a537c8"
    database = memory.perf_database.PerfDatabase("test_xpu", "vllm", "0.28.0", systems_root=str(tmp_path))
    config = ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1)
    with pytest.raises(ValueError, match="Model backend sglang differs from database backend vllm"):
        get_model(path, config, "sglang", database=database)
    with pytest.raises(ValueError, match="Model release 0.30.0 differs from database release 0.28.0"):
        get_model(path, config, "vllm", "0.30.0", database=database)
    for change, stock in (
        ({}, True),
        ({"image": runtime["image"] + "@" + index, "image_digest": manifest}, True),
        ({"image": "index.docker.io/" + runtime["image"] + "@" + index, "image_digest": manifest}, True),
        ({"image": "registry-1.docker.io/" + runtime["image"] + "@" + manifest, "image_digest": manifest}, True),
        ({"image": "example.org/" + runtime["image"] + "@" + index, "image_digest": manifest}, False),
        ({"image": runtime["image"] + "@" + manifest, "image_digest": index}, False),
        ({"framework": "vllm"}, False),
        ({"version": "0.29.0"}, False),
        ({"image": "custom/vllm:v0.28.0"}, False),
        ({"image_digest": "sha256:" + "0" * 64}, False),
    ):
        metadata.write_text(json.dumps({"schema_version": 1, "runtime": runtime | change, "tables": table}))
        if not stock:
            with pytest.raises(ValueError, match="runtime profile"):
                estimate_kv_bytes_per_token(path, tp_size=1, **args)
            with pytest.raises(ValueError, match="runtime profile"):
                get_model(path, config, "vllm", database=database)
        else:
            assert estimate_kv_bytes_per_token(path, tp_size=1, **args) == 225_280
            assert estimate_kv_bytes_per_token(path, tp_size=4, **args) == 61_440
            assert get_model(path, config, "vllm", database=database).get_kvcache_bytes_per_sequence(1) == 225_280
    first = tmp_path / "first-root"
    incomplete = first / directory.relative_to(tmp_path)
    incomplete.mkdir(parents=True)
    (first / "test_xpu.yaml").write_bytes((tmp_path / "test_xpu.yaml").read_bytes())
    (incomplete / "INCOMPLETE.txt").write_text("collection failed")
    metadata.write_text(json.dumps({"schema_version": 1, "runtime": runtime, "tables": table}))
    draft_config = ModelConfig(
        tp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        speculation=SpeculationConfig("draft_model", {"num_speculative_tokens": 2}, draft_model_path=path),
    )
    target = "google/gemma-4-26B-A4B"
    for kwargs in ({"database": database}, {"backend_version": "0.32.0"}):
        model = get_model(target, draft_config, "vllm", **kwargs)
        assert model.spec_scheme.draft_kv_bytes_per_sequence(model, 1) == 225_280
    assert (
        estimate_kv_bytes_per_token(path, tp_size=4, **(args | {"systems_paths": [str(first), str(tmp_path)]}))
        == 61_440
    )
    event = dict(collector_ref="test", collector_hash="test", case_plan_hash="test", collected_at="2026-01-01")
    custom = runtime | {"image": "custom/vllm:v0.28.0"}
    for default, measured, stock in (
        (custom, [runtime], True),
        (runtime, [custom], False),
        (runtime, [custom, runtime], False),
        (runtime, [None], True),
    ):
        events = [event | table["gemm_perf"] | ({"runtime": value} if value else {}) for value in measured]
        tables = {"gemm_perf": {"status": "complete", "rows": len(events), "collections": events}}
        metadata.write_text(json.dumps({"schema_version": 2, "runtime": default, "tables": tables}))
        if stock:
            assert estimate_kv_bytes_per_token(path, tp_size=4, **args) == 61_440
        else:
            with pytest.raises(ValueError, match="runtime profile"):
                estimate_kv_bytes_per_token(path, tp_size=4, **args)
    # A reduced schema-1 cell may omit rows. A custom retained measurement
    # contradicts the official GEMM cell; an empty Parquet table contributes no evidence.
    import pyarrow as pa
    import pyarrow.parquet as pq

    attention = tmp_path / "data/test_xpu/attention/vllm/0.28.0"
    attention.mkdir(parents=True)
    metadata.write_text(json.dumps({"schema_version": 1, "runtime": runtime, "tables": table}))
    (attention / "collection_meta.yaml").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provenance": "local",
                "runtime": custom,
                "tables": {"generation_attention_perf": {"status": "complete"}},
            }
        )
    )
    retained = attention / "generation_attention_perf.parquet"
    for latencies in ([0.5], []):
        pq.write_table(pa.table({"latency": pa.array(latencies, type=pa.float32())}), retained)
        if latencies:
            with pytest.raises(ValueError, match="runtime profile"):
                get_model(path, config, "vllm", database=database)
        else:
            assert get_model(path, config, "vllm", database=database).get_kvcache_bytes_per_sequence(1) == 225_280
    # Only a collector-valid reduced measurement can qualify a stock image.
    gemm_table = directory / "gemm_perf.txt"
    saved_gemm = gemm_table.rename(tmp_path / gemm_table.name)
    reduced = {"schema_version": 1, "provenance": "local", "runtime": runtime}
    pq.write_table(pa.table({"latency": [0.5]}), retained)
    for entry in ({}, {"status": "failed"}, {"status": "partial"}):
        (attention / "collection_meta.yaml").write_text(
            json.dumps(reduced | {"tables": {"generation_attention_perf": entry}})
        )
        if entry.get("status") == "partial":
            from unittest.mock import patch

            # A new file revision needs one footer read, shared by repeat requests.
            pq.write_table(pa.table({"latency": [0.6]}), retained)
            with patch.object(pq, "read_metadata", wraps=pq.read_metadata) as footer:
                for _ in range(2):
                    assert (
                        get_model(path, config, "vllm", database=database).get_kvcache_bytes_per_sequence(1) == 225_280
                    )
                assert footer.call_count == 1
            empty = attention / "empty.parquet"
            pq.write_table(pa.table({"latency": pa.array([], type=pa.float32())}), empty)
            empty.replace(retained)
        with pytest.raises(ValueError, match="runtime profile"):
            get_model(path, config, "vllm", database=database)
    # A malformed nonempty measurement must also prevent other stock tables from qualifying the cell.
    saved_gemm.rename(gemm_table)
    metadata.write_text(json.dumps({"schema_version": 1, "runtime": runtime, "tables": table}))
    pq.write_table(pa.table({"latency": [0.5]}), retained)
    for invalid in (
        {"schema_version": True},
        {"schema_version": 1.0},
        {"extra_field": True},
        {"runtime": runtime | {"extra_field": True}},
        {"runtime": runtime | {"source_commit": 42}},
        {"runtime": runtime | {"transport": "nccl"}},
    ):
        (attention / "collection_meta.yaml").write_text(
            json.dumps(reduced | invalid | {"tables": {"generation_attention_perf": {"status": "complete"}}})
        )
        with pytest.raises(ValueError, match="runtime profile"):
            get_model(path, config, "vllm", database=database)
    for entry in ({}, {"status": "failed", "rows": 2}, {"status": "complete", "rows": 2, "collected_at": None}):
        (attention / "collection_meta.yaml").write_text(
            json.dumps(reduced | {"runtime": custom, "tables": {"generation_attention_perf": entry}})
        )
        with pytest.raises(ValueError, match="runtime profile"):
            get_model(path, config, "vllm", database=database)
    (directory / "gemm_perf.txt").unlink()
    with pytest.raises(ValueError, match="runtime profile"):
        estimate_kv_bytes_per_token(path, tp_size=1, **args)
    with pytest.raises(ValueError, match="runtime profile"):
        get_model(target, draft_config, "vllm", database=database)


def test_strict_catalog_refusal_precedes_naive_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("AIC_STRICT_PROVENANCE", "1")
    (tmp_path / "test_h100.yaml").write_text("data_dir: data/test_h100\ngpu: {name: NVIDIA H100}\n")
    directory = tmp_path / "data/test_h100/gemm/vllm/0.30.0"
    directory.mkdir(parents=True)
    (directory / "collection_meta.yaml").write_text("tables: [")
    (directory / "gemm_perf.txt").write_text("m,n,k,latency\n1,1,1,1\n")
    options = _missing_system_cache_options(tmp_path, tp_size=1, capacity=80 * _GIB)
    with pytest.raises(memory.perf_database.StrictProvenanceError, match="failed to parse collection_meta"):
        memory.estimate_num_gpu_blocks("facebook/opt-125m", "test_h100", "vllm", "0.30.0", **options)


def test_gemma4_v1_retained_table_uses_the_directory_runtime_and_physical_rows(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisim_core.sdk.models.gemma4 import _gemma4_collection_images

    version = "0.30.0"
    runtime = dict(
        framework="vllm",
        version=version,
        image="vllm/vllm-openai:v0.30.0",
        image_digest="sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90",
    )
    entry = dict(collector_ref="rev", collector_hash="h", case_plan_hash="p", collected_at="2026-01-01", rows=1)
    name = "context_attention_perf"
    sidecar = tmp_path / "collection_meta.yaml"
    # The selected Parquet takes precedence over a retained legacy text file for the same table.
    (tmp_path / f"{name}.txt").write_text("latency\n1\n")
    parquet = tmp_path / f"{name}.parquet"
    pq.write_table(pa.table({"latency": [1]}), parquet)
    for status in ("complete", "partial"):
        for change, table, expected in (
            ({}, entry, [True]),
            ({}, entry | {"rows": 0}, [False]),
            ({}, entry | {"rows": 2}, [False]),
            ({"framework": False}, entry, [False]),
            ({"framework": "sglang"}, entry, []),
            ({}, entry | {"runtime": {"framework": "sglang"}}, [False]),
            ({}, entry | {"runtime": {"framework": False}}, [False]),
            ({}, None, [False]),
        ):
            tables = {name: table | {"status": status}} if table is not None else {}
            sidecar.write_text(json.dumps({"schema_version": 1, "runtime": runtime | change, "tables": tables}))
            assert list(_gemma4_collection_images(tmp_path, version, "vllm", cache=False)) == expected
    for tier, summary, expected in (
        ({}, {"rows": 1}, [True]),
        ({"provenance": "local"}, {"rows": 1}, [False]),
        ({"provenance": "other"}, {"rows": 1}, [False]),
        ({"provenance": "collected"}, {"rows": 1, "collected_at": "2026-01-01"}, [True]),
    ):
        tables = {name: {"status": "complete", **summary}}
        sidecar.write_text(json.dumps({"schema_version": 1, "runtime": runtime, "tables": tables, **tier}))
        assert list(_gemma4_collection_images(tmp_path, version, "vllm", cache=False)) == expected
    sidecar.write_text(json.dumps({"schema_version": 1, "runtime": runtime, "tables": []}))
    assert list(_gemma4_collection_images(tmp_path, version, "vllm", cache=False)) == [False]

    sidecar.unlink()
    assert list(_gemma4_collection_images(tmp_path, version, "vllm", cache=False)) == [False]
    tables = {name: entry | {"status": "complete"}}
    sidecar.write_text(json.dumps({"schema_version": 1, "runtime": runtime, "tables": tables}))
    pq.write_table(pa.table({"latency": pa.array([], type=pa.int64())}), parquet)
    assert list(_gemma4_collection_images(tmp_path, version, "vllm", cache=False)) == []
    parquet.unlink()
    assert list(_gemma4_collection_images(tmp_path, version, "vllm", cache=False)) == [True]


def test_gemma4_v2_per_event_runtime_and_zero_rows(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisim_core.sdk.models.gemma4 import _gemma4_collection_images

    version = "0.30.0"
    runtime = dict(
        framework="vllm",
        version=version,
        image="vllm/vllm-openai:v0.30.0",
        image_digest="sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90",
    )
    custom = runtime | {"image": "custom/vllm"}
    event = dict(
        collector_ref="rev", collector_hash="h", case_plan_hash="p", collected_at="2026-01-01", status="partial"
    )
    name = "context_attention_perf"
    # FPM publishes separate metadata for its backend-first timing table.
    pq.write_table(pa.table({"latency": [1]}), tmp_path / "fpm_forward_perf.parquet")
    for stock_rows, foreign_rows, merged_rows, retained_rows, expected in (
        (1, 0, 1, 1, [True]),
        (2, 0, 1, 1, [True]),
        (0, 1, 1, 1, [False]),
        (1, 0, 2, 1, [False]),
        (1, 0, 2, 2, [False]),
        (0, 0, 1, 1, [False]),
    ):
        pq.write_table(pa.table({"latency": range(retained_rows)}), tmp_path / f"{name}.parquet")
        events = [
            event | {"rows": stock_rows, "runtime": runtime, "classified_failures": 1},
            event | {"rows": foreign_rows},
        ]
        tables = {name: {"rows": merged_rows, "status": "partial", "collections": events}}
        (tmp_path / "collection_meta.yaml").write_text(
            json.dumps({"schema_version": 2, "runtime": custom, "tables": tables})
        )
        assert list(_gemma4_collection_images(tmp_path, version, "vllm", cache=False)) == expected


@pytest.mark.parametrize(
    "version,manifest,previous",
    [
        (
            "0.31.0",
            "sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b",
            "sha256:3f7dd5b777d34d1724456ce71f87385dca288c3bb23029ab27dee358f5d2b971",
        ),
        (
            "0.32.0",
            "sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9",
            "sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b",
        ),
    ],
)
def test_gemma4_frozen_sparse_measurements_bind_every_retained_row_to_the_runtime(
    tmp_path, version, manifest, previous, monkeypatch
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisim_core.sdk.models.gemma4 import _gemma4_collection_images

    runtime = dict(
        framework="vllm", version=version, image="vllm/vllm-openai", image_arch="amd64", image_digest=manifest
    )
    name = "glm53_attention_module_perf"
    event = dict(
        status="complete", rows=2, data_sha256="e" * 64, collector="collector.vllm.glm53flash_attention_runner"
    )
    (tmp_path / "collection_meta.yaml").write_text(
        json.dumps({"schema_version": 1, "runtime": runtime, "tables": {name: event}})
    )
    reads = []
    read = pq.ParquetFile.read

    def record_read(file, *args, **kwargs):
        reads.append(kwargs.get("columns"))
        return read(file, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read", record_read)
    for data, qualifies in (
        ({"runtime_digest": [manifest, manifest]}, True),
        ({"runtime_digest": [previous, previous]}, False),
        ({"runtime_digest": [manifest, previous]}, False),
        ({"runtime_digest": [manifest, None]}, False),
        ({"runtime_digest": [manifest]}, False),
        ({"latency": [1, 2]}, False),
    ):
        pq.write_table(pa.table(data), tmp_path / f"{name}.parquet")
        assert list(_gemma4_collection_images(tmp_path, version, "vllm", cache=False)) == [qualifies]
        scans = len(reads)
        assert list(_gemma4_collection_images(tmp_path, version, "vllm", cache=True)) == [qualifies]
        assert len(reads) == scans


def test_gemma4_adapter_compares_each_ranks_cache_without_deployment_evidence(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    doc = _load_pre_downloaded_hf_config("google/gemma-4-26B-A4B")
    text = doc["text_config"]
    heads = text.pop("num_global_key_value_heads")
    dim = text.pop("global_head_dim")
    text["per_layer_config"] = {
        str(i): {"head_dim": dim, "num_key_value_heads": heads}
        for i, kind in enumerate(text["layer_types"])
        if kind == "full_attention"
    }
    for allow in (False, True):
        text["allow_global_per_layer_attribute_access"] = allow
        path = _write_config(tmp_path, doc)
        for tp in (1, 4, 8):
            args = dict(tp_size=tp, pp_size=1, backend="vllm", backend_version="0.28.0")
            if allow and tp == 8:
                assert NaiveKVCacheEstimator.from_hf_config(doc, **args).kv_bytes_per_token() == 35_840
                assert estimate_kv_bytes_per_token(path, **args) == 35_840
            else:
                with pytest.raises(ValueError, match="runtime profile"):
                    NaiveKVCacheEstimator.from_hf_config(doc, **args)
                with pytest.raises(ValueError, match="runtime profile"):
                    estimate_kv_bytes_per_token(path, **args)


def test_gemma4_reader_agreement_reaches_native_capacity_and_rank_local_models(tmp_path, monkeypatch):
    from aisim_core.sdk.config import ModelConfig
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    doc = _load_pre_downloaded_hf_config("google/gemma-4-26B-A4B")
    text = doc["text_config"]
    text.update(num_global_key_value_heads=8, allow_global_per_layer_attribute_access=True)
    text["per_layer_config"] = {
        str(i): {"head_dim": 512, "num_key_value_heads": 2}
        for i, kind in enumerate(text["layer_types"])
        if kind == "full_attention"
    }
    text["per_layer_config"]["0"] = {"num_key_value_heads": 4}
    path = _write_config(tmp_path, doc)
    config = ModelConfig(tp_size=8, moe_tp_size=8, moe_ep_size=1)
    assert get_model(path, config, "vllm").get_kvcache_bytes_per_sequence(1) == 35_840
    for release in (None, "0.28.0", "0.32.0"):
        estimator = NaiveKVCacheEstimator.from_hf_config(doc, tp_size=8, pp_size=1, backend_version=release)
        assert estimator.kv_bytes_per_token() == 35_840
    with pytest.raises(ValueError, match="one Gemma 4 cache geometry"):
        NaiveKVCacheEstimator.from_hf_config(doc, tp_size=8, pp_size=1, backend="sglang", backend_version="0.5.23")
    with pytest.raises(ValueError, match="one Gemma 4 cache geometry"):
        NaiveKVCacheEstimator.from_hf_config(doc, tp_size=1, pp_size=1, backend_version="0.32.0")

    del text["per_layer_config"]["0"]
    model_dir = tmp_path / "native"
    model_dir.mkdir()
    path = _write_config(model_dir, doc)
    monkeypatch.setattr(
        memory.perf_database,
        "get_database",
        lambda *a, **kw: SimpleNamespace(
            version="0.30.0",
            backend="vllm",
            system="custom",
            systems_root=str(tmp_path),
            strict_provenance=False,
            system_spec={},
        ),
    )
    with pytest.raises(ValueError, match="stock collection or a runtime profile"):
        memory.KVCacheEstimator.from_request(
            path, "custom", "vllm", "0.30.0", systems_path=str(tmp_path), max_num_tokens=2560, max_batch_size=12
        )
    monkeypatch.setattr(
        NaiveKVCacheEstimator, "from_model_path", lambda *a, **kw: pytest.fail("selected runtime was already refused")
    )
    opts = _missing_system_cache_options(tmp_path, tp_size=1, capacity=80 * _GIB)
    with pytest.raises(ValueError, match="stock collection or a runtime profile"):
        memory.estimate_num_gpu_blocks(path, "custom", "vllm", "0.30.0", **opts)


def test_gemma4_adapter_uses_only_global_fields_read_by_its_serving_release():
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    doc = _load_pre_downloaded_hf_config("google/gemma-4-26B-A4B")
    text = doc["text_config"]
    text.update(
        global_head_dim=256, num_global_key_value_heads=8, expert_intermediate_size=704, moe_intermediate_size=704
    )
    text["per_layer_config"] = {"0": {"expert_intermediate_size": 768}}
    for release in ("0.31.0", "0.32.0"):
        assert (
            NaiveKVCacheEstimator.from_hf_config(
                doc, tp_size=1, pp_size=1, backend_version=release
            ).kv_bytes_per_token()
            == 245_760
        )
    with pytest.raises(ValueError, match="expert_intermediate_size"):
        NaiveKVCacheEstimator.from_hf_config(doc, tp_size=1, pp_size=1, backend_version="0.30.0")


def test_gemma4_implicit_and_explicit_layer_settings_use_the_same_chunk_and_window(tmp_path):
    from aisim_core.sdk.config import ModelConfig
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    doc = _load_pre_downloaded_hf_config("google/gemma-4-26B-A4B")
    text = doc["text_config"]
    text.update(attention_chunk_size=512, sliding_window=1024, allow_global_per_layer_attribute_access=True)
    rows = text["per_layer_config"] = {
        str(i): {"sliding_window": None, "head_dim": 512, "num_key_value_heads": 2}
        for i, kind in enumerate(text["layer_types"])
        if kind == "full_attention"
    }
    for explicit in (False, True):
        if explicit:
            rows["0"] = {}
        folder = tmp_path / str(explicit)
        folder.mkdir()
        config = ModelConfig(tp_size=1, moe_ep_size=1, moe_tp_size=1)
        assert (
            get_model(_write_config(folder, doc), config, "vllm", "0.32.0").get_kvcache_bytes_per_sequence(1) == 225_280
        )
        with pytest.raises(ValueError, match="cannot combine attention_chunk_size and sliding_window"):
            NaiveKVCacheEstimator.from_hf_config(doc, tp_size=1, pp_size=1, backend_version="0.32.0")
    for i, kind in enumerate(text["layer_types"]):
        if kind == "sliding_attention":
            rows[str(i)] = {"attention_chunk_size": None}
    assert (
        NaiveKVCacheEstimator.from_hf_config(doc, tp_size=1, pp_size=1, backend_version="0.32.0").kv_bytes_per_token()
        == 225_280
    )


def test_gemma4_agreeing_adapter_readers_use_the_saved_geometry_without_a_catalog(tmp_path, monkeypatch):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.config import ModelConfig
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    document = _load_pre_downloaded_hf_config("google/gemma-4-26B-A4B")
    text = document["text_config"]
    full_dim = text.pop("global_head_dim")
    del text["num_global_key_value_heads"]
    text["per_layer_config"] = {
        str(i): {"head_dim": full_dim, "num_key_value_heads": text["num_key_value_heads"]}
        for i, kind in enumerate(text["layer_types"])
        if kind == "full_attention"
    }
    path = _write_config(tmp_path, document)
    monkeypatch.setattr(memory.perf_database, "_iter_backend_version_dirs", lambda *args: pytest.fail("catalog lookup"))
    config = ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1)
    assert get_model(path, config, "vllm").get_kvcache_bytes_per_sequence(1) == 286_720
    for version in ("0.28.0", "0.29.0", "0.30.0", "0.31.0"):
        assert (
            estimate_kv_bytes_per_token(path, tp_size=1, pp_size=1, system="gb300", backend_version=version) == 286_720
        )


def test_selected_decoder_uses_only_the_fields_needed_by_the_estimate(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim.generator.naive import _estimate_model_weight_bytes
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    text = _load_pre_downloaded_hf_config("Qwen/Qwen3-0.6B")
    del text["max_position_embeddings"]
    wrapper = {"architectures": ["LlavaForConditionalGeneration"], "model_type": "llava", "text_config": text}
    path = _write_config(tmp_path, wrapper)
    assert estimate_kv_bytes_per_token(path, tp_size=1, pp_size=1) == 114_688
    # Qwen3 projects 16 query heads and eight KV heads of width 128 from a 1,024-wide hidden state.
    attention = 2 * 1024 * 128 * (16 + 8)
    text_weights = 2 * (2 * 151936 * 1024 + 28 * (attention + 3 * 1024 * 3072))
    # The default CLIP tower adds 23 resident layers on vLLM, plus the patch embeddings and connector.
    assert _estimate_model_weight_bytes(path) >= text_weights + 23 * 2 * (4 * 1024**2 + 2 * 1024 * 4096)

    del text["vocab_size"]
    other = tmp_path / "without-vocab"
    other.mkdir()
    path = _write_config(other, wrapper)
    assert estimate_kv_bytes_per_token(path, tp_size=1, pp_size=1) == 114_688
    with pytest.raises(RuntimeError, match="insufficient raw model metadata"):
        _estimate_model_weight_bytes(path)
    with pytest.raises(ValueError, match="config parser refuses"):
        NaiveKVCacheEstimator.from_hf_config(text, tp_size=1, pp_size=1)


def test_registered_gemma4_with_release_independent_heads_skips_the_catalog(monkeypatch):
    from aisim.capacity import estimate_kv_bytes_per_token

    monkeypatch.delenv("AICONFIGURATOR_SYSTEMS_PATH", raising=False)
    monkeypatch.setattr(memory.perf_database, "get_systems_paths", lambda: pytest.fail("unexpected catalog lookup"))
    for version in (None, "current"):
        assert (
            estimate_kv_bytes_per_token(
                "google/gemma-4-26B-A4B",
                tp_size=1,
                pp_size=1,
                system="h200_sxm",
                backend="sglang",
                backend_version=version,
            )
            == 225_280
        )


@pytest.mark.parametrize("name", ["deepseek-ai/DeepSeek-V3", "moonshotai/Kimi-K2.5"])
def test_selected_deepseek_mla_uses_the_constructor_dimensions(tmp_path, name):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config(name)
    text = raw.get("text_config", raw)
    for key in ("kv_lora_rank", "qk_rope_head_dim", "v_head_dim", "qk_nope_head_dim"):
        del text[key]
    model = _write_config(tmp_path, {"text_config": text})
    opts = _missing_system_cache_options(tmp_path, tp_size=64, capacity=80 * _GIB)
    opts.update(moe_ep_size=64, moe_tp_size=1)
    width = 61 * (512 + 64) * 2
    assert estimate_kv_bytes_per_token(model, tp_size=64, pp_size=1, backend_version="0.33.0") == width
    blocks = memory.estimate_num_gpu_blocks(model, "missing", "vllm", "0.33.0", **opts)
    assert 0 < blocks * 16 * width <= 80 * _GIB
    text["kv_lora_rank"] = None
    with pytest.raises(ValueError, match="kv_lora_rank"):
        NaiveKVCacheEstimator.from_hf_config({"text_config": text}, tp_size=1, pp_size=1)


@pytest.mark.parametrize("named", [True, False])
def test_selected_qwen_dense_weights_use_the_constructor_intermediate_size(named):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("Qwen/Qwen3-0.6B")
    if not named:
        del raw["architectures"]
    del raw["intermediate_size"]
    est = NaiveKVCacheEstimator.from_hf_config({"text_config": raw}, tp_size=1, pp_size=1)
    raw["intermediate_size"] = 22_016
    explicit = NaiveKVCacheEstimator.from_hf_config({"text_config": raw}, tp_size=1, pp_size=1)
    assert est.geometry["inter"] == 22_016
    assert est.weight_bytes() == explicit.weight_bytes()
    raw["intermediate_size"] = None
    with pytest.raises(ValueError, match="intermediate_size"):
        NaiveKVCacheEstimator.from_hf_config({"text_config": raw}, tp_size=1, pp_size=1)


def test_selected_qwen2_uses_its_default_ffn_to_size_an_eight_gib_cache(tmp_path):
    text = dict(
        model_type="qwen2",
        hidden_size=1536,
        num_attention_heads=12,
        num_key_value_heads=2,
        num_hidden_layers=28,
        vocab_size=151936,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
    )
    opts = _missing_system_cache_options(tmp_path, tp_size=1, capacity=8 * _GIB)
    opts.update(memory_fraction_value=1, max_num_tokens=2560, max_batch_size=12)
    del opts["naive_kv_reservation"]
    for dimensions in ({}, {"intermediate_size": 22_016}):
        wrapper = {"architectures": ["LlavaForConditionalGeneration"], "text_config": text | dimensions}
        model = _write_config(tmp_path, wrapper)
        assert memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.23", **opts) == 1448


@pytest.mark.parametrize(
    "wrapper_class,model_type",
    [("LlavaOnevisionForConditionalGeneration", "llava_onevision"), ("InternVLForConditionalGeneration", "internvl")],
)
def test_multimodal_default_qwen2_constructor_sizes_weights(tmp_path, wrapper_class, model_type):
    text = dict(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
        num_hidden_layers=24,
        vocab_size=152000,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
    )
    wrapper = {
        "architectures": [wrapper_class],
        "model_type": model_type,
        "torch_dtype": "float16",
        "text_config": text,
    }
    opts = _missing_system_cache_options(tmp_path, tp_size=1, capacity=4 * _GIB)
    opts.update(memory_fraction_value=1, max_num_tokens=2560, max_batch_size=12)
    del opts["naive_kv_reservation"]
    for fields in ({}, {"model_type": "qwen2"}, {"model_type": "qwen2", "intermediate_size": 22_016}):
        path = _write_config(tmp_path, wrapper | {"text_config": text | fields})
        # Both wrappers retain the vision tower while their bare decoder below reserves only language weights.
        assert memory.estimate_num_gpu_blocks(path, "missing", "vllm", "0.32.0", **opts) == (
            46 if model_type == "llava_onevision" else 789
        )
    path = _write_config(tmp_path, text | {"model_type": "qwen2"})
    assert memory.estimate_num_gpu_blocks(path, "missing", "vllm", "0.32.0", **opts) == 3073
    path = _write_config(tmp_path, wrapper | {"text_config": text | {"model_type": None}})
    with pytest.raises(ValueError, match="model_type must be a string"):
        NaiveKVCacheEstimator.from_model_path(path, tp_size=1, pp_size=1, allow_hf_config_download=False)


@pytest.mark.parametrize("wrapped", [False, True])
def test_qwen2_enabled_sliding_window_uses_its_constructor_default(wrapped):
    text = dict(
        model_type="qwen2",
        architectures=["Qwen2ForCausalLM"],
        hidden_size=1536,
        num_hidden_layers=28,
        num_attention_heads=12,
        num_key_value_heads=2,
        vocab_size=151936,
        intermediate_size=8960,
        max_position_embeddings=32768,
        use_sliding_window=True,
        max_window_layers=0,
    )
    doc = {"architectures": ["LlavaForConditionalGeneration"], "text_config": text} if wrapped else text
    for window in ({}, {"sliding_window": 4096}):
        text.update(window)
        estimator = NaiveKVCacheEstimator.from_hf_config(doc, tp_size=1, pp_size=1, backend_version="0.32.0")
        assert estimator.kv_bytes_per_token() == 28_672
        assert estimator.geometry["num_swa_layers"] == 28
        assert estimator.geometry["sliding_window"] == 4096
    text["max_window_layers"] = 14
    for fields in ({}, {"layer_types": ["full_attention"] * 14 + ["sliding_attention"] * 14}):
        text.update(fields)
        with pytest.raises(ValueError, match="vLLM Qwen2 does not support"):
            NaiveKVCacheEstimator.from_hf_config(doc, tp_size=1, pp_size=1, backend_version="0.32.0")
        estimator = NaiveKVCacheEstimator.from_hf_config(
            doc, tp_size=1, pp_size=1, backend="sglang", backend_version="0.5.20"
        )
        assert estimator.geometry["num_swa_layers"] is None and estimator.kv_bytes_per_token() == 28_672
    text["sliding_window"] = 0
    with pytest.raises(ValueError, match="positive sliding window"):
        NaiveKVCacheEstimator.from_hf_config(doc, tp_size=1, pp_size=1, backend_version="0.32.0")


def test_mixed_attention_follows_the_serving_model_selected_by_the_wrapper():
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    text = _load_pre_downloaded_hf_config("Qwen/Qwen3-0.6B")
    text.update(use_sliding_window=True, max_window_layers=14)
    text.pop("layer_types", None)
    text.pop("sliding_window", None)
    wrapped = {"architectures": ["LlavaForConditionalGeneration"], "text_config": text}
    with pytest.raises(ValueError, match="vLLM Qwen3 does not support"):
        NaiveKVCacheEstimator.from_hf_config(wrapped, tp_size=1, pp_size=1, backend_version="0.32.0")
    sglang = NaiveKVCacheEstimator.from_hf_config(
        wrapped, tp_size=1, pp_size=1, backend="sglang", backend_version="0.5.20"
    )
    assert sglang.geometry["num_swa_layers"] is None
    assert sglang.kv_bytes_per_token() == 114_688

    # The converse uses Qwen2Model on a LlamaConfig. Its extra window flag does
    # not generate Qwen2Config's mixed layer list.
    llama = {**text, "model_type": "llama", "architectures": ["Qwen2LMHeadModel", "LlamaForCausalLM"]}
    dispatched = {**wrapped, "text_config": llama}
    assert (
        NaiveKVCacheEstimator.from_hf_config(
            dispatched, tp_size=1, pp_size=1, backend_version="0.32.0"
        ).kv_bytes_per_token()
        == 57_344
    )

    # Idefics3 uses LlamaModel with Qwen2Config, so Qwen2's mixed-layer rejection does not apply.
    text.update(model_type="qwen2", architectures=["Qwen2ForCausalLM"])
    fixed = {"architectures": ["Idefics3ForConditionalGeneration"], "text_config": text}
    assert (
        NaiveKVCacheEstimator.from_hf_config(fixed, tp_size=1, pp_size=1, backend_version="0.32.0").kv_bytes_per_token()
        == 114_688
    )


def test_registered_qwen2_sliding_defaults_follow_the_sdk_parser(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token

    text = dict(
        architectures=["Qwen2ForCausalLM"],
        model_type="qwen2",
        hidden_size=1536,
        num_hidden_layers=28,
        num_attention_heads=12,
        num_key_value_heads=2,
        vocab_size=151936,
        intermediate_size=8960,
        max_position_embeddings=32768,
        use_sliding_window=True,
    )
    layer_error = "must define 'num_hidden_layers' or a parseable layer pattern"
    for missing, overrides, reason in (
        ([], {"num_hidden_layers": None}, layer_error),
        (["num_hidden_layers"], {}, layer_error),
        (["vocab_size"], {"max_window_layers": "invalid"}, "KeyError: 'vocab_size'"),
    ):
        raw = text | overrides
        for field in missing:
            del raw[field]
        path = _write_config(tmp_path, raw)
        with pytest.raises(ValueError, match=reason):
            NaiveKVCacheEstimator.from_hf_config(raw, tp_size=1, pp_size=1)
        with pytest.raises(ValueError, match=reason):
            estimate_kv_bytes_per_token(path, tp_size=1, pp_size=1)
    text["max_window_layers"] = "invalid"
    with pytest.raises(ValueError, match="max_window_layers must be an integer"):
        estimate_kv_bytes_per_token(_write_config(tmp_path, text), tp_size=1, pp_size=1)


@pytest.mark.parametrize("wrapper", ["Idefics3ForConditionalGeneration", "SmolVLMForConditionalGeneration"])
def test_fixed_llama_wrapper_uses_its_decoder_independently_of_the_training_class(tmp_path, wrapper):
    from aisim.capacity import estimate_kv_bytes_per_token

    text = {
        "model_type": "llama",
        "architectures": ["VLlama3ForCausalLM"],
        "num_hidden_layers": 30,
        "hidden_size": 576,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "intermediate_size": 1536,
        "vocab_size": 49152,
    }
    for arch in ("VLlama3ForCausalLM", "DeepseekV3ForCausalLM"):
        model = _write_config(tmp_path, {"architectures": [wrapper], "text_config": text | {"architectures": [arch]}})
        assert estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1) == 23_040
    model = _write_config(tmp_path, {"architectures": [wrapper], "text_config": text | {"model_type": "mistral"}})
    assert estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1, backend_version="0.32.0") == 23_040
    bare = estimate_kv_bytes_per_token(_write_config(tmp_path, text), tp_size=1, pp_size=1)
    for decoder_key in ("text_config", "llm_config"):
        model = _write_config(tmp_path, {"architectures": ["UnknownConditionalGeneration"], decoder_key: text})
        assert estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1) == bare == 23_040
    training = {k: v for k, v in text.items() if k != "num_hidden_layers"}
    assert estimate_kv_bytes_per_token(_write_config(tmp_path, training), tp_size=1, pp_size=1) == 24_576
    text["architectures"] = ["DeepseekV3ForCausalLM"]
    normalized = wrapper.replace("ForConditionalGeneration", "LMHeadModel")
    for names in (["OldCheckpoint", wrapper], ["OldCheckpoint", normalized], ["AlbertModel", wrapper]):
        model = _write_config(tmp_path, {"architectures": names, "text_config": text})
        assert estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1, backend_version="0.32.0") == 23_040
    for names, reason in (
        (["LlavaNextForConditionalGeneration", wrapper], "conflicts with architectures"),
        (["Phi3ForCausalLM", wrapper], "selected wrapper"),
        (["AlbertModel", normalized], "selected wrapper"),
        (["LlamaForCausalLM", wrapper], "config parser refuses.*LlamaForCausalLM"),
    ):
        model = _write_config(tmp_path, {"architectures": names, "text_config": text})
        with pytest.raises(ValueError, match=reason):
            estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1, backend_version="0.32.0")
    model = _write_config(tmp_path, {"architectures": ["LlavaNextForConditionalGeneration"], "text_config": text})
    with pytest.raises(ValueError, match="conflicts with architectures"):
        estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1)
    model = _write_config(tmp_path, {"architectures": ["LlavaForConditionalGeneration"], "text_config": text})
    assert (
        estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1, backend="sglang", backend_version="0.5.20") == 23_040
    )


def test_config_class_defaults_and_explicit_vllm_decoder_dispatch():
    def width(document):
        return NaiveKVCacheEstimator.from_hf_config(
            document, tp_size=1, pp_size=1, backend_version="0.32.0"
        ).kv_bytes_per_token()

    qwen = {"model_type": "qwen2", "num_attention_heads": 64, "hidden_size": 8192}
    assert width(qwen) == width({"text_config": qwen}) == 524_288
    for key in ("text_config", "llm_config"):
        assert width({key: qwen | {"architectures": False}}) == 524_288
    onevision = {"architectures": ["LlavaOnevisionForConditionalGeneration"]}
    assert width(onevision | {"text_config": {"vocab_size": 152064}}) == 524_288
    for value in ([], {}, False):
        with pytest.raises(ValueError, match="architectures"):
            width(onevision | {"text_config": {"vocab_size": 152064, "architectures": value}})

    parent = {"architectures": ["LlavaNextForConditionalGeneration"]}
    text = dict(
        architectures=["MistralForCausalLM"],
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=32,
        hidden_size=4096,
        intermediate_size=11008,
        max_position_embeddings=4096,
        vocab_size=32064,
    )
    assert width(parent | {"text_config": text}) == 131_072
    text.update(
        model_type="llama",
        architectures=["OPTForCausalLM"],
        do_layer_norm_before=True,
        layer_norm_elementwise_affine=True,
        ffn_dim=11008,
        enable_bias=True,
        activation_function="relu",
        word_embed_proj_dim=4096,
        _remove_final_layer_norm=False,
    )
    assert width(parent | {"text_config": text}) == 524_288

    def weights(positions):
        model = parent | {"text_config": text | {"max_position_embeddings": positions}}
        return NaiveKVCacheEstimator.from_hf_config(
            model, tp_size=2, pp_size=1, backend_version="0.32.0"
        ).weight_bytes()

    assert weights(8192) - weights(4096) == 4096 * 4096 * 2
    del text["ffn_dim"]
    with pytest.raises(ValueError, match="OPT decoder"):
        width(parent | {"text_config": text})


@pytest.mark.parametrize(
    "classes,expected",
    [
        (["VLlama3ForCausalLM", "LlamaForCausalLM"], 81_920),
        (["VLlama3ForCausalLM", "LlamaLMHeadModel"], 81_920),
        (["VLlama3ForCausalLM", "Qwen2LMHeadModel"], 163_840),
        (["LlamaForCausalLM", "Qwen2LMHeadModel"], 81_920),
        (["Qwen2LMHeadModel", "LlamaForCausalLM"], 163_840),
        (["VLlama3ForCausalLM", "OPTForCausalLM", "LlamaForCausalLM"], 819_200),
    ],
)
def test_vllm_dispatches_the_first_available_text_decoder(tmp_path, classes, expected):
    from aisim.capacity import estimate_kv_bytes_per_token

    # Each of 40 layers stores K and V in FP16. Llama uses 8 heads of width 64,
    # Qwen2 uses 8 of width 5120 / 40, and OPT uses all 40 of width 5120 / 40.
    text = dict(
        model_type="llama",
        architectures=classes,
        hidden_size=5120,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=64,
        num_hidden_layers=40,
        intermediate_size=13824,
        max_position_embeddings=4096,
        vocab_size=32064,
        ffn_dim=13824,
        word_embed_proj_dim=5120,
        enable_bias=True,
        activation_function="relu",
        layer_norm_elementwise_affine=True,
        do_layer_norm_before=False,
    )
    path = _write_config(tmp_path, {"architectures": ["LlavaNextForConditionalGeneration"], "text_config": text})
    assert estimate_kv_bytes_per_token(path, tp_size=1, pp_size=1, backend_version="0.32.0") == expected
    assert _compiled_vllm_transfer_widths(path, "0.32.0") == (expected, expected)


def _compiled_vllm_transfer_widths(path, version, *, kvcache_quant_mode=None):
    from aisim.compiler import prediction_to_replay_spec
    from aisim.config import CorePredictionConfig

    worker = dict(
        parallelism={"tensor": 1},
        timing={"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
        kv_cache={"capacity": {"type": "fixed", "blocks": 256}},
    )
    if kvcache_quant_mode is not None:
        worker["timing"] = {"type": "default"}
    engine = dict(
        model=path,
        mode="disaggregated",
        hardware="gb300",
        backend="vllm",
        backend_version=version,
        kvcache_quant_mode=kvcache_quant_mode,
        context_length=4096,
        workers={"prefill": worker, "decode": worker},
        kv_transfer={"bytes_per_token": "auto", "bandwidth_gb_per_second": 50},
    )
    deployment = prediction_to_replay_spec(CorePredictionConfig.model_validate({"engine": engine})).backend_deployment
    return deployment.prefill_engine_args["kv_transfer_bytes_per_token"], deployment.decode_engine_args[
        "kv_transfer_bytes_per_token"
    ]


def _onevision_qwen2_7b(child_dtype):
    return {
        "architectures": ["LlavaOnevisionForConditionalGeneration"],
        "model_type": "llava_onevision",
        "torch_dtype": "float16",
        "text_config": {
            "model_type": "qwen2",
            "torch_dtype": child_dtype,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "vocab_size": 152064,
        },
    }


def _synthetic_multimodal_w8():
    return {
        "architectures": ["LlavaOnevisionForConditionalGeneration"],
        "model_type": "llava_onevision",
        "torch_dtype": "float16",
        "text_config": dict(
            model_type="qwen2",
            hidden_size=768,
            intermediate_size=1536,
            num_hidden_layers=3,
            num_attention_heads=12,
            num_key_value_heads=2,
            vocab_size=2048,
        ),
        "vision_config": dict(
            model_type="siglip_vision_model",
            hidden_size=64,
            intermediate_size=192,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=64,
            patch_size=8,
        ),
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "config_groups": {
                "projection": {
                    "targets": ["Linear"],
                    "weights": {"type": "float", "num_bits": 8, "strategy": "channel"},
                    "input_activations": {"type": "float", "num_bits": 8, "strategy": "token", "dynamic": True},
                }
            },
            "ignore": ["vision_tower.output", "multi_modal_projector.linear_2", "language_model.lm_head"],
        },
    }


def test_vllm_onevision_fp8_decoder_projection_coverage(tmp_path):
    document = _synthetic_multimodal_w8()
    quant = document["quantization_config"]
    args = _missing_system_cache_options(tmp_path, tp_size=1, capacity=32 * 2**20)
    args.update(
        max_num_tokens=512, max_batch_size=1, memory_fraction_value=1, naive_kv_reservation=0.8, tolerance_fraction=0
    )
    model = _write_config(tmp_path, document)
    assert _compiled_vllm_transfer_widths(model, "0.32.0") == (1536, 1536)
    assert memory.estimate_num_gpu_blocks(model, "missing", "vllm", "0.32.0", **args) == 352
    for target in ("language_model.model.layers.2.self_attn.o_proj", r"re:.*\.mlp\.down_proj$"):
        quant["ignore"].append(target)
        model = _write_config(tmp_path, document)
        with pytest.raises(ValueError, match="no KV budget"):
            memory.estimate_num_gpu_blocks(model, "missing", "vllm", "0.32.0", **args)
        quant["ignore"].pop()
    quant["kv_cache_scheme"] = {"num_bits": 8, "type": "float", "strategy": "tensor", "symmetric": True}
    document["compression_config"] = document.pop("quantization_config")
    document["text_config"]["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "config_groups": {},
        "format": "dense",
    }
    model = _write_config(tmp_path, document)
    assert _compiled_vllm_transfer_widths(model, "0.32.0") == (1536, 1536)
    with pytest.raises(ValueError, match="no KV budget"):
        memory.estimate_num_gpu_blocks(model, "missing", "vllm", "0.32.0", **args)


def test_vllm_onevision_w8_group_scales_and_activation_scheme():
    document = _synthetic_multimodal_w8()
    quant = document["quantization_config"]
    group = quant["config_groups"]["projection"]
    options = dict(backend="vllm", backend_version="0.32.0", tp_size=1, pp_size=1)
    quant["format"] = "pack-quantized"
    group["weights"].update(type="int", strategy="group", group_size=128)
    group["input_activations"] = None
    matrix_weights = 3 * (2 * 768 * 64 * (12 + 2) + 3 * 768 * 1536)
    assert NaiveKVCacheEstimator.from_hf_config(document, **options).weight_bytes() == 22_708_352 + 2 * (
        matrix_weights // 128
    )
    group["weights"]["group_size"] = 16
    assert NaiveKVCacheEstimator.from_hf_config(document, **options).weight_bytes() == 37_453_952
    quant["format"] = "int-quantized"
    group["weights"].update(strategy="channel", group_size=None)
    group["input_activations"] = {"num_bits": 8, "type": "int", "strategy": "token", "dynamic": True}
    assert NaiveKVCacheEstimator.from_hf_config(document, **options).weight_bytes() == 22_708_352
    group["input_activations"]["strategy"] = "tensor"
    with pytest.raises(ValueError, match="W8 weights"):
        NaiveKVCacheEstimator.from_hf_config(document, **options)
    group["input_activations"] = {"type": "int"}
    assert NaiveKVCacheEstimator.from_hf_config(document, **options).weight_bytes() == 22_708_352


@pytest.mark.parametrize(
    ("version", "group_activations", "group_packed_weights"),
    [
        ("0.10.0", False, False),
        ("0.10.1rc1", True, False),
        ("0.10.1.1", True, False),
        ("0.10.2rc1", True, True),
        ("0.11.0", True, True),
    ],
)
def test_vllm_per_group_weight_formats(tmp_path, version, group_activations, group_packed_weights):
    from aisim.generator.naive import _estimate_model_weight_bytes

    document = _synthetic_multimodal_w8()
    quant = document["quantization_config"]
    group = quant["config_groups"]["projection"]
    for supported_format, per_group in (
        ("float-quantized", group_activations),
        ("pack-quantized", group_packed_weights),
    ):
        if supported_format == "pack-quantized":
            group["weights"]["type"] = "int"
            group["input_activations"] = None
        for root, override, supported in (
            (supported_format, "dense", not per_group),
            ("dense", supported_format, per_group),
        ):
            quant["format"] = root
            group["format"] = override
            checkpoint = tmp_path / f"{root}-{override}"
            checkpoint.mkdir()
            model = _write_config(checkpoint, document)
            if supported:
                assert _estimate_model_weight_bytes(model, backend="vllm", backend_version=version) == 22_708_352
            else:
                with pytest.raises(RuntimeError, match="W8 weights"):
                    _estimate_model_weight_bytes(model, backend="vllm", backend_version=version)


def test_auxiliary_weights_are_separate_from_marginal_decoder_kv(tmp_path):
    from aisim.generator.naive import _estimate_model_weight_bytes

    source = _synthetic_multimodal_w8()
    text = {**source["text_config"], "quantization_config": source["quantization_config"]}
    unknown = {**source, "architectures": ["UnknownVisionWrapper"], "model_type": "unknown"}
    neither = {k: v for k, v in unknown.items() if k != "vision_config"}
    opts = dict(tp_size=1, pp_size=1, backend_version="0.32.0")
    for document in (source, text, unknown, neither):
        assert NaiveKVCacheEstimator.from_hf_config(document, **opts).kv_bytes_per_token() == 1536
    assert NaiveKVCacheEstimator.from_hf_config(text, **opts).weight_bytes() > 0
    assert NaiveKVCacheEstimator.from_hf_config(neither, **opts).weight_bytes() > 0
    with pytest.raises(RuntimeError, match="resident weights for the vision tower and projector"):
        _estimate_model_weight_bytes(_write_config(tmp_path, unknown), backend_version="0.32.0")


def test_vllm_implicit_and_partial_vision_towers():
    text = _synthetic_multimodal_w8()["text_config"]
    opts = dict(tp_size=1, pp_size=1, backend_version="0.32.0")
    qwen = {"model_type": "qwen2_vl", "architectures": ["Qwen2VLForConditionalGeneration"], "text_config": text}
    for raw in (
        qwen,
        {**qwen, "vision_config": {}},
        {**qwen, "architectures": ["UnknownNativeCheckpoint", *qwen["architectures"]]},
    ):
        estimator = NaiveKVCacheEstimator.from_hf_config(raw, **opts)
        assert estimator.kv_bytes_per_token() == 1536
        with pytest.raises(ValueError, match="resident weights for the vision tower"):
            estimator.weight_bytes()

    def resident(wrapper, **extra):
        return NaiveKVCacheEstimator.from_hf_config(
            {"architectures": [wrapper], "text_config": text, **extra}, **opts
        ).weight_bytes()

    clip = {"model_type": "clip_vision_model", "patch_size": 32}
    assert resident("LlavaForConditionalGeneration", vision_config={}) == resident(
        "LlavaForConditionalGeneration", vision_config=clip
    )
    intern = resident("InternVLForConditionalGeneration", model_type="internvl")
    assert intern == resident(
        "InternVLForConditionalGeneration", model_type="internvl", vision_config={"model_type": "internvl_vision"}
    )


def test_vllm_effective_quantizer_class_sidecar_and_modelopt_exclusions(tmp_path):
    from pathlib import Path

    from aisim.generator.naive import _estimate_model_weight_bytes

    document = _synthetic_multimodal_w8()
    document["text_config"]["torch_dtype"] = "float8_e4m3fn"
    settings = document.pop("quantization_config")
    settings.pop("quant_method")
    settings["kv_cache_scheme"] = dict(type="float", num_bits=8, strategy="tensor", symmetric=True)
    document["text_config"]["quantization_config"] = settings
    document["compression_config"] = {"quant_method": "compressed-tensors"}
    model = _write_config(tmp_path, document)
    assert _compiled_vllm_transfer_widths(model, "0.32.0") == (768, 768)
    mixed = {"quant_method": "modelopt", "quant_algo": "FP8", "ignore": ["*mlp*", "*lm_head*"]}
    assert _estimate_model_weight_bytes(model, backend_version="0.32.0") == 22_708_352
    adjacent = Path(model) / "hf_quant_config.json"
    adjacent.write_text(json.dumps(mixed))
    assert _compiled_vllm_transfer_widths(model, "0.32.0") == (1536, 1536)
    weights = _estimate_model_weight_bytes(model, backend_version="0.32.0")
    inline_dir = tmp_path / "inline"
    inline_dir.mkdir()
    inline = _write_config(inline_dir, {**document, "quantization_config": mixed})
    assert _estimate_model_weight_bytes(inline, backend_version="0.32.0") == weights == 37_453_952
    opts = dict(backend_version="0.32.0", tp_size=1, pp_size=1)
    complete = {**mixed, "ignore": []}
    assert (
        NaiveKVCacheEstimator.from_hf_config({**document, "quantization_config": complete}, **opts).weight_bytes()
        < weights
    )
    mx = []
    for method in ("modelopt_mxfp8", "mxfp8"):
        declared = {**document, "quantization_config": {"quant_method": method}}
        with pytest.raises(ValueError, match="MXFP8 requires bfloat16"):
            NaiveKVCacheEstimator.from_hf_config(declared, **opts)
        mx.append(NaiveKVCacheEstimator.from_hf_config({**declared, "torch_dtype": "bfloat16"}, **opts).weight_bytes())
    assert mx[0] == mx[1] < weights
    with pytest.raises(ValueError, match="quant_algo"):
        NaiveKVCacheEstimator.from_hf_config(
            {**document, "quantization_config": {**complete, "quant_algo": "unknown"}}, **opts
        )


def test_compressed_tensors_fused_shards_and_packed_activations():
    document = _synthetic_multimodal_w8()
    quant = document["quantization_config"]
    opts = dict(tp_size=1, pp_size=1, backend_version="0.32.0")
    base = "language_model.model.layers.1.self_attn."
    quant["ignore"].append(base + "q_proj")
    with pytest.raises(ValueError, match="same quantization scheme"):
        NaiveKVCacheEstimator.from_hf_config(document, **opts)
    quant["ignore"].extend([base + "k_proj", base + "v_proj"])
    assert NaiveKVCacheEstimator.from_hf_config(document, **opts).weight_bytes() == 37_453_952
    quant["ignore"] = []
    quant["format"] = "pack-quantized"
    group = quant["config_groups"]["projection"]
    group["weights"] = {"type": "int", "num_bits": 8, "group_size": 128}
    group["input_activations"] = None
    inferred = NaiveKVCacheEstimator.from_hf_config(document, **opts).weight_bytes()
    group["weights"]["strategy"] = "group"
    assert NaiveKVCacheEstimator.from_hf_config(document, **opts).weight_bytes() == inferred
    group["weights"] = dict(type="int", strategy="channel", num_bits=8)
    weight_only = NaiveKVCacheEstimator.from_hf_config(document, **opts).weight_bytes()
    group["input_activations"] = {"type": "int", "num_bits": 8, "strategy": "tensor", "dynamic": False}
    with pytest.raises(ValueError, match="W8 weights"):
        NaiveKVCacheEstimator.from_hf_config(document, **opts)
    group["output_activations"] = group["input_activations"]
    assert NaiveKVCacheEstimator.from_hf_config(document, **opts).weight_bytes() == weight_only
    for strategy in ({}, {"type": "int"}, {"strategy": "group"}, {"type": "INT", "strategy": "GROUP"}):
        group["input_activations"] = {"dynamic": True, "group_size": 128, **strategy}
        assert NaiveKVCacheEstimator.from_hf_config(document, **opts).weight_bytes() == weight_only
        group["input_activations"]["dynamic"] = False
        with pytest.raises(ValueError, match="W8 weights"):
            NaiveKVCacheEstimator.from_hf_config(document, **opts)


def test_vllm_mxfp8_routed_experts_keep_the_execution_width_reserve(tmp_path):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("deepseek-ai/DeepSeek-V3")
    raw.update(architectures=["DeepseekV2ForCausalLM"], quantization_config={"quant_method": "modelopt_mxfp8"})
    opts = _missing_system_cache_options(tmp_path, tp_size=8, capacity=80 * _GIB)
    opts.update(moe_tp_size=8, moe_ep_size=1, naive_kv_reservation=1)
    with pytest.raises(ValueError, match="no KV budget"):
        memory.estimate_num_gpu_blocks(_write_config(tmp_path, raw), "missing", "vllm", "0.32.0", **opts)


def test_modelopt_mxfp8_uses_checkpoint_dtype_and_rank_local_projection_widths(tmp_path):
    doc = _synthetic_multimodal_w8()
    del doc["torch_dtype"]
    quant = doc["quantization_config"] = {"quant_method": "modelopt_mxfp8"}
    model = _write_config(tmp_path, doc)
    opts = dict(tp_size=1, pp_size=1, backend_version="0.32.0")
    for dtype in ("F16", "BF16"):
        header = json.dumps({"embed.weight": {"dtype": dtype, "shape": [1], "data_offsets": [0, 2]}}).encode()
        header += b" " * (-len(header) % 8)
        (tmp_path / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + bytes(2))
        if dtype == "F16":
            with pytest.raises(ValueError, match="MXFP8 requires bfloat16"):
                _compiled_vllm_transfer_widths(model, "0.32.0")
        else:
            assert _compiled_vllm_transfer_widths(model, "0.32.0") == (1536, 1536)
    with pytest.raises(ValueError, match="checkpoint headers"):
        NaiveKVCacheEstimator.from_hf_config(doc, **opts)
    doc["torch_dtype"] = "bfloat16"
    doc["text_config"].update(num_attention_heads=16, num_key_value_heads=8, intermediate_size=29_568)
    assert NaiveKVCacheEstimator.from_hf_config(doc, **(opts | {"tp_size": 4})).weight_bytes() > 0
    opts["tp_size"] = 8
    for ignored in ([], ["*.self_attn.o_proj"], ["language_model.model.layers.0.mlp.down_proj"]):
        quant["ignored_layers"] = ignored
        with pytest.raises(ValueError, match="MXFP8.*32"):
            NaiveKVCacheEstimator.from_hf_config(doc, **opts)
    quant["ignored_layers"] = ["*.mlp.down_proj"]
    assert NaiveKVCacheEstimator.from_hf_config(doc, **opts).weight_bytes() > 0


def test_borrowed_gemma_and_gpt_neo_construct_before_vllm_dispatch(tmp_path):
    opt = dict(
        architectures=["OPTForCausalLM"],
        hidden_size=768,
        num_attention_heads=12,
        ffn_dim=3072,
        enable_bias=True,
        word_embed_proj_dim=768,
        activation_function="relu",
        layer_norm_elementwise_affine=True,
        do_layer_norm_before=False,
    )
    child = {**opt, "model_type": "gemma4_text", "allow_global_per_layer_attribute_access": True}
    wrapper = {"architectures": ["LlavaForConditionalGeneration"], "model_type": "llava", "text_config": child}
    opts = dict(backend_version="0.32.0", tp_size=1, pp_size=1)
    inherited = NaiveKVCacheEstimator.from_hf_config(wrapper, **opts)
    assert _compiled_vllm_transfer_widths(_write_config(tmp_path, wrapper), "0.32.0") == (92_160, 92_160)
    child.update(num_hidden_layers=30, vocab_size=262144, max_position_embeddings=131072)
    assert NaiveKVCacheEstimator.from_hf_config(wrapper, **opts).weight_bytes() == inherited.weight_bytes()
    gemma3 = {**opt, "model_type": "gemma3_text"}
    for raw in (gemma3, {**wrapper, "text_config": gemma3}):
        assert _compiled_vllm_transfer_widths(_write_config(tmp_path, raw), "0.32.0") == (79_872, 79_872)
    with pytest.raises(ValueError, match="OPTForCausalLM"):
        NaiveKVCacheEstimator.from_hf_config({**opt, "model_type": "falcon"}, **opts)

    # The same registered OPT class uses all 12 attention heads when loaded as a bare model.
    bare = {**child, "head_dim": 64, "global_head_dim": 64, "num_global_key_value_heads": 1, "attention_k_eq_v": True}
    path = _write_config(tmp_path, bare)
    assert _compiled_vllm_transfer_widths(path, "0.32.0") == (92_160, 92_160)
    capacity = 4 * _GIB
    args = _missing_system_cache_options(tmp_path, tp_size=1, capacity=capacity)
    args.update(max_num_tokens=512, max_batch_size=1, memory_fraction_value=1.0, tolerance_fraction=0)
    blocks = memory.estimate_num_gpu_blocks(path, "missing", "vllm", "0.32.0", **args)
    assert 0 < blocks <= capacity // (16 * 92_160)

    child.update(
        architectures=["MistralForCausalLM"],
        num_hidden_layers=12,
        num_key_value_heads=12,
        head_dim=64,
        layer_types=["full_attention"] * 12,
        hidden_act="silu",
        intermediate_size=3072,
        num_global_key_value_heads=1,
        global_head_dim=64,
        attention_k_eq_v=True,
    )
    assert _compiled_vllm_transfer_widths(_write_config(tmp_path, wrapper), "0.32.0") == (36_864, 36_864)
    assert _compiled_vllm_transfer_widths(_write_config(tmp_path, child), "0.32.0") == (36_864, 36_864)
    child.update(architectures=["LlamaForCausalLM"], use_bidirectional_attention="all")
    with pytest.raises(ValueError, match="no persistent vLLM KV"):
        _compiled_vllm_transfer_widths(_write_config(tmp_path, wrapper), "0.32.0")

    child = {**opt, "model_type": "gpt_neo", "num_layers": 12, "num_heads": 12}
    wrapper["text_config"] = child
    with pytest.raises(ValueError, match="attention_types"):
        NaiveKVCacheEstimator.from_hf_config(wrapper, **opts)
    child["attention_types"] = [[["global", "local"], 6]]
    assert _compiled_vllm_transfer_widths(_write_config(tmp_path, wrapper), "0.32.0") == (36_864, 36_864)


@pytest.mark.parametrize("child_dtype", ["bfloat16", "float32", "float8_e4m3fn"])
def test_vllm_wrapper_uses_the_execution_dtype_for_auto_kv_and_unquantized_weights(tmp_path, child_dtype):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim.generator.naive import _estimate_model_weight_bytes

    # The OneVision Qwen2 7B checkpoint runs in the parent's FP16 dtype, including its linear weights.
    model = _write_config(tmp_path, _onevision_qwen2_7b(child_dtype))
    weights = 16_074_109_568  # FP16 Qwen2 projections, vocabulary, SigLIP tower, connector and vectors.
    for version in ("0.31.0", "0.32.0", "0.33.0"):
        assert _estimate_model_weight_bytes(model, backend="vllm", backend_version=version) == weights
        for mode, width in ((None, 57_344), ("fp8", 28_672)):
            assert (
                estimate_kv_bytes_per_token(
                    model, tp_size=1, pp_size=1, backend_version=version, kvcache_quant_mode=mode
                )
                == width
            )
            assert _compiled_vllm_transfer_widths(model, version, kvcache_quant_mode=mode) == (width, width)


@pytest.mark.parametrize("quantize_linears", [False, True])
@pytest.mark.parametrize("child_dtype", ["float16", "float8_e4m3fn"])
def test_vllm_compressed_linear_weights_and_kv_have_independent_storage(tmp_path, quantize_linears, child_dtype):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim.generator.naive import _estimate_model_weight_bytes

    # RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic applies W8A8 to decoder Linear layers, excluding lm_head.
    fp8 = {"num_bits": 8, "type": "float", "dynamic": False, "symmetric": True}
    groups = {}
    if quantize_linears:
        groups["group_0"] = {
            "targets": ["Linear"],
            "weights": fp8 | {"strategy": "channel"},
            "input_activations": fp8 | {"strategy": "token", "dynamic": True},
        }
    quant = {
        "quant_method": "compressed-tensors",
        "format": "float-quantized" if groups else "dense",
        "config_groups": groups,
        "ignore": ["lm_head"] if groups else [],
        "kv_cache_scheme": fp8 | {"strategy": "tensor"},
    }
    model = _write_config(tmp_path, _onevision_qwen2_7b(child_dtype) | {"quantization_config": quant})
    args = _missing_system_cache_options(tmp_path, tp_size=1, capacity=12 * _GIB)
    args.update(max_num_tokens=512, max_batch_size=1, memory_fraction_value=1, tolerance_fraction=0)
    for version in ("0.31.0", "0.32.0", "0.33.0"):
        assert estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1, backend_version=version) == 28_672
        assert _compiled_vllm_transfer_widths(model, version) == (28_672, 28_672)
        weights = _estimate_model_weight_bytes(model, backend="vllm", backend_version=version)
        if quantize_linears:
            assert 8_706_862_080 <= weights < 12 * _GIB
            assert memory.estimate_num_gpu_blocks(model, "missing", "vllm", version, **args) > 0
        else:
            assert weights >= 15_232_150_528
            with pytest.raises(ValueError, match="no KV budget"):
                memory.estimate_num_gpu_blocks(model, "missing", "vllm", version, **args)
    # Released KV-only checkpoints can carry an attention group with activation scales and no weights.
    groups["attention"] = {"targets": ["Qwen2Attention"], "weights": None, "input_activations": fp8}
    model = _write_config(tmp_path, _onevision_qwen2_7b(child_dtype) | {"quantization_config": quant})
    assert _compiled_vllm_transfer_widths(model, "0.32.0") == (28_672, 28_672)
    assert _estimate_model_weight_bytes(model, backend="vllm", backend_version="0.32.0") == weights
    if quantize_linears:
        group = groups["group_0"]
        group["format"] = "dense"
        model = _write_config(tmp_path, _onevision_qwen2_7b(child_dtype) | {"quantization_config": quant})
        with pytest.raises(ValueError, match="W8 weights in 'dense'"):
            memory.estimate_num_gpu_blocks(model, "missing", "vllm", "0.32.0", **args)
        group["format"] = "float-quantized"
        quant["format"] = "dense"
        model = _write_config(tmp_path, _onevision_qwen2_7b(child_dtype) | {"quantization_config": quant})
        assert _estimate_model_weight_bytes(model, backend="vllm", backend_version="0.32.0") == weights
        for order in ("group", "dynamic"):
            groups["group_0"]["weights"]["actorder"] = order
            model = _write_config(tmp_path, _onevision_qwen2_7b(child_dtype) | {"quantization_config": quant})
            with pytest.raises(ValueError, match="activation ordering"):
                memory.estimate_num_gpu_blocks(model, "missing", "vllm", "0.32.0", **args)
    else:
        quant["quant_method"] = "COMPRESSED_TENSORS"
        model = _write_config(tmp_path, _onevision_qwen2_7b(child_dtype) | {"compression_config": quant})
        assert _compiled_vllm_transfer_widths(model, "0.32.0") == (28_672, 28_672)
        groups["attention"]["targets"] = ["Linear"]
        model = _write_config(tmp_path, _onevision_qwen2_7b(child_dtype) | {"compression_config": quant})
        with pytest.raises(ValueError, match="weight scheme"):
            _compiled_vllm_transfer_widths(model, "0.32.0")


@pytest.mark.parametrize(
    "architecture", ["Qwen2ForCausalLM", "Qwen3ForCausalLM", "LlamaForCausalLM", "MistralForCausalLM"]
)
def test_vllm_encoder_only_decoder_has_no_persistent_transfer(tmp_path, architecture):
    from aisim.capacity import estimate_kv_bytes_per_token

    kind = architecture.removesuffix("ForCausalLM").lower()
    child = {"model_type": kind, "architectures": [architecture], "is_causal": False}
    model = _write_config(tmp_path, {"architectures": ["LlavaNextForConditionalGeneration"], "text_config": child})
    with pytest.raises(ValueError, match="no persistent vLLM KV cache"):
        estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1, backend_version="0.32.0")
    with pytest.raises(ValueError, match="no persistent vLLM KV cache"):
        _compiled_vllm_transfer_widths(model, "0.32.0")


@pytest.mark.parametrize("kind", ["apertus", "deepseek_v3"])
def test_vllm_apertus_and_deepseek_encoder_attention_has_no_persistent_transfer(tmp_path, kind):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    if kind == "deepseek_v3":
        child = _load_pre_downloaded_hf_config("deepseek-ai/DeepSeek-V3")
        child["architectures"] = ["DeepseekV2ForCausalLM"]
    else:
        child = dict(
            model_type=kind,
            architectures=["ApertusForCausalLM"],
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=14336,
            vocab_size=131072,
        )
    parent = {"architectures": ["LlavaForConditionalGeneration"], "text_config": child}
    child["is_causal"] = True
    assert _compiled_vllm_transfer_widths(_write_config(tmp_path, parent), "0.32.0")[0] > 0
    child["is_causal"] = False
    with pytest.raises(ValueError, match="no persistent vLLM KV cache"):
        _compiled_vllm_transfer_widths(_write_config(tmp_path, parent), "0.32.0")


@pytest.mark.parametrize("key", ["quantization_config", "compression_config"])
def test_vllm_opt_with_compressed_kv_and_unquantized_weights(tmp_path, key):
    child = dict(model_type="opt", architectures=["OPTForCausalLM"], vocab_size=50272)
    quant = {
        "quant_method": "compressed-tensors",
        "config_groups": {},
        "format": "dense",
        "kv_cache_scheme": {"type": "float", "num_bits": 8, "strategy": "tensor", "symmetric": True},
    }
    path = _write_config(
        tmp_path, {"architectures": ["LlavaForConditionalGeneration"], "text_config": child, key: quant}
    )
    assert _compiled_vllm_transfer_widths(path, "0.32.0") == (18_432, 18_432)
    opts = _missing_system_cache_options(tmp_path, tp_size=1, capacity=2 * _GIB)
    assert memory.estimate_num_gpu_blocks(path, "missing", "vllm", "0.32.0", **opts) > 0
    quant.update(quant_method="modelopt")
    path = _write_config(
        tmp_path, {"architectures": ["LlavaForConditionalGeneration"], "text_config": child, key: quant}
    )
    with pytest.raises(ValueError, match="quant_algo"):
        memory.estimate_num_gpu_blocks(path, "missing", "vllm", "0.32.0", **opts)


@pytest.mark.parametrize(
    "parent_type,wrappers,child_type,head_dim,expected",
    [
        ("llava_next", ["LlavaNextForConditionalGeneration"], "qwen2", 64, 147_456),
        ("interns1", ["InternS1ForConditionalGeneration", "InternVLForConditionalGeneration"], "qwen3", 128, 147_456),
        ("interns1", ["InternVLLMHeadModel", "InternS1ForConditionalGeneration"], None, 64, 73_728),
        ("internvl", ["InternS1ForConditionalGeneration", "InternVLForConditionalGeneration"], None, 64, 147_456),
    ],
)
def test_vllm_uses_the_parent_config_and_registered_language_model(
    tmp_path, parent_type, wrappers, child_type, head_dim, expected
):
    from aisim.capacity import estimate_kv_bytes_per_token

    # Both InternVL names load InternS1 in vLLM. Its HF parents default to Qwen3
    # and Qwen2 respectively. Qwen3 reads head_dim, while Qwen2 uses 4096 / 32.
    child = dict(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=36,
        head_dim=head_dim,
        intermediate_size=12288,
        max_position_embeddings=65536,
        vocab_size=153216,
    )
    if child_type:
        child["model_type"] = child_type
    model = _write_config(tmp_path, {"model_type": parent_type, "architectures": wrappers, "text_config": child})
    for version in ("0.31.0", "0.32.0", "0.33.0"):
        assert estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1, backend_version=version) == expected
        assert _compiled_vllm_transfer_widths(model, version) == (expected, expected)


@pytest.mark.parametrize("config_type", ["falcon", "gpt2", "gptj", "gpt-sw3", "gpt_neo", "gemma4_text", "deepseek_v3"])
def test_vllm_opt_uses_its_head_count_with_foreign_config_aliases(tmp_path, config_type):
    child = dict(
        model_type=config_type,
        architectures=["OPTForCausalLM"],
        multi_query=True,
        num_key_value_heads=1,
        ffn_dim=3072,
        word_embed_proj_dim=768,
        enable_bias=True,
        activation_function="relu",
        layer_norm_elementwise_affine=True,
        do_layer_norm_before=False,
        vocab_size=50272,
    )
    if config_type in ("gpt2", "gptj", "gpt-sw3"):
        child.update(n_head=12, n_layer=12, n_embd=768, n_positions=2048)
    elif config_type == "gpt_neo":
        child.update(
            num_heads=12,
            num_layers=12,
            hidden_size=768,
            max_position_embeddings=2048,
            attention_types=[[["global", "local"], 6]],
        )
    else:
        child.update(num_attention_heads=12, num_hidden_layers=12, hidden_size=768, max_position_embeddings=2048)
    if config_type == "gemma4_text":
        child.update(
            layer_types=["full_attention"] * 12, global_head_dim=64, num_global_key_value_heads=1, attention_k_eq_v=True
        )
    path = _write_config(tmp_path, {"architectures": ["LlavaNextForConditionalGeneration"], "text_config": child})
    assert _compiled_vllm_transfer_widths(path, "0.32.0") == (36_864, 36_864)


def test_qwen2_moe_config_dispatches_dense_weights_and_its_constructed_layer_types(tmp_path):
    from aisim.generator.naive import _estimate_model_weight_bytes

    child = dict(model_type="qwen2_moe", architectures=["Qwen2ForCausalLM"], num_experts=8, moe_intermediate_size=128)
    parent = {"architectures": ["LlavaForConditionalGeneration"], "text_config": child}
    path = _write_config(tmp_path, parent)
    assert _estimate_model_weight_bytes(path, backend="vllm", backend_version="0.32.0") == 4_336_543_744
    assert _compiled_vllm_transfer_widths(path, "0.32.0") == (196_608, 196_608)
    child.update(use_sliding_window=True, max_window_layers=12)
    path = _write_config(tmp_path, parent)
    with pytest.raises(ValueError, match="sliding attention for only some layers"):
        _compiled_vllm_transfer_widths(path, "0.32.0")
    child["sliding_window"] = None
    with pytest.raises(ValueError, match="sliding attention for only some layers"):
        _compiled_vllm_transfer_widths(_write_config(tmp_path, parent), "0.32.0")


@pytest.mark.parametrize(
    ("kind", "architectures", "width"),
    [("ministral3", None, 139_264), ("phi3", ["LlamaForCausalLM"], 393_216)],
)
def test_vllm_dense_class_can_use_another_transformers_config(tmp_path, kind, architectures, width):
    child = {"model_type": kind}
    if architectures is not None:
        child["architectures"] = architectures
    path = _write_config(tmp_path, {"architectures": ["LlavaNextForConditionalGeneration"], "text_config": child})
    assert _compiled_vllm_transfer_widths(path, "0.32.0") == (width, width)


def test_wrapped_gemma_config_uses_llama_projection_weights_and_default_head_dimension(tmp_path):
    from aisim.generator.naive import _estimate_model_weight_bytes

    child = dict(model_type="gemma", architectures=["LlamaForCausalLM"], hidden_act="silu")
    for fields in ({}, {"head_dim": 256}):
        path = _write_config(
            tmp_path, {"architectures": ["LlavaForConditionalGeneration"], "text_config": child | fields}
        )
        # GemmaConfig supplies 16 heads, 256 elements per head and 28 layers to Llama's projections.
        assert _compiled_vllm_transfer_widths(path, "0.32.0") == (458_752, 458_752)
        assert _estimate_model_weight_bytes(path, backend="vllm", backend_version="0.32.0") >= 18_647_875_584


@pytest.mark.parametrize(
    ("child_dtype", "activation", "scales"),
    [("bfloat16", "dynamic", 7), ("float8_e4m3fn", "dynamic", 7), ("bfloat16", "static", 14)],
)
def test_vllm_fp8_linear_weights_keep_parent_dtype_for_embeddings(tmp_path, child_dtype, activation, scales):
    from aisim.generator.naive import _estimate_model_weight_bytes

    document = _onevision_qwen2_7b(child_dtype)
    document["quantization_config"] = {"quant_method": "fp8", "activation_scheme": activation}
    document["text_config"].update(num_hidden_layers=1, hidden_size=4096, num_attention_heads=32)
    path = _write_config(tmp_path, document)
    # Seven FP8 projections each have a weight scale and, for static input quantization, an activation scale.
    opts = dict(backend="vllm", backend_version="0.32.0", tp_size=1, pp_size=1)
    unquantized = {k: v for k, v in document.items() if k != "quantization_config"}
    full = NaiveKVCacheEstimator.from_hf_config(unquantized, **opts).weight_bytes()
    projections = 2 * 4096 * 128 * (32 + 4) + 3 * 4096 * 18944
    assert (
        full - _estimate_model_weight_bytes(path, backend="vllm", backend_version="0.32.0") == projections - 4 * scales
    )
    quant = document["quantization_config"]
    quant["ignored_layers"] = ["language_model.model.layers.0.mlp.down_proj"]
    assert (
        NaiveKVCacheEstimator.from_hf_config(document, **opts).weight_bytes()
        == NaiveKVCacheEstimator.from_hf_config(unquantized, **opts).weight_bytes()
    )
    del quant["activation_scheme"]
    with pytest.raises(ValueError, match="activation_scheme"):
        NaiveKVCacheEstimator.from_hf_config(document, **opts)


def test_vllm_native_block_fp8_counts_rank_local_scale_tiles(tmp_path):
    from aisim.generator.naive import _estimate_model_weight_bytes

    # Intern-S1-mini-FP8 has seven block-FP8 Qwen3 projections per layer and BF16 execution.
    document = {
        "architectures": ["InternS1ForConditionalGeneration"],
        "model_type": "interns1",
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["model.vision_tower.proj", "model.language_model.embed_tokens", "lm_head"],
        },
        "vision_config": dict(
            model_type="interns1_vision",
            hidden_size=64,
            intermediate_size=192,
            num_hidden_layers=2,
            image_size=[64, 64],
            patch_size=[8, 8],
        ),
        "text_config": {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "torch_dtype": "bfloat16",
            "hidden_size": 4096,
            "head_dim": 128,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 36,
            "intermediate_size": 12288,
            "vocab_size": 153216,
        },
    }
    model = _write_config(tmp_path, document)
    full = {key: value for key, value in document.items() if key != "quantization_config"}
    for tp, projections, scales in ((1, 6_945_767_424, 1_695_744), (16, 452_984_832, 110_592)):
        # QKV replicates one 128-wide KV head on each TP16 rank. The four local fused matrices own their scale tiles.
        opts = dict(backend="vllm", backend_version="0.32.0", tp_size=tp, pp_size=1)
        serving_dtype = NaiveKVCacheEstimator.from_hf_config(full, **opts).weight_bytes()
        physical = NaiveKVCacheEstimator.from_hf_config(document, **opts).weight_bytes()
        assert serving_dtype - physical == projections - scales
        if tp == 1:
            assert _estimate_model_weight_bytes(model, backend="vllm", backend_version="0.32.0") == physical
    options = _missing_system_cache_options(tmp_path, tp_size=1, capacity=16 * _GIB)
    options.update(max_num_tokens=512, max_batch_size=1, memory_fraction_value=1, tolerance_fraction=0)
    assert memory.estimate_num_gpu_blocks(model, "missing", "vllm", "0.32.0", **options) > 0


def test_dispatched_deepseek_counts_resident_weights_and_routed_layers(tmp_path):
    from aisim.generator.naive import _estimate_model_weight_bytes
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("deepseek-ai/DeepSeek-V3")
    raw.pop("quantization_config", None)
    raw.pop("n_routed_experts", None)
    raw.update(architectures=["DeepseekV2ForCausalLM"], first_k_dense_replace=30)
    for dtype in ("bfloat16", "float8_e4m3fn"):
        rank = NaiveKVCacheEstimator.from_hf_config(
            raw | {"torch_dtype": dtype},
            backend="vllm",
            backend_version="0.32.0",
            tp_size=8,
            pp_size=1,
            moe_tp_size=8,
        )
        # The 31 resident routed FFNs alone exceed an 80 GiB rank.
        assert rank.weight_bytes() >= 31 * 256 * 3 * 7168 * 2048 * 2 // 8 > 80 * _GIB
    raw.update(first_k_dense_replace=61, torch_dtype="bfloat16")
    path = _write_config(tmp_path, raw)
    for backend, release in (("vllm", "0.32.0"), ("sglang", "0.5.23")):
        metadata = {}
        assert (
            _estimate_model_weight_bytes(path, backend=backend, backend_version=release, model_metadata=metadata)
            == 74_893_717_504
        )
        assert metadata["is_moe"] is False
    raw.update(first_k_dense_replace=60, quantization_config={"quant_method": "fp8", "activation_scheme": "dynamic"})
    estimator = NaiveKVCacheEstimator.from_hf_config(
        raw, backend="sglang", backend_version="0.5.23", tp_size=8, pp_size=1, moe_tp_size=8
    )
    assert estimator.geometry["num_routed_layers"] == 1
    assert 0 < estimator.weight_bytes() < 14 * _GIB
    raw["first_k_dense_replace"] = 30
    raw["quantization_config"]["weight_block_size"] = [128, 128]
    options = dict(backend="sglang", backend_version="0.5.23", tp_size=8, pp_size=1, moe_tp_size=8)
    block_fp8 = NaiveKVCacheEstimator.from_hf_config(raw, **options).weight_bytes()
    raw["quantization_config"]["ignored_layers"] = ["mlp.experts"]
    bf16 = NaiveKVCacheEstimator.from_hf_config(raw, **options).weight_bytes()
    # Each rank keeps 256 experts in 31 layers, each with three 7168 x 256 matrices and FP32 scales.
    params = 31 * 256 * 3 * 7168 * 256
    assert bf16 - block_fp8 == params - 4 * (params // (128 * 128))
    assert block_fp8 < 80 * _GIB < bf16
    for release, expected in (("0.5.9", block_fp8), ("0.5.10rc0", bf16)):
        assert (
            NaiveKVCacheEstimator.from_hf_config(raw, **(options | {"backend_version": release})).weight_bytes()
            == expected
        )
    del raw["quantization_config"]["weight_block_size"]
    raw["quantization_config"]["ignored_layers"] = []
    for activation, scales in (("dynamic", 3), ("static", 5)):
        raw["quantization_config"]["activation_scheme"] = activation
        tensor_fp8 = NaiveKVCacheEstimator.from_hf_config(raw, **options).weight_bytes()
        assert bf16 - tensor_fp8 == params - 31 * 256 * 4 * scales


def test_sglang_llava_uses_its_pinned_transformers_causal_lm(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("deepseek-ai/DeepSeek-V3.2")
    path = _write_config(tmp_path, {"architectures": ["LlavaForConditionalGeneration"], "text_config": raw})
    (tmp_path / "hopper.yaml").write_text("gpu: {sm_version: 90}\n")
    args = dict(tp_size=1, pp_size=1, system="hopper", systems_paths=[str(tmp_path)], backend="sglang")
    assert estimate_kv_bytes_per_token(path, backend_version="0.5.23", **args) == 78_324
    with pytest.raises(ValueError, match="no causal LM"):
        estimate_kv_bytes_per_token(path, backend_version="0.5.14", **args)


@pytest.mark.parametrize("config_type", ["phi3", "gemma"])
def test_registered_opt_uses_its_projections_with_a_different_config(tmp_path, config_type):
    config = dict(
        model_type=config_type,
        hidden_size=768,
        num_attention_heads=12,
        num_key_value_heads=1,
        head_dim=32,
        num_hidden_layers=12,
        intermediate_size=3072,
        ffn_dim=3072,
        max_position_embeddings=2048,
        vocab_size=50272,
        word_embed_proj_dim=768,
        enable_bias=True,
        activation_function="relu",
        layer_norm_elementwise_affine=True,
        do_layer_norm_before=False,
    )
    for names in (["OPTForCausalLM"], ["UnknownCheckpoint", "OPTLMHeadModel"]):
        config["architectures"] = names
        path = _write_config(tmp_path, {"architectures": ["LlavaNextForConditionalGeneration"], "text_config": config})
        assert _compiled_vllm_transfer_widths(path, "0.32.0") == (36_864, 36_864)
        options = _missing_system_cache_options(tmp_path, tp_size=1, capacity=600 << 20)
        options["naive_kv_reservation"] = 0.8
        with pytest.raises(ValueError, match="no KV budget"):
            memory.estimate_num_gpu_blocks(path, "missing", "vllm", "0.32.0", **options)
        options["gpu_memory_capacity_bytes_override"] = 2 * _GIB
        assert memory.estimate_num_gpu_blocks(path, "missing", "vllm", "0.32.0", **options) > 0


def test_qwen2_projection_uses_the_constructed_config(tmp_path):
    parent = {"architectures": ["LlavaNextForConditionalGeneration"]}
    # Phi3Config derives its KV head count and Qwen2 initializes the rotary parameters.
    phi3 = {"model_type": "phi3", "architectures": ["Qwen2ForCausalLM"]}
    for fields in ({}, {"num_key_value_heads": 32, "rope_parameters": {"rope_type": "default", "rope_theta": 10000}}):
        path = _write_config(tmp_path, parent | {"text_config": phi3 | fields})
        assert _compiled_vllm_transfer_widths(path, "0.32.0") == (393_216, 393_216)
    vision = dict(
        model_type="clip_vision_model",
        hidden_size=1024,
        num_attention_heads=16,
        num_hidden_layers=24,
        intermediate_size=4096,
    )
    for fields in ({}, {"architectures": ["Qwen2ForCausalLM"]}):
        path = _write_config(tmp_path, parent | {"text_config": vision | fields})
        with pytest.raises(ValueError, match="causal LM|Qwen2ForCausalLM requires"):
            _compiled_vllm_transfer_widths(path, "0.32.0")
    path = _write_config(tmp_path, {"architectures": ["LlavaForConditionalGeneration"], "text_config": vision})
    with pytest.raises(ValueError, match="no causal LM"):
        NaiveKVCacheEstimator.from_model_path(
            path, tp_size=1, pp_size=1, backend="sglang", backend_version="0.5.20", allow_hf_config_download=False
        )


def test_vllm_stops_at_a_registered_decoder_even_when_its_geometry_is_unknown():
    def width(first, version, last="LlamaForCausalLM", auto_map=None):
        text = {"model_type": "llama", "architectures": [first, last], "auto_map": auto_map}
        wrapped = {"architectures": ["LlavaNextForConditionalGeneration"], "text_config": text}
        return NaiveKVCacheEstimator.from_hf_config(
            wrapped, tp_size=1, pp_size=1, backend_version=version
        ).kv_bytes_per_token()

    # Dory first appeared in v0.33.0. Older registries reach the default 32-layer Llama.
    assert width("DoryForCausalLM", "0.32.0") == 524_288
    for version in ("v0.32.0", "0.31.0rc5", "0.33.0rc3"):
        assert width("VLlama3ForCausalLM", version) == 524_288
    with pytest.raises(ValueError, match="conflicts with architectures='DoryForCausalLM'"):
        width("DoryForCausalLM", "0.33.0")
    with pytest.raises(ValueError, match="depends on the backend release"):
        width("DoryForCausalLM", None)
    with pytest.raises(ValueError, match="conflicts with architectures='Phi3ForCausalLM'"):
        width("Phi3ForCausalLM", "0.32.0")
    assert width("VLlama3ForCausalLM", "0.32.0", "LlamaLMHeadModel", {"AutoConfig": "remote.Config"}) == 524_288
    assert width("ReformerModel", "0.32.0", "LlamaLMHeadModel", {"AutoModel": "remote.Model"}) == 524_288
    for first, auto_map in (
        ("BertConfig", None),
        ("AlbertModel", {"AutoModel": "remote.Model"}),
        ("VLlama3ForCausalLM", {"AutoModel": "remote.Model"}),
    ):
        with pytest.raises(ValueError, match="conflicts with architectures"):
            width(first, "0.32.0", "LlamaLMHeadModel", auto_map)


def test_bare_normalized_decoders_use_their_local_layers_and_experts(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    # Both registered choices use the generic estimate, even when the chosen class varies by release.
    generic = {**_UNKNOWN_DECODER, "model_type": "falcon", "architectures": ["DoryForCausalLM", "BloomForCausalLM"]}
    assert NaiveKVCacheEstimator.from_hf_config(generic, tp_size=1, pp_size=1).kv_bytes_per_token() == 1_024
    nested = {
        "text_config": _UNKNOWN_DECODER,
        "model_type": "falcon",
        "architectures": ["DoryForCausalLM", "OPTForCausalLM"],
    }
    assert NaiveKVCacheEstimator.from_hf_config(nested, tp_size=1, pp_size=1).kv_bytes_per_token() == 1_024

    path = _write_config(
        tmp_path, {"model_type": "llama", "architectures": ["LLaMAForCausalLM"], "num_key_value_heads": 8}
    )
    assert estimate_kv_bytes_per_token(path, tp_size=1, pp_size=2, backend_version="0.32.0") == 65_536
    with pytest.raises(ValueError, match="divisible by tp_size"):
        estimate_kv_bytes_per_token(path, tp_size=3, pp_size=1, backend_version="0.32.0")

    llama = _load_pre_downloaded_hf_config("meta-llama/Meta-Llama-3.1-8B")
    llama["architectures"] = ["LLaMAForCausalLM"]
    opts = _missing_system_cache_options(tmp_path, tp_size=2, capacity=80 * _GIB)
    del opts["dcp_size"]
    model = _write_config(tmp_path, llama)
    assert memory.estimate_num_gpu_blocks(model, "missing", "vllm", "0.32.0", **opts) == memory.estimate_num_gpu_blocks(
        model, "missing", "vllm", "0.32.0", dcp_size=1, **opts
    )
    del llama["num_hidden_layers"]
    for names in (False, {}, [], ""):
        with pytest.raises(ValueError, match="architectures"):
            _compiled_vllm_transfer_widths(_write_config(tmp_path, llama | {"architectures": names}), "0.32.0")

    deepseek = _load_pre_downloaded_hf_config("deepseek-ai/DeepSeek-V3")
    deepseek.pop("quantization_config", None)
    deepseek["architectures"] = ["DeepseekV2ForCausalLM"]
    options = dict(tp_size=8, pp_size=1, moe_tp_size=8, moe_ep_size=1, backend_version="0.32.0")
    explicit = NaiveKVCacheEstimator.from_hf_config(deepseek, **options).weight_bytes()
    del deepseek["num_hidden_layers"], deepseek["n_routed_experts"]
    inferred = NaiveKVCacheEstimator.from_hf_config(deepseek, **options).weight_bytes()
    # The routed BF16 FFNs alone span 58 layers, each with 256 experts sharded over 8 GPUs.
    assert explicit == inferred >= 58 * 256 * 3 * 7168 * 2048 * 2 // 8 > 80 * _GIB
    # Zero routed layers retain the native MLA and dense matrices (10,977,280,000 BF16 bytes/rank).
    dense = deepseek | {"first_k_dense_replace": 61}
    for model in (dense, {"architectures": ["LlavaNextForConditionalGeneration"], "text_config": dense}):
        path = _write_config(tmp_path, model)
        assert _compiled_vllm_transfer_widths(path, "0.32.0") == (70_272, 70_272)
        opts = _missing_system_cache_options(tmp_path, tp_size=8, capacity=10 * _GIB)
        opts.update(moe_tp_size=8, moe_ep_size=1, naive_kv_reservation=0.8)
        with pytest.raises(ValueError, match="no KV budget"):
            memory.estimate_num_gpu_blocks(path, "missing", "vllm", "0.32.0", **opts)
        opts["gpu_memory_capacity_bytes_override"] = 12 * _GIB
        assert memory.estimate_num_gpu_blocks(path, "missing", "vllm", "0.32.0", **opts) == (
            1353 if model is dense else 837
        )
    deepseek["architectures"] = ["DeepseekV3BidirectionalModel", "DeepseekV3ForCausalLM"]
    parent = {"architectures": ["LlavaNextForConditionalGeneration"], "text_config": deepseek}
    assert NaiveKVCacheEstimator.from_hf_config(parent, **options).kv_bytes_per_token() == 70_272


def test_real_opt_child_reserves_positions_and_defaults_the_embedding_projection(tmp_path):
    options = _missing_system_cache_options(tmp_path, tp_size=1, capacity=1000 << 20)
    parent = {"architectures": ["LlavaNextForConditionalGeneration"]}
    for names in ({}, {"architectures": ["OPTForCausalLM"]}):
        text = {"model_type": "opt", **names}
        for embedding in ({}, {"word_embed_proj_dim": None}, {"word_embed_proj_dim": 768}):
            config = parent | {"text_config": text | embedding}
            estimate = NaiveKVCacheEstimator.from_hf_config(config, tp_size=1, pp_size=1, backend_version="0.32.0")
            assert estimate.kv_bytes_per_token() == 36_864
            assert 900 << 20 > estimate.weight_bytes() > 800 << 20
        path = _write_config(tmp_path, parent | {"text_config": text | {"max_position_embeddings": 131072}})
        with pytest.raises(ValueError, match="no KV budget"):
            memory.estimate_num_gpu_blocks(path, "missing", "vllm", "0.32.0", **options)


@pytest.mark.parametrize("classes", [["UnknownOPTCheckpoint", "OPTForCausalLM"], ["OPTLMHeadModel"]])
def test_flat_opt_dispatch_reserves_learned_positions(tmp_path, classes):
    cfg = {
        "model_type": "opt",
        "torch_dtype": "bfloat16",
        "architectures": classes,
        "vocab_size": 50272,
        "max_position_embeddings": 131072,
    }
    path = _write_config(tmp_path, cfg)
    opts = _missing_system_cache_options(tmp_path, tp_size=1, capacity=400 << 20)
    opts["memory_fraction_value"] = 1
    for version in ("0.31.0", "0.32.0", "0.33.0"):
        # The tied BF16 weights alone occupy 448,659,456 bytes. The config-only
        # estimate conservatively reserves 526,218,240 bytes, including both vocabulary matrices.
        with pytest.raises(ValueError, match=r"non-KV memory \(526218240 bytes\)"):
            memory.estimate_num_gpu_blocks(path, "missing", "vllm", version, **opts)
        # One physical block is reserved by the vLLM scheduler.
        assert (
            memory.estimate_num_gpu_blocks(
                path, "missing", "vllm", version, **(opts | {"gpu_memory_capacity_bytes_override": 600 << 20})
            )
            == ((600 << 20) - 526_218_240) // (16 * 36_864) - 1
        )
    model = _write_config(tmp_path, cfg | {"architectures": ["BloomForCausalLM", "OPTForCausalLM"]})
    with pytest.raises(ValueError, match="conflicts with architectures='BloomForCausalLM'"):
        memory.estimate_num_gpu_blocks(model, "missing", "vllm", "0.32.0", **opts)


@pytest.mark.parametrize("model_type", ["", "fake_llama_config"])
def test_registered_wrapper_resolves_the_child_config_before_the_architecture(model_type):
    text = dict(_UNKNOWN_DECODER, model_type=model_type, architectures=["LlamaForCausalLM"])
    for parent in ("LlavaNextForConditionalGeneration", "Idefics3ForConditionalGeneration"):
        config = {"architectures": [parent], "text_config": text}
        with pytest.raises(ValueError, match="text_config.model_type"):
            NaiveKVCacheEstimator.from_hf_config(config, tp_size=1, pp_size=1, backend_version="0.32.0")


def test_gemma_readers_compare_only_the_rank_dimensions_used_by_the_adapter():
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    doc = _load_pre_downloaded_hf_config("google/gemma-4-26B-A4B")
    text = doc["text_config"]
    text["attention_k_eq_v"] = False
    assert (
        NaiveKVCacheEstimator.from_hf_config(doc, tp_size=8, pp_size=1, backend="sglang").kv_bytes_per_token() == 35_840
    )
    with pytest.raises(ValueError, match="SGLang release"):
        NaiveKVCacheEstimator.from_hf_config(doc, tp_size=1, pp_size=1, backend="sglang")
    text.update(
        head_dim=256, global_head_dim=256, num_key_value_heads=8, num_global_key_value_heads=8, attention_k_eq_v=True
    )
    text["per_layer_config"] = {
        str(i): {"num_key_value_heads": 2} for i, kind in enumerate(text["layer_types"]) if kind == "full_attention"
    }
    for changes in ({}, {"0": {"num_key_value_heads": 4}}):
        text["per_layer_config"].update(changes)
        estimator = NaiveKVCacheEstimator.from_hf_config(doc, tp_size=8, pp_size=1, backend_version="0.31.0")
        assert estimator.kv_bytes_per_token() == 30_720
        with pytest.raises(ValueError, match="cache geometry|runtime profile"):
            NaiveKVCacheEstimator.from_hf_config(doc, tp_size=1, pp_size=1, backend_version="0.31.0")


@pytest.mark.parametrize("dtype_key", ["dtype", "torch_dtype"])
def test_registered_decoder_with_an_unset_dtype_matches_its_default_serialization(tmp_path, dtype_key):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    model = _load_pre_downloaded_hf_config("meta-llama/Meta-Llama-3.1-8B")
    for key in ("dtype", "torch_dtype"):
        model.pop(key, None)
    for precision in ({}, {dtype_key: None}, {dtype_key: "float16"}):
        checkpoint = _write_config(tmp_path, model | precision)
        assert estimate_kv_bytes_per_token(checkpoint, tp_size=1, pp_size=1) == 131_072
    checkpoint = _write_config(tmp_path, model | {dtype_key: {"invalid": "dtype"}})
    with pytest.raises(ValueError, match="must be a nonempty string"):
        estimate_kv_bytes_per_token(checkpoint, tp_size=1, pp_size=1)


def test_selected_llama_uses_its_constructor_hidden_size_for_cache_geometry(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token

    text = dict(
        architectures=["LlamaForCausalLM"],
        model_type="llama",
        head_dim=128,
        num_hidden_layers=32,
        num_key_value_heads=32,
        max_position_embeddings=4096,
        vocab_size=32064,
        torch_dtype="float16",
    )
    wrapper = {"architectures": ["LlavaForConditionalGeneration"], "text_config": text}
    assert estimate_kv_bytes_per_token(_write_config(tmp_path, wrapper), tp_size=1, pp_size=1) == 524_288
    with pytest.raises(ValueError, match="config parser refuses"):
        NaiveKVCacheEstimator.from_hf_config(text, tp_size=1, pp_size=1)


def test_selected_moe_weights_use_the_constructor_count_and_routed_width(tmp_path):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    opts = _missing_system_cache_options(tmp_path, tp_size=1, capacity=24 * _GIB)
    for name, count, width, backend, version in (
        ("deepseek-ai/DeepSeek-V3", 256, 2048, "sglang", "0.5.20"),
        ("moonshotai/Kimi-K2.5", 256, 2048, "sglang", "0.5.20"),
        ("MiniMaxAI/MiniMax-M2.5", 256, 1536, "sglang", "0.5.20"),
        ("Qwen/Qwen3-VL-30B-A3B-Instruct", 60, 768, "sglang", "0.5.20"),
        ("mistralai/Mixtral-8x7B-v0.1", 8, 14336, "vllm", "0.33.0"),
    ):
        checkpoint = _load_pre_downloaded_hf_config(name)
        raw = checkpoint.get("text_config", checkpoint)
        raw.pop("num_local_experts", None)
        raw.pop("num_experts", None)
        raw.pop("n_routed_experts", None)
        if name.startswith("mistralai/"):
            raw["moe_intermediate_size"] = 1024
        wrapped = {"text_config": raw}
        model = _write_config(tmp_path, wrapped)
        shape = NaiveKVCacheEstimator.from_hf_config(wrapped, tp_size=1, pp_size=1).geometry
        assert (shape["num_experts"], shape["moe_inter"]) == (count, width)
        for estimate in (memory.estimate_num_gpu_blocks, memory.estimate_kv_cache):
            with pytest.raises(ValueError, match="no KV budget"):
                estimate(model, "missing", backend, version, **opts)


@pytest.mark.parametrize(
    ("backend", "release", "value_dim", "supported"),
    [
        ("sglang", "0.5.6.post2", 256, True),
        ("sglang", "0.5.7", 256, False),
        ("sglang", "0.5.18", 0, False),
        ("sglang", "0.5.19", 0, True),
        ("sglang", "0.5.20", 128, True),
        ("sglang", "0.5.20", 256, False),
        ("vllm", "0.33.0", 256, True),
    ],
)
def test_selected_dense_value_dimension_matches_the_serving_pool(tmp_path, backend, release, value_dim, supported):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    text = _load_pre_downloaded_hf_config("Qwen/Qwen3-0.6B")
    text["v_head_dim"] = value_dim
    model = _write_config(tmp_path, {"architectures": ["LlavaForConditionalGeneration"], "text_config": text})
    opts = _missing_system_cache_options(tmp_path, tp_size=1, capacity=24 * _GIB)
    for fn, args, kwargs in (
        (estimate_kv_bytes_per_token, (model,), dict(tp_size=1, pp_size=1, backend=backend, backend_version=release)),
        (memory.estimate_kv_cache, (model, "missing", backend, release), opts),
        (memory.estimate_num_gpu_blocks, (model, "missing", backend, release), opts),
    ):
        if supported:
            assert fn(*args, **kwargs)
        else:
            with pytest.raises(ValueError, match="asymmetric SGLang"):
                fn(*args, **kwargs)


@pytest.mark.parametrize(
    ("release", "headers", "configured", "gpu", "error"),
    [
        ("0.32.0", ("F16",), None, {"sm_version": 90}, None),
        ("0.32.1rc0", ("F16",), None, {"sm_version": 90}, "vLLM float16 model dtype"),
        ("0.33.0", ("BF16",), None, {"sm_version": 90}, None),
        ("0.33.0", ("F16",), "bfloat16", {"sm_version": 90}, None),
        ("0.33.0", ("BF16",), "float16", {"sm_version": 90}, "vLLM float16 model dtype"),
        ("0.33.0", ("F32", "F16"), None, {"sm_version": 90}, None),
        ("0.33.0", ("F32", "F16"), None, {"sm_version": 75}, "vLLM float16 model dtype"),
        ("0.33.0", ("F8_E4M3",), None, {"sm_version": 90}, None),
        ("0.33.0", ("F8_E4M3",), None, {"sm_version": 75}, "vLLM float16 model dtype"),
        ("0.33.0", (), "float8_e4m3fn", {"sm_version": 75}, "vLLM float16 model dtype"),
        ("0.33.0", ("F8_E5M2", "BF16"), None, {"sm_version": 90}, None),
        ("0.33.0", (), "float32", {"sm_version": 75}, "vLLM float16 model dtype"),
        ("0.33.0", (), "float32", {"name": "Intel B60"}, None),
        ("0.33.0", ("F64", "BF16"), None, {"sm_version": 90}, None),
        ("0.33.0", ("F16", "F64"), None, {"sm_version": 90}, None),
        ("0.33.0", ("F16", "BF16"), None, {"sm_version": 90}, "resolved vLLM model dtype"),
        ("0.33.0", ("F64", "BF16"), None, {"name": "Intel B60"}, None),
        ("0.33.0", (), "float64", {"name": "Intel B60"}, None),
        ("0.33.0", ("F16",), None, {"name": "Intel B60"}, "vLLM float16 model dtype"),
        ("0.33.0", ("F64", "BF16"), None, {"sm_version": 70}, "vLLM float16 model dtype"),
        ("0.33.0", ("F64", "BF16"), None, {}, "preferred vLLM model dtype"),
    ],
)
def test_explicit_vllm_bfloat16_cache_reads_local_weight_headers(tmp_path, release, headers, configured, gpu, error):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    text = _load_pre_downloaded_hf_config("Qwen/Qwen3-0.6B")
    text.pop("dtype", None)
    text.pop("torch_dtype", None)
    if configured:
        text["dtype"] = configured
    model = _write_config(tmp_path, {"text_config": text})
    for i, dtype in enumerate(headers):
        size = {"F8_E4M3": 1, "F8_E5M2": 1, "F16": 2, "BF16": 2, "F32": 4, "F64": 8}[dtype]
        tensor = {"dtype": dtype, "shape": [1], "data_offsets": [0, size]}
        header = json.dumps({f"weight_{i}": tensor}).encode()
        header += b" " * (-len(header) % 8)
        (tmp_path / f"model-{i}.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + bytes(size))
    (tmp_path / "target.yaml").write_text(json.dumps({"gpu": gpu}))
    opts = _missing_system_cache_options(tmp_path, tp_size=1, capacity=24 * _GIB)
    for fn, args, kwargs in (
        (
            estimate_kv_bytes_per_token,
            (model,),
            dict(
                tp_size=1,
                pp_size=1,
                backend="vllm",
                backend_version=release,
                system="target",
                systems_paths=[str(tmp_path)],
            ),
        ),
        (memory.estimate_kv_cache, (model, "target", "vllm", release), opts),
        (memory.estimate_num_gpu_blocks, (model, "target", "vllm", release), opts),
    ):
        if error:
            with pytest.raises(ValueError, match=error):
                fn(*args, **kwargs, kvcache_quant_mode="bfloat16")
        else:
            assert fn(*args, **kwargs, kvcache_quant_mode="bfloat16")


def test_sglang_pool_heads_and_vllm_float_cache_use_the_serving_config(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("Qwen/Qwen3-0.6B")
    raw["n_head_kv"] = 16
    model = _write_config(tmp_path, {"text_config": raw})
    with pytest.raises(ValueError, match="pool KV heads"):
        estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1, backend="sglang", backend_version="0.5.20")
    assert estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1, backend_version="0.33.0") == 114_688
    raw.update(n_head_kv=8, num_kv_heads=16, torch_dtype="float16", dtype="float16")
    _write_config(tmp_path, {"text_config": raw})
    opts = _missing_system_cache_options(tmp_path, tp_size=1, capacity=24 * _GIB)
    for backend, version, mismatch in (
        ("sglang", "0.5.20", False),
        ("vllm", "0.32.0", False),
        ("vllm", "0.32.1rc0", True),
        ("vllm", "0.33.0rc0", True),
        ("vllm", "0.33.0", True),
    ):
        width = dict(tp_size=1, pp_size=1, backend=backend, backend_version=version, kvcache_quant_mode="bfloat16")
        if mismatch:
            with pytest.raises(ValueError, match="vLLM float16 model dtype"):
                estimate_kv_bytes_per_token(model, **width)
            with pytest.raises(ValueError, match="vLLM float16 model dtype"):
                memory.estimate_num_gpu_blocks(
                    model, "missing", backend, version, **opts, kvcache_quant_mode="bfloat16"
                )
        else:
            assert estimate_kv_bytes_per_token(model, **width) == 114_688
            assert (
                memory.estimate_num_gpu_blocks(
                    model, "missing", backend, version, **opts, kvcache_quant_mode="bfloat16"
                )
                > 0
            )
    _write_config(tmp_path, {"dtype": "bfloat16", "text_config": raw})
    assert estimate_kv_bytes_per_token(model, **width) == 114_688

    raw.pop("dtype")
    raw.pop("torch_dtype")
    raw.pop("architectures")
    raw["multi_query"] = True
    for subconfig in ("vision_config", "encoder_config"):
        _write_config(tmp_path, {"text_config": raw, subconfig: {"dtype": "float16"}})
        with pytest.raises(ValueError, match="vLLM float16 model dtype"):
            estimate_kv_bytes_per_token(model, **width)
        _write_config(tmp_path, {"text_config": raw, subconfig: {"dtype": "bfloat16"}})
        assert estimate_kv_bytes_per_token(model, **width) == 114_688

    llama = _load_pre_downloaded_hf_config("meta-llama/Meta-Llama-3.1-8B")
    width = dict(tp_size=1, pp_size=1, backend="sglang", backend_version="0.5.20")
    _write_config(tmp_path, {"text_config": dict(llama, n_head_kv=8)})
    assert estimate_kv_bytes_per_token(model, **width) == 131_072
    _write_config(tmp_path, {"text_config": dict(llama, n_head_kv=16)})
    with pytest.raises(ValueError, match="pool KV heads"):
        estimate_kv_bytes_per_token(model, **width)


def test_fp8_expert_tiles_follow_implicit_vllm_tp_and_the_qwen_vl_expert_module(tmp_path):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    fp8 = _load_pre_downloaded_hf_config("Qwen/Qwen3-30B-A3B-FP8")
    model = _write_config(tmp_path, {"text_config": fp8})
    opts = _missing_system_cache_options(tmp_path, tp_size=8, capacity=192 * _GIB)
    opts.update(moe_tp_size=None, moe_ep_size=None)
    for estimate in (memory.estimate_num_gpu_blocks, memory.estimate_kv_cache):
        with pytest.raises(ValueError, match="weight_block_size=128"):
            estimate(model, "missing", "vllm", "0.28.0", **opts)
        assert estimate(model, "missing", "vllm", "0.29.0rc1", **opts)
    raw = _load_pre_downloaded_hf_config("Qwen/Qwen3-VL-30B-A3B-Instruct")["text_config"]
    quant = fp8["quantization_config"]
    raw["quantization_config"] = quant
    opts.update(moe_tp_size=8, moe_ep_size=1)
    for exclusions in (["model.language_model.layers.1.mlp.experts"], ["mlp.experts"], ["language_model"]):
        quant["modules_to_not_convert"] = exclusions
        _write_config(tmp_path, {"text_config": raw})
        if exclusions != ["model.language_model.layers.1.mlp.experts"]:
            assert memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.20", **opts) > 0
        else:
            with pytest.raises(ValueError, match="weight_block_size=128"):
                memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.20", **opts)


def test_selected_gemma4_uses_its_effective_window_for_single_request_retention():
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("google/gemma-4-26B-A4B")["text_config"]
    raw.pop("sliding_window", None)
    for attention, window in ((None, 512), ("vision", 512), ("all", 257)):
        text = dict(raw, use_bidirectional_attention=attention)
        implicit = memory.NaiveKVCacheEstimator.from_hf_config({"text_config": text}, tp_size=1, pp_size=1)
        explicit = memory.NaiveKVCacheEstimator.from_hf_config(
            {"text_config": dict(raw, sliding_window=window)},
            tp_size=1,
            pp_size=1,
        )
        assert implicit.geometry["sliding_window"] == window
        assert implicit.get_kvcache_max_tokens(_GIB) == explicit.get_kvcache_max_tokens(_GIB)


def test_selected_runtime_validates_kv_scales_and_explicit_dtype(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("Qwen/Qwen3-0.6B")
    scheme = {"strategy": "tensor", "type": "float", "num_bits": 8, "symmetric": True}
    quant = {"quant_method": "compressed-tensors", "kv_cache_scheme": scheme}
    model = _write_config(tmp_path, {"text_config": raw, "quantization_config": quant})

    def rate(version, **opts):
        return estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1, backend="vllm", backend_version=version, **opts)

    assert rate("0.14.0") == 114_688
    assert rate("0.33.0") == 57_344
    with pytest.raises(ValueError, match="NVFP4 KV cache requires"):
        rate("0.14.0", kvcache_quant_mode="nvfp4")
    for strategy, symmetric in (("attn_head", True), ("tensor", False)):
        scheme.update(strategy=strategy, symmetric=symmetric)
        _write_config(tmp_path, {"text_config": raw, "quantization_config": quant})
        with pytest.raises(ValueError, match="symmetric FP8"):
            rate("0.14.0")
        if symmetric:
            assert rate("0.15.0") == 114_688
            (tmp_path / "hopper.yaml").write_text("data_dir: data/hopper\n")
            (tmp_path / "data/hopper/vllm/0.32.0").mkdir(parents=True)
            (tmp_path / "query_versions.yaml").write_text(
                "schema_version: 1\noverrides: {hopper: {vllm: {current: '0.32.0', previous: '0.14.0'}}}\n"
            )
            roots = dict(system="hopper", systems_paths=[str(tmp_path)])
            for slot in (None, "current"):
                assert rate(slot, **roots) == rate("0.32.0", **roots) == 57_344
            with pytest.raises(ValueError, match="symmetric FP8"):
                rate("previous", **roots)


@pytest.mark.parametrize(
    ("backend", "before", "supported"), [("vllm", "0.14.0", "0.20.0rc1"), ("sglang", "0.5.15.post1", "0.5.16")]
)
def test_selected_decoder_override_requires_the_serving_cache_dtype(tmp_path, backend, before, supported):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    model = _write_config(tmp_path, {"text_config": _load_pre_downloaded_hf_config("Qwen/Qwen3-0.6B")})
    args = dict(tp_size=1, pp_size=1, backend=backend)
    for version in (before, supported):
        with pytest.raises(ValueError, match="no literal int8 KV cache"):
            estimate_kv_bytes_per_token(model, **args, backend_version=version, kvcache_quant_mode="int8")
        assert estimate_kv_bytes_per_token(model, **args, backend_version=version, kvcache_quant_mode="fp8") == 57_344
    with pytest.raises(ValueError, match="NVFP4 KV cache requires"):
        estimate_kv_bytes_per_token(model, **args, backend_version=before, kvcache_quant_mode="nvfp4")
    assert estimate_kv_bytes_per_token(model, **args, backend_version=supported, kvcache_quant_mode="nvfp4") == 32_256


def test_sglang_expert_tiles_use_checkpoint_scope_and_module_exclusions(tmp_path):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("Qwen/Qwen3-30B-A3B-FP8")
    quant = raw.pop("quantization_config")
    model = _write_config(tmp_path, {"text_config": raw, "compression_config": quant})
    opts = _missing_system_cache_options(tmp_path, tp_size=8, capacity=192 * _GIB) | {
        "moe_tp_size": 8,
        "moe_ep_size": 1,
    }
    with pytest.raises(ValueError, match="weight_block_size=128"):
        memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.20", **opts)
    quant["modules_to_not_convert"] = [f"model.layers.{i}.mlp.experts" for i in range(raw["num_hidden_layers"])]
    _write_config(tmp_path, {"text_config": raw, "compression_config": quant})
    explicit = memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.20", **opts)
    assert explicit > 0
    partial = quant["modules_to_not_convert"][:-1]

    for selectors, version, unquantized in (
        (partial, "0.5.20", False),
        (["mlp.experts"], "0.5.20", True),
        (["mlp.expert"], "0.5.10rc0", True),
        (["mlp.expert"], "0.5.10", True),
        (["mlp.expert"], "0.5.11", False),
        (["mlp.experts."], "0.5.12", False),
        (["mlp.experts."], "0.5.13", True),
    ):
        quant["modules_to_not_convert"] = selectors
        _write_config(tmp_path, {"text_config": raw, "compression_config": quant})
        if unquantized:
            assert memory.estimate_num_gpu_blocks(model, "missing", "sglang", version, **opts) == explicit
        else:
            with pytest.raises(ValueError, match="weight_block_size=128"):
                memory.estimate_num_gpu_blocks(model, "missing", "sglang", version, **opts)

    aligned = dict(opts, tp_size=2, moe_tp_size=2)
    quant["modules_to_not_convert"] = []
    _write_config(tmp_path, {"text_config": raw, "compression_config": quant})
    expected = memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.20", **aligned)
    quant["modules_to_not_convert"] = None
    _write_config(tmp_path, {"text_config": raw, "compression_config": quant})
    assert memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.20", **aligned) == expected
    omitted = {key: value for key, value in opts.items() if key not in ("moe_tp_size", "moe_ep_size")}
    with pytest.raises(ValueError, match="weight_block_size=128"):
        memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.20", **omitted)
    quant.update(modules_to_not_convert=["mlp.experts"], weight_block_size=[128])
    _write_config(tmp_path, {"text_config": raw, "compression_config": quant})
    with pytest.raises(ValueError, match="two positive integer"):
        memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.20", **opts)

    raw = _load_pre_downloaded_hf_config("deepseek-ai/DeepSeek-V3")
    opts.update(tp_size=32, moe_tp_size=32)
    for exclusions in (["mlp"], []):
        raw["quantization_config"]["modules_to_not_convert"] = exclusions
        _write_config(tmp_path, {"text_config": raw})
        if exclusions:
            assert memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.20", **opts) > 0
        else:
            with pytest.raises(ValueError, match="weight_block_size=128"):
                memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.20", **opts)


@pytest.mark.parametrize("release", ["0.5.20", "0.5.23"])
def test_sglang_llava_fp8_exclusions_use_the_language_model_expert_prefix(tmp_path, release):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    text = _load_pre_downloaded_hf_config("Qwen/Qwen3-30B-A3B-FP8")
    quant = text.pop("quantization_config")
    wrapper = {"architectures": ["LlavaForConditionalGeneration"], "model_type": "llava", "text_config": text}
    opts = _missing_system_cache_options(tmp_path, tp_size=8, capacity=80 * _GIB)
    opts.update(moe_tp_size=8, moe_ep_size=1, memory_fraction_value=1.0)
    del opts["naive_kv_reservation"]
    model = _write_config(tmp_path, wrapper)
    args = (model, "missing", "sglang", release)
    blocks = memory.estimate_num_gpu_blocks(*args, **opts)
    assert blocks == 158_028
    wrapper["quantization_config"] = quant
    for excluded in (
        ["language_model"],
        [f"language_model.model.layers.{i}.mlp.experts" for i in range(text["num_hidden_layers"])],
        ["vision_tower"],
        [],
    ):
        quant["modules_to_not_convert"] = excluded
        _write_config(tmp_path, wrapper)
        if excluded and excluded[0].startswith("language_model"):
            assert memory.estimate_num_gpu_blocks(*args, **opts) == blocks
        else:
            with pytest.raises(ValueError, match="weight_block_size=128"):
                memory.estimate_num_gpu_blocks(*args, **opts)


def test_sparse_index_transfer_uses_the_platform_and_resident_layers(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    common = dict(tp_size=8, pp_size=1)
    raw = _load_pre_downloaded_hf_config("zai-org/GLM-5")
    model = _write_config(tmp_path, {"text_config": raw})
    for backend, version in (("vllm", "0.33.0"), ("sglang", "0.5.20")):
        assert (
            estimate_kv_bytes_per_token(model, **common, backend=backend, backend_version=version, system="b60")
            == 100_152
        )
    raw = _load_pre_downloaded_hf_config("zai-org/GLM-5.2")
    _write_config(tmp_path, {"text_config": raw})
    assert (
        estimate_kv_bytes_per_token(model, **common, backend_version="0.33.0", system="h200_sxm")
        == 78 * 1152 + 21 * 132
    )
    for system in ("h200_sxm", "b200_sxm"):
        assert (
            estimate_kv_bytes_per_token(
                model, **common, backend_version="0.33.0", system=system, kvcache_quant_mode="bfloat16"
            )
            == 92_628
        )
    with pytest.raises(ValueError, match="SM120 sparse backend"):
        estimate_kv_bytes_per_token(
            model, **common, backend_version="0.33.0", system="rtx_pro_6000_server", kvcache_quant_mode="bfloat16"
        )
    for freq in (None, 0):
        if freq is None:
            del raw["index_topk_freq"]
        else:
            raw["index_topk_freq"] = freq
        _write_config(tmp_path, {"text_config": raw})
        assert (
            estimate_kv_bytes_per_token(model, **common, backend="sglang", backend_version="0.5.20", system="h200_sxm")
            == 92_628
        )
    with pytest.raises(ValueError, match="index_topk_freq"):
        estimate_kv_bytes_per_token(model, **common, backend="sglang", backend_version="0.5.19", system="h200_sxm")
    with pytest.raises(ValueError, match="index_topk_freq"):
        estimate_kv_bytes_per_token(model, **common, backend_version="0.33.0", system="h200_sxm")
    raw["index_topk_freq"] = 4
    raw["index_skip_topk_offset"] = 0
    _write_config(tmp_path, {"text_config": raw})
    # The checkpoint's explicit indexer_types wins over the invalid offset on SGLang 0.5.20.
    assert (
        estimate_kv_bytes_per_token(model, **common, backend="sglang", backend_version="0.5.20", system="h200_sxm")
        == 92_628
    )
    raw.pop("indexer_types")
    _write_config(tmp_path, {"text_config": raw})
    with pytest.raises(ValueError, match="index_skip_topk_offset"):
        estimate_kv_bytes_per_token(model, **common, backend="sglang", backend_version="0.5.20", system="h200_sxm")
    raw["index_skip_topk_offset"] = 1
    _write_config(tmp_path, {"text_config": raw})
    assert (
        estimate_kv_bytes_per_token(model, **common, backend="sglang", backend_version="0.5.20", system="h200_sxm")
        == 92_496
    )
    raw["index_topk_pattern"] = []
    _write_config(tmp_path, {"text_config": raw})
    assert estimate_kv_bytes_per_token(model, **common, backend_version="0.33.0", system="h200_sxm") == 100_152
    raw = _load_pre_downloaded_hf_config("deepseek-ai/DeepSeek-V3.2")
    _write_config(tmp_path, {"text_config": dict(raw, index_head_dim=64)})
    with pytest.raises(ValueError, match="divisible by 128"):
        estimate_kv_bytes_per_token(model, **common, backend_version="0.33.0", system="h200_sxm")
    _write_config(tmp_path, {"text_config": raw})
    with pytest.raises(ValueError, match="runtime latent and indexer"):
        estimate_kv_bytes_per_token(
            model, **common, backend_version="0.33.0", system="rtx_pro_6000_server", kvcache_quant_mode="bfloat16"
        )
    assert (
        estimate_kv_bytes_per_token(
            model, **common, backend_version="0.33.0", system="b60", kvcache_quant_mode="bfloat16"
        )
        == 78_324
    )
    hidden = dict(common, system="rtx_pro_6000_server", systems_paths=[str(tmp_path)], kvcache_quant_mode="bfloat16")
    with pytest.raises(ValueError, match="sparse MLA.*target GPU"):
        estimate_kv_bytes_per_token(model, **hidden, backend_version="0.33.0")
    assert estimate_kv_bytes_per_token(model, **hidden, backend_version="0.22.0") == 78_324
    raw["index_head_dim"] = 64
    _write_config(tmp_path, {"text_config": raw})
    with pytest.raises(ValueError, match="index_head_dim=128"):
        estimate_kv_bytes_per_token(model, **common, backend="sglang", backend_version="0.5.20", system="h200_sxm")


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize(("name", "layers"), [("nvidia/GLM-5.3-NVFP4", 78), ("deepseek-ai/DeepSeek-V3.2", 61)])
def test_nested_sparse_mla_marginal_includes_indexer_and_bf16_rope(tmp_path, named, name, layers):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.config_builders import build_model_config
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config(name)
    if not named:
        raw.pop("architectures")
    model = _write_config(tmp_path, {"architectures": [_UNKNOWN_WRAPPER], "text_config": raw})
    opts = _missing_system_cache_options(tmp_path, tp_size=16, capacity=192 * _GIB)
    for mode, width in [("fp8", 1), ("bfloat16", 2)]:
        native = get_model(
            name,
            build_model_config(
                tp_size=16, pp_size=1, attention_dp_size=1, moe_tp_size=1, moe_ep_size=16, kvcache_quant_mode=mode
            ),
            "sglang",
        )
        for backend in ("vllm", "sglang"):
            if mode == "fp8":
                with pytest.raises(ValueError, match="runtime latent and indexer cache formats"):
                    estimate_kv_bytes_per_token(model, tp_size=16, pp_size=1, backend=backend, kvcache_quant_mode=mode)
            else:
                marginal = estimate_kv_bytes_per_token(
                    model,
                    tp_size=16,
                    pp_size=1,
                    backend=backend,
                    kvcache_quant_mode=mode,
                    backend_version="0.22.0" if backend == "vllm" else "0.5.16",
                )
                assert marginal == layers * (512 * width + 64 * 2 + 132)
                assert marginal == native.get_kvcache_bytes_per_sequence(257) - native.get_kvcache_bytes_per_sequence(
                    256
                )
        for fn in (memory.estimate_kv_cache, memory.estimate_num_gpu_blocks):
            with pytest.raises(ValueError, match="runtime indexer allocations"):
                fn(model, "missing", "sglang", "0.5.20", **opts, kvcache_quant_mode=mode)


def test_weight_only_sparse_mla_uses_the_default_cache_of_its_serving_gpu(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config, _load_pre_downloaded_hf_quant_config

    raw = _load_pre_downloaded_hf_config("deepseek-ai/DeepSeek-V3.2")
    model = _write_config(tmp_path, {"text_config": raw})
    for system, sm in (("hopper", 90), ("blackwell", 100), ("sm120", 120)):
        (tmp_path / f"{system}.yaml").write_text(f"gpu: {{sm_version: {sm}}}\n")
        for backend, version in (("vllm", "0.33.0"), ("sglang", "0.5.20")):
            args = dict(
                tp_size=8,
                pp_size=1,
                backend=backend,
                backend_version=version,
                system=system,
                systems_paths=[str(tmp_path)],
            )
            if sm == 120 or (backend == "sglang" and sm == 100):
                with pytest.raises(ValueError, match="runtime latent and indexer cache formats"):
                    estimate_kv_bytes_per_token(model, **args)
            else:
                assert estimate_kv_bytes_per_token(model, **args) == 61 * (512 * 2 + 64 * 2 + 132) == 78_324

    name = "nvidia/DeepSeek-V3.2-NVFP4"
    raw = _load_pre_downloaded_hf_config(name)
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(_load_pre_downloaded_hf_quant_config(name)))
    _write_config(tmp_path, {"text_config": raw})
    args = dict(tp_size=8, pp_size=1, backend="vllm", backend_version="0.33.0", systems_paths=[str(tmp_path)])
    for system in ("hopper", "blackwell"):
        assert estimate_kv_bytes_per_token(model, **args, system=system, kvcache_quant_mode="bfloat16") == 78_324
        with pytest.raises(ValueError, match="runtime latent and indexer cache formats"):
            estimate_kv_bytes_per_token(model, **args, system=system)
    with pytest.raises(ValueError, match="runtime latent and indexer cache formats"):
        estimate_kv_bytes_per_token(model, **args, system="sm120", kvcache_quant_mode="bfloat16")


def test_sparse_auto_transfer_with_missing_gpu_uses_only_hardware_independent_defaults(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim.compiler import prediction_to_replay_spec
    from aisim.config import CorePredictionConfig
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("zai-org/GLM-5-FP8")
    model = _write_config(tmp_path, {**raw, "architectures": [_UNKNOWN_WRAPPER], "text_config": raw})
    for target in ({}, {"system": "gb200", "systems_paths": [str(tmp_path)]}):
        args = dict(tp_size=8, pp_size=1, **target)
        assert estimate_kv_bytes_per_token(model, **args, backend_version="0.23.0") == 100_152
        for version in ("0.24.0rc1", "0.33.0", "current"):
            with pytest.raises(ValueError, match="sparse MLA.*target GPU"):
                estimate_kv_bytes_per_token(model, **args, backend_version=version)
        assert estimate_kv_bytes_per_token(model, **args, kvcache_quant_mode="bfloat16") == 100_152
        with pytest.raises(ValueError, match="runtime latent and indexer cache formats"):
            estimate_kv_bytes_per_token(model, **args, kvcache_quant_mode="fp8")

    worker = dict(
        parallelism={"tensor": 8},
        timing={"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
        kv_cache={"capacity": {"type": "fixed", "blocks": 256}},
    )
    engine = dict(
        mode="disaggregated",
        model=model,
        hardware="gb200",
        backend="vllm",
        backend_version="0.33.0",
        systems_paths=[str(tmp_path)],
        workers={"prefill": worker, "decode": worker},
        kv_transfer={"bytes_per_token": "auto", "bandwidth_gb_per_second": 50},
        context_length=4096,
    )
    with pytest.raises(ValueError, match="sparse MLA.*target GPU"):
        prediction_to_replay_spec(CorePredictionConfig.model_validate({"engine": engine}))
    for timing, transfer, controls in (
        (worker["timing"], 100_152, {}),
        ({"type": "default"}, "auto", {"kvcache_quant_mode": "bfloat16"}),
        ({"type": "default", "kvcache_quant_mode": "bfloat16"}, "auto", {}),
    ):
        worker["timing"] = timing
        engine["kv_transfer"]["bytes_per_token"] = transfer
        config = CorePredictionConfig.model_validate({"engine": {**engine, **controls}})
        deployment = prediction_to_replay_spec(config).backend_deployment
        assert deployment.prefill_engine_args["kv_transfer_bytes_per_token"] == 100_152
        assert deployment.decode_engine_args["kv_transfer_bytes_per_token"] == 100_152

    deepseek = _load_pre_downloaded_hf_config("deepseek-ai/DeepSeek-V3.2")
    _write_config(tmp_path, {"text_config": deepseek})
    worker["timing"] = {"type": "default"}
    engine.update(hardware="rtx_pro_6000_server", kvcache_quant_mode="bfloat16")
    with pytest.raises(ValueError, match="sparse MLA.*target GPU"):
        prediction_to_replay_spec(CorePredictionConfig.model_validate({"engine": engine}))
    engine["backend_version"] = "0.23.0"
    deployment = prediction_to_replay_spec(CorePredictionConfig.model_validate({"engine": engine})).backend_deployment
    assert deployment.prefill_engine_args["kv_transfer_bytes_per_token"] == 78_324
    assert deployment.decode_engine_args["kv_transfer_bytes_per_token"] == 78_324

    raw["learnable_sink"] = True
    _write_config(tmp_path, {"text_config": raw})
    args = dict(tp_size=8, pp_size=1, backend="sglang")
    with pytest.raises(ValueError, match="sparse MLA.*target GPU"):
        estimate_kv_bytes_per_token(model, **args, backend_version="0.5.19")
    assert estimate_kv_bytes_per_token(model, **args, backend_version="0.5.20") == 100_152


def test_nested_sparse_index_declaration_requires_a_complete_known_layout(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("deepseek-ai/DeepSeek-V3.2")
    raw.pop("architectures")
    raw.pop("model_type")
    for label, declaration in (
        ("unknown-indexer", {}),
        (
            "missing-width",
            {"layer_types": ["deepseek_sparse_attention"] * raw["num_hidden_layers"], "index_head_dim": 0},
        ),
    ):
        folder = tmp_path / label
        folder.mkdir()
        model = _write_config(folder, {"text_config": {**raw, **declaration}})
        with pytest.raises(ValueError, match="complete supported MLA and indexer geometry"):
            estimate_kv_bytes_per_token(model, tp_size=16, pp_size=1)


def test_homogeneous_pipeline_groups_transfer_only_the_resident_layers(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    gemma = _load_pre_downloaded_hf_config("google/gemma-4-26B-A4B")["text_config"]
    gemma["layer_types"] = ["full_attention"] * 30
    qwen = _load_pre_downloaded_hf_config("Qwen/Qwen3-0.6B")
    qwen.update(
        use_sliding_window=True, sliding_window=1024, max_window_layers=0, layer_types=["sliding_attention"] * 28
    )
    for raw, width in ((gemma, 122_880), (qwen, 114_688)):
        model = _write_config(tmp_path, {"text_config": raw})
        for backend in ("vllm", "sglang"):
            for pp_size in (1, 2):
                assert (
                    estimate_kv_bytes_per_token(model, tp_size=1, pp_size=pp_size, backend=backend) == width // pp_size
                )


def test_serialized_gemma4_uses_the_vllm_release_for_transfer_and_capacity(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("google/gemma-4-26B-A4B")["text_config"]
    model = _write_config(tmp_path, {"text_config": raw})
    opts = _missing_system_cache_options(tmp_path, tp_size=8, capacity=192 * _GIB)
    args = (model, "missing", "vllm", "0.33.0")
    blocks = memory.estimate_num_gpu_blocks(*args, **opts)
    assert blocks > 0

    # Gemma4TextConfig.to_dict moves the global widths into per-layer overrides.
    head_dim = raw.pop("global_head_dim")
    heads = raw.pop("num_global_key_value_heads")
    raw["per_layer_config"] = {
        f"{i:02}": {"head_dim": head_dim, "num_key_value_heads": heads}
        for i, kind in enumerate(raw["layer_types"])
        if kind == "full_attention"
    }
    model = _write_config(tmp_path, {"text_config": raw})
    with pytest.raises(ValueError, match="vLLM release"):
        estimate_kv_bytes_per_token(model, tp_size=8, pp_size=1)
    for version in ("0.28.0", "0.29.0", "0.30.0", "0.31.0"):
        with pytest.raises(ValueError, match="stock collection or a runtime profile"):
            estimate_kv_bytes_per_token(model, tp_size=8, pp_size=1, backend_version=version)
    for version in ("0.32.0", "0.33.0"):
        assert estimate_kv_bytes_per_token(model, tp_size=8, pp_size=1, backend_version=version) == 35_840
    assert memory.estimate_num_gpu_blocks(*args, **opts) == blocks
    assert memory.estimate_kv_cache(*args, **opts)["total_kv_size_tokens"] == blocks * 16


@pytest.mark.parametrize("form", ["flat", "wrapped", "registered", "registered-without-inner-type"])
def test_gemma4_explicit_layout_uses_the_required_final_full_layer(tmp_path, form):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    packaged = _load_pre_downloaded_hf_config("google/gemma-4-26B-A4B")
    raw = {**packaged["text_config"], "layer_types": ["sliding_attention"] * 30}
    if form == "flat":
        config = raw
    elif form == "wrapped":
        config = {"architectures": [_UNKNOWN_WRAPPER], "text_config": raw}
    else:
        text = dict(raw)
        if form == "registered-without-inner-type":
            text.pop("model_type")
        config = {**packaged, "text_config": text}
    model = _write_config(tmp_path, config)
    assert estimate_kv_bytes_per_token(model, tp_size=8, pp_size=1) == 29 * 1024 + 2048 == 31_744
    weights = NaiveKVCacheEstimator.from_hf_config(raw, tp_size=8, pp_size=1, moe_ep_size=8).weight_bytes()
    opts = _missing_system_cache_options(tmp_path, tp_size=8, capacity=weights + (64 << 20))
    args = (model, "missing", "vllm", "0.33.0")
    assert memory.estimate_kv_cache(*args, **opts)["total_kv_size_tokens"] == 68 * 16
    assert memory.estimate_num_gpu_blocks(*args, **opts) == 68


@pytest.mark.parametrize("named", [True, False])
def test_wrapping_preserves_expert_defaults_and_replicated_sglang_prefill_cp(tmp_path, named):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    opts = dict(
        max_num_tokens=4096,
        max_batch_size=32,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        allow_naive_fallback=True,
        systems_path=str(tmp_path),
        gpu_memory_capacity_bytes_override=80 * _GIB,
        dcp_size=1,
        scheduler_block_size=16,
    )
    for model, backend, topology in [
        ("openai/gpt-oss-20b", "vllm", {"tp_size": 2}),
        ("Qwen/Qwen3-0.6B", "sglang", {"cp_size": 2}),
    ]:
        raw = _load_pre_downloaded_hf_config(model)
        if not named:
            raw.pop("architectures")
        (tmp_path / backend).mkdir()
        wrapped = _write_config(tmp_path / backend, {"architectures": [_UNKNOWN_WRAPPER], "text_config": raw})
        for function in (memory.estimate_kv_cache, memory.estimate_num_gpu_blocks):
            native_config = function(model, "missing", backend, **opts, **topology)
            assert function(wrapped, "missing", backend, **opts, **topology) == native_config
            if backend == "sglang":
                assert native_config == function(model, "missing", backend, **opts, cp_size=1)
            else:
                assert native_config != function(model, "missing", backend, **opts, **topology, moe_ep_size=2)
                assert function(wrapped, "missing", backend, **opts, **topology, moe_ep_size=2) == function(
                    model, "missing", backend, **opts, **topology, moe_ep_size=2
                )


@pytest.mark.parametrize(
    "name",
    ["deepseek-ai/DeepSeek-V3", "nvidia/DeepSeek-V3.1-NVFP4", "moonshotai/Kimi-K2.5", "deepseek-ai/DeepSeek-V3.2"],
)
def test_sglang_prefill_cp_replication_requires_the_complete_latent_cache(tmp_path, name):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config(name)
    decoder = raw.get("text_config") or raw.get("llm_config") or raw
    wrapped = _write_config(tmp_path, {"architectures": [_UNKNOWN_WRAPPER], "text_config": decoder})
    opts = dict(
        max_num_tokens=4096,
        max_batch_size=32,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        allow_naive_fallback=True,
        naive_kv_reservation=1,
        systems_path=str(tmp_path),
        gpu_memory_capacity_bytes_override=192 * _GIB,
        cp_size=16 if name == "moonshotai/Kimi-K2.5" else 8,
        dcp_size=1,
        moe_tp_size=1,
        moe_ep_size=16 if name == "moonshotai/Kimi-K2.5" else 8,
        kvcache_quant_mode="bfloat16",
    )
    for model in (name, wrapped):
        args = (model, "missing", "sglang", "0.5.20")
        if name.endswith("V3.2"):
            for function, extra in (
                (memory.estimate_kv_cache, {}),
                (memory.estimate_num_gpu_blocks, {"scheduler_block_size": 16}),
            ):
                with pytest.raises(ValueError, match="context parallelism"):
                    function(*args, **opts, **extra)
        else:
            result = memory.estimate_kv_cache(*args, **opts)
            assert (result["source"], result["kv_size_per_token_bytes"], result["total_kv_size_tokens"]) == (
                "naive_fallback",
                70_272,
                675_056 if name == "moonshotai/Kimi-K2.5" else 77_488,
            )
            assert memory.estimate_num_gpu_blocks(*args, **opts, scheduler_block_size=16) == (
                42_191 if name == "moonshotai/Kimi-K2.5" else 4843
            )


@pytest.mark.parametrize("name", ["Qwen/Qwen3-30B-A3B", "openai/gpt-oss-20b"])
def test_vllm_estimate_prices_the_rank_with_the_most_experts(tmp_path, name):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config(name)
    wrapped = _write_config(tmp_path, {"text_config": raw})
    estimator = NaiveKVCacheEstimator.from_hf_config(raw, tp_size=1, pp_size=1)
    g = estimator.geometry
    experts_per_rank = (g["num_experts"] + 2) // 3
    non_expert = 2 * g["vocab"] * g["hidden"] + g["layers"] * 4 * g["hidden"] ** 2
    expert = g["layers"] * experts_per_rank * 3 * g["hidden"] * g["moe_inter"]
    weights = int((non_expert + expert) * estimator.weight_dtype_bytes)
    opts = _missing_system_cache_options(tmp_path, tp_size=1, capacity=weights + (64 << 20))
    opts.update(attention_dp_size=3, moe_ep_size=3)
    for checkpoint in (name, wrapped):
        args = (checkpoint, "missing", "vllm", "0.33.0")
        result = memory.estimate_kv_cache(*args, **opts)
        assert result["total_kv_size_bytes"] == 64 << 20
        assert memory.estimate_num_gpu_blocks(*args, **opts) == result["total_kv_size_tokens"] // 16


@pytest.mark.parametrize("mode", [None, "fp8_block"])
@pytest.mark.parametrize("named", [True, False])
@pytest.mark.parametrize(
    ("name", "tp"),
    [("stepfun-ai/Step-3.7-Flash-FP8", 4), ("Qwen/Qwen3-30B-A3B-FP8", 8), ("Qwen/Qwen3-235B-A22B-FP8", 8)],
)
def test_wrapped_fp8_moe_shards_follow_the_serving_loader(tmp_path, name, tp, named, mode):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config(name)
    if not named:
        raw.pop("architectures")
    model = _write_config(tmp_path, {"text_config": raw})
    opts = _missing_system_cache_options(tmp_path, tp_size=tp, capacity=192 * _GIB)
    for backend, version in (("vllm", "0.33.0"), ("sglang", "0.5.20")):
        args = (model, "missing", backend, version)
        for fn in (memory.estimate_kv_cache, memory.estimate_num_gpu_blocks):
            sharded = opts | {"moe_tp_size": tp, "moe_ep_size": 1}
            if backend == "sglang":
                with pytest.raises(ValueError, match="weight_block_size=128"):
                    fn(*args, **sharded, moe_quant_mode=mode)
            else:
                assert fn(*args, **sharded, moe_quant_mode=mode)
                with pytest.raises(ValueError, match="weight_block_size=128"):
                    fn(model, "missing", "vllm", "0.28.0", **sharded, moe_quant_mode=mode)
            valid = opts | {"moe_tp_size": 2, "moe_ep_size": tp // 2}
            if name.startswith("stepfun") and backend == "sglang":
                with pytest.raises(ValueError, match="separate full and sliding pool sizes"):
                    fn(*args, **valid, moe_quant_mode=mode)
            else:
                assert fn(*args, **valid, moe_quant_mode=mode)


@pytest.mark.parametrize(
    ("name", "shape", "tp", "reason"),
    [
        ("deepseek-ai/DeepSeek-V3", [128, 128], 32, "weight_block_size=128"),
        ("Qwen/Qwen3-30B-A3B-FP8", [32, 128], 8, "weight_block_size=128"),
        ("Qwen/Qwen3-30B-A3B-FP8", [128], 2, "two positive integer"),
        ("Qwen/Qwen3-30B-A3B-FP8", [32, 128], 2, None),
    ],
)
def test_sglang_fp8_experts_must_fit_both_axes_of_the_declared_tile(tmp_path, name, shape, tp, reason):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config(name)
    raw.pop("architectures")
    raw["quantization_config"]["weight_block_size"] = shape
    model = _write_config(tmp_path, {"text_config": raw})
    opts = _missing_system_cache_options(tmp_path, tp_size=tp, capacity=192 * _GIB)
    opts.update(moe_tp_size=tp, moe_ep_size=1)
    for fn in (memory.estimate_kv_cache, memory.estimate_num_gpu_blocks):
        if reason:
            with pytest.raises(ValueError, match=reason):
                fn(model, "missing", "sglang", "0.5.20", **opts)
        else:
            assert fn(model, "missing", "sglang", "0.5.20", **opts)


@pytest.mark.parametrize(
    ("tp", "shape", "excluded", "error"),
    [
        (8, [128, 128], [], None),
        (16, [128, 128], [], "mlp.gate_up_proj.*1600"),
        (16, [64, 128], [], "mlp.down_proj.*1600"),
        (16, [128, 128], ["mlp"], None),
        (16, [128, 128], ["model.layers.0.mlp"], "mlp.gate_up_proj.*1600"),
        (8, [256, 128], ["mlp"], "self_attn.qkv_proj"),
        (8, [256, 128], ["mlp", "q_proj", "k_proj"], "all shards"),
        (8, [256, 128], ["mlp", "q_proj", "k_proj", "v_proj"], None),
        (8, [128, 128], ["language_model.model.layers.62.self_attn.q_proj"], "all shards"),
        (8, [128], [], "two positive integer block dimensions"),
    ],
)
def test_sglang_dense_fp8_linear_shards_fit_the_serialized_blocks(tmp_path, tp, shape, excluded, error):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    text = _load_pre_downloaded_hf_config("Qwen/Qwen3-32B-FP8")
    text["quantization_config"].update(weight_block_size=shape, modules_to_not_convert=excluded)
    model = _write_config(tmp_path, {"architectures": ["LlavaForConditionalGeneration"], "text_config": text})
    opts = _missing_system_cache_options(tmp_path, tp_size=tp, capacity=80 * _GIB)
    for fn in (memory.estimate_kv_cache, memory.estimate_num_gpu_blocks):
        if error:
            with pytest.raises(ValueError, match=error):
                fn(model, "missing", "sglang", "0.5.20", **opts)
        else:
            assert fn(model, "missing", "sglang", "0.5.20", **opts)


def test_sglang_fp8_checks_the_dense_layers_in_a_routed_model(tmp_path):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("deepseek-ai/DeepSeek-V3")
    opts = _missing_system_cache_options(tmp_path, tp_size=32, capacity=80 * _GIB)
    opts.update(moe_tp_size=1, moe_ep_size=32)
    for prefix, frequency, supported in ((3, 1, False), (0, 1, True), (0, 2, False)):
        raw.update(first_k_dense_replace=prefix, moe_layer_freq=frequency)
        model = _write_config(tmp_path, {"text_config": raw})
        for fn in (memory.estimate_kv_cache, memory.estimate_num_gpu_blocks):
            if supported:
                assert fn(model, "missing", "sglang", "0.5.20", **opts)
            else:
                with pytest.raises(ValueError, match=r"mlp.gate_up_proj.*576"):
                    fn(model, "missing", "sglang", "0.5.20", **opts)


def test_sglang_mla_prefill_cp_replicates_the_dense_mlp(tmp_path):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("deepseek-ai/DeepSeek-V3")
    model = _write_config(tmp_path, {"text_config": raw})
    (tmp_path / "hopper.yaml").write_text("gpu: {sm_version: 90}\n")
    opts = _missing_system_cache_options(tmp_path, tp_size=1, capacity=80 * _GIB)
    opts.update(moe_ep_size=32, cp_size=32, memory_fraction_value=1, max_num_tokens=2560, max_batch_size=1)
    del opts["naive_kv_reservation"]
    for release in ("0.5.13", "0.5.20"):
        assert memory.estimate_num_gpu_blocks(model, "hopper", "sglang", release, **opts) == 10_057
    with pytest.raises(ValueError, match=r"mlp.gate_up_proj.*576"):
        memory.estimate_num_gpu_blocks(model, "hopper", "sglang", "0.5.12.post1", **opts)


def test_sglang_fp8_uses_the_release_attention_group_for_output_shards(tmp_path):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("Qwen/Qwen3-0.6B")
    raw["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 256],
        "modules_to_not_convert": ["mlp"],
    }
    model = _write_config(tmp_path, {"architectures": ["LlavaForConditionalGeneration"], "text_config": raw})
    opts = _missing_system_cache_options(tmp_path, tp_size=16, capacity=80 * _GIB)
    opts.update(attention_dp_size=2, moe_tp_size=32)
    assert memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.21", **opts)
    for version in ("0.5.22", "0.5.23"):
        with pytest.raises(ValueError, match=r"self_attn.o_proj.*128"):
            memory.estimate_num_gpu_blocks(model, "missing", "sglang", version, **opts)


def test_sglang_fp8_uses_model_packed_projections_and_versioned_exclusions(tmp_path):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("Qwen/Qwen3-32B-FP8")
    wrapper = {"architectures": ["LlavaForConditionalGeneration"], "text_config": raw}
    quant = raw["quantization_config"]
    quant.update(
        modules_to_not_convert=["q_proj", "kv_proj"], packed_modules_mapping={"qkv_proj": ["q_proj", "kv_proj"]}
    )
    opts = _missing_system_cache_options(tmp_path, tp_size=8, capacity=80 * _GIB)
    with pytest.raises(ValueError, match="same FP8 precision"):
        memory.estimate_num_gpu_blocks(_write_config(tmp_path, wrapper), "missing", "sglang", "0.5.20", **opts)
    quant["modules_to_not_convert"] = ["q_proj", "k_proj", "v_proj"]
    assert memory.estimate_num_gpu_blocks(_write_config(tmp_path, wrapper), "missing", "sglang", "0.5.20", **opts)

    opts = _missing_system_cache_options(tmp_path, tp_size=16, capacity=80 * _GIB)
    quant["modules_to_not_convert"] = ["model.language_model"]
    model = _write_config(tmp_path, wrapper)
    with pytest.raises(ValueError, match=r"mlp.gate_up_proj.*1600"):
        memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.9", **opts)
    assert memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.10rc0", **opts)
    quant["modules_to_not_convert"] = ["language_model"]
    assert memory.estimate_num_gpu_blocks(_write_config(tmp_path, wrapper), "missing", "sglang", "0.5.9", **opts)


def test_sglang_mxfp8_experts_use_the_fixed_checkpoint_block(tmp_path):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("Qwen/Qwen3-30B-A3B-FP8")
    raw["quantization_config"].update(quant_method="mxfp8", kv_cache_quant_algo="FP8")
    model = _write_config(tmp_path, {"text_config": raw})
    opts = _missing_system_cache_options(tmp_path, tp_size=2, capacity=192 * _GIB)
    (tmp_path / "blackwell.yaml").write_text("gpu: {sm_version: 100}\n")
    result = memory.estimate_kv_cache(model, "blackwell", "sglang", "0.5.20", **opts)
    assert result["kv_size_per_token_bytes"] == 24_576
    assert (
        memory.estimate_num_gpu_blocks(model, "blackwell", "sglang", "0.5.20", **opts)
        == result["total_kv_size_tokens"] // 16
    )

    with pytest.raises(ValueError, match="backend version 0.5.9"):
        memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.8.post1", **opts)

    with pytest.raises(ValueError, match="MXFP8.*require a Blackwell"):
        memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.20", **opts)
    (tmp_path / "hopper.yaml").write_text("gpu: {sm_version: 90}\n")
    with pytest.raises(ValueError, match="MXFP8.*require a Blackwell"):
        memory.estimate_kv_cache(model, "hopper", "sglang", "0.5.20", **opts)
    (tmp_path / "sm120.yaml").write_text("gpu: {sm_version: 120}\n")
    with pytest.raises(ValueError, match="SM100 family"):
        memory.estimate_num_gpu_blocks(model, "sm120", "sglang", "0.5.20", **opts)

    raw["architectures"] = ["Qwen3ForCausalLM"]
    _write_config(tmp_path, {"text_config": raw})
    with pytest.raises(ValueError, match="qwen3_moe.*conflicts.*Qwen3ForCausalLM"):
        memory.estimate_num_gpu_blocks(model, "blackwell", "sglang", "0.5.20", **opts)


@pytest.mark.parametrize("estimate", [memory.estimate_kv_cache, memory.estimate_num_gpu_blocks])
def test_sglang_mxfp8_gpu_requirements_follow_expert_exclusions(tmp_path, estimate):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    # InternVL 3.5 uses the BF16 Qwen3-30B-A3B decoder under llm_config.
    text = _load_pre_downloaded_hf_config("Qwen/Qwen3-30B-A3B")
    wrapper = {"architectures": ["InternVLChatModel"], "llm_config": text}
    model = _write_config(tmp_path, wrapper)
    opts = _missing_system_cache_options(tmp_path, tp_size=2, capacity=80 * _GIB)
    opts.update(moe_tp_size=2, moe_ep_size=1)
    for name, sm in (("hopper", 90), ("sm100", 100), ("sm120", 120)):
        (tmp_path / f"{name}.yaml").write_text(f"gpu: {{sm_version: {sm}}}\n")
    plain = estimate(model, "sm120", "sglang", "0.5.20", **opts)

    quant = {
        "quant_method": "mxfp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [1, 32],
        "modules_to_not_convert": ["model", "lm_head"],
    }
    _write_config(tmp_path, dict(wrapper, quantization_config=quant))
    assert estimate(model, "sm120", "sglang", "0.5.20", **opts) == plain
    with pytest.raises(ValueError, match="SM100"):
        estimate(model, "hopper", "sglang", "0.5.20", **opts)
    with pytest.raises(ValueError, match="backend version 0.5.9"):
        estimate(model, "sm120", "sglang", "0.5.8.post1", **opts)
    with pytest.raises(ValueError, match="SM100 family"):
        estimate(model, "sm120", "sglang", "0.5.9", **opts)  # This release always quantizes FusedMoE.

    quant.pop("weight_block_size")
    quant["activation_scheme"] = "static"
    _write_config(tmp_path, dict(wrapper, quantization_config=quant))
    assert estimate(model, "sm120", "sglang", "0.5.15.post1", **opts) == plain
    with pytest.raises(ValueError, match="dynamic activations"):
        estimate(model, "sm120", "sglang", "0.5.16", **opts)
    quant["activation_scheme"] = "dynamic"
    quant["modules_to_not_convert"] = ["model.layers.47", "lm_head"]
    _write_config(tmp_path, dict(wrapper, quantization_config=quant))
    for mode in (None, "bfloat16", "fp8"):
        with pytest.raises(ValueError, match="SM100 family"):
            estimate(model, "sm120", "sglang", "0.5.20", **opts, moe_quant_mode=mode)
    assert estimate(model, "sm100", "sglang", "0.5.20", **opts)

    mixtral = {"text_config": _load_pre_downloaded_hf_config("mistralai/Mixtral-8x7B-v0.1")}
    _write_config(tmp_path, mixtral)
    plain = estimate(model, "sm120", "sglang", "0.5.20", **opts)
    quant["modules_to_not_convert"] = ["model", "lm_head"]
    _write_config(tmp_path, dict(mixtral, quantization_config=quant))
    with pytest.raises(ValueError, match="SM100 family"):
        estimate(model, "sm120", "sglang", "0.5.20", **opts)
    quant["modules_to_not_convert"].append("layers")
    _write_config(tmp_path, dict(mixtral, quantization_config=quant))
    assert estimate(model, "sm120", "sglang", "0.5.20", **opts) == plain

    dense = {"text_config": _load_pre_downloaded_hf_config("Qwen/Qwen3-0.6B"), "quantization_config": quant}
    _write_config(tmp_path, dense)
    with pytest.raises(ValueError, match="SM100"):
        estimate(model, "hopper", "sglang", "0.5.20", **opts)
    assert estimate(model, "sm100", "sglang", "0.5.20", **opts)


@pytest.mark.parametrize("named", [True, False])
@pytest.mark.parametrize(
    "source", ["sidecar", "unlabelled_sidecar", "root_inline", "text_inline", "weight_only", "unquantized"]
)
def test_wrapped_checkpoint_kv_precision_prices_transfers_and_blocks(tmp_path, named, source):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim.compiler import prediction_to_replay_spec
    from aisim.config import CorePredictionConfig
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config, _load_pre_downloaded_hf_quant_config

    name = "nvidia/Llama-3.1-70B-Instruct-FP8"
    raw = _load_pre_downloaded_hf_config(name)
    sidecar = _load_pre_downloaded_hf_quant_config(name)
    if not named:
        raw.pop("architectures")
    wrapper = {"text_config": raw}
    if source in ("root_inline", "text_inline"):
        owner = raw if source == "text_inline" else wrapper
        owner["quantization_config"] = {
            "quant_method": "modelopt",
            "quant_algo": "FP8",
            "kv_cache_scheme": {"num_bits": 8, "type": "float", "dynamic": False},
        }
    elif source == "weight_only":
        raw["quantization_config"] = {"quant_method": "fp8", "activation_scheme": "dynamic"}
    else:
        if source == "unquantized":
            sidecar["quantization"]["kv_cache_quant_algo"] = None
        elif source == "unlabelled_sidecar":
            sidecar.pop("producer")
        (tmp_path / "hf_quant_config.json").write_text(json.dumps(sidecar))
    model = _write_config(tmp_path, wrapper)
    # The unregistered wrapper supplies raw BF16 geometry and no native vLLM decoder class.
    weights = 2 * (2 * 128256 * 8192 + 80 * (4 * 8192**2 + 3 * 8192 * 28672)) // 8
    budget = 16 * 64 * 40_960
    opts = _missing_system_cache_options(tmp_path, tp_size=8, capacity=weights + budget)
    for backend, version in (("vllm", "0.33.0"), ("sglang", "0.5.20")):
        worker = dict(
            parallelism={"tensor": 8},
            timing={"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
            kv_cache={"capacity": {"type": "fixed", "blocks": 256}},
        )
        engine = dict(
            mode="disaggregated",
            model=model,
            hardware="gb300",
            backend=backend,
            context_length=4096,
            workers={"prefill": worker, "decode": worker},
            kv_transfer={"bytes_per_token": "auto", "bandwidth_gb_per_second": 50},
        )
        # At TP8 each rank holds one KV head: two 128-element arrays in each of 80 layers.
        fp8 = source in ("root_inline", "sidecar") or (
            source in ("text_inline", "unlabelled_sidecar") and backend == "sglang"
        )
        automatic = 20_480 if fp8 else 40_960
        deployment = prediction_to_replay_spec(
            CorePredictionConfig.model_validate({"engine": engine})
        ).backend_deployment
        assert deployment.prefill_engine_args["kv_transfer_bytes_per_token"] == automatic
        assert deployment.decode_engine_args["kv_transfer_bytes_per_token"] == automatic
        for override, width in ((None, automatic), ("fp8", 20_480), ("bfloat16", 40_960)):
            assert (
                estimate_kv_bytes_per_token(model, tp_size=8, pp_size=1, backend=backend, kvcache_quant_mode=override)
                == width
            )
            args = (model, "missing", backend, version)
            result = memory.estimate_kv_cache(*args, **opts, kvcache_quant_mode=override)
            blocks = budget // (16 * width) - int(backend == "vllm")
            assert (result["kv_size_per_token_bytes"], result["total_kv_size_tokens"]) == (width, blocks * 16)
            assert memory.estimate_num_gpu_blocks(*args, **opts, kvcache_quant_mode=override) == blocks


@pytest.mark.parametrize(
    ("method", "backend", "version", "width"),
    [
        ("compressed-tensors", "vllm", "0.14.0", 16_384),
        ("compressed-tensors", "vllm", "0.21.0", 16_384),
        ("compressed-tensors", "vllm", "0.22.0", 8_192),
        ("compressed-tensors", "sglang", "0.5.10", 16_384),
        ("compressed-tensors", "sglang", "0.5.18", 16_384),
        ("compressed-tensors", "sglang", "0.5.19", 8_192),
        ("modelopt", "vllm", "0.19.1", 16_384),
        ("modelopt", "vllm", "0.20.0", 4_608),
    ],
)
def test_versioned_checkpoint_cache_reaches_both_compiled_workers_and_block_capacity(
    tmp_path, method, backend, version, width
):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim.compiler import prediction_to_replay_spec
    from aisim.config import CorePredictionConfig
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("meta-llama/Meta-Llama-3.1-8B")
    raw["quantization_config"] = {
        "quant_method": method,
        "kv_cache_scheme": {
            "num_bits": 4 if method == "modelopt" else 8,
            "type": "float",
            "strategy": "tensor",
            "symmetric": True,
            "dynamic": False,
        },
    }
    model = _write_config(tmp_path, {**raw, "architectures": [_UNKNOWN_WRAPPER], "text_config": raw})
    worker = dict(
        parallelism={"tensor": 8},
        timing={"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
        kv_cache={"capacity": {"type": "fixed", "blocks": 256}},
    )
    engine = dict(
        mode="disaggregated",
        model=model,
        hardware="gb200",
        backend=backend,
        backend_version=version,
        context_length=4096,
        workers={"prefill": worker, "decode": worker},
        kv_transfer={"bytes_per_token": "auto", "bandwidth_gb_per_second": 50},
    )
    ranks = prediction_to_replay_spec(CorePredictionConfig.model_validate({"engine": engine})).backend_deployment
    assert (
        ranks.prefill_engine_args["kv_transfer_bytes_per_token"]
        == ranks.decode_engine_args["kv_transfer_bytes_per_token"]
        == width
    )
    # TP8 stores one KV head per rank, 128 elements per head in 32 layers. NVFP4 uses 9 bytes per 16 elements.
    assert estimate_kv_bytes_per_token(model, tp_size=8, pp_size=1, backend=backend, backend_version=version) == width
    assert (
        estimate_kv_bytes_per_token(
            model, tp_size=8, pp_size=1, backend=backend, backend_version=version, kvcache_quant_mode="bfloat16"
        )
        == 16_384
    )
    weights = NaiveKVCacheEstimator.from_hf_config(raw, tp_size=8, pp_size=1).weight_bytes()
    budget = 16 << 20
    opts = _missing_system_cache_options(tmp_path, tp_size=8, capacity=weights + budget)
    result = memory.estimate_kv_cache(model, "missing", backend, version, **opts)
    assert result["kv_size_per_token_bytes"] == width
    assert memory.estimate_num_gpu_blocks(model, "missing", backend, version, **opts) == budget // (16 * width) - (
        backend == "vllm"
    )


@pytest.mark.parametrize("metadata_field", ["quantization_config", "compression_config"])
def test_auto_cache_precision_uses_the_hardware_version_slot(tmp_path, monkeypatch, metadata_field):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk import perf_database
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("meta-llama/Meta-Llama-3.1-8B")
    model = _write_config(
        tmp_path,
        {
            "text_config": raw,
            metadata_field: {
                "quant_method": "compressed-tensors",
                "kv_cache_scheme": {"type": "float", "num_bits": 8, "strategy": "tensor", "symmetric": True},
            },
        },
    )

    def slots(system, backend, systems_paths):
        assert (system, backend, systems_paths) == ("custom", "vllm", [str(tmp_path)])
        return {"current": "0.22.0", "previous": "0.20.0"}

    monkeypatch.setattr(perf_database, "get_version_slots", slots)
    for selector, width in ((None, 8_192), ("current", 8_192), ("previous", 16_384), ("next", 16_384)):
        assert (
            estimate_kv_bytes_per_token(
                model,
                tp_size=8,
                pp_size=1,
                backend="vllm",
                backend_version=selector,
                system="custom",
                systems_paths=[str(tmp_path)],
            )
            == width
        )


def test_gemma3_implicit_layer_pattern_matches_its_explicit_physical_pool(tmp_path):
    raw = dict(
        model_type="gemma3_text",
        hidden_size=2560,
        intermediate_size=10240,
        num_hidden_layers=34,
        num_attention_heads=8,
        num_key_value_heads=4,
        vocab_size=262208,
        sliding_window=1024,
        dtype="bfloat16",
        max_position_embeddings=131072,
    )
    # Transformers supplies a 256-element head and one global layer per six layers, totaling five global layers.
    budget = 101 * 34 * 65_536
    serialized = {
        **raw,
        "head_dim": 256,
        "layer_types": ["full_attention" if i % 6 == 0 else "sliding_attention" for i in range(1, 35)],
    }
    model = _write_config(tmp_path, {**serialized, "architectures": ["Gemma3ForCausalLM"]})
    estimator = NaiveKVCacheEstimator.from_model_path(model, tp_size=1, pp_size=1, allow_hf_config_download=False)
    for text in (raw, serialized):
        wrapped = {"architectures": ["Gemma3ForConditionalGeneration"], "text_config": text}
        combined = NaiveKVCacheEstimator.from_hf_config(wrapped, tp_size=1, pp_size=1)
        with pytest.raises(ValueError, match="resident weights for the vision tower"):
            combined.weight_bytes()
        assert combined.kv_bytes_per_token() == estimator.kv_bytes_per_token()
        assert (combined.geometry["num_swa_layers"], combined.geometry["num_global_layers"]) == (29, 5)
        sglang = NaiveKVCacheEstimator.from_hf_config(
            wrapped, tp_size=1, pp_size=1, backend="sglang", backend_version="0.5.23"
        )
        with pytest.raises(ValueError, match="resident weights for the vision tower"):
            sglang.weight_bytes()
    opts = _missing_system_cache_options(tmp_path, tp_size=1, capacity=estimator.weight_bytes() + budget)
    assert memory.estimate_num_gpu_blocks(model, "missing", "vllm", "0.33.0", **opts) == 97
    with pytest.raises(ValueError, match="separate full and sliding pool sizes"):
        memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.20", **opts)


@pytest.mark.parametrize("omit_defaults", [False, True])
@pytest.mark.parametrize("omit_class", [False, True])
def test_wrapped_decoder_capacity_uses_the_layers_on_each_pipeline_rank(tmp_path, omit_defaults, omit_class):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("Qwen/Qwen3-0.6B")
    if omit_defaults:
        raw.pop("head_dim")  # Qwen3Config supplies 128, while hidden_size // num_attention_heads is 64.
    if omit_class:
        raw.pop("architectures")
    model = _write_config(tmp_path, {"text_config": raw})
    opts = _missing_system_cache_options(tmp_path, tp_size=1, capacity=80 * _GIB)
    for pp, width in ((1, 114_688), (2, 57_344), (4, 28_672)):
        assert estimate_kv_bytes_per_token(model, tp_size=1, pp_size=pp) == width
        result = memory.estimate_kv_cache(model, "missing", "sglang", "0.5.20", **opts, pp_size=pp)
        assert result["kv_size_per_token_bytes"] == width
    # Stage zero owns the entire vocabulary and seven layers. These matrices alone require 531,365,888 bytes.
    budget = 700 << 20
    opts["gpu_memory_capacity_bytes_override"] = budget
    blocks = memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.20", **opts, pp_size=4)
    assert 0 < blocks <= (budget - 531_365_888) // (16 * 28_672) == 441
    qwen_moe = _load_pre_downloaded_hf_config("Qwen/Qwen3-30B-A3B")
    _write_config(tmp_path, {"text_config": qwen_moe})
    with pytest.raises(ValueError, match="resident weights on each stage"):
        memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.20", **opts, pp_size=4)
    _write_config(tmp_path, {"text_config": raw})
    with pytest.raises(ValueError, match="resident attention layers"):
        estimate_kv_bytes_per_token(model, tp_size=1, pp_size=3)


@pytest.mark.parametrize("remove_class", [False, True])
def test_wrapped_block_quantization_and_expert_tensor_shards_follow_the_width(tmp_path, remove_class):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("Qwen/Qwen3-30B-A3B-FP8")
    if remove_class:
        raw.pop("architectures")
        raw["model_type"] = "unregistered_sparse_decoder"
    model = _write_config(tmp_path, {"text_config": raw})
    for function in (memory.estimate_kv_cache, memory.estimate_num_gpu_blocks):
        for ranks, mode, reason in (
            (5, "bfloat16", "divisible by moe_tp_size"),
            (8, "fp8_block", None),  # The 96-wide shard supports vLLM's 32x32 scale refinement.
            (32, "fp8_block", "weight_block_size=128"),
            (3, "bfloat16", None),
        ):
            opts = _missing_system_cache_options(tmp_path, tp_size=1, capacity=192 * _GIB)
            opts.update(attention_dp_size=ranks, moe_ep_size=1, moe_tp_size=ranks, moe_quant_mode=mode)
            if reason:
                with pytest.raises(ValueError, match=reason):
                    function(model, "missing", "vllm", "0.33.0", **opts)
            else:
                assert function(model, "missing", "vllm", "0.33.0", **opts)


def test_nested_state_and_draft_cache_need_physical_reservations(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    inkling = {
        **_UNKNOWN_DECODER,
        "use_sconv": True,
        "sconv_kernel_size": 4,
        "local_layer_ids": [0, 1, 2],
        "sliding_window_size": 128,
        "swa_num_key_value_heads": 4,
    }
    model = _write_config(tmp_path, {"text_config": inkling})
    with pytest.raises(ValueError, match="state reservations"):
        estimate_kv_bytes_per_token(model, tp_size=1, pp_size=1)
    opts = _missing_system_cache_options(tmp_path, tp_size=1, capacity=80 * _GIB)
    for function in (memory.estimate_kv_cache, memory.estimate_num_gpu_blocks):
        with pytest.raises(ValueError, match="state reservations"):
            function(model, "missing", "vllm", "0.33.0", **opts)

    raw = _load_pre_downloaded_hf_config("deepseek-ai/DeepSeek-V3")
    model = _write_config(tmp_path, {"text_config": raw})
    opts = _missing_system_cache_options(tmp_path, tp_size=8, capacity=192 * _GIB)
    for function in (memory.estimate_kv_cache, memory.estimate_num_gpu_blocks):
        assert function(model, "missing", "sglang", "0.5.20", **opts, nextn=0)
        with pytest.raises(ValueError, match="runtime draft weights and cache"):
            function(model, "missing", "sglang", "0.5.20", **opts, nextn=1)


def test_gpt_oss_implicit_and_explicit_windows_have_the_same_shared_capacity(tmp_path):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("openai/gpt-oss-20b")
    weights = NaiveKVCacheEstimator.from_hf_config(raw, tp_size=1, pp_size=1).weight_bytes()
    opts = _missing_system_cache_options(tmp_path, tp_size=1, capacity=weights + (96 << 20) + (384 << 10))
    for types in (raw["layer_types"], None):
        if types is None:
            raw.pop("layer_types")
            raw.pop("sliding_window")
            raw.pop("architectures")
        model = _write_config(tmp_path, {"text_config": raw})
        assert memory.estimate_num_gpu_blocks(model, "missing", "vllm", "0.33.0", **opts) == 127
        with pytest.raises(ValueError, match="separate full and sliding pool sizes"):
            memory.estimate_num_gpu_blocks(model, "missing", "sglang", "0.5.20", **opts)


def test_naive_sizing_with_an_offline_catalog_uses_the_unknown_version_bound(tmp_path, monkeypatch):
    from aisim_core.data.ensure import MissingDataError
    from aisim_core.sdk import perf_database

    def unavailable(*args, **kwargs):
        raise MissingDataError("offline catalog is unavailable")

    monkeypatch.setattr(perf_database, "get_version_slots", unavailable)
    opts = _missing_system_cache_options(tmp_path, tp_size=1, capacity=80 * _GIB)
    for function in (memory.estimate_kv_cache, memory.estimate_num_gpu_blocks):
        for version in (None, "current", "previous", "next"):
            assert function("Qwen/Qwen3-0.6B", "missing", "vllm", version, **opts)


@pytest.mark.parametrize("method", ["fp8", "mxfp8"])
def test_sglang_native_fp8_needs_a_separate_kv_declaration(tmp_path, method):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("Qwen/Qwen3-30B-A3B-FP8")
    quant = raw.pop("quantization_config") | {"quant_method": method}
    for placement in ("root", "text_config"):
        document = {"text_config": dict(raw)}
        owner = document if placement == "root" else document["text_config"]
        for algo, rate in ((None, 24_576), ("FP8", 12_288)):
            owner["quantization_config"] = {**quant, "kv_cache_quant_algo": algo}
            model = _write_config(tmp_path, document)
            for version, expected in (("0.5.17", 24_576), ("0.5.18", rate), ("0.5.20", rate), (None, 24_576)):
                assert (
                    estimate_kv_bytes_per_token(model, tp_size=8, pp_size=1, backend="sglang", backend_version=version)
                    == expected
                )
            assert estimate_kv_bytes_per_token(model, tp_size=8, pp_size=1, backend="vllm") == 24_576


@pytest.mark.parametrize("placement", ["text_config", "root", "sidecar_only"])
def test_static_qwen_modelopt_prefers_each_backends_runtime_metadata(tmp_path, placement):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config, _load_pre_downloaded_hf_quant_config

    name = "Qwen/Qwen3-32B-FP8-Static-PerTensor"
    raw = _load_pre_downloaded_hf_config(name)
    inline = raw.pop("quantization_config")
    wrapper = {"text_config": raw}
    if placement != "sidecar_only":
        (raw if placement == "text_config" else wrapper)["quantization_config"] = inline
    model = _write_config(tmp_path, wrapper)
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(_load_pre_downloaded_hf_quant_config(name)))
    for backend in ("vllm", "sglang"):
        # Both legacy readers use FP8. The inline string lacks vLLM's quant_method and SGLang's kv_cache_quant_algo.
        fp8 = placement == "sidecar_only" or (backend == "vllm" and placement == "text_config")
        assert estimate_kv_bytes_per_token(model, tp_size=8, pp_size=1, backend=backend) == (16_384 if fp8 else 32_768)
        for override, expected in (("fp8", 16_384), ("bfloat16", 32_768)):
            assert (
                estimate_kv_bytes_per_token(model, tp_size=8, pp_size=1, backend=backend, kvcache_quant_mode=override)
                == expected
            )


def test_fixed_capacity_pd_uses_the_prefill_backends_sharded_kv_geometry(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim.compiler import prediction_to_replay_spec
    from aisim.config import CorePredictionConfig
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    raw = _load_pre_downloaded_hf_config("stepfun-ai/Step-3.7-Flash")
    raw["attention_other_setting"]["num_attention_groups"] = raw["swa_num_key_value_heads"] = 16
    checkpoint = _write_config(tmp_path, {"text_config": raw})
    worker = dict(
        parallelism={"tensor": 1},
        timing={"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
        kv_cache={"capacity": {"type": "fixed", "blocks": 256}},
    )
    for backend, width in (("sglang", 319_488), ("vllm", 184_320)):
        engine = dict(
            mode="disaggregated",
            model=checkpoint,
            hardware="gb300",
            backend=backend,
            context_length=4096,
            workers={"prefill": worker, "decode": worker},
            kv_transfer={"bytes_per_token": "auto", "bandwidth_gb_per_second": 50},
        )
        deployment = prediction_to_replay_spec(
            CorePredictionConfig.model_validate({"engine": engine})
        ).backend_deployment
        assert estimate_kv_bytes_per_token(checkpoint, tp_size=1, pp_size=1, backend=backend) == width
        assert deployment.prefill_engine_args["kv_transfer_bytes_per_token"] == width
        assert deployment.decode_engine_args["kv_transfer_bytes_per_token"] == width


def test_wrapped_marginal_rejects_an_impossible_attention_partition(tmp_path):
    from aisim.capacity import estimate_kv_bytes_per_token
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    model = _write_config(tmp_path, {"text_config": _load_pre_downloaded_hf_config("openai/gpt-oss-20b")})
    for checkpoint in ("openai/gpt-oss-20b", model):
        with pytest.raises(ValueError, match="num_attention_heads 64 must be divisible by tp_size 3"):
            estimate_kv_bytes_per_token(checkpoint, tp_size=3, pp_size=1)
        assert estimate_kv_bytes_per_token(checkpoint, tp_size=2, pp_size=1) > 0


def test_fallback_preserves_the_known_systems_version_slots(tmp_path, monkeypatch):
    from aisim_core.sdk import perf_database

    monkeypatch.delenv("AIC_ALLOW_UNLISTED_VERSIONS", raising=False)
    (tmp_path / "a100_sxm.yaml").write_text("data_dir: data/a100_sxm\n")
    (tmp_path / "data/a100_sxm/vllm/0.14.0").mkdir(parents=True)
    (tmp_path / "query_versions.yaml").write_text(
        "schema_version: 1\noverrides: {a100_sxm: {vllm: {current: '0.14.0'}}}\n"
    )
    options = dict(
        systems_path=str(tmp_path),
        max_batch_size=2,
        max_num_tokens=4096,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        gpu_memory_capacity_bytes_override=60 * _GIB,
        allow_naive_fallback=True,
    )
    model = "openai/gpt-oss-20b"
    for function, extra in (
        (memory.estimate_kv_cache, {}),
        (memory.estimate_num_gpu_blocks, {"scheduler_block_size": 16}),
    ):
        for version, message in [("0.1.0", "not a queryable version"), ("next", "has no 'next' version")]:
            with pytest.raises(perf_database.UnlistedQueryVersionError, match=message):
                function(model, "a100_sxm", "vllm", version, **options, **extra)
        # The accepted alias and literal still permit an incomplete hardware specification to use the override.
        assert function(model, "a100_sxm", "vllm", "current", **options, **extra) == function(
            model, "a100_sxm", "vllm", "0.14.0", **options, **extra
        )


@pytest.mark.parametrize("wrapped", [True, False])
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"layer_types": [{"kind": "sliding_attention"}] * 24}, "layer_types entries must be nonempty strings"),
        ({"layer_types": [""] * 24}, "layer_types entries must be nonempty strings"),
        ({"layer_types": ["custom_sliding"] * 24}, "layer_types must contain only"),
        ({"use_sliding_window": False}, "conflicts with sliding attention layers"),
        ({"use_sliding_window": True, "sliding_window": 0}, "requires a positive sliding window"),
        (
            {"sliding_window": 0, "layer_types": ["full_attention", "sliding_attention"]},
            "2 entries for 24 decoder layers",
        ),
    ],
)
def test_gpt_oss_layout_must_support_its_reported_retention(tmp_path, wrapped, change, message):
    from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

    decoder = _load_pre_downloaded_hf_config("openai/gpt-oss-20b") | change
    model = _write_config(
        tmp_path, {"architectures": [_UNKNOWN_WRAPPER], "text_config": decoder} if wrapped else decoder
    )
    opts = dict(
        systems_path=str(tmp_path),
        max_num_tokens=4096,
        max_batch_size=1,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        allow_naive_fallback=True,
        gpu_memory_capacity_bytes_override=100 * _GIB,
    )
    for function, extra in (
        (memory.estimate_kv_cache, {}),
        (memory.estimate_num_gpu_blocks, {"scheduler_block_size": 16}),
    ):
        with pytest.raises(ValueError, match=message):
            function(model, "missing-memory-spec", "vllm", "0.33.0", **opts, **extra)


# --------------------------------------------------------------------------- #
# estimate_kv_cache: routing between native and naive fallback.
# --------------------------------------------------------------------------- #


def test_estimate_kv_cache_falls_back_when_breakdown_raises(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("perf DB not available")

    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(_boom))
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "_load_config", lambda *a, **k: dict(_RAW_UNSUPPORTED))
    out = memory.estimate_kv_cache(
        "foo/bar-unknown-arch",
        "h200_sxm",
        "trtllm",
        max_num_tokens=8192,
        max_batch_size=256,
        memory_fraction_kind="of_free",
        memory_fraction_value=0.9,
        gpu_memory_capacity_bytes_override=200 * _GIB,
        allow_naive_fallback=True,
    )
    assert out["source"] == "naive_fallback"
    assert out["tolerance_adjusted"] is None
    assert "kvcache_quant_mode" not in out


@pytest.mark.parametrize("knob", ["cp_size", "dcp_size"])
def test_estimate_kv_cache_rejects_malformed_context_parallel_sizes_before_any_build(monkeypatch, knob):
    # Validation runs before the model build and the naive fallback: neither
    # may turn True (int 1) into a cp=dcp=1 estimate.
    def _never(*args, **kwargs):
        raise AssertionError("model build must not be reached")

    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(_never))
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "from_model_path", classmethod(_never))
    with pytest.raises(ValueError, match=f"{knob} must be a positive integer"):
        memory.estimate_kv_cache(
            "foo/bar",
            "h200_sxm",
            "vllm",
            max_num_tokens=8192,
            max_batch_size=256,
            memory_fraction_kind="of_total",
            memory_fraction_value=0.9,
            allow_naive_fallback=True,
            **{knob: True},
        )


def test_estimate_kv_cache_never_falls_back_after_a_context_parallel_capability_rejection(monkeypatch):
    # get_model rejects dcp/cp>1 for families without the sharded path with
    # NotImplementedError; the naive estimator knows neither knob, so falling
    # back would silently return an unstriped capacity for a striped request.
    def _unsupported(*args, **kwargs):
        raise NotImplementedError("Decode context parallelism (dcp_size=2) is not supported")

    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(_unsupported))
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "_load_config", lambda *a, **k: dict(_RAW_UNSUPPORTED))
    with pytest.raises(ValueError, match="context parallelism is not supported"):
        memory.estimate_kv_cache(
            "foo/bar-unknown-arch",
            "h200_sxm",
            "vllm",
            max_num_tokens=8192,
            max_batch_size=256,
            memory_fraction_kind="of_total",
            memory_fraction_value=0.9,
            gpu_memory_capacity_bytes_override=200 * _GIB,
            dcp_size=2,
            allow_naive_fallback=True,
        )
    # Without a CP knob the same rejection keeps the documented naive fallback.
    out = memory.estimate_kv_cache(
        "foo/bar-unknown-arch",
        "h200_sxm",
        "vllm",
        max_num_tokens=8192,
        max_batch_size=256,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        gpu_memory_capacity_bytes_override=200 * _GIB,
        allow_naive_fallback=True,
    )
    assert out["source"] == "naive_fallback"


def test_estimate_kv_cache_propagates_when_fallback_disabled(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("perf DB not available")

    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(_boom))
    with pytest.raises(ValueError, match="unsupported model/backend/GPU"):
        memory.estimate_kv_cache(
            "definitely/not-a-real-model-xyz",
            "h200_sxm",
            "trtllm",
            max_num_tokens=8192,
            max_batch_size=256,
            memory_fraction_kind="of_free",
            memory_fraction_value=0.9,
            gpu_memory_capacity_bytes_override=200 * _GIB,
            allow_naive_fallback=False,
        )


def test_estimate_kv_cache_keeps_version_slot_rejection_unwrapped(monkeypatch):
    # A backend version outside the queryable slots is a configuration error
    # with its own fix; it must not be re-labelled as an unsupported model.
    from aisim_core.sdk import perf_database

    slot_error = perf_database.UnlistedQueryVersionError("sglang version '0.5.10' is not a queryable version")

    def _reject(*args, **kwargs):
        raise slot_error

    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(_reject))
    with pytest.raises(perf_database.UnlistedQueryVersionError) as excinfo:
        memory.estimate_kv_cache(
            "Qwen/Qwen3-32B",
            "b300_sxm",
            "sglang",
            backend_version="0.5.10",
            max_num_tokens=8192,
            max_batch_size=256,
            memory_fraction_kind="of_total",
            memory_fraction_value=0.9,
            allow_naive_fallback=False,
        )
    assert excinfo.value is slot_error
    assert "unsupported model/backend/GPU" not in str(excinfo.value)


def test_estimate_kv_cache_nextn_reaches_breakdown(monkeypatch):
    # Acceptance is not part of the aic-core memory API.
    reached = {"n": 0}

    def _spy(*args, **kwargs):
        reached["n"] += 1
        reached["nextn"] = kwargs.get("nextn")
        raise RuntimeError("stop here")

    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(_spy))

    with pytest.raises(ValueError, match="unsupported model/backend/GPU"):
        memory.estimate_kv_cache(
            "Qwen/Qwen3-32B",
            "h200_sxm",
            "trtllm",
            max_num_tokens=8192,
            max_batch_size=256,
            memory_fraction_kind="of_free",
            memory_fraction_value=0.9,
            nextn=2,
            allow_naive_fallback=False,
        )
    assert reached["n"] == 1
    assert reached["nextn"] == 2


def test_estimate_kv_cache_rejects_bad_fraction_before_breakdown(monkeypatch):
    # Fraction validation must run before any breakdown attempt: a TRT-LLM +
    # of_total request fails even though the breakdown would also have raised.
    called = {"n": 0}

    def _spy(*args, **kwargs):
        called["n"] += 1
        raise RuntimeError("should not be reached")

    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(_spy))
    with pytest.raises(ValueError, match="incompatible memory fraction"):
        memory.estimate_kv_cache(
            "Qwen/Qwen3-32B",
            "h200_sxm",
            "trtllm",
            max_num_tokens=8192,
            max_batch_size=256,
            memory_fraction_kind="of_total",
            memory_fraction_value=0.9,
            allow_naive_fallback=True,
        )
    assert called["n"] == 0


def test_estimate_kv_cache_native_uses_synthetic_breakdown(monkeypatch):
    bd = _breakdown(60.0 * _GIB, 327_680.0, 141.0 * _GIB)
    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(lambda cls, *a, **k: cls(bd)))
    out = memory.estimate_kv_cache(
        "Qwen/Qwen3-32B",
        "h200_sxm",
        "trtllm",
        max_num_tokens=8192,
        max_batch_size=256,
        memory_fraction_kind="of_free",
        memory_fraction_value=0.9,
    )
    assert out["source"] == "native"
    assert out["total_kv_size_bytes"] == int((141.0 * _GIB - 60.0 * _GIB) * 0.9)
    assert out["tolerance_adjusted"] is None
    # The internal KV-curve inverse callable must not leak into the result dict
    # (it would not survive the PyO3 boundary).
    assert "tokens_from_kv_bytes" not in out


@pytest.mark.parametrize(
    ("backend", "memory_fraction_kind", "expected_reduction"),
    [
        ("vllm", "of_total", 8 * _GIB),
        # SGLang's explicit value is additional to graph/runtime headroom
        # already encoded in mem_fraction_static.
        ("sglang", "of_total", 8 * _GIB),
        ("trtllm", "of_free", int(8 * _GIB * 0.9)),
    ],
)
def test_cuda_graph_reservation_reduces_native_kv_budget(
    monkeypatch, backend, memory_fraction_kind, expected_reduction
):
    bd = _breakdown(60.0 * _GIB, 327_680.0, 141.0 * _GIB)
    monkeypatch.setattr(
        memory.KVCacheEstimator,
        "from_request",
        classmethod(lambda cls, *a, **k: cls(bd)),
    )
    baseline = memory.estimate_kv_cache(
        "Qwen/Qwen3-32B",
        "h200_sxm",
        backend,
        max_num_tokens=8192,
        max_batch_size=256,
        memory_fraction_kind=memory_fraction_kind,
        memory_fraction_value=0.9,
    )
    reserved = memory.estimate_kv_cache(
        "Qwen/Qwen3-32B",
        "h200_sxm",
        backend,
        max_num_tokens=8192,
        max_batch_size=256,
        memory_fraction_kind=memory_fraction_kind,
        memory_fraction_value=0.9,
        cuda_graph_reserved_bytes=8 * _GIB,
    )

    assert baseline["total_kv_size_bytes"] - reserved["total_kv_size_bytes"] == pytest.approx(expected_reduction, abs=1)
    assert reserved["memory_breakdown"]["cuda_graph_reserved_bytes"] == 8 * _GIB


def test_cuda_graph_reservation_rejected_before_breakdown(monkeypatch):
    called = {"n": 0}

    def _spy(*args, **kwargs):
        called["n"] += 1
        raise RuntimeError("should not be reached")

    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(_spy))
    for bad in (-1, 1.5, True, (1 << 53) + 1, 1 << 64):
        with pytest.raises(ValueError, match="cuda_graph_reserved_bytes"):
            memory.estimate_kv_cache(
                "Qwen/Qwen3-32B",
                "h200_sxm",
                "vllm",
                max_num_tokens=8192,
                max_batch_size=256,
                memory_fraction_kind="of_total",
                memory_fraction_value=0.9,
                cuda_graph_reserved_bytes=bad,
            )
    assert called["n"] == 0


def test_cuda_graph_reservation_preserves_exact_integer_in_native_breakdown(monkeypatch):
    max_exact = 1 << 53
    bd = _breakdown(1.0, 1.0, float(1 << 54))
    monkeypatch.setattr(
        memory.KVCacheEstimator,
        "from_request",
        classmethod(lambda cls, *a, **k: cls(bd)),
    )

    out = memory.estimate_kv_cache(
        "Qwen/Qwen3-32B",
        "h200_sxm",
        "vllm",
        max_num_tokens=8192,
        max_batch_size=256,
        memory_fraction_kind="of_total",
        memory_fraction_value=1.0,
        cuda_graph_reserved_bytes=max_exact,
    )

    assert out["memory_breakdown"]["cuda_graph_reserved_bytes"] == max_exact


def test_cuda_graph_reservation_reduces_naive_fallback_without_breakdown(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("perf DB not available")

    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(_boom))
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "_load_config", lambda *a, **k: dict(_RAW_UNSUPPORTED))
    common = {
        "max_num_tokens": 8192,
        "max_batch_size": 256,
        "memory_fraction_kind": "of_free",
        "memory_fraction_value": 0.9,
        "gpu_memory_capacity_bytes_override": 200 * _GIB,
        "allow_naive_fallback": True,
    }

    baseline = memory.estimate_kv_cache("foo/bar-unknown-arch", "h200_sxm", "trtllm", **common)
    reserved = memory.estimate_kv_cache(
        "foo/bar-unknown-arch",
        "h200_sxm",
        "trtllm",
        cuda_graph_reserved_bytes=8 * _GIB,
        **common,
    )

    assert baseline["total_kv_size_bytes"] - reserved["total_kv_size_bytes"] == pytest.approx(
        int(8 * _GIB * 0.8), abs=1
    )
    assert reserved["memory_breakdown"] is None


# --------------------------------------------------------------------------- #
# tolerance_fraction: validation (up front) + application (native & naive).
#
# This math moved out of Rust `apply_tolerance` into the Python
# `estimate_kv_cache`; the assertions reproduce the formula the Rust tests
# previously asserted (`adj_bytes = floor(raw * (1 - t))`, tokens floor-divided),
# so this is the parity guarantee that the port preserves the old numbers.
# --------------------------------------------------------------------------- #


def test_tolerance_applied_to_native_estimate(monkeypatch):
    bd = _breakdown(60.0 * _GIB, 327_680.0, 141.0 * _GIB)
    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(lambda cls, *a, **k: cls(bd)))
    out = memory.estimate_kv_cache(
        "Qwen/Qwen3-32B",
        "h200_sxm",
        "trtllm",
        max_num_tokens=8192,
        max_batch_size=256,
        memory_fraction_kind="of_free",
        memory_fraction_value=0.9,
        tolerance_fraction=0.05,
    )
    adj = out["tolerance_adjusted"]
    assert adj is not None
    assert adj["tolerance_fraction"] == pytest.approx(0.05)
    # adj bytes = floor(raw * 0.95); tokens recomputed by floor-divide.
    expected_bytes = int(out["total_kv_size_bytes"] * 0.95)
    assert adj["total_kv_size_bytes"] == expected_bytes
    assert adj["total_kv_size_tokens"] == expected_bytes // out["kv_size_per_token_bytes"]
    # Raw fields are left untouched.
    assert out["total_kv_size_bytes"] == int((141.0 * _GIB - 60.0 * _GIB) * 0.9)


def test_tolerance_applied_to_naive_fallback(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("perf DB not available")

    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(_boom))
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "_load_config", lambda *a, **k: dict(_RAW_UNSUPPORTED))
    out = memory.estimate_kv_cache(
        "foo/bar-unknown-arch",
        "h200_sxm",
        "trtllm",
        max_num_tokens=8192,
        max_batch_size=256,
        memory_fraction_kind="of_free",
        memory_fraction_value=0.9,
        gpu_memory_capacity_bytes_override=200 * _GIB,
        tolerance_fraction=0.05,
        allow_naive_fallback=True,
    )
    assert out["source"] == "naive_fallback"
    adj = out["tolerance_adjusted"]
    assert adj is not None
    expected_bytes = int(out["total_kv_size_bytes"] * 0.95)
    assert adj["total_kv_size_bytes"] == expected_bytes
    assert adj["total_kv_size_tokens"] == expected_bytes // out["kv_size_per_token_bytes"]


def test_tolerance_out_of_range_rejected_before_breakdown(monkeypatch):
    # Tolerance validation must run before any breakdown attempt: a bad tolerance
    # fails even though the breakdown would have succeeded (spy never called).
    called = {"n": 0}

    def _spy(*args, **kwargs):
        called["n"] += 1
        raise RuntimeError("should not be reached")

    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(_spy))
    for bad in (1.0, 1.5, -0.1, float("nan")):
        with pytest.raises(ValueError, match="tolerance_fraction"):
            memory.estimate_kv_cache(
                "Qwen/Qwen3-32B",
                "h200_sxm",
                "trtllm",
                max_num_tokens=8192,
                max_batch_size=256,
                memory_fraction_kind="of_free",
                memory_fraction_value=0.9,
                tolerance_fraction=bad,
                allow_naive_fallback=True,
            )
    assert called["n"] == 0


def test_tolerance_zero_is_noop_margin(monkeypatch):
    # t = 0.0 is accepted (half-open [0, 1)): adjusted bytes == raw bytes.
    bd = _breakdown(60.0 * _GIB, 327_680.0, 141.0 * _GIB)
    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", classmethod(lambda cls, *a, **k: cls(bd)))
    out = memory.estimate_kv_cache(
        "Qwen/Qwen3-32B",
        "h200_sxm",
        "trtllm",
        max_num_tokens=8192,
        max_batch_size=256,
        memory_fraction_kind="of_free",
        memory_fraction_value=0.9,
        tolerance_fraction=0.0,
    )
    adj = out["tolerance_adjusted"]
    assert adj is not None
    assert adj["total_kv_size_bytes"] == out["total_kv_size_bytes"]


def test_breakdown_charges_a_colocated_encoder_and_reserved_bytes(monkeypatch):
    # A rank hosting the vision tower keeps its weights and the multimodal
    # embedding cache outside the KV pool.
    class _EncoderOp:
        def get_weights(self):
            return 2 * _GIB

    class _StubModel:
        def __init__(self):
            self.encoder_ops = [_EncoderOp()]

        def get_kvcache_bytes_per_sequence(self, seq_len):
            return 100.0 * seq_len

        def get_kvcache_max_tokens(self, budget):
            return int(budget // 100)

        def get_kvcache_batch_capacity(self, budget, batch_size):
            return int(budget // 100)

    class _StubBackend:
        def _get_memory_usage(self, *a, **k):
            return {"weights": 1.0, "activations": 1.0, "others": 1.0, "nccl": 1.0, "kvcache": 0.0}

    class _StubDB:
        def __init__(self):
            self.version = "1.3.0rc20"
            self.system_spec = {"gpu": {"mem_capacity": 100 * _GIB}}

    monkeypatch.setattr(memory, "get_model", lambda *a, **k: _StubModel())
    monkeypatch.setattr(memory, "get_backend", lambda backend: _StubBackend())
    monkeypatch.setattr(memory.perf_database, "get_database", lambda *a, **k: _StubDB())
    built = {}
    real_build = memory.build_model_config

    def _build(**kwargs):
        built.update(kwargs)
        return real_build(**kwargs)

    monkeypatch.setattr(memory, "build_model_config", _build)

    plain = memory.KVCacheEstimator.from_request(
        "Qwen/Qwen3-VL-8B-Instruct", "h200_sxm", "sglang", max_num_tokens=8192, max_batch_size=8
    ).breakdown
    assert "enable_encoder_dp" not in built
    hosted = memory.KVCacheEstimator.from_request(
        "Qwen/Qwen3-VL-8B-Instruct",
        "h200_sxm",
        "sglang",
        max_num_tokens=8192,
        max_batch_size=8,
        colocated_encoder=True,
        reserved_bytes=100 << 20,
    ).breakdown
    assert hosted["weights_bytes"] - plain["weights_bytes"] == 2 * _GIB
    assert hosted["non_kv_bytes"] - plain["non_kv_bytes"] == 2 * _GIB + (100 << 20)
    # A hosted tower is sharded unless dp is asked for, as SGLang lays it out.
    assert built["enable_encoder_dp"] is False

    common = dict(
        model_path="Qwen/Qwen3-VL-8B-Instruct",
        system="h200_sxm",
        backend="sglang",
        max_num_tokens=8192,
        max_batch_size=8,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        colocated_encoder=True,
    )
    plain_estimate = memory.estimate_kv_cache(**common)
    reserved_estimate = memory.estimate_kv_cache(**common, reserved_bytes=100 << 20)
    assert plain_estimate["total_kv_size_bytes"] - reserved_estimate["total_kv_size_bytes"] == 100 << 20
    assert plain_estimate["memory_breakdown"]["reserved_bytes"] == 0
    assert reserved_estimate["memory_breakdown"]["reserved_bytes"] == 100 << 20
