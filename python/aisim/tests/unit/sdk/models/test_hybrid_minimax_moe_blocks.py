# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused MoE-block wiring for retained MiMo and synthetic local hybrid configs."""

from __future__ import annotations

import json

import pytest

import aisim_core.sdk.operations as ops
from aisim_core.sdk import config
from aisim_core.sdk.models import get_model

pytestmark = pytest.mark.unit

MIMO = "XiaomiMiMo/MiMo-V2-Flash"  # 256 experts, 48 layers (24 global_moe / 23 swa_moe / 1 swa_dense)


def _build(model_path, backend="sglang", **cfg_kwargs):
    cfg_kwargs.setdefault("tp_size", 8)
    cfg_kwargs.setdefault("pp_size", 1)
    cfg_kwargs.setdefault("moe_tp_size", 1)
    cfg_kwargs.setdefault("moe_ep_size", 8)
    cfg_kwargs.setdefault("attention_dp_size", 1)
    return get_model(model_path, config.ModelConfig(**cfg_kwargs), backend)


def _names(op_list):
    return [op._name for op in op_list]


def _op(op_list, name):
    """The single top-level op with this name."""
    (found,) = [op for op in op_list if op._name == name]
    return found


def _dispatches(op_list):
    # Composite getters (``overlap._group_a``) re-wrap children as the Rust
    # base classes, so match by class name instead of the Python shell type.
    return [op for op in op_list if type(op).__name__ == "MoEDispatch"]


@pytest.fixture
def synthetic_hybrid(tmp_path):
    """Independently authored dimensions; no excluded checkpoint contents."""

    def write(num_experts):
        model_dir = tmp_path / f"hybrid-{num_experts}"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["MiMoV2FlashForCausalLM"],
                    "num_hidden_layers": 4,
                    "hidden_size": 256,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 2,
                    "head_dim": 32,
                    "intermediate_size": 512,
                    "moe_intermediate_size": 128,
                    "num_experts_per_tok": 2,
                    "n_routed_experts": num_experts,
                    "vocab_size": 512,
                    "max_position_embeddings": 2048,
                    "hybrid_layer_pattern": [1, 0, 1, 0],
                    "moe_layer_freq": [1, 1, 0, 0],
                    "sliding_window_size": 128,
                }
            )
        )
        return str(model_dir)

    return write


@pytest.fixture(params=["retained_mimo", "synthetic"])
def hybrid_model_path(request, synthetic_hybrid):
    return MIMO if request.param == "retained_mimo" else synthetic_hybrid(16)


class TestHybridMoEBlocksViaBuilder:
    @pytest.mark.parametrize("num_experts", [16, 128])
    def test_synthetic_router_emission_boundary(self, synthetic_hybrid, num_experts):
        """Routers exist below and at the old 128-expert gate, only on MoE spans."""
        model = _build(synthetic_hybrid(num_experts))
        for phase, op_list in (("context", model.context_ops), ("generation", model.generation_ops)):
            routers = [op for op in op_list if "router_gemm" in op._name]
            assert _names(routers) == [f"{phase}_global_router_gemm", f"{phase}_swa_router_gemm"]
            for router in routers:
                assert isinstance(router, ops.GEMM)
                assert (router._n, router._k, router._scale_factor) == (num_experts, 256, 1)
                assert router._quant_mode.name == "bfloat16"
                index = op_list.index(router)
                assert _names(op_list[index + 1 : index + 4]) == [
                    router._name.replace("router_gemm", suffix)
                    for suffix in ("moe_pre_dispatch", "moe", "moe_post_dispatch")
                ]

    def test_synthetic_moe_comm_backend_is_rejected(self, synthetic_hybrid):
        with pytest.raises(ValueError, match="large-EP is not wired for the HYBRIDMOE family"):
            _build(
                synthetic_hybrid(16),
                moe_comm_backend={"context": "deepep_ht", "generation": "deepep_ll"},
            )

    def test_mimo_context_sequence_unchanged(self):
        """>=128-expert models already emitted the router: byte-identical emission."""
        model = _build(MIMO)
        assert _names(model.context_ops) == [
            "context_embedding",
            "context_global_attn_norm",
            "context_global_qkv_gemm",
            "context_attention",
            "context_global_proj_gemm",
            "context_global_moe_norm",
            "context_global_router_gemm",
            "context_global_moe_pre_dispatch",
            "context_global_moe",
            "context_global_moe_post_dispatch",
            "context_swa_attn_norm",
            "context_swa_qkv_gemm",
            "context_attention",
            "context_swa_proj_gemm",
            "context_swa_moe_norm",
            "context_swa_router_gemm",
            "context_swa_moe_pre_dispatch",
            "context_swa_moe",
            "context_swa_moe_post_dispatch",
            "context_swa_dense_attn_norm",
            "context_swa_dense_qkv_gemm",
            "context_attention",
            "context_swa_dense_proj_gemm",
            "context_swa_dense_ffn_norm",
            "context_swa_dense_gate_up_gemm",
            "context_swa_dense_act",
            "context_swa_dense_down_gemm",
            "context_logits_gemm",
            "context_p2p",
        ]

    def test_dispatch_ops_stay_quant_agnostic(self, hybrid_model_path):
        """Legacy hybrid dispatches never carried quant_mode; the builder's fused
        path always forwards it, so the family resets it on the returned ops
        (quant-aware dispatch modeling is a deliberate follow-up, not a rewire
        side effect)."""
        model = _build(hybrid_model_path)
        dispatches = _dispatches(model.context_ops) + _dispatches(model.generation_ops)
        assert len(dispatches) == 8  # (global + swa) x (pre + post) x (context + generation)
        assert all(op._quant_mode is None for op in dispatches)

    def test_cp_mutations_land_on_builder_ops(self, hybrid_model_path):
        """The post-hoc CP wiring mutates the builder-returned ops: prefill router/MoE
        get seq_split, dispatches get attn_cp_size; decode dispatches are re-routed
        to the decode-CP comm path (attn_cp_size + is_context=False)."""
        model = _build(hybrid_model_path, tp_size=1, moe_ep_size=2, attention_dp_size=1, cp_size=2)
        assert _op(model.context_ops, "context_global_router_gemm")._seq_split == 2
        assert _op(model.context_ops, "context_global_moe")._seq_split == 2
        context_dispatches = _dispatches(model.context_ops)
        assert len(context_dispatches) == 4
        assert all(op._attn_cp_size == 2 for op in context_dispatches)
        assert all(op._is_context for op in context_dispatches)
        generation_dispatches = _dispatches(model.generation_ops)
        assert len(generation_dispatches) == 4
        assert all(op._attn_cp_size == 2 for op in generation_dispatches)
        assert all(not op._is_context for op in generation_dispatches)
