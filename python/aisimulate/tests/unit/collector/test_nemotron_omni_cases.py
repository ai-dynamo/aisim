# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NemotronH_Omni_Reasoning_V3 collection case file."""

import pytest
from collector.case_generator import (
    get_attention_context_shape_sweeps,
    get_attention_head_configs,
    get_common_mamba2_test_cases,
    get_common_moe_test_cases,
    moe_model_allows_quantization,
)
from collector.model_cases import (
    build_collection_case_plan,
    default_architecture_cases_path,
)

pytestmark = pytest.mark.unit

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
ARCHITECTURE = "NemotronH_Omni_Reasoning_V3"


@pytest.mark.parametrize(
    ("backend", "ops"),
    [
        ("sglang", ["attention_context", "attention_generation", "compute_scale", "gemm", "moe"]),
        ("trtllm", ["attention_context", "attention_generation", "compute_scale", "gemm", "mamba2", "moe"]),
        ("vllm", ["attention_context", "attention_generation", "compute_scale", "gemm", "moe"]),
    ],
)
def test_model_path_resolves_the_architecture_case_file(backend, ops):
    plan = build_collection_case_plan(backend=backend, model_path=MODEL, sm_version=100)
    assert plan.model_architecture == ARCHITECTURE
    assert plan.model_cases_paths == [default_architecture_cases_path(ARCHITECTURE)]
    assert plan.ops == ops


def test_latent_moe_row_carries_the_checkpoint_geometry_and_routing(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL)
    cases = get_common_moe_test_cases()
    assert len(cases) == 117
    assert {(c.model_name, c.hidden_size, c.inter_size, c.topk, c.num_experts) for c in cases} == {
        (MODEL, 1024, 2688, 22, 512)
    }
    assert {
        (
            c.sglang_moe_activation,
            c.sglang_moe_is_gated,
            c.sglang_moe_scoring_func,
            c.sglang_moe_routing_method_type,
            c.sglang_moe_routed_scaling_factor,
            c.sglang_moe_renormalize,
            c.sglang_moe_has_correction_bias,
            c.sglang_moe_num_expert_group,
            c.sglang_moe_topk_group,
        )
        for c in cases
    } == {("relu2", False, "sigmoid", "DeepSeekV3", 5.0, True, True, 1, 1)}
    assert moe_model_allows_quantization("sglang", MODEL, "nvfp4")
    assert moe_model_allows_quantization("trtllm", MODEL, "nvfp4")
    # vLLM's FlashInfer NVFP4 MoE path stops at top-k 10, and this checkpoint routes to 22 experts.
    assert not moe_model_allows_quantization("vllm", MODEL, "nvfp4")


def test_mamba2_row_keeps_the_full_model_heads(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL)
    cases = get_common_mamba2_test_cases()
    assert [(c.model_name, c.phase, c.d_model, c.d_state, c.d_conv, c.nheads, c.head_dim) for c in cases] == [
        (MODEL, "context", 4096, 128, 4, 128, 64),
        (MODEL, "generation", 4096, 128, 4, 128, 64),
    ]
    assert {(c.n_groups, c.chunk_size) for c in cases} == {(8, 128)}


@pytest.mark.parametrize(
    ("sm_version", "kernel_source"), [(90, "fa3"), (100, "trtllm_mha"), (103, "trtllm_mha"), (120, "flashinfer")]
)
def test_sglang_attention_profile_keeps_the_head_tuple(monkeypatch, sm_version, kernel_source):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL)
    configs = [
        cfg
        for sweep in get_attention_context_shape_sweeps("sglang")
        for cfg in get_attention_head_configs(sweep, phase="context", backend="sglang", sm_version=sm_version)
    ]
    assert {(c.num_heads, c.num_kv_heads, c.head_dim, c.window_size, c.kernel_source) for c in configs} == {
        (heads, kv_heads, 128, 0, kernel_source)
        for heads, kv_heads in ((32, 2), (16, 1), (8, 1), (4, 1), (2, 1), (1, 1))
    }
    assert {c.architecture for c in configs} == {ARCHITECTURE}
