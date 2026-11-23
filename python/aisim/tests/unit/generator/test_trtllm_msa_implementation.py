# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 x TRT-LLM SM100-family deployments must prescribe the msa
(fmha_sm100) sparse-attention implementation: rc23 serving defaults to the
Triton reference, while the shipped SM100/103 perf tables are collected with
implementation="msa" — the emitted engine config makes deployments run
exactly the configuration the data represents (PR #1507 review 4969690316).
Non-MSA models and non-SM100 systems must NOT carry the block."""

from __future__ import annotations

import pytest
import yaml

from aisim.generator.api import generate_backend_artifacts

pytestmark = pytest.mark.unit

_BACKEND_VERSION = "1.3.0rc23"


def _params(model_cfg: dict) -> dict:
    return {
        "ServiceConfig": {
            "model_path": "MiniMaxAI/MiniMax-M3",
            "served_model_path": "MiniMaxAI/MiniMax-M3",
            "served_model_name": "MiniMax-M3",
            "include_frontend": True,
        },
        "K8sConfig": {"name_prefix": "test", "k8s_namespace": "default"},
        "DynConfig": {"mode": "agg"},
        "WorkerConfig": {"agg_workers": 1, "agg_gpus_per_worker": 8},
        "NodeConfig": {"num_gpus_per_node": 8},
        "SlaConfig": {"isl": 1024, "osl": 256},
        "ModelConfig": {"is_moe": True, "prefix": 0, **model_cfg},
        "BenchConfig": {},
        "params": {
            "agg": {
                "tensor_parallel_size": 8,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "moe_tensor_parallel_size": 1,
                "moe_expert_parallel_size": 8,
                "max_batch_size": 32,
                "max_num_tokens": 4096,
                "max_seq_len": 4096,
            },
        },
    }


def _engine_yaml(artifacts) -> dict:
    for name, content in artifacts.items():
        if "extra_engine_args" in name and "agg" in name:
            return yaml.safe_load(content)
    raise AssertionError(f"no agg engine args artifact in {sorted(artifacts)}")


def test_msa_implementation_rendered_when_bridge_sets_it():
    artifacts = generate_backend_artifacts(
        backend="trtllm",
        backend_version=_BACKEND_VERSION,
        params=_params({"msa_sparse_implementation": "msa"}),
    )
    engine = _engine_yaml(artifacts)
    assert engine.get("sparse_attention_config") == {"implementation": "msa"}


def test_no_sparse_block_without_the_bridge_field():
    artifacts = generate_backend_artifacts(
        backend="trtllm",
        backend_version=_BACKEND_VERSION,
        params=_params({}),
    )
    engine = _engine_yaml(artifacts)
    assert "sparse_attention_config" not in engine


class _FakeTask:
    def __init__(self, backend="trtllm", model="MiniMaxAI/MiniMax-M3", system="b200_sxm"):
        self.primary_backend_name = backend
        self.primary_model_path = model
        self.primary_system_name = system


@pytest.mark.parametrize(
    ("backend", "model", "system", "expected"),
    [
        ("trtllm", "deepseek-ai/DeepSeek-V3", "b200_sxm", None),
        ("vllm", "MiniMaxAI/MiniMax-M3", "b200_sxm", None),
    ],
)
def test_bridge_prescribes_msa_only_for_m3_on_the_sm100_family(backend, model, system, expected):
    from aisim.generator.module_bridge import _msa_sparse_implementation

    assert _msa_sparse_implementation(_FakeTask(backend, model, system)) == expected


# --- rc29: sparse_attention_config.algorithm (required on every SM) ---------------------
_RC29 = "1.3.0rc29"


def test_rc29_template_renders_algorithm_for_m3_without_implementation():
    """SM90 M3 (no fmha_sm100 prescription) still needs algorithm=minimax_m3 at rc29."""
    artifacts = generate_backend_artifacts(
        backend="trtllm", backend_version=_RC29, params=_params({"msa_sparse_algorithm": "minimax_m3"})
    )
    assert _engine_yaml(artifacts).get("sparse_attention_config") == {"algorithm": "minimax_m3"}


def test_rc29_template_renders_both_keys_on_sm100():
    artifacts = generate_backend_artifacts(
        backend="trtllm",
        backend_version=_RC29,
        params=_params({"msa_sparse_algorithm": "minimax_m3", "msa_sparse_implementation": "msa"}),
    )
    assert _engine_yaml(artifacts).get("sparse_attention_config") == {
        "algorithm": "minimax_m3",
        "implementation": "msa",
    }


def test_rc23_template_ignores_the_algorithm_key():
    """Prior template untouched: rc23 never read `algorithm`, so it must not appear."""
    artifacts = generate_backend_artifacts(
        backend="trtllm", backend_version=_BACKEND_VERSION, params=_params({"msa_sparse_algorithm": "minimax_m3"})
    )
    assert "sparse_attention_config" not in _engine_yaml(artifacts)


@pytest.mark.parametrize(
    ("backend", "model", "expected"),
    [
        ("trtllm", "deepseek-ai/DeepSeek-V3", None),
        ("sglang", "MiniMaxAI/MiniMax-M3", None),
    ],
)
def test_algorithm_prescription_is_architecture_keyed_and_trtllm_only(backend, model, expected):
    from aisim.generator.module_bridge import _msa_sparse_algorithm

    assert _msa_sparse_algorithm(_FakeTask(backend, model, "h200_sxm")) == expected
