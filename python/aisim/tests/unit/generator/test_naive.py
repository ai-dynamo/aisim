# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for generator naive module — nvbug 5941223."""

import json
import logging
import re
from unittest.mock import patch

import pytest

from aisim.generator.naive import (
    _ENGINE_LIMIT_KEYS,
    _calculate_min_tp,
    _estimate_model_weight_bytes,
    _sanitize_rfc1123,
    build_naive_generator_params,
)

_RFC1123_LABEL_RE = re.compile(r"^[a-z0-9]([a-z0-9\-.]*[a-z0-9])?$")
_UNSUPPORTED_GPT_NEOX_CONFIG = {
    "architectures": ["GPTNeoXForCausalLM"],
    "hidden_size": 512,
    "intermediate_size": 2048,
    "num_attention_heads": 8,
    "num_hidden_layers": 6,
    "torch_dtype": "float16",
    "vocab_size": 50304,
}
_UNSUPPORTED_MOE_CONFIG = {
    **_UNSUPPORTED_GPT_NEOX_CONFIG,
    "architectures": ["UnsupportedMoeForCausalLM"],
    "num_local_experts": 8,
}


@pytest.mark.unit
class TestSanitizeRfc1123:
    """Verify _sanitize_rfc1123 produces valid RFC 1123 subdomain labels."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Qwen/Qwen3-32B", "qwen-qwen3-32b"),
            ("meta-llama/Llama-3.1-70B", "meta-llama-llama-3.1-70b"),
            ("deepseek-ai/DeepSeek-V3", "deepseek-ai-deepseek-v3"),
            ("simple-model", "simple-model"),
            ("ALLCAPS", "allcaps"),
            ("a" * 100, "a" * 63),
        ],
    )
    def test_known_models(self, raw, expected):
        assert _sanitize_rfc1123(raw) == expected

    @pytest.mark.parametrize("bad_input", [None, "", "---", "///"])
    def test_fallback_to_dynamo(self, bad_input):
        assert _sanitize_rfc1123(bad_input) == "dynamo"

    @pytest.mark.parametrize(
        "raw",
        [
            "Qwen/Qwen3-32B",
            "meta-llama/Llama-3.1-70B",
            "nvidia/Nemotron-4-340B",
            "a",
            "a-b.c",
        ],
    )
    def test_result_matches_rfc1123(self, raw):
        result = _sanitize_rfc1123(raw)
        assert _RFC1123_LABEL_RE.match(result), f"{result!r} is not RFC 1123 compliant"
        assert len(result) <= 63


@pytest.mark.unit
class TestBuildNaiveGeneratorParams:
    """Verify build_naive_generator_params produces correct keys for the rendering engine."""

    @patch(
        "aisim.generator.naive._estimate_model_weight_bytes",
        return_value=30 * 1024**3,
    )
    @patch(
        "aisim.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_uses_service_config_and_k8s_config_keys(self, _mock_sys, _mock_est):
        result = build_naive_generator_params(
            model_name="Qwen/Qwen3-32B",
            total_gpus=8,
            system_name="h200_sxm",
            backend_name="vllm",
        )
        assert "ServiceConfig" in result, "expected ServiceConfig key, got 'service'"
        assert "K8sConfig" in result, "expected K8sConfig key, got 'k8s'"
        assert "service" not in result
        assert "k8s" not in result

    @patch(
        "aisim.generator.naive._estimate_model_weight_bytes",
        return_value=30 * 1024**3,
    )
    @patch(
        "aisim.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_name_prefix_is_rfc1123_valid(self, _mock_sys, _mock_est):
        result = build_naive_generator_params(
            model_name="Qwen/Qwen3-32B",
            total_gpus=8,
            system_name="h200_sxm",
            backend_name="vllm",
        )
        prefix = result["K8sConfig"]["name_prefix"]
        assert prefix is not None
        assert _RFC1123_LABEL_RE.match(prefix), f"{prefix!r} is not RFC 1123 compliant"

    @patch(
        "aisim.generator.naive._estimate_model_weight_bytes",
        return_value=30 * 1024**3,
    )
    @patch(
        "aisim.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_model_path_propagated(self, _mock_sys, _mock_est):
        result = build_naive_generator_params(
            model_name="Qwen/Qwen3-32B",
            total_gpus=8,
            system_name="h200_sxm",
            backend_name="vllm",
        )
        assert result["ServiceConfig"]["model_path"] == "Qwen/Qwen3-32B"
        assert result["ServiceConfig"]["model_name"] == "Qwen/Qwen3-32B"

    @patch(
        "aisim.generator.naive._estimate_model_weight_bytes",
        return_value=30 * 1024**3,
    )
    @patch(
        "aisim.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_agg_mode_set(self, _mock_sys, _mock_est):
        result = build_naive_generator_params(
            model_name="Qwen/Qwen3-32B",
            total_gpus=8,
            system_name="h200_sxm",
            backend_name="vllm",
        )
        assert result["DynConfig"]["mode"] == "agg"

    @patch(
        "aisim.generator.naive._estimate_model_weight_bytes",
        return_value=30 * 1024**3,
    )
    @patch(
        "aisim.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_include_frontend_true(self, _mock_sys, _mock_est):
        result = build_naive_generator_params(
            model_name="Qwen/Qwen3-32B",
            total_gpus=8,
            system_name="h200_sxm",
            backend_name="vllm",
        )
        assert result["ServiceConfig"]["include_frontend"] is True

    @patch(
        "aisim.generator.naive.get_model_config_from_model_path",
        return_value={"architecture": "Qwen3ForCausalLM", "num_experts": 0},
    )
    @patch(
        "aisim.generator.naive._estimate_model_weight_bytes",
        return_value=30 * 1024**3,
    )
    @patch(
        "aisim.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_generator_dynamo_version_sets_backend_image_default(self, _mock_sys, _mock_est, _mock_model):
        result = build_naive_generator_params(
            model_name="Qwen/Qwen3-32B",
            total_gpus=8,
            system_name="h200_sxm",
            backend_name="vllm",
            generator_dynamo_version="1.2.0",
        )

        assert result["generator_dynamo_version"] == "1.2.0"
        assert result["K8sConfig"]["k8s_image"] == "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0"

    @patch(
        "aisim.generator.naive.get_model_config_from_model_path",
        return_value={"architecture": "Qwen3ForCausalLM", "num_experts": 0},
    )
    @patch(
        "aisim.generator.naive._estimate_model_weight_bytes",
        return_value=30 * 1024**3,
    )
    @patch(
        "aisim.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_generator_overrides_are_applied_before_schema_defaults(self, _mock_sys, _mock_est, _mock_model):
        result = build_naive_generator_params(
            model_name="Qwen/Qwen3-32B",
            total_gpus=8,
            system_name="h200_sxm",
            backend_name="sglang",
            generator_dynamo_version="1.2.0",
            generator_overrides={
                "BenchConfig": {"name": "custom-bench"},
                "K8sConfig": {"k8s_namespace": "custom-ns"},
                "LlmdConfig": {"routing_proxy_enabled": False},
                "SflowConfig": {"slurm_partition": "debug"},
                "Workers": {"agg": {"max_batch_size": 256}},
            },
        )

        assert result["BenchConfig"]["name"] == "custom-bench"
        assert result["K8sConfig"]["k8s_namespace"] == "custom-ns"
        assert result["K8sConfig"]["k8s_image"] == "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0"
        assert result["LlmdConfig"]["routing_proxy_enabled"] is False
        assert result["SflowConfig"]["slurm_partition"] == "debug"
        assert result["params"]["agg"]["max_batch_size"] == 256


@pytest.mark.unit
class TestEstimateModelWeightBytes:
    def test_estimates_unsupported_architecture_from_raw_config(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps(_UNSUPPORTED_GPT_NEOX_CONFIG))

        assert _estimate_model_weight_bytes(str(tmp_path)) == 153354240

    def test_refuses_an_invalid_config_of_a_supported_architecture_with_the_parser_message(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"], "hidden_size": 128}))

        with pytest.raises(RuntimeError) as error:
            _estimate_model_weight_bytes(str(tmp_path))
        assert str(error.value) == (
            f"Could not estimate model size for {str(tmp_path)!r}: the config parser refuses the HF config of the "
            "supported architecture LlamaForCausalLM: ValueError: Model config must define 'num_hidden_layers' or a "
            "parseable layer pattern."
        )

    def test_logs_the_raw_config_estimator_as_the_step_after_a_failed_parse(self, tmp_path, caplog):
        (tmp_path / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"], "hidden_size": 128}))

        with caplog.at_level(logging.INFO, logger="aisim.generator.naive"), pytest.raises(RuntimeError):
            _estimate_model_weight_bytes(str(tmp_path))
        assert f"Normalized model parsing failed for {tmp_path}; trying the raw-config estimator." in caplog.messages

    def test_sglang_generator_selects_its_own_mixed_attention_decoder(self, tmp_path):
        from aisim.generator.api import generate_naive_config
        from aisim_core.sdk.utils import _load_pre_downloaded_hf_config

        raw = _load_pre_downloaded_hf_config("Qwen/Qwen3-0.6B")
        raw.update(use_sliding_window=True, max_window_layers=14)
        raw.pop("layer_types", None)
        raw.pop("sliding_window", None)
        wrapper = {"architectures": ["LlavaForConditionalGeneration"], "text_config": raw}
        (tmp_path / "config.json").write_text(json.dumps(wrapper))
        result = generate_naive_config(str(tmp_path), 1, "h100_sxm", "sglang", backend_version="0.5.20")
        assert result["parallelism"]["tp"] == 1
        with pytest.raises(RuntimeError, match="vLLM Qwen3 does not support sliding attention"):
            _estimate_model_weight_bytes(str(tmp_path), backend="vllm", backend_version="0.32.0")

    def test_estimates_an_unknown_wrapper_from_its_decoder(self, tmp_path):
        decoder = {
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 1024,
            "vocab_size": 1000,
            "torch_dtype": "bfloat16",
        }
        wrapper = {
            "architectures": ["UnknownWrapperForConditionalGeneration"],
            "hidden_size": 4096,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "text_config": decoder,
        }
        (tmp_path / "config.json").write_text(json.dumps(wrapper))

        # 2 * 1,000 * 256 + 4 * (4 * 256**2 + 3 * 256 * 1,024) = 4,706,304 parameters at 2 bytes each.
        assert _estimate_model_weight_bytes(str(tmp_path)) == 9_412_608

    @patch(
        "aisim.generator.naive._load_model_config_from_model_path",
        return_value=_UNSUPPORTED_GPT_NEOX_CONFIG,
    )
    def test_remote_unsupported_architecture_loads_config_once(self, mock_load_config):
        assert _estimate_model_weight_bytes("EleutherAI/pythia-70m") == 153354240
        mock_load_config.assert_called_once_with("EleutherAI/pythia-70m")

    def test_remote_sidecar_and_root_are_reused_across_generations(self):
        from aisim.sdk.memory import NaiveKVCacheEstimator
        from aisim_core.sdk import utils

        root = {
            "architectures": ["LlavaForConditionalGeneration"],
            "torch_dtype": "float16",
            "text_config": {"model_type": "llama", "vocab_size": 2048},
        }
        sidecar = {"quant_method": "modelopt", "quantization": {"quant_algo": "FP8", "exclude_modules": ["*mlp*"]}}
        utils._load_model_config_from_model_path.cache_clear()
        with (
            patch.object(utils, "_download_hf_config", return_value=root) as config_read,
            patch.object(utils, "_download_hf_json", return_value=sidecar) as sidecar_read,
            patch.object(
                NaiveKVCacheEstimator, "from_model_path", side_effect=AssertionError("unexpected remote metadata read")
            ),
        ):
            first = _estimate_model_weight_bytes("test/remote-sidecar", backend_version="0.32.0")
            expected = NaiveKVCacheEstimator.from_hf_config(
                {**root, "hf_quant_config": sidecar}, tp_size=1, pp_size=1, backend_version="0.32.0"
            ).weight_bytes()
            assert first == expected
            assert _estimate_model_weight_bytes("test/remote-sidecar", backend_version="0.32.0") == first
            config_read.assert_called_once_with("test/remote-sidecar")
            sidecar_read.assert_called_once_with("test/remote-sidecar", "hf_quant_config.json", raise_on_404=False)

    @patch(
        "aisim.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 80 * 1024**3},
    )
    def test_builds_params_for_unsupported_architecture(self, _mock_system, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps(_UNSUPPORTED_GPT_NEOX_CONFIG))

        result = build_naive_generator_params(
            model_name=str(tmp_path),
            total_gpus=8,
            system_name="h100_sxm",
            backend_name="vllm",
        )

        assert result["params"]["agg"]["tensor_parallel_size"] == 1
        assert result["ModelConfig"]["fits_in_memory"] is True

    @patch(
        "aisim.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 512 * 1024**2},
    )
    def test_builds_tep_params_for_unsupported_moe_architecture(self, _mock_system, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps(_UNSUPPORTED_MOE_CONFIG))

        result = build_naive_generator_params(
            model_name=str(tmp_path),
            total_gpus=8,
            system_name="h100_sxm",
            backend_name="vllm",
        )

        assert result["ModelConfig"]["is_moe"] is True
        assert result["params"]["agg"]["tensor_parallel_size"] == 1
        assert result["params"]["agg"]["moe_tensor_parallel_size"] == 2
        assert result["params"]["agg"]["moe_expert_parallel_size"] == 1

    @patch("aisim.generator.naive._load_model_config_from_model_path")
    def test_raises_when_config_download_fails(self, mock_get_config):
        mock_get_config.side_effect = Exception(
            "Failed to download nonexistent-org/fake-model-12345's config.json from HuggingFace: "
            "HuggingFace returned HTTP error 401: Unauthorized."
        )
        with pytest.raises(RuntimeError, match=r"Model .* not found or config unavailable"):
            _estimate_model_weight_bytes("nonexistent-org/fake-model-12345")

    @patch("aisim.generator.naive._get_system_config")
    @patch("aisim.generator.naive._estimate_model_weight_bytes")
    def test_build_naive_generator_params_propagates_model_not_found(self, mock_est, _mock_sys):
        mock_est.side_effect = RuntimeError("Model 'nonexistent-org/fake-model-12345' not found or config unavailable")
        with pytest.raises(RuntimeError, match="not found or config unavailable"):
            build_naive_generator_params(
                model_name="nonexistent-org/fake-model-12345",
                total_gpus=8,
                system_name="h200_sxm",
                backend_name="vllm",
            )


@pytest.mark.unit
class TestCalculateMinTp:
    """Verify _calculate_min_tp memory-fit floor, including multi-node sweeps."""

    # H100: 80 GiB per GPU.  DeepSeek R1 FP8: ~671 GiB of weights.
    _H100_VRAM = 80 * 1024**3
    _R1_FP8_WEIGHTS = 671 * 1024**3

    def test_single_node_caps_at_gpus_per_node(self):
        """Dense default: min_tp capped at gpus_per_node even if model is larger."""
        min_gpus, fits, tp = _calculate_min_tp(
            model_weight_bytes=self._R1_FP8_WEIGHTS,
            vram_per_gpu=self._H100_VRAM,
            gpus_per_node=8,
            total_gpus=32,
        )

        assert fits is False  # cannot fit in single node configuration
        assert min_gpus == 8  # capped
        assert tp == 16  # required TP still computed

    def test_multi_node_allows_crossing_node_boundary(self):
        """MoE wide-EP: min_tp can span nodes, floored to power-of-2 fit. Model fits when spanning multiple nodes."""
        min_gpus, fits, tp = _calculate_min_tp(
            model_weight_bytes=self._R1_FP8_WEIGHTS,
            vram_per_gpu=self._H100_VRAM,
            gpus_per_node=8,
            total_gpus=32,
            allow_multi_node=True,
        )

        assert fits is True
        assert tp == 16
        assert min_gpus == 16

    def test_multi_node_capped_by_total_gpus(self):
        """Insufficient total GPUs → should not fit."""
        min_gpus, fits, tp = _calculate_min_tp(
            model_weight_bytes=self._R1_FP8_WEIGHTS,
            vram_per_gpu=self._H100_VRAM,
            gpus_per_node=8,
            total_gpus=8,
            allow_multi_node=True,
        )

        assert fits is False
        assert min_gpus == 8
        assert tp == 16

    def test_small_model_fits_on_one_gpu(self):
        """Tiny model should trivially fit."""
        min_gpus, fits, tp = _calculate_min_tp(
            model_weight_bytes=10 * 1024**3,
            vram_per_gpu=self._H100_VRAM,
            gpus_per_node=8,
            total_gpus=8,
        )

        assert fits is True
        assert min_gpus == 1
        assert tp == 1


@pytest.mark.unit
class TestPreserveEngineLimits:
    """The FPM collector's declared entry point: preserve_engine_limits=True."""

    @patch(
        "aisim.generator.naive._estimate_model_weight_bytes",
        return_value=30 * 1024**3,
    )
    @patch(
        "aisim.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_strips_engine_limit_keys_and_sets_guard(self, _mock_sys, _mock_est):
        overrides = {
            "params": {
                "agg": {
                    "max_num_tokens": 4096,
                    "max_seq_len": 8192,
                    "tokens_per_block": 64,
                    "gpu_memory_utilization": 0.9,
                    "compilation_config": "{}",
                    "cuda_graph_batch_sizes": [1, 2, 4],
                    "trust_remote_code": True,
                }
            }
        }
        result = build_naive_generator_params(
            model_name="Qwen/Qwen3-32B",
            total_gpus=8,
            system_name="h200_sxm",
            backend_name="vllm",
            generator_overrides=overrides,
            preserve_engine_limits=True,
        )

        assert result["preserve_engine_limits"] is True
        assert result["params"]
        for role_params in result["params"].values():
            assert not set(role_params) & set(_ENGINE_LIMIT_KEYS)
        assert result["params"]["agg"]["trust_remote_code"] is True

    @patch(
        "aisim.generator.naive._estimate_model_weight_bytes",
        return_value=30 * 1024**3,
    )
    @patch(
        "aisim.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_default_keeps_naive_engine_limits(self, _mock_sys, _mock_est):
        result = build_naive_generator_params(
            model_name="Qwen/Qwen3-32B",
            total_gpus=8,
            system_name="h200_sxm",
            backend_name="vllm",
        )

        assert "preserve_engine_limits" not in result
        assert result["params"]["agg"]["max_batch_size"] == 128


@pytest.mark.unit
class TestRenderingNameFallback:
    """Verify prepare_template_context uses 'dynamo' fallback for missing name_prefix."""

    def test_none_name_prefix_becomes_dynamo(self):
        from aisim.generator.rendering.engine import prepare_template_context

        params = {
            "K8sConfig": {},
            "ServiceConfig": {"model_path": "test"},
            "DynConfig": {"mode": "agg"},
            "params": {"agg": {}},
            "WorkerConfig": {},
        }
        ctx = prepare_template_context(params, "vllm")
        assert ctx["name_prefix"] == "dynamo"
        assert ctx["name"] == "dynamo-agg"

    def test_include_frontend_yields_one_replica(self):
        from aisim.generator.rendering.engine import prepare_template_context

        params = {
            "K8sConfig": {"name_prefix": "test"},
            "ServiceConfig": {"model_path": "test", "include_frontend": True},
            "DynConfig": {"mode": "agg"},
            "params": {"agg": {}},
            "WorkerConfig": {},
        }
        ctx = prepare_template_context(params, "vllm")
        assert ctx["frontend_replicas"] == 1

    def test_no_include_frontend_yields_zero_replica(self):
        from aisim.generator.rendering.engine import prepare_template_context

        params = {
            "K8sConfig": {"name_prefix": "test"},
            "ServiceConfig": {"model_path": "test"},
            "DynConfig": {"mode": "agg"},
            "params": {"agg": {}},
            "WorkerConfig": {},
        }
        ctx = prepare_template_context(params, "vllm")
        assert ctx["frontend_replicas"] == 0

    def test_benchmark_prefix_preserves_explicit_model_zero(self):
        from aisim.generator.rendering.engine import prepare_template_context

        params = {
            "K8sConfig": {"name_prefix": "test"},
            "ServiceConfig": {"model_path": "test", "prefix": 1024},
            "ModelConfig": {"prefix": 0},
            "BenchConfig": {},
            "DynConfig": {"mode": "agg"},
            "params": {"agg": {}},
            "WorkerConfig": {},
        }
        ctx = prepare_template_context(params, "vllm")
        assert ctx["BenchConfig"]["prefix"] == 0


@pytest.mark.unit
def test_frozen_model_config_renders_without_model_resolution(monkeypatch):
    # E3: with a frozen config the builder must not touch the filesystem or
    # network for model metadata, even for unreachable checkpoints.
    import aisim.generator.naive as naive_module

    def _boom(*_a, **_k):
        raise AssertionError("model resolution must not run when model_config is provided")

    monkeypatch.setattr("aisim.sdk.utils.get_model_config_from_model_path", _boom)
    frozen = {
        "layers": 4,
        "hidden_size": 128,
        "inter_size": 256,
        "vocab": 2048,
        "num_experts": 8,
        "moe_inter_size": 64,
        "architecture": "GlmMoeDsaForCausalLM",
    }
    params = naive_module.build_naive_generator_params(
        "/cluster-only/private-model",
        4,
        "b200_sxm",
        "vllm",
        model_config=frozen,
    )
    assert params["ModelConfig"]["is_moe"] is True


@pytest.mark.unit
def test_preserve_engine_limits_strips_disagg_roles():
    params = build_naive_generator_params(
        "/cluster-only/private-model",
        4,
        "b200_sxm",
        "vllm",
        mode="disagg",
        preserve_engine_limits=True,
        model_config={
            "layers": 2,
            "hidden_size": 64,
            "inter_size": 128,
            "vocab": 1000,
            "num_experts": 0,
            "moe_inter_size": 0,
            "architecture": "Qwen2ForCausalLM",
        },
    )
    for role in ("prefill", "decode"):
        role_params = params["params"][role]
        assert "max_batch_size" not in role_params
        assert "gpu_memory_utilization" not in role_params
    assert params["preserve_engine_limits"] is True
