# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint precision must survive local-directory loading and engine compilation."""

import json

import pytest

from aisim.sdk import common, config
from aisim.sdk.models.helpers import resolve_dsv4_moe_arch
from aisim_core import _native
from aisim_core.sdk import utils
from aisim_core.sdk.engine import compile_engine

pytestmark = pytest.mark.unit


@pytest.fixture
def local_kimi(tmp_path, monkeypatch):
    """Invented small text-only topology; no removed checkpoint contents."""
    raw = {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "num_hidden_layers": 4,
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "intermediate_size": 512,
        "moe_intermediate_size": 128,
        "n_routed_experts": 16,
        "num_experts_per_token": 2,
        "vocab_size": 1024,
        "max_position_embeddings": 2048,
        "q_lora_rank": 32,
        "kv_lora_rank": 32,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 16,
        "v_head_dim": 32,
        "routed_expert_hidden_size": 128,
        "linear_attn_config": {"kda_layers": [1, 3], "num_heads": 8, "head_dim": 32},
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "config_groups": {"experts": {"weights": {"num_bits": 4, "type": "float"}}},
            "ignore": ["self_attn", "mlp.shared_experts", "lm_head"],
        },
    }
    # An unrelated basename rules out fixing this with a path-name heuristic.
    path = tmp_path / "checkpoint-copy"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(raw))

    def no_network(*args, **kwargs):
        pytest.fail("local estimator preparation must not download metadata")

    monkeypatch.setattr(utils, "_download_hf_json", no_network)
    yield path, raw
    utils._load_model_config_from_model_path.cache_clear()
    utils.get_model_config_from_model_path.cache_clear()


@pytest.mark.parametrize(
    "system,backend,explicit,expected",
    [
        ("b200_sxm", "sglang", None, "w4a8_mxfp4_mxfp8"),
        ("b300_sxm", "sglang", None, "w4a8_mxfp4_mxfp8"),
        ("h200_sxm", "sglang", None, "w4a16_mxfp4"),
        ("b200_sxm", "vllm", None, "w4a16_mxfp4"),
        ("b200_sxm", "sglang", "fp8", "fp8"),
    ],
)
def test_local_kimi_estimator_compiles_the_correct_moe_lane(local_kimi, system, backend, explicit, expected):
    path, _ = local_kimi
    blob = compile_engine(str(path), system, backend, "current", moe_quant_mode=explicit)
    spec = json.loads(_native.engine_spec_json_from_bincode(blob))
    assert spec["engine"]["moe_dtype"] == expected
    # Check the emitted operations too: changing identity alone cannot fix pricing.
    moe_ops = [op["Moe"] for op in spec["context_ops"] + spec["generation_ops"] if "Moe" in op]
    assert moe_ops
    assert all(op["quant_mode"] == expected for op in moe_ops)


def test_local_and_hf_alias_resolve_the_same_moe_lane(local_kimi, monkeypatch):
    path, raw = local_kimi
    monkeypatch.setattr(utils, "_download_hf_config", lambda _: json.loads(json.dumps(raw)))
    monkeypatch.setattr(utils, "_download_hf_json", lambda *args, **kwargs: None)
    for model_path in (str(path), "moonshotai/Kimi-K3", "acme/Synthetic-Kimi-Copy"):
        cfg = config.ModelConfig()
        resolve_dsv4_moe_arch(cfg, model_path, system_name="b200_sxm", backend_name="sglang")
        assert cfg.moe_quant_mode is common.MoEQuantMode.w4a8_mxfp4_mxfp8


def test_canonical_estimator_uses_local_checkpoint_precision(local_kimi, monkeypatch):
    from aisim_core.sdk import engine
    from aisim_core.sdk.rust_engine_step import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    path, _ = local_kimi
    compiled = []

    def trace_compile(*args, **kwargs):
        blob = compile_engine(*args, **kwargs)
        compiled.append(json.loads(_native.engine_spec_json_from_bincode(blob)))
        return blob

    monkeypatch.setattr(engine, "compile_engine", trace_compile)
    model = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model=str(path),
            system="b200_sxm",
            backend="sglang",
            worker_type="aggregated",
            estimation_mode="op_level",
        )
    )
    assert isinstance(model, RustForwardPassPerfModel)
    assert compiled and all(spec["engine"]["moe_dtype"] == "w4a8_mxfp4_mxfp8" for spec in compiled)


@pytest.mark.parametrize("dense_only", [False, True], ids=["no_quantization", "dense_only_mixed_precision"])
def test_kimi_missing_precision_fails_and_explicit_mode_is_preserved(local_kimi, dense_only):
    path, raw = local_kimi
    del raw["quantization_config"]
    if dense_only:
        raw["quantization_config"] = {
            "quant_method": "mixed_precision",
            "quantized_layers": {"model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"}},
        }
    (path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="requires expert quantization metadata"):
        compile_engine(str(path), "b200_sxm", "sglang", "current")
    blob = compile_engine(str(path), "b200_sxm", "sglang", "current", moe_quant_mode="bfloat16")
    assert json.loads(_native.engine_spec_json_from_bincode(blob))["engine"]["moe_dtype"] == "bfloat16"


@pytest.mark.parametrize(
    "selector",
    ["mlp.experts", r"re:.*mlp\.experts\..*", r"re:.*mlp\.experts.*", "re:.*experts.*"],
)
def test_local_kimi_explicitly_unquantized_experts_keep_bfloat16(local_kimi, selector):
    from aisim.sdk.task_v2 import Task

    path, raw = local_kimi
    raw["quantization_config"]["ignore"].append(selector)
    (path / "config.json").write_text(json.dumps(raw))
    blob = compile_engine(str(path), "b200_sxm", "sglang", "current")
    spec = json.loads(_native.engine_spec_json_from_bincode(blob))
    assert spec["engine"]["moe_dtype"] == "bfloat16"
    moe_ops = [op["Moe"] for op in spec["context_ops"] + spec["generation_ops"] if "Moe" in op]
    assert moe_ops and all(op["quant_mode"] == "bfloat16" for op in moe_ops)
    task = Task(model_path=str(path), system_name="b200_sxm", backend_name="sglang", total_gpus=1)
    assert task.build_model_config(role="agg").moe_quant_mode is common.MoEQuantMode.bfloat16


@pytest.mark.parametrize(
    "selector",
    [
        "model.layers.0.mlp.experts.0.down_proj",
        "model.layers.0.mlp.experts",
        "mlp.experts.0",
        r"re:.*mlp\.experts\.0\..*",
        r"re:.*mlp\.experts\..*\.down_proj",
    ],
)
def test_local_kimi_partial_expert_exclusions_fail(local_kimi, selector):
    from aisim.sdk.task_v2 import Task

    path, raw = local_kimi
    raw["quantization_config"]["ignore"].append(selector)
    (path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="partial or unrecognized routed-expert exclusions"):
        compile_engine(str(path), "b200_sxm", "sglang", "current")
    with pytest.raises(ValueError, match="partial or unrecognized routed-expert exclusions"):
        Task(model_path=str(path), system_name="b200_sxm", backend_name="sglang", total_gpus=1)
    blob = compile_engine(str(path), "b200_sxm", "sglang", "current", moe_quant_mode="bfloat16")
    assert json.loads(_native.engine_spec_json_from_bincode(blob))["engine"]["moe_dtype"] == "bfloat16"
    task = Task(
        model_path=str(path),
        system_name="b200_sxm",
        backend_name="sglang",
        total_gpus=1,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
    )
    assert task.build_model_config(role="agg").moe_quant_mode is common.MoEQuantMode.bfloat16


@pytest.mark.parametrize("scope", ["root", "text_config"])
@pytest.mark.parametrize("format_location", ["header", "group", "both", "group_override"])
def test_local_kimi_packed_format_layouts(local_kimi, scope, format_location):
    path, raw = local_kimi
    quant = raw["quantization_config"]
    if format_location != "header":
        quant["config_groups"]["experts"]["format"] = "mxfp4-pack-quantized"
    if format_location == "group":
        del quant["format"]
    elif format_location == "group_override":
        quant["format"] = "pack-quantized"
    if scope == "text_config":
        raw["text_config"] = {key: raw.pop(key) for key in list(raw) if key != "architectures"}
    (path / "config.json").write_text(json.dumps(raw))
    loaded = utils._load_model_config_from_model_path(str(path))
    assert utils._get_language_quantization_config(loaded) == quant
    blob = compile_engine(str(path), "b200_sxm", "sglang", "current")
    spec = json.loads(_native.engine_spec_json_from_bincode(blob))
    assert spec["engine"]["moe_dtype"] == "w4a8_mxfp4_mxfp8"
    moe_ops = [op["Moe"] for op in spec["context_ops"] + spec["generation_ops"] if "Moe" in op]
    assert moe_ops and all(op["quant_mode"] == "w4a8_mxfp4_mxfp8" for op in moe_ops)


@pytest.mark.parametrize("packed_format", ["nvfp4-pack-quantized", "pack-quantized", None])
@pytest.mark.parametrize("format_location", ["header", "group_override"])
def test_local_kimi_other_packed_formats_fail(local_kimi, packed_format, format_location):
    from aisim.sdk.task_v2 import Task

    path, raw = local_kimi
    quant = raw["quantization_config"]
    if format_location == "header":
        quant["format"] = packed_format
    else:
        quant["config_groups"]["experts"]["format"] = packed_format
        if packed_format is None:
            del quant["format"]
    (path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="4-bit float.*require format 'mxfp4-pack-quantized'"):
        compile_engine(str(path), "b200_sxm", "sglang", "current")
    with pytest.raises(ValueError, match="4-bit float.*require format 'mxfp4-pack-quantized'"):
        Task(model_path=str(path), system_name="b200_sxm", backend_name="sglang", total_gpus=1)


def test_local_kimi_mxfp4_label_requires_packed_format_metadata(local_kimi):
    path, raw = local_kimi
    raw["quantization_config"] = {"quant_method": "mxfp4"}
    (path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Kimi-K3 MXFP4 inference requires compressed-tensors format"):
        compile_engine(str(path), "b200_sxm", "sglang", "current")


def test_local_kimi_fp8_export_keeps_its_declared_precision(local_kimi):
    path, raw = local_kimi
    raw["quantization_config"] = {"quant_method": "fp8"}
    (path / "config.json").write_text(json.dumps(raw))
    blob = compile_engine(str(path), "b200_sxm", "sglang", "current")
    assert json.loads(_native.engine_spec_json_from_bincode(blob))["engine"]["moe_dtype"] == "fp8"


@pytest.mark.parametrize("explicit", [None, common.MoEQuantMode.bfloat16])
def test_local_kimi_sweeper_quant_resolution(local_kimi, explicit):
    from aisim.sdk.task_v2 import Task

    path, raw = local_kimi
    if explicit is not None:
        del raw["quantization_config"]
        (path / "config.json").write_text(json.dumps(raw))
    task = Task(
        model_path=str(path),
        system_name="b200_sxm",
        backend_name="sglang",
        moe_quant_mode=explicit,
        total_gpus=1,
    )
    assert task.build_model_config(role="agg").moe_quant_mode is (explicit or common.MoEQuantMode.w4a8_mxfp4_mxfp8)
