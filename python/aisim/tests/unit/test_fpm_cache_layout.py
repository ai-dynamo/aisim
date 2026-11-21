# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cache layout check that config intake and the FPM collector share."""

from __future__ import annotations

import pytest

from aisim.fpm_cache_layout import cache_layout_markers, recognized_layer_types

pytestmark = pytest.mark.unit

_STATE_FIELDS = (
    "ssm_cfg",
    "mamba_d_state",
    "linear_conv_kernel_dim",
    "linear_attn_config",
    "ssm_state_size",
    "mamba_num_heads",
)


@pytest.mark.parametrize("architecture", [None, "LlamaForCausalLM", "Glm5NextForConditionalGeneration"])
def test_sparse_attention_is_recognized_only_for_the_dsa_decoders(architecture):
    assert recognized_layer_types(architecture) == {"full_attention", "sliding_attention"}
    for decoder in ("GlmMoeDsaForCausalLM", "DeepseekV32ForCausalLM"):
        assert recognized_layer_types(decoder) == {"full_attention", "sliding_attention", "deepseek_sparse_attention"}


@pytest.mark.parametrize(
    "raw,architecture,expected",
    [
        ({}, None, ()),
        ({"layer_types": ["full_attention", "sliding_attention"]}, None, ()),
        (
            {"layer_types": ["linear_attention", "full_attention", "linear_attention"]},
            None,
            ("layer_types linear_attention x2",),
        ),
        ({"layer_types": ["deepseek_sparse_attention"]}, "GlmMoeDsaForCausalLM", ()),
        # A multimodal wrapper is not a DSA decoder.
        (
            {"layer_types": ["deepseek_sparse_attention"]},
            "Glm5NextForConditionalGeneration",
            ("layer_types deepseek_sparse_attention x1",),
        ),
        # Malformed lists are structural errors that each caller raises itself; they are not markers.
        ({"layer_types": [1, "", None]}, None, ()),
        ({"layer_types": "linear_attention"}, None, ()),
        # 1 is full attention; every other integer is one reason per distinct value.
        ({"attn_type_list": [1, 1]}, None, ()),
        ({"attn_type_list": [0, 1, 0, 2]}, None, ("attn_type_list 0 x2", "attn_type_list 2 x1")),
        ({"attn_type_list": [True, "0", 1.5]}, None, ()),
        # Null state fields are absent, not markers.
        (dict.fromkeys(_STATE_FIELDS), None, ()),
        *(({field: 0}, None, (f"recurrent-state field {field}",)) for field in _STATE_FIELDS),
        # Architecture names match in any letter case and are reported as declared.
        ({}, "FalconMambaForCausalLM", ("recurrent architecture FalconMambaForCausalLM",)),
        ({}, "ToyJAMBAForCausalLM", ("recurrent architecture ToyJAMBAForCausalLM",)),
        ({}, "Rwkv7ForCausalLM", ("recurrent architecture Rwkv7ForCausalLM",)),
        # Every marker counts, also next to recognized layer names, and the reasons come back sorted.
        (
            {"layer_types": ["linear_attention", "full_attention"], "attn_type_list": [0, 1], "ssm_state_size": 128},
            "ToyMambaForCausalLM",
            (
                "attn_type_list 0 x1",
                "layer_types linear_attention x1",
                "recurrent architecture ToyMambaForCausalLM",
                "recurrent-state field ssm_state_size",
            ),
        ),
        # The rule for architectures that intake does not recognize is not part of this check.
        ({}, "MiMoForCausalLM", ()),
    ],
)
def test_cache_layout_markers(raw, architecture, expected):
    assert cache_layout_markers(raw, architecture) == expected


def test_marker_layout_declarations_have_one_implementation():
    from collector.fpm_forward import capabilities

    from aisim import fpm_cache_layout
    from aisim.support import config_profile

    assert config_profile._integer is fpm_cache_layout.layout_integer
    assert config_profile._cache_window is fpm_cache_layout.cache_window
    assert config_profile._layer_types is fpm_cache_layout.layer_types
    assert config_profile._attn_type_list is fpm_cache_layout.attn_type_list
    assert config_profile._INT_ALIASES["num_hidden_layers"] is fpm_cache_layout.LAYER_COUNT_FIELDS
    assert config_profile.check_marker_layout_declarations is fpm_cache_layout.check_marker_layout_declarations
    assert capabilities.check_marker_layout_declarations is fpm_cache_layout.check_marker_layout_declarations
    assert not {"_WINDOW_FIELDS", "_LAYER_COUNT_FIELDS", "_intake_integer"} & vars(capabilities).keys()


_NOT_AN_INTEGER = "must be an integer >={}, not a boolean, float or numeric string"


@pytest.mark.parametrize(
    "raw,error",
    [
        ({"sliding_window": "128"}, "sliding_window " + _NOT_AN_INTEGER.format(0)),
        (
            {"sliding_window": 64, "sliding_window_size": 128},
            "conflicting sliding-window fields: {'sliding_window': 64, 'sliding_window_size': 128}",
        ),
        ({"num_layers": 0}, "num_layers " + _NOT_AN_INTEGER.format(1)),
        (
            {"num_hidden_layers": 2, "n_layer": 3},
            "conflicting num_hidden_layers fields: {'num_hidden_layers': 2, 'n_layer': 3}",
        ),
        ({"num_hidden_layers": 2, "layer_types": ["a"]}, "layer_types must have one entry per num_hidden_layers"),
        ({"layer_types": ["a", ""]}, "layer_types entries must be nonempty strings"),
        ({"n_layer": 2, "attn_type_list": [1]}, "attn_type_list must have one entry per num_hidden_layers"),
        ({"attn_type_list": [1, "0"]}, "attn_type_list entries must be integers"),
        # A list is required also when no layer count states its length.
        ({"layer_types": "ab"}, "layer_types must have one entry per num_hidden_layers"),
        ({"attn_type_list": "01"}, "attn_type_list must have one entry per num_hidden_layers"),
        # The window is checked before the layer count.
        ({"sliding_window": True, "num_hidden_layers": "2"}, "sliding_window " + _NOT_AN_INTEGER.format(0)),
        # layer_types is checked before attn_type_list.
        ({"layer_types": ["a", ""], "attn_type_list": [1, "0"]}, "layer_types entries must be nonempty strings"),
    ],
)
def test_marker_layout_declarations_refuse_each_malformed_declaration(raw, error):
    from aisim.fpm_cache_layout import check_marker_layout_declarations

    with pytest.raises(ValueError) as refusal:
        check_marker_layout_declarations(raw)
    assert str(refusal.value) == error


def test_marker_layout_declarations_return_the_layer_count():
    from aisim.fpm_cache_layout import check_marker_layout_declarations

    raw = {"sliding_window": None, "sliding_window_size": 0, "num_layers": 3, "layer_types": ["a", "b", "c"]}
    assert check_marker_layout_declarations(raw) == 3
    assert check_marker_layout_declarations({"attn_type_list": [1, 0]}) is None


def test_intake_and_collector_run_the_shared_marker_block(tmp_path, monkeypatch):
    # Each side calls the one function for a config with a layout marker, and neither repeats its four checks.
    import json

    from collector.fpm_forward import capabilities

    from aisim.support import config_profile

    calls = []

    def record(side):
        def checked(raw):
            calls.append(side)

        return checked

    monkeypatch.setattr(config_profile, "check_marker_layout_declarations", record("intake"))
    monkeypatch.setattr(capabilities, "check_marker_layout_declarations", record("collector"))
    config = {
        "architectures": ["MiniMaxText01ForCausalLM"],
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "intermediate_size": 256,
        "vocab_size": 1024,
        "max_position_embeddings": 4096,
        "torch_dtype": "bfloat16",
        "attn_type_list": [0, 1],
    }

    def load_both(declarations):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({**config, **declarations}))
        config_profile.load_model_config(path)
        capabilities.resolve_model_capability(
            backend="vllm",
            model_path=str(path),
            model_architecture=None,
            selected_ops={"attention_context", "attention_generation"},
            has_model_cases=False,
            system="b200_sxm",
            requested_weight_quantizations=(),
            requested_kv_cache_dtypes=("auto",),
            database_version="0.27.0",
            checkpoint_native_dtypes=True,
        )

    load_both({})
    assert calls == ["intake", "collector"]
    # A malformed layer_types that only the shared checks refuse: neither side keeps a copy of them.
    calls.clear()
    load_both({"layer_types": ["linear_attention"]})
    assert calls == ["intake", "collector"]
