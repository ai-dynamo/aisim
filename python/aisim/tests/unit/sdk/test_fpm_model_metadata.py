# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The decoder key that FPM config intake, the FPM collector and the FPM profile check share.

The file also pins the nested-decoder rule: a decoder that holds a decoder key is ambiguous. A decoder under the key
must also be a nonempty object. A registered wrapper that holds only the other decoder key is refused too.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from aisim_core.decoder_config import (
    LLM_CONFIG_WRAPPERS,
    TEXT_CONFIG_WRAPPERS,
    decoder_config_problem,
    nested_decoder_ambiguity,
    wrapper_key_problem,
)
from aisim_core.sdk.common import MULTIMODAL_TEXT_CONFIG_KEY
from aisim_core.sdk.fpm_model_metadata import decoder_config_key

pytestmark = pytest.mark.unit

_DECODER = {"hidden_size": 16}
_BOTH_KEYS = {"text_config": _DECODER, "llm_config": _DECODER}
# MULTIMODAL_TEXT_CONFIG_KEY registers llm_config for this wrapper (#562).
_VL = "NemotronH_Omni_Reasoning_V3"
_WRAPPER = "ExampleForConditionalGeneration"


@pytest.mark.parametrize(
    "document,key",
    [
        ({"architectures": [_VL], "llm_config": _DECODER}, "llm_config"),
        ({"architectures": [_VL], "llm_config": _DECODER, "text_config": _DECODER}, "llm_config"),
        ({"architectures": [_VL], "llm_config": None, "text_config": _DECODER}, "llm_config"),
        ({"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": _DECODER}, "text_config"),
        ({"architectures": [_WRAPPER], "text_config": _DECODER, "llm_config": _DECODER}, "text_config"),
        # A malformed text_config is still chosen, so intake keeps reporting it.
        ({"architectures": [_WRAPPER], "text_config": None}, "text_config"),
        ({"architectures": ["InternVLChatModel"], "llm_config": _DECODER}, "llm_config"),
        # A registered wrapper holds llm_config whenever the key is present, as the op-level parser does.
        ({"architectures": [_VL], "llm_config": None}, "llm_config"),
        ({"architectures": [_VL], "llm_config": {}}, "llm_config"),
        # Without a registered wrapper, llm_config counts only as a nonempty object, so it adds no error.
        ({"llm_config": [_DECODER]}, None),
        ({"llm_config": "decoder"}, None),
        # A malformed wrapper architecture never raises here.
        ({"architectures": _VL, "llm_config": _DECODER}, "llm_config"),
        ({"architectures": [{"name": _VL}], "llm_config": _DECODER}, "llm_config"),
        ({"architectures": [], "text_config": _DECODER}, "text_config"),
        # With both keys, a malformed architectures value must neither raise nor select llm_config, and a wrapper
        # registered to text_config keeps text_config.
        ({"architectures": _VL, **_BOTH_KEYS}, "text_config"),
        ({"architectures": [{"name": _VL}], **_BOTH_KEYS}, "text_config"),
        ({"architectures": [], **_BOTH_KEYS}, "text_config"),
        ({"architectures": {"0": _VL}, **_BOTH_KEYS}, "text_config"),
        ({"architectures": ["Qwen3_5ForConditionalGeneration"], **_BOTH_KEYS}, "text_config"),
        ({"architectures": ["NemotronHForCausalLM"], "mamba_num_heads": 128}, None),
        ({}, None),
    ],
    ids=[
        "registered-llm-config",
        "registered-key-before-text-config",
        "registered-key-not-an-object",
        "registered-text-config",
        "text-config-before-llm-config",
        "malformed-text-config",
        "unregistered-llm-config",
        "registered-null-llm-config",
        "registered-empty-llm-config",
        "list-llm-config",
        "string-llm-config",
        "string-architectures",
        "object-architecture-entry",
        "empty-architectures",
        "both-keys-string-architectures",
        "both-keys-object-architecture-entry",
        "both-keys-empty-architectures",
        "both-keys-dict-architectures",
        "both-keys-wrapper-registered-to-text-config",
        "flat",
        "empty",
    ],
)
def test_decoder_config_key(document, key):
    assert decoder_config_key(document) == key


# The key of each registered wrapper, stated apart from the two sets that the registry is built from.
_WRAPPER_KEYS = {
    "DeepseekV41ForCausalLM": "text_config",
    "Gemma4ForConditionalGeneration": "text_config",
    "Glm5NextForConditionalGeneration": "text_config",
    "KimiK25ForConditionalGeneration": "text_config",
    "KimiK3ForConditionalGeneration": "text_config",
    "Llama4ForConditionalGeneration": "text_config",
    "MiniMaxM3SparseForConditionalGeneration": "text_config",
    "Mistral3ForConditionalGeneration": "text_config",
    "MuseGlimmerForConditionalGeneration": "text_config",
    "NemotronH_Omni_Reasoning_V3": "llm_config",
    "Qwen3VLForConditionalGeneration": "text_config",
    "Qwen3VLMoeForConditionalGeneration": "text_config",
    "Qwen3_5ForConditionalGeneration": "text_config",
    "Qwen3_5MoeForConditionalGeneration": "text_config",
    "Step3p7FlashForCausalLM": "text_config",
    "Step3p7ForConditionalGeneration": "text_config",
}


def test_the_registry_maps_wrappers_only_to_keys_that_the_helper_orders():
    # The helper orders only these two keys, so a third key would go unseen. It reads the llm_config wrappers from
    # LLM_CONFIG_WRAPPERS, so the registry must map exactly those wrappers to llm_config.
    assert set(MULTIMODAL_TEXT_CONFIG_KEY.values()) <= {"text_config", "llm_config"}
    assert {name for name, key in MULTIMODAL_TEXT_CONFIG_KEY.items() if key == "llm_config"} == LLM_CONFIG_WRAPPERS
    assert {name for name, key in MULTIMODAL_TEXT_CONFIG_KEY.items() if key == "text_config"} == TEXT_CONFIG_WRAPPERS
    assert not LLM_CONFIG_WRAPPERS & TEXT_CONFIG_WRAPPERS
    assert MULTIMODAL_TEXT_CONFIG_KEY == _WRAPPER_KEYS


def test_the_decoder_key_of_a_registered_wrapper_loads_no_sdk_module():
    # FPM config intake must stay usable in a process that does not import aisim_core.sdk.
    probe = (
        "import json, sys\n"
        "from aisim_core.decoder_config import decoder_config_key\n"
        "assert decoder_config_key(json.loads(sys.argv[1])) == 'llm_config'\n"
        "loaded = [name for name in sys.modules if name.startswith('aisim_core.sdk')]\n"
        "assert not loaded, loaded\n"
    )
    document = json.dumps({"architectures": [_VL], **_BOTH_KEYS})
    result = subprocess.run([sys.executable, "-c", probe, document], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


_QWEN = "Qwen3_5MoeForConditionalGeneration"


@pytest.mark.parametrize(
    ("document", "problem"),
    [
        (
            {"architectures": [_VL], "text_config": _DECODER},
            f"{_VL} keeps its text decoder under llm_config, but this config holds text_config and no llm_config",
        ),
        (
            {"architectures": [_VL], "text_config": None},
            f"{_VL} keeps its text decoder under llm_config, but this config holds text_config and no llm_config",
        ),
        (
            {"architectures": [_QWEN], "llm_config": _DECODER},
            f"{_QWEN} keeps its text decoder under text_config, but this config holds llm_config and no text_config",
        ),
        # A null or empty llm_config is not held, so the config of a text_config wrapper stays flat.
        ({"architectures": [_QWEN], "llm_config": None}, None),
        ({"architectures": [_QWEN], "llm_config": {}}, None),
        ({"architectures": [_QWEN], "hidden_size": 16}, None),
        ({"architectures": [_VL], **_BOTH_KEYS}, None),
        ({"architectures": [_QWEN], **_BOTH_KEYS}, None),
        ({"architectures": [_WRAPPER], "llm_config": _DECODER}, None),
        ({"architectures": _VL, "text_config": _DECODER}, None),
        ({"text_config": _DECODER}, None),
        # The wrapper is the first architecture.
        (
            {"architectures": [_QWEN, _WRAPPER], "llm_config": _DECODER},
            f"{_QWEN} keeps its text decoder under text_config, but this config holds llm_config and no text_config",
        ),
        ({"architectures": [_WRAPPER, _QWEN], "llm_config": _DECODER}, None),
        # A first architecture that is no string names no wrapper.
        ({"architectures": [{"name": _VL}], "text_config": _DECODER}, None),
    ],
)
def test_wrapper_key_problem(document, problem):
    assert wrapper_key_problem(document) == problem


def test_every_registered_wrapper_refuses_the_other_decoder_key():
    assert len(_WRAPPER_KEYS) == 16
    for wrapper, own in _WRAPPER_KEYS.items():
        other = "llm_config" if own == "text_config" else "text_config"
        assert wrapper_key_problem({"architectures": [wrapper], other: _DECODER}) == (
            f"{wrapper} keeps its text decoder under {own}, but this config holds {other} and no {own}"
        )


def test_the_wrapper_key_problem_of_a_registered_wrapper_loads_no_sdk_module():
    # FPM config intake must stay usable in a process that does not import aisim_core.sdk.
    probe = (
        "import json, sys\n"
        "from aisim_core.decoder_config import wrapper_key_problem\n"
        "assert wrapper_key_problem(json.loads(sys.argv[1])) == sys.argv[2]\n"
        "loaded = [name for name in sys.modules if name.startswith('aisim_core.sdk')]\n"
        "assert not loaded, loaded\n"
    )
    document = json.dumps({"architectures": [_QWEN], "llm_config": _DECODER})
    problem = f"{_QWEN} keeps its text decoder under text_config, but this config holds llm_config and no text_config"
    result = subprocess.run(
        [sys.executable, "-c", probe, document, problem], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "key,decoder,ambiguity",
    [
        ("text_config", {"text_config": _DECODER}, "ambiguous nested text_config"),
        ("text_config", {"text_config": None}, "ambiguous nested text_config"),
        ("llm_config", {"llm_config": _DECODER}, "ambiguous nested llm_config"),
        ("text_config", {"llm_config": _DECODER}, "ambiguous nested llm_config in text_config"),
        ("llm_config", {"text_config": _DECODER}, "ambiguous nested text_config in llm_config"),
        ("llm_config", {"text_config": None}, "ambiguous nested text_config in llm_config"),
        # The decoder's own key comes first, so intake keeps its text of today.
        ("text_config", _BOTH_KEYS, "ambiguous nested text_config"),
        ("llm_config", _BOTH_KEYS, "ambiguous nested llm_config"),
        # llm_config is held only as a nonempty object, as decoder_config_key holds it.
        ("llm_config", {"llm_config": None}, None),
        ("llm_config", {"llm_config": {}}, None),
        ("text_config", {"llm_config": [_DECODER]}, None),
        ("text_config", {"llm_config": "decoder"}, None),
        ("text_config", _DECODER, None),
        ("llm_config", _DECODER, None),
    ],
    ids=[
        "own-text-config",
        "own-null-text-config",
        "own-llm-config",
        "llm-config-in-text-config",
        "text-config-in-llm-config",
        "null-text-config-in-llm-config",
        "both-keys-in-text-config",
        "both-keys-in-llm-config",
        "own-null-llm-config",
        "own-empty-llm-config",
        "list-llm-config-in-text-config",
        "string-llm-config-in-text-config",
        "flat-text-config",
        "flat-llm-config",
    ],
)
def test_nested_decoder_ambiguity(key, decoder, ambiguity):
    assert nested_decoder_ambiguity(key, decoder) == ambiguity


_NOT_A_DECODER = "text_config must be a nonempty object describing one text decoder"


@pytest.mark.parametrize(
    "key,decoder,problem",
    [
        # text_config is held whenever the key is present, so a value that is no nonempty object reaches the helper.
        ("text_config", None, _NOT_A_DECODER),
        ("text_config", [], _NOT_A_DECODER),
        ("text_config", "decoder", _NOT_A_DECODER),
        ("text_config", {}, _NOT_A_DECODER),
        ("text_config", 0, _NOT_A_DECODER),
        # A nonempty object is checked for a nested decoder, and the diagnosis stays that of nested_decoder_ambiguity.
        ("llm_config", {"text_config": _DECODER}, "ambiguous nested text_config in llm_config"),
        ("text_config", _DECODER, None),
        ("llm_config", _DECODER, None),
    ],
    ids=[
        "null",
        "list",
        "string",
        "empty-object",
        "number",
        "text-config-in-llm-config",
        "flat-text-config",
        "flat-llm-config",
    ],
)
def test_decoder_config_problem(key, decoder, problem):
    assert decoder_config_problem(key, decoder) == problem


# decoder_config_key and nested_decoder_ambiguity each decide whether a value under llm_config or text_config holds a
# decoder. The two tests below fail when those two rules drift apart.
_KEY_VALUES = [
    pytest.param(None, id="null"),
    pytest.param({}, id="empty-object"),
    pytest.param([], id="empty-list"),
    pytest.param("decoder", id="string"),
    pytest.param(0, id="number"),
    pytest.param(_DECODER, id="decoder"),
]


@pytest.mark.parametrize("value", _KEY_VALUES)
def test_both_helpers_hold_an_llm_config_by_one_rule(value):
    picked = decoder_config_key({"llm_config": value}) == "llm_config"
    assert (nested_decoder_ambiguity("text_config", {"llm_config": value}) is not None) == picked


@pytest.mark.parametrize("value", _KEY_VALUES)
def test_both_helpers_hold_a_text_config_by_one_rule(value):
    picked = decoder_config_key({"text_config": value}) == "text_config"
    assert (nested_decoder_ambiguity("llm_config", {"text_config": value}) is not None) == picked
