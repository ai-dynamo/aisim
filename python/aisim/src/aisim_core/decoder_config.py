# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read text decoder keys and cache-layout declarations from a checkpoint config.

FPM config intake, the FPM collector, the FPM profile check, the context lookup of replay and the naive KV-cache
estimator call decoder_config_key, so all five read one decoder.
LLM_CONFIG_WRAPPERS and TEXT_CONFIG_WRAPPERS name the wrappers whose text decoder is under llm_config and under
text_config. Config intake, the FPM collector, the FPM profile check and the context lookup of replay call
wrapper_key_problem and refuse a wrapper of either set that holds only the other decoder key.
The module imports only the standard library. Config intake must stay usable in a process that does not import
aisim_core.sdk. The cold-process tests in tests/unit/test_support_topology.py and
tests/unit/test_onboard_topology.py check that, and tests/unit/sdk/test_fpm_model_metadata.py checks it for a wrapper
in each set.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

WINDOW_FIELDS = ("sliding_window", "sliding_window_size")


# The wrappers whose text decoder is under llm_config, and those whose text decoder is under text_config.
# aisim_core.sdk.common builds MULTIMODAL_TEXT_CONFIG_KEY from these two sets, so the op-level parser and the callers
# of this module agree on the decoder key of every registered wrapper.
LLM_CONFIG_WRAPPERS = frozenset({"NemotronH_Omni_Reasoning_V3"})
TEXT_CONFIG_WRAPPERS = frozenset(
    {
        "DeepseekV41ForCausalLM",
        "Glm5NextForConditionalGeneration",
        "KimiK25ForConditionalGeneration",
        "KimiK3ForConditionalGeneration",
        # Step-3.7/3.5-Flash ship a vision tower and nest the whole decoder under
        # text_config; without this the parser reads the top level and rejects real
        # checkpoints for having no num_hidden_layers.
        "Step3p7ForConditionalGeneration",
        "Step3p7FlashForCausalLM",
        "Llama4ForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Gemma4ForConditionalGeneration",
        "MuseGlimmerForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
        "MiniMaxM3SparseForConditionalGeneration",
        "Mistral3ForConditionalGeneration",
    }
)


def decoder_config_key(document: Mapping[str, Any]) -> str | None:
    """Return the key that nests the text decoder in a checkpoint config, or None for a flat config.

    A wrapper in LLM_CONFIG_WRAPPERS, named by the first architecture, holds llm_config whenever the key is present,
    as the op-level parser does. Otherwise text_config is held whenever the key is present, and then llm_config only
    as a nonempty object, so a null, empty or non-object llm_config leaves the config flat. A held key is chosen even
    when its value is malformed, so the caller reports it. The function never raises.

    MULTIMODAL_TEXT_CONFIG_KEY maps exactly TEXT_CONFIG_WRAPPERS to text_config and LLM_CONFIG_WRAPPERS to llm_config,
    and tests/unit/sdk/test_fpm_model_metadata.py checks that.
    """
    architectures = document.get("architectures")
    wrapper = architectures[0] if isinstance(architectures, list) and architectures else None
    if isinstance(wrapper, str) and wrapper in LLM_CONFIG_WRAPPERS and "llm_config" in document:
        return "llm_config"
    if "text_config" in document:
        return "text_config"
    llm_config = document.get("llm_config")
    return "llm_config" if isinstance(llm_config, dict) and llm_config else None


def nested_decoder_ambiguity(key: str, decoder: Mapping[str, Any]) -> str | None:
    """Name the decoder key that a decoder config holds, or return None when it holds none.

    key is the result of decoder_config_key and decoder is the object under it. A decoder that holds a decoder key
    nests a second decoder, so no caller can tell which one to read. The held rule is that of decoder_config_key for a
    document that names no wrapper in LLM_CONFIG_WRAPPERS: text_config is held whenever the key is present, and
    llm_config only as a nonempty object. The decoder's own key comes first, then the other decoder key.

    The text is the diagnosis only: "ambiguous nested <key>" for the own key and "ambiguous nested <inner> in <key>"
    for the other key. Config intake adds its remedy after "; ", and so does the collector through
    decoder_config_problem. The remedies differ, because intake accepts a complete --fpm-profile and the collector
    does not. The FPM profile check and the context lookup of replay put their own text before the diagnosis and add
    no remedy. The function never raises and never reads the registry.
    """
    llm_config = decoder.get("llm_config")
    held = {"text_config": "text_config" in decoder, "llm_config": isinstance(llm_config, dict) and bool(llm_config)}
    if held.get(key):
        return f"ambiguous nested {key}"
    inner = "llm_config" if key == "text_config" else "text_config"
    return f"ambiguous nested {inner} in {key}" if held[inner] else None


def decoder_config_problem(key: str, decoder: object) -> str | None:
    """Name what keeps the value under a decoder key from describing one text decoder, or return None.

    key is the result of decoder_config_key and decoder is the value under it. The value must be a nonempty object
    that holds no decoder key. text_config, and the llm_config of a wrapper in LLM_CONFIG_WRAPPERS, are held whenever
    the key is present, so a null, empty or non-object value reaches this function. Any other llm_config is held only
    as a nonempty object, so it never fails the first check.

    The text is the diagnosis only: "<key> must be a nonempty object describing one text decoder", or the text of
    nested_decoder_ambiguity. The collector adds its remedy after "; ". The FPM profile check, the context lookup of
    replay and the naive KV-cache estimator put their own text before the diagnosis and add no remedy. Config intake
    keeps its own two checks, because it words the remedy of each differently. The function never raises.
    """
    if not isinstance(decoder, dict) or not decoder:
        return f"{key} must be a nonempty object describing one text decoder"
    return nested_decoder_ambiguity(key, decoder)


def wrapper_key_problem(document: Mapping[str, Any]) -> str | None:
    """Name a registered wrapper that holds only the other decoder key, or return None.

    The wrapper is the first architecture, as in decoder_config_key. Its own key is llm_config for a wrapper in
    LLM_CONFIG_WRAPPERS and text_config for a wrapper in TEXT_CONFIG_WRAPPERS. When the document lacks the own key and
    decoder_config_key holds the other one, a reader of that key and the op-level parser, which reads the wrapper root
    when the own key is missing, would see two decoders. A null or empty llm_config is not held, so the config of a
    text_config wrapper with such a value stays flat.

    The text is the diagnosis only. Config intake and the collector add their remedy after "; ", and the FPM profile
    check and the context lookup of replay put their own text before it. The function never raises and never reads
    the registry.
    """
    architectures = document.get("architectures")
    wrapper = architectures[0] if isinstance(architectures, list) and architectures else None
    if not isinstance(wrapper, str):
        return None
    if wrapper in LLM_CONFIG_WRAPPERS:
        own, other = "llm_config", "text_config"
    elif wrapper in TEXT_CONFIG_WRAPPERS:
        own, other = "text_config", "llm_config"
    else:
        return None
    if own in document or decoder_config_key(document) != other:
        return None
    return f"{wrapper} keeps its text decoder under {own}, but this config holds {other} and no {own}"


def layout_integer(value: Any, field: str, *, minimum: int = 1, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        bound = f"{minimum}..{maximum}" if maximum is not None else f">={minimum}"
        raise ValueError(f"{field} must be an integer {bound}, not a boolean, float or numeric string")
    return value


def cache_window(raw: Mapping[str, Any]) -> int | None:
    supplied = {key: layout_integer(raw[key], key, minimum=0) for key in WINDOW_FIELDS if raw.get(key) is not None}
    if len(set(supplied.values())) > 1:
        raise ValueError(f"conflicting sliding-window fields: {supplied}")
    return next(iter(supplied.values()), None)


def layer_types(raw: Mapping[str, Any], layers: int | None) -> list[str]:
    types = raw["layer_types"]
    if not isinstance(types, list) or (layers is not None and len(types) != layers):
        raise ValueError("layer_types must have one entry per num_hidden_layers")
    if any(not isinstance(value, str) or not value.strip() for value in types):
        raise ValueError("layer_types entries must be nonempty strings")
    return types
