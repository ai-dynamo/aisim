# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
# SPDX-FileCopyrightText: Copyright 2025-2026 the HuggingFace Team
# SPDX-FileCopyrightText: Copyright the safetensors contributors
# SPDX-License-Identifier: Apache-2.0
# Adapted grouping: https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/kv_cache_planning.py
# Step geometry: https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/configs/model_config.py
# Checkpoint KV: https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/utils/torch_utils.py
# Checkpoint KV: https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/mem_cache/kv_cache_dtype.py
# GPT-OSS defaults: https://github.com/huggingface/transformers/blob/eeee11099135592f1d03c2871fbd3d8132e5a4d7/src/transformers/models/gpt_oss/configuration_gpt_oss.py
# Gemma defaults: https://github.com/huggingface/transformers/blob/eeee11099135592f1d03c2871fbd3d8132e5a4d7/src/transformers/models/gemma4/configuration_gemma4.py
# Historical group-size rules and modifications are listed in THIRD_PARTY_NOTICES.md.

"""KV-cache capacity estimation.

This module implements scalar rank-local KV-cache capacity estimation. The
legacy Rust ``aisim_core::memory::estimate_kv_cache`` forwards here and
rebuilds the scalar result. Grouped FPM profiles delegate budget arithmetic,
page rounding and retention footprints to the canonical Rust
``ForwardPassPerfModelConfig::estimate_cache_budget`` API; they have no scalar
token capacity.

Resource sources:

- **FPM profile**: caller-declared rank-local resource bounds bypass model
  construction and timing data. Linear profiles use the scalar arithmetic below;
  grouped profiles use the native cache-budget API. Both check the scheduler
  against the declared resource envelope.

- **Native** (:class:`KVCacheEstimator`): ``from_request`` reuses AIC's full
  backend memory model (``BaseBackend._get_memory_usage`` plus the model's
  MLA-aware per-token KV size), then ``estimate`` does the OfFree (TRT-LLM,
  fraction of FREE memory) / OfTotal (vLLM/SGLang, fraction of TOTAL memory)
  budget math.
- **Naive fallback** (:class:`NaiveKVCacheEstimator`): for models/backends AIC
  cannot build natively, ``from_model_path`` parses the HF ``config.json``
  (reusing the SDK's existing loaders +
  :func:`~aisim_core.sdk.utils._parse_hf_config_json`) and ``estimate``
  reserves a fraction (default 80%) of post-weight memory for KV. It handles
  standard and compressed attention. Shared scheduler sizing charges in-flight
  tokens on every attention layer. DeepSeek V4 exposes marginal token bytes;
  its packed groups require an allocator to determine capacity.

The scalar budget math is factored into :func:`kv_cache_budget_bytes` so the deferred
consolidation of ``InferenceSummary._check_and_set_kv_cache_oom`` onto a single
formula is a small follow-up (tracked in #1208).
"""

from __future__ import annotations

import enum
import json
import math
import os
import re
from collections.abc import Callable, Collection
from fnmatch import fnmatchcase
from functools import cache, partial
from importlib.resources import files
from itertools import pairwise
from pathlib import Path
from typing import Any

from packaging.version import Version

from aisim_core.data.ensure import MissingDataError
from aisim_core.decoder_config import (
    FPM_DECODER_ARCHITECTURES,
    attn_type_list,
    cache_layout_markers,
    cache_window,
    decoder_config_key,
    decoder_config_problem,
    layer_types,
    layout_integer,
)
from aisim_core.sdk import perf_database
from aisim_core.sdk.backends.factory import get_backend
from aisim_core.sdk.common import (
    ARCHITECTURE_TO_MODEL_FAMILY,
    MULTIMODAL_TEXT_CONFIG_KEY,
    CommQuantMode,
    DeepSeekV4Config,
    DefaultHFModels,
    FMHAQuantMode,
    Gemma4MixConfig,
    GEMMQuantMode,
    KVCacheQuantMode,
    MoEQuantMode,
    indexer_cache_entry_bytes,
    parse_support_matrix_version,
)
from aisim_core.sdk.config import validate_parallel_size
from aisim_core.sdk.config_builders import apply_nextn, build_model_config, validate_moe_controls, validate_nextn
from aisim_core.sdk.fpm_model_metadata import attach_quantization_sidecar, inherit_decoder_precision
from aisim_core.sdk.fpm_profile import FpmModelProfile, _load_fpm_timing_profile
from aisim_core.sdk.memory.vllm_registry import (
    default_causal_lm,
    has_remote_model,
    is_known_transformers_config,
    select_architecture,
    select_wrapper_architecture,
    wrapper_runtime,
)
from aisim_core.sdk.models import get_model
from aisim_core.sdk.models.base import _MODEL_REGISTRY, BaseModel
from aisim_core.sdk.models.deepseek_v32 import _dsa_full_layer_fraction
from aisim_core.sdk.models.gemma4 import UnverifiedGemmaRuntimeError, check_gemma4_adapter_runtime
from aisim_core.sdk.models.helpers import (
    _apply_model_quant_defaults,
    _infer_quant_modes_from_raw_config,
    resolve_sglang_mla_compute,
    resolve_trtllm_mla_compute,
)
from aisim_core.sdk.utils import (
    GEMMA4_DEFAULT_INTERMEDIATE_SIZE,
    _attach_inferred_quant_fields,
    _download_hf_config,
    _download_hf_json,
    _load_local_config,
    _load_local_quant_config,
    _load_model_config_from_model_path,
    _load_pre_downloaded_hf_config,
    _load_pre_downloaded_hf_quant_config,
    _parse_hf_config_json,
    fp8_weight_block_shape,
    gemma4_cache_requires_release,
    gemma4_cache_shapes,
    validate_fp8_moe_shard,
)

# Text-only identities supported by the SDK in addition to the FPM intake architectures.
_SELECTED_DECODER_ARCHITECTURES = {
    **FPM_DECODER_ARCHITECTURES,
    "gpt_oss": "GptOssForCausalLM",
    "kimi_k2": "DeepseekV3ForCausalLM",
    "mimo": "MiMoForCausalLM",
    "mimo_v2_flash": "MiMoV2FlashForCausalLM",
    "opt": "OPTForCausalLM",
    "qwen2_moe": "Qwen2MoeForCausalLM",
    "step3p5": "Step3p5ForCausalLM",
    "step3p7": "Step3p5ForCausalLM",
}

_VLLM_DENSE_DECODERS = {
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "OPTForCausalLM",
}
_TEXT_ONLY_DECODERS = set(_SELECTED_DECODER_ARCHITECTURES.values()) | _VLLM_DENSE_DECODERS

_TEXT_CONFIG_DEFAULTS = {
    "InternS1ForConditionalGeneration": ("interns1", "qwen3"),
    "InternVLForConditionalGeneration": ("internvl", "qwen2"),
    "LlavaForConditionalGeneration": ("llava", "llama"),
    "LlavaNextForConditionalGeneration": ("llava_next", "llama"),
    "LlavaNextVideoForConditionalGeneration": ("llava_next_video", "llama"),
    "LlavaOnevisionForConditionalGeneration": ("llava_onevision", "qwen2"),
}
_TEXT_DEFAULTS_BY_MODEL_TYPE = {
    **dict(_TEXT_CONFIG_DEFAULTS.values()),
    "idefics2": "mistral",
    "idefics3": "llama",
    "smolvlm": "llama",
}

_FIXED_LLAMA_WRAPPERS = {"Idefics3ForConditionalGeneration", "SmolVLMForConditionalGeneration"}
_TEXT_WRAPPERS = _TEXT_CONFIG_DEFAULTS.keys() | _FIXED_LLAMA_WRAPPERS
_VLLM_NO_FP16_MODEL_TYPES = ("gemma2", "gemma3", "gemma3_text", "glm4")

_VLLM_FP8_WEIGHT_DTYPES = {
    "F8_E4M3": "float8_e4m3fn",
    "F8_E4M3FNUZ": "float8_e4m3fnuz",
    "F8_E5M2": "float8_e5m2",
    "F8_E5M2FNUZ": "float8_e5m2fnuz",
}


# A missing field in a known Transformers config can have a fixed class default.
# An explicit null still follows the model's own derived-dimension path.
_DECODER_HEAD_DEFAULTS = {
    # model_type: (head_dim, num_key_value_heads)
    "gemma3_text": (256, 4),
    "gemma4_text": (256, 4),
    "glm4_moe": (None, 8),
    "gpt_oss": (64, 8),
    "llama": (None, None),
    "llama4_text": (128, 8),
    "minimax_m2": (128, 8),
    "mistral": (None, 8),
    "mixtral": (None, 8),
    "qwen2": (None, 32),
    "qwen2_moe": (None, 16),
    "qwen3": (128, 32),
    "qwen3_moe": (None, 4),
    "qwen3_vl_text": (128, 32),
    "qwen3_vl_moe_text": (None, 16),
}


# Structural defaults of the selected Transformers constructors used for KV and weight sizing.
# Both Kimi-K2.5 serving loaders instantiate its kimi_k2 text config as DeepseekV3Config.
_DECODER_GEOMETRY_DEFAULTS = {
    "llama": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "intermediate_size": 11008},
    "mistral": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "intermediate_size": 14336},
    "opt": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "ffn_dim": 3072,
        "max_position_embeddings": 2048,
        "vocab_size": 50272,
    },
    "qwen2": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "intermediate_size": 22016},
    "qwen3": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "intermediate_size": 22016},
    "deepseek_v3": {
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "intermediate_size": 18432,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
        "first_k_dense_replace": 3,
    },
}


# These parent configurations always construct their own text class, including for a sparse child dictionary.
_QWEN3_VL_TEXT_DEFAULTS = {
    "qwen3_vl": {"vocab_size": 151936, **_DECODER_GEOMETRY_DEFAULTS["qwen3"]},
    "qwen3_vl_moe": {
        "vocab_size": 151936,
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 5632,
    },
}


# Config classes that can also provide the attributes of an explicitly selected Qwen2 or OPT.
_BORROWED_DECODER_DEFAULTS = {
    "phi3": dict(
        hidden_size=3072,
        num_hidden_layers=32,
        num_attention_heads=32,
        intermediate_size=8192,
        hidden_act="silu",
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        vocab_size=32064,
    ),
    "ministral3": dict(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        num_hidden_layers=34,
        intermediate_size=14336,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        max_position_embeddings=262144,
        vocab_size=131072,
    ),
    "qwen2_moe": dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=16,
        num_hidden_layers=24,
        intermediate_size=5632,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        max_position_embeddings=32768,
        vocab_size=151936,
    ),
    "gemma": dict(
        hidden_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=256,
        intermediate_size=24576,
        hidden_act="gelu_pytorch_tanh",
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        vocab_size=256000,
    ),
    "gemma3_text": dict(
        hidden_size=2304,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        intermediate_size=9216,
        max_position_embeddings=131072,
        rms_norm_eps=1e-6,
        vocab_size=262208,
    ),
}


def _deepseek_routed_layers(decoder: dict, layers: int) -> range:
    first = layout_integer(decoder.get("first_k_dense_replace", 0), "first_k_dense_replace", minimum=0)
    frequency = layout_integer(decoder.get("moe_layer_freq", 1), "moe_layer_freq")
    return range(-(-first // frequency) * frequency, layers, frequency)


def _apply_decoder_geometry_defaults(decoder: dict, constructor: str | None) -> None:
    for field, default in _DECODER_GEOMETRY_DEFAULTS.get(constructor, {}).items():
        value = decoder.setdefault(field, default)
        if value is None and field in ("q_lora_rank", "num_key_value_heads"):
            continue
        layout_integer(value, field, minimum=0 if field == "first_k_dense_replace" else 1)
    head_dim, kv_heads = _DECODER_HEAD_DEFAULTS.get(constructor, (None, None))
    if head_dim is not None:
        decoder.setdefault("head_dim", head_dim)
    if kv_heads is not None:
        decoder.setdefault("num_key_value_heads", kv_heads)


@cache
def _sglang_causal_lms() -> dict[int, dict[str, str]]:
    releases: dict[int, dict[str, str]] = {}
    with files(__package__).joinpath("sglang_text_architectures.txt").open(encoding="utf-8") as source:
        for line in source:
            if line.startswith("#"):
                continue
            first, last, *pairs = line.split()
            mapping = dict(pair.split("=", 1) for pair in pairs)
            for patch in range(int(first), int(last) + 1):
                releases.setdefault(patch, {}).update(mapping)
    return releases


def _sglang_llava_causal_lm(kind: str, version: str | None) -> str | None:
    # LLaVA maps the concrete Transformers config class to AutoModelForCausalLM under SGLang's own dependency pin.
    release = parse_support_matrix_version(version)
    mappings = _sglang_causal_lms()
    if (
        release is not None
        and release.release[:2] == (0, 5)
        and not release.is_prerelease
        and release.micro in mappings
    ):
        return mappings[release.micro].get(kind)
    choices = {mapping.get(kind) for mapping in mappings.values()}
    if len(choices) != 1:
        raise ValueError(f"SGLang's LLaVA decoder for {kind!r} depends on the backend release")
    return choices.pop()


@cache
def _sglang_model_registries() -> dict[int, dict[str, set[str]]]:
    releases: dict[int, dict[str, set[str]]] = {}
    with files(__package__).joinpath("sglang_multimodal_architectures.txt").open(encoding="utf-8") as source:
        for line in source:
            if not line.startswith("#"):
                kind, first, last, *names = line.split()
                if kind == "compatible":
                    suffix, *prefixes = names
                    names = [prefix + suffix if suffix != "-" else prefix for prefix in prefixes]
                for patch in range(int(first), int(last) + 1):
                    releases.setdefault(patch, {}).setdefault(kind, set()).update(names)
    return releases


def _sglang_wrapper(
    names: list[str], version: str | None, document: dict, decoder: dict
) -> tuple[str | None, bool, bool]:
    """The first registered entrypoint wins. Only a known text decoder establishes decoder-only residency."""
    auto_map = document.get("auto_map")
    remote = has_remote_model(auto_map)
    if not names:
        return None, remote, False
    release = parse_support_matrix_version(version)
    registries = _sglang_model_registries()
    if (
        release is not None
        and release.micro in registries
        and release.public == ("0.5.6.post2" if release.micro == 6 else f"0.5.{release.micro}")
    ):
        candidates = (registries[release.micro],)
    else:
        candidates = registries.values()
    choices = set()
    needs_weights = False
    transformers = True
    for entries in candidates:
        native = next((name for name in names if name in entries["native"]), None)
        choices.add(native or names[0])
        if native is None:
            fallback = bool(entries["transformers"].intersection(names))
            remote_auto = isinstance(auto_map, dict) and "AutoModel" in auto_map
            if (
                fallback
                and not remote_auto
                and any(name not in entries["transformers"] and name not in entries["compatible"] for name in names)
            ):
                raise ValueError(
                    "SGLang Transformers fallback requires a compatible model class for every architecture"
                )
            fallback |= remote_auto
            needs_weights |= fallback or remote or bool(entries["multimodal"].intersection(names))
        else:
            fallback = native.startswith("Transformers")
            needs_weights |= native in entries["multimodal"] or native not in _TEXT_ONLY_DECODERS
        if fallback and "TransformersMultiModalForCausalLM" not in entries["native"]:
            # SGLang 0.5.6 through 0.5.9 constructs its generic attention layers and output head from the parent.
            kind = decoder.get("model_type")
            if isinstance(kind, str):
                _apply_decoder_geometry_defaults(decoder, kind)
            if kind == "mistral":
                decoder.setdefault("vocab_size", 32000)
                if decoder.get("num_key_value_heads") is None:
                    decoder["num_key_value_heads"] = decoder["num_attention_heads"]
            dimensions = (
                "vocab_size",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
            )
            if any(document.get(k) is None or document[k] != decoder.get(k) for k in dimensions):
                raise ValueError(
                    "sizing a legacy SGLang Transformers model requires matching parent and decoder dimensions"
                )
            size = layout_integer(document["hidden_size"], "hidden_size") // layout_integer(
                document["num_attention_heads"], "num_attention_heads"
            )
            if document.get("head_dim", size) != (decoder.get("head_dim") or size):
                raise ValueError(
                    "sizing a legacy SGLang Transformers model requires matching parent and decoder head_dim"
                )
        transformers &= fallback
    if len(choices) == 1:
        return choices.pop(), needs_weights, transformers
    if choices & _TEXT_WRAPPERS:
        raise ValueError(f"SGLang's selected wrapper for {names!r} depends on the backend release")
    return names[0], needs_weights, transformers


def _uses_bare_decoder(config: dict, backend: str, version: str | None) -> bool:
    kind = config.get("model_type")
    if not isinstance(kind, str):
        return False
    if kind in _DECODER_GEOMETRY_DEFAULTS:
        return True
    # A bare HF config can provide the attributes for a different registered vLLM decoder.
    names = config.get("architectures")
    return (
        backend == "vllm"
        and isinstance(names, list)
        and bool(names)
        and all(isinstance(name, str) for name in names)
        and is_known_transformers_config(kind, version)
        and select_architecture(names, version, config.get("auto_map"), modeled=_VLLM_DENSE_DECODERS) is not None
    )


def _select_text_decoder(
    decoder: dict,
    key: str | None,
    parent: str | None,
    parent_type: str | None,
    backend: str,
    backend_version: str | None,
) -> tuple[str | None, str | None]:
    """Resolve the Transformers config class and the model class selected by its serving wrapper."""
    registered_submodel = key == "text_config" and parent in _TEXT_CONFIG_DEFAULTS
    sglang_llava = key == "text_config" and backend == "sglang" and parent == "LlavaForConditionalGeneration"
    fixed_llama = key == "text_config" and parent in _FIXED_LLAMA_WRAPPERS
    kind = decoder.get("model_type")
    if registered_submodel or fixed_llama:
        default_kind = (
            _TEXT_DEFAULTS_BY_MODEL_TYPE.get(parent_type) or _TEXT_CONFIG_DEFAULTS.get(parent, (None, "llama"))[1]
        )
        kind = decoder.setdefault("model_type", default_kind)
        if not isinstance(kind, str):
            raise ValueError(f"{key}.model_type must be a string")
        if not sglang_llava and not is_known_transformers_config(kind, backend_version):
            raise ValueError(f"{key}.model_type={kind!r} does not identify an available Transformers config")
    elif kind is not None and not isinstance(kind, str):
        raise ValueError(f"{key}.model_type must be a string")
    if fixed_llama and (backend == "vllm" or kind == "llama"):
        # Idefics3 builds LlamaModel with the selected config's attributes.
        decoder["architectures"] = ["LlamaForCausalLM"]
        return kind or "llama", "LlamaForCausalLM"
    selected = _SELECTED_DECODER_ARCHITECTURES.get(kind)
    dynamic = registered_submodel and backend == "vllm"
    native_dispatch = backend == "vllm" and (registered_submodel or key is None)
    classes = decoder.get("architectures")
    if sglang_llava:
        causal = _sglang_llava_causal_lm(kind, backend_version)
        if causal is None:
            raise ValueError(f"SGLang has no causal LM for {key}.model_type={kind!r}")
        classes = decoder["architectures"] = [causal]
    if dynamic and classes is None:
        default = default_causal_lm(kind, backend_version)
        if default is None:
            raise ValueError(f"vLLM has no default causal LM for {key}.model_type={kind!r}")
        classes = [default]
    if not native_dispatch and not classes:
        classes = None
    if classes is not None and (not isinstance(classes, list) or not all(isinstance(n, str) for n in classes)):
        raise ValueError(f"HF config architectures must be a list of strings, got {classes!r}")
    if native_dispatch and classes == []:
        raise ValueError("No model architectures are specified for the vLLM text decoder")
    named = classes[0] if classes else None
    if native_dispatch and classes:
        named = select_architecture(classes, backend_version, decoder.get("auto_map"))
        if named is None:
            # A flat checkpoint can supply its own geometry under a training-only class.
            if key is None:
                required = _DECODER_GEOMETRY_DEFAULTS[kind].keys()
                if kind == "llama" and classes == ["VLlama3ForCausalLM"]:
                    # The released SmolVLM text checkpoint supplies Llama projection geometry
                    # under its training label. Its layer count can inherit LlamaConfig's default.
                    required = required - {"num_hidden_layers"} | {"num_key_value_heads", "vocab_size"}
                if all(decoder.get(k) is not None for k in required):
                    return kind, selected
            raise ValueError(f"model_type={kind!r} conflicts with architectures={classes[0]!r}")
        decoder["architectures"] = [named]
    elif native_dispatch:
        named = selected
    if (
        native_dispatch
        and named in ("LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM", "OPTForCausalLM")
        and kind in _BORROWED_DECODER_DEFAULTS
    ):
        for field, default in _BORROWED_DECODER_DEFAULTS[kind].items():
            decoder.setdefault(field, default)
        if kind == "phi3" and decoder.get("num_key_value_heads") is None:
            decoder["num_key_value_heads"] = decoder["num_attention_heads"]
    if native_dispatch and kind == "gemma4_text" and named in _VLLM_DENSE_DECODERS:
        # Construct the source config before the selected attention class reads its attributes.
        for field, default in (
            ("hidden_size", 2304),
            ("num_hidden_layers", 30),
            ("num_attention_heads", 8),
            ("intermediate_size", GEMMA4_DEFAULT_INTERMEDIATE_SIZE),
            ("max_position_embeddings", 131072),
            ("vocab_size", 262144),
            ("rms_norm_eps", 1e-6),
            ("head_dim", 256),
            ("num_key_value_heads", 4),
        ):
            decoder.setdefault(field, default)
        if decoder.get("use_bidirectional_attention") == "all":
            decoder["is_causal"] = False
        if named != "OPTForCausalLM" and decoder.get("layer_types") is None:
            layers = layout_integer(decoder["num_hidden_layers"], "num_hidden_layers")
            decoder["layer_types"] = [
                "full_attention" if (i + 1) % 6 == 0 or i == layers - 1 else "sliding_attention" for i in range(layers)
            ]
        if named != "OPTForCausalLM":
            decoder.setdefault("sliding_window", 512)
            # Use the dispatched projection geometry, including the ordinary KV heads, below.
            decoder["model_type"] = {
                "LlamaForCausalLM": "llama",
                "MistralForCausalLM": "mistral",
                "Qwen2ForCausalLM": "qwen2",
                "Qwen3ForCausalLM": "qwen3",
            }[named]
    if native_dispatch and named == "Qwen2ForCausalLM":
        # Qwen2 derives the QKV projection width even if the config declares head_dim.
        if kind not in ("llama", "mistral", "qwen2", "qwen3"):
            required = (
                "num_key_value_heads",
                "max_position_embeddings",
                "rms_norm_eps",
                "intermediate_size",
                "vocab_size",
                "num_hidden_layers",
            )
            if any(decoder.get(field) is None for field in required) or decoder.get("hidden_act") != "silu":
                raise ValueError(f"Qwen2ForCausalLM requires its decoder configuration for model_type={kind!r}")
        _apply_decoder_geometry_defaults(decoder, kind)
        if decoder.get("hidden_act", "silu") != "silu":
            raise ValueError("Qwen2ForCausalLM requires the silu activation")
        hidden = layout_integer(decoder.get("hidden_size"), "hidden_size")
        heads = layout_integer(decoder.get("num_attention_heads"), "num_attention_heads")
        decoder["head_dim"] = hidden // heads
        if kind not in ("qwen2", "qwen3", "qwen2_moe"):
            # Only these Qwen constructors derive layer_types from this flag.
            decoder.pop("use_sliding_window", None)
        return kind, named
    if kind is None and named:
        kind = next((k for k, name in _SELECTED_DECODER_ARCHITECTURES.items() if name == named), None)
    if kind == "opt":
        _apply_decoder_geometry_defaults(decoder, kind)
        decoder.setdefault("enable_bias", True)
        decoder.setdefault("activation_function", "relu")
        decoder.setdefault("layer_norm_elementwise_affine", True)
        decoder.setdefault("do_layer_norm_before", True)
        decoder.setdefault("_remove_final_layer_norm", False)
        if decoder.get("word_embed_proj_dim") is None:
            decoder["word_embed_proj_dim"] = decoder["hidden_size"]
        named = named or "OPTForCausalLM"
        decoder["architectures"] = [named]
    if named == "OPTForCausalLM" and (native_dispatch or kind == "opt"):
        # These GPT configs expose their serialized dimensions under the canonical attributes read by OPT.
        gpt_defaults = {
            "gpt2": (768, 12, 12, 1024, 50257),
            "gpt-sw3": (768, 12, 12, 1024, 50257),
            "gpt_bigcode": (768, 12, 12, 1024, 50257),
            "gptj": (4096, 16, 28, 2048, 50400),
            "gpt_neo": (2048, 16, 24, 2048, 50257),
            "codegen": (4096, 16, 28, 2048, 50400),
            "openai-gpt": (768, 12, 12, 512, 40478),
        }
        if kind in gpt_defaults:
            attributes = ("hidden_size", "num_attention_heads", "num_hidden_layers", "max_position_embeddings")
            aliases = (
                ("hidden_size", "num_heads", "num_layers", "max_position_embeddings")
                if kind == "gpt_neo"
                else ("n_embd", "n_head", "n_layer", "n_positions")
            )
            for field, alias, default in zip(attributes, aliases, gpt_defaults[kind][:-1], strict=True):
                decoder.setdefault(field, decoder.get(alias, default))
            decoder.setdefault("vocab_size", gpt_defaults[kind][-1])
            if kind == "gpt_neo":
                schedule = decoder.get("attention_types")
                if schedule is None:
                    schedule = [[["global", "local"], 12]]
                if (
                    not isinstance(schedule, (list, tuple))
                    or any(
                        not isinstance(item, (list, tuple))
                        or len(item) != 2
                        or not isinstance(item[0], (list, tuple))
                        or type(item[1]) is not int
                        for item in schedule
                    )
                    or sum(len(types) * max(0, repeats) for types, repeats in schedule) != decoder["num_hidden_layers"]
                ):
                    raise ValueError("GPT-Neo attention_types must expand to num_layers entries")
        # OPT uses full multi-head attention and its own FFN fields even when a
        # different Transformers config provides the actual constructor arguments.
        needed = (
            "ffn_dim",
            "word_embed_proj_dim",
            "enable_bias",
            "activation_function",
            "layer_norm_elementwise_affine",
            "do_layer_norm_before",
        )
        if any(field not in decoder for field in needed) or (
            decoder["do_layer_norm_before"] and "_remove_final_layer_norm" not in decoder
        ):
            raise ValueError("OPTForCausalLM requires its OPT decoder configuration")
        _apply_decoder_geometry_defaults(decoder, kind)
        if any(decoder.get(field) is None for field in _DECODER_GEOMETRY_DEFAULTS["opt"]):
            raise ValueError(f"OPTForCausalLM requires source config dimensions for model_type={kind!r}")
        heads = layout_integer(decoder.get("num_attention_heads"), "num_attention_heads")
        hidden = layout_integer(decoder.get("hidden_size"), "hidden_size")
        if hidden % heads:
            raise ValueError("OPT hidden size must be divisible by its attention head count")
        decoder.update(num_key_value_heads=heads, head_dim=hidden // heads, intermediate_size=decoder["ffn_dim"])
        if native_dispatch:
            # OPT constructs full multi-head attention in every layer, independent of the source config class.
            decoder["model_type"] = "opt"
            for field in (
                "layer_types",
                "sliding_window",
                "sliding_window_pattern",
                "use_sliding_window",
                "per_layer_config",
            ):
                decoder.pop(field, None)
            return "opt", named
        return kind, named
    if native_dispatch and named in ("LlamaForCausalLM", "MistralForCausalLM") and named != selected:
        if kind not in ("llama", "mistral", "qwen2", "qwen3"):
            required = ("hidden_size", "num_attention_heads", "num_hidden_layers", "intermediate_size", "rms_norm_eps")
            if any(decoder.get(field) is None for field in required) or decoder.get("hidden_act") != "silu":
                raise ValueError(f"{named} requires its decoder configuration for model_type={kind!r}")
            if decoder.get("num_key_value_heads", 1) is None:
                raise ValueError(f"{named} requires a valid num_key_value_heads")
        return kind, named
    if native_dispatch and (kind, named) in (
        ("qwen3_vl_text", "Qwen3ForCausalLM"),
        ("qwen3_vl_moe_text", "Qwen3MoeForCausalLM"),
    ):
        # The native Qwen3 decoder layers read their VL text config's dimensions directly.
        return kind, named
    if native_dispatch and not selected and named:
        default = default_causal_lm(kind, backend_version)
        if not default or named != select_architecture([default], backend_version, None):
            raise ValueError(f"model_type={kind!r} conflicts with architectures={named!r}")
        return kind, named
    if not named or not selected or named == selected:
        return kind, selected
    if selected == "Step3p5ForCausalLM" and ARCHITECTURE_TO_MODEL_FAMILY.get(named) == "STEP3P7":
        return kind, selected
    if native_dispatch and kind == "deepseek_v3" and named == "DeepseekV2ForCausalLM":
        # DeepseekV3ForCausalLM inherits the V2 implementation without overriding it.
        return kind, named
    if native_dispatch or named in ARCHITECTURE_TO_MODEL_FAMILY or named in _SELECTED_DECODER_ARCHITECTURES.values():
        raise ValueError(f"model_type={kind!r} conflicts with architectures={named!r}")
    return kind, selected


# Transformers constructor fields and aliases used by the selected expert implementations.
# (count, alias applied after construction, default count, routed intermediate width, default width)
_DECODER_EXPERT_DEFAULTS = {
    "deepseek_v3": ("n_routed_experts", "num_local_experts", 256, "moe_intermediate_size", 2048),
    "glm4_moe": ("n_routed_experts", "num_local_experts", 128, "moe_intermediate_size", 1408),
    "gpt_oss": ("num_local_experts", "num_experts", 128, "intermediate_size", 2880),
    "llama4_text": ("num_local_experts", None, 16, "intermediate_size", 8192),
    "minimax_m2": ("num_local_experts", "num_experts", 256, "intermediate_size", 1536),
    "mixtral": ("num_local_experts", "num_experts", 8, "intermediate_size", 14336),
    "qwen2_moe": ("num_experts", None, 60, "moe_intermediate_size", 1408),
    "qwen3_moe": ("num_experts", "num_local_experts", 128, "moe_intermediate_size", 768),
    "qwen3_vl_moe_text": ("num_experts", "num_local_experts", 60, "moe_intermediate_size", 1408),
}


def _config_only_backend_version(
    system: str | None, backend: str, requested: str | None, systems_path: str | list[str] | None = None
) -> str | None:
    """Resolve cache semantics for a config-only estimate, including a deployment with no local timing data."""
    if backend not in ("vllm", "sglang") or requested not in (None, "current", "previous", "next"):
        return requested
    if system is None:
        return None
    try:
        roots = perf_database.resolve_deployment_systems_paths(systems_path)
        slots = perf_database.get_version_slots(system, backend, systems_paths=roots) or {}
        if requested is None and not slots:
            return perf_database.get_latest_database_version(system, backend, systems_paths=roots)
        return slots.get(requested or "current")
    except MissingDataError:
        return None


def _config_only_system_spec(system: str | None, systems_path: str | list[str] | None) -> dict:
    """Load the named platform offline, including non-CUDA targets without an SM version."""
    try:
        return (
            perf_database.load_system_spec(system, perf_database.resolve_deployment_systems_paths(systems_path))
            if system
            else {}
        )
    except MissingDataError:
        return {}


def _sglang_checkpoint_quant(document: dict, raw: dict) -> dict | None:
    for quant in (
        document.get("quantization_config"),
        (document.get("text_config") or {}).get("quantization_config"),
        document.get("compression_config"),
        raw.get("hf_quant_config"),
    ):
        if quant is not None:
            if not isinstance(quant, dict):
                raise ValueError("SGLang checkpoint quantization metadata must be an object")
            return quant
    return None


def _sglang_skips_modules(
    ignored: list[str],
    layers: Collection[int],
    release: Version,
    module: str,
    *,
    shards: tuple[str, ...] = (),
) -> bool:
    """Match SGLang FP8 exclusions against the layers that instantiate this module."""
    names = set(ignored)
    if release >= Version("0.5.10rc0"):
        bare = {name.removeprefix("model.") for name in names}
        names = bare | {"model." + name for name in bare}
    elif module.endswith(".experts"):
        return False  # Earlier FusedMoE constructors always select Fp8MoEMethod.
    dotted = release >= Version("0.5.11")
    strip_trailing = release >= Version("0.5.13")

    def skipped(prefix: str) -> bool:
        for name in names:
            # A checkpoint can also list a parameter below its fused expert module.
            if module.endswith(".experts") and prefix in name:
                return True
            if strip_trailing:
                name = name.rstrip(".")
            if name in prefix and (not dotted or f".{name}." in f".{prefix}."):
                return True
        return False

    def layer_skipped(prefix: str) -> bool:
        if shards:
            states = {skipped(prefix.rsplit(".", 1)[0] + "." + name) for name in shards}
            if len(states) != 1:
                raise ValueError(f"SGLang requires all shards of {prefix} to have the same FP8 precision")
            return states.pop()
        if prefix.endswith(".gate_up_proj") and all(
            prefix.replace("gate_up_proj", name) in names for name in ("gate_proj", "up_proj")
        ):
            return True
        return skipped(prefix)

    all_skipped = True
    for i in layers:
        if not layer_skipped(module.format(i)):
            if not shards:
                return False
            all_skipped = False  # A different layer can still contain a partially quantized fused projection.
    return all_skipped


def _validate_sglang_dense_fp8(
    quant: dict,
    geom: dict,
    attn_tp: int,
    dense_tp: int,
    release: Version | None,
    ignored: list[str],
    layer_prefix: str,
    dense_layers: Collection[int],
) -> None:
    """Check the actual dense MLP and ordinary attention shards of the selected decoder."""
    block_n, block_k = fp8_weight_block_shape(quant, sglang=True)
    checks = []
    if dense_layers:
        intermediate = layout_integer(geom["inter"], "intermediate_size")
        if intermediate % dense_tp:
            raise ValueError("dense intermediate_size must be divisible by the tensor parallel size")
        mlp_shard = intermediate // dense_tp
        checks.append(("mlp.gate_up_proj", (mlp_shard,), block_n, dense_layers))
        if dense_tp > 1:
            checks.append(("mlp.down_proj", (mlp_shard,), block_k, dense_layers))
    if not geom.get("kv_lora_rank") and not geom.get("num_experts") and geom.get("head_dim"):
        layers = range(geom["layers"])
        query_heads = layout_integer(geom["num_attention_heads"], "num_attention_heads")
        kv_heads = layout_integer(geom["num_kv_heads"], "num_key_value_heads")
        if query_heads % attn_tp or max(kv_heads, attn_tp) % min(kv_heads, attn_tp):
            raise ValueError(
                "dense attention heads must partition or replicate uniformly over the tensor parallel ranks"
            )
        query_shard = query_heads // attn_tp * geom["head_dim"]
        kv_shard = max(1, kv_heads // attn_tp) * geom["head_dim"]
        checks.append(("self_attn.qkv_proj", (query_shard, kv_shard), block_n, layers))
        # Since 0.5.22 the validator reads the TP group of the linear layer itself.
        if attn_tp > 1 and (attn_tp == dense_tp or release is None or release >= Version("0.5.22")):
            checks.append(("self_attn.o_proj", (query_shard,), block_k, layers))
    fused = {}
    if release is not None and release >= Version("0.5.11"):
        # The CUDA loader supplies the serving model's packed modules, replacing checkpoint mappings.
        fused = {"gate_up_proj": ("gate_proj", "up_proj"), "qkv_proj": ("q_proj", "k_proj", "v_proj")}
    for module, widths, block, layers in checks:
        aligned = all(width % block == 0 for width in widths)
        shards = fused.get(module.rsplit(".", 1)[-1], ())
        if (
            ignored
            and release is not None
            and (shards or not aligned)
            and _sglang_skips_modules(ignored, layers, release, layer_prefix + module, shards=shards)
        ):
            continue
        if not aligned:
            raise ValueError(
                f"SGLang FP8 {module} shard widths {widths} must be divisible by weight_block_size={block}"
            )


def _outside_dense_decoder_projections(target: object) -> bool:
    """Recognize literal compressed-tensors exclusions outside the dense decoder's projection matrices."""
    if not isinstance(target, str) or target.startswith("re:"):
        return False
    module = target.rsplit(".", 1)[-1]
    namespace = target.removeprefix("model.").partition(".")[0]
    return module in ("lm_head", "embed_tokens", "embed_positions") or namespace in (
        "vision_tower",
        "vision_model",
        "multi_modal_projector",
        "connector",
    )


def _check_fused_decoder_exclusions(ignored: list, layers: int, prefix: str) -> None:
    """All logical shards of a compressed-tensors fused projection must agree."""
    targets = [target for target in ignored if not _outside_dense_decoder_projections(target)]
    if not targets:
        return
    if not all(isinstance(target, str) for target in targets):
        raise ValueError("compressed-tensors ignore must contain module names or regular expressions")
    exact = {target for target in targets if not target.startswith("re:")}
    patterns = [re.compile(target[3:]) for target in targets if target.startswith("re:")]

    def matches(name: str) -> bool:
        return name in exact or any(pattern.match(name) for pattern in patterns)

    for i in range(layers):
        for module, fused, shards in (
            ("self_attn", "qkv_proj", ("q_proj", "k_proj", "v_proj")),
            ("mlp", "gate_up_proj", ("gate_proj", "up_proj")),
        ):
            base = f"{prefix}model.layers.{i}.{module}."
            if not matches(base + fused):
                skipped = [matches(base + shard) for shard in shards]
                if any(skipped) and not all(skipped):
                    raise ValueError(f"vLLM requires the same quantization scheme for all shards of {base}{fused}")


def _vllm_fp8_linear_scale_bytes(quant: dict, geom: dict, tp: int) -> int:
    """Scale storage for one layer of rank-local QKV, output, gate/up and down projections."""
    hidden, inter = geom["hidden"], geom["inter"]
    if inter % tp:
        raise ValueError("vLLM dense intermediate_size must be divisible by the tensor parallel size")
    if quant.get("weight_block_size") is None:
        # Seven logical weights each have a per-tensor FP32 scale. Static inputs have another seven.
        return 7 * 4 * (2 if quant["activation_scheme"] == "static" else 1)

    block_n, block_k = fp8_weight_block_shape(quant, sglang=False)
    dim = layout_integer(geom["head_dim"], "head_dim")
    q = layout_integer(geom["num_attention_heads"], "num_attention_heads") // tp * dim
    kv = max(1, layout_integer(geom["num_kv_heads"], "num_key_value_heads") // tp) * dim
    ffn = inter // tp
    # Column-parallel fused shards must start at tile boundaries. TP also splits each row-parallel input.
    if q % block_n or kv % block_n or ffn % block_n or (tp > 1 and (q % block_k or ffn % block_k)):
        raise ValueError("vLLM FP8 projection shards must align with the checkpoint weight_block_size")
    matrices = ((q + 2 * kv, hidden), (hidden, q), (2 * ffn, hidden), (hidden, ffn))
    return 4 * sum(((out + block_n - 1) // block_n) * ((inp + block_k - 1) // block_k) for out, inp in matrices)


def _check_modelopt_mxfp8_inputs(geom: dict, tp: int, ignored: list[str], wrapper: str | None, opt: bool) -> None:
    # KMxfp8Static groups the input of each quantized rank-local matrix into sets of 32.
    if wrapper in _FIXED_LLAMA_WRAPPERS:
        # Idefics3 and SmolVLM construct an inner LlamaModel directly, without a checkpoint-name mapper.
        prefix = "model.text_model.layers."
    else:
        prefix = ("language_model." if wrapper is not None else "") + (
            "model.decoder.layers." if opt else "model.layers."
        )
        # The serving mapper expands wildcard suffixes before renaming the checkpoint's module prefixes.
        mapped = []
        for pattern in ignored:
            patterns = (
                (pattern[:-1], pattern[:-1] + ".*")
                if len(pattern) >= 2 and pattern.endswith("*") and pattern[-2] != "."
                else (pattern,)
            )
            for name in patterns:
                if wrapper in _TEXT_CONFIG_DEFAULTS and name.startswith("model.language_model."):
                    name = "language_model.model." + name.removeprefix("model.language_model.")
                if opt and name.startswith("decoder."):
                    name = "model." + name
                mapped.append(name)
        ignored = mapped
    hidden = layout_integer(geom["hidden"], "hidden_size")
    inter = layout_integer(geom["inter"], "intermediate_size")
    q = layout_integer(geom["num_attention_heads"], "num_attention_heads") * layout_integer(
        geom["head_dim"], "head_dim"
    )
    blocks = (
        ("self_attn.qkv_proj", hidden, ("q_proj", "k_proj", "v_proj")),
        ("self_attn.out_proj" if opt else "self_attn.o_proj", q, ()),
        ("fc1" if opt else "mlp.gate_up_proj", hidden, () if opt else ("gate_proj", "up_proj")),
        ("fc2" if opt else "mlp.down_proj", inter, ()),
    )
    for module, width, shards in blocks:
        row_parallel = module.endswith(("o_proj", "out_proj", "down_proj", "fc2"))
        aligned = width % (32 * (tp if row_parallel else 1)) == 0
        if aligned and (not ignored or not shards):
            continue
        if row_parallel and width % tp:
            raise ValueError(f"vLLM {module} input width must be divisible by tensor parallel size")
        for i in range(geom["layers"]):
            name = f"{prefix}{i}.{module}"
            if name in ignored:
                continue
            if shards:
                excluded = [name.rsplit(".", 1)[0] + "." + shard in ignored for shard in shards]
                if any(excluded):
                    if not all(excluded):
                        raise ValueError(f"vLLM requires the same quantization scheme for all shards of {name}")
                    continue
            if aligned or any(pattern in name or fnmatchcase(name, pattern) for pattern in ignored):
                continue
            raise ValueError(f"vLLM ModelOpt MXFP8 requires {name} rank-local input width divisible by 32")


def _compressed_tensor_strategy(args: dict) -> str:
    strategy = args.get("strategy")
    group_size = args.get("group_size")
    if strategy is not None:
        if not isinstance(strategy, str):
            raise ValueError("compressed-tensors strategy must be a string")
        strategy = strategy.lower()
        if strategy in ("group", "tensor_group"):
            layout_integer(group_size, "compressed-tensors group_size")
        elif isinstance(group_size, int) and group_size > 0:
            raise ValueError("compressed-tensors group_size requires a grouped strategy")
        return strategy
    if group_size is None:
        return "tensor"
    if group_size == -1:
        return "channel"
    layout_integer(group_size, "compressed-tensors group_size")
    return "group"


def _compressed_tensor_type(args: dict) -> str:
    value = args.get("type", "int")
    if not isinstance(value, str):
        raise ValueError("compressed-tensors type must be a string")
    return value.lower()


def _vllm_weight_config(document: dict) -> dict | None:
    # get_quant_config loads the checkpoint's settings after ModelArchConfigConvertor chooses the quantizer class.
    text = document.get("text_config")
    for config, key in (
        (document, "quantization_config"),
        (text, "quantization_config"),
        (document, "compression_config"),
    ):
        if isinstance(config, dict) and (quant := config.get(key)) is not None:
            if not isinstance(quant, dict):
                raise ValueError(f"vLLM {key} must be a JSON object")
            if config is text and isinstance(parent_scheme := document.get("compression_config"), dict):
                selected = str(parent_scheme.get("quant_method", "")).lower().replace("_", "-")
                loaded = str(quant.get("quant_method", "")).lower().replace("_", "-")
                if selected and loaded and selected != loaded:
                    raise ValueError("vLLM selects a different quantizer from the text checkpoint settings")
                if selected and not loaded:
                    return {**quant, "quant_method": selected}
            return quant
    return None


def _automatic_decoder_kv_mode(
    document: dict, raw: dict, backend: str, backend_version: str | None
) -> KVCacheQuantMode | None:
    """Resolve the serving backend's automatic cache dtype, retaining the checkpoint metadata's original scope."""
    if backend not in ("vllm", "sglang"):
        return None
    root_quant = document.get("quantization_config")
    sidecar = raw.get("hf_quant_config")
    # SGLang checks inline metadata before opening a sidecar. vLLM's automatic ModelOpt
    # cache dtype reads the root HF config, where its loader installs a sidecar if needed.
    quant = _sglang_checkpoint_quant(document, raw) if backend == "sglang" else root_quant
    if quant is None:
        quant = sidecar
    from_weight_config = quant is None and backend == "vllm"
    if from_weight_config:
        # The vLLM attention layer also reads the selected compressed-tensors object's KV scheme.
        quant = _vllm_weight_config(document)
    if not isinstance(quant, dict):
        return None
    method = str(quant.get("quant_method", "")).lower().replace("_", "-")
    if from_weight_config and method != "compressed-tensors":
        return None
    release = parse_support_matrix_version(backend_version)
    scheme = quant.get("kv_cache_scheme")
    if backend == "sglang" and method in ("fp8", "mxfp8"):
        if release is None or release < Version("0.5.18"):
            return None
        algo = quant.get("kv_cache_quant_algo")  # Both names use SGLang's Fp8Config.
        return KVCacheQuantMode.fp8 if isinstance(algo, str) and algo.upper() == "FP8" else None
    if method == "compressed-tensors" and scheme is not None:
        fp8 = isinstance(scheme, dict) and scheme.get("type") == "float" and scheme.get("num_bits") == 8
        if backend == "sglang":
            if release is None or release < Version("0.5.19"):
                return None
            if (
                fp8
                and scheme.get("strategy") == "tensor"
                and scheme.get("symmetric", True)
                and not scheme.get("dynamic")
            ):
                return KVCacheQuantMode.fp8
            return None
        # Older vLLM validates the scales on attention construction even when auto keeps the model dtype.
        per_head = release is not None and release >= Version("0.15.0rc1")
        if not (
            fp8
            and scheme.get("symmetric")
            and (scheme.get("strategy") == "tensor" or (per_head and scheme.get("strategy") == "attn_head"))
        ):
            raise ValueError("vLLM requires a supported symmetric FP8 compressed-tensors KV scheme")
        return KVCacheQuantMode.fp8 if release is not None and release >= Version("0.22.0") else None
    section = quant.get("quantization") or {}
    producer = (quant.get("producer") or {}).get("name")
    if not (
        method.startswith("modelopt")
        or producer == "modelopt"
        or "modelopt_quant_config" in section
        or (backend == "sglang" and quant is sidecar and section.get("quant_algo"))
    ):
        return None
    if backend == "sglang":
        if quant.get("quant_algo") is not None:
            if isinstance(scheme, dict):
                return KVCacheQuantMode.fp8 if scheme.get("type") == "float" and scheme.get("num_bits") == 8 else None
            # The flat reader learned to fall back to kv_cache_quant_algo in 0.5.15.
            if release is None or release < Version("0.5.15"):
                return None
            section = quant
        algo = section.get("kv_cache_quant_algo")
        return KVCacheQuantMode.fp8 if isinstance(algo, str) and algo.upper() == "FP8" else None
    if not method.startswith("modelopt") and not section.get("quant_algo"):
        return None  # vLLM normalizes a legacy ModelOpt section to a quant_method before resolving its cache.
    algo = (
        section.get("kv_cache_scheme")
        or scheme
        or section.get("kv_cache_quant_algo")
        or quant.get("kv_cache_quant_algo")
    )
    if isinstance(algo, dict):
        if algo.get("type") == "float" and algo.get("num_bits") == 8 and algo.get("dynamic") is False:
            return KVCacheQuantMode.fp8
        if algo.get("type") == "float" and algo.get("num_bits") == 4:
            algo = "nvfp4"
        else:
            return None
    if str(algo).lower() == "nvfp4":
        return KVCacheQuantMode.nvfp4 if release is not None and release >= Version("0.20.0") else None
    return KVCacheQuantMode.fp8 if str(algo).lower() == "fp8" else None


_ONE_GIB = 1 << 30
_MAX_EXACT_BYTE_COUNT = 1 << 53

# A KV byte-budget -> token-count inverse. Every scalar estimation path produces one of
# these (native: the model's ``get_kvcache_max_tokens``; naive:
# ``NaiveKVCacheEstimator.get_kvcache_max_tokens``; the linear
# ``_linear_tokens_from_bytes`` is the default for constant-per-token growth), so
# the raw and tolerance-adjusted token counts invert the same curve through a
# single, uniform type.
TokensFromBytes = Callable[[float], int]

# Default fraction of post-weight memory reserved for KV in the naive fallback
# (exposed as the ``naive_kv_reservation`` arg so callers can tune it).
_DEFAULT_NAIVE_KV_RESERVATION = 0.80

# Backend -> the memory-fraction kind that backend accepts. TRT-LLM applies the
# fraction to FREE memory (`free_gpu_memory_fraction`); vLLM/SGLang apply it to
# TOTAL memory (`gpu_memory_utilization` / `mem_fraction_static`).
_BACKEND_FRACTION_KIND = {
    "trtllm": "of_free",
    "vllm": "of_total",
    "sglang": "of_total",
}


# --------------------------------------------------------------------------- #
# Validation (runs before any model build, so bad inputs fail cheaply).
# --------------------------------------------------------------------------- #


def _validate_memory_fraction(backend: str, memory_fraction_kind: str, memory_fraction_value: float) -> None:
    """Validate the ``memory_fraction_kind`` against the backend, then the range.

    Runs FIRST in :func:`estimate_kv_cache`, before any model build, so a wrong
    kind or out-of-range value fails deterministically without touching the perf
    database. ``of_free`` is TRT-LLM only; ``of_total`` is vLLM / SGLang.
    """
    expected = _BACKEND_FRACTION_KIND.get(backend)
    if expected is None:
        raise ValueError(f"unknown backend {backend!r} for KV-cache estimation")
    if memory_fraction_kind != expected:
        raise ValueError(
            f"incompatible memory fraction: backend {backend!r} does not accept "
            f"memory_fraction_kind {memory_fraction_kind!r} (expected {expected!r})"
        )
    f = float(memory_fraction_value)
    if not (math.isfinite(f) and 0.0 <= f <= 1.0):
        raise ValueError(f"kv_cache_memory_fraction must be finite and in [0, 1], got {f}")


def _validate_naive_reservation(naive_kv_reservation: float) -> None:
    """Validate the naive-fallback KV reservation fraction.

    Must be finite and in ``[0, 1]``. Runs FIRST in :func:`estimate_kv_cache`
    (before any model build), alongside the other fail-fast validators, so a bad
    value is rejected deterministically even on the native path where the
    fraction is never used.
    """
    r = float(naive_kv_reservation)
    if not (math.isfinite(r) and 0.0 <= r <= 1.0):
        raise ValueError(f"naive_kv_reservation must be finite and in [0, 1], got {naive_kv_reservation}")


def _validate_tolerance(tolerance_fraction: float | None) -> None:
    """Validate the optional KV-cache safety-margin ``tolerance_fraction``.

    Must be finite and in the half-open interval ``[0, 1)`` (``t = 1.0`` would
    reserve the entire budget). Runs FIRST in :func:`estimate_kv_cache`, before
    the expensive breakdown, so a bad value fails cheaply and deterministically.
    """
    if tolerance_fraction is None:
        return
    t = float(tolerance_fraction)
    if not (math.isfinite(t) and 0.0 <= t < 1.0):
        raise ValueError(f"tolerance_fraction must be finite and in [0, 1), got {tolerance_fraction}")


def _validate_encoder_parallel(encoder_parallel: str | None) -> None:
    if encoder_parallel not in (None, "tp", "dp"):
        raise ValueError(f"encoder_parallel must be 'tp' or 'dp', got {encoder_parallel!r}")


def _validate_reserved_bytes(reserved_bytes: int) -> None:
    if isinstance(reserved_bytes, bool) or not isinstance(reserved_bytes, int) or reserved_bytes < 0:
        raise ValueError(f"reserved_bytes must be a non-negative integer, got {reserved_bytes!r}")


def _validate_cuda_graph_reservation(cuda_graph_reserved_bytes: int) -> None:
    """Validate the fixed, rank-local CUDA graph reservation."""
    if (
        isinstance(cuda_graph_reserved_bytes, bool)
        or not isinstance(cuda_graph_reserved_bytes, int)
        or cuda_graph_reserved_bytes < 0
        or cuda_graph_reserved_bytes > _MAX_EXACT_BYTE_COUNT
    ):
        raise ValueError(
            "cuda_graph_reserved_bytes must be a non-negative integer no greater than "
            f"{_MAX_EXACT_BYTE_COUNT}, got "
            f"{cuda_graph_reserved_bytes!r}"
        )


# --------------------------------------------------------------------------- #
# Shared scalar budget math.
#
# The scalar formula both the estimate and (in a follow-up PR) the sweep's
# `InferenceSummary._check_and_set_kv_cache_oom` should use. The OOM check folds
# `reserved` (block-allocator overhead) and `tolerance` into the fraction
# multiplicatively; the estimate passes `reserved=tolerance=0` here and applies
# its tolerance separately via `_apply_tolerance` (so `tolerance_adjusted` stays
# a distinct field). Unit-agnostic: pass capacity/non_kv in the same unit.
# TODO(#1208): point `_check_and_set_kv_cache_oom` at this helper.
# --------------------------------------------------------------------------- #


def kv_cache_budget_bytes(
    *,
    capacity: float,
    non_kv: float,
    fraction: float,
    of_free: bool,
    reserved: float = 0.0,
    tolerance: float = 0.0,
) -> float:
    """KV-cache budget for one rank.

    ``of_free`` (TRT-LLM) applies the fraction to the FREE pool
    ``(capacity - non_kv)``; otherwise (vLLM/SGLang ``of_total``) it applies the
    fraction to TOTAL memory then subtracts ``non_kv``. ``reserved`` and
    ``tolerance`` shrink the fraction multiplicatively (both default to a no-op).
    """
    scale = (1.0 - reserved) * (1.0 - tolerance)
    if of_free:
        return (capacity - non_kv) * fraction * scale
    return capacity * fraction * scale - non_kv


def _linear_tokens_from_bytes(per_token_bytes: float) -> TokensFromBytes:
    """The linear-growth inverse ``floor(budget / per_token)`` as a callable.

    The default :data:`TokensFromBytes` for models with a constant per-token KV
    size (MHA / GQA / MLA), where token capacity scales linearly with the budget.
    Hybrid / compressed-attention paths supply their own non-linear inverse
    instead.
    """

    def tokens_from_bytes(budget_bytes: float) -> int:
        budget = float(budget_bytes)
        if budget <= 0.0 or per_token_bytes <= 0.0:
            return 0
        return int(budget // per_token_bytes)

    return tokens_from_bytes


def _apply_tolerance(estimate: dict[str, Any], tolerance_fraction: float | None) -> dict[str, Any]:
    """Populate ``tolerance_adjusted`` on a finished raw estimate (in place).

    ``adj_bytes = floor(total_kv_size_bytes * (1 - t))`` (a ``t`` of 0.05 keeps
    95% of raw, reserving a 5% safety margin); ``adj_tokens`` is the token count
    that fits in ``adj_bytes``, derived through the estimate's
    ``tokens_from_kv_bytes`` inverse -- the SAME callable used for the raw count,
    so the raw and adjusted counts follow the same KV curve. When
    ``tolerance_fraction`` is ``None``, ``tolerance_adjusted`` stays ``None`` (raw
    only).

    The emitted dict keys (``tolerance_fraction`` / ``total_kv_size_bytes`` /
    ``total_kv_size_tokens``) match what the Rust ``estimate_from_dict`` parses
    back, so the numbers are identical across the PyO3 boundary.
    """
    if tolerance_fraction is None:
        return estimate
    t = float(tolerance_fraction)
    raw_bytes = int(estimate["total_kv_size_bytes"])
    adj_bytes = math.floor(raw_bytes * (1.0 - t))
    tokens_from_bytes: TokensFromBytes = estimate["tokens_from_kv_bytes"]
    adj_tokens = int(tokens_from_bytes(adj_bytes))
    estimate["tolerance_adjusted"] = {
        "tolerance_fraction": t,
        "total_kv_size_bytes": adj_bytes,
        "total_kv_size_tokens": adj_tokens,
    }
    return estimate


# --------------------------------------------------------------------------- #
# Native path: AIC's full backend memory model.
#
# `KVCacheEstimator` builds the model + backend + perf DB (the part that fails
# for models AIC cannot model -> the caller falls back to NaiveKVCacheEstimator),
# computes the `BaseBackend._get_memory_usage` breakdown + per-token KV byte size
# + capacity, then sizes the rank-local KV budget via the OfFree (TRT-LLM) /
# OfTotal (vLLM/SGLang) formula. Mirrors NaiveKVCacheEstimator: build via
# from_request(...), then estimate().
#
# Everything is in BYTES (not GiB) so no downstream unit scaling is needed:
# `_get_memory_usage` returns GiB, so the breakdown components are multiplied by
# 1 GiB; `get_kvcache_bytes_per_sequence(1)` is already bytes;
# `system_spec["gpu"]["mem_capacity"]` is already bytes.
# --------------------------------------------------------------------------- #


class KVCacheEstimator:
    """Native KV-cache estimator: AIC's full backend memory model.

    :meth:`from_request` builds the model / backend / perf DB and the non-KV
    memory breakdown -- the part that raises for models AIC cannot build, which is
    the signal for the caller to fall back to :class:`NaiveKVCacheEstimator`.
    :meth:`estimate` then does the pure OfFree (TRT-LLM) / OfTotal (vLLM/SGLang)
    budget math, whose ``ValueError``\\ s (no KV budget, zero per-token) propagate
    rather than triggering the fallback.
    """

    def __init__(self, breakdown: dict[str, Any]):
        self._breakdown = breakdown

    @classmethod
    def from_request(
        cls,
        model_path: str,
        system: str,
        backend: str,
        backend_version: str | None = None,
        *,
        max_num_tokens: int,
        max_batch_size: int,
        tp_size: int = 1,
        pp_size: int = 1,
        attention_dp_size: int = 1,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
        cp_size: int = 1,
        dcp_size: int = 1,
        gemm_quant_mode: str | None = None,
        moe_quant_mode: str | None = None,
        kvcache_quant_mode: str | None = None,
        fmha_quant_mode: str | None = None,
        comm_quant_mode: str | None = None,
        moe_backend: str | None = None,
        moe_kernel_source: str | None = None,
        attention_backend: str | None = None,
        enable_eplb: bool = False,
        wideep_num_slots: int | None = None,
        nextn: int = 0,
        systems_path: str | None = None,
        colocated_encoder: bool = False,
        reserved_bytes: int = 0,
        encoder_parallel: str | None = None,
    ) -> KVCacheEstimator:
        """Build the model/backend/perf-DB and the non-KV memory breakdown.

        ``colocated_encoder`` keeps the model's vision-encoder weights resident on
        this rank, laid out per ``encoder_parallel`` (``"tp"`` shards the tower
        over the tensor-parallel group, ``"dp"`` replicates it; sharded when
        unset, as SGLang lays the tower out by default); ``reserved_bytes`` is a
        fixed rank-local reservation outside the KV pool (SGLang's multimodal
        embedding cache).

        Reuses the exact AIC machinery the latency path uses: ``build_model_config``
        + ``apply_nextn`` (so the built model is spec-decode aware) + ``get_model``
        (quant inferred inside ``get_model``), ``get_backend``, and
        ``perf_database.get_database``. Calls ``BaseBackend._get_memory_usage`` with
        ``num_tokens = max_num_tokens`` and ``mtp_activation_scaling=False`` (so
        activations track ``BuildConfig.max_num_tokens`` -- the engine's per-iteration
        token budget that already bounds draft tokens -- the way TRT-LLM's
        ``_memory_usage_kwargs_for_agg`` does, without re-applying the ``(nextn+1)``
        decode multiplier); ``isl``/``osl``/``max_seq_len`` only
        feed the discarded ``kvcache`` key, so they are set to a neutral ``1``. Note
        ``max_batch_size`` does NOT affect non-KV memory (activations use
        ``num_tokens``, KV is recomputed per token), but it is accepted to mirror the
        request shape. SGLang retains peak activations in the breakdown for
        diagnostics but excludes them from the static weights/KV pool: its
        ``mem_fraction_static`` already reserves transient execution headroom.

        Raises when AIC cannot build the model/backend or the system spec is missing --
        the signal for the caller to fall back to the naive estimator.
        """
        resolved_moe_tp = moe_tp_size if moe_tp_size is not None else 1
        resolved_moe_ep = moe_ep_size if moe_ep_size is not None else 1
        validate_moe_controls(
            model_path=model_path,
            enable_eplb=enable_eplb,
            wideep_num_slots=wideep_num_slots,
            moe_backend=moe_backend,
            moe_kernel_source=moe_kernel_source,
        )
        # A hosted tower is sharded unless dp is asked for, as SGLang lays it out.
        encoder_layout = (
            {"enable_encoder_dp": encoder_parallel == "dp"} if encoder_parallel is not None or colocated_encoder else {}
        )
        model_config = build_model_config(
            tp_size=tp_size,
            pp_size=pp_size,
            attention_dp_size=attention_dp_size,
            moe_tp_size=resolved_moe_tp,
            moe_ep_size=resolved_moe_ep,
            gemm_quant_mode=gemm_quant_mode,
            kvcache_quant_mode=kvcache_quant_mode,
            fmha_quant_mode=fmha_quant_mode,
            moe_quant_mode=moe_quant_mode,
            comm_quant_mode=comm_quant_mode,
            moe_backend=moe_backend,
            moe_kernel_source=moe_kernel_source,
            attention_backend=attention_backend,
            enable_eplb=enable_eplb,
            wideep_num_slots=wideep_num_slots,
            cp_size=cp_size,
            dcp_size=dcp_size,
            **encoder_layout,
        )
        # Apply nextn/MTP onto the config BEFORE get_model so the built model is
        # spec-decode aware (e.g. for any draft-module weights). This does NOT scale
        # the capacity activation by (nextn+1); that multiplier is suppressed below
        # (see mtp_activation_scaling). Mirrors the agg/disagg/static estimate paths.
        # Memory is cost-side only; accepted-token progress never enters
        # capacity math.
        apply_nextn(model_config, nextn)
        # Capacity needs model/system metadata, including when external FPM
        # timing has no backend data directory.
        database = perf_database.get_database(
            system,
            backend,
            "current" if backend_version is None else backend_version,
            systems_paths=systems_path,
            allow_missing_data=True,
        )
        resolve_sglang_mla_compute(model_config, model_path, backend, database.version, database.system_spec)
        resolve_trtllm_mla_compute(model_config, model_path, backend, database.system_spec)
        model = get_model(model_path, model_config, backend, database=database)
        if backend == "vllm" and getattr(model, "architecture", None) == "Gemma4ForConditionalGeneration":
            raw = _load_model_config_from_model_path(model_path)
            text = raw.get("text_config", raw)
            if "per_layer_config" in text:
                gemma4_cache_shapes(text, backend, database.version, agreement_tp_size=tp_size)
        backend_obj = get_backend(backend)

        # num_tokens = max_num_tokens -> activations track BuildConfig.max_num_tokens
        # (TRT-LLM `_memory_usage_kwargs_for_agg`). With num_tokens > 0 passed
        # explicitly, isl/osl/max_seq_len only feed the discarded `kvcache` key.
        #
        # mtp_activation_scaling=False: max_num_tokens is the engine's per-iteration
        # token budget, which already caps total per-forward tokens INCLUDING the nextn
        # draft tokens. The latency sweep's (nextn+1) activation multiplier would
        # double-count here -- it inflated the prefill worker's non-KV memory past GPU
        # capacity and drove the KV budget negative once the draft length grew
        # (AIC-1110). With it suppressed the breakdown is nextn-independent: the draft
        # module's marginal weight/KV cost (~nextn extra layers) is not separately
        # modeled, so the estimate slightly OVERSTATES available KV (an optimistic
        # approximation) -- but far closer to reality than the gross prior over-count.
        memory = backend_obj._get_memory_usage(
            model,
            database,
            batch_size=int(max_batch_size),
            beam_width=1,
            isl=1,
            osl=1,
            num_tokens=int(max_num_tokens),
            prefix=0,
            max_seq_len=1,
            mtp_activation_scaling=False,
        )

        weights_bytes = float(memory["weights"]) * _ONE_GIB
        # Some timing graphs approximate layer counts or embedding lookup
        # shapes. Prefer model-owned resident weights when that layout is known.
        weight_memory = getattr(model, "get_weight_memory_bytes", None)
        if weight_memory is not None:
            resident_weights = weight_memory()
            if resident_weights is not None:
                weights_bytes = float(resident_weights)
        if colocated_encoder:
            # The language worker loads the vision tower; its weights never enter
            # the context-op inventory above.
            weights_bytes += float(sum(op.get_weights() for op in getattr(model, "encoder_ops", ())))
        activations_bytes = float(memory["activations"]) * _ONE_GIB
        runtime_overhead_bytes = float(memory["others"]) * _ONE_GIB
        comm_overhead_bytes = float(memory["nccl"]) * _ONE_GIB
        reserved_bytes = float(reserved_bytes)
        non_kv_bytes = weights_bytes + runtime_overhead_bytes + comm_overhead_bytes + reserved_bytes
        # SGLang sizes its static weights/KV pool before allocating peak forward
        # activations. mem_fraction_static already leaves headroom for those
        # transient allocations; charging them inside the pool counts them twice.
        # Keep the activation estimate in diagnostics, but not in its KV budget.
        if backend != "sglang":
            non_kv_bytes += activations_bytes

        # Per-RANK persistent KV. Prefill CP keeps the full KV on every rank;
        # decode CP stripes the token-linear KV so each rank holds 1/dcp of every
        # token, while rank-local per-request state (Kimi-K3 KDA) stays whole
        # (BaseModel.get_kvcache_rank_*). The per-token figure and the
        # byte-budget -> token-count inverse must see the same striping.
        # Subtract the zero-token footprint so fixed request state (e.g. Mamba)
        # is not exported as a per-token byte rate. The model's capacity inverse
        # still reserves that state for the scheduler slots.
        # Duck-typed model doubles without the rank hooks keep the full KV.
        rank_bytes_per_sequence = getattr(
            model, "get_kvcache_rank_bytes_per_sequence", model.get_kvcache_bytes_per_sequence
        )

        def tokens_from_kv_bytes(kv_budget_bytes: float) -> int:
            # Resolved lazily: some callers never invert (stubs without the
            # capacity method).
            capacity = getattr(model, "get_kvcache_rank_batch_capacity", None) or model.get_kvcache_batch_capacity
            return int(capacity(float(kv_budget_bytes), max_batch_size))

        return cls(
            {
                "weights_bytes": weights_bytes,
                "activations_bytes": activations_bytes,
                "runtime_overhead_bytes": runtime_overhead_bytes,
                "comm_overhead_bytes": comm_overhead_bytes,
                "reserved_bytes": reserved_bytes,
                "non_kv_bytes": non_kv_bytes,
                # SGLang measures free memory after distributed/CUDA setup,
                # before loading weights, and applies its static fraction there.
                "pre_model_load_overhead_bytes": (
                    runtime_overhead_bytes + comm_overhead_bytes if backend == "sglang" else 0.0
                ),
                "kv_size_per_token_bytes": float(rank_bytes_per_sequence(1) - rank_bytes_per_sequence(0)),
                "kvcache_quant_mode": (
                    model_config.kvcache_quant_mode.name if model_config.kvcache_quant_mode is not None else None
                ),
                "gpu_memory_capacity_bytes": float(database.system_spec["gpu"]["mem_capacity"]),
                "tokens_from_kv_bytes": tokens_from_kv_bytes,
            }
        )

    @property
    def breakdown(self) -> dict[str, Any]:
        """The non-KV memory breakdown (BYTE quantities + the ``tokens_from_kv_bytes``
        inverse) computed by :meth:`from_request`."""
        return self._breakdown

    def estimate(
        self,
        *,
        is_of_free: bool,
        fraction: float,
        gpu_memory_capacity_bytes_override: int | None,
        cuda_graph_reserved_bytes: int = 0,
    ) -> dict[str, Any]:
        """Size the rank-local KV budget from the breakdown (raw estimate dict)."""
        return self._estimate_from_breakdown(
            self._breakdown,
            is_of_free=is_of_free,
            fraction=fraction,
            gpu_memory_capacity_bytes_override=gpu_memory_capacity_bytes_override,
            cuda_graph_reserved_bytes=cuda_graph_reserved_bytes,
        )

    @staticmethod
    def _estimate_from_breakdown(
        breakdown: dict[str, Any],
        *,
        is_of_free: bool,
        fraction: float,
        gpu_memory_capacity_bytes_override: int | None,
        cuda_graph_reserved_bytes: int = 0,
    ) -> dict[str, Any]:
        """Native budget math over a breakdown dict (pure; no model build).

        Uses the shared :func:`kv_cache_budget_bytes` (``reserved=tolerance=0``;
        tolerance is applied separately by :func:`_apply_tolerance`). Floors to whole
        bytes/tokens. The token count comes from the breakdown's
        ``tokens_from_kv_bytes`` inverse (the model's KV-curve-aware
        ``get_kvcache_max_tokens``), so hybrid / sliding-window models do not get
        capacity extrapolated from the ``seq_len=1`` slope; when absent (e.g. a
        synthetic breakdown) it defaults to the linear
        :func:`_linear_tokens_from_bytes`. Raises ``ValueError`` for a non-positive
        per-token KV size or budget (these propagate; they do NOT trigger the naive
        fallback, which only fires when the breakdown itself cannot be built).
        """
        kv_per_token = float(breakdown["kv_size_per_token_bytes"])
        if not (math.isfinite(kv_per_token) and kv_per_token > 0.0):
            raise ValueError("insufficient model metadata; missing: kv_size_per_token_bytes")

        if gpu_memory_capacity_bytes_override is not None:
            capacity = float(gpu_memory_capacity_bytes_override)
        else:
            capacity = float(breakdown["gpu_memory_capacity_bytes"])
        if not (math.isfinite(capacity) and capacity > 0.0):
            raise ValueError(f"GPU capacity must be finite and positive, got {capacity}")

        cuda_graph = float(cuda_graph_reserved_bytes)
        pre_load_overhead = float(breakdown.get("pre_model_load_overhead_bytes", 0.0))
        non_kv = float(breakdown["non_kv_bytes"]) - pre_load_overhead + cuda_graph
        total_kv_size_bytes_f = kv_cache_budget_bytes(
            capacity=capacity - pre_load_overhead, non_kv=non_kv, fraction=fraction, of_free=is_of_free
        )

        if total_kv_size_bytes_f <= 0.0:
            raise ValueError(
                f"no KV budget: non-KV memory ({int(max(non_kv, 0.0))} bytes) meets/exceeds the "
                f"KV-cache memory limit (capacity={int(max(capacity, 0.0))} bytes)"
            )

        tokens_from_bytes: TokensFromBytes = breakdown.get("tokens_from_kv_bytes") or _linear_tokens_from_bytes(
            kv_per_token
        )
        total_kv_size_tokens = int(tokens_from_bytes(total_kv_size_bytes_f))

        return {
            "total_gpu_capacity_bytes": int(max(capacity, 0.0)),
            "total_kv_size_bytes": math.floor(total_kv_size_bytes_f),
            "kv_size_per_token_bytes": math.floor(kv_per_token),
            "total_kv_size_tokens": total_kv_size_tokens,
            **(
                {"kvcache_quant_mode": breakdown["kvcache_quant_mode"]}
                if breakdown.get("kvcache_quant_mode") is not None
                else {}
            ),
            # Carried so _apply_tolerance derives adjusted tokens the same way;
            # stripped by estimate_kv_cache before returning.
            "tokens_from_kv_bytes": tokens_from_bytes,
            "source": "native",
            "memory_breakdown": {
                "weights_bytes": int(max(float(breakdown["weights_bytes"]), 0.0)),
                "activations_bytes": int(max(float(breakdown["activations_bytes"]), 0.0)),
                "runtime_overhead_bytes": int(max(float(breakdown["runtime_overhead_bytes"]), 0.0)),
                "comm_overhead_bytes": int(max(float(breakdown["comm_overhead_bytes"]), 0.0)),
                "reserved_bytes": int(max(float(breakdown.get("reserved_bytes", 0.0)), 0.0)),
                "cuda_graph_reserved_bytes": cuda_graph_reserved_bytes,
            },
            "tolerance_adjusted": None,
        }


# --------------------------------------------------------------------------- #
# Naive fallback: NaiveKVCacheEstimator (HF-config-only) + reserve a fraction of
# post-weight memory.
#
# The estimator reuses the SDK's config loaders + `_parse_hf_config_json`, which
# REJECTS architectures AIC does not support -- exactly the fallback's target --
# so a supported arch (perf DB missing) uses the normalized parse, while an
# unsupported arch reads the raw config keys directly. Both feed one geometry dict
# that drives the per-token KV, weight, and token-capacity math.
# --------------------------------------------------------------------------- #


class NaiveKVCacheMode(enum.Enum):
    """Which KV-growth model :class:`NaiveKVCacheEstimator` inverts."""

    LINEAR = "linear"  # constant bytes/token (MHA/GQA/MLA, or an unknown layout)
    HYBRID = "hybrid"  # SWA layers cap at the window; global layers keep growing


class NaiveKVCacheEstimator:
    """Config-only KV-cache estimator for models AIC cannot build natively.

    Owns the HF-``config.json``-derived geometry and the byte<->token math the
    native backend memory model would otherwise provide, and mirrors the native
    :meth:`~aisim_core.sdk.models.base.Model.get_kvcache_max_tokens` surface so
    the naive fallback plugs into the same :data:`TokensFromBytes` pipeline.
    Handles standard K/V and MLA attention, and (via :class:`NaiveKVCacheMode`)
    hybrid sliding-window vs linear KV growth.

    The static ``_load_config`` / ``_dtype_bytes`` / ``_swa_layout`` / ``_geometry``
    helpers turn an HF config into the geometry dict; :meth:`from_model_path` wires
    them together. The instance methods compute the per-token / weight byte sizes
    and the token-capacity inverse from that geometry.
    """

    def __init__(
        self,
        geometry: dict,
        *,
        dtype_bytes: float,
        tp_size: int,
        pp_size: int,
        moe_ep_size: int = 1,
        moe_tp_size: int = 1,
    ):
        self.geometry = geometry
        # HF ``architectures[0]`` when built from a config; callers route model-specific layouts by it.
        self.architecture: str | None = None
        self.dtype_bytes = dtype_bytes  # KV precision, fractional for NVFP4
        self.weight_dtype_bytes = dtype_bytes
        self._weight_scale_group: tuple[int, int] | None = None
        self._vllm_fp8_linear: dict | None = None
        self._embedding_dtype_bytes = dtype_bytes
        self._quantized_weights = False
        self.tp_size = validate_parallel_size("tp_size", tp_size)
        self.pp_size = validate_parallel_size("pp_size", pp_size)
        self.moe_ep_size = validate_parallel_size("moe_ep_size", moe_ep_size)
        self.moe_tp_size = validate_parallel_size("moe_tp_size", moe_tp_size)
        self._v4_cache_mode: KVCacheQuantMode | None = None
        self._selected_decoder: dict | None = None
        self._is_nested_decoder = False
        self._wrapper_config: dict | None = None
        self._wrapper_requires_residency = False
        self._sglang_layer_prefix = "model.layers.{}."
        self._scalar_cache_problem: str | None = None
        self._nested_index_head_dim: int | None = None
        self._sparse_index_layers: int | None = None
        self._sparse_cache_needs_target = False
        self._cache_backend: str | None = None
        self._cache_release: Version | None = None
        self._vllm_native_decoder = False
        self._transformers_wrapper = False
        self._vllm_check_model_dtype = False
        self._vllm_model_dtype: str | None = None
        self._vllm_dtype_target: tuple[str | None, str | list[str] | None] = (None, None)
        self._checkpoint_dir: Path | None = None
        self._vllm_sparse_auto_precision: tuple[float, bool] | None = None
        self._vllm_packed_sparse = False

    @classmethod
    def from_model_path(
        cls,
        model_path: str,
        *,
        tp_size: int,
        pp_size: int,
        allow_hf_config_download: bool,
        moe_ep_size: int = 1,
        moe_tp_size: int = 1,
        backend: str = "vllm",
        backend_version: str | None = None,
        system: str | None = None,
        systems_paths: str | list[str] | None = None,
    ) -> NaiveKVCacheEstimator:
        """Build from an HF ``config.json`` (local dir / pre-cached / optional download).

        A config of a supported architecture goes through the normalized parse, which
        also unwraps registered wrappers; a config that the parse refuses is refused
        here with its message. A config of an unsupported architecture, the target of
        this fallback, is read from its raw keys, through its text decoder when it
        nests one. Raises ``ValueError`` when no config can be obtained offline.
        """
        hf_config = cls._load_config(model_path, allow_hf_config_download)
        if hf_config is None:
            raise ValueError(
                f"insufficient model metadata: no HF config for {model_path!r} (not a local "
                "directory / not pre-cached, and HF download is disabled)"
            )
        key = decoder_config_key(hf_config)
        decoder = hf_config.get(key, hf_config)
        names = hf_config.get("architectures")
        architecture = names[0] if isinstance(names, list) and names and isinstance(names[0], str) else None
        gemma4 = isinstance(decoder, dict) and (
            architecture == "Gemma4ForConditionalGeneration" or decoder.get("model_type") == "gemma4_text"
        )
        fallback = isinstance(decoder, dict) and architecture not in ARCHITECTURE_TO_MODEL_FAMILY
        gemma_release = backend in ("sglang", "vllm") and gemma4 and gemma4_cache_requires_release(decoder, backend)
        version = backend_version
        if gemma_release or (fallback and (key is not None or backend == "vllm")):
            version = _config_only_backend_version(system, backend, backend_version, systems_paths)
        fallback = fallback and (key is not None or _uses_bare_decoder(hf_config, backend, version))
        if fallback or gemma_release:
            backend_version = version
        # vLLM installs an adjacent checkpoint declaration into an empty root before class or child dispatch.
        sidecar = None
        if fallback:
            if backend != "vllm" or hf_config.get("quantization_config") is None:
                if os.path.isdir(model_path):
                    sidecar = _load_local_quant_config(model_path)
                elif model_path in DefaultHFModels:
                    sidecar = _load_pre_downloaded_hf_quant_config(model_path)
                elif allow_hf_config_download:
                    sidecar = _download_hf_json(model_path, "hf_quant_config.json", raise_on_404=False)
            if backend == "vllm" and sidecar is not None:
                hf_config = {**hf_config, "quantization_config": sidecar}
        estimator = cls._from_resolved_hf_config(
            hf_config,
            tp_size=tp_size,
            pp_size=pp_size,
            moe_ep_size=moe_ep_size,
            moe_tp_size=moe_tp_size,
            backend=backend,
            backend_version=backend_version,
            checkpoint_dir=Path(model_path) if os.path.isdir(model_path) else None,
            dtype_target=(system, systems_paths),
        )

        if backend == "vllm" and gemma4 and estimator.architecture not in _VLLM_DENSE_DECODERS:
            check_gemma4_adapter_runtime(
                hf_config, backend_version, tp_size, system=system, systems_paths=systems_paths
            )

        if estimator._selected_decoder is not None:
            if sidecar is not None and not estimator._vllm_native_decoder:
                estimator._selected_decoder = attach_quantization_sidecar(estimator._selected_decoder, sidecar)
            raw = estimator._selected_decoder
            if (sidecar is not None and not estimator._vllm_native_decoder) or (
                system is not None and estimator._nested_index_head_dim is not None
            ):
                target = _config_only_system_spec(system, systems_paths) if estimator._nested_index_head_dim else {}
                estimator._set_checkpoint_cache_precision(raw, backend, backend_version, hf_config, system_spec=target)
        elif "deepseek_v4" in estimator.geometry:
            estimator._set_checkpoint_cache_precision(
                _load_model_config_from_model_path(model_path), backend, backend_version
            )
        return estimator

    @classmethod
    def from_hf_config(
        cls,
        hf_config: dict,
        *,
        tp_size: int,
        pp_size: int,
        moe_ep_size: int = 1,
        moe_tp_size: int = 1,
        backend: str = "vllm",
        backend_version: str | None = None,
        system: str | None = None,
        systems_paths: str | list[str] | None = None,
    ) -> NaiveKVCacheEstimator:
        """Build directly from an already-loaded HF ``config.json`` dictionary.

        Callers that already loaded a config can reuse it without issuing a second HF request.
        """
        backend_version = _config_only_backend_version(system, backend, backend_version, systems_paths)
        estimator = cls._from_resolved_hf_config(
            hf_config,
            tp_size=tp_size,
            pp_size=pp_size,
            moe_ep_size=moe_ep_size,
            moe_tp_size=moe_tp_size,
            backend=backend,
            backend_version=backend_version,
            dtype_target=(system, systems_paths),
        )
        if backend == "vllm" and estimator.architecture not in _VLLM_DENSE_DECODERS:
            check_gemma4_adapter_runtime(
                hf_config, backend_version, tp_size, system=system, systems_paths=systems_paths
            )
        return estimator

    @classmethod
    def _from_resolved_hf_config(
        cls,
        hf_config: dict,
        *,
        tp_size: int,
        pp_size: int,
        moe_ep_size: int,
        moe_tp_size: int,
        backend: str,
        backend_version: str | None,
        resolve_bare: bool = True,
        checkpoint_dir: Path | None = None,
        dtype_target: tuple[str | None, str | list[str] | None] = (None, None),
    ) -> NaiveKVCacheEstimator:
        if (
            backend == "vllm"
            and hf_config.get("quantization_config") is None
            and hf_config.get("hf_quant_config") is not None
        ):
            hf_config = {**hf_config, "quantization_config": hf_config["hf_quant_config"]}
        architectures = hf_config.get("architectures") or []
        if not isinstance(architectures, list) or not all(isinstance(name, str) for name in architectures):
            raise ValueError(f"HF config architectures must be a list of strings, got {architectures!r}")
        architecture = next(iter(architectures), None)
        # V4.1 shares a compressed representation with source-layer ownership and position-dependent growth.
        # The original V4 checkpoints, including sgl-project's deepseek_ref exports, have no such ownership.
        model_type = hf_config.get("model_type")
        deepseek_v4 = decoder_config_key(hf_config) is None and (
            model_type in ("deepseek_v4", "deepseek_ref") or architecture == "DeepseekV4ForCausalLM"
        )
        if deepseek_v4 and architecture is None:
            architecture = "DeepseekV4ForCausalLM"
            hf_config = {**hf_config, "architectures": [architecture]}
        try:
            parsed = _parse_hf_config_json(hf_config)  # supported arch -> normalized parse, or its refusal
        except Exception as error:
            if architecture in ARCHITECTURE_TO_MODEL_FAMILY:
                raise ValueError(
                    f"the config parser refuses the HF config of the supported architecture {architecture}: "
                    f"{type(error).__name__}: {error}"
                ) from error
            parsed = None  # unsupported arch (the fallback's target) -> raw-key read, of its decoder when nested
            key = decoder_config_key(hf_config)
            if key is not None or (resolve_bare and _uses_bare_decoder(hf_config, backend, backend_version)):
                if key is None:
                    raw_decoder = dict(hf_config)
                else:
                    problem = decoder_config_problem(key, hf_config[key])
                    if problem is not None:
                        raise ValueError(f"HF config: {problem}") from error
                    raw_decoder = inherit_decoder_precision(hf_config, hf_config[key])[0]
                # The parent config constructs the child before vLLM selects the wrapper implementation.
                parent_type = (
                    model_type if isinstance(model_type, str) else _TEXT_CONFIG_DEFAULTS.get(architecture, (None,))[0]
                )
                if key == "text_config" and parent_type in _TEXT_DEFAULTS_BY_MODEL_TYPE:
                    raw_decoder.setdefault("model_type", _TEXT_DEFAULTS_BY_MODEL_TYPE[parent_type])
                needs_wrapper_weights = False
                transformers_wrapper = False
                text_only = False
                parent_output_head = False
                if key is not None and backend == "vllm":
                    needs_wrapper_weights, transformers_wrapper, native_text = wrapper_runtime(
                        architectures, backend_version, hf_config.get("auto_map"), _TEXT_ONLY_DECODERS
                    )
                    architecture = native_text or select_wrapper_architecture(
                        architectures,
                        backend_version,
                        hf_config.get("auto_map"),
                        _TEXT_WRAPPERS if key == "text_config" else set(),
                    )
                    text_only = native_text is not None
                    if text_only:
                        if key == "text_config" and parent_type in _QWEN3_VL_TEXT_DEFAULTS:
                            raw_decoder["model_type"] = f"{parent_type}_text"
                            for field, default in _QWEN3_VL_TEXT_DEFAULTS[parent_type].items():
                                raw_decoder.setdefault(field, default)
                        # vLLM remaps llm_config to text_config before these constructors read get_text_config().
                        parent_output_head = architecture in (
                            "MiMoForCausalLM",
                            "MiMoV2FlashForCausalLM",
                            "Qwen3ForCausalLM",
                        )
                        child_reader = parent_output_head or architecture in ("Qwen2ForCausalLM", "Qwen3MoeForCausalLM")
                        if not child_reader:
                            raw_decoder = {k: v for k, v in hf_config.items() if k not in ("llm_config", "text_config")}
                        raw_decoder["architectures"] = [architecture]
                elif key is not None and backend == "sglang":
                    architecture, needs_wrapper_weights, transformers_wrapper = _sglang_wrapper(
                        architectures, backend_version, hf_config, raw_decoder
                    )
                selected_type, selected = _select_text_decoder(
                    raw_decoder, None if text_only else key, architecture, parent_type, backend, backend_version
                )
                resolved_vllm_decoder = backend == "vllm" and (
                    text_only
                    or (key == "text_config" and architecture in _TEXT_WRAPPERS)
                    or (
                        key is None
                        and selected is not None
                        and (not architecture or selected == (raw_decoder.get("architectures") or [None])[0])
                    )
                )
                constructor = "deepseek_v3" if selected_type == "kimi_k2" else selected_type
                _apply_decoder_geometry_defaults(raw_decoder, constructor)
                # These vLLM output heads read the parent, while their layers use the constructed text config.
                if parent_output_head and any(
                    hf_config.get(k) is None or hf_config[k] != raw_decoder.get(k)
                    for k in ("vocab_size", "hidden_size")
                ):
                    raise ValueError(
                        f"{architecture} sizing requires matching parent and decoder head dimensions"
                    ) from error
                if selected_type in _DECODER_HEAD_DEFAULTS or (
                    resolved_vllm_decoder and selected in _VLLM_DENSE_DECODERS
                ):
                    # These projection classes read the canonical field, while SGLang sizes its pool separately.
                    canonical = raw_decoder.get("num_key_value_heads") or raw_decoder.get("num_attention_heads")
                    pool_heads = next(
                        (raw_decoder[k] for k in ("n_head_kv", "num_kv_heads") if raw_decoder.get(k) is not None), None
                    )
                    if backend == "sglang" and (pool_heads is not None or raw_decoder.get("multi_query")):
                        if selected_type == "gemma4_text":
                            canonical = gemma4_cache_shapes(
                                raw_decoder, backend, backend_version, agreement_tp_size=tp_size
                            )[1][0]
                        if (1 if raw_decoder.get("multi_query") else pool_heads) != canonical:
                            raise ValueError(
                                "SGLang pool KV heads disagree with the selected decoder's KV projections"
                            ) from error
                    for alias in ("attention_head_dim", "num_kv_heads", "n_head_kv", "n_kv_heads", "multi_query"):
                        raw_decoder.pop(alias, None)
                if resolved_vllm_decoder and selected in _VLLM_DENSE_DECODERS:
                    # Dense attention and FFNs ignore the source config's MLA, indexer and expert dimensions.
                    for field in (
                        "kv_lora_rank",
                        "qk_rope_head_dim",
                        "index_head_dim",
                        "index_n_heads",
                        "index_topk",
                        "num_experts",
                        "num_local_experts",
                        "n_routed_experts",
                        "moe_num_experts",
                    ):
                        raw_decoder.pop(field, None)
                elif constructor in _DECODER_EXPERT_DEFAULTS:
                    field, alias, default, width, default_width = _DECODER_EXPERT_DEFAULTS[constructor]
                    count = raw_decoder.get(alias, raw_decoder.get(field, default))
                    for name in ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts"):
                        raw_decoder.pop(name, None)
                    raw_decoder[field] = layout_integer(count, field)
                    raw_decoder["moe_intermediate_size"] = layout_integer(
                        raw_decoder.setdefault(width, default_width), width
                    )
                if selected_type == "gemma3_text":
                    if "sliding_window_size" not in raw_decoder:
                        raw_decoder.setdefault("sliding_window", 4096)
                    if raw_decoder.get("layer_types") is None:
                        raw_decoder.pop("layer_types", None)
                        layout_integer(raw_decoder.setdefault("sliding_window_pattern", 6), "sliding_window_pattern")
                    if raw_decoder.get("use_bidirectional_attention") and raw_decoder.get("sliding_window"):
                        raw_decoder["sliding_window"] = (
                            layout_integer(raw_decoder["sliding_window"], "sliding_window") // 2 + 1
                        )
                if raw_decoder.get("model_type") == "gemma4_text":
                    raw_decoder.setdefault("global_head_dim", 512)
                    raw_decoder.setdefault("sliding_window", 512)
                if not raw_decoder.get("architectures") and selected_type in (
                    "deepseek_v32",
                    "glm_moe_dsa",
                    "step3p5",
                    "step3p7",
                ):
                    # These layouts require the native parser even when the class is omitted.
                    raw_decoder = {**raw_decoder, "architectures": [selected]}
                decoder = cls._from_resolved_hf_config(
                    # The native timing parser also projects context and vocabulary. A selected
                    # decoder can supply KV geometry without either; weights still require vocabulary.
                    {"max_position_embeddings": None, "vocab_size": None, **raw_decoder},
                    tp_size=tp_size,
                    pp_size=pp_size,
                    moe_ep_size=moe_ep_size,
                    moe_tp_size=moe_tp_size,
                    backend=backend,
                    backend_version=backend_version,
                    resolve_bare=False,
                    checkpoint_dir=checkpoint_dir,
                    dtype_target=dtype_target,
                )
                if (
                    raw_decoder.get("model_type") == "gemma4_text"
                    and raw_decoder.get("use_bidirectional_attention") == "all"
                ):
                    # The layer-map reader above consumes the original serialized window.
                    window = layout_integer(raw_decoder["sliding_window"], "sliding_window") // 2 + 1
                    decoder.geometry["sliding_window"] = raw_decoder["sliding_window"] = window
                if not resolved_vllm_decoder and hf_config.get("hf_quant_config") is not None:
                    raw_decoder = attach_quantization_sidecar(raw_decoder, hf_config["hf_quant_config"])
                # The family supplies deployment capabilities. Ordinary raw geometry does not require a
                # complete native timing config (for example max_position_embeddings).
                decoder.architecture = decoder.architecture or selected
                if backend == "sglang":
                    if architecture == "LlavaForConditionalGeneration":
                        decoder._sglang_layer_prefix = "language_model.model.layers.{}."
                    elif selected_type in ("qwen3_vl_text", "qwen3_vl_moe_text"):
                        decoder._sglang_layer_prefix = "model.language_model.layers.{}."
                    quant = _sglang_checkpoint_quant(hf_config, raw_decoder)
                    if quant is not None:
                        raw_decoder = {**raw_decoder, "quantization_config": quant}
                if (
                    (backend == "vllm" or (backend == "sglang" and (key is None or architecture in _TEXT_WRAPPERS)))
                    and constructor == "deepseek_v3"
                    and selected in ("DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM")
                ):
                    decoder.geometry["num_routed_layers"] = len(
                        _deepseek_routed_layers(raw_decoder, decoder.geometry["layers"])
                    )
                decoder._is_nested_decoder = key is not None and not text_only
                if decoder._is_nested_decoder:
                    decoder._wrapper_config = {**hf_config, "architectures": [architecture]}
                    decoder._wrapper_requires_residency = needs_wrapper_weights
                    decoder._transformers_wrapper = transformers_wrapper
                decoder._vllm_native_decoder = resolved_vllm_decoder
                decoder._checkpoint_dir = checkpoint_dir
                decoder._vllm_dtype_target = dtype_target
                release = parse_support_matrix_version(backend_version)
                if (
                    decoder._vllm_native_decoder
                    and not raw_decoder.get("is_causal", True)
                    and (
                        selected
                        in (
                            "LlamaForCausalLM",
                            "MistralForCausalLM",
                            "Qwen2ForCausalLM",
                            "Qwen3ForCausalLM",
                            "Gemma3ForCausalLM",
                            "ApertusForCausalLM",
                        )
                        or (
                            selected in ("DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM")
                            and (release is None or release >= Version("0.29.0rc1"))
                        )
                    )
                ):
                    # These native layers construct EncoderOnlyAttention, whose get_kv_cache_spec returns None.
                    decoder._scalar_cache_problem = f"{selected} with is_causal=False has no persistent vLLM KV cache"
                decoder._selected_decoder = _attach_inferred_quant_fields(dict(raw_decoder))
                decoder._resolve_nested_cache(raw_decoder, architecture, backend, backend_version)
                decoder._set_checkpoint_cache_precision(decoder._selected_decoder, backend, backend_version, hf_config)
                decoder.architecture = decoder.architecture or architecture
                return decoder
        if deepseek_v4:
            if architecture != "DeepseekV4ForCausalLM" or model_type not in (
                None,
                "deepseek_v4",
                "deepseek_ref",
            ):
                raise ValueError(f"model_type {model_type} conflicts with architecture {architecture}")
            if any(hf_config.get(field) for field in ("kv_source_layer_ids", "index_source_layer_ids")):
                raise ValueError("source-layer cache ownership requires native sequence geometry or a runtime profile")
            ratios = hf_config.get("compress_ratios")
            layers = layout_integer(hf_config.get("num_hidden_layers"), "num_hidden_layers")
            mtp = layout_integer(hf_config.get("num_nextn_predict_layers", 0), "num_nextn_predict_layers", minimum=0)
            if (
                not isinstance(ratios, list)
                or len(ratios) not in (layers, layers + mtp)
                or any(type(ratio) is not int or ratio not in {0, 4, 128} for ratio in ratios)
                or not any(ratios[:layers])
                or any(ratios[layers:])
            ):
                raise ValueError(
                    "DeepSeek V4 compress_ratios must cover the decoder and optional MTP layers "
                    "with supported integer ratios and uncompressed MTP layers"
                )
        estimator = cls(
            cls._geometry(hf_config, parsed, backend, backend_version, tp_size),
            dtype_bytes=cls._dtype_bytes(
                hf_config,
                generic=parsed is None
                and not (isinstance(model_type, str) and model_type in _SELECTED_DECODER_ARCHITECTURES),
            ),
            tp_size=tp_size,
            pp_size=pp_size,
            moe_ep_size=moe_ep_size,
            moe_tp_size=moe_tp_size,
        )
        estimator.architecture = architecture
        effective = hf_config.get(MULTIMODAL_TEXT_CONFIG_KEY.get(architecture), hf_config)
        if effective.get("model_type") == "deepseek_v41_text" or (
            effective.get("compress_ratios") is not None and "deepseek_v4" not in estimator.geometry
        ):
            estimator._scalar_cache_problem = (
                "compressed KV storage varies with position; use the native sequence geometry or a runtime profile"
            )
        if "deepseek_v4" in estimator.geometry:
            estimator._set_checkpoint_cache_precision(
                _attach_inferred_quant_fields(dict(hf_config)), backend, backend_version
            )
        return estimator

    def _set_checkpoint_cache_precision(
        self,
        raw: dict,
        backend: str,
        backend_version: str | None,
        document: dict | None = None,
        *,
        system_spec: dict | None = None,
    ) -> None:
        if "deepseek_v4" in self.geometry:
            self._v4_cache_mode = _infer_quant_modes_from_raw_config(raw, self.architecture).get("kvcache_quant_mode")
            return
        release = parse_support_matrix_version(backend_version)
        self._cache_backend = backend
        self._cache_release = release
        # vLLM resolves model dtype from the parent, text, vision, then encoder config.
        parent = document or {}
        dtype = next(
            (
                value
                for config in (parent, raw, parent.get("vision_config"), parent.get("encoder_config"))
                if isinstance(config, dict) and (value := config.get("dtype") or config.get("torch_dtype"))
            ),
            None,
        )
        # These model types cannot run FP16 in vLLM and are recast by its automatic dtype selection.
        model_type = parent.get("model_type") or raw.get("model_type")
        self._vllm_model_dtype = dtype
        self._vllm_check_model_dtype = (
            backend == "vllm"
            and release is not None
            and release >= Version("0.32.1rc0")
            and model_type not in _VLLM_NO_FP16_MODEL_TYPES
        )
        mode = _automatic_decoder_kv_mode(document or {}, raw, backend, backend_version)
        if self._nested_index_head_dim is not None:
            sm_version = (system_spec or {}).get("gpu", {}).get("sm_version")
            unknown_gpu = sm_version is None and not perf_database.is_xpu_system_spec(system_spec)
            layers = self.geometry["layers"]
            stage_layers = layers // self.pp_size
            self._sparse_index_layers = stage_layers
            pattern = raw.get("index_topk_pattern")
            if backend == "sglang" and release is not None and release >= Version("0.5.18"):
                indexers = raw.get("indexer_types") if release >= Version("0.5.20") else None
                if indexers is not None:
                    if not isinstance(indexers, list):
                        raise ValueError("SGLang indexer_types must be a layer list")
                    pattern = ["S" if kind == "shared" else "F" for kind in indexers]
                elif raw.get("cli_factor") is not None:
                    factor = layout_integer(raw["cli_factor"], "cli_factor")
                    if factor > 1:
                        pattern = ["S" if i % factor else "F" for i in range(layers)]
            if pattern is not None:
                raw = {**raw, "index_topk_pattern": pattern, "index_topk_freq": 1}
            elif (
                backend == "sglang"
                and release is not None
                and release >= Version("0.5.14")
                and raw.get("index_skip_topk_offset") is not None
            ):
                layout_integer(raw["index_skip_topk_offset"], "index_skip_topk_offset")
            if (
                raw.get("index_topk_freq", 1) != 1
                or raw.get("index_topk_pattern")
                or (
                    backend == "sglang"
                    and release is None
                    and (raw.get("indexer_types") is not None or raw.get("cli_factor"))
                )
            ):
                # Older releases computed shared indices but still allocated an index cache at every layer.
                shares_storage = (backend == "vllm" and release is not None and release >= Version("0.24.0rc1")) or (
                    backend == "sglang" and release is not None and release >= Version("0.5.18")
                )
                if raw.get("index_topk_pattern") is None and release is not None:
                    freq = raw.get("index_topk_freq", 1)
                    if backend == "vllm" and (
                        shares_storage or (release >= Version("0.21.0") and raw.get("use_index_cache"))
                    ):
                        if type(freq) is not int or not freq:
                            raise ValueError("vLLM sparse index_topk_freq must be a nonzero integer")
                        if freq < 0:
                            raw = {**raw, "index_topk_freq": abs(freq)}
                    if backend == "sglang" and release >= Version("0.5.14") and freq is not None:
                        layout_integer(freq, "index_topk_freq")
                if shares_storage:
                    cumulative = [
                        round(end * _dsa_full_layer_fraction(raw, end)) for end in range(0, layers + 1, stage_layers)
                    ]
                    counts = {end - start for start, end in pairwise(cumulative)}
                    self._sparse_index_layers = counts.pop() if len(counts) == 1 else None
                elif release is None:
                    self._sparse_index_layers = None
            if backend == "sglang":
                # SGLang selects the DSA pool before loading the quantized weights.
                bf16_sinks = raw.get("learnable_sink") and release is not None and release >= Version("0.5.20")
                self._sparse_cache_needs_target = not bf16_sinks and unknown_gpu
                mode = (
                    KVCacheQuantMode.fp8
                    if sm_version is not None and sm_version >= 100 and not bf16_sinks
                    else KVCacheQuantMode.bfloat16
                )
            elif backend == "vllm":
                # FlashMLA on Hopper keeps the model dtype under auto, independently of FP8 expert weights.
                declared = raw.get("kv_cache_quant_algo")
                if mode is None and declared in ("fp8", "nvfp4"):
                    mode = KVCacheQuantMode[declared]
                # From 0.24 the SM120 backend changes auto to fp8_ds_mla even on a weight-only checkpoint.
                self._sparse_cache_needs_target = (
                    mode is None and unknown_gpu and (release is None or release >= Version("0.24.0rc1"))
                )
                sm120 = sm_version is not None and 120 <= sm_version < 130
                self._vllm_packed_sparse = sm120 and (release is None or release >= Version("0.24.0rc1"))
                if sm_version is not None and (
                    self._vllm_packed_sparse
                    or (
                        100 <= sm_version < 110
                        and self.architecture == "GlmMoeDsaForCausalLM"
                        and (release is None or release >= Version("0.32.0"))
                    )
                ):
                    mode = mode or KVCacheQuantMode.fp8
            else:
                mode = _infer_quant_modes_from_raw_config(raw, self.architecture).get("kvcache_quant_mode")
        model_dtype_bytes = self.weight_dtype_bytes
        if self._transformers_wrapper and (backend == "vllm" or model_dtype_bytes == 4):
            # vLLM uses its supported GPU dtype; SGLang downcasts FP32 to FP16 or BF16.
            model_dtype_bytes = 2
        self._quantized_weights = bool(
            raw.get("quantization_config") or raw.get("hf_quant_config") or raw.get("quant_algo")
        )
        if self._vllm_native_decoder:
            # with_hf_config retains the parent's execution dtype. On supported GPU targets vLLM
            # keeps FP16/BF16 or casts to the preferred one, including for FP32 and FP8 inputs.
            model_dtype_bytes = 2
            # A cache scheme alone does not quantize any linear weights. The compressed-tensors
            # target map prices weights independently; reserve model dtype for mixed or unknown targets.
            quant = _vllm_weight_config(parent or raw)
            if quant is None:
                quant = raw.get("hf_quant_config") or {}
            self.weight_dtype_bytes = model_dtype_bytes
            self._embedding_dtype_bytes = model_dtype_bytes
            self._weight_scale_group = None
            self._vllm_fp8_linear = None
            method = str(quant.get("quant_method", "")).lower().replace("_", "-")
            self._quantized_weights = bool(quant or raw.get("quant_algo"))
            if method.startswith("modelopt") or method == "mxfp8" or "quantization" in quant:
                section = quant.get("quantization", quant)
                if not isinstance(section, dict):
                    raise ValueError("ModelOpt quantization must be a mapping")
                algo = str(section.get("quant_algo") or "").upper()
                if not algo and method in ("modelopt-mxfp8", "mxfp8") and "quantization" not in quant:
                    algo = "MXFP8"
                if method.startswith("modelopt") and not algo:
                    raise ValueError("ModelOpt checkpoint is missing quant_algo")
                if algo and algo not in {
                    "FP8",
                    "FP8_PER_CHANNEL_PER_TOKEN",
                    "FP8_PB_WO",
                    "MXFP8",
                    "NVFP4",
                    "W4A16_NVFP4",
                    "MIXED_PRECISION",
                }:
                    raise ValueError(f"unsupported ModelOpt quant_algo {algo!r}")
                if method in ("modelopt-mxfp8", "mxfp8") and algo != "MXFP8":
                    raise ValueError("ModelOpt MXFP8 checkpoints require quant_algo MXFP8")
                if algo == "MXFP8" and self._vllm_execution_dtype(
                    needs_preferred=True, allow_fp16=model_type not in _VLLM_NO_FP16_MODEL_TYPES
                ) not in ("bf16", "bfloat16"):
                    raise ValueError("vLLM ModelOpt MXFP8 requires bfloat16 execution")
                self._quantized_weights = bool(algo)
                ignored = section.get("exclude_modules", []) if "quantization" in quant else quant.get("ignore", [])
                if method in ("modelopt-mxfp8", "mxfp8") and not section.get("quant_algo"):
                    ignored = quant.get("ignored_layers", [])
                if not isinstance(ignored, list) or not all(isinstance(target, str) for target in ignored):
                    raise ValueError("ModelOpt exclusions must be a list of module names")
                if (
                    algo == "MXFP8"
                    and self.architecture in _VLLM_DENSE_DECODERS
                    and not self.geometry.get("num_experts")
                ):
                    _check_modelopt_mxfp8_inputs(
                        self.geometry,
                        self.tp_size,
                        ignored,
                        self._wrapper_config["architectures"][0] if self._wrapper_config else None,
                        self.architecture == "OPTForCausalLM",
                    )
                # For mixed or potentially excluded linears retain the execution-width reserve.
                if all(_outside_dense_decoder_projections(target.strip("*")) for target in ignored):
                    if algo == "FP8" and self.architecture in _VLLM_DENSE_DECODERS - {"OPTForCausalLM"}:
                        self.weight_dtype_bytes = 1
                        self._vllm_fp8_linear = {"activation_scheme": "static"}
                    elif (
                        algo == "MXFP8"
                        and self.architecture in _VLLM_DENSE_DECODERS
                        and not self.geometry.get("num_experts")
                    ):
                        # Only dense residency includes rank-local scale storage. Keep the execution-width MoE reserve.
                        self.weight_dtype_bytes = 1
                        self._weight_scale_group = (32, 1)  # E8M0 scale per group of MXFP8 weights.
            elif method == "compressed-tensors":
                groups = quant.get("config_groups", {})
                if not isinstance(groups, dict):
                    raise ValueError("compressed-tensors config_groups must be a mapping")
                weight_groups = []
                for group in groups.values():
                    targets = group.get("targets") if isinstance(group, dict) else None
                    if not isinstance(targets, list) or not all(isinstance(target, str) for target in targets):
                        raise ValueError("compressed-tensors targets must be a list of module names")
                    if not targets or (
                        len(targets) == 1
                        and targets[0].endswith("Attention")
                        and release is not None
                        and release >= Version("0.15.0rc0")
                    ):
                        # vLLM applies attention quantization through kv_cache_scheme.
                        continue
                    weights = group.get("weights")
                    if not isinstance(weights, dict):
                        raise ValueError("compressed-tensors groups require a weight scheme")
                    if weights.get("actorder") not in (None, False, "static", "weight") and (
                        release is None or release >= Version("0.30.0rc1")
                    ):
                        raise ValueError("vLLM does not support this compressed-tensors weight activation ordering")
                    weight_groups.append(group)
                self._quantized_weights = bool(weight_groups)
                if self.pp_size == 1 and self.architecture in _VLLM_DENSE_DECODERS and len(weight_groups) == 1:
                    group = weight_groups[0]
                    weights = dict(group["weights"])
                    weights["strategy"] = _compressed_tensor_strategy(weights)
                    weights["type"] = _compressed_tensor_type(weights)
                    ignored = quant.get("ignore", [])
                    if isinstance(ignored, list) and self.architecture != "OPTForCausalLM":
                        _check_fused_decoder_exclusions(
                            ignored, self.geometry["layers"], "language_model." if self._is_nested_decoder else ""
                        )
                    if (
                        group["targets"] == ["Linear"]
                        and isinstance(ignored, list)
                        and all(_outside_dense_decoder_projections(target) for target in ignored)
                        and weights.get("num_bits", 8) == 8
                        and weights.get("type") in ("float", "int")
                        and weights.get("symmetric", True)
                        and weights.get("strategy") in ("tensor", "channel", "block", "group")
                        and not weights.get("dynamic", False)
                    ):
                        group_format = group.get("format")
                        root_format = quant.get("format")
                        if group_format is None or (release is not None and release < Version("0.10.1rc1")):
                            group_format = root_format
                        elif release is None and group_format != root_format:
                            raise ValueError("a compressed-tensors group format override requires the vLLM release")
                        activation_format = group_format in (
                            "naive-quantized",
                            "int-quantized",
                            "float-quantized",
                            "nvfp4-pack-quantized",
                        )
                        # v0.10.1 reads activation formats per group, but still dispatches packed weights globally.
                        packed_format = (
                            root_format if release is not None and release < Version("0.10.2rc1") else group_format
                        )
                        inputs, outputs = group.get("input_activations"), group.get("output_activations")
                        static_int8_io = (
                            release is not None
                            and release >= Version("0.23.0rc2")
                            and all(
                                isinstance(args, dict)
                                and _compressed_tensor_type(args) == "int"
                                and args.get("num_bits", 8) == 8
                                and _compressed_tensor_strategy(args) == "tensor"
                                and not args.get("dynamic", False)
                                for args in (inputs, outputs)
                            )
                        )
                        packed_dynamic = (
                            release is not None
                            and release >= Version("0.26.0rc1")
                            and isinstance(inputs, dict)
                            and _compressed_tensor_type(inputs) == "int"
                            and inputs.get("num_bits", 8) in (4, 8)
                            and _compressed_tensor_strategy(inputs) in ("token", "group")
                            and inputs.get("dynamic") is True
                            and inputs.get("symmetric", True)
                        )
                        packed = (
                            packed_format == "pack-quantized"
                            and weights["strategy"] in ("channel", "group")
                            and (inputs is None or (weights["type"] == "int" and (static_int8_io or packed_dynamic)))
                        )
                        if activation_format and weights["type"] == "int" and not packed:
                            # For W8A8, vLLM supports static per-tensor or dynamic per-token activations.
                            if not isinstance(inputs, dict) or not inputs:
                                raise ValueError("vLLM compressed-tensors W8A8 requires an input activation scheme")
                            activation_format = (
                                inputs.get("num_bits", 8) == 8
                                and (_compressed_tensor_strategy(inputs), inputs.get("dynamic", False))
                                in (("tensor", False), ("token", True))
                                and weights["strategy"] in ("tensor", "channel")
                            )
                        if not activation_format and not packed:
                            raise ValueError(
                                f"vLLM cannot load these compressed-tensors W8 weights in {group_format!r}"
                            )
                        if weights["strategy"] != "group":
                            self.weight_dtype_bytes = 1
                        elif packed and weights["type"] == "int" and inputs is None:
                            size = layout_integer(weights.get("group_size"), "compressed-tensors W8 group_size")
                            geom = self.geometry
                            hidden, inter = geom["hidden"], geom["inter"]
                            heads, dim = geom["num_attention_heads"], geom["head_dim"]
                            sharded = size * self.tp_size
                            partitioned_scales = weights.get("actorder") in (None, False, "static", "weight")
                            if (
                                size in (32, 64, 128)
                                and partitioned_scales
                                and all((hidden, inter, heads, dim))
                                and not (hidden % size or inter % sharded or heads * dim % sharded)
                            ):
                                self._weight_scale_group = (size, model_dtype_bytes)
                                self.weight_dtype_bytes = 1
            elif method == "fp8":
                if quant.get("activation_scheme") not in ("dynamic", "static"):
                    raise ValueError("vLLM FP8 checkpoints require a static or dynamic activation_scheme")
                if quant.get("weight_block_size") is not None:
                    fp8_weight_block_shape(quant, sglang=False)
                    if quant["activation_scheme"] != "dynamic":
                        raise ValueError("vLLM block-quantized FP8 weights require dynamic activations")
                ignored = quant.get("ignored_layers") or quant.get("modules_to_not_convert") or []
                if isinstance(ignored, list) and all(_outside_dense_decoder_projections(target) for target in ignored):
                    if self.architecture in _VLLM_DENSE_DECODERS and not self.geometry.get("num_experts"):
                        self.weight_dtype_bytes = (
                            1  # Fp8LinearMethod allocates FP8 weights for BF16 and FP16 execution.
                        )
                        self._vllm_fp8_linear = quant
                    else:
                        self.weight_dtype_bytes = self._dtype_bytes(raw)
            elif self._quantized_weights and raw.get("quant_algo"):
                self.weight_dtype_bytes = self._dtype_bytes(raw)
        self.dtype_bytes = mode.value.memory if mode is not None else model_dtype_bytes
        if (
            backend == "vllm"
            and self.architecture == "DeepseekV32ForCausalLM"
            and self._nested_index_head_dim is not None
        ):
            # The model rewrites an explicit BF16 *after* automatic checkpoint precision was resolved.
            self._vllm_sparse_auto_precision = (
                KVCacheQuantMode.fp8.value.memory if self._vllm_packed_sparse else model_dtype_bytes,
                unknown_gpu and (release is None or release >= Version("0.24.0rc1")),
            )

    def _resolve_nested_cache(self, decoder: dict, wrapper: str | None, backend: str, version: str | None) -> None:
        """Use the selected decoder's storage declarations before admitting it to the generic scalar estimator."""
        layers = self.geometry.get("layers")
        if self.pp_size > 1 and (
            not layers
            or layers % self.pp_size
            or (self.geometry.get("num_swa_layers") and self.geometry.get("num_global_layers"))
            or "deepseek_v4" in self.geometry
        ):
            self._scalar_cache_problem = (
                "per-rank pipeline KV sizing requires the resident attention layers on each stage"
            )
        if "deepseek_v4" in self.geometry or self._scalar_cache_problem:
            return
        if backend == "sglang" and not self.geometry.get("kv_lora_rank"):
            release = parse_support_matrix_version(version)
            head_dim = self.geometry.get("head_dim")
            value_dim = decoder.get("v_head_dim", head_dim)
            if release is not None and (
                (value_dim is None and release >= Version("0.5.14"))
                or (value_dim == 0 and release >= Version("0.5.19"))
            ):
                value_dim = head_dim
            sliding_dim = self.geometry.get("swa_head_dim") or head_dim
            sliding_value = decoder.get("swa_v_head_dim", sliding_dim)
            if sliding_value is None and release is not None and release >= Version("0.5.14"):
                sliding_value = sliding_dim
            if (release is None or release >= Version("0.5.7")) and (
                value_dim != head_dim
                or (
                    self.geometry.get("num_swa_layers")
                    and (release is None or release >= Version("0.5.9"))
                    and sliding_value != sliding_dim
                )
            ):
                self._scalar_cache_problem = (
                    "asymmetric SGLang key/value head dimensions require runtime cache geometry"
                )
                return
        types = decoder.get("layer_types") or ()
        if "attn_type_list" in decoder:
            attn_type_list(decoder, self.geometry.get("layers"))
        index_fields = ("index_head_dim", "index_n_heads", "index_topk")
        indexed = any(decoder.get(field) is not None for field in index_fields) or "deepseek_sparse_attention" in types
        # The layer declaration and the checkpoint model type both survive removal of HF's optional architecture.
        dsa = ARCHITECTURE_TO_MODEL_FAMILY.get(self.architecture) == "DEEPSEEKV32" or (
            types and set(types) == {"deepseek_sparse_attention"}
        )
        identity = "DeepseekV32ForCausalLM" if dsa else self.architecture or decoder.get("model_type")
        markers = cache_layout_markers(decoder, identity if isinstance(identity, str) else None)
        if (
            markers
            or decoder.get("hybrid_override_pattern")
            or decoder.get("layers_block_type")
            or wrapper == "InklingForConditionalGeneration"
            or decoder.get("use_sconv")
            or decoder.get("local_layer_ids")
        ):
            self._scalar_cache_problem = (
                "config-only sizing of this decoder requires runtime state reservations or cache groups"
            )
            return
        # Step's parser supplies per-group dimensions for its alternate attention settings.
        per_layer = (
            "sparse_attention_config",
            "block_configs",
            "hybrid_layer_pattern",
            "attention_chunk_size",
            "num_kv_shared_layers",
            "kv_source_layer_ids",
            "index_source_layer_ids",
        )
        step = ARCHITECTURE_TO_MODEL_FAMILY.get(self.architecture) == "STEP3P7"
        # vLLM uses Step's decoder-wide KV heads on every attention layer.
        separate_heads = not (step and backend == "vllm") and any(
            decoder.get(key) for key in ("swa_num_key_value_heads", "swa_head_dim")
        )
        if (
            any(decoder.get(key) for key in per_layer)
            or (decoder.get("per_layer_config") and decoder.get("model_type") != "gemma4_text")
            or (separate_heads and not (self.geometry.get("swa_kv_heads") or self.geometry.get("swa_head_dim")))
            or (decoder.get("attention_other_setting") and not step)
        ):
            self._scalar_cache_problem = "per-layer cache sizing of this decoder requires runtime cache geometry"
            return
        if indexed or dsa:
            if (
                not dsa
                or (types and set(types) != {"deepseek_sparse_attention"})
                or not all(decoder.get(k) for k in ("kv_lora_rank", "qk_rope_head_dim", *index_fields))
            ):
                self._scalar_cache_problem = (
                    "sparse decoder cache sizing requires the complete supported MLA and indexer geometry"
                )
                return
            for key in ("kv_lora_rank", "qk_rope_head_dim", *index_fields):
                layout_integer(decoder[key], key)
            self._nested_index_head_dim = decoder["index_head_dim"]
            if backend == "sglang" and self._nested_index_head_dim != 128:
                self._scalar_cache_problem = "SGLang's DSA KV pool requires index_head_dim=128"
            if backend == "vllm" and self._nested_index_head_dim % 128:
                self._scalar_cache_problem = "vLLM's sparse indexer requires index_head_dim divisible by 128"

    # ----------------------------- config -> geometry ----------------------------- #

    @staticmethod
    def _load_config(model_path: str, allow_hf_config_download: bool) -> dict | None:
        """Load a raw HF ``config.json`` dict, or ``None`` when unavailable offline.

        Reuses the SDK loaders: a local directory -> the pre-downloaded
        ``model_configs`` cache -> an HF download (ONLY when
        ``allow_hf_config_download`` is set). Returns ``None`` when no config can be
        obtained without downloading; :meth:`from_model_path` raises in that case.
        """
        if os.path.isdir(model_path):
            return _load_local_config(model_path)
        if model_path in DefaultHFModels:
            return _load_pre_downloaded_hf_config(model_path)
        if allow_hf_config_download:
            return _download_hf_config(model_path)
        return None

    @staticmethod
    def _dtype_bytes(hf_config: dict, *, generic: bool = False) -> int:
        """dtype byte width from ``torch_dtype``/``dtype``. Defaults to 2 (bf16)."""
        value = hf_config.get("torch_dtype")
        if not isinstance(value, str) or not value.strip():
            value = hf_config.get("dtype")
        if not isinstance(value, str) or not value.strip():
            # An unregistered decoder can carry an untyped torch_dtype. Its generic estimate defaults to BF16.
            if generic and value is None and not isinstance(hf_config.get("torch_dtype"), str):
                return 2
            if hf_config.get("torch_dtype") is not None or value is not None:
                raise ValueError("HF config torch_dtype or dtype must be a nonempty string")
            return 2
        dtype_str = value.strip().lower()
        if dtype_str in ("float32", "fp32"):
            return 4
        if dtype_str in ("bfloat16", "float16", "bf16", "fp16", "half"):
            return 2
        if "fp8" in dtype_str or "float8" in dtype_str or "int8" in dtype_str:
            return 1
        return 2

    @staticmethod
    def _raw_dimension(hf_config: dict, *keys: str) -> int | None:
        """Return the first usable raw-config dimension among equivalent HF keys.

        Hugging Face config families use several names for the same dimensions and
        may serialize an unused preferred key as ``null``. Only positive integers
        are dimensions; invalid preferred values are skipped in favor of an alias.
        """
        for key in keys:
            value = hf_config.get(key)
            if isinstance(value, bool):
                continue
            try:
                parsed = int(value)
            except (OverflowError, TypeError, ValueError):
                continue
            if isinstance(value, float) and not value.is_integer():
                continue
            if parsed > 0:
                return parsed
        return None

    @classmethod
    def _swa_layout(
        cls, hf_config: dict, parsed: dict | None = None, backend: str = "vllm"
    ) -> tuple[int | None, int | None, int | None]:
        """Sliding-window size + (SWA, global) layer counts from an HF config.

        Returns ``(sliding_window, num_swa_layers, num_global_layers)`` for a hybrid
        sliding-window model, or ``(None, None, None)`` when the layout cannot be
        pinned down (so the estimator stays on its linear per-token mode). Step-3.7
        uses its parsed decoder pattern. Otherwise two raw signals are recognized:

        - ``layer_types`` (Gemma-3/4-style explicit per-layer list): count the
          ``sliding_*`` entries vs the rest.
        - ``sliding_window`` + ``sliding_window_pattern`` (an integer ``P``): the
          Gemma convention of one global layer every ``P`` layers, the rest sliding.

        A bare ``sliding_window`` with no per-layer signal is left unresolved -- the
        split is too model-specific to guess, and linear is the safe default.
        """
        family = ARCHITECTURE_TO_MODEL_FAMILY.get(parsed.get("architecture")) if parsed is not None else None
        extra = parsed.get("extra_params") if parsed is not None else None
        normalized = family == "STEP3P7" or isinstance(extra, Gemma4MixConfig)
        if normalized:
            key = MULTIMODAL_TEXT_CONFIG_KEY.get(parsed["architecture"])
            hf_config = hf_config.get(key, hf_config)
        gpt_oss = hf_config.get("model_type") == "gpt_oss" or (parsed and parsed["architecture"] == "GptOssForCausalLM")
        enabled = hf_config.get("use_sliding_window")
        qwen_sliding = enabled is True and (
            hf_config.get("model_type") in ("qwen2", "qwen3", "qwen2_moe")
            or (hf_config.get("architectures") or [None])[0] in ("Qwen2ForCausalLM", "Qwen3ForCausalLM")
        )
        if qwen_sliding and "sliding_window" not in hf_config:
            hf_config = {**hf_config, "sliding_window": 4096}
        window = cache_window(hf_config)
        qwen_moe = hf_config.get("model_type") == "qwen2_moe"
        if qwen_sliding and not qwen_moe and hf_config["sliding_window"] is None:
            enabled = False
        if gpt_oss and "sliding_window" not in hf_config:
            window = 128  # GptOssConfig default.
        if enabled is not None and type(enabled) is not bool:
            raise ValueError("use_sliding_window must be a boolean")
        if enabled and not window and not (qwen_moe and hf_config.get("sliding_window") is None):
            raise ValueError("use_sliding_window=true requires a positive sliding window")
        num_layers = (
            cls._raw_dimension(parsed, "layers")
            if parsed is not None
            else cls._raw_dimension(hf_config, "num_hidden_layers", "n_layer", "num_layers")
        )
        gemma_text = isinstance(extra, Gemma4MixConfig) or hf_config.get("model_type") == "gemma4_text"
        if family == "STEP3P7":
            # The parser trims extra MTP layers from the external layer list to the actual decoder.
            num_swa = extra.attn_layer_pattern.count(0)
            num_global = extra.attn_layer_pattern.count(1)
        elif qwen_sliding and hf_config.get("layer_types") is None and num_layers:
            # Qwen2Config and Qwen3Config mark layers at and after max_window_layers as sliding.
            full = layout_integer(hf_config.get("max_window_layers", 28), "max_window_layers", minimum=0)
            if not enabled:
                num_swa, num_global = 0, num_layers
            elif qwen_moe:
                # Qwen2MoeConfig uses even-indexed layers strictly before max_window_layers.
                num_swa = (min(num_layers, full) + 1) // 2
                num_global = num_layers - num_swa
            else:
                num_global = min(num_layers, full)
                num_swa = num_layers - num_global
        elif gpt_oss and hf_config.get("layer_types") is None and num_layers:
            # GptOssConfig defaults even-numbered layers to sliding and odd-numbered layers to full attention.
            num_swa = (num_layers + 1) // 2
            num_global = num_layers - num_swa
        elif "layer_types" in hf_config and not (gemma_text and hf_config["layer_types"] is None):
            types = layer_types(hf_config, None)
            if num_layers and len(types) != num_layers:
                raise ValueError(f"layer_types has {len(types)} entries for {num_layers} decoder layers")
            num_swa = types.count("sliding_attention")
            # Other registered decoders declare sparse MLA or recurrent layers; they use no sliding-window split.
            if (num_swa or gemma_text or (parsed and parsed["architecture"] == "GptOssForCausalLM")) and any(
                name not in {"sliding_attention", "full_attention"} for name in types
            ):
                raise ValueError("layer_types must contain only 'sliding_attention' or 'full_attention'")
            # Gemma4TextConfig forces the final layer to full attention even for an explicit list.
            if gemma_text and types and types[-1] == "sliding_attention":
                num_swa -= 1
            num_global = len(types) - num_swa
        elif gemma_text and num_layers:
            # Gemma4TextConfig defaults every sixth layer and the final layer to full attention.
            num_global = (num_layers + 5) // 6
            num_swa = num_layers - num_global
        elif window and num_layers and (pattern := hf_config.get("sliding_window_pattern")):
            interval = layout_integer(pattern, "sliding_window_pattern")
            num_global = num_layers // interval
            num_swa = num_layers - num_global
        else:
            return None, None, None
        if enabled is False and num_swa:
            raise ValueError("use_sliding_window=false conflicts with sliding attention layers")
        names = hf_config.get("architectures")
        implementation = names[0] if names else _SELECTED_DECODER_ARCHITECTURES.get(hf_config.get("model_type"))
        if implementation in ("Qwen2ForCausalLM", "Qwen3ForCausalLM") and num_swa:
            if backend == "sglang" and not hf_config.get("is_hybrid_swa"):
                return None, None, None
            if backend == "vllm" and num_global and hf_config.get("max_window_layers", 28) != num_layers:
                name = implementation.removesuffix("ForCausalLM")
                raise ValueError(f"vLLM {name} does not support sliding attention for only some layers")
        return window, num_swa, num_global

    @classmethod
    def _geometry(
        cls,
        hf_config: dict,
        parsed: dict | None,
        backend: str = "vllm",
        backend_version: str | None = None,
        tp_size: int = 1,
    ) -> dict:
        """Normalized attention/FFN scalars the byte formulas consume.

        From :func:`~aisim_core.sdk.utils._parse_hf_config_json` output when the
        architecture is supported (``parsed`` not ``None``), else from the HF config
        keys directly (unsupported architecture). Missing values stay ``None`` /
        ``0`` so the formulas can fail closed.
        """
        sliding_window, num_swa_layers, num_global_layers = cls._swa_layout(hf_config, parsed, backend)
        if parsed is not None:
            extra = parsed.get("extra_params")
            if isinstance(extra, dict):
                kv_lora = extra.get("kv_lora_rank")
                qk_rope = extra.get("qk_rope_head_dim")
            else:
                # Some supported archs carry a dataclass extra_params (e.g. NemotronH,
                # Gemma4Mix, DeepSeek-V4); read MLA latent fields by attribute when
                # present. V4's compressed cache uses its own geometry below.
                kv_lora = getattr(extra, "kv_lora_rank", None)
                qk_rope = getattr(extra, "qk_rope_head_dim", None)
            heads = parsed.get("n_kv") or parsed.get("n")
            dim = parsed.get("d")
            swa_heads = swa_dim = swa_queries = None
            if isinstance(extra, Gemma4MixConfig):
                decoder = hf_config.get("text_config", hf_config)
                (swa_heads, swa_dim), (heads, dim) = gemma4_cache_shapes(
                    decoder, backend, backend_version, agreement_tp_size=tp_size
                )
            elif ARCHITECTURE_TO_MODEL_FAMILY.get(parsed.get("architecture")) == "STEP3P7":
                key = MULTIMODAL_TEXT_CONFIG_KEY.get(parsed["architecture"])
                decoder = hf_config.get(key, hf_config)
                other = decoder.get("attention_other_setting") or {}
                swa_queries = decoder.get("num_sliding_attention_heads")
                if not swa_queries and other.get("attention_type") == "sliding_attention":
                    swa_queries = other.get("num_attention_heads")
                if backend == "sglang":
                    # The SGLang pool uses sliding KV heads. vLLM uses the decoder-wide count on both types.
                    attention_heads = other.get("num_attention_groups", heads)
                    swa_heads = layout_integer(
                        decoder.get("swa_num_key_value_heads", attention_heads), "sliding attention KV heads"
                    )
                    if "num_attention_groups" in other and swa_heads != attention_heads:
                        raise ValueError("Step sliding attention KV heads disagree with the SGLang cache pool")
                    if other.get("num_attention_heads"):
                        swa_queries = other["num_attention_heads"]
                if not decoder.get("head_dim"):
                    if decoder.get("is_heterogeneous"):
                        raise ValueError("Step heterogeneous query heads require an explicit head_dim for cache sizing")
                    if swa_queries:
                        swa_dim = parsed["hidden_size"] // layout_integer(swa_queries, "sliding attention query heads")
            types = getattr(extra, "layer_types", ()) or hf_config.get("layer_types") or ()
            return {
                **({"deepseek_v4": extra} if isinstance(extra, DeepSeekV4Config) else {}),
                **({"recurrent_cache": True} if "linear_attention" in types else {}),
                "layers": parsed.get("layers"),
                "num_attention_heads": parsed.get("n"),
                "swa_query_heads": swa_queries,
                "num_kv_heads": heads,
                "head_dim": dim,
                **({"swa_kv_heads": swa_heads, "swa_head_dim": swa_dim} if swa_heads or swa_dim else {}),
                "kv_lora_rank": kv_lora,
                "qk_rope_head_dim": qk_rope,
                "vocab": parsed.get("vocab"),
                "hidden": parsed.get("hidden_size"),
                "inter": parsed.get("inter_size"),
                "num_experts": parsed.get("num_experts") or 0,
                "moe_inter": parsed.get("moe_inter_size") or 0,
                "sliding_window": sliding_window,
                "num_swa_layers": num_swa_layers,
                "num_global_layers": num_global_layers,
            }

        dimension = cls._raw_dimension
        hidden = dimension(hf_config, "hidden_size", "n_embd", "n_embed")
        layers = dimension(hf_config, "num_hidden_layers", "n_layer", "num_layers")
        heads = dimension(hf_config, "num_attention_heads", "n_head", "num_heads")
        head_dim = dimension(hf_config, "head_dim", "attention_head_dim")
        if head_dim is None and hidden and heads:
            head_dim = int(hidden) // int(heads)
        num_experts = (
            dimension(
                hf_config,
                "n_routed_experts",
                "num_local_experts",
                "num_experts",
            )
            or 0
        )
        inter = dimension(hf_config, "intermediate_size", "ffn_dim", "n_inner", "ffn_hidden_size")
        if inter is None and hidden and not num_experts:
            # Dense decoder families conventionally use a 4x hidden FFN when the
            # config omits the intermediate dimension (or serializes it as null).
            inter = 4 * int(hidden)
        num_kv_heads = dimension(hf_config, "num_key_value_heads", "num_kv_heads", "n_head_kv", "n_kv_heads")
        # Legacy Falcon configs retain num_kv_heads=num_attention_heads while
        # multi_query selects one shared K/V head. New-decoder configs use the
        # explicit KV-head count instead.
        if hf_config.get("multi_query") and not hf_config.get("new_decoder_architecture"):
            num_kv_heads = 1
        swa_heads = swa_dim = None
        if hf_config.get("model_type") == "gemma4_text":
            if "intermediate_size" not in hf_config:
                inter = GEMMA4_DEFAULT_INTERMEDIATE_SIZE
            (swa_heads, swa_dim), (num_kv_heads, head_dim) = gemma4_cache_shapes(
                hf_config, backend, backend_version, agreement_tp_size=tp_size
            )
        return {
            "layers": layers,
            **({"recurrent_cache": True} if "linear_attention" in (hf_config.get("layer_types") or ()) else {}),
            **({"swa_kv_heads": swa_heads, "swa_head_dim": swa_dim} if swa_heads or swa_dim else {}),
            "num_attention_heads": heads,
            "num_kv_heads": num_kv_heads or heads,
            "head_dim": head_dim,
            "kv_lora_rank": hf_config.get("kv_lora_rank"),
            "qk_rope_head_dim": hf_config.get("qk_rope_head_dim"),
            "vocab": hf_config.get("vocab_size"),
            "hidden": hidden,
            "inter": inter,
            "num_experts": num_experts,
            "moe_inter": hf_config.get("moe_intermediate_size") or inter or 0,
            "sliding_window": sliding_window,
            "num_swa_layers": num_swa_layers,
            "num_global_layers": num_global_layers,
        }

    # --------------------------- geometry -> byte sizes --------------------------- #

    def kv_bytes_per_token(self) -> float | None:
        """Per-token KV byte size (the ``seq_len=1`` rate), MLA-aware, or ``None``.

        Standard K/V and MLA/compressed-latent attention are handled separately
        (MLA stores one shared latent cache per layer, not per-head K and V):

        - MLA (``kv_lora_rank`` present): ``layers * (kv_lora_rank + qk_rope_head_dim)
          * dtype_bytes`` (the latent is shared across heads, NOT sharded by TP).
          Fails closed (``None``) when ``qk_rope_head_dim`` is absent so the caller
          raises rather than guessing the latent width.
        - A selected sparse MLA decoder also stores a scaled FP8 index. Its RoPE
          remains BF16 when the latent is quantized.
        - Standard (GQA/MHA): ``2 * ceil(num_kv_heads / tp) * head_dim * layers *
          dtype_bytes``.
        """
        geom = self.geometry
        if self._selected_decoder is not None or self.architecture in ARCHITECTURE_TO_MODEL_FAMILY:
            self._validate_attention_tp()
        if self._scalar_cache_problem:
            raise ValueError(self._scalar_cache_problem)
        layers = geom.get("layers")
        if self._selected_decoder is not None and self.pp_size > 1:
            layers //= self.pp_size  # _resolve_nested_cache established uniform, equally sized stages.
        if geom.get("kv_lora_rank") and self.dtype_bytes == KVCacheQuantMode.nvfp4.value.memory:
            raise ValueError("an nvfp4 KV cache is modeled only for standard K/V attention")
        if self._nested_index_head_dim is not None:
            if self._sparse_cache_needs_target:
                raise ValueError(
                    "automatic sparse MLA cache precision requires the target GPU or a runtime cache profile"
                )
            if self.dtype_bytes != KVCacheQuantMode.bfloat16.value.memory:
                # FlashMLA stores per-128 latent scales and BF16 RoPE. Other sparse backends use a different layout.
                raise ValueError("quantized sparse MLA requires the runtime latent and indexer cache formats")
            if self._sparse_index_layers is None:
                raise ValueError("shared sparse indices require a resolved release and uniform pipeline stage geometry")
            latent = (geom["kv_lora_rank"] + geom["qk_rope_head_dim"]) * self.dtype_bytes
            return layers * latent + self._sparse_index_layers * indexer_cache_entry_bytes(self._nested_index_head_dim)
        if "deepseek_v4" in geom:
            from aisim_core.sdk.memory.deepseek_v4 import deepseek_v4_state_cache

            if self._v4_cache_mode not in (None, KVCacheQuantMode.fp8):
                raise ValueError("DeepSeek V4 stores fp8_ds_mla KV; use kvcache_quant_mode fp8")
            return deepseek_v4_state_cache(geom["deepseek_v4"], indexer_cache_dtype="auto")["kv_bytes_per_token"]
        if not layers:
            return None

        kv_lora = geom.get("kv_lora_rank")
        if kv_lora:
            qk_rope = geom.get("qk_rope_head_dim")
            if not qk_rope:
                # MLA but latent width unclear -> fail closed (None; the caller raises).
                return None
            return int(layers) * (int(kv_lora) + int(qk_rope)) * self.dtype_bytes

        if not geom.get("num_kv_heads") or not geom.get("head_dim"):
            return None
        if (geom.get("swa_kv_heads") or geom.get("swa_head_dim")) and geom.get("num_swa_layers") is not None:
            return sum(self._attention_group_rates())
        return self._kv_layer_bytes(geom["num_kv_heads"], geom["head_dim"]) * int(layers)

    def _validate_attention_tp(self) -> None:
        for key in ("num_attention_heads", "swa_query_heads"):
            heads = self.geometry.get(key)
            if heads and layout_integer(heads, key) % self.tp_size:
                raise ValueError(f"{key} {heads} must be divisible by tp_size {self.tp_size}")
        for key in ("num_kv_heads", "swa_kv_heads"):
            heads = self.geometry.get(key)
            if heads and max(heads, self.tp_size) % min(heads, self.tp_size):
                raise ValueError(f"{key} {heads} cannot partition or replicate uniformly over tp_size {self.tp_size}")

    def _kv_layer_bytes(self, kv_heads: int, head_dim: int) -> float:
        heads_per_rank = -(-int(kv_heads) // max(self.tp_size, 1))
        return 2 * heads_per_rank * int(head_dim) * self.dtype_bytes

    def _attention_layer_rates(self) -> tuple[float, float]:
        geom = self.geometry
        sliding = self._kv_layer_bytes(
            geom.get("swa_kv_heads") or geom["num_kv_heads"], geom.get("swa_head_dim") or geom["head_dim"]
        )
        full = self._kv_layer_bytes(geom["num_kv_heads"], geom["head_dim"])
        return sliding, full

    def _attention_group_rates(self) -> tuple[float, float]:
        sliding, full = self._attention_layer_rates()
        stages = self.pp_size if self._selected_decoder is not None else 1
        return sliding * (self.geometry["num_swa_layers"] // stages), full * (
            self.geometry["num_global_layers"] // stages
        )

    def _local_vllm_model_dtype(self) -> str:
        """Resolve the floating weight dtype from local safetensors headers when the config omits it."""
        problem = "a resolved vLLM model dtype requires config or checkpoint headers"
        if self._checkpoint_dir is None:
            raise ValueError(problem)
        dtypes = set()
        for path in self._checkpoint_dir.glob("*.safetensors"):
            try:
                with path.open("rb") as weights:
                    size = int.from_bytes(weights.read(8), "little")
                    if not 0 < size <= 100_000_000:  # Safetensors' maximum header size.
                        raise ValueError("invalid safetensors header length")
                    header = json.loads(weights.read(size))
                for name, info in header.items():
                    if name != "__metadata__":
                        dtype = info["dtype"]
                        if not isinstance(dtype, str):
                            raise ValueError("invalid safetensors dtype")
                        dtypes.add(dtype)
            except (OSError, ValueError, AttributeError, KeyError, TypeError) as exc:
                raise ValueError(f"{problem}: {path.name}") from exc
        # The serving reader ignores types absent from safetensors.torch's dtype map, including E8M0 scales.
        dtypes &= _VLLM_FP8_WEIGHT_DTYPES.keys() | {
            "F16",
            "BF16",
            "F32",
            "F64",
            "C64",
            "BOOL",
            "U8",
            "U16",
            "U32",
            "U64",
            "I8",
            "I16",
            "I32",
            "I64",
        }
        if not dtypes:
            return "float32"  # vLLM's default when no weights report a dtype.
        floating = {dtype for dtype in dtypes if dtype.startswith(("F", "BF", "C"))} - _VLLM_FP8_WEIGHT_DTYPES.keys()
        if not floating:
            for header, dtype in _VLLM_FP8_WEIGHT_DTYPES.items():
                if header in dtypes:
                    return dtype  # All FP8 formats resolve to the same preferred execution dtype.
        if "F64" in floating and floating <= {"F16", "BF16", "F32", "F64"}:
            return "float64"
        if "F32" in floating and floating <= {"F16", "BF16", "F32"}:
            return "float32"
        if floating == {"BF16"}:
            return "bfloat16"
        if floating == {"F16"}:
            return "float16"
        raise ValueError(problem)

    def _vllm_execution_dtype(self, *, needs_preferred: bool = False, allow_fp16: bool = True) -> str:
        dtype = self._vllm_model_dtype or self._local_vllm_model_dtype()
        fp8 = dtype in _VLLM_FP8_WEIGHT_DTYPES.values()
        if dtype in ("float16", "fp16", "half") and allow_fp16:
            return dtype
        if fp8 or dtype in ("float32", "fp32", "float", "float64", "double", "bfloat16", "bf16") or not allow_fp16:
            spec = _config_only_system_spec(*self._vllm_dtype_target)
            sm = spec.get("gpu", {}).get("sm_version")
            if sm is not None:
                return "bfloat16" if sm >= 80 else "float16"
            if spec.get("gpu") and perf_database.is_xpu_system_spec(spec):
                return "bfloat16"
            if fp8 or dtype in ("float64", "double") or (needs_preferred and dtype not in ("bfloat16", "bf16")):
                raise ValueError("resolving the preferred vLLM model dtype requires a deployment target")
        return dtype

    def set_kvcache_quant_mode(self, mode: str) -> None:
        """Apply an explicit KV precision independently of the checkpoint's weight width."""
        if mode == "bfloat16" and self._vllm_sparse_auto_precision is not None:
            # vLLM rewrites DeepSeek V3.2 BF16 to auto before the backend selects its physical cache.
            self.dtype_bytes, self._sparse_cache_needs_target = self._vllm_sparse_auto_precision
            return
        if mode == "bfloat16" and self._vllm_check_model_dtype:
            dtype = self._vllm_execution_dtype()
            if dtype in ("float16", "fp16", "half"):
                raise ValueError("explicit bfloat16 KV cache dtype disagrees with the vLLM float16 model dtype")
        if mode == "bfloat16" and self._vllm_packed_sparse:
            raise ValueError("the vLLM SM120 sparse backend requires the packed FP8 MLA cache")
        if mode == "int8" and self._cache_backend in ("vllm", "sglang"):
            raise ValueError(f"{self._cache_backend} has no literal int8 KV cache dtype")
        if mode == "nvfp4" and self._cache_release is not None:
            if self._cache_backend == "vllm" and self._cache_release < Version("0.20.0rc1"):
                raise ValueError("vLLM NVFP4 KV cache requires backend version 0.20.0 or later")
            if self._cache_backend == "sglang" and self._cache_release < Version("0.5.16"):
                raise ValueError("SGLang NVFP4 KV cache requires backend version 0.5.16 or later")
        if mode == "nvfp4" and (self.geometry.get("kv_lora_rank") or self.architecture == "DeepseekV41ForCausalLM"):
            raise ValueError("an nvfp4 KV cache is modeled only for standard K/V attention")
        try:
            quantization = KVCacheQuantMode[mode]
            self.dtype_bytes = quantization.value.memory
            self._sparse_cache_needs_target = False
            if "deepseek_v4" in self.geometry:
                self._v4_cache_mode = quantization
        except KeyError as error:
            raise ValueError(f"unsupported kvcache_quant_mode {mode!r}") from error

    def weight_bytes(self) -> int | None:
        """Estimate the whole rank's resident decoder and colocated auxiliary weights."""
        decoder = self._decoder_weight_bytes()
        if decoder is None or self._wrapper_config is None:
            return decoder
        document = self._wrapper_config
        wrapper = document["architectures"][0]
        vision = document.get("vision_config")
        family = document.get("model_type") or _TEXT_CONFIG_DEFAULTS.get(wrapper, (None,))[0]
        llava = wrapper in _TEXT_CONFIG_DEFAULTS and wrapper.startswith("Llava")
        intern = wrapper in _TEXT_CONFIG_DEFAULTS and wrapper.startswith("Intern")
        clip = family in ("llava", "llava_next", "llava_next_video")
        if any(document.get(key) is not None for key in ("audio_config", "encoder_config", "audio_tower")):
            raise ValueError("whole-rank capacity requires resident weights for the wrapper's auxiliary encoder")
        if vision is None:
            if family == "llava_onevision":
                vision = dict(
                    hidden_size=1152, intermediate_size=4304, num_hidden_layers=26, image_size=384, patch_size=14
                )
            elif clip:
                vision = dict(
                    hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, image_size=336, patch_size=14
                )
            elif family == "internvl":
                vision = {}
            elif (
                wrapper not in _TEXT_WRAPPERS
                and not self._wrapper_requires_residency
                and not document.get("vision_tower")
                and not document.get("mm_vision_tower")
                and document.get("projector_config") is None
                and document.get("mm_projector_type") is None
            ):
                return decoder
        default_kind = (
            "internvl_vision" if family == "internvl" else ("clip_vision_model" if clip else "siglip_vision_model")
        )
        vision_kind = vision.get("model_type", default_kind) if isinstance(vision, dict) else None
        if not (
            (llava and vision_kind in ("siglip_vision_model", "clip_vision_model"))
            or (intern and vision_kind in ("interns1_vision", "internvl_vision"))
        ):
            raise ValueError("whole-rank capacity requires resident weights for the vision tower and projector")

        # Reserve the entire ViT and connector on a rank. This bounds replicated or sharded encoder
        # execution without assuming text TP also partitions either the vision tower or the projector.
        def pair(value: object, field: str) -> tuple[int, int]:
            if type(value) is int:
                value = (value, value)
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"vision_config.{field} requires two spatial dimensions")
            return layout_integer(value[0], field), layout_integer(value[1], field)

        # LLaVA supplies its own dimensions for an omitted tower. A partial dict uses the vision class defaults.
        if vision_kind == "internvl_vision":
            defaults = (1024, 4096, 24, 448, 14)
        elif llava:
            defaults = (768, 3072, 12, 224, 32 if vision_kind == "clip_vision_model" else 16)
        else:
            defaults = (None, None, None, None, None)
        hidden = layout_integer(vision.get("hidden_size", defaults[0]), "vision hidden_size")
        inter = layout_integer(vision.get("intermediate_size", defaults[1]), "vision intermediate_size")
        layers = layout_integer(vision.get("num_hidden_layers", defaults[2]), "vision num_hidden_layers")
        channels = layout_integer(vision.get("num_channels", 3), "vision num_channels")
        ih, iw = pair(vision.get("image_size", defaults[3]), "image_size")
        ph, pw = pair(vision.get("patch_size", defaults[4]), "patch_size")
        text = self.geometry["hidden"]
        features = document.get("vision_feature_layer")
        projector_in = hidden * (len(features) if llava and isinstance(features, list) else 1)
        if intern:
            ratio = document.get("downsample_ratio", 0.5)
            if type(ratio) not in (float, int) or not 0 < ratio <= 1 or not (1 / ratio).is_integer():
                raise ValueError("vision projector requires an inverse-integer downsample_ratio")
            projector_in *= int(1 / ratio) ** 2
        # Q/K/V/O and two MLP weights per block, patch embeddings, learned positions and both projector matrices.
        matrices = layers * (4 * hidden**2 + 2 * hidden * inter) + channels * ph * pw * hidden
        matrices += ((ih // ph) * (iw // pw) + 1) * hidden + (projector_in + text) * text
        # Reserve projection biases, norms, layer scales, special embeddings and connector norms at FP32.
        vectors = layers * (13 * hidden + inter) + 8 * hidden + 2 * projector_in + 3 * text
        return decoder + max(2, self._embedding_dtype_bytes) * matrices + 4 * vectors

    def _decoder_weight_bytes(self) -> int | None:
        """Rough per-rank weight estimate (bytes) from the geometry, or ``None``.

        Coarse param count (embeddings + per-layer attention + dense/MoE FFN), times
        dtype bytes, divided by TP*PP. NOT calibrated; for models AIC cannot fully
        model. Attention-DP replicates rather than shards, so it does NOT divide
        here. Hidden size and layer count accept their standard Hugging Face aliases;
        dense models without an explicit intermediate size use the conventional
        ``4 * hidden_size`` expansion. Returns ``None`` (the caller raises) when a
        required dimension is still missing, rather than fabricating a placeholder.
        """
        geom = self.geometry
        hidden = geom.get("hidden")
        layers = geom.get("layers")
        vocab = geom.get("vocab")
        inter = geom.get("inter")
        if not hidden or not layers or not vocab or not inter:
            return None
        hidden = int(hidden)
        layers = int(layers)
        vocab = int(vocab)
        inter = int(inter)

        if self._selected_decoder is not None and self.pp_size > 1:
            raw = self._selected_decoder
            if (
                self.architecture not in ("LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM")
                or self.tp_size != 1
                or layers % self.pp_size
                or (geom.get("num_swa_layers") and geom.get("num_global_layers"))
                or geom.get("num_experts")
                or self.weight_dtype_bytes < 2
                or self._quantized_weights
            ):
                raise ValueError("pipeline capacity requires the resident weights on each stage, including embeddings")
            heads, kv_heads, dim = geom.get("num_attention_heads"), geom.get("num_kv_heads"), geom.get("head_dim")
            if not all((heads, kv_heads, dim)) or not raw.get("intermediate_size"):
                raise ValueError(
                    "pipeline capacity requires the dense decoder's attention and FFN projection dimensions"
                )
            # At TP1 the endpoint owns one unsharded vocabulary matrix, padded to a multiple of 64 in both
            # runtimes. Each stage has Q/K/V/output projections and the three matrices of the gated MLP.
            padded_vocab = -(-vocab // 64) * 64
            resident_layers = layers // self.pp_size
            matrices = padded_vocab * hidden + resident_layers * (
                2 * hidden * dim * (heads + kv_heads) + 3 * hidden * inter
            )
            # Allow all projection biases, attention/MLP and Q/K norms, the output norm and vocabulary bias at FP32.
            vectors = (
                padded_vocab
                + hidden
                + resident_layers * ((heads + 2 * kv_heads) * dim + 4 * hidden + 2 * inter + 2 * dim)
            )
            return int(matrices * self.weight_dtype_bytes + 4 * vectors)

        if self.architecture == "OPTForCausalLM" and self._selected_decoder is not None:
            # The vocabulary and attention/FFN matrices shard over TP. OPT's learned
            # positions and optional input/output embedding projections are replicated.
            raw = self._selected_decoder
            embedding = layout_integer(raw.get("word_embed_proj_dim"), "word_embed_proj_dim")
            positions = layout_integer(raw.get("max_position_embeddings"), "max_position_embeddings")
            if self.weight_dtype_bytes < 2 or self._quantized_weights or inter % self.tp_size:
                raise ValueError("OPT capacity requires unquantized weights and a divisible FFN width")
            vocabulary = -(-vocab // 64) * 64
            matrices = 2 * vocabulary * embedding + layers * (4 * hidden**2 + 2 * hidden * inter)
            replicated = (positions + 2) * hidden
            if embedding != hidden:
                replicated += 2 * hidden * embedding
            # Reserve both vocabulary endpoints and all projection biases and norms.
            vectors = 2 * hidden + layers * (9 * hidden + inter)
            sharded = (matrices + self.tp_size - 1) // self.tp_size
            return int((sharded + replicated) * self.weight_dtype_bytes + 4 * vectors)

        # Embeddings: input + (untied) output projection ~ 2 * vocab * hidden.
        embed_params = 2 * vocab * hidden
        # Per-layer attention q/k/v/o ~ 4 * hidden**2.
        attn_params_per_layer = 4 * hidden * hidden

        n_experts = int(geom.get("num_experts") or 0)
        pp = max(self.pp_size, 1)
        tp = max(self.tp_size, 1)
        if n_experts > 0:
            moe_inter = int(geom.get("moe_inter") or inter)
            # Expert FFN is sharded by moe_tp * moe_ep, not by the attention TP.
            local_experts = -(-n_experts // self.moe_ep_size)
            expert_params = local_experts * 3 * hidden * moe_inter // self.moe_tp_size
            non_expert_bytes = (
                embed_params * self._embedding_dtype_bytes + layers * attn_params_per_layer * self.weight_dtype_bytes
            ) // (tp * pp)
            raw = self._selected_decoder
            if "num_routed_layers" in geom and raw is not None:
                routed = geom["num_routed_layers"]
                shared = raw.get("n_shared_experts", 1)
                shared = 0 if shared is None else layout_integer(shared, "n_shared_experts", minimum=0)
                q_rank = raw["q_lora_rank"] or 0
                kv_rank, rope, nope, value = (
                    raw[k] for k in ("kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim")
                )
                heads = raw["num_attention_heads"]
                if heads % tp or ((layers > routed) and inter % tp) or (routed and shared * moe_inter % tp):
                    raise ValueError("DeepSeek projections must partition over the tensor-parallel ranks")
                # MLA replicates the Q/KV A matrices and shards the Q/KV B and output matrices by head.
                attention = hidden * (q_rank + kv_rank + rope) + (heads // tp) * (
                    (q_rank or hidden) * (nope + rope) + kv_rank * (nope + value) + hidden * value
                )
                padded_vocab = -(-vocab // 64) * 64
                vocab_rows = -(-padded_vocab // tp)
                embeddings = 2 * vocab_rows * hidden * self._embedding_dtype_bytes
                matrices = layers * attention + routed * expert_params
                matrices += 3 * hidden * ((layers - routed) * inter + routed * shared * moe_inter) // tp
                # Norms, replicated gates and their correction biases may use FP32.
                vectors = hidden + layers * (2 * hidden + q_rank + kv_rank) + routed * n_experts * (hidden + 1)
                resident = embeddings + matrices * self.weight_dtype_bytes + 4 * vectors
                quant = raw.get("quantization_config") or {}
                if self._cache_backend == "sglang" and quant.get("quant_method") == "fp8":
                    activation = quant.get("activation_scheme")
                    if activation not in ("dynamic", "static"):
                        raise ValueError("SGLang FP8 experts require a static or dynamic activation_scheme")
                    ignored = quant.get("ignored_layers", quant.get("modules_to_not_convert"))
                    if ignored is None:
                        ignored = []
                    if not isinstance(ignored, list) or not all(isinstance(name, str) and name for name in ignored):
                        raise ValueError("SGLang FP8 exclusions must be a list of module names")
                    quantized = routed
                    if ignored:
                        quantized = (
                            sum(
                                not _sglang_skips_modules(
                                    ignored, (i,), self._cache_release, self._sglang_layer_prefix + "mlp.experts"
                                )
                                for i in _deepseek_routed_layers(raw, layers)
                            )
                            if self._cache_release is not None
                            else 0
                        )
                    shard = moe_inter // self.moe_tp_size
                    # w13 has two (shard, hidden) matrices and w2 has one (hidden, shard) matrix per local expert.
                    if quant.get("weight_block_size") is None:
                        scales = 3 + (2 if activation == "static" else 0)
                    else:
                        if activation != "dynamic":
                            raise ValueError("SGLang block-quantized FP8 weights require dynamic activations")
                        block_n, block_k = fp8_weight_block_shape(quant, sglang=True, experts=True)
                        scales = 2 * math.ceil(shard / block_n) * math.ceil(hidden / block_k)
                        scales += math.ceil(hidden / block_n) * math.ceil(shard / block_k)
                    resident += quantized * (expert_params * (1 - self.weight_dtype_bytes) + 4 * local_experts * scales)
                return int(resident)
            expert_bytes = layers * expert_params * self.weight_dtype_bytes // pp
            resident = non_expert_bytes + expert_bytes
            if (
                self._vllm_native_decoder
                and self.architecture == "Qwen3MoeForCausalLM"
                and raw.get("model_type") == "qwen3_vl_moe_text"
            ):
                # Qualify the rough reserve against the same rank-local inventory as the bare native decoder.
                problem = "Qwen3 MoE capacity requires measured rank-local resident weights"
                if (
                    self._quantized_weights
                    or self.weight_dtype_bytes != 2
                    or not raw.get("tie_word_embeddings", True)
                    or raw.get("mlp_only_layers")
                    or raw.get("decoder_sparse_step", 1) != 1
                    or raw.get("shared_expert_intermediate_size")
                    or raw.get("attention_bias")
                    or vocab % max(64, tp)
                    or tp != self.moe_tp_size * self.moe_ep_size
                    or n_experts < tp
                    or n_experts % self.moe_ep_size
                    or moe_inter % self.moe_tp_size
                ):
                    raise ValueError(problem)
                self._validate_attention_tp()
                config = build_model_config(tp, pp, 1, self.moe_tp_size, self.moe_ep_size)
                _apply_model_quant_defaults(config, raw, self.architecture, "vllm")
                family = ARCHITECTURE_TO_MODEL_FAMILY[self.architecture]
                info = {
                    **_parse_hf_config_json({"max_position_embeddings": None, **raw}),
                    "model_path": str(self._checkpoint_dir or ""),
                    "model_family": family,
                }
                native = _MODEL_REGISTRY[family].create(info, config, "vllm")
                # The native ops include the router and KV projections. Norms sit outside that inventory.
                norms = 4 * (hidden + layers * (2 * hidden + 2 * geom["head_dim"]))
                if resident < native.get_resident_weights_bytes() + norms:
                    raise ValueError(problem)
            return resident
        else:
            rank_vectors = 0
            if self._vllm_native_decoder and self.architecture in _VLLM_DENSE_DECODERS:
                # Keep the existing coarse reservation as a floor. Wide queries and
                # replicated KV heads can exceed four square matrices on a rank.
                heads, kv_heads, dim = geom["num_attention_heads"], geom["num_kv_heads"], geom["head_dim"]
                if heads and kv_heads and dim:
                    self._validate_attention_tp()
                    qkv_output = 2 * hidden * dim * (heads // tp + max(1, kv_heads // tp))
                    if self._wrapper_config and self._wrapper_config["architectures"][0] in _TEXT_CONFIG_DEFAULTS:
                        # The language decoder's rank-local Q/K/V/O matrices are known. Charge their actual
                        # size alongside the vision tower and allow FP32 norms, linear biases and channel scales.
                        if inter % tp:
                            raise ValueError(
                                "vLLM dense intermediate_size must be divisible by the tensor parallel size"
                            )
                        attn_params_per_layer = qkv_output * tp
                        output_rows = dim * (heads // tp + 2 * max(1, kv_heads // tp)) + 2 * hidden + 2 * inter // tp
                        rank_vectors = 4 * (hidden + layers * (2 * output_rows + 4 * hidden + 2 * dim))
                        padded_vocab = -(-vocab // 64) * 64
                        embed_params = 2 * (-(-padded_vocab // tp)) * tp * hidden
                    else:
                        attn_params_per_layer = max(attn_params_per_layer, qkv_output * tp)
            ffn_params_per_layer = 3 * hidden * inter
            layer_params = layers * (attn_params_per_layer + ffn_params_per_layer)
            weights = embed_params * self._embedding_dtype_bytes + layer_params * self.weight_dtype_bytes
            if self._weight_scale_group:
                size, scale_bytes = self._weight_scale_group
                weights += -(-layer_params // size) * scale_bytes
            if self._vllm_fp8_linear is not None:
                return (
                    rank_vectors
                    + weights // (tp * pp)
                    + (layers // pp) * _vllm_fp8_linear_scale_bytes(self._vllm_fp8_linear, geom, tp)
                )
            return rank_vectors + weights // (tp * pp)

    # --------------------------- byte budget -> tokens ---------------------------- #

    @property
    def mode(self) -> NaiveKVCacheMode:
        """HYBRID when the config exposes a usable sliding-window layout, else LINEAR."""
        return NaiveKVCacheMode.HYBRID if self._hybrid_tokens_from_bytes() is not None else NaiveKVCacheMode.LINEAR

    def _require_scalar_capacity(self) -> None:
        if self._scalar_cache_problem:
            raise ValueError(self._scalar_cache_problem)
        if "deepseek_v4" in self.geometry:
            raise ValueError("DeepSeek V4 packed cache capacity requires runtime group allocations")

    def get_kvcache_max_tokens(
        self, kv_budget_bytes: float, mode: NaiveKVCacheMode | None = None, *, max_batch_size: int = 1
    ) -> int:
        """KV byte budget -> token capacity, mirroring ``Model.get_kvcache_max_tokens``.

        ``mode`` defaults to :attr:`mode` (auto: HYBRID when a sliding-window layout
        is known, else LINEAR); pass it explicitly to force one. HYBRID caps the SWA
        layers at the window while global layers keep growing; LINEAR floor-divides
        the budget by the per-token size. An explicit HYBRID with no usable layout
        falls back to LINEAR. Each concurrent request may fill a whole sliding
        window. DeepSeek V4 exposes marginal token bytes only. Its packed groups
        require runtime allocations to determine token capacity.
        """
        validate_parallel_size("max_batch_size", max_batch_size)
        self._require_scalar_capacity()
        hybrid = self._hybrid_tokens_from_bytes(max_batch_size)
        use_hybrid = mode is NaiveKVCacheMode.HYBRID or (mode is None and hybrid is not None)
        if use_hybrid and hybrid is not None:
            return hybrid(kv_budget_bytes)
        return _linear_tokens_from_bytes(float(self.kv_bytes_per_token() or 0.0))(kv_budget_bytes)

    def estimate(
        self,
        *,
        capacity: int | None,
        naive_kv_reservation: float,
        cuda_graph_reserved_bytes: int = 0,
        max_batch_size: int = 1,
        scheduler_backend: str | None = None,
        scheduler_block_size: int | None = None,
        scheduler_version: str | None = None,
    ) -> dict[str, Any]:
        """Conservative naive estimate: reserve ``naive_kv_reservation`` of post-weight memory.

        The naive counterpart to :meth:`KVCacheEstimator.estimate`. ``capacity`` is
        the per-rank GPU memory (the ``gpu_memory_capacity_bytes_override`` the caller
        must supply, since the native SystemSpec capacity is unavailable when this
        path runs). Does NOT honor the requested ``memory_fraction`` -- the
        post-weight reservation is the crude budget. By default this direct method
        estimates retention after eviction. Shared scheduling also covers live
        prefill and independent pages for each attention group.
        RAISES ``ValueError`` when ``capacity`` is missing or the config lacks the
        metadata to estimate the weights / per-token KV (rather than guessing with a
        placeholder constant).
        """
        self._require_scalar_capacity()
        if (
            self._selected_decoder is not None
            and self.pp_size > 1
            and scheduler_backend not in (None, "vllm", "sglang")
        ):
            raise ValueError("pipeline capacity requires the resident weights on each stage, including embeddings")
        if scheduler_backend is not None:
            if self.geometry.get("recurrent_cache"):
                raise ValueError("shared capacity for recurrent layers requires runtime state reservations")
            if self._nested_index_head_dim is not None:
                raise ValueError("shared capacity for a nested sparse MLA decoder requires runtime indexer allocations")
        if not (capacity and int(capacity) > 0):
            raise ValueError(
                "naive fallback requires gpu_memory_capacity_bytes_override (the native "
                "path failed, so SystemSpec capacity is unavailable)"
            )
        capacity_bytes = float(capacity)

        kv_per_token = self.kv_bytes_per_token()
        if kv_per_token is None or not (math.isfinite(kv_per_token) and kv_per_token > 0.0):
            raise ValueError(
                "insufficient model metadata; missing: per-token KV geometry "
                "(layer count + KV attention heads/head dimension, or the MLA latent dims)"
            )
        weight_bytes = self.weight_bytes()
        if weight_bytes is None:
            raise ValueError(
                "insufficient model metadata; missing: weight-estimate fields "
                "(hidden size, layer count, vocab_size, FFN intermediate size)"
            )
        kv_per_token = float(kv_per_token)
        weight_bytes = float(weight_bytes)

        cuda_graph = float(cuda_graph_reserved_bytes)
        post_weight = max(capacity_bytes - weight_bytes - cuda_graph, 0.0)
        total_kv_size_bytes_f = post_weight * float(naive_kv_reservation)
        if total_kv_size_bytes_f <= 0.0:
            non_kv_bytes = weight_bytes + cuda_graph
            raise ValueError(
                f"no KV budget: non-KV memory ({int(max(non_kv_bytes, 0.0))} bytes) meets/exceeds the "
                f"KV-cache memory limit (capacity={int(max(capacity_bytes, 0.0))} bytes)"
            )

        tokens_from_kv_bytes = partial(self.get_kvcache_max_tokens, max_batch_size=max_batch_size)
        if scheduler_backend is not None:
            tokens_from_kv_bytes = self._shared_tokens_from_bytes(
                kv_per_token, scheduler_backend, scheduler_block_size, scheduler_version
            )
        return {
            "total_gpu_capacity_bytes": int(max(capacity_bytes, 0.0)),
            "total_kv_size_bytes": math.floor(total_kv_size_bytes_f),
            "kv_size_per_token_bytes": math.floor(kv_per_token),
            "total_kv_size_tokens": tokens_from_kv_bytes(total_kv_size_bytes_f),
            # Carried so _apply_tolerance derives adjusted tokens the same way;
            # stripped by estimate_kv_cache before returning.
            "tokens_from_kv_bytes": tokens_from_kv_bytes,
            "source": "naive_fallback",
            "memory_breakdown": None,
            "tolerance_adjusted": None,
        }

    def _shared_tokens_from_bytes(
        self, token_bytes: float, backend: str, block_size: int | None, version: str | None
    ) -> TokensFromBytes:
        """Bound cold, independent requests with one resident page in every cache group."""
        geom = self.geometry
        sliding, full = geom.get("num_swa_layers"), geom.get("num_global_layers")
        pools = [(token_bytes, token_bytes)]  # (bytes/token for a physical row, bytes/token for all groups)
        if sliding and full and not geom.get("kv_lora_rank"):
            if backend == "sglang":
                raise ValueError("SGLang hybrid cache capacity requires separate full and sliding pool sizes")
            widest = max(self._attention_layer_rates())
            if backend == "vllm":
                # Adapted from vLLM's uniform-page grouping. Each type pads its last group to row_layers.
                # An unverified version must fit either group size used by the supported schedulers.
                smaller, larger = sorted((sliding, full))
                if version in {"0.31.0", "0.32.0", "0.33.0"}:
                    widths = (larger if 2 * larger < 3 * smaller else smaller,)
                elif version == "0.14.0":
                    widths = (larger if 4 * larger < 5 * smaller else smaller,)
                else:
                    widths = (smaller, larger)
                # A connector can disable hybrid allocation. The original full-width row then holds all layers.
                for width in widths:
                    groups = -(-sliding // width) + -(-full // width)
                    pools.append((widest * width, widest * width * groups))
            else:
                return _linear_tokens_from_bytes(widest * (sliding + full))
        if backend != "vllm":
            return _linear_tokens_from_bytes(token_bytes)
        if block_size is None:
            # For any integer page size B, admitting k>=1 pages costs B*(k*total + row).
            # k*B <= budget/(total + row) is sufficient even for a single, maximally large page.
            return _linear_tokens_from_bytes(max(total + row for row, total in pools))

        def schedulable_tokens(budget: float) -> int:
            blocks = min(int((budget - block_size * row) // (block_size * total)) for row, total in pools)
            return max(0, blocks) * block_size

        return schedulable_tokens

    def _hybrid_tokens_from_bytes(self, max_batch_size: int = 1) -> TokensFromBytes | None:
        """Piecewise sliding-window inverse closure, or ``None`` when not hybrid.

        For a STANDARD-attention model with a known sliding-window layout: caps the
        SWA layers at the window while the global layers keep growing, so capacity
        is not extrapolated from the (larger) ``seq_len=1`` rate. ``None`` (caller
        stays LINEAR) for non-hybrid models, MLA hybrids (latent cache; out of scope
        for this simple heuristic), or incomplete geometry.
        """
        geom = self.geometry
        if geom.get("kv_lora_rank"):  # MLA latent cache: piecewise split N/A here
            return None
        window = geom.get("sliding_window")
        num_swa = geom.get("num_swa_layers")
        num_global = geom.get("num_global_layers")
        head_dim = geom.get("head_dim")
        num_kv_heads = geom.get("num_kv_heads")
        if not window or not num_swa or num_global is None or not head_dim or not num_kv_heads:
            return None

        # With T live tokens, the sliding layers can retain up to min(T, batch * window).
        window = int(window) * max_batch_size
        swa_const, global_const = self._attention_group_rates()

        def tokens_from_bytes(budget_bytes: float) -> int:
            budget = float(budget_bytes)
            if budget <= 0.0:
                return 0
            within_window_rate = swa_const + global_const  # every layer grows up to the window
            bytes_at_window = within_window_rate * window
            if budget <= bytes_at_window:
                return int(budget // within_window_rate) if within_window_rate > 0 else 0
            if global_const <= 0:  # all layers capped -> KV can't grow past the window
                return window
            return window + int((budget - bytes_at_window) // global_const)

        return tokens_from_bytes


# --------------------------------------------------------------------------- #
# Public entry points.
# --------------------------------------------------------------------------- #


def estimate_kv_cache(
    model_path: str,
    system: str,
    backend: str,
    backend_version: str | None = None,
    *,
    max_num_tokens: int,
    max_batch_size: int,
    memory_fraction_kind: str,
    memory_fraction_value: float,
    tp_size: int = 1,
    pp_size: int = 1,
    attention_dp_size: int = 1,
    moe_tp_size: int | None = None,
    moe_ep_size: int | None = None,
    cp_size: int = 1,
    dcp_size: int | None = None,
    gemm_quant_mode: str | None = None,
    moe_quant_mode: str | None = None,
    kvcache_quant_mode: str | None = None,
    fmha_quant_mode: str | None = None,
    comm_quant_mode: str | None = None,
    moe_backend: str | None = None,
    moe_kernel_source: str | None = None,
    attention_backend: str | None = None,
    enable_eplb: bool = False,
    wideep_num_slots: int | None = None,
    nextn: int = 0,
    systems_path: str | None = None,
    gpu_memory_capacity_bytes_override: int | None = None,
    cuda_graph_reserved_bytes: int = 0,
    tolerance_fraction: float | None = None,
    naive_kv_reservation: float = _DEFAULT_NAIVE_KV_RESERVATION,
    allow_naive_fallback: bool = False,
    allow_hf_config_download: bool = False,
    fpm_profile: dict | str | FpmModelProfile | None = None,
    worker_type: str = "aggregated",
    context_length: int | None = None,
    request_occupancy_tokens: int | None = None,
    scheduler_block_size: int | None = None,
    colocated_encoder: bool = False,
    reserved_bytes: int = 0,
    encoder_parallel: str | None = None,
) -> dict[str, Any]:
    """Compute the KV-cache memory estimate (raw + optional tolerance margin).

    This function validates inputs and implements scalar native/naive budget
    math and tolerance margins. Grouped and runtime-memory FPM profiles delegate
    budget arithmetic to ``RustForwardPassPerfModel.estimate_cache_budget``.
    Grouped token-capacity and bytes-per-token fields are ``None``. The legacy Rust
    ``aisim_core::memory::estimate_kv_cache`` transport calls this function
    and rebuilds scalar results; it rejects grouped results. That transport does not forward
    the scheduler block size or DCP width. A config-only nested decoder at TP>1 therefore
    needs the Python entry point with an explicit DCP width, or native/profile memory coverage.

    Native path: :meth:`KVCacheEstimator.from_request` builds AIC's full backend
    memory model and :meth:`KVCacheEstimator.estimate` does the OfFree (TRT-LLM) /
    OfTotal (vLLM/SGLang) budget math. Naive fallback: when the native model build
    is unsupported (``from_request`` raises) AND ``allow_naive_fallback`` is set,
    :meth:`NaiveKVCacheEstimator.estimate` parses the HF ``config.json`` and
    reserves ``naive_kv_reservation`` of post-weight memory for KV; this requires
    ``gpu_memory_capacity_bytes_override``.

    Args:
        memory_fraction_kind: ``"of_total"`` (vLLM / SGLang) or ``"of_free"``
            (TRT-LLM); validated against ``backend`` BEFORE any model build.
        memory_fraction_value: the fraction in ``[0, 1]``.
        gpu_memory_capacity_bytes_override: when set, wins over the SystemSpec
            capacity on the native path; REQUIRED on the naive fallback.
        cuda_graph_reserved_bytes: fixed rank-local bytes reserved by CUDA graphs
            before KV-cache allocation. For SGLang this is an additional
            reservation beyond the graph/runtime headroom already encoded in
            ``mem_fraction_static``. Must be no greater than ``2**53`` so the
            floating-point budget math preserves the integer exactly. Defaults
            to zero for wire compatibility.
        dcp_size: decode context parallel width. A config-only nested decoder at TP greater than one
            needs the explicit width to distinguish an unsharded cache from DCP.
        scheduler_block_size: optional tokens per scheduler block. The config-only vLLM estimate
            reserves one physical row and counts whole blocks across all padded groups. Without
            a block size it returns a lower bound valid for an independently allocated block of
            any size. SGLang hybrid capacity requires the sizes of both cache pools. Recurrent
            state and the indexer in a nested sparse MLA decoder require runtime allocation information.
        tolerance_fraction: optional safety margin in ``[0, 1)``; when set,
            ``tolerance_adjusted`` is populated with ``floor(raw * (1 - t))``
            bytes and the recomputed token count. ``None`` -> raw estimate only.
        naive_kv_reservation: fraction of post-weight memory the naive fallback
            reserves for KV (default ``0.80``). Ignored on the native path.
        allow_naive_fallback: fall back to the naive heuristic when the native
            model build is unsupported (default off -> the error propagates).
        allow_hf_config_download: allow the naive fallback to download a
            ``config.json`` from HuggingFace when it is not local / pre-cached.
        fpm_profile: explicit FPM identity and rank-local memory evidence. Pending
            memory fails before estimation. Complete declared bytes require the
            requested scheduler envelope to fit the profile; runtime capacity
            requires its exact scheduler settings and memory fraction. This
            route never constructs an analytical model or uses naive fallback.
            Separate CUDA graph bytes apply only to declared profile overheads.
        worker_type: canonical serving role selecting this deployment's resources.
        cp_size: prefill context parallelism (folds into the attention width); also
            the FPM profile cell identity, whose recorded cells are CP1 today.
        context_length: optional context bound for a grouped profile's per-request
            cache peak, also checked against runtime memory's ``max_model_len``.
            Defaults to the profile context and does not change its byte budget.
        request_occupancy_tokens: optional logical request length for a separate
            native one-token decode occupancy estimate. Requires an FPM profile;
            context and scheduler admission still use their original settings.
        colocated_encoder, encoder_parallel, reserved_bytes: the resident vision
            tower, its layout over the tensor-parallel group, and a fixed rank-local
            reservation, charged by the model build path only; an FPM profile
            records the rank's memory as measured and rejects them.

    Returns:
        A flat dict with ``total_gpu_capacity_bytes``, ``total_kv_size_bytes``,
        ``kv_size_per_token_bytes``, ``total_kv_size_tokens``, ``source``
        (``"native"`` | ``"naive_fallback"`` | ``"profile"``), ``memory_breakdown`` (dict on the
        native/declared paths, ``None`` for runtime capacity or naive fallback), and ``tolerance_adjusted`` (a dict
        when ``tolerance_fraction`` is set, else ``None``).
        Native and profile estimates also report the resolved ``kvcache_quant_mode``;
        the naive fallback does not infer a quantization mode from its byte width.

    Raises:
        ValueError: incompatible / out-of-range memory fraction, out-of-range
            tolerance, no KV budget, insufficient model metadata, or (with the
            fallback off) an unsupported model/backend.
    """
    _validate_memory_fraction(backend, memory_fraction_kind, memory_fraction_value)
    _validate_tolerance(tolerance_fraction)
    _validate_naive_reservation(naive_kv_reservation)
    _validate_cuda_graph_reservation(cuda_graph_reserved_bytes)
    _validate_encoder_parallel(encoder_parallel)
    _validate_reserved_bytes(reserved_bytes)
    # Before the model build and the naive fallback, which would swallow the error.
    validate_parallel_size("cp_size", cp_size)
    explicit_dcp = dcp_size is not None
    dcp_size = validate_parallel_size("dcp_size", 1 if dcp_size is None else dcp_size)
    if scheduler_block_size is not None:
        validate_parallel_size("scheduler_block_size", scheduler_block_size)
    fraction = float(memory_fraction_value)
    is_of_free = memory_fraction_kind == "of_free"

    # Validate the compute-side MTP depth before any fallback path.
    validate_nextn(nextn)
    if request_occupancy_tokens is not None and fpm_profile is None:
        raise ValueError("request_occupancy_tokens requires an FPM profile")
    if fpm_profile is not None:
        if colocated_encoder or reserved_bytes or encoder_parallel is not None:
            # The profile records the rank's memory as measured; it cannot be reshaped by
            # a reservation or an encoder layout the recording did not carry.
            raise ValueError(
                "FPM profile resources carry the measured rank memory; colocated_encoder, "
                "reserved_bytes and encoder_parallel apply to the model build path only"
            )
        profile = _load_fpm_timing_profile(fpm_profile)
        if backend_version is None:
            raise ValueError("FPM profile memory estimation requires the profile's literal backend_version")
        deployment = profile.select(
            model=model_path,
            system=system,
            backend=backend,
            backend_version=backend_version,
            worker_type=worker_type,
            tp_size=tp_size,
            pp_size=pp_size,
            attention_dp_size=attention_dp_size,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            cp_size=cp_size,
        )
        if nextn and deployment.execution_profile is None:
            raise ValueError("FPM profile resources support plain autoregressive execution; nextn must be 0")
        deployment.validate_overrides(
            gemm_quant_mode=gemm_quant_mode,
            moe_quant_mode=moe_quant_mode,
            kvcache_quant_mode=kvcache_quant_mode,
            fmha_quant_mode=fmha_quant_mode,
            comm_quant_mode=comm_quant_mode,
            attention_backend=attention_backend,
            moe_backend=moe_backend,
            enable_eplb=enable_eplb,
            wideep_num_slots=wideep_num_slots,
        )
        resources = deployment.require_resources()
        resources.require_memory()
        resources.validate_envelope(max_num_tokens=max_num_tokens, max_batch_size=max_batch_size)
        if (
            resources.runtime_memory is None
            and resources.non_kv_bytes + cuda_graph_reserved_bytes > _MAX_EXACT_BYTE_COUNT
        ):
            raise ValueError("total FPM non-KV bytes including CUDA graphs must not exceed 2**53")
        capacity = gpu_memory_capacity_bytes_override
        if capacity is None:
            capacity = perf_database.load_system_spec(system, systems_paths=systems_path)["gpu"]["mem_capacity"]
        if (
            resources.cache_layout == "grouped"
            or resources.runtime_memory is not None
            or request_occupancy_tokens is not None
            or deployment.execution_profile is not None
        ):
            from aisim_core.sdk.rust_engine_step import RustForwardPassPerfModel

            estimate = RustForwardPassPerfModel.estimate_cache_budget(
                {
                    "model": model_path,
                    "system": system,
                    "backend": backend,
                    "backend_version": backend_version,
                    "worker_type": worker_type,
                    "tp": tp_size,
                    "pp": pp_size,
                    "attention_dp": attention_dp_size,
                    "moe_tp_size": deployment.moe_tp,
                    "moe_ep_size": deployment.moe_ep,
                    "cp_size": cp_size,
                    # This API's default DCP1 means no sharding, not a recorded
                    # DCP selector. Preserve nontrivial DCP for native validation.
                    **({"dcp": dcp_size} if dcp_size != 1 else {}),
                    "fpm_profile": profile.model_dump(mode="json"),
                    **(
                        {
                            "nextn": nextn,
                            "estimation_mode": "fpm_interpolation",
                            "fallback_policy": "deny",
                            "estimator_config": {"fpm_interpolation": {"method": "direct"}},
                        }
                        if deployment.execution_profile is not None
                        else {}
                    ),
                },
                {
                    "total_gpu_capacity_bytes": capacity,
                    "memory_fraction_kind": memory_fraction_kind,
                    "memory_fraction_value": memory_fraction_value,
                    "max_num_tokens": max_num_tokens,
                    "max_batch_size": max_batch_size,
                    "context_length": context_length,
                    "request_occupancy_tokens": request_occupancy_tokens,
                    "cuda_graph_reserved_bytes": cuda_graph_reserved_bytes,
                    "tolerance_fraction": tolerance_fraction,
                },
            )
            estimate["kvcache_quant_mode"] = deployment.kv_cache_dtype
            return estimate
        estimate = KVCacheEstimator._estimate_from_breakdown(
            {
                "weights_bytes": resources.weights_bytes,
                "activations_bytes": resources.activations_bytes,
                "runtime_overhead_bytes": resources.runtime_overhead_bytes,
                "comm_overhead_bytes": resources.comm_overhead_bytes,
                "non_kv_bytes": resources.non_kv_bytes,
                "kv_size_per_token_bytes": resources.kv_bytes_per_token,
                "gpu_memory_capacity_bytes": capacity,
            },
            is_of_free=is_of_free,
            fraction=fraction,
            gpu_memory_capacity_bytes_override=gpu_memory_capacity_bytes_override,
            cuda_graph_reserved_bytes=cuda_graph_reserved_bytes,
        )
        estimate["source"] = "profile"
        estimate["kvcache_quant_mode"] = deployment.kv_cache_dtype
        estimate["resource_provenance"] = resources.provenance
        _apply_tolerance(estimate, tolerance_fraction)
        estimate.pop("tokens_from_kv_bytes", None)
        return estimate

    for name, value in (("tp_size", tp_size), ("pp_size", pp_size), ("attention_dp_size", attention_dp_size)):
        validate_parallel_size(name, value)
    for name, value in (("moe_tp_size", moe_tp_size), ("moe_ep_size", moe_ep_size)):
        if value is not None:
            validate_parallel_size(name, value)
    validate_parallel_size("max_batch_size", max_batch_size)
    validate_parallel_size("max_num_tokens", max_num_tokens)
    for field, mode, options in (
        ("gemm_quant_mode", gemm_quant_mode, GEMMQuantMode),
        ("moe_quant_mode", moe_quant_mode, MoEQuantMode),
        ("kvcache_quant_mode", kvcache_quant_mode, KVCacheQuantMode),
        ("fmha_quant_mode", fmha_quant_mode, FMHAQuantMode),
        ("comm_quant_mode", comm_quant_mode, CommQuantMode),
    ):
        if mode is not None and (not isinstance(mode, str) or mode not in options.__members__):
            raise ValueError(f"unsupported {field} {mode!r}")
    validate_moe_controls(
        model_path=model_path,
        enable_eplb=enable_eplb,
        wideep_num_slots=wideep_num_slots,
        moe_backend=moe_backend,
        moe_kernel_source=moe_kernel_source,
    )
    try:
        native = KVCacheEstimator.from_request(
            model_path,
            system,
            backend,
            backend_version,
            max_num_tokens=int(max_num_tokens),
            max_batch_size=int(max_batch_size),
            tp_size=int(tp_size),
            pp_size=int(pp_size),
            attention_dp_size=int(attention_dp_size),
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            cp_size=int(cp_size),
            dcp_size=int(dcp_size),
            gemm_quant_mode=gemm_quant_mode,
            moe_quant_mode=moe_quant_mode,
            kvcache_quant_mode=kvcache_quant_mode,
            fmha_quant_mode=fmha_quant_mode,
            comm_quant_mode=comm_quant_mode,
            moe_backend=moe_backend,
            moe_kernel_source=moe_kernel_source,
            attention_backend=attention_backend,
            enable_eplb=enable_eplb,
            wideep_num_slots=wideep_num_slots,
            nextn=int(nextn),
            systems_path=systems_path,
            colocated_encoder=colocated_encoder,
            reserved_bytes=reserved_bytes,
            encoder_parallel=encoder_parallel,
        )
    except (perf_database.UnlistedQueryVersionError, perf_database.StrictProvenanceError):
        # A selected catalog explicitly rejected the request. Fallback cannot qualify a different root.
        raise
    except Exception as exc:  # native model build unsupported (model/backend/perf DB)
        if isinstance(exc, NotImplementedError) and (int(cp_size) > 1 or int(dcp_size) > 1):
            # A context-parallel capability rejection must not degrade into the
            # naive estimator, which knows neither knob and would return an
            # unstriped (cp=dcp=1) capacity for a request that asked otherwise.
            raise ValueError(
                f"context parallelism is not supported for KV-cache estimation of this model/backend "
                f"(model={model_path}, backend={backend}, cp_size={cp_size}, dcp_size={dcp_size}): {exc}"
            ) from exc
        if (
            not allow_naive_fallback
            or isinstance(exc, UnverifiedGemmaRuntimeError)
            or enable_eplb
            or wideep_num_slots is not None
            or moe_backend not in (None, "default")
            or moe_kernel_source is not None
            or attention_backend is not None
            or colocated_encoder
            or reserved_bytes
            or encoder_parallel is not None
        ):
            raise ValueError(
                f"unsupported model/backend/GPU for KV-cache estimation: "
                f"model={model_path}, backend={backend}, gpu_sku={system}: {exc}"
            ) from exc
        version = _config_only_backend_version(system, backend, backend_version, systems_path)
        naive = NaiveKVCacheEstimator.from_model_path(
            model_path,
            tp_size=int(tp_size),
            pp_size=int(pp_size),
            allow_hf_config_download=allow_hf_config_download,
            backend=backend,
            backend_version=version,
            system=system,
            systems_paths=systems_path,
            moe_ep_size=int(moe_ep_size or 1),
            moe_tp_size=int(moe_tp_size or 1),
        )
        naive._require_scalar_capacity()
        if naive._selected_decoder is not None and nextn:
            raise ValueError(
                "speculative decoding capacity requires the runtime draft weights and cache allocation"
            ) from exc
        if naive._is_nested_decoder and tp_size > 1 and not explicit_dcp:
            raise ValueError(
                "config-only sizing of a nested decoder at TP>1 requires explicit dcp_size. "
                "Set dcp_size=1 for an unsharded cache"
            ) from exc
        family = ARCHITECTURE_TO_MODEL_FAMILY.get(naive.architecture)
        model_class = _MODEL_REGISTRY.get(family)
        # SGLang prefill CP replicates the ordinary K/V or dense MLA cache on each rank.
        # Sparse MLA has an additional indexer allocation outside this config-only estimate.
        replicated_cp = (
            backend == "sglang"
            and tp_size == attention_dp_size == 1
            and model_class is not None
            and model_class.supports_cp(backend)
            and (
                not naive.geometry.get("kv_lora_rank")
                or model_class.get_kvcache_bytes_per_sequence is BaseModel.get_kvcache_bytes_per_sequence
            )
        )
        if dcp_size != 1 or (cp_size != 1 and not replicated_cp):
            raise ValueError(
                "config-only cache sizing cannot represent context parallelism for this model/backend"
            ) from exc
        if model_class is not None or naive._selected_decoder is not None:
            naive._validate_attention_tp()
        experts = naive.geometry.get("num_experts") if naive.geometry.get("num_routed_layers", 1) else 0
        if experts and (moe_tp_size is not None or moe_ep_size is not None):
            ranks = build_model_config(
                tp_size=tp_size,
                pp_size=pp_size,
                cp_size=cp_size,
                attention_dp_size=attention_dp_size,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
            )
            naive.moe_tp_size, naive.moe_ep_size = ranks.resolve_moe_parallelism()
            # vLLM distributes the remainder one expert at a time over the first EP ranks.
            if experts < naive.moe_ep_size or (backend != "vllm" and experts % naive.moe_ep_size):
                raise ValueError(
                    f"{experts} experts cannot partition uniformly over moe_ep_size {naive.moe_ep_size}"
                ) from exc
        raw = naive._selected_decoder
        expert_tp = naive.moe_tp_size
        if backend in ("sglang", "vllm") and moe_tp_size is None and moe_ep_size is None:
            # Both backends default to EP1. Validate the actual shard even with the coarse unsharded reservation.
            expert_tp = tp_size * attention_dp_size * cp_size
        if raw is not None and experts:
            intermediate = naive.geometry["moe_inter"]
            if not intermediate or intermediate % expert_tp:
                raise ValueError("expert intermediate width must be positive and divisible by moe_tp_size") from exc
        quant = (raw or {}).get("quantization_config") or {}
        block = quant.get("weight_block_size")
        mxfp8 = backend == "sglang" and quant.get("quant_method") == "mxfp8"
        sglang_fp8 = mxfp8 or (backend == "sglang" and quant.get("quant_method") == "fp8" and block is not None)
        if sglang_fp8 or (raw is not None and experts and moe_quant_mode in (None, "fp8_block")):
            release = parse_support_matrix_version(version)
            # Starting in 0.5.16 MXFP8 sets its fixed block shape before checking activations.
            if (
                sglang_fp8
                and quant.get("activation_scheme") != "dynamic"
                and not (
                    mxfp8
                    and block is None
                    and quant.get("activation_scheme") == "static"
                    and release is not None
                    and release < Version("0.5.16")
                )
            ):
                raise ValueError("SGLang block-quantized FP8 weights require dynamic activations") from exc
            sdk_tiles = hasattr(model_class, "_validate_fp8_block_quantized_moe_config") or (
                model_class is None and "weight_block_size" in quant
            )
            check_tiles = sglang_fp8 or (
                sdk_tiles
                and (
                    moe_quant_mode == "fp8_block"
                    or _infer_quant_modes_from_raw_config(raw, naive.architecture).get("moe_quant_mode")
                    is MoEQuantMode.fp8_block
                )
            )
            quantized_experts = bool(experts)
            deepseek = naive.architecture in (
                "DeepseekV2ForCausalLM",
                "DeepseekV3ForCausalLM",
                "DeepseekV32ForCausalLM",
            )
            module = None
            if naive.architecture == "MixtralForCausalLM":
                # Mixtral constructs its layer list with the literal "layers" prefix.
                module = "layers.{}.block_sparse_moe.experts"
            elif (
                naive.architecture == "Qwen3MoeForCausalLM" or raw.get("model_type") == "qwen3_vl_moe_text" or deepseek
            ):
                module = naive._sglang_layer_prefix + "mlp.experts"
            if sglang_fp8:
                ignored = quant.get("ignored_layers", quant.get("modules_to_not_convert", []))
                if ignored is None:
                    ignored = []
                if not isinstance(ignored, list) or not all(isinstance(name, str) and name for name in ignored):
                    raise ValueError("SGLang FP8 exclusions must be a list of module names") from exc
                layers = range(naive.geometry["layers"])
                routed = set(layers) if experts else set()
                if deepseek and experts:
                    routed = set(_deepseek_routed_layers(raw, len(layers)))
                dense_layers = [i for i in layers if i not in routed]
                if dense_layers:
                    dense_tp = tp_size * attention_dp_size * cp_size
                    if (
                        cp_size > 1
                        and naive.architecture == "DeepseekV3ForCausalLM"
                        and release is not None
                        and release >= Version("0.5.13")
                    ):
                        # The SGLang DeepSeek MLA prefill-CP override replicates its dense MLPs.
                        dense_tp = 1
                    # Qwen3 shards attention across the attention TP group. Llama and Qwen2 use the full TP group.
                    attn_tp = (
                        dense_tp
                        if naive.architecture in ("LlamaForCausalLM", "Qwen2ForCausalLM", "MistralForCausalLM")
                        else tp_size
                    )
                    _validate_sglang_dense_fp8(
                        quant,
                        naive.geometry,
                        attn_tp,
                        dense_tp,
                        release,
                        ignored,
                        naive._sglang_layer_prefix,
                        dense_layers,
                    )
                if module is not None and release is not None:
                    quantized_experts = not _sglang_skips_modules(ignored, routed, release, module)
            if mxfp8:
                if release is None or release < Version("0.5.9"):
                    raise ValueError("SGLang MXFP8 checkpoints require backend version 0.5.9 or later") from exc
                sm = _config_only_system_spec(system, systems_path).get("gpu", {}).get("sm_version")
                if not (sm is not None and sm >= 100):
                    raise ValueError(
                        "SGLang checkpoints declaring MXFP8 require a Blackwell GPU (SM100) or later"
                    ) from exc
                if quantized_experts and sm >= 110:
                    raise ValueError(
                        "SGLang MXFP8 expert weights require a Blackwell CUDA GPU in the SM100 family"
                    ) from exc
            if experts and check_tiles:
                validate_fp8_moe_shard(
                    raw,
                    intermediate,
                    expert_tp,
                    sglang_tp_size=max(tp_size * attention_dp_size, expert_tp) if backend == "sglang" else None,
                    quantized_experts=quantized_experts,
                    vllm_refine_hidden_size=(
                        naive.geometry["hidden"]
                        if backend == "vllm" and release is not None and release >= Version("0.29.0rc1")
                        else None
                    ),
                )
        # With no deployment specified, preserve the conservative unsharded expert weight estimate.
        if kvcache_quant_mode is not None:
            naive.set_kvcache_quant_mode(kvcache_quant_mode)
        estimate = naive.estimate(
            capacity=gpu_memory_capacity_bytes_override,
            naive_kv_reservation=float(naive_kv_reservation),
            cuda_graph_reserved_bytes=cuda_graph_reserved_bytes,
            max_batch_size=max_batch_size,
            scheduler_backend=backend,
            scheduler_block_size=scheduler_block_size,
            scheduler_version=version,
        )
    else:
        # Native model built: budget-math failures (no KV budget, zero per-token)
        # propagate; they do NOT trigger the naive fallback.
        estimate = native.estimate(
            is_of_free=is_of_free,
            fraction=fraction,
            gpu_memory_capacity_bytes_override=gpu_memory_capacity_bytes_override,
            cuda_graph_reserved_bytes=cuda_graph_reserved_bytes,
        )

    # Apply the tolerance margin (no-op when `tolerance_fraction` is None),
    # identically for the native and naive estimates. _apply_tolerance reads the
    # carried `tokens_from_kv_bytes` inverse; strip it afterwards so the result is
    # the flat, serializable dict the Rust forwarder parses.
    _apply_tolerance(estimate, tolerance_fraction)
    estimate.pop("tokens_from_kv_bytes", None)
    return estimate


def estimate_num_gpu_blocks(
    model_path: str,
    system: str,
    backend: str,
    backend_version: str | None = None,
    *,
    scheduler_block_size: int,
    max_num_tokens: int,
    max_batch_size: int,
    memory_fraction_kind: str,
    memory_fraction_value: float,
    tp_size: int = 1,
    pp_size: int = 1,
    attention_dp_size: int = 1,
    moe_tp_size: int | None = None,
    moe_ep_size: int | None = None,
    cp_size: int = 1,
    dcp_size: int = 1,
    gemm_quant_mode: str | None = None,
    moe_quant_mode: str | None = None,
    kvcache_quant_mode: str | None = None,
    fmha_quant_mode: str | None = None,
    comm_quant_mode: str | None = None,
    moe_backend: str | None = None,
    moe_kernel_source: str | None = None,
    attention_backend: str | None = None,
    enable_eplb: bool = False,
    wideep_num_slots: int | None = None,
    nextn: int = 0,
    systems_path: str | None = None,
    gpu_memory_capacity_bytes_override: int | None = None,
    cuda_graph_reserved_bytes: int = 0,
    tolerance_fraction: float | None = None,
    naive_kv_reservation: float = _DEFAULT_NAIVE_KV_RESERVATION,
    allow_naive_fallback: bool = False,
    allow_hf_config_download: bool = False,
    diagnostics: dict[str, Any] | None = None,
    fpm_profile: dict | str | FpmModelProfile | None = None,
    worker_type: str = "aggregated",
    context_length: int | None = None,
    colocated_encoder: bool = False,
    reserved_bytes: int = 0,
    encoder_parallel: str | None = None,
) -> int:
    """Convert the KV-cache token capacity to a scheduler block count.

    Thin convenience over :func:`estimate_kv_cache`: computes the per-rank
    KV-cache token capacity and returns ``floor(total_kv_size_tokens /
    scheduler_block_size)``, using the tolerance-adjusted token count when
    ``tolerance_fraction`` is set (the raw count otherwise).

    Args:
        scheduler_block_size: KV block size the scheduler allocates in (tokens).
        memory_fraction_kind: ``"of_total"`` (vLLM / SGLang) or ``"of_free"``
            (TRT-LLM); validated against ``backend``.
        memory_fraction_value: the fraction in ``[0, 1]``.
        diagnostics: optional output mapping populated with the exact memory
            estimate used for this block count, including its source and units.
        (remaining kwargs mirror :func:`estimate_kv_cache`.)

    Returns:
        ``floor(total_kv_size_tokens / scheduler_block_size)``, using the
        tolerance-adjusted tokens when ``tolerance_fraction`` is set.

    Raises:
        ValueError: propagated from :func:`estimate_kv_cache` (unsupported
            model/backend, incompatible memory fraction, out-of-range tolerance,
            no KV budget, etc.).
    """
    block_size = validate_parallel_size("scheduler_block_size", scheduler_block_size)

    estimate = estimate_kv_cache(
        model_path,
        system,
        backend,
        backend_version=backend_version,
        scheduler_block_size=block_size,
        max_num_tokens=max_num_tokens,
        max_batch_size=max_batch_size,
        memory_fraction_kind=memory_fraction_kind,
        memory_fraction_value=float(memory_fraction_value),
        tp_size=tp_size,
        pp_size=pp_size,
        attention_dp_size=attention_dp_size,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        # Raw values: estimate_kv_cache validates them (positive, integral, not
        # bool) before any coercion, so True / 1.9 cannot become cp=dcp=1 here.
        cp_size=cp_size,
        dcp_size=dcp_size,
        gemm_quant_mode=gemm_quant_mode,
        moe_quant_mode=moe_quant_mode,
        kvcache_quant_mode=kvcache_quant_mode,
        fmha_quant_mode=fmha_quant_mode,
        comm_quant_mode=comm_quant_mode,
        moe_backend=moe_backend,
        moe_kernel_source=moe_kernel_source,
        attention_backend=attention_backend,
        enable_eplb=enable_eplb,
        wideep_num_slots=wideep_num_slots,
        nextn=nextn,
        systems_path=systems_path,
        gpu_memory_capacity_bytes_override=gpu_memory_capacity_bytes_override,
        cuda_graph_reserved_bytes=cuda_graph_reserved_bytes,
        tolerance_fraction=tolerance_fraction,
        naive_kv_reservation=float(naive_kv_reservation),
        allow_naive_fallback=allow_naive_fallback,
        allow_hf_config_download=allow_hf_config_download,
        fpm_profile=fpm_profile,
        worker_type=worker_type,
        context_length=context_length,
        colocated_encoder=colocated_encoder,
        reserved_bytes=reserved_bytes,
        encoder_parallel=encoder_parallel,
    )

    if estimate.get("cache_layout") == "mamba_align":
        raise ValueError("Mamba FPM cache requires replay_cache from estimate_kv_cache, including state reservations")
    if estimate.get("cache_layout") == "grouped":
        raise ValueError(
            "grouped FPM caches require cache_groups and the byte budget from estimate_kv_cache; "
            "a scalar num_gpu_blocks or token capacity cannot represent windowed cache groups"
        )
    adjusted = estimate.get("tolerance_adjusted")
    if adjusted is not None:
        tokens = int(adjusted["total_kv_size_tokens"])
    else:
        tokens = int(estimate["total_kv_size_tokens"])

    if diagnostics is not None:
        diagnostics.update(estimate)
        diagnostics["scheduler_block_size_tokens"] = block_size
        diagnostics["num_gpu_blocks"] = tokens // block_size
    return tokens // block_size
