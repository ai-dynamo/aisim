# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config markers of a cache layout that config.json cannot describe.

Config intake and the FPM collector detect these layouts with this one check. Intake leaves their cache
geometry to the runtime probe; the collector refuses an FPM attention template for them. The module also holds
the structure checks of a layout with markers, which intake and the collector run.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

RECURRENT_STATE_FIELDS = (
    "ssm_cfg",
    "mamba_d_state",
    "linear_conv_kernel_dim",
    "linear_attn_config",
    "ssm_state_size",
    "mamba_num_heads",
)
RECURRENT_ARCHITECTURES = ("mamba", "jamba", "rwkv")
LAYER_COUNT_FIELDS = ("num_hidden_layers", "n_layer", "num_layers")


# The supervisor imports this module while building its parser. Load the decoder only on config intake.
def layout_integer(value: Any, field: str, *, minimum: int = 1, maximum: int | None = None) -> int:
    from aisim_core import decoder_config

    return decoder_config.layout_integer(value, field, minimum=minimum, maximum=maximum)


def cache_window(raw: Mapping[str, Any]) -> int | None:
    from aisim_core import decoder_config

    return decoder_config.cache_window(raw)


def layer_count(raw: Mapping[str, Any]) -> int | None:
    supplied = {field: layout_integer(raw[field], field) for field in LAYER_COUNT_FIELDS if field in raw}
    if len(set(supplied.values())) > 1:
        raise ValueError(f"conflicting num_hidden_layers fields: {supplied}")
    return next(iter(supplied.values()), None)


def layer_types(raw: Mapping[str, Any], layers: int | None) -> list[str]:
    from aisim_core import decoder_config

    return decoder_config.layer_types(raw, layers)


def attn_type_list(raw: Mapping[str, Any], layers: int | None) -> list[int]:
    values = raw["attn_type_list"]
    if not isinstance(values, list) or (layers is not None and len(values) != layers):
        raise ValueError("attn_type_list must have one entry per num_hidden_layers")
    if any(type(value) is not int for value in values):
        raise ValueError("attn_type_list entries must be integers")
    return values


def check_marker_layout_declarations(raw: Mapping[str, Any]) -> int | None:
    """Check the declarations of a layout with markers, in this order; return the layer count, None when absent.

    The window fields, the layer count from every field in LAYER_COUNT_FIELDS, layer_types and attn_type_list.
    A marker never excuses a malformed declaration, and the runtime probe measures the cache geometry itself.
    """
    cache_window(raw)
    layers = layer_count(raw)
    if "layer_types" in raw:
        layer_types(raw, layers)
    if "attn_type_list" in raw:
        attn_type_list(raw, layers)
    return layers


def recognized_layer_types(architecture: str | None) -> frozenset[str]:
    names = {"full_attention", "sliding_attention"}
    if architecture in {"GlmMoeDsaForCausalLM", "DeepseekV32ForCausalLM"}:
        names.add("deepseek_sparse_attention")
    return frozenset(names)


def cache_layout_markers(raw: Mapping[str, Any], architecture: str | None) -> tuple[str, ...]:
    """Sorted reasons the config declares a cache layout it cannot describe; empty when it declares none.

    Layer types, attention-type lists and recurrent-state markers count independently, so recognized values
    never hide declared state. ``architecture`` is the decoder architecture the layer types are checked against.
    Malformed lists are not markers: each caller raises its own structure errors for them.
    """
    reasons = []
    layer_types = raw.get("layer_types")
    if isinstance(layer_types, list):
        recognized = recognized_layer_types(architecture)
        unrecognized = Counter(
            value for value in layer_types if isinstance(value, str) and value.strip() and value not in recognized
        )
        reasons.extend(f"layer_types {name} x{count}" for name, count in unrecognized.items())
    attn_types = raw.get("attn_type_list")
    if isinstance(attn_types, list):
        # 1 is full attention; any other integer is a layer type the config does not describe.
        other = Counter(value for value in attn_types if type(value) is int and value != 1)
        reasons.extend(f"attn_type_list {value} x{count}" for value, count in other.items())
    reasons.extend(f"recurrent-state field {key}" for key in RECURRENT_STATE_FIELDS if raw.get(key) is not None)
    if any(name in (architecture or "").lower() for name in RECURRENT_ARCHITECTURES):
        reasons.append(f"recurrent architecture {architecture}")
    return tuple(sorted(reasons))
