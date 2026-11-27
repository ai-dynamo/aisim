# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config markers of a cache layout that config.json cannot describe.

Config intake and the FPM collector detect these layouts with this one check. Intake leaves their cache
geometry to the runtime probe; the collector refuses an FPM attention template for them. The module also holds
the structure checks of a layout with markers, which intake and the collector run.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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
    from aisim_core import decoder_config

    return decoder_config.attn_type_list(raw, layers)


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
    from aisim_core import decoder_config

    return decoder_config.recognized_layer_types(architecture)


def cache_layout_markers(raw: Mapping[str, Any], architecture: str | None) -> tuple[str, ...]:
    from aisim_core import decoder_config

    return decoder_config.cache_layout_markers(raw, architecture)
