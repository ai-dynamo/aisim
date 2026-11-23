# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified processor/topology test derivatives (Apache-2.0), copyright 2026
# the HuggingFace Inc. team and HuggingFace Team, and copyright contributors
# to the vLLM project. Processor override dictionaries are synthetic fixtures.
# https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/kimi_k25/image_processing_kimi_k25.py
# https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/kimi_k25/video_processing_kimi_k25.py
# https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/vllm/model_executor/models/kimi_k25_vit.py

"""Kimi K3 image/video encoder parsing, construction, and runtime tests."""

import dataclasses
import inspect

import pytest

from aisim.sdk import common, config
from aisim.sdk.models import get_model

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("config_type", [common.VisionEncoderConfig, common.Gemma4VisionEncoderConfig])
def test_kimi_config_additions_preserve_existing_positional_constructors(config_type):
    positional_fields = (
        "depth",
        "hidden_size",
        "num_heads",
        "intermediate_size",
        "patch_size",
        "temporal_patch_size",
        "spatial_merge_size",
        "out_hidden_size",
        "deepstack_visual_indexes",
        "projector_dims",
        "projector_n_instances",
        "partial_rotary_factor",
        "in_channels",
        "image_size",
        "has_cls_token",
        "max_num_tiles",
        "resize_to_max_canvas",
        "add_global_tile",
        "prompt_image_tokens",
        "prompt_tokens_per_local_tile",
        "final_norm",
        "pool_temporal",
        "video_attention_type",
        "resize_mode",
        "image_max_patches",
        "video_max_patches",
        "max_patches_per_side",
        "max_video_frames",
        "projector_replicated",
    )
    if config_type is common.Gemma4VisionEncoderConfig:
        positional_fields += (
            "num_key_value_heads",
            "head_dim",
            "pooling_kernel_size",
            "position_embedding_size",
            "soft_tokens_per_image",
            "supported_soft_token_budgets",
            "standardize",
        )
    signature = inspect.signature(config_type)
    assert (
        tuple(
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        == positional_fields
    )
    existing = config_type(27, 1024, 16, 4096, 14, 1, 2, 7168)
    reconstructed = config_type(*(getattr(existing, name) for name in positional_fields))
    assert reconstructed == existing
    updated = dataclasses.replace(reconstructed, qkv_hidden_size=1536, projector_pre_norm=False)
    assert updated.qkv_hidden_size == 1536
    assert updated.projector_pre_norm is False
    assert all(getattr(updated, name) == getattr(existing, name) for name in positional_fields)


def _model_config(tp_size: int = 1, *, enable_encoder_dp: bool = True, nextn: int = 0) -> config.ModelConfig:
    return config.ModelConfig(
        tp_size=tp_size,
        attention_dp_size=1,
        moe_tp_size=tp_size,
        moe_ep_size=1,
        enable_encoder_dp=enable_encoder_dp,
        nextn=nextn,
    )


@pytest.fixture
def kimi_k3_model():
    return get_model("moonshotai/Kimi-K3", _model_config(), "sglang")
