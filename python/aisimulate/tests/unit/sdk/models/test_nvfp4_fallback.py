# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aisimulate_core.sdk import common
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.models.helpers import resolve_nvfp4_for_system

pytestmark = pytest.mark.unit


def _model_config(**overrides) -> ModelConfig:
    defaults = dict(tp_size=8, moe_tp_size=1, moe_ep_size=8)
    defaults.update(overrides)
    return ModelConfig(**defaults)


def test_nvfp4_remapped_to_nvfp4_wo_on_hopper():
    """nvfp4 → nvfp4_wo unconditionally on non-Blackwell (hardware fact)."""
    mc = _model_config(
        gemm_quant_mode=common.GEMMQuantMode.nvfp4,
        moe_quant_mode=common.MoEQuantMode.nvfp4,
    )
    resolve_nvfp4_for_system(mc, "h100_sxm")
    assert mc.gemm_quant_mode == common.GEMMQuantMode.nvfp4_wo
    assert mc.moe_quant_mode == common.MoEQuantMode.nvfp4_wo


def test_nvfp4_unchanged_on_blackwell():
    """Blackwell has native FP4 TC — nvfp4 stays as-is."""
    mc = _model_config(
        gemm_quant_mode=common.GEMMQuantMode.nvfp4,
        moe_quant_mode=common.MoEQuantMode.nvfp4,
    )
    resolve_nvfp4_for_system(mc, "b200_sxm")
    assert mc.gemm_quant_mode == common.GEMMQuantMode.nvfp4
    assert mc.moe_quant_mode == common.MoEQuantMode.nvfp4


@pytest.mark.parametrize("pinned_sm,default_sm", [(100, 90), (90, 100), (100, None)])
@pytest.mark.parametrize("root_form", ["path", "string", "path_list", "string_list"])
def test_nvfp4_classification_uses_explicit_systems_roots(tmp_path, monkeypatch, pinned_sm, default_sm, root_form):
    from aisimulate_core.sdk import perf_database

    pinned = tmp_path / "pinned"
    default = tmp_path / "default"
    pinned.mkdir()
    default.mkdir()
    (pinned / "custom_sku.yaml").write_text(f"gpu:\n  sm_version: {pinned_sm}\n")
    if default_sm is not None:
        (default / "custom_sku.yaml").write_text(f"gpu:\n  sm_version: {default_sm}\n")
    monkeypatch.setattr(perf_database, "_SYSTEMS_PATHS", [str(default)])
    roots = {"path": pinned, "string": str(pinned), "path_list": [pinned], "string_list": [str(pinned)]}[root_form]
    assert perf_database.is_blackwell_system("custom_sku", systems_paths=roots) is (pinned_sm >= 100)
    assert perf_database.is_blackwell_system("custom_sku") is (default_sm is not None and default_sm >= 100)

    mc = _model_config(gemm_quant_mode=common.GEMMQuantMode.nvfp4, moe_quant_mode=common.MoEQuantMode.nvfp4)
    resolve_nvfp4_for_system(mc, "custom_sku", systems_paths=roots)
    suffix = "nvfp4" if pinned_sm >= 100 else "nvfp4_wo"
    assert mc.gemm_quant_mode == common.GEMMQuantMode[suffix]
    assert mc.moe_quant_mode == common.MoEQuantMode[suffix]


def test_bfloat16_unchanged_on_hopper():
    mc = _model_config(
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
    )
    resolve_nvfp4_for_system(mc, "h100_sxm")
    assert mc.gemm_quant_mode == common.GEMMQuantMode.bfloat16
    assert mc.moe_quant_mode == common.MoEQuantMode.bfloat16


def test_nvfp4_wo_memory_matches_nvfp4():
    assert common.GEMMQuantMode.nvfp4_wo.value.memory == common.GEMMQuantMode.nvfp4.value.memory
    assert common.MoEQuantMode.nvfp4_wo.value.memory == common.MoEQuantMode.nvfp4.value.memory


def test_nvfp4_wo_compute_matches_bfloat16():
    assert common.GEMMQuantMode.nvfp4_wo.value.compute == common.GEMMQuantMode.bfloat16.value.compute
    assert common.MoEQuantMode.nvfp4_wo.value.compute == common.MoEQuantMode.bfloat16.value.compute
