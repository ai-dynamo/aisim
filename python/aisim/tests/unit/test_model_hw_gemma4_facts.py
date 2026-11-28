# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the (model, hardware) facts the parallel search and AFD consumers read.

``check_is_moe`` resolves the ``GEMMA4MIX`` family from the parsed expert
configuration, so a dense Gemma 4 checkpoint stops looking like a routed-MoE
model. These assertions call the real ``resolve_model_hardware`` path, covering
the derived ``mla``/``enable_wideep`` facts and the expert count as well as the
classification itself. The bundled ``gb300`` system spec keeps the test
hermetic.
"""

import pytest

from aisim.sweeper.model_hw import resolve_model_hardware

pytestmark = pytest.mark.unit

DENSE_GEMMA4 = "nvidia/Gemma-4-31B-IT-NVFP4"
ROUTED_GEMMA4 = "google/gemma-4-26B-A4B"


@pytest.mark.parametrize(
    ("model", "is_moe", "mla", "enable_wideep", "num_experts"),
    [
        # Dense 31B: no routed experts, so no MoE-only facts either.
        (DENSE_GEMMA4, False, False, False, 0),
        # Routed 26B-A4B: 128 experts, MLA-marked and small enough for one node.
        (ROUTED_GEMMA4, True, True, False, 128),
    ],
)
def test_gemma4_hardware_facts_follow_the_expert_configuration(model, is_moe, mla, enable_wideep, num_experts):
    facts = resolve_model_hardware(model, "gb300", backend="vllm")

    assert facts.is_moe is is_moe
    assert facts.mla is mla
    assert facts.enable_wideep is enable_wideep
    assert facts.num_experts == num_experts
