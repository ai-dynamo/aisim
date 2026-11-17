# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fp8_block MoE expert-TP limit is a cited kernel limit, not a defect.

C3.4's vLLM full grid (2026-11-02) recorded nine MoE cases as ``unexpected``
ValueErrors - every one of them vLLM's own
``The output_size of gate's and up's weight = N is not divisible by weight
quantization block_n = 128`` from validate_fp8_block_shape_moe
(quantization/utils/fp8_utils.py:1263-1289 @0.31.0). The limit is not a plain
modulo: vLLM first refines the 128-wide checkpoint block down to
``gcd(128, local_inter, hidden)`` when that is >= 32
(refine_fp8_moe_block_shape, fused_moe/oracle/fp8.py:296-315), which is why
the same grid measured fp8_block at moe_tp 1, 2 and 4 (640/320/160 -> 128/64/
32-wide blocks) and failed only at 8, 16 and 32 (80/40/20 -> 16/8/4).

The shard is expert-TP only. ``FusedMoEConfig.__post_init__`` sets
``intermediate_size_per_partition = intermediate_size // tp_size``
(fused_moe/config.py:1323) and ``FusedMoEParallelConfig.make`` collapses TP to
1 whenever expert parallelism is enabled - "each device owns a set of experts
fully. There is no tensor parallel" (fused_moe/config.py:1182-1226). The
serving topology is moe_tp 1 / ep 8, and an earlier revision of this guard
keyed on ``ep`` instead and refused 27 cases, dropping that topology; these
tests pin the EP cases as measurable so it cannot regress again.

A deterministic, cited framework limit must be counted apart from collector
defects, so the guard refuses the case before FusedMoE allocates weights and
its message starts with ``FIXME(kernel-limit)``
(collector/failure_handling.md). The module imports torch and vllm at module
scope, so the pure helper is AST-extracted.
"""

import ast
import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = REPO_ROOT / "collector" / "vllm" / "collect_moe.py"
HIDDEN = 2560
INTER = 640


def _load_local_inter_error():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_fp8_block_moe_local_inter_error"
    )
    # Only the function node is compiled, so supply the module-level imports
    # the helper closes over.
    namespace: dict = {"math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace["_fp8_block_moe_local_inter_error"]


@pytest.mark.parametrize("moe_type", ["bfloat16", "fp8"])
def test_non_block_dtypes_are_never_refused(moe_type):
    local_inter_error = _load_local_inter_error()

    # bf16 and per-tensor fp8 have no 128-wide weight block, so the same
    # per-rank 20-wide expert that fails block-FP8 stays measurable.
    assert local_inter_error(moe_type, INTER, HIDDEN, 32, 1) is None


@pytest.mark.parametrize("moe_ep_size", [1, 2, 4, 8, 16, 32, 64, 128, 256])
def test_expert_parallel_never_shards_the_intermediate_dim(moe_ep_size):
    """EP splits experts, so every rank keeps the full 128-block-aligned expert.

    This is the regression that matters: moe_tp 1 / ep 8 is Qwen3.8-Flash-Next's
    serving topology, and ep 2-256 were all measured by the accepted grid.
    """
    local_inter_error = _load_local_inter_error()

    assert local_inter_error("fp8_block", INTER, HIDDEN, 1, moe_ep_size) is None


@pytest.mark.parametrize(
    ("moe_tp_size", "local_inter"),
    [
        pytest.param(1, 640, id="tp1-whole-128-block"),
        pytest.param(2, 320, id="tp2-refines-to-64"),
        pytest.param(4, 160, id="tp4-refines-to-32"),
    ],
)
def test_every_tp_the_full_grid_measured_stays_measurable(moe_tp_size, local_inter):
    """Pin what the real 2026-11-02 full grid collected.

    tp1 accepted (ep 1-256), tp2/tp4 were collected because vLLM refined the
    checkpoint block to 64/32. Guarding them would drop 27 measurable cases
    per TP and contradict rows from the accepted run.
    """
    local_inter_error = _load_local_inter_error()

    assert local_inter_error("fp8_block", INTER, HIDDEN, moe_tp_size, 1) is None


@pytest.mark.parametrize(
    ("moe_tp_size", "local_inter", "refinement"),
    [
        pytest.param(8, 80, 16, id="tp8-80-refines-to-16"),
        pytest.param(16, 40, 8, id="tp16-40"),
        pytest.param(32, 20, 4, id="tp32-20"),
    ],
)
def test_unrefinable_expert_tp_shards_are_cited_kernel_limits(moe_tp_size, local_inter, refinement):
    local_inter_error = _load_local_inter_error()

    error = local_inter_error("fp8_block", INTER, HIDDEN, moe_tp_size, 1)

    assert error is not None
    message = str(error)
    # collect._classify_exception keys on exactly this prefix; anything else
    # records the case as an unexpected collector failure on every rerun.
    assert message.lstrip().startswith("FIXME(kernel-limit):")
    assert f"inter_size={INTER} / {moe_tp_size} = {local_inter}" in message
    assert f"= {refinement} < 32" in message
    assert "oracle/fp8.py:296-315" in message


def test_hidden_size_participates_in_the_refinement():
    local_inter_error = _load_local_inter_error()

    # refine = gcd(gcd(128, local_inter), hidden), so a hidden size that
    # removes the last common power of two below 32 makes an otherwise
    # refinable shard unrefinable. tp2 is 320: gcd(128, 320) = 64.
    assert local_inter_error("fp8_block", INTER, 2560, 2, 1) is None  # gcd(64, 2560) = 64
    assert local_inter_error("fp8_block", INTER, 96, 2, 1) is None  # gcd(64, 96) = 32
    assert local_inter_error("fp8_block", INTER, 48, 2, 1) is not None  # gcd(64, 48) = 16


def test_guard_is_raised_before_fused_moe_is_constructed():
    # Ordering is the contract: refusing before construction means no expert
    # weights are allocated for a shape that must fail, and the raised type is
    # what collect.py classifies.
    source = SOURCE_PATH.read_text(encoding="utf-8")
    guard = source.index(
        "_fp8_block_moe_local_inter_error(moe_type, inter_size, hidden_size, moe_tp_size, moe_ep_size)"
    )
    construct = source.index("moe_module = FusedMoEFactory(")
    assert guard < construct
