# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba2Kernel carries its recurrent-state dtype and speculative width."""

from __future__ import annotations

import copy
import pickle

import pytest

import aisim.sdk.operations as ops

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"state_dtype": "float16", "token_width": 4}, {"state_dtype": "float16", "token_width": 4}),
        ({}, {}),
    ],
)
def test_qualification_kwargs_survive_copy_and_pickle(kwargs, expected):
    op = ops.Mamba2Kernel(
        "generation_mamba_ssm", 1, "selective_state_update", "generation", 4096, 64, 64, 128, 4, 4, 128, **kwargs
    )
    for clone in (copy.deepcopy(op), pickle.loads(pickle.dumps(op))):
        assert clone.__getnewargs_ex__()[1] == expected


class LegacyKernel(ops.Mamba2Kernel):
    """A subclass written against the pre-qualification constructor."""

    def __new__(cls, name, scale, kernel, phase, hidden_size, nheads, head_dim, d_state, d_conv, n_groups, chunk_size):
        return super().__new__(
            cls, name, scale, kernel, phase, hidden_size, nheads, head_dim, d_state, d_conv, n_groups, chunk_size
        )


def test_legacy_subclass_still_unpickles():
    op = LegacyKernel("context_mamba_ssm", 1, "mamba_chunk_scan_combined", "context", 4096, 64, 64, 128, 4, 4, 128)
    clone = pickle.loads(pickle.dumps(op))
    assert type(clone) is LegacyKernel
    assert clone.__getnewargs_ex__() == op.__getnewargs_ex__()


def test_ad_hoc_query_takes_the_op_width_and_rejects_partial_requests():
    from aisim.sdk.perf_database import get_database

    database = get_database("b200_sxm", "trtllm", "1.3.0rc20", allow_unlisted_version=True)
    if database is None:
        pytest.skip("b200_sxm/trtllm/1.3.0rc20 data missing")
    op = ops.Mamba2Kernel(
        "generation_mamba_ssm",
        1,
        "selective_state_update",
        "generation",
        4096,
        64,
        64,
        128,
        4,
        4,
        128,
        state_dtype="bfloat16",
        token_width=4,
    )
    # Eight tokens at width four are two whole requests.
    result = op._engine_query(database, batch_size=8, s=16, x=8)
    assert float(result) > 0
    assert result.source == "sol"
    with pytest.raises(Exception, match="not a multiple of token_width 4"):
        op._engine_query(database, batch_size=5, s=16, x=5)
