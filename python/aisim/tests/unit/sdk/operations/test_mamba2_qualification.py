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
