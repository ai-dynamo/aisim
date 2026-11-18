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


@pytest.mark.parametrize(
    ("batch_size", "ctx_tokens", "requests", "new_tokens"),
    [
        # One request with 1096 new tokens under a 2048-token budget: one state.
        (1, 2048, 1, 1096),
        # Two complete cached requests keep both states.
        (16, 8192, 2, 2192),
    ],
)
def test_run_agg_prices_scheduled_prefill_requests_not_budget(
    monkeypatch, batch_size, ctx_tokens, requests, new_tokens
):
    from aisim.sdk.inference_session import InferenceSession
    from aisim_core.sdk import common, config
    from aisim_core.sdk.backends.factory import get_backend
    from aisim_core.sdk.models import get_model, nemotron_h
    from aisim_core.sdk.perf_database import get_database

    database = get_database("b200_sxm", "trtllm", "1.3.0rc20", allow_unlisted_version=True)
    if database is None:
        pytest.skip("b200_sxm/trtllm/1.3.0rc20 data missing")
    database.set_default_database_mode(common.DatabaseMode.HYBRID)
    real_ops = nemotron_h.ops

    def kernel(name, scale, source, phase, *args, **kwargs):
        return real_ops.Mamba2Kernel(name, scale, source, phase, *args, state_dtype="bfloat16", token_width=1, **kwargs)

    class QualifiedOps:
        def __getattr__(self, name):
            return kernel if name == "Mamba2Kernel" else getattr(real_ops, name)

    monkeypatch.setattr(nemotron_h, "ops", QualifiedOps())
    model_config = config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1)
    session = InferenceSession(
        get_model("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", model_config, "trtllm"),
        database,
        get_backend("trtllm", "b200_sxm"),
    )
    summary = session.run_agg(
        config.RuntimeConfig(batch_size=batch_size, isl=4096, osl=64, prefix=3000), ctx_tokens=ctx_tokens
    )
    # Bytes / bandwidth for 23 Mamba layers at b200_sxm's 7.7e12 B/s. Per token the
    # scan moves (4096 + 2 * 8 * 128 + 64) * 2 + 4096 * 2 = 20,608 bytes; each
    # request reads and writes its 1 MiB BF16 state once.
    expected_ms = 23 * (new_tokens * 20_608 + requests * 2 * 1_048_576) / 7.7e12 * 1000
    got = summary.get_per_ops_data()["mix_step"]["context_mamba_ssm"]
    assert got == pytest.approx(expected_ms, rel=1e-9)
