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
    # Eight tokens at width four are two whole requests: 8 * 18,560 activation
    # bytes, plus each request's 1 MiB BF16 state read once and written four
    # times, at b200_sxm's 7.7e12 B/s.
    result = op._engine_query(database, batch_size=8, s=16, x=8)
    assert float(result) == pytest.approx(10_634_240 / 7.7e12 * 1000, rel=1e-9)
    assert result.source == "sol"
    with pytest.raises(Exception, match="not a multiple of token_width 4"):
        op._engine_query(database, batch_size=5, s=16, x=5)


def _qualified_nano_session(monkeypatch):
    from aisim.sdk.inference_session import InferenceSession
    from aisim_core.sdk import config
    from aisim_core.sdk.backends.factory import get_backend
    from aisim_core.sdk.models import get_model, nemotron_h
    from aisim_core.sdk.perf_database import get_database_view

    # An isolated view: the shared cached database keeps its default mode.
    database = get_database_view("b200_sxm", "trtllm", "1.3.0rc20", database_mode="HYBRID", allow_unlisted_version=True)
    if database is None:
        pytest.skip("b200_sxm/trtllm/1.3.0rc20 data missing")
    real_ops = nemotron_h.ops

    def kernel(name, scale, source, phase, *args, **kwargs):
        # float16 keeps the 2-byte state and never reads BF16 scan rows, so
        # the scan always takes the analytic estimate the oracle derives.
        return real_ops.Mamba2Kernel(name, scale, source, phase, *args, state_dtype="float16", token_width=1, **kwargs)

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

    def run(batch_size, ctx_tokens, prefix):
        return session.run_agg(
            config.RuntimeConfig(batch_size=batch_size, isl=4096, osl=64, prefix=prefix), ctx_tokens=ctx_tokens
        )

    return run


def _scan_ms(requests, new_tokens):
    # Bytes / bandwidth for 23 Mamba layers at b200_sxm's 7.7e12 B/s. Per token the
    # scan moves (4096 + 2 * 8 * 128 + 64) * 2 + 4096 * 2 = 20,608 bytes; each
    # request reads and writes its 1 MiB FP16 state once.
    return 23 * (new_tokens * 20_608 + requests * 2 * 1_048_576) / 7.7e12 * 1000


@pytest.mark.parametrize(
    ("batch_size", "ctx_tokens", "requests", "new_tokens"),
    [
        # One request with 1096 new tokens under a 2048-token budget: one state.
        (1, 2048, 1, 1096),
        # The same lone request under a budget of two prompts: still one state.
        (1, 8192, 1, 1096),
        # Two complete cached requests keep both states.
        (16, 8192, 2, 2192),
    ],
)
def test_run_agg_prices_scheduled_prefill_requests_not_budget(
    monkeypatch, batch_size, ctx_tokens, requests, new_tokens
):
    run = _qualified_nano_session(monkeypatch)
    summary = run(batch_size, ctx_tokens, prefix=3000)
    got = summary.get_per_ops_data()["mix_step"]["context_mamba_ssm"]
    assert got == pytest.approx(_scan_ms(requests, new_tokens), rel=1e-9)


def test_run_agg_lone_request_ignores_unused_budget(monkeypatch):
    # A budget above one prompt schedules the same single request as an exact
    # one-prompt budget, for every operation, not only Mamba.
    run = _qualified_nano_session(monkeypatch)
    exact, large = run(1, 4096, prefix=3000), run(1, 8192, prefix=3000)
    assert large.get_per_ops_data()["mix_step"] == exact.get_per_ops_data()["mix_step"]
    assert large.get_summary_df().iloc[0]["ttft"] == exact.get_summary_df().iloc[0]["ttft"]
    # The step's activations follow the scheduled prompt, not the budget.
    assert large.get_memory() == exact.get_memory()


def test_run_agg_cache_separates_prefixes(monkeypatch):
    run = _qualified_nano_session(monkeypatch)
    uncached, cached = run(1, 2048, prefix=0), run(1, 2048, prefix=3000)
    assert uncached.get_per_ops_data()["mix_step"]["context_mamba_ssm"] == pytest.approx(_scan_ms(1, 2048), rel=1e-9)
    assert cached.get_per_ops_data()["mix_step"]["context_mamba_ssm"] == pytest.approx(_scan_ms(1, 1096), rel=1e-9)
    # A repeated identical call reuses the matching result.
    assert run(1, 2048, prefix=3000) is cached
