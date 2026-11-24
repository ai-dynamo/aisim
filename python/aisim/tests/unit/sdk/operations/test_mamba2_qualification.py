# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba2Kernel carries its recurrent-state dtype and speculative width."""

from __future__ import annotations

import copy
import json
import pickle

import pytest

import aisim.sdk.operations as ops

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "speculation,qualified",
    [
        ({"kind": "ngram", "params": {"num_speculative_tokens": 3}}, False),
        ({"kind": "mtp", "params": {"depth": 3}}, True),
    ],
)
def test_hybrid_rebuild_preserves_construction_time_mtp_qualification(speculation, qualified):
    from aisim_core.sdk.config import ModelConfig
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.speculation.base import SpeculationConfig

    model = get_model(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, speculation=SpeculationConfig(**speculation)),
        "vllm",
    )
    assert model._nextn == 3  # Both schemes materialize the same verification width.

    def specs():
        return [
            json.loads(op._spec_json())["Mamba2"]
            for op in (*model.context_ops, *model.generation_ops)
            if isinstance(op, ops.Mamba2Kernel)
        ]

    before = specs()
    model.set_hybrid_config(copy.deepcopy(model._hybrid_config))
    after = specs()
    assert len(after) == 4
    assert after == before
    assert all((op["state_dtype"] != "unrecorded") == qualified for op in after)
    assert [op["token_width"] for op in after] == ([1, 1, 4, 4] if qualified else [1, 1, 1, 1])


@pytest.mark.parametrize(
    "mtp",
    [
        {"nextn": 1},
        {"nextn": 3},
        {"speculation": {"kind": "mtp", "params": {"depth": 3}}},
    ],
)
@pytest.mark.parametrize(("backend", "version"), [("vllm", "0.24.0"), ("trtllm", "1.3.0rc20")])
def test_canonical_mtp_prices_resolved_mamba_state_and_width(tmp_path, mtp, backend, version):
    from aisim_core.sdk.rust_engine_step import RustForwardPassPerfModel

    config = {
        "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "system": "b200_sxm",
        "backend": backend,
        "backend_version": version,
        "estimation_mode": "op_level",
        "fallback_policy": "deny",
        "worker_type": "aggregated",
        "database_mode": "HYBRID",
        "tp": 1,
        "moe_tp_size": 1,
        "moe_ep_size": 1,
        **mtp,
    }
    model = RustForwardPassPerfModel.best_available(config)
    nextn = mtp.get("nextn", 3)
    width = nextn + 1
    scheduled = {"num_decode_requests": 8 * width, "sum_decode_kv_tokens": 8 * width * 1025}
    # Nano's bundled config has 23 Mamba layers, 52 total layers, 64 heads
    # of dimension 64, 8 groups and state dimension 128. The declared FP32
    # recurrent state is 2 MiB/request. A scan transfers 20,608 activation
    # bytes/token; prefill reads+writes one state, decode reads once and
    # writes each of its `width` checkpoints. B200 bandwidth is 7.7e12 B/s.
    state_bytes = 2_097_152
    prefill_bytes = 17 * 20_608 + 2 * 2 * state_bytes
    decode_bytes = 8 * width * 20_608 + 8 * (1 + width) * state_bytes
    expected_decode = decode_bytes * 23 * (52 + nextn) / 52 / 7.7e12 * 1000
    # The convolution's 6,144 channels transfer 73,728 activation bytes per
    # token for its width-four window; it has no recurrent-state term.
    expected_conv = 8 * width * 73_728 * 23 * (52 + nextn) / 52 / 7.7e12 * 1000
    expected_prefill = prefill_bytes * 23 / 7.7e12 * 1000
    payloads = []
    for prefill in ({}, {"num_prefill_requests": 2, "sum_prefill_tokens": 17}):
        payload = [{"scheduled_requests": {**prefill, **scheduled}}]
        detail = model.estimate_forward_pass_detailed(payload)
        detail = json.loads(detail) if isinstance(detail, str) else detail
        rows = {op["name"]: op for op in detail["ranks"][0]["operations"]}
        assert rows["generation_mamba_ssm"]["latency_ms"] == pytest.approx(expected_decode, rel=1e-9)
        assert rows["generation_mamba_ssm"]["source"] == "sol"
        assert rows["generation_mamba_conv1d"]["latency_ms"] == pytest.approx(expected_conv, rel=1e-9)
        assert rows["generation_mamba_conv1d"]["source"] == "sol"
        if prefill:
            assert rows["context_mamba_ssm"]["latency_ms"] == pytest.approx(expected_prefill, rel=1e-9)
            assert rows["context_mamba_ssm"]["source"] == "sol"
        else:
            assert "context_mamba_ssm" not in rows or rows["context_mamba_ssm"]["latency_ms"] == 0
        payloads.append(payload)

    # Restoring the saved canonical configuration must rebuild the qualified
    # graph without constructor patches or an external execution profile.
    pinned = model.save_state_snapshot(tmp_path / "snapshot")
    restored = RustForwardPassPerfModel.best_available(json.loads(json.dumps(pinned)))
    for payload in payloads:
        assert restored.estimate_forward_pass_time_ms(payload) == model.estimate_forward_pass_time_ms(payload)


@pytest.mark.parametrize(
    ("state_dtype", "token_width", "message"),
    [
        ("fp16", 1, "unsupported state_dtype"),
        ("bfloat16", 0, "must be positive"),
        ("unrecorded", 0, "must be positive"),
        ("unrecorded", 4, "1 for an unrecorded state_dtype"),
    ],
)
def test_constructor_rejects_invalid_qualification(state_dtype, token_width, message):
    with pytest.raises(ValueError, match=message):
        ops.Mamba2Kernel(
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
            state_dtype=state_dtype,
            token_width=token_width,
        )


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

    def run(batch_size, ctx_tokens, prefix, *, isl=4096):
        return session.run_agg(
            config.RuntimeConfig(batch_size=batch_size, isl=isl, osl=64, prefix=prefix), ctx_tokens=ctx_tokens
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


def test_run_agg_lone_request_memory_ignores_unused_budget():
    # vLLM sizes the step's activations from its scheduled tokens (TensorRT-LLM
    # reserves max_num_tokens instead), so a budget above one prompt must not
    # enlarge the footprint of the single request it schedules.
    from aisim.sdk.inference_session import InferenceSession
    from aisim_core.sdk import config
    from aisim_core.sdk.backends.factory import get_backend
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.perf_database import get_database_view

    database = get_database_view("b200_sxm", "vllm", "0.24.0", database_mode="HYBRID", allow_unlisted_version=True)
    if database is None:
        pytest.skip("b200_sxm/vllm/0.24.0 data missing")
    model_config = config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1)
    session = InferenceSession(
        get_model("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", model_config, "vllm"),
        database,
        get_backend("vllm", "b200_sxm"),
    )

    def memory(ctx_tokens, isl=4096, prefix=3000):
        runtime = config.RuntimeConfig(batch_size=1, isl=isl, osl=64, prefix=prefix)
        return session.run_agg(runtime, ctx_tokens=ctx_tokens).get_memory()

    assert memory(8192) == memory(4096)
    # The activations follow the step's 1096 new tokens, as for an uncached
    # 1096-token prompt, not the 4096-token prompt with its cached prefix.
    assert memory(4096)["activations"] == memory(1096, isl=1096, prefix=0)["activations"]
    assert memory(4096)["activations"] < memory(4096, prefix=0)["activations"]


def _ttft(summary):
    return float(summary.get_summary_df().iloc[0]["ttft"])


@pytest.mark.parametrize(("prefix", "chunks", "qf"), [(3000, 1, 1.90), (0, 2, 1.95)])
def test_run_agg_ttft_counts_uncached_prefill_steps(monkeypatch, prefix, chunks, qf):
    # A 4096-token prompt under a 2048-token budget: with 3000 cached tokens the
    # 1096 uncached tokens finish in one step; uncached, the prompt needs two.
    # TensorRT-LLM's TTFT is step * chunks * qf, with qf = 2 + (steps - 3) / 20
    # for the prefill steps and no dispatch or encoder term.
    run = _qualified_nano_session(monkeypatch)
    summary = run(1, 2048, prefix=prefix)
    step_ms = sum(summary.get_per_ops_data()["mix_step"].values())
    assert _ttft(summary) == pytest.approx(step_ms * chunks * qf, abs=1e-3)


def test_run_agg_ttft_prices_uneven_uncached_chunks(monkeypatch):
    run = _qualified_nano_session(monkeypatch)
    summary = run(1, 512, prefix=3000)
    # Price each actual native chunk independently: 512, 512, then 72 fresh
    # tokens, with the prefix growing after each step. Every chunk pays one
    # state read/write pair. A full-size tail would overprice this request.
    chunks = [run(1, end, prefix=past, isl=end) for end, past in [(3512, 3000), (4024, 3512), (4096, 4024)]]
    chunk_ms = [sum(chunk.get_per_ops_data()["mix_step"].values()) for chunk in chunks]
    scan_ms = sum(chunk.get_per_ops_data()["mix_step"]["context_mamba_ssm"] for chunk in chunks)
    assert scan_ms == pytest.approx(_scan_ms(3, 1096), rel=1e-9)
    assert chunk_ms[-1] < chunk_ms[0]
    # TensorRT-LLM applies qf = 2 + (3 - 3) / 20 once to the three-step sum.
    assert _ttft(summary) == pytest.approx(sum(chunk_ms) * 2.0, abs=1e-3)


def test_run_agg_one_step_ttft_does_not_depend_on_the_budget(monkeypatch):
    # 128 uncached tokens fit a 128-token and a 4096-token budget alike: one
    # step at the same coordinates, one chunk, and qf = 2 + (1 - 3) / 20.
    run = _qualified_nano_session(monkeypatch)
    tight, full = run(1, 128, prefix=3968), run(1, 4096, prefix=3968)
    step_ms = sum(full.get_per_ops_data()["mix_step"].values())
    assert tight.get_per_ops_data()["mix_step"] == full.get_per_ops_data()["mix_step"]
    assert _ttft(tight) == _ttft(full) == pytest.approx(step_ms * 1.90, abs=1e-3)


def test_run_agg_rejects_a_fully_cached_prompt(monkeypatch):
    run = _qualified_nano_session(monkeypatch)
    with pytest.raises(Exception, match="isl must be greater than"):
        run(1, 2048, prefix=4096)


def test_run_agg_reuses_results_only_for_the_same_graph_and_database_mode():
    from aisim.sdk.inference_session import InferenceSession
    from aisim_core.sdk import common, config, rust_engine_step
    from aisim_core.sdk.backends.factory import get_backend
    from aisim_core.sdk.models import get_model, nemotron_h
    from aisim_core.sdk.perf_database import get_database, get_database_view

    model_name = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    hybrid = get_database_view("b200_sxm", "trtllm", "1.3.0rc20", database_mode="HYBRID", allow_unlisted_version=True)
    if hybrid is None:
        pytest.skip("b200_sxm/trtllm/1.3.0rc20 data missing")
    model_config = config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1)
    real_ops = nemotron_h.ops

    class QualifiedOps:
        def __getattr__(self, name):
            if name != "Mamba2Kernel":
                return getattr(real_ops, name)
            return lambda *a, **k: real_ops.Mamba2Kernel(*a, state_dtype="float32", token_width=1, **k)

    legacy = get_model(model_name, model_config, "trtllm")
    nemotron_h.ops = QualifiedOps()
    try:
        qualified = get_model(model_name, model_config, "trtllm")
    finally:
        nemotron_h.ops = real_ops

    def run(backend, model, database):
        runtime = config.RuntimeConfig(batch_size=1, isl=4096, osl=64, prefix=3000)
        return _ttft(InferenceSession(model, database, backend).run_agg(runtime, ctx_tokens=2048))

    def cold(model, database):
        rust_engine_step._engine_handle_cache_clear()
        return run(get_backend("trtllm", "b200_sxm"), model, database)

    expected = {id(legacy): cold(legacy, hybrid), id(qualified): cold(qualified, hybrid)}
    assert expected[id(legacy)] != expected[id(qualified)]
    for first, second in ((legacy, qualified), (qualified, legacy)):
        # One backend for both graphs, and fresh backends over one handle cache.
        rust_engine_step._engine_handle_cache_clear()
        backend = get_backend("trtllm", "b200_sxm")
        run(backend, first, hybrid)
        assert run(backend, second, hybrid) == expected[id(second)]
        rust_engine_step._engine_handle_cache_clear()
        run(get_backend("trtllm", "b200_sxm"), first, hybrid)
        assert run(get_backend("trtllm", "b200_sxm"), second, hybrid) == expected[id(second)]
    # A query mode changed in place on the same database object. The SOL
    # reference comes first, so the HYBRID handle and result stay warm.
    sol = get_database_view("b200_sxm", "trtllm", "1.3.0rc20", database_mode="SOL", allow_unlisted_version=True)
    sol_ttft = cold(legacy, sol)
    shared = get_database("b200_sxm", "trtllm", "1.3.0rc20", allow_unlisted_version=True)
    original = shared.get_default_database_mode()
    try:
        shared.set_default_database_mode(common.DatabaseMode.HYBRID)
        backend = get_backend("trtllm", "b200_sxm")
        hybrid_ttft = run(backend, legacy, shared)
        # The modes price differently, so a stale HYBRID result cannot pass.
        assert hybrid_ttft != sol_ttft
        shared.set_default_database_mode(common.DatabaseMode.SOL)
        # The backend's result cache, then a fresh backend over the handle cache.
        assert run(backend, legacy, shared) == sol_ttft
        assert run(get_backend("trtllm", "b200_sxm"), legacy, shared) == sol_ttft
    finally:
        shared.set_default_database_mode(original)


def _nano_with_state_dtypes(**dtypes):
    """A Nano graph whose Mamba2 ops take the state dtype named by op suffix
    (``conv1d`` or ``ssm``) in both phases; a missing suffix stays legacy."""
    from aisim_core.sdk import config
    from aisim_core.sdk.models import get_model, nemotron_h

    real_ops = nemotron_h.ops

    def kernel(name, *args, **kwargs):
        dtype = dtypes.get(name.rsplit("_", 1)[1])
        if dtype is None:
            return real_ops.Mamba2Kernel(name, *args, **kwargs)
        return real_ops.Mamba2Kernel(name, *args, state_dtype=dtype, token_width=1, **kwargs)

    class Ops:
        def __getattr__(self, name):
            return kernel if name == "Mamba2Kernel" else getattr(real_ops, name)

    nemotron_h.ops = Ops()
    try:
        model_config = config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1)
        return get_model("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", model_config, "trtllm")
    finally:
        nemotron_h.ops = real_ops


def _hybrid_trtllm_database():
    from aisim_core.sdk.perf_database import get_database_view

    database = get_database_view("b200_sxm", "trtllm", "1.3.0rc20", database_mode="HYBRID", allow_unlisted_version=True)
    if database is None:
        pytest.skip("b200_sxm/trtllm/1.3.0rc20 data missing")
    return database


def _agg_ttft(backend, model, database):
    from aisim.sdk.inference_session import InferenceSession
    from aisim_core.sdk import config

    runtime = config.RuntimeConfig(batch_size=1, isl=4096, osl=64, prefix=3000)
    return _ttft(InferenceSession(model, database, backend).run_agg(runtime, ctx_tokens=2048))


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_run_agg_reuse_observes_appended_qualified_operation(phase):
    import aisim_core
    from aisim.sdk.inference_session import InferenceSession
    from aisim_core.sdk import config, rust_engine_step
    from aisim_core.sdk.backends.factory import get_backend

    database = _hybrid_trtllm_database()
    model = _nano_with_state_dtypes(conv1d="float16", ssm="float16")
    cold = _nano_with_state_dtypes(conv1d="float16", ssm="float16")
    backend = get_backend("trtllm", "b200_sxm")
    runtime = config.RuntimeConfig(batch_size=1, isl=4096, osl=64, prefix=3000)

    def run(backend, model, budget):
        return InferenceSession(model, database, backend).run_agg(runtime, ctx_tokens=budget)

    rust_engine_step._engine_handle_cache_clear()
    warm = run(backend, model, 2048)
    old_identity = rust_engine_step.engine_identity(model, database)
    for graph in (model, cold):
        ops = getattr(graph, f"{phase}_ops")
        spec = json.loads(next(op for op in ops if op._name == f"{phase}_mamba_ssm")._spec_json())
        spec["Mamba2"]["name"] = f"extra_{phase}_ssm"
        ops.append(aisim_core.op_from_spec_json(json.dumps(spec)))
    assert rust_engine_step.engine_identity(model, database) != old_identity
    # First reuse the existing workload and result cache, then query a new
    # workload through the previously warmed compiled-handle cache. Compute
    # the cold references afterwards so they cannot prime either warm cache.
    actual = [run(backend, model, budget) for budget in (2048, 512)]
    assert actual[0] is not warm
    assert actual[0].get_per_ops_data() != warm.get_per_ops_data()
    rust_engine_step._engine_handle_cache_clear()
    for budget, got in zip((2048, 512), actual, strict=True):
        expected = run(get_backend("trtllm", "b200_sxm"), cold, budget)
        assert got.get_per_ops_data() == expected.get_per_ops_data()
        assert got.get_summary_df().equals(expected.get_summary_df())
        assert run(backend, model, budget) is got


def test_run_agg_reuse_keeps_each_mamba_op_state_dtype():
    # Swapping float16 and float32 between the convolution and the scan keeps
    # the same set of (phase, dtype, width) fields, but the scan's state traffic
    # differs, so neither the result nor the compiled handle may be shared.
    from aisim_core.sdk import rust_engine_step
    from aisim_core.sdk.backends.factory import get_backend

    database = _hybrid_trtllm_database()
    first = _nano_with_state_dtypes(conv1d="float16", ssm="float32")
    second = _nano_with_state_dtypes(conv1d="float32", ssm="float16")
    rust_engine_step._engine_handle_cache_clear()
    expected = _agg_ttft(get_backend("trtllm", "b200_sxm"), second, database)
    rust_engine_step._engine_handle_cache_clear()
    assert _agg_ttft(get_backend("trtllm", "b200_sxm"), first, database) != expected
    # The handle cache is warm with the first graph: a fresh backend, then the
    # aggregate cache of a backend that already priced the first graph.
    assert _agg_ttft(get_backend("trtllm", "b200_sxm"), second, database) == expected
    rust_engine_step._engine_handle_cache_clear()
    backend = get_backend("trtllm", "b200_sxm")
    _agg_ttft(backend, first, database)
    assert _agg_ttft(backend, second, database) == expected


def test_run_agg_reuse_keeps_the_partial_qualification_rejection():
    # A graph with one legacy convolution must still fail the compiled engine's
    # all-or-none qualification check after a fully qualified graph warmed both caches.
    from aisim_core.sdk import rust_engine_step
    from aisim_core.sdk.backends.factory import get_backend

    database = _hybrid_trtllm_database()
    full = _nano_with_state_dtypes(conv1d="float16", ssm="float16")
    partial = _nano_with_state_dtypes(ssm="float16")
    rust_engine_step._engine_handle_cache_clear()
    backend = get_backend("trtllm", "b200_sxm")
    _agg_ttft(backend, full, database)
    for warm in (backend, get_backend("trtllm", "b200_sxm")):
        with pytest.raises(Exception, match="must all record a state_dtype"):
            _agg_ttft(warm, partial, database)


def test_run_agg_reuse_keeps_a_wrapped_mamba_op_distinct():
    # A draft query width wraps the op in TokenScale, and a qualified op may
    # not be nested, so the wrapped graph fails cold and after the plain graph
    # with the same constructor arguments warmed both caches.
    from aisim_core.sdk import rust_engine_step
    from aisim_core.sdk.backends.factory import get_backend

    database = _hybrid_trtllm_database()
    plain = _nano_with_state_dtypes(conv1d="float16", ssm="float16")
    wrapped = _nano_with_state_dtypes(conv1d="float16", ssm="float16")
    next(op for op in wrapped.context_ops if op._name == "context_mamba_ssm")._draft_token_width = (1, 1)
    rust_engine_step._engine_handle_cache_clear()
    with pytest.raises(Exception, match="nested"):
        _agg_ttft(get_backend("trtllm", "b200_sxm"), wrapped, database)
    rust_engine_step._engine_handle_cache_clear()
    backend = get_backend("trtllm", "b200_sxm")
    _agg_ttft(backend, plain, database)
    for warm in (backend, get_backend("trtllm", "b200_sxm")):
        with pytest.raises(Exception, match="nested"):
            _agg_ttft(warm, wrapped, database)


def _nano_with_nested_scan(state_dtype):
    """A legacy Nano graph whose context scan sits inside an ``Overlap`` composite."""
    import json

    import aisim_core

    model = _nano_with_state_dtypes()
    index = next(i for i, op in enumerate(model.context_ops) if op._name == "context_mamba_ssm")
    scan = json.loads(model.context_ops[index]._spec_json())
    scan["Mamba2"]["state_dtype"] = state_dtype
    overlap = {"Overlap": {"name": "context_mamba_pair", "group_a": [scan], "group_b": []}}
    model.context_ops[index] = aisim_core.op_from_spec_json(json.dumps(overlap))
    return model


def test_run_agg_reuse_keeps_a_nested_mamba_op_distinct():
    # A qualified op may not sit in a composite. The identity reads the compiled
    # spec, so a nested qualified scan fails cold and after a legacy graph with
    # the same layout warmed both caches.
    from aisim_core.sdk import rust_engine_step
    from aisim_core.sdk.backends.factory import get_backend

    database = _hybrid_trtllm_database()
    legacy = _nano_with_nested_scan("unrecorded")
    qualified = _nano_with_nested_scan("float16")
    rust_engine_step._engine_handle_cache_clear()
    with pytest.raises(Exception, match="nested"):
        _agg_ttft(get_backend("trtllm", "b200_sxm"), qualified, database)
    rust_engine_step._engine_handle_cache_clear()
    backend = get_backend("trtllm", "b200_sxm")
    _agg_ttft(backend, legacy, database)
    for warm in (backend, get_backend("trtllm", "b200_sxm")):
        with pytest.raises(Exception, match="nested"):
            _agg_ttft(warm, qualified, database)


def test_engine_identity_keeps_fpm_wrapped_mamba_ops_distinct():
    # A whole-model FPM op carries the op-level graph in its compiled spec, so
    # two FPM graphs that differ only in a Mamba2 state dtype differ in identity.
    from aisim_core.sdk import rust_engine_step
    from aisim_core.sdk.operations import FPMForwardOp

    def fpm(**dtypes):
        model = _nano_with_state_dtypes(**dtypes)
        model.context_ops = [FPMForwardOp("prefill", model.config, model.model_path, sol_ops=model.context_ops)]
        model.generation_ops = []
        return model

    database = _hybrid_trtllm_database()
    first = fpm(conv1d="float16", ssm="float16")
    second = fpm(conv1d="float32", ssm="float16")
    assert rust_engine_step._mamba_qualification_identity(first)
    assert rust_engine_step.engine_identity(first, database) != rust_engine_step.engine_identity(second, database)


@pytest.mark.parametrize("wrapped", [False, True])
def test_unsupported_mamba_neighbor_rejects_with_a_warm_handle(monkeypatch, wrapped):
    from types import SimpleNamespace

    from aisim_core.sdk import engine, rust_engine_step
    from aisim_core.sdk.operations import FPMForwardOp

    database = _hybrid_trtllm_database()
    valid = _nano_with_state_dtypes(conv1d="float16", ssm="float16")
    invalid = _nano_with_state_dtypes(conv1d="float16", ssm="float16")
    invalid.context_ops.append(SimpleNamespace(_name="unsupported_neighbor"))
    if wrapped:
        invalid.context_ops = [FPMForwardOp("prefill", invalid.config, invalid.model_path, sol_ops=invalid.context_ops)]
    rust_engine_step._engine_handle_cache_clear()
    with pytest.raises(rust_engine_step.RustEngineUnsupportedError, match="unsupported_neighbor"):
        rust_engine_step._cached_engine_handle(invalid, database)
    rust_engine_step._engine_handle_cache_clear()
    handle = rust_engine_step._cached_engine_handle(valid, database)
    equivalent = _nano_with_state_dtypes(conv1d="float16", ssm="float16")
    assert rust_engine_step._cached_engine_handle(equivalent, database) is handle
    real_convert = engine._as_engine_op
    conversions = []

    def convert(op):
        conversions.append(op)
        return real_convert(op)

    monkeypatch.setattr(engine, "_as_engine_op", convert)
    for attempt in range(2):
        previous = len(conversions)
        with pytest.raises(rust_engine_step.RustEngineUnsupportedError, match="unsupported_neighbor"):
            rust_engine_step._cached_engine_handle(invalid, database)
        if attempt and not wrapped:
            assert len(conversions) == previous  # normal compilation rejection is cached
        assert rust_engine_step._cached_engine_handle(valid, database) is handle
    assert conversions  # the malformed graph never reuses the valid identity
    conversions.clear()
    assert rust_engine_step._cached_engine_handle(valid, database) is handle
    assert conversions == []


def _renamed(model, **names):
    """``model`` with each context op named by a key of ``names`` renamed to its value."""
    import json

    import aisim_core

    for index, op in enumerate(model.context_ops):
        if op._name in names:
            spec = json.loads(op._spec_json())
            (body,) = spec.values()
            body["name"] = names[op._name]
            model.context_ops[index] = aisim_core.op_from_spec_json(json.dumps(spec))
    return model


def test_run_agg_reuse_keeps_a_renamed_neighbor_rejection():
    # Build validation reads every op name: a neighbor named like the qualified
    # scan must fail cold and after the valid graph warmed both caches.
    from aisim_core.sdk import rust_engine_step
    from aisim_core.sdk.backends.factory import get_backend

    database = _hybrid_trtllm_database()
    valid = _nano_with_state_dtypes(conv1d="float16", ssm="float16")
    clash = _renamed(_nano_with_state_dtypes(conv1d="float16", ssm="float16"), context_mamba_norm="context_mamba_ssm")
    rust_engine_step._engine_handle_cache_clear()
    with pytest.raises(Exception, match="invalid engine config"):
        _agg_ttft(get_backend("trtllm", "b200_sxm"), clash, database)
    rust_engine_step._engine_handle_cache_clear()
    backend = get_backend("trtllm", "b200_sxm")
    _agg_ttft(backend, valid, database)
    for warm in (backend, get_backend("trtllm", "b200_sxm")):
        with pytest.raises(Exception, match="invalid engine config"):
            _agg_ttft(warm, clash, database)


def test_mixed_power_coverage_counts_each_mamba_query(tmp_path):
    # The qualified convolution prices a powered measured row and the scan
    # prices unpowered SOL. Under one name their rows fold, so coverage must
    # come from each query, not from the folded row's energy.
    import json

    # Synthetic row: exactly 2 ms at 500 W for the convolution. There is no
    # scan row, so its qualified FP16 query uses unpowered SOL independently
    # of any installed performance data.
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    import aisim_core
    from aisim_core.sdk import rust_engine_step
    from aisim_core.sdk.perf_database import PerfDatabase

    systems = tmp_path / "systems"
    systems.mkdir()
    source = Path(aisim_core.__file__).parent / "systems" / "b200_sxm.yaml"
    (systems / source.name).write_text(source.read_text())
    data = systems / "data/b200_sxm/linear_attention/trtllm/1.3.0rc20"
    data.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "kernel_source": ["causal_conv1d_fn"],
                "phase": ["context"],
                "batch_size": [1],
                "seq_len": [2048],
                "d_model": [2688],
                "d_state": [128],
                "d_conv": [4],
                "nheads": [64],
                "head_dim": [64],
                "n_groups": [8],
                "chunk_size": [128],
                "latency": [2.0],
                "power": [500.0],
            }
        ),
        data / "mamba2_perf.parquet",
    )
    database = PerfDatabase(
        "b200_sxm", "trtllm", "1.3.0rc20", str(systems), database_mode="HYBRID", strict_provenance=False
    )

    def step(model):
        return rust_engine_step.estimate_mixed_step_breakdown_with_rust(
            model, database, ctx_tokens=2048, gen_tokens=0, isl=2048, osl=64, prefix=0
        )

    apart = _nano_with_state_dtypes(conv1d="bfloat16", ssm="float16")
    shared = _nano_with_state_dtypes(conv1d="bfloat16", ssm="float16")
    for model in (apart, shared):
        model.context_ops = [
            op for op in model.context_ops if op._name in ("context_mamba_conv1d", "context_mamba_ssm")
        ]
        model.generation_ops = []
    index = next(i for i, op in enumerate(shared.context_ops) if op._name == "context_mamba_ssm")
    scan = json.loads(shared.context_ops[index]._spec_json())
    scan["Mamba2"]["name"] = "context_mamba_conv1d"
    shared.context_ops[index] = aisim_core.op_from_spec_json(json.dumps(scan))
    one, two = step(apart), step(shared)
    latency, energy = one["per_op_latency_ms"], one["per_op_energy_wms"]
    assert energy["context_mamba_conv1d"] > 0 and energy["context_mamba_ssm"] == 0
    # Coverage counts the whole powered convolution and none of the scan.
    covered = one["covered_latency_ms"]
    assert latency["context_mamba_conv1d"] == pytest.approx(46.0)  # 23 layers * 2 ms
    assert covered == pytest.approx(46.0)
    folded = two["per_op_latency_ms"]["context_mamba_conv1d"]
    assert folded == pytest.approx(latency["context_mamba_conv1d"] + latency["context_mamba_ssm"], rel=1e-12)
    assert two["covered_latency_ms"] == pytest.approx(one["covered_latency_ms"], rel=1e-12)


def test_hybrid_rebuild_invalidates_warm_engine_and_aggregate_caches():
    from dataclasses import replace

    from aisim_core.sdk import config, rust_engine_step
    from aisim_core.sdk.backends.factory import get_backend
    from aisim_core.sdk.models import get_model

    database = _hybrid_trtllm_database()
    model = get_model(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, nextn=3),
        "trtllm",
    )
    backend = get_backend("trtllm", "b200_sxm")
    first = _agg_ttft(backend, model, database)
    identity = rust_engine_step.engine_identity(model, database)
    handle = rust_engine_step._cached_engine_handle(model, database)
    assert rust_engine_step._cached_engine_handle(model, database) is handle
    model.set_hybrid_config(replace(model._hybrid_config, mamba_ssm_cache_dtype="float16"))
    assert rust_engine_step.engine_identity(model, database) != identity
    assert rust_engine_step._cached_engine_handle(model, database) is not handle
    rebuilt = _agg_ttft(backend, model, database)
    fresh = get_model(model.model_path, copy.deepcopy(model.config), "trtllm")
    fresh.set_hybrid_config(copy.deepcopy(model._hybrid_config))
    assert rebuilt == _agg_ttft(get_backend("trtllm", "b200_sxm"), fresh, database)
    assert rebuilt != first


@pytest.mark.parametrize("reserved", ["context_attention", "context_attention (scaled)"])
def test_qualified_scan_rejects_reserved_attention_names(monkeypatch, reserved):
    # The SDK reports these names as amortized attention, which XPU TTFT
    # rescales. The compiled build rejects them on every SDK path, and the
    # canonical model does not fall back to another estimator.
    from aisim_core.sdk import common, config, rust_engine_step
    from aisim_core.sdk.backends.factory import get_backend
    from aisim_core.sdk.models import nemotron_h
    from aisim_core.sdk.rust_engine_step import RustForwardPassPerfModel
    from aisim_core.sdk.step_estimate import MixedStepInput

    database = _hybrid_trtllm_database()
    runtime = config.RuntimeConfig(batch_size=1, beam_width=1, isl=2048, osl=64, prefix=0)
    for name, rejected in (("context_ssm", False), (reserved, True)):
        model = _renamed(_nano_with_state_dtypes(conv1d="bfloat16", ssm="bfloat16"), context_mamba_ssm=name)
        rust_engine_step._engine_handle_cache_clear()
        calls = (
            lambda m=model: rust_engine_step.estimate_mixed_step_breakdown_with_rust(
                m, database, ctx_tokens=2048, gen_tokens=0, isl=2048, osl=64, prefix=0
            ),
            # The mixed step the XPU TTFT path prices.
            lambda m=model: get_backend("trtllm").run_mixed(m, database, runtime, MixedStepInput(2048, 0)),
        )
        for call in calls:
            if rejected:
                with pytest.raises(ValueError, match="reserved attention name"):
                    call()
            else:
                call()

    real_ops = nemotron_h.ops

    def kernel(name, *args, **kwargs):
        name = reserved if name == "context_mamba_ssm" else name
        return real_ops.Mamba2Kernel(name, *args, state_dtype="bfloat16", token_width=1, **kwargs)

    class Ops:
        def __getattr__(self, attr):
            return kernel if attr == "Mamba2Kernel" else getattr(real_ops, attr)

    monkeypatch.setattr(nemotron_h, "ops", Ops())
    with pytest.raises(ValueError, match="reserved attention name"):
        RustForwardPassPerfModel.best_available(
            {
                "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                "system": "b200_sxm",
                "backend": "trtllm",
                "backend_version": "1.3.0rc20",
                "estimation_mode": "auto",
                "tp": 1,
                "moe_tp_size": 1,
                "moe_ep_size": 1,
                "worker_type": "aggregated",
                "database_mode": common.DatabaseMode.HYBRID.name,
            }
        )


def test_generation_only_qualification_rejects_shared_u32_overflow():
    from aisim_core.sdk.config import ModelConfig
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.perf_database import get_database_view
    from aisim_core.sdk.rust_engine_step import estimate_mixed_step_latency_with_rust

    model = get_model(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, nextn=3),
        "vllm",
    )
    model.context_ops = [ops.ElementWise("norm", 1, 1, 0)]
    database = get_database_view("b200_sxm", "vllm", "0.24.0", database_mode="SOL")
    requests = (2**32 - 1) // 4

    def estimate(tokens):
        return estimate_mixed_step_latency_with_rust(
            model, database, ctx_tokens=tokens, gen_tokens=requests, isl=8, osl=2, prefix=0
        )

    # 4_294_967_292 packed decode positions fit; three more reach u32::MAX.
    # Eight prefill positions must reject, never silently saturate the norm.
    assert estimate(3) > 0
    with pytest.raises(ValueError, match="shared tokens overflow"):
        estimate(8)
