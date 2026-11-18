# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Guangli Liu
# SPDX-License-Identifier: Apache-2.0

"""Explicit SGLang chunks survive search, saved prediction and native replay."""

import pytest

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.recommend import _candidate_prediction, recommendation_to_sweeper
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper.config import SearchSpace
from aisimulate.sweeper.deploy import build_backend_deployment
from aisimulate.sweeper.parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.replay import ReplaySpec
from aisimulate.sweeper.sample import unroll_sample
from aisimulate.sweeper.search_space import branch_knob_choices

pytestmark = pytest.mark.unit


def _config(role="aggregated", chunk=None):
    roles = ("aggregated",) if role == "aggregated" else ("prefill", "decode")
    workers = {
        name: {
            "scheduler": {"max_batched_tokens": 8192, "max_sequences": 8},
            "kv_cache": {"block_size": 1, "capacity": {"type": "fixed", "blocks": 8192}},
            "timing": {"type": "fixed", "prefill_ms": 10, "decode_ms": 1},
        }
        for name in roles
    }
    if chunk is not None:
        workers[role]["scheduler"]["chunked_prefill_size"] = chunk
    return {
        "engine": {
            "model": "Qwen/Qwen3-32B",
            "hardware": "h200_sxm",
            "backend": "sglang",
            "backend_version": "0.5.22",
            "context_length": 4096,
            "mode": "aggregated" if role == "aggregated" else "disaggregated",
            "workers": workers,
        },
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 256, "output_tokens": 4},
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 1},
        },
    }


@pytest.mark.parametrize("role", ["aggregated", "prefill", "decode"])
@pytest.mark.parametrize(
    "domain,expected,bounds",
    [
        (128, None, None),
        ({"choices": [64, 128, 256]}, [64, 128, 256], None),
        ({"range": {"min": 64, "max": 256, "step": 64}}, [64, 128, 192, 256], None),
        ({"range": {"min": 64, "max": 256, "scale": "log"}}, [64], [64, 256]),
    ],
)
def test_chunk_domain_round_trip(role, domain, expected, bounds):
    source = CoreRecommendationConfig.model_validate({**_config(role, domain), "optimization": {}})
    smart = recommendation_to_sweeper(source)
    space = smart.search_space
    mode = "agg" if role == "aggregated" else "disagg"
    internal_role = "agg" if role == "aggregated" else role
    key = f"{internal_role}_chunked_prefill_size"
    knobs = branch_knob_choices(space, mode)
    if expected is None:
        assert key not in knobs
        assert getattr(space, key) == 128
    else:
        assert knobs[key] == expected
    assert space.engine_integer_log_ranges.get(key) == bounds
    replica = ReplicaParallelConfig(ParallelShape(tp=1, dp=2, moe_tp=1, moe_ep=1), replicas=1)
    parallel = replica if mode == "agg" else DisaggParallelConfig(replica, replica)
    selection = {"deployment_mode": mode, "backend": "sglang"}
    for name in ("agg",) if mode == "agg" else ("prefill", "decode"):
        selection[f"{name}_max_num_batched_tokens"] = 8192
        selection[f"{name}_max_num_seqs"] = 8
    if expected is not None:
        selection[key] = 128
    sample = unroll_sample(search_space=space, selection=selection, parallel_config=parallel)
    assert sample[key] == 128
    deployment = build_backend_deployment(sample, backend_version="0.5.22")
    args = getattr(deployment, f"{internal_role}_engine_args")
    # Keep the global chunk intact: the native scheduler divides it by DP once.
    assert args["aic_attention_dp_size"] == 2
    assert args["sglang"]["chunked_prefill_size"] == 128
    assert args["sglang"]["max_prefill_tokens"] == 8192
    spec = ReplaySpec(backend_deployment=deployment, workload=smart.workload.model_dump(mode="json"), goal={})
    saved = _candidate_prediction(source, sample, spec, adapter_sections={})
    assert saved["engine"]["workers"][role]["scheduler"]["chunked_prefill_size"] == 128
    reloaded = CorePredictionConfig.model_validate(saved)
    replay = prediction_to_replay_spec(reloaded)
    assert getattr(replay.backend_deployment, f"{internal_role}_engine_args")["sglang"]["chunked_prefill_size"] == 128


@pytest.mark.parametrize(
    "value", [0, -1, True, 1.5, {"choices": []}, {"choices": [64, 0]}, {"range": {"min": 0, "max": 64, "step": 32}}]
)
def test_recommend_rejects_invalid_chunks(value):
    with pytest.raises(ValueError, match="chunked_prefill_size"):
        CoreRecommendationConfig.model_validate({**_config(chunk=value), "optimization": {}})


@pytest.mark.parametrize("backend", ["vllm", "trtllm", {"choices": ["sglang", "vllm"]}])
def test_recommend_rejects_chunks_with_other_backends(backend):
    raw = _config(chunk=128)
    raw["engine"]["backend"] = backend
    with pytest.raises(ValueError, match="chunked_prefill_size.*sglang"):
        CoreRecommendationConfig.model_validate({**raw, "optimization": {}})


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
def test_predict_rejects_chunks_with_other_backends(backend):
    raw = _config(chunk=128)
    raw["engine"]["backend"] = backend
    raw["engine"]["workers"]["aggregated"]["kv_cache"]["block_size"] = 64
    with pytest.raises(ValueError, match="chunked_prefill_size.*sglang"):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize("value", [[], [0], -1, True, 1.5])
def test_internal_search_space_rejects_invalid_chunks(value):
    with pytest.raises(ValueError, match="chunked_prefill_size"):
        SearchSpace(
            model_name="example/model", hardware_sku="example", backend=["sglang"], agg_chunked_prefill_size=value
        )


@pytest.mark.parametrize("chunk,expected", [(None, None), (128, 128)])
def test_prediction_preserves_omitted_chunk(chunk, expected):
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(_config(chunk=chunk)))
    assert spec.backend_deployment.agg_engine_args.get("sglang", {}).get("chunked_prefill_size") == expected


@pytest.mark.parametrize("role", ["aggregated", "prefill", "decode"])
@pytest.mark.parametrize("dp,chunk", [(1, 63), (2, 64), (2, 127)])
def test_prediction_rejects_sub_block_per_rank_chunks(role, dp, chunk):
    raw = _config(role, chunk)
    worker = raw["engine"]["workers"][role]
    worker["kv_cache"]["block_size"] = 64
    worker["parallelism"] = {"attention_data": dp}

    with pytest.raises(ValueError, match="chunked_prefill_size=.*at least one block.*attention_data"):
        prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))


@pytest.mark.parametrize("role", ["aggregated", "prefill", "decode"])
@pytest.mark.parametrize("dp,chunk", [(1, 64), (2, 128), (2, 192), (2, None)])
def test_prediction_accepts_at_least_one_block_per_rank_or_omitted_chunk(role, dp, chunk):
    raw = _config(role, chunk)
    worker = raw["engine"]["workers"][role]
    worker["kv_cache"]["block_size"] = 64
    worker["parallelism"] = {"attention_data": dp}

    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    internal_role = "agg" if role == "aggregated" else role
    args = getattr(spec.backend_deployment, f"{internal_role}_engine_args")
    assert args.get("sglang", {}).get("chunked_prefill_size") == chunk


@pytest.mark.parametrize("role", ["aggregated", "prefill", "decode"])
@pytest.mark.parametrize("chunk", [64, 128, 192])
def test_sweeper_requires_at_least_one_block_per_rank(role, chunk):
    raw = _config(role, chunk)
    raw["engine"]["workers"][role]["kv_cache"]["block_size"] = 64
    source = CoreRecommendationConfig.model_validate({**raw, "optimization": {}})
    smart = recommendation_to_sweeper(source)
    mode = "agg" if role == "aggregated" else "disagg"
    replica = ReplicaParallelConfig(ParallelShape(tp=1, dp=2, moe_tp=1, moe_ep=1), replicas=1)
    selection = {"deployment_mode": mode, "backend": "sglang"}
    for name in ("agg",) if mode == "agg" else ("prefill", "decode"):
        selection[f"{name}_max_num_batched_tokens"] = 8192
        selection[f"{name}_max_num_seqs"] = 8
    sample = unroll_sample(
        search_space=smart.search_space,
        selection=selection,
        parallel_config=replica if mode == "agg" else DisaggParallelConfig(replica, replica),
    )

    if chunk < 128:
        with pytest.raises(ValueError, match="chunked_prefill_size=.*at least one block.*attention_data"):
            build_backend_deployment(sample, backend_version="0.5.22")
    else:
        deployment = build_backend_deployment(sample, backend_version="0.5.22")
        internal_role = "agg" if role == "aggregated" else role
        assert getattr(deployment, f"{internal_role}_engine_args")["sglang"]["chunked_prefill_size"] == chunk


def test_explicit_chunk_replay_uses_evaluated_prefill_ceiling():
    raw = _config(chunk=16384)
    worker = raw["engine"]["workers"]["aggregated"]
    worker["scheduler"]["max_sequences"] = 64
    worker["kv_cache"].update(prefix_caching=False, capacity={"type": "fixed", "blocks": 65536})
    raw["traffic"]["source"]["output_tokens"] = 1
    raw["traffic"]["load"]["concurrency"] = 64
    raw["traffic"]["stop"]["requests"] = 64
    source = CoreRecommendationConfig.model_validate({**raw, "optimization": {}})
    smart = recommendation_to_sweeper(source)
    sample = unroll_sample(
        search_space=smart.search_space,
        selection={
            "deployment_mode": "agg",
            "backend": "sglang",
            "agg_max_num_batched_tokens": 8192,
            "agg_max_num_seqs": 64,
        },
        parallel_config=ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1),
    )
    deployment = build_backend_deployment(sample, backend_version="0.5.22")
    spec = ReplaySpec(backend_deployment=deployment, workload=smart.workload.model_dump(mode="json"), goal={})
    saved = _candidate_prediction(source, sample, spec, adapter_sections={})
    reloaded = prediction_to_replay_spec(CorePredictionConfig.model_validate(saved))

    for replay in (spec, reloaded):
        assert replay.backend_deployment.agg_engine_args["sglang"] == {
            "chunked_prefill_size": 16384,
            "max_prefill_tokens": 8192,
        }
        report = EngineReplayRunnerFactory().create(0).run(replay)
        assert report.metrics["completed_requests"] == 64
        # Two 8192-token batches, each containing 32 uncached 256-token prompts.
        assert report.metrics["mean_ttft_ms"] == pytest.approx(15)


@pytest.mark.parametrize("chunk,expected", [(None, 8192), (128, 128)])
def test_explicit_chunk_overrides_host_loop_fallback(chunk, expected):
    raw = _config(chunk=chunk)
    worker = raw["engine"]["workers"]["aggregated"]
    worker["host_loop"] = True
    worker.pop("timing")  # Host-loop replay requires native timing.
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    assert spec.backend_deployment.agg_engine_args["sglang"]["chunked_prefill_size"] == expected


@pytest.mark.parametrize("dp", [1, 2])
def test_chunk_controls_native_prefill_passes(dp):
    observed = []
    for chunk in (64 * dp, 128 * dp, 256 * dp):
        raw = _config(chunk=chunk)
        raw["engine"]["workers"]["aggregated"]["parallelism"] = {"attention_data": dp}
        spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
        report = EngineReplayRunnerFactory().create(0).run(spec)
        assert report.metrics["completed_requests"] == 1
        observed.append(report.metrics["mean_ttft_ms"])
    # One 256-token prompt, fixed 10 ms per prefill pass: four, two, one passes.
    assert observed == pytest.approx([40, 20, 10])
