# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from aisim import EngineReplayRunnerFactory
from aisim.config import CorePredictionConfig, CoreRecommendationConfig
from aisim.predict import run_prediction
from aisim.recommend import run_recommendation
from aisim.sweeper.result import CandidateStatus

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.fixture
def config():
    return {
        "engine": {
            "model": "Qwen/Qwen3-30B-A3B",
            "hardware": "h200_sxm",
            "backend": {"choices": ["vllm", "sglang"]},
            "context_length": 256,
            "mode": "aggregated",
            "workers": {
                "aggregated": {
                    "scheduler": {
                        "max_sequences": 8,
                        "max_batched_tokens": 64,
                        "prefill_schedule_interval": {"choices": [1, 4]},
                        "prefill_decode_interval": {"choices": [0, 4]},
                    },
                    "timing": {"type": "fixed", "prefill_ms": 3, "decode_ms": 1},
                    "kv_cache": {"block_size": 16, "capacity": {"type": "fixed", "blocks": 512}},
                }
            },
        },
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 64, "output_tokens": 12},
            "load": {"type": "concurrency", "concurrency": 8},
            "stop": {"requests": 24},
        },
        "optimization": {"target": "throughput", "constraints": {"max_candidate_gpus": 2}},
        "optimizer": {"algorithm": "random", "max_trials": 6, "parallelism": 1, "seed": 42},
    }


@pytest.mark.parametrize("algorithm", ["random", "bayesian"])
def test_interval_candidates_replay_only_on_matching_aggregate_dp(config, algorithm):
    expected = {
        ("vllm", 1, 1, 0),
        ("vllm", 2, 1, 0),
        ("vllm", 2, 4, 0),
        ("sglang", 1, 1, 0),
        ("sglang", 2, 1, 0),
        ("sglang", 2, 1, 4),
    }
    if algorithm == "random":
        config["engine"]["backend"]["choices"].append("trtllm")
        expected.update({("trtllm", 1, 1, 0), ("trtllm", 2, 1, 0)})
    config["optimizer"].update(algorithm=algorithm, max_trials=len(expected))
    config["engine"]["workers"]["aggregated"]["parallelism"] = {
        "preset": False,
        "replicas": 1,
        "tensor": 1,
        "pipeline": 1,
        "moe_tensor": 1,
        "attention_data": {"range": {"min": 1, "max": 2, "scale": "log"}},
        "moe_expert": {"choices": [1, 2]},
    }
    factory = EngineReplayRunnerFactory(determinism="canonical_v1")
    result = run_recommendation(
        CoreRecommendationConfig.model_validate(config), stack="engine", runner_factory=factory, show_progress=False
    )
    timings = {}
    for candidate in result.candidates:
        assert candidate.status == CandidateStatus.FEASIBLE, candidate.reason
        saved = CorePredictionConfig.model_validate(candidate.prediction_config)
        worker = saved.engine.workers.aggregated
        dp = worker.parallelism.attention_data
        intervals = (worker.scheduler.prefill_schedule_interval, worker.scheduler.prefill_decode_interval)
        key = (saved.engine.backend, dp, *intervals)
        assert key not in timings
        direct = run_prediction(saved, stack="engine", runner_factory=factory)
        timings[key] = candidate.metrics["output_throughput_tok_s"]
        assert timings[key] == direct.summary["output_throughput_tok_s"]
    assert set(timings) == expected
    for backend, changed in (("vllm", (4, 0)), ("sglang", (1, 4))):
        assert timings[backend, 2, 1, 0] != timings[backend, 2, *changed]


def test_mtp_depth_and_interval_are_independent_on_dp2(config):
    engine = config["engine"]
    engine.update(backend="vllm", backend_version="0.24.0", database_mode="SOL", estimation_mode="op_level")
    engine["speculation"] = {"kind": "mtp", "depth": {"choices": [0, 1]}, "acceptance_rates": [0.75]}
    worker = engine["workers"]["aggregated"]
    worker["parallelism"] = {
        "preset": [{"replicas": 1, "tensor": 1, "pipeline": 1, "attention_data": 2, "moe_tensor": 1, "moe_expert": 2}]
    }
    del worker["scheduler"]["prefill_decode_interval"]
    del worker["timing"]
    factory = EngineReplayRunnerFactory(determinism="canonical_v1")
    result = run_recommendation(
        CoreRecommendationConfig.model_validate(config), stack="engine", runner_factory=factory, show_progress=False
    )
    pairs = []
    for candidate in result.candidates:
        assert candidate.status == CandidateStatus.FEASIBLE, candidate.reason
        saved = CorePredictionConfig.model_validate(candidate.prediction_config)
        depth = saved.engine.speculation.num_speculative_tokens if saved.engine.speculation else 0
        pairs.append((depth, saved.engine.workers.aggregated.scheduler.prefill_schedule_interval))
        direct = run_prediction(saved, stack="engine", runner_factory=factory)
        assert direct.summary["output_throughput_tok_s"] == candidate.metrics["output_throughput_tok_s"]
    assert sorted(pairs) == [(0, 1), (0, 4), (1, 1), (1, 4)]


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_interval_domain_rejects_disaggregated_worker(config, role):
    engine = config["engine"]
    engine["mode"] = "disaggregated"
    worker = engine["workers"].pop("aggregated")
    engine["workers"] = {"prefill": {}, "decode": {}, role: worker}
    with pytest.raises(ValidationError, match="requires an aggregated"):
        CoreRecommendationConfig.model_validate(config)
