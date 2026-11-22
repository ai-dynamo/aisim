# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from pydantic import ValidationError

from aisim import EngineReplayRunnerFactory
from aisim.config import CorePredictionConfig, CoreRecommendationConfig
from aisim.config.engine import MtpDepthSearch
from aisim.predict import run_prediction
from aisim.recommend import run_recommendation
from aisim.sweeper.result import CandidateStatus

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_depth_curve_is_conditional_and_excludes_bonus():
    sweep = MtpDepthSearch(kind="mtp", depth={"choices": [0, 1, 3]}, acceptance_rates=[0.5, 0.5, 0.25])
    assert sweep.at_depth(0) is None
    assert sweep.at_depth(1).expected_accepted_tokens == 0.5
    # P(reach 1) + P(reach 2) + P(reach 3).
    assert sweep.at_depth(3).expected_accepted_tokens == 0.5 + 0.25 + 0.0625
    with pytest.raises(ValidationError, match="largest MTP depth"):
        MtpDepthSearch(kind="mtp", depth={"choices": [0, 3]}, acceptance_rates=[1])
    with pytest.raises(ValidationError):
        MtpDepthSearch(kind="mtp", depth={"choices": [8]}, acceptance_rates=[1] * 8)


@pytest.mark.parametrize("hybrid", [False, True], ids=["attention", "hybrid-offload"])
def test_mtp_search_replays_each_saved_depth_and_topology(hybrid, tmp_path):
    raw = {
        "engine": {
            "model": "Qwen/Qwen3-8B",
            "hardware": "h200_sxm",
            "backend": "vllm",
            "backend_version": "0.24.0",
            "mode": "aggregated",
            "context_length": 128,
            "estimation_mode": "op_level",
            "database_mode": "SOL",
            "speculation": {
                "kind": "mtp",
                "depth": {"choices": [0, 3, 7]},
                "acceptance_rates": [1, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5],
                "seed": 7,
                "disable_eagle_block_drop": True,
            },
            "workers": {
                "aggregated": {
                    "parallelism": {
                        "preset": [
                            {
                                "replicas": 1,
                                "tensor": tp,
                                "pipeline": 1,
                                "attention_data": 1,
                                "moe_tensor": 1,
                                "moe_expert": 1,
                            }
                            for tp in (1, 2)
                        ]
                    },
                    "scheduler": {"max_batched_tokens": 128, "max_sequences": 8},
                    "kv_cache": {
                        "block_size": 64,
                        "prefix_match_unit": 64,
                        "bytes_per_token": 128,
                        "capacity": {"type": "fixed", "blocks": 64},
                        "state_cache": {"bytes_per_request": 8192},
                        "host_offload": {
                            "scope": "dp_rank_local",
                            "num_host_blocks": 32,
                            "connector": "offloading_v0_32",
                        },
                    },
                }
            },
        },
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 8, "output_tokens": 20},
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 2},
        },
        "optimization": {"target": "throughput", "constraints": {"max_candidate_gpus": 2}},
        "optimizer": {"algorithm": "random", "max_trials": 6, "parallelism": 1, "seed": 42},
    }
    if hybrid:
        trace = tmp_path / "play.json"
        play = {
            "id": "test",
            "models": ["model"],
            "block_size": 64,
            "hash_id_scope": "local",
            "requests": [{"type": "s", "t": t, "model": "model", "in": 64, "out": 20, "hash_ids": [1]} for t in (0, 1)],
        }
        trace.write_text(json.dumps(play), encoding="utf-8")
        raw["traffic"] = {
            "source": {"type": "trace", "format": "weka", "paths": [str(trace)], "block_size": 64},
            "load": {"type": "trace_timestamps", "agentic_lanes": 1},
        }
    else:
        cache = raw["engine"]["workers"]["aggregated"]["kv_cache"]
        for key in ("state_cache", "prefix_match_unit", "host_offload"):
            del cache[key]
    config = CoreRecommendationConfig.model_validate(raw)
    factory = EngineReplayRunnerFactory(determinism="canonical_v1")
    result = run_recommendation(config, stack="engine", runner_factory=factory, show_progress=False)
    pairs = set()
    for candidate in result.candidates:
        assert candidate.status == CandidateStatus.FEASIBLE, candidate.reason
        saved = CorePredictionConfig.model_validate(candidate.prediction_config)
        depth = candidate.config["mtp_depth"]
        pairs.add((depth, saved.engine.workers.aggregated.parallelism.tensor))
        assert (saved.engine.speculation.num_speculative_tokens if saved.engine.speculation else 0) == depth
        if depth:
            assert saved.engine.speculation.acceptance_rates == raw["engine"]["speculation"]["acceptance_rates"][:depth]
        if hybrid:
            state = saved.engine.workers.aggregated.kv_cache.state_cache
            assert (state.num_speculative_states or 0) == depth
        direct = run_prediction(saved, stack="engine", runner_factory=factory)
        assert direct.summary["total_output_tokens"] == 40
        for key in ("total_output_tokens", "output_throughput_tok_s", "mean_output_token_throughput_per_user"):
            assert candidate.metrics[key] == direct.summary[key]
    assert pairs == {(depth, tp) for depth in (0, 3, 7) for tp in (1, 2)}
