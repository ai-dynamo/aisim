# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAFD split MoE stage provider tests."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig
from aisimulate.legacy_cli.api import cli_estimate
from aisimulate.sdk.inference_session import AFDInferenceSession
from aisimulate.sweeper import afd_fastafd
from aisimulate.sweeper.afd_fastafd import FastAFDMoEStagePerformanceModel
from aisimulate.sweeper.afd_parallel import AFDInfeasible, AFDTopology
from aisimulate.sweeper.afd_perfmodel import AFDLayerTimes, AFDMeasurementRequest, AICAFDPerformanceModel
from aisimulate_core.sdk.fastafd_profile import FASTAFD_PROFILE_SCHEMA

MODEL = "Qwen/Qwen3-235B-A22B-FP8"
SYSTEM = "gb200"
LAYERS = 94
STAGE_MS = 11.213456
COMPONENTS = {
    "attention_compute_ms": 0.09,
    "a_combine_ms": 0.006,
    "router_ms": 0.02,
    "a_router_ms": 0.002,
    "ffn_compute_ms": 0.6,
    "f_collective_ms": 0.036,
}


def _entry(**key_overrides) -> dict:
    key = {
        "model_path": MODEL,
        "system": SYSTEM,
        "stage": "afd",
        "topology": "64A8F",
        "logical_batch_per_source_rank": 128,
        "mtp_nextn": 0,
        "microbatches": 2,
        "moe_layers": LAYERS,
        "routed_topk": 8,
        "moe_precision": "w4a8_mxfp4_mxfp8",
        "moe_backend": "megamoe",
    }
    key.update(key_overrides)
    return {
        "key": key,
        "model_profile": "qwen3_235b_fp4",
        "latency_ms": STAGE_MS,
        "measurement": {
            "scope": "complete_moe_stage",
            "method": "cuda_event",
            "method_version": "torch 2.10.0+cu130; cuda 13.0",
            "statistic": "p50",
            "sample_count": 40,
            "cv_percent": 0.63,
            "routing": "balanced",
            "source_tree_sha256": "c" * 64,
            "raw_artifact": "qwen3_235b_fp4_mega_64a8f_s128_n0_mb2_balanced.json",
            "raw_sha256": "b" * 64,
        },
        "validation": {"stable": True, "correctness": None},
    }


def _profile(tmp_path, entries=None) -> str:
    path = tmp_path / "megamoe-m2n.json"
    path.write_text(
        json.dumps(
            {
                "schema": FASTAFD_PROFILE_SCHEMA,
                "lookup_policy": "exact-only",
                "source": {
                    "repository": "https://github.com/example/FastAFD",
                    "commit": "0" * 40,
                    "benchmark": "fastafd.moe-m2n-model-benchmark.v2",
                    "boundary": "split AFD MoE stage",
                },
                "entries": [_entry()] if entries is None else entries,
            }
        )
    )
    return str(path)


def _topology(**overrides) -> AFDTopology:
    values = dict(
        n_a_nodes=16,
        n_f_nodes=2,
        gpus_per_node=4,
        tp_a=1,
        a_batch_size=128,
        f_moe_ep_size=8,
        num_microbatches=2,
        phase="decode",
        combined_with_pd=True,
        is_moe=True,
        num_experts=128,
    )
    values.update(overrides)
    return AFDTopology(**values)


def _request(topology: AFDTopology, *, nextn: int = 0) -> AFDMeasurementRequest:
    return AFDMeasurementRequest(
        model_name=MODEL,
        hardware_sku=SYSTEM,
        backend="sglang",
        backend_version="0.5.14",
        topology=topology,
        input_length=1024,
        output_length=128,
        nextn=nextn,
    )


class _Attention:
    """An attention-side provider exposing the per-layer boundary components."""

    def __init__(self, *, num_layers: int = LAYERS, components: dict | None = COMPONENTS) -> None:
        self.num_layers = num_layers
        self.components = components
        self.requests = []

    def measure(self, request):
        self.requests.append(request)
        provenance = {"provider": "aic", "backend": request.backend}
        if self.components is not None:
            provenance["components"] = dict(self.components)
        return (
            AFDLayerTimes(
                phase="decode",
                attention_ms=0.096,
                ffn_ms=0.636,
                a_to_f_ms=0.0106,
                f_to_a_ms=0.0106,
                num_layers=self.num_layers,
                provenance=provenance,
            ),
        )


def _provider(tmp_path, attention=None, **overrides) -> FastAFDMoEStagePerformanceModel:
    values = dict(moe_backend="megamoe", attention_model=attention or _Attention())
    values.update(overrides)
    if "profile_path" not in values:
        values["profile_path"] = _profile(tmp_path)
    return FastAFDMoEStagePerformanceModel(**values)


def test_stage_latency_is_spread_over_layer_microbatch_units(tmp_path):
    attention = _Attention()
    (measurement,) = _provider(tmp_path, attention).measure(_request(_topology()))

    assert measurement.phase.value == "decode"
    assert measurement.ffn_ms == pytest.approx(STAGE_MS / (LAYERS * 2))
    assert measurement.attention_ms == pytest.approx(0.09 + 0.002)
    assert measurement.a_to_f_ms == 0.0
    assert measurement.f_to_a_ms == 0.0
    assert measurement.num_layers == LAYERS
    assert len(attention.requests) == 1

    provenance = measurement.provenance
    assert provenance["provider"] == "fastafd"
    assert provenance["stage"]["key"]["topology"] == "64A8F"
    assert provenance["stage"]["key"]["routed_topk"] == 8
    assert provenance["stage"]["latency_ms"] == STAGE_MS
    assert provenance["stage"]["raw_artifact"].endswith("_balanced.json")
    assert provenance["stage"]["profile_sha256"]
    assert provenance["attention"]["components"] == COMPONENTS
    assert provenance["attention"]["provider"] == "aic"
    assert set(provenance["composition"]) == {"attention_ms", "ffn_ms", "a_to_f_ms", "f_to_a_ms"}


@pytest.mark.parametrize("nextn", [0, 1])
def test_builtin_provider_recomputes_the_fastafd_router_on_a(tmp_path, monkeypatch, nextn):
    """Native SOL evidence with the synthetic 64A8F stage fixture, not serving accuracy."""
    local_router_batches = []
    sum_latency = AFDInferenceSession._sum_latency

    def trace_router_query(self, ops, **kwargs):
        ops = list(ops)
        if [op._name for op in ops] == ["generation_router_gemm"]:
            local_router_batches.append(kwargs["batch_size"])
        return sum_latency(self, ops, **kwargs)

    monkeypatch.setattr(AFDInferenceSession, "_sum_latency", trace_router_query)

    def estimate(*args, **kwargs):
        # Measure verification compute, without projecting service throughput
        # by accepted tokens (the legacy projection drops layer metadata).
        return cli_estimate(*args, **kwargs, database_mode="SOL", nextn_accepted=0.0 if nextn else None)

    request = replace(_request(_topology(), nextn=nextn), max_seq_len=2048)
    profile = _profile(tmp_path, [_entry(mtp_nextn=nextn)])
    (measurement,) = _provider(tmp_path, AICAFDPerformanceModel(estimate), profile_path=profile).measure(request)

    components = measurement.provenance["attention"]["components"]
    # 128 requests / 2 microbatches on each A, widened for verification;
    # the F query instead processes 64 times that many tokens.
    assert local_router_batches == [64 * (nextn + 1)]
    assert components["router_ms"] > components["a_router_ms"] > 0.0
    assert measurement.attention_ms == pytest.approx(components["attention_compute_ms"] + components["a_router_ms"])
    assert measurement.attention_ms < components["attention_compute_ms"] + components["router_ms"]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"phase": "both", "combined_with_pd": False}, "decode only"),
        ({"tp_a": 2}, "tp_a=2"),
        ({"f_moe_ep_size": 4}, "every F GPU"),
        ({"boundary_on_attn": False}, "boundary_on_attn=true"),
        ({"a_batch_size": 64}, "no FastAFD split MoE stage measurement"),
        ({"num_microbatches": 1}, "no FastAFD split MoE stage measurement"),
        ({"n_f_nodes": 4, "f_moe_ep_size": 16}, "no FastAFD split MoE stage measurement"),
    ],
)
def test_unmeasured_or_unsupported_topologies_fail_closed(tmp_path, overrides, match):
    with pytest.raises(AFDInfeasible, match=match):
        _provider(tmp_path).measure(_request(_topology(**overrides)))


def test_mtp_degree_is_part_of_the_key(tmp_path):
    with pytest.raises(AFDInfeasible, match="no FastAFD split MoE stage measurement"):
        _provider(tmp_path).measure(_request(_topology(), nextn=1))


def test_precision_is_selected_explicitly_when_several_entries_match(tmp_path):
    entries = [_entry(), _entry(moe_precision="fp8")]
    with pytest.raises(AFDInfeasible, match="set engine.moe_quant_mode"):
        _provider(tmp_path, profile_path=_profile(tmp_path, entries)).measure(_request(_topology()))

    (measurement,) = _provider(tmp_path, profile_path=_profile(tmp_path, entries), moe_precision="fp8").measure(
        _request(_topology())
    )
    assert measurement.provenance["stage"]["key"]["moe_precision"] == "fp8"


def test_layer_count_must_match_the_attention_model(tmp_path):
    with pytest.raises(AFDInfeasible, match="no FastAFD split MoE stage measurement"):
        _provider(tmp_path, _Attention(num_layers=95)).measure(_request(_topology()))


def test_routed_topk_must_match_the_resolved_model(tmp_path):
    profile = _profile(tmp_path, [_entry(routed_topk=4)])
    with pytest.raises(AFDInfeasible, match="no FastAFD split MoE stage measurement") as error:
        _provider(tmp_path, profile_path=profile).measure(_request(_topology()))
    assert error.value.provenance["routed_topk"] == 8


@pytest.mark.parametrize("reverse", [False, True])
def test_other_layer_counts_and_topk_do_not_make_an_exact_match_ambiguous(tmp_path, reverse):
    entries = [_entry(moe_layers=1), _entry(routed_topk=4), _entry()]
    if reverse:
        entries.reverse()
    (measurement,) = _provider(tmp_path, profile_path=_profile(tmp_path, entries)).measure(_request(_topology()))
    assert measurement.provenance["stage"]["key"]["moe_layers"] == LAYERS
    assert measurement.provenance["stage"]["key"]["routed_topk"] == 8


@pytest.mark.parametrize("topk", [None, 0, -1, True, "8", 8.5])
def test_unknown_or_invalid_model_topk_fails_closed(tmp_path, monkeypatch, topk):
    monkeypatch.setattr(afd_fastafd, "get_model_config_from_model_path", lambda _: {"topk": topk})
    with pytest.raises(AFDInfeasible, match="routed top-k must be a known positive integer"):
        _provider(tmp_path).measure(_request(_topology()))


def test_unresolvable_model_topk_fails_closed(tmp_path, monkeypatch):
    def missing_model(_):
        raise ValueError("unknown model")

    monkeypatch.setattr(afd_fastafd, "get_model_config_from_model_path", missing_model)
    with pytest.raises(AFDInfeasible, match="cannot resolve the model's routed top-k"):
        _provider(tmp_path).measure(_request(_topology()))


@pytest.mark.parametrize(
    ("components", "match"),
    [
        (None, "does not expose"),
        ({**COMPONENTS, "a_router_ms": 0.0}, "no router time"),
        ({"attention_compute_ms": 0.09}, "does not expose"),
    ],
)
def test_attention_provider_must_expose_the_boundary_components(tmp_path, components, match):
    with pytest.raises(AFDInfeasible, match=match):
        _provider(tmp_path, _Attention(components=components)).measure(_request(_topology()))


@pytest.mark.parametrize("missing", ["attention_compute_ms", "a_combine_ms", "a_router_ms"])
def test_each_consumed_attention_component_is_required(tmp_path, missing):
    components = {name: value for name, value in COMPONENTS.items() if name != missing}
    with pytest.raises(AFDInfeasible, match="does not expose"):
        _provider(tmp_path, _Attention(components=components)).measure(_request(_topology()))


def test_f_side_router_is_not_a_substitute_for_the_local_a_router(tmp_path):
    components = {name: value for name, value in COMPONENTS.items() if name != "a_router_ms"}
    assert components["router_ms"] > 0.0
    with pytest.raises(AFDInfeasible, match="a_router_ms"):
        _provider(tmp_path, _Attention(components=components)).measure(_request(_topology()))


@pytest.mark.parametrize("name", ["attention_compute_ms", "a_combine_ms", "a_router_ms"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.001, None, "0.002", True])
def test_invalid_attention_components_fail_closed(tmp_path, name, value):
    with pytest.raises(AFDInfeasible, match=f"{name} must be a finite non-negative number"):
        _provider(tmp_path, _Attention(components={**COMPONENTS, name: value})).measure(_request(_topology()))


@pytest.mark.parametrize("name", ["attention_compute_ms", "a_combine_ms"])
def test_attention_components_must_reconcile_with_the_headline(tmp_path, name):
    # The provider still reports 0.096 ms, but its components now sum to 0.106 ms.
    components = {**COMPONENTS, name: COMPONENTS[name] + 0.01}
    with pytest.raises(AFDInfeasible, match="do not reconcile with attention_ms"):
        _provider(tmp_path, _Attention(components=components)).measure(_request(_topology()))


def test_attention_component_reconciliation_tolerates_floating_point_roundoff(tmp_path):
    components = {**COMPONENTS, "attention_compute_ms": 0.09 + 1e-15}
    (measurement,) = _provider(tmp_path, _Attention(components=components)).measure(_request(_topology()))
    # The valid fixture keeps 0.09 ms of compute and adds the 0.002 ms router.
    assert measurement.attention_ms == pytest.approx(0.092)


def _prediction(profile_path: str) -> dict:
    return {
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 1024, "output_tokens": 128},
            "load": {"type": "concurrency", "concurrency": 16},
            "stop": {"requests": 32},
        },
        "engine": {
            "mode": "afd",
            "model": MODEL,
            "hardware": SYSTEM,
            "backend": "sglang",
            "afd": {
                "phase": "decode",
                "combined_with_pd": True,
                "n_a_nodes": 16,
                "n_f_nodes": 2,
                "tp_a": 1,
                "a_batch_size": 128,
                "f_moe_ep_size": 8,
                "num_microbatches": 2,
                "fastafd_profile_path": profile_path,
                "fastafd_moe_backend": "megamoe",
            },
            "workers": {
                "prefill": {
                    "parallelism": {"tensor": 4, "moe_expert": 4},
                    "timing": {"type": "fixed", "prefill_ms": 2.0, "decode_ms": 2.0},
                }
            },
        },
    }


def test_prediction_config_attaches_the_profile_stage(tmp_path):
    attention = _Attention()
    config = CorePredictionConfig.model_validate(_prediction(_profile(tmp_path)))

    spec = prediction_to_replay_spec(config, afd_performance_model=attention)

    metadata = spec.backend_deployment.performance_model_metadata["afd"]
    assert metadata["provider"] == "fastafd"
    assert metadata["measurement_required"] is False
    (measurement,) = metadata["measurements"]
    assert measurement["phase"] == "decode"
    assert measurement["ffn_ms"] == pytest.approx(STAGE_MS / (LAYERS * 2))
    assert measurement["a_to_f_ms"] == 0.0
    assert measurement["provenance"]["stage"]["key"]["topology"] == "64A8F"
    assert measurement["provenance"]["stage"]["key"]["routed_topk"] == 8
    assert spec.backend_deployment.parallel_config["afd"]["a_batch_size"] == 128
    assert "fastafd_profile_path" not in spec.backend_deployment.parallel_config["afd"]
    assert attention.requests[0].topology.ffn_gpus == 8


@pytest.mark.parametrize("precision", ["w4a8_mxfp4_mxfp8", "fp8"])
@pytest.mark.parametrize("companion_timing", ["default", "fixed"])
def test_prediction_config_selects_profile_precision(tmp_path, precision, companion_timing):
    fp8_entry = {**_entry(moe_precision="fp8"), "latency_ms": STAGE_MS * 2}
    raw = _prediction(_profile(tmp_path, [_entry(), fp8_entry]))
    raw["engine"]["moe_quant_mode"] = precision
    if companion_timing == "default":
        raw["engine"]["workers"]["prefill"].pop("timing")

    config = CorePredictionConfig.model_validate(raw)
    config = CorePredictionConfig.model_validate_json(config.model_dump_json())
    assert config.engine.moe_quant_mode == precision

    deployment = prediction_to_replay_spec(config, afd_performance_model=_Attention()).backend_deployment
    (measurement,) = deployment.performance_model_metadata["afd"]["measurements"]
    stage = measurement["provenance"]["stage"]
    expected_ms = fp8_entry["latency_ms"] if precision == "fp8" else STAGE_MS
    assert stage["key"]["moe_precision"] == precision
    assert measurement["ffn_ms"] == pytest.approx(expected_ms / (LAYERS * 2))
    # The precision selects the measured stage, not the companion's estimator.
    assert "moe_quant_mode" not in deployment.performance_model_metadata["prefill"]["config"]


@pytest.mark.parametrize("precision", [None, "unmeasured"])
def test_prediction_config_rejects_ambiguous_or_unmeasured_precision(tmp_path, precision):
    raw = _prediction(_profile(tmp_path, [_entry(), _entry(moe_precision="fp8")]))
    raw["engine"]["moe_quant_mode"] = precision
    config = CorePredictionConfig.model_validate(raw)
    match = "set engine.moe_quant_mode" if precision is None else "no FastAFD split MoE stage measurement"
    with pytest.raises(AFDInfeasible, match=match):
        prediction_to_replay_spec(config, afd_performance_model=_Attention())


def test_afd_precision_control_still_requires_a_profile(tmp_path):
    raw = _prediction(_profile(tmp_path))
    raw["engine"]["moe_quant_mode"] = "fp8"
    raw["engine"]["afd"].update(fastafd_profile_path=None, fastafd_moe_backend=None)
    with pytest.raises(ValidationError, match="engine model controls require default timing"):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize(
    "controls",
    [
        {"gemm_quant_mode": "fp8"},
        {"enable_eplb": True},
        {"nextn": 1, "nextn_accepted": 0.5},
    ],
)
def test_profile_precision_does_not_enable_other_afd_model_controls(tmp_path, controls):
    raw = _prediction(_profile(tmp_path))
    raw["engine"].update(moe_quant_mode="fp8", **controls)
    with pytest.raises(ValidationError, match="engine model controls require default timing"):
        CorePredictionConfig.model_validate(raw)


def test_prediction_config_rejects_a_profile_for_another_topk(tmp_path):
    config = CorePredictionConfig.model_validate(_prediction(_profile(tmp_path, [_entry(routed_topk=4)])))
    with pytest.raises(AFDInfeasible, match="no FastAFD split MoE stage measurement"):
        prediction_to_replay_spec(config, afd_performance_model=_Attention())


@pytest.mark.parametrize(
    ("afd_overrides", "match"),
    [
        ({"fastafd_moe_backend": None}, "configured together"),
        ({"fastafd_profile_path": None}, "configured together"),
        ({"fastafd_moe_backend": "unsupported"}, "Input should be"),
        ({"fastafd_profile_path": " "}, "cannot be empty"),
        ({"phase": "both", "combined_with_pd": False}, "requires phase='decode'"),
    ],
)
def test_profile_controls_are_validated(tmp_path, afd_overrides, match):
    raw = _prediction(_profile(tmp_path))
    raw["engine"]["afd"].update(afd_overrides)
    if raw["engine"]["afd"]["combined_with_pd"] is False:
        raw["engine"].pop("workers")
    with pytest.raises(ValidationError, match=match):
        CorePredictionConfig.model_validate(raw)
