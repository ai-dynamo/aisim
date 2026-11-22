# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAFD split MoE stage provider tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig
from aisimulate.legacy_cli.api import cli_estimate
from aisimulate.runner import EngineReplayRunnerFactory, _afd_layer_measurements, _afd_topology
from aisimulate.sdk.inference_session import AFDInferenceSession
from aisimulate.sweeper import afd_fastafd
from aisimulate.sweeper.afd_engine import AFDForegroundEngine, evaluate_afd_phase
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
                    "boundary_components": ["dispatch", "experts", "return", "combine"],
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


def test_attention_provenance_snapshots_validated_components(tmp_path):
    class RetainingAttention(_Attention):
        def measure(self, request):
            measurements = super().measure(request)
            self.retained_components = measurements[0].provenance["components"]
            self.retained_components["diagnostic_ms"] = 0.1
            return measurements

    attention = RetainingAttention()
    (measurement,) = _provider(tmp_path, attention).measure(_request(_topology()))
    components = measurement.provenance["attention"]["components"]
    assert components == {**COMPONENTS, "diagnostic_ms": 0.1}
    # The provider keeps its mutable mapping after returning the measurement.
    attention.retained_components["a_router_ms"] = 0.004
    attention.retained_components["diagnostic_ms"] = 0.2
    assert components == {**COMPONENTS, "diagnostic_ms": 0.1}
    assert measurement.attention_ms == pytest.approx(0.092)  # 0.09 compute + 0.002 router
    assert type(components["attention_compute_ms"]) is float


@pytest.mark.parametrize(
    "components",
    [None, ["dispatch", "experts", "return"], ["router", "dispatch", "experts", "return", "combine"]],
)
def test_matching_capture_with_incompatible_boundary_is_infeasible(tmp_path, components):
    path = Path(_profile(tmp_path))
    payload = json.loads(path.read_text())
    payload["source"]["boundary_components"] = components
    path.write_text(json.dumps(payload))
    # The key matches, but recomposition cannot remove combine or charge a
    # router twice just because the description calls this a complete stage.
    with pytest.raises(AFDInfeasible, match="boundary_components"):
        _provider(tmp_path, profile_path=path).measure(_request(_topology()))


@pytest.mark.parametrize("response", ["extra-prefill", "extra-invalid", "prefill", "invalid", "empty", "duplicate"])
def test_attention_provider_response_must_be_exactly_one_decode_measurement(tmp_path, response):
    request = _request(_topology())
    (decode,) = _Attention().measure(request)
    prefill = replace(decode, phase="prefill")
    measurements = {
        "extra-prefill": (decode, prefill),
        "extra-invalid": (decode, object()),
        "prefill": (prefill,),
        "invalid": (object(),),
        "empty": (),
        "duplicate": (decode, decode),
    }[response]
    attention = SimpleNamespace(measure=lambda _: measurements)

    with pytest.raises(AFDInfeasible, match="must return exactly one decode AFDLayerTimes"):
        _provider(tmp_path, attention).measure(request)


@pytest.mark.parametrize("factor", [0.5, 2.0, 100.0, 1.0000000001])
def test_non_unit_communication_calibration_fails_before_measurement(tmp_path, factor):
    attention = _Attention()
    provider = _provider(tmp_path, attention, profile_path=tmp_path / "missing-profile.json")
    with pytest.raises(AFDInfeasible, match="FastAFD requires comm_overhead_factor=1") as error:
        provider.measure(_request(_topology(comm_overhead_factor=factor)))
    assert error.value.provenance["comm_overhead_factor"] == factor
    assert attention.requests == []


@pytest.mark.parametrize("nextn", [0, 1])
def test_builtin_provider_recomputes_the_fastafd_router_on_a(tmp_path, monkeypatch, nextn, fastafd_attention_sol):
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

    local_router_batches.clear()
    (attention_only,) = _provider(tmp_path, AICAFDPerformanceModel(), profile_path=profile).measure(request)
    assert local_router_batches == [64 * (nextn + 1)]
    # The full provider subtracts its combine term; the A-only path never
    # queries it. Their recomposition differs only by floating-point rounding.
    assert attention_only.attention_ms == pytest.approx(measurement.attention_ms, rel=1e-14, abs=1e-16)
    assert attention_only.ffn_ms == measurement.ffn_ms


@pytest.fixture
def fastafd_attention_sol(monkeypatch):
    """Native SOL timings and real memory accounting; no serving-accuracy claim."""
    from aisimulate.sdk import perf_database

    load = perf_database.get_database_view

    def sol_view(system, backend, version, **kwargs):
        kwargs.update(database_mode="SOL", allow_missing_data=True)
        return load(system, backend, version, **kwargs)

    monkeypatch.setattr(perf_database, "get_database_view", sol_view)


def _kimi_request_and_profile(tmp_path, precision, *, moe_backend="megamoe"):
    model = "moonshotai/Kimi-K2-Instruct"
    entry = _entry(model_path=model, topology="64A4F", moe_layers=61, moe_precision=precision, moe_backend=moe_backend)
    request = replace(
        _request(_topology(n_f_nodes=1, f_moe_ep_size=4, num_experts=384)), model_name=model, max_seq_len=2048
    )
    return request, _profile(tmp_path, [entry])


@pytest.mark.parametrize("moe_backend", ["megamoe", "deepep_deepgemm"])
def test_builtin_fastafd_qualifies_selected_low_bit_f_memory(tmp_path, monkeypatch, fastafd_attention_sol, moe_backend):
    """Kimi FP8 F weights exceed GB200 capacity; the measured 4-bit stage fits."""
    from aisimulate_core.sdk import RustForwardPassPerfModel

    request, profile = _kimi_request_and_profile(tmp_path, "w4a8_mxfp4_mxfp8", moe_backend=moe_backend)
    with pytest.raises(RuntimeError, match="OOM"):
        cli_estimate(
            request.model_name,
            SYSTEM,
            mode="afd",
            backend_name=request.backend,
            backend_version=request.backend_version,
            database_mode="SOL",
            n_a_nodes=16,
            n_f_nodes=1,
            a_batch_size=128,
            f_moe_ep_size=4,
            num_microbatches=2,
            afd_combined_with_pd=False,
            isl=request.input_length,
            osl=request.output_length,
            max_seq_len=request.max_seq_len,
        )

    queries = []
    sum_latency = AFDInferenceSession._sum_latency

    def attention_only(self, ops, **kwargs):
        assert kwargs["model"].config.tp_size == 1, "discarded F timing must never be queried"
        queries.append(kwargs["seq_len"])
        return sum_latency(self, ops, **kwargs)

    def no_communication(*args, **kwargs):
        raise AssertionError("communication is already inside the measured stage")

    monkeypatch.setattr(AFDInferenceSession, "_sum_latency", attention_only)
    monkeypatch.setattr(AFDInferenceSession, "_build_afd_comm_ops", no_communication)
    (measurement,) = _provider(
        tmp_path, AICAFDPerformanceModel(), profile_path=profile, moe_backend=moe_backend
    ).measure(request)
    provenance = measurement.provenance["attention"]
    assert provenance["measurement_scope"] == "attention_only"
    assert provenance["config"] == measurement.provenance["config"]
    assert provenance["config"]["moe_quant_mode"] == "w4a8_mxfp4_mxfp8"
    assert provenance["config"]["afd"]["ffn_gpus"] == 4
    memory = provenance["memory"]
    assert memory["f_worker"]["kvcache"] == 0.0
    assert memory["a_worker"]["kvcache"] > 0.0
    assert 0.0 < memory["f_worker"]["weights"] < 198674743296 / (1 << 30)
    assert queries == [1025, 1057, 1089, 1121, 1025]
    assert measurement.ffn_ms == pytest.approx(STAGE_MS / (61 * 2))
    assert measurement.attention_ms > provenance["components"]["a_router_ms"] > 0.0

    # Reconstruction uses the resolved precision and pinned pool identity too.
    restored = RustForwardPassPerfModel.best_available(provenance["config"])
    attention = restored.estimate_fastafd_attention(isl=1024, osl=128, max_seq_len=2048)
    assert attention["provenance"]["memory"] == memory
    assert attention["attention_ms"] == provenance["components"]["attention_compute_ms"]


def test_external_aic_attention_layer_count_must_match_the_selected_stage(tmp_path):
    def estimate(*args, **kwargs):
        return SimpleNamespace(
            raw={
                "afd_layer_measurements": {
                    "decode": {
                        "attention_ms": 0.096,
                        "ffn_ms": 0.6,
                        "a_to_f_ms": 0.01,
                        "f_to_a_ms": 0.01,
                        "num_layers": 32,
                        "components": COMPONENTS,
                    }
                }
            }
        )

    with pytest.raises(AFDInfeasible, match="layer count does not match"):
        _provider(tmp_path, AICAFDPerformanceModel(estimate)).measure(_request(_topology()))


@pytest.mark.parametrize(("precision", "max_seq_len", "pool"), [("fp8", 2048, "F"), ("w4a8_mxfp4_mxfp8", 131072, "A")])
def test_builtin_fastafd_still_rejects_oom_in_either_selected_pool(
    tmp_path, fastafd_attention_sol, precision, max_seq_len, pool
):
    request, profile = _kimi_request_and_profile(tmp_path, precision)
    request = replace(request, max_seq_len=max_seq_len)
    with pytest.raises(AFDInfeasible, match=f"OOM: FastAFD {pool} pool"):
        _provider(tmp_path, AICAFDPerformanceModel(), profile_path=profile).measure(request)


@pytest.mark.parametrize("nextn", [0, 1])
def test_ordinary_afd_evaluates_nested_native_routers(monkeypatch, nextn):
    """Shared-expert decode graphs must work without a FastAFD profile."""
    router_queries = []
    sum_latency = AFDInferenceSession._sum_latency

    def trace_router_query(self, ops, **kwargs):
        ops = list(ops)
        result = sum_latency(self, ops, **kwargs)
        if ops and all("router" in op._name for op in ops):
            # Overlap getters reconstruct native leaves without Python shims.
            assert all(not hasattr(op, "_engine_query") for op in ops)
            router_queries.append((kwargs["batch_size"], result))
        return result

    monkeypatch.setattr(AFDInferenceSession, "_sum_latency", trace_router_query)
    result = cli_estimate(
        "deepseek-ai/DeepSeek-V3",
        "b200_sxm",
        mode="afd",
        backend_name="vllm",
        backend_version="current",
        database_mode="SOL",
        n_a_nodes=2,
        n_f_nodes=1,
        a_batch_size=129,
        num_microbatches=2,
        f_moe_ep_size=8,
        afd_combined_with_pd=False,
        isl=1024,
        osl=128,
        max_seq_len=2048,
        nextn=nextn,
        nextn_accepted=0.0 if nextn else None,
    )

    assert len(router_queries) == 2
    batch, (total_ms, per_op) = router_queries[0]
    assert batch == 65 * (nextn + 1)  # ceil(129 / 2), widened for MTP
    assert set(per_op) == {"generation_router_gemm"}
    assert all(ms > 0.0 for ms in per_op.values())
    assert total_ms == pytest.approx(sum(per_op.values()))
    layer_times = result.raw["afd_layer_measurements"]["decode"]
    assert layer_times["components"]["a_router_ms"] == pytest.approx(total_ms / layer_times["num_layers"])
    f_batch, (f_total_ms, _) = router_queries[1]
    assert f_batch == 16 * 65 * (nextn + 1)
    assert layer_times["components"]["router_ms"] == pytest.approx(f_total_ms / layer_times["num_layers"])
    assert layer_times["components"]["router_ms"] > layer_times["components"]["a_router_ms"] > 0


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


@pytest.mark.parametrize("name", list(COMPONENTS))
@pytest.mark.parametrize("value", [True, "0.002", None, float("nan"), float("inf"), -0.001])
def test_aic_adapter_rejects_malformed_components_before_fastafd(tmp_path, name, value):
    def estimator(*args, **kwargs):
        return SimpleNamespace(
            raw={
                "afd_layer_measurements": {
                    "decode": {
                        "attention_ms": 0.096,
                        "ffn_ms": 0.636,
                        "a_to_f_ms": 0.0106,
                        "f_to_a_ms": 0.0106,
                        "num_layers": LAYERS,
                        "components": {**COMPONENTS, name: value},
                    }
                }
            }
        )

    with pytest.raises(AFDInfeasible, match=f"{name} must be a finite non-negative number"):
        _provider(tmp_path, AICAFDPerformanceModel(estimator)).measure(_request(_topology()))


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


def test_prediction_config_uses_builtin_attention_with_low_bit_f_pool(tmp_path, fastafd_attention_sol):
    request, profile = _kimi_request_and_profile(tmp_path, "w4a8_mxfp4_mxfp8")
    raw = _prediction(profile)
    raw["engine"].update(model=request.model_name, context_length=2048, backend_version="0.5.14")
    raw["engine"]["afd"].update(n_f_nodes=1, f_moe_ep_size=4)
    # The independent prefill pool still serves the checkpoint at FP8.
    raw["engine"]["workers"]["prefill"]["parallelism"].update(tensor=8, moe_expert=8)
    config = CorePredictionConfig.model_validate(raw)
    for candidate in (config, CorePredictionConfig.model_validate_json(config.model_dump_json())):
        spec = prediction_to_replay_spec(candidate)
        (measurement,) = spec.backend_deployment.performance_model_metadata["afd"]["measurements"]
        provenance = measurement["provenance"]
        assert provenance["attention"]["measurement_scope"] == "attention_only"
        assert provenance["config"]["moe_quant_mode"] == "w4a8_mxfp4_mxfp8"
        assert provenance["attention"]["config"] == provenance["config"]
        assert measurement["ffn_ms"] == pytest.approx(STAGE_MS / (61 * 2))


def test_serial_decode_evaluates_the_replayed_profile_stage(tmp_path):
    raw = _prediction(_profile(tmp_path))
    raw["engine"]["afd"]["pipeline_model"] = "serial"
    config = CorePredictionConfig.model_validate(raw)
    spec = prediction_to_replay_spec(config, afd_performance_model=_Attention())

    # Exercise the runner's reconstruction of the serialized measurement and
    # topology before evaluating the complete decode step.
    (measurement,) = _afd_layer_measurements(spec.backend_deployment)
    evaluation = evaluate_afd_phase(
        _afd_topology(spec.backend_deployment),
        measurement,
        input_length=1024,
        output_length=128,
    )

    # 94 layers * 2 microbatches * 0.092 ms of A + 11.213456 ms of F.
    assert evaluation.step_latency_ms == pytest.approx(28.509456)


def _decode_reference(topology, samples, ffn_ms, output_length):
    """Use the session decode integrator on the same engine-measured A trace."""
    session = object.__new__(AFDInferenceSession)
    session._afd_config = SimpleNamespace(num_microbatches=2, pipeline_model=topology.pipeline_model.value)
    session._nextn = 0
    by_seq = {1025 + 32 * i: ms for i, (ms, _) in enumerate(samples)}
    session._sum_latency = lambda ops, **kw: (
        (by_seq[kw["seq_len"]] if ops == "A" else ffn_ms) * LAYERS,
        {},
    )
    return session._integrate_decode_phase(
        a_partition=SimpleNamespace(attn_ops="A"),
        f_partition=SimpleNamespace(ffn_ops="F"),
        a_model=None,
        f_model=None,
        runtime_config=SimpleNamespace(),
        isl=1024,
        osl=output_length,
        a_batch_size=64,
        b_batch_size=512,
        num_layers=LAYERS,
        brk_t_a_per_layer=0.0,
        brk_t_f_per_layer=0.0,
        t_a2f_layer=0.0,
        t_f2a_layer=0.0,
    )[3]


@pytest.mark.parametrize(
    "pipeline_model,expected", [("optimistic", 23.575), ("conservative", 23.575), ("serial", 37.6)]
)
def test_fastafd_crossing_trace_survives_replay_serialization(tmp_path, pipeline_model, expected):
    request = replace(_request(_topology(pipeline_model=pipeline_model)), output_length=65)
    (attention,) = _Attention().measure(request)
    components = {**COMPONENTS, "attention_compute_ms": 0.098}
    attention = replace(
        attention,
        attention_ms=0.104,
        provenance={
            "components": components,
            "decode_samples": [
                {"repeat": 32, "components": {**components, "attention_compute_ms": ms}} for ms in (0.048, 0.148)
            ],
        },
    )
    entry = {**_entry(), "latency_ms": 18.8}  # F = 18.8 / (94 * 2) = 0.10 ms.
    raw = _prediction(_profile(tmp_path, [entry]))
    raw["engine"]["afd"]["pipeline_model"] = pipeline_model
    raw["traffic"]["source"]["output_tokens"] = 65
    # Isolate service TPOT from queueing behind other decode batches.
    raw["traffic"]["load"]["concurrency"] = 1
    raw["traffic"]["stop"]["requests"] = 1
    spec = prediction_to_replay_spec(
        CorePredictionConfig.model_validate(raw),
        afd_performance_model=SimpleNamespace(measure=lambda _: (attention,)),
    )
    # JSON round trip exercises the serialized lists used in saved replay specs.
    deployment = spec.backend_deployment
    metadata = json.loads(json.dumps(deployment.performance_model_metadata))
    (measurement,) = _afd_layer_measurements(replace(deployment, performance_model_metadata=metadata))
    assert measurement.decode_attention_samples == ((0.05, 32), (0.15, 32))
    planned = AFDForegroundEngine(request.topology, (measurement,)).execute_pass(
        phase="decode", now_ms=0.0, input_length=1024, output_length=65
    )
    # For overlap, sample steps are 18.85 and 28.30 ms: mean 23.575.
    # The averaged-input maximum would incorrectly give only 18.90 ms.
    assert planned.end_ms == pytest.approx(expected)
    assert planned.intervals[-1].end_ms == pytest.approx(expected)
    assert planned.evaluation.step_latency_ms == pytest.approx(
        _decode_reference(request.topology, measurement.decode_attention_samples, 0.1, 65)
    )
    report = EngineReplayRunnerFactory().create(0).run(spec)
    assert report.metrics["mean_tpot_ms"] == pytest.approx(expected)
    assert report.metrics["completed_requests"] == 1


@pytest.mark.parametrize("provider_path", ["attention_only", "cli"])
@pytest.mark.parametrize("pipeline_model", ["optimistic", "conservative", "serial"])
def test_builtin_fastafd_integrates_native_attention_samples(
    tmp_path, fastafd_attention_sol, provider_path, pipeline_model
):
    request = replace(_request(_topology(pipeline_model=pipeline_model)), output_length=66, max_seq_len=2048)
    provider = AICAFDPerformanceModel()
    if provider_path == "cli":
        provider = AICAFDPerformanceModel(lambda *a, **kw: cli_estimate(*a, **kw, database_mode="SOL"))
    path = _profile(tmp_path)
    fastafd = _provider(tmp_path, provider, profile_path=path)
    (initial,) = fastafd.measure(request)
    samples = initial.decode_attention_samples
    assert [repeat for _, repeat in samples] == [32, 32, 1]
    lo, hi = min(ms for ms, _ in samples), max(ms for ms, _ in samples)
    assert hi > lo  # Real Rust SOL attention increases with KV length.
    ffn_ms = (lo + hi) / 2
    payload = json.loads(Path(path).read_text())
    payload["entries"][0]["latency_ms"] = ffn_ms * LAYERS * 2
    Path(path).write_text(json.dumps(payload))
    (measurement,) = fastafd.measure(request)
    evaluation = evaluate_afd_phase(request.topology, measurement, input_length=1024, output_length=66)
    assert evaluation.step_latency_ms == pytest.approx(
        _decode_reference(request.topology, samples, ffn_ms, 66), rel=1e-12
    )
    if pipeline_model != "serial":
        averaged = evaluate_afd_phase(
            request.topology, replace(measurement, decode_attention_samples=()), input_length=1024, output_length=66
        )
        assert evaluation.step_latency_ms > averaged.step_latency_ms


@pytest.mark.parametrize(
    "trace,match",
    [
        ([], "non-empty sequence"),
        ([{"repeat": 0, "components": COMPONENTS}], "repeat must be a positive integer"),
        ([{"repeat": 1, "components": {**COMPONENTS, "attention_compute_ms": float("nan")}}], "finite"),
        ([{"repeat": 1, "components": {**COMPONENTS, "a_router_ms": 0}}], "positive A-side router"),
        ([{"repeat": 1, "components": {**COMPONENTS, "attention_compute_ms": 0.5}}], "does not reconcile"),
    ],
)
def test_fastafd_rejects_invalid_or_inconsistent_sample_traces(tmp_path, trace, match):
    (attention,) = _Attention().measure(_request(_topology()))
    attention = replace(attention, provenance={**attention.provenance, "decode_samples": trace})
    with pytest.raises(AFDInfeasible, match=match):
        _provider(tmp_path, SimpleNamespace(measure=lambda _: (attention,))).measure(_request(_topology()))


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
        {"nextn": 1, "nextn_accepted": 1},
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


@pytest.mark.parametrize("factor", [0.5, 2.0, 100.0, 1.0000000001])
def test_prediction_config_rejects_non_unit_fastafd_communication_calibration(tmp_path, factor):
    raw = _prediction(_profile(tmp_path))
    raw["engine"]["afd"]["comm_overhead_factor"] = factor
    with pytest.raises(ValidationError, match="fastafd_profile_path requires comm_overhead_factor=1"):
        CorePredictionConfig.model_validate(raw)


def test_modeled_afd_still_accepts_communication_calibration(tmp_path):
    raw = _prediction(_profile(tmp_path))
    raw["engine"]["afd"].update(fastafd_profile_path=None, fastafd_moe_backend=None, comm_overhead_factor=2.0)
    config = CorePredictionConfig.model_validate(raw)
    assert config.engine.afd.comm_overhead_factor == 2.0


def test_unit_fastafd_communication_calibration_survives_config_round_trip(tmp_path):
    raw = _prediction(_profile(tmp_path))
    raw["engine"]["afd"]["comm_overhead_factor"] = 1.0
    config = CorePredictionConfig.model_validate(raw)
    restored = CorePredictionConfig.model_validate_json(config.model_dump_json())
    attention = _Attention()
    spec = prediction_to_replay_spec(restored, afd_performance_model=attention)
    assert attention.requests[0].topology.comm_overhead_factor == 1.0
    (measurement,) = spec.backend_deployment.performance_model_metadata["afd"]["measurements"]
    assert measurement["ffn_ms"] == pytest.approx(STAGE_MS / (LAYERS * 2))


def test_canonical_stage_api_parquet_json_and_saved_config(tmp_path, monkeypatch):
    """Native measurements agree across raw imports, parquet, and saved identity."""
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel
    from aisimulate_core.sdk.fastafd_profile import FastAFDMoEStageProfile

    original = _profile(tmp_path, [_entry(), _entry(moe_precision="fp8"), _entry(routed_topk=4)])
    profile = FastAFDMoEStageProfile.load(original)
    parquet = tmp_path / "stage.parquet"
    profile.write_parquet(parquet)
    assert FastAFDMoEStageProfile.load(parquet).entries == profile.entries

    def no_python_table(*args, **kwargs):
        raise AssertionError("production stage timing must use the canonical Rust table")

    monkeypatch.setattr(FastAFDMoEStageProfile, "load", no_python_table)
    for path in (original, parquet):
        (measurement,) = _provider(tmp_path, profile_path=path, moe_precision="fp8").measure(_request(_topology()))
        saved = measurement.provenance["config"]
        assert saved["estimation_mode"] == "fastafd_stage"
        assert saved["fallback_policy"] == "deny"
        assert saved["afd"]["attention_gpus"] == 64
        assert saved["afd"]["routed_topk"] == 8
        assert saved["backend_version"] == "0.5.14"
        assert saved["estimator_config"]["fastafd_stage"]["profile_path"] == str(path)
        for config in (saved, ForwardPassPerfModelConfig(**saved)):
            restored = RustForwardPassPerfModel.best_available(config)
            stage = restored.estimate_fastafd_stage(128)
            assert stage["ffn_ms"] == measurement.ffn_ms
            assert stage["stage"] == measurement.provenance["stage"]
            assert restored.diagnostics()["source"] == "fastafd_stage"
            with pytest.raises(ValueError, match="no FastAFD split MoE stage measurement"):
                restored.estimate_fastafd_stage(127)
            with pytest.raises(ValueError, match="not a whole forward pass"):
                restored.estimate_forward_pass_time_ms([])


def test_parquet_precision_ambiguity_and_corruption_fail_closed(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisimulate_core.sdk.fastafd_profile import FastAFDMoEStageProfile

    profile = FastAFDMoEStageProfile.load(_profile(tmp_path, [_entry(), _entry(moe_precision="fp8")]))
    parquet = tmp_path / "stage.parquet"
    profile.write_parquet(parquet)
    with pytest.raises(AFDInfeasible, match="set engine.moe_quant_mode"):
        _provider(tmp_path, profile_path=parquet).measure(_request(_topology()))
    rows = pq.read_table(parquet).to_pylist()
    capture = json.loads(rows[0]["capture_json"])
    rows[0]["capture_json"] = json.dumps({**capture, "key": _entry()["key"]})
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    with pytest.raises(ValueError, match="parquet capture must contain exactly"):
        FastAFDMoEStageProfile.load(parquet)
    with pytest.raises(AFDInfeasible, match="unknown field"):
        _provider(tmp_path, profile_path=parquet).measure(_request(_topology()))
    rows[0]["capture_json"] = json.dumps(capture)
    rows[0]["latency_ms"] = -1.0
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    with pytest.raises(AFDInfeasible, match="unqualified FastAFD measurement"):
        _provider(tmp_path, profile_path=parquet).measure(_request(_topology()))


def test_native_stage_query_pins_profile_until_reconstruction(tmp_path):
    from aisimulate_core.sdk import RustForwardPassPerfModel

    path = _profile(tmp_path)
    (measurement,) = _provider(tmp_path, profile_path=path).measure(_request(_topology()))
    saved = measurement.provenance["config"]
    model = RustForwardPassPerfModel.best_available(saved)
    original = model.estimate_fastafd_stage(128)
    payload = json.loads(Path(path).read_text())
    payload["entries"][0]["latency_ms"] = 2 * STAGE_MS
    with open(path, "w") as stream:
        json.dump(payload, stream)
    assert model.estimate_fastafd_stage(128) == original
    refreshed = RustForwardPassPerfModel.best_available(saved).estimate_fastafd_stage(128)
    assert refreshed["ffn_ms"] == 2 * original["ffn_ms"]
    assert refreshed["stage"]["profile_sha256"] != original["stage"]["profile_sha256"]


def test_canonical_stage_pins_batch_and_resolved_precision(tmp_path):
    from aisimulate_core.sdk import RustForwardPassPerfModel

    other = {**_entry(logical_batch_per_source_rank=64, moe_precision="fp8"), "latency_ms": 2 * STAGE_MS}
    path = _profile(tmp_path, [_entry(), other])
    (measurement,) = _provider(tmp_path, profile_path=path).measure(_request(_topology()))
    saved = measurement.provenance["config"]
    assert saved["afd"]["logical_batch_per_source_rank"] == 128
    assert saved["moe_quant_mode"] == "w4a8_mxfp4_mxfp8"
    model = RustForwardPassPerfModel.best_available(saved)
    with pytest.raises(ValueError, match="no FastAFD split MoE stage measurement"):
        model.estimate_fastafd_stage(64)
    # A separate request can still select the other batch's sole precision,
    # preserving the legacy provider's exact per-request selection policy.
    (smaller,) = _provider(tmp_path, profile_path=path).measure(_request(_topology(a_batch_size=64)))
    assert smaller.provenance["config"]["moe_quant_mode"] == "fp8"
    assert smaller.ffn_ms == 2 * measurement.ffn_ms
