# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public grouped-cache workflows with fictional resources and synthetic timings."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aisim import main as cli
from aisim.capacity import materialize_aic_num_gpu_blocks
from aisim.compiler import prediction_to_replay_spec
from aisim.config import CorePredictionConfig, CoreRecommendationConfig
from aisim.support.plan import create_plan
from aisim.support.schema import SupportRequest
from aisim.sweeper.config import Workload
from aisim.sweeper.kv_load import resolve_kv_load
from aisim.sweeper.model_hw import parallel_configs_for
from aisim_core.ais_perf_model import AisPerfModelDeprecationWarning

pytestmark = pytest.mark.unit

# AISim still writes the old estimator fields into the configs it saves, so reading one back warns.
reads_saved_config = pytest.mark.filterwarnings("ignore::aisim_core.ais_perf_model.AisPerfModelDeprecationWarning")


@pytest.fixture
def grouped_case(tmp_path, monkeypatch):
    import aisim_core
    from aisim_core.sdk import engine, memory

    def no_graph(*_args, **_kwargs):
        pytest.fail("grouped FPM workflow constructed an analytical model")

    monkeypatch.setattr(engine, "get_model", no_graph)
    monkeypatch.setattr(engine, "build_model_config", no_graph)
    monkeypatch.setattr(memory.KVCacheEstimator, "from_request", no_graph)
    monkeypatch.setattr(memory.NaiveKVCacheEstimator, "from_model_path", no_graph)
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    resources = {
        "weights_bytes": 100,
        "activations_bytes": 20,
        "runtime_overhead_bytes": 30,
        "comm_overhead_bytes": 50,
        "cache_layout": "grouped",
        "cache_groups": [
            {
                "name": "full",
                "kind": "attention",
                "num_layers": 1,
                "block_size_tokens": 64,
                "page_size_bytes": 64,
            },
            {
                "name": "window",
                "kind": "attention",
                "num_layers": 2,
                "block_size_tokens": 16,
                "page_size_bytes": 16,
                "sliding_window": 512,
            },
            {
                "name": "conv",
                "kind": "convolution",
                "num_layers": 3,
                "block_size_tokens": 4,
                "page_size_bytes": 8,
                "sliding_window": 4,
            },
        ],
        "max_num_tokens": 128,
        "max_batch_size": 8,
        "provenance": "Hand-sized fake cache pages including aggregate group layers; not measured GPU resources.",
    }
    deployment = {
        "system": "h200_sxm",
        "backend": "vllm",
        "backend_version": "0.25.1",
        "gemm_quant_mode": "fp8",
        "moe_quant_mode": "fp8",
        "fmha_quant_mode": "bfloat16",
        "comm_quant_mode": "half",
        "kv_cache_dtype": "fp8",
        "resources": resources,
    }
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"architectures": ["UnregisteredWindowMoeForCausalLM"]}))
    profile = {
        "schema_version": 1,
        "model": str(checkpoint),
        "model_revision": "synthetic-v1",
        "architecture": "UnregisteredWindowMoeForCausalLM",
        "context_length": 2048,
        "num_experts": 4,
        "provenance": "Fictional model and timings; functional coverage only, no silicon accuracy claim.",
        "deployments": [
            {**deepcopy(deployment), "tp": 2, "dp": 1, "moe_tp": 2, "moe_ep": 1},
            {**deepcopy(deployment), "tp": 1, "dp": 2, "moe_tp": 1, "moe_ep": 2},
        ],
    }
    root = tmp_path / "systems"
    root.mkdir()
    system = yaml.safe_load((Path(aisim_core.__file__).parent / "systems/h200_sxm.yaml").read_text())
    # A deliberately tiny fake GPU creates meaningful byte pressure in CPU tests.
    system["gpu"]["mem_capacity"] = 4096
    (root / "h200_sxm.yaml").write_text(yaml.safe_dump(system))
    _write_timings(root, profile)
    return profile, root


def _write_timings(
    root,
    profile,
    *,
    decode_ceiling=8192,
    version="0.25.1",
    prefill_tokens=(128, 256, 512, 1024),
    prefill_kvs=(0, 1, 512, 1024, 2048, 8192),
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisim_core.sdk.fpm_profile import load_fpm_profile

    rows = []
    for index, deployment in enumerate(load_fpm_profile(profile).deployments):
        identity = deployment.model_dump(mode="json", exclude={"resources"})
        for batch in (1, 2, 4, 8):
            for phase, tokens, kvs in (
                ("prefill", (batch, *prefill_tokens), prefill_kvs),
                ("decode", (0,), (0, 1, 512, decode_ceiling)),
            ):
                for new_tokens in sorted(set(tokens)):
                    for kv in sorted(set(kvs)):
                        rows.append(
                            {
                                **identity,
                                "cell_id": f"synthetic-{index}-{phase}-{batch}-{new_tokens}-{kv}",
                                "model_path": profile["model"],
                                "weight_quantization": "synthetic",
                                "workload_kind": phase,
                                "partition_policy": "balanced_v1",
                                "batch_size": batch,
                                "total_prefill_tokens": new_tokens,
                                "total_kv_read_tokens": kv,
                                "latency_ms": 1 + batch + new_tokens / 128 + kv / 1024,
                                "kv_seed_regime": "real_kv",
                            }
                        )
    path = root / f"data/h200_sxm/vllm/{version}/fpm_forward_perf.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "schema_name": "aic_fpm_forward_perf",
                "schema_version": 6,
                "coordinate_system": "iteration_totals_balanced_v1",
                "measurement_policy": "dynamo_native_single_sample_v1",
                "system": "h200_sxm",
                "backend": "vllm",
                "backend_version": version,
                "row_count": len(rows),
                "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    )


def _prediction(profile, root, *, dep=False):
    return {
        "engine": {
            "mode": "aggregated",
            "model": profile["model"],
            "fpm_profile": profile,
            "hardware": "h200_sxm",
            "backend": "vllm",
            "backend_version": "0.25.1",
            "context_length": 2048,
            "systems_paths": [str(root)],
            "workers": {
                "aggregated": {
                    "parallelism": {
                        "tensor": 1 if dep else 2,
                        "attention_data": 2 if dep else 1,
                        "moe_tensor": 1 if dep else 2,
                        "moe_expert": 2 if dep else 1,
                    },
                    "scheduler": {"max_batched_tokens": 128, "max_sequences": 4},
                    "kv_cache": {"prefix_caching": False},
                    "timing": {
                        "estimation_mode": "fpm_interpolation",
                        "fallback_policy": "deny",
                        "estimator_config": {
                            "fpm_interpolation": {
                                "method": "direct",
                                "collect_coverage": True,
                            }
                        },
                    },
                }
            },
        },
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 1152, "output_tokens": 4},
            "load": {"type": "concurrency", "concurrency": 2},
            "stop": {"requests": 2},
        },
    }


def _mixed_profile(profile):
    profile = deepcopy(profile)
    resources = profile["deployments"][0]["resources"]
    resources["cache_layout"] = "linear"
    resources["kv_bytes_per_token"] = 1
    resources.pop("cache_groups")
    # A full-only grouped deployment is valid beside a linear deployment.
    profile["deployments"][1]["resources"]["cache_groups"] = profile["deployments"][1]["resources"]["cache_groups"][:1]
    return profile


# Two pool blocks of 64 bytes for every running request, whatever its length.
_STATE_GROUP = {
    "name": "state",
    "kind": "state",
    "num_layers": 2,
    "block_size_tokens": 64,
    "page_size_bytes": 64,
    "blocks_per_request": 2,
}


def _with_state_group(profile):
    profile = deepcopy(profile)
    for deployment in profile["deployments"]:
        deployment["resources"]["cache_groups"].append(dict(_STATE_GROUP))
    return profile


# A convolution group as the probe import writes one for Inkling: a 4-token window, marked cacheable. Its block and
# page fit _with_prefix_evidence.
_CONVOLUTION_GROUP = {
    "name": "conv",
    "kind": "convolution",
    "num_layers": 3,
    "block_size_tokens": 64,
    "page_size_bytes": 64,
    "sliding_window": 4,
    "prefix_cacheable": True,
}


def _with_prefix_evidence(profile, *, hash_block=64, version="0.31.0"):
    """Runtime memory of a hand-made profile with prefix caching on, and every group marked cacheable.

    The Rust gate needs one page size, a hash block that divides every cacheable block and one running block
    per state group, so every page is 64 bytes and every block is one hash block. Grouped prefix caching refuses
    convolution groups, so the fictional convolution group goes.
    """
    profile = deepcopy(profile)
    for deployment in profile["deployments"]:
        deployment["backend_version"] = version
        resources = deployment["resources"]
        resources["cache_groups"] = [group for group in resources["cache_groups"] if group["kind"] != "convolution"]
        for name in ("weights_bytes", "activations_bytes", "runtime_overhead_bytes", "comm_overhead_bytes"):
            resources.pop(name)
        # Runtime memory requires the exact recorded scheduler envelope of _prediction.
        resources["max_batch_size"] = 4
        resources["runtime_memory"] = {
            "kv_cache_bytes": 3456,
            "gpu_memory_utilization": 0.9,
            # The Rust gate admits this pool at 1024 tokens, as vLLM's startup check would.
            "max_model_len": 1024,
            "provenance": "Synthetic pool of 54 blocks of 64 bytes from a fictional probe. No GPU qualification.",
            "enable_prefix_caching": True,
            "mamba_cache_mode": "align",
            "hash_block_size_tokens": hash_block,
        }
        for group in resources["cache_groups"]:
            group.update(block_size_tokens=hash_block, page_size_bytes=64, prefix_cacheable=True)
            if group["kind"] == "state":
                group["blocks_per_request"] = 1
                if version in ("0.32.0", "0.33.0"):
                    group["prefill_checkpoint_blocks"] = 0
    return profile


def _prefix_caching_prediction(profile, root, version):
    """Two identical 520-token requests in sequence on a grouped rank with prefix caching."""
    raw = _prediction(profile, root)
    raw["engine"]["backend_version"] = version
    # The rank runs at the probe's max_model_len. The Rust gate needs a pool for one prompt of max_model_len plus
    # the state blocks of one request: 16 + 13 + 2 = 31 of the 54 blocks at 1024 tokens.
    raw["engine"]["context_length"] = profile["deployments"][0]["resources"]["runtime_memory"]["max_model_len"]
    raw["engine"]["workers"]["aggregated"]["kv_cache"]["prefix_caching"] = True
    raw["traffic"] = {
        "source": {"type": "synthetic", "input_tokens": 520, "output_tokens": 4, "cached_prefix_tokens": 520},
        "load": {"type": "concurrency", "concurrency": 1},
        "stop": {"requests": 2},
    }
    return raw


_PD_PAIRING = (
    "grouped FPM cache P/D requires prefill and decode deployments with equal tensor parallelism and identical "
    "cache groups"
)
_PD_MARKERS = (
    "grouped FPM cache P/D requires every cache group to declare prefix_cacheable and transfer_bytes_per_block; "
    "set both on each cache group of the profile, with the transfer_bytes_per_block from evidence for each group, "
    "for the deployment's vLLM version"
)
_PD_VERSION = "grouped FPM cache P/D is modeled for vLLM 0.31.0, 0.32.0 and 0.33.0 only"
_PD_INTERVAL = (
    "grouped FPM cache P/D on vLLM 0.33.0 requires prefill_schedule_interval=1; vLLM 0.33.0 holds prefills at "
    "attention DP size 1 too, and replay does not model that"
)
_PD_TRANSFER = "grouped FPM cache derives kv_transfer.bytes_per_token from its cache groups; set it to auto"
_PD_REPLAY_WINDOW = (
    "kv_cache_groups with worker_type=prefill or decode do not model a sliding-window attention group with "
    "prefix_cacheable=false; describe a ring buffer as a state group with blocks_per_request: 1 and "
    "prefix_cacheable: false, and replay a bounded replay window on an aggregated rank with prefix caching off"
)


def _with_transfer_markers(profile, *, version="0.31.0"):
    """The P/D markers of a hand-made profile on every grouped deployment, with the page as the transfer bytes."""
    profile = deepcopy(profile)
    for deployment in profile["deployments"]:
        deployment["backend_version"] = version
        for group in deployment["resources"].get("cache_groups", []):
            group.setdefault("prefix_cacheable", True)
            group["transfer_bytes_per_block"] = group["page_size_bytes"]
    return profile


def _pd_prediction(profile, root, *, bandwidth=None):
    """A prefill and a decode worker of tensor 2 on the profile's vLLM version, with a KV transfer at a bandwidth."""
    raw = _prediction(profile, root)
    engine = raw["engine"]
    engine["mode"] = "disaggregated"
    engine["backend_version"] = profile["deployments"][0]["backend_version"]
    worker = engine["workers"].pop("aggregated")
    engine["workers"] = {"prefill": deepcopy(worker), "decode": worker}
    if bandwidth is not None:
        engine["kv_transfer"] = {"bytes_per_token": "auto", "bandwidth_gb_per_second": bandwidth}
    return raw


def _pd_search(profile, root):
    """The search of _pd_prediction: disaggregated workers without prefix caching."""
    return {
        "deployment_mode": ["disagg"],
        "backend": ["vllm"],
        "backend_version": profile["deployments"][0]["backend_version"],
        "model_name": profile["model"],
        "hardware_sku": "h200_sxm",
        "fpm_profile": profile,
        "systems_paths": [str(root)],
        "gpu_budget": 4,
        "prefill_max_num_batched_tokens": [128],
        "prefill_max_num_seqs": [4],
        "prefill_enable_prefix_caching": False,
        "decode_max_num_batched_tokens": [128],
        "decode_max_num_seqs": [4],
    }


@pytest.mark.parametrize("unrelated", ["topology", "hardware", "version"])
def test_mixed_profile_predict_preserves_selected_linear_prefix_cache(grouped_case, tmp_path, unrelated):
    profile, root = grouped_case
    profile = _mixed_profile(profile)
    if unrelated != "topology":
        profile["deployments"][1].update(tp=2, dp=1, moe_tp=2, moe_ep=1)
        profile["deployments"][1].update(
            {"system": "h100_sxm"} if unrelated == "hardware" else {"backend_version": "0.25.2"}
        )
    raw = _prediction(profile, root)
    raw["engine"]["workers"]["aggregated"]["kv_cache"]["prefix_caching"] = True
    CorePredictionConfig.model_validate(raw)
    path = tmp_path / "linear.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "linear-prediction"
    assert cli.main(["predict", "-c", str(path), "--output-dir", str(output), "--detail", "memory"]) == 0
    report = json.loads((output / "prediction.json").read_text())
    assert report["completed_requests"] == 2
    assert report["details"]["sections"]["memory"]["roles"]["aggregated"]["total_kv_size_tokens"] > 0


@pytest.mark.parametrize("preset", [False, True])
def test_mixed_profile_recommendation_preserves_pinned_linear_prefix_cache(grouped_case, tmp_path, monkeypatch, preset):
    from aisim.recommend import _run_recommendation, recommendation_to_sweeper
    from aisim.sweeper.search_space import enumerate_branches

    profile, root = grouped_case
    raw = _prediction(_mixed_profile(profile), root)
    worker = raw["engine"]["workers"]["aggregated"]
    worker["kv_cache"]["prefix_caching"] = True
    worker["parallelism"]["preset"] = False
    if preset:
        worker["parallelism"] = {"preset": [{**worker["parallelism"], "replicas": 1, "pipeline": 1}]}
        worker["parallelism"]["preset"][0].pop("preset")
    raw["optimization"] = {"constraints": {"max_candidate_gpus": 2}}
    raw["optimizer"] = {"algorithm": "random", "max_trials": 1, "parallelism": 1}
    smart = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw))
    branches = enumerate_branches(smart)
    assert len(branches) == 1
    assert {candidate.shape.tp for candidate in branches[0].parallel_configs} == {2}
    monkeypatch.setattr("aisim.recommend.run_recommendation", _run_recommendation)
    path = tmp_path / "linear-recommend.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "linear-recommendation"
    assert cli.main(["recommend", "-c", str(path), "--output-dir", str(output)]) == 0
    with pytest.warns(AisPerfModelDeprecationWarning):
        exported = CorePredictionConfig.from_yaml(output / "recommendations/0001.yaml")
    assert exported.engine.workers.aggregated.parallelism.tensor == 2
    assert exported.engine.workers.aggregated.kv_cache.prefix_caching is True


def _mixed_search(profile, root):
    return {
        "deployment_mode": ["agg"],
        "backend": ["vllm"],
        "backend_version": "0.25.1",
        "model_name": profile["model"],
        "hardware_sku": "h200_sxm",
        "fpm_profile": _mixed_profile(profile),
        "systems_paths": [str(root)],
        "gpu_budget": 2,
        "agg_max_num_batched_tokens": [128],
        "agg_max_num_seqs": [4],
        "agg_enable_prefix_caching": True,
    }


@pytest.mark.parametrize("filter_by", ["hardware", "version", "budget", "pin", "mode_pin", "choices", "log_range"])
def test_mixed_search_ignores_unreachable_grouped_deployments(grouped_case, filter_by):
    from aisim.sweeper.config import SmartSearchConfig
    from aisim.sweeper.search_space import enumerate_branches

    profile, root = grouped_case
    raw = _mixed_search(profile, root)
    grouped = raw["fpm_profile"]["deployments"][1]
    if filter_by == "hardware":
        grouped["system"] = "h100_sxm"
    elif filter_by == "version":
        grouped["backend_version"] = "0.25.2"
        raw["backend_version"] = {"vllm": " 0.25.1 "}
    elif filter_by == "budget":
        grouped.update(tp=4, dp=1, moe_tp=1, moe_ep=4)
    elif filter_by in {"pin", "mode_pin"}:
        pinned = [{"tp": 2, "moe_tp": 2}]
        raw["parallel_configs" if filter_by == "pin" else "parallel_configs_by_mode"] = (
            pinned if filter_by == "pin" else {"agg": pinned}
        )
    elif filter_by == "choices":
        raw["parallel_independent_by_mode"] = {"agg": {"tp": [2]}}
    else:
        raw["parallel_independent_by_mode"] = {"agg": {"tp": None}}
        raw["parallel_independent_log_ranges_by_mode"] = {"agg": {"tp": [2, 4]}}
    config = SmartSearchConfig(search_space=raw, workload=Workload(isl=1152, osl=4, concurrency=2, request_count=2))
    branches = enumerate_branches(config)
    assert len(branches) == 1
    assert {candidate.shape.tp for candidate in branches[0].parallel_configs} == {2}


@pytest.mark.parametrize("placement", ["aggregated", "disaggregated", "heterogeneous"])
def test_mixed_recommendation_linear_fixed_capacity_and_disaggregation(grouped_case, placement):
    from aisim.recommend import recommendation_to_sweeper
    from aisim.sweeper.search_space import enumerate_branches

    profile, root = grouped_case
    raw = _prediction(_mixed_profile(profile), root)
    # Unselected deployments must not reach resource-envelope validation either.
    raw["engine"]["fpm_profile"]["deployments"][1]["resources"]["max_num_tokens"] = 64
    worker = raw["engine"]["workers"]["aggregated"]
    worker["kv_cache"].update(prefix_caching=True, capacity={"type": "fixed", "blocks": 64})
    disaggregated = placement != "aggregated"
    if disaggregated:
        raw["engine"]["mode"] = "disaggregated"
        raw["engine"]["workers"] = {"prefill": deepcopy(worker), "decode": deepcopy(worker)}
    if placement == "heterogeneous":
        linear = deepcopy(raw["engine"]["fpm_profile"]["deployments"][0])
        linear["system"] = "h100_sxm"
        raw["engine"]["fpm_profile"]["deployments"].append(linear)
        raw["engine"]["workers"]["decode"]["hardware"] = "h100_sxm"
        (root / "h100_sxm.yaml").write_bytes((root / "h200_sxm.yaml").read_bytes())
    CorePredictionConfig.model_validate(raw)
    for worker in raw["engine"]["workers"].values():
        worker["parallelism"]["preset"] = False
        worker["parallelism"]["replicas"] = 1
    if placement == "heterogeneous":
        parallel = raw["engine"]["workers"]["prefill"]["parallelism"]
        parallel.pop("preset")
        raw["engine"]["workers"]["prefill"]["parallelism"] = {"preset": [{**parallel, "pipeline": 1}]}
    raw["optimization"] = {"constraints": {"max_candidate_gpus": 4 if disaggregated else 2}}
    branches = enumerate_branches(recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw)))
    assert len(branches) == 1
    assert len(branches[0].parallel_configs) == 1
    candidate = branches[0].parallel_configs[0]
    assert (candidate.prefill.shape.tp if disaggregated else candidate.shape.tp) == 2


def test_mixed_search_keeps_both_layouts_when_grouped_settings_are_supported(grouped_case):
    from aisim.sweeper.config import SmartSearchConfig
    from aisim.sweeper.search_space import enumerate_branches

    profile, root = grouped_case
    raw = _mixed_search(profile, root)
    raw["agg_enable_prefix_caching"] = False
    config = SmartSearchConfig(search_space=raw, workload=Workload(isl=1152, osl=4, concurrency=2, request_count=2))
    assert {candidate.shape.tp for candidate in enumerate_branches(config)[0].parallel_configs} == {1, 2}


@pytest.mark.parametrize(
    "update,error",
    [
        ({}, "prefix_caching=false"),
        ({"agg_enable_prefix_caching": False, "agg_native_host_offload": {"num_host_blocks": 64}}, "only HBM"),
        ({"agg_enable_prefix_caching": False, "agg_num_gpu_blocks": 64}, "not scalar capacity"),
        ({"agg_enable_prefix_caching": False, "agg_kv_bytes_per_token": 1}, "not scalar capacity"),
        (
            {
                "agg_enable_prefix_caching": False,
                "speculation": {"kind": "ngram", "num_speculative_tokens": 1, "acceptance_rates": [0.5]},
            },
            "grouped FPM cache requires vLLM without speculative decoding",
        ),
        # The fixture's vLLM 0.25.1 is no P/D version. A prefill or decode rank with prefix caching on gets that text
        # first, as with prefix caching off below, and not the remedy of turning prefix caching off.
        ({"deployment_mode": ["disagg"], "gpu_budget": 4}, _PD_VERSION),
        (
            {
                "deployment_mode": ["disagg"],
                "gpu_budget": 4,
                "prefill_enable_prefix_caching": False,
                "decode_native_host_offload": {"num_host_blocks": 64},
            },
            "only HBM",
        ),
        (
            {
                "deployment_mode": ["disagg"],
                "gpu_budget": 4,
                "prefill_enable_prefix_caching": False,
                "decode_num_gpu_blocks": 64,
            },
            "not scalar capacity",
        ),
        ({"deployment_mode": ["disagg"], "gpu_budget": 4, "prefill_enable_prefix_caching": False}, _PD_VERSION),
    ],
)
def test_mixed_search_rejects_reachable_grouped_unsupported_settings(grouped_case, update, error):
    from aisim.sweeper.config import SearchSpace

    profile, root = grouped_case
    raw = _mixed_search(profile, root)
    raw.update(update)
    with pytest.raises(ValidationError, match=error):
        SearchSpace.model_validate(raw)


def test_public_recommendation_rejects_reachable_grouped_before_estimating(grouped_case, tmp_path, monkeypatch, capsys):
    from aisim.recommend import _run_recommendation
    from aisim_core.sdk import RustForwardPassPerfModel

    def no_estimator(*_args, **_kwargs):
        pytest.fail("unsupported grouped recommendation reached estimator construction")

    profile, root = grouped_case
    raw = _prediction(_mixed_profile(profile), root)
    raw["engine"]["workers"]["aggregated"]["parallelism"] = {"preset": "default"}
    raw["engine"]["workers"]["aggregated"]["kv_cache"]["prefix_caching"] = True
    raw["optimization"] = {"constraints": {"max_candidate_gpus": 2}}
    raw["optimizer"] = {"algorithm": "random", "max_trials": 1, "parallelism": 1}
    CoreRecommendationConfig.model_validate(raw)
    monkeypatch.setattr("aisim.recommend.run_recommendation", _run_recommendation)
    monkeypatch.setattr(RustForwardPassPerfModel, "best_available", no_estimator)
    path = tmp_path / "unsupported-recommend.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(SystemExit) as error:
        cli.main(["recommend", "-c", str(path), "--output-dir", str(tmp_path / "unsupported")])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert "grouped FPM cache requires agg_enable_prefix_caching=false" in output.out + output.err


@pytest.mark.parametrize("dep", [False, True])
@pytest.mark.parametrize("layout", ["mixed", "window_only", "full_only"])
def test_public_predict_uses_group_bytes_with_tp_and_dep(grouped_case, tmp_path, dep, layout):
    profile, root = grouped_case
    if layout != "mixed":
        for deployment in profile["deployments"]:
            groups = deployment["resources"]["cache_groups"]
            deployment["resources"]["cache_groups"] = groups[:1] if layout == "full_only" else groups[1:]
    raw = _prediction(profile, root, dep=dep)
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    diagnostics = {}
    lowered = materialize_aic_num_gpu_blocks(spec.backend_deployment.agg_engine_args, memory_diagnostics=diagnostics)
    assert "num_gpu_blocks" not in lowered
    assert lowered["kv_cache_capacity_bytes"] == 3486  # floor(4096 * .9 - 200)
    assert lowered["kv_cache_groups"] == diagnostics["cache_groups"]
    assert diagnostics["total_kv_size_tokens"] is None
    path = tmp_path / "predict.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "prediction"
    assert (
        cli.main(
            [
                "predict",
                "-c",
                str(path),
                "--output-dir",
                str(output),
                "--detail",
                "memory",
                "--capture-per-request",
            ]
        )
        == 0
    )
    report = json.loads((output / "prediction.json").read_text())
    assert report["completed_requests"] == 2
    assert all(row["output_length"] == 4 for row in report["per_request"])
    assert report["fpm_query_coverage"]["status"] == "covered"
    memory = report["details"]["sections"]["memory"]["roles"]["aggregated"]
    assert memory["total_kv_size_bytes"] == 3486
    assert memory["total_kv_size_tokens"] is None
    assert "estimated_num_gpu_blocks" not in memory


@pytest.mark.parametrize("dep", [False, True])
def test_public_predict_runs_with_a_state_group(grouped_case, tmp_path, dep):
    profile, root = grouped_case
    raw = _prediction(_with_state_group(profile), root, dep=dep)
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    diagnostics = {}
    lowered = materialize_aic_num_gpu_blocks(spec.backend_deployment.agg_engine_args, memory_diagnostics=diagnostics)
    # The state group reaches the engine configuration with its block count. No other group has the key.
    assert lowered["kv_cache_groups"] == diagnostics["cache_groups"]
    groups = {group["name"]: group for group in lowered["kv_cache_groups"]}
    assert groups["state"] == {**_STATE_GROUP, "sliding_window": None}
    assert all("blocks_per_request" not in group for name, group in groups.items() if name != "state")
    assert lowered["kv_cache_capacity_bytes"] == 3486
    # 2048 full-history + 656 sliding-window + 272 convolution + 128 state bytes (2 blocks of 64).
    assert diagnostics["request_peak_cache_bytes"] == 3104
    path = tmp_path / "predict.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "prediction"
    arguments = ["predict", "-c", str(path), "--output-dir", str(output), "--detail", "memory", "--capture-per-request"]
    assert cli.main(arguments) == 0
    report = json.loads((output / "prediction.json").read_text())
    assert report["completed_requests"] == 2
    assert all(row["output_length"] == 4 for row in report["per_request"])
    assert report["fpm_query_coverage"]["status"] == "covered"
    memory = report["details"]["sections"]["memory"]["roles"]["aggregated"]
    assert memory["total_kv_size_bytes"] == 3486
    assert memory["total_kv_size_tokens"] is None


def test_public_predict_does_not_clamp_logical_context_to_window(grouped_case, tmp_path):
    profile, root = grouped_case
    # Window-only allocation is bounded, but the first decode after the
    # prefill-emitted token must read all 1152 prompt KV tokens. A table ending
    # at 512 must fail, not shrink the logical context or physical byte budget.
    for deployment in profile["deployments"]:
        deployment["resources"]["cache_groups"] = deployment["resources"]["cache_groups"][1:]
    _write_timings(root, profile, decode_ceiling=512)
    raw = _prediction(profile, root)
    raw["traffic"]["load"]["concurrency"] = 1
    raw["traffic"]["stop"]["requests"] = 1
    path = tmp_path / "uncovered.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "uncovered"
    assert cli.main(["predict", "-c", str(path), "--output-dir", str(output)]) != 0
    coverage = json.loads((output / "fpm-coverage.json").read_text())
    gaps = coverage["roles"][0]["coverage"]["gaps"]
    assert any(gap["coordinates"]["total_kv_read_tokens"] == 1152 for gap in gaps)


@reads_saved_config
@pytest.mark.parametrize("relative_load", [False, True])
@pytest.mark.parametrize("observed_memory", [False, True])
def test_grouped_recommendation_exports_a_runnable_prediction(
    grouped_case, tmp_path, monkeypatch, relative_load, observed_memory
):
    from aisim.recommend import _run_recommendation

    monkeypatch.setattr("aisim.recommend.run_recommendation", _run_recommendation)
    profile, root = grouped_case
    if observed_memory:
        for deployment in profile["deployments"]:
            resources = deployment["resources"]
            for name in ("weights_bytes", "activations_bytes", "runtime_overhead_bytes", "comm_overhead_bytes"):
                resources.pop(name)
            resources["max_batch_size"] = 4
            resources["runtime_memory"] = {
                "kv_cache_bytes": 3486,
                "max_model_len": 2048,
                "gpu_memory_utilization": 0.9,
                "provenance": "Synthetic initialized cache pool; no GPU qualification.",
            }
    raw = _prediction(profile, root)
    raw["engine"]["workers"]["aggregated"]["parallelism"] = {"preset": "default"}
    raw["optimization"] = {"constraints": {"max_candidate_gpus": 2}}
    raw["optimizer"] = {"algorithm": "random", "max_trials": 2, "parallelism": 1}
    if relative_load:
        raw["traffic"]["load"] = {"type": "kv_capacity_fraction", "fraction": 1.0}
    CoreRecommendationConfig.model_validate(raw)
    path = tmp_path / "recommend.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "recommendation"
    assert cli.main(["recommend", "-c", str(path), "--output-dir", str(output)]) == 0
    exported_path = output / "recommendations/0001.yaml"
    exported = CorePredictionConfig.from_yaml(exported_path)
    assert exported.engine.fpm_profile.deployments[0].resources.cache_layout == "grouped"
    assert exported.engine.workers.aggregated.kv_cache.prefix_caching is False
    assert (
        cli.main(
            [
                "predict",
                "-c",
                str(exported_path),
                "--output-dir",
                str(tmp_path / "exported-prediction"),
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("kv_cache_capacity_bytes", 9999, "byte capacity conflicts"),
        ("kv_cache_groups", None, "cache groups conflict"),
        ("tensor_parallel_size", 4, "tensor_parallel_size|topology"),
        ("dp_size", 2, "dp_size|topology"),
    ],
)
def test_native_grouped_rematerialization_checks_identity_and_budget(grouped_case, field, value, error):
    from aisim import _runtime
    from aisim.runner import EngineReplayRunnerFactory

    profile, root = grouped_case
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(_prediction(profile, root)))

    class ChangedNativeInput:
        def run_replay_json(self, serialized):
            payload = json.loads(serialized)
            engine = payload.get("spec", payload)["engine"]
            if field in {"tensor_parallel_size", "dp_size"}:
                engine[field] = value
            elif field == "kv_cache_groups":
                # Wrong but internally valid physical page geometry must not
                # replace the canonical profile after Python materialization.
                engine["rank"][field][0]["page_size_bytes"] += 1
            else:
                engine["rank"][field] = value
            return _runtime.run_replay_json(json.dumps(payload))

    with pytest.raises(RuntimeError, match=error):
        EngineReplayRunnerFactory(runtime=ChangedNativeInput()).create(0).run(spec)


def test_grouped_capacity_preserves_graph_reservation_and_rejects_fixed_blocks(grouped_case):
    profile, root = grouped_case
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(_prediction(profile, root)))
    args = deepcopy(spec.backend_deployment.agg_engine_args)
    args["cuda_graph_reserved_bytes"] = 1000
    args["timing_model"]["config"]["cuda_graph_reserved_bytes"] = 1000
    budget = materialize_aic_num_gpu_blocks(args)
    assert budget["kv_cache_capacity_bytes"] == 2486
    args["num_gpu_blocks"] = 1
    with pytest.raises(ValueError, match="fixed num_gpu_blocks"):
        materialize_aic_num_gpu_blocks(args)


def test_grouped_topology_admission_uses_each_rank_byte_budget_before_collection(grouped_case, monkeypatch):
    from aisim_core.sdk import RustForwardPassPerfModel

    profile, root = grouped_case
    # DEP has an explicitly larger rank-local weight reservation. It leaves
    # 1086 bytes, insufficient for even the 2048-byte full-history group.
    profile["deployments"][1]["resources"]["weights_bytes"] = 2500

    def no_timing(*_args, **_kwargs):
        pytest.fail("candidate fit constructed a timing estimator before collection")

    monkeypatch.setattr(RustForwardPassPerfModel, "best_available", no_timing)
    candidates = parallel_configs_for(
        profile["model"],
        "h200_sxm",
        backend="vllm",
        backend_version="0.25.1",
        deployment_mode="agg",
        gpu_budget=2,
        max_num_tokens=128,
        max_batch_size=4,
        fpm_profile=profile,
        systems_paths=[str(root)],
    )
    assert [candidate.shape.strategy for candidate in candidates] == ["tp"]


def test_grouped_kv_load_uses_bytes_without_inventing_token_capacity(grouped_case):
    profile, root = grouped_case
    candidates = parallel_configs_for(
        profile["model"],
        "h200_sxm",
        backend="vllm",
        backend_version="0.25.1",
        deployment_mode="agg",
        gpu_budget=2,
        max_num_tokens=128,
        max_batch_size=4,
        fpm_profile=profile,
        systems_paths=[str(root)],
    )
    assert len(candidates) == 2
    for candidate in candidates:
        sample = {
            "model_name": profile["model"],
            "hardware_sku": "h200_sxm",
            "backend": "vllm",
            "fpm_profile": profile,
            "systems_paths": [str(root)],
            "agg_block_size": 64,
            "agg_max_num_batched_tokens": 128,
            "agg_max_num_seqs": 4,
            "agg_gpu_memory_utilization": 0.9,
        }
        result = resolve_kv_load(
            sample,
            workload=Workload(isl=1152, osl=4, kv_load_ratio=1.0, request_count=2),
            parallel_config=candidate,
            ratio=1.0,
            backend_version="0.25.1",
        )
        # 1216 full-history + 528 sliding-window + 16 convolution bytes.
        assert result.role_request_cache_bytes == {"agg": 1760}
        assert result.role_capacity_bytes == {"agg": 3486 * candidate.shape.dp}
        assert result.concurrency_capacity == candidate.shape.dp
        assert result.role_capacity_tokens == {}


def test_grouped_kv_load_sizes_each_pd_role_by_its_bytes(grouped_case):
    profile, root = grouped_case
    profile = _with_transfer_markers(profile)
    search = {
        "backend": "vllm",
        "backend_version": "0.31.0",
        "deployment_mode": "disagg",
        "gpu_budget": 4,
        "max_num_tokens": 128,
        "max_batch_size": 4,
        "fpm_profile": profile,
        "systems_paths": [str(root)],
    }
    candidates = parallel_configs_for(profile["model"], "h200_sxm", **search)
    # The shared deployments pair only with their own shape.
    assert {(candidate.prefill.shape.tp, candidate.decode.shape.tp) for candidate in candidates} == {(2, 2), (1, 1)}
    sample = {
        "model_name": profile["model"],
        "hardware_sku": "h200_sxm",
        "backend": "vllm",
        "fpm_profile": profile,
        "systems_paths": [str(root)],
        **{
            f"{role}_{name}": value
            for role in ("prefill", "decode")
            for name, value in (
                ("block_size", 64),
                ("max_num_batched_tokens", 128),
                ("max_num_seqs", 4),
                ("gpu_memory_utilization", 0.9),
            )
        },
    }
    workload = Workload(isl=1152, osl=4, kv_load_ratio=1.0, request_count=2)
    for candidate in candidates:
        result = resolve_kv_load(
            sample, workload=workload, parallel_config=candidate, ratio=1.0, backend_version="0.31.0"
        )
        # Each role holds the 1,760 bytes per request of the aggregated rank, and the decode role sets the load.
        assert result.role_request_cache_bytes == {"prefill": 1760, "decode": 1760}
        assert result.role_capacity_bytes == {
            "prefill": 3486 * candidate.prefill.shape.dp,
            "decode": 3486 * candidate.decode.shape.dp,
        }
        assert result.concurrency_capacity == candidate.decode.shape.dp
        assert result.role_capacity_tokens == {}
    # A P/D role refuses its own fixed blocks, and the topology budget refuses speculation.
    with pytest.raises(ValueError, match="^grouped FPM cache requires replay without fixed scalar blocks$"):
        resolve_kv_load(
            {**sample, "decode_num_gpu_blocks": 64},
            workload=workload,
            parallel_config=candidates[0],
            ratio=1.0,
            backend_version="0.31.0",
        )
    with pytest.raises(ValueError, match="^grouped FPM cache requires replay without speculation$"):
        parallel_configs_for(profile["model"], "h200_sxm", nextn=1, **search)


def test_grouped_kv_load_sizes_each_pd_role_with_its_own_context_length(grouped_case, monkeypatch):
    profile, root = grouped_case
    profile = _with_transfer_markers(profile)
    for deployment in profile["deployments"]:
        resources = deployment["resources"]
        for name in ("weights_bytes", "activations_bytes", "runtime_overhead_bytes", "comm_overhead_bytes"):
            resources.pop(name)
        resources["max_batch_size"] = 4
        resources["runtime_memory"] = {
            "kv_cache_bytes": 3486,
            "max_model_len": 1024,
            "gpu_memory_utilization": 0.9,
            "provenance": "Synthetic initialized cache pool; no GPU qualification.",
        }
    role_context = {"prefill": 1024, "decode": 512}
    candidates = parallel_configs_for(
        profile["model"],
        "h200_sxm",
        backend="vllm",
        backend_version="0.31.0",
        deployment_mode="disagg",
        gpu_budget=4,
        max_num_tokens=128,
        max_batch_size=4,
        fpm_profile=profile,
        systems_paths=[str(root)],
        max_seq_len=2048,
        role_max_seq_len=role_context,
    )
    # The global context is above the probe's max_model_len, and each role sets its own context within it, as a
    # role deployment reads it. The two roles differ, so one role's context cannot size both.
    sample = {
        "model_name": profile["model"],
        "hardware_sku": "h200_sxm",
        "backend": "vllm",
        "fpm_profile": profile,
        "systems_paths": [str(root)],
        "context_length": 2048,
        **{
            f"{role}_{name}": value
            for role in ("prefill", "decode")
            for name, value in (
                ("block_size", 64),
                ("max_num_batched_tokens", 128),
                ("max_num_seqs", 4),
                ("gpu_memory_utilization", 0.9),
                ("context_length", role_context[role]),
            )
        },
    }
    from aisim.sweeper import kv_load

    sized = {}
    estimate = kv_load.estimate_grouped_cache_budget

    def record(*args, **kwargs):
        sized[kwargs["worker_type"]] = kwargs["context_length"]
        return estimate(*args, **kwargs)

    monkeypatch.setattr(kv_load, "estimate_grouped_cache_budget", record)
    result = resolve_kv_load(
        sample,
        workload=Workload(isl=256, osl=4, kv_load_ratio=1.0, request_count=2),
        parallel_config=candidates[0],
        ratio=1.0,
        backend_version="0.31.0",
    )
    assert sized == {"prefill": 1024, "decode": 512}
    assert set(result.role_request_cache_bytes) == {"prefill", "decode"}


def test_grouped_state_group_adds_fixed_bytes_per_request(grouped_case):
    profile, root = grouped_case
    profile = _with_state_group(profile)
    candidates = parallel_configs_for(
        profile["model"],
        "h200_sxm",
        backend="vllm",
        backend_version="0.25.1",
        deployment_mode="agg",
        gpu_budget=2,
        max_num_tokens=128,
        max_batch_size=4,
        fpm_profile=profile,
        systems_paths=[str(root)],
    )
    assert len(candidates) == 2
    for candidate in candidates:
        sample = {
            "model_name": profile["model"],
            "hardware_sku": "h200_sxm",
            "backend": "vllm",
            "fpm_profile": profile,
            "systems_paths": [str(root)],
            "agg_block_size": 64,
            "agg_max_num_batched_tokens": 128,
            "agg_max_num_seqs": 4,
            "agg_gpu_memory_utilization": 0.9,
        }
        result = resolve_kv_load(
            sample,
            workload=Workload(isl=1152, osl=4, kv_load_ratio=1.0, request_count=2),
            parallel_config=candidate,
            ratio=1.0,
            backend_version="0.25.1",
        )
        # The 1760 bytes of the other groups plus 2 state blocks of 64 bytes. The capacity does not change.
        assert result.role_request_cache_bytes == {"agg": 1888}
        assert result.role_capacity_bytes == {"agg": 3486 * candidate.shape.dp}
        assert result.concurrency_capacity == candidate.shape.dp
        assert result.role_capacity_tokens == {}


@pytest.mark.parametrize(
    "update,error",
    [
        ({"prefix_caching": True}, "prefix_caching=false"),
        ({"capacity": {"type": "fixed", "blocks": 1024}}, "byte budget"),
        ({"bytes_per_token": 8}, "byte budget"),
        ({"host_offload": {"num_host_blocks": 64}}, "only HBM"),
    ],
)
def test_grouped_incompatible_cache_settings_fail_before_replay(grouped_case, update, error):
    profile, root = grouped_case
    raw = _prediction(profile, root)
    raw["engine"]["workers"]["aggregated"]["kv_cache"].update(update)
    with pytest.raises(ValidationError, match=error):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize(
    "changes",
    [
        {"nextn": 1, "nextn_accepted": 1.0},
        {"speculation": {"kind": "ngram", "num_speculative_tokens": 1, "acceptance_rates": [0.5]}},
    ],
)
def test_grouped_unsupported_execution_modes_fail_before_replay(grouped_case, changes):
    profile, root = grouped_case
    raw = _prediction(profile, root)
    raw["engine"].update(changes)
    with pytest.raises(ValidationError, match="grouped FPM cache supports only vLLM without speculative decoding"):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize(
    "case,error",
    [
        ("vllm_0310", None),
        ("vllm_0320", None),
        ("vllm_0330", None),
        ("no_markers", _PD_MARKERS),
        ("vllm_0300", _PD_VERSION),
        ("unequal_tp", _PD_PAIRING),
        ("other_groups", _PD_PAIRING),
        ("linear_decode", _PD_PAIRING),
        ("transfer_bytes", _PD_TRANSFER),
        ("non_cacheable_window", _PD_REPLAY_WINDOW),
        ("ring_state", None),
    ],
)
def test_grouped_pd_requires_equal_tp_identical_groups_and_markers(grouped_case, case, error):
    from aisim.sweeper.config import SearchSpace

    profile, root = grouped_case
    version = {"vllm_0320": "0.32.0", "vllm_0330": "0.33.0", "vllm_0300": "0.30.0"}.get(case, "0.31.0")
    if case == "linear_decode":
        profile = _mixed_profile(profile)
    profile = _with_transfer_markers(profile, version=version)
    dep = {"tensor": 1, "attention_data": 2, "moe_tensor": 1, "moe_expert": 2}
    if case == "no_markers":
        for deployment in profile["deployments"]:
            for group in deployment["resources"]["cache_groups"]:
                del group["prefix_cacheable"], group["transfer_bytes_per_block"]
    elif case == "non_cacheable_window":
        # The window group of the fixture as a bounded replay window, which the transfer rules do not model. The
        # window group that takes part in prefix caching passes in the other cases.
        for deployment in profile["deployments"]:
            deployment["resources"]["cache_groups"][1]["prefix_cacheable"] = False
    elif case == "ring_state":
        # A ring described as a state group of one block that does not take part in prefix caching.
        for deployment in profile["deployments"]:
            ring = {**_STATE_GROUP, "name": "ring", "blocks_per_request": 1, "prefix_cacheable": False}
            deployment["resources"]["cache_groups"].append(
                {**ring, "transfer_bytes_per_block": ring["page_size_bytes"]}
            )
    elif case == "other_groups":
        # Role-tagged deployments of one shape: the decode deployment lacks the convolution group.
        prefill = profile["deployments"][0]
        decode = {**deepcopy(prefill), "worker_type": "decode"}
        decode["resources"]["cache_groups"] = decode["resources"]["cache_groups"][:-1]
        profile["deployments"][:1] = [{**prefill, "worker_type": "prefill"}, decode]
    raw = _pd_prediction(profile, root)
    search = _pd_search(profile, root)
    workers = raw["engine"]["workers"]
    # A search drops the enumerated pairs that fail the pairing rule, so it names such a pair with a pin.
    tensor_pin, dep_pin = {"tp": 2, "moe_tp": 2}, {"tp": 1, "attention_dp": 2, "moe_ep": 2}
    if case == "unequal_tp":
        workers["decode"]["parallelism"] = dep
        search["parallel_configs_by_mode"] = {"disagg": [{"prefill": tensor_pin, "decode": dep_pin}]}
    elif case == "other_groups":
        search["parallel_configs_by_mode"] = {"disagg": [{"prefill": tensor_pin, "decode": tensor_pin}]}
    elif case == "linear_decode":
        # The mixed profile groups the DEP deployment and keeps the tensor-parallel one linear.
        workers["prefill"]["parallelism"] = dep
        search["parallel_configs_by_mode"] = {"disagg": [{"prefill": dep_pin, "decode": tensor_pin}]}
    elif case == "transfer_bytes":
        raw["engine"]["kv_transfer"] = {"bytes_per_token": 7}
        search["kv_transfer_bytes_per_token"] = 7
    if error is None:
        CorePredictionConfig.model_validate(raw)
        SearchSpace.model_validate(search)
        return
    with pytest.raises(ValidationError, match=re.escape(error)):
        CorePredictionConfig.model_validate(raw)
    with pytest.raises(ValidationError, match=re.escape(error)):
        SearchSpace.model_validate(search)


def test_grouped_pd_of_a_probe_profile_gets_the_markers_text(grouped_case):
    # A profile as the import of a vLLM 0.33.0 runtime probe writes it for a cache outside the measured shape:
    # prefix_cacheable on every group, the recorded version 0.33.0 and no transfer bytes. Replay models P/D for vLLM
    # 0.33.0, so the profile gets the text of the markers.
    from aisim.sweeper.config import SearchSpace

    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version="0.33.0")
    for deployment in profile["deployments"]:
        deployment["resources"]["runtime_memory"]["runtime_framework_version"] = "0.33.0"
    with pytest.raises(ValidationError, match=re.escape(_PD_MARKERS)):
        CorePredictionConfig.model_validate(_pd_prediction(profile, root))
    with pytest.raises(ValidationError, match=re.escape(_PD_MARKERS)):
        SearchSpace.model_validate(_pd_search(profile, root))


@pytest.mark.parametrize("prefix_caching", [False, True])
def test_grouped_pd_accepts_vllm_0330_with_prefix_caching_on_and_off(grouped_case, prefix_caching):
    # Replay has grouped prefix-caching rules and P/D transfer rules for vLLM 0.33.0. A profile of vLLM 0.33.0 with
    # prefix evidence and both P/D markers passes the prediction and the search, with prefix caching on or off.
    from aisim.sweeper.config import SearchSpace

    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version="0.33.0")
    profile = _with_transfer_markers(profile, version="0.33.0")
    raw = _pd_prediction(profile, root)
    raw["engine"]["context_length"] = profile["deployments"][0]["resources"]["runtime_memory"]["max_model_len"]
    for worker in raw["engine"]["workers"].values():
        worker["kv_cache"]["prefix_caching"] = prefix_caching
    search = _pd_search(profile, root)
    search["prefill_enable_prefix_caching"] = prefix_caching
    CorePredictionConfig.model_validate(raw)
    SearchSpace.model_validate(search)


@pytest.mark.parametrize("prefix_caching", [False, True])
def test_grouped_pd_rank_on_vllm_0330_refuses_a_prefill_schedule_interval_above_one(grouped_case, prefix_caching):
    # vLLM 0.33.0 applies the interval without data parallelism, with prefix caching on or off, and replay does not
    # model that. A prefill or decode rank gets the P/D text first, whose remedy holds with prefix caching off too.
    profile, root = grouped_case

    def prediction(version, role, interval):
        versioned = _with_prefix_evidence(_with_state_group(profile), version=version)
        versioned = _with_transfer_markers(versioned, version=version)
        raw = _pd_prediction(versioned, root)
        raw["engine"]["context_length"] = versioned["deployments"][0]["resources"]["runtime_memory"]["max_model_len"]
        for worker in raw["engine"]["workers"].values():
            worker["kv_cache"]["prefix_caching"] = prefix_caching
        raw["engine"]["workers"][role]["scheduler"]["prefill_schedule_interval"] = interval
        return raw

    for role in ("prefill", "decode"):
        with pytest.raises(ValidationError, match=re.escape(_PD_INTERVAL)):
            CorePredictionConfig.model_validate(prediction("0.33.0", role, 2))
        CorePredictionConfig.model_validate(prediction("0.33.0", role, 1))
        # The interval comes before the markers.
        unmarked = prediction("0.33.0", role, 2)
        for deployment in unmarked["engine"]["fpm_profile"]["deployments"]:
            for group in deployment["resources"]["cache_groups"]:
                del group["transfer_bytes_per_block"]
        with pytest.raises(ValidationError, match=re.escape(_PD_INTERVAL)):
            CorePredictionConfig.model_validate(unmarked)
        # The rows of 0.31.0 and 0.32.0 keep the interval.
        for version in ("0.31.0", "0.32.0"):
            CorePredictionConfig.model_validate(prediction(version, role, 2))
    if not prefix_caching:
        # A prefill worker beside the aggregated worker of an aggregated prediction is no P/D rank. It fails on its
        # role, not on the P/D text.
        raw = _prediction(_with_prefix_evidence(_with_state_group(profile), version="0.33.0"), root)
        raw["engine"].update(mode="aggregated", backend_version="0.33.0")
        workers = raw["engine"]["workers"]
        workers["prefill"] = deepcopy(workers["aggregated"])
        workers["prefill"]["kv_cache"]["prefix_caching"] = False
        workers["prefill"]["scheduler"]["prefill_schedule_interval"] = 2
        with pytest.raises(ValidationError, match="aggregated mode rejects prefill/decode workers") as refused:
            CorePredictionConfig.model_validate(raw)
        assert _PD_INTERVAL not in str(refused.value)


def test_grouped_pd_refuses_roles_whose_evidence_records_two_vllm_versions(grouped_case):
    # A prefill deployment with vLLM 0.32.0 evidence beside a decode deployment with vLLM 0.33.0 evidence, under the
    # label 0.33.0. Replay follows the rules of one version for a handoff, so the pair is refused.
    from aisim.sweeper.config import SearchSpace

    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version="0.33.0")
    profile = _with_transfer_markers(profile, version="0.33.0")
    prefill = profile["deployments"][0]
    decode = {**deepcopy(prefill), "worker_type": "decode"}
    prefill["resources"]["runtime_memory"]["runtime_framework_version"] = "0.32.0"
    decode["resources"]["runtime_memory"]["runtime_framework_version"] = "0.33.0"
    profile["deployments"][:1] = [{**prefill, "worker_type": "prefill"}, decode]
    search = _pd_search(profile, root)
    pin = {"tp": 2, "moe_tp": 2}
    search["parallel_configs_by_mode"] = {"disagg": [{"prefill": pin, "decode": pin}]}
    error = (
        "grouped FPM cache P/D requires runtime_memory.runtime_framework_version (0.32.0) to equal the deployment's "
        "backend_version (0.33.0)"
    )
    with pytest.raises(ValidationError, match=re.escape(error)):
        CorePredictionConfig.model_validate(_pd_prediction(profile, root))
    with pytest.raises(ValidationError, match=re.escape(error)):
        SearchSpace.model_validate(search)


@pytest.mark.parametrize(("case", "error"), [("non_cacheable_window", _PD_REPLAY_WINDOW), ("vllm_0300", _PD_VERSION)])
def test_grouped_pd_with_prefix_caching_gives_the_texts_whose_remedies_hold_on_a_pd_rank(grouped_case, case, error):
    # With prefix caching on, the prefix-caching rule runs before the P/D rule. A bounded replay window and a version
    # that P/D does not model fail on a prefill or decode rank with prefix caching off too. So the window text of the
    # prefix-caching rule ("turn prefix caching off") would point to a setting that is refused again, and its version
    # text would name prefix caching for a rank that fails on its P/D version. A prefill or decode rank checks them
    # first and gets the P/D texts, as the Rust gate does.
    from aisim.sweeper.config import SearchSpace

    profile, root = grouped_case
    version = "0.30.0" if case == "vllm_0300" else "0.31.0"
    profile = _with_prefix_evidence(_with_state_group(profile), version=version)
    profile = _with_transfer_markers(profile, version=version)
    if case == "non_cacheable_window":
        for deployment in profile["deployments"]:
            deployment["resources"]["cache_groups"][1]["prefix_cacheable"] = False
    raw = _pd_prediction(profile, root)
    for worker in raw["engine"]["workers"].values():
        worker["kv_cache"]["prefix_caching"] = True
    search = _pd_search(profile, root)
    search["prefill_enable_prefix_caching"] = True
    search["decode_enable_prefix_caching"] = True
    with pytest.raises(ValidationError, match=re.escape(error)):
        CorePredictionConfig.model_validate(raw)
    with pytest.raises(ValidationError, match=re.escape(error)):
        SearchSpace.model_validate(search)


def test_grouped_pd_of_a_probe_profile_under_another_label_gets_the_recorded_version_text(grouped_case):
    # A profile from the vLLM 0.33.0 runtime probe under the label 0.32.0, for a cache outside the measured shape:
    # prefix_cacheable on every group, the recorded version 0.33.0 and no transfer bytes. The rule on the recorded
    # version comes before the markers.
    from aisim.sweeper.config import SearchSpace

    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version="0.32.0")
    for deployment in profile["deployments"]:
        deployment["resources"]["runtime_memory"]["runtime_framework_version"] = "0.33.0"
    error = (
        "grouped FPM cache P/D requires runtime_memory.runtime_framework_version (0.33.0) to equal the deployment's "
        "backend_version (0.32.0)"
    )
    with pytest.raises(ValidationError, match=re.escape(error)):
        CorePredictionConfig.model_validate(_pd_prediction(profile, root))
    with pytest.raises(ValidationError, match=re.escape(error)):
        SearchSpace.model_validate(_pd_search(profile, root))


def test_grouped_pd_refuses_evidence_of_another_vllm_version(grouped_case):
    # A profile that records vLLM 0.33.0 evidence under the label 0.32.0 cannot replay P/D with the rules of the label.
    from aisim.sweeper.config import SearchSpace

    profile, root = grouped_case
    profile = _with_transfer_markers(
        _with_prefix_evidence(_with_state_group(profile), version="0.32.0"), version="0.32.0"
    )
    # A profile without the field, or with the label's own version, passes.
    for recorded in (None, "0.32.0"):
        for deployment in profile["deployments"]:
            deployment["resources"]["runtime_memory"]["runtime_framework_version"] = recorded
        CorePredictionConfig.model_validate(_pd_prediction(profile, root))
        SearchSpace.model_validate(_pd_search(profile, root))
    for deployment in profile["deployments"]:
        deployment["resources"]["runtime_memory"]["runtime_framework_version"] = "0.33.0"
    error = (
        "grouped FPM cache P/D requires runtime_memory.runtime_framework_version (0.33.0) to equal the deployment's "
        "backend_version (0.32.0)"
    )
    with pytest.raises(ValidationError, match=re.escape(error)):
        CorePredictionConfig.model_validate(_pd_prediction(profile, root))
    with pytest.raises(ValidationError, match=re.escape(error)):
        SearchSpace.model_validate(_pd_search(profile, root))


def test_grouped_pd_recommendation_with_fixed_unequal_tp_gives_the_pairing_text(grouped_case):
    from aisim.recommend import recommendation_to_sweeper

    profile, root = grouped_case
    raw = _pd_prediction(_with_transfer_markers(profile), root)
    workers = raw["engine"]["workers"]
    workers["decode"]["parallelism"] = {"tensor": 1, "attention_data": 2, "moe_tensor": 1, "moe_expert": 2}
    # A preset list for each role lowers to the pinned pairs of the two lists.
    for worker in workers.values():
        worker["parallelism"] = {"preset": [{**worker["parallelism"], "replicas": 1, "pipeline": 1}]}
    raw["optimization"] = {"constraints": {"max_candidate_gpus": 4}}
    config = CoreRecommendationConfig.model_validate(raw)
    with pytest.raises(ValidationError, match=re.escape(_PD_PAIRING)):
        recommendation_to_sweeper(config)


def test_disaggregated_prediction_with_a_grouped_aggregated_worker_keeps_the_role_error(grouped_case):
    # Only the prefill and the decode deployment decide the P/D rule. Here both are linear, so the grouped aggregated
    # worker fails on its role, not on a P/D text.
    profile, root = grouped_case
    raw = _pd_prediction(_mixed_profile(profile), root)
    workers = raw["engine"]["workers"]
    workers["aggregated"] = deepcopy(workers["prefill"])
    workers["aggregated"]["parallelism"] = {"tensor": 1, "attention_data": 2, "moe_tensor": 1, "moe_expert": 2}
    with pytest.raises(ValidationError, match="disaggregated mode rejects aggregated workers"):
        CorePredictionConfig.model_validate(raw)


def test_grouped_prediction_whose_roles_do_not_fit_its_mode_keeps_the_role_error(grouped_case):
    # A prefill or decode worker of a vLLM 0.33.0 profile with prefix caching on, in a prediction that is no P/D pair:
    # beside the aggregated worker of an aggregated prediction, or as the only worker of a disaggregated prediction.
    # The P/D rule does not run there, so the worker fails on its role, not on the P/D version text.
    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version="0.33.0")
    profile = _with_transfer_markers(profile, version="0.33.0")
    for mode, error in (
        ("aggregated", "aggregated mode rejects prefill/decode workers"),
        ("disaggregated", "disaggregated mode requires prefill and decode workers"),
    ):
        for role in ("prefill", "decode"):
            raw = _prediction(profile, root)
            engine = raw["engine"]
            engine["mode"] = mode
            engine["backend_version"] = "0.33.0"
            workers = engine["workers"]
            stray = deepcopy(workers["aggregated"])
            stray["kv_cache"]["prefix_caching"] = True
            if mode == "disaggregated":
                workers.clear()
            workers[role] = stray
            with pytest.raises(ValidationError, match=re.escape(error)) as refused:
                CorePredictionConfig.model_validate(raw)
            assert _PD_VERSION not in str(refused.value), (mode, role)
    # The same stray worker with a bounded replay window, in both shapes, gets the window text of the prefix-caching
    # rule, not the one of the P/D rule.
    windowed = deepcopy(profile)
    for deployment in windowed["deployments"]:
        deployment["resources"]["cache_groups"][1]["prefix_cacheable"] = False
    prefix_window = "kv_cache_groups with enable_prefix_caching=true do not model a sliding-window attention group"
    for mode, role in (("aggregated", "prefill"), ("disaggregated", "prefill"), ("disaggregated", "decode")):
        raw = _prediction(windowed, root)
        raw["engine"].update(mode=mode, backend_version="0.33.0")
        raw["engine"]["context_length"] = windowed["deployments"][0]["resources"]["runtime_memory"]["max_model_len"]
        workers = raw["engine"]["workers"]
        stray = deepcopy(workers["aggregated"])
        stray["kv_cache"]["prefix_caching"] = True
        if mode == "disaggregated":
            workers.clear()
        workers[role] = stray
        with pytest.raises(ValidationError, match=re.escape(prefix_window)) as refused:
            CorePredictionConfig.model_validate(raw)
        assert _PD_REPLAY_WINDOW not in str(refused.value), (mode, role)


@pytest.mark.parametrize("version", ["0.31.0", "0.32.0", "0.33.0"])
def test_grouped_pd_prediction_replays_one_p_one_d(grouped_case, tmp_path, version):
    profile, root = grouped_case
    profile = _with_transfer_markers(_with_state_group(profile), version=version)
    _write_timings(root, profile, version=version)
    raw = _pd_prediction(profile, root, bandwidth=1.0)
    deployment = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)).backend_deployment
    prefill, decode = deployment.prefill_engine_args, deployment.decode_engine_args
    for args in (prefill, decode):
        # A grouped rank derives the transfer payload from its groups, so it has no scalar bytes per token.
        assert "kv_transfer_bytes_per_token" not in args
        assert (args["kv_transfer_bandwidth"], args["kv_transfer_timing_mode"]) == (1.0, "destination_missing")
        # The Rust gate refuses a grouped prefill or decode rank without vLLM 0.31.0, 0.32.0 or 0.33.0, also with
        # prefix caching off.
        assert (args["enable_prefix_caching"], args["vllm_version"]) == (False, version)
        assert "prefix_match_unit" not in args
    groups = materialize_aic_num_gpu_blocks(prefill)["kv_cache_groups"]
    assert [group["name"] for group in groups] == ["full", "window", "conv", "state"]
    assert materialize_aic_num_gpu_blocks(decode)["kv_cache_groups"] == groups
    path = tmp_path / "predict.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "prediction"
    assert cli.main(["predict", "-c", str(path), "--output-dir", str(output)]) == 0
    assert json.loads((output / "prediction.json").read_text())["completed_requests"] == 2


@pytest.mark.parametrize("version", ["0.31.0", "0.32.0", "0.33.0"])
def test_grouped_pd_prediction_with_prefix_caching_on_both_ranks(grouped_case, tmp_path, version):
    # Planning writes prefix_caching: true into an accepted role's worker.yaml, so a composed P/D prediction of two
    # planned roles runs both ranks in block mode.
    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version=version)
    profile = _with_transfer_markers(profile, version=version)
    _write_timings(root, profile, version=version)
    raw = _pd_prediction(profile, root, bandwidth=1.0)
    aggregated = _prefix_caching_prediction(profile, root, version)
    raw["engine"]["context_length"] = aggregated["engine"]["context_length"]
    raw["traffic"] = aggregated["traffic"]
    for worker in raw["engine"]["workers"].values():
        worker["kv_cache"]["prefix_caching"] = True
    deployment = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)).backend_deployment
    for args in (deployment.prefill_engine_args, deployment.decode_engine_args):
        assert (args["enable_prefix_caching"], args["prefix_match_unit"], args["vllm_version"]) == (True, 64, version)
    path = tmp_path / "predict.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "prediction"
    assert cli.main(["predict", "-c", str(path), "--output-dir", str(output), "--capture-per-request"]) == 0
    report = json.loads((output / "prediction.json").read_text())
    assert report["completed_requests"] == 2
    rows = report["per_request"]
    # The prefill rank computes 519 tokens and keeps the state at 512, so the identical second request reuses 512.
    assert [row["reused_input_tokens"] for row in rows] == [0, 512]
    # The first transfer moves 9 full-attention blocks, 9 window blocks and 1 state block, 64 bytes each: 1,216
    # bytes. The 512-token window skips 519 - 512 + 1 = 8 of the 519 tokens, less than a block, so all 9 window
    # blocks lie inside it. The decode rank keeps the 8 full blocks that the first load cached, so the second moves
    # 1 full block and the same fixed bytes: 704 bytes. At 1 GB/s a byte takes 1e-6 ms.
    delays = [row["destination_activated_ms"] - row["destination_reserved_ms"] for row in rows]
    assert delays == [pytest.approx(1216e-6, abs=1e-9), pytest.approx(704e-6, abs=1e-9)]


@pytest.mark.parametrize("version", ["0.31.0", "0.32.0", "0.33.0"])
def test_grouped_pd_prediction_with_prefix_caching_on_the_prefill_rank_only(grouped_case, tmp_path, version):
    # A search with prefix evidence defaults to prefix caching on prefill workers and off on decode workers.
    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version=version)
    profile = _with_transfer_markers(profile, version=version)
    _write_timings(root, profile, version=version)
    raw = _pd_prediction(profile, root, bandwidth=1.0)
    aggregated = _prefix_caching_prediction(profile, root, version)
    raw["engine"]["context_length"] = aggregated["engine"]["context_length"]
    raw["traffic"] = aggregated["traffic"]
    raw["engine"]["workers"]["prefill"]["kv_cache"]["prefix_caching"] = True
    deployment = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)).backend_deployment
    prefill, decode = deployment.prefill_engine_args, deployment.decode_engine_args
    assert (prefill["enable_prefix_caching"], prefill["prefix_match_unit"]) == (True, 64)
    assert (decode["enable_prefix_caching"], "prefix_match_unit" in decode) == (False, False)
    assert prefill["vllm_version"] == decode["vllm_version"] == version
    path = tmp_path / "predict.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "prediction"
    assert cli.main(["predict", "-c", str(path), "--output-dir", str(output), "--capture-per-request"]) == 0
    report = json.loads((output / "prediction.json").read_text())
    assert report["completed_requests"] == 2
    rows = report["per_request"]
    # The prefill rank reuses 512 tokens as with prefix caching on both ranks. The cold decode rank keeps no blocks,
    # so both transfers move the full 1,216 bytes.
    assert [row["reused_input_tokens"] for row in rows] == [0, 512]
    delays = [row["destination_activated_ms"] - row["destination_reserved_ms"] for row in rows]
    assert delays == [pytest.approx(1216e-6, abs=1e-9), pytest.approx(1216e-6, abs=1e-9)]


def test_grouped_onboarding_profile_reaches_real_collector_preview_and_render(
    grouped_case, tmp_path, monkeypatch, capsys
):
    from collector.fpm_forward import cli as collector_cli
    from collector.fpm_forward import runner as collector_runner

    profile, _root = grouped_case
    profile = deepcopy(profile)
    for deployment in profile["deployments"]:
        deployment["gemm_quant_mode"] = "fp8_static"
        deployment["fmha_quant_mode"] = "fp8"
    local_config = tmp_path / "config.json"
    local_config.write_text(
        json.dumps(
            {
                "architectures": [profile["architecture"]],
                "model_type": "example_moe",
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 3,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 1024,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 128,
                "max_position_embeddings": 2048,
                "torch_dtype": "bfloat16",
                "quantization_config": {"quant_method": "fp8", "kv_cache_scheme": "FP8"},
                "sliding_window": 512,
                "layer_types": ["full_attention", "sliding_attention", "sliding_attention"],
            }
        )
    )
    request = SupportRequest.model_validate(
        {
            "identity": {
                "model": profile["model"],
                "model_revision": profile["model_revision"],
                "model_kind": "moe",
                "framework_version": "0.25.1",
                "gpu": "h200_sxm",
                "interconnect": "nvswitch",
            },
            "search": {
                "tensor_parallel": 2,
                "attention_data_parallel": 1,
                "moe_tensor_parallel": 2,
                "moe_expert_parallel": 1,
                "context_length": 2048,
            },
            "collection": {"max_num_tokens": 128, "max_batch_size": 4},
            "fpm_profile": profile,
        }
    )
    plan_dir = tmp_path / "collection"
    create_plan(request, plan_dir)
    command = json.loads((plan_dir / "commands.json").read_text())["fpm_plan_local"][3:]
    command += ["--fpm-model-config", str(local_config)]
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    assert collector_cli.main(command) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["fpm_profile"] == request.fpm_profile.model_dump(mode="json")
    assert preview["point_generation"]["method"] == "native_self_benchmark"
    assert preview["point_generation"]["planned_point_count"] is None
    assert {cell["parallel_strategy"] for cell in preview["cells"]} == {"pure_tp"}
    assert (
        preview["topology_memory_admission"][0]["estimates"][0]["provenance"]
        == (profile["deployments"][0]["resources"]["provenance"])
    )

    def render_without_launch(_args, resolved):
        plan, overrides = resolved
        assert plan.to_dict()["fpm_profile"] == preview["fpm_profile"]
        for cell in plan.cells:
            target = tmp_path / f"render-{cell.workload_kind}"
            target.mkdir()
            collector_runner._render_cell(plan, cell, target, overrides)
            script = (target / "run.sh").read_text()
            assert f"--model {profile['model']}" in script
            assert "--max-num-batched-tokens 128" in script
            assert "--max-num-seqs 4" in script
        return []

    monkeypatch.setattr(collector_cli, "run_resolved", render_without_launch)
    assert collector_cli.main([arg for arg in command if arg != "--plan-only"]) == 0


def _plan_and_validate_weka(profile, tmp_path, monkeypatch, *, dep=False, version="0.25.1", context_length=2048):
    """Plan the onboarding of a grouped profile and replay a windowed Weka trace with validate-fpm.

    A shorter context, such as a probe's max_model_len, gets a workload and a second request that fit in it. The
    second request still crosses the 512-token window.
    """
    from aisim import supervision

    # The default workload of 1,024 input and 128 output tokens needs a context of 1,152 tokens.
    workload = {} if context_length >= 1152 else {"workload": {"input_tokens": 512, "output_tokens": 4}}
    second = min(1152, context_length - 64)
    request = SupportRequest.model_validate(
        {
            "identity": {
                "model": profile["model"],
                "model_revision": profile["model_revision"],
                "model_kind": "moe",
                "framework_version": version,
                "gpu": "h200_sxm",
                "interconnect": "nvswitch",
            },
            "search": {
                "tensor_parallel": 1 if dep else 2,
                "attention_data_parallel": 2 if dep else 1,
                "moe_tensor_parallel": 1 if dep else 2,
                "moe_expert_parallel": 2 if dep else 1,
                "context_length": context_length,
            },
            **workload,
            "collection": {"max_num_tokens": 128, "max_batch_size": 4},
            "fpm_profile": profile,
        }
    )
    plan = tmp_path / "collection"
    create_plan(request, plan)
    _write_timings(plan / "systems", profile, version=version)
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "id": "window-boundary-play",
                "models": ["trace/source-model"],
                "block_size": 64,
                "hash_id_scope": "local",
                "requests": [
                    {
                        "t": 0.0,
                        "type": "s",
                        "model": "trace/source-model",
                        "in": 512,
                        "out": 3,
                        "hash_ids": list(range(8)),
                    },
                    {
                        "t": 0.1,
                        "type": "s",
                        "model": "trace/source-model",
                        "in": second,
                        "out": 4,
                        "hash_ids": list(range(second // 64)),
                    },
                ],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(supervision, "main", cli.main)
    output = tmp_path / "validation"
    assert (
        cli.main(
            [
                "onboard",
                "validate-fpm",
                "--config",
                str(plan / "request.yaml"),
                "--output-dir",
                str(plan),
                "--trace",
                str(trace),
                "--validation-output-dir",
                str(output),
            ]
        )
        == 0
    )
    return plan, output


@reads_saved_config
@pytest.mark.parametrize("dep", [False, True])
def test_onboard_validation_replays_windowed_weka_with_full_context(grouped_case, tmp_path, monkeypatch, dep):
    profile, _root = grouped_case
    _plan, output = _plan_and_validate_weka(profile, tmp_path, monkeypatch, dep=dep)
    validation = json.loads((output / "validation.json").read_text())
    assert validation["status"] == "covered"
    assert validation["accuracy"] == "not_assessed"
    report = json.loads((output / "prediction/prediction.json").read_text())
    assert report["completed_requests"] == 2
    assert report["agentic_model_projection"]["target_model"] == profile["model"]
    saved = CorePredictionConfig.from_yaml(output / "predict.yaml")
    assert saved.engine.workers.aggregated.kv_cache.prefix_caching is False
    assert saved.engine.fpm_profile.deployments[0].resources.cache_layout == "grouped"


@pytest.mark.parametrize("profile_field", ["aic_fpm_profile", "fpm_profile"])
def test_grouped_runner_capability_rejects_unsupported_stack(grouped_case, profile_field):
    from aisim.runner import EngineReplayRunnerFactory
    from aisim.sweeper.replay import RunnerCapabilities

    profile, root = grouped_case
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(_prediction(profile, root)))
    with pytest.raises(ValueError, match="grouped FPM caches; use --stack engine"):
        RunnerCapabilities(supported_backend_topologies=(("vllm", "agg"),)).require_compatible(spec)
    EngineReplayRunnerFactory().capabilities().require_compatible(spec)
    linear_spec = prediction_to_replay_spec(
        CorePredictionConfig.model_validate(_prediction(_mixed_profile(profile), root))
    )
    RunnerCapabilities(supported_backend_topologies=(("vllm", "agg"),)).require_compatible(linear_spec)

    from dataclasses import replace

    for profile_variant, grouped in ((profile, True), (_mixed_profile(profile), False)):
        selected = profile_variant["deployments"][0]
        flat = {
            "aic_model_path": profile_variant["model"],
            "aic_system": selected["system"],
            profile_field: profile_variant,
            "aic_tp_size": selected["tp"],
            "aic_attention_dp_size": selected["dp"],
            "aic_moe_tp_size": selected["moe_tp"],
            "aic_moe_ep_size": selected["moe_ep"],
        }
        flat_spec = replace(spec, backend_deployment=replace(spec.backend_deployment, agg_engine_args=flat))
        capabilities = RunnerCapabilities(supported_backend_topologies=(("vllm", "agg"),))
        if grouped:
            with pytest.raises(ValueError, match="grouped FPM caches"):
                capabilities.require_compatible(flat_spec)
        else:
            capabilities.require_compatible(flat_spec)


def test_aggregated_only_grouped_profile_skips_unavailable_disagg(grouped_case):
    from aisim.sweeper.config import SmartSearchConfig
    from aisim.sweeper.search_space import enumerate_branches

    profile, root = grouped_case
    for deployment in profile["deployments"]:
        deployment["worker_type"] = "aggregated"
    raw = _mixed_search(profile, root)
    raw.update(fpm_profile=profile, agg_enable_prefix_caching=False)
    raw.pop("deployment_mode")
    config = SmartSearchConfig(search_space=raw, workload=Workload(isl=1152, osl=4, concurrency=1, request_count=2))
    with pytest.warns(UserWarning, match="disagg.*skipped"):
        branches = enumerate_branches(config, max_seq_len=2048)
    assert [branch.deployment_mode for branch in branches] == ["agg"]


def test_grouped_profile_without_a_valid_pair_skips_disagg(grouped_case):
    from aisim.sweeper.config import SmartSearchConfig
    from aisim.sweeper.search_space import enumerate_branches

    profile, root = grouped_case
    profile = _with_transfer_markers(profile)
    tensor, dep = profile["deployments"]
    # Prefill workers run only at tensor 2 and decode workers only at DEP, so no pair has equal tensor parallelism.
    profile["deployments"] = [
        {**deepcopy(tensor), "worker_type": "aggregated"},
        {**tensor, "worker_type": "prefill"},
        {**dep, "worker_type": "decode"},
    ]
    raw = {
        **_pd_search(profile, root),
        "deployment_mode": ["disagg", "agg"],
        "agg_max_num_batched_tokens": [128],
        "agg_max_num_seqs": [4],
        "agg_enable_prefix_caching": False,
    }
    # Validation warns about nothing. The root configuration turns any warning outside pytest.warns into an error.
    config = SmartSearchConfig(search_space=raw, workload=Workload(isl=1152, osl=4, concurrency=1, request_count=2))
    reason = "no prefill/decode deployment pair with equal tensor parallelism and identical cache groups"
    # One skip warning carries the reason. pytest.warns raises any other warning again, so a second one fails.
    with pytest.warns(UserWarning, match=f"^smart-sweep: deployment_mode='disagg' skipped — {reason}$"):
        branches = enumerate_branches(config, max_seq_len=2048)
    assert [branch.deployment_mode for branch in branches] == ["agg"]


def test_grouped_pd_recommendation_enumerates_equal_tp_pairs(grouped_case):
    from aisim.recommend import recommendation_to_sweeper
    from aisim.sweeper.search_space import enumerate_branches

    profile, root = grouped_case
    profile = _with_transfer_markers(profile)
    tensor, dep = profile["deployments"]
    # The role-tagged deployments at tensor 2 pair. The DEP deployment serves only decode workers and pairs with none.
    profile["deployments"] = [
        {**deepcopy(tensor), "worker_type": "prefill"},
        {**tensor, "worker_type": "decode"},
        {**dep, "worker_type": "decode"},
    ]
    raw = _pd_prediction(profile, root, bandwidth=1.0)
    for worker in raw["engine"]["workers"].values():
        worker["parallelism"] = {"preset": "default"}
    raw["optimization"] = {"constraints": {"max_candidate_gpus": 4}}
    branches = enumerate_branches(recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw)))
    assert [branch.deployment_mode for branch in branches] == ["disagg"]
    candidates = branches[0].parallel_configs
    assert candidates
    assert all(candidate.prefill.shape.tp == candidate.decode.shape.tp == 2 for candidate in candidates)


def test_grouped_heterogeneous_pd_search_keeps_only_valid_pairs(grouped_case):
    from aisim.sweeper.config import SmartSearchConfig
    from aisim.sweeper.search_space import enumerate_branches

    profile, root = grouped_case
    profile = _with_transfer_markers(profile)
    profile["deployments"] += [{**deepcopy(deployment), "system": "h100_sxm"} for deployment in profile["deployments"]]
    (root / "h100_sxm.yaml").write_bytes((root / "h200_sxm.yaml").read_bytes())
    raw = {**_pd_search(profile, root), "decode_hardware_sku": "h100_sxm"}
    config = SmartSearchConfig(search_space=raw, workload=Workload(isl=1152, osl=4, concurrency=1, request_count=2))
    (branch,) = enumerate_branches(config, max_seq_len=2048)
    # Each role is enumerated on its own hardware, and the pairs keep equal tensor parallelism.
    assert {(candidate.prefill.shape, candidate.decode.shape) for candidate in branch.parallel_configs} == {
        (shape, shape) for shape in {candidate.prefill.shape for candidate in branch.parallel_configs}
    }
    assert {candidate.prefill.shape.tp for candidate in branch.parallel_configs} == {1, 2}


@pytest.mark.parametrize(
    "prefix_caching",
    [{"prefill": False, "decode": False}, {"prefill": True, "decode": True}, {}],
    ids=["off", "on", "defaults"],
)
@pytest.mark.parametrize(
    "contexts",
    [
        {},
        {"context_length": 1024},
        {"context_length": 1024, "prefill_context_length": 512, "decode_context_length": 768},
    ],
    ids=["profile-context", "shared-context", "role-context"],
)
def test_grouped_sweeper_pd_candidate_ranks_take_the_vllm_version(grouped_case, prefix_caching, contexts):
    from aisim.sweeper.config import SearchSpace
    from aisim.sweeper.deploy import build_backend_deployment
    from aisim.sweeper.parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig
    from aisim.sweeper.sample import unroll_sample

    profile, root = grouped_case
    profile = _with_transfer_markers(_with_prefix_evidence(_with_state_group(profile)))
    raw = {
        **_pd_search(profile, root),
        # A recommendation with a KV transfer lowers bytes_per_token "auto" into the search.
        "kv_transfer_bytes_per_token": "auto",
        "kv_transfer_bandwidth": 1.0,
        **contexts,
    }
    # Without settings, the search keeps its defaults: prefix caching on prefill workers and off on decode workers.
    raw.pop("prefill_enable_prefix_caching")
    raw.update({f"{role}_enable_prefix_caching": value for role, value in prefix_caching.items()})
    worker = ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=2, moe_ep=1), replicas=1)
    sample = unroll_sample(
        search_space=SearchSpace.model_validate(raw),
        selection={
            "deployment_mode": "disagg",
            "backend": "vllm",
            "prefill_max_num_batched_tokens": 128,
            "prefill_max_num_seqs": 4,
            "decode_max_num_batched_tokens": 128,
            "decode_max_num_seqs": 4,
        },
        parallel_config=DisaggParallelConfig(prefill=worker, decode=worker),
    )
    deployment = build_backend_deployment(sample, backend_version="0.31.0")
    defaults = {"prefill": True, "decode": False}
    for role, args in (("prefill", deployment.prefill_engine_args), ("decode", deployment.decode_engine_args)):
        # Each grouped rank reads the transfer rules of its version and derives the payload from its groups.
        assert (args["vllm_version"], args["kv_transfer_bandwidth"]) == ("0.31.0", 1.0)
        assert "kv_transfer_bytes_per_token" not in args
        assert args["kv_transfer_timing_mode"] == "destination_missing"
        assert args["enable_prefix_caching"] is prefix_caching.get(role, defaults[role])
        # Cold and prefix-cached ranks keep explicit contexts or fall back to the profile's.
        assert args["max_model_len"] == contexts.get(f"{role}_context_length", contexts.get("context_length", 2048))
        if args["enable_prefix_caching"]:
            assert args["prefix_match_unit"] == 64
        else:
            assert "prefix_match_unit" not in args


@reads_saved_config
@pytest.mark.parametrize("version", ["0.31.0", "0.33.0"])
@pytest.mark.parametrize("prefix_caching", [False, True])
def test_grouped_pd_recommendation_replays_every_candidate(
    grouped_case, tmp_path, monkeypatch, prefix_caching, version
):
    from aisim.recommend import _run_recommendation
    from aisim.sweeper import SweepResult
    from aisim.sweeper.result import CandidateStatus

    monkeypatch.setattr("aisim.recommend.run_recommendation", _run_recommendation)
    profile, root = grouped_case
    if prefix_caching:
        profile = _with_prefix_evidence(_with_state_group(profile), version=version)
    profile = _with_transfer_markers(profile, version=version)
    _write_timings(root, profile, version=version)
    raw = _pd_prediction(profile, root, bandwidth=1.0)
    if prefix_caching:
        aggregated = _prefix_caching_prediction(profile, root, version)
        raw["engine"]["context_length"] = aggregated["engine"]["context_length"]
        raw["traffic"] = aggregated["traffic"]
    for worker in raw["engine"]["workers"].values():
        worker["parallelism"] = {"preset": "default"}
        worker["kv_cache"]["prefix_caching"] = prefix_caching
    raw["optimization"] = {"constraints": {"max_candidate_gpus": 4}}
    raw["optimizer"] = {"algorithm": "random", "max_trials": 2, "parallelism": 1}
    path = tmp_path / "recommend.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "recommendation"
    assert cli.main(["recommend", "-c", str(path), "--output-dir", str(output)]) == 0
    result = SweepResult.from_json((output / "recommendation.json").read_text())
    # Both equal-TP pairs replay. A candidate that failed its build or the Rust gate would not be feasible.
    assert len(result.candidates) == 2
    assert all(candidate.status is CandidateStatus.FEASIBLE for candidate in result.candidates)
    exported = CorePredictionConfig.from_yaml(output / "recommendations/0001.yaml")
    assert exported.engine.kv_transfer.bandwidth_gb_per_second == 1.0
    assert exported.engine.workers.prefill.kv_cache.prefix_caching is prefix_caching


def _prefix_search(profile, root, version):
    return {**_mixed_search(profile, root), "fpm_profile": profile, "backend_version": version}


@pytest.mark.parametrize(
    "case,error",
    [
        ("no_evidence", "prefix_caching=false for cold replay"),
        ("probe_without_prefix_caching", "prefix_caching=false for cold replay"),
        ("probe_without_hash_block", "prefix_caching=false for cold replay"),
        (
            "unmarked_group",
            "every cache group to declare prefix_cacheable; "
            "set prefix_cacheable \\(true or false\\) on each cache group",
        ),
        ("convolution_group", "grouped prefix caching does not support convolution cache groups"),
        (
            "non_cacheable_window",
            "kv_cache_groups with enable_prefix_caching=true do not model a sliding-window attention group with "
            "prefix_cacheable=false; describe a ring buffer as a state group with blocks_per_request: 1 and "
            "prefix_cacheable: false, and turn prefix caching off for a bounded replay window, which replay does not "
            "model",
        ),
        ("non_cacheable_full_attention", None),
        ("state_without_align", "requires mamba_cache_mode=align for a prefix-cacheable state group"),
        ("vllm_0320", None),
        ("vllm_0330", None),
        ("vllm_0300", "is modeled for vLLM 0.31.0, 0.32.0 and 0.33.0 only"),
        ("unequal_pages", "requires equal page_size_bytes across cache groups"),
        ("other_unit", "must equal the profile's hash_block_size_tokens for a grouped FPM cache"),
    ],
)
def test_grouped_prefix_caching_needs_vllm_evidence_with_align_mode(grouped_case, case, error):
    from aisim.sweeper.config import SearchSpace

    profile, root = grouped_case
    version = {"no_evidence": "0.25.1", "vllm_0320": "0.32.0", "vllm_0330": "0.33.0", "vllm_0300": "0.30.0"}.get(
        case, "0.31.0"
    )
    profile = _with_state_group(profile)
    if case != "no_evidence":
        profile = _with_prefix_evidence(profile, version=version)
    for deployment in profile["deployments"]:
        resources = deployment["resources"]
        if case == "probe_without_prefix_caching":
            resources["runtime_memory"].update(enable_prefix_caching=False, mamba_cache_mode="none")
        elif case == "probe_without_hash_block":
            del resources["runtime_memory"]["hash_block_size_tokens"]
        elif case == "unmarked_group":
            del resources["cache_groups"][1]["prefix_cacheable"]
        elif case == "convolution_group":
            resources["cache_groups"].append(dict(_CONVOLUTION_GROUP))
        elif case == "non_cacheable_window":
            # The window group of the fixture, as the Rust rule table refuses it.
            resources["cache_groups"][1]["prefix_cacheable"] = False
        elif case == "non_cacheable_full_attention":
            # A full-attention group that opts out takes no part in hits, as in vLLM.
            resources["cache_groups"][0]["prefix_cacheable"] = False
        elif case == "state_without_align":
            resources["runtime_memory"]["mamba_cache_mode"] = "none"
        elif case == "unequal_pages":
            resources["cache_groups"][1]["page_size_bytes"] = 16
    raw = _prediction(profile, root)
    raw["engine"]["backend_version"] = version
    cache = raw["engine"]["workers"]["aggregated"]["kv_cache"]
    cache["prefix_caching"] = True
    if case == "other_unit":
        cache["prefix_match_unit"] = 32
    # The search space has no prefix_match_unit, so only the prediction can state another unit.
    search = None if case == "other_unit" else _prefix_search(profile, root, version)
    if error is None:
        CorePredictionConfig.model_validate(raw)
        SearchSpace.model_validate(search)
        return
    with pytest.raises(ValidationError, match=error):
        CorePredictionConfig.model_validate(raw)
    if search is not None:
        with pytest.raises(ValidationError, match=error):
            SearchSpace.model_validate(search)


def test_grouped_cold_replay_text_names_the_field_of_each_gate(grouped_case):
    from aisim.sweeper.config import SearchSpace

    profile, root = grouped_case
    profile = _with_state_group(profile)  # no prefix-caching evidence
    raw = _prediction(profile, root)
    raw["engine"]["workers"]["aggregated"]["kv_cache"]["prefix_caching"] = True
    # pydantic puts "Value error, " before the text of a ValueError and " [type=" after it.
    text = "Value error, grouped FPM cache requires {}=false for cold replay [type="
    with pytest.raises(ValidationError, match=re.escape(text.format("kv_cache.prefix_caching"))):
        CorePredictionConfig.model_validate(raw)
    with pytest.raises(ValidationError, match=re.escape(text.format("agg_enable_prefix_caching"))):
        SearchSpace.model_validate(_prefix_search(profile, root, "0.25.1"))
    # A prefill or decode rank names its own search field. The markers give the profile a P/D version.
    pd_profile = _with_transfer_markers(profile)
    for role in ("prefill", "decode"):
        with pytest.raises(ValidationError, match=re.escape(text.format(f"{role}_enable_prefix_caching"))):
            SearchSpace.model_validate({**_pd_search(pd_profile, root), f"{role}_enable_prefix_caching": True})


@pytest.mark.parametrize("version", ["0.31.0", "0.32.0", "0.33.0"])
def test_grouped_prefix_caching_replays_with_reuse(grouped_case, tmp_path, version):
    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version=version)
    _write_timings(root, profile, version=version)
    raw = _prefix_caching_prediction(profile, root, version)
    args = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)).backend_deployment.agg_engine_args
    assert (args["enable_prefix_caching"], args["prefix_match_unit"], args["vllm_version"]) == (True, 64, version)
    path = tmp_path / "predict.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "prediction"
    assert cli.main(["predict", "-c", str(path), "--output-dir", str(output), "--capture-per-request"]) == 0
    report = json.loads((output / "prediction.json").read_text())
    assert report["completed_requests"] == 2
    # Every version keeps the state at 512, the last 64-token boundary below the 520-token prompt, so the
    # identical second request reuses 512 tokens.
    assert [row["reused_input_tokens"] for row in report["per_request"]] == [0, 512]


@pytest.mark.parametrize("version", ["0.31.0", "0.32.0", "0.33.0"])
@pytest.mark.parametrize(
    "host",
    [
        {"num_host_blocks": 64},
        {"num_host_blocks": 64, "connector": "offloading_v0_32"},
        {"num_host_blocks": 64, "scope": "cluster_shared"},
    ],
)
def test_grouped_prefix_caching_refuses_host_offload(grouped_case, version, host):
    profile, root = grouped_case
    raw = _prefix_caching_prediction(_with_prefix_evidence(profile, version=version), root, version)
    CorePredictionConfig.model_validate(raw)
    raw["engine"]["workers"]["aggregated"]["kv_cache"]["host_offload"] = host
    with pytest.raises(ValidationError, match="only HBM"):
        CorePredictionConfig.model_validate(raw)


def test_grouped_prefix_caching_rank_takes_the_probe_hash_block(grouped_case):
    profile, root = grouped_case
    raw = _prefix_caching_prediction(_with_prefix_evidence(_with_state_group(profile)), root, "0.31.0")

    def engine_args():
        return prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)).backend_deployment.agg_engine_args

    implicit = engine_args()
    raw["engine"]["workers"]["aggregated"]["kv_cache"]["prefix_match_unit"] = 64
    assert engine_args() == implicit
    # The Rust gate also refuses a rank without max_model_len. The compiler writes it for every vLLM rank.
    assert (implicit["prefix_match_unit"], implicit["vllm_version"], implicit["max_model_len"]) == (64, "0.31.0", 1024)
    raw["engine"]["workers"]["aggregated"]["kv_cache"] = {"prefix_caching": False}
    assert not {"prefix_match_unit", "vllm_version"} & engine_args().keys()


def test_grouped_rank_without_prefix_caching_refuses_prefix_match_unit(grouped_case):
    # Only prefix caching reads the unit of a grouped rank, and a grouped profile cannot use state_cache.
    profile, root = grouped_case
    raw = _prediction(profile, root)
    raw["engine"]["workers"]["aggregated"]["kv_cache"]["prefix_match_unit"] = 64
    error = "kv_cache.prefix_match_unit on a grouped FPM cache requires kv_cache.prefix_caching=true"
    with pytest.raises(ValidationError, match=error):
        CorePredictionConfig.model_validate(raw)
    # A linear deployment keeps the text of a linear cache.
    raw = _prediction(_mixed_profile(profile), root)
    raw["engine"]["workers"]["aggregated"]["kv_cache"]["prefix_match_unit"] = 64
    with pytest.raises(ValidationError, match="prefix_match_unit currently requires state_cache"):
        CorePredictionConfig.model_validate(raw)


@pytest.mark.parametrize(
    "hash_block,budget,refused",
    [(2176, 8190, True), (2176, 8192, False), (2174, 8192, True)],
)
def test_glm_grid_refuses_off_grid_units_and_budgets(grouped_case, hash_block, budget, refused):
    from aisim.sweeper.config import SearchSpace

    profile, root = grouped_case
    profile = _with_prefix_evidence(profile, hash_block=hash_block)
    profile["architecture"] = "Glm5NextForConditionalGeneration"
    for deployment in profile["deployments"]:
        # Runtime memory requires the exact recorded budget.
        deployment["resources"]["max_num_tokens"] = budget
    raw = _prediction(profile, root)
    raw["engine"]["backend_version"] = "0.31.0"
    worker = raw["engine"]["workers"]["aggregated"]
    worker["kv_cache"]["prefix_caching"] = True
    worker["scheduler"]["max_batched_tokens"] = budget
    search = {**_prefix_search(profile, root, "0.31.0"), "agg_max_num_batched_tokens": [budget]}
    if not refused:
        CorePredictionConfig.model_validate(raw)
        SearchSpace.model_validate(search)
        return
    error = "GLM-5.3-Flash prefix caching requires prefix_match_unit and max_batched_tokens that are multiples of 4"
    with pytest.raises(ValidationError, match=error):
        CorePredictionConfig.model_validate(raw)
    with pytest.raises(ValidationError, match=error):
        SearchSpace.model_validate(search)


@reads_saved_config
def test_grouped_plan_keeps_prefix_caching_on_with_evidence_and_validation_cold(grouped_case, tmp_path, monkeypatch):
    profile, _root = grouped_case
    profile = _with_prefix_evidence(profile)
    # Onboarding keeps the context within the probe's max_model_len (support/cli.py:317-320).
    plan, output = _plan_and_validate_weka(profile, tmp_path, monkeypatch, version="0.31.0", context_length=1024)
    for name in ("predict/pilot.yaml", "recommend/pilot.yaml"):
        worker = yaml.safe_load((plan / name).read_text())["engine"]["workers"]["aggregated"]
        assert worker["kv_cache"]["prefix_caching"] is True
    validation = json.loads((output / "validation.json").read_text())
    assert validation["status"] == "covered"
    # validate-fpm tests FPM timing, so its replay stays cold.
    saved = CorePredictionConfig.from_yaml(output / "predict.yaml")
    assert saved.engine.workers.aggregated.kv_cache.prefix_caching is False


def test_cold_validation_drops_the_unit_that_only_prefix_caching_reads(grouped_case, tmp_path, monkeypatch):
    from aisim.support.validation import _prediction_config

    profile, _root = grouped_case
    plan, _output = _plan_and_validate_weka(
        _with_prefix_evidence(profile), tmp_path, monkeypatch, version="0.31.0", context_length=1024
    )
    # The profile's hash block is a valid prefix_match_unit for a grouped rank with prefix caching. The plan does not
    # write it, and validate-fpm refuses a changed pilot, so the test gives such a pilot to _prediction_config.
    pilot = yaml.safe_load((plan / "predict/pilot.yaml").read_text())
    cache = pilot["engine"]["workers"]["aggregated"]["kv_cache"]
    cache["prefix_match_unit"] = 64
    CorePredictionConfig.model_validate(pilot)
    edited = tmp_path / "edited"
    (edited / "predict").mkdir(parents=True)
    (edited / "predict/pilot.yaml").write_text(yaml.safe_dump(pilot))
    # On a grouped rank only prefix caching reads the unit, so the cold replay turns prefix caching off and drops it.
    cold = _prediction_config(SupportRequest.from_yaml(plan / "request.yaml"), edited, tmp_path / "trace.jsonl")
    del cache["prefix_match_unit"]
    assert cold["engine"]["workers"]["aggregated"]["kv_cache"] == {**cache, "prefix_caching": False}


def test_grouped_plan_without_the_probe_hash_block_stays_cold(grouped_case, tmp_path, monkeypatch):
    # Replay hashes prompts in blocks of the profile's hash block, so runtime memory without it is no evidence.
    profile, _root = grouped_case
    profile = _with_prefix_evidence(profile)
    for deployment in profile["deployments"]:
        del deployment["resources"]["runtime_memory"]["hash_block_size_tokens"]
    plan, _output = _plan_and_validate_weka(profile, tmp_path, monkeypatch, version="0.31.0", context_length=1024)
    for name in ("predict/pilot.yaml", "recommend/pilot.yaml"):
        worker = yaml.safe_load((plan / name).read_text())["engine"]["workers"]["aggregated"]
        assert worker["kv_cache"]["prefix_caching"] is False


@pytest.mark.parametrize("version", ["0.30.0", "0.25.1"])
def test_grouped_plan_on_an_unmodeled_vllm_version_stays_cold(grouped_case, tmp_path, monkeypatch, version):
    # Replay models grouped prefix caching for vLLM 0.31.0, 0.32.0 and 0.33.0 only, so a hand-made profile of another
    # version with evidence and markers plans the cold replay that it planned before.
    profile, _root = grouped_case
    profile = _with_prefix_evidence(profile, version=version)
    plan, output = _plan_and_validate_weka(profile, tmp_path, monkeypatch, version=version, context_length=1024)
    for name in ("predict/pilot.yaml", "recommend/pilot.yaml"):
        worker = yaml.safe_load((plan / name).read_text())["engine"]["workers"]["aggregated"]
        assert worker["kv_cache"]["prefix_caching"] is False
    assert json.loads((output / "validation.json").read_text())["status"] == "covered"


@pytest.mark.parametrize("unsupported", ["state_without_align", "convolution_group"])
def test_grouped_plan_that_fails_a_prefix_caching_requirement_stays_cold(
    grouped_case, tmp_path, monkeypatch, unsupported
):
    # A hand-made profile with evidence and markers on a modeled version still plans the cold replay that it planned
    # before when the deployment fails another grouped prefix-caching requirement, and that replay runs.
    profile, _root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile))
    for deployment in profile["deployments"]:
        resources = deployment["resources"]
        if unsupported == "state_without_align":
            resources["runtime_memory"]["mamba_cache_mode"] = "none"
        else:
            resources["cache_groups"].append(dict(_CONVOLUTION_GROUP))
    plan, output = _plan_and_validate_weka(profile, tmp_path, monkeypatch, version="0.31.0", context_length=1024)
    for name in ("predict/pilot.yaml", "recommend/pilot.yaml"):
        worker = yaml.safe_load((plan / name).read_text())["engine"]["workers"]["aggregated"]
        assert worker["kv_cache"]["prefix_caching"] is False
    assert json.loads((output / "validation.json").read_text())["status"] == "covered"
    assert json.loads((output / "prediction/prediction.json").read_text())["completed_requests"] == 2


@pytest.mark.parametrize("prefix_caching", [False, True])
def test_grouped_sweeper_candidate_rank_takes_the_probe_hash_block(grouped_case, prefix_caching):
    from aisim.sweeper.config import SearchSpace
    from aisim.sweeper.deploy import build_backend_deployment
    from aisim.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
    from aisim.sweeper.sample import unroll_sample

    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile))
    raw = {**_prefix_search(profile, root, "0.31.0"), "agg_enable_prefix_caching": prefix_caching}
    sample = unroll_sample(
        search_space=SearchSpace.model_validate(raw),
        selection={
            "deployment_mode": "agg",
            "backend": "vllm",
            "agg_max_num_batched_tokens": 128,
            "agg_max_num_seqs": 4,
        },
        parallel_config=ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=2, moe_ep=1), replicas=1),
    )
    args = build_backend_deployment(sample, backend_version="0.31.0").agg_engine_args
    if prefix_caching:
        # Without a search context length, the rank takes the profile's.
        assert (args["prefix_match_unit"], args["vllm_version"], args["max_model_len"]) == (64, "0.31.0", 2048)
    else:
        assert not {"prefix_match_unit", "vllm_version", "max_model_len"} & args.keys()


@reads_saved_config
def test_grouped_prefix_caching_recommendation_replays_every_candidate(grouped_case, tmp_path, monkeypatch):
    from aisim.recommend import _run_recommendation
    from aisim.sweeper import SweepResult
    from aisim.sweeper.result import CandidateStatus

    monkeypatch.setattr("aisim.recommend.run_recommendation", _run_recommendation)
    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile))
    _write_timings(root, profile, version="0.31.0")
    raw = _prefix_caching_prediction(profile, root, "0.31.0")
    raw["engine"]["workers"]["aggregated"]["parallelism"] = {"preset": "default"}
    raw["optimization"] = {"constraints": {"max_candidate_gpus": 2}}
    raw["optimizer"] = {"algorithm": "random", "max_trials": 2, "parallelism": 1}
    path = tmp_path / "recommend.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "recommendation"
    assert cli.main(["recommend", "-c", str(path), "--output-dir", str(output)]) == 0
    result = SweepResult.from_json((output / "recommendation.json").read_text())
    # Both topologies replay. A candidate that the Rust gate refused would be failed with "replay failed".
    assert len(result.candidates) == 2
    assert all(candidate.status is CandidateStatus.FEASIBLE for candidate in result.candidates)
    exported = CorePredictionConfig.from_yaml(output / "recommendations/0001.yaml")
    assert exported.engine.workers.aggregated.kv_cache.prefix_caching is True


def test_grouped_sweeper_checks_each_candidate_budget_before_replay(grouped_case):
    from aisim.sweeper.config import SearchSpace
    from aisim.sweeper.deploy import build_backend_deployment
    from aisim.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
    from aisim.sweeper.sample import unroll_sample

    profile, root = grouped_case
    profile = _with_prefix_evidence(profile, hash_block=2176)
    profile["architecture"] = "Glm5NextForConditionalGeneration"
    # The search gate checks the largest budget, 8192. The 8190 candidate is refused when its rank is built.
    search = SearchSpace.model_validate(
        {**_prefix_search(profile, root, "0.31.0"), "agg_max_num_batched_tokens": [8190, 8192]}
    )
    sample = unroll_sample(
        search_space=search,
        selection={
            "deployment_mode": "agg",
            "backend": "vllm",
            "agg_max_num_batched_tokens": 8190,
            "agg_max_num_seqs": 4,
        },
        parallel_config=ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=2, moe_ep=1), replicas=1),
    )
    with pytest.raises(ValueError, match="max_batched_tokens that are multiples of 4"):
        build_backend_deployment(sample, backend_version="0.31.0")


def _grouped_agg_sample(profile, root, budget):
    from aisim.sweeper.config import SearchSpace
    from aisim.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
    from aisim.sweeper.sample import unroll_sample

    search = SearchSpace.model_validate(
        {**_prefix_search(profile, root, "0.31.0"), "agg_max_num_batched_tokens": [budget]}
    )
    return unroll_sample(
        search_space=search,
        selection={
            "deployment_mode": "agg",
            "backend": "vllm",
            "agg_max_num_batched_tokens": budget,
            "agg_max_num_seqs": 4,
        },
        parallel_config=ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=2, moe_ep=1), replicas=1),
    )


def test_grouped_sweeper_loads_the_profile_once_per_candidate(grouped_case, monkeypatch):
    from aisim.sweeper import deploy
    from aisim.sweeper.config import SearchSpace
    from aisim.sweeper.parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig
    from aisim.sweeper.sample import unroll_sample
    from aisim_core.fpm_profile import FpmModelProfile
    from aisim_core.sdk.fpm_profile import load_fpm_profile as load

    # Every name of the loader ends in this one validation, so the count sees a load through any of them.
    loads = []
    validate = FpmModelProfile.model_validate

    def counted(cls, value, **kwargs):
        loads.append(value)
        return validate(value, **kwargs)

    monkeypatch.setattr(FpmModelProfile, "model_validate", classmethod(counted))
    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile))
    sample = _grouped_agg_sample(profile, root, 128)
    # The search validation that makes a sample loads the profile too, so the count starts after it.
    loads.clear()
    args = deploy.build_backend_deployment(sample, backend_version="0.31.0").agg_engine_args
    # An aggregated rank with prefix caching reads the profile once.
    assert len(loads) == 1
    # A profile given as JSON text or as a loaded model builds the same rank as the dict that the search stores.
    for value in (json.dumps(sample["fpm_profile"]), load(sample["fpm_profile"])):
        assert (
            deploy.build_backend_deployment({**sample, "fpm_profile": value}, backend_version="0.31.0").agg_engine_args
            == args
        )
    # An aggregated rank without prefix caching never reads the profile. So it does not load it, and a profile that
    # the loader refuses builds the same rank.
    loads.clear()
    off = {**sample, "agg_enable_prefix_caching": False}
    off_args = deploy.build_backend_deployment(off, backend_version="0.31.0").agg_engine_args
    refused = {**off, "fpm_profile": {**sample["fpm_profile"], "context_length": 0}}
    assert deploy.build_backend_deployment(refused, backend_version="0.31.0").agg_engine_args == off_args
    assert not loads

    # A disaggregated candidate reads the profile for both roles and for the transfer size of each role, four times in
    # all, and loads it once.
    raw = {
        **_pd_search(_with_transfer_markers(profile), root),
        "prefill_enable_prefix_caching": True,
        "kv_transfer_bytes_per_token": "auto",
        "kv_transfer_bandwidth": 1.0,
    }
    worker = ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=2, moe_ep=1), replicas=1)
    sample = unroll_sample(
        search_space=SearchSpace.model_validate(raw),
        selection={
            "deployment_mode": "disagg",
            "backend": "vllm",
            "prefill_max_num_batched_tokens": 128,
            "prefill_max_num_seqs": 4,
            "decode_max_num_batched_tokens": 128,
            "decode_max_num_seqs": 4,
        },
        parallel_config=DisaggParallelConfig(prefill=worker, decode=worker),
    )
    loads.clear()
    deployment = deploy.build_backend_deployment(sample, backend_version="0.31.0")
    assert deployment.prefill_engine_args["prefix_match_unit"] == 64
    assert len(loads) == 1


def test_a_build_error_outside_the_prefix_caching_rules_stays_failed(grouped_case):
    from aisim.runner import EngineReplayRunnerFactory
    from aisim.sweeper import deploy
    from aisim.sweeper import search as search_module
    from aisim.sweeper.config import SmartSearchConfig
    from aisim.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
    from aisim.sweeper.replay import ForwardPassEstimatorSpec
    from aisim.sweeper.result import ReasonCategory

    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile))
    sample = _grouped_agg_sample(profile, root, 128)
    # The profile holds no deployment of vLLM 0.29.0. That is a fault of the inputs, not an engine setting.
    with pytest.raises(ValueError, match="no matching FPM deployment profile") as error:
        deploy.build_backend_deployment(sample, backend_version="0.29.0")
    assert not isinstance(error.value, deploy.InfeasibleEngineSetting)
    # A profile that the loader refuses is a fault of the inputs too.
    refused = {**sample, "fpm_profile": {**sample["fpm_profile"], "context_length": 0}}
    with pytest.raises(ValueError, match="context_length") as error:
        deploy.build_backend_deployment(refused, backend_version="0.31.0")
    assert not isinstance(error.value, deploy.InfeasibleEngineSetting)

    class OtherVersion:
        def resolve_candidate(self, sample):
            config = {
                "model": sample["model_name"],
                "system": "h200_sxm",
                "backend": "vllm",
                "backend_version": "0.29.0",
                "database_mode": "SILICON",
                "estimation_mode": "fpm_interpolation",
            }
            return {"agg": ForwardPassEstimatorSpec(config=config)}

    config = SmartSearchConfig(
        search_space=_prefix_search(profile, root, "0.31.0"),
        workload=Workload(isl=1152, osl=4, concurrency=1, request_count=2),
    )
    prepared, result = search_module._materialize_one(
        {"deployment_mode": "agg", "backend": "vllm", "agg_max_num_batched_tokens": 128, "agg_max_num_seqs": 4},
        ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=2, moe_ep=1), replicas=1),
        config=config,
        goal=config.goal,
        providers={},
        provider_plans={},
        runner_factory=EngineReplayRunnerFactory(),
        estimator_resolver=OtherVersion(),
    )
    assert prepared is None
    assert (result.outcome, result.reason_category) == ("failed", ReasonCategory.CANDIDATE_MATERIALIZATION)
    assert result.reason.startswith("candidate build failed: ValueError: no matching FPM deployment profile")


def test_grouped_sweeper_marks_an_off_grid_budget_infeasible_with_a_short_reason(grouped_case, caplog):
    from aisim.runner import EngineReplayRunnerFactory
    from aisim.sweeper import search as search_module
    from aisim.sweeper.config import SmartSearchConfig
    from aisim.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
    from aisim.sweeper.result import ReasonCategory

    profile, root = grouped_case
    profile = _with_prefix_evidence(profile, hash_block=2176)
    profile["architecture"] = "Glm5NextForConditionalGeneration"
    # The search gate checks the largest budget, 8192. The 8190 candidate is refused when its rank is built.
    config = SmartSearchConfig(
        search_space={**_prefix_search(profile, root, "0.31.0"), "agg_max_num_batched_tokens": [8190, 8192]},
        workload=Workload(isl=1152, osl=4, concurrency=1, request_count=2),
    )

    class NoEstimators:
        # The synthetic checkpoint names another architecture, so the exact estimator cannot be built. The candidate
        # takes the search's backend version, as a candidate without estimators does.
        def resolve_candidate(self, sample):
            return {}

    prepared, result = search_module._materialize_one(
        {"deployment_mode": "agg", "backend": "vllm", "agg_max_num_batched_tokens": 8190, "agg_max_num_seqs": 4},
        ReplicaParallelConfig(ParallelShape(tp=2, dp=1, moe_tp=2, moe_ep=1), replicas=1),
        config=config,
        goal=config.goal,
        providers={},
        provider_plans={},
        runner_factory=EngineReplayRunnerFactory(),
        estimator_resolver=NoEstimators(),
    )
    assert prepared is None
    assert (result.outcome, result.reason_category) == ("infeasible", ReasonCategory.BACKEND_TOPOLOGY)
    assert result.reason == (
        "candidate engine settings infeasible: GLM-5.3-Flash prefix caching requires prefix_match_unit and "
        "max_batched_tokens that are multiples of 4"
    )
    assert "Sweeper candidate build failed" not in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def _onboarding_plan(profile, tmp_path, capsys, version, runtime_version=None, *, worker_type=None, max_num_tokens=128):
    """Run `aisim onboard plan` for one grouped profile and return its stderr and its files, with tmp_path removed."""
    request = SupportRequest.model_validate(
        {
            **({} if worker_type is None else {"worker_type": worker_type}),
            "identity": {
                "model": profile["model"],
                "model_revision": profile["model_revision"],
                "model_kind": "moe",
                "framework_version": version,
                "runtime_framework_version": runtime_version,
                "gpu": "h200_sxm",
                "interconnect": "nvswitch",
            },
            "search": {
                "tensor_parallel": 2,
                "attention_data_parallel": 1,
                "moe_tensor_parallel": 2,
                "moe_expert_parallel": 1,
                "context_length": 1024,
            },
            "workload": {"input_tokens": 512, "output_tokens": 4},
            "collection": {"max_num_tokens": max_num_tokens, "max_batch_size": 4},
            "fpm_profile": profile,
        }
    )
    path = tmp_path / "request.yaml"
    path.write_text(yaml.safe_dump(request.model_dump(mode="json", exclude_none=True), sort_keys=False))
    plan = tmp_path / "collection"
    capsys.readouterr()
    assert cli.main(["onboard", "plan", "--config", str(path), "--output-dir", str(plan)]) == 0
    stderr = capsys.readouterr().err
    files = {
        str(item.relative_to(plan)): item.read_bytes().replace(str(tmp_path).encode(), b"<tmp>")
        for item in sorted(plan.rglob("*"))
        if item.is_file()
    }
    return stderr, files


def _not_modeled(version):
    return (
        f"Grouped prefix caching is not modeled for vLLM {version}, so the plan of this profile runs without prefix "
        "caching, although its runtime memory records prefix caching on."
    )


@reads_saved_config
def test_grouped_plan_of_a_version_without_rules_records_and_prints_that_prefix_caching_is_not_modeled(
    grouped_case, tmp_path, capsys
):
    # A hand-made profile of vLLM 0.30.0 records prefix caching on, but the rule table models vLLM 0.31.0, 0.32.0
    # and 0.33.0 only. The plan stays cold, and the plan file and the command output say why.
    profile, _root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version="0.30.0")
    stderr, files = _onboarding_plan(profile, tmp_path, capsys, "0.30.0")
    assert stderr.strip() == _not_modeled("0.30.0")
    for name in ("predict/pilot.yaml", "recommend/pilot.yaml"):
        worker = yaml.safe_load(files[name])["engine"]["workers"]["aggregated"]
        assert worker["kv_cache"]["prefix_caching"] is False
    assert _plan_reason(files) == _not_modeled("0.30.0")
    # The other plan files hold no note.
    assert all(b"not modeled" not in content for name, content in files.items() if name != "support-plan.json")


@reads_saved_config
@pytest.mark.parametrize(
    "version,probe_prefix_caching", [("0.33.0", False), ("0.33.0", True), ("0.31.0", True), ("0.32.0", True)]
)
def test_grouped_plan_prints_no_prefix_caching_note_without_probe_prefix_caching_or_on_a_modeled_version(
    grouped_case, tmp_path, capsys, version, probe_prefix_caching
):
    profile, _root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version=version)
    if not probe_prefix_caching:
        for deployment in profile["deployments"]:
            deployment["resources"]["runtime_memory"].update(enable_prefix_caching=False, mamba_cache_mode="none")
    stderr, files = _onboarding_plan(profile, tmp_path, capsys, version)
    assert "prefix caching" not in stderr
    worker = yaml.safe_load(files["predict/pilot.yaml"])["engine"]["workers"]["aggregated"]
    assert worker["kv_cache"]["prefix_caching"] is probe_prefix_caching


@reads_saved_config
@pytest.mark.parametrize("label", ["0.31.0", "0.32.0", "custom-vllm"])
def test_grouped_plan_decides_prefix_caching_by_the_observed_vllm_version(grouped_case, tmp_path, capsys, label):
    # `onboard probe-runtime` keeps an explicit dataset label and records the vLLM version that it observed as
    # runtime_framework_version. The evidence is that of vLLM 0.33.0, which has grouped rules, but the label differs
    # and replay would take the rules of the label. The plan stays cold, and the note names both versions.
    profile, _root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version=label)
    stderr, files = _onboarding_plan(profile, tmp_path, capsys, label, runtime_version="0.33.0")
    assert stderr.strip() == _label_differs("0.33.0", label)
    for name in ("predict/pilot.yaml", "recommend/pilot.yaml"):
        worker = yaml.safe_load(files[name])["engine"]["workers"]["aggregated"]
        assert worker["kv_cache"]["prefix_caching"] is False


def _label_differs(version, label):
    return (
        f"The evidence of this profile comes from vLLM {version}, but its label is {label}, and replay takes the "
        "grouped prefix-caching rules from the label, so the plan of this profile runs without prefix caching, "
        "although its runtime memory records prefix caching on."
    )


@reads_saved_config
def test_grouped_plan_of_evidence_without_rules_under_another_label_names_the_version_of_the_evidence(
    grouped_case, tmp_path, capsys
):
    # Both notes hold for evidence of vLLM 0.30.0 under the label 0.31.0. The version without rules comes first.
    profile, _root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version="0.31.0")
    stderr, files = _onboarding_plan(profile, tmp_path, capsys, "0.31.0", runtime_version="0.30.0")
    assert stderr.strip() == _plan_reason(files) == _not_modeled("0.30.0")


@reads_saved_config
@pytest.mark.parametrize("label", ["custom-vllm", "0.33.0", "0.30.0"])
def test_grouped_plan_with_modeled_evidence_under_another_label_is_cold(grouped_case, tmp_path, capsys, label):
    # The evidence is of vLLM 0.32.0, which has grouped rules, but replay would take the rules of the label. The plan
    # stays cold and says why, and so does the plan that the runtime resolution of collect-fpm writes.
    from aisim.support import versioning
    from aisim.support.schema import FPMDeployment

    profile, _root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version=label)
    for name in ("observed", "label"):
        (tmp_path / name).mkdir()
    stderr, files = _onboarding_plan(profile, tmp_path / "observed", capsys, label, runtime_version="0.32.0")
    assert stderr.strip() == _label_differs("0.32.0", label)
    for name in ("predict/pilot.yaml", "recommend/pilot.yaml"):
        worker = yaml.safe_load(files[name])["engine"]["workers"]["aggregated"]
        assert worker["kv_cache"]["prefix_caching"] is False
    # The request of the runtime resolution has no observed version yet; the step detects vLLM 0.32.0 on the image.
    stderr, _files = _onboarding_plan(profile, tmp_path / "label", capsys, label)
    request = SupportRequest.from_yaml(tmp_path / "label" / "request.yaml")
    original = versioning.detect_runtime_versions
    versioning.detect_runtime_versions = lambda deployment: {"backend_version": "0.32.0"}
    try:
        resolved, root = versioning.resolve_runtime_request(
            request, FPMDeployment(image="registry.example/image:1"), tmp_path / "label" / "collection"
        )
    finally:
        versioning.detect_runtime_versions = original
    assert resolved.identity.runtime_framework_version == "0.32.0"
    worker = yaml.safe_load((root / "predict/pilot.yaml").read_text())["engine"]["workers"]["aggregated"]
    assert worker["kv_cache"]["prefix_caching"] is False


@reads_saved_config
@pytest.mark.parametrize("version", ["0.31.0", "0.32.0", "0.33.0"])
def test_grouped_plan_with_the_evidence_version_of_its_label_keeps_prefix_caching_on(
    grouped_case, tmp_path, capsys, version
):
    profile, _root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version=version)
    for deployment in profile["deployments"]:
        deployment["resources"]["runtime_memory"]["runtime_framework_version"] = version
    stderr, files = _onboarding_plan(profile, tmp_path, capsys, version)
    assert "prefix caching" not in stderr
    for name in ("predict/pilot.yaml", "recommend/pilot.yaml"):
        worker = yaml.safe_load(files[name])["engine"]["workers"]["aggregated"]
        assert worker["kv_cache"]["prefix_caching"] is True


def _refuse_prefix_caching(*_args, **_kwargs):
    raise ValueError("an earlier planner")


def _plan_cold_like_an_earlier_planner(monkeypatch):
    """Make `create_plan` of this module write the plan that AISim wrote before replay modeled grouped prefix caching.

    That planner gave every grouped profile `prefix_caching: false`. The planner plans cold when a deployment fails the
    grouped prefix-caching requirements, so refusing them and removing the recorded reason gives the same files.
    """
    import sys

    from aisim.support import plan as plan_module

    module = sys.modules[__name__]
    original = module.create_plan

    def cold(*args, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(plan_module, "grouped_prefix_caching_requirements", _refuse_prefix_caching)
            result = original(*args, **kwargs)
        _drop_cold_reason(Path(args[1]))
        return result

    monkeypatch.setattr(module, "create_plan", cold)
    return original


def _prefix_caching_of(plan):
    return [
        yaml.safe_load((plan / name).read_text())["engine"]["workers"]["aggregated"]["kv_cache"]["prefix_caching"]
        for name in ("predict/pilot.yaml", "recommend/pilot.yaml")
    ]


@reads_saved_config
@pytest.mark.parametrize("version", ["0.31.0", "0.32.0", "0.33.0"])
def test_grouped_plan_saved_cold_by_an_earlier_planner_still_verifies(
    grouped_case, tmp_path, monkeypatch, capsys, version
):
    import sys

    from aisim.support.plan import check_plan

    profile, _root = grouped_case
    profile = _with_prefix_evidence(profile, version=version)
    original = _plan_cold_like_an_earlier_planner(monkeypatch)
    # validate-fpm verifies the saved plan before its replay and after it, and returns 0 only when both pass.
    plan, _output = _plan_and_validate_weka(profile, tmp_path, monkeypatch, version=version, context_length=1024)
    assert _prefix_caching_of(plan) == [False, False]
    config = plan / "request.yaml"
    request = SupportRequest.from_yaml(config)
    check_plan(request, plan)
    # The other commands that verify the plan stop at their next step, which needs the collector or a collection.
    from aisim.support import collection_readiness, finalization, fpm

    def reached(*_args, **_kwargs):
        raise ValueError("reached the step after the plan check")

    monkeypatch.setattr(fpm, "_resolve_execution", reached)
    monkeypatch.setattr(collection_readiness, "assess_readiness", reached)
    monkeypatch.setattr(finalization, "_verify_collection", reached)
    capsys.readouterr()
    collect = ["onboard", "collect-fpm", "--config", str(config), "--output-dir", str(plan)]
    finalize = ["onboard", "finalize", "--config", str(config), "--output-dir", str(plan)]
    for command in (
        collect,
        [*collect, "--check-readiness"],
        [*finalize, "--resolved-output-dir", str(tmp_path / "r")],
    ):
        with pytest.raises(SystemExit):
            cli.main(command)
        assert "reached the step after the plan check" in capsys.readouterr().err
    # The saved plan decides until `onboard plan --overwrite` writes the plan of the planner, with prefix caching on.
    monkeypatch.setattr(sys.modules[__name__], "create_plan", original)
    assert cli.main(["onboard", "plan", "--config", str(config), "--output-dir", str(plan), "--overwrite"]) == 0
    assert _prefix_caching_of(plan) == [True, True]
    check_plan(request, plan)


@reads_saved_config
@pytest.mark.parametrize(
    "version,change,refused",
    [
        ("0.31.0", "requests", True),
        ("0.30.0", "prefix_caching", True),
        ("0.31.0", "prefix_caching", False),
        ("0.33.0", "prefix_caching", False),
    ],
)
def test_grouped_plan_saved_cold_refuses_any_other_change(
    grouped_case, tmp_path, capsys, monkeypatch, version, change, refused
):
    from aisim.support import plan as plan_module

    profile, _root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version=version)
    monkeypatch.setattr(plan_module, "grouped_prefix_caching_requirements", _refuse_prefix_caching)
    _onboarding_plan(profile, tmp_path, capsys, version)
    monkeypatch.undo()
    plan = tmp_path / "collection"
    _drop_cold_reason(plan)
    saved = json.loads((plan / "support-plan.json").read_text())
    for name in ("predict/pilot.yaml", "recommend/pilot.yaml"):
        pilot = yaml.safe_load((plan / name).read_text())
        if change == "requests":
            # A cold plan with one more change is neither the plan of the planner nor that of the earlier planner.
            pilot["traffic"]["stop"]["requests"] += 1
        else:
            # Prefix caching on in both pilots is the plan of the planner for a version with grouped rules. A vLLM
            # 0.30.0 profile has no grouped rules and plans cold under both planners, so there it is a rendering of
            # neither.
            pilot["engine"]["workers"]["aggregated"]["kv_cache"]["prefix_caching"] = True
        # The hashes of the plan are written again, so only the comparison with the renderings can refuse it.
        data = yaml.safe_dump(pilot, sort_keys=False).encode()
        (plan / name).write_bytes(data)
        saved["generated_files_sha256"][name] = hashlib.sha256(data).hexdigest()
    (plan / "support-plan.json").write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
    request = SupportRequest.from_yaml(plan / "request.yaml")
    if not refused:
        plan_module.check_plan(request, plan)
        return
    with pytest.raises(ValueError, match="support-plan.json was modified; choose a new output directory"):
        plan_module.check_plan(request, plan)


@reads_saved_config
def test_a_grouped_vllm_0330_plan_saved_cold_still_verifies(grouped_case, tmp_path, capsys, monkeypatch):
    # Before replay had grouped rules for vLLM 0.33.0, the planner wrote this profile's plan cold, the same bytes as
    # the earlier planner writes.
    from aisim.support import plan as plan_module

    profile, _root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version="0.33.0")
    monkeypatch.setattr(plan_module, "grouped_prefix_caching_requirements", _refuse_prefix_caching)
    _onboarding_plan(profile, tmp_path, capsys, "0.33.0")
    monkeypatch.undo()
    plan = tmp_path / "collection"
    _drop_cold_reason(plan)
    request = SupportRequest.from_yaml(plan / "request.yaml")
    pilot = plan / "predict/pilot.yaml"
    before = pilot.read_bytes()
    assert _prefix_caching_of(plan) == [False, False]
    plan_module.check_plan(request, plan)
    assert pilot.read_bytes() == before
    # A changed file of the saved plan is still refused.
    pilot.write_bytes(before.replace(b"prefix_caching: false", b"prefix_caching: true"))
    with pytest.raises(ValueError, match="was modified or missing; choose a new output directory"):
        plan_module.check_plan(request, plan)
    pilot.write_bytes(before)
    # `onboard plan --overwrite` writes the plan of the planner, which turns prefix caching on.
    config = tmp_path / "request.yaml"
    assert cli.main(["onboard", "plan", "--config", str(config), "--output-dir", str(plan), "--overwrite"]) == 0
    assert _prefix_caching_of(plan) == [True, True]
    plan_module.check_plan(request, plan)


@reads_saved_config
@pytest.mark.parametrize("version", ["0.30.0", "0.25.1"])
def test_a_version_without_rules_plans_the_files_of_the_earlier_planner_and_records_why(
    grouped_case, tmp_path, capsys, monkeypatch, version
):
    # A version without grouped rules plans cold, as the earlier planner did for every grouped profile. The two plans
    # differ only in the reason that the planner records, and the plan of the earlier planner still verifies.
    from aisim.support import plan as plan_module

    profile, _root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version=version)
    stderr, files = _onboarding_plan(profile, tmp_path, capsys, version)
    assert stderr.strip() == _plan_reason(files) == _not_modeled(version)
    (tmp_path / "collection").rename(tmp_path / "planner")
    calls = []

    def earlier_planner(*args):
        # The earlier planner planned every grouped profile cold and recorded no reason.
        calls.append(args)
        return "an earlier planner"

    monkeypatch.setattr(plan_module, "_grouped_cold_reason", earlier_planner)
    _onboarding_plan(profile, tmp_path, capsys, version)
    monkeypatch.undo()
    assert calls
    plan = tmp_path / "collection"
    _drop_cold_reason(plan)
    earlier = {
        str(item.relative_to(plan)): item.read_bytes().replace(str(tmp_path).encode(), b"<tmp>")
        for item in sorted(plan.rglob("*"))
        if item.is_file()
    }
    planner = json.loads(files.pop("support-plan.json"))
    assert planner["fpm"].pop("prefix_caching_reason") == _not_modeled(version)
    assert planner == json.loads(earlier.pop("support-plan.json"))
    assert files == earlier
    worker = yaml.safe_load(files["predict/pilot.yaml"])["engine"]["workers"]["aggregated"]
    assert worker["kv_cache"]["prefix_caching"] is False
    plan_module.check_plan(SupportRequest.from_yaml(plan / "request.yaml"), plan)


_COLD = "The plan of this profile runs without prefix caching: "


def _plan_reason(files):
    return json.loads(files["support-plan.json"])["fpm"].get("prefix_caching_reason")


@reads_saved_config
@pytest.mark.parametrize(
    "case,text",
    [
        (
            "unmarked_group",
            "grouped FPM cache prefix caching requires every cache group to declare prefix_cacheable; "
            "set prefix_cacheable (true or false) on each cache group of a hand-made profile, or import the runtime "
            "probe again for a profile that was imported before the import wrote this field",
        ),
        (
            "convolution_group",
            "grouped prefix caching does not support convolution cache groups; turn prefix caching off "
            "to replay them cold",
        ),
        (
            "state_without_align",
            "grouped FPM cache prefix caching requires mamba_cache_mode=align for a prefix-cacheable state group",
        ),
    ],
)
def test_grouped_cold_plan_records_and_prints_its_reason(grouped_case, tmp_path, capsys, case, text):
    profile, _root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version="0.31.0")
    for deployment in profile["deployments"]:
        resources = deployment["resources"]
        if case == "unmarked_group":
            del resources["cache_groups"][1]["prefix_cacheable"]
        elif case == "convolution_group":
            resources["cache_groups"].append(dict(_CONVOLUTION_GROUP))
        else:
            resources["runtime_memory"]["mamba_cache_mode"] = "none"
    stderr, files = _onboarding_plan(profile, tmp_path, capsys, "0.31.0")
    assert stderr.strip() == _plan_reason(files) == _COLD + text
    worker = yaml.safe_load(files["predict/pilot.yaml"])["engine"]["workers"]["aggregated"]
    assert worker["kv_cache"]["prefix_caching"] is False


@reads_saved_config
@pytest.mark.parametrize("version,runtime_version", [("0.33.0", None), ("0.30.0", None), ("0.31.0", "0.33.0")])
def test_grouped_plan_of_a_probe_without_prefix_caching_records_the_reason_and_prints_nothing(
    grouped_case, tmp_path, capsys, version, runtime_version
):
    # The notes on the vLLM version of the evidence hold only for a probe that ran with prefix caching, so a version
    # without rules and evidence under another label record the reason of the probe too.
    profile, _root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version=version)
    for deployment in profile["deployments"]:
        deployment["resources"]["runtime_memory"].update(enable_prefix_caching=False, mamba_cache_mode="none")
    stderr, files = _onboarding_plan(profile, tmp_path, capsys, version, runtime_version)
    assert "prefix caching" not in stderr
    assert _plan_reason(files) == _COLD + "grouped FPM cache requires kv_cache.prefix_caching=false for cold replay"


@reads_saved_config
def test_grouped_warm_plan_records_no_reason(grouped_case, tmp_path, capsys):
    profile, _root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version="0.31.0")
    stderr, files = _onboarding_plan(profile, tmp_path, capsys, "0.31.0")
    assert stderr.strip() == "" and _plan_reason(files) is None


def _drop_cold_reason(plan):
    """Give a saved plan the bytes that a planner wrote before plans recorded the reason of a cold replay."""
    path = plan / "support-plan.json"
    saved = json.loads(path.read_text())
    saved["fpm"].pop("prefix_caching_reason")
    path.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")


@reads_saved_config
def test_grouped_plan_saved_before_the_reason_field_still_verifies(grouped_case, tmp_path, capsys, monkeypatch):
    from aisim.support import collection_readiness, finalization, fpm
    from aisim.support.plan import check_plan

    profile, _root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version="0.31.0")
    for deployment in profile["deployments"]:
        deployment["resources"]["runtime_memory"]["mamba_cache_mode"] = "none"
    _onboarding_plan(profile, tmp_path, capsys, "0.31.0")
    plan, config = tmp_path / "collection", tmp_path / "request.yaml"
    _drop_cold_reason(plan)
    before = (plan / "support-plan.json").read_bytes()
    request = SupportRequest.from_yaml(plan / "request.yaml")
    check_plan(request, plan)

    def reached(*_args, **_kwargs):
        raise ValueError("reached the step after the plan check")

    monkeypatch.setattr(fpm, "_resolve_execution", reached)
    monkeypatch.setattr(collection_readiness, "assess_readiness", reached)
    monkeypatch.setattr(finalization, "_verify_collection", reached)
    capsys.readouterr()
    collect = ["onboard", "collect-fpm", "--config", str(config), "--output-dir", str(plan)]
    finalize = ["onboard", "finalize", "--config", str(config), "--output-dir", str(plan)]
    for command in (
        collect,
        [*collect, "--check-readiness"],
        [*finalize, "--resolved-output-dir", str(tmp_path / "r")],
    ):
        with pytest.raises(SystemExit):
            cli.main(command)
        assert "reached the step after the plan check" in capsys.readouterr().err
    assert (plan / "support-plan.json").read_bytes() == before
    # A saved plan with another change is still refused.
    (plan / "support-plan.json").write_bytes(before.replace(b'"status": "planned"', b'"status": "changed"'))
    with pytest.raises(ValueError, match="support-plan.json was modified; choose a new output directory"):
        check_plan(request, plan)
    (plan / "support-plan.json").write_bytes(before)
    # `onboard plan --overwrite` writes the reason.
    assert cli.main(["onboard", "plan", "--config", str(config), "--output-dir", str(plan), "--overwrite"]) == 0
    assert json.loads((plan / "support-plan.json").read_text())["fpm"]["prefix_caching_reason"].startswith(_COLD)


def _set_saved_reason(plan, value):
    """Write a saved plan whose recorded reason is ``value``; every other byte is that of the planner."""
    path = plan / "support-plan.json"
    saved = json.loads(path.read_text())
    saved["fpm"]["prefix_caching_reason"] = value
    path.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")


def _unmarked_cold_plan(grouped_case, tmp_path, capsys):
    profile, _root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version="0.31.0")
    for deployment in profile["deployments"]:
        del deployment["resources"]["cache_groups"][1]["prefix_cacheable"]
    _onboarding_plan(profile, tmp_path, capsys, "0.31.0")
    plan = tmp_path / "collection"
    return plan, SupportRequest.from_yaml(plan / "request.yaml")


@reads_saved_config
@pytest.mark.parametrize(
    "text",
    [
        # Two other wordings of this requirement, and any other text.
        _COLD + "grouped FPM cache prefix caching requires every cache group to declare prefix_cacheable; "
        "set prefix_cacheable (true or false) on each cache group of the profile",
        _COLD + "grouped FPM cache prefix caching requires every cache group to declare prefix_cacheable; "
        "set prefix_cacheable (true or false) on each cache group of a hand-made profile, or import the runtime "
        "probe again for a profile that the import wrote",
        "Another text.",
        "",
    ],
)
def test_grouped_cold_plan_verifies_with_the_reason_text_that_it_was_saved_with(
    grouped_case, tmp_path, capsys, monkeypatch, text
):
    # The recorded reason explains why the plan is cold. The plan check compares the decision, not these words, so a
    # later change of a requirement message does not refuse saved work.
    from aisim.support import collection_readiness, finalization
    from aisim.support.plan import check_plan

    plan, request = _unmarked_cold_plan(grouped_case, tmp_path, capsys)
    _set_saved_reason(plan, text)
    before = (plan / "support-plan.json").read_bytes()
    check_plan(request, plan)

    def reached(*_args, **_kwargs):
        raise ValueError("reached the step after the plan check")

    monkeypatch.setattr(collection_readiness, "assess_readiness", reached)
    monkeypatch.setattr(finalization, "_verify_collection", reached)
    config = tmp_path / "request.yaml"
    collect = ["onboard", "collect-fpm", "--config", str(config), "--output-dir", str(plan), "--check-readiness"]
    finalize = ["onboard", "finalize", "--config", str(config), "--output-dir", str(plan)]
    for command in (collect, [*finalize, "--resolved-output-dir", str(tmp_path / "r")]):
        capsys.readouterr()
        with pytest.raises(SystemExit):
            cli.main(command)
        assert "reached the step after the plan check" in capsys.readouterr().err
    assert (plan / "support-plan.json").read_bytes() == before
    # `onboard plan --overwrite` writes the text of the planner.
    assert cli.main(["onboard", "plan", "--config", str(config), "--output-dir", str(plan), "--overwrite"]) == 0
    saved = json.loads((plan / "support-plan.json").read_text())
    assert saved["fpm"]["prefix_caching_reason"] == _COLD + _REFUSAL_TEXT["unmarked_group"]


@reads_saved_config
@pytest.mark.parametrize("value", [None, 0, 1.5, True, False, ["text"], {"text": "text"}])
def test_grouped_cold_plan_with_a_recorded_reason_that_is_not_a_string_is_refused(
    grouped_case, tmp_path, capsys, value
):
    from aisim.support.plan import check_plan

    plan, request = _unmarked_cold_plan(grouped_case, tmp_path, capsys)
    _set_saved_reason(plan, value)
    with pytest.raises(ValueError, match="support-plan.json was modified; choose a new output directory"):
        check_plan(request, plan)


@reads_saved_config
@pytest.mark.parametrize("fpm", [["text"], "text", None])
def test_grouped_cold_plan_whose_fpm_block_is_not_an_object_is_refused(grouped_case, tmp_path, capsys, fpm):
    from aisim.support.plan import check_plan

    plan, request = _unmarked_cold_plan(grouped_case, tmp_path, capsys)
    path = plan / "support-plan.json"
    saved = json.loads(path.read_text())
    if fpm is None:
        del saved["fpm"]
    else:
        saved["fpm"] = fpm
    path.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="support-plan.json was modified; choose a new output directory"):
        check_plan(request, plan)


@reads_saved_config
@pytest.mark.parametrize("worker_type", [None, "decode"])
def test_grouped_plan_saved_cold_with_its_reason_verifies_when_a_later_release_plans_it_warm(
    grouped_case, tmp_path, capsys, monkeypatch, worker_type
):
    # A later release that models one more vLLM version plans this profile warm. A plan that was saved cold, with
    # its reason or without it, keeps verifying until `onboard plan --overwrite` writes it again.
    from aisim.config import engine
    from aisim.support import plan as plan_module

    profile, _root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version="0.30.0")
    for deployment in profile["deployments"]:
        if worker_type is not None:
            deployment["worker_type"] = worker_type
    stderr, files = _onboarding_plan(profile, tmp_path, capsys, "0.30.0", worker_type=worker_type)
    assert stderr.strip() == _plan_reason(files) == _not_modeled("0.30.0")
    plan = tmp_path / "collection"
    request = SupportRequest.from_yaml(plan / "request.yaml")
    with_reason = (plan / "support-plan.json").read_bytes()
    later = ("0.30.0", *engine.GROUPED_PREFIX_CACHING_VERSIONS)
    monkeypatch.setattr(engine, "GROUPED_PREFIX_CACHING_VERSIONS", later)
    monkeypatch.setattr(engine, "GROUPED_PD_VERSIONS", later)
    monkeypatch.setattr(plan_module, "GROUPED_PREFIX_CACHING_VERSIONS", later)
    # The stand-in for the later release plans the profile warm.
    current, documents = plan_module._plan_documents(request, plan)
    assert "prefix_caching_reason" not in current["fpm"]
    pilot = "worker.yaml" if worker_type else "predict/pilot.yaml"
    worker = yaml.safe_load(documents[Path(pilot)])["engine"]["workers"][worker_type or "aggregated"]
    assert worker["kv_cache"]["prefix_caching"] is True
    plan_module.check_plan(request, plan)
    _drop_cold_reason(plan)
    plan_module.check_plan(request, plan)
    # The pilots of the saved plan stay cold; a changed pilot is still refused.
    (plan / "support-plan.json").write_bytes(with_reason)
    pilot_bytes = (plan / pilot).read_bytes()
    assert b"prefix_caching: false" in pilot_bytes
    (plan / pilot).write_bytes(pilot_bytes.replace(b"prefix_caching: false", b"prefix_caching: true"))
    with pytest.raises(ValueError, match="was modified or missing; choose a new output directory"):
        plan_module.check_plan(request, plan)
    (plan / pilot).write_bytes(pilot_bytes)
    config = tmp_path / "request.yaml"
    assert cli.main(["onboard", "plan", "--config", str(config), "--output-dir", str(plan), "--overwrite"]) == 0
    assert "prefix_caching_reason" not in json.loads((plan / "support-plan.json").read_text())["fpm"]


@reads_saved_config
@pytest.mark.parametrize("reason", [None, "Another text."])
def test_a_plan_of_a_warm_profile_whose_pilots_were_saved_cold_verifies_with_or_without_a_reason(
    grouped_case, tmp_path, capsys, reason
):
    # Cold pilots for a profile that plans warm are the plan of an earlier planner, which the plan check accepts. Such
    # a plan verifies without a recorded reason as an earlier planner wrote it, and with one as a planner that records
    # its reason wrote it before replay modeled prefix caching for the profile; a reason on warm pilots is refused.
    from aisim.support.plan import check_plan

    profile, _root = grouped_case
    _onboarding_plan(_with_prefix_evidence(_with_state_group(profile), version="0.31.0"), tmp_path, capsys, "0.31.0")
    plan = tmp_path / "collection"
    request = SupportRequest.from_yaml(plan / "request.yaml")
    path = plan / "support-plan.json"
    saved = json.loads(path.read_text())
    if reason is not None:
        saved["fpm"]["prefix_caching_reason"] = reason
        path.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
        with pytest.raises(ValueError, match="support-plan.json was modified; choose a new output directory"):
            check_plan(request, plan)
    for name in ("predict/pilot.yaml", "recommend/pilot.yaml"):
        pilot = yaml.safe_load((plan / name).read_text())
        pilot["engine"]["workers"]["aggregated"]["kv_cache"]["prefix_caching"] = False
        data = yaml.safe_dump(pilot, sort_keys=False).encode()
        (plan / name).write_bytes(data)
        saved["generated_files_sha256"][name] = hashlib.sha256(data).hexdigest()
    path.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
    check_plan(request, plan)


def _leaves(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaves(item, (*path, index))
    else:
        yield path, value


def _edited(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if value is None:
        return "edited"
    return value + " edited"


@reads_saved_config
def test_grouped_cold_plan_with_another_reason_text_still_compares_every_other_field(grouped_case, tmp_path, capsys):
    # Each edit of another field is refused with the same error whether the plan holds its own reason text or another.
    from aisim.support.plan import check_plan

    plan, request = _unmarked_cold_plan(grouped_case, tmp_path, capsys)
    path = plan / "support-plan.json"
    own = path.read_bytes()
    check_plan(request, plan)
    _set_saved_reason(plan, "Another text.")
    check_plan(request, plan)
    edits = 0
    for location, value in _leaves(json.loads(own)):
        if location == ("fpm", "prefix_caching_reason"):
            continue
        errors = []
        for text in (None, "Another text."):
            saved = json.loads(own)
            if text is not None:
                saved["fpm"]["prefix_caching_reason"] = text
            parent = saved
            for key in location[:-1]:
                parent = parent[key]
            parent[location[-1]] = _edited(value)
            path.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
            with pytest.raises(ValueError) as error:
                check_plan(request, plan)
            errors.append(str(error.value))
        assert errors[0] == errors[1], location
        edits += 1
    path.write_bytes(own)
    assert edits > 30


@reads_saved_config
@pytest.mark.parametrize("kind", ["warm", "linear", "no_profile"])
def test_a_recorded_reason_on_a_plan_that_records_none_is_refused(grouped_case, tmp_path, capsys, kind):
    from aisim.support.plan import check_plan

    profile, _root = grouped_case
    if kind == "warm":
        _onboarding_plan(
            _with_prefix_evidence(_with_state_group(profile), version="0.31.0"), tmp_path, capsys, "0.31.0"
        )
    elif kind == "linear":
        _onboarding_plan(_mixed_profile(profile), tmp_path, capsys, "0.25.1")
    else:
        request = SupportRequest.model_validate(
            {
                "identity": {
                    "model": "fictional/no-profile-model",
                    "model_revision": "synthetic-v1",
                    "model_kind": "moe",
                    "framework_version": "0.31.0",
                    "gpu": "h200_sxm",
                    "interconnect": "nvswitch",
                },
                "search": {"tensor_parallel": 2, "moe_tensor_parallel": 2, "context_length": 1024},
                "workload": {"input_tokens": 512, "output_tokens": 4},
            }
        )
        create_plan(request, tmp_path / "collection")
    plan = tmp_path / "collection"
    request = SupportRequest.from_yaml(plan / "request.yaml")
    check_plan(request, plan)
    assert "prefix_caching_reason" not in json.loads((plan / "support-plan.json").read_text())["fpm"]
    _set_saved_reason(plan, _COLD + "Another text.")
    with pytest.raises(ValueError, match="support-plan.json was modified; choose a new output directory"):
        check_plan(request, plan)


# The texts of grouped_prefix_caching_requirements for the two refusals.
_REFUSAL_TEXT = {
    "unmarked_group": (
        "grouped FPM cache prefix caching requires every cache group to declare prefix_cacheable; "
        "set prefix_cacheable (true or false) on each cache group of a hand-made profile, or import the runtime probe "
        "again for a profile that was imported before the import wrote this field"
    ),
    "non_cacheable_window": (
        "kv_cache_groups with enable_prefix_caching=true do not model a sliding-window attention group with "
        "prefix_cacheable=false; describe a ring buffer as a state group with blocks_per_request: 1 and "
        "prefix_cacheable: false, and turn prefix caching off for a bounded replay window, which replay does not "
        "model"
    ),
}


@reads_saved_config
@pytest.mark.parametrize("case", ["unmarked_group", "non_cacheable_window"])
@pytest.mark.parametrize("version", ["0.31.0", "0.32.0", "0.33.0"])
def test_the_marker_and_window_refusals_hold_on_each_version(grouped_case, tmp_path, capsys, case, version):
    from aisim.sweeper.config import SearchSpace

    error = {
        "unmarked_group": "every cache group to declare prefix_cacheable",
        "non_cacheable_window": "do not model a sliding-window attention group with prefix_cacheable=false",
    }[case]
    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version=version)
    for deployment in profile["deployments"]:
        groups = deployment["resources"]["cache_groups"]
        if case == "unmarked_group":
            del groups[1]["prefix_cacheable"]
        else:
            # The window group of the fixture, as the Rust rule table refuses it.
            assert groups[1]["kind"] == "attention" and groups[1]["sliding_window"]
            groups[1]["prefix_cacheable"] = False
    with pytest.raises(ValidationError, match=error):
        CorePredictionConfig.model_validate(_prefix_caching_prediction(profile, root, version))
    with pytest.raises(ValidationError, match=error):
        SearchSpace.model_validate(_prefix_search(profile, root, version))
    stderr, files = _onboarding_plan(profile, tmp_path, capsys, version)
    assert stderr.strip() == _plan_reason(files) == _COLD + _REFUSAL_TEXT[case]
    assert _prefix_caching_of(tmp_path / "collection") == [False, False]


@reads_saved_config
@pytest.mark.parametrize(
    "role,case,text",
    [
        ("prefill", "unmarked_group", _REFUSAL_TEXT["unmarked_group"]),
        ("decode", "unmarked_group", _REFUSAL_TEXT["unmarked_group"]),
        # A prefill or decode rank refuses the window group with prefix caching off too, so its text names no such
        # remedy.
        ("prefill", "non_cacheable_window", _PD_REPLAY_WINDOW),
        ("decode", "non_cacheable_window", _PD_REPLAY_WINDOW),
        # An explicit aggregated worker keeps the texts of an aggregated rank.
        ("aggregated", "non_cacheable_window", _REFUSAL_TEXT["non_cacheable_window"]),
        # A role of a profile that passes every requirement keeps prefix caching on.
        ("prefill", "marked", None),
        ("decode", "marked", None),
    ],
)
def test_grouped_plan_of_one_worker_type_records_and_prints_its_reason_or_stays_warm(
    grouped_case, tmp_path, capsys, role, case, text
):
    profile, _root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version="0.31.0")
    for deployment in profile["deployments"]:
        deployment["worker_type"] = role
        groups = deployment["resources"]["cache_groups"]
        if case == "unmarked_group":
            del groups[1]["prefix_cacheable"]
        elif case == "non_cacheable_window":
            groups[1]["prefix_cacheable"] = False
    stderr, files = _onboarding_plan(profile, tmp_path, capsys, "0.31.0", worker_type=role)
    pilot = "predict/pilot.yaml" if role == "aggregated" else "worker.yaml"
    worker = yaml.safe_load(files[pilot])["engine"]["workers"][role]
    if text is None:
        assert stderr.strip() == "" and _plan_reason(files) is None
        assert worker["kv_cache"]["prefix_caching"] is True
        return
    assert stderr.strip() == _plan_reason(files) == _COLD + text
    assert worker["kv_cache"]["prefix_caching"] is False


@reads_saved_config
def test_grouped_glm_plan_with_a_token_budget_off_the_grid_records_and_prints_its_reason(
    grouped_case, tmp_path, capsys
):
    # The GLM-5.3-Flash requirement reads the token budget of the plan, so the note must use the same budget.
    profile, _root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version="0.33.0")
    profile["architecture"] = "Glm5NextForConditionalGeneration"
    (Path(profile["model"]) / "config.json").write_text(json.dumps({"architectures": [profile["architecture"]]}))
    for deployment in profile["deployments"]:
        deployment["resources"]["max_num_tokens"] = 126
    stderr, files = _onboarding_plan(profile, tmp_path, capsys, "0.33.0", max_num_tokens=126)
    assert (
        stderr.strip()
        == _plan_reason(files)
        == _COLD
        + ("GLM-5.3-Flash prefix caching requires prefix_match_unit and max_batched_tokens that are multiples of 4")
    )
    worker = yaml.safe_load(files["predict/pilot.yaml"])["engine"]["workers"]["aggregated"]
    assert worker["kv_cache"]["prefix_caching"] is False


@pytest.mark.parametrize("label,recorded", [("0.32.0", "0.33.0"), ("0.33.0", "0.32.0")])
def test_grouped_prefix_caching_refuses_evidence_of_another_vllm_version(grouped_case, label, recorded):
    # A profile that records the evidence of one vLLM version under the label of another cannot replay with the rules
    # of the label.
    from aisim.sweeper.config import SearchSpace

    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version=label)
    CorePredictionConfig.model_validate(_prefix_caching_prediction(profile, root, label))
    SearchSpace.model_validate(_prefix_search(profile, root, label))
    for deployment in profile["deployments"]:
        deployment["resources"]["runtime_memory"]["runtime_framework_version"] = recorded
    error = (
        rf"grouped FPM cache prefix caching requires runtime_memory.runtime_framework_version \({recorded}\) to equal "
        rf"the deployment's backend_version \({label}\)"
    )
    with pytest.raises(ValidationError, match=error):
        CorePredictionConfig.model_validate(_prefix_caching_prediction(profile, root, label))
    with pytest.raises(ValidationError, match=error):
        SearchSpace.model_validate(_prefix_search(profile, root, label))
    # With prefix caching off, the profile replays cold as before.
    cold = _prefix_caching_prediction(profile, root, label)
    cold["engine"]["workers"]["aggregated"]["kv_cache"] = {"prefix_caching": False}
    CorePredictionConfig.model_validate(cold)


_INTERVAL_TEXT = (
    "grouped prefix caching on vLLM 0.33.0 requires prefill_schedule_interval=1; vLLM 0.33.0 holds prefills at "
    "attention DP size 1 too, and replay does not model that"
)


@pytest.mark.parametrize("version,interval,refused", [("0.33.0", 1, False), ("0.33.0", 2, True), ("0.32.0", 2, False)])
def test_grouped_prefix_caching_on_vllm_0330_refuses_a_prefill_schedule_interval_above_one(
    grouped_case, version, interval, refused
):
    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version=version)
    raw = _prefix_caching_prediction(profile, root, version)
    raw["engine"]["workers"]["aggregated"]["scheduler"]["prefill_schedule_interval"] = interval
    if not refused:
        CorePredictionConfig.model_validate(raw)
        return
    with pytest.raises(ValidationError, match=_INTERVAL_TEXT):
        CorePredictionConfig.model_validate(raw)
    # With prefix caching off the rank does not read the version, so the cold replay keeps the interval.
    raw["engine"]["workers"]["aggregated"]["kv_cache"] = {"prefix_caching": False}
    CorePredictionConfig.model_validate(raw)


@reads_saved_config
@pytest.mark.parametrize("label,recorded", [("0.32.0", "0.33.0"), ("0.33.0", "0.32.0")])
def test_grouped_plan_with_recorded_evidence_of_another_version_is_cold(
    grouped_case, tmp_path, capsys, label, recorded
):
    profile, _root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version=label)
    for deployment in profile["deployments"]:
        deployment["resources"]["runtime_memory"]["runtime_framework_version"] = recorded
    stderr, files = _onboarding_plan(profile, tmp_path, capsys, label)
    assert stderr.strip() == _label_differs(recorded, label)
    for name in ("predict/pilot.yaml", "recommend/pilot.yaml"):
        worker = yaml.safe_load(files[name])["engine"]["workers"]["aggregated"]
        assert worker["kv_cache"]["prefix_caching"] is False


@reads_saved_config
def test_grouped_vllm_0330_plan_with_an_unmarked_window_group_is_cold_and_prints_the_missing_marker(
    grouped_case, tmp_path, capsys
):
    # The probe import writes no prefix_cacheable for a group with a SlidingWindowMLASpec layer. The version has
    # rules and equals the label, so the note names the missing marker, not the version.
    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version="0.33.0")
    for deployment in profile["deployments"]:
        window = next(group for group in deployment["resources"]["cache_groups"] if group.get("sliding_window"))
        del window["prefix_cacheable"]
    stderr, files = _onboarding_plan(profile, tmp_path, capsys, "0.33.0")
    assert stderr.strip() == _plan_reason(files) == _COLD + _REFUSAL_TEXT["unmarked_group"]
    for name in ("predict/pilot.yaml", "recommend/pilot.yaml"):
        worker = yaml.safe_load(files[name])["engine"]["workers"]["aggregated"]
        assert worker["kv_cache"]["prefix_caching"] is False
    with pytest.raises(ValidationError, match="every cache group to declare prefix_cacheable"):
        CorePredictionConfig.model_validate(_prefix_caching_prediction(profile, root, "0.33.0"))


def test_grouped_vllm_0330_profile_without_checkpoint_fields_is_refused_by_native_replay(grouped_case):
    # The Python gates do not read the checkpoint fields, so the prediction validates. The rule table refuses the
    # rank with the text of its own row.
    from aisim.runner import EngineReplayRunnerFactory

    profile, root = grouped_case
    profile = _with_prefix_evidence(_with_state_group(profile), version="0.33.0")
    for deployment in profile["deployments"]:
        for group in deployment["resources"]["cache_groups"]:
            group.pop("prefill_checkpoint_blocks", None)
    _write_timings(root, profile, version="0.33.0")
    config = CorePredictionConfig.model_validate(_prefix_caching_prediction(profile, root, "0.33.0"))
    error = "kv_cache_groups with vllm_version=0.33.0 require prefill_checkpoint_blocks on every prefix-cacheable state"
    with pytest.raises(RuntimeError, match=error):
        EngineReplayRunnerFactory().create(0).run(prediction_to_replay_spec(config))


_GLM_PAGE = 13_045_120


def _glm_shaped_profile(profile, version):
    """A hand-made profile of vLLM 0.31.0, 0.32.0 or 0.33.0 with six GLM-5.3-Flash TP 4 cache groups and 64 blocks.

    The kinds, blocks and layer counts of the groups are those that the vLLM 0.33.0 runtime probe records (fixture
    glm53flash_vllm_0330_probe): MLA and indexer attention of 22 layers and four KDA MambaSpec states of 9, 9, 8 and 8
    layers with 2,176-token blocks, and a CircularBufferSpec ring of 11 layers with 4-token blocks. The prefix-caching
    evidence equals the probe's: prefix caching on, align mode and a hash block of 2,176 tokens. The page size, the pool
    and the group order are hand-made. The state groups keep prefill_checkpoint_blocks 0, which every rule row accepts.
    The page is the sum of the pages of the attention layers, 13,045,120 bytes, the block row of vLLM 0.31.0.
    vLLM 0.32.0 rounds the row up to the least common multiple of the block-stride alignments of all layers (v0.32.0
    vllm/v1/core/kv_cache_utils.py:1533-1551): 13,516,800 bytes for this model at TP 4 on GB300, 25 times the least
    common multiple of the 16,384-byte alignment of the MLA layers and the 4,224-byte alignment of the indexer layers.
    The fixture keeps the page of the attention layers. The tests count blocks and tokens, which do not depend on the
    page.
    """
    profile = deepcopy(profile)
    profile["architecture"] = "Glm5NextForConditionalGeneration"
    profile["context_length"] = 32768
    groups = [
        {
            "name": "runtime_group_0",
            "kind": "attention",
            "num_layers": 22,
            "block_size_tokens": 2176,
            "page_size_bytes": _GLM_PAGE,
            "prefix_cacheable": True,
        },
        {
            "name": "runtime_group_1",
            "kind": "state",
            "num_layers": 11,
            "block_size_tokens": 4,
            "page_size_bytes": _GLM_PAGE,
            "blocks_per_request": 1,
            "prefix_cacheable": False,
        },
    ]
    for index, layers in ((2, 9), (3, 9), (4, 8), (5, 8)):
        groups.append(
            {
                "name": f"runtime_group_{index}",
                "kind": "state",
                "num_layers": layers,
                "block_size_tokens": 2176,
                "page_size_bytes": _GLM_PAGE,
                "blocks_per_request": 1,
                "prefix_cacheable": True,
                "prefill_checkpoint_blocks": 0,
            }
        )
    for deployment in profile["deployments"]:
        deployment["backend_version"] = version
        resources = deployment["resources"]
        for name in ("weights_bytes", "activations_bytes", "runtime_overhead_bytes", "comm_overhead_bytes"):
            resources.pop(name)
        resources.update(cache_groups=deepcopy(groups), max_num_tokens=8192, max_batch_size=4)
        resources["runtime_memory"] = {
            "kv_cache_bytes": 64 * _GLM_PAGE,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 32768,
            "provenance": "Synthetic pool of 64 GLM-5.3-Flash pages. Not a probe result.",
            "enable_prefix_caching": True,
            "mamba_cache_mode": "align",
            "hash_block_size_tokens": 2176,
        }
    return profile


@pytest.mark.parametrize("version", ["0.31.0", "0.32.0", "0.33.0"])
def test_glm_shaped_grouped_profile_replays_a_multi_turn_trace_with_hits(grouped_case, tmp_path, monkeypatch, version):
    # A trace has no known host memory peak, so the CLI replays it only in a supervised child. Model the
    # supervisor context of that child, as test_unified_cli.py does.
    monkeypatch.setenv(
        "_AISIM_SUPERVISED_BUDGET",
        json.dumps(
            {
                "supervisor_pid": os.getpid(),
                "memory_limit_bytes": 4_000_000_000,
                "cpu_limit": 1,
                "reserved_host_memory_bytes": 1_000_000_000,
                "events_path": str(tmp_path / "events.jsonl"),
                "ready_path": str(tmp_path / "ready"),
            }
        ),
    )
    profile, root = grouped_case
    profile = _glm_shaped_profile(profile, version)
    (Path(profile["model"]) / "config.json").write_text(json.dumps({"architectures": [profile["architecture"]]}))
    import aisim_core

    # The real H200 memory holds the 64 GLM pages that the tiny fake GPU of grouped_case cannot.
    system = (Path(aisim_core.__file__).parent / "systems/h200_sxm.yaml").read_text()
    (root / "h200_sxm.yaml").write_text(system)
    _write_timings(
        root,
        profile,
        decode_ceiling=16384,
        version=version,
        prefill_tokens=(2048, 4096, 8192),
        prefill_kvs=(0, 1, 8192, 16384),
    )
    # One conversation of three turns, each resending the earlier prompt. A trace block of 2,176 tokens fills
    # block i with hash id i, so the turns share the blocks of the Rust test
    # grouped_glm_shaped_multi_turn_replay_reports_hit_tokens_and_block_counts. A fourth turn shares only the
    # first three blocks, so it branches below every state that the earlier turns keep.
    trace = tmp_path / "turns.jsonl"
    turns = [
        (9000, [1, 2, 3, 4, 5]),
        (13000, [1, 2, 3, 4, 5, 6]),
        (13100, [1, 2, 3, 4, 5, 6, 7]),
        (8704, [1, 2, 3, 8]),
    ]
    trace.write_text(
        "".join(
            json.dumps({"timestamp": 1000 * index, "input_length": length, "output_length": 4, "hash_ids": hashes})
            + "\n"
            for index, (length, hashes) in enumerate(turns)
        )
    )
    raw = _prediction(profile, root)
    raw["engine"]["backend_version"] = version
    raw["engine"]["context_length"] = 32768
    worker = raw["engine"]["workers"]["aggregated"]
    worker["scheduler"] = {"max_batched_tokens": 8192, "max_sequences": 4}
    worker["kv_cache"]["prefix_caching"] = True
    raw["traffic"] = {
        "source": {"type": "trace", "paths": [str(trace)], "format": "mooncake", "block_size": 2176},
        "load": {"type": "concurrency", "concurrency": 1},
    }
    path = tmp_path / "predict.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "prediction"
    assert cli.main(["predict", "-c", str(path), "--output-dir", str(output), "--capture-per-request"]) == 0
    report = json.loads((output / "prediction.json").read_text())
    assert report["completed_requests"] == 4
    # Turn two hits the state that turn one keeps at (9000 - 1) / 2176 * 2176 = 8704, and turn three the one that
    # turn two keeps at (13000 - 1) / 2176 * 2176 = 10880. Every version reuses 19,584 tokens. Turn four finds
    # 6,528 tokens of cached full-attention blocks, which a linear pool would reuse. Under retention 0 no state is
    # kept at 6,528 or below, so the grouped path reuses nothing.
    assert [row["reused_input_tokens"] for row in report["per_request"]] == [0, 8704, 10880, 0]
    assert report["prefix_cache_reused_ratio"] == pytest.approx((8704 + 10880) / (9000 + 13000 + 13100 + 8704))


# The bytes that vLLM's NIXL connector reads for one block of each GLM-5.3-Flash group at TP 4 on one rank: attention,
# ring and four KDA states. A live run measured the attention value and the sum of the other five groups
# (benchmarks/evidence/nixl-pd-transfer-bytes-20261115). The split of that sum among the groups is a reading of the
# connector code. On vLLM 0.33.0 a log of each group measured the six values
# (benchmarks/evidence/nixl-pd-transfer-bytes-20261117).
_GLM_TRANSFER_BYTES = {
    "0.31.0": (13_045_120, 13_045_120, 9_768_960, 9_768_960, 9_768_960, 9_768_960),
    "0.32.0": (21_978_496, 92_288, 9_768_960, 9_768_960, 9_768_960, 9_768_960),
    "0.33.0": (21_978_496, 92_288, 9_768_960, 9_768_960, 9_768_960, 9_768_960),
}


def _glm_pd_transfer_ms(profile, root, tmp_path, version):
    """The mean TTFT of one 1,000-token P/D request with a transfer at 1 GB/s, less its TTFT without a bandwidth."""
    (Path(profile["model"]) / "config.json").write_text(json.dumps({"architectures": [profile["architecture"]]}))
    import aisim_core

    # The real H200 memory holds the 64 GLM pages that the tiny fake GPU of grouped_case cannot.
    system = (Path(aisim_core.__file__).parent / "systems/h200_sxm.yaml").read_text()
    (root / "h200_sxm.yaml").write_text(system)
    _write_timings(
        root,
        profile,
        decode_ceiling=16384,
        version=version,
        prefill_tokens=(2048, 4096, 8192),
        prefill_kvs=(0, 1, 8192, 16384),
    )
    mean_ttft = {}
    for bandwidth in (None, 1.0):
        raw = _pd_prediction(profile, root, bandwidth=bandwidth)
        raw["engine"]["context_length"] = 32768
        for worker in raw["engine"]["workers"].values():
            worker["scheduler"] = {"max_batched_tokens": 8192, "max_sequences": 4}
        raw["traffic"] = {
            "source": {"type": "synthetic", "input_tokens": 1000, "output_tokens": 2},
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 1},
        }
        path = tmp_path / f"predict-{bandwidth}.yaml"
        path.write_text(yaml.safe_dump(raw))
        output = tmp_path / f"prediction-{bandwidth}"
        assert cli.main(["predict", "-c", str(path), "--output-dir", str(output)]) == 0
        mean_ttft[bandwidth] = json.loads((output / "prediction.json").read_text())["mean_ttft_ms"]
    return mean_ttft[1.0] - mean_ttft[None]


@pytest.mark.parametrize("version", ["0.31.0", "0.32.0", "0.33.0"])
def test_glm_shaped_grouped_pd_transfer_time_matches_the_oracle(grouped_case, tmp_path, version):
    profile, root = grouped_case
    profile = _glm_shaped_profile(profile, version)
    for deployment in profile["deployments"]:
        for group, transfer_bytes in zip(
            deployment["resources"]["cache_groups"], _GLM_TRANSFER_BYTES[version], strict=True
        ):
            group["transfer_bytes_per_block"] = transfer_bytes
    # The prefill computes 999 tokens, so one 2,176-token attention block moves with one block of each state group:
    # the measured row of 1,000 tokens, 65,166,080 bytes on vLLM 0.31.0 and 61,146,624 on vLLM 0.32.0 and 0.33.0, over
    # 1 GB/s.
    measured_bytes = {"0.31.0": 65_166_080, "0.32.0": 61_146_624, "0.33.0": 61_146_624}[version]
    assert _glm_pd_transfer_ms(profile, root, tmp_path, version) == pytest.approx(measured_bytes / 1e6, abs=1e-3)


_EVIDENCE = Path(__file__).resolve().parents[1] / "benchmarks/evidence"
_EVIDENCE_0320 = "nixl-pd-transfer-bytes-20261115"
_EVIDENCE_0330 = "nixl-pd-transfer-bytes-20261117"


def _evidence_versions(folder):
    data = json.loads((_EVIDENCE / folder / "measurements.json").read_text())
    assert data["schema"] == "nixl-pd-transfer-bytes/v2"
    assert set(data["marks"]) == {"measured", "derived", "unmarked"}
    return data["versions"]


def _evidence_rows(block):
    return block["rows"] if "rows" in block else [row for arm in block["arms"].values() for row in arm["rows"]]


@pytest.mark.parametrize(
    "folder,version,block_bytes,fixed_bytes",
    [
        (_EVIDENCE_0320, "0.31.0", 13_045_120, 52_120_960),
        (_EVIDENCE_0320, "0.32.0", 21_978_496, 39_168_128),
        (_EVIDENCE_0330, "0.33.0", 21_978_496, 39_168_128),
    ],
)
def test_nixl_transfer_evidence_marks_every_number_and_keeps_every_row_on_the_line(
    folder, version, block_bytes, fixed_bytes
):
    block = _evidence_versions(folder)[version]
    sources = block["sources"]
    assert set(sources) - {"region_sums"} == {"regions", "bytes_per_block", "formula", "rows"}
    assert set(sources["bytes_per_block"]) == set(block["bytes_per_block"])
    marks = [sources["regions"], sources["formula"], *sources["bytes_per_block"].values(), *sources["rows"].values()]
    assert set(marks) <= {"measured", "derived"}
    assert (block["bytes_per_block"]["attention"], sources["bytes_per_block"]["attention"]) == (block_bytes, "measured")
    rows = _evidence_rows(block)
    assert set().union(*rows) - {"prompt", "prompt_tokens"} == set(sources["rows"])
    for row in rows:
        assert row["bytes_per_rank"] == block_bytes * row["attention_blocks_moved"] + fixed_bytes


def test_nixl_transfer_evidence_of_vllm_0310_and_0320_marks_the_split_as_derived():
    split = {"attention": "measured", "ring": "derived", "kda_9_layers": "derived", "kda_8_layers": "derived"}
    for version, block in _evidence_versions(_EVIDENCE_0320).items():
        assert block["sources"]["bytes_per_block"] == split, version
        assert block["sources"]["region_sums"] == "derived", version
        rows = block["sources"]["rows"]
        assert (rows["transferred_tokens"], rows["prefill_attention_blocks"]) == ("measured", "measured"), version


def test_nixl_transfer_evidence_of_vllm_0330_measures_every_group():
    block = _evidence_versions(_EVIDENCE_0330)["0.33.0"]
    groups = {"attention": 21_978_496, "ring": 92_288, **{f"kda_group_{index}": 9_768_960 for index in range(4)}}
    assert block["bytes_per_block"] == groups
    assert set(block["sources"]["bytes_per_block"].values()) == {"measured"}
    assert block["sources"]["rows"]["group_bytes"] == "measured"
    rows = block["sources"]["rows"]
    assert (rows["transferred_tokens"], rows["prefill_attention_blocks"]) == ("measured", "measured")
    assert {name: len(arm["rows"]) for name, arm in block["arms"].items()} == {"A": 14, "B": 8, "C": 6}
    for row in block["arms"]["C"]["rows"]:
        assert row["group_bytes"] == {**groups, "attention": 21_978_496 * row["attention_blocks_moved"]}
        assert sum(row["group_bytes"].values()) == row["bytes_per_rank"]


def test_nixl_transfer_evidence_of_vllm_0330_repeats_the_rows_of_vllm_0320():
    def key(row):
        return (row["prompt_tokens"], row["attention_blocks_moved"], row["bytes_per_rank"], row["descriptors_per_rank"])

    old = _evidence_versions(_EVIDENCE_0320)["0.32.0"]["rows"]
    new = _evidence_versions(_EVIDENCE_0330)["0.33.0"]["arms"]["A"]["rows"]
    assert [key(row) for row in new] == [key(row) for row in old]


def test_nixl_transfer_evidence_of_vllm_0330_holds_the_region_table_of_the_live_connector():
    table = json.loads((_EVIDENCE / _EVIDENCE_0330 / "region-table.json").read_text())
    lengths, group_ids = table["block_len_per_layer"], table["region_group_ids"]
    assert table["num_regions"] == len(lengths) == len(group_ids) == len(table["region_names"]) == 40
    assert table["physical_blocks_per_logical"] == 1
    assert lengths == [71_808] + [1_114_112] * 8 + [71_808] * 10 + [1_114_112] * 11 + [2_048] * 10
    assert group_ids == [-1] * 9 + [4] * 21 + [5] * 10
    assert table["ssm_region_indices"] == list(range(9))
    assert table["scratch_region_indices"] == [0, *range(30, 40)]
    assert table["mamba_ssm_size"] == [36_864, 1_048_576]
    # The regions give the measured bytes of each group: an attention block reads every region, the ring its scratch
    # regions, and a Mamba group the state bytes in every state region.
    measured = _evidence_versions(_EVIDENCE_0330)["0.33.0"]["bytes_per_block"]
    assert sum(lengths) == measured["attention"]
    assert sum(lengths[index] for index in table["scratch_region_indices"]) == measured["ring"]
    assert len(table["ssm_region_indices"]) * sum(table["mamba_ssm_size"]) == measured["kda_group_0"]


_PROBE_FIXTURE = (
    Path(__file__).resolve().parents[1] / "python/aisim/tests/unit/collector/fixtures/glm53flash_vllm_0330_probe"
)


def test_grouped_pd_prediction_runs_on_the_cache_groups_of_a_vllm_0330_probe_import(grouped_case, tmp_path):
    # The import of the tracked probe record of GLM-5.3-Flash gives the six cache groups with their transfer bytes.
    # A P/D prediction on vLLM 0.33.0 takes them as they are. The pool of 64 pages is hand-made, as in
    # _glm_shaped_profile. The fixture leaves out the staged model configuration, which equals the bundled one.
    import shutil

    from collector.fpm_forward.runtime_observations import validate_observation_phase

    import aisim_core

    campaign = tmp_path / "probe"
    shutil.copytree(_PROBE_FIXTURE, campaign)
    index = campaign / "observations.json"
    ((label, saved),) = json.loads(index.read_text())["configurations"].items()
    bundled = Path(aisim_core.__file__).parent / "model_configs/zai-org--GLM-5.3-Flash_config.json"
    for reference in saved["attempts"][0]["phases"]["prefill"]["artifacts"]:
        if Path(reference["path"]).name == "model-config.json":
            shutil.copyfile(bundled, index.parent / reference["path"])
    imported = validate_observation_phase(
        index, label, saved["launch"], phase="prefill", attempt_id=saved["active_attempt_id"]
    )
    groups = imported["resources"]["cache_groups"]
    assert [group["transfer_bytes_per_block"] for group in groups] == [9_768_960] * 4 + [21_978_496, 92_288]
    profile, root = grouped_case
    profile = _glm_shaped_profile(profile, "0.33.0")
    for deployment in profile["deployments"]:
        resources = deployment["resources"]
        resources["cache_groups"] = deepcopy(groups)
        resources["runtime_memory"].update(kv_cache_bytes=64 * 13_516_800, runtime_framework_version="0.33.0")
    # One attention block and one block of each state group move: the measured row of 1,000 tokens, 61,146,624 bytes.
    assert _glm_pd_transfer_ms(profile, root, tmp_path, "0.33.0") == pytest.approx(61_146_624 / 1e6, abs=1e-3)
