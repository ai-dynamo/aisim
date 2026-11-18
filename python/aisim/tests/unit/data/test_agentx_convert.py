# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentX bundles convert to aisim-e2e/v1 with canonical names and unchanged silicon statistics."""

import json

import pytest

from aisim_core.data.agentx_convert import (
    AgentxBundleError,
    agentx_bundle_to_configuration,
    agentx_bundles_to_configurations,
)

pytestmark = pytest.mark.unit


def _row(key, phase, start, end, out, ttft, tpot, status="completed", cached=None):
    return {
        "key": key,
        "identity_valid": True,
        "phase": phase,
        "status": status,
        "failure_reason": None,
        "start_ms": start,
        "end_ms": end,
        "input_length": 1000,
        "output_length": out,
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "e2e_ms": end - start,
        "cached_tokens": cached,
    }


def _bundle(root, name, *, category="self_collected", lanes=4, with_keys=True, model="Qwen/Qwen3.5-397B-A17B"):
    path = root / name
    path.mkdir()
    manifest = {
        "schema": "aisimulate.agentx.benchmark.v1",
        "id": name,
        "category": category,
        "model": model,
        "hardware": "B300",
        "backend": "vllm",
        "measured_backend_version": "0.31.0",
        "tensor_parallel": 8,
        "lanes": lanes,
        "corpus_id": "weka-062126-context-262144",
        "measured_requests": 3,
    }
    rows = [
        _row("k0", "preparation", 0.0, 100.0, 1, 100.0, None),
        _row("k1", "profile", 1000.0, 2000.0, 11, 200.0, 10.0, cached=64),
        _row("k2", "profile", 1500.0, 3000.0, 21, 400.0, 20.0),
        _row("k3", "profile", 2000.0, 2500.0, 5, 0.0, 0.0, status="failed"),
    ]
    configuration = {
        "schema": "aisimulate.agentx.silicon-config.v1",
        "identity": {},
        "evidence_gaps": [],
        "implementation_identity": {"image_sha256": "a" * 64},
        "effective": [{"rank": 0, "scheduler": {"max_num_seqs": 64}, "cache_config": {"block_size": 528}}],
    }
    reference = [
        {"key": "k1", "conversation_id": "c1", "source_trace_id": "t1", "source_outer_idx": 3, "agent_depth": 0}
    ]
    if not with_keys:
        reference = [{"request_id": "r1", "conversation_id": "c1"}]
    workload = {
        "variant": {"lanes": lanes, "profile_seconds": 3600},
        "silicon_reference": {
            "reference": reference,
            "silicon": {
                "metric_definitions": {"running": {"kind": "gauge", "unit": "requests"}},
                "phases": {
                    "profile": {"series": {"running": [{"start": 0.0, "end": 1.0, "value": 2.0, "coverage": 1.0}]}}
                },
                "provenance": {"aiperf_version": "1.0.12"},
            },
        },
        "other_silicon_settings_and_observations": [
            {
                "id": "agentx.weka",
                "parameters": [{"name": "context_limit", "value": 262144}, {"name": "seed", "value": 42}],
            }
        ],
    }
    for file, value in (
        ("manifest.json", manifest),
        ("configuration.json", configuration),
        ("workload.json", workload),
        ("observations.json", rows),
        ("features.json", {"features": [{"id": "speculation", "state": "disabled", "parameters": []}]}),
    ):
        (path / file).write_text(json.dumps(value))
    return path


def test_bundle_converts_with_canonical_names_and_lineage(tmp_path):
    data = agentx_bundle_to_configuration(_bundle(tmp_path, "qwen-weka-l4"))
    body = data.body
    assert (body.origin, body.hardware, body.framework, body.framework_version) == (
        "self-collected",
        "b300_sxm",
        "vllm",
        "0.31.0",
    )
    assert body.workload.kind == "agentic_trace" and body.workload.seed == 42
    assert body.settings["scheduler.max_num_seqs"].authority == "observed"
    assert body.source_ids["agentx"]["hardware"] == "B300"
    obs = data.observations[0]
    assert obs.concurrency == 4 and obs.observer == "client" and obs.valid_for_submission is False
    assert (obs.requests_total, obs.requests_completed, obs.requests_failed) == (3, 2, 1)
    phases = {r.request_id: r.phase for r in data.requests}
    assert phases == {"k0": "warmup", "k1": "profile", "k2": "profile", "k3": "profile"}
    k1 = next(r for r in data.requests if r.request_id == "k1")
    assert k1.cached_tokens == 64 and k1.source_trace_id == "t1" and k1.source_outer_idx == 3
    assert data.series[0].metric == "running_requests" and data.series[0].end_ms == 1000.0


def test_bundles_with_one_serving_identity_merge_into_one_configuration(tmp_path):
    configs = agentx_bundles_to_configurations([_bundle(tmp_path, "l1", lanes=1), _bundle(tmp_path, "l4", lanes=4)])
    assert len(configs) == 1
    assert [o.observation_id for o in configs[0].observations] == ["l1", "l4"]
    assert {o.concurrency for o in configs[0].observations} == {1, 4}


def test_official_bundle_without_reference_keys_records_a_lineage_gap(tmp_path):
    data = agentx_bundle_to_configuration(_bundle(tmp_path, "agentx-1", category="agentx", with_keys=False))
    assert data.body.origin == "official" and data.observations[0].valid_for_submission is None
    assert any(g["field"] == "requests.lineage" for g in data.body.evidence_gaps)
    assert all(r.source_trace_id is None for r in data.requests)


def test_unsupported_bundle_schema_is_rejected(tmp_path):
    path = _bundle(tmp_path, "bad")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["schema"] = "something-else/v9"
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AgentxBundleError):
        agentx_bundle_to_configuration(path)
