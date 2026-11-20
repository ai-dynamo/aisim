# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from aisim.generator.api import generate_from_request
from aisim.generator.rendering.engine import prepare_template_context
from aisim.generator.request import ModelFacts, from_sweeper_candidate

from .test_sweeper_request import _agg_candidate, _cli_flag_value, _disagg_candidate


def test_agg_sglang_candidate_renders_evaluated_prefill_token_limit():
    request = from_sweeper_candidate(
        _agg_candidate(backend="sglang", backend_version="0.5.18"),
        workload={"isl": 4000, "osl": 1000},
        model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
        generator_overrides={"K8sConfig": {"k8s_image": "example/sglang:0.5.18"}},
    )

    artifacts = generate_from_request(request)

    assert request.topology.roles["agg"].extra["max_prefill_tokens"] == 8192
    assert _cli_flag_value(artifacts["cli_args_agg"], "--max-prefill-tokens") == "8192"


@pytest.mark.parametrize("version", ["0.5.0", "0.5.18"])
@pytest.mark.parametrize("legacy", [False, True])
def test_null_chunk_matches_omitted_chunk(version, legacy):
    candidate = _agg_candidate(backend="sglang", backend_version=version)
    candidate["config"].pop("adapters", None)
    request = from_sweeper_candidate(
        candidate,
        workload={"isl": 4000, "osl": 1000},
        model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
        generator_overrides={"K8sConfig": {"k8s_image": f"example/sglang:{version}"}},
    )
    sizing = request.topology.roles["agg"]
    extra = dict(sizing.extra)
    if legacy:
        extra["enable_chunked_prefill"] = True
    else:
        extra.pop("enable_chunked_prefill", None)

    def render(extra):
        roles = {"agg": replace(sizing, extra=extra)}
        configured = replace(request, topology=replace(request.topology, roles=roles))
        return generate_from_request(configured)["cli_args_agg"]

    omitted = render(extra)
    explicit_null = render({**extra, "chunked_prefill_size": None})
    assert explicit_null == omitted
    assert '--chunked-prefill-size "None"' not in explicit_null
    if legacy:
        assert "--chunked-prefill-size" in explicit_null


@pytest.mark.parametrize("legacy_value", [True, False])
def test_explicit_chunk_overrides_later_legacy_mapping(legacy_value):
    context = prepare_template_context(
        {
            "params": {
                "agg": {"max_num_tokens": 8192, "chunked_prefill_size": 128, "enable_chunked_prefill": legacy_value}
            }
        },
        "sglang",
    )
    assert context["agg_chunked-prefill-size"] == 128


def test_disagg_sglang_candidate_renders_each_role_prefill_token_limit():
    adapters = {
        "dynamo.router": {
            "mode": "kv",
            "overlap_score_credit": 0.75,
            "router_temperature": 0.2,
        },
        "dynamo.planner": {
            "scaling_policy": "throughput",
            "enable_throughput_scaling": True,
            "enable_load_scaling": False,
            "environment": "kubernetes",
            "mode": "disagg",
        },
    }
    request = from_sweeper_candidate(
        _disagg_candidate(
            backend="sglang",
            backend_version="0.5.18",
            adapters=adapters,
        ),
        workload={"isl": 8192, "osl": 1024},
        model_facts=ModelFacts(is_moe=True, architecture="DeepseekV3ForCausalLM"),
        generator_overrides={"K8sConfig": {"k8s_image": "example/sglang:0.5.18"}},
    )

    artifacts = generate_from_request(request)

    assert _cli_flag_value(artifacts["cli_args_prefill"], "--max-prefill-tokens") == "16384"
    assert _cli_flag_value(artifacts["cli_args_decode"], "--max-prefill-tokens") == "8192"


@pytest.mark.parametrize("backend_version", ["current", "0.5.10.post1", "0.5.9", "0.5.8"])
def test_sglang_candidate_omits_unspecified_kv_cache_dtype(backend_version):
    request = from_sweeper_candidate(
        _agg_candidate(backend="sglang", backend_version=backend_version),
        workload={"isl": 1024, "osl": 128},
        model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
        generator_overrides={"K8sConfig": {"k8s_image": "example/sglang:release"}},
    )

    artifacts = generate_from_request(request)

    assert "--kv-cache-dtype" not in artifacts["cli_args_agg"]
    assert "--kv-cache-dtype" not in artifacts["k8s_deploy.yaml"]


@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize(
    "version", ["0.5.1.post1", "0.5.3.post2", "0.5.4.post3", "0.5.6.post2", "0.5.8", "0.5.9", "0.5.10.post1", "0.5.18"]
)
def test_explicit_chunk_is_preserved_separately_from_prefill_ceiling(mode, version):
    candidate = (_agg_candidate if mode == "agg" else _disagg_candidate)(backend="sglang", backend_version=version)
    candidate["config"].pop("adapters", None)
    roles = ("agg",) if mode == "agg" else ("prefill", "decode")
    chunks = {role: 4096 * (index + 1) for index, role in enumerate(roles)}
    for role, chunk in chunks.items():
        candidate["config"][f"{role}_chunked_prefill_size"] = chunk
    request = from_sweeper_candidate(
        candidate,
        workload={"isl": 4000, "osl": 1000},
        model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
        generator_overrides={"K8sConfig": {"k8s_image": f"example/sglang:{version}"}},
    )
    artifacts = generate_from_request(request)
    for role, chunk in chunks.items():
        assert _cli_flag_value(artifacts[f"cli_args_{role}"], "--chunked-prefill-size") == str(chunk)
        assert _cli_flag_value(artifacts[f"cli_args_{role}"], "--max-prefill-tokens") == str(
            candidate["config"][f"{role}_max_num_batched_tokens"]
        )


def test_generator_does_not_silently_round_an_evaluated_chunk():
    candidate = _agg_candidate(backend="sglang", backend_version="0.5.18")
    candidate["config"]["agg_chunked_prefill_size"] = 65
    with pytest.raises(ValueError, match="multiple of 64"):
        from_sweeper_candidate(
            candidate,
            workload={"isl": 4000, "osl": 1000},
            model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
        )


@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize("chunk", [65, 127, 128])
@pytest.mark.parametrize("rule", [None, "benchmark"])
def test_direct_request_preserves_or_rejects_evaluated_chunks(mode, chunk, rule):
    candidate = (_agg_candidate if mode == "agg" else _disagg_candidate)(backend="sglang", backend_version="0.5.18")
    candidate["config"].pop("adapters", None)
    request = from_sweeper_candidate(
        candidate,
        workload={"isl": 4000, "osl": 1000},
        model_facts=ModelFacts(is_moe=False, architecture="Qwen3ForCausalLM"),
        generator_overrides={"K8sConfig": {"k8s_image": "example/sglang:0.5.18"}},
    )
    # Inject RoleSizing.extra directly, bypassing from_sweeper_candidate's chunk check.
    role = "agg" if mode == "agg" else "decode"
    roles = dict(request.topology.roles)
    sizing = roles[role]
    roles[role] = replace(sizing, extra={**sizing.extra, "chunked_prefill_size": chunk})
    request = replace(
        request,
        topology=replace(request.topology, roles=roles),
        overrides=replace(request.overrides, raw={**request.overrides.raw, "rule": rule}),
    )

    if chunk % 64:
        with pytest.raises(ValueError, match=rf"Cannot preserve evaluated {role}\.chunked_prefill_size={chunk}"):
            generate_from_request(request)
    else:
        artifacts = generate_from_request(request)
        assert _cli_flag_value(artifacts[f"cli_args_{role}"], "--chunked-prefill-size") == str(chunk)
        assert _cli_flag_value(artifacts[f"cli_args_{role}"], "--max-prefill-tokens") == str(
            candidate["config"][f"{role}_max_num_batched_tokens"]
        )
