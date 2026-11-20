# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic imported native proof; these tests do not qualify GPU execution."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from collector.fpm_forward import planner, runtime_probe
from collector.fpm_forward.capabilities import resolve_model_capability
from collector.fpm_forward.config import _freeze_benchmark_points
from collector.fpm_forward.runtime_instrumentation import load_instrumentation, runtime_binding
from collector.fpm_forward.runtime_observations import expected_ranks, validate_observations
from collector.fpm_forward.types import ParallelTopology

from aisimulate_core.sdk.fpm_profile import load_fpm_profile

from .test_fpm_forward import _kimi_k31_config
from .test_fpm_profile_collection import no_models_or_timing_data  # noqa: F401
from .test_k31_runtime_observations import _complete_fixture, _decoder_cache_names, _kernel_view_fixture, _moe
from .test_runtime_observations import _replace_launch, _write

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_models_or_timing_data")]


@pytest.fixture
def inputs(tmp_path, monkeypatch, request):
    settings = getattr(request, "param", "trtllm_mla")
    workload_kind = "prefill" if settings == "prefill" else "decode"
    index_path, launches = _complete_fixture(tmp_path, monkeypatch, workload_kind=workload_kind)
    config_path, text = _kimi_k31_config(tmp_path)
    # Keep the independent checkpoint layout small and consistent with the
    # synthetic native views, including two distinct KDA decoder branches.
    text["num_hidden_layers"] = 4
    text["linear_attn_config"].update(kda_layers=[1, 2], full_attn_layers=[3, 4])
    text["dsa_config"]["dsa_layers"] = [3, 4]
    config = json.loads(config_path.read_text())
    config["text_config"] = text
    config_path.write_text(json.dumps(config))
    launch = launches["tp2"]
    if workload_kind == "prefill":
        settings = "trtllm_mla"
    if settings == "explicit-moe":
        launch["precision"]["moe_backend"] = "flashinfer_trtllm"
        settings = "trtllm_mla"
    attention_backend, packed_indexer = (settings, False) if isinstance(settings, str) else settings
    launch["precision"]["attention_backend"] = attention_backend
    launch["model_config"] = {"path": str(config_path), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}
    launch["topology"].update(dp=1, moe_tp=2, moe_ep=1)
    launch = runtime_probe.normalize_probe_launch(launch)
    launches["tp2"] = launch
    _replace_launch(index_path, launches)
    index = json.loads(index_path.read_text())
    attempt = index["configurations"]["tp2"]["attempts"][0]
    bundle = load_instrumentation(tmp_path / attempt["bundle"]["manifest"])
    phase = attempt["phases"][workload_kind]
    context_path = tmp_path / phase["launch_manifest"]["path"]
    context = json.loads(context_path.read_text())
    context.update(expected_ranks=expected_ranks(launch["topology"]), instrumentation_binding=runtime_binding())
    phase["launch_manifest"]["sha256"] = _write(context_path, context)["sha256"]
    if packed_indexer:
        cache, scheduled = [_decoder_cache_names(value) for value in _kernel_view_fixture(monkeypatch)]
        for mapping in cache["kernel_block_views"].values():
            mapping["kernel_block_table_shape"][1] = launch["collection"]["max_num_seqs"]
    retained = []
    for ref in phase["artifacts"]:
        record_path = tmp_path / ref["path"]
        record = json.loads(record_path.read_text())
        if record["dp_rank"] != 0:
            continue
        record["model_config_sha256"] = launch["model_config"]["sha256"]
        if record["kind"] == "worker":
            record["initialized_moe"] = _moe(indices=range(1, text["num_hidden_layers"]))
        record["resolved_config"]["parallel_config"].update(
            data_parallel_size=1, data_parallel_size_local=1, enable_expert_parallel=False
        )
        record["resolved_config"]["kernel_config"]["moe_backend"] = launch["precision"]["moe_backend"]
        if packed_indexer:
            record["cache"] = copy.deepcopy(cache if record["kind"] == "worker" else scheduled)
        if attention_backend == "flashmla":
            config = record["resolved_config"]
            config["cache_config"]["cache_dtype"] = "fp8_ds_mla"
            config["attention_config"]["backend"] = "FLASHMLA_SPARSE"
            for group in record["cache"]["groups"]:
                specs = group.get("layer_specs", dict.fromkeys(group["layer_names"], group))
                for name, spec in specs.items():
                    if name.endswith(".self_attn.attn"):
                        spec["cache_dtype_str"] = "fp8_ds_mla"
            if record["kind"] == "worker":
                record["attention_backend"][0] = "FLASHMLA_SPARSE"
                record["attention_groups"][0]["backend"] = "FLASHMLA_SPARSE"
        record["instrumentation_binding"] = {
            **runtime_binding(),
            **{key: record[key] for key in ("attempt_id", "configuration", "phase", "bundle_sha256")},
            "status": "verified",
            "expected_bundle_sha256": bundle.sha256,
            "hostname": "synthetic-host",
            "pid": 123,
            "root": "/frozen",
            "files": bundle.files,
            "imports": {
                "observer": {
                    "file": "observer.py",
                    "path": "/frozen/observer.py",
                    "sha256": bundle.files["observer.py"],
                }
            },
            "classes": {key: bundle.manifest[key] for key in ("worker_class", "scheduler_class")},
        }
        record["cpu_affinity_observation"] = {"status": "complete", "hostname": "synthetic-host", "pid": 123}
        ref["sha256"] = _write(record_path, record)["sha256"]
        retained.append(ref)
    phase["artifacts"] = retained
    index_path.write_text(json.dumps(index))
    result = validate_observations(index_path, launches)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    probe = runtime_probe.build_runtime_probe_plan("tp2", launch, bundle)
    options = probe.options
    if workload_kind == "prefill":
        points_path = tmp_path / "prefill-points.json"
        _write(
            points_path,
            {
                "schema_version": 1,
                "prefill": [{"batch_size": 1, "total_prefill_tokens": 128, "total_kv_read_tokens": 0}],
                "decode": [],
            },
        )
        payload, digest = _freeze_benchmark_points(str(points_path))
        options = replace(options, benchmark_points_json=payload, benchmark_points_sha256=digest)
    capability = resolve_model_capability(
        backend="vllm",
        model_architecture=None,
        model_path=launch["identity"]["model"],
        system="gb300",
        selected_ops=set(),
        has_model_cases=False,
        requested_weight_quantizations=(),
        requested_kv_cache_dtypes=("fp8",),
        model_config_path=str(config_path),
        database_version=launch["identity"]["framework_version"],
        checkpoint_native_dtypes=True,
    )
    profile = {
        "schema_version": 1,
        "model": launch["identity"]["model"],
        "model_revision": launch["identity"]["model_revision"],
        "architecture": capability.architecture,
        "context_length": 4096,
        "num_experts": 8,
        "provenance": "Synthetic test only",
        "deployments": [
            {
                "system": "gb300",
                "backend": "vllm",
                "backend_version": launch["identity"]["framework_version"],
                **launch["topology"],
                **{key: value for key, value in launch["precision"].items() if key != "kvcache_quant_mode"},
                "kv_cache_dtype": launch["precision"]["kvcache_quant_mode"],
                "worker_type": workload_kind,
                "resources": result["resources"],
            }
        ],
    }
    return {
        "capability": capability,
        "profile": profile,
        "topologies": (ParallelTopology(**launch["topology"]),),
        "model_path": profile["model"],
        "system": "gb300",
        "backend": "vllm",
        "options": options,
        "runtime_instrumentation": bundle,
        "runtime_launch": launch,
    }


def bind(inputs):
    return planner._observed_k31_capability(**{**inputs, "profile": load_fpm_profile(inputs["profile"])})


def edit_proof(inputs, mutation):
    memory = inputs["profile"]["deployments"][0]["resources"]["runtime_memory"]
    proof = json.loads(memory["provenance"])
    mutation(proof)
    memory["provenance"] = json.dumps(proof)


def test_valid_imported_native_proof_changes_only_moe_precision(inputs):
    original = inputs["capability"]
    assert original.dtype.moe_quant_mode == "w4a16_mxfp4"
    observed = bind(inputs)
    assert observed == replace(original, dtype=replace(original.dtype, moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm"))
    assert original.dtype.moe_quant_mode == "w4a16_mxfp4"


def test_planner_rejects_inline_worker_missing_a_configured_moe_layer(inputs):
    def omit_layer(proof):
        worker = next(item["evidence"] for item in proof["artifacts"] if item["evidence"]["kind"] == "worker")
        worker["initialized_moe"]["layers"].pop()
        worker["initialized_moe"]["expected_layer_names"].pop()

    edit_proof(inputs, omit_layer)
    with pytest.raises(ValueError, match="model configuration inventory"):
        bind(inputs)


@pytest.mark.parametrize("instrumentation", [None, [], "verified"])
def test_inline_proof_rejects_non_object_instrumentation(inputs, instrumentation):
    edit_proof(inputs, lambda proof: proof.update(instrumentation=instrumentation))
    with pytest.raises(ValueError, match="provenance is not bound"):
        bind(inputs)


@pytest.mark.parametrize(
    "inputs",
    [("trtllm_mla", False), ("trtllm_mla", True), ("flashmla", True), "prefill"],
    ids=["manager-blocks", "packed-indexer-trtllm", "packed-indexer-flashmla", "zero-history-prefill"],
    indirect=True,
)
@pytest.mark.parametrize("onboarding", [False, True], ids=["inline-proof", "onboarding-proof"])
def test_formal_plan_uses_observed_precision_and_keeps_final_launch_validation(inputs, monkeypatch, onboarding):
    if onboarding:
        onboarding_wrapper(inputs)
    calls = []
    validate = runtime_probe.validate_collection_probe_launch

    def record(*args, **kwargs):
        calls.append(True)
        return validate(*args, **kwargs)

    monkeypatch.setattr(runtime_probe, "validate_collection_probe_launch", record)
    plan_args = dict(
        backend=inputs["backend"],
        model_path=inputs["model_path"],
        system=inputs["system"],
        selected_ops=set(),
        options=inputs["options"],
        has_model_cases=False,
        model_config_path=inputs["runtime_launch"]["model_config"]["path"],
        fpm_profile=inputs["profile"],
        runtime_instrumentation=inputs["runtime_instrumentation"],
        runtime_launch=inputs["runtime_launch"],
        generator_overrides=runtime_probe.probe_generator_overrides(inputs["runtime_launch"]),
    )
    plan = planner.build_collection_plan(**plan_args)
    assert calls == [True]
    assert plan.cells and {cell.moe_quant_mode for cell in plan.cells} == {"w4a8_mxfp4_mxfp8_trtllm"}
    assert plan.runtime_instrumentation.sha256 == inputs["runtime_instrumentation"].sha256
    if inputs["runtime_launch"]["worker_type"] == "prefill":
        from collector.fpm_forward import runner

        assert {cell.workload_kind for cell in plan.cells} == {"prefill"}
        assert {p["total_kv_read_tokens"] for p in json.loads(plan.options.benchmark_points_json)["prefill"]} == {0}
        for cell in plan.cells:
            generated = runner._cell_generator_overrides(plan, cell, plan_args["generator_overrides"])
            assert "--no-enable-prefix-caching" in generated["params"]["agg"]["extra_cli_args"]
    plan_args["options"] = replace(inputs["options"], warmup_iterations=inputs["options"].warmup_iterations + 1)
    with pytest.raises(ValueError, match="warmup_iterations"):
        planner.build_collection_plan(**plan_args)


def test_formal_plan_freezes_the_bundle_used_for_precision_validation(inputs, monkeypatch):
    from collector.fpm_forward import runtime_instrumentation

    bundle = inputs["runtime_instrumentation"]
    calls = []
    load = runtime_instrumentation.load_instrumentation

    def load_once(*args, **kwargs):
        calls.append(True)
        # A second read must not substitute different bytes after validation.
        return bundle if len(calls) == 1 else replace(bundle, sha256="0" * 64)

    monkeypatch.setattr(runtime_instrumentation, "load_instrumentation", load_once)
    plan = planner.build_collection_plan(
        backend=inputs["backend"],
        model_path=inputs["model_path"],
        system=inputs["system"],
        selected_ops=set(),
        options=inputs["options"],
        has_model_cases=False,
        model_config_path=inputs["runtime_launch"]["model_config"]["path"],
        fpm_profile=inputs["profile"],
        runtime_instrumentation=bundle.manifest_path,
        runtime_launch=inputs["runtime_launch"],
        generator_overrides=runtime_probe.probe_generator_overrides(inputs["runtime_launch"]),
    )
    assert calls == [True]
    monkeypatch.setattr(runtime_instrumentation, "load_instrumentation", load)
    assert plan.runtime_instrumentation.sha256 == bundle.sha256
    root = bundle.root / "collection"
    frozen = root / "runtime-observations/parent-attempt/instrumentation"
    source = bundle.root / next(iter(bundle.files))
    source.write_bytes(source.read_bytes() + b"\n# changed after planning\n")
    with pytest.raises(ValueError, match="instrumentation.*changed"):
        runtime_probe.prepare_collection_observations(plan, root, "parent-attempt")
    assert not frozen.exists()


@pytest.mark.parametrize("different_device", [False, True])
def test_inline_proof_requires_matching_rank_hardware(inputs, different_device):
    def mutate(proof):
        for artifact in proof["artifacts"]:
            worker = artifact["evidence"]
            if different_device and worker["kind"] == "worker" and worker["tp_rank"] == 1:
                worker["hardware"]["device_name"] += " variant"

    edit_proof(inputs, mutate)
    if different_device:
        with pytest.raises(ValueError, match="hardware across phases/ranks"):
            bind(inputs)
    else:
        assert bind(inputs).dtype.moe_quant_mode == "w4a8_mxfp4_mxfp8_trtllm"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(source="declared"),
        lambda p: p.update(bundle_sha256="0" * 64),
        lambda p: p["runtime"].update(version="0.31.0"),
        lambda p: p["runtime"]["source_files"].pop(next(iter(p["runtime"]["source_files"]))),
        lambda p: p["launch"]["identity"].update(model_revision="different-checkpoint"),
        lambda p: p["launch"]["topology"].update(tp=1),
        lambda p: p["instrumentation"].update(import_binding="unverified_legacy"),
        lambda p: p["artifacts"].pop(0),
        lambda p: p["artifacts"].append(copy.deepcopy(p["artifacts"][0])),
        lambda p: p["artifacts"][0].update(sha256="not-a-hash"),
        lambda p: p["artifacts"][0]["evidence"].update(initialized_moe=None),
        lambda p: p["artifacts"][0]["evidence"]["initialized_moe"]["layers"][0]["descriptors"]["_a1"].update(
            dtype="bfloat16"
        ),
        lambda p: p["artifacts"][0]["evidence"].update(phase="prefill"),
        lambda p: p["artifacts"][0]["evidence"].update(unresolved_fields=["precision"]),
        lambda p: p["artifacts"][0]["evidence"]["lifecycle"].update(warmup_completed=False),
        lambda p: p["artifacts"][0]["evidence"]["instrumentation_binding"].update(status="unverified"),
        lambda p: p["rank_capacities"].pop(),
        lambda p: p["rank_capacities"].append(copy.deepcopy(p["rank_capacities"][0])),
        lambda p: p["rank_capacities"][0].update(kv_cache_bytes=9999),
    ],
)
def test_missing_stale_partial_or_relabelled_observations_cannot_override_precision(inputs, mutation):
    edit_proof(inputs, mutation)
    with pytest.raises(ValueError):
        bind(inputs)
    assert inputs["capability"].dtype.moe_quant_mode == "w4a16_mxfp4"


@pytest.mark.parametrize("field", ["runtime_instrumentation", "runtime_launch"])
def test_formal_binding_cannot_be_replaced_by_profile_label(inputs, field):
    inputs[field] = None
    with pytest.raises(ValueError, match="formal runtime instrumentation and launch"):
        bind(inputs)


def test_current_model_payload_is_bound_independently_of_public_name(inputs):
    capability = inputs["capability"]
    model = capability.model_config
    altered = dict(model.payload, test_only_changed_checkpoint=True)
    inputs["capability"] = replace(
        capability,
        model_config=type(model)(altered, source_kind=model.source_kind, source_reference=model.source_reference),
    )
    with pytest.raises(ValueError, match="current checkpoint"):
        bind(inputs)


def test_unobserved_checkpoint_native_identity_is_not_promoted(inputs):
    inputs["profile"]["deployments"][0]["moe_quant_mode"] = "w4a16_mxfp4"
    inputs["runtime_instrumentation"] = inputs["runtime_launch"] = None
    with pytest.raises(ValueError, match="formal runtime instrumentation and launch"):
        bind(inputs)


def test_formal_plan_rejects_unobserved_inferred_moe_precision(inputs):
    inputs["profile"]["deployments"][0]["moe_quant_mode"] = "w4a16_mxfp4"
    with pytest.raises(ValueError, match="formal runtime instrumentation and launch"):
        planner.build_collection_plan(
            backend=inputs["backend"],
            model_path=inputs["model_path"],
            system=inputs["system"],
            selected_ops=set(),
            options=inputs["options"],
            has_model_cases=False,
            model_config_path=inputs["runtime_launch"]["model_config"]["path"],
            fpm_profile=inputs["profile"],
        )


def test_matching_inferred_moe_label_still_requires_observed_arithmetic(inputs):
    inputs["profile"]["deployments"][0]["moe_quant_mode"] = "w4a16_mxfp4"
    with pytest.raises(ValueError, match="from the selected profile"):
        bind(inputs)


@pytest.mark.parametrize("matching", [False, True])
def test_inline_tp2_proof_requires_matching_effective_graph_settings(inputs, matching):
    def captured(proof):
        for artifact in proof["artifacts"]:
            record = artifact["evidence"]
            if record["kind"] != "worker":
                continue
            graph = record["resolved_config"]["compilation_config"]
            record["initial_compilation_config"] = copy.deepcopy(graph)
            if matching or record["tp_rank"] == 1:
                graph["cudagraph_mode"] = "PIECEWISE"

    edit_proof(inputs, captured)
    if matching:
        assert bind(inputs).dtype.moe_quant_mode == "w4a8_mxfp4_mxfp8_trtllm"
    else:
        with pytest.raises(ValueError, match="effective runtime/graph settings across phases/ranks"):
            bind(inputs)


@pytest.mark.parametrize("matching", [False, True])
@pytest.mark.parametrize("difference", ["settings", "geometry"])
def test_inline_and_imported_dp_proofs_require_matching_full_settings(inputs, matching, difference):
    # Preserve two physical workers while changing the topology to TP1/DP2.
    # Each DP scheduler agrees with its worker even in the mismatching case.
    launch = inputs["runtime_launch"]
    launch["topology"].update(tp=1, dp=2, moe_tp=1, moe_ep=2)
    inputs["topologies"] = (ParallelTopology(**launch["topology"]),)
    deployment = inputs["profile"]["deployments"][0]
    deployment.update(launch["topology"])
    memory = deployment["resources"]["runtime_memory"]
    proof = json.loads(memory["provenance"])
    proof["launch"] = copy.deepcopy(launch)
    artifacts = []
    scheduler = next(a for a in proof["artifacts"] if a["evidence"]["kind"] == "scheduler")
    for artifact in proof["artifacts"]:
        record = artifact["evidence"]
        if record["kind"] != "worker":
            continue
        dp = record["tp_rank"]
        record.update(dp_rank=dp, tp_rank=0)
        copied_scheduler = copy.deepcopy(scheduler)
        copied_scheduler["path"] += f".dp{dp}"
        copied_scheduler["evidence"]["dp_rank"] = dp
        for item in (artifact, copied_scheduler):
            observed = item["evidence"]
            observed["launch"] = copy.deepcopy(launch)
            observed["resolved_config"]["parallel_config"].update(
                tensor_parallel_size=1,
                data_parallel_size=2,
                data_parallel_size_local=1 if difference == "settings" and not matching and dp == 1 else 2,
                enable_expert_parallel=True,
            )
            if difference == "geometry" and not matching and dp == 1:
                observed["cache"]["groups"][-1]["layer_names"].reverse()
            if difference == "geometry":
                cache = observed["cache"]
                cache["pool_id"] = f"pool-dp{dp}"
                for group in cache["groups"]:
                    group["pool_id"] = cache["pool_id"]
                if observed["kind"] == "scheduler":
                    cache["group_pool_ids"] = [cache["pool_id"]] * len(cache["groups"])
            artifacts.append(item)
    proof["artifacts"] = artifacts
    for rank in proof["rank_capacities"]:
        rank.update(dp_rank=rank["tp_rank"], tp_rank=0)
    memory["provenance"] = json.dumps(proof)
    if matching:
        assert bind(inputs).dtype.moe_quant_mode == "w4a8_mxfp4_mxfp8_trtllm"
    else:
        message = "effective runtime/graph settings" if difference == "settings" else "cache geometry"
        with pytest.raises(ValueError, match=message + " across phases/ranks"):
            bind(inputs)

    # Rebuild the hashed artifact route from the same records to verify that
    # importing and inline planning apply the same cross-rank contract.
    index_path = Path(proof["observations_index"]["path"])
    root = index_path.parent
    index = json.loads(index_path.read_text())
    index["configurations"]["tp2"]["launch"] = launch
    attempt = index["configurations"]["tp2"]["attempts"][0]
    phase = attempt["phases"]["decode"]
    context_path = root / phase["launch_manifest"]["path"]
    context = json.loads(context_path.read_text())
    context.update(launch=launch, expected_ranks=expected_ranks(launch["topology"]))
    phase["launch_manifest"]["sha256"] = _write(context_path, context)["sha256"]
    phase["artifacts"] = [
        {"kind": "observation", **_write(root / a["path"], a["evidence"]), "path": a["path"]} for a in artifacts
    ]
    index_path.write_text(json.dumps(index))
    imported = validate_observations(index_path, {"tp2": launch})["tp2"]
    assert imported["status"] == ("complete" if matching else "incomplete"), imported
    if not matching:
        assert message + " across phases/ranks" in str(imported["diagnostics"])


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda r: r.update(max_num_tokens=2048), "resource envelope"),
        (lambda r: r.update(max_batch_size=32), "resource envelope"),
        (lambda r: r["runtime_memory"].update(max_model_len=8192), "resource envelope"),
        (lambda r: r["runtime_memory"].update(gpu_memory_utilization=0.8), "resource envelope"),
        (lambda r: r["cache_groups"][0].update(page_size_bytes=1024), "profile cache groups"),
        (lambda r: r["cache_groups"][-1].update(blocks_per_request=2), "profile cache groups"),
        (lambda r: r["cache_groups"].pop(), "profile cache groups"),
    ],
)
def test_resource_fields_cannot_be_relabelled_around_valid_precision_proof(inputs, mutation, match):
    mutation(inputs["profile"]["deployments"][0]["resources"])
    with pytest.raises(ValueError, match=match):
        bind(inputs)


def onboarding_wrapper(inputs):
    memory = inputs["profile"]["deployments"][0]["resources"]["runtime_memory"]
    proof = json.loads(memory["provenance"])
    index = Path(proof["observations_index"]["path"])
    (index.parent / "launch.json").write_text(json.dumps(inputs["runtime_launch"]))
    (index.parent / "model-config.json").write_bytes(
        Path(inputs["runtime_launch"]["model_config"]["path"]).read_bytes()
    )
    resources = copy.deepcopy(inputs["profile"]["deployments"][0]["resources"])
    resources.pop("provenance", None)
    resources["runtime_memory"].pop("provenance", None)
    wrapper = {
        "schema_version": "aisimulate-onboarding-runtime-probe/v1",
        "configuration": proof["configuration"],
        "checkpoint": str(index.parent / "checkpoint.json"),
        "model_metadata": {key: inputs["profile"][key] for key in ("architecture", "context_length", "num_experts")},
        "user_overrides": {},
        "observations_index": str(index),
        "source_artifacts": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(index.parent.rglob("*"))
            if path.is_file()
        ],
        "observed_resources": resources,
    }
    memory["provenance"] = json.dumps(wrapper)
    return wrapper


def test_canonical_onboarding_wrapper_revalidates_native_evidence(inputs):
    expected = bind(inputs)
    onboarding_wrapper(inputs)
    assert bind(inputs) == expected


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(configuration="missing"),
        lambda p: p["model_metadata"].update(num_experts=999),
        lambda p: p.update(user_overrides=None),
        lambda p: p.update(source_artifacts=[]),
        lambda p: p["source_artifacts"][0].update(sha256="0" * 64),
        lambda p: p["source_artifacts"].append(copy.deepcopy(p["source_artifacts"][0])),
        lambda p: p["observed_resources"]["runtime_memory"].update(kv_cache_bytes=9999),
    ],
)
def test_onboarding_wrapper_cannot_replace_immutable_native_proof(inputs, mutation):
    onboarding_wrapper(inputs)
    edit_proof(inputs, mutation)
    with pytest.raises(ValueError):
        bind(inputs)


def test_onboarding_wrapper_rejects_changed_source_even_if_inline_claim_is_valid(inputs):
    wrapper = onboarding_wrapper(inputs)
    index = Path(wrapper["observations_index"])
    index.write_text(index.read_text() + "\n")
    with pytest.raises(ValueError, match="source artifact changed"):
        bind(inputs)


def test_onboarding_wrapper_does_not_bypass_native_validator(inputs):
    memory = inputs["profile"]["deployments"][0]["resources"]["runtime_memory"]
    proof = json.loads(memory["provenance"])
    index_path = Path(proof["observations_index"]["path"])
    index = json.loads(index_path.read_text())
    phase = index["configurations"]["tp2"]["attempts"][0]["phases"]["decode"]
    for ref in phase["artifacts"]:
        path = index_path.parent / ref["path"]
        worker = json.loads(path.read_text())
        if worker["kind"] == "worker":
            worker["lifecycle"]["warmup_completed"] = False
            ref["sha256"] = _write(path, worker)["sha256"]
            break
    index_path.write_text(json.dumps(index))
    onboarding_wrapper(inputs)
    with pytest.raises(ValueError, match="worker warmup and graph capture completion evidence is missing"):
        bind(inputs)
