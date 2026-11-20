# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic source-bound snapshot identity tests; no GPU qualification."""

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.fpm_forward import execution_evidence
from collector.fpm_forward.runtime import fpm_memory_observer as observer

from .test_fpm_profile_collection import no_models_or_timing_data  # noqa: F401
from .test_fpm_repeatability import _write_campaign
from .test_fpm_runner import _write_provenance
from .test_fpm_runtime_memory import _typed, _vllm_config
from .test_k31_planner_precision import inputs  # noqa: F401
from .test_k31_runtime_observations import _set_state_dtypes
from .test_runtime_observations import _mutate, _replace_launch, _write

pytestmark = pytest.mark.unit


@pytest.fixture
def native_source(tmp_path, monkeypatch):
    """Synthetic installed packages with real on-disk source hashes."""
    from collector.fpm_forward.runtime_k31 import VERSION

    sources = {}
    packages = {}
    for name in ("vllm", "dynamo"):
        root = tmp_path / "installed" / name
        root.mkdir(parents=True)
        source = root / "source.py"
        source.write_text("# synthetic installed source fixture\n")
        sources[f"{name}/source.py"] = hashlib.sha256(source.read_bytes()).hexdigest()
        packages[name] = SimpleNamespace(__path__=[str(root)], __version__=VERSION)
        monkeypatch.setitem(sys.modules, name, packages[name])
    monkeypatch.setitem(sys.modules, "vllm._version", SimpleNamespace(__commit_id__="g266e88c05"))
    expert = Path(packages["vllm"].__path__[0]) / "model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py"
    expert.parent.mkdir(parents=True)
    expert.write_text("# synthetic installed expert implementation fixture\n")
    sources["vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py"] = hashlib.sha256(
        expert.read_bytes()
    ).hexdigest()
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "runtime": {
                    "version": VERSION,
                    "source_revision": hashlib.sha256(
                        json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "source_files": sources,
                }
            }
        )
    )
    monkeypatch.setenv("AISIMULATE_RUNTIME_INSTRUMENTATION", str(manifest))
    monkeypatch.setattr(observer.importlib.metadata, "version", lambda _: VERSION)
    return packages, manifest


@pytest.mark.parametrize(
    "corruption,match",
    [
        (None, None),
        ("vllm_source", "source hash differs: vllm/source.py"),
        ("expert_source", "source hash differs: vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py"),
        ("dynamo_source", "source hash differs: dynamo/source.py"),
        ("missing_source", "source path must resolve uniquely"),
        ("duplicate_source", "source path must resolve uniquely"),
        ("imported_version", "imported vllm package reports"),
        ("metadata_version", "requires audited vLLM"),
        ("commit", "VCS identifier contradicts"),
        ("manifest_version", "manifest="),
    ],
)
def test_k31_execution_checks_installed_source_before_observed(native_source, tmp_path, monkeypatch, corruption, match):
    from collector.fpm_forward.runtime_k31 import VERSION

    packages, manifest = native_source
    if corruption in {"vllm_source", "dynamo_source", "missing_source"}:
        name = "dynamo" if corruption == "dynamo_source" else "vllm"
        source = Path(packages[name].__path__[0]) / "source.py"
        if corruption == "missing_source":
            source.unlink()
        else:
            source.write_text("# modified installed source\n")
    elif corruption == "expert_source":
        expert = Path(packages["vllm"].__path__[0]) / "model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py"
        expert.write_text("# changed synthetic expert implementation\n")
    elif corruption == "duplicate_source":
        packages["vllm"].__path__ *= 2
    elif corruption == "imported_version":
        packages["vllm"].__version__ = "0.33.0"
    elif corruption == "metadata_version":
        monkeypatch.setattr(observer.importlib.metadata, "version", lambda _: "0.28.1")
    elif corruption == "commit":
        monkeypatch.setitem(sys.modules, "vllm._version", SimpleNamespace(__commit_id__="g99999999"))
    elif corruption == "manifest_version":
        document = json.loads(manifest.read_text())
        document["runtime"]["version"] = "0.33.0"
        manifest.write_text(json.dumps(document))
    _write_provenance(tmp_path / "collector-provenance.json", cell_id="synthetic-cell")
    monkeypatch.setattr(observer, "RESULTS_DIR", tmp_path)
    group = SimpleNamespace(
        backend=type("ResolvedBackend", (), {"__module__": "actual.runtime"}),
        layer_names=["layer.0"],
        kv_cache_group_id=0,
    )
    worker = SimpleNamespace(vllm_config=_vllm_config(), model_runner=SimpleNamespace(attn_groups=[[group]]))
    observer.observe_execution(worker, dp_rank=0, tp_rank=0, pp_rank=0, audited_version=VERSION)
    result = json.loads((tmp_path / "fpm-execution-worker-dp0-tp0-pp0.json").read_text())
    if corruption is None:
        assert result["status"] == "observed", result
        assert result["runtime"]["source_files"] == json.loads(manifest.read_text())["runtime"]["source_files"]
    else:
        assert result["status"] == "unresolved", result
        assert match in result["error"]
        assert "attention_groups" not in result


@pytest.fixture
def snapshot(request):
    bound_inputs = request.getfixturevalue("inputs")
    memory = bound_inputs["profile"]["deployments"][0]["resources"]["runtime_memory"]
    proof = json.loads(memory["provenance"])
    index_path = Path(proof["observations_index"]["path"])
    root = index_path.parent
    if getattr(request, "param", False):
        # A locally loaded Hub snapshot can omit HF's config commit hash.
        launch = bound_inputs["runtime_launch"]
        revision = "a" * 40
        launch["identity"]["model_revision"] = revision
        _replace_launch(index_path, {"tp2": launch})

        def pinned_snapshot(record):
            model = record["resolved_config"]["model_config"]
            model.update(
                model=f"/cache/models--{launch['identity']['model'].replace('/', '--')}/snapshots/{revision}",
                revision=revision,
            )
            model.pop("loaded_config_commit_hash", None)

        _mutate(index_path, pinned_snapshot)
        from collector.fpm_forward.runtime_observations import validate_observations

        imported = validate_observations(index_path, {"tp2": launch})["tp2"]
        assert imported["status"] == "complete", imported["diagnostics"]
        bound_inputs["profile"]["model_revision"] = revision
        bound_inputs["profile"]["deployments"][0]["resources"] = imported["resources"]
    # The canonical collector archive lives beside cells/<cell>/raw.
    target = root / "runtime-observations.json"
    index = json.loads(index_path.read_text())
    entry = index["configurations"]["tp2"]
    attempt = next(a for a in entry["attempts"] if a["attempt_id"] == entry["active_attempt_id"])
    phase = attempt["phases"]["decode"]
    phase.update(cell_id="native-cell", collector_attempt_id="native-attempt")
    context_path = root / phase["launch_manifest"]["path"]
    context = json.loads(context_path.read_text())
    context.update(cell_id="native-cell", collector_attempt_id="native-attempt")
    phase["launch_manifest"]["sha256"] = _write(context_path, context)["sha256"]
    for ref in phase["artifacts"]:
        path = root / ref["path"]
        record = json.loads(path.read_text())
        record.update(cell_id="native-cell", collector_attempt_id="native-attempt")
        ref["sha256"] = _write(path, record)["sha256"]
    target.write_text(json.dumps(index))
    raw = root / "cells/native-cell/raw"
    raw.mkdir(parents=True)
    plan = SimpleNamespace(
        runtime_configuration="tp2",
        runtime_launch=bound_inputs["runtime_launch"],
        runtime_instrumentation=bound_inputs["runtime_instrumentation"],
    )
    cell = SimpleNamespace(cell_id="native-cell", workload_kind="decode")
    collection = SimpleNamespace(collector_attempt_id="native-attempt")
    return plan, cell, raw, collection, target


def test_snapshot_identity_uses_revalidated_same_attempt_worker(snapshot):
    plan, cell, raw, collection, index = snapshot
    models, source = execution_evidence._runtime_worker_evidence(plan, cell, raw, collection)
    assert set(models) == {(0, 0, 0), (0, 1, 0)}
    assert source["sha256"] == hashlib.sha256(index.read_bytes()).hexdigest()
    revision = "a" * 40
    public = plan.runtime_launch["identity"]["model"]
    model = {"model": f"/cache/models--{public.replace('/', '--')}/snapshots/{revision}", "revision": revision}
    native = dict(model)
    assert not execution_evidence._matches_pinned_hub_snapshot(public, model, {revision})
    assert execution_evidence._matches_pinned_hub_snapshot(public, model, {revision}, native)


def test_snapshot_identity_is_checked_on_the_validated_manifest(snapshot, monkeypatch):
    from collector.fpm_forward import runtime_observations

    plan, cell, raw, collection, index_path = snapshot
    index = json.loads(index_path.read_text())
    phase = index["configurations"]["tp2"]["attempts"][0]["phases"]["decode"]
    manifest_path = index_path.parent / phase["launch_manifest"]["path"]
    wrong = json.loads(manifest_path.read_text())
    correct = copy.deepcopy(wrong)
    wrong["collector_attempt_id"] = "different-native-attempt"
    phase["launch_manifest"]["sha256"] = _write(manifest_path, wrong)["sha256"]
    index_path.write_text(json.dumps(index))
    read = runtime_observations.read_json

    def replace_after_read(path, **kwargs):
        value = read(path, **kwargs)
        if Path(path) == manifest_path:
            manifest_path.write_text(json.dumps(correct))
        return value

    monkeypatch.setattr(runtime_observations, "read_json", replace_after_read)
    with pytest.raises(ValueError, match="launch manifest collector_attempt_id"):
        execution_evidence._runtime_worker_evidence(plan, cell, raw, collection)


@pytest.fixture
def native_execution(snapshot, native_source, request, monkeypatch):
    from collector.fpm_forward import planner, runtime_probe
    from collector.fpm_forward.native_artifact import validate_native_collection

    bound = request.getfixturevalue("inputs")
    _, _, _, collection, index_path = snapshot
    generator_overrides = runtime_probe.probe_generator_overrides(bound["runtime_launch"])
    plan = planner.build_collection_plan(
        backend=bound["backend"],
        model_path=bound["model_path"],
        system=bound["system"],
        selected_ops=set(),
        options=bound["options"],
        has_model_cases=False,
        model_config_path=bound["runtime_launch"]["model_config"]["path"],
        fpm_profile=bound["profile"],
        runtime_instrumentation=bound["runtime_instrumentation"],
        runtime_configuration="tp2",
        runtime_launch=bound["runtime_launch"],
        generator_overrides=generator_overrides,
    )
    root = index_path.parent
    _write_campaign(
        root,
        root / "checkpoint.json",
        plan,
        attempt_id=collection.collector_attempt_id,
        generator_overrides=generator_overrides,
    )
    cell = plan.cells[0]
    raw = root / "cells" / cell.cell_id / "raw"
    for benchmark in raw.glob("**/benchmark*.json"):
        payload = json.loads(benchmark.read_text())
        if payload.get("artifact_type") == "rank":
            payload["kvwarm"]["initialization_strategy"] = "native_exact_context"
            benchmark.write_text(json.dumps(payload))
    index = json.loads(index_path.read_text())
    phase = index["configurations"]["tp2"]["attempts"][0]["phases"]["decode"]
    phase["cell_id"] = cell.cell_id
    path = root / phase["launch_manifest"]["path"]
    context = json.loads(path.read_text())
    context["cell_id"] = cell.cell_id
    phase["launch_manifest"]["sha256"] = _write(path, context)["sha256"]
    for ref in phase["artifacts"]:
        path = root / ref["path"]
        record = json.loads(path.read_text())
        record["cell_id"] = cell.cell_id
        ref["sha256"] = _write(path, record)["sha256"]
    index_path.write_text(json.dumps(index))
    workers, _ = execution_evidence._runtime_worker_evidence(plan, cell, raw, collection)
    classes = {
        "FLASHMLA_SPARSE": "vllm.v1.attention.backends.mla.flashmla_sparse.FlashMLASparseBackend",
        "FLASHINFER_MLA_SPARSE": (
            "vllm.v1.attention.backends.mla.flashinfer_mla_sparse.FlashInferMLASparseTRTLLMBackend"
        ),
        "DEEPSEEK_V32_INDEXER": "vllm.models.kimi_k3.common.dsa_cache.KimiK3IndexerBackend",
        "KIMI_K3_KDA": "vllm.models.kimi_k3.nvidia.kda_metadata.KimiK3KDAAttentionBackend",
    }
    version = observer.importlib.metadata.version
    monkeypatch.setattr(
        observer.importlib.metadata,
        "version",
        lambda name: plan.capability.aic_database_version if name == "vllm" else version(name),
    )
    for path in raw.glob("**/fpm-execution-worker-*.json"):
        payload = json.loads(path.read_text())
        worker = workers[tuple(payload[key] for key in ("dp_rank", "tp_rank", "pp_rank"))]
        for section, values in worker["resolved_config"].items():
            if isinstance(values, dict) and isinstance(payload["resolved_config"].get(section), dict):
                payload["resolved_config"][section].update(copy.deepcopy(values))
            else:
                payload["resolved_config"][section] = copy.deepcopy(values)
        # Recreate initialized runtime objects, then use the actual producer.
        # Copying the native record into the execution JSON masks fields that
        # the execution serializer omits.
        resolved = payload["resolved_config"]
        model = resolved["model_config"]
        hf_config = {"architectures": model.pop("architectures")}
        if "loaded_config_commit_hash" in model:
            hf_config["_commit_hash"] = model.pop("loaded_config_commit_hash")
        config = SimpleNamespace(
            **{section: SimpleNamespace(**values) for section, values in resolved.items() if isinstance(values, dict)}
        )
        config.model_config.hf_config = SimpleNamespace(**hf_config)
        quant = resolved["quantization_config"]
        config.quant_config = _typed(*quant["type"].rsplit(".", 1))
        for key, value in quant.items():
            if key != "type":
                setattr(config.quant_config, key, value)
        groups = [
            SimpleNamespace(
                kv_cache_group_id=group["group_id"],
                backend=type(_typed(*classes[group["backend"]].rsplit(".", 1))),
                layer_names=group["layer_names"],
            )
            for group in worker["attention_groups"]
        ]
        owner = SimpleNamespace(vllm_config=config, model_runner=SimpleNamespace(attn_groups=[groups]))
        path.unlink()
        monkeypatch.setattr(observer, "RESULTS_DIR", path.parent)
        observer.observe_execution(
            owner,
            **{key: payload[key] for key in ("dp_rank", "tp_rank", "pp_rank")},
            audited_version=plan.capability.aic_database_version,
        )
        emitted = json.loads(path.read_text())
        assert emitted["status"] == "observed"
        # The installed-source fixture above exercises the real producer with
        # synthetic file hashes. Bind only its source identity to the canonical
        # map used by the synthetic native observation fixture, leaving all
        # producer-serialized configuration and attention fields intact.
        synthetic_runtime = json.loads(native_source[1].read_text())["runtime"]
        assert emitted["runtime"] == {"framework": "vllm", **synthetic_runtime}
        for key in ("source_files", "source_revision"):
            emitted["runtime"][key] = worker["runtime"][key]
        path.write_text(json.dumps(emitted))
    collection = validate_native_collection(
        cell, raw, expected_plan_sha256=plan.sha256, expected_attempt_id=collection.collector_attempt_id
    )
    return plan, cell, raw, collection, index_path


@pytest.mark.parametrize("inputs", ["trtllm_mla", "flashmla"], indirect=True)
@pytest.mark.usefixtures("inputs")
def test_native_execution_producer_emits_required_model_and_attention_identity(native_execution):
    plan, cell, raw, collection, _ = native_execution
    for path in raw.glob("**/fpm-execution-worker-*.json"):
        config = json.loads(path.read_text())["resolved_config"]
        assert config["model_config"]["architectures"] == ["KimiK31ForConditionalGeneration"]
        assert config["attention_config"] == {
            "backend": "FLASHMLA_SPARSE"
            if cell.backend_policy.aic_fields["attention_backend"] == "flashmla"
            else "FLASHINFER_MLA_SPARSE",
            "indexer_kv_dtype": "fp8",
        }
    result = execution_evidence.inspect_execution_evidence(cell, raw, collection, plan=plan)
    assert result["status"] == "qualified", (result["missing_evidence"], result["failures"])


@pytest.mark.parametrize("image", [False, True])
def test_execution_accepts_matching_audited_runtime_identity(native_execution, image):
    plan, cell, raw, collection, index_path = native_execution
    if image:
        workers, _ = execution_evidence._runtime_worker_evidence(plan, cell, raw, collection)
        for path in raw.glob("**/fpm-execution-worker-*.json"):
            payload = json.loads(path.read_text())
            rank = tuple(payload[key] for key in ("dp_rank", "tp_rank", "pp_rank"))
            payload["runtime"]["image"] = workers[rank]["runtime"]["image"]
            path.write_text(json.dumps(payload))
    result = execution_evidence.inspect_execution_evidence(cell, raw, collection, plan=plan)
    assert result["status"] == "qualified", (result["missing_evidence"], result["failures"])


@pytest.mark.parametrize(
    "mutation", ["missing", "source_files", "missing_file", "source_revision", "framework", "version", "image"]
)
def test_execution_rejects_sidecar_runtime_contradicting_audited_worker(native_execution, mutation):
    plan, cell, raw, collection, _ = native_execution
    path = next(raw.glob("**/fpm-execution-worker-*.json"))
    payload = json.loads(path.read_text())
    runtime = payload["runtime"]
    if mutation == "missing":
        payload.pop("runtime")
    elif mutation in {"source_files", "missing_file"}:
        name = next(iter(runtime["source_files"]))
        if mutation == "missing_file":
            runtime["source_files"].pop(name)
        else:
            runtime["source_files"][name] = "0" * 64
        runtime["source_revision"] = hashlib.sha256(
            json.dumps(runtime["source_files"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    else:
        runtime[mutation] = "different-runtime-identity"
    path.write_text(json.dumps(payload))
    result = execution_evidence.inspect_execution_evidence(cell, raw, collection, plan=plan)
    assert result["status"] == "failed", result
    assert any("execution runtime source identity differs" in reason for reason in result["failures"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_parallel_size", 1),
        ("pipeline_parallel_size", 2),
        ("data_parallel_size", 1),
        ("data_parallel_size_local", 1),
        ("data_parallel_external_lb", True),
        ("enable_expert_parallel", False),
        ("enable_eplb", True),
        ("decode_context_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
    ],
)
def test_execution_rejects_placement_differing_from_initialized_worker(tmp_path, monkeypatch, field, value):
    from collector.fpm_forward.config import FPMCollectionOptions
    from collector.fpm_forward.runtime_observations import _launch_cell, validate_observations

    from .test_fpm_forward import _args
    from .test_k31_runtime_observations import _complete_fixture

    path, launches = _complete_fixture(tmp_path, monkeypatch)
    launch = launches["tp2"]
    imported = validate_observations(path, launches)["tp2"]
    assert imported["status"] == "complete", imported["diagnostics"]
    worker = next(a["evidence"] for a in imported["provenance"]["artifacts"] if a["evidence"]["kind"] == "worker")
    config = copy.deepcopy(worker["resolved_config"])
    payload = {"resolved_config": config, "graph_config": config["compilation_config"]}
    plan = SimpleNamespace(
        options=FPMCollectionOptions.from_args(
            _args(
                fpm_max_model_len=launch["collection"]["max_model_len"],
                fpm_gpu_memory_utilization=launch["collection"]["gpu_memory_utilization"],
            )
        ),
        fpm_profile=SimpleNamespace(model_revision=launch["identity"]["model_revision"]),
    )
    cell = _launch_cell(launch, "decode")
    missing, failures = [], []
    execution_evidence._inspect_config(payload, cell, plan, None, missing, failures, worker)
    assert not failures
    assert config["parallel_config"][field] != value
    config["parallel_config"][field] = value
    execution_evidence._inspect_config(payload, cell, plan, None, missing, failures, worker)
    assert f"runtime parallel_config.{field} differs from the validated initialized worker" in failures
    if field == "data_parallel_size_local":
        # Both placements are individually valid: local DP1 versus local
        # DP2, each within global DP2. Only binding the records detects it.
        assert config["parallel_config"][field] <= config["parallel_config"]["data_parallel_size"]
        assert not any("local DP size exceeds" in reason for reason in failures)


def test_execution_checks_initialized_worker_local_dp_bound(native_execution):
    plan, cell, raw, collection, index_path = native_execution
    # Keep the valid sidecar and global topology; the native configuration's
    # local placement alone contradicts its global DP size.
    _mutate(index_path, lambda record: record["resolved_config"]["parallel_config"].update(data_parallel_size_local=3))
    result = execution_evidence.inspect_execution_evidence(cell, raw, collection, plan=plan)
    assert result["status"] == "failed", result
    assert any("initialized worker local DP size exceeds global DP size" in s for s in result["failures"])


@pytest.mark.parametrize("damage", [None, "moe_precision", "state_precision"])
def test_publication_requires_matching_initialized_native_precision(native_execution, damage):
    from dataclasses import replace

    from collector.fpm_forward.database import aggregate_cell

    plan, cell, raw, collection, index = native_execution
    for path in raw.glob("**/fpm-execution-worker-*.json"):
        config = json.loads(path.read_text())["resolved_config"]["attention_config"]
        (path.parent / "resolved-config-native.json").write_text(
            json.dumps(
                {
                    "config": {
                        "engine_args": {
                            "attention_config": {**config, "backend": "AttentionBackendEnum." + config["backend"]}
                        }
                    }
                }
            )
        )
    if damage == "state_precision":
        from collector.fpm_forward.runtime_observations import validate_observations

        _mutate(index, lambda record: _set_state_dtypes(record, "bfloat16", "bfloat16", "bfloat16", "bfloat16"))
        standalone = validate_observations(index, {"tp2": plan.runtime_launch})["tp2"]
        assert standalone["status"] == "complete", standalone["diagnostics"]
        with pytest.raises(ValueError, match="formal collection requires effective float32 recurrent state"):
            aggregate_cell(plan, cell, raw.parent, expected_attempt_id=collection.collector_attempt_id)
    elif damage == "moe_precision":
        cell = replace(cell, moe_quant_mode="w4a16_mxfp4")
        with pytest.raises(ValueError, match="publication MoE precision differs"):
            aggregate_cell(plan, cell, raw.parent, expected_attempt_id=collection.collector_attempt_id)
    else:
        rows = aggregate_cell(plan, cell, raw.parent, expected_attempt_id=collection.collector_attempt_id)
        assert rows and {row["moe_quant_mode"] for row in rows} == {"w4a8_mxfp4_mxfp8_trtllm"}


@pytest.mark.parametrize("inputs", ["flashmla"], indirect=True)
@pytest.mark.parametrize("cache_dtype", ["fp8_ds_mla", "fp8"])
@pytest.mark.usefixtures("inputs")
def test_flashmla_native_execution_requires_matching_observed_cache_layout(native_execution, cache_dtype):
    plan, cell, raw, collection, _ = native_execution
    for path in raw.glob("**/fpm-execution-worker-*.json"):
        payload = json.loads(path.read_text())
        assert payload["resolved_config"]["cache_config"]["cache_dtype"] == "fp8_ds_mla"
        payload["resolved_config"]["cache_config"]["cache_dtype"] = cache_dtype
        path.write_text(json.dumps(payload))
    result = execution_evidence.inspect_execution_evidence(cell, raw, collection, plan=plan)
    assert result["status"] == ("qualified" if cache_dtype == "fp8_ds_mla" else "failed"), (
        result["missing_evidence"],
        result["failures"],
    )
    if cache_dtype != "fp8_ds_mla":
        assert any("KV precision/layout differs" in reason for reason in result["failures"])


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "backend",
        "layers",
        "group",
        "duplicate",
        "absent_index",
        "bad_index",
        "bad_hash",
        "bad_artifact",
        "wrong_attempt",
        "bundle_error",
    ],
)
def test_execution_requires_consistent_native_attention_and_bound_evidence(native_execution, mutation, monkeypatch):
    plan, cell, raw, collection, index_path = native_execution
    if mutation in {"backend", "layers", "group", "duplicate"}:
        path = next(raw.glob("**/fpm-execution-worker-*.json"))
        payload = json.loads(path.read_text())
        groups = payload["attention_groups"]
        if mutation == "backend":
            groups[0]["backend_class"] = "vllm.v1.attention.backends.mla.flashmla_sparse.FlashMLASparseBackend"
        elif mutation == "layers":
            groups[0]["layer_names"], groups[1]["layer_names"] = groups[1]["layer_names"], groups[0]["layer_names"]
        elif mutation == "group":
            groups[0]["kv_cache_group_id"] = 1
        else:
            groups.append(copy.deepcopy(groups[0]))
        path.write_text(json.dumps(payload))
    elif mutation == "absent_index":
        index_path.unlink()
    elif mutation == "bad_index":
        index_path.write_text("{malformed")
    elif mutation in {"bad_hash", "bad_artifact", "wrong_attempt"}:
        index = json.loads(index_path.read_text())
        phase = index["configurations"]["tp2"]["attempts"][0]["phases"]["decode"]
        if mutation == "bad_hash":
            phase["launch_manifest"]["sha256"] = "0" * 64
        elif mutation == "bad_artifact":
            phase["artifacts"][-1]["sha256"] = "0" * 64
        else:
            phase["collector_attempt_id"] = "wrong-attempt"
        index_path.write_text(json.dumps(index))
    reads = []
    original = type(plan).runtime_instrumentation

    def bundle_read(value):
        reads.append(True)
        if mutation == "bundle_error":
            raise ValueError("synthetic invalid instrumentation bundle")
        return original.fget(value)

    monkeypatch.setattr(type(plan), "runtime_instrumentation", property(bundle_read))
    result = execution_evidence.inspect_execution_evidence(cell, raw, collection, plan=plan)
    expected = "qualified" if mutation == "none" else "incomplete" if mutation == "absent_index" else "failed"
    assert result["status"] == expected, (result["missing_evidence"], result["failures"])
    assert len(reads) == 1
    assert len(result["observed_workers"]) == cell.topology.tp
    if mutation == "absent_index":
        assert not result["failures"]
        assert any("snapshot index is missing" in reason for reason in result["missing_evidence"])
    elif mutation != "none":
        assert result["failures"]
    if mutation in {"absent_index", "bad_index", "bad_hash", "bad_artifact", "wrong_attempt", "bundle_error"}:
        diagnostics = [*result["missing_evidence"], *result["failures"]]
        assert sum(reason.startswith("runtime model snapshot evidence:") for reason in diagnostics) == 1


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("other_attempt", "belongs to a different cell or native attempt"),
        ("other_cell", "belongs to a different cell or native attempt"),
        ("wrong_context", "launch manifest collector_attempt_id"),
        ("bad_config_hash", "loaded model configuration hash"),
        ("symlink", "snapshot index cannot be a symlink"),
    ],
)
def test_snapshot_identity_rejects_unbound_or_changed_native_evidence(snapshot, mutation, match):
    plan, cell, raw, collection, index_path = snapshot
    index = json.loads(index_path.read_text())
    entry = index["configurations"]["tp2"]
    attempt = next(a for a in entry["attempts"] if a["attempt_id"] == entry["active_attempt_id"])
    phase = attempt["phases"]["decode"]
    if mutation == "other_attempt":
        phase["collector_attempt_id"] = "other"
    elif mutation == "other_cell":
        phase["cell_id"] = "other"
    elif mutation == "wrong_context":
        path = index_path.parent / phase["launch_manifest"]["path"]
        context = json.loads(path.read_text())
        context["collector_attempt_id"] = "other"
        phase["launch_manifest"]["sha256"] = _write(path, context)["sha256"]
    elif mutation == "bad_config_hash":
        for ref in phase["artifacts"]:
            path = index_path.parent / ref["path"]
            record = json.loads(path.read_text())
            if record["kind"] == "worker":
                record["model_config_sha256"] = "0" * 64
                ref["sha256"] = _write(path, record)["sha256"]
                break
    index_path.write_text(json.dumps(index))
    if mutation == "symlink":
        moved = index_path.with_suffix(".original")
        index_path.rename(moved)
        index_path.symlink_to(moved)
    with pytest.raises(ValueError, match=match):
        execution_evidence._runtime_worker_evidence(plan, cell, raw, collection)


@pytest.mark.parametrize(
    "mutation", ["wrong_native_model", "wrong_native_dtype", "wrong_commit", "wrong_loaded_hash", "missing_none_field"]
)
def test_snapshot_alias_fallback_does_not_weaken_identity(snapshot, mutation):
    plan, _, _, _, _ = snapshot
    revision = "a" * 40
    public = plan.runtime_launch["identity"]["model"]
    model = {
        "model": f"/cache/models--{public.replace('/', '--')}/snapshots/{revision}",
        "revision": revision,
        "dtype": "torch.bfloat16",
    }
    native = copy.deepcopy(model)
    if mutation == "wrong_native_model":
        native["model"] += "-other"
    elif mutation == "wrong_native_dtype":
        native["dtype"] = "torch.float16"
    elif mutation == "wrong_commit":
        model["model"] = model["model"].replace(revision, "0" * 40)
    elif mutation == "missing_none_field":
        model["loaded_config_commit_hash"] = None
    else:
        model["loaded_config_commit_hash"] = "0" * 40
        native["loaded_config_commit_hash"] = "0" * 40
    assert not execution_evidence._matches_pinned_hub_snapshot(public, model, {revision}, native)


@pytest.mark.parametrize(
    "mode", ["verified", "no_native_evidence", "different_quantization", "different_model_dtype", "wrong_cell_gemm"]
)
def test_mxfp4_execution_requires_matching_initialized_native_evidence(snapshot, mode):
    from collector.fpm_forward.runtime_observations import _launch_cell
    from collector.fpm_forward.runtime_probe import build_runtime_probe_plan

    plan, cell, raw, collection, _ = snapshot
    workers, _ = execution_evidence._runtime_worker_evidence(plan, cell, raw, collection)
    worker = workers[(0, 0, 0)]
    config = copy.deepcopy(worker["resolved_config"])
    payload = {"resolved_config": config, "graph_config": config["compilation_config"]}
    probe = build_runtime_probe_plan(plan.runtime_configuration, plan.runtime_launch, plan.runtime_instrumentation)
    active = SimpleNamespace(options=probe.options, fpm_profile=SimpleNamespace(model_revision="immutable"))
    native_cell = _launch_cell(plan.runtime_launch, "decode")
    if mode == "different_quantization":
        config["quantization_config"]["type"] = "other.Quantization"
    elif mode == "different_model_dtype":
        config["model_config"]["dtype"] = "torch.float16"
    elif mode == "wrong_cell_gemm":
        from dataclasses import replace

        native_cell = replace(native_cell, gemm_quant_mode="fp8")
    missing, failures = [], []
    execution_evidence._inspect_config(
        payload, native_cell, active, None, missing, failures, None if mode == "no_native_evidence" else worker
    )
    if mode == "verified":
        assert not failures
        # This snapshot fixture omits the optional block-count override.
        assert missing == ["runtime cache_config.num_gpu_blocks_override is unreported"]
    assert ("runtime quantization mapping is unaudited: mxfp4" in missing) == (mode == "no_native_evidence")
    assert ("runtime MXFP4 execution differs from the validated initialized worker" in failures) == mode.startswith(
        "different_"
    )
    assert any("planned GEMM/MoE identity" in failure for failure in failures) == (mode == "wrong_cell_gemm")


@pytest.mark.parametrize("inputs", ["flashmla"], indirect=True)
@pytest.mark.parametrize("architectures", [None, [], ["OtherModel"]])
@pytest.mark.usefixtures("inputs")
def test_execution_cannot_bypass_bound_native_architecture(native_execution, architectures):
    plan, cell, raw, collection, _ = native_execution
    path = next(raw.glob("**/fpm-execution-worker-*.json"))
    payload = json.loads(path.read_text())
    model = payload["resolved_config"]["model_config"]
    if architectures is None:
        model.pop("architectures")
    else:
        model["architectures"] = architectures
    payload["resolved_config"]["cache_config"]["cache_dtype"] = "fp8"
    path.write_text(json.dumps(payload))
    result = execution_evidence.inspect_execution_evidence(cell, raw, collection, plan=plan)
    assert result["status"] == "failed"
    assert any("architecture differs" in reason for reason in result["failures"])
    assert any("KV precision/layout differs" in reason for reason in result["failures"])


@pytest.mark.parametrize("graph_mode", ["FULL_AND_PIECEWISE", "PIECEWISE"])
def test_execution_graphs_must_match_bound_worker(native_execution, graph_mode):
    plan, cell, raw, collection, _ = native_execution
    path = next(raw.glob("**/fpm-execution-worker-*.json"))
    payload = json.loads(path.read_text())
    assert payload["graph_config"]["cudagraph_mode"] == "FULL_AND_PIECEWISE"
    for graph in (payload["graph_config"], payload["resolved_config"]["compilation_config"]):
        graph["cudagraph_mode"] = graph_mode
        # Equivalent serialized compilation modes normalize before comparison.
        graph["mode"] = 3
    path.write_text(json.dumps(payload))
    result = execution_evidence.inspect_execution_evidence(cell, raw, collection, plan=plan)
    assert result["status"] == ("qualified" if graph_mode == "FULL_AND_PIECEWISE" else "failed"), result
    if graph_mode == "PIECEWISE":
        assert any("graph settings differ" in reason for reason in result["failures"])


@pytest.mark.parametrize("shapes", [["w13_weight", "w2_weight"], None, "weights"])
def test_execution_reports_malformed_initialized_weight_shapes(native_execution, shapes):
    plan, cell, raw, collection, index_path = native_execution

    def corrupt(record):
        if record["kind"] == "worker":
            record["initialized_moe"]["layers"][0]["weight_shapes"] = shapes

    _mutate(index_path, corrupt)
    result = execution_evidence.inspect_execution_evidence(cell, raw, collection, plan=plan)
    assert result["status"] == "failed"
    assert any("initialized expert weights are incomplete" in reason for reason in result["failures"])


@pytest.mark.parametrize("inputs", ["explicit-moe"], indirect=True)
@pytest.mark.parametrize("kernel", [["invalid"], "invalid", 1, None, {"moe_backend": "flashinfer_trtllm"}])
@pytest.mark.usefixtures("inputs")
def test_explicit_moe_pin_reports_malformed_native_kernel_config(native_execution, kernel):
    plan, cell, raw, collection, index_path = native_execution
    assert cell.backend_policy.aic_fields["moe_backend"] == "flashinfer_trtllm"

    def corrupt(record):
        if record["kind"] == "worker":
            record["resolved_config"]["kernel_config"] = kernel

    _mutate(index_path, corrupt)
    result = execution_evidence.inspect_execution_evidence(cell, raw, collection, plan=plan)
    assert result["status"] == ("qualified" if isinstance(kernel, dict) else "failed"), result
    if not isinstance(kernel, dict):
        assert any("kernel_config object" in reason for reason in result["failures"])


@pytest.mark.parametrize(
    "mutation",
    [
        "null_phase",
        "null_hardware",
        "list_hardware",
        "list_classes",
        "null_placement",
        "null_cache",
        "null_group",
        "null_component",
        "null_scheduler_group",
    ],
)
def test_execution_reports_malformed_native_objects_as_failed(native_execution, mutation):
    plan, cell, raw, collection, index_path = native_execution
    if mutation == "null_phase":
        index = json.loads(index_path.read_text())
        index["configurations"]["tp2"]["attempts"][0]["phases"]["decode"] = None
        index_path.write_text(json.dumps(index))
    else:

        def corrupt(record):
            if record["kind"] == "worker" and mutation != "null_scheduler_group":
                if mutation in {"null_hardware", "list_hardware"}:
                    record["hardware"] = None if mutation == "null_hardware" else []
                elif mutation == "list_classes":
                    group = record["cache"]["groups"][-1]
                    group["layer_classes"] = group["layer_names"]
                elif mutation == "null_cache":
                    record["cache"] = None
                elif mutation == "null_group":
                    record["cache"]["groups"][0] = None
                elif mutation == "null_component":
                    record["cache"]["layer_components"][record["cache"]["groups"][0]["layer_names"][0]] = [None]
                else:
                    record["cache"]["tensor_allocations"][0] = None
            elif record["kind"] == "scheduler" and mutation == "null_scheduler_group":
                record["cache"]["groups"][0] = None

        _mutate(index_path, corrupt)
    result = execution_evidence.inspect_execution_evidence(cell, raw, collection, plan=plan)
    assert result["status"] == "failed", result
    message = {
        "null_phase": "phase must be an object",
        "null_hardware": "hardware must be an object",
        "list_hardware": "hardware must be an object",
        "list_classes": "class coverage",
        "null_placement": "placement",
        "null_cache": "cache must be an object",
        "null_group": "cache group must be an object",
        "null_component": "layer component must be an object",
        "null_scheduler_group": "cache group must be an object",
    }
    assert any(message[mutation] in reason for reason in result["failures"])


@pytest.mark.parametrize("role", ["worker", "scheduler"])
@pytest.mark.parametrize("lifecycle", [None, [], "initialized"])
def test_execution_reports_malformed_lifecycle_as_failed(native_execution, role, lifecycle):
    plan, cell, raw, collection, index_path = native_execution

    def corrupt(record):
        if record["kind"] == role:
            record["lifecycle"] = lifecycle

    _mutate(index_path, corrupt)
    result = execution_evidence.inspect_execution_evidence(cell, raw, collection, plan=plan)
    assert result["status"] == "failed", result
    assert any("lifecycle must be an object" in reason for reason in result["failures"])


@pytest.mark.parametrize("group_index", [0, 1, 2])
def test_matching_sidecars_cannot_override_selected_native_backend(native_execution, group_index):
    plan, cell, raw, collection, index_path = native_execution

    def corrupt(record):
        if record["kind"] == "worker":
            record["attention_groups"][group_index]["backend"] = "FLASHMLA_SPARSE"

    _mutate(index_path, corrupt)
    for path in raw.glob("**/fpm-execution-worker-*.json"):
        payload = json.loads(path.read_text())
        payload["attention_groups"][group_index]["backend_class"] = (
            "vllm.v1.attention.backends.mla.flashmla_sparse.FlashMLASparseBackend"
        )
        path.write_text(json.dumps(payload))
    result = execution_evidence.inspect_execution_evidence(cell, raw, collection, plan=plan)
    assert result["status"] == "failed", result
    assert any("backend differs from the selected policy" in reason for reason in result["failures"])


@pytest.mark.parametrize("snapshot", [True], indirect=True)
@pytest.mark.parametrize("has_native_evidence", [True, False])
def test_hashless_hub_snapshot_alias_qualifies_only_with_bound_worker(native_execution, has_native_evidence):
    plan, cell, raw, collection, index_path = native_execution
    for path in raw.glob("**/fpm-execution-worker-*.json"):
        model = json.loads(path.read_text())["resolved_config"]["model_config"]
        assert model["model"].endswith("/snapshots/" + "a" * 40)
        assert "loaded_config_commit_hash" not in model
    if not has_native_evidence:
        index_path.unlink()
    result = execution_evidence.inspect_execution_evidence(cell, raw, collection, plan=plan)
    assert result["status"] == ("qualified" if has_native_evidence else "failed"), result
    if has_native_evidence:
        assert all(worker["runtime_observation_binding"] for worker in result["observed_workers"])
    else:
        assert any("snapshot index is missing" in reason for reason in result["missing_evidence"])
        assert "runtime model_config.model differs from generated --model" in result["failures"]


@pytest.mark.parametrize("role", ["worker", "scheduler"])
@pytest.mark.parametrize("field", ["cell_id", "collector_attempt_id"])
@pytest.mark.parametrize("stale", [False, True])
def test_each_initialized_record_is_bound_to_selected_collector_attempt(snapshot, role, field, stale):
    plan, cell, raw, collection, index_path = snapshot

    def unbind(record):
        if record["kind"] == role and (role != "worker" or record["tp_rank"] == 1):
            if stale:
                record[field] = "previous-collector-value"
            else:
                record.pop(field)

    # The phase and launch manifest retain the current binding. Rehashing an
    # older record must not let it supply the retry's initialized evidence.
    _mutate(index_path, unbind)
    with pytest.raises(ValueError, match=field):
        execution_evidence._runtime_worker_evidence(plan, cell, raw, collection)


@pytest.mark.parametrize("inputs", ["explicit-moe"], indirect=True)
@pytest.mark.parametrize("backend", ["flashinfer_trtllm", "auto"])
@pytest.mark.usefixtures("inputs")
def test_execution_moe_backend_matches_each_bound_native_worker(native_execution, backend):
    plan, cell, raw, collection, _ = native_execution
    path = sorted(raw.glob("**/fpm-execution-worker-*.json"))[-1]
    payload = json.loads(path.read_text())
    payload["resolved_config"]["kernel_config"]["moe_backend"] = backend
    path.write_text(json.dumps(payload))
    result = execution_evidence.inspect_execution_evidence(cell, raw, collection, plan=plan)
    assert result["status"] == ("qualified" if backend == "flashinfer_trtllm" else "failed"), result
    if backend == "auto":
        assert any("kernel_config.moe_backend differs" in reason for reason in result["failures"])


@pytest.mark.parametrize("kernel", [None, []])
def test_execution_reports_unresolved_native_moe_backend_without_crashing(native_execution, kernel):
    plan, cell, raw, collection, index_path = native_execution
    _mutate(index_path, lambda record: record["resolved_config"].update(kernel_config=kernel))
    result = execution_evidence.inspect_execution_evidence(cell, raw, collection, plan=plan)
    assert result["status"] == "failed", result
    assert any("kernel_config.moe_backend differs" in reason for reason in result["failures"])


def test_packaged_k31_manifest_binds_selected_trtllm_expert_implementation():
    from collector.fpm_forward import runtime_k31

    manifest = json.loads((Path(runtime_k31.__file__).parent / "runtime/vllm-kimi-k31.example.json").read_text())
    runtime = manifest["runtime"]
    source = "vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py"
    # Exact upstream Git blob at the immutable revision cited in source notes.
    assert runtime["source_files"][source] == "6cd47f0edf4dd941138e482b63cd1d366a25d7290bbaa57579827ef309600b6f"
    assert (
        runtime["source_revision"]
        == hashlib.sha256(
            json.dumps(runtime["source_files"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert runtime_k31.is_audited_runtime(runtime)
    changed = copy.deepcopy(runtime)
    changed["source_files"][source] = "0" * 64
    assert not runtime_k31.is_audited_runtime(changed)


def test_cell_acceptance_revalidates_archived_worker_evidence(snapshot):
    from collector.fpm_forward.database import require_k31_worker_evidence

    plan, cell, raw, collection, index = snapshot
    plan.capability = SimpleNamespace(architecture="KimiK31ForConditionalGeneration")
    cell.moe_quant_mode = "w4a8_mxfp4_mxfp8_trtllm"
    require_k31_worker_evidence(plan, cell, raw, collection)

    def unresolved(record):
        if record["kind"] == "worker" and record["tp_rank"] == 1:
            record.update(status="unresolved", error="synthetic initialized evidence failure")

    _mutate(index, unresolved)
    with pytest.raises(ValueError, match="synthetic initialized evidence failure|unresolved|observed"):
        require_k31_worker_evidence(plan, cell, raw, collection)
