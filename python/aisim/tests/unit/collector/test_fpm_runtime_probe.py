# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch-only probes use real rendering and synthetic executor evidence."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from .test_fpm_profile_collection import no_models_or_timing_data  # noqa: F401

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_models_or_timing_data")]


@pytest.fixture(autouse=True)
def shared_private_delivery_root(tmp_path, monkeypatch):
    # Synthetic executors see this caller-selected shared-storage stand-in.
    monkeypatch.setenv("AISIMULATE_PRIVATE_DELIVERY_ROOT", str(tmp_path / "shared-private-delivery"))


def _inputs(tmp_path, *, graph_policy="runtime"):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "architectures": ["ExternalMoeForCausalLM"],
                "hidden_size": 128,
                "intermediate_size": 256,
                "moe_intermediate_size": 256,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "num_hidden_layers": 5,
                "max_position_embeddings": 8192,
                "vocab_size": 1024,
                "num_experts": 8,
                "torch_dtype": "bfloat16",
            }
        )
    )
    source = tmp_path / "observer.py"
    marker = tmp_path / "observer-was-imported"
    source.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('never import')\n")
    (tmp_path / "source-notes.txt").write_text("Synthetic source mapping fixture; no live verification.")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "aisimulate-runtime-instrumentation/v1",
                "runtime": {
                    "framework": "vllm",
                    "version": "0.28.0",
                    "source_revision": "a" * 40,
                    "source_files": {"vllm/v1/core/block_pool.py": "b" * 64},
                },
                "files": ["observer.py", "source-notes.txt"],
                "worker_class": "observer.ObservedWorker",
                "scheduler_class": "observer.ObservedScheduler",
                "observation_schema": "aisimulate-runtime-observation/v1",
                "source_notes": "source-notes.txt",
            }
        )
    )
    launch = {
        "identity": {
            "model": "example/external-moe",
            "model_revision": "pinned-model-revision",
            "model_kind": "moe",
            "framework": "vllm",
            "framework_version": "0.28.0",
            "gpu": "gb200",
            "interconnect": "nvlink",
        },
        "topology": {"tp": 4, "pp": 1, "dp": 1, "moe_tp": 1, "moe_ep": 4, "cp": 1},
        "precision": {
            "gemm_quant_mode": "bfloat16",
            "moe_quant_mode": "bfloat16",
            "fmha_quant_mode": "bfloat16",
            "kvcache_quant_mode": "bfloat16",
            "comm_quant_mode": "half",
            "moe_backend": "auto",
            "attention_backend": "auto",
            "enable_wideep": False,
            "enable_eplb": False,
        },
        "collection": {
            "max_model_len": 4096,
            "max_num_batched_tokens": 1024,
            "max_num_seqs": 64,
            "gpu_memory_utilization": 0.85,
            "prefill_cudagraph_policy": graph_policy,
            "max_prefill_cudagraph_size": 512 if graph_policy == "explicit" else None,
        },
        "model_config": {"path": str(config), "sha256": hashlib.sha256(config.read_bytes()).hexdigest()},
        "deployment": {
            "executor": "slurm",
            "image": "image@sha256:synthetic",
            "container_mount": ["/cache:/cache"],
            "cpus_per_task": 16,
            "cpu_bind": "cores",
        },
    }
    return launch, manifest, marker


@pytest.mark.parametrize("mode", [None, "NONE"])
def test_probe_automatically_selects_eager_from_v41_config(tmp_path, mode):
    from collector.fpm_forward import runner, runtime_probe
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation

    launch, manifest, marker = _inputs(tmp_path)
    # Keep the arbitrary model identifier: selection must use actual config metadata.
    config = Path(launch["model_config"]["path"])
    payload = json.loads(config.read_text())
    payload["architectures"] = ["DeepseekV41ForCausalLM"]
    config.write_text(json.dumps(payload))
    launch["model_config"]["sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    launch["worker_type"] = "aggregated"
    launch["collection"]["cudagraph_mode"] = mode
    bundle = load_instrumentation(manifest)
    plan = runtime_probe.build_runtime_probe_plan("v41", launch, bundle)
    assert plan.launch["collection"]["enforce_eager"] is True
    assert runtime_probe.build_runtime_probe_plan("v41", plan.launch, bundle).sha256 == plan.sha256
    for cell in plan.cells:
        request = runner._cell_generator_overrides(plan, cell, {})
        assert "--enforce-eager" in request["params"]["agg"]["extra_cli_args"]
    assert not marker.exists()
    assert "enforce_eager" not in launch["collection"]


class _SyntheticExecutor:
    def __init__(self, plan, cell, cell_dir, executions, *, fail_configuration=None):
        self.plan, self.cell, self.cell_dir = plan, cell, cell_dir
        self.executions = executions
        self.fail_configuration = fail_configuration

    def cleanup(self):
        pass

    def apply(self):
        pass

    def wait_ready(self, expected_nodes):
        return ["synthetic-node", *(f"synthetic-node-{index}" for index in range(1, expected_nodes))]

    def stage(self, pods, files):
        self.files = {path.name: path for path in files}
        assert "runtime-instrumentation.zip" in self.files
        assert "runtime-probe-context.json" in self.files

    def prepare_attempt(self, pods, **kwargs):
        self.identity = kwargs

    def execute(self, pods):
        context = json.loads(self.files["runtime-probe-context.json"].read_text())
        self.executions.append((context["configuration"], context["phase"]))
        for pod in pods:
            raw = self.cell_dir / "raw" / pod
            raw.mkdir(parents=True, exist_ok=True)
            (raw / "synthetic-observation.json").write_text(
                json.dumps({**context, "schema_version": "aisimulate-runtime-observation/v1", "synthetic": True})
            )
        if context["configuration"] == self.fail_configuration:
            raise RuntimeError("synthetic runtime failure")

    def collect(self, pods, *, require_benchmark=True):
        pass


@pytest.mark.parametrize("worker_type", ["prefill", "decode", "aggregated"])
@pytest.mark.parametrize("graph_policy", ["runtime", "explicit"])
def test_serving_role_probe_preserves_engine_settings_and_only_exercises_its_phases(
    tmp_path, monkeypatch, worker_type, graph_policy
):
    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, _ = _inputs(tmp_path, graph_policy=graph_policy)
    launch["worker_type"] = worker_type
    mode = {"prefill": "PIECEWISE", "decode": "FULL_DECODE_ONLY", "aggregated": "FULL_AND_PIECEWISE"}[worker_type]
    launch["collection"]["cudagraph_mode"] = mode
    if graph_policy == "runtime":
        launch["collection"]["cudagraph_capture_sizes"] = [1, 2, 4, 8, 16, 32, 64]
    monkeypatch.setattr(runner, "_cell_runner", lambda *_args: pytest.fail("preview launched an executor"))
    result = runtime_probe.probe_runtime({"selected": launch}, instrumentation=manifest, output_dir=tmp_path / "probe")
    assert result["status"] == "preview", result
    expected = {"prefill", "decode"} if worker_type == "aggregated" else {worker_type}
    phases = result["configurations"]["selected"]["phases"]
    assert set(phases) == expected
    configs = []
    for phase in phases.values():
        request = json.loads(Path(phase["generator_request"]).read_text())
        args = request["params"]["agg"]["extra_cli_args"]
        configs.append(json.loads(args[args.index("--compilation-config") + 1]))
        assert args[args.index("--max-num-batched-tokens") + 1] == "1024"
        assert args[args.index("--max-num-seqs") + 1] == "64"
    assert all(config == configs[0] for config in configs)
    assert configs[0]["cudagraph_mode"] == mode
    assert configs[0]["max_cudagraph_capture_size"] == (512 if graph_policy == "explicit" else 64)


def test_disaggregated_probe_failure_keeps_other_role_resumable(tmp_path, monkeypatch):
    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, _ = _inputs(tmp_path)
    configurations = {role: {**copy.deepcopy(launch), "worker_type": role} for role in ("prefill", "decode")}
    executions = []
    monkeypatch.setattr(
        runner,
        "_cell_runner",
        lambda plan, cell, _manifest, directory: _SyntheticExecutor(
            plan, cell, directory, executions, fail_configuration="prefill"
        ),
    )
    output = tmp_path / "probe"
    result = runtime_probe.probe_runtime(configurations, instrumentation=manifest, output_dir=output, execute=True)
    assert result["status"] == "partial"
    assert result["configurations"]["prefill"]["status"] == "failed"
    assert result["configurations"]["decode"]["status"] == "completed"
    assert executions == [("prefill", "prefill"), ("decode", "decode")]
    monkeypatch.setattr(runner, "_cell_runner", lambda *_args: pytest.fail("completed role launched again"))
    resumed = runtime_probe.probe_runtime(
        {"decode": configurations["decode"]}, instrumentation=manifest, output_dir=output, execute=True, resume=True
    )
    assert resumed["configurations"]["decode"]["resumed"] is True


@pytest.mark.parametrize(
    "changes",
    [
        {"cudagraph_capture_sizes": [8, 4]},
        {"cudagraph_capture_sizes": [4, 8], "max_cudagraph_capture_size": 16},
        {"max_cudagraph_capture_size": 2048},
        {"cudagraph_mode": "NONE", "cudagraph_capture_sizes": [4]},
        {"cudagraph_mode": "UNKNOWN"},
    ],
)
def test_invalid_role_graph_settings_fail_before_execution_without_blocking_other_role(tmp_path, monkeypatch, changes):
    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, _ = _inputs(tmp_path)
    invalid = {**copy.deepcopy(launch), "worker_type": "prefill"}
    invalid["collection"].update(changes)
    valid = {**copy.deepcopy(launch), "worker_type": "decode"}
    monkeypatch.setattr(runner, "_cell_runner", lambda *_args: pytest.fail("preview launched an executor"))
    result = runtime_probe.probe_runtime(
        {"prefill": invalid, "decode": valid}, instrumentation=manifest, output_dir=tmp_path / "probe"
    )
    assert result["status"] == "partial"
    assert result["configurations"]["prefill"]["status"] == "failed"
    assert result["configurations"]["decode"]["status"] == "preview"


@pytest.mark.parametrize("graph_policy", ["runtime", "explicit"])
def test_preview_requires_no_cache_geometry_and_never_imports_bundle(tmp_path, monkeypatch, graph_policy):
    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, marker = _inputs(tmp_path, graph_policy=graph_policy)
    monkeypatch.setattr(runner, "_cell_runner", lambda *_args: pytest.fail("preview launched an executor"))
    result = runtime_probe.probe_runtime(
        {"arbitrary-label": launch}, instrumentation=manifest, output_dir=tmp_path / "probe"
    )

    assert result["status"] == "preview"
    assert not marker.exists()
    preview = result["configurations"]["arbitrary-label"]
    assert preview["minimum_gpus"] == 4
    assert set(preview["phases"]) == {"prefill", "decode"}
    for phase, info in preview["phases"].items():
        params = json.loads(Path(info["generator_request"]).read_text())
        args = params["params"]["agg"]["extra_cli_args"]
        assert args[args.index("--max-model-len") + 1] == "4096"
        assert args[args.index("--max-num-batched-tokens") + 1] == "1024"
        assert args[args.index("--max-num-seqs") + 1] == "64"
        assert params["params"]["agg"]["kv_cache_free_gpu_memory_fraction"] == 0.85
        assert args[args.index("--revision") + 1] == launch["identity"]["model_revision"]
        assert args.count("--revision") == 1
        assert args[args.index("--worker-cls") + 1] == "fpm_runtime_instrumentation.ObservedWorker"
        assert "--no-async-scheduling" in args
        batch_flag = "--prefix-max-batch-size-samples" if phase == "prefill" else "--decode-max-batch-size-samples"
        context = json.loads((tmp_path / "probe" / info["launch_manifest"]["path"]).read_text())
        assert int(args[args.index(batch_flag) + 1]) == context["sampling"]["max_batch_size_samples"] == 2
        if phase == "prefill":
            assert ("--compilation-config" in args) == (graph_policy == "explicit")
            assert args[args.index("--prefill-max-new-token-samples") + 1] == "2"
            if graph_policy == "explicit":
                graph = json.loads(args[args.index("--compilation-config") + 1])
                assert graph["max_cudagraph_capture_size"] == 512


@pytest.mark.parametrize("executor", ["kubernetes", "slurm"])
def test_multinode_preview_labels_every_resource(tmp_path, monkeypatch, executor):
    from collector.fpm_forward import runner, runtime_probe
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation

    launch, manifest, marker = _inputs(tmp_path)
    launch["identity"]["gpu"] = "gb300"
    if executor == "kubernetes":
        launch["deployment"] = {"executor": executor, "image": launch["deployment"]["image"]}
    launch["topology"].update(tp=8, dp=1, moe_tp=8, moe_ep=1)
    dep = copy.deepcopy(launch)
    dep["topology"].update(tp=1, dp=8, moe_tp=1, moe_ep=8)
    configurations = {"tp8": launch, "dep8": dep}
    monkeypatch.setattr(runner, "_cell_runner", lambda *_args: pytest.fail("preview launched an executor"))
    output = tmp_path / "probe"

    result = runtime_probe.probe_runtime(configurations, instrumentation=manifest, output_dir=output)

    assert result["status"] == "preview", result
    assert not marker.exists()
    bundle = load_instrumentation(manifest)
    for name, facts in configurations.items():
        plan = runtime_probe.build_runtime_probe_plan(name, facts, bundle)
        preview = result["configurations"][name]
        assert preview["minimum_gpus"] == 8
        assert set(preview["phases"]) == {"prefill", "decode"}
        for cell in plan.cells:
            phase = preview["phases"][cell.workload_kind]
            resource = output / phase["resource_manifest"]["path"]
            documents = runner._manifest_documents(resource)
            assert [document["kind"] for document in documents] == ["ComputeDomain", "LeaderWorkerSet"]
            assert runner._expected_nodes(resource) == 2
            expected = {
                "aiconfigurator.nvidia.com/owned-by": "fpm-forward-collector",
                "aiconfigurator.nvidia.com/plan": plan.sha256[:16],
                runner.FPM_CELL_LABEL: cell.cell_id,
            }
            for document in documents:
                assert expected.items() <= document["metadata"]["labels"].items()
            assert documents[0]["spec"]["numNodes"] == 0


@pytest.mark.parametrize("flag", [["--revision", "another-model-revision"], ["--revision=another-model-revision"]])
def test_runtime_revision_cannot_be_replaced_by_backend_policy(tmp_path, flag):
    from dataclasses import replace

    from collector.fpm_forward import runner, runtime_probe
    from collector.fpm_forward.planner import BackendPolicy
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation

    launch, manifest, _marker = _inputs(tmp_path)
    plan = runtime_probe.build_runtime_probe_plan("worker", launch, load_instrumentation(manifest))
    cell = replace(
        plan.cells[0], backend_policy=BackendPolicy("test", {"params": {"agg": {"extra_cli_args": flag}}}, {})
    )
    with pytest.raises(ValueError, match="selected model revision"):
        runner._cell_generator_overrides(plan, cell, runtime_probe.probe_generator_overrides(plan.launch))


def test_execute_covers_each_configuration_and_both_phases_then_resumes(tmp_path, monkeypatch):
    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, marker = _inputs(tmp_path)
    # Fresh callers may omit the default CPU proposal. Preview freezes it,
    # and resume must recover that same explicit identity from the saved plan.
    launch["deployment"].pop("cpus_per_task")
    launch["deployment"].pop("cpu_bind")
    second = copy.deepcopy(launch)
    second["topology"].update(tp=1, dp=4)
    configurations = {"tp4": launch, "dep4": second}
    executions = []
    monkeypatch.setattr(
        runner,
        "_cell_runner",
        lambda plan, cell, _manifest, directory: _SyntheticExecutor(plan, cell, directory, executions),
    )
    output = tmp_path / "probe"
    runtime_probe.probe_runtime(configurations, instrumentation=manifest, output_dir=output)
    result = runtime_probe.probe_runtime(configurations, instrumentation=manifest, output_dir=output, execute=True)
    assert result["status"] == "completed"
    assert executions == [(label, phase) for label in configurations for phase in ("prefill", "decode")]
    assert not marker.exists()
    index = json.loads((output / "observations.json").read_text())
    original = {
        str(output / artifact["path"]): (output / artifact["path"]).read_bytes()
        for entry in index["configurations"].values()
        for phase in entry["attempts"][0]["phases"].values()
        for artifact in phase["artifacts"]
        if artifact["kind"] == "observation"
    }
    assert len(original) == 4
    for entry in index["configurations"].values():
        assert len(entry["attempts"]) == 1
        assert set(entry["attempts"][0]["phases"]) == {"prefill", "decode"}
    runtime_probe.probe_runtime(configurations, instrumentation=manifest, output_dir=output, execute=True, resume=True)
    assert len(executions) == 4
    assert all(Path(path).read_bytes() == contents for path, contents in original.items())
    # Resume rechecks collector-owned launch bytes as well as observations.
    altered = output / index["configurations"]["tp4"]["attempts"][0]["phases"]["prefill"]["artifacts"][0]["path"]
    altered.write_bytes(altered.read_bytes() + b"\n")
    runtime_probe.probe_runtime(configurations, instrumentation=manifest, output_dir=output, execute=True, resume=True)
    assert len(executions) == 6
    updated = json.loads((output / "observations.json").read_text())
    assert len(updated["configurations"]["tp4"]["attempts"]) == 2
    assert len(updated["configurations"]["dep4"]["attempts"]) == 1


def test_configuration_failure_preserves_successes_and_new_bundle_gets_new_attempt(tmp_path, monkeypatch):
    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, _marker = _inputs(tmp_path)
    second = copy.deepcopy(launch)
    second["topology"].update(tp=1, dp=4)
    executions = []
    monkeypatch.setattr(
        runner, "_salvage_artifacts", lambda resource, _cell: resource.collect([], require_benchmark=False)
    )
    monkeypatch.setattr(
        runner,
        "_cell_runner",
        lambda plan, cell, _manifest, directory: _SyntheticExecutor(
            plan, cell, directory, executions, fail_configuration="broken"
        ),
    )
    output = tmp_path / "probe"
    result = runtime_probe.probe_runtime(
        {"broken": launch, "good": second}, instrumentation=manifest, output_dir=output, execute=True
    )
    assert result["status"] == "partial"
    assert len(executions) == 4
    index = json.loads((output / "observations.json").read_text())
    first = index["configurations"]["good"]["attempts"][0]
    preserved = (output / first["bundle"]["manifest"]).read_bytes()
    (tmp_path / "observer.py").write_text("# revised synthetic observer\n")
    runtime_probe.probe_runtime(
        {"good": second}, instrumentation=manifest, output_dir=output, execute=True, resume=True
    )
    updated = json.loads((output / "observations.json").read_text())
    attempts = updated["configurations"]["good"]["attempts"]
    assert len(attempts) == 2
    assert attempts[0] == first
    assert attempts[0]["bundle"]["sha256"] != attempts[1]["bundle"]["sha256"]
    assert (output / first["bundle"]["manifest"]).read_bytes() == preserved


@pytest.mark.parametrize(
    "provided", [{"cpus_per_task": 32}, {"cpu_bind": "none"}, {"cpus_per_task": 8}, {"cpu_bind": "cores"}]
)
def test_probe_partial_resume_keeps_saved_cpu_policy_or_rejects_changes(tmp_path, monkeypatch, provided):
    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, _ = _inputs(tmp_path)
    launch["deployment"].update(cpus_per_task=32, cpu_bind="none")
    executions = []
    monkeypatch.setattr(runner, "_cell_runner", lambda p, c, m, d: _SyntheticExecutor(p, c, d, executions))
    output = tmp_path / "probe"
    assert (
        runtime_probe.probe_runtime({"tp4": launch}, instrumentation=manifest, output_dir=output, execute=True)[
            "status"
        ]
        == "completed"
    )
    index_path = output / "observations.json"
    before = index_path.read_bytes()
    launch["deployment"].pop("cpus_per_task")
    launch["deployment"].pop("cpu_bind")
    launch["deployment"].update(provided)
    result = runtime_probe.probe_runtime(
        {"tp4": launch}, instrumentation=manifest, output_dir=output, execute=True, resume=True
    )
    same = provided in ({"cpus_per_task": 32}, {"cpu_bind": "none"})
    assert result["status"] == ("completed" if same else "failed"), result
    if same:
        assert result["configurations"]["tp4"]["resumed"] is True
        assert result["configurations"]["tp4"]["launch"]["deployment"]["cpus_per_task"] == 32
        assert result["configurations"]["tp4"]["launch"]["deployment"]["cpu_bind"] == "none"
    else:
        assert "launch settings changed" in str(result)
    assert index_path.read_bytes() == before
    assert len(executions) == 2


@pytest.mark.parametrize("tamper", ["hash", "launch", "missing"])
def test_probe_cpu_policy_inheritance_requires_verified_saved_context(tmp_path, monkeypatch, tamper):
    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, _ = _inputs(tmp_path)
    launch["deployment"].update(cpus_per_task=32, cpu_bind="none")
    monkeypatch.setattr(runner, "_cell_runner", lambda p, c, m, d: _SyntheticExecutor(p, c, d, []))
    output = tmp_path / "probe"
    runtime_probe.probe_runtime({"tp4": launch}, instrumentation=manifest, output_dir=output, execute=True)
    index_path = output / "observations.json"
    index = json.loads(index_path.read_text())
    entry = index["configurations"]["tp4"]
    if tamper == "missing":
        entry["attempts"][0]["phases"] = {}
    else:
        reference = entry["attempts"][0]["phases"]["prefill"]["launch_manifest"]
        path = output / reference["path"]
        context = json.loads(path.read_text())
        context["launch"]["deployment"]["cpu_bind"] = "cores"
        runner._atomic_json(path, context)
        if tamper == "launch":
            reference.update(runner._file_metadata(path))
    runner._atomic_json(index_path, index)
    before = index_path.read_bytes()
    launch["deployment"].pop("cpu_bind")
    monkeypatch.setattr(runner, "_cell_runner", lambda *args: pytest.fail("invalid CPU evidence reached an executor"))
    result = runtime_probe.probe_runtime(
        {"tp4": launch}, instrumentation=manifest, output_dir=output, execute=True, resume=True
    )
    assert result["status"] == "failed", result
    assert index_path.read_bytes() == before


def test_resume_without_an_archived_probe_uses_fresh_cpu_defaults(tmp_path, monkeypatch):
    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, _ = _inputs(tmp_path)
    launch["deployment"].pop("cpus_per_task")
    launch["deployment"].pop("cpu_bind")
    monkeypatch.setattr(runner, "_cell_runner", lambda p, c, m, d: _SyntheticExecutor(p, c, d, []))
    result = runtime_probe.probe_runtime(
        {"tp4": launch}, instrumentation=manifest, output_dir=tmp_path / "probe", execute=True, resume=True
    )
    assert result["status"] == "completed", result
    assert result["configurations"]["tp4"]["launch"]["deployment"]["cpus_per_task"] == 16
    assert result["configurations"]["tp4"]["launch"]["deployment"]["cpu_bind"] == "cores"


@pytest.mark.parametrize("completed", [True, False])
def test_legacy_probe_reuse_is_read_only_and_new_execution_does_not_replace_attempt(tmp_path, monkeypatch, completed):
    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, _ = _inputs(tmp_path)
    executions = []
    monkeypatch.setattr(runner, "_cell_runner", lambda p, c, m, d: _SyntheticExecutor(p, c, d, executions))
    output = tmp_path / "probe"
    runtime_probe.probe_runtime({"tp4": launch}, instrumentation=manifest, output_dir=output, execute=True)
    # Build an explicitly historical fixture, retaining the old launch representation
    # in its SHA-bound context. No CPU observations or policy are invented for it.
    index_path = output / "observations.json"
    index = json.loads(index_path.read_text())
    entry = index["configurations"]["tp4"]
    for field in ("cpus_per_task", "cpu_bind"):
        launch["deployment"].pop(field)
        entry["launch"]["deployment"].pop(field)
    for phase in entry["attempts"][0]["phases"].values():
        reference = phase["launch_manifest"]
        path = output / reference["path"]
        context = json.loads(path.read_text())
        context["launch"] = entry["launch"]
        runner._atomic_json(path, context)
        reference.update(runner._file_metadata(path))
        if not completed:
            phase["status"] = "failed"
    if not completed:
        entry["attempts"][0]["status"] = "failed"
    runner._atomic_json(index_path, index)
    before = index_path.read_bytes()
    attempts = sorted((output / "attempts").rglob("runtime-probe-context.json"))
    monkeypatch.setattr(runner, "_cell_runner", lambda *args: pytest.fail("legacy resume reached an executor"))
    result = runtime_probe.probe_runtime(
        {"tp4": launch}, instrumentation=manifest, output_dir=output, execute=True, resume=True
    )
    assert result["status"] == ("completed" if completed else "failed"), result
    if not completed:
        assert "no frozen CPU policy" in str(result)
    assert index_path.read_bytes() == before
    assert sorted((output / "attempts").rglob("runtime-probe-context.json")) == attempts


def test_dense_tensor_parallel_probe_keeps_dense_moe_axes(tmp_path):
    from collector.fpm_forward import runtime_probe

    launch, manifest, _marker = _inputs(tmp_path)
    launch["identity"]["model_kind"] = "dense"
    launch["topology"].update(moe_ep=1)
    config = Path(launch["model_config"]["path"])
    payload = json.loads(config.read_text())
    payload.pop("num_experts")
    config.write_text(json.dumps(payload))
    launch["model_config"]["sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    result = runtime_probe.probe_runtime({"dense": launch}, instrumentation=manifest, output_dir=tmp_path / "probe")
    assert result["status"] == "preview"
    assert result["configurations"]["dense"]["launch"]["topology"] == launch["topology"]


def test_probe_rejects_ignored_checkpoint_precision_and_resolves_explicit_graph_default(tmp_path):
    from collector.fpm_forward import runtime_probe

    launch, manifest, _marker = _inputs(tmp_path, graph_policy="explicit")
    launch["collection"]["max_prefill_cudagraph_size"] = None
    wrong = copy.deepcopy(launch)
    wrong["precision"]["gemm_quant_mode"] = "nvfp4"
    result = runtime_probe.probe_runtime(
        {"valid": launch, "wrong": wrong}, instrumentation=manifest, output_dir=tmp_path / "probe"
    )
    assert result["status"] == "partial"
    assert "checkpoint-native" in result["configurations"]["wrong"]["diagnostics"][0]
    assert result["configurations"]["valid"]["launch"]["collection"]["max_prefill_cudagraph_size"] == 2048


def _formal_cli_inputs(tmp_path):
    from .test_fpm_profile_collection import _profile

    launch, manifest, _marker = _inputs(tmp_path)
    profile = _profile()
    profile.update(
        model=launch["identity"]["model"],
        model_revision=launch["identity"]["model_revision"],
        architecture="ExternalMoeForCausalLM",
        context_length=4096,
        num_experts=8,
    )
    profile["deployments"] = [profile["deployments"][0]]
    deployment = profile["deployments"][0]
    deployment.update(system="gb200", backend_version="0.28.0", kv_cache_dtype="bfloat16", **launch["topology"])
    deployment.update({key: value for key, value in launch["precision"].items() if key != "kvcache_quant_mode"})
    deployment["resources"].update(max_num_tokens=1024, max_batch_size=64)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    launch_path = tmp_path / "launch.json"
    launch_path.write_text(json.dumps(launch))
    argv = [
        "--model-path",
        launch["identity"]["model"],
        "--gpu",
        "gb200",
        "--fpm-model-profile",
        str(profile_path),
        "--fpm-model-config",
        launch["model_config"]["path"],
        "--fpm-max-gpus",
        "4",
        "--fpm-gpu-counts",
        "4",
        "--fpm-parallel-presets",
        "tep",
        "--fpm-kv-cache-dtypes",
        "bfloat16",
        "--fpm-executor",
        "slurm",
        "--fpm-slurm-container-image",
        launch["deployment"]["image"],
        "--fpm-slurm-container-mount",
        "/cache:/cache",
        "--fpm-max-model-len",
        "4096",
        "--fpm-max-num-batched-tokens",
        "1024",
        "--fpm-max-num-seqs",
        "64",
        "--fpm-gpu-memory-utilization",
        "0.85",
        "--fpm-prefill-cudagraph-policy",
        "runtime",
    ]
    extra = [
        "--fpm-runtime-instrumentation",
        str(manifest),
        "--fpm-runtime-launch",
        str(launch_path),
        "--fpm-runtime-configuration",
        "tp4",
    ]
    return argv, extra, launch


def test_formal_collector_cli_requires_all_instrumentation_inputs(tmp_path, capsys, monkeypatch):
    from collector.fpm_forward import cli

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "")
    argv, extra, _launch = _formal_cli_inputs(tmp_path)
    with pytest.raises(SystemExit) as error:
        cli.main([*argv, *extra[:2], "--plan-only"])
    assert error.value.code == 2
    assert "required together" in capsys.readouterr().err
    assert cli.main([*argv, *extra, "--plan-only"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["runtime_memory_policy"]["observation"] == "enabled"
    assert plan["runtime_observation"]["configuration"] == "tp4"
    assert plan["runtime_observation"]["launch"]["identity"]["framework_version"] == "0.28.0"


def test_repeatability_roundtrips_and_hashes_instrumented_formal_plan(tmp_path):
    from collector.fpm_forward import cli, entry, repeatability
    from collector.fpm_forward.runtime_memory import validate_saved_plan
    from collector.model_cases import build_collection_case_plan

    argv, extra, _launch = _formal_cli_inputs(tmp_path)
    args = cli._parser().parse_args([*argv, *extra])
    case_plan = build_collection_case_plan(backend="vllm", model_path=args.model_path, gpu_type=args.gpu)
    plan, _overrides = entry.resolve_inputs(args, case_plan)
    saved = plan.to_dict()
    validate_saved_plan(saved)
    (tmp_path / "collection-plan.json").write_text(json.dumps(saved))
    loaded = repeatability.load_repeatability_source(tmp_path)
    assert loaded.to_dict() == saved
    assert loaded.runtime_instrumentation.sha256 == plan.runtime_instrumentation.sha256
    selected = {
        "cell_id": plan.cells[0].cell_id,
        "benchmark_points": {
            "schema_version": 3,
            "prefill": [{"batch_size": 1, "total_prefill_tokens": 16, "total_kv_read_tokens": 0}],
            "decode": [],
        },
    }
    subset = repeatability._subset_plan(loaded, selected)
    validate_saved_plan(subset.to_dict())
    assert subset.to_dict()["runtime_observation"] == saved["runtime_observation"]
    assert subset.sha256 != plan.sha256


def test_formal_collection_instruments_resolved_memory_and_archives_phase_attempts(tmp_path, monkeypatch):
    from collector.fpm_forward import cli, entry, runner
    from collector.model_cases import build_collection_case_plan

    argv, extra, _launch = _formal_cli_inputs(tmp_path)
    args = cli._parser().parse_args([*argv, *extra])
    case_plan = build_collection_case_plan(backend="vllm", model_path=args.model_path, gpu_type=args.gpu)
    plan, overrides = entry.resolve_inputs(args, case_plan)
    assert plan.deployment_profile(plan.cells[0]).resources.memory_source != "pending"
    executions = []
    monkeypatch.setattr(
        runner,
        "_cell_runner",
        lambda plan, cell, _manifest, directory: _SyntheticExecutor(plan, cell, directory, executions),
    )
    monkeypatch.setattr(runner, "_runtime_collection_summary", lambda *_args, **_kwargs: {})
    errors = runner.run_collection(
        plan,
        generator_overrides=overrides,
        checkpoint_dir=str(tmp_path / "checkpoint"),
        artifact_root=str(tmp_path / "collection"),
        resume=False,
        retry_failed=False,
        publish_database=False,
    )
    assert errors == []
    root = tmp_path / "collection" / plan.sha256[:16]
    index = json.loads((root / "runtime-observations.json").read_text())
    attempt = index["configurations"]["tp4"]["attempts"][0]
    assert attempt["status"] == "captured"
    assert set(attempt["phases"]) == {"prefill", "decode"}
    contexts = [
        json.loads((root / phase["launch_manifest"]["path"]).read_text()) for phase in attempt["phases"].values()
    ]
    assert {context["attempt_id"] for context in contexts} == {attempt["attempt_id"]}
    assert len({context["collector_attempt_id"] for context in contexts}) == 2
    assert all(len(items) == 1 for items in attempt["phase_attempts"].values())
    for path in (root / "cells").glob("*/generator-request.json"):
        args = json.loads(path.read_text())["params"]["agg"]["extra_cli_args"]
        assert args[args.index("--revision") + 1] == plan.runtime_launch["identity"]["model_revision"]
        assert args.count("--revision") == 1


def _sidecar_inputs(tmp_path):
    launch, manifest, marker = _inputs(tmp_path)
    sidecar = tmp_path / "hf_quant_config.json"
    sidecar.write_text(json.dumps({"quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": None}}))
    launch["model_config"]["source_files"] = {sidecar.name: hashlib.sha256(sidecar.read_bytes()).hexdigest()}
    launch["precision"].update(gemm_quant_mode="nvfp4", moe_quant_mode="nvfp4")
    return launch, manifest, marker, sidecar


def test_probe_binds_raw_primary_and_adjacent_quantization_metadata(tmp_path):
    from collector.fpm_forward import runtime_probe

    launch, manifest, marker, sidecar = _sidecar_inputs(tmp_path)
    primary = Path(launch["model_config"]["path"])
    original = {"model-config.json": primary.read_bytes(), sidecar.name: sidecar.read_bytes()}
    output = tmp_path / "probe"
    result = runtime_probe.probe_runtime({"tp4": launch}, instrumentation=manifest, output_dir=output)
    assert result["status"] == "preview", result
    assert primary.read_bytes() == original["model-config.json"]
    assert not marker.exists()
    for phase in result["configurations"]["tp4"]["phases"].values():
        context = json.loads((output / phase["launch_manifest"]["path"]).read_text())
        assert context["launch"]["model_config"] == launch["model_config"]
        artifacts = {Path(a["path"]).name: a for a in phase["artifacts"]}
        for name, content in original.items():
            assert (output / artifacts[name]["path"]).read_bytes() == content
            assert artifacts[name]["sha256"] == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize("mutation", ["missing", "changed", "malformed", "unrecorded", "escape", "symlink", "bad-hash"])
def test_probe_rejects_unbound_quantization_metadata(tmp_path, mutation):
    from collector.fpm_forward import runtime_probe

    launch, manifest, _marker, sidecar = _sidecar_inputs(tmp_path)
    # Keep native precision unchanged so rejection cannot be a precision mismatch.
    launch["precision"].update(gemm_quant_mode="bfloat16", moe_quant_mode="bfloat16")
    sidecar.write_text("{}")
    launch["model_config"]["source_files"][sidecar.name] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    if mutation == "missing":
        sidecar.unlink()
    elif mutation == "changed":
        sidecar.write_text("{}\n")
    elif mutation == "malformed":
        sidecar.write_text("[]")
        launch["model_config"]["source_files"][sidecar.name] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    elif mutation == "unrecorded":
        launch["model_config"].pop("source_files")
    elif mutation == "escape":
        launch["model_config"]["source_files"] = {"../hf_quant_config.json": "a" * 64}
    elif mutation == "symlink":
        sidecar.rename(tmp_path / "other.json")
        (tmp_path / "other.json").write_text("{}\n")
        sidecar.symlink_to(tmp_path / "other.json")
    elif mutation == "bad-hash":
        launch["model_config"]["source_files"][sidecar.name] = "not-a-digest"
    result = runtime_probe.probe_runtime({"tp4": launch}, instrumentation=manifest, output_dir=tmp_path / "probe")
    assert result["status"] == "failed", result


def test_probe_preserves_logical_adjacency_for_hf_cache_symlinks(tmp_path):
    from collector.fpm_forward import runtime_probe

    launch, manifest, _marker, sidecar = _sidecar_inputs(tmp_path)
    primary = Path(launch["model_config"]["path"])
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    for path, blob_name in ((primary, "primary-blob"), (sidecar, "quantization-blob")):
        target = blobs / blob_name
        path.rename(target)
        path.symlink_to(target)
    result = runtime_probe.probe_runtime({"tp4": launch}, instrumentation=manifest, output_dir=tmp_path / "probe")
    assert result["status"] == "preview", result


def test_probe_rechecks_sidecar_between_plan_and_stage(tmp_path, monkeypatch):
    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, _marker, sidecar = _sidecar_inputs(tmp_path)
    original_render = runner._render_cell

    def render_then_mutate(*args, **kwargs):
        result = original_render(*args, **kwargs)
        sidecar.write_bytes(sidecar.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(runner, "_render_cell", render_then_mutate)
    result = runtime_probe.probe_runtime({"tp4": launch}, instrumentation=manifest, output_dir=tmp_path / "probe")
    assert result["status"] == "failed"
    assert any("SHA-256" in item for item in result["configurations"]["tp4"]["diagnostics"])


@pytest.mark.parametrize("version", ["0.27.0", "0.28.0"])
@pytest.mark.parametrize("pending", [False, True])
def test_instrumented_saved_plan_hash_round_trip(tmp_path, version, pending):
    from collector.fpm_forward import cli, entry, runtime_memory
    from collector.model_cases import build_collection_case_plan

    argv, extra, _launch = _formal_cli_inputs(tmp_path)
    for filename in ("profile.json", "manifest.json", "launch.json"):
        path = tmp_path / filename
        payload = json.loads(path.read_text().replace("0.28.0", version))
        if pending and filename == "profile.json":
            resources = payload["deployments"][0]["resources"]
            for field in ("weights_bytes", "activations_bytes", "runtime_overhead_bytes", "comm_overhead_bytes"):
                resources.pop(field, None)
        path.write_text(json.dumps(payload))
    args = cli._parser().parse_args([*argv, *extra])
    case_plan = build_collection_case_plan(backend="vllm", model_path=args.model_path, gpu_type=args.gpu)
    plan, _overrides = entry.resolve_inputs(args, case_plan)
    assert (plan.deployment_profile(plan.cells[0]).resources.memory_source == "pending") is pending
    payload = plan.to_dict()
    runtime_memory.validate_saved_plan(payload)
    for field in ("runtime_observation", "runtime_memory_policy"):
        changed = copy.deepcopy(payload)
        changed[field]["bundle_sha256"] = "f" * 64
        with pytest.raises(ValueError, match="SHA-256"):
            runtime_memory.validate_saved_plan(changed)


@pytest.mark.parametrize(
    "mutation",
    [
        "context",
        "binding",
        "manifest",
        "redirect",
        "ownership",
        "compute_domain_ownership",
        "compute_domain_missing_ownership",
    ],
)
def test_probe_resume_rejects_changed_ownership_before_external_commands(tmp_path, monkeypatch, mutation):
    import shutil

    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, _marker = _inputs(tmp_path)
    launch["deployment"] = {"executor": "kubernetes", "image": "image@sha256:synthetic"}
    auxiliary_ownership = mutation in {"compute_domain_ownership", "compute_domain_missing_ownership"}
    if auxiliary_ownership:
        launch["identity"]["gpu"] = "gb300"
        launch["topology"].update(tp=8, dp=1, moe_tp=8, moe_ep=1)
    configuration = "tp8" if auxiliary_ownership else "tp4"
    real_runner = runner._cell_runner

    class Interrupted(_SyntheticExecutor):
        def execute(self, pods):
            super().execute(pods)
            raise KeyboardInterrupt()

    monkeypatch.setattr(runner, "_salvage_artifacts", lambda *_args: None)
    monkeypatch.setattr(runner, "_cell_runner", lambda p, c, m, d: Interrupted(p, c, d, []))
    output = tmp_path / "probe"
    with pytest.raises(KeyboardInterrupt):
        runtime_probe.probe_runtime({configuration: launch}, instrumentation=manifest, output_dir=output, execute=True)
    index_path = output / "observations.json"
    index = json.loads(index_path.read_text())
    phase = index["configurations"][configuration]["attempts"][0]["phases"]["prefill"]
    context = output / phase["launch_manifest"]["path"]
    resource = context.parent / runner.FPM_MANIFEST_FILENAME
    if mutation == "context":
        context.write_bytes(context.read_bytes() + b"\n")
    elif mutation == "binding":
        payload = json.loads(context.read_text())
        payload["attempt_id"] = "unrelated-attempt"
        context.write_text(json.dumps(payload))
        phase["launch_manifest"]["sha256"] = hashlib.sha256(context.read_bytes()).hexdigest()
    elif mutation in {"manifest", "ownership"} or auxiliary_ownership:
        documents = list(yaml.safe_load_all(resource.read_text()))
        if auxiliary_ownership:
            metadata = next(document for document in documents if document["kind"] == "ComputeDomain")["metadata"]
        else:
            metadata = runner._workload_document(documents)["metadata"]
            metadata["name"] = "unrelated-production-worker"
        if mutation == "compute_domain_missing_ownership":
            for label in (
                "aiconfigurator.nvidia.com/owned-by",
                "aiconfigurator.nvidia.com/plan",
                runner.FPM_CELL_LABEL,
            ):
                metadata["labels"].pop(label)
        else:
            metadata["labels"][runner.FPM_CELL_LABEL] = "unrelated-cell"
        resource.write_text(yaml.safe_dump_all(documents))
        if (mutation == "ownership" or auxiliary_ownership) and "resource_manifest" in phase:
            phase["resource_manifest"]["sha256"] = hashlib.sha256(resource.read_bytes()).hexdigest()
    else:
        redirected = output / "redirected"
        shutil.copytree(context.parent, redirected)
        phase["launch_manifest"]["path"] = str((redirected / context.name).relative_to(output))
    index_path.write_text(json.dumps(index))
    monkeypatch.setattr(runner, "_cell_runner", real_runner)
    monkeypatch.setattr(runner, "_kubectl_command", lambda: ["kubectl"])
    commands = []

    def no_external(args, **kwargs):
        commands.append(args)
        raise RuntimeError("unexpected external command")

    monkeypatch.setattr(runner, "_run_command", no_external)
    result = runtime_probe.probe_runtime(
        {configuration: launch}, instrumentation=manifest, output_dir=output, execute=True, resume=True
    )
    assert result["status"] == "failed"
    assert commands == []
    if auxiliary_ownership:
        assert any("collector ownership" in item for item in result["configurations"][configuration]["diagnostics"])


@pytest.mark.parametrize(
    ("executor", "changed_bundle", "configuration"),
    [
        ("kubernetes", False, "tp4"),
        ("kubernetes", True, "tp4"),
        ("slurm", False, "tp4"),
        ("slurm", True, "tp4"),
        ("kubernetes", False, "tp8"),
        ("slurm", False, "dep8"),
    ],
)
@pytest.mark.parametrize("saved_binding", ["current", "legacy", "older-loader"])
def test_probe_resume_preserves_valid_interrupted_recovery(
    tmp_path, monkeypatch, executor, changed_bundle, configuration, saved_binding
):
    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, _marker = _inputs(tmp_path)
    if executor == "kubernetes":
        launch["deployment"] = {"executor": executor, "image": launch["deployment"]["image"]}
    if configuration == "tp8":
        launch["identity"]["gpu"] = "gb300"
        launch["topology"].update(tp=8, dp=1, moe_tp=8, moe_ep=1)
    elif configuration == "dep8":
        launch["identity"]["gpu"] = "gb300"
        launch["topology"].update(tp=1, dp=8, moe_tp=1, moe_ep=8)
    executions, cleanups, salvages = [], [], []

    class InterruptedOnce(_SyntheticExecutor):
        def cleanup(self):
            cleanups.append(str(self.cell_dir))

        def execute(self, pods):
            super().execute(pods)
            if len(executions) == 1:
                raise KeyboardInterrupt()

    monkeypatch.setattr(runner, "_salvage_artifacts", lambda resource, _cell: salvages.append(str(resource.cell_dir)))
    monkeypatch.setattr(runner, "_cell_runner", lambda p, c, m, d: InterruptedOnce(p, c, d, executions))
    output = tmp_path / "probe"
    with pytest.raises(KeyboardInterrupt):
        runtime_probe.probe_runtime({configuration: launch}, instrumentation=manifest, output_dir=output, execute=True)
    index_path = output / "observations.json"
    index = json.loads(index_path.read_text())
    previous = index["configurations"][configuration]["attempts"][0]
    reference = previous["phases"]["prefill"]["launch_manifest"]
    saved_path = output / reference["path"]
    context = json.loads(saved_path.read_text())
    if saved_binding == "legacy":
        context.pop("instrumentation_binding")
    elif saved_binding == "older-loader":
        context["instrumentation_binding"]["loader_sha256"] = "f" * 64
    saved_path.write_text(json.dumps(context))
    reference.update(runner._file_metadata(saved_path))
    index_path.write_text(json.dumps(index))
    interrupted_dir = str((output / previous["phases"]["prefill"]["launch_manifest"]["path"]).parent)
    if changed_bundle:
        (tmp_path / "observer.py").write_text("# a revised observer retained alongside the interrupted bundle\n")
    result = runtime_probe.probe_runtime(
        {configuration: launch}, instrumentation=manifest, output_dir=output, execute=True, resume=True
    )
    assert result["status"] == "completed", result
    assert len(executions) == 3
    assert cleanups.count(interrupted_dir) == 3
    assert salvages.count(interrupted_dir) == 2
    attempts = json.loads((output / "observations.json").read_text())["configurations"][configuration]["attempts"]
    previous = attempts[0]
    assert any(item["kind"] == "observation" for item in previous["phases"]["prefill"]["recovery_artifacts"])
    for phase in attempts[-1]["phases"].values():
        from collector.fpm_forward.runtime_instrumentation import runtime_binding

        new_context = json.loads((output / phase["launch_manifest"]["path"]).read_text())
        assert new_context["instrumentation_binding"] == runtime_binding()
        observations = [item for item in phase["artifacts"] if item["kind"] == "observation"]
        assert len(observations) == (1 if configuration == "tp4" else 2)


@pytest.mark.parametrize("saved_status", ["running", "passed"])
@pytest.mark.parametrize("archive_failure", [None, "copy", "index", "context", "stale"])
def test_formal_resume_archives_recovered_native_evidence_before_skip(
    tmp_path, monkeypatch, saved_status, archive_failure
):
    from collector.fpm_forward import cli, entry, runner, runtime_probe
    from collector.model_cases import build_collection_case_plan

    from .test_fpm_runner import _native_payload

    argv, extra, _launch = _formal_cli_inputs(tmp_path)
    args = cli._parser().parse_args([*argv, *extra])
    case_plan = build_collection_case_plan(backend="vllm", model_path=args.model_path, gpu_type=args.gpu)
    plan, overrides = entry.resolve_inputs(args, case_plan)
    artifact_root = tmp_path / "collection"
    root = artifact_root / plan.sha256[:16]
    parent = "interrupted-parent"
    frozen = runtime_probe.prepare_collection_observations(plan, root, parent)
    prefill = next(cell for cell in plan.cells if cell.workload_kind == "prefill")
    cell_dir = root / "cells" / prefill.cell_id
    cell_dir.mkdir(parents=True)
    runner._render_cell(plan, prefill, cell_dir, overrides)
    context = runtime_probe.launch_context(plan, prefill, configuration=plan.runtime_configuration, attempt_id=parent)
    context.update(collector_attempt_id="prefill-native-attempt", cell_id=prefill.cell_id)
    runtime_probe.stage_runtime_instrumentation(frozen, cell_dir, context)
    raw = cell_dir / "raw" / "synthetic-node"
    raw.mkdir(parents=True)

    def native_files(cell, directory, attempt_id):
        (directory / "collector-provenance.json").write_text(
            json.dumps(
                {
                    "schema_name": "aic_fpm_collector_provenance",
                    "schema_version": 1,
                    "cell_id": cell.cell_id,
                    "plan_sha256": plan.sha256,
                    "attempt_id": attempt_id,
                    "runtime": {"backend": "vllm", "backend_version": "0.28.0"},
                }
            )
        )
        (directory / "benchmark.json").write_text(json.dumps(_native_payload(phase=cell.workload_kind, rank=0, dp=1)))

    native_files(prefill, raw, "prefill-native-attempt")
    observation = raw / "synthetic-observation.json"
    observation.write_text(
        json.dumps({**context, "schema_version": "aisimulate-runtime-observation/v1", "synthetic": True})
    )
    original_observation = observation.read_bytes()
    if archive_failure == "stale":
        # A prior failed snapshot can predate the final observation write.
        observation.unlink()
        runtime_probe.record_collection_observations(
            plan, root, parent, prefill, "prefill-native-attempt", cell_dir, "failed"
        )
        observation.write_bytes(original_observation)
    runner._runtime_collection_summary(
        prefill, cell_dir / "raw", expected_plan_sha256=plan.sha256, expected_attempt_id="prefill-native-attempt"
    )
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "fpm_forward.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "schema": runner.CHECKPOINT_SCHEMA,
                "plan_sha256": plan.sha256,
                "runtime_observation_attempt_id": parent,
                "cells": {prefill.cell_id: {"status": saved_status, "attempt_id": "prefill-native-attempt"}},
            }
        )
    )
    executions = []

    class NativeExecutor(_SyntheticExecutor):
        def execute(self, pods):
            super().execute(pods)
            native_files(self.cell, self.cell_dir / "raw" / "synthetic-node", self.identity["attempt_id"])

    monkeypatch.setattr(runner, "_cell_runner", lambda p, c, m, d: NativeExecutor(p, c, d, executions))
    original_copy = runtime_probe.shutil.copy2
    original_atomic = runner._atomic_json
    context_path = cell_dir / runtime_probe.CONTEXT_FILENAME
    original_context = context_path.read_bytes()
    if archive_failure == "context":
        corrupted = {**context, "collector_attempt_id": "unrelated-attempt"}
        context_path.write_text(json.dumps(corrupted))
    if archive_failure == "copy":

        def interrupted_copy(source, target, *args, **kwargs):
            if Path(source) == context_path:
                raise OSError("synthetic observation snapshot interruption")
            return original_copy(source, target, *args, **kwargs)

        monkeypatch.setattr(runtime_probe.shutil, "copy2", interrupted_copy)
    if archive_failure == "index":

        def interrupted_index(path, payload):
            if Path(path).name == "runtime-observations.json":
                raise OSError("synthetic observation index interruption")
            return original_atomic(path, payload)

        monkeypatch.setattr(runner, "_atomic_json", interrupted_index)
    kwargs = dict(
        generator_overrides=overrides,
        checkpoint_dir=str(checkpoint_dir),
        artifact_root=str(artifact_root),
        resume=True,
        retry_failed=True,
        publish_database=False,
    )
    errors = runner.run_collection(plan, **kwargs)
    if archive_failure not in {None, "stale"}:
        assert any(item["classification"] == "runtime_observation_archive_failed" for item in errors)
        assert json.loads(checkpoint_path.read_text())["cells"][prefill.cell_id]["status"] == "failed"
        assert observation.read_bytes() == original_observation
        monkeypatch.setattr(runtime_probe.shutil, "copy2", original_copy)
        monkeypatch.setattr(runner, "_atomic_json", original_atomic)
        context_path.write_bytes(original_context)
        assert runner.run_collection(plan, **kwargs) == []
    else:
        assert errors == []
    index_path = root / "runtime-observations.json"
    index = json.loads(index_path.read_text())
    attempt = index["configurations"]["tp4"]["attempts"][0]
    assert set(attempt["phases"]) == {"prefill", "decode"}
    assert attempt["status"] == "captured"
    assert attempt["phases"]["prefill"]["collector_attempt_id"] == "prefill-native-attempt"
    archived = [item for item in attempt["phases"]["prefill"]["artifacts"] if item["kind"] == "observation"]
    assert len(archived) == 1
    assert (root / archived[0]["path"]).read_bytes() == original_observation
    if archive_failure == "stale":
        assert [phase["status"] for phase in attempt["phase_attempts"]["prefill"]] == ["failed", "passed"]
        assert not any(item["kind"] == "observation" for item in attempt["phase_attempts"]["prefill"][0]["artifacts"])
    assert observation.read_bytes() == original_observation
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["runtime_observations"] == str(index_path)
    assert all(entry["status"] == "passed" for entry in checkpoint["cells"].values())
    assert executions == [("tp4", "decode")]
    before = index_path.read_bytes()
    assert runner.run_collection(plan, **kwargs) == []
    assert index_path.read_bytes() == before
    assert executions == [("tp4", "decode")]


def test_probe_uses_observed_runtime_and_ignores_dynamo_release_for_compatibility(tmp_path):
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation
    from collector.fpm_forward.runtime_probe import build_runtime_probe_plan

    launch, manifest, _ = _inputs(tmp_path)
    launch["identity"].update(framework_version="custom-vllm-build", runtime_framework_version="0.28.0")
    bundle = load_instrumentation(manifest)
    plan = build_runtime_probe_plan("custom", launch, bundle)
    assert plan.capability.aic_database_version == "custom-vllm-build"
    assert plan.runtime_backend_version == "0.28.0"
    launch["deployment"]["dynamo_version"] = "unknown-dynamo-build"
    assert build_runtime_probe_plan("custom", launch, bundle).sha256 == plan.sha256
    launch["identity"]["runtime_framework_version"] = "0.27.0"
    with pytest.raises(ValueError, match="instrumentation runtime version"):
        build_runtime_probe_plan("custom", launch, bundle)


def test_custom_probe_label_without_observed_version_stays_pending_in_preview(tmp_path):
    from collector.fpm_forward.runtime_probe import probe_runtime

    launch, _manifest, _ = _inputs(tmp_path)
    launch["identity"]["framework_version"] = "custom-vllm-build"
    result = probe_runtime({"custom": launch}, output_dir=tmp_path / "preview")
    assert result["status"] == "pending_runtime_version"
    assert not (tmp_path / "preview").exists()
    with pytest.raises(ValueError, match="target-container version detection"):
        probe_runtime({"custom": launch}, output_dir=tmp_path / "execute", execute=True)


def test_pending_probe_launch_validates_other_facts_before_runtime_resolution(tmp_path):
    from collector.fpm_forward.runtime_probe import normalize_probe_launch

    launch, _manifest, _ = _inputs(tmp_path)
    launch["identity"].pop("framework_version")
    pending = normalize_probe_launch(launch, allow_pending_version=True)
    assert "framework_version" not in pending["identity"]
    with pytest.raises(ValueError, match="identity.framework_version"):
        normalize_probe_launch(launch)
    launch["topology"]["tp"] = 0
    with pytest.raises(ValueError, match="topology.tp"):
        normalize_probe_launch(launch, allow_pending_version=True)


def _k31_native_probe_inputs(tmp_path):
    from collector.fpm_forward import runtime_probe

    from .test_fpm_forward import _kimi_k31_config

    launch, manifest, marker = _inputs(tmp_path)
    config, _ = _kimi_k31_config(tmp_path)
    launch["model_config"] = {"path": str(config), "sha256": hashlib.sha256(config.read_bytes()).hexdigest()}
    launch["worker_type"] = "decode"
    launch["identity"].update(model="example/k31", framework_version="0.1.dev24216+g266e88c05")
    launch["topology"].update(tp=8, moe_tp=8, moe_ep=1)
    launch["precision"].update(
        moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm", kvcache_quant_mode="fp8", attention_backend="trtllm_mla"
    )
    packaged = Path(runtime_probe.__file__).parent / "runtime/vllm-kimi-k31.example.json"
    document = json.loads(manifest.read_text())
    document["runtime"] = json.loads(packaged.read_text())["runtime"]
    manifest.write_text(json.dumps(document))
    return launch, manifest, marker


@pytest.mark.parametrize("worker_type", ["prefill", "decode"])
@pytest.mark.parametrize("fmha", ["fp8", "half", "bfloat16"])
def test_k31_probe_rejects_fmha_without_checkpoint_compute_mapping(tmp_path, worker_type, fmha):
    from collector.fpm_forward import runtime_probe
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation

    launch, manifest, marker = _k31_native_probe_inputs(tmp_path)
    launch["worker_type"] = worker_type
    launch["precision"]["fmha_quant_mode"] = fmha
    if fmha == "bfloat16":
        plan = runtime_probe.build_runtime_probe_plan("k31", launch, load_instrumentation(manifest))
        assert plan.cells[0].fmha_quant_mode == "bfloat16"
        assert plan.cells[0].kv_cache_dtype == "fp8"
    else:
        with pytest.raises(ValueError, match="FMHA precision matching the supported checkpoint compute dtype"):
            runtime_probe.build_runtime_probe_plan("k31", launch, load_instrumentation(manifest))
    assert not marker.exists()


def test_k31_probe_binds_native_activation_candidate_and_private_decode_state(tmp_path):
    from collector.fpm_forward import runner, runtime_probe
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation

    launch, manifest, _ = _k31_native_probe_inputs(tmp_path)
    plan = runtime_probe.build_runtime_probe_plan("k31-decode", launch, load_instrumentation(manifest))
    assert plan.capability.architecture == "KimiK31ForConditionalGeneration"
    assert len(plan.cells) == 1
    cell = plan.cells[0]
    assert cell.moe_quant_mode == "w4a8_mxfp4_mxfp8_trtllm"
    request = runner._cell_generator_overrides(plan, cell, {})
    args = request["params"]["agg"]["extra_cli_args"]
    assert args[args.index("--mamba-cache-mode") + 1] == "none"
    assert args[args.index("--mamba-ssm-cache-dtype") + 1] == "float32"
    assert args.count("--no-enable-prefix-caching") == 1
    assert "--language-model-only" in args
    assert "--trust-remote-code" in args
    assert json.loads(args[args.index("--attention-config") + 1]) == {
        "backend": "FLASHINFER_MLA_SPARSE",
        "indexer_kv_dtype": "fp8",
    }
    assert runner._configured_sampling_metadata(plan, cell, smoke=True)["decode_prefix_caching"] == "disabled"


def test_probe_deployment_environment_reaches_startup_and_engine(tmp_path):
    from aisimulate.support.schema import FPMDeployment
    from collector.fpm_forward import runner, runtime_probe
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation

    launch, manifest, _ = _inputs(tmp_path)
    environment = [
        {"name": "PYTHONPATH", "value": "/runtime/dynamo:/runtime/wheel"},
        {"name": "PYTHONNOUSERSITE", "value": "1"},
        {"name": "PYTHONUSERBASE", "value": "/runtime/userbase"},
        {"name": "HF_HOME", "value": "/runtime/hf"},
        {"name": "HF_HUB_OFFLINE", "value": "1"},
        {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
    ]
    launch["deployment"] = FPMDeployment(**launch["deployment"], extra_env=environment).model_dump(exclude_none=True)
    plan = runtime_probe.build_runtime_probe_plan("custom-runtime", launch, load_instrumentation(manifest))
    overrides = runner._cell_generator_overrides(plan, plan.cells[0], runtime_probe.probe_generator_overrides(launch))
    assert all(item in overrides["K8sConfig"]["extra_env"] for item in environment)
    runner._write_runtime_environment(tmp_path, overrides)
    startup = (tmp_path / runner.RUNTIME_ENV_FILENAME).read_text()
    for item in environment:
        assert f"export {item['name']}={item['value']}" in startup


@pytest.mark.parametrize("worker_type", ["aggregated", None])
def test_k31_private_state_probe_requires_an_independent_role(tmp_path, worker_type):
    from collector.fpm_forward import runtime_probe
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation

    launch, manifest, _ = _k31_native_probe_inputs(tmp_path)
    launch["worker_type"] = worker_type
    with pytest.raises(ValueError, match="independent prefill or decode"):
        runtime_probe.build_runtime_probe_plan("k31", launch, load_instrumentation(manifest))


def test_k31_prefill_probe_preserves_native_sampling_with_prefix_caching_disabled(tmp_path):
    from collector.fpm_forward import runner, runtime_probe
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation

    launch, manifest, _ = _k31_native_probe_inputs(tmp_path)
    launch["worker_type"] = "prefill"
    plan = runtime_probe.build_runtime_probe_plan("k31-prefill", launch, load_instrumentation(manifest))
    assert [cell.workload_kind for cell in plan.cells] == ["prefill"]
    overrides = runner._cell_generator_overrides(plan, plan.cells[0], {})
    args = overrides["params"]["agg"]["extra_cli_args"]
    assert args[args.index("--benchmark-mode") + 1] == "prefill"
    assert args[args.index("--mamba-cache-mode") + 1] == "none"
    assert args[args.index("--mamba-ssm-cache-dtype") + 1] == "float32"
    assert args.count("--no-enable-prefix-caching") == 1
    assert args.count("--no-async-scheduling") == 1
    assert "--enable-prefix-caching" not in args
    assert "--benchmark-points-file" not in args
    for flag in (
        "--prefill-max-new-token-samples",
        "--prefill-max-kv-read-token-samples",
        "--prefix-max-batch-size-samples",
    ):
        assert args[args.index(flag) + 1] == "2"
    environment = {entry["name"]: entry["value"] for entry in overrides["K8sConfig"]["extra_env"]}
    assert environment["DYN_BENCH_PREFILL_REAL_SEED"] == "on"


@pytest.mark.parametrize("worker_type", ["prefill", "decode"])
@pytest.mark.parametrize("historical_kv", [0, 128])
def test_k31_explicit_points_require_zero_historical_kv_for_every_prefill(tmp_path, worker_type, historical_kv):
    from dataclasses import replace

    from collector.fpm_forward import runner, runtime_probe
    from collector.fpm_forward.config import _freeze_benchmark_points
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation

    launch, manifest, _ = _k31_native_probe_inputs(tmp_path)
    launch["worker_type"] = worker_type
    plan = runtime_probe.build_runtime_probe_plan("k31", launch, load_instrumentation(manifest))
    points = {
        "schema_version": 1,
        "prefill": [
            {"batch_size": 1, "total_prefill_tokens": 128, "total_kv_read_tokens": 0},
            {"batch_size": 2, "total_prefill_tokens": 512, "total_kv_read_tokens": historical_kv},
        ],
        "decode": [{"batch_size": 1, "total_kv_read_tokens": 128}],
    }
    points_path = tmp_path / "points.json"
    points_path.write_text(json.dumps(points))
    canonical, digest = _freeze_benchmark_points(str(points_path))
    plan = replace(
        plan,
        options=replace(plan.options, benchmark_points_json=canonical, benchmark_points_sha256=digest),
    )
    if historical_kv:
        with pytest.raises(ValueError, match="total_kv_read_tokens=0 in every explicit point"):
            runner._cell_generator_overrides(plan, plan.cells[0], {})
    else:
        overrides = runner._cell_generator_overrides(plan, plan.cells[0], {})
        args = overrides["params"]["agg"]["extra_cli_args"]
        assert args.count("--benchmark-points-file") == 1
        assert args[args.index("--benchmark-points-file") + 1].endswith("/benchmark-points.json")
        assert runner._frozen_points(plan) == canonical


@pytest.mark.parametrize("worker_type", ["prefill", "decode"])
@pytest.mark.parametrize(
    "flag",
    [
        "--enable-prefix-caching",
        "--enable-prefix-caching=true",
        "--enable_prefix_caching",
        "--enable_prefix_caching=true",
    ],
)
def test_k31_native_state_policy_cannot_reenable_prefix_caching(tmp_path, worker_type, flag):
    from dataclasses import replace

    from collector.fpm_forward import runner, runtime_probe
    from collector.fpm_forward.planner import BackendPolicy
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation

    launch, manifest, _ = _k31_native_probe_inputs(tmp_path)
    launch["worker_type"] = worker_type
    plan = runtime_probe.build_runtime_probe_plan("k31", launch, load_instrumentation(manifest))
    cell = replace(
        plan.cells[0], backend_policy=BackendPolicy("test", {"params": {"agg": {"extra_cli_args": [flag]}}}, {})
    )
    with pytest.raises(ValueError, match="cannot change K3.1 observed private state, prefix caching"):
        runner._cell_generator_overrides(plan, cell, {})


@pytest.mark.parametrize(
    "flag,message",
    [
        ("--mamba_ssm_cache_dtype=bfloat16", "cannot change K3.1"),
        ("--mamba_cache_dtype=float32", "cannot change K3.1"),
        ("--mamba_cache_mode=align", "cannot change K3.1"),
        ("--speculative_config={}", "cannot change K3.1"),
        ("--no_language_model_only", "cannot change K3.1"),
        ("--worker_cls=OtherWorker", "observer worker class"),
        ("--scheduler_cls=OtherScheduler", "observer scheduler class"),
        ("--kv_cache_memory_bytes=1234", "automatic cache sizing"),
        ("--num_gpu_blocks_override=123", "automatic cache sizing"),
        ("--async_scheduling", "requires synchronous scheduling"),
        ("--gpu_memory_utilization=0.5", "cannot override --fpm-gpu-memory-utilization"),
        ("--enforce_eager", "eager execution must be supplied"),
        ("--benchmark_points_file=other.json", "benchmark points must be supplied"),
    ],
)
def test_runtime_observation_policy_guards_accept_parser_option_spellings(tmp_path, flag, message):
    from dataclasses import replace

    from collector.fpm_forward import runner, runtime_probe
    from collector.fpm_forward.planner import BackendPolicy
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation

    launch, manifest, _ = _k31_native_probe_inputs(tmp_path)
    plan = runtime_probe.build_runtime_probe_plan("k31", launch, load_instrumentation(manifest))
    cell = replace(
        plan.cells[0], backend_policy=BackendPolicy("test", {"params": {"agg": {"extra_cli_args": [flag]}}}, {})
    )
    with pytest.raises(ValueError, match=message):
        runner._cell_generator_overrides(plan, cell, {})


@pytest.mark.parametrize("worker_type", ["prefill", "decode"])
def test_k31_activation_candidate_requires_exact_source_map(tmp_path, worker_type):
    from collector.fpm_forward import runtime_probe
    from collector.fpm_forward.runtime_instrumentation import freeze_instrumentation, load_instrumentation

    launch, manifest, _ = _k31_native_probe_inputs(tmp_path)
    launch["worker_type"] = worker_type
    frozen = freeze_instrumentation(load_instrumentation(manifest), tmp_path / "changed-runtime")
    changed = frozen.manifest
    changed["runtime"]["source_files"][next(iter(changed["runtime"]["source_files"]))] = "f" * 64
    frozen.manifest_path.write_text(json.dumps(changed))
    bundle = load_instrumentation(frozen.manifest_path)
    with pytest.raises(ValueError, match="requires an audited instrumentation bundle"):
        runtime_probe.build_runtime_probe_plan("k31", launch, bundle)


def test_probe_environment_roundtrip_preserves_supplementary_unicode(tmp_path):
    from collector.fpm_forward.runtime_probe import probe_generator_overrides

    launch, _, _ = _inputs(tmp_path)
    environment = [{"name": "HF_HOME", "value": "/cache/𠮷"}]
    launch["deployment"]["extra_env"] = environment
    resolved = probe_generator_overrides(launch)
    assert resolved["K8sConfig"]["extra_env"] == environment
    assert resolved["K8sConfig"]["extra_env"][0]["value"].encode("utf-8") == b"/cache/\xf0\xa0\xae\xb7"


@pytest.mark.parametrize("location", ["wrapper", "decoder"])
@pytest.mark.parametrize("reverse", [False, True])
def test_k31_probe_rejects_multiple_architectures_before_launch(tmp_path, location, reverse):
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation
    from collector.fpm_forward.runtime_probe import build_runtime_probe_plan

    launch, manifest, marker = _k31_native_probe_inputs(tmp_path)
    path = Path(launch["model_config"]["path"])
    config = json.loads(path.read_text())
    target = config if location == "wrapper" else config["text_config"]
    target["architectures"] = ["KimiK31ForConditionalGeneration", "KimiK3ForConditionalGeneration"]
    if reverse:
        target["architectures"].reverse()
        if location == "wrapper":
            config["text_config"].pop("architectures", None)
        else:
            config["architectures"] = ["KimiK3ForConditionalGeneration"]
    path.write_text(json.dumps(config))
    launch["model_config"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="single-entry architecture lists"):
        build_runtime_probe_plan("k31", launch, load_instrumentation(manifest))
    assert not marker.exists()


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("warmup", [None, "on", "off", "OFF", "0", "false"])
def test_k31_disabled_decode_warmup_rejected_before_scheduling(tmp_path, role, warmup):
    from collector.fpm_forward import runner, runtime_probe
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation

    launch, manifest, marker = _k31_native_probe_inputs(tmp_path)
    launch["worker_type"] = role
    plan = runtime_probe.build_runtime_probe_plan("k31", launch, load_instrumentation(manifest))
    base = {"K8sConfig": {"extra_env": [{"name": "DYN_BENCH_KV_WARMUP", "value": warmup}]}} if warmup else {}
    if role == "decode" and (warmup or "").lower() in {"off", "0", "false"}:
        with pytest.raises(ValueError, match="requires DYN_BENCH_KV_WARMUP enabled"):
            runner._cell_generator_overrides(plan, plan.cells[0], base)
    else:
        rendered = runner._cell_generator_overrides(plan, plan.cells[0], base)
        actual = {item["name"]: item["value"] for item in rendered["K8sConfig"]["extra_env"]}
        assert actual["DYN_BENCH_KV_WARMUP"] == (warmup or "on")
    assert not marker.exists()


def test_k31_decode_rendering_uses_one_native_state_resolution(tmp_path):
    from collector.fpm_forward import runner, runtime_probe
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation

    launch, manifest, _ = _k31_native_probe_inputs(tmp_path)
    base = runtime_probe.build_runtime_probe_plan("k31", launch, load_instrumentation(manifest))

    # Other rendering paths legitimately read the bundle to configure the
    # observer; isolate the duplicate decode-policy resolution itself.
    from unittest.mock import patch

    original = runner._kimi_k31_native_state
    calls = []

    def resolve(plan, *args, **kwargs):
        calls.append(True)
        return original(plan, *args, **kwargs)

    with patch.object(runner, "_kimi_k31_native_state", resolve):
        rendered = runner._cell_generator_overrides(base, base.cells[0], {})
    assert len(calls) == 1
    assert "--no-enable-prefix-caching" in rendered["params"]["agg"]["extra_cli_args"]


@pytest.mark.parametrize(
    "failure", ["worker_evidence", "backend_markers", "before_staging", "private_cleanup", "worker_interrupt"]
)
def test_formal_cell_retries_rejected_evidence_and_pre_staging_failures(tmp_path, monkeypatch, failure):
    from collector.fpm_forward import cli, database, entry, runner, runtime_probe
    from collector.model_cases import build_collection_case_plan

    from .test_fpm_runner import _native_payload

    argv, extra, launch = _formal_cli_inputs(tmp_path)
    token = "synthetic-secret-746-runtime"
    environment = [{"name": "HF_TOKEN", "value": token}]
    launch["deployment"]["extra_env"] = environment
    if failure == "backend_markers":
        # Use a supported explicit policy to require a resolved-config sidecar.
        launch["precision"]["moe_backend"] = "triton"
        profile_path = Path(argv[argv.index("--fpm-model-profile") + 1])
        profile = json.loads(profile_path.read_text())
        profile["deployments"][0]["moe_backend"] = "triton"
        profile_path.write_text(json.dumps(profile))
        argv.extend(["--fpm-moe-backend", "triton"])
    Path(extra[extra.index("--fpm-runtime-launch") + 1]).write_text(json.dumps(launch))
    argv.extend(["--generator-set", "K8sConfig.extra_env=" + json.dumps(environment)])
    args = cli._parser().parse_args([*argv, *extra])
    cases = build_collection_case_plan(backend="vllm", model_path=args.model_path, gpu_type=args.gpu)
    plan, overrides = entry.resolve_inputs(args, cases)
    executions = []
    rejected = set()
    delivery_directories = []

    class NativeExecutor(_SyntheticExecutor):
        def stage(self, pods, files):
            super().stage(pods, files)
            delivery = self.files["run.sh"].parent
            assert delivery != self.cell_dir
            assert delivery.stat().st_mode & 0o777 == 0o700
            assert runner.shlex.quote(token) in self.files["run.sh"].read_text()
            from collector.fpm_forward.slurm import SlurmCellRunner

            SlurmCellRunner.stage(self, pods, files)
            assert self.runtime_dir.parent == delivery
            assert not delivery.is_relative_to(self.cell_dir.parent.parent)
            assert runner.shlex.quote(token) in (self.runtime_dir / "run.sh").read_text()
            delivery_directories.append(delivery)

        def execute(self, pods):
            super().execute(pods)
            raw = self.cell_dir / "raw" / pods[0]
            provenance = {
                "schema_name": "aic_fpm_collector_provenance",
                "schema_version": 1,
                "cell_id": self.cell.cell_id,
                "plan_sha256": plan.sha256,
                "attempt_id": self.identity["attempt_id"],
                "runtime": {"backend": "vllm", "backend_version": "0.28.0"},
            }
            (raw / "collector-provenance.json").write_text(json.dumps(provenance))
            payload = _native_payload(phase=self.cell.workload_kind, rank=0, dp=1, prefill_history=0)
            (raw / "benchmark.json").write_text(json.dumps(payload))
            (raw / "raw-config-private.json").write_text(json.dumps({"environment": {"HF_TOKEN": token}}))
            if failure == "backend_markers" and len(executions) != 1:
                (raw / "resolved-config-native.json").write_text(
                    json.dumps({"config": {"engine_args": {"kernel_config": {"moe_backend": "triton"}}}})
                )
            if failure == "worker_evidence" and len(executions) == 1:
                rejected.add(self.identity["attempt_id"])

    monkeypatch.setattr(runner, "_cell_runner", lambda p, c, m, d: NativeExecutor(p, c, d, executions))
    from collector.fpm_forward import private_inputs

    cleanup = private_inputs.SlurmDelivery.cleanup
    cleanup_failed = False

    def cleanup_delivery(delivery):
        nonlocal cleanup_failed
        if failure == "private_cleanup" and len(executions) == 1 and not cleanup_failed:
            cleanup_failed = True
            raise OSError("synthetic private staging deletion failure")
        cleanup(delivery)

    monkeypatch.setattr(private_inputs.SlurmDelivery, "cleanup", cleanup_delivery)
    # Isolate cell lifecycle from model-specific native inventory. The actual
    # archived-worker validator has a separate real-envelope regression.
    monkeypatch.setattr(database, "_is_k31", lambda plan: True)
    # The synthetic model above exercises both phases in one campaign, unlike
    # K3.1's independent workers. Its entry requirements are covered separately
    # by the profileless K3.1 execution rejection and native rendering tests.
    monkeypatch.setattr(runner, "_require_k31_execution_inputs", lambda plan: None)

    def qualify(plan, cell, raw, collection, *, allow_staged=False):
        index = json.loads((raw.parent.parent.parent / "runtime-observations.json").read_text())
        active = index["configurations"][plan.runtime_configuration]["attempts"][-1]["phases"][cell.workload_kind]
        assert active["collector_attempt_id"] == collection.collector_attempt_id
        assert allow_staged and active["status"] == "validating"
        if failure == "worker_interrupt":
            raise KeyboardInterrupt
        if collection.collector_attempt_id in rejected:
            raise ValueError("synthetic unresolved worker evidence")

    monkeypatch.setattr(database, "require_k31_worker_evidence", qualify)
    artifact_root = tmp_path / "collection"
    checkpoint_dir = tmp_path / "checkpoint"
    kwargs = dict(
        generator_overrides=overrides,
        checkpoint_dir=str(checkpoint_dir),
        artifact_root=str(artifact_root),
        resume=False,
        retry_failed=False,
        publish_database=False,
    )
    assert token not in json.dumps(plan.to_dict())

    def assert_private_artifacts():
        remaining = [directory for directory in delivery_directories if directory.exists()]
        assert delivery_directories
        assert len(remaining) == (1 if failure == "private_cleanup" and len(executions) == 2 else 0)
        for retained in artifact_root.rglob("*"):
            if retained.is_file():
                assert token.encode() not in retained.read_bytes(), retained

    if failure == "worker_interrupt":
        with pytest.raises(KeyboardInterrupt):
            runner.run_collection(plan, **kwargs)
        saved = json.loads(next(artifact_root.rglob("runtime-observations.json")).read_text())
        attempt = saved["configurations"][plan.runtime_configuration]["attempts"][-1]
        assert attempt["status"] == "incomplete"
        assert attempt["phases"]["prefill"]["status"] == "validating"
        return

    errors = runner.run_collection(plan, **kwargs)
    assert_private_artifacts()
    checkpoint_path = checkpoint_dir / "fpm_forward.json"
    first = json.loads(checkpoint_path.read_text())
    prefill = next(c for c in plan.cells if c.workload_kind == "prefill")
    decode = next(c for c in plan.cells if c.workload_kind == "decode")
    original_id = first["cells"][prefill.cell_id]["attempt_id"]
    assert first["cells"][decode.cell_id]["status"] == "passed"
    root = artifact_root / plan.sha256[:16]
    cell_dir = root / "cells" / prefill.cell_id
    if failure in {"worker_evidence", "backend_markers"}:
        assert errors and first["cells"][prefill.cell_id]["status"] == "failed"
        message = (
            "synthetic unresolved worker evidence"
            if failure == "worker_evidence"
            else "requires resolved-config evidence"
        )
        assert message in first["cells"][prefill.cell_id]["error"]
        observation = json.loads((root / "runtime-observations.json").read_text())
        attempt = observation["configurations"][plan.runtime_configuration]["attempts"][-1]
        assert attempt["status"] == "incomplete"
        assert attempt["phases"]["prefill"]["status"] == "failed"
        assert all(item["status"] == "failed" for item in attempt["phase_attempts"]["prefill"])
    elif failure == "private_cleanup":
        record = first["cells"][prefill.cell_id]
        assert errors and record["status"] == "cleanup_failed"
        assert record["private_cleanup_error"] == "synthetic private staging deletion failure"
        assert record["completed_at"] and record["duration_seconds"] >= 0
        observation = json.loads((root / "runtime-observations.json").read_text())
        phase = observation["configurations"][plan.runtime_configuration]["attempts"][-1]["phases"]["prefill"]
        assert phase["status"] == "cleanup_failed" and phase["artifacts"]
        assert (root / "run-manifest.json").exists()
    else:
        assert errors == []
        first["cells"][prefill.cell_id]["status"] = "failed"
        checkpoint_path.write_text(json.dumps(first))
        # Retain an already archived launch, but incomplete native timings
        # prevent zero-work recovery of the deliberately failed old attempt.
        (cell_dir / "raw/synthetic-node/benchmark.json").unlink()
        render = runner._render_cell

        def fail_render(*args, **kwargs):
            raise OSError("synthetic render failure before staging")

        monkeypatch.setattr(runner, "_render_cell", fail_render)
        assert runner.run_collection(plan, **{**kwargs, "resume": True, "retry_failed": True})
        assert_private_artifacts()
        failed = json.loads(checkpoint_path.read_text())["cells"][prefill.cell_id]
        assert failed["status"] == "failed" and "observation_error" not in failed
        assert "synthetic render failure before staging" in failed["error"]
        assert not (cell_dir / runtime_probe.CONTEXT_FILENAME).exists()
        monkeypatch.setattr(runner, "_render_cell", render)
    assert runner.run_collection(plan, **{**kwargs, "resume": True, "retry_failed": True}) == []
    assert_private_artifacts()
    final = json.loads(checkpoint_path.read_text())
    assert all(item["status"] == "passed" for item in final["cells"].values())
    assert final["cells"][prefill.cell_id]["attempt_id"] != original_id
    assert final["cells"][decode.cell_id]["attempt_id"] == first["cells"][decode.cell_id]["attempt_id"]
    history = json.loads((root / "runtime-observations.json").read_text())["configurations"][
        plan.runtime_configuration
    ]["attempts"][-1]["phase_attempts"]["prefill"]
    assert any(item["collector_attempt_id"] == original_id for item in history)
    assert executions == [("tp4", "prefill"), ("tp4", "decode"), ("tp4", "prefill")]
    assert "/__aisimulate_environment_reference__/HF_TOKEN/" in (cell_dir / "run.sh").read_text()
    if failure == "backend_markers":
        from collector.fpm_forward.config import archive_generator_overrides

        monkeypatch.delenv("HF_TOKEN", raising=False)
        for config in (root / "cells").rglob("resolved-config-native.json"):
            payload = json.loads(config.read_text())
            payload["unrelated_private_value"] = token
            config.write_text(json.dumps(payload))
        recovery = {**kwargs, "generator_overrides": archive_generator_overrides(overrides), "resume": True}
        assert runner.run_collection(plan, **recovery) == []
        index_path = root / "runtime-observations.json"
        before = index_path.read_bytes()
        assert runner.run_collection(plan, **recovery) == []
        assert index_path.read_bytes() == before
        assert len(executions) == 3
        for cell in plan.cells:
            database._validate_backend_markers(cell, root / "cells" / cell.cell_id)
        assert_private_artifacts()


@pytest.mark.parametrize("transport", ["slurm", "kubernetes"])
@pytest.mark.parametrize("fail_runtime", [False, True])
def test_probe_keeps_credentials_out_of_launch_evidence_and_delivers_to_runtime(
    tmp_path, monkeypatch, transport, fail_runtime
):
    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, _ = _inputs(tmp_path)
    if transport == "kubernetes":
        launch["deployment"] = {"executor": "kubernetes", "image": launch["deployment"]["image"]}
    token = 'synthetic-"private"-$746'
    environment = [
        {"name": "HF_TOKEN", "value": token},
        {"name": "VLLM_SSM_CONV_STATE_LAYOUT", "value": "DS"},
        {"name": "FPM_READINESS_TIMEOUT_SECONDS", "value": "1200"},
    ]
    launch["deployment"]["extra_env"] = environment
    delivered = []
    staging = []
    node_local = tmp_path / "node-local-tmp"
    node_local.mkdir()
    from collector.fpm_forward import private_inputs

    monkeypatch.setattr(private_inputs.tempfile, "tempdir", str(node_local))

    class Executor(_SyntheticExecutor):
        def stage(self, pods, files):
            if transport == "slurm":
                from collector.fpm_forward.slurm import SlurmCellRunner

                SlurmCellRunner.stage(self, pods, files)
                staging.append(self.runtime_dir)
                # Pyxis bind mounts must remain on the campaign's shared
                # filesystem even when the login node's default tmp is local.
                assert not self.runtime_dir.is_relative_to(output)
                assert self.runtime_dir.parent.stat().st_mode & 0o777 == 0o700
                assert runner.shlex.quote(token) in (self.runtime_dir / "run.sh").read_text()
            self.staged_scripts = {path.name: path.read_text() for path in files if path.suffix == ".sh"}
            super().stage(pods, files)

        def execute(self, pods):
            saved = json.loads((self.cell_dir / "generator-request.json").read_text())
            auth = next(item for item in saved["K8sConfig"]["extra_env"] if item["name"] == "HF_TOKEN")
            assert auth == {
                "name": "HF_TOKEN",
                "value_from_env": "HF_TOKEN",
                "value_sha256": hashlib.sha256(token.encode()).hexdigest(),
            }
            script = self.staged_scripts["run.sh"]
            assert runner.shlex.quote(token) in script
            delivered.append(self.cell.workload_kind)
            super().execute(pods)
            raw = self.cell_dir / "raw" / pods[0]
            (raw / "resolved-config-synthetic.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "engine_args": {"kernel_config": {"moe_backend": "triton"}},
                            "environment": {"HF_TOKEN": token},
                        }
                    }
                )
            )
            logs = self.cell_dir / "logs"
            logs.mkdir(exist_ok=True)
            (logs / "synthetic.stderr.log").write_text("worker context " + token)
            if fail_runtime:
                raise RuntimeError("synthetic worker failure after staging " + token)

    monkeypatch.setattr(runner, "_cell_runner", lambda p, c, m, d: Executor(p, c, d, []))
    output = tmp_path / "probe"

    def assert_archived(value):
        if isinstance(value, dict):
            assert value.get("value") != token
            if value.get("name") == "HF_TOKEN":
                assert value == {
                    "name": "HF_TOKEN",
                    "value_from_env": "HF_TOKEN",
                    "value_sha256": hashlib.sha256(token.encode()).hexdigest(),
                }
            for item in value.values():
                assert_archived(item)
        elif isinstance(value, list):
            for item in value:
                assert_archived(item)
        elif isinstance(value, str):
            assert token not in value

    monkeypatch.delenv("HF_TOKEN", raising=False)
    offline = runtime_probe.probe_runtime(
        {"tp4": runtime_probe.normalize_probe_launch(launch)},
        instrumentation=manifest,
        output_dir=output,
        execute=False,
    )
    assert offline["status"] == "preview"
    assert_archived(offline)

    def assert_files_archived():
        for path in output.rglob("*"):
            if path.is_file():
                contents = path.read_bytes()
                assert token.encode() not in contents, path
                assert json.dumps(token)[1:-1].encode() not in contents, path
                assert runner.shlex.quote(token).encode() not in contents, path
                if path.suffix == ".json":
                    assert_archived(json.loads(contents))

    assert_files_archived()
    for execute in (False, True):
        report = runtime_probe.probe_runtime(
            {"tp4": launch}, instrumentation=manifest, output_dir=output, execute=execute
        )
        expected = "failed" if fail_runtime else "completed"
        assert report["status"] == (expected if execute else "preview"), report
        assert_archived(report)
        assert_files_archived()
    for phase in ("prefill", "decode"):
        sidecars = list(output.glob(f"attempts/*/*/{phase}/evidence-*/raw/**/resolved-config-synthetic.json"))
        assert len(sidecars) == 1, (phase, sidecars)
        assert json.loads(sidecars[0].read_text())["config"]["engine_args"]["kernel_config"]["moe_backend"] == "triton"
        assert "<redacted>" in sidecars[0].read_text()
    assert delivered == ["prefill", "decode"]
    assert len(staging) == (2 if transport == "slurm" else 0)
    assert all(not directory.exists() for directory in staging)
    assert not list(output.glob("aisimulate-probe-delivery-*"))
    assert not list(node_local.iterdir())

    saved = json.loads((output / "probe-plan.json").read_text())
    frozen = saved["configurations"]["tp4"]
    assert (
        next(item for item in frozen["deployment"]["extra_env"] if item["name"] == "VLLM_SSM_CONV_STATE_LAYOUT")[
            "value"
        ]
        == "DS"
    )
    assert runtime_probe.normalize_probe_launch(frozen) == frozen


@pytest.mark.parametrize("layout", ["DS", "SD", None])
def test_probe_layout_is_literal_for_offline_validation(tmp_path, monkeypatch, layout):
    from collector.fpm_forward import runtime_probe

    launch, _, _ = _inputs(tmp_path)
    if layout is not None:
        launch["deployment"]["extra_env"] = [{"name": "VLLM_SSM_CONV_STATE_LAYOUT", "value": layout}]
    normalized = runtime_probe.normalize_probe_launch(launch)
    assert runtime_probe.normalize_probe_launch(normalized) == normalized
    if layout is None:
        return
    assert normalized["deployment"]["extra_env"] == launch["deployment"]["extra_env"]
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", layout)
    launch["deployment"]["extra_env"] = [
        {
            "name": "VLLM_SSM_CONV_STATE_LAYOUT",
            "value_from_env": "VLLM_SSM_CONV_STATE_LAYOUT",
            "value_sha256": hashlib.sha256(layout.encode()).hexdigest(),
        }
    ]
    with pytest.raises(ValueError, match="must be a literal SD or DS"):
        runtime_probe.normalize_probe_launch(launch)


def test_k31_probe_rejects_unaudited_bundle_before_generic_precision_checks(tmp_path):
    from collector.fpm_forward import runtime_probe
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation

    launch, _, marker = _k31_native_probe_inputs(tmp_path)
    (tmp_path / "unaudited").mkdir()
    _, unaudited_manifest, _ = _inputs(tmp_path / "unaudited")
    with pytest.raises(ValueError, match="requires an audited instrumentation bundle"):
        runtime_probe.build_runtime_probe_plan("k31", launch, load_instrumentation(unaudited_manifest))
    assert not marker.exists()


@pytest.mark.parametrize("dtype,accepted", [("float16", False), ("bfloat16", True)])
def test_k31_flashmla_checkpoint_dtype_is_checked_before_launch(tmp_path, dtype, accepted):
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation
    from collector.fpm_forward.runtime_probe import build_runtime_probe_plan

    launch, manifest, marker = _k31_native_probe_inputs(tmp_path)
    model = Path(launch["model_config"]["path"])
    config = json.loads(model.read_text())
    config["text_config"]["torch_dtype"] = dtype
    config["text_config"]["dtype"] = dtype
    model.write_text(json.dumps(config))
    launch["model_config"]["sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
    launch["precision"].update(
        attention_backend="flashmla",
        fmha_quant_mode="half" if dtype == "float16" else "bfloat16",
        gemm_quant_mode="half" if dtype == "float16" else "bfloat16",
    )
    if accepted:
        assert build_runtime_probe_plan("k31", launch, load_instrumentation(manifest)).cells
    else:
        with pytest.raises(ValueError, match="FLASHMLA_SPARSE requires a bfloat16 checkpoint"):
            build_runtime_probe_plan("k31", launch, load_instrumentation(manifest))
    assert not marker.exists()


@pytest.mark.parametrize("conflicting", [False, True])
def test_k31_probe_rejects_conflicting_singleton_outer_identity(tmp_path, conflicting):
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation
    from collector.fpm_forward.runtime_probe import build_runtime_probe_plan

    launch, manifest, _ = _k31_native_probe_inputs(tmp_path)
    model = Path(launch["model_config"]["path"])
    config = json.loads(model.read_text())
    config["architectures"] = ["OtherModel" if conflicting else "KimiK31ForConditionalGeneration"]
    config["text_config"]["architectures"] = [
        "KimiK31ForConditionalGeneration" if conflicting else "KimiK3ForConditionalGeneration"
    ]
    model.write_text(json.dumps(config))
    launch["model_config"]["sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
    if conflicting:
        with pytest.raises(ValueError, match="top-level architecture identity"):
            build_runtime_probe_plan("k31", launch, load_instrumentation(manifest))
    else:
        assert build_runtime_probe_plan("k31", launch, load_instrumentation(manifest)).cells


@pytest.mark.parametrize("interrupt", [False, True])
def test_private_probe_cleanup_failure_persists_phase_and_preserves_interrupt(tmp_path, monkeypatch, interrupt):
    from types import SimpleNamespace

    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, _ = _inputs(tmp_path)
    launch["deployment"]["extra_env"] = [{"name": "HF_TOKEN", "value": "synthetic-private-cleanup"}]
    directories = []

    def delivery(*args):
        path = tmp_path / f"private-{len(directories)}"
        path.mkdir()
        directories.append(path)

        def cleanup():
            raise OSError("private deletion denied")

        return SimpleNamespace(name=str(path), cleanup=cleanup)

    class Executor(_SyntheticExecutor):
        def execute(self, pods):
            super().execute(pods)
            if interrupt:
                raise KeyboardInterrupt

    monkeypatch.setattr(runtime_probe, "new_delivery_directory", delivery)
    monkeypatch.setattr(runner, "_cell_runner", lambda p, c, m, d: Executor(p, c, d, []))
    output = tmp_path / "probe"
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            runtime_probe.probe_runtime({"tp4": launch}, instrumentation=manifest, output_dir=output, execute=True)
    else:
        report = runtime_probe.probe_runtime({"tp4": launch}, instrumentation=manifest, output_dir=output, execute=True)
        assert report["status"] == "failed", report
    index = json.loads((output / "observations.json").read_text())
    phase = index["configurations"]["tp4"]["attempts"][0]["phases"]["prefill"]
    assert phase["status"] == ("interrupted" if interrupt else "cleanup_failed")
    assert phase["private_cleanup_error"] == "private deletion denied"
    assert phase["artifacts"]


def test_slurm_resume_reclaims_abandoned_private_staging_only_after_teardown(tmp_path, monkeypatch):
    from collector.fpm_forward import runner, runtime_probe
    from collector.fpm_forward.private_inputs import RECEIPT

    launch, manifest, _ = _inputs(tmp_path)
    launch["deployment"]["extra_env"] = [{"name": "HF_TOKEN", "value": "synthetic-abandoned-staging"}]
    blocked = True
    staged = []
    executions = []

    class Executor(_SyntheticExecutor):
        def stage(self, pods, files):
            super().stage(pods, files)
            staged.append(self.files["run.sh"].parent)

        def execute(self, pods):
            super().execute(pods)
            if blocked:
                raise KeyboardInterrupt

        def cleanup(self):
            if blocked and executions:
                raise RuntimeError("Slurm step still running")

    monkeypatch.setattr(runner, "_cell_runner", lambda p, c, m, d: Executor(p, c, d, executions))
    monkeypatch.setattr(runner, "_salvage_artifacts", lambda *args: None)
    output = tmp_path / "probe"
    with pytest.raises(KeyboardInterrupt):
        runtime_probe.probe_runtime({"tp4": launch}, instrumentation=manifest, output_dir=output, execute=True)
    old = staged[0]
    assert old.exists() and not old.is_relative_to(output)
    receipt = next(output.rglob(RECEIPT))
    failed = runtime_probe.probe_runtime(
        {"tp4": launch}, instrumentation=manifest, output_dir=output, execute=True, resume=True
    )
    assert failed["status"] == "failed" and old.exists() and receipt.exists()
    blocked = False
    report = runtime_probe.probe_runtime(
        {"tp4": launch}, instrumentation=manifest, output_dir=output, execute=True, resume=True
    )
    assert report["status"] == "completed", report
    assert not old.exists() and not receipt.exists()
    assert all(not path.exists() for path in staged)


@pytest.mark.parametrize("interrupt,cleanup_fails", [(False, False), (True, False), (False, True)])
def test_probe_sanitization_failure_persists_terminal_phase(tmp_path, monkeypatch, interrupt, cleanup_fails):
    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, _ = _inputs(tmp_path)
    executions = []

    class Executor(_SyntheticExecutor):
        def execute(self, pods):
            super().execute(pods)
            if interrupt:
                raise KeyboardInterrupt

        def cleanup(self):
            if cleanup_fails and executions:
                raise OSError("teardown denied")

    def sanitize(directory, values):
        logs = Path(directory) / "logs"
        logs.mkdir(exist_ok=True)
        (logs / "unsafe.log").write_text("synthetic private diagnostic")
        raise OSError("diagnostic rewrite denied")

    monkeypatch.setattr(runtime_probe, "redact_diagnostics", sanitize)
    monkeypatch.setattr(runner, "_cell_runner", lambda p, c, m, d: Executor(p, c, d, executions))
    monkeypatch.setattr(runner, "_salvage_artifacts", lambda *a: None)
    output = tmp_path / "probe"
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            runtime_probe.probe_runtime({"tp4": launch}, instrumentation=manifest, output_dir=output, execute=True)
    else:
        result = runtime_probe.probe_runtime({"tp4": launch}, instrumentation=manifest, output_dir=output, execute=True)
        assert result["status"] == "failed"
        assert "diagnostic sanitization" in str(result)
    index = json.loads((output / "observations.json").read_text())
    phase = index["configurations"]["tp4"]["attempts"][0]["phases"]["prefill"]
    assert phase["status"] == ("interrupted" if interrupt else "cleanup_failed" if cleanup_fails else "failed")
    assert phase["diagnostic_error"] == "diagnostic rewrite denied"
    assert not list(output.rglob("unsafe.log"))
    assert phase["artifacts"] == []
    assert not list(output.rglob("evidence-*"))
    if cleanup_fails:
        assert phase["cleanup_error"] == "teardown denied"


def test_probe_indexes_phase_before_private_render_interruption(tmp_path, monkeypatch):
    from collector.fpm_forward import runner, runtime_probe
    from collector.fpm_forward.private_inputs import RECEIPT

    launch, manifest, _ = _inputs(tmp_path)
    launch["deployment"]["extra_env"] = [{"name": "HF_TOKEN", "value": "synthetic-render-interrupt"}]
    original = runner._render_cell
    output = tmp_path / "probe"
    private_paths = []

    def render(plan, cell, directory, *args, **kwargs):
        original(plan, cell, directory, *args, **kwargs)
        if kwargs.get("resolve_environment", True):
            saved = json.loads((output / "observations.json").read_text())
            phase = saved["configurations"]["tp4"]["attempts"][-1]["phases"][cell.workload_kind]
            assert phase["status"] == "running"
            assert phase["launch_manifest"] and phase["resource_manifest"]
            private_paths.append(Path(directory))
            # Model a host loss: no catch/finally runs after this render.
            raise SystemExit("synthetic host loss")

    # SystemExit is still subject to finally in-process, so leave delivery for
    # resume as an abruptly killed process would, then use the saved running index.
    monkeypatch.setattr(runner, "_render_cell", render)
    from collector.fpm_forward import private_inputs

    cleanup = private_inputs.SlurmDelivery.cleanup
    monkeypatch.setattr(private_inputs.SlurmDelivery, "cleanup", lambda self: None)
    with pytest.raises(SystemExit):
        runtime_probe.probe_runtime({"tp4": launch}, instrumentation=manifest, output_dir=output, execute=True)
    assert private_paths[0].exists()
    monkeypatch.setattr(private_inputs.SlurmDelivery, "cleanup", cleanup)
    monkeypatch.setattr(runner, "_render_cell", original)
    monkeypatch.setattr(runner, "_cell_runner", lambda p, c, m, d: _SyntheticExecutor(p, c, d, []))
    report = runtime_probe.probe_runtime(
        {"tp4": launch}, instrumentation=manifest, output_dir=output, execute=True, resume=True
    )
    assert report["status"] == "completed", report
    assert not private_paths[0].exists()
    assert not list(output.glob("attempts/*/*/prefill/" + RECEIPT))


@pytest.mark.parametrize("credential", ["matching", "missing", "changed"])
def test_kubernetes_recovery_sanitizes_salvaged_diagnostics(tmp_path, monkeypatch, credential):
    from collector.fpm_forward import runner, runtime_probe
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation

    launch, manifest, _ = _inputs(tmp_path)
    token = "synthetic-salvaged-credential"
    launch["deployment"] = {
        "executor": "kubernetes",
        "image": "synthetic:image",
        "extra_env": [{"name": "HF_TOKEN", "value": token}],
    }

    class Executor(_SyntheticExecutor):
        def execute(self, pods):
            super().execute(pods)
            raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_cell_runner", lambda p, c, m, d: Executor(p, c, d, []))
    monkeypatch.setattr(runner, "_salvage_artifacts", lambda *args: None)
    output = tmp_path / "probe"
    with pytest.raises(KeyboardInterrupt):
        runtime_probe.probe_runtime({"tp4": launch}, instrumentation=manifest, output_dir=output, execute=True)
    index = json.loads((output / "observations.json").read_text())
    entry = index["configurations"]["tp4"]
    attempt = entry["attempts"][0]
    if credential == "matching":
        monkeypatch.setenv("HF_TOKEN", token)
    elif credential == "changed":
        monkeypatch.setenv("HF_TOKEN", "wrong-original-value")
    else:
        monkeypatch.delenv("HF_TOKEN", raising=False)

    def salvage(resource, cell_id):
        raw = resource.cell_dir / "raw/recovered"
        raw.mkdir(parents=True)
        (raw / "resolved-config-private.json").write_text(json.dumps({"environment": {"HF_TOKEN": token}}))
        (raw / "benchmark.json").write_text('{"latency_ms": 1.25}')
        logs = resource.cell_dir / "logs"
        logs.mkdir(exist_ok=True)
        (logs / "recovered.log").write_text(token)

    monkeypatch.setattr(runner, "_salvage_artifacts", salvage)
    plan = runtime_probe.build_runtime_probe_plan("tp4", entry["launch"], load_instrumentation(manifest))
    runtime_probe._cleanup_previous_attempt(output, plan, attempt)
    references = attempt["phases"]["prefill"]["recovery_artifacts"]
    assert any(ref["path"].endswith("recovered/benchmark.json") for ref in references)
    configs = [ref for ref in references if ref["path"].endswith("resolved-config-private.json")]
    assert bool(configs) == (credential == "matching")
    for path in output.rglob("*"):
        if path.is_file():
            assert token.encode() not in path.read_bytes(), path


def test_slurm_delivery_removes_private_directory_on_receipt_failure(tmp_path, monkeypatch):
    from collector.fpm_forward import private_inputs, runner

    private = tmp_path / "private"
    monkeypatch.setenv("AISIMULATE_PRIVATE_DELIVERY_ROOT", str(private))

    def fail(*args):
        raise OSError("receipt failed")

    monkeypatch.setattr(runner, "_atomic_json", fail)
    campaign = tmp_path / "campaign"
    with pytest.raises(OSError, match="receipt failed"):
        private_inputs.SlurmDelivery(campaign, campaign / "cell")
    assert list(private.iterdir()) == []


@pytest.mark.parametrize("name", ["FPM_SLURM_CPUS_PER_TASK", "FPM_SLURM_CPU_BIND"])
def test_slurm_rejects_deployment_environment_overriding_launcher(tmp_path, name):
    from collector.fpm_forward import runner, runtime_probe
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation

    launch, manifest, _ = _inputs(tmp_path)
    assert launch["deployment"]["executor"] == "slurm"
    plan = runtime_probe.build_runtime_probe_plan("custom", launch, load_instrumentation(manifest))
    overrides = {"K8sConfig": {"extra_env": [{"name": name, "value": "32"}]}}
    with pytest.raises(ValueError, match="owned by the Slurm launcher"):
        runner._cell_generator_overrides(plan, plan.cells[0], overrides)
