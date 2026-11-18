# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host side of an SGLang self-benchmark collection: entry gates, plan, staging, provenance and validation.

No SGLang, no GPU and no cluster are involved. The cell builder (``sglang_fpm_builder.py``) is replaced with a
double where a test reaches it, except in the campaign test, which renders with the real one. Every result file
is a FAKE of ``test_fpm_sglbench_artifact.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import re
import shlex
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import collector.fpm_forward as fpm_forward
import pytest
from collector.fpm_forward import cli, planner, runner, slurm
from collector.fpm_forward.config import FPMCollectionOptions
from collector.fpm_forward.database import aggregate_cell
from collector.fpm_forward.native_artifact import validate_native_collection
from collector.fpm_forward.planner import build_collection_plan
from collector.fpm_forward.repeatability import load_repeatability_source
from collector.fpm_forward.runtime import sglbench_artifact as artifact
from collector.fpm_forward.runtime.sglbench_compat import TESTED_VERSIONS
from collector.fpm_forward.runtime_memory import saved_plan_identity
from collector.fpm_forward.topology import topology_strategy
from collector.fpm_forward.types import ParallelTopology

from aisim_core.sdk.fpm_identity import LEGACY_EXECUTION_IDENTITY

from .test_fpm_runner import _cell, _native_payload, _plan, _write_provenance
from .test_fpm_sglbench_artifact import (
    LAUNCHER_PRODUCER,
    POD_SGLANG_VERSION,
    _pod,
    _points,
    _rank_artifacts,
    _sglang_plan,
)

pytestmark = pytest.mark.unit

RUNTIME = Path(__file__).resolve().parents[3] / "collector/fpm_forward/runtime"

MODEL = "private-org/dense-model"
MOE_MODEL = "private-org/moe-model"
SYSTEM = "h200_sxm"
DENSE_CONFIG = {
    "architectures": ["Phi3ForCausalLM"],
    "hidden_size": 3072,
    "intermediate_size": 8192,
    "num_hidden_layers": 32,
    "num_attention_heads": 24,
    "num_key_value_heads": 8,
    "vocab_size": 200064,
    "max_position_embeddings": 131072,
    "torch_dtype": "bfloat16",
}
MOE_CONFIG = {
    "architectures": ["NewMoeDsaForCausalLM"],
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "hidden_size": 4096,
    "intermediate_size": 8192,
    "num_hidden_layers": 4,
    "vocab_size": 32000,
    "n_routed_experts": 8,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "max_position_embeddings": 4096,
}

# The hash and the cell ids of the plan of _plan_inputs with backend "vllm". They were computed with planner.py
# as it was before SGLang was admitted (commit e26ff186), so they move only when plan identity itself changes.
VLLM_PLAN_SHA256 = "4ce0a4bad64ad2381905bae7a2f416107cd6adacfccdf0c5904d090da85c14ca"
VLLM_CELL_IDS = ["fpm-f28a6836df315638", "fpm-b80365929b21ac58", "fpm-f3124a62ef358b3e", "fpm-a7b1dd3127d7797a"]

# The program a pod runs to write its provenance file, as it was before the backend switch, and the check that
# follows it when the plan names a runtime version.
VLLM_PROVENANCE_PROGRAM = (
    "import importlib.metadata, json, pathlib, sys; "
    "payload = json.loads(sys.argv[1]); "
    "payload['runtime'] = {'backend': 'vllm', "
    "'backend_version': importlib.metadata.version('vllm')}; "
    "path = pathlib.Path('/results') / sys.argv[2]; "
    "\ntry:\n"
    "    payload['runtime']['dynamo_version'] = importlib.metadata.version('ai-dynamo')\n"
    "except importlib.metadata.PackageNotFoundError:\n"
    "    pass\n"
    "path.write_text(json.dumps(payload, sort_keys=True) + '\\n')"
)
RUNTIME_VERSION_CHECK = (
    "\nactual = payload['runtime']['backend_version']\n"
    "if actual != sys.argv[3]:\n"
    "    raise RuntimeError("
    "f'FPM observed runtime changed: actual={actual!r}, expected={sys.argv[3]!r}. '"
    "'Resolve the collection runtime again in a new output directory.')\n"
)

# What every cell stages before its backend's own files, and the launcher's files as the pattern finds them.
STAGED_FOR_EVERY_CELL = ["run.sh", "fpm_env.sh", "collector-runtime-env.sh", "fpm_exec.sh"]
SGLBENCH_FILES = sorted(path.name for path in RUNTIME.glob("sglbench_*.py"))


@pytest.fixture(autouse=True)
def _hermetic_plans(monkeypatch):
    """Plan without a git checkout and without a memory estimate, so a plan hash depends on its inputs alone."""

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("memory estimate intentionally unavailable")

    monkeypatch.setattr(planner, "_git_revision", lambda: "test-revision")
    monkeypatch.setattr("collector.fpm_forward.memory_admission.KVCacheEstimator.from_request", unavailable)


def _options(**overrides) -> FPMCollectionOptions:
    values = {"fpm_max_gpus": 2, "fpm_prefill_cudagraph_policy": "runtime", **overrides}
    return FPMCollectionOptions.from_args(argparse.Namespace(**values))


def _config(tmp_path: Path, document: dict = DENSE_CONFIG) -> str:
    """Write a model config and return its path, as --fpm-model-config takes it. One file per architecture."""
    path = tmp_path / f"{document['architectures'][0]}.json"
    path.write_text(json.dumps(document))
    return str(path)


def _plan_inputs(tmp_path: Path, **overrides) -> dict:
    """The inputs of a dense TP collection on one and two GPUs, with everything an SGLang plan must give."""
    inputs = {
        "backend": "sglang",
        "model_path": MODEL,
        "model_architecture": "Phi3ForCausalLM",
        "system": SYSTEM,
        "selected_ops": {"attention_context", "attention_generation"},
        "has_model_cases": False,
        "options": _options(),
        "model_config_path": _config(tmp_path),
        # --fpm-runtime-backend-version and --fpm-backend-version (entry.py, resolve_inputs).
        "collector_config": {"runtime_backend_version": "0.5.23", "aic_database_version": "0.5.23"},
    }
    return {**inputs, **overrides}


def _moe_plan_inputs(tmp_path: Path, **overrides) -> dict:
    """The inputs of a collection of the FAKE MoE model on one and two GPUs, with the TEP preset."""
    inputs = {
        "model_path": MOE_MODEL,
        "model_architecture": "NewMoeDsaForCausalLM",
        "model_config_path": _config(tmp_path, MOE_CONFIG),
        # --fpm-parallel-presets tep. The default preset of a MoE model also plans DEP cells.
        "options": _options(fpm_parallel_presets=["tep"]),
    }
    return _plan_inputs(tmp_path, **{**inputs, **overrides})


def _cells(plan) -> dict[str, list]:
    return {phase: [cell for cell in plan.cells if cell.workload_kind == phase] for phase in ("prefill", "decode")}


# ---------------------------------------------------------------------------------------------------------------
# Entry gates
# ---------------------------------------------------------------------------------------------------------------


def _cli_argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        *("--backend", "sglang", "--model-path", MODEL, "--model-architecture", "Phi3ForCausalLM"),
        *("--fpm-model-config", _config(tmp_path), "--gpu", SYSTEM, "--fpm-max-gpus", "2", "--plan-only"),
        *extra,
    ]


def test_the_cli_accepts_sglang_and_keeps_vllm_as_the_default(capsys):
    parser = cli._parser()

    assert parser.parse_args(["--backend", "sglang"]).backend == "sglang"
    assert parser.parse_args([]).backend == "vllm"
    with pytest.raises(SystemExit) as refused:
        parser.parse_args(["--backend", "trtllm"])
    assert refused.value.code == 2
    assert "invalid choice: 'trtllm'" in capsys.readouterr().err


def test_the_cli_plans_an_sglang_collection(tmp_path, monkeypatch, capsys):
    # main() sets this variable; registering it here restores the environment after the test.
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL)
    versions = ("--fpm-backend-version", "0.5.23", "--fpm-runtime-backend-version", "0.5.23")

    assert cli.main(_cli_argv(tmp_path, "--fpm-prefill-cudagraph-policy", "runtime", *versions)) == 0

    printed = json.loads(capsys.readouterr().out)
    assert (printed["backend"], printed["backend_version"], printed["runtime_backend_version"]) == (
        "sglang",
        "0.5.23",
        "0.5.23",
    )
    assert [
        (cell["workload_kind"], cell["parallel_strategy"], cell["topology"]["tp"]) for cell in printed["cells"]
    ] == [
        ("prefill", "single", 1),
        ("prefill", "tp", 2),
        ("decode", "single", 1),
        ("decode", "tp", 2),
    ]
    assert saved_plan_identity(printed).backend == "sglang"


def test_the_cli_reports_a_refused_sglang_plan_as_a_usage_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL)

    with pytest.raises(SystemExit) as refused:
        cli.main(_cli_argv(tmp_path))

    assert refused.value.code == 2
    error = capsys.readouterr().err
    assert "error: SGLang FPM collection requires --fpm-runtime-backend-version" in error
    assert "Traceback" not in error


def test_a_backend_other_than_vllm_and_sglang_is_still_refused(tmp_path):
    with pytest.raises(ValueError, match="supports only backend=vllm and backend=sglang, got 'trtllm'"):
        build_collection_plan(**_plan_inputs(tmp_path, backend="trtllm"))


# ---------------------------------------------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------------------------------------------


def test_a_dense_tp_plan_builds_and_its_saved_form_is_accepted(tmp_path):
    plan = build_collection_plan(**_plan_inputs(tmp_path))

    assert (plan.backend, plan.backend_version, plan.runtime_backend_version) == ("sglang", "0.5.23", "0.5.23")
    assert plan.capability.aic_database_version == "0.5.23"
    # One cell per phase and GPU count. The one-GPU cell of the TP preset has the strategy "single".
    assert [(cell.workload_kind, cell.parallel_strategy, cell.topology.to_dict()) for cell in plan.cells] == [
        (phase, strategy, {"tp": tp, "pp": 1, "dp": 1, "moe_tp": 1, "moe_ep": 1, "cp": 1})
        for phase in ("prefill", "decode")
        for strategy, tp in (("single", 1), ("tp", 2))
    ]
    assert {cell.execution_identity for cell in plan.cells} == {LEGACY_EXECUTION_IDENTITY}
    # No vLLM observer and no vLLM marker is planned for it.
    assert {tuple(cell.backend_policy.expected_markers) for cell in plan.cells} == {()}
    saved = plan.to_dict()
    assert "runtime_memory_policy" not in saved and "runtime_observation" not in saved

    # The saved plan names the launcher as the owner of its points, and no Dynamo scheduler.
    assert saved["point_generation"]["owner"] == "collector.fpm_forward.runtime.sglbench_launcher"
    assert saved["point_generation"]["precondition"] == "sglang_engine_initialized"
    assert saved["point_generation"]["point_admission"] == "sglbench_feasibility_rules"
    # The reader of saved plans recomputes the hash from the saved form and rebuilds every cell.
    identity = saved_plan_identity(json.loads(json.dumps(saved)))
    assert (identity.backend, identity.sha256) == ("sglang", plan.sha256)
    assert (identity.backend_version, identity.runtime_backend_version) == ("0.5.23", "0.5.23")
    assert identity.cells == plan.cells


def test_admitting_sglang_leaves_the_vllm_plan_of_the_same_inputs_as_it_was(tmp_path, monkeypatch):
    sglang_plan = build_collection_plan(**_plan_inputs(tmp_path))
    # A vLLM plan never enters the SGLang checks.
    for name in ("_require_sglang_launcher_inputs", "_require_sglang_launcher_cells"):
        monkeypatch.setattr(planner, name, lambda **_kwargs: pytest.fail("a vLLM plan entered an SGLang check"))

    vllm_plan = build_collection_plan(**_plan_inputs(tmp_path, backend="vllm"))

    assert vllm_plan.sha256 == VLLM_PLAN_SHA256
    assert [cell.cell_id for cell in vllm_plan.cells] == VLLM_CELL_IDS
    saved = vllm_plan.to_dict()
    assert saved["point_generation"]["owner"] == "dynamo.vllm.instrumented_scheduler.InstrumentedScheduler"
    assert saved["point_generation"]["precondition"] == "vllm_engine_initialized"
    assert saved["point_generation"]["point_admission"] == "dynamo_live_scheduler"
    assert saved_plan_identity(saved).sha256 == VLLM_PLAN_SHA256

    # The backend is part of the plan hash and of every cell id, and of nothing else in a cell.
    assert sglang_plan.sha256 != vllm_plan.sha256
    for sglang_cell, vllm_cell in zip(sglang_plan.cells, vllm_plan.cells, strict=True):
        assert sglang_cell.cell_id != vllm_cell.cell_id
        assert dataclasses.replace(sglang_cell, cell_id=vllm_cell.cell_id) == vllm_cell


def test_each_tested_release_is_admitted_and_is_part_of_the_plan_hash(tmp_path):
    plans = [
        build_collection_plan(
            **_plan_inputs(
                tmp_path, collector_config={"runtime_backend_version": version, "aic_database_version": "my-label"}
            )
        )
        for version in TESTED_VERSIONS
    ]

    assert [plan.runtime_backend_version for plan in plans] == ["0.5.22", "0.5.23"]
    # The data label is the user's and need not be a release.
    assert {(plan.backend_version, plan.capability.aic_database_version) for plan in plans} == {
        ("my-label", "my-label")
    }
    assert plans[0].sha256 != plans[1].sha256


def _refused(tmp_path: Path, message: str, **overrides) -> None:
    with pytest.raises(ValueError, match=message):
        build_collection_plan(**_plan_inputs(tmp_path, **overrides))


def test_an_sglang_plan_must_name_a_tested_release_and_a_data_label(tmp_path):
    _refused(
        tmp_path,
        r"requires --fpm-runtime-backend-version, the SGLang release of the image \(0\.5\.22, 0\.5\.23\)",
        collector_config={"aic_database_version": "0.5.23"},
    )
    _refused(
        tmp_path,
        "requires --fpm-runtime-backend-version",
        collector_config={"runtime_backend_version": None, "aic_database_version": "0.5.23"},
    )
    # 0.5.21 is the release the operator collectors are pinned to; the launcher was not tested with it.
    for untested in ("0.5.21", "0.5.24", "0.5.23.post1", "0.33.0"):
        _refused(
            tmp_path,
            rf"--fpm-runtime-backend-version '{untested}' is not an SGLang release the launcher was tested with",
            collector_config={"runtime_backend_version": untested, "aic_database_version": "0.5.23"},
        )
    _refused(
        tmp_path,
        "requires --fpm-backend-version: without it the plan's database version would be the newest curated SGLang",
        collector_config={"runtime_backend_version": "0.5.23"},
    )


@pytest.mark.parametrize(
    ("inputs", "flag"),
    [
        ({"fpm_profile": {}}, "--fpm-model-profile"),
        ({"runtime_instrumentation": "manifest.json"}, "--fpm-runtime-instrumentation"),
        ({"runtime_launch": {}}, "--fpm-runtime-launch"),
        ({"runtime_configuration": "probe"}, "--fpm-runtime-configuration"),
    ],
    ids=["model-profile", "runtime-instrumentation", "runtime-launch", "runtime-configuration"],
)
def test_the_vllm_only_inputs_of_a_plan_are_refused_by_name(tmp_path, inputs, flag):
    _refused(tmp_path, rf"^{flag} is not supported with --backend sglang: ", **inputs)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"fpm_prefill_cudagraph_policy": None}, "requires --fpm-prefill-cudagraph-policy runtime"),
        ({"fpm_prefill_cudagraph_policy": "explicit"}, "requires --fpm-prefill-cudagraph-policy runtime"),
        ({"fpm_worker_type": "aggregated", "fpm_cudagraph_mode": "NONE"}, "^--fpm-cudagraph-mode is a vLLM"),
        (
            {"fpm_worker_type": "aggregated", "fpm_cudagraph_capture_sizes": [1, 2]},
            "^--fpm-cudagraph-capture-sizes is a vLLM",
        ),
        (
            {"fpm_worker_type": "aggregated", "fpm_max_cudagraph_capture_size": 8},
            "^--fpm-max-cudagraph-capture-size is a vLLM",
        ),
        ({"fpm_enforce_eager": True}, "^--fpm-enforce-eager is a vLLM engine setting"),
        ({"fpm_decoder_replay": True}, "^--fpm-decoder-replay is a vLLM engine setting"),
    ],
    ids=[
        "default-graph-policy",
        "explicit-graph-policy",
        "cudagraph-mode",
        "capture-sizes",
        "max-capture-size",
        "enforce-eager",
        "decoder-replay",
    ],
)
def test_the_vllm_engine_settings_of_a_plan_are_refused_by_name(tmp_path, options, message):
    _refused(tmp_path, message, options=_options(**options))


def test_a_vllm_compilation_config_is_refused_whatever_filled_it(tmp_path, monkeypatch):
    # Today the config is made of the three graph options, which are refused by name before it is looked at.
    monkeypatch.setattr(FPMCollectionOptions, "compilation_config", property(lambda _self: {"fake_key": 1}))

    _refused(tmp_path, r"cannot pass a vLLM compilation config \(\['fake_key'\]\)")


def test_the_engine_limits_an_sglang_cell_can_take_are_admitted(tmp_path):
    # --fpm-max-model-len -1 is the default: the cell then passes no context length.
    assert build_collection_plan(**_plan_inputs(tmp_path)).options.vllm_max_model_len == -1
    limits = {
        "fpm_max_model_len": 8192,
        "fpm_max_num_batched_tokens": 4096,
        "fpm_max_num_seqs": 64,
        "fpm_gpu_memory_utilization": 0.8,
        "fpm_worker_type": "prefill",
    }

    plan = build_collection_plan(**_plan_inputs(tmp_path, options=_options(**limits)))

    assert (plan.options.vllm_max_model_len, plan.options.max_num_batched_tokens) == (8192, 4096)
    assert [cell.workload_kind for cell in plan.cells] == ["prefill", "prefill"]


def test_a_moe_plan_with_the_tep_preset_is_frozen_with_a_single_and_a_tep_cell(tmp_path):
    plan = build_collection_plan(**_moe_plan_inputs(tmp_path))

    # The capability of the checkpoint: expert layers, and no pure TP for this backend.
    assert (plan.capability.is_moe, plan.capability.allow_pure_tp) == (True, False)
    # One cell per phase and GPU count. The one-GPU cell of the TEP preset has the strategy "single" and one
    # expert-parallel rank; the two-GPU cell has one expert-parallel rank per tensor-parallel rank.
    assert [(cell.workload_kind, cell.parallel_strategy, cell.topology.to_dict()) for cell in plan.cells] == [
        (phase, strategy, {"tp": tp, "pp": 1, "dp": 1, "moe_tp": 1, "moe_ep": tp, "cp": 1})
        for phase in ("prefill", "decode")
        for strategy, tp in (("single", 1), ("tep", 2))
    ]
    assert {cell.execution_identity for cell in plan.cells} == {LEGACY_EXECUTION_IDENTITY}
    # Memory admission looked at both topologies and dropped none. The estimate is unavailable in this suite
    # (_hermetic_plans); test_fpm_sglang_moe.py plans a packaged MoE model with the real estimate.
    assert [(decision.topology, decision.disposition) for decision in plan.topology_memory_admission] == [
        (topology, "unknown") for topology in plan.topologies
    ]
    assert [topology.to_dict()["tp"] for topology in plan.topologies] == [1, 2]
    # The one backend policy is the automatic baseline: no engine flag, no vLLM marker, and the four backend
    # identity columns of every row say that the engine decided.
    (policy,) = plan.backend_policies
    assert (policy.policy_id, policy.generator_overrides, policy.expected_markers) == ("baseline_auto", {}, {})
    assert all(cell.backend_policy == policy for cell in plan.cells)
    assert planner.backend_identity_columns(policy) == {
        "moe_backend": "auto",
        "attention_backend": "auto",
        "enable_wideep": False,
        "enable_eplb": False,
    }

    # The saved form names the strategies, and its reader recomputes the hash and rebuilds every cell.
    saved = plan.to_dict()
    assert [(topology["strategy"], topology["tp"], topology["moe_ep"]) for topology in saved["topologies"]] == [
        ("single", 1, 1),
        ("tep", 2, 2),
    ]
    assert saved["counts"]["memory_rejected_topologies"] == 0
    assert saved["point_generation"]["owner"] == "collector.fpm_forward.runtime.sglbench_launcher"
    identity = saved_plan_identity(json.loads(json.dumps(saved)))
    assert (identity.backend, identity.sha256) == ("sglang", plan.sha256)
    assert identity.cells == plan.cells


def test_the_cli_plans_a_moe_collection_with_the_tep_preset(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MOE_MODEL)
    arguments = [
        *("--backend", "sglang", "--model-path", MOE_MODEL, "--model-architecture", "NewMoeDsaForCausalLM"),
        *("--fpm-model-config", _config(tmp_path, MOE_CONFIG), "--gpu", SYSTEM, "--plan-only"),
        *("--fpm-prefill-cudagraph-policy", "runtime"),
        *("--fpm-backend-version", "0.5.23", "--fpm-runtime-backend-version", "0.5.23"),
        *("--fpm-max-gpus", "2", "--fpm-gpu-counts", "1", "2"),
    ]

    assert cli.main([*arguments, "--fpm-parallel-presets", "tep"]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert [
        (cell["workload_kind"], cell["parallel_strategy"], cell["topology"]["tp"], cell["topology"]["moe_ep"])
        for cell in printed["cells"]
    ] == [
        ("prefill", "single", 1, 1),
        ("prefill", "tep", 2, 2),
        ("decode", "single", 1, 1),
        ("decode", "tep", 2, 2),
    ]
    assert saved_plan_identity(printed).backend == "sglang"

    # Without the preset the plan of a MoE model holds DEP cells, and the refusal names the flag to pass.
    with pytest.raises(SystemExit) as refused:
        cli.main(arguments)
    assert refused.value.code == 2
    assert "error: SGLang FPM collection does not support DP attention yet" in capsys.readouterr().err


def test_memory_admission_still_drops_a_moe_cell_that_does_not_fit(tmp_path, monkeypatch):
    # FAKE sizes: the weights do not fit one GPU of the FAKE capacity and fit when two ranks share them.
    capacity = 100 * 2**30

    def estimate(*_args, tp_size, moe_ep_size, **_kwargs):
        assert moe_ep_size == tp_size
        non_kv = 120 * 2**30 // tp_size
        return SimpleNamespace(breakdown={"non_kv_bytes": non_kv, "gpu_memory_capacity_bytes": capacity})

    monkeypatch.setattr("collector.fpm_forward.memory_admission.KVCacheEstimator.from_request", estimate)

    plan = build_collection_plan(**_moe_plan_inputs(tmp_path))

    assert [(decision.topology.tp, decision.disposition) for decision in plan.topology_memory_admission] == [
        (1, "rejected"),
        (2, "admitted"),
    ]
    assert [(cell.workload_kind, cell.parallel_strategy, cell.topology.tp) for cell in plan.cells] == [
        ("prefill", "tep", 2),
        ("decode", "tep", 2),
    ]


@pytest.mark.parametrize(
    ("presets", "message"),
    [
        # The default preset "auto" plans TEP and DEP for a MoE model, and DEP needs DP attention.
        (None, r"does not support DP attention yet \(topology \{'tp': 1, 'pp': 1, 'dp': 2, 'moe_tp': 1, 'moe_ep': 2"),
        (["auto"], "does not support DP attention yet"),
        (["dep"], "does not support DP attention yet"),
        (["tep", "dep"], "does not support DP attention yet"),
        # Refused before the SGLang check: pure TP is admitted for vLLM only, and plain TP for a dense model only.
        (["pure_tp"], "pure TP was requested, but the AIC model/runtime capability profile does not explicitly admit"),
        (["tp"], "MoE models use TEP or DEP; plain TP is not an independent typical preset"),
    ],
    ids=["default", "auto", "dep", "tep-and-dep", "pure-tp", "tp"],
)
def test_the_other_presets_of_a_moe_model_are_refused(tmp_path, presets, message):
    _refused(tmp_path, message, **_moe_plan_inputs(tmp_path, options=_options(fpm_parallel_presets=presets)))


@pytest.mark.parametrize("presets", [None, ["auto"], ["tep"], ["dep"]], ids=["default", "auto", "tep", "dep"])
def test_a_one_gpu_plan_of_a_moe_model_holds_the_single_cell_under_these_presets(tmp_path, presets):
    # On one GPU the TEP and the DEP preset give the same topology, so no preset plans a cell with DP attention.
    options = _options(fpm_max_gpus=1, fpm_parallel_presets=presets)

    plan = build_collection_plan(**_moe_plan_inputs(tmp_path, options=options))

    assert [(cell.workload_kind, cell.parallel_strategy, cell.topology.to_dict()) for cell in plan.cells] == [
        (phase, "single", {"tp": 1, "pp": 1, "dp": 1, "moe_tp": 1, "moe_ep": 1, "cp": 1})
        for phase in ("prefill", "decode")
    ]


def test_the_dp_attention_refusal_names_the_preset_to_pass(tmp_path):
    with pytest.raises(ValueError) as refused:
        build_collection_plan(**_moe_plan_inputs(tmp_path, options=_options()))

    assert str(refused.value) == (
        "SGLang FPM collection does not support DP attention yet (topology "
        "{'tp': 1, 'pp': 1, 'dp': 2, 'moe_tp': 1, 'moe_ep': 2, 'cp': 1}); pass --fpm-parallel-presets tep to "
        "collect the tensor- and expert-parallel cells of a MoE model"
    )


@pytest.mark.parametrize(
    "name",
    [
        "DYN_BENCH_PREFILL_CONTENT",
        "DYN_BENCH_KV_WARMUP_DATASET",
        "DYN_BENCH_KV_WARMUP_CACHE_DIR",
        "DYN_BENCH_KV_WARMUP_SHA256",
    ],
)
def test_a_content_variable_of_the_deployment_inputs_is_part_of_the_plan_hash(tmp_path, name):
    def plan(*environment: dict):
        overrides = {"K8sConfig": {"k8s_image": "example/sglang:test", "extra_env": list(environment)}}
        return build_collection_plan(**_moe_plan_inputs(tmp_path, generator_overrides=overrides))

    # FAKE values: the planner does not read them, it freezes them. The launcher reads them in the pod.
    plans = [plan(), plan({"name": name, "value": "first"}), plan({"name": name, "value": "second"})]

    # K8sConfig.extra_env is the one way such a value reaches a cell, and each value is another plan.
    assert len({saved.sha256 for saved in plans}) == 3
    assert len({saved.generator_config_sha256 for saved in plans}) == 3
    # The cells are the same cells: their ids name the backend, the model, the topology and the dtypes.
    assert len({tuple(cell.cell_id for cell in saved.cells) for saved in plans}) == 1


def test_the_tep_preset_of_a_dense_model_is_still_refused(tmp_path):
    _refused(
        tmp_path,
        r"dense models support only the TP typical preset, got \['tep'\]",
        options=_options(fpm_parallel_presets=["tep"]),
    )


def test_a_model_with_an_execution_identity_is_refused(tmp_path, monkeypatch):
    # The identity DeepSeek V4.1 and GLM-5.3-Flash get: a config hash in the first column.
    identity = ("0" * 64, "full", "none", "text")
    monkeypatch.setattr(planner, "execution_identity", lambda *_args, **_kwargs: identity)

    _refused(tmp_path, r"does not support a model with an FPM execution identity \(\{'model_config_sha256': '0000")
    # The refusal does not depend on the model kind: a MoE model with such an identity is refused the same way.
    _refused(
        tmp_path,
        r"does not support a model with an FPM execution identity \(\{'model_config_sha256': '0000",
        **_moe_plan_inputs(tmp_path),
    )


@pytest.mark.parametrize("axis", ["pp", "dp", "moe_tp", "moe_ep", "cp"])
def test_only_tensor_parallel_topologies_of_a_dense_model_are_admitted(axis):
    sizes = {"tp": 2, "pp": 1, "dp": 1, "moe_tp": 1, "moe_ep": 1, "cp": 1}
    dense = {"is_moe": False, "execution": LEGACY_EXECUTION_IDENTITY}
    tensor_parallel = tuple(ParallelTopology(**{**sizes, "tp": tp}) for tp in (1, 2, 4, 8))

    planner._require_sglang_launcher_cells(**dense, topologies=tensor_parallel)

    other = ParallelTopology(**{**sizes, axis: 2})
    with pytest.raises(ValueError, match=rf"supports tensor-parallel cells only .*got topology .*'{axis}': 2"):
        planner._require_sglang_launcher_cells(**dense, topologies=(*tensor_parallel, other))


def _moe_topology(**sizes: int) -> ParallelTopology:
    """A topology of a MoE model: one expert-parallel rank per tensor-parallel rank unless ``sizes`` say otherwise."""
    tp = sizes.get("tp", 2)
    return ParallelTopology(**{"tp": tp, "pp": 1, "dp": 1, "moe_tp": 1, "moe_ep": tp, "cp": 1, **sizes})


def test_the_single_and_the_tep_topologies_of_a_moe_model_are_admitted():
    moe = {"is_moe": True, "execution": LEGACY_EXECUTION_IDENTITY}
    admitted = tuple(_moe_topology(tp=tp) for tp in (1, 2, 4, 8))
    assert [topology_strategy(topology, is_moe=True) for topology in admitted] == ["single", "tep", "tep", "tep"]

    planner._require_sglang_launcher_cells(**moe, topologies=admitted)

    # A dense model has no expert-parallel rank to spread: the same topologies above one GPU are refused for it.
    with pytest.raises(ValueError, match=r"supports tensor-parallel cells only .*got topology .*'moe_ep': 2"):
        planner._require_sglang_launcher_cells(is_moe=False, execution=LEGACY_EXECUTION_IDENTITY, topologies=admitted)


@pytest.mark.parametrize(
    ("sizes", "message"),
    [
        # DEP at two and four GPUs, and DP attention beside tensor parallelism.
        ({"tp": 1, "dp": 2, "moe_ep": 2}, r"does not support DP attention yet \(topology .*'dp': 2.*\); pass "),
        ({"tp": 1, "dp": 4, "moe_ep": 4}, r"does not support DP attention yet \(topology .*'dp': 4.*\); pass "),
        ({"tp": 2, "dp": 2, "moe_ep": 4}, r"does not support DP attention yet \(topology .*'dp': 2.*\); pass "),
        # Pure TP, an expert-parallel size that is not the tensor-parallel size, and the two fixed axes.
        ({"tp": 2, "moe_tp": 2, "moe_ep": 1}, r"moe_ep must equal tp\); got topology .*'moe_tp': 2, 'moe_ep': 1"),
        ({"tp": 2, "moe_ep": 1}, r"moe_ep must equal tp\); got topology .*'tp': 2, .*'moe_ep': 1"),
        ({"tp": 4, "moe_ep": 2}, r"moe_ep must equal tp\); got topology .*'tp': 4, .*'moe_ep': 2"),
        ({"tp": 1, "moe_ep": 2}, r"moe_ep must equal tp\); got topology .*'tp': 1, .*'moe_ep': 2"),
        ({"tp": 4, "moe_tp": 2, "moe_ep": 2}, r"moe_ep must equal tp\); got topology .*'moe_tp': 2, 'moe_ep': 2"),
        # The expert-parallel size is the tensor-parallel size, and only moe_tp is not 1.
        ({"tp": 2, "moe_tp": 2}, r"moe_ep must equal tp\); got topology .*'tp': 2, .*'moe_tp': 2, 'moe_ep': 2"),
        ({"pp": 2}, r"moe_ep must equal tp\); got topology .*'pp': 2"),
        ({"cp": 2}, r"moe_ep must equal tp\); got topology .*'cp': 2"),
    ],
    ids=[
        *("dep-2", "dep-4", "dp-with-tp", "pure-tp", "no-ep", "ep-below-tp", "ep-above-tp", "mixed"),
        *("moe-tp-beside-ep", "pp", "cp"),
    ],
)
def test_every_other_topology_of_a_moe_model_is_refused(sizes, message):
    moe = {"is_moe": True, "execution": LEGACY_EXECUTION_IDENTITY}
    admitted = (_moe_topology(tp=1), _moe_topology(tp=2))

    with pytest.raises(ValueError, match=message) as refused:
        planner._require_sglang_launcher_cells(**moe, topologies=(*admitted, _moe_topology(**sizes)))

    if "DP attention" not in message:
        assert str(refused.value).startswith(
            "SGLang FPM collection supports the tensor- and expert-parallel cells of a MoE model only (pp, dp, cp "
            "and moe_tp must be 1, and moe_ep must equal tp); got topology {"
        )


# ---------------------------------------------------------------------------------------------------------------
# The paths that stay vLLM only
# ---------------------------------------------------------------------------------------------------------------


def test_the_vllm_observers_stay_off_for_an_sglang_plan():
    cell = _cell()

    def plan(backend: str) -> SimpleNamespace:
        # Everything that switches both observers on for vLLM: a version that an audited vLLM release line
        # matches, and a profile whose memory is still pending.
        return SimpleNamespace(
            backend=backend,
            runtime_instrumentation=None,
            runtime_backend_version="0.33.0",
            fpm_profile=object(),
            deployment_profile=lambda _cell: SimpleNamespace(resources=SimpleNamespace(memory_source="pending")),
        )

    assert runner._observe_runtime_memory(plan("vllm"), cell) is True
    assert runner._observe_runtime_execution(plan("vllm"), cell) is True
    assert runner._observe_runtime_memory(plan("sglang"), cell) is False
    assert runner._observe_runtime_execution(plan("sglang"), cell) is False


def test_repeatability_refuses_an_sglang_campaign(tmp_path):
    campaigns = {}
    for backend in ("vllm", "sglang"):
        plan = build_collection_plan(**_plan_inputs(tmp_path, backend=backend))
        campaigns[backend] = tmp_path / backend
        campaigns[backend].mkdir()
        (campaigns[backend] / "collection-plan.json").write_text(json.dumps(plan.to_dict()))

    assert load_repeatability_source(campaigns["vllm"]).sha256 == VLLM_PLAN_SHA256
    with pytest.raises(ValueError, match="supports only vLLM campaigns; the source has backend 'sglang'"):
        load_repeatability_source(campaigns["sglang"])


# ---------------------------------------------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------------------------------------------


class _VllmRenderStarted(Exception):
    """Raised where the vLLM path of ``_render_cell`` begins, in the place of a Generator render."""


def test_an_sglang_cell_is_rendered_by_the_sglang_builder_from_the_vllm_render(tmp_path, monkeypatch):
    plan = build_collection_plan(**_plan_inputs(tmp_path))
    cell = plan.cells[0]
    overrides = {"K8sConfig": {"k8s_image": "example/sglang:test"}}
    calls = []

    def render_cell(sglang_plan, sglang_cell, cell_dir, generator_overrides, *, smoke=False, render_vllm_cell):
        calls.append((sglang_plan, sglang_cell, cell_dir, generator_overrides, smoke, render_vllm_cell))
        return {"rendered_by": "the double"}

    def vllm_path(render_plan, *_args, **_kwargs):
        raise _VllmRenderStarted(render_plan.backend)

    double = types.ModuleType("collector.fpm_forward.sglang_fpm_builder")
    double.render_cell = render_cell
    monkeypatch.setitem(sys.modules, double.__name__, double)
    monkeypatch.setattr(fpm_forward, "sglang_fpm_builder", double, raising=False)
    monkeypatch.setattr(runner, "_cell_generator_overrides", vllm_path)

    assert runner._render_cell(plan, cell, tmp_path, overrides, smoke=True) == {"rendered_by": "the double"}

    assert calls == [(plan, cell, tmp_path, overrides, True, runner._render_cell)]
    # What the builder is handed renders the vLLM form of the cell, and does not come back to the builder.
    with pytest.raises(_VllmRenderStarted, match="^vllm$"):
        calls[0][5](dataclasses.replace(plan, backend="vllm"), cell, tmp_path, overrides, smoke=True)
    assert len(calls) == 1


# ---------------------------------------------------------------------------------------------------------------
# Staging, the attempt and the checkpoint record
# ---------------------------------------------------------------------------------------------------------------


def _run_cells(monkeypatch, root: Path, plan):
    """Run every cell of ``plan`` on an in-process cluster.

    Return, per cell id, the names it staged and the keywords of its attempt; the backends the collection
    summary was asked for; and the cells of the checkpoint.
    """
    seen = SimpleNamespace(staged={}, attempts={}, summary_backends=[])

    def render_cell(_plan, _cell, cell_dir, _overrides, **_kwargs):
        (cell_dir / "k8s_deploy.yaml").write_text("apiVersion: v1\nkind: Pod\nmetadata:\n  name: cell\n")
        for name in ("run.sh", "fpm_env.sh", "collector-runtime-env.sh"):
            (cell_dir / name).write_text("#!/bin/sh\n")

    class FakeResource:
        def __init__(self, _manifest, cell_dir, **_slurm_settings):
            self.cell_id = cell_dir.name

        def apply(self):
            pass

        def wait_ready(self, _expected_nodes):
            return ["pod-0"]

        def stage(self, _pods, files):
            seen.staged[self.cell_id] = [path.name for path in files]

        def prepare_attempt(self, _pods, **keywords):
            seen.attempts[self.cell_id] = keywords

        def execute(self, _pods):
            pass

        def collect(self, _pods, *, require_benchmark=True):
            pass

        def cleanup(self):
            pass

    def summary(_cell, _raw_root, **keywords):
        seen.summary_backends.append(keywords["backend"])
        return {"measured_point_count": 3, "measured_new_token_axis_count": 2}

    monkeypatch.setattr(runner, "_render_cell", render_cell)
    monkeypatch.setattr(runner, "KubernetesCellRunner", FakeResource)
    monkeypatch.setattr(slurm, "SlurmCellRunner", FakeResource)
    monkeypatch.setattr(runner, "_runtime_collection_summary", summary)
    errors = runner.run_collection(
        plan,
        generator_overrides={},
        checkpoint_dir=str(root / "checkpoints"),
        artifact_root=str(root / "artifacts"),
        resume=False,
        retry_failed=False,
        cell_limit=len(plan.cells),
        publish_database=False,
    )
    assert errors == []
    checkpoint = json.loads((root / "checkpoints" / "fpm_forward.json").read_text())
    return seen, checkpoint["cells"]


def test_the_pattern_finds_the_launcher_and_its_preflight():
    assert set(SGLBENCH_FILES) >= {
        "sglbench_artifact.py",
        "sglbench_compat.py",
        "sglbench_content.py",
        "sglbench_driver.py",
        "sglbench_grid.py",
        "sglbench_launcher.py",
        "sglbench_preflight.py",
        "sglbench_recorder.py",
    }


@pytest.mark.parametrize("executor", ["kubernetes", "slurm"])
def test_an_sglang_cell_stages_the_launcher_in_the_place_of_the_vllm_preflight(tmp_path, monkeypatch, executor):
    transport = {"fpm_executor": "slurm", "fpm_slurm_container_image": "example/sglang:test"}
    options = _options(**(transport if executor == "slurm" else {}))
    plan = build_collection_plan(**_plan_inputs(tmp_path, options=options))

    seen, records = _run_cells(monkeypatch, tmp_path, plan)

    # The gate runs a CPU observation on every Slurm cell and loads this one file for it.
    slurm_observer = ["fpm_memory_observer.py"] if executor == "slurm" else []
    assert seen.staged == {
        cell.cell_id: [*STAGED_FOR_EVERY_CELL, *slurm_observer, *SGLBENCH_FILES] for cell in plan.cells
    }
    assert not any("preflight.py" in names for names in seen.staged.values())
    # The pod is told which distribution to record, and the plan's two versions as for any plan.
    for cell in plan.cells:
        keywords = dict(seen.attempts[cell.cell_id])
        assert len(keywords.pop("attempt_id")) == 32
        assert keywords == {
            "cell_id": cell.cell_id,
            "plan_sha256": plan.sha256,
            "expected_backend_version": "0.5.23",
            "runtime_backend_version": "0.5.23",
            "backend": "sglang",
        }
    # The collected files are validated as those of an SGLang cell.
    assert seen.summary_backends == ["sglang"] * len(plan.cells)

    # The checkpoint holds the cap the cell was given beside the number of new-token values it measured.
    prefill, decode = _cells(plan)["prefill"][0], _cells(plan)["decode"][0]
    assert records[prefill.cell_id]["status"] == "passed"
    assert (
        records[prefill.cell_id]["prefill_max_new_token_samples"],
        records[prefill.cell_id]["measured_new_token_axis_count"],
    ) == (8192, 2)
    assert records[decode.cell_id]["decode_radix_cache"] == "enabled"


def test_a_vllm_cell_stages_what_it_staged_and_its_pod_is_told_no_backend(tmp_path, monkeypatch):
    plan = build_collection_plan(**_plan_inputs(tmp_path, backend="vllm"))

    seen, _records = _run_cells(monkeypatch, tmp_path, plan)

    assert seen.staged == {cell.cell_id: [*STAGED_FOR_EVERY_CELL, "preflight.py"] for cell in plan.cells}
    # The keywords the test doubles of the vLLM suites take: no backend among them.
    for cell in plan.cells:
        keywords = dict(seen.attempts[cell.cell_id])
        assert len(keywords.pop("attempt_id")) == 32
        assert keywords == {
            "cell_id": cell.cell_id,
            "plan_sha256": plan.sha256,
            "expected_backend_version": "0.5.23",
            "runtime_backend_version": "0.5.23",
        }
    assert seen.summary_backends == ["vllm"] * len(plan.cells)


# ---------------------------------------------------------------------------------------------------------------
# The provenance program
# ---------------------------------------------------------------------------------------------------------------

ATTEMPT = {"cell_id": "cell", "plan_sha256": "plan", "attempt_id": "attempt"}


def test_the_vllm_provenance_program_is_the_text_it_was():
    for keywords in ({}, {"backend": "vllm"}):
        command = runner._attempt_provenance_command(**ATTEMPT, **keywords)
        assert command[:2] == ["python3", "-c"]
        assert command[2] == VLLM_PROVENANCE_PROGRAM
        assert json.loads(command[3]) == {
            "schema_name": "aic_fpm_collector_provenance",
            "schema_version": 1,
            **ATTEMPT,
        }
        assert command[4:] == ["collector-provenance.json"]

        pinned = runner._attempt_provenance_command(**ATTEMPT, runtime_backend_version="0.27.0", **keywords)
        assert pinned[2] == VLLM_PROVENANCE_PROGRAM + RUNTIME_VERSION_CHECK
        assert pinned[4:] == ["collector-provenance.json", "0.27.0"]


def test_the_sglang_provenance_program_differs_in_the_backend_and_the_distribution_only():
    command = runner._attempt_provenance_command(**ATTEMPT, runtime_backend_version="0.5.23", backend="sglang")

    assert command[2] == (VLLM_PROVENANCE_PROGRAM + RUNTIME_VERSION_CHECK).replace("'vllm'", "'sglang'")
    assert VLLM_PROVENANCE_PROGRAM.count("'vllm'") == 2
    # The JSON payload and the file name keep their places, and the payload names no backend.
    assert command[3:] == [
        runner._attempt_provenance_command(**ATTEMPT)[3],
        "collector-provenance.json",
        "0.5.23",
    ]


def test_the_sglang_provenance_program_records_the_installed_sglang(tmp_path, monkeypatch):
    def installed(name):
        # An SGLang image: no vllm and no ai-dynamo distribution.
        if name == "sglang":
            return "0.5.23"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(runner, "FPM_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(importlib.metadata, "version", installed)
    identity = {**ATTEMPT, "expected_backend_version": "my-label", "backend": "sglang"}
    command = runner._attempt_provenance_command(**identity, runtime_backend_version="0.5.23")

    monkeypatch.setattr(sys, "argv", ["-c", *command[3:]])
    exec(command[2], {})

    written = json.loads((tmp_path / "collector-provenance.json").read_text())
    assert written == {
        "schema_name": "aic_fpm_collector_provenance",
        "schema_version": 1,
        **ATTEMPT,
        "backend_version": "my-label",
        "runtime": {"backend": "sglang", "backend_version": "0.5.23"},
    }

    # A pod with another release than the plan names writes the file and then fails the attempt.
    drifted = runner._attempt_provenance_command(**identity, runtime_backend_version="0.5.22")
    monkeypatch.setattr(sys, "argv", ["-c", *drifted[3:]])
    with pytest.raises(RuntimeError, match="observed runtime changed: actual='0.5.23', expected='0.5.22'"):
        exec(drifted[2], {})

    # The vLLM program fails in such a pod before it writes anything, which is why the backend is passed.
    (tmp_path / "collector-provenance.json").unlink()
    vllm = runner._attempt_provenance_command(**ATTEMPT)
    monkeypatch.setattr(sys, "argv", ["-c", *vllm[3:]])
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        exec(vllm[2], {})
    assert not (tmp_path / "collector-provenance.json").exists()


@pytest.mark.parametrize("backend", ["trtllm", "sglang'); import os; ('", ""])
def test_the_provenance_program_takes_its_backend_from_a_closed_table(backend):
    with pytest.raises(ValueError, match="knows no runtime distribution for backend"):
        runner._attempt_provenance_command(**ATTEMPT, backend=backend)


def test_both_runners_hand_the_backend_of_an_attempt_to_the_program():
    commands = []
    kubernetes = object.__new__(runner.KubernetesCellRunner)
    kubernetes._exec_checked = lambda _pod, command, timeout: commands.append(command)
    slurm_step = object.__new__(slurm.SlurmCellRunner)
    slurm_step._exec = lambda _unit, command, *, timeout: commands.append(command)
    versions = {"expected_backend_version": "my-label", "runtime_backend_version": "0.5.23"}

    for resource in (kubernetes, slurm_step):
        resource.prepare_attempt(["pod-0"], **ATTEMPT, **versions)
        assert commands[-1] == runner._attempt_provenance_command(**ATTEMPT, **versions)
        resource.prepare_attempt(["pod-0"], **ATTEMPT, **versions, backend="sglang")
        assert commands[-1] == runner._attempt_provenance_command(**ATTEMPT, **versions, backend="sglang")
        assert "version('sglang')" in commands[-1][2]


# ---------------------------------------------------------------------------------------------------------------
# Validation of the collected files
# ---------------------------------------------------------------------------------------------------------------


def _collected(root: Path, phase: str = "prefill", *, installed: str = POD_SGLANG_VERSION, edit=None):
    """A collected SGLang cell with FAKE times: the provenance file of the pod and one complete rank artifact."""
    cell = _cell(phase=phase, strategy="tp")
    pod = _pod(root, cell, sglang_version=installed)
    (payload,) = _rank_artifacts(phase, _points(phase))
    if edit is not None:
        edit(payload)
    artifact.atomic_write_json(str(pod / "benchmark.json"), payload)
    return cell


def test_the_validator_accepts_sglang_provenance_only_when_it_is_asked_for(tmp_path):
    cell = _collected(tmp_path / "sglang")
    raw = tmp_path / "sglang" / "raw"

    collection = validate_native_collection(cell, raw, expected_backend="sglang")
    assert collection.backend_version == POD_SGLANG_VERSION
    assert len(collection.points) == len(_points("prefill"))
    # Without the keyword, and for every caller that has no plan, the collection must be vLLM's.
    for keywords in ({}, {"expected_backend": "vllm"}):
        with pytest.raises(ValueError, match="Collector provenance has invalid runtime identity"):
            validate_native_collection(cell, raw, **keywords)

    # A vLLM collection is accepted as before, with no producer block, and is not an SGLang collection.
    vllm_pod = tmp_path / "vllm" / "pod-0"
    vllm_pod.mkdir(parents=True)
    vllm_cell = _cell()
    _write_provenance(vllm_pod / "collector-provenance.json", cell_id=vllm_cell.cell_id)
    (vllm_pod / "benchmark.json").write_text(json.dumps(_native_payload(phase="prefill", rank=0, dp=1)))
    assert validate_native_collection(vllm_cell, tmp_path / "vllm").backend_version == "0.24.0"
    # Whatever its plan says of expert layers: what an SGLang pod records is asked of an SGLang collection only.
    for expected_moe in (True, False, None):
        collection = validate_native_collection(vllm_cell, tmp_path / "vllm", expected_moe=expected_moe)
        assert collection.backend_version == "0.24.0"
    with pytest.raises(ValueError, match="Collector provenance has invalid runtime identity"):
        validate_native_collection(vllm_cell, tmp_path / "vllm", expected_backend="sglang")


def _drop_producer(payload: dict) -> None:
    del payload["producer"]


def _differs(installed: str, launcher: str) -> str:
    return re.escape(
        "native SGLang producer.sglang_version differs from the Collector provenance runtime: "
        f"expected {installed!r}, found {launcher!r}: "
    )


NO_RELEASE = re.escape("native SGLang rank artifact has no producer.sglang_version: ")
VERSION_MISMATCHES = {
    # name -> (the release installed in the pod, an edit of the launcher's artifact or None, the refusal)
    "another-installed-release": ("0.5.22", None, _differs("0.5.22", "0.5.23")),
    "another-launcher-release": (
        "0.5.23",
        lambda payload: payload["producer"].update(sglang_version="0.5.22"),
        _differs("0.5.23", "0.5.22"),
    ),
    "no-release-in-the-producer-block": (
        "0.5.23",
        lambda payload: payload["producer"].pop("sglang_version"),
        NO_RELEASE,
    ),
    "release-is-not-a-string": (
        "0.5.23",
        lambda payload: payload["producer"].update(sglang_version=523),
        NO_RELEASE,
    ),
    "release-is-empty": ("0.5.23", lambda payload: payload["producer"].update(sglang_version=""), NO_RELEASE),
    "no-producer-block": ("0.5.23", _drop_producer, NO_RELEASE),
}


@pytest.mark.parametrize("name", sorted(VERSION_MISMATCHES))
def test_the_validator_wants_the_launchers_release_to_be_the_installed_one(tmp_path, name):
    installed, edit, refusal = VERSION_MISMATCHES[name]
    # The unedited artifact carries the release of the FAKE pod.
    assert LAUNCHER_PRODUCER == {"sglang_version": POD_SGLANG_VERSION} and POD_SGLANG_VERSION == "0.5.23"
    cell = _collected(tmp_path, installed=installed, edit=edit)

    with pytest.raises(ValueError, match=refusal):
        validate_native_collection(cell, tmp_path / "raw", expected_backend="sglang")


def test_publication_and_the_collection_summary_validate_with_the_backend_of_the_plan(tmp_path):
    cell = _collected(tmp_path, "decode")

    rows = aggregate_cell(_sglang_plan(cell), cell, tmp_path, expected_attempt_id="attempt")
    assert {(row["backend"], row["runtime_backend_version"]) for row in rows} == {("sglang", POD_SGLANG_VERSION)}
    assert len(rows) == len(_points("decode"))
    with pytest.raises(ValueError, match="Collector provenance has invalid runtime identity"):
        aggregate_cell(_plan(cell), cell, tmp_path, expected_attempt_id="attempt")

    summary = runner._runtime_collection_summary(cell, tmp_path / "raw", backend="sglang")
    assert summary["measured_point_count"] == len(_points("decode"))
    with pytest.raises(ValueError, match="Collector provenance has invalid runtime identity"):
        runner._runtime_collection_summary(cell, tmp_path / "raw")


def test_a_failed_sglang_cell_with_complete_files_is_recovered_as_an_sglang_collection(tmp_path):
    cell = _collected(tmp_path / "cells" / "cell-prefill")
    assert cell.cell_id == "cell-prefill"
    entry = {"status": "failed", "attempt_id": "attempt", "error_type": "RuntimeError", "error": "FAKE: copy failed"}

    recovered = runner._recover_completed_attempt(_sglang_plan(cell), cell, tmp_path, entry)

    assert recovered is not None
    assert (recovered["status"], recovered["artifact_recovery"]["original_status"]) == ("passed", "failed")
    # The same files are no vLLM collection, so a vLLM plan does not recover them.
    assert runner._recover_completed_attempt(_plan(cell), cell, tmp_path, entry) is None


def test_resuming_a_passed_sglang_cell_refreshes_its_record_from_an_sglang_collection(tmp_path, monkeypatch, caplog):
    cell = _collected(tmp_path / "artifacts" / "plan-sha" / "smoke" / "cells" / "cell-prefill")
    plan = _sglang_plan(cell)
    assert (plan.sha256[:16], cell.cell_id) == ("plan-sha", "cell-prefill")
    checkpoint_path = tmp_path / "checkpoints" / "fpm_forward_smoke.json"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_text(
        json.dumps(
            {
                "schema": runner.CHECKPOINT_SCHEMA,
                "plan_sha256": plan.sha256,
                "cells": {cell.cell_id: {"status": "passed", "attempt_id": "attempt"}},
            }
        )
    )

    def reject_rerun(*_args, **_kwargs):
        raise AssertionError("a passed cell must not run again")

    monkeypatch.setattr(runner, "_render_cell", reject_rerun)
    with caplog.at_level("WARNING", logger=runner.logger.name):
        errors = runner.run_collection(
            plan,
            generator_overrides={},
            checkpoint_dir=str(checkpoint_path.parent),
            artifact_root=str(tmp_path / "artifacts"),
            resume=True,
            retry_failed=False,
            smoke=True,
            cell_limit=1,
        )

    assert errors == []
    record = json.loads(checkpoint_path.read_text())["cells"][cell.cell_id]
    # The refresh validated the files of the cell as those of an SGLang collection and counted its points.
    assert (record["status"], record["measured_point_count"]) == ("passed", len(_points("prefill")))
    assert not [message for message in caplog.messages if "Skipping metadata refresh" in message]


# ---------------------------------------------------------------------------------------------------------------
# A whole campaign: the real cell builder, the files a pod leaves, publication and resume
# ---------------------------------------------------------------------------------------------------------------


def _fake_cluster(plan):
    """A cluster double for ``plan``: one pod per cell, which leaves the files of a complete FAKE launcher run.

    Unlike ``_run_cells`` it replaces neither the render nor the validation: a cell is rendered by the real
    builder and its files are read by the real validator. Returns the class and the list of the pods it made.
    """
    cells = {cell.cell_id: cell for cell in plan.cells}
    created = []

    class FakeCluster:
        def __init__(self, _manifest, cell_dir, **_slurm_settings):
            self.cell = cells[cell_dir.name]
            self.pod = cell_dir / "raw" / "pod-0"
            self.staged: list[str] = []
            created.append(self)

        def apply(self):
            pass

        def cleanup(self):
            pass

        def wait_ready(self, _expected_nodes):
            return ["pod-0"]

        def stage(self, _pods, files):
            self.staged = [path.name for path in files]

        def prepare_attempt(self, _pods, **keywords):
            # What the provenance program writes in a pod whose installed release is the one the plan names
            # (test_the_sglang_provenance_program_records_the_installed_sglang).
            self.pod.mkdir(parents=True, exist_ok=True)
            provenance = {
                "schema_name": "aic_fpm_collector_provenance",
                "schema_version": 1,
                "cell_id": keywords["cell_id"],
                "plan_sha256": keywords["plan_sha256"],
                "attempt_id": keywords["attempt_id"],
                "backend_version": keywords["expected_backend_version"],
                "runtime": {"backend": keywords["backend"], "backend_version": keywords["runtime_backend_version"]},
            }
            (self.pod / "collector-provenance.json").write_text(json.dumps(provenance))

        def execute(self, _pods):
            phase = self.cell.workload_kind
            (payload,) = _rank_artifacts(phase, _points(phase))
            artifact.atomic_write_json(str(self.pod / "benchmark.json"), payload)

        def collect(self, _pods, *, require_benchmark=True):
            pass

    return FakeCluster, created


def test_an_sglang_campaign_renders_with_the_real_builder_publishes_and_resumes(tmp_path, monkeypatch):
    plan = build_collection_plan(**_plan_inputs(tmp_path, options=_options(fpm_max_gpus=1)))
    prefill, decode = plan.cells
    cluster, pods = _fake_cluster(plan)
    monkeypatch.setattr(runner, "KubernetesCellRunner", cluster)
    campaign = {
        "generator_overrides": {"K8sConfig": {"k8s_image": "example/sglang:test"}},
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "artifact_root": str(tmp_path / "artifacts"),
        "retry_failed": False,
        "database_root": str(tmp_path / "systems"),
    }

    assert runner.run_collection(plan, resume=False, **campaign) == []

    # Each cell went through the branch of _render_cell to sglang_fpm_builder.py and staged the launcher.
    root = tmp_path / "artifacts" / plan.sha256[:16]
    assert [pod.cell for pod in pods] == [prefill, decode]
    commands = {}
    for pod in pods:
        run_script = (root / "cells" / pod.cell.cell_id / "run.sh").read_text()
        (line,) = [line for line in run_script.splitlines() if line.startswith("engine_command=(")]
        commands[pod.cell.cell_id] = shlex.split(line[len("engine_command=(") : -1])
        assert commands[pod.cell.cell_id][:2] == ["python3", "/tmp/fpm-bench/sglbench_launcher.py"]
        assert pod.staged == [*STAGED_FOR_EVERY_CELL, *SGLBENCH_FILES]
    # The files of both pods were validated as SGLang files and published under the data label of the plan.
    checkpoint_path = tmp_path / "checkpoints" / "fpm_forward.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    assert {record["status"] for record in checkpoint["cells"].values()} == {"passed"}
    # The two caps in the record of the prefill cell are the ones in the command the builder wrote.
    for flag, key in (
        ("--prefill-max-new-token-samples", "prefill_max_new_token_samples"),
        ("--prefill-max-kv-read-token-samples", "prefill_max_kv_read_token_samples"),
    ):
        command = commands[prefill.cell_id]
        assert command[command.index(flag) + 1] == str(checkpoint["cells"][prefill.cell_id][key])
    assert checkpoint["database"]["status"] == "passed"
    table = tmp_path / "systems" / SYSTEM / "sglang" / "0.5.23"
    metadata = json.loads((table / "fpm_forward_perf.metadata.json").read_text())
    assert (metadata["backend"], metadata["backend_version"]) == ("sglang", "0.5.23")
    assert metadata["row_count"] == len(_points("prefill")) + len(_points("decode"))

    # A resume reads the files of both cells again, with no cluster work: it recovers the attempt whose record
    # says it was interrupted and refreshes the record of the passed one.
    attempt_id = checkpoint["cells"][decode.cell_id]["attempt_id"]
    checkpoint["cells"][decode.cell_id] = {"status": "interrupted", "attempt_id": attempt_id}
    del checkpoint["cells"][prefill.cell_id]["measured_point_count"]
    del checkpoint["database"]
    checkpoint_path.write_text(json.dumps(checkpoint))
    monkeypatch.setattr(runner, "_render_cell", lambda *_a, **_k: pytest.fail("a resume ran an SGLang cell again"))

    assert runner.run_collection(plan, resume=True, **campaign) == []

    resumed = json.loads(checkpoint_path.read_text())
    assert resumed["cells"][decode.cell_id]["status"] == "passed"
    assert resumed["cells"][decode.cell_id]["artifact_recovery"]["original_status"] == "interrupted"
    assert resumed["cells"][prefill.cell_id]["measured_point_count"] == len(_points("prefill"))
    assert resumed["database"]["status"] == "passed"


# ---------------------------------------------------------------------------------------------------------------
# The checkpoint record of a cell
# ---------------------------------------------------------------------------------------------------------------


def _vllm_arguments(plan, cell, *, smoke: bool = False) -> list[str]:
    """The arguments of the vLLM render the SGLang builder translates: the same cell of the plan as a vLLM plan."""
    vllm_form = dataclasses.replace(plan, backend="vllm")
    return runner._cell_generator_overrides(vllm_form, cell, {}, smoke=smoke)["params"]["agg"]["extra_cli_args"]


def test_the_record_of_an_sglang_cell_says_what_the_cell_was_given(tmp_path):
    plan = build_collection_plan(**_plan_inputs(tmp_path))
    prefill, decode = _cells(plan)["prefill"][0], _cells(plan)["decode"][0]
    kv_read_cap = plan.options.prefill_sampling.max_kv_read_token_samples

    record = runner._configured_sampling_metadata(plan, prefill, smoke=False)
    assert record == {
        "prefill_max_new_token_samples": 8192,
        "prefill_max_kv_read_token_samples": kv_read_cap,
        "prefill_new_token_axis_source": "sglang_engine_cuda_graph_sizes",
    }
    # The vLLM render of the cell carries that token budget and that KV-read cap, and no new-token cap: the
    # builder passes the budget as the cap.
    arguments = _vllm_arguments(plan, prefill)
    assert arguments[arguments.index("--max-num-batched-tokens") + 1] == "8192"
    assert arguments[arguments.index("--prefill-max-kv-read-token-samples") + 1] == str(kv_read_cap)
    assert "--prefill-max-new-token-samples" not in arguments

    # A smoke cell keeps the two caps of its vLLM render.
    assert runner._configured_sampling_metadata(plan, prefill, smoke=True) == {
        "prefill_max_new_token_samples": 2,
        "prefill_max_kv_read_token_samples": 2,
        "prefill_new_token_axis_source": "sglang_engine_cuda_graph_sizes",
    }
    arguments = _vllm_arguments(plan, prefill, smoke=True)
    assert arguments[arguments.index("--prefill-max-new-token-samples") + 1] == "2"
    assert arguments[arguments.index("--prefill-max-kv-read-token-samples") + 1] == "2"

    for smoke in (False, True):
        assert runner._configured_sampling_metadata(plan, decode, smoke=smoke) == {"decode_radix_cache": "enabled"}

    # The records of the same cells in a vLLM plan are what they were: vLLM's capture counts and its
    # prefix-caching rule.
    vllm_plan = dataclasses.replace(plan, backend="vllm")
    assert runner._configured_sampling_metadata(vllm_plan, prefill, smoke=False) == {
        "prefill_cudagraph_capture_size_count": None,
        "prefill_requested_new_token_axis_count": None,
        "prefill_max_new_token_samples": None,
        "prefill_cudagraph_policy": "runtime",
    }
    assert runner._configured_sampling_metadata(vllm_plan, decode, smoke=False) == {"decode_prefix_caching": "disabled"}


def test_the_record_of_an_sglang_cell_with_a_points_file_names_the_file(tmp_path):
    points = tmp_path / "points.json"
    points.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "prefill": [{"batch_size": 1, "total_prefill_tokens": 128, "total_kv_read_tokens": 0}],
                "decode": [
                    {"batch_size": 1, "total_kv_read_tokens": 64},
                    {"batch_size": 2, "total_kv_read_tokens": 256},
                ],
            }
        )
    )
    plan = build_collection_plan(**_plan_inputs(tmp_path, options=_options(fpm_benchmark_points_file=str(points))))
    prefill, decode = _cells(plan)["prefill"][0], _cells(plan)["decode"][0]
    digest = plan.options.benchmark_points_sha256

    # The points are given, so no cap and no engine axis applies.
    assert runner._configured_sampling_metadata(plan, prefill, smoke=False) == {
        "benchmark_points_sha256": digest,
        "requested_point_count": 1,
    }
    assert runner._configured_sampling_metadata(plan, decode, smoke=False) == {
        "benchmark_points_sha256": digest,
        "requested_point_count": 2,
        "decode_radix_cache": "enabled",
    }
