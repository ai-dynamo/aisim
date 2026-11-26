# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host side of an SGLang self-benchmark collection: entry gates, plan, staging, provenance and validation.

No SGLang, no GPU and no cluster are involved. The tests that run cells on the in-process cluster replace the
whole render (``runner._render_cell``, see ``_run_cells``), so the cell builder (``sglang_fpm_builder.py``) is not
reached there; the campaign test renders with the real one. Every result file is a FAKE of
``test_fpm_sglbench_artifact.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import re
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

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
from collector.fpm_forward.types import ParallelTopology

from aisim_core.sdk.fpm_identity import LEGACY_EXECUTION_IDENTITY

from .test_fpm_runner import _cell, _native_payload, _plan, _write_provenance
from .test_fpm_sglbench_artifact import (
    LAUNCHER_PRODUCER,
    POD_KV_CACHE,
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

# The hash and the cell ids of the plan of _plan_inputs with backend "vllm". They were computed with a planner
# that has no SGLang check, so they move only when plan identity itself changes: first with planner.py as it was
# before SGLang was admitted, and again with the collector of main alone once that had the power option, with
# the options below (power off, as an SGLang plan has it).
VLLM_PLAN_SHA256 = "4ce0a4bad64ad2381905bae7a2f416107cd6adacfccdf0c5904d090da85c14ca"
VLLM_CELL_IDS = ["fpm-f28a6836df315638", "fpm-b80365929b21ac58", "fpm-f3124a62ef358b3e", "fpm-a7b1dd3127d7797a"]

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
    # The backend is among the arguments, as on the command line: the default of the power option depends on it.
    values = {"backend": "sglang", "fpm_max_gpus": 2, "fpm_prefill_cudagraph_policy": "runtime", **overrides}
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
        # --fpm-parallel-presets tep. The default preset of a MoE model also plans DEP and pure TP cells.
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


@pytest.mark.parametrize(
    ("arguments", "measured"),
    [
        ({"backend": "sglang"}, False),
        ({"backend": "sglang", "fpm_measure_power": None}, False),
        ({"backend": "sglang", "fpm_measure_power": False}, False),
        ({"backend": "sglang", "fpm_measure_power": True}, True),
        ({"backend": "vllm"}, True),
        ({"backend": "vllm", "fpm_measure_power": False}, False),
        ({}, True),
    ],
    ids=["sglang", "sglang-unset", "sglang-off", "sglang-on", "vllm", "vllm-off", "no-backend"],
)
def test_the_power_option_is_off_by_default_for_sglang_alone(arguments, measured):
    values = {"fpm_max_gpus": 2, "fpm_prefill_cudagraph_policy": "runtime", **arguments}

    assert FPMCollectionOptions.from_args(argparse.Namespace(**values)).measure_power is measured


def test_an_sglang_plan_measures_no_power_and_refuses_to_be_asked_for_it(tmp_path):
    plan = build_collection_plan(**_plan_inputs(tmp_path))
    # No power in the plan, so no power window in a cell and the plan identity of a plan without the option.
    assert plan.options.measure_power is False
    assert "measure_power" not in json.dumps(plan.to_dict())
    # Switching the option off is accepted and gives the plan of the default command.
    switched_off = build_collection_plan(**_plan_inputs(tmp_path, options=_options(fpm_measure_power=False)))
    assert switched_off.sha256 == plan.sha256

    _refused(
        tmp_path,
        "^--fpm-measure-power is not supported with --backend sglang: the launcher has no power window, so an "
        "SGLang plan measures no power and its rows carry no power_w; leave the option out$",
        options=_options(fpm_measure_power=True),
    )


def test_a_moe_plan_with_the_tep_preset_is_frozen_with_a_single_and_a_tep_cell(tmp_path):
    plan = build_collection_plan(**_moe_plan_inputs(tmp_path))

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

    # The saved form names the strategies, and its reader recomputes the hash and rebuilds every cell.
    saved = plan.to_dict()
    assert [(topology["strategy"], topology["tp"], topology["moe_ep"]) for topology in saved["topologies"]] == [
        ("single", 1, 1),
        ("tep", 2, 2),
    ]
    identity = saved_plan_identity(json.loads(json.dumps(saved)))
    assert (identity.backend, identity.sha256) == ("sglang", plan.sha256)
    assert identity.cells == plan.cells


def test_a_moe_plan_with_the_pure_tp_preset_holds_the_one_gpu_cell_and_a_pure_tp_cell(tmp_path):
    plan = build_collection_plan(**_moe_plan_inputs(tmp_path, options=_options(fpm_parallel_presets=["pure_tp"])))

    # One cell per phase and GPU count. The two-GPU cell has the strategy "pure_tp": every tensor-parallel rank
    # holds a slice of every expert, so moe_tp is the tensor-parallel size and there is one expert-parallel rank.
    assert [(cell.workload_kind, cell.parallel_strategy, cell.topology.to_dict()) for cell in plan.cells] == [
        (phase, strategy, {"tp": tp, "pp": 1, "dp": 1, "moe_tp": tp, "moe_ep": 1, "cp": 1})
        for phase in ("prefill", "decode")
        for strategy, tp in (("single", 1), ("pure_tp", 2))
    ]
    assert {cell.execution_identity for cell in plan.cells} == {LEGACY_EXECUTION_IDENTITY}
    # Memory admission looked at both topologies and dropped none (no estimate in this suite, _hermetic_plans).
    assert [(decision.topology, decision.disposition) for decision in plan.topology_memory_admission] == [
        (topology, "unknown") for topology in plan.topologies
    ]

    # The saved form names the strategy and both expert sizes, and its reader rebuilds every cell.
    saved = plan.to_dict()
    assert [
        (topology["strategy"], topology["tp"], topology["moe_tp"], topology["moe_ep"])
        for topology in saved["topologies"]
    ] == [("single", 1, 1, 1), ("pure_tp", 2, 2, 1)]
    identity = saved_plan_identity(json.loads(json.dumps(saved)))
    assert (identity.backend, identity.sha256) == ("sglang", plan.sha256)
    assert identity.cells == plan.cells


# What a preset plans per phase for the FAKE MoE model on one and two GPUs: (strategy, tp, dp, moe_tp, moe_ep).
SINGLE_CELL, TEP_CELL, DEP_CELL = ("single", 1, 1, 1, 1), ("tep", 2, 1, 1, 2), ("dep", 1, 2, 1, 2)
PURE_TP_CELL = ("pure_tp", 2, 1, 2, 1)


@pytest.mark.parametrize(
    ("presets", "per_phase"),
    [
        # The default preset "auto" plans DEP, pure TP and TEP for a MoE model, as it does for vLLM.
        (None, [SINGLE_CELL, DEP_CELL, PURE_TP_CELL, TEP_CELL]),
        (["auto"], [SINGLE_CELL, DEP_CELL, PURE_TP_CELL, TEP_CELL]),
        (["pure_tp", "tep", "dep"], [SINGLE_CELL, DEP_CELL, PURE_TP_CELL, TEP_CELL]),
        # The two presets by name plan what the default preset planned before pure TP was admitted.
        (["tep", "dep"], [SINGLE_CELL, DEP_CELL, TEP_CELL]),
        (["dep"], [SINGLE_CELL, DEP_CELL]),
    ],
    ids=["default", "auto", "all-three", "tep-and-dep", "dep"],
)
def test_a_moe_plan_holds_a_dep_cell_under_the_default_preset_and_under_the_dep_preset(tmp_path, presets, per_phase):
    plan = build_collection_plan(**_moe_plan_inputs(tmp_path, options=_options(fpm_parallel_presets=presets)))

    assert [
        (
            cell.workload_kind,
            cell.parallel_strategy,
            cell.topology.tp,
            cell.topology.dp,
            cell.topology.moe_tp,
            cell.topology.moe_ep,
        )
        for cell in plan.cells
    ] == [(phase, *sizes) for phase in ("prefill", "decode") for sizes in per_phase]
    # The DEP cell: one GPU and one expert-parallel rank per data-parallel rank, and nothing else above 1.
    for cell in plan.cells:
        if cell.parallel_strategy == "dep":
            assert cell.topology == ParallelTopology(tp=1, pp=1, dp=2, moe_tp=1, moe_ep=2, cp=1)
            assert cell.topology.total_gpus == 2
    assert {cell.execution_identity for cell in plan.cells} == {LEGACY_EXECUTION_IDENTITY}
    # Memory admission looked at every topology and dropped none (no estimate in this suite, _hermetic_plans).
    assert [(decision.topology, decision.disposition) for decision in plan.topology_memory_admission] == [
        (topology, "unknown") for topology in plan.topologies
    ]
    # The saved form names the strategy, and its reader recomputes the hash and rebuilds every cell.
    saved = plan.to_dict()
    assert [
        (topology["strategy"], topology["tp"], topology["dp"], topology["moe_tp"], topology["moe_ep"])
        for topology in saved["topologies"]
    ] == per_phase
    identity = saved_plan_identity(json.loads(json.dumps(saved)))
    assert (identity.backend, identity.sha256) == ("sglang", plan.sha256)
    assert identity.cells == plan.cells


@pytest.mark.parametrize(("max_gpus", "cells_per_phase"), [(8, 10)])
def test_the_default_preset_plans_a_dep_a_pure_tp_and_a_tep_cell_for_every_gpu_count_above_one(
    tmp_path, max_gpus, cells_per_phase
):
    def plan(presets, **overrides):
        options = _options(fpm_max_gpus=max_gpus, fpm_parallel_presets=presets)
        return build_collection_plan(**_moe_plan_inputs(tmp_path, options=options, **overrides))

    # The default GPU counts are the powers of two up to --fpm-max-gpus. For each count above 1, in the order of
    # the plan: (strategy, tp, dp, moe_tp, moe_ep).
    widths = [width for width in (2, 4, 8) if width <= max_gpus]
    expected = [SINGLE_CELL]
    for width in widths:
        expected += [("dep", 1, width, 1, width), ("pure_tp", width, 1, width, 1), ("tep", width, 1, 1, width)]
    by_default = _cells(plan(None))
    for phase in ("prefill", "decode"):
        assert [
            (cell.parallel_strategy, cell.topology.tp, cell.topology.dp, cell.topology.moe_tp, cell.topology.moe_ep)
            for cell in by_default[phase]
        ] == expected
        assert len(by_default[phase]) == cells_per_phase
        assert {cell.topology.total_gpus for cell in by_default[phase]} == {1, *widths}
    # The vLLM plan of the same inputs holds the same topologies in the same order.
    assert plan(None, backend="vllm").topologies == plan(None).topologies


def test_a_content_variable_of_the_deployment_inputs_is_part_of_the_plan_hash(tmp_path):
    def plan(*environment: dict):
        overrides = {"K8sConfig": {"k8s_image": "example/sglang:test", "extra_env": list(environment)}}
        return build_collection_plan(**_moe_plan_inputs(tmp_path, generator_overrides=overrides))

    name = "DYN_BENCH_PREFILL_CONTENT"
    # FAKE values: the planner does not read them, it freezes them. The launcher reads them in the pod.
    plans = [plan(), plan({"name": name, "value": "first"}), plan({"name": name, "value": "second"})]

    # K8sConfig.extra_env is the one way such a value reaches a cell, and each value is another plan.
    assert len({saved.sha256 for saved in plans}) == 3
    assert len({saved.generator_config_sha256 for saved in plans}) == 3
    # The cells are the same cells: their ids name the backend, the model, the topology and the dtypes.
    assert len({tuple(cell.cell_id for cell in saved.cells) for saved in plans}) == 1


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


# The end of the rule for a cell without DP attention, before the topology: it names the two layouts admitted.
NEITHER_LAYOUT = (
    r"either moe_tp of 1 with moe_ep equal to tp \(TEP\) or moe_ep of 1 with moe_tp equal to tp \(pure TP\)"
)


@pytest.mark.parametrize(
    ("sizes", "message"),
    [
        # With data-parallel ranks: attention TP inside a rank, an expert-parallel size that is not the number
        # of ranks, and the three fixed axes.
        ({"tp": 2, "dp": 2, "moe_ep": 4}, r"moe_ep must equal dp; .*\); got topology .*'tp': 2, .*'dp': 2"),
        ({"tp": 1, "dp": 2, "moe_ep": 1}, r"moe_ep must equal dp; .*\); got topology .*'dp': 2, .*'moe_ep': 1"),
        ({"tp": 1, "dp": 2, "moe_ep": 4}, r"moe_ep must equal dp; .*\); got topology .*'dp': 2, .*'moe_ep': 4"),
        ({"tp": 1, "dp": 2, "moe_tp": 2, "moe_ep": 2}, r"moe_ep must equal dp; .*\); got topology .*'moe_tp': 2"),
        ({"tp": 1, "dp": 2, "moe_ep": 2, "pp": 2}, r"moe_ep must equal dp; .*\); got topology .*'pp': 2"),
        ({"tp": 1, "dp": 2, "moe_ep": 2, "cp": 2}, r"moe_ep must equal dp; .*\); got topology .*'cp': 2"),
        # Without them. Plain TP of a MoE model, where neither expert size is the tensor-parallel size.
        ({"tp": 2, "moe_ep": 1}, rf"{NEITHER_LAYOUT}; got topology .*'tp': 2, .*'moe_tp': 1, 'moe_ep': 1"),
        # A mixed layout, with both expert sizes above 1: their product is the tensor-parallel size.
        ({"tp": 4, "moe_tp": 2, "moe_ep": 2}, rf"{NEITHER_LAYOUT}; got topology .*'moe_tp': 2, 'moe_ep': 2"),
        # One expert-parallel rank, and a moe_tp that is neither 1 nor the tensor-parallel size.
        ({"tp": 4, "moe_tp": 2, "moe_ep": 1}, rf"{NEITHER_LAYOUT}; got topology .*'tp': 4, .*'moe_tp': 2, 'moe_ep': 1"),
        # The two fixed axes, on the TEP shape.
        ({"pp": 2}, rf"{NEITHER_LAYOUT}; got topology .*'pp': 2"),
        ({"cp": 2}, rf"{NEITHER_LAYOUT}; got topology .*'cp': 2"),
    ],
    ids=[
        *("dp-with-tp", "dp-without-ep", "ep-above-dp", "dp-with-moe-tp", "dp-with-pp", "dp-with-cp"),
        *("no-ep", "mixed", "moe-tp-below-tp", "pp", "cp"),
    ],
)
def test_every_other_topology_of_a_moe_model_is_refused(sizes, message):
    moe = {"is_moe": True, "execution": LEGACY_EXECUTION_IDENTITY}
    # The four kinds of cell planning admits: the one-GPU cell and, on two GPUs, TEP, pure TP and DEP.
    admitted = (
        _moe_topology(tp=1),
        _moe_topology(tp=2),
        _moe_topology(tp=2, moe_tp=2, moe_ep=1),
        _moe_topology(tp=1, dp=2, moe_ep=2),
    )
    planner._require_sglang_launcher_cells(**moe, topologies=admitted)

    with pytest.raises(ValueError, match=message):
        planner._require_sglang_launcher_cells(**moe, topologies=(*admitted, _moe_topology(**sizes)))


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

    for observes in (runner._observe_runtime_memory, runner._observe_runtime_execution):
        assert (observes(plan("vllm"), cell), observes(plan("sglang"), cell)) == (True, False)


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


def test_the_pattern_finds_the_launcher():
    # The staging test compares with this list, so the list must not be empty of what a pod needs.
    assert "sglbench_launcher.py" in SGLBENCH_FILES


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
    # The unedited artifact carries the release of the FAKE pod, and the KV cache of the fp8 cell it is held
    # against, so that only the release is what a case of this test refuses.
    assert LAUNCHER_PRODUCER == {
        "sglang_version": POD_SGLANG_VERSION,
        "kv_cache_dtype": "fp8_e4m3",
        "kv_cache_stored_dtype": "torch.float8_e4m3fn",
    }
    assert POD_SGLANG_VERSION == "0.5.23"
    cell = _collected(tmp_path, installed=installed, edit=edit)

    with pytest.raises(ValueError, match=refusal):
        validate_native_collection(cell, tmp_path / "raw", expected_backend="sglang")


# Config keys by which a checkpoint declares an NVFP4 KV cache of its own, as AISim reads them. Typed in.
NVFP4_KV_CACHE_DECLARATIONS = {
    "hf-quant-config": {"hf_quant_config": {"quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": "NVFP4"}}},
    "quantization-config": {"quantization_config": {"kv_cache_quant_algo": "nvfp4"}},
}
_NO_NVFP4_IN_A_PLAN = re.escape("unsupported FPM KV-cache dtype 'nvfp4'")


@pytest.mark.parametrize("name", sorted(NVFP4_KV_CACHE_DECLARATIONS))
def test_no_plan_carries_the_kv_cache_dtype_nvfp4_that_the_validator_has_a_rule_for(tmp_path, name):
    # The validator holds an nvfp4 cell against the FP4 dtype and the argument nvfp4. No plan hands it such a
    # cell: planning refuses the checkpoint AISim infers the dtype for, and the option cannot name it.
    document = {**DENSE_CONFIG, **NVFP4_KV_CACHE_DECLARATIONS[name]}
    # A folder of its own: ``_plan_inputs`` writes the plain config under the same file name.
    folder = tmp_path / "declares-an-nvfp4-kv-cache"
    folder.mkdir()
    inputs = _plan_inputs(tmp_path, model_config_path=_config(folder, document))

    with pytest.raises(ValueError, match=_NO_NVFP4_IN_A_PLAN):
        build_collection_plan(**inputs)
    # The plain checkpoint, with the option asked to name the dtype.
    with pytest.raises(ValueError, match=_NO_NVFP4_IN_A_PLAN):
        build_collection_plan(**_plan_inputs(tmp_path, options=_options(fpm_kv_cache_dtypes=["nvfp4"])))

    # With a dtype the option can name, the same checkpoint is planned, and its cells carry that dtype.
    for label in ("bfloat16", "fp8"):
        plan = build_collection_plan(**{**inputs, "options": _options(fpm_kv_cache_dtypes=[label])})
        assert plan.capability.dtype.native_kv_cache_dtype == "nvfp4"
        assert {cell.kv_cache_dtype for cell in plan.cells} == {label}


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
            # What the launcher records of the layout in a pod of a dense cell (sglbench_launcher.py,
            # _start_up_checks and _resolved_settings): SGLang's TP size, one attention-DP rank and no block of
            # expert layers. The host holds the three against the cell of a dense plan. It holds the dtype of
            # the KV pool against the KV-cache label of the cell, so the FAKE pod records the one of its label.
            producer = {
                **LAUNCHER_PRODUCER,
                **POD_KV_CACHE[self.cell.kv_cache_dtype],
                "tp_size": self.cell.topology.tp,
                "attn_dp_size": 1,
                "moe": None,
            }
            (payload,) = _rank_artifacts(phase, _points(phase), producer=producer)
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
    record = runner._configured_sampling_metadata(plan, decode, smoke=False)
    assert (record["benchmark_points_sha256"], record["requested_point_count"]) == (digest, 2)
