# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of ``sglang_fpm_builder.py``, the cell builder of the SGLang self-benchmark. No SGLang, no GPU, no pod.

Every cell is rendered by the real ``runner._render_cell``, so by the real Generator, from a real collection
plan: one dense model on two GPUs, or one MoE model with the TEP or the DEP preset on one and two GPUs, or with
the pure TP and the TEP preset on two. The plan is built as a vLLM plan and turned into an SGLang plan with
``dataclasses.replace``: the tests therefore need neither the planner's admission of ``sglang`` nor the runner's
branch that calls the builder.

What is FAKE is named so: the models (configs typed in here, no checkpoint), the image name, and the
``python3`` that one test puts first on ``PATH`` to see how ``run.sh`` starts its command. Nothing of the
launcher runs in a pod here; what the tests show is the text of the cell's files and that the launcher's own
flag reader takes the command.

The tests that read SGLang's source need a local checkout of SGLang for each release the launcher accepts.
``AISIM_SGLBENCH_SGLANG_SOURCES`` names a folder that holds them as ``v<version>``, the variable and the layout
of ``test_fpm_sglbench_preflight.py``; without it those tests are skipped.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

import collector.fpm_forward.runner as fpm_runner
from collector.fpm_forward import planner
from collector.fpm_forward import sglang_fpm_builder as builder
from collector.fpm_forward.config import FPMCollectionOptions
from collector.fpm_forward.planner import BackendPolicy, FPMCell, FPMCollectionPlan, build_collection_plan
from collector.fpm_forward.runner import _render_cell
from collector.fpm_forward.types import ParallelTopology

AISIM = Path(__file__).resolve().parents[3]
RUNTIME = AISIM / "collector/fpm_forward/runtime"
# In the pod the runtime files lie flat in one directory and import each other by bare name. The directory goes
# to the END of sys.path, so that it shadows nothing.
if str(RUNTIME) not in sys.path:
    sys.path.append(str(RUNTIME))

import sglbench_compat as compat
import sglbench_grid as grid
import sglbench_launcher as launcher

SOURCES_ENV = "AISIM_SGLBENCH_SGLANG_SOURCES"

MODEL = "private-org/dense-model"
FAKE_MODEL_CONFIG = {
    "architectures": ["PrivateDenseForCausalLM"],
    "hidden_size": 3072,
    "intermediate_size": 8192,
    "num_hidden_layers": 32,
    "num_attention_heads": 24,
    "num_key_value_heads": 8,
    "vocab_size": 200064,
}
MOE_MODEL = "private-org/moe-model"
FAKE_MOE_MODEL_CONFIG = {
    "architectures": ["PrivateMoeForCausalLM"],
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "vocab_size": 151936,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 768,
}
FAKE_IMAGE = "example.invalid/sglang:test"
DEPLOYMENT = {"K8sConfig": {"k8s_image": FAKE_IMAGE}}

COMMAND_LINE_START = "engine_command=("
# The expected commands are typed as text; no token of them holds a blank, so ``split`` gives the token list.
LAUNCHER = "python3 /tmp/fpm-bench/sglbench_launcher.py"
# What a cell of the plan below passes to SGLang's parser before the benchmark flags.
ENGINE = f"--model-path {MODEL} --served-model-name {MODEL} --tp-size 2 --kv-cache-dtype fp8_e4m3"
BENCHMARK = "--benchmark-timeout 10200 --benchmark-mode {phase} --benchmark-warmup-iterations 5"
RESULT_FILES = "--dump-config-to /results/resolved-config-node0.json --benchmark-output-path /results/benchmark.json"
TOKEN_BUDGET = "--chunked-prefill-size 8192 --max-prefill-tokens 8192"
PREFILL_COMMAND = (
    f"{LAUNCHER} {ENGINE} {BENCHMARK.format(phase='prefill')} --prefill-max-kv-read-token-samples 29"
    f" {TOKEN_BUDGET} {RESULT_FILES} --prefill-max-new-token-samples 8192"
).split()
DECODE_COMMAND = f"{LAUNCHER} {ENGINE} {BENCHMARK.format(phase='decode')} {RESULT_FILES}".split()

# The flags the builder writes, by the parser that takes them.
LAUNCHER_FLAGS = {
    "--benchmark-mode",
    "--benchmark-points-file",
    "--benchmark-warmup-iterations",
    "--benchmark-output-path",
    "--benchmark-timeout",
    "--dump-config-to",
    "--prefill-max-new-token-samples",
    "--prefill-max-kv-read-token-samples",
    "--prefix-max-batch-size-samples",
    "--decode-max-kv-read-token-samples",
    "--decode-max-batch-size-samples",
}
SGLANG_FLAGS = {
    "--model-path",
    "--served-model-name",
    "--tp-size",
    "--attn-dp-size",
    "--ep-size",
    "--kv-cache-dtype",
    "--mem-fraction-static",
    "--context-length",
    "--chunked-prefill-size",
    "--max-prefill-tokens",
    "--max-running-requests",
}
# What the launcher refuses when it is switched on or is not 1 (sglbench_launcher.py ``_RESOLVED_CHECKS``; the
# overlap schedule is refused from the recorder headers, sglbench_driver.py), with SGLang's aliases.
REFUSED_BY_THE_LAUNCHER = {
    "--disable-overlap-schedule",
    "--disable-radix-cache",
    "--enable-mixed-chunk",
    "--skip-tokenizer-init",
    "--dp-size",
    "--data-parallel-size",
    "--pp-size",
    "--pipeline-parallel-size",
}
# The other spellings of the attention-DP size, which the builder does not write: SGLang's alias of
# --attn-dp-size, and the deprecated switch that makes an attention-DP size of --dp-size.
OTHER_ATTENTION_DP_SPELLINGS = {"--attention-data-parallel-size", "--enable-dp-attention"}


# ---------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------


def _sglang_plan(tmp_path: Path, monkeypatch, model: str, config: dict, **options) -> FPMCollectionPlan:
    """A vLLM plan of a FAKE model on one node, turned into an SGLang plan.

    The defaults are what an SGLang plan has: the runtime graph policy, under which the vLLM command holds no
    capture list. ``options`` are attributes of the parsed ``--fpm-*`` arguments.
    """
    monkeypatch.setattr(planner, "_git_revision", lambda: "test-revision")

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("a unit test has no checkpoint")

    monkeypatch.setattr("collector.fpm_forward.memory_admission.KVCacheEstimator.from_request", unavailable)
    config_dir = tmp_path / f"model-config-{config['architectures'][0]}"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.json").write_text(json.dumps(config))
    arguments = {
        # The backend is among the arguments, as on the command line: the power option is off by default for it.
        "backend": "sglang",
        "fpm_max_gpus": 2,
        "fpm_kv_cache_dtypes": ["fp8"],
        "fpm_prefill_cudagraph_policy": "runtime",
        **options,
    }
    vllm_plan = build_collection_plan(
        backend="vllm",
        model_path=model,
        model_architecture=config["architectures"][0],
        system="h200_sxm",
        selected_ops={"attention_context", "attention_generation"},
        has_model_cases=False,
        options=FPMCollectionOptions.from_args(argparse.Namespace(**arguments)),
        collector_config={"aic_database_version": "0.5.22"},
        model_config_path=str(config_dir),
    )
    return dataclasses.replace(vllm_plan, backend="sglang")


def _plan(tmp_path: Path, monkeypatch, **options) -> FPMCollectionPlan:
    """An SGLang plan of the FAKE dense model on two GPUs of one node: one prefill and one decode cell at TP 2."""
    plan = _sglang_plan(tmp_path, monkeypatch, MODEL, FAKE_MODEL_CONFIG, **{"fpm_gpu_counts": [2], **options})
    assert [(cell.workload_kind, cell.parallel_strategy, cell.topology.tp) for cell in plan.cells] == [
        ("prefill", "tp", 2),
        ("decode", "tp", 2),
    ]
    return plan


def _moe_plan(tmp_path: Path, monkeypatch, **options) -> FPMCollectionPlan:
    """An SGLang plan of the FAKE MoE model with the TEP preset: per phase a one-GPU cell and a TEP cell at TP 2."""
    presets = {"fpm_gpu_counts": [1, 2], "fpm_parallel_presets": ["tep"]}
    plan = _sglang_plan(tmp_path, monkeypatch, MOE_MODEL, FAKE_MOE_MODEL_CONFIG, **{**presets, **options})
    assert [
        (cell.workload_kind, cell.parallel_strategy, cell.topology.tp, cell.topology.moe_ep) for cell in plan.cells
    ] == [
        ("prefill", "single", 1, 1),
        ("prefill", "tep", 2, 2),
        ("decode", "single", 1, 1),
        ("decode", "tep", 2, 2),
    ]
    return plan


def _dep_plan(tmp_path: Path, monkeypatch, **options) -> FPMCollectionPlan:
    """An SGLang plan of the FAKE MoE model with the DEP preset: per phase a one-GPU and a two-rank DEP cell."""
    presets = {"fpm_gpu_counts": [1, 2], "fpm_parallel_presets": ["dep"]}
    plan = _sglang_plan(tmp_path, monkeypatch, MOE_MODEL, FAKE_MOE_MODEL_CONFIG, **{**presets, **options})
    assert [
        (cell.workload_kind, cell.parallel_strategy, cell.topology.tp, cell.topology.dp, cell.topology.moe_ep)
        for cell in plan.cells
    ] == [
        ("prefill", "single", 1, 1, 1),
        ("prefill", "dep", 1, 2, 2),
        ("decode", "single", 1, 1, 1),
        ("decode", "dep", 1, 2, 2),
    ]
    return plan


def _cell(plan: FPMCollectionPlan, phase: str, strategy: str | None = None) -> FPMCell:
    """The cell of ``phase``; ``strategy`` picks one where the plan has several per phase."""
    return next(
        cell
        for cell in plan.cells
        if cell.workload_kind == phase and (strategy is None or cell.parallel_strategy == strategy)
    )


def _render(plan, cell, cell_dir: Path, overrides=None, *, smoke: bool = False, render=_render_cell) -> dict:
    """Render ``cell`` as an SGLang cell with the builder. ``render`` stands for ``runner._render_cell``."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    return builder.render_cell(
        plan, cell, cell_dir, DEPLOYMENT if overrides is None else overrides, smoke=smoke, render_vllm_cell=render
    )


def _render_vllm(plan, cell, cell_dir: Path) -> dict:
    """The vLLM render of the same cell, to compare with."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    return _render_cell(dataclasses.replace(plan, backend="vllm"), cell, cell_dir, DEPLOYMENT)


def _edited_render(edit):
    """The real render followed by ``edit(cell_dir)``: a render the Generator does not write today."""

    def render(plan, cell, cell_dir, generator_overrides, *, smoke=False):
        artifacts = _render_cell(plan, cell, cell_dir, generator_overrides, smoke=smoke)
        edit(cell_dir)
        return artifacts

    return render


def _edit_file(name: str, change):
    def edit(cell_dir: Path) -> None:
        path = cell_dir / name
        path.write_text(change(path.read_text()))

    return edit


def _command(cell_dir: Path) -> list[str]:
    """The tokens of the one command line of the cell's ``run.sh``."""
    lines = [line for line in (cell_dir / "run.sh").read_text().splitlines() if line.startswith(COMMAND_LINE_START)]
    assert len(lines) == 1
    assert lines[0].endswith(")")
    return shlex.split(lines[0][len(COMMAND_LINE_START) : -1])


def _files(cell_dir: Path) -> dict[str, str]:
    return {path.name: path.read_text() for path in sorted(cell_dir.iterdir())}


def _policy(*extra_cli_args: str, **deployment) -> BackendPolicy:
    overrides: dict = {"params": {"agg": {"extra_cli_args": list(extra_cli_args)}}}
    if deployment:
        overrides["K8sConfig"] = deployment
    return BackendPolicy("custom", overrides, {})


def _with_environment(*entries: tuple[str, str]) -> dict:
    return {
        "K8sConfig": {"k8s_image": FAKE_IMAGE, "extra_env": [{"name": name, "value": value} for name, value in entries]}
    }


def _vllm(*flags: str) -> list[str]:
    return ["python3", "-m", "dynamo.vllm", *flags]


def _written_flags() -> set[str]:
    """Every flag a row of the translation table can write."""
    return {name for row in builder._TRANSLATION.values() for name in row.writes}


def _refusal(text: str):
    return pytest.raises(ValueError, match=re.escape(f"SGLang self-benchmark cell refused: {text}"))


# ---------------------------------------------------------------------------------------------------------------
# The cell of a formal run
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("phase", "expected"), [("prefill", PREFILL_COMMAND), ("decode", DECODE_COMMAND)])
def test_dense_tp_cell_command(tmp_path, monkeypatch, phase, expected):
    plan = _plan(tmp_path, monkeypatch)

    _render(plan, _cell(plan, phase), tmp_path / "cell")

    assert _command(tmp_path / "cell") == expected


MOE_ENGINE = f"--model-path {MOE_MODEL} --served-model-name {MOE_MODEL} --tp-size {{sizes}} --kv-cache-dtype fp8_e4m3"
MOE_PREFILL = (
    f"{LAUNCHER} {MOE_ENGINE} {BENCHMARK.format(phase='prefill')} --prefill-max-kv-read-token-samples 29"
    f" {TOKEN_BUDGET} {RESULT_FILES} --prefill-max-new-token-samples 8192"
)
MOE_DECODE = f"{LAUNCHER} {MOE_ENGINE} {BENCHMARK.format(phase='decode')} {RESULT_FILES}"


@pytest.mark.parametrize(
    ("phase", "strategy", "expected"),
    [
        # The TEP cell: one expert-parallel rank per tensor-parallel rank, written right after --tp-size.
        ("prefill", "tep", MOE_PREFILL.format(sizes="2 --ep-size 2")),
        ("decode", "tep", MOE_DECODE.format(sizes="2 --ep-size 2")),
        # The one-GPU cell of the same preset: no size is written, so SGLang's default of 1 applies.
        ("prefill", "single", MOE_PREFILL.format(sizes="1")),
        ("decode", "single", MOE_DECODE.format(sizes="1")),
    ],
    ids=["prefill-tep", "decode-tep", "prefill-single", "decode-single"],
)
def test_moe_cell_command(tmp_path, monkeypatch, phase, strategy, expected):
    plan = _moe_plan(tmp_path, monkeypatch)
    cell = _cell(plan, phase, strategy)
    _render_vllm(plan, cell, tmp_path / "vllm")

    _render(plan, cell, tmp_path / "sglang")

    command = _command(tmp_path / "sglang")
    assert command == expected.split()
    # What was translated is the Generator's own vLLM command of the cell: the three parallel sizes, and then
    # the expert-parallel switch exactly when the cell has more than one expert-parallel rank.
    vllm = _command(tmp_path / "vllm")
    sizes = f"--tensor-parallel-size {cell.topology.tp} --pipeline-parallel-size 1 --data-parallel-size 1".split()
    start = vllm.index("--tensor-parallel-size")
    assert vllm[start : start + 6] == sizes
    assert (vllm[start + 6] == "--enable-expert-parallel") is (strategy == "tep")
    assert ("--enable-expert-parallel" in vllm) is (strategy == "tep")
    # The launcher's own reader of flags takes the command and hands both sizes to SGLang's parser.
    environment = {"DYN_BENCH_KV_WARMUP": "on", "DYN_BENCH_PREFILL_REAL_SEED": "on"}
    options = launcher._read_options(command[2:], environment)
    assert (options.rejection, options.output_problem, options.mode) == (None, None, phase)
    engine_arguments = MOE_ENGINE.format(sizes="2 --ep-size 2" if strategy == "tep" else "1").split()
    assert options.sglang_argv[: len(engine_arguments)] == engine_arguments


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_pure_tp_cell_command_has_the_tensor_parallel_size_and_no_expert_parallel_size(tmp_path, monkeypatch, phase):
    # The pure TP cell and the TEP cell of the same two GPUs, in one plan.
    plan = _sglang_plan(
        tmp_path,
        monkeypatch,
        MOE_MODEL,
        FAKE_MOE_MODEL_CONFIG,
        fpm_gpu_counts=[2],
        fpm_parallel_presets=["pure_tp", "tep"],
    )
    cell = _cell(plan, phase, "pure_tp")
    assert cell.topology == ParallelTopology(tp=2, pp=1, dp=1, moe_tp=2, moe_ep=1, cp=1)

    _render(plan, cell, tmp_path / "sglang")

    # The pure TP cell: the tensor-parallel size above 1 and no expert-parallel size after it, so SGLang's
    # default of one expert-parallel rank applies and every rank holds a slice of every expert. No attention-DP
    # size is written either.
    command = _command(tmp_path / "sglang")
    assert command == (MOE_PREFILL if phase == "prefill" else MOE_DECODE).format(sizes="2").split()
    assert command[command.index("--tp-size") + 1] == "2"
    assert "--ep-size" not in command and "--attn-dp-size" not in command
    # The launcher's own reader of flags takes the command and hands the one size to SGLang's parser.
    environment = {"DYN_BENCH_KV_WARMUP": "on", "DYN_BENCH_PREFILL_REAL_SEED": "on"}
    options = launcher._read_options(command[2:], environment)
    assert (options.rejection, options.output_problem, options.mode) == (None, None, phase)
    engine_arguments = MOE_ENGINE.format(sizes="2").split()
    assert options.sglang_argv[: len(engine_arguments)] == engine_arguments


# The DEP cell of two GPUs. SGLang's --tp-size and --ep-size count the ranks of both data-parallel ranks, and
# --attn-dp-size, in the place of vLLM's --data-parallel-size, makes the two ranks attention-DP ranks. SGLang
# divides the chunk size by the number of those ranks, so the cell writes twice the token budget for it and
# each rank gets the budget; the other token bound and the cap of the new-token axis stay the budget.
DEP_SIZES = "2 --ep-size 2 --attn-dp-size 2"
DEP_TOKEN_BUDGET = "--chunked-prefill-size 16384 --max-prefill-tokens 8192"
assert MOE_PREFILL.count(TOKEN_BUDGET) == 1
DEP_PREFILL_COMMAND = MOE_PREFILL.replace(TOKEN_BUDGET, DEP_TOKEN_BUDGET).format(sizes=DEP_SIZES).split()
DEP_DECODE_COMMAND = MOE_DECODE.format(sizes=DEP_SIZES).split()
# What the Generator's vLLM command of that cell says about its ranks.
VLLM_DEP_SIZES = [
    *("--tensor-parallel-size", "1", "--pipeline-parallel-size", "1", "--data-parallel-size", "2"),
    "--enable-expert-parallel",
]


@pytest.mark.parametrize(("phase", "expected"), [("prefill", DEP_PREFILL_COMMAND), ("decode", DEP_DECODE_COMMAND)])
def test_dep_cell_command(tmp_path, monkeypatch, phase, expected):
    plan = _dep_plan(tmp_path, monkeypatch)
    cell = _cell(plan, phase, "dep")
    assert cell.topology == ParallelTopology(tp=1, pp=1, dp=2, moe_tp=1, moe_ep=2, cp=1)
    _render_vllm(plan, cell, tmp_path / "vllm")

    _render(plan, cell, tmp_path / "sglang")

    command = _command(tmp_path / "sglang")
    assert command == expected
    # What was translated is the Generator's own vLLM command of the cell: one tensor-parallel rank in each of
    # two data-parallel ranks, the expert-parallel switch, and the start-up of the data-parallel engine cores.
    vllm = _command(tmp_path / "vllm")
    start = vllm.index("--tensor-parallel-size")
    assert vllm[start : start + len(VLLM_DEP_SIZES)] == VLLM_DEP_SIZES
    assert vllm[vllm.index("--data-parallel-backend") + 1] == "mp"
    # No data-parallel name of vLLM reaches SGLang's parser, where --data-parallel-size counts whole replicas,
    # and the deprecated spelling of DP attention is not written either.
    assert not {"--data-parallel-size", "--data-parallel-backend", "--dp-size", "--enable-dp-attention"} & set(command)
    # The token budget of the vLLM cell, which each of its two data-parallel ranks gets whole: SGLang halves the
    # chunk size for a rank, so twice the budget is written for it. SGLang does not divide the other bound.
    if phase == "prefill":
        budget = vllm[vllm.index("--max-num-batched-tokens") + 1]
        assert budget == "8192"
        assert command[command.index("--chunked-prefill-size") + 1] == "16384"
        assert command[command.index("--max-prefill-tokens") + 1] == budget
        assert command[command.index("--prefill-max-new-token-samples") + 1] == budget
    # The launcher's own reader of flags takes the command and hands the three sizes to SGLang's parser.
    environment = {"DYN_BENCH_KV_WARMUP": "on", "DYN_BENCH_PREFILL_REAL_SEED": "on"}
    options = launcher._read_options(command[2:], environment)
    assert (options.rejection, options.output_problem, options.mode) == (None, None, phase)
    engine_arguments = MOE_ENGINE.format(sizes=DEP_SIZES).split()
    assert options.sglang_argv[: len(engine_arguments)] == engine_arguments


def test_dep_cell_asks_the_gate_for_two_rank_files_and_the_pod_for_two_gpus(tmp_path, monkeypatch):
    plan = _dep_plan(tmp_path, monkeypatch)
    cell = _cell(plan, "decode", "dep")
    _render_vllm(plan, cell, tmp_path / "vllm")

    _render(plan, cell, tmp_path / "sglang")

    vllm, sglang = _files(tmp_path / "vllm"), _files(tmp_path / "sglang")
    # The environment of the gate is the vLLM render's: one node, and two data-parallel ranks on it. In its
    # mode without etcd the gate waits for one rank file per local rank (runtime/fpm_exec.sh), which is what
    # the launcher writes for a cell with two attention-DP ranks.
    assert sglang["fpm_env.sh"] == vllm["fpm_env.sh"]
    exports = sglang["fpm_env.sh"].splitlines()
    for line in ("export FPM_NODE_COUNT=1", "export FPM_DATA_PARALLEL_SIZE=2", "export FPM_LOCAL_DATA_PARALLEL_SIZE=2"):
        assert line in exports
    # The pod asks for one GPU per rank.
    pod = yaml.safe_load(sglang["k8s_deploy.yaml"])
    (container,) = pod["spec"]["containers"]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "2"
    assert "workingDir" not in container


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_only_the_command_line_of_run_script_changes(tmp_path, monkeypatch, phase):
    plan = _plan(tmp_path, monkeypatch)
    cell = _cell(plan, phase)
    _render_vllm(plan, cell, tmp_path / "vllm")

    _render(plan, cell, tmp_path / "sglang")

    vllm, sglang = _files(tmp_path / "vllm"), _files(tmp_path / "sglang")
    assert (
        list(sglang)
        == list(vllm)
        == [
            "collector-runtime-env.sh",
            "fpm_env.sh",
            "generator-request.json",
            "k8s_deploy.yaml",
            "run.sh",
        ]
    )
    vllm_lines, sglang_lines = vllm["run.sh"].split("\n"), sglang["run.sh"].split("\n")
    changed = [index for index, pair in enumerate(zip(vllm_lines, sglang_lines, strict=True)) if pair[0] != pair[1]]
    assert len(changed) == 1
    assert vllm_lines[changed[0]].startswith(f"{COMMAND_LINE_START}python3 -m dynamo.vllm ")
    assert sglang_lines[changed[0]] == f"{COMMAND_LINE_START}{shlex.join(_command(tmp_path / 'sglang'))})"
    # The gate signals the process group of run.sh, and the launcher must be started by path.
    assert sglang["run.sh"].rstrip("\n").splitlines()[-1] == (
        "exec python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \"${engine_command[@]}\""
    )
    # The hardware environment of the Generator stays.
    assert "export NCCL_CUMEM_ENABLE=1" in sglang_lines
    for name in ("fpm_env.sh", "generator-request.json"):
        assert (tmp_path / "sglang" / name).read_bytes() == (tmp_path / "vllm" / name).read_bytes()
    for name in ("run.sh", "fpm_env.sh"):
        subprocess.run(["bash", "-n", str(tmp_path / "sglang" / name)], check=True)
        assert (tmp_path / "sglang" / name).stat().st_mode == (tmp_path / "vllm" / name).stat().st_mode


def test_manifest_loses_only_the_working_dir(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    cell = _cell(plan, "prefill")
    _render_vllm(plan, cell, tmp_path / "vllm")

    _render(plan, cell, tmp_path / "sglang")

    vllm_lines = (tmp_path / "vllm" / "k8s_deploy.yaml").read_text().splitlines(keepends=True)
    sglang_text = (tmp_path / "sglang" / "k8s_deploy.yaml").read_text()
    assert [line.strip() for line in vllm_lines if "workingDir" in line] == [
        "workingDir: /workspace/examples/backends/vllm"
    ]
    assert sglang_text == "".join(line for line in vllm_lines if "workingDir" not in line)
    pod = yaml.safe_load(sglang_text)
    assert pod["kind"] == "Pod"
    (container,) = pod["spec"]["containers"]
    assert "workingDir" not in container
    assert container["image"] == FAKE_IMAGE
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "2"
    # The render got the SGLang plan's own hash, and the runner selects the pods of a cell by its label.
    labels = pod["metadata"]["labels"]
    assert labels["aiconfigurator.nvidia.com/plan"] == plan.sha256[:16]
    assert labels["aiconfigurator.nvidia.com/cell"] == cell.cell_id


def test_runtime_environment_gets_the_gate_switch(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    cell = _cell(plan, "decode")
    _render_vllm(plan, cell, tmp_path / "vllm")

    _render(plan, cell, tmp_path / "sglang")

    vllm_text = (tmp_path / "vllm" / "collector-runtime-env.sh").read_text()
    sglang_text = (tmp_path / "sglang" / "collector-runtime-env.sh").read_text()
    assert sglang_text == vllm_text + "export AISIMULATE_SGLBENCH_GATE=1\n"
    # The gate still asks for the readiness budget before it looks at the switch.
    assert sglang_text.splitlines() == [
        "# Generated Collector startup environment; engine settings remain in run.sh.",
        "export FPM_READINESS_TIMEOUT_SECONDS=900",
        "export DYN_BENCH_KV_WARMUP=on",
        "export DYN_BENCH_PREFILL_REAL_SEED=on",
        "export AISIMULATE_SGLBENCH_GATE=1",
    ]
    # The switch travels in the collector's file only: fpm_env.sh exports the contract's names and no other.
    assert "AISIMULATE_SGLBENCH_GATE" not in (tmp_path / "sglang" / "fpm_env.sh").read_text()
    assert "AISIMULATE_SGLBENCH_GATE" not in (tmp_path / "sglang" / "run.sh").read_text()
    # Sourced in the gate's order, the two files give the switch and the one node its mode without etcd asks for.
    sourced = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1/fpm_env.sh"; source "$1/collector-runtime-env.sh"; '
            'printf "%s\\n" "$AISIMULATE_SGLBENCH_GATE" "$FPM_NODE_COUNT" "$FPM_NODE_RANK" '
            '"$FPM_READINESS_TIMEOUT_SECONDS"',
            "bash",
            str(tmp_path / "sglang"),
        ],
        env={"PATH": os.environ["PATH"]},
        check=True,
        capture_output=True,
        text=True,
    )
    assert sourced.stdout.splitlines() == ["1", "1", "0", "900"]


def test_run_script_starts_the_launcher_by_path_in_its_own_session(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    cell = _cell(plan, "prefill")
    _render(plan, cell, tmp_path / "cell")
    # A FAKE python3: it runs the ``-c`` program of the last line of run.sh with the real interpreter. That
    # program calls setsid and starts ``python3`` again, which is this FAKE once more, now with the launcher's
    # path and flags. It writes them down and ends; nothing of the launcher runs.
    record = tmp_path / "started.json"
    recorder = tmp_path / "fake_python3.py"
    recorder.write_text(
        "import json, os, sys\n"
        "if sys.argv[1] == '-c':\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        "started = {'argv': sys.argv[1:], 'session_leader': os.getsid(0) == os.getpid(), 'environ': dict(os.environ)}\n"
        f"with open({str(record)!r}, 'w') as handle:\n"
        "    json.dump(started, handle)\n"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python3 = bin_dir / "python3"
    fake_python3.write_text(
        f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} {shlex.quote(str(recorder))} "$@"\n'
    )
    fake_python3.chmod(0o755)

    completed = subprocess.run(
        ["bash", str(tmp_path / "cell" / "run.sh")],
        env={"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    started = json.loads(record.read_text())
    assert started["argv"] == PREFILL_COMMAND[1:]
    assert started["session_leader"] is True
    environ = started["environ"]
    assert environ["DYN_FPM_WORKER_ID"] == f"{cell.cell_id}-node0"
    assert environ["DYN_FPM_BENCHMARK_OUTPUT_PATH"] == "/results/benchmark.json"
    assert (environ["DYN_BENCH_KV_WARMUP"], environ["DYN_BENCH_PREFILL_REAL_SEED"]) == ("on", "on")
    assert (environ["FPM_NODE_COUNT"], environ["FPM_NODE_RANK"]) == ("1", "0")

    # The launcher's own reader of flags and variables takes them, and SGLang's parser gets the rest.
    options = launcher._read_options(started["argv"][1:], environ)
    assert (options.rejection, options.output_problem) == (None, None)
    assert (options.mode, options.timeout, options.warmup_iterations) == ("prefill", 10200, 5)
    assert options.caps == grid.SampleCaps(prefill_max_new_token_samples=8192, prefill_max_kv_read_token_samples=29)
    assert (options.output_path, options.dump_config_to, options.points_file) == (
        "/results/benchmark.json",
        "/results/resolved-config-node0.json",
        None,
    )
    assert options.sglang_argv == f"{ENGINE} {TOKEN_BUDGET} {launcher.STREAMING_FLAG}".split()


def test_gate_deadline_leaves_the_launcher_more_than_one_engine_call(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    _render(plan, _cell(plan, "prefill"), tmp_path / "cell")

    command = _command(tmp_path / "cell")
    launcher_deadline = int(command[command.index("--benchmark-timeout") + 1])
    (gate_deadline,) = re.findall(
        r"^export FPM_WAIT_TIMEOUT_SECONDS=(\d+)$", (tmp_path / "cell" / "fpm_env.sh").read_text(), flags=re.MULTILINE
    )
    assert (launcher_deadline, int(gate_deadline)) == (10200, 11400)
    assert builder._DEADLINE_LEAD_SECONDS == 600
    # What remains holds the point in flight and the launcher's last wait, with time for the shutdown and the write.
    assert (
        int(gate_deadline) - launcher_deadline
        >= launcher.ENGINE_CALL_TIMEOUT_SECONDS + launcher.FINAL_WAIT_SECONDS + 60
    )


# ---------------------------------------------------------------------------------------------------------------
# Other cells
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("phase", "caps"),
    [
        (
            "prefill",
            "--prefill-max-new-token-samples 2 --prefill-max-kv-read-token-samples 2 --prefix-max-batch-size-samples 1",
        ),
        ("decode", "--decode-max-kv-read-token-samples 2 --decode-max-batch-size-samples 2"),
    ],
)
def test_smoke_cell_keeps_its_own_caps(tmp_path, monkeypatch, phase, caps):
    plan = _plan(tmp_path, monkeypatch)

    _render(plan, _cell(plan, phase), tmp_path / "cell", smoke=True)

    # A smoke prefill cell has no token budget and no need of one: its cap of 2 is already there.
    expected = f"{LAUNCHER} {ENGINE} {BENCHMARK.format(phase=phase)} {caps} {RESULT_FILES}"
    assert _command(tmp_path / "cell") == expected.split()


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_points_file_cell_gets_no_new_token_cap(tmp_path, monkeypatch, phase):
    points = tmp_path / "points.json"
    points.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "prefill": [{"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 0}],
                "decode": [{"batch_size": 1, "total_kv_read_tokens": 64}],
            }
        )
    )
    plan = _plan(tmp_path, monkeypatch, fpm_benchmark_points_file=str(points))

    _render(plan, _cell(plan, phase), tmp_path / "cell")

    command = _command(tmp_path / "cell")
    # The staged file is the runner's; the launcher only reads it. A manifest replaces the generated grid, so
    # no cap applies.
    assert command[command.index("--benchmark-points-file") + 1] == "/tmp/fpm-bench/benchmark-points.json"
    assert "--prefill-max-new-token-samples" not in command
    if phase == "prefill":
        assert command[command.index("--chunked-prefill-size") + 1] == "8192"


def test_context_length_memory_fraction_and_decode_batch(tmp_path, monkeypatch):
    plan = _plan(
        tmp_path,
        monkeypatch,
        fpm_max_model_len=32768,
        fpm_gpu_memory_utilization=0.85,
        fpm_max_decode_batch_size=64,
    )

    _render(plan, _cell(plan, "prefill"), tmp_path / "prefill")
    _render(plan, _cell(plan, "decode"), tmp_path / "decode")

    engine = ENGINE.replace("--kv-cache-dtype", "--mem-fraction-static 0.85 --kv-cache-dtype")
    prefill = (
        f"{LAUNCHER} {engine} {BENCHMARK.format(phase='prefill')} --context-length 32768"
        f" --prefill-max-kv-read-token-samples 29 {TOKEN_BUDGET} {RESULT_FILES}"
        " --prefill-max-new-token-samples 8192"
    )
    decode = (
        f"{LAUNCHER} {engine} {BENCHMARK.format(phase='decode')} --context-length 32768"
        f" --max-running-requests 64 {RESULT_FILES}"
    )
    assert _command(tmp_path / "prefill") == prefill.split()
    assert _command(tmp_path / "decode") == decode.split()


@pytest.mark.parametrize(
    ("cell_dtype", "sglang_dtype"),
    [("fp8", "fp8_e4m3"), ("bfloat16", "auto"), ("auto", "auto")],
)
def test_kv_cache_dtype_is_translated_as_the_generator_does_for_sglang(tmp_path, monkeypatch, cell_dtype, sglang_dtype):
    plan = _plan(tmp_path, monkeypatch)
    cell = dataclasses.replace(_cell(plan, "decode"), kv_cache_dtype=cell_dtype)

    _render(plan, cell, tmp_path / "cell")

    command = _command(tmp_path / "cell")
    assert command[command.index("--kv-cache-dtype") + 1] == sglang_dtype


def test_slurm_cell_needs_no_kubernetes_image(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch, fpm_executor="slurm", fpm_slurm_container_image=FAKE_IMAGE)

    _render(plan, _cell(plan, "decode"), tmp_path / "cell", {})

    assert _command(tmp_path / "cell") == DECODE_COMMAND


def test_image_of_the_backend_policy_counts(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    cell = dataclasses.replace(_cell(plan, "decode"), backend_policy=BackendPolicy("custom", DEPLOYMENT, {}))

    _render(plan, cell, tmp_path / "cell", {})

    assert yaml.safe_load((tmp_path / "cell" / "k8s_deploy.yaml").read_text())["spec"]["containers"][0]["image"] == (
        FAKE_IMAGE
    )


@pytest.mark.parametrize("value", ["on", "TRUE"])
def test_real_state_switches_that_say_on_are_kept(tmp_path, monkeypatch, value):
    plan = _plan(tmp_path, monkeypatch)
    overrides = _with_environment(("DYN_BENCH_KV_WARMUP", value), ("DYN_BENCH_PREFILL_REAL_SEED", value))

    _render(plan, _cell(plan, "decode"), tmp_path / "cell", overrides)

    script = (tmp_path / "cell" / "run.sh").read_text().splitlines()
    assert f"export DYN_BENCH_KV_WARMUP={value}" in script
    assert f"export DYN_BENCH_PREFILL_REAL_SEED={value}" in script


# ---------------------------------------------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------------------------------------------


def test_missing_image_on_kubernetes_is_refused_before_the_render(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)

    with _refusal("K8sConfig.k8s_image must name the SGLang image"):
        _render(plan, _cell(plan, "prefill"), tmp_path / "cell", {})

    assert list((tmp_path / "cell").iterdir()) == []


@pytest.mark.parametrize(
    ("name", "value"),
    [("DYN_BENCH_KV_WARMUP", "off"), ("DYN_BENCH_PREFILL_REAL_SEED", "off"), ("DYN_BENCH_KV_WARMUP", "False")],
)
def test_real_state_switch_that_says_off_is_refused_before_the_render(tmp_path, monkeypatch, name, value):
    plan = _plan(tmp_path, monkeypatch)

    with _refusal(f"{name}={value!r} in K8sConfig.extra_env: the launcher stages real cached state only"):
        _render(plan, _cell(plan, "prefill"), tmp_path / "cell", _with_environment((name, value)))

    assert list((tmp_path / "cell").iterdir()) == []


def test_real_state_switch_of_the_backend_policy_is_refused(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    policy = _policy(extra_env=[{"name": "DYN_BENCH_PREFILL_REAL_SEED", "value": "off"}])
    cell = dataclasses.replace(_cell(plan, "prefill"), backend_policy=policy)

    with _refusal("DYN_BENCH_PREFILL_REAL_SEED='off' in K8sConfig.extra_env"):
        _render(plan, cell, tmp_path / "cell")


def test_real_state_switch_is_checked_again_in_the_rendered_run_script(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    name = "DYN_BENCH_KV_WARMUP"
    render = _edited_render(_edit_file("run.sh", lambda text: text.replace(f"export {name}=on", f"export {name}=off")))

    with _refusal(f"{name}='off' in run.sh"):
        _render(plan, _cell(plan, "decode"), tmp_path / "cell", render=render)


CONTENT_VARIABLE = "DYN_BENCH_PREFILL_CONTENT"
# Values of the content variable: the three the launcher takes, then words and spellings it refuses.
CONTENT_WORDS_ACCEPTED = ("", "random", "sharegpt_chain")
CONTENT_WORDS_REFUSED = ("sharegpt", "zeros", "RANDOM", "Sharegpt_Chain", " random", "sharegpt_chain ", "0", "off")
# FAKE expert keys, as the launcher reads them from the model config of a model with expert layers.
FAKE_EXPERT_KEYS = {"num_experts": 60}


def _launcher_refuses_content(value: str, experts: dict) -> bool:
    """Whether the launcher refuses ``value`` of the content variable for a model with these expert keys."""
    if launcher._refused_variables({launcher.ENV_PREFILL_CONTENT: value}):
        return True
    try:
        launcher._is_text_run(value or None, experts)
    except Exception as refusal:
        assert str(refusal).startswith("rejected: ")
        return True
    return False


@pytest.mark.parametrize("is_moe", [False, True], ids=["dense", "moe"])
@pytest.mark.parametrize("value", [*CONTENT_WORDS_ACCEPTED, *CONTENT_WORDS_REFUSED])
def test_content_word_is_refused_at_render_exactly_when_the_launcher_refuses_it(value, is_moe):
    refused = False
    try:
        builder._check_content_word(value, "run.sh", is_moe=is_moe)
    except ValueError:
        refused = True

    assert refused is _launcher_refuses_content(value, FAKE_EXPERT_KEYS if is_moe else {})
    # What each side refuses: every word but the accepted three, and the seeded word for a model with expert layers.
    assert refused is (value not in CONTENT_WORDS_ACCEPTED or (is_moe and value == "random"))


def test_content_word_of_a_text_run_is_kept_for_a_moe_plan(tmp_path, monkeypatch):
    plan = _moe_plan(tmp_path, monkeypatch)
    value = "sharegpt_chain"

    _render(plan, _cell(plan, "prefill", "tep"), tmp_path / "cell", _with_environment((CONTENT_VARIABLE, value)))

    assert f"export {CONTENT_VARIABLE}={shlex.quote(value)}" in (tmp_path / "cell" / "run.sh").read_text().splitlines()


@pytest.mark.parametrize(
    ("value", "executor"),
    [("sharegpt", "kubernetes"), (None, "kubernetes"), ("sharegpt", "slurm")],
    ids=["sharegpt", "None", "sharegpt-on-slurm"],
)
def test_content_word_the_launcher_refuses_is_refused_before_the_render(tmp_path, monkeypatch, value, executor):
    if executor == "slurm":
        # Slurm skips the image check and not the environment checks: its inputs name no Kubernetes image.
        plan = _plan(tmp_path, monkeypatch, fpm_executor="slurm", fpm_slurm_container_image=FAKE_IMAGE)
        overrides = {"K8sConfig": {"extra_env": [{"name": CONTENT_VARIABLE, "value": value}]}}
    else:
        plan = _plan(tmp_path, monkeypatch)
        overrides = _with_environment((CONTENT_VARIABLE, value))

    with _refusal(f"{CONTENT_VARIABLE}={value!r} in K8sConfig.extra_env"):
        _render(plan, _cell(plan, "prefill"), tmp_path / "cell", overrides)

    assert list((tmp_path / "cell").iterdir()) == []


def test_seeded_content_word_is_refused_for_a_moe_plan_before_the_render(tmp_path, monkeypatch):
    plan = _moe_plan(tmp_path, monkeypatch)
    text = (
        f"{CONTENT_VARIABLE}=random in K8sConfig.extra_env: the launcher does not support it for a model with "
        "expert layers, whose cells send dataset text"
    )

    with _refusal(text):
        _render(plan, _cell(plan, "decode", "tep"), tmp_path / "cell", _with_environment((CONTENT_VARIABLE, "random")))

    assert list((tmp_path / "cell").iterdir()) == []


def _with_content_lines(*values: str):
    """A render whose ``run.sh`` exports the content variable once per value, in this order.

    The inputs of the cell do not name the variable, so only the check of the rendered script sees it.
    """
    lines = "\n".join(f"export {CONTENT_VARIABLE}={shlex.quote(value)}" for value in values)
    return _edited_render(
        _edit_file("run.sh", lambda text: text.replace("\nengine_command=(", f"\n{lines}\nengine_command=("))
    )


def test_seeded_content_word_in_the_rendered_run_script_is_refused_for_a_moe_plan_only(tmp_path, monkeypatch):
    dense = _plan(tmp_path, monkeypatch)
    _render(dense, _cell(dense, "decode"), tmp_path / "dense-cell", render=_with_content_lines("random"))
    assert f"export {CONTENT_VARIABLE}=random" in (tmp_path / "dense-cell" / "run.sh").read_text().splitlines()

    moe = _moe_plan(tmp_path, monkeypatch)
    with _refusal(f"{CONTENT_VARIABLE}=random in run.sh: the launcher does not support it for a model with expert"):
        _render(moe, _cell(moe, "decode", "tep"), tmp_path / "moe-cell", render=_with_content_lines("random"))


def test_the_last_export_line_of_the_content_variable_decides(tmp_path, monkeypatch):
    # The shell keeps the last export of a name, so the launcher reads that one.
    dense = _plan(tmp_path, monkeypatch)
    _render(dense, _cell(dense, "decode"), tmp_path / "dense", render=_with_content_lines("sharegpt", "random"))

    moe = _moe_plan(tmp_path, monkeypatch)
    _render(moe, _cell(moe, "decode", "tep"), tmp_path / "moe", render=_with_content_lines("random", "sharegpt_chain"))
    with _refusal(f"{CONTENT_VARIABLE}=random in run.sh: the launcher does not support it for a model with expert"):
        _render(
            moe, _cell(moe, "decode", "tep"), tmp_path / "moe-2", render=_with_content_lines("sharegpt_chain", "random")
        )


def _with_worker_environment(*entries: tuple[str, str]) -> dict:
    """Deployment inputs that name variables in the worker's pod spec, and none in ``K8sConfig.extra_env``."""
    overrides = _with_environment()
    overrides["K8sConfig"]["worker_extra_pod_spec"] = {
        "mainContainer": {"env": [{"name": name, "value": value} for name, value in entries]}
    }
    return overrides


def test_content_word_of_the_worker_pod_spec_alone_is_refused_in_the_rendered_run_script(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)

    with _refusal(f"{CONTENT_VARIABLE}='sharegpt' in run.sh: the launcher does not support it"):
        _render(
            plan, _cell(plan, "decode"), tmp_path / "cell", _with_worker_environment((CONTENT_VARIABLE, "sharegpt"))
        )


def test_unknown_flag_is_refused_and_the_vllm_render_stays(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    cell = dataclasses.replace(_cell(plan, "prefill"), backend_policy=_policy("--enable-something"))
    _render_vllm(plan, cell, tmp_path / "vllm")

    with _refusal("the vLLM flag --enable-something has no SGLang translation"):
        _render(plan, cell, tmp_path / "sglang")

    # Never a mix of the two: no file was rewritten, so the directory holds what the cell was refused for.
    assert _files(tmp_path / "sglang") == _files(tmp_path / "vllm")
    assert _command(tmp_path / "sglang")[:3] == ["python3", "-m", "dynamo.vllm"]


def test_cell_on_more_than_one_node_is_refused(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    # Sixteen GPUs are two nodes of this system.
    cell = dataclasses.replace(
        _cell(plan, "decode"), topology=ParallelTopology(tp=16, pp=1, dp=1, moe_tp=1, moe_ep=1, cp=1)
    )

    with _refusal("the launcher runs on one node, and fpm_env.sh exports 2 as FPM_NODE_COUNT"):
        _render(plan, cell, tmp_path / "cell")


def test_timeout_that_leaves_the_launcher_under_a_minute_is_refused():
    with _refusal("--benchmark-timeout 659: the launcher's deadline is 600 s earlier and must be at least 60 s"):
        builder._translate_command(_vllm("--benchmark-mode", "decode", "--benchmark-timeout", "659"))

    command = builder._translate_command(_vllm("--benchmark-mode", "decode", "--benchmark-timeout=660"))
    assert command[command.index("--benchmark-timeout") + 1] == "60"


# A run.sh the Generator does not write today: (the change, the refusal).
@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        (
            lambda text: text.replace(COMMAND_LINE_START, "engine_arguments=("),
            f"run.sh has 0 lines that start with {COMMAND_LINE_START}, expected one",
        ),
        # What the shell would expand is not a list of words this module can read.
        (
            lambda text: text.replace(f"{COMMAND_LINE_START}python3 ", f'{COMMAND_LINE_START}"$PYTHON" '),
            "the command line of run.sh is not a plain list of quoted words: ",
        ),
        (
            lambda text: text.replace("-m dynamo.vllm", "-m dynamo.sglang"),
            "the command starts with 'python3 -m dynamo.sglang', not with 'python3 -m dynamo.vllm'",
        ),
        # The gate signals the process group of run.sh, so its last line must start the command in one.
        (
            lambda text: text + 'echo "done"\n',
            'the last line of run.sh does not start the command in its own process group: echo "done"',
        ),
        # A flag added after the assignment would reach the launcher without a translation.
        (
            lambda text: text.replace(
                "\nif (( FPM_NODE_COUNT > 1 )); then\n",
                "\nengine_command+=(--enable-something)\nif (( FPM_NODE_COUNT > 1 )); then\n",
            ),
            "run.sh names the command array in a line this builder does not know: engine_command+=(--enable-something)",
        ),
    ],
    ids=["no-command-line", "no-plain-list-of-words", "not-the-vllm-worker", "no-setsid-line", "a-later-append"],
)
def test_run_script_the_generator_does_not_write_is_refused(tmp_path, monkeypatch, change, refusal):
    plan = _plan(tmp_path, monkeypatch)

    with _refusal(refusal):
        _render(plan, _cell(plan, "decode"), tmp_path / "cell", render=_edited_render(_edit_file("run.sh", change)))


def test_manifest_without_a_working_dir_is_refused(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    render = _edited_render(
        _edit_file(
            "k8s_deploy.yaml",
            lambda text: "".join(line for line in text.splitlines(keepends=True) if "workingDir" not in line),
        )
    )

    with _refusal("k8s_deploy.yaml has 0 workingDir lines, expected one"):
        _render(plan, _cell(plan, "decode"), tmp_path / "cell", render=render)

    # The manifest is read after run.sh was translated in memory. Nothing of the translation is on disk.
    assert _command(tmp_path / "cell")[:3] == ["python3", "-m", "dynamo.vllm"]
    assert builder.GATE_SWITCH_ENV not in (tmp_path / "cell" / "collector-runtime-env.sh").read_text()


# ---------------------------------------------------------------------------------------------------------------
# The translation table
# ---------------------------------------------------------------------------------------------------------------


def test_every_row_of_the_table_in_one_command():
    values = {
        # Two data-parallel ranks, so that the attention-DP size is written too. With it the three numbers of
        # one rank are written as their product with it: the tensor-parallel size (with the expert-parallel
        # size after it), the chunk size of the token budget and the request cap.
        "--data-parallel-size": "2",
        "--data-parallel-backend": "mp",
        "--kv-cache-dtype": "bfloat16",
        "--benchmark-mode": "decode",
        "--benchmark-timeout": "7200",
        "--max-model-len": "4096",
        "--max-num-batched-tokens": "2048",
    }
    tokens = _vllm()
    for flag, row in builder._TRANSLATION.items():
        tokens.extend([flag, values.get(flag, "1")] if row.takes_value else [flag])

    command = builder._translate_command(tokens)

    assert (
        command
        == (
            f"{LAUNCHER} --model-path 1 --served-model-name 1 --tp-size 2 --ep-size 2 --attn-dp-size 2"
            " --kv-cache-dtype auto --mem-fraction-static 1 --benchmark-mode decode"
            " --benchmark-warmup-iterations 1 --benchmark-points-file 1"
            " --benchmark-output-path 1 --dump-config-to 1 --prefill-max-new-token-samples 1"
            " --prefill-max-kv-read-token-samples 1 --prefix-max-batch-size-samples 1"
            " --decode-max-kv-read-token-samples 1 --decode-max-batch-size-samples 1 --benchmark-timeout 6600"
            " --context-length 4096 --chunked-prefill-size 4096 --max-prefill-tokens 2048 --max-running-requests 2"
        ).split()
    )
    # The table is the whole of what can be written.
    assert {token for token in command if token.startswith("--")} == _written_flags()


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ("--benchmark-mode=decode --benchmark-timeout=7200", "--benchmark-mode decode --benchmark-timeout 6600"),
        ("--benchmark-mode decode --model=/models/a=b", "--benchmark-mode decode --model-path /models/a=b"),
        ("--benchmark-mode decode --max-model-len -1", "--benchmark-mode decode"),
        # A cap must be at least 2 (sglbench_grid.py ``SampleCaps``).
        (
            "--benchmark-mode prefill --max-num-batched-tokens 1",
            "--benchmark-mode prefill --chunked-prefill-size 1 --max-prefill-tokens 1"
            " --prefill-max-new-token-samples 2",
        ),
        # A cap that is already there stays, and then no token budget is needed.
        (
            "--benchmark-mode prefill --prefill-max-new-token-samples 7",
            "--benchmark-mode prefill --prefill-max-new-token-samples 7",
        ),
        # The expert-parallel switch is written as the size, which is the tensor-parallel size, right after it:
        # in the order of the Generator's command, with other flags between the two, and with the switch first.
        (
            "--tensor-parallel-size 4 --pipeline-parallel-size 1 --data-parallel-size 1 --enable-expert-parallel"
            " --benchmark-mode decode",
            "--tp-size 4 --ep-size 4 --benchmark-mode decode",
        ),
        (
            "--tensor-parallel-size 2 --kv-cache-dtype fp8 --benchmark-mode decode --enable-expert-parallel",
            "--tp-size 2 --ep-size 2 --kv-cache-dtype fp8_e4m3 --benchmark-mode decode",
        ),
        (
            "--enable-expert-parallel --benchmark-mode decode --tensor-parallel-size=8",
            "--benchmark-mode decode --tp-size 8 --ep-size 8",
        ),
        # Without the switch no size is written, and SGLang's default of one expert-parallel rank applies: the
        # sizes of a dense TP cell and of a pure TP cell.
        ("--benchmark-mode decode --tensor-parallel-size 2", "--benchmark-mode decode --tp-size 2"),
        # The command of a DEP cell. --tp-size and --ep-size count the ranks of every data-parallel rank, the
        # data-parallel size is written as the attention-DP size in the place of the vLLM flag, and the
        # start-up of vLLM's data-parallel engine cores is dropped.
        (
            "--tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 2 --enable-expert-parallel"
            " --benchmark-mode decode --data-parallel-backend mp",
            "--tp-size 2 --ep-size 2 --attn-dp-size 2 --benchmark-mode decode",
        ),
        (
            "--tensor-parallel-size 1 --data-parallel-size=8 --enable-expert-parallel --benchmark-mode decode",
            "--tp-size 8 --ep-size 8 --attn-dp-size 8 --benchmark-mode decode",
        ),
        # The product of the two sizes, whatever the order of the flags. No plan holds such a cell, and the
        # launcher refuses the layout.
        (
            "--data-parallel-size 4 --benchmark-mode decode --enable-expert-parallel --tensor-parallel-size 2",
            "--attn-dp-size 4 --benchmark-mode decode --tp-size 8 --ep-size 8",
        ),
        # One data-parallel rank writes nothing, with or without the start-up flag, and the tensor-parallel
        # size then stays the text it was: nothing is computed.
        (
            "--benchmark-mode decode --tensor-parallel-size 02 --data-parallel-size 1 --data-parallel-backend mp",
            "--benchmark-mode decode --tp-size 02",
        ),
        (
            "--tensor-parallel-size 4 --data-parallel-size 1 --enable-expert-parallel --benchmark-mode decode",
            "--tp-size 4 --ep-size 4 --benchmark-mode decode",
        ),
        ("--benchmark-mode decode --data-parallel-size 1", "--benchmark-mode decode"),
    ],
)
def test_translation_of_single_flags(flags, expected):
    assert builder._translate_command(_vllm(*flags.split())) == f"{LAUNCHER} {expected}".split()


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        ("--benchmark-mode agg", "--benchmark-mode agg: a cell measures prefill or decode"),
        (f"--model {MODEL}", "the command has no --benchmark-mode"),
        (
            "--benchmark-mode prefill",
            "a prefill cell without --prefill-max-new-token-samples needs --max-num-batched-tokens",
        ),
        ("--benchmark-mode decode --max-num-seqs 8 --max-num-seqs=16", "--max-num-seqs is given twice"),
        ("--benchmark-mode", "--benchmark-mode has no value"),
        ("--no-async-scheduling=true", "--no-async-scheduling takes no value, got 'true'"),
        ("--max-model-len 0", "--max-model-len 0: a positive integer is required"),
        ("--max-model-len -2", "--max-model-len -2: a positive integer is required"),
        ("--benchmark-timeout soon", "--benchmark-timeout soon: a positive integer is required"),
        ("--kv-cache-dtype float16", "--kv-cache-dtype float16: SGLang accepts auto, fp8_e5m2, fp8_e4m3"),
        # What the runner adds for a vLLM cell with memory or execution observation, or with a runtime probe.
        ("--worker-cls fpm_memory_worker.FpmExecutionWorker", "the vLLM flag --worker-cls has no SGLang translation"),
        # The expert-parallel switch has no size of its own.
        (
            "--benchmark-mode decode --enable-expert-parallel",
            "--enable-expert-parallel is written with the value of --tensor-parallel-size, which the command does "
            "not give",
        ),
        # Data-parallel ranks without the expert-parallel switch: whole replicas of the model.
        (
            "--tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 2 --benchmark-mode decode"
            " --data-parallel-backend mp",
            "--data-parallel-size 2 without --enable-expert-parallel: the launcher runs data-parallel ranks only as "
            "DP attention with one expert-parallel rank on each",
        ),
        # With the switch, each of the two sizes of the product must be a positive integer.
        (
            "--tensor-parallel-size 0 --data-parallel-size 2 --enable-expert-parallel --benchmark-mode decode",
            "--tensor-parallel-size 0: a positive integer is required",
        ),
        ("--data-parallel-size 0", "--data-parallel-size 0: a positive integer is required"),
        # Another start-up of vLLM's data-parallel engine cores than the one the Generator writes.
        ("--data-parallel-backend ray", "--data-parallel-backend ray: only mp, the value the Generator writes, is"),
        ("--pipeline-parallel-size 2", "--pipeline-parallel-size 2: the launcher runs one pipeline stage"),
        # The other MoE flags of a vLLM cell.
        ("--kernel-config {}", "the vLLM flag --kernel-config has no SGLang translation"),
        ("--enable-eplb", "the vLLM flag --enable-eplb has no SGLang translation"),
        # What the runner adds for a model of the architecture GlmMoeDsaForCausalLM: its cells are refused.
        ("--trust-remote-code", "the vLLM flag --trust-remote-code has no SGLang translation"),
        ("--reasoning-parser=glm45", "the vLLM flag --reasoning-parser has no SGLang translation"),
    ],
)
def test_translation_refusals(flags, message):
    with _refusal(message):
        builder._translate_command(_vllm(*flags.split()))


@pytest.mark.parametrize("size", ["1", "01", "2", "4", "016"])
def test_each_rank_of_a_dep_cell_gets_the_whole_token_budget_and_request_cap(size):
    command = builder._translate_command(
        _vllm(
            *("--tensor-parallel-size", "1", "--data-parallel-size", size, "--enable-expert-parallel"),
            *("--benchmark-mode", "prefill", "--max-num-batched-tokens", "8192", "--max-num-seqs", "64"),
        )
    )

    # What SGLang gives one rank is the written number divided by the attention-DP size, rounded down
    # (``test_sglang_divides_the_chunk_size_and_the_request_cap_of_a_rank_at_each_tested_release``).
    ranks = int(size)
    assert command[command.index("--chunked-prefill-size") + 1] == str(8192 * ranks)
    assert command[command.index("--max-running-requests") + 1] == str(64 * ranks)
    # SGLang does not divide the new tokens of one prefill batch, and the cap of the new-token axis is the
    # chunk size of one rank.
    assert command[command.index("--max-prefill-tokens") + 1] == "8192"
    assert command[command.index("--prefill-max-new-token-samples") + 1] == "8192"
    # With one rank the command is the one of a cell without data-parallel ranks.
    assert ("--attn-dp-size" in command) is (ranks > 1)


def test_limits_given_before_the_data_parallel_size_are_multiplied_by_it():
    # The two limits stand before the data-parallel size they are multiplied by.
    limits = ("--max-num-batched-tokens=4096", "--max-num-seqs=7")
    ranks = ("--tensor-parallel-size", "1", "--data-parallel-size=3", "--enable-expert-parallel")

    command = builder._translate_command(_vllm(*limits, *ranks, "--benchmark-mode", "decode"))

    assert command[command.index("--chunked-prefill-size") + 1] == "12288"
    assert command[command.index("--max-prefill-tokens") + 1] == "4096"
    assert command[command.index("--max-running-requests") + 1] == "21"
    assert command[command.index("--tp-size") + 1] == command[command.index("--ep-size") + 1] == "3"


def test_written_flags_are_the_launchers_or_sglangs():
    written = _written_flags()
    declared_by_the_launcher = {
        option for action in launcher._build_parser()._actions for option in action.option_strings
    }

    assert written & declared_by_the_launcher == LAUNCHER_FLAGS
    assert written - declared_by_the_launcher == SGLANG_FLAGS
    # The sixth cap, the end of the decode batch axis, is not written: the cell bounds the batch on the engine.
    assert "--benchmark-max-batch-size" in declared_by_the_launcher - written


def test_nothing_the_launcher_refuses_can_be_written():
    written = _written_flags()

    assert written.isdisjoint(REFUSED_BY_THE_LAUNCHER)
    assert written.isdisjoint(OTHER_ATTENTION_DP_SPELLINGS)
    assert written.isdisjoint(flag for _dest, flag, _variable, _why in launcher._REFUSED_SWITCHES)
    assert written.isdisjoint(flag for _dest, flag, _variable, _use in launcher._LEGACY_FLAGS)
    # The two flags a vLLM cell switches scheduling and caching with have no counterpart in the cell.
    assert builder._TRANSLATION["--no-async-scheduling"].writes == ()
    assert builder._TRANSLATION["--no-enable-prefix-caching"].writes == ()


def _sglang_field_files(version: str) -> list[Path]:
    """The field files of one SGLang release in a local checkout. Skips the test when the variable is unset."""
    base = os.environ.get(SOURCES_ENV)
    if not base:
        pytest.skip(f"{SOURCES_ENV} is not set; it names a folder with one SGLang checkout per tested release")
    files = sorted((Path(base) / f"v{version}" / "python/sglang/srt/arg_groups/fields").glob("*.py"))
    # A variable that is set and wrong is a mistake to show, not a reason to skip.
    assert files, f"{SOURCES_ENV} holds no checkout of SGLang {version}"
    return files


def _declared_arguments(files: list[Path]) -> dict[str, ast.Call | None]:
    """Every flag the field files declare, with the ``Arg(...)`` call of its field when it has one.

    A field ``name: A[type, "help"]`` or ``name: A[type, Arg(...)]`` is the flag ``--name`` with dashes, or the
    ``cli_name`` of its ``Arg``, plus its ``aliases``; a field with ``no_cli=True`` is none
    (srt/arg_groups/arg_utils.py ``add_cli_args_from_dataclass``).
    """
    declared: dict[str, ast.Call | None] = {}
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
                continue
            annotation = node.annotation
            if not isinstance(annotation, ast.Subscript) or not isinstance(annotation.slice, ast.Tuple):
                continue
            # The first metadata entry that is an ``Arg`` or a help text counts; other calls are skipped.
            metadata = next(
                (
                    entry
                    for entry in annotation.slice.elts[1:]
                    if not isinstance(entry, ast.Call) or getattr(entry.func, "id", None) == "Arg"
                ),
                None,
            )
            if metadata is None:
                continue
            arg = metadata if isinstance(metadata, ast.Call) else None
            keywords = {keyword.arg: keyword.value for keyword in arg.keywords} if arg is not None else {}
            if "no_cli" in keywords and ast.literal_eval(keywords["no_cli"]):
                continue
            names = [ast.literal_eval(keywords["cli_name"])] if "cli_name" in keywords else []
            names = names or ["--" + node.target.id.replace("_", "-")]
            names.extend(ast.literal_eval(keywords["aliases"]) if "aliases" in keywords else [])
            for name in names:
                declared[name] = arg
    return declared


@pytest.mark.parametrize("version", compat.TESTED_VERSIONS)
def test_sglang_flags_are_declared_at_each_tested_release(version):
    declared = _declared_arguments(_sglang_field_files(version))

    assert sorted(SGLANG_FLAGS - set(declared)) == []
    # What the launcher refuses is a real flag of that release, so leaving it out is a choice and no accident.
    assert sorted(REFUSED_BY_THE_LAUNCHER - set(declared)) == []


@pytest.mark.parametrize("version", compat.TESTED_VERSIONS)
def test_sglang_divides_the_chunk_size_and_the_request_cap_of_a_rank_at_each_tested_release(version):
    # What the note at the translation table says SGLang does with the two numbers of a DEP cell, which is why
    # the cell writes them multiplied by the attention-DP size.
    srt = _sglang_field_files(version)[0].parents[2]
    hook = ast.parse((srt / "arg_groups" / "parallel_hook.py").read_text())
    (data_parallelism,) = [
        node for node in ast.walk(hook) if isinstance(node, ast.FunctionDef) and node.name == "handle_data_parallelism"
    ]
    source = ast.unparse(data_parallelism)
    assert "chunked_prefill_size=cfg.chunked_prefill_size // cfg.attn_dp_size" in source
    assert "assert cfg.tp_size % cfg.attn_dp_size == 0" in source
    # The new tokens of one prefill batch are not divided there.
    assert "max_prefill_tokens" not in source

    configurator = ast.parse((srt / "mem_cache" / "kv_cache_configurator.py").read_text())
    (per_rank_requests,) = [
        node
        for node in ast.walk(configurator)
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_max_num_reqs"
    ]
    division = "max_num_reqs // self.attn_dp_size"
    assert division in ast.unparse(per_rank_requests)
    # Only the later release keeps at least one request for a rank.
    assert (f"max(1, {division})" in ast.unparse(per_rank_requests)) is (version == "0.5.23")


@pytest.mark.parametrize("version", compat.TESTED_VERSIONS)
def test_kv_cache_dtypes_are_sglangs_choices_at_each_tested_release(version):
    kv_cache_dtype = _declared_arguments(_sglang_field_files(version))["--kv-cache-dtype"]

    choices = {keyword.arg: keyword.value for keyword in kv_cache_dtype.keywords}["choices"]
    assert tuple(ast.literal_eval(choices)) == builder._SGLANG_KV_CACHE_DTYPES
    assert set(builder._KV_CACHE_DTYPES.values()) <= set(builder._SGLANG_KV_CACHE_DTYPES)


# ---------------------------------------------------------------------------------------------------------------
# The new-token cap, with the launcher's own grid
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize(
    ("graph_sizes", "low_end"),
    [
        # The longest axis there can be: a graph for every token count up to the budget. One token has a graph
        # of its own, so no batch runs without one and the low end is the 1 that is a graph size.
        (tuple(range(1, 513)), [1]),
        # Graph sizes above the budget add nothing below it. The smallest size, 256, is at most twice the tokens
        # of a batch from 128 tokens on.
        ((256, 1024, 4096), [1, 127, 128]),
    ],
)
def test_token_budget_as_the_cap_never_cuts_the_new_token_axis(ranks, graph_sizes, low_end):
    budget = 512
    sizes = ("--tensor-parallel-size", "1", "--data-parallel-size", str(ranks), "--enable-expert-parallel")
    command = builder._translate_command(
        _vllm(*sizes, "--benchmark-mode", "prefill", "--max-num-batched-tokens", str(budget))
    )
    cap = int(command[command.index("--prefill-max-new-token-samples") + 1])
    # What an engine started with the cell's two token flags reports, with FAKE sizes for the rest. Under DP
    # attention a scheduler reports the written chunk size divided by the number of ranks.
    limits = grid.Limits(
        page_size=1,
        max_req_input_len=131072,
        max_total_num_tokens=1048576,
        max_running_requests=64,
        max_prefill_tokens=int(command[command.index("--max-prefill-tokens") + 1]),
        chunked_prefill_size=int(command[command.index("--chunked-prefill-size") + 1]) // ranks,
        prefill_graph_sizes=graph_sizes,
    )

    axis = grid.prefill_new_token_axis(limits, grid.SampleCaps(prefill_max_new_token_samples=cap))

    n_limit = grid.largest_prefill_new_tokens(limits)
    uncut = grid.graph_axis(graph_sizes, n_limit)
    # The cap of the cell cuts nothing: the axis is the whole graph axis with the low end, which is put on the
    # axis after the cut and does not count against the cap.
    assert axis == sorted({*uncut, *low_end})
    assert len(uncut) <= budget == cap
    if len(uncut) > grid.SampleCaps().prefill_max_new_token_samples:
        # Without the cap of the cell the launcher's default would cut this axis.
        assert len(grid.prefill_new_token_axis(limits, grid.SampleCaps())) < len(axis)


# ---------------------------------------------------------------------------------------------------------------
# What the module repeats from its neighbours
# ---------------------------------------------------------------------------------------------------------------


def test_repeated_constants_equal_their_sources():
    assert builder._RUNTIME_ENV_FILENAME == fpm_runner.RUNTIME_ENV_FILENAME
    assert builder._REMOTE_WORKDIR == fpm_runner.REMOTE_WORKDIR
    assert list(builder.LAUNCHER_COMMAND) == LAUNCHER.split()
    assert (RUNTIME / Path(builder.LAUNCHER_COMMAND[1]).name).is_file()
    assert dict(launcher._REAL_STATE_SWITCHES) == builder._REAL_STATE_SWITCHES
    assert (builder._CONTENT_VARIABLE, builder._SEEDED_CONTENT, builder._TEXT_CONTENT) == (
        launcher.ENV_PREFILL_CONTENT,
        launcher.SEEDED_CONTENT,
        launcher.TEXT_CONTENT,
    )
    assert builder._CONTENT_VARIABLE == CONTENT_VARIABLE
    assert builder._MIN_SAMPLE_CAP == 2
    with pytest.raises(ValueError, match="prefill_max_new_token_samples"):
        grid.SampleCaps(prefill_max_new_token_samples=builder._MIN_SAMPLE_CAP - 1)
