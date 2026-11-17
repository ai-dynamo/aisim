# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of ``sglang_fpm_builder.py``, the cell builder of the SGLang self-benchmark. No SGLang, no GPU, no pod.

Every cell is rendered by the real ``runner._render_cell``, so by the real Generator, from a real collection
plan of one dense model on two GPUs. The plan is built as a vLLM plan and turned into an SGLang plan with
``dataclasses.replace``: the tests therefore need neither the planner's admission of ``sglang`` nor the
runner's branch that calls the builder.

What is FAKE is named so: the model (a config typed in here, no checkpoint), the image name, and the
``python3`` that one test puts first on ``PATH`` to see how ``run.sh`` starts its command. Nothing of the
launcher runs in a pod here; what the tests show is the text of the cell's files and that the launcher's own
flag reader takes the command.

The two tests that read SGLang's field declarations need a local checkout of SGLang for each release the
launcher accepts. ``AISIM_SGLBENCH_SGLANG_SOURCES`` names a folder that holds them as ``v<version>``, the
variable and the layout of ``test_fpm_sglbench_preflight.py``; without it those two tests are skipped.
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

RUNTIME = Path(__file__).resolve().parents[3] / "collector/fpm_forward/runtime"
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
    "--attn-dp-size",
    "--attention-data-parallel-size",
    "--enable-dp-attention",
}


# ---------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------


def _plan(tmp_path: Path, monkeypatch, **options) -> FPMCollectionPlan:
    """An SGLang plan of the FAKE dense model on two GPUs of one node: one prefill and one decode cell at TP 2.

    The defaults are what an SGLang plan has: the runtime graph policy, under which the vLLM command holds no
    capture list. ``options`` are attributes of the parsed ``--fpm-*`` arguments.
    """
    monkeypatch.setattr(planner, "_git_revision", lambda: "test-revision")

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("a unit test has no checkpoint")

    monkeypatch.setattr("collector.fpm_forward.memory_admission.KVCacheEstimator.from_request", unavailable)
    config_dir = tmp_path / "model-config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.json").write_text(json.dumps(FAKE_MODEL_CONFIG))
    arguments = {
        "fpm_max_gpus": 2,
        "fpm_gpu_counts": [2],
        "fpm_kv_cache_dtypes": ["fp8"],
        "fpm_prefill_cudagraph_policy": "runtime",
        **options,
    }
    vllm_plan = build_collection_plan(
        backend="vllm",
        model_path=MODEL,
        model_architecture="PrivateDenseForCausalLM",
        system="h200_sxm",
        selected_ops={"attention_context", "attention_generation"},
        has_model_cases=False,
        options=FPMCollectionOptions.from_args(argparse.Namespace(**arguments)),
        collector_config={"aic_database_version": "0.5.22"},
        model_config_path=str(config_dir),
    )
    assert [(cell.workload_kind, cell.parallel_strategy, cell.topology.tp) for cell in vllm_plan.cells] == [
        ("prefill", "tp", 2),
        ("decode", "tp", 2),
    ]
    return dataclasses.replace(vllm_plan, backend="sglang")


def _cell(plan: FPMCollectionPlan, phase: str) -> FPMCell:
    return next(cell for cell in plan.cells if cell.workload_kind == phase)


def _render(plan, cell, cell_dir: Path, overrides=None, *, smoke: bool = False, render=_render_cell) -> dict:
    """Render ``cell`` as an SGLang cell with the builder. ``render`` stands for ``runner._render_cell``."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    return builder.render_cell(
        plan, cell, cell_dir, DEPLOYMENT if overrides is None else overrides, smoke=smoke, render_vllm_cell=render
    )


def _render_vllm(plan, cell, cell_dir: Path, overrides=None, *, smoke: bool = False) -> dict:
    """The vLLM render of the same cell, to compare with."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    return _render_cell(
        dataclasses.replace(plan, backend="vllm"),
        cell,
        cell_dir,
        DEPLOYMENT if overrides is None else overrides,
        smoke=smoke,
    )


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


def test_the_vllm_render_is_asked_for_the_same_plan_and_the_artifacts_are_the_cell_files(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    cell = _cell(plan, "prefill")
    calls = []

    def render(vllm_plan, *args, **kwargs):
        calls.append((vllm_plan, args, kwargs))
        return _render_cell(vllm_plan, *args, **kwargs)

    artifacts = _render(plan, cell, tmp_path / "cell", smoke=True, render=render)

    ((vllm_plan, args, kwargs),) = calls
    assert vllm_plan.backend == "vllm"
    assert vllm_plan.sha256 == plan.sha256
    assert dataclasses.replace(vllm_plan, backend="sglang") == plan
    assert args == (cell, tmp_path / "cell", DEPLOYMENT)
    assert kwargs == {"smoke": True}
    files = _files(tmp_path / "cell")
    assert set(artifacts) == {"k8s_deploy.yaml", "fpm_env.sh", "run.sh"}
    assert artifacts == {name: files[name] for name in artifacts}


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
    [
        ("fp8", "fp8_e4m3"),
        ("bfloat16", "auto"),
        ("auto", "auto"),
        ("fp8_e5m2", "fp8_e5m2"),
        ("fp8_e4m3", "fp8_e4m3"),
        ("mxfp8", "mxfp8"),
        ("bf16", "bf16"),
        ("nvfp4", "nvfp4"),
        ("fp4_mx_block16", "fp4_mx_block16"),
    ],
)
def test_kv_cache_dtype_is_translated_as_the_generator_does_for_sglang(tmp_path, monkeypatch, cell_dtype, sglang_dtype):
    plan = _plan(tmp_path, monkeypatch)
    cell = dataclasses.replace(_cell(plan, "decode"), kv_cache_dtype=cell_dtype)

    _render(plan, cell, tmp_path / "cell")

    command = _command(tmp_path / "cell")
    assert command[command.index("--kv-cache-dtype") + 1] == sglang_dtype


def test_default_kv_cache_dtype_of_the_plan(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch, fpm_kv_cache_dtypes=None)
    cell = _cell(plan, "prefill")
    assert cell.kv_cache_dtype == "bfloat16"

    _render(plan, cell, tmp_path / "cell")

    assert _command(tmp_path / "cell") == " ".join(PREFILL_COMMAND).replace("fp8_e4m3", "auto").split()


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


@pytest.mark.parametrize("value", ["on", "1", "true", "TRUE", "On"])
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


@pytest.mark.parametrize("value", ["off", "0", "false", "OFF", "False"])
@pytest.mark.parametrize("name", ["DYN_BENCH_KV_WARMUP", "DYN_BENCH_PREFILL_REAL_SEED"])
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


@pytest.mark.parametrize(
    ("name", "accepted"),
    [("DYN_BENCH_KV_WARMUP", "on, 1, true"), ("DYN_BENCH_PREFILL_REAL_SEED", "on, 1, true, auto")],
)
def test_real_state_switch_is_checked_again_in_the_rendered_run_script(tmp_path, monkeypatch, name, accepted):
    plan = _plan(tmp_path, monkeypatch)
    render = _edited_render(_edit_file("run.sh", lambda text: text.replace(f"export {name}=on", f"export {name}=off")))

    with _refusal(f"{name}='off' in run.sh: the launcher stages real cached state only (accepted: {accepted})"):
        _render(plan, _cell(plan, "decode"), tmp_path / "cell", render=render)


def test_unknown_flag_is_refused_and_the_vllm_render_stays(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    cell = dataclasses.replace(_cell(plan, "prefill"), backend_policy=_policy("--enable-something"))
    _render_vllm(plan, cell, tmp_path / "vllm")

    with _refusal("the vLLM flag --enable-something has no SGLang translation"):
        _render(plan, cell, tmp_path / "sglang")

    # Never a mix of the two: no file was rewritten, so the directory holds what the cell was refused for.
    assert _files(tmp_path / "sglang") == _files(tmp_path / "vllm")
    assert _command(tmp_path / "sglang")[:3] == ["python3", "-m", "dynamo.vllm"]


def test_enforce_eager_is_refused(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    plan = dataclasses.replace(plan, options=dataclasses.replace(plan.options, enforce_eager=True))

    with _refusal("the vLLM flag --enforce-eager has no SGLang translation"):
        _render(plan, _cell(plan, "prefill"), tmp_path / "cell")


def test_kv_cache_dtype_that_sglang_does_not_have_is_refused(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    cell = dataclasses.replace(_cell(plan, "decode"), kv_cache_dtype="float16")

    with _refusal("--kv-cache-dtype float16: SGLang accepts auto, fp8_e5m2, fp8_e4m3, mxfp8, bf16, bfloat16, nvfp4,"):
        _render(plan, cell, tmp_path / "cell")


def test_vllm_capture_list_is_refused(tmp_path, monkeypatch):
    # The explicit graph policy, which the planner does not admit for SGLang: the list is vLLM's.
    plan = _plan(tmp_path, monkeypatch, fpm_prefill_cudagraph_policy=None)

    with _refusal("the vLLM flag --compilation-config has no SGLang translation"):
        _render(plan, _cell(plan, "prefill"), tmp_path / "cell")


@pytest.mark.parametrize(
    ("topology", "flag"),
    [
        (ParallelTopology(tp=1, pp=1, dp=2, moe_tp=1, moe_ep=1, cp=1), "--data-parallel-size 2"),
        (ParallelTopology(tp=1, pp=2, dp=1, moe_tp=1, moe_ep=1, cp=1), "--pipeline-parallel-size 2"),
    ],
)
def test_parallel_size_other_than_one_is_refused(tmp_path, monkeypatch, topology, flag):
    plan = _plan(tmp_path, monkeypatch)
    cell = dataclasses.replace(_cell(plan, "prefill"), topology=topology)

    with _refusal(f"{flag}: the launcher runs one data-parallel replica and one pipeline stage"):
        _render(plan, cell, tmp_path / "cell")


def test_cell_on_more_than_one_node_is_refused(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    # Sixteen GPUs are two nodes of this system.
    cell = dataclasses.replace(
        _cell(plan, "decode"), topology=ParallelTopology(tp=16, pp=1, dp=1, moe_tp=1, moe_ep=1, cp=1)
    )

    with _refusal("the launcher runs on one node, and fpm_env.sh exports 2 as FPM_NODE_COUNT"):
        _render(plan, cell, tmp_path / "cell")


def test_timeout_that_leaves_the_launcher_under_a_minute_is_refused(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    short = dataclasses.replace(_cell(plan, "decode"), backend_policy=_policy("--benchmark-timeout", "659"))
    shortest = dataclasses.replace(_cell(plan, "decode"), backend_policy=_policy("--benchmark-timeout=660"))

    with _refusal("--benchmark-timeout 659: the launcher's deadline is 600 s earlier and must be at least 60 s"):
        _render(plan, short, tmp_path / "short")
    _render(plan, shortest, tmp_path / "shortest")

    command = _command(tmp_path / "shortest")
    assert command[command.index("--benchmark-timeout") + 1] == "60"


@pytest.mark.parametrize(
    ("change", "count"),
    [
        (lambda text: text.replace(COMMAND_LINE_START, "engine_arguments=("), 0),
        (
            lambda text: text.replace("\nfor index in", f"\n{COMMAND_LINE_START}python3 -m dynamo.vllm)\nfor index in"),
            2,
        ),
    ],
)
def test_run_script_without_exactly_one_command_line_is_refused(tmp_path, monkeypatch, change, count):
    plan = _plan(tmp_path, monkeypatch)

    with _refusal(f"run.sh has {count} lines that start with {COMMAND_LINE_START}, expected one"):
        _render(plan, _cell(plan, "decode"), tmp_path / "cell", render=_edited_render(_edit_file("run.sh", change)))


@pytest.mark.parametrize(
    "change",
    [
        # What the shell would expand is not a list of words this module can read.
        lambda text: text.replace(f"{COMMAND_LINE_START}python3 ", f'{COMMAND_LINE_START}"$PYTHON" '),
        lambda text: text.replace(f"{COMMAND_LINE_START}python3 ", f"{COMMAND_LINE_START}python3  "),
        lambda text: text.replace("/results/benchmark.json)\n", "/results/benchmark.json\n"),
        lambda text: text.replace("/results/benchmark.json)\n", "'/results/benchmark.json)\n"),
    ],
)
def test_command_line_that_is_not_a_plain_list_of_words_is_refused(tmp_path, monkeypatch, change):
    plan = _plan(tmp_path, monkeypatch)

    with _refusal("the command line of run.sh is not a plain list of quoted words: "):
        _render(plan, _cell(plan, "decode"), tmp_path / "cell", render=_edited_render(_edit_file("run.sh", change)))


def test_command_that_is_not_the_vllm_worker_is_refused(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    render = _edited_render(_edit_file("run.sh", lambda text: text.replace("-m dynamo.vllm", "-m dynamo.sglang")))

    with _refusal("the command starts with 'python3 -m dynamo.sglang', not with 'python3 -m dynamo.vllm'"):
        _render(plan, _cell(plan, "decode"), tmp_path / "cell", render=render)


def test_run_script_that_does_not_end_with_the_setsid_line_is_refused(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    render = _edited_render(_edit_file("run.sh", lambda text: text + 'echo "done"\n'))

    with _refusal('the last line of run.sh does not start the command in its own process group: echo "done"'):
        _render(plan, _cell(plan, "decode"), tmp_path / "cell", render=render)


@pytest.mark.parametrize(
    ("change", "line"),
    [
        # A flag added after the assignment would reach the launcher without a translation.
        (
            lambda text: text.replace(
                "\nif (( FPM_NODE_COUNT > 1 )); then\n",
                "\nengine_command+=(--enable-something)\nif (( FPM_NODE_COUNT > 1 )); then\n",
            ),
            "engine_command+=(--enable-something)",
        ),
        # A second assignment inside a block would start the vLLM worker after all.
        (
            lambda text: text.replace("\nfor index in", "\n  engine_command=(python3 -m dynamo.vllm)\nfor index in"),
            "  engine_command=(python3 -m dynamo.vllm)",
        ),
        (lambda text: text.replace("\nfor index in", "\nengine_command[1]=-I\nfor index in"), "engine_command[1]=-I"),
    ],
)
def test_run_script_that_changes_the_command_elsewhere_is_refused(tmp_path, monkeypatch, change, line):
    plan = _plan(tmp_path, monkeypatch)

    with _refusal(f"run.sh names the command array in a line this builder does not know: {line}"):
        _render(plan, _cell(plan, "decode"), tmp_path / "cell", render=_edited_render(_edit_file("run.sh", change)))


def test_command_array_is_named_only_where_the_builder_knows_it(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    _render(plan, _cell(plan, "prefill"), tmp_path / "cell")

    lines = (tmp_path / "cell" / "run.sh").read_text().split("\n")
    start = lines.index(builder._MULTINODE_BLOCK_START)
    end = lines.index(builder._BLOCK_END, start)
    naming = [index for index, line in enumerate(lines) if "engine_command" in line]
    # On one node the array is assigned once, passed through the placeholder loop and started by the last line.
    assert [lines[index] for index in naming if not start < index < end] == [
        f"{COMMAND_LINE_START}{shlex.join(PREFILL_COMMAND)})",
        *builder._ONE_NODE_ARRAY_LINES,
    ]
    # Everything else is an append of the multinode block, which a cell of one node never enters.
    inside = [lines[index].strip() for index in naming if start < index < end]
    assert inside
    assert all(line.startswith("engine_command+=(") for line in inside)


@pytest.mark.parametrize(
    ("change", "count"),
    [
        (lambda text: "".join(line for line in text.splitlines(keepends=True) if "workingDir" not in line), 0),
        (lambda text: text.replace("    workingDir: ", "    workingDir: /elsewhere\n    workingDir: "), 2),
    ],
)
def test_manifest_without_exactly_one_working_dir_is_refused(tmp_path, monkeypatch, change, count):
    plan = _plan(tmp_path, monkeypatch)
    render = _edited_render(_edit_file("k8s_deploy.yaml", change))

    with _refusal(f"k8s_deploy.yaml has {count} workingDir lines, expected one"):
        _render(plan, _cell(plan, "decode"), tmp_path / "cell", render=render)

    # The manifest is read after run.sh was translated in memory. Nothing of the translation is on disk.
    assert _command(tmp_path / "cell")[:3] == ["python3", "-m", "dynamo.vllm"]
    assert builder.GATE_SWITCH_ENV not in (tmp_path / "cell" / "collector-runtime-env.sh").read_text()


# ---------------------------------------------------------------------------------------------------------------
# The translation table
# ---------------------------------------------------------------------------------------------------------------


def test_every_row_of_the_table_in_one_command():
    values = {
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
            f"{LAUNCHER} --model-path 1 --served-model-name 1 --tp-size 1 --kv-cache-dtype auto --mem-fraction-static 1"
            " --benchmark-mode decode --benchmark-warmup-iterations 1 --benchmark-points-file 1"
            " --benchmark-output-path 1 --dump-config-to 1 --prefill-max-new-token-samples 1"
            " --prefill-max-kv-read-token-samples 1 --prefix-max-batch-size-samples 1"
            " --decode-max-kv-read-token-samples 1 --decode-max-batch-size-samples 1 --benchmark-timeout 6600"
            " --context-length 4096 --chunked-prefill-size 2048 --max-prefill-tokens 2048 --max-running-requests 1"
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
        ("--model --benchmark-mode decode", "--model has no value"),
        ("--no-async-scheduling=true", "--no-async-scheduling takes no value, got 'true'"),
        ("--max-model-len 0", "--max-model-len 0: a positive integer is required"),
        ("--max-model-len -2", "--max-model-len -2: a positive integer is required"),
        ("--max-num-batched-tokens -1", "--max-num-batched-tokens -1: a positive integer is required"),
        ("--max-num-batched-tokens 8k", "--max-num-batched-tokens 8k: a positive integer is required"),
        ("--benchmark-timeout soon", "--benchmark-timeout soon: a positive integer is required"),
        ("--kv-cache-dtype float16", "--kv-cache-dtype float16: SGLang accepts auto, fp8_e5m2, fp8_e4m3"),
        ("--kv-cache-dtype int8", "--kv-cache-dtype int8: SGLang accepts"),
        # vLLM's parser also takes the underscore spelling; the table does not.
        ("--max_model_len 4096", "the vLLM flag --max_model_len has no SGLang translation"),
        ("serve", "the vLLM flag serve has no SGLang translation"),
        # What the runner adds for a vLLM cell with memory or execution observation, or with a runtime probe.
        ("--worker-cls fpm_memory_worker.FpmExecutionWorker", "the vLLM flag --worker-cls has no SGLang translation"),
        ("--revision main", "the vLLM flag --revision has no SGLang translation"),
    ],
)
def test_translation_refusals(flags, message):
    with _refusal(message):
        builder._translate_command(_vllm(*flags.split()))


def test_written_flags_are_the_launchers_or_sglangs():
    written = _written_flags()
    declared_by_the_launcher = {
        option for action in launcher._build_parser()._actions for option in action.option_strings
    }

    assert written & declared_by_the_launcher == LAUNCHER_FLAGS
    assert written - declared_by_the_launcher == SGLANG_FLAGS
    # The sixth cap, the end of the decode batch axis, is not written: the cell bounds the batch on the engine.
    assert "--benchmark-max-batch-size" in declared_by_the_launcher - written


def test_nothing_the_launcher_refuses_can_be_written(tmp_path, monkeypatch):
    written = _written_flags()

    assert written.isdisjoint(REFUSED_BY_THE_LAUNCHER)
    assert written.isdisjoint(flag for _dest, flag, _variable, _why in launcher._REFUSED_SWITCHES)
    assert written.isdisjoint(flag for _dest, flag, _variable, _use in launcher._LEGACY_FLAGS)
    # The two flags a vLLM cell switches scheduling and caching with have no counterpart in the cell.
    assert builder._TRANSLATION["--no-async-scheduling"].writes == ()
    assert builder._TRANSLATION["--no-enable-prefix-caching"].writes == ()
    plan = _plan(tmp_path, monkeypatch)
    for phase in ("prefill", "decode"):
        _render(plan, _cell(plan, phase), tmp_path / phase)
        assert REFUSED_BY_THE_LAUNCHER.isdisjoint(_command(tmp_path / phase))


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
    assert sorted(REFUSED_BY_THE_LAUNCHER - {"--enable-dp-attention"} - set(declared)) == []


@pytest.mark.parametrize("version", compat.TESTED_VERSIONS)
def test_kv_cache_dtypes_are_sglangs_choices_at_each_tested_release(version):
    kv_cache_dtype = _declared_arguments(_sglang_field_files(version))["--kv-cache-dtype"]

    choices = {keyword.arg: keyword.value for keyword in kv_cache_dtype.keywords}["choices"]
    assert tuple(ast.literal_eval(choices)) == builder._SGLANG_KV_CACHE_DTYPES
    assert set(builder._KV_CACHE_DTYPES.values()) <= set(builder._SGLANG_KV_CACHE_DTYPES)


# ---------------------------------------------------------------------------------------------------------------
# The new-token cap, with the launcher's own grid
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize(
    "graph_sizes",
    [
        (),
        (4, 8, 16, 32, 64, 128, 256, 512),
        # The longest axis there can be: a graph for every token count up to the budget.
        tuple(range(1, 513)),
        # Graph sizes above the budget add nothing below it.
        (256, 1024, 4096),
    ],
)
def test_token_budget_as_the_cap_never_cuts_the_new_token_axis(page_size, graph_sizes):
    budget = 512
    command = builder._translate_command(_vllm("--benchmark-mode", "prefill", "--max-num-batched-tokens", str(budget)))
    cap = int(command[command.index("--prefill-max-new-token-samples") + 1])
    # What an engine started with the cell's two token flags reports, with FAKE sizes for the rest.
    limits = grid.Limits(
        page_size=page_size,
        max_req_input_len=131072,
        max_total_num_tokens=1048576,
        max_running_requests=64,
        max_prefill_tokens=int(command[command.index("--max-prefill-tokens") + 1]),
        chunked_prefill_size=int(command[command.index("--chunked-prefill-size") + 1]),
        prefill_graph_sizes=graph_sizes,
    )

    axis = grid.prefill_new_token_axis(limits, grid.SampleCaps(prefill_max_new_token_samples=cap))

    uncut = grid.graph_axis(graph_sizes, grid.largest_prefill_new_tokens(limits))
    assert axis == (uncut if uncut[0] == 1 else [1, *uncut])
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
    assert builder._MIN_SAMPLE_CAP == 2
    with pytest.raises(ValueError, match="prefill_max_new_token_samples"):
        grid.SampleCaps(prefill_max_new_token_samples=builder._MIN_SAMPLE_CAP - 1)


def test_module_does_not_import_the_runner():
    tree = ast.parse(Path(builder.__file__).read_text())

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update({node.module or "", *(alias.name for alias in node.names)})
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    # The runner imports this module inside ``_render_cell`` and hands itself over as ``render_vllm_cell``.
    assert not any(name.split(".")[-1] == "runner" for name in imported)
