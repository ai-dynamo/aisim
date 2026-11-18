# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cell builder of the SGLang self-benchmark: the vLLM FPM render of a cell, rewritten for the launcher.

The Generator owns the resource Pod and the hardware facts of a cell, and its FPM target renders vLLM cells
only. An SGLang cell is therefore the vLLM FPM render of the same cell with three files rewritten:

- ``run.sh``: the one ``engine_command=(...)`` line becomes the command of ``runtime/sglbench_launcher.py``,
  token by token through the closed table ``_TRANSLATION``. Every other line stays.
- ``k8s_deploy.yaml``: the container's ``workingDir`` is removed. It names the examples folder of the Dynamo
  vLLM image, which a stock SGLang image does not have.
- ``collector-runtime-env.sh``: one more line switches the gate (``runtime/fpm_exec.sh``) to its mode without
  etcd.

``fpm_env.sh`` and ``generator-request.json`` stay as rendered. The request file records the request that was
made to the Generator, which is a vLLM request.

Whatever cannot be translated refuses the cell with a ``ValueError`` that names what was found. Nothing is
passed on untranslated. A value of the two real-state switches or of the content variable that the inputs of
the cell or its rendered ``run.sh`` hold, and that the launcher would refuse in the pod, refuses the cell in
the same way, before a pod is scheduled. Two cases stay with the launcher: a value that reaches the pod in
another way, for example from the image, and the seeded content word for a model that the plan does not take
for an MoE model. A cell that is refused after the render keeps the vLLM files in its directory; the runner
fails the cell before it stages anything.

Where the flag names come from
    The launcher's own flags are those ``runtime/sglbench_launcher.py`` declares (``_build_parser``). Every
    other flag of the command goes to SGLang's parser as it is. Those names were read from SGLang's field
    declarations, ``python/sglang/srt/arg_groups/fields/*.py`` of ``sgl-project/sglang`` (Apache-2.0), at the
    two releases the launcher accepts::

        v0.5.22  https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d
        v0.5.23  https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90

    A field is the flag of its name with dashes (``python/sglang/srt/arg_groups/arg_utils.py``). The fields
    used here are ``model_path``, ``context_length`` and ``kv_cache_dtype`` (``model.py``), ``tp_size`` and
    ``ep_size`` (``parallel.py``), ``mem_fraction_static``, ``max_running_requests``, ``chunked_prefill_size``
    and ``max_prefill_tokens`` (``schedule.py``) and ``served_model_name`` (``serving.py``). No SGLang code is
    copied.
"""

from __future__ import annotations

import dataclasses
import re
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aisim.fpm_contract import FPM_ENV_FILENAME, FPM_MANIFEST_FILENAME, FPM_RUN_SCRIPT_FILENAME

from .planner import FPMCell, FPMCollectionPlan

# The same values as runner.py ``RUNTIME_ENV_FILENAME`` and ``REMOTE_WORKDIR``. The runner imports this module,
# so the two are repeated here; a test compares them.
_RUNTIME_ENV_FILENAME = "collector-runtime-env.sh"
_REMOTE_WORKDIR = "/tmp/fpm-bench"

# The gate reads this from collector-runtime-env.sh (runtime/fpm_exec.sh). It is no FPM_* name: fpm_env.sh
# exports exactly the names of aisim.fpm_contract.FPM_ENV_EXPORTED_VARS.
GATE_SWITCH_ENV = "AISIMULATE_SGLBENCH_GATE"

_VLLM_COMMAND = ("python3", "-m", "dynamo.vllm")
# The launcher is started by path: Python then puts the directory of the staged files on ``sys.path``, where
# the launcher finds its sibling modules. With ``-m`` it does not (runtime/sglbench_launcher.py, "Where it
# runs").
LAUNCHER_COMMAND = ("python3", f"{_REMOTE_WORKDIR}/sglbench_launcher.py")

_COMMAND_LINE_START = "engine_command=("
# The last line of the Generator's run.sh (src/aisim/generator/builders/fpm_builder.py ``_render_run_script``).
# It starts the command in a session of its own, so the gate can signal the launcher's process group, and it
# runs the command's own first token with the rest, so the launcher is started by path.
_EXEC_LINE = (
    "exec python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \"${engine_command[@]}\""
)

# Only the one assignment is translated, so no other line may change the array on a cell of one node. The
# Generator's run.sh names the array in three more lines there: the two of the loop that puts the node rank
# into a placeholder (the command of a one-node cell holds none) and the last line. A cell of several nodes
# adds vLLM flags to the array inside the block that starts with ``_MULTINODE_BLOCK_START`` and ends with the
# next ``fi`` at the start of a line; a one-node cell never enters it (``_require_one_node``). Any other line
# that names the array refuses the cell.
_COMMAND_ARRAY = "engine_command"
_ONE_NODE_ARRAY_LINES = (
    'for index in "${!engine_command[@]}"; do',
    '  engine_command[$index]="${engine_command[$index]//__FPM_NODE_RANK__/$FPM_NODE_RANK}"',
    _EXEC_LINE,
)
_MULTINODE_BLOCK_START = "if (( FPM_NODE_COUNT > 1 )); then"
_BLOCK_END = "fi"

# The Generator sets the gate's deadline to the vLLM benchmark timeout plus 600 s
# (src/aisim/generator/builders/fpm_builder.py ``_benchmark_wait_timeout_seconds``), and the gate counts it
# from the start of run.sh. The launcher looks at its own deadline between points only, and one engine call
# may wait 900 s (runtime/sglbench_launcher.py ``ENGINE_CALL_TIMEOUT_SECONDS``). With the launcher's deadline
# this much before the vLLM timeout, 1,200 s remain for one engine call at its limit, the engine shutdown,
# the write of the result files and the launcher's last wait of 5 s. A point is up to three engine calls, and
# six when it is staged again. A point that needs more than this margin ends the cell at the gate's deadline
# (status 124); no result is published from such a cell.
_DEADLINE_LEAD_SECONDS = 600
# The launcher counts its deadline from its own start, so the engine build is inside it. Less than this
# cannot hold a build.
_MIN_DEADLINE_SECONDS = 60

_MODE_FLAG = "--benchmark-mode"
_TENSOR_PARALLEL_FLAG = "--tensor-parallel-size"
_POINTS_FILE_FLAG = "--benchmark-points-file"
_TOKEN_BUDGET_FLAG = "--max-num-batched-tokens"
# The cap of the prefill new-token axis. An SGLang plan has the runtime graph policy, under which the vLLM
# command carries no cap: the host has no list of the graph sizes, the launcher reads them from the engine.
# The launcher's own default of 64 would then cut graph sizes off the axis (runtime/sglbench_grid.py
# ``SampleCaps``). The axis holds distinct integers from 1 to the largest prefill size, which is at most the
# chunk size (``largest_prefill_new_tokens``), and the cell sets the chunk size to the token budget. An axis
# that is not longer than its cap is not cut (``limit_graph_axis``). So the token budget as the cap cuts
# nothing, whatever sizes the engine captures. A cap below 2 is refused by ``SampleCaps``.
_NEW_TOKEN_CAP_FLAG = "--prefill-max-new-token-samples"
_MIN_SAMPLE_CAP = 2

# The Generator's own translation for SGLang (src/aisim/generator/config/backend_config_mapping.yaml,
# ``kv_cache_dtype``), so a cell passes what AISim's generated SGLang serving deployment passes for the same
# dtype. SGLang has no plain "fp8".
#
# "auto" is the model's dtype only in the plain case. At both tested releases SGLang stores FP8 for a
# checkpoint whose quantization config asks for an FP8 KV cache (``python/sglang/srt/mem_cache/kv_cache_dtype.py``
# ``configure_kv_cache_dtype``), and its per-model overrides turn "auto" into "fp8_e4m3" for some
# architectures (``python/sglang/srt/arg_groups/overrides.py``). The rows of a bfloat16 cell carry "bfloat16"
# as their KV-cache mode whatever SGLang stored: nothing compares the stored dtype with the cell's.
# "bfloat16" is itself a choice of SGLang and would force BF16, which the serving deployment does not do.
_KV_CACHE_DTYPES = {"fp8": "fp8_e4m3", "bfloat16": "auto"}
# The choices of SGLang's ``kv_cache_dtype`` field (``fields/model.py``), the same at both tags.
_SGLANG_KV_CACHE_DTYPES = ("auto", "fp8_e5m2", "fp8_e4m3", "mxfp8", "bf16", "bfloat16", "nvfp4", "fp4_mx_block16")

# The two switches of real cached state and the values the launcher accepts for each
# (runtime/sglbench_launcher.py ``_REAL_STATE_SWITCHES``; a test compares the two tables). The host admits
# off, 0 and false as well (config.py ``with_kv_warmup_defaults``). The launcher would refuse those only
# after the pod was scheduled.
_REAL_STATE_SWITCHES = {
    "DYN_BENCH_KV_WARMUP": ("on", "1", "true"),
    "DYN_BENCH_PREFILL_REAL_SEED": ("on", "1", "true", "auto"),
}

# The variable that chooses the content of a cell, and the two words the launcher accepts for it
# (runtime/sglbench_launcher.py ``ENV_PREFILL_CONTENT``, ``SEEDED_CONTENT`` and ``TEXT_CONTENT``; a test
# compares the constants and the two refusals). The launcher takes the value as it stands, so another
# spelling of a word is another word, and an empty value counts as unset. It refuses the seeded word for a
# model with expert layers once it has read the model config in the pod. The plan has its own answer before
# (``capability.is_moe``, capabilities.py: one of three expert counts of the decoder config is above 1, or
# the model cases select the DSA op pair), and the host refuses the word for an MoE plan here. The
# launcher's test, a truthy value of any of six keys on the config SGLang loads
# (runtime/sglbench_compat.py ``EXPERT_KEYS``), is wider than the count test, so the launcher also refuses
# the word for some models that the plan takes for dense. A plan that is MoE by the DSA op pair alone would
# be refused here although the launcher finds no expert key; every packaged config of the two DSA
# architectures has an expert count above 1. Without these checks the launcher would refuse the cell only
# after the pod was scheduled.
_CONTENT_VARIABLE = "DYN_BENCH_PREFILL_CONTENT"
_SEEDED_CONTENT = "random"
_TEXT_CONTENT = "sharegpt_chain"


def _refused(reason: str) -> ValueError:
    return ValueError(f"SGLang self-benchmark cell refused: {reason}")


def _positive_int(flag: str, value: str) -> int:
    if not re.fullmatch(r"[0-9]+", value) or int(value) < 1:
        raise _refused(f"{flag} {value}: a positive integer is required")
    return int(value)


def _one_rank(flag: str, value: str) -> str:
    if value != "1":
        raise _refused(f"{flag} {value}: the launcher runs one data-parallel replica and one pipeline stage")
    return value


def _kv_cache_dtype(flag: str, value: str) -> str:
    translated = _KV_CACHE_DTYPES.get(value, value)
    if translated not in _SGLANG_KV_CACHE_DTYPES:
        raise _refused(f"{flag} {value}: SGLang accepts {', '.join(_SGLANG_KV_CACHE_DTYPES)}")
    return translated


def _benchmark_mode(flag: str, value: str) -> str:
    if value not in ("prefill", "decode"):
        raise _refused(f"{flag} {value}: a cell measures prefill or decode")
    return value


def _launcher_deadline(flag: str, value: str) -> str:
    deadline = _positive_int(flag, value) - _DEADLINE_LEAD_SECONDS
    if deadline < _MIN_DEADLINE_SECONDS:
        raise _refused(
            f"{flag} {value}: the launcher's deadline is {_DEADLINE_LEAD_SECONDS} s earlier and must be at least "
            f"{_MIN_DEADLINE_SECONDS} s"
        )
    return str(deadline)


def _context_length(flag: str, value: str) -> str | None:
    # Deviation note (no context length under vLLM's auto-fit).
    # What differs: --max-model-len -1, the default of a plan, lets vLLM fit the context to the memory it
    # has. SGLang has no such value, so no --context-length is written and the limit of the model's own
    # config applies (``context_length`` in ``fields/model.py``), on every system.
    # Direction of the error: none known for a collected point. The limit bounds which contexts the grid
    # holds, so the largest context of the table can differ from the one of a vLLM table of the same cell.
    # How to retire it: give the plan a positive --fpm-max-model-len.
    if value == "-1":
        return None
    return str(_positive_int(flag, value))


def _token_budget(flag: str, value: str) -> str:
    return str(_positive_int(flag, value))


@dataclasses.dataclass(frozen=True)
class _Row:
    """What one flag of the vLLM command becomes in the launcher's command."""

    # The flags written for it, each followed by the value. Empty: the flag is dropped.
    writes: tuple[str, ...] = ()
    # Whether a value follows the vLLM flag.
    takes_value: bool = True
    # Checks the value and returns the one to write, or None when this value writes nothing. Without it the
    # value is written as it is.
    value: Callable[[str, str], str | None] | None = None
    # For a switch that SGLang takes as a size: the vLLM flag whose written value it is written with. Its
    # flags follow the flags of that one directly, and a command in which that flag wrote nothing is refused.
    value_of: str | None = None


def _unchanged(flag: str) -> _Row:
    return _Row(writes=(flag,))


_DROPPED = _Row()
_DROPPED_SWITCH = _Row(takes_value=False)

# Every flag the vLLM command of a cell may hold. A flag that is not here refuses the cell, so a new flag of
# the runner or of the Generator is never passed to SGLang unseen; --enforce-eager and --compilation-config
# are refused that way, because the launcher has nothing to answer them with.
#
# Deviation note (mixed chunk and the decode graph sizes are SGLang's defaults).
# What differs: AISim's generated SGLang serving deployment passes --enable-mixed-chunk and its own
# --cuda-graph-bs-decode list (src/aisim/generator/rule_plugin/sglang.rule). The cell passes neither: the
# launcher refuses mixed chunk (runtime/sglbench_launcher.py ``_RESOLVED_CHECKS``), and no row writes a graph
# list. The table is measured with mixed chunk off and with the decode graph sizes SGLang captures by itself,
# on every system.
# Direction of the error: not known.
# How to retire it: a launcher that covers mixed chunk, and a plan option that carries the serving graph list.
_TRANSLATION: dict[str, _Row] = {
    "--model": _Row(writes=("--model-path",)),
    "--served-model-name": _unchanged("--served-model-name"),
    _TENSOR_PARALLEL_FLAG: _Row(writes=("--tp-size",)),
    # SGLang's defaults are 1, and the launcher refuses any other resolved value.
    "--pipeline-parallel-size": _Row(value=_one_rank),
    "--data-parallel-size": _Row(value=_one_rank),
    # vLLM's switch carries no size: it spreads the experts of a MoE model over the ranks of the engine, and a
    # cell has one data-parallel replica, so those are its tensor-parallel ranks. SGLang takes the size
    # (``ep_size`` in ``fields/parallel.py``, default 1), so the switch is written as --ep-size with the
    # tensor-parallel size. AISim's generated SGLang serving deployment passes the same size under the alias
    # --expert-parallel-size (backend_config_mapping.yaml, ``moe_expert_parallel_size``). The Generator sets
    # the switch only for an expert-parallel size above 1 (src/aisim/generator/rule_plugin/vllm.rule), so a
    # one-GPU MoE cell writes no --ep-size and runs with SGLang's default. No other MoE flag has a row:
    # --kernel-config, --enable-eplb and --data-parallel-backend refuse the cell like any unknown flag.
    "--enable-expert-parallel": _Row(writes=("--ep-size",), takes_value=False, value_of=_TENSOR_PARALLEL_FLAG),
    "--kv-cache-dtype": _Row(writes=("--kv-cache-dtype",), value=_kv_cache_dtype),
    # The same pair as the Generator's (backend_config_mapping.yaml, ``kv_cache_free_gpu_memory_fraction``).
    # The number is passed as it is. In SGLang it is the share of the memory that the weights and the KV pool
    # get (``fields/schedule.py``).
    "--gpu-memory-utilization": _Row(writes=("--mem-fraction-static",)),
    # vLLM's worker start-up and request log. SGLang logs no request unless it is asked to.
    "--distributed-executor-backend": _DROPPED,
    "--distributed-timeout-seconds": _DROPPED,
    "--no-enable-log-requests": _DROPPED_SWITCH,
    # Never --disable-overlap-schedule: the launcher refuses a run without overlap scheduling
    # (runtime/sglbench_driver.py, the recorder headers).
    "--no-async-scheduling": _DROPPED_SWITCH,
    # Never --disable-radix-cache: the launcher refuses a run without the radix cache (``_RESOLVED_CHECKS``),
    # because every cached context of a point is real state in it.
    "--no-enable-prefix-caching": _DROPPED_SWITCH,
    _MODE_FLAG: _Row(writes=(_MODE_FLAG,), value=_benchmark_mode),
    "--benchmark-warmup-iterations": _unchanged("--benchmark-warmup-iterations"),
    _POINTS_FILE_FLAG: _unchanged(_POINTS_FILE_FLAG),
    "--benchmark-output-path": _unchanged("--benchmark-output-path"),
    "--dump-config-to": _unchanged("--dump-config-to"),
    _NEW_TOKEN_CAP_FLAG: _unchanged(_NEW_TOKEN_CAP_FLAG),
    "--prefill-max-kv-read-token-samples": _unchanged("--prefill-max-kv-read-token-samples"),
    "--prefix-max-batch-size-samples": _unchanged("--prefix-max-batch-size-samples"),
    "--decode-max-kv-read-token-samples": _unchanged("--decode-max-kv-read-token-samples"),
    "--decode-max-batch-size-samples": _unchanged("--decode-max-batch-size-samples"),
    "--benchmark-timeout": _Row(writes=("--benchmark-timeout",), value=_launcher_deadline),
    "--max-model-len": _Row(writes=("--context-length",), value=_context_length),
    # vLLM's token budget of one step. SGLang has two bounds for it: the chunk of one request and the new
    # tokens of one prefill batch.
    _TOKEN_BUDGET_FLAG: _Row(writes=("--chunked-prefill-size", "--max-prefill-tokens"), value=_token_budget),
    "--max-num-seqs": _Row(writes=("--max-running-requests",)),
}


def _translate_command(tokens: list[str]) -> list[str]:
    """The launcher's command for the vLLM worker command of the same cell, in the order of the vLLM flags.

    A flag may carry its value as the next token or after ``=``. A flag that is given twice is refused: the
    Generator and the two parsers each read the last one, and this module would have to guess the same. What a
    row with ``value_of`` writes stands directly after the flags of the flag it names, wherever the switch
    itself stands in the vLLM command.
    """
    if tuple(tokens[:3]) != _VLLM_COMMAND:
        raise _refused(f"the command starts with {shlex.join(tokens[:3])!r}, not with {shlex.join(_VLLM_COMMAND)!r}")
    given: dict[str, str | None] = {}
    # The tokens written for each vLLM flag, in the order of the vLLM flags, and the value they carry.
    written: dict[str, list[str]] = {}
    written_values: dict[str, str] = {}
    position = len(_VLLM_COMMAND)
    while position < len(tokens):
        flag, joined, value = tokens[position].partition("=")
        position += 1
        row = _TRANSLATION.get(flag)
        if row is None:
            raise _refused(f"the vLLM flag {flag} has no SGLang translation")
        if flag in given:
            raise _refused(f"{flag} is given twice")
        if not row.takes_value:
            if joined:
                raise _refused(f"{flag} takes no value, got {value!r}")
            given[flag] = None
            continue
        if not joined:
            if position == len(tokens) or tokens[position].startswith("--"):
                raise _refused(f"{flag} has no value")
            value = tokens[position]
            position += 1
        given[flag] = value
        translated = value if row.value is None else row.value(flag, value)
        if translated is not None and row.writes:
            written[flag] = [token for name in row.writes for token in (name, translated)]
            written_values[flag] = translated

    for flag, row in _TRANSLATION.items():
        if row.value_of is None or flag not in given:
            continue
        if row.value_of not in written:
            raise _refused(f"{flag} is written with the value of {row.value_of}, which the command does not give")
        size = written_values[row.value_of]
        written[row.value_of].extend(token for name in row.writes for token in (name, size))
    command = [*LAUNCHER_COMMAND, *(token for flag_tokens in written.values() for token in flag_tokens)]

    if _MODE_FLAG not in given:
        raise _refused(f"the command has no {_MODE_FLAG}")
    if given[_MODE_FLAG] == "prefill" and _POINTS_FILE_FLAG not in given and _NEW_TOKEN_CAP_FLAG not in given:
        budget = given.get(_TOKEN_BUDGET_FLAG)
        if budget is None:
            raise _refused(
                f"a prefill cell without {_NEW_TOKEN_CAP_FLAG} needs {_TOKEN_BUDGET_FLAG}, which bounds its "
                "new-token axis"
            )
        command.extend([_NEW_TOKEN_CAP_FLAG, str(max(_MIN_SAMPLE_CAP, int(budget)))])
    return command


def _check_real_state_switch(name: str, value: object, source: str) -> None:
    accepted = _REAL_STATE_SWITCHES[name]
    if not isinstance(value, str) or value.strip().lower() not in accepted:
        raise _refused(
            f"{name}={value!r} in {source}: the launcher stages real cached state only (accepted: "
            f"{', '.join(accepted)})"
        )


def _check_content_word(value: object, source: str, *, is_moe: bool) -> None:
    """Refuse a value of the content variable that the launcher would refuse in the pod."""
    if value == "":
        return
    if value not in (_SEEDED_CONTENT, _TEXT_CONTENT):
        raise _refused(
            f"{_CONTENT_VARIABLE}={value!r} in {source}: the launcher does not support it (accepted: "
            f"{_SEEDED_CONTENT}, {_TEXT_CONTENT})"
        )
    if value == _SEEDED_CONTENT and is_moe:
        raise _refused(
            f"{_CONTENT_VARIABLE}={_SEEDED_CONTENT} in {source}: the launcher does not support it for a model "
            "with expert layers, whose cells send dataset text"
        )


def _check_overrides(plan: FPMCollectionPlan, cell: FPMCell, generator_overrides: dict[str, Any]) -> None:
    """Refuse, before anything is rendered, what the deployment inputs of the cell already show."""
    # The backend policy of a cell is merged over the inputs of the campaign (runner.py
    # ``_cell_generator_overrides``).
    deployments = [
        source.get("K8sConfig") or {} for source in (cell.backend_policy.generator_overrides, generator_overrides)
    ]
    # On Slurm the image is --fpm-slurm-container-image and the manifest's is not used.
    if getattr(plan.options, "executor", "kubernetes") != "slurm" and not any(
        deployment.get("k8s_image") for deployment in deployments
    ):
        raise _refused("K8sConfig.k8s_image must name the SGLang image; the Generator's default is a Dynamo vLLM image")
    for deployment in deployments:
        for item in deployment.get("extra_env") or []:
            if not isinstance(item, dict):
                continue
            if item.get("name") in _REAL_STATE_SWITCHES:
                _check_real_state_switch(item["name"], item.get("value"), "K8sConfig.extra_env")
            elif item.get("name") == _CONTENT_VARIABLE:
                _check_content_word(item.get("value"), "K8sConfig.extra_env", is_moe=plan.capability.is_moe)


def _require_one_node(env_script: str) -> None:
    # The gate's mode without etcd has no multinode form and exits 2 for any other count
    # (runtime/fpm_exec.sh), after every node was scheduled.
    counts = re.findall(r"^export FPM_NODE_COUNT=(.*)$", env_script, flags=re.MULTILINE)
    if counts != ["1"]:
        raise _refused(
            f"the launcher runs on one node, and {FPM_ENV_FILENAME} exports {', '.join(counts) or 'nothing'} as "
            "FPM_NODE_COUNT"
        )


def _shell_word(quoted: str) -> str:
    """The value of an ``export NAME=<word>`` line of run.sh, which the Generator writes with ``shlex.quote``."""
    try:
        words = shlex.split(quoted)
    except ValueError:
        return quoted
    return words[0] if len(words) == 1 else quoted


def _require_known_array_lines(lines: list[str], command_line: int) -> None:
    """Refuse a ``run.sh`` that names the command array where the Generator's script of a one-node cell does not."""
    in_multinode_block = False
    for index, line in enumerate(lines):
        if line == _MULTINODE_BLOCK_START:
            in_multinode_block = True
        elif in_multinode_block and line == _BLOCK_END:
            in_multinode_block = False
        elif (
            index != command_line
            and not in_multinode_block
            and _COMMAND_ARRAY in line
            and not line.lstrip().startswith("#")
            and line not in _ONE_NODE_ARRAY_LINES
        ):
            raise _refused(
                f"{FPM_RUN_SCRIPT_FILENAME} names the command array in a line this builder does not know: {line}"
            )


def _sglang_run_script(run_script: str, *, is_moe: bool) -> str:
    """``run.sh`` with its one ``engine_command=(...)`` line translated and every other line as it was.

    ``is_moe`` says whether the model of the plan has expert layers; the content variable is tested with it.
    """
    # Deviation note (the hardware environment of the vLLM render is kept).
    # What differs: the lines that stay hold the environment the Generator renders for the hardware of the
    # cell, for example NCCL_CUMEM_ENABLE=1 on Hopper systems (src/aisim/generator/facts/hardware.yaml).
    # SGLang chooses that variable by itself only when it is unset, and then writes 0 unless symmetric
    # memory is on (python/sglang/srt/entrypoints/engine.py ``_set_envs_and_config``), so the cell runs with
    # the Generator's value where a plain SGLang start has SGLang's own. AISim's generated SGLang serving
    # deployment gets the same variables from the same hardware facts
    # (src/aisim/generator/builders/k8s_builder.py), so against that deployment the cell does not differ.
    # Direction of the error: not known against a plain SGLang start.
    # How to retire it: nothing to retire while the Generator gives its SGLang workers these variables; an
    # FPM target of the Generator that renders SGLang cells would write the same lines.
    lines = run_script.split("\n")
    found = [index for index, line in enumerate(lines) if line.startswith(_COMMAND_LINE_START)]
    if len(found) != 1:
        raise _refused(
            f"{FPM_RUN_SCRIPT_FILENAME} has {len(found)} lines that start with {_COMMAND_LINE_START}, expected one"
        )
    line = lines[found[0]]
    try:
        tokens = shlex.split(line[len(_COMMAND_LINE_START) : -1]) if line.endswith(")") else None
    except ValueError:
        tokens = None
    # Only the Generator's own form is read: a plain list of quoted words. Anything the shell would expand
    # is not a token list.
    if tokens is None or f"{_COMMAND_LINE_START}{shlex.join(tokens)})" != line:
        raise _refused(f"the command line of {FPM_RUN_SCRIPT_FILENAME} is not a plain list of quoted words: {line}")
    _require_known_array_lines(lines, found[0])
    last_line = run_script.rstrip("\n").rsplit("\n", 1)[-1]
    if last_line != _EXEC_LINE:
        raise _refused(
            f"the last line of {FPM_RUN_SCRIPT_FILENAME} does not start the command in its own process group: "
            f"{last_line}"
        )
    # The complete check: these lines also hold what a backend policy and the collector's defaults declared.
    for name in _REAL_STATE_SWITCHES:
        for quoted in re.findall(rf"^export {name}=(.*)$", run_script, flags=re.MULTILINE):
            _check_real_state_switch(name, _shell_word(quoted), FPM_RUN_SCRIPT_FILENAME)
    # The shell keeps the last export of a name, so that line holds the value the launcher reads. Two lines
    # arise when ``K8sConfig.worker_extra_pod_spec`` names the variable too: the Generator writes those
    # entries before the ones of ``K8sConfig.extra_env``.
    # The script is read line by line, as for the two switches above. A value that holds a line break is
    # therefore not read as the shell reads it: the Generator writes it over several lines, and a line of it
    # that looks like an export of the variable is taken for one. Such a cell can be refused here although
    # the launcher would run it, or pass here and be refused by the launcher in the pod.
    content_words = re.findall(rf"^export {_CONTENT_VARIABLE}=(.*)$", run_script, flags=re.MULTILINE)
    if content_words:
        _check_content_word(_shell_word(content_words[-1]), FPM_RUN_SCRIPT_FILENAME, is_moe=is_moe)
    lines[found[0]] = f"{_COMMAND_LINE_START}{shlex.join(_translate_command(tokens))})"
    return "\n".join(lines)


def _without_working_dir(manifest: str) -> str:
    """The resource Pod without its container's ``workingDir``, every other line as it was."""
    lines = manifest.splitlines(keepends=True)
    found = [index for index, line in enumerate(lines) if re.match(r" +workingDir: \S", line)]
    if len(found) != 1:
        raise _refused(f"{FPM_MANIFEST_FILENAME} has {len(found)} workingDir lines, expected one")
    del lines[found[0]]
    return "".join(lines)


def render_cell(
    plan: FPMCollectionPlan,
    cell: FPMCell,
    cell_dir: Path,
    generator_overrides: dict[str, Any],
    *,
    smoke: bool = False,
    render_vllm_cell: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Render one SGLang cell into ``cell_dir`` and return the artifacts, as ``runner._render_cell`` does.

    ``render_vllm_cell`` is ``runner._render_cell``. It gets the same plan with the backend ``vllm``; the
    plan's hash is kept, so the labels of the resource Pod carry the hash of the SGLang plan.
    """
    _check_overrides(plan, cell, generator_overrides)
    artifacts = render_vllm_cell(
        dataclasses.replace(plan, backend="vllm"), cell, cell_dir, generator_overrides, smoke=smoke
    )
    run_path = cell_dir / FPM_RUN_SCRIPT_FILENAME
    manifest_path = cell_dir / FPM_MANIFEST_FILENAME
    runtime_env_path = cell_dir / _RUNTIME_ENV_FILENAME

    # Every check passes before the first file is written, so the directory never holds a mix of the two.
    _require_one_node((cell_dir / FPM_ENV_FILENAME).read_text())
    run_script = _sglang_run_script(run_path.read_text(), is_moe=plan.capability.is_moe)
    manifest = _without_working_dir(manifest_path.read_text())
    runtime_env = runtime_env_path.read_text()
    if not runtime_env.endswith("\n"):
        runtime_env += "\n"
    runtime_env += f"export {GATE_SWITCH_ENV}=1\n"

    run_path.write_text(run_script)
    manifest_path.write_text(manifest)
    runtime_env_path.write_text(runtime_env)
    return {**artifacts, FPM_RUN_SCRIPT_FILENAME: run_script, FPM_MANIFEST_FILENAME: manifest}
