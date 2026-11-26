# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# The dataset record in the kvwarm block of a result file (FAKE_DATASET) has the key names of ai-dynamo/dynamo
# (Apache-2.0, https://github.com/ai-dynamo/dynamo), components/src/dynamo/vllm/instrumented_scheduler.py at
# b20fe6fe94021b854597965c7305931175e8f482. See THIRD_PARTY_NOTICES.md.

"""MoE cells of an SGLang self-benchmark collection, seen from the host.

One packaged MoE model is planned on one and two GPUs through the functions of the entry point, with the TEP
preset or with the default preset, which also plans the DEP cell of two data-parallel ranks and the pure TP
cell of two tensor-parallel ranks. Its capability, the memory estimate of each topology and its backend policy
are therefore the ones a user gets. The cells are rendered by the real Generator and the real cell builder, a
cluster double leaves the files of a launcher run, and the real validator, row builder and table writer read
them.

The rule that holds the result files of a dense plan against their cells has its tests here too, beside the
rule of a MoE plan: one packaged dense model is planned the same way, with a one-GPU cell and a TP cell of two
GPUs, and its result files are typed in like the others.

So has the rule that holds the dtype of the engine's KV pool against the KV-cache label of the cell, which is
asked of every SGLang rank file: both models are planned with bfloat16 cells, with fp8 cells and with both, and
the two keys of the producer block are typed in like the rest of it. No engine stored a KV cache for them. For
the label nvfp4, which no plan carries, a planned cell is relabelled.

No SGLang, no GPU and no cluster are involved, and a fixture counts every attempt to open a network
connection. Every result file is a FAKE of ``test_fpm_sglbench_artifact.py``: its times were typed in and
never measured, and the dataset it names was never read. That holds for the two rank files of a DEP cell and
for the file of a pure TP cell too: no launcher run with DP attention or with pure TP produced them.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import re
import shlex
import socket
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from collector.fpm_forward import cli, native_artifact, planner, runner
from collector.fpm_forward.database import aggregate_cell
from collector.fpm_forward.entry import resolve_inputs
from collector.fpm_forward.native_artifact import plan_is_moe, validate_native_collection
from collector.fpm_forward.runtime import sglbench_artifact as artifact
from collector.fpm_forward.runtime import sglbench_content as content
from collector.fpm_forward.runtime_memory import saved_plan_identity
from collector.fpm_forward.types import ParallelTopology
from collector.model_cases import build_collection_case_plan

from .test_fpm_sglbench_artifact import (
    EXPECTED_KVWARM,
    POD_KV_CACHE,
    POD_SGLANG_VERSION,
    _points,
    _rank_artifacts,
)

pytestmark = pytest.mark.unit

MODEL = "Qwen/Qwen3-30B-A3B"
SYSTEM = "h200_sxm"
FAKE_IMAGE = "example.invalid/sglang:test"
# AISim's data holds the operator tables of this SGLang version for the system, so the planner gets a memory
# estimate for it.
LABEL_WITH_TABLES = "0.5.21"

# What a text run records about its dataset in the kvwarm block of a decode artifact: the four keys of Dynamo's
# vLLM self-benchmark. Every value is FAKE.
FAKE_DATASET = {
    "path": "/models/fpm_datasets/fake-conversations.json",
    "sha256": "0123456789abcdef" * 4,
    "collection_pool": "conversation_sha256_even",
    "conversations": 3,
}


def _fake_producer(cell) -> dict:
    """The producer block of a text run of ``cell``, under the names the launcher and the driver write.

    The host reads the release, the dtype of the KV pool and, for a cell of a MoE plan, the content source and
    the three sizes (``native_artifact._validate_sglang_moe_producer``). The sizes are the ones SGLang resolves
    for the cell: its ``tp_size`` counts the ranks of every data-parallel rank. The two keys of the KV cache
    are those of the KV-cache label of the cell (``POD_KV_CACHE``). Every other value is FAKE and has no reader.
    """
    ranks = cell.topology.tp * cell.topology.dp
    producer = {
        "sglang_version": POD_SGLANG_VERSION,
        **POD_KV_CACHE[cell.kv_cache_dtype],
        "tp_size": ranks,
        "attn_dp_size": cell.topology.dp,
        # One scheduler process per rank of the engine.
        "scheduler_processes": ranks,
        "staging": "one_batch_call" if cell.topology.dp == 1 else "balanced_batch_call",
        "expert_keys": {"num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 768},
        "moe": {
            "ep_size": cell.topology.moe_ep,
            "moe_a2a_backend": "none",
            "moe_runner_backend": "auto",
            "moe_dense_tp_size": None,
            "enable_eplb": False,
        },
        "content": {
            "source": "dataset_text_chains",
            "dataset": {**FAKE_DATASET, "url": None, "downloaded": False},
            "pool": {
                "id_range": [0, 151936],
                "conversations": 3,
                "tokens": 4096,
                "dropped_tokens": 2,
                "dropped_conversations": 0,
            },
        },
    }
    if cell.topology.dp > 1:
        producer["dp_attention"] = {
            "local_control_broadcast": False,
            "input_blocker": True,
            "placement": "round_robin_checked",
        }
    return producer


@pytest.fixture(autouse=True)
def _no_git_and_no_network(monkeypatch):
    """Plan without a git checkout, and fail the test when anything tries to open a network connection."""
    attempts = []

    def refuse(*arguments, **_keywords):
        attempts.append(arguments[-1] if arguments else None)
        raise OSError("this suite has no network")

    monkeypatch.setattr(planner, "_git_revision", lambda: "test-revision")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield
    assert attempts == []


def _arguments(*extra: str, presets: tuple[str, ...] = ("tep",)) -> list[str]:
    """What a user passes to ``python -m collector.fpm_forward`` for the MoE cells of one and two GPUs.

    ``presets`` are the values of ``--fpm-parallel-presets``. Without one the option is left out and the
    default preset applies, which plans the one-GPU cell, the DEP cell, the pure TP cell and the TEP cell.
    """
    return [
        *("--backend", "sglang", "--model-path", MODEL, "--gpu", SYSTEM),
        *("--fpm-max-gpus", "2", "--fpm-gpu-counts", "1", "2"),
        *(("--fpm-parallel-presets", *presets) if presets else ()),
        *("--fpm-prefill-cudagraph-policy", "runtime"),
        *("--fpm-runtime-backend-version", POD_SGLANG_VERSION, "--fpm-backend-version", LABEL_WITH_TABLES),
        *("--generator-set", f"K8sConfig.k8s_image={FAKE_IMAGE}"),
        *extra,
    ]


def _resolved(*extra: str, presets: tuple[str, ...] = ("tep",)):
    """The plan and the deployment inputs as the entry point resolves them (cli.py ``main``)."""
    arguments = cli._parser().parse_args(_arguments(*extra, presets=presets))
    case_plan = build_collection_case_plan(
        backend=arguments.backend,
        model_path=arguments.model_path,
        model_architecture=arguments.model_architecture,
        gpu_type=arguments.gpu,
        sm_version=arguments.sm,
        model_cases_path=arguments.model_cases,
    )
    return resolve_inputs(arguments, case_plan)


def _cell(plan, phase: str, strategy: str):
    (cell,) = [cell for cell in plan.cells if cell.workload_kind == phase and cell.parallel_strategy == strategy]
    return cell


def _launcher_files(cell, producer: dict | None = None) -> list[dict]:
    """The rank artifacts a launcher run leaves for ``cell``: one per data-parallel rank, whatever the TP size.

    A decode artifact names the dataset in its kvwarm block, as a run with dataset text does. A prefill
    artifact has no such block. ``producer`` replaces the producer block of every rank.
    """
    phase = cell.workload_kind
    keywords = {"producer": _fake_producer(cell) if producer is None else producer}
    if phase == "decode":
        keywords["kvwarm_dataset"] = dict(FAKE_DATASET)
    return _rank_artifacts(phase, _points(phase), dp=cell.topology.dp, **keywords)


def _pod(root: Path, plan, cell) -> Path:
    """The folder of the one pod of ``cell`` under ``root``, with the provenance file the collector writes."""
    pod = root / "raw" / "pod-0"
    pod.mkdir(parents=True)
    provenance = {
        "schema_name": "aic_fpm_collector_provenance",
        "schema_version": 1,
        "cell_id": cell.cell_id,
        "plan_sha256": plan.sha256,
        "attempt_id": "attempt",
        "runtime": {"backend": "sglang", "backend_version": POD_SGLANG_VERSION},
    }
    (pod / "collector-provenance.json").write_text(json.dumps(provenance))
    return pod


def _collected(root: Path, plan, cell, payloads: list[dict]) -> dict:
    """Write ``payloads`` as the launcher does, into the pod folder of ``cell``. Returns the paths."""
    return artifact.write_artifacts(str(_pod(root, plan, cell) / "benchmark.json"), payloads)


def _written_past_the_writer(root: Path, plan, cell, payloads: list[dict]) -> list[Path]:
    """Write ``payloads`` under the names of the launcher's writer, which is not asked whether it would take them.

    No merged file is written: the validator does not read one.
    """
    pod = _pod(root, plan, cell)
    paths = [Path(artifact.rank_output_path(str(pod / "benchmark.json"), rank)) for rank in range(len(payloads))]
    for rank_path, payload in zip(paths, payloads, strict=True):
        artifact.atomic_write_json(str(rank_path), payload)
    return paths


def _validate(plan, cell, root: Path, **keywords):
    """Validate as the runner and the row builder do: with the backend of the plan and what it says of the model."""
    keywords.setdefault("expected_moe", plan_is_moe(plan))
    return validate_native_collection(
        cell,
        root / "raw",
        expected_plan_sha256=plan.sha256,
        expected_attempt_id="attempt",
        expected_backend="sglang",
        **keywords,
    )


# ---------------------------------------------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------------------------------------------


def test_the_default_preset_of_the_moe_model_plans_a_single_a_dep_a_pure_tp_and_a_tep_cell(monkeypatch, capsys):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL)
    arguments = _arguments("--plan-only", presets=())
    assert "--fpm-parallel-presets" not in arguments

    assert cli.main(arguments) == 0

    printed = json.loads(capsys.readouterr().out)
    # Per phase: the one-GPU cell, then on two GPUs the DEP cell (one GPU and one expert-parallel rank per
    # data-parallel rank), the pure TP cell (a slice of every expert on each tensor-parallel rank) and the TEP
    # cell (one expert-parallel rank per tensor-parallel rank).
    per_phase = [
        ("single", {"tp": 1, "pp": 1, "dp": 1, "moe_tp": 1, "moe_ep": 1, "cp": 1}),
        ("dep", {"tp": 1, "pp": 1, "dp": 2, "moe_tp": 1, "moe_ep": 2, "cp": 1}),
        ("pure_tp", {"tp": 2, "pp": 1, "dp": 1, "moe_tp": 2, "moe_ep": 1, "cp": 1}),
        ("tep", {"tp": 2, "pp": 1, "dp": 1, "moe_tp": 1, "moe_ep": 2, "cp": 1}),
    ]
    assert [(cell["workload_kind"], cell["parallel_strategy"], cell["topology"]) for cell in printed["cells"]] == [
        (phase, strategy, topology) for phase in ("prefill", "decode") for strategy, topology in per_phase
    ]
    assert [(topology["strategy"], topology["dp"]) for topology in printed["topologies"]] == [
        ("single", 1),
        ("dep", 2),
        ("pure_tp", 1),
        ("tep", 1),
    ]
    # Memory admission estimated the four topologies for this backend and found each below the capacity of one
    # GPU. A rank of each two-GPU cell holds half of the expert weights, as half of the experts or as half of
    # every expert, so it needs less than the one-GPU cell.
    admission = printed["topology_memory_admission"]
    assert [(decision["topology"]["dp"], decision["disposition"]) for decision in admission] == [
        (1, "admitted"),
        (2, "admitted"),
        (1, "admitted"),
        (1, "admitted"),
    ]
    single, dep, pure_tp, tep = (decision["estimates"][0] for decision in admission)
    for two_gpu_cell in (dep, pure_tp, tep):
        assert 0 < two_gpu_cell["estimated_non_kv_bytes"] < single["estimated_non_kv_bytes"]
    assert single["estimated_non_kv_bytes"] < single["gpu_capacity_bytes"]
    assert (printed["counts"]["memory_rejected_topologies"], printed["counts"]["memory_unknown_topologies"]) == (0, 0)
    identity = saved_plan_identity(printed)
    assert (identity.backend, identity.sha256, len(identity.cells)) == ("sglang", printed["sha256"], 8)
    # The one-GPU and the TEP cells are the cells of the plan with the TEP preset.
    tep_plan, _overrides = _resolved()
    assert {cell.cell_id for cell in tep_plan.cells} == {
        cell["cell_id"] for cell in printed["cells"] if cell["parallel_strategy"] in ("single", "tep")
    }


def test_the_pure_tp_preset_and_the_tep_and_dep_presets_share_out_the_cells_of_the_default_preset():
    default_plan, _overrides = _resolved(presets=())

    def cells_of_the_default_plan(*strategies: str) -> list:
        return [cell for cell in default_plan.cells if cell.parallel_strategy in strategies]

    # --fpm-parallel-presets pure_tp: the one-GPU cell and the pure TP cell of each phase.
    pure_tp_plan, _overrides = _resolved(presets=("pure_tp",))
    assert list(pure_tp_plan.cells) == cells_of_the_default_plan("single", "pure_tp")
    # --fpm-parallel-presets tep dep: every other cell, which is what the default preset planned before pure TP
    # was admitted for this backend.
    tep_and_dep_plan, _overrides = _resolved(presets=("tep", "dep"))
    assert list(tep_and_dep_plan.cells) == cells_of_the_default_plan("single", "dep", "tep")
    # The cells are the same cells under each choice of presets, with the same ids, and each choice is another
    # plan.
    assert len({default_plan.sha256, pure_tp_plan.sha256, tep_and_dep_plan.sha256}) == 3


# ---------------------------------------------------------------------------------------------------------------
# Validation of one cell
# ---------------------------------------------------------------------------------------------------------------


def test_the_files_of_one_data_parallel_rank_are_no_result_of_a_dep_cell(tmp_path):
    plan, _overrides = _resolved()
    tep = _cell(plan, "decode", "tep")
    # The DEP cell of the same two GPUs, under the id of the TEP cell: two data-parallel ranks.
    dep = dataclasses.replace(
        tep, parallel_strategy="dep", topology=ParallelTopology(tp=1, pp=1, dp=2, moe_tp=1, moe_ep=2, cp=1)
    )
    _collected(tmp_path, plan, tep, _launcher_files(tep))

    assert len(_validate(plan, tep, tmp_path).points) == len(_points("decode"))
    with pytest.raises(ValueError, match="native DP metadata mismatch"):
        _validate(plan, dep, tmp_path)


# ---------------------------------------------------------------------------------------------------------------
# What the pod recorded, held against the cell
# ---------------------------------------------------------------------------------------------------------------


def _not_text(found: str) -> str:
    return (
        "native SGLang rank artifact of a MoE plan did not send dataset text: "
        f"expected producer.content.source 'dataset_text_chains', found {found}: "
    )


def _not_the_layout(tp: int, ep: int, found_tp: str, found_ep: str, *, dp: int = 1, found_dp: str = "1") -> str:
    """The refusal of a layout. ``tp`` is SGLang's ``tp_size`` for the cell: every rank of the engine."""
    return (
        "native SGLang rank artifact of a MoE plan did not run the layout of its cell: "
        f"expected producer.tp_size {tp}, producer.moe.ep_size {ep} and producer.attn_dp_size {dp}, "
        f"found {found_tp}, {found_ep} and {found_dp}: "
    )


def _no_expert_layers_found(producer: dict) -> None:
    """What a pod records whose SGLang finds no expert count in the model config: no keys and no moe block."""
    producer.update(expert_keys={}, moe=None)


def _seeded_ids(producer: dict) -> None:
    """Such a pod without a content word: the launcher sends seeded token ids."""
    _no_expert_layers_found(producer)
    producer["content"] = {"source": "seeded_token_ids", "seed": "0", "token_range": [1000, 2000], "issued": 7}


def _dep_refusal(found_tp: str, found_ep: str, found_dp: str) -> str:
    """The refusal of a layout for the DEP cell of two ranks, for which SGLang resolves 2, 2 and 2."""
    return _not_the_layout(2, 2, found_tp, found_ep, dp=2, found_dp=found_dp)


# name: (the strategy of the cell, the change to the producer block, the refusal)
DIFFERS_FROM_THE_CELL = {
    # The pod found no expert layers, although the plan's config has them.
    "seeded-ids": ("single", _seeded_ids, _not_text("'seeded_token_ids'")),
    "no-content-block": ("single", lambda producer: producer.pop("content"), _not_text("None")),
    # Text was asked for by name in such a pod: the content is text, and no layout was recorded.
    "text-without-expert-layers": ("single", _no_expert_layers_found, _not_the_layout(1, 1, "1", "None")),
    # The command reached the pod without --ep-size: every rank holds a slice of every expert. That is the layout
    # of the pure TP cell of the same GPUs, which the launcher accepts, and it is no result of the TEP cell.
    "tep-without-the-expert-parallel-size": (
        "tep",
        lambda producer: producer["moe"].update(ep_size=1),
        _not_the_layout(2, 2, "2", "1"),
    ),
    "single-on-two-ranks": ("single", lambda producer: producer.update(tp_size=2), _not_the_layout(1, 1, "2", "1")),
    # True equals 1 and is no size.
    "a-bool": ("single", lambda producer: producer["moe"].update(ep_size=True), _not_the_layout(1, 1, "1", "True")),
    # The attention-DP size of a cell with one data-parallel rank must be 1. A pod of the TEP cell that ran DP
    # attention ran another cell on the same GPUs.
    "tep-run-with-dp-attention": (
        "tep",
        lambda producer: producer.update(attn_dp_size=2),
        _not_the_layout(2, 2, "2", "2", found_dp="2"),
    ),
    # A block without the key is no record of one rank: the launcher writes the size into every result file,
    # with 1 for a run without DP attention.
    "tep-without-an-attention-dp-size": (
        "tep",
        lambda producer: producer.pop("attn_dp_size"),
        _not_the_layout(2, 2, "2", "2", found_dp="None"),
    ),
    # The DEP cell of two ranks. SGLang's tp_size counts both ranks, so the sizes of the cell are 2, 2 and 2.
    # The command reached the pod without --attn-dp-size: two tensor-parallel ranks of one scheduler group.
    "dep-run-without-dp-attention": (
        "dep",
        lambda producer: producer.update(attn_dp_size=1),
        _dep_refusal("2", "2", "1"),
    ),
    # A block without the key holds no attention-DP size, in a DEP cell as in any other.
    "dep-without-an-attention-dp-size": (
        "dep",
        lambda producer: producer.pop("attn_dp_size"),
        _dep_refusal("2", "2", "None"),
    ),
    "dep-without-the-expert-parallel-size": (
        "dep",
        lambda producer: producer["moe"].update(ep_size=1),
        _dep_refusal("2", "1", "2"),
    ),
    # The sizes as vLLM writes them for the cell: one tensor-parallel rank in each data-parallel rank.
    "dep-with-the-tensor-parallel-size-of-one-rank": (
        "dep",
        lambda producer: producer.update(tp_size=1),
        _dep_refusal("1", "2", "2"),
    ),
    # The pure TP cell of two GPUs, for which SGLang resolves 2, 1 and 1. The command reached the pod with the
    # --ep-size of the TEP cell: one expert-parallel rank per tensor-parallel rank. The launcher accepts that
    # layout too, so this comparison alone tells the two cells apart.
    "pure-tp-run-with-the-expert-parallel-size-of-the-tep-cell": (
        "pure_tp",
        lambda producer: producer["moe"].update(ep_size=2),
        _not_the_layout(2, 1, "2", "2"),
    ),
}


@pytest.mark.parametrize("name", list(DIFFERS_FROM_THE_CELL))
def test_a_moe_result_that_differs_from_its_cell_is_refused(tmp_path, name):
    strategy, change, message = DIFFERS_FROM_THE_CELL[name]
    phase = "decode"
    # The DEP cell and the pure TP cell are cells of the plan with the default preset.
    plan, _overrides = _resolved(presets=()) if strategy in ("dep", "pure_tp") else _resolved()
    assert plan_is_moe(plan) is True
    cell = _cell(plan, phase, strategy)
    # One file per data-parallel rank, each with the same change.
    payloads = _launcher_files(cell)
    assert len(payloads) == cell.topology.dp
    for payload in payloads:
        change(payload["producer"])
    # Written past the launcher's own writer, which is not asked whether it would take the block.
    _written_past_the_writer(tmp_path, plan, cell, payloads)
    refusal = "^" + re.escape(message)

    with pytest.raises(ValueError, match=refusal):
        _validate(plan, cell, tmp_path)
    # The row builder and the summary the runner keeps make the same comparison, from the plan.
    with pytest.raises(ValueError, match=refusal):
        aggregate_cell(plan, cell, tmp_path, expected_attempt_id="attempt")
    with pytest.raises(ValueError, match=refusal):
        runner._runtime_collection_summary(cell, tmp_path / "raw", backend="sglang", is_moe=plan_is_moe(plan))

    # A caller without a plan does not make the comparison: every other rule accepts the files. The plan of a
    # model without expert layers makes a comparison of its own, which has its tests further down.
    collection = _validate(plan, cell, tmp_path, expected_moe=None)
    assert len(collection.points) == len(_points(phase))
    summary = runner._runtime_collection_summary(cell, tmp_path / "raw", backend="sglang")
    assert summary["measured_point_count"] == len(_points(phase))


def test_the_text_source_the_validator_expects_is_the_one_the_content_plan_writes():
    # The validator does not import the runtime modules, so the two values are held together here.
    assert native_artifact._SGLANG_TEXT_CONTENT_SOURCE == content.CONTENT_TEXT_CHAINS == "dataset_text_chains"


# ---------------------------------------------------------------------------------------------------------------
# What the pod recorded, held against the cell of a dense plan
# ---------------------------------------------------------------------------------------------------------------

# A packaged model without expert layers. Its plan holds tensor-parallel cells only.
DENSE_MODEL = "Qwen/Qwen3-0.6B"


def _dense_plan():
    """The plan of the dense model on one and two GPUs: per phase the one-GPU cell and the TP cell of two GPUs.

    The model is given once more after the one of ``_arguments``, and the parser keeps the last. No preset is
    named: the default preset of a dense model is tensor parallelism.
    """
    plan, _overrides = _resolved("--model-path", DENSE_MODEL, presets=())
    assert (plan.model_path, plan_is_moe(plan)) == (DENSE_MODEL, False)
    cells = [(cell.workload_kind, cell.parallel_strategy, cell.topology.tp, cell.topology.dp) for cell in plan.cells]
    assert cells == [
        (phase, strategy, tp, 1) for phase in ("prefill", "decode") for strategy, tp in (("single", 1), ("tp", 2))
    ]
    return plan


def _fake_dense_producer(cell) -> dict:
    """The producer block of a run with seeded ids of ``cell``, under the names the launcher and the driver write.

    For a cell of a dense plan the host reads the release, the dtype of the KV pool and three keys of the
    launcher (``native_artifact._validate_sglang_dense_producer``): SGLang's ``tp_size``, its ``attn_dp_size``
    and the ``moe`` block, which is null when the launcher finds no expert key. The two keys of the KV cache
    are those of the KV-cache label of the cell (``POD_KV_CACHE``). Every other value is FAKE and has no
    reader.
    """
    return {
        "sglang_version": POD_SGLANG_VERSION,
        **POD_KV_CACHE[cell.kv_cache_dtype],
        "tp_size": cell.topology.tp,
        "attn_dp_size": 1,
        "scheduler_processes": cell.topology.tp,
        "staging": "one_batch_call",
        "expert_keys": {},
        "moe": None,
        "content": {"source": "seeded_token_ids", "seed": "0", "token_range": [1000, 2000], "issued": 7},
    }


def _dense_files(cell, change=None) -> list[dict]:
    """The one rank artifact a launcher run with seeded ids leaves for ``cell``: its kvwarm block names no dataset.

    ``change`` changes the producer block.
    """
    producer = _fake_dense_producer(cell)
    if change is not None:
        change(producer)
    return _rank_artifacts(cell.workload_kind, _points(cell.workload_kind), producer=producer)


def _expert_layer_files(cell, change=None) -> list[dict]:
    """The one rank artifact of a pod of ``cell`` whose launcher finds expert layers where the plan finds none.

    The two keys are among the six the launcher looks up (sglbench_compat.py, ``EXPERT_KEYS``) and not among the
    three counts the plan reads (capabilities.py, ``_is_moe``). Such a pod sends dataset text and records a
    ``moe`` block with the expert-parallel size SGLang resolved: 1 for the command of a dense cell, which has
    no ``--ep-size``. ``change`` changes the producer block.
    """
    producer = _fake_producer(cell)
    producer["expert_keys"] = {"moe_num_experts": 8, "num_experts_per_tok": 2}
    assert (producer["tp_size"], producer["attn_dp_size"], producer["moe"]["ep_size"]) == (cell.topology.tp, 1, 1)
    if change is not None:
        change(producer)
    return _launcher_files(cell, producer)


def _tep_on(ranks: int):
    """The change that records the TEP layout of ``ranks`` GPUs: whole experts on every tensor-parallel rank."""

    def change(producer: dict) -> None:
        producer["tp_size"] = ranks
        producer["moe"]["ep_size"] = ranks

    return change


def _not_the_dense_sizes(tp: int, found_tp: str, found_dp: str = "1") -> str:
    """The refusal of the two sizes. ``tp`` is the tensor-parallel size of the cell."""
    return (
        "native SGLang rank artifact of a dense plan did not run the layout of its cell: "
        f"expected producer.tp_size {tp} and producer.attn_dp_size 1, found {found_tp} and {found_dp}: "
    )


def _not_without_expert_parallelism(found: str) -> str:
    """The refusal of the ``moe`` block. ``found`` is what the refusal says of the block."""
    return (
        "native SGLang rank artifact of a dense plan did not run the layout of its cell: "
        f"expected producer.moe null or producer.moe.ep_size 1, found {found}: "
    )


@pytest.mark.parametrize(
    ("strategy", "phase", "files", "ep_size"),
    [
        ("tp", "prefill", _dense_files, None),
        ("single", "decode", _dense_files, None),
        # The launcher took the model for one with expert layers and ran it with SGLang's default: every
        # tensor-parallel rank holds a slice of every expert. That is tensor parallelism alone, so the files pass.
        ("tp", "decode", _expert_layer_files, 1),
    ],
    ids=["tp-prefill", "single-decode", "a-moe-block-with-the-expert-parallel-size-1"],
)
def test_the_files_of_a_dense_cell_are_accepted_with_the_sizes_of_the_cell(tmp_path, strategy, phase, files, ep_size):
    plan = _dense_plan()
    cell = _cell(plan, phase, strategy)
    payloads = files(cell)
    recorded = payloads[0]["producer"]
    assert (recorded["tp_size"], recorded["attn_dp_size"]) == (cell.topology.tp, 1)
    # No block of expert layers, or one with the expert-parallel size 1.
    assert (recorded["moe"] and recorded["moe"]["ep_size"]) == ep_size
    _collected(tmp_path, plan, cell, payloads)

    # ``_validate`` passes what the plan says of the model, which is False here.
    assert len(_validate(plan, cell, tmp_path).points) == len(_points(phase))
    # The row builder and the summary the runner keeps make the same comparison, from the plan.
    rows = aggregate_cell(plan, cell, tmp_path, expected_attempt_id="attempt")
    assert [(row["parallel_strategy"], row["tp"]) for row in rows] == [(strategy, cell.topology.tp)] * len(rows)
    summary = runner._runtime_collection_summary(cell, tmp_path / "raw", backend="sglang", is_moe=plan_is_moe(plan))
    assert summary["measured_point_count"] == len(rows) == len(_points(phase))


# name: (the strategy of the cell, the files of the pod, the change to their producer block, the refusal)
DIFFERS_FROM_THE_DENSE_CELL = {
    # The command reached the pod with another --tp-size, or without one: the launcher runs the size SGLang
    # resolved to the end, and the times are those of another number of GPUs.
    "tp-on-one-rank": ("tp", _dense_files, lambda producer: producer.update(tp_size=1), _not_the_dense_sizes(2, "1")),
    "single-on-two-ranks": (
        "single",
        _dense_files,
        lambda producer: producer.update(tp_size=2),
        _not_the_dense_sizes(1, "2"),
    ),
    # A block without the key is no record of the size: the launcher writes it into every result file.
    "no-tensor-parallel-size": (
        "tp",
        _dense_files,
        lambda producer: producer.pop("tp_size"),
        _not_the_dense_sizes(2, "None"),
    ),
    # True equals 1 and is no size.
    "a-bool": ("single", _dense_files, lambda producer: producer.update(tp_size=True), _not_the_dense_sizes(1, "True")),
    # A cell of a dense plan has one data-parallel rank, so the attention-DP size must be 1.
    "an-attention-dp-size-of-2": (
        "tp",
        _dense_files,
        lambda producer: producer.update(attn_dp_size=2),
        _not_the_dense_sizes(2, "2", "2"),
    ),
    # The launcher found expert layers and the command reached the pod with an --ep-size of the tensor-parallel
    # size: the TEP layout, which the launcher accepts and no cell of a dense plan describes.
    "a-tep-run": (
        "tp",
        _expert_layer_files,
        _tep_on(2),
        _not_without_expert_parallelism("producer.moe.ep_size 2"),
    ),
    # A moe block that is no mapping says nothing of the layout.
    "a-moe-block-that-is-no-mapping": (
        "single",
        _expert_layer_files,
        lambda producer: producer.update(moe=[1]),
        _not_without_expert_parallelism("producer.moe.ep_size None"),
    ),
    # A block without the key is no record of a model without expert layers: the launcher writes the key into
    # every result file, with null for such a model.
    "no-moe-key": (
        "tp",
        _dense_files,
        lambda producer: producer.pop("moe"),
        _not_without_expert_parallelism("no producer.moe"),
    ),
}


@pytest.mark.parametrize("name", list(DIFFERS_FROM_THE_DENSE_CELL))
def test_a_dense_result_that_differs_from_its_cell_is_refused(tmp_path, name):
    strategy, files, change, message = DIFFERS_FROM_THE_DENSE_CELL[name]
    phase = "decode"
    plan = _dense_plan()
    cell = _cell(plan, phase, strategy)
    # Written past the launcher's own writer, which is not asked whether it would take the block.
    (path,) = _written_past_the_writer(tmp_path, plan, cell, files(cell, change))

    with pytest.raises(ValueError) as refused:
        _validate(plan, cell, tmp_path)
    assert str(refused.value) == message + str(path)
    # The row builder and the summary the runner keeps make the same comparison, from the plan.
    with pytest.raises(ValueError, match="^" + re.escape(message)):
        aggregate_cell(plan, cell, tmp_path, expected_attempt_id="attempt")
    with pytest.raises(ValueError, match="^" + re.escape(message)):
        runner._runtime_collection_summary(cell, tmp_path / "raw", backend="sglang", is_moe=plan_is_moe(plan))

    # A caller without a plan does not make the comparison: every other rule accepts the files.
    assert len(_validate(plan, cell, tmp_path, expected_moe=None).points) == len(_points(phase))
    summary = runner._runtime_collection_summary(cell, tmp_path / "raw", backend="sglang")
    assert summary["measured_point_count"] == len(_points(phase))


# ---------------------------------------------------------------------------------------------------------------
# The dtype of the engine's KV pool, held against the KV-cache label of the cell
# ---------------------------------------------------------------------------------------------------------------

# The stored dtypes that each label with a rule stands for, as the refusal words them, and for ``nvfp4`` the
# resolved argument it wants beside the dtype. Typed here on their own.
_STANDS_FOR = {
    "bfloat16": "'torch.bfloat16' or 'torch.float16'",
    "fp8": "'torch.float8_e4m3fn'",
    "nvfp4": "'torch.float4_e2m1fn_x2' with producer.kv_cache_dtype 'nvfp4'",
}


def _not_the_kv_cache(label: str, found: str, resolved: str) -> str:
    """The refusal of a stored dtype. ``found`` and ``resolved`` are what the refusal says of the two keys."""
    return (
        "native SGLang rank artifact did not store the KV cache of its cell: the cell's kv_cache_dtype "
        f"{label!r} stands for producer.kv_cache_stored_dtype {_STANDS_FOR[label]}, found {found} "
        f"(producer.kv_cache_dtype {resolved}): "
    )


def _no_rule_for_the_label(label: str, found: str, resolved: str) -> str:
    """The refusal of a cell whose KV-cache label no rule knows."""
    return (
        "native SGLang rank artifact cannot be held against the KV cache of its cell: no rule says which "
        f"producer.kv_cache_stored_dtype the cell's kv_cache_dtype {label!r} stands for, found {found} "
        f"(producer.kv_cache_dtype {resolved}): "
    )


def _plan_of(kind: str, *kv_cache_dtypes: str):
    """The dense plan or the MoE plan of this suite, with ``--fpm-kv-cache-dtypes`` when dtypes are named."""
    option = ("--fpm-kv-cache-dtypes", *kv_cache_dtypes) if kv_cache_dtypes else ()
    if kind == "dense":
        plan, _overrides = _resolved("--model-path", DENSE_MODEL, *option, presets=())
    else:
        plan, _overrides = _resolved(*option)
    assert plan_is_moe(plan) is (kind == "moe")
    return plan


def _two_gpu_cell(plan, phase: str, label: str):
    """The cell of two GPUs of ``plan`` with this phase and KV-cache label: the TP cell or the TEP cell."""
    (cell,) = [
        cell for cell in plan.cells if (cell.workload_kind, cell.topology.tp, cell.kv_cache_dtype) == (phase, 2, label)
    ]
    return cell


def _files_of(plan, cell, change=None) -> list[dict]:
    """The files a pod of ``cell`` leaves under ``plan``, typed in; ``change`` changes each producer block."""
    payloads = _launcher_files(cell) if plan_is_moe(plan) else _dense_files(cell)
    if change is not None:
        for payload in payloads:
            change(payload["producer"])
    return payloads


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_a_plan_of_both_kv_cache_dtypes_takes_the_files_of_each_cell_for_that_cell_only(tmp_path, kind, phase):
    plan = _plan_of(kind, "bfloat16", "fp8")
    cells = {label: _two_gpu_cell(plan, phase, label) for label in ("bfloat16", "fp8")}
    assert cells["bfloat16"].cell_id != cells["fp8"].cell_id

    for label, cell in cells.items():
        other = cells["fp8" if label == "bfloat16" else "bfloat16"]
        recorded = POD_KV_CACHE[label]
        payloads = _files_of(plan, cell)
        assert {key: payloads[0]["producer"][key] for key in recorded} == recorded

        # The files of the cell, in the pod of the cell: accepted, and the rows carry the label.
        own = tmp_path / f"{label}-own"
        _collected(own, plan, cell, payloads)
        assert len(_validate(plan, cell, own).points) == len(_points(phase))
        rows = aggregate_cell(plan, cell, own, expected_attempt_id="attempt")
        assert {row["kv_cache_dtype"] for row in rows} == {label} and len(rows) == len(_points(phase))

        # The same files in the pod of the cell with the other label: refused, by the plan and without one.
        swapped = tmp_path / f"{label}-in-the-pod-of-the-other"
        written = _collected(swapped, plan, other, payloads)
        found, resolved = repr(recorded["kv_cache_stored_dtype"]), repr(recorded["kv_cache_dtype"])
        message = _not_the_kv_cache(other.kv_cache_dtype, found, resolved)
        for keywords in ({}, {"expected_moe": None}):
            with pytest.raises(ValueError) as refused:
                _validate(plan, other, swapped, **keywords)
            assert str(refused.value) == message + written["rank_files"][0]
        with pytest.raises(ValueError, match="^" + re.escape(message)):
            aggregate_cell(plan, other, swapped, expected_attempt_id="attempt")


def test_a_float16_pool_is_accepted_for_a_bfloat16_cell_and_for_no_fp8_cell(tmp_path):
    phase = "decode"
    plan = _plan_of("dense", "bfloat16", "fp8")
    bfloat16, fp8 = (_two_gpu_cell(plan, phase, label) for label in ("bfloat16", "fp8"))

    def float16(producer: dict) -> None:
        producer.update(kv_cache_dtype="auto", kv_cache_stored_dtype="torch.float16")

    _collected(tmp_path / "bfloat16", plan, bfloat16, _files_of(plan, bfloat16, float16))
    assert len(_validate(plan, bfloat16, tmp_path / "bfloat16").points) == len(_points(phase))
    rows = aggregate_cell(plan, bfloat16, tmp_path / "bfloat16", expected_attempt_id="attempt")
    assert {row["kv_cache_dtype"] for row in rows} == {"bfloat16"}

    written = _collected(tmp_path / "fp8", plan, fp8, _files_of(plan, fp8, float16))
    with pytest.raises(ValueError) as refused:
        _validate(plan, fp8, tmp_path / "fp8")
    assert str(refused.value) == _not_the_kv_cache("fp8", "'torch.float16'", "'auto'") + written["rank_files"][0]


# name: (the change to the producer block of a bfloat16 cell, what the refusal says it found, what it says of
# the resolved argument)
NOT_THE_KV_CACHE_OF_A_BFLOAT16_CELL = {
    # SGLang stored FP8 for ``auto``: the quantization config of the loaded model asked for it.
    "fp8-stored-for-auto": (
        lambda producer: producer.update(kv_cache_stored_dtype="torch.float8_e4m3fn"),
        "'torch.float8_e4m3fn'",
        "'auto'",
    ),
    # A dtype no label stands for.
    "float32": (lambda producer: producer.update(kv_cache_stored_dtype="torch.float32"), "'torch.float32'", "'auto'"),
    # A block without the key is no record of a dtype: the driver writes it into every complete result file.
    "no-stored-dtype": (
        lambda producer: producer.pop("kv_cache_stored_dtype"),
        "no producer.kv_cache_stored_dtype",
        "'auto'",
    ),
    # A value that is not the text of a dtype: null, as the failed artifact of a run without accepted headers
    # holds it.
    "null": (lambda producer: producer.update(kv_cache_stored_dtype=None), "None", "'auto'"),
}


# Every row on the dense plan, and one on the MoE plan.
@pytest.mark.parametrize(
    ("kind", "name"),
    [*(("dense", name) for name in NOT_THE_KV_CACHE_OF_A_BFLOAT16_CELL), ("moe", "fp8-stored-for-auto")],
)
def test_a_result_whose_engine_did_not_store_the_kv_cache_of_its_label_is_refused(tmp_path, kind, name):
    change, found, resolved = NOT_THE_KV_CACHE_OF_A_BFLOAT16_CELL[name]
    # No KV-cache dtype is named: both models of this suite are planned with bfloat16 cells.
    plan = _plan_of(kind)
    cell = _two_gpu_cell(plan, "decode", "bfloat16")
    # Written past the launcher's own writer, which is not asked whether it would take the block.
    (path,) = _written_past_the_writer(tmp_path, plan, cell, _files_of(plan, cell, change))
    message = _not_the_kv_cache("bfloat16", found, resolved)

    with pytest.raises(ValueError) as refused:
        _validate(plan, cell, tmp_path)
    assert str(refused.value) == message + str(path)
    # The row builder and the summary the runner keeps make the same comparison.
    with pytest.raises(ValueError, match="^" + re.escape(message)):
        aggregate_cell(plan, cell, tmp_path, expected_attempt_id="attempt")
    with pytest.raises(ValueError, match="^" + re.escape(message)):
        runner._runtime_collection_summary(cell, tmp_path / "raw", backend="sglang", is_moe=plan_is_moe(plan))
    # So does a caller without a plan: the rule is asked of every SGLang rank file.
    with pytest.raises(ValueError) as refused:
        _validate(plan, cell, tmp_path, expected_moe=None)
    assert str(refused.value) == message + str(path)
    with pytest.raises(ValueError, match="^" + re.escape(message)):
        runner._runtime_collection_summary(cell, tmp_path / "raw", backend="sglang")


# What the file of one rank of a DEP cell says differently from the file of the other rank: the change, the
# refusal, and whether a caller without a plan is refused too. The layout is compared from the plan alone; the
# KV cache is held against the label of the cell for every caller.
ONE_RANK_DIFFERS = {
    "attention-dp-size": (lambda producer: producer.update(attn_dp_size=1), _dep_refusal("2", "2", "1"), False),
    "kv-cache-stored-dtype": (
        lambda producer: producer.update(kv_cache_stored_dtype="torch.float8_e4m3fn"),
        _not_the_kv_cache("bfloat16", "'torch.float8_e4m3fn'", "'auto'"),
        True,
    ),
}


@pytest.mark.parametrize("faulty_rank", [0, 1])
@pytest.mark.parametrize("name", list(ONE_RANK_DIFFERS))
def test_every_rank_file_is_held_against_the_cell(tmp_path, name, faulty_rank):
    change, message, without_a_plan_too = ONE_RANK_DIFFERS[name]
    plan, _overrides = _resolved(presets=())
    cell = _cell(plan, "prefill", "dep")
    assert (cell.topology.dp, cell.kv_cache_dtype) == (2, "bfloat16")
    payloads = _launcher_files(cell)
    change(payloads[faulty_rank]["producer"])
    written = _collected(tmp_path, plan, cell, payloads)

    with pytest.raises(ValueError) as refused:
        _validate(plan, cell, tmp_path)

    assert str(refused.value) == message + written["rank_files"][faulty_rank]
    if without_a_plan_too:
        with pytest.raises(ValueError) as refused:
            _validate(plan, cell, tmp_path, expected_moe=None)
        assert str(refused.value) == message + written["rank_files"][faulty_rank]
    else:
        assert len(_validate(plan, cell, tmp_path, expected_moe=None).points) == len(_points("prefill"))


def test_a_cell_with_the_kv_cache_label_int8_which_has_no_rule_is_refused(tmp_path):
    # ``int8`` is the one label without a rule that an option can name. No cell of this suite carries it, so a
    # cell is relabelled.
    label, stored = "int8", "torch.bfloat16"
    plan = _plan_of("dense")
    planned = _two_gpu_cell(plan, "decode", "bfloat16")
    cell = dataclasses.replace(planned, kv_cache_dtype=label)
    payloads = _files_of(plan, planned, lambda producer: producer.update(kv_cache_stored_dtype=stored))
    (path,) = _written_past_the_writer(tmp_path, plan, cell, payloads)
    message = _no_rule_for_the_label(label, repr(stored), "'auto'")

    for keywords in ({}, {"expected_moe": None}):
        with pytest.raises(ValueError) as refused:
            _validate(plan, cell, tmp_path, **keywords)
        assert str(refused.value) == message + str(path)
    with pytest.raises(ValueError, match="^" + re.escape(message)):
        aggregate_cell(plan, cell, tmp_path, expected_attempt_id="attempt")


# ---------------------------------------------------------------------------------------------------------------
# The label ``nvfp4``: the stored dtype and the resolved argument together
# ---------------------------------------------------------------------------------------------------------------


def _nvfp4_cell(plan, phase: str):
    """A cell of two GPUs of ``plan`` relabelled ``nvfp4``, and the cell it was planned as.

    No plan carries that label: AISim infers it for a checkpoint whose quantization config declares an NVFP4
    KV cache, planning refuses such a checkpoint, and no option names the label (test_fpm_sglang_host.py). The
    validator's rule is for a caller that hands it such a cell.
    """
    planned = _two_gpu_cell(plan, phase, "bfloat16")
    return dataclasses.replace(planned, kv_cache_dtype="nvfp4"), planned


def test_the_files_of_an_nvfp4_cell_are_accepted_with_the_fp4_dtype_and_the_argument_nvfp4(tmp_path):
    phase = "decode"
    plan = _plan_of("dense")
    cell, planned = _nvfp4_cell(plan, phase)
    payloads = _files_of(plan, planned, lambda producer: producer.update(POD_KV_CACHE["nvfp4"]))
    _collected(tmp_path, plan, cell, payloads)

    assert len(_validate(plan, cell, tmp_path).points) == len(_points(phase))
    # The row builder takes the files too, and the rows carry the label of the cell.
    rows = aggregate_cell(plan, cell, tmp_path, expected_attempt_id="attempt")
    assert {row["kv_cache_dtype"] for row in rows} == {"nvfp4"} and len(rows) == len(_points(phase))


# name: (the change to the producer block of an nvfp4 cell, what the refusal says it found, what it says of the
# resolved argument)
NOT_THE_KV_CACHE_OF_AN_NVFP4_CELL = {
    # SGLang's other FP4 recipe is stored in the same dtype: the argument alone tells the two apart.
    "the-other-fp4-recipe": (
        lambda producer: producer.update(kv_cache_dtype="fp4_mx_block16"),
        "'torch.float4_e2m1fn_x2'",
        "'fp4_mx_block16'",
    ),
    # A block without the argument is no record of the recipe: the launcher writes it into every result file of
    # a run that got as far as building its engine.
    "no-argument": (lambda producer: producer.pop("kv_cache_dtype"), "'torch.float4_e2m1fn_x2'", "None"),
    # The right argument beside another stored dtype.
    "an-fp8-pool": (
        lambda producer: producer.update(kv_cache_stored_dtype="torch.float8_e4m3fn"),
        "'torch.float8_e4m3fn'",
        "'nvfp4'",
    ),
}


@pytest.mark.parametrize("name", list(NOT_THE_KV_CACHE_OF_AN_NVFP4_CELL))
def test_an_nvfp4_cell_is_refused_for_another_stored_dtype_and_for_another_resolved_argument(tmp_path, name):
    change, found, resolved = NOT_THE_KV_CACHE_OF_AN_NVFP4_CELL[name]
    plan = _plan_of("dense")
    cell, planned = _nvfp4_cell(plan, "decode")

    def edit(producer: dict) -> None:
        producer.update(POD_KV_CACHE["nvfp4"])
        change(producer)

    paths = _written_past_the_writer(tmp_path, plan, cell, _files_of(plan, planned, edit))
    message = _not_the_kv_cache("nvfp4", found, resolved)

    for keywords in ({}, {"expected_moe": None}):
        with pytest.raises(ValueError) as refused:
            _validate(plan, cell, tmp_path, **keywords)
        assert str(refused.value) == message + str(paths[0])
    with pytest.raises(ValueError, match="^" + re.escape(message)):
        aggregate_cell(plan, cell, tmp_path, expected_attempt_id="attempt")


# ---------------------------------------------------------------------------------------------------------------
# A whole campaign: render, the files a pod leaves, validation, publication
# ---------------------------------------------------------------------------------------------------------------


def _fake_cluster(plan, changed=None):
    """A cluster double for ``plan``: one pod per cell, which leaves the files of a complete FAKE launcher run.

    ``changed`` maps a cell id to a function that changes the producer block of that cell's result.
    """
    cells = {cell.cell_id: cell for cell in plan.cells}
    changed = changed or {}

    class FakeCluster:
        def __init__(self, _manifest, cell_dir, **_slurm_settings):
            self.cell = cells[cell_dir.name]
            self.pod = cell_dir / "raw" / "pod-0"

        def apply(self):
            pass

        def cleanup(self):
            pass

        def wait_ready(self, _expected_nodes):
            return ["pod-0"]

        def stage(self, _pods, files):
            pass

        def prepare_attempt(self, _pods, **keywords):
            # What the provenance program writes in a pod whose installed release is the one the plan names.
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
            payloads = _launcher_files(self.cell)
            for payload in payloads:
                changed.get(self.cell.cell_id, lambda _producer: None)(payload["producer"])
            artifact.write_artifacts(str(self.pod / "benchmark.json"), payloads)

        def collect(self, _pods, *, require_benchmark=True):
            pass

    return FakeCluster


def test_a_campaign_of_the_default_preset_renders_the_sizes_of_each_layout_and_publishes_dep_and_pure_tp_rows(
    tmp_path, monkeypatch
):
    plan, generator_overrides = _resolved(presets=())
    assert [(cell.workload_kind, cell.parallel_strategy) for cell in plan.cells] == [
        (phase, strategy) for phase in ("prefill", "decode") for strategy in ("single", "dep", "pure_tp", "tep")
    ]
    monkeypatch.setattr(runner, "KubernetesCellRunner", _fake_cluster(plan))

    errors = runner.run_collection(
        plan,
        generator_overrides=generator_overrides,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        artifact_root=str(tmp_path / "artifacts"),
        resume=False,
        retry_failed=False,
        database_root=str(tmp_path / "systems"),
    )

    assert errors == []
    # Each cell was rendered by the Generator and the cell builder. A DEP cell carries the three sizes SGLang
    # takes for it, and its pod left one rank file per data-parallel rank and the merged file. A pure TP cell
    # carries the tensor-parallel size alone, where the TEP cell of the same GPUs has the expert-parallel size
    # after it: without --ep-size SGLang runs one expert-parallel rank, which is the pure TP layout.
    root = tmp_path / "artifacts" / plan.sha256[:16]
    expected_sizes = {
        "single": ["--tp-size", "1"],
        "pure_tp": ["--tp-size", "2"],
        "tep": ["--tp-size", "2", "--ep-size", "2"],
        "dep": ["--tp-size", "2", "--ep-size", "2", "--attn-dp-size", "2"],
    }
    rendered = set()
    for cell in plan.cells:
        cell_dir = root / "cells" / cell.cell_id
        (line,) = [
            line for line in (cell_dir / "run.sh").read_text().splitlines() if line.startswith("engine_command=(")
        ]
        command = shlex.split(line[len("engine_command=(") : -1])
        assert command[:2] == ["python3", "/tmp/fpm-bench/sglbench_launcher.py"]
        assert (
            command[command.index("--tp-size") : command.index("--kv-cache-dtype")]
            == (expected_sizes[cell.parallel_strategy])
        )
        if cell.parallel_strategy == "pure_tp":
            # Nowhere in the command of either phase, not only between the two flags above.
            assert "--ep-size" not in command and "--attn-dp-size" not in command
            rendered.add(cell.workload_kind)
        results = sorted(path.name for path in (cell_dir / "raw" / "pod-0").glob("benchmark*.json"))
        if cell.parallel_strategy == "dep":
            assert results == ["benchmark.json", "benchmark_dp1.json", "benchmark_merged.json"]
        else:
            assert results == ["benchmark.json", "benchmark_merged.json"]
    assert rendered == {"prefill", "decode"}
    # Every cell passed. The record of a cell counts its rank files.
    checkpoint = json.loads((tmp_path / "checkpoints" / "fpm_forward.json").read_text())
    assert {record["status"] for record in checkpoint["cells"].values()} == {"passed"}
    assert {cell.parallel_strategy: checkpoint["cells"][cell.cell_id]["runtime_rank_count"] for cell in plan.cells} == {
        "single": 1,
        "pure_tp": 1,
        "tep": 1,
        "dep": 2,
    }
    # The record of a decode cell keeps the kvwarm block with the dataset.
    for cell in plan.cells:
        record = checkpoint["cells"][cell.cell_id]
        if cell.workload_kind == "decode":
            assert record["native_kvwarm"] == {**EXPECTED_KVWARM, "dataset": FAKE_DATASET}
        else:
            assert "native_kvwarm" not in record
    assert checkpoint["database"]["status"] == "passed"

    # The table holds the rows of the eight cells with their labels: a DEP row says tp 1 and dp 2, and a pure
    # TP row says moe_tp 2 and moe_ep 1 where a TEP row says moe_tp 1 and moe_ep 2.
    table = tmp_path / "systems" / SYSTEM / "sglang" / LABEL_WITH_TABLES
    rows = pq.read_table(table / "fpm_forward_perf.parquet").to_pylist()
    assert len(rows) == 4 * (len(_points("prefill")) + len(_points("decode")))
    # The table lies under the data label of the plan, and its metadata names the backend and counts the rows.
    metadata = json.loads((table / "fpm_forward_perf.metadata.json").read_text())
    assert (metadata["backend"], metadata["backend_version"]) == ("sglang", LABEL_WITH_TABLES)
    assert metadata["row_count"] == len(rows)
    # The table has no column for the content of the prompts.
    published = pq.read_table(table / "fpm_forward_perf.parquet")
    assert not [name for name in published.schema.names if "dataset" in name or "content" in name]
    labels = collections.Counter(
        (
            row["workload_kind"],
            row["parallel_strategy"],
            row["tp"],
            row["dp"],
            row["moe_tp"],
            row["moe_ep"],
            row["kv_seed_regime"],
        )
        for row in rows
    )
    assert labels == {
        ("prefill", "single", 1, 1, 1, 1, "n/a"): len(_points("prefill")),
        ("prefill", "dep", 1, 2, 1, 2, "n/a"): len(_points("prefill")),
        ("prefill", "pure_tp", 2, 1, 2, 1, "n/a"): len(_points("prefill")),
        ("prefill", "tep", 2, 1, 1, 2, "n/a"): len(_points("prefill")),
        ("decode", "single", 1, 1, 1, 1, "real_kv"): len(_points("decode")),
        ("decode", "dep", 1, 2, 1, 2, "real_kv"): len(_points("decode")),
        ("decode", "pure_tp", 2, 1, 2, 1, "real_kv"): len(_points("decode")),
        ("decode", "tep", 2, 1, 1, 2, "real_kv"): len(_points("decode")),
    }
    # The pure TP rows and the TEP rows of each phase carry the ids of two cells.
    assert len({row["cell_id"] for row in rows if row["parallel_strategy"] in ("pure_tp", "tep")}) == 4
    # A DEP row carries the batch of one rank, as the points of the FAKE launcher run name it.
    dep_decode = sorted(
        (row for row in rows if (row["workload_kind"], row["parallel_strategy"]) == ("decode", "dep")),
        key=lambda row: row["total_kv_read_tokens"],
    )
    assert [(row["batch_size"], row["total_kv_read_tokens"]) for row in dep_decode] == sorted(
        ((point["batch_size"], point["total_kv_read_tokens"]) for point in _points("decode")),
        key=lambda sizes: sizes[1],
    )

    # A resume reads the two rank files of a DEP cell again and takes them without a rerun.
    dep = _cell(plan, "decode", "dep")
    attempt_id = checkpoint["cells"][dep.cell_id]["attempt_id"]
    checkpoint["cells"][dep.cell_id] = {"status": "interrupted", "attempt_id": attempt_id}
    del checkpoint["database"]
    (tmp_path / "checkpoints" / "fpm_forward.json").write_text(json.dumps(checkpoint))
    monkeypatch.setattr(runner, "_render_cell", lambda *_a, **_k: pytest.fail("a resume ran a cell again"))

    assert (
        runner.run_collection(
            plan,
            generator_overrides=generator_overrides,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            artifact_root=str(tmp_path / "artifacts"),
            resume=True,
            retry_failed=False,
            database_root=str(tmp_path / "systems"),
        )
        == []
    )

    resumed = json.loads((tmp_path / "checkpoints" / "fpm_forward.json").read_text())
    assert resumed["cells"][dep.cell_id]["status"] == "passed"
    assert resumed["cells"][dep.cell_id]["artifact_recovery"]["original_status"] == "interrupted"
    assert resumed["cells"][dep.cell_id]["runtime_rank_count"] == 2


def test_a_campaign_whose_pod_sent_seeded_ids_for_a_moe_cell_fails_that_cell_and_publishes_nothing(
    tmp_path, monkeypatch
):
    plan, generator_overrides = _resolved()
    # The pod of the one-GPU decode cell found no expert layers in the model config and sent seeded token ids.
    # Its run is complete, and the launcher's writer and every other rule of the validator accept its files.
    seeded = _cell(plan, "decode", "single")
    monkeypatch.setattr(runner, "KubernetesCellRunner", _fake_cluster(plan, {seeded.cell_id: _seeded_ids}))

    errors = runner.run_collection(
        plan,
        generator_overrides=generator_overrides,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        artifact_root=str(tmp_path / "artifacts"),
        resume=False,
        retry_failed=False,
        database_root=str(tmp_path / "systems"),
    )

    result = tmp_path / "artifacts" / plan.sha256[:16] / "cells" / seeded.cell_id / "raw" / "pod-0" / "benchmark.json"
    refusal = _not_text("'seeded_token_ids'") + str(result)
    assert errors == [
        {
            "module": "fpm_forward",
            "cell_id": seeded.cell_id,
            "error_type": "ValueError",
            "error_message": refusal,
            "classification": "campaign_cell_failed",
        }
    ]
    checkpoint = json.loads((tmp_path / "checkpoints" / "fpm_forward.json").read_text())
    assert {cell_id: record["status"] for cell_id, record in checkpoint["cells"].items()} == {
        cell.cell_id: "failed" if cell is seeded else "passed" for cell in plan.cells
    }
    assert checkpoint["cells"][seeded.cell_id]["error"] == refusal
    # No table is written for a campaign with a failed cell, so no row carries the label of that cell.
    assert "database" not in checkpoint
    assert not (tmp_path / "systems").exists()

    # A resume looks at the complete files of a failed cell again, to take them without a rerun. They are refused
    # again, so the cell stays failed and is not run: the cluster double of the resume would leave good files.
    def reject_rerun(*_arguments, **_keywords):
        raise AssertionError("a failed cell is not run again without --resume-retry-failed")

    monkeypatch.setattr(runner, "_render_cell", reject_rerun)
    runner.run_collection(
        plan,
        generator_overrides=generator_overrides,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        artifact_root=str(tmp_path / "artifacts"),
        resume=True,
        retry_failed=False,
        database_root=str(tmp_path / "systems"),
    )

    resumed = json.loads((tmp_path / "checkpoints" / "fpm_forward.json").read_text())
    assert {cell_id: record["status"] for cell_id, record in resumed["cells"].items()} == {
        cell.cell_id: "failed" if cell is seeded else "passed" for cell in plan.cells
    }
    assert "artifact_recovery" not in resumed["cells"][seeded.cell_id]
    assert "database" not in resumed
    assert not (tmp_path / "systems").exists()
    # The files of a failed cell that differ from good ones in nothing are taken: the refusal is this rule's.
    good = _cell(plan, "decode", "tep")
    root = tmp_path / "artifacts" / plan.sha256[:16]
    as_failed = {**resumed["cells"][good.cell_id], "status": "failed"}
    assert runner._recover_completed_attempt(plan, good, root, as_failed)["status"] == "passed"
    assert runner._recover_completed_attempt(plan, seeded, root, resumed["cells"][seeded.cell_id]) is None
