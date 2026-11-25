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
cluster double leaves the files of a launcher run, and the real validator, gate, row builder and table writer
read them.

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
import types
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from collector.fpm_forward import capabilities, cli, native_artifact, planner, runner, sglang_fpm_builder
from collector.fpm_forward.database import aggregate_cell
from collector.fpm_forward.entry import resolve_inputs
from collector.fpm_forward.native_artifact import plan_is_moe, validate_native_collection
from collector.fpm_forward.runtime import sglbench_artifact as artifact
from collector.fpm_forward.runtime import sglbench_content as content
from collector.fpm_forward.runtime_memory import saved_plan_identity
from collector.fpm_forward.types import KVWARM_STRATEGIES, ParallelTopology
from collector.model_cases import build_collection_case_plan

from aisim.sdk import common
from aisim_core.sdk.fpm_identity import LEGACY_EXECUTION_IDENTITY

from .test_fpm_sglbench_artifact import (
    EXPECTED_KVWARM,
    EXPECTED_REGIMES,
    GATE_ACCEPT,
    POD_KV_CACHE,
    POD_SGLANG_VERSION,
    _gate,
    _points,
    _rank_artifacts,
)

pytestmark = pytest.mark.unit

MODEL = "Qwen/Qwen3-30B-A3B"
SYSTEM = "h200_sxm"
FAKE_IMAGE = "example.invalid/sglang:test"
# AISim's data holds the operator tables of this SGLang version for the system, so the planner gets a memory
# estimate for it. A data label without tables leaves the estimate unavailable and the cells planned.
LABEL_WITH_TABLES = "0.5.21"
# No table is kept under this label, and no release has it. A release is no such label for good: the data held
# no tables of the pod's release when these tests were written and holds them now.
LABEL_WITHOUT_TABLES = "0.0.0"

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


def _older_producer(cell, **sizes: int) -> dict:
    """The producer block of the launcher before it could run DP attention: no ``attn_dp_size``.

    Its ``staging`` is ``one_batch_call``, the one value that launcher wrote, and it has no ``dp_attention``.
    ``sizes`` replace the two sizes such a launcher recorded. The block keeps the two keys of the KV cache,
    which a launcher of that time did not write either: without them the rule of the KV cache refuses the
    files before any rule that reads the sizes
    (``test_result_files_written_before_the_pod_recorded_its_kv_cache_are_refused_for_every_caller``).
    """
    producer = _fake_producer(cell)
    for key in ("attn_dp_size", "dp_attention"):
        producer.pop(key, None)
    producer["staging"] = "one_batch_call"
    producer["tp_size"] = sizes.get("tp_size", producer["tp_size"])
    producer["moe"]["ep_size"] = sizes.get("ep_size", producer["moe"]["ep_size"])
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


def _arguments(*extra: str, label: str = LABEL_WITH_TABLES, presets: tuple[str, ...] = ("tep",)) -> list[str]:
    """What a user passes to ``python -m collector.fpm_forward`` for the MoE cells of one and two GPUs.

    ``presets`` are the values of ``--fpm-parallel-presets``. Without one the option is left out and the
    default preset applies, which plans the one-GPU cell, the DEP cell, the pure TP cell and the TEP cell.
    """
    return [
        *("--backend", "sglang", "--model-path", MODEL, "--gpu", SYSTEM),
        *("--fpm-max-gpus", "2", "--fpm-gpu-counts", "1", "2"),
        *(("--fpm-parallel-presets", *presets) if presets else ()),
        *("--fpm-prefill-cudagraph-policy", "runtime"),
        *("--fpm-runtime-backend-version", POD_SGLANG_VERSION, "--fpm-backend-version", label),
        *("--generator-set", f"K8sConfig.k8s_image={FAKE_IMAGE}"),
        *extra,
    ]


def _resolved(*extra: str, label: str = LABEL_WITH_TABLES, presets: tuple[str, ...] = ("tep",)):
    """The plan and the deployment inputs as the entry point resolves them (cli.py ``main``)."""
    arguments = cli._parser().parse_args(_arguments(*extra, label=label, presets=presets))
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


def test_the_entry_point_freezes_a_moe_plan_with_a_single_and_a_tep_cell(monkeypatch, capsys):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL)

    assert cli.main(_arguments("--plan-only")) == 0

    printed = json.loads(capsys.readouterr().out)
    assert (printed["backend"], printed["backend_version"], printed["runtime_backend_version"]) == (
        "sglang",
        LABEL_WITH_TABLES,
        POD_SGLANG_VERSION,
    )
    # The capability of the packaged config: a MoE model, for which this backend plans pure TP too. The TEP
    # preset of this plan leaves the pure TP cell out.
    capability = printed["capability"]
    assert (capability["architecture"], capability["is_moe"], capability["allow_pure_tp"]) == (
        "Qwen3MoeForCausalLM",
        True,
        True,
    )
    assert capability["model_config"]["source_kind"] == "aic_cache"
    # One cell per phase and GPU count: the one-GPU cell is "single", the two-GPU cell is "tep" with one
    # expert-parallel rank per tensor-parallel rank.
    assert [(cell["workload_kind"], cell["parallel_strategy"], cell["topology"]) for cell in printed["cells"]] == [
        (phase, strategy, {"tp": tp, "pp": 1, "dp": 1, "moe_tp": 1, "moe_ep": tp, "cp": 1})
        for phase in ("prefill", "decode")
        for strategy, tp in (("single", 1), ("tep", 2))
    ]
    assert [(topology["strategy"], topology["tp"]) for topology in printed["topologies"]] == [("single", 1), ("tep", 2)]
    # Memory admission estimated both topologies and found each below the capacity of one GPU. A rank of the
    # TEP cell holds half of the experts, so it needs less than the one-GPU cell.
    admission = printed["topology_memory_admission"]
    assert [(decision["topology"]["tp"], decision["disposition"]) for decision in admission] == [
        (1, "admitted"),
        (2, "admitted"),
    ]
    single, tep = (decision["estimates"][0] for decision in admission)
    assert 0 < tep["estimated_non_kv_bytes"] < single["estimated_non_kv_bytes"] < single["gpu_capacity_bytes"]
    assert (printed["counts"]["memory_rejected_topologies"], printed["counts"]["memory_unknown_topologies"]) == (0, 0)
    # The backend policy is the automatic baseline: no engine flag and no vLLM marker.
    (policy,) = printed["backend_policies"]
    assert (policy["policy_id"], policy["generator_overrides"], policy["expected_markers"]) == (
        "baseline_auto",
        {},
        {},
    )
    assert printed["point_generation"]["owner"] == "collector.fpm_forward.runtime.sglbench_launcher"
    # The reader of saved plans recomputes the hash from the printed form and rebuilds every cell.
    identity = saved_plan_identity(printed)
    assert (identity.backend, identity.sha256, len(identity.cells)) == ("sglang", printed["sha256"], 4)


def test_a_data_label_without_tables_leaves_the_estimate_unknown_and_the_cells_planned():
    plan, _overrides = _resolved(label=LABEL_WITHOUT_TABLES)

    assert [(decision.topology.tp, decision.disposition) for decision in plan.topology_memory_admission] == [
        (1, "unknown"),
        (2, "unknown"),
    ]
    assert [(cell.workload_kind, cell.parallel_strategy) for cell in plan.cells] == [
        ("prefill", "single"),
        ("prefill", "tep"),
        ("decode", "single"),
        ("decode", "tep"),
    ]
    # The label is part of the plan: the same cells under another label are another plan.
    assert plan.sha256 != _resolved()[0].sha256


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
    assert [(cell.workload_kind, cell.parallel_strategy) for cell in pure_tp_plan.cells] == [
        (phase, strategy) for phase in ("prefill", "decode") for strategy in ("single", "pure_tp")
    ]
    assert list(pure_tp_plan.cells) == cells_of_the_default_plan("single", "pure_tp")
    # --fpm-parallel-presets tep dep: every other cell, which is what the default preset planned before pure TP
    # was admitted for this backend.
    tep_and_dep_plan, _overrides = _resolved(presets=("tep", "dep"))
    assert [(cell.workload_kind, cell.parallel_strategy) for cell in tep_and_dep_plan.cells] == [
        (phase, strategy) for phase in ("prefill", "decode") for strategy in ("single", "dep", "tep")
    ]
    assert list(tep_and_dep_plan.cells) == cells_of_the_default_plan("single", "dep", "tep")
    # The cells are the same cells under each choice of presets, with the same ids, and each choice is another
    # plan.
    assert len({default_plan.sha256, pure_tp_plan.sha256, tep_and_dep_plan.sha256}) == 3


# ---------------------------------------------------------------------------------------------------------------
# Validation and rows of one cell
# ---------------------------------------------------------------------------------------------------------------


def test_a_tep_decode_result_that_names_its_dataset_is_accepted_and_gives_tep_rows(tmp_path):
    plan, _overrides = _resolved()
    cell = _cell(plan, "decode", "tep")
    assert cell.topology == ParallelTopology(tp=2, pp=1, dp=1, moe_tp=1, moe_ep=2, cp=1)
    # The strategy for which the validator demands an eligible kvwarm block of a decode cell.
    assert cell.parallel_strategy in KVWARM_STRATEGIES
    (payload,) = _launcher_files(cell)
    # Two tensor-parallel ranks, one file set: the artifact holds one data-parallel rank.
    assert (payload["dp"]["rank"], payload["dp"]["size"]) == (0, 1)
    assert (payload["producer"]["tp_size"], payload["producer"]["moe"]["ep_size"]) == (2, 2)
    assert payload["kvwarm"] == {**EXPECTED_KVWARM, "dataset": FAKE_DATASET}
    cell_dir = tmp_path / cell.cell_id

    written = _collected(cell_dir, plan, cell, [payload])

    # The validator accepts the file set and carries the block along as it is.
    collection = _validate(plan, cell, cell_dir)
    assert collection.backend_version == POD_SGLANG_VERSION
    assert [measurement.point for measurement in collection.points] == _points("decode")
    assert [measurement.kv_seed_regime for measurement in collection.points] == EXPECTED_REGIMES["decode"]
    assert collection.kvwarm_meta == {**EXPECTED_KVWARM, "dataset": FAKE_DATASET}
    # No execution identity, so no input provenance: the three input columns of a row stay empty.
    assert (cell.execution_identity, collection.input_provenance) == (LEGACY_EXECUTION_IDENTITY, None)
    # The gate accepts the same files.
    gate = _gate(written["rank_files"], "decode")
    assert gate.returncode == GATE_ACCEPT, gate.stderr

    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    assert len(rows) == len(_points("decode"))
    for row, point in zip(rows, _points("decode"), strict=True):
        # The labels of the cell: TEP on two GPUs, two expert-parallel ranks.
        assert (row["parallel_strategy"], row["tp"], row["pp"], row["dp"], row["moe_tp"], row["moe_ep"], row["cp"]) == (
            "tep",
            2,
            1,
            1,
            1,
            2,
            1,
        )
        assert row["kv_seed_regime"] == "real_kv"
        assert (row["backend"], row["runtime_backend_version"], row["backend_version"]) == (
            "sglang",
            POD_SGLANG_VERSION,
            LABEL_WITH_TABLES,
        )
        assert (row["model_path"], row["cell_id"], row["source_plan_sha256"]) == (MODEL, cell.cell_id, plan.sha256)
        assert (row["workload_kind"], row["batch_size"], row["total_kv_read_tokens"]) == (
            "decode",
            point["batch_size"],
            point["total_kv_read_tokens"],
        )
        # The engine decided the four backend identity columns, and SGLang's values for them are not recorded.
        assert (row["moe_backend"], row["attention_backend"], row["enable_wideep"], row["enable_eplb"]) == (
            "auto",
            "auto",
            False,
            False,
        )
        # No column says which text was sent: the dataset stays in the result file and in the checkpoint.
        assert (row["input_text_sha256"], row["input_token_ids_sha256"], row["input_tokenizer_revision"]) == (
            None,
            None,
            None,
        )
        assert not [column for column in row if "dataset" in column or "content" in column]
    assert len({row["latency_ms"] for row in rows}) == len(rows)

    # The summary the runner keeps in the checkpoint of the cell holds the block, and so the dataset.
    summary = runner._runtime_collection_summary(cell, cell_dir / "raw", backend="sglang", is_moe=plan_is_moe(plan))
    assert summary["native_kvwarm"]["dataset"] == FAKE_DATASET
    assert summary["native_kv_seed_regime_counts"] == {"real_kv": len(rows)}


def test_the_single_moe_decode_cell_and_the_tep_prefill_cell_give_rows_with_their_own_labels(tmp_path):
    plan, _overrides = _resolved()
    expected = {
        # (phase, strategy): (tp, moe_ep, the regime of every row)
        ("decode", "single"): (1, 1, "real_kv"),
        ("prefill", "tep"): (2, 2, "n/a"),
        ("prefill", "single"): (1, 1, "n/a"),
    }
    for (phase, strategy), (tp, moe_ep, regime) in expected.items():
        cell = _cell(plan, phase, strategy)
        cell_dir = tmp_path / cell.cell_id
        payloads = _launcher_files(cell)
        # A prefill artifact has no kvwarm block, so it names no dataset.
        assert ("kvwarm" in payloads[0]) is (phase == "decode")
        _collected(cell_dir, plan, cell, payloads)

        collection = _validate(plan, cell, cell_dir)
        rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

        assert (collection.kvwarm_meta is None) is (phase == "prefill")
        assert len(rows) == len(_points(phase))
        assert {
            (row["parallel_strategy"], row["tp"], row["dp"], row["moe_tp"], row["moe_ep"], row["kv_seed_regime"])
            for row in rows
        } == {(strategy, tp, 1, 1, moe_ep, regime)}


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_a_pure_tp_result_with_one_expert_parallel_rank_is_accepted_and_gives_pure_tp_rows(tmp_path, phase):
    plan, _overrides = _resolved(presets=())
    cell = _cell(plan, phase, "pure_tp")
    assert cell.topology == ParallelTopology(tp=2, pp=1, dp=1, moe_tp=2, moe_ep=1, cp=1)
    # The strategy for which the validator demands an eligible kvwarm block of a decode cell.
    assert cell.parallel_strategy in KVWARM_STRATEGIES
    (payload,) = _launcher_files(cell)
    # Two tensor-parallel ranks, one file set. The sizes SGLang resolves for a command with --tp-size 2 and no
    # --ep-size: both ranks, one expert-parallel rank, which is its default, and one attention-DP rank.
    assert (payload["dp"]["rank"], payload["dp"]["size"]) == (0, 1)
    producer = payload["producer"]
    assert (producer["tp_size"], producer["moe"]["ep_size"], producer["attn_dp_size"]) == (2, 1, 1)
    assert (producer["staging"], "dp_attention" in producer) == ("one_batch_call", False)
    cell_dir = tmp_path / cell.cell_id

    written = _collected(cell_dir, plan, cell, [payload])

    # The validator accepts the file set, and the gate accepts the same files.
    points = _points(phase)
    collection = _validate(plan, cell, cell_dir)
    assert collection.backend_version == POD_SGLANG_VERSION
    assert [measurement.point for measurement in collection.points] == points
    assert [measurement.kv_seed_regime for measurement in collection.points] == EXPECTED_REGIMES[phase]
    assert collection.kvwarm_meta == ({**EXPECTED_KVWARM, "dataset": FAKE_DATASET} if phase == "decode" else None)
    gate = _gate(written["rank_files"], phase)
    assert gate.returncode == GATE_ACCEPT, gate.stderr

    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    assert len(rows) == len(points)
    for row, point in zip(rows, points, strict=True):
        # The labels of the cell: pure TP on two GPUs, where moe_tp is the tensor-parallel size and there is one
        # expert-parallel rank. The TEP rows of the same GPUs differ from these in the two expert sizes.
        assert (row["parallel_strategy"], row["tp"], row["pp"], row["dp"], row["moe_tp"], row["moe_ep"], row["cp"]) == (
            "pure_tp",
            2,
            1,
            1,
            2,
            1,
            1,
        )
        # A decode row is real cached state, and a prefill row has the regime of every other MoE prefill row.
        assert row["kv_seed_regime"] == ("real_kv" if phase == "decode" else "n/a")
        assert (row["backend"], row["runtime_backend_version"], row["backend_version"]) == (
            "sglang",
            POD_SGLANG_VERSION,
            LABEL_WITH_TABLES,
        )
        assert (row["model_path"], row["cell_id"], row["source_plan_sha256"]) == (MODEL, cell.cell_id, plan.sha256)
        assert (row["workload_kind"], row["batch_size"]) == (phase, point["batch_size"])
        assert (row["total_prefill_tokens"], row["total_kv_read_tokens"]) == (
            point["total_prefill_tokens"],
            point["total_kv_read_tokens"],
        )
        # The engine decided the four backend identity columns, as for every other cell of this backend.
        assert (row["moe_backend"], row["attention_backend"], row["enable_wideep"], row["enable_eplb"]) == (
            "auto",
            "auto",
            False,
            False,
        )
    assert len({row["latency_ms"] for row in rows}) == len(rows)

    # What the runner keeps in the checkpoint of the cell: the points, and one rank file.
    summary = runner._runtime_collection_summary(cell, cell_dir / "raw", backend="sglang", is_moe=plan_is_moe(plan))
    assert summary["measured_point_count"] == len(points)
    assert runner._runtime_timing_summary(cell_dir / "raw")["runtime_rank_count"] == 1
    if phase == "decode":
        assert summary["native_kv_seed_regime_counts"] == {"real_kv": len(rows)}
        assert summary["native_kvwarm"]["dataset"] == FAKE_DATASET


def test_the_result_of_a_tep_cell_is_no_result_of_the_pure_tp_cell_of_the_same_gpus_and_the_reverse(tmp_path):
    plan, _overrides = _resolved(presets=())
    tep, pure_tp = _cell(plan, "decode", "tep"), _cell(plan, "decode", "pure_tp")
    # Both cells have two tensor-parallel ranks and one data-parallel rank, so a pod of either leaves one rank
    # file. Of what a file records, only the expert-parallel size says which layout ran.
    assert (tep.topology.tp, tep.topology.dp) == (pure_tp.topology.tp, pure_tp.topology.dp) == (2, 1)
    for ran, other, found_ep in ((tep, pure_tp, "2"), (pure_tp, tep, "1")):
        root = tmp_path / ran.parallel_strategy
        written = _collected(root, plan, ran, _launcher_files(ran))
        # The other cell of the same GPUs, under the id of the cell that ran.
        as_the_other = dataclasses.replace(ran, parallel_strategy=other.parallel_strategy, topology=other.topology)

        assert len(_validate(plan, ran, root).points) == len(_points("decode"))
        with pytest.raises(ValueError) as refused:
            _validate(plan, as_the_other, root)

        assert str(refused.value) == (
            _not_the_layout(2, other.topology.moe_ep, "2", found_ep) + written["rank_files"][0]
        )


def test_a_tep_decode_result_without_an_eligible_kvwarm_block_is_refused(tmp_path):
    plan, _overrides = _resolved()
    cell = _cell(plan, "decode", "tep")
    refusals = {
        "removed": (
            lambda payload: payload.pop("kvwarm"),
            "native decode result lacks required KV warm-up metadata for strategy 'tep'",
        ),
        "skipped": (
            lambda payload: payload["kvwarm"].update(warm_eligible=False, skip_reason="dataset_unavailable"),
            "native KV warm-up protocol mismatch for strategy 'tep': enabled=True, skip_reason='dataset_unavailable'",
        ),
    }
    for name, (edit, message) in refusals.items():
        (payload,) = _launcher_files(cell)
        edit(payload)
        # The launcher's own writer refuses such a file, so it is written past it.
        artifact.atomic_write_json(str(_pod(tmp_path / name, plan, cell) / "benchmark.json"), payload)

        with pytest.raises(ValueError, match=message):
            _validate(plan, cell, tmp_path / name)


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


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_a_dep_result_of_two_ranks_is_accepted_and_gives_dep_rows_with_the_sizes_of_one_rank(tmp_path, phase):
    plan, _overrides = _resolved(presets=())
    cell = _cell(plan, phase, "dep")
    assert cell.topology == ParallelTopology(tp=1, pp=1, dp=2, moe_tp=1, moe_ep=2, cp=1)
    # The strategy for which the validator demands an eligible kvwarm block of a decode cell.
    assert cell.parallel_strategy in KVWARM_STRATEGIES
    payloads = _launcher_files(cell)
    # One file per attention-DP rank. Each records the sizes SGLang resolves for the cell: both ranks in
    # tp_size and ep_size, where the cell says tp 1 and dp 2.
    assert [(payload["dp"]["rank"], payload["dp"]["size"]) for payload in payloads] == [(0, 2), (1, 2)]
    for payload in payloads:
        producer = payload["producer"]
        assert (producer["tp_size"], producer["moe"]["ep_size"], producer["attn_dp_size"]) == (2, 2, 2)
        assert producer["dp_attention"] == {
            "local_control_broadcast": False,
            "input_blocker": True,
            "placement": "round_robin_checked",
        }
        assert producer["staging"] == "balanced_batch_call"
    cell_dir = tmp_path / cell.cell_id

    written = _collected(cell_dir, plan, cell, payloads)

    # The names of the launcher's writer: rank 0 at the output path, rank 1 beside it, and the merged file.
    assert [Path(name).name for name in written["rank_files"]] == ["benchmark.json", "benchmark_dp1.json"]
    assert Path(written["merged_output_path"]).name == "benchmark_merged.json"
    points = _points(phase)
    collection = _validate(plan, cell, cell_dir)
    assert collection.backend_version == POD_SGLANG_VERSION
    assert [measurement.point for measurement in collection.points] == points
    assert [measurement.kv_seed_regime for measurement in collection.points] == EXPECTED_REGIMES[phase]
    # Each point has the time of both ranks, and each rank its own elapsed time.
    assert [[rank for rank, _time in measurement.rank_wall_times] for measurement in collection.points] == [
        [0, 1]
    ] * len(points)
    assert [rank for rank, _elapsed, _measured in collection.rank_timings] == [0, 1]
    assert collection.kvwarm_meta == ({**EXPECTED_KVWARM, "dataset": FAKE_DATASET} if phase == "decode" else None)
    # The gate accepts the same two rank files with their merged file.
    gate = _gate(written["rank_files"], phase)
    assert gate.returncode == GATE_ACCEPT, gate.stderr

    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    assert len(rows) == len(points)
    for row, point, measurement in zip(rows, points, collection.points, strict=True):
        # The labels of the cell: DEP on two GPUs, one tensor-parallel rank in each of two data-parallel ranks.
        assert (row["parallel_strategy"], row["tp"], row["pp"], row["dp"], row["moe_tp"], row["moe_ep"], row["cp"]) == (
            "dep",
            1,
            1,
            2,
            1,
            2,
            1,
        )
        # The sizes of a row are those of ONE rank: the engine ran twice the batch and twice the tokens.
        assert (row["workload_kind"], row["batch_size"]) == (phase, point["batch_size"])
        assert (row["total_prefill_tokens"], row["total_kv_read_tokens"]) == (
            point["total_prefill_tokens"],
            point["total_kv_read_tokens"],
        )
        # The time of a row is the slower rank's. The FAKE times of the two ranks differ at every point.
        times = [time for _rank, time in measurement.rank_wall_times]
        assert min(times) < max(times)
        assert row["latency_ms"] == max(times) * 1000.0
        assert row["kv_seed_regime"] == ("real_kv" if phase == "decode" else "n/a")
        assert (row["backend"], row["runtime_backend_version"], row["cell_id"]) == (
            "sglang",
            POD_SGLANG_VERSION,
            cell.cell_id,
        )
        # No column says that the cell ran DP attention with the input blocker: that stays in the result file.
        assert not [column for column in row if "attn" in column or "blocker" in column or "staging" in column]

    # What the runner keeps in the checkpoint of the cell: the counts of the points and the two ranks.
    summary = runner._runtime_collection_summary(cell, cell_dir / "raw", backend="sglang", is_moe=plan_is_moe(plan))
    assert summary["measured_point_count"] == len(points)
    timing = runner._runtime_timing_summary(cell_dir / "raw")
    assert timing["runtime_rank_count"] == 2
    # Every rank reports the sum of the slower rank's times, so the two files agree on it.
    assert timing["measured_iteration_seconds"] == pytest.approx(
        sum(max(time for _rank, time in measurement.rank_wall_times) for measurement in collection.points)
    )


def test_result_files_of_a_launcher_without_dp_attention_are_refused_for_every_cell_of_a_plan_made_now(tmp_path):
    # A launcher of that time wrote no attention-DP size. Its files belong to a plan of that source revision, and
    # under a plan made now a block without the size is no record of one rank: the launcher writes the size into
    # every result file. The two sizes such a block holds are those of the cell, and it is refused all the same.
    # The block is typed in with the two keys of the KV cache, which no launcher of that time wrote, so that
    # this test pins the rule of the sizes alone: a file as that launcher wrote it is refused for its KV cache
    # first (``_older_producer``).
    plan, _overrides = _resolved(presets=())
    for phase in ("prefill", "decode"):
        for strategy, sizes in (("single", (1, 1)), ("tep", (2, 2))):
            cell = _cell(plan, phase, strategy)
            cell_dir = tmp_path / cell.cell_id
            payloads = _launcher_files(cell, _older_producer(cell))
            assert "attn_dp_size" not in payloads[0]["producer"]
            written = _collected(cell_dir, plan, cell, payloads)
            refusal = _not_the_layout(*sizes, str(sizes[0]), str(sizes[1]), found_dp="None")

            with pytest.raises(ValueError) as refused:
                _validate(plan, cell, cell_dir)

            assert str(refused.value) == refusal + written["rank_files"][0]
            with pytest.raises(ValueError, match="^" + re.escape(refusal)):
                aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
            # Every other rule accepts the files: only this comparison reads the key.
            assert len(_validate(plan, cell, cell_dir, expected_moe=None).points) == len(_points(phase))

    # The DEP cell of the same plan. Two rank files with such a block are refused whatever two sizes the block
    # holds: those SGLang resolves for the DEP cell, which are also those of the TEP cell of the same GPUs;
    # those of the one-GPU cell; and the cell's own tp beside its moe_ep.
    for phase in ("prefill", "decode"):
        dep = _cell(plan, phase, "dep")
        for name, sizes, found in (
            ("as-recorded", {}, ("2", "2")),
            ("one-rank", {"tp_size": 1, "ep_size": 1}, ("1", "1")),
            ("the-cells-own-tp", {"tp_size": 1}, ("1", "2")),
        ):
            root = tmp_path / f"dep-{phase}-{name}"
            payloads = _launcher_files(dep, _older_producer(dep, **sizes))
            written = _collected(root, plan, dep, payloads)

            with pytest.raises(ValueError) as refused:
                _validate(plan, dep, root)

            assert str(refused.value) == _dep_refusal(*found, "None") + written["rank_files"][0]
            with pytest.raises(ValueError, match=re.escape(_dep_refusal(*found, "None"))):
                aggregate_cell(plan, dep, root, expected_attempt_id="attempt")


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


def _sizes(tp_size: int, ep_size: int, attn_dp_size: int | None = None):
    """The change that records these sizes. Without ``attn_dp_size`` the recorded one stays."""

    def change(producer: dict) -> None:
        producer["tp_size"] = tp_size
        producer["moe"]["ep_size"] = ep_size
        if attn_dp_size is not None:
            producer["attn_dp_size"] = attn_dp_size

    return change


def _dep_refusal(found_tp: str, found_ep: str, found_dp: str) -> str:
    """The refusal of a layout for the DEP cell of two ranks, for which SGLang resolves 2, 2 and 2."""
    return _not_the_layout(2, 2, found_tp, found_ep, dp=2, found_dp=found_dp)


# name: (the strategy of the cell, the change to the producer block, the refusal)
DIFFERS_FROM_THE_CELL = {
    # The pod found no expert layers, although the plan's config has them. In a TEP pod the launcher refuses
    # such a run for its --ep-size, so these files are those of a pod that the flag did not reach either.
    "seeded-ids": ("single", _seeded_ids, _not_text("'seeded_token_ids'")),
    "seeded-ids-in-a-tep-pod": ("tep", _seeded_ids, _not_text("'seeded_token_ids'")),
    "no-content-block": ("single", lambda producer: producer.pop("content"), _not_text("None")),
    "a-content-block-that-is-no-mapping": ("tep", lambda producer: producer.update(content="text"), _not_text("None")),
    # Only the one word is dataset text: no other word that looks like it, and no other spelling of it.
    "another-text-word": (
        "tep",
        lambda producer: producer["content"].update(source="dataset_text"),
        _not_text("'dataset_text'"),
    ),
    "another-spelling-of-the-word": (
        "single",
        lambda producer: producer["content"].update(source="Dataset_Text_Chains"),
        _not_text("'Dataset_Text_Chains'"),
    ),
    # Text was asked for by name in such a pod: the content is text, and no layout was recorded.
    "text-without-expert-layers": ("single", _no_expert_layers_found, _not_the_layout(1, 1, "1", "None")),
    # The command reached the pod without --ep-size: every rank holds a slice of every expert. That is the layout
    # of the pure TP cell of the same GPUs, which the launcher accepts, and it is no result of the TEP cell.
    "tep-without-the-expert-parallel-size": (
        "tep",
        lambda producer: producer["moe"].update(ep_size=1),
        _not_the_layout(2, 2, "2", "1"),
    ),
    "tep-on-one-rank": ("tep", lambda producer: producer.update(tp_size=1), _not_the_layout(2, 2, "1", "2")),
    "single-on-two-ranks": ("single", lambda producer: producer.update(tp_size=2), _not_the_layout(1, 1, "2", "1")),
    # The two recorded sizes agree with each other and are not the cell's: the command reached the pod without
    # both sizes, or with those of another GPU count. The launcher accepts each of these layouts.
    "tep-run-as-the-one-gpu-cell": ("tep", _sizes(1, 1), _not_the_layout(2, 2, "1", "1")),
    "tep-run-on-four-ranks": ("tep", _sizes(4, 4), _not_the_layout(2, 2, "4", "4")),
    "single-run-as-the-tep-cell": ("single", _sizes(2, 2), _not_the_layout(1, 1, "2", "2")),
    "no-tensor-parallel-size": ("tep", lambda producer: producer.pop("tp_size"), _not_the_layout(2, 2, "None", "2")),
    "no-expert-parallel-size": (
        "tep",
        lambda producer: producer["moe"].pop("ep_size"),
        _not_the_layout(2, 2, "2", "None"),
    ),
    "a-moe-block-that-is-no-mapping": (
        "tep",
        lambda producer: producer.update(moe=[2]),
        _not_the_layout(2, 2, "2", "None"),
    ),
    # True equals 1 and 2.0 equals 2, and neither is a size; nor is the written form of a number.
    "a-bool": ("single", lambda producer: producer["moe"].update(ep_size=True), _not_the_layout(1, 1, "1", "True")),
    "a-float": ("tep", lambda producer: producer["moe"].update(ep_size=2.0), _not_the_layout(2, 2, "2", "2.0")),
    "a-string": ("tep", lambda producer: producer.update(tp_size="2"), _not_the_layout(2, 2, "'2'", "2")),
    # The attention-DP size of a cell with one data-parallel rank must be 1. A pod of the TEP cell that ran DP
    # attention ran another cell on the same GPUs.
    "tep-run-with-dp-attention": (
        "tep",
        lambda producer: producer.update(attn_dp_size=2),
        _not_the_layout(2, 2, "2", "2", found_dp="2"),
    ),
    "an-attention-dp-size-that-is-a-bool": (
        "single",
        lambda producer: producer.update(attn_dp_size=True),
        _not_the_layout(1, 1, "1", "1", found_dp="True"),
    ),
    # A key without a size is no size, and neither is the written form of the number.
    "an-attention-dp-size-of-none": (
        "tep",
        lambda producer: producer.update(attn_dp_size=None),
        _not_the_layout(2, 2, "2", "2", found_dp="None"),
    ),
    "an-attention-dp-size-that-is-a-string": (
        "single",
        lambda producer: producer.update(attn_dp_size="1"),
        _not_the_layout(1, 1, "1", "1", found_dp="'1'"),
    ),
    # A block without the key is no record of one rank: the launcher writes the size into every result file,
    # with 1 for a run without DP attention.
    "single-without-an-attention-dp-size": (
        "single",
        lambda producer: producer.pop("attn_dp_size"),
        _not_the_layout(1, 1, "1", "1", found_dp="None"),
    ),
    "tep-without-an-attention-dp-size": (
        "tep",
        lambda producer: producer.pop("attn_dp_size"),
        _not_the_layout(2, 2, "2", "2", found_dp="None"),
    ),
    "pure-tp-without-an-attention-dp-size": (
        "pure_tp",
        lambda producer: producer.pop("attn_dp_size"),
        _not_the_layout(2, 1, "2", "1", found_dp="None"),
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
    "dep-run-on-four-ranks": ("dep", _sizes(4, 4, 4), _dep_refusal("4", "4", "4")),
    "dep-run-as-the-one-gpu-cell": ("dep", _sizes(1, 1, 1), _dep_refusal("1", "1", "1")),
    "dep-with-attention-tp-inside-a-rank": ("dep", _sizes(4, 4, 2), _dep_refusal("4", "4", "2")),
    "dep-with-a-float": ("dep", lambda producer: producer.update(attn_dp_size=2.0), _dep_refusal("2", "2", "2.0")),
    # The pure TP cell of two GPUs, for which SGLang resolves 2, 1 and 1. The command reached the pod with the
    # --ep-size of the TEP cell: one expert-parallel rank per tensor-parallel rank. The launcher accepts that
    # layout too, so this comparison alone tells the two cells apart.
    "pure-tp-run-with-the-expert-parallel-size-of-the-tep-cell": (
        "pure_tp",
        lambda producer: producer["moe"].update(ep_size=2),
        _not_the_layout(2, 1, "2", "2"),
    ),
    "pure-tp-run-as-the-one-gpu-cell": ("pure_tp", _sizes(1, 1), _not_the_layout(2, 1, "1", "1")),
    "pure-tp-run-on-four-ranks": ("pure_tp", _sizes(4, 1), _not_the_layout(2, 1, "4", "1")),
    "pure-tp-with-the-sizes-of-the-dep-cell": (
        "pure_tp",
        _sizes(2, 2, 2),
        _not_the_layout(2, 1, "2", "2", found_dp="2"),
    ),
    # A pod of the pure TP cell that found no expert layers. Its command has no --ep-size for which the launcher
    # could refuse the run, so the pod runs it as the TP cell of a dense model: with seeded ids, or with text
    # when text was asked for by name. The host refuses the files of both.
    "seeded-ids-in-a-pure-tp-pod": ("pure_tp", _seeded_ids, _not_text("'seeded_token_ids'")),
    "pure-tp-text-without-expert-layers": ("pure_tp", _no_expert_layers_found, _not_the_layout(2, 1, "2", "None")),
    # True equals 1, the expert-parallel size this cell expects, and is no size.
    "pure-tp-with-a-bool": (
        "pure_tp",
        lambda producer: producer["moe"].update(ep_size=True),
        _not_the_layout(2, 1, "2", "True"),
    ),
}


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("name", list(DIFFERS_FROM_THE_CELL))
def test_a_moe_result_that_differs_from_its_cell_is_refused(tmp_path, phase, name):
    strategy, change, message = DIFFERS_FROM_THE_CELL[name]
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


def test_the_comparison_names_the_file_it_refuses(tmp_path):
    plan, _overrides = _resolved()
    cell = _cell(plan, "decode", "tep")
    (payload,) = _launcher_files(cell)
    payload["producer"]["moe"]["ep_size"] = 1
    path = _pod(tmp_path, plan, cell) / "benchmark.json"
    artifact.atomic_write_json(str(path), payload)

    with pytest.raises(ValueError) as refused:
        _validate(plan, cell, tmp_path)

    assert str(refused.value) == _not_the_layout(2, 2, "2", "1") + str(path)


# What the file of one rank of a DEP cell says differently from the file of the other rank.
ONE_RANK_DIFFERS = {
    "expert-parallel-size": (lambda producer: producer["moe"].update(ep_size=1), _dep_refusal("2", "1", "2")),
    "attention-dp-size": (lambda producer: producer.update(attn_dp_size=1), _dep_refusal("2", "2", "1")),
    "no-attention-dp-size": (lambda producer: producer.pop("attn_dp_size"), _dep_refusal("2", "2", "None")),
    "tensor-parallel-size": (lambda producer: producer.update(tp_size=1), _dep_refusal("1", "2", "2")),
}


@pytest.mark.parametrize("faulty_rank", [0, 1])
@pytest.mark.parametrize("name", list(ONE_RANK_DIFFERS))
def test_every_rank_file_is_held_against_the_cell(tmp_path, name, faulty_rank):
    change, message = ONE_RANK_DIFFERS[name]
    plan, _overrides = _resolved(presets=())
    cell = _cell(plan, "prefill", "dep")
    payloads = _launcher_files(cell)
    change(payloads[faulty_rank]["producer"])
    written = _collected(tmp_path, plan, cell, payloads)

    with pytest.raises(ValueError) as refused:
        _validate(plan, cell, tmp_path)

    assert str(refused.value) == message + written["rank_files"][faulty_rank]
    assert len(_validate(plan, cell, tmp_path, expected_moe=None).points) == len(_points("prefill"))


@pytest.mark.parametrize("producer", [None, [], "producer"])
def test_a_producer_block_that_is_no_mapping_did_not_send_text(tmp_path, producer):
    # The validator refuses such a block for its release before this rule is reached, so the rule is called alone.
    plan, _overrides = _resolved()
    path = tmp_path / "benchmark.json"

    with pytest.raises(ValueError) as refused:
        native_artifact._validate_sglang_moe_producer(_cell(plan, "decode", "tep"), producer, path)

    assert str(refused.value) == _not_text("None") + str(path)


@pytest.mark.parametrize("ranks", [2, 4, 8])
def test_the_recorded_expert_parallel_size_tells_a_pure_tp_result_from_a_tep_result_at_every_gpu_count(tmp_path, ranks):
    # The plan of this suite ends at two GPUs, so the rule is called alone with the cells of a wider plan: the
    # topologies the pure TP and the TEP preset give for this many GPUs.
    plan, _overrides = _resolved(presets=())
    pure_tp = dataclasses.replace(
        _cell(plan, "decode", "pure_tp"),
        topology=ParallelTopology(tp=ranks, pp=1, dp=1, moe_tp=ranks, moe_ep=1, cp=1),
    )
    tep = dataclasses.replace(
        _cell(plan, "decode", "tep"), topology=ParallelTopology(tp=ranks, pp=1, dp=1, moe_tp=1, moe_ep=ranks, cp=1)
    )
    path = tmp_path / "benchmark.json"
    # What a pod of each cell records: every rank in tp_size, one attention-DP rank, and the expert-parallel
    # size of its layout, 1 for pure TP and the tensor-parallel size for TEP.
    ran_pure_tp, ran_tep = _fake_producer(pure_tp), _fake_producer(tep)
    assert (ran_pure_tp["tp_size"], ran_pure_tp["moe"]["ep_size"], ran_pure_tp["attn_dp_size"]) == (ranks, 1, 1)
    assert (ran_tep["tp_size"], ran_tep["moe"]["ep_size"], ran_tep["attn_dp_size"]) == (ranks, ranks, 1)

    # Each block is accepted for its own cell.
    native_artifact._validate_sglang_moe_producer(pure_tp, ran_pure_tp, path)
    native_artifact._validate_sglang_moe_producer(tep, ran_tep, path)
    # Each is refused for the other cell of the same GPUs, and the refusal names the size that differs.
    with pytest.raises(ValueError) as refused:
        native_artifact._validate_sglang_moe_producer(pure_tp, ran_tep, path)
    assert str(refused.value) == (
        "native SGLang rank artifact of a MoE plan did not run the layout of its cell: "
        f"expected producer.tp_size {ranks}, producer.moe.ep_size 1 and producer.attn_dp_size 1, "
        f"found {ranks}, {ranks} and 1: {path}"
    )
    with pytest.raises(ValueError) as refused:
        native_artifact._validate_sglang_moe_producer(tep, ran_pure_tp, path)
    assert str(refused.value) == _not_the_layout(ranks, ranks, str(ranks), "1") + str(path)


def test_the_text_source_the_validator_expects_is_the_one_the_content_plan_writes():
    # The validator does not import the runtime modules, so the two values are held together here.
    assert native_artifact._SGLANG_TEXT_CONTENT_SOURCE == content.CONTENT_TEXT_CHAINS == "dataset_text_chains"


def test_a_plan_says_whether_its_model_has_expert_layers_and_a_stand_in_may_say_nothing():
    plan, _overrides = _resolved()
    assert (plan.capability.is_moe, plan_is_moe(plan)) == (True, True)

    def stand_in(**capability):
        return types.SimpleNamespace(capability=types.SimpleNamespace(**capability))

    assert plan_is_moe(stand_in(is_moe=False)) is False
    assert plan_is_moe(stand_in(is_moe=True)) is True
    # Only a bool is an answer.
    for silent in (
        object(),
        types.SimpleNamespace(capability=None),
        stand_in(),
        stand_in(is_moe=1),
        stand_in(is_moe="no"),
    ):
        assert plan_is_moe(silent) is None


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


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("strategy", ["single", "tp"])
def test_the_files_of_a_dense_cell_are_accepted_with_the_sizes_of_the_cell(tmp_path, phase, strategy):
    plan = _dense_plan()
    cell = _cell(plan, phase, strategy)
    payloads = _dense_files(cell)
    recorded = payloads[0]["producer"]
    assert (recorded["tp_size"], recorded["attn_dp_size"], recorded["moe"]) == (cell.topology.tp, 1, None)
    _collected(tmp_path, plan, cell, payloads)

    # ``_validate`` passes what the plan says of the model, which is False here.
    assert len(_validate(plan, cell, tmp_path).points) == len(_points(phase))
    # The row builder and the summary the runner keeps make the same comparison, from the plan.
    rows = aggregate_cell(plan, cell, tmp_path, expected_attempt_id="attempt")
    assert [(row["parallel_strategy"], row["tp"]) for row in rows] == [(strategy, cell.topology.tp)] * len(rows)
    summary = runner._runtime_collection_summary(cell, tmp_path / "raw", backend="sglang", is_moe=plan_is_moe(plan))
    assert summary["measured_point_count"] == len(rows) == len(_points(phase))


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("strategy", ["single", "tp"])
def test_a_moe_block_with_the_expert_parallel_size_1_is_the_layout_of_a_dense_cell(tmp_path, phase, strategy):
    # The launcher took the model for one with expert layers and ran it with SGLang's default: every
    # tensor-parallel rank holds a slice of every expert. That is tensor parallelism alone, so the files pass.
    plan = _dense_plan()
    cell = _cell(plan, phase, strategy)
    payloads = _expert_layer_files(cell)
    assert payloads[0]["producer"]["moe"]["ep_size"] == 1
    _collected(tmp_path, plan, cell, payloads)

    assert len(_validate(plan, cell, tmp_path).points) == len(_points(phase))
    assert len(aggregate_cell(plan, cell, tmp_path, expected_attempt_id="attempt")) == len(_points(phase))


# name: (the strategy of the cell, the files of the pod, the change to their producer block, the refusal)
DIFFERS_FROM_THE_DENSE_CELL = {
    # The command reached the pod with another --tp-size, or without one: the launcher runs the size SGLang
    # resolved to the end, and the times are those of another number of GPUs.
    "tp-on-one-rank": ("tp", _dense_files, lambda producer: producer.update(tp_size=1), _not_the_dense_sizes(2, "1")),
    "tp-on-four-ranks": (
        "tp",
        _dense_files,
        lambda producer: producer.update(tp_size=4),
        _not_the_dense_sizes(2, "4"),
    ),
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
    "single-without-a-tensor-parallel-size": (
        "single",
        _dense_files,
        lambda producer: producer.pop("tp_size"),
        _not_the_dense_sizes(1, "None"),
    ),
    # True equals 1 and 2.0 equals 2, and neither is a size; nor is the written form of a number.
    "a-bool": ("single", _dense_files, lambda producer: producer.update(tp_size=True), _not_the_dense_sizes(1, "True")),
    "a-float": ("tp", _dense_files, lambda producer: producer.update(tp_size=2.0), _not_the_dense_sizes(2, "2.0")),
    "a-string": ("tp", _dense_files, lambda producer: producer.update(tp_size="2"), _not_the_dense_sizes(2, "'2'")),
    # A cell of a dense plan has one data-parallel rank, so the attention-DP size must be 1.
    "an-attention-dp-size-of-2": (
        "tp",
        _dense_files,
        lambda producer: producer.update(attn_dp_size=2),
        _not_the_dense_sizes(2, "2", "2"),
    ),
    "no-attention-dp-size": (
        "tp",
        _dense_files,
        lambda producer: producer.pop("attn_dp_size"),
        _not_the_dense_sizes(2, "2", "None"),
    ),
    "single-without-an-attention-dp-size": (
        "single",
        _dense_files,
        lambda producer: producer.pop("attn_dp_size"),
        _not_the_dense_sizes(1, "1", "None"),
    ),
    "an-attention-dp-size-that-is-a-bool": (
        "single",
        _dense_files,
        lambda producer: producer.update(attn_dp_size=True),
        _not_the_dense_sizes(1, "1", "True"),
    ),
    "an-attention-dp-size-of-none": (
        "tp",
        _dense_files,
        lambda producer: producer.update(attn_dp_size=None),
        _not_the_dense_sizes(2, "2", "None"),
    ),
    # The launcher found expert layers and the command reached the pod with an --ep-size of the tensor-parallel
    # size: the TEP layout, which the launcher accepts and no cell of a dense plan describes.
    "a-tep-run": (
        "tp",
        _expert_layer_files,
        _tep_on(2),
        _not_without_expert_parallelism("producer.moe.ep_size 2"),
    ),
    # The sizes are compared first: a TEP run of another GPU count is refused for its tensor-parallel size.
    "a-tep-run-on-four-ranks": ("tp", _expert_layer_files, _tep_on(4), _not_the_dense_sizes(2, "4")),
    # A moe block must hold the size as an int: True equals 1 and 1.0 equals 1, and neither is a size. A block
    # without one, and a block that is no mapping, say nothing of the layout.
    "an-expert-parallel-size-that-is-a-bool": (
        "single",
        _expert_layer_files,
        lambda producer: producer["moe"].update(ep_size=True),
        _not_without_expert_parallelism("producer.moe.ep_size True"),
    ),
    "an-expert-parallel-size-that-is-a-float": (
        "tp",
        _expert_layer_files,
        lambda producer: producer["moe"].update(ep_size=1.0),
        _not_without_expert_parallelism("producer.moe.ep_size 1.0"),
    ),
    "a-moe-block-without-the-size": (
        "tp",
        _expert_layer_files,
        lambda producer: producer["moe"].pop("ep_size"),
        _not_without_expert_parallelism("producer.moe.ep_size None"),
    ),
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


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("name", list(DIFFERS_FROM_THE_DENSE_CELL))
def test_a_dense_result_that_differs_from_its_cell_is_refused(tmp_path, phase, name):
    strategy, files, change, message = DIFFERS_FROM_THE_DENSE_CELL[name]
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


@pytest.mark.parametrize("producer", [None, [], "producer"])
def test_a_producer_block_that_is_no_mapping_holds_no_size_of_a_dense_cell(tmp_path, producer):
    # The validator refuses such a block for its release before this rule is reached, so the rule is called alone.
    cell = _cell(_dense_plan(), "decode", "tp")
    path = tmp_path / "benchmark.json"

    with pytest.raises(ValueError) as refused:
        native_artifact._validate_sglang_dense_producer(cell, producer, path)

    assert str(refused.value) == _not_the_dense_sizes(2, "None", "None") + str(path)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_the_recorded_tensor_parallel_size_tells_the_dense_cells_of_a_plan_apart(tmp_path, tp):
    # The plan of this suite ends at two GPUs, so the rule is called alone with the cells of a wider plan.
    template = _cell(_dense_plan(), "prefill", "tp")
    cells = {
        size: dataclasses.replace(template, topology=dataclasses.replace(template.topology, tp=size))
        for size in (1, 2, 4, 8)
    }
    path = tmp_path / "benchmark.json"

    # What a pod of each cell records is accepted for its own cell and refused for every other.
    for size, ran in cells.items():
        if size == tp:
            native_artifact._validate_sglang_dense_producer(cells[tp], _fake_dense_producer(ran), path)
            continue
        with pytest.raises(ValueError) as refused:
            native_artifact._validate_sglang_dense_producer(cells[tp], _fake_dense_producer(ran), path)
        assert str(refused.value) == _not_the_dense_sizes(tp, str(size)) + str(path)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_each_plan_holds_the_files_against_its_own_rule_and_not_against_the_other(tmp_path, phase):
    # The TEP cell of the MoE plan gives both file sets: two tensor-parallel ranks with dataset text.
    plan, _overrides = _resolved()
    assert plan_is_moe(plan) is True
    cell = _cell(plan, phase, "tep")

    # As the cell records it, with one expert-parallel rank per tensor-parallel rank. The rule of a MoE plan
    # accepts the files, so the rule of a dense plan did not run: it refuses them when it is asked.
    as_recorded = tmp_path / "as-recorded"
    written = _collected(as_recorded, plan, cell, _launcher_files(cell))
    assert len(_validate(plan, cell, as_recorded).points) == len(_points(phase))
    assert len(aggregate_cell(plan, cell, as_recorded, expected_attempt_id="attempt")) == len(_points(phase))
    with pytest.raises(ValueError) as refused:
        _validate(plan, cell, as_recorded, expected_moe=False)
    assert str(refused.value) == _not_without_expert_parallelism("producer.moe.ep_size 2") + written["rank_files"][0]

    # With the expert-parallel size 1, the pure TP layout. The rule of a MoE plan still runs and refuses the
    # files for the TEP cell; the rule of a dense plan accepts them when it is asked.
    without_ep = tmp_path / "without-ep"
    payloads = _launcher_files(cell)
    payloads[0]["producer"]["moe"]["ep_size"] = 1
    written = _collected(without_ep, plan, cell, payloads)
    with pytest.raises(ValueError) as refused:
        _validate(plan, cell, without_ep)
    assert str(refused.value) == _not_the_layout(2, 2, "2", "1") + written["rank_files"][0]
    with pytest.raises(ValueError, match="^" + re.escape(_not_the_layout(2, 2, "2", "1"))):
        aggregate_cell(plan, cell, without_ep, expected_attempt_id="attempt")
    assert len(_validate(plan, cell, without_ep, expected_moe=False).points) == len(_points(phase))
    # A caller without a plan asks for neither rule.
    for root in (as_recorded, without_ep):
        assert len(_validate(plan, cell, root, expected_moe=None).points) == len(_points(phase))


@pytest.mark.parametrize("faulty_rank", [0, 1])
def test_every_rank_file_is_held_against_the_cell_of_a_dense_plan(tmp_path, faulty_rank):
    # No dense plan has a cell with two rank files. The DEP cell of the MoE plan lends its two files here,
    # validated with the answer of a dense plan: its tensor-parallel size is 1.
    plan, _overrides = _resolved(presets=())
    cell = _cell(plan, "prefill", "dep")
    assert (cell.topology.tp, cell.topology.dp) == (1, 2)

    def files(faulty: int | None) -> list[dict]:
        payloads = _launcher_files(cell)
        for rank, payload in enumerate(payloads):
            payload["producer"] = _fake_dense_producer(cell)
            if rank == faulty:
                payload["producer"]["tp_size"] = 2
        return payloads

    _written_past_the_writer(tmp_path / "both-fit", plan, cell, files(None))
    assert len(_validate(plan, cell, tmp_path / "both-fit", expected_moe=False).points) == len(_points("prefill"))

    paths = _written_past_the_writer(tmp_path / "one-differs", plan, cell, files(faulty_rank))
    with pytest.raises(ValueError) as refused:
        _validate(plan, cell, tmp_path / "one-differs", expected_moe=False)
    assert str(refused.value) == _not_the_dense_sizes(1, "2") + str(paths[faulty_rank])


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


def test_the_labels_with_a_rule_are_bfloat16_fp8_and_nvfp4_and_the_label_int8_has_none():
    # The two tables of the rule, typed here on their own: the stored dtypes of each label, and the resolved
    # argument of the one label whose stored dtype does not say which KV cache it is.
    assert native_artifact._SGLANG_STORED_KV_DTYPES == {
        "bfloat16": ("torch.bfloat16", "torch.float16"),
        "fp8": ("torch.float8_e4m3fn",),
        "nvfp4": ("torch.float4_e2m1fn_x2",),
    }
    assert native_artifact._SGLANG_KV_CACHE_ARGUMENTS == {"nvfp4": "nvfp4"}
    # AISim's KV-cache labels: what an option value is normalised to, and the KV-cache modes AISim can infer
    # for a checkpoint. An option cannot name ``nvfp4``, and planning refuses the checkpoint AISim infers it
    # for (test_fpm_sglang_host.py), so no plan carries that label today.
    assert set(capabilities._KV_ALIASES.values()) - {"auto"} == {"bfloat16", "fp8", "int8"}
    labels = {*capabilities._KV_ALIASES.values(), *(mode.name for mode in common.KVCacheQuantMode)} - {"auto"}
    assert labels == {"bfloat16", "fp8", "int8", "nvfp4"}
    assert labels - set(native_artifact._SGLANG_STORED_KV_DTYPES) == {"int8"}
    # A label with a rule is one the cell builder writes a command for, with the argument the FAKE pods of
    # this suite record for it: two labels by its table, and ``nvfp4`` under its own name.
    translated = {label: recorded["kv_cache_dtype"] for label, recorded in POD_KV_CACHE.items()}
    assert translated == {**sglang_fpm_builder._KV_CACHE_DTYPES, "nvfp4": "nvfp4"}
    assert translated == {
        label: sglang_fpm_builder._kv_cache_dtype("--kv-cache-dtype", label) for label in POD_KV_CACHE
    }
    assert {label: recorded["kv_cache_stored_dtype"] for label, recorded in POD_KV_CACHE.items()} == {
        label: dtypes[0] for label, dtypes in native_artifact._SGLANG_STORED_KV_DTYPES.items()
    }
    # The argument the rule wants for ``nvfp4`` is the one such a pod records.
    assert native_artifact._SGLANG_KV_CACHE_ARGUMENTS["nvfp4"] == POD_KV_CACHE["nvfp4"]["kv_cache_dtype"]


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


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_a_float16_pool_is_accepted_for_a_bfloat16_cell_and_for_no_fp8_cell(tmp_path, kind, phase):
    plan = _plan_of(kind, "bfloat16", "fp8")
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
    # SGLang's resolution made ``fp8_e4m3`` of ``auto``, and the engine stored FP8.
    "resolution-made-fp8": (
        lambda producer: producer.update(kv_cache_dtype="fp8_e4m3", kv_cache_stored_dtype="torch.float8_e4m3fn"),
        "'torch.float8_e4m3fn'",
        "'fp8_e4m3'",
    ),
    # Dtypes no label stands for.
    "float32": (lambda producer: producer.update(kv_cache_stored_dtype="torch.float32"), "'torch.float32'", "'auto'"),
    "another-fp8-dtype": (
        lambda producer: producer.update(kv_cache_stored_dtype="torch.float8_e5m2"),
        "'torch.float8_e5m2'",
        "'auto'",
    ),
    # A block without the key is no record of a dtype: the driver writes it into every complete result file.
    "no-stored-dtype": (
        lambda producer: producer.pop("kv_cache_stored_dtype"),
        "no producer.kv_cache_stored_dtype",
        "'auto'",
    ),
    # A value that is not the text of a dtype: null, as the failed artifact of a run without accepted headers
    # holds it, a number, a list, the resolved argument, the name without its prefix, another letter case.
    "null": (lambda producer: producer.update(kv_cache_stored_dtype=None), "None", "'auto'"),
    "a-number": (lambda producer: producer.update(kv_cache_stored_dtype=16), "16", "'auto'"),
    "a-list": (
        lambda producer: producer.update(kv_cache_stored_dtype=["torch.bfloat16"]),
        "['torch.bfloat16']",
        "'auto'",
    ),
    "the-argument-in-its-place": (lambda producer: producer.update(kv_cache_stored_dtype="auto"), "'auto'", "'auto'"),
    "the-name-without-torch": (
        lambda producer: producer.update(kv_cache_stored_dtype="bfloat16"),
        "'bfloat16'",
        "'auto'",
    ),
    "another-letter-case": (
        lambda producer: producer.update(kv_cache_stored_dtype="torch.BFloat16"),
        "'torch.BFloat16'",
        "'auto'",
    ),
    # The refusal names the resolved argument as the block holds it, also when the block holds none.
    "neither-key": (
        lambda producer: [producer.pop(key) for key in ("kv_cache_dtype", "kv_cache_stored_dtype")],
        "no producer.kv_cache_stored_dtype",
        "None",
    ),
}


@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("name", list(NOT_THE_KV_CACHE_OF_A_BFLOAT16_CELL))
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


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_the_resolved_argument_of_a_bfloat16_cell_is_compared_with_nothing(tmp_path, kind):
    plan = _plan_of(kind)
    cell = _two_gpu_cell(plan, "prefill", "bfloat16")
    # Without the key, and with the argument of an fp8 cell beside the dtype of a bfloat16 cell.
    for name, change in (
        ("no-argument", lambda producer: producer.pop("kv_cache_dtype")),
        ("another-argument", lambda producer: producer.update(kv_cache_dtype="fp8_e4m3")),
        ("no-text", lambda producer: producer.update(kv_cache_dtype=None)),
    ):
        _written_past_the_writer(tmp_path / name, plan, cell, _files_of(plan, cell, change))
        assert len(_validate(plan, cell, tmp_path / name).points) == len(_points("prefill"))


@pytest.mark.parametrize("faulty_rank", [0, 1])
def test_every_rank_file_is_held_against_the_kv_cache_label_of_the_cell(tmp_path, faulty_rank):
    plan, _overrides = _resolved(presets=())
    cell = _cell(plan, "prefill", "dep")
    assert (cell.topology.dp, cell.kv_cache_dtype) == (2, "bfloat16")
    payloads = _launcher_files(cell)
    payloads[faulty_rank]["producer"]["kv_cache_stored_dtype"] = "torch.float8_e4m3fn"
    written = _collected(tmp_path, plan, cell, payloads)

    for keywords in ({}, {"expected_moe": None}):
        with pytest.raises(ValueError) as refused:
            _validate(plan, cell, tmp_path, **keywords)
        message = _not_the_kv_cache("bfloat16", "'torch.float8_e4m3fn'", "'auto'")
        assert str(refused.value) == message + written["rank_files"][faulty_rank]


@pytest.mark.parametrize("stored", ["torch.bfloat16", "torch.float8_e4m3fn", "torch.float4_e2m1fn_x2", None])
@pytest.mark.parametrize("label", ["int8", "float16", "auto", "fp8_e4m3", "fp4_mx_block16", "BFLOAT16", "NVFP4"])
def test_a_cell_whose_kv_cache_label_has_no_rule_is_refused_whatever_its_files_record(tmp_path, label, stored):
    # ``int8`` is a label an option can name; the others are no label planning gives, among them SGLang's name
    # of its other FP4 recipe and other spellings of labels with a rule. No cell of this suite carries one, so
    # a cell is relabelled.
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


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_the_files_of_an_nvfp4_cell_are_accepted_with_the_fp4_dtype_and_the_argument_nvfp4(tmp_path, kind, phase):
    plan = _plan_of(kind)
    cell, planned = _nvfp4_cell(plan, phase)
    payloads = _files_of(plan, planned, lambda producer: producer.update(POD_KV_CACHE["nvfp4"]))
    assert payloads[0]["producer"]["kv_cache_dtype"] == "nvfp4"
    assert payloads[0]["producer"]["kv_cache_stored_dtype"] == "torch.float4_e2m1fn_x2"
    _collected(tmp_path, plan, cell, payloads)

    # By the plan and for a caller without one.
    for keywords in ({}, {"expected_moe": None}):
        assert len(_validate(plan, cell, tmp_path, **keywords).points) == len(_points(phase))
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
    "the-argument-auto": (
        lambda producer: producer.update(kv_cache_dtype="auto"),
        "'torch.float4_e2m1fn_x2'",
        "'auto'",
    ),
    "another-letter-case": (
        lambda producer: producer.update(kv_cache_dtype="NVFP4"),
        "'torch.float4_e2m1fn_x2'",
        "'NVFP4'",
    ),
    # A block without the argument is no record of the recipe: the launcher writes it into every result file of
    # a run that got as far as building its engine.
    "no-argument": (lambda producer: producer.pop("kv_cache_dtype"), "'torch.float4_e2m1fn_x2'", "None"),
    "an-argument-that-is-null": (
        lambda producer: producer.update(kv_cache_dtype=None),
        "'torch.float4_e2m1fn_x2'",
        "None",
    ),
    "an-argument-that-is-a-list": (
        lambda producer: producer.update(kv_cache_dtype=["nvfp4"]),
        "'torch.float4_e2m1fn_x2'",
        "['nvfp4']",
    ),
    # The right argument beside another stored dtype.
    "an-fp8-pool": (
        lambda producer: producer.update(kv_cache_stored_dtype="torch.float8_e4m3fn"),
        "'torch.float8_e4m3fn'",
        "'nvfp4'",
    ),
    "a-bfloat16-pool": (
        lambda producer: producer.update(kv_cache_stored_dtype="torch.bfloat16"),
        "'torch.bfloat16'",
        "'nvfp4'",
    ),
    "the-name-without-torch": (
        lambda producer: producer.update(kv_cache_stored_dtype="float4_e2m1fn_x2"),
        "'float4_e2m1fn_x2'",
        "'nvfp4'",
    ),
    "no-stored-dtype": (
        lambda producer: producer.pop("kv_cache_stored_dtype"),
        "no producer.kv_cache_stored_dtype",
        "'nvfp4'",
    ),
    # What a bfloat16 cell and an fp8 cell record.
    "the-files-of-a-bfloat16-cell": (
        lambda producer: producer.update(POD_KV_CACHE["bfloat16"]),
        "'torch.bfloat16'",
        "'auto'",
    ),
    "the-files-of-an-fp8-cell": (
        lambda producer: producer.update(POD_KV_CACHE["fp8"]),
        "'torch.float8_e4m3fn'",
        "'fp8_e4m3'",
    ),
}


@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("name", list(NOT_THE_KV_CACHE_OF_AN_NVFP4_CELL))
def test_an_nvfp4_cell_is_refused_for_another_stored_dtype_and_for_another_resolved_argument(tmp_path, name, kind):
    change, found, resolved = NOT_THE_KV_CACHE_OF_AN_NVFP4_CELL[name]
    plan = _plan_of(kind)
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


@pytest.mark.parametrize("label", ["bfloat16", "fp8"])
@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_the_files_of_an_nvfp4_pod_are_no_result_of_a_bfloat16_cell_or_of_an_fp8_cell(tmp_path, kind, label):
    plan = _plan_of(kind, "bfloat16", "fp8")
    cell = _two_gpu_cell(plan, "decode", label)
    payloads = _files_of(plan, cell, lambda producer: producer.update(POD_KV_CACHE["nvfp4"]))
    written = _collected(tmp_path, plan, cell, payloads)

    with pytest.raises(ValueError) as refused:
        _validate(plan, cell, tmp_path)
    message = _not_the_kv_cache(label, "'torch.float4_e2m1fn_x2'", "'nvfp4'")
    assert str(refused.value) == message + written["rank_files"][0]


@pytest.mark.parametrize("producer", [None, [], "producer"])
def test_a_producer_block_that_is_no_mapping_holds_no_dtype_of_the_kv_pool(tmp_path, producer):
    # The validator refuses such a block for its release before this rule is reached, so the rule is called alone.
    cell = _two_gpu_cell(_plan_of("dense"), "decode", "bfloat16")
    path = tmp_path / "benchmark.json"

    with pytest.raises(ValueError) as refused:
        native_artifact._validate_sglang_kv_cache_dtype(cell, producer, path)

    assert str(refused.value) == _not_the_kv_cache("bfloat16", "no producer.kv_cache_stored_dtype", "None") + str(path)


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_result_files_written_before_the_pod_recorded_its_kv_cache_are_refused_for_every_caller(tmp_path, kind):
    # A launcher of an earlier revision wrote neither key. Its files belong to a plan of that revision.
    plan = _plan_of(kind)
    cell = _two_gpu_cell(plan, "decode", "bfloat16")

    def earlier(producer: dict) -> None:
        for key in ("kv_cache_dtype", "kv_cache_stored_dtype"):
            del producer[key]

    payloads = _files_of(plan, cell, earlier)

    # Under a plan made from this source the plan hash differs from the one in the pod's provenance file, and
    # the files are refused for their plan before the rule of the KV cache is reached.
    older = tmp_path / "plan-of-the-earlier-revision"
    pod = _pod(older, plan, cell)
    provenance = json.loads((pod / "collector-provenance.json").read_text())
    provenance["plan_sha256"] = "FAKE: the hash of the plan of the earlier revision"
    (pod / "collector-provenance.json").write_text(json.dumps(provenance))
    artifact.write_artifacts(str(pod / "benchmark.json"), payloads)
    with pytest.raises(ValueError, match="^Collector provenance plan mismatch: "):
        _validate(plan, cell, older)

    # Where the collector revision is held at one value the plan hash is the same, the files reach the rule,
    # and it refuses them for the missing key, with a plan and without one.
    same = tmp_path / "plan-with-the-revision-held"
    written = _collected(same, plan, cell, payloads)
    message = _not_the_kv_cache("bfloat16", "no producer.kv_cache_stored_dtype", "None")
    for keywords in ({}, {"expected_moe": None}):
        with pytest.raises(ValueError) as refused:
            _validate(plan, cell, same, **keywords)
        assert str(refused.value) == message + written["rank_files"][0]
    with pytest.raises(ValueError, match="^" + re.escape(message)):
        aggregate_cell(plan, cell, same, expected_attempt_id="attempt")


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


def test_a_moe_campaign_renders_the_expert_parallel_size_and_publishes_single_and_tep_rows(tmp_path, monkeypatch):
    plan, generator_overrides = _resolved()
    assert generator_overrides == {"K8sConfig": {"k8s_image": FAKE_IMAGE}}
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
    # Each cell was rendered by the Generator and the cell builder. The TEP cells carry the expert-parallel
    # size right after the tensor-parallel size; the one-GPU cells carry none.
    root = tmp_path / "artifacts" / plan.sha256[:16]
    for cell in plan.cells:
        run_script = (root / "cells" / cell.cell_id / "run.sh").read_text()
        (line,) = [line for line in run_script.splitlines() if line.startswith("engine_command=(")]
        command = shlex.split(line[len("engine_command=(") : -1])
        assert command[:2] == ["python3", "/tmp/fpm-bench/sglbench_launcher.py"]
        sizes = command[command.index("--tp-size") : command.index("--kv-cache-dtype")]
        if cell.parallel_strategy == "tep":
            assert sizes == ["--tp-size", "2", "--ep-size", "2"]
        else:
            assert sizes == ["--tp-size", "1"]
            assert "--ep-size" not in command
    # Every cell passed, and the record of a decode cell keeps the kvwarm block with the dataset.
    checkpoint = json.loads((tmp_path / "checkpoints" / "fpm_forward.json").read_text())
    assert {record["status"] for record in checkpoint["cells"].values()} == {"passed"}
    for cell in plan.cells:
        record = checkpoint["cells"][cell.cell_id]
        if cell.workload_kind == "decode":
            assert record["native_kvwarm"] == {**EXPECTED_KVWARM, "dataset": FAKE_DATASET}
        else:
            assert "native_kvwarm" not in record
    assert checkpoint["database"]["status"] == "passed"

    # The table under the data label of the plan holds the rows of the four cells with their labels.
    table = tmp_path / "systems" / SYSTEM / "sglang" / LABEL_WITH_TABLES
    metadata = json.loads((table / "fpm_forward_perf.metadata.json").read_text())
    assert (metadata["backend"], metadata["backend_version"]) == ("sglang", LABEL_WITH_TABLES)
    published = pq.read_table(table / "fpm_forward_perf.parquet")
    rows = published.to_pylist()
    assert metadata["row_count"] == len(rows) == 2 * (len(_points("prefill")) + len(_points("decode")))
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
        ("prefill", "tep", 2, 1, 1, 2, "n/a"): len(_points("prefill")),
        ("decode", "single", 1, 1, 1, 1, "real_kv"): len(_points("decode")),
        ("decode", "tep", 2, 1, 1, 2, "real_kv"): len(_points("decode")),
    }
    assert {row["model_path"] for row in rows} == {MODEL}
    # The table has no column for the content of the prompts.
    assert not [name for name in published.schema.names if "dataset" in name or "content" in name]


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
    assert checkpoint["database"]["status"] == "passed"

    # The table holds the rows of the eight cells with their labels: a DEP row says tp 1 and dp 2, and a pure
    # TP row says moe_tp 2 and moe_ep 1 where a TEP row says moe_tp 1 and moe_ep 2.
    table = tmp_path / "systems" / SYSTEM / "sglang" / LABEL_WITH_TABLES
    rows = pq.read_table(table / "fpm_forward_perf.parquet").to_pylist()
    assert len(rows) == 4 * (len(_points("prefill")) + len(_points("decode")))
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


def test_a_campaign_whose_dep_pod_ran_without_dp_attention_fails_that_cell(tmp_path, monkeypatch):
    plan, generator_overrides = _resolved(presets=())
    # The pod of the DEP decode cell recorded one attention-DP rank. A launcher writes one rank file per rank
    # it ran, so only a faulty launcher leaves two files with this block; the rule refuses them all the same.
    faulty = _cell(plan, "decode", "dep")
    without_dp_attention = {faulty.cell_id: lambda producer: producer.update(attn_dp_size=1)}
    monkeypatch.setattr(runner, "KubernetesCellRunner", _fake_cluster(plan, without_dp_attention))

    errors = runner.run_collection(
        plan,
        generator_overrides=generator_overrides,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        artifact_root=str(tmp_path / "artifacts"),
        resume=False,
        retry_failed=False,
        database_root=str(tmp_path / "systems"),
    )

    result = tmp_path / "artifacts" / plan.sha256[:16] / "cells" / faulty.cell_id / "raw" / "pod-0" / "benchmark.json"
    assert [(error["cell_id"], error["error_type"], error["error_message"]) for error in errors] == [
        (faulty.cell_id, "ValueError", _dep_refusal("2", "2", "1") + str(result))
    ]
    checkpoint = json.loads((tmp_path / "checkpoints" / "fpm_forward.json").read_text())
    assert {cell_id: record["status"] for cell_id, record in checkpoint["cells"].items()} == {
        cell.cell_id: "failed" if cell is faulty else "passed" for cell in plan.cells
    }
    # No table is written for a campaign with a failed cell.
    assert "database" not in checkpoint
    assert not (tmp_path / "systems").exists()


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


def test_a_cell_that_an_older_checkpoint_calls_passed_is_refused_when_its_rows_are_built(tmp_path, monkeypatch, caplog):
    plan, generator_overrides = _resolved()
    monkeypatch.setattr(runner, "KubernetesCellRunner", _fake_cluster(plan))
    settings = {
        "generator_overrides": generator_overrides,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "artifact_root": str(tmp_path / "artifacts"),
        "retry_failed": False,
        "database_root": str(tmp_path / "systems"),
    }
    # Every cell passes, and no table is written yet.
    assert runner.run_collection(plan, resume=False, publish_database=False, **settings) == []
    # The files of the TEP decode cell as a pod without --ep-size leaves them. A collector without this rule
    # called such a cell passed, and its checkpoint is the one a resume finds.
    tep = _cell(plan, "decode", "tep")
    result = tmp_path / "artifacts" / plan.sha256[:16] / "cells" / tep.cell_id / "raw" / "pod-0" / "benchmark.json"
    payload = json.loads(result.read_text())
    payload["producer"]["moe"]["ep_size"] = 1
    artifact.atomic_write_json(str(result), payload)
    refusal = _not_the_layout(2, 2, "2", "1") + str(result)

    def reject_rerun(*_arguments, **_keywords):
        raise AssertionError("a passed cell must not run again")

    monkeypatch.setattr(runner, "_render_cell", reject_rerun)
    with caplog.at_level("WARNING", logger=runner.logger.name):
        errors = runner.run_collection(plan, resume=True, **settings)

    # The record of the cell is not refreshed from files the rule refuses, and the row builder refuses them.
    assert [message for message in caplog.messages if "Skipping metadata refresh" in message] == [
        f"Skipping metadata refresh for passed FPM cell {tep.cell_id}: {refusal}"
    ]
    assert [(error["error_type"], error["error_message"]) for error in errors] == [("ValueError", refusal)]
    checkpoint = json.loads((tmp_path / "checkpoints" / "fpm_forward.json").read_text())
    assert (checkpoint["database"]["status"], checkpoint["database"]["error"]) == ("failed", refusal)
    assert not (tmp_path / "systems").exists()
