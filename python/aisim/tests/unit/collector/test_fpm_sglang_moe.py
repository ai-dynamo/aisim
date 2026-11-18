# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# The dataset record in the kvwarm block of a result file (FAKE_DATASET) has the key names of ai-dynamo/dynamo
# (Apache-2.0, https://github.com/ai-dynamo/dynamo), components/src/dynamo/vllm/instrumented_scheduler.py at
# b20fe6fe94021b854597965c7305931175e8f482. See THIRD_PARTY_NOTICES.md.

"""MoE cells of an SGLang self-benchmark collection, seen from the host.

One packaged MoE model is planned with the TEP preset on one and two GPUs through the functions of the entry
point, so its capability, the memory estimate of each topology and its backend policy are the ones a user
gets. The cells are rendered by the real Generator and the real cell builder, a cluster double leaves the
files of a launcher run, and the real validator, gate, row builder and table writer read them.

No SGLang, no GPU and no cluster are involved, and a fixture counts every attempt to open a network
connection. Every result file is a FAKE of ``test_fpm_sglbench_artifact.py``: its times were typed in and
never measured, and the dataset it names was never read.
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
from collector.fpm_forward import cli, native_artifact, planner, runner
from collector.fpm_forward.database import aggregate_cell
from collector.fpm_forward.entry import resolve_inputs
from collector.fpm_forward.native_artifact import plan_is_moe, validate_native_collection
from collector.fpm_forward.runtime import sglbench_artifact as artifact
from collector.fpm_forward.runtime import sglbench_content as content
from collector.fpm_forward.runtime_memory import saved_plan_identity
from collector.fpm_forward.types import KVWARM_STRATEGIES, ParallelTopology
from collector.model_cases import build_collection_case_plan

from aisim_core.sdk.fpm_identity import LEGACY_EXECUTION_IDENTITY

from .test_fpm_sglbench_artifact import (
    EXPECTED_KVWARM,
    EXPECTED_REGIMES,
    GATE_ACCEPT,
    POD_SGLANG_VERSION,
    _gate,
    _points,
    _rank_artifacts,
)

pytestmark = pytest.mark.unit

MODEL = "Qwen/Qwen3-30B-A3B"
SYSTEM = "h200_sxm"
FAKE_IMAGE = "example.invalid/sglang:test"
# The operator tables of this SGLang version are packaged for the system, so the planner gets a memory estimate
# for it. A data label without packaged tables, such as a release the launcher accepts, leaves the estimate
# unavailable and the cells planned.
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

    The host reads the release and, for a cell of a MoE plan, the content source and the two sizes
    (``native_artifact._validate_sglang_moe_producer``). Every other value is FAKE and has no reader.
    """
    return {
        "sglang_version": POD_SGLANG_VERSION,
        "tp_size": cell.topology.tp,
        # One scheduler process per tensor-parallel rank.
        "scheduler_processes": cell.topology.tp,
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


def _arguments(*extra: str, label: str = LABEL_WITH_TABLES) -> list[str]:
    """What a user passes to ``python -m collector.fpm_forward`` for the MoE cells of one and two GPUs."""
    return [
        *("--backend", "sglang", "--model-path", MODEL, "--gpu", SYSTEM),
        *("--fpm-max-gpus", "2", "--fpm-gpu-counts", "1", "2", "--fpm-parallel-presets", "tep"),
        *("--fpm-prefill-cudagraph-policy", "runtime"),
        *("--fpm-runtime-backend-version", POD_SGLANG_VERSION, "--fpm-backend-version", label),
        *("--generator-set", f"K8sConfig.k8s_image={FAKE_IMAGE}"),
        *extra,
    ]


def _resolved(*extra: str, label: str = LABEL_WITH_TABLES):
    """The plan and the deployment inputs as the entry point resolves them (cli.py ``main``)."""
    arguments = cli._parser().parse_args(_arguments(*extra, label=label))
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


def _launcher_files(cell) -> list[dict]:
    """The rank artifacts a launcher run leaves for ``cell``: one data-parallel rank, whatever the TP size.

    A decode artifact names the dataset in its kvwarm block, as a run with dataset text does. A prefill
    artifact has no such block.
    """
    assert cell.topology.dp == 1
    phase = cell.workload_kind
    keywords = {"producer": _fake_producer(cell)}
    if phase == "decode":
        keywords["kvwarm_dataset"] = dict(FAKE_DATASET)
    return _rank_artifacts(phase, _points(phase), dp=1, **keywords)


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
    # The capability of the packaged config: a MoE model, for which this backend plans no pure TP.
    capability = printed["capability"]
    assert (capability["architecture"], capability["is_moe"], capability["allow_pure_tp"]) == (
        "Qwen3MoeForCausalLM",
        True,
        False,
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


def test_a_data_label_without_packaged_tables_leaves_the_estimate_unknown_and_the_cells_planned():
    plan, _overrides = _resolved(label=POD_SGLANG_VERSION)

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


def test_the_default_preset_of_the_moe_model_is_refused_with_the_preset_to_pass(monkeypatch, capsys):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL)
    arguments = _arguments("--plan-only")
    preset = arguments.index("--fpm-parallel-presets")
    del arguments[preset : preset + 2]

    with pytest.raises(SystemExit) as refused:
        cli.main(arguments)

    assert refused.value.code == 2
    error = capsys.readouterr().err
    assert "error: SGLang FPM collection does not support DP attention yet (topology {'tp': 1, 'pp': 1, " in error
    assert "pass --fpm-parallel-presets tep to collect the tensor- and expert-parallel cells of a MoE model" in error


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
    # The DEP cell of the same two GPUs, which planning refuses for this backend: two data-parallel ranks.
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


def _not_the_layout(tp: int, ep: int, found_tp: str, found_ep: str) -> str:
    return (
        "native SGLang rank artifact of a MoE plan did not run the layout of its cell: "
        f"expected producer.tp_size {tp} and producer.moe.ep_size {ep}, found {found_tp} and {found_ep}: "
    )


def _no_expert_layers_found(producer: dict) -> None:
    """What a pod records whose SGLang finds no expert count in the model config: no keys and no moe block."""
    producer.update(expert_keys={}, moe=None)


def _seeded_ids(producer: dict) -> None:
    """Such a pod without a content word: the launcher sends seeded token ids."""
    _no_expert_layers_found(producer)
    producer["content"] = {"source": "seeded_token_ids", "seed": "0", "token_range": [1000, 2000], "issued": 7}


def _sizes(tp_size: int, ep_size: int):
    """The change that records these two sizes."""

    def change(producer: dict) -> None:
        producer["tp_size"] = tp_size
        producer["moe"]["ep_size"] = ep_size

    return change


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
    # The command reached the pod without --ep-size: every rank holds a slice of every expert.
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
}


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("name", list(DIFFERS_FROM_THE_CELL))
def test_a_moe_result_that_differs_from_its_cell_is_refused(tmp_path, phase, name):
    strategy, change, message = DIFFERS_FROM_THE_CELL[name]
    plan, _overrides = _resolved()
    assert plan_is_moe(plan) is True
    cell = _cell(plan, phase, strategy)
    (payload,) = _launcher_files(cell)
    change(payload["producer"])
    # Written past the launcher's own writer, which is not asked whether it would take the block.
    artifact.atomic_write_json(str(_pod(tmp_path, plan, cell) / "benchmark.json"), payload)
    refusal = "^" + re.escape(message)

    with pytest.raises(ValueError, match=refusal):
        _validate(plan, cell, tmp_path)
    # The row builder and the summary the runner keeps make the same comparison, from the plan.
    with pytest.raises(ValueError, match=refusal):
        aggregate_cell(plan, cell, tmp_path, expected_attempt_id="attempt")
    with pytest.raises(ValueError, match=refusal):
        runner._runtime_collection_summary(cell, tmp_path / "raw", backend="sglang", is_moe=plan_is_moe(plan))

    # A caller without a plan, and the plan of a model without expert layers, do not make the comparison: every
    # other rule accepts the files.
    for expected_moe in (None, False):
        collection = _validate(plan, cell, tmp_path, expected_moe=expected_moe)
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


@pytest.mark.parametrize("faulty_rank", [0, 1])
def test_every_rank_file_is_held_against_the_cell(tmp_path, faulty_rank):
    plan, _overrides = _resolved()
    tep = _cell(plan, "prefill", "tep")
    # A cell of two data-parallel ranks, which planning refuses for this backend: the only way to two rank files.
    cell = dataclasses.replace(tep, topology=dataclasses.replace(tep.topology, dp=2))
    payloads = _rank_artifacts("prefill", _points("prefill"), dp=2, producer=_fake_producer(cell))
    # The file of one rank says another expert-parallel size than the file of the other.
    payloads[faulty_rank]["producer"]["moe"]["ep_size"] = 1
    written = _collected(tmp_path, plan, cell, payloads)

    with pytest.raises(ValueError) as refused:
        _validate(plan, cell, tmp_path)

    assert str(refused.value) == _not_the_layout(2, 2, "2", "1") + written["rank_files"][faulty_rank]
    assert len(_validate(plan, cell, tmp_path, expected_moe=None).points) == len(_points("prefill"))


@pytest.mark.parametrize("producer", [None, [], "producer"])
def test_a_producer_block_that_is_no_mapping_did_not_send_text(tmp_path, producer):
    # The validator refuses such a block for its release before this rule is reached, so the rule is called alone.
    plan, _overrides = _resolved()
    path = tmp_path / "benchmark.json"

    with pytest.raises(ValueError) as refused:
        native_artifact._validate_sglang_moe_producer(_cell(plan, "decode", "tep"), producer, path)

    assert str(refused.value) == _not_text("None") + str(path)


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
