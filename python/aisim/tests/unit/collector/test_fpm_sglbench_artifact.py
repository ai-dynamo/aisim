# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# The expected result envelope follows ai-dynamo/dynamo (Apache-2.0, https://github.com/ai-dynamo/dynamo),
# components/src/dynamo/vllm/instrumented_scheduler.py and worker_factory.py at
# b20fe6fe94021b854597965c7305931175e8f482. See THIRD_PARTY_NOTICES.md.

"""CPU-only checks of ``runtime/sglbench_artifact.py`` against AISim's real checkers.

No SGLang and no GPU are involved. Every time value in this file is FAKE: it was typed in to exercise the
artifact arithmetic and was never measured on an engine.

The two checkers are used as shipped, not copied:

- the validator, ``collector.fpm_forward.native_artifact.validate_native_collection``;
- the gate, the Python text inside ``check_result_files()`` of ``runtime/fpm_exec.sh``, run in a subprocess.
"""

from __future__ import annotations

import ast
import collections
import copy
import decimal
import enum
import fnmatch
import functools
import itertools
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

import pytest
from collector.fpm_forward import native_artifact
from collector.fpm_forward.capabilities import GENERIC_RUNTIME_TEMPLATE_ID
from collector.fpm_forward.database import aggregate_cell
from collector.fpm_forward.measurement_evidence import extract_measurement_evidence
from collector.fpm_forward.native_artifact import select_native_measurements, validate_native_collection
from collector.fpm_forward.runner import _runtime_timing_summary
from collector.fpm_forward.runtime import sglbench_artifact as artifact

from aisim.fpm_contract import (
    FPM_BENCHMARK_RESULT_GLOB,
    FPM_NATIVE_BENCHMARK_RESULT_SCHEMA_VERSION,
    fpm_benchmark_result_name,
)

from .test_fpm_runner import _cell, _plan

pytestmark = pytest.mark.unit

RUNTIME = Path(__file__).resolve().parents[3] / "collector/fpm_forward/runtime"
FPM_EXEC = RUNTIME / "fpm_exec.sh"

# Exit statuses of the gate's checker (fpm_exec.sh: pending_status = 10, then 11, invalid() exits 20).
GATE_ACCEPT, GATE_RANK_PENDING, GATE_MERGED_PENDING, GATE_INVALID = 0, 10, 11, 20

# The SGLang release of the FAKE pod, one of the two the launcher accepts. The collector records the installed
# release in the provenance file, the launcher writes the release it runs on into the producer block of every
# rank artifact (sglbench_launcher.py, _start_up_checks), and the validator requires the two to agree for an
# SGLang cell.
POD_SGLANG_VERSION = "0.5.23"
# What the FAKE pod of a cell records about its KV cache, by the KV-cache label of the cell: the argument SGLang
# resolves for the command the cell builder writes (``auto`` for bfloat16, ``fp8_e4m3`` for fp8), which the
# launcher records, and the text of the torch dtype the engine then builds its KV pool with for a model that
# loads in bfloat16, which the recorder reports and the driver records. The validator requires the stored
# dtype to be one the label of the cell stands for. Typed here; no engine wrote these values. The third label
# is the one AISim infers for a checkpoint with an NVFP4 KV cache of its own, which no plan carries today: the
# cell builder passes it on under its own name, SGLang stores its FP4 dtype for it, and the validator wants
# the argument as well.
POD_KV_CACHE = {
    "bfloat16": {"kv_cache_dtype": "auto", "kv_cache_stored_dtype": "torch.bfloat16"},
    "fp8": {"kv_cache_dtype": "fp8_e4m3", "kv_cache_stored_dtype": "torch.float8_e4m3fn"},
    "nvfp4": {"kv_cache_dtype": "nvfp4", "kv_cache_stored_dtype": "torch.float4_e2m1fn_x2"},
}
# The keys of the producer block that the validator asks of every rank artifact of an SGLang cell. The KV
# cache is that of an fp8 cell, which the stand-in cell of test_fpm_runner.py is.
LAUNCHER_PRODUCER = {"sglang_version": POD_SGLANG_VERSION, **POD_KV_CACHE["fp8"]}

FAKE_RUN_ID = "fake-run-1"
FAKE_GRID_DIGEST = "fake-grid-digest"
FAKE_ELAPSED_SECONDS = 12.5

# The top-level keys the module must never write, because an AISim reader expects vLLM-shaped content under each
# of them (the comment at FORBIDDEN_TOP_LEVEL_KEYS in the module names the readers). Typed here independently
# of the module.
BANNED_TOP_LEVEL_KEYS = {
    "measurement_protocol",
    "warmup_evidence",
    "engine",
    "cudagraph",
    "limits",
    "recurrent_state",
    "worker_probe_path",
    "input_provenance",
    "execution_mode",
    "warmup_results",
}

ALWAYS_WRITTEN_KEYS = {
    "schema_version",
    "artifact_type",
    "status",
    "valid",
    "usable",
    "timing_valid",
    "stop_reason",
    "error",
    "errors",
    "run_id",
    "grid_digest",
    "config",
    "coverage",
    "dp",
    "results",
    "iteration_groups",
    "skipped_points",
    "missing_phases",
    "timing",
    "measurement_policy",
    "producer",
}

# The kvwarm block of a decode or agg artifact, typed here independently of the module.
EXPECTED_KVWARM = {"enabled": True, "warm_eligible": True, "skip_reason": None, "method": "radix_cache_seed"}

# A FAKE record of the dataset whose text the prompts of a run were made of, as a caller hands it over: no such
# file exists and nothing was hashed. The keys are those of the record Dynamo's vLLM self-benchmark writes into
# its kvwarm block (instrumented_scheduler.py at the revision in the header of this file, _kvwarm_load_texts).
FAKE_KVWARM_DATASET = {
    "path": "/fake-cache/fpm_datasets/fake_dataset.json",
    "sha256": "0123456789abcdef" * 4,
    "collection_pool": "conversation_sha256_even",
    "conversations": 1234,
}
# The kvwarm block of a decode or agg artifact of such a run, typed here independently of the module.
EXPECTED_KVWARM_WITH_DATASET = {
    "enabled": True,
    "warm_eligible": True,
    "skip_reason": None,
    "method": "radix_cache_seed",
    "dataset": {
        "path": "/fake-cache/fpm_datasets/fake_dataset.json",
        "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "collection_pool": "conversation_sha256_even",
        "conversations": 1234,
    },
}


# ---------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------


@functools.cache
def _gate_checker_source() -> str:
    script = FPM_EXEC.read_text(encoding="utf-8")  # the script holds a non-ASCII dash
    assert script.count("check_result_files() {") == 1
    return script.split("check_result_files() {", 1)[1].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]


def _gate(rank_paths, mode: str, *, start_rank: int = 0) -> subprocess.CompletedProcess:
    """Run the shipped checker as fpm_exec.sh does: ``python3 - <start_rank> <mode> <schema> <rank paths...>``."""
    return subprocess.run(
        [
            sys.executable,
            "-",
            str(start_rank),
            mode,
            str(FPM_NATIVE_BENCHMARK_RESULT_SCHEMA_VERSION),
            *(str(path) for path in rank_paths),
        ],
        input=_gate_checker_source(),
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )


def _gate_shell_rank_path(output_path: str, dp_rank: int) -> str:
    """The rank path as the gate's own shell function ``benchmark_path_for_dp_rank`` derives it."""
    script = FPM_EXEC.read_text(encoding="utf-8")  # the script holds a non-ASCII dash
    assert script.count("benchmark_path_for_dp_rank() {") == 1
    body = script.split("benchmark_path_for_dp_rank() {", 1)[1].split("\n}\n", 1)[0]
    program = f'benchmark_path_for_dp_rank() {{{body}\n}}\nbenchmark_path_for_dp_rank "$1"\n'
    done = subprocess.run(
        ["bash", "-c", program, "bash", str(dp_rank)],
        env={**os.environ, "FPM_BENCHMARK_OUTPUT_PATH": output_path},
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    return done.stdout.removesuffix("\n")


def _point(phase: str, benchmark_id: int, batch: int, total_new: int, total_kv: int, reasons=()) -> dict:
    return {
        "point_type": phase,
        "benchmark_id": benchmark_id,
        "total_prefill_tokens": total_new,
        "total_kv_read_tokens": total_kv,
        "batch_size": batch,
        "sample_reasons": list(reasons),
    }


def _prefill_points() -> list[dict]:
    return [
        _point("prefill", 1, 2, 256, 128, ["prefill_real_seed"]),  # 2 requests of 128 new and 64 cached tokens
        _point("prefill", 2, 2, 256, 0),  # the same without cached tokens
        _point("prefill", 3, 1, 48, 32, ["explicit", "prefill_real_seed"]),  # manifest point, cached
        _point("prefill", 4, 4, 64, 0, ["explicit"]),  # manifest point, not cached
    ]


def _decode_points() -> list[dict]:
    return [
        _point("decode", 1, 2, 0, 2048, ["kvwarm_real_kv"]),  # 2 requests with 1024 past tokens each
        _point("decode", 2, 2, 0, 8, ["kvwarm_real_kv"]),  # the 4-token minimum context
        _point("decode", 3, 3, 0, 300, ["explicit", "kvwarm_real_kv"]),  # manifest point
    ]


def _points(phase: str) -> list[dict]:
    return _prefill_points() if phase == "prefill" else _decode_points()


# The regimes of the points above, written out by hand: real_prefix for a prefill point with cached tokens,
# not_applicable for one without, real_kv for a decode point.
EXPECTED_REGIMES = {
    "prefill": ["real_prefix", "not_applicable", "real_prefix", "not_applicable"],
    "decode": ["real_kv", "real_kv", "real_kv"],
}


def _fake_wall_time(benchmark_id: int, rank: int) -> float:
    """FAKE seconds. Rank 0 is the slower rank on odd ids and rank 1 on even ids."""
    return 0.004 + 0.0007 * benchmark_id + (0.0003 if (benchmark_id + rank) % 2 else 0.0)


def _fake_decode_steps(benchmark_id: int, rank: int) -> list[float]:
    """FAKE seconds for six decode steps: a short first step, a short last step, unsorted kept steps."""
    base = _fake_wall_time(benchmark_id, rank)
    return [base * 0.5, base * 1.4, base * 1.2, base * 0.9, base, base * 0.3]


def _expected_wall_time(point: dict, rank: int) -> float:
    """The sample the artifact must carry, worked out without the module."""
    if point["point_type"] == "decode":
        # Decode steps 3 to 5 are the list items 2, 3 and 4; the upper median of three values is the middle one.
        return statistics.median_high(_fake_decode_steps(point["benchmark_id"], rank)[2:5])
    return _fake_wall_time(point["benchmark_id"], rank)


def _measurement(point: dict, dp: int) -> dict:
    fpms = {}
    for rank in range(dp):
        if point["point_type"] == "decode":
            steps = _fake_decode_steps(point["benchmark_id"], rank)
            extra = {"sglbench_decode_steps": steps, "worker_id": f"fake-worker-{rank}"}
            fpms[rank] = artifact.build_fpm(point, rank, artifact.decode_wall_time(steps), extra=extra)
        else:
            fpms[rank] = artifact.build_fpm(point, rank, _fake_wall_time(point["benchmark_id"], rank))
    return artifact.build_measurement(point, fpms)


def _build_kwargs(mode: str, points: list[dict], *, dp: int = 1, completed: int | None = None) -> dict:
    measured = points if completed is None else points[:completed]
    return {
        "dp_rank": 0,
        "dp_size": dp,
        "mode": mode,
        "run_id": FAKE_RUN_ID,
        "grid_digest": FAKE_GRID_DIGEST,
        "expected_points": len(points),
        "measurements": [_measurement(point, dp) for point in measured],
        "benchmark_elapsed_seconds": FAKE_ELAPSED_SECONDS,
    }


def _rank_artifacts(mode: str, points: list[dict], *, dp: int = 1, completed: int | None = None, **extra) -> list[dict]:
    kwargs = {**_build_kwargs(mode, points, dp=dp, completed=completed), **extra}
    # Without a producer block of its own, an artifact carries what the launcher always adds to one.
    kwargs.setdefault("producer", LAUNCHER_PRODUCER)
    elapsed = kwargs.pop("benchmark_elapsed_seconds")
    kwargs.pop("dp_rank")
    # Ranks may report different elapsed times; the validator allows that.
    return [
        artifact.build_rank_artifact(dp_rank=rank, benchmark_elapsed_seconds=elapsed + rank, **kwargs)
        for rank in range(dp)
    ]


def _pod(root: Path, cell, *, sglang_version: str = POD_SGLANG_VERSION) -> Path:
    """The collected layout the validator needs: ``<raw>/<pod>/benchmark*.json`` beside a provenance file.

    The provenance is the one the collector writes in the pod of an SGLang cell (runner.py,
    ``_attempt_provenance_command``): backend ``sglang`` and the version of the installed distribution.
    """
    pod = root / "raw" / "pod-0"
    pod.mkdir(parents=True)
    provenance = {
        "schema_name": "aic_fpm_collector_provenance",
        "schema_version": 1,
        "cell_id": cell.cell_id,
        "plan_sha256": "plan-sha",
        "attempt_id": "attempt",
        "runtime": {"backend": "sglang", "backend_version": sglang_version},
    }
    (pod / "collector-provenance.json").write_text(json.dumps(provenance))
    return pod


def _validate(cell, root: Path):
    """The validator as a caller without a plan calls it for an SGLang cell: it says nothing of expert layers.

    With a plan the host also passes ``expected_moe``, True for a MoE model and False for a dense one, and
    what the pod recorded of its layout is then held against the cell (test_fpm_sglang_moe.py).
    """
    return validate_native_collection(
        cell, root / "raw", expected_plan_sha256="plan-sha", expected_attempt_id="attempt", expected_backend="sglang"
    )


def _sglang_plan(cell):
    """The helper plan of the vLLM runner suite, with the backend of an SGLang collection."""
    plan = _plan(cell)
    plan.backend = "sglang"
    return plan


def _result_files(pod: Path) -> list[str]:
    return sorted(path.name for path in pod.iterdir() if path.name != "collector-provenance.json")


# ---------------------------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------------------------


def test_constants_have_the_expected_values_and_match_aisim():
    assert artifact.SCHEMA_VERSION == FPM_NATIVE_BENCHMARK_RESULT_SCHEMA_VERSION == 2
    assert (artifact.STATUS_COMPLETE, artifact.STATUS_PARTIAL, artifact.STATUS_FAILED) == (
        "complete",
        "partial",
        "failed",
    )
    assert artifact.STOP_REASON_TIMEOUT == "timeout"
    assert artifact.DECODE_RAW_STEPS == 6
    assert artifact.DECODE_KEPT_STEP_INDICES == (2, 3, 4)
    assert artifact.DECODE_STEPS_KEY == "sglbench_decode_steps"
    assert artifact.MEASUREMENT_POLICY == {
        "prefill": "single_extend_batch",
        "decode": "upper_median_of_decode_steps_3_to_5",
    }
    assert set(artifact.FORBIDDEN_TOP_LEVEL_KEYS) == BANNED_TOP_LEVEL_KEYS


# ---------------------------------------------------------------------------------------------------------------
# Complete artifacts: both checkers accept
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["single", "tp", "dep"])
@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_complete_artifacts_pass_the_validator_and_the_gate(tmp_path, phase, dp, strategy):
    # "single" and "tp" are what a dense SGLang cell gets; "dep" makes the kvwarm block mandatory for decode.
    cell = _cell(phase=phase, dp=dp, strategy=strategy)
    pod = _pod(tmp_path, cell)
    points = _points(phase)
    output_path = str(pod / "benchmark.json")

    written = artifact.write_artifacts(output_path, _rank_artifacts(phase, points, dp=dp))

    collection = _validate(cell, tmp_path)
    assert [measurement.point for measurement in collection.points] == points
    assert [measurement.kv_seed_regime for measurement in collection.points] == EXPECTED_REGIMES[phase]
    for point, measurement in zip(points, collection.points, strict=True):
        # The floats reach the reader unrounded.
        assert measurement.rank_wall_times == tuple((rank, _expected_wall_time(point, rank)) for rank in range(dp))
    assert (collection.runtime_run_id, collection.runtime_grid_digest) == (FAKE_RUN_ID, FAKE_GRID_DIGEST)
    assert [elapsed for _rank, elapsed, _measured in collection.rank_timings] == [
        FAKE_ELAPSED_SECONDS + rank for rank in range(dp)
    ]
    if phase == "decode":
        assert collection.kvwarm_meta == {
            "enabled": True,
            "warm_eligible": True,
            "skip_reason": None,
            "method": "radix_cache_seed",
        }
    else:
        assert collection.kvwarm_meta is None

    gate = _gate(written["rank_files"], phase)
    assert gate.returncode == GATE_ACCEPT, gate.stderr
    assert written == {
        "rank_files": [fpm_benchmark_result_name(output_path, rank) for rank in range(dp)],
        "merged_output_path": str(pod / "benchmark_merged.json"),
    }
    expected_files = ["benchmark.json", "benchmark_merged.json"] + [f"benchmark_dp{rank}.json" for rank in range(1, dp)]
    assert _result_files(pod) == sorted(expected_files)


@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_a_run_of_exactly_one_point_is_complete_and_passes_both_checkers(tmp_path, phase, dp):
    """The smallest complete run, which is what a one-point manifest produces."""
    cell = _cell(phase=phase, dp=dp, strategy="dep")
    pod = _pod(tmp_path, cell)
    points = _points(phase)[:1]

    payloads = _rank_artifacts(phase, points, dp=dp)
    for payload in payloads:
        assert (payload["status"], payload["valid"], payload["usable"]) == ("complete", True, True)
        assert payload["coverage"] == {"expected_points": 1, "completed_points": 1, "skipped_points": 0}
        assert len(payload["results"]) == len(payload["iteration_groups"]) == 1
    written = artifact.write_artifacts(str(pod / "benchmark.json"), payloads)

    collection = _validate(cell, tmp_path)
    assert [measurement.point for measurement in collection.points] == points
    assert [measurement.kv_seed_regime for measurement in collection.points] == EXPECTED_REGIMES[phase][:1]
    gate = _gate(written["rank_files"], phase)
    assert gate.returncode == GATE_ACCEPT, gate.stderr


def test_rank_artifact_envelope_for_a_complete_run():
    points = _decode_points()
    rank0, rank1 = _rank_artifacts(
        "decode", points, dp=2, config={"decode_max_batch_size_samples": 128}, producer={"page_size": 64}
    )

    assert set(rank0) == ALWAYS_WRITTEN_KEYS | {"kvwarm"}
    assert rank0["artifact_type"] == "rank"
    assert (rank0["status"], rank0["valid"], rank0["usable"], rank0["timing_valid"]) == ("complete", True, True, True)
    assert (rank0["stop_reason"], rank0["error"], rank0["errors"]) == (None, None, [])
    assert (rank0["skipped_points"], rank0["missing_phases"]) == ([], [])
    assert rank0["config"] == {"mode": "decode", "decode_max_batch_size_samples": 128}
    assert rank0["producer"] == {"name": "aisim_sglbench", "artifact_format": 1, "page_size": 64}
    assert rank0["coverage"] == {"expected_points": 3, "completed_points": 3, "skipped_points": 0}
    assert all(type(value) is int for value in rank0["coverage"].values())
    assert (rank0["dp"], rank1["dp"]) == ({"size": 2, "rank": 0}, {"size": 2, "rank": 1})
    assert rank0["measurement_policy"] == artifact.MEASUREMENT_POLICY
    assert rank0["measurement_policy"] is not artifact.MEASUREMENT_POLICY

    # Rows hold the rank's own fpm; groups hold both and are identical in both rank files.
    assert rank0["iteration_groups"] == rank1["iteration_groups"]
    for index, point in enumerate(points):
        group = rank0["iteration_groups"][index]
        assert group["benchmark_id"] == point["benchmark_id"] == index + 1
        assert group["point"] == point
        assert (group["expected_dp_ranks"], group["complete"]) == ([0, 1], True)
        assert [item["dp_rank"] for item in group["rank_results"]] == [0, 1]
        for rank, payload in enumerate((rank0, rank1)):
            row = payload["results"][index]
            assert set(row) == {"point", "kv_seed_regime", "fpms"}
            assert row["point"] == point
            assert row["kv_seed_regime"] == "real_kv"
            assert row["fpms"] == group["rank_results"][rank]["fpms"]
            (fpm,) = row["fpms"]
            assert fpm == {
                "counter_id": point["benchmark_id"],
                "dp_rank": rank,
                "wall_time": _expected_wall_time(point, rank),
                "scheduled_requests": {
                    "num_prefill_requests": 0,
                    "sum_prefill_tokens": 0,
                    "sum_prefill_kv_tokens": 0,
                    "num_decode_requests": point["batch_size"],
                    "sum_decode_kv_tokens": point["total_kv_read_tokens"],
                },
                "sglbench_decode_steps": _fake_decode_steps(point["benchmark_id"], rank),
                "worker_id": f"fake-worker-{rank}",
            }
        # The group time is the slower rank's float, untouched.
        slowest = max(_expected_wall_time(point, rank) for rank in (0, 1))
        assert group["wall_time"] == slowest
    slower_ranks = [max((0, 1), key=lambda rank, point=point: _expected_wall_time(point, rank)) for point in points]
    assert set(slower_ranks) == {0, 1}, "the fake times must make each rank the slower one at least once"


def test_prefill_artifact_has_no_kvwarm_block_and_agg_has_one():
    assert set(_rank_artifacts("prefill", _prefill_points())[0]) == ALWAYS_WRITTEN_KEYS
    agg_points = _prefill_points() + [
        {**point, "benchmark_id": point["benchmark_id"] + 4} for point in _decode_points()
    ]
    assert "kvwarm" in _rank_artifacts("agg", agg_points)[0]


# ---------------------------------------------------------------------------------------------------------------
# The dataset record of a run whose prompts were dataset text
# ---------------------------------------------------------------------------------------------------------------


def _agg_points() -> list[dict]:
    return _prefill_points() + [{**point, "benchmark_id": point["benchmark_id"] + 4} for point in _decode_points()]


# name -> (mode, points of the grid, keyword arguments that end the run, status)
DATASET_RUNS = {
    "decode-complete": ("decode", _decode_points, {}, "complete"),
    "agg-complete": ("agg", _agg_points, {}, "complete"),
    "decode-partial": ("decode", _decode_points, {"completed": 1, "stop_reason": "timeout"}, "partial"),
    "decode-failed": (
        "decode",
        _decode_points,
        {"completed": 1, "failure": {"point": None, "reason": "fake reason"}},
        "failed",
    ),
    "agg-missing-phase": ("agg", _prefill_points, {"missing_phases": ("decode",)}, "failed"),
}


@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("name", sorted(DATASET_RUNS))
def test_the_kvwarm_block_carries_the_dataset_record_it_is_given_and_nothing_else_changes(name, dp):
    mode, points, ending, status = DATASET_RUNS[name]
    # The caller's dict, with its keys in another order than the artifact writes them.
    given = {key: FAKE_KVWARM_DATASET[key] for key in ("conversations", "collection_pool", "sha256", "path")}

    plain = _rank_artifacts(mode, points(), dp=dp, **ending)
    with_none = _rank_artifacts(mode, points(), dp=dp, kvwarm_dataset=None, **ending)
    with_record = _rank_artifacts(mode, points(), dp=dp, kvwarm_dataset=given, **ending)

    for rank in range(dp):
        assert plain[rank]["status"] == status
        # Without a record the artifact is the one that is built without the argument, to the byte.
        assert plain[rank]["kvwarm"] == EXPECTED_KVWARM
        assert json.dumps(with_none[rank]) == json.dumps(plain[rank])
        # With one, the block gains the key "dataset" and the artifact differs in nothing else.
        block = with_record[rank]["kvwarm"]
        assert block == EXPECTED_KVWARM_WITH_DATASET
        assert json.dumps({**with_record[rank], "kvwarm": EXPECTED_KVWARM}) == json.dumps(plain[rank])
        # The keys are written in one order, whatever the order of the caller's dict.
        assert list(block) == ["enabled", "warm_eligible", "skip_reason", "method", "dataset"]
        assert list(block["dataset"]) == ["path", "sha256", "collection_pool", "conversations"]
        assert type(block["dataset"]["conversations"]) is int
    # The artifact holds a copy of the record, and each rank a block of its own.
    given.update(path="/fake-cache/edited.json", fake_note="added after the build")
    assert [payload["kvwarm"] for payload in with_record] == [EXPECTED_KVWARM_WITH_DATASET] * dp
    with_record[0]["kvwarm"]["dataset"]["conversations"] = 1
    assert all(payload["kvwarm"] == EXPECTED_KVWARM_WITH_DATASET for payload in with_record[1:])


@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("strategy", ["single", "tp", "tep"])
def test_a_decode_artifact_with_a_dataset_record_passes_both_checkers_and_is_published_as_before(
    tmp_path, strategy, dp
):
    # "tep" is the strategy of a cell with expert parallelism: for it the validator needs the kvwarm block.
    cell = _cell(phase="decode", dp=dp, strategy=strategy)
    points = _decode_points()
    rows = {}
    for label, dataset in (("ids", None), ("text", FAKE_KVWARM_DATASET)):
        root = tmp_path / label
        pod = _pod(root, cell)
        payloads = _rank_artifacts("decode", points, dp=dp, kvwarm_dataset=dataset)
        written = artifact.write_artifacts(str(pod / "benchmark.json"), payloads)
        expected_block = EXPECTED_KVWARM if dataset is None else EXPECTED_KVWARM_WITH_DATASET

        # Every rank file and the merged file carry the block.
        for path in (*written["rank_files"], written["merged_output_path"]):
            assert json.loads(Path(path).read_text(encoding="utf-8"))["kvwarm"] == expected_block
        # The validator reads three keys of the block and hands the whole of it on, the record included.
        collection = _validate(cell, root)
        assert collection.kvwarm_meta == expected_block
        assert [measurement.point for measurement in collection.points] == points
        assert [measurement.kv_seed_regime for measurement in collection.points] == EXPECTED_REGIMES["decode"]
        gate = _gate(written["rank_files"], "decode")
        assert gate.returncode == GATE_ACCEPT, gate.stderr
        rows[label] = aggregate_cell(_sglang_plan(cell), cell, root, expected_attempt_id="attempt")

    # Publication does not read the record: the rows are those of the run without one.
    assert rows["text"] == rows["ids"]
    assert [row["kv_seed_regime"] for row in rows["text"]] == ["real_kv"] * len(points)
    assert {row["parallel_strategy"] for row in rows["text"]} == {strategy}


def test_an_agg_artifact_with_a_dataset_record_passes_the_gate_in_agg_mode(tmp_path):
    payloads = _rank_artifacts("agg", _agg_points(), dp=2, kvwarm_dataset=FAKE_KVWARM_DATASET)
    written = artifact.write_artifacts(str(tmp_path / "benchmark.json"), payloads)
    assert _gate(written["rank_files"], "agg").returncode == GATE_ACCEPT
    merged = json.loads(Path(written["merged_output_path"]).read_text(encoding="utf-8"))
    assert merged["kvwarm"] == EXPECTED_KVWARM_WITH_DATASET


def _dataset(**changes) -> dict:
    return {**FAKE_KVWARM_DATASET, **changes}


def _dataset_without(key: str) -> dict:
    return {name: value for name, value in FAKE_KVWARM_DATASET.items() if name != key}


NOT_A_SHA256 = r"\['sha256'\] must be the 64 lower-case hexadecimal digits of a SHA-256, got "
EXACT_KEYS = r" must have exactly the keys \['path', 'sha256', 'collection_pool', 'conversations'\], got "

# name -> (what is given as the record, exception, the message after the place of the record)
BAD_KVWARM_DATASETS = {
    "a-list-of-pairs": (list(FAKE_KVWARM_DATASET.items()), TypeError, r" must be a dict, got list$"),
    "a-path": ("/fake-cache/fpm_datasets/fake_dataset.json", TypeError, r" must be a dict, got str$"),
    "true": (True, TypeError, r" must be a dict, got bool$"),
    "no-keys": ({}, ValueError, EXACT_KEYS + r"\[\]$"),
    "path-missing": (
        _dataset_without("path"),
        ValueError,
        EXACT_KEYS + r"\['collection_pool', 'conversations', 'sha256'\]$",
    ),
    "sha256-missing": (_dataset_without("sha256"), ValueError, EXACT_KEYS),
    "collection-pool-missing": (_dataset_without("collection_pool"), ValueError, EXACT_KEYS),
    "conversations-missing": (_dataset_without("conversations"), ValueError, EXACT_KEYS),
    # A fifth key, as the launcher's own record of the dataset has: the artifact takes Dynamo's four only.
    "a-fifth-key": (_dataset(url="https://example.invalid/fake_dataset.json"), ValueError, EXACT_KEYS),
    "a-key-that-is-not-a-str": ({**FAKE_KVWARM_DATASET, 1: "x"}, ValueError, EXACT_KEYS),
    "path-empty": (_dataset(path=""), ValueError, r"\['path'\] must not be empty$"),
    "path-none": (_dataset(path=None), TypeError, r"\['path'\] must be a str, got None$"),
    "path-bytes": (_dataset(path=b"/fake-cache/fake.json"), TypeError, r"\['path'\] must be a str, got b'"),
    "sha256-none": (_dataset(sha256=None), TypeError, r"\['sha256'\] must be a str, got None$"),
    "sha256-an-int": (_dataset(sha256=0x1234), TypeError, r"\['sha256'\] must be a str, got 4660$"),
    "sha256-bytes": (_dataset(sha256=b"0123456789abcdef" * 4), TypeError, r"\['sha256'\] must be a str, got b'"),
    "sha256-empty": (_dataset(sha256=""), ValueError, NOT_A_SHA256 + r"''$"),
    "sha256-one-digit-short": (_dataset(sha256="0123456789abcdef" * 3 + "0123456789abcde"), ValueError, NOT_A_SHA256),
    "sha256-one-digit-long": (_dataset(sha256="0123456789abcdef" * 4 + "0"), ValueError, NOT_A_SHA256),
    "sha256-upper-case": (_dataset(sha256="0123456789ABCDEF" * 4), ValueError, NOT_A_SHA256),
    "sha256-not-hexadecimal": (_dataset(sha256="g" * 64), ValueError, NOT_A_SHA256),
    "sha256-with-a-prefix": (_dataset(sha256="sha256:" + "0" * 57), ValueError, NOT_A_SHA256),
    "sha256-ends-with-a-newline": (_dataset(sha256="0" * 63 + "\n"), ValueError, NOT_A_SHA256),
    "collection-pool-empty": (_dataset(collection_pool=""), ValueError, r"\['collection_pool'\] must not be empty$"),
    "collection-pool-a-list": (
        _dataset(collection_pool=["conversation_sha256_even"]),
        TypeError,
        r"\['collection_pool'\] must be a str, got \['conversation_sha256_even'\]$",
    ),
    "conversations-zero": (_dataset(conversations=0), ValueError, r"\['conversations'\] must be >= 1, got 0$"),
    "conversations-negative": (_dataset(conversations=-1), ValueError, r"\['conversations'\] must be >= 1, got -1$"),
    "conversations-a-bool": (_dataset(conversations=True), TypeError, r"\['conversations'\] must be an int, got True$"),
    "conversations-a-float": (
        _dataset(conversations=1234.0),
        TypeError,
        r"\['conversations'\] must be an int, got 1234\.0$",
    ),
    "conversations-a-str": (
        _dataset(conversations="1234"),
        TypeError,
        r"\['conversations'\] must be an int, got '1234'$",
    ),
    "conversations-none": (_dataset(conversations=None), TypeError, r"\['conversations'\] must be an int, got None$"),
}


@pytest.mark.parametrize("mode", ["decode", "agg"])
@pytest.mark.parametrize("name", sorted(BAD_KVWARM_DATASETS))
def test_a_bad_dataset_record_is_refused_when_the_artifact_is_built(name, mode):
    record, error, message = BAD_KVWARM_DATASETS[name]
    kwargs = _build_kwargs(mode, _decode_points() if mode == "decode" else _agg_points())
    # The baseline is sound, with the record and without it.
    for sound in (None, dict(FAKE_KVWARM_DATASET)):
        assert artifact.build_rank_artifact(**kwargs, kvwarm_dataset=sound)["status"] == "complete"
    with pytest.raises(error, match=rf"^kvwarm_dataset{message}"):
        artifact.build_rank_artifact(**kwargs, kvwarm_dataset=record)


def test_a_prefill_artifact_refuses_a_dataset_record():
    points = _prefill_points()
    message = r"^kvwarm_dataset was given for a prefill artifact, which has no kvwarm block$"
    endings = (
        {},
        {"completed": 1, "stop_reason": "timeout"},
        {"completed": 1, "failure": {"point": None, "reason": "fake reason"}},
    )
    for ending in endings:
        completed = ending.get("completed")
        kwargs = {**_build_kwargs("prefill", points, completed=completed), **ending}
        kwargs.pop("completed", None)
        # A prefill artifact has no kvwarm block to put the record into, whatever the status of the run ...
        for record in (dict(FAKE_KVWARM_DATASET), {}, "fake_dataset.json"):
            with pytest.raises(ValueError, match=message):
                artifact.build_rank_artifact(**kwargs, kvwarm_dataset=record)
        # ... and it is built as before when none is given.
        assert "kvwarm" not in artifact.build_rank_artifact(**kwargs, kvwarm_dataset=None)
        assert artifact.build_rank_artifact(**kwargs, kvwarm_dataset=None) == artifact.build_rank_artifact(**kwargs)


def test_prefill_rows_carry_prefill_counters():
    (payload,) = _rank_artifacts("prefill", _prefill_points())
    assert payload["results"][0]["fpms"][0]["scheduled_requests"] == {
        "num_prefill_requests": 2,
        "sum_prefill_tokens": 256,
        "sum_prefill_kv_tokens": 128,
        "num_decode_requests": 0,
        "sum_decode_kv_tokens": 0,
    }
    assert [row["kv_seed_regime"] for row in payload["results"]] == EXPECTED_REGIMES["prefill"]


def test_measured_time_is_the_sum_of_group_times_and_may_equal_elapsed(tmp_path):
    points = _decode_points()
    kwargs = _build_kwargs("decode", points, dp=2)
    group_times = [max(_expected_wall_time(point, rank) for rank in (0, 1)) for point in points]
    total = 0.0
    for value in group_times:  # the validator adds the group times left to right, in id order
        total += value
    assert total == pytest.approx(math.fsum(group_times), rel=1e-12)

    payload = artifact.build_rank_artifact(**{**kwargs, "benchmark_elapsed_seconds": total})
    assert payload["timing"] == {"benchmark_elapsed_seconds": total, "measured_iteration_seconds": total}

    # One float step less elapsed time than was measured is a producer bug.
    with pytest.raises(ValueError, match="more than the elapsed"):
        artifact.build_rank_artifact(**{**kwargs, "benchmark_elapsed_seconds": math.nextafter(total, 0.0)})

    # The boundary case (elapsed exactly equal to the sum) is accepted by the validator as well.
    cell = _cell(phase="decode", dp=2, strategy="tp")
    pod = _pod(tmp_path, cell)
    payloads = [
        artifact.build_rank_artifact(
            **{**kwargs, "dp_rank": rank, "benchmark_elapsed_seconds": total}, producer=LAUNCHER_PRODUCER
        )
        for rank in (0, 1)
    ]
    artifact.write_artifacts(str(pod / "benchmark.json"), payloads)
    collection = _validate(cell, tmp_path)
    assert collection.rank_timings == ((0, total, total), (1, total, total))


def test_a_deadline_after_the_last_point_leaves_the_run_complete():
    (payload,) = _rank_artifacts("decode", _decode_points(), stop_reason="timeout")
    assert (payload["status"], payload["valid"], payload["stop_reason"], payload["errors"]) == (
        "complete",
        True,
        None,
        [],
    )


def test_elapsed_time_is_written_as_given(tmp_path):
    # A whole number of seconds is a legitimate elapsed time and is not converted; step times must be floats.
    cell = _cell(phase="prefill", strategy="single")
    pod = _pod(tmp_path, cell)
    (payload,) = _rank_artifacts("prefill", _prefill_points(), benchmark_elapsed_seconds=13)
    assert payload["timing"]["benchmark_elapsed_seconds"] == 13
    assert type(payload["timing"]["benchmark_elapsed_seconds"]) is int
    artifact.write_artifacts(str(pod / "benchmark.json"), [payload])
    assert _validate(cell, tmp_path).rank_timings[0][1] == 13.0


# ---------------------------------------------------------------------------------------------------------------
# Partial and failed artifacts: both checkers reject
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("completed", [0, 1, 2])
@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_partial_run_is_written_and_rejected_by_both_checkers(tmp_path, phase, dp, completed):
    cell = _cell(phase=phase, dp=dp, strategy="tp")
    pod = _pod(tmp_path, cell)
    points = _points(phase)

    payloads = _rank_artifacts(phase, points, dp=dp, completed=completed, stop_reason="timeout")
    for payload in payloads:
        assert (payload["status"], payload["valid"], payload["timing_valid"]) == ("partial", False, True)
        assert payload["usable"] is (completed >= 1)
        assert (payload["stop_reason"], payload["error"]) == ("timeout", None)
        assert payload["errors"] == [f"benchmark stopped by timeout after {completed} of {len(points)} points"]
        assert payload["coverage"] == {
            "expected_points": len(points),
            "completed_points": completed,
            "skipped_points": 0,
        }
        assert len(payload["results"]) == len(payload["iteration_groups"]) == completed
        assert (payload["skipped_points"], payload["missing_phases"]) == ([], [])
        assert set(payload) & BANNED_TOP_LEVEL_KEYS == set()
        # A group that is in the artifact was measured on every rank: it is complete although the run is not.
        assert all(group["complete"] is True for group in payload["iteration_groups"])
        # The kvwarm block says how the decode context was produced; it does not depend on the status.
        assert ("kvwarm" in payload) is (phase == "decode")
        if phase == "decode":
            assert payload["kvwarm"] == EXPECTED_KVWARM
    written = artifact.write_artifacts(str(pod / "benchmark.json"), payloads)

    # A partial run still gets its merged file; the gate fails on the rank file before reading it.
    merged = json.loads(Path(written["merged_output_path"]).read_text())
    assert (merged["artifact_type"], merged["status"], merged["valid"]) == ("merged", "partial", False)
    with pytest.raises(ValueError, match="invalid native terminal result envelope"):
        _validate(cell, tmp_path)
    gate = _gate(written["rank_files"], phase)
    assert gate.returncode == GATE_INVALID
    assert "status='partial' valid=False" in gate.stderr
    assert "stopped by timeout" in gate.stderr  # the gate prints "errors", which carries the reason


@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_failed_point_is_recorded_and_rejected_by_both_checkers(tmp_path, phase, dp):
    cell = _cell(phase=phase, dp=dp, strategy="tp")
    pod = _pod(tmp_path, cell)
    points = _points(phase)
    # A reason is free text to the module. This one is a reason string of Dynamo's vLLM self-benchmark
    # (instrumented_scheduler.py at the revision in the header of this file, _bench_fpm_validation_failure).
    failure = {"point": points[2], "reason": "measured_batch_size_mismatch"}

    payloads = _rank_artifacts(phase, points, dp=dp, completed=2, failure=failure)
    message = "benchmark_id=3: benchmark point failed: measured_batch_size_mismatch"
    for payload in payloads:
        assert (payload["status"], payload["valid"], payload["usable"]) == ("failed", False, False)
        assert (payload["error"], payload["errors"], payload["stop_reason"]) == (message, [message], None)
        assert payload["skipped_points"] == [
            {
                "point": points[2],
                "kv_seed_regime": EXPECTED_REGIMES[phase][2],
                "reason": "measured_batch_size_mismatch",
            }
        ]
        assert payload["coverage"] == {
            "expected_points": len(points),
            "completed_points": 2,
            "skipped_points": 1,
        }
        assert [row["point"]["benchmark_id"] for row in payload["results"]] == [1, 2]  # the evidence is kept
        assert set(payload) & BANNED_TOP_LEVEL_KEYS == set()
        # The run failed, the times of the two rows it kept did not: nothing is written with timing_valid false.
        assert payload["timing_valid"] is True
        assert all(group["complete"] is True for group in payload["iteration_groups"])
        assert ("kvwarm" in payload) is (phase == "decode")
        if phase == "decode":
            assert payload["kvwarm"] == EXPECTED_KVWARM
    written = artifact.write_artifacts(str(pod / "benchmark.json"), payloads)

    assert written["merged_output_path"] is None
    assert _result_files(pod) == sorted(["benchmark.json"] + [f"benchmark_dp{rank}.json" for rank in range(1, dp)])
    with pytest.raises(ValueError, match="invalid native terminal result envelope"):
        _validate(cell, tmp_path)
    gate = _gate(written["rank_files"], phase)
    assert gate.returncode == GATE_INVALID
    assert "status='failed' valid=False" in gate.stderr
    assert "measured_batch_size_mismatch" in gate.stderr


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_missing_phase_fails_the_run_and_is_rejected_by_both_checkers(tmp_path, phase):
    cell = _cell(phase=phase, strategy="tp")
    pod = _pod(tmp_path, cell)

    payload = artifact.build_rank_artifact(
        dp_rank=0,
        dp_size=1,
        mode=phase,
        run_id=FAKE_RUN_ID,
        grid_digest=FAKE_GRID_DIGEST,
        expected_points=0,
        measurements=[],
        benchmark_elapsed_seconds=0.25,
        missing_phases=(phase,),
    )
    message = f"no feasible benchmark points for phase(s): {phase}"
    assert (payload["status"], payload["valid"], payload["usable"]) == ("failed", False, False)
    assert (payload["error"], payload["errors"], payload["missing_phases"]) == (message, [message], [phase])
    assert payload["coverage"] == {"expected_points": 0, "completed_points": 0, "skipped_points": 0}
    assert (payload["timing_valid"], payload["stop_reason"]) == (True, None)
    assert ("kvwarm" in payload) is (phase == "decode")
    if phase == "decode":
        assert payload["kvwarm"] == EXPECTED_KVWARM
    written = artifact.write_artifacts(str(pod / "benchmark.json"), [payload])

    assert written["merged_output_path"] is None
    assert _result_files(pod) == ["benchmark.json"]
    with pytest.raises(ValueError, match="invalid native terminal result envelope"):
        _validate(cell, tmp_path)
    gate = _gate(written["rank_files"], phase)
    assert gate.returncode == GATE_INVALID
    assert "no feasible benchmark points" in gate.stderr


def test_failure_bookkeeping():
    points = _decode_points()
    base = _build_kwargs("decode", points, completed=1)

    # A failure outside any point: the reason is the error and nothing is listed as skipped.
    payload = artifact.build_rank_artifact(**base, failure={"point": None, "reason": "recorder header missing"})
    assert (payload["status"], payload["error"], payload["errors"]) == (
        "failed",
        "recorder header missing",
        ["recorder header missing"],
    )
    assert (payload["skipped_points"], payload["coverage"]["skipped_points"]) == ([], 0)
    # Nothing of such a run may be used, although the row measured before the failure stays in the artifact.
    assert (payload["valid"], payload["usable"]) == (False, False)
    assert payload["timing_valid"] is True  # the time of that row is sound
    assert [row["point"] for row in payload["results"]] == [points[0]]
    assert (payload["coverage"]["expected_points"], payload["coverage"]["completed_points"]) == (3, 1)

    # A failed point after the deadline keeps both facts. The reason is again a reason string of Dynamo's vLLM
    # self-benchmark (_bench_save_current_point).
    payload = artifact.build_rank_artifact(
        **base, failure={"point": points[1], "reason": "no_fpm_before_deadline"}, stop_reason="timeout"
    )
    assert (payload["status"], payload["stop_reason"]) == ("failed", "timeout")
    assert payload["error"] == "benchmark_id=2: benchmark point failed: no_fpm_before_deadline"
    # The deadline is in stop_reason only: "errors" holds the one error and nothing about the timeout.
    assert payload["errors"] == ["benchmark_id=2: benchmark point failed: no_fpm_before_deadline"]

    # In an agg run both phases can be missing, prefill first; a failure given with them decides the error.
    agg = {**base, "mode": "agg", "measurements": [], "expected_points": 0}
    payload = artifact.build_rank_artifact(**agg, missing_phases=["prefill", "decode"])
    assert payload["error"] == "no feasible benchmark points for phase(s): prefill, decode"
    assert payload["missing_phases"] == ["prefill", "decode"]
    payload = artifact.build_rank_artifact(
        **agg, missing_phases=("decode",), failure={"point": None, "reason": "engine exited"}
    )
    assert (payload["error"], payload["missing_phases"]) == ("engine exited", ["decode"])
    # A missing phase reported after the deadline keeps both facts, as a failed point does.
    payload = artifact.build_rank_artifact(**agg, missing_phases=("decode",), stop_reason="timeout")
    assert (payload["status"], payload["stop_reason"]) == ("failed", "timeout")
    assert payload["errors"] == ["no feasible benchmark points for phase(s): decode"]

    # The failed point is copied, so later changes by the caller do not reach the artifact. The copy is deep:
    # the list inside the point is not shared either.
    failed_point = copy.deepcopy(points[1])
    payload = artifact.build_rank_artifact(**base, failure={"point": failed_point, "reason": "fake reason"})
    failed_point["batch_size"] = 99
    failed_point["sample_reasons"].append("edited")
    assert payload["skipped_points"][0]["point"] == points[1]

    # The list of missing phases is copied as well.
    phases = ["prefill", "decode"]
    payload = artifact.build_rank_artifact(**agg, missing_phases=phases)
    phases.clear()
    assert payload["missing_phases"] == ["prefill", "decode"]


@pytest.mark.parametrize("dp", [1, 2])
def test_a_missing_phase_fails_a_run_that_measured_every_point_of_its_other_phase(tmp_path, dp):
    """An agg run whose grid has prefill points and no feasible decode point: all of its points are measured."""
    points = _prefill_points()
    payloads = _rank_artifacts("agg", points, dp=dp, missing_phases=("decode",))

    message = "no feasible benchmark points for phase(s): decode"
    for payload in payloads:
        assert (payload["status"], payload["valid"], payload["usable"]) == ("failed", False, False)
        assert (payload["error"], payload["errors"], payload["missing_phases"]) == (message, [message], ["decode"])
        assert payload["coverage"] == {"expected_points": 4, "completed_points": 4, "skipped_points": 0}
        # The measured rows stay in the artifact as evidence.
        assert [row["point"] for row in payload["results"]] == points
        assert len(payload["iteration_groups"]) == 4

    # It is written as a failed run (no merged file), and the gate rejects it with the reason.
    written = artifact.write_artifacts(str(tmp_path / "benchmark.json"), payloads)
    assert written["merged_output_path"] is None
    assert _result_files(tmp_path) == sorted(["benchmark.json"] + [f"benchmark_dp{rank}.json" for rank in range(1, dp)])
    gate = _gate(written["rank_files"], "agg")
    assert gate.returncode == GATE_INVALID
    assert "status='failed' valid=False" in gate.stderr
    assert "no feasible benchmark points" in gate.stderr


@pytest.mark.parametrize("elapsed", [0.0, 0])
def test_a_run_that_failed_before_any_time_passed_may_report_zero_elapsed_seconds(elapsed):
    payload = artifact.build_rank_artifact(
        dp_rank=0,
        dp_size=1,
        mode="decode",
        run_id=FAKE_RUN_ID,
        grid_digest=FAKE_GRID_DIGEST,
        expected_points=3,
        measurements=[],
        benchmark_elapsed_seconds=elapsed,
        failure={"point": None, "reason": "fake failure before the first point"},
    )
    assert (payload["status"], payload["error"]) == ("failed", "fake failure before the first point")
    assert payload["timing"] == {"benchmark_elapsed_seconds": 0, "measured_iteration_seconds": 0.0}
    assert type(payload["timing"]["benchmark_elapsed_seconds"]) is type(elapsed)


# ---------------------------------------------------------------------------------------------------------------
# Gate: pending states and write order
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_gate_reports_pending_until_the_merged_file_exists(tmp_path, phase):
    output_path = str(tmp_path / "benchmark.json")
    assert _gate([output_path], phase).returncode == GATE_RANK_PENDING

    written = artifact.write_artifacts(output_path, _rank_artifacts(phase, _points(phase)))
    merged_path = Path(written["merged_output_path"])
    merged_text = merged_path.read_text()
    merged_path.unlink()
    # Rank files valid, merged file missing: still finalizing.
    assert _gate(written["rank_files"], phase).returncode == GATE_MERGED_PENDING

    merged_path.write_text(merged_text)
    assert _gate(written["rank_files"], phase).returncode == GATE_ACCEPT


def test_rank_files_are_written_highest_rank_first_and_the_merged_file_last(tmp_path, monkeypatch):
    renamed = []
    real_replace = os.replace

    def recording_replace(source, destination):
        renamed.append(Path(destination).name)
        real_replace(source, destination)

    monkeypatch.setattr(artifact.os, "replace", recording_replace)
    artifact.write_artifacts(str(tmp_path / "benchmark.json"), _rank_artifacts("decode", _decode_points(), dp=3))
    assert renamed == ["benchmark_dp2.json", "benchmark_dp1.json", "benchmark.json", "benchmark_merged.json"]

    renamed.clear()
    failed = _rank_artifacts(
        "decode", _decode_points(), dp=2, completed=0, failure={"point": None, "reason": "fake failure"}
    )
    artifact.write_artifacts(str(tmp_path / "failed.json"), failed)
    assert renamed == ["failed_dp1.json", "failed.json"]

    # A partial run is the case the order exists for: the gate fails the pod when it reads rank 0's file.
    renamed.clear()
    partial = _rank_artifacts("decode", _decode_points(), dp=3, completed=1, stop_reason="timeout")
    artifact.write_artifacts(str(tmp_path / "partial.json"), partial)
    assert renamed == ["partial_dp2.json", "partial_dp1.json", "partial.json", "partial_merged.json"]


def test_gate_reacts_to_a_partial_run_only_once_rank_zero_is_on_disk(tmp_path):
    """Why rank 0 is written last: the gate stops at the first missing rank file, in rank order."""
    points = _decode_points()
    rank0, rank1 = _rank_artifacts("decode", points, dp=2, completed=1, stop_reason="timeout")
    paths = [artifact.rank_output_path(str(tmp_path / "benchmark.json"), rank) for rank in (0, 1)]

    artifact.atomic_write_json(paths[1], rank1)
    assert _gate(paths, "decode").returncode == GATE_RANK_PENDING  # only rank 1: the gate keeps waiting
    Path(paths[1]).unlink()
    artifact.atomic_write_json(paths[0], rank0)
    assert _gate(paths, "decode").returncode == GATE_INVALID  # only rank 0: the gate fails the pod at once


# ---------------------------------------------------------------------------------------------------------------
# Publication and the readiness reader
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("strategy", ["single", "tp", "dep"])
def test_decode_rows_are_published_as_real_kv(tmp_path, strategy, dp):
    cell = _cell(phase="decode", dp=dp, strategy=strategy)
    pod = _pod(tmp_path, cell)
    points = _decode_points()
    artifact.write_artifacts(str(pod / "benchmark.json"), _rank_artifacts("decode", points, dp=dp))

    rows = aggregate_cell(_sglang_plan(cell), cell, tmp_path, expected_attempt_id="attempt")

    assert [row["kv_seed_regime"] for row in rows] == ["real_kv"] * len(points)
    assert [(row["batch_size"], row["total_prefill_tokens"], row["total_kv_read_tokens"]) for row in rows] == [
        (2, 0, 2048),
        (2, 0, 8),
        (3, 0, 300),
    ]
    for point, row in zip(points, rows, strict=True):
        slowest = max(_expected_wall_time(point, rank) for rank in range(dp))
        assert row["latency_ms"] == slowest * 1000.0


@pytest.mark.parametrize("template_id", [None, GENERIC_RUNTIME_TEMPLATE_ID])
def test_prefill_rows_are_published_with_real_prefix_provenance(tmp_path, template_id):
    cell = _cell(phase="prefill", strategy="tp")
    pod = _pod(tmp_path, cell)
    points = _prefill_points()
    artifact.write_artifacts(str(pod / "benchmark.json"), _rank_artifacts("prefill", points))
    plan = _sglang_plan(cell)
    # The generic template refuses a cached prefill row whose native regime is not "real_prefix".
    plan.capability.template_id = template_id

    rows = aggregate_cell(plan, cell, tmp_path, expected_attempt_id="attempt")

    assert [row["kv_seed_regime"] for row in rows] == ["n/a"] * len(points)  # the formal prefill label
    selected = select_native_measurements(_validate(cell, tmp_path), cell_id=cell.cell_id)
    assert [measurement.kv_seed_regime for measurement in selected] == EXPECTED_REGIMES["prefill"]


@pytest.mark.parametrize("evidence_version", [1, 2])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_readiness_reader_accepts_an_artifact_without_measurement_protocol(tmp_path, phase, evidence_version):
    cell = _cell(phase=phase, dp=2, strategy="tp")
    pod = _pod(tmp_path, cell)
    payloads = _rank_artifacts(
        phase,
        _points(phase),
        dp=2,
        config={"content_seed": "0"},
        producer={**LAUNCHER_PRODUCER, "run_started_at": 111.0},
    )
    artifact.write_artifacts(str(pod / "benchmark.json"), payloads)
    collection = _validate(cell, tmp_path)

    evidence = extract_measurement_evidence(tmp_path / "raw", collection, evidence_version=evidence_version)

    assert evidence["measurement_protocol"] is None
    assert evidence["status"] == "unestablished"
    assert "legacy artifact has no measurement protocol" in evidence["reasons"]
    assert len(evidence["points"]) == len(_points(phase))
    # "config" has this reader: it is copied here and compared between launches, so it must stay run-invariant.
    assert evidence["benchmark_config"] == {
        "0": {"mode": phase, "content_seed": "0"},
        "1": {"mode": phase, "content_seed": "0"},
    }


@pytest.mark.parametrize("status", ["complete", "partial", "failed"])
@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_keys_with_vllm_shaped_readers_are_never_written(mode, status):
    points = _points(mode)
    extra = {
        "complete": {},
        "partial": {"completed": 1, "stop_reason": "timeout"},
        "failed": {"completed": 1, "failure": {"point": points[1], "reason": "fake reason"}},
    }[status]
    payloads = _rank_artifacts(mode, points, dp=2, **extra)
    documents = list(payloads)
    if status != "failed":
        documents.append(artifact.build_merged_artifact(payloads, ["/results/benchmark.json", "/results/b_dp1.json"]))

    for document in documents:
        assert document["status"] == status
        assert set(document) & BANNED_TOP_LEVEL_KEYS == set()
        for row in document["results"]:
            for fpm in row["fpms"]:
                assert "benchmark_measurement" not in fpm


# ---------------------------------------------------------------------------------------------------------------
# agg mode
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_agg_artifact_passes_the_gate_in_agg_mode_and_no_cell(tmp_path, phase):
    cell = _cell(phase=phase, strategy="tp")
    pod = _pod(tmp_path, cell)
    points = _prefill_points() + [{**point, "benchmark_id": point["benchmark_id"] + 4} for point in _decode_points()]

    written = artifact.write_artifacts(str(pod / "benchmark.json"), _rank_artifacts("agg", points))

    assert _gate(written["rank_files"], "agg").returncode == GATE_ACCEPT
    # AISim launches one phase per cell, so an agg file never matches a cell ...
    with pytest.raises(ValueError, match="native benchmark mode mismatch"):
        _validate(cell, tmp_path)
    # ... and a gate started for one phase refuses it as well.
    assert _gate(written["rank_files"], phase).returncode == GATE_INVALID


# ---------------------------------------------------------------------------------------------------------------
# Per-point pieces
# ---------------------------------------------------------------------------------------------------------------


def test_decode_wall_time_is_the_upper_median_of_steps_three_to_five():
    values = [0.011, 0.012, 0.013, 0.014, 0.015, 0.016]  # FAKE seconds, all distinct
    for steps in itertools.permutations(values):
        chosen = artifact.decode_wall_time(list(steps))
        kept = sorted([steps[2], steps[3], steps[4]])  # decode steps 3, 4 and 5
        assert chosen == kept[1]
        assert chosen == statistics.median_high(steps[2:5])

    steps = (0.5, 0.7, 0.031, 0.029, 0.030, 0.001)  # a tuple is accepted; steps 1, 2 and 6 never count
    chosen = artifact.decode_wall_time(steps)
    assert chosen == 0.030
    assert any(chosen is step for step in steps), "the sample must be one of the inputs, unchanged"
    # Two equal kept steps: the upper median of (a, a, b) with a < b is a.
    assert artifact.decode_wall_time([9.0, 9.0, 0.02, 0.02, 0.05, 9.0]) == 0.02


class _FloatSubclass(float):
    """Stands in for numpy.float64, which is a subclass of float. The module wants exactly float."""


@pytest.mark.parametrize(
    ("steps", "error"),
    [
        ([0.01] * 5, ValueError),
        ([0.01] * 7, ValueError),
        ([], ValueError),
        ([0.01, 0.01, 0.0, 0.01, 0.01, 0.01], ValueError),
        ([0.01, 0.01, 0.01, 0.01, 0.01, -0.01], ValueError),
        ([float("nan"), 0.01, 0.01, 0.01, 0.01, 0.01], ValueError),  # not a kept step, still refused
        ([0.01, 0.01, 0.01, float("inf"), 0.01, 0.01], ValueError),
        ([0.01, 0.01, 1, 0.01, 0.01, 0.01], TypeError),
        ([0.01, 0.01, True, 0.01, 0.01, 0.01], TypeError),
        ([0.01, 0.01, "0.01", 0.01, 0.01, 0.01], TypeError),
        ([0.01, 0.01, None, 0.01, 0.01, 0.01], TypeError),
        ([0.01, 0.01, _FloatSubclass(0.01), 0.01, 0.01, 0.01], TypeError),
        ("123456", TypeError),
        (None, TypeError),
        ({0: 0.01}, TypeError),
    ],
)
def test_decode_wall_time_rejects_bad_step_lists(steps, error):
    with pytest.raises(error):
        artifact.decode_wall_time(steps)


def test_scheduled_requests_and_point_rules_agree_with_the_validator():
    """Every small coordinate: the module accepts a point exactly when the validator's own rule does."""
    accepted = 0
    for phase, batch, total_new, total_kv in itertools.product(
        ("prefill", "decode"), range(4), range(-1, 5), range(-1, 5)
    ):
        point = _point(phase, 1, batch, total_new, total_kv)
        try:
            expected = native_artifact._expected_scheduled(point)
        except ValueError:
            with pytest.raises(ValueError):
                artifact.validate_point(point)
            with pytest.raises(ValueError):
                artifact.scheduled_requests(point)
        else:
            accepted += 1
            artifact.validate_point(point)
            assert artifact.scheduled_requests(point) == expected
    # Hand count over batch 1..3, new and cached tokens -1..4.
    # Prefill needs new >= batch and cached >= 0: (4 + 3 + 2) * 5 = 45.
    # Decode needs new == 0 and context >= batch: 4 + 3 + 2 = 9.
    assert accepted == 45 + 9


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (lambda point: point.pop("point_type"), ValueError),
        (lambda point: point.pop("benchmark_id"), ValueError),
        (lambda point: point.pop("total_prefill_tokens"), ValueError),
        (lambda point: point.pop("total_kv_read_tokens"), ValueError),
        (lambda point: point.pop("batch_size"), ValueError),
        (lambda point: point.pop("sample_reasons"), ValueError),
        (lambda point: point.update(point_type="agg"), ValueError),
        (lambda point: point.update(point_type=None), ValueError),
        (lambda point: point.update(benchmark_id=0), ValueError),
        (lambda point: point.update(benchmark_id=True), TypeError),
        (lambda point: point.update(benchmark_id="1"), TypeError),
        (lambda point: point.update(batch_size=0), ValueError),
        (lambda point: point.update(batch_size=True), TypeError),
        (lambda point: point.update(batch_size=2.0), TypeError),
        (lambda point: point.update(total_kv_read_tokens=64.0), TypeError),
        (lambda point: point.update(total_kv_read_tokens=None), TypeError),
        (lambda point: point.update(total_prefill_tokens=False), TypeError),
        (lambda point: point.update(sample_reasons=None), TypeError),
        (lambda point: point.update(sample_reasons=("kvwarm_real_kv",)), TypeError),
        (lambda point: point.update(sample_reasons="kvwarm_real_kv"), TypeError),
        (lambda point: point.update(sample_reasons=["kvwarm_real_kv", 7]), TypeError),
    ],
)
def test_validate_point_rejects_malformed_points(change, error):
    point = _point("decode", 1, 2, 0, 64, ["kvwarm_real_kv"])
    artifact.validate_point(point)
    steps = [0.011, 0.012, 0.013, 0.014, 0.015, 0.016]  # FAKE seconds; their sample is 0.014
    assert artifact.build_fpm(point, 0, 0.014, extra={"sglbench_decode_steps": steps})["counter_id"] == 1
    change(point)
    for function in (artifact.validate_point, artifact.scheduled_requests, artifact.kv_seed_regime):
        with pytest.raises(error):
            function(point)
    # build_fpm is called both ways, so that no refusal of the steps can stand in for the refusal of the point.
    # With consistent steps the point is the only reason to raise: a decode fpm without steps is refused for a
    # sound point too. Without steps, a point whose type was changed is not refused for carrying decode steps.
    with pytest.raises(error):
        artifact.build_fpm(point, 0, 0.014, extra={"sglbench_decode_steps": steps})
    with pytest.raises(error):
        artifact.build_fpm(point, 0, 0.01)
    with pytest.raises(error):
        artifact.build_measurement(point, {0: {}})


@pytest.mark.parametrize("point", [None, [], "decode", 3])
def test_validate_point_rejects_non_dicts(point):
    with pytest.raises(TypeError):
        artifact.validate_point(point)


def test_validate_point_accepts_extra_keys_and_boundary_totals():
    artifact.validate_point({**_point("prefill", 7, 3, 3, 0), "note": "one new token per request"})
    artifact.validate_point(_point("decode", 7, 3, 0, 3, ["kvwarm_real_kv"]))


def test_kv_seed_regime_agrees_with_the_validator_on_every_stamp_combination(tmp_path):
    """The module accepts a point exactly when the validator accepts its row with the module's regime."""
    stamps = ["prefill_real_seed", "prefill_fake_prefix", "kvwarm_real_kv", "kvwarm_fake_fallback"]
    kinds = {
        "real_kv": _point("decode", 1, 2, 0, 64),
        "real_prefix": _point("prefill", 1, 2, 16, 32),
        "not_applicable": _point("prefill", 1, 2, 16, 0),
    }
    accepted = []
    for (regime, base), size, other in itertools.product(kinds.items(), range(5), ([], ["explicit", "capture"])):
        for chosen in itertools.combinations(stamps, size):
            point = {**base, "sample_reasons": [*other, *chosen]}
            try:
                native_artifact._validate_seed_regime({"point": point, "kv_seed_regime": regime}, tmp_path)
            except ValueError:
                with pytest.raises(ValueError, match="stamps"):
                    artifact.kv_seed_regime(point)
            else:
                assert artifact.kv_seed_regime(point) == regime
                accepted.append((regime, tuple(point["sample_reasons"])))
    # Exactly one stamp set is right per kind, with or without the free reasons.
    assert sorted(accepted) == sorted(
        [
            ("real_kv", ("kvwarm_real_kv",)),
            ("real_kv", ("explicit", "capture", "kvwarm_real_kv")),
            ("real_prefix", ("prefill_real_seed",)),
            ("real_prefix", ("explicit", "capture", "prefill_real_seed")),
            ("not_applicable", ()),
            ("not_applicable", ("explicit", "capture")),
        ]
    )


def test_one_cached_token_in_total_is_a_cached_prefill_point(tmp_path):
    """The smallest cached prefix: one request with one cached token, which is legal at page size 1."""
    point = _point("prefill", 1, 1, 4, 1, ["prefill_real_seed"])
    assert artifact.kv_seed_regime(point) == "real_prefix"
    # The refusal starts with the point.
    message = r"^benchmark_id=1: regime 'real_prefix' needs the stamps \['prefill_real_seed'\]"
    with pytest.raises(ValueError, match=message):
        artifact.kv_seed_regime({**point, "sample_reasons": []})
    # One cached token fewer is the uncached regime, which takes no stamp.
    uncached = _point("prefill", 1, 1, 4, 0)
    assert artifact.kv_seed_regime(uncached) == "not_applicable"
    with pytest.raises(ValueError, match=r"^benchmark_id=1: regime 'not_applicable' needs the stamps \[\]"):
        artifact.kv_seed_regime({**uncached, "sample_reasons": ["prefill_real_seed"]})

    # The validator and the gate accept a run of that one point, and the validator reads the same regime.
    cell = _cell(phase="prefill", strategy="single")
    pod = _pod(tmp_path, cell)
    written = artifact.write_artifacts(str(pod / "benchmark.json"), _rank_artifacts("prefill", [point]))
    collection = _validate(cell, tmp_path)
    assert [(measurement.point, measurement.kv_seed_regime) for measurement in collection.points] == [
        (point, "real_prefix")
    ]
    assert _gate(written["rank_files"], "prefill").returncode == GATE_ACCEPT


def test_a_refused_stamp_set_names_the_point_the_regime_and_both_stamp_sets():
    """The whole text, at an id other than 1: the id in front is the only place the refusal names the point."""
    with pytest.raises(ValueError) as refused:
        artifact.kv_seed_regime(_point("decode", 7, 2, 0, 64, ["explicit", "prefill_real_seed"]))
    assert str(refused.value) == (
        "benchmark_id=7: regime 'real_kv' needs the stamps ['kvwarm_real_kv'] in sample_reasons, "
        "found ['prefill_real_seed']"
    )
    # The builders pass the same text on: a measurement, and the failed point of a run.
    point = _point("prefill", 2, 2, 16, 32)
    with pytest.raises(ValueError, match=r"^benchmark_id=2: regime 'real_prefix' needs the stamps"):
        artifact.build_measurement(point, {0: artifact.build_fpm(point, 0, 0.01)})
    kwargs = _build_kwargs("prefill", _prefill_points(), completed=1)
    with pytest.raises(ValueError, match=r"^benchmark_id=2: regime 'real_prefix' needs the stamps"):
        artifact.build_rank_artifact(**kwargs, failure={"point": point, "reason": "fake reason"})


def test_build_fpm_copies_extras_and_keeps_the_wall_time_object():
    point = _point("decode", 5, 2, 0, 64, ["kvwarm_real_kv"])
    steps = [0.011, 0.012, 0.013, 0.014, 0.015, 0.016]
    wall_time = artifact.decode_wall_time(steps)

    fpm = artifact.build_fpm(point, 1, wall_time, extra={"sglbench_decode_steps": steps, "worker_id": "fake-worker"})

    assert fpm["wall_time"] is wall_time
    assert fpm == {
        "counter_id": 5,
        "dp_rank": 1,
        "wall_time": 0.014,
        "scheduled_requests": {
            "num_prefill_requests": 0,
            "sum_prefill_tokens": 0,
            "sum_prefill_kv_tokens": 0,
            "num_decode_requests": 2,
            "sum_decode_kv_tokens": 64,
        },
        "sglbench_decode_steps": [0.011, 0.012, 0.013, 0.014, 0.015, 0.016],
        "worker_id": "fake-worker",
    }
    steps.append(99.0)
    assert len(fpm["sglbench_decode_steps"]) == 6
    # Without extras an fpm has the four core keys. Only a prefill fpm can be built that way: a decode fpm
    # needs its six raw steps.
    prefill = artifact.build_fpm(_point("prefill", 5, 2, 16, 0), 0, 0.01)
    assert set(prefill) == {"counter_id", "dp_rank", "wall_time", "scheduled_requests"}


@pytest.mark.parametrize(
    ("extra", "error"),
    [
        ({"counter_id": 1}, ValueError),
        ({"dp_rank": 0}, ValueError),
        ({"wall_time": 0.5}, ValueError),
        ({"scheduled_requests": {}}, ValueError),
        ({"benchmark_measurement": {"schema_version": 1}}, ValueError),
        ({"worker_id": "w", "benchmark_measurement": None}, ValueError),
        ({"benchmark_sample": {}}, ValueError),
        ({"worker_id": "w", "benchmark_sample": None}, ValueError),
        ({1: "not a str key"}, TypeError),
        ([("worker_id", "w")], TypeError),
        ("worker_id", TypeError),
    ],
)
def test_build_fpm_rejects_bad_extras(extra, error):
    # A prefill point: its fpm is complete without extras, so the extra is the only reason to raise.
    point = _point("prefill", 1, 2, 16, 0)
    assert artifact.build_fpm(point, 0, 0.01, extra={"worker_id": "w"})["worker_id"] == "w"
    with pytest.raises(error):
        artifact.build_fpm(point, 0, 0.01, extra=extra)


@pytest.mark.parametrize("key", ["benchmark_measurement", "benchmark_sample"])
def test_both_keys_that_measurement_evidence_reads_are_reserved(key):
    """measurement_evidence.py reads both keys of an fpm (``_read_measurement``, ``_observed_execution``)."""
    point = _point("prefill", 1, 2, 16, 0)
    with pytest.raises(ValueError, match=rf"extra may not contain the fpm keys \['{key}'\]"):
        artifact.build_fpm(point, 0, 0.01, extra={key: {}})
    good = artifact.build_fpm(point, 0, 0.01)
    with pytest.raises(ValueError, match=rf"fpm may not contain \['{key}'\]"):
        artifact.build_measurement(point, {0: {**good, key: {}}})
    measurement = artifact.build_measurement(point, {0: good})
    measurement["rank_fpms"][0][key] = {}
    with pytest.raises(ValueError, match=rf"fpm may not contain \['{key}'\]"):
        artifact.build_rank_artifact(**{**_build_kwargs("prefill", [point]), "measurements": [measurement]})


@pytest.mark.parametrize(
    ("wall_time", "error"),
    [
        (0.0, ValueError),
        (-0.001, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (1, TypeError),
        (True, TypeError),
        ("0.01", TypeError),
        (None, TypeError),
        (_FloatSubclass(0.01), TypeError),  # exactly float: a subclass is not converted, it is refused
    ],
)
def test_bad_wall_times_are_rejected_everywhere(wall_time, error):
    point = _point("prefill", 1, 2, 16, 0)
    with pytest.raises(error):
        artifact.build_fpm(point, 0, wall_time)
    # A hand-made fpm gets the same check when it is put into a measurement or an artifact.
    fpm = {**artifact.build_fpm(point, 0, 0.01), "wall_time": wall_time}
    with pytest.raises(error):
        artifact.build_measurement(point, {0: fpm})
    measurement = artifact.build_measurement(point, {0: artifact.build_fpm(point, 0, 0.01)})
    measurement["rank_fpms"][0]["wall_time"] = wall_time
    with pytest.raises(error):
        artifact.build_rank_artifact(**{**_build_kwargs("prefill", [point]), "measurements": [measurement]})


@pytest.mark.parametrize(
    ("dp_rank", "error"), [(-1, ValueError), (True, TypeError), (0.0, TypeError), ("0", TypeError), (None, TypeError)]
)
def test_build_fpm_rejects_bad_ranks(dp_rank, error):
    with pytest.raises(error):
        artifact.build_fpm(_point("prefill", 1, 2, 16, 0), dp_rank, 0.01)


def test_build_measurement_checks_every_fpm_and_copies_its_input():
    point = _point("prefill", 2, 2, 16, 32, ["prefill_real_seed"])
    fpms = {1: artifact.build_fpm(point, 1, 0.02), 0: artifact.build_fpm(point, 0, 0.01)}

    measurement = artifact.build_measurement(point, fpms)

    assert measurement == {"point": point, "kv_seed_regime": "real_prefix", "rank_fpms": fpms}
    assert list(measurement["rank_fpms"]) == [0, 1]
    point["batch_size"] = 99
    fpms[0]["wall_time"] = 5.0
    assert measurement["point"]["batch_size"] == 2
    assert measurement["rank_fpms"][0]["wall_time"] == 0.01
    # The copies are deep: the list inside the point and the dict inside an fpm are not shared either.
    point["sample_reasons"].append("edited")
    fpms[0]["scheduled_requests"]["num_prefill_requests"] = 99
    assert measurement["point"]["sample_reasons"] == ["prefill_real_seed"]
    assert measurement["rank_fpms"][0]["scheduled_requests"]["num_prefill_requests"] == 2

    point = _point("prefill", 2, 2, 16, 32, ["prefill_real_seed"])
    good = artifact.build_fpm(point, 0, 0.01)
    other_point = artifact.build_fpm(_point("prefill", 3, 2, 16, 32, ["prefill_real_seed"]), 0, 0.01)
    other_shape = artifact.build_fpm(_point("prefill", 2, 2, 17, 32, ["prefill_real_seed"]), 0, 0.01)
    for bad, error in [
        ({}, ValueError),  # no rank at all
        ([good], TypeError),
        ({0: other_point}, ValueError),  # counter_id is another point's
        ({1: good}, ValueError),  # the fpm says rank 0
        ({0: other_shape}, ValueError),  # scheduled_requests of another shape
        ({0: {**good, "benchmark_measurement": {}}}, ValueError),
        ({0: {**good, "benchmark_sample": {}}}, ValueError),
        ({0: {key: value for key, value in good.items() if key != "scheduled_requests"}}, ValueError),
        ({0: {**good, "counter_id": True}}, ValueError),
        ({0: None}, TypeError),
        ({0: [good]}, TypeError),  # a list and a str support "key in fpm", so they need their own check
        ({0: "fpm"}, TypeError),
        ({"0": good}, TypeError),
        ({True: good}, TypeError),
        ({-1: good}, ValueError),
    ]:
        with pytest.raises(error):
            artifact.build_measurement(point, bad)
    # A cached prefill point without its stamp has no regime.
    with pytest.raises(ValueError, match="stamps"):
        artifact.build_measurement(_point("prefill", 2, 2, 16, 32), {0: good})


def test_a_measurement_shares_no_nested_object_with_the_callers_point_and_fpms():
    """A driver that reuses its point, its step list or its evidence for the next point must not change this one."""
    point = _point("decode", 1, 2, 0, 64, ["explicit", "kvwarm_real_kv"])
    fpms = {
        rank: artifact.build_fpm(
            point, rank, 0.014, extra={STEPS_KEY: list(FAKE_STEPS), "fake_evidence": {"fake_list": [1]}}
        )
        for rank in (0, 1)
    }
    measurement = artifact.build_measurement(point, fpms)
    before = copy.deepcopy(measurement)

    point["sample_reasons"].append("edited")
    for fpm in fpms.values():
        fpm["scheduled_requests"]["num_decode_requests"] = 99
        fpm[STEPS_KEY].append(9.0)
        fpm["fake_evidence"]["fake_list"].append(2)

    assert measurement == before
    assert measurement["point"]["sample_reasons"] == ["explicit", "kvwarm_real_kv"]
    for rank in (0, 1):
        assert measurement["rank_fpms"][rank]["scheduled_requests"]["num_decode_requests"] == 2
        assert measurement["rank_fpms"][rank][STEPS_KEY] == FAKE_STEPS
        assert measurement["rank_fpms"][rank]["fake_evidence"] == {"fake_list": [1]}


def test_a_bool_counter_or_rank_is_refused_even_where_it_compares_equal_to_the_int():
    """``True == 1`` and ``False == 0``: on point 1 and rank 0 only the type check can refuse a bool."""
    point = _point("prefill", 1, 2, 16, 0)
    good = artifact.build_fpm(point, 0, 0.01)
    assert artifact.build_measurement(point, {0: good})["rank_fpms"][0] == good
    for key, value, message in [
        ("counter_id", True, "rank=0: fpm counter_id is True"),
        ("dp_rank", False, "rank=0: fpm dp_rank is False"),
    ]:
        assert good[key] == value  # equal as numbers, so a comparison alone would let it through
        with pytest.raises(ValueError, match=message):
            artifact.build_measurement(point, {0: {**good, key: value}})
        measurement = artifact.build_measurement(point, {0: good})
        measurement["rank_fpms"][0][key] = value
        with pytest.raises(ValueError, match=message):
            artifact.build_rank_artifact(**{**_build_kwargs("prefill", [point]), "measurements": [measurement]})


# ---------------------------------------------------------------------------------------------------------------
# A decode fpm carries its six raw step times
# ---------------------------------------------------------------------------------------------------------------

STEPS_KEY = "sglbench_decode_steps"
# FAKE seconds. Decode steps 3 to 5 are 0.013, 0.014 and 0.015, so the sample (their upper median) is 0.014.
FAKE_STEPS = [0.011, 0.012, 0.013, 0.014, 0.015, 0.016]
_ABSENT = object()

# Every refusal starts with the place: the point and the rank of the fpm, which is point 1 on rank 0 in the test
# that uses this table. A reason that comes from decode_wall_time follows the fpm key.
PLACE = r"^benchmark_id=1 rank=0: "
STEP_LIST = PLACE + r"fpm sglbench_decode_steps: "

# name -> (wall_time, what is stored under the steps key, exception, start of the message)
BAD_DECODE_FPMS = {
    "key-missing": (0.014, _ABSENT, ValueError, PLACE + r"a decode fpm must carry"),
    "five-values": (0.014, FAKE_STEPS[:5], ValueError, STEP_LIST + r"a decode run has exactly 6 decode steps, got 5"),
    "seven-values": (
        0.014,
        [*FAKE_STEPS, 0.017],
        ValueError,
        STEP_LIST + r"a decode run has exactly 6 decode steps, got 7",
    ),
    "no-values": (0.014, [], ValueError, STEP_LIST + r"a decode run has exactly 6 decode steps, got 0"),
    "an-int-step": (0.014, [1, *FAKE_STEPS[1:]], TypeError, STEP_LIST + r"raw_step_seconds\[0\] must be a float"),
    "a-bool-step": (0.014, [*FAKE_STEPS[:5], True], TypeError, STEP_LIST + r"raw_step_seconds\[5\] must be a float"),
    "a-string-step": (
        0.014,
        ["0.011", *FAKE_STEPS[1:]],
        TypeError,
        STEP_LIST + r"raw_step_seconds\[0\] must be a float",
    ),
    "a-none-step": (0.014, [None, *FAKE_STEPS[1:]], TypeError, STEP_LIST + r"raw_step_seconds\[0\] must be a float"),
    "a-float-subclass-step": (
        0.014,
        [_FloatSubclass(0.011), *FAKE_STEPS[1:]],
        TypeError,
        STEP_LIST + r"raw_step_seconds\[0\] must be a float",
    ),
    "a-zero-step": (
        0.014,
        [0.0, *FAKE_STEPS[1:]],
        ValueError,
        STEP_LIST + r"raw_step_seconds\[0\] must be finite and > 0",
    ),
    "a-negative-step": (
        0.014,
        [*FAKE_STEPS[:5], -0.016],
        ValueError,
        STEP_LIST + r"raw_step_seconds\[5\] must be finite",
    ),
    "a-nan-step": (
        0.014,
        [float("nan"), *FAKE_STEPS[1:]],
        ValueError,
        STEP_LIST + r"raw_step_seconds\[0\] must be finite",
    ),
    "an-infinite-step": (
        0.014,
        [*FAKE_STEPS[:5], float("inf")],
        ValueError,
        STEP_LIST + r"raw_step_seconds\[5\] must be finite",
    ),
    "steps-none": (0.014, None, TypeError, PLACE + r"fpm sglbench_decode_steps must be a list"),
    "steps-a-tuple": (0.014, tuple(FAKE_STEPS), TypeError, PLACE + r"fpm sglbench_decode_steps must be a list"),
    "steps-a-dict": (
        0.014,
        dict(enumerate(FAKE_STEPS)),
        TypeError,
        PLACE + r"fpm sglbench_decode_steps must be a list",
    ),
    "wall-time-is-the-lowest-kept-step": (
        0.013,
        FAKE_STEPS,
        ValueError,
        PLACE + r"fpm wall_time 0.013 is not the sample 0.014",
    ),
    "wall-time-is-the-highest-kept-step": (
        0.015,
        FAKE_STEPS,
        ValueError,
        PLACE + r"fpm wall_time 0.015 is not the sample 0.014",
    ),
    "wall-time-is-the-first-step": (
        0.011,
        FAKE_STEPS,
        ValueError,
        PLACE + r"fpm wall_time 0.011 is not the sample 0.014",
    ),
    "wall-time-is-the-last-step": (
        0.016,
        FAKE_STEPS,
        ValueError,
        PLACE + r"fpm wall_time 0.016 is not the sample 0.014",
    ),
    "wall-time-is-no-step": (
        0.0141,
        FAKE_STEPS,
        ValueError,
        PLACE + r"fpm wall_time 0.0141 is not the sample 0.014",
    ),
    # The comparison is exact: the next float above the sample is refused.
    "wall-time-is-one-float-above-the-sample": (
        math.nextafter(0.014, 1.0),
        FAKE_STEPS,
        ValueError,
        PLACE + r"fpm wall_time 0\.014000000000000002 is not the sample 0\.014 of its raw steps",
    ),
    # ... and so is the next float below it.
    "wall-time-is-one-float-below-the-sample": (
        math.nextafter(0.014, 0.0),
        FAKE_STEPS,
        ValueError,
        PLACE + r"fpm wall_time 0\.013999999999999999 is not the sample 0\.014 of its raw steps",
    ),
}


@pytest.mark.parametrize("name", sorted(BAD_DECODE_FPMS))
def test_a_decode_fpm_without_consistent_raw_steps_is_refused_by_every_builder(name):
    wall_time, stored, error, message = BAD_DECODE_FPMS[name]
    point = _point("decode", 1, 2, 0, 64, ["kvwarm_real_kv"])
    good = artifact.build_fpm(point, 0, 0.014, extra={STEPS_KEY: list(FAKE_STEPS), "worker_id": "fake-worker"})
    assert (good["wall_time"], good[STEPS_KEY]) == (0.014, FAKE_STEPS)  # the sound baseline

    extra = {"worker_id": "fake-worker"} if stored is _ABSENT else {"worker_id": "fake-worker", STEPS_KEY: stored}
    with pytest.raises(error, match=message):
        artifact.build_fpm(point, 0, wall_time, extra=extra)
    if stored is _ABSENT:
        with pytest.raises(error, match=message):
            artifact.build_fpm(point, 0, wall_time)

    # A hand-made fpm gets the same check when it is put into a measurement or an artifact.
    bad = {**good, "wall_time": wall_time, STEPS_KEY: stored}
    if stored is _ABSENT:
        del bad[STEPS_KEY]
    with pytest.raises(error, match=message):
        artifact.build_measurement(point, {0: bad})
    measurement = artifact.build_measurement(point, {0: good})
    measurement["rank_fpms"][0] = bad
    kwargs = {**_build_kwargs("decode", [point]), "measurements": [measurement]}
    with pytest.raises(error, match=message):
        artifact.build_rank_artifact(**kwargs)
    measurement["rank_fpms"][0] = good
    assert artifact.build_rank_artifact(**kwargs)["status"] == "complete"


@pytest.mark.parametrize(
    ("steps", "error", "reason"),
    [
        (FAKE_STEPS[:5], ValueError, "a decode run has exactly 6 decode steps, got 5"),
        ([*FAKE_STEPS[:5], 0.0], ValueError, "raw_step_seconds[5] must be finite and > 0, got 0.0"),
        ([*FAKE_STEPS[:5], 1], TypeError, "raw_step_seconds[5] must be a float in seconds, got 1"),
    ],
    ids=["five-values", "a-zero-step", "an-int-step"],
)
def test_a_refused_step_list_names_the_point_the_rank_and_the_fpm_key(steps, error, reason):
    """The step reduction knows neither the point nor the rank; the fpm check puts both in front of its reason."""
    with pytest.raises(error) as bare:
        artifact.decode_wall_time(steps)
    assert str(bare.value) == reason  # the reduction alone names no place

    # The fpm of point 2 on rank 1, alone and inside a run of 3 points on 2 ranks.
    with pytest.raises(error) as at_the_fpm:
        artifact.build_fpm(_decode_points()[1], 1, 0.014, extra={STEPS_KEY: steps})
    kwargs = _decode_run()
    kwargs["measurements"][1]["rank_fpms"][1][STEPS_KEY] = steps
    with pytest.raises(error) as at_the_artifact:
        artifact.build_rank_artifact(**kwargs)
    for refused in (at_the_fpm, at_the_artifact):
        assert type(refused.value) is error
        assert str(refused.value) == f"benchmark_id=2 rank=1: fpm sglbench_decode_steps: {reason}"
        # The reduction's own error is kept as the cause.
        assert type(refused.value.__cause__) is error
        assert str(refused.value.__cause__) == reason

    # The far end of a run of 3 points on 4 ranks: the fpm of point 3 on rank 3, alone, in a measurement and
    # inside the run.
    point = _decode_points()[2]
    with pytest.raises(error) as at_the_fpm:
        artifact.build_fpm(point, 3, 0.014, extra={STEPS_KEY: steps})
    fpms = _measurement(point, 4)["rank_fpms"]
    fpms[3][STEPS_KEY] = steps
    with pytest.raises(error) as at_the_measurement:
        artifact.build_measurement(point, fpms)
    kwargs = _decode_run(dp=4)
    kwargs["measurements"][2]["rank_fpms"][3][STEPS_KEY] = steps
    with pytest.raises(error) as at_the_artifact:
        artifact.build_rank_artifact(**kwargs)
    for refused in (at_the_fpm, at_the_measurement, at_the_artifact):
        assert type(refused.value) is error
        assert str(refused.value) == f"benchmark_id=3 rank=3: fpm sglbench_decode_steps: {reason}"


@pytest.mark.parametrize("stored", [FAKE_STEPS, [], None, 0], ids=["six-steps", "no-steps", "none", "zero"])
def test_a_prefill_fpm_may_not_carry_decode_steps(stored):
    """The key itself is refused, whatever is stored under it: an empty or a false value is no exception."""
    stored = copy.deepcopy(stored)
    point = _point("prefill", 1, 2, 16, 0)
    message = r"^benchmark_id=1 rank=0: a prefill fpm may not contain 'sglbench_decode_steps'$"
    with pytest.raises(ValueError, match=message):
        artifact.build_fpm(point, 0, 0.014, extra={STEPS_KEY: stored})
    good = artifact.build_fpm(point, 0, 0.014)
    bad = {**good, STEPS_KEY: stored}
    with pytest.raises(ValueError, match=message):
        artifact.build_measurement(point, {0: bad})
    measurement = artifact.build_measurement(point, {0: good})
    measurement["rank_fpms"][0] = bad
    with pytest.raises(ValueError, match=message):
        artifact.build_rank_artifact(**{**_build_kwargs("prefill", [point]), "measurements": [measurement]})


def test_every_decode_fpm_of_an_artifact_holds_the_steps_its_sample_came_from():
    """What the deviation note of the kept decode steps promises: the raw steps are in the file, next to the sample."""
    for payload in _rank_artifacts("decode", _decode_points(), dp=2):
        fpms = [row["fpms"][0] for row in payload["results"]]
        fpms += [item["fpms"][0] for group in payload["iteration_groups"] for item in group["rank_results"]]
        assert len(fpms) == 3 + 3 * 2
        for fpm in fpms:
            steps = fpm[STEPS_KEY]
            assert len(steps) == artifact.DECODE_RAW_STEPS
            assert fpm["wall_time"] == statistics.median_high(steps[2:5])
    for payload in _rank_artifacts("prefill", _prefill_points(), dp=2):
        assert all(STEPS_KEY not in row["fpms"][0] for row in payload["results"])


# ---------------------------------------------------------------------------------------------------------------
# Evidence must be JSON data when it is handed over
# ---------------------------------------------------------------------------------------------------------------


class _FakeEnum(enum.Enum):
    WARM = 1


# name -> (a value json.dumps(allow_nan=False) refuses, exception)
NOT_JSON = {
    "nan": (float("nan"), ValueError),
    "infinity": (float("inf"), ValueError),
    "minus-infinity": (float("-inf"), ValueError),
    "bytes": (b"\x00fake-digest", TypeError),
    "path": (Path("/models/fake"), TypeError),
    "set": ({"seed", "warm"}, TypeError),
    "enum": (_FakeEnum.WARM, TypeError),
    "tuple-key": ({(1, 2): 3}, TypeError),
}


@pytest.mark.parametrize("nested", [False, True], ids=["top-level", "nested"])
@pytest.mark.parametrize("name", sorted(NOT_JSON))
def test_evidence_that_is_not_json_is_refused_by_the_call_that_receives_it(name, nested):
    """A bad evidence value must stop the point or the run where it is produced, not after the sweep."""
    value, error = NOT_JSON[name]
    evidence = {"fake_evidence": {"deep": [1, value]} if nested else value}

    prefill = _point("prefill", 1, 2, 16, 0)
    with pytest.raises(error, match="extra is not JSON data"):
        artifact.build_fpm(prefill, 0, 0.01, extra=evidence)
    decode = _point("decode", 1, 2, 0, 64, ["kvwarm_real_kv"])
    with pytest.raises(error, match="extra is not JSON data"):
        artifact.build_fpm(decode, 0, 0.014, extra={STEPS_KEY: list(FAKE_STEPS), **evidence})

    kwargs = _build_kwargs("decode", _decode_points(), dp=2)
    sound = artifact.build_rank_artifact(**kwargs, config={"fake_evidence": [1]}, producer={"fake_evidence": [1]})
    assert sound["status"] == "complete"
    with pytest.raises(error, match="config is not JSON data"):
        artifact.build_rank_artifact(**kwargs, config=evidence)
    with pytest.raises(error, match="producer is not JSON data"):
        artifact.build_rank_artifact(**kwargs, producer=evidence)
    # A failed run is refused the same way: the bad value has to be fixed, it cannot be written.
    failed = {**kwargs, "measurements": [], "failure": {"point": None, "reason": "fake failure"}}
    sound = artifact.build_rank_artifact(**failed, config={"fake_evidence": [1]}, producer={"fake_evidence": [1]})
    assert sound["status"] == "failed"
    with pytest.raises(error, match="config is not JSON data"):
        artifact.build_rank_artifact(**failed, config=evidence)
    with pytest.raises(error, match="producer is not JSON data"):
        artifact.build_rank_artifact(**failed, producer=evidence)


class _StrSubclass(str):
    """A dict key of this type is a str for the key rule, which asks for a str and not for exactly str."""


# A dict key that is not a str. json writes the first four as the texts "1", "1.5", "true" and "null", and
# refuses the last two itself with a text that names the type only.
NOT_STR_KEYS = {"int": 1, "float": 1.5, "bool": True, "none": None, "tuple": (1, 2), "bytes": b"fake-key"}


def _evidence_with_a_nested_key(key, shape: str) -> dict:
    inner = {"fake_sound_key": 1, key: "x"}
    return {
        "dict-in-dict": {"fake_evidence": inner},
        "dict-in-list": {"fake_evidence": {"deep": [1, [inner]]}},
        "dict-in-tuple": {"fake_evidence": ({"deeper": inner},)},
    }[shape]


@pytest.mark.parametrize("shape", ["dict-in-dict", "dict-in-list", "dict-in-tuple"])
@pytest.mark.parametrize("kind", sorted(NOT_STR_KEYS))
def test_a_dict_key_that_is_not_a_str_is_refused_at_every_depth_of_the_evidence(kind, shape):
    """Below the top level too: the refusal is a TypeError that names the argument and the key."""
    key = NOT_STR_KEYS[kind]
    evidence = _evidence_with_a_nested_key(key, shape)
    reason = f"is not JSON data: dict keys must be str at every depth, got {key!r}"
    # The same shape with str keys only is accepted, so the key is the only reason to raise.
    sound = _evidence_with_a_nested_key(_StrSubclass("fake_key"), shape)

    prefill = _point("prefill", 1, 2, 16, 0)
    decode = _point("decode", 1, 2, 0, 64, ["kvwarm_real_kv"])
    assert artifact.build_fpm(prefill, 0, 0.01, extra=sound)["fake_evidence"] == sound["fake_evidence"]
    with pytest.raises(TypeError) as refused:
        artifact.build_fpm(prefill, 0, 0.01, extra=evidence)
    assert str(refused.value) == f"extra {reason}"
    with pytest.raises(TypeError) as refused:
        artifact.build_fpm(decode, 0, 0.014, extra={STEPS_KEY: list(FAKE_STEPS), **evidence})
    assert str(refused.value) == f"extra {reason}"

    complete = _build_kwargs("decode", _decode_points(), dp=2)
    failed = {**complete, "measurements": [], "failure": {"point": None, "reason": "fake failure"}}
    for kwargs, status in ((complete, "complete"), (failed, "failed")):
        assert artifact.build_rank_artifact(**kwargs, config=sound, producer=sound)["status"] == status
        for name in ("config", "producer"):
            with pytest.raises(TypeError) as refused:
                artifact.build_rank_artifact(**kwargs, **{name: evidence})
            assert str(refused.value) == f"{name} {reason}"


def test_two_keys_that_json_would_write_as_the_same_text_are_refused():
    """Why a key must be a str: json turns ``1`` into ``"1"``, beside a ``"1"`` that is already there."""
    twice = {1: "int", "1": "str"}
    text = json.dumps({"fake_evidence": twice})
    assert text == '{"fake_evidence": {"1": "int", "1": "str"}}'  # one JSON object, the key "1" twice
    assert json.loads(text) == {"fake_evidence": {"1": "str"}}  # a reader keeps one of the two values

    reason = "is not JSON data: dict keys must be str at every depth, got 1"
    with pytest.raises(TypeError) as refused:
        artifact.build_fpm(_point("prefill", 1, 2, 16, 0), 0, 0.01, extra={"fake_evidence": twice})
    assert str(refused.value) == f"extra {reason}"
    kwargs = _build_kwargs("decode", _decode_points())
    for name in ("config", "producer"):
        with pytest.raises(TypeError) as refused:
            artifact.build_rank_artifact(**kwargs, **{name: {"fake_evidence": twice}})
        assert str(refused.value) == f"{name} {reason}"


def test_a_tuple_in_the_evidence_is_accepted_and_written_as_a_list(tmp_path):
    point = _point("prefill", 1, 2, 16, 0)
    fpm = artifact.build_fpm(point, 0, 0.01, extra={"fake_shape": (1, (2, 3)), "fake_caps": {"a": ("b",)}})
    assert fpm["fake_shape"] == (1, (2, 3))  # in memory it stays what the caller gave
    kwargs = {**_build_kwargs("prefill", [point]), "measurements": [artifact.build_measurement(point, {0: fpm})]}
    payload = artifact.build_rank_artifact(
        **kwargs, config={"fake_sizes": (1, 2)}, producer={"fake_notes": ("a", {"b": (3,)})}
    )
    assert (payload["config"]["fake_sizes"], payload["producer"]["fake_notes"]) == ((1, 2), ("a", {"b": (3,)}))

    written = artifact.write_artifacts(str(tmp_path / "benchmark.json"), [payload])
    for path in (*written["rank_files"], written["merged_output_path"]):
        on_disk = json.loads(Path(path).read_text())
        assert on_disk["config"]["fake_sizes"] == [1, 2]
        assert on_disk["producer"]["fake_notes"] == ["a", {"b": [3]}]
        assert on_disk["results"][0]["fpms"][0]["fake_shape"] == [1, [2, 3]]
        assert on_disk["results"][0]["fpms"][0]["fake_caps"] == {"a": ["b"]}


@pytest.mark.parametrize("container", ["dict", "list"])
def test_evidence_that_holds_itself_is_refused_as_not_json(container):
    """A reference cycle has no JSON form; the key check must not follow it forever."""
    cycle = {"fake_self": None} if container == "dict" else [1]
    if container == "dict":
        cycle["fake_self"] = cycle
    else:
        cycle.append(cycle)
    kwargs = _build_kwargs("decode", _decode_points())
    for name in ("config", "producer"):
        with pytest.raises(ValueError, match=rf"^{name} is not JSON data: "):
            artifact.build_rank_artifact(**kwargs, **{name: {"fake_evidence": cycle}})


# ---------------------------------------------------------------------------------------------------------------
# Producer bugs raise
# ---------------------------------------------------------------------------------------------------------------


def _decode_run(dp: int = 2) -> dict:
    return _build_kwargs("decode", _decode_points(), dp=dp)


def _with(**changes):
    return lambda kwargs: kwargs.update(changes)


def _drop_measurement(index: int):
    return lambda kwargs: kwargs["measurements"].pop(index)


def _edit_measurement(index: int, edit):
    return lambda kwargs: edit(kwargs["measurements"][index])


def _duplicate_coordinate(kwargs: dict) -> None:
    # Same batch and context as point 1 under a new id: AISim raises on it at publication.
    clone = {**_decode_points()[0], "benchmark_id": 3}
    kwargs["measurements"][2] = _measurement(clone, 2)


def _prefill_point_in_a_decode_run(kwargs: dict) -> None:
    kwargs["measurements"][2] = _measurement(_point("prefill", 3, 1, 8, 0), 2)


# name -> (change applied to a sound decode run of 3 points on 2 ranks, exception, fragment of the message)
PRODUCER_BUGS = {
    "ids-start-at-2": (_drop_measurement(0), ValueError, r"benchmark_id is 2, expected 1"),
    "id-gap": (_drop_measurement(1), ValueError, r"benchmark_id is 3, expected 2"),
    "ids-out-of-order": (lambda kwargs: kwargs["measurements"].reverse(), ValueError, r"benchmark_id is 3, expected 1"),
    "repeated-id": (
        lambda kwargs: kwargs["measurements"].__setitem__(1, kwargs["measurements"][0]),
        ValueError,
        r"benchmark_id is 1, expected 2",
    ),
    "missing-rank": (_edit_measurement(1, lambda m: m["rank_fpms"].pop(1)), ValueError, r"rank_fpms has ranks \[0\]"),
    "extra-rank": (
        _edit_measurement(1, lambda m: m["rank_fpms"].update({2: {**m["rank_fpms"][1], "dp_rank": 2}})),
        ValueError,
        r"rank_fpms has ranks \[0, 1, 2\]",
    ),
    "string-rank-key": (
        # What a JSON round trip of a measurement would produce; it is refused, not converted back.
        _edit_measurement(1, lambda m: m.update(rank_fpms={str(k): v for k, v in m["rank_fpms"].items()})),
        TypeError,
        r"rank_fpms key must be an int, got '0'",
    ),
    "rank-fpms-not-a-dict": (
        _edit_measurement(1, lambda m: m.update(rank_fpms=list(m["rank_fpms"].values()))),
        TypeError,
        r"rank_fpms must be a dict",
    ),
    "fpm-of-another-point": (
        _edit_measurement(1, lambda m: m["rank_fpms"][0].update(counter_id=7)),
        ValueError,
        r"fpm counter_id is 7",
    ),
    "fpm-of-another-rank": (
        _edit_measurement(1, lambda m: m["rank_fpms"][0].update(dp_rank=1)),
        ValueError,
        r"rank=0: fpm dp_rank is 1",
    ),
    "fpm-of-another-shape": (
        _edit_measurement(1, lambda m: m["rank_fpms"][0]["scheduled_requests"].update(num_decode_requests=9)),
        ValueError,
        r"scheduled_requests do not match the point",
    ),
    "fpm-with-vllm-evidence-key": (
        _edit_measurement(1, lambda m: m["rank_fpms"][0].update(benchmark_measurement={})),
        ValueError,
        r"fpm may not contain \['benchmark_measurement'\]",
    ),
    "regime-differs-from-point": (
        _edit_measurement(1, lambda m: m.update(kv_seed_regime="real_prefix")),
        ValueError,
        r"kv_seed_regime is 'real_prefix', the point says 'real_kv'",
    ),
    "stamp-removed-after-build": (
        _edit_measurement(1, lambda m: m["point"].update(sample_reasons=[])),
        ValueError,
        r"^benchmark_id=2: regime 'real_kv' needs the stamps \['kvwarm_real_kv'\]",
    ),
    "measurement-without-point": (_edit_measurement(1, lambda m: m.pop("point")), ValueError, r"lacks \['point'\]"),
    "measurement-not-a-dict": (
        lambda kwargs: kwargs["measurements"].__setitem__(1, None),
        TypeError,
        r"measurements\[1\] must be a dict",
    ),
    "measurements-not-a-list": (_with(measurements={"1": None}), TypeError, r"measurements must be a list or tuple"),
    "prefill-point-in-decode-run": (
        _prefill_point_in_a_decode_run,
        ValueError,
        r"a 'prefill' point is not valid in mode 'decode'",
    ),
    "duplicate-coordinate": (_duplicate_coordinate, ValueError, r"second measurement of coordinate"),
    "more-measurements-than-points": (_with(expected_points=2), ValueError, r"3 measurements for a grid of 2 points"),
    "partial-without-stop-reason": (_drop_measurement(2), ValueError, r"only 2 of 3 points completed"),
    "no-points-at-all": (_with(measurements=[], expected_points=0), ValueError, r"cannot be complete"),
    "sum-above-elapsed": (_with(benchmark_elapsed_seconds=0.001), ValueError, r"more than the elapsed"),
    "elapsed-negative": (_with(benchmark_elapsed_seconds=-1.0), ValueError, r"must be finite and >= 0"),
    "elapsed-nan": (_with(benchmark_elapsed_seconds=float("nan")), ValueError, r"must be finite and >= 0"),
    "elapsed-infinite": (_with(benchmark_elapsed_seconds=float("inf")), ValueError, r"must be finite and >= 0"),
    "elapsed-bool": (_with(benchmark_elapsed_seconds=True), TypeError, r"benchmark_elapsed_seconds must be a number"),
    "elapsed-string": (
        _with(benchmark_elapsed_seconds="12.5"),
        TypeError,
        r"benchmark_elapsed_seconds must be a number",
    ),
    "config-overrides-mode": (_with(config={"mode": "prefill"}), ValueError, r"config may not contain 'mode'"),
    "config-repeats-mode": (_with(config={"mode": "decode"}), ValueError, r"config may not contain 'mode'"),
    "config-not-a-dict": (_with(config=[("seed", "0")]), TypeError, r"config must be a dict"),
    # A key that is not a str, at the top level of each block: json would write it as the text "1".
    "config-key-not-a-str": (_with(config={1: "x"}), TypeError, r"^config keys must be str, got 1$"),
    "producer-key-not-a-str": (_with(producer={1: "x"}), TypeError, r"^producer keys must be str, got 1$"),
    "producer-overrides-name": (_with(producer={"name": "someone-else"}), ValueError, r"may not contain \['name'\]"),
    "producer-overrides-format": (
        _with(producer={"artifact_format": 2}),
        ValueError,
        r"may not contain \['artifact_format'\]",
    ),
    "producer-not-a-dict": (_with(producer="sglang"), TypeError, r"producer must be a dict"),
    "rank-equal-to-size": (_with(dp_rank=2), ValueError, r"dp_rank 2 is outside range\(2\)"),
    "rank-negative": (_with(dp_rank=-1), ValueError, r"dp_rank must be >= 0"),
    "rank-bool": (_with(dp_rank=False), TypeError, r"dp_rank must be an int"),
    "size-zero": (_with(dp_size=0), ValueError, r"dp_size must be >= 1"),
    "size-float": (_with(dp_size=2.0), TypeError, r"dp_size must be an int"),
    "unknown-mode": (_with(mode="bench"), ValueError, r"mode must be one of"),
    "empty-run-id": (_with(run_id=""), ValueError, r"run_id must not be empty"),
    "run-id-not-a-string": (_with(run_id=17), TypeError, r"run_id must be a str"),
    "empty-grid-digest": (_with(grid_digest=""), ValueError, r"grid_digest must not be empty"),
    "grid-digest-none": (_with(grid_digest=None), TypeError, r"grid_digest must be a str"),
    "unknown-stop-reason": (_with(stop_reason="banana"), ValueError, r"stop_reason must be None or 'timeout'"),
    "expected-points-bool": (_with(expected_points=True), TypeError, r"expected_points must be an int"),
    "expected-points-float": (_with(expected_points=3.0), TypeError, r"expected_points must be an int"),
    "expected-points-negative": (_with(expected_points=-1), ValueError, r"expected_points must be >= 0"),
    "missing-phases-bare-string": (
        _with(missing_phases="decode"),
        TypeError,
        r"missing_phases must be a list or tuple",
    ),
    "missing-phase-of-another-mode": (
        _with(missing_phases=("prefill",)),
        ValueError,
        r"'prefill' is not a phase of mode 'decode'",
    ),
    "missing-phase-repeated": (_with(missing_phases=("decode", "decode")), ValueError, r"unique and ordered"),
    "missing-phases-out-of-order": (
        _with(mode="agg", missing_phases=("decode", "prefill")),
        ValueError,
        r"unique and ordered",
    ),
    "failure-not-a-dict": (_with(failure="boom"), TypeError, r"failure must be a dict"),
    "failure-without-point-key": (_with(failure={"reason": "boom"}), ValueError, r"exactly the keys"),
    "failure-with-extra-key": (
        _with(failure={"point": None, "reason": "boom", "detail": 1}),
        ValueError,
        r"exactly the keys",
    ),
    "failure-empty-reason": (_with(failure={"point": None, "reason": ""}), ValueError, r"reason must not be empty"),
    "failure-reason-not-a-string": (_with(failure={"point": None, "reason": 3}), TypeError, r"reason must be a str"),
    # All three points completed, so no id is left for a failed point.
    "failed-point-already-measured": (
        _with(failure={"point": _decode_points()[0], "reason": "x"}),
        ValueError,
        r"3 of 3 points completed before it",
    ),
    "failed-point-beyond-the-grid": (
        _with(failure={"point": {**_decode_points()[0], "benchmark_id": 4}, "reason": "x"}),
        ValueError,
        r"3 of 3 points completed before it",
    ),
    "failed-point-malformed": (
        _with(failure={"point": {"benchmark_id": 4}, "reason": "x"}),
        ValueError,
        r"benchmark point lacks",
    ),
}


@pytest.mark.parametrize("name", sorted(PRODUCER_BUGS))
def test_producer_bugs_raise(name):
    kwargs = _decode_run()
    assert artifact.build_rank_artifact(**copy.deepcopy(kwargs))["status"] == "complete"  # the baseline is sound
    change, error, message = PRODUCER_BUGS[name]
    change(kwargs)
    with pytest.raises(error, match=message):
        artifact.build_rank_artifact(**kwargs)


def test_a_failed_point_must_be_the_next_one_in_id_order():
    points = _decode_points()
    kwargs = _build_kwargs("decode", points, completed=1)
    assert artifact.build_rank_artifact(**kwargs, failure={"point": points[1], "reason": "x"})["status"] == "failed"
    for wrong in (points[0], points[2]):  # already measured; would leave a gap
        with pytest.raises(ValueError, match="completed before it"):
            artifact.build_rank_artifact(**kwargs, failure={"point": wrong, "reason": "x"})
    with pytest.raises(ValueError, match="is a 'prefill' point in mode 'decode'"):
        artifact.build_rank_artifact(**kwargs, failure={"point": _point("prefill", 2, 1, 8, 0), "reason": "x"})


@pytest.mark.parametrize(
    "first",
    [
        _point("decode", 1, 4, 0, 64, ["kvwarm_real_kv"]),
        _point("prefill", 1, 2, 16, 32, ["prefill_real_seed"]),
        _point("prefill", 1, 2, 16, 0),
    ],
    ids=["decode", "cached-prefill", "zero-cached-prefill"],
)
def test_a_second_sample_of_one_coordinate_is_refused_because_publication_would_raise(first):
    # The same work again under the next id, as a manifest point: only the id and a free reason differ.
    second = {**first, "benchmark_id": 2, "sample_reasons": ["explicit", *first["sample_reasons"]]}
    mode = first["point_type"]

    # AISim's own publication step refuses the pair, after the whole run has been collected ...
    collection = native_artifact.NativeCollection(
        points=tuple(
            native_artifact.NativePointMeasurement(
                point=point, rank_wall_times=((0, 0.01),), kv_seed_regime=artifact.kv_seed_regime(point)
            )
            for point in (first, second)
        ),
        rank_timings=((0, 1.0, 0.02),),
        backend_version="fake",
        collector_attempt_id="attempt",
        runtime_run_id=FAKE_RUN_ID,
        runtime_grid_digest=FAKE_GRID_DIGEST,
    )
    with pytest.raises(ValueError, match="conflicting FPM measurements for physical coordinate"):
        select_native_measurements(collection, cell_id="cell")

    # ... so the module refuses it in the pod, when the artifact is built.
    kwargs = _build_kwargs(mode, [first, second])
    with pytest.raises(ValueError, match="benchmark_id=2: second measurement of coordinate"):
        artifact.build_rank_artifact(**kwargs)


# ---------------------------------------------------------------------------------------------------------------
# Merged file
# ---------------------------------------------------------------------------------------------------------------


def test_merged_file_is_the_first_rank_plus_paths_ranks_and_all_fpms():
    points = _decode_points()
    payloads = _rank_artifacts("decode", points, dp=2)
    before = copy.deepcopy(payloads)

    merged = artifact.build_merged_artifact(payloads, ["/results//benchmark.json", "/results/benchmark_dp1.json"])

    assert merged["artifact_type"] == "merged"
    assert merged["rank_files"] == ["/results/benchmark.json", "/results/benchmark_dp1.json"]
    assert merged["merged_output_path"] == "/results/benchmark_merged.json"
    assert merged["dp"] == {"size": 2, "ranks": [0, 1], "source_ranks": [0, 1], "managed_size": 2, "global_size": 2}
    assert "worker_probe_path" not in merged
    for index, row in enumerate(merged["results"]):
        assert row["fpms"] == [payloads[0]["results"][index]["fpms"][0], payloads[1]["results"][index]["fpms"][0]]
        assert [fpm["dp_rank"] for fpm in row["fpms"]] == [0, 1]
    changed = {"artifact_type", "rank_files", "merged_output_path", "dp", "results"}
    assert {key: value for key, value in merged.items() if key not in changed} == {
        key: value for key, value in payloads[0].items() if key not in changed
    }
    # A deep copy: editing the merged document leaves the rank artifacts alone.
    merged["config"]["mode"] = "edited"
    merged["results"][0]["point"]["batch_size"] = 99
    assert payloads == before


def test_merged_file_of_a_later_node_binds_its_own_ranks(tmp_path):
    """Ranks 2 and 3 of a four-rank run: the shape a second node would write (the gate's start_rank is 2)."""
    points = _decode_points()
    payloads = _rank_artifacts("decode", points, dp=4)[2:]
    output_path = str(tmp_path / "benchmark.json")
    paths = [artifact.rank_output_path(output_path, rank) for rank in (2, 3)]

    merged = artifact.build_merged_artifact(payloads, paths)
    assert merged["dp"] == {
        "size": 4,
        "ranks": [0, 1, 2, 3],
        "source_ranks": [2, 3],
        "managed_size": 2,
        "global_size": 4,
    }
    assert merged["merged_output_path"] == str(tmp_path / "benchmark_dp2_merged.json")
    for path, payload in zip(paths, payloads, strict=True):
        artifact.atomic_write_json(path, payload)
    artifact.atomic_write_json(merged["merged_output_path"], merged)

    gate = _gate(paths, "decode", start_rank=2)
    assert gate.returncode == GATE_ACCEPT, gate.stderr


def _two_ranks() -> list[dict]:
    return _rank_artifacts("decode", _decode_points(), dp=2)


# name -> (change applied to the two sound rank artifacts of one run, fragment of the message)
MERGE_BUGS = {
    "swapped-order": (lambda ranks: ranks.reverse(), r"consecutive ascending rank order, got ranks \[1, 0\]"),
    "same-rank-twice": (
        lambda ranks: ranks.__setitem__(1, copy.deepcopy(ranks[0])),
        r"consecutive ascending rank order, got ranks \[0, 0\]",
    ),
    "different-run-id": (lambda ranks: ranks[1].update(run_id="another-run"), r"disagree on run_id"),
    "different-grid-digest": (lambda ranks: ranks[1].update(grid_digest="another-digest"), r"disagree on grid_digest"),
    "different-mode": (lambda ranks: ranks[1]["config"].update(mode="prefill"), r"disagree on config.mode"),
    "different-coverage": (lambda ranks: ranks[1]["coverage"].update(completed_points=2), r"disagree on coverage"),
    "different-dp-size": (lambda ranks: ranks[1]["dp"].update(size=3), r"disagree on dp.size"),
    "different-points": (
        lambda ranks: ranks[1]["results"][0]["point"].update(batch_size=5),
        r"disagree on result points",
    ),
    "different-groups": (
        lambda ranks: ranks[1]["iteration_groups"][0].update(wall_time=9.0),
        r"disagree on iteration_groups",
    ),
    "fewer-rows": (lambda ranks: ranks[1]["results"].pop(), r"disagree on result points"),
    # The kvwarm block is one per run: a rank without it, and a rank that alone names a dataset.
    "kvwarm-block-in-one-rank-only": (
        lambda ranks: ranks[1].pop("kvwarm"),
        r"^ranks disagree on kvwarm: rank 1 differs from rank 0$",
    ),
    "dataset-record-in-one-rank-only": (
        lambda ranks: ranks[0]["kvwarm"].update(dataset=dict(FAKE_KVWARM_DATASET)),
        r"^ranks disagree on kvwarm: rank 1 differs from rank 0$",
    ),
    "row-with-another-ranks-fpm": (
        lambda ranks: ranks[1]["results"][0]["fpms"][0].update(dp_rank=0),
        r"a result row holds the fpm of rank 0",
    ),
    "row-with-two-fpms": (
        lambda ranks: ranks[1]["results"][0]["fpms"].append(ranks[1]["results"][0]["fpms"][0]),
        r"exactly one fpm",
    ),
    "merged-passed-as-rank": (lambda ranks: ranks[0].update(artifact_type="merged"), r"not a schema-2 rank artifact"),
    "another-schema-version": (lambda ranks: ranks[0].update(schema_version=3), r"not a schema-2 rank artifact"),
    "unknown-status": (lambda ranks: [rank.update(status="running") for rank in ranks], r"unknown status 'running'"),
    "worker-probe-advertised": (
        lambda ranks: ranks[0].update(worker_probe_path="/results/probe.json"),
        r"contains the keys \['worker_probe_path'\]",
    ),
    "measurement-protocol-added": (
        lambda ranks: ranks[1].update(measurement_protocol={}),
        r"contains the keys \['measurement_protocol'\]",
    ),
    "rank-outside-size": (
        lambda ranks: [rank["dp"].update(size=1) for rank in ranks],
        r"dp.rank 1 is outside range\(1\)",
    ),
    "no-dp-block": (lambda ranks: ranks[0].pop("dp"), r"no integer dp.size and dp.rank"),
    # Hand-made payloads: each is refused with a ValueError that names the fault, never with a KeyError or
    # AttributeError.
    "dp-size-is-a-bool": (lambda ranks: ranks[0]["dp"].update(size=True), r"no integer dp.size and dp.rank"),
    "dp-rank-is-a-bool": (lambda ranks: ranks[0]["dp"].update(rank=False), r"no integer dp.size and dp.rank"),
    "dp-rank-negative": (lambda ranks: ranks[0]["dp"].update(rank=-1), r"dp.rank -1 is outside range\(2\)"),
    "config-removed": (lambda ranks: ranks[0].pop("config"), r"rank_payloads\[0\] has no config or coverage object"),
    "config-not-a-dict": (
        lambda ranks: ranks[1].update(config=[("mode", "decode")]),
        r"rank_payloads\[1\] has no config or coverage object",
    ),
    "coverage-removed": (
        lambda ranks: ranks[1].pop("coverage"),
        r"rank_payloads\[1\] has no config or coverage object",
    ),
    "results-not-a-list": (
        lambda ranks: ranks[0].update(results=tuple(ranks[0]["results"])),
        r"rank_payloads\[0\] has no results or iteration_groups list",
    ),
    "results-removed": (
        lambda ranks: ranks[1].pop("results"),
        r"rank_payloads\[1\] has no results or iteration_groups list",
    ),
    "iteration-groups-removed": (
        lambda ranks: ranks[1].pop("iteration_groups"),
        r"rank_payloads\[1\] has no results or iteration_groups list",
    ),
    "row-not-a-dict": (
        lambda ranks: ranks[0]["results"].__setitem__(1, None),
        r"rank_payloads\[0\]: every result row must hold exactly one fpm",
    ),
    "row-fpm-not-a-dict": (
        lambda ranks: ranks[1]["results"][0].update(fpms=[None]),
        r"rank_payloads\[1\]: every result row must hold exactly one fpm",
    ),
}


@pytest.mark.parametrize("name", sorted(MERGE_BUGS))
def test_rank_sets_that_do_not_belong_together_are_refused_before_any_write(tmp_path, name):
    ranks = _two_ranks()
    change, message = MERGE_BUGS[name]
    change(ranks)

    with pytest.raises(ValueError, match=message):
        artifact.build_merged_artifact(ranks, ["/results/benchmark.json", "/results/benchmark_dp1.json"])
    with pytest.raises(ValueError, match=message):
        artifact.write_artifacts(str(tmp_path / "benchmark.json"), ranks)
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------------------------------------------
# A complete rank artifact that was edited after it was built
# ---------------------------------------------------------------------------------------------------------------


def _set(key: str, value):
    return lambda payload: payload.__setitem__(key, value)


def _replace_row_fpm(payload: dict) -> None:
    # A new fpm object in the row; the group keeps the old one. (Editing the row's fpm in place would change
    # the group as well: inside one artifact the row and the group hold the same object.)
    row = payload["results"][0]
    row["fpms"] = [{**row["fpms"][0], "worker_id": "edited"}]


def _change_rank_wall_times(payload: dict) -> None:
    # Every rank's fpm of point 2 gets a new time, in the row and in the group alike; the group time stays.
    for item in payload["iteration_groups"][1]["rank_results"]:
        item["fpms"][0]["wall_time"] = 0.75


def _empty_the_run(payload: dict) -> None:
    # No row, no group, and a coverage and a sum that agree with that: only "complete with no point" is wrong.
    payload["results"].clear()
    payload["iteration_groups"].clear()
    payload["coverage"].update(expected_points=0, completed_points=0)
    payload["timing"]["measured_iteration_seconds"] = 0.0


def _first_group_entry(payload: dict) -> dict:
    return payload["iteration_groups"][0]["rank_results"][0]


def _add_a_second_fpm_to_a_group_entry(payload: dict) -> None:
    fpms = _first_group_entry(payload)["fpms"]
    fpms.append(copy.deepcopy(fpms[0]))


def _add_a_group_rank_entry(payload: dict) -> None:
    # One entry more than the run has ranks: a copy of the last entry under the next rank number, so the
    # largest rank time of the group stays what it was and the count is the only fault.
    entries = payload["iteration_groups"][0]["rank_results"]
    entries.append({**copy.deepcopy(entries[-1]), "dp_rank": len(entries)})


# name -> (one edit of a complete decode rank artifact, the field the refusal names)
COMPLETE_ARTIFACT_EDITS = {
    "valid-false": (_set("valid", False), r"valid"),
    "usable-false": (_set("usable", False), r"usable"),
    "timing-valid-false": (_set("timing_valid", False), r"timing_valid"),
    "stop-reason-timeout": (_set("stop_reason", "timeout"), r"stop_reason"),
    "error-text": (_set("error", "x"), r"error"),
    "skipped-point-listed": (
        lambda payload: payload["skipped_points"].append({"point": _decode_points()[0], "reason": "x"}),
        r"skipped_points",
    ),
    "missing-phase-listed": (_set("missing_phases", ["decode"]), r"missing_phases"),
    # The same fields with a value that a looser comparison would let through: a true value that is not True,
    # an empty value that is not None, and no list where an empty list belongs.
    "usable-is-the-int-1": (_set("usable", 1), r"usable must be True in a complete artifact, got 1$"),
    "timing-valid-is-a-string": (
        _set("timing_valid", "yes"),
        r"timing_valid must be True in a complete artifact, got 'yes'$",
    ),
    "stop-reason-is-an-empty-string": (
        _set("stop_reason", ""),
        r"stop_reason must be None in a complete artifact, got ''$",
    ),
    "error-is-an-empty-string": (_set("error", ""), r"error must be None in a complete artifact, got ''$"),
    "skipped-points-none": (
        _set("skipped_points", None),
        r"skipped_points must be \[\] in a complete artifact, got None$",
    ),
    "missing-phases-removed": (
        lambda payload: payload.pop("missing_phases"),
        r"missing_phases must be \[\] in a complete artifact, got None$",
    ),
    "coverage-expects-more": (
        lambda payload: payload["coverage"].update(expected_points=4),
        r"coverage\.expected_points",
    ),
    "coverage-completed-fewer": (
        lambda payload: payload["coverage"].update(completed_points=2),
        r"coverage\.completed_points",
    ),
    "coverage-skipped-one": (lambda payload: payload["coverage"].update(skipped_points=1), r"coverage\.skipped_points"),
    # False == 0, so the validator lets this one through; the gate wants exact ints and rejects it.
    "coverage-count-is-a-bool": (
        lambda payload: payload["coverage"].update(skipped_points=False),
        r"coverage\.skipped_points",
    ),
    "row-removed": (lambda payload: payload["results"].pop(), r"coverage\.expected_points"),
    "no-rows-at-all": (_empty_the_run, r"results is empty"),
    "group-removed": (lambda payload: payload["iteration_groups"].pop(), r"iteration_groups"),
    "group-added": (
        lambda payload: payload["iteration_groups"].append(copy.deepcopy(payload["iteration_groups"][-1])),
        r"iteration_groups holds 4 groups for 3 results$",
    ),
    "row-fpm-differs-from-group": (_replace_row_fpm, r"results\[0\]\.fpms"),
    "row-wall-time-changed": (
        lambda payload: payload["results"][0].update(fpms=[{**payload["results"][0]["fpms"][0], "wall_time": 0.02}]),
        r"results\[0\]\.fpms",
    ),
    "group-wall-time-changed": (
        lambda payload: payload["iteration_groups"][1].update(wall_time=9.0),
        r"iteration_groups\[1\]\.wall_time",
    ),
    "rank-wall-times-changed-under-the-group": (_change_rank_wall_times, r"iteration_groups\[1\]\.wall_time"),
    "group-rank-entry-removed": (
        lambda payload: payload["iteration_groups"][2]["rank_results"].pop(),
        r"iteration_groups\[2\]\.rank_results must hold one entry per rank",
    ),
    "group-rank-entry-mislabelled": (
        lambda payload: _first_group_entry(payload).update(dp_rank=7),
        r"iteration_groups\[0\]\.rank_results\[0\] must hold dp_rank 0 and exactly one fpm",
    ),
    "group-rank-entry-without-fpm": (
        lambda payload: _first_group_entry(payload).update(fpms=[]),
        r"iteration_groups\[0\]\.rank_results\[0\] must hold dp_rank 0 and exactly one fpm",
    ),
    "group-fpm-without-wall-time": (
        lambda payload: _first_group_entry(payload)["fpms"][0].update(wall_time=None),
        r"iteration_groups\[0\]\.rank_results\[0\] has no float fpm wall_time",
    ),
    "group-fpm-wall-time-is-an-int": (
        lambda payload: _first_group_entry(payload)["fpms"][0].update(wall_time=1),
        r"iteration_groups\[0\]\.rank_results\[0\] has no float fpm wall_time",
    ),
    # Groups and rank entries of the wrong shape: each is refused with the ValueError that names the field,
    # never with an AttributeError or a TypeError from reading it.
    "group-not-a-dict": (
        lambda payload: payload["iteration_groups"].__setitem__(0, None),
        r"iteration_groups\[0\]\.rank_results must hold one entry per rank",
    ),
    "group-rank-results-none": (
        lambda payload: payload["iteration_groups"][0].update(rank_results=None),
        r"iteration_groups\[0\]\.rank_results must hold one entry per rank",
    ),
    "group-rank-entry-added": (
        _add_a_group_rank_entry,
        r"iteration_groups\[0\]\.rank_results must hold one entry per rank",
    ),
    "group-rank-entry-not-a-dict": (
        lambda payload: payload["iteration_groups"][0]["rank_results"].__setitem__(0, None),
        r"iteration_groups\[0\]\.rank_results\[0\] must hold dp_rank 0 and exactly one fpm",
    ),
    "group-rank-entry-with-two-fpms": (
        _add_a_second_fpm_to_a_group_entry,
        r"iteration_groups\[0\]\.rank_results\[0\] must hold dp_rank 0 and exactly one fpm",
    ),
    "group-fpm-not-a-dict": (
        lambda payload: _first_group_entry(payload).update(fpms=[None]),
        r"iteration_groups\[0\]\.rank_results\[0\] must hold dp_rank 0 and exactly one fpm",
    ),
    "measured-seconds-doubled": (
        lambda payload: payload["timing"].update(
            measured_iteration_seconds=payload["timing"]["measured_iteration_seconds"] * 2
        ),
        r"timing\.measured_iteration_seconds",
    ),
    "timing-removed": (lambda payload: payload.pop("timing"), r"timing\.measured_iteration_seconds"),
    "kvwarm-removed": (lambda payload: payload.pop("kvwarm"), r"kvwarm"),
    "kvwarm-not-eligible": (
        lambda payload: payload["kvwarm"].update(warm_eligible=False, skip_reason="fake reason"),
        r"kvwarm",
    ),
    # 1 == True, so the block still compares equal to the builder's; the validator wants a real boolean.
    "kvwarm-enabled-is-an-int": (lambda payload: payload["kvwarm"].update(enabled=1), r"kvwarm"),
    "row-regime-fake-fallback": (
        lambda payload: payload["results"][0].update(kv_seed_regime="fake_fallback"),
        r"results\[0\]\.kv_seed_regime",
    ),
    "row-stamp-removed": (
        lambda payload: payload["results"][0]["point"].update(sample_reasons=[]),
        r"results\[0\]\.point",
    ),
    # No point at all in the row: the point check raises TypeError, the refusal is still the ValueError.
    "row-point-none": (
        lambda payload: payload["results"][0].update(point=None),
        r"results\[0\]\.point is not a point of a known regime",
    ),
}


@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("name", sorted(COMPLETE_ARTIFACT_EDITS))
def test_a_complete_artifact_edited_after_the_build_is_refused_before_any_write(tmp_path, name, dp):
    """Each edit makes a ``complete`` artifact that a checker rejects, so the module must not write it."""
    edit, field = COMPLETE_ARTIFACT_EDITS[name]
    cell = _cell(phase="decode", dp=dp, strategy="dep")
    points = _decode_points()

    # Unedited, the artifacts are accepted by the validator and the gate.
    sound = tmp_path / "sound"
    pod = _pod(sound, cell)
    written = artifact.write_artifacts(str(pod / "benchmark.json"), _rank_artifacts("decode", points, dp=dp))
    assert len(_validate(cell, sound).points) == len(points)
    assert _gate(written["rank_files"], "decode").returncode == GATE_ACCEPT

    # The same edit in every rank file, so that the ranks still agree with each other.
    payloads = _rank_artifacts("decode", points, dp=dp)
    for payload in payloads:
        edit(payload)
        assert payload["status"] == "complete"

    # Put on disk as they are, the edited rank files are rejected: by the validator, except for the one edit
    # that only the gate catches. (There is no merged file here, so a gate that finds no fault says "pending".)
    forced = tmp_path / "forced"
    pod = _pod(forced, cell)
    rank_paths = [artifact.rank_output_path(str(pod / "benchmark.json"), rank) for rank in range(dp)]
    for path, payload in zip(rank_paths, payloads, strict=True):
        artifact.atomic_write_json(path, payload)
    if name == "coverage-count-is-a-bool":
        assert len(_validate(cell, forced).points) == len(points)
        assert _gate(rank_paths, "decode").returncode == GATE_INVALID
    else:
        with pytest.raises((ValueError, TypeError)):
            _validate(cell, forced)
        gate_catches_it = name.startswith("coverage-") or name in {
            "valid-false",
            "row-removed",
            "no-rows-at-all",
            "row-point-none",
        }
        expected_gate = GATE_INVALID if gate_catches_it else GATE_MERGED_PENDING
        assert _gate(rank_paths, "decode").returncode == expected_gate

    # ... so the module refuses them, naming the rank and the field, and writes nothing.
    refused = tmp_path / "refused"
    refused.mkdir()
    message = rf"rank 0: .*{field}"
    with pytest.raises(ValueError, match=message):
        artifact.write_artifacts(str(refused / "benchmark.json"), payloads)
    with pytest.raises(ValueError, match=message):
        artifact.build_merged_artifact(payloads, [f"/results/benchmark_dp{rank}.json" for rank in range(dp)])
    assert list(refused.iterdir()) == []


def test_the_refusal_of_an_edited_complete_artifact_names_the_rank_that_was_edited(tmp_path):
    ranks = _rank_artifacts("decode", _decode_points(), dp=3)
    ranks[2]["usable"] = False
    with pytest.raises(ValueError, match=r"rank 2: usable must be True in a complete artifact, got False"):
        artifact.write_artifacts(str(tmp_path / "benchmark.json"), ranks)

    ranks = _rank_artifacts("decode", _decode_points(), dp=3)
    ranks[1]["timing"]["measured_iteration_seconds"] = 0.5
    with pytest.raises(ValueError, match=r"rank 1: timing\.measured_iteration_seconds is 0\.5, the group times sum to"):
        artifact.write_artifacts(str(tmp_path / "benchmark.json"), ranks)

    # The later ranks of a run (what a second node would merge) are checked against their own group entries.
    ranks = _rank_artifacts("decode", _decode_points(), dp=4)[2:]
    paths = ["/results/benchmark_dp2.json", "/results/benchmark_dp3.json"]
    assert artifact.build_merged_artifact(ranks, paths)["dp"]["source_ranks"] == [2, 3]
    row = ranks[1]["results"][0]
    row["fpms"] = [{**row["fpms"][0], "worker_id": "edited"}]
    with pytest.raises(ValueError, match=r"rank 3: results\[0\]\.fpms\[0\] differs"):
        artifact.build_merged_artifact(ranks, paths)
    assert list(tmp_path.iterdir()) == []


def test_the_complete_envelope_is_checked_in_every_mode_and_only_for_complete_artifacts(tmp_path):
    output_path = str(tmp_path / "benchmark.json")

    # prefill: the same rules, and no kvwarm block is asked for.
    (prefill,) = _rank_artifacts("prefill", _prefill_points())
    assert "kvwarm" not in prefill
    prefill["error"] = "x"
    with pytest.raises(ValueError, match=r"rank 0: error must be None in a complete artifact, got 'x'"):
        artifact.write_artifacts(output_path, [prefill])

    # agg: the builder writes a kvwarm block, so its absence is an edit.
    agg_points = _prefill_points() + [
        {**point, "benchmark_id": point["benchmark_id"] + 4} for point in _decode_points()
    ]
    (agg,) = _rank_artifacts("agg", agg_points)
    del agg["kvwarm"]
    with pytest.raises(ValueError, match=r"rank 0: kvwarm must be .* in a complete 'agg' artifact, got None"):
        artifact.write_artifacts(output_path, [agg])
    assert list(tmp_path.iterdir()) == []

    # A partial artifact is not valid, has a stop reason and fewer rows than points ON PURPOSE: it is written.
    (partial,) = _rank_artifacts("decode", _decode_points(), completed=1, stop_reason="timeout")
    assert (partial["valid"], partial["stop_reason"], len(partial["results"])) == (False, "timeout", 1)
    assert artifact.write_artifacts(output_path, [partial])["merged_output_path"] is not None


def test_an_extra_key_in_the_kvwarm_block_is_refused_although_no_checker_rejects_it(tmp_path):
    """The complete-envelope check asks for exactly the block the builder writes; the validator is looser."""
    cell = _cell(phase="decode", strategy="dep")
    pod = _pod(tmp_path, cell)
    (payload,) = _rank_artifacts("decode", _decode_points())
    payload["kvwarm"]["fake_note"] = "added after the build"
    edited_block = {
        "enabled": True,
        "warm_eligible": True,
        "skip_reason": None,
        "method": "radix_cache_seed",
        "fake_note": "added after the build",
    }

    # Put on disk as it is, the rank file passes the validator, and the gate finds no fault in it (there is
    # no merged file here, so the gate says "pending").
    rank_path = str(pod / "benchmark.json")
    artifact.atomic_write_json(rank_path, payload)
    assert _validate(cell, tmp_path).kvwarm_meta == edited_block
    assert _gate([rank_path], "decode").returncode == GATE_MERGED_PENDING

    # The module still refuses it: the block is not the builder's.
    refused = tmp_path / "refused"
    refused.mkdir()
    message = r"^rank 0: kvwarm must be .* in a complete 'decode' artifact, got .*'fake_note': 'added after the build'"
    with pytest.raises(ValueError, match=message):
        artifact.write_artifacts(str(refused / "benchmark.json"), [payload])
    with pytest.raises(ValueError, match=message):
        artifact.build_merged_artifact([payload], ["/results/benchmark.json"])
    assert list(refused.iterdir()) == []


NOT_THE_BUILDERS_BLOCK = r"kvwarm must be .*, alone or with a 'dataset' record, in a complete 'decode' artifact, got "

# name -> (one edit of the kvwarm block of a complete decode rank artifact that names a dataset, the refusal)
DATASET_RECORD_EDITS = {
    "record-none": (lambda block: block.update(dataset=None), r"kvwarm\.dataset must be a dict, got NoneType$"),
    "record-a-list": (
        lambda block: block.update(dataset=list(block["dataset"].items())),
        r"kvwarm\.dataset must be a dict, got list$",
    ),
    "record-emptied": (
        lambda block: block["dataset"].clear(),
        r"kvwarm\.dataset must have exactly the keys .* got \[\]$",
    ),
    "path-removed": (
        lambda block: block["dataset"].pop("path"),
        r"kvwarm\.dataset must have exactly the keys .* got \['collection_pool', 'conversations', 'sha256'\]$",
    ),
    "a-fifth-key-in-the-record": (
        lambda block: block["dataset"].update(url="https://example.invalid/fake_dataset.json"),
        r"kvwarm\.dataset must have exactly the keys ",
    ),
    "path-emptied": (lambda block: block["dataset"].update(path=""), r"kvwarm\.dataset\['path'\] must not be empty$"),
    "sha256-upper-case": (
        lambda block: block["dataset"].update(sha256=block["dataset"]["sha256"].upper()),
        r"kvwarm\.dataset\['sha256'\] must be the 64 lower-case hexadecimal digits of a SHA-256, got '0123456789ABCDEF",
    ),
    "sha256-cut": (
        lambda block: block["dataset"].update(sha256=block["dataset"]["sha256"][:40]),
        r"kvwarm\.dataset\['sha256'\] must be the 64 lower-case hexadecimal digits of a SHA-256, got ",
    ),
    "collection-pool-none": (
        lambda block: block["dataset"].update(collection_pool=None),
        r"kvwarm\.dataset\['collection_pool'\] must be a str, got None$",
    ),
    "conversations-zero": (
        lambda block: block["dataset"].update(conversations=0),
        r"kvwarm\.dataset\['conversations'\] must be >= 1, got 0$",
    ),
    # 1234.0 == 1234 and True == 1: the record still compares equal to a sound one, its type is wrong.
    "conversations-a-float": (
        lambda block: block["dataset"].update(conversations=1234.0),
        r"kvwarm\.dataset\['conversations'\] must be an int, got 1234\.0$",
    ),
    "conversations-a-bool": (
        lambda block: block["dataset"].update(conversations=True),
        r"kvwarm\.dataset\['conversations'\] must be an int, got True$",
    ),
    # A sound record does not open the block to other keys.
    "another-key-beside-the-record": (
        lambda block: block.update(fake_note="added after the build"),
        NOT_THE_BUILDERS_BLOCK + r".*'fake_note': 'added after the build'",
    ),
    "the-record-under-another-key": (
        lambda block: block.update(datasets=block.pop("dataset")),
        NOT_THE_BUILDERS_BLOCK + r".*'datasets': ",
    ),
}


@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("name", sorted(DATASET_RECORD_EDITS))
def test_a_dataset_record_edited_after_the_build_is_refused_although_no_checker_rejects_it(tmp_path, name, dp):
    """Neither checker has a rule for the record, so only the module keeps a ``complete`` artifact to its form."""
    edit, refusal = DATASET_RECORD_EDITS[name]
    cell = _cell(phase="decode", dp=dp, strategy="tep")
    points = _decode_points()
    # The same edit in every rank file, so that the ranks still agree with each other.
    payloads = _rank_artifacts("decode", points, dp=dp, kvwarm_dataset=FAKE_KVWARM_DATASET)
    for payload in payloads:
        edit(payload["kvwarm"])
        assert payload["status"] == "complete"

    # Put on disk as they are, the rank files pass the validator, which hands the edited block on, and the
    # gate finds no fault in them (there is no merged file here, so the gate says "pending").
    pod = _pod(tmp_path, cell)
    rank_paths = [artifact.rank_output_path(str(pod / "benchmark.json"), rank) for rank in range(dp)]
    for path, payload in zip(rank_paths, payloads, strict=True):
        artifact.atomic_write_json(path, payload)
    collection = _validate(cell, tmp_path)
    assert collection.kvwarm_meta == json.loads(json.dumps(payloads[0]["kvwarm"]))
    assert len(collection.points) == len(points)
    assert _gate(rank_paths, "decode").returncode == GATE_MERGED_PENDING

    # The module refuses them, naming the rank and the field, and writes nothing.
    refused = tmp_path / "refused"
    refused.mkdir()
    message = rf"^rank 0: {refusal}"
    with pytest.raises(ValueError, match=message):
        artifact.write_artifacts(str(refused / "benchmark.json"), payloads)
    with pytest.raises(ValueError, match=message):
        artifact.build_merged_artifact(payloads, [f"/results/benchmark_dp{rank}.json" for rank in range(dp)])
    assert list(refused.iterdir()) == []


# name -> one edit of the four fixed keys of a kvwarm block that names a dataset
FIXED_KEY_EDITS = {
    "method-removed": lambda block: block.pop("method"),
    "method-changed": lambda block: block.update(method="fake_method"),
    "not-eligible": lambda block: block.update(warm_eligible=False, skip_reason="dataset_unavailable"),
    # 1 == True, so the fixed keys still compare equal to the builder's; the validator wants a real boolean.
    "enabled-is-an-int": lambda block: block.update(enabled=1),
}


@pytest.mark.parametrize("name", sorted(FIXED_KEY_EDITS))
def test_the_fixed_keys_of_a_kvwarm_block_are_held_to_the_builders_values_beside_a_dataset_record(tmp_path, name):
    payloads = _rank_artifacts("decode", _decode_points(), dp=2, kvwarm_dataset=FAKE_KVWARM_DATASET)
    assert artifact.build_merged_artifact(payloads, ["/results/b.json", "/results/b_dp1.json"])["kvwarm"] == (
        EXPECTED_KVWARM_WITH_DATASET
    )
    for payload in payloads:
        FIXED_KEY_EDITS[name](payload["kvwarm"])
    with pytest.raises(ValueError, match=rf"^rank 0: {NOT_THE_BUILDERS_BLOCK}.*'dataset': "):
        artifact.write_artifacts(str(tmp_path / "benchmark.json"), payloads)
    assert list(tmp_path.iterdir()) == []


def test_a_complete_artifact_may_carry_either_form_of_the_block_but_every_rank_the_same(tmp_path):
    # The check knows the two forms the builder writes, not which of them the run was built with: a sound
    # record that is put into every rank file after the build, or taken out of every one, leaves a block
    # of the other form.
    paths = ["/results/benchmark.json", "/results/benchmark_dp1.json"]
    added = _rank_artifacts("decode", _decode_points(), dp=2)
    for payload in added:
        payload["kvwarm"]["dataset"] = dict(FAKE_KVWARM_DATASET)
    assert artifact.build_merged_artifact(added, paths)["kvwarm"] == EXPECTED_KVWARM_WITH_DATASET
    removed = _rank_artifacts("decode", _decode_points(), dp=2, kvwarm_dataset=FAKE_KVWARM_DATASET)
    for payload in removed:
        del payload["kvwarm"]["dataset"]
    assert artifact.build_merged_artifact(removed, paths)["kvwarm"] == EXPECTED_KVWARM

    # In one rank file only, it is two blocks for one run.
    del added[1]["kvwarm"]["dataset"]
    with pytest.raises(ValueError, match=r"^ranks disagree on kvwarm: rank 1 differs from rank 0$"):
        artifact.build_merged_artifact(added, paths)
    with pytest.raises(ValueError, match=r"^ranks disagree on kvwarm: rank 1 differs from rank 0$"):
        artifact.write_artifacts(str(tmp_path / "benchmark.json"), added)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("status", ["complete", "partial", "failed"])
def test_ranks_that_name_different_datasets_are_refused_whatever_the_status_of_the_run(tmp_path, status):
    points = _decode_points()
    ending = {
        "complete": {},
        "partial": {"completed": 1, "stop_reason": "timeout"},
        "failed": {"completed": 1, "failure": {"point": points[1], "reason": "fake reason"}},
    }[status]
    cell = _cell(phase="decode", dp=3, strategy="tep")
    ranks = _rank_artifacts("decode", points, dp=3, kvwarm_dataset=FAKE_KVWARM_DATASET, **ending)
    # Rank 2 was built with the record of another file: a sound record, and not the one of ranks 0 and 1.
    other = _rank_artifacts("decode", points, dp=3, kvwarm_dataset=_dataset(sha256="f" * 64), **ending)
    ranks[2] = other[2]
    assert [payload["status"] for payload in ranks] == [status] * 3

    if status == "complete":
        # The validator compares three of the fixed keys across ranks, so it accepts these files and hands on
        # the block of the first one.
        pod = _pod(tmp_path / "forced", cell)
        for rank, payload in enumerate(ranks):
            artifact.atomic_write_json(artifact.rank_output_path(str(pod / "benchmark.json"), rank), payload)
        assert _validate(cell, tmp_path / "forced").kvwarm_meta == EXPECTED_KVWARM_WITH_DATASET

    refused = tmp_path / "refused"
    refused.mkdir()
    message = r"^ranks disagree on kvwarm: rank 2 differs from rank 0$"
    with pytest.raises(ValueError, match=message):
        artifact.write_artifacts(str(refused / "benchmark.json"), ranks)
    with pytest.raises(ValueError, match=message):
        artifact.build_merged_artifact(ranks, [f"/results/benchmark_dp{rank}.json" for rank in range(3)])
    assert list(refused.iterdir()) == []


def _group_time_one_float_up(payload: dict) -> None:
    group = payload["iteration_groups"][0]
    group["wall_time"] = math.nextafter(group["wall_time"], math.inf)


def _group_time_one_float_down(payload: dict) -> None:
    group = payload["iteration_groups"][0]
    group["wall_time"] = math.nextafter(group["wall_time"], 0.0)


def _measured_sum_one_float_up(payload: dict) -> None:
    timing = payload["timing"]
    timing["measured_iteration_seconds"] = math.nextafter(timing["measured_iteration_seconds"], math.inf)


def _measured_sum_one_float_down(payload: dict) -> None:
    timing = payload["timing"]
    timing["measured_iteration_seconds"] = math.nextafter(timing["measured_iteration_seconds"], 0.0)


def _group_fpm_time_a_float_subclass(payload: dict) -> None:
    fpm = payload["iteration_groups"][0]["rank_results"][0]["fpms"][0]
    fpm["wall_time"] = _FloatSubclass(fpm["wall_time"])


# name -> (edit of a complete decode artifact, the module's refusal)
EXACT_TIME_EDITS = {
    "group-wall-time-one-float-up": (
        _group_time_one_float_up,
        r"^rank 0: iteration_groups\[0\]\.wall_time is .*, the largest rank wall_time is ",
    ),
    "group-wall-time-one-float-down": (
        _group_time_one_float_down,
        r"^rank 0: iteration_groups\[0\]\.wall_time is .*, the largest rank wall_time is ",
    ),
    "measured-sum-one-float-up": (
        _measured_sum_one_float_up,
        r"^rank 0: timing\.measured_iteration_seconds is .*, the group times sum to ",
    ),
    "measured-sum-one-float-down": (
        _measured_sum_one_float_down,
        r"^rank 0: timing\.measured_iteration_seconds is .*, the group times sum to ",
    ),
    "group-fpm-wall-time-a-float-subclass": (
        _group_fpm_time_a_float_subclass,
        r"^rank 0: iteration_groups\[0\]\.rank_results\[0\] has no float fpm wall_time$",
    ),
}


@pytest.mark.parametrize("name", sorted(EXACT_TIME_EDITS))
def test_a_time_that_is_not_exactly_the_builders_is_refused_although_no_checker_rejects_it(tmp_path, name):
    """The complete-envelope check wants exactly the builder's times, and exactly the type float.

    The validator allows a relative tolerance, so it accepts a group time or a sum that is one float away from
    the builder's, in either direction. A float subclass is written as the same number as the float it holds:
    the file of that edit is the unedited file byte for byte, so no checker can see the edit at all, and the
    module's refusal of it pins the exact type and not an exact time.
    """
    edit, message = EXACT_TIME_EDITS[name]
    cell = _cell(phase="decode", strategy="dep")
    pod = _pod(tmp_path, cell)
    (payload,) = _rank_artifacts("decode", _decode_points())
    unedited = copy.deepcopy(payload)
    edit(payload)

    # Put on disk as it is, the rank file passes the validator, and the gate finds no fault in it (there is
    # no merged file here, so the gate says "pending").
    rank_path = str(pod / "benchmark.json")
    artifact.atomic_write_json(rank_path, payload)
    _validate(cell, tmp_path)
    assert _gate([rank_path], "decode").returncode == GATE_MERGED_PENDING
    # What the checkers were given differs from the unedited file in one number, except for the float subclass.
    unedited_path = tmp_path / "unedited.json"
    artifact.atomic_write_json(str(unedited_path), unedited)
    same_file = Path(rank_path).read_bytes() == unedited_path.read_bytes()
    assert same_file is (name == "group-fpm-wall-time-a-float-subclass")

    # The module still refuses it: the value is one float away from the builder's, or its type is not float.
    refused = tmp_path / "refused"
    refused.mkdir()
    with pytest.raises(ValueError, match=message):
        artifact.write_artifacts(str(refused / "benchmark.json"), [payload])
    with pytest.raises(ValueError, match=message):
        artifact.build_merged_artifact([payload], ["/results/benchmark.json"])
    assert list(refused.iterdir()) == []


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_an_int_time_placed_alike_in_the_fpm_the_group_and_the_sum_is_refused_by_the_type_rule_alone(tmp_path, phase):
    """``1 == 1.0``: every comparison of the complete-envelope check holds for this edit, only the type is wrong."""
    cell = _cell(phase=phase, strategy="dep")
    pod = _pod(tmp_path, cell)
    point = _points(phase)[0]
    if phase == "decode":
        steps = [0.5, 1.4, 1.2, 0.9, 1.0, 0.3]  # FAKE seconds; the upper median of steps 3 to 5 is exactly 1.0
        fpm = artifact.build_fpm(point, 0, artifact.decode_wall_time(steps), extra={STEPS_KEY: steps})
    else:
        fpm = artifact.build_fpm(point, 0, 1.0)  # FAKE: a batch of exactly one second
    kwargs = {**_build_kwargs(phase, [point]), "measurements": [artifact.build_measurement(point, {0: fpm})]}
    payload = artifact.build_rank_artifact(**kwargs, producer=LAUNCHER_PRODUCER)
    assert payload["status"] == "complete"
    assert artifact.build_merged_artifact([payload], ["/results/benchmark.json"])["status"] == "complete"

    # The int 1 in place of the float 1.0, in all three places. (The row and the group hold the same fpm object.)
    payload["results"][0]["fpms"][0]["wall_time"] = 1
    payload["iteration_groups"][0]["wall_time"] = 1
    payload["timing"]["measured_iteration_seconds"] = 1
    assert payload["iteration_groups"][0]["rank_results"][0]["fpms"][0]["wall_time"] == 1

    # Put on disk as it is, the rank file passes the validator, which reads the time as 1.0, and the gate finds
    # no fault in it (there is no merged file here, so the gate says "pending").
    rank_path = str(pod / "benchmark.json")
    artifact.atomic_write_json(rank_path, payload)
    collection = _validate(cell, tmp_path)
    assert [measurement.rank_wall_times for measurement in collection.points] == [((0, 1.0),)]
    assert collection.rank_timings == ((0, FAKE_ELAPSED_SECONDS, 1.0),)
    assert _gate([rank_path], phase).returncode == GATE_MERGED_PENDING

    # The module still refuses it: a time in an artifact is a float.
    refused = tmp_path / "refused"
    refused.mkdir()
    message = r"^rank 0: iteration_groups\[0\]\.rank_results\[0\] has no float fpm wall_time$"
    with pytest.raises(ValueError, match=message):
        artifact.write_artifacts(str(refused / "benchmark.json"), [payload])
    with pytest.raises(ValueError, match=message):
        artifact.build_merged_artifact([payload], ["/results/benchmark.json"])
    assert list(refused.iterdir()) == []


# ---------------------------------------------------------------------------------------------------------------
# The raw decode steps of a complete rank artifact that was edited after it was built
# ---------------------------------------------------------------------------------------------------------------


def _group_fpm(payload: dict, index: int, rank: int) -> dict:
    """The fpm of ``rank`` in group ``index``. In that rank's own file the row holds this same object."""
    return payload["iteration_groups"][index]["rank_results"][rank]["fpms"][0]


def _refresh_times(payload: dict) -> None:
    """Derive every group time and the measured sum from the fpm times again, with the builder's arithmetic."""
    total = 0.0
    for group in payload["iteration_groups"]:
        group["wall_time"] = max(item["fpms"][0]["wall_time"] for item in group["rank_results"])
        total += group["wall_time"]
    payload["timing"]["measured_iteration_seconds"] = total


def _edit_in_every_rank_file(payloads: list[dict], index: int, rank: int, edit) -> None:
    """One edit of one fpm, made alike in every rank file, with the times derived from the fpms afterwards.

    The files then still agree with each other, each row still equals its group entry, and every group time
    and sum still follows from the fpm times: the edited fpm itself is all that is wrong.
    """
    for payload in payloads:
        unchanged = copy.deepcopy(payload)
        _refresh_times(payload)
        assert payload == unchanged, "the helper must repeat the builder's arithmetic"
        edit(_group_fpm(payload, index, rank))
        _refresh_times(payload)
        assert payload["status"] == "complete"


def _other_kept_steps(fpm: dict) -> None:
    # FAKE seconds in place of decode steps 3 to 5. Their upper median is 9.25, and wall_time is none of them.
    fpm[STEPS_KEY] = [*fpm[STEPS_KEY][:2], 9.0, 9.5, 9.25, fpm[STEPS_KEY][5]]


def _wall_time_one_float_up(fpm: dict) -> None:
    fpm["wall_time"] = math.nextafter(fpm["wall_time"], math.inf)


# name -> (edit of one decode fpm, exception, the refusal after the place; "old" and "new" are the fpm's
# wall_time before and after the edit)
COMPLETE_STEP_EDITS = {
    "steps-removed": (
        lambda fpm: fpm.pop(STEPS_KEY),
        ValueError,
        "a decode fpm must carry its 6 raw step times under 'sglbench_decode_steps'",
    ),
    "five-values": (
        lambda fpm: fpm.update({STEPS_KEY: fpm[STEPS_KEY][:5]}),
        ValueError,
        "fpm sglbench_decode_steps: a decode run has exactly 6 decode steps, got 5",
    ),
    "a-zero-step": (
        lambda fpm: fpm.update({STEPS_KEY: [0.0, *fpm[STEPS_KEY][1:]]}),
        ValueError,
        "fpm sglbench_decode_steps: raw_step_seconds[0] must be finite and > 0, got 0.0",
    ),
    "an-int-step": (
        lambda fpm: fpm.update({STEPS_KEY: [*fpm[STEPS_KEY][:5], 1]}),
        TypeError,
        "fpm sglbench_decode_steps: raw_step_seconds[5] must be a float in seconds, got 1",
    ),
    "steps-none": (
        lambda fpm: fpm.update({STEPS_KEY: None}),
        TypeError,
        "fpm sglbench_decode_steps must be a list, got NoneType",
    ),
    "steps-changed-under-the-wall-time": (
        _other_kept_steps,
        ValueError,
        "fpm wall_time {old!r} is not the sample 9.25 of its raw steps",
    ),
    # The group time and the measured sum are derived again from the moved time, so every time in the envelope
    # still agrees with every other.
    "wall-time-one-float-up-with-its-group-time-and-the-sum": (
        _wall_time_one_float_up,
        ValueError,
        "fpm wall_time {new!r} is not the sample {old!r} of its raw steps",
    ),
}


@pytest.mark.parametrize(
    ("dp", "rank", "index"),
    # The edited fpm is the slowest of its group: rank 0 is the slower rank at point 3, rank 1 at point 2.
    [(1, 0, 1), (2, 0, 2), (2, 1, 1)],
    ids=["dp1", "dp2-own-entry-of-rank-0", "dp2-entry-of-the-other-rank"],
)
@pytest.mark.parametrize("name", sorted(COMPLETE_STEP_EDITS))
def test_a_complete_artifact_whose_decode_steps_were_edited_is_refused_although_no_checker_rejects_it(
    tmp_path, name, dp, rank, index
):
    """No AISim reader parses the raw steps, so the module is the only place that can refuse such an edit."""
    edit, error, reason = COMPLETE_STEP_EDITS[name]
    cell = _cell(phase="decode", dp=dp, strategy="dep")
    points = _decode_points()
    payloads = _rank_artifacts("decode", points, dp=dp)
    old = _group_fpm(payloads[0], index, rank)["wall_time"]
    assert old == payloads[0]["iteration_groups"][index]["wall_time"], "the edited fpm must be its group's slowest"

    _edit_in_every_rank_file(payloads, index, rank, edit)
    new = _group_fpm(payloads[0], index, rank)["wall_time"]
    if name == "wall-time-one-float-up-with-its-group-time-and-the-sum":
        assert new == math.nextafter(old, math.inf)
        assert payloads[0]["iteration_groups"][index]["wall_time"] == new  # the group time moved with it
    else:
        assert new == old

    # Put on disk as they are, the rank files pass the validator, and the gate finds no fault in them (there
    # is no merged file here, so the gate says "pending").
    forced = tmp_path / "forced"
    pod = _pod(forced, cell)
    rank_paths = [artifact.rank_output_path(str(pod / "benchmark.json"), file_rank) for file_rank in range(dp)]
    for path, payload in zip(rank_paths, payloads, strict=True):
        artifact.atomic_write_json(path, payload)
    assert [measurement.point for measurement in _validate(cell, forced).points] == points
    assert _gate(rank_paths, "decode").returncode == GATE_MERGED_PENDING

    # The module refuses them with the error build_fpm raises for the same fault, in front of it the rank file
    # it was checking, the group and the rank entry. Nothing is written.
    refused = tmp_path / "refused"
    refused.mkdir()
    message = f"rank 0: iteration_groups[{index}].rank_results[{rank}]: " + reason.format(old=old, new=new)
    with pytest.raises(error) as at_the_write:
        artifact.write_artifacts(str(refused / "benchmark.json"), payloads)
    with pytest.raises(error) as at_the_merge:
        artifact.build_merged_artifact(payloads, [f"/results/benchmark_dp{file_rank}.json" for file_rank in range(dp)])
    for raised in (at_the_write, at_the_merge):
        assert type(raised.value) is error
        assert str(raised.value) == message
    assert list(refused.iterdir()) == []

    # The same fpm is refused by the builders for the same reason, with the point and the rank as its place.
    bad_fpm = _group_fpm(payloads[0], index, rank)
    with pytest.raises(error) as at_the_measurement:
        artifact.build_measurement(points[index], {rank: bad_fpm})
    assert type(at_the_measurement.value) is error
    assert str(at_the_measurement.value) == f"benchmark_id={index + 1} rank={rank}: " + reason.format(old=old, new=new)


@pytest.mark.parametrize("stored", [FAKE_STEPS, [], None], ids=["six-steps", "no-steps", "none"])
@pytest.mark.parametrize(("dp", "rank"), [(1, 0), (2, 0), (2, 1)])
def test_a_complete_prefill_artifact_whose_fpm_was_given_decode_steps_is_refused(tmp_path, dp, rank, stored):
    """A prefill fpm has no decode steps: the key is refused in a complete artifact as it is by ``build_fpm``."""
    cell = _cell(phase="prefill", dp=dp, strategy="tp")
    points = _prefill_points()
    payloads = _rank_artifacts("prefill", points, dp=dp)
    _edit_in_every_rank_file(payloads, 3, rank, lambda fpm: fpm.update({STEPS_KEY: copy.deepcopy(stored)}))

    # Both checkers ignore an fpm key they do not know.
    forced = tmp_path / "forced"
    pod = _pod(forced, cell)
    rank_paths = [artifact.rank_output_path(str(pod / "benchmark.json"), file_rank) for file_rank in range(dp)]
    for path, payload in zip(rank_paths, payloads, strict=True):
        artifact.atomic_write_json(path, payload)
    assert [measurement.point for measurement in _validate(cell, forced).points] == points
    assert _gate(rank_paths, "prefill").returncode == GATE_MERGED_PENDING

    refused = tmp_path / "refused"
    refused.mkdir()
    message = (
        rf"^rank 0: iteration_groups\[3\]\.rank_results\[{rank}\]: "
        r"a prefill fpm may not contain 'sglbench_decode_steps'$"
    )
    with pytest.raises(ValueError, match=message):
        artifact.write_artifacts(str(refused / "benchmark.json"), payloads)
    with pytest.raises(ValueError, match=message):
        artifact.build_merged_artifact(payloads, [f"/results/benchmark_dp{file_rank}.json" for file_rank in range(dp)])
    assert list(refused.iterdir()) == []


def test_the_steps_of_a_complete_artifact_are_checked_in_every_group_for_every_rank_with_the_rows_point_type():
    # agg: prefill rows and decode rows in one artifact. Each fpm is checked with the point type of its own row.
    agg_points = _prefill_points() + [
        {**point, "benchmark_id": point["benchmark_id"] + 4} for point in _decode_points()
    ]
    paths = ["/results/benchmark.json"]
    assert artifact.build_merged_artifact(_rank_artifacts("agg", agg_points), paths)["status"] == "complete"
    for index in range(len(agg_points)):
        (payload,) = _rank_artifacts("agg", agg_points)
        if agg_points[index]["point_type"] == "prefill":
            _group_fpm(payload, index, 0)[STEPS_KEY] = list(FAKE_STEPS)
            reason = "a prefill fpm may not contain 'sglbench_decode_steps'"
        else:
            del _group_fpm(payload, index, 0)[STEPS_KEY]
            reason = "a decode fpm must carry its 6 raw step times under 'sglbench_decode_steps'"
        with pytest.raises(ValueError) as refused:
            artifact.build_merged_artifact([payload], paths)
        assert str(refused.value) == f"rank 0: iteration_groups[{index}].rank_results[0]: {reason}"

    # The later ranks of a run (what a second node would merge): the entry of every rank of the run is checked,
    # also of a rank whose file is not among those passed, and the refusal names the first file passed.
    paths = ["/results/benchmark_dp2.json", "/results/benchmark_dp3.json"]
    for index, rank in itertools.product(range(3), range(4)):
        ranks = _rank_artifacts("decode", _decode_points(), dp=4)[2:]
        _edit_in_every_rank_file(ranks, index, rank, lambda fpm: fpm.update({STEPS_KEY: fpm[STEPS_KEY][:5]}))
        with pytest.raises(ValueError) as refused:
            artifact.build_merged_artifact(ranks, paths)
        assert str(refused.value) == (
            f"rank 2: iteration_groups[{index}].rank_results[{rank}]: "
            "fpm sglbench_decode_steps: a decode run has exactly 6 decode steps, got 5"
        )

    # The row's point decides the type, not the group's. A row that was given a decode point (its group still
    # holds the prefill point) has an fpm without raw steps.
    (payload,) = _rank_artifacts("prefill", _prefill_points())
    payload["results"][1].update(point=_point("decode", 2, 2, 0, 8, ["kvwarm_real_kv"]), kv_seed_regime="real_kv")
    assert payload["iteration_groups"][1]["point"]["point_type"] == "prefill"
    with pytest.raises(ValueError) as refused:
        artifact.build_merged_artifact([payload], ["/results/benchmark.json"])
    assert str(refused.value) == (
        "rank 0: iteration_groups[1].rank_results[0]: "
        "a decode fpm must carry its 6 raw step times under 'sglbench_decode_steps'"
    )


@pytest.mark.parametrize("status", ["partial", "failed"])
def test_the_steps_of_a_partial_or_failed_artifact_are_not_checked_again_when_it_is_written(tmp_path, status):
    """Both checkers reject such an artifact whatever it holds; the module checks its fpms when it is built."""
    points = _decode_points()
    extra = {
        "partial": {"completed": 2, "stop_reason": "timeout"},
        "failed": {"completed": 2, "failure": {"point": points[2], "reason": "fake reason"}},
    }[status]
    payloads = _rank_artifacts("decode", points, dp=2, **extra)
    for payload in payloads:
        del _group_fpm(payload, 1, 0)[STEPS_KEY]
        assert payload["status"] == status

    written = artifact.write_artifacts(str(tmp_path / "benchmark.json"), payloads)
    on_disk = json.loads(Path(written["rank_files"][0]).read_text())
    assert STEPS_KEY not in on_disk["results"][1]["fpms"][0]
    assert STEPS_KEY in on_disk["results"][0]["fpms"][0]


def test_the_artifact_shares_no_object_with_the_callers_inputs():
    """Editing what was handed to the builders, after the build, must not change what will be written."""
    kwargs = _build_kwargs("decode", _decode_points(), dp=2)
    config = {"fake_caps": {"decode": [1, 2]}}
    producer = {"fake_limits": {"page_size": 64}, "fake_notes": ["a"]}
    payloads = [
        artifact.build_rank_artifact(**{**kwargs, "dp_rank": rank, "config": config, "producer": producer})
        for rank in (0, 1)
    ]
    before = copy.deepcopy(payloads)

    config["fake_caps"]["decode"].append(3)
    config["fake_caps"]["prefill"] = [4]
    producer["fake_limits"]["page_size"] = 1
    producer["fake_notes"].append("b")
    for measurement in kwargs["measurements"]:
        measurement["point"]["batch_size"] = 99
        measurement["point"]["sample_reasons"].append("edited")
        for fpm in measurement["rank_fpms"].values():
            fpm["wall_time"] = 0.5
            fpm["worker_id"] = "edited"
            fpm[STEPS_KEY].append(9.0)
            fpm["scheduled_requests"]["num_decode_requests"] = 99
    assert payloads == before

    # The merged file holds its own copies too, of the envelope and of every rank's fpm.
    merged = artifact.build_merged_artifact(payloads, ["/results/benchmark.json", "/results/benchmark_dp1.json"])
    merged_before = copy.deepcopy(merged)
    for payload in payloads:
        payload["config"]["fake_caps"]["decode"].append(5)
        for row in payload["results"]:
            row["point"]["batch_size"] = 98
            row["fpms"][0]["wall_time"] = 0.25
            row["fpms"][0][STEPS_KEY].append(8.0)
    assert merged == merged_before
    # ... and in the other direction: editing the merged rows leaves the rank artifacts alone.
    payloads = copy.deepcopy(before)
    merged = artifact.build_merged_artifact(payloads, ["/results/benchmark.json", "/results/benchmark_dp1.json"])
    for row in merged["results"]:
        for fpm in row["fpms"]:
            fpm["wall_time"] = 0.125
            fpm["scheduled_requests"]["num_decode_requests"] = 97
    assert payloads == before


def test_write_artifacts_refuses_incomplete_or_unserialisable_input_with_the_disk_untouched(tmp_path):
    output_path = str(tmp_path / "benchmark.json")
    complete = _two_ranks()
    partial = _rank_artifacts("decode", _decode_points(), dp=2, completed=1, stop_reason="timeout")

    with pytest.raises(ValueError, match="needs the artifacts of ranks"):
        artifact.write_artifacts(output_path, complete[:1])  # rank 0 of a two-rank run
    with pytest.raises(ValueError, match="needs the artifacts of ranks"):
        artifact.write_artifacts(output_path, complete[1:])  # rank 1 alone
    with pytest.raises(ValueError, match="disagree on status"):
        artifact.write_artifacts(output_path, [complete[0], partial[1]])
    with pytest.raises(ValueError, match="no rank artifact"):
        artifact.write_artifacts(output_path, [])
    with pytest.raises(TypeError):
        artifact.write_artifacts(output_path, {"0": complete[0]})
    with pytest.raises(TypeError):
        artifact.write_artifacts(output_path, [complete[0], None])

    # A value JSON cannot hold, in the LAST rank, whose file is the first one written: nothing may be on disk
    # when it is found. (The test after this one puts the value into the files that are written later.)
    for bad in (float("inf"), float("nan"), {1, 2}, b"bytes", {(1, 2): 3}):
        ranks = _two_ranks()
        ranks[1]["producer"]["fake_value"] = bad
        with pytest.raises((ValueError, TypeError)):
            artifact.write_artifacts(output_path, ranks)
    failed = _rank_artifacts("decode", _decode_points(), dp=2, completed=0, failure={"point": None, "reason": "x"})
    failed[1]["producer"]["fake_value"] = float("nan")
    with pytest.raises(ValueError):
        artifact.write_artifacts(output_path, failed)

    for bad_path in (output_path + "/", "", None, tmp_path / "benchmark.json"):
        with pytest.raises((ValueError, TypeError)):
            artifact.write_artifacts(bad_path, _two_ranks())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("status", ["complete", "partial", "failed"])
@pytest.mark.parametrize("bad_rank", [0, 1, 2])
@pytest.mark.parametrize("bad", [float("nan"), {1, 2}, {(1, 2): 3}], ids=["nan", "set", "tuple-key"])
def test_every_document_is_serialised_before_the_first_file_is_written(tmp_path, bad, bad_rank, status):
    """The files are written highest rank first, so a value JSON cannot hold in a lower rank is met late.

    If a document were serialised only when its file is written, the files of the higher ranks would already
    be on disk when the value is found in rank 1 or in rank 0.
    """
    extra = {
        "complete": {},
        "partial": {"completed": 1, "stop_reason": "timeout"},
        "failed": {"completed": 1, "failure": {"point": None, "reason": "fake failure"}},
    }[status]
    sound = tmp_path / "sound"
    sound.mkdir()
    written = artifact.write_artifacts(
        str(sound / "benchmark.json"), _rank_artifacts("decode", _decode_points(), dp=3, **extra)
    )
    assert len(written["rank_files"]) == 3  # without the bad value the same run is written

    ranks = _rank_artifacts("decode", _decode_points(), dp=3, **extra)
    ranks[bad_rank]["producer"]["fake_value"] = bad
    refused = tmp_path / "refused"
    refused.mkdir()
    with pytest.raises(ValueError if isinstance(bad, float) else TypeError):
        artifact.write_artifacts(str(refused / "benchmark.json"), ranks)
    assert list(refused.iterdir()) == []


def test_merged_file_needs_one_distinct_path_per_rank_and_a_run_that_did_not_fail():
    ranks = _two_ranks()
    with pytest.raises(ValueError, match="rank paths"):
        artifact.build_merged_artifact(ranks, ["/results/benchmark.json"])
    with pytest.raises(ValueError, match="not distinct"):
        artifact.build_merged_artifact(ranks, ["/results/benchmark.json", "/results//benchmark.json"])
    with pytest.raises(ValueError, match="must name a file"):
        artifact.build_merged_artifact(ranks, ["/results/benchmark.json", "/results/benchmark_dp1.json/"])
    with pytest.raises(TypeError):
        artifact.build_merged_artifact(ranks, "/results/benchmark.json")
    failed = _rank_artifacts("decode", _decode_points(), dp=2, completed=0, failure={"point": None, "reason": "x"})
    with pytest.raises(ValueError, match="failed run has no merged file"):
        artifact.build_merged_artifact(failed, ["/results/benchmark.json", "/results/benchmark_dp1.json"])


# ---------------------------------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------------------------------

PATH_VECTORS = [
    "/results/benchmark.json",
    "/results/run.v1/metrics.final.json",  # dotted directory, several dots
    "/results/benchmark.results.v2.json",
    "/results/plain-noext",
    "/results/.hidden",
    "/results/sub dir/benchmark results.json",
    "/results//benchmark.json",  # doubled slash
    "//results/benchmark.json",
    "results/benchmark.json",  # relative
    "benchmark.json",
    "/results/benchmark_dp1.json",
]


@pytest.mark.parametrize("dp_rank", [0, 1, 13])
@pytest.mark.parametrize("output_path", PATH_VECTORS)
def test_rank_output_path_equals_the_aisim_contract(output_path, dp_rank):
    assert artifact.rank_output_path(output_path, dp_rank) == fpm_benchmark_result_name(output_path, dp_rank)


def test_path_helpers_on_hand_computed_cases():
    assert artifact.rank_output_path("/results/benchmark.json", 0) == "/results/benchmark.json"
    assert artifact.rank_output_path("/results//benchmark.json", 0) == "/results//benchmark.json"  # unchanged
    assert artifact.rank_output_path("/results/benchmark.json", 1) == "/results/benchmark_dp1.json"
    assert artifact.rank_output_path("/results/run.v1/metrics.final.json", 12) == (
        "/results/run.v1/metrics.final_dp12.json"
    )
    assert artifact.rank_output_path("/results/plain-noext", 3) == "/results/plain-noext_dp3"

    assert artifact.merged_output_path("/results/benchmark.json") == "/results/benchmark_merged.json"
    assert artifact.merged_output_path("/results/benchmark_dp2.json") == "/results/benchmark_dp2_merged.json"
    assert artifact.merged_output_path("/results/run.v1/metrics.final.json") == (
        "/results/run.v1/metrics.final_merged.json"
    )
    assert artifact.merged_output_path("/results/plain-noext") == "/results/plain-noext_merged"
    assert artifact.merged_output_path("/results//benchmark.json") == "/results/benchmark_merged.json"


@pytest.mark.parametrize(
    "bad_path", ["/results/benchmark.json/", "/results/", "/", "", ".", None, 7, b"/results/b.json"]
)
def test_path_helpers_refuse_paths_without_a_file_name(bad_path):
    for dp_rank in (0, 1):
        with pytest.raises((ValueError, TypeError)):
            artifact.rank_output_path(bad_path, dp_rank)
    with pytest.raises((ValueError, TypeError)):
        artifact.merged_output_path(bad_path)


@pytest.mark.parametrize(
    ("dp_rank", "error"), [(-1, ValueError), (True, TypeError), (1.0, TypeError), ("1", TypeError)]
)
def test_rank_output_path_rejects_bad_ranks(dp_rank, error):
    with pytest.raises(error):
        artifact.rank_output_path("/results/benchmark.json", dp_rank)


@pytest.mark.parametrize(
    "relative_output",
    [
        "benchmark.json",
        "nested//benchmark.json",  # doubled slash: the gate's shell keeps it, pathlib collapses it
        "run.v1/metrics.final.json",  # dotted directory and several dots
        "plain-noext",
        "sub/../benchmark.json",  # a ".." segment: pathlib keeps it, and so must the recorded strings
    ],
)
def test_files_land_where_the_gates_own_naming_looks_for_them(tmp_path, relative_output):
    """End to end: the gate's shell naming gives the argv, the module writes, the gate's checker accepts.

    Exit 0 proves three things at once: the rank files are at the shell-derived paths, the merged file is where
    the checker looks, and the path strings recorded inside it equal the checker's ``str(pathlib.Path(argv))``.
    """
    output_path = f"{tmp_path}/{relative_output}"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    argv = [_gate_shell_rank_path(output_path, rank) for rank in (0, 1)]
    assert argv[0] == output_path

    written = artifact.write_artifacts(output_path, _rank_artifacts("prefill", _prefill_points(), dp=2))

    gate = _gate(argv, "prefill")
    assert gate.returncode == GATE_ACCEPT, gate.stderr
    assert written["rank_files"] == [str(Path(path)) for path in argv]
    merged = json.loads(Path(written["merged_output_path"]).read_text())
    assert merged["rank_files"] == written["rank_files"]
    assert merged["merged_output_path"] == written["merged_output_path"]


def test_recorded_paths_keep_a_parent_segment():
    """``..`` is not folded away: the gate compares with ``str(pathlib.Path(argv))``, which keeps it."""
    paths = ["/results/sub/../benchmark.json", "/results/sub/../benchmark_dp1.json"]
    assert [str(Path(path)) for path in paths] == paths  # the gate's form of these two paths
    merged = artifact.build_merged_artifact(_two_ranks(), paths)
    assert merged["rank_files"] == paths
    assert merged["merged_output_path"] == "/results/sub/../benchmark_merged.json"
    assert artifact.rank_output_path("/results/sub/../benchmark.json", 1) == "/results/sub/../benchmark_dp1.json"


def test_recorded_paths_are_not_resolved(tmp_path):
    real = tmp_path / "real-results"
    real.mkdir()
    link = tmp_path / "results-link"
    link.symlink_to(real, target_is_directory=True)
    output_path = str(link / "benchmark.json")

    written = artifact.write_artifacts(output_path, _rank_artifacts("decode", _decode_points(), dp=2))

    # The strings keep the symlink's name, although the files live in the real directory.
    assert written == {
        "rank_files": [str(link / "benchmark.json"), str(link / "benchmark_dp1.json")],
        "merged_output_path": str(link / "benchmark_merged.json"),
    }
    merged = json.loads((real / "benchmark_merged.json").read_text())
    assert merged["rank_files"] == written["rank_files"]
    assert merged["merged_output_path"] == written["merged_output_path"]
    assert sorted(path.name for path in real.iterdir()) == [
        "benchmark.json",
        "benchmark_dp1.json",
        "benchmark_merged.json",
    ]
    gate = _gate([output_path, str(link / "benchmark_dp1.json")], "decode")
    assert gate.returncode == GATE_ACCEPT, gate.stderr


# ---------------------------------------------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------------------------------------------


def _spy_on_replace(monkeypatch) -> list[tuple[str, object, object]]:
    """Record, at the moment of each rename: temporary name, temporary content, target content before it."""
    seen = []
    real_replace = os.replace

    def spying_replace(source, destination):
        target = Path(destination)
        before = json.loads(target.read_text()) if target.exists() else None
        seen.append((Path(source).name, json.loads(Path(source).read_text()), before))
        real_replace(source, destination)

    monkeypatch.setattr(artifact.os, "replace", spying_replace)
    return seen


def test_atomic_write_replaces_an_existing_target_in_one_rename(tmp_path, monkeypatch):
    target = tmp_path / "benchmark.json"
    target.write_text('{"old": true}')
    seen = _spy_on_replace(monkeypatch)

    artifact.atomic_write_json(str(target), {"new": 1})
    artifact.atomic_write_json(target, {"new": 2})  # a path object is accepted too

    assert json.loads(target.read_text()) == {"new": 2}
    # At each rename the old target was still whole and the temporary already held the whole new document.
    assert [(content, before) for _name, content, before in seen] == [
        ({"new": 1}, {"old": True}),
        ({"new": 2}, {"new": 1}),
    ]
    names = [name for name, _content, _before in seen]
    assert len(set(names)) == 2, "two writes must not share a temporary name"
    for name in names:
        assert name.startswith(".benchmark.json.") and name.endswith(".tmp")
        assert not fnmatch.fnmatch(name, FPM_BENCHMARK_RESULT_GLOB)
        # The whole name: a dot, the target's name, this process's pid, 16 hex characters of randomness, ".tmp".
        assert re.fullmatch(rf"\.benchmark\.json\.{os.getpid()}\.[0-9a-f]{{16}}\.tmp", name), name
    # Nothing but the target is left, and the result sweep sees only the target.
    assert [path.name for path in tmp_path.iterdir()] == ["benchmark.json"]
    assert list(tmp_path.glob(f"**/{FPM_BENCHMARK_RESULT_GLOB}")) == [target]
    assert list(tmp_path.glob("**/*.tmp")) == []


@pytest.mark.parametrize("with_old_target", [False, True])
@pytest.mark.parametrize(
    "payload",
    [
        {"x": float("nan")},
        {"x": [1.0, float("inf")]},
        {"x": float("-inf")},
        {"x": object()},
        {"x": {1, 2}},
        {"x": {(1, 2): 3}},  # a dict key json cannot write: it is refused, not left out
    ],
    ids=["nan", "inf", "minus-inf", "object", "set", "tuple-key"],
)
def test_a_failing_dump_leaves_no_partial_target_and_no_temporary(tmp_path, payload, with_old_target):
    target = tmp_path / "benchmark.json"
    if with_old_target:
        target.write_text('{"old": true}')

    with pytest.raises((ValueError, TypeError)):
        artifact.atomic_write_json(str(target), payload)

    if with_old_target:
        assert [path.name for path in tmp_path.iterdir()] == ["benchmark.json"]
        assert target.read_text() == '{"old": true}'
    else:
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failing_call", ["fsync", "replace"])
def test_a_failing_write_removes_its_temporary_and_keeps_the_old_target(tmp_path, monkeypatch, failing_call):
    target = tmp_path / "benchmark.json"
    target.write_text('{"old": true}')
    created = []

    def fail(*args):
        created.extend(path.name for path in tmp_path.iterdir() if path.name != "benchmark.json")
        raise OSError("fake disk failure")

    monkeypatch.setattr(artifact.os, failing_call, fail)
    with pytest.raises(OSError, match="fake disk failure"):
        artifact.atomic_write_json(str(target), {"new": 1})
    monkeypatch.undo()

    assert len(created) == 1 and created[0].endswith(".tmp"), "the failure must hit after the temporary exists"
    assert [path.name for path in tmp_path.iterdir()] == ["benchmark.json"]
    assert target.read_text() == '{"old": true}'


@pytest.mark.parametrize("with_old_target", [False, True])
@pytest.mark.parametrize("failing_call", ["fsync", "replace"])
def test_an_interrupted_write_removes_its_temporary_too(tmp_path, monkeypatch, failing_call, with_old_target):
    """KeyboardInterrupt is what a signal during the write looks like, and it is not an OSError."""
    target = tmp_path / "benchmark.json"
    if with_old_target:
        target.write_text('{"old": true}')
    created = []

    def interrupt(*args):
        created.extend(path.name for path in tmp_path.iterdir() if path.name != "benchmark.json")
        raise KeyboardInterrupt("fake interrupt")

    monkeypatch.setattr(artifact.os, failing_call, interrupt)
    with pytest.raises(KeyboardInterrupt, match="fake interrupt"):
        artifact.atomic_write_json(str(target), {"new": 1})
    monkeypatch.undo()

    assert len(created) == 1 and created[0].endswith(".tmp"), "the interrupt must hit after the temporary exists"
    if with_old_target:
        assert [path.name for path in tmp_path.iterdir()] == ["benchmark.json"]
        assert target.read_text() == '{"old": true}'
    else:
        assert list(tmp_path.iterdir()) == []


def test_the_temporary_already_holds_every_byte_when_it_is_synced(tmp_path, monkeypatch):
    """fsync before flush would sync an empty file: a small document is still in Python's buffers then."""
    sizes_at_fsync = []
    real_fsync = os.fsync

    def recording_fsync(descriptor):
        sizes_at_fsync.append(os.fstat(descriptor).st_size)
        real_fsync(descriptor)

    monkeypatch.setattr(artifact.os, "fsync", recording_fsync)

    target = tmp_path / "benchmark.json"
    artifact.atomic_write_json(str(target), {"fake": [1, 2, 3]})
    assert sizes_at_fsync == [len(target.read_bytes())]
    assert target.read_bytes() == b'{"fake": [1, 2, 3]}'

    # The rank file of a run that failed before its first point is as small as a real file gets.
    sizes_at_fsync.clear()
    failed = _rank_artifacts("decode", _decode_points(), completed=0, failure={"point": None, "reason": "fake"})
    written = artifact.write_artifacts(str(tmp_path / "failed.json"), failed)
    (rank_file,) = written["rank_files"]
    size = len(Path(rank_file).read_bytes())
    assert 0 < size < 4096, "the document must stay inside Python's write buffers for this test to bite"
    assert sizes_at_fsync == [size]


def test_a_temporary_left_by_a_killed_process_is_neither_reused_nor_removed(tmp_path):
    # A temporary named by pid alone would collide here: a container restart can repeat the pid, so a file of
    # that name may be left over from a writer that was killed. The random token in the name avoids it.
    stale = tmp_path / f".benchmark.json.{os.getpid()}.tmp"
    stale.write_text("left behind by a killed process")

    artifact.atomic_write_json(str(tmp_path / "benchmark.json"), {"new": 1})

    assert json.loads((tmp_path / "benchmark.json").read_text()) == {"new": 1}
    assert stale.read_text() == "left behind by a killed process"
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted([stale.name, "benchmark.json"])
    # A stale temporary beside the results does not disturb the result sweep.
    assert [path.name for path in tmp_path.glob(f"**/{FPM_BENCHMARK_RESULT_GLOB}")] == ["benchmark.json"]


def test_an_existing_file_under_the_temporary_name_is_neither_written_into_nor_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact.secrets, "token_hex", lambda size: "fixedtoken")  # make the name predictable
    occupied = tmp_path / f".benchmark.json.{os.getpid()}.fixedtoken.tmp"
    occupied.write_text("not this call's file")
    target = tmp_path / "benchmark.json"
    target.write_text('{"old": true}')

    with pytest.raises(FileExistsError):
        artifact.atomic_write_json(str(target), {"new": 1})

    # Exclusive create: the existing file is not truncated; and cleanup removes only what this call created.
    assert occupied.read_text() == "not this call's file"
    assert target.read_text() == '{"old": true}'


def test_a_leftover_temporary_is_invisible_to_every_result_reader(tmp_path):
    cell = _cell(phase="decode", dp=2, strategy="tp")
    pod = _pod(tmp_path, cell)
    points = _decode_points()
    written = artifact.write_artifacts(str(pod / "benchmark.json"), _rank_artifacts("decode", points, dp=2))
    # What a writer killed in mid-write leaves behind: half a document under the temporary's name.
    (pod / f".benchmark.json.{os.getpid()}.0123456789abcdef.tmp").write_text('{"schema_version": 2, "resul')

    assert len(_validate(cell, tmp_path).points) == len(points)
    # The timing summary parses every "benchmark*.json" under the raw directory, merged file included.
    group_times = [max(_expected_wall_time(point, rank) for rank in (0, 1)) for point in points]
    assert _runtime_timing_summary(tmp_path / "raw") == {
        "runtime_rank_count": 2,
        "benchmark_elapsed_seconds": FAKE_ELAPSED_SECONDS + 1,
        "measured_iteration_seconds": pytest.approx(math.fsum(group_times), rel=1e-12),
    }
    assert _gate(written["rank_files"], "decode").returncode == GATE_ACCEPT


def test_atomic_write_needs_an_existing_directory_and_a_file_name(tmp_path):
    with pytest.raises(FileNotFoundError):
        artifact.atomic_write_json(str(tmp_path / "missing" / "benchmark.json"), {"a": 1})
    assert not (tmp_path / "missing").exists()
    with pytest.raises(ValueError, match="must name a file"):
        artifact.atomic_write_json(f"{tmp_path}/", {"a": 1})
    with pytest.raises(TypeError):
        artifact.atomic_write_json(os.fsencode(tmp_path / "benchmark.json"), {"a": 1})
    assert list(tmp_path.iterdir()) == []


def test_written_files_are_ascii_json_readable_under_any_encoding(tmp_path):
    payload = {"producer": {"note": "caf" + chr(0xE9) + " " + chr(0x6D4B) + chr(0x8BD5)}}  # not ASCII
    artifact.atomic_write_json(str(tmp_path / "benchmark.json"), payload)
    raw = (tmp_path / "benchmark.json").read_bytes()
    assert raw.isascii()
    assert json.loads(raw.decode("ascii")) == payload


def test_written_rank_files_round_trip_exactly(tmp_path):
    payloads = _rank_artifacts("decode", _decode_points(), dp=2, producer={"limits": {"page_size": 64}})
    written = artifact.write_artifacts(str(tmp_path / "benchmark.json"), payloads)
    for path, payload in zip(written["rank_files"], payloads, strict=True):
        assert json.loads(Path(path).read_text(encoding="utf-8")) == payload
    merged = json.loads(Path(written["merged_output_path"]).read_text(encoding="utf-8"))
    assert merged == artifact.build_merged_artifact(payloads, written["rank_files"])


# ---------------------------------------------------------------------------------------------------------------
# Corners of the key rule, of the step re-check and of a run that is not complete
# ---------------------------------------------------------------------------------------------------------------

_Pair = collections.namedtuple("_Pair", "first second")

NESTED_KEYS = {
    # name -> (evidence, the key the refusal names)
    "counter": ({"fake": collections.Counter([1, 1])}, 1),  # a dict subclass: json writes it as an object
    "ordered-dict": ({"fake": [collections.OrderedDict({"a": {None: 1}})]}, None),
    "named-tuple": ({"fake": _Pair(1, {2: "x"})}, 2),  # a tuple subclass: json writes it as a list
    "six-containers-down": ({"l1": [({"l3": [[{1: "x"}]]},)]}, 1),
    "twelve-containers-down": ({"a": [[[[[[[[[[{"b": {True: "x"}}]]]]]]]]]]}, True),
    "decimal-key": ({"fake": {decimal.Decimal("1.5"): 1}}, decimal.Decimal("1.5")),  # repr and str differ
    "nan-key": ({"fake": {float("nan"): 1}}, float("nan")),  # json would refuse it itself, as a ValueError
}


@pytest.mark.parametrize("name", sorted(NESTED_KEYS))
def test_the_key_rule_searches_subclasses_of_the_json_containers_at_any_depth_and_names_the_key_by_its_repr(name):
    evidence, key = NESTED_KEYS[name]
    reason = f"is not JSON data: dict keys must be str at every depth, got {key!r}"
    with pytest.raises(TypeError) as refused:
        artifact.build_fpm(_point("prefill", 1, 2, 16, 0), 0, 0.01, extra=evidence)
    assert str(refused.value) == f"extra {reason}"
    kwargs = _build_kwargs("decode", _decode_points())
    for argument in ("config", "producer"):
        with pytest.raises(TypeError) as refused:
            artifact.build_rank_artifact(**kwargs, **{argument: evidence})
        assert str(refused.value) == f"{argument} {reason}"


def test_evidence_that_holds_one_container_in_two_places_is_accepted():
    """Shared is not cyclic: json writes the container twice."""
    limits = {"fake_max_tokens": 4096, "fake_sizes": [1, 2]}
    evidence = {"fake_grid_limits": limits, "fake_engine_limits": limits, "fake_list": [limits, (limits,)]}
    evidence.update(fake_empty_a=(), fake_empty_b=())  # the empty tuple is one object in CPython
    fpm = artifact.build_fpm(_point("prefill", 1, 2, 16, 0), 0, 0.01, extra=evidence)
    assert (fpm["fake_grid_limits"], fpm["fake_engine_limits"]) == (limits, limits)
    payload = artifact.build_rank_artifact(
        **_build_kwargs("decode", _decode_points()), config=evidence, producer=evidence
    )
    assert payload["status"] == "complete"
    assert payload["producer"]["fake_list"] == [limits, (limits,)]


# FAKE whole and half seconds per rank: the upper median of steps 3 to 5 is 2.0 on rank 0 and 1.0 on rank 1.
FAKE_WHOLE_STEPS = {0: [0.5, 1.0, 2.0, 1.0, 3.0, 4.0], 1: [0.5, 1.0, 1.0, 1.0, 3.0, 4.0]}


def _whole_second_ranks(dp: int) -> list[dict]:
    measurements = []
    for point in _decode_points():
        fpms = {
            rank: artifact.build_fpm(
                point,
                rank,
                artifact.decode_wall_time(FAKE_WHOLE_STEPS[rank]),
                extra={STEPS_KEY: list(FAKE_WHOLE_STEPS[rank])},
            )
            for rank in range(dp)
        }
        measurements.append(artifact.build_measurement(point, fpms))
    kwargs = {**_build_kwargs("decode", _decode_points(), dp=dp), "measurements": measurements}
    kwargs.pop("dp_rank")
    return [artifact.build_rank_artifact(dp_rank=rank, **kwargs) for rank in range(dp)]


def _whole_steps_as_ints(fpm: dict) -> None:
    fpm[STEPS_KEY] = [int(step) if step == int(step) else step for step in fpm[STEPS_KEY]]


def test_the_steps_are_checked_in_every_rank_file_and_in_the_groups_own_fpm_object():
    """``1 == 1.0``: an int step for a whole second leaves every comparison of the envelope true."""
    reason = "fpm sglbench_decode_steps: raw_step_seconds[1] must be a float in seconds, got 1"
    paths = ["/results/benchmark.json", "/results/benchmark_dp1.json"]

    # Only in the file of rank 1: the group entry of rank 0 holds ints. The two files still compare equal.
    ranks = _whole_second_ranks(2)
    assert artifact.build_merged_artifact(ranks, paths)["status"] == "complete"
    _whole_steps_as_ints(_group_fpm(ranks[1], 0, 0))
    assert ranks[0]["iteration_groups"] == ranks[1]["iteration_groups"]
    with pytest.raises(TypeError) as refused:
        artifact.build_merged_artifact(ranks, paths)
    assert str(refused.value) == f"rank 1: iteration_groups[0].rank_results[0]: {reason}"

    # A row that no longer shares its group entry (as after a JSON round trip): the group's fpm holds the ints.
    (payload,) = _whole_second_ranks(1)
    payload["results"][0]["fpms"] = [copy.deepcopy(payload["results"][0]["fpms"][0])]
    fpm = _group_fpm(payload, 0, 0)
    _whole_steps_as_ints(fpm)
    assert payload["results"][0]["fpms"][0] == fpm
    with pytest.raises(TypeError) as refused:
        artifact.build_merged_artifact([payload], paths[:1])
    assert str(refused.value) == f"rank 0: iteration_groups[0].rank_results[0]: {reason}"


def _row_steps_as_ints(fpm: dict) -> None:
    _whole_steps_as_ints(fpm)


def _row_steps_as_bools(fpm: dict) -> None:
    fpm[STEPS_KEY] = [True if step == 1.0 else step for step in fpm[STEPS_KEY]]


def _row_time_as_an_int(fpm: dict) -> None:
    fpm["wall_time"] = int(fpm["wall_time"])


ROW_COPY_EDITS = {
    # name -> (edit of the row's own copy of its fpm, exception, the text after "rank R: results[1].fpms[0]")
    "int-steps": (_row_steps_as_ints, TypeError, ": fpm sglbench_decode_steps: raw_step_seconds[1] must be a float"),
    "bool-steps": (_row_steps_as_bools, TypeError, ": fpm sglbench_decode_steps: raw_step_seconds[1] must be a float"),
    "int-time": (_row_time_as_an_int, ValueError, " has no float fpm wall_time"),
}


@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("name", sorted(ROW_COPY_EDITS))
def test_a_rows_own_copy_of_its_fpm_is_held_to_the_types_of_its_group_entry(tmp_path, name, dp):
    """A row that is not the same object as its group entry, with a whole-number time or step as int or bool.

    The copy still equals the entry, the group times and the sum, so the envelope rules hold; both checkers
    would accept the file.
    """
    edit, exception, reason = ROW_COPY_EDITS[name]
    ranks = _whole_second_ranks(dp)
    rank = dp - 1
    row = ranks[rank]["results"][1]
    row["fpms"] = [copy.deepcopy(row["fpms"][0])]
    assert row["fpms"][0] is not _group_fpm(ranks[rank], 1, rank)
    (tmp_path / "sound").mkdir()
    (tmp_path / "edited").mkdir()
    # The copy alone is sound: the file is written.
    assert artifact.write_artifacts(str(tmp_path / "sound" / "benchmark.json"), ranks)["merged_output_path"]
    edit(row["fpms"][0])
    assert row["fpms"][0] == _group_fpm(ranks[rank], 1, rank)  # equal, and of another type

    paths = ["/results/benchmark.json", "/results/benchmark_dp1.json"][:dp]
    for call in (
        lambda: artifact.build_merged_artifact(ranks, paths),
        lambda: artifact.write_artifacts(str(tmp_path / "edited" / "benchmark.json"), ranks),
    ):
        with pytest.raises(exception) as refused:
            call()
        assert str(refused.value).startswith(f"rank {rank}: results[1].fpms[0]{reason}")
    assert list((tmp_path / "edited").iterdir()) == []


def test_the_place_of_a_step_fault_is_the_position_of_the_group_not_its_benchmark_id():
    (payload,) = _rank_artifacts("decode", _decode_points())
    payload["iteration_groups"][1]["benchmark_id"] = 7
    del _group_fpm(payload, 1, 0)[STEPS_KEY]
    with pytest.raises(ValueError) as refused:
        artifact.build_merged_artifact([payload], ["/results/benchmark.json"])
    assert str(refused.value) == (
        "rank 0: iteration_groups[1].rank_results[0]: "
        "a decode fpm must carry its 6 raw step times under 'sglbench_decode_steps'"
    )


def test_of_two_faults_the_step_fault_of_an_earlier_row_and_of_a_lower_rank_entry_is_the_one_reported():
    """The order the docstring of ``_check_complete_payload`` gives."""
    missing = "a decode fpm must carry its 6 raw step times under 'sglbench_decode_steps'"
    paths = ["/results/benchmark.json", "/results/benchmark_dp1.json"]
    # steps removed in row 0, a wrong regime in row 1
    (payload,) = _rank_artifacts("decode", _decode_points())
    del _group_fpm(payload, 0, 0)[STEPS_KEY]
    payload["results"][1]["kv_seed_regime"] = "real_prefix"
    with pytest.raises(ValueError) as refused:
        artifact.build_merged_artifact([payload], paths[:1])
    assert str(refused.value) == f"rank 0: iteration_groups[0].rank_results[0]: {missing}"
    # steps removed in row 0, a wrong measured sum
    (payload,) = _rank_artifacts("decode", _decode_points())
    del _group_fpm(payload, 0, 0)[STEPS_KEY]
    payload["timing"]["measured_iteration_seconds"] = 1.0
    with pytest.raises(ValueError) as refused:
        artifact.build_merged_artifact([payload], paths[:1])
    assert str(refused.value) == f"rank 0: iteration_groups[0].rank_results[0]: {missing}"
    # steps removed in both rank entries of one group
    ranks = _rank_artifacts("decode", _decode_points(), dp=2)
    for payload in ranks:
        del _group_fpm(payload, 0, 0)[STEPS_KEY]
        del _group_fpm(payload, 0, 1)[STEPS_KEY]
    with pytest.raises(ValueError) as refused:
        artifact.build_merged_artifact(ranks, paths)
    assert str(refused.value) == f"rank 0: iteration_groups[0].rank_results[0]: {missing}"


FAKE_CALLER_BLOCKS = {
    "config": {"fake_model": "fake-model", "fake_limits": {"fake_page_size": 16}},
    "producer": {"fake_grid_limits": {"fake_max_tokens": 4096}},
}

NOT_COMPLETE_RUNS = {
    # name -> (mode, numbers of completed points, keyword arguments for that number, status, stop_reason)
    "deadline": ("decode", (0, 1, 2), lambda points, done: {"stop_reason": "timeout"}, "partial", "timeout"),
    "failure-outside-a-point": (
        "decode",
        (0, 1, 2, 3),
        lambda points, done: {"failure": {"point": None, "reason": "fake engine exit"}},
        "failed",
        None,
    ),
    "failed-point": (
        "decode",
        (0, 1, 2),
        lambda points, done: {"failure": {"point": points[done], "reason": "fake reason"}},
        "failed",
        None,
    ),
    "failure-after-the-deadline": (
        "decode",
        (0, 1, 2),
        lambda points, done: {"failure": {"point": None, "reason": "fake engine exit"}, "stop_reason": "timeout"},
        "failed",
        "timeout",
    ),
    "failed-point-after-the-deadline": (
        "decode",
        (0, 1, 2),
        lambda points, done: {"failure": {"point": points[done], "reason": "fake reason"}, "stop_reason": "timeout"},
        "failed",
        "timeout",
    ),
    "missing-phase": ("agg", (3,), lambda points, done: {"missing_phases": ("prefill",)}, "failed", None),
    "missing-phase-after-the-deadline": (
        "agg",
        (3,),
        lambda points, done: {"missing_phases": ("prefill",), "stop_reason": "timeout"},
        "failed",
        "timeout",
    ),
}


@pytest.mark.parametrize("dp", [1, 2, 3])
@pytest.mark.parametrize("ending", sorted(NOT_COMPLETE_RUNS))
def test_whatever_ends_a_run_its_artifact_keeps_the_callers_blocks_and_the_rows_of_the_completed_points(ending, dp):
    """Neither checker reads a rejected artifact, so only this pins what a person finds in one."""
    mode, done_counts, arguments, status, stop_reason = NOT_COMPLETE_RUNS[ending]
    points = _decode_points()
    whole = _rank_artifacts("decode", points, dp=dp, **FAKE_CALLER_BLOCKS)
    assert [payload["status"] for payload in whole] == ["complete"] * dp
    for done in done_counts:
        payloads = _rank_artifacts(mode, points, dp=dp, completed=done, **FAKE_CALLER_BLOCKS, **arguments(points, done))
        fake_group_times = [max(_expected_wall_time(point, rank) for rank in range(dp)) for point in points[:done]]
        fake_sum = 0.0
        for fake_time in fake_group_times:
            fake_sum += fake_time
        for rank, payload in enumerate(payloads):
            assert (payload["status"], payload["stop_reason"]) == (status, stop_reason)
            assert payload["config"] == {"mode": mode, **FAKE_CALLER_BLOCKS["config"]}
            assert payload["producer"] == {
                "name": "aisim_sglbench",
                "artifact_format": 1,
                **FAKE_CALLER_BLOCKS["producer"],
            }
            assert payload["measurement_policy"] == whole[rank]["measurement_policy"] != {}
            assert ("kvwarm" in payload, payload.get("kvwarm")) == (True, EXPECTED_KVWARM)
            assert payload["results"] == whole[rank]["results"][:done]
            assert payload["iteration_groups"] == whole[rank]["iteration_groups"][:done]
            assert [group["wall_time"] for group in payload["iteration_groups"]] == fake_group_times
            assert [
                [entry["fpms"][0]["wall_time"] for entry in group["rank_results"]]
                for group in payload["iteration_groups"]
            ] == [[_expected_wall_time(point, entry_rank) for entry_rank in range(dp)] for point in points[:done]]
            assert [row["fpms"][0][STEPS_KEY] for row in payload["results"]] == [
                _fake_decode_steps(point["benchmark_id"], rank) for point in points[:done]
            ]
            assert payload["timing"] == {
                "benchmark_elapsed_seconds": FAKE_ELAPSED_SECONDS + rank,
                "measured_iteration_seconds": fake_sum,
            }
            assert payload["coverage"]["completed_points"] == done
            assert payload["dp"] == {"size": dp, "rank": rank}


@pytest.mark.parametrize("dp", [1, 3])
def test_a_failed_run_returns_its_rank_paths_in_the_recorded_form_and_writes_every_rank_file(tmp_path, dp):
    payloads = _rank_artifacts(
        "decode", _decode_points(), dp=dp, completed=1, failure={"point": None, "reason": "fake engine exit"}
    )
    (tmp_path / "sub").mkdir()
    written = artifact.write_artifacts(f"{tmp_path}/sub//benchmark.json", payloads)
    names = ["benchmark.json", "benchmark_dp1.json", "benchmark_dp2.json"][:dp]
    assert written == {"rank_files": [str(tmp_path / "sub" / name) for name in names], "merged_output_path": None}
    assert sorted(path.name for path in (tmp_path / "sub").iterdir()) == sorted(names)
    for path, payload in zip(written["rank_files"], payloads, strict=True):
        assert json.loads(Path(path).read_text(encoding="utf-8")) == payload


# ---------------------------------------------------------------------------------------------------------------
# The module is a standard-library leaf
# ---------------------------------------------------------------------------------------------------------------


def test_module_imports_with_the_standard_library_only():
    """Import by flat name with site-packages off: this is how the pod, which has no AISim, loads the file."""
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import sglbench_artifact; print(sglbench_artifact.SCHEMA_VERSION)"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(RUNTIME)], capture_output=True, text=True, timeout=60, check=False
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "2"


def test_module_source_has_no_foreign_import_and_reads_no_environment():
    source = (RUNTIME / "sglbench_artifact.py").read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative import: the pod loads the file by its flat name"
            imported.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Name):
            # A dynamic import is seen by no import statement; the pod would meet it only when the call runs.
            assert node.id != "__import__", "dynamic import in a pod module"
    assert imported <= set(sys.stdlib_module_names), sorted(imported - set(sys.stdlib_module_names))
    assert "importlib" not in imported, "dynamic import in a pod module"
    assert not any(name.startswith("sglbench") for name in imported)
    assert "environ" not in source and "getenv" not in source
    assert source.startswith(
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        "# The result envelope, its status rule, the point checks, the decode sample and the merged file are "
        "re-implemented\n"
        "# from ai-dynamo/dynamo (Apache-2.0, https://github.com/ai-dynamo/dynamo), components/src/dynamo/vllm/\n"
        "# instrumented_scheduler.py, benchmark_points.py and worker_factory.py and components/src/dynamo/common/\n"
        "# forward_pass_metrics.py at b20fe6fe94021b854597965c7305931175e8f482. See THIRD_PARTY_NOTICES.md.\n\n"
    )
    assert (
        "No upstream function or block of source is copied. What follows SGLang and Dynamo is named in the header "
        "of this\nfile and recorded in THIRD_PARTY_NOTICES.md. "
    ) in source
