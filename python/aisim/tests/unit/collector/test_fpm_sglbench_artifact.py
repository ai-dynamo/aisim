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

import copy
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
from collector.fpm_forward.native_artifact import select_native_measurements, validate_native_collection
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


# ---------------------------------------------------------------------------------------------------------------
# The dataset record of a run whose prompts were dataset text
# ---------------------------------------------------------------------------------------------------------------

# name -> (mode, points of the grid, keyword arguments that end the run, status)
DATASET_RUNS = {
    "decode-complete": ("decode", _decode_points, {}, "complete"),
    "decode-partial": ("decode", _decode_points, {"completed": 1, "stop_reason": "timeout"}, "partial"),
    "decode-failed": (
        "decode",
        _decode_points,
        {"completed": 1, "failure": {"point": None, "reason": "fake reason"}},
        "failed",
    ),
}


@pytest.mark.parametrize("name", sorted(DATASET_RUNS))
def test_the_kvwarm_block_carries_the_dataset_record_it_is_given_and_nothing_else_changes(name):
    mode, points, ending, status = DATASET_RUNS[name]
    dp = 2
    given = dict(FAKE_KVWARM_DATASET)

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


def _dataset(**changes) -> dict:
    return {**FAKE_KVWARM_DATASET, **changes}


def _dataset_without(key: str) -> dict:
    return {name: value for name, value in FAKE_KVWARM_DATASET.items() if name != key}


NOT_A_SHA256 = r"\['sha256'\] must be the 64 lower-case hexadecimal digits of a SHA-256, got "
EXACT_KEYS = r" must have exactly the keys \['path', 'sha256', 'collection_pool', 'conversations'\], got "

# name -> (what is given as the record, exception, the message after the place of the record)
BAD_KVWARM_DATASETS = {
    "a-path": ("/fake-cache/fpm_datasets/fake_dataset.json", TypeError, r" must be a dict, got str$"),
    "path-missing": (
        _dataset_without("path"),
        ValueError,
        EXACT_KEYS + r"\['collection_pool', 'conversations', 'sha256'\]$",
    ),
    # A fifth key, as the launcher's own record of the dataset has: the artifact takes Dynamo's four only.
    "a-fifth-key": (_dataset(url="https://example.invalid/fake_dataset.json"), ValueError, EXACT_KEYS),
    "sha256-one-digit-short": (_dataset(sha256="0123456789abcdef" * 3 + "0123456789abcde"), ValueError, NOT_A_SHA256),
    "conversations-zero": (_dataset(conversations=0), ValueError, r"\['conversations'\] must be >= 1, got 0$"),
    "conversations-a-bool": (_dataset(conversations=True), TypeError, r"\['conversations'\] must be an int, got True$"),
}


@pytest.mark.parametrize("name", sorted(BAD_KVWARM_DATASETS))
def test_a_bad_dataset_record_is_refused_when_the_artifact_is_built(name):
    record, error, message = BAD_KVWARM_DATASETS[name]
    kwargs = _build_kwargs("decode", _decode_points())
    # The baseline is sound, with the record and without it.
    for sound in (None, dict(FAKE_KVWARM_DATASET)):
        assert artifact.build_rank_artifact(**kwargs, kvwarm_dataset=sound)["status"] == "complete"
    with pytest.raises(error, match=rf"^kvwarm_dataset{message}"):
        artifact.build_rank_artifact(**kwargs, kvwarm_dataset=record)
    # A prefill artifact has no kvwarm block to put the record into: there a sound record is refused too.
    no_block = r"^kvwarm_dataset was given for a prefill artifact, which has no kvwarm block$"
    with pytest.raises(ValueError, match=no_block):
        artifact.build_rank_artifact(**_build_kwargs("prefill", _prefill_points()), kvwarm_dataset=FAKE_KVWARM_DATASET)


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

    # A failure given with a missing phase decides the error.
    agg = {**base, "mode": "agg", "measurements": [], "expected_points": 0}
    payload = artifact.build_rank_artifact(
        **agg, missing_phases=("decode",), failure={"point": None, "reason": "engine exited"}
    )
    assert (payload["error"], payload["missing_phases"]) == ("engine exited", ["decode"])
    # A missing phase reported after the deadline keeps both facts, as a failed point does.
    payload = artifact.build_rank_artifact(**agg, missing_phases=("decode",), stop_reason="timeout")
    assert (payload["status"], payload["stop_reason"]) == ("failed", "timeout")
    assert payload["errors"] == ["no feasible benchmark points for phase(s): decode"]


# ---------------------------------------------------------------------------------------------------------------
# Gate: pending states and write order
# ---------------------------------------------------------------------------------------------------------------


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
# Publication
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


@pytest.mark.parametrize(
    ("steps", "error"),
    [
        ([0.01] * 5, ValueError),
        ([0.01] * 7, ValueError),
        ([0.01, 0.01, 0.0, 0.01, 0.01, 0.01], ValueError),
        ([float("nan"), 0.01, 0.01, 0.01, 0.01, 0.01], ValueError),  # not a kept step, still refused
        ([0.01, 0.01, 1, 0.01, 0.01, 0.01], TypeError),
        (None, TypeError),
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
        (lambda point: point.update(point_type="agg"), ValueError),
        (lambda point: point.update(benchmark_id=0), ValueError),
        (lambda point: point.update(batch_size=True), TypeError),
        (lambda point: point.update(total_kv_read_tokens=64.0), TypeError),
        (lambda point: point.update(sample_reasons=("kvwarm_real_kv",)), TypeError),
    ],
)
def test_validate_point_rejects_malformed_points(change, error):
    point = _point("decode", 1, 2, 0, 64, ["kvwarm_real_kv"])
    artifact.validate_point(point)
    steps = [0.011, 0.012, 0.013, 0.014, 0.015, 0.016]  # FAKE seconds; their sample is 0.014
    assert artifact.build_fpm(point, 0, 0.014, extra={"sglbench_decode_steps": steps})["counter_id"] == 1
    change(point)
    with pytest.raises(error):
        artifact.validate_point(point)
    # build_fpm is called both ways, so that no refusal of the steps can stand in for the refusal of the point.
    # With consistent steps the point is the only reason to raise: a decode fpm without steps is refused for a
    # sound point too. Without steps, a point whose type was changed is not refused for carrying decode steps.
    with pytest.raises(error):
        artifact.build_fpm(point, 0, 0.014, extra={"sglbench_decode_steps": steps})
    with pytest.raises(error):
        artifact.build_fpm(point, 0, 0.01)


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


@pytest.mark.parametrize(
    ("extra", "error"),
    [
        ({"counter_id": 1}, ValueError),
        ({"scheduled_requests": {}}, ValueError),
        ({"benchmark_measurement": {"schema_version": 1}}, ValueError),
        ([("worker_id", "w")], TypeError),
    ],
)
def test_build_fpm_rejects_bad_extras(extra, error):
    # A prefill point: its fpm is complete without extras, so the extra is the only reason to raise.
    point = _point("prefill", 1, 2, 16, 0)
    assert artifact.build_fpm(point, 0, 0.01, extra={"worker_id": "w"})["worker_id"] == "w"
    with pytest.raises(error):
        artifact.build_fpm(point, 0, 0.01, extra=extra)


@pytest.mark.parametrize(
    ("wall_time", "error"),
    [
        (0.0, ValueError),
        (-0.001, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (1, TypeError),
    ],
)
def test_bad_wall_times_are_rejected_everywhere(wall_time, error):
    point = _point("prefill", 1, 2, 16, 0)
    with pytest.raises(error):
        artifact.build_fpm(point, 0, wall_time)
    # A hand-made fpm gets the same check when it is put into an artifact.
    measurement = artifact.build_measurement(point, {0: artifact.build_fpm(point, 0, 0.01)})
    measurement["rank_fpms"][0]["wall_time"] = wall_time
    with pytest.raises(error):
        artifact.build_rank_artifact(**{**_build_kwargs("prefill", [point]), "measurements": [measurement]})


def test_build_measurement_checks_every_fpm_and_orders_the_ranks():
    point = _point("prefill", 2, 2, 16, 32, ["prefill_real_seed"])
    fpms = {1: artifact.build_fpm(point, 1, 0.02), 0: artifact.build_fpm(point, 0, 0.01)}

    measurement = artifact.build_measurement(point, fpms)

    assert measurement == {"point": point, "kv_seed_regime": "real_prefix", "rank_fpms": fpms}
    assert list(measurement["rank_fpms"]) == [0, 1]

    good = artifact.build_fpm(point, 0, 0.01)
    other_point = artifact.build_fpm(_point("prefill", 3, 2, 16, 32, ["prefill_real_seed"]), 0, 0.01)
    other_shape = artifact.build_fpm(_point("prefill", 2, 2, 17, 32, ["prefill_real_seed"]), 0, 0.01)
    for bad, error in [
        ({}, ValueError),  # no rank at all
        ({0: other_point}, ValueError),  # counter_id is another point's
        ({1: good}, ValueError),  # the fpm says rank 0
        ({0: other_shape}, ValueError),  # scheduled_requests of another shape
    ]:
        with pytest.raises(error):
            artifact.build_measurement(point, bad)
    # A cached prefill point without its stamp has no regime.
    with pytest.raises(ValueError, match="stamps"):
        artifact.build_measurement(_point("prefill", 2, 2, 16, 32), {0: good})


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
    "an-int-step": (0.014, [1, *FAKE_STEPS[1:]], TypeError, STEP_LIST + r"raw_step_seconds\[0\] must be a float"),
    "steps-a-tuple": (0.014, tuple(FAKE_STEPS), TypeError, PLACE + r"fpm sglbench_decode_steps must be a list"),
    "wall-time-is-the-lowest-kept-step": (
        0.013,
        FAKE_STEPS,
        ValueError,
        PLACE + r"fpm wall_time 0.013 is not the sample 0.014",
    ),
    # The comparison is exact: the next float above the sample is refused.
    "wall-time-is-one-float-above-the-sample": (
        math.nextafter(0.014, 1.0),
        FAKE_STEPS,
        ValueError,
        PLACE + r"fpm wall_time 0\.014000000000000002 is not the sample 0\.014 of its raw steps",
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


# ---------------------------------------------------------------------------------------------------------------
# Evidence must be JSON data when it is handed over
# ---------------------------------------------------------------------------------------------------------------

# name -> (a value that is not JSON data, exception, what the refusal says after "is not JSON data: ")
NOT_JSON = {
    # Values json.dumps(allow_nan=False) refuses.
    "nan": (float("nan"), ValueError, ""),
    "bytes": (b"\x00fake-digest", TypeError, ""),
    "set": ({"seed", "warm"}, TypeError, ""),
    # A dict key that is not a str, below the top level of the evidence. json would write these two as the
    # texts "1" and "null", so only the key rule refuses them.
    "nested-int-key": ({1: "x"}, TypeError, "dict keys must be str"),
    "nested-none-key": ({None: "x"}, TypeError, "dict keys must be str"),
}


@pytest.mark.parametrize("nested", [False, True], ids=["top-level", "nested"])
@pytest.mark.parametrize("name", sorted(NOT_JSON))
def test_evidence_that_is_not_json_is_refused_by_the_call_that_receives_it(name, nested):
    """A bad evidence value must stop the point or the run where it is produced, not after the sweep."""
    value, error, reason = NOT_JSON[name]
    evidence = {"fake_evidence": {"deep": [1, value]} if nested else value}

    prefill = _point("prefill", 1, 2, 16, 0)
    with pytest.raises(error, match=f"extra is not JSON data: {reason}"):
        artifact.build_fpm(prefill, 0, 0.01, extra=evidence)

    kwargs = _build_kwargs("decode", _decode_points(), dp=2)
    sound = artifact.build_rank_artifact(**kwargs, config={"fake_evidence": [1]}, producer={"fake_evidence": [1]})
    assert sound["status"] == "complete"
    with pytest.raises(error, match=f"config is not JSON data: {reason}"):
        artifact.build_rank_artifact(**kwargs, config=evidence)
    with pytest.raises(error, match=f"producer is not JSON data: {reason}"):
        artifact.build_rank_artifact(**kwargs, producer=evidence)


# ---------------------------------------------------------------------------------------------------------------
# Producer bugs raise
# ---------------------------------------------------------------------------------------------------------------


def _decode_run() -> dict:
    return _build_kwargs("decode", _decode_points(), dp=2)


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
    "id-gap": (_drop_measurement(1), ValueError, r"benchmark_id is 3, expected 2"),
    "ids-out-of-order": (lambda kwargs: kwargs["measurements"].reverse(), ValueError, r"benchmark_id is 3, expected 1"),
    "missing-rank": (_edit_measurement(1, lambda m: m["rank_fpms"].pop(1)), ValueError, r"rank_fpms has ranks \[0\]"),
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
    "elapsed-nan": (_with(benchmark_elapsed_seconds=float("nan")), ValueError, r"must be finite and >= 0"),
    "config-overrides-mode": (_with(config={"mode": "prefill"}), ValueError, r"config may not contain 'mode'"),
    # A key that is not a str, at the top level of a block: json would write it as the text "1".
    "config-key-not-a-str": (_with(config={1: "x"}), TypeError, r"^config keys must be str, got 1$"),
    "producer-overrides-name": (_with(producer={"name": "someone-else"}), ValueError, r"may not contain \['name'\]"),
    "rank-equal-to-size": (_with(dp_rank=2), ValueError, r"dp_rank 2 is outside range\(2\)"),
    "unknown-mode": (_with(mode="bench"), ValueError, r"mode must be one of"),
    "empty-run-id": (_with(run_id=""), ValueError, r"run_id must not be empty"),
    "unknown-stop-reason": (_with(stop_reason="banana"), ValueError, r"stop_reason must be None or 'timeout'"),
    "missing-phase-of-another-mode": (
        _with(missing_phases=("prefill",)),
        ValueError,
        r"'prefill' is not a phase of mode 'decode'",
    ),
    "failure-without-point-key": (_with(failure={"reason": "boom"}), ValueError, r"exactly the keys"),
    "failure-empty-reason": (_with(failure={"point": None, "reason": ""}), ValueError, r"reason must not be empty"),
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
}


@pytest.mark.parametrize("name", sorted(PRODUCER_BUGS))
def test_producer_bugs_raise(name):
    kwargs = _decode_run()
    assert artifact.build_rank_artifact(**copy.deepcopy(kwargs))["status"] == "complete"  # the baseline is sound
    change, error, message = PRODUCER_BUGS[name]
    change(kwargs)
    with pytest.raises(error, match=message):
        artifact.build_rank_artifact(**kwargs)


def test_a_second_sample_of_one_coordinate_is_refused_because_publication_would_raise():
    first = _point("decode", 1, 4, 0, 64, ["kvwarm_real_kv"])
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


def _two_ranks() -> list[dict]:
    return _rank_artifacts("decode", _decode_points(), dp=2)


# name -> (change applied to the two sound rank artifacts of one run, fragment of the message)
MERGE_BUGS = {
    "swapped-order": (lambda ranks: ranks.reverse(), r"consecutive ascending rank order, got ranks \[1, 0\]"),
    "different-run-id": (lambda ranks: ranks[1].update(run_id="another-run"), r"disagree on run_id"),
    "different-groups": (
        lambda ranks: ranks[1]["iteration_groups"][0].update(wall_time=9.0),
        r"disagree on iteration_groups",
    ),
    "fewer-rows": (lambda ranks: ranks[1]["results"].pop(), r"disagree on result points"),
    # The kvwarm block is one per run: a rank without it.
    "kvwarm-block-in-one-rank-only": (
        lambda ranks: ranks[1].pop("kvwarm"),
        r"^ranks disagree on kvwarm: rank 1 differs from rank 0$",
    ),
    "row-with-another-ranks-fpm": (
        lambda ranks: ranks[1]["results"][0]["fpms"][0].update(dp_rank=0),
        r"a result row holds the fpm of rank 0",
    ),
    "merged-passed-as-rank": (lambda ranks: ranks[0].update(artifact_type="merged"), r"not a schema-2 rank artifact"),
    "unknown-status": (lambda ranks: [rank.update(status="running") for rank in ranks], r"unknown status 'running'"),
    "worker-probe-advertised": (
        lambda ranks: ranks[0].update(worker_probe_path="/results/probe.json"),
        r"contains the keys \['worker_probe_path'\]",
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


def _empty_the_run(payload: dict) -> None:
    # No row, no group, and a coverage and a sum that agree with that: only "complete with no point" is wrong.
    payload["results"].clear()
    payload["iteration_groups"].clear()
    payload["coverage"].update(expected_points=0, completed_points=0)
    payload["timing"]["measured_iteration_seconds"] = 0.0


def _first_group_entry(payload: dict) -> dict:
    return payload["iteration_groups"][0]["rank_results"][0]


# name -> (one edit of a complete decode rank artifact, the field the refusal names)
COMPLETE_ARTIFACT_EDITS = {
    "valid-false": (_set("valid", False), r"valid"),
    "usable-false": (_set("usable", False), r"usable"),
    "stop-reason-timeout": (_set("stop_reason", "timeout"), r"stop_reason"),
    "error-text": (_set("error", "x"), r"error"),
    "skipped-point-listed": (
        lambda payload: payload["skipped_points"].append({"point": _decode_points()[0], "reason": "x"}),
        r"skipped_points",
    ),
    "coverage-expects-more": (
        lambda payload: payload["coverage"].update(expected_points=4),
        r"coverage\.expected_points",
    ),
    "coverage-skipped-one": (lambda payload: payload["coverage"].update(skipped_points=1), r"coverage\.skipped_points"),
    # False == 0, so the validator lets this one through; the gate wants exact ints and rejects it.
    "coverage-count-is-a-bool": (
        lambda payload: payload["coverage"].update(skipped_points=False),
        r"coverage\.skipped_points",
    ),
    "no-rows-at-all": (_empty_the_run, r"results is empty"),
    "group-removed": (lambda payload: payload["iteration_groups"].pop(), r"iteration_groups"),
    "row-fpm-differs-from-group": (_replace_row_fpm, r"results\[0\]\.fpms"),
    "group-wall-time-changed": (
        lambda payload: payload["iteration_groups"][1].update(wall_time=9.0),
        r"iteration_groups\[1\]\.wall_time",
    ),
    "group-rank-entry-removed": (
        lambda payload: payload["iteration_groups"][2]["rank_results"].pop(),
        r"iteration_groups\[2\]\.rank_results must hold one entry per rank",
    ),
    "group-rank-entry-without-fpm": (
        lambda payload: _first_group_entry(payload).update(fpms=[]),
        r"iteration_groups\[0\]\.rank_results\[0\] must hold dp_rank 0 and exactly one fpm",
    ),
    "group-fpm-without-wall-time": (
        lambda payload: _first_group_entry(payload)["fpms"][0].update(wall_time=None),
        r"iteration_groups\[0\]\.rank_results\[0\] has no float fpm wall_time",
    ),
    "measured-seconds-doubled": (
        lambda payload: payload["timing"].update(
            measured_iteration_seconds=payload["timing"]["measured_iteration_seconds"] * 2
        ),
        r"timing\.measured_iteration_seconds",
    ),
    "timing-removed": (lambda payload: payload.pop("timing"), r"timing\.measured_iteration_seconds"),
    "kvwarm-not-eligible": (
        lambda payload: payload["kvwarm"].update(warm_eligible=False, skip_reason="fake reason"),
        r"kvwarm",
    ),
    "row-regime-fake-fallback": (
        lambda payload: payload["results"][0].update(kv_seed_regime="fake_fallback"),
        r"results\[0\]\.kv_seed_regime",
    ),
    "row-stamp-removed": (
        lambda payload: payload["results"][0]["point"].update(sample_reasons=[]),
        r"results\[0\]\.point",
    ),
    # The raw decode steps of an fpm. No AISim reader parses them, so neither checker rejects this edit: the
    # module is the only place that can, and it does so with the error build_fpm raises for the same fault.
    "steps-removed": (
        lambda payload: _first_group_entry(payload)["fpms"][0].pop(STEPS_KEY),
        r"iteration_groups\[0\]\.rank_results\[0\]: a decode fpm must carry its 6 raw step times",
    ),
}


@pytest.mark.parametrize("name", sorted(COMPLETE_ARTIFACT_EDITS))
def test_a_complete_artifact_edited_after_the_build_is_refused_before_any_write(tmp_path, name):
    """Each edit makes a ``complete`` artifact that a checker rejects or, for the raw decode steps, that only the
    module can refuse; the module must not write it.
    """
    edit, field = COMPLETE_ARTIFACT_EDITS[name]
    dp = 2
    cell = _cell(phase="decode", dp=dp, strategy="dep")
    points = _decode_points()

    # The same edit in every rank file, so that the ranks still agree with each other.
    payloads = _rank_artifacts("decode", points, dp=dp)
    for payload in payloads:
        edit(payload)
        assert payload["status"] == "complete"

    # Put on disk as they are, the edited rank files are rejected: by the validator, except for the one edit
    # that only the gate catches and the one of the raw decode steps, which no checker reads. (There is no
    # merged file here, so a gate that finds no fault says "pending".)
    forced = tmp_path / "forced"
    pod = _pod(forced, cell)
    rank_paths = [artifact.rank_output_path(str(pod / "benchmark.json"), rank) for rank in range(dp)]
    for path, payload in zip(rank_paths, payloads, strict=True):
        artifact.atomic_write_json(path, payload)
    if name == "coverage-count-is-a-bool":
        assert len(_validate(cell, forced).points) == len(points)
        assert _gate(rank_paths, "decode").returncode == GATE_INVALID
    elif name == "steps-removed":
        assert [measurement.point for measurement in _validate(cell, forced).points] == points
        assert _gate(rank_paths, "decode").returncode == GATE_MERGED_PENDING
    else:
        with pytest.raises((ValueError, TypeError)):
            _validate(cell, forced)
        gate_catches_it = name.startswith("coverage-") or name in {"valid-false", "no-rows-at-all"}
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


# ---------------------------------------------------------------------------------------------------------------
# Isolation from the caller, and nothing on disk before every document is serialised
# ---------------------------------------------------------------------------------------------------------------


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


def test_write_artifacts_refuses_incomplete_or_unserialisable_input_with_the_disk_untouched(tmp_path):
    output_path = str(tmp_path / "benchmark.json")
    complete = _two_ranks()
    partial = _rank_artifacts("decode", _decode_points(), dp=2, completed=1, stop_reason="timeout")

    with pytest.raises(ValueError, match="needs the artifacts of ranks"):
        artifact.write_artifacts(output_path, complete[:1])  # rank 0 of a two-rank run
    with pytest.raises(ValueError, match="disagree on status"):
        artifact.write_artifacts(output_path, [complete[0], partial[1]])
    with pytest.raises(ValueError, match="no rank artifact"):
        artifact.write_artifacts(output_path, [])

    # A value JSON cannot hold, in the LAST rank, whose file is the first one written: nothing may be on disk
    # when it is found. (The test after this one puts the value into the file that is written last.)
    ranks = _two_ranks()
    ranks[1]["producer"]["fake_value"] = {1, 2}
    with pytest.raises((ValueError, TypeError)):
        artifact.write_artifacts(output_path, ranks)

    with pytest.raises((ValueError, TypeError)):
        artifact.write_artifacts(output_path + "/", _two_ranks())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("status", ["complete", "partial", "failed"])
def test_every_document_is_serialised_before_the_first_file_is_written(tmp_path, status):
    """The files are written highest rank first, so a value JSON cannot hold in a lower rank is met late.

    If a document were serialised only when its file is written, the files of the higher ranks would already
    be on disk when the value is found in rank 0.
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
    ranks[0]["producer"]["fake_value"] = float("nan")
    refused = tmp_path / "refused"
    refused.mkdir()
    with pytest.raises(ValueError):
        artifact.write_artifacts(str(refused / "benchmark.json"), ranks)
    assert list(refused.iterdir()) == []


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


def test_written_files_are_ascii_json_readable_under_any_encoding(tmp_path):
    payload = {"producer": {"note": "caf" + chr(0xE9) + " " + chr(0x6D4B) + chr(0x8BD5)}}  # not ASCII
    artifact.atomic_write_json(str(tmp_path / "benchmark.json"), payload)
    raw = (tmp_path / "benchmark.json").read_bytes()
    assert raw.isascii()
    assert json.loads(raw.decode("ascii")) == payload


# ---------------------------------------------------------------------------------------------------------------
# A run that is not complete
# ---------------------------------------------------------------------------------------------------------------

FAKE_CALLER_BLOCKS = {
    "config": {"fake_model": "fake-model", "fake_limits": {"fake_page_size": 16}},
    "producer": {"fake_grid_limits": {"fake_max_tokens": 4096}},
}

NOT_COMPLETE_RUNS = {
    # name -> (mode, numbers of completed points, keyword arguments for that number, status, stop_reason)
    "deadline": ("decode", (2,), lambda points, done: {"stop_reason": "timeout"}, "partial", "timeout"),
    "failed-point": (
        "decode",
        (2,),
        lambda points, done: {"failure": {"point": points[done], "reason": "fake reason"}},
        "failed",
        None,
    ),
    "failure-after-the-deadline": (
        "decode",
        (2,),
        lambda points, done: {"failure": {"point": None, "reason": "fake engine exit"}, "stop_reason": "timeout"},
        "failed",
        "timeout",
    ),
}


@pytest.mark.parametrize("ending", sorted(NOT_COMPLETE_RUNS))
def test_whatever_ends_a_run_its_artifact_keeps_the_callers_blocks_and_the_rows_of_the_completed_points(ending):
    """Neither checker reads a rejected artifact, so only this pins what a person finds in one."""
    mode, done_counts, arguments, status, stop_reason = NOT_COMPLETE_RUNS[ending]
    dp = 2
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
