# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The collector's gate, ``fpm_exec.sh``, for an SGLang cell with two data-parallel ranks. No SGLang, no GPU.

A DEP cell of two GPUs runs on one node with two attention-DP ranks. Its ``fpm_env.sh`` is the one of the vLLM
render of the cell and exports ``FPM_DATA_PARALLEL_SIZE=2`` and ``FPM_LOCAL_DATA_PARALLEL_SIZE=2``
(test_fpm_sglang_builder.py renders it), and the cell builder switches the gate to its mode without etcd. The
gate then waits for one rank file per local rank and for the merged file that names both.

The whole script runs, staged by the helpers of test_fpm_sglbench_gate.py with those two values. The engine
is the FAKE launcher of that file: a short Python program that writes result files and ends the way
``sglbench_launcher.py`` ends. It is not the launcher, and no engine produced the files. Where a test says so,
the files are built by the launcher's own artifact module from typed-in times; otherwise they are the
typed-in results of test_fpm_exec.py.

The gate itself is not changed for these cells. What the tests pin is that it needs no change.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.fpm_forward.runtime import sglbench_artifact as artifact

from .test_fpm_exec import _benchmark_result, _finalized_artifacts, _run
from .test_fpm_sglbench_artifact import _points, _rank_artifacts
from .test_fpm_sglbench_gate import ENGINE_EXITED, _engine_report, _fake_launcher, _stage_cell, launcher

pytestmark = pytest.mark.unit

# What fpm_env.sh of a two-GPU DEP cell exports for its ranks.
TWO_LOCAL_RANKS = {"FPM_DATA_PARALLEL_SIZE": 2, "FPM_LOCAL_DATA_PARALLEL_SIZE": 2}
# How long the FAKE launcher stays alive after its last file where a test does not look at that wait. The
# launcher's own wait is ``launcher.FINAL_WAIT_SECONDS``; one test keeps it.
SHORT_FINAL_WAIT = 0.5


def _stage_two_ranks(tmp_path: Path) -> SimpleNamespace:
    """One staged pod of a cell with two data-parallel ranks on its one node, in the gate's mode without etcd."""
    staged = _stage_cell(tmp_path, env_overrides=TWO_LOCAL_RANKS)
    # The paths the gate derives for the two ranks (``benchmark_path_for_dp_rank``).
    staged.rank_paths = [str(staged.benchmark_output), str(staged.benchmark_output.with_name("benchmark_dp1.json"))]
    staged.merged_path = str(staged.benchmark_output.with_name("benchmark_merged.json"))
    return staged


def _typed_in_files(staged: SimpleNamespace, ranks=(0, 1)) -> dict:
    """Complete rank files that say ``ranks`` in their records, in the gate's order, and their merged file."""
    files = {path: _benchmark_result(dp_rank=rank) for path, rank in zip(staged.rank_paths, ranks, strict=False)}
    return _finalized_artifacts(files)


def _writer_files(staged: SimpleNamespace) -> dict:
    """The two rank files and the merged file as the launcher's artifact module builds them, with FAKE times.

    The order is the one ``sglbench_artifact.write_artifacts`` writes in: the highest rank first, rank 0 next
    and the merged file last.
    """
    payloads = _rank_artifacts("prefill", _points("prefill"), dp=2)
    paths = [artifact.rank_output_path(str(staged.benchmark_output), rank) for rank in range(2)]
    assert paths == staged.rank_paths
    merged = artifact.build_merged_artifact(payloads, paths)
    assert merged["merged_output_path"] == staged.merged_path
    return {paths[1]: payloads[1], paths[0]: payloads[0], staged.merged_path: merged}


def _results(staged: SimpleNamespace) -> list[str]:
    return sorted(path.name for path in staged.results.glob("benchmark*"))


@pytest.mark.parametrize("built_by", ["typed-in", "the-launchers-artifact-module"])
def test_two_rank_files_and_the_merged_file_that_names_both_give_0(tmp_path, built_by):
    staged = _stage_two_ranks(tmp_path)
    files = _typed_in_files(staged) if built_by == "typed-in" else _writer_files(staged)
    assert sorted(files) == sorted([*staged.rank_paths, staged.merged_path])
    # With the files of the artifact module the FAKE launcher also waits as long as the launcher does.
    final_wait = SHORT_FINAL_WAIT if built_by == "typed-in" else launcher.FINAL_WAIT_SECONDS
    _fake_launcher(tmp_path, staged, files=files, exit_code=launcher.EXIT_COMPLETE, final_wait=final_wait)
    done = _run(staged)

    assert done.returncode == 0, done.stderr
    assert _results(staged) == ["benchmark.json", "benchmark_dp1.json", "benchmark_merged.json"]
    merged = json.loads(Path(staged.merged_path).read_text())
    assert (merged["rank_files"], merged["dp"]["source_ranks"]) == (staged.rank_paths, [0, 1])
    # The mode of an SGLang cell: the launcher's preflight, no etcd and no endpoint.
    assert _engine_report(staged) == {
        "etcd_endpoints": None,
        "pythonpath": staged.env.get("PYTHONPATH"),
        "sglbench_preflight_ran": True,
        "dynamo_preflight_ran": False,
    }
    assert not (staged.results / "etcd.log").exists()
    assert "sending SIGKILL" not in done.stderr


def test_one_rank_file_of_two_is_not_a_result_of_the_cell(tmp_path):
    staged = _stage_two_ranks(tmp_path)
    # A launcher that ran one rank although the cell has two: its one rank file and a merged file of one rank,
    # both complete. The gate keeps waiting for the file of rank 1 until the launcher has ended.
    files = _typed_in_files(staged, ranks=(0,))
    assert sorted(files) == sorted([staged.rank_paths[0], staged.merged_path])
    _fake_launcher(tmp_path, staged, files=files, exit_code=launcher.EXIT_COMPLETE, final_wait=SHORT_FINAL_WAIT)
    done = _run(staged)

    assert done.returncode == 1
    assert done.stderr.endswith(ENGINE_EXITED + "\n")
    assert _results(staged) == ["benchmark.json", "benchmark_merged.json"]


def test_a_rank_file_that_holds_the_records_of_the_other_rank_gives_1(tmp_path):
    staged = _stage_two_ranks(tmp_path)
    # Both files are there, and the file at the path of rank 1 says rank 0 in every record.
    files = _typed_in_files(staged, ranks=(0, 0))
    _fake_launcher(tmp_path, staged, files=files, exit_code=launcher.EXIT_COMPLETE, final_wait=SHORT_FINAL_WAIT)
    done = _run(staged)

    assert done.returncode == 1
    assert f"Invalid FPM benchmark result {staged.rank_paths[1]}: FPM dp_ranks [0] != [1]" in done.stderr
    assert ENGINE_EXITED not in done.stderr


def test_a_merged_file_that_names_one_rank_does_not_bind_two_complete_rank_files(tmp_path):
    staged = _stage_two_ranks(tmp_path)
    # Both rank files are complete, and the merged file is the one of a run with rank 0 alone.
    one_rank = _typed_in_files(staged, ranks=(0,))
    files = {
        staged.rank_paths[1]: _benchmark_result(dp_rank=1),
        staged.rank_paths[0]: one_rank[staged.rank_paths[0]],
        staged.merged_path: one_rank[staged.merged_path],
    }
    _fake_launcher(tmp_path, staged, files=files, exit_code=launcher.EXIT_COMPLETE, final_wait=SHORT_FINAL_WAIT)
    done = _run(staged)

    assert done.returncode == 1
    assert (
        f"Invalid FPM benchmark result {staged.merged_path}: "
        "merged paths or source ranks do not bind the local DP shard"
    ) in done.stderr


@pytest.mark.parametrize("stale", ["benchmark_dp1.json", "benchmark_merged.json"])
def test_an_output_of_an_earlier_run_ends_the_cell_before_the_launcher_starts(tmp_path, stale):
    staged = _stage_two_ranks(tmp_path)
    # The file of rank 1 is an output of this cell only because the cell has two local ranks.
    leftover = staged.results / stale
    leftover.write_text("{}")
    _fake_launcher(tmp_path, staged, files=_typed_in_files(staged), exit_code=launcher.EXIT_COMPLETE)
    done = _run(staged)

    assert done.returncode == 1
    assert done.stderr == f"Refusing to overwrite existing benchmark output: {leftover}\n"
    assert not staged.engine_report.exists()
    assert leftover.read_text() == "{}"
