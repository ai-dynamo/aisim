# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The collector's gate, ``fpm_exec.sh``, in the mode of an SGLang self-benchmark cell. No SGLang, no GPU.

The whole script runs, staged by the helpers of test_fpm_exec.py. What differs from those tests:

- The mode is switched on the way the collector does it, by a line in ``collector-runtime-env.sh``.
- ``PATH`` holds the commands the gate runs and nothing else, so no ``etcd`` of the machine can be found.
  Tests that have to show that etcd was NOT started keep the trace-writing stub of test_fpm_exec.py there.
- Both preflight files are stand-ins that leave a mark. The preflight itself is tested in
  test_fpm_sglbench_preflight.py; one test here stages the shipped file, to show that the gate starts it by
  path and stops on its failure.
- The engine is a FAKE launcher: a short Python program that writes typed-in result files and ends the way
  ``sglbench_launcher.py`` ends, with its exit codes and its wait after the last file. It is not the launcher
  and no engine produced the files. The test of a cell with two data-parallel ranks takes its files from the
  launcher's own artifact module, built from FAKE times.

The last test runs the same script without the switch, to show that a Dynamo cell still starts etcd and runs
its own preflight.
"""

from __future__ import annotations

import json
import shlex
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.fpm_forward import sglang_fpm_builder
from collector.fpm_forward.runtime import sglbench_artifact as artifact

from .test_fpm_exec import (
    _benchmark_result,
    _finalized_artifacts,
    _run,
    _setsid_run_script,
    _stage,
    _tree_killing_engine_prelude,
    _write_executable,
)
from .test_fpm_sglbench_artifact import _points, _rank_artifacts

RUNTIME = Path(__file__).resolve().parents[3] / "collector/fpm_forward/runtime"
# The pod holds the runtime files flat in one directory and imports them by their plain names. The directory
# goes to the END of the path so that it shadows nothing.
if str(RUNTIME) not in sys.path:
    sys.path.append(str(RUNTIME))

import sglbench_launcher as launcher

pytestmark = pytest.mark.unit

# The cell builder writes the line and the gate reads it; neither file can import the other. A name that
# changed on one side only would leave every SGLang cell with the gate of a Dynamo cell, which waits for an
# etcd the image does not have. The tests write the builder's name, so such a change fails the first of them.
SWITCH = sglang_fpm_builder.GATE_SWITCH_ENV

# Every command the gate runs: bash for run.sh, python3 for the preflight and the checker, awk for the two
# engine streams, mktemp and mkfifo for the folder of the two pipes the awks read and rm
# to remove it, sleep for the polls and the wait for the awks, dirname and mkdir for the result folders, and
# rm and tail around etcd, which a cell in the mode never starts.
GATE_COMMANDS = ("bash", "python3", "awk", "mktemp", "mkfifo", "sleep", "dirname", "mkdir", "rm", "tail")

ENGINE_EXITED = "Engine exited before writing all FPM benchmark outputs (rank, merged, and advertised worker probe)"
ONE_NODE_ONLY = f"{SWITCH}=1 requires FPM_NODE_COUNT=1 and FPM_NODE_RANK=0"
UNKNOWN_VALUE = f"{SWITCH} must be unset, empty or 1"
LIVENESS_TEST = '  if ! kill -0 "$engine_pid" 2>/dev/null; then\n'


def _stage_cell(tmp_path: Path, *, switch: str | None = "1", etcd_stub: bool = False, **stage_options):
    """One staged pod. ``switch`` is the value of the mode line, None for a cell without the line."""
    staged = _stage(tmp_path, run_script="", **stage_options)
    bin_dir = tmp_path / "bin"
    if not etcd_stub:
        (bin_dir / "etcd").unlink()
    for command in GATE_COMMANDS:
        found = shutil.which(command)
        assert found, f"the gate needs {command}"
        _write_executable(bin_dir / command, f'#!/bin/sh\nexec {shlex.quote(found)} "$@"\n')
    staged.env["PATH"] = str(bin_dir)
    assert (shutil.which("etcd", path=staged.env["PATH"]) is not None) == etcd_stub
    # Neither name may come from the shell that runs the tests.
    for name in (SWITCH, "ETCD_ENDPOINTS"):
        staged.env.pop(name, None)

    staged.sglbench_preflight_ran = tmp_path / "sglbench-preflight-ran"
    staged.dynamo_preflight_ran = tmp_path / "dynamo-preflight-ran"
    staged.engine_report = tmp_path / "engine-report.json"
    (staged.workdir / "sglbench_preflight.py").write_text(
        f"open({str(staged.sglbench_preflight_ran)!r}, 'w').close()\n"
    )
    (staged.workdir / "preflight.py").write_text(f"open({str(staged.dynamo_preflight_ran)!r}, 'w').close()\n")
    if switch is not None:
        with (staged.workdir / "collector-runtime-env.sh").open("a") as handle:
            handle.write(f"export {SWITCH}={switch}\n")
    return staged


def _fake_launcher(
    tmp_path: Path,
    staged: SimpleNamespace,
    *,
    files: dict,
    exit_code: int,
    final_wait: float = launcher.FINAL_WAIT_SECONDS,
    wait_for: Path | None = None,
    kill_itself: bool = False,
    ends_on_sigterm: bool = False,
) -> None:
    """Stage the FAKE launcher as ``run.sh``.

    It reports what it was started with, waits for ``wait_for`` when one is given, writes ``files`` in their
    order by rename (the rank file before the merged one) and then, like ``sglbench_launcher.py``, ignores
    SIGTERM, stays alive for ``final_wait`` seconds and ends with ``exit_code``. With ``kill_itself`` it dies
    of SIGKILL before it writes anything, as the launcher does when SGLang kills it. With ``ends_on_sigterm``
    it idles after its files and ends on SIGTERM, which is what a Dynamo worker does.
    """
    body = f"""\
import json
import os
import pathlib
import signal
import sys
import time

pathlib.Path({str(staged.engine_report)!r}).write_text(json.dumps({{
    "etcd_endpoints": os.environ.get("ETCD_ENDPOINTS"),
    "pythonpath": os.environ.get("PYTHONPATH"),
    "sglbench_preflight_ran": os.path.exists({str(staged.sglbench_preflight_ran)!r}),
    "dynamo_preflight_ran": os.path.exists({str(staged.dynamo_preflight_ran)!r}),
}}))
wait_for = {str(wait_for) if wait_for else None!r}
deadline = time.monotonic() + 30
while wait_for and not os.path.exists(wait_for):
    if time.monotonic() > deadline:
        sys.exit("the FAKE launcher waited for the gate in vain")
    time.sleep(0.02)
if {kill_itself!r}:
    os.kill(os.getpid(), signal.SIGKILL)
if not {ends_on_sigterm!r}:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
for path, payload in json.loads({json.dumps(files)!r}).items():
    partial = path + ".tmp"
    pathlib.Path(partial).write_text(json.dumps(payload))
    os.replace(partial, path)
if {ends_on_sigterm!r}:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(0.1)
time.sleep({final_wait})
sys.exit({exit_code})
"""
    (staged.workdir / "run.sh").write_text(_setsid_run_script(tmp_path, body))


def _complete_files(staged: SimpleNamespace) -> dict:
    return _finalized_artifacts({str(staged.benchmark_output): _benchmark_result()})


def _unfinished_rank(status: str, error: str) -> dict:
    rank = _benchmark_result(status=status, valid=False)
    rank["errors"] = [error]
    return rank


def _engine_report(staged: SimpleNamespace) -> dict:
    return json.loads(staged.engine_report.read_text())


def _hold_the_liveness_test(staged: SimpleNamespace, first_check_done: Path) -> None:
    """Make the gate wait, between its file check and its liveness test, until the engine is gone.

    The mark tells the FAKE launcher that the first check is over. The launcher then writes its files and
    ends, so the liveness test finds complete files on disk and no process: the window that the launcher's
    own wait keeps shut in a real cell.
    """
    script = staged.script.read_text()
    assert script.count(LIVENESS_TEST) == 1
    hold = (
        f'  : > {shlex.quote(str(first_check_done))}\n  while kill -0 "$engine_pid" 2>/dev/null; do sleep 0.1; done\n'
    )
    staged.script.write_text(script.replace(LIVENESS_TEST, hold + LIVENESS_TEST))


# ---------------------------------------------------------------------------------------------------------------
# The mode
# ---------------------------------------------------------------------------------------------------------------


def test_complete_files_give_0_with_no_etcd_no_endpoint_and_the_sglang_preflight(tmp_path):
    staged = _stage_cell(tmp_path)
    _fake_launcher(tmp_path, staged, files=_complete_files(staged), exit_code=launcher.EXIT_COMPLETE)
    done = _run(staged)

    assert done.returncode == 0, done.stderr
    assert not (staged.results / "etcd.log").exists()
    assert "etcd exited" not in done.stderr and "etcd readiness timeout" not in done.stderr
    assert staged.sglbench_preflight_ran.exists() and not staged.dynamo_preflight_ran.exists()
    assert _engine_report(staged) == {
        "etcd_endpoints": None,
        # The launcher is started by path and needs no import path from the gate.
        "pythonpath": staged.env.get("PYTHONPATH"),
        "sglbench_preflight_ran": True,
        "dynamo_preflight_ran": False,
    }
    # The launcher ended by itself inside the grace period; nothing had to be killed.
    assert "sending SIGKILL" not in done.stderr


@pytest.mark.parametrize(
    ("exit_code", "kill_itself", "expected"),
    [(launcher.EXIT_REJECTED, False, 64), (0, True, 137)],
    ids=["rejected-without-an-artifact", "killed"],
)
def test_no_file_gives_the_status_the_launcher_ended_with(tmp_path, exit_code, kill_itself, expected):
    staged = _stage_cell(tmp_path)
    _fake_launcher(tmp_path, staged, files={}, exit_code=exit_code, final_wait=0, kill_itself=kill_itself)
    done = _run(staged)

    assert done.returncode == expected
    assert ENGINE_EXITED in done.stderr
    assert list(staged.results.glob("benchmark*")) == []


def test_what_the_launcher_prints_after_the_engine_shutdown_is_in_the_exec_stream_and_in_the_log(tmp_path):
    """SGLang's engine shutdown sends SIGKILL to every descendant of the launcher's process, and the launcher
    writes its files and prints its outcome after that. A cell that passes must keep the outcome line. A
    filter in the launcher's tree is dead by then, so the line has no reader and the print itself fails."""

    staged = _stage_cell(tmp_path)
    before = "[sglbench] SGLang 0.5.23: the start-up checks passed; building the engine"
    after = f"[sglbench] complete: wrote {staged.benchmark_output}"
    body = (
        _tree_killing_engine_prelude()
        + f"""import json
import pathlib
import sys
import time

print({before!r}, file=sys.stderr, flush=True)
_kill_descendants()
signal.signal(signal.SIGTERM, signal.SIG_IGN)
for path, payload in json.loads({json.dumps(_complete_files(staged))!r}).items():
    partial = path + ".tmp"
    pathlib.Path(partial).write_text(json.dumps(payload))
    os.replace(partial, path)
print({after!r}, file=sys.stderr, flush=True)
time.sleep(1)
sys.exit({launcher.EXIT_COMPLETE})
"""
    )
    (staged.workdir / "run.sh").write_text(_setsid_run_script(tmp_path, body))

    done = _run(staged)

    assert done.returncode == 0, done.stderr
    assert before in done.stderr and after in done.stderr
    assert (staged.results / "engine.stderr.log").read_text().splitlines() == [before, after]
    assert list(staged.gate_tmp.iterdir()) == []


@pytest.mark.parametrize(
    ("switch", "env_overrides", "message"),
    [
        ("1", {"FPM_NODE_COUNT": 2, "FPM_NODE_RANK": 1}, ONE_NODE_ONLY),
        ("0", {}, UNKNOWN_VALUE),
    ],
    ids=["two-nodes-follower", "switch-0"],
)
def test_more_than_one_node_or_another_switch_value_gives_2_before_anything_starts(
    tmp_path, switch, env_overrides, message
):
    staged = _stage_cell(tmp_path, switch=switch, etcd_stub=True, env_overrides=env_overrides)
    _fake_launcher(tmp_path, staged, files=_complete_files(staged), exit_code=0, ends_on_sigterm=True)
    done = _run(staged)

    assert done.returncode == 2
    assert done.stderr == message + "\n"
    assert not staged.etcd_trace.exists()
    assert not staged.sglbench_preflight_ran.exists() and not staged.dynamo_preflight_ran.exists()
    assert not staged.engine_report.exists()


@pytest.mark.parametrize("mode", ["sglbench", "dynamo"])
def test_files_that_are_complete_only_after_the_launcher_ended_are_accepted_in_the_mode_alone(tmp_path, mode):
    staged = _stage_cell(tmp_path, switch="1" if mode == "sglbench" else None, etcd_stub=mode == "dynamo")
    first_check_done = tmp_path / "first-check-done"
    _hold_the_liveness_test(staged, first_check_done)
    files = _complete_files(staged)
    _fake_launcher(tmp_path, staged, files=files, exit_code=0, final_wait=0, wait_for=first_check_done)
    done = _run(staged)

    # In both modes the files on disk are complete and were written after the first check.
    assert first_check_done.exists()
    for path, payload in files.items():
        assert json.loads(Path(path).read_text()) == payload
    if mode == "sglbench":
        assert done.returncode == 0, done.stderr
        assert done.stderr == ""
    else:
        # A Dynamo worker that ends by itself stays a failure, as it is today.
        assert done.returncode == 1
        assert ENGINE_EXITED in done.stderr


def test_the_second_check_gives_1_for_a_file_that_is_not_complete_as_the_first_does(tmp_path):
    status, exit_code = "failed", launcher.EXIT_FAILED
    staged = _stage_cell(tmp_path)
    first_check_done = tmp_path / "first-check-done"
    _hold_the_liveness_test(staged, first_check_done)
    error = f"{status}: FAKE reason"
    files = {str(staged.benchmark_output): _unfinished_rank(status, error)}
    _fake_launcher(tmp_path, staged, files=files, exit_code=exit_code, final_wait=0, wait_for=first_check_done)
    done = _run(staged)

    # No check saw the file while the launcher lived. The result is still the one of a check that did: 1 and
    # the checker's line, not the launcher's code, so the status of a cell does not depend on the timing.
    assert done.returncode == 1
    assert f"status={status!r} valid=False errors=[{error!r}]" in done.stderr
    assert ENGINE_EXITED not in done.stderr


@pytest.mark.parametrize(("exit_code", "expected"), [(launcher.EXIT_COMPLETE, 1)], ids=["launcher-0"])
def test_the_second_check_does_not_accept_a_rank_file_without_its_merged_file(tmp_path, exit_code, expected):
    staged = _stage_cell(tmp_path)
    first_check_done = tmp_path / "first-check-done"
    _hold_the_liveness_test(staged, first_check_done)
    # A complete rank file alone is "finalization pending" for the checker, and nothing can finalize it now.
    files = {str(staged.benchmark_output): _benchmark_result()}
    _fake_launcher(tmp_path, staged, files=files, exit_code=exit_code, final_wait=0, wait_for=first_check_done)
    done = _run(staged)

    assert done.returncode == expected
    assert done.stderr.endswith(ENGINE_EXITED + "\n")
    assert json.loads(staged.benchmark_output.read_text())["status"] == "complete"
    assert not staged.benchmark_output.with_name("benchmark_merged.json").exists()


def test_the_shipped_preflight_under_the_gate_ends_a_cell_whose_image_has_no_sglang(tmp_path):
    staged = _stage_cell(tmp_path)
    # The pod's python3 is the interpreter of the tests without the packages of this machine, so that no
    # SGLang can be found whatever is installed here.
    _write_executable(tmp_path / "bin" / "python3", f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -S "$@"\n')
    # The launcher's files and the preflight as the collector stages them: flat, beside the gate's inputs.
    for path in RUNTIME.glob("sglbench_*.py"):
        text = path.read_text(encoding="utf-8")
        if path.name == "sglbench_preflight.py":
            assert text.count('"/results/') == 1
            text = text.replace('"/results/', f'"{staged.results}/')
        (staged.workdir / path.name).write_text(text, encoding="utf-8")
    _fake_launcher(tmp_path, staged, files=_complete_files(staged), exit_code=0)
    done = _run(staged)

    assert done.returncode == 1
    audit = json.loads((staged.results / "runtime-preflight.json").read_text())
    assert audit["status"] == "failed" and audit["sglang_version"] is None
    assert done.stderr.splitlines() == [f"[sglbench] preflight FAILED: {failure}" for failure in audit["failures"]]
    # Started by path, it found every sibling without PYTHONPATH: all failures are about SGLang.
    assert audit["launcher_modules"] and not [failure for failure in audit["failures"] if "sglbench_" in failure]
    assert not staged.engine_report.exists()


# ---------------------------------------------------------------------------------------------------------------
# Two data-parallel ranks
# ---------------------------------------------------------------------------------------------------------------

# A DEP cell of two GPUs runs on one node with two attention-DP ranks, and the gate waits for one rank file per
# local rank and for the merged file that names both. The gate was not changed for these cells.

# What fpm_env.sh of a two-GPU DEP cell exports for its ranks.
TWO_LOCAL_RANKS = {"FPM_DATA_PARALLEL_SIZE": 2, "FPM_LOCAL_DATA_PARALLEL_SIZE": 2}


def _stage_two_ranks(tmp_path: Path) -> SimpleNamespace:
    """One staged pod of a cell with two data-parallel ranks on its one node, in the gate's mode without etcd."""
    staged = _stage_cell(tmp_path, env_overrides=TWO_LOCAL_RANKS)
    # The paths the gate derives for the two ranks (``benchmark_path_for_dp_rank``).
    staged.rank_paths = [str(staged.benchmark_output), str(staged.benchmark_output.with_name("benchmark_dp1.json"))]
    staged.merged_path = str(staged.benchmark_output.with_name("benchmark_merged.json"))
    return staged


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


def test_two_rank_files_and_the_merged_file_that_names_both_give_0(tmp_path):
    staged = _stage_two_ranks(tmp_path)
    files = _writer_files(staged)
    assert sorted(files) == sorted([*staged.rank_paths, staged.merged_path])
    # The FAKE launcher waits as long as the launcher does: ``launcher.FINAL_WAIT_SECONDS`` is its default.
    _fake_launcher(tmp_path, staged, files=files, exit_code=launcher.EXIT_COMPLETE)
    done = _run(staged)

    assert done.returncode == 0, done.stderr
    assert _results(staged) == ["benchmark.json", "benchmark_dp1.json", "benchmark_merged.json"]
    merged = json.loads(Path(staged.merged_path).read_text())
    assert (merged["rank_files"], merged["dp"]["source_ranks"]) == (staged.rank_paths, [0, 1])


# ---------------------------------------------------------------------------------------------------------------
# Without the switch
# ---------------------------------------------------------------------------------------------------------------


def test_without_the_switch_a_failing_preflight_still_ends_the_cell_with_its_status(tmp_path):
    staged = _stage_cell(tmp_path, switch=None, etcd_stub=True)
    # The stand-in waits for the trace of the etcd stub, so that the order of the two is not left to chance.
    (staged.workdir / "preflight.py").write_text(
        "import os, sys, time\n"
        "deadline = time.monotonic() + 30\n"
        f"while not os.path.exists({str(staged.etcd_trace)!r}) and time.monotonic() < deadline:\n"
        "    time.sleep(0.02)\n"
        "sys.exit(7)\n"
    )
    _fake_launcher(tmp_path, staged, files=_complete_files(staged), exit_code=0, ends_on_sigterm=True)
    done = _run(staged)

    assert done.returncode == 7
    assert staged.etcd_trace.exists(), "etcd is started before the preflight"
    assert not staged.engine_report.exists() and not staged.sglbench_preflight_ran.exists()
