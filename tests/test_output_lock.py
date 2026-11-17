# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One AISim run owns an output directory at a time."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import aisim.output as output
from aisim.output import (
    OUTPUT_LOCK_ENVIRONMENT,
    lock_output_directory,
    prepare_output_directory,
    release_output_directory_lock,
)

_HOLDER = textwrap.dedent(
    """
    import sys
    from aisim.output import prepare_output_directory

    prepare_output_directory(sys.argv[1], overwrite=False)
    print("locked", flush=True)
    sys.stdin.readline()
    """
)


@pytest.fixture(autouse=True)
def _isolated_locks(monkeypatch):
    monkeypatch.delenv(OUTPUT_LOCK_ENVIRONMENT, raising=False)
    before = set(output._HELD_LOCKS)
    yield
    for key in set(output._HELD_LOCKS) - before:
        release_output_directory_lock(key)


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop(OUTPUT_LOCK_ENVIRONMENT, None)
    return env


class _Holder:
    """Another process that has prepared, and so locked, an output directory."""

    def __init__(self, root: Path) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-c", _HOLDER, str(root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=_child_env(),
        )
        assert self.process.stdout.readline().strip() == "locked"

    def release(self) -> None:
        self.process.stdin.write("\n")
        self.process.stdin.flush()
        assert self.process.wait(timeout=30) == 0

    def __enter__(self) -> _Holder:
        return self

    def __exit__(self, *exc) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=30)
        self.process.stdin.close()
        self.process.stdout.close()


def test_second_process_fails_fast_while_a_run_holds_the_directory(tmp_path) -> None:
    root = tmp_path / "out"
    with _Holder(root) as holder:
        (root / "prediction.json").write_text("{}\n")
        with pytest.raises(
            ValueError, match="another AISim run is using output directory"
        ):
            prepare_output_directory(root, overwrite=True)
        # The refused run removed nothing, even with --overwrite.
        assert (root / "prediction.json").read_text() == "{}\n"
        holder.release()
    prepare_output_directory(root, overwrite=True)
    assert not (root / "prediction.json").exists()


def test_lock_blocks_failed_run_cleanup_by_another_process(tmp_path) -> None:
    root = tmp_path / "out"
    root.mkdir()
    lock_output_directory(root)
    (root / "execution-events.jsonl").write_text("")
    (root / "resource-runtime.json").write_text('{"status": "failed"}\n')
    script = textwrap.dedent(
        """
        import sys
        from aisim.output import prepare_output_directory
        try:
            prepare_output_directory(sys.argv[1], overwrite=False)
        except ValueError as exc:
            print(exc)
            sys.exit(2)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        capture_output=True,
        text=True,
        env=_child_env(),
        timeout=60,
    )
    assert result.returncode == 2
    assert "another AISim run is using output directory" in result.stdout
    assert sorted(entry.name for entry in root.iterdir()) == [
        "execution-events.jsonl",
        "resource-runtime.json",
    ]


def test_lock_is_reentrant_within_a_process(tmp_path) -> None:
    root = tmp_path / "out"
    prepare_output_directory(root, overwrite=False)
    prepare_output_directory(root, overwrite=True)
    assert output._HELD_LOCKS[os.path.realpath(root)] >= 0


def test_worker_of_the_holding_supervisor_does_not_relock(tmp_path) -> None:
    root = tmp_path / "out"
    prepare_output_directory(root, overwrite=False)
    assert os.environ[OUTPUT_LOCK_ENVIRONMENT] == os.path.realpath(root)
    # A worker inherits the environment and writes under its parent's lock.
    script = textwrap.dedent(
        """
        import sys
        from aisim import output
        output.prepare_output_directory(sys.argv[1], overwrite=False)
        assert not output._HELD_LOCKS
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_symlinked_path_shares_the_lock(tmp_path) -> None:
    root = tmp_path / "out"
    alias = tmp_path / "alias"
    with _Holder(root):
        alias.symlink_to(root, target_is_directory=True)
        with pytest.raises(ValueError, match="another AISim run"):
            prepare_output_directory(alias, overwrite=False)


def test_lock_adds_no_file_to_the_directory(tmp_path) -> None:
    root = tmp_path / "out"
    prepare_output_directory(root, overwrite=False)
    assert list(root.iterdir()) == []


def test_release_lets_another_process_take_the_directory(tmp_path) -> None:
    root = tmp_path / "out"
    prepare_output_directory(root, overwrite=False)
    release_output_directory_lock(root)
    assert OUTPUT_LOCK_ENVIRONMENT not in os.environ
    with _Holder(root) as holder:
        holder.release()
