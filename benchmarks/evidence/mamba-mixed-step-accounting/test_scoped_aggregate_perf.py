# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate evidence must reject stale sources and free-form revision labels."""

import subprocess

import pytest
from scoped_aggregate_perf import verify_source


def test_backend_source_matches_pinned_git_blob(tmp_path, monkeypatch):
    source = tmp_path / "python/aisim/src/aisim_core/sdk/backends/base_backend.py"
    source.parent.mkdir(parents=True)
    source.write_text("class BaseBackend: pass\n")
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "fixture"], check=True)
    monkeypatch.chdir(tmp_path)
    revision = verify_source(source, "HEAD")
    assert len(revision) == 40
    source.write_text("class BaseBackend: edited = True\n")
    with pytest.raises(ValueError, match="backend source .* does not match"):
        verify_source(source, revision)
    with pytest.raises(subprocess.CalledProcessError):
        verify_source(source, "missing-revision")
