# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An extra source file, untracked or ignored, must fail the revision check."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from committed_sources import require_committed


def _repo(tmp_path):
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "a.py").write_text("A = 1\n")
    (tmp_path / ".gitignore").write_text("pkg/ignored.py\n")
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "init"], check=True)
    return package


@pytest.mark.parametrize("name", ["untracked.py", "ignored.py"])
def test_extra_source_is_rejected(tmp_path, name):
    package = _repo(tmp_path)
    require_committed(tmp_path, "HEAD", list(package.rglob("*.py")))
    (package / name).write_text("B = 2\n")
    with pytest.raises(ValueError, match=f"pkg/{name}"):
        require_committed(tmp_path, "HEAD", list(package.rglob("*.py")))
