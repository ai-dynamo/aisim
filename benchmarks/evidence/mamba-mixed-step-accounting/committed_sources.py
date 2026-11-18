# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject source files that a recorded Git revision does not contain."""

import os
import subprocess
from pathlib import Path


def require_committed(repo, revision, paths):
    """Raise unless every path in `paths` is a file of `revision`'s tree.

    `git diff HEAD` ignores untracked and ignored files, so a hashed source
    inventory can hold files the revision does not. Paths are absolute or
    relative to `repo`. Each path's own name is checked, without following
    symlinks: an uncommitted link to a committed file is still uncommitted.
    """
    repo = os.path.abspath(repo)
    relative = sorted(
        {Path(os.path.relpath(os.path.abspath(os.path.join(repo, path)), repo)).as_posix() for path in paths}
    )
    tree = subprocess.run(
        ["git", "-C", repo, "ls-tree", "-r", "--name-only", "-z", revision, "--", *relative],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    missing = sorted(set(relative) - set(filter(None, tree.split("\0"))))
    if missing:
        raise ValueError(f"source files absent from {revision}: {', '.join(missing[:5])}")
