# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject source files that a recorded Git revision does not contain."""

import subprocess
from pathlib import Path


def require_committed(repo, revision, paths):
    """Raise unless every path in `paths` is a file of `revision`'s tree.

    `git diff HEAD` ignores untracked and ignored files, so a hashed source
    inventory can hold files the revision does not. Paths are absolute or
    relative to `repo`.
    """
    repo = Path(repo).resolve()
    relative = sorted({(repo / path).resolve().relative_to(repo).as_posix() for path in paths})
    tree = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "-z", revision, "--", *relative],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    missing = sorted(set(relative) - set(filter(None, tree.split("\0"))))
    if missing:
        raise ValueError(f"source files absent from {revision}: {', '.join(missing[:5])}")
