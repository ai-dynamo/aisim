#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hand the downloaded op-latency dataset from one CI job to the others.

Every Full CI job that reads operation tables needs the pinned ``ai-dynamo/aisim-data`` revision, thousands of
files. Downloading it in every parallel job gets the runners rate limited by Hugging Face (HTTP 429) and the jobs
time out. One job downloads it once (``pack``) and uploads the archive as a workflow artifact; the others
restore it (``unpack``) and then compose their systems root from the warm cache::

    python scripts/ci/op_latency_handoff.py pack --output op-latency-data.tar
    python scripts/ci/op_latency_handoff.py unpack op-latency-data.tar

The archive holds only the downloaded dataset files (``<owner>__<name>/<sha>/...``), never a composed systems root,
so a job still verifies every file's hash while composing. The cache root is ``AISIM_DATA_CACHE`` or the user cache
directory, as everywhere else.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "aisim" / "src"))

COMPOSED_ROOTS = "systems"  # composed roots are rebuilt by every job, never handed over


def cache_root() -> Path:
    from aisim_core.data.fetch import default_cache_dir

    return Path(default_cache_dir())


def dataset_directories(root: Path) -> list[Path]:
    """The downloaded dataset directories (``<owner>__<name>``) under a cache root."""
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and "__" in p.name and p.name != COMPOSED_ROOTS)


def pack(output: Path, *, download: bool = True, root: Path | None = None) -> int:
    """Download and verify the pinned dataset (unless ``download`` is false), then archive the cache files."""
    if download:
        from aisim_core.data.default_systems import default_systems_path

        default_systems_path()  # fetches every op-latency entry into the cache and verifies it
    root = cache_root() if root is None else root
    directories = dataset_directories(root)
    if not directories:
        print(f"no downloaded dataset under {root}", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(output, "w") as archive:  # parquet is already compressed
        for directory in directories:
            archive.add(directory, arcname=directory.name)
            count += sum(1 for p in directory.rglob("*") if p.is_file())
    print(f"packed {count} files from {root} into {output} ({output.stat().st_size / 2**20:.1f} MiB)")
    return 0


def _safe(member: tarfile.TarInfo) -> bool:
    """Plain files and directories under a relative path with no ``..``: nothing that can escape the cache root."""
    path = PurePosixPath(member.name)
    return (
        (member.isfile() or member.isdir())
        and not path.is_absolute()
        and ".." not in path.parts
        and bool(path.parts)
        and path.parts[0] != COMPOSED_ROOTS
    )


def unpack(archive: Path, *, root: Path | None = None) -> int:
    """Extract an archive made by :func:`pack` into the cache root, replacing files of the same name."""
    root = cache_root() if root is None else root
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r") as tar:
        members = tar.getmembers()
        unsafe = [m.name for m in members if not _safe(m)]
        if unsafe:
            print(f"refusing an archive with unexpected members: {unsafe[:5]}", file=sys.stderr)
            return 1
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(root, members=members, **kwargs)
    files = sum(1 for m in members if m.isfile())
    print(f"restored {files} files into {root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    pack_parser = commands.add_parser("pack", help="download the pinned dataset and archive it")
    pack_parser.add_argument("--output", type=Path, required=True)
    pack_parser.add_argument("--no-download", action="store_true", help="archive the cache as it is")
    unpack_parser = commands.add_parser("unpack", help="restore an archive into the cache")
    unpack_parser.add_argument("archive", type=Path)
    args = parser.parse_args(argv)
    if args.command == "pack":
        return pack(args.output, download=not args.no_download)
    return unpack(args.archive)


if __name__ == "__main__":
    raise SystemExit(main())
