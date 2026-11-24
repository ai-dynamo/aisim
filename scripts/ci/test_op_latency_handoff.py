# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the op-latency dataset hand-off between CI jobs."""

import io
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import op_latency_handoff as h


def _cache(root: Path) -> Path:
    data = root / "ai-dynamo__aisim-data" / ("a" * 40) / "data" / "op-latency" / "b200_sxm"
    data.mkdir(parents=True)
    (data / "gemm_perf.parquet").write_bytes(b"table bytes")
    (data.parent / "manifest.json").write_text("{}")
    (root / "systems" / "composed").mkdir(parents=True)  # a composed root must not travel
    (root / "systems" / "composed" / "b200_sxm.yaml").write_text("x: 1")
    (root / "config.json").write_text("{}")  # nor the saved init choice
    return root


def test_pack_archives_only_the_downloaded_dataset(tmp_path, capsys):
    source = _cache(tmp_path / "source")
    archive = tmp_path / "out" / "data.tar"
    assert h.pack(archive, download=False, root=source) == 0
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert any(n.endswith("gemm_perf.parquet") for n in names)
    assert not any(n.startswith("systems") or n == "config.json" for n in names)
    assert "packed 2 files" in capsys.readouterr().out


def test_unpack_restores_identical_files_and_replaces_existing_ones(tmp_path):
    source = _cache(tmp_path / "source")
    archive = tmp_path / "data.tar"
    h.pack(archive, download=False, root=source)
    target = tmp_path / "target"
    stale = target / "ai-dynamo__aisim-data" / ("a" * 40) / "data" / "op-latency" / "b200_sxm" / "gemm_perf.parquet"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    assert h.unpack(archive, root=target) == 0
    assert stale.read_bytes() == b"table bytes"
    assert (stale.parent.parent / "manifest.json").read_text() == "{}"
    assert not (target / "systems").exists()


def test_pack_without_a_downloaded_dataset_fails(tmp_path, capsys):
    assert h.pack(tmp_path / "data.tar", download=False, root=tmp_path / "empty") == 1
    assert "no downloaded dataset" in capsys.readouterr().err


@pytest.mark.parametrize(
    "name", ["../escape.txt", "/abs/escape.txt", "ai-dynamo__x/../../escape.txt", "systems/composed/b200_sxm.yaml"]
)
def test_unpack_refuses_members_that_could_leave_the_cache(tmp_path, name):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo(name)
        info.size = 3
        tar.addfile(info, io.BytesIO(b"bad"))
    root = tmp_path / "cache"
    assert h.unpack(archive, root=root) == 1
    assert not (tmp_path / "escape.txt").exists() and not list(root.rglob("*.txt"))


def test_unpack_refuses_links(tmp_path):
    archive = tmp_path / "link.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("ai-dynamo__aisim-data/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    assert h.unpack(archive, root=tmp_path / "cache") == 1


def test_main_routes_the_commands(tmp_path, monkeypatch):
    # The cache root comes from the data library (``AISIM_DATA_CACHE``); stand in for it here.
    root = {"path": _cache(tmp_path / "source")}
    monkeypatch.setattr(h, "cache_root", lambda: root["path"])
    archive = tmp_path / "data.tar"
    assert h.main(["pack", "--output", str(archive), "--no-download"]) == 0
    root["path"] = tmp_path / "other"
    assert h.main(["unpack", str(archive)]) == 0
    assert (tmp_path / "other" / "ai-dynamo__aisim-data").is_dir()
