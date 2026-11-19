# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ensure_data with fake transports: cold cache, warm offline, threshold, precise errors."""

import json
from pathlib import Path

import pytest

from aisim_core.data.ensure import (
    DataPin,
    DataTooLargeError,
    MissingDataError,
    ensure_data,
    pin_from_environment,
)
from tests.unit.data.test_fetch import REPO, SHA, FakeTransport, build_repo

pytestmark = pytest.mark.unit

CONFIG = {"system": "h200", "backend": "vllm", "backend_version": "0.1", "model": "m1"}


@pytest.fixture
def files(tmp_path):
    return build_repo(tmp_path)


def pin(tmp_path, **kw):
    return DataPin(repo=REPO, revision=SHA, cache_dir=tmp_path / "cache", **kw)


def test_no_pin_is_a_noop():
    out = ensure_data(CONFIG, None)
    assert out.config == CONFIG and out.datasets == {}


def test_offline_cold_cache_is_a_precise_error(tmp_path, files):
    with pytest.raises(MissingDataError, match=r"offline: .*not fully cached.*catalog/dataset.json") as e:
        ensure_data(CONFIG, pin(tmp_path, offline=True), transport=FakeTransport(files))
    assert e.value.to_dict()["error"] == "missing-data"


def test_over_threshold_refuses_with_size_and_command(tmp_path, files):
    transport = FakeTransport(files)
    with pytest.raises(DataTooLargeError) as e:
        ensure_data(CONFIG, pin(tmp_path, max_bytes=1, categories=("fpm",)), transport=transport)
    msg = str(e.value)
    assert str(e.value.uncached_bytes) in msg and "aisim data fetch --category fpm" in msg
    assert "--model m1" in msg and f"--revision {SHA}" in msg
    assert "data/fpm/m1/data.parquet" not in transport.opened  # nothing downloaded


def test_missing_data_is_precise_and_never_falls_back(tmp_path, files):
    with pytest.raises(MissingDataError, match=r"model=other.*system=h200|system=h200.*model=other"):
        ensure_data({**CONFIG, "model": "other"}, pin(tmp_path, categories=("fpm",)), transport=FakeTransport(files))


def test_local_override_selection_needs_no_network_or_cache(tmp_path, files):
    root = tmp_path / "local"
    for path, data in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(data)
    with pytest.raises(MissingDataError, match="no fpm data"):
        ensure_data(
            {**CONFIG, "model": "nope"}, DataPin(local_root=root, categories=("fpm",), cache_dir=tmp_path / "cache")
        )


def test_pin_from_environment():
    assert pin_from_environment({}) is None
    p = pin_from_environment({"AISIM_DATASET": f"{REPO}@{SHA}", "AISIM_DATA_MAX_BYTES": "5", "AISIM_DATA_OFFLINE": "1"})
    assert (p.repo, p.revision, p.max_bytes, p.offline) == (REPO, SHA, 5, True)
    with pytest.raises(ValueError):
        pin_from_environment({"AISIM_DATASET": "no-revision"})


def test_check_manifest_accepts_fpm_execution_profile_only(tmp_path, files):
    from aisim_core.data import Manifest, check_manifest

    root = tmp_path / "r"
    for path, data in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(data)
    assert check_manifest(root, "data/fpm/m1/manifest.json").category == "fpm"
    doc = json.loads(files["data/fpm/m1/manifest.json"])
    assert Manifest.model_validate(doc).category == "fpm"


def test_default_pin_names_only_the_op_database():
    assert DataPin(local_root=Path("x")).categories == ("op-latency",)
