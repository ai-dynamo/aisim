# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ensure_data with fake transports: cold cache, warm offline, threshold, precise errors."""

import hashlib
import json

import pytest

from aisim_core.data import ensure as ensure_mod
from aisim_core.data.ensure import (
    DataConflictError,
    DataPin,
    DataTooLargeError,
    MissingDataError,
    ensure_data,
    pin_from_environment,
)
from aisim_core.data.fetch import Cache
from aisim_core.data.loaders import FpmData, FpmLoader
from tests.unit.data.test_fetch import REPO, SHA, FakeTransport, build_repo

pytestmark = pytest.mark.unit

CONFIG = {"system": "h200", "backend": "vllm", "backend_version": "0.1", "model": "m1"}
PROFILE = {"path": "/materialized/profile.json", "sha256": "f" * 64}


@pytest.fixture(autouse=True)
def stub_prepare(monkeypatch):
    """Skip the Rust readiness check; the loader's verification still runs."""

    def load(self, root, manifest_path, *, output_dir=None, config=None):
        manifest = self.read_manifest(root, manifest_path, config)
        output_dir.mkdir(parents=True)
        prepared = {
            "systems_paths": [str(output_dir / "systems")],
            "estimator_config": {"op_level": {"execution_profile": PROFILE}},
        }
        (output_dir / "config.json").write_text(json.dumps(prepared))
        return FpmData(manifest, config=prepared)

    monkeypatch.setattr(FpmLoader, "load", load)


@pytest.fixture
def files(tmp_path):
    return build_repo(tmp_path)


def pin(tmp_path, **kw):
    return DataPin(repo=REPO, revision=SHA, cache_dir=tmp_path / "cache", **kw)


def test_no_pin_is_a_noop():
    out = ensure_data(CONFIG, None)
    assert out.config == CONFIG and out.datasets == {}


def test_cold_cache_downloads_wires_config_and_reports_revision(tmp_path, files):
    transport = FakeTransport(files)
    out = ensure_data(CONFIG, pin(tmp_path), transport=transport)
    assert out.revisions == {"fpm": SHA}
    assert out.datasets["fpm"]["source"] == "hf-download"
    assert out.datasets["fpm"]["downloaded_bytes"] > 0
    assert out.config["estimator_config"]["op_level"]["execution_profile"] == PROFILE
    assert out.config["systems_paths"][0].endswith("/systems")
    assert "fpm/m1/data.parquet" in transport.opened
    assert "fpm/m2/data.parquet" not in transport.opened  # selection excludes other models


def test_warm_cache_works_offline_with_zero_network(tmp_path, files):
    ensure_data(CONFIG, pin(tmp_path), transport=FakeTransport(files))
    cold = FakeTransport(files)
    out = ensure_data(CONFIG, pin(tmp_path, offline=True), transport=cold)
    assert cold.opened == [] and cold.resolves == 0
    assert out.datasets["fpm"]["source"] == "hf-cache"
    assert out.revisions == {"fpm": SHA}
    # Online but warm: still no transfer.
    warm = FakeTransport(files)
    ensure_data(CONFIG, pin(tmp_path), transport=warm)
    assert warm.opened == []


def test_offline_cold_cache_is_a_precise_error(tmp_path, files):
    with pytest.raises(MissingDataError, match=r"offline: .*not fully cached.*catalog/dataset.json") as e:
        ensure_data(CONFIG, pin(tmp_path, offline=True), transport=FakeTransport(files))
    assert e.value.to_dict()["error"] == "missing-data"


def test_over_threshold_refuses_with_size_and_command(tmp_path, files):
    transport = FakeTransport(files)
    with pytest.raises(DataTooLargeError) as e:
        ensure_data(CONFIG, pin(tmp_path, max_bytes=1), transport=transport)
    msg = str(e.value)
    assert str(e.value.uncached_bytes) in msg and "aisim data fetch --category fpm" in msg
    assert "--model m1" in msg and f"--revision {SHA}" in msg
    assert "fpm/m1/data.parquet" not in transport.opened  # nothing downloaded


def test_missing_data_is_precise_and_never_falls_back(tmp_path, files):
    with pytest.raises(MissingDataError, match=r"model=other.*system=h200|system=h200.*model=other"):
        ensure_data({**CONFIG, "model": "other"}, pin(tmp_path), transport=FakeTransport(files))


def test_local_override_needs_no_network_or_cache(tmp_path, files):
    root = tmp_path / "local"
    for path, data in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(data)
    out = ensure_data(CONFIG, DataPin(local_root=root, cache_dir=tmp_path / "cache"))
    assert out.datasets["fpm"]["source"] == "local" and out.revisions["fpm"] == f"local:{root}"
    with pytest.raises(MissingDataError, match="no fpm data"):
        ensure_data({**CONFIG, "model": "nope"}, DataPin(local_root=root, cache_dir=tmp_path / "cache"))


def test_corrupt_local_file_is_an_error(tmp_path, files):
    root = tmp_path / "local"
    for path, data in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(data if "m1/data" not in path else b"corrupt")
    with pytest.raises(MissingDataError, match="failed verification|unusable"):
        ensure_data(CONFIG, DataPin(local_root=root, cache_dir=tmp_path / "cache"))


def test_user_pinned_profile_conflict(tmp_path, files):
    config = {**CONFIG, "estimator_config": {"op_level": {"execution_profile": {"sha256": "0" * 64}}}}
    with pytest.raises(DataConflictError, match="pins execution profile"):
        ensure_data(config, pin(tmp_path), transport=FakeTransport(files))


def test_pin_from_environment():
    assert pin_from_environment({}) is None
    p = pin_from_environment({"AISIM_DATASET": f"{REPO}@{SHA}", "AISIM_DATA_MAX_BYTES": "5", "AISIM_DATA_OFFLINE": "1"})
    assert (p.repo, p.revision, p.max_bytes, p.offline) == (REPO, SHA, 5, True)
    with pytest.raises(ValueError):
        pin_from_environment({"AISIM_DATASET": "no-revision"})


def test_ensure_guard_blocks_recursion(tmp_path, files):
    seen = []
    real = FpmLoader.load

    def spy(self, *a, **k):
        seen.append(ensure_mod.ensuring())
        return real(self, *a, **k)

    FpmLoader.load = spy
    try:
        ensure_data(CONFIG, pin(tmp_path), transport=FakeTransport(files))
    finally:
        FpmLoader.load = real
    assert seen == [True] and not ensure_mod.ensuring()
    assert Cache(tmp_path / "cache").root.exists()


def test_check_manifest_accepts_fpm_execution_profile_only(tmp_path, files):
    from aisim_core.data import Manifest, check_manifest

    root = tmp_path / "r"
    for path, data in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(data)
    assert check_manifest(root, "fpm/m1/manifest.json").category == "fpm"
    doc = json.loads(files["fpm/m1/manifest.json"])
    assert Manifest.model_validate(doc).category == "fpm"


def test_the_over_budget_command_keeps_the_pinned_configuration_id(tmp_path, files):
    index = json.loads(files["catalog/fpm.json"])
    ids = [e["configuration_id"] for e in index["entries"]]
    assert len(ids) == 2  # both models match the hardware, so a command without the ID would fetch both
    with pytest.raises(DataTooLargeError) as e:
        ensure_data(CONFIG, pin(tmp_path, max_bytes=1, configuration_id=ids[0]), transport=FakeTransport(files))
    assert f"--configuration {ids[0]}" in str(e.value) and ids[1] not in str(e.value)


def test_an_fpm_dataset_republished_in_place_is_not_served_from_the_old_preparation(tmp_path):
    import pyarrow as pa

    from aisim_core.data.parquet import write_parquet
    from tests.unit.data.test_fetch import _repin

    files = build_repo(tmp_path)
    root = tmp_path / "local"

    def publish(files):
        for path, data in files.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_bytes(data)

    publish(files)
    local = DataPin(local_root=root, cache_dir=tmp_path / "cache")
    first = ensure_data(CONFIG, local).config["systems_paths"][0]
    assert ensure_data(CONFIG, local).config["systems_paths"][0] == first  # unchanged content is reused
    # Corrected measurements: same identity and configuration ID, different table bytes and manifest digests.
    table = write_parquet(pa.table({"x": [7, 8, 9, 10]}), tmp_path / "new.parquet", "aisim.test/v1").read_bytes()
    files["fpm/m1/data.parquet"] = table
    manifest = json.loads(files["fpm/m1/manifest.json"])
    manifest["artifacts"][0].update(sha256=hashlib.sha256(table).hexdigest(), bytes=len(table), rows=4)
    files["fpm/m1/manifest.json"] = json.dumps(manifest).encode()
    _repin(files, "fpm/m1/manifest.json")
    publish(files)
    assert ensure_data(CONFIG, local).config["systems_paths"][0] != first


def _roots(tmp_path, fpm_bytes, op_bytes):
    fpm, op = tmp_path / "fpm-root", tmp_path / "op-root"
    for root, data in ((fpm, fpm_bytes), (op, op_bytes)):
        table = root / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet"
        table.parent.mkdir(parents=True)
        table.write_bytes(data)
    return fpm, op


@pytest.mark.parametrize("categories", [("fpm", "op-latency"), ("op-latency", "fpm")])
def test_overlapping_fpm_and_op_tables_conflict_whatever_the_category_order(tmp_path, monkeypatch, categories):
    fpm, op = _roots(tmp_path, b"fpm copy", b"op copy")

    def fake_ensure_category(out, category, pin, transport):
        roots = {"fpm": [str(fpm)], "op-latency": [str(op)]}
        out.config["systems_paths"] = roots[category] if category == "op-latency" else [*roots[category], "user"]
        out.roots[category] = roots[category]
        out.datasets[category] = {"configuration_id": "cfg-fpm", "configuration_ids": ["cfg-op-1", "cfg-op-2"]}

    monkeypatch.setattr(ensure_mod, "_ensure_category", fake_ensure_category)
    with pytest.raises(DataConflictError, match=r"cfg-fpm.*cfg-op-1, cfg-op-2.*gemm_perf"):
        ensure_data(CONFIG, DataPin(local_root=tmp_path, categories=categories))


@pytest.mark.parametrize("categories", [("fpm", "op-latency"), ("op-latency", "fpm")])
def test_identical_overlap_gives_the_same_roots_in_either_order(tmp_path, monkeypatch, categories):
    fpm, op = _roots(tmp_path, b"same", b"same")

    def fake_ensure_category(out, category, pin, transport):
        roots = {"fpm": [str(fpm)], "op-latency": [str(op)]}
        out.config["systems_paths"] = roots[category] if category == "op-latency" else [*roots[category], "user"]
        out.roots[category] = roots[category]
        out.datasets[category] = {"configuration_id": "cfg-fpm", "configuration_ids": ["cfg-op"]}

    monkeypatch.setattr(ensure_mod, "_ensure_category", fake_ensure_category)
    out = ensure_data(CONFIG, DataPin(local_root=tmp_path, categories=categories))
    assert out.config["systems_paths"] == [str(fpm), str(op)]
