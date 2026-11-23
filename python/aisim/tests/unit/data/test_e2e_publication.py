# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publication, evidence revision and evaluator regressions for E2E datasets."""

import dataclasses
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aisim_core.data import e2e as module
from aisim_core.data.e2e import (
    ProvenanceFile,
    SnapshotAcquisition,
    SnapshotMember,
    aiperf_to_configuration,
    evaluation_points,
    load_e2e,
    make_metric,
    read_provenance,
    read_raw_record,
    read_snapshot,
    store_provenance,
    write_configuration,
    write_snapshot,
)
from tests.unit.data.test_e2e import _aiperf_records, _aiperf_summary, _dev_body, _dev_config, _snapshot_args

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("path", ["../escape.json", "/escape.json", "nested/../../escape.json"])
def test_snapshot_rejects_escaping_paths_before_writing(tmp_path, path):
    members, kwargs = _snapshot_args("o1")
    root = tmp_path / "dataset"
    with pytest.raises(ValueError, match="relative path"):
        write_snapshot(root, path, members, **kwargs)
    assert not root.exists()


@pytest.mark.parametrize("prefix", ["../escape", "/escape", "safe/../../escape"])
def test_configuration_rejects_escaping_prefix_before_writing(tmp_path, prefix):
    root = tmp_path / "dataset"
    with pytest.raises(ValueError, match="relative path"):
        write_configuration(root, _dev_config(), prefix=prefix)
    assert not root.exists()


@pytest.mark.parametrize("name", ["../../../../escape", "/escape", "safe/../../../escape"])
def test_provenance_rejects_escaping_names_before_writing(tmp_path, name):
    root = tmp_path / "dataset"
    with pytest.raises(ValueError, match="relative path"):
        store_provenance(root, "valid", ProvenanceFile("raw-output", b"raw", name=name))
    assert not root.exists()


@pytest.mark.parametrize("operation", ["configuration", "snapshot", "provenance"])
def test_output_paths_refuse_existing_symlink_parents(tmp_path, operation):
    root, outside = tmp_path / "dataset", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "e2e").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        if operation == "configuration":
            write_configuration(root, _dev_config())
        elif operation == "snapshot":
            members, kwargs = _snapshot_args("o1")
            write_snapshot(root, "e2e/snap.json", members, **kwargs)
        else:
            store_provenance(root, "e2e", ProvenanceFile("raw-output", b"raw"))
    assert list(outside.iterdir()) == []


def test_publication_rechecks_parent_components_and_nested_paths_work(tmp_path, monkeypatch):
    root, outside = tmp_path / "dataset", tmp_path / "outside"
    outside.mkdir()
    real_publish = module._publish_staged

    def redirect_after_staging(stage, destination, relatives):
        (destination / "nested").symlink_to(outside, target_is_directory=True)
        return real_publish(stage, destination, relatives)

    monkeypatch.setattr(module, "_publish_staged", redirect_after_staging)
    with pytest.raises(ValueError, match="symlink"):
        write_configuration(root, _dev_config(), prefix="nested/data")
    assert list(outside.iterdir()) == []
    (root / "nested").unlink()
    monkeypatch.setattr(module, "_publish_staged", real_publish)
    path = write_configuration(root, _dev_config(), prefix="nested/data")
    assert load_e2e(root, path).observations[0].observation_id == "dev:1"
    members, kwargs = _snapshot_args("o1")
    write_snapshot(root, "nested/snapshots/a.json", members, **kwargs)
    assert read_snapshot(root, "nested/snapshots/a.json")[1] == members
    artifact, _ = store_provenance(root, "nested", ProvenanceFile("raw-output", b"raw", name="valid/raw"))
    assert (root / artifact.path).is_file()


def test_readers_keep_the_old_configuration_during_and_after_publication(tmp_path, monkeypatch):
    original = _dev_config()
    path = write_configuration(tmp_path, original)
    (tmp_path / "pinned.json").write_bytes((tmp_path / path).read_bytes())
    old_tokens = original.observations[0].input_tokens
    real_replace = module.os.replace
    reads = []

    def interleaved_replace(src, dst):
        real_replace(src, dst)
        if str(dst).endswith(".parquet"):
            reads.append(load_e2e(tmp_path, path).observations[0].input_tokens)

    monkeypatch.setattr(module.os, "replace", interleaved_replace)
    changed = original.observations[0].model_copy(update={"input_tokens": 999})
    assert write_configuration(tmp_path, dataclasses.replace(original, observations=[changed])) == path
    assert reads and set(reads) == {old_tokens}
    assert load_e2e(tmp_path, path).observations[0].input_tokens == 999
    assert load_e2e(tmp_path, "pinned.json").observations[0].input_tokens == old_tokens


def test_snapshot_readers_keep_their_members_during_and_after_publication(tmp_path, monkeypatch):
    members, kwargs = _snapshot_args("old")
    write_snapshot(tmp_path, "snapshot.json", members, **kwargs)
    (tmp_path / "pinned.json").write_bytes((tmp_path / "snapshot.json").read_bytes())
    new_members, _ = _snapshot_args("new")
    real_replace = module.os.replace
    reads = []

    def interleaved_replace(src, dst):
        real_replace(src, dst)
        if str(dst).endswith(".parquet"):
            reads.append(read_snapshot(tmp_path, "snapshot.json")[1])

    monkeypatch.setattr(module.os, "replace", interleaved_replace)
    write_snapshot(tmp_path, "snapshot.json", new_members, **kwargs)
    assert reads and all(rows == members for rows in reads)
    assert read_snapshot(tmp_path, "snapshot.json")[1] == new_members
    assert read_snapshot(tmp_path, "pinned.json")[1] == members


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_rollback_attempts_all_successful_replacements_and_retains_recovery(tmp_path, monkeypatch, rollback_fails):
    root, stage = tmp_path / "root", tmp_path / "stage"
    root.mkdir()
    stage.mkdir()
    for name in ("a", "b", "c"):
        (root / name).write_text("old")
        (stage / name).write_text("new")
    real_replace = module.os.replace
    fail_at = "c" if rollback_fails else "b"

    def fail_replace(src, dst):
        if dst == root / fail_at:
            raise OSError("destination refuses renames")
        if rollback_fails and dst == root / "b" and ".previous" in Path(src).parts:
            raise OSError("rollback also failed")
        return real_replace(src, dst)

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="recovery files retained" if rollback_fails else "refuses renames"):
        module._publish_staged(stage, root, ["a", "b", "c"])
    assert (root / "a").read_text() == "old"
    assert (root / "c").read_text() == "old"
    if rollback_fails:
        assert (stage / ".previous/b").read_text() == "old"
    else:
        assert (root / "b").read_text() == "old"


def test_configuration_writer_keeps_the_recovery_directory_when_rollback_fails(tmp_path, monkeypatch):
    path = write_configuration(tmp_path, _dev_config())
    previous = (tmp_path / path).read_bytes()
    real_replace, real_unlink = module.os.replace, Path.unlink

    def fail_manifest(src, dst):
        if dst == tmp_path / path:
            raise OSError("cannot publish manifest")
        return real_replace(src, dst)

    def fail_one_cleanup(destination, *args, **kwargs):
        if destination.name == "observations.parquet":
            raise OSError("cannot remove new table")
        return real_unlink(destination, *args, **kwargs)

    monkeypatch.setattr(module.os, "replace", fail_manifest)
    monkeypatch.setattr(Path, "unlink", fail_one_cleanup)
    with pytest.raises(OSError, match="recovery files retained") as error:
        write_configuration(tmp_path, _dev_config())
    (stage,) = list(tmp_path.glob(".e2e-stage-*"))
    assert str(stage) in str(error.value)
    assert (stage / ".previous" / path).read_bytes() == previous
    assert load_e2e(tmp_path, path).observations[0].observation_id == "dev:1"


@pytest.mark.parametrize("reverse", [False, True])
def test_per_device_throughput_rejects_conflicting_normalizations(tmp_path, reverse):
    dataset = load_e2e(tmp_path, write_configuration(tmp_path, _dev_config()))
    rows = [
        make_metric("dev:1", "output_throughput", "value", 400, "tokens/s", normalization="total", device_count=4),
        make_metric("dev:1", "output_throughput", "value", 110, "tokens/s", normalization="per_device"),
    ]
    if reverse:
        rows.reverse()
    with pytest.raises(ValueError, match="ambiguous per-device throughput"):
        evaluation_points(dataclasses.replace(dataset, metrics=rows))


@pytest.mark.parametrize("reverse", [False, True])
def test_per_device_throughput_selects_value_then_mean_deterministically(tmp_path, reverse):
    dataset = load_e2e(tmp_path, write_configuration(tmp_path, _dev_config()))
    rows = [
        make_metric("dev:1", "output_throughput", stat, value, "tokens/s", normalization="per_device")
        for stat, value in [("value", 100), ("mean", 110), ("median", 120)]
    ]
    if reverse:
        rows.reverse()
    (point,) = evaluation_points(dataclasses.replace(dataset, metrics=rows))
    assert point.per_device_throughput["output_throughput"] == 100
    assert len(point.metrics) == 3
    (point,) = evaluation_points(dataclasses.replace(dataset, metrics=[r for r in rows if r.statistic != "value"]))
    assert point.per_device_throughput["output_throughput"] == 110


def test_aiperf_request_corrections_change_snapshot_revision_and_preserve_raw_audit(tmp_path):
    original = _dev_config()
    records = _aiperf_records()
    records[0]["metrics"]["time_to_first_token"]["value"] += 5
    corrected = aiperf_to_configuration(_aiperf_summary(), records, _dev_body(), observation_id="dev:1", device_count=8)
    before, after = original.observations[0], corrected.observations[0]
    assert before.observation_id == after.observation_id and before.record_sha256 != after.record_sha256
    assert before.raw_record_sha256 == after.raw_record_sha256
    ids = []
    for index, data in enumerate((original, corrected)):
        root = tmp_path / str(index)
        path = write_configuration(root, data)
        dataset = load_e2e(root, path)
        assert json.loads(read_raw_record(root, dataset, "dev:1")) == _aiperf_summary()
        member = SnapshotMember(
            configuration_id=before.configuration_id,
            observation_id="dev:1",
            record_sha256=data.observations[0].record_sha256,
            manifest_path=path,
        )
        ids.append(
            write_snapshot(
                root, "snapshot.json", [member], source_scope={}, selection_policy="p", completeness="complete"
            ).snapshot_id
        )
    assert ids[0] != ids[1]


def _replace_after_hashing(monkeypatch, path, replacement):
    """Replace ``path`` right after the integrity check hashed its current bytes, as a concurrent refresh could."""
    import aisim_core.data as data_module

    original = path.read_bytes()

    def sha256(content=b"", **kwargs):  # a check that hashes bytes it already holds
        digest = hashlib.sha256(content, **kwargs)
        if content == original:
            path.write_bytes(replacement)
        return digest

    def file_digest(stream, *args, **kwargs):  # a check that hashes the open file
        digest = hashlib.file_digest(stream, *args, **kwargs)
        if Path(getattr(stream, "name", "")) == path:
            path.write_bytes(replacement)
        return digest

    spy = SimpleNamespace(**{**vars(hashlib), "sha256": sha256, "file_digest": file_digest})
    monkeypatch.setattr(data_module, "hashlib", spy)


@pytest.mark.parametrize("table", ["members", "acquisition"])
def test_snapshot_parses_the_tables_that_passed_verification(tmp_path, monkeypatch, table):
    members, kwargs = _snapshot_args("o1")
    acquisition = [SnapshotAcquisition(observation_id="o1", source_uri="https://benchmark.example/data")]
    snapshot = write_snapshot(tmp_path / "live", "snapshot.json", members, **kwargs, acquisition=acquisition)
    # Another valid table with the same schema: a refresh that lands between the digest check and the parse.
    other_members, _ = _snapshot_args("o2")
    other_acquisition = [SnapshotAcquisition(observation_id="o2", source_uri="https://other.example/data")]
    other = write_snapshot(tmp_path / "other", "snapshot.json", other_members, **kwargs, acquisition=other_acquisition)
    live_table = getattr(snapshot, table).path
    replacement = (tmp_path / "other" / getattr(other, table).path).read_bytes()
    _replace_after_hashing(monkeypatch, tmp_path / "live" / live_table, replacement)
    loaded, rows = read_snapshot(tmp_path / "live", "snapshot.json")
    assert (loaded, rows) == (snapshot, members)  # the verified bytes were parsed, not the replacement


def test_provenance_returns_the_bytes_that_passed_verification(tmp_path, monkeypatch):
    dataset = load_e2e(tmp_path, write_configuration(tmp_path, _dev_config()))
    artifact = next(a for a in dataset.manifest.artifacts if a.format == "gzip")
    expected = read_provenance(tmp_path, artifact)
    _replace_after_hashing(monkeypatch, tmp_path / artifact.path, gzip.compress(b"tampered after the check"))
    assert read_provenance(tmp_path, artifact) == expected
