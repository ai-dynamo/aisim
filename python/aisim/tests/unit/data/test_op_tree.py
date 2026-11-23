# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-tree op-latency loading."""

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisim_core.data.loaders import LoaderError
from aisim_core.data.op_tree import load_op_latency_tree, op_latency_entries, verify_op_latency_support

from .test_op_publish import _build, tree  # noqa: F401  (fixture reuse)

pytestmark = pytest.mark.unit


@pytest.fixture
def published(tree, tmp_path):  # noqa: F811
    out = tmp_path / "published"
    _build(tree, out)
    return tree, out


def test_load_reproduces_packaged_rows_and_lookup_files(published, tmp_path):
    systems, out = published
    loaded = load_op_latency_tree(out, tmp_path / "root", packaged_systems=None)
    assert loaded.entries == 2 and set(loaded.lookup_files) == {
        "perf_data_reuse_manifest.yaml",
        "query_versions.yaml",
    }
    for name in loaded.lookup_files:
        assert (loaded.root / name).read_bytes() == (systems / name).read_bytes()


def test_load_copies_packaged_specs_but_never_packaged_tables(published, tmp_path):
    systems, out = published
    (systems / "h200_sxm.yaml").write_text("gpu: {}\nnode: {}\n")
    loaded = load_op_latency_tree(out, tmp_path / "root", packaged_systems=systems)
    assert (loaded.root / "h200_sxm.yaml").is_file()
    assert (loaded.root / "perf_data_reuse_manifest.yaml").read_bytes() == (
        systems / "perf_data_reuse_manifest.yaml"
    ).read_bytes()


def test_selection_filters(published):
    _, out = published
    assert len(op_latency_entries(out)) == 2
    assert op_latency_entries(out, hardware=["l4"]) == []
    assert len(op_latency_entries(out, framework=["vllm"], hardware=["h200_sxm"])) == 2


def test_tampered_tree_is_rejected_and_leaves_no_root(published, tmp_path):
    _, out = published
    table = next((out / "data/op-latency/h200_sxm/gemm/vllm/1.0").glob("*.parquet"))
    pq.write_table(pa.table({"b": [1]}), table)
    with pytest.raises(ValueError, match="mismatch"):
        load_op_latency_tree(out, tmp_path / "root", packaged_systems=None)
    assert not (tmp_path / "root").exists()


def test_existing_destination_and_empty_selection_are_refused(published, tmp_path):
    _, out = published
    (tmp_path / "root").mkdir()
    with pytest.raises(FileExistsError):
        load_op_latency_tree(out, tmp_path / "root")
    with pytest.raises(LoaderError, match="no op-latency entries"):
        load_op_latency_tree(out, tmp_path / "other", hardware=["l4"])
    assert not (tmp_path / "other").exists()


@pytest.mark.parametrize("missing", ["query_versions", "perf_data_reuse_manifest", "both"])
def test_indexed_policy_must_pin_both_lookups_before_loading(published, tmp_path, missing):
    from .test_ensure_op_latency import _write_op_index

    _, out = published
    index = json.loads((out / "catalog/op-latency.json").read_bytes())
    index["support"] = [a for a in index["support"] if missing != "both" and missing not in a["path"]]
    _write_op_index(out, index)
    with pytest.raises(LoaderError, match="required lookup support"):
        verify_op_latency_support(out)
    with pytest.raises(LoaderError, match="required lookup support"):
        load_op_latency_tree(out, tmp_path / "root", packaged_systems=None)
    assert not (tmp_path / "root").exists()


def test_legacy_local_layout_explicitly_omits_the_support_field(published, tmp_path):
    from .test_ensure_op_latency import _write_op_index

    _, out = published
    index = json.loads((out / "catalog/op-latency.json").read_bytes())
    del index["support"]
    _write_op_index(out, index)
    loaded = load_op_latency_tree(out, tmp_path / "root", packaged_systems=None)
    assert set(loaded.lookup_files) == {"query_versions.yaml", "perf_data_reuse_manifest.yaml"}


def _support_declaration(root, change):
    dataset_path = root / "catalog/dataset.json"
    dataset = json.loads(dataset_path.read_bytes())
    reference = dataset["categories"]["op-latency"]["index"]
    path = root / reference["path"]
    index = json.loads(path.read_bytes())
    change(index)
    path.write_text(json.dumps(index))
    reference.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    dataset_path.write_text(json.dumps(dataset))


@pytest.mark.parametrize("name", ["perf_data_reuse_manifest", "query_versions"])
@pytest.mark.parametrize("declaration", ["empty", "partial"])
def test_declared_support_rejects_unpinned_lookups_in_validation_and_materialization(
    published, tmp_path, name, declaration
):
    from aisim_core.data.ensure import MissingDataError, _load_op_latency
    from aisim_core.data.fetch import Cache, Selection

    _, root = published
    relative = f"data/op-latency/{name}.parquet"

    def unpin(index):
        index["support"] = [] if declaration == "empty" else [a for a in index["support"] if a["path"] != relative]

    _support_declaration(root, unpin)
    with pytest.raises(LoaderError, match="required lookup support"):
        verify_op_latency_support(root)
    with pytest.raises(LoaderError, match="required lookup support"):
        load_op_latency_tree(root, tmp_path / "untrusted", packaged_systems=None)
    assert not (tmp_path / "untrusted").exists()
    with pytest.raises(MissingDataError, match="required lookup support"):
        _load_op_latency(
            root,
            op_latency_entries(root),
            {},
            Cache(tmp_path / "cache"),
            "local",
            Selection(category="op-latency"),
            str(root),
        )


def test_legacy_index_without_support_keeps_local_lookup_compatibility(published, tmp_path):
    systems, root = published
    _support_declaration(root, lambda index: index.pop("support"))
    verify_op_latency_support(root)
    loaded = load_op_latency_tree(root, tmp_path / "legacy", packaged_systems=None)
    for name in loaded.lookup_files:
        assert (loaded.root / name).read_bytes() == (systems / name).read_bytes()


@pytest.mark.parametrize("name", ["perf_data_reuse_manifest", "query_versions"])
def test_pinned_lookup_digest_is_still_required(published, tmp_path, name):
    _, root = published
    path = root / f"data/op-latency/{name}.parquet"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_op_latency_support(root)
    with pytest.raises(ValueError, match="integrity mismatch"):
        load_op_latency_tree(root, tmp_path / "tampered", packaged_systems=None)
    assert not (tmp_path / "tampered").exists()
