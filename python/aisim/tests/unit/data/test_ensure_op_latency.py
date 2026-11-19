# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The op-latency pin composes every operation family with the system specs into one verified root."""

import shutil

import pyarrow.parquet as pq
import pytest

from aisim_core.data.ensure import DataConflictError, DataPin, MissingDataError, ensure_data
from aisim_core.sdk import rust_engine_step as res

from .test_op_publish import _build, _table, tree  # noqa: F401  (fixture reuse)

pytestmark = pytest.mark.unit

CONFIG = {"system": "h200_sxm", "backend": "vllm", "backend_version": "1.0", "model": "any/model"}


@pytest.fixture
def two_families(tree, tmp_path):  # noqa: F811
    """gemm and attention at one hardware/backend/version, as the publisher emits them."""
    v1 = tree / "data/h200_sxm/gemm/vllm/1.0"
    attention = tree / "data/h200_sxm/attention/vllm/1.0"
    attention.mkdir(parents=True)
    pq.write_table(_table(3), attention / "attention_perf.parquet")
    shutil.copyfile(v1 / "collection_meta.yaml", attention / "collection_meta.yaml")
    out = tmp_path / "published"
    _build(tree, out)
    return out


def _pin(root, tmp_path, **kw):
    return DataPin(local_root=root, categories=("op-latency",), cache_dir=tmp_path / "cache", **kw)


def test_all_families_and_the_system_spec_arrive_in_one_root(two_families, tmp_path):
    out = ensure_data(dict(CONFIG), _pin(two_families, tmp_path))
    root = out.config["systems_paths"][0]
    families = {p.parent.parent.parent.name for p in (tmp_path / "cache").rglob("*_perf.parquet")}
    assert {"gemm", "attention"} <= families
    from pathlib import Path

    assert (Path(root) / "h200_sxm.yaml").is_file(), "the native database needs the system spec beside its tables"
    assert (Path(root) / "data/h200_sxm/attention/vllm/1.0").is_dir()
    assert len(out.datasets["op-latency"]["configuration_ids"]) == 2


def test_the_best_available_consumer_receives_the_complete_root(two_families, tmp_path):
    from pathlib import Path

    ensured = res._ensure_dataset(dict(CONFIG), _pin(two_families, tmp_path))
    first = Path(ensured.config["systems_paths"][0])
    assert (first / "h200_sxm.yaml").is_file() and (first / "data/h200_sxm/gemm/vllm/1.0").is_dir()


def test_second_call_reuses_the_materialized_root(two_families, tmp_path):
    first = ensure_data(dict(CONFIG), _pin(two_families, tmp_path)).config["systems_paths"][0]
    again = ensure_data(dict(CONFIG), _pin(two_families, tmp_path)).config["systems_paths"][0]
    assert first == again


def test_two_entries_for_one_family_are_a_real_conflict(two_families, tmp_path, monkeypatch):
    from aisim_core.data import ensure as mod

    real = mod._matching_entries
    monkeypatch.setattr(mod, "_matching_entries", lambda root, sel: [*real(root, sel), real(root, sel)[0]])
    with pytest.raises(DataConflictError, match="one operation family"):
        ensure_data(dict(CONFIG), _pin(two_families, tmp_path))


def test_configuration_id_still_selects_one_entry(two_families, tmp_path):
    from aisim_core.data.ensure import _matching_entries
    from aisim_core.data.loaders import OpLatencyLoader

    entries = _matching_entries(two_families, OpLatencyLoader().select(CONFIG))
    out = ensure_data(dict(CONFIG), _pin(two_families, tmp_path, configuration_id=entries[0].configuration_id))
    assert out.datasets["op-latency"]["configuration_ids"] == [entries[0].configuration_id]


def test_unpublished_version_has_no_fallback(two_families, tmp_path):
    with pytest.raises(MissingDataError):
        ensure_data({**CONFIG, "backend_version": "9.9"}, _pin(two_families, tmp_path))
