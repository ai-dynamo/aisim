# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset source selection reaches the native predictor, including saved init settings."""

import os
import shutil
from pathlib import Path

import pytest

from aisim.main import main
from aisim_core.data import ensure as ensure_mod
from aisim_core.sdk import rust_engine_step as res
from tests.unit.data.test_ensure_op_latency import native_dataset  # noqa: F401
from tests.unit.data.test_fetch import REPO, SHA, SHA2, FakeTransport
from tests.unit.data.test_loaders import _tree

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_dataset_environment(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith("AISIM_DATA"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AISIM_DATA_CACHE", str(tmp_path / "cache"))


def test_no_pin_does_not_touch_config(monkeypatch):
    monkeypatch.delenv("AISIM_DATASET", raising=False)
    monkeypatch.delenv("AISIM_DATA_LOCAL", raising=False)
    assert res._ensure_dataset({"system": "h200"}, None) is None


def test_pin_from_environment_reaches_ensure(monkeypatch, tmp_path):
    monkeypatch.setenv("AISIM_DATA_LOCAL", str(tmp_path))
    with pytest.raises(ensure_mod.MissingDataError):  # empty local root: precise error, no fallback
        res._ensure_dataset({"system": "h200", "backend": "vllm", "backend_version": "0.1", "model": "m"}, None)


def test_reentrancy_guard_skips(monkeypatch, tmp_path):
    monkeypatch.setenv("AISIM_DATA_LOCAL", str(tmp_path))
    with ensure_mod._guard():
        assert res._ensure_dataset({}, None) is None


@pytest.mark.parametrize("override", ["saved", "environment-repo", "environment-local", "explicit-pin"])
def test_init_source_reaches_native_prediction(native_dataset, monkeypatch, override):  # noqa: F811
    published, systems, config = native_dataset
    requested_sources = set()

    class Transport(FakeTransport):
        def open_file(self, repo, sha, path):
            requested_sources.add((repo, sha))
            return super().open_file(repo, sha, path)

    transport = Transport(_tree(published))
    monkeypatch.setattr("aisim_core.data.fetch.HfTransport", lambda: transport)
    assert main(["init", "--on-demand", "--repo", REPO, "--revision", SHA]) == 0
    assert not transport.opened  # init records the source without downloading it

    expected_revision, expected_sources = SHA, {(REPO, SHA)}
    pin = None
    if override in {"environment-repo", "explicit-pin"}:
        monkeypatch.setenv("AISIM_DATASET", f"other/dataset@{SHA2}")
        expected_revision, expected_sources = SHA2, {("other/dataset", SHA2)}
    if override in {"environment-local", "explicit-pin"}:
        if override == "environment-local":
            monkeypatch.setenv("AISIM_DATA_LOCAL", str(published))
        else:
            pin = ensure_mod.DataPin(local_root=published)
        expected_revision, expected_sources = f"local:{published}", set()

    shutil.rmtree(systems / "data")  # prediction must consume the published tables, not the producer's
    payload = config.to_dict()
    payload.pop("systems_paths")
    model = res.RustForwardPassPerfModel.best_available(payload, dataset=pin)

    assert model.dataset_revisions == {"op-latency": expected_revision}
    assert requested_sources == expected_sources
    selected = Path(model.diagnostics()["provenance"]["selected_systems_root"])
    assert selected.is_relative_to(Path(os.environ["AISIM_DATA_CACHE"]) / "materialized")
    assert (selected / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet").is_file()
    assert model.predict_prefill_latency(1, 4) >= 7.0  # the published 7 ms attention measurement is consumed
