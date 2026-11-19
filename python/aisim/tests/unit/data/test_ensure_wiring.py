# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""best_available wiring is inert without a pin and fails loudly with one."""

import pytest

from aisim_core.data import ensure as ensure_mod
from aisim_core.sdk import rust_engine_step as res

pytestmark = pytest.mark.unit


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


def test_snapshot_bearing_config_skips_the_pin(monkeypatch, tmp_path):
    monkeypatch.setenv("AISIM_DATA_LOCAL", str(tmp_path))  # would raise MissingDataError for an ordinary config
    config = {"system": "h200", "estimator_config": {"state_snapshot": {"path": str(tmp_path)}}}
    assert res._ensure_dataset(config, None) is None


def test_every_instance_has_dataset_revisions():
    assert res.RustForwardPassPerfModel(object()).dataset_revisions == {}


def test_typed_config_pin_does_not_resolve_default_systems_root(monkeypatch, tmp_path):
    from aisim_core.sdk import perf_database

    def cold_default_root():
        raise AssertionError("default systems root resolved before the pinned dataset was wired")

    monkeypatch.setattr(perf_database, "get_systems_paths", cold_default_root)
    monkeypatch.setenv("AISIM_DATA_LOCAL", str(tmp_path))
    config = res.ForwardPassPerfModelConfig(
        model="m", system="h200", backend="vllm", backend_version="0.1", worker_type="aggregated"
    )
    assert config.to_dict(resolve_default_systems=False)["systems_paths"] == ()
    with pytest.raises(ensure_mod.MissingDataError):  # reaches the pinned loader, not the default root
        res._ensure_dataset(config, None)
