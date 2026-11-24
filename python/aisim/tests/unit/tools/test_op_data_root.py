# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation tools read the selected op-latency root, never the deleted in-tree ``systems/data`` directory."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from tools.prediction_regression_gate import grid

pytestmark = pytest.mark.unit

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
PERF_DATABASE = PACKAGE_ROOT / "tools" / "perf_database"


def _load(name: str):
    sys.path.insert(0, str(PERF_DATABASE))  # the tools import their sibling modules by bare name
    try:
        spec = importlib.util.spec_from_file_location(name, PERF_DATABASE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(PERF_DATABASE))


@pytest.fixture
def op_data_root_module():
    return _load("op_data_root")


@pytest.fixture
def composed(tmp_path, monkeypatch):
    """A stand-in for the composed default systems root; the dataset itself is never fetched."""
    import aisim_core.data.default_systems as default_systems

    root = tmp_path / "composed"
    (root / "data").mkdir(parents=True)
    monkeypatch.delenv("AICONFIGURATOR_SYSTEMS_PATH", raising=False)
    monkeypatch.setattr(default_systems, "default_systems_path", lambda **_: str(root))
    return root


def test_default_root_is_the_composed_dataset_not_the_source_tree(op_data_root_module, composed):
    root = op_data_root_module.op_data_root()
    assert root == composed / "data"
    assert (PACKAGE_ROOT / "src") not in root.parents


def test_environment_root_and_explicit_flag_take_precedence(op_data_root_module, composed, tmp_path, monkeypatch):
    monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", str(tmp_path / "env"))
    assert op_data_root_module.op_data_root() == tmp_path / "env" / "data"
    assert op_data_root_module.op_data_root(tmp_path / "flag") == tmp_path / "flag"


def test_prediction_gate_discovers_combos_in_the_composed_root(composed, monkeypatch):
    monkeypatch.setattr(grid, "DATA_ROOT", None)  # the default, after the packaged tree was removed
    assert not (PACKAGE_ROOT / "src" / "aisim_core" / "systems" / "data").exists()
    (composed / "b200_sxm.yaml").write_text("data_dir: data\n")
    for relative in ("gemm/vllm/0.19.0/gemm_perf.parquet", "moe/vllm/0.22.0/moe_perf.parquet"):
        path = composed / "data" / "b200_sxm" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")
    combos = grid.enumerate_combos(systems=["b200_sxm"], versions="all")
    assert sorted(c.version for c in combos) == ["0.19.0", "0.22.0"]


def test_collector_data_check_defaults_to_the_composed_root(composed, monkeypatch):
    module = _load("check_collector_data")
    seen = []
    monkeypatch.setattr(module, "run_checks", lambda root, catalog: seen.append(root) or {r: [] for r in module.RULES})
    assert module.main([]) == 0
    assert seen == [composed / "data"]


def test_backend_facts_check_defaults_to_the_composed_root(composed, monkeypatch):
    module = _load("backend_facts")
    seen = []

    def scan(root):
        seen.append(root)
        raise SystemExit(0)

    monkeypatch.setattr(module, "scan", scan)
    monkeypatch.setattr(sys, "argv", ["backend_facts.py", "--check"])
    with pytest.raises(SystemExit):
        module.main()
    assert seen == [composed / "data"]
