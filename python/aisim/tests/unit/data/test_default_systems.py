# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The default systems root is composed from the pinned dataset, never from packaged tables."""

import importlib.util
import re
import shutil
import tomllib
from pathlib import Path

import pytest

from aisim_core.data import default_systems as ds
from aisim_core.data.ensure import MissingDataError

pytestmark = pytest.mark.unit

_spec = importlib.util.spec_from_file_location("op_publish_fixtures", Path(__file__).with_name("test_op_publish.py"))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
tree, _build = _fixtures.tree, _fixtures._build

SHA = "b" * 40


class DirTransport:
    """Serves a local publish tree as if it were the Hub repository at ``SHA``."""

    def __init__(self, root: Path):
        self.root, self.opened = root, []

    def resolve_revision(self, repo, revision):
        return SHA

    def open_file(self, repo, sha, path):
        self.opened.append(path)
        yield (self.root / path).read_bytes()


@pytest.fixture
def published(tree, tmp_path):
    out = tmp_path / "published"
    _build(tree, out)
    return out


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    for name in (ds.DATASET_ENV, ds.LOCAL_ENV, "AISIM_DATA_OFFLINE", "AISIM_DATA_MAX_BYTES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AISIM_DATA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(ds, "op_latency_pin", lambda: ("o/n", SHA))
    ds.clear_default_systems_cache()
    yield
    ds.clear_default_systems_cache()


def test_pin_file_names_a_full_commit_sha():
    import json

    pin = json.loads(ds.PIN_FILE.read_text())
    assert pin["repo_id"] == "ai-dynamo/aisim-data"
    assert re.fullmatch(r"[0-9a-f]{40}", pin["revision"])


def test_composes_root_from_the_pinned_revision_and_reuses_it(published, monkeypatch):
    transport = DirTransport(published)
    first = Path(ds.default_systems_path(transport=transport))
    assert (first / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet").is_file()
    assert (first / "query_versions.yaml").is_file() and (first / "perf_data_reuse_manifest.yaml").is_file()
    assert (first / "h200_sxm.yaml").is_file()  # packaged specs travel with the data
    assert transport.opened

    shutil.rmtree(first)  # only the root is gone: the downloaded dataset rebuilds it
    ds.clear_default_systems_cache()
    again = DirTransport(published)
    assert Path(ds.default_systems_path(transport=again)) == first
    assert again.opened == []  # warm cache: no network

    ds.clear_default_systems_cache()  # as in a new process
    monkeypatch.setattr(ds, "_ensure_tree", lambda *args: pytest.fail("planned and re-hashed the cached dataset"))
    assert ds.default_systems_path() == str(first)


def test_a_branch_revision_is_resolved_through_the_dataset(published, monkeypatch):
    monkeypatch.setenv(ds.DATASET_ENV, "o/n@main")
    root = Path(ds.default_systems_path(transport=DirTransport(published)))
    assert root.name.startswith(f"o__n-{SHA}-")  # named by the commit, not by the branch


def test_offline_with_cold_cache_fails_instead_of_falling_back(monkeypatch):
    monkeypatch.setenv("AISIM_DATA_OFFLINE", "1")
    with pytest.raises(MissingDataError, match="no fallback"):
        ds.default_systems_path()


def test_local_override_needs_no_network(published, monkeypatch):
    monkeypatch.setenv(ds.LOCAL_ENV, str(published))
    root = Path(ds.default_systems_path())
    assert (root / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet").is_file()


def test_local_override_that_is_not_a_dataset_fails(tmp_path, monkeypatch):
    monkeypatch.setenv(ds.LOCAL_ENV, str(tmp_path))
    with pytest.raises(MissingDataError, match="dataset-layout"):
        ds.default_systems_path()


def test_explicit_systems_paths_skip_the_dataset(tmp_path):
    from aisim_core.sdk import perf_database as pdb

    explicit = tmp_path / "mine"
    explicit.mkdir()
    assert pdb._normalize_systems_paths([str(explicit)]) == [str(explicit)]  # nothing fetched, no pin read


def test_default_download_transport_is_a_base_dependency():
    """The first default prediction downloads through HfTransport, so a plain `pip install aisim` must ship it."""
    project = tomllib.loads(Path(__file__).parents[3].joinpath("pyproject.toml").read_text())["project"]
    base = {
        re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0].lower().replace("-", "_")
        for requirement in project["dependencies"]
    }
    assert "huggingface_hub" in base


@pytest.fixture
def cold_offline(monkeypatch):
    """No default data anywhere: composing the default root would raise, so success proves it was not touched."""
    from aisim_core.sdk import perf_database as pdb

    monkeypatch.setenv("AISIM_DATA_OFFLINE", "1")
    monkeypatch.setattr(pdb, "_SYSTEMS_PATHS", None)
    monkeypatch.delenv("AICONFIGURATOR_SYSTEMS_PATH", raising=False)
    with pytest.raises(MissingDataError):
        ds.default_systems_path()
    return pdb


def test_environment_root_is_used_before_the_default_is_composed(cold_offline, tmp_path, monkeypatch):
    from aisim_core.sdk.rust_engine_step import _resolve_forward_pass_systems_paths

    custom = tmp_path / "custom"
    custom.mkdir()
    monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", str(custom))
    assert _resolve_forward_pass_systems_paths(()) == [str(custom)]


def test_a_process_override_is_used_without_composing_the_default(cold_offline, tmp_path, monkeypatch):
    from aisim_core.sdk.rust_engine_step import _resolve_forward_pass_systems_paths

    custom, other = tmp_path / "custom", tmp_path / "other"
    custom.mkdir()
    other.mkdir()
    cold_offline.set_systems_paths(str(custom))
    assert _resolve_forward_pass_systems_paths(()) == [str(custom)]
    monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", str(other))  # an explicit override still outranks the variable
    assert _resolve_forward_pass_systems_paths(()) == [str(custom)]


def test_without_an_override_the_default_is_still_composed(cold_offline):
    from aisim_core.sdk.rust_engine_step import _resolve_forward_pass_systems_paths

    with pytest.raises(MissingDataError, match="no fallback"):
        _resolve_forward_pass_systems_paths(())


def test_a_dataset_pin_selects_its_root_before_the_default_dataset_resolves(cold_offline, tmp_path, monkeypatch):
    import aisim_core.data.ensure as ensure_module
    from aisim_core.data.ensure import DataPin
    from aisim_core.sdk import ForwardPassPerfModelConfig
    from aisim_core.sdk.rust_engine_step import _ensure_dataset

    seen = []
    monkeypatch.setattr(ensure_module, "ensure_data", lambda payload, pin: seen.append(payload) or "selected")
    identity = {"model": "m", "system": "s", "backend": "vllm", "worker_type": "decode"}
    pin = DataPin(local_root=tmp_path, categories=("op-latency",))
    typed = ForwardPassPerfModelConfig(**identity)
    assert _ensure_dataset(typed, pin) == _ensure_dataset(identity, pin) == "selected"
    assert len(seen) == 2 and not any(payload.get("systems_paths") for payload in seen)  # left for the pin to fill
    with pytest.raises(MissingDataError):  # the remaining default still resolves when no pin wired a root
        typed.to_dict()


def test_custom_roots_never_compose_the_default_even_when_the_environment_names_one(
    cold_offline, tmp_path, monkeypatch
):
    from aisim_core.sdk.rust_engine_step import _resolve_forward_pass_systems_paths

    custom, environment = tmp_path / "custom", tmp_path / "environment"
    custom.mkdir()
    environment.mkdir()
    monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", str(environment))
    # Only the default root alone is an expansion of unset roots, and it is not composed in this process.
    assert _resolve_forward_pass_systems_paths((str(custom),)) == [str(custom)]


def test_naming_roots_is_decided_without_composing_the_default(cold_offline, tmp_path, monkeypatch):
    from aisim_core.sdk.rust_engine_step import _names_systems_paths

    custom = tmp_path / "custom"
    custom.mkdir()
    assert _names_systems_paths({"systems_paths": [str(custom)]}) is True
    assert _names_systems_paths({"systems_paths": ["Default"]}) is True  # the literal token is an explicit request
    assert _names_systems_paths({}) is False  # unset: the default root, which a saved source may answer instead
    cold_offline.set_systems_paths(str(custom))
    assert _names_systems_paths({}) is True
    monkeypatch.setattr(cold_offline, "_SYSTEMS_PATHS", None)  # drop the override; only the variable names roots
    monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", str(custom))
    assert _names_systems_paths({}) is True


def test_restoring_an_unresolved_state_never_composes_the_default(cold_offline, tmp_path):
    custom, other = tmp_path / "custom", tmp_path / "other"
    custom.mkdir()
    other.mkdir()
    previous = cold_offline.explicit_systems_paths()
    assert previous is None  # nothing resolved yet, and reading that state did not compose the default
    cold_offline.set_systems_paths(str(custom))
    cold_offline.restore_systems_paths(previous)
    assert cold_offline.explicit_systems_paths() is None
    cold_offline.set_systems_paths(str(custom))
    cold_offline.restore_systems_paths([str(other)])
    assert cold_offline.explicit_systems_paths() == [str(other)]


def test_an_estimate_with_explicit_roots_never_composes_the_default(cold_offline, tmp_path, monkeypatch):
    import aisim.legacy_cli.api as api

    custom = tmp_path / "custom"
    custom.mkdir()
    seen = []
    monkeypatch.setattr(cold_offline, "get_latest_database_version", lambda system, backend, systems_paths=None: "v1")
    monkeypatch.setattr(api, "_run_agg_estimate", lambda **kwargs: seen.append(kwargs["resolved_version"]) or "done")
    result = api.cli_estimate(
        model_path="Qwen/Qwen3-32B",
        system_name="h200_sxm",
        mode="agg",
        database_mode="SOL",
        batch_size=1,
        systems_paths=str(custom),
    )
    assert result == "done" and seen == ["v1"]
    assert cold_offline.explicit_systems_paths() is None  # the call left the process as it found it
