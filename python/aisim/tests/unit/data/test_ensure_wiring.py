# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset source selection reaches the native predictor, including saved init settings."""

import json
import os
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from aisim.main import main
from aisim_core.data import default_systems
from aisim_core.data import ensure as ensure_mod
from aisim_core.sdk import perf_database
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
    monkeypatch.delenv("AICONFIGURATOR_SYSTEMS_PATH", raising=False)
    monkeypatch.setattr(perf_database, "_SYSTEMS_PATHS", None)  # no override held; restored afterwards


@pytest.fixture(autouse=True)
def default_root(monkeypatch, tmp_path):
    """A stand-in for the composed default systems root, so no test fetches or composes the pinned dataset."""
    root = tmp_path / "default-systems"
    root.mkdir()
    monkeypatch.setattr(default_systems, "_resolved", {})
    monkeypatch.setattr(default_systems, "_compose", lambda *args, **kwargs: root)
    return str(root)


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


@pytest.mark.parametrize("mode", ["fpm_interpolation", "fpm_regression", "fpm_learned"])
def test_a_saved_source_leaves_fpm_estimators_and_their_systems_paths_alone(monkeypatch, tmp_path, mode):
    assert main(["init", "--on-demand", "--repo", REPO, "--revision", SHA]) == 0
    seen = []
    monkeypatch.setattr(ensure_mod, "ensure_data", lambda payload, pin: seen.append(payload) or "ensured")
    config = {"estimation_mode": mode, "system": "h200_sxm", "systems_paths": [str(tmp_path / "fpm-tables")]}
    # The saved op-latency source would replace systems_paths, which carries these modes' FPM tables.
    assert res._ensure_dataset(config, None) is None and seen == []
    # An operation-table run that names no roots still uses the saved source ...
    unnamed = {k: v for k, v in config.items() if k != "systems_paths"}
    assert res._ensure_dataset({**unnamed, "estimation_mode": "op_level"}, None) == "ensured"
    # ... and a source the environment or the caller names applies to every mode.
    monkeypatch.setenv("AISIM_DATASET", f"{REPO}@{SHA}")
    assert res._ensure_dataset(config, None) == "ensured"
    monkeypatch.delenv("AISIM_DATASET")
    assert res._ensure_dataset(config, ensure_mod.DataPin(repo=REPO, revision=SHA)) == "ensured"


@pytest.mark.parametrize("mode", ["auto", "op_level"])
def test_a_saved_source_never_replaces_the_roots_the_caller_named(monkeypatch, tmp_path, mode):
    # ``auto`` falls back to the FPM estimators, which read their tables from the named roots.
    assert main(["init", "--on-demand", "--repo", REPO, "--revision", SHA]) == 0
    seen = []
    monkeypatch.setattr(ensure_mod, "ensure_data", lambda payload, pin: seen.append(payload) or "ensured")
    named = {"estimation_mode": mode, "system": "h200_sxm", "systems_paths": [str(tmp_path / "tables")]}
    assert res._ensure_dataset(named, None) is None and seen == []
    config = res.ForwardPassPerfModelConfig(
        model="m",
        system="h200_sxm",
        backend="vllm",
        worker_type="prefill",
        estimation_mode=mode,
        systems_paths=(str(tmp_path),),
    )
    assert res._ensure_dataset(config, None) is None  # the object form names roots the same way
    unnamed = {"estimation_mode": mode, "system": "h200_sxm"}
    assert res._ensure_dataset(unnamed, None) == "ensured"  # no roots named: the op-latency source applies
    monkeypatch.setenv("AISIM_DATASET", f"{REPO}@{SHA}")
    assert res._ensure_dataset(named, None) == "ensured"  # a source the environment names overrides them


@pytest.mark.parametrize("mode", ["auto", "op_level"])
def test_a_saved_source_never_replaces_the_literal_default_roots(monkeypatch, mode):
    # The literal token explicitly asks for the bundled tables, so the saved source must not replace them.
    assert main(["init", "--on-demand", "--repo", REPO, "--revision", SHA]) == 0
    seen = []
    monkeypatch.setattr(ensure_mod, "ensure_data", lambda payload, pin: seen.append(payload) or "ensured")
    for spelling in ("default", "Default"):
        explicit = {"estimation_mode": mode, "system": "h200_sxm", "systems_paths": [spelling]}
        assert res._ensure_dataset(explicit, None) is None and seen == []
    assert res._ensure_dataset({"estimation_mode": mode, "system": "h200_sxm"}, None) == "ensured"


def _sweeper_request(model, systems_paths=None, mode="op_level"):
    """The request the recommendation consumer builds; it expands unset roots to the default ones itself."""
    from aisim.sweeper.config import SearchSpace
    from aisim.sweeper.forward_pass_estimator import ForwardPassEstimatorResolver

    space = {
        "model_name": str(model),
        "hardware_sku": "h200_sxm",
        "backend": ["vllm"],
        "backend_version": "1.0",
        "estimation_mode": mode,
        "estimator_config": {"correction": {"enabled": False}},
    }
    if systems_paths is not None:
        space["systems_paths"] = systems_paths
    sample = {
        "backend": "vllm",
        "hardware_sku": "h200_sxm",
        "prefill_moe_tp": 1,
        "prefill_moe_ep": 1,
        "prefill_tp": 1,
        "prefill_pp": 1,
        "prefill_attention_dp": 1,
        "prefill_block_size": None,
    }
    resolver = ForwardPassEstimatorResolver(SearchSpace.model_validate(space))
    return resolver, resolver._request(sample, "prefill")


@pytest.mark.parametrize("mode", ["op_level", "auto"])
@pytest.mark.parametrize("configure_defaults", [False, True])
def test_the_recommendation_consumer_gets_the_saved_source_unless_the_user_names_roots(
    native_dataset,  # noqa: F811
    monkeypatch,
    tmp_path,
    mode,
    configure_defaults,
):
    published, systems, config = native_dataset
    transport = FakeTransport(_tree(published))
    monkeypatch.setattr("aisim_core.data.fetch.HfTransport", lambda: transport)
    assert main(["init", "--on-demand", "--repo", REPO, "--revision", SHA]) == 0
    model = config.model
    if configure_defaults:
        res._configure_default_data_roots()

    # No roots given: the consumer still expands them to the default ones, and the saved source wins over those.
    resolver, request = _sweeper_request(model, mode=mode)
    assert request.systems_paths, "the consumer hands over the expanded default roots"
    shutil.rmtree(systems / "data")
    built = res.RustForwardPassPerfModel.best_available(request)
    assert built.dataset_revisions == {"op-latency": SHA}
    selected = Path(built.diagnostics()["provenance"]["selected_systems_root"])
    assert selected.is_relative_to(Path(os.environ["AISIM_DATA_CACHE"]) / "materialized")
    assert built.predict_prefill_latency(1, 4) >= 7.0  # the published 7 ms attention measurement is consumed

    # Roots the user gave win over the saved source.
    own = tmp_path / "own-roots"
    own.mkdir()
    _, request = _sweeper_request(model, systems_paths=[str(own)], mode=mode)
    assert res._ensure_dataset(request, None) is None


@pytest.mark.parametrize("mode", ["op_level", "auto"])
def test_the_recommendation_consumer_keeps_an_explicit_default_root_distinct_from_unset_roots(
    native_dataset,  # noqa: F811
    monkeypatch,
    mode,
):
    """Both expand to the default root, but only the literal ``default`` outranks the saved source."""
    published, _, config = native_dataset
    transport = FakeTransport(_tree(published))
    monkeypatch.setattr("aisim_core.data.fetch.HfTransport", lambda: transport)
    assert main(["init", "--on-demand", "--repo", REPO, "--revision", SHA]) == 0
    seen = []
    monkeypatch.setattr(ensure_mod, "ensure_data", lambda payload, pin: seen.append(payload) or "ensured")
    _, explicit = _sweeper_request(config.model, systems_paths=["default"], mode=mode)
    assert res._ensure_dataset(explicit, None) is None and seen == []
    assert "default" in explicit.systems_paths
    _, unset = _sweeper_request(config.model, mode=mode)
    assert res._ensure_dataset(unset, None) == "ensured"


def test_an_explicit_default_root_survives_native_serialization_with_an_environment_root(
    monkeypatch, tmp_path, default_root
):
    environment = tmp_path / "environment"
    environment.mkdir()
    monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", str(environment))

    def serialized(**roots):
        config = res.ForwardPassPerfModelConfig(
            model="m", system="h200_sxm", backend="vllm", worker_type="prefill", estimation_mode="op_level", **roots
        )
        return json.loads(res._forward_pass_config_json(config))["systems_paths"]

    # An explicit default is the default data even though the environment names other roots; unset roots follow it.
    assert serialized(systems_paths=("default",)) == [default_root]
    assert serialized() == [str(environment)]
    # The raw mapping form resolves once as well.
    mapping = {"estimation_mode": "op_level", "systems_paths": ["default"]}
    assert json.loads(res._forward_pass_config_json(mapping))["systems_paths"] == [default_root]


def test_request_roots_keep_the_configured_order_when_an_environment_root_is_set(monkeypatch, tmp_path, default_root):
    from aisim.sweeper.forward_pass_estimator import request_systems_paths

    custom, environment = tmp_path / "custom", tmp_path / "environment"
    custom.mkdir()
    environment.mkdir()
    monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", str(environment))
    configured = [default_root, "default", str(custom)]
    # The environment root replaces only an unset list; explicit entries stay in order, the token stays a token.
    assert request_systems_paths(configured) == (default_root, "default", str(custom))
    assert request_systems_paths(["default", str(custom)]) == ("default", str(custom))


@pytest.mark.parametrize("how", ["environment", "set_systems_paths"])
@pytest.mark.parametrize("mode", ["op_level", "auto"])
def test_configured_roots_outrank_the_saved_source_whether_or_not_the_request_names_them(
    native_dataset,  # noqa: F811
    monkeypatch,
    tmp_path,
    how,
    mode,
):
    """``AICONFIGURATOR_SYSTEMS_PATH`` and ``set_systems_paths`` are choices; the Sweeper's default expansion is not."""
    published, systems, config = native_dataset
    transport = FakeTransport(_tree(published))
    monkeypatch.setattr("aisim_core.data.fetch.HfTransport", lambda: transport)
    assert main(["init", "--on-demand", "--repo", REPO, "--revision", SHA]) == 0
    own = tmp_path / "configured-roots"
    own.mkdir()
    if how == "environment":
        monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", str(own))
    else:
        perf_database.set_systems_paths([str(own)])

    # The request names no roots; the consumer expands them to the configured ones, and those win.
    _, request = _sweeper_request(config.model, mode=mode)
    assert list(request.systems_paths) == [str(own)]
    assert res._ensure_dataset(request, None) is None
    # The same holds when the roots are not named in the request at all.
    unnamed = {"estimation_mode": mode, "system": "h200_sxm", "backend": "vllm"}
    assert res._ensure_dataset(unnamed, None) is None

    # Nothing configured: the default root is a default, and the saved source applies again.
    monkeypatch.delenv("AICONFIGURATOR_SYSTEMS_PATH", raising=False)
    perf_database.set_systems_paths(None)
    _, request = _sweeper_request(config.model, mode=mode)
    assert res._ensure_dataset(request, None) is not None


@pytest.mark.parametrize("mode", ["op_level", "auto"])
@pytest.mark.parametrize("as_mapping", [False, True])
@pytest.mark.parametrize("roots", ["expanded", "unset"])
def test_environment_root_reaches_prediction_for_packaged_only_paths(
    native_dataset,  # noqa: F811
    monkeypatch,
    mode,
    as_mapping,
    roots,
):
    published, systems, config = native_dataset
    transport = FakeTransport(_tree(published))
    monkeypatch.setattr("aisim_core.data.fetch.HfTransport", lambda: transport)
    assert main(["init", "--on-demand", "--repo", REPO, "--revision", SHA]) == 0
    perf_database.set_systems_paths(None)
    entries = {"expanded": tuple(perf_database.get_systems_paths()), "unset": ()}[roots]
    monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", str(systems))
    config = replace(config, estimation_mode=mode, systems_paths=entries)
    request = asdict(config) if as_mapping else config

    assert res._ensure_dataset(request, None) is None
    model = res.RustForwardPassPerfModel.best_available(request)
    assert model.dataset_revisions == {}
    assert Path(model.diagnostics()["provenance"]["selected_systems_root"]) == systems
    assert model.predict_prefill_latency(1, 4) >= 7.0  # the fixture's 7 ms attention measurement is consumed
    assert not transport.opened


def test_a_literal_default_stays_the_default_data_when_the_environment_names_a_root(
    tmp_path, monkeypatch, default_root
):
    perf_database.set_systems_paths(None)
    assert perf_database.get_systems_paths() == [default_root]
    monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", str(tmp_path))
    assert res._resolve_forward_pass_systems_paths(("default",)) == [default_root]  # an explicit request for it
    assert res._resolve_forward_pass_systems_paths((default_root,)) == [str(tmp_path)]  # the expansion of unset roots
    assert res._resolve_forward_pass_systems_paths(()) == [str(tmp_path)]


@pytest.mark.parametrize("configure_defaults", [False, True])
def test_an_environment_choice_of_packaged_roots_outranks_the_saved_source(
    native_dataset,  # noqa: F811
    monkeypatch,
    configure_defaults,
):
    published, _, config = native_dataset
    transport = FakeTransport(_tree(published))
    monkeypatch.setattr("aisim_core.data.fetch.HfTransport", lambda: transport)
    assert main(["init", "--on-demand", "--repo", REPO, "--revision", SHA]) == 0
    perf_database.set_systems_paths(None)
    if configure_defaults:
        res._configure_default_data_roots()
    monkeypatch.setenv("AICONFIGURATOR_SYSTEMS_PATH", perf_database.get_systems_paths()[0])
    assert res._ensure_dataset(replace(config, systems_paths=("default",)), None) is None
    assert not transport.opened


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
