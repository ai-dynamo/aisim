# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loader reference tests: outputs equal the existing code paths for one configuration."""

import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pytest

from aisim_core.data import Entry, Identity, configuration_id, prepare_execution_profile
from aisim_core.data.fetch import Cache, Selection, SelectionError, fetch
from aisim_core.data.loaders import (
    LOADERS,
    E2eLoader,
    FpmLoader,
    LoaderError,
    OpLatencyLoader,
)
from aisim_core.data.parquet import write_parquet
from aisim_core.sdk import RustForwardPassPerfModel
from tests.unit.data.test_fetch import FakeTransport
from tests.unit.sdk.test_dataset_contracts import artifact, dataset, write  # noqa: F401

pytestmark = pytest.mark.unit

CONFIG = {"system": "h200_sxm", "backend": "vllm", "backend_version": "1.0", "model": "meta-llama/Meta-Llama-3.1-8B"}


def _tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("damage", ["missing-table", "edited-table", "profile", "config"])
@pytest.mark.parametrize("relative_cache", [False, True])
def test_cached_fpm_preparation_reuses_intact_files_and_repairs_damage(
    dataset,  # noqa: F811
    tmp_path,
    monkeypatch,
    damage,
    relative_cache,
):
    from aisim_core.data import Artifact, Entry, Manifest
    from aisim_core.data.ensure import _load

    root, path, doc = dataset
    manifest = Manifest.model_validate(doc)
    data = (root / path).read_bytes()
    entry = Entry(
        configuration_id=manifest.configuration_id,
        purpose=manifest.purpose,
        identity=manifest.identity,
        manifest=Artifact(
            artifact_id="manifest",
            path=path,
            role="manifest",
            format="json",
            schema=manifest.schema_version,
            sha256=hashlib.sha256(data).hexdigest(),
            bytes=len(data),
        ),
    )
    loader = FpmLoader()
    real_load = loader.load
    loads = []

    def counted(*args, **kwargs):
        loads.append(True)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(loader, "load", counted)
    monkeypatch.chdir(tmp_path)
    output = Path("materialized") if relative_cache else tmp_path / "materialized"
    first = _load(loader, "fpm", root, entry, CONFIG, output)
    systems = Path(first.config["systems_paths"][0])
    pristine = _tree(systems.parent)
    assert _load(loader, "fpm", root, entry, CONFIG, output).config == first.config
    assert len(loads) == 1  # no native reconstruction for an intact cache
    runtime = next(a.runtime_path for a in manifest.artifacts if a.runtime_path is not None)
    if damage == "missing-table":
        (systems / runtime).unlink()
    else:
        victim = systems / runtime if damage == "edited-table" else systems.parent / f"{damage}.json"
        victim.write_bytes(b"damaged")
    repaired = _load(loader, "fpm", root, entry, CONFIG, output)
    assert repaired.config == first.config and _tree(systems.parent) == pristine
    assert len(loads) == 2


def _op_manifest(root, manifest, **identity_overrides):
    """Operation-tables manifest over the fixture's table + hardware spec."""
    identity = Identity(
        hardware="h200_sxm",
        framework="vllm",
        framework_version="1.0",
        operation_family=None,
        model=None,
        effective_settings={},
        unresolved_identity=None,
    ).model_copy(update=identity_overrides)
    doc = dict(
        manifest,
        purpose="operation-tables",
        configuration_id=configuration_id("op-latency", "operation-tables", identity),
        identity=identity.model_dump(),
        artifacts=[a for a in manifest["artifacts"] if a.get("runtime_path")],
    )
    write(root, "metadata/op.json", doc)
    return "metadata/op.json"


# ------------------------------------------------------------------ select


def test_select_builds_catalog_selections():
    op = OpLatencyLoader().select(CONFIG)
    assert op == Selection(
        category="op-latency",
        hardware="h200_sxm",
        framework="vllm",
        framework_version="1.0",
        purpose="operation-tables",
    )  # model-independent
    fpm = FpmLoader().select(CONFIG)
    assert fpm.model == CONFIG["model"] and fpm.category == "fpm"
    assert E2eLoader().select(CONFIG).category == "e2e"
    with pytest.raises(SelectionError):
        FpmLoader().select({"system": "h200_sxm"})  # fpm needs a model
    assert set(LOADERS) == {"op-latency", "fpm", "e2e"}


# ------------------------------------------------------------------ op-latency


def test_op_latency_reproduces_systems_root(dataset, tmp_path):  # noqa: F811
    root, _, manifest = dataset
    path = _op_manifest(root, manifest)
    out = OpLatencyLoader().load(root, path, output_dir=tmp_path / "out", config=CONFIG)
    # Reference: the systems root the existing execution-profile preparation produces.
    prepared = prepare_execution_profile(root, "metadata/manifest.json", tmp_path / "prepared")
    assert _tree(out.systems_root) == _tree(tmp_path / "prepared" / "systems")
    assert out.files == tuple(sorted(a["runtime_path"] for a in manifest["artifacts"] if a.get("runtime_path")))
    assert prepared["systems_paths"] == [str(tmp_path / "prepared" / "systems")]
    # And it is discoverable like any systems root.
    from aisim_core.sdk import perf_database as pdb

    assert [r[:3] for r in pdb._discover_database_refs([str(out.systems_root)])] == [
        r[:3] for r in pdb._discover_database_refs([str(tmp_path / "prepared" / "systems")])
    ]


def test_op_latency_rejections(dataset, tmp_path):  # noqa: F811
    root, _, manifest = dataset
    path = _op_manifest(root, manifest)
    with pytest.raises(LoaderError, match="differs from configuration"):
        OpLatencyLoader().load(root, path, output_dir=tmp_path / "a", config={**CONFIG, "backend_version": "2"})
    assert not (tmp_path / "a").exists()
    with pytest.raises(LoaderError, match="model-independent"):
        OpLatencyLoader().load(root, _op_manifest(root, manifest, model="m"), output_dir=tmp_path / "b")
    with pytest.raises(LoaderError, match="cannot load purpose"):
        OpLatencyLoader().load(root, "metadata/manifest.json", output_dir=tmp_path / "c")
    table = root / manifest["artifacts"][0]["path"]
    table.write_bytes(table.read_bytes() + b"x")
    with pytest.raises(ValueError, match="integrity"):
        OpLatencyLoader().load(root, path, output_dir=tmp_path / "d")
    assert not (tmp_path / "d").exists()


def test_op_latency_refuses_existing_output(dataset, tmp_path):  # noqa: F811
    root, _, manifest = dataset
    path = _op_manifest(root, manifest)
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        OpLatencyLoader().load(root, path, output_dir=tmp_path / "out")


# ------------------------------------------------------------------ fpm


def test_fpm_loader_matches_prepare_execution_profile_and_predicts(dataset, tmp_path):  # noqa: F811
    root, path, _ = dataset
    shutil.rmtree(root.parent / "producer-systems")
    loaded = FpmLoader().load(root, path, output_dir=tmp_path / "fpm", config=CONFIG)
    reference = prepare_execution_profile(root, path, tmp_path / "ref")
    strip = lambda c: {k: v for k, v in c.items() if k not in ("systems_paths",) and k != "estimator_config"}
    assert strip(loaded.config) == strip(reference)
    assert _tree(tmp_path / "fpm" / "systems") == _tree(tmp_path / "ref" / "systems")
    assert (tmp_path / "fpm" / "profile.json").read_bytes() == (tmp_path / "ref" / "profile.json").read_bytes()
    a = RustForwardPassPerfModel.best_available(loaded.config)
    b = RustForwardPassPerfModel.best_available(reference)
    assert a.predict_prefill_latency(1, 4, 0) == b.predict_prefill_latency(1, 4, 0) == 6.0
    assert a.predict_prefill_latency(1, 8, 4) == b.predict_prefill_latency(1, 8, 4) == 15.0


@pytest.mark.parametrize("damage", ["systems", "table", "profile", "config", "incomplete"])
def test_cached_fpm_preparation_is_verified_and_rebuilt_before_native_use(dataset, tmp_path, monkeypatch, damage):  # noqa: F811
    from aisim_core.data.ensure import _load

    root, path, manifest = dataset
    entry = Entry(
        configuration_id=manifest["configuration_id"],
        purpose=manifest["purpose"],
        identity=manifest["identity"],
        manifest=artifact(root, path, "manifest"),
    )
    loader = FpmLoader()
    target = tmp_path / "materialized"
    first = _load(loader, "fpm", root, entry, CONFIG, target)
    prepared = next(p for p in tmp_path.glob("materialized-*") if p.is_dir())
    # A valid warm hit must not reconstruct the native model or rewrite the outputs.
    real_load = loader.load

    def no_rebuild(*args, **kwargs):
        pytest.fail("a complete cached preparation must be reused")

    monkeypatch.setattr(loader, "load", no_rebuild)
    assert _load(loader, "fpm", root, entry, CONFIG, target).config == first.config
    monkeypatch.setattr(loader, "load", real_load)
    if damage == "systems":
        shutil.rmtree(prepared / "systems")
    elif damage == "table":
        runtime = next(a["runtime_path"] for a in manifest["artifacts"] if a.get("runtime_path"))
        (prepared / "systems" / runtime).write_bytes(b"corrupt")
    elif damage == "profile":
        (prepared / "profile.json").unlink()
    elif damage == "config":
        config = json.loads((prepared / "config.json").read_bytes())
        config["estimator_config"]["op_level"]["execution_profile"]["sha256"] = "0" * 64
        (prepared / "config.json").write_text(json.dumps(config))
    else:
        (prepared / "config.json").unlink()
    repaired = _load(loader, "fpm", root, entry, CONFIG, target)
    assert repaired.config == first.config
    model = RustForwardPassPerfModel.best_available(repaired.config)
    assert model.predict_prefill_latency(1, 4, 0) == 6.0


def test_fpm_loader_requires_output_dir_and_checks_identity(dataset, tmp_path):  # noqa: F811
    root, path, _ = dataset
    with pytest.raises(LoaderError, match="output_dir"):
        FpmLoader().load(root, path)
    with pytest.raises(LoaderError, match="differs from configuration"):
        FpmLoader().load(root, path, output_dir=tmp_path / "x", config={**CONFIG, "system": "b200_sxm"})


def test_fpm_loader_routes_fpm_profile_artifact_through_load_fpm_profile(dataset, monkeypatch):  # noqa: F811
    # Wiring test: building a valid FpmModelProfile needs a probe-sized fixture, and
    # load_fpm_profile itself is covered elsewhere. Check the verified bytes reach it.
    import aisim_core.fpm_profile as fpm_profile

    seen = []
    monkeypatch.setattr(fpm_profile, "load_fpm_profile", lambda text: seen.append(text) or "PROFILE")
    root, _, manifest = dataset
    (root / "metadata/fpm.json").write_text('{"schema_version": 1}')
    art = artifact(root, "metadata/fpm.json", "fpm-profile")
    write(root, "metadata/fpm-manifest.json", dict(manifest, artifacts=[art]))
    loaded = FpmLoader().load(root, "metadata/fpm-manifest.json")
    assert loaded.profile == "PROFILE" and seen == ['{"schema_version": 1}'] and loaded.config is None
    (root / "metadata/fpm.json").write_text('{"schema_version": 2}')
    with pytest.raises(ValueError, match="integrity"):
        FpmLoader().load(root, "metadata/fpm-manifest.json")


# ------------------------------------------------------------------ e2e


def _e2e_dataset(tmp_path, schema="aisim.test-e2e/v1"):
    root = tmp_path / "e2e"
    table = pa.table({"request": [1, 2, 3], "ttft_ms": [10.0, 11.0, 12.5]})
    path = "e2e/m/requests.parquet"
    (root / path).parent.mkdir(parents=True)
    write_parquet(table, root / path, schema)
    data = (root / path).read_bytes()
    identity = Identity(
        hardware="h200_sxm",
        framework="vllm",
        framework_version="1.0",
        operation_family=None,
        model="m",
        effective_settings={},
        unresolved_identity=None,
    )
    art = {
        "artifact_id": "requests",
        "path": path,
        "role": "e2e-table",
        "format": "parquet",
        "schema": schema,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "rows": 3,
    }
    doc = {
        "schema_version": "aisim-artifact-manifest/v1",
        "category": "e2e",
        "purpose": "e2e-configuration",
        "configuration_id": configuration_id("e2e", "e2e-configuration", identity),
        "identity": identity.model_dump(),
        "requested_settings": None,
        "artifacts": [art],
        "provenance": {},
        "evidence_gaps": [],
    }
    (root / "e2e/m/manifest.json").write_text(json.dumps(doc))
    return root, table


def test_e2e_returns_table_views_equal_to_source(tmp_path):
    root, table = _e2e_dataset(tmp_path)
    cfg = {"system": "h200_sxm", "backend": "vllm", "backend_version": "1.0", "model": "m"}
    loaded = E2eLoader().load(root, "e2e/m/manifest.json", config=cfg)
    assert loaded["requests"].equals(table)
    projected = E2eLoader().load(root, "e2e/m/manifest.json", columns={"requests": ["ttft_ms"]})
    assert projected["requests"].column_names == ["ttft_ms"]


def test_e2e_rejects_unaccepted_schema_and_wrong_model(tmp_path):
    root, _ = _e2e_dataset(tmp_path)
    with pytest.raises(LoaderError, match="unsupported table schema"):
        E2eLoader(accepted_schemas=("aisim.other/v1",)).load(root, "e2e/m/manifest.json")
    with pytest.raises(LoaderError, match="manifest model"):
        E2eLoader().load(
            root,
            "e2e/m/manifest.json",
            config={"system": "h200_sxm", "backend": "vllm", "backend_version": "1.0", "model": "other"},
        )


def test_e2e_rejects_corrupt_and_path_escape(tmp_path):
    root, _ = _e2e_dataset(tmp_path)
    with pytest.raises(ValueError, match="canonical"):
        E2eLoader().load(root, "../manifest.json")
    p = root / "e2e/m/requests.parquet"
    p.write_bytes(p.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="integrity"):
        E2eLoader().load(root, "e2e/m/manifest.json")


def test_e2e_tables_are_parsed_from_the_verified_bytes(tmp_path, monkeypatch):
    """A file replaced after verification, with the same schema ID, never reaches the caller."""
    import os

    from aisim_core.data import loaders as loaders_module
    from aisim_core.data.e2e import load_e2e, write_configuration
    from tests.unit.data.test_e2e import _dev_config

    expected = _dev_config()
    path = write_configuration(tmp_path, expected)
    table = next(tmp_path.rglob("metrics.parquet"))
    schema_id = next(
        a["schema"]
        for a in json.loads((tmp_path / path).read_text())["artifacts"]
        if a["path"].endswith("metrics.parquet")
    )
    replacement = write_parquet(pa.table({"observation_id": ["x"]}), tmp_path / "evil.parquet", schema_id)
    real = loaders_module._verified_bytes
    swapped = []

    def verify_then_swap(root, artifact):
        data = real(root, artifact)
        if artifact.path.endswith("metrics.parquet") and not swapped:
            swapped.append(os.replace(replacement, table))  # replaced right after the check
        return data

    monkeypatch.setattr(loaders_module, "_verified_bytes", verify_then_swap)
    assert len(load_e2e(tmp_path, path).metrics) == len(expected.metrics)


def test_legacy_fpm_cold_fetch_keeps_auxiliary_measurement_provenance(dataset, tmp_path):  # noqa: F811
    root, path, manifest = dataset
    auxiliary = "metadata/measurement-provenance.json"
    write(root, auxiliary, {"provenance": "Synthetic fixture, not measured evidence."})
    manifest["artifacts"].append(artifact(root, auxiliary, "measurement-provenance"))
    # Catalog this legacy execution-profile bundle under FPM, as FpmLoader.select requests.
    manifest["category"] = "fpm"
    manifest["configuration_id"] = configuration_id("fpm", "execution-profile", Identity(**manifest["identity"]))
    write(root, path, manifest)
    write(
        root,
        "catalog/fpm/index.json",
        {
            "schema_version": "aisim-category-index/v1",
            "category": "fpm",
            "entries": [
                {
                    "configuration_id": manifest["configuration_id"],
                    "purpose": manifest["purpose"],
                    "identity": manifest["identity"],
                    "manifest": artifact(root, path, "manifest"),
                }
            ],
        },
    )
    write(
        root,
        "catalog/dataset.json",
        {
            "schema_version": "aisim-dataset-index/v1",
            "dataset_id": "org/dataset",
            "categories": {
                "fpm": {
                    "root": "data/fpm",
                    "layout": "category",
                    "index": artifact(root, "catalog/fpm/index.json", "index"),
                }
            },
        },
    )
    transport = FakeTransport({str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()})
    fetched = fetch(
        FpmLoader().select(CONFIG),
        repo="org/dataset",
        revision="a" * 40,
        transport=transport,
        cache=Cache(tmp_path / "cold-cache"),
    )
    assert auxiliary in transport.opened
    loaded = FpmLoader().load(fetched.root, path, output_dir=tmp_path / "loaded", config=CONFIG)
    model = RustForwardPassPerfModel.best_available(loaded.config)
    assert model.predict_prefill_latency(1, 4, 0) == 6.0


def test_catalog_parallelism_keeps_omitted_topology_distinct_from_explicit_all_ones():
    from aisim_core.data.loaders import _parallelism
    from aisim_core.sdk import ForwardPassPerfModelConfig

    assert _parallelism({"model": "m", "system": "h200"}) is None  # omitted: the catalog stays unfiltered
    assert _parallelism({"tp": 1}) == ()  # explicit all-ones selects the single-GPU identity
    assert _parallelism({"tp": 4, "attention_dp": 1, "moe_ep_size": 4}) == (("ep", 4), ("tp", 4))
    # A typed configuration always carries topology (tp=1 by default), so it requests the single-GPU entry.
    typed = ForwardPassPerfModelConfig(
        model="m", system="h200", backend="vllm", backend_version="1", worker_type="aggregated"
    )
    assert _parallelism(typed.to_dict()) == ()


def test_moved_fpm_embedded_keys_reach_the_native_profile_consumer(dataset, tmp_path):  # noqa: F811
    root, _, manifest = dataset
    logical_dir = "data/Meta-Llama/h200/vllm/1/tp1"
    for item in manifest["artifacts"]:
        original = root / item["path"]
        item["path"] = f"{logical_dir}/{original.name}"
        destination = root / item["path"].replace("data/", "data/fpm/", 1)
        destination.parent.mkdir(parents=True, exist_ok=True)
        original.rename(destination)
    logical_manifest = f"{logical_dir}/manifest.json"
    stored_manifest = logical_manifest.replace("data/", "data/fpm/", 1)
    write(root, stored_manifest, manifest)
    manifest_ref = artifact(root, stored_manifest, "manifest")
    manifest_ref["path"] = logical_manifest
    write(
        root,
        "catalog/fpm.json",
        {
            "schema_version": "aisim-category-index/v1",
            "category": "fpm",
            "entries": [
                {
                    "configuration_id": manifest["configuration_id"],
                    "identity": manifest["identity"],
                    "purpose": manifest["purpose"],
                    "manifest": manifest_ref,
                }
            ],
        },
    )
    write(
        root,
        "catalog/dataset.json",
        {
            "schema_version": "aisim-dataset-index/v1",
            "dataset_id": "ai-dynamo/aisim-data",
            "categories": {
                "fpm": {"root": "data/fpm", "layout": "category", "index": artifact(root, "catalog/fpm.json", "index")}
            },
        },
    )
    shutil.rmtree(root.parent / "producer-systems")
    loaded = FpmLoader().load(root, logical_manifest, output_dir=tmp_path / "moved", config=CONFIG)
    assert [a.path for a in loaded.manifest.artifacts] == [a["path"] for a in manifest["artifacts"]]
    model = RustForwardPassPerfModel.best_available(loaded.config)
    # Independent fixture oracle: scale factor 3 times the retained measured cells of 2 ms and 5 ms.
    assert model.predict_prefill_latency(1, 4, 0) == 6.0
    assert model.predict_prefill_latency(1, 8, 4) == 15.0
