# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import shutil
import sys

import pytest

from aisim_core.data import Artifact, CategoryIndex, DatasetIndex, Identity, Manifest, configuration_id
from aisim_core.data import catalog as catalog_module
from aisim_core.data.catalog import (
    Catalog,
    CatalogBudget,
    build_catalog,
    catalog_files,
    check_catalog_size,
    known_system_specs,
    shard_path,
)
from aisim_core.data.eval import EvalRecord, PredictionIdentity, write_record
from aisim_core.data.parquet import read_parquet_schema_id
from aisim_core.data.systems import spec_sha256

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_coverage_replacement_preserves_previous_declaration(tmp_path, monkeypatch, failure):
    root, systems = tmp_path / "dataset", tmp_path / "systems"
    systems.mkdir()
    build_catalog(root, systems_dir=systems, expected_models=["original"], expected_frameworks=["vllm"])
    path = root / "catalog/coverage.json"
    before = path.read_bytes()
    original = catalog_module.tempfile.NamedTemporaryFile

    def fail_replace(*args):
        raise OSError("synthetic replacement failure")

    class BrokenWrite:
        def __enter__(self):
            self.handle = original("w", dir=path.parent, delete=False)
            self.name = self.handle.name
            return self

        def write(self, value):
            self.handle.write(value[:8])
            raise OSError("synthetic partial write")

        def __exit__(self, *args):
            self.handle.close()

    if failure == "write":
        monkeypatch.setattr(catalog_module.tempfile, "NamedTemporaryFile", lambda *a, **kw: BrokenWrite())
    else:
        monkeypatch.setattr(catalog_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic"):
        build_catalog(root, systems_dir=systems, expected_models=["replacement"])
    assert path.read_bytes() == before
    assert catalog_module.read_catalog_coverage(root).expected_models == ["original"]
    assert not [p for p in path.parent.iterdir() if p.is_file() and p.name.startswith("tmp")]


@pytest.mark.parametrize("hardware_count", [1, 2])
def test_catalog_rejects_projected_grid_before_materializing_rows(tmp_path, monkeypatch, hardware_count):
    monkeypatch.setattr(catalog_module, "known_system_specs", lambda _: {f"gpu{i}": NEW for i in range(hardware_count)})

    def unexpected(*args):
        pytest.fail("oversized grid reached row materialization")

    monkeypatch.setattr(catalog_module, "_summarize", unexpected)
    # The same dimensions fit on one hardware only; hardware must count too.
    count = 1000 if hardware_count == 1 else 200
    frameworks = 1000 if hardware_count == 1 else 200
    with pytest.raises(ValueError, match="projected catalog grid.*limit"):
        build_catalog(
            tmp_path,
            expected_models=[f"model{i}" for i in range(count)],
            expected_frameworks=[f"framework{i}" for i in range(frameworks)],
            write=False,
        )
    assert not (tmp_path / "catalog").exists()


SPEC = {"gpu": {"flops": 1}, "node": {"gpus": 8}}
NEW = spec_sha256(SPEC)
OLD = "0" * 64
QWEN, DSV4 = "Qwen/Qwen3-32B", "deepseek-ai/DeepSeek-V4-Pro"


def _ident(model, hw="b200_sxm", fw="vllm", ver="0.11.0", family=None):
    return Identity(
        hardware=hw,
        framework=fw,
        framework_version=ver,
        operation_family=family,
        model=model,
        effective_settings={},
        unresolved_identity=None,
    )


def _artifact(path, payload=b"x"):
    return Artifact(
        artifact_id=path,
        path=path,
        role="manifest",
        format="json",
        schema="s/v1",
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
    )


def _write_category(root, category, purpose, idents, gaps=0):
    entries = []
    for i, ident in enumerate(idents):
        cid = configuration_id(category, purpose, ident)
        path = f"data/{category}/m{i}.json"
        manifest = Manifest(
            schema_version="aisim-artifact-manifest/v1",
            category=category,
            purpose=purpose,
            configuration_id=cid,
            identity=ident,
            requested_settings=None,
            artifacts=[_artifact(f"data/{category}/payload{i}.parquet")],
            provenance={},
            evidence_gaps=[{"field": "g"}] * gaps,
        )
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(manifest.model_dump_json(by_alias=True))
        entries.append(
            {"configuration_id": cid, "purpose": purpose, "identity": ident.model_dump(), "manifest": _artifact(path)}
        )
    index = CategoryIndex.model_validate(
        {"schema_version": "aisim-category-index/v1", "category": category, "entries": entries}
    )
    (root / "catalog").mkdir(exist_ok=True)
    rel = f"catalog/{category}.json"
    (root / rel).write_text(index.model_dump_json(by_alias=True))
    return rel


def _record(ident, layer, mape, spec, created, gpu_seconds=None):
    pred = PredictionIdentity.build("fpm_interpolation", {"p": "b" * 64}, spec)
    gpu = {"authority": "unknown"}
    if gpu_seconds is not None:
        gpu = {"authority": "measured", "gpu_count": 8, "wall_seconds": gpu_seconds / 8}
    return EvalRecord.model_validate(
        {
            "schema_version": "aisim-eval/v1",
            "prediction": pred.model_dump(),
            "truth": {
                "layer": layer,
                "configuration_ids": ["cfg-" + "c" * 64],
                "identity": ident.model_dump(),
                "artifacts": [_artifact("truth.parquet").model_dump(by_alias=True)],
                "truth_table_artifact_id": "truth.parquet",
            },
            "results": {
                "n_total": 10,
                "n_scored": 9,
                "mape_pct": mape,
                "residual_quantiles_pct": None,
                "worst_n": [],
                "coverage": 0.9,
                "exclusions": [],
                "collection_quality": {"gaps": 1},
            },
            "assumptions": [],
            "gpu_time": gpu,
            "simulation_cost": {},
            "producer": {"kind": "ci", "name": "t", "evaluator_revision": "r"},
            "tier": "submitted",
            "created_at": created,
        }
    )


@pytest.fixture
def dataset(tmp_path):
    systems = tmp_path / "systems"
    systems.mkdir()
    import yaml

    for name in ("b200_sxm", "h200_sxm"):
        (systems / f"{name}.yaml").write_text(yaml.safe_dump(SPEC))
    (systems / "query_versions.yaml").write_text("not: a spec\n")  # skipped: not a system spec
    root = tmp_path / "ds"
    root.mkdir()
    fpm = _write_category(root, "fpm", "execution-profile", [_ident(QWEN), _ident(DSV4)], gaps=2)
    e2e = _write_category(root, "e2e", "e2e-configuration", [_ident(QWEN)])
    ref = lambda c, rel: {"root": f"data/{c}", "index": _artifact(rel).model_dump(by_alias=True), "layout": "category"}
    (root / "catalog" / "dataset.json").write_text(
        DatasetIndex.model_validate(
            {
                "schema_version": "aisim-dataset-index/v1",
                "dataset_id": "nvidia/test",
                "categories": {"fpm": ref("fpm", fpm), "e2e": ref("e2e", e2e)},
            }
        ).model_dump_json(by_alias=True)
    )
    write_record(root, _record(_ident(QWEN), "fpm", 3.0, NEW, "2026-11-01T00:00:00Z", gpu_seconds=800.0))
    write_record(root, _record(_ident(QWEN), "fpm", 9.0, NEW, "2026-11-02T00:00:00Z", gpu_seconds=1600.0))
    write_record(root, _record(_ident(DSV4), "fpm", 12.0, OLD, "2026-11-02T00:00:00Z"))
    write_record(root, _record(_ident(QWEN), "e2e", 8.0, NEW, "2026-11-02T00:00:00Z"))
    return root, systems


@pytest.fixture
def built(dataset):
    root, systems = dataset
    rows = build_catalog(root, systems_dir=systems)
    return root, systems, rows


def _cell(rows, layer, model, hw):
    (match,) = [r for r in rows if (r.layer, r.model, r.hardware) == (layer, model, hw)]
    return match


def test_rows_status_accuracy_quality_and_gpu_time(built):
    _, _, rows = built
    qwen = _cell(rows, "fpm", QWEN, "b200_sxm")
    assert qwen.status == "have" and qwen.n_configs == 1 and qwen.n_evidence_gaps == 2
    assert qwen.acc_mape_pct == 9.0  # latest created_at wins
    assert qwen.gpu_seconds == 2400.0 and qwen.gpu_n_recorded == 2
    assert qwen.collection_quality_json == '{"gaps":1}'
    dsv4 = _cell(rows, "fpm", DSV4, "b200_sxm")
    assert dsv4.status == "stale" and dsv4.spec_sha256_current == NEW and dsv4.acc_spec_sha256 == OLD
    assert dsv4.gpu_seconds is None  # unknown stays null, never zero
    e2e_dsv4 = _cell(rows, "e2e", DSV4, "b200_sxm")
    assert e2e_dsv4.status == "missing" and e2e_dsv4.acc_mape_pct is None


def test_known_systems_without_data_are_all_missing(built):
    root, systems, rows = built
    h200 = [r for r in rows if r.hardware == "h200_sxm"]
    assert h200 and {r.status for r in h200} == {"missing"}
    assert {(r.layer, r.model) for r in h200} >= {("fpm", QWEN), ("e2e", DSV4), ("op", None)}
    assert Catalog(root).systems() == {"b200_sxm": NEW, "h200_sxm": NEW} == known_system_specs(systems)


def test_sharded_zstd_parquet_with_schema_id(built):
    root, _, _ = built
    assert shard_path(root, "fpm", "b200_sxm").is_file() and shard_path(root, "e2e", "h200_sxm").is_file()
    files = catalog_files(root)
    assert files and all(read_parquet_schema_id(f) for f in files)
    assert {f.parent.name for f in files} <= {"catalog", "op-latency", "fpm", "e2e"}


def test_rebuild_is_deterministic_and_drops_stale_shards(built):
    root, systems, rows = built
    before = {f: f.read_bytes() for f in catalog_files(root)}
    orphan = root / "catalog" / "fpm" / "gone_gpu.parquet"
    shutil.copy(shard_path(root, "fpm", "b200_sxm"), orphan)
    assert build_catalog(root, systems_dir=systems) == rows
    assert not orphan.exists()
    assert {f: f.read_bytes() for f in catalog_files(root)} == before


def test_query_filters(built):
    root, _, _ = built
    cat = Catalog(root)
    assert {r.hardware for r in cat.rows(hardware="b200_sxm")} == {"b200_sxm"}
    assert {r.layer for r in cat.rows(layer="e2e")} == {"e2e"}
    assert {r.model for r in cat.rows(model=DSV4, layer="fpm")} == {DSV4}
    assert [r.hardware for r in cat.stale_cells()] == ["b200_sxm"]
    assert cat.hardware() == ["b200_sxm", "h200_sxm"]
    unevaluated = cat.unevaluated_cells("b200_sxm")
    assert unevaluated == [] or all(r.status == "have" for r in unevaluated)


def test_exit_gate_worst_and_empty_cells_from_catalog_alone(built, tmp_path, monkeypatch):
    """An agent finds worst/empty cells for one GPU with only the catalog downloaded, reading no payloads."""
    root, _, _ = built
    only_catalog = tmp_path / "download"
    shutil.copytree(root / "catalog", only_catalog / "catalog", ignore=shutil.ignore_patterns("*.json"))
    assert not (only_catalog / "eval").exists() and not (only_catalog / "data").exists()

    # Parquet is read by native code (invisible to audit hooks), so spy on the reader too.
    opened = []
    real_read = catalog_module.read_parquet
    monkeypatch.setattr(
        catalog_module, "read_parquet", lambda path, *a, **k: (opened.append(str(path)), real_read(path, *a, **k))[1]
    )
    # An audit hook cannot be removed, so gate it: later tests in the session must not feed ``opened``.
    active = [True]
    sys.addaudithook(lambda event, args: opened.append(str(args[0])) if active[0] and event == "open" else None)
    cat = Catalog(only_catalog)
    worst = cat.worst_cells("b200_sxm", n=2)
    worst_reads = [p for p in opened if str(only_catalog) in p]
    empty = cat.empty_cells("h200_sxm", layer="fpm")
    active[0] = False

    assert [(r.model, r.layer, r.acc_mape_pct) for r in worst] == [(DSV4, "fpm", 12.0), (QWEN, "fpm", 9.0)]
    assert {r.model for r in empty} == {QWEN, DSV4}
    reads = [p for p in opened if str(only_catalog) in p]
    assert reads, "expected catalog files to be read"
    assert all(p.endswith(".parquet") and "/catalog/" in p for p in reads), reads
    # a b200 query touches no h200 shard (checked on its own reads, before the h200 query ran)
    assert any(p.endswith("b200_sxm.parquet") for p in worst_reads), worst_reads
    assert not any(p.endswith("h200_sxm.parquet") for p in worst_reads), worst_reads


def test_size_check_against_configurable_budget(built):
    root, _, _ = built
    report = check_catalog_size(root)
    assert report.within_budget and report.total_bytes == sum(report.shard_bytes.values())
    tight = check_catalog_size(root, CatalogBudget(soft_total=1, hard_total=2, soft_shard=1))
    assert tight.refused and not tight.within_budget
    soft = check_catalog_size(root, CatalogBudget(soft_total=1, hard_total=10**9, soft_shard=10**9))
    assert not soft.refused and not soft.within_budget


def test_size_scales_with_cells_not_payload(tmp_path):
    """10 models x 10 GPUs x 3 frameworks: shards stay under the proposed 256 KiB each, total under 1 MiB."""
    root = tmp_path / "ds"
    systems = tmp_path / "systems"
    systems.mkdir()
    import yaml

    for i in range(10):
        (systems / f"gpu{i}.yaml").write_text(yaml.safe_dump({**SPEC, "gpu": {"flops": i}}))
    build_catalog(
        root,
        systems_dir=systems,
        expected_models=[f"org/model-{i}" for i in range(10)],
        expected_frameworks=["vllm", "sglang", "trtllm"],
    )
    report = check_catalog_size(root)
    assert report.within_budget, report.findings
    assert len(Catalog(root).empty_cells("gpu3")) == 10 * 3 * 2 + 3  # fpm+e2e per model, plus op


def test_missing_catalog_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        Catalog(tmp_path)


@pytest.mark.parametrize("hardware", ["/tmp/report", "../x", "a/b", "..", ""])
def test_shard_path_rejects_hardware_that_escapes_the_catalog(tmp_path, hardware):
    with pytest.raises(ValueError, match="plain name"):
        shard_path(tmp_path, "fpm", hardware)


def test_build_rejects_a_manifest_that_disagrees_with_its_index_entry(dataset):
    root, systems = dataset
    path = root / "data/fpm/m0.json"
    manifest = Manifest.model_validate_json(path.read_bytes())
    other = manifest.identity.model_copy(update={"hardware": "h200_sxm"})
    path.write_text(manifest.model_copy(update={"identity": other}).model_dump_json(by_alias=True))
    with pytest.raises(ValueError, match="disagrees with its fpm index entry"):
        build_catalog(root, systems_dir=systems)
