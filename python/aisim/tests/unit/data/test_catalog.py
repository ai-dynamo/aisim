# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
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
from tests.unit.data.test_fetch import _repin

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_coverage_replacement_preserves_previous_declaration(tmp_path, monkeypatch, failure):
    root, systems = tmp_path / "dataset", tmp_path / "systems"
    systems.mkdir()
    build_catalog(root, systems_dir=systems, expected_models=["original"], expected_frameworks=["vllm"])
    path = root / "catalog/coverage.json"
    before = path.read_bytes()
    original = catalog_module.tempfile.NamedTemporaryFile

    original_replace = catalog_module.os.replace
    replacement_reached = []

    def fail_replace(src, dst):
        if dst == path:
            replacement_reached.append(True)
            raise OSError("synthetic replacement failure")
        return original_replace(src, dst)

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
    if failure == "replace":
        assert replacement_reached == [True]
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


def _file_artifact(root, path):
    """The artifact record of a file already written under ``root`` (real size and digest)."""
    return _artifact(path, (root / path).read_bytes())


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
            {
                "configuration_id": cid,
                "purpose": purpose,
                "identity": ident.model_dump(),
                "manifest": _file_artifact(root, path),
            }
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
    ref = lambda c, rel: {
        "root": f"data/{c}",
        "index": _file_artifact(root, rel).model_dump(by_alias=True),
        "layout": "category",
    }
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
    sys.addaudithook(lambda event, args: opened.append(str(args[0])) if event == "open" else None)
    cat = Catalog(only_catalog)
    worst = cat.worst_cells("b200_sxm", n=2)
    empty = cat.empty_cells("h200_sxm", layer="fpm")

    assert [(r.model, r.layer, r.acc_mape_pct) for r in worst] == [(DSV4, "fpm", 12.0), (QWEN, "fpm", 9.0)]
    assert {r.model for r in empty} == {QWEN, DSV4}
    reads = [p for p in opened if str(only_catalog) in p]
    assert reads, "expected catalog files to be read"
    assert all(p.endswith(".parquet") and "/catalog/" in p for p in reads), reads
    # hardware-scoped queries touch only that hardware's shards
    assert not any("h200_sxm" in p for p in reads if "b200" in p)


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


def test_a_manifest_edited_after_indexing_is_not_counted(dataset):
    root, systems = dataset
    path = root / "data/fpm/m0.json"
    manifest = Manifest.model_validate_json(path.read_bytes())
    # Schema-valid edit that keeps the identity: only the evidence gaps change.
    path.write_text(
        manifest.model_copy(update={"evidence_gaps": [{"field": "new"}] * 7}).model_dump_json(by_alias=True)
    )
    with pytest.raises(ValueError, match="integrity mismatch"):
        build_catalog(root, systems_dir=systems)


@pytest.mark.parametrize("field", ["identity", "category"])
def test_catalog_rejects_a_digest_pinned_manifest_for_another_cell(dataset, field):
    root, systems = dataset
    path = "data/fpm/m0.json"
    doc = json.loads((root / path).read_bytes())
    if field == "identity":
        doc["identity"]["hardware"] = "h100_sxm"
    else:
        doc["category"] = "e2e"
    (root / path).write_text(json.dumps(doc))
    index_path = root / "catalog/fpm.json"
    index = json.loads(index_path.read_bytes())
    index["entries"][0]["manifest"] = _file_artifact(root, path).model_dump(by_alias=True)
    index_path.write_text(json.dumps(index))
    dataset_path = root / "catalog/dataset.json"
    document = json.loads(dataset_path.read_bytes())
    document["categories"]["fpm"]["index"] = _file_artifact(root, "catalog/fpm.json").model_dump(by_alias=True)
    dataset_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="catalog and manifest identity mismatch"):
        build_catalog(root, systems_dir=systems)


def test_a_category_index_edited_after_indexing_is_refused(dataset):
    root, systems = dataset
    index = root / "catalog/e2e.json"
    index.write_text(index.read_text() + "\n")
    with pytest.raises(ValueError, match="integrity mismatch"):
        build_catalog(root, systems_dir=systems)


@pytest.mark.parametrize("target", ["catalog/fpm.json", "data/fpm/m0.json"])
def test_catalog_parses_the_bytes_that_passed_verification(dataset, monkeypatch, target):
    root, systems = dataset
    victim = root / target
    original = victim.read_bytes()
    changed = json.loads(original)
    if target.startswith("catalog/"):
        changed["entries"] = []
    else:
        changed["evidence_gaps"] = []
    replacement = json.dumps(changed, separators=(",", ":")).encode()
    replacement += b" " * (len(original) - len(replacement))
    assert len(replacement) == len(original)
    real_open = type(victim).open
    replaced = False

    def swap_after_open(path, *args, **kwargs):
        nonlocal replaced
        stream = real_open(path, *args, **kwargs)
        if path == victim and not replaced and kwargs.get("mode", args[0] if args else "r") == "rb":
            replaced = True
            temporary = victim.with_suffix(".replacement")
            with real_open(temporary, "wb") as out:
                out.write(replacement)
            os.replace(temporary, victim)
        return stream

    monkeypatch.setattr(type(victim), "open", swap_after_open)
    rows = build_catalog(root, systems_dir=systems)
    assert replaced
    cell = _cell(rows, "fpm", QWEN, "b200_sxm")
    assert cell.n_configs == 1 and cell.n_evidence_gaps == 2


@pytest.mark.parametrize(
    "field,value", [("identity", "h200_sxm"), ("purpose", "operation-tables"), ("category", "e2e")]
)
def test_catalog_rejects_digest_valid_manifest_identity_mismatches(dataset, field, value):
    root, systems = dataset
    files = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*.json")}
    path = "data/fpm/m0.json"
    document = json.loads(files[path])
    if field == "identity":
        document[field]["hardware"] = value
    else:
        document[field] = value
    files[path] = json.dumps(document).encode()
    # Refresh both enclosing pins: digest integrity alone cannot establish identity equality.
    index = json.loads(files["catalog/fpm.json"])
    index["entries"][0]["manifest"] = _artifact(path, files[path]).model_dump(by_alias=True)
    files["catalog/fpm.json"] = json.dumps(index).encode()
    _repin(files, "catalog/fpm.json")
    for name, data in files.items():
        (root / name).write_bytes(data)
    with pytest.raises(ValueError, match="catalog and manifest identity mismatch"):
        build_catalog(root, systems_dir=systems)


@pytest.mark.parametrize("hardware", ["/tmp/victim", "../escape", "a/b", "..", ""])
def test_hardware_cannot_name_a_path_outside_the_shard_directory(tmp_path, hardware):
    with pytest.raises(ValueError, match="path component"):
        shard_path(tmp_path, "fpm", hardware)


def test_the_latest_accuracy_is_chosen_by_time_not_by_string(dataset):
    older = _record(_ident(QWEN), "fpm", 1.0, NEW, "2026-11-01T00:00:00Z")
    newer = _record(_ident(QWEN), "fpm", 2.0, NEW, "2026-11-01T00:00:00.1Z")
    assert "2026-11-01T00:00:00Z" > "2026-11-01T00:00:00.1Z"  # as strings the older one sorts last
    for order in ([older, newer], [newer, older]):
        assert catalog_module._latest_accuracy(order).results.mape_pct == 2.0


def test_a_manifest_beside_eval_records_is_not_loaded_as_a_record(dataset):
    from aisim_core.data.eval import record_dir

    root, systems = dataset
    directory = record_dir(root, _ident(QWEN))
    (directory / "manifest.json").write_text('{"schema_version": "aisim-artifact-manifest/v1"}')
    rows = build_catalog(root, systems_dir=systems)
    assert any(r.model == QWEN and r.acc_mape_pct is not None for r in rows)
    # Corrupt content at a valid record filename is still refused.
    victim = next(p for p in directory.glob("*.json") if p.name != "manifest.json")
    victim.write_text(victim.read_text().replace("2026-11", "2027-01"))
    with pytest.raises(ValueError, match="content hash"):
        build_catalog(root, systems_dir=systems)


def test_build_rejects_a_manifest_that_disagrees_with_its_index_entry(dataset):
    root, systems = dataset
    path = root / "data/fpm/m0.json"
    manifest = Manifest.model_validate_json(path.read_bytes())
    other = manifest.identity.model_copy(update={"hardware": "h200_sxm"})
    path.write_text(manifest.model_copy(update={"identity": other}).model_dump_json(by_alias=True))
    with pytest.raises(ValueError, match="disagrees with its fpm index entry"):
        build_catalog(root, systems_dir=systems)


def test_catalog_versions_preserve_v1_and_write_v2(tmp_path):
    import pyarrow as pa
    from pydantic import ValidationError

    from aisim_core.data.catalog import CATALOG_SCHEMA_ID, CatalogRowV1
    from aisim_core.data.parquet import ParquetSchemaError, read_parquet, write_parquet
    from aisim_core.data.schemas import SCHEMA_MODELS

    legacy = CatalogRowV1(
        layer="e2e",
        model="synthetic/model",
        hardware="h100_sxm",
        framework="vllm",
        status="have",
        n_configs=1,
        n_evidence_gaps=0,
        n_eval_records=0,
    )
    path = shard_path(tmp_path, "e2e", "h100_sxm")
    write_parquet(pa.Table.from_pylist([legacy.model_dump()]), path, "aisim-catalog/v1")
    row = Catalog(tmp_path).rows(hardware="h100_sxm", layer="e2e")[0]
    assert row.acc_prediction_fingerprint is None
    row.acc_prediction_fingerprint = "a" * 64
    catalog_module._write(tmp_path, [row], {})
    assert CATALOG_SCHEMA_ID == "aisim-catalog/v2"
    assert read_parquet_schema_id(path) == CATALOG_SCHEMA_ID
    assert Catalog(tmp_path).rows(hardware="h100_sxm", layer="e2e")[0] == row
    with pytest.raises(ParquetSchemaError):
        read_parquet(path, "aisim-catalog/v1")  # an older reader rejects the new version explicitly
    with pytest.raises(ValidationError):
        SCHEMA_MODELS["aisim-catalog/v1"].model_validate(row.model_dump())
    assert "acc_prediction_fingerprint" not in SCHEMA_MODELS["aisim-catalog/v1"].model_fields
