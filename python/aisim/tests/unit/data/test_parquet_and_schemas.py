# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from jsonschema import Draft202012Validator

from aisim_core.data.parquet import (
    SCHEMA_METADATA_KEY,
    ParquetSchemaError,
    read_parquet,
    read_parquet_schema_id,
    write_parquet,
)
from aisim_core.data.schemas import (
    PACKAGED_SNAPSHOT_DIR,
    SCHEMA_MODELS,
    check_snapshots,
    render_all,
    render_snapshot,
    snapshot_filename,
    write_snapshots,
)

pytestmark = pytest.mark.unit

SCHEMA = "aisim-e2e-test/v1"


def _table():
    return pa.table({"a": [1, 2, None], "b": ["x", None, "z"], "c": [1.5, None, 2.5]})


def test_round_trip_zstd_with_schema_id(tmp_path):
    path = write_parquet(_table(), tmp_path / "sub" / "t.parquet", SCHEMA, extra_metadata={"k": "v"})
    assert read_parquet_schema_id(path) == SCHEMA
    assert read_parquet(path, SCHEMA).equals(_table().replace_schema_metadata(None))
    meta = pq.read_metadata(path)
    assert meta.row_group(0).column(0).compression == "ZSTD"
    assert meta.metadata[b"k"] == b"v"
    assert not list(path.parent.glob(".*.tmp"))


def test_nulls_stay_null(tmp_path):
    out = read_parquet(write_parquet(_table(), tmp_path / "t.parquet", SCHEMA), SCHEMA)
    assert out.column("a").null_count == 1 and out.column("c").null_count == 1


def test_schema_mismatch_and_missing_rejected(tmp_path):
    path = write_parquet(_table(), tmp_path / "t.parquet", SCHEMA)
    with pytest.raises(ParquetSchemaError):
        read_parquet(path, "other/v1")
    assert read_parquet(path, ("other/v1", SCHEMA)).num_rows == 3
    bare = tmp_path / "bare.parquet"
    pq.write_table(_table(), bare)
    assert read_parquet_schema_id(bare) is None
    with pytest.raises(ParquetSchemaError):
        read_parquet(bare, SCHEMA)


def test_cannot_override_schema_key(tmp_path):
    with pytest.raises(ValueError):
        write_parquet(_table(), tmp_path / "t.parquet", SCHEMA, extra_metadata={SCHEMA_METADATA_KEY.decode(): "x"})
    with pytest.raises(ValueError):
        write_parquet(_table(), tmp_path / "t.parquet", "")


def test_failed_write_leaves_no_file(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pq, "write_table", boom)
    with pytest.raises(RuntimeError):
        write_parquet(_table(), tmp_path / "t.parquet", SCHEMA)
    assert list(tmp_path.iterdir()) == []


def test_writes_are_deterministic(tmp_path):
    a = write_parquet(_table(), tmp_path / "a.parquet", SCHEMA).read_bytes()
    b = write_parquet(_table(), tmp_path / "b.parquet", SCHEMA).read_bytes()
    assert a == b


def test_packaged_snapshots_have_no_drift():
    assert check_snapshots(PACKAGED_SNAPSHOT_DIR) == []


def test_snapshots_are_valid_json_schema():
    for text in render_all().values():
        Draft202012Validator.check_schema(json.loads(text))
    assert set(render_all()) == {snapshot_filename(s) for s in SCHEMA_MODELS}


def test_drift_is_detected(tmp_path):
    write_snapshots(tmp_path)
    assert check_snapshots(tmp_path) == []
    victim = next(tmp_path.glob("*.json"))
    victim.write_text(victim.read_text() + " ")
    (tmp_path / "stray.v9.json").write_text("{}")
    problems = check_snapshots(tmp_path)
    assert any("differs" in p for p in problems) and any("unexpected" in p for p in problems)
    victim.unlink()
    assert any("missing" in p for p in check_snapshots(tmp_path))
    # write_snapshots repairs drift and removes stale files.
    write_snapshots(tmp_path)
    assert check_snapshots(tmp_path) == []


def test_check_on_missing_dir(tmp_path):
    assert len(check_snapshots(tmp_path / "nope")) == len(SCHEMA_MODELS)


def test_gpu_time_schema_registered():
    from aisim_core.data.schemas import SCHEMA_MODELS

    assert "aisim-gpu-time/v1" in SCHEMA_MODELS


# ---------------------------------------------------------------- schema constraints mirror the model validators


def _validator(schema_id):
    return Draft202012Validator(json.loads(render_snapshot(schema_id)))


def _metric(**over):
    row = {"observation_id": "o", "metric": "ttft", "statistic": "mean", "unit": "ms", "status": "valid", "value": 1.0}
    return {**row, **over}


@pytest.mark.parametrize(
    ("document", "valid"),
    [
        (_metric(), True),
        (_metric(status="missing", value=None), True),
        (_metric(value=None), False),  # valid without a value
        (_metric(status="missing", value=0), False),  # missing with a value
    ],
)
def test_metric_schema_matches_status_value_rule(document, valid):
    from aisim_core.data.e2e import MetricRow

    assert _validator("aisim-e2e-metrics/v1").is_valid(document) is valid
    try:
        MetricRow.model_validate(document)
        accepted = True
    except ValueError:
        accepted = False
    assert accepted is valid


@pytest.mark.parametrize(
    ("document", "valid"),
    [
        ({"authority": "unknown"}, True),
        ({"authority": "measured", "gpu_count": 8, "wall_seconds": 10.0}, True),
        ({"authority": "measured"}, False),  # no GPU count or time
        ({"authority": "unknown", "gpu_seconds": 12}, False),  # unknown must carry nothing
    ],
)
def test_gpu_time_schema_matches_authority_contract(document, valid):
    from aisim_core.data.gpu_time import GpuTime

    assert _validator("aisim-gpu-time/v1").is_valid(document) is valid
    try:
        GpuTime.model_validate(document)
        accepted = True
    except ValueError:
        accepted = False
    assert accepted is valid


@pytest.mark.parametrize("path", ["../outside", "/abs", "a//b", "a/./b", "a\\b", ""])
def test_artifact_path_restriction_is_published(path):
    from aisim_core.data import RELATIVE_PATH_PATTERN, Artifact

    assert re.match(RELATIVE_PATH_PATTERN, path) is None
    document = {
        "artifact_id": "a",
        "path": path,
        "role": "r",
        "format": "json",
        "schema": "s/v1",
        "sha256": "0" * 64,
        "bytes": 0,
    }
    with pytest.raises(ValueError):
        Artifact.model_validate(document)
    schema = json.loads(render_snapshot("aisim-artifact-manifest/v1"))
    assert not Draft202012Validator({"$ref": "#/$defs/Artifact", "$defs": schema["$defs"]}).is_valid(document)
    document["path"] = "data/ok.json"
    assert Draft202012Validator({"$ref": "#/$defs/Artifact", "$defs": schema["$defs"]}).is_valid(document)


def test_verify_compares_the_stored_schema_id_with_the_manifest_declaration(tmp_path):
    import hashlib

    from aisim_core.data import Artifact, _verify

    path = write_parquet(_table(), tmp_path / "t.parquet", SCHEMA)
    data = path.read_bytes()
    declared = dict(
        artifact_id="t",
        path="t.parquet",
        role="r",
        format="parquet",
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        rows=_table().num_rows,
    )
    assert _verify(tmp_path, Artifact(**declared, schema=SCHEMA)) == path
    with pytest.raises(ValueError, match="schema ID differs"):
        _verify(tmp_path, Artifact(**declared, schema="aisim-other/v1"))
