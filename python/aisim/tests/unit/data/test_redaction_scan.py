# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit checks for the disclosure scanner's credential, name and URL rules."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisim_core.data.redaction import DEFAULT_POLICY, _check_table_text, check_disclosure, scan_json_file, scan_value

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("key", ["AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "DOCKER_AUTH_CONFIG"])
def test_cloud_and_registry_credential_fields_are_refused(key):
    assert {f.code for f in scan_value({key: "abc123"}, DEFAULT_POLICY, "p", "w")} == {"secret"}
    assert not scan_value({key: "[REDACTED]"}, DEFAULT_POLICY, "p", "w")


def test_aggregate_text_rejects_urls_but_keeps_plain_status():
    row = {"n_scored": 5, "status": "complete"}
    assert not check_disclosure([row])
    row["status"] = "See https://benchmark.example/report"
    assert "not-aggregate" in {f.code for f in check_disclosure([row])}


def test_repeated_key_sidecar_is_refused_not_scanned_as_last_value(tmp_path):
    path = tmp_path / "s.json"
    path.write_text('{"a": 1, "a": 2}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        scan_json_file(path, DEFAULT_POLICY, "s.json", "JSON sidecar")


@pytest.mark.parametrize("nested", [False, True])
def test_field_names_are_scanned_without_rows_or_values(tmp_path, nested):
    name = "token_hf_" + "a" * 24
    dtype = pa.struct([pa.field(name, pa.string())]) if nested else pa.string()
    top = "col" if nested else name
    table = pa.table({top: pa.array([], type=dtype)})
    path = tmp_path / "t.parquet"
    pq.write_table(table, path)
    codes = {f.code for f in _check_table_text(table, "t.parquet", DEFAULT_POLICY, "column", path=path)}
    assert "secret" in codes
    clean = pa.table({"col": pa.array([], type=pa.string())})
    pq.write_table(clean, path)
    assert not _check_table_text(clean, "t.parquet", DEFAULT_POLICY, "column", path=path)
