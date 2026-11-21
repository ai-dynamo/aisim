# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest

from aisimulate_core.sdk.fastafd_profile import (
    FASTAFD_PROFILE_SCHEMA,
    FastAFDMoEStageKey,
    FastAFDMoEStageProfile,
)

pytestmark = pytest.mark.unit

_REPOSITORY = "https://github.com/example/FastAFD"
_COMMIT = "0" * 40


def _entry():
    return {
        "key": {
            "model_path": "deepseek-ai/DeepSeek-V4-Flash",
            "system": "gb200",
            "stage": "afd",
            "topology": "16A8F",
            "logical_batch_per_source_rank": 48,
            "mtp_nextn": 0,
            "microbatches": 2,
            "moe_layers": 43,
            "routed_topk": 6,
            "moe_precision": "w4a8_mxfp4_mxfp8",
            "moe_backend": "megamoe",
        },
        "model_profile": "deepseek_v4_flash_fp4",
        "latency_ms": 16.7,
        "measurement": {
            "scope": "complete_moe_stage",
            "method": "cuda_event",
            "method_version": "torch 2.10.0+cu130; cuda 13.0",
            "statistic": "p50",
            "sample_count": 40,
            "cv_percent": 0.4,
            "routing": "balanced",
            "source_tree_sha256": "c" * 64,
            "raw_artifact": "deepseek_v4_flash_fp4_mega_16a8f_s48_n0_mb2_balanced.json",
            "raw_sha256": "b" * 64,
        },
        "validation": {"stable": True, "correctness": None},
    }


def _payload(entries=None):
    return {
        "schema": FASTAFD_PROFILE_SCHEMA,
        "lookup_policy": "exact-only",
        "source": {
            "repository": _REPOSITORY,
            "commit": _COMMIT,
            "benchmark": "fastafd.moe-m2n-model-benchmark.v2",
            "boundary": "split AFD MoE stage",
        },
        "entries": [_entry()] if entries is None else entries,
    }


def _write(tmp_path, payload):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_exact_lookup_preserves_source_and_method(tmp_path):
    profile = FastAFDMoEStageProfile.load(_write(tmp_path, _payload()))
    key = FastAFDMoEStageKey(**_entry()["key"])
    measurement = profile.require(key)

    assert measurement.latency_ms == pytest.approx(16.7)
    assert profile.find(FastAFDMoEStageKey(**{**_entry()["key"], "topology": "32A8F"})) is None
    assert len(profile.profile_sha256) == 64
    provenance = measurement.provenance()
    assert provenance["repository"] == _REPOSITORY
    assert provenance["source_commit"] == _COMMIT
    assert provenance["benchmark"] == "fastafd.moe-m2n-model-benchmark.v2"
    assert provenance["method"] == "cuda_event"
    assert provenance["source_tree_sha256"] == "c" * 64
    assert provenance["correctness"] is None


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("repository", "", "non-empty string"),
        ("commit", "main", "invalid digest"),
        ("boundary", "", "non-empty string"),
    ],
)
def test_rejects_incomplete_source(tmp_path, field, value, match):
    payload = _payload()
    payload["source"][field] = value
    with pytest.raises(ValueError, match=match):
        FastAFDMoEStageProfile.load(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("scope", "per_rank_gpu_span", "complete_moe_stage"),
        ("method", "", "non-empty string"),
        ("statistic", "mean", "must be p50"),
        ("sample_count", 0, "integer >= 1"),
        ("cv_percent", -1.0, ">= 0.0"),
        ("source_tree_sha256", "bad", "invalid digest"),
        ("raw_sha256", "bad", "invalid digest"),
    ],
)
def test_rejects_unqualified_measurement(tmp_path, field, value, match):
    payload = _payload()
    payload["entries"][0]["measurement"][field] = value
    with pytest.raises(ValueError, match=match):
        FastAFDMoEStageProfile.load(_write(tmp_path, payload))


def test_rejects_failed_validation_and_duplicate_keys(tmp_path):
    payload = _payload()
    payload["entries"][0]["validation"]["stable"] = False
    with pytest.raises(ValueError, match="not a qualified measurement"):
        FastAFDMoEStageProfile.load(_write(tmp_path, payload))

    payload["entries"][0]["validation"] = {"stable": True, "correctness": False}
    with pytest.raises(ValueError, match="not a qualified measurement"):
        FastAFDMoEStageProfile.load(_write(tmp_path, payload))

    with pytest.raises(ValueError, match="duplicate FastAFD MoE stage key"):
        FastAFDMoEStageProfile.load(_write(tmp_path, _payload([_entry(), _entry()])))


def test_rejects_other_schemas_and_duplicate_json_fields(tmp_path):
    payload = _payload()
    payload["schema"] = "unsupported.schema"
    with pytest.raises(ValueError, match="unsupported FastAFD profile schema"):
        FastAFDMoEStageProfile.load(_write(tmp_path, payload))

    path = _write(tmp_path, _payload())
    path.write_text(path.read_text().replace('"stable": true', '"stable": true, "stable": false'))
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        FastAFDMoEStageProfile.load(path)


def test_parquet_entries_and_digest_use_the_same_snapshot(tmp_path, monkeypatch):
    profile = FastAFDMoEStageProfile.load(_write(tmp_path, _payload()))
    path = tmp_path / "stage.parquet"
    profile.write_parquet(path)
    original = path.read_bytes()
    replacement = _payload()
    replacement["entries"][0]["latency_ms"] = 24.0
    replacement_path = tmp_path / "replacement.parquet"
    FastAFDMoEStageProfile.load(_write(tmp_path, replacement)).write_parquet(replacement_path)
    read_bytes = Path.read_bytes

    def replace_after_read(self):
        raw = read_bytes(self)
        if self == path:
            replacement_path.replace(path)
        return raw

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    loaded = FastAFDMoEStageProfile.load(path)

    assert loaded.entries == profile.entries
    assert loaded.profile_sha256 == hashlib.sha256(original).hexdigest()
    assert read_bytes(path) != original


@pytest.mark.parametrize(
    "column", ["logical_batch_per_source_rank", "mtp_nextn", "microbatches", "moe_layers", "routed_topk"]
)
@pytest.mark.parametrize("encoding", ["int32", "uint32"])
def test_rejects_non_int64_parquet_counts(tmp_path, column, encoding):
    import pyarrow as pa
    import pyarrow.parquet as pq

    profile = FastAFDMoEStageProfile.load(_write(tmp_path, _payload()))
    path = tmp_path / "stage.parquet"
    profile.write_parquet(path)
    table = pq.read_table(path)
    index = table.schema.get_field_index(column)
    table = table.set_column(index, column, table[column].cast(getattr(pa, encoding)()))
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=f"count column '{column}' must be signed INT64"):
        FastAFDMoEStageProfile.load(path)


def test_int64_parquet_counts_load_through_canonical_model(tmp_path):
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    payload = _payload()
    # Synthetic complete stage: 12 ms for 3 layers and 2 microbatches,
    # or 2 ms per pipeline unit.
    payload["entries"][0]["key"]["moe_layers"] = 3
    payload["entries"][0]["latency_ms"] = 12.0
    profile = FastAFDMoEStageProfile.load(_write(tmp_path, payload))
    path = tmp_path / "stage.parquet"
    profile.write_parquet(path)
    assert FastAFDMoEStageProfile.load(path).entries == profile.entries
    config = ForwardPassPerfModelConfig(
        model=_entry()["key"]["model_path"],
        system="gb200",
        backend="sglang",
        backend_version="0.5.14",
        worker_type="decode",
        estimation_mode="fastafd_stage",
        moe_backend="megamoe",
        afd={
            "logical_batch_per_source_rank": 48,
            "attention_gpus": 16,
            "ffn_gpus": 8,
            "f_moe_ep_size": 8,
            "microbatches": 2,
            "moe_layers": 3,
            "routed_topk": 6,
            "boundary_on_attn": True,
        },
        estimator_config={"fastafd_stage": {"profile_path": str(path)}},
    )
    model = RustForwardPassPerfModel.best_available(config)
    assert model.estimate_fastafd_stage(48)["ffn_ms"] == 2.0
