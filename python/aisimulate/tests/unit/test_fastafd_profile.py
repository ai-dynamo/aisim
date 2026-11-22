# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace
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
            "boundary_components": ["dispatch", "experts", "return", "combine"],
        },
        "entries": [_entry()] if entries is None else entries,
    }


def _write(tmp_path, payload):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _canonical_config(path, *, moe_layers=43):
    from aisimulate_core.sdk import ForwardPassPerfModelConfig

    return ForwardPassPerfModelConfig(
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
            "moe_layers": moe_layers,
            "routed_topk": 6,
            "boundary_on_attn": True,
        },
        estimator_config={"fastafd_stage": {"profile_path": str(path)}},
    )


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
    assert provenance["boundary_components"] == ["dispatch", "experts", "return", "combine"]
    assert provenance["method"] == "cuda_event"
    assert provenance["source_tree_sha256"] == "c" * 64
    assert provenance["correctness"] is None


@pytest.mark.parametrize("field", ["repository", "commit", "benchmark", "boundary"])
def test_parquet_writer_rejects_mixed_sources_before_overwriting(tmp_path, field):
    profile = FastAFDMoEStageProfile.load(_write(tmp_path, _payload()))
    other = replace(profile.entries[0], source=replace(profile.source, **{field: "1" * 40}))
    mixed = FastAFDMoEStageProfile(
        (profile.entries[0], replace(other, key=replace(other.key, microbatches=3))),
        source=profile.source,
        path=profile.path,
        profile_sha256=profile.profile_sha256,
    )
    path = tmp_path / "profile.parquet"
    profile.write_parquet(path)
    original = path.read_bytes()
    assert FastAFDMoEStageProfile.load(path).entries == profile.entries

    with pytest.raises(ValueError, match="mixed FastAFD source provenance"):
        mixed.write_parquet(path)
    assert path.read_bytes() == original


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


@pytest.mark.parametrize(
    "field", ["logical_batch_per_source_rank", "mtp_nextn", "microbatches", "moe_layers", "routed_topk"]
)
@pytest.mark.parametrize("value", [2**32 - 1, 2**32])
@pytest.mark.parametrize("suffix", ["json", "parquet"])
def test_key_counts_respect_native_u32_bounds(tmp_path, field, value, suffix):
    payload = _payload()
    payload["entries"][0]["key"][field] = value
    if suffix == "json":
        path = _write(tmp_path, payload)
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq

        profile = FastAFDMoEStageProfile.load(_write(tmp_path, _payload()))
        path = tmp_path / "profile.parquet"
        profile.write_parquet(path)
        table = pq.read_table(path)
        index = table.schema.get_field_index(field)
        table = table.set_column(index, field, pa.array([value], type=pa.int64()))
        pq.write_table(table, path)

    if value == 2**32:
        with pytest.raises(ValueError, match=rf"key\.{field}.*4294967295"):
            FastAFDMoEStageProfile.load(path)
    else:
        profile = FastAFDMoEStageProfile.load(path)
        key = FastAFDMoEStageKey(**payload["entries"][0]["key"])
        assert profile.require(key).key == key


@pytest.mark.parametrize("sample_count", [2**32 - 1, 2**32, 2**64 - 1])
@pytest.mark.parametrize("suffix", ["json", "parquet"])
def test_measurement_sample_count_is_not_limited_to_u32(tmp_path, sample_count, suffix):
    from aisimulate_core.sdk import RustForwardPassPerfModel

    payload = _payload()
    payload["entries"][0]["measurement"]["sample_count"] = sample_count
    path = _write(tmp_path, payload)
    profile = FastAFDMoEStageProfile.load(path)
    if suffix == "parquet":
        path = tmp_path / "profile.parquet"
        profile.write_parquet(path)

    assert FastAFDMoEStageProfile.load(path).entries[0].sample_count == sample_count
    model = RustForwardPassPerfModel.best_available(_canonical_config(path))
    estimate = model.estimate_fastafd_stage(48)
    assert estimate["stage"]["sample_count"] == sample_count
    assert estimate["stage"]["latency_ms"] == 16.7


@pytest.mark.parametrize("suffix", ["json", "parquet"])
def test_measurement_sample_count_rejects_u64_overflow(tmp_path, suffix):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisimulate_core.sdk import RustForwardPassPerfModel

    payload = _payload()
    payload["entries"][0]["measurement"]["sample_count"] = 2**64
    if suffix == "json":
        path = _write(tmp_path, payload)
    else:
        profile = FastAFDMoEStageProfile.load(_write(tmp_path, _payload()))
        path = tmp_path / "profile.parquet"
        profile.write_parquet(path)
        table = pq.read_table(path)
        capture = json.loads(table["capture_json"][0].as_py())
        capture["measurement"]["sample_count"] = 2**64
        index = table.schema.get_field_index("capture_json")
        table = table.set_column(index, "capture_json", pa.array([json.dumps(capture)]))
        pq.write_table(table, path)

    with pytest.raises(ValueError, match=r"sample_count.*18446744073709551615"):
        FastAFDMoEStageProfile.load(path)
    with pytest.raises(ValueError, match=r"sample_count|out of range|invalid type"):
        RustForwardPassPerfModel.best_available(_canonical_config(path))


@pytest.mark.parametrize("nested_leaves", [1, 2])
def test_canonical_model_rejects_nested_parquet_before_positional_reads(tmp_path, nested_leaves):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisimulate_core.sdk import RustForwardPassPerfModel
    from aisimulate_core.sdk.errors import PerfDataNotAvailableError

    profile = FastAFDMoEStageProfile.load(_write(tmp_path, _payload()))
    path = tmp_path / "profile.parquet"
    profile.write_parquet(path)
    table = pq.read_table(path)
    # Keep every key and provenance field ahead of the nested group, so their
    # reads still succeed. A two-leaf group shifts the latency leaf index onto
    # the neighboring 24 ms DOUBLE although the actual measurement is 16.7 ms.
    table = table.select([name for name in table.column_names if name != "latency_ms"] + ["latency_ms"])
    nested = pa.array([{f"leaf_{i}": 0.0 for i in range(nested_leaves)}])
    table = table.add_column(len(table.column_names) - 1, "nested", nested)
    table = table.append_column("neighbor_ms", pa.array([24.0]))
    pq.write_table(table, path)

    with pytest.raises(PerfDataNotAvailableError, match="flat parquet schema"):
        RustForwardPassPerfModel.best_available(_canonical_config(path))


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


@pytest.mark.parametrize("location", ["direct", "symlink", "extensionless"])
@pytest.mark.parametrize("profile_format", ["json", "parquet", "parquet-float"])
@pytest.mark.parametrize("mode_field", ["estimation_mode", "forward_model"])
def test_profile_loads_through_canonical_model_preserving_source(tmp_path, location, profile_format, mode_field):
    from aisimulate_core.sdk import RustForwardPassPerfModel

    payload = _payload()
    # Synthetic complete stage: 12 ms for 3 layers and 2 microbatches,
    # or 2 ms per pipeline unit.
    payload["entries"][0]["key"]["moe_layers"] = 3
    payload["entries"][0]["latency_ms"] = 12.0
    path = _write(tmp_path, payload)
    profile = FastAFDMoEStageProfile.load(path)
    if profile_format != "json":
        path = tmp_path / "stage.parquet"
        profile.write_parquet(path)
        if profile_format == "parquet-float":
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            index = table.schema.get_field_index("latency_ms")
            table = table.set_column(index, "latency_ms", table.column(index).cast(pa.float32()))
            pq.write_table(table, path)
    if location != "direct":
        target = path.with_suffix("")
        path.rename(target)
        path.symlink_to(target)
        if location == "extensionless":
            path = target
    assert FastAFDMoEStageProfile.load(path).entries == profile.entries
    config = _canonical_config(path, moe_layers=3)
    if mode_field == "forward_model":
        config = config.to_dict()
        config["forward_model"] = config.pop("estimation_mode")
    model = RustForwardPassPerfModel.best_available(config)
    estimate = model.estimate_fastafd_stage(48)
    assert estimate["ffn_ms"] == 2.0
    assert estimate["stage"]["boundary_components"] == ["dispatch", "experts", "return", "combine"]
    assert estimate["stage"]["profile_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert estimate["stage"]["profile_path"] == str(path.resolve())
    saved = model.diagnostics()["provenance"]["config"]
    assert saved["estimator_config"]["fastafd_stage"]["profile_path"] == str(path.resolve())
    restored = RustForwardPassPerfModel.best_available(saved)
    assert restored.estimate_fastafd_stage(48) == model.estimate_fastafd_stage(48)
    with pytest.raises(ValueError, match="cannot be tuned"):
        model.tune_with_fpms([])


@pytest.mark.parametrize(
    "components",
    [None, ["dispatch", "experts", "return"], ["router", "dispatch", "experts", "return", "combine"]],
)
def test_inspection_rejects_unverified_or_incompatible_boundary(tmp_path, components):
    payload = _payload()
    payload["source"]["boundary_components"] = components
    with pytest.raises(ValueError, match="boundary_components"):
        FastAFDMoEStageProfile.load(_write(tmp_path, payload))
