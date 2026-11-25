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
            "timed_span": "all_moe_layers_and_microbatches",
            "microbatch_split": "equal_padded",
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
    assert provenance["timed_span"] == "all_moe_layers_and_microbatches"
    assert provenance["method"] == "cuda_event"
    assert provenance["source_tree_sha256"] == "c" * 64
    assert provenance["correctness"] is None


@pytest.mark.parametrize("padding", [" ", "\x1c", "\x1d", "\x1e", "\x1f", "\u00a0"])
@pytest.mark.parametrize("field", ["model_path", "system", "stage", "topology", "moe_precision", "moe_backend"])
def test_key_trimming_agrees_for_inspection_and_canonical_estimation(tmp_path, padding, field):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisimulate_core.sdk import RustForwardPassPerfModel

    payload = _payload()
    expected_key = FastAFDMoEStageKey(**payload["entries"][0]["key"])
    raw_value = f"{padding}{getattr(expected_key, field)}{padding}"
    payload["entries"][0]["key"][field] = raw_value
    json_path = _write(tmp_path, payload)
    profile = FastAFDMoEStageProfile.load(json_path)
    parquet_path = tmp_path / "stage.parquet"
    profile.write_parquet(parquet_path)
    table = pq.read_table(parquet_path)
    # Preserve the raw padded key in direct Parquet input, before either
    # loader has a chance to normalize it.
    table = table.set_column(table.schema.get_field_index(field), field, pa.array([raw_value]))
    pq.write_table(table, parquet_path)

    for path in (json_path, parquet_path):
        inspected = FastAFDMoEStageProfile.load(path).require(expected_key)
        assert inspected.key == expected_key
        # JSON follows the legacy-to-Parquet conversion; Parquet goes straight
        # to Rust. Both must select the same unpadded key as Python inspection.
        estimate = RustForwardPassPerfModel.best_available(_canonical_config(path)).estimate_fastafd_stage(48)
        assert FastAFDMoEStageKey(**estimate["stage"]["key"]) == expected_key
        assert estimate["stage"]["latency_ms"] == inspected.latency_ms == 16.7


def test_parquet_writer_preserves_direct_integer_latency_through_canonical_estimator(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisimulate_core.sdk import RustForwardPassPerfModel

    profile = FastAFDMoEStageProfile.load(_write(tmp_path, _payload()))
    # Direct construction accepts an integer latency. This synthetic complete
    # stage has six pipeline units (3 layers x 2 microbatches), each costing 2 ms.
    measurement = replace(profile.entries[0], latency_ms=12, key=replace(profile.entries[0].key, moe_layers=3))
    direct = FastAFDMoEStageProfile(
        (measurement,), source=profile.source, path=profile.path, profile_sha256=profile.profile_sha256
    )
    path = tmp_path / "stage.parquet"
    direct.write_parquet(path)

    schema = pq.read_schema(path)
    assert schema.field("latency_ms").type == pa.float64()
    for field in ("logical_batch_per_source_rank", "mtp_nextn", "microbatches", "moe_layers", "routed_topk"):
        assert schema.field(field).type == pa.int64()
    assert FastAFDMoEStageProfile.load(path).entries == (measurement,)
    estimate = RustForwardPassPerfModel.best_available(_canonical_config(path, moe_layers=3)).estimate_fastafd_stage(48)
    assert estimate["stage"]["latency_ms"] == 12.0
    assert estimate["ffn_ms"] == 2.0


@pytest.mark.parametrize("suffix", ["json", "parquet"])
def test_canonical_estimator_rejects_latency_underflow(tmp_path, suffix):
    from aisimulate_core.sdk import RustForwardPassPerfModel

    payload = _payload()
    payload["entries"][0]["latency_ms"] = float.fromhex("0x0.0000000000001p-1022")
    path = _write(tmp_path, payload)
    if suffix == "parquet":
        profile = FastAFDMoEStageProfile.load(path)
        path = path.with_suffix(".parquet")
        profile.write_parquet(path)
    model = RustForwardPassPerfModel.best_available(_canonical_config(path))
    with pytest.raises(ValueError, match="normalized per-unit latency must be finite and positive"):
        model.estimate_fastafd_stage(48)


def test_canonical_estimator_rejects_unused_fpm_attention_selector(tmp_path):
    from aisimulate_core.sdk import RustForwardPassPerfModel

    config = _canonical_config(_write(tmp_path, _payload())).to_dict()
    config["fpm_fmha_quant_mode"] = "fp8"
    with pytest.raises(ValueError, match="no unrelated estimator/graph controls"):
        RustForwardPassPerfModel.best_available(config)


def test_entries_cannot_be_replaced_after_exact_index_is_built(tmp_path):
    other = _entry()
    other["key"]["microbatches"] = 3
    profile = FastAFDMoEStageProfile.load(_write(tmp_path, _payload([_entry(), other])))
    first, second = profile.entries
    assert profile.require(first.key) is first
    with pytest.raises(AttributeError):
        profile.entries = (second, first)
    assert profile.require(first.key) is first
    assert profile.require(second.key) is second


@pytest.mark.parametrize("codec", ["NONE", "SNAPPY", "ZSTD", "GZIP"])
def test_parquet_codec_inspection_agrees_with_native_reader(tmp_path, codec):
    import pyarrow.parquet as pq

    from aisimulate_core.sdk import RustForwardPassPerfModel

    profile = FastAFDMoEStageProfile.load(_write(tmp_path, _payload()))
    path = tmp_path / "profile.parquet"
    profile.write_parquet(path)
    table = pq.read_table(path)
    # Compress just one column differently, so inspection must check every
    # column rather than infer codec support from the first column alone.
    pq.write_table(table, path, compression={"latency_ms": codec})
    if codec == "GZIP":
        with pytest.raises(ValueError, match="unsupported FastAFD Parquet codec: GZIP"):
            FastAFDMoEStageProfile.load(path)
        with pytest.raises(ValueError, match="(?i)gzip|flate2"):
            RustForwardPassPerfModel.best_available(_canonical_config(path))
    else:
        assert FastAFDMoEStageProfile.load(path).entries == profile.entries
        estimate = RustForwardPassPerfModel.best_available(_canonical_config(path)).estimate_fastafd_stage(48)
        assert estimate["stage"]["latency_ms"] == 16.7


@pytest.mark.parametrize("stage,topology", [("agg", "ep{}"), ("afd", "{}A8F"), ("afd", "16A{}F")])
@pytest.mark.parametrize("value", [2**32 - 1, 2**32])
@pytest.mark.parametrize("suffix", ["json", "parquet"])
def test_topology_bounds_validate_every_row_before_selection(tmp_path, stage, topology, value, suffix):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisimulate_core.sdk import RustForwardPassPerfModel

    other = _entry()
    other["key"].update(stage=stage, topology=topology.format(value))
    if suffix == "json":
        path = _write(tmp_path, _payload([_entry(), other]))
    else:
        profile = FastAFDMoEStageProfile.load(_write(tmp_path, _payload()))
        path = tmp_path / "profile.parquet"
        profile.write_parquet(path)
        rows = pq.read_table(path).to_pylist()
        rows.append({**rows[0], "stage": stage, "topology": other["key"]["topology"]})
        pq.write_table(pa.Table.from_pylist(rows), path)

    if value == 2**32:
        with pytest.raises(ValueError, match=r"entry 1.key.topology.*4294967295"):
            FastAFDMoEStageProfile.load(path)
        # The valid first row is still an exact match. Native loading must
        # reject the malformed neighboring row instead of selecting past it.
        with pytest.raises(ValueError, match="unqualified FastAFD measurement|key.topology"):
            RustForwardPassPerfModel.best_available(_canonical_config(path))
    else:
        profile = FastAFDMoEStageProfile.load(path)
        assert profile.require(FastAFDMoEStageKey(**other["key"])).key.topology == other["key"]["topology"]
        estimate = RustForwardPassPerfModel.best_available(_canonical_config(path)).estimate_fastafd_stage(48)
        assert estimate["stage"]["key"]["topology"] == "16A8F"


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


@pytest.mark.parametrize("nested_kind", ["struct", "struct_pair", "list", "map"])
def test_inspection_and_canonical_model_agree_on_flat_parquet_schema(tmp_path, nested_kind):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisimulate_core.sdk import RustForwardPassPerfModel
    from aisimulate_core.sdk.errors import PerfDataNotAvailableError

    profile = FastAFDMoEStageProfile.load(_write(tmp_path, _payload()))
    path = tmp_path / "profile.parquet"
    profile.write_parquet(path)
    table = pq.read_table(path)
    # An unused scalar column is valid, even when its literal name has a dot.
    table = table.append_column("unused.scalar", pa.array([0.0]))
    pq.write_table(table, path)
    assert FastAFDMoEStageProfile.load(path).entries == profile.entries
    estimate = RustForwardPassPerfModel.best_available(_canonical_config(path)).estimate_fastafd_stage(48)
    assert estimate["stage"]["latency_ms"] == 16.7
    # Keep every key and provenance field ahead of the nested group, so their
    # reads still succeed. A two-leaf group shifts the latency leaf index onto
    # the neighboring 24 ms DOUBLE although the actual measurement is 16.7 ms.
    table = table.select([name for name in table.column_names if name != "latency_ms"] + ["latency_ms"])
    nested = {
        "struct": pa.array([{"leaf": 0.0}]),
        "struct_pair": pa.array([{"leaf": 0.0, "other": 0.0}]),
        "list": pa.array([[0.0]]),
        "map": pa.array([[("leaf", 0.0)]], type=pa.map_(pa.string(), pa.float64())),
    }[nested_kind]
    table = table.add_column(len(table.column_names) - 1, "nested", nested)
    table = table.append_column("neighbor_ms", pa.array([24.0]))
    pq.write_table(table, path)

    with pytest.raises(ValueError, match="flat parquet schema"):
        FastAFDMoEStageProfile.load(path)
    with pytest.raises(PerfDataNotAvailableError, match="flat parquet schema"):
        RustForwardPassPerfModel.best_available(_canonical_config(path))


def test_inspection_and_native_loading_reject_legacy_repeated_primitive(tmp_path):
    import base64
    import zlib

    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisimulate_core.sdk import RustForwardPassPerfModel
    from aisimulate_core.sdk.errors import PerfDataNotAvailableError

    # Self-authored synthetic _payload(), written with parquet-rs 59.3.0's
    # SerializedFileWriter. All 14 profile fields are REQUIRED, plus one unused
    # REPEATED DOUBLE (one zero value, definition level 1, repetition level 0).
    # PyArrow writes lists as groups and cannot generate this legacy encoding.
    # One row, no compression; latency 16.7 ms, counts and provenance as _payload.
    encoded = (
        "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"
    )
    path = tmp_path / "legacy.parquet"
    path.write_bytes(zlib.decompress(base64.b64decode(encoded)))
    schema = pq.ParquetFile(path).schema
    assert len(schema) == 15
    assert schema.column(14).path == "unused"
    assert schema.column(14).max_repetition_level == 1
    assert pa.types.is_list(pq.read_schema(path).field("unused").type)
    with pytest.raises(ValueError, match="flat parquet schema"):
        FastAFDMoEStageProfile.load(path)
    with pytest.raises(PerfDataNotAvailableError, match="flat parquet schema"):
        RustForwardPassPerfModel.best_available(_canonical_config(path))

    # The same complete profile with an unused scalar DOUBLE remains valid.
    profile = FastAFDMoEStageProfile.load(_write(tmp_path, _payload()))
    table = pq.read_table(path).drop(["unused"]).append_column("unused", pa.array([0.0]))
    pq.write_table(table, path)
    assert FastAFDMoEStageProfile.load(path).entries == profile.entries
    estimate = RustForwardPassPerfModel.best_available(_canonical_config(path)).estimate_fastafd_stage(48)
    assert estimate["stage"]["latency_ms"] == 16.7


@pytest.mark.parametrize("span", [None, "one_microbatch", "one_moe_layer", 1])
@pytest.mark.parametrize("suffix", ["json", "parquet"])
@pytest.mark.parametrize("field", ["timed_span", "microbatch_split"])
def test_inspection_and_native_loading_require_capture_contract(tmp_path, span, suffix, field):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisimulate_core.sdk import RustForwardPassPerfModel

    payload = _payload()
    if span is None:
        del payload["source"][field]
    else:
        payload["source"][field] = span
    if suffix == "json":
        path = _write(tmp_path, payload)
    else:
        profile = FastAFDMoEStageProfile.load(_write(tmp_path, _payload()))
        path = tmp_path / "profile.parquet"
        profile.write_parquet(path)
        table = pq.read_table(path)
        index = table.schema.get_field_index("source_json")
        table = table.set_column(index, "source_json", pa.array([json.dumps(payload["source"])]))
        pq.write_table(table, path)
    with pytest.raises(ValueError, match=field):
        FastAFDMoEStageProfile.load(path)
    with pytest.raises(ValueError, match=f"{field}|invalid type"):
        RustForwardPassPerfModel.best_available(_canonical_config(path))


@pytest.mark.parametrize("column", ["capture_json", "source_json"])
@pytest.mark.parametrize("encoding", ["binary", "string", "large_string"])
def test_parquet_metadata_requires_utf8_for_inspection_and_native_loading(tmp_path, column, encoding):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisimulate_core.sdk import RustForwardPassPerfModel

    profile = FastAFDMoEStageProfile.load(_write(tmp_path, _payload()))
    path = tmp_path / "profile.parquet"
    profile.write_parquet(path)
    table = pq.read_table(path)
    table = table.set_column(table.schema.get_field_index(column), column, table[column].cast(getattr(pa, encoding)()))
    pq.write_table(table, path)
    if encoding == "binary":
        with pytest.raises(ValueError, match=rf"{column}.*UTF-8"):
            FastAFDMoEStageProfile.load(path)
        with pytest.raises(ValueError, match="(?i)string|str"):
            RustForwardPassPerfModel.best_available(_canonical_config(path))
    else:
        assert FastAFDMoEStageProfile.load(path).entries == profile.entries
        estimate = RustForwardPassPerfModel.best_available(_canonical_config(path)).estimate_fastafd_stage(48)
        assert estimate["stage"]["latency_ms"] == 16.7


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
@pytest.mark.parametrize("encoding", ["int32", "uint32", "timestamp"])
def test_rejects_non_int64_parquet_counts(tmp_path, column, encoding):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisimulate_core.sdk import RustForwardPassPerfModel
    from aisimulate_core.sdk.errors import PerfDataNotAvailableError

    profile = FastAFDMoEStageProfile.load(_write(tmp_path, _payload()))
    path = tmp_path / "stage.parquet"
    profile.write_parquet(path)
    table = pq.read_table(path)
    index = table.schema.get_field_index(column)
    data_type = pa.timestamp("ns") if encoding == "timestamp" else getattr(pa, encoding)()
    table = table.set_column(index, column, table[column].cast(data_type))
    pq.write_table(table, path)

    with pytest.raises(ValueError, match=f"count column '{column}' must be signed INT64"):
        FastAFDMoEStageProfile.load(path)
    with pytest.raises(PerfDataNotAvailableError, match=f'count column "{column}" must be signed INT64'):
        RustForwardPassPerfModel.best_available(_canonical_config(path))


@pytest.mark.parametrize("location", ["direct", "symlink", "extensionless"])
def test_lfs_pointer_reports_recovery_before_format_dispatch(tmp_path, location):
    from aisimulate_core.sdk import RustForwardPassPerfModel

    target = tmp_path / "capture"
    target.write_bytes(b"version https://git-lfs/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 1234\n")
    path = tmp_path / "capture.parquet"
    if location == "direct":
        target.rename(path)
    else:
        path.symlink_to(target)
        if location == "extensionless":
            path = target
    for load in (FastAFDMoEStageProfile.load, lambda p: RustForwardPassPerfModel.best_available(_canonical_config(p))):
        with pytest.raises(ValueError, match="unresolved git-lfs pointer.*git lfs pull") as error:
            load(path)
        assert str(path.resolve()) in str(error.value)


def test_rejects_int64_parquet_latency(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisimulate_core.sdk import RustForwardPassPerfModel

    payload = _payload()
    payload["entries"][0]["latency_ms"] = 12.0
    profile = FastAFDMoEStageProfile.load(_write(tmp_path, payload))
    path = tmp_path / "stage.parquet"
    profile.write_parquet(path)
    table = pq.read_table(path)
    index = table.schema.get_field_index("latency_ms")
    table = table.set_column(index, "latency_ms", table["latency_ms"].cast(pa.int64()))
    pq.write_table(table, path)

    with pytest.raises(ValueError, match="Cannot access Long as Float"):
        RustForwardPassPerfModel.best_available(_canonical_config(path))
    with pytest.raises(ValueError, match="latency_ms column must be FLOAT or DOUBLE"):
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
