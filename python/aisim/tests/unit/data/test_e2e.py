# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""aisim-e2e/v1: InferenceX and developer fixtures share one evaluator loader."""

import gzip

import pytest

from aisim_core.data.e2e import (
    LOG_CAP_BYTES,
    E2eManifestBody,
    MetricRow,
    ProvenanceFile,
    Setting,
    SnapshotAcquisition,
    SnapshotMember,
    aiperf_to_configuration,
    inferencex_to_configurations,
    make_metric,
    public_release_problems,
    read_provenance,
    read_snapshot,
    redact,
    store_provenance,
    throughput_per_device,
    to_canonical,
    write_snapshot,
)

pytestmark = pytest.mark.unit


def _infx_record(**over):
    # Same shape as the InferenceX cache fixture in tests/test_e2e_hf_measurements.py.
    rec = dict(
        model="test-model",
        hardware="h100_sxm",
        framework="vllm",
        precision="fp8",
        spec_method="none",
        disagg=False,
        is_multinode=False,
        prefill_tp=1,
        prefill_ep=1,
        prefill_dp_attention=False,
        prefill_num_workers=0,
        decode_tp=8,
        decode_ep=1,
        decode_dp_attention=False,
        decode_num_workers=1,
        num_prefill_gpu=0,
        num_decode_gpu=8,
        id=123,
        date="2026-10-17",
        isl=1024,
        osl=128,
        conc=32,
        metrics={
            "mean_ttft": 0.1,
            "median_ttft": 0.09,
            "p99_ttft": 0.3,
            "mean_tpot": 0.01,
            "median_tpot": 0.0,
            "mean_e2el": 1.5,
            "tput_per_gpu": 1200.0,
            "output_tput_per_gpu": 150.0,
            "mean_intvty": 100.0,
            "weird_metric": 7,
        },
        image="test-image",
        benchmark_type="single_turn",
        run_url="https://github.com/SemiAnalysisAI/InferenceX/actions/runs/987/attempts/2",
    )
    rec.update(over)
    return rec


def _dev_body():
    return E2eManifestBody(
        origin="self-collected",
        model_id="test-model",
        deployment="aggregated",
        hardware="h100_sxm",
        framework="vllm",
        framework_version="0.11.0",
        parallelism={"aggregated": {"tp": 8, "ep": 1}},
        parallelism_complete=True,
        precision={"weights": "fp8"},
        settings={"max_num_seqs": Setting(value=256, authority="observed")},
        source_kind="developer",
    )


def _aiperf_summary(**over):
    s = {
        "is_complete": True,
        "was_cancelled": False,
        "request_count": {"avg": 2},
        "error_request_count": {"avg": 0},
        "time_to_first_token": {"unit": "ms", "avg": 100.0, "p50": 90.0, "p90": 150.0, "p99": 300.0},
        "inter_token_latency": {"unit": "ms", "avg": 10.0, "p50": 9.0, "p90": 14.0},
        "request_latency": {"unit": "seconds", "avg": 1.5, "p50": 1.4, "p90": 2.0},
        "output_token_throughput": {"unit": "tokens/second", "avg": 9600.0},
    }
    s.update(over)
    return s


def _aiperf_records():
    return [
        {
            "request_id": i,
            "error": None,
            "metrics": {
                "input_sequence_length": {"value": 1024, "unit": "tokens"},
                "output_sequence_length": {"value": 128, "unit": "tokens"},
                "time_to_first_token": {"value": 0.1, "unit": "seconds"},
                "inter_token_latency": {"value": 10, "unit": "ms"},
            },
        }
        for i in range(2)
    ]


def _dev_config(**kw):
    return aiperf_to_configuration(
        _aiperf_summary(), _aiperf_records(), _dev_body(), observation_id="dev:1", device_count=8, **kw
    )


# ---------------------------------------------------------------- metrics


def test_missing_is_never_zero():
    assert make_metric("o", "ttft", "mean", None, "s").status == "missing"
    zero = make_metric("o", "ttft", "mean", 0, "s")
    assert zero.status == "valid" and zero.value == 0.0
    for bad in (float("nan"), float("inf"), -1.0, True, "1"):
        row = make_metric("o", "ttft", "mean", bad, "s")
        assert row.status == "invalid" and row.value is None
    with pytest.raises(ValueError):
        MetricRow(observation_id="o", metric="ttft", statistic="mean", unit="ms", status="missing", value=0.0)
    with pytest.raises(ValueError):
        MetricRow(observation_id="o", metric="ttft", statistic="mean", unit="ms", status="valid")


def test_unit_conversions():
    assert to_canonical(0.25, "s", "ms") == 250.0
    assert to_canonical(250, "milliseconds", "ms") == 250
    assert to_canonical(500, "us", "ms") == pytest.approx(0.5)
    assert to_canonical(3, "tok/s", "tokens/s") == 3
    with pytest.raises(ValueError):
        to_canonical(1, "furlongs", "ms")
    with pytest.raises(ValueError):
        to_canonical(1, "tokens/s", "ms")
    row = make_metric("o", "ttft", "p90", 0.2, "s")
    assert (row.value, row.unit, row.source_value, row.source_unit) == (200.0, "ms", 0.2, "s")


def test_per_device_throughput_is_explicit():
    per = make_metric("o", "output_throughput", "value", 150.0, "tokens/s", normalization="per_device", device_count=8)
    total = make_metric("o", "output_throughput", "value", 1200.0, "tokens/s", normalization="total", device_count=8)
    assert throughput_per_device(per) == 150.0 and throughput_per_device(total) == 150.0
    unknown = total.model_copy(update={"device_count": None})
    with pytest.raises(ValueError, match="device_count"):
        throughput_per_device(unknown)
    with pytest.raises(ValueError):
        throughput_per_device(make_metric("o", "ttft", "mean", 1, "ms"))


# ---------------------------------------------------------------- round trip through one loader


def test_inferencex_disaggregated_groups_and_identity():
    configs = inferencex_to_configurations(
        [
            _infx_record(),
            _infx_record(id=124, conc=64),
            _infx_record(id=125, disagg=True, prefill_tp=4, prefill_ep=2, prefill_num_workers=1, num_prefill_gpu=4),
        ]
    )
    assert sorted(len(c.observations) for c in configs) == [1, 2]
    dis = next(c for c in configs if c.body.deployment == "disaggregated")
    assert dis.body.parallelism == {"decode": {"tp": 8}, "prefill": {"tp": 4, "ep": 2}}
    assert dis.observations[0].device_count == 12
    assert dis.body.configuration_id() != next(c for c in configs if c is not dis).body.configuration_id()


def test_open_settings_do_not_change_identity():
    base = _dev_body()
    more = base.model_copy(update={"settings": {**base.settings, "new_flag": Setting(value=True, authority="unknown")}})
    assert base.configuration_id() == more.configuration_id()
    other = base.model_copy(update={"parallelism": {"aggregated": {"tp": 4}}})
    assert base.configuration_id() != other.configuration_id()


# ---------------------------------------------------------------- provenance


def test_provenance_cap_redaction_and_hash_link(tmp_path):
    big = b"A" * 100 + b"x" * (LOG_CAP_BYTES + 10) + b"Z" * 100
    artifact, info = store_provenance(tmp_path, "d", ProvenanceFile("log", big, "o"))
    assert info.truncated and info.tier == 3 and info.full_bytes == len(big)
    assert info.full_sha256 != artifact.sha256
    kept = read_provenance(tmp_path, artifact)
    assert kept.startswith(b"AAAA") and kept.endswith(b"ZZZZ") and len(kept) < len(big)
    small, small_info = store_provenance(tmp_path, "d", ProvenanceFile("log", b"hello", "o", name="s"))
    assert not small_info.truncated and gzip.decompress((tmp_path / small.path).read_bytes()) == b"hello"
    with pytest.raises(ValueError, match="never truncated"):
        store_provenance(tmp_path, "d", ProvenanceFile("raw-output", b"x" * 11, "o", cap_bytes=10))
    assert b"hf_" not in redact(b"token=hf_" + b"a" * 30)
    body = _dev_body().model_copy(update={"provenance_artifacts": [info]})
    assert public_release_problems(body) == [f"unredacted tier-3 artifact: {info.artifact_id}"]


# ---------------------------------------------------------------- snapshot


def test_snapshot_round_trip_and_tamper(tmp_path):
    members = [
        SnapshotMember(
            configuration_id="cfg-" + "a" * 64, observation_id="o2", record_sha256="b" * 64, manifest_path="m"
        ),
        SnapshotMember(
            configuration_id="cfg-" + "a" * 64, observation_id="o1", record_sha256="c" * 64, manifest_path="m"
        ),
    ]
    snap = write_snapshot(
        tmp_path,
        "snap.json",
        members,
        source_scope={"kind": "api-current-model-view"},
        selection_policy="p",
        completeness="complete",
        acquisition=[SnapshotAcquisition(observation_id="o1", source_uri="https://x")],
    )
    got, rows = read_snapshot(tmp_path, "snap.json")
    assert got == snap and [r.observation_id for r in rows] == ["o1", "o2"]
    data = (tmp_path / "snap.json").read_text().replace('"complete"', '"partial"')
    (tmp_path / "snap.json").write_text(data)
    assert read_snapshot(tmp_path, "snap.json")[0].completeness == "partial"
    (tmp_path / "snap.json").write_text(data.replace(snap.snapshot_id, "0" * 64))
    with pytest.raises(ValueError, match="snapshot ID"):
        read_snapshot(tmp_path, "snap.json")
