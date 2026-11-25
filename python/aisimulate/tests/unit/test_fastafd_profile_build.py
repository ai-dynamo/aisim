# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from aisimulate.fastafd_profile import FastAFDMoEStageKey, FastAFDMoEStageProfile, build_profile, main

pytestmark = pytest.mark.unit

_REPOSITORY = "https://github.com/example/FastAFD"
_MODEL = {
    "key": "qwen3_235b_fp4",
    "model_path": "Qwen/Qwen3-235B-A22B-FP8",
    "routed_top_k": 8,
    "moe_quant_contract": "w4a8_mxfp4_mxfp8",
}
_STACK = [{"torch": "2.10.0+cu130", "cuda_runtime": "13.0"}]


def _colocated(tokens, *, routing="balanced", stable=True, p50=12.0, correctness=None):
    return {
        "schema": "fastafd.megamoe-colocated-model-benchmark.v2",
        "system_label": "gb200",
        "measurement_boundary": "colocated MoE stage",
        "measurement_boundary_components": ["dispatch", "experts", "return", "combine"],
        "measurement_timed_span": "all_moe_layers_and_microbatches",
        "measurement_microbatch_split": "equal_padded",
        "primary_backend": "mega",
        "model_profile": _MODEL,
        "topology": {"ep_size": 8},
        "workload": {"logical_tokens_per_rank": tokens, "mtp_nextn": 0, "layers": 94, "routing": routing},
        "stage_cuda": {"samples": 40, "p50_ms": p50, "cv_percent": 0.5},
        "stable": stable,
        "correctness_passed": correctness,
        "source": {"commit": "0" * 40, "source_tree_sha256": "c" * 64},
        "provenance_by_rank": _STACK,
    }


def _split(sequences):
    return {
        "schema": "fastafd.moe-m2n-model-benchmark.v2",
        "system_label": "gb200",
        "measurement_boundary": "split AFD MoE stage",
        "measurement_boundary_components": ["dispatch", "experts", "return", "combine"],
        "measurement_timed_span": "all_moe_layers_and_microbatches",
        "measurement_microbatch_split": "equal_padded",
        "moe_backend": "megamoe",
        "model_profile": _MODEL,
        "topology": {"ag_size": 16, "eg_size": 8},
        "workload": {
            "sequences_per_ag_rank": sequences,
            "mtp_nextn": 1,
            "microbatches": 2,
            "layers": 94,
            "routing": "balanced",
        },
        "stage_cuda": {"samples": 40, "p50_ms": 11.4, "cv_percent": 0.6},
        "stable": True,
        "source": {"commit": "0" * 40, "source_tree_sha256": "d" * 64},
        "provenance_by_rank": _STACK,
    }


def _write(directory, name, payload):
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_builds_colocated_profile_and_skips_unqualified_results(tmp_path):
    results = tmp_path / "colocated"
    raw = _write(results, "t48.json", _colocated(48))
    _write(results, "t96.json", _colocated(96, p50=14.0))
    _write(results, "t48_hotset.json", _colocated(48, routing="hotset"))
    _write(results, "t8.json", _colocated(8, stable=False))
    _write(results, "t16.json", _colocated(16, correctness=False))

    profile, summary = build_profile(results, repository=_REPOSITORY)

    assert summary == {"accepted": 2, "skipped_routing": 1, "skipped_unstable": 1, "skipped_correctness": 1}
    assert profile["source"] == {
        "repository": _REPOSITORY,
        "commit": "0" * 40,
        "benchmark": "fastafd.megamoe-colocated-model-benchmark.v2",
        "boundary": "colocated MoE stage",
        "boundary_components": ["dispatch", "experts", "return", "combine"],
        "timed_span": "all_moe_layers_and_microbatches",
        "microbatch_split": "equal_padded",
    }
    entry = profile["entries"][0]
    assert entry["key"] == {
        "model_path": "Qwen/Qwen3-235B-A22B-FP8",
        "system": "gb200",
        "stage": "agg",
        "topology": "ep8",
        "logical_batch_per_source_rank": 48,
        "mtp_nextn": 0,
        "microbatches": 1,
        "moe_layers": 94,
        "routed_topk": 8,
        "moe_precision": "w4a8_mxfp4_mxfp8",
        "moe_backend": "megamoe",
    }
    assert entry["measurement"]["raw_artifact"] == "t48.json"
    assert entry["measurement"]["raw_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert entry["measurement"]["method_version"] == "torch 2.10.0+cu130; cuda 13.0"
    assert entry["validation"] == {"stable": True, "correctness": None}


@pytest.mark.parametrize("stable", ["true", "false", 0, 1, None, [], {}])
@pytest.mark.parametrize("suffix", ["json", "parquet"])
def test_malformed_stability_among_valid_results_prevents_publication(tmp_path, stable, suffix):
    results = tmp_path / "results"
    _write(results, "a-valid.json", _colocated(48))
    _write(results, "b-malformed.json", _colocated(96, stable=stable))
    output = tmp_path / f"profile.{suffix}"
    original = b"existing profile"
    output.write_bytes(original)

    with pytest.raises(ValueError, match="b-malformed.json: stable must be a boolean"):
        main([str(results), "--repository", _REPOSITORY, "--output", str(output)])

    assert output.read_bytes() == original


@pytest.mark.parametrize("suffix", ["json", "parquet"])
def test_builds_split_profile_through_the_command_line(tmp_path, suffix):
    results = tmp_path / "split"
    _write(results, "s64.json", _split(64))
    output = tmp_path / f"profile.{suffix}"

    assert main([str(results), "--repository", _REPOSITORY, "--output", str(output)]) == 0

    profile = FastAFDMoEStageProfile.load(output)
    key = FastAFDMoEStageKey(
        model_path="Qwen/Qwen3-235B-A22B-FP8",
        system="gb200",
        stage="afd",
        topology="16A8F",
        logical_batch_per_source_rank=64,
        mtp_nextn=1,
        microbatches=2,
        moe_layers=94,
        routed_topk=8,
        moe_precision="w4a8_mxfp4_mxfp8",
        moe_backend="megamoe",
    )
    assert profile.require(key).latency_ms == pytest.approx(11.4)
    assert profile.require(key).method_version == "torch 2.10.0+cu130; cuda 13.0"
    assert profile.source.benchmark == "fastafd.moe-m2n-model-benchmark.v2"


def test_rejects_mixed_directories_and_repeated_measurements(tmp_path):
    mixed = tmp_path / "mixed"
    _write(mixed, "a.json", _colocated(48))
    _write(mixed, "b.json", _split(64))
    with pytest.raises(ValueError, match="mixed benchmark"):
        build_profile(mixed, repository=_REPOSITORY)

    repeated = tmp_path / "repeated"
    _write(repeated, "a.json", _colocated(48))
    _write(repeated, "b.json", _colocated(48, p50=12.5))
    with pytest.raises(ValueError, match="repeats the measurement"):
        build_profile(repeated, repository=_REPOSITORY)

    with pytest.raises(ValueError, match="no qualified hotset"):
        build_profile(repeated, repository=_REPOSITORY, routing="hotset")


def test_rejects_duplicate_raw_json_members(tmp_path):
    results = tmp_path / "results"
    raw = _write(results, "a.json", _colocated(48))
    raw.write_text(raw.read_text().replace('"p50_ms": 12.0', '"p50_ms": 12.0, "p50_ms": 1.0'))

    with pytest.raises(ValueError, match="duplicate JSON object key: 'p50_ms'"):
        build_profile(results, repository=_REPOSITORY)


@pytest.mark.parametrize("p50", [-1.0, 0.0, float("nan"), float("inf")])
def test_build_profile_rejects_invalid_latency_directly(tmp_path, p50):
    _write(tmp_path, "a.json", _colocated(48, p50=p50))

    with pytest.raises(ValueError, match="latency_ms must be"):
        build_profile(tmp_path, repository=_REPOSITORY)


def test_build_profile_accepts_positive_latency_directly(tmp_path):
    _write(tmp_path, "a.json", _colocated(48, p50=12.0))

    profile, summary = build_profile(tmp_path, repository=_REPOSITORY)

    assert profile["entries"][0]["latency_ms"] == 12.0
    assert summary == {"accepted": 1}


@pytest.mark.parametrize("microbatches", [0, False, None, ""])
@pytest.mark.parametrize("suffix", ["json", "parquet"])
def test_rejects_explicit_invalid_colocated_microbatches_before_publication(tmp_path, microbatches, suffix):
    results = tmp_path / "results"
    payload = _colocated(48)
    payload["workload"]["microbatches"] = microbatches
    _write(results, "a.json", payload)
    output = tmp_path / f"profile.{suffix}"

    with pytest.raises(ValueError, match="microbatches must be an integer >= 1"):
        main([str(results), "--repository", _REPOSITORY, "--output", str(output)])

    assert not output.exists()


@pytest.mark.parametrize("suffix", ["json", "parquet"])
def test_omitted_colocated_microbatches_default_to_one(tmp_path, suffix):
    results = tmp_path / "results"
    _write(results, "a.json", _colocated(48))
    output = tmp_path / f"profile.{suffix}"

    assert main([str(results), "--repository", _REPOSITORY, "--output", str(output)]) == 0

    assert FastAFDMoEStageProfile.load(output).entries[0].key.microbatches == 1


@pytest.mark.parametrize("suffix", ["json", "parquet"])
def test_invalid_timing_preserves_existing_profile(tmp_path, suffix):
    results = tmp_path / "results"
    _write(results, "a.json", _colocated(48))
    output = tmp_path / f"profile.{suffix}"
    args = [str(results), "--repository", _REPOSITORY, "--output", str(output)]
    assert main(args) == 0
    original = output.read_bytes()
    _write(results, "a.json", _colocated(48, p50=-1.0))

    with pytest.raises(ValueError, match="latency_ms must be > 0"):
        main(args)

    assert output.read_bytes() == original
    assert FastAFDMoEStageProfile.load(output).entries[0].latency_ms == 12.0
    assert sorted(path.name for path in tmp_path.iterdir()) == [output.name, "results"]


@pytest.mark.parametrize("suffix", ["json", "parquet"])
def test_repeated_build_excludes_only_designated_output(tmp_path, suffix):
    _write(tmp_path, "capture.json", _split(64))
    output = tmp_path / f"profile.{suffix}"
    args = [str(tmp_path), "--repository", _REPOSITORY, "--output", str(output)]
    assert main(args) == 0
    original = output.read_bytes()
    assert main(args) == 0
    assert output.read_bytes() == original
    assert FastAFDMoEStageProfile.load(output).entries[0].latency_ms == 11.4

    # Arbitrary unsupported inputs must still fail; only --output is excluded.
    _write(tmp_path, "unsupported.json", {"schema": "unsupported"})
    with pytest.raises(ValueError, match="unsupported benchmark schema"):
        main(args)
    assert output.read_bytes() == original


def test_output_cannot_replace_unrelated_parquet_in_results(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    _write(tmp_path, "capture.json", _split(64))
    output = tmp_path / "summary.parquet"
    pq.write_table(pa.table({"unrelated": [12.0]}), output)
    original = output.read_bytes()

    with pytest.raises(ValueError, match="summary.parquet: output must be an existing generated profile"):
        main([str(tmp_path), "--repository", _REPOSITORY, "--output", str(output)])

    assert output.read_bytes() == original


@pytest.mark.parametrize("correctness", ["false", "true", 0, 1, [], {}])
@pytest.mark.parametrize("routing,stable", [("hotset", True), ("balanced", False), ("balanced", True)])
@pytest.mark.parametrize("suffix", ["json", "parquet"])
def test_malformed_correctness_prevents_publication_before_filtering(tmp_path, correctness, routing, stable, suffix):
    results = tmp_path / "results"
    _write(results, "a-valid.json", _colocated(48))
    _write(results, "b-malformed.json", _colocated(96, routing=routing, stable=stable, correctness=correctness))
    output = tmp_path / f"profile.{suffix}"
    output.write_bytes(b"previous output")

    with pytest.raises(ValueError, match="b-malformed.json: correctness_passed must be a boolean or null"):
        main([str(results), "--repository", _REPOSITORY, "--output", str(output)])

    assert output.read_bytes() == b"previous output"


@pytest.mark.parametrize("field", ["torch", "cuda_runtime"])
@pytest.mark.parametrize("version", [None, "", "  ", 13, False])
@pytest.mark.parametrize("suffix", ["json", "parquet"])
def test_invalid_runtime_provenance_prevents_publication(tmp_path, field, version, suffix):
    results = tmp_path / "results"
    payload = _split(64)
    payload["provenance_by_rank"] = [{**_STACK[0], field: version}]
    _write(results, "capture.json", payload)
    output = tmp_path / f"profile.{suffix}"
    output.write_bytes(b"previous output")

    with pytest.raises(ValueError, match=f"capture.json: {field} must be a non-empty string"):
        main([str(results), "--repository", _REPOSITORY, "--output", str(output)])

    assert output.read_bytes() == b"previous output"


def test_output_cannot_replace_a_raw_capture(tmp_path):
    raw = _write(tmp_path, "a.json", _split(64))
    _write(tmp_path, "b.json", _split(128))
    original = raw.read_bytes()
    with pytest.raises(ValueError, match="output.*generated profile"):
        main([str(tmp_path), "--repository", _REPOSITORY, "--output", str(raw)])
    assert raw.read_bytes() == original


def test_symlink_output_is_rejected_consistently(tmp_path):
    _write(tmp_path, "capture.json", _split(64))
    target = tmp_path / "profile.json"
    args = [str(tmp_path), "--repository", _REPOSITORY, "--output", str(target)]
    assert main(args) == 0
    original = target.read_bytes()
    output = tmp_path / "latest.json"
    output.symlink_to(target.name)
    for _ in range(2):
        with pytest.raises(ValueError, match="output.*symlink"):
            main([str(tmp_path), "--repository", _REPOSITORY, "--output", str(output)])
        assert output.is_symlink()
        assert target.read_bytes() == original


@pytest.mark.parametrize("routing", [None, [], ["balanced"], {}, True, 1, "", "  "])
@pytest.mark.parametrize("suffix", ["json", "parquet"])
def test_malformed_routing_prevents_publication(tmp_path, routing, suffix):
    results = tmp_path / "results"
    _write(results, "a-valid.json", _colocated(48))
    _write(results, "b-malformed.json", _colocated(96, routing=routing))
    output = tmp_path / f"profile.{suffix}"
    original = b"previous profile"
    output.write_bytes(original)
    with pytest.raises(ValueError, match="b-malformed.json: routing must be a non-empty string"):
        main([str(results), "--repository", _REPOSITORY, "--output", str(output)])
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    "components",
    [None, ["dispatch", "experts", "return"], ["router", "dispatch", "experts", "return", "combine"]],
)
def test_builder_rejects_unverified_or_incompatible_boundary(tmp_path, components):
    payload = _split(64)
    if components is None:
        del payload["measurement_boundary_components"]
    else:
        payload["measurement_boundary_components"] = components
    _write(tmp_path, "capture.json", payload)
    output = tmp_path / "profile.json"
    with pytest.raises(ValueError, match="boundary_components"):
        main([str(tmp_path), "--repository", _REPOSITORY, "--output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("span", [None, "one_microbatch", "one_moe_layer", "", 1])
@pytest.mark.parametrize("suffix", ["json", "parquet"])
@pytest.mark.parametrize("field", ["timed_span", "microbatch_split"])
def test_builder_rejects_unverified_capture_contract_without_publication(tmp_path, span, suffix, field):
    payload = _split(64)
    capture = _write(tmp_path, "capture.json", payload)
    output = tmp_path / f"profile.{suffix}"
    args = [str(tmp_path), "--repository", _REPOSITORY, "--output", str(output)]
    assert main(args) == 0
    original = output.read_bytes()
    if span is None:
        del payload[f"measurement_{field}"]
    else:
        payload[f"measurement_{field}"] = span
    capture.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        main(args)
    assert output.read_bytes() == original
