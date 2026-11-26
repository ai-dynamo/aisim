# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native collection accepts the rank artifacts of the GLM-5.3-Flash vLLM producer.

The artifacts are built in the shape of ``Glm53FlashPrefixSeedScheduler._glm_enrich``: a native rank artifact plus the
producer's execution, input, serving and evidence blocks and a repetition-evidence sidecar next to it.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest
from collector import glm53flash_attention_tokens as seeded
from collector.fpm_forward import database, native_artifact
from collector.fpm_forward.capabilities import GENERIC_RUNTIME_TEMPLATE_ID
from collector.fpm_forward.native_artifact import (
    NativeCollection,
    _expected_scheduled,
    _validate_execution_provenance,
    select_native_measurements,
    validate_native_collection,
)
from collector.fpm_forward.planner import FPMCell, GlmProducerDigests
from collector.fpm_forward.runtime_memory import cell_from_dict

from aisim.fpm_contract import fpm_benchmark_result_name
from aisim_core.sdk.fpm_identity import EXECUTION_COLUMNS, execution_identity
from aisim_core.sdk.glm53flash import MODEL_REVISIONS
from aisim_core.sdk.utils import get_model_config_from_model_path

from .test_fpm_forward import _write_v41_token_streams
from .test_fpm_glm53flash_prefix_seed import load_producer
from .test_fpm_runner import _cell, _native_payload, _write_provenance

pytestmark = pytest.mark.unit

GLM_MODEL = "zai-org/GLM-5.3-Flash"
PLAN_SHA256 = "plan-sha"
ATTEMPT_ID = "a1"
EVIDENCE_FILE = "benchmark.repetitions.jsonl"
# The producer files that the collector ships and the launch mounts. The producer records their digests.
PRODUCER_DIR = Path(native_artifact.__file__).parent / "runtime" / "glm53flash"

PREFILL_COLD = {
    "batch_size": 1,
    "total_prefill_tokens": 64,
    "total_kv_read_tokens": 0,
    "sample_reasons": ["explicit"],
}
PREFILL_CACHED = {
    "batch_size": 1,
    "total_prefill_tokens": 64,
    "total_kv_read_tokens": 2176,
    "sample_reasons": ["explicit", "prefill_real_seed"],
}
DECODE_POINTS = [
    {
        "batch_size": 1,
        "total_prefill_tokens": 0,
        "total_kv_read_tokens": 4096,
        "sample_reasons": ["explicit", "kvwarm_real_kv"],
    },
    {
        "batch_size": 2,
        "total_prefill_tokens": 0,
        "total_kv_read_tokens": 8192,
        "sample_reasons": ["explicit", "kvwarm_real_kv"],
    },
]


@functools.cache
def glm_identity() -> tuple[str, ...]:
    """The execution identity that the producer computes from the loaded checkpoint configuration."""
    raw_config = get_model_config_from_model_path(GLM_MODEL)["raw_config"]
    return execution_identity(raw_config, backend="vllm", input_modality="text")


def glm_cell(phase: str, *, dp: int = 1) -> FPMCell:
    """A pure-TP cell that carries the real GLM-5.3-Flash execution identity."""
    return replace(
        _cell(phase=phase, dp=dp, strategy="pure_tp"),
        execution_identity=glm_identity(),
        glm_producer_digests=GlmProducerDigests.from_runtime(PRODUCER_DIR),
    )


def glm_native_payload(phase: str, *, points: list[dict], rank: int = 0, dp: int = 1) -> dict:
    """The rank artifact that the producer writes for ``points`` of one phase.

    A point gives ``batch_size``, ``total_prefill_tokens``, ``total_kv_read_tokens`` and ``sample_reasons``. Each
    row holds one FPM that schedules exactly what its point asks for. The evidence manifest and the per-row digests
    stay empty until ``write_glm_rank`` writes the sidecar that they describe.
    """
    payload = _native_payload(phase=phase, rank=rank, dp=dp)
    rows, groups = [], []
    for benchmark_id, spec in enumerate(points, start=1):
        point = {"point_type": phase, "benchmark_id": benchmark_id, **copy.deepcopy(spec)}
        fpms = [
            {
                "counter_id": benchmark_id,
                "dp_rank": dp_rank,
                "wall_time": 0.01 * benchmark_id + dp_rank / 1000,
                "scheduled_requests": _expected_scheduled(point),
            }
            for dp_rank in range(dp)
        ]
        if phase == "decode":
            regime = "real_kv"
        else:
            regime = "real_prefix" if point["total_kv_read_tokens"] > 0 else "not_applicable"
        rows.append(
            {"point": point, "fpms": [fpms[rank]], "kv_seed_regime": regime, "repetition_evidence_sha256": None}
        )
        groups.append(
            {
                "benchmark_id": benchmark_id,
                "point": point,
                "expected_dp_ranks": list(range(dp)),
                "complete": True,
                "wall_time": max(fpm["wall_time"] for fpm in fpms),
                "rank_results": [{"dp_rank": fpm["dp_rank"], "fpms": [fpm]} for fpm in fpms],
                # The producer stamps every group with the decode regime, whatever its phase.
                "kv_seed_regime": "real_kv",
            }
        )
    measured = sum(group["wall_time"] for group in groups)
    context_policy = {
        "measured_context_limit": 131072,
        "runtime_context_length": 131079,
        "native_admission_headroom": 7,
    }
    payload.update(
        coverage={"expected_points": len(rows), "completed_points": len(rows), "skipped_points": 0},
        results=rows,
        iteration_groups=groups,
        timing={"benchmark_elapsed_seconds": measured + 1.0, "measured_iteration_seconds": measured},
        context_policy=context_policy,
        execution_identity=dict(zip(EXECUTION_COLUMNS, glm_identity(), strict=True)),
        execution_mode="native_graph_policy",
        observation_purpose="fpm",
        ops_instrumented=False,
        timing_boundary="dynamo_vllm_instrumented_scheduler_fpm_wall_time",
        input_provenance={
            "source": "seeded_tokens",
            "generator": seeded.GENERATOR,
            "generator_version": seeded.GENERATOR_VERSION,
            "seed": seeded.SEED,
            "count": 131072 + 32 * seeded.REQUEST_STRIDE,
            "vocab_low": seeded.VOCAB_LOW,
            "vocab_high_exclusive": seeded.VOCAB_HIGH,
            "tokenizer_json_sha256": seeded.TOKENIZER_JSON_SHA256,
            "tokenizer_revision": MODEL_REVISIONS[GLM_MODEL],
            "pinned_model": GLM_MODEL,
            "pinned_revision": MODEL_REVISIONS[GLM_MODEL],
            "checkpoint_dir": f"/hf-hub/models--zai-org--GLM-5.3-Flash/snapshots/{MODEL_REVISIONS[GLM_MODEL]}",
            # The digest and the distinct-token count of the real stream of seed 53, generator version 2.
            "token_ids_sha256": "b72027be2cc23b86e212485dc13b24b62969b583eae6ee40ac0124429f7cc15e",
            "unique_token_count": 126303,
            "sampling": "prompt = concatenated [offset, length] rotations of the seeded stream (see prompt specs)",
            "chunk_start_alignment": 4,
            "decode_contexts_sha256": None,
            "context_policy": context_policy,
            "repetition_evidence_manifest": {
                "schema_version": 1,
                "file": Path(fpm_benchmark_result_name("benchmark.json", rank)).with_suffix(".repetitions.jsonl").name,
                "sha256": None,
                "records": 0,
            },
        },
        serving_config={
            "enable_prefix_caching": True,
            "mamba_cache_mode": "align",
            "prefix_match_unit": None,
            "hash_block_size": 2176,
            "block_size": 2176,
            "need_mamba_block_aligned_split": True,
            "cache_dtype": "fp8_e4m3",
            "async_scheduling": False,
            "mamba_prefill_checkpoint": {"blocks": True, "alignment": 16},
        },
        kvwarm={
            "enabled": True,
            "warm_eligible": True,
            "skip_reason": None,
            "method": "native_prefix_cache_real_seed",
            "state_protocol": "glm53flash_prefix_cache_real_seed_v1",
            "max_batch": 32,
            "max_context": 131072,
        },
        producer={
            "instrumentation_revision": "54960177085413259859c88bd34ed0734d4c2ea9",
            "runtime_scheduler_revision": "9abd2ef0d908c4bb323e1b8956ea76a5772a1d41",
            "runtime_distribution": "ai-dynamo 1.7.0.dev20261109",
            "real_seed_port_source": "ai-dynamo/dynamo@99dae1f53e5a0534c274223ff9daa97f0b1fc2ea "
            "instrumented_scheduler._bench_realseed_*",
            "queued_metrics_port_source": "ai-dynamo/dynamo@9abd2ef0d908c4bb323e1b8956ea76a5772a1d41 "
            "instrumented_scheduler._compute_queued",
            "vllm_package_version": "0.33.0",
            "reviewed_scheduler_api_revision": "4ba787ec9959e566d3b587a796dff07673324da5",
            "runtime_source_manifest_sha256": hashlib.sha256(
                (PRODUCER_DIR / "runtime-source-sha256.json").read_bytes()
            ).hexdigest(),
            "overlay_sha256": hashlib.sha256(
                (PRODUCER_DIR / "glm53flash_prefix_scheduler.py").read_bytes()
            ).hexdigest(),
            "context_policy_version": 1,
            "warmup_repeats": 5,
            "measurement_repeats": 10,
        },
    )
    return payload


def write_glm_rank(directory: Path, payload: dict) -> Path:
    """Write ``payload`` as ``benchmark.json`` and its sidecar of repetition evidence next to it.

    The sidecar holds one line per result row. As in the producer, a row's digest is the digest of its line without
    the line break, and the manifest holds the digest of the whole file. ``payload`` is updated in place with these
    values and the path of the artifact is returned. Rank 0 writes ``benchmark.json``. Another rank writes
    ``benchmark_dp<rank>.json`` with a sidecar of the same stem, as the engine names them.
    """
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in payload["results"]:
        point = row["point"]
        fpm = row["fpms"][0]
        raw = fpm.get("benchmark_measurement", {}).get("raw_fpms")
        repetitions = []
        forward = -1
        for index in range(15):
            observed = raw[index - 5] if isinstance(raw, list) and 5 <= index < 5 + len(raw) else None
            steps = 2 if point["point_type"] == "decode" else 1
            observed = observed if isinstance(observed, dict) else {}
            sample = observed.get("benchmark_sample")
            sample = sample if isinstance(sample, dict) else {}
            forward = sample.get("forward_index", forward + steps)
            measured = {
                "counter_id": point["benchmark_id"],
                "dp_rank": payload["dp"]["rank"],
                "scheduled_requests": _expected_scheduled(point),
                "wall_time": fpm["wall_time"],
                **observed,
                "benchmark_sample": {**sample, "forward_index": forward, "sample_index": steps - 1},
            }
            fpms = [measured]
            if steps == 2:
                first = copy.deepcopy(measured)
                first["scheduled_requests"] = {
                    **_expected_scheduled(point),
                    "sum_decode_kv_tokens": point["total_kv_read_tokens"] - point["batch_size"],
                }
                first["benchmark_sample"].update(sample_index=0, forward_index=forward - 1)
                fpms.insert(0, first)
            repetitions.append(
                {"attempt": index, "repetition": index, "role": "warmup" if index < 5 else "measurement", "fpms": fpms}
            )
        record = {"benchmark_id": point["benchmark_id"], "point_type": point["point_type"], "repetitions": repetitions}
        lines.append(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
    for row, line in zip(payload["results"], lines, strict=True):
        row["repetition_evidence_sha256"] = hashlib.sha256(line).hexdigest()
    sidecar = b"".join(line + b"\n" for line in lines)
    manifest = payload["input_provenance"]["repetition_evidence_manifest"]
    manifest.update(sha256=hashlib.sha256(sidecar).hexdigest(), records=len(lines))
    (directory / manifest["file"]).write_bytes(sidecar)
    artifact = directory / fpm_benchmark_result_name("benchmark.json", payload["dp"]["rank"])
    artifact.write_text(json.dumps(payload))
    return artifact


def _glm_producer_shape(tmp_path, breakage=None):
    """Rewrite a GLM-5.3-Flash rank artifact into the shape that the producer writes on Dynamo 1.7.

    The fake of the native scheduler runs the producer once for each phase. Its measurement protocol and the
    ``benchmark_sample`` of its first measured repetition are the templates. Each point then gets the record of the
    native save as the producer rewrites it: ten raw samples of the point's own FPM, each numbered by its position and
    with its own timing interval, and the producer's upper median of samples 1 to 9 as the retained wall time. With no
    global warm-up, Dynamo records an empty warm-up list. ``breakage`` moves the timing interval of one raw sample
    (``interval``) or moves the retained wall time to the slowest repetition (``median``), in the first point of each
    artifact.
    """
    from collector.fpm_forward.measurement_evidence import _content_key

    from .test_fpm_glm53flash_prefix_seed import Engine, Point, _run_prefill_point, make_scheduler
    from .test_fpm_glm53flash_prefix_seed import run as run_producer

    templates, estimates = {}, {}
    with pytest.MonkeyPatch.context() as patch:
        module = load_producer(patch)
        for phase, point in (
            ("prefill", Point("prefill", 1, 1, 32, 120, rows=[[32, 120]])),
            ("decode", Point("decode", 1, 3, 0, 3 * 201)),
        ):
            (tmp_path / f"producer-{phase}").mkdir(parents=True)
            scheduler = make_scheduler(module, tmp_path / f"producer-{phase}", [point])
            if phase == "prefill":
                _run_prefill_point(scheduler, measured_walls=[0.4] * 10)
            else:
                run_producer(scheduler, Engine(scheduler), "decode")
            ((produced,),) = [result.fpms for result in scheduler._bench_results]
            measurement = produced["benchmark_measurement"]
            assert (measurement["expected_internal_samples"], measurement["estimate"]["method"]) == (
                10,
                "adjacent_upper_median",
            )
            templates[phase] = measurement["raw_fpms"][0]["benchmark_sample"]
            estimates[phase] = measurement["estimate"]
        protocol = scheduler._bench_measurement_protocol()

    def shape(phase, payload):
        payload["measurement_protocol"] = copy.deepcopy(protocol)
        payload["warmup_evidence"] = {"status": "recorded", "records": []}
        forward = 0
        for position, (row, group) in enumerate(zip(payload["results"], payload["iteration_groups"], strict=True)):
            fpm = group["rank_results"][0]["fpms"][0]
            assert row["fpms"][0] is fpm
            walls = [fpm["wall_time"] * (1 + 0.01 * (index - 4)) for index in range(10)]
            tokens = row["point"]["total_prefill_tokens"] or row["point"]["batch_size"]
            samples = []
            forward += 11 if phase == "decode" else 5
            for index, wall in enumerate(walls):
                sample = copy.deepcopy(templates[phase])
                start = 100.0 * position + index
                # The producer numbers each raw sample by its position, as Dynamo's merge requires.
                sample["sample_index"] = index
                sample["forward_index"] = forward
                sample["timing"].update(start_monotonic=start, end_monotonic=start + wall)
                sample["cudagraph"].update(num_unpadded_tokens=tokens, num_padded_tokens=tokens, num_paddings=0)
                # A decode repetition records two FPMs, and the second is its steady sample.
                forward += 2 if phase == "decode" else 1
                raw = {**copy.deepcopy(fpm), "wall_time": wall, "benchmark_sample": sample}
                if phase == "decode":
                    raw["kvwarm_steady_sample"] = True
                samples.append(raw)
            if phase == "decode":
                fpm["kvwarm_steady_sample"] = True
            # The producer's reduction: the upper median of the samples that its estimate names.
            chosen = sorted(walls[index] for index in estimates[phase]["raw_sample_indices"])
            fpm["wall_time"] = group["wall_time"] = chosen[len(chosen) // 2]
            fpm["benchmark_measurement"] = {
                "schema_version": 1,
                "point_key": _content_key(row["point"]),
                "dp_rank": 0,
                # The producer injects no prompt hashes.
                "prompts": {"status": "unavailable", "sha256": None, "requests": []},
                "preparation": {
                    "grid_digest": payload["grid_digest"],
                    "completed_points_before": position,
                    "kv_seed_regime": row["kv_seed_regime"],
                    "warmup_records_before": 0,
                },
                "expected_internal_samples": 10,
                "raw_fpms": samples,
                "estimate": copy.deepcopy(estimates[phase]),
            }
        first = payload["iteration_groups"][0]["rank_results"][0]["fpms"][0]
        if breakage == "interval":
            first["benchmark_measurement"]["raw_fpms"][3]["benchmark_sample"]["timing"]["start_monotonic"] -= 0.1
        elif breakage == "median":
            # The slowest repetition instead of the median.
            first["wall_time"] = max(sample["wall_time"] for sample in first["benchmark_measurement"]["raw_fpms"])
            payload["iteration_groups"][0]["wall_time"] = first["wall_time"]
        groups = payload["iteration_groups"]
        payload["timing"]["measured_iteration_seconds"] = sum(group["wall_time"] for group in groups)

    return shape


def _pod(tmp_path: Path, cell: FPMCell, *, backend_version: str = "0.33.0") -> Path:
    """The pod folder of a launch whose Collector provenance names the vLLM runtime ``backend_version``."""
    pod = tmp_path / "raw" / "node0000"
    pod.mkdir(parents=True)
    provenance = pod / "collector-provenance.json"
    _write_provenance(provenance, cell_id=cell.cell_id, plan_sha256=PLAN_SHA256, attempt_id=ATTEMPT_ID)
    record = json.loads(provenance.read_text())
    record["runtime"]["backend_version"] = backend_version
    provenance.write_text(json.dumps(record))
    return pod


def _validate(tmp_path: Path, cell: FPMCell):
    return validate_native_collection(
        cell, tmp_path / "raw", expected_plan_sha256=PLAN_SHA256, expected_attempt_id=ATTEMPT_ID
    )


def _glm_plan() -> SimpleNamespace:
    """The parts of a collection plan that publication reads, for the GLM-5.3-Flash checkpoint."""
    return SimpleNamespace(
        sha256=PLAN_SHA256,
        aic_revision="test-revision",
        model_path=GLM_MODEL,
        system="gb300",
        backend="vllm",
        options=SimpleNamespace(warmup_iterations=0),
        capability=SimpleNamespace(
            support_level="native_runtime",
            template_id=GENERIC_RUNTIME_TEMPLATE_ID,
            template_version=1,
            aic_database_version="0.33.0",
        ),
    )


def _restamp(measurement, stamps: list[str], regime: str):
    """The measurement of a point that carries ``stamps`` and whose row reports ``regime``."""
    return replace(
        measurement, point={**measurement.point, "sample_reasons": ["explicit", *stamps]}, kv_seed_regime=regime
    )


def _publish(monkeypatch, tmp_path: Path, cell: FPMCell, collection: NativeCollection, plan=None) -> list[dict]:
    """Publish ``collection`` as if the native validator had returned it.

    The validator refuses artifacts that the producer never writes, such as a GLM decode artifact with a warm-up
    skip. A test starts from a collection that the validator accepted and changes it, to reach the publication rules.
    The tests of this helper write plain artifacts without measurement records, which the measurement record reader
    refuses. The reader also refuses a collection whose points, row regimes or wall times differ from the artifact, as
    a restamped point with a new stamp or regime does. So it is replaced too, and other tests of this file publish
    through the real reader.
    """
    monkeypatch.setattr(database, "validate_native_collection", lambda *_args, **_kwargs: collection)
    monkeypatch.setattr(database, "extract_measurement_evidence", lambda *_args, **_kwargs: {})
    return database.aggregate_cell(plan or _glm_plan(), cell, tmp_path, expected_attempt_id=ATTEMPT_ID)


def test_glm_producer_artifact_validates_and_selects_its_points(tmp_path):
    cell = glm_cell("prefill")
    write_glm_rank(_pod(tmp_path, cell), glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED]))
    collection = _validate(tmp_path, cell)
    assert [m.kv_seed_regime for m in collection.points] == ["not_applicable", "real_prefix"]
    assert collection.input_provenance["source"] == "seeded_tokens"
    selected = select_native_measurements(collection, cell_id=cell.cell_id)
    assert [m.point["total_kv_read_tokens"] for m in selected] == [0, 2176]


def test_glm_producer_decode_artifact_validates(tmp_path):
    cell = glm_cell("decode")
    write_glm_rank(_pod(tmp_path, cell), glm_native_payload("decode", points=DECODE_POINTS))
    collection = _validate(tmp_path, cell)
    assert [m.point["batch_size"] for m in collection.points] == [1, 2]
    assert {m.kv_seed_regime for m in collection.points} == {"real_kv"}
    assert collection.kvwarm_meta["method"] == "native_prefix_cache_real_seed"


def test_glm_producer_artifact_without_repetition_evidence_is_refused(tmp_path):
    cell = glm_cell("prefill")
    pod = _pod(tmp_path, cell)
    write_glm_rank(pod, glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED]))
    (pod / EVIDENCE_FILE).unlink()
    with pytest.raises(ValueError, match="native GLM-5.3-Flash result lacks valid repetition_evidence_manifest"):
        _validate(tmp_path, cell)


def test_glm_producer_row_regime_must_follow_its_point(tmp_path):
    cell = glm_cell("decode")
    payload = glm_native_payload("decode", points=DECODE_POINTS)
    payload["results"][0]["kv_seed_regime"] = "real_prefix"
    write_glm_rank(_pod(tmp_path, cell), payload)
    with pytest.raises(ValueError, match="native GLM-5.3-Flash result lacks valid kv_seed_regime"):
        _validate(tmp_path, cell)


def test_unknown_execution_mode_is_refused(tmp_path):
    cell = glm_cell("prefill")
    payload = glm_native_payload("prefill", points=[PREFILL_COLD])
    payload["execution_mode"] = "graph"
    write_glm_rank(_pod(tmp_path, cell), payload)
    with pytest.raises(ValueError, match="native execution identity cell has an unaudited execution mode: 'graph'"):
        _validate(tmp_path, cell)


def test_v41_artifacts_keep_their_contract(tmp_path):
    identity = ("c" * 64, "full", "hbm_tp_sharded", "text")
    cell = replace(_cell(phase="prefill", strategy="pure_tp"), execution_identity=identity, input_text_sha256="a" * 64)
    payload = _native_payload(phase="prefill", rank=0, dp=1)
    payload.update(
        execution_identity=dict(zip(EXECUTION_COLUMNS, identity, strict=True)),
        execution_mode="eager",
        input_provenance={
            "source": "tokenizer_text",
            "text_sha256": "a" * 64,
            "token_ids_sha256": "b" * 64,
            "tokenizer_revision": "pinned",
            "token_count": 100,
            "unique_token_count": 20,
        },
    )
    (_pod(tmp_path, cell) / "benchmark.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="V4.1 native result lacks token-stream manifest"):
        _validate(tmp_path, cell)


_V41_IDENTITY = ("c" * 64, "full", "hbm_tp_sharded", "text")


def test_v41_cell_refuses_an_artifact_that_declares_the_glm_contract(tmp_path):
    # The frozen cell, not the artifact, selects the producer contract. A V4.1 artifact that declares the GLM mode
    # with its own execution identity and the GLM provenance would otherwise skip the V4.1 eager and text checks.
    cell = replace(
        _cell(phase="prefill", strategy="pure_tp"), execution_identity=_V41_IDENTITY, input_text_sha256="a" * 64
    )
    payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
    payload["execution_identity"] = dict(zip(EXECUTION_COLUMNS, _V41_IDENTITY, strict=True))
    artifact = write_glm_rank(_pod(tmp_path, cell), payload)
    with pytest.raises(ValueError) as error:
        _validate(tmp_path, cell)
    assert str(error.value) == (
        f"native execution identity cell expects execution mode 'eager', found 'native_graph_policy': {artifact}"
    )


def test_glm_cell_refuses_an_artifact_that_declares_the_v41_contract(tmp_path):
    # The reverse case: the GLM cell accepts only the GLM producer's mode, whatever V4.1 evidence the artifact holds.
    cell = glm_cell("prefill")
    payload = _native_payload(phase="prefill", rank=0, dp=1)
    payload.update(
        execution_identity=dict(zip(EXECUTION_COLUMNS, glm_identity(), strict=True)),
        execution_mode="eager",
        input_provenance={
            "source": "tokenizer_text",
            "text_sha256": "a" * 64,
            "token_ids_sha256": "b" * 64,
            "tokenizer_revision": "pinned",
            "token_count": 100,
            "unique_token_count": 20,
        },
    )
    artifact = _pod(tmp_path, cell) / "benchmark.json"
    _write_v41_token_streams(payload, artifact)
    artifact.write_text(json.dumps(payload))
    with pytest.raises(ValueError) as error:
        _validate(tmp_path, cell)
    assert str(error.value) == (
        f"native execution identity cell expects execution mode 'native_graph_policy', found 'eager': {artifact}"
    )


def test_cell_without_an_audited_contract_refuses_an_audited_mode(tmp_path):
    # An execution identity whose Engram residency no producer contract has stays refused, whatever mode it declares.
    identity = ("c" * 64, "full", "cpu_offloaded", "text")
    cell = SimpleNamespace(execution_identity=identity, input_text_sha256="a" * 64)
    payload = {"execution_identity": dict(zip(EXECUTION_COLUMNS, identity, strict=True)), "execution_mode": "eager"}
    path = tmp_path / "benchmark.json"
    with pytest.raises(ValueError) as error:
        _validate_execution_provenance(cell, payload, path)
    assert str(error.value) == (
        f"native execution identity cell has no audited producer contract: 'cpu_offloaded': {path}"
    )


def _drop_a_line(payload: dict, sidecar: Path) -> None:
    lines = sidecar.read_bytes().splitlines(keepends=True)[:-1]
    sidecar.write_bytes(b"".join(lines))
    payload["input_provenance"]["repetition_evidence_manifest"]["sha256"] = hashlib.sha256(b"".join(lines)).hexdigest()


def _add_a_line(payload: dict, sidecar: Path) -> None:
    raw = sidecar.read_bytes() + b'{"benchmark_id":99}\n'
    sidecar.write_bytes(raw)
    manifest = payload["input_provenance"]["repetition_evidence_manifest"]
    manifest.update(sha256=hashlib.sha256(raw).hexdigest(), records=len(raw.splitlines()))


def _evidence(payload: dict) -> dict:
    return payload["input_provenance"]


def _rewrite_sidecar(payload: dict, sidecar: Path, lines: list[bytes]) -> None:
    """Write ``lines`` as the sidecar, each with a line break, and give the manifest their digest and their count."""
    raw = b"".join(line + b"\n" for line in lines)
    sidecar.write_bytes(raw)
    manifest = payload["input_provenance"]["repetition_evidence_manifest"]
    manifest.update(sha256=hashlib.sha256(raw).hexdigest(), records=len(lines))


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda record: record.pop("repetitions"), "5 warmups and 10 measurements"),
        (lambda record: record["repetitions"].pop(4), "5 warmups and 10 measurements"),
        (lambda record: record["repetitions"][4].update(role="measurement"), "invalid accepted warmup"),
        (lambda record: record["repetitions"][5].update(repetition=4), "invalid accepted measurement"),
        (lambda record: record["repetitions"][5].update(attempt=4), "invalid accepted measurement"),
        (
            lambda record: record["repetitions"][4].update(rejection_reason="missing dispatch"),
            "invalid accepted warmup",
        ),
        (lambda record: record["repetitions"][4].update(fpms=[]), "invalid forward for warmup"),
        (lambda record: record["repetitions"][4]["fpms"].append({}), "invalid forward for warmup"),
        (
            lambda record: record["repetitions"][5]["fpms"][-1]["benchmark_sample"]["cudagraph"].update(
                runtime_mode="NONE"
            ),
            "differs from its saved raw sample",
        ),
        (
            lambda record: record["repetitions"][4]["fpms"][-1]["scheduled_requests"].update(sum_prefill_tokens=0),
            "invalid forward for warmup",
        ),
        (
            lambda record: record["repetitions"][4]["fpms"][-1]["benchmark_sample"].update(forward_index=3),
            "repeated or unordered forward",
        ),
        (
            lambda record: record["repetitions"][5]["fpms"][-1].update(wall_time=9.0),
            "differs from its saved raw sample",
        ),
        (lambda record: record.update(point_type="decode"), "5 warmups and 10 measurements"),
    ],
)
def test_glm_publication_checks_the_contents_of_each_repetition(tmp_path, change, expected):
    cell = glm_cell("prefill")
    payload = glm_native_payload("prefill", points=[PREFILL_COLD])
    _glm_producer_shape(tmp_path)("prefill", payload)
    artifact = write_glm_rank(_pod(tmp_path, cell), payload)
    _check_altered_repetition(tmp_path, cell, payload, artifact, change, expected)


@pytest.mark.parametrize(
    "change",
    [
        lambda record: record["repetitions"][4]["fpms"].pop(0),
        lambda record: record["repetitions"][4]["fpms"][0]["scheduled_requests"].update(num_decode_requests=2),
        lambda record: record["repetitions"][4]["fpms"][0]["scheduled_requests"].update(sum_decode_kv_tokens=4096),
    ],
    ids=["missing-first-step", "wrong-first-batch", "wrong-first-kv-count"],
)
def test_glm_publication_requires_both_decode_steps(tmp_path, change):
    cell = glm_cell("decode")
    payload = glm_native_payload("decode", points=[DECODE_POINTS[0]])
    _glm_producer_shape(tmp_path)("decode", payload)
    artifact = write_glm_rank(_pod(tmp_path, cell), payload)
    _check_altered_repetition(tmp_path, cell, payload, artifact, change, "invalid forward for warmup")


def _check_altered_repetition(tmp_path, cell, payload, artifact, change, expected):
    sidecar = artifact.with_suffix(".repetitions.jsonl")
    record = json.loads(sidecar.read_bytes())
    change(record)
    line = json.dumps(record).encode()
    _rewrite_sidecar(payload, sidecar, [line])
    payload["results"][0]["repetition_evidence_sha256"] = hashlib.sha256(line).hexdigest()
    artifact.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=expected):
        database.aggregate_cell(_glm_plan(), cell, tmp_path, expected_attempt_id=ATTEMPT_ID)


def _swap_the_row_digests(payload: dict, _sidecar: Path) -> None:
    first, second = payload["results"]
    first["repetition_evidence_sha256"], second["repetition_evidence_sha256"] = (
        second["repetition_evidence_sha256"],
        first["repetition_evidence_sha256"],
    )


def _sidecar_lines(sidecar: Path) -> list[bytes]:
    return sidecar.read_bytes().splitlines()


def _repeat_the_first_id(payload: dict, sidecar: Path) -> None:
    lines = [json.loads(line) for line in _sidecar_lines(sidecar)]
    lines[1]["benchmark_id"] = lines[0]["benchmark_id"]
    _rewrite_sidecar(
        payload, sidecar, [json.dumps(line, sort_keys=True, separators=(",", ":")).encode() for line in lines]
    )


def _first_id_as_a_string(payload: dict, sidecar: Path) -> None:
    lines = _sidecar_lines(sidecar)
    first = json.loads(lines[0])
    first["benchmark_id"] = str(first["benchmark_id"])
    _rewrite_sidecar(payload, sidecar, [json.dumps(first, sort_keys=True, separators=(",", ":")).encode(), *lines[1:]])


def _first_line_is_no_json(payload: dict, sidecar: Path) -> None:
    _rewrite_sidecar(payload, sidecar, [b"not json", *_sidecar_lines(sidecar)[1:]])


def _first_line_is_no_object(payload: dict, sidecar: Path) -> None:
    _rewrite_sidecar(payload, sidecar, [b"[1]", *_sidecar_lines(sidecar)[1:]])


def _first_line_is_no_utf8(payload: dict, sidecar: Path) -> None:
    _rewrite_sidecar(payload, sidecar, [b"\xff", *_sidecar_lines(sidecar)[1:]])


def _second_line_names_another_point(payload: dict, sidecar: Path) -> None:
    # Rows 1 and 2 with sidecar lines 1 and 3. The second row carries the digest of the line with ID 3, so its own ID 2
    # names no line.
    lines = _sidecar_lines(sidecar)
    second = json.loads(lines[1])
    second["benchmark_id"] = 3
    line = json.dumps(second, sort_keys=True, separators=(",", ":")).encode()
    _rewrite_sidecar(payload, sidecar, [lines[0], line])
    payload["results"][1]["repetition_evidence_sha256"] = hashlib.sha256(line).hexdigest()


def _first_line_id_is_a_boolean(payload: dict, sidecar: Path) -> None:
    # The row of ID 1 carries the digest of the changed line, so only the type of the line's ID is wrong.
    lines = _sidecar_lines(sidecar)
    first = json.loads(lines[0])
    first["benchmark_id"] = True
    line = json.dumps(first, sort_keys=True, separators=(",", ":")).encode()
    _rewrite_sidecar(payload, sidecar, [line, *lines[1:]])
    payload["results"][0]["repetition_evidence_sha256"] = hashlib.sha256(line).hexdigest()


REFUSALS = [
    pytest.param(
        "execution_identity",
        lambda p, _: p["execution_identity"].update(execution_profile="decoder_bounded"),
        id="identity",
    ),
    pytest.param("input_provenance", lambda p, _: p.update(input_provenance=[]), id="provenance-not-a-mapping"),
    pytest.param(
        "input_provenance.source", lambda p, _: _evidence(p).update(source="tokenizer_text"), id="input-source"
    ),
    pytest.param("input_provenance.generator", lambda p, _: _evidence(p).update(generator=""), id="generator"),
    pytest.param(
        "input_provenance.generator_version",
        lambda p, _: _evidence(p).update(generator_version="1"),
        id="generator-version",
    ),
    pytest.param("input_provenance.sampling", lambda p, _: _evidence(p).update(sampling=None), id="sampling"),
    pytest.param("input_provenance.seed", lambda p, _: _evidence(p).update(seed=True), id="seed"),
    pytest.param("input_provenance.seed", lambda p, _: _evidence(p).update(seed=-1), id="seed-negative"),
    pytest.param("input_provenance.seed", lambda p, _: _evidence(p).update(seed=2**64), id="seed-above-64-bits"),
    pytest.param("input_provenance.count", lambda p, _: _evidence(p).update(count="262240"), id="count"),
    pytest.param("input_provenance.count", lambda p, _: _evidence(p).update(count=0), id="count-zero"),
    pytest.param("input_provenance.count", lambda p, _: _evidence(p).update(count=-1), id="count-negative"),
    pytest.param("input_provenance.count", lambda p, _: _evidence(p).update(count=True), id="count-is-a-boolean"),
    pytest.param(
        "input_provenance.unique_token_count",
        lambda p, _: _evidence(p).update(unique_token_count=_evidence(p)["count"] + 1),
        id="unique-count-above-count",
    ),
    pytest.param(
        "input_provenance.unique_token_count",
        lambda p, _: _evidence(p).update(unique_token_count=0),
        id="unique-count-zero",
    ),
    pytest.param(
        "input_provenance.unique_token_count",
        lambda p, _: _evidence(p).update(unique_token_count=True),
        id="unique-count-is-a-boolean",
    ),
    pytest.param(
        "input_provenance.unique_token_count",
        lambda p, _: _evidence(p).pop("unique_token_count"),
        id="unique-count-absent",
    ),
    # The pinned vocabulary has 154,820 ids, so a stream of 262,240 tokens cannot hold more distinct tokens than that.
    pytest.param(
        "input_provenance.unique_token_count",
        lambda p, _: _evidence(p).update(unique_token_count=_evidence(p)["count"]),
        id="unique-count-equals-count-above-the-vocabulary",
    ),
    pytest.param(
        "input_provenance.unique_token_count",
        lambda p, _: _evidence(p).update(unique_token_count=seeded.VOCAB_HIGH - seeded.VOCAB_LOW + 1),
        id="unique-count-one-above-the-vocabulary",
    ),
    # The vocabulary is the half-open range of its low and high ids, so ids 50 to 149 are 100 ids and not 150.
    pytest.param(
        "input_provenance.unique_token_count",
        lambda p, _: _evidence(p).update(vocab_low=50, vocab_high_exclusive=150, unique_token_count=101),
        id="unique-count-above-a-vocabulary-that-starts-above-zero",
    ),
    pytest.param("input_provenance.vocab_low", lambda p, _: _evidence(p).update(vocab_low=-1), id="vocab-low-negative"),
    pytest.param(
        "input_provenance.vocab_low", lambda p, _: _evidence(p).update(vocab_low=False), id="vocab-low-is-a-boolean"
    ),
    pytest.param(
        "input_provenance.vocab_low", lambda p, _: _evidence(p).update(vocab_low=0.0), id="vocab-low-is-a-float"
    ),
    pytest.param("input_provenance.vocab_low", lambda p, _: _evidence(p).pop("vocab_low"), id="vocab-low-absent"),
    pytest.param(
        "input_provenance.vocab_high_exclusive",
        lambda p, _: _evidence(p).update(vocab_high_exclusive=True),
        id="vocab-high-is-a-boolean",
    ),
    pytest.param(
        "input_provenance.vocab_high_exclusive",
        lambda p, _: _evidence(p).update(vocab_high_exclusive=float(seeded.VOCAB_HIGH)),
        id="vocab-high-is-a-float",
    ),
    pytest.param(
        "input_provenance.vocab_high_exclusive",
        lambda p, _: _evidence(p).pop("vocab_high_exclusive"),
        id="vocab-high-absent",
    ),
    pytest.param(
        "input_provenance.vocab_high_exclusive",
        lambda p, _: _evidence(p).update(vocab_high_exclusive=_evidence(p)["vocab_low"]),
        id="vocab-range-empty",
    ),
    pytest.param(
        "input_provenance.vocab_high_exclusive",
        lambda p, _: _evidence(p).update(vocab_low=10, vocab_high_exclusive=5),
        id="vocab-range-inverted",
    ),
    pytest.param(
        "input_provenance.vocab_high_exclusive",
        lambda p, _: _evidence(p).update(vocab_high_exclusive=-1),
        id="vocab-high-negative",
    ),
    pytest.param(
        "input_provenance.vocab_low",
        lambda p, _: _evidence(p).update(vocab_low=-10, vocab_high_exclusive=-5),
        id="vocab-range-negative",
    ),
    pytest.param(
        "input_provenance.tokenizer_json_sha256",
        lambda p, _: _evidence(p).update(tokenizer_json_sha256="abc"),
        id="tokenizer-hash",
    ),
    pytest.param(
        "input_provenance.token_ids_sha256",
        lambda p, _: _evidence(p).update(token_ids_sha256="A" * 64),
        id="token-ids-hash-uppercase",
    ),
    pytest.param(
        "input_provenance.tokenizer_revision",
        lambda p, _: _evidence(p).update(tokenizer_revision="main"),
        id="tokenizer-revision",
    ),
    pytest.param(
        "input_provenance.chunk_start_alignment",
        lambda p, _: _evidence(p).update(chunk_start_alignment=1),
        id="chunk-alignment",
    ),
    pytest.param(
        "repetition_evidence_manifest",
        lambda p, _: _evidence(p).pop("repetition_evidence_manifest"),
        id="manifest-absent",
    ),
    pytest.param(
        "repetition_evidence_manifest",
        lambda p, _: _evidence(p)["repetition_evidence_manifest"].update(schema_version=2),
        id="manifest-schema",
    ),
    pytest.param(
        "repetition_evidence_manifest",
        lambda p, _: _evidence(p)["repetition_evidence_manifest"].update(schema_version=True),
        id="manifest-schema-is-a-boolean",
    ),
    pytest.param(
        "repetition_evidence_manifest",
        lambda p, _: _evidence(p)["repetition_evidence_manifest"].update(file="other.repetitions.jsonl"),
        id="manifest-file-name",
    ),
    pytest.param(
        "repetition_evidence_manifest",
        lambda p, _: _evidence(p)["repetition_evidence_manifest"].update(file=f"../{EVIDENCE_FILE}"),
        id="manifest-file-outside-the-directory",
    ),
    pytest.param(
        "repetition_evidence_manifest",
        lambda p, _: _evidence(p)["repetition_evidence_manifest"].update(sha256="0" * 64),
        id="manifest-digest",
    ),
    pytest.param(
        "repetition_evidence_manifest",
        lambda p, _: _evidence(p)["repetition_evidence_manifest"].update(records=3),
        id="manifest-records",
    ),
    pytest.param(
        "repetition_evidence_manifest",
        lambda p, _: _evidence(p)["repetition_evidence_manifest"].update(records=2.0),
        id="manifest-records-is-a-float",
    ),
    pytest.param("repetition_evidence_manifest", _drop_a_line, id="sidecar-has-fewer-lines-than-records"),
    pytest.param("repetition_evidence_manifest", _add_a_line, id="sidecar-has-more-lines-than-rows"),
    pytest.param("kvwarm.enabled", lambda p, _: p.pop("kvwarm"), id="kvwarm-absent"),
    pytest.param("kvwarm.enabled", lambda p, _: p["kvwarm"].update(enabled=False), id="kvwarm-disabled"),
    pytest.param("kvwarm.enabled", lambda p, _: p["kvwarm"].update(enabled=1), id="kvwarm-enabled-is-not-a-boolean"),
    pytest.param(
        "kvwarm.warm_eligible", lambda p, _: p["kvwarm"].update(warm_eligible=False), id="kvwarm-not-eligible"
    ),
    pytest.param("kvwarm.skip_reason", lambda p, _: p["kvwarm"].update(skip_reason="skipped"), id="kvwarm-skip-reason"),
    pytest.param(
        "kvwarm.method",
        lambda p, _: p["kvwarm"].update(method="native_radix_cache_real_seed"),
        id="kvwarm-method",
    ),
    pytest.param(
        "kvwarm.state_protocol",
        lambda p, _: p["kvwarm"].update(state_protocol="other_protocol"),
        id="kvwarm-state-protocol",
    ),
    pytest.param("timing_boundary", lambda p, _: p.update(timing_boundary="wall_clock"), id="timing-boundary"),
    pytest.param(
        "serving_config.enable_prefix_caching",
        lambda p, _: p["serving_config"].update(enable_prefix_caching=False),
        id="prefix-caching-off",
    ),
    pytest.param(
        "serving_config.async_scheduling",
        lambda p, _: p["serving_config"].update(async_scheduling=True),
        id="async-scheduling-on",
    ),
    pytest.param(
        "serving_config.mamba_cache_mode",
        lambda p, _: p["serving_config"].update(mamba_cache_mode="none"),
        id="mamba-mode",
    ),
    *[
        pytest.param(
            "serving_config.block_size",
            lambda p, _, size=size: p["serving_config"].update(block_size=size, hash_block_size=size),
            id=f"invalid-block-size-{size}",
        )
        for size in (0, -4, 3)
    ],
    pytest.param(
        "serving_config.block_size",
        lambda p, _: p["serving_config"].update(block_size=16),
        id="scheduler-block-differs-from-the-hash-block",
    ),
    pytest.param(
        "serving_config.need_mamba_block_aligned_split",
        lambda p, _: p["serving_config"].update(need_mamba_block_aligned_split=False),
        id="no-aligned-split",
    ),
    pytest.param(
        "serving_config.block_size",
        lambda p, _: p["serving_config"].update(block_size=2176.0, hash_block_size=2176.0),
        id="blocks-are-floats",
    ),
    pytest.param(
        "serving_config.block_size",
        lambda p, _: p["serving_config"].update(hash_block_size=2176.0),
        id="hash-block-is-a-float",
    ),
    pytest.param(
        "serving_config.need_mamba_block_aligned_split",
        lambda p, _: p["serving_config"].update(need_mamba_block_aligned_split=1),
        id="aligned-split-is-not-a-boolean",
    ),
    pytest.param(
        "serving_config.enable_prefix_caching", lambda p, _: p.pop("serving_config"), id="serving-config-absent"
    ),
    pytest.param(
        "producer.runtime_source_manifest_sha256",
        lambda p, _: p["producer"].update(runtime_source_manifest_sha256=None),
        id="runtime-source-manifest",
    ),
    pytest.param("producer", lambda p, _: p.pop("producer"), id="producer-absent"),
    pytest.param(
        "repetition_evidence_sha256",
        lambda p, _: p["results"][1].pop("repetition_evidence_sha256"),
        id="row-digest-absent",
    ),
    pytest.param("repetition_evidence_sha256", _swap_the_row_digests, id="row-digests-swapped"),
    pytest.param(
        "repetition_evidence_sha256",
        lambda p, _: p["results"][1].update(repetition_evidence_sha256=p["results"][0]["repetition_evidence_sha256"]),
        id="row-digest-repeated",
    ),
    pytest.param(
        "repetition_evidence_sha256",
        lambda p, _: p["results"][0].update(repetition_evidence_sha256=hashlib.sha256(b"another point").hexdigest()),
        id="row-digest-of-no-line",
    ),
    pytest.param("repetition_evidence", _repeat_the_first_id, id="sidecar-repeats-an-id"),
    pytest.param("repetition_evidence", _first_id_as_a_string, id="sidecar-id-is-a-string"),
    pytest.param("repetition_evidence", _first_line_is_no_json, id="sidecar-line-is-no-json"),
    pytest.param("repetition_evidence", _first_line_is_no_object, id="sidecar-line-is-no-object"),
    pytest.param("repetition_evidence", _first_line_is_no_utf8, id="sidecar-line-is-no-utf8"),
    pytest.param("repetition_evidence", _first_line_id_is_a_boolean, id="sidecar-id-is-a-boolean"),
    pytest.param("repetition_evidence_sha256", _second_line_names_another_point, id="row-id-of-no-line"),
    pytest.param(
        "kv_seed_regime", lambda p, _: p["results"][0].update(kv_seed_regime="real_prefix"), id="regime-of-cold-row"
    ),
    pytest.param(
        "kv_seed_regime",
        lambda p, _: p["results"][1].update(kv_seed_regime="not_applicable"),
        id="regime-of-cached-row",
    ),
    pytest.param(
        "input_provenance.generator",
        lambda p, _: _evidence(p).update(generator="other"),
        id="generator-is-not-the-pinned-one",
    ),
    pytest.param(
        "input_provenance.generator_version",
        lambda p, _: _evidence(p).update(generator_version=1),
        id="generator-version-1",
    ),
    pytest.param(
        "input_provenance.generator_version",
        lambda p, _: _evidence(p).update(generator_version=3),
        id="generator-version-3",
    ),
    pytest.param("input_provenance.seed", lambda p, _: _evidence(p).update(seed=54), id="seed-is-not-the-pinned-one"),
    pytest.param(
        "input_provenance.count", lambda p, _: _evidence(p).update(count=262239), id="count-is-not-the-producer-count"
    ),
    pytest.param(
        "input_provenance.vocab_low",
        lambda p, _: _evidence(p).update(vocab_low=1),
        id="vocab-low-is-not-the-pinned-one",
    ),
    pytest.param(
        "input_provenance.vocab_high_exclusive",
        lambda p, _: _evidence(p).update(vocab_high_exclusive=154821),
        id="vocab-high-is-not-the-pinned-one",
    ),
    pytest.param(
        "input_provenance.tokenizer_json_sha256",
        lambda p, _: _evidence(p).update(tokenizer_json_sha256="0" * 64),
        id="tokenizer-digest-is-not-the-pinned-one",
    ),
    pytest.param(
        "input_provenance.token_ids_sha256",
        lambda p, _: _evidence(p).update(token_ids_sha256=hashlib.sha256(b"another stream").hexdigest()),
        id="stream-digest-of-another-stream",
    ),
    pytest.param(
        "input_provenance.unique_token_count",
        lambda p, _: _evidence(p).update(unique_token_count=126302),
        id="distinct-count-of-another-stream",
    ),
]


@pytest.mark.parametrize(("field", "tamper"), REFUSALS)
def test_glm_producer_artifact_refuses_each_unaudited_item(tmp_path, field, tamper):
    cell = glm_cell("prefill")
    pod = _pod(tmp_path, cell)
    payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
    artifact = write_glm_rank(pod, payload)
    tamper(payload, pod / EVIDENCE_FILE)
    artifact.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=rf"native GLM-5\.3-Flash result lacks valid {re.escape(field)}: "):
        _validate(tmp_path, cell)


def test_glm_producer_token_evidence_must_be_the_pinned_stream(tmp_path, monkeypatch):
    producer = load_producer(monkeypatch)
    # The producer draws its largest measured context of 131,072 tokens plus a window of 4,099 tokens for each of up
    # to 32 requests, from seed 53 over the pinned vocabulary of 154,820 ids. Its distinct-token count is that of the
    # real stream, which does not draw every id.
    count = producer.MAX_CONTEXT + producer.MAX_BATCH * seeded.REQUEST_STRIDE
    tokens = seeded.generate(seeded.SEED, count, seeded.VOCAB_LOW, seeded.VOCAB_HIGH)
    assert (seeded.SEED, count, len(set(tokens))) == (53, 262240, 126303)
    cases = {
        "producer": ({"seed": seeded.SEED, "count": count, "unique_token_count": len(set(tokens))}, None),
        # The smallest seed and a stream of one token.
        "smallest": ({"seed": 0, "count": 1, "unique_token_count": 1}, "input_provenance.seed"),
        # The largest seed and a stream that draws every id of the vocabulary, the most distinct tokens that it allows.
        "largest": (
            {"seed": 2**64 - 1, "count": count, "unique_token_count": seeded.VOCAB_HIGH - seeded.VOCAB_LOW},
            "input_provenance.seed",
        ),
        # The ids 50 to 149 are 100 ids, because the upper bound is exclusive. They cap the distinct tokens of a longer
        # stream at 100.
        "narrow-vocabulary": (
            {"count": count, "unique_token_count": 100, "vocab_low": 50, "vocab_high_exclusive": 150},
            "input_provenance.vocab_low",
        ),
    }
    for name, (fields, refused) in cases.items():
        cell = glm_cell("prefill")
        payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
        payload["input_provenance"].update(fields)
        write_glm_rank(_pod(tmp_path / name, cell), payload)
        if refused is None:
            assert len(_validate(tmp_path / name, cell).points) == 2
        else:
            with pytest.raises(ValueError, match=rf"native GLM-5\.3-Flash result lacks valid {re.escape(refused)}: "):
                _validate(tmp_path / name, cell)


def test_fixture_holds_the_real_token_stream_record():
    tokens = seeded.generate(seeded.SEED, 262240, seeded.VOCAB_LOW, seeded.VOCAB_HIGH)
    digest = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
    evidence = glm_native_payload("prefill", points=[PREFILL_COLD])["input_provenance"]
    record = native_artifact._glm_input_record(seeded.GENERATOR_VERSION)
    assert (digest, len(set(tokens))) == (evidence["token_ids_sha256"], evidence["unique_token_count"])
    assert (digest, len(set(tokens))) == (record["token_ids_sha256"], record["unique_token_count"])
    assert native_artifact._glm_input_record(1) is None


def test_seeded_input_record_cannot_be_changed_by_a_caller(tmp_path):
    record = native_artifact._glm_input_record(seeded.GENERATOR_VERSION)
    try:
        with pytest.raises(TypeError):
            record["seed"] = 54
        with pytest.raises(TypeError):
            del record["seed"]
        # Every later validation of the process still compares with the pinned seed.
        cell = glm_cell("prefill")
        payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
        payload["input_provenance"]["seed"] = 54
        write_glm_rank(_pod(tmp_path, cell), payload)
        with pytest.raises(ValueError, match=r"native GLM-5\.3-Flash result lacks valid input_provenance\.seed: "):
            _validate(tmp_path, cell)
    finally:
        # A record that this test could change is drawn again for the tests that follow it.
        native_artifact._glm_input_record.cache_clear()


@pytest.mark.parametrize(
    ("benchmark_id", "refusal"),
    [
        pytest.param(False, "native point field 'benchmark_id' must be an integer, got False", id="boolean"),
        pytest.param(2.0, "native point field 'benchmark_id' must be an integer, got 2.0", id="float"),
        pytest.param("1", "native point field 'benchmark_id' must be an integer, got '1'", id="string"),
    ],
)
def test_row_id_that_is_no_integer_is_refused_by_the_check_of_the_row_ids(tmp_path, benchmark_id, refusal):
    # The row carries the digest of the line with ID 1. A plain lookup by its ID would not give that digest: no line has
    # the key False (0), and the key 2.0 finds the line of the second row. The digest rule compares only a row whose ID
    # is an integer and not a bool, so the check of the row IDs names the ID.
    cell = glm_cell("prefill")
    pod = _pod(tmp_path, cell)
    payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
    artifact = write_glm_rank(pod, payload)
    payload["results"][0]["point"]["benchmark_id"] = benchmark_id
    artifact.write_text(json.dumps(payload))
    with pytest.raises(ValueError) as error:
        _validate(tmp_path, cell)
    assert str(error.value) == refusal


def test_digest_rule_passes_a_row_whose_id_is_a_list_without_an_error(tmp_path):
    # The digest rule compares only a row whose ID is an integer, so a list ID passes it and every other rule of the
    # GLM-5.3-Flash reader. The check of the row IDs comes after it.
    cell = glm_cell("prefill")
    pod = _pod(tmp_path, cell)
    payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
    artifact = write_glm_rank(pod, payload)
    payload["results"][0]["point"]["benchmark_id"] = [1]
    artifact.write_text(json.dumps(payload))
    stored = json.loads(artifact.read_text())
    assert native_artifact._validate_glm_provenance(cell, stored, artifact) == stored["input_provenance"]


def _other_execution_profile(payload: dict, _sidecar: Path) -> None:
    payload["execution_identity"].update(execution_profile="decoder_bounded")


def _no_chunk_alignment(payload: dict, _sidecar: Path) -> None:
    _evidence(payload).update(chunk_start_alignment=1)


def _kvwarm_disabled(payload: dict, _sidecar: Path) -> None:
    payload["kvwarm"].update(enabled=False)


def _wall_clock_boundary(payload: dict, _sidecar: Path) -> None:
    payload.update(timing_boundary="wall_clock")


def _async_scheduling(payload: dict, _sidecar: Path) -> None:
    payload["serving_config"].update(async_scheduling=True)


def _other_warmup_repeats(payload: dict, _sidecar: Path) -> None:
    payload["producer"].update(warmup_repeats=6)


def _cold_row_with_a_prefix_regime(payload: dict, _sidecar: Path) -> None:
    payload["results"][0].update(kv_seed_regime="real_prefix")


def _second_row_without_a_digest(payload: dict, _sidecar: Path) -> None:
    payload["results"][1].pop("repetition_evidence_sha256")


def _second_row_with_the_digest_of_no_line(payload: dict, _sidecar: Path) -> None:
    payload["results"][1].update(repetition_evidence_sha256=hashlib.sha256(b"another point").hexdigest())


def _float_record_count(payload: dict, _sidecar: Path) -> None:
    _evidence(payload)["repetition_evidence_manifest"].update(records=2.0)


def _mamba_mode_none(payload: dict, _sidecar: Path) -> None:
    payload["serving_config"].update(mamba_cache_mode="none")


def _small_scheduler_block(payload: dict, _sidecar: Path) -> None:
    payload["serving_config"].update(block_size=16)


def _no_aligned_split(payload: dict, _sidecar: Path) -> None:
    payload["serving_config"].update(need_mamba_block_aligned_split=False)


def _other_seed(payload: dict, _sidecar: Path) -> None:
    _evidence(payload).update(seed=54)


# The reader checks, in this order: the execution identity, the input fields, the sidecar manifest with the type of its
# record count, kvwarm, the timing boundary, the serving keys enable_prefix_caching and async_scheduling, the Mamba
# cache mode, the scheduler block, the aligned split, the producer record, the sidecar lines, then for each row in turn
# the form of its digest, its digest against its line and its regime, and last the seeded input record. Each case names
# a check that runs first and a check that runs second. An artifact with both faults is refused for the check that runs
# first, whichever fault was written first.
ORDERED_FAULTS = [
    pytest.param(
        _other_execution_profile,
        "execution_identity",
        _float_record_count,
        "repetition_evidence_manifest",
        id="execution-identity-before-the-record-count",
    ),
    pytest.param(
        _no_chunk_alignment,
        "input_provenance.chunk_start_alignment",
        _float_record_count,
        "repetition_evidence_manifest",
        id="input-fields-before-the-record-count",
    ),
    pytest.param(
        _float_record_count,
        "repetition_evidence_manifest",
        _kvwarm_disabled,
        "kvwarm.enabled",
        id="record-count-before-kvwarm",
    ),
    pytest.param(
        _float_record_count,
        "repetition_evidence_manifest",
        _wall_clock_boundary,
        "timing_boundary",
        id="record-count-before-the-timing-boundary",
    ),
    pytest.param(
        _async_scheduling,
        "serving_config.async_scheduling",
        _mamba_mode_none,
        "serving_config.mamba_cache_mode",
        id="async-scheduling-before-the-mamba-mode",
    ),
    pytest.param(
        _async_scheduling,
        "serving_config.async_scheduling",
        _small_scheduler_block,
        "serving_config.block_size",
        id="async-scheduling-before-the-scheduler-block",
    ),
    pytest.param(
        _async_scheduling,
        "serving_config.async_scheduling",
        _no_aligned_split,
        "serving_config.need_mamba_block_aligned_split",
        id="async-scheduling-before-the-aligned-split",
    ),
    pytest.param(
        _mamba_mode_none,
        "serving_config.mamba_cache_mode",
        _small_scheduler_block,
        "serving_config.block_size",
        id="mamba-mode-before-the-scheduler-block",
    ),
    pytest.param(
        _mamba_mode_none,
        "serving_config.mamba_cache_mode",
        _no_aligned_split,
        "serving_config.need_mamba_block_aligned_split",
        id="mamba-mode-before-the-aligned-split",
    ),
    pytest.param(
        _small_scheduler_block,
        "serving_config.block_size",
        _no_aligned_split,
        "serving_config.need_mamba_block_aligned_split",
        id="scheduler-block-before-the-aligned-split",
    ),
    pytest.param(
        _mamba_mode_none,
        "serving_config.mamba_cache_mode",
        _other_warmup_repeats,
        "producer.warmup_repeats",
        id="mamba-mode-before-the-producer-record",
    ),
    pytest.param(
        _small_scheduler_block,
        "serving_config.block_size",
        _other_warmup_repeats,
        "producer.warmup_repeats",
        id="scheduler-block-before-the-producer-record",
    ),
    pytest.param(
        _no_aligned_split,
        "serving_config.need_mamba_block_aligned_split",
        _other_warmup_repeats,
        "producer.warmup_repeats",
        id="aligned-split-before-the-producer-record",
    ),
    pytest.param(
        _other_warmup_repeats,
        "producer.warmup_repeats",
        _first_line_is_no_json,
        "repetition_evidence",
        id="producer-record-before-the-sidecar-lines",
    ),
    pytest.param(
        _first_line_is_no_json,
        "repetition_evidence",
        _cold_row_with_a_prefix_regime,
        "kv_seed_regime",
        id="sidecar-lines-before-the-rows",
    ),
    pytest.param(
        _other_warmup_repeats,
        "producer.warmup_repeats",
        _swap_the_row_digests,
        "repetition_evidence_sha256",
        id="producer-record-before-the-row-digests",
    ),
    pytest.param(
        _swap_the_row_digests,
        "repetition_evidence_sha256",
        _cold_row_with_a_prefix_regime,
        "kv_seed_regime",
        id="row-digest-before-the-regime-of-its-row",
    ),
    pytest.param(
        _cold_row_with_a_prefix_regime,
        "kv_seed_regime",
        _second_row_with_the_digest_of_no_line,
        "repetition_evidence_sha256",
        id="regime-of-the-first-row-before-the-digest-of-the-second-row",
    ),
    pytest.param(
        _cold_row_with_a_prefix_regime,
        "kv_seed_regime",
        _other_seed,
        "input_provenance.seed",
        id="row-regimes-before-the-seeded-input",
    ),
    pytest.param(
        _second_row_without_a_digest,
        "repetition_evidence_sha256",
        _other_seed,
        "input_provenance.seed",
        id="row-digests-before-the-seeded-input",
    ),
]


@pytest.mark.parametrize(("runs_first", "first_field", "runs_second", "second_field"), ORDERED_FAULTS)
def test_two_faults_are_refused_for_the_first_in_the_order_of_the_reader(
    tmp_path, runs_first, first_field, runs_second, second_field
):
    cell = glm_cell("prefill")
    for name, tampers, field in (
        ("first", [runs_first], first_field),
        ("second", [runs_second], second_field),
        ("both", [runs_first, runs_second], first_field),
        ("both-written-the-other-way", [runs_second, runs_first], first_field),
    ):
        pod = _pod(tmp_path / name, cell)
        payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
        artifact = write_glm_rank(pod, payload)
        for tamper in tampers:
            tamper(payload, pod / EVIDENCE_FILE)
        artifact.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match=rf"native GLM-5\.3-Flash result lacks valid {re.escape(field)}: "):
            _validate(tmp_path / name, cell)


def test_sidecar_lines_in_another_order_than_the_rows_still_validate(tmp_path):
    # The producer writes the lines in the order in which the points ran, so a row is bound to its line by its
    # benchmark_id and not by its position.
    cell = glm_cell("prefill")
    pod = _pod(tmp_path, cell)
    payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
    artifact = write_glm_rank(pod, payload)
    _rewrite_sidecar(payload, pod / EVIDENCE_FILE, list(reversed(_sidecar_lines(pod / EVIDENCE_FILE))))
    artifact.write_text(json.dumps(payload))
    assert len(_validate(tmp_path, cell).points) == 2


def test_record_count_of_one_must_be_an_integer(tmp_path):
    # True equals 1 in Python, so only the type of the count tells it from the one record of this sidecar.
    cell = glm_cell("prefill")
    pod = _pod(tmp_path, cell)
    payload = glm_native_payload("prefill", points=[PREFILL_COLD])
    artifact = write_glm_rank(pod, payload)
    _evidence(payload)["repetition_evidence_manifest"]["records"] = True
    artifact.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=r"native GLM-5\.3-Flash result lacks valid repetition_evidence_manifest: "):
        _validate(tmp_path, cell)


def test_malformed_sidecar_manifest_of_another_contract_is_refused_with_a_text(tmp_path):
    # The GLM-5.3-Flash reader does not read a token-stream manifest, so the comparison of the ranks meets it first.
    cell = glm_cell("prefill", dp=2)
    pod = _pod(tmp_path, cell)
    for rank in range(2):
        payload = glm_native_payload("prefill", points=[PREFILL_COLD], rank=rank, dp=2)
        payload["input_provenance"]["token_stream_manifest"] = "x"
        write_glm_rank(pod, payload)
    with pytest.raises(ValueError, match="native input provenance token_stream_manifest must be a mapping, found 'x'"):
        _validate(tmp_path, cell)


@pytest.mark.parametrize(
    ("high", "drawable"),
    [(2**63 + 1, True), (2**63 + 2, False)],
    ids=["range-at-the-generator-cap", "range-above-the-generator-cap"],
)
def test_glm_producer_vocabulary_range_follows_the_generator_cap(tmp_path, high, drawable):
    # The generator takes at most 2**63 ids in its range, and so does the form check of the validator. A lower bound of
    # 1 puts exactly 2**63 ids in the first range and one more in the second, and both upper bounds exceed 2**63. Only
    # the pinned vocabulary passes the reader, so the first range is refused after the form check and the second by it.
    low = 1
    cell = glm_cell("prefill")
    payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
    payload["input_provenance"].update(vocab_low=low, vocab_high_exclusive=high)
    write_glm_rank(_pod(tmp_path, cell), payload)
    if drawable:
        assert len(seeded.generate(seeded.SEED, 1, low, high)) == 1
        with pytest.raises(ValueError, match=r"native GLM-5\.3-Flash result lacks valid input_provenance\.vocab_low: "):
            _validate(tmp_path, cell)
    else:
        with pytest.raises(ValueError, match="invalid token generator request"):
            seeded.generate(seeded.SEED, 1, low, high)
        with pytest.raises(
            ValueError, match=r"native GLM-5\.3-Flash result lacks valid input_provenance\.vocab_high_exclusive: "
        ):
            _validate(tmp_path, cell)


def test_glm_producer_dp_ranks_must_agree_on_input_provenance(tmp_path):
    cell = glm_cell("prefill", dp=2)
    pod = _pod(tmp_path, cell)
    payloads = [glm_native_payload("prefill", points=[PREFILL_COLD], rank=rank, dp=2) for rank in range(2)]
    artifacts = [write_glm_rank(pod, payload) for payload in payloads]
    # Each rank names its own sidecar, and the names alone are no disagreement.
    assert [artifact.name for artifact in artifacts] == ["benchmark.json", "benchmark_dp1.json"]
    assert len(_validate(tmp_path, cell).points) == 1
    assert (
        _evidence(payloads[0])["repetition_evidence_manifest"]["sha256"]
        != _evidence(payloads[1])["repetition_evidence_manifest"]["sha256"]
    )
    _evidence(payloads[1]).update(checkpoint_dir="/another/checkpoint")
    artifacts[1].write_text(json.dumps(payloads[1]))
    with pytest.raises(ValueError, match="native DP ranks disagree on input provenance"):
        _validate(tmp_path, cell)


_NO_KEY = "no key"
_WITH_CHECKPOINT = {"blocks": True, "alignment": 16}
_WITHOUT_CHECKPOINT = {"blocks": False, "alignment": None}


@pytest.mark.parametrize(
    ("first", "second", "refused"),
    [
        pytest.param(_WITH_CHECKPOINT, _WITHOUT_CHECKPOINT, True, id="different-values"),
        pytest.param(_NO_KEY, _WITH_CHECKPOINT, True, id="first-rank-lacks-the-key"),
        pytest.param(_WITH_CHECKPOINT, _NO_KEY, True, id="second-rank-lacks-the-key"),
        pytest.param(_NO_KEY, _NO_KEY, False, id="no-rank-has-the-key"),
        pytest.param(_WITH_CHECKPOINT, dict(_WITH_CHECKPOINT), False, id="equal-values"),
    ],
)
def test_collection_refuses_ranks_that_disagree_on_the_checkpoint_evidence(tmp_path, first, second, refused):
    cell = glm_cell("prefill", dp=2)
    pod = _pod(tmp_path, cell)
    payloads = [glm_native_payload("prefill", points=[PREFILL_COLD], rank=rank, dp=2) for rank in range(2)]
    for payload, value in zip(payloads, (first, second), strict=True):
        if value == _NO_KEY:
            del payload["serving_config"]["mamba_prefill_checkpoint"]
        else:
            payload["serving_config"]["mamba_prefill_checkpoint"] = value
    for payload in payloads:
        write_glm_rank(pod, payload)
    if refused:
        with pytest.raises(
            ValueError,
            match=r"^native ranks disagree on serving_config\.mamba_prefill_checkpoint: .+benchmark_dp1\.json$",
        ):
            _validate(tmp_path, cell)
    else:
        assert len(_validate(tmp_path, cell).points) == 1


REAL_KV_REFUSAL = "execution-identity cells require real KV measurements for cached prefill and decode, got {}"
# Dynamo skips the decode warm-up of a pure tensor-parallel MoE model and stamps every point as a fake fallback.
WARMUP_SKIP = {"enabled": True, "warm_eligible": False, "skip_reason": "moe_tp_balanced_by_construction"}


def test_glm_producer_rows_publish_real_regimes(tmp_path):
    rows = {}
    shape = _glm_producer_shape(tmp_path)
    for phase, points in (("prefill", [PREFILL_COLD, PREFILL_CACHED]), ("decode", DECODE_POINTS)):
        cell = glm_cell(phase)
        payload = glm_native_payload(phase, points=points)
        shape(phase, payload)
        write_glm_rank(_pod(tmp_path / phase, cell), payload)
        rows[phase] = database.aggregate_cell(_glm_plan(), cell, tmp_path / phase, expected_attempt_id=ATTEMPT_ID)
    assert [(row["total_kv_read_tokens"], row["kv_seed_regime"]) for row in rows["prefill"]] == [
        (0, "n/a"),
        (2176, "real_kv"),
    ]
    assert [(row["total_kv_read_tokens"], row["kv_seed_regime"]) for row in rows["decode"]] == [
        (4096, "real_kv"),
        (8192, "real_kv"),
    ]
    for row in [*rows["prefill"], *rows["decode"]]:
        assert row["model_template_id"] == GENERIC_RUNTIME_TEMPLATE_ID
        assert row["execution_profile"] == "full"
        assert tuple(row[name] for name in EXECUTION_COLUMNS) == glm_identity()


def test_glm_publication_keeps_the_retained_values_of_a_producer_record(tmp_path):
    shape = _glm_producer_shape(tmp_path)
    for phase, points in (("prefill", [PREFILL_COLD, PREFILL_CACHED]), ("decode", DECODE_POINTS)):
        cell = glm_cell(phase)
        payload = glm_native_payload(phase, points=points)
        original = [row["fpms"][0]["wall_time"] for row in payload["results"]]
        shape(phase, payload)
        retained = [row["fpms"][0]["wall_time"] for row in payload["results"]]
        # The builder spreads ten repetitions around the original wall time, 1% apart from 0.96 to 1.05 times it, and
        # retains the upper median of repetitions 1 to 9: repetition 5, at 1.01 times the original.
        assert retained == [wall * (1 + 0.01 * (5 - 4)) for wall in original]
        write_glm_rank(_pod(tmp_path / phase, cell), payload)
        rows = database.aggregate_cell(_glm_plan(), cell, tmp_path / phase, expected_attempt_id=ATTEMPT_ID)
        # One rank, so the latency of a row is the retained wall time of its point.
        assert [row["latency_ms"] for row in rows] == [wall * 1000.0 for wall in retained]


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_glm_publication_refuses_a_producer_record_whose_retained_median_moved(tmp_path, phase):
    cell = glm_cell(phase)
    points = [PREFILL_COLD, PREFILL_CACHED] if phase == "prefill" else DECODE_POINTS
    payload = glm_native_payload(phase, points=points)
    _glm_producer_shape(tmp_path, "median")(phase, payload)
    write_glm_rank(_pod(tmp_path, cell), payload)
    # The retained wall time, the group wall time and the timing totals moved together to the slowest repetition, so
    # the native validator accepts the artifact. Only the reduction rule of its measurement record refuses it.
    _validate(tmp_path, cell)
    with pytest.raises(ValueError) as error:
        database.aggregate_cell(_glm_plan(), cell, tmp_path, expected_attempt_id=ATTEMPT_ID)
    assert "benchmark_measurement reduction disagrees with retained timing" in str(error.value)


def _publication_refusal(tmp_path, phase, tamper) -> str:
    """Verify publication refuses the artifact and return the measurement reader's diagnosis.

    A raw sample can also disagree with the independent, unmodified repetition sidecar.
    """
    cell = glm_cell(phase)
    points = [PREFILL_COLD, PREFILL_CACHED] if phase == "prefill" else DECODE_POINTS
    payload = glm_native_payload(phase, points=points)
    _glm_producer_shape(tmp_path)(phase, payload)
    artifact = write_glm_rank(_pod(tmp_path, cell), payload)
    collection = _validate(tmp_path, cell)
    tamper(payload)
    artifact.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        database.aggregate_cell(_glm_plan(), cell, tmp_path, expected_attempt_id=ATTEMPT_ID)
    with pytest.raises(ValueError) as error:
        database.extract_measurement_evidence(tmp_path / "raw", collection)
    return str(error.value)


def _first_record(payload) -> dict:
    return payload["results"][0]["fpms"][0]["benchmark_measurement"]


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_glm_publication_refuses_a_raw_sample_of_another_workload(tmp_path, phase):
    def tamper(payload):
        scheduled = _first_record(payload)["raw_fpms"][3]["scheduled_requests"]
        if phase == "prefill":
            scheduled["sum_prefill_tokens"] += 4
        else:
            scheduled["sum_decode_kv_tokens"] += 1

    refusal = _publication_refusal(tmp_path, phase, tamper)
    assert "benchmark_measurement raw sample 3 does not hold the workload of its point" in refusal


@pytest.mark.parametrize("count", [1, 5])
def test_glm_publication_refuses_a_global_warmup_in_the_protocol(tmp_path, count):
    def tamper(payload):
        payload["measurement_protocol"]["preparation"]["warmup_iterations"] = count

    refusal = _publication_refusal(tmp_path, "prefill", tamper)
    assert (
        "measurement_protocol.preparation.warmup_iterations must be 0 for the GLM-5.3-Flash producer: "
        f"expected 0, found {count}"
    ) in refusal


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_glm_publication_refuses_a_raw_sample_whose_count_is_below_its_point(tmp_path, phase):
    # The decode case holds the count of the first recorded step of a decode repetition: the point's KV total less one
    # token for each request.
    def tamper(payload):
        row = payload["results"][0]
        scheduled = row["fpms"][0]["benchmark_measurement"]["raw_fpms"][3]["scheduled_requests"]
        if phase == "prefill":
            scheduled["sum_prefill_tokens"] -= 4
        else:
            scheduled["sum_decode_kv_tokens"] -= row["point"]["batch_size"]

    refusal = _publication_refusal(tmp_path, phase, tamper)
    assert "benchmark_measurement raw sample 3 does not hold the workload of its point" in refusal


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_glm_publication_refuses_a_raw_sample_of_another_workload_in_a_later_point(tmp_path, phase):
    def tamper(payload):
        row = payload["results"][-1]
        assert row is not payload["results"][0]
        scheduled = row["fpms"][0]["benchmark_measurement"]["raw_fpms"][5]["scheduled_requests"]
        if phase == "prefill":
            scheduled["sum_prefill_tokens"] += 4
        else:
            scheduled["sum_decode_kv_tokens"] += 1

    refusal = _publication_refusal(tmp_path, phase, tamper)
    assert "benchmark_measurement raw sample 5 does not hold the workload of its point" in refusal


_ABSENT = object()


def _set_scheduled(index, key, value):
    def tamper(payload):
        sample = _first_record(payload)["raw_fpms"][index]
        if key is None:
            sample["scheduled_requests"] = value
        elif value is _ABSENT:
            del sample["scheduled_requests"][key]
        else:
            sample["scheduled_requests"][key] = value(sample["scheduled_requests"][key])

    return tamper


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize(
    ("index", "tamper"),
    [
        pytest.param(0, _set_scheduled(0, None, None), id="first-sample-without-a-mapping"),
        pytest.param(9, _set_scheduled(9, None, []), id="last-sample-holds-a-list"),
        pytest.param(3, _set_scheduled(3, "num_decode_requests", _ABSENT), id="key-absent"),
        pytest.param(3, _set_scheduled(3, "num_prefill_requests", float), id="count-is-a-float"),
        pytest.param(3, _set_scheduled(3, "sum_prefill_kv_tokens", bool), id="count-is-a-boolean"),
        pytest.param(3, _set_scheduled(3, "sum_decode_kv_tokens", str), id="count-is-a-string"),
    ],
)
def test_glm_publication_refuses_a_raw_sample_whose_workload_is_malformed(tmp_path, phase, index, tamper):
    # The producer copies the integer counts of Dynamo's FPM into each raw sample, so a count of another JSON type that
    # compares equal (1.0, False for 0) is no workload that the producer writes.
    refusal = _publication_refusal(tmp_path, phase, tamper)
    assert f"benchmark_measurement raw sample {index} does not hold the workload of its point" in refusal


@pytest.mark.parametrize("value", [True, 0.0, 1.0, "0", None], ids=["boolean", "zero-float", "float", "string", "null"])
def test_glm_publication_keeps_the_form_refusal_of_a_warmup_count_that_is_no_integer(tmp_path, value):
    def tamper(payload):
        payload["measurement_protocol"]["preparation"]["warmup_iterations"] = value

    refusal = _publication_refusal(tmp_path, "prefill", tamper)
    assert "measurement_protocol has malformed preparation settings" in refusal


def _global_warmup(payload):
    payload["measurement_protocol"]["preparation"]["warmup_iterations"] = 5


def _warmup_history_of_another_status(payload):
    payload["warmup_evidence"]["status"] = "pending"


def _unobserved_is_text(payload):
    payload["measurement_protocol"]["unobserved"] = "kv_cache_tensors"


def _raw_sample_of_another_workload(payload):
    _first_record(payload)["raw_fpms"][3]["scheduled_requests"]["sum_prefill_tokens"] += 4


def _eleven_expected_samples(payload):
    _first_record(payload)["expected_internal_samples"] = 11


def _raw_sample_is_a_list(payload):
    _first_record(payload)["raw_fpms"][3] = []


def _raw_sample_without_its_sample_record(payload):
    del _first_record(payload)["raw_fpms"][3]["benchmark_sample"]


@pytest.mark.parametrize(
    ("first", "second", "text"),
    [
        pytest.param(
            _unobserved_is_text,
            _global_warmup,
            "measurement_protocol must describe unobserved inputs",
            id="unobserved-inputs-before-the-global-warmup",
        ),
        pytest.param(
            _global_warmup,
            _warmup_history_of_another_status,
            "measurement_protocol.preparation.warmup_iterations must be 0 for the GLM-5.3-Flash producer: "
            "expected 0, found 5",
            id="global-warmup-before-the-warmup-history",
        ),
        pytest.param(
            _eleven_expected_samples,
            _raw_sample_of_another_workload,
            "benchmark_measurement does not hold the producer's ten measured repetitions",
            id="ten-repetitions-before-the-raw-sample-workload",
        ),
        pytest.param(
            _raw_sample_is_a_list,
            lambda payload: None,
            "benchmark_measurement has invalid raw counter/rank/timing evidence",
            id="raw-sample-form-before-the-raw-sample-workload",
        ),
        pytest.param(
            _raw_sample_of_another_workload,
            _raw_sample_without_its_sample_record,
            "benchmark_measurement raw sample 3 does not hold the workload of its point",
            id="raw-sample-workload-before-the-observed-execution",
        ),
    ],
)
def test_glm_publication_reads_the_measurement_record_in_its_order(tmp_path, first, second, text):
    # The check of the measurement record that runs first names its fault when an artifact holds two faults. A raw
    # sample that is no mapping holds no workload, so the form check of the raw samples names it.
    def tamper(payload):
        first(payload)
        second(payload)

    assert text in _publication_refusal(tmp_path, "prefill", tamper)


def test_glm_publication_refuses_an_artifact_without_the_producer_protocol(tmp_path):
    # The producer writes its measurement protocol into every rank artifact, so an artifact without it is not the
    # producer's, although the native validator accepts it.
    cell = glm_cell("prefill")
    write_glm_rank(_pod(tmp_path, cell), glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED]))
    _validate(tmp_path, cell)
    with pytest.raises(ValueError) as error:
        database.aggregate_cell(_glm_plan(), cell, tmp_path, expected_attempt_id=ATTEMPT_ID)
    assert "measurement_protocol of a GLM-5.3-Flash producer artifact must be present" in str(error.value)


@pytest.mark.parametrize(
    ("stamps", "regime", "published"),
    [
        # The native row reports real_prefix, and publication labels the same prefix real_kv.
        pytest.param(["prefill_real_seed"], "real_prefix", "real_kv", id="real-seed"),
        pytest.param(["prefill_fake_prefix"], "fake_prefix", None, id="fake-prefix"),
    ],
)
def test_execution_identity_cached_prefill_accepts_real_prefix_stamp(tmp_path, monkeypatch, stamps, regime, published):
    cell = glm_cell("prefill")
    write_glm_rank(_pod(tmp_path, cell), glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED]))
    valid = _validate(tmp_path, cell)
    cold, cached = valid.points
    collection = replace(valid, points=(cold, _restamp(cached, stamps, regime)))
    plan = _glm_plan()
    if published is None:
        with pytest.raises(ValueError) as error:
            _publish(monkeypatch, tmp_path, cell, collection, plan)
        assert str(error.value) == REAL_KV_REFUSAL.format(regime)
    else:
        rows = _publish(monkeypatch, tmp_path, cell, collection, plan)
        assert [row["kv_seed_regime"] for row in rows] == ["n/a", published]


@pytest.mark.parametrize("kvwarm", [None, WARMUP_SKIP], ids=["no-warmup-block", "warmup-skip"])
def test_cached_prefill_regime_follows_its_stamp_not_the_decode_warmup_block(tmp_path, monkeypatch, kvwarm):
    cell = glm_cell("prefill")
    write_glm_rank(_pod(tmp_path, cell), glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED]))
    collection = replace(_validate(tmp_path, cell), kvwarm_meta=kvwarm)
    rows = _publish(monkeypatch, tmp_path, cell, collection)
    assert [row["kv_seed_regime"] for row in rows] == ["n/a", "real_kv"]


def test_cached_prefill_with_the_real_kv_stamp_follows_the_warmup_block(tmp_path, monkeypatch):
    """No adapter writes both stamps. A point with the real-KV stamp is classified by the warm-up block, as before."""
    cell = glm_cell("prefill")
    write_glm_rank(_pod(tmp_path, cell), glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED]))
    valid = _validate(tmp_path, cell)
    cold, cached = valid.points
    both = _restamp(cached, ["prefill_real_seed", "kvwarm_real_kv"], "real_prefix")
    collection = replace(valid, points=(cold, both), kvwarm_meta=WARMUP_SKIP)
    with pytest.raises(ValueError) as error:
        _publish(monkeypatch, tmp_path, cell, collection)
    assert str(error.value) == REAL_KV_REFUSAL.format("skip:moe_tp_balanced_by_construction")


def test_glm_identity_decode_without_real_kv_is_refused(tmp_path, monkeypatch):
    cell = glm_cell("decode")
    write_glm_rank(_pod(tmp_path, cell), glm_native_payload("decode", points=DECODE_POINTS))
    valid = _validate(tmp_path, cell)
    fallback = tuple(_restamp(measurement, ["kvwarm_fake_fallback"], "fake_fallback") for measurement in valid.points)
    skipped = replace(valid, kvwarm_meta=WARMUP_SKIP, points=fallback)
    with pytest.raises(ValueError) as error:
        _publish(monkeypatch, tmp_path, cell, skipped)
    assert str(error.value) == REAL_KV_REFUSAL.format("skip:moe_tp_balanced_by_construction")


def test_glm_decode_artifact_with_a_warmup_skip_is_refused_at_validation(tmp_path):
    cell = glm_cell("decode")
    fallback_points = [{**point, "sample_reasons": ["explicit", "kvwarm_fake_fallback"]} for point in DECODE_POINTS]
    payload = glm_native_payload("decode", points=fallback_points)
    payload["kvwarm"] = dict(WARMUP_SKIP)
    for row in payload["results"]:
        row["kv_seed_regime"] = "fake_fallback"
    write_glm_rank(_pod(tmp_path, cell), payload)
    with pytest.raises(ValueError, match="native GLM-5.3-Flash result lacks valid kvwarm"):
        database.aggregate_cell(_glm_plan(), cell, tmp_path, expected_attempt_id=ATTEMPT_ID)


def test_published_glm_pair_loads_through_the_rust_fpm_loader(tmp_path):
    """The loader refuses a cached prefill or decode row of a cell with an execution identity unless it is real_kv."""
    import aisim_core
    from aisim_core.sdk.rust_engine_step import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    plan = _glm_plan()
    plan.backend_version = "glm-published-pair"
    rows = []
    shape = _glm_producer_shape(tmp_path)
    for phase, points in (("prefill", [PREFILL_COLD, PREFILL_CACHED]), ("decode", DECODE_POINTS)):
        # The loader selects cells by the quantization that the SDK resolves for the checkpoint.
        cell = replace(
            glm_cell(phase),
            weight_quantization="fp8_block",
            gemm_quant_mode="fp8_block",
            moe_quant_mode="fp8_block",
            fmha_quant_mode="bfloat16",
        )
        payload = glm_native_payload(phase, points=points)
        shape(phase, payload)
        write_glm_rank(_pod(tmp_path / phase, cell), payload)
        rows += database.aggregate_cell(plan, cell, tmp_path / phase, expected_attempt_id=ATTEMPT_ID)
    systems = tmp_path / "systems"
    _parquet, metadata, _skipped = database.write_formal_database(plan, rows, systems_root=systems / "data")
    assert {
        (
            row["measurement_policy"],
            row["global_warmup_iterations"],
            row["warmup_repeats"],
            row["measurement_repeats"],
            row["state_protocol"],
            row["timing_boundary"],
        )
        for row in rows
    } == {
        (
            "vllm_native_fpm_prefix_seed_median_v1",
            0,
            5,
            10,
            "glm53flash_prefix_cache_real_seed_v1",
            "dynamo_vllm_instrumented_scheduler_fpm_wall_time",
        )
    }
    sidecar = json.loads(metadata.read_text())
    assert (sidecar["schema_version"], sidecar["measurement_policy"]) == (7, "vllm_native_fpm_prefix_seed_median_v1")
    assert (sidecar["warmup_repeats"], sidecar["measurement_repeats"]) == (5, 10)
    system_file = f"{plan.system}.yaml"
    shutil.copyfile(Path(aisim_core.__file__).parent / "systems" / system_file, systems / system_file)
    engine = dict(
        schema_version=1,
        model_name=GLM_MODEL,
        system_name=plan.system,
        backend=plan.backend,
        backend_version=plan.backend_version,
        systems_path=str(systems),
        enable_shared_layer=False,
        strict_provenance=True,
        tp_size=1,
        pp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        attention_dp_size=1,
        database_mode="SILICON",
        forward_model="fpm",
    )
    model = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig.from_legacy_engine_config(engine, "aggregated")
    )
    # Each published point is a measured coordinate, so the model returns its latency.
    for row in rows:
        decode = row["workload_kind"] == "decode"
        scheduled = dict(
            num_prefill_requests=0 if decode else row["batch_size"],
            num_decode_requests=row["batch_size"] if decode else 0,
            sum_prefill_tokens=row["total_prefill_tokens"],
            sum_prefill_kv_tokens=0 if decode else row["total_kv_read_tokens"],
            sum_decode_kv_tokens=row["total_kv_read_tokens"] if decode else 0,
            var_prefill_length=0.0,
            var_decode_kv_tokens=0.0,
        )
        fpm = dict(version=1, wall_time=1.0, scheduled_requests=scheduled)
        assert model.estimate_forward_pass_time_ms(fpm) == pytest.approx(row["latency_ms"])


@pytest.mark.parametrize("mode", [None, "eager"], ids=["no-execution-mode", "eager"])
def test_rows_of_other_models_keep_the_single_sample_policy(tmp_path, monkeypatch, mode):
    # A cell without an execution identity has no execution mode, and a DeepSeek V4.1 cell has the audited mode eager.
    # Only the mode of the GLM-5.3-Flash producer takes the median label.
    cell = glm_cell("prefill")
    write_glm_rank(_pod(tmp_path, cell), glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED]))
    valid = _validate(tmp_path, cell)
    rows = _publish(monkeypatch, tmp_path, cell, replace(valid, execution_mode=mode))
    assert len(rows) == 2
    for row in rows:
        assert (row["measurement_policy"], row["warmup_repeats"], row["measurement_repeats"]) == (
            database.SINGLE_SAMPLE_MEASUREMENT_POLICY,
            0,
            1,
        )
        assert not {"state_protocol", "timing_boundary"} & row.keys()


def _producer_cells(tmp_path: Path) -> dict[str, tuple[FPMCell, Path]]:
    """The prefill and the decode cell of GLM-5.3-Flash, each with an artifact in the shape that the producer writes."""
    shape = _glm_producer_shape(tmp_path)
    cells = {}
    for phase, points in (("prefill", [PREFILL_COLD, PREFILL_CACHED]), ("decode", DECODE_POINTS)):
        cell = glm_cell(phase)
        payload = glm_native_payload(phase, points=points)
        shape(phase, payload)
        write_glm_rank(_pod(tmp_path / phase, cell), payload)
        cells[phase] = (cell, tmp_path / phase)
    return cells


@pytest.mark.parametrize("warmup", [1, 5, -1, False, 0.0], ids=["one", "five", "negative", "boolean", "zero-float"])
def test_glm_cell_under_a_plan_with_a_global_warmup_is_refused_before_a_pair_is_written(tmp_path, warmup):
    # The loader of a published pair reads a median row only with a global warm-up of the integer 0, so a median row of
    # this plan would make the whole pair unreadable. The single-sample label of a pair sealed with it has no such rule.
    plan = _glm_plan()
    plan.options = SimpleNamespace(warmup_iterations=warmup)
    cell, cell_dir = _producer_cells(tmp_path)["prefill"]
    with pytest.raises(ValueError) as error:
        database.aggregate_cell(plan, cell, cell_dir, expected_attempt_id=ATTEMPT_ID)
    assert str(error.value) == (
        f"{cell.cell_id} is a GLM-5.3-Flash producer cell under a plan with warmup_iterations={warmup!r}, and its rows "
        "carry measurement_policy='vllm_native_fpm_prefix_seed_median_v1', which requires a global warm-up of the "
        "integer 0"
    )
    rows = database.aggregate_cell(plan, cell, cell_dir, expected_attempt_id=ATTEMPT_ID, median_policy=False)
    assert len(rows) == 2
    assert [(row["measurement_policy"], repr(row["global_warmup_iterations"])) for row in rows] == [
        (database.SINGLE_SAMPLE_MEASUREMENT_POLICY, repr(warmup))
    ] * len(rows)


def _powered_publication(monkeypatch, tmp_path, *, mode="native_graph_policy", measure_power=True):
    """A cell whose validated collection holds power-window artifacts of ``mode``, and a plan with ``measure_power``."""
    cell = glm_cell("prefill")
    write_glm_rank(_pod(tmp_path, cell), glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED]))
    valid = _validate(tmp_path, cell)
    assert valid.execution_mode == "native_graph_policy"
    powered = replace(
        valid,
        execution_mode=mode,
        artifact_schema_version=database.FPM_NATIVE_BENCHMARK_POWER_RESULT_SCHEMA_VERSION,
        points=tuple(replace(point, power_w=100.0) for point in valid.points),
    )
    plan = _glm_plan()
    plan.options = SimpleNamespace(warmup_iterations=0, measure_power=measure_power)
    monkeypatch.setattr(database, "validate_native_collection", lambda *_args, **_kwargs: powered)
    monkeypatch.setattr(database, "extract_measurement_evidence", lambda *_args, **_kwargs: {})
    return plan, cell


@pytest.mark.parametrize("median_policy", [True, False], ids=["median-label", "single-sample-label"])
def test_glm_cell_with_power_window_artifacts_is_refused(tmp_path, monkeypatch, median_policy):
    # The median label holds no power, and the power-window label names a single sample, so no label is true for the
    # rows of a producer cell with a power window.
    plan, cell = _powered_publication(monkeypatch, tmp_path)
    with pytest.raises(ValueError) as error:
        database.aggregate_cell(plan, cell, tmp_path, expected_attempt_id=ATTEMPT_ID, median_policy=median_policy)
    assert str(error.value) == (
        f"{cell.cell_id} wrote power-window artifacts for a GLM-5.3-Flash producer cell, and no FPM measurement "
        "policy describes the rows of such a cell with a power window"
    )


@pytest.mark.parametrize("median_policy", [True, False], ids=["median-label", "single-sample-label"])
def test_glm_cell_with_power_window_artifacts_under_a_plan_without_measure_power_keeps_that_refusal(
    tmp_path, monkeypatch, median_policy
):
    # A producer cell with power-window artifacts under a plan without measure_power holds two faults. The refusal of
    # the plan runs first, so the cell reads its text.
    plan, cell = _powered_publication(monkeypatch, tmp_path, measure_power=False)
    with pytest.raises(ValueError) as error:
        database.aggregate_cell(plan, cell, tmp_path, expected_attempt_id=ATTEMPT_ID, median_policy=median_policy)
    assert str(error.value) == f"{cell.cell_id} wrote power-window artifacts under a plan without measure_power"


@pytest.mark.parametrize("mode", [None, "eager"], ids=["no-execution-mode", "eager"])
def test_powered_cell_of_another_mode_keeps_the_power_window_label(tmp_path, monkeypatch, mode):
    # Only the mode of the GLM-5.3-Flash producer is refused with power-window artifacts. A cell without an execution
    # mode and a cell of the audited mode eager keep the power-window label and its power.
    plan, cell = _powered_publication(monkeypatch, tmp_path, mode=mode)
    rows = database.aggregate_cell(plan, cell, tmp_path, expected_attempt_id=ATTEMPT_ID)
    assert len(rows) == 2
    assert {(row["measurement_policy"], row["power_w"]) for row in rows} == {
        (database.POWER_WINDOW_MEASUREMENT_POLICY, 100.0)
    }


@pytest.mark.parametrize("order", ["median-first", "single-first"])
def test_glm_rows_and_single_sample_rows_cannot_share_a_database(tmp_path, order):
    plan = _glm_plan()
    (prefill, prefill_dir), (decode, decode_dir) = _producer_cells(tmp_path).values()
    median = database.aggregate_cell(plan, prefill, prefill_dir, expected_attempt_id=ATTEMPT_ID)
    single = database.aggregate_cell(plan, decode, decode_dir, expected_attempt_id=ATTEMPT_ID, median_policy=False)
    first, second = (median, single) if order == "median-first" else (single, median)
    systems = tmp_path / "systems"
    parquet, metadata, _skipped = database.write_formal_database(plan, first, systems_root=systems)
    sealed = (parquet.read_bytes(), metadata.read_bytes())
    with pytest.raises(ValueError) as error:
        database.write_formal_database(plan, second, systems_root=systems)
    assert str(error.value) == (
        "cannot mix GLM-5.3-Flash producer rows and other FPM rows in one database: the sidecar holds one "
        "measurement_policy, and 'vllm_native_fpm_prefix_seed_median_v1' binds every row. Publish to a clean "
        f"destination: {parquet}"
    )
    assert (parquet.read_bytes(), metadata.read_bytes()) == sealed


@pytest.mark.parametrize("order", ["median-first", "power-first"])
def test_glm_rows_and_power_window_rows_read_the_power_window_refusal(tmp_path, order):
    # A pair cannot hold power-window rows and median rows. The power-window refusal runs first, so it names the fault.
    plan = _glm_plan()
    (prefill, prefill_dir), (decode, decode_dir) = _producer_cells(tmp_path).values()
    median = database.aggregate_cell(plan, prefill, prefill_dir, expected_attempt_id=ATTEMPT_ID)
    single = database.aggregate_cell(plan, decode, decode_dir, expected_attempt_id=ATTEMPT_ID, median_policy=False)
    power = [
        {**row, "measurement_policy": database.POWER_WINDOW_MEASUREMENT_POLICY, "power_w": 120.0} for row in single
    ]
    first, second = (median, power) if order == "median-first" else (power, median)
    systems = tmp_path / "systems"
    parquet, metadata, _skipped = database.write_formal_database(plan, first, systems_root=systems)
    sealed = (parquet.read_bytes(), metadata.read_bytes())
    with pytest.raises(ValueError) as error:
        database.write_formal_database(plan, second, systems_root=systems)
    assert str(error.value) == (
        "cannot mix power-window and latency-only FPM rows in one database: schema 8 requires every row to carry "
        "measurement_policy='dynamo_native_single_sample_power_window_v1'. Publish to a clean destination, or collect "
        f"every cell with the same --fpm-measure-power setting: {parquet}"
    )
    assert (parquet.read_bytes(), metadata.read_bytes()) == sealed


def test_glm_rows_cannot_join_rows_without_a_measurement_policy(tmp_path):
    plan = _glm_plan()
    (prefill, prefill_dir), (decode, decode_dir) = _producer_cells(tmp_path).values()
    median = database.aggregate_cell(plan, prefill, prefill_dir, expected_attempt_id=ATTEMPT_ID)
    single = database.aggregate_cell(plan, decode, decode_dir, expected_attempt_id=ATTEMPT_ID, median_policy=False)
    bare = [{key: value for key, value in row.items() if key != "measurement_policy"} for row in single]
    systems = tmp_path / "systems"
    parquet, metadata, _skipped = database.write_formal_database(plan, bare, systems_root=systems)
    sealed = (parquet.read_bytes(), metadata.read_bytes())
    with pytest.raises(ValueError) as error:
        database.write_formal_database(plan, median, systems_root=systems)
    assert str(error.value) == (
        "cannot mix GLM-5.3-Flash producer rows and other FPM rows in one database: the sidecar holds one "
        "measurement_policy, and 'vllm_native_fpm_prefix_seed_median_v1' binds every row. Publish to a clean "
        f"destination: {parquet}"
    )
    assert (parquet.read_bytes(), metadata.read_bytes()) == sealed


def test_only_verified_old_single_sample_rows_can_select_the_historical_policy():
    cell = glm_cell("prefill")
    old = replace(cell, glm_producer_digests=None)
    hashes = {cell.cell_id: "a" * 64}
    metadata = {"measurement_policy": database.SINGLE_SAMPLE_MEASUREMENT_POLICY, "cell_rows": hashes}
    assert not database.published_median_policy(metadata, old, hashes)
    assert database.published_median_policy(metadata, cell, hashes)
    assert database.published_median_policy(metadata, old, None)
    assert database.published_median_policy(metadata, old, {cell.cell_id: "b" * 64})
    metadata["measurement_policy"] = database.VLLM_PREFIX_SEED_MEDIAN_POLICY
    assert database.published_median_policy(metadata, old, hashes)


@pytest.mark.parametrize(
    "field,value",
    [
        ("warmup_repeats", 5.0),
        ("warmup_repeats", True),
        ("measurement_repeats", 9),
        ("measurement_repeats", None),
        ("measurement_policy", "sglang_native_fpm_radix_seed_median_v1"),
        ("measurement_policy", "unknown"),
        ("backend", "sglang"),
        ("schema_version", 6),
        ("schema_version", 7.0),
    ],
)
def test_invalid_median_sidecar_cannot_be_reopened_or_republished(tmp_path, field, value):
    plan = _glm_plan()
    (prefill, prefill_dir), (decode, decode_dir) = _producer_cells(tmp_path).values()
    plan.cells = (prefill,)
    rows = database.aggregate_cell(plan, prefill, prefill_dir, expected_attempt_id=ATTEMPT_ID)
    incoming = database.aggregate_cell(plan, decode, decode_dir, expected_attempt_id=ATTEMPT_ID)
    systems = tmp_path / "systems"
    parquet, metadata, _ = database.write_formal_database(plan, rows, systems_root=systems)
    sidecar = json.loads(metadata.read_text())
    if value is None:
        del sidecar[field]
    else:
        sidecar[field] = value
    metadata.write_text(json.dumps(sidecar))
    before = (parquet.read_bytes(), metadata.read_bytes())
    with pytest.raises(ValueError, match="median|frozen plan|historical schema-6"):
        database.validate_formal_database_commit(parquet, metadata, plan)
    with pytest.raises(ValueError, match="median"):
        database.write_formal_database(plan, incoming, systems_root=systems)
    assert (parquet.read_bytes(), metadata.read_bytes()) == before


def test_sealed_sglang_median_publication_keeps_its_native_policy(tmp_path):
    from collector.fpm_forward import sglang_native_fpm

    plan = _glm_plan()
    (prefill, prefill_dir), _ = _producer_cells(tmp_path).values()
    plan.cells = (prefill,)
    rows = database.aggregate_cell(plan, prefill, prefill_dir, expected_attempt_id=ATTEMPT_ID)
    plan.backend = "sglang"
    for row in rows:
        row.update(
            backend="sglang",
            measurement_policy="sglang_native_fpm_radix_seed_median_v1",
            state_protocol=sglang_native_fpm.STATE_PROTOCOL,
            timing_boundary=sglang_native_fpm.TIMING_BOUNDARY,
        )
    parquet, metadata, _ = database.write_formal_database(plan, rows, systems_root=tmp_path / "systems")
    assert json.loads(metadata.read_text())["measurement_policy"] == "sglang_native_fpm_radix_seed_median_v1"
    database.validate_formal_database_commit(parquet, metadata, plan)


@pytest.mark.parametrize(
    "field,value",
    [
        ("global_warmup_iterations", False),
        ("warmup_repeats", 4),
        ("measurement_repeats", 10.0),
        ("state_protocol", "other-producer"),
        ("timing_boundary", "other-timer"),
        ("kv_seed_regime", "fake_fallback"),
    ],
)
def test_median_writer_rejects_rows_that_violate_the_native_contract(tmp_path, field, value):
    plan = _glm_plan()
    (prefill, prefill_dir), _ = _producer_cells(tmp_path).values()
    rows = database.aggregate_cell(plan, prefill, prefill_dir, expected_attempt_id=ATTEMPT_ID)
    for row in rows:
        row[field] = value
    destination = tmp_path / "systems"
    with pytest.raises(ValueError, match="FPM median row"):
        database.write_formal_database(plan, rows, systems_root=destination)
    assert not list(destination.rglob("*.parquet"))


@pytest.mark.parametrize("case", ["single-sample-label", "median-label"])
def test_publication_verifies_with_its_frozen_cells_and_sealed_checkpoint_rows(tmp_path, case):
    import pyarrow.parquet as pq

    plan = _glm_plan()
    cells = _producer_cells(tmp_path)
    median = case == "median-label"
    if not median:
        cells = {
            phase: (replace(cell, glm_producer_digests=None), directory) for phase, (cell, directory) in cells.items()
        }
    rows = [
        row
        for cell, cell_dir in cells.values()
        for row in database.aggregate_cell(plan, cell, cell_dir, expected_attempt_id=ATTEMPT_ID, median_policy=median)
    ]
    parquet, metadata, _skipped = database.write_formal_database(plan, rows, systems_root=tmp_path / "systems")
    # The runner saves these hashes independently in the terminal collection checkpoint.
    published = database.validate_formal_database_commit(
        parquet, metadata, plan, required_cells=tuple(cell for cell, _cell_dir in cells.values())
    )
    checkpoint_rows = published["cell_rows"]
    again = [
        row
        for cell, cell_dir in cells.values()
        for row in database.aggregate_cell(
            plan,
            cell,
            cell_dir,
            expected_attempt_id=ATTEMPT_ID,
            median_policy=database.published_median_policy(published, cell, checkpoint_rows),
        )
    ]
    canonical = lambda row: json.dumps(row, sort_keys=True, allow_nan=False)
    assert sorted(map(canonical, again)) == sorted(map(canonical, pq.read_table(parquet).to_pylist()))


def test_publication_and_readiness_aggregate_with_the_label_of_the_producer_contract():
    """Publication and readiness aggregate with the label of the producer contract. Finalization and collection
    validation consult the frozen cell and checkpoint; tests/unit/test_onboard_runtime.py exercises both readers."""
    import ast

    import collector.fpm_forward

    import aisim.support

    sources = {
        "collection_readiness": Path(aisim.support.__file__).parent / "collection_readiness.py",
        "runner": Path(collector.fpm_forward.__file__).parent / "runner.py",
    }
    keywords = {}
    for name, path in sources.items():
        calls = [
            node
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "aggregate_cell"
        ]
        keywords[name] = [
            [ast.unparse(keyword.value) for keyword in call.keywords if keyword.arg == "median_policy"]
            for call in calls
        ]
    assert keywords == {
        "collection_readiness": [[]],
        "runner": [[]],
    }


def test_glm_producer_record_must_name_the_supported_vllm(tmp_path):
    # vLLM 0.33.1 is another runtime, although its version shares the release line of the supported one.
    cell = glm_cell("prefill")
    payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
    payload["producer"]["vllm_package_version"] = "0.33.1"
    write_glm_rank(_pod(tmp_path, cell), payload)
    with pytest.raises(
        ValueError,
        match=r"^native GLM-5\.3-Flash result lacks valid producer\.vllm_package_version: expected '0\.33\.0', "
        r"found '0\.33\.1': .+benchmark\.json$",
    ):
        _validate(tmp_path, cell)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instrumentation_revision", "0" * 40),
        ("runtime_scheduler_revision", "0" * 40),
        ("runtime_distribution", "ai-dynamo 1.6.0.dev20261010"),
        ("real_seed_port_source", "ai-dynamo/dynamo@main instrumented_scheduler._bench_realseed_*"),
        ("queued_metrics_port_source", "ai-dynamo/dynamo@main instrumented_scheduler._compute_queued"),
        ("vllm_package_version", "0.33.0+local"),
        ("reviewed_scheduler_api_revision", "0" * 40),
        ("runtime_source_manifest_sha256", "0" * 64),
        ("overlay_sha256", "0" * 64),
        # The same number in another JSON type is another record.
        ("context_policy_version", 1.0),
        ("warmup_repeats", 4),
        ("measurement_repeats", 9),
        ("measurement_repeats", None),
    ],
)
def test_glm_producer_record_must_be_the_supported_producers(tmp_path, field, value):
    cell = glm_cell("prefill")
    payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
    expected = payload["producer"][field]
    if value is None:
        del payload["producer"][field]
    else:
        payload["producer"][field] = value
    write_glm_rank(_pod(tmp_path, cell), payload)
    with pytest.raises(ValueError) as refusal:
        _validate(tmp_path, cell)
    assert re.fullmatch(
        rf"native GLM-5\.3-Flash result lacks valid producer\.{field}: expected {re.escape(repr(expected))}, "
        rf"found {re.escape(repr(value))}: .+benchmark\.json",
        str(refusal.value),
    ), str(refusal.value)


def test_glm_producer_record_must_not_add_a_field(tmp_path):
    cell = glm_cell("prefill")
    payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
    payload["producer"]["patched"] = True
    write_glm_rank(_pod(tmp_path, cell), payload)
    with pytest.raises(
        ValueError,
        match=r"^native GLM-5\.3-Flash result lacks valid producer\.patched: expected no such field, found True: "
        r".+benchmark\.json$",
    ):
        _validate(tmp_path, cell)


def test_glm_producer_record_digests_are_those_of_the_shipped_producer_files():
    record = native_artifact._glm_producer_record()
    for field, name in (
        ("runtime_source_manifest_sha256", "runtime-source-sha256.json"),
        ("overlay_sha256", "glm53flash_prefix_scheduler.py"),
    ):
        assert record[field] == hashlib.sha256((PRODUCER_DIR / name).read_bytes()).hexdigest()
    assert record["vllm_package_version"] == "0.33.0"


def _receipt(cell_dir: Path, cell: FPMCell, **changes) -> Path:
    """Write a producer receipt for the test cell, with ``changes`` applied to its top-level keys."""
    receipt = {
        "schema_name": "aic_fpm_producer_receipt",
        "schema_version": 1,
        "plan_sha256": PLAN_SHA256,
        "cell_id": cell.cell_id,
        "attempt_id": ATTEMPT_ID,
        "producer": {"runtime_source_manifest_sha256": "a" * 64, "overlay_sha256": "b" * 64},
        **changes,
    }
    path = cell_dir / "producer-receipt.json"
    path.write_text(json.dumps(receipt))
    return path


def _cell_of_another_producer_revision(tmp_path: Path) -> FPMCell:
    """A prefill cell whose artifact names the producer digests "a" * 64 and "b" * 64, not those of this checkout."""
    cell = replace(glm_cell("prefill"), glm_producer_digests=GlmProducerDigests("a" * 64, "b" * 64))
    payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
    payload["producer"].update(runtime_source_manifest_sha256="a" * 64, overlay_sha256="b" * 64)
    write_glm_rank(_pod(tmp_path, cell), payload)
    return cell


def test_reader_without_a_receipt_keeps_the_rule_of_the_reading_checkout(tmp_path):
    cell = _cell_of_another_producer_revision(tmp_path)
    with pytest.raises(ValueError) as refusal:
        _validate(tmp_path, cell)
    text = str(refusal.value)
    assert text.startswith(
        "native GLM-5.3-Flash result lacks valid producer.runtime_source_manifest_sha256: expected '"
    ), text
    assert "found 'aaaa" in text


def test_reader_takes_the_producer_digests_of_a_receipt(tmp_path):
    cell = _cell_of_another_producer_revision(tmp_path)
    _receipt(tmp_path, cell)
    assert len(_validate(tmp_path, cell).points) == 2


@pytest.mark.parametrize(
    ("change", "name", "key", "expected"),
    [
        ({"plan_sha256": "0" * 64}, "plan", "plan_sha256", PLAN_SHA256),
        ({"cell_id": "fpm-other"}, "cell", "cell_id", None),
        ({"attempt_id": "a2"}, "attempt", "attempt_id", ATTEMPT_ID),
        # A receipt of another plan, cell and attempt names the plan, the first binding that the reader compares.
        ({"plan_sha256": "0" * 64, "cell_id": "fpm-other", "attempt_id": "a2"}, "plan", "plan_sha256", PLAN_SHA256),
    ],
    ids=["plan", "cell", "attempt", "plan-cell-and-attempt"],
)
def test_receipt_of_another_plan_cell_or_attempt_is_refused(tmp_path, change, name, key, expected):
    cell = _cell_of_another_producer_revision(tmp_path)
    path = _receipt(tmp_path, cell, **change)
    expected = cell.cell_id if expected is None else expected
    with pytest.raises(ValueError) as refusal:
        _validate(tmp_path, cell)
    assert str(refusal.value) == (
        f"producer receipt names another {name}: {key} actual={change[key]!r} expected={expected!r}: {path}"
    )


_RECEIPT_KEYS = "['attempt_id', 'cell_id', 'plan_sha256', 'producer', 'schema_name', 'schema_version']"
_RECEIPT_OBJECT = f"it must be a JSON object with exactly the keys {_RECEIPT_KEYS}"
_RECEIPT_VERSION = "schema_version must be the integer 1"
_RECEIPT_PRODUCER = (
    "producer must be a JSON object with exactly the keys ['overlay_sha256', 'runtime_source_manifest_sha256']"
)
_RECEIPT_LINK = "it must be a regular file, not a symbolic link"
_RECEIPT_TEXT = "it must be JSON text in UTF-8"


def _write_receipt_text(text: str):
    def write(cell_dir: Path, _cell: FPMCell) -> None:
        (cell_dir / "producer-receipt.json").write_text(text)

    return write


def _write_receipt_bytes(data: bytes):
    def write(cell_dir: Path, _cell: FPMCell) -> None:
        (cell_dir / "producer-receipt.json").write_bytes(data)

    return write


def _write_receipt_without(key: str):
    def write(cell_dir: Path, cell: FPMCell) -> None:
        path = _receipt(cell_dir, cell)
        receipt = json.loads(path.read_text())
        del receipt[key]
        path.write_text(json.dumps(receipt))

    return write


def _write_receipt_with(**changes):
    def write(cell_dir: Path, cell: FPMCell) -> None:
        _receipt(cell_dir, cell, **changes)

    return write


def _write_receipt_symlink(cell_dir: Path, cell: FPMCell) -> None:
    """A symbolic link at the receipt path to a valid receipt in a folder beside the cell folder."""
    elsewhere = cell_dir.parent / "elsewhere"
    elsewhere.mkdir()
    target = _receipt(elsewhere, cell)
    (cell_dir / "producer-receipt.json").symlink_to(target)


def _write_receipt_dangling_symlink(cell_dir: Path, _cell: FPMCell) -> None:
    (cell_dir / "producer-receipt.json").symlink_to(cell_dir.parent / "missing.json")


def _write_receipt_directory(cell_dir: Path, _cell: FPMCell) -> None:
    (cell_dir / "producer-receipt.json").mkdir()


@pytest.mark.parametrize(
    ("write", "rule"),
    [
        (_write_receipt_text("[]"), _RECEIPT_OBJECT),
        (_write_receipt_text("null"), _RECEIPT_OBJECT),
        (_write_receipt_text("5"), _RECEIPT_OBJECT),
        (_write_receipt_text("{"), _RECEIPT_TEXT),
        (_write_receipt_bytes(b"\xff\xfe{}"), _RECEIPT_TEXT),
        (_write_receipt_without("attempt_id"), _RECEIPT_OBJECT),
        (_write_receipt_with(note="added"), _RECEIPT_OBJECT),
        (_write_receipt_with(schema_version="1"), _RECEIPT_VERSION),
        (_write_receipt_with(schema_version=True), _RECEIPT_VERSION),
        (_write_receipt_with(schema_version=2), _RECEIPT_VERSION),
        (_write_receipt_with(schema_version=1.0), _RECEIPT_VERSION),
        (_write_receipt_with(schema_name="other"), "schema_name must be 'aic_fpm_producer_receipt'"),
        (
            _write_receipt_with(producer={"runtime_source_manifest_sha256": "a" * 63, "overlay_sha256": "b" * 64}),
            "producer.runtime_source_manifest_sha256 must be 64 lowercase hexadecimal characters",
        ),
        (
            _write_receipt_with(producer={"runtime_source_manifest_sha256": "A" * 64, "overlay_sha256": "b" * 64}),
            "producer.runtime_source_manifest_sha256 must be 64 lowercase hexadecimal characters",
        ),
        (
            _write_receipt_with(
                producer={"runtime_source_manifest_sha256": "a" * 64, "overlay_sha256": "b" * 64, "note": "added"}
            ),
            _RECEIPT_PRODUCER,
        ),
        (_write_receipt_with(producer=[]), _RECEIPT_PRODUCER),
        (_write_receipt_with(producer=5), _RECEIPT_PRODUCER),
        (_write_receipt_symlink, _RECEIPT_LINK),
        (_write_receipt_dangling_symlink, _RECEIPT_LINK),
        (_write_receipt_directory, "it must be a regular file"),
    ],
    ids=[
        "list",
        "null",
        "number",
        "no-json",
        "no-utf8",
        "no-attempt-id",
        "added-key",
        "version-string",
        "version-bool",
        "version-2",
        "version-float",
        "other-schema",
        "short-digest",
        "upper-case-digest",
        "producer-added-key",
        "producer-list",
        "producer-number",
        "symlink",
        "dangling-symlink",
        "directory",
    ],
)
def test_malformed_receipt_is_refused(tmp_path, write, rule):
    cell_dir = tmp_path / "cell"
    cell = _cell_of_another_producer_revision(cell_dir)
    write(cell_dir, cell)
    path = cell_dir / "producer-receipt.json"
    with pytest.raises(ValueError) as refusal:
        _validate(cell_dir, cell)
    assert str(refusal.value) == f"invalid producer receipt: {rule}: {path}"


@pytest.mark.parametrize("legacy_plan", [False, True])
def test_matching_receipt_and_artifact_cannot_replace_the_frozen_producer(tmp_path, legacy_plan):
    cell = _cell_of_another_producer_revision(tmp_path)
    cell = replace(cell, glm_producer_digests=None if legacy_plan else GlmProducerDigests.from_runtime(PRODUCER_DIR))
    cell = cell_from_dict(cell.to_dict())
    _receipt(tmp_path, cell)
    with pytest.raises(ValueError, match="producer digests differ from the producer expected for this collection plan"):
        _validate(tmp_path, cell)


@pytest.mark.parametrize(("mask", "mode"), [(0o022, 0o644), (0o002, 0o664), (0o077, 0o600)])
def test_producer_receipt_respects_the_campaign_umask(tmp_path, mask, mode):
    cell = glm_cell("prefill")

    def write():
        os.umask(mask)
        native_artifact.write_producer_receipt(cell, tmp_path, plan_sha256=PLAN_SHA256, attempt_id=ATTEMPT_ID)

    process = multiprocessing.get_context("fork").Process(target=write)
    process.start()
    process.join()
    assert process.exitcode == 0
    receipt = tmp_path / native_artifact.PRODUCER_RECEIPT_FILENAME
    assert stat.S_IMODE(receipt.stat().st_mode) == mode


def test_concurrent_receipt_writers_use_their_own_file(tmp_path, monkeypatch):
    cell = glm_cell("prefill")
    replace_file = native_artifact.os.replace
    barrier = Barrier(4)
    files = []

    def finish(source, destination):
        files.append(source)
        barrier.wait(timeout=10)
        replace_file(source, destination)

    monkeypatch.setattr(native_artifact.os, "replace", finish)

    def write(_):
        return native_artifact.write_producer_receipt(cell, tmp_path, plan_sha256=PLAN_SHA256, attempt_id=ATTEMPT_ID)

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(write, range(4)))
    assert len(set(files)) == 4
    assert all(path.parent == tmp_path for path in files)
    assert set(paths) == set(tmp_path.iterdir()) == {tmp_path / native_artifact.PRODUCER_RECEIPT_FILENAME}
    assert json.loads(paths[0].read_text())["producer"] == cell.glm_producer_digests._asdict()


def test_failed_receipt_replace_preserves_the_last_receipt_and_cleans_up(tmp_path, monkeypatch):
    cell = glm_cell("prefill")
    receipt = native_artifact.write_producer_receipt(cell, tmp_path, plan_sha256=PLAN_SHA256, attempt_id=ATTEMPT_ID)
    previous = receipt.read_bytes()

    def fail(_src, _dst):
        raise OSError("replacement interrupted")

    monkeypatch.setattr(native_artifact.os, "replace", fail)
    with pytest.raises(OSError, match="replacement interrupted"):
        native_artifact.write_producer_receipt(cell, tmp_path, plan_sha256=PLAN_SHA256, attempt_id="another")
    assert list(tmp_path.iterdir()) == [receipt]
    assert receipt.read_bytes() == previous


def test_receipt_leaves_the_other_producer_fields_to_the_reader(tmp_path):
    cell = replace(glm_cell("prefill"), glm_producer_digests=GlmProducerDigests("a" * 64, "b" * 64))
    payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
    payload["producer"].update(runtime_source_manifest_sha256="a" * 64, overlay_sha256="b" * 64, warmup_repeats=6)
    write_glm_rank(_pod(tmp_path / "warmup", cell), payload)
    _receipt(tmp_path / "warmup", cell)
    with pytest.raises(ValueError) as refusal:
        _validate(tmp_path / "warmup", cell)
    assert "lacks valid producer.warmup_repeats: expected 5, found 6" in str(refusal.value)

    payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
    payload["producer"].update(runtime_source_manifest_sha256="c" * 64, overlay_sha256="c" * 64)
    write_glm_rank(_pod(tmp_path / "digests", cell), payload)
    _receipt(tmp_path / "digests", cell)
    with pytest.raises(ValueError) as refusal:
        _validate(tmp_path / "digests", cell)
    # The expected digest is the one of the receipt, and the text says so.
    assert str(refusal.value).startswith(
        f"native GLM-5.3-Flash result lacks valid producer.runtime_source_manifest_sha256: expected {'a' * 64!r} "
        f"(the value of the producer receipt of the cell), found {'c' * 64!r}: "
    ), str(refusal.value)


def test_receipt_is_not_read_for_a_cell_without_the_producer_contract(tmp_path):
    # The artifact of test_v41_artifacts_keep_their_contract, with a receipt that no reader of the GLM-5.3-Flash
    # producer contract would accept in its cell folder.
    identity = ("c" * 64, "full", "hbm_tp_sharded", "text")
    cell = replace(_cell(phase="prefill", strategy="pure_tp"), execution_identity=identity, input_text_sha256="a" * 64)
    payload = _native_payload(phase="prefill", rank=0, dp=1)
    payload.update(
        execution_identity=dict(zip(EXECUTION_COLUMNS, identity, strict=True)),
        execution_mode="eager",
        input_provenance={
            "source": "tokenizer_text",
            "text_sha256": "a" * 64,
            "token_ids_sha256": "b" * 64,
            "tokenizer_revision": "pinned",
            "token_count": 100,
            "unique_token_count": 20,
        },
    )
    (_pod(tmp_path, cell) / "benchmark.json").write_text(json.dumps(payload))
    (tmp_path / "producer-receipt.json").write_text("[]")
    with pytest.raises(ValueError, match="V4.1 native result lacks token-stream manifest"):
        _validate(tmp_path, cell)


def test_no_receipt_is_written_for_a_cell_without_the_producer_contract(tmp_path):
    written = native_artifact.write_producer_receipt(
        _cell(phase="prefill"), tmp_path, plan_sha256=PLAN_SHA256, attempt_id=ATTEMPT_ID
    )
    assert written is None
    assert not (tmp_path / "producer-receipt.json").exists()


def test_campaign_with_receipts_is_read_after_the_producer_files_change(tmp_path, monkeypatch):
    shape = _glm_producer_shape(tmp_path)
    cell = glm_cell("prefill")
    payload = glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED])
    shape("prefill", payload)
    write_glm_rank(_pod(tmp_path / "prefill", cell), payload)
    receipt = native_artifact.write_producer_receipt(
        cell, tmp_path / "prefill", plan_sha256=PLAN_SHA256, attempt_id=ATTEMPT_ID
    )
    assert receipt == tmp_path / "prefill" / "producer-receipt.json"
    expected = {
        "attempt_id": ATTEMPT_ID,
        "cell_id": cell.cell_id,
        "plan_sha256": PLAN_SHA256,
        "producer": {
            "overlay_sha256": hashlib.sha256(
                (PRODUCER_DIR / "glm53flash_prefix_scheduler.py").read_bytes()
            ).hexdigest(),
            "runtime_source_manifest_sha256": hashlib.sha256(
                (PRODUCER_DIR / "runtime-source-sha256.json").read_bytes()
            ).hexdigest(),
        },
        "schema_name": "aic_fpm_producer_receipt",
        "schema_version": 1,
    }
    # The receipt is the JSON text with an indent of two, sorted keys and a final line break, and the write through a
    # temporary file leaves no other file in the cell folder.
    assert receipt.read_text() == json.dumps(expected, indent=2, sort_keys=True) + "\n"
    assert sorted(item.name for item in (tmp_path / "prefill").iterdir()) == ["producer-receipt.json", "raw"]

    # A later revision of the producer: one more line in the scheduler file of the reading checkout.
    later = tmp_path / "later"
    later.mkdir()
    for name in ("glm53flash_prefix_scheduler.py", "runtime-source-sha256.json"):
        shutil.copyfile(PRODUCER_DIR / name, later / name)
    with (later / "glm53flash_prefix_scheduler.py").open("ab") as scheduler:
        scheduler.write(b"\n# a later revision\n")
    monkeypatch.setattr(native_artifact, "_GLM_RUNTIME", later)
    native_artifact._glm_producer_record.cache_clear()
    try:
        assert len(_validate(tmp_path / "prefill", cell).points) == 2
        rows = database.aggregate_cell(_glm_plan(), cell, tmp_path / "prefill", expected_attempt_id=ATTEMPT_ID)
        assert len(rows) == 2
        receipt.unlink()
        with pytest.raises(ValueError, match=r"lacks valid producer\.overlay_sha256"):
            _validate(tmp_path / "prefill", cell)
    finally:
        native_artifact._glm_producer_record.cache_clear()


def test_glm_producer_vllm_must_be_the_collection_runtime(tmp_path):
    # The producer and the record are those of vLLM 0.33.0, and the Collector provenance binds the launch to 0.33.1.
    cell = glm_cell("prefill")
    write_glm_rank(
        _pod(tmp_path, cell, backend_version="0.33.1"),
        glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED]),
    )
    with pytest.raises(
        ValueError,
        match=r"^native GLM-5\.3-Flash producer\.vllm_package_version differs from the Collector provenance runtime: "
        r"expected '0\.33\.1', found '0\.33\.0': .+benchmark\.json$",
    ):
        _validate(tmp_path, cell)


def test_glm_collection_carries_the_contract_of_its_cell(tmp_path):
    cell = glm_cell("prefill")
    write_glm_rank(_pod(tmp_path, cell), glm_native_payload("prefill", points=[PREFILL_COLD, PREFILL_CACHED]))
    assert _validate(tmp_path, cell).execution_mode == "native_graph_policy"
