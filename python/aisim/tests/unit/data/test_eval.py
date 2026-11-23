# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
from pathlib import Path

import pytest

from aisim_core.data import Artifact, Identity
from aisim_core.data.eval import (
    EvalRecord,
    Exclusion,
    GpuTime,
    PredictionIdentity,
    attest,
    compute_results,
    current_records,
    load_record,
    prediction_fingerprint,
    record_dir,
    record_files,
    record_path,
    verify_record,
    write_record,
    write_truth_table,
)

pytestmark = pytest.mark.unit

SPEC = "a" * 64
PIN = "b" * 64


def _predict(inputs):
    return inputs["tokens"] * 1.1 + (5.0 if inputs["tokens"] == 40 else 0.0)


def _identity():
    return Identity(
        hardware="b200_sxm",
        framework="vllm",
        framework_version="0.11.0",
        operation_family=None,
        model="Qwen/Qwen3-32B",
        effective_settings={},
        unresolved_identity=None,
        parallelism={"aggregated": {"tp": 4, "ep": 1}},
        parallelism_complete=True,
        deployment="aggregated",
    )


@pytest.fixture
def built(tmp_path):
    rows = [(f"s{i}", {"tokens": 10 * i}, 10.0 * i) for i in range(1, 6)] + [("bad", {"tokens": 1}, None)]
    art = write_truth_table(tmp_path / "truth.parquet", rows)
    cases = [(c, t, _predict(i)) for c, i, t in rows]
    results = compute_results(cases)
    pred = PredictionIdentity.build("fpm_interpolation", {"profile": PIN}, SPEC)
    record = EvalRecord.model_validate(
        {
            "schema_version": "aisim-eval/v1",
            "prediction": pred.model_dump(),
            "truth": {
                "layer": "fpm",
                "snapshot_id": None,
                "configuration_ids": ["cfg-" + "c" * 64],
                "identity": _identity().model_dump(),
                "artifacts": [art.model_dump(by_alias=True)],
                "truth_table_artifact_id": "truth-table",
            },
            "results": results.model_dump(),
            "assumptions": [{"name": "kv_dtype", "value": "fp8", "status": "estimated", "kind": "calibration"}],
            "gpu_time": GpuTime(authority="unknown").model_dump(),
            "simulation_cost": {"wall_seconds": 1.5, "gpu_seconds": 0.0},
            "producer": {"kind": "agent", "name": "t", "evaluator_revision": "r1"},
            "tier": "submitted",
            "created_at": "2026-11-07T00:00:00Z",
        }
    )
    return tmp_path, record


def test_fingerprint_ignores_unrelated_and_tracks_spec():
    base = prediction_fingerprint("p", {"a": PIN}, SPEC)
    assert base == prediction_fingerprint("p", {"a": PIN}, SPEC)
    assert base != prediction_fingerprint("p", {"a": PIN}, "d" * 64)
    with pytest.raises(ValueError):
        PredictionIdentity(code_path="p", data_pins={"a": PIN}, spec_sha256=SPEC, fingerprint="e" * 64)


def test_unknown_gpu_time_cannot_be_zero():
    with pytest.raises(ValueError):
        GpuTime(authority="unknown", gpu_seconds=0.0)


def test_exclusion_and_metrics(built):
    _, record = built
    assert record.results.n_total == 6 and record.results.n_scored == 5
    assert [e.case_id for e in record.results.exclusions] == ["bad"]
    assert record.results.coverage == pytest.approx(5 / 6)


@pytest.mark.parametrize("mutate", [False, True])
def test_attestation_requires_the_exact_record_that_was_verified(built, mutate):
    root, record = built
    report = verify_record(record, root, _predict, record.prediction.fingerprint)
    assert report.ok and report.record_id == record.record_id
    assert attest(record, report).verification.verified_record_id == record.record_id
    other = record if mutate else record.model_copy(deep=True)
    other.producer.name = "another producer"
    with pytest.raises(ValueError, match="different record content"):
        attest(other, report)


def test_write_verify_supersede_exit_gate(built):
    root, record = built
    path = write_record(root, record)
    assert path == record_path(root, record) and path.stem == record.record_id
    assert write_record(root, record) == path  # idempotent
    original = path.read_bytes()

    report = verify_record(load_record(path), root, _predict, record.prediction.fingerprint)
    assert report.ok, report.problems

    verified = attest(record, report)
    vpath = write_record(root, verified)
    assert vpath != path and path.read_bytes() == original  # original untouched
    assert verified.supersedes == record.record_id and verified.tier == "verified"
    assert [r.record_id for r in current_records(path.parent)] == [verified.record_id]


def test_verifier_rejects_wrong_claims_and_fingerprint(built):
    root, record = built
    report = verify_record(record, root, lambda i: _predict(i) * 1.5, record.prediction.fingerprint)
    assert not report.ok and any("mape_pct" in p for p in report.problems)
    assert not verify_record(record, root, _predict, "f" * 64).ok
    with pytest.raises(ValueError):
        attest(record, report)


def test_tampered_truth_table_fails(built):
    root, record = built
    (root / "truth.parquet").write_bytes(b"x")
    assert "hash" in verify_record(record, root, _predict, record.prediction.fingerprint).problems[0]


def test_immutability_and_supersede_rules(built):
    root, record = built
    path = write_record(root, record)
    path.write_text(path.read_text().replace("2026-11-07", "2027-01-01"))
    with pytest.raises(ValueError, match="content hash"):
        load_record(path)
    other = EvalRecord.model_validate({**record.model_dump(by_alias=True), "supersedes": "9" * 64})
    with pytest.raises(ValueError, match="not found"):
        write_record(root, other)
    with pytest.raises(ValueError):
        EvalRecord.model_validate({**record.model_dump(by_alias=True), "tier": "verified"})
    with pytest.raises(ValueError):
        EvalRecord.model_validate({**record.model_dump(by_alias=True), "status": "failed"})


def test_eval_uses_gpu_time_module_type():
    from aisim_core.data import gpu_time

    assert GpuTime is gpu_time.GpuTime


def test_record_gpu_time_archival_warns_qualified_errors(built):
    from aisim_core.data.eval import check_record_gpu_time

    _, record = built
    unknown = check_record_gpu_time(record)
    assert [(f.severity, f.code) for f in unknown] == [("warn", "gpu_time.missing")]
    absent = record.model_copy(update={"gpu_time": None})
    assert check_record_gpu_time(absent)[0].severity == "warn"
    doc = record.model_dump(by_alias=True)
    doc["gpu_time"] = None
    doc.update(
        tier="qualified",
        verification={
            "verifier_version": "1",
            "recomputed_fingerprint": record.prediction.fingerprint,
            "verified_record_id": record.record_id,
        },
    )
    with pytest.raises(ValueError, match="qualified record needs gpu_time"):
        EvalRecord.model_validate(doc)
    doc["gpu_time"] = GpuTime(authority="unknown").model_dump(by_alias=True)
    with pytest.raises(ValueError, match="qualified record needs gpu_time"):
        EvalRecord.model_validate(doc)
    doc["gpu_time"] = GpuTime(authority="measured", gpu_count=4, wall_seconds=0.0).model_dump(by_alias=True)
    assert EvalRecord.model_validate(doc).tier == "qualified"  # zero is recorded, not missing


def test_verification_scores_the_bytes_the_digest_covered(built, monkeypatch):
    root, record = built
    path = root / "truth.parquet"
    rows = [(f"s{i}", {"tokens": 10 * i}, 99.0 * i) for i in range(1, 6)] + [("bad", {"tokens": 1}, None)]
    write_truth_table(root / "replacement.parquet", rows)
    replacement = (root / "replacement.parquet").read_bytes()
    real_read_bytes = Path.read_bytes

    def swap_after_the_digest_read(self):
        data = real_read_bytes(self)
        if self == path:
            path.write_bytes(replacement)  # an atomic replace between the digest check and the decode
        return data

    monkeypatch.setattr(Path, "read_bytes", swap_after_the_digest_read)
    report = verify_record(record, root, _predict, record.prediction.fingerprint)
    assert report.ok, report.problems


@pytest.mark.parametrize(
    "model,field",
    [
        ("Qwen/Qwen3-32B", "hardware"),
        ("Qwen/Qwen3-32B", "framework"),
        ("Qwen/Qwen3-32B", "framework_version"),
        ("Qwen/Qwen3-32B", "model"),
        (None, "hardware"),
        (None, "operation_family"),
    ],
)
def test_record_dir_keeps_identity_values_inside_the_eval_root(tmp_path, field, model):
    identity = _identity().model_copy(update={"model": model, "operation_family": "gemm"})
    normal = record_dir(tmp_path, identity)
    assert normal.relative_to(tmp_path / "eval").parts[0] == ("Qwen--Qwen3-32B" if model else "b200_sxm")
    for value in ("../../escape", "a/b"):
        path = record_dir(tmp_path, identity.model_copy(update={field: value}))
        assert path.resolve().is_relative_to((tmp_path / "eval").resolve())
        assert len(path.relative_to(tmp_path / "eval").parts) == len(normal.relative_to(tmp_path / "eval").parts)
    for value in ("..", ".", "a\\b"):
        with pytest.raises(ValueError, match="path component"):
            record_dir(tmp_path, identity.model_copy(update={field: value}))


def test_write_record_refuses_an_identity_that_would_leave_the_eval_root(built):
    root, record = built
    truth = record.truth.model_copy(update={"identity": record.truth.identity.model_copy(update={"hardware": ".."})})
    escaped = record.model_copy(update={"truth": truth})
    with pytest.raises(ValueError, match="path component"):
        write_record(root, escaped)
    assert not (root / "eval").exists()


@pytest.mark.parametrize("inputs_json", [None, "not json", "[1, 2]", "3"])
def test_a_truth_row_without_an_inputs_object_fails_verification(built, inputs_json):
    import hashlib

    import pyarrow as pa

    from aisim_core.data.eval import TRUTH_SCHEMA_ID
    from aisim_core.data.parquet import write_parquet

    root, record = built
    table = pa.table(
        {"case_id": ["s1", "s2"], "inputs_json": [inputs_json, '{"tokens": 20}'], "truth_value": [10.0, 20.0]}
    )
    path = write_parquet(table, root / "truth.parquet", TRUTH_SCHEMA_ID)
    data = path.read_bytes()
    art = record.truth.artifacts[0].model_copy(
        update={"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "rows": 2}
    )
    broken = record.model_copy(update={"truth": record.truth.model_copy(update={"artifacts": [art]})})
    report = verify_record(broken, root, _predict, broken.prediction.fingerprint)
    assert not report.ok and "inputs_json" in report.problems[0]


def test_outlier_truth_is_compared_with_the_truth_table(built):
    root, record = built
    outlier = record.results.worst_n[0]
    forged = record.results.model_copy(
        update={"worst_n": [outlier.model_copy(update={"truth": outlier.truth + 50.0}), *record.results.worst_n[1:]]}
    )
    lying = record.model_copy(update={"results": forged})
    report = verify_record(lying, root, _predict, lying.prediction.fingerprint)
    assert not report.ok and any("worst_n values" in p for p in report.problems)
    assert verify_record(record, root, _predict, record.prediction.fingerprint).ok


def test_verifier_does_not_trust_a_producers_exclusion_of_a_valid_error(built):
    root, record = built
    cases = [(f"s{i}", 10.0 * i, _predict({"tokens": 10 * i})) for i in range(1, 6)] + [("bad", None, 1.1)]
    forged = record.model_copy(update={"results": compute_results(cases, [Exclusion(case_id="s4", reason="skip")])})
    report = verify_record(forged, root, _predict, forged.prediction.fingerprint)
    assert not report.ok
    assert any("exclusions not justified by recomputation" in p for p in report.problems)
    assert report.recomputed.n_scored == 5 and report.recomputed.mape_pct > forged.results.mape_pct
    assert [e.case_id for e in report.recomputed.exclusions] == ["bad"]


def test_attestation_cannot_reuse_another_records_successful_report(built):
    root, record = built
    report = verify_record(record, root, _predict, record.prediction.fingerprint)
    assert report.ok and report.record_id == record.record_id
    other = record.model_copy(update={"created_at": "2026-11-08T00:00:00Z"})
    with pytest.raises(ValueError, match="different record"):
        attest(other, report)
    assert attest(record, report).verification.verified_record_id == record.record_id


def test_nonfinite_truth_and_unavailable_predictions_remain_excluded():
    results = compute_results(
        [
            ("infinite", float("inf"), 1.0),
            ("nan", float("nan"), 1.0),
            ("missing", None, 1.0),
            ("unavailable", 1.0, None),
            ("valid", 1.0, 2.0),
        ]
    )
    assert results.n_scored == 1 and results.mape_pct == 100.0
    assert {e.case_id for e in results.exclusions} == {"infinite", "nan", "missing", "unavailable"}


def test_a_finite_pair_whose_residual_overflows_is_excluded_without_changing_finite_scores():
    baseline = [("exact", 10.0, 10.0), ("ten-percent", 100.0, 110.0)]
    got = compute_results([*baseline, ("overflowing-ratio", 1e-300, 1e300)])
    reference = compute_results(baseline)
    assert got.n_scored == reference.n_scored == 2 and got.mape_pct == reference.mape_pct == 5.0
    assert got.residual_quantiles_pct == reference.residual_quantiles_pct
    assert {e.case_id: e.reason for e in got.exclusions} == {"overflowing-ratio": "non-finite residual"}


@pytest.mark.parametrize("mutate_results", [False, True])
def test_verification_uses_a_private_record_snapshot_before_prediction(built, mutate_results):
    root, record = built
    original = record.model_copy(deep=True)
    replacement = write_truth_table(root / "replacement.parquet", [("different", {"tokens": 1}, 999.0)])

    def mutating_predict(inputs):
        # A predictor may close over and mutate the caller's nested objects.
        record.truth.artifacts[0].path = replacement.path
        record.truth.artifacts[0].sha256 = replacement.sha256
        record.truth.artifacts[0].bytes = replacement.bytes
        if mutate_results:
            record.results.mape_pct = -1.0
        return _predict(inputs)

    report = verify_record(record, root, mutating_predict, record.prediction.fingerprint)
    assert report.ok, report.problems
    assert report.record_id == original.record_id
    assert report.recomputed.mape_pct == original.results.mape_pct
    with pytest.raises(ValueError, match="different record content"):
        attest(record, report)
    assert attest(original, report).verification.verified_record_id == original.record_id


def test_two_promotions_cannot_fork_the_current_record(built):
    root, record = built
    # Synthetic declared cost makes qualification valid without claiming a hardware measurement.
    record = record.model_copy(update={"gpu_time": GpuTime(authority="declared", gpu_count=1, wall_seconds=1.0)})
    write_record(root, record)
    report = verify_record(record, root, _predict, record.prediction.fingerprint)
    verified, qualified = attest(record, report), attest(record, report, "qualified")
    path = write_record(root, verified)
    assert write_record(root, verified) == path
    with pytest.raises(ValueError, match="already has successor"):
        write_record(root, qualified)
    assert current_records(path.parent) == [verified]
    next_report = verify_record(verified, root, _predict, verified.prediction.fingerprint)
    final = attest(verified, next_report, "qualified")
    write_record(root, final)
    assert current_records(path.parent) == [final]


def _concurrent_promotion(root, document, start, results):
    record = EvalRecord.model_validate(document)
    start.wait()
    try:
        write_record(root, record)
        results.put("written")
    except ValueError as exc:
        results.put(str(exc))


@pytest.mark.parametrize("identical", [False, True])
def test_concurrent_process_promotions_are_serialized(built, identical):
    import multiprocessing

    root, record = built
    # Synthetic declared cost makes qualification valid without claiming a hardware measurement.
    record = record.model_copy(update={"gpu_time": GpuTime(authority="declared", gpu_count=1, wall_seconds=1.0)})
    path = write_record(root, record)
    report = verify_record(record, root, _predict, record.prediction.fingerprint)
    verified = attest(record, report)
    other = verified if identical else attest(record, report, "qualified")
    context = multiprocessing.get_context("spawn")
    start, results = context.Event(), context.Queue()
    workers = [
        context.Process(target=_concurrent_promotion, args=(root, r.model_dump(by_alias=True), start, results))
        for r in (verified, other)
    ]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(20)
        if worker.is_alive():
            worker.terminate()
            worker.join()
            pytest.fail("record writer did not finish")
        assert worker.exitcode == 0
    outcomes = [results.get(timeout=5) for _ in workers]
    assert outcomes.count("written") == (2 if identical else 1)
    if not identical:
        assert any("already has successor" in outcome for outcome in outcomes)
    assert len(current_records(path.parent)) == 1
    assert not list(path.parent.glob(".*lock"))


def test_oversized_records_are_rejected_before_publication(built, monkeypatch):
    from aisim_core.data import eval as module
    from aisim_core.data.hygiene import CandidatePolicy

    root, record = built
    limit = 2048
    monkeypatch.setattr(module, "CandidatePolicy", lambda: CandidatePolicy(max_json_bytes=limit))
    document = record.model_dump(by_alias=True)
    document["assumptions"][0]["value"] = "x" * limit
    oversized = EvalRecord.model_validate(document)
    for _ in range(2):
        with pytest.raises(ValueError, match="JSON exceeds"):
            write_record(root, oversized)
        assert not record_path(root, oversized).exists()


def test_a_verification_report_cannot_attest_another_record(built):
    root, record = built
    report = verify_record(record, root, _predict, record.prediction.fingerprint)
    assert report.ok and report.record_id == record.record_id
    changed = record.model_copy(update={"results": record.results.model_copy(update={"mape_pct": 99.0})})
    with pytest.raises(ValueError, match="different record content"):
        attest(changed, report)
    assert attest(record, report).supersedes == record.record_id


def test_a_declared_findings_file_with_a_record_like_name_is_not_a_record(built):
    root, record = built
    directory = write_record(root, record).parent
    payload = b'{"note": "synthetic findings"}'
    digest = hashlib.sha256(payload).hexdigest()
    (directory / f"{digest}.json").write_bytes(payload)  # a content-addressed name, like a record's
    findings = Artifact(
        artifact_id="findings",
        path=f"{digest}.json",
        role="findings",
        format="json",
        schema="findings/v1",
        sha256=digest,
        bytes=len(payload),
    )
    successor = record.model_copy(update={"findings": [findings], "supersedes": record.record_id})
    # The successor check of the write loads every record beside it, including this sidecar's directory.
    assert write_record(root, successor).parent == directory
    assert [p.stem for p in record_files(directory)] == sorted([record.record_id, successor.record_id])
    assert [r.record_id for r in current_records(directory)] == [successor.record_id]
    # Undeclared content at a record-like name is still a corrupt record that callers must see.
    (directory / f"{'e' * 64}.json").write_bytes(b"{}")
    assert (directory / f"{'e' * 64}.json") in record_files(directory)
    with pytest.raises(ValueError):
        current_records(directory)


def test_attest_cannot_demote_a_record_but_can_keep_or_raise_its_tier(built):
    root, record = built
    # Qualification needs a known GPU time.
    measured = GpuTime(authority="measured", gpu_count=4, wall_seconds=1.0)
    record = EvalRecord.model_validate({**record.model_dump(by_alias=True), "gpu_time": measured.model_dump()})

    def report_for(current):
        return verify_record(current, root, _predict, current.prediction.fingerprint)

    verified = attest(record, report_for(record))
    qualified = attest(verified, report_for(verified), "qualified")
    assert (verified.tier, qualified.tier) == ("verified", "qualified")
    with pytest.raises(ValueError, match="cannot demote a qualified record to verified"):
        attest(qualified, report_for(qualified))  # the default target tier is "verified"
    with pytest.raises(ValueError, match="cannot demote a qualified record to verified"):
        attest(qualified, report_for(qualified), "verified")
    assert attest(qualified, report_for(qualified), "qualified").tier == "qualified"  # equal-tier re-attestation
    assert attest(verified, report_for(verified)).tier == "verified"


def _successor_with_findings(record, tag):
    """A successor of ``record`` that links its own hash-named findings file; returns it with the file's bytes."""
    payload = f'{{"note": "{tag}"}}'.encode()
    digest = hashlib.sha256(payload).hexdigest()
    findings = Artifact(
        artifact_id="findings",
        path=f"{digest}.json",
        role="findings",
        format="json",
        schema="findings/v1",
        sha256=digest,
        bytes=len(payload),
    )
    successor = record.model_copy(update={"findings": [findings], "supersedes": record.record_id})
    return successor, {findings.path: payload}


def _leftovers(directory):
    return sorted(p.name for p in directory.iterdir() if p.name.startswith(".") or p.suffix == ".tmp")


@pytest.mark.parametrize("reverse", [False, True])
def test_two_pending_findings_files_leave_one_successor_and_a_clean_failure(built, reverse):
    root, record = built
    directory = write_record(root, record).parent
    first, first_files = _successor_with_findings(record, "one")
    second, second_files = _successor_with_findings(record, "two")
    pairs = [(first, first_files), (second, second_files)]
    winner, loser = pairs[::-1] if reverse else pairs
    # Neither caller has written anything beside the records: each findings file is published with its record.
    write_record(root, winner[0], sidecars=winner[1])
    with pytest.raises(ValueError, match=f"already has successor {winner[0].record_id}"):
        write_record(root, loser[0], sidecars=loser[1])
    assert [r.record_id for r in current_records(directory)] == [winner[0].record_id]
    (kept,) = winner[1]
    (dropped,) = loser[1]
    assert (directory / kept).read_bytes() == winner[1][kept]
    assert not (directory / dropped).exists()  # the competing writer left nothing behind
    assert _leftovers(directory) == []
    # The winner can be written again, and the loser's retry fails the same way, not on a corrupt record.
    assert write_record(root, winner[0], sidecars=winner[1]).parent == directory
    with pytest.raises(ValueError, match="already has successor"):
        write_record(root, loser[0], sidecars=loser[1])


def test_concurrent_successors_with_pending_findings_publish_exactly_one(built):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    root, record = built
    directory = write_record(root, record).parent
    candidates = [_successor_with_findings(record, tag) for tag in ("one", "two", "three", "four")]
    barrier = Barrier(len(candidates))

    def attempt(candidate):
        barrier.wait()
        try:
            return write_record(root, candidate[0], sidecars=candidate[1])
        except ValueError as exc:
            return exc

    with ThreadPoolExecutor(len(candidates)) as pool:
        results = list(pool.map(attempt, candidates))
    published = [r for r in results if isinstance(r, Path)]
    failures = [r for r in results if isinstance(r, ValueError)]
    assert len(published) == 1 and len(failures) == len(candidates) - 1
    assert all("already has successor" in str(f) for f in failures)
    (current,) = current_records(directory)
    assert published[0] == record_path(root, current)
    assert sorted(p.name for p in directory.glob("*.json")) == sorted(
        [f"{record.record_id}.json", f"{current.record_id}.json", current.findings[0].path]
    )
    assert _leftovers(directory) == []


def test_a_retry_completes_the_sidecars_of_an_already_published_record(built):
    root, record = built
    directory = write_record(root, record).parent
    successor, files = _successor_with_findings(record, "one")
    path = write_record(root, successor, sidecars=files)
    (name,) = files
    (directory / name).unlink()  # a failure between the record and its sidecar
    assert write_record(root, successor, sidecars=files) == path
    assert (directory / name).read_bytes() == files[name]
    assert [r.record_id for r in current_records(directory)] == [successor.record_id]


def test_sidecars_are_checked_against_the_record_before_anything_is_published(built):
    root, record = built
    directory = write_record(root, record).parent
    successor, files = _successor_with_findings(record, "one")
    (name,) = files
    with pytest.raises(ValueError, match="not an artifact of the record"):
        write_record(root, successor, sidecars={**files, "other.json": b"{}"})
    with pytest.raises(ValueError, match="declared size or digest"):
        write_record(root, successor, sidecars={name: b'{"note": "tampered"}'})
    (directory / name).write_bytes(b'{"note": "someone else"}')  # a different file already holds the name
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_record(root, successor, sidecars=files)
    assert not record_path(root, successor).exists()
    assert _leftovers(directory) == []


def _hold_lock_in_another_process(directory, **environment):
    """A process that holds the record lock of ``directory`` until it is killed."""
    import subprocess
    import sys

    import aisim_core

    code = (
        "import sys, time\nfrom pathlib import Path\nfrom aisim_core.data.eval import _record_lock\n"
        "with _record_lock(Path(sys.argv[1])):\n    print('held', flush=True)\n    time.sleep(120)\n"
    )
    source = str(Path(aisim_core.__file__).parents[1])
    env = {**os.environ, "PYTHONPATH": source, **environment}
    holder = subprocess.Popen([sys.executable, "-c", code, str(directory)], stdout=subprocess.PIPE, text=True, env=env)
    assert holder.stdout.readline().strip() == "held"
    return holder


def test_a_killed_writer_does_not_block_the_next_one(built, monkeypatch):
    from aisim_core.data import eval as eval_module

    root, record = built
    directory = write_record(root, record).parent
    successor, files = _successor_with_findings(record, "one")
    monkeypatch.setattr(eval_module, "_RECORD_LOCK_WAIT_SECONDS", 0.3)
    holder = _hold_lock_in_another_process(directory)
    try:
        # While the holder is alive the directory is busy, so no competing successor can be published.
        with pytest.raises(ValueError, match="another record writer is publishing"):
            write_record(root, successor, sidecars=files)
        assert not record_path(root, successor).exists()
    finally:
        holder.kill()  # no cleanup runs: the lock is released by the operating system
        holder.wait()
    monkeypatch.setattr(eval_module, "_RECORD_LOCK_WAIT_SECONDS", 5.0)
    assert write_record(root, successor, sidecars=files) == record_path(root, successor)
    assert _leftovers(directory) == []  # no lock file is left in the dataset


def test_the_record_lock_is_exclusive_between_live_writers_of_one_directory(built, monkeypatch):
    from aisim_core.data import eval as eval_module

    root, record = built
    directory = write_record(root, record).parent
    successor, files = _successor_with_findings(record, "one")
    monkeypatch.setattr(eval_module, "_RECORD_LOCK_WAIT_SECONDS", 0.2)
    with (
        eval_module._record_lock(directory),
        pytest.raises(ValueError, match="another record writer is publishing"),
    ):
        write_record(root, successor, sidecars=files)
    write_record(root, successor, sidecars=files)  # released on exit, however the holder leaves


def _successor_declaring(record, relative):
    payload = b'{"note": "nested sidecar"}'
    digest = hashlib.sha256(payload).hexdigest()
    artifact = Artifact(
        artifact_id="findings",
        path=relative,
        role="findings",
        format="json",
        schema="findings/v1",
        sha256=digest,
        bytes=len(payload),
    )
    successor = record.model_copy(update={"findings": [artifact], "supersedes": record.record_id})
    return successor, {relative: payload}


def test_sidecars_cannot_follow_a_symlinked_directory_out_of_the_dataset(built, tmp_path):
    root, record = built
    directory = write_record(root, record).parent
    outside = tmp_path / "outside"
    outside.mkdir()
    (directory / "linked").symlink_to(outside, target_is_directory=True)
    successor, files = _successor_declaring(record, "linked/note.json")
    with pytest.raises(ValueError, match="symlink"):
        write_record(root, successor, sidecars=files)
    assert list(outside.iterdir()) == []  # nothing was created outside the dataset
    assert not record_path(root, successor).exists()  # and no record was published
    assert _leftovers(directory) == []
    # An ordinary nested directory is still supported.
    nested, files = _successor_declaring(record, "nested/note.json")
    path = write_record(root, nested, sidecars=files)
    assert path.exists() and (directory / "nested/note.json").read_bytes() == files["nested/note.json"]


def test_the_record_lock_does_not_depend_on_the_temporary_directory_setting(built, monkeypatch, tmp_path):
    import tempfile

    from aisim_core.data import eval as eval_module

    root, record = built
    directory = write_record(root, record).parent
    successor, files = _successor_with_findings(record, "one")
    holder_tmp, writer_tmp = tmp_path / "holder-tmp", tmp_path / "writer-tmp"
    holder_tmp.mkdir()
    writer_tmp.mkdir()
    holder = _hold_lock_in_another_process(directory, TMPDIR=str(holder_tmp), TEMP=str(holder_tmp), TMP=str(holder_tmp))
    try:
        # A writer whose environment names another temporary directory must still meet the holder's lock.
        for name in ("TMPDIR", "TEMP", "TMP"):
            monkeypatch.setenv(name, str(writer_tmp))
        monkeypatch.setattr(tempfile, "tempdir", None)
        assert tempfile.gettempdir() == str(writer_tmp)
        monkeypatch.setattr(eval_module, "_RECORD_LOCK_WAIT_SECONDS", 0.3)
        with pytest.raises(ValueError, match="another record writer is publishing"):
            write_record(root, successor, sidecars=files)
        assert not record_path(root, successor).exists()
        assert not list(writer_tmp.iterdir()) and not list(holder_tmp.iterdir())  # no lock file depends on either
    finally:
        holder.kill()
        holder.wait()


def _successor_with_two_sidecars(record, tag):
    first, first_files = _successor_with_findings(record, f"{tag}-one")
    second = f'{{"note": "{tag}-two"}}'.encode()
    digest = hashlib.sha256(second).hexdigest()
    other = Artifact(
        artifact_id="more-findings",
        path=f"{digest}.json",
        role="findings",
        format="json",
        schema="findings/v1",
        sha256=digest,
        bytes=len(second),
    )
    successor = first.model_copy(update={"findings": [*first.findings, other]})
    return successor, {**first_files, other.path: second}


def test_a_failed_sidecar_publication_withdraws_the_record_so_no_successor_builds_on_it(built, monkeypatch):
    root, record = built
    directory = write_record(root, record).parent
    predecessor, files = _successor_with_two_sidecars(record, "predecessor")
    last = list(files)[-1]
    real_link = os.link

    def failing_link(source, target, *args, **kwargs):
        if Path(target).name == last:
            raise OSError("the second sidecar cannot be linked")
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "link", failing_link)
    with pytest.raises(OSError, match="cannot be linked"):
        write_record(root, predecessor, sidecars=files)
    monkeypatch.setattr(os, "link", real_link)
    # Nothing of the failed write remains: not the record, not the sidecar that did link, no staged copy.
    assert not record_path(root, predecessor).exists()
    assert [p.name for p in directory.glob("*.json")] == [f"{record.record_id}.json"]
    assert _leftovers(directory) == []
    # A successor cannot be accepted on top of the withdrawn record, so it cannot claim the missing path either.
    rival, rival_files = _successor_with_findings(predecessor, "rival")
    with pytest.raises(ValueError, match="not found next to the new record"):
        write_record(root, rival, sidecars=rival_files)
    # Retrying the whole write completes it; only then can a successor build on it.
    assert write_record(root, predecessor, sidecars=files) == record_path(root, predecessor)
    assert all((directory / name).read_bytes() == data for name, data in files.items())
    write_record(root, rival, sidecars=rival_files)
    assert [r.record_id for r in current_records(directory)] == [rival.record_id]
