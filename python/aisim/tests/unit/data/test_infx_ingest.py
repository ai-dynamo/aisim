# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InferenceX ingestion: retries, rate limits, incomplete pages, valid-empty, contradictions, corrections.

All HTTP is faked; the response shapes mirror the live documented endpoints
(views/options, availability, benchmarks?exact=true, workflow-info).
"""

import ast
import gzip
import hashlib
import http.client
import io
import json
import shutil
import socket
import urllib.error
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aisim_core.data import e2e as e2e_module
from aisim_core.data import infx_ingest, infx_job
from aisim_core.data.e2e import evaluation_points, load_e2e, read_raw_record, read_snapshot
from aisim_core.data.infx_ingest import (
    HttpResponse,
    HttpStatusError,
    InfxClient,
    RateLimitedError,
    RetryPolicy,
    TransientExhaustedError,
    parse_retry_after,
)
from aisim_core.data.infx_job import PromotionError, promote, run_ingestion, staleness

pytestmark = pytest.mark.unit

NOW = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
DATE = "2026-11-05"
RUN = "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/987"
OPTIONS = {"models": [{"name": "Test-Model", "dbKeys": ["testmodel"]}, {"name": "Other", "dbKeys": ["other"]}]}


def row(rid=1, hardware="h100", **over):
    value = dict(
        id=str(rid), model="testmodel", hardware=hardware, framework="vllm", precision="fp8", spec_method="none",
        disagg=False, is_multinode=False, prefill_tp=1, prefill_ep=1, prefill_dp_attention=False,
        prefill_num_workers=0, decode_tp=8, decode_ep=1, decode_dp_attention=False, decode_num_workers=1,
        num_prefill_gpu=0, num_decode_gpu=8, benchmark_type="single_turn", isl=1024, osl=128, conc=32,
        date=DATE, image="img", run_url=RUN, metrics={"mean_ttft": 0.1, "mean_tpot": 0.01, "mean_e2el": 1.5},
    )  # fmt: skip
    value.update(over)
    return value


def avail(hardware="h100", model="testmodel"):
    return dict(model=model, isl=1024, osl=128, precision="fp8", hardware=hardware, framework="vllm",
                spec_method="none", disagg=False, benchmark_type="single_turn", date=DATE)  # fmt: skip


WORKFLOW = {
    "runs": [{"html_url": RUN, "run_attempt": 2, "github_run_id": "987"}],
    "runConfigs": [{"html_url": RUN, "head_sha": "a" * 40}],
    "changelogs": [],
    "configs": [],
}


class FakeInfx:
    """Scripted transport: route -> list of responses (last one repeats); records every call."""

    def __init__(self, rows=None, availability=None, workflow=WORKFLOW):
        self.rows = [row(1), row(2, "h200")] if rows is None else rows
        self.availability = [avail("h100"), avail("h200")] if availability is None else availability
        self.workflow = workflow
        self.script: dict[str, list] = {}
        self.calls: list[tuple[str, dict]] = []

    def respond(self, route, *responses):
        self.script[route] = list(responses)

    def __call__(self, url, headers, timeout):
        route = url.split("/api/v1/")[-1].split("?")[0]
        self.calls.append((url, dict(headers)))
        queue = self.script.get(route)
        if queue:
            item = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(item, Exception):
                raise item
            return item
        body = {"views/options": OPTIONS, "availability": self.availability, "benchmarks": self.rows,
                "workflow-info": self.workflow}[route]  # fmt: skip
        if route == "benchmarks" and isinstance(body, dict):  # per-date rows (exact=true is "as of" that date)
            body = body[url.split("date=")[1].split("&")[0]]
        body_bytes = json.dumps(body).encode()
        etag = f'"{hashlib.sha256(body_bytes).hexdigest()}"'
        if headers.get("If-None-Match") == etag:
            return HttpResponse(304, {"etag": etag}, b"")
        return HttpResponse(200, {"etag": etag}, body_bytes)


def ok(body, **headers):
    return HttpResponse(200, {k.lower(): v for k, v in headers.items()}, json.dumps(body).encode())


def make_run(tmp_path, fake, **kw):
    sleeps = []
    result = run_ingestion(
        tmp_path / "root", tmp_path / "cache", transport=fake, sleep=sleeps.append, now=lambda: NOW,
        policy=kw.pop("policy", RetryPolicy(max_attempts=3, base_delay=1, max_retry_after=10)), **kw,
    )  # fmt: skip
    return result, sleeps


@pytest.fixture
def root(tmp_path):
    (tmp_path / "root").mkdir()
    return tmp_path / "root"


def tree(path: Path):
    return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("attempts", [(None, 2), (2, None), (2, 3)])
def test_run_attempt_merge_requires_two_known_conflicting_values(tmp_path, attempts):
    fake = FakeInfx()
    pages = []
    for attempt in attempts:
        run = {"html_url": RUN}
        if attempt is not None:
            run["run_attempt"] = attempt
        pages.append(ok({**WORKFLOW, "runs": [run]}))
    fake.respond("workflow-info", *pages)
    acquirer = infx_ingest.Acquirer(InfxClient(fake), infx_ingest.AcquisitionCache(tmp_path))
    if attempts == (2, 3):
        with pytest.raises(infx_ingest.ContradictionError, match="run_attempt"):
            acquirer._runs([DATE, "2026-11-06"])
    else:
        assert acquirer._runs([DATE, "2026-11-06"])[RUN]["run_attempt"] == 2


def test_overflow_metric_writes_cli_failure_report_and_preserves_qualification(root, tmp_path, monkeypatch):
    first, _ = make_run(tmp_path, FakeInfx())
    promote(root, first.snapshot_id, "reviewer", NOW)
    before = tree(root)
    fake = FakeInfx(rows=[row(1, metrics={"mean_ttft": 10**400}), row(2, "h200")])
    run = infx_job.run_ingestion
    monkeypatch.setattr(infx_job, "run_ingestion", lambda *a, **kw: run(*a, **kw, transport=fake))
    report = tmp_path / "report.json"
    assert (
        infx_job.main(
            ["run", "--root", str(root), "--cache-dir", str(tmp_path / "cache"), "--report", str(report), "--no-notify"]
        )
        == 1
    )
    failure = json.loads(report.read_text())
    assert failure["status"] == "failed" and "IncompleteAcquisitionError" in failure["error"]
    assert tree(root) == before


def test_large_workflow_response_is_retained_in_published_candidate(root, tmp_path):
    from aisim_core.data import Manifest, _verify

    raw = json.dumps({**WORKFLOW, "padding": "x" * (17 * 1024 * 1024)}).encode()
    fake = FakeInfx(rows=[row()], availability=[avail()])
    fake.respond("workflow-info", HttpResponse(200, {}, raw))
    result, _ = make_run(tmp_path, fake)
    assert result.status == "candidate", result.error
    _, members = read_snapshot(root, infx_job.load_state(root)["candidate"]["snapshot_path"])
    manifest = Manifest.model_validate_json((root / members[0].manifest_path).read_bytes())
    bodies = [gzip.decompress(_verify(root, a).read_bytes()) for a in manifest.artifacts if a.format == "gzip"]
    assert raw in bodies


def test_deep_state_writes_cli_failure_report(root, tmp_path):
    (root / infx_job.STATE_FILE).write_bytes(b"[" * 10000 + b"0" + b"]" * 10000)
    before = tree(root)
    report = tmp_path / "report.json"
    assert (
        infx_job.main(
            ["run", "--root", str(root), "--cache-dir", str(tmp_path / "cache"), "--report", str(report), "--no-notify"]
        )
        == 1
    )
    failure = json.loads(report.read_text())
    assert "ValueError" in failure["error"] and "nesting" in failure["error"]
    assert failure["requests_made"] == 0 and tree(root) == before


def test_empty_acquisition_rechecks_concurrent_qualification(root, tmp_path):
    first, _ = make_run(tmp_path, FakeInfx())
    promoted = []
    fake = FakeInfx(rows=[], availability=[])

    def interleave(url, headers, timeout):
        if not promoted:
            promoted.append(promote(root, first.snapshot_id, "reviewer", NOW))
        return fake(url, headers, timeout)

    result, _ = make_run(tmp_path, interleave)
    assert result.status == "failed" and "state changed" in result.error
    assert infx_job.load_state(root)["qualified"] == promoted[0]


@pytest.mark.parametrize("operation", ["promote", "reuse", "reuse-orphan"])
def test_partial_snapshot_cannot_be_qualified_or_reused(root, tmp_path, operation):
    first, _ = make_run(tmp_path, FakeInfx())
    path = root / infx_job.load_state(root)["candidate"]["snapshot_path"]
    document = json.loads(path.read_bytes())
    document["completeness"] = "partial"
    path.write_text(json.dumps(document))
    if operation == "reuse-orphan":
        state = infx_job.load_state(root)
        infx_job.save_state(root, {**state, "candidate": None}, expected=state)
    before = tree(root)
    if operation == "promote":
        with pytest.raises(PromotionError, match="must be complete"):
            promote(root, first.snapshot_id, "reviewer", NOW)
    else:
        result, _ = make_run(tmp_path, FakeInfx())
        assert result.status == "failed" and "must be complete" in result.error
    assert tree(root) == before


# --------------------------------------------------------------------------- client


def test_retry_after_parsing():
    assert parse_retry_after("7") == 7.0
    assert parse_retry_after(None) is None and parse_retry_after("soon") is None
    assert parse_retry_after("Sat, 07 Nov 2026 12:00:30 GMT", now=NOW.timestamp()) == pytest.approx(30.0)


def test_transient_5xx_and_transport_errors_are_retried_with_backoff():
    fake = FakeInfx()
    fake.respond("availability", HttpResponse(503, {}, b""), ConnectionError("reset"), ok([]))
    sleeps = []
    client = InfxClient(fake, RetryPolicy(max_attempts=4, base_delay=1), sleeps.append)
    fetched = client.get("availability")
    assert fetched.attempts == 3 and sleeps == [1, 2]


def test_rate_limit_honors_retry_after_then_succeeds():
    fake = FakeInfx()
    fake.respond("availability", HttpResponse(429, {"retry-after": "5"}, b""), ok([]))
    sleeps = []
    InfxClient(fake, RetryPolicy(max_attempts=3, max_retry_after=10), sleeps.append).get("availability")
    assert sleeps == [5.0]


def test_retry_after_beyond_cap_fails_instead_of_waiting():
    fake = FakeInfx()
    fake.respond("availability", HttpResponse(429, {"retry-after": "3600"}, b""))
    sleeps = []
    with pytest.raises(RateLimitedError):
        InfxClient(fake, RetryPolicy(max_attempts=3, max_retry_after=10), sleeps.append).get("availability")
    assert sleeps == [] and len(fake.calls) == 1


def test_retries_are_bounded_and_client_errors_are_not_retried():
    fake = FakeInfx()
    fake.respond("availability", HttpResponse(500, {}, b""))
    with pytest.raises(TransientExhaustedError):
        InfxClient(fake, RetryPolicy(max_attempts=3), lambda _: None).get("availability")
    assert len(fake.calls) == 3
    fake.calls.clear()
    fake.respond("availability", HttpResponse(400, {}, b"bad"))
    with pytest.raises(HttpStatusError):
        InfxClient(fake, RetryPolicy(max_attempts=3), lambda _: None).get("availability")
    assert len(fake.calls) == 1


def test_truncated_body_is_retried_and_request_budget_is_enforced():
    fake = FakeInfx()
    fake.respond("availability", HttpResponse(200, {"content-length": "99"}, b"[1"), ok([]))
    assert InfxClient(fake, RetryPolicy(max_attempts=2), lambda _: None).get("availability").attempts == 2
    fake.respond("availability", HttpResponse(500, {}, b""))
    with pytest.raises(RateLimitedError):
        InfxClient(fake, RetryPolicy(max_attempts=9, max_requests=2), lambda _: None).get("availability")


@pytest.mark.parametrize(
    "error", [http.client.IncompleteRead(b"[1", 99), http.client.BadStatusLine(""), http.client.LineTooLong("status")]
)
def test_http_protocol_errors_from_the_transport_are_retried(error):
    fake = FakeInfx()
    fake.respond("availability", error, ok([]))
    assert InfxClient(fake, RetryPolicy(max_attempts=2), lambda _: None).get("availability").attempts == 2


def test_exhausted_http_protocol_errors_write_failure_report_and_alert(root, tmp_path, monkeypatch):
    first, _ = make_run(tmp_path, FakeInfx())
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    before = tree(root)
    fake = FakeInfx()
    fake.respond("availability", http.client.BadStatusLine(""))
    original = infx_job.run_ingestion
    monkeypatch.setattr(
        infx_job,
        "run_ingestion",
        lambda *a, **k: original(*a, **k, transport=fake, sleep=lambda _: None, policy=RetryPolicy(max_attempts=2)),
    )
    alerts = []
    monkeypatch.setattr(infx_job, "_env_alert", lambda subject, text: alerts.append(subject) or [])
    report = tmp_path / "failure.json"
    assert infx_job.main(["run", "--root", str(root), "--cache-dir", str(tmp_path / "c"), "--report", str(report)]) == 1
    result = json.loads(report.read_text())
    assert result["status"] == "failed" and "transport error" in result["error"]
    assert alerts == ["ingestion failed"] and tree(root) == before


# --------------------------------------------------------------------------- end to end


def test_first_run_writes_candidate_with_stable_shards_and_never_qualifies(root, tmp_path):
    result, _ = make_run(tmp_path, FakeInfx())
    assert result.status == "candidate" and len(result.changes["added"]) == 2
    state = infx_job.load_state(root)
    assert state["qualified"] is None and state["candidate"]["snapshot_id"] == result.snapshot_id
    snapshot, members = read_snapshot(root, state["candidate"]["snapshot_path"])
    assert snapshot.completeness == "complete" and snapshot.counts == {"observations": 2, "configurations": 2}
    assert all(m.manifest_path.startswith("revisions/") and "/e2e/testmodel/" in m.manifest_path for m in members)
    # One shared loader, producer metadata from workflow-info reached the observation row.
    dataset = load_e2e(root, members[0].manifest_path, config=None)
    assert dataset.observations[0].producer_revision == "a" * 40 and dataset.observations[0].producer_attempt == 2
    assert evaluation_points(dataset)


def test_unchanged_refresh_writes_nothing_and_uses_conditional_requests(root, tmp_path):
    make_run(tmp_path, FakeInfx())
    before = tree(root)
    fake = FakeInfx()
    result, _ = make_run(tmp_path, fake)
    assert result.status == "unchanged" and tree(root) == before
    assert any("If-None-Match" in headers for _, headers in fake.calls)


@pytest.mark.parametrize("moved", [False, True])
def test_promotion_requires_the_candidates_qualified_baseline_to_be_current(root, tmp_path, moved):
    first, _ = make_run(tmp_path, FakeInfx())
    baseline = tmp_path / "baseline"  # the default branch as it is now
    baseline.mkdir()
    state = infx_job.load_state(root)
    if moved:  # another candidate was qualified and merged after this one was staged
        qualified = {
            **state["candidate"],
            "snapshot_id": "b" * 64,
            "qualified_at": "2026-11-06T00:00:00Z",
            "reviewer": "r",
        }
        state = {**state, "qualified": qualified, "candidate": None}
    infx_job.save_state(baseline, state | {"candidate": None})
    if moved:
        with pytest.raises(PromotionError, match="different qualified baseline"):
            promote(root, first.snapshot_id, "Reviewer", now=NOW, baseline=baseline)
        assert infx_job.load_state(root)["qualified"] is None
    else:
        promote(root, first.snapshot_id, "Reviewer", now=NOW, baseline=baseline)
        assert infx_job.load_state(root)["qualified"]["snapshot_id"] == first.snapshot_id


def test_verify_pending_checks_the_candidate_and_its_member_artifacts(root, tmp_path):
    first, _ = make_run(tmp_path, FakeInfx())
    assert infx_job.main(["verify", "--root", str(root), "--snapshot-id", first.snapshot_id]) == 0
    with pytest.raises(PromotionError):
        infx_job.verify_pending(root, "0" * 64)
    next(root.rglob("metrics.parquet")).unlink()
    with pytest.raises(ValueError, match="missing dataset artifact"):
        infx_job.verify_pending(root, first.snapshot_id)


def test_candidate_is_not_promoted_by_a_later_run_and_promotion_needs_a_reviewer(root, tmp_path):
    first, _ = make_run(tmp_path, FakeInfx())
    with pytest.raises(PromotionError):
        promote(root, first.snapshot_id, "  ")
    with pytest.raises(PromotionError):
        promote(root, "0" * 64, "reviewer")
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    state = infx_job.load_state(root)
    assert state["qualified"]["snapshot_id"] == first.snapshot_id and state["candidate"] is None
    second, _ = make_run(tmp_path, FakeInfx(rows=[row(1), row(2, "h200"), row(3, "b200")],
                                            availability=[avail("h100"), avail("h200"), avail("b200")]))  # fmt: skip
    state = infx_job.load_state(root)
    assert second.status == "candidate" and state["qualified"]["snapshot_id"] == first.snapshot_id
    assert state["candidate"]["snapshot_id"] == second.snapshot_id


def test_correction_keeps_observation_id_changes_hash_and_preserves_old_snapshot(root, tmp_path):
    first, _ = make_run(tmp_path, FakeInfx())
    old_doc = (root / infx_job.snapshot_path(first.snapshot_id)).read_bytes()
    _, old_members = read_snapshot(root, infx_job.snapshot_path(first.snapshot_id))
    old_data = {m.observation_id: load_e2e(root, m.manifest_path) for m in old_members}
    old_shards = {m.manifest_path: tree((root / m.manifest_path).parent) for m in old_members}
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    fixed = row(2, "h200", metrics={"mean_ttft": 0.2, "mean_tpot": 0.01, "mean_e2el": 1.5})
    second, _ = make_run(tmp_path, FakeInfx(rows=[row(1), fixed]))
    assert second.status == "candidate" and second.snapshot_id != first.snapshot_id
    _, corrected_members = read_snapshot(root, infx_job.snapshot_path(second.snapshot_id))
    assert second.changes == {
        "added": [],
        "corrected": ["inferencex:2"],
        "removed": [],
        "revised_shards": sorted({m.manifest_path for m in corrected_members}),
    }
    assert (root / infx_job.snapshot_path(first.snapshot_id)).read_bytes() == old_doc  # old pin stays readable
    for member in old_members:
        data = load_e2e(root, member.manifest_path)
        assert data.observations == old_data[member.observation_id].observations
        assert data.metrics == old_data[member.observation_id].metrics
        assert (
            next(o for o in data.observations if o.observation_id == member.observation_id).record_sha256
            == member.record_sha256
        )
        assert tree((root / member.manifest_path).parent) == old_shards[member.manifest_path]
        assert json.loads(read_raw_record(root, data, member.observation_id))["metrics"]["mean_ttft"] == 0.1
    _, new_members = read_snapshot(root, infx_job.snapshot_path(second.snapshot_id))
    old_paths = {m.observation_id: m.manifest_path for m in old_members}
    for member in new_members:
        data = load_e2e(root, member.manifest_path)
        expected_ttft = 0.2 if member.observation_id == "inferencex:2" else 0.1
        assert json.loads(read_raw_record(root, data, member.observation_id))["metrics"]["mean_ttft"] == expected_ttft
        assert evaluation_points(data)[0].metrics[("ttft", "mean")] == expected_ttft * 1000
        # The original full HTTP page changed too, so even the unchanged row gets a new evidence revision.
        assert member.manifest_path != old_paths[member.observation_id]
    assert infx_job.load_state(root)["qualified"]["snapshot_id"] == first.snapshot_id


@pytest.mark.parametrize("change", ["add", "remove"])
def test_configuration_membership_changes_preserve_qualified_shards(root, tmp_path, change):
    first, _ = make_run(tmp_path, FakeInfx(rows=[row(1), row(3)], availability=[avail()]))
    _, old_members = read_snapshot(root, infx_job.snapshot_path(first.snapshot_id))
    old_path = old_members[0].manifest_path
    before = tree((root / old_path).parent)
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    rows = [row(1), row(3), row(4)] if change == "add" else [row(1)]
    second, _ = make_run(tmp_path, FakeInfx(rows=rows, availability=[avail()]))
    assert second.status == "candidate"
    _, new_members = read_snapshot(root, infx_job.snapshot_path(second.snapshot_id))
    assert new_members[0].manifest_path != old_path
    assert tree((root / old_path).parent) == before
    assert {o.observation_id for o in load_e2e(root, old_path).observations} == {"inferencex:1", "inferencex:3"}
    assert {o.observation_id for o in load_e2e(root, new_members[0].manifest_path).observations} == {
        f"inferencex:{r['id']}" for r in rows
    }


def test_producer_metadata_correction_retains_the_old_configuration_revision(root, tmp_path):
    first, _ = make_run(tmp_path, FakeInfx())
    _, old_members = read_snapshot(root, infx_job.snapshot_path(first.snapshot_id))
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    workflow = {**WORKFLOW, "runConfigs": [{"html_url": RUN, "head_sha": "b" * 40}]}
    second, _ = make_run(tmp_path, FakeInfx(workflow=workflow))
    assert second.status == "candidate" and second.snapshot_id != first.snapshot_id
    assert second.to_json()["change_counts"] == {"added": 0, "corrected": 0, "removed": 0, "revised_shards": 2}
    _, new_members = read_snapshot(root, infx_job.snapshot_path(second.snapshot_id))
    for old, new in zip(old_members, new_members, strict=True):
        assert old.record_sha256 == new.record_sha256  # workflow-info, rather than the benchmark, changed
        assert old.manifest_path != new.manifest_path
        assert load_e2e(root, old.manifest_path).observations[0].producer_revision == "a" * 40
        assert load_e2e(root, new.manifest_path).observations[0].producer_revision == "b" * 40


def test_refresh_preserves_legacy_configuration_paths(root, tmp_path):
    from aisim_core.data.e2e import SnapshotMember, snapshot_id, write_configuration, write_snapshot

    records = [row(1), row(2, "h200")]
    members = []
    for cfg in e2e_module.inferencex_to_configurations(records):
        manifest = write_configuration(root, cfg)
        members.extend(
            SnapshotMember(
                configuration_id=o.configuration_id,
                observation_id=o.observation_id,
                record_sha256=o.record_sha256,
                manifest_path=manifest,
            )
            for o in cfg.observations
        )
    scope = {"source": "inferencex", "models": "all"}
    sid = snapshot_id(members, scope, infx_ingest.SELECTION_POLICY)
    path = infx_job.snapshot_path(sid)
    (root / path).parent.mkdir(parents=True)
    write_snapshot(
        root, path, members, source_scope=scope, selection_policy=infx_ingest.SELECTION_POLICY, completeness="complete"
    )
    infx_job.save_state(
        root,
        {
            "schema_version": infx_job.STATE_SCHEMA,
            "qualified": None,
            "candidate": {"snapshot_id": sid, "snapshot_path": path},
        },
    )
    promote(root, sid, "Reviewer", now=NOW)
    legacy = tree(root / "e2e")
    result, _ = make_run(tmp_path, FakeInfx(rows=[row(1), row(2, "h200", metrics={"mean_ttft": 0.2})]))
    assert result.status == "candidate"
    assert tree(root / "e2e") == legacy
    for member in read_snapshot(root, path)[1]:
        data = load_e2e(root, member.manifest_path)
        assert data.observations[0].record_sha256 == member.record_sha256
        assert evaluation_points(data)[0].metrics[("ttft", "mean")] == 100.0


def test_removed_upstream_observation_is_reported(root, tmp_path):
    first, _ = make_run(tmp_path, FakeInfx())
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    second, _ = make_run(tmp_path, FakeInfx(rows=[row(1)], availability=[avail("h100")]))
    assert second.changes["removed"] == ["inferencex:2"]


# --------------------------------------------------------------------------- failure keeps last qualified


@pytest.mark.parametrize(
    ("route", "response", "error"),
    [
        ("benchmarks", HttpResponse(503, {}, b""), "TransientExhaustedError"),
        ("benchmarks", HttpResponse(429, {"retry-after": "9999"}, b""), "RateLimitedError"),
        ("benchmarks", HttpResponse(200, {}, b'[{"id": 1'), "IncompleteAcquisitionError"),
        ("benchmarks", ok({"rows": []}), "IncompleteAcquisitionError"),
        ("workflow-info", HttpResponse(500, {}, b""), "TransientExhaustedError"),
    ],
)
def test_failures_write_nothing_and_retain_last_qualified(root, tmp_path, route, response, error):
    first, _ = make_run(tmp_path, FakeInfx())
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    before = tree(root)
    fake = FakeInfx(rows=[row(1), row(2, "h200", metrics={"mean_ttft": 9.0})])  # would otherwise be a new candidate
    fake.respond(route, response)
    result, _ = make_run(tmp_path, fake)
    assert result.status == "failed" and error in result.error
    assert tree(root) == before
    assert infx_job.load_state(root)["qualified"]["snapshot_id"] == first.snapshot_id


def test_transient_failure_then_success_still_produces_the_snapshot(root, tmp_path):
    fake = FakeInfx()
    fake.respond("benchmarks", HttpResponse(502, {}, b""), HttpResponse(429, {"retry-after": "2"}, b""),
                 ok(fake.rows))  # fmt: skip
    result, sleeps = make_run(tmp_path, fake)
    assert result.status == "candidate" and sleeps == [1, 2.0]


def test_incomplete_page_missing_declared_configuration_fails(root, tmp_path):
    result, _ = make_run(tmp_path, FakeInfx(rows=[row(1)]))  # availability declares h200 too
    assert result.status == "failed" and "IncompleteAcquisitionError" in result.error and "absent" in result.error
    assert not (root / infx_job.STATE_FILE).exists()


def test_declared_but_empty_benchmarks_is_incomplete_not_valid_empty(root, tmp_path):
    result, _ = make_run(tmp_path, FakeInfx(rows=[]))
    assert result.status == "failed" and "IncompleteAcquisitionError" in result.error


def test_valid_empty_scope_is_not_a_failure_and_writes_nothing(root, tmp_path):
    result, _ = make_run(tmp_path, FakeInfx(availability=[]))
    assert result.status == "empty" and result.error is None and not (root / infx_job.STATE_FILE).exists()


def test_valid_empty_after_data_existed_is_a_contradiction(root, tmp_path):
    first, _ = make_run(tmp_path, FakeInfx())
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    result, _ = make_run(tmp_path, FakeInfx(availability=[]))
    assert result.status == "failed" and "ContradictionError" in result.error


def test_availability_differing_only_by_workload_shape_is_declared_separately(root, tmp_path):
    long_context = {**avail("h100"), "isl": 2048}
    result, _ = make_run(tmp_path, FakeInfx(rows=[row(1)], availability=[avail("h100"), long_context]))
    assert result.status == "failed" and "IncompleteAcquisitionError" in result.error and "absent" in result.error


def test_unknown_model_selector_fails_but_known_model_without_data_is_empty(root, tmp_path):
    result, _ = make_run(tmp_path, FakeInfx(), models=["Test-Modle"])
    assert result.status == "failed" and "unknown model selector" in result.error
    assert make_run(tmp_path, FakeInfx(), models=["other"])[0].status == "empty"  # a db key is accepted too


def test_model_filter_selecting_nothing_is_valid_empty(root, tmp_path):
    result, _ = make_run(tmp_path, FakeInfx(), models=["Other"])
    assert result.status == "empty"


@pytest.mark.parametrize("selector", ["Other", "other"])
def test_empty_model_scope_does_not_compare_another_models_prior_data(root, tmp_path, selector):
    first, _ = make_run(tmp_path, FakeInfx())
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    before = tree(root)
    assert make_run(tmp_path, FakeInfx(), models=[selector])[0].status == "empty"
    assert tree(root) == before


@pytest.mark.parametrize("artifact", ["manifest.json", "metrics.parquet"])
def test_promotion_verifies_member_configuration_artifacts(root, tmp_path, artifact):
    first, _ = make_run(tmp_path, FakeInfx())
    state = infx_job.load_state(root)
    _, members = read_snapshot(root, state["candidate"]["snapshot_path"])
    directory = (root / members[0].manifest_path).parent
    path = directory / artifact
    if not path.exists():
        path = next(directory.glob("*metrics*.parquet"))
    path.unlink()
    before = (root / infx_job.STATE_FILE).read_bytes()
    with pytest.raises((OSError, ValueError)):
        promote(root, first.snapshot_id, "Reviewer", now=NOW)
    assert (root / infx_job.STATE_FILE).read_bytes() == before


def test_snapshot_document_publication_is_atomic_and_retryable(root, tmp_path, monkeypatch):
    original = infx_ingest.os.replace

    def interrupt(source, destination):
        if str(destination).endswith("/snapshot.json"):
            raise OSError("interrupted snapshot publication")
        return original(source, destination)

    monkeypatch.setattr(infx_ingest.os, "replace", interrupt)
    failed, _ = make_run(tmp_path, FakeInfx())
    assert failed.status == "failed" and "interrupted" in failed.error
    assert not list(root.rglob("snapshot.json")) and not (root / infx_job.STATE_FILE).exists()
    monkeypatch.setattr(infx_ingest.os, "replace", original)
    result, _ = make_run(tmp_path, FakeInfx())
    assert read_snapshot(root, infx_job.snapshot_path(result.snapshot_id))[0].snapshot_id == result.snapshot_id


@pytest.mark.parametrize("qualified", [False, True])
def test_unchanged_refresh_checks_member_artifacts(root, tmp_path, qualified):
    first, _ = make_run(tmp_path, FakeInfx())
    if qualified:
        promote(root, first.snapshot_id, "Reviewer", now=NOW)
    _, members = read_snapshot(root, infx_job.snapshot_path(first.snapshot_id))
    data = load_e2e(root, members[0].manifest_path)
    metrics = next(a for a in data.manifest.artifacts if "metrics" in a.artifact_schema)
    (root / metrics.path).unlink()
    before = tree(root)
    result, _ = make_run(tmp_path, FakeInfx())
    assert result.status == "failed" and result.error
    assert tree(root) == before


def test_existing_candidate_is_verified_before_registration(root, tmp_path):
    acquirer = infx_ingest.Acquirer(InfxClient(FakeInfx()), infx_ingest.AcquisitionCache(tmp_path / "cache"))
    acquisition = acquirer.acquire()
    plan = infx_ingest.plan_snapshot(acquisition)
    infx_ingest.write_candidate(root, acquisition, plan)
    data = load_e2e(root, plan.members[0].manifest_path)
    metrics = next(a for a in data.manifest.artifacts if "metrics" in a.artifact_schema)
    (root / metrics.path).unlink()
    with pytest.raises((ValueError, OSError)):
        infx_ingest.write_candidate(root, acquisition, plan)
    assert not (root / infx_job.STATE_FILE).exists()


def test_promote_rejects_valid_but_different_snapshot_document(root, tmp_path):
    first, _ = make_run(tmp_path, FakeInfx())
    second, _ = make_run(tmp_path, FakeInfx(rows=[row(1, conc=64), row(2, "h200")]))
    # Both documents and their referenced membership tables are internally valid.
    shutil.copyfile(root / infx_job.snapshot_path(first.snapshot_id), root / infx_job.snapshot_path(second.snapshot_id))
    assert read_snapshot(root, infx_job.snapshot_path(second.snapshot_id))[0].snapshot_id == first.snapshot_id
    before = tree(root)
    with pytest.raises(PromotionError, match="snapshot ID differs"):
        promote(root, second.snapshot_id, "Reviewer", now=NOW)
    assert tree(root) == before


@pytest.mark.parametrize("failure", ["state", "snapshot-read", "publication"])
def test_storage_failures_write_reports_and_notify(root, tmp_path, monkeypatch, failure):
    first, _ = make_run(tmp_path, FakeInfx())
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    state = (root / infx_job.STATE_FILE).read_bytes()
    fake = FakeInfx(rows=[row(1, conc=64), row(2, "h200")])
    if failure == "state":
        (root / infx_job.STATE_FILE).write_text("{")
        state = b"{"
    elif failure == "snapshot-read":
        (root / infx_job.snapshot_path(first.snapshot_id)).write_text("{")
    else:
        original = infx_ingest.os.replace

        def fail_publication(source, destination):
            if str(destination).endswith("/snapshot.json"):
                raise OSError("fixture storage failure")
            return original(source, destination)

        monkeypatch.setattr(infx_ingest.os, "replace", fail_publication)
    monkeypatch.setattr(
        infx_job,
        "run_ingestion",
        lambda root, cache_dir, **kw: run_ingestion(root, cache_dir, transport=fake, sleep=lambda _: None),
    )
    alerts = []
    monkeypatch.setattr(infx_job, "_env_alert", lambda *args: alerts.append(args) or [])
    report = tmp_path / "storage-report.json"
    assert (
        infx_job.main(["run", "--root", str(root), "--cache-dir", str(tmp_path / "cache"), "--report", str(report)])
        == 1
    )
    result = json.loads(report.read_text())
    assert result["status"] == "failed" and result["error"]
    assert alerts and (root / infx_job.STATE_FILE).read_bytes() == state
    assert not root.with_name(".root.ingestion-state.lock").exists()


def test_existing_corrupt_snapshot_is_not_registered(root, tmp_path):
    acq = infx_ingest.Acquirer(InfxClient(FakeInfx()), infx_ingest.AcquisitionCache(tmp_path / "cache")).acquire()
    plan = infx_ingest.plan_snapshot(acq)
    path = root / infx_job.snapshot_path(plan.snapshot_id)
    path.parent.mkdir(parents=True)
    path.write_text('{"truncated":')
    with pytest.raises(ValueError):
        infx_ingest.write_candidate(root, acq, plan)
    assert not (root / infx_job.STATE_FILE).exists()


def test_malformed_workflow_info_fails_without_changing_qualified_state(root, tmp_path):
    first, _ = make_run(tmp_path, FakeInfx())
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    before = tree(root)
    result, _ = make_run(tmp_path, FakeInfx(workflow={"error": "unavailable"}))
    assert result.status == "failed" and "workflow-info" in result.error
    assert tree(root) == before


# --------------------------------------------------------------------------- contradictions


def test_duplicate_id_with_different_bodies_is_a_contradiction(root, tmp_path):
    rows = [row(1), row(1, "h200")]
    result, _ = make_run(tmp_path, FakeInfx(rows=rows))
    assert result.status == "failed" and "ContradictionError" in result.error


def test_same_benchmark_across_as_of_dates_is_deduplicated_ignoring_curve_context(root, tmp_path):
    day2 = "2026-11-06"
    rows = {DATE: [row(1, curve_date=DATE, curve_workflow_run_id="1")],
            day2: [row(1, curve_date=day2, curve_workflow_run_id="2")]}  # fmt: skip
    availability = [avail("h100"), {**avail("h100"), "date": day2}]
    result, _ = make_run(tmp_path, FakeInfx(rows=rows, availability=availability))
    assert result.status == "candidate" and result.changes["added"] == ["inferencex:1"]
    _, members = read_snapshot(root, infx_job.load_state(root)["candidate"]["snapshot_path"])
    assert len(members) == 1
    import gzip

    shutil.rmtree(tmp_path / "cache")
    data = load_e2e(root, members[0].manifest_path)
    originals = [a for a in data.manifest.artifacts if "inferencex-response-" in a.path]
    assert len(originals) == 4  # two benchmark pages and two dated workflow-info responses
    assert {gzip.decompress((root / a.path).read_bytes()) for a in originals} == {
        *[json.dumps(page).encode() for page in rows.values()],
        json.dumps(WORKFLOW).encode(),
    }


def test_row_for_a_different_model_is_a_contradiction(root, tmp_path):
    result, _ = make_run(tmp_path, FakeInfx(rows=[row(1), row(2, "h200", model="other")]))
    assert result.status == "failed" and "ContradictionError" in result.error


def test_conflicting_head_sha_between_workflow_pages_is_a_contradiction(root, tmp_path):
    workflow = {
        **WORKFLOW,
        "runConfigs": [{"html_url": RUN, "head_sha": "a" * 40}, {"html_url": RUN, "head_sha": "b" * 40}],
    }
    result, _ = make_run(tmp_path, FakeInfx(workflow=workflow))
    assert result.status == "failed" and "head_sha" in result.error


def test_availability_lagging_benchmarks_only_warns(root, tmp_path):
    result, _ = make_run(tmp_path, FakeInfx(availability=[avail("h100")], rows=[row(1), row(2, "h200")]))
    assert result.status == "candidate" and result.warnings


def test_row_missing_required_fields_is_incomplete(root, tmp_path):
    bad = row(1)
    del bad["decode_tp"]
    result, _ = make_run(tmp_path, FakeInfx(rows=[bad, row(2, "h200")]))
    assert result.status == "failed" and "IncompleteAcquisitionError" in result.error


@pytest.mark.parametrize("field,value", [("benchmark_type", None), ("id", "invalid"), ("id", "²")])
def test_malformed_benchmark_writes_failure_report(root, tmp_path, monkeypatch, field, value):
    first, _ = make_run(tmp_path, FakeInfx())
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    before = tree(root)
    bad = row()
    if value is None:
        del bad[field]
    else:
        bad[field] = value
    fake = FakeInfx(rows=[bad, row(2, "h200")])
    monkeypatch.setattr(
        infx_job,
        "run_ingestion",
        lambda root, cache_dir, **kw: run_ingestion(root, cache_dir, transport=fake, sleep=lambda _: None),
    )
    monkeypatch.setattr(infx_job, "_env_alert", lambda *args: [])
    report = tmp_path / "report.json"
    assert (
        infx_job.main(["run", "--root", str(root), "--cache-dir", str(tmp_path / "cache"), "--report", str(report)])
        == 1
    )
    result = json.loads(report.read_text())
    assert result["status"] == "failed" and "IncompleteAcquisitionError" in result["error"]
    if field == "id" and value is not None:
        assert "benchmark id must be numeric" in result["error"]
    assert tree(root) == before


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"models": None},
        {"models": {}},
        {"models": [1]},
        {"models": [{"dbKeys": ["testmodel"]}]},
        {"models": [{"name": 1, "dbKeys": ["testmodel"]}]},
        {"models": [{"name": "Test-Model", "dbKeys": "testmodel"}]},
        {"models": [{"name": "Test-Model", "dbKeys": [1]}]},
    ],
)
def test_malformed_model_options_fail_without_writing_state(root, tmp_path, options):
    first, _ = make_run(tmp_path, FakeInfx())
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    before = tree(root)
    fake = FakeInfx()
    fake.respond("views/options", ok(options))
    result, _ = make_run(tmp_path, fake)
    assert result.status == "failed" and "IncompleteAcquisitionError" in result.error
    assert tree(root) == before


@pytest.mark.parametrize("model", ["dsr1", "DEEPSEEK-AI/DEEPSEEK-R1-0528"])
def test_original_responses_follow_canonical_model_names(root, tmp_path, model):
    rows = [row(model=model)]
    fake = FakeInfx(rows=rows, availability=[avail(model=model)])
    fake.respond("views/options", ok({"models": [{"name": "DeepSeek R1", "dbKeys": [model]}]}))
    result, _ = make_run(tmp_path, fake)
    assert result.status == "candidate"
    _, members = read_snapshot(root, infx_ingest.snapshot_path(result.snapshot_id))
    loaded = load_e2e(root, members[0].manifest_path)
    responses = [a for a in loaded.manifest.artifacts if "inferencex-response-" in a.path]
    assert len(responses) == 2
    assert json.dumps(rows).encode() in {gzip.decompress((root / a.path).read_bytes()) for a in responses}
    assert loaded.body.model_id == "deepseek-ai/DeepSeek-R1-0528"


@pytest.mark.parametrize("selector", ["dsr1", "DeepSeek R1"])
def test_scoped_refresh_preserves_prior_canonical_observations(root, tmp_path, selector):
    options = {"models": [{"name": "DeepSeek R1", "dbKeys": ["dsr1"]}]}
    fake = FakeInfx(rows=[row(model="dsr1")], availability=[avail(model="dsr1")])
    fake.respond("views/options", ok(options))
    first, _ = make_run(tmp_path, fake)
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    # A different selector changes the scope, but the observations are inherited, not added again.
    result, _ = make_run(tmp_path, fake, models=[selector])
    assert result.status == "candidate"
    assert result.changes == {"added": [], "corrected": [], "removed": [], "revised_shards": []}
    before = tree(root)
    empty = FakeInfx(rows=[], availability=[])
    empty.respond("views/options", ok(options))
    failed, _ = make_run(tmp_path, empty, models=[selector])
    assert failed.status == "failed" and "ContradictionError" in failed.error
    assert tree(root) == before


# --------------------------------------------------------------------------- staleness and alerts


@pytest.mark.parametrize(
    ("days", "level"), [(0, "ok"), (7.9, "ok"), (8, "warn"), (13.9, "warn"), (14, "stale"), (40, "stale")]
)
def test_staleness_thresholds_from_last_qualified(days, level):
    state = {
        "qualified": {"snapshot_id": "x", "qualified_at": (NOW - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")}
    }
    assert staleness(state, NOW)["level"] == level
    assert staleness({"qualified": None}, NOW)["level"] == "never"


def test_alert_goes_to_slack_and_email_and_one_channel_failing_does_not_hide_the_other():
    sent = []

    def boom(_):
        raise OSError("slack down")

    failures = infx_job.alert("s", "t", slack=boom, mail=lambda subject, text: sent.append((subject, text)))
    assert sent == [("s", "t")] and failures == ["slack: slack down"]
    got = []
    assert infx_job.alert("s", "t", slack=got.append, mail=None) == ["email: not configured"] and got == ["s\nt"]


def test_cli_failure_alerts_owner_channels_and_exits_nonzero(root, tmp_path, monkeypatch):
    alerts = []
    monkeypatch.setattr(infx_job, "_env_alert", lambda subject, text: alerts.append(subject) or [])
    monkeypatch.setattr(infx_job, "run_ingestion", lambda *a, **k: infx_job.RunResult("failed", error="x"))
    assert infx_job.main(["run", "--root", str(root), "--cache-dir", str(tmp_path / "c")]) == 1
    assert alerts == ["ingestion failed"]


# --------------------------------------------------------------------------- evaluators never call InferenceX

EVALUATOR_MODULES = (
    "e2e",
    "loaders",
    "eval",
    "ensure",
    "fetch",
    "hygiene",
    "catalog",
    "validate",
    "parquet",
    "materialize",
)


def _imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.add(node.module or "")
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def test_evaluator_modules_do_not_import_the_ingestion_adapter_or_http_clients():
    data = Path(e2e_module.__file__).parent
    for name in EVALUATOR_MODULES:
        imports = _imports(data / f"{name}.py")
        assert not [i for i in imports if "infx_ingest" in i or "infx_job" in i], name
        assert not imports & {"urllib.request", "requests", "http.client", "socket"}, name


def test_evaluator_paths_make_zero_inferencex_calls(root, tmp_path, monkeypatch):
    make_run(tmp_path, FakeInfx())
    snapshot_file = infx_job.load_state(root)["candidate"]["snapshot_path"]
    calls = []
    monkeypatch.setattr(InfxClient, "get", lambda *a, **k: calls.append("get"))
    monkeypatch.setattr(infx_ingest, "urllib_transport", lambda *a, **k: calls.append("transport"))

    def no_network(*args, **kwargs):
        calls.append("socket")
        raise AssertionError("evaluator opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    _, members = read_snapshot(root, snapshot_file)
    for member in members:
        assert evaluation_points(load_e2e(root, member.manifest_path, config=None))
    assert calls == []


def test_interrupted_configuration_write_can_retry_and_qualify(root, tmp_path, monkeypatch):
    original = Path.write_text

    def interrupt(path, text, *args, **kwargs):
        if path.name == "manifest.json":
            path.write_bytes(b'{"partial":')
            raise OSError("interrupted manifest write")
        return original(path, text, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", interrupt)
        failed, _ = make_run(tmp_path, FakeInfx())
        assert failed.status == "failed" and "interrupted manifest" in failed.error
    assert not list(root.rglob("manifest.json"))
    result, _ = make_run(tmp_path, FakeInfx())
    assert result.status == "candidate"
    promote(root, result.snapshot_id, "Reviewer", now=NOW)
    assert infx_job.load_state(root)["qualified"]["snapshot_id"] == result.snapshot_id


@pytest.mark.parametrize("kind", ["availability", "metrics", "planning"])
def test_malformed_acquisition_writes_report_and_notifies(root, tmp_path, monkeypatch, kind):
    first, _ = make_run(tmp_path, FakeInfx())
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    before = tree(root)
    fake = FakeInfx(rows=[row(metrics=1), row(2, "h200")]) if kind == "metrics" else FakeInfx()
    if kind == "availability":
        fake.availability[0]["isl"] = []
    if kind == "planning":

        def bad_plan(_):
            raise infx_ingest.IncompleteAcquisitionError("incomplete retained response")

        monkeypatch.setattr(infx_job, "plan_snapshot", bad_plan)
    monkeypatch.setattr(
        infx_job,
        "run_ingestion",
        lambda root, cache_dir, **kw: run_ingestion(root, cache_dir, transport=fake, sleep=lambda _: None),
    )
    alerts = []
    monkeypatch.setattr(infx_job, "_env_alert", lambda *args: alerts.append(args) or [])
    report = tmp_path / "failed.json"
    assert (
        infx_job.main(["run", "--root", str(root), "--cache-dir", str(tmp_path / "cache"), "--report", str(report)])
        == 1
    )
    report_data = json.loads(report.read_text())
    assert report_data["status"] == "failed"
    if kind == "metrics":
        assert "metrics must be a mapping" in report_data["error"]
    assert alerts and tree(root) == before


def test_conflicting_attempts_across_dates_fail(root, tmp_path):
    date2 = "2026-11-06"
    fake = FakeInfx(
        rows={DATE: [row(1)], date2: [row(2, date=date2)]}, availability=[avail(), {**avail(), "date": date2}]
    )
    workflow2 = {**WORKFLOW, "runs": [{**WORKFLOW["runs"][0], "run_attempt": 3}]}
    fake.respond("workflow-info", ok(WORKFLOW), ok(workflow2))
    result, _ = make_run(tmp_path, fake)
    assert result.status == "failed" and "run_attempt" in result.error
    assert not list(root.iterdir())


@pytest.mark.parametrize(
    "corruption", ["truncated", "list", "missing-time", "bad-time", "etag", "url", "size", "unreadable"]
)
def test_corrupt_cache_metadata_is_refetched(root, tmp_path, monkeypatch, corruption):
    first, _ = make_run(tmp_path, FakeInfx())
    cache = infx_ingest.AcquisitionCache(tmp_path / "cache")
    _, path = cache._paths(InfxClient().url("views/options"))
    meta = json.loads(path.read_bytes())
    if corruption == "truncated":
        path.write_bytes(b"{")
    elif corruption == "list":
        path.write_text("[]")
    elif corruption == "unreadable":
        original = Path.read_bytes

        def read_bytes(candidate):
            if candidate == path:
                raise OSError("fixture unreadable cache metadata")
            return original(candidate)

        monkeypatch.setattr(Path, "read_bytes", read_bytes)
    else:
        if corruption == "missing-time":
            del meta["fetched_at"]
        elif corruption == "bad-time":
            meta["fetched_at"] = "not a timestamp"
        elif corruption == "etag":
            meta["etag"] = ["invalid"]
        elif corruption == "url":
            meta["url"] = "https://wrong.example.test/"
        else:
            meta["bytes"] += 1
        path.write_text(json.dumps(meta))
    fake = FakeInfx()
    result, _ = make_run(tmp_path, fake)
    assert result.status == "unchanged" and result.snapshot_id == first.snapshot_id
    options = next(headers for url, headers in fake.calls if "views/options" in url)
    assert "If-None-Match" not in options
    assert any("If-None-Match" in headers for url, headers in fake.calls if "views/options" not in url)


def test_duplicate_benchmark_metric_keys_fail_before_conversion(root, tmp_path):
    fake = FakeInfx()
    body = json.dumps(fake.rows).encode().replace(b'"mean_ttft": 0.1', b'"mean_ttft": 0.1, "mean_ttft": 0.2', 1)
    fake.respond("benchmarks", HttpResponse(200, {}, body))
    result, _ = make_run(tmp_path, fake)
    assert result.status == "failed" and "duplicate JSON key: 'mean_ttft'" in result.error
    assert not tree(root)


@pytest.mark.parametrize("selector,count", [("testmodel", 2), ("Test-Model", 3)])
def test_db_key_selector_drops_unselected_sibling_rows(root, tmp_path, selector, count):
    fake = FakeInfx(rows=[row(1), row(2, "h200"), row(3, model="other")])
    fake.respond("views/options", ok({"models": [{"name": "Test-Model", "dbKeys": ["testmodel", "other"]}]}))
    result, _ = make_run(tmp_path, fake, models=[selector])
    assert result.status == "candidate"
    _, members = read_snapshot(root, infx_job.snapshot_path(result.snapshot_id))
    assert len(members) == count
    assert {m.observation_id for m in members} == {f"inferencex:{i}" for i in range(1, count + 1)}


def test_workflow_response_evidence_survives_cache_removal(root, tmp_path):
    fake = FakeInfx()
    result, _ = make_run(tmp_path, fake)
    shutil.rmtree(tmp_path / "cache")
    _, members = read_snapshot(root, infx_job.snapshot_path(result.snapshot_id))
    for member in members:
        data = load_e2e(root, member.manifest_path)
        responses = [a for a in data.manifest.artifacts if "inferencex-response-" in a.path]
        raw = {gzip.decompress((root / a.path).read_bytes()) for a in responses}
        assert json.dumps(WORKFLOW).encode() in raw
        assert json.dumps(fake.rows).encode() in raw
        observation = next(o for o in data.observations if o.observation_id == member.observation_id)
        assert observation.producer_revision == "a" * 40 and observation.producer_attempt == 2


def test_reused_acquirer_keeps_only_current_response_evidence(root, tmp_path):
    fake = FakeInfx()
    fake.respond(
        "views/options",
        ok({"models": [{"name": "Test-Model", "dbKeys": ["testmodel"]}, {"name": "Other", "dbKeys": ["other"]}]}),
    )
    fake.rows = [row(1)]
    fake.availability = [avail(), avail(model="other")]
    fake.respond("benchmarks", ok([row(3, model="other")]), ok([row(1)]))
    acquirer = infx_ingest.Acquirer(InfxClient(fake), infx_ingest.AcquisitionCache(tmp_path / "cache"))
    first = acquirer.acquire()
    assert any("model=Other" in url for url in first.responses)
    second = acquirer.acquire(models=["testmodel"])
    assert not any("model=Other" in url for url in second.responses)
    plan = infx_ingest.plan_snapshot(second)
    path = infx_ingest.write_candidate(root, second, plan)
    assert {m.observation_id for m in read_snapshot(root, path)[1]} == {"inferencex:1"}


def test_ci_can_delegate_failure_notification_to_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(infx_job, "run_ingestion", lambda *a, **k: infx_job.RunResult("failed", error="fixture"))
    alerts = []
    monkeypatch.setattr(infx_job, "_env_alert", lambda *a: alerts.append(a))
    report = tmp_path / "report.json"
    assert (
        infx_job.main(
            [
                "run",
                "--root",
                str(tmp_path),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--report",
                str(report),
                "--no-notify",
            ]
        )
        == 1
    )
    assert not alerts and json.loads(report.read_text())["error"] == "fixture"


def test_model_filtered_refresh_reports_all_qualified_removals(root, tmp_path):
    rows = [row(1), row(2, model="other")]
    fake = FakeInfx(rows=rows, availability=[avail(), avail(model="other")])
    fake.respond("benchmarks", ok([rows[1]]), ok([rows[0]]))  # Other is queried before Test-Model
    first, _ = make_run(tmp_path, fake)
    assert first.status == "candidate", first.error
    promote(root, first.snapshot_id, "Reviewer", now=NOW)
    result, _ = make_run(tmp_path, FakeInfx(rows=[row(1)], availability=[avail()]), models=["Test-Model"])
    assert result.status == "candidate" and result.changes["removed"] == ["inferencex:2"]
    promote(root, result.snapshot_id, "Reviewer", now=NOW)
    _, members = read_snapshot(root, infx_ingest.snapshot_path(result.snapshot_id))
    assert {m.observation_id for m in members} == {"inferencex:1"}


@pytest.mark.parametrize("conflicting", [False, True])
def test_model_keys_require_consistent_display_names(root, tmp_path, conflicting):
    fake = FakeInfx()
    fake.respond(
        "views/options",
        ok(
            {
                "models": [
                    OPTIONS["models"][0],
                    {"name": "Different" if conflicting else "Test-Model", "dbKeys": ["testmodel"]},
                ]
            }
        ),
    )
    result, _ = make_run(tmp_path, fake)
    assert result.status == ("failed" if conflicting else "candidate")
    if conflicting:
        assert "ContradictionError" in result.error and tree(root) == {}


@pytest.mark.parametrize(
    "field,value",
    [
        ("decode_tp", "eight"),
        ("prefill_ep", True),
        ("decode_num_workers", -1),
        ("num_decode_gpu", "8"),
        ("is_multinode", "false"),
        ("decode_dp_attention", 1),
        # The live source reports a boolean for both roles, also on aggregated rows; null cannot be converted.
        ("prefill_dp_attention", None),
        ("decode_dp_attention", None),
    ],
)
def test_malformed_deployment_preserves_qualified_state(root, tmp_path, field, value):
    fake = FakeInfx()
    first, _ = make_run(tmp_path, fake)
    promote(root, first.snapshot_id, "reviewer", NOW)
    before = tree(root)
    fake.rows[0][field] = value
    result, _ = make_run(tmp_path, fake)
    assert result.status == "failed" and field in result.error
    assert tree(root) == before


@pytest.mark.parametrize("degree", [None, 0, 8])
def test_deployment_unknown_and_zero_sentinels_remain_supported(root, tmp_path, degree):
    result, _ = make_run(tmp_path, FakeInfx(rows=[row(decode_tp=degree)], availability=[avail()]))
    assert result.status == "candidate", result.error


def test_promotion_during_acquisition_is_not_reverted(root, tmp_path):
    first, _ = make_run(tmp_path, FakeInfx())
    promoted = []
    fake = FakeInfx(rows=[row(1, metrics={"mean_ttft": 0.2}), row(2, "h200")])

    def interleave(url, headers, timeout):
        if not promoted:
            promoted.append(promote(root, first.snapshot_id, "reviewer", NOW))
            promoted.append(tree(root))
        return fake(url, headers, timeout)

    result, _ = make_run(tmp_path, interleave)
    state = infx_job.load_state(root)
    assert promoted and state["qualified"] == promoted[0]
    assert tree(root) == promoted[1]
    assert state["candidate"] is None
    assert result.status == "failed" and "state changed" in result.error
    retry, _ = make_run(tmp_path, fake)
    assert retry.status == "candidate"
    assert infx_job.load_state(root)["qualified"] == promoted[0]


@pytest.mark.parametrize("size", [16, 17])
def test_transport_bounds_response_before_ingestion_reports_failure(root, tmp_path, monkeypatch, size):
    class Response(io.BytesIO):
        status = 200

        def __init__(self, data):
            super().__init__(data)
            self.headers = {}

        def read(self, amount=-1):
            assert amount == 17  # one sentinel byte beyond the limit, never an unbounded read
            return super().read(amount)

    response = Response(b" " * (size - 2) + b"{}")
    monkeypatch.setattr(infx_ingest, "MAX_RESPONSE_BYTES", 16)
    monkeypatch.setattr(infx_ingest.urllib.request, "urlopen", lambda *a, **k: response)
    if size == 16:
        assert len(infx_ingest.urllib_transport("https://example.test", {}, 1).body) == 16
    else:
        report = tmp_path / "oversized-report.json"
        assert (
            infx_job.main(
                [
                    "run",
                    "--root",
                    str(root),
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--report",
                    str(report),
                    "--no-notify",
                ]
            )
            == 1
        )
        result = json.loads(report.read_text())
        assert result["status"] == "failed" and "response exceeds" in result["error"]
        assert result["requests_made"] == 1 and not tree(root)
    assert response.closed


def test_transport_closes_http_errors_without_reading_the_body(monkeypatch):
    class Unreadable(io.BytesIO):
        def read(self, *args):
            pytest.fail("unused HTTP error body was read")

    body = Unreadable(b"unused")
    error = urllib.error.HTTPError("https://example.test", 429, "limited", {"Retry-After": "3"}, body)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(infx_ingest.urllib.request, "urlopen", fail)
    response = infx_ingest.urllib_transport("https://example.test", {}, 1)
    assert response.status == 429 and response.headers == {"retry-after": "3"} and response.body == b""
    assert body.closed


@pytest.mark.parametrize("missing", ["both", "head_sha", "run_attempt", "run_url"])
def test_missing_optional_producer_metadata_is_reported_for_candidates_and_noops(root, tmp_path, missing):
    workflow = json.loads(json.dumps(WORKFLOW))
    rows = None
    if missing == "both":
        workflow = {"runs": [], "runConfigs": []}
    elif missing == "run_url":
        rows = [row(1, run_url=None), row(2, "h200", run_url=None)]
    else:
        workflow["runs" if missing == "run_attempt" else "runConfigs"][0].pop(missing)
    fake = FakeInfx(rows=rows, workflow=workflow)
    result, _ = make_run(tmp_path, fake)
    assert result.status == "candidate" and any("missing producer metadata:" in w for w in result.warnings)
    _, members = read_snapshot(root, infx_job.snapshot_path(result.snapshot_id))
    observation = load_e2e(root, members[0].manifest_path).observations[0]
    assert (observation.producer_revision is None) is (missing in ("both", "head_sha", "run_url"))
    assert (observation.producer_attempt is None) is (missing in ("both", "run_attempt", "run_url"))
    repeated, _ = make_run(tmp_path, fake)
    assert repeated.status == "unchanged" and repeated.warnings == result.warnings


@pytest.mark.parametrize(
    "value",
    [
        [],
        None,
        1,
        {"schema_version": infx_job.STATE_SCHEMA, "candidate": []},
        {"schema_version": infx_job.STATE_SCHEMA, "qualified": "bad"},
        {"schema_version": infx_job.STATE_SCHEMA, "candidate": {}},
        {"schema_version": infx_job.STATE_SCHEMA, "candidate": {"snapshot_id": [], "snapshot_path": 1}},
    ],
)
def test_malformed_state_shapes_write_a_failure_report(root, tmp_path, value):
    path = root / infx_job.STATE_FILE
    path.write_text(json.dumps(value))
    before = tree(root)
    report = tmp_path / "state-report.json"
    assert (
        infx_job.main(
            ["run", "--root", str(root), "--cache-dir", str(tmp_path / "cache"), "--report", str(report), "--no-notify"]
        )
        == 1
    )
    result = json.loads(report.read_text())
    assert result["status"] == "failed" and "ValueError" in result["error"]
    assert result["requests_made"] == 0 and tree(root) == before


def test_ingestion_refuses_a_qualified_snapshot_id_path_mismatch(root, tmp_path):
    first, _ = make_run(tmp_path, FakeInfx())
    promote(root, first.snapshot_id, "Reviewer", NOW)
    fake = FakeInfx(rows=[row(1, metrics={"mean_ttft": 0.2}), row(2, "h200")])
    second, _ = make_run(tmp_path, fake)
    state = infx_job.load_state(root)
    state["qualified"]["snapshot_path"] = infx_job.snapshot_path(second.snapshot_id)
    infx_job.save_state(root, state)
    before = tree(root)
    result, _ = make_run(tmp_path, fake)
    assert result.status == "failed" and "qualified snapshot ID differs" in result.error
    assert result.requests_made == 0 and tree(root) == before


def test_noop_refuses_a_candidate_replaced_during_acquisition(root, tmp_path):
    first, _ = make_run(tmp_path, FakeInfx())
    replacement = []
    original = FakeInfx()

    def interleave(url, headers, timeout):
        if not replacement:
            result, _ = make_run(tmp_path, FakeInfx(rows=[row(1, metrics={"mean_ttft": 0.2}), row(2, "h200")]))
            assert result.status == "candidate" and result.snapshot_id != first.snapshot_id
            replacement.append(tree(root))
        return original(url, headers, timeout)

    result, _ = make_run(tmp_path, interleave)
    assert result.status == "failed" and "state changed" in result.error and "retry" in result.error
    assert tree(root) == replacement[0]
    retry, _ = make_run(tmp_path, original)
    assert retry.status == "candidate" and retry.snapshot_id == first.snapshot_id


@pytest.mark.parametrize("value", [[], {}, False, 0, "", "not-a-sha", "a" * 39, "g" * 40])
def test_malformed_producer_sha_is_rejected_with_a_report(root, tmp_path, monkeypatch, value):
    workflow = {**WORKFLOW, "runConfigs": [{"html_url": RUN, "head_sha": value}]}
    _assert_failed_acquisition_report(root, tmp_path, monkeypatch, FakeInfx(workflow=workflow), "head_sha")


@pytest.mark.parametrize("value", [[], {}, False, 1])
def test_malformed_benchmark_run_url_is_rejected_with_a_report(root, tmp_path, monkeypatch, value):
    _assert_failed_acquisition_report(root, tmp_path, monkeypatch, FakeInfx(rows=[row(1, run_url=value)]), "run_url")


def test_deep_acquisition_json_is_rejected_with_a_report(root, tmp_path, monkeypatch):
    fake = FakeInfx()
    fake.respond("views/options", HttpResponse(200, {}, b"[" * 10000 + b"0" + b"]" * 10000))
    _assert_failed_acquisition_report(root, tmp_path, monkeypatch, fake, "not valid JSON")


def _assert_failed_acquisition_report(root, tmp_path, monkeypatch, fake, message):
    first, _ = make_run(tmp_path, FakeInfx())
    promote(root, first.snapshot_id, "Reviewer", NOW)
    before = tree(root)
    original = infx_job.run_ingestion
    monkeypatch.setattr(
        infx_job, "run_ingestion", lambda *a, **k: original(*a, **k, transport=fake, sleep=lambda _: None)
    )
    report = tmp_path / "failure-report.json"
    assert (
        infx_job.main(
            [
                "run",
                "--root",
                str(root),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--report",
                str(report),
                "--no-notify",
            ]
        )
        == 1
    )
    result = json.loads(report.read_text())
    assert result["status"] == "failed" and "IncompleteAcquisitionError" in result["error"]
    assert message in result["error"] and tree(root) == before


def test_aggregated_zero_decode_workers_are_normalized_with_evidence(root, tmp_path):
    result, _ = make_run(tmp_path, FakeInfx(rows=[row(1, decode_num_workers=0), row(2, "h200")]))
    assert result.status == "candidate"
    _, members = read_snapshot(root, infx_ingest.snapshot_path(result.snapshot_id))
    member = next(m for m in members if m.observation_id == "inferencex:1")
    dataset = load_e2e(root, member.manifest_path)
    assert dataset.body.workers["aggregated"].num_workers == 1
    assert json.loads(dataset.observations[0].source_config_json)["decode_num_workers"] == 0
    assert any("zero sentinel" in gap["reason"] for gap in dataset.body.evidence_gaps)
    assert evaluation_points(dataset)[0].metrics[("ttft", "mean")] == 100.0


def test_duplicate_ingestion_state_keys_are_rejected_and_missing_is_first_run(root, tmp_path):
    assert infx_job.load_state(root) == {"schema_version": infx_job.STATE_SCHEMA, "qualified": None, "candidate": None}
    first, _ = make_run(tmp_path, FakeInfx())
    promote(root, first.snapshot_id, "Reviewer", NOW)
    state = infx_job.load_state(root)
    path = root / infx_job.STATE_FILE
    path.write_text(json.dumps(state)[:-1] + ', "qualified": null}')
    before = tree(root)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        infx_job.load_state(root)
    result, _ = make_run(tmp_path, FakeInfx())
    assert result.status == "failed" and "duplicate JSON key" in result.error
    assert tree(root) == before


def test_unconditional_304_does_not_revalidate_cached_evidence(root, tmp_path):
    fake = FakeInfx()
    fake.respond("views/options", ok(OPTIONS))  # the cached 200 has no ETag
    first, _ = make_run(tmp_path, fake)
    assert first.status == "candidate"
    before = tree(root)
    fake.respond("views/options", HttpResponse(304, {}, b""))
    result, _ = make_run(tmp_path, fake)
    assert result.status == "failed" and "conditional request validator" in result.error
    assert "If-None-Match" not in fake.calls[-1][1]
    assert tree(root) == before


@pytest.mark.parametrize("attempt", [-1, 0, True, "2", 1.5, [], {}])
def test_invalid_workflow_attempts_cannot_enter_a_snapshot(root, tmp_path, attempt):
    workflow = {**WORKFLOW, "runs": [{"html_url": RUN, "run_attempt": attempt}]}
    result, _ = make_run(tmp_path, FakeInfx(workflow=workflow))
    assert result.status == "failed" and "run_attempt must be a positive integer or null" in result.error
    assert not tree(root)


@pytest.mark.parametrize("spelling", ["p99.9_ttft", "p99_9_ttft", "both-equal", "both-conflicting"])
def test_canonical_metric_aliases_are_unambiguous(root, tmp_path, spelling):
    metrics = {"mean_ttft": 0.1}
    if spelling.startswith("both"):
        metrics.update({"p99.9_ttft": 0.1, "p99_9_ttft": 0.2 if spelling == "both-conflicting" else 0.1})
    else:
        metrics[spelling] = 0.1
    result, _ = make_run(tmp_path, FakeInfx(rows=[row(1, metrics=metrics)], availability=[avail()]))
    if spelling == "both-conflicting":
        assert result.status == "failed" and "conflicting canonical metric" in result.error
        assert not tree(root)
    else:
        assert result.status == "candidate"
        _, members = read_snapshot(root, infx_ingest.snapshot_path(result.snapshot_id))
        point = evaluation_points(load_e2e(root, members[0].manifest_path))[0]
        assert point.metrics[("ttft", "p99.9")] == 100.0  # independent seconds -> milliseconds expectation


@pytest.mark.parametrize("stale_manifest", ["none", "old-record", "extra-observation"])
def test_orphan_revision_reuse_checks_the_planned_observations(root, tmp_path, stale_manifest):
    def acquire(fake):
        return infx_ingest.Acquirer(InfxClient(fake), infx_ingest.AcquisitionCache(tmp_path / "cache")).acquire()

    old_rows = [row(1), row(2, "h200")]
    if stale_manifest == "extra-observation":
        old_rows.append(row(3))
    old = acquire(FakeInfx(rows=old_rows))
    old_plan = infx_ingest.plan_snapshot(old)
    infx_ingest.write_candidate(root, old, old_plan)
    new_row = row(1) if stale_manifest == "extra-observation" else row(1, metrics={"mean_ttft": 0.3})
    corrected = acquire(FakeInfx(rows=[new_row, row(2, "h200")]))
    plan = infx_ingest.plan_snapshot(corrected)
    path = infx_ingest.write_candidate(root, corrected, plan)
    (root / path).unlink()  # interrupted publication left valid orphan revisions
    member = next(m for m in plan.members if m.observation_id == "inferencex:1")
    if stale_manifest != "none":
        old_member = next(m for m in old_plan.members if m.observation_id == member.observation_id)
        (root / member.manifest_path).write_bytes((root / old_member.manifest_path).read_bytes())
    before = tree(root)
    if stale_manifest != "none":
        with pytest.raises(ValueError, match="snapshot member does not match"):
            infx_ingest.write_candidate(root, corrected, plan)
        assert tree(root) == before
    else:
        assert infx_ingest.write_candidate(root, corrected, plan) == path
        point = evaluation_points(load_e2e(root, member.manifest_path))[0]
        assert point.metrics[("ttft", "mean")] == 300.0


@pytest.mark.parametrize("missing", ["candidate", "qualified"])
def test_existing_ingestion_state_requires_explicit_pointers(root, tmp_path, missing):
    path = root / infx_job.STATE_FILE
    state = {"schema_version": infx_job.STATE_SCHEMA, "candidate": None, "qualified": None}
    path.write_text(json.dumps(state))
    assert infx_job.load_state(root) == state
    del state[missing]
    path.write_text(json.dumps(state))
    before = tree(root)
    with pytest.raises(ValueError, match=f"missing required pointer {missing}"):
        infx_job.load_state(root)
    result, _ = make_run(tmp_path, FakeInfx())
    assert result.status == "failed" and f"missing required pointer {missing}" in result.error
    assert tree(root) == before


def test_fake_etag_tracks_exact_response_bytes():
    fake = FakeInfx(rows=[row(1)])
    url = InfxClient(fake).url("benchmarks")
    first = fake(url, {}, 1)
    fake.rows = [dict(reversed(list(fake.rows[0].items())))]
    second = fake(url, {"If-None-Match": first.headers["etag"]}, 1)
    assert second.status == 200 and second.body != first.body
    assert second.headers["etag"] != first.headers["etag"]
    assert fake(url, {"If-None-Match": second.headers["etag"]}, 1).status == 304


def test_retry_after_accepts_only_ascii_delta_seconds():
    assert parse_retry_after("²") is None
    assert parse_retry_after("12") == 12.0


def test_ambiguous_model_selector_is_refused(tmp_path):
    fake = FakeInfx()
    fake.respond(
        "views/options", ok({"models": [{"name": "shared", "dbKeys": ["a"]}, {"name": "B", "dbKeys": ["shared"]}]})
    )
    fake.availability = [avail(model="a")]
    acquirer = infx_ingest.Acquirer(InfxClient(fake), infx_ingest.AcquisitionCache(tmp_path / "cache"))
    with pytest.raises(infx_ingest.InfxError, match="ambiguous"):
        acquirer.acquire(models=["shared"])


@pytest.mark.parametrize("value", ["2", 1.5, True, -1])
def test_malformed_parallelism_metric_is_refused(tmp_path, value):
    fake = FakeInfx(rows=[row(1, metrics={"mean_ttft": 0.1, "pp": value, "decode_pp": 1})])
    acquirer = infx_ingest.Acquirer(InfxClient(fake), infx_ingest.AcquisitionCache(tmp_path / "cache"))
    with pytest.raises(infx_ingest.IncompleteAcquisitionError, match="metrics.pp"):
        acquirer.acquire()


def test_request_budget_applies_to_each_acquisition(tmp_path):
    fake = FakeInfx()
    client = InfxClient(fake, RetryPolicy(max_requests=6))
    acquirer = infx_ingest.Acquirer(client, infx_ingest.AcquisitionCache(tmp_path / "cache"))
    acquirer.acquire()
    acquirer.acquire()  # a shared counter would exhaust the budget here


def test_failed_cache_replace_leaves_no_temporary_file(tmp_path, monkeypatch):
    cache = infx_ingest.AcquisitionCache(tmp_path / "cache")

    def fail(*_):
        raise OSError("disk full")

    monkeypatch.setattr(infx_ingest.os, "replace", fail)
    with pytest.raises(OSError):
        cache.put("https://x/y", b"{}", None, "2026-11-07T12:00:00Z")
    assert not [p for p in (tmp_path / "cache").rglob("*") if p.is_file()]


def test_report_write_failure_still_alerts_and_exits_nonzero(root, tmp_path, monkeypatch):
    fake = FakeInfx()
    fake.respond("availability", http.client.BadStatusLine(""))
    original = infx_job.run_ingestion
    monkeypatch.setattr(
        infx_job,
        "run_ingestion",
        lambda *a, **k: original(*a, **k, transport=fake, sleep=lambda _: None, policy=RetryPolicy(max_attempts=2)),
    )
    alerts = []
    monkeypatch.setattr(infx_job, "_env_alert", lambda subject, text: alerts.append(subject) or [])
    missing = tmp_path / "no-such-dir" / "report.json"
    args = ["run", "--root", str(root), "--cache-dir", str(tmp_path / "c"), "--report", str(missing)]
    assert infx_job.main(args) == 1 and alerts == ["ingestion failed"]
