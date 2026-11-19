# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InferenceX ingestion: retries, rate limits, incomplete pages, valid-empty, contradictions, corrections.

All HTTP is faked; the response shapes mirror the live documented endpoints
(views/options, availability, benchmarks?exact=true, workflow-info).
"""

import ast
import json
import shutil
import socket
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
        etag = f'"{abs(hash(json.dumps(body, sort_keys=True))) % 10**8}"'
        if headers.get("If-None-Match") == etag:
            return HttpResponse(304, {"etag": etag}, b"")
        return HttpResponse(200, {"etag": etag}, json.dumps(body).encode())


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


def test_incomplete_read_from_the_transport_is_retried():
    import http.client

    fake = FakeInfx()
    fake.respond("availability", http.client.IncompleteRead(b"[1", 99), ok([]))
    assert InfxClient(fake, RetryPolicy(max_attempts=2), lambda _: None).get("availability").attempts == 2


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
    assert second.changes == {"added": [], "corrected": ["inferencex:2"], "removed": []}
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
    with pytest.raises(OSError, match="interrupted"):
        make_run(tmp_path, FakeInfx())
    assert not list(root.rglob("snapshot.json")) and not (root / infx_job.STATE_FILE).exists()
    monkeypatch.setattr(infx_ingest.os, "replace", original)
    result, _ = make_run(tmp_path, FakeInfx())
    assert read_snapshot(root, infx_job.snapshot_path(result.snapshot_id))[0].snapshot_id == result.snapshot_id


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
    assert len(originals) == 2
    assert {gzip.decompress((root / a.path).read_bytes()) for a in originals} == {
        json.dumps(page).encode() for page in rows.values()
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
