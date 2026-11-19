# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weekly/manual InferenceX ingestion job: state, promotion, staleness and alerts.

``python -m aisim_core.data.infx_job {run,promote,staleness}``. The internal GitLab
job (``scripts/ci/gitlab/aisim-infx-ingest.gitlab-ci.yml``) only calls these entry points.

Rules enforced here (AISIM-2334 / DEP #435):

* An **unchanged** refresh (snapshot ID equals the qualified or pending candidate) writes
  nothing, so the job has no commit to make.
* An **empty** run writes nothing under the data root. A **failed** publication may
  leave completed immutable revisions for retry; the qualification pointer stays
  unchanged, and the failed GitLab job does not push its checkout.
* A successful run with new content writes a **candidate** only. The job has no code path
  that sets ``qualified``; :func:`promote` needs a named reviewer and is a separate manual step.
* A model-filtered candidate replaces the qualified scope. Its change report
  includes removals from unselected models, so review sees every effect of promotion.
* Staleness is measured from the last qualified snapshot: warn at 8 days, stale at 14
  (alert only; nothing is deleted or demoted).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.request
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aisim_core.data.e2e import SnapshotMember, load_e2e, read_snapshot, verify_snapshot_members
from aisim_core.data.hygiene import bounded_json_bytes, json_sidecar_value
from aisim_core.data.infx_ingest import (
    Acquirer,
    AcquisitionCache,
    InfxClient,
    InfxError,
    RetryPolicy,
    Transport,
    diff_members,
    plan_snapshot,
    urllib_transport,
    write_candidate,
)
from aisim_core.data.infx_ingest import (
    snapshot_path as snapshot_path,
)

STATE_FILE = "ingestion-state.json"
STATE_SCHEMA = "aisim-infx-ingestion-state/v1"
OWNER = "Simone Chen"
SLACK_CHANNEL = "#swdl-dynamo-aisim-daily"
WARN_DAYS = 8
STALE_DAYS = 14


class PromotionError(Exception):
    pass


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state(root: Path) -> dict[str, Any]:
    path = Path(root) / STATE_FILE
    if not path.exists():
        return {"schema_version": STATE_SCHEMA, "qualified": None, "candidate": None}
    try:
        state = json_sidecar_value(bounded_json_bytes(path))
    except RecursionError as exc:
        raise ValueError("ingestion state JSON exceeds the decoder nesting limit") from exc
    if not isinstance(state, dict):
        raise ValueError("ingestion state must be an object")
    if state.get("schema_version") != STATE_SCHEMA:
        raise ValueError(f"unsupported ingestion state schema {state.get('schema_version')!r}")
    for key in ("qualified", "candidate"):
        entry = state.get(key)
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"ingestion state {key} must be an object or null")
        for name in ("snapshot_id", "snapshot_path", *(("qualified_at", "reviewer") if key == "qualified" else ())):
            if not isinstance(entry.get(name), str) or not entry[name].strip():
                raise ValueError(f"ingestion state {key}.{name} must be a nonempty string")
        sid = entry["snapshot_id"]
        if len(sid) != 64 or any(c not in "0123456789abcdef" for c in sid):
            raise ValueError(f"ingestion state {key}.snapshot_id must be a SHA-256")
        for name in ("created_at", "qualified_at"):
            if name not in entry:
                continue
            value = entry[name]
            if not isinstance(value, str) or datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is None:
                raise ValueError(f"ingestion state {key}.{name} must include a timezone")
        counts = entry.get("change_counts", {})
        if not isinstance(counts, dict) or any(type(n) is not int or n < 0 for n in counts.values()):
            raise ValueError(f"ingestion state {key}.change_counts must contain nonnegative integer counts")
    return state


@contextmanager
def _state_lock(root: Path):
    """Serialize state commits; stale locks after a crash need operator inspection."""
    lock = root.resolve().parent / f".{root.resolve().name}.ingestion-state.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise InfxError("another writer is updating ingestion state; retry the operation") from exc
    try:
        yield
    finally:
        os.close(fd)
        lock.unlink()


def _check_state(root: Path, expected: Mapping[str, Any] | None) -> None:
    if expected is not None and load_state(root) != expected:
        raise InfxError("ingestion state changed during the operation; retry with current state")


def _save_state(root: Path, state: Mapping[str, Any]) -> None:
    pending = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=root, delete=False) as handle:
            pending = Path(handle.name)
            handle.write(json.dumps(state, indent=2, sort_keys=True) + "\n")
        os.replace(pending, root / STATE_FILE)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def save_state(root: Path, state: Mapping[str, Any], *, expected: Mapping[str, Any] | None = None) -> None:
    """Replace state under a writer lock, refusing stale acquisition/promotion results."""
    root = Path(root)
    with _state_lock(root):
        _check_state(root, expected)
        _save_state(root, state)


def _members_of(root: Path, entry: Mapping[str, Any] | None) -> dict[str, SnapshotMember]:
    if not entry:
        return {}
    snapshot, members = read_snapshot(root, entry["snapshot_path"])
    if snapshot.snapshot_id != entry["snapshot_id"]:
        raise ValueError("qualified snapshot ID differs from ingestion state")
    return {m.observation_id: m for m in members}


@dataclass
class RunResult:
    status: str  # candidate | unchanged | empty | failed
    snapshot_id: str | None = None
    changes: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    requests_made: int = 0

    def to_json(self) -> dict[str, Any]:
        counts = {k: len(v) for k, v in self.changes.items()}
        return {
            "status": self.status,
            "snapshot_id": self.snapshot_id,
            "change_counts": counts,
            "warnings": self.warnings,
            "error": self.error,
            "requests_made": self.requests_made,
        }


def run_ingestion(
    root: Path,
    cache_dir: Path,
    *,
    models: list[str] | None = None,
    transport: Transport = urllib_transport,
    policy: RetryPolicy = RetryPolicy(),  # noqa: B008
    sleep: Callable[[float], None] | None = None,
    now: Callable[[], datetime] | None = None,
) -> RunResult:
    """Acquire, plan and (only if new) write a candidate. Never changes ``qualified``."""
    root = Path(root)
    clock = now or (lambda: datetime.now(UTC))
    kwargs = {"sleep": sleep} if sleep else {}
    client = InfxClient(transport, policy, **kwargs)
    try:
        state = load_state(root)
        expected = dict(state)
        qualified, candidate = state.get("qualified"), state.get("candidate")
        prior = _members_of(root, qualified)
        prior_models = {}
        if models and qualified:
            _, members = read_snapshot(root, qualified["snapshot_path"])
            by_path = {path: load_e2e(root, path).body.model_id for path in {m.manifest_path for m in members}}
            prior_models = {m.observation_id: by_path[m.manifest_path] for m in members}
        acquirer = Acquirer(client, AcquisitionCache(cache_dir), clock)
        acq = acquirer.acquire(models, list(prior), prior_models)
        plan = None if acq.empty_scope else plan_snapshot(acq)
        if acq.empty_scope:
            with _state_lock(root):
                _check_state(root, expected)
            return RunResult(
                "empty", warnings=["no InferenceX data declared for the scope"], requests_made=acq.requests_made
            )
        assert plan is not None
        known = {e["snapshot_id"] for e in (qualified, candidate) if e}
        changes = diff_members({oid: m.record_sha256 for oid, m in prior.items()}, plan.members)
        changes["revised_shards"] = sorted(
            {
                m.manifest_path
                for m in plan.members
                if m.observation_id in prior and m.manifest_path != prior[m.observation_id].manifest_path
            }
        )
        warnings = [f"availability lags benchmarks: {u}" for u in acq.undeclared]
        warnings.extend(f"missing producer metadata: {u}" for u in acq.missing_producer_metadata)
        if plan.snapshot_id in known:
            entry = next(e for e in (qualified, candidate) if e and e["snapshot_id"] == plan.snapshot_id)
            snapshot, members = read_snapshot(root, entry["snapshot_path"])
            if snapshot.snapshot_id != plan.snapshot_id:
                raise ValueError("existing snapshot ID differs from the planned snapshot")
            if snapshot.completeness != "complete":
                raise ValueError("existing snapshot must be complete")
            verify_snapshot_members(root, members)
            with _state_lock(root):
                _check_state(root, expected)
            return RunResult("unchanged", plan.snapshot_id, warnings=warnings, requests_made=acq.requests_made)
        with _state_lock(root):
            _check_state(root, expected)  # stale acquisitions must not stage any candidate bytes
            path = write_candidate(root, acq, plan)
            state["candidate"] = {
                "snapshot_id": plan.snapshot_id,
                "snapshot_path": path,
                "created_at": _stamp(clock()),
                "change_counts": {k: len(v) for k, v in changes.items()},
            }
            _save_state(root, state)
        return RunResult("candidate", plan.snapshot_id, changes, warnings, requests_made=acq.requests_made)
    except (InfxError, OSError, ValueError) as exc:
        return RunResult("failed", error=f"{type(exc).__name__}: {exc}", requests_made=client.requests_made)


def promote(root: Path, snapshot_id: str, reviewer: str, now: datetime | None = None) -> dict[str, Any]:
    """Manual review step: the pending candidate becomes the qualified snapshot."""
    root = Path(root)
    if not reviewer.strip():
        raise PromotionError("promotion requires a named reviewer")
    state = load_state(root)
    expected = dict(state)
    candidate = state.get("candidate")
    if not candidate or candidate["snapshot_id"] != snapshot_id:
        raise PromotionError(f"{snapshot_id} is not the pending candidate")
    snapshot, members = read_snapshot(root, candidate["snapshot_path"])
    if snapshot.snapshot_id != snapshot_id:
        raise PromotionError("candidate snapshot ID differs from the requested ID")
    if snapshot.completeness != "complete":
        raise PromotionError("candidate snapshot must be complete")
    verify_snapshot_members(root, members)
    state["qualified"] = {**candidate, "qualified_at": _stamp(now or datetime.now(UTC)), "reviewer": reviewer}
    state["candidate"] = None
    try:
        save_state(root, state, expected=expected)
    except InfxError as exc:
        raise PromotionError(str(exc)) from exc
    return state["qualified"]


def staleness(state: Mapping[str, Any], now: datetime) -> dict[str, Any]:
    """``level`` is ok < 8 days <= warn < 14 days <= stale; ``never`` when nothing was qualified."""
    qualified = state.get("qualified")
    if not qualified:
        return {"level": "never", "age_days": None}
    age = (now - datetime.fromisoformat(qualified["qualified_at"].replace("Z", "+00:00"))).total_seconds() / 86400
    level = "stale" if age >= STALE_DAYS else "warn" if age >= WARN_DAYS else "ok"
    return {"level": level, "age_days": round(age, 2), "snapshot_id": qualified["snapshot_id"]}


# --------------------------------------------------------------------------- alerts


def post_slack(webhook: str, text: str) -> None:
    request = urllib.request.Request(
        webhook, json.dumps({"channel": SLACK_CHANNEL, "text": text}).encode(), {"Content-Type": "application/json"}
    )
    urllib.request.urlopen(request, timeout=30).close()


def send_mail(host: str, sender: str, to: str, subject: str, text: str) -> None:
    import smtplib
    from email.message import EmailMessage

    message = EmailMessage()
    message["From"], message["To"], message["Subject"] = sender, to, subject
    message.set_content(text)
    with smtplib.SMTP(host, timeout=30) as smtp:
        smtp.send_message(message)


def alert(
    subject: str,
    text: str,
    *,
    slack: Callable[[str], None] | None,
    mail: Callable[[str, str], None] | None,
) -> list[str]:
    """Send to Slack and to the owner's email independently; returns delivery failures (never raises)."""
    failures = []
    for name, send in (
        ("slack", (lambda: slack(f"{subject}\n{text}")) if slack else None),
        ("email", (lambda: mail(subject, text)) if mail else None),
    ):
        if send is None:
            failures.append(f"{name}: not configured")
            continue
        try:
            send()
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    return failures


def _env_alert(subject: str, text: str) -> list[str]:
    hook, host = os.environ.get("SLACK_WEBHOOK_URL"), os.environ.get("SMTP_HOST")
    to = os.environ.get("AISIM_JOB_OWNER_EMAIL")
    return alert(
        f"[aisim-infx-ingest] {subject}",
        f"owner: {OWNER}\n{text}",
        slack=(lambda t: post_slack(hook, t)) if hook else None,
        mail=(lambda s, t: send_mail(host, os.environ.get("SMTP_FROM", to or ""), to, s, t)) if host and to else None,
    )


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="infx_job")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--root", type=Path, required=True, help="working tree of the private cache repository")
    run.add_argument("--cache-dir", type=Path, required=True, help="mutable acquisition cache (not committed)")
    run.add_argument("--model", action="append", help="display name or DB key; default all")
    run.add_argument("--no-notify", action="store_true", help="let the CI fallback job deliver failure alerts")
    run.add_argument("--report", type=Path, help="write the run report JSON here")
    pro = sub.add_parser("promote")
    pro.add_argument("--root", type=Path, required=True)
    pro.add_argument("--snapshot-id", required=True)
    pro.add_argument("--reviewer", required=True)
    notify = sub.add_parser("notify", help="alert the owner channels about a job failure outside ingestion")
    notify.add_argument("--subject", required=True)
    notify.add_argument("--text", required=True)
    stale = sub.add_parser("staleness")
    stale.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "run":
        result = run_ingestion(args.root, args.cache_dir, models=args.model)
        report = result.to_json()
        if args.report:
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True))
        if result.status == "failed":
            if not args.no_notify:
                _env_alert("ingestion failed", f"{result.error}\nThe last qualified snapshot is retained.")
            return 1
        return 0
    if args.command == "promote":
        print(json.dumps(promote(args.root, args.snapshot_id, args.reviewer), sort_keys=True))
        return 0
    if args.command == "notify":
        failures = _env_alert(args.subject, args.text)
        print(json.dumps({"delivery_failures": failures}))
        return 0 if len(failures) < 2 else 1
    verdict = staleness(load_state(args.root), datetime.now(UTC))
    print(json.dumps(verdict, sort_keys=True))
    if verdict["level"] in ("warn", "stale", "never"):
        _env_alert(f"data {verdict['level']}", json.dumps(verdict, sort_keys=True))
    return {"ok": 0, "warn": 10, "stale": 11, "never": 11}[verdict["level"]]


if __name__ == "__main__":
    sys.exit(main())
