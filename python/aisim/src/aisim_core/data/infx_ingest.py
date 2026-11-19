# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InferenceX REST ingestion adapter: acquisition cache -> immutable snapshot (``aisim-e2e/v1``).

This is an *ingestion source*, not an evaluator input. Evaluators read a frozen
snapshot through :func:`aisim_core.data.e2e.load_e2e` /
:func:`~aisim_core.data.e2e.read_snapshot` and never import this module.

Layers
------
1. **Client** (:class:`InfxClient`): documented public GET endpoints
   (``views/options``, ``availability``, ``benchmarks?exact=true``,
   ``workflow-info``) with bounded retries for 429/5xx/transport errors/truncated
   bodies, honoring ``Retry-After`` (capped; a longer request fails rather than
   waits). The transport and sleep are injected, so tests use fakes.
2. **Acquisition cache** (:class:`AcquisitionCache`): mutable, raw bytes + body
   hash + ETag per URL; conditional requests make an unchanged refresh cheap.
3. **Snapshot** (:func:`write_candidate`): converts through
   :func:`inferencex_to_configurations`, writes content-versioned shard paths
   (``revisions/<content-sha256>/e2e/<model>/.../<configuration-id>``) and an immutable
   ``snapshots/<snapshot_id>/`` document. Corrections keep ``observation_id`` and
   change ``record_sha256``, which changes the snapshot ID; old snapshot
   documents and their referenced configuration shards are never rewritten.

Acquisition fails closed: a required page that is incomplete (exhausted retries,
truncated or non-JSON body, configurations declared by ``availability`` but absent
from ``benchmarks``) or contradictory raises, so no snapshot is produced. A valid
empty response (``[]`` where nothing is declared) is a distinct, non-failure outcome.
"""

from __future__ import annotations

import email.utils
import hashlib
import http.client
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from aisim_core.data.e2e import (
    INFERENCEX_CONFIG_FIELDS,
    E2eConfigurationData,
    ProvenanceFile,
    SnapshotAcquisition,
    SnapshotMember,
    configuration_dir,
    inferencex_to_configurations,
    load_e2e,
    read_snapshot,
    snapshot_id,
    verify_snapshot_members,
    write_configuration,
    write_snapshot,
)
from aisim_core.data.hygiene import json_sidecar_value
from aisim_core.data.names import normalize

BASE_URL = "https://inferencex.semianalysis.com/api/v1/"
USER_AGENT = "aisim-infx-ingest/1"
SELECTION_POLICY = "inferencex-all-observations-in-scope/v1"
REQUIRED_ROW_KEYS = (*INFERENCEX_CONFIG_FIELDS, "id", "metrics", "isl", "osl", "benchmark_type")
# ``exact=true&date=D`` returns the logical snapshot as of D, so one benchmark id can reappear on later
# dates. Only these per-view context fields differ between appearances (verified against the live API);
# they are not measured content and are excluded from normalized records and contradiction checks.
# The original HTTP responses are retained unchanged as hash-linked provenance.
VIEW_CONTEXT_PREFIX = "curve_"
_AVAILABILITY_KEY = (
    "model", "hardware", "framework", "precision", "spec_method", "disagg", "benchmark_type", "isl", "osl",
)  # fmt: skip


class InfxError(Exception):
    """Base class; any subclass means the snapshot must not be produced."""


class TransientExhaustedError(InfxError):
    """Retries for a 429/5xx/transport failure were used up."""


class RateLimitedError(InfxError):
    """``Retry-After`` exceeds the cap or the request budget is exhausted."""


class HttpStatusError(InfxError):
    """A non-retryable status (for example 400/404)."""


class IncompleteAcquisitionError(InfxError):
    """A required page is truncated, malformed or missing declared configurations."""


class ContradictionError(InfxError):
    """Sources disagree (duplicate ids with different content, wrong model, suspicious empty)."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]  # lower-case names
    body: bytes


Transport = Callable[[str, Mapping[str, str], float], HttpResponse]


def urllib_transport(url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
    """Stdlib transport; HTTP error statuses are responses, network failures raise ``OSError``."""
    request = urllib.request.Request(url, headers=dict(headers))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(response.status, {k.lower(): v for k, v in response.headers.items()}, response.read())
    except urllib.error.HTTPError as exc:
        return HttpResponse(exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read())
    except urllib.error.URLError as exc:
        raise ConnectionError(str(exc.reason)) from exc


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 30.0
    max_retry_after: float = 120.0
    timeout: float = 60.0
    max_requests: int = 5000  # budget across a run, retries included


def parse_retry_after(value: str | None, now: float | None = None) -> float | None:
    """Seconds from a ``Retry-After`` value (delta-seconds or HTTP-date); ``None`` if absent or invalid."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, when.timestamp() - (time.time() if now is None else now))


@dataclass(frozen=True)
class Fetched:
    url: str
    status: int  # 200, or 304 when the cached body is still current
    body: bytes
    etag: str | None
    attempts: int


class InfxClient:
    def __init__(
        self,
        transport: Transport = urllib_transport,
        policy: RetryPolicy = RetryPolicy(),  # noqa: B008 - frozen
        sleep: Callable[[float], None] = time.sleep,
        base_url: str = BASE_URL,
    ):
        self.transport, self.policy, self.sleep, self.base_url = transport, policy, sleep, base_url
        self.requests_made = 0

    def url(self, route: str, params: Mapping[str, Any] | None = None) -> str:
        query = urlencode(sorted((params or {}).items()))
        return self.base_url + route + (f"?{query}" if query else "")

    def get(self, route: str, params: Mapping[str, Any] | None = None, *, etag: str | None = None) -> Fetched:
        url = self.url(route, params)
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json", "X-InferenceX-Traffic": "ci"}
        if etag:
            headers["If-None-Match"] = etag
        last = "no attempt"
        for attempt in range(1, self.policy.max_attempts + 1):
            if self.requests_made >= self.policy.max_requests:
                raise RateLimitedError(f"request budget of {self.policy.max_requests} exhausted before {url}")
            self.requests_made += 1
            delay = min(self.policy.max_delay, self.policy.base_delay * 2 ** (attempt - 1))
            try:
                response = self.transport(url, headers, self.policy.timeout)
            except (OSError, http.client.HTTPException) as exc:
                last = f"transport error: {exc!r}"
            else:
                status = response.status
                if status in (200, 304):
                    declared = response.headers.get("content-length")
                    if (
                        status == 200
                        and declared is not None
                        and declared.isdigit()
                        and int(declared) != len(response.body)
                    ):
                        last = f"truncated body ({len(response.body)} of {declared} bytes)"
                    else:
                        return Fetched(url, status, response.body, response.headers.get("etag"), attempt)
                elif status == 429 or 500 <= status < 600:
                    last = f"HTTP {status}"
                    asked = parse_retry_after(response.headers.get("retry-after"))
                    if asked is not None:
                        if asked > self.policy.max_retry_after:
                            raise RateLimitedError(
                                f"{url}: Retry-After {asked:.0f}s exceeds the {self.policy.max_retry_after:.0f}s cap"
                            )
                        delay = asked
                else:
                    raise HttpStatusError(f"{url}: HTTP {status}")
            if attempt < self.policy.max_attempts:
                self.sleep(delay)
        raise TransientExhaustedError(f"{url}: {self.policy.max_attempts} attempts failed ({last})")


# --------------------------------------------------------------------------- acquisition cache


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _utc(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class AcquisitionCache:
    """Mutable cache of raw response bytes keyed by request URL; never part of a snapshot."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _paths(self, url: str) -> tuple[Path, Path]:
        key = _sha(url.encode())
        return self.root / key[:2] / f"{key}.body", self.root / key[:2] / f"{key}.meta.json"

    def get(self, url: str) -> tuple[dict[str, Any], bytes] | None:
        body_path, meta_path = self._paths(url)
        try:
            meta = json_sidecar_value(meta_path.read_bytes())
            if (
                not isinstance(meta, dict)
                or meta.get("url") != url
                or not isinstance(meta.get("fetched_at"), str)
                or not meta["fetched_at"]
                or (meta.get("etag") is not None and not isinstance(meta["etag"], str))
                or type(meta.get("bytes")) is not int
            ):
                return None
            datetime.fromisoformat(meta["fetched_at"].replace("Z", "+00:00"))
            body = body_path.read_bytes()
            if meta.get("body_sha256") != _sha(body) or meta["bytes"] != len(body):
                return None
        except (OSError, ValueError, RecursionError):
            return None  # a torn or unreadable cache entry is replaceable, never truth
        return meta, body

    def put(self, url: str, body: bytes, etag: str | None, fetched_at: str) -> dict[str, Any]:
        body_path, meta_path = self._paths(url)
        body_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"url": url, "etag": etag, "body_sha256": _sha(body), "bytes": len(body), "fetched_at": fetched_at}
        for path, data in ((body_path, body), (meta_path, (json.dumps(meta, sort_keys=True) + "\n").encode())):
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                handle.write(data)
            os.replace(handle.name, path)  # body first, meta last: a torn write fails the hash check
        return meta


# --------------------------------------------------------------------------- acquisition


@dataclass
class Acquisition:
    records: list[dict[str, Any]]
    runs: dict[str, dict[str, Any]]
    scope: dict[str, Any]
    acquisitions: list[SnapshotAcquisition]
    empty_scope: bool = False  # valid-empty: nothing declared for the scope (not a failure)
    undeclared: list[str] = field(default_factory=list)  # benchmark rows availability did not list (warning)
    requests_made: int = 0
    responses: dict[str, bytes] = field(default_factory=dict)  # immutable original benchmark response evidence


def _json(fetched: Fetched, body: bytes, expect: type) -> Any:
    try:
        value = json_sidecar_value(body)
    except ValueError as exc:
        raise IncompleteAcquisitionError(f"{fetched.url}: body is not valid JSON ({exc})") from exc
    if not isinstance(value, expect):
        raise IncompleteAcquisitionError(f"{fetched.url}: expected {expect.__name__}, got {type(value).__name__}")
    return value


class Acquirer:
    def __init__(self, client: InfxClient, cache: AcquisitionCache, now: Callable[[], datetime] | None = None):
        self.client, self.cache = client, cache
        self.now = now or (lambda: datetime.now(UTC))
        self.responses: dict[str, bytes] = {}
        self.selected_models: set[str] | None = None  # canonical stored model IDs after applying a selector

    def _page(self, route: str, params: Mapping[str, Any] | None, expect: type) -> tuple[Any, str, str]:
        """Fetch with a conditional request; returns (value, acquired_at, status 'acquired'|'inherited')."""
        url = self.client.url(route, params)
        cached = self.cache.get(url)
        fetched = self.client.get(route, params, etag=cached[0].get("etag") if cached else None)
        if fetched.status == 304:
            if cached is None:
                raise IncompleteAcquisitionError(f"{url}: 304 without a cached body")
            self.responses[url] = cached[1]
            return _json(fetched, cached[1], expect), cached[0]["fetched_at"], "inherited"
        value = _json(fetched, fetched.body, expect)  # validate before it can enter the cache
        stamp = _utc(self.now())
        self.cache.put(url, fetched.body, fetched.etag, stamp)
        self.responses[url] = fetched.body
        return value, stamp, "acquired"

    def acquire(
        self,
        models: Sequence[str] | None = None,
        prior_observation_ids: Sequence[str] = (),
        prior_observation_models: Mapping[str, str] | None = None,
    ) -> Acquisition:
        self.responses = {}
        self.selected_models = None
        options, _, _ = self._page("views/options", None, dict)
        by_key: dict[str, str] = {}
        if not isinstance(options.get("models"), list):
            raise IncompleteAcquisitionError("views/options: models must be a list")
        for entry in options["models"]:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("name"), str)
                or not isinstance(entry.get("dbKeys"), list)
                or any(not isinstance(key, str) for key in entry["dbKeys"])
            ):
                raise IncompleteAcquisitionError("views/options: model entries need a string name and string dbKeys")
            for key in entry["dbKeys"]:
                if key in by_key and by_key[key] != entry["name"]:
                    raise ContradictionError(f"views/options: model {key!r} has conflicting display names")
                by_key[key] = entry["name"]
        availability, _, _ = self._page("availability", None, list)
        wanted = set(models) if models else None
        if wanted is not None:  # a misspelled selector must not read as "no data"
            known = set(by_key) | set(by_key.values())
            if unknown := sorted(wanted - known):
                raise InfxError(f"unknown model selector(s) {unknown}; known: {sorted(set(by_key.values()))}")
            # Stored observation model IDs are normalized DB keys, not display names.
            self.selected_models = {
                normalize("model", k) for k, name in by_key.items() if k in wanted or name in wanted
            }
            if prior_observation_models is not None:
                prior_observation_ids = [
                    oid for oid in prior_observation_ids if prior_observation_models.get(oid) in self.selected_models
                ]
        declared: dict[tuple[str, str], set[tuple]] = {}
        for row in availability:
            if not isinstance(row, dict) or not {*_AVAILABILITY_KEY, "date"} <= row.keys():
                raise IncompleteAcquisitionError(f"availability row lacks required fields: {row!r:.200}")
            _check_availability_values(row, "availability", include_date=True)
            if wanted is not None and row["model"] not in wanted and by_key.get(row["model"]) not in wanted:
                continue
            if row["model"] not in by_key:
                raise IncompleteAcquisitionError(f"availability model {row['model']!r} is not in views/options")
            declared.setdefault((by_key[row["model"]], row["date"]), set()).add(
                tuple(row[k] for k in _AVAILABILITY_KEY)
            )
        scope = {"source": "inferencex", "models": sorted(wanted) if wanted else "all"}
        if not declared:
            if prior_observation_ids:
                raise ContradictionError(
                    f"empty availability for scope but {len(prior_observation_ids)} observations were ingested before"
                )
            return Acquisition([], {}, scope, [], empty_scope=True, requests_made=self.client.requests_made)

        records: dict[str, dict[str, Any]] = {}
        acquired: dict[str, SnapshotAcquisition] = {}
        undeclared: list[str] = []
        for (name, date), expected in sorted(declared.items()):
            rows, stamp, status = self._page("benchmarks", {"model": name, "date": date, "exact": "true"}, list)
            url = self.client.url("benchmarks", {"model": name, "date": date, "exact": "true"})
            valid_keys = {k for k, n in by_key.items() if n == name}
            seen: set[tuple] = set()
            for row in rows:
                if not isinstance(row, dict) or any(k not in row for k in REQUIRED_ROW_KEYS):
                    raise IncompleteAcquisitionError(f"{url}: row lacks required fields: {row!r:.200}")
                _check_availability_values(row, url)
                if not isinstance(row["metrics"], dict):
                    raise IncompleteAcquisitionError(f"{url}: metrics must be a mapping")
                if row["model"] not in valid_keys:
                    raise ContradictionError(f"{url}: row {row['id']} is for model {row['model']!r}, not {name!r}")
                if wanted is not None and row["model"] not in wanted and name not in wanted:
                    continue  # display-name requests may also return unselected sibling DB keys
                row = {k: v for k, v in row.items() if not k.startswith(VIEW_CONTEXT_PREFIX)}
                rid = str(row["id"])
                if not rid.isascii() or not rid.isdigit():
                    raise IncompleteAcquisitionError(f"{url}: benchmark id must be numeric: {rid!r:.200}")
                if rid in records and _canonical(records[rid]) != _canonical(row):
                    raise ContradictionError(f"{url}: benchmark id {rid} has two different bodies in one acquisition")
                records[rid] = row
                acquired[rid] = SnapshotAcquisition(
                    observation_id=f"inferencex:{rid}", acquired_at=stamp, source_uri=url, status=status
                )
                seen.add(tuple(row[k] for k in _AVAILABILITY_KEY))
            missing = expected - seen
            if missing:
                raise IncompleteAcquisitionError(
                    f"{url}: availability declares {len(missing)} configuration(s) absent from benchmarks "
                    f"(returned {len(rows)} rows), for example {sorted(map(str, missing))[0]}"
                )
            if seen - expected:  # availability is a cached view and may lag benchmarks: warn, do not fail
                undeclared.append(f"{name}/{date}/{sorted(map(str, seen - expected))[0]}")

        runs = self._runs(sorted({date for _, date in declared}))
        return Acquisition(
            records=[records[k] for k in sorted(records, key=int)],
            runs=runs,
            scope=scope,
            acquisitions=[acquired[k] for k in sorted(acquired, key=int)],
            undeclared=undeclared,
            requests_made=self.client.requests_made,
            responses={
                url: body for url, body in self.responses.items() if "/benchmarks?" in url or "/workflow-info?" in url
            },
        )

    def _runs(self, dates: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Producer metadata keyed by run URL; a run whose ``head_sha`` differs between pages is a contradiction."""
        runs: dict[str, dict[str, Any]] = {}
        for date in dates:
            info, _, _ = self._page("workflow-info", {"date": date}, dict)
            if not all(isinstance(info.get(key), list) for key in ("runs", "runConfigs")):
                raise IncompleteAcquisitionError(f"workflow-info {date}: runs and runConfigs must be lists")
            for key in ("runs", "runConfigs"):
                if any(not isinstance(item, dict) or not isinstance(item.get("html_url"), str) for item in info[key]):
                    raise IncompleteAcquisitionError(f"workflow-info {date}: malformed {key} entry")
            for run in info.get("runs", []):
                entry = runs.setdefault(run["html_url"], {})
                attempt = run.get("run_attempt")
                if "run_attempt" in entry and entry["run_attempt"] != attempt:
                    raise ContradictionError(
                        f"workflow-info {date}: run {run['html_url']} reports two run_attempt values"
                    )
                entry["run_attempt"] = attempt
            for config in info.get("runConfigs", []):
                entry = runs.setdefault(config["html_url"], {})
                sha = config.get("head_sha")
                if sha and entry.get("head_sha") not in (None, sha):
                    raise ContradictionError(f"workflow-info {date}: run {config['html_url']} reports two head_sha")
                if sha:
                    entry["head_sha"] = sha
        return runs


# --------------------------------------------------------------------------- snapshot layer


@dataclass(frozen=True)
class SnapshotPlan:
    """The snapshot an acquisition would produce, computed without touching disk."""

    snapshot_id: str
    members: list[SnapshotMember]
    configurations: list[E2eConfigurationData]
    configuration_prefixes: dict[str, str]


def _configuration_prefix(data: E2eConfigurationData) -> str:
    # Include the converted rows and producer metadata, not just raw record hashes:
    # a normalization or workflow-info correction must retain the old truth too.
    content = {
        "body": data.body.model_dump(mode="json"),
        "tables": [
            [row.model_dump(mode="json") for row in rows]
            for rows in (data.observations, data.metrics, data.requests, data.series)
        ],
        "provenance": [{**vars(item), "data": _sha(item.data)} for item in data.provenance],
    }
    return f"revisions/{_sha(_canonical(content))}"


def plan_snapshot(acq: Acquisition) -> SnapshotPlan:
    configurations = inferencex_to_configurations(acq.records, acq.runs)
    # Keep original response bytes as well as the normalized, deduplicated record bundle.
    # These hash-linked artifacts survive removal of the mutable acquisition cache.
    originals: dict[str, list[ProvenanceFile]] = {}
    models = {data.body.model_id for data in configurations}
    for url, body in sorted(acq.responses.items()):
        artifact = ProvenanceFile(
            "raw-output", body, name=f"inferencex-response-{_sha(url.encode() + body)}", source_tool=url
        )
        response_models = (
            models
            if "/workflow-info?" in url
            else {normalize("model", row["model"]) for row in json_sidecar_value(body)}
        )
        if not response_models & models:
            raise IncompleteAcquisitionError(f"{url}: retained response matches no configuration")
        for model in response_models:
            originals.setdefault(model, []).append(artifact)
    configurations = [
        replace(data, provenance=[*data.provenance, *originals.get(data.body.model_id, [])]) for data in configurations
    ]
    prefixes = {data.body.configuration_id(): _configuration_prefix(data) for data in configurations}
    members = [
        SnapshotMember(
            configuration_id=o.configuration_id,
            observation_id=o.observation_id,
            record_sha256=o.record_sha256,
            manifest_path=f"{prefixes[o.configuration_id]}/{configuration_dir(data.body)}/manifest.json",
        )
        for data in configurations
        for o in data.observations
    ]
    return SnapshotPlan(snapshot_id(members, acq.scope, SELECTION_POLICY), members, configurations, prefixes)


def snapshot_path(sid: str) -> str:
    return f"snapshots/{sid}/snapshot.json"


def write_candidate(root: Path, acq: Acquisition, plan: SnapshotPlan) -> str:
    """Write shards and the immutable snapshot document; returns its path. An existing snapshot is never rewritten."""
    root = Path(root)
    path = snapshot_path(plan.snapshot_id)
    if (root / path).exists():
        snapshot, members = read_snapshot(root, path)
        if snapshot.snapshot_id != plan.snapshot_id:
            raise ValueError("existing snapshot ID differs from the planned snapshot")
        verify_snapshot_members(root, members)
        return path
    for data in plan.configurations:
        prefix = plan.configuration_prefixes[data.body.configuration_id()]
        manifest_path = f"{prefix}/{configuration_dir(data.body)}/manifest.json"
        if (root / manifest_path).exists():
            load_e2e(root, manifest_path)  # verify reused shards without replacing any of their bytes
        else:
            # The manifest is the last file in a multi-file shard. Publish the
            # entire immutable revision only after all writes and verification
            # succeed, so an interruption cannot poison the next identical run.
            with tempfile.TemporaryDirectory(dir=root) as tmp:
                staging = Path(tmp)
                write_configuration(staging, data, prefix=prefix)
                load_e2e(staging, manifest_path)
                destination = root / prefix
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.rename(staging / prefix, destination)
                except FileExistsError:
                    load_e2e(root, manifest_path)
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    pending = path + ".pending"
    try:
        snapshot = write_snapshot(
            root,
            pending,
            plan.members,
            source_scope=acq.scope,
            selection_policy=SELECTION_POLICY,
            completeness="complete",
            acquisition=acq.acquisitions,
        )
        os.replace(root / pending, root / path)
    finally:
        (root / pending).unlink(missing_ok=True)
    assert snapshot.snapshot_id == plan.snapshot_id
    return path


def diff_members(before: Mapping[str, str], after: Sequence[SnapshotMember]) -> dict[str, list[str]]:
    """Change summary against ``{observation_id: record_sha256}``: added, corrected, removed."""
    now = {m.observation_id: m.record_sha256 for m in after}
    return {
        "added": sorted(set(now) - set(before)),
        "corrected": sorted(o for o in now if o in before and now[o] != before[o]),
        "removed": sorted(set(before) - set(now)),
    }


def _check_availability_values(row: Mapping[str, Any], where: str, *, include_date: bool = False) -> None:
    for key in (*_AVAILABILITY_KEY, "date") if include_date else _AVAILABILITY_KEY:
        value = row.get(key)
        if key in ("isl", "osl"):
            valid = type(value) is int and value >= 0
        elif key == "disagg":
            valid = type(value) is bool
        elif key == "spec_method":
            valid = value is None or isinstance(value, str)
        else:
            valid = isinstance(value, str) and bool(value)
        if not valid:
            raise IncompleteAcquisitionError(f"{where}: invalid scalar value for {key}")
    if not include_date:
        for key in INFERENCEX_CONFIG_FIELDS:
            value = row.get(key)
            if key in {"is_multinode", "prefill_dp_attention", "decode_dp_attention"}:
                valid = value is None or type(value) is bool
            elif key.endswith(("_tp", "_ep", "_num_workers", "_gpu")):
                valid = value is None or (type(value) is int and value >= 0)
            else:
                continue  # the shared availability fields were checked above
            if not valid:
                raise IncompleteAcquisitionError(f"{where}: invalid deployment value for {key}")
