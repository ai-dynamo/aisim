# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport, concurrent cache replacement and pre-selection download accounting."""

import json
import sys
from types import SimpleNamespace

import pytest

from aisim_core.data import fetch as fetch_module
from aisim_core.data.fetch import (
    BudgetExceededError,
    Cache,
    HfTransport,
    IntegrityError,
    TransferError,
    fetch,
    fetch_support,
    load_category_index,
    plan,
)
from aisim_core.data.fpm_layout import physical_path
from aisim_core.data.repos import PUBLIC_REPO, RETIRED_FPM_REPO
from tests.unit.data.test_fetch import REPO, SEL, SHA, FakeTransport, _artifact_json, _repin, _sha, build_repo

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("kind", ["artifact", "metadata", "support"])
def test_a_failed_concurrent_replacement_keeps_the_verified_winner(tmp_path, monkeypatch, kind):
    files = build_repo(tmp_path)
    cache = Cache(tmp_path / "cache")
    if kind == "metadata":
        victim = "catalog/dataset.json"
    elif kind == "artifact":
        victim = next(p for p in files if p.endswith(".parquet"))
    else:
        victim = "policy.json"
        files[victim] = b'{"policy": "pinned"}'
        index = json.loads(files["catalog/fpm.json"])
        index["support"] = [_artifact_json("policy", victim, files[victim], format="json", schema="s/v1")]
        files["catalog/fpm.json"] = json.dumps(index).encode()
        _repin(files, "catalog/fpm.json")
    real_valid = cache.valid
    raced = False

    def another_caller_wins(repo, sha, path, expected):
        nonlocal raced
        if path == victim and not raced:
            raced = True
            winner = cache.path_for(repo, sha, path)
            winner.parent.mkdir(parents=True, exist_ok=True)
            winner.write_bytes(files[path])
            return False  # this caller had already planned a miss
        return real_valid(repo, sha, path, expected)

    monkeypatch.setattr(cache, "valid", another_caller_wins)
    transport = FakeTransport(files, fail_after={victim: 2})
    with pytest.raises((TransferError, ConnectionError)):
        if kind == "support":
            fetch_support("fpm", repo=REPO, revision=SHA, transport=transport, cache=cache)
        else:
            fetch(SEL, repo=REPO, revision=SHA, transport=transport, cache=cache)
    assert raced and cache.path_for(REPO, SHA, victim).read_bytes() == files[victim]


def _successor_files(files):
    index = json.loads(files["catalog/fpm.json"])
    moved = {}
    for entry in index["entries"]:
        old_manifest = entry["manifest"]["path"]
        manifest = json.loads(files.pop(old_manifest))
        for artifact in manifest["artifacts"]:
            old = artifact["path"]
            logical = "data/" + old.removeprefix("fpm/")
            artifact["path"] = logical
            moved[physical_path(logical)] = files.pop(old)
        logical_manifest = "data/" + old_manifest.removeprefix("fpm/")
        content = json.dumps(manifest).encode()
        moved[physical_path(logical_manifest)] = content
        entry["manifest"].update(path=logical_manifest, sha256=_sha(content), bytes=len(content))
    files.update(moved)
    files["catalog/fpm.json"] = json.dumps(index).encode()
    _repin(files, "catalog/fpm.json")
    return files


def test_successor_alias_maps_only_transport_paths_and_preserves_logical_keys(tmp_path):
    files = _successor_files(build_repo(tmp_path))
    cache = Cache(tmp_path / "cache")
    transport = FakeTransport(files)
    planned = plan(SEL, repo=RETIRED_FPM_REPO, revision="main", transport=transport, cache=cache)
    assert planned.repo == PUBLIC_REPO
    assert not any(f.path.startswith("data/fpm/") for f in planned.files)
    result = fetch(SEL, repo=RETIRED_FPM_REPO, revision="main", transport=transport, cache=cache)
    assert all(path in files for path in transport.opened)
    assert any(path.startswith("data/fpm/") for path in transport.opened)
    assert set(result.paths) == {f.path for f in result.plan.files}
    assert all(path.is_file() for path in result.paths.values())
    # A full historical SHA stays in its original repository and keeps its original transport paths.
    historical = {p.replace("data/fpm/", "data/"): data for p, data in files.items()}
    direct = FakeTransport(historical)
    old = fetch(SEL, repo=RETIRED_FPM_REPO, revision=SHA, transport=direct, cache=Cache(tmp_path / "historical"))
    assert old.plan.repo == RETIRED_FPM_REPO
    assert all(not path.startswith("data/fpm/") for path in direct.opened)


@pytest.mark.parametrize("already_resolved", [False, True])
def test_successor_mapping_leaves_other_category_paths_alone(tmp_path, already_resolved):
    from aisim_core.data.fetch import _resolve_source

    path = "data/op-latency/policy.json"
    files = {path: b'{"policy": "pinned"}'}
    index = {
        "schema_version": "aisim-category-index/v1",
        "category": "op-latency",
        "entries": [],
        "support": [_artifact_json("policy", path, files[path], format="json", schema="s/v1")],
    }
    files["catalog/op.json"] = json.dumps(index).encode()
    dataset = {
        "schema_version": "aisim-dataset-index/v1",
        "dataset_id": PUBLIC_REPO,
        "categories": {
            "op-latency": {
                "root": "data/op-latency",
                "layout": "category",
                "index": _artifact_json(
                    "index", "catalog/op.json", files["catalog/op.json"], format="json", schema="s/v1"
                ),
            }
        },
    }
    files["catalog/dataset.json"] = json.dumps(dataset).encode()
    transport = FakeTransport(files)
    if already_resolved:  # ensure_data shares this adapter across FPM and op-latency fetches
        repo, revision, adapted = _resolve_source(RETIRED_FPM_REPO, "main", transport)
        fetch_support("op-latency", repo=repo, revision=revision, transport=adapted, cache=Cache(tmp_path / "cache"))
    else:
        fetch_support(
            "op-latency", repo=RETIRED_FPM_REPO, revision="main", transport=transport, cache=Cache(tmp_path / "cache")
        )
    assert path in transport.opened


def test_support_fetch_applies_one_budget_before_indexes_and_support(tmp_path):
    files = build_repo(tmp_path)
    files["policy.json"] = b'{"policy": "pinned"}'
    index = json.loads(files["catalog/fpm.json"])
    index["support"] = [_artifact_json("policy", "policy.json", files["policy.json"], format="json", schema="s/v1")]
    files["catalog/fpm.json"] = json.dumps(index).encode()
    _repin(files, "catalog/fpm.json")
    cache = Cache(tmp_path / "cache")
    transport = FakeTransport(files)
    with pytest.raises(BudgetExceededError):
        fetch_support("fpm", repo=REPO, revision=SHA, transport=transport, cache=cache, max_bytes=0)
    assert transport.opened == []
    indexes_size = sum(len(files[p]) for p in ("catalog/dataset.json", "catalog/fpm.json"))
    with pytest.raises(BudgetExceededError):
        fetch_support("fpm", repo=REPO, revision=SHA, transport=transport, cache=cache, max_bytes=indexes_size)
    assert "policy.json" not in transport.opened
    fetch_support("fpm", repo=REPO, revision=SHA, transport=transport, cache=cache, max_bytes=len(files["policy.json"]))
    transport.opened.clear()
    fetch_support("fpm", repo=REPO, revision=SHA, transport=transport, cache=cache, max_bytes=0)
    assert transport.opened == []


class SizedTransport(FakeTransport):
    """A transport that, like Hugging Face, can report a file's size but cannot stop a transfer once it starts."""

    def file_size(self, repo, sha, path):
        return len(self.files[path])


def test_a_file_without_a_declared_size_is_budgeted_before_the_transfer_starts(tmp_path):
    files = build_repo(tmp_path)
    catalog = "catalog/dataset.json"  # the root catalog declares no size
    transport = SizedTransport(files)
    with pytest.raises(BudgetExceededError):
        load_category_index(
            "fpm",
            repo=REPO,
            revision=SHA,
            transport=transport,
            cache=Cache(tmp_path / "cache"),
            max_bytes=len(files[catalog]) - 1,
        )
    assert transport.opened == []  # nothing was requested, so nothing was downloaded
    # A warm cache needs no probe and no transfer.
    cache = Cache(tmp_path / "warm")
    load_category_index("fpm", repo=REPO, revision=SHA, transport=SizedTransport(files), cache=cache)
    quiet = SizedTransport(files)
    quiet.file_size = lambda *a: pytest.fail("a cache hit must not probe the transport")
    load_category_index("fpm", repo=REPO, revision=SHA, transport=quiet, cache=cache)
    assert quiet.opened == []


def test_a_file_reported_over_the_metadata_cap_is_refused_before_the_transfer_starts(tmp_path, monkeypatch):
    transport = SizedTransport(build_repo(tmp_path))
    monkeypatch.setattr(fetch_module, "_METADATA_CAP_BYTES", 10)
    with pytest.raises(IntegrityError, match="over the 10-byte cap"):
        load_category_index("fpm", repo=REPO, revision=SHA, transport=transport, cache=Cache(tmp_path / "cache"))
    assert transport.opened == []


def test_the_hub_transport_reports_file_sizes_from_repository_metadata(monkeypatch):
    calls = []

    class FakeApi:
        def __init__(self, token=None):
            calls.append(("token", token))

        def get_paths_info(self, repo, paths, *, repo_type, revision):
            calls.append((repo, tuple(paths), repo_type, revision))
            if paths == ["broken"]:
                raise ConnectionError("hub unavailable")
            return [SimpleNamespace(size=None if paths == ["folder"] else 1234)]

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    transport = HfTransport(token=False)
    assert transport.file_size(REPO, SHA, "catalog/dataset.json") == 1234
    assert (REPO, ("catalog/dataset.json",), "dataset", SHA) in calls and ("token", False) in calls
    assert transport.file_size(REPO, SHA, "folder") is None  # no size reported
    assert transport.file_size(REPO, SHA, "broken") is None  # a failed probe loses only the early check


def test_the_successor_alias_probes_the_physical_path(tmp_path):
    from aisim_core.data.fetch import _AliasTransport
    from aisim_core.data.repos import resolve_ref

    probed = []

    class Probe(FakeTransport):
        def file_size(self, repo, sha, path):
            probed.append(path)
            return 1

    reference = resolve_ref(RETIRED_FPM_REPO, "main")
    assert _AliasTransport(Probe({}), reference).file_size(PUBLIC_REPO, SHA, "data/m1/manifest.json") == 1
    assert probed == [physical_path("data/m1/manifest.json")]
    assert _AliasTransport(FakeTransport({}), reference).file_size(PUBLIC_REPO, SHA, "data/m1/manifest.json") is None
