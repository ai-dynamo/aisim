# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable HF inputs must retain identities, checksums and workload semantics."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_agentx_accuracy import result, silicon_inputs

from scripts.e2e_accuracy import run_agentx_accuracy as agentx_run
from scripts.e2e_accuracy.agentx_contract import canonical, validate_catalog
from scripts.e2e_accuracy.agentx_inputs import (
    checked,
    finite_trace,
    load_runs,
    manual_geometry,
    reconstruct,
    replay_config,
    safe_path,
)
from scripts.e2e_accuracy.benchmark_features import SILICON_FEATURES, validate


def test_thin_catalog_and_relative_bundle_round_trip(tmp_path):
    catalog = silicon_inputs(tmp_path, result())
    assert validate_catalog(catalog) == catalog
    runs = load_runs(tmp_path, catalog)
    assert runs[0]["id"] == result()["id"]
    assert {
        r["id"]
        for r in json.loads(checked(tmp_path, runs[0]["artifacts"]["feature_metadata"]).read_bytes())["features"]
    } == SILICON_FEATURES
    manifest = checked(tmp_path, catalog["runs"][0]["manifest"])
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        load_runs(tmp_path, catalog)


@pytest.mark.parametrize("path", ["../escape.json", "/absolute", "a//b", "a/./b", "a/../b", "a\\b"])
def test_unsafe_bundle_paths_rejected(tmp_path, path):
    with pytest.raises(ValueError, match="Unsafe"):
        safe_path(tmp_path, path)


def test_symlink_escape_rejected(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe"):
        safe_path(tmp_path, "link/escape.json")


@pytest.mark.parametrize("child_link", [False, True])
def test_trusted_root_with_symlinked_ancestor(tmp_path, child_link):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    root = alias / "inputs"
    root.mkdir()
    target = root / "bundle"
    target.mkdir()
    if child_link:
        (root / "link").symlink_to(target, target_is_directory=True)
        with pytest.raises(ValueError, match="Unsafe"):
            safe_path(root, "link/manifest.json")
    else:
        assert safe_path(root, "bundle/manifest.json") == target / "manifest.json"


@pytest.mark.parametrize("root_link", [False, True])
def test_checked_and_load_runs_through_trusted_symlink(tmp_path, root_link):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    root = alias if root_link else alias / "inputs"
    root.mkdir(exist_ok=True)
    catalog = silicon_inputs(root, result())
    run = load_runs(root, catalog)[0]
    assert checked(root, run["artifacts"]["feature_metadata"]).is_file()


@pytest.mark.parametrize("directory", [False, True])
def test_in_bundle_symlinks_rejected_even_with_contained_targets(tmp_path, directory):
    target = tmp_path / "actual"
    target.mkdir() if directory else target.write_text("intact")
    (tmp_path / "link").symlink_to(target, target_is_directory=directory)
    with pytest.raises(ValueError, match="Unsafe"):
        safe_path(tmp_path, "link/manifest.json" if directory else "link")


def test_corpus_filter_preserves_original_bytes_and_order(tmp_path):
    source = tmp_path / "source.jsonl"
    lines = [
        b'{"id":"small", "requests":[{"in":4,"out":2}]}\n',
        b'{"id":"large","requests":[{"in":9,"out":2}]}\n',
    ]
    source.write_bytes(b"".join(lines))
    descriptor = {
        "repository": "org/corpus",
        "revision": "a" * 40,
        "filename": "traces.jsonl",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "derived_sha256": hashlib.sha256(lines[0]).hexdigest(),
        "transformations": [
            {
                "operation": "retain_complete_traces",
                "maximum_tokens": 8,
                "eligible": ["small"],
                "excluded": ["large"],
                "serialization": "original_jsonl_lines",
            }
        ],
    }
    target = tmp_path / "filtered.jsonl"
    reconstruct(descriptor, source, target)
    assert target.read_bytes() == lines[0]
    descriptor["transformations"][0]["eligible"] = ["invented"]
    with pytest.raises(ValueError, match="reviewed workload"):
        reconstruct(descriptor, source, target)
    assert not target.exists()


def test_finite_trace_preserves_token_equality_and_warmup_barrier(tmp_path):
    rows = [
        {
            "request_id": identity,
            "cell_id": "case",
            "lane_id": 0,
            "turn_index": turn,
            "phase": phase,
            "input_token_ids": prompt,
            "output_token_ids": output,
            "dependency_ids": [],
            "think_time_ms": 100,
        }
        for identity, turn, phase, prompt, output in [
            ("warm", -1, "warmup", [80, 90], [70]),
            ("measured", 0, "measured", [80, 90, 70], [60]),
            ("kernel", -2, "kernel_warmup", [1], [2]),
        ]
    ]
    raw = tmp_path / "raw.jsonl"
    raw.write_bytes(b"".join(canonical(r) for r in rows))
    run = {
        "model": "test/model",
        "artifacts": {
            "raw": {
                "path": raw.name,
                "bytes": raw.stat().st_size,
                "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            }
        },
    }
    target = tmp_path / "trace.jsonl"
    finite_trace(tmp_path, run, target)
    records = [json.loads(line) for line in target.read_text().splitlines()][1:]
    assert {r["request_id"] for r in records} == {"warm", "measured"}
    measured = next(r for r in records if r["request_id"] == "measured")
    warm = next(r for r in records if r["request_id"] == "warm")
    assert measured["hash_ids"] == [80, 90, 70]
    assert warm["output_token_ids"] == [2]
    assert measured["dependencies"] == [
        {
            "request_id": "warm",
            "trigger": "completion",
            "delay_ms": 100,
            "relation": "sequence",
        }
    ]


def test_curated_features_keep_actual_draft_length_and_unknown_scope(tmp_path):
    catalog = silicon_inputs(tmp_path, result())
    run = load_runs(tmp_path, catalog)[0]
    value = json.loads(checked(tmp_path, run["artifacts"]["feature_metadata"]).read_bytes())
    row = next(r for r in value["features"] if r["id"] == "speculation")
    row.update(
        state="enabled",
        parameters=[
            {"name": "method", "value": "dspark"},
            {"name": "draft_length", "value": 7},
        ],
        evidence=[{"sha256": "a" * 64, "pointer": "/runtime"}],
    )
    assert validate(copy.deepcopy(value)) == value
    assert next(r for r in value["features"] if r["id"] == "cache.g2")["parameters"] == [
        {"name": "scope", "value": "unknown"}
    ]
    row["parameters"][0]["value"] = "off"
    with pytest.raises(ValueError, match="not applicable"):
        validate(value)


def test_partial_measurement_snapshot_cannot_pass_public_gate():
    from test_agentx_accuracy import bundle

    from scripts.pages.agentx_pages import validate_snapshot

    snapshot = json.loads(bundle()["summary.json"])["snapshot"]
    snapshot["measurement_campaign_status"] = "partial"
    with pytest.raises(ValueError, match="require a preview"):
        validate_snapshot(snapshot)
    snapshot["preview"] = True
    validate_snapshot(snapshot, preview=True)
    snapshot["measurement_campaign_status"] = "complete"
    with pytest.raises(ValueError, match="require a preview"):
        validate_snapshot(snapshot, preview=True)


def sglang_replay_inputs(tmp_path, *, depth=3, acceptance=None):
    run = result()
    run["artifacts"] = {}
    documents = {
        "recipe": {
            "base": {
                "roles": {
                    "agg": {
                        "args": {
                            "context-length": 8192,
                            "chunked-prefill-size": 512,
                            "max-running-requests": 32,
                            "speculative-num-steps": depth,
                        }
                    }
                }
            }
        },
        "feature_point_receipt": {
            "metrics": {
                "kv_cache_pool_tokens": 8192,
                **({"speculative_acceptance": acceptance} if acceptance is not None else {}),
            }
        },
        "source_manifest": {
            "initial_selection": {"seed": 42, "corpus_cursor": 1, "source_play_id": "trace", "t_star_ms": 0}
        },
        "workload": {"silicon_reference": {"namespace_map": {"trace": "weka:1111111111111111"}}},
        "feature_harness_receipt": {
            "input_config": {"phases": [{"duration": 60, "system_idle_gap_cap_seconds": 2, "grace_period": 5}]}
        },
    }
    for role, value in documents.items():
        # Nonfinite fixture values exercise validation after checksum verification.
        content = json.dumps(value).encode()
        target = tmp_path / f"{role}.json"
        target.write_bytes(content)
        run["artifacts"][role] = {
            "path": target.name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
    return run


@pytest.mark.parametrize("accepted", [1.25, 2.75, None, True, -1, 4, float("inf"), float("nan"), "1.99"])
@pytest.mark.parametrize("depth", [2, 3])
def test_sglang_replay_uses_only_depth_matched_retained_acceptance(tmp_path, accepted, depth):
    receipt = {"verification_depth": depth, "accepted_draft_tokens": accepted}
    run = sglang_replay_inputs(tmp_path, acceptance=receipt if accepted is not None else None)
    if depth == 3 and type(accepted) is float and accepted in (1.25, 2.75):
        engine = replay_config(tmp_path, run)["engine"]
        assert engine["nextn"] == 3
        assert engine["nextn_accepted"] == accepted
    else:
        with pytest.raises(ValueError, match="acceptance"):
            replay_config(tmp_path, run)


def test_non_speculative_replay_validates_without_acceptance(tmp_path):
    run = sglang_replay_inputs(tmp_path, depth=0)
    config = replay_config(tmp_path, run)
    # The predictor consumes this selector before parsing the public config.
    config["engine"]["backend_version"] = config.pop("prediction_database_version")
    # Pages CI runs without an installed AISim wheel. Validate the real source
    # models in a subprocess without importing the runtime into the test worker.
    source = Path(__file__).resolve().parents[1] / "python/aisim/src"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from aisim.config import CorePredictionConfig\n"
            "config = CorePredictionConfig.model_validate_json(sys.stdin.read())\n"
            "assert config.engine.nextn == 0\n"
            "assert config.engine.nextn_accepted is None\n"
            "assert CorePredictionConfig.model_validate_json(config.model_dump_json()) == config\n",
        ],
        input=json.dumps(config),
        text=True,
        env={**os.environ, "PYTHONPATH": str(source)},
        check=True,
    )


def hub_replay_inputs(tmp_path, monkeypatch, acceptance):
    hub = tmp_path / "hub"
    catalog = silicon_inputs(hub, result())
    manifest_path = hub / catalog["runs"][0]["manifest"]["path"]
    manifest = json.loads(manifest_path.read_bytes())
    replay = tmp_path / "replay"
    replay.mkdir()
    extra = sglang_replay_inputs(replay, acceptance=acceptance)

    def put(role, value):
        content = canonical(value)
        path = manifest_path.parent / f"{role}.json"
        path.write_bytes(content)
        manifest["artifacts"][role] = {
            "path": path.name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }

    for role, entry in extra["artifacts"].items():
        if role != "workload":
            put(role, json.loads((replay / entry["path"]).read_bytes()))
    workload = json.loads((manifest_path.parent / manifest["artifacts"]["workload"]["path"]).read_bytes())
    workload.update(
        corpus_id="test-corpus",
        silicon_reference={"namespace_map": {"trace": "weka:1111111111111111"}},
    )
    put("workload", workload)
    manifest["corpus_id"] = "test-corpus"
    content = canonical(manifest)
    manifest_path.write_bytes(content)
    catalog["runs"][0]["manifest"].update(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    del catalog["snapshot_id"]
    catalog["snapshot_id"] = hashlib.sha256(canonical(catalog)).hexdigest()
    (hub / agentx_run.INDEX).write_bytes(canonical(catalog))

    corpus = tmp_path / "upstream.jsonl"
    corpus.write_bytes(b'{"id":"trace","requests":[{"in":4,"out":2}]}\n')
    checksum = hashlib.sha256(corpus.read_bytes()).hexdigest()
    registry = {
        "schema": "aisimulate.agentx.corpora.v1",
        "sources": {
            "test-corpus": {
                "repository": "org/corpus",
                "revision": "b" * 40,
                "filename": "traces.jsonl",
                "sha256": checksum,
                "derived_sha256": checksum,
                "transformations": [],
            }
        },
    }
    (hub / agentx_run.CORPORA).write_bytes(canonical(registry))
    downloads = []

    def download(repo, name, **kwargs):
        downloads.append((repo, name))
        return str(hub / name) if repo == agentx_run.REPO else str(corpus)

    monkeypatch.setattr(agentx_run, "hf_hub_download", download)
    return downloads, corpus


@pytest.mark.parametrize("acceptance", [None, {"verification_depth": 2, "accepted_draft_tokens": 1.25}])
def test_fetch_preflights_acceptance_before_downloading_corpus(tmp_path, monkeypatch, acceptance):
    downloads, _ = hub_replay_inputs(tmp_path, monkeypatch, acceptance)
    output = tmp_path / "inputs"
    with pytest.raises(ValueError, match="agentx-446063.*acceptance"):
        agentx_run.fetch("a" * 40, output)
    assert all(repo == agentx_run.REPO for repo, _ in downloads)
    assert not (output / "corpora").exists()


@pytest.mark.parametrize("input_error", ["missing", "null_base", "null_override", "empty_phases"])
def test_fetch_reports_malformed_input_and_later_acceptance(tmp_path, monkeypatch, input_error):
    downloads, _ = hub_replay_inputs(tmp_path, monkeypatch, None)
    hub = tmp_path / "hub"
    catalog = json.loads((hub / agentx_run.INDEX).read_bytes())
    first_path = hub / catalog["runs"][0]["manifest"]["path"]
    first = json.loads(first_path.read_bytes())
    second_path = first_path.parent.parent / "agentx-446068" / first_path.name
    second_path.parent.mkdir()
    for source in first_path.parent.iterdir():
        if source.is_file():
            (second_path.parent / source.name).write_bytes(source.read_bytes())
    second = copy.deepcopy(first)
    second.update(id="agentx-446068", lanes=4)
    for role in ("feature_metadata", "workload"):
        entry = second["artifacts"][role]
        path = second_path.parent / entry["path"]
        value = json.loads(path.read_bytes())
        value["benchmark_id"] = second["id"]
        path.write_bytes(canonical(value))
        entry.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    if input_error == "empty_phases":
        for role in ("feature_point_receipt", "feature_harness_receipt"):
            entry = first["artifacts"][role]
            path = first_path.parent / entry["path"]
            value = json.loads(path.read_bytes())
            if role == "feature_point_receipt":
                value["metrics"]["speculative_acceptance"] = {"verification_depth": 3, "accepted_draft_tokens": 1.25}
            else:
                value["input_config"]["phases"] = []
            path.write_bytes(canonical(value))
            entry.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    elif input_error == "missing":
        del first["artifacts"]["recipe"]
    else:
        entry = first["artifacts"]["recipe"]
        path = first_path.parent / entry["path"]
        recipe = json.loads(path.read_bytes())
        recipe["base" if input_error == "null_base" else "override_c1"] = None
        path.write_bytes(canonical(recipe))
        entry.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    catalog["runs"] = []
    for manifest, path in ((first, first_path), (second, second_path)):
        path.write_bytes(canonical(manifest))
        entry = {key: manifest[key] for key in ("id", "category", "configuration_id")}
        entry["manifest"] = {
            "path": path.relative_to(hub).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        catalog["runs"].append(entry)
    del catalog["snapshot_id"]
    catalog["snapshot_id"] = hashlib.sha256(canonical(catalog)).hexdigest()
    (hub / agentx_run.INDEX).write_bytes(canonical(catalog))
    with pytest.raises(ValueError) as caught:
        agentx_run.fetch("a" * 40, tmp_path / "inputs")
    assert "agentx-446063:" in str(caught.value)
    assert "agentx-446068: Missing depth-matched measured speculative acceptance" in str(caught.value)
    assert all(repo == agentx_run.REPO for repo, _ in downloads)


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_replay_rejects_empty_recorded_phases(tmp_path, cache_workers, backend):
    run = (
        sglang_replay_inputs(tmp_path, acceptance={"verification_depth": 3, "accepted_draft_tokens": 1.25})
        if backend == "sglang"
        else result(
            backend="vllm",
            measured_backend_version="0.31.0",
            model="Qwen/Qwen3.5-397B-A17B",
            precision="BF16",
            corpus_id="test-corpus",
            artifacts={},
        )
    )

    def put(role, value):
        path = tmp_path / f"{role}.json"
        content = canonical(value)
        path.write_bytes(content)
        run["artifacts"][role] = {
            "path": path.name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }

    if backend == "sglang":
        put("feature_harness_receipt", {"input_config": {"phases": []}})
    else:
        for worker in cache_workers:
            worker["model"]["max_model_len"] = 8192
            worker["scheduler"].update(max_num_seqs=32, max_num_batched_tokens=512)
        put("cache_receipt", {"workers": cache_workers})
        put("workload", {"silicon_reference": {"namespace_map": {"trace": "weka:1111111111111111"}}})
        put("snapshot_receipt", [{"lane_index": 0, "root_id": "trace", "snapshot": {"t_star_ms": 0}}])
        put("harness_receipt", {"phases": [], "datasets": [{"random_seed": 42}]})
        registry = tmp_path / agentx_run.CORPORA
        registry.parent.mkdir(parents=True)
        registry.write_bytes(canonical({"sources": {"test-corpus": {"transformations": []}}}))
    with pytest.raises(ValueError, match="phases"):
        replay_config(tmp_path, run)


def test_fetch_preflights_valid_inputs_and_preserves_corpus(tmp_path, monkeypatch):
    downloads, corpus = hub_replay_inputs(
        tmp_path, monkeypatch, {"verification_depth": 3, "accepted_draft_tokens": 1.25}
    )
    output = tmp_path / "inputs"
    agentx_run.fetch("a" * 40, output)
    assert ("org/corpus", "traces.jsonl") in downloads
    assert (output / "corpora/test-corpus.jsonl").read_bytes() == corpus.read_bytes()


def test_metadata_fetch_retains_invalid_inputs_for_diagnosis(tmp_path, monkeypatch):
    downloads, _ = hub_replay_inputs(tmp_path, monkeypatch, None)
    output = tmp_path / "inputs"
    agentx_run.fetch("a" * 40, output, metadata_only=True)
    assert len(load_runs(output, json.loads((output / agentx_run.INDEX).read_bytes()))) == 1
    assert all(repo == agentx_run.REPO for repo, _ in downloads)


def test_metadata_fetch_retains_catalog_when_workload_validation_fails(tmp_path, monkeypatch):
    downloads, _ = hub_replay_inputs(tmp_path, monkeypatch, None)
    hub = tmp_path / "hub"
    catalog_path = hub / agentx_run.INDEX
    catalog = json.loads(catalog_path.read_bytes())
    manifest_entry = catalog["runs"][0]["manifest"]
    manifest_path = hub / manifest_entry["path"]
    manifest = json.loads(manifest_path.read_bytes())
    workload_entry = manifest["artifacts"]["workload"]
    workload_path = manifest_path.parent / workload_entry["path"]
    workload = json.loads(workload_path.read_bytes())
    workload["schema"] = "invalid-workload-schema"
    workload_path.write_bytes(canonical(workload))
    workload_entry.update(
        sha256=hashlib.sha256(workload_path.read_bytes()).hexdigest(), bytes=workload_path.stat().st_size
    )
    manifest_path.write_bytes(canonical(manifest))
    manifest_entry.update(
        sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(), bytes=manifest_path.stat().st_size
    )
    del catalog["snapshot_id"]
    catalog["snapshot_id"] = hashlib.sha256(canonical(catalog)).hexdigest()
    catalog_path.write_bytes(canonical(catalog))

    output = tmp_path / "inputs"
    with pytest.raises(ValueError, match="Silicon workload identity mismatch"):
        agentx_run.fetch("a" * 40, output, metadata_only=True)
    assert (output / agentx_run.INDEX).read_bytes() == catalog_path.read_bytes()
    assert (output / workload_path.relative_to(hub)).read_bytes() == workload_path.read_bytes()
    assert all(repo == agentx_run.REPO for repo, _ in downloads)


def test_rejected_replay_does_not_reconstruct_trace(tmp_path, monkeypatch):
    hub_replay_inputs(tmp_path, monkeypatch, None)
    root = tmp_path / "inputs"
    agentx_run.fetch("a" * 40, root, metadata_only=True)
    run = load_runs(root, json.loads((root / agentx_run.INDEX).read_bytes()))[0]
    monkeypatch.setattr(
        agentx_run,
        "prepare_trace",
        lambda *args: pytest.fail("Reconstructed a rejected trace"),
    )
    with pytest.raises(ValueError, match="acceptance"):
        agentx_run.run_one(root, run, tmp_path / "output", 1, {})


@pytest.fixture
def cache_workers():
    groups = [
        {
            "layer_names": [f"layer-{g * 15 + i}" for i in range(15)],
            "spec_type": "FullAttentionSpec" if g == 0 else "MambaSpec",
            "page_size_bytes": 256,
            "spec": {"block_size": 64, **({"mamba_cache_mode": "align"} if g else {})},
        }
        for g in range(4)
    ]
    worker = {
        "vllm_version": "0.31.0",
        "speculative_config": None,
        "kv_transfer_config": None,
        "model": {"model": "Qwen/Qwen3.5-397B-A17B"},
        "parallelism": {
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "prefill_context_parallel_size": 1,
            "decode_context_parallel_size": 1,
        },
        "scheduler": {
            "async_scheduling": False,
            "enable_chunked_prefill": True,
            "disable_hybrid_kv_cache_manager": False,
        },
        "cache_config": {
            "block_size": 64,
            "mamba_cache_mode": "align",
            "enable_prefix_caching": True,
            "prefix_match_unit": None,
            "use_replayssm": False,
        },
        "prefix_cache_retention_interval": 0,
        "num_blocks": 10,
        "groups": groups,
        "tensors": [
            {"size": 38400, "offset": 0, "layers": g["layer_names"], "layer_stride": 2560, "block_stride": 256}
            for g in groups
        ],
    }
    return [dict(copy.deepcopy(worker), rank=rank) for rank in range(8)]


def test_qwen_geometry_retains_shared_pool_and_reserved_blocks(cache_workers):
    assert manual_geometry(cache_workers) == {
        "block_size": 64,
        "prefix_match_unit": 64,
        "bytes_per_token": 60,
        "prefix_caching": True,
        "capacity": {"type": "fixed", "blocks": 9},
        "state_cache": {"bytes_per_request": 11520},
    }


@pytest.mark.parametrize(
    "drift", ["missing_rank", "duplicate_rank", "out_of_range_rank", "bool_rank", "capacity", "scheduler", "tensor"]
)
def test_qwen_geometry_rejects_rank_or_allocation_drift(cache_workers, drift):
    if drift == "missing_rank":
        cache_workers.pop()
    elif drift == "duplicate_rank":
        cache_workers[-1]["rank"] = 0
    elif drift == "out_of_range_rank":
        cache_workers[-1]["rank"] = 8
    elif drift == "bool_rank":
        cache_workers[0]["rank"] = False
    elif drift == "capacity":
        cache_workers[-1]["num_blocks"] += 1
    elif drift == "scheduler":
        cache_workers[-1]["scheduler"]["max_num_seqs"] = 32
    else:
        cache_workers[-1]["tensors"][0]["size"] += 1
    with pytest.raises(ValueError):
        manual_geometry(cache_workers)
