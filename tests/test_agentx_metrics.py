# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentX agreed metrics, learned-FPM step inputs, and their publication contract."""

import gzip
import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_agentx_accuracy import bundle, observation, result, silicon_inputs, simulation_record

from scripts.e2e_accuracy import agentx_learned
from scripts.e2e_accuracy.agentx_contract import (
    canonical,
    detail_path,
    driver_digest,
    score,
    validate_result,
)
from scripts.e2e_accuracy.agentx_metrics import (
    percentile,
    point_metrics,
    quantile_error,
    request_metrics,
    step_accumulator,
    step_summary,
)
from scripts.e2e_accuracy.run_agentx_accuracy import apply_performance_model, cpu_overhead, performance_models


def pair(silicon_tpot, predicted_tpot, silicon_ttft=100.0, predicted_ttft=120.0):
    return {
        "silicon_tpot_ms": silicon_tpot,
        "predicted_tpot_ms": predicted_tpot,
        "tpot_ms_ape": abs(predicted_tpot / silicon_tpot - 1) * 100,
        "silicon_ttft_ms": silicon_ttft,
        "predicted_ttft_ms": predicted_ttft,
        "ttft_ms_ape": abs(predicted_ttft / silicon_ttft - 1) * 100,
    }


def test_percentile_interpolates_like_numpy_linear():
    values = [1.0, 2.0, 4.0, 8.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 8.0
    assert percentile(values, 50) == 3.0
    assert percentile(values, 10) == pytest.approx(1.3)


def test_request_errors_are_unsigned_and_exclude_burst_tpot():
    pairs = [pair(10.0, 12.0), pair(20.0, 18.0), pair(0.04, 3.0)]
    tpot = point_metrics(pairs, "tpot_ms")
    # The 0.04 ms burst-delivery request is excluded and counted; means 15 vs 15 give 0%.
    assert tpot["n"] == 2 and tpot["excluded"] == 1
    assert tpot["e2e"] == pytest.approx(0.0)
    assert tpot["quantile"] == quantile_error([10.0, 20.0], [12.0, 18.0]) > 0
    ttft = point_metrics(pairs, "ttft_ms")
    assert ttft["n"] == 3 and ttft["excluded"] == 0 and ttft["e2e"] == pytest.approx(20.0)
    # Over-prediction and under-prediction both count positively.
    assert point_metrics([pair(10.0, 5.0)], "tpot_ms")["e2e"] == pytest.approx(50.0)


def test_request_metrics_bootstrap_is_deterministic_and_skips_unmatched():
    pairs = [pair(10.0 + i, 11.0 + i * 1.1) for i in range(30)] + [
        {**pair(10.0, 10.0), "tpot_ms_ape": None, "ttft_ms_ape": None}
    ]
    first, second = request_metrics(pairs), request_metrics(pairs)
    assert first == second
    assert first["tpot_ms"]["n"] == 30
    assert 0 < first["tpot_ms"]["e2e_se"] < first["tpot_ms"]["e2e"]
    single = request_metrics([pair(10.0, 15.0)])
    assert single["tpot_ms"]["e2e"] == pytest.approx(50.0) and single["tpot_ms"]["e2e_se"] is None


def test_step_sums_pool_exactly():
    left, right = step_accumulator([1.0, 3.0]), step_accumulator([3.0, 5.0])
    pooled = {key: left[key] + right[key] for key in left}
    summary = step_summary(pooled)
    assert summary["mape"] == 3.0 and summary["n"] == 4
    assert summary["se"] == pytest.approx(math.sqrt(8 / 3 / 4))
    assert step_summary(step_accumulator([2.0])) is None


def published(**changes):
    run = result(**changes)
    run.update(
        performance_model="op_level",
        config_summary=[{"name": "Estimation", "value": "op_level"}],
        accuracy_metrics={
            **request_metrics(run["pairs"]),
            "step": {
                "seed": 7,
                "dropped_shifted": 0,
                "decode": step_accumulator([1.0, 2.0]),
                "prefill": step_accumulator([]),
            },
        },
        cpu_overhead={
            "tokenization": {"workers": 1, "base_ms": 0.0, "per_input_token_ms": 0.001},
            "extra_cpu_overhead": None,
        },
    )
    return run


def test_published_metrics_reconcile_with_pairs_and_estimation():
    run = published()
    validate_result(run)
    assert detail_path(run) == "details/agentx-446063-op-level.json"
    assert detail_path(result()) == "details/agentx-446063.json"
    for change in (
        lambda r: r["accuracy_metrics"]["tpot_ms"].update(e2e=1.0),
        lambda r: r["accuracy_metrics"]["ttft_ms"].update(n=5),
        lambda r: r.update(performance_model="fpm_learned"),
        lambda r: r.update(performance_model="magic"),
        lambda r: r["accuracy_metrics"]["step"].update(decode={"n": 1}),
        lambda r: r["cpu_overhead"].update(tokenization={"workers": "many"}),
        lambda r: r.pop("cpu_overhead"),
    ):
        broken = json.loads(json.dumps(run))
        change(broken)
        with pytest.raises(ValueError):
            validate_result(broken)


def test_unavailable_runs_publish_no_request_metrics():
    run = published()
    unavailable = score([observation()], [], run["boundary"])
    run.update(
        outcome="failed",
        reason="replay_process_failed",
        prediction=unavailable["prediction"],
        point_errors=unavailable["point_errors"],
        request_errors=unavailable["request_errors"],
    )
    for item in run["pairs"]:
        item.update(ttft_ms_ape=None, tpot_ms_ape=None)
    with pytest.raises(ValueError):
        validate_result(run)
    run["accuracy_metrics"].update(tpot_ms=None, ttft_ms=None)
    validate_result(run)


def test_driver_digest_covers_metrics_and_learned_modules():
    root = Path(__file__).resolve().parents[1]
    files = {}

    def read(name):
        files[name] = True
        return (root / name).read_bytes()

    driver_digest(read)
    assert {"scripts/e2e_accuracy/agentx_metrics.py", "scripts/e2e_accuracy/agentx_learned.py"} <= set(files)


def test_learned_models_run_only_where_step_captures_are_pinned(monkeypatch):
    monkeypatch.setattr(agentx_learned, "STEP_CAPTURES", {"Qwen/Qwen3.5-397B-A17B": {}})
    import scripts.e2e_accuracy.run_agentx_accuracy as runner

    monkeypatch.setattr(runner, "STEP_CAPTURES", agentx_learned.STEP_CAPTURES)
    assert performance_models({"model": "Qwen/Qwen3.5-397B-A17B"}) == ("op_level", "fpm_learned")
    assert performance_models({"model": "nvidia/GLM-5.3-NVFP4"}) == ("op_level",)
    # Release wheels without the learned FPM still produce every op-based result.
    assert performance_models({"model": "Qwen/Qwen3.5-397B-A17B"}, learned=False) == ("op_level",)


def test_learned_support_follows_the_evaluated_wheel(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name="aisimulate_core.sdk.fpm_learned")

    monkeypatch.setattr(agentx_learned, "runtime_module", missing)
    assert agentx_learned.supported() is False

    def broken(name):
        raise ModuleNotFoundError(name="sklearn")

    monkeypatch.setattr(agentx_learned, "runtime_module", broken)
    with pytest.raises(ModuleNotFoundError):
        agentx_learned.supported()


def test_learned_replay_is_frozen_and_never_falls_back():
    config = {"engine": {"workers": {"aggregated": {"timing": {"type": "default"}}}}}
    assert apply_performance_model(json.loads(json.dumps(config)), "op_level") == config
    timing = apply_performance_model(config, "fpm_learned", Path("/a.json"))["engine"]["workers"]["aggregated"][
        "timing"
    ]
    assert timing["estimation_mode"] == "fpm_learned" and timing["fallback_policy"] == "deny"
    assert timing["estimator_config"]["fpm_learned"] == {"artifact_path": "/a.json", "adaptation": "frozen"}
    with pytest.raises(ValueError):
        apply_performance_model(config, "fpm_learned")


def test_cpu_overhead_reads_the_compiled_replay_terms():
    deployment = {
        "tokenization": {"workers": 1, "base_ms": 0.0, "per_input_token_ms": 0.0012, "queue": "ignored"},
        "extra_cpu_overhead": {"base_ms": 12.6, "per_input_token_ms": 0.0003},
    }
    assert cpu_overhead(deployment) == {
        "tokenization": {"workers": 1, "base_ms": 0.0, "per_input_token_ms": 0.0012},
        "extra_cpu_overhead": {"base_ms": 12.6, "per_input_token_ms": 0.0003},
    }
    assert cpu_overhead({"tokenization": None}) == {"tokenization": None, "extra_cpu_overhead": None}
    assert cpu_overhead(None) is None


def fpm(prefill=0, decode=0, wall=0.01, categories=None):
    scheduled = {"num_prefill_requests": prefill, "num_decode_requests": decode}
    if categories is not None:
        scheduled["device_time_s_by_category"] = categories
    return {"wall_time": wall, "scheduled_requests": scheduled}


def test_unattributed_steps_and_their_previous_record_are_dropped():
    records = [
        fpm(decode=4, categories={"target_verify": 0.01}),
        fpm(decode=4, categories={"target_verify": 0.02}),  # Absorbed the next record's forward.
        fpm(decode=4, categories={"eagle_draft": 0.001}),  # Target forward recorded on the previous record.
        fpm(prefill=1, categories={"eagle_draft": 0.001}),  # Prefill without extend.
        fpm(),  # Idle heartbeat.
        fpm(prefill=1, categories={"extend": 0.3}),
        fpm(decode=2),  # No categories recorded: kept.
    ]
    assert agentx_learned.unattributed(records) == [False, True, True, True, False, False, False]
    spec = {"drop_unattributed": True}
    kept, dropped = agentx_learned.kept(spec, records)
    assert dropped == 3 and len(kept) == 4
    assert agentx_learned.kept({"drop_unattributed": False}, records) == (records, 0)
    assert [agentx_learned.step_kind(r) for r in (records[0], records[3], records[4], fpm(prefill=1, decode=1))] == [
        "decode",
        "prefill",
        None,
        None,
    ]


def test_dynamo_native_windows_follow_the_published_selection(tmp_path, monkeypatch):
    lines = [json.dumps({"event": {"fpm": fpm(decode=i + 1)}}) for i in range(5)] + ["{}"]
    trace = tmp_path / "fpm.1.jsonl.gz"
    with gzip.open(trace, "wt") as stream:
        stream.write("\n".join(lines) + "\n")
    digest = agentx_learned._sha256(trace)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps([{"path": "traces/aggregated/l8/fpm.1.jsonl.gz", "sha256": digest, "first_line": 2, "last_line": 4}])
    )
    requested = []

    def download(spec, name, cache):
        requested.append(name)
        return selection if name.endswith("selection.json") else trace

    monkeypatch.setattr(agentx_learned, "_download", download)
    spec = {"layout": "dynamo_native"}
    windows = list(agentx_learned.windows(spec, 7, tmp_path))
    assert requested == ["training/seed7.selection.json", "native-step/seed7/l8/traces/fpm.1.jsonl.gz"]
    assert [(lanes, [r["scheduled_requests"]["num_decode_requests"] for r in rows]) for lanes, rows in windows] == [
        (8, [2, 3, 4])
    ]
    selection.write_text(selection.read_text().replace(digest, "0" * 64))
    with pytest.raises(ValueError, match="checksum"):
        list(agentx_learned.windows(spec, 7, tmp_path))


def test_qualification_requires_every_performance_model(tmp_path, monkeypatch):
    import scripts.e2e_accuracy.run_agentx_accuracy as runner

    op = published()
    op["simulation_features"] = simulation_record()
    learned = json.loads(json.dumps(op))
    learned["performance_model"] = "fpm_learned"
    learned["config_summary"] = [{"name": "Estimation", "value": "fpm_learned"}]
    catalog = silicon_inputs(tmp_path, op)
    monkeypatch.setattr(runner, "STEP_CAPTURES", {op["model"]: {}})
    snapshot = json.loads(bundle()["summary.json"])["snapshot"]
    del snapshot["completed_at"]
    snapshot.update(cohort_sha256=catalog["snapshot_id"], driver_sha256=driver_digest())
    args = Namespace(inputs=tmp_path, shards=tmp_path, shard_count=1, output=tmp_path / "public")
    for selected in ([op], [op, op], [learned, learned]):
        (tmp_path / "shard-0.json").write_bytes(
            canonical({"snapshot": snapshot, "learned_fpm": True, "runs": selected})
        )
        with pytest.raises(ValueError):
            runner.qualify(args)
    (tmp_path / "shard-0.json").write_bytes(
        canonical({"snapshot": snapshot, "learned_fpm": True, "runs": [op, learned]})
    )
    runner.qualify(args)
    summary = json.loads((args.output / "summary.json").read_bytes())
    assert [r["detail_path"] for r in summary["runs"]] == [
        "details/agentx-446063-op-level.json",
        "details/agentx-446063-fpm-learned.json",
    ]
    from scripts.pages.agentx_pages import validate_bundle

    files = {p.relative_to(args.output).as_posix(): p.read_bytes() for p in args.output.rglob("*.json")}
    assert len(validate_bundle(files)["runs"]) == 2
    # A wheel without the learned FPM qualifies on op-based results alone, and only then.
    (tmp_path / "shard-0.json").write_bytes(canonical({"snapshot": snapshot, "learned_fpm": False, "runs": [op]}))
    runner.qualify(Namespace(**{**vars(args), "output": tmp_path / "op-only"}))
    (tmp_path / "shard-0.json").write_bytes(canonical({"snapshot": snapshot, "runs": [op]}))
    with pytest.raises(KeyError):
        runner.qualify(Namespace(**{**vars(args), "output": tmp_path / "unrecorded"}))


def test_sglang_hook_windows_follow_the_published_manifest(tmp_path, monkeypatch):
    shards = {}
    for seed, lanes, window in ((7, 8, "warmup"), (7, 8, "profiling"), (7, 14, "profiling"), (42, 8, "profiling")):
        path = tmp_path / f"s{seed}-c{lanes}-{window}.jsonl.gz"
        with gzip.open(path, "wt") as stream:
            stream.write(json.dumps({"sequence": 1, "fpm": fpm(decode=lanes)}) + "\n")
        shards[f"step-capture/seed{seed}/c{lanes}/{window}.jsonl.gz"] = path
    manifest = tmp_path / "aisim-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [{"artifact_id": key, "sha256": agentx_learned._sha256(p)} for key, p in shards.items()]
                + [{"artifact_id": "collection/windows.json", "sha256": "0" * 64}]
            }
        )
    )
    monkeypatch.setattr(
        agentx_learned,
        "_download",
        lambda spec, name, cache: manifest if name == "aisim-manifest.json" else shards[name],
    )
    spec = {"layout": "sglang_hook"}
    assert [lanes for lanes, _ in agentx_learned.windows(spec, 7, tmp_path)] == [14, 8, 8]
    assert [lanes for lanes, _ in agentx_learned.windows(spec, 7, tmp_path, scoring=True)] == [14, 8]
    assert [lanes for lanes, _ in agentx_learned.windows(spec, 42, tmp_path)] == [8]


def test_shifted_pairs_flag_both_records_of_a_split_step():
    records = [fpm(prefill=1, wall=0.010), fpm(prefill=1, wall=0.002), fpm(prefill=1, wall=0.030), fpm(), fpm(decode=1)]
    # Predicted 10 ms each: 0.2x next to 3x is one step split across two records; idle records never pair.
    assert agentx_learned.shifted_pairs(records, lambda r: 10.0) == [False, True, True, False, False]
    assert agentx_learned.shifted_pairs(records[:2], lambda r: 10.0) == [False, False]


def test_operation_level_decode_is_packed_by_verification_width(tmp_path, monkeypatch):
    step = fpm(decode=1, wall=0.01)
    # Capture-shaped: per-request lists describe the one logical request.
    step["scheduled_requests"].update(
        sum_decode_kv_tokens=2584, extend_lengths=[4], past_kv_lengths=[2584], spec_accept_lengths=[3]
    )
    # Zero-prefix prefill as SGLang hooks record it: KV total counts the 8,192 fresh tokens.
    prefill = fpm(prefill=1, wall=0.01)
    prefill["scheduled_requests"].update(
        sum_prefill_tokens=8192, sum_prefill_kv_tokens=8192, past_kv_lengths=[0], extend_lengths=[8192]
    )
    seen = []

    class Estimator:
        def __init__(self, config):
            self.config = config

        def estimate_forward_pass_time_ms(self, metrics):
            seen.append((self.config["estimation_mode"], metrics[0]["scheduled_requests"]))
            return 10.0

    monkeypatch.setattr(agentx_learned, "STEP_CAPTURES", {"m": {"drop_unattributed": False}})
    monkeypatch.setattr(agentx_learned, "windows", lambda *args, **kwargs: iter([(8, [step, prefill])]))
    monkeypatch.setattr(agentx_learned, "_estimator", Estimator)
    op = {"estimation_mode": "op_level", "nextn": 3}
    agentx_learned.score_steps("m", 8, op, tmp_path)
    # Packed counts without the logical per-request lists, which the native validator would reject.
    assert seen[0] == ("op_level", {"num_prefill_requests": 0, "num_decode_requests": 4, "sum_decode_kv_tokens": 10336})
    # Operation-level prefill KV is the cached prefix only.
    assert seen[1][1] == {
        "num_prefill_requests": 1,
        "num_decode_requests": 0,
        "sum_prefill_tokens": 8192,
        "sum_prefill_kv_tokens": 0,
    }
    seen.clear()
    agentx_learned.score_steps("m", 8, {**op, "estimation_mode": "fpm_learned"}, tmp_path)
    # The learned FPM reads captures as recorded.
    assert [x[1] for x in seen] == [step["scheduled_requests"], prefill["scheduled_requests"]]


def test_wheels_without_learned_mtp_publish_op_based_results_without_step_metrics(tmp_path, monkeypatch):
    def construct(config):
        if config["estimation_mode"] == "fpm_learned":
            raise ValueError("invalid engine config: fpm_learned does not support speculative decoding")
        return object()

    monkeypatch.setattr(agentx_learned, "STEP_CAPTURES", {"m": {"drop_unattributed": True}})
    monkeypatch.setattr(agentx_learned, "_estimator", construct)
    op, learned = {"estimation_mode": "op_level", "nextn": 3}, {"estimation_mode": "fpm_learned", "nextn": 3}
    assert agentx_learned.score_steps("m", 8, op, tmp_path, learned) is None
    assert agentx_learned.score_steps("m", 8, learned, tmp_path, learned) is None

    def broken(config):
        raise ValueError("invalid engine config: something else")

    monkeypatch.setattr(agentx_learned, "_estimator", broken)
    with pytest.raises(ValueError, match="something else"):
        agentx_learned.score_steps("m", 8, op, tmp_path, learned)


def test_learned_models_retrain_every_evaluation(tmp_path, monkeypatch):
    trained = []
    learned = type(
        "Learned",
        (),
        {
            "FEATURE_PRESETS": {"p": ()},
            "group_iterations": staticmethod(lambda records: records),
            "train": staticmethod(lambda *args, **kwargs: trained.append(1) or {"wheel": len(trained)}),
        },
    )
    monkeypatch.setattr(agentx_learned, "runtime_module", lambda name: learned)
    monkeypatch.setattr(agentx_learned, "windows", lambda *args, **kwargs: iter([]))
    spec = {"repository": "r", "revision": "v", "root": "x", "features": "p", "drop_unattributed": False}
    output = tmp_path / "model.json"
    agentx_learned.train_artifact(spec, (42,), tmp_path, output)
    agentx_learned.train_artifact(spec, (42,), tmp_path, output)
    assert len(trained) == 2 and json.loads(output.read_text()) == {"wheel": 2}
