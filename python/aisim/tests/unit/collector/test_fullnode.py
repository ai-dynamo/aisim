# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for full-node collector orchestration."""

import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from collector import fullnode, provenance
from collector.framework_manifest import CollectorRuntime

_COLLECTOR_DIR = str(Path(__file__).resolve().parents[3] / "collector")
if _COLLECTOR_DIR not in sys.path:
    sys.path.insert(0, _COLLECTOR_DIR)


pytestmark = pytest.mark.unit


class _Checkpoint:
    instances: ClassVar[list] = []

    def __init__(self, **_kwargs):
        self.passed = []
        self.failed = []
        self.attempted = []
        self.flushed = False
        self.__class__.instances.append(self)

    def mark_attempted(self, task_id):
        self.attempted.append(task_id)

    def mark_attempted_many(self, task_ids):
        self.attempted.extend(task_ids)

    def has_passed(self, task_id):
        return task_id in self.passed

    def mark_passed(self, task_id):
        self.passed.append(task_id)

    def mark_failed(self, task_id):
        self.failed.append(task_id)

    def flush(self, force=False):
        self.flushed = force


def _logger():
    return SimpleNamespace(info=lambda *_args: None, warning=lambda *_args: None, exception=lambda *_args: None)


def test_shape_index_selects_exact_case_before_limit(monkeypatch):
    cases = [{"shape": 0}, {"shape": 1}, {"shape": 2}]
    monkeypatch.setenv("DEEPEP_LL_SHAPE_INDEX", "2")

    assert fullnode.select_cases("deepep_ll", cases, limit=1) == [{"shape": 2}]


def test_shape_index_rejects_out_of_range(monkeypatch):
    monkeypatch.setenv("DEEPEP_NORMAL_SHAPE_INDEX", "3")

    with pytest.raises(RuntimeError, match=r"DEEPEP_NORMAL_SHAPE_INDEX=3.*0\.\.1"):
        fullnode.select_cases("deepep_normal", [{"shape": 0}, {"shape": 1}], limit=None)


def test_native_tp_route_uses_common_limit_without_collective_shape_selector(monkeypatch):
    monkeypatch.setenv("DEEPEP_LL_SHAPE_INDEX", "999")
    cases = [{"tp": 1}, {"tp": 2}, {"tp": 4}]
    assert fullnode.select_cases("kimi_k31_dsa_context_module", cases, limit=2) == cases[:2]
    assert fullnode.select_cases("kimi_k31_dsa_generation_module", cases, limit=None) == cases


@pytest.fixture(autouse=True)
def isolated_collector_workdir(monkeypatch, tmp_path):
    monkeypatch.delenv("COLLECTOR_LOG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("sm", [103, None])
@pytest.mark.parametrize("maturity", [{"unverified": True}, {"unverified_sms": (103,)}])
def test_fullnode_preserves_registry_maturity_gate_before_import(maturity, sm):
    _Checkpoint.instances.clear()
    errors = fullnode.collect_sglang_fullnode_op(
        {"name": "sglang", "type": "kimi_k31_dsa_context_module", "module": "must_not_import", **maturity},
        runtime_version="0.0.0.dev1",
        limit=None,
        shuffle=False,
        shuffle_seed=42,
        backend="sglang",
        resume_options=None,
        model_path=None,
        case_plan=None,
        sm_version=sm,
        case_filters=None,
        get_test_cases_for_model=lambda *_args: pytest.fail("planned unverified collector"),
        resume_checkpoint_cls=_Checkpoint,
        logger=_logger(),
    )
    assert errors[0]["error_type"] == "UnverifiedCollector"
    assert errors[0]["classification"] == "unverified_skipped"
    assert not _Checkpoint.instances


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_native_sm121_maturity_gate_prevents_import_and_queueing(phase):
    from collector.sglang.registry import REGISTRY
    from collector.version_resolver import build_collections

    _Checkpoint.instances.clear()
    (collection,) = build_collections(
        REGISTRY, "sglang", "0.0.0.dev1+g41208373c.d20261104", [f"kimi_k31_dsa_{phase}_module"]
    )
    assert 103 not in collection["unverified_sms"]
    errors = fullnode.collect_sglang_fullnode_op(
        {**collection, "module": "must_not_import"},
        runtime_version="0.0.0.dev1+g41208373c.d20261104",
        limit=None,
        shuffle=False,
        shuffle_seed=42,
        backend="sglang",
        resume_options=None,
        model_path=None,
        case_plan=None,
        sm_version=121,
        case_filters=None,
        get_test_cases_for_model=lambda *_args: pytest.fail("planned unverified SM121"),
        resume_checkpoint_cls=_Checkpoint,
        logger=_logger(),
    )
    assert errors[0]["classification"] == "unverified_skipped"
    assert not _Checkpoint.instances


def test_runner_maps_reported_failure_to_exact_checkpoint_task(monkeypatch):
    cases = [
        {"hidden_size": 2048, "num_experts": 128, "topk": 8},
        {"hidden_size": 8192, "num_experts": 512, "topk": 22},
    ]

    def get_cases():
        return list(cases)

    def run_cases(*, perf_filename, limit, cases):
        assert perf_filename == "perf.txt"
        assert limit is None
        return {"succeeded": 1, "failed": [cases[1]]}

    module_name = "fake_fullnode_collector"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(__compat__=">=0.5.0", get_cases=get_cases, run_cases=run_cases),
    )
    monkeypatch.setattr(fullnode, "filter_cases", lambda values, **_kwargs: (values, []))
    _Checkpoint.instances.clear()

    errors = fullnode.collect_sglang_fullnode_op(
        {
            "name": "fake",
            "type": "deepep_ll",
            "module": module_name,
            "get_func": "get_cases",
            "run_func": "run_cases",
            "perf_filename": "perf.txt",
        },
        runtime_version="0.5.12",
        limit=None,
        shuffle=False,
        shuffle_seed=42,
        backend="sglang",
        resume_options=None,
        model_path=None,
        case_plan=None,
        sm_version=100,
        case_filters=None,
        get_test_cases_for_model=lambda get_func, _model_path: get_func(),
        resume_checkpoint_cls=_Checkpoint,
        logger=_logger(),
    )

    checkpoint = _Checkpoint.instances[-1]
    assert len(checkpoint.attempted) == 2
    assert len(checkpoint.passed) == 1
    assert len(checkpoint.failed) == 1
    assert checkpoint.flushed is True
    assert len(errors) == 1
    assert errors[0]["error_type"] == "FullNodeCaseFailure"
    assert errors[0]["task_params"] == str(cases[1])


def test_runner_fails_closed_when_failure_list_is_missing(monkeypatch):
    case = {"hidden_size": 2048, "num_experts": 128, "topk": 8}
    module_name = "fake_fullnode_collector_missing_failures"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(
            __compat__=">=0.5.0",
            get_cases=lambda: [case],
            run_cases=lambda **_kwargs: None,
        ),
    )
    monkeypatch.setattr(fullnode, "filter_cases", lambda values, **_kwargs: (values, []))
    _Checkpoint.instances.clear()

    errors = fullnode.collect_sglang_fullnode_op(
        {
            "name": "fake",
            "type": "deepep_ll",
            "module": module_name,
            "get_func": "get_cases",
            "run_func": "run_cases",
            "perf_filename": "perf.txt",
        },
        runtime_version="0.5.12",
        limit=None,
        shuffle=False,
        shuffle_seed=42,
        backend="sglang",
        resume_options=None,
        model_path=None,
        case_plan=None,
        sm_version=100,
        case_filters=None,
        get_test_cases_for_model=lambda get_func, _model_path: get_func(),
        resume_checkpoint_cls=_Checkpoint,
        logger=_logger(),
    )

    assert len(_Checkpoint.instances[-1].attempted) == 1
    assert len(_Checkpoint.instances[-1].failed) == 1
    assert errors[0]["error_type"] == "FullNodeCollectionFailure"
    assert "must return" in errors[0]["error_message"]


def test_native_failure_summary_aggregates_by_consumer_group(monkeypatch, tmp_path):
    import collect as collect_mod
    from collector.op_plans import ModuleCase

    model = "moonshotai/Kimi-K3.1-Pro"
    cases = [ModuleCase(length, 1, 48, "fp8", "bfloat16", "bfloat16", model, 0) for length in (4096, 8192)]
    module_name = "fake_native_failure_groups"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(
            __compat__="sglang==0.0.0.dev1+g41208373c.d20261104",
            get_cases=lambda: cases,
            run_cases=lambda **kwargs: {"succeeded": 0, "failed": kwargs["cases"]},
        ),
    )
    monkeypatch.setattr(fullnode, "filter_cases", lambda values, **_kwargs: (values, []))
    errors = fullnode.collect_sglang_fullnode_op(
        {
            "name": "sglang",
            "type": "kimi_k31_dsa_generation_module",
            "module": module_name,
            "get_func": "get_cases",
            "run_func": "run_cases",
            "perf_filename": str(tmp_path / "dsa_generation_module_perf.txt"),
        },
        runtime_version="0.0.0.dev1+g41208373c.d20261104",
        limit=None,
        shuffle=False,
        shuffle_seed=42,
        backend="sglang",
        resume_options=None,
        model_path=None,
        case_plan=None,
        sm_version=103,
        case_filters=None,
        get_test_cases_for_model=lambda get_func, _model_path: get_func(),
        resume_checkpoint_cls=_Checkpoint,
        logger=_logger(),
    )
    expected_group = json.dumps(
        dict(
            model_path=model,
            num_heads=48,
            kv_cache_dtype="fp8",
            compute_dtype="bfloat16",
            gemm_type="bfloat16",
            batch_size=1,
        ),
        sort_keys=True,
    )
    assert len(errors) == 2
    assert {error["group"] for error in errors} == {expected_group}
    monkeypatch.setenv("COLLECTOR_LOG_DIR", str(tmp_path))
    collect_mod.generate_collection_summary(errors, "sglang", "test")
    summary = json.loads((tmp_path / "collection_summary_sglang.json").read_text())["summary"]
    assert summary["errors_by_group"] == {f"sglang.kimi_k31_dsa_generation_module:{expected_group}": 2}


def test_replay_selects_full_node_cases_like_the_pool_ops(monkeypatch):
    import collect as collect_mod

    cases = [{"shape": 0}, {"shape": 1}, {"shape": 2}]
    ran = []

    def run_cases(*, perf_filename, limit, cases):
        ran.extend(cases)
        return {"succeeded": len(cases), "failed": []}

    module_name = "fake_fullnode_collector_replay"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(__compat__=">=0.5.0", get_cases=lambda: list(cases), run_cases=run_cases),
    )
    monkeypatch.setattr(fullnode, "filter_cases", lambda values, **_kwargs: (values, []))
    replay = collect_mod.ReplaySpec("errors.json", {str(cases[1])})

    errors = fullnode.collect_sglang_fullnode_op(
        {
            "name": "fake",
            "type": "deepep_ll",
            "module": module_name,
            "get_func": "get_cases",
            "run_func": "run_cases",
            "perf_filename": "perf.txt",
        },
        runtime_version="0.5.12",
        limit=None,
        shuffle=False,
        shuffle_seed=42,
        backend="sglang",
        resume_options=None,
        model_path=None,
        case_plan=None,
        sm_version=100,
        case_filters=None,
        replay=replay,
        get_test_cases_for_model=lambda get_func, _model_path: get_func(),
        resume_checkpoint_cls=_Checkpoint,
        logger=_logger(),
    )

    assert errors == []
    assert ran == [cases[1]]
    assert replay.report()["matched_by_op"] == {"deepep_ll": 1}


def test_collect_sglang_passes_only_keywords_the_full_node_runner_accepts():
    """collect_sglang runs full-node ops after the pool ops: a keyword the runner does not
    accept is a TypeError that loses the run's summary and provenance."""
    import ast
    import inspect

    import collect as collect_mod

    tree = ast.parse(Path(collect_mod.__file__).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "collect_sglang_fullnode_op"
    ]
    assert calls
    accepted = set(inspect.signature(fullnode.collect_sglang_fullnode_op).parameters)
    for call in calls:
        assert {keyword.arg for keyword in call.keywords} <= accepted


REAL_FULLNODE_MODULE = "collector.wideep.sglang.collect_deepep_ll"


def _real_collection() -> dict:
    return {
        "name": "sglang",
        "type": "deepep_ll",
        "module": REAL_FULLNODE_MODULE,
        "get_func": "get_deepep_ll_test_cases",
        "run_func": "run_deepep_ll_fullnode",
        "perf_filename": "wideep_deepep_ll_perf.txt",
    }


def _run_with_real_checkpoint(
    monkeypatch,
    tmp_path,
    *,
    cases,
    failed_cases=(),
    resume=False,
    retry_failed=False,
    run_error=False,
):
    import collect as collect_mod

    calls = []

    def run_cases(*, perf_filename, limit, cases):
        calls.append(list(cases))
        if run_error:
            raise RuntimeError("launcher failed before producing a row")
        return {"succeeded": len(cases) - len(failed_cases), "failed": list(failed_cases)}

    monkeypatch.setitem(
        sys.modules,
        REAL_FULLNODE_MODULE,
        SimpleNamespace(
            __compat__=">=0.5.0",
            get_deepep_ll_test_cases=lambda: list(cases),
            run_deepep_ll_fullnode=run_cases,
        ),
    )
    monkeypatch.setattr(fullnode, "filter_cases", lambda values, **_kwargs: (values, []))
    checkpoint_dir = tmp_path / "checkpoint"
    errors = fullnode.collect_sglang_fullnode_op(
        _real_collection(),
        runtime_version="0.5.14",
        limit=None,
        shuffle=False,
        shuffle_seed=42,
        backend="sglang",
        resume_options={
            "checkpoint_dir": str(checkpoint_dir),
            "resume": resume,
            "retry_failed": retry_failed,
        },
        model_path=None,
        case_plan=None,
        sm_version=100,
        case_filters=None,
        get_test_cases_for_model=lambda get_func, _model_path: get_func(),
        resume_checkpoint_cls=collect_mod.ResumeCheckpoint,
        logger=_logger(),
    )
    checkpoint_path = checkpoint_dir / "sglang" / "sglang.deepep_ll.json"
    return errors, calls, checkpoint_path


def _checkpoint_sets(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: set(data.get(key, [])) for key in ("done", "failed", "attempted")}


def test_successful_retry_retains_run_diagnostic_without_unresolved_summary(monkeypatch, tmp_path):
    cases = [{"shape": 1}]
    errors, _, checkpoint = _run_with_real_checkpoint(monkeypatch, tmp_path, cases=cases, run_error=True)
    assert errors[0]["error_type"] == "FullNodeCollectionFailure"
    assert _checkpoint_sets(checkpoint)["failed"]

    errors, calls, _ = _run_with_real_checkpoint(monkeypatch, tmp_path, cases=cases, resume=True, retry_failed=True)
    assert calls == [cases]
    assert errors == []
    assert not _checkpoint_sets(checkpoint)["failed"]
    diagnostic = checkpoint.with_name("errors_sglang.deepep_ll.json")
    assert json.loads(diagnostic.read_text())[0]["error_message"] == "launcher failed before producing a row"
    errors, calls, _ = _run_with_real_checkpoint(monkeypatch, tmp_path, cases=cases, resume=True)
    assert errors == []
    assert calls == []


def test_optional_diagnostic_mirror_failure_keeps_measured_case_passed(monkeypatch, tmp_path):
    log_dir = tmp_path / "unwritable-mirror"
    monkeypatch.setenv("COLLECTOR_LOG_DIR", str(log_dir))
    write_text = Path.write_text

    def write(path, *args, **kwargs):
        if path.parent == log_dir:
            raise PermissionError("optional mirror unavailable")
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write)
    cases = [{"shape": 1}, {"shape": 2}]
    errors, calls, checkpoint = _run_with_real_checkpoint(monkeypatch, tmp_path, cases=cases, failed_cases=cases[:1])
    assert calls == [cases]
    assert len(errors) == 1
    assert errors[0]["task_params"] == str(cases[0])
    state = _checkpoint_sets(checkpoint)
    assert len(state["done"]) == len(state["failed"]) == 1
    assert state["done"].isdisjoint(state["failed"])
    authoritative = json.loads(checkpoint.with_name("errors_sglang.deepep_ll.json").read_text())
    assert authoritative == errors
    errors, calls, _ = _run_with_real_checkpoint(monkeypatch, tmp_path, cases=cases, resume=True)
    assert calls == []
    assert len(errors) == 1  # The real failed case remains visible.


def _finalize_pending_fullnode_event(tmp_path, checkpoint_path):
    import collect as collect_mod

    pending = _checkpoint_sets(checkpoint_path)["attempted"]
    output_root = tmp_path / "out"
    output_root.mkdir()
    parquet_path = output_root / "wideep_deepep_ll_perf.parquet"
    pq.write_table(pa.table({"shape": [1, 2], "latency": [1.0, 2.0]}), parquet_path)
    staging_path = parquet_path.with_suffix(".txt")
    staging_path.write_text("retained staging\n", encoding="utf-8")
    staging_stat = staging_path.stat()
    runtime = CollectorRuntime(
        framework="sglang",
        version="0.5.14",
        images={"default": "lmsysorg/sglang:v0.5.14@sha256:" + "0" * 64},
    )
    collect_mod._write_collector_provenance(
        output_root,
        [parquet_path],
        {
            "installed_version": runtime.version,
            "runtime": runtime,
            "sm_version": 100,
            "collections": [_real_collection()],
        },
        run_errors=[],
        backend="sglang",
        checkpoint_dir=str(tmp_path / "checkpoint"),
        finalization_info={
            parquet_path.resolve(): collect_mod.PerfFinalizationInfo(
                new_rows=2,
                merged_existing=False,
                source_digest="sha256:" + hashlib.sha256(staging_path.read_bytes()).hexdigest(),
                source_device=staging_stat.st_dev,
                source_inode=staging_stat.st_ino,
            )
        },
    )
    table = yaml.safe_load((output_root / "collection_meta.yaml").read_text(encoding="utf-8"))["tables"][
        "wideep_deepep_ll_perf"
    ]
    return pending, table


def test_real_checkpoint_fresh_fullnode_event_finalizes_exact_selected_set(monkeypatch, tmp_path):
    cases = [{"shape": 1}, {"shape": 2}]
    errors, calls, checkpoint_path = _run_with_real_checkpoint(monkeypatch, tmp_path, cases=cases)

    assert errors == []
    assert calls == [cases]
    before = _checkpoint_sets(checkpoint_path)
    assert before["done"] == before["attempted"]
    assert len(before["attempted"]) == 2

    pending, table = _finalize_pending_fullnode_event(tmp_path, checkpoint_path)
    assert pending == before["attempted"]
    assert table["case_plan_hash"] == provenance.case_plan_hash(sorted(before["attempted"]))
    assert _checkpoint_sets(checkpoint_path)["attempted"] == set()


def test_real_checkpoint_interrupted_resume_unions_old_and_new_fullnode_cases(monkeypatch, tmp_path):
    cases = [{"shape": 1}, {"shape": 2}]
    _, _, checkpoint_path = _run_with_real_checkpoint(monkeypatch, tmp_path, cases=[cases[0]])

    errors, calls, _ = _run_with_real_checkpoint(monkeypatch, tmp_path, cases=cases, resume=True)

    assert errors == []
    assert calls == [[cases[1]]]
    state = _checkpoint_sets(checkpoint_path)
    assert state["done"] == state["attempted"]
    assert len(state["attempted"]) == 2


def test_real_checkpoint_retry_after_closed_event_attests_only_retry(monkeypatch, tmp_path):
    cases = [{"shape": 1}, {"shape": 2}]
    _, _, checkpoint_path = _run_with_real_checkpoint(
        monkeypatch,
        tmp_path,
        cases=cases,
        failed_cases=[cases[1]],
    )
    _pending, _table = _finalize_pending_fullnode_event(tmp_path, checkpoint_path)
    assert _checkpoint_sets(checkpoint_path)["attempted"] == set()

    _, calls, _ = _run_with_real_checkpoint(
        monkeypatch,
        tmp_path,
        cases=cases,
        resume=True,
        retry_failed=True,
    )

    assert calls == [[cases[1]]]
    state = _checkpoint_sets(checkpoint_path)
    assert len(state["attempted"]) == 1
    assert state["attempted"] <= state["done"]
    assert state["failed"] == set()


def test_real_checkpoint_zero_work_preserves_pending_and_failure_marks_all_selected(monkeypatch, tmp_path):
    cases = [{"shape": 1}, {"shape": 2}]
    recorded_errors, _, checkpoint_path = _run_with_real_checkpoint(
        monkeypatch,
        tmp_path,
        cases=cases,
        failed_cases=cases,
    )
    pending = _checkpoint_sets(checkpoint_path)["attempted"]

    errors, calls, _ = _run_with_real_checkpoint(monkeypatch, tmp_path, cases=cases, resume=True)

    assert calls == []
    assert _checkpoint_sets(checkpoint_path)["attempted"] == pending
    assert len(_checkpoint_sets(checkpoint_path)["failed"]) == 2
    assert errors == recorded_errors
    assert len(errors) == 2
    assert {error["task_params"] for error in errors} == set(map(str, cases))


@pytest.mark.parametrize("sm", [103, None])
def test_pool_unknown_sm_cannot_bypass_registry_gate(sm):
    import collect as collect_mod

    errors = collect_mod.collect_ops(
        1,
        [{"name": "sglang", "type": "test", "module": "must_not_import", "unverified_sms": (103,)}],
        sm_version=sm,
    )
    assert len(errors) == 1
    assert errors[0]["classification"] == "unverified_skipped"


def test_native_failure_records_survive_interruption_and_plain_resume(monkeypatch, tmp_path):
    import logging

    import collector.sglang.collect_kimi_k31_dsa_v1 as native
    from collector.collect import ReplaySpec, ResumeCheckpoint, load_replay_cases
    from collector.op_plans import ModuleCase

    cases = [ModuleCase(4096, 1, heads, "fp8", "bfloat16", "bfloat16", native.MODEL, 0) for heads in (96, 48)]
    calls = []
    report_dir = tmp_path / "logs"
    monkeypatch.setenv("COLLECTOR_LOG_DIR", str(report_dir))

    def run(*, cases, **kwargs):
        calls.extend(cases)
        if len(calls) == 1:
            return {
                "failed": cases,
                "failure_details": {
                    str(cases[0]): {
                        "error_type": "RuntimeError",
                        "error_message": "FIXME(kernel-limit): observed native refusal",
                    }
                },
            }
        if len(calls) == 2:
            # A real process/job interruption loses all in-memory errors.
            assert (report_dir / "errors_sglang.kimi_k31_dsa_context_module.json").exists()
            raise KeyboardInterrupt("job stopped during next case")
        return {"failed": []}

    class Checkpoint(ResumeCheckpoint):
        def mark_failed(self, task_id):
            saved = json.loads((report_dir / "errors_sglang.kimi_k31_dsa_context_module.json").read_text())
            assert any(error["task_id"] == task_id for error in saved)
            super().mark_failed(task_id)

    monkeypatch.setattr(native, "run_kimi_k31_dsa_fullnode", run)

    def collect():
        return fullnode.collect_sglang_fullnode_op(
            {
                "name": "sglang",
                "type": "kimi_k31_dsa_context_module",
                "module": native.__name__,
                "get_func": "get_kimi_k31_dsa_context_test_cases",
                "run_func": "run_kimi_k31_dsa_fullnode",
                "perf_filename": str(tmp_path / "context.txt"),
            },
            runtime_version=native.RUNTIME_VERSION,
            limit=None,
            shuffle=False,
            shuffle_seed=42,
            backend="sglang",
            resume_options={"checkpoint_dir": str(tmp_path / "checkpoint"), "resume": True},
            model_path=native.MODEL,
            case_plan=None,
            sm_version=103,
            case_filters=None,
            get_test_cases_for_model=lambda *_: cases,
            resume_checkpoint_cls=Checkpoint,
            logger=logging.getLogger(__name__),
        )

    with pytest.raises(KeyboardInterrupt):
        collect()
    report = report_dir / "errors_sglang.kimi_k31_dsa_context_module.json"
    before = json.loads(report.read_text())
    assert len(before) == 1
    error = before[0]
    assert error["task_params"] == str(cases[0])
    assert error["classification"] == "kernel_limit"
    assert error["error_type"] == "RuntimeError"
    assert "observed native refusal" in error["error_message"]
    assert json.loads(error["group"])["num_heads"] == 96
    requested = load_replay_cases(str(report))
    replay = ReplaySpec(str(report), requested)
    assert replay.select("kimi_k31_dsa_context_module", cases, random.Random(42)) == [cases[0]]
    assert collect() == before
    assert calls == [cases[0], cases[1], cases[1]]
    assert collect() == before  # zero-work resumes retain exact records too
