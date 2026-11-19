# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native K31 collection keeps its own architecture and real TP geometry."""

import copy
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.framework_manifest import get_collector_runtime, require_collector_runtime
from collector.op_plans import OP_PLANS, ModuleCase, plan
from collector.sglang.collect_kimi_k31_dsa_v1 import (
    ARCHITECTURE,
    MODEL,
    RUNTIME_VERSION,
    SOURCE_COMMIT,
    _extend_history,
    executed_kernel_source,
    perf_item,
    run_kimi_k31_dsa_fullnode,
    run_kimi_k31_dsa_worker,
    single_layer_config,
    validate_case,
    validate_forward_batch,
)
from collector.sglang.registry import REGISTRY
from collector.version_resolver import _check_compat, resolve_module

pytestmark = pytest.mark.unit


@pytest.fixture
def config():
    # Independently written contract fixture, not a checkpoint config copy.
    return {
        "architectures": [ARCHITECTURE],
        "auto_map": {"AutoConfig": "untrusted.CheckpointConfig"},
        "text_config": {
            "hidden_size": 7168,
            "num_attention_heads": 96,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "mla_use_output_gate": True,
            "num_hidden_layers": 93,
            "num_nextn_predict_layers": 5,
            "linear_attn_config": {"full_attn_layers": [4, 8], "kda_layers": [1, 2, 3]},
            "dsa_config": {
                "dsa_layers": [4, 8],
                "index_topk": 2048,
                "index_head_dim": 128,
                "index_num_heads": 32,
                "index_rope_dim": 0,
                "index_k_norm_type": "layernorm",
                "index_topk_freq": 1,
                "index_skip_topk_offset": 1,
            },
        },
    }


def test_native_builder_preserves_geometry_and_input_config(config):
    before = copy.deepcopy(config)
    result = single_layer_config(config)
    text = result["text_config"]
    assert config == before
    assert "auto_map" not in result
    assert text["num_attention_heads"] == 96
    assert text["num_hidden_layers"] == 1
    assert text["num_nextn_predict_layers"] == 0
    assert text["linear_attn_config"] == {"full_attn_layers": [1], "kda_layers": []}
    assert text["dsa_config"]["dsa_layers"] == [1]


@pytest.mark.parametrize(("field", "wrong"), [("index_rope_dim", 64), ("index_topk_freq", 2)])
def test_changed_upstream_indexer_contract_fails_closed(config, field, wrong):
    config["text_config"]["dsa_config"][field] = wrong
    with pytest.raises(ValueError, match=field):
        single_layer_config(config)


def test_case_plan_uses_native_tp_shards_and_supported_cache_precision(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL)
    for phase in ("context", "generation"):
        op = f"kimi_k31_dsa_{phase}_module"
        structural = OP_PLANS[op].expand()
        assert {case.num_heads for case in structural} == {96, 48, 24, 12}
        planned = plan(op, "sglang", 103)
        cases = planned.cases
        assert not planned.removed
        assert cases
        assert {case.num_heads for case in cases} == {96, 48, 24, 12}
        assert {case.kv_cache_dtype for case in cases} == {"fp8"}
        assert {case.gemm_type for case in cases} == {"bfloat16"}
        assert {validate_case(case) for case in cases} == {1, 2, 4, 8}
        assert all(case.seq_len + case.prefix_len <= 1048576 for case in cases)
        assert not plan(op, "vllm", 103).cases


def test_existing_dsa_key_represents_complete_decode_history():
    case = ModuleCase(8192, 4, 24, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    item = perf_item(case, "generation", 1.5)
    assert item["isl"] + item["step"] == 8192
    assert item["tp_size"] == 4
    assert item["architecture"] == ARCHITECTURE
    assert item["num_heads"] == 24


def test_stock_runtime_cannot_select_native_preview_collector():
    for entry in REGISTRY:
        if not entry.op.startswith("kimi_k31_dsa_"):
            continue
        runtime = require_collector_runtime(
            "sglang",
            RUNTIME_VERSION,
            requested_ops={entry.op},
            model_path=MODEL,
        )
        assert runtime.version == RUNTIME_VERSION
        assert runtime.source_commit == SOURCE_COMMIT
        assert "@sha256:" in runtime.image()
        assert entry.run_func == "run_kimi_k31_dsa_fullnode"
        assert not entry.unverified
        assert set(entry.unverified_sms) == {80, 86, 89, 90, 100, 120}
        assert resolve_module(entry, runtime.version) == "collector.sglang.collect_kimi_k31_dsa_v1"
        assert resolve_module(entry, get_collector_runtime("sglang").version) is None
        assert _check_compat(f"sglang=={RUNTIME_VERSION}", RUNTIME_VERSION)


def test_kernel_source_reports_executed_native_backend_and_eager_mode():
    backend = SimpleNamespace(use_mha=False, dsa_prefill_impl="flashmla_kv", dsa_decode_impl="flashmla_kv")
    hybrid = SimpleNamespace(full_attn_backend=backend)
    pool = SimpleNamespace(dsa_kv_cache_store_fp8=True, fp8_cast_on_write=False, kv_cache_dim=656)
    assert executed_kernel_source(hybrid, pool, "context") == "sglang_dsa_indexer_flashmla_kv_kv_flashmla_packed_eager"
    assert (
        executed_kernel_source(hybrid, pool, "generation") == "sglang_dsa_indexer_flashmla_kv_kv_flashmla_packed_eager"
    )
    backend.use_mha = True
    backend.device_sm_major = 10
    assert (
        executed_kernel_source(hybrid, pool, "context") == "sglang_dsa_dense_mha_trtllm_ragged_kv_flashmla_packed_eager"
    )
    pool.dsa_kv_cache_store_fp8 = False
    pool.fp8_cast_on_write = True
    pool.kv_cache_dim = 576
    with pytest.raises(RuntimeError, match=r"FIXME\(kernel-limit\).*raw FP8"):
        executed_kernel_source(hybrid, pool, "context")
    pool.kv_cache_dim = 656
    with pytest.raises(RuntimeError, match="physical layout"):
        executed_kernel_source(hybrid, pool, "context")


def test_native_runtime_rejects_bf16_cache():
    case = ModuleCase(4096, 1, 96, "bfloat16", "bfloat16", "bfloat16", MODEL, 0)
    with pytest.raises(ValueError, match="requires FP8 KV"):
        validate_case(case)


def test_collector_rejects_fabricated_local_head_geometry():
    case = ModuleCase(4096, 1, 16, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    with pytest.raises(ValueError, match="unsupported native"):
        validate_case(case)


@pytest.mark.parametrize(("heads", "tp"), [(48, 2), (24, 4), (12, 8)])
def test_multi_gpu_cases_require_standalone_exclusive_ownership(heads, tp, monkeypatch):
    case = ModuleCase(4096, 1, heads, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    assert validate_case(case) == tp
    import collector.sglang.collect_kimi_k31_dsa_v1 as module

    monkeypatch.setattr(module.subprocess, "Popen", lambda *_a, **_k: pytest.fail("launched an unowned GPU group"))
    with pytest.raises(RuntimeError, match="exclusive ownership"):
        run_kimi_k31_dsa_worker(*case, device="cuda:0", perf_filename="dsa_generation_module_perf.txt")


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_fullnode_runs_each_requested_tp_group_and_retains_exact_failed_case(monkeypatch, tmp_path, phase):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module

    cases = [ModuleCase(4096, 1, heads, "fp8", "bfloat16", "bfloat16", MODEL, 0) for heads in (96, 48, 24)]
    observed = []
    popen = subprocess.Popen

    def launch(command, **kwargs):
        case = ModuleCase(*json.loads(command[command.index("--case") + 1]))
        assert command[command.index("--device") + 1] == "cuda:0"
        assert command[command.index("--phase") + 1] == phase
        observed.append(case)
        return popen([sys.executable, "-c", f"raise SystemExit({int(case.num_heads == 48)})"], **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    # Directory names must not turn a generation run into a context run.
    output = tmp_path / "context_sweep" / f"dsa_{phase}_module_perf.txt"
    getter = f"get_kimi_k31_dsa_{phase}_test_cases"
    monkeypatch.setattr(module, getter, lambda: cases)
    other = "generation" if phase == "context" else "context"
    monkeypatch.setattr(module, f"get_kimi_k31_dsa_{other}_test_cases", lambda: pytest.fail("wrong phase getter"))
    result = run_kimi_k31_dsa_fullnode(perf_filename=output)
    assert observed == cases
    assert result["succeeded"] == 2
    assert result["failed"] == [cases[1]]
    assert result["failure_details"][str(cases[1])]["error_type"] == "RuntimeError"


@pytest.mark.parametrize(
    ("reason", "classification"),
    [
        ("FIXME(kernel-limit): does not fit: native TP2 requires 2 available GPUs", "kernel_limit"),
        ("native initialization failed", "unexpected"),
        ("native failure mentioning FIXME(kernel-limit): is not a capacity refusal", "unexpected"),
    ],
)
def test_native_child_preserves_failure_classification_and_output(reason, classification, monkeypatch, capfd, tmp_path):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module

    monkeypatch.syspath_prepend(str(Path(module.__file__).parents[1]))
    from collect import _classify_exception

    popen = subprocess.Popen

    def launch(_command, **kwargs):
        return popen(
            [sys.executable, "-c", f"print('native stdout', flush=True); raise RuntimeError({reason!r})"], **kwargs
        )

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    case = ModuleCase(4096, 1, 96, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    with pytest.raises(RuntimeError) as error:
        run_kimi_k31_dsa_worker(*case, perf_filename=tmp_path / "dsa_generation_module_perf.txt")
    assert _classify_exception(error.value) == classification
    assert reason in str(error.value)
    if classification == "kernel_limit":
        assert str(error.value) == reason
    else:
        assert str(error.value).startswith("native K31 worker exited 1:")
    output = capfd.readouterr()
    assert "native stdout" in output.out
    assert f"RuntimeError: {reason}" in output.err


def test_native_success_keeps_stderr_visible(monkeypatch, capfd, tmp_path):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module

    popen = subprocess.Popen
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda _command, **kwargs: popen(
            [sys.executable, "-c", "import sys; print('native warning', file=sys.stderr)"], **kwargs
        ),
    )
    case = ModuleCase(4096, 1, 96, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    run_kimi_k31_dsa_worker(*case, perf_filename=tmp_path / "dsa_context_module_perf.txt")
    assert "native warning" in capfd.readouterr().err


@pytest.mark.skipif(os.name != "posix", reason="native SGLang TP workers use POSIX process groups")
def test_native_timeout_kills_tp_children_and_fullnode_continues(monkeypatch, capfd, tmp_path):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module
    import psutil

    popen = subprocess.Popen
    children = []
    rank_pid = tmp_path / "rank.pid"
    cases = [ModuleCase(4096, 1, heads, "fp8", "bfloat16", "bfloat16", MODEL, 0) for heads in (48, 96)]

    def launch(command, **kwargs):
        case = ModuleCase(*json.loads(command[command.index("--case") + 1]))
        code = ""
        if case.num_heads == 48:
            code = (
                "import subprocess, sys, time; from pathlib import Path; "
                "rank = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                f"Path({str(rank_pid)!r}).write_text(str(rank.pid)); "
                "print('waiting on TP rank', file=sys.stderr, flush=True); time.sleep(60)"
            )
        child = popen([sys.executable, "-c", code], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(module, "NATIVE_CASE_TIMEOUT_S", 1)
    monkeypatch.setattr(module.subprocess, "Popen", launch)
    result = run_kimi_k31_dsa_fullnode(perf_filename=tmp_path / "dsa_generation_module_perf.txt", cases=cases)
    assert result["succeeded"] == 1
    assert result["failed"] == [cases[0]]
    assert "timed out" in result["failure_details"][str(cases[0])]["error_message"]
    assert all(child.poll() is not None for child in children)
    pid = int(rank_pid.read_text())
    try:
        status = psutil.Process(pid).status()
    except psutil.NoSuchProcess:
        status = None
    assert status in (None, psutil.STATUS_ZOMBIE)
    assert "waiting on TP rank" in capfd.readouterr().err


@pytest.mark.skipif(os.name != "posix", reason="native SGLang TP workers use POSIX process groups")
@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, RuntimeError])
def test_native_exception_kills_launcher_and_tp_children(exception_type, monkeypatch, capfd, caplog, tmp_path):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module
    import psutil

    popen = subprocess.Popen
    children = []
    rank_pid = tmp_path / "rank.pid"
    error = exception_type("interrupted native communication")

    def launch(_command, **kwargs):
        code = (
            "import subprocess, sys, time; from pathlib import Path; "
            "rank = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "print('native cancellation diagnostic', file=sys.stderr, flush=True); "
            f"Path({str(rank_pid)!r}).write_text(str(rank.pid)); time.sleep(60)"
        )
        child = popen([sys.executable, "-c", code], **kwargs)
        children.append(child)
        communicate = child.communicate
        interrupted = False

        def interrupt_once(*args, **kwargs):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                deadline = time.monotonic() + 5
                while not rank_pid.exists() or not rank_pid.read_text().strip():
                    assert time.monotonic() < deadline, "TP rank did not start"
                    time.sleep(0.01)
                raise error
            return communicate(*args, **kwargs)

        monkeypatch.setattr(child, "communicate", interrupt_once)
        return child

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    case = ModuleCase(4096, 1, 48, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    try:
        kwargs = {"cases": [case], "perf_filename": tmp_path / "dsa_generation_module_perf.txt"}
        if issubclass(exception_type, Exception):
            result = run_kimi_k31_dsa_fullnode(**kwargs)
            assert result["failed"] == [case]
            assert result["failure_details"][str(case)]["error_message"] == str(error)
            assert any(record.exc_info and record.exc_info[1] is error for record in caplog.records)
        else:
            with pytest.raises(exception_type) as raised:
                run_kimi_k31_dsa_fullnode(**kwargs)
            assert raised.value is error
        assert children[0].poll() is not None
        try:
            status = psutil.Process(int(rank_pid.read_text())).status()
        except psutil.NoSuchProcess:
            status = None
        assert status in (None, psutil.STATUS_ZOMBIE)
        assert "native cancellation diagnostic" in capfd.readouterr().err
    finally:
        # A regression must not leave the test's real process group running.
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)


def test_native_forward_batch_must_match_persisted_prefix_and_history():
    case = ModuleCase(512, 1, 96, "fp8", "bfloat16", "bfloat16", MODEL, 2048)
    batch = SimpleNamespace(
        input_ids=SimpleNamespace(numel=lambda: 512),
        seq_lens=SimpleNamespace(tolist=lambda: [2560]),
        extend_prefix_lens_cpu=[2048],
        extend_seq_lens_cpu=[512],
    )
    assert validate_forward_batch(batch, case, "context") == {
        "seq_lens": [2560],
        "extend_prefix_lens": [2048],
        "extend_seq_lens": [512],
    }
    batch.extend_prefix_lens_cpu = [0]
    with pytest.raises(RuntimeError, match="cached-prefix"):
        validate_forward_batch(batch, case, "context")
    batch.seq_lens = SimpleNamespace(tolist=lambda: [512])
    with pytest.raises(RuntimeError, match="KV history"):
        validate_forward_batch(batch, case, "context")


def test_native_history_continuation_covers_every_token_without_replacing_requests():
    reqs = [SimpleNamespace(kv=SimpleNamespace(req_pool_idx=i + 1)) for i in range(2)]
    observed = [[] for _ in reqs]

    def extend(actual):
        assert actual is reqs
        assert sum(req.extend_end - req.prefix_len for req in actual) <= 6
        for index, req in enumerate(actual):
            observed[index].extend(range(req.prefix_len, req.extend_end))
        return (
            [req.extend_end for req in actual],
            None,
            SimpleNamespace(
                seq_lens_cpu=SimpleNamespace(tolist=lambda: [req.extend_end for req in actual]),
                prefix_lens=[req.prefix_len for req in actual],
                extend_lens=[req.extend_end - req.prefix_len for req in actual],
            ),
        )

    tokens, _, receipt = _extend_history(SimpleNamespace(extend=extend), reqs, 7, chunk_tokens=6)
    assert observed == [list(range(7)), list(range(7))]
    assert tokens == [7, 7]
    assert receipt == {
        "protocol": "native-req-continuation-v1",
        "chunk_tokens": 6,
        "chunks": 3,
        "history_per_request": 7,
        "request_slots_preserved": True,
    }


@pytest.mark.parametrize("failure", ["prefix", "length", "request_slot", "unallocated", "native_error"])
def test_native_history_continuation_fails_closed(failure):
    req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=1))
    calls = 0

    def extend(reqs):
        nonlocal calls
        calls += 1
        lengths = [req.extend_end]
        prefixes = [req.prefix_len]
        if calls == 2:
            if failure == "native_error":
                raise RuntimeError("native prefill failed")
            if failure == "prefix":
                prefixes = [0]
            if failure == "length":
                lengths = [1]
            if failure == "request_slot":
                req.kv.req_pool_idx = 2
            if failure == "unallocated":
                req.kv.req_pool_idx = None
        return (
            [0],
            None,
            SimpleNamespace(
                seq_lens_cpu=SimpleNamespace(tolist=lambda: lengths),
                prefix_lens=prefixes,
                extend_lens=[req.extend_end - req.prefix_len],
            ),
        )

    with pytest.raises(RuntimeError, match="native"):
        _extend_history(SimpleNamespace(extend=extend), [req], 7, chunk_tokens=3)
    assert calls == 2


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_capacity_refuses_before_native_allocation_and_checks_every_rank(phase):
    from collector.sglang.collect_kimi_k31_dsa_v1 import _check_capacity

    case = ModuleCase(1048576, 16, 48, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    visited = []

    def memory(gpu):
        visited.append(gpu)
        return (2**40 if gpu == 2 else 1, 2**40)

    with pytest.raises(RuntimeError, match=r"^FIXME\(kernel-limit\): does not fit.*cuda:3"):
        _check_capacity(case, phase, SimpleNamespace(mem_get_info=memory), 2)
    assert visited == [2, 3]
    _check_capacity(case, phase, SimpleNamespace(mem_get_info=lambda _gpu: (2**40, 2**40)), 0)


@pytest.mark.parametrize("sm", [80, 86, 89])
@pytest.mark.parametrize("phase", ["context", "generation"])
def test_native_k31_plan_applies_pre_hopper_floor(sm, phase, monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL)
    planned = plan(f"kimi_k31_dsa_{phase}_module", "sglang", sm)
    assert not planned.cases
    assert planned.removed


def test_native_fullnode_failure_reaches_persisted_artifact(monkeypatch, tmp_path):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module
    from collector import fullnode
    from collector.collect import ResumeCheckpoint

    case = ModuleCase(4096, 1, 12, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    reason = "FIXME(kernel-limit): does not fit: native TP8 requires 8 available GPUs"

    def fail(*_args):
        raise RuntimeError(reason)

    monkeypatch.setattr(module, "_run_native_case", fail)
    errors = fullnode.collect_sglang_fullnode_op(
        {
            "name": "sglang",
            "type": "kimi_k31_dsa_context_module",
            "module": module.__name__,
            "get_func": "get_kimi_k31_dsa_context_test_cases",
            "run_func": "run_kimi_k31_dsa_fullnode",
            "perf_filename": str(tmp_path / "context.txt"),
        },
        runtime_version=RUNTIME_VERSION,
        limit=None,
        shuffle=False,
        shuffle_seed=42,
        backend="sglang",
        resume_options={"checkpoint_dir": str(tmp_path / "checkpoint")},
        model_path=MODEL,
        case_plan=None,
        sm_version=103,
        case_filters=None,
        get_test_cases_for_model=lambda *_args: [case],
        resume_checkpoint_cls=ResumeCheckpoint,
        logger=logging.getLogger(__name__),
    )
    assert len(errors) == 1
    assert errors[0]["error_message"] == reason
    assert errors[0]["error_type"] == "RuntimeError"
    assert errors[0]["classification"] == "kernel_limit"
