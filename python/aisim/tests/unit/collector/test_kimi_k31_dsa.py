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
from itertools import pairwise
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


@pytest.mark.parametrize("model_path", [None, MODEL])
def test_case_plan_uses_native_tp_shards_and_supported_cache_precision(monkeypatch, model_path):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module

    if model_path is None:
        monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    else:
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    monkeypatch.setattr(module, "get_sm_version", lambda: 103)
    for phase in ("context", "generation"):
        op = f"kimi_k31_dsa_{phase}_module"
        structural = OP_PLANS[op].expand()
        assert len(set(structural)) == len(structural)
        assert {case.num_heads for case in structural} == {96, 48, 24, 12}
        planned = plan(op, "sglang", 103)
        cases = planned.cases
        assert cases == getattr(module, f"get_kimi_k31_dsa_{phase}_test_cases")() == structural
        assert not planned.removed
        assert cases
        assert {case.num_heads for case in cases} == {96, 48, 24, 12}
        assert {case.kv_cache_dtype for case in cases} == {"fp8"}
        assert {case.gemm_type for case in cases} == {"bfloat16"}
        assert {validate_case(case, phase) for case in cases} == {1, 2, 4, 8}
        assert all(case.seq_len + case.prefix_len <= 1048576 for case in cases)
        assert not plan(op, "vllm", 103).cases


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_explicit_native_request_rejected_before_stock_route_filtering(phase):
    from collector.version_resolver import build_collections

    op = f"kimi_k31_dsa_{phase}_module"
    with pytest.raises(RuntimeError, match="native Kimi K3.1 requires exactly"):
        build_collections(REGISTRY, "sglang", "0.5.21", [op])
    ordinary = build_collections(REGISTRY, "sglang", "0.5.21")
    assert ordinary
    assert not any(entry["type"].startswith("kimi_k31_dsa_") for entry in ordinary)


@pytest.mark.parametrize("model_scoped", [False, True])
def test_cli_generated_plan_keeps_native_ops_model_scoped(model_scoped):
    import collect
    from collector.version_resolver import build_collections

    selection = ["--model-path", MODEL] if model_scoped else ["--model-cases-full"]
    result = subprocess.run(
        [sys.executable, collect.__file__, "--backend", "sglang", "--plan-only", *selection],
        check=True,
        capture_output=True,
        text=True,
    )
    ops = json.loads(result.stdout)["ops"]
    native = {f"kimi_k31_dsa_{phase}_module" for phase in ("context", "generation")}
    if model_scoped:
        assert set(ops) == native
        collections = build_collections(REGISTRY, "sglang", RUNTIME_VERSION, ops)
        assert {entry["type"] for entry in collections} == native
    else:
        assert native.isdisjoint(ops)
        assert "gemm" in ops
        assert build_collections(REGISTRY, "sglang", "0.5.21", ops)


@pytest.mark.parametrize("ambient", [None, "0", "1", "512", "8192"])
@pytest.mark.parametrize(
    ("variable", "expected"),
    [
        ("SGLANG_DSA_FUSE_TOPK", "1"),
        ("SGLANG_OPT_USE_TOPK_V2", "1"),
        ("SGLANG_DSA_TOPK_BROADCAST", "0"),
        ("SGLANG_FP8_PAGED_MQA_LOGITS_TORCH", "0"),
        ("SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR", None),
        ("SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD", "2048"),
        ("SGLANG_MAX_KV_CHUNK_CAPACITY", "131072"),
        ("SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN", "0"),
    ],
)
def test_native_dispatch_setting_is_fixed_for_child_and_standalone(monkeypatch, tmp_path, ambient, variable, expected):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module

    if ambient is None:
        monkeypatch.delenv(variable, raising=False)
    else:
        monkeypatch.setenv(variable, ambient)
    popen = subprocess.Popen
    output = tmp_path / "dsa_generation_module_perf.txt"

    def launch(command, **kwargs):
        stage = command[command.index("--output") + 1]
        code = f"import os; from pathlib import Path; Path({stage!r}).write_text(repr(os.environ.get({variable!r})))"
        return popen([sys.executable, "-c", code], **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    case = ModuleCase(8192, 1, 96, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    monkeypatch.setattr(
        sys, "argv", ["collector", "--case", json.dumps(list(case)), "--phase", "generation", "--output", str(output)]
    )
    module.main()
    assert output.read_text() == repr(expected)
    assert os.environ.get(variable) == ambient
    monkeypatch.setattr(module.os, "environ", dict(os.environ))
    module.os.environ[variable] = ambient or "0"

    class RuntimeReached(Exception):
        pass

    def runtime():
        assert os.environ.get(variable) == expected
        raise RuntimeReached

    monkeypatch.setattr(module, "_assert_runtime", runtime)
    with module.native_publication_lock(output) as fd:
        sys.argv.extend(["--publication-lock-fd", str(fd)])
        with pytest.raises(RuntimeReached):
            module.main()


def test_standalone_recovers_retained_output_from_package_entrypoint(tmp_path):
    package = Path(__file__).resolve().parents[3]
    output = tmp_path / "result"
    script = """
import json, sys
from pathlib import Path
from collector.sglang import collect_kimi_k31_dsa_v1 as native

out = Path(sys.argv[1])
native.collector_retained_path(out).write_bytes(b"prior")
native._run_native_case = lambda case, device, stage, **options: stage.write_bytes(b"measured")
sys.argv = ["native", "--phase", "context", "--output", str(out), "--case",
            json.dumps([4096, 1, 96, "fp8", "bfloat16", "bfloat16", native.MODEL, 0])]
native.main()
assert out.read_bytes() == b"measured"
"""
    subprocess.run(
        [sys.executable, "-c", script, str(output)],
        cwd=package,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(package), str(package / "src")])},
        check=True,
        timeout=30,
    )


def test_native_cleanup_deadline_stops_allocation(monkeypatch, tmp_path):
    import io

    import collector.sglang.collect_kimi_k31_dsa_v1 as native

    timeouts = []

    def stalled(timeout):
        timeouts.append(timeout)
        raise subprocess.TimeoutExpired("worker", timeout)

    pipe = io.StringIO()
    monkeypatch.setattr(
        native.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=123, stderr=pipe, communicate=stalled)
    )
    monkeypatch.setattr(native.os, "killpg", lambda pid, sig: None)
    case = ModuleCase(4096, 1, 96, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    with pytest.raises(native.NativeAllocationFailure, match="stop this allocation"):
        native._run_native_case(case, "cuda:0", tmp_path / "stage")
    assert timeouts == [native.NATIVE_CASE_TIMEOUT_S, native.NATIVE_CLEANUP_TIMEOUT_S]
    assert pipe.closed


@pytest.mark.parametrize("existing", [False, True])
def test_sigterm_cleanup_timeout_preserves_live_stage_and_writer_lock(monkeypatch, tmp_path, existing):
    import io

    import collector.sglang.collect_kimi_k31_dsa_v1 as native

    stage = tmp_path / "stage"
    writer_lock = Path(str(stage) + ".lock")
    if existing:
        stage.write_bytes(b"prior rows\n")
    timeouts = []
    original_handler = signal.getsignal(signal.SIGTERM)

    def stalled(timeout):
        timeouts.append(timeout)
        if len(timeouts) == 1:
            with stage.open("ab") as stream:
                stream.write(b"unconfirmed row\n")
            writer_lock.write_bytes(b"live writer")
            os.kill(os.getpid(), signal.SIGTERM)
        raise subprocess.TimeoutExpired("worker", timeout)

    monkeypatch.setattr(
        native.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=123, stderr=io.StringIO(), communicate=stalled)
    )
    monkeypatch.setattr(native.os, "killpg", lambda pid, sig: None)
    case = ModuleCase(4096, 1, 96, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    with pytest.raises(native.NativeAllocationFailure, match="stop this allocation"):
        native._run_native_case(case, "cuda:0", stage)
    assert timeouts == [native.NATIVE_CASE_TIMEOUT_S, native.NATIVE_CLEANUP_TIMEOUT_S]
    assert stage.read_bytes() == (b"prior rows\n" if existing else b"") + b"unconfirmed row\n"
    assert writer_lock.read_bytes() == b"live writer"
    assert signal.getsignal(signal.SIGTERM) == original_handler


def test_physical_sm_validation_checks_every_selected_device():
    from collector.sglang.collect_kimi_k31_dsa_v1 import _check_device_sm

    cuda = SimpleNamespace(get_device_capability=lambda gpu: (10, 3) if gpu == 1 else (10, 0))
    _check_device_sm(cuda, 1, 1, 103)
    with pytest.raises(RuntimeError, match="cuda:2 has SM100, expected SM103"):
        _check_device_sm(cuda, 1, 2, 103)


def test_standalone_shares_supervisor_lock(tmp_path, monkeypatch):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module

    output = tmp_path / "rows"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collector",
            "--case",
            json.dumps([4096, 1, 96, "fp8", "bfloat16", "bfloat16", MODEL, 0]),
            "--phase",
            "generation",
            "--output",
            str(output),
        ],
    )
    calls = []
    monkeypatch.setattr(module, "_collect_native", lambda args, *_: calls.append(args.output))
    with module.native_publication_lock(output) as fd:
        with pytest.raises(RuntimeError, match="publication owner"):
            module.main()
        sys.argv.extend(["--publication-lock-fd", str(fd)])
        module.main()
    assert calls == [str(output)]


@pytest.mark.parametrize("sm", [90, 103, 107])
def test_native_plan_lists_only_qualified_hardware(monkeypatch, sm):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL)
    for op in ("kimi_k31_dsa_context_module", "kimi_k31_dsa_generation_module"):
        assert bool(plan(op, "sglang", sm).cases) == (sm == 103)


def test_standalone_preserves_pending_fullnode_publication(monkeypatch, tmp_path):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module

    output = tmp_path / "generation.txt"
    output.write_text("previous row")
    receipt = tmp_path / ".generation.txt.pending"
    receipt.write_text("outstanding full-node receipt")
    case = ModuleCase(4096, 1, 96, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    monkeypatch.setattr(
        sys, "argv", ["collector", "--case", json.dumps(list(case)), "--phase", "generation", "--output", str(output)]
    )
    for invoke in (module.main, lambda: module._run_native_case(case, "cuda:0", output)):
        with pytest.raises(RuntimeError, match="Resume the full-node job"):
            invoke()
    assert receipt.read_text() == "outstanding full-node receipt"
    assert output.read_text() == "previous row"


@pytest.mark.parametrize("alias", [False, True])
def test_supervisor_and_child_share_the_target_lock(tmp_path, monkeypatch, alias):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module

    target = tmp_path / "rows.txt"
    output = tmp_path / "generation.txt" if alias else target
    if alias:
        output.symlink_to(target)
    popen = subprocess.Popen

    def launch(command, **kwargs):
        child_path = command[command.index("--output") + 1]
        fd = int(command[command.index("--publication-lock-fd") + 1])
        assert child_path == str(target)
        code = (
            "from pathlib import Path\nfrom collector.helper import native_publication_lock\n"
            f"path = Path({child_path!r})\nwith native_publication_lock(path, {fd}): path.write_text('measured')"
        )
        return popen([sys.executable, "-c", code], **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    case = ModuleCase(4096, 1, 96, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    module._run_native_case(case, "cuda:0", output)
    assert output.read_text() == "measured"
    assert output.is_symlink() == alias


@pytest.mark.skipif(os.name != "posix", reason="native SGLang TP workers use POSIX process groups")
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("outcome", ["success", "rank_failure", "timeout"])
def test_standalone_bounds_rank_join_and_publishes_only_success(tmp_path, monkeypatch, existing, outcome):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module
    import psutil

    # An arbitrary basename ensures the explicit context phase is forwarded.
    output = tmp_path / "rows.txt"
    initial = "existing row\n" if existing else ""
    if existing:
        output.write_text(initial)
    rank_pid = tmp_path / "rank.pid"
    release = tmp_path / "release"
    rank_code = (
        "import os, time\nfrom pathlib import Path\n"
        f"Path({str(rank_pid)!r}).write_text(str(os.getpid()))\n"
        f"while not Path({str(release)!r}).exists(): time.sleep(0.01)\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collector",
            "--case",
            json.dumps([4096, 1, 48, "fp8", "bfloat16", "bfloat16", MODEL, 0]),
            "--phase",
            "context",
            "--output",
            str(output),
            "--device",
            "cuda:1",
            "--sm",
            "103",
        ],
    )
    popen = subprocess.Popen
    children = []
    attempt_outcome = outcome

    def launch(command, **kwargs):
        rank_pid.unlink(missing_ok=True)
        release.unlink(missing_ok=True)
        # Exercise the real child entrypoint and inherited stage lock, replacing
        # only GPU measurement with a writer and a rank whose join can stall.
        code = (
            "import subprocess, sys\nfrom pathlib import Path\n"
            "import collector.sglang.collect_kimi_k31_dsa_v1 as module\n"
            "def ranks(args, case, tp):\n"
            "    assert tp == 2 and args.phase == 'context'\n"
            "    assert args.device == 'cuda:1' and args.sm == 103\n"
            f"    assert Path(args.output) != Path({str(output)!r})\n"
            "    with Path(args.output).open('a') as stream: stream.write('new row\\n')\n"
            f"    rank = subprocess.Popen([sys.executable, '-c', {rank_code!r}])\n"
            "    rank.wait()\n"
            f"    if {attempt_outcome == 'rank_failure'!r}: raise RuntimeError('rank 1 failed after rank 0 wrote')\n"
            "module._collect_native = ranks\nmodule.main()\n"
        )
        child = popen([sys.executable, "-c", code, *command[3:]], **kwargs)
        children.append(child)
        deadline = time.monotonic() + 30
        while not rank_pid.exists() or not rank_pid.read_text().strip():
            assert time.monotonic() < deadline, "TP rank did not start"
            assert child.poll() is None, "native launcher exited before starting its rank"
            time.sleep(0.01)
        assert child.poll() is None
        assert output.exists() == existing
        if existing:
            assert output.read_text() == initial
        with pytest.raises(RuntimeError, match="publication owner"), module.native_publication_lock(output):
            pass
        if attempt_outcome != "timeout":
            release.touch()
        # Start the execution deadline after interpreter/rank startup.
        monkeypatch.setattr(module, "NATIVE_CASE_TIMEOUT_S", 0.1 if attempt_outcome == "timeout" else 30)
        return child

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    monkeypatch.setattr(module, "_collect_native", lambda *_: pytest.fail("standalone bypassed supervision"))
    try:
        if outcome == "success":
            module.main()
        else:
            error, message = (
                (subprocess.TimeoutExpired, "timed out") if outcome == "timeout" else (RuntimeError, "rank 1 failed")
            )
            with pytest.raises(error, match=message):
                module.main()
            assert output.exists() == existing
            if existing:
                assert output.read_text() == initial
        assert all(child.poll() is not None for child in children)
        try:
            status = psutil.Process(int(rank_pid.read_text())).status()
        except psutil.NoSuchProcess:
            status = None
        assert status in (None, psutil.STATUS_ZOMBIE)
        assert not list(tmp_path.glob(".rows.txt.stage-*"))
        with module.native_publication_lock(output):
            pass
        if outcome != "success":
            # A retry must acquire the same output and publish exactly one row.
            attempt_outcome = "success"
            module.main()
        assert output.read_text() == initial + "new row\n"
    finally:
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)


@pytest.mark.parametrize("existing", [False, True])
def test_native_failure_restores_staging_existence_and_length(monkeypatch, tmp_path, existing):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module

    output = tmp_path / "dsa_context_module_perf.txt"
    if existing:
        output.write_text("previous row\n")
    popen = subprocess.Popen

    def launch(_command, **kwargs):
        return popen(
            [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(output)!r}).open('a').write('partial row'); exit(1)",
            ],
            **kwargs,
        )

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    case = ModuleCase(4096, 1, 96, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    with pytest.raises(RuntimeError, match="native K31 worker exited"):
        module._run_native_case(case, "cuda:0", output)
    assert output.exists() == existing
    if existing:
        assert output.read_text() == "previous row\n"


@pytest.mark.parametrize("phase", ["context", "generation"])
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tensor_parallel_sizes", [], "non-empty tensor_parallel_sizes"),
        ("kv_cache_dtypes", [], "non-empty kv_cache_dtypes"),
        ("kv_cache_dtypes", ["bfloat16"], "requires FP8 KV"),
        ("kv_cache_dtypes", ["fp8", "bfloat16"], "requires FP8 KV"),
        ("compute_dtype", "float32", "requires BF16 compute and projections"),
        ("gemm_type", "fp8", "requires BF16 compute and projections"),
        ("native_num_heads", 192, "requires native_num_heads=96"),
        ("tensor_parallel_sizes", [3], "invalid native Kimi K3.1 TP geometry"),
        ("tensor_parallel_sizes", [16], "invalid native Kimi K3.1 TP geometry"),
    ],
)
def test_native_plan_rejects_unresolvable_declarations(monkeypatch, phase, field, value, message):
    import collector.op_plans as module

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL)
    rows = [dict(row) for row in module._model_case_values("kimi_k31_dsa_module")]
    assert rows
    rows[0][field] = value
    if field == "native_num_heads":
        rows[0]["tensor_parallel_sizes"] = [2]  # 192 / TP2 must not masquerade as native TP1.
    monkeypatch.setattr(module, "_model_case_values", lambda _op: rows)
    with pytest.raises(ValueError, match=message):
        OP_PLANS[f"kimi_k31_dsa_{phase}_module"].expand()


@pytest.mark.parametrize(("length", "prefix"), [(1, 0), (8192, 64)])
def test_generation_coordinates_fail_before_native_runtime(monkeypatch, tmp_path, length, prefix):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module

    case = ModuleCase(length, 1, 96, "fp8", "bfloat16", "bfloat16", MODEL, prefix)
    assert validate_case(case, "context") == 1
    monkeypatch.setattr(module, "_assert_runtime", lambda: pytest.fail("loaded native runtime"))
    monkeypatch.setattr(
        sys, "argv", ["collector", "--case", json.dumps(list(case)), "--phase", "generation", "--output", str(tmp_path)]
    )
    with pytest.raises(ValueError, match="generation requires"):
        module.main()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_a, **_k: pytest.fail("launched invalid native case"))
    with pytest.raises(ValueError, match="generation requires"):
        module._run_native_case(case, "cuda:0", tmp_path / "dsa_generation_module_perf.txt")
    assert validate_case(case._replace(seq_len=8192, prefix_len=0), "generation") == 1


@pytest.mark.parametrize("op", ["kimi_k31_dsa_context_module", "kimi_k31_dsa_generation_module"])
@pytest.mark.parametrize("version", [RUNTIME_VERSION, "0.0.0.dev1+other", "0.0.0.dev1", "0.5.21"])
def test_native_registry_checks_full_runtime_before_queueing(monkeypatch, op, version):
    import importlib.metadata

    from collector import collect as module
    from collector.sglang.registry import require_kimi_k31_runtime

    if version == RUNTIME_VERSION:
        require_kimi_k31_runtime(version, {op})
    else:
        from collector.framework_manifest import resolve_op_runtime

        stock = resolve_op_runtime("sglang", op).version
        version = stock if version == "0.5.21" else version
        monkeypatch.setattr(importlib.metadata, "version", lambda _: version)
        monkeypatch.setattr(module, "collect_ops", lambda *_a, **_k: pytest.fail("queued pool cases"))
        import collector.fullnode

        monkeypatch.setattr(
            collector.fullnode, "collect_sglang_fullnode_op", lambda *_a, **_k: pytest.fail("queued native cases")
        )
        with pytest.raises(RuntimeError, match="native Kimi K3.1 requires exactly"):
            module.collect_sglang(1, ops=[op], model_path=None if version == stock else MODEL, sm_version=103)
    # Other ops keep the shared public-version policy.
    from collector.framework_manifest import require_collector_runtime, resolve_op_runtime

    stock = resolve_op_runtime("sglang", "gemm")
    assert require_collector_runtime("sglang", stock.version + "+vendor", requested_ops={"gemm"}) == stock


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
        assert set(entry.unverified_sms) == {80, 86, 89, 90, 100, 120, 121}
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
        output = command[command.index("--output") + 1]
        if case.num_heads == 24:
            assert Path(output).read_text() == "synthetic result row\n"
        script = (
            "from pathlib import Path; "
            f"p=Path({output!r}); p.parent.mkdir(parents=True, exist_ok=True); "
            f"p.open('a').write('synthetic result row\\n'); raise SystemExit({int(case.num_heads == 48)})"
        )
        return popen([sys.executable, "-c", script], **kwargs)

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
    assert output.read_text() == "synthetic result row\n" * 2


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
    output = tmp_path / "dsa_context_module_perf.txt"
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda _command, **kwargs: popen(
            [
                sys.executable,
                "-c",
                "import sys; from pathlib import Path; "
                f"Path({str(output)!r}).write_text('synthetic row'); print('native warning', file=sys.stderr)",
            ],
            **kwargs,
        ),
    )
    case = ModuleCase(4096, 1, 96, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    run_kimi_k31_dsa_worker(*case, perf_filename=output)
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
        output = command[command.index("--output") + 1]
        code = f"from pathlib import Path; Path({output!r}).write_text('synthetic row')"
        if case.num_heads == 48:
            rank_code = (
                "import os, time; from pathlib import Path; "
                f"Path({str(rank_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"
            )
            code = (
                "import subprocess, sys, time; from pathlib import Path; "
                f"rank = subprocess.Popen([sys.executable, '-c', {rank_code!r}]); "
                "print('waiting on TP rank', file=sys.stderr, flush=True); time.sleep(60)"
            )
        child = popen([sys.executable, "-c", code], **kwargs)
        children.append(child)
        # Interpreter startup is outside each execution deadline. The rank
        # itself publishes readiness before the hang timeout begins.
        if case.num_heads == 48:
            deadline = time.monotonic() + 30
            while not rank_pid.exists() or not rank_pid.read_text().strip():
                assert time.monotonic() < deadline, "TP rank did not start"
                time.sleep(0.01)
            monkeypatch.setattr(module, "NATIVE_CASE_TIMEOUT_S", 1)
        else:
            deadline = time.monotonic() + 30
            while not Path(output).exists():
                assert time.monotonic() < deadline, "success control did not start"
                time.sleep(0.01)
            monkeypatch.setattr(module, "NATIVE_CASE_TIMEOUT_S", 30)
        return child

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    try:
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
    finally:
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)


@pytest.mark.skipif(os.name != "posix", reason="native SGLang TP workers use POSIX process groups")
@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, RuntimeError])
def test_native_exception_kills_launcher_and_tp_children(exception_type, monkeypatch, capfd, caplog, tmp_path):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module
    import psutil

    popen = subprocess.Popen
    children = []
    rank_pid = tmp_path / "rank.pid"
    error = exception_type("interrupted native communication")
    output = tmp_path / "dsa_generation_module_perf.txt"
    output.write_text("previous successful row\n")

    def launch(_command, **kwargs):
        code = (
            "import subprocess, sys, time; from pathlib import Path; "
            "rank = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "print('native cancellation diagnostic', file=sys.stderr, flush=True); "
            f"Path({str(output)!r}).open('a').write('failed partial row\\n'); "
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
        kwargs = {"cases": [case], "perf_filename": output}
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
        assert output.read_text() == "previous successful row\n"
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


@pytest.mark.skipif(os.name != "posix", reason="native SGLang TP workers use POSIX process groups")
@pytest.mark.parametrize("during_launch", [False, True])
def test_parent_only_sigterm_kills_native_group_and_rolls_back(tmp_path, during_launch):
    import psutil

    output = tmp_path / "dsa_generation_module_perf.txt"
    output.write_text("previous row\n")
    rank_pid = tmp_path / "rank.pid"
    launcher_pid = tmp_path / "launcher.pid"
    rank_code = (
        "import os, time; from pathlib import Path; "
        f"Path({str(rank_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    launcher_code = (
        "import os, subprocess, sys, time; from pathlib import Path; "
        f"Path({str(output)!r}).open('a').write('partial row'); "
        f"Path({str(launcher_pid)!r}).write_text(str(os.getpid())); "
        f"subprocess.Popen([sys.executable, '-c', {rank_code!r}]); time.sleep(60)"
    )
    driver = tmp_path / "collector_driver.py"
    driver.write_text(
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "import collector.sglang.collect_kimi_k31_dsa_v1 as module\n"
        "import signal\n"
        "real_signal = signal.signal\n"
        "def install(sig, handler):\n"
        "    if not callable(handler): return real_signal(sig, handler)\n"
        "    def observed(*args):\n"
        "        handler(*args)\n"
        f"        Path({str(tmp_path / 'handled')!r}).touch()\n"
        "    return real_signal(sig, observed)\n"
        "signal.signal = install\n"
        "real_popen = subprocess.Popen\n"
        "def launch(_command, **kwargs):\n"
        f"    child = real_popen([sys.executable, '-c', {launcher_code!r}], **kwargs)\n"
        f"    while {during_launch!r} and not Path({str(tmp_path / 'release')!r}).exists(): time.sleep(0.01)\n"
        "    communicate = child.communicate\n"
        "    def observed_communicate(*args, **kwargs):\n"
        f"        Path({str(tmp_path / 'owned')!r}).touch()\n"
        "        return communicate(*args, **kwargs)\n"
        "    child.communicate = observed_communicate\n"
        "    return child\n"
        "module.subprocess.Popen = launch\n"
        f"case = module.ModuleCase(4096, 1, 48, 'fp8', 'bfloat16', 'bfloat16', {MODEL!r}, 0)\n"
        f"module.run_kimi_k31_dsa_fullnode(perf_filename={str(output)!r}, cases=[case])\n"
    )
    child_env = dict(os.environ)
    checkout = str(Path(__file__).resolve().parents[3])
    child_env["PYTHONPATH"] = os.pathsep.join(filter(None, [checkout, child_env.get("PYTHONPATH")]))
    with subprocess.Popen([sys.executable, str(driver)], stderr=subprocess.PIPE, text=True, env=child_env) as parent:
        try:
            deadline = time.monotonic() + 30
            while not rank_pid.exists() or not rank_pid.read_text().strip():
                assert time.monotonic() < deadline, "native rank did not become ready"
                assert parent.poll() is None, "collector exited before native rank was ready"
                time.sleep(0.01)
            # Send only to the collector PID, never its process group.
            if not during_launch:
                while not (tmp_path / "owned").exists():
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
            os.kill(parent.pid, signal.SIGTERM)
            if during_launch:
                while not (tmp_path / "handled").exists():
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                assert not (tmp_path / "owned").exists()
            (tmp_path / "release").touch()
            _, stderr = parent.communicate(timeout=30)
            assert parent.returncode == 128 + signal.SIGTERM, stderr
            assert output.read_text() == "previous row\n"
            for pid_file in (launcher_pid, rank_pid):
                try:
                    status = psutil.Process(int(pid_file.read_text())).status()
                except psutil.NoSuchProcess:
                    status = None
                assert status in (None, psutil.STATUS_ZOMBIE)
        finally:
            if parent.poll() is None:
                parent.kill()
            if launcher_pid.exists():
                try:
                    os.killpg(int(launcher_pid.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_context_prefix_preparation_covers_million_tokens_in_bounded_chunks(monkeypatch):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module

    case = ModuleCase(1, 1, 96, "fp8", "bfloat16", "bfloat16", MODEL, 1048575)
    reqs = [SimpleNamespace(prefix_len=0, extend_end=1048576, kv=SimpleNamespace(req_pool_idx=1))]
    runner = SimpleNamespace(device="cpu", forward=lambda batch: None)
    intervals = []

    def extend(actual):
        assert actual is reqs
        req = actual[0]
        intervals.append((req.prefix_len, req.extend_end))
        assert req.extend_end - req.prefix_len <= module.HISTORY_CHUNK_TOKENS
        batch = SimpleNamespace(
            seq_lens_cpu=SimpleNamespace(tolist=lambda: [req.extend_end]),
            prefix_lens=[req.prefix_len],
            extend_lens=[req.extend_end - req.prefix_len],
            input_ids=SimpleNamespace(numel=lambda: req.extend_end - req.prefix_len),
            seq_lens=SimpleNamespace(tolist=lambda: [req.extend_end]),
            extend_prefix_lens_cpu=[req.prefix_len],
            extend_seq_lens_cpu=[req.extend_end - req.prefix_len],
        )
        runner.forward(batch)
        return [0], None, batch

    monkeypatch.setitem(
        sys.modules,
        "sglang.benchmark.one_batch",
        SimpleNamespace(prepare_synthetic_inputs_for_latency_test=lambda batch, length: reqs),
    )
    monkeypatch.setattr(module, "_validate_tp_inputs", lambda *_args: None)
    bench = SimpleNamespace(torch_runner=runner, clear=lambda: None, extend=extend)
    batch, history = module._prepare_forward(bench, case, "context")
    assert intervals[0][0] == 0
    assert all(left[1] == right[0] for left, right in pairwise(intervals))
    assert intervals[-2][1] == 1048575
    assert intervals[-1] == (1048575, 1048576)
    assert sum(end - start for start, end in intervals[:-1]) == 1048575
    assert history["history_per_request"] == 1048575
    assert history["chunks"] == 128
    assert validate_forward_batch(batch, case, "context")["extend_prefix_lens"] == [1048575]


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


@pytest.mark.parametrize(
    ("seq_len", "batch_size", "prefix_len", "peak_hidden_tokens"),
    [
        (1, 1, 1048575, 8192),
        (1, 2, 262144, 8192),
        (1, 1, 512, 512),
        (16384, 1, 16384, 16384),
        (16384, 1, 0, 16384),
        (8192, 2, 512, 16384),
    ],
)
def test_context_capacity_accounts_for_history_chunks_and_final_forward(
    monkeypatch, seq_len, batch_size, prefix_len, peak_hidden_tokens
):
    from collector.sglang.collect_kimi_k31_dsa_v1 import _check_capacity

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL)
    case = ModuleCase(seq_len, batch_size, 96, "fp8", "bfloat16", "bfloat16", MODEL, prefix_len)
    assert case in OP_PLANS["kimi_k31_dsa_context_module"].expand()
    cache_bytes = batch_size * (seq_len + prefix_len + 64) * (656 + 128 + 4)
    required = cache_bytes + peak_hidden_tokens * 7168 * 2
    # All these declared cases fit the accounted buffers within 8 GiB, even
    # when their complete cached prefixes cannot be prepared in one forward.
    assert required < 8 * 2**30
    _check_capacity(case, "context", SimpleNamespace(mem_get_info=lambda _gpu: (8 * 2**30, 8 * 2**30)), 0)
    _check_capacity(case, "context", SimpleNamespace(mem_get_info=lambda _gpu: (required, 8 * 2**30)), 0)
    with pytest.raises(RuntimeError, match=r"^FIXME\(kernel-limit\): does not fit"):
        _check_capacity(case, "context", SimpleNamespace(mem_get_info=lambda _gpu: (required - 1, 8 * 2**30)), 0)


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

    def fail(*_args, **_kwargs):
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
    artifact = tmp_path / "checkpoint/sglang/errors_sglang.kimi_k31_dsa_context_module.json"
    assert json.loads(artifact.read_text()) == errors
    assert errors[0]["task_params"] == str(case)
    assert len(errors) == 1
    assert errors[0]["error_message"] == reason
    assert errors[0]["error_type"] == "RuntimeError"
    assert errors[0]["classification"] == "kernel_limit"


@pytest.mark.parametrize("alias", [False, True])
def test_native_fullnode_resumes_after_last_completed_case(monkeypatch, tmp_path, alias):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module
    from collector import fullnode
    from collector.collect import ResumeCheckpoint

    cases = [ModuleCase(4096, 1, heads, "fp8", "bfloat16", "bfloat16", MODEL, 0) for heads in (96, 48)]
    selected = cases
    output = tmp_path / "dsa_context_module_perf.txt"
    target = tmp_path / "physical-rows" if alias else output
    if alias:
        output.symlink_to(target)
    ran = []

    def native(case, _device, filename, *, expected_sm, phase):
        assert phase == "context"
        assert expected_sm == 103
        with pytest.raises(RuntimeError, match="already has a publication owner"):
            collect()
        ran.append(case)
        if len(ran) == 2:
            with Path(filename).open("a") as stream:
                stream.write("unconfirmed row\n")
            raise KeyboardInterrupt("allocation cancelled during B")
        with Path(filename).open("a") as stream:
            stream.write(str(case) + "\n")

    monkeypatch.setattr(module, "_run_native_case", native)
    options = {"checkpoint_dir": str(tmp_path / "checkpoint"), "resume": True}

    def collect():
        return fullnode.collect_sglang_fullnode_op(
            {
                "name": "sglang",
                "type": "kimi_k31_dsa_context_module",
                "module": module.__name__,
                "get_func": "get_kimi_k31_dsa_context_test_cases",
                "run_func": "run_kimi_k31_dsa_fullnode",
                "perf_filename": str(output),
            },
            runtime_version=RUNTIME_VERSION,
            limit=None,
            shuffle=False,
            shuffle_seed=42,
            backend="sglang",
            resume_options=options,
            model_path=MODEL,
            case_plan=None,
            sm_version=103,
            case_filters=None,
            get_test_cases_for_model=lambda *_args: selected,
            resume_checkpoint_cls=ResumeCheckpoint,
            logger=logging.getLogger(__name__),
        )

    with pytest.raises(KeyboardInterrupt, match="during B"):
        collect()
    checkpoint_path = tmp_path / "checkpoint" / "sglang" / "sglang.kimi_k31_dsa_context_module.json"
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint = json.loads(checkpoint_bytes)
    assert len(checkpoint["done"]) == 1
    assert not checkpoint["failed"]
    assert output.read_text().splitlines() == [str(cases[0])]
    receipt = target.with_name(f".{target.name}.pending")
    transaction = receipt.read_bytes()
    stage = target.with_name(json.loads(transaction)["stage"])
    stage_bytes = stage.read_bytes()
    selected = cases[:1]
    with pytest.raises(RuntimeError, match="select every unfinished attempt"):
        collect()
    assert receipt.read_bytes() == transaction
    assert stage.read_bytes() == stage_bytes
    assert checkpoint_path.read_bytes() == checkpoint_bytes
    assert output.read_text().splitlines() == [str(cases[0])]
    selected = cases

    # Simulate the launcher dying while a child retains the inherited flock.
    with module.native_publication_lock(stage) as fd:
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.read()"],
            pass_fds=(fd,),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert child.stdout.readline().strip() == "ready"
    try:
        lock_path = stage.with_name(f".{stage.name}.publication-lock")
        lock_inode = lock_path.stat().st_ino
        with pytest.raises(RuntimeError, match="already has a publication owner"):
            collect()
        assert lock_path.stat().st_ino == lock_inode
        assert receipt.read_bytes() == transaction
        assert stage.read_bytes() == stage_bytes
        assert checkpoint_path.read_bytes() == checkpoint_bytes
        assert ran == cases
    finally:
        try:
            child.communicate(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    monkeypatch.setenv("COLLECTOR_K31_FINALIZE_ONLY", "1")
    with pytest.raises(RuntimeError, match="passed checkpoint for every selected case"):
        collect()
    monkeypatch.delenv("COLLECTOR_K31_FINALIZE_ONLY")
    assert collect() == []
    monkeypatch.setenv("COLLECTOR_K31_FINALIZE_ONLY", "1")
    assert collect() == []
    assert ran == [cases[0], cases[1], cases[1]]
    assert output.is_symlink() == alias
    assert target.read_text().splitlines() == list(map(str, cases))


@pytest.mark.parametrize("failure", ["runner", "result", "flush"])
def test_native_batch_failure_leaves_unsubmitted_cases_runnable(monkeypatch, tmp_path, failure):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module
    from collector import fullnode
    from collector.collect import ResumeCheckpoint

    cases = [ModuleCase(4096, 1, heads, "fp8", "bfloat16", "bfloat16", MODEL, 0) for heads in (96, 48, 24)]
    ran = []

    def run(*, cases, **_kwargs):
        ran.extend(cases)
        if len(ran) == 2 and failure == "runner":
            raise RuntimeError("runner failed")
        if len(ran) == 2 and failure == "result":
            return None
        return {"failed": []}

    class Checkpoint(ResumeCheckpoint):
        fail_once = failure == "flush"

        def flush(self, force=False):
            if self.fail_once and len(self._done) == 1:
                self.__class__.fail_once = False
                raise OSError("checkpoint flush failed")
            return super().flush(force=force)

    monkeypatch.setattr(module, "run_kimi_k31_dsa_fullnode", run)

    def collect():
        return fullnode.collect_sglang_fullnode_op(
            {
                "name": "sglang",
                "type": "kimi_k31_dsa_context_module",
                "module": module.__name__,
                "get_func": "get_kimi_k31_dsa_context_test_cases",
                "run_func": "run_kimi_k31_dsa_fullnode",
                "perf_filename": str(tmp_path / "dsa_context_module_perf.txt"),
            },
            runtime_version=RUNTIME_VERSION,
            limit=None,
            shuffle=False,
            shuffle_seed=42,
            backend="sglang",
            resume_options={"checkpoint_dir": str(tmp_path / "checkpoint"), "resume": True},
            model_path=MODEL,
            case_plan=None,
            sm_version=103,
            case_filters=None,
            get_test_cases_for_model=lambda *_args: cases,
            resume_checkpoint_cls=Checkpoint,
            logger=logging.getLogger(__name__),
        )

    assert any(error["error_type"] == "FullNodeCollectionFailure" for error in collect())
    assert ran == (cases[:1] if failure == "flush" else cases[:2])
    checkpoint = json.loads(
        (tmp_path / "checkpoint" / "sglang" / "sglang.kimi_k31_dsa_context_module.json").read_text()
    )
    assert len(checkpoint["failed"]) == (0 if failure == "flush" else 1)
    assert len(checkpoint["attempted"]) == len(ran)
    collect()  # ordinary resume skips the submitted failure and runs later cases
    assert ran == cases
    checkpoint = json.loads(
        (tmp_path / "checkpoint" / "sglang" / "sglang.kimi_k31_dsa_context_module.json").read_text()
    )
    assert len(checkpoint["done"]) == (3 if failure == "flush" else 2)
    assert len(checkpoint["failed"]) == (0 if failure == "flush" else 1)


def test_native_success_without_result_row_is_not_checkpointable(monkeypatch, tmp_path):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module

    popen = subprocess.Popen
    monkeypatch.setattr(module.subprocess, "Popen", lambda _cmd, **kw: popen([sys.executable, "-c", "pass"], **kw))
    case = ModuleCase(4096, 1, 96, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    result = run_kimi_k31_dsa_fullnode(perf_filename=tmp_path / "dsa_context_module_perf.txt", cases=[case])
    assert result["succeeded"] == 0
    assert result["failed"] == [case]
    assert "without writing its result row" in result["failure_details"][str(case)]["error_message"]


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_native_input_draw_is_identical_despite_rank_local_rng(monkeypatch, request, phase):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module
    import numpy as np

    initial_state = np.random.get_state()
    request.addfinalizer(lambda: np.random.set_state(initial_state))
    draws, checked = [], []

    def prepare(batch_size, length):
        ids = np.random.randint(0, 10000, size=(batch_size, length)).tolist()
        draws.append(ids)
        return [SimpleNamespace(origin_input_ids=seq) for seq in ids]

    monkeypatch.setitem(
        sys.modules, "sglang.benchmark.one_batch", SimpleNamespace(prepare_synthetic_inputs_for_latency_test=prepare)
    )
    monkeypatch.setattr(
        module, "_validate_tp_inputs", lambda reqs, _device: checked.append([r.origin_input_ids for r in reqs])
    )
    case = ModuleCase(8, 2, 48, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    for seed in (73, 891):
        np.random.seed(seed)
        runner = SimpleNamespace(device="cuda:0", forward=lambda batch: None)

        def extend(reqs):
            assert checked[-1] == [r.origin_input_ids for r in reqs]
            runner.forward("captured batch")
            return "tokens", "batch"

        bench = SimpleNamespace(
            torch_runner=runner, clear=lambda: None, extend=extend, decode=lambda *_a: runner.forward("captured batch")
        )
        monkeypatch.setattr(module, "_extend_history", lambda bench, reqs, *_a: (*bench.extend(reqs), None))
        assert module._prepare_forward(bench, case, phase)[0] == "captured batch"
    assert draws[0] == draws[1]


@pytest.mark.parametrize("mismatch", [False, True])
def test_native_tp_input_check_observes_other_ranks(monkeypatch, mismatch):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module
    import numpy as np

    def gather(copies, tokens):
        for other in copies:
            other[:] = tokens
        if mismatch:
            copies[1][0, 0] += 1

    torch = SimpleNamespace(
        tensor=lambda values, **_kw: np.asarray(values),
        int64=None,
        empty_like=np.empty_like,
        equal=np.array_equal,
        distributed=SimpleNamespace(get_world_size=lambda: 2, all_gather=gather),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    reqs = [SimpleNamespace(origin_input_ids=[1, 2, 3]), SimpleNamespace(origin_input_ids=[4, 5, 6])]
    if mismatch:
        with pytest.raises(RuntimeError, match=r"FIXME\(kernel-limit\).*token IDs differ"):
            module._validate_tp_inputs(reqs, "cuda:0")
    else:
        module._validate_tp_inputs(reqs, "cuda:0")


@pytest.mark.parametrize("power", ["true", "1"])
def test_native_power_request_is_rejected_before_import_or_queue(monkeypatch, tmp_path, power):
    from collector import fullnode

    monkeypatch.setenv("COLLECTOR_MEASURE_POWER", power)
    with pytest.raises(ValueError, match="does not support --measure_power"):
        fullnode.collect_sglang_fullnode_op(
            {"name": "sglang", "type": "kimi_k31_dsa_context_module", "module": "must_not_import"},
            runtime_version=RUNTIME_VERSION,
            limit=None,
            shuffle=False,
            shuffle_seed=42,
            backend="sglang",
            resume_options=None,
            model_path=MODEL,
            case_plan=None,
            sm_version=103,
            case_filters=None,
            get_test_cases_for_model=lambda *_: pytest.fail("queued power case"),
            resume_checkpoint_cls=None,
            logger=logging.getLogger(__name__),
        )
    case = ModuleCase(4096, 1, 96, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    with pytest.raises(ValueError, match="does not support --measure_power"):
        run_kimi_k31_dsa_worker(*case, perf_filename=tmp_path / "generation.txt")


@pytest.mark.skipif(os.name != "posix", reason="native SGLang TP workers use POSIX process groups")
def test_timeout_inside_writer_releases_lock_for_next_success(monkeypatch, tmp_path):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module

    real_popen = subprocess.Popen
    children = []
    output = tmp_path / "dsa_generation_module_perf.txt"
    lock = Path(str(output) + ".lock")
    # Use the real writer, pausing exactly after exclusive-create acquired the
    # lock and before opening its CSV. No GPU/runtime package is needed.
    writer = (
        "import os, time; from collector.helper import log_perf; "
        "real_open = os.open\n"
        "def pause_after_lock(path, flags, *args, **kwargs):\n"
        "    fd = real_open(path, flags, *args, **kwargs)\n"
        f"    if str(path) == {str(lock)!r}: time.sleep(60)\n"
        "    return fd\n"
        "os.open = pause_after_lock\n"
    )
    item = dict(perf_item(ModuleCase(4096, 1, 96, "fp8", "bfloat16", "bfloat16", MODEL, 0), "generation", 1.25))
    call = (
        f"log_perf([{item!r}], 'SGLang', {RUNTIME_VERSION!r}, 'test_gpu', "
        f"'dsa_generation_module', 'test_source', {str(output)!r})"
    )

    def launch(command, **kwargs):
        case = ModuleCase(*json.loads(command[command.index("--case") + 1]))
        env = dict(kwargs["env"])
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(Path(__file__).resolve().parents[3]), env.get("PYTHONPATH")])
        )
        kwargs["env"] = env
        code = writer + call if case.num_heads == 48 else "from collector.helper import log_perf; " + call
        child = real_popen([sys.executable, "-c", code], **kwargs)
        children.append(child)
        if case.num_heads == 48:
            deadline = time.monotonic() + 30
            while not lock.exists():
                assert child.poll() is None, "writer exited before locking"
                assert time.monotonic() < deadline, "writer did not lock"
                time.sleep(0.01)
            monkeypatch.setattr(module, "NATIVE_CASE_TIMEOUT_S", 0.1)
        else:
            monkeypatch.setattr(module, "NATIVE_CASE_TIMEOUT_S", 30)
        return child

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    cases = [ModuleCase(4096, 1, heads, "fp8", "bfloat16", "bfloat16", MODEL, 0) for heads in (48, 96)]
    try:
        result = run_kimi_k31_dsa_fullnode(perf_filename=output, cases=cases)
        assert result["failed"] == [cases[0]]
        assert result["succeeded"] == 1
        assert "timed out" in result["failure_details"][str(cases[0])]["error_message"]
        assert output.read_text().count("1.2500") == 1
        assert not lock.exists()
    finally:
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)


def test_native_output_lock_owned_before_attempt_is_preserved(monkeypatch, tmp_path):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module

    output = tmp_path / "generation.txt"
    lock = Path(str(output) + ".lock")
    lock.write_text("other writer")
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_a, **_k: pytest.fail("started with a foreign lock"))
    case = ModuleCase(4096, 1, 96, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    with pytest.raises(RuntimeError, match="already has a writer lock"):
        run_kimi_k31_dsa_worker(*case, perf_filename=output)
    assert lock.read_text() == "other writer"
