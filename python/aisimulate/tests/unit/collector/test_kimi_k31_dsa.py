# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native K31 collection keeps its own architecture and real TP geometry."""

import copy
import json
import subprocess
import sys
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
        assert set(entry.unverified_sms) == {80, 86, 89, 120}
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
    assert executed_kernel_source(hybrid, pool, "context") == "sglang_dsa_dense_mha_trtllm_ragged_kv_raw_fp8_eager"
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

    monkeypatch.setattr(module.subprocess, "run", lambda *_a, **_k: pytest.fail("launched an unowned GPU group"))
    with pytest.raises(RuntimeError, match="exclusive ownership"):
        run_kimi_k31_dsa_worker(*case, device="cuda:0", perf_filename="dsa_generation_module_perf.txt")


def test_fullnode_runs_each_requested_tp_group_and_retains_exact_failed_case(monkeypatch, tmp_path):
    import collector.sglang.collect_kimi_k31_dsa_v1 as module

    cases = [ModuleCase(4096, 1, heads, "fp8", "bfloat16", "bfloat16", MODEL, 0) for heads in (96, 48, 24)]
    observed = []

    def run(command, *, check):
        assert check
        case = ModuleCase(*json.loads(command[command.index("--case") + 1]))
        assert command[command.index("--device") + 1] == "cuda:0"
        assert command[command.index("--phase") + 1] == "generation"
        observed.append(case)
        if case.num_heads == 48:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(module.subprocess, "run", run)
    result = run_kimi_k31_dsa_fullnode(perf_filename=tmp_path / "dsa_generation_module_perf.txt", cases=cases)
    assert observed == cases
    assert result == {"succeeded": 2, "failed": [cases[1]]}


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
    ("tp", "nodes", "rank", "expected"), [(1, 1, 0, 1), (4, 1, 0, 4), (8, 2, 0, 4), (8, 2, 1, 4), (8, 4, 3, 2)]
)
def test_native_tp_layout_preserves_global_group_and_local_devices(tp, nodes, rank, expected):
    from collector.sglang.collect_kimi_k31_dsa_v1 import local_tp_size

    assert local_tp_size(tp, nodes, rank) == expected
    ranks = [node * expected + local for node in range(nodes) for local in range(expected)]
    assert ranks == list(range(tp))


@pytest.mark.parametrize(("nodes", "rank"), [(0, 0), (3, 0), (2, -1), (2, 2)])
def test_native_tp_layout_rejects_incomplete_groups(nodes, rank):
    from collector.sglang.collect_kimi_k31_dsa_v1 import local_tp_size

    with pytest.raises(ValueError, match="divide evenly"):
        local_tp_size(8, nodes, rank)


def _native_main_argv(tmp_path):
    case = ModuleCase(4096, 1, 12, "fp8", "bfloat16", "bfloat16", MODEL, 0)
    return [
        "collect_kimi_k31_dsa_v1",
        "--case",
        json.dumps(case),
        "--phase",
        "context",
        "--output",
        str(tmp_path / "perf.txt"),
    ]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--nnodes", "2"], "requires the same --dist-init-addr"),
        (["--nnodes", "2", "--node-rank", "-1", "--dist-init-addr", "master:1234"], "valid node ranks"),
        (["--nnodes", "2", "--node-rank", "2", "--dist-init-addr", "master:1234"], "valid node ranks"),
        (["--node-rank", "1"], "valid node ranks"),
    ],
)
def test_native_main_rejects_invalid_rendezvous_before_runtime_lookup(monkeypatch, tmp_path, options, message):
    from collector.sglang import collect_kimi_k31_dsa_v1 as native

    monkeypatch.setattr(sys, "argv", _native_main_argv(tmp_path) + options)
    monkeypatch.setattr(native, "_assert_runtime", lambda: pytest.fail("invalid launch reached runtime lookup"))
    with pytest.raises(ValueError, match=message):
        native.main()


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize(("nodes", "rank"), [(1, 0), (2, 0), (2, 1)])
def test_native_main_wires_platform_launch_and_offline_tokenizer(monkeypatch, tmp_path, config, offline, nodes, rank):
    from collector.sglang import collect_kimi_k31_dsa_v1 as native

    config_path = tmp_path / "checkpoint-config.json"
    config_path.write_text(json.dumps(config))
    tokenizer = str(tmp_path / "tokenizer")
    calls = []

    def hf_hub_download(model, filename, *, revision):
        assert (model, filename, revision) == (MODEL, "config.json", native.MODEL_REVISION)
        return str(config_path)

    def snapshot_download(model, *, revision, local_files_only, allow_patterns):
        assert (model, revision) == (MODEL, native.MODEL_REVISION)
        assert local_files_only is offline
        assert set(allow_patterns) == {
            "tokenizer_config.json",
            "tokenization_kimi.py",
            "encoding_k3.py",
            "tiktoken.model",
        }
        calls.append("tokenizer")
        return tokenizer

    class ServerArgs(SimpleNamespace):
        def resolve_once(self):
            # main() must write the single-layer config before native setup.
            with open(self.model_path + "/config.json") as file:
                assert json.load(file) == single_layer_config(config)
            assert self.tokenizer_path == tokenizer
            assert (self.tp_size, self.nnodes, self.node_rank) == (8, nodes, rank)
            assert self.dist_init_addr == ("master:1234" if nodes > 1 else None)
            calls.append("resolve")

    ports = object()

    def set_envs(server_args):
        assert isinstance(server_args, ServerArgs)
        calls.append("environment")

    def init_ports(server_args):
        assert isinstance(server_args, ServerArgs)
        calls.append("ports")
        return ports

    def spawn(function, *, args, nprocs, join):
        server_args, actual_ports, case_values, phase, output, first_gpu = args
        assert function is native._rank_main
        assert isinstance(server_args, ServerArgs)
        assert actual_ports is ports
        assert native.validate_case(ModuleCase(*case_values)) == 8
        assert (phase, output, first_gpu, nprocs, join) == ("context", str(tmp_path / "perf.txt"), 1, 8 // nodes, True)
        calls.append("spawn")

    # Exercise parsing and orchestration without GPUs, model downloads, or a
    # serving-framework install. The real single_layer_config still executes.
    modules = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(device_count=lambda: 1 + 8 // nodes),
            multiprocessing=SimpleNamespace(spawn=spawn),
        ),
        "huggingface_hub": SimpleNamespace(
            constants=SimpleNamespace(HF_HUB_OFFLINE=offline),
            hf_hub_download=hf_hub_download,
            snapshot_download=snapshot_download,
        ),
        "sglang.srt.entrypoints.engine": SimpleNamespace(_set_envs_and_config=set_envs),
        "sglang.srt.server_args": SimpleNamespace(ServerArgs=ServerArgs, PortArgs=SimpleNamespace(init_new=init_ports)),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(native, "_assert_runtime", lambda: None)
    options = ["--device", "cuda:1"]
    if nodes > 1:
        options += ["--nnodes", str(nodes), "--node-rank", str(rank), "--dist-init-addr", "master:1234"]
    monkeypatch.setattr(sys, "argv", _native_main_argv(tmp_path) + options)
    native.main()
    assert calls == ["tokenizer", "resolve", "environment", "ports", "spawn"]
