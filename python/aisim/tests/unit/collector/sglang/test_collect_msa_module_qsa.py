# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8-Flash-Next QSA route of the SGLang MSA module collector.

QSA dispatches through the existing ``msa_*_module`` op family (one op → one
collector module), so the plan-time contract it inherits from MSA is verified
here: one case per declared per-rank head shard, BF16 end to end, the served
compressed page size, and a ``kernel_source`` label that is declared in the
curated translation table.

The GPU-side construction checks (``_validate_qsa_module``) need a real
ModelRunner and are exercised by the C3.2 smoke, not here.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

MODEL_PATH = "Qwen/Qwen3.8-Flash-Next"
FP8_PATH = "Qwen/Qwen3.8-Flash-Next-FP8"
NATIVE_Q_HEADS = 24
NATIVE_KV_HEADS = 2
REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "src" / "aisim_core" / "model_configs" / "Qwen--Qwen3.8-Flash-Next_config.json"
BACKEND_MAP = REPO_ROOT / "collector" / "kernel_source_backends.yaml"


def _import_module(monkeypatch, sm: int = 90, cuda_memory: int | None = None):
    torch = types.ModuleType("torch")
    device_properties = types.SimpleNamespace(total_memory=cuda_memory)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_memory is not None,
        empty_cache=lambda: None,
        set_device=lambda *_: None,
        get_device_properties=lambda _device: device_properties,
    )
    torch.Tensor = object
    torch.bfloat16 = "bfloat16"
    monkeypatch.setitem(sys.modules, "torch", torch)
    mod_name = "collector.sglang.collect_msa_module"
    monkeypatch.delitem(sys.modules, mod_name, raising=False)
    spec = importlib.util.spec_from_file_location(mod_name, "collector/sglang/collect_msa_module.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, mod_name, mod)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "get_sm_version", lambda: sm)
    return mod


def _bundled_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def _qsa_cases(mod, getter):
    return [case for case in getter() if case[6] == MODEL_PATH]


def test_qsa_cases_cover_exactly_the_declared_per_rank_head_shards(monkeypatch):
    mod = _import_module(monkeypatch)
    for getter in (mod.get_msa_context_module_test_cases, mod.get_msa_generation_module_test_cases):
        cases = _qsa_cases(mod, getter)
        assert cases
        # 24 native query heads at TP 1/2/4/8 (get_mla_module_model_head_counts
        # answers 24/12/6/3); the shared MSA/DSA head grid has no 6/3, so these
        # counts can only come from the model's own declaration.
        assert {case[2] for case in cases} == {3, 6, 12, 24}
        assert {case[2] * case[7] for case in cases} == {NATIVE_Q_HEADS}
        assert {case[7] for case in cases} == {1, 2, 4, 8}
        # Both Qwen3.8-Flash-Next artifacts keep the QSA layer and its KV in
        # BF16, so there is exactly one precision triple.
        assert {(case[3], case[4], case[5]) for case in cases} == {("bfloat16", "bfloat16", "bfloat16")}
        # The plan manifest is bound at plan time and is never empty.
        assert all(case[8] for case in cases)


def test_qsa_context_plan_exercises_the_prefix_route(monkeypatch):
    """The chunk-prefill kernel_source label is only reachable with prefix > 0."""
    mod = _import_module(monkeypatch)
    prefix_lens = {
        prefix for case in _qsa_cases(mod, mod.get_msa_context_module_test_cases) for (_b, _s, prefix) in case[8]
    }
    assert 0 in prefix_lens
    assert any(prefix > 0 for prefix in prefix_lens)


def test_qsa_context_prefixes_align_to_the_compressed_block_width(monkeypatch):
    """The QSA write plan asserts prefix % compress_ratio == 0 on the device
    (qwen_sparse_attn_backend.py:527@v0.5.21) and a violation aborts the whole
    sweep subprocess, so every planned prefix must be group-aligned.

    The op plan raises the declared anchors to a group boundary and dedupes in
    the aligned space; a raised point that would cross the RoPE table is
    dropped rather than moved.
    """
    mod = _import_module(monkeypatch)
    ratio = mod._qsa_compress_ratio(MODEL_PATH)
    assert ratio == _bundled_config()["text_config"]["indexer_compress_ratio"] == 4

    from collector.op_plans import _compressed_prefix_points

    # Sub-group prefixes round up to the smallest addressable one, and two
    # anchors can collide there (1 -> 4 is the declared 4).
    assert _compressed_prefix_points([(1, 1, 1), (1, 1, 4), (2, 512, 1023)], ratio, None) == [
        (1, 1, 4),
        (2, 512, 1024),
    ]
    # Rounding up must keep the RoPE table bound, so a point that would cross
    # it is dropped instead of raised.
    assert _compressed_prefix_points([(1, 1, 1021), (1, 1, 1016)], ratio, 1023) == [(1, 1, 1016)]

    max_pos = mod._model_max_position_embeddings(MODEL_PATH)
    cases = _qsa_cases(mod, mod.get_msa_context_module_test_cases)
    assert cases
    prefixes = set()
    for case in cases:
        assert case[8], f"empty plan for batch {case[1]}"
        assert all(prefix % ratio == 0 for (_b, _s, prefix) in case[8]), case[1]
        assert len({tuple(point) for point in case[8]}) == len(case[8]), f"duplicate plan points for batch {case[1]}"
        assert all(prefix + s <= max_pos for (_b, s, prefix) in case[8]), case[1]
        prefixes |= {prefix for (_b, _s, prefix) in case[8]}
    assert 0 in prefixes
    assert any(prefix > 0 for prefix in prefixes)


def test_qsa_prefix_alignment_leaves_the_msa_grid_alone(monkeypatch):
    """MiniMax-M3 has no compressed pool: its plan points, including the 1M
    anchor, are queued exactly as the shared grid declares them."""
    mod = _import_module(monkeypatch)
    rows = [row for row in mod.get_msa_context_module_test_cases() if row[6] == "MiniMaxAI/MiniMax-M3"]
    assert rows
    points = {tuple(point) for row in rows for point in row[8]}
    # The 1M ceiling anchor is not a multiple of the QSA group width, so it
    # proves alignment was not applied to the MSA route.
    assert (1, 1, 1048575) in points
    assert any(prefix % 4 for (_b, _s, prefix) in points)


def test_qsa_compress_ratio_requires_a_declared_group_width(monkeypatch):
    mod = _import_module(monkeypatch)
    with pytest.raises(RuntimeError, match="indexer_compress_ratio"):
        mod._qsa_compress_ratio("MiniMaxAI/MiniMax-M3")


def test_worker_refuses_a_misaligned_ad_hoc_qsa_prefix(monkeypatch):
    """The plan aligns its prefixes; a hand-run shape can still be off a group,
    and the device-side assert it would hit poisons the CUDA context for every
    shape that follows it in the sweep."""
    mod = _import_module(monkeypatch)
    mod._require_aligned_qsa_prefixes([(1, 1, 4)], model_path=MODEL_PATH)
    with pytest.raises(RuntimeError, match="compressed-block width"):
        mod._require_aligned_qsa_prefixes([(1, 1, 1)], model_path=MODEL_PATH)


def test_qsa_plan_page_size_is_the_serving_pinned_64(monkeypatch):
    mod = _import_module(monkeypatch)
    assert mod._QSA_PLAN_PAGE_SIZE == 64
    assert mod._kv_cost_estimate("qsa", NATIVE_Q_HEADS, "bfloat16", MODEL_PATH)[0] == 64
    # MSA keeps its own page size; the two pools are different objects.
    assert mod._kv_cost_estimate("msa", 64, "bfloat16", "MiniMaxAI/MiniMax-M3")[0] == mod._PLAN_PAGE_SIZE


def test_qsa_bytes_per_token_shards_kv_like_serving(monkeypatch):
    """2 KV heads over TP 1/2 keeps 2/1; TP 4/8 replicate the single head."""
    mod = _import_module(monkeypatch)
    text_config = _bundled_config()["text_config"]
    assert [mod._qsa_local_kv_heads(text_config, tp) for tp in (1, 2, 4, 8)] == [2, 1, 1, 1]
    head_dim = int(text_config["head_dim"])
    bf16_per_token = mod._qsa_bytes_per_token(NATIVE_Q_HEADS, "bfloat16", text_config)
    # The paged main K/V alone is the lower bound; the three bf16 side buffers
    # and the int64 MRoPE position ring are on top.
    main_only = 2 * NATIVE_KV_HEADS * head_dim * 2
    assert bf16_per_token > main_only
    # fp8 only halves the main K/V; the side buffers stay bf16 by contract.
    fp8_per_token = mod._qsa_bytes_per_token(NATIVE_Q_HEADS, "fp8", text_config)
    assert fp8_per_token == bf16_per_token - main_only // 2
    # A per-rank shard costs less than the full head count.
    assert mod._qsa_bytes_per_token(NATIVE_Q_HEADS // 8, "bfloat16", text_config) < bf16_per_token


def test_qsa_prefix_workspace_is_the_backend_gather(monkeypatch):
    """A prefixed chunk gathers every row's whole context out of the pool: one
    ``index_select`` per request plus the ``torch.cat`` of the parts, for K and
    again for V (layers/attention/qwen_sparse_attn_backend.py:1326-1361@v0.5.21).
    That is two full copies of the layer's main K/V per token, allocated per
    measured shape ON TOP of the resident pool — not part of the pool."""
    mod = _import_module(monkeypatch)
    text_config = _bundled_config()["text_config"]
    head_dim = int(text_config["head_dim"])
    # TP1 keeps both native KV heads, TP8 replicates the single one.
    for num_heads, kv_heads in ((NATIVE_Q_HEADS, NATIVE_KV_HEADS), (NATIVE_Q_HEADS // 8, 1)):
        workspace = mod._qsa_prefix_transient_bytes_per_token(num_heads, "bfloat16", text_config)
        assert workspace == 2 * (2 * kv_heads * head_dim * 2)
        # fp8 halves the gathered K/V exactly like the pool does.
        assert mod._qsa_prefix_transient_bytes_per_token(num_heads, "fp8", text_config) == workspace // 2
        assert mod._kv_cost_estimate("qsa", num_heads, "bfloat16", MODEL_PATH)[2] == workspace
    # MSA has no prefix gather of its own: its pool estimate is the footprint.
    assert mod._kv_cost_estimate("msa", 64, "bfloat16", "MiniMaxAI/MiniMax-M3")[2] == 0


def test_qsa_footprint_refusal_covers_the_pool_and_the_workspace(monkeypatch):
    """The C3.3 gap: 215 of 11888 SGLang context shapes aborted with CUDA OOM
    mid-sweep because the plan bounded the pool but not the pool-resident case's
    per-shape gather workspace. The worker refuses such a point before it
    allocates anything, with the classified kernel-limit reason.

    Measured one shape per fresh process on H200: TP1 b=128 prefix 131072 peaks
    at 104.67 GiB and measures; prefix 262140 needs ~197 GiB and OOMs.
    """
    device = 150_109_880_320
    mod = _import_module(monkeypatch, cuda_memory=device)
    page_size, bytes_per_token, workspace_bytes = mod._kv_cost_estimate("qsa", NATIVE_Q_HEADS, "bfloat16", MODEL_PATH)

    def reason(shape):
        return mod._qsa_footprint_refusal(
            shape,
            page_size=page_size,
            bytes_per_token=bytes_per_token,
            workspace_bytes=workspace_bytes,
            total_memory=device,
        )

    assert reason((1, 1, 0)) is None
    assert reason((128, 1, 131072)) is None
    refused = reason((128, 1, 262140))
    assert refused is not None and refused.startswith(mod.KERNEL_LIMIT)
    assert "does not fit" in refused and "4096 B/token prefix gather" in refused
    # Without the workspace term the 262140 ceiling looks affordable, which is
    # exactly the estimate that let the C3.3 sweep OOM mid-case.
    tokens = mod.required_kv_alloc_tokens(128, 1, 262140, page_size, is_prefill=True)
    assert tokens * bytes_per_token <= int(device * mod._MEMORY_BUDGET_SAFETY_FACTOR)


def test_qsa_getter_queues_the_plan_verbatim(monkeypatch):
    """Plans do not depend on the device: capacity is refused by the worker
    (layer_permissions.md), so the getter stays the one-line plan call."""
    from collector.op_plans import plan_cases

    mod = _import_module(monkeypatch)
    for op, getter in (
        ("msa_context_module", mod.get_msa_context_module_test_cases),
        ("msa_generation_module", mod.get_msa_generation_module_test_cases),
    ):
        assert getter() == plan_cases(op, "sglang", mod.get_sm_version())
    # The serving-relevant short/medium context range is planned at batch 1.
    tp1 = {case[1]: case for case in _qsa_cases(mod, mod.get_msa_context_module_test_cases) if case[7] == 1}
    assert [1, 1, 0] in tp1[1][8]
    assert [1, 1, 16384] in tp1[1][8]


def test_qsa_override_args_reproduce_one_target_tp_rank(monkeypatch):
    mod = _import_module(monkeypatch)
    config = _bundled_config()
    for target_tp in (1, 2, 4, 8):
        num_heads = NATIVE_Q_HEADS // target_tp
        override = mod._build_qsa_model_override_args(config, num_heads, target_tp, mod.QSA_BENCH_NUM_LAYERS)
        text = override["text_config"]
        assert set(override) == {"text_config"}, "the outer multimodal config must stay untouched (MRoPE)"
        assert text["num_attention_heads"] == num_heads
        assert text["num_key_value_heads"] == max(1, NATIVE_KV_HEADS // target_tp)
        assert text["num_hidden_layers"] == mod.QSA_BENCH_NUM_LAYERS
        assert text["layer_types"] == mod.QSA_BENCH_LAYER_TYPES
        # Exactly one full-attention layer, and it is the benchmarked one.
        assert [i for i, kind in enumerate(text["layer_types"]) if kind == "full_attention"] == [mod.QSA_BENCH_LAYER]
        assert text["ple_layer_ids"] == []
        assert text["num_experts"] == mod.QSA_BENCH_NUM_EXPERTS


def test_qsa_override_args_reject_a_mismatched_or_text_only_config(monkeypatch):
    mod = _import_module(monkeypatch)
    config = _bundled_config()
    with pytest.raises(ValueError, match="num_heads"):
        mod._build_qsa_model_override_args(config, 12, 1, mod.QSA_BENCH_NUM_LAYERS)
    with pytest.raises(ValueError, match="text_config"):
        mod._build_qsa_model_override_args(
            {"num_attention_heads": NATIVE_Q_HEADS}, NATIVE_Q_HEADS, 1, mod.QSA_BENCH_NUM_LAYERS
        )


def test_qsa_model_resolution_accepts_the_fp8_alias_only(monkeypatch):
    mod = _import_module(monkeypatch)
    canonical = mod._resolve_model_spec(MODEL_PATH)
    assert canonical.attention_type == "qsa"
    assert canonical.architecture == "Qwen4ExpForConditionalGeneration"
    assert canonical.native_num_heads == NATIVE_Q_HEADS
    # A targeted run may name the -FP8 artifact; it shares the one physical
    # case (model_aliases), so it resolves back to the canonical row.
    assert mod._resolve_model_spec(FP8_PATH).model_path == MODEL_PATH
    assert mod._attention_type_for_model(MODEL_PATH) == "qsa"
    assert mod._attention_type_for_model("MiniMaxAI/MiniMax-M3") == "msa"
    with pytest.raises(ValueError):
        mod._resolve_model_spec("Qwen/Qwen3.8-Flash-Next-NVFP4")


class _QsaBackendModule(types.ModuleType):
    """Stand-in for sglang's qwen_sparse_attn_backend route resolvers."""

    def __init__(self, *, trtllm=None, varlen=None):
        super().__init__("sglang.srt.layers.attention.qwen_sparse_attn_backend")
        self.sparse_gqa_fwd_interface_triton = object()
        self.sparse_gqa_fwd_interface_triton_ck = object()
        self.sparse_gqa_packed_decode_triton = object()
        self._resolve_trtllm_sparse_decode = lambda: trtllm
        self._resolve_flash_attn_varlen_func = lambda: varlen


class _Varlen:
    def __init__(self, name, module):
        self.__name__ = name
        self.__module__ = module


def _declared_labels(framework: str) -> set[str]:
    table = yaml.safe_load(BACKEND_MAP.read_text())
    return {entry["kernel_source"] for entry in table["mappings"] if entry["framework"] == framework}


@pytest.mark.parametrize(
    ("prefix_len", "expected"),
    [
        (0, "sglang_qwen4exp_qsa_prefill_triton"),
        (1, "sglang_qwen4exp_qsa_prefill_triton_ck"),
        (4096, "sglang_qwen4exp_qsa_prefill_triton_ck"),
    ],
)
def test_qsa_prefill_labels(monkeypatch, prefix_len, expected):
    mod = _import_module(monkeypatch)
    monkeypatch.setattr(mod, "_qsa_backend_module", lambda: _QsaBackendModule())
    monkeypatch.setattr(mod, "_qsa_full_attn_backend", lambda _runner: object())
    assert mod._qsa_prefill_kernel_source(object(), prefix_len) == expected
    assert expected in _declared_labels("sglang")


def _fa4_cute_wrapper(module: str = "flash_attn.cute.interface"):
    """The wrapper SGLang builds when FA2 is absent (qwen_sparse_attn_backend.py:83-91)."""

    def flash_attn_varlen_func(*args, **kwargs):
        output = cute_varlen_func(*args, **kwargs)
        return output[0] if isinstance(output, tuple) else output

    def cute_varlen_func(*args, **kwargs):
        return args

    cute_varlen_func.__module__ = module
    return flash_attn_varlen_func


def test_qsa_decode_labels(monkeypatch):
    mod = _import_module(monkeypatch)
    sglang_utils = types.ModuleType("sglang.srt.utils")
    monkeypatch.setitem(sys.modules, "sglang", types.ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", types.ModuleType("sglang.srt"))
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", sglang_utils)
    monkeypatch.setattr(mod, "_qsa_full_attn_backend", lambda _runner: object())

    fa2 = _Varlen("flash_attn_varlen_func", "flash_attn.flash_attn_interface")
    sm121 = _Varlen("qwen38_qsa_sm121_varlen", "sglang.srt.layers.attention.qwen_sparse_attn_backend")

    monkeypatch.setattr(mod, "_qsa_backend_module", lambda: _QsaBackendModule(trtllm=object(), varlen=fa2))
    assert mod._qsa_decode_kernel_source(object()) == "sglang_qwen4exp_qsa_decode_trtllm_sparse"

    sglang_utils.is_hip = lambda: True
    monkeypatch.setattr(mod, "_qsa_backend_module", lambda: _QsaBackendModule(varlen=fa2))
    assert mod._qsa_decode_kernel_source(object()) == "sglang_qwen4exp_qsa_decode_triton_packed"

    sglang_utils.is_hip = lambda: False
    monkeypatch.setattr(mod, "_qsa_backend_module", lambda: _QsaBackendModule(varlen=sm121))
    assert mod._qsa_decode_kernel_source(object()) == "sglang_qwen4exp_qsa_decode_sm121_varlen"

    monkeypatch.setattr(mod, "_qsa_backend_module", lambda: _QsaBackendModule(varlen=fa2))
    assert mod._qsa_decode_kernel_source(object()) == "sglang_qwen4exp_qsa_decode_fa2_varlen"

    monkeypatch.setattr(mod, "_qsa_backend_module", lambda: _QsaBackendModule(varlen=_fa4_cute_wrapper()))
    assert mod._qsa_decode_kernel_source(object()) == "sglang_qwen4exp_qsa_decode_fa4_cute_varlen"

    declared = _declared_labels("sglang")
    assert {
        "sglang_qwen4exp_qsa_decode_trtllm_sparse",
        "sglang_qwen4exp_qsa_decode_triton_packed",
        "sglang_qwen4exp_qsa_decode_sm121_varlen",
        "sglang_qwen4exp_qsa_decode_fa2_varlen",
        "sglang_qwen4exp_qsa_decode_fa4_cute_varlen",
    } <= declared


def test_qsa_decode_refuses_an_unnamed_kernel(monkeypatch):
    """A fallback the collector cannot name must raise, not be mislabelled."""
    mod = _import_module(monkeypatch)
    sglang_utils = types.ModuleType("sglang.srt.utils")
    sglang_utils.is_hip = lambda: False
    monkeypatch.setitem(sys.modules, "sglang", types.ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", types.ModuleType("sglang.srt"))
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", sglang_utils)
    monkeypatch.setattr(mod, "_qsa_full_attn_backend", lambda _runner: object())
    monkeypatch.setattr(
        mod,
        "_qsa_backend_module",
        lambda: _QsaBackendModule(varlen=_Varlen("paged_varlen_kernel", "some_other_library.ops")),
    )
    with pytest.raises(RuntimeError, match="no registered kernel_source label"):
        mod._qsa_decode_kernel_source(object())


def test_qsa_decode_refuses_a_varlen_wrapper_over_a_third_party_backend(monkeypatch):
    """Only the FA4-cute implementation may be named from a wrapper's closure."""
    mod = _import_module(monkeypatch)
    sglang_utils = types.ModuleType("sglang.srt.utils")
    sglang_utils.is_hip = lambda: False
    monkeypatch.setitem(sys.modules, "sglang", types.ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", types.ModuleType("sglang.srt"))
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", sglang_utils)
    monkeypatch.setattr(mod, "_qsa_full_attn_backend", lambda _runner: object())
    monkeypatch.setattr(
        mod,
        "_qsa_backend_module",
        lambda: _QsaBackendModule(varlen=_fa4_cute_wrapper("some_other_library.ops")),
    )
    with pytest.raises(RuntimeError, match="no registered kernel_source label"):
        mod._qsa_decode_kernel_source(object())


def test_qsa_kernel_source_symbols_are_required(monkeypatch):
    """An upstream rename must fail loud instead of guessing a label."""
    mod = _import_module(monkeypatch)
    monkeypatch.setattr(mod, "_qsa_backend_module", lambda: _QsaBackendModule())
    monkeypatch.setattr(mod, "_qsa_full_attn_backend", lambda _runner: object())
    with pytest.raises(RuntimeError, match="sparse_gqa_fwd_interface_triton_ck"):
        # prefix > 0 needs the chunk-prefill symbol, which the stub lacks.
        stub = _QsaBackendModule()
        del stub.sparse_gqa_fwd_interface_triton_ck
        monkeypatch.setattr(mod, "_qsa_backend_module", lambda: stub)
        mod._qsa_prefill_kernel_source(object(), 128)


def test_an_unpersistable_row_aborts_the_sweep_with_the_real_cause(monkeypatch):
    """``log_perf`` raising PerfLogError is fatal for the whole sweep.

    A resumed run that finds the perf txt and its parked ``.retained`` twin both
    present (helper.py:793-795@HEAD) cannot append ANY row, so retrying the next
    shape only reproduces the same failure. Before this conversion the generic
    per-point handler counted one failure per remaining shape and the subprocess
    then died with "persisted no rows" — 230 identical failures and 15 minutes of
    H200 time per case, with the actual cause hidden (C3.3 heal, 2026-11-01).
    The row writer must therefore raise the fatal PerfLogWriteError, which
    ``run_msa_module`` re-raises immediately.
    """
    mod = _import_module(monkeypatch)
    monkeypatch.setattr(mod, "get_version", lambda _name: "0.5.21")

    def _conflicting(**_kwargs):
        raise mod.PerfLogError("Conflicting retained performance file for /tmp/x_perf.txt")

    monkeypatch.setattr(mod, "log_perf", _conflicting)
    with pytest.raises(mod.PerfLogWriteError) as excinfo:
        mod._log_msa_row(
            perf_filename="/tmp/x_perf.txt",
            model_path=MODEL_PATH,
            architecture=mod.QSA_ARCHITECTURE,
            compute_dtype="bfloat16",
            kv_cache_dtype="bfloat16",
            gemm_type="bfloat16",
            num_heads=NATIVE_Q_HEADS,
            batch_size=128,
            isl=1,
            step=131072,
            target_tp_size=1,
            latency_ms=1.0,
            op_name="sglang_qwen4exp_qsa_prefill_triton_ck",
            kernel_source="sglang_qwen4exp_qsa_prefill_triton_ck",
            device_name="NVIDIA H200",
            power_stats=None,
        )
    assert "Conflicting retained performance file" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, mod.PerfLogError)
