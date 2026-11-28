# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public MSA getter and worker contracts for the TRT-LLM and vLLM collectors."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = "MiniMaxAI/MiniMax-M3"
ARCHITECTURE = "MiniMaxM3ForCausalLM"
QWEN4EXP_PATH = "Qwen/Qwen3.8-Flash-Next"
_QWEN4EXP_TEXT_CONFIG = json.loads(
    (REPO_ROOT / "src" / "aisim_core" / "model_configs" / "Qwen--Qwen3.8-Flash-Next_config.json").read_text()
)["text_config"]


class _Dummy:
    def __init__(self, *_args, **kwargs):
        self.__dict__.update(kwargs)


class _StubPerfLogError(RuntimeError):
    """Stand-in for ``collector.helper.PerfLogError`` in the import stub."""


def _noop(*_args, **_kwargs):
    return None


def _stub_module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    module.__path__ = []
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _stub_torch(monkeypatch):
    return _stub_module(
        monkeypatch,
        "torch",
        Tensor=object,
        bfloat16="bfloat16",
        float8_e4m3fn="float8_e4m3fn",
        float32="float32",
        int32="int32",
        long="long",
        uint8="uint8",
        cuda=SimpleNamespace(
            OutOfMemoryError=RuntimeError,
            Stream=_noop,
            empty_cache=_noop,
            is_available=lambda: False,
        ),
        device=lambda value: value,
        inference_mode=nullcontext,
    )


class _PeriodicGCStub:
    def maybe_collect(self):
        return False


def _install_helper_stub(monkeypatch):
    _stub_module(
        monkeypatch,
        "collector.helper",
        PeriodicGarbageCollector=_PeriodicGCStub,
        PerfLogError=_StubPerfLogError,
        _resolve_local_model_path=lambda path: path,
        benchmark_with_power=_noop,
        get_sm_version=lambda: 100,
        log_perf=lambda **_kwargs: True,
    )


def _install_trtllm_stubs(monkeypatch):
    _stub_torch(monkeypatch)
    transformers = _stub_module(monkeypatch, "transformers", PretrainedConfig=_Dummy)
    trtllm = _stub_module(monkeypatch, "tensorrt_llm", __version__="1.4.0rc2")
    for package in (
        "tensorrt_llm._torch",
        "tensorrt_llm._torch.attention",
        "tensorrt_llm._torch.attention.backends",
        "tensorrt_llm._torch.attention.backends.sparse",
        "tensorrt_llm._torch.attention.backends.sparse.minimax_m3",
        "tensorrt_llm._torch.models",
        "tensorrt_llm._torch.modules",
        "tensorrt_llm._torch.pyexecutor",
        "tensorrt_llm.bindings",
        "tensorrt_llm.bindings.internal",
        "tensorrt_llm.llmapi",
        "tensorrt_llm.models",
        "tensorrt_llm.quantization",
    ):
        _stub_module(monkeypatch, package)
    _stub_module(monkeypatch, "tensorrt_llm._torch.attention.backends.interface", AttentionRuntimeFeatures=_Dummy)
    _stub_module(monkeypatch, "tensorrt_llm._torch.attention.backends.utils", get_attention_backend=_noop)
    _stub_module(
        monkeypatch,
        "tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend",
        MiniMaxM3MsaSparseAttentionMetadata=_Dummy,
    )
    _stub_module(monkeypatch, "tensorrt_llm._torch.autotuner", autotune=nullcontext)
    _stub_module(monkeypatch, "tensorrt_llm._torch.modules.linear", Linear=_Dummy)
    _stub_module(monkeypatch, "tensorrt_llm._torch.metadata", KVCacheParams=_Dummy)
    _stub_module(monkeypatch, "tensorrt_llm._torch.model_config", ModelConfig=_Dummy)
    _stub_module(
        monkeypatch,
        "tensorrt_llm._torch.models.modeling_minimaxm3",
        MiniMaxM3DecoderLayer=_Dummy,
    )
    _stub_module(
        monkeypatch,
        "tensorrt_llm._torch.pyexecutor._util",
        get_kv_cache_manager_cls=_noop,
    )
    _stub_module(monkeypatch, "tensorrt_llm._torch.pyexecutor.config_utils", _CONFIG_REGISTRY={})
    _stub_module(
        monkeypatch,
        "tensorrt_llm._torch.pyexecutor.model_loader",
        initialize_dummy_weights=_noop,
    )
    _stub_module(
        monkeypatch,
        "tensorrt_llm._torch.utils",
        AuxStreamType=SimpleNamespace(MoeChunkingOverlap="moe"),
        get_model_extra_attrs=lambda: {},
        model_extra_attrs=lambda *_args: SimpleNamespace(__enter__=_noop, __exit__=_noop),
    )
    _stub_module(monkeypatch, "tensorrt_llm._utils", torch_dtype_to_binding=_noop)
    _stub_module(
        monkeypatch,
        "tensorrt_llm.llmapi.llm_args",
        KvCacheConfig=_Dummy,
        MiniMaxM3SparseAttentionConfig=_Dummy,
    )
    _stub_module(
        monkeypatch,
        "tensorrt_llm.bindings.internal.batch_manager",
        CacheType=SimpleNamespace(SELF="self"),
    )
    _stub_module(monkeypatch, "tensorrt_llm.functional", AllReduceStrategy=SimpleNamespace(AUTO="auto"))
    _stub_module(
        monkeypatch,
        "tensorrt_llm.models.modeling_utils",
        QuantConfig=_Dummy,
    )
    _stub_module(monkeypatch, "tensorrt_llm.quantization.mode", QuantAlgo=SimpleNamespace())
    configs = _stub_module(monkeypatch, "tensorrt_llm._torch.configs")
    trtllm._torch = SimpleNamespace(configs=configs)
    configs.PretrainedConfig = transformers.PretrainedConfig
    _install_helper_stub(monkeypatch)


def _install_vllm_stubs(monkeypatch):
    _stub_torch(monkeypatch)
    _stub_module(monkeypatch, "vllm")
    for package in (
        "vllm.transformers_utils",
        "vllm.v1",
        "vllm.v1.worker",
        "collector.vllm",
    ):
        _stub_module(monkeypatch, package)
    _stub_module(monkeypatch, "vllm.config", set_current_vllm_config=_noop)
    _stub_module(monkeypatch, "vllm.forward_context", set_forward_context=_noop)
    _stub_module(monkeypatch, "vllm.transformers_utils.config", _CONFIG_REGISTRY={})
    _stub_module(monkeypatch, "vllm.v1.worker.workspace", init_workspace_manager=_noop)
    _stub_module(monkeypatch, "vllm.version", __version__="0.24.0")
    _stub_module(
        monkeypatch,
        "collector.vllm.utils",
        BatchSpec=_Dummy,
        common_attn_num_computed_tokens_cpu=_noop,
        create_common_attn_metadata=_noop,
        create_vllm_config=_noop,
        enable_engine_fused_ops=_noop,
        setup_distributed=_noop,
        with_exit_stack=lambda function: function,
    )
    _install_helper_stub(monkeypatch)


def _load_backend(monkeypatch, backend: str):
    import collector.op_plans  # noqa: F401  (the real plans, before collector.helper is stubbed)

    monkeypatch.setattr(sys, "path", list(sys.path))
    if backend == "trtllm":
        _install_trtllm_stubs(monkeypatch)
    else:
        _install_vllm_stubs(monkeypatch)
    module_name = f"msa_{backend}_getter_contract"
    path = REPO_ROOT / "collector" / backend / "collect_msa_module.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    if backend == "vllm":
        # The stub above makes _resolve_local_model_path an identity, so the
        # plan-time QSA geometry read must be fed the real bundled checkpoint
        # config instead of hitting the filesystem.
        monkeypatch.setattr(module, "_qsa_text_config", lambda _path: dict(_QWEN4EXP_TEXT_CONFIG))
    return module


def _invocation_identity(case) -> tuple[object, ...]:
    """Hashable rendering of a queued row (TRT-LLM rows carry their points as a list)."""

    def freeze(value):
        return tuple(freeze(item) for item in value) if isinstance(value, (list, tuple)) else value

    return freeze(case)


# Two checkpoint families are collected by the same module op: MiniMax-M3
# (MSA) and Qwen3.8-Flash-Next (QSA). Both persist into the MSA tables, keyed
# by the architecture the SDK/model config declares, so the consumer identity
# must carry the per-case architecture rather than the MSA model's constant.
MODEL_ARCHITECTURES = {
    MODEL_PATH: ARCHITECTURE,
    QWEN4EXP_PATH: "Qwen4ExpForConditionalGeneration",
}


def _consumer_keys(case, phase: str) -> list[tuple[object, ...]]:
    """The perf-table key each queued point of ``case`` fills.

    TRT-LLM queues a series row per module (``[points, batch, heads, ...]``), so
    one row fills one key per point; the XML-era per-point backends queue one
    case per point.
    """
    first, batch_size, num_heads, kv_dtype, compute_dtype, gemm_type, model_path, *rest = case
    architecture = MODEL_ARCHITECTURES[model_path]
    if isinstance(first, list):
        return [
            (architecture, compute_dtype, kv_dtype, gemm_type, num_heads, prefix, seq_len, batch_size)
            for seq_len, prefix in first
        ]
    if phase == "context":
        prefix_len = rest[0] if rest else 0
        return [(architecture, compute_dtype, kv_dtype, gemm_type, num_heads, prefix_len, first, batch_size)]
    return [(architecture, kv_dtype, gemm_type, num_heads, batch_size, first + 1)]


@pytest.mark.parametrize("backend", ["trtllm", "vllm"])
def test_public_msa_getters_queue_the_op_plan(monkeypatch, backend):
    from collector.op_plans import plan

    module = _load_backend(monkeypatch, backend)
    # TRT-LLM leaves the Qwen QSA route out (collector/skips.py).
    allowed_models = {MODEL_PATH} if backend == "trtllm" else set(MODEL_ARCHITECTURES)
    for op, getter in (
        ("msa_context_module", module.get_msa_context_module_test_cases),
        ("msa_generation_module", module.get_msa_generation_module_test_cases),
    ):
        rows = getter()
        assert rows and rows == plan(op, backend, 100).rows
        assert {row[6] for row in rows} == allowed_models


@pytest.mark.parametrize("backend", ["trtllm", "vllm"])
def test_public_msa_getters_populate_both_phases_without_duplicate_identities(monkeypatch, backend):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    module = _load_backend(monkeypatch, backend)
    allowed_models = {MODEL_PATH} if backend == "trtllm" else set(MODEL_ARCHITECTURES)
    for phase, cases in (
        ("context", module.get_msa_context_module_test_cases()),
        ("generation", module.get_msa_generation_module_test_cases()),
    ):
        assert cases, f"{backend}/{phase} returned no cases"
        assert {case[6] for case in cases} == allowed_models
        assert len(cases) == len({_invocation_identity(case) for case in cases})
        consumer_keys = [key for case in cases for key in _consumer_keys(case, phase)]
        assert len(consumer_keys) == len(set(consumer_keys)), f"{backend}/{phase} repeats a consumer key"


def test_trtllm_series_rows_fit_the_series_worker(monkeypatch):
    module = _load_backend(monkeypatch, "trtllm")
    captured = []
    monkeypatch.setattr(module, "run_msa_module_series", lambda *args, **kwargs: captured.append((args, kwargs)))

    row = module.get_msa_context_module_test_cases()[0]
    module.run_msa_module_series_worker(*row, perf_filename="context.txt")

    points, batch_size, num_heads, kv_dtype, compute, gemm, model_path = row
    assert captured == [
        (
            (points, batch_size, num_heads, kv_dtype, compute, gemm, "context.txt"),
            {"model_path": model_path, "device": "cuda:0"},
        )
    ]


def test_vllm_msa_getters_populate_qsa_per_rank_heads(monkeypatch):
    """Qwen4-Exp contributes its own exact per-rank heads, not the MSA grid."""
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    module = _load_backend(monkeypatch, "vllm")

    for getter in (module.get_msa_context_module_test_cases, module.get_msa_generation_module_test_cases):
        cases = [case for case in getter() if case[6] == QWEN4EXP_PATH]
        assert cases
        # 24 native query heads at TP 1/2/4/8; the QSA layer is bf16 only.
        assert {case[2] for case in cases} == {24, 12, 6, 3}
        assert {(case[3], case[4], case[5]) for case in cases} == {("bfloat16", "bfloat16", "bfloat16")}
    # The worker resolves the attention type from the model path; an unknown
    # path must fail closed rather than silently run MSA.
    assert module._attention_type_for_model(QWEN4EXP_PATH) == "qsa"
    assert module._attention_type_for_model(MODEL_PATH) == "msa"
    # Rows are persisted under the canonical model_path only, so a bare alias
    # is never a case identity and fails closed.
    with pytest.raises(ValueError):
        module._attention_type_for_model("Qwen/Qwen3.8-Flash-Next-FP8")


@pytest.mark.parametrize("backend", ["trtllm", "vllm"])
def test_targeted_msa_getters_keep_prefixes_and_worker_tuple_compatibility(monkeypatch, backend):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL_PATH)
    module = _load_backend(monkeypatch, backend)
    context_cases = module.get_msa_context_module_test_cases()
    generation_cases = module.get_msa_generation_module_test_cases()

    assert context_cases and generation_cases
    assert {case[6] for case in context_cases} == {MODEL_PATH}
    assert {case[6] for case in generation_cases} == {MODEL_PATH}
    if backend == "trtllm":
        # TRT-LLM rows carry their (seq_len, prefix_len) points inline.
        context_prefixes = {prefix for case in context_cases for _seq, prefix in case[0]}
        generation_prefixes = {prefix for case in generation_cases for _seq, prefix in case[0]}
    else:
        # vLLM cases are one point each, with the prefix as their last field.
        context_prefixes = {case[7] for case in context_cases}
        generation_prefixes = {case[7] for case in generation_cases}
    # A targeted run keeps the declared context prefixes; generation carries
    # none (a decode point prefills nothing).
    assert context_prefixes > {0}
    assert generation_prefixes == {0}


def test_vllm_cases_fit_the_point_worker(monkeypatch):
    module = _load_backend(monkeypatch, "vllm")
    captured = []
    monkeypatch.setattr(module, "run_msa_module", lambda **kwargs: captured.append(kwargs) or 1.0)

    context = next(case for case in module.get_msa_context_module_test_cases() if case.prefix_len)
    generation = module.get_msa_generation_module_test_cases()[0]
    module.run_msa_module_worker(*context, perf_filename="context.txt")
    module.run_msa_module_worker(*generation, perf_filename="generation.txt")

    assert captured[0]["prefix_len"] == context.prefix_len
    assert captured[0]["perf_filename"] == "context.txt"
    assert captured[1]["prefix_len"] == 0
    assert captured[1]["seq_len"] == generation.seq_len


def test_vllm_refuses_a_decode_point_whose_kv_cache_cannot_fit_before_allocating(monkeypatch):
    module = _load_backend(monkeypatch, "vllm")
    monkeypatch.setattr(
        module.torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=2**30), raising=False
    )
    monkeypatch.setattr(module, "setup_distributed", lambda _device: pytest.fail("allocated for a refused point"))

    with pytest.raises(RuntimeError, match=r"^FIXME\(kernel-limit\): does not fit: the KV cache needs"):
        module.run_msa_module(
            None,
            seq_len=131072,
            batch_size=64,
            num_heads=64,
            kv_cache_dtype="bfloat16",
            compute_dtype="bfloat16",
            gemm_type="bfloat16",
            perf_filename="msa_generation_module_perf.txt",
            model_path=MODEL_PATH,
        )


def test_vllm_standalone_cli_runs_the_plan_cases_and_quick_defaults_to_prefix_zero(monkeypatch):
    from collector.op_plans import ModuleCase

    module = _load_backend(monkeypatch, "vllm")
    cases = [
        ModuleCase(16, 1, 64, "bfloat16", "bfloat16", "bfloat16", MODEL_PATH, 0),
        ModuleCase(16, 1, 64, "bfloat16", "bfloat16", "bfloat16", MODEL_PATH, 512),
    ]
    monkeypatch.setattr(module, "plan", lambda *_args: SimpleNamespace(rows=cases))
    calls = []
    monkeypatch.setattr(module, "run_msa_module", lambda **kwargs: calls.append(kwargs))

    monkeypatch.setattr(sys, "argv", ["collect_msa_module.py", "--mode", "context"])
    module.main()
    assert [call["prefix_len"] for call in calls] == [0, 512]

    calls.clear()
    monkeypatch.setattr(sys, "argv", ["collect_msa_module.py", "--mode", "context", "--quick"])
    module.main()
    assert calls
    assert {call["prefix_len"] for call in calls} == {0}


def test_vllm_qsa_kernel_source_labels_are_declared():
    """Every label ``_qsa_kernel_source`` can emit is in the curated table.

    The translation table turns a persisted ``kernel_source`` into a real
    backend; an undeclared label silently degrades to ``unverified`` in
    op_backend_facts, which would hide that the QSA rows ran a named Triton
    route.
    """
    import yaml

    table = yaml.safe_load((REPO_ROOT / "collector" / "kernel_source_backends.yaml").read_text())
    declared = {entry["kernel_source"] for entry in table["mappings"] if entry["framework"] == "vllm"}
    assert {"vllm_qwen4exp_qsa_prefill_triton", "vllm_qwen4exp_qsa_decode_triton"} <= declared


def test_vllm_qsa_generation_footprint_is_the_checkpoint_geometry(monkeypatch):
    """The QSA filter must price the checkpoint's own KV, not the M3 constants.

    Per token at 24 heads: paged main K/V (2 x 2 heads x 256 dim x 2 B =
    2048) plus the compressed key (128 x 2 B / 4 = 64) = 2112 B. The raw-key
    ring is NOT token-linear: ``QSAKeyStateCache.get_kv_cache_spec`` declares a
    ``CircularBufferSpec`` of ``compress_ratio`` rows per request
    (``common/qsa_cache.py:844-862@v0.32.0``), each carrying a 128-dim BF16 key
    rounded to the int64 word plus three int64 MRoPE axes (140 words), so one
    request costs 4 * 140 * 2 = 1120 B whatever its cached length. The M3
    fallback prices 768 B/token, i.e. three times more tokens admitted than the
    device can hold.
    """
    module = _load_backend(monkeypatch, "vllm")

    footprint = module._qsa_generation_kv_footprint_bytes(QWEN4EXP_PATH)

    assert footprint(1, 24, "bfloat16", num_requests=1) == 2112 + 1120
    assert footprint(1, 24, "fp8", num_requests=1) == 1088 + 1120
    # A TP-sharded head shard keeps one KV head (max(1, 6 // 12)).
    assert footprint(1, 6, "bfloat16", num_requests=1) == 1088 + 1120
    # The ring follows the batch, not the token count.
    assert footprint(1, 24, "bfloat16", num_requests=8) == 2112 + 8 * 1120
    assert footprint(8192, 24, "bfloat16", num_requests=1) == 2112 * 8192 + 1120
    # The M3 default filter is the one this replaces.
    assert module._generation_kv_footprint_bytes(1, 24, "bfloat16") == 768


def test_vllm_qsa_generation_filter_charges_the_ring_per_request(monkeypatch):
    """The batch reaches the footprint, and the boundary follows the ring.

    A per-token total that folded the ring in would grow with the cached
    prefix and shrink with the batch; the per-request ring does neither. Check
    the forwarding and a device budget that sits between the two totals.
    """
    module = _load_backend(monkeypatch, "vllm")
    footprint = module._qsa_generation_kv_footprint_bytes(QWEN4EXP_PATH)

    seen = []

    def probe(total_tokens, num_heads, dtype, *, num_requests):
        seen.append(num_requests)
        return 1

    assert module._kv_does_not_fit(10, 24, "bfloat16", 1 << 40, footprint=probe, num_requests=8) is None
    assert seen == [8]

    total_tokens, batch = 8192, 8
    corrected = footprint(total_tokens, 24, "bfloat16", num_requests=batch)
    # The old per-token charge: ring (128 x 2 B + 3 int64 axes) per token.
    old_per_token_total = total_tokens * (2112 + 280)
    assert corrected < old_per_token_total

    safety = module._MEMORY_BUDGET_SAFETY_FACTOR
    inside = (corrected + old_per_token_total) / 2
    assert (
        module._kv_does_not_fit(
            total_tokens,
            24,
            "bfloat16",
            inside / safety,
            footprint=footprint,
            num_requests=batch,
        )
        is None
    )
    below = module._kv_does_not_fit(
        total_tokens, 24, "bfloat16", (corrected - 4096) / safety, footprint=footprint, num_requests=batch
    )
    assert below is not None and "does not fit" in below


def test_vllm_mrope_detection_follows_the_rope_parameters(monkeypatch):
    """``_uses_mrope_from_config`` mirrors vLLM's ``uses_mrope`` predicate."""
    module = _load_backend(monkeypatch, "vllm")

    assert module._uses_mrope_from_config(dict(_QWEN4EXP_TEXT_CONFIG)) is True
    assert module._uses_mrope_from_config({}) is False
    assert module._uses_mrope_from_config({"rope_parameters": {"rope_type": "default"}}) is False
    # The legacy ``rope_scaling`` spelling is accepted too.
    assert module._uses_mrope_from_config({"rope_scaling": {"mrope_section": [8, 8, 8]}}) is True


@pytest.mark.parametrize("failure", ["returns_false", "raises"])
def test_vllm_persist_msa_row_fails_the_case_on_every_write_failure(monkeypatch, failure):
    """A lost row must abort the case, not shrink the table one point at a time."""
    module = _load_backend(monkeypatch, "vllm")

    def log_perf(**_kwargs):
        if failure == "raises":
            raise module.PerfLogError("retained twin present")
        return False

    monkeypatch.setattr(module, "log_perf", log_perf)

    with pytest.raises(module.PerfLogWriteError, match="failed to persist MSA row"):
        module._persist_msa_row(
            item={},
            vllm_version="0.31.0",
            device_name="NVIDIA H200",
            op_name="msa_generation_module",
            kernel_source="vllm_qwen4exp_qsa_decode_triton",
            perf_filename="msa_generation_module_perf.txt",
            power_stats=None,
        )


def test_vllm_qsa_decode_points_are_refused_on_the_checkpoint_geometry(monkeypatch):
    """A QSA decode point is priced on the checkpoint's KV, not the M3 constants.

    At 24 heads a QSA point costs 2392 B/token against the M3 layer's 768 B, so
    the M3 filter would admit ~3x more tokens than the device holds.
    """
    module = _load_backend(monkeypatch, "vllm")
    monkeypatch.setattr(
        module.torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=2**30), raising=False
    )
    monkeypatch.setattr(module, "setup_distributed", lambda _device: pytest.fail("allocated for a refused point"))

    # 64 x 8192 tokens x 2392 B = 1.17 GiB against a 0.97 GiB allowance: the
    # same point priced at the M3 768 B/token (0.38 GiB) would have been run.
    with pytest.raises(RuntimeError, match=r"KV cache needs 1\.[0-9] GiB"):
        module.run_msa_module(
            None,
            seq_len=8192,
            batch_size=64,
            num_heads=24,
            kv_cache_dtype="bfloat16",
            compute_dtype="bfloat16",
            gemm_type="bfloat16",
            perf_filename="msa_generation_module_perf.txt",
            model_path=QWEN4EXP_PATH,
        )
