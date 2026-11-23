# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the real GEMM timing/publication block without a GPU or vLLM install.

Layer construction and kernel selection are stubbed; the real rotation loop and
timing/publication block execute against a recording benchmark context. This
checks distinct weight reuse, per-operation normalization, graph-off requests,
and publication of an eager result even if the helper changes its behavior.
"""

import ast
import contextlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[3] / "collector" / "vllm" / "collect_gemm.py"


def _fake_tuning_bucket(m, max_tuned_tokens=None):
    """helper.flashinfer_tuning_bucket for the m values below: FlashInfer's hybrid ceil."""
    return {1: 1, 48: 64, 257: 512, 4097: 8192}[m]


def _run_timing_block(
    gemm_type,
    result,
    *,
    capture_error=None,
    shape=(1, 768, 7168),
    l2_cache_bytes=60 << 20,
    kernel_source="test-selected-kernel",
    m=None,
    sm=120,
):
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_gemm")
    start = next(i for i, n in enumerate(run.body) if isinstance(n, ast.FunctionDef) and n.name == "kernel_func")
    rotation_start = next(
        i
        for i, n in enumerate(run.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "outside_loop_count" for target in n.targets)
    )
    rotation_end = next(i + 1 for i in range(rotation_start, start) if isinstance(run.body[i], ast.For))
    calls = {"benchmark": [], "weights": [], "forward": [], "published": [], "autotuned": []}
    x = object()

    def create_gemm():
        weight = object()
        calls["weights"].append(weight)
        return SimpleNamespace(forward=lambda arg: calls["forward"].append((weight, arg)))

    @contextmanager
    def benchmark_with_power(**kwargs):
        calls["benchmark"].append(kwargs)
        # Fail at the boundary if a dtype-specific eager exception is restored.
        assert kwargs["use_cuda_graph"] is True
        assert kwargs["allow_graph_fail"] is False
        if capture_error is not None:
            raise capture_error
        kwargs["kernel_func"]()
        kwargs["kernel_func"]()
        yield result

    @contextmanager
    def flashinfer_autotuned(num_tokens, tune_fn, *, label=""):
        calls["autotuned"].append((num_tokens, label))
        yield True

    namespace = {
        "contextlib": contextlib,
        "torch": SimpleNamespace(
            device=lambda d: d,
            cuda=SimpleNamespace(
                get_device_name=lambda d: "test GPU",
                get_device_properties=lambda d: SimpleNamespace(L2_cache_size=l2_cache_bytes),
            ),
        ),
        "benchmark_with_power": benchmark_with_power,
        "flashinfer_autotuned": flashinfer_autotuned,
        "flashinfer_tuning_bucket": _fake_tuning_bucket,
        "has_flashinfer": lambda: True,
        "current_platform": SimpleNamespace(has_device_capability=lambda capability: sm >= capability),
        "log_perf": lambda **kwargs: calls["published"].append(kwargs),
        "device": "cuda:0",
        "gemm_type": gemm_type,
        "m": shape[0] if m is None else m,
        "n": shape[1],
        "k": shape[2],
        "x": x,
        "create_gemm": create_gemm,
        "vllm_version": "test-version",
        "kernel_source": kernel_source,
        "perf_filename": "test-perf-file",
    }
    body = run.body[rotation_start:rotation_end] + run.body[start:]
    code = compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec")
    return lambda: exec(code, namespace), calls, x


@pytest.mark.parametrize("gemm_type,copies", [("bfloat16", 6), ("fp8", 6), ("fp8_block", 6), ("nvfp4", 1)])
def test_gemm_rotates_weights_and_publishes_per_operation_graph_timing(gemm_type, copies):
    run, calls, x = _run_timing_block(gemm_type, {"used_cuda_graph": True, "latency_ms": 0.012, "power_stats": None})
    run()
    assert len(calls["weights"]) == copies
    assert len({id(weight) for weight in calls["weights"]}) == copies
    assert calls["forward"] == [(weight, x) for weight in calls["weights"]] * 2
    assert len(calls["benchmark"]) == 1
    benchmark = calls["benchmark"][0]
    assert (benchmark["num_warmups"], benchmark["num_runs"], benchmark["repeat_n"]) == (3, 6, 1)
    assert len(calls["published"]) == 1
    published = calls["published"][0]
    assert published["kernel_source"] == "test-selected-kernel"
    assert published["item_list"][0]["gemm_dtype"] == gemm_type
    assert published["item_list"][0]["latency"] == pytest.approx(0.012 / copies)


@pytest.mark.parametrize(
    "shape,l2_cache_bytes,copies",
    [
        ((1, 1024, 1024), 1 << 20, 2),
        ((1, 11264, 1024), 10 << 20, 2),  # A matrix 10% larger than L2 still rotates.
        ((1, 20480, 1024), 10 << 20, 1),  # Exactly twice L2.
        ((1, 20480, 1024), (10 << 20) + 1, 2),
        ((1, 20480, 1024), (10 << 20) - 1, 1),
        ((1, 1024, 1024), 60 << 20, 6),  # Bound graph size for tiny weights.
    ],
)
def test_fp8_block_rotation_targets_twice_l2_with_a_six_copy_cap(shape, l2_cache_bytes, copies):
    run, calls, x = _run_timing_block(
        "fp8_block",
        {"used_cuda_graph": True, "latency_ms": 0.012, "power_stats": None},
        shape=shape,
        l2_cache_bytes=l2_cache_bytes,
    )
    run()
    assert len(calls["weights"]) == copies
    assert calls["forward"] == [(weight, x) for weight in calls["weights"]] * 2
    assert calls["published"][0]["item_list"][0]["latency"] == pytest.approx(0.012 / copies)


def test_fp8_block_large_weights_are_allocated_only_once():
    # Six 1 GiB weights plus vLLM's FP32 preparation buffers can OOM even
    # though the original single-copy case fits. The matrix already exceeds twice L2.
    run, calls, _ = _run_timing_block(
        "fp8_block",
        {"used_cuda_graph": True, "latency_ms": 0.012, "power_stats": None},
        shape=(1, 16384, 65536),
    )
    run()
    assert len(calls["weights"]) == 1
    assert calls["published"][0]["item_list"][0]["latency"] == pytest.approx(0.012)


@pytest.mark.parametrize("gemm_type,copies", [("bfloat16", 6), ("fp8", 6), ("nvfp4", 1)])
def test_l2_limit_preserves_other_precision_rotation(gemm_type, copies):
    run, calls, _ = _run_timing_block(
        gemm_type,
        {"used_cuda_graph": True, "latency_ms": 0.012, "power_stats": None},
        l2_cache_bytes=1,
    )
    run()
    assert len(calls["weights"]) == copies


@pytest.mark.parametrize("graph_flag", [False, None, 0, 1, "false", "true", "missing"])
def test_eager_or_unproven_result_is_not_published(graph_flag):
    result = {"latency_ms": 0.15, "power_stats": None}
    if graph_flag != "missing":
        result["used_cuda_graph"] = graph_flag
    run, calls, _ = _run_timing_block("fp8_block", result)
    with pytest.raises(RuntimeError, match="refusing to publish eager timing"):
        run()
    assert calls["published"] == []


def test_capture_failure_propagates_without_publication():
    failure = RuntimeError("capture failed")
    run, calls, _ = _run_timing_block("fp8_block", {}, capture_error=failure)
    with pytest.raises(RuntimeError, match="capture failed") as exc:
        run()
    assert exc.value is failure
    assert calls["published"] == []


def _run_config_block(qc):
    """Execute run_gemm's VllmConfig block, from its construction to entering it."""
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_gemm")
    start = next(
        i
        for i, n in enumerate(run.body)
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "vllm_config" for t in n.targets)
    )
    end = next(i for i, n in enumerate(run.body) if i > start and "set_current_vllm_config" in ast.unparse(n))
    built = []
    entered = []

    class VllmConfig:
        def __init__(self, **kwargs):
            built.append(kwargs)
            self.model_config = None

    namespace = {
        "VllmConfig": VllmConfig,
        "SimpleNamespace": SimpleNamespace,
        "set_current_vllm_config": lambda config: config,
        "exit_stack": SimpleNamespace(enter_context=entered.append),
        "dtype": "bf16",
        "qc": qc,
    }
    code = compile(ast.Module(body=run.body[start : end + 1], type_ignores=[]), str(SOURCE), "exec")
    exec(code, namespace)
    return built, entered


@pytest.mark.parametrize("qc", [None, object()], ids=["bfloat16", "quantized"])
def test_vllm_config_carries_the_layer_quant_config(qc):
    # Serving derives custom ops from VllmConfig.quant_config: block FP8 turns on
    # the CUDA quant_fp8 op (vllm/config/vllm.py has_blocked_weights). A bare
    # VllmConfig() would time the torch-native activation quant instead.
    built, entered = _run_config_block(qc)
    assert built == [{"quant_config": qc}]
    assert len(entered) == 1
    assert entered[0].model_config.hf_text_config.model_type == ""


@pytest.mark.parametrize(
    ("kernel_source", "sm", "tuned"),
    [
        ("FlashInferCutlassNvFp4LinearKernel", 120, True),
        ("CutlassFP8ScaledMMLinearKernel", 90, True),
        ("FlashInferCuteDslNvFp4LinearKernel", 100, False),
        # Warmup autotunes on SM90 and newer only (kernel_warmup.py:261-268).
        ("CutlassFP8ScaledMMLinearKernel", 89, False),
    ],
)
def test_flashinfer_autotune_follows_serving_skip_rule(kernel_source, sm, tuned):
    # Serving tunes every FlashInfer op except fp4_gemm under the cute-dsl kernel
    # (vllm warmup/kernel_warmup.py _flashinfer_autotune_skip_ops); the tuner leaves
    # kernels it does not own untouched, so only the cute-dsl lane skips the pass.
    run, calls, _ = _run_timing_block(
        "nvfp4", {"used_cuda_graph": True, "latency_ms": 1.0, "power_stats": None}, kernel_source=kernel_source, sm=sm
    )
    run()
    assert calls["autotuned"] == ([(1, kernel_source)] if tuned else [])
    assert len(calls["published"]) == 1


@pytest.mark.parametrize(("m", "bucket"), [(1, 1), (48, 64), (257, 512), (4097, 8192)])
def test_flashinfer_autotune_tunes_the_bucket_mm_fp4_maps_m_to(m, bucket):
    # Warmup profiles the hybrid buckets and mm_fp4 maps a runtime m up to the next
    # one, so m is timed on that bucket's tactic; the row keeps m.
    run, calls, _ = _run_timing_block(
        "nvfp4",
        {"used_cuda_graph": True, "latency_ms": 1.0, "power_stats": None},
        kernel_source="FlashInferCutlassNvFp4LinearKernel",
        m=m,
    )
    run()
    assert calls["autotuned"] == [(bucket, "FlashInferCutlassNvFp4LinearKernel")]
    assert calls["published"][0]["item_list"][0]["m"] == m
