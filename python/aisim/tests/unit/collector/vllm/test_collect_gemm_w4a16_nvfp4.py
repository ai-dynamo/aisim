# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-weight preparation of the vLLM w4a16_nvfp4 GEMM lane.

Packing uses real torch tensors; capacity checks use device metadata fakes.
vLLM is replaced by a minimal import surface, so these tests check the
collector-owned packing and branch selection, not vLLM's quantizer numerics.
"""

import importlib
import importlib.metadata
import importlib.util
import sys
import types
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

COLLECT_GEMM = Path(__file__).resolve().parents[4] / "collector" / "vllm" / "collect_gemm.py"
# E2M1 code -> magnitude, from the OCP MX format definition (sign in bit 3).
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _decode(torch, packed):
    """Independent decoder: low nibble is the even element, high nibble the odd one."""
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(packed.shape[0], -1).long()
    magnitude = torch.tensor(E2M1)[codes & 0x7]
    return torch.where((codes & 0x8).bool(), -magnitude, magnitude)


def _initialized_torch():
    """The real torch package if this process already initialized it, else None.

    A later fake can hide it under sys.modules["torch"], but its loaded
    submodules keep a reference to the package.
    """
    tensor_module = sys.modules.get("torch._tensor")
    package = getattr(tensor_module, "torch", None)
    return package if isinstance(package, types.ModuleType) else None


@pytest.fixture
def torch(monkeypatch):
    """The real torch, also when another test module cached a MagicMock as torch.

    torch can be initialized only once per process, so an initialized package
    is reused instead of being imported again.
    """
    # Ask the installed distributions, not sys.modules: other collector test
    # modules cache a MagicMock under sys.modules["torch"] at collection time.
    try:
        importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("real torch required for tensor operations")
    module = sys.modules.get("torch")
    if not isinstance(module, types.ModuleType):
        module = _initialized_torch()
    if module is None:
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        module = importlib.import_module("torch")
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


@pytest.fixture
def collect_gemm(torch, monkeypatch):
    return _load_collect_gemm(torch, monkeypatch)


@pytest.fixture
def capacity_gemm(monkeypatch):
    """Capacity accounting uses CUDA metadata only, so it needs no tensor library."""
    torch = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", torch)
    return _load_collect_gemm(torch, monkeypatch)


def _load_collect_gemm(torch, monkeypatch):
    for name in [
        "vllm",
        "vllm.envs",
        "vllm.version",
        "vllm._custom_ops",
        "vllm.config",
        "vllm.model_executor",
        "vllm.model_executor.kernels",
        "vllm.model_executor.kernels.linear",
        "vllm.model_executor.kernels.linear.nvfp4",
        "vllm.model_executor.kernels.linear.nvfp4.marlin",
        "vllm.model_executor.kernels.linear.scaled_mm",
        "vllm.model_executor.kernels.linear.scaled_mm.flashinfer",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.linear",
        "vllm.model_executor.layers.quantization",
        "vllm.model_executor.layers.quantization.compressed_tensors",
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
        "vllm.model_executor.layers.quantization.fp8",
        "vllm.model_executor.layers.quantization.modelopt",
        "vllm.model_executor.layers.quantization.utils",
        "vllm.model_executor.layers.quantization.utils.marlin_utils",
        "vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils",
        "vllm.utils",
        "vllm.utils.deep_gemm",
    ]:
        module = types.ModuleType(name)
        module.__getattr__ = lambda attr: MagicMock()
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["vllm.version"].__version__ = "0.0.0"
    monkeypatch.setitem(
        sys.modules,
        "collector.vllm.utils",
        types.SimpleNamespace(setup_distributed=lambda *a, **k: None, with_exit_stack=lambda f: f),
    )
    spec = importlib.util.spec_from_file_location("collect_gemm_under_test", COLLECT_GEMM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # The memory tests supply framework dispatch/padding results, without loading CUDA.
    class MarlinKernel:
        pass

    module.MarlinNvFp4LinearKernel = MarlinKernel
    module.init_nvfp4_linear_kernel = lambda **kw: MarlinKernel() if module.get_sm_version() < 100 else object()
    module.MARLIN_MAX_BLOCKS_PER_SM = 4
    module.USE_FP32_REDUCE_DEFAULT = True
    padded_shapes = {
        (65536, 51200): (65536, 51200),
        (128, 4096): (128, 4096),
        (65536, 16): (65536, 128),
        (65536, 1024): (65536, 1024),
        (32, 1024): (64, 1024),
        (64, 1040): (64, 1152),
        (128, 16): (128, 64),
    }
    module.marlin_padded_nk = lambda n, k, group_size: padded_shapes[n, k]
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _d: types.SimpleNamespace(multi_processor_count=132)
    )
    return module


def test_pack_e2m1_round_trips_through_an_independent_decoder(torch, collect_gemm):
    generator = torch.Generator().manual_seed(0)
    signs = torch.where(torch.rand(4, 32, generator=generator) < 0.5, -1.0, 1.0)
    values = signs * torch.tensor(E2M1)[torch.randint(0, 8, (4, 32), generator=generator)]

    packed = collect_gemm._pack_e2m1(values)

    assert packed.dtype == torch.uint8
    assert packed.shape == (4, 16)
    assert torch.equal(_decode(torch, packed), values)


@pytest.mark.parametrize("sm", [80, 89, 90])
@pytest.mark.parametrize("chunk_rows", [2, 3, 8])
def test_pre_blackwell_weights_use_the_reference_quantizer(torch, collect_gemm, monkeypatch, sm, chunk_rows):
    n, k = 8, 64
    weight = torch.randn(n, k, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    calls = []
    # Distinct, FP8-representable block scales in every row and block, so their order is checked.
    scales = torch.tensor([m * 2.0**e for e in range(-2, 6) for m in (1, 1.25, 1.5, 1.75)]).reshape(n, k // 16)

    def reference(x, global_scale, block_size):
        calls.append((x, global_scale, block_size))
        start = x.storage_offset() // k
        values = torch.tensor(E2M1).repeat(len(x), k // len(E2M1))
        return values, scales[start : start + len(x)]

    def blackwell_only(*args, **kwargs):
        raise AssertionError("scaled_fp4_quant needs an SM100+ kernel")

    monkeypatch.setattr(collect_gemm, "get_sm_version", lambda: sm)
    monkeypatch.setattr(collect_gemm, "_W4A16_NVFP4_CHUNK_ELEMENTS", chunk_rows * k)
    monkeypatch.setattr(collect_gemm, "_ref_nvfp4_quant", reference)
    monkeypatch.setattr(collect_gemm, "_scaled_fp4_quant", blackwell_only)

    weight_fp4, weight_scale, weight_scale_2 = collect_gemm._w4a16_nvfp4_checkpoint_weights(weight)

    amax = float(weight.abs().max())
    assert weight_scale_2 == pytest.approx(amax / 2688.0)
    assert len(calls) == (n + chunk_rows - 1) // chunk_rows
    assert torch.equal(torch.cat([x for x, _, _ in calls]), weight)
    for x, global_scale, block_size in calls:
        assert len(x) <= chunk_rows and block_size == 16
        assert float(global_scale) == pytest.approx(2688.0 / amax)
    assert weight_fp4.dtype == torch.uint8 and weight_fp4.shape == (n, k // 2)
    assert torch.equal(_decode(torch, weight_fp4), torch.tensor(E2M1).repeat(n, k // len(E2M1)))
    assert weight_scale.dtype == torch.float8_e4m3fn and weight_scale.shape == (n, k // 16)
    assert torch.equal(weight_scale.float(), scales)


def test_weight_preparation_fit_check_releases_cached_memory_first(capacity_gemm, monkeypatch):
    collect_gemm = capacity_gemm
    gib, mib = 2**30, 2**20
    cuda = collect_gemm.torch.cuda
    state = {"free": 0, "cached": 0}

    def device(free, cached=0):
        state.update(free=free, cached=cached)

    def empty_cache():
        state["free"] += state.pop("cached", 0)

    monkeypatch.setattr(cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(cuda, "mem_get_info", lambda _device: (state["free"], 80 * gib))
    monkeypatch.setattr(collect_gemm, "get_sm_version", lambda: 90)
    # The full SM80-plan weight needs 9.77 GiB for the layer, BF16 input and prepared
    # packed outputs, plus bounded scratch. Unused cached segments count after release.
    # The old full-matrix reference quantizer needed over 40 GiB for this shape.
    n, k = 65536, 51200
    device(4 * gib, cached=6 * gib)
    assert collect_gemm._w4a16_nvfp4_does_not_fit(1, n, k, "cuda:0") is None
    device(4 * gib, cached=5 * gib)
    assert collect_gemm._w4a16_nvfp4_does_not_fit(1, n, k, "cuda:0").startswith("FIXME(kernel-limit): does not fit")
    # Include the activation alongside preparation and chunk scratch.
    device(18 * mib)
    assert collect_gemm._w4a16_nvfp4_does_not_fit(1, 128, 4096, "cuda:0") is None
    device(18 * mib)
    assert collect_gemm._w4a16_nvfp4_does_not_fit(256, 128, 4096, "cuda:0").startswith("FIXME(kernel-limit)")
    # The forward's BF16 output counts as well: a small weight with a 4 GiB output does not fit in 3 GiB.
    device(3 * gib)
    assert collect_gemm._w4a16_nvfp4_does_not_fit(32768, 65536, 16, "cuda:0").startswith("FIXME(kernel-limit)")
    # The layer's FP8 block scales count too: activation, packed weight and output fit in 4192 MiB,
    # the 4 MiB of scales do not.
    monkeypatch.setattr(collect_gemm, "get_sm_version", lambda: 100)
    device(4192 * mib)
    assert collect_gemm._w4a16_nvfp4_does_not_fit(32768, 65536, 1024, "cuda:0").startswith("FIXME(kernel-limit)")


@pytest.mark.parametrize(
    "m,n,k,sms,available_mib,fits_mib",
    [
        # Aligned output: 8.25 MiB FP32 reduction scratch plus locks.
        (32768, 65536, 1024, 132, 4200, 4205),
        # N=32 pads to 64: 4 MiB physical output, rather than 2 MiB.
        (32768, 32, 1024, 132, 75, 77),
        # With fewer SMs, the 2 MiB contiguous slice copy dominates reduction scratch.
        (32768, 32, 1024, 1, 69, 71),
        # K padding retains the logical activation and adds a 72 MiB padded activation.
        (32768, 64, 1040, 132, 145, 150),
        # A single row uses 16, rather than 64, rows of reduction storage.
        (1, 128, 16, 132, 2, 3),
    ],
)
def test_marlin_physical_forward_capacity(capacity_gemm, monkeypatch, m, n, k, sms, available_mib, fits_mib):
    collect_gemm, torch = capacity_gemm, capacity_gemm.torch
    monkeypatch.setattr(collect_gemm, "get_sm_version", lambda: 90)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _d: types.SimpleNamespace(multi_processor_count=sms)
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _d: (available_mib * 2**20, 80 * 2**30))
    assert collect_gemm._w4a16_nvfp4_does_not_fit(m, n, k, "cuda:0").startswith("FIXME(kernel-limit): does not fit")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _d: (fits_mib * 2**20, 80 * 2**30))
    assert collect_gemm._w4a16_nvfp4_does_not_fit(m, n, k, "cuda:0") is None


@pytest.mark.parametrize("sm,marlin", [(90, True), (100, True), (100, False)])
def test_forward_budget_uses_selected_kernel(capacity_gemm, monkeypatch, sm, marlin):
    collect_gemm, torch = capacity_gemm, capacity_gemm.torch
    monkeypatch.setattr(collect_gemm, "get_sm_version", lambda: sm)
    selected = collect_gemm.MarlinNvFp4LinearKernel() if marlin else object()
    selector = MagicMock(return_value=selected)
    monkeypatch.setattr(collect_gemm, "init_nvfp4_linear_kernel", selector)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _d: (75 * 2**20, 80 * 2**30))
    refusal = collect_gemm._w4a16_nvfp4_does_not_fit(32768, 32, 1024, "cuda:0")
    assert (refusal is not None) == marlin
    selector.assert_called_once_with(use_a16=True)


@pytest.mark.parametrize(
    "m,n,k,available",
    [(1, 65536, 51200, 8 * 2**30), (32768, 65536, 1024, 4200 * 2**20), (32768, 32, 1024, 75 * 2**20)],
)
def test_capacity_refusal_precedes_tensor_allocation(capacity_gemm, monkeypatch, m, n, k, available):
    collect_gemm, torch = capacity_gemm, capacity_gemm.torch
    monkeypatch.setattr(collect_gemm.envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_a: None)
    monkeypatch.setattr(torch, "set_default_dtype", lambda *_a: None)
    monkeypatch.setattr(torch, "set_default_device", lambda *_a: None)
    monkeypatch.setattr(collect_gemm, "get_sm_version", lambda: 80)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _d: (available, 40 * 2**30))
    stack = MagicMock()
    monkeypatch.setattr(collect_gemm, "VllmConfig", lambda: types.SimpleNamespace(model_config=None))

    def unexpected_allocation(*_a, **_kw):
        raise AssertionError("must refuse before allocating the activation or layer weights")

    monkeypatch.setattr(torch, "randn", unexpected_allocation)
    with pytest.raises(RuntimeError, match="^FIXME\\(kernel-limit\\): does not fit"):
        collect_gemm.run_gemm(stack, "w4a16_nvfp4", m, n, k, perf_filename="unused")


def test_marlin_budget_keeps_serving_context_and_publishes_logical_shape(capacity_gemm, monkeypatch):
    collect_gemm, torch = capacity_gemm, capacity_gemm.torch
    state = {"in_context": False}
    selected = collect_gemm.MarlinNvFp4LinearKernel()

    @contextmanager
    def config_context(config):
        assert config.model_config.dtype is torch.bfloat16
        state["in_context"] = True
        yield
        state["in_context"] = False

    def select_kernel(**kwargs):
        assert state["in_context"] and kwargs == {"use_a16": True}
        return selected

    monkeypatch.setattr(collect_gemm.envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(collect_gemm, "get_sm_version", lambda: 90)
    monkeypatch.setattr(collect_gemm, "VllmConfig", lambda: types.SimpleNamespace(model_config=None))
    monkeypatch.setattr(collect_gemm, "set_current_vllm_config", config_context)
    monkeypatch.setattr(collect_gemm, "init_nvfp4_linear_kernel", select_kernel)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _d: (80 * 2**30, 80 * 2**30))
    layer = MagicMock(spec=["to", "weight", "weight_scale", "weight_scale_2", "quant_method", "forward"])
    layer.quant_method.kernel = selected
    monkeypatch.setattr(collect_gemm, "RowParallelLinear", lambda **kw: layer)
    monkeypatch.setattr(collect_gemm, "_w4a16_nvfp4_checkpoint_weights", lambda _w: (object(), object(), 1.0))

    @contextmanager
    def benchmark(**kwargs):
        kwargs["kernel_func"]()
        yield {"used_cuda_graph": True, "latency_ms": 1.0, "power_stats": {}}

    monkeypatch.setattr(collect_gemm, "benchmark_with_power", benchmark)
    publish = MagicMock()
    monkeypatch.setattr(collect_gemm, "log_perf", publish)
    with ExitStack() as stack:
        collect_gemm.run_gemm(stack, "w4a16_nvfp4", 32768, 32, 1024, perf_filename="unused")
    assert not state["in_context"]
    row = publish.call_args.kwargs["item_list"][0]
    assert (row["m"], row["n"], row["k"]) == (32768, 32, 1024)
    assert row["gemm_dtype"] == "w4a16_nvfp4"
    assert publish.call_args.kwargs["kernel_source"] == type(selected).__name__
    assert layer.forward.call_count == 2  # Dry run and fake benchmark.


def test_blackwell_keeps_the_scaled_fp4_quant_kernel(torch, collect_gemm, monkeypatch):
    weight = torch.randn(4, 32).to(torch.bfloat16)
    sentinel = (torch.zeros(4, 16, dtype=torch.uint8), torch.zeros(4, 2))
    calls = []
    monkeypatch.setattr(collect_gemm, "get_sm_version", lambda: 100)
    monkeypatch.setattr(collect_gemm, "_scaled_fp4_quant", lambda *a, **kw: calls.append(kw) or sentinel)

    weight_fp4, weight_scale, _ = collect_gemm._w4a16_nvfp4_checkpoint_weights(weight)

    assert calls == [{"is_sf_swizzled_layout": False}]
    assert weight_fp4 is sentinel[0] and weight_scale is sentinel[1]
