# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-weight preparation of the vLLM w4a16_nvfp4 GEMM lane.

Real torch tensors; vLLM itself is replaced by a minimal import surface, so
these tests check the collector-owned packing and branch selection, not
vLLM's quantizer numerics.
"""

import importlib
import importlib.metadata
import importlib.util
import sys
import types
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
    for name in [
        "vllm",
        "vllm.envs",
        "vllm.version",
        "vllm._custom_ops",
        "vllm.config",
        "vllm.model_executor",
        "vllm.model_executor.kernels",
        "vllm.model_executor.kernels.linear",
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
def test_pre_blackwell_weights_use_the_reference_quantizer(torch, collect_gemm, monkeypatch, sm):
    n, k = 8, 64
    weight = torch.randn(n, k, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    calls = []
    # Distinct, FP8-representable block scales in every row and block, so their order is checked.
    scales = torch.tensor([m * 2.0**e for e in range(-2, 6) for m in (1, 1.25, 1.5, 1.75)]).reshape(n, k // 16)

    def reference(x, global_scale, block_size):
        calls.append((x, global_scale, block_size))
        values = torch.tensor(E2M1).repeat(n, k // len(E2M1))
        return values, scales

    def blackwell_only(*args, **kwargs):
        raise AssertionError("scaled_fp4_quant needs an SM100+ kernel")

    monkeypatch.setattr(collect_gemm, "get_sm_version", lambda: sm)
    monkeypatch.setattr(collect_gemm, "_ref_nvfp4_quant", reference)
    monkeypatch.setattr(collect_gemm, "_scaled_fp4_quant", blackwell_only)

    weight_fp4, weight_scale, weight_scale_2 = collect_gemm._w4a16_nvfp4_checkpoint_weights(weight)

    amax = float(weight.abs().max())
    assert weight_scale_2 == pytest.approx(amax / 2688.0)
    ((x, global_scale, block_size),) = calls
    assert x is weight and block_size == 16
    assert float(global_scale) == pytest.approx(2688.0 / amax)
    assert weight_fp4.dtype == torch.uint8 and weight_fp4.shape == (n, k // 2)
    assert torch.equal(_decode(torch, weight_fp4), torch.tensor(E2M1).repeat(n, k // len(E2M1)))
    assert weight_scale.dtype == torch.float8_e4m3fn and weight_scale.shape == (n, k // 16)
    assert torch.equal(weight_scale.float(), scales)


def test_weight_preparation_counts_reusable_allocator_memory(torch, collect_gemm, monkeypatch):
    gib = 2**30
    monkeypatch.setattr(collect_gemm, "get_sm_version", lambda: 90)
    monkeypatch.setattr(collect_gemm.torch.cuda, "mem_get_info", lambda _device: (1 * gib, 80 * gib))
    monkeypatch.setattr(collect_gemm.torch.cuda, "memory_allocated", lambda _device: 2 * gib)
    # Below SM100 a 1 Gi-element weight needs at least 6.5 GiB.
    n, k = gib, 1
    monkeypatch.setattr(collect_gemm.torch.cuda, "memory_reserved", lambda _device: 8 * gib)
    assert collect_gemm._w4a16_nvfp4_does_not_fit(1, n, k, "cuda:0") is None  # 1 free + 6 reusable
    monkeypatch.setattr(collect_gemm.torch.cuda, "memory_reserved", lambda _device: 6 * gib)
    refusal = collect_gemm._w4a16_nvfp4_does_not_fit(1, n, k, "cuda:0")  # 1 free + 4 reusable
    assert refusal.startswith("FIXME(kernel-limit): does not fit")
    # The activation is budgeted too: 1 GiB more of it no longer fits in 1 free + 6 reusable.
    monkeypatch.setattr(collect_gemm.torch.cuda, "memory_reserved", lambda _device: 8 * gib)
    assert collect_gemm._w4a16_nvfp4_does_not_fit(gib // 2, n, 1, "cuda:0").startswith("FIXME(kernel-limit)")
    # The forward's BF16 output counts as well: a small weight with a 4 GiB output does not fit in 3 GiB.
    monkeypatch.setattr(collect_gemm.torch.cuda, "memory_reserved", lambda _device: 4 * gib)
    assert collect_gemm._w4a16_nvfp4_does_not_fit(32768, 65536, 16, "cuda:0").startswith("FIXME(kernel-limit)")


def test_blackwell_keeps_the_scaled_fp4_quant_kernel(torch, collect_gemm, monkeypatch):
    weight = torch.randn(4, 32).to(torch.bfloat16)
    sentinel = (torch.zeros(4, 16, dtype=torch.uint8), torch.zeros(4, 2))
    calls = []
    monkeypatch.setattr(collect_gemm, "get_sm_version", lambda: 100)
    monkeypatch.setattr(collect_gemm, "_scaled_fp4_quant", lambda *a, **kw: calls.append(kw) or sentinel)

    weight_fp4, weight_scale, _ = collect_gemm._w4a16_nvfp4_checkpoint_weights(weight)

    assert calls == [{"is_sf_swizzled_layout": False}]
    assert weight_fp4 is sentinel[0] and weight_scale is sentinel[1]
