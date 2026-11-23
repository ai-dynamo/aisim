# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan-time inner-manifest binding for the SGLang MSA collector (review
4969690316 Standards-2): every queued case carries the EXACT (b, s, prefix)
list it will execute — task IDs, resume checkpoints, and failure records bind
the retained shapes; the worker never re-derives the grid."""

import importlib.util
import sys
import types

import pytest

pytestmark = pytest.mark.unit


def _import_module(monkeypatch, sm: int = 100, cuda: bool = False):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda, empty_cache=lambda: None, set_device=lambda *_: None)
    torch.Tensor = object
    monkeypatch.setitem(sys.modules, "torch", torch)
    mod_name = "collector.sglang.collect_msa_module"
    monkeypatch.delitem(sys.modules, mod_name, raising=False)
    spec = importlib.util.spec_from_file_location(mod_name, "collector/sglang/collect_msa_module.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, mod_name, mod)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "get_sm_version", lambda: sm)
    return mod


def test_worker_requires_the_manifest(monkeypatch):
    """The worker never re-derives the grid: without a manifest (and without
    --quick) it must raise instead of silently rebuilding shapes."""
    mod = _import_module(monkeypatch)
    with pytest.raises(RuntimeError, match=r"empty-manifest|manifest"):
        mod.run_msa_module(
            num_heads=8,
            model_path="MiniMaxAI/MiniMax-M3",
            kv_cache_dtype="bfloat16",
            compute_dtype="bfloat16",
            gemm_type="bfloat16",
            is_prefill=False,
            gpu_id=0,
            inner_manifest=(),
        )
