# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic native-interface fixtures, not GPU or performance qualification."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.fpm_forward import runtime_k31, runtime_memory
from collector.fpm_forward.runtime import fpm_memory_observer as legacy
from collector.fpm_forward.runtime_instrumentation import load_instrumentation
from collector.fpm_forward.runtime_observations import _SEMANTICS, _physical_views, validate_observations

from .test_fpm_runner import _cell
from .test_fpm_runtime_memory import _Storage, _typed
from .test_runtime_observations import _mutate, _replace_launch, observation_fixture

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("runtime", [None, [], {}, {"version": "0.28.0"}])
def test_unaudited_runtime_does_not_read_k31_manifest(monkeypatch, runtime):
    def unexpected_read(*args, **kwargs):
        raise AssertionError("unrelated runtime read the K3.1 manifest")

    monkeypatch.setattr(Path, "read_text", unexpected_read)
    assert runtime_k31.is_audited_runtime(runtime) is False


def test_audited_runtime_still_requires_complete_source_map():
    manifest = Path(runtime_k31.__file__).parent / "runtime/vllm-kimi-k31.example.json"
    runtime = json.loads(manifest.read_text())["runtime"]
    assert runtime_k31.is_audited_runtime(runtime)
    runtime["source_files"].pop(next(iter(runtime["source_files"])))
    assert not runtime_k31.is_audited_runtime(runtime)


def _adapter(monkeypatch):
    monkeypatch.setitem(sys.modules, "fpm_memory_observer", legacy)
    return importlib.import_module("collector.fpm_forward.runtime.instrumentation.kimi_k31")


class _Tensor:
    def __init__(self, storage, offset, shape, dtype, element_size, block_stride=512):
        self.storage, self.offset, self.shape = storage, offset, shape
        self.dtype, self.size = dtype, element_size
        self.block_stride = block_stride

    def untyped_storage(self):
        return self.storage

    def element_size(self):
        return self.size

    def storage_offset(self):
        return self.offset // self.size

    def stride(self):
        result, step = [], 1
        for dim in reversed(self.shape[1:]):
            result.append(step)
            step *= dim
        return (self.block_stride // self.size, *reversed(result))


def _native_fixture(monkeypatch, *, uniform=False, scheduler_projection=False, attention_block=4):
    adapter = _adapter(monkeypatch)
    scale = attention_block // 4
    pool_page = 512 * scale
    storage = _Storage(1234, 10 * pool_page)
    layers, groups, placements = {}, [], []
    for stem, page, width in (("mla", 128 * scale, 32), ("indexer", 64 * scale, 16), ("kda", 64, None)):
        spec = _typed("vllm.v1.kv_cache_interface", "MambaSpec" if stem == "kda" else "MLAAttentionSpec")
        spec.block_size, spec.page_size_bytes = (4096 if stem == "kda" else attention_block), page
        if stem == "kda":
            spec.shapes, spec.dtypes = ((2, 4), (2, 4)), ("torch.bfloat16", "torch.float32")
            spec.state_content_size_bytes, spec.page_size_padded = 48, 64
            spec.mamba_cache_mode, spec.num_speculative_blocks = "none", 0
            spec.num_prefill_checkpoint_blocks, spec.prefill_checkpoint_alignment = 1, 64
            spec.max_num_blocks_per_req = lambda *_: 1
        else:
            spec.dtype, spec.cache_dtype_str = "torch.uint8", "fp8"
            spec.unpadded_page_size_bytes = attention_block * width
        names = [f"{stem}.{i}" for i in range(2)]
        groups.append(SimpleNamespace(layer_names=names, kv_cache_spec=spec, host_resident=False, is_eagle_group=False))
        placements.append(
            SimpleNamespace(
                size=10 * pool_page,
                layers=names,
                layer_stride=page,
                block_stride=pool_page,
                offset=0,
                host_resident=False,
            )
        )
        for i, name in enumerate(names):
            module, class_name = {
                "kda": ("vllm.models.kimi_k3.nvidia.kda", "KimiK3DeltaAttention"),
                "mla": ("vllm.model_executor.layers.attention.mla_attention", "MLAAttention"),
                "indexer": ("vllm.models.kimi_k3.common.dsa_cache", "KimiK3IndexerCache"),
            }[stem]
            layer = _typed(module, class_name)
            layer.kv_cache = (
                (
                    _Tensor(storage, i * page, (10, 2, 4), "torch.bfloat16", 2, pool_page),
                    _Tensor(storage, i * page + 16, (10, 2, 4), "torch.float32", 4, pool_page),
                )
                if stem == "kda"
                else _Tensor(storage, i * page, (10, attention_block, width), "torch.uint8", 1, pool_page)
            )
            layers[name] = layer
    if uniform:
        # Native packed grouping places MLA and indexer in the SAME token
        # block table while retaining their different per-layer page sizes.
        placements[1].offset = 256 * scale
        for name in groups[1].layer_names:
            layers[name].kv_cache.offset += 256 * scale
        wrapped = []
        for entries in (groups[:2], groups[2:]):
            spec = _typed("vllm.v1.kv_cache_interface", "UniformTypeKVCacheSpecs")
            spec.kv_cache_specs = {name: entry.kv_cache_spec for entry in entries for name in entry.layer_names}
            spec.block_size = entries[0].kv_cache_spec.block_size
            spec.page_size_bytes = sum(child.page_size_bytes for child in spec.kv_cache_specs.values())
            wrapped.append(
                SimpleNamespace(
                    layer_names=list(spec.kv_cache_specs), kv_cache_spec=spec, host_resident=False, is_eagle_group=False
                )
            )
        groups = wrapped
    config = SimpleNamespace(num_blocks=10, kv_cache_groups=groups, kv_cache_tensors=placements)
    vllm = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=4096, hf_config=SimpleNamespace(architectures=["KimiK31ForConditionalGeneration"])
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False, mamba_cache_mode="none"),
        speculative_config=None,
        compilation_config=SimpleNamespace(static_forward_context=layers),
    )
    worker = SimpleNamespace(
        vllm_config=vllm,
        _fpm_cache_initialized=True,
        _fpm_available_cache_bytes=10 * pool_page + 80,
        model_runner=SimpleNamespace(kv_cache_config=config, kv_caches=[x.kv_cache for x in layers.values()]),
    )
    pool = SimpleNamespace(
        num_gpu_blocks=10, get_num_free_blocks=lambda: 9, null_block=SimpleNamespace(is_null=True, block_id=0)
    )
    scheduler = SimpleNamespace(
        vllm_config=vllm,
        kv_cache_manager=SimpleNamespace(
            block_pool=pool,
            watermark_blocks=0,
            coordinator=SimpleNamespace(single_type_managers=[SimpleNamespace(block_pool=pool) for _ in groups]),
        ),
    )
    scheduler_config = copy.deepcopy(config)
    if scheduler_projection:
        for group in scheduler_config.kv_cache_groups:
            group.kv_cache_spec = next(iter(group.kv_cache_spec.kv_cache_specs.values()))
    cache, scheduled = adapter.worker_memory(worker), adapter.scheduler_memory(scheduler, scheduler_config)
    for record in (cache, scheduled):
        record["pool_id"] = "pool"
        for group in record["groups"]:
            group.update(pool_id="pool", extra_retained_tokens=0, attention_chunk_size=None)
    cache["semantics"] = {**_SEMANTICS, "retention": "full_or_state"}
    scheduled.update(group_pool_ids=["pool"] * len(groups), permanent_reserved_block_ids=[0])
    return cache, scheduled, worker


def test_native_mixed_state_and_overlaid_groups_charge_actual_padded_pool(monkeypatch):
    cache, scheduled, _ = _native_fixture(monkeypatch)
    groups, accounting = _physical_views(cache, scheduled, "fp8", context=4096)
    assert accounting["page_size_bytes"] == 512
    assert [group["kind"] for group in groups] == ["attention", "attention", "state"]
    assert groups[-1]["blocks_per_request"] == 1
    assert groups[-1]["page_size_bytes"] == 512
    assert [v["dtype"] for v in cache["layer_components"]["kda.0"]] == ["torch.bfloat16", "torch.float32"]
    assert cache["allocated_cache_bytes"] == 5120
    assert cache["groups"][-1]["state_content_size_bytes"] == 48
    assert cache["groups"][-1]["spec_page_size_bytes"] == 64


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("truncated", [False, True])
def test_direct_attention_views_cover_native_payload_with_optional_page_padding(monkeypatch, padded, truncated):
    cache, scheduled, worker = _native_fixture(monkeypatch)
    spec = worker.model_runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec
    spec.unpadded_page_size_bytes = 96 if padded else 128
    for name in ("mla.0", "mla.1"):
        tensor = worker.vllm_config.compilation_config.static_forward_context[name].kv_cache
        tensor.shape = (10, 4, 1 if truncated else spec.unpadded_page_size_bytes // 4)
    cache = _adapter(monkeypatch).worker_memory(worker)
    cache["pool_id"] = "pool"
    cache["semantics"] = {**_SEMANTICS, "retention": "full_or_state"}
    for group in cache["groups"]:
        group.update(pool_id="pool", extra_retained_tokens=0, attention_chunk_size=None)
    scheduled["groups"][0].update(unpadded_page_size_bytes=spec.unpadded_page_size_bytes)
    if truncated:
        with pytest.raises(ValueError, match="attention.*payload"):
            _physical_views(cache, scheduled, "fp8", context=4096)
    else:
        _, accounting = _physical_views(cache, scheduled, "fp8", context=4096)
        assert cache["groups"][0]["unpadded_page_size_bytes"] == spec.unpadded_page_size_bytes
        assert cache["groups"][0]["spec_page_size_bytes"] == 128
        assert accounting["page_size_bytes"] == 512
        assert cache["allocated_cache_bytes"] == 5120


@pytest.mark.parametrize("elements", [3, 4])
def test_mixed_state_component_offsets_must_be_element_aligned(monkeypatch, elements):
    cache, scheduled, _ = _native_fixture(monkeypatch)
    for record in (cache, scheduled):
        state = record["groups"][-1]
        state["state_shapes"][0] = [elements]
        state["state_content_size_bytes"] = elements * 2 + 32
    for name in cache["groups"][-1]["layer_names"]:
        convolution, recurrence = cache["layer_components"][name]
        convolution.update(shape=[10, elements], stride_bytes=[512, 2])
        recurrence["storage_offset_bytes"] = convolution["storage_offset_bytes"] + elements * 2
    if elements == 3:
        with pytest.raises(ValueError, match="offsets"):
            _physical_views(cache, scheduled, "fp8", context=4096)
    else:
        assert _physical_views(cache, scheduled, "fp8", context=4096)[0][-1]["blocks_per_request"] == 1


@pytest.mark.parametrize("omit", ["state", "attention"])
def test_complete_native_layout_requires_both_attention_and_kda_state(monkeypatch, omit):
    cache, scheduled, _ = _native_fixture(monkeypatch)
    omitted = {name for group in cache["groups"] if group["kind"] == omit for name in group["layer_names"]}
    for record in (cache, scheduled):
        record["groups"] = [group for group in record["groups"] if not omitted.intersection(group["layer_names"])]
    scheduled["group_pool_ids"] = ["pool"] * len(scheduled["groups"])
    cache["tensor_allocations"] = [a for a in cache["tensor_allocations"] if not omitted.intersection(a["layers"])]
    for name in omitted:
        del cache["layer_components"][name]
    with pytest.raises(ValueError, match="requires KDA state and attention"):
        _physical_views(cache, scheduled, "fp8", context=4096)


def test_native_uniform_wrapper_preserves_one_attention_block_table(monkeypatch):
    cache, scheduled, _ = _native_fixture(monkeypatch, uniform=True)
    groups, accounting = _physical_views(cache, scheduled, "fp8", context=4096)
    assert len(groups) == 2
    assert groups[0]["kind"] == "attention" and groups[0]["num_layers"] == 4
    assert groups[1]["kind"] == "state" and groups[1]["blocks_per_request"] == 1
    assert accounting["page_size_bytes"] == 512
    assert cache["groups"][0]["spec_page_size_bytes"] == 384
    assert cache["groups"][0]["layer_specs"]["indexer.0"]["spec_page_size_bytes"] == 64
    cache["tensor_allocations"][1]["offset"] = 128
    for name in ("indexer.0", "indexer.1"):
        cache["layer_components"][name][0]["storage_offset_bytes"] -= 128
    with pytest.raises(ValueError, match="within one group overlap"):
        _physical_views(cache, scheduled, "fp8", context=4096)


def test_raw_uniform_wrapper_reads_children_without_abstract_scalar_property(monkeypatch):
    _, _, worker = _native_fixture(monkeypatch, uniform=True)
    observer = importlib.import_module("collector.fpm_forward.runtime.instrumentation.observer")
    config = worker.model_runner.kv_cache_config

    def abstract_state_content(_):
        raise NotImplementedError

    for group in config.kv_cache_groups:
        monkeypatch.setattr(
            type(group.kv_cache_spec), "state_content_size_bytes", property(abstract_state_content), raising=False
        )
    with pytest.raises(NotImplementedError):
        observer._raw_cache(config)
    raw = observer._raw_cache(config, kimi_k31=True)
    assert raw["num_blocks"] == 10
    assert raw["groups"][0]["page_size_bytes"] == 384
    children = raw["groups"][0]["kv_cache_specs"]
    assert set(children) == set(raw["groups"][0]["layer_names"])
    assert children["mla.0"]["page_size_bytes"] == 128
    assert children["indexer.0"]["page_size_bytes"] == 64
    state = raw["groups"][1]["kv_cache_specs"]["kda.0"]
    assert state["state_content_size_bytes"] == 48
    assert state["dtypes"] == ["torch.bfloat16", "torch.float32"]
    assert raw["tensor_allocations"][1]["offset"] == 256
    child = config.kv_cache_groups[1].kv_cache_spec.kv_cache_specs["kda.0"]
    del child.state_content_size_bytes
    monkeypatch.setattr(type(child), "state_content_size_bytes", property(abstract_state_content), raising=False)
    with pytest.raises(NotImplementedError):
        observer._raw_cache(config, kimi_k31=True)


def test_scheduler_projection_uses_observed_representative_not_serialized_key_order(monkeypatch):
    cache, scheduled, _ = _native_fixture(monkeypatch, uniform=True, scheduler_projection=True)
    cache = json.loads(json.dumps(cache, sort_keys=True))
    assert next(iter(cache["groups"][0]["layer_specs"])) == "indexer.0"
    assert cache["groups"][0]["representative_layer"] == "mla.0"
    groups, accounting = _physical_views(cache, scheduled, "fp8", context=4096)
    assert len(groups) == 2 and groups[0]["num_layers"] == 4
    assert accounting["page_size_bytes"] == 512


@pytest.mark.parametrize("field", ["layer_names", "pool_id", "kind", "layer_classes"])
def test_scheduler_projection_rejects_child_group_metadata(monkeypatch, field):
    cache, scheduled, _ = _native_fixture(monkeypatch, uniform=True, scheduler_projection=True)
    group = cache["groups"][0]
    value = ["mla.0"] if field == "layer_names" else group[field]
    group["layer_specs"][group["representative_layer"]][field] = value
    if field == "layer_names":
        scheduled["groups"][0][field] = value
    with pytest.raises(ValueError, match="group metadata"):
        _physical_views(cache, scheduled, "fp8", context=4096)


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("block,declared", [(32, 32), (32, 4096), (4096, 32)])
def test_state_page_must_cover_reviewed_context(monkeypatch, uniform, block, declared):
    cache, scheduled, _ = _native_fixture(monkeypatch, uniform=uniform)
    for record in (cache, scheduled):
        state = record["groups"][-1]
        state["block_size_tokens"] = block
        for spec in state.get("layer_specs", {"state": state}).values():
            spec.update(block_size_tokens=block, max_model_len=declared)
    with pytest.raises(ValueError, match="one non-speculative state page"):
        _physical_views(cache, scheduled, "fp8", context=4096)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda c, s: c["groups"][0].pop("representative_layer"), "observed representative"),
        (lambda c, s: c["groups"][0].update(representative_layer="indexer.0"), "geometry disagrees"),
        (lambda c, s: s["groups"][0].update(spec_page_size_bytes=129), "geometry disagrees"),
        (lambda c, s: s["groups"][0]["layer_names"].pop(), "geometry disagrees"),
        (lambda c, s: c["groups"][0]["layer_specs"]["indexer.0"].update(block_size_tokens=8), "incompatible per-layer"),
    ],
)
def test_scheduler_projection_retains_exact_native_geometry_checks(monkeypatch, mutation, match):
    cache, scheduled, _ = _native_fixture(monkeypatch, uniform=True, scheduler_projection=True)
    mutation(cache, scheduled)
    with pytest.raises(ValueError, match=match):
        _physical_views(cache, scheduled, "fp8", context=4096)


def _kernel_view_fixture(monkeypatch):
    cache, scheduled, _ = _native_fixture(monkeypatch, uniform=True, scheduler_projection=True, attention_block=128)
    cache["kernel_block_views"] = {}
    for name in ("indexer.0", "indexer.1"):
        cache["layer_components"][name][0].update(shape=[146, 64, 16], stride_bytes=[1024, 16, 1])
        cache["kernel_block_views"][name] = {
            "attention_group_class": "vllm.v1.worker.utils.AttentionGroup",
            "backend_class": "vllm.models.kimi_k3.common.dsa_cache.KimiK3IndexerBackend",
            "kernel_block_size": 64,
            "kernel_block_stride": 16,
            "kernel_block_offsets": [0, 1],
            # CP1: ceil(4096 / 128) manager entries, aligned to 128
            # tokens, then two kernel entries per manager block.
            "kernel_block_table_shape": [1, 8, 64],
            "kernel_block_table_dtype": "torch.int32",
            "maps_slot_mapping": False,
            "writes_kv_cache_from_block_table": True,
        }
    return cache, scheduled


def test_native_indexer_kernel_view_charges_only_actual_shared_allocation(monkeypatch):
    cache, scheduled = _kernel_view_fixture(monkeypatch)
    groups, accounting = _physical_views(cache, scheduled, "fp8", context=4096, sequences=8)
    assert len(groups) == 2
    assert accounting["num_blocks"] == 10 and accounting["page_size_bytes"] == 16384
    assert cache["allocated_cache_bytes"] == 163840
    assert cache["layer_components"]["indexer.0"][0]["shape"][0] == 146


@pytest.mark.parametrize("payload,padded", [(1920, 2048), (2048, 4096)])
def test_strided_indexer_views_cover_unpadded_payload_and_keep_pool_padding(monkeypatch, payload, padded):
    cache, scheduled = _kernel_view_fixture(monkeypatch)
    # Each indexer addresses 128 * 16 = 2048 bytes. Its padded footprint
    # may include padding before the next layer in this pool.
    group = cache["groups"][0]
    for name in ("indexer.0", "indexer.1"):
        group["layer_specs"][name].update(spec_page_size_bytes=padded, unpadded_page_size_bytes=payload)
    group["spec_page_size_bytes"] = sum(child["spec_page_size_bytes"] for child in group["layer_specs"].values())
    cache["tensor_allocations"][1]["layer_stride"] = padded
    cache["layer_components"]["indexer.1"][0]["storage_offset_bytes"] = 8192 + padded
    if payload == 2048:
        _, accounting = _physical_views(cache, scheduled, "fp8", context=4096, sequences=8)
        assert accounting["page_size_bytes"] == 16384
        assert cache["allocated_cache_bytes"] == 10 * 16384
    else:
        with pytest.raises(ValueError, match="layer page coverage"):
            _physical_views(cache, scheduled, "fp8", context=4096, sequences=8)


def test_native_kernel_table_width_includes_context_alignment(monkeypatch):
    cache, scheduled = _kernel_view_fixture(monkeypatch)
    for record in (cache, scheduled):
        state = record["groups"][-1]
        for spec in state.get("layer_specs", dict.fromkeys(state["layer_names"], state)).values():
            spec.update(block_size_tokens=4097, max_model_len=4097)
        state["block_size_tokens"] = 4097
    # 4097 tokens need 33 manager blocks (already 128-token aligned),
    # hence 66 entries in the 64-token kernel table.
    for mapping in cache["kernel_block_views"].values():
        mapping["kernel_block_table_shape"][-1] = 66
    assert _physical_views(cache, scheduled, "fp8", context=4097, sequences=8)[1]["num_blocks"] == 10
    cache["kernel_block_views"]["indexer.0"]["kernel_block_table_shape"][-1] = 65
    with pytest.raises(ValueError, match="kernel-block table"):
        _physical_views(cache, scheduled, "fp8", context=4097, sequences=8)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda c: c["kernel_block_views"].pop("indexer.0"), "coverage"),
        (lambda c: c["kernel_block_views"]["indexer.0"].update(kernel_block_stride=15), "stride"),
        (lambda c: c["kernel_block_views"]["indexer.0"].update(kernel_block_size=3), "subdivision"),
        (lambda c: c["kernel_block_views"]["indexer.0"].update(kernel_block_offsets=[0, 2]), "offsets"),
        (lambda c: c["kernel_block_views"]["indexer.0"].update(kernel_block_table_dtype="torch.int64"), "table"),
        (lambda c: c["kernel_block_views"]["indexer.0"].update(kernel_block_table_shape=[1, 8, 31]), "table"),
        (lambda c: c["kernel_block_views"]["indexer.0"].update(backend_class="other.Backend"), "address mapping"),
        (
            lambda c: c["kernel_block_views"]["indexer.0"].update(writes_kv_cache_from_block_table=False),
            "address mapping",
        ),
        (lambda c: c["kernel_block_views"].update({"unknown.layer": {}}), "layer coverage"),
        (lambda c: c["layer_components"]["indexer.0"][0].update(shape=[145, 64, 16]), "coverage"),
        (lambda c: c["layer_components"]["indexer.0"][0].update(stride_bytes=[31, 16, 1]), "stride"),
        (lambda c: c["layer_components"]["indexer.0"][0].update(storage_offset_bytes=8193), "offsets"),
        (lambda c: c["tensor_allocations"][1].update(layer_stride=32), "overlap"),
    ],
)
def test_kernel_view_mapping_cannot_hide_pool_or_layer_corruption(monkeypatch, mutation, match):
    cache, scheduled = _kernel_view_fixture(monkeypatch)
    mutation(cache)
    with pytest.raises(ValueError, match=match):
        _physical_views(cache, scheduled, "fp8", context=4096, sequences=8)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda g: g["layer_specs"].pop("indexer.0"), "every native per-layer"),
        (lambda g: g.update(spec_page_size_bytes=385), "page sum"),
        (lambda g: g["layer_specs"]["indexer.0"].update(block_size_tokens=8), "incompatible per-layer"),
    ],
)
def test_native_uniform_wrapper_requires_exact_child_geometry(monkeypatch, mutation, match):
    cache, scheduled, _ = _native_fixture(monkeypatch, uniform=True)
    for record in (cache, scheduled):
        mutation(record["groups"][0])
    with pytest.raises(ValueError, match=match):
        _physical_views(cache, scheduled, "fp8", context=4096)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda c, s: c["layer_components"]["kda.0"].pop(), "component count"),
        (lambda c, s: c["layer_components"]["kda.0"][1].update(dtype="torch.uint8"), "shape/dtype"),
        (lambda c, s: c["layer_components"]["kda.0"][1].update(storage_offset_bytes=0), "offsets"),
        (lambda c, s: c["layer_components"]["kda.0"][0]["stride_bytes"].__setitem__(0, 256), "coverage"),
        (lambda c, s: c["tensor_allocations"][-1].update(layer_stride=32), "overlap"),
        (lambda c, s: c["tensor_allocations"][-1].update(block_stride=64), "block-outer"),
        (lambda c, s: c["tensor_allocations"][-1].update(host_resident=True), "device"),
        (lambda c, s: c["tensor_allocations"].pop(), "omit"),
        (lambda c, s: c["layer_components"]["mla.0"][0]["shape"].__setitem__(0, 20), "coverage"),
        (lambda c, s: s["groups"][-1].update(mamba_cache_mode="align"), "geometry"),
    ],
)
def test_native_component_or_allocation_corruption_is_rejected(monkeypatch, mutation, match):
    cache, scheduled, _ = _native_fixture(monkeypatch)
    mutation(cache, scheduled)
    with pytest.raises(ValueError, match=match):
        _physical_views(cache, scheduled, "fp8", context=4096)


def test_uniform_attention_child_cannot_override_parent_kind_with_state_components(monkeypatch):
    cache, scheduled, _ = _native_fixture(monkeypatch, uniform=True)
    child = cache["groups"][0]["layer_specs"]["mla.0"]
    child.update(kind="state", state_shapes=[[4, 8], [4, 4]], state_dtypes=["torch.bfloat16", "torch.float32"])
    scheduled["groups"][0]["layer_specs"]["mla.0"].update(copy.deepcopy(child))
    component = cache["layer_components"]["mla.0"][0]
    convolution = {
        **component,
        "shape": [10, 4, 8],
        "stride_bytes": [512, 16, 2],
        "dtype": "torch.bfloat16",
        "element_size_bytes": 2,
    }
    recurrence = {
        **component,
        "shape": [10, 4, 4],
        "stride_bytes": [512, 16, 4],
        "dtype": "torch.float32",
        "element_size_bytes": 4,
        "storage_offset_bytes": 64,
    }
    cache["layer_components"]["mla.0"] = [convolution, recurrence]
    with pytest.raises(ValueError, match="group metadata"):
        _physical_views(cache, scheduled, "fp8", context=4096)


def test_nonuniform_group_cannot_override_its_geometry_with_child_specs(monkeypatch):
    cache, scheduled, _ = _native_fixture(monkeypatch)
    for record in (cache, scheduled):
        record["groups"][0]["layer_specs"] = {"mla.0": {"block_size_tokens": 1}}
    cache["layer_components"]["mla.0"][0]["shape"][1] = 1
    with pytest.raises(ValueError, match="per-layer specs require a native uniform group"):
        _physical_views(cache, scheduled, "fp8", context=4096)
    with pytest.raises(ValueError, match="per-layer specs require a native uniform group"):
        runtime_k31.validate_views(cache, "fp8", 10, 512)


def test_kernel_offsets_reject_huge_geometry_without_allocating_from_it(monkeypatch):
    cache, _ = _kernel_view_fixture(monkeypatch)
    mapping = cache["kernel_block_views"]["indexer.0"]
    # Prevent a regression from attempting a multi-petabyte list in the test.
    monkeypatch.setattr(
        runtime_k31, "range", lambda *_: pytest.fail("range allocation from untrusted geometry"), raising=False
    )
    with pytest.raises(ValueError, match="kernel-block offsets"):
        runtime_k31._kernel_view_geometry(mapping, {"block_size_tokens": 2**53}, {}, 10, 512, 64)


@pytest.mark.parametrize("mutation", ["none", "out_of_bounds", "overlap"])
def test_legacy_resource_resolver_cannot_accept_native_k31_layouts(tmp_path, monkeypatch, mutation):
    from .test_fpm_runtime_memory import _edit_memory, _evidence, _resolve

    monkeypatch.setattr(legacy, "_offloader_type", lambda: "vllm.model_executor.offloader.base.NoopOffloader")
    cell = replace(_evidence(tmp_path, monkeypatch), kv_cache_dtype="fp8")
    cache, scheduled, _ = _native_fixture(monkeypatch)
    if mutation == "out_of_bounds":
        cache["tensor_allocations"][0]["offset"] = 512
    elif mutation == "overlap":
        cache["tensor_allocations"][0]["layer_stride"] = 64

    def edit(record):
        record["cache"] = copy.deepcopy(cache if record["kind"] == "worker" else scheduled)
        record["resolved_config"]["cache_config"]["cache_dtype"] = "fp8"

    _edit_memory(tmp_path, edit)
    if mutation != "none":
        match = "exceeds its pool block slot" if mutation == "out_of_bounds" else "within one group overlap"
        with pytest.raises(ValueError, match=match):
            _physical_views(cache, scheduled, "fp8", context=4096)
    with pytest.raises(ValueError, match="require the source-bound runtime observation importer"):
        _resolve(cell, tmp_path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("mamba_cache_mode", "align"),
        ("num_speculative_blocks", 1),
        ("blocks_per_request", 2),
        ("block_size_tokens", 128),
    ],
)
def test_native_state_mode_cannot_be_relabelled_fixed_one_page(monkeypatch, key, value):
    cache, scheduled, _ = _native_fixture(monkeypatch)
    for record in (cache, scheduled):
        record["groups"][-1][key] = value
    with pytest.raises(ValueError, match="one non-speculative state page"):
        _physical_views(cache, scheduled, "fp8", context=4096)


def _moe(*, indices=(1,)):
    evidence = {
        "enumeration": "model.named_modules:RoutedExperts",
        "expected_layer_names": ["model.layers.1.experts"],
        "layers": [
            {
                "name": "model.layers.1.experts",
                "layer_type": "vllm.model_executor.layers.fused_moe.routed_experts.RoutedExperts",
                "method_type": "vllm.model_executor.layers.quantization.mxfp4.Mxfp4MoEMethod",
                "kernel_type": "vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEKernel",
                "backend": "FLASHINFER_TRTLLM_MXFP4_MXFP8",
                "descriptors": {
                    x: {"dtype": d} for x, d in (("_a1", "mxfp8"), ("_a2", "mxfp8"), ("_w1", "mxfp4"), ("_w2", "mxfp4"))
                },
                "weight_shapes": {"w13_weight": [8, 64, 32], "w2_weight": [8, 32, 64]},
            }
        ],
    }
    layer = evidence["layers"][0]
    evidence["layers"] = [{**copy.deepcopy(layer), "name": f"model.layers.{index}.experts"} for index in indices]
    evidence["expected_layer_names"] = [item["name"] for item in evidence["layers"]]
    return evidence


def _decoder_cache_names(value):
    """Give synthetic native views the decoder ownership in our four-layer config."""
    names = {
        **{f"kda.{i}": f"model.layers.{i}.self_attn" for i in range(2)},
        **{f"mla.{i}": f"model.layers.{i + 2}.self_attn.attn" for i in range(2)},
        **{f"indexer.{i}": f"model.layers.{i + 2}.self_attn.indexer.k_cache" for i in range(2)},
    }
    if isinstance(value, str):
        return names.get(value, value)
    if isinstance(value, list):
        return [_decoder_cache_names(item) for item in value]
    if isinstance(value, dict):
        return {_decoder_cache_names(key): _decoder_cache_names(item) for key, item in value.items()}
    return value


@pytest.mark.parametrize("wrapped", [False, True])
def test_initialized_moe_coverage_matches_independent_config_inventory(wrapped):
    config = {"num_hidden_layers": 7, "num_experts": 8, "first_k_dense_replace": 2, "moe_layer_freq": 2}
    if wrapped:
        config = {"text_config": config}
    complete = _moe(indices=(2, 4, 6))
    assert runtime_k31.moe_precision(complete, model_config=config) == "w4a8_mxfp4_mxfp8_trtllm"
    # Dropping the layer from both worker-generated lists used to pass.
    for indices in ((2, 6), (0, 2, 4, 6), (2, 4, 5), (1, 3, 5)):
        with pytest.raises(ValueError, match="model configuration inventory"):
            runtime_k31.moe_precision(_moe(indices=indices), model_config=config)


def test_initialized_moe_inventory_uses_native_config_defaults():
    config = {"num_hidden_layers": 3, "num_experts": 8}
    assert runtime_k31.moe_precision(_moe(indices=(0, 1, 2)), model_config=config) == "w4a8_mxfp4_mxfp8_trtllm"
    with pytest.raises(ValueError, match="model configuration inventory"):
        runtime_k31.moe_precision(_moe(indices=(1, 2)), model_config=config)


@pytest.mark.parametrize("first,frequency,indices", [(0, 3, (0, 3, 6)), (2, 3, (3, 6)), (6, 3, (6,))])
def test_moe_inventory_count_includes_only_eligible_decoder_indices(first, frequency, indices):
    config = {"num_hidden_layers": 7, "num_experts": 8, "first_k_dense_replace": first, "moe_layer_freq": frequency}
    assert runtime_k31.moe_precision(_moe(indices=indices), model_config=config) == "w4a8_mxfp4_mxfp8_trtllm"


def test_moe_inventory_rejects_huge_declared_layer_count_without_enumeration(monkeypatch):
    def forbidden_range(*args):
        raise AssertionError("inventory validation enumerated the declared decoder range")

    monkeypatch.setattr(runtime_k31, "range", forbidden_range, raising=False)
    config = {"num_hidden_layers": 2**53, "num_experts": 8}
    with pytest.raises(ValueError, match="model configuration inventory"):
        runtime_k31.moe_precision(_moe(), model_config=config)


def test_initialized_expert_arithmetic_selects_exact_mode_and_rejects_metadata_guess():
    evidence = _moe()
    assert runtime_k31.moe_precision(evidence) == "w4a8_mxfp4_mxfp8_trtllm"
    config = {
        "model_config": {
            "dtype": "bfloat16",
            "quantization": "mxfp4",
            "architectures": ["KimiK31ForConditionalGeneration"],
        },
        "quantization_config": {"type": "vllm.model_executor.layers.quantization.mxfp4.Mxfp4Config"},
    }
    cell = replace(_cell(), gemm_quant_mode="bfloat16", moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm")
    runtime_memory._validate_precision(config, cell, evidence=evidence)
    with pytest.raises(ValueError, match="initialized MoE backend"):
        runtime_memory._validate_precision(config, cell)
    for mode in ("w4a16_mxfp4", "w4a8_mxfp4_mxfp8", "bfloat16"):
        with pytest.raises(ValueError, match="planned GEMM/MoE"):
            runtime_memory._validate_precision(config, replace(cell, moe_quant_mode=mode), evidence=evidence)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda e: e["layers"][0]["descriptors"]["_a2"].update(dtype="None"), "activation/weight"),
        (lambda e: e["layers"][0]["descriptors"]["_w2"].update(dtype="nvfp4"), "activation/weight"),
        (lambda e: e["layers"][0].update(backend="FLASHINFER_TRTLLM_MXFP4_BF16"), "backend"),
        (lambda e: e["layers"][0].update(kernel_type="NoneType"), "method/kernel"),
        (lambda e: e["expected_layer_names"].append("unobserved.experts"), "coverage"),
        (lambda e: e["layers"].append(copy.deepcopy(e["layers"][0])), "coverage"),
    ],
)
def test_initialized_precision_is_not_inferred_from_model_dtype(mutation, match):
    evidence = _moe()
    mutation(evidence)
    with pytest.raises(ValueError, match=match):
        runtime_k31.moe_precision(evidence)


@pytest.mark.parametrize("name", [None, "", " ", 42, [], {}])
@pytest.mark.parametrize("target", ["both", "expected", "observed"])
def test_initialized_moe_requires_actual_layer_names(name, target):
    evidence = _moe()
    if target in {"both", "expected"}:
        evidence["expected_layer_names"] = [name]
    if target in {"both", "observed"}:
        evidence["layers"][0]["name"] = name
    with pytest.raises(ValueError, match="layer names must be nonempty strings"):
        runtime_k31.moe_precision(evidence)


def test_preview_bundle_uses_exact_source_map_not_invented_git_commit():
    path = Path(runtime_k31.__file__).parent / "runtime/vllm-kimi-k31.example.json"
    bundle = load_instrumentation(path)
    assert runtime_k31.is_audited_runtime(bundle.manifest["runtime"])
    altered = copy.deepcopy(bundle.manifest["runtime"])
    altered["source_files"]["vllm/v1/kv_cache_interface.py"] = "0" * 64
    assert not runtime_k31.is_audited_runtime(altered)
    assert "instrumentation/kimi_k31.py" in bundle.files
    assert json.loads(path.read_text())["runtime"]["version"] == runtime_k31.VERSION


def test_native_expert_enumeration_requires_every_initialized_kimi_moe(monkeypatch):
    adapter = _adapter(monkeypatch)
    routed = _typed("vllm.model_executor.layers.fused_moe.routed_experts", "RoutedExperts")
    kimi = _typed("vllm.models.kimi_k3.nvidia.model", "KimiMoE")
    method = _typed("vllm.model_executor.layers.quantization.mxfp4", "Mxfp4MoEMethod")
    method.moe_kernel = _typed("vllm.model_executor.layers.fused_moe.modular_kernel", "FusedMoEKernel")
    method.mxfp4_backend = SimpleNamespace(name="FLASHINFER_TRTLLM_MXFP4_MXFP8")
    method.moe_quant_config = SimpleNamespace(
        **{
            key: SimpleNamespace(dtype=dtype)
            for key, dtype in (("_a1", "mxfp8"), ("_a2", "mxfp8"), ("_w1", "mxfp4"), ("_w2", "mxfp4"))
        }
    )
    routed.quant_method = method
    routed.w13_weight, routed.w2_weight = SimpleNamespace(shape=(8, 64, 32)), SimpleNamespace(shape=(8, 32, 64))
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.fused_moe.routed_experts", SimpleNamespace(RoutedExperts=type(routed))
    )
    monkeypatch.setitem(sys.modules, "vllm.models.kimi_k3.nvidia.model", SimpleNamespace(KimiMoE=type(kimi)))
    modules = {"layer.1.moe": kimi, "layer.1.moe.experts": routed}
    owner = SimpleNamespace(model_runner=SimpleNamespace(model=SimpleNamespace(named_modules=lambda: modules.items())))
    evidence = adapter.initialized_moe(owner)
    assert evidence["expected_layer_names"] == ["layer.1.moe.experts"]
    assert runtime_k31.moe_precision(evidence) == "w4a8_mxfp4_mxfp8_trtllm"
    method.moe_quant_config._a2.dtype = None
    with pytest.raises(ValueError, match="activation/weight"):
        runtime_k31.moe_precision(adapter.initialized_moe(owner))
    modules["layer.2.moe"] = type(kimi)()
    with pytest.raises(ValueError, match="exactly one audited RoutedExperts"):
        adapter.initialized_moe(owner)


@pytest.mark.parametrize("setting,value", [("enable_prefix_caching", True), ("mamba_cache_mode", "align")])
def test_observer_preserves_unsupported_runtime_mode_as_failure(monkeypatch, setting, value):
    adapter = _adapter(monkeypatch)
    _, _, worker = _native_fixture(monkeypatch)
    setattr(worker.vllm_config.cache_config, setting, value)
    with pytest.raises(ValueError, match="prefix caching off"):
        adapter.worker_memory(worker)


def test_preview_source_digest_checks_actual_files_and_wheel_commit(monkeypatch, tmp_path):
    _adapter(monkeypatch)
    observer = importlib.import_module("collector.fpm_forward.runtime.instrumentation.observer")
    path = tmp_path / "source.py"
    path.write_text("# synthetic source identity fixture\n")
    source_map = {"vllm/source.py": hashlib.sha256(path.read_bytes()).hexdigest()}
    source_revision = hashlib.sha256(json.dumps(source_map, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {
        "runtime": {"version": runtime_k31.VERSION, "source_revision": source_revision, "source_files": source_map}
    }
    monkeypatch.setattr(legacy.importlib.metadata, "version", lambda _: runtime_k31.VERSION)
    monkeypatch.setattr(legacy, "_installed_file", lambda _: path)
    real_import = legacy.importlib.import_module
    build = SimpleNamespace(__commit_id__="g266e88c05")
    monkeypatch.setattr(
        legacy.importlib, "import_module", lambda name: build if name == "vllm._version" else real_import(name)
    )
    payload = {"launch": {"deployment": {"image": "fixture@sha256:" + "a" * 64}}}
    observer._runtime(payload, manifest, runtime_k31.VERSION)
    assert payload["source_identity"]["vllm_commit_id"] == "g266e88c05"
    assert payload["runtime"]["source_files"] == source_map
    path.write_text("# changed after source audit\n")
    with pytest.raises(ValueError, match="source hash differs"):
        observer._runtime(payload, manifest, runtime_k31.VERSION)
    build.__commit_id__ = "g99999999"
    with pytest.raises(ValueError, match="VCS identifier contradicts"):
        observer._runtime(payload, manifest, runtime_k31.VERSION)


def _complete_fixture(tmp_path, monkeypatch, *, workload_kind="decode", packed_indexer=False):
    path, launches = observation_fixture(tmp_path, version=runtime_k31.VERSION)
    launch = launches["tp2"]
    model_path = Path(launch["model_config"]["path"])
    model_path.write_text(
        json.dumps(
            {
                "architectures": ["KimiK31ForConditionalGeneration"],
                "text_config": {
                    "num_hidden_layers": 4,
                    "num_experts": 8,
                    "first_k_dense_replace": 1,
                    "linear_attn_config": {
                        "kda_layers": [1, 2],
                        "full_attn_layers": [3, 4],
                        "use_full_rank_gate": True,
                    },
                    "dsa_config": {"dsa_layers": [3, 4]},
                },
            }
        )
    )
    launch["model_config"]["sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    launch["worker_type"] = workload_kind
    launch["precision"].update(
        gemm_quant_mode="bfloat16",
        moe_quant_mode="w4a8_mxfp4_mxfp8_trtllm",
        kvcache_quant_mode="fp8",
        attention_backend="trtllm_mla",
    )
    _replace_launch(path, launches)
    manifest_path = tmp_path / "manifest.yaml"
    manifest = json.loads(manifest_path.read_text())
    pinned = json.loads((Path(runtime_k31.__file__).parent / "runtime/vllm-kimi-k31.example.json").read_text())[
        "runtime"
    ]
    manifest["runtime"] = pinned
    manifest_path.write_text(json.dumps(manifest))
    bundle = load_instrumentation(manifest_path)
    if packed_indexer:
        cache, scheduled = _kernel_view_fixture(monkeypatch)
        for mapping in cache["kernel_block_views"].values():
            mapping["kernel_block_table_shape"][1] = launch["collection"]["max_num_seqs"]
    else:
        cache, scheduled, _ = _native_fixture(monkeypatch, uniform=True)
    index = json.loads(path.read_text())
    attempt = index["configurations"]["tp2"]["attempts"][0]
    attempt["bundle"]["sha256"] = bundle.sha256
    attempt["phases"] = {workload_kind: attempt["phases"][workload_kind]}
    phase = attempt["phases"][workload_kind]
    context_path = tmp_path / phase["launch_manifest"]["path"]
    context = json.loads(context_path.read_text())
    context["bundle_sha256"] = bundle.sha256
    context_path.write_text(json.dumps(context))
    phase["launch_manifest"]["sha256"] = hashlib.sha256(context_path.read_bytes()).hexdigest()
    for ref in phase["artifacts"]:
        record_path = tmp_path / ref["path"]
        record = json.loads(record_path.read_text())
        record["bundle_sha256"] = bundle.sha256
        record["model_config_sha256"] = launch["model_config"]["sha256"]
        record["runtime"] = {**pinned, "image": launch["deployment"]["image"]}
        config = record["resolved_config"]
        config["parallel_config"].update(
            data_parallel_size_local=launch["topology"]["dp"], data_parallel_external_lb=False, enable_eplb=False
        )
        config["model_config"].update(
            dtype="bfloat16", quantization="mxfp4", architectures=["KimiK31ForConditionalGeneration"]
        )
        config["quantization_config"] = {"type": "vllm.model_executor.layers.quantization.mxfp4.Mxfp4Config"}
        config["cache_config"].update(
            cache_dtype="fp8",
            mamba_cache_mode="none",
            enable_prefix_caching=False,
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="auto",
        )
        config["attention_config"] = {"backend": "FLASHINFER_MLA_SPARSE", "indexer_kv_dtype": "fp8"}
        record["cache"] = copy.deepcopy(cache if record["kind"] == "worker" else scheduled)
        if record["kind"] == "worker":
            record["initialized_moe"] = _moe(indices=(1, 2, 3))
            record["attention_backend"] = ["FLASHINFER_MLA_SPARSE", "KIMI_K3_KDA", "DEEPSEEK_V32_INDEXER"]
            record["attention_groups"] = [
                {"group_id": 0, "backend": "FLASHINFER_MLA_SPARSE", "layer_names": ["mla.0", "mla.1"]},
                {"group_id": 0, "backend": "DEEPSEEK_V32_INDEXER", "layer_names": ["indexer.0", "indexer.1"]},
                {"group_id": 1, "backend": "KIMI_K3_KDA", "layer_names": ["kda.0", "kda.1"]},
            ]
        record = _decoder_cache_names(record)
        record_path.write_text(json.dumps(record))
        ref["sha256"] = hashlib.sha256(record_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(index))
    return path, launches


def test_import_rejects_worker_with_self_consistent_but_missing_configured_moe_layer(tmp_path, monkeypatch):
    from collector.fpm_forward.runtime_observations import validate_observation_phase

    path, launches = _complete_fixture(tmp_path, monkeypatch)
    launch = launches["tp2"]
    assert validate_observations(path, launches)["tp2"]["status"] == "complete"

    def omit_layer(record):
        if record["kind"] == "worker" and record["dp_rank"] == 0 and record["tp_rank"] == 1:
            evidence = record["initialized_moe"]
            evidence["layers"].pop()
            evidence["expected_layer_names"].pop()

    _mutate(path, omit_layer)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete" and result["resources"] is None
    assert any("model configuration inventory" in message for message in result["diagnostics"])
    with pytest.raises(ValueError, match="model configuration inventory"):
        validate_observation_phase(path, "tp2", launch, phase="decode", attempt_id="attempt-001")


def test_import_rejects_matching_worker_scheduler_omission_of_one_kda_group(tmp_path, monkeypatch):
    path, launches = _complete_fixture(tmp_path, monkeypatch)

    def split_kda_groups(record):
        cache = record["cache"]
        state = cache["groups"].pop()
        for name in state["layer_names"]:
            group = copy.deepcopy(state)
            group.update(
                layer_names=[name],
                representative_layer=name,
                layer_specs={name: state["layer_specs"][name]},
                spec_page_size_bytes=state["layer_specs"][name]["spec_page_size_bytes"],
            )
            if record["kind"] == "worker":
                group["layer_classes"] = {name: state["layer_classes"][name]}
            cache["groups"].append(group)
        if record["kind"] == "worker":
            original = record["attention_groups"].pop()
            record["attention_groups"].extend(
                {**original, "group_id": gid, "layer_names": group["layer_names"]}
                for gid, group in enumerate(cache["groups"][1:], start=1)
            )
        else:
            cache["group_pool_ids"] = [cache["pool_id"]] * len(cache["groups"])

    _mutate(path, split_kda_groups)
    assert validate_observations(path, launches)["tp2"]["status"] == "complete"

    def omit_group(record):
        cache = record["cache"]
        omitted = set(cache["groups"].pop()["layer_names"])
        assert len(cache["groups"]) == 2  # Another KDA group remains.
        if record["kind"] == "worker":
            record["attention_groups"].pop()
            for name in omitted:
                cache["layer_components"].pop(name)
            for allocation in cache["tensor_allocations"]:
                allocation["layers"] = [name for name in allocation["layers"] if name not in omitted]
        else:
            cache["group_pool_ids"].pop()

    _mutate(path, omit_group)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete" and result["resources"] is None
    assert "cache layers differ from the checkpoint decoder inventory" in str(result["diagnostics"])


@pytest.mark.parametrize("rows", [1, 63, 64, 65])
def test_import_checks_kernel_table_sequence_capacity(tmp_path, monkeypatch, rows):
    path, launches = _complete_fixture(tmp_path, monkeypatch, packed_indexer=True)
    cache, _ = [_decoder_cache_names(value) for value in _kernel_view_fixture(monkeypatch)]
    for mapping in cache["kernel_block_views"].values():
        mapping["kernel_block_table_shape"][1] = rows

    def set_kernel_views(record):
        if record["kind"] == "worker":
            record["cache"]["kernel_block_views"] = copy.deepcopy(cache["kernel_block_views"])
            for name in cache["kernel_block_views"]:
                record["cache"]["layer_components"][name] = copy.deepcopy(cache["layer_components"][name])

    _mutate(path, set_kernel_views)
    assert launches["tp2"]["collection"]["max_num_seqs"] == 64
    result = validate_observations(path, launches)["tp2"]
    if rows == 64:
        assert result["status"] == "complete", result["diagnostics"]
    else:
        assert result["status"] == "incomplete", result
        assert any("kernel-block table" in message for message in result["diagnostics"])


@pytest.mark.parametrize("width", [32, 62, 64, 66])
def test_import_checks_kernel_table_width_against_reviewed_context(tmp_path, monkeypatch, width):
    path, launches = _complete_fixture(tmp_path, monkeypatch, packed_indexer=True)
    cache, _ = [_decoder_cache_names(value) for value in _kernel_view_fixture(monkeypatch)]
    for mapping in cache["kernel_block_views"].values():
        mapping["kernel_block_table_shape"] = [1, 64, width]

    def set_kernel_views(record):
        if record["kind"] == "worker":
            record["cache"]["kernel_block_views"] = copy.deepcopy(cache["kernel_block_views"])
            for name in cache["kernel_block_views"]:
                record["cache"]["layer_components"][name] = copy.deepcopy(cache["layer_components"][name])

    _mutate(path, set_kernel_views)
    assert launches["tp2"]["collection"]["max_model_len"] == 4096
    result = validate_observations(path, launches)["tp2"]
    if width == 64:
        assert result["status"] == "complete", result["diagnostics"]
    else:
        assert result["status"] == "incomplete" and result["resources"] is None
        assert any("kernel-block table" in message for message in result["diagnostics"])


@pytest.mark.parametrize("kernel", [2, 16, 32])
def test_import_rejects_self_consistent_unsupported_cuda_kernel_blocks(tmp_path, monkeypatch, kernel):
    path, launches = _complete_fixture(tmp_path, monkeypatch, packed_indexer=True)

    def change_kernel(record):
        if record["kind"] != "worker":
            return
        cache = record["cache"]
        subdivisions, stride = 128 // kernel, 16384 // (kernel * 16)
        for name, mapping in cache["kernel_block_views"].items():
            mapping.update(
                kernel_block_size=kernel,
                kernel_block_stride=stride,
                kernel_block_offsets=list(range(subdivisions)),
                kernel_block_table_shape=[1, 64, 4096 // kernel],
            )
            cache["layer_components"][name][0].update(
                shape=[9 * stride + subdivisions, kernel, 16], stride_bytes=[kernel * 16, 16, 1]
            )

    _mutate(path, change_kernel)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete" and result["resources"] is None
    assert any("CUDA indexer requires 64-token" in message for message in result["diagnostics"])


@pytest.mark.parametrize("convolution", ["auto", "bfloat16", "float32"])
@pytest.mark.parametrize("recurrence", ["auto", "bfloat16", "float32"])
@pytest.mark.parametrize("observed", ["bfloat16", "float32"])
@pytest.mark.parametrize("observed_convolution", ["bfloat16", "float32"])
def test_import_binds_native_state_dtypes_to_resolved_cache_settings(
    tmp_path, monkeypatch, convolution, recurrence, observed, observed_convolution
):
    path, launches = _complete_fixture(tmp_path, monkeypatch)
    conv = "bfloat16" if convolution == "auto" else convolution
    expected = "float32" if recurrence == "auto" else recurrence

    def change_state(record):
        record["resolved_config"]["cache_config"].update(
            mamba_cache_dtype=convolution, mamba_ssm_cache_dtype=recurrence
        )
        state = record["cache"]["groups"][-1]
        sizes = [2 if dtype == "bfloat16" else 4 for dtype in (observed_convolution, observed)]
        for spec in state["layer_specs"].values():
            spec.update(
                state_dtypes=[f"torch.{observed_convolution}", f"torch.{observed}"],
                state_content_size_bytes=8 * sum(sizes),
            )
        if record["kind"] == "worker":
            for name in state["layer_names"]:
                components = record["cache"]["layer_components"][name]
                offset = components[0]["storage_offset_bytes"]
                for component, dtype, size in zip(components, (observed_convolution, observed), sizes, strict=True):
                    component.update(
                        dtype=f"torch.{dtype}",
                        element_size_bytes=size,
                        stride_bytes=[512, 4 * size, size],
                        storage_offset_bytes=offset,
                    )
                    offset += 8 * size

    _mutate(path, change_state)
    result = validate_observations(path, launches)["tp2"]
    if observed == expected and observed_convolution == conv:
        assert result["status"] == "complete", result["diagnostics"]
    else:
        assert result["status"] == "incomplete" and result["resources"] is None
        assert any("state dtypes differ from the resolved" in message for message in result["diagnostics"])


@pytest.mark.parametrize("workload_kind", ["prefill", "decode"])
@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3"])
def test_complete_rank_import_preserves_native_grouping_state_and_precision(
    tmp_path, monkeypatch, workload_kind, cache_dtype
):
    path, launches = _complete_fixture(tmp_path, monkeypatch, workload_kind=workload_kind)

    def resolve_dtype(record):
        record["resolved_config"]["cache_config"]["cache_dtype"] = cache_dtype
        group = record["cache"]["groups"][0]
        if "layer_specs" in group:
            for name, spec in group["layer_specs"].items():
                if name.endswith(".self_attn.attn"):
                    spec["cache_dtype_str"] = cache_dtype
        else:
            group["cache_dtype_str"] = cache_dtype

    _mutate(path, resolve_dtype)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "complete", result["diagnostics"]
    assert len(result["resources"]["cache_groups"]) == 2
    assert result["resources"]["cache_groups"][-1]["blocks_per_request"] == 1
    assert result["resources"]["runtime_memory"]["kv_cache_bytes"] == 9 * 512
    assert len(result["provenance"]["rank_capacities"]) == 4
    assert {capacity["phase"] for capacity in result["provenance"]["rank_capacities"]} == {workload_kind}
    for artifact in result["provenance"]["artifacts"]:
        evidence = artifact["evidence"]
        if evidence["kind"] == "worker":
            assert runtime_k31.moe_precision(evidence["initialized_moe"]) == "w4a8_mxfp4_mxfp8_trtllm"


def test_import_rejects_e4m3_spelling_for_indexer_even_with_matching_scheduler(tmp_path, monkeypatch):
    path, launches = _complete_fixture(tmp_path, monkeypatch)

    def corrupt(record):
        group = record["cache"]["groups"][0]
        if "layer_specs" in group:
            for name, spec in group["layer_specs"].items():
                if name.endswith(".self_attn.indexer.k_cache"):
                    spec["cache_dtype_str"] = "fp8_e4m3"

    _mutate(path, corrupt)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete" and result["resources"] is None, result
    assert "cache spec dtype differs" in str(result["diagnostics"])


@pytest.mark.parametrize("quantization", ["absent", "wrong_method", "wrong_class"])
def test_k31_import_requires_audited_quantization_even_for_reviewed_bf16_launch(tmp_path, monkeypatch, quantization):
    path, launches = _complete_fixture(tmp_path, monkeypatch)
    launches["tp2"]["precision"]["moe_quant_mode"] = "bfloat16"
    _replace_launch(path, launches)

    def corrupt(record):
        config = record["resolved_config"]
        if quantization == "absent":
            config["model_config"]["quantization"] = None
            config["quantization_config"] = None
        elif quantization == "wrong_method":
            config["model_config"]["quantization"] = "fp8"
        else:
            config["quantization_config"]["type"] = "vllm.model_executor.layers.quantization.fp8.Fp8Config"
        record.pop("initialized_moe", None)

    _mutate(path, corrupt)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete" and result["resources"] is None, result
    assert "K3.1 runtime weight precision requires audited MXFP4" in str(result["diagnostics"])


@pytest.mark.parametrize("omit", ["indexer", "mla"])
def test_import_requires_initialized_indexed_mla_despite_consistent_cache_maps(tmp_path, monkeypatch, omit):
    path, launches = _complete_fixture(tmp_path, monkeypatch)

    def remove_layers(record):
        cache = record["cache"]
        group = cache["groups"][0]
        omitted = {name for name in group["layer_names"] if name in _decoder_cache_names([f"{omit}.0", f"{omit}.1"])}
        group["layer_names"] = [name for name in group["layer_names"] if name not in omitted]
        for field in ("layer_specs", "layer_classes"):
            if field in group:
                group[field] = {name: value for name, value in group[field].items() if name not in omitted}
        group["representative_layer"] = group["layer_names"][0]
        group["spec_page_size_bytes"] = sum(spec["spec_page_size_bytes"] for spec in group["layer_specs"].values())
        if record["kind"] == "worker":
            cache["layer_components"] = {
                name: value for name, value in cache["layer_components"].items() if name not in omitted
            }
            cache["tensor_allocations"] = [
                a for a in cache["tensor_allocations"] if not omitted.intersection(a["layers"])
            ]
            record["attention_groups"] = [
                a for a in record["attention_groups"] if not omitted.intersection(a["layer_names"])
            ]
            # Keep the summary unchanged: only per-layer evidence establishes
            # that the indexed MLA path was actually initialized.

    _mutate(path, remove_layers)
    index = json.loads(path.read_text())
    references = index["configurations"]["tp2"]["attempts"][0]["phases"]["decode"]["artifacts"]
    records = [json.loads((path.parent / ref["path"]).read_text()) for ref in references]
    worker = next(record for record in records if record["kind"] == "worker")
    scheduler = next(record for record in records if record["kind"] == "scheduler")
    # Physical accounting alone still accepts the mutually consistent maps.
    _physical_views(worker["cache"], scheduler["cache"], "fp8", context=4096)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete" and result["resources"] is None, result
    assert "requires initialized MLA and DSA indexer layers" in str(result["diagnostics"])


@pytest.mark.parametrize("layer,dtype", [("mla.0", "nvfp4"), ("mla.0", "fp8_ds_mla"), ("indexer.0", "fp8_ds_mla")])
def test_native_spec_dtype_must_match_each_resolved_cache_layout(tmp_path, monkeypatch, layer, dtype):
    layer = _decoder_cache_names(layer)
    path, launches = _complete_fixture(tmp_path, monkeypatch)

    def corrupt(record):
        cache = record["cache"]
        group = cache["groups"][0]
        # Update the scheduler projection too: equality alone is insufficient.
        if "layer_specs" in group:
            group["layer_specs"][layer]["cache_dtype_str"] = dtype
        elif layer.endswith(".self_attn.attn"):
            group["cache_dtype_str"] = dtype

    _mutate(path, corrupt)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete", result
    assert any("cache spec dtype differs" in reason for reason in result["diagnostics"])


def test_import_verifies_the_observation_bytes_it_parses(tmp_path, monkeypatch):
    from collector.fpm_forward import runtime_observations

    path, launches = _complete_fixture(tmp_path, monkeypatch)
    read = runtime_observations.read_json
    replaced = []

    def replace_before_read(artifact, **kwargs):
        record = json.loads(artifact.read_text())
        if record.get("kind") == "worker" and not replaced:
            record["hardware"]["total_memory_bytes"] += 1
            artifact.write_text(json.dumps(record))
            replaced.append(True)
        return read(artifact, **kwargs)

    monkeypatch.setattr(runtime_observations, "read_json", replace_before_read)
    result = validate_observations(path, launches)["tp2"]
    assert replaced == [True]
    assert result["status"] == "incomplete", result
    assert any("runtime artifact hash mismatch" in reason for reason in result["diagnostics"])


@pytest.mark.parametrize("role", ["worker", "scheduler"])
@pytest.mark.parametrize("architectures", [None, [], ["OtherModel"], ["KimiK31ForConditionalGeneration", "OtherModel"]])
def test_native_layout_requires_exact_k31_architecture_on_each_role(tmp_path, monkeypatch, role, architectures):
    path, launches = _complete_fixture(tmp_path, monkeypatch)
    index = json.loads(path.read_text())
    phase = index["configurations"]["tp2"]["attempts"][0]["phases"]["decode"]
    for ref in phase["artifacts"]:
        record_path = path.parent / ref["path"]
        record = json.loads(record_path.read_text())
        if record["kind"] != role:
            continue
        model = record["resolved_config"]["model_config"]
        if architectures is None:
            model.pop("architectures", None)
        else:
            model["architectures"] = architectures
        # BF16 is valid generically and must not bypass the K3.1 baseline.
        record["resolved_config"]["cache_config"]["enable_prefix_caching"] = True
        ref["sha256"] = hashlib.sha256(json.dumps(record).encode()).hexdigest()
        record_path.write_text(json.dumps(record))
    path.write_text(json.dumps(index))
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete"
    assert any(
        "requires the observed KimiK31ForConditionalGeneration architecture" in reason
        for reason in result["diagnostics"]
    )


def test_one_rank_missing_initialized_precision_keeps_import_incomplete(tmp_path, monkeypatch):
    path, launches = _complete_fixture(tmp_path, monkeypatch)
    index = json.loads(path.read_text())
    refs = index["configurations"]["tp2"]["attempts"][0]["phases"]["decode"]["artifacts"]
    target = tmp_path / refs[0]["path"]
    record = json.loads(target.read_text())
    record.pop("initialized_moe")
    target.write_text(json.dumps(record))
    refs[0]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    path.write_text(json.dumps(index))
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete" and result["resources"] is None
    assert "initialized MoE backend" in str(result["diagnostics"])


def test_import_rejects_state_context_shorter_than_reviewed_launch(tmp_path, monkeypatch):
    from .test_runtime_observations import _mutate

    path, launches = _complete_fixture(tmp_path, monkeypatch)
    assert launches["tp2"]["collection"]["max_model_len"] == 4096

    def shorter_context(record):
        state = record["cache"]["groups"][-1]
        state["block_size_tokens"] = 32
        for child in state["layer_specs"].values():
            child.update(block_size_tokens=32, max_model_len=32)

    _mutate(path, shorter_context)
    result = validate_observations(path, launches)["tp2"]
    assert result["status"] == "incomplete" and result["resources"] is None
    assert "one non-speculative state page" in str(result["diagnostics"])
