# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate K3.1 native observations, never infer missing runtime evidence.

Source contract: runtime/instrumentation/vllm-kimi-k31.md.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

VERSION = "0.1.dev24216+g266e88c05"
LAYOUT = "kimi-k31-block-outer/v1"
SPEC = "vllm.v1.kv_cache_interface."
KDA = "vllm.models.kimi_k3.nvidia.kda.KimiK3DeltaAttention"
ATTENTION_CLASSES = {
    "vllm.model_executor.layers.attention.mla_attention.MLAAttention",
    "vllm.models.kimi_k3.common.dsa_cache.KimiK3IndexerCache",
}
_NATIVE_SPEC_FIELDS = {
    "spec_type",
    "block_size_tokens",
    "spec_page_size_bytes",
    "unpadded_page_size_bytes",
    "sliding_window",
    "attention_chunk_size",
    "extra_retained_tokens",
    "state_shapes",
    "state_dtypes",
    "state_content_size_bytes",
    "page_size_padded",
    "mamba_cache_mode",
    "num_speculative_blocks",
    "num_prefill_checkpoint_blocks",
    "prefill_checkpoint_alignment",
    "max_model_len",
    "blocks_per_request",
    "dtype",
    "cache_dtype_str",
}


def is_audited_runtime(runtime):
    """Exact inspected version and source map, shared by planning and import."""
    if not isinstance(runtime, dict) or runtime.get("version") != VERSION:
        return False
    pinned = json.loads((Path(__file__).parent / "runtime/vllm-kimi-k31.example.json").read_text())["runtime"]
    return {key: runtime.get(key) for key in pinned} == pinned


def _positive(value, label):
    if type(value) is not int or not 0 < value <= 2**53:
        raise ValueError(f"K3.1 {label} must be a positive integer")
    return value


def _nonnegative(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"K3.1 {label} must be a nonnegative integer")
    return value


def moe_precision(evidence, *, model_config=None):
    if not isinstance(evidence, dict) or evidence.get("enumeration") != "model.named_modules:RoutedExperts":
        raise ValueError("Kimi K3.1 runtime weight precision requires initialized MoE backend and activation evidence")
    layers = evidence.get("layers")
    expected = evidence.get("expected_layer_names")
    if not isinstance(layers, list) or not layers or not isinstance(expected, list) or not expected:
        raise ValueError("K3.1 initialized MoE layer coverage is incomplete")
    names = [layer.get("name") if isinstance(layer, dict) else None for layer in layers]
    if any(not isinstance(name, str) or not name.strip() for name in [*names, *expected]):
        raise ValueError("K3.1 initialized MoE layer names must be nonempty strings")
    if len(set(names)) != len(names) or set(names) != set(expected) or len(expected) != len(set(expected)):
        raise ValueError("K3.1 initialized MoE layer coverage is inconsistent")
    if model_config is not None:
        # Pinned models/kimi_k3/nvidia/model.py::KimiDecoderLayer selects
        # expert layers by decoder index, dense prefix and MoE frequency.
        # kimi_linear.py::KimiLinearConfig defaults the latter two to 0/1.
        # Import currently supports PP1, so each worker owns this inventory;
        # module-name equality across workers is neither needed nor sufficient.
        config = model_config.get("text_config", model_config)
        if not isinstance(config, dict):
            raise ValueError("K3.1 decoder configuration must be an object")
        count = _positive(config.get("num_hidden_layers"), "decoder layer count")
        _positive(config.get("num_experts"), "configured expert count")
        first = _nonnegative(config.get("first_k_dense_replace", 0), "dense layer prefix")
        frequency = _positive(config.get("moe_layer_freq", 1), "MoE layer frequency")
        # Count multiples in [first, count) without enumerating an untrusted
        # decoder size. Work and retained memory scale with observed layers.
        eligible = max(0, (count - 1) // frequency - (first + frequency - 1) // frequency + 1)
        if len(names) != eligible:
            raise ValueError("K3.1 initialized MoE layer coverage differs from the model configuration inventory")
        matches = [re.search(r"(?:^|\.)layers\.(\d+)\.", name) for name in names]
        indices = [int(match[1]) for match in matches if match is not None]
        if (
            len(indices) != len(names)
            or len(set(indices)) != len(names)
            or any(not first <= index < count or index % frequency for index in indices)
        ):
            raise ValueError("K3.1 initialized MoE layer coverage differs from the model configuration inventory")
    modes = set()
    for layer in layers:
        if (
            layer.get("method_type") != "vllm.model_executor.layers.quantization.mxfp4.Mxfp4MoEMethod"
            or layer.get("layer_type") != "vllm.model_executor.layers.fused_moe.routed_experts.RoutedExperts"
            or layer.get("kernel_type") != "vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEKernel"
        ):
            raise ValueError("K3.1 initialized MoE method/kernel has no audited precision mapping")
        descriptors = layer.get("descriptors", {})
        if descriptors != {
            key: {"dtype": dtype}
            for key, dtype in (("_a1", "mxfp8"), ("_a2", "mxfp8"), ("_w1", "mxfp4"), ("_w2", "mxfp4"))
        }:
            raise ValueError("K3.1 initialized MoE activation/weight descriptors are unsupported")
        backend = layer.get("backend")
        if backend != "FLASHINFER_TRTLLM_MXFP4_MXFP8":
            raise ValueError("K3.1 initialized MoE backend has no audited precision mapping")
        modes.add("w4a8_mxfp4_mxfp8_trtllm")
        shapes = layer.get("weight_shapes", {})
        if not isinstance(shapes, dict) or set(shapes) != {"w13_weight", "w2_weight"}:
            raise ValueError("K3.1 initialized expert weights are incomplete")
        for shape in shapes.values():
            if not isinstance(shape, list) or not shape:
                raise ValueError("K3.1 initialized expert weight shape is missing")
            for size in shape:
                _positive(size, "expert weight dimension")
    if len(modes) != 1:
        raise ValueError("K3.1 initialized expert precision differs across layers")
    return modes.pop()


def normalize_groups(cache, scheduled, page_bytes, *, context):
    context = _positive(context, "reviewed maximum context")
    if cache.get("allocation_layout") != LAYOUT or scheduled.get("allocation_layout") != LAYOUT:
        raise ValueError("K3.1 worker/scheduler allocation mapping disagrees")
    raw, other = cache.get("groups"), scheduled.get("groups")
    if not isinstance(raw, list) or not raw or not isinstance(other, list) or len(raw) != len(other):
        raise ValueError("K3.1 cache group coverage is incomplete")
    names, result = set(), []
    for index, (group, scheduler) in enumerate(zip(raw, other, strict=True)):
        if not isinstance(group, dict) or not isinstance(scheduler, dict):
            raise ValueError("K3.1 cache group must be an object")
        if "layer_specs" in group and group.get("spec_type") != SPEC + "UniformTypeKVCacheSpecs":
            raise ValueError("K3.1 per-layer specs require a native uniform group")
        if group.get("spec_type") == SPEC + "UniformTypeKVCacheSpecs":
            children = group.get("layer_specs")
            if (
                not isinstance(children, dict)
                or not children
                or any(not isinstance(child, dict) for child in children.values())
            ):
                raise ValueError("K3.1 uniform group requires every native per-layer spec")
            for child in children.values():
                if child.get("spec_type") == SPEC + "UniformTypeKVCacheSpecs":
                    raise ValueError("nested K3.1 uniform specs are unsupported")
                if set(child) - _NATIVE_SPEC_FIELDS:
                    raise ValueError("K3.1 child spec contains non-native spec fields or group metadata")
        expected = {key: value for key, value in group.items() if key not in {"kind", "layer_classes"}}
        if group.get("spec_type") == SPEC + "UniformTypeKVCacheSpecs":
            # Pinned kv_cache_utils.generate_scheduler_kv_cache_config selects
            # the first native child; JSON serialization may reorder its keys.
            children = expected.pop("layer_specs", None)
            representative = expected.pop("representative_layer", None)
            if not isinstance(children, dict) or not isinstance(representative, str) or representative not in children:
                raise ValueError("K3.1 scheduler projection requires the observed representative layer")
            expected.update(children[representative])
        if expected != scheduler:
            raise ValueError("K3.1 worker and scheduler cache group geometry disagrees")
        layers = group.get("layer_names")
        if (
            not isinstance(layers, list)
            or not layers
            or any(not isinstance(x, str) or not x for x in layers)
            or len(set(layers)) != len(layers)
            or names.intersection(layers)
        ):
            raise ValueError("K3.1 cache layer names must be nonempty and unique")
        names.update(layers)
        classes = group.get("layer_classes", {})
        if (
            not isinstance(classes, dict)
            or set(classes) != set(layers)
            or any(not isinstance(x, str) or not x for x in classes.values())
        ):
            raise ValueError("K3.1 cache layer class coverage is incomplete")
        block = _positive(group.get("block_size_tokens"), "block size")
        _positive(group.get("spec_page_size_bytes"), "spec page size")
        kind = group.get("kind")
        item = {
            "name": f"runtime_group_{index}",
            "kind": kind,
            "num_layers": len(layers),
            "block_size_tokens": block,
            "page_size_bytes": page_bytes,
            "sliding_window": None,
        }
        if (
            group.get("sliding_window") is not None
            or group.get("attention_chunk_size") is not None
            or group.get("extra_retained_tokens") != 0
            or type(group.get("extra_retained_tokens")) is not int
        ):
            raise ValueError("K3.1 alternate cache retention is unsupported")
        if group.get("spec_type") == SPEC + "UniformTypeKVCacheSpecs":
            children = group.get("layer_specs")
            if not isinstance(children, dict) or set(children) != set(layers):
                raise ValueError("K3.1 uniform group requires every native per-layer spec")
            if (
                sum(_positive(child.get("spec_page_size_bytes"), "layer page bytes") for child in children.values())
                != group["spec_page_size_bytes"]
            ):
                raise ValueError("K3.1 uniform group page sum contradicts native per-layer specs")
            child_groups = [
                {**child, "layer_names": [name], "kind": kind, "layer_classes": {name: classes[name]}}
                for name, child in children.items()
            ]
            child_scheduled = [
                {key: value for key, value in child.items() if key not in {"kind", "layer_classes"}}
                for child in child_groups
            ]
            normalized = normalize_groups(
                {"allocation_layout": LAYOUT, "groups": child_groups},
                {"allocation_layout": LAYOUT, "groups": child_scheduled},
                page_bytes,
                context=context,
            )
            if any(child["block_size_tokens"] != block or child["kind"] != kind for child in normalized):
                raise ValueError("K3.1 uniform group has incompatible per-layer retention")
            if kind == "state":
                item["blocks_per_request"] = 1
            result.append(item)
            continue
        if group.get("spec_type") == SPEC + "MambaSpec" and kind == "state":
            if (
                set(classes.values()) != {KDA}
                or group.get("mamba_cache_mode") != "none"
                or group.get("num_speculative_blocks") != 0
                or type(group.get("num_speculative_blocks")) is not int
                or group.get("blocks_per_request") != 1
                or type(group.get("blocks_per_request")) is not int
                or _positive(group.get("max_model_len"), "maximum context") != context
                or block < context
            ):
                raise ValueError("K3.1 state requires native KDA, mamba none and one non-speculative state page")
            shapes, dtypes = group.get("state_shapes"), group.get("state_dtypes")
            if not isinstance(shapes, list) or len(shapes) != 2 or not isinstance(dtypes, list) or len(dtypes) != 2:
                raise ValueError("K3.1 KDA requires both convolution and recurrence components")
            content = 0
            for shape, dtype in zip(shapes, dtypes, strict=True):
                if not isinstance(shape, list) or not shape or dtype not in {"torch.bfloat16", "torch.float32"}:
                    raise ValueError("K3.1 state shape/dtype is unsupported")
                content += math.prod(_positive(x, "state dimension") for x in shape) * (
                    2 if dtype == "torch.bfloat16" else 4
                )
            if content != group.get("state_content_size_bytes") or content > group["spec_page_size_bytes"]:
                raise ValueError("K3.1 state payload differs from native page geometry")
            padded = group.get("page_size_padded")
            if (padded is None and content != group["spec_page_size_bytes"]) or (
                padded is not None and padded != group["spec_page_size_bytes"]
            ):
                raise ValueError("K3.1 state padding differs from native page geometry")
            item["blocks_per_request"] = 1
        elif (
            group.get("spec_type") != SPEC + "MLAAttentionSpec"
            or kind != "attention"
            or set(classes.values()) - ATTENTION_CLASSES
        ):
            raise ValueError("K3.1 cache spec/kind has no audited mapping")
        elif any(
            group.get("cache_dtype_str")
            not in ({None, "fp8"} if cls.endswith("KimiK3IndexerCache") else {"fp8", "fp8_e4m3", "fp8_ds_mla"})
            for cls in classes.values()
        ):
            raise ValueError("K3.1 attention cache dtype has no audited FP8 mapping")
        if kind == "attention":
            payload = _positive(group.get("unpadded_page_size_bytes"), "attention unpadded payload bytes")
            if payload > group["spec_page_size_bytes"] or payload % block:
                raise ValueError("K3.1 attention payload differs from native page geometry")
        result.append(item)
    return result


def _validate_state_dtypes(spec, config):
    # Pinned KimiK3DeltaAttention.get_state_dtype delegates to
    # mamba_utils.MambaStateDtypeCalculator.kda_state_dtype: convolution uses
    # mamba_cache_dtype (auto follows model dtype), auto recurrence is FP32.
    cache = config["cache_config"]
    conv = cache.get("mamba_cache_dtype")
    if conv == "auto":
        conv = config["model_config"]["dtype"].removeprefix("torch.")
    recurrent = cache.get("mamba_ssm_cache_dtype")
    if recurrent == "auto":
        recurrent = "float32"
    if conv not in {"bfloat16", "float32"} or recurrent not in {"bfloat16", "float32"}:
        raise ValueError("K3.1 resolved KDA cache dtypes are missing or unsupported")
    if spec.get("state_dtypes") != [f"torch.{conv}", f"torch.{recurrent}"]:
        raise ValueError("K3.1 native state dtypes differ from the resolved cache configuration")


def _validate_cache_inventory(cache, model_config):
    """Check PP1 cache ownership against the independently hash-bound decoder."""
    config = model_config.get("text_config", model_config)
    if not isinstance(config, dict):
        raise ValueError("K3.1 decoder configuration must be an object")
    count = _positive(config.get("num_hidden_layers"), "decoder layer count")

    def layer_indices(section, field):
        if section is None:
            return set()
        if not isinstance(section, dict) or not isinstance(section.get(field), list):
            raise ValueError("K3.1 checkpoint cache layer inventory is malformed")
        values = section[field]
        if any(type(value) is not int or not 1 <= value <= count for value in values):
            raise ValueError("K3.1 checkpoint cache layer indices must be one-based decoder indices")
        return {value - 1 for value in values}

    # Pinned KimiLinearConfig.is_kda_layer/is_dsa_layer use one-based lists;
    # KimiDecoderLayer selects KDA first, then MLA with an optional DSA indexer.
    kda = layer_indices(config.get("linear_attn_config"), "kda_layers")
    if kda and config["linear_attn_config"].get("use_full_rank_gate") is not True:
        raise ValueError("K3.1 native KDA requires checkpoint use_full_rank_gate=true")
    dsa_config = config.get("dsa_config")
    dsa_field = "layers" if isinstance(dsa_config, dict) and "dsa_layers" not in dsa_config else "dsa_layers"
    indexer = layer_indices(dsa_config, dsa_field) - kda
    mla_class = "vllm.model_executor.layers.attention.mla_attention.MLAAttention"
    indexer_class = "vllm.models.kimi_k3.common.dsa_cache.KimiK3IndexerCache"
    observed = {cls: set() for cls in (KDA, mla_class, indexer_class)}
    for group in cache["groups"]:
        for name in group["layer_names"]:
            cls = group["layer_classes"].get(name)
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            if cls not in observed or match is None:
                raise ValueError("K3.1 cache layers differ from the checkpoint decoder inventory")
            index = int(match[1])
            if index in observed[cls] or not 0 <= index < count:
                raise ValueError("K3.1 cache layers differ from the checkpoint decoder inventory")
            observed[cls].add(index)
    if (
        observed[KDA] != kda
        or observed[indexer_class] != indexer
        or len(observed[mla_class]) != count - len(kda)
        or observed[mla_class].intersection(kda)
    ):
        raise ValueError("K3.1 cache layers differ from the checkpoint decoder inventory")


def _validate_state_shapes(spec, config, model_config, conv_state_layout):
    # vLLM 266e88c05 KimiK3DeltaAttention.get_state_shape delegates to
    # MambaStateShapeCalculator.kda_state_shape. No speculation is admitted.
    decoder = model_config.get("text_config", model_config)
    kda = decoder.get("linear_attn_config")
    if not isinstance(kda, dict):
        raise ValueError("K3.1 checkpoint KDA state configuration is missing")
    heads = _positive(kda.get("num_heads"), "KDA heads")
    dim = _positive(kda.get("head_dim"), "KDA head dimension")
    conv = _positive(kda.get("short_conv_kernel_size"), "KDA convolution size")
    tp = _positive(config["parallel_config"].get("tensor_parallel_size"), "KDA TP size")
    if heads % tp or conv < 2 or conv_state_layout not in {"SD", "DS"}:
        raise ValueError("K3.1 checkpoint KDA state geometry is unsupported")
    local_heads = heads // tp
    convolution = [conv - 1, 3 * local_heads * dim]
    if conv_state_layout == "DS":
        convolution.reverse()
    if spec.get("state_shapes") != [convolution, [local_heads, dim, dim]]:
        raise ValueError("K3.1 native state shapes differ from the checkpoint and TP size")


def metadata_builder_count(config):
    # Pinned gpu/attn_utils.py::init_attn_backend calls
    # ubatch_utils.get_num_ubatches: one plane when use_ubatching is false.
    parallel = config["parallel_config"]
    enabled = parallel.get("use_ubatching")
    if type(enabled) is not bool:
        raise ValueError("K3.1 kernel-block table requires the resolved microbatching configuration")
    return _positive(parallel.get("num_ubatches"), "metadata builder count") if enabled else 1


def _validate_mla_payload(spec, config, model_config):
    # vLLM 266e88c051ca46607fd9de20216bd58426f80052:
    # models/kimi_k3/nvidia/mla.py::MultiHeadLatentAttention sets head_size to
    # kv_lora_rank + qk_rope_head_dim. MLAAttention.get_kv_cache_spec and
    # v1/kv_cache_interface.py::MLAAttentionSpec use that FP8 latent width
    # for FLASHINFER_MLA_SPARSE. flashmla_sparse.py documents the separate
    # 656-byte fp8_ds_mla row: 512 FP8 NoPE + 16 scale + 128 BF16 RoPE bytes.
    decoder = model_config.get("text_config", model_config)
    latent = _positive(decoder.get("kv_lora_rank"), "MLA latent dimension")
    rope = _positive(decoder.get("qk_rope_head_dim"), "MLA RoPE dimension")
    backend = config["attention_config"]["backend"]
    if backend == "FLASHINFER_MLA_SPARSE":
        token_bytes = latent + rope
        block_multiple = 32
    elif backend == "FLASHMLA_SPARSE" and (latent, rope) == (512, 64):
        token_bytes = 656
        block_multiple = 64
    else:
        raise ValueError("K3.1 checkpoint MLA geometry has no audited backend payload mapping")
    # The pinned sparse MLA backends' get_supported_kernel_block_sizes
    # constrain MLA independently of the indexer's 64-token kernel blocks.
    if _positive(spec.get("block_size_tokens"), "MLA block size") % block_multiple:
        raise ValueError(f"K3.1 {backend} MLA block size must be a multiple of {block_multiple} tokens")
    # Check logical payload independently of native page padding and allocation.
    if spec.get("unpadded_page_size_bytes") != token_bytes * spec["block_size_tokens"]:
        raise ValueError("K3.1 MLA payload differs from the checkpoint and backend geometry")


def validate_attention_assignments(worker, *, model_config=None, conv_state_layout="SD"):
    """Reconcile initialized layer backends and dtypes with resolved policy."""
    cache = worker["cache"]
    config = worker["resolved_config"]
    selected = config["attention_config"]["backend"]
    classes = {
        "vllm.model_executor.layers.attention.mla_attention.MLAAttention": selected,
        "vllm.models.kimi_k3.common.dsa_cache.KimiK3IndexerCache": "DEEPSEEK_V32_INDEXER",
        KDA: "KIMI_K3_KDA",
    }
    expected = {}
    for gid, group in enumerate(cache["groups"]):
        if not isinstance(group, dict):
            raise ValueError("K3.1 cache group must be an object")
        if not isinstance(group.get("layer_classes"), dict):
            raise ValueError("K3.1 cache layer class coverage is incomplete")
        for name in group["layer_names"]:
            cls = group["layer_classes"][name]
            if cls not in classes:
                raise ValueError("K3.1 attention layer class has no audited backend mapping")
            expected[(gid, name)] = classes[cls]
            spec = group["layer_specs"][name] if "layer_specs" in group else group
            if not isinstance(spec, dict):
                raise ValueError("K3.1 native cache spec must be an object")
            if cls == KDA:
                _validate_state_dtypes(spec, config)
                if model_config is not None:
                    _validate_state_shapes(spec, config, model_config, conv_state_layout)
            if cls in ATTENTION_CLASSES:
                dtype = "fp8" if cls.endswith("KimiK3IndexerCache") else config["cache_config"]["cache_dtype"]
                observed = spec.get("cache_dtype_str")
                if cls.endswith("KimiK3IndexerCache"):
                    # Pinned DeepseekV32IndexerCache.get_kv_cache_spec leaves
                    # MLAAttentionSpec.cache_dtype_str at None. The resolved
                    # attention configuration, not that optional marker, owns
                    # the indexer's format.
                    if config["attention_config"].get("indexer_kv_dtype") != "fp8":
                        raise ValueError("K3.1 indexer requires resolved FP8 cache configuration")
                    if observed is None:
                        observed = dtype
                    if model_config is not None:
                        decoder = model_config.get("text_config", model_config)
                        dsa = decoder.get("dsa_config")
                        if not isinstance(dsa, dict):
                            raise ValueError("K3.1 checkpoint indexer configuration is missing")
                        # Pinned kimi_k3/common/dsa_cache.py::indexer_cache_entry_bytes:
                        # each FP8 key carries one FP32 scale. Bind the logical
                        # payload only; native padded pages still own placement.
                        token_bytes = _positive(dsa.get("index_head_dim"), "indexer head dimension") + 4
                        if spec.get("unpadded_page_size_bytes") != token_bytes * spec["block_size_tokens"]:
                            raise ValueError("K3.1 indexer payload differs from the checkpoint FP8 geometry")
                if observed != dtype:
                    raise ValueError("K3.1 native cache spec dtype differs from the resolved cache layout")
                if not cls.endswith("KimiK3IndexerCache") and model_config is not None:
                    _validate_mla_payload(spec, config, model_config)
    assignments = worker.get("attention_groups")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("K3.1 native attention group assignments are missing")
    observed = {}
    for group in assignments:
        if not isinstance(group, dict) or type(group.get("group_id")) is not int:
            raise ValueError("K3.1 native attention group assignment is malformed")
        names = group.get("layer_names")
        if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names):
            raise ValueError("K3.1 native attention group assignment is malformed")
        for name in names:
            key = (group["group_id"], name)
            if key in observed or key not in expected or group.get("backend") != expected[key]:
                raise ValueError("K3.1 native attention layer backend differs from the selected policy")
            observed[key] = group["backend"]
    if observed != expected:
        raise ValueError("K3.1 native attention assignments omit initialized layers")
    if not {selected, "DEEPSEEK_V32_INDEXER"}.issubset(observed.values()):
        raise ValueError("K3.1 native attention requires initialized MLA and DSA indexer layers")
    if model_config is not None:
        _validate_cache_inventory(cache, model_config)
    return observed


def _kernel_view_geometry(
    mapping, group, view, count, page, width, *, context=None, sequences=None, metadata_builders=1
):
    """Check the observed AttentionGroup's manager-to-kernel address mapping."""
    if (
        not isinstance(mapping, dict)
        or mapping.get("attention_group_class") != "vllm.v1.worker.utils.AttentionGroup"
        or mapping.get("backend_class") != "vllm.models.kimi_k3.common.dsa_cache.KimiK3IndexerBackend"
        or type(mapping.get("maps_slot_mapping")) is not bool
        or mapping.get("writes_kv_cache_from_block_table") is not True
    ):
        raise ValueError("K3.1 kernel-block view has no audited native address mapping")
    kernel = _positive(mapping.get("kernel_block_size"), "kernel block size")
    stride = _positive(mapping.get("kernel_block_stride"), "kernel block stride")
    block = group["block_size_tokens"]
    if kernel >= block or block % kernel:
        raise ValueError("K3.1 kernel-block subdivision contradicts the native manager block")
    # KimiK3IndexerBackend delegates FP8 to DeepseekV32IndexerBackend;
    # its pinned get_supported_kernel_block_sizes supports only 64 on CUDA.
    if kernel != 64:
        raise ValueError("K3.1 CUDA indexer requires 64-token kernel blocks")
    subdivisions = block // kernel
    offsets = mapping.get("kernel_block_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != subdivisions
        or any(type(value) is not int or value != index for index, value in enumerate(offsets))
    ):
        raise ValueError("K3.1 native kernel-block offsets do not cover the manager block")
    table = mapping.get("kernel_block_table_shape")
    # CP1, full-history AttentionSpec.max_num_blocks_per_req and
    # worker/block_table.py::get_block_table_width at vLLM 266e88c05:
    # align manager capacity to 128 tokens before kernel subdivision.
    manager_entries = (_positive(context, "reviewed maximum context") + block - 1) // block
    alignment = 128 // math.gcd(128, block)
    expected_width = ((manager_entries + alignment - 1) // alignment) * alignment * subdivisions
    if (
        mapping.get("kernel_block_table_dtype") != "torch.int32"
        or not isinstance(table, list)
        or len(table) != 3
        or any(type(value) is not int or value < 1 for value in table)
        or table[0] != metadata_builders
        or table[1] != _positive(sequences, "reviewed maximum sequences")
        or table[-1] != expected_width
    ):
        raise ValueError("K3.1 native kernel-block table shape/dtype is inconsistent")
    shape, strides = view["shape"], view["stride_bytes"]
    if len(shape) != 3 or shape[1] != kernel or strides[1:] != [shape[2], 1]:
        raise ValueError("K3.1 kernel-block component has no native contiguous token layout")
    kernel_bytes = kernel * shape[2]
    # AttentionGroup.map_kv_cache_to_kernel_blocks and map_to_kernel_block_table:
    # kernel page j of manager block b is b * stride + j. The strided tensor
    # spans other layers' gaps; only these addressed pages belong to this layer.
    if (
        strides[0] != kernel_bytes
        or stride * kernel_bytes != page
        or subdivisions * kernel_bytes != group["unpadded_page_size_bytes"]
        or shape[0] != (count - 1) * stride + subdivisions
    ):
        raise ValueError("K3.1 kernel-block view contradicts native pool stride or layer page coverage")
    return kernel


def validate_views(cache, precision, count, page, *, context=None, sequences=None, metadata_builders=1):
    """Verify block-outer native placements, including safe cross-group overlays.

    A block ID belongs to exactly one cache group. Different groups may overlap
    within that block's byte slot; layers in the same group may never overlap.
    """
    if precision != "fp8":
        raise ValueError("K3.1 native adapter requires FP8 attention/indexer cache")
    if {group["kind"] for group in cache["groups"]} != {"state", "attention"}:
        raise ValueError("K3.1 complete native layout requires KDA state and attention groups")
    storages = cache.get("storages")
    if not isinstance(storages, list) or len(storages) != 1 or storages[0].get("size_bytes") != count * page:
        raise ValueError("K3.1 native allocator requires one exact shared storage")
    storage = storages[0].get("storage_id")
    if not isinstance(storage, str) or not storage:
        raise ValueError("K3.1 backing storage identity is missing")
    by_name = {}
    for index, group in enumerate(cache["groups"]):
        uniform = group.get("spec_type") == SPEC + "UniformTypeKVCacheSpecs"
        if "layer_specs" in group and not uniform:
            raise ValueError("K3.1 per-layer specs require a native uniform group")
        for name in group["layer_names"]:
            spec = {**group, **group["layer_specs"][name], "kind": group["kind"]} if uniform else group
            by_name[name] = (index, spec)
            if (
                group["layer_classes"].get(name) == "vllm.models.kimi_k3.common.dsa_cache.KimiK3IndexerCache"
                and spec.get("dtype") != "torch.uint8"
            ):
                raise ValueError("K3.1 native FP8 indexer requires torch.uint8 storage")
            if spec["kind"] == "attention" and spec.get("dtype") not in {"torch.uint8", "torch.float8_e4m3fn"}:
                raise ValueError("K3.1 attention storage contradicts FP8 precision")
    views = cache.get("layer_components")
    if not isinstance(views, dict) or set(views) != set(by_name):
        raise ValueError("K3.1 bound component coverage is incomplete")
    kernel_views = cache.get("kernel_block_views", {})
    if not isinstance(kernel_views, dict) or set(kernel_views) - set(by_name):
        raise ValueError("K3.1 kernel-block view layer coverage is inconsistent")
    allocations = cache.get("tensor_allocations")
    if not isinstance(allocations, list) or not allocations:
        raise ValueError("K3.1 native placements are missing")
    covered, intervals = set(), {}
    for allocation in allocations:
        if (
            not isinstance(allocation, dict)
            or allocation.get("storage_id") != storage
            or allocation.get("size") != count * page
            or allocation.get("block_stride") != page
            or allocation.get("host_resident") is not False
        ):
            raise ValueError("K3.1 native placement is not an exact device block-outer allocation")
        layers = allocation.get("layers")
        if (
            not isinstance(layers, list)
            or not layers
            or any(name not in by_name for name in layers)
            or len(set(layers)) != len(layers)
            or covered.intersection(layers)
        ):
            raise ValueError("K3.1 native placement layer coverage is inconsistent")
        step = _positive(allocation.get("layer_stride"), "layer stride")
        start = _nonnegative(allocation.get("offset"), "placement offset")
        for index, name in enumerate(layers):
            gid, group = by_name[name]
            offset, width = start + index * step, group["spec_page_size_bytes"]
            if offset + width > page:
                raise ValueError("K3.1 placement exceeds its pool block slot")
            for lo, hi in intervals.setdefault(gid, []):
                if offset < hi and lo < offset + width:
                    raise ValueError("K3.1 layers within one group overlap")
            intervals[gid].append((offset, offset + width))
            components = views[name]
            state = group["kind"] == "state"
            mapping = kernel_views.get(name)
            if name in kernel_views and (
                state or group["layer_classes"].get(name) != "vllm.models.kimi_k3.common.dsa_cache.KimiK3IndexerCache"
            ):
                raise ValueError("K3.1 kernel-block mapping is not an observed native indexer cache")
            if (
                group["layer_classes"].get(name) == "vllm.models.kimi_k3.common.dsa_cache.KimiK3IndexerCache"
                and group["block_size_tokens"] != 64
                and name not in kernel_views
            ):
                raise ValueError("K3.1 CUDA indexer requires 64-token blocks or native kernel-block mapping coverage")
            if not isinstance(components, list) or len(components) != (2 if state else 1):
                raise ValueError("K3.1 layer component count contradicts native spec")
            component_offset = offset
            for component_index, view in enumerate(components):
                if not isinstance(view, dict):
                    raise ValueError("K3.1 layer component must be an object")
                size = _positive(view.get("element_size_bytes"), "component element bytes")
                shape, strides = view.get("shape"), view.get("stride_bytes")
                if (
                    not isinstance(shape, list)
                    or not shape
                    or not isinstance(strides, list)
                    or len(shape) != len(strides)
                    or view.get("storage_id") != storage
                    or view.get("storage_offset_bytes") != component_offset
                    or component_offset % size
                ):
                    raise ValueError("K3.1 component block coverage, offsets or storage disagrees")
                for value in shape:
                    _positive(value, "component dimension")
                for value in strides:
                    if _nonnegative(value, "component stride") % size:
                        raise ValueError("K3.1 component stride is not element aligned")
                component_block = group["block_size_tokens"]
                if name in kernel_views:
                    component_block = _kernel_view_geometry(
                        mapping,
                        group,
                        view,
                        count,
                        page,
                        width,
                        context=context,
                        sequences=sequences,
                        metadata_builders=metadata_builders,
                    )
                elif shape[0] != count or strides[0] != page:
                    raise ValueError("K3.1 component block coverage, offsets or storage disagrees")
                if state:
                    dtype = group["state_dtypes"][component_index]
                    if (
                        shape[1:] != group["state_shapes"][component_index]
                        or view.get("dtype") != dtype
                        or size != (2 if dtype == "torch.bfloat16" else 4)
                    ):
                        raise ValueError("K3.1 component shape/dtype differs from native state spec")
                elif size != 1 or view.get("dtype") != group["dtype"] or len(shape) != 3 or shape[1] != component_block:
                    raise ValueError("K3.1 attention component shape/dtype differs from native cache spec")
                if (
                    not state
                    and group["layer_classes"].get(name)
                    == "vllm.model_executor.layers.attention.mla_attention.MLAAttention"
                    and (
                        group.get("cache_dtype_str") == "fp8"
                        or (group.get("cache_dtype_str") == "fp8_ds_mla" and component_block > 64)
                    )
                    and strides[0] % shape[2]
                ):
                    # Pinned sparse_utils.flat_kv_row_view addresses complete
                    # latent/RoPE rows, including gaps between native blocks.
                    raise ValueError("K3.1 sparse MLA block stride must contain whole cache rows")
                span = size
                for stride, dimension in sorted(zip(strides[1:], shape[1:], strict=True)):
                    if dimension > 1:
                        if stride < span:
                            raise ValueError("K3.1 component dimensions overlap")
                        span += (dimension - 1) * stride
                if component_offset + span > offset + width:
                    raise ValueError("K3.1 component exceeds its native page interval")
                logical = math.prod(shape[1:]) * size
                if state:
                    expected_stride = size
                    for dimension, stride in reversed(list(zip(shape[1:], strides[1:], strict=True))):
                        if dimension > 1 and stride != expected_stride:
                            raise ValueError("K3.1 state components must retain native axis-order contiguous payloads")
                        expected_stride *= dimension
                    if span != logical:
                        raise ValueError("K3.1 state components must retain native contiguous payloads")
                if (
                    not state
                    and name not in kernel_views
                    and group["layer_classes"].get(name) == "vllm.models.kimi_k3.common.dsa_cache.KimiK3IndexerCache"
                    and strides[1:] != [shape[2], 1]
                ):
                    raise ValueError("K3.1 direct indexer requires native BLHNC token-byte strides")
                if (
                    not state
                    and name not in kernel_views
                    and group["layer_classes"].get(name)
                    == "vllm.model_executor.layers.attention.mla_attention.MLAAttention"
                    and group.get("cache_dtype_str") in {"fp8", "fp8_e4m3"}
                    and strides[1:] != [shape[2], 1]
                ):
                    # flashinfer_mla_sparse.py -> sparse_utils.flat_kv_row_view
                    # addresses contiguous token rows, retaining inter-block padding.
                    raise ValueError("K3.1 direct FlashInfer MLA requires native token-byte strides")
                if (
                    not state
                    and name not in kernel_views
                    and (logical != group["unpadded_page_size_bytes"] or span != logical)
                ):
                    raise ValueError("K3.1 attention components must retain complete native contiguous payloads")
                component_offset += logical
        covered.update(layers)
    if covered != set(by_name):
        raise ValueError("K3.1 native placements omit initialized layers")
