# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence checks of one vLLM 0.33.0 phase that the import runs after its own checks of the phase.

The import runs them last, so a refusal of one of its own checks keeps its text.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .runtime.fpm_memory_observer import _MAMBA_SPEC, _UNIFORM_SPEC
from .runtime_instrumentation import canonical_json


def validate_phase(
    phase: str,
    workers: dict[tuple[int, int], dict[str, Any]],
    schedulers: dict[int, dict[str, Any]],
    source_files: Mapping[str, str],
) -> None:
    """Run the evidence checks of one vLLM 0.33.0 phase.

    Raise ValueError for the first relation that a record breaks.
    """
    for _, scheduler in sorted(schedulers.items()):
        _check_size_tokens(scheduler)
    for dp in sorted(schedulers):
        _check_worker_block_count(phase, dp, [worker for (rank, _), worker in sorted(workers.items()) if rank == dp])
    for _, record in (*sorted(schedulers.items()), *sorted(workers.items())):
        _check_layer_specs(record)
    for _, worker in sorted(workers.items()):
        _check_audited_sources(worker, source_files)
    for _, record in (*sorted(schedulers.items()), *sorted(workers.items())):
        _check_raw_cache_config(
            record,
            [record]
            if record["kind"] == "worker"
            else [worker for (rank, _), worker in sorted(workers.items()) if rank == record["dp_rank"]],
        )
    for _, record in (*sorted(schedulers.items()), *sorted(workers.items())):
        _check_mamba_pages(record)


# hf_overrides is a launch argument with the default {} (vLLM 0.33.0 config/model.py:348, engine/arg_utils.py:1035),
# which vLLM applies on top of the loaded config.json (config/model.py:616-630, :680-692,
# transformers_utils/config.py:989-994). The reviewed launch sets none, and the config.json hash does not cover another
# value. The observer records a callable as a string (runtime/instrumentation/observer.py, _resolved). Unlike the
# checks of validate_phase, the import calls this one for each worker record beside its config format check.
def _check_hf_overrides(record: dict[str, Any]) -> None:
    value = record["resolved_config"]["model_config"].get("hf_overrides")
    if value is not None and value != {}:
        raise ValueError(f"loaded model configuration overrides require another source mapping: hf_overrides={value!r}")


# The observer refuses a runner or a backend whose source file the bundle does not hash, a layer in two attention
# groups, and attention groups that do not cover every cache layer (runtime/instrumentation/observer.py,
# _layer_views). vLLM builds the attention groups of a cache group from that group's layers, one for each backend,
# layer spec and Q-head count (vLLM 0.33.0 v1/worker/gpu/attn_utils.py:217-265,
# v1/worker/gpu_model_runner.py:6648-6726). Every cache group, a Mamba or a ring group too, gets one: only a spec
# without layer views gets none (attn_utils.py:221-223, kv_cache_interface.py:329-336, :510-512, :546-548), and the
# observer supports no such spec.
def _check_audited_sources(record: dict[str, Any], source_files: Mapping[str, str]) -> None:
    views = "attention backend views do not cover every physical cache layer"
    entries = record.get("attention_groups")
    if not isinstance(entries, list) or not entries or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError(views)
    for entry in entries:
        _check_audited_class(entry.get("backend_class"), source_files, "backend")
    _check_audited_class(record["cache"]["model_runner"]["class"], source_files, "model runner")
    groups = record["cache"]["groups"]
    seen: set[str] = set()
    for entry in entries:
        index, names = entry.get("kv_cache_group_id"), entry.get("layer_names")
        if (
            type(index) is not int
            or not 0 <= index < len(groups)
            or not isinstance(names, list)
            or any(not isinstance(name, str) for name in names)
            or not set(names) <= set(groups[index]["layer_names"])
        ):
            raise ValueError(f"attention group layers lie outside their cache group: {names}")
        for name in names:
            if name in seen:
                raise ValueError(f"duplicate attention group layer: {name}")
            seen.add(name)
    if seen != {name for group in groups for name in group["layer_names"]}:
        raise ValueError(views)


# The observer records a class as f"{module}.{qualname}", and a qualified name can hold a dot, so any dotted prefix
# of the string can be its module.
def _check_audited_class(name: Any, source_files: Mapping[str, str], role: str) -> None:
    parts = name.split(".") if isinstance(name, str) else []
    if not any(".".join(parts[:end]).replace(".", "/") + ".py" in source_files for end in range(len(parts) - 1, 0, -1)):
        path = "/".join(parts[:-1]) + ".py" if len(parts) > 1 else f"{role} class {name!r}"
        raise ValueError(f"selected {role} source has not been audited: {path}")


# vLLM stores both capacity values in one call (vLLM 0.33.0 v1/core/kv_cache_utils.py:1136-1142), which the engine
# core makes once, before it builds the scheduler (v1/engine/core.py:352). The token capacity is the integer part of
# the concurrency times max_model_len (kv_cache_utils.py:1120-1128). The concurrency is the pool blocks over the
# blocks of one request, a positive float for every layout with a cache group (:1076-1093,
# v1/engine/core.py:337-352).
# The startup concurrency check skips a layout with a sliding-window group, so this check refuses a concurrency that
# is no positive finite number itself, with the text of that check.
def _check_size_tokens(record: dict[str, Any]) -> None:
    cache = record["cache"]
    label = f"{record['phase']} scheduler of DP rank {record['dp_rank']}"
    concurrency = cache.get("kv_cache_max_concurrency")
    if isinstance(concurrency, bool) or not isinstance(concurrency, (int, float)) or not 0 < concurrency < math.inf:
        raise ValueError(f"{label} lacks a positive kv_cache_max_concurrency")
    value = cache.get("kv_cache_size_tokens")
    expected = int(concurrency * record["resolved_config"]["model_config"]["max_model_len"])
    if type(value) is not int or value != expected:
        raise ValueError(
            f"{label} records kv_cache_size_tokens {value!r}, "
            f"but its kv_cache_max_concurrency and max_model_len give {expected}"
        )


# Each worker fits available_memory // bytes_per_block blocks in the budget that its determine_available_memory
# returns (vLLM 0.33.0 v1/core/kv_cache_planning.py:1568-1573, v1/executor/abstract.py:156-157,
# v1/engine/core.py:310), and every worker then takes the smallest count of its engine (kv_cache_planning.py:1583-1595).
# The default builder of CUDA (platforms/interface.py:1474-1493) allocates bytes_per_block * num_blocks
# (kv_cache_planning.py:1715-1718, :681-683), so the allocation over the count gives the bytes of one block.
def _check_worker_block_count(phase: str, dp: int, workers: list[dict[str, Any]]) -> None:
    expected = min(
        worker["cache"]["available_cache_bytes"]
        // (worker["cache"]["allocated_cache_bytes"] // worker["cache"]["num_blocks"])
        for worker in workers
    )
    for worker in workers:
        count = worker["cache"]["num_blocks"]
        if count != expected:
            raise ValueError(
                f"{phase} worker block count of DP rank {dp} contradicts the profiled cache budgets: "
                f"{count} blocks recorded, {expected} fit"
            )


# The observer writes every layer spec of a group that is not uniform from the one group spec
# (runtime/fpm_memory_observer.py, cache_groups). The scheduler group of a uniform worker group is one inner spec
# (vLLM 0.33.0 v1/core/kv_cache_utils.py:1110-1116), so a scheduler group holds equal layer specs too.
def _check_layer_specs(record: dict[str, Any]) -> None:
    for group in record["cache"]["groups"]:
        if group["spec_type"] == _UNIFORM_SPEC:
            continue
        specs = list(group["layer_specs"].values())
        if any(canonical_json(spec) != canonical_json(specs[0]) for spec in specs) or any(
            canonical_json(spec.get("dtype")) != canonical_json(group.get("dtype")) for spec in specs
        ):
            raise ValueError(f"cache group layer specs contradict its group spec: {group['layer_names']}")


_MAMBA_STATE_FIELDS = (
    "mamba_cache_mode",
    "num_speculative_blocks",
    "num_prefill_checkpoint_blocks",
    "prefill_checkpoint_alignment",
    "shapes",
    "dtypes",
)


# The observer writes both blocks from the one KVCacheConfig of the process, in one call and with one serializer
# (runtime/instrumentation/observer.py, observe and _raw_cache; runtime/fpm_memory_observer.py, cache_groups,
# _layer_strided_allocations and _json_value). A raw group holds a spec field only when the spec has it: a ring holds
# no sliding_window, attention_chunk_size or cache_role, which the cache record gives as None. The raw block of a
# scheduler holds the tensor allocations, which its cache record does not: vLLM builds the scheduler configuration as
# a deep copy of the first worker's (vLLM 0.33.0 v1/core/kv_cache_utils.py:1109-1117), and with one pipeline stage,
# the only one that the import accepts, every worker of an engine core is planned from the groups of the whole model
# with the smallest block count (v1/core/kv_cache_planning.py:1436-1437, :1505-1508, :1583-1595), one spec for each
# layer in all workers (:1620-1628). The raw fields that the cache record does not hold (kv_cache_layout; head_size,
# num_kv_heads, kv_quant_mode; the padded page of a group that is no MambaSpec) are not compared, and neither is
# extra_retained_tokens: the cache record gives 0 for a class without the field, whose raw group holds no key, and the
# retention check of the import requires 0.
def _check_raw_cache_config(record: dict[str, Any], workers: Sequence[dict[str, Any]]) -> None:
    def refuse(what: str) -> ValueError:
        return ValueError(f"{label}: raw cache configuration contradicts the cache record: {what}")

    label = f"{record['phase']} {record['kind']} of DP rank {record['dp_rank']}"
    if record["kind"] == "worker":
        label += f" and TP rank {record['tp_rank']}"

    raw, cache = record.get("raw_cache_config"), record["cache"]
    if not isinstance(raw, dict):
        raise refuse("missing")
    if "num_blocks" not in raw or canonical_json(raw["num_blocks"]) != canonical_json(cache["num_blocks"]):
        raise refuse("num_blocks")
    groups = raw.get("groups")
    if not isinstance(groups, list) or len(groups) != len(cache["groups"]):
        raise refuse("group count")
    for index, (entry, group) in enumerate(zip(groups, cache["groups"], strict=True)):
        if not isinstance(entry, dict):
            raise refuse(f"group {index} layer_names")
        for field, value in (
            ("layer_names", group["layer_names"]),
            ("spec_type", group["spec_type"]),
            ("block_size", group["block_size_tokens"]),
            ("page_size_bytes", group["spec_page_size_bytes"]),
        ):
            if field not in entry or canonical_json(entry[field]) != canonical_json(value):
                raise refuse(f"group {index} {field}")
        if group["spec_type"] == _UNIFORM_SPEC:
            continue
        first = group["layer_specs"][group["layer_names"][0]]
        expected: list[tuple[str, Any]] = [(field, first.get(field)) for field in ("tokens_per_state", "cache_role")]
        if group["spec_type"] != _MAMBA_SPEC:
            expected += [(field, group.get(field)) for field in ("dtype", "sliding_window", "attention_chunk_size")]
        for field, value in expected:
            if canonical_json(entry.get(field)) != canonical_json(value):
                raise refuse(f"group {index} {field}")
        if group["spec_type"] == _MAMBA_SPEC:
            state = group.get("state")
            for field in _MAMBA_STATE_FIELDS:
                if (
                    field not in entry
                    or not isinstance(state, dict)
                    or field not in state
                    or canonical_json(entry[field]) != canonical_json(state[field])
                ):
                    raise refuse(f"group {index} {field}")
    allocations = raw.get("tensor_allocations")
    if not isinstance(allocations, list) or any(
        canonical_json(allocations)
        != canonical_json(
            [
                {key: value for key, value in item.items() if key != "storage_id"}
                for item in worker["cache"]["tensor_allocations"]
            ]
        )
        for worker in workers
    ):
        raise refuse("tensor_allocations")


# MambaSpec.page_size_bytes is page_size_padded when it is set, else the bytes of the state tensors (vLLM 0.33.0
# v1/kv_cache_interface.py:1116-1129); the padded size comes from the cache configuration
# (model_executor/layers/mamba/abstract.py:75-80) or from the grouping (v1/core/kv_cache_planning.py:380-389,
# :1105-1108). A uniform group holds a Mamba layer only beside specs that the first
# MambaSpec accepts, each with its page (kv_cache_interface.py:1155-1166, :1347-1361), and the import refuses a group
# that mixes a MambaSpec with another class. _check_raw_cache_config has passed the raw groups.
def _check_mamba_pages(record: dict[str, Any]) -> None:
    for group, entry in zip(record["cache"]["groups"], record["raw_cache_config"]["groups"], strict=True):
        if group["spec_type"] == _MAMBA_SPEC:
            # A raw group without the key gives no page.
            padded = entry.get("page_size_padded") if "page_size_padded" in entry else "absent"
            state = group.get("state")
            page = state.get("real_page_size_bytes") if padded is None and isinstance(state, dict) else padded
            if type(page) is not int or page != group["spec_page_size_bytes"]:
                raise ValueError(f"Mamba group page contradicts its padded page: {group['layer_names']}")
        specs = list(group["layer_specs"].values())
        if any(spec.get("spec_type") == _MAMBA_SPEC for spec in specs) and any(
            canonical_json(spec.get("page_size_bytes")) != canonical_json(specs[0].get("page_size_bytes"))
            for spec in specs
        ):
            raise ValueError(f"Mamba layers of one cache group disagree on their page: {group['layer_names']}")
