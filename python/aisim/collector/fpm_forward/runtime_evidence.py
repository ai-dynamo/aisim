# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence checks of one vLLM 0.33.0 phase that stand apart from the older import checks.

The import runs them after every older check of a phase, so an older refusal keeps its text.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .runtime.fpm_memory_observer import _UNIFORM_SPEC
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


# hf_overrides is a launch argument with the default {} (vLLM 0.33.0 config/model.py:348, engine/arg_utils.py:1035),
# which vLLM applies on top of the loaded config.json (config/model.py:616-630). The reviewed launch sets none, and
# the config.json hash does not cover another value. The observer records a callable as a string
# (runtime/instrumentation/observer.py, _resolved).
def _check_hf_overrides(record: dict[str, Any]) -> None:
    value = record["resolved_config"]["model_config"].get("hf_overrides")
    if value is not None and value != {}:
        raise ValueError(f"loaded model configuration overrides require another source mapping: hf_overrides={value!r}")


# The observer refuses a runner or a backend whose source file the bundle does not hash, a layer in two attention
# groups, and attention groups that do not cover every cache layer (runtime/instrumentation/observer.py,
# _layer_views). vLLM builds the attention groups of a cache group from that group's layers, one for each backend and
# layer spec (vLLM 0.33.0 v1/worker/gpu/attn_utils.py:217-265, v1/worker/gpu_model_runner.py:6692-6713).
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
        path = "/".join(parts[:-1]) + ".py" if len(parts) > 1 else repr(name)
        raise ValueError(f"selected {role} source has not been audited: {path}")


# vLLM stores both capacity values in one call (vLLM 0.33.0 v1/core/kv_cache_utils.py:1136-1142), which the engine
# core makes once, before it builds the scheduler (v1/engine/core.py:352). The token capacity is the integer part of
# the concurrency times max_model_len (kv_cache_utils.py:1120-1128). The concurrency is the pool blocks over the
# blocks of one request, a positive float for every layout with a cache group (:1076-1093, engine/core.py:337-352).
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
