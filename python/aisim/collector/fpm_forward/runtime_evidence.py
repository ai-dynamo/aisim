# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence checks of one vLLM 0.33.0 phase that stand apart from the older import checks.

The import runs them after every older check of a phase, so an older refusal keeps its text.
"""

from __future__ import annotations

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


# vLLM stores both capacity values in one call (vLLM 0.33.0 v1/core/kv_cache_utils.py:1136-1142), which the engine
# core makes once, before it builds the scheduler (v1/engine/core.py:352). The token capacity is the integer part of
# the concurrency times max_model_len (kv_cache_utils.py:1120-1128). The import has checked the concurrency before.
def _check_size_tokens(record: dict[str, Any]) -> None:
    cache = record["cache"]
    value = cache.get("kv_cache_size_tokens")
    expected = int(cache["kv_cache_max_concurrency"] * record["resolved_config"]["model_config"]["max_model_len"])
    if type(value) is not int or value != expected:
        raise ValueError(
            f"{record['phase']} scheduler of DP rank {record['dp_rank']} records kv_cache_size_tokens {value!r}, "
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
