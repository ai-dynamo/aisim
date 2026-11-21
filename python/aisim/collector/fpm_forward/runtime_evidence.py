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
    for _, record in (*sorted(schedulers.items()), *sorted(workers.items())):
        _check_layer_specs(record)


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
