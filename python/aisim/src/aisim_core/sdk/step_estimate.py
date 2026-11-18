# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward-pass workload and estimate contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from aisim_core.sdk.performance_result import MoECommFallback


@dataclass(frozen=True)
class MixedStepInput:
    """Scheduled work for one mixed prefill/decode engine iteration.

    ``num_decode_requests`` is the number of logical decode sequences. It is
    deliberately distinct from the target-model query-token count, which is
    derived from the model's speculative draft depth.

    For op-level pricing, ``context_tokens`` is the packed prompt budget in
    full prompt tokens:
    ``context_tokens // isl`` complete requests of the effective ``isl``
    tokens, prefix included. A nonzero remainder adds one partial request,
    and an exact multiple of ``isl`` has none. A qualified Mamba op caps that
    partial request at ``isl - prefix`` new tokens. The budget is not the
    count of newly scheduled tokens after prefix-cache hits. A whole-model FPM
    instead reads ``context_tokens`` as new tokens. It derives
    ``ceil(context_tokens / (isl - prefix))`` requests, each with ``prefix``
    cached tokens, and its query holds ``context_tokens`` new tokens plus the
    packed decode tokens. A budget below ``isl - prefix`` prices chunks of
    one request.
    """

    context_tokens: int
    num_decode_requests: int

    def __post_init__(self) -> None:
        if self.context_tokens <= 0:
            raise ValueError("context_tokens must be positive for a mixed step")
        if self.num_decode_requests < 0:
            raise ValueError("num_decode_requests must be non-negative")


@dataclass(frozen=True)
class StepEstimate:
    """Raw wall-time estimate for one scheduled engine iteration."""

    latency_ms: float
    energy_wms: float
    component_latency_ms: dict[str, float] = field(default_factory=dict)
    component_energy_wms: dict[str, float] = field(default_factory=dict)
    per_op_latency_ms: dict[str, float] = field(default_factory=dict)
    per_op_source: dict[str, str] = field(default_factory=dict)
    context_tokens: int = 0
    num_decode_requests: int = 0
    num_decode_query_tokens: int = 0
    moe_comm_fallbacks: tuple[MoECommFallback, ...] = ()
    per_op_energy_wms: dict[str, float] = field(default_factory=dict)
    covered_latency_ms: float = 0.0

    def legacy_tuple(self) -> tuple[float, float, dict[str, float], dict[str, str]]:
        """Return the pre-contract private-helper shape."""
        return self.latency_ms, self.energy_wms, self.per_op_latency_ms, self.per_op_source


__all__ = ["MixedStepInput", "StepEstimate"]
