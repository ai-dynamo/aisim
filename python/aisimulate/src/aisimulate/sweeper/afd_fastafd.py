# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AFD layer measurements whose MoE stage comes from a FastAFD split-stage profile."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aisimulate_core.sdk import RustForwardPassPerfModel
from aisimulate_core.sdk.utils import get_model_config_from_model_path

from .afd_parallel import AFDInfeasible, AFDPhase, AFDReasonCategory
from .afd_perfmodel import (
    AFD_MEASUREMENT_API_VERSION,
    AFDLayerTimes,
    AFDMeasurementRequest,
    AFDPerformanceModel,
    AICAFDPerformanceModel,
)

# Per-layer attention components an attention provider must expose for the
# recomposition to be exact: FastAFD runs the router on the A side outside its
# measured stage and folds the top-k combine into that stage.
_ATTENTION_COMPONENTS = ("attention_compute_ms", "a_combine_ms", "a_router_ms")


def _infeasible(detail: str, **provenance: Any) -> AFDInfeasible:
    return AFDInfeasible(AFDReasonCategory.INVALID_MEASUREMENT, detail, provenance=provenance)


@dataclass(frozen=True)
class FastAFDMoEStagePerformanceModel:
    """Combine a measured FastAFD split MoE stage with an attention-side provider.

    The profile entry is an exact measurement of the complete decode MoE stage for
    one ``<A>A<F>F`` topology, per-A-rank batch, MTP degree, microbatch count, and
    layer count. Its latency is spread evenly over ``moe_layers * microbatches``
    pipeline units and becomes ``ffn_ms``; both transfer stages are zero because
    dispatch and return are inside the measured stage. ``attention_ms`` is the
    attention provider's A-side compute without its own combine term, plus the
    router measured at the A-side microbatch. Lookup is exact-only.
    """

    profile_path: str | Path
    moe_backend: str
    attention_model: AFDPerformanceModel
    moe_precision: str | None = None

    def measure(self, request: AFDMeasurementRequest) -> tuple[AFDLayerTimes, ...]:
        import aisimulate_core

        topology = request.topology
        if topology.comm_overhead_factor != 1.0:
            raise _infeasible(
                "FastAFD requires comm_overhead_factor=1; communication is included "
                "in the measured MoE stage and cannot be calibrated separately",
                comm_overhead_factor=topology.comm_overhead_factor,
            )
        try:
            # Reject unsupported pools before invoking an expensive attention
            # provider, using the canonical model's same Rust validation.
            aisimulate_core.RustForwardPassPerfModel._validate_fastafd_topology(
                "decode" if topology.phase is AFDPhase.DECODE else "aggregated",
                topology.tp_a,
                topology.attention_gpus,
                topology.ffn_gpus,
                topology.f_moe_ep_size,
                topology.num_microbatches,
                topology.boundary_on_attn,
            )
        except ValueError as exc:
            raise _infeasible(str(exc)) from exc
        if type(self.attention_model) is AICAFDPerformanceModel:
            # Select precision before the built-in provider constructs either
            # pool. F memory must describe this stage, not checkpoint defaults.
            model_layers = get_model_config_from_model_path(request.model_name)["layers"]
            estimate, config, model = self._select(request, model_layers)
            attention = self._attention(request, model)
        else:
            attention = self._attention(request)
            estimate, config, _model = self._select(request, attention.num_layers)
        if attention.num_layers != estimate["num_layers"]:
            raise _infeasible("the attention provider's layer count does not match the selected FastAFD stage")
        components = self._components(attention)
        return (
            AFDLayerTimes(
                phase=AFDPhase.DECODE,
                attention_ms=components["attention_compute_ms"] + components["a_router_ms"],
                ffn_ms=estimate["ffn_ms"],
                a_to_f_ms=0.0,
                f_to_a_ms=0.0,
                num_layers=estimate["num_layers"],
                decode_attention_samples=self._decode_samples(attention),
                provenance={
                    "provider": "fastafd",
                    "api_version": AFD_MEASUREMENT_API_VERSION,
                    "units": "milliseconds_per_layer_microbatch",
                    "composition": {
                        "attention_ms": "attention_compute_ms + a_router_ms",
                        "ffn_ms": "stage_latency_ms / (moe_layers * microbatches)",
                        "a_to_f_ms": "inside the measured stage",
                        "f_to_a_ms": "inside the measured stage",
                    },
                    "stage": estimate["stage"],
                    "config": config,
                    "attention": {
                        **dict(attention.provenance),
                        "components": {**attention.provenance["components"], **components},
                    },
                },
            ),
        )

    def _attention(self, request: AFDMeasurementRequest, stage_model: Any = None) -> AFDLayerTimes:
        measurements = tuple(
            self.attention_model.measure_attention(request, stage_model)
            if type(self.attention_model) is AICAFDPerformanceModel and stage_model is not None
            else self.attention_model.measure(request)
        )
        if (
            len(measurements) != 1
            or not isinstance(measurements[0], AFDLayerTimes)
            or measurements[0].phase is not AFDPhase.DECODE
        ):
            raise _infeasible("the attention provider must return exactly one decode AFDLayerTimes")
        return measurements[0]

    @staticmethod
    def _decode_samples(attention: AFDLayerTimes) -> tuple[tuple[float, int], ...]:
        raw = attention.provenance.get("decode_samples")
        if "decode_samples" not in attention.provenance:
            # Legacy/custom providers without a trace describe constant A time.
            return ()
        if not isinstance(raw, (list, tuple)) or not raw:
            raise _infeasible("the attention provider decode_samples must be a non-empty sequence")
        samples = []
        total_repeat = 0
        compute_sum = combine_sum = router_sum = 0.0
        for sample in raw:
            if not isinstance(sample, Mapping):
                raise _infeasible("invalid attention provider decode sample")
            repeat = sample.get("repeat")
            components = sample.get("components")
            if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
                raise _infeasible("decode sample repeat must be a positive integer")
            if not isinstance(components, Mapping) or any(name not in components for name in _ATTENTION_COMPONENTS):
                raise _infeasible("decode sample must expose the attention boundary components")
            # Validate the boundary for every KV sample, including the
            # router's presence and exact removal of the A-side combine.
            compute = components["attention_compute_ms"]
            combine = components["a_combine_ms"]
            for name in _ATTENTION_COMPONENTS:
                value = components[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise _infeasible(f"decode sample {name} must be a finite non-negative number")
            if components["a_router_ms"] <= 0:
                raise _infeasible("decode sample must include positive A-side router time")
            samples.append((compute + components["a_router_ms"], repeat))
            total_repeat += repeat
            compute_sum += compute * repeat
            combine_sum += combine * repeat
            router_sum += components["a_router_ms"] * repeat
        averaged = FastAFDMoEStagePerformanceModel._components(attention)
        for name, total in zip(_ATTENTION_COMPONENTS, (compute_sum, combine_sum, router_sum), strict=True):
            if not math.isclose(total / total_repeat, averaged[name], rel_tol=1e-9, abs_tol=1e-12):
                raise _infeasible(f"decode sample {name} does not reconcile with its averaged component")
        return tuple(samples)

    @staticmethod
    def _components(attention: AFDLayerTimes) -> dict[str, float]:
        raw = attention.provenance.get("components")
        if not isinstance(raw, Mapping) or any(name not in raw for name in _ATTENTION_COMPONENTS):
            raise _infeasible(
                "the attention provider does not expose the per-layer attention_compute_ms, "
                "a_combine_ms, and a_router_ms components needed to recompose the FastAFD boundary",
                provider=attention.provenance.get("provider"),
            )
        components = {}
        for name in _ATTENTION_COMPONENTS:
            value = raw[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise _infeasible(
                    f"the attention provider component {name} must be a finite non-negative number",
                    field=name,
                    value=value,
                )
            components[name] = float(value)
        if not math.isclose(
            components["attention_compute_ms"] + components["a_combine_ms"],
            attention.attention_ms,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise _infeasible(
                "the attention provider components attention_compute_ms + a_combine_ms "
                "do not reconcile with attention_ms",
                components=components,
                attention_ms=attention.attention_ms,
            )
        if components["a_router_ms"] <= 0.0:
            raise _infeasible(
                "the attention provider measured no router time on the A side; "
                "FastAFD excludes the router from its measured stage, so it cannot be recomposed"
            )
        return components

    def _select(
        self,
        request: AFDMeasurementRequest,
        model_layers: int,
    ) -> tuple[dict[str, Any], dict[str, Any], RustForwardPassPerfModel]:
        topology = request.topology
        try:
            routed_topk = get_model_config_from_model_path(request.model_name).get("topk")
        except (OSError, ValueError, KeyError) as exc:
            raise _infeasible(f"cannot resolve the model's routed top-k: {exc}", model=request.model_name) from exc
        if isinstance(routed_topk, bool) or not isinstance(routed_topk, int) or routed_topk < 1:
            raise _infeasible(
                "the model's routed top-k must be a known positive integer",
                model=request.model_name,
                routed_topk=routed_topk,
            )
        config = {
            "model": request.model_name,
            "system": request.hardware_sku,
            "backend": request.backend,
            "backend_version": request.backend_version,
            "worker_type": "decode" if topology.phase is AFDPhase.DECODE else "aggregated",
            "estimation_mode": "fastafd_stage",
            "tp": topology.tp_a,
            "moe_backend": self.moe_backend,
            "moe_quant_mode": self.moe_precision,
            "nextn": request.nextn,
            "afd": {
                "logical_batch_per_source_rank": topology.a_batch_size,
                "attention_gpus": topology.attention_gpus,
                "ffn_gpus": topology.ffn_gpus,
                "f_moe_ep_size": topology.f_moe_ep_size,
                "microbatches": topology.num_microbatches,
                "moe_layers": model_layers,
                "routed_topk": routed_topk,
                "boundary_on_attn": topology.boundary_on_attn,
            },
            "estimator_config": {"fastafd_stage": {"profile_path": str(Path(self.profile_path).expanduser())}},
        }
        try:
            model = RustForwardPassPerfModel.best_available(config)
            estimate = model.estimate_fastafd_stage(topology.a_batch_size)
        except (ValueError, RuntimeError, OSError) as exc:
            raise _infeasible(
                str(exc),
                routed_topk=routed_topk,
                canonical_config=config,
                logical_batch_per_source_rank=topology.a_batch_size,
            ) from exc
        return estimate, model.diagnostics()["provenance"]["config"], model


__all__ = ["FastAFDMoEStagePerformanceModel"]
