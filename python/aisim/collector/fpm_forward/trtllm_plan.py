# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze the two serving points qualified by the rc29 TRT-LLM pilot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

from .config import FPMCollectionOptions
from .native_artifact import TRTLLM_RUNTIME_VERSION
from .planner import BackendPolicy, FPMCell
from .runtime_instrumentation import InstrumentationBundle, canonical_json
from .types import ParallelTopology

MODEL = "Qwen/Qwen3-32B-FP8"
MODEL_REVISION = "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
SYSTEM = "h200_sxm"
DECODE_KV_TOLERANCE = 128
# amd64 manifest of nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc29.
IMAGE_SHA256 = "e9dab9fccc550dcc04356962ff152908d692e0e47ba143f57402a24a4ad64b77"
IMAGE = f"nvcr.io#nvidia/tensorrt-llm/release:sha256:{IMAGE_SHA256}"
POINTS = {
    "prefill": {"batch_size": 4, "total_prefill_tokens": 8192, "total_kv_read_tokens": 0},
    "decode": {"batch_size": 4, "total_prefill_tokens": 0, "total_kv_read_tokens": 32768},
}


@dataclass(frozen=True, slots=True)
class TrtllmCapability:
    support_level: str = "exact"
    template_id: str = "trtllm_observer:qwen3_dense"
    template_version: int = 1
    aic_database_version: str = TRTLLM_RUNTIME_VERSION


@dataclass(frozen=True, slots=True)
class TrtllmPlan:
    options: FPMCollectionOptions
    runtime_instrumentation: InstrumentationBundle
    cells: tuple[FPMCell, ...]
    aic_revision: str
    container_sha256: str
    sha256: str
    capability: TrtllmCapability = TrtllmCapability()
    model_path: str = MODEL
    system: str = SYSTEM
    backend: str = "trtllm"
    backend_version: str = TRTLLM_RUNTIME_VERSION
    runtime_backend_version: str = TRTLLM_RUNTIME_VERSION

    def to_dict(self):
        return {
            "schema_name": "aisim_trtllm_fpm_plan",
            "schema_version": 1,
            "sha256": self.sha256,
            "aic_revision": self.aic_revision,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "system": self.system,
            "model_path": self.model_path,
            "model_revision": MODEL_REVISION,
            "bundle_sha256": self.runtime_instrumentation.sha256,
            "container_sha256": self.container_sha256,
            "points": json.loads(self.options.benchmark_points_json),
            "decode_kv_tolerance": DECODE_KV_TOLERANCE,
            "cells": [{**cell.to_dict(), "point_source": "trtllm_observer"} for cell in self.cells],
        }


def build_plan(
    model: str,
    system: str,
    options: FPMCollectionOptions,
    bundle: InstrumentationBundle,
    aic_revision: str,
    container_sha256: str,
) -> TrtllmPlan:
    if (model, system) != (MODEL, SYSTEM) or not options.measure_power or options.benchmark_points_json is None:
        raise ValueError("TRT-LLM FPM requires Qwen/Qwen3-32B-FP8, h200_sxm, power, and explicit points")
    data = json.loads(options.benchmark_points_json)
    if any(data[phase] != [point] for phase, point in POINTS.items()):
        raise ValueError("TRT-LLM FPM requires one pinned prefill point and one pinned decode point")
    if options.max_gpus != 2:
        raise ValueError("TRT-LLM FPM requires --fpm-max-gpus 2")
    if options.worker_type not in (None, "aggregated") or options.decoder_replay:
        raise ValueError("TRT-LLM FPM supports only the two non-speculative serving points")
    options = replace(options, warmup_iterations=1)
    launch_identity = {
        "aic_revision": aic_revision,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "system": SYSTEM,
        "trtllm": TRTLLM_RUNTIME_VERSION,
        "container_sha256": container_sha256,
        "bundle_sha256": bundle.sha256,
        "points": data,
        "decode_kv_tolerance": DECODE_KV_TOLERANCE,
    }
    digest = hashlib.sha256(canonical_json(launch_identity).encode()).hexdigest()
    cells = tuple(
        FPMCell(
            cell_id=f"trtllm-{phase}-{digest[:16]}",
            workload_kind=phase,
            topology=ParallelTopology(tp=2, pp=1, dp=1, moe_tp=2, moe_ep=1, cp=1),
            backend_policy=BackendPolicy("baseline_auto", {}, {}),
            parallel_strategy="tp",
            weight_quantization="fp8_block",
            gemm_quant_mode="fp8_block",
            kv_cache_dtype="bfloat16",
            fmha_quant_mode="bfloat16",
            moe_quant_mode="bfloat16",
            comm_quant_mode="half",
        )
        for phase in POINTS
    )
    return TrtllmPlan(options, bundle, cells, aic_revision, container_sha256, digest)


BUNDLED_MANIFEST = Path(__file__).parent / "runtime" / "trtllm" / "manifest.json"
