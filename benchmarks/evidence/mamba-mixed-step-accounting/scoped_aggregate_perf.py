# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare aggregate callers on one optimized wheel with an unchanged native oracle.

Usage: scoped_aggregate_perf.py PINNED_BASE_BACKEND.py BASE_SHA CANDIDATE_SHA OUT.json
The baseline replaces only the three changed BaseBackend methods with the pinned
source. Native code and all other installed sources are shared. This measures
aggregate cache misses with a warm native handle, then aggregate cache hits;
it does not measure first-time native compilation or serving accuracy.
"""

import hashlib
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import aisim._runtime
from aisim_core.sdk import config
from aisim_core.sdk.backends.base_backend import BaseBackend
from aisim_core.sdk.backends.factory import get_backend
from aisim_core.sdk.models import get_model
from aisim_core.sdk.perf_database import get_database_view


def measure(backend, model, database, runtime):
    backend._agg_cache.clear()
    start = time.perf_counter_ns()
    summary = backend.run_agg(model, database, runtime, ctx_tokens=512)
    miss_us = (time.perf_counter_ns() - start) / 1000
    start = time.perf_counter_ns()
    for _ in range(200):
        assert backend.run_agg(model, database, runtime, ctx_tokens=512) is summary
    hit_us = (time.perf_counter_ns() - start) / 200_000
    row = summary.get_result_dict()
    group = summary.get_aggregate_energy_breakdown()["mix_step"]
    return {
        "miss_us": miss_us,
        "hit_us": hit_us,
        "prediction": {
            "mixed_steps": summary.get_step_estimates()["scheduling"]["num_mix_steps"],
            # Use the same scalar type in this diagnostic reduction. Python
            # 3.12's float sum and numpy scalar accumulation differ by an ulp.
            "weighted_mamba_ms": sum(float(v) for k, v in group.per_op_latency_ms.items() if "mamba" in k),
            "ttft_ms": row["ttft"],
            "tpot_ms": row["tpot"],
            "tokens_s": row["tokens/s"],
        },
    }


def verify_source(path, revision):
    source = "python/aisim/src/aisim_core/sdk/backends/base_backend.py"
    revision = subprocess.check_output(["git", "rev-parse", "--verify", f"{revision}^{{commit}}"], text=True).strip()
    expected = subprocess.check_output(["git", "show", f"{revision}:{source}"])
    if Path(path).read_bytes() != expected:
        raise ValueError(f"backend source {path} does not match {revision}")
    return revision


def main():
    baseline_path, baseline_sha, candidate_sha, output = sys.argv[1:]
    baseline_sha = verify_source(baseline_path, baseline_sha)
    candidate_sha = verify_source(sys.modules[BaseBackend.__module__].__file__, candidate_sha)
    spec = importlib.util.spec_from_file_location("scoped_aggregate_baseline", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    results = {}
    for label, model_path, nextn, prefix in (
        ("mtp_prefix3072", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", 3, 3072),
        (
            "mtp_prefix3000_tail72",
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            3,
            3000,
        ),
        ("mtp_cold_control", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", 3, 0),
        ("non_mamba_control", "Qwen/Qwen3-0.6B", 0, 3072),
    ):
        model = get_model(
            model_path,
            config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, nextn=nextn),
            "vllm",
        )
        database = get_database_view("b200_sxm", "vllm", "0.24.0", database_mode="SOL")
        assert database is not None
        candidate = get_backend("vllm")
        original = type(
            "PinnedVLLMBackend",
            (type(candidate),),
            {
                name: getattr(baseline.BaseBackend, name)
                for name in (
                    "run_agg",
                    "_run_agg_mixed_step",
                    "_schedule_qualified_mamba_prefill",
                )
            },
        )()
        backends = {"baseline": original, "candidate": candidate}
        runtime = config.RuntimeConfig(batch_size=4, isl=4096, prefix=prefix, osl=64)
        # Prime model compilation, native query caches and summary presentation.
        for backend in backends.values():
            measure(backend, model, database, runtime)
        rounds = []
        for index in range(6):
            order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
            samples = {side: [] for side in order}
            for _ in range(30):
                for side in order:
                    samples[side].append(measure(backends[side], model, database, runtime))
            rounds.append(
                {"order": list(order)}
                | {
                    side: {
                        metric: statistics.median(sample[metric] for sample in samples[side])
                        for metric in ("miss_us", "hit_us")
                    }
                    for side in order
                }
            )
        predictions = {side: samples[side][-1]["prediction"] for side in backends}
        if label.endswith("control"):
            assert predictions["baseline"] == predictions["candidate"]
        results[label] = {
            "model": model_path,
            "nextn": nextn,
            "prefix": prefix,
            "rounds": rounds,
            **predictions,
        }
    Path(output).write_text(
        json.dumps(
            {
                "baseline": baseline_sha,
                "candidate": candidate_sha,
                "baseline_backend_sha256": hashlib.sha256(Path(baseline_path).read_bytes()).hexdigest(),
                "candidate_backend_sha256": hashlib.sha256(
                    Path(sys.modules[BaseBackend.__module__].__file__).read_bytes()
                ).hexdigest(),
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "native_sha256": hashlib.sha256(Path(aisim._runtime.__file__).read_bytes()).hexdigest(),
                "python": platform.python_version(),
                "inputs": {
                    "system": "b200_sxm",
                    "backend": "vllm",
                    "version": "0.24.0",
                    "mode": "SOL",
                    "batch_size": 4,
                    "isl": 4096,
                    "osl": 64,
                    "ctx_tokens": 512,
                },
                "scope": "aggregate miss (warm native handle) and aggregate hit, excluding result formatting",
                "cases": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
