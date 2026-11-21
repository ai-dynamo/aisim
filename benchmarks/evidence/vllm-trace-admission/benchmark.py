# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure one release build. Run once per revision with the same trace subset."""

import argparse
import hashlib
import json
import time
from pathlib import Path


def context_lengths(play):
    for turn in play["requests"]:
        if "in" in turn:
            yield turn["in"] + turn["out"]
        if "requests" in turn:
            yield from context_lengths(turn)


def select_plays(source, target):
    # Copy whole plays, including child requests, without changing recorded tokens or times.
    with source.open("rb") as rows, target.open("xb") as out:
        count = 0
        for line in rows:
            if max(context_lengths(json.loads(line)), default=0) <= 262144:
                out.write(line)
                count += 1
                if count == 32:
                    return
    raise ValueError(f"only {count} eligible plays in {source}")


def run(args):
    from aisim import _runtime
    from aisim.compiler import prediction_to_replay_spec
    from aisim.config.cli import CorePredictionConfig
    from aisim.runner import _materialize_engine_execution_spec

    hybrid = args.model == "hybrid"
    tp = 4 if hybrid else 8
    cache = {"block_size": 16, "prefix_caching": True, "capacity": {"type": "fixed", "blocks": 98304}}
    if hybrid:
        # TP4: 16 full-attention layers, 48 Gated DeltaNet layers. BF16 convolution and FP32 recurrent state.
        cache.update(bytes_per_token=16384, prefix_match_unit=16, state_cache={"bytes_per_request": 38486016})
    worker = {
        "parallelism": {"replicas": 1, "tensor": tp, "moe_tensor": tp, "moe_expert": 1},
        "scheduler": {"max_batched_tokens": 8192, "max_sequences": 128},
        "kv_cache": cache,
        "timing": {"type": "default", "database_mode": "SOL", "estimation_mode": "op_level", "fallback_policy": "deny"},
    }
    config = CorePredictionConfig.model_validate(
        {
            "engine": {
                "model": "Qwen/Qwen3.5-27B" if hybrid else "Qwen/Qwen3-Coder-480B-A35B-Instruct",
                "backend": "vllm",
                "backend_version": "0.24.0",
                "hardware": "gb200",
                "mode": "aggregated",
                "context_length": 262144,
                "workers": {"aggregated": worker},
            },
            "traffic": {
                "source": {
                    "type": "trace",
                    "format": "weka",
                    "paths": [str(args.trace.resolve())],
                    "block_size": 64,
                    "client": "agentx",
                    "nested_timestamp_basis": "absolute",
                },
                "load": {
                    "type": "trace_timestamps",
                    "agentic_lanes": args.lanes,
                    "agentic_snapshot": {"seed": 42},
                    "agentic_warmup": True,
                    "agentic_warmup_policy": "live_trajectory_v1",
                    "agentic_profile": {
                        "duration_seconds": args.seconds,
                        "global_idle_cap_seconds": 10,
                        "tree_idle_cap_seconds": 300,
                    },
                },
            },
        }
    )
    spec = _materialize_engine_execution_spec(
        prediction_to_replay_spec(config), trace_block_size=64, record_per_request=True, memory_diagnostics=None
    )
    spec["determinism"] = "canonical_v1"
    spec["traffic"]["trace_graph_cache"] = True
    payload = json.dumps(spec)
    for iteration in range(args.repeats + 1):
        cpu, start = time.process_time(), time.perf_counter()
        raw = _runtime.run_replay_json(payload)
        cpu, wall = time.process_time() - cpu, time.perf_counter() - start
        report = json.loads(raw)
        replay_wall = report.pop("wall_time_ms") / 1000
        # These rates use the simulator's host wall clock as their denominator.
        report.pop("processed_tokens_per_s")
        report.pop("processed_output_tokens_per_s")
        digest = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        print(
            json.dumps(
                {
                    "iteration": iteration,
                    "warmup": iteration == 0,
                    "cpu_s": cpu,
                    "wall_s": wall,
                    "replay_wall_s": replay_wall,
                    "report_sha256": digest,
                    "requests": report["completed_requests"],
                    "output_tokens": report["total_output_tokens"],
                    "lifecycle": report["agentic_lifecycle_digest"],
                    "prefix_cache_reused_ratio": report["prefix_cache_reused_ratio"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("source", type=Path)
    select.add_argument("target", type=Path)
    bench = commands.add_parser("run")
    bench.add_argument("trace", type=Path)
    bench.add_argument("--model", choices=["coder", "hybrid"], required=True)
    bench.add_argument("--lanes", type=int, required=True)
    bench.add_argument("--seconds", type=int, default=180)
    bench.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.command == "select":
        select_plays(args.source, args.target)
    else:
        run(args)
