# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure one release build. Run once per revision with the same trace subset."""

import argparse
import hashlib
import json
import subprocess
import time
from contextlib import ExitStack
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

    # At TP16 the K3 state spans 384-token physical pages. Both hash units divide the page.
    cache = {
        "block_size": 384,
        "prefix_match_unit": args.prefix_unit,
        "prefix_caching": True,
        "capacity": {"type": "fixed", "blocks": 4096},
        "state_cache": {},
    }
    worker = {
        "parallelism": {"replicas": 1, "tensor": 16, "moe_tensor": 16, "moe_expert": 1},
        "scheduler": {"max_batched_tokens": 8192, "max_sequences": 128},
        "kv_cache": cache,
        "timing": {"type": "default", "database_mode": "SOL", "estimation_mode": "op_level", "fallback_policy": "deny"},
    }
    config = CorePredictionConfig.model_validate(
        {
            "engine": {
                "model": "moonshotai/Kimi-K3",
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
        if args.step:
            input()
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


def compare(args):
    """Alternate warm runs on two native builds and reject any difference in the simulated report."""
    command = [
        str(Path(__file__).resolve()),
        "run",
        str(args.trace),
        "--step",
        "--lanes",
        str(args.lanes),
        "--prefix-unit",
        str(args.prefix_unit),
        "--seconds",
        str(args.seconds),
        "--repeats",
        str(args.repeats),
    ]
    with ExitStack() as stack:
        workers = {}
        for label, python in (("baseline", args.baseline_python), ("candidate", args.candidate_python)):
            worker = stack.enter_context(
                subprocess.Popen(
                    [str(python), *command],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                )
            )
            stack.callback(worker.terminate)
            workers[label] = worker
        for iteration in range(args.repeats + 1):
            order = ("candidate", "baseline") if iteration % 2 == 0 else ("baseline", "candidate")
            results = {}
            for label in order:
                worker = workers[label]
                worker.stdin.write("\n")
                worker.stdin.flush()
                line = worker.stdout.readline()
                if not line:
                    raise RuntimeError(f"{label} exited with status {worker.wait()}")
                results[label] = json.loads(line)
            if results["baseline"]["report_sha256"] != results["candidate"]["report_sha256"]:
                raise ValueError(f"replay reports differ at iteration {iteration}")
            print(json.dumps(results), flush=True)
        for label, worker in workers.items():
            if worker.wait():
                raise RuntimeError(f"{label} failed after the final replay")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("source", type=Path)
    select.add_argument("target", type=Path)
    bench = argparse.ArgumentParser(add_help=False)
    bench.add_argument("trace", type=Path)
    bench.add_argument("--prefix-unit", type=int, choices=[64, 384], default=384)
    bench.add_argument("--lanes", type=int, required=True)
    bench.add_argument("--seconds", type=int, default=120)
    bench.add_argument("--repeats", type=int, default=3)
    single = commands.add_parser("run", parents=[bench])
    single.add_argument("--step", action="store_true", help=argparse.SUPPRESS)
    paired = commands.add_parser("compare", parents=[bench])
    paired.add_argument("--baseline-python", type=Path, required=True)
    paired.add_argument("--candidate-python", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        select_plays(args.source, args.target)
    elif args.command == "compare":
        compare(args)
    else:
        run(args)
