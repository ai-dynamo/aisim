# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical MTP CLI comparison.

Usage: canonical_mtp_perf.py PYTHON BASE_SDK ROUNDS OUT.json REPO BASE_REV CANDIDATE_REV

PYTHON imports the candidate and BASE_SDK contains the pinned baseline packages.
Use identical optimized native extensions and dependencies. There are no Mamba
constructor shims. The Nano case uses canonical MTP3; Qwen3-0.6B without MTP is
an unchanged control. Generated configuration and output stay below TMPDIR.
"""

import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from committed_sources import package_inventory

if len(sys.argv) != 8:
    sys.exit(__doc__)
python, base_sdk, rounds, output = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    sys.argv[4],
)
repo, base_revision, candidate_revision = sys.argv[5:]
inventories = {
    "baseline": package_inventory(repo, base_revision),
    "candidate": package_inventory(repo, candidate_revision),
}
if rounds <= 0 or rounds % 2:
    sys.exit("ROUNDS must be a positive even number to balance build positions")
record_identity = """import atexit, hashlib, json, os
from pathlib import Path
import aisim_core.sdk.models.nemotron_h as nh
from aisim import _runtime
import aisim, aisim_core
from committed_sources import verify_imported_packages
def record():
    with open(os.environ["REPAIR_SOURCE_INVENTORY"]) as source:
        provenance = verify_imported_packages(json.load(source), [aisim, aisim_core])
    identity = {
        **provenance,
        "constructor_sha256": hashlib.sha256(Path(nh.__file__).read_bytes()).hexdigest(),
        "native_sha256": hashlib.sha256(Path(_runtime.__file__).read_bytes()).hexdigest(),
        "native_sources_sha256": hashlib.sha256(_runtime._build_source_files_json().encode()).hexdigest(),
    }
    with open(os.environ["REPAIR_BUILD_RECORD"], "a") as out:
        out.write(json.dumps(identity) + "\\n")
atexit.register(record)
"""
base = {
    "traffic": {
        "source": {
            "type": "synthetic",
            "input_tokens": 2048,
            "cached_prefix_tokens": 512,
            "output_tokens": 32,
        },
        "load": {"type": "concurrency", "concurrency": 64},
        "stop": {"requests": 1000},
    },
    "engine": {
        "mode": "aggregated",
        "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "hardware": "b200_sxm",
        "backend": "trtllm",
        "backend_version": "1.3.0rc20",
        "context_length": 8192,
        "ais_perf_model": {
            "estimation_mode": "op_level",
            "fallback_policy": "deny",
            "database_mode": "HYBRID",
        },
        "workers": {
            "aggregated": {
                "parallelism": {
                    "replicas": 1,
                    "tensor": 1,
                    "pipeline": 1,
                    "attention_data": 1,
                    "moe_tensor": 1,
                    "moe_expert": 1,
                },
                "scheduler": {"max_batched_tokens": 4096, "max_sequences": 64},
                "kv_cache": {
                    "block_size": 64,
                    "prefix_caching": True,
                    "capacity": {"type": "fixed", "blocks": 8192},
                },
            }
        },
    },
}
result = {
    "config": base,
    "qualification": "canonical MTP3, checkpoint-declared FP32 state; control has no MTP",
    "python_revisions": {side: inventory["revision"] for side, inventory in inventories.items()},
    "native_scope": "Both Python revisions share the candidate native oracle; its hashes are recorded separately.",
    "host": {
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        ),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    },
    "mode": "CPU native engine simulation, not accelerator measurements",
    "env_overrides": {"AIC_ALLOW_UNLISTED_VERSIONS": "1"},
    "rounds": [],
}
raw_requests = {}
metrics = (
    "completed_requests",
    "total_input_tokens",
    "total_output_tokens",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "duration_ms",
    "wall_time_ms",
)
with tempfile.TemporaryDirectory(prefix="mamba-repair-replay-") as temp:
    root = Path(temp)
    (root / "sitecustomize.py").write_text(record_identity)
    for round_index in range(rounds):
        order = ["baseline", "candidate"] if round_index % 2 == 0 else ["candidate", "baseline"]
        record = {"order": order, "results": {}}
        for build in order:
            record["results"][build] = {}
            for case, model in (
                ("nano", base["engine"]["model"]),
                ("control", "Qwen/Qwen3-0.6B"),
            ):
                config = json.loads(json.dumps(base))
                config["engine"]["model"] = model
                if case == "nano":
                    config["engine"].update(nextn=3, nextn_accepted=2.2)
                config_path = root / "config.json"
                config_path.write_text(json.dumps(config))
                dest = root / f"{round_index}-{build}-{case}"
                build_record = root / "build.jsonl"
                build_record.write_text("")
                env = os.environ | {
                    "AIC_ALLOW_UNLISTED_VERSIONS": "1",
                    "REPAIR_BUILD_RECORD": str(build_record),
                }
                inventory_path = root / "source-inventory.json"
                inventory_path.write_text(json.dumps(inventories[build]))
                env["REPAIR_SOURCE_INVENTORY"] = str(inventory_path)
                env["PYTHONPATH"] = os.pathsep.join(
                    [str(root), str(Path(__file__).resolve().parent)] + ([base_sdk] if build == "baseline" else [])
                )
                # Check imports before timing or invoking the workload. The
                # record() exit hook also verifies every measured child.
                subprocess.run([python, "-c", record_identity + "\nrecord()"], env=env, check=True)
                build_record.write_text("")
                start = time.perf_counter()
                command = [
                    python,
                    "-m",
                    "aisim",
                    "predict",
                    "--stack",
                    "engine",
                    "-c",
                    str(config_path),
                    "--output-dir",
                    str(dest),
                    "--format",
                    "json",
                    "--capture-per-request",
                ]
                proc = subprocess.run(
                    command,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                elapsed = time.perf_counter() - start
                if proc.returncode:
                    sys.exit(proc.stderr)
                report = json.loads((dest / "prediction.json").read_text())
                resources = json.loads((dest / "resource-runtime.json").read_text())
                requests_bytes = (dest / "requests.jsonl").read_bytes()
                requests = [json.loads(line) for line in requests_bytes.splitlines()]
                assert len(requests) == report["completed_requests"]
                requests_sha = hashlib.sha256(requests_bytes).hexdigest()
                raw_requests.setdefault(requests_sha, requests)
                identities = [json.loads(line) for line in build_record.read_text().splitlines()]
                if not identities:
                    sys.exit(f"{build}: no verified build identity recorded")
                assert report["completed_requests"] == 1000
                assert report["total_input_tokens"] == 2_048_000 and report["total_output_tokens"] == 32_000
                assert len({identity["native_sha256"] for identity in identities}) == 1
                record["results"][build][case] = {
                    "config": config,
                    "command": command,
                    "correctness": "1000 completions; input/output totals match; control duration equal",
                    "requests_sha256": requests_sha,
                    **{key: report[key] for key in metrics},
                    "cli_seconds": elapsed,
                    "peak_rss_bytes": resources["peak_observed_rss_bytes"],
                    "build": identities[-1],
                }
        assert (
            record["results"]["baseline"]["control"]["duration_ms"]
            == record["results"]["candidate"]["control"]["duration_ms"]
        )
        assert (
            record["results"]["baseline"]["nano"]["build"]["constructor_sha256"]
            != record["results"]["candidate"]["nano"]["build"]["constructor_sha256"]
        )
        assert (
            record["results"]["baseline"]["nano"]["build"]["native_sha256"]
            == record["results"]["candidate"]["nano"]["build"]["native_sha256"]
        )
        result["rounds"].append(record)
result["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
companion = Path(output).with_suffix(".requests.json.gz")
companion.write_bytes(gzip.compress(json.dumps(raw_requests, separators=(",", ":")).encode(), mtime=0))
result["raw_requests"] = {
    "path": companion.name,
    "sha256": hashlib.sha256(companion.read_bytes()).hexdigest(),
    "format": "gzip JSON mapping requests.jsonl SHA256 to complete per-request records",
}
Path(output).write_text(json.dumps(result, indent=1) + "\n")
print(f"Verified {rounds} balanced rounds, 1000 completed requests per build and case")
