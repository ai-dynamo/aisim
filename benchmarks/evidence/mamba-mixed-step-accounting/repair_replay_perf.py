# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matched native CLI comparison. Usage: repair_replay_perf.py PYTHON BASE_SDK ROUNDS OUT.json

PYTHON imports the candidate; BASE_SDK contains the baseline's installed packages.
Each child's package, model constructor and native import paths are checked so
an empty or incorrect BASE_SDK cannot silently select the candidate installation.
Build both native extensions with identical optimized flags before running this.
All generated configs and CLI output live below TMPDIR. Mamba qualification is a
test-only constructor shim, as in telem.py; production defaults are not changed.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

python, base_sdk, rounds, output = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
if rounds <= 0 or rounds % 2:
    sys.exit("ROUNDS must be a positive even number to balance build positions")
if not base_sdk.strip():
    sys.exit("BASE_SDK must name the baseline's installed-packages directory")
base_sdk = Path(base_sdk).resolve()
identity_packages = {"package_path": "aisim", "constructor_path": "aisim_core", "native_path": "aisim"}
shim = """import atexit, hashlib, json, os, sys
import aisim
import aisim_core.sdk.models.nemotron_h as nh
from aisim import _runtime
from pathlib import Path
real_ops, calls = nh.ops, []
class QualifiedOps:
    def __getattr__(self, name):
        if name != "Mamba2Kernel":
            return getattr(real_ops, name)
        def construct(*args, **kwargs):
            kwargs.setdefault("state_dtype", "bfloat16")
            kwargs.setdefault("token_width", 1)
            calls.append(1)
            return real_ops.Mamba2Kernel(*args, **kwargs)
        return construct
nh.ops = QualifiedOps()
def record():
    identity = {
        "calls": len(calls),
        "package_path": str(Path(aisim.__file__).resolve()),
        "constructor_path": str(Path(nh.__file__).resolve()),
        "native_path": str(Path(_runtime.__file__).resolve()),
        "native_sha256": hashlib.sha256(Path(_runtime.__file__).read_bytes()).hexdigest(),
        "native_sources_sha256": hashlib.sha256(_runtime._build_source_files_json().encode()).hexdigest(),
    }
    with open(os.environ["REPAIR_BUILD_RECORD"], "a") as out:
        out.write(json.dumps(identity) + "\\n")
atexit.register(record)
"""
base = {
    "traffic": {
        "source": {"type": "synthetic", "input_tokens": 2048, "cached_prefix_tokens": 512, "output_tokens": 32},
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
        "ais_perf_model": {"estimation_mode": "op_level", "fallback_policy": "deny", "database_mode": "HYBRID"},
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
                "kv_cache": {"block_size": 64, "prefix_caching": True, "capacity": {"type": "fixed", "blocks": 8192}},
            }
        },
    },
}
result = {"config": base, "qualification": "test-only BF16, width one", "rounds": []}
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
    (root / "sitecustomize.py").write_text(shim)
    for round_index in range(rounds):
        order = ["baseline", "candidate"] if round_index % 2 == 0 else ["candidate", "baseline"]
        record = {"order": order, "results": {}}
        for build in order:
            record["results"][build] = {}
            for case, model in (("nano", base["engine"]["model"]), ("control", "Qwen/Qwen3-0.6B")):
                config = json.loads(json.dumps(base))
                config["engine"]["model"] = model
                config_path = root / "config.json"
                config_path.write_text(json.dumps(config))
                dest = root / f"{round_index}-{build}-{case}"
                build_record = root / "build.jsonl"
                build_record.write_text("")
                env = os.environ | {"AIC_ALLOW_UNLISTED_VERSIONS": "1", "REPAIR_BUILD_RECORD": str(build_record)}
                env["PYTHONPATH"] = os.pathsep.join([str(root)] + ([str(base_sdk)] if build == "baseline" else []))
                start = time.perf_counter()
                proc = subprocess.run(
                    [
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
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                )
                elapsed = time.perf_counter() - start
                if proc.returncode:
                    sys.exit(proc.stderr)
                identities = [json.loads(line) for line in build_record.read_text().splitlines()]
                if not identities:
                    sys.exit(f"{build}: no child build identity recorded")
                if build == "baseline":
                    for identity in identities:
                        for key, package in identity_packages.items():
                            loaded = identity.get(key)
                            if not loaded or not Path(loaded).resolve().is_relative_to(base_sdk / package):
                                sys.exit(f"baseline: {key} imported outside BASE_SDK/{package}: {loaded}")
                report = json.loads((dest / "prediction.json").read_text())
                resources = json.loads((dest / "resource-runtime.json").read_text())
                assert report["completed_requests"] == 1000
                assert report["total_input_tokens"] == 2_048_000 and report["total_output_tokens"] == 32_000
                if case == "nano":
                    assert any(identity["calls"] > 0 for identity in identities)
                assert len({identity["native_sha256"] for identity in identities}) == 1
                record["results"][build][case] = {
                    **{key: report[key] for key in metrics},
                    "cli_seconds": elapsed,
                    "peak_rss_bytes": resources["peak_observed_rss_bytes"],
                    "build": identities,
                }
        for key in identity_packages:
            paths_by_build = {
                build: {
                    identity[key]
                    for case in ("nano", "control")
                    for identity in record["results"][build][case]["build"]
                }
                for build in ("baseline", "candidate")
            }
            if paths_by_build["baseline"] & paths_by_build["candidate"]:
                sys.exit(f"baseline: {key} resolves to the candidate installation")
        assert (
            record["results"]["baseline"]["control"]["duration_ms"]
            == record["results"]["candidate"]["control"]["duration_ms"]
        )
        result["rounds"].append(record)
result["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
Path(output).write_text(json.dumps(result, indent=1) + "\n")
print(f"Verified {rounds} balanced rounds, 1000 completed requests per build and case")
