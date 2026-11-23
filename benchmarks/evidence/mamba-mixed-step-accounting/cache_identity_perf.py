# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time identity checks through native mixed calls and aggregate cache hits/misses.

Run from the repository root: cache_identity_perf.py BASE_SDK CANDIDATE_SDK OUT.json.
The baseline SDK differs only in sdk/rust_engine_step.py, restored from d32fb0aa.
Both SDKs must use the same optimized candidate native extension.
"""

import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(output):
    from aisim import _runtime
    from aisim_core.sdk import config, engine, rust_engine_step
    from aisim_core.sdk.backends.factory import get_backend
    from aisim_core.sdk.models import get_model, nemotron_h
    from aisim_core.sdk.perf_database import get_database_view

    sys.path.insert(0, str(Path.cwd() / "python/aisim"))
    from tools.forward_perf_gate.measurement import measure_cold_and_warm

    result = {}
    for label, model_path, nextn in [
        ("nano", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", 3),
        ("qwen", "Qwen/Qwen3-0.6B", 0),
    ]:
        start = time.perf_counter_ns()
        model = get_model(model_path, config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, nextn=nextn), "vllm")
        database = get_database_view("b200_sxm", "vllm", "0.24.0", database_mode="SOL", allow_unlisted_version=True)
        backend = get_backend("vllm")
        setup_us = (time.perf_counter_ns() - start) / 1000
        runtime = config.RuntimeConfig(batch_size=4, isl=4096, prefix=3000, osl=64)

        def mixed():
            return rust_engine_step.estimate_mixed_step_latency_with_rust(
                model, database, ctx_tokens=512, gen_tokens=3, isl=4096, osl=64, prefix=3000
            )

        def agg_hit():
            return backend.run_agg(model, database, runtime, ctx_tokens=512).get_result_dict()["ttft"]

        def agg_miss():
            backend._agg_cache.clear()
            return agg_hit()

        record = {"setup_us": setup_us, "cases": {}}
        for name, call in [("native_mixed", mixed), ("aggregate_miss", agg_miss), ("aggregate_hit", agg_hit)]:
            value, cold, samples, warm = measure_cold_and_warm(call, warmup=1000, iterations=100)
            record["cases"][name] = {
                "prediction": value,
                "first_query_us": cold,
                "warm_samples_us": samples,
                "warm": warm,
            }
        result[label] = record
    result["identity"] = {
        "native_path": str(Path(_runtime.__file__).resolve()),
        "bridge_path": str(Path(rust_engine_step.__file__).resolve()),
        "native_sha256": fingerprint(_runtime.__file__),
        "bridge_sha256": fingerprint(rust_engine_step.__file__),
        "compiler_sha256": fingerprint(engine.__file__),
        "constructor_sha256": fingerprint(nemotron_h.__file__),
    }
    output.write_text(json.dumps(result))


def main():
    baseline, candidate, output = map(Path, sys.argv[1:])
    sdks = {"baseline": baseline.resolve(), "candidate": candidate.resolve()}
    result = {
        "rounds": [],
        "scope": "SDK-to-native mixed calls, aggregate cache misses and hits; no operation edits during timing",
        "python": sys.version,
        "platform": platform.platform(),
        "source_baseline": "d32fb0aa013ae4a0ab98074093d6d71d36fc72c9",
        "source_candidate": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "driver_sha256": fingerprint(__file__),
        "warmup_calls": 1000,
        "timed_calls": 100,
        "variant_difference": (
            "Only sdk/rust_engine_step.py is restored; both use the candidate optimized native extension."
        ),
    }
    with tempfile.TemporaryDirectory() as temporary:
        for index in range(6):
            order = ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
            row = {"order": order, "results": {}}
            for label in order:
                dest = Path(temporary) / f"identity-perf-{index}-{label}.json"
                subprocess.run(
                    [sys.executable, __file__, "worker", str(dest)],
                    env=os.environ | {"PYTHONPATH": str(sdks[label]), "AIC_ALLOW_UNLISTED_VERSIONS": "1"},
                    check=True,
                )
                record = json.loads(dest.read_text())
                for key in ("native_path", "bridge_path"):
                    assert Path(record["identity"][key]).is_relative_to(sdks[label])
                row["results"][label] = record
            identities = [row["results"][label]["identity"] for label in ("baseline", "candidate")]
            assert identities[0]["native_sha256"] == identities[1]["native_sha256"]
            assert identities[0]["constructor_sha256"] == identities[1]["constructor_sha256"]
            assert identities[0]["compiler_sha256"] == identities[1]["compiler_sha256"]
            assert identities[0]["bridge_sha256"] != identities[1]["bridge_sha256"]
            for model in ("nano", "qwen"):
                for case in ("native_mixed", "aggregate_miss", "aggregate_hit"):
                    assert (
                        row["results"]["baseline"][model]["cases"][case]["prediction"]
                        == row["results"]["candidate"][model]["cases"][case]["prediction"]
                    )
            result["rounds"].append(row)
    output.write_text(json.dumps(result, indent=1) + "\n")
    for model in ("nano", "qwen"):
        for case in ("native_mixed", "aggregate_miss", "aggregate_hit"):
            values = {
                label: [r["results"][label][model]["cases"][case]["warm"]["call_median_us"] for r in result["rounds"]]
                for label in ("baseline", "candidate")
            }
            print(model, case, {label: (statistics.median(v), min(v), max(v)) for label, v in values.items()})


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "worker":
        worker(Path(sys.argv[2]))
    else:
        main()
