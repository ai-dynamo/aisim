# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical MTP forward timings. Usage: canonical_mtp_forward.py OUT.json

Run from each installed build, selecting baseline packages with PYTHONPATH when
needed. Uses the forward gate's first-query and warm-query timing boundaries.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3] / "python/aisim"))
from tools.forward_perf_gate.measurement import measure_cold_and_warm

from aisim import _runtime
from aisim_core.sdk.models import nemotron_h
from aisim_core.sdk.rust_engine_step import RustForwardPassPerfModel

config = {
    "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "system": "b200_sxm",
    "backend": "trtllm",
    "backend_version": "1.3.0rc20",
    "estimation_mode": "op_level",
    "nextn": 3,
    "tp": 1,
    "moe_tp_size": 1,
    "moe_ep_size": 1,
    "fallback_policy": "deny",
    "worker_type": "aggregated",
    "database_mode": "HYBRID",
}
cases = {
    "decode": {"num_decode_requests": 32, "sum_decode_kv_tokens": 32800},
    "mixed_ragged": {
        "num_prefill_requests": 2,
        "sum_prefill_tokens": 17,
        "num_decode_requests": 32,
        "sum_decode_kv_tokens": 32800,
    },
    "prefill_uniform": {"num_prefill_requests": 2, "sum_prefill_tokens": 4096},
    "prefill_ragged": {"num_prefill_requests": 2, "sum_prefill_tokens": 4097},
}
result = {"config": config, "cases": cases, "builds": {}}
for label, model_id, nextn in [("nano", config["model"], 3), ("control", "Qwen/Qwen3-0.6B", 0)]:
    resolved = config | {"model": model_id, "nextn": nextn}
    start = time.perf_counter()
    model = RustForwardPassPerfModel.best_available(resolved)
    setup_ms = (time.perf_counter() - start) * 1000
    model.estimate_forward_pass_time_ms([{"scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 64}}])
    record = {"config": resolved, "setup_ms": setup_ms, "queries": {}}
    for name, scheduled in cases.items():
        payload = [{"scheduled_requests": scheduled}]
        value, cold_us, samples_us, warm = measure_cold_and_warm(
            lambda payload=payload: model.estimate_forward_pass_time_ms(payload), warmup=10, iterations=100
        )
        assert value is not None and value > 0
        record["queries"][name] = {
            "predicted_ms": value,
            "first_query_us": cold_us,
            "warm_samples_us": samples_us,
            "warm": warm,
        }
    result["builds"][label] = record
result["identity"] = {
    "constructor_path": str(Path(nemotron_h.__file__).resolve()),
    "native_path": str(Path(_runtime.__file__).resolve()),
    "constructor_sha256": hashlib.sha256(Path(nemotron_h.__file__).read_bytes()).hexdigest(),
    "native_sha256": hashlib.sha256(Path(_runtime.__file__).read_bytes()).hexdigest(),
}
Path(sys.argv[1]).write_text(json.dumps(result, indent=1) + "\n")
