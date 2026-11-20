# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed-telemetry MTP3 diagnostic. Usage: telem.py OUT.json"""

import json
import sys
import tempfile
from pathlib import Path

if len(sys.argv) != 2:
    sys.exit("usage: telem.py OUT.json")
NEXTN = 3
import aisim_core.sdk.models.nemotron_h as nh

real = nh.ops.Mamba2Kernel


def wrapped(name, scale, kernel, phase, *args, **kw):
    # Keep this historical BF16 diagnostic independent of the checkpoint's
    # resolved FP32 state, now passed by the canonical MTP constructor.
    kw["state_dtype"] = "bfloat16"
    kw["token_width"] = (NEXTN + 1) if phase == "generation" else 1
    return real(name, scale, kernel, phase, *args, **kw)


class _Shim:
    def __init__(self):
        self._o = nh.ops

    def __getattr__(self, n):
        return wrapped if n == "Mamba2Kernel" else getattr(self._o, n)


nh.ops = _Shim()
from aisim_core.sdk.rust_engine_step import RustForwardPassPerfModel

config = {
    "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "system": "b200_sxm",
    "backend": "trtllm",
    "backend_version": "1.3.0rc20",
    "estimation_mode": "op_level",
    "nextn": NEXTN,
    "tp": 1,
    "moe_tp_size": 1,
    "moe_ep_size": 1,
    "fallback_policy": "deny",
    "worker_type": "aggregated",
    "database_mode": "HYBRID",
}
model = RustForwardPassPerfModel.best_available(config)
# Packed op-level decode convention: num_decode_requests = B * (nextn + 1),
# sum_decode_kv_tokens = num_decode_requests * s (s = context incl. current token).
B, S = 8, 1025
decode = {"num_decode_requests": B * (NEXTN + 1), "sum_decode_kv_tokens": B * (NEXTN + 1) * S}
cases = {
    "pure_decode_8req": {"scheduled_requests": dict(decode)},
    "mixed_1x2048_plus_8req": {"scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 2048, **decode}},
}
out = {"config": config, "qualification": {"state_dtype": "bfloat16", "token_width_generation": NEXTN + 1}, "cases": {}}
for name, fpm in cases.items():
    total = model.estimate_forward_pass_time_ms([fpm])
    detail = model.estimate_forward_pass_detailed([fpm])
    detail = json.loads(detail) if isinstance(detail, str) else detail
    ops = detail["ranks"][0]["operations"]
    out["cases"][name] = {
        "forward_pass_metrics": fpm,
        "total_ms": total,
        "mamba_ops": [{k: op.get(k) for k in ("name", "latency_ms", "source")} for op in ops if "mamba" in op["name"]],
        "non_mamba_ms": sum(op["latency_ms"] for op in ops if "mamba" not in op["name"]),
    }
# Use the native snapshot fingerprint of the loaded data, without copying its
# tables. Saving also rejects data changed since this model first loaded it.
out["diagnostics"] = model.diagnostics()
with tempfile.TemporaryDirectory() as temporary:
    snapshot = Path(temporary) / "snapshot"
    model.save_state_snapshot(snapshot)
    out["performance_data"] = json.loads((snapshot / "manifest.json").read_text())["native"]["perf_db"]
with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: v["total_ms"] for k, v in out["cases"].items()}))
