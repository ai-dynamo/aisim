# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixed-step Mamba prefill shapes through InferenceSession.run_agg. Usage: agg_caller.py OUT.json"""

import json
import sys

import aisim_core.sdk.models.nemotron_h as nh

real = nh.ops.Mamba2Kernel


def wrapped(name, scale, kernel, phase, *args, **kw):
    kw.setdefault("state_dtype", "bfloat16")
    kw.setdefault("token_width", 1)
    return real(name, scale, kernel, phase, *args, **kw)


class _Shim:
    def __init__(self):
        self._o = nh.ops

    def __getattr__(self, n):
        return wrapped if n == "Mamba2Kernel" else getattr(self._o, n)


nh.ops = _Shim()
from aisim.sdk.inference_session import InferenceSession
from aisim_core.sdk import common, config
from aisim_core.sdk.backends.factory import get_backend
from aisim_core.sdk.models import get_model
from aisim_core.sdk.perf_database import get_database

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
# (batch, isl, osl, prefix, ctx_tokens): ctx_tokens is run_agg's prefill budget per step.
CASES = {
    "one_request_budget_above_new_tokens": (1, 4096, 64, 3000, 2048),
    "one_request_budget_above_prompt": (1, 4096, 64, 3000, 8192),
    "two_complete_cached_requests": (16, 4096, 64, 3000, 8192),
    "split_budget_cached": (16, 4096, 64, 1024, 8128),
    "chunked_uncached": (16, 6000, 64, 0, 2048),
}
out = {"model": MODEL, "system": "b200_sxm", "backend": "trtllm", "version": "1.3.0rc20", "mode": "HYBRID"}
out["cases"] = {}
for name, (b, isl, osl, prefix, ctx) in CASES.items():
    db = get_database("b200_sxm", "trtllm", "1.3.0rc20")
    db.set_default_database_mode(common.DatabaseMode.HYBRID)
    mc = config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1)
    session = InferenceSession(get_model(MODEL, mc, "trtllm"), db, get_backend("trtllm", "b200_sxm"))
    summary = session.run_agg(config.RuntimeConfig(batch_size=b, isl=isl, osl=osl, prefix=prefix), ctx_tokens=ctx)
    row = summary.get_summary_df().iloc[0].to_dict()
    mix = summary.get_per_ops_data()["mix_step"]
    out["cases"][name] = {
        "inputs": {"batch_size": b, "isl": isl, "osl": osl, "prefix": prefix, "ctx_tokens": ctx},
        "ttft_ms": float(row["ttft"]),
        "tpot_ms": float(row["tpot"]),
        "mix_step_mamba_ms": {k: float(v) for k, v in sorted(mix.items()) if "mamba" in k},
        "mix_step_non_mamba_ms": float(sum(v for k, v in mix.items() if "mamba" not in k)),
    }
# One session, same batch, ISL and budget: prefix 0, then 3000, then 3000 again.
db = get_database("b200_sxm", "trtllm", "1.3.0rc20")
db.set_default_database_mode(common.DatabaseMode.HYBRID)
mc = config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1)
session = InferenceSession(get_model(MODEL, mc, "trtllm"), db, get_backend("trtllm", "b200_sxm"))
calls = [
    session.run_agg(config.RuntimeConfig(batch_size=1, isl=4096, osl=64, prefix=prefix), ctx_tokens=2048)
    for prefix in (0, 3000, 3000)
]
out["same_session_prefixes"] = {
    "inputs": {"batch_size": 1, "isl": 4096, "osl": 64, "ctx_tokens": 2048, "prefixes": [0, 3000, 3000]},
    "context_mamba_ssm_ms": [float(c.get_per_ops_data()["mix_step"]["context_mamba_ssm"]) for c in calls],
    "ttft_ms": [float(c.get_summary_df().iloc[0]["ttft"]) for c in calls],
    "repeat_reuses_result": calls[2] is calls[1],
}
with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=1)
