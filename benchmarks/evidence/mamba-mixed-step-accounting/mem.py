# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-request vLLM run_agg memory for Nemotron Nano. Usage: mem.py"""

import json
import sys

from aisim.sdk.inference_session import InferenceSession
from aisim_core.sdk import config
from aisim_core.sdk.backends.factory import get_backend
from aisim_core.sdk.models import get_model
from aisim_core.sdk.perf_database import get_database_view

db = get_database_view("b200_sxm", "vllm", "0.24.0", database_mode="HYBRID", allow_unlisted_version=True)
if db is None:
    sys.exit("mem.py needs the b200_sxm vLLM 0.24.0 performance data")
mc = config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1)
session = InferenceSession(
    get_model("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", mc, "vllm"), db, get_backend("vllm", "b200_sxm")
)
out = {}
for prefix, ctx in [(3000, 2048), (3000, 4096), (3000, 8192), (0, 2048)]:
    memory = session.run_agg(
        config.RuntimeConfig(batch_size=1, isl=4096, osl=64, prefix=prefix), ctx_tokens=ctx
    ).get_memory()
    out[f"prefix={prefix},ctx_tokens={ctx}"] = {
        k: round(float(v), 6) for k, v in memory.items() if isinstance(v, (int, float))
    }
print(json.dumps(out, indent=1))
