# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-token cross-check for one mixed step. Usage: shared_tokens.py

A qualified graph at budget 8128 (isl 4096, prefix 1024) schedules two capped
3072-token suffixes, so its shared token-major ops see 6144 prefill tokens. A
legacy graph sees `ctx - prefix * floor(ctx / isl)`: 6144 at budget 8192 and
7104 at budget 8128. Each shared op of the qualified step must equal the legacy
step at 8192 and differ from the legacy step at 8128.
"""

import aisim_core.sdk.models.nemotron_h as nh

real = nh.ops.Mamba2Kernel
real_ops = nh.ops
qualify = False


def wrapped(name, scale, kernel, phase, *args, **kw):
    if qualify:
        kw.setdefault("state_dtype", "bfloat16")
        kw.setdefault("token_width", 1)
    return real(name, scale, kernel, phase, *args, **kw)


class _Shim:
    def __getattr__(self, n):
        return wrapped if n == "Mamba2Kernel" else getattr(real_ops, n)


from aisim_core.sdk import common, config
from aisim_core.sdk.backends.factory import get_backend
from aisim_core.sdk.models import get_model
from aisim_core.sdk.perf_database import get_database
from aisim_core.sdk.step_estimate import MixedStepInput

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def step(qualified, ctx):
    global qualify
    qualify = qualified
    db = get_database("b200_sxm", "trtllm", "1.3.0rc20")
    db.set_default_database_mode(common.DatabaseMode.HYBRID)
    nh.ops = _Shim()
    try:
        model = get_model(MODEL, config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1), "trtllm")
    finally:
        nh.ops = real_ops
    runtime = config.RuntimeConfig(batch_size=16, isl=4096, osl=64, prefix=1024)
    step = MixedStepInput(context_tokens=ctx, num_decode_requests=13)
    return get_backend("trtllm", "b200_sxm").run_mixed(model, db, runtime, step).per_op_latency_ms


def main():
    qualified, legacy_6144, legacy_7104 = step(True, 8128), step(False, 8192), step(False, 8128)
    for name, ms in qualified.items():
        if name not in legacy_6144:
            print(f"{name:28s} {ms:.6f}  (qualified graph only)")
            continue
        match = "own geometry"
        if ms == legacy_6144[name]:
            match = "= legacy 6144"
        elif ms == legacy_7104[name]:
            match = "= legacy 7104"
        print(f"{name:28s} {ms:.6f}  legacy 6144 {legacy_6144[name]:.6f}  legacy 7104 {legacy_7104[name]:.6f}  {match}")

    for name in ("context_mamba_in_proj_gemm", "context_moe"):
        assert qualified[name] == legacy_6144[name], f"{name}: expected the 6144-token query"
        assert qualified[name] != legacy_7104[name], f"{name}: control does not distinguish token counts"


if __name__ == "__main__":
    main()
