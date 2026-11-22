# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host time of the native entry points the d6b02759 accounting fixes touch.

Usage: affected_perf.py ROOT ROUNDS OUT.json BUILD...

Each BUILD is a directory under ROOT with the checkout in `src` and its environment in
`venv`, as for host_perf.py. Round r runs the builds rotated by r. Each run is a fresh
process: it builds every handle, primes it with a query of another shape, then times one
unseen query (cold) and 100 repeats after 10 warmup calls (warm median) with the forward
perf gate's measure_cold_and_warm. OUT.json holds each round's order and maps
build -> case -> per-round cold, warm median, raw warm samples and returned value.
"""

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def worker(src):
    sys.path.insert(0, os.path.join(src, "python/aisim"))
    sys.path.insert(0, os.path.join(src, "python/aisim/tests/unit/sdk"))
    from tools.forward_perf_gate.measurement import measure_cold_and_warm

    import aisim_core.sdk.models.nemotron_h as nh

    real, real_ops, qualify = nh.ops.Mamba2Kernel, nh.ops, {"on": False}

    def wrapped(name, scale, kernel, phase, *args, **kw):
        if qualify["on"]:
            kw.setdefault("state_dtype", "bfloat16")
            kw.setdefault("token_width", 1)
        return real(name, scale, kernel, phase, *args, **kw)

    class _Shim:
        def __getattr__(self, n):
            return wrapped if n == "Mamba2Kernel" else getattr(real_ops, n)

    nh.ops = _Shim()
    from aisim_core.sdk import common
    from aisim_core.sdk import config as sdk_config
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.perf_database import get_database
    from aisim_core.sdk.rust_engine_step import RustForwardPassPerfModel, _cached_engine_handle

    def fpm_model(model, qualified):
        qualify["on"] = qualified
        cfg = {
            "model": model,
            "system": "b200_sxm",
            "backend": "trtllm",
            "backend_version": "1.3.0rc20",
            "estimation_mode": "op_level",
            "tp": 1,
            "moe_tp_size": 1,
            "moe_ep_size": 1,
            "fallback_policy": "deny",
            "worker_type": "aggregated",
            "database_mode": "HYBRID",
        }
        return RustForwardPassPerfModel.best_available(cfg)

    def handle(model, qualified):
        qualify["on"] = qualified
        db = get_database("b200_sxm", "trtllm", "1.3.0rc20")
        db.set_default_database_mode(common.DatabaseMode.HYBRID)
        mc = sdk_config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1)
        return _cached_engine_handle(get_model(model, mc, "trtllm"), db)

    def packed(requests, tokens):
        return [{"scheduled_requests": {"num_prefill_requests": requests, "sum_prefill_tokens": tokens}}]

    nano, qwen = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "Qwen/Qwen3-32B"
    cases = {}
    for name, qualified in (("nano_qualified", True), ("nano_legacy", False)):
        m = fpm_model(nano, qualified)
        m.estimate_forward_pass_time_ms(packed(2, 2048))  # prime
        for label, tokens in (("uneven_4097", 4097), ("even_4096", 4096)):
            cases[f"{name}.pure_prefill_{label}"] = lambda m=m, t=tokens: m.estimate_forward_pass_time_ms(packed(2, t))
    q = fpm_model(qwen, False)
    q.estimate_forward_pass_time_ms(packed(2, 2048))
    cases["qwen.pure_prefill_uneven_4097"] = lambda: q.estimate_forward_pass_time_ms(packed(2, 4097))
    for name, model, qualified in (
        ("nano_qualified", nano, True),
        ("nano_legacy", nano, False),
        ("qwen", qwen, False),
    ):
        h = handle(model, qualified)
        h._run_static_per_op_with_coverage(batch_size=2, isl=512, osl=8, mode="static", stride=1)  # prime
        cases[f"{name}.static_coverage"] = lambda h=h: h._run_static_per_op_with_coverage(
            batch_size=8, isl=1024, osl=33, mode="static", stride=1
        )[3]
    # Whole-model FPM mixed step on the synthetic fpm_caller table.
    import shutil
    import tempfile

    import test_fpm_forward as t

    from aisim_core.sdk import models
    from aisim_core.sdk.perf_database import PerfDatabase

    fm = models.get_model("Qwen/Qwen3-0.6B", t._model_config(forward_model="fpm"), t.BACKEND)
    identity = dict(zip(t._CELL_MATCH_COLUMNS, fm.context_ops[0]._match_identity, strict=True))
    rows = [
        t._row(phase, *key, ms, model_path=fm.model_path, identity=identity)
        for phase, key, ms in (
            ("prefill", (1, 256, 256), 12.0),
            ("prefill", (2, 512, 512), 30.0),
            ("prefill", (1, 512, 0), 22.0),
            ("prefill", (1, 520, 0), 22.5),
            ("decode", (1, 0, 514), 3.0),
            ("decode", (8, 0, 4112), 6.0),
        )
    ]
    root = tempfile.mkdtemp()
    shutil.copy(os.path.join(t._CORE_SYSTEMS, f"{t.SYSTEM}.yaml"), os.path.join(root, f"{t.SYSTEM}.yaml"))
    t._write_pair(os.path.join(root, "data", t.SYSTEM, t.BACKEND, t.VERSION), rows)
    fh = _cached_engine_handle(fm, PerfDatabase(t.SYSTEM, t.BACKEND, t.VERSION, systems_root=root))
    fh._mixed_step_breakdown_per_op_with_metadata(256, 1, 512, 2, 256)  # prime
    cases["fpm.mixed_coverage"] = lambda: fh._mixed_step_breakdown_per_op_with_metadata(512, 8, 512, 2, 0)[3]

    out = {}
    for name, call in cases.items():
        value, cold, warm_samples, warm = measure_cold_and_warm(call, warmup=10, iterations=100)
        out[name] = {
            "value": value,
            "cold_us": cold,
            "warm_us": warm["call_median_us"],
            "warm_samples_us": warm_samples,
        }
    shutil.rmtree(root)
    sys.stdout.write("\n" + json.dumps(out) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        worker(sys.argv[2])
        sys.exit()
    root, rounds, out, builds = Path(sys.argv[1]).resolve(), int(sys.argv[2]), sys.argv[3], sys.argv[4:]
    if rounds <= 0:
        sys.exit(f"ROUNDS must be positive, got {rounds}")
    res, orders = {b: {} for b in builds}, []
    for r in range(rounds):
        order = builds[r % len(builds) :] + builds[: r % len(builds)]
        orders.append(order)
        for b in order:
            proc = subprocess.run(
                [str(root / b / "venv/bin/python"), os.path.abspath(__file__), "--worker", str(root / b / "src")],
                cwd=root / b,
                capture_output=True,
                text=True,
                check=True,
            )
            for name, row in json.loads(proc.stdout.splitlines()[-1]).items():
                cell = res[b].setdefault(name, {"cold": [], "warm": [], "warm_samples_us": [], "value": []})
                cell["cold"].append(row["cold_us"])
                cell["warm"].append(row["warm_us"])
                cell["warm_samples_us"].append(row["warm_samples_us"])
                cell["value"].append(row["value"])
    with open(out, "w") as f:
        json.dump({"orders": orders, "results": res}, f, indent=1)
    for name in res[builds[0]]:
        cells = [res[b][name] for b in builds]
        print(name, *(f"{statistics.median(c['warm']):.2f}/{statistics.median(c['cold']):.1f}" for c in cells))
