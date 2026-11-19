# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alternate the forward-perf gate worker over builds on its 13 SILICON Qwen3-32B and
Nemotron-3-Super cases. Usage: host_perf.py ROOT ROUNDS OUT.json BUILD...

Each BUILD is a directory under ROOT that holds the build's checkout in `src` and its
environment in `venv`. `HOST_PERF_MODELS` selects model-id prefixes (comma-separated) and
`HOST_PERF_MODES` database modes (default `SILICON`). Round r runs the builds rotated by
r, so with ROUNDS a multiple of the build count each build runs in each position equally
often. OUT.json holds each round's order
and maps build -> case -> per-round cold and warm medians (us).
"""

import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

root, rounds, out, builds = Path(sys.argv[1]).resolve(), int(sys.argv[2]), sys.argv[3], sys.argv[4:]
if rounds <= 0:
    sys.exit(f"ROUNDS must be positive, got {rounds}")
sys.path.insert(0, str(root / builds[0] / "src/python/aisim"))
from tools.forward_perf_gate.cases import expand_cases

prefixes = os.environ.get("HOST_PERF_MODELS")
modes = os.environ.get("HOST_PERF_MODES", "SILICON").split(",")


def selected(model_id):
    if prefixes:
        return model_id.startswith(tuple(prefixes.split(",")))
    return model_id == "qwen3-32b" or model_id.startswith("nemotron-3-super")


cases = [c for c in expand_cases() if selected(c["model_id"]) and c["database_mode"] in modes]
res = {b: {} for b in builds}
orders = []
for r in range(rounds):
    order = builds[r % len(builds) :] + builds[: r % len(builds)]
    orders.append(order)
    for b in order:
        request = {"protocol_version": 2, "revision": b, "cases": cases, "warmup": 10, "iterations": 100}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(request, f)
        proc = subprocess.run(
            [str(root / b / "venv/bin/python"), "tools/forward_perf_gate/worker.py", "--request", f.name],
            cwd=root / b / "src/python/aisim",
            capture_output=True,
            text=True,
            check=True,
        )
        for c in json.loads(proc.stdout)["results"]:
            if c.get("status") != "OK":
                sys.exit(f"{b} {c['case_id']}: {c.get('status')} {json.dumps(c.get('error', {}))[:200]}")
            row = res[b].setdefault(c["case_id"], {"cold": [], "warm": []})
            row["cold"].append(c["cold_us"])
            row["warm"].append(c["warm"]["call_median_us"])
with open(out, "w") as f:
    json.dump({"orders": orders, "results": res}, f, indent=1)
for case in sorted(res[builds[0]]):
    print(case, *(f"{statistics.median(res[b][case]['warm']):.2f}" for b in builds))
