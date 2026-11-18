# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alternate the forward-perf gate worker over builds on its 13 SILICON Qwen3-32B and
Nemotron-3-Super cases. Usage: host_perf.py ROOT ROUNDS OUT.json BUILD...

Each BUILD is a directory under ROOT that holds the build's checkout in `src` and its
environment in `venv`. OUT.json maps build -> case -> per-round cold and warm medians (us).
"""

import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

root, rounds, out, builds = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4:]
sys.path.insert(0, str(root / builds[0] / "src/python/aisim"))
from tools.forward_perf_gate.cases import expand_cases

cases = [
    c
    for c in expand_cases()
    if (c["model_id"] == "qwen3-32b" or c["model_id"].startswith("nemotron-3-super"))
    and c["database_mode"] == "SILICON"
]
res = {b: {} for b in builds}
for _ in range(rounds):
    for b in builds:  # builds run in turn within each round
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
                sys.exit(f"{b} {c['case_id']}: {c.get('status')} {c.get('error', '')[:200]}")
            row = res[b].setdefault(c["case_id"], {"cold": [], "warm": []})
            row["cold"].append(c["cold_us"])
            row["warm"].append(c["warm"]["call_median_us"])
with open(out, "w") as f:
    json.dump(res, f, indent=1)
for case in sorted(res[builds[0]]):
    print(case, *(f"{statistics.median(res[b][case]['warm']):.2f}" for b in builds))
