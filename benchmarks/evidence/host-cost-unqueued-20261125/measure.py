# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lane-1 GLM host overhead: client TTFT minus the engine's prefill span, per request.

Client: aiperf profile_export.jsonl (request_start_ns, time_to_first_token, input tokens), profiling phase only.
Engine: step records; a request's engine span runs from its first prefill step's scheduled time to its last
prefill step's emitted time (server monotonic clock, mapped to unix via the recorder's receive clock pair).
Usage: measure.py PROFILE_EXPORT STEPS_JSONL[.gz]"""

import gzip, json, sys
import numpy as np

prof, steps_path = sys.argv[1], sys.argv[2]
reqs = []
for l in open(prof):
    r = json.loads(l)
    m = r["metadata"]
    x = r["metrics"]
    if m.get("benchmark_phase") != "profiling" or m.get("was_cancelled") or "time_to_first_token" not in x:
        continue
    reqs.append(
        (m["request_start_ns"] / 1e9, x["time_to_first_token"]["value"] / 1e3, x["input_sequence_length"]["value"])
    )
reqs.sort()
op = gzip.open if steps_path.endswith(".gz") else open
pre = []
offs = []
for l in op(steps_path, "rt"):
    r = json.loads(l)
    f = r["fpm"]
    s = f["scheduled_requests"]
    offs.append(r["received_unix_s"] - r["received_monotonic_s"])
    if s["num_prefill_requests"] > 0 and "host_scheduled_at_s" in s:
        pre.append((s["host_scheduled_at_s"], s["host_emitted_at_s"], s["sum_prefill_tokens"]))
off = float(np.median(offs))
pre = np.array([(a + off, b + off, n) for a, b, n in pre])
rows = []
for start, ttft, inp in reqs:
    end = start + ttft
    sel = pre[(pre[:, 0] >= start) & (pre[:, 0] <= end)]
    if not len(sel):
        continue
    eng0, eng1 = sel[:, 0].min(), sel[:, 1].max()
    rows.append((inp, ttft * 1e3, (eng0 - start) * 1e3, (eng1 - eng0) * 1e3, (end - eng1) * 1e3, sel[:, 2].sum()))
a = np.array(rows)
print(f"requests {len(reqs)} matched {len(a)}")
for name, col in (("before engine", 2), ("engine prefill span", 3), ("after engine", 4)):
    print(f"  {name:20s} median {np.median(a[:, col]):7.1f} ms  mean {a[:, col].mean():7.1f} ms")
ovh = a[:, 2] + a[:, 4]
X = np.c_[np.ones(len(a)), a[:, 0] / 1e3]
for name, y in (("before engine", a[:, 2]), ("after engine", a[:, 4]), ("outside engine (sum)", ovh)):
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"  fit {name:22s} = {c[0]:6.1f} ms + {c[1] * 1e3 / 1e3:5.3f} us/token x input   (R2 {r2:.2f})")
print(f"  median input {np.median(a[:, 0]):.0f} tokens; median prefilled {np.median(a[:, 5]):.0f}")
