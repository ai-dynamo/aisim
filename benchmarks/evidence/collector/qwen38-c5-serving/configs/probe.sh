#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Minimal cold-vs-warm test for (b=1, kv=0, prefill=1).
# The published parquet row is NOT modified; this only tests first-touch.
set -uo pipefail
MODEL=Qwen/Qwen3.8-Flash-Next
TOK=/hf-cache/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540
LOG=/scratch/server.log

vllm serve "$TOK" --served-model-name "$MODEL" --tensor-parallel-size 8 \
  --enable-expert-parallel --gpu-memory-utilization 0.90 --max-model-len 9344 \
  --max-num-seqs 1 --no-enable-prefix-caching \
  --engram-config '{"cpu_offload": false}' --port 8000 > "$LOG" 2>&1 &
PID=$!
echo "start $(date -u +%H:%M:%SZ)"

for _ in $(seq 1 300); do
  curl -sf localhost:8000/health >/dev/null 2>&1 && break
  kill -0 "$PID" 2>/dev/null || { echo "server died"; tail -40 "$LOG"; exit 1; }
  sleep 4
done
curl -sf localhost:8000/health >/dev/null 2>&1 || { echo "not ready"; tail -40 "$LOG"; exit 1; }
echo "healthy $(date -u +%H:%M:%SZ)"

# ttft: one streamed completion, timed to the first data line carrying text.
ttft() {
  python3 -c '
import json, sys, time, urllib.request
n, lbl = int(sys.argv[1]), sys.argv[2]
body = json.dumps({"model": "Qwen/Qwen3.8-Flash-Next",
                   "prompt": [1000 + i for i in range(n)],
                   "max_tokens": 1, "ignore_eos": True,
                   "stream": True, "temperature": 0.0}).encode()
req = urllib.request.Request("http://127.0.0.1:8000/v1/completions", data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=600) as r:
    for line in r:
        if line.startswith(b"data:") and b"\"text\"" in line:
            break
print("%s n=%d ttft=%.1f" % (lbl, n, (time.time() - t0) * 1000.0))
' "$1" "$2"
}

echo "== cold: first requests after readiness =="
ttft 1 cold-1
ttft 4 cold-4

echo "== warmups: 3 x prefill=64 =="
for _ in 1 2 3; do ttft 64 warmup >/dev/null; done

echo "== warm: prefill=1 x20, neighbours x5 =="
for _ in $(seq 20); do ttft 1 warm-1; done
for n in 4 8 16 32 64 97; do
  for _ in 1 2 3 4 5; do ttft "$n" "warm-$n"; done
done

grep -iE "Model loading took|Available KV cache|GPU KV cache size" "$LOG" | head -3 | sed 's/^/  /'
kill -TERM "$PID" 2>/dev/null; sleep 8; kill -KILL "$PID" 2>/dev/null
echo "end $(date -u +%H:%M:%SZ)"
