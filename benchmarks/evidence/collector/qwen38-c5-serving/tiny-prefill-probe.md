<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Supplementary: is the tiny-prefill row a first-touch artifact?

Answers the open item from
[`c37_prefill_admission_exclusions`](../../../model-onboarding/onboarding-checkpoint.json):
both C3.7 prefill parquets contain one anomalous coordinate,
`(batch_size=1, total_kv_read_tokens=0, total_prefill_tokens=1)`, at 225.558 ms
(BF16) and 211.8 ms (FP8) against a ~36 ms neighbour floor at prefill=4. It was
the first point each sweep injected, with `warmup_repeats=0` and
`measurement_repeats=1`.

**This does not modify or rehabilitate that row.** The publish copies still have
1589 rows each and the row stays denied by non-publication. This is evidence
about the mechanism.

## Method

`model-onboarding/c5/05-probe/probe.sh`, run in `qwen38-tiny-prefill` on the
same node, image (`vllm-openai:v0.31.0`, `sha256:c1c9f6fd…67c3`) and topology
(TP8/EP8, BF16) as the C3.7 collection, with the serving flags of the C5 cells.

TTFT is the client's time to the first streamed chunk carrying generated text,
through `POST /v1/completions` with an explicit token-id prompt (`max_tokens 1`,
`ignore_eos`, `temperature 0`). Phase A fires prefill=1 as the very first
request after `/health` returns; then prefill=4; then three warmups at 64; then
20 × prefill=1 and 5 × each of 4/8/16/32/64/97.

Raw log: `model-onboarding/c5/logs/tiny-prefill.log`.
Server side: 43.53 GiB/rank, model load 295.9 s.

## Results

| point | n | median (ms) | min | max |
| --- | --- | --- | --- | --- |
| cold prefill=1 (first request) | 1 | 285.6 | — | — |
| cold prefill=4 (second request) | 1 | 1214.1 | — | — |
| warm prefill=1 | 20 | 58.1 | 55.9 | 65.6 |
| warm prefill=4 | 5 | 145.5 | 142.6 | 152.5 |
| warm prefill=8 | 5 | 148.6 | 142.0 | 1227.6 |
| warm prefill=16 | 5 | 145.1 | 142.0 | 708.0 |
| warm prefill=32 | 5 | 146.8 | 145.5 | 955.8 |
| warm prefill=64 | 5 | 153.1 | 150.3 | 153.3 |
| warm prefill=97 | 5 | 153.9 | 151.1 | 708.9 |

## What it supports, and what it does not

**Supported:** a cold effect is real and is *not confined to the first
request*. The first prefill=1 after readiness is 285.6 ms against a warm median
of 58.1 ms — **4.9×**, the same direction and comparable magnitude to the
collector's 6.3×. But the *second* request (prefill=4) was 1214.1 ms, slower
than the first, so "the very first injected point" is not the whole story: the
engine needs a few requests to settle, which is exactly what `warmup_repeats=0`
denies.

**Not supported:** the absolute values do not transfer. This serving path sits
at 58 ms for prefill=1 and ~146–154 ms for prefill≥4, whereas the collector's
Dynamo self-benchmark measured ~36 ms across the same small-prefill range. The
probe therefore cannot be used to correct the published numbers; the two
harnesses differ in frontend overhead and scheduling. A correction needs the
collector re-run with `measurement_repeats > 1` (and ideally a real warmup),
which is still the remedy named in the exclusion record.

**New finding, and the most consequential one:** even fully warm, 4 of 30
prefill≥4 requests (13.3%) exceeded 500 ms, up to 1227.6 ms, with no pattern
tied to request order or prompt length that this probe can identify. A
single-sample collector (`measurement_repeats=1`) is exposed to that tail at
*every* small-prefill coordinate, not just the first one. The practical
consequence for admission is stronger than "one bad row": the small-prefill
corner of these parquets is a single draw from a distribution with a heavy
right tail, and no row there is repeatability-qualified.

## Reproduce

```bash
export KUBECONFIG=~/psap/kubeconfig-psap-janus
kubectl -n aisimulate-qwen38 create configmap qwen38-tiny-prefill \
  --from-file=probe.sh=model-onboarding/c5/05-probe/probe.sh --dry-run=client -o yaml \
  | kubectl apply -f -
kubectl apply -f model-onboarding/c5/05-tiny-prefill-probe.yaml
kubectl -n aisimulate-qwen38 logs -f qwen38-tiny-prefill
```
