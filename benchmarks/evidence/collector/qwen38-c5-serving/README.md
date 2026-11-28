<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.8-Flash-Next C5: single-stream serving check on H200/sm90

This records the A6/D3 aggregated serving lane for
`Qwen/Qwen3.8-Flash-Next`: the checkpoint served at the topology AISimulate
prices (TP8, `moe_ep_size` 8, aggregated) and one fixed workload replayed
through each runtime's own bench client, next to the same workload through
`aisimulate predict --stack engine`.

**What this is:** four single-stream points (2 backends × 2 precisions) at
8192 input / 128 output tokens, measured on the serving path with CUDA graphs
enabled.

**What this is not:** a concurrency sweep, a disaggregated measurement, or a
vision lane. It is one workload per cell, eight requests, concurrency 1. Any
conclusion here is about steady-state single-stream latency at one shape.

## Environment

| item | value |
| --- | --- |
| node | `psap-de-h200-cluster-6mgmp-gpu-h200-7zgjh`, 8 × H200 (sm90), 250 G root with 116 G free at start |
| namespace | `aisimulate-qwen38`, PVC `qwen38-model-cache` (1 Ti RWX NFS, `pvc-84822c11-…`) |
| vLLM | `v0.31.0`, `sha256:c1c9f6fd…67c3` |
| SGLang | `v0.5.21`, `sha256:b1259f3e…f469` |
| BF16 checkpoint | `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d43b917e7706784d8bb445c9af86a3540`, 360,023,351,514 B |
| FP8 checkpoint | `Qwen/Qwen3.8-Flash-Next-FP8` @ `236dfdf285828023ca3bcd3f37366c58a3469b13`, 185,563,783,577 B |

Both snapshots VERIFY-OK against a download manifest of 545,587,135,091 B.

Server flags were the C2 load-tested set (`model-onboarding/c2`) at a 9344
context, plus `--no-enable-prefix-caching` / `--disable-radix-cache` so the
comparison is not silently cache-assisted:

- vLLM: `--tensor-parallel-size 8 --enable-expert-parallel
  --gpu-memory-utilization 0.90 --max-model-len 9344 --max-num-seqs 1
  --no-enable-prefix-caching --engram-config '{"cpu_offload": false}'`
- SGLang: `--tp-size 8 --ep-size 8 --no-ple-offload-embedding
  --disable-radix-cache --context-length 9344 --mem-fraction-static 0.85`

Workload: `isl 8192`, `osl 128`, `concurrency 1`, `8` requests, warmups
excluded by the client (vLLM `--num-warmups 2`, SGLang `--warmup-requests 3`).

## What the engines actually resolved

The KV dtype is the field the FP8 QSA key-derivation fix depends on, so it is
read back from each engine rather than assumed:

| backend | precision | KV dtype reported by engine | KV capacity |
| --- | --- | --- | --- |
| vLLM | BF16 | `cache_dtype=auto` → bf16 | 80.05 GiB pool |
| vLLM | FP8 | `cache_dtype=auto` → bf16 | 99.64 GiB pool |
| SGLang | BF16 | `torch.bfloat16` | 3,197,248 tokens (K 18.30 GB + V 18.30 GB) |
| SGLang | FP8 | `torch.bfloat16` | 4,080,832 tokens (K 23.35 GB + V 23.35 GB) |

All four are bf16 KV, which matches the predictor's `kv_cache_precision`
(`aisimulate_policy` → `bfloat16`) for both checkpoints, and is the runtime
confirmation that the FP8 checkpoint's routed-expert-only quantization leaves
QSA and its cache in bf16. Weight footprint: 43.53 GiB/rank (BF16), 23.51
GiB/rank (FP8) in vLLM.

## Measured vs predicted

Predicted values are `aisimulate predict --stack engine` on the identical
traffic (`max_batched_tokens 8192`, `max_sequences 8`, concurrency 1).

The committed `predictions/*.json` were rendered before the QSA fused halves
moved inside the MSA module op, so their per-op lists still name
`context_qsa_output_gate_gemm` / `context_qsa_indexer_k_gemm` (and the
generation pair). Those widths are now `output_gate_dim` / `indexer_k_dim` on
the module and are charged inside it. The move is FLOPS-exact, so the totals
below are unchanged; only the op attribution differs.

| backend | precision | TTFT meas (ms) | TTFT pred | Δ | TPOT meas | TPOT pred | Δ | out tok/s meas | pred | Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vLLM | BF16 | 194.96 | 144.09 | +35.3% | 5.77 | 5.85 | −1.4% | 138.01 | 144.30 | −4.4% |
| vLLM | FP8 | 211.85 | 144.13 | +47.0% | 6.26 | 6.15 | +1.8% | 127.06 | 138.39 | −8.2% |
| SGLang | BF16 | 189.80 | 162.71 | +16.6% | 6.22 | 5.61 | +10.9% | 130.33 | 146.33 | −10.9% |
| SGLang | FP8 | 189.08 | 160.44 | +17.9% | 6.31 | 5.56 | +13.5% | 128.84 | 147.73 | −12.8% |

Distribution shape, where it was captured: vLLM BF16 TTFT mean 194.96 /
median 195.53; SGLang BF16 189.80 / 189.78 / p99 191.59 and SGLang FP8 189.08 /
188.29 / p99 194.10, with per-request TTFTs spread over 187–194 ms. The two
SGLang cells are tight enough that the mean is representative; the vLLM cells
did not have their per-request arrays copied out (see gaps).

End-to-end, measured vs predicted: SGLang FP8 990.84 vs 866.46 (+14.4%),
SGLang BF16 not captured / 874.72, vLLM not captured / 887.03 (BF16) and
924.91 (FP8).

## Interpretation

- **TTFT is systematically under-predicted by the runtime**, 16.6%–47.0% high
  in measured terms. This is the one direction that is consistent across both
  backends and both precisions, so it is a modelling gap rather than a
  measurement artefact.
- **TPOT is close for vLLM** (−1.4%, +1.8%) and **biased low for SGLang**
  (+10.9%, +13.5%). Decode is the better-modelled half of the two.
- **Throughput is 4%–13% optimistic** everywhere, which is the direct
  consequence of the TTFT error at this shape.
- **FP8 does not buy prefill latency on vLLM here** (+17 ms TTFT, +0.49 ms
  TPOT vs BF16), while **SGLang shows almost no precision difference**
  (−0.7 ms TTFT, +0.09 ms TPOT). Both are consistent with the C3.7 prefill
  finding that at ≤8192 scheduled tokens only GEMM/MoE are quantized and EP8
  dispatch outweighs the arithmetic saving.
- The TTFT gap is **not** attributed to a specific stage by this dataset. With
  concurrency 1 there is no queueing component to separate out, so the
  residual sits in the prefill estimate itself (op-level rows measured at
  b=1..128 prefill buckets) or in the fixed per-request overhead the predictor
  does not model. Separating those needs the disaggregated and concurrency
  lanes, which this bundle does not contain.

## Runs, retries and rejections

Every attempt is listed because the accepted number is only meaningful next to
what was discarded.

| cell | attempt | outcome |
| --- | --- | --- |
| vLLM BF16 | 1 | failed: no `/dev/shm` mount, engine-core init (TP8 NCCL) |
| vLLM BF16 | 2 | server healthy, bench failed resolving the HF id (closed network, no `refs/main`); fixed with `--tokenizer <snapshot>` |
| vLLM BF16 | 3 | **rejected**: concurrency 8, TTFT mean 12.9 s, dominated by queueing, not an accuracy point |
| vLLM BF16 | 4 | accepted (concurrency 1) |
| vLLM FP8 | 1 | accepted (concurrency 1) |
| SGLang BF16 | 1 | server healthy; bench failed — `--dataset-name random` fetches ShareGPT via `hf_hub_download` at dataset build |
| SGLang BF16 | 2 | **rejected**: `random-ids` without `--random-range-ratio 1.0` sampled lengths uniformly in [1, 8192] (38,412 in / 750 out tokens instead of 65,536 / 1,024) |
| SGLang BF16 | 3 | accepted |
| SGLang FP8 | 1 | **rejected**: same length-distribution defect |
| SGLang FP8 | 2 | accepted |

Two harness defects were found by these runs and are now pinned in
`benchmarks/evidence/collector/qwen38-c5-serving/configs/00-serving-harness-configmap.yaml`: SGLang's
`--random-range-ratio` default is not a fixed length, and `random-ids` +
`--tokenize-prompt` is the only offline-safe way to get exact token counts
(vLLM's random dataset samples vocabulary ids, so the regime matches).

## Provenance gaps, stated plainly

1. **The two vLLM cells have no retained raw logs.** Their pods were deleted at
   cell teardown before the logs were copied out, and the vLLM result JSON was
   written to the pod's `emptyDir` and died with it. The vLLM numbers in the
   table are transcribed from the bench summary that the harness printed into
   the pod log (`Successful 8`, `Failed 0`, duration 7.42 s / 8.06 s), together
   with the model-load and KV-pool lines from the same log. They are not
   independently re-readable from this bundle. The two SGLang cells do have
   retained logs: `logs/sglang-fp8.log` (`sglang-bf16` was captured in the same
   form but its pod was deleted first; re-read from the transcript above).
2. **One sample per cell.** Eight requests, one run, no repeat of the whole
   cell, so this says nothing about run-to-run variance.
3. **The predictor arms are BF16-precision model rows applied to the FP8
   checkpoint**, which is correct here because only routed experts are
   quantized (see the KV table above), not because FP8 GEMM was measured at
   this shape.

## Reproduce

```bash
export KUBECONFIG=~/psap/kubeconfig-psap-janus
kubectl apply -f benchmarks/evidence/collector/qwen38-c5-serving/configs/00-serving-harness-configmap.yaml
kubectl apply -f benchmarks/evidence/collector/qwen38-c5-serving/configs/01-vllm-bf16.yaml     # or 02/03/04
kubectl -n aisimulate-qwen38 logs -f qwen38-serve-vllm-bf16
```

The predictor side is the matching `predict-*.yaml` at concurrency 1 / 8
requests:

```bash
aisimulate predict --stack engine --config benchmarks/evidence/collector/qwen38-c5-serving/configs/predict-vllm-bf16.yaml \
  --format json --output-dir ./out/predict-vllm-bf16-conc1
```
