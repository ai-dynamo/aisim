---
name: aisim-validate-benchmarks
description: Compare AISim predictions with matched serving benchmarks and diagnose latency, throughput or coverage gaps. Use when assessing measured-versus-simulated accuracy, including AgentX results, rather than merely explaining a prediction report or collecting a new FPM table.
license: Apache-2.0
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Validate against a serving benchmark

Produce a reproducible, scoped accuracy assessment from an AISim prediction
and measured serving evidence. Prefer existing artifacts before launching
additional GPU work. Preserve the user's comparison metrics and tolerance;
do not invent a universal acceptable error threshold.

## Establish a matched experiment

Freeze the original prediction and its provenance before examining tuning
alternatives. Identify the reference run, AISim source/wheel, timing database
or FPM profile, model/checkpoint/tokenizer revisions, quantization, GPU topology,
backend/runtime build, engine limits and graph policy. For speculative decoding,
include draft identity, configured length and acceptance definition.

Match request lengths, input/prefix/cache history, arrivals or concurrency,
session/dependency/tool-delay behavior, seed, source selection, warmup and
measurement/grace boundaries. For agentic closed-loop workloads, preserve both
complete cohorts: changed latency can change which future requests are issued
and completed. A paired request subset is useful diagnosis, not a replacement
headline population.

Reject an unqualified comparison before computing a headline accuracy score:
missing run identity, different workload semantics, unaccounted failed requests,
or incomplete preparation can dominate a result. Record mismatches rather than
silently adjusting measured artifacts to fit a prediction.

## Define and recompute the metrics

Read the [native report contract](https://github.com/ai-dynamo/aisim/blob/main/docs/getting-started/understand-results.md)
and the benchmark producer's matching-version contract. Use the same units,
window, sample eligibility and percentile estimator on both sides.

| Comparison | Required distinction |
| --- | --- |
| TTFT | Arrival to first client-visible output; distinguish frontend, queueing and GPU boundaries. |
| TPOT / ITL | Per-request `(last - first) / (output_tokens - 1)` for at least two output tokens versus pooled token gaps or iteration time. Retain one-token TTFT samples. |
| Mean user token rate | Mean of eligible per-request reciprocals, separate from the reciprocal of mean TPOT. |
| Throughput | Completed requests/tokens over the declared observed span versus a fixed admission/profile duration. |
| Percentiles | Native AISim nearest-rank versus AIPerf linear interpolation; recompute with a common estimator when raw samples permit. |
| Completion/coverage | Attempted, completed, failed, canceled, unmatched and latency-eligible counts; missing metrics are not zero. |

Independently recompute statistics from per-request evidence where available.
Do not discard outliers or short/failed requests contrary to the agreed policy.
Report measured value, predicted value, signed difference and absolute error.
Relative absolute error is `abs(predicted / measured - 1)` for finite values
with a positive measured denominator. A zero reference leaves relative error
undefined; report its absolute difference. Negative or nonfinite latency/rate
values are invalid evidence. Use unrounded values for acceptance decisions.
State the statistical limits of a single run; matched configuration alone does
not measure repeatability.

## Diagnose a gap before changing the model

Separate workload/accounting mismatches from FPM timing error, scheduler/queue
behavior, cache/recomputation effects, transfer and frontend overhead. Inspect
request milestones, actual cache-use evidence, same-geometry forward passes
and relevant worker/rank telemetry. Choose one falsifiable hypothesis and a
bounded check. Keep CPU simulator execution speed separate from modeled latency.

The existing
[AgentX gap-debug skill](https://github.com/ai-dynamo/aisim/blob/main/.agents/skills/aisim-agentx-gap-debug/SKILL.md)
provides deeper diagnostic patterns for qualified AgentX evidence. Its
[collection companion](https://github.com/ai-dynamo/aisim/blob/main/.agents/skills/aisim-agentx-collect/SKILL.md)
describes serving-feature receipts for self-collected runs. Use applicable
public guidance and available artifacts; an inaccessible internal dataset
worker or private receipt store is a named limitation, not an implicit
prerequisite for analyzing a user-supplied benchmark.

If a new simulation is needed, preserve the matched configuration, check help
and write a fresh capture:

```bash
aisim predict --stack engine --config matched.yaml \
  --capture-per-request --output-dir benchmark-replay
```

Do not allocate hardware, broaden the benchmark or modify the performance
model merely to explain an existing discrepancy. Continue new measurements
or repairs when they are already within the user's authorized scope and budget.

## Deliver the assessment

Include the comparison identity and selection, metric definitions/windows,
counts, measured/predicted values, tolerance result and remaining confounders.
Distinguish original predictive accuracy from post-result calibration and
held-out validation. Link retained commands, configs, reports and per-request
evidence. State whether the assessment is qualified, failed or incomplete and
what evidence would resolve it. Functional support, interpolation coverage,
forward-pass accuracy and end-to-end serving accuracy are separate verdicts.
