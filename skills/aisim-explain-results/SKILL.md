---
name: aisim-explain-results
description: Explain AISim prediction and recommendation artifacts, including latency, throughput, goodput, incomplete requests, queueing and cache bottlenecks. Use when interpreting a simulation result; use matched benchmark validation for claims about prediction accuracy.
license: Apache-2.0
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Explain an AISim result

Turn the saved artifacts into an evidence-based explanation of serving behavior
and the next useful experiment. Start with files the user already supplied.
Reading a report does not require a new simulation or a serving GPU.

## Establish what ran

Inspect `prediction.json`, its input YAML, AISim version/source revision, and
the selected stack and runner. For a recommendation, also inspect
`recommendation.json` and the concrete candidate YAML. Match candidate IDs;
selected files alone omit failed or rejected trials.

Use the report contract for that runner. The fields below describe native
engine replay; analytical or external runners can have different populations.
`--format json` controls the stdout summary, not the durable report.
Consult the [results guide](https://github.com/ai-dynamo/aisim/blob/main/docs/getting-started/understand-results.md)
at the source revision matching the installed package when fields differ.

## Check populations before interpreting speed

Compare input/issued requests, completed requests, failures, cancellations,
unfinished work, and latency sample counts. A zero latency with no samples is
unavailable evidence. Headline native latency and output-token throughput use
completed requests; incomplete requests can make those columns look optimistic.
The absence of a terminal record does not establish that no request was admitted.

| Quantity | Native replay interpretation |
| --- | --- |
| TTFT | Arrival to first output token for completed requests with output; includes modeled queueing. |
| TPOT / reported ITL | One average per completed request with at least two output tokens: `(last - first) / (output_tokens - 1)`. It is not the worst token gap or a pooled mean over all gaps. |
| Request throughput | Completed requests divided by the modeled reporting interval in seconds. |
| Output throughput | Completed requests' output tokens divided by that interval. |
| Goodput | Completed requests satisfying the configured per-request SLA, with the corresponding qualifying token total. |
| `wall_time_ms` | CPU execution time of the simulator, separate from modeled serving latency and duration. |
| `gpu_hours` | Provisioned GPU time integrated over the simulated interval, including all modeled roles. |

For throughput per GPU, derive average provisioned GPUs from
`gpu_hours / (duration_ms / 3600000)` and divide by that value only when both
inputs are positive. Do not substitute peak allocation for an autoscaling run.
For continuous agentic profiles, inspect the recorded observed cohort interval;
it can differ from the configured admission duration and include grace successes.

Native percentiles use nearest rank; AIPerf uses linear interpolation. Mean
user token rate is the mean of eligible per-request `1000 / TPOT`, which need
not equal `1000 / mean_tpot_ms`. Strict SLA filtering checks configured
aggregate means and required evidence; it is not a p99 or every-request guarantee.

## Localize a suspected bottleneck

Use per-request milestones and native telemetry when present. Distinguish
observations from hypotheses; high TTFT alone cannot identify compute,
queueing, frontend or transfer time.

| Observation | Evidence to inspect next |
| --- | --- |
| Long wait before admission | Running/waiting queues, router pending requests, arrival rate and scheduler limits, grouped by worker and rank. |
| High prefill elapsed time | Prompt lengths, prefix-reuse attribution, prefill work and the selected timing model. |
| Slow decode or falling user token rate | Output lengths, concurrent decodes, decode timing, KV occupancy and preemption/recomputation evidence. |
| Little benefit from a warm cache | Actual G1 lookup matched/queried tokens and first-admission token sources; router overlap is not measured cache reuse. |
| Missing completion or unstable throughput | Stop controls, profile/grace boundaries, cancellations, pending work and retained interval counters. |

Keep clock domains aligned: `requests.jsonl` is profiling-only and
barrier-relative; `telemetry-requests.jsonl` includes preparation and uses the
absolute virtual clock. Use the manifest's profile origin before joining them.
Do not interpolate scheduler snapshots into an invented continuous trace.
Missing cache tiers/counters stay unavailable; a zero denominator is not a
measured zero-percent hit rate. Interval delivered-token counters may include
tokens from canceled requests and differ from headline completed-token totals.

If a new capture is authorized and needed, retain the user's configuration and
write a fresh output directory:

```bash
aisim predict --stack engine --config prediction.yaml \
  --capture-per-request --capture-telemetry --output-dir prediction-diagnostics
```

Check the installed command's help first. Telemetry support and capture memory
depend on the runner and topology; do not silently change the deployment to
enable it. `--export-html` can render a simulation dashboard when supported.

## Deliver the explanation

State the modeled deployment/workload, completion and sample counts, metric
units/window, the observations supporting each diagnosis, and any missing
evidence. Propose a bounded comparison that changes one relevant input while
preserving the rest. Keep predicted behavior separate from measured silicon
accuracy; the [benchmark validation skill](https://github.com/ai-dynamo/aisim/tree/main/skills/aisim-validate-benchmarks)
covers a matched comparison.
