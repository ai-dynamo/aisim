---
name: aisim-replay-workload
description: Prepare and replay a real request or agentic workload trace with AISim, preserving timing, token lengths, sessions, dependencies and cache identity. Use for reproducible CPU simulations of an existing workload rather than synthetic configuration search or GPU benchmark collection.
license: Apache-2.0
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Replay a real workload

Produce a reproducible simulation from the user's trace and target deployment.
Offline engine replay runs on the CPU; `engine.hardware` names modeled GPUs
and does not require allocating them or downloading model weights.

## Inspect the input and installed contract

Record the AISim version/source revision, trace provenance and content hash,
model/config revision, hardware, backend/version and concrete engine settings.
Inspect `aisim predict --help` and the
[workload contract](https://github.com/ai-dynamo/aisim/blob/main/docs/replay/workloads.md)
matching that source before converting a new producer format. Current source
documentation may describe capabilities absent from an older wheel.

Examine representative rows plus whole-input counts and schema validity.
Determine timestamp units and origin, full-prompt versus delta lengths,
session/turn structure, dependencies, tool delays and prefix identity. Ask only
for facts the supplied data and deployment metadata cannot establish.

| Input | Replay choice and important boundary |
| --- | --- |
| Mooncake JSONL | `mooncake` for complete prompts; `mooncake-delta` only for new-input deltas on later turns. Delta replay appends prior outputs and would double-count a full prompt. |
| Dependency graph JSONL | `agentic_mooncake`; explicit unique request IDs and an acyclic `wait_for` graph control release. |
| Weka / AgentX corpus | `weka`; preserve complete plays and children. Resolve nested absolute/relative timestamp conventions from producer evidence. |
| Dynamo request trace | `dynamo`; use its embedded hash-block size and the matching standard/agentic load mode. Reading it does not require the Dynamo stack. |
| Applied Compute agentic sessions | `applied_compute_agentic`; use concurrency load because these rows lack first-session timestamps. |
| Another CSV/JSON/export format | Write an explicit conversion into a supported format and document each field mapping; renaming the file is insufficient. |

For conversion, preserve request identities, units, ordering and the complete
dependency/session structure. Validate required lengths, nonfinite/negative
values, duplicate IDs and missing parents. Keep a source-to-output mapping
and count rejected rows with reasons. Preserve originals and token privacy.

For Mooncake, `hash_ids` is required, the first session row under timestamp
load needs a millisecond timestamp, and output length is required. Hash blocks
describe prompt identity at `source.block_size`; they are not arbitrary
request IDs. Do not fabricate shared prefixes from length-only data. If genuine
cache identity cannot be recovered, state that limitation and agree on an
explicit cache assumption before using the result for cache-sensitive decisions.

## Preserve load semantics

Use `trace_timestamps` with `speedup: 1.0` to preserve authored pacing.
Concurrency load replaces first-arrival pacing and rejects `speedup`; use it
only for the user's chosen closed-loop experiment. A timestamp speedup also
scales documented inter-turn/dependency waits, not just initial arrivals.
Same-timestamp requests retain source order, which can affect scheduling.

Start with finite replay unless the user requested a continuous campaign.
Without `agentic_profile`, omitting `traffic.stop` runs to the end of the trace.
Do not introduce a cutoff that silently censors a workload. For a bounded smoke
test, select one or more complete sessions/plays and state that selection.

An illustrative traffic block for an already validated full-prompt Mooncake
trace is below. Merge it into the user's concrete engine config; replace the
path and block size with the actual trace's values.

```yaml
traffic:
  source:
    type: trace
    format: mooncake
    paths: [/absolute/path/requests.jsonl]
    block_size: 512
  load:
    type: trace_timestamps
    speedup: 1.0
```

Agentic lanes, seeded snapshots and warmup change how the corpus is exercised.
For those workflows, follow the
[AgentX quickstart](https://github.com/ai-dynamo/aisim/blob/main/docs/replay/agentic/quickstart.md)
and [continuous-profile contract](https://github.com/ai-dynamo/aisim/blob/main/docs/replay/agentic/continuous-profiles.md).
Continuous profiles require positive lanes and a seeded snapshot; do not add
them to reproduce an originally finite turn-zero run. Preserve source selection,
seed, warmup, initial cuts, idle caps and preparation/profile/grace boundaries.

## Run and qualify the replay

Check the exact model/hardware/backend path and required timing/profile data.
Keep an explicit failure for missing coverage rather than changing precision,
backend or timing method to obtain a successful run. Fixed latency can test
workload plumbing, but cannot establish hardware performance or timing accuracy.

With the validated input saved as `replay.yaml`:

```bash
aisim predict --stack engine --config replay.yaml \
  --capture-per-request --output-dir workload-replay
```

Retain the input, conversion/mapping when applicable, version and data pins,
exact command/exit status, `prediction.json` and `requests.jsonl`. Use a fresh
output directory when changing a run. Add supported telemetry only when needed.

Reconcile source rows, selected plays, issued requests, completed requests,
latency samples, failures and unfinished work. These counts need not match
when snapshots, warmup or recycling are enabled; explain the accounting.
Report the selected workload, load/cache assumptions, functional completion
and unresolved coverage. Successful replay alone does not establish agreement
with the original deployment's measured latency.
