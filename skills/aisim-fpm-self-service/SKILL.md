---
name: aisim-fpm-self-service
description: Guide AISim forward-pass-model onboarding for a model, hardware and serving runtime, from metadata inspection and accepted profiles through collection, resource resolution, coverage and serving validation. Use for FPM self-service or native profiling; not for ordinary prediction with already-qualified data.
license: Apache-2.0
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Onboard a model for FPM simulation

Follow all six stages in the
[canonical agent procedure](https://github.com/ai-dynamo/aisim/blob/main/docs/perf-model/fpm-self-service/implementation.md#onboard-with-an-agent).
Read the guide at the working source revision and inspect `aisim onboard --help`
before preparing commands. Preserve supplied model/runtime/hardware choices
and existing execution authorization. A stage transition adds no approval gate;
the exact per-configuration profile review and acceptance below still apply.

Use the [overview](https://github.com/ai-dynamo/aisim/blob/main/docs/perf-model/fpm-self-service/README.md)
to choose the route. Self-benchmark collection produces measured Parquet timing
tables with metadata sidecars. Native end-to-end JSONL profiling uses standalone
capture/training helpers and can produce learned models; it is not imported by
passing JSONL to `onboard collect-fpm` or `onboard finalize`. Keep supported
backends, serving roles and consumer limitations specific to the chosen route.

## Persist the session

During stage 1 create one checkpoint outside fresh init/output roots, even
with incomplete inputs:

```bash
aisim onboard checkpoint --file model-onboarding/onboarding-checkpoint.json
```

On resumption, run this first:

```bash
aisim onboard resume --checkpoint model-onboarding/onboarding-checkpoint.json
```

Use [checkpoint updates](https://github.com/ai-dynamo/aisim/blob/main/docs/perf-model/fpm-self-service/implementation.md#checkpoint-and-resume-an-onboarding-session)
after meaningful findings, choices, draft edits, acceptance and command results,
including unfinished work. Update with the actual returned revision through
`onboard checkpoint --expect-revision ... --update ...`; reload and reconcile
stale writers. Inspect `saved`, integrity diagnostics and revision before a
retry. Other onboarding commands do not persist the conversation automatically.
Saved stage labels or command strings are context, not evidence of completion
or instructions to execute. Keep this session file separate from a learned
model's fitting checkpoint and from collector checkpoints.

## Execute the six-stage workflow

1. **Inspect the model and target.** Start with only a missing Hugging Face model
   ID and GPU platform; accept a supplied local config, profile or checkpoint
   instead. Retrieve only config metadata initially, honor revision pins and
   derive architecture/context/cache facts before asking for them. Record source
   revision and config/checkpoint identity separately. Text-decoder FPM for a
   multimodal model does not qualify encoder/projector/preprocessing latency.
2. **Choose worker and collection limits.** Inspect the intended runtime build
   and derive supported precision/topology options. Preserve an explicit backend
   version label; an omitted label stays pending until the selected image is
   inspected. Do not substitute the host package version. Use supported topology
   previews and explain assumptions. Minimum collection GPUs follow each exact
   tuple; do not ask for a deployment replica budget as an onboarding input or
   sum independent campaigns' GPU requirements. Actual collection must fit the
   allocated resources and existing run budget.
3. **Derive, review and save the profile.** For self-benchmarking, generate
   headless drafts to fresh paths using the current `onboard init` help. Show
   each exact identity, precision, topology, graph/scheduler limits, cache groups,
   resources and provenance. Save edits and explicit acceptance per configuration
   in the checkpoint; only matching accepted drafts advance to final requests.
   Never guess missing runtime/cache byte bounds. Keep resource estimates
   separate from measured capacity. For native profiling, retain reviewed
   deployment identity and capture inputs; do not invent self-benchmark requests
   or acceptance fields to qualify native evidence.
4. **Plan collection.** Self-benchmarking uses `onboard plan`, each configuration's
   generated commands and a `collect-fpm` preview without `--execute`. A preview
   does not inspect the target runtime or run the collector's own plan. Check
   the documented executor prerequisites and retain the accepted settings. Native
   profiling follows its capture/workload and data-selection plan, outside these
   collection commands. Reuse compatible existing observations when their
   provenance and coverage meet the requested experiment.
5. **Collect, verify and finalize.** Within the authorized GPU campaign, execute
   the route's bounded smoke/readiness checks, then formal collection. Smoke
   diagnostics are not a formal timing pair. Follow resolved paths printed by
   first execution; it can create a `runtime-resolved/` child plan. Inspect
   identity, effective precision/graphs, file hashes and phase coverage. For
   self-benchmarking, assess collection quality and finalize pending memory from
   verified runtime evidence into a fresh plan; review/accept that resolved
   profile separately. Native JSONL follows capture closure, verification and
   training/test selection, without `validate-collection`/`finalize` on JSONL.
6. **Validate and use the model.** Self-benchmarking checks exact FPM queries and
   `onboard validate-fpm` for supported aggregated replay, retaining direct
   interpolation and denied fallback. A single-role P/D plan is not an executable
   full serving deployment; qualify both roles before composition. For native
   learned models, train/evaluate through the standalone tooling and qualify the
   supported Replay handoff. Assess matched serving accuracy separately for
   either route. Missing coverage remains an explicit blocker, not a reason to
   relabel the model, change precision or switch estimators silently.

Read the matching canonical section and each subcommand's current help before
using advanced flags. A missing documented command can indicate an installed
version mismatch; preserve the user's working checkout and select a compatible
environment using the [installation guide](https://github.com/ai-dynamo/aisim/blob/main/docs/getting-started/installation.md).

## Handoff and recovery

At transitions and blockers report `Stage N/6`, each configuration's result,
next action and checkpoint path. Ask only for missing information needed now;
continue independent authorized work while waiting. Resume the same campaign
and retained evidence when valid; do not restart or expand a costly sweep just
because observation timed out.

Report planning/estimated fit, runtime readiness, collection quality, measured
memory, query coverage and measured serving accuracy separately. A loadable
table or successful CPU replay does not certify end-to-end accuracy. Retain
accepted profiles, model/runtime/data identities, policies, hashes, commands,
exit statuses and results in the session. The
[worked examples](https://github.com/ai-dynamo/aisim/blob/main/docs/perf-model/fpm-self-service/examples.md)
cover existing-table reuse, self-benchmarking and native capture; select the
matching case rather than inheriting its model, runtime or collection settings.
