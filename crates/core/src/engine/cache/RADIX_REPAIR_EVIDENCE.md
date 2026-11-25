<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Scoped radix repair evidence

The production correction is `e2f62f787b54bc6440d8e3b290e5327b3ee6e8dc`,
descended from pinned PR head `9664b5ba9bc33ac02e7e0f727679c00587b08733`.
The SGLang dependency remains `f0fda25fc95c357368b714f90a13a69c1b67f50a`.
These synthetic tests establish cache behavior and host cost, not measured
serving accuracy.

## Recency and request-key ownership

`PRRT_kwDOT0t9uc65JEIW`: Belady carries the retained prefix's original recency
across native suffix splits, while recomputing next use from each exposed tail.
A real core access supersedes the saved recency. Evicted physical pages lose
that metadata immediately; snapshots preserve it. Tests cover repeated splits,
separate eviction calls, real accesses, and checkpoint restoration. The existing
selection fixture now expects `[1,0]`, then `[2]`, matching main's equal-priority
order. No numerical prediction golden changed.

`PRRT_kwDOT0t9uc65Jpg9` and `PRRT_kwDOT0t9uc65KZVn`: the lease owns an
`Arc<Vec<LocalBlockHash>>`. A query shares that allocation and exposes only its
materialized prefix. The pinned native insert constructs its owned walk key
from the new suffix, so it never retains the lease's full buffer. Ordinary
growth therefore appends into the uniquely owned vector. Native anchor
validation, topology, lock transfers, and overlapping-suffix reconciliation
are preserved. A regression counts all owned key atoms constructed during
one-page extension and bounds them independently of retained context length;
it also verifies stable reserved storage, unique ownership after insertion,
overlap canonicalization, cleanup, and repeat reuse.

## Lifecycle outputs

`PRRT_kwDOT0t9uc65JEI1` and `PRRT_kwDOT0t9uc65JphC`: the identical
[probe](../kv_manager/radix_lifecycle_evidence.rs) runs at pre-integration main
`8288046ac88fccaee124714cd60e62b2ad4e184d` and at the production correction.
It admits leases with the normal `input_len - 1` cap, uses four-token pages and
24 token slots, and checks actual published removal hashes and repeat-admission
reuse. It drains every cache back to its original capacity. The
[machine-readable result](radix-lifecycle-comparison.json) records exact hashes,
source revisions, the probe checksum, and commands.
Its isolated-build rerun uses probe revision
`f2254e0d6e47bea54c5fe0af7a6db5cbf736493e`, whose production code is unchanged
from the correction.

| Scenario | Main | Native adapter |
| --- | --- | --- |
| Cache an eight-token prompt, finish, evict four tokens, repeat the prompt | Removes the second hash, frees four tokens, reuses four | Removes both hashes in prefix order, frees eight tokens, reuses zero |
| Cache A, admit A plus one incomplete token, cache unrelated B, finish A, evict four tokens, repeat A | Removes A, reuses zero | Removes B, reuses four |

Whole-leaf eviction intentionally follows SGLang's compressed-leaf policy.
Completion intentionally refreshes only the retained endpoint when it adds no
complete page, as implemented by prior commit
`335a02280b23cbc2f806bf32a2b4523869cdcbe9`. Adding a complete page still refreshes
the full path. The existing endpoint/ancestor regression covers all three cases:
zero, one incomplete, and four newly completed tokens. No other lifecycle
output difference was observed by these probes.

Reproduce both strict comparisons from the repository root, using a fresh
external work directory each time:

```bash
python3 scripts/compare_radix_eviction.py \
  --baseline 8288046ac88fccaee124714cd60e62b2ad4e184d \
  --candidate e2f62f787b54bc6440d8e3b290e5327b3ee6e8dc \
  --work-dir "$TMPDIR/selection-evidence"
python3 scripts/compare_radix_eviction.py --probe lifecycle \
  --baseline 8288046ac88fccaee124714cd60e62b2ad4e184d \
  --candidate f2254e0d6e47bea54c5fe0af7a6db5cbf736493e \
  --work-dir "$TMPDIR/lifecycle-evidence"
```

The helper isolates each snapshot's Cargo artifacts beneath `CARGO_TARGET_DIR`
and disables incremental compilation, including when a caller supplies a shared
target root. This prevents reuse of binaries containing another snapshot's
embedded manifest directory.

## Growth cost and retained state

The affected operation runs whenever SGLang materializes another complete page
during prefill or decode. The former query construction copied every retained
hash at each extension, accumulating quadratic hash copying over a growing
request. Sharing the lease buffer removes that copy; native-owned key
construction now copies only new suffix hashes. Native ancestor validation,
lock walks, tree mutations, and event handling still run. This does not claim
that total request growth is linear.

The representation adds one `Vec` header per owned key allocation. The full
request buffer remains owned only by its lease between calls, as asserted by
the regression; nodes retain their suffix storage. Belady's optional correction
map is bounded by resident physical pages and is absent for LRU. Cleanup and
checkpoint tests cover both state lifetimes.

The [cache microbenchmark results](radix-growth-comparison.json) compare pinned
PR head with the correction using distinct optimized test binaries. Both were
built before measurement, their hashes differ, and only the candidate inventory
contains the new recency and ownership regressions. Rust 1.91.1, release profile,
one codegen unit, thin LTO, the same lockfile, and test-only inspection were used.
The candidate also includes the ownership counter. Measurements used CPU 0,
one warmup, then five alternating paired samples in fresh processes on WSL2
with a two-CPU quota and 5 GiB container memory limit.

Each row times 50 complete cache lifecycles: allocation, initial caching,
one-page extension, finish, full eviction, and cleanup. Pages contain 32 tokens.
The 64-token control adds two pages; the larger contexts add 128. Overlap means
another request inserts a suffix halfway through growth, requiring adoption of
its canonical pages and freeing duplicates. All lifecycle assertions pass.
Median milliseconds for 50 lifecycles follow; raw ranges are in the JSON.

| Initial tokens | Overlap | Pinned head | Correction |
| --- | --- | --- | --- |
| 64 | No | 0.264 | 0.277 |
| 64 | Yes | 0.195 | 0.209 |
| 4096 | No | 8.424 | 7.902 |
| 4096 | Yes | 8.696 | 8.370 |
| 32768 | No | 11.168 | 8.583 |
| 32768 | Yes | 10.274 | 9.672 |

The small control has slight added ownership overhead. Larger-context medians
improve, including overlap reconciliation. Process peak RSS was 36,300 KiB on
both sides, but that coarse measure includes launcher overhead and does not
establish exact retained-memory parity. Reproduce a single microbenchmark
sample using the archived probe from
`f2254e0d6e47bea54c5fe0af7a6db5cbf736493e`, with SHA-256
`a83069941bdad532ec2c8e4532eded72670f380db907771e20da2b3b090dfb38`.
This is the probe recorded in the JSON: 50 lifecycles for each of the six
context/overlap combinations above, with two growth pages for the control and
128 for the larger contexts. Overlay exactly these bytes on both recorded
production revisions; their source and lockfiles otherwise remain unchanged.
From the repository root, with `TMPDIR` outside the checkout:

```bash
set -euo pipefail
growth_work=$(mktemp -d "${TMPDIR:?}/radix-growth.XXXXXXXX")
growth_targets="${CARGO_TARGET_DIR:-$growth_work/target}/$(basename "$growth_work")"
probe_revision=f2254e0d6e47bea54c5fe0af7a6db5cbf736493e
probe_path=crates/core/src/engine/kv_manager/radix_lifecycle_evidence.rs
git show "$probe_revision:$probe_path" > "$growth_work/probe.rs"
printf '%s  %s\n' \
  a83069941bdad532ec2c8e4532eded72670f380db907771e20da2b3b090dfb38 \
  "$growth_work/probe.rs" | sha256sum --check

for side in baseline candidate; do
  if [ "$side" = baseline ]; then
    revision=9664b5ba9bc33ac02e7e0f727679c00587b08733
  else
    revision=f2254e0d6e47bea54c5fe0af7a6db5cbf736493e
  fi
  snapshot="$growth_work/$side"
  mkdir "$snapshot"
  git archive "$revision" Cargo.toml Cargo.lock crates/core \
    python/aisimulate/tests/fixtures | tar -x -C "$snapshot"
  cp "$growth_work/probe.rs" "$snapshot/$probe_path"
  module="$snapshot/crates/core/src/engine/kv_manager/sglang_backend.rs"
  if ! grep -q '^mod radix_lifecycle_evidence;$' "$module"; then
    printf '\n#[cfg(test)]\n#[path = "radix_lifecycle_evidence.rs"]\nmod radix_lifecycle_evidence;\n' >> "$module"
  fi
  (cd "$snapshot"
    CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0 \
      CARGO_TARGET_DIR="$growth_targets/$side" \
      cargo test --release --locked --lib radix_growth_measurement --no-run \
      > "$growth_work/$side-build.log" 2>&1)
done

# Build both optimized binaries before timing either side.
for side in baseline candidate; do
  (cd "$growth_work/$side"
    CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0 \
      CARGO_TARGET_DIR="$growth_targets/$side" \
      AISIM_EVICTION_EVIDENCE_OUTPUT="$growth_work/$side.json" \
      cargo test --release --locked --lib radix_growth_measurement -- \
      --ignored --nocapture > "$growth_work/$side-run.log" 2>&1)
done
printf 'Growth observations and build/run logs: %s\n' "$growth_work"
```

This produces one fresh sample per side, with the historical lifecycle
assertions intact. To compare timings, use the matched affinity/resources,
warmup and alternating paired sampling described above; the historical timing
values are host-specific. The current checkout's `radix_growth_measurement` is
a separate benchmark: it uses 20 repetitions, initial contexts of 64 and 4096
tokens, and growth sizes of 2, 128, 512 and 2048 pages. Running it directly does
not reproduce this historical table.

## Production CLI comparison

[The CLI report](radix-cli-comparison.json) records the exact configurations,
model metadata and provenance, native hashes, all 20 timing/memory samples,
and both canonical prediction reports. Release wheels were built first and
installed into separate scratch environments. The baseline native SHA-256 is
`7e4cb269adade83d3c0754f4103628f6c9ff697b41ec45fe4e718c92c8098660`;
the candidate is
`093c1556ef16ecb1c66ab8a1e4f7c0237b278cfa33c046b3b6de6bd5cf6162a0`.
These production extensions have no test-only inspection or ownership counter.

The real consumer was `python -m aisimulate predict --stack engine --config
CONFIG --output-dir OUTPUT --format json`. Both sides use one SGLang worker,
concurrency one, page size 32, 8192 cache blocks, fixed 1 ms prefill/decode
timing, and 32 synthetic requests. The model is the repository's tiny config
with its context limit extended to 131072; no weights were downloaded. The
control has 64 input and 64 output tokens per request; growth has 32768 input
and 4096 output tokens. Both use a 1.2 GB execution budget and one CPU with the
default 1 GB host reserve. All builds and other test jobs finished before these
CPU-0 measurements: one warmup per side/case, then five alternating paired runs.

Every run completed all requests and input/output tokens, exited successfully,
and reported complete supervised termination. Every prediction field matches
exactly between sides and repetitions except the three host timer/rate fields
`wall_time_ms`, `processed_tokens_per_s`, and `processed_output_tokens_per_s`.
The growth case processes 1,048,576 input and 131,072 output tokens. This fixture
qualifies engine execution and host cost, not serving prediction accuracy.

| Case / metric | Pinned head median (min–max) | Correction median (min–max) |
| --- | --- | --- |
| Control / native wall ms | 1.503 (1.450–1.597) | 1.556 (1.498–1.618) |
| Control / full CLI seconds | 2.193 (2.167–2.334) | 2.169 (2.165–2.274) |
| Growth / native wall ms | 83.740 (82.961–85.334) | 82.054 (81.402–89.433) |
| Growth / full CLI seconds | 2.271 (2.266–2.273) | 2.270 (2.242–2.311) |
| Control / supervised peak RSS MB | 197.927 (196.555–198.058) | 197.493 (197.267–197.841) |
| Growth / supervised peak RSS MB | 197.542 (197.284–197.808) | 197.513 (197.263–197.546) |

Native wall time covers `Replayer::run_inner` through collector finalization;
full CLI time additionally covers model/session preparation, Python startup,
and serialization. RSS is the supervisor's sampled process-tree peak, not a
precise allocation counter. The CLI ranges overlap: these measurements show
equivalent completed work and no material integrated regression, not a
statistically established CLI speedup. The microbenchmark and ownership bound
provide the more direct evidence for removing repeated key copies.

To reproduce, create the report's `generated_model` as a local `config.json`,
set each recorded configuration's `engine.model` to that directory, and use
separately built release wheels from the recorded revisions. Run the command
above with fresh output directories, matched affinity/resources, a warmup, and
alternating paired samples. Do not compare shared or unidentified native builds.

## Current validation and environment corrections

The corrected production code passed the full Rust workspace: 1792 unit tests,
103 integration tests, and three doc tests; two existing/manual cases were
ignored. The 386 native engine-step/compile-engine numerical parity cases
passed with the original assertions and 120-second timeout after building the
optimized wheel. All 16 numerical sentinels passed. Targeted engine/replay,
recency/ownership, and isolated lifecycle comparisons passed. Rust formatting,
Ruff, copyright/legal, CODEOWNERS, and documentation checks passed.
The combined CODEOWNERS, workflow-contract, qualification, and release-policy
suite passed all 434 tests; Python syntax compilation also passed with bytecode
redirected into scratch storage.

Earlier unsuccessful invocations are retained in the repair scratch logs:

- Initial caller/probe compile errors and Ruff formatting were corrected before
  the passing regressions; missing baseline build tooling was installed in its
  scratch environment before rebuilding the release wheel.
- Shared Cargo artifacts initially reused another snapshot's binary and embedded
  data path. Those timings were discarded. The first full workspace attempt
  consequently had 354 data-path failures (1438 passes). After clearing that
  crate's invalid dev artifacts and disabling incremental reuse, the full
  workspace rerun passed. Final comparison snapshots have isolated targets;
  measured release binaries were rebuilt and checked by hash and test inventory.
- A 2 GB CLI budget was rejected by the unchanged host-headroom guard, including
  a retry after builds finished. A 1 GB budget completed the control but failed
  the growth session's allocation estimate. Both final workloads use 1.2 GB,
  satisfying the unchanged guards and completing every paired run. No workload,
  test assertion, timeout, or memory reserve was reduced to obtain these results.
- The first broader policy run had 429 passes and five failures because its
  required `pip-licenses` and Node.js tools were absent. Installing
  `pip-licenses==5.5.5` and Node.js 24.19.0 into the scratch environment and
  adding its bin directory to `PATH` produced the passing 434-test rerun.

This repair covers only the five selected findings. Existing dependency release
contract/indexing decisions and merge conflicts remain outside this repair.
Hosted workflow/log access returned HTTP 403; no hosted CI success, approval,
or merge readiness is claimed. Fresh CI, reviews, and technical reapproval
remain with the controller.
