<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Completed-prompt warm readmission

This follows up `PRRT_kwDOT0t9uc67wHNZ`, separately from the growth-only
measurement in [exclusive growth qualification](RADIX_EXCLUSIVE_GROWTH_EVIDENCE.md).
The baseline is legacy main `8288046ac88fccaee124714cd60e62b2ad4e184d`, an
ancestor of the PR. The native baseline is
`e1c69bdb17a0c48dd653e8558c36ec9c58c5110c`, and the corrected native source is
`3c800f55ba78ccb1e90d999df5174935807d14a2`. All three arms use the same immutable
probe from the native baseline; its hash is recorded below.

## Work and timing boundaries

The identical probe starts with a 64-token request and appends 2, 128, 512, or
2,048 pages, one page per call. Pages contain 32 tokens; capacity is 131,072
tokens. After finishing that request, it separately times 100 cache checkpoints
(including destruction of each snapshot), then 100 complete-prompt warm
admissions. Admission timing includes begin, allocation/matching, the manager's
transactional checkpoint, and commit. Finish is outside that timer. Initial
population and growth are outside both timers.

Every warm admission must reuse the entire prompt. Each run checks that warm
admissions do not change node count and that final eviction restores the full
capacity. This manager probe deliberately permits a complete-prompt hit; the
production SGLang scheduler caps matching before the last input token. It
isolates the requested cache lifecycle rather than claiming serving accuracy or
end-to-end replay throughput.

All three builds use Rust 1.91.1, release opt-level 3, codegen-units 1, thin LTO,
the empty default feature, and pinned registry packages. The native arms have
identical lockfiles. Legacy main has the same versions/checksums for all shared
registry packages. Native adds hashbrown 0.16.1, sha2-asm 0.6.4, thiserror 1.0.69
and its derive crate, and enables the sha2 asm feature. The report records both
lockfile hashes and the added packages; the legacy-to-native comparison includes
these intended dependency changes. Each arm has one warmup followed by five
measured triplets with alternating forward/reverse order, pinned to CPU 0 on
the same WSL2 container
(two-CPU quota, 5 GiB memory cap). All builds finish before timing. The
[machine-readable report](radix-warm-readmission-comparison.json) retains every
sample, source and probe checksums, binary hashes/sizes, and host settings.

## Results and correction

The completed-prompt measurement confirms the reported topology cost. Legacy
main retains two nodes including the root, while both native arms retain
`growth_pages + 2`. The correction preserves every native node, recency boundary,
lock, and eviction victim. Child-edge maps now share immutable storage across
checkpoints and copy only on insertion, removal, or splitting. This removes one
hash-map clone and allocation per non-leaf node from each checkpoint. Keys and
page-value buffers were already shared. Snapshot-isolation tests exercise
independent splits, new branches, and eviction across three cache versions.

Median microseconds per call, including snapshot destruction:

| Added pages | Legacy checkpoint | Native before | Native corrected | Legacy warm admission | Native before | Native corrected |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.175 | 0.502 | 0.583 | 0.235 | 0.835 | 0.788 |
| 128 | 0.080 | 9.938 | 5.872 | 0.845 | 16.489 | 12.668 |
| 512 | 0.080 | 39.225 | 23.847 | 2.628 | 67.680 | 53.533 |
| 2048 | 0.081 | 180.064 | 108.614 | 9.661 | 265.582 | 221.160 |

All five measured samples remain in the report. At 2,048 added pages, native before warm admission spans 252.056–289.678 µs; candidate warm admission spans 191.917–236.938 µs.

Checkpoint cost remains O(resident node metadata), and matching/locking is
O(reused native nodes). Complete rollback requires preserving recency, locks,
and possible eviction changes. The chosen snapshot shares immutable keys,
values, and topology while keeping mutable metadata independent. A mutation
journal would change every affected mutation/rollback boundary; this correction
keeps the existing transactional contract. Coalescing the chain would change
native eviction boundaries. The remaining native cost is visible here; these
results do not establish legacy-speed or constant-time admission.

Native test builds retain the PageKey exposure counters used by the earlier
growth tests; the legacy implementation has no such instrumentation. Production
wheel comparisons below omit those counters. Treat the manager numbers as a
focused lifecycle measurement, not a serving-accuracy or replay-throughput claim.

Peak binary-process RSS across the five measured runs (KiB, min–max):

- baseline: 15584–16004.
- native before: 15664–16128.
- candidate: 15724–16216.

These peaks include population, all four prompt sizes, snapshots, admission,
and cleanup. They are process peaks, not isolated per-node retained memory.
The shared map adds reference-counted storage per node; dropping a checkpoint
releases its references, and structural edits detach only the modified maps.

## Production consumer and host cost

[Repeated optimized-wheel CLI measurements](radix-checkpoint-cli-comparison.json)
compare native production source before and after the correction through
`predict --stack engine`. The before wheel was built at
`2f5cb91a7339a16adcf2ffaafffb76f40ddd221c`; its production source is identical to
the test-only successor `e1c69bdb17a0c48dd653e8558c36ec9c58c5110c` used by the
manager probe. The report names the actual artifact revision. The Full-only CLI
workloads do not invoke the separate SWA ring validation correction.
Each arm has one warmup and five alternating measured runs per size, on CPU 0,
with identical release optimization, dependencies, model metadata, and explicit
`extension-module,python` features. Loaded extension hashes match their wheels.
The report retains every host timing and peak RSS sample, full configurations,
and all stable summary fields. It excludes only the three host timing/rate
fields when checking summary equality.

Every run completes eight requests, 512 input tokens, and exactly eight times
the configured output count. Page size is one, capacity is 32,768 blocks,
concurrency and worker count are one, and timing is fixed at 1 ms. These growing
requests exercise repeated appends, resident state, admission checkpoints, and
terminal cleanup. The native recency/eviction policy remains unchanged.

Median native `wall_time_ms`:

| Output tokens/request | Before, ms | Corrected, ms | Full CLI before/corrected, ms | Peak CLI RSS before/corrected, KiB |
| --- | --- | --- | --- | --- |
| 64 | 0.989 | 1.569 | 3466.3 / 3459.3 | 147668 / 145952 |
| 512 | 8.396 | 10.150 | 3943.1 / 3596.0 | 147768 / 147412 |
| 2048 | 25.796 | 28.008 | 2986.4 / 2846.6 | 154360 / 154920 |

The correction is a workload tradeoff, not a general replay speedup. Sharing
maps adds one reference-counted allocation per node and reference-count work
while eliminating map allocations and copies at each checkpoint. The growing,
non-reusing CLI cases above have higher median native times despite improved
warm-readmission costs. Their measured native-time ranges overlap:

| Output tokens/request | Before min–max, ms | Corrected min–max, ms |
| --- | --- | --- |
| 64 | 0.961–1.926 | 1.063–2.489 |
| 512 | 5.959–13.607 | 6.297–12.162 |
| 2048 | 21.925–41.349 | 23.598–36.332 |

These short native intervals remain sensitive to host variation. Full CLI
medians are similar or lower, but Python startup dominates them and cannot
establish unchanged native execution cost. The explicit remaining cost is the
chosen tradeoff for repeated checkpoint savings. A full mutation journal was
considered; it would require redesigning recency, lock, split, and eviction
rollback together. Coalescing nodes would change the intended native eviction
policy. Neither contract is changed by this repair.

The native timer covers `Replayer::run_inner` through collector finalization;
it excludes timing-model construction, Python startup, input preparation, and
serialization. Full CLI timings include those phases and are dominated by
startup on these small fixtures. RSS uses a fresh Python wrapper with one child and Linux wait4 resource
accounting: `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` after child exit.
Full CLI elapsed time also includes the wrapper's startup. RSS covers the whole
CLI process. Fixed timing
is a semantic control: this comparison makes no hardware-accuracy claim.
All non-host summary fields match across every repetition and both arms.
The production artifact sizes and `.text` measurements are recorded beside
the verified extension hashes; test-binary sizes include different test suites.

To reproduce the production rows, use the isolated release-wheel build,
installation, and loaded-extension hash procedure in
[exclusive growth qualification](RADIX_EXCLUSIVE_GROWTH_EVIDENCE.md#verification-and-reproduction),
substituting the two native revisions named here. Materialize each `rows[].config`
from the CLI report, replacing only `$SCRATCH/tiny-model` with a local directory
containing `model_provenance.config` as `config.json`. Run the report's command
with a fresh output directory for every warmup and sample, matching the paired
schedule. Preserve all resource limits and canonical-summary exclusions.

## Reproduction

Run from the repository root with all immutable revisions available locally.
Keep snapshots and targets outside the source tree. This explicitly overlays
the same probe into legacy main; merely naming the test on that revision would
run zero matching tests.

```bash
set -euo pipefail
warm_work=$(mktemp -d "${TMPDIR:?}/warm-readmission.XXXXXXXX")
warm_targets="${CARGO_TARGET_DIR:-$warm_work/target}/$(basename "$warm_work")"
probe_revision=e1c69bdb17a0c48dd653e8558c36ec9c58c5110c
probe_path=crates/core/src/engine/kv_manager/radix_lifecycle_evidence.rs
git show "$probe_revision:$probe_path" > "$warm_work/probe.rs"
printf '%s  %s\n' \
  9d1182aa5d96e111beb458483f662a52275ebe8ca4032a8b5e7d48f2611b0267 \
  "$warm_work/probe.rs" | sha256sum --check
for side in baseline native_before candidate; do
  revision=8288046ac88fccaee124714cd60e62b2ad4e184d
  if [ "$side" = native_before ]; then revision=$probe_revision; fi
  if [ "$side" = candidate ]; then revision=3c800f55ba78ccb1e90d999df5174935807d14a2; fi
  snapshot="$warm_work/$side"
  mkdir "$snapshot"
  git archive "$revision" Cargo.toml Cargo.lock crates/core python/aisimulate | tar -x -C "$snapshot"
  cp "$warm_work/probe.rs" "$snapshot/$probe_path"
  module="$snapshot/crates/core/src/engine/kv_manager/sglang_backend.rs"
  if ! rg -q '^mod radix_lifecycle_evidence;$' "$module"; then
    printf '\n#[cfg(test)]\n#[path = "radix_lifecycle_evidence.rs"]\nmod radix_lifecycle_evidence;\n' >> "$module"
  fi
  (cd "$snapshot"
    CARGO_TARGET_DIR="$warm_targets/$side" cargo test --release --workspace --locked \
      --lib radix_warm_readmission_measurement --no-run --jobs 1 \
      > "$warm_work/$side-build.log" 2>&1)
done
warm_test=engine::kv_manager::sglang_backend::radix_lifecycle_evidence::radix_warm_readmission_measurement
for pair in 0 1 2 3 4 5; do
  order='baseline native_before candidate'
  if (( pair % 2 )); then order='candidate native_before baseline'; fi
  for side in $order; do
    (cd "$warm_work/$side"
      CARGO_TARGET_DIR="$warm_targets/$side" \
        AISIM_EVICTION_EVIDENCE_OUTPUT="$warm_work/$pair-$side.json" \
        taskset -c 0 cargo test --release --workspace --locked --lib \
        radix_warm_readmission_measurement -- --ignored --exact "$warm_test")
  done
done
```

Pair zero is warmup. Divide each nanosecond total by its recorded repetition
count and by 1,000 to obtain microseconds per call; report all five remaining
samples and their median, separately for each size and arm. Preserve the
reported node counts rather than assuming the implementations have equal
topology. Record hashes of the executed test binaries under each isolated
target's `release/deps` directory before interpreting the measurements.

## Repair verification and setup retries

The corrected native source passes the optimized workspace suite (2,042 tests;
four manual benchmarks ignored) and all 13 public engine CLI prediction cases
against the corrected optimized wheel, including transfer rejection, auxiliary
lock/cursor transitions, SWA ring path validation, suspended tombstone
recovery, and checkpoint snapshot isolation. The explicit manual probe above
passes for all three revisions, reuses every prompt, preserves node counts,
and evicts back to full capacity. No numerical goldens or timeouts changed.

Early fixture/setup failures and successful reruns remain in the repair
scratch logs. Backup fixtures required normal write-through acknowledgments;
SWA fixtures required valid ancestor windows. The CI-policy test environment
needed Node and pip-licenses, and documentation checks needed markdown-it-py.
The final runs use the scratch venv and its dependencies, with pytest plugin
autoload disabled only for the standalone policy tests. A crate-only wheel was
replaced by the mixed application wheel built from `python/aisimulate`; loaded
extension hashes are verified in both CLI reports. Measurement preflight
corrected an overstrict identical-lockfile assertion (the declared native
dependencies differ), used wait4 because GNU time is unavailable, accepted the
public CLI's flat summary format, and gave each run a fresh output directory.
Those preflight failures are retained and are not measured samples. All final
benchmark assertions remain enabled, and no final timing sample is discarded.
