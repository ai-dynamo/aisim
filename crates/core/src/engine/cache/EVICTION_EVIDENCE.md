<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native radix eviction selection evidence

This is a synthetic cache-selection comparison for PR #363, not a measurement
of serving accuracy or simulator speed. The production cache sources are pinned
to main `8288046ac88fccaee124714cd60e62b2ad4e184d` and PR head
`e2f62f787b54bc6440d8e3b290e5327b3ee6e8dc`. That corrective commit preserves
Belady recency across suffix splits and shares request hash buffers for growth. The native dependency is SGLang
`f0fda25fc95c357368b714f90a13a69c1b67f50a`.

## Inputs and reproduction

[The scenario set](eviction-selection-scenarios.json) fixes seed 363, physical
capacity, page size, insert/access/lock operations, eviction requests, and
Belady future-input order. There is no random sampling. Page labels expand to
tokens with the recorded seed rule; physical IDs start at zero. Insertions
allocate real pool pages and reuse canonical shared prefixes. The probe checks
that freed tokens equal freed pages times page size, free capacity increases
by that amount, no page is freed twice within a call, and every scenario drains
to its original capacity after releasing its locks.

From the repository root, with Rust and Python 3.12 available:

```bash
python3 scripts/compare_radix_eviction.py \
  --baseline 8288046ac88fccaee124714cd60e62b2ad4e184d \
  --candidate e2f62f787b54bc6440d8e3b290e5327b3ee6e8dc \
  --work-dir "$TMPDIR/radix-eviction-evidence"
cargo test -p aisimulate-core --locked --lib engine::cache::radix_cache
```

Use a fresh work directory outside the checkout. The script archives each local
revision without fetching refs, overlays the identical test probe and fixtures,
and runs the recorded Cargo command with the revision's lockfile. It respects
`CARGO_TARGET_DIR` and limits Cargo build jobs to two. Snapshot sources, logs,
observed JSON, and the generated comparison stay in that work directory.
Neither the script nor the test updates committed expectations.

[The machine-readable comparison](eviction-selection-comparison.json) records
the source SHAs, toolchain, test profile, features, seed, probe/input checksums,
command lines, and all 22 eviction calls on each side. Each call includes
ordered physical victim IDs, requested and freed pages/tokens, LRU overshoot,
remaining prefix matches, and allocator capacity. Node IDs are deliberately
excluded because the two tree implementations use different node identities.
The ordinary Rust test checks candidate outputs against this record; the
reproduction script checks both recorded sides.

## Observed differences

| Scenario | Main | Native adapter | Explanation |
| --- | --- | --- | --- |
| `lru_compressed_overshoot`, first call requests one page | Frees `[2]` | Frees `[0,1,2]`, overshoot 2 | Whole compressed leaf eviction replaces suffix truncation. The next call frees `[3,4]` instead of `[1]`, so the final drain is empty. |
| `lru_locked_branch`, first call requests one page | Frees `[2]` | Frees `[1,2]`, overshoot 1 | The shared page 0 remains locked; the entire unlocked suffix is reclaimed. Later drains differ by the already freed page. Both reclaim page 0 only after unlock. |
| `lru_recycled_page` | Frees `[1]`, then `[0]`, then `[1]` | Frees `[0,1]`, then `[1]`, then `[]` | Whole-leaf eviction removes the old prefix earlier. Both reuse physical page 1 for the new prompt, but only main retains the older page 0. |
| `belady_equal_priority_recency` | Frees `[1,0]`, then `[2]` | Frees `[1,0]`, then `[2]` | The adapter preserves pre-split recency while recomputing priority from the new tail. |

The original native adapter chose `[1,2]`, then `[0]` for equal Belady priorities.
The correction to [the recency finding](https://github.com/ai-dynamo/aisim/pull/363#discussion_r4619189045)
(`PRRT_kwDOT0t9uc65JEIW`) makes this scenario match main. Additional tests cover
separate eviction calls, admission snapshots, and invalidation after real touches.
The adapter saves the original tie-break only for retained suffix-split prefixes,
keyed by their tail page and the core's split tick. A changed core access tick
supersedes it; freeing a physical page removes it. The table is bounded by resident
pages, and stale entries are reclaimed during each eviction step.

All calls in the other five scenarios match: single-page LRU recency, Belady
tail reranking, global forecast retirement, locked-prefix preservation with
four-token pages, and equal local hashes beneath different prefixes. The JSON
explains every differing call and its dependent prefix/capacity changes.

## Two-leaf split recency

The native default LRU deliberately retains the pinned SGLang split policy,
including its Full-KV access-counter refresh on the untouched suffix. This is
an additional behavior change from the legacy simulator, separate from
whole-leaf over-eviction. The component linked-list positions preserved by
`split_node_` do not determine Full LRU ranking; Full uses `last_access_counter`.
Belady retains its separate adapter correction and is unchanged by this case.

An identical synthetic probe was run against legacy main
`8288046ac88fccaee124714cd60e62b2ad4e184d` and published native head
`5f0da35ae7ef81cde8b68d49abb89ec168a0ff25`, using Rust 1.91.1, the unoptimized
test profile, each revision's lockfile, and no Cargo features. Each case starts
with capacity four and page size one. Insert `[A,B] = [10,20]` into physical
pages `[0,1]`, then `[C] = [30]` into physical page `[2]`. Match and lock the
indicated key, then request one page of eviction:

| Matched and locked key | Legacy physical victim | Native physical victim |
| --- | --- | --- |
| `[A]` (partial match; splits `[A,B]`) | `[1]` (B) | `[2]` (C) |
| `[A,B]` (full-match control; no split) | `[2]` (C) | `[2]` (C) |

Both revisions free exactly one page and retain the lock. After unlocking,
both drain the remaining two pages and restore all four free pages. A partial
match therefore changes subsequent prefix reuse under eviction pressure; it is
not equivalent to the legacy suffix-recency policy. The native adapter keeps
this production SGLang behavior as intended by the integration. No change to
the pinned tree's split algorithm or serving-accuracy claim follows from this
synthetic evidence.

`radix_cache::tests::lru_split_recency_victim_matches_native_policy` asserts the
native physical victim, surviving prefix, allocator accounting, and final
drain for both cases. Run it with:

```bash
cargo test -p aisimulate-core --locked --lib lru_split_recency_victim_matches_native_policy
```

To reproduce the comparison, archive each revision into a separate fresh
scratch directory (outside the checkout), append the identical probe below to
its `crates/core/src/engine/cache/radix_cache.rs`, and set `split_revision` to
that snapshot's full SHA. Run from the snapshot with a separate target directory
so embedded manifest paths cannot be reused from the other snapshot:

```bash
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$TMPDIR/radix-target}/lru-split-$split_revision" \
  cargo test -p aisimulate-core --locked --lib scoped_review_lru_split_victim_evidence -- --nocapture
```

The probe uses only APIs present in both revisions and prints the observed
physical victims without imposing either revision's victim policy:

```rust
#[cfg(test)]
mod scoped_split_recency_evidence {
    use super::*;
    #[test]
    fn scoped_review_lru_split_victim_evidence() {
        for partial_match in [true, false] {
            let mut cache = RadixCache::new(4, 1);
            let ab = cache.page_pool.allocate(2).unwrap();
            cache.insert(&[10, 20], &ab);
            let c = cache.page_pool.allocate(1).unwrap();
            cache.insert(&[30], &c);
            assert_eq!(ab, vec![0, 1]);
            assert_eq!(c, vec![2]);
            let key: &[u32] = if partial_match { &[10] } else { &[10, 20] };
            let (matched, locked) = cache.match_prefix(key);
            assert_eq!(matched, key.len());
            cache.inc_lock_ref(locked);
            let (freed, victims) = cache.evict(1);
            assert_eq!(freed, 1);
            assert_eq!(cache.available_tokens(), 2);
            println!("LRU_SPLIT partial={partial_match} victims={victims:?}");
            cache.dec_lock_ref(locked);
            assert_eq!(cache.evict(4).0, 2);
            assert_eq!(cache.available_tokens(), 4);
        }
    }
}
```

## Engine-golden coverage and validation

The existing `engine_step.json`, `compile_engine.json`, and `per_op.json`
goldens remain unchanged. They cover forward-pass timing, energy, compiled
operation graphs, and numerical query outputs. Their aggregate/disaggregate
estimates do not run the SGLang radix-cache eviction state machine and cover
none of the victim-selection paths above. The new scenario fixture and native
Rust regression supply that missing coverage through the actual `RadixCache`,
allocator, lock, and forecast implementations. They run with ordinary Cargo
unit tests in Full CI.

The original evidence-only repair at `9664b5ba9bc33ac02e7e0f727679c00587b08733`
used Rust 1.91.1, Python 3.12.12, the frozen Python lockfile,
and an optimized wheel. To reproduce the validation, activate a scratch Python
environment with the frozen development dependencies, pip, and maturin installed.
From the repository root, build into a fresh wheel directory, install that exact
wheel, and verify the loaded extension before running the suites. Every `python`
command below must use that same environment's interpreter:

```bash
set -euo pipefail
parity_wheel_dir=$(mktemp -d "$TMPDIR/radix-parity-wheels.XXXXXX")
(cd python/aisimulate && python -m maturin build \
  --release --locked --auditwheel skip --out "$parity_wheel_dir")
parity_wheels=("$parity_wheel_dir"/aisimulate-*.whl)
test "${#parity_wheels[@]}" -eq 1
python -m pip install --no-deps --force-reinstall "${parity_wheels[0]}"
python - "${parity_wheels[0]}" <<'PY'
import hashlib
from importlib.metadata import distribution
from pathlib import Path
import sys
from zipfile import ZipFile

import aisimulate._runtime as runtime

wheel_path = Path(sys.argv[1]).resolve()
loaded_path = Path(runtime.__file__).resolve()
member = f"aisimulate/{loaded_path.name}"
installed_path = Path(distribution("aisimulate").locate_file(member)).resolve()
with ZipFile(wheel_path) as wheel:
    wheel_sha256 = hashlib.sha256(wheel.read(member)).hexdigest()
loaded_sha256 = hashlib.sha256(loaded_path.read_bytes()).hexdigest()
print(f"interpreter: {sys.executable}")
print(f"wheel: {wheel_path}")
print(f"loaded extension: {loaded_path}")
print(f"loaded SHA-256: {loaded_sha256}")
print(f"wheel extension SHA-256: {wheel_sha256}")
if loaded_path != installed_path or loaded_sha256 != wheel_sha256:
    raise SystemExit("Loaded extension does not match the installed wheel")
PY
python -m pytest -q -rx -n 4 -c python/aisimulate/pytest.ini \
  crates/core/parity_tests/perfmodel/test_engine_step_parity.py
python -m pytest -q -rx -n 4 -c python/aisimulate/pytest.ini \
  crates/core/parity_tests/perfmodel/test_compile_engine_parity.py
python scripts/check_prediction_numerics.py --fetch-baseline-only
python scripts/check_prediction_numerics.py --output "$TMPDIR/numerical-sentinels.json"
```

The historical suites passed **301/301** and **85/85** cases, respectively, with their
original assertions and 120-second per-test timeout. All **16 numerical
sentinels** passed; their baseline commit was already present, so no fetch was
needed. Native wheel features were `extension-module` (including `python` and
PyO3's extension-module support); native radix defaults were disabled. The
loaded `aisimulate/_runtime.abi3.so` SHA-256 was
`89a8b87daf5a638887d48cd3fcf1d9c083eb07f44489b8918376ad738a760ceb`.
The selection probe uses the ordinary Cargo test profile and the native
dependency's test-only `inspection` feature, not the optimized wheel.

These are local results for the commands in `engine-golden-regression`. Hosted
workflow/log access returned HTTP 403 in the repair container; no hosted job
success is claimed. The controller must obtain current-commit Full CI, reviews,
and technical reapproval before merge.

Current corrective validation, lifecycle events, and growth measurements are in
[the scoped repair evidence](RADIX_REPAIR_EVIDENCE.md).

## Matching and locking a compressed prefix

The additional `match_and_lock_compressed_prefix` lifecycle probe caches two
pages A1/A2, then one unrelated page B. It admits a request reusing A1, leaving
that prefix locked, and evicts exactly one page before repeating A and B.
The capacity is 24 tokens, with four tokens per page. Actual manager admission,
physical page allocation, lock receipts, and `Removed` events are exercised;
the final drain returns all 24 tokens to the pool.

[The recorded comparison](radix-matched-prefix-comparison.json) runs the same
probe against main `8288046ac88fccaee124714cd60e62b2ad4e184d` and the corrective
source commit `a59c6d0641ceb9b775dc6740786cbdeba8f935cc`. It records the
probe checksum, Rust 1.91.1 toolchain, ordinary test profile, empty feature list,
commands, revision IDs, and observations from all three lifecycle scenarios.
The published starting head `54be42ae1d8dd1392e76860bed2810f801433b77`
was also archived and tested separately with its original lockfile and an
isolated target directory. Its complete observations match the corrective
source commit, as recorded under `published_starting_head`. Thus this ordering
predates these local corrections. Replacing `--candidate` below with that
starting SHA reproduces the published-head comparison.

| Observation after one-page pressure | Main | Native adapter |
| --- | --- | --- |
| Removed page | A2 | B |
| Removed block hash | `4945711292740353085` | `8906183847324352865` |
| Freed tokens | 4 | 4 |
| Repeat A reused tokens | 4 | 8 |
| Repeat B reused tokens | 4 | 0 |

Main preserves the unmatched suffix's old timestamp when splitting. The pinned
SGLang core stamps that suffix with a new access counter, making A2 newer than B
for the adapter's LRU ordering. A1 remains locked throughout the eviction.
This is an intentional native-policy difference that changes subsequent reuse;
it is independent of whole-leaf overshoot. The repair preserves that policy.
These synthetic observations establish cache lifecycle behavior, not serving
accuracy, timing-model parity, or simulator throughput.

Reproduce using a fresh scratch directory outside the checkout:

```bash
python3 scripts/compare_radix_eviction.py --probe lifecycle \
  --baseline 8288046ac88fccaee124714cd60e62b2ad4e184d \
  --candidate a59c6d0641ceb9b775dc6740786cbdeba8f935cc \
  --work-dir "$TMPDIR/radix-matched-prefix"
cargo test --workspace --locked --lib radix_lifecycle_matches_recorded_behavior
```

The first comparison attempt did not compile on main because it used the new
`protected_size()` method where main exposes a field. The successful probe
instead asserts the reused lease length and its node's lock reference through
interfaces shared by both revisions. The before/after comparison above comes
only from that successful rerun, using separate snapshot source and target
directories. Neither production revision's sources or lockfile were rewritten.

The local native corrections have separate regressions in
`sglang_radix_regressions.rs`; all ten original regressions fail on the unpatched
bundle. During fixture qualification, backup tests initially omitted the
acknowledgement that registers host coexistence; the final fixtures execute
`mark_write_through_pending` and `finish_write_through`. A proposed Mamba-only
lock fixture violated Full/Mamba lock ordering and was replaced with valid
Full-only and Full+Mamba receipts. Sanity assertions remain enabled. Initial
Python line-length and patch-context whitespace check failures were corrected
before publication. Failed-attempt logs and successful reruns remain in the
repair run's scratch directory; none is represented as a final passing test.

## Public CLI reuse under pressure

Addresses `PRRT_kwDOT0t9uc67wy1x`. The
[integrated comparison](radix-reuse-pressure-comparison.json) runs the public
`predict --stack engine` consumer on legacy main
`8288046ac88fccaee124714cd60e62b2ad4e184d` and native corrective source
`3c800f55ba78ccb1e90d999df5174935807d14a2`. Both optimized wheels use Rust
1.91.1, opt-level 3, codegen-units 1, thin LTO, and the explicit features
`extension-module,python`. Each wheel runs twice in separate processes, pinned
to CPU 0, after all compilation has finished. The report verifies each loaded
extension against the corresponding wheel and records both hashes and sizes.

The six locally authored Mooncake rows repeat three distinct prompts, with
block hashes `[10,20,30]`, `[40,50]`, `[60,70]`, then the same sequence again.
One SGLang worker uses LRU, concurrency one, one sequence, 32-token pages, and
six cache blocks. Every run completes six requests, 448 input tokens, and six
output tokens. No trace sampling or model weights are involved. The report
contains the exact rows, trace checksum, configuration, tiny-model metadata,
and all emitted summary fields.

| Summary field | Legacy main | Native |
| --- | --- | --- |
| `prefix_cache_reused_ratio` | 0.2857142857142857 | 0.0 |
| `first_admission_prefix_cache_reused_ratio` | 0.2857142857142857 | 0.0 |
| `committed_prefill_tokens` | 320 | 448 |
| Mean and p99 TTFT, ms | 1 | 1 |
| Modeled duration, ms | 6 | 6 |
| Modeled request throughput, requests/s | 1000 | 1000 |

Legacy suffix reclamation retains 128 tokens that later requests reuse. Native
whole-leaf reclamation removes the compressed prompts, so their repeats miss.
Those 128 tokens account for the complete difference in committed prefill work.
This fixture exercises the whole-leaf policy; the separate partial-match
fixture above establishes split-recency behavior. Fixed 1 ms prefill/decode
timing does not depend on prefill token count, so this comparison establishes
no TTFT or serving-throughput change. Every other summary field matches.

Each revision's two summaries are byte-stable after excluding only
`wall_time_ms`, `processed_tokens_per_s`, and `processed_output_tokens_per_s`;
their digests are recorded. Those three host-cost fields vary and are retained
in the report. Two tiny CLI runs are not a simulator-speed qualification. The
native `wall_time_ms` interval includes replay setup through collector
finalization, excluding model construction, Python startup, trace loading,
and output serialization.

### Downstream release note: Dynamo and Mocker

SGLang replay now uses the pinned native tree's whole-leaf LRU reclamation and
split-recency policy. Dynamo and Mocker consumers can observe different prefix
residency, reuse ratios, and prefill work at cache-capacity boundaries. With
token-dependent timing, those changes can also affect simulated TTFT and
throughput. Requalify affected replay expectations against the native policy;
do not assume legacy suffix eviction or unchanged replay output. The synthetic
case above measures a reuse change, with unchanged fixed-timing TTFT and
throughput. It is not a production accuracy claim or a Dynamo runtime test.

### Reproduce the integrated case

Use isolated snapshots and release wheels for the two immutable revisions.
The wheel build/install/hash procedure in
[growth qualification](RADIX_EXCLUSIVE_GROWTH_EVIDENCE.md#verification-and-reproduction)
applies, substituting the two revisions in this section. Match Python and the
locked runtime dependencies between arms. Do not build while executing either
arm. Set `pressure_work` to a fresh scratch directory outside the checkout and
materialize the inputs directly from the committed report:

```bash
pressure_work=$(mktemp -d "${TMPDIR:?}/radix-pressure.XXXXXXXX")
python3 - "$pressure_work" <<'PYCODE'
import hashlib, json, pathlib, sys
work = pathlib.Path(sys.argv[1])
report = json.loads(pathlib.Path(
    "crates/core/src/engine/cache/radix-reuse-pressure-comparison.json"
).read_text())
trace = "".join(json.dumps(row) + "\n" for row in report["trace_rows"])
assert hashlib.sha256(trace.encode()).hexdigest() == report["trace_sha256"]
(work / "trace.jsonl").write_text(trace)
(work / "tiny-model").mkdir()
(work / "tiny-model/config.json").write_text(json.dumps(report["model_provenance"]["config"]))
config = report["config"]
config["engine"]["model"] = str(work / "tiny-model")
config["traffic"]["source"]["paths"] = [str(work / "trace.jsonl")]
(work / "config.json").write_text(json.dumps(config))
PYCODE
# Set pressure_python to each isolated venv's Python, with its verified wheel.
# Execute twice per arm, using a fresh output directory for each invocation.
taskset -c 0 "$pressure_python" -m aisimulate predict --stack engine \
  --config "$pressure_work/config.json" \
  --output-dir "$pressure_work/$side-$repeat" --format json
```

Read each `prediction.json` as the summary (or its `summary` member if wrapped); verify completion and token totals,
then compare all summary keys with exactly the three host-metric exclusions
above. A setup attempt incorrectly supplied `traffic.stop.requests` for trace
traffic and was rejected by normal validation. Removing that unsupported stop
condition makes both arms consume the complete six-row trace. The recorded
comparison contains only that successful rerun; the rejected setup log remains
in the repair scratch directory.
