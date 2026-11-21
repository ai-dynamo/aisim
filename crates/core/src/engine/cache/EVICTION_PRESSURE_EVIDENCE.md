<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Incremental eviction under cache pressure

This repair compares starting head `e911d3a4a07750db05dddc3b6911c283358c044c`
with production correction `2440f4a8bff822eececbfa649874f79f6c33a0e8`
(including `cfe9b1cd0ba62c3ecf2d467f91d0b55533f8e4c1`). The native dependency
remains SGLang `f0fda25fc95c357368b714f90a13a69c1b67f50a`. These are synthetic
cache and engine measurements, not serving-accuracy evidence.

## Requested fixes

- `PRRT_kwDOT0t9uc65JEH-`: Belady maintains an ordered candidate index and a
  reverse index from forecast hashes to candidates. `changes_since` refreshes
  changed priorities, including a buried victim after remote retirement.
- `PRRT_kwDOT0t9uc65JEIm`: LRU uses the same maintained index, without forecast
  bookkeeping. It removes a whole native leaf using the full-length selected
  suffix API, preserving compressed-leaf overshoot and physical page order.
- `PRRT_kwDOT0t9uc65JEIJ`: key splits and a local `RadixValue` adapter share the
  retained prefix, copy the removed suffix, and compact after the visible
  prefix falls below half its backing allocation. Draining N pages copies
  O(N) buffer entries rather than copying every retained prefix again.

The native tree remains authoritative for eligibility, topology, locks, access
ticks, and actual removal. Inserts, unlocks, and exposed parents introduce
candidates. Native access ticks only increase, so stale access priorities can
be refreshed at the minimum; ineligible entries are removed there. Forecast
priorities can improve while buried, so their reverse-index updates happen
before victim selection. Every Belady page removal reranks the newly exposed
tail, preserving the existing saved-recency correction.

Index state is O(resident nodes), with no growing lazy-heap tombstones. Ordinary
one-page pressure updates O(log N) ordered-index state, rather than visiting N
unchanged leaves. Forecast refresh work scales with changed hashes and their
indexed candidates. Admission checkpoints save the native tree, allocator,
leases, and recency as before, but do not clone this derived index. A restored
or independently used checkpoint reconstructs it once at its first eviction;
later evictions are incremental. Thus successful fallible admissions do not
pay for another resident-index copy.

Adopted values still own compact buffers. Retained split buffers occupy at most
twice their visible length; child-map keys own only their page. A live checkpoint
can retain its original buffers independently, as required for rollback. The
ordinary drain regression bounds total constructed key atoms by 5N (including
three one-atom edge keys per split/removal), copied value entries by 2N, and
the active retained value allocation by twice its visible pages. Both live-
checkpoint and unshared cases are covered at 33, 1,024, and 8,192 pages.

## Reproduction

The identical [optimized probe](eviction_pressure_probe.rs) runs in isolated
baseline and candidate targets with Rust 1.91.1, the frozen lockfile, release
optimization, one codegen unit, thin LTO, and the dependency's test-only
`inspection` feature. Archive each immutable revision into a fresh scratch
directory, overlay this probe, and append its `#[cfg(test)]` module declaration
to the baseline adapter. Build both before collecting samples:

```bash
cargo test --release --locked --lib eviction_pressure_measurement --no-run
AISIM_PRESSURE_OUTPUT="$TMPDIR/pressure-probe.json" \
  cargo test --release --locked --lib eviction_pressure_measurement -- --ignored
cargo test -p aisimulate-core --locked --lib engine::cache::radix_cache
```

Use distinct `CARGO_TARGET_DIR` subdirectories for the revisions and disable
incremental compilation. For paired timing, run each already-built test binary
in a fresh CPU-0 process, with one warmup per side followed by five alternating
pairs. The probe checks every victim, physical-capacity restoration, and
replacement insert. Setup and snapshot construction are outside its timers.
Do not run builds or other tests while measuring.

For the integrated comparison, build each revision's wheel with the CI profile
before installing it into separate scratch package directories:

```bash
python -m maturin build --release --locked --auditwheel skip --out "$TMPDIR/wheels"
python -m pip install --no-deps --target "$TMPDIR/runtime" "$TMPDIR/wheels/"*.whl
PYTHONPATH="$TMPDIR/runtime" python -c \
  'import aisimulate._runtime as n, hashlib; print(n.__file__); print(hashlib.sha256(open(n.__file__, "rb").read()).hexdigest())'
```

From the repository root, prepare the selected report configuration (use
`control`, `pressure_1024`, or `pressure_8192`) and the local model metadata:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

scratch = Path(os.environ["TMPDIR"])
report = json.loads(Path(
    "crates/core/src/engine/cache/eviction-pressure-cli-comparison.json"
).read_text())
config = report["configs"]["control"]["config"]
model = json.loads(Path(
    "tests/e2e/configs/unified_cli/fixtures/tiny-model/config.json"
).read_text())
model["max_position_embeddings"] = 131072
model_dir = scratch / "pressure-tiny-model"
model_dir.mkdir(exist_ok=True)
(model_dir / "config.json").write_text(json.dumps(model))
config["engine"]["model"] = str(model_dir)
(scratch / "pressure-config.json").write_text(json.dumps(config))
PY
PYTHONPATH="$TMPDIR/runtime" taskset -c 0 python -m aisimulate predict \
  --stack engine --config "$TMPDIR/pressure-config.json" \
  --output-dir "$TMPDIR/prediction" --format json
```

Run the build from `python/aisimulate`, and use a separate output directory for
each CLI invocation. Use the same frozen Python environment and lockfile for
both wheels. The comparison report records each loaded extension's hash, the
full configurations, and the tiny local model configuration. No model weights
are downloaded; serving times are fixed at 1 ms. The model
configuration comes from
`tests/e2e/configs/unified_cli/fixtures/tiny-model/config.json`, with only its
context limit enlarged to 131,072 for these synthetic cases.

The control runs 32 requests against 64 physical blocks. Pressure cases run
4,096 requests against 1,024 blocks and 16,384 against 8,192 blocks. All use
one SGLang worker, concurrency one, block size 32, distinct 32-token prompts,
and one output token. They therefore exercise completed request admissions,
checkpoint creation, cache insertion, and sustained LRU replacement. The
compressed Belady and forecast-retirement cases are measured separately by
the native probe. The 1.2 GB execution budget, one-CPU limit, and default
1 GB host reserve are unchanged on both sides.

For every CLI sample, require the configured request count, 32 input tokens
and one output token per request, successful resource-supervisor completion,
and complete worker termination. Compare the complete prediction JSON after
removing only `wall_time_ms`, `processed_tokens_per_s`, and
`processed_output_tokens_per_s` from its summary. Those three fields measure
host execution cost; all simulated serving results must remain identical.

The native `wall_time_ms` interval includes replay setup, execution, and
collector finalization. It excludes timing-model construction before replay
and final evidence/power aggregation, Python startup, and output serialization.
The full CLI timer covers those additional phases and process supervision.
Peak RSS is the supervisor's sampled process-tree measurement, not an exact
allocation count. Do not interpret these host costs as simulated serving
latency or predictive accuracy.

## Measurements

All timings below are milliseconds, reported as median [minimum, maximum]
of five measured samples per side. Each side also completed one excluded
warmup. Pairs alternate execution order in fresh CPU-0 processes. The host
was Linux/WSL2 x86-64, with a two-CPU cgroup quota and 5 GiB memory limit.
The CLI itself was limited to one CPU. Optimized wheels have no test counters;
the native probe retains candidate-only test instrumentation. Exact toolchain,
lockfile, probe, wheel, and loaded-binary hashes are in the reports.

### Repeated one-page pressure

| Policy | Resident leaves | Evictions + replacements | Baseline ms | Candidate ms |
| --- | ---: | ---: | ---: | ---: |
| lru | 64 | 63 | 0.085 [0.078, 0.093] | 0.064 [0.061, 0.075] |
| belady | 64 | 63 | 0.107 [0.103, 0.112] | 0.054 [0.053, 0.055] |
| belady_retirement | 64 | 63 | 0.119 [0.115, 0.120] | 0.072 [0.072, 0.085] |
| lru | 1,024 | 256 | 1.255 [1.215, 1.412] | 0.193 [0.184, 0.226] |
| belady | 1,024 | 256 | 4.100 [3.841, 4.230] | 0.224 [0.216, 0.228] |
| belady_retirement | 1,024 | 256 | 4.714 [4.680, 4.916] | 0.293 [0.286, 0.317] |
| lru | 8,192 | 256 | 10.922 [10.531, 12.539] | 0.262 [0.227, 0.268] |
| belady | 8,192 | 256 | 33.216 [32.028, 36.574] | 0.312 [0.252, 0.434] |
| belady_retirement | 8,192 | 256 | 40.619 [40.112, 47.294] | 0.499 [0.436, 0.518] |

The [native report](eviction-pressure-comparison.json) preserves every raw
sample, including warmups. Belady retirement changes a buried next victim
after an initial selection; the timed loop retires the next request, asserts
its physical victim, reuses the allocator page, and inserts a new leaf.
At 8,192 leaves the median improvements are 41.6x for LRU, 106.6x for
Belady, and 81.4x for retirement. The bounded candidate-query regression
independently establishes that selection no longer scans unchanged leaves.

### Compressed Belady drain

| Pages drained | Live checkpoint | Baseline ms | Candidate ms |
| --- | --- | ---: | ---: |
| 256 | no | 0.137 [0.137, 0.142] | 0.158 [0.151, 0.162] |
| 256 | yes | 0.130 [0.127, 0.134] | 0.149 [0.142, 0.159] |
| 2,048 | no | 1.287 [1.261, 1.388] | 1.176 [1.125, 1.226] |
| 2,048 | yes | 1.302 [1.278, 1.335] | 1.203 [1.135, 1.207] |
| 8,192 | no | 11.676 [11.639, 14.663] | 4.805 [4.693, 4.871] |
| 8,192 | yes | 11.794 [11.472, 13.658] | 4.784 [4.774, 5.043] |

The 256-page case adds about 0.02 ms of maintained-order bookkeeping over
the whole drain, with no many-leaf scan to eliminate. At 8,192 pages, the
candidate is 2.4x faster with or without a checkpoint. The copy-count tests
establish the linear bound independently of these timings. Per-page ordering
and forecast-index maintenance remain necessary to compare an exposed tail
against other leaves; there is no new size-dependent scan on that path.

### Complete native CLI

| Case | Baseline native ms | Candidate native ms | Baseline full CLI ms | Candidate full CLI ms |
| --- | ---: | ---: | ---: | ---: |
| control | 0.280 [0.270, 0.997] | 0.346 [0.285, 0.492] | 2560.474 [2531.550, 2805.042] | 2706.203 [2580.676, 2892.327] |
| pressure_1024 | 29.107 [26.594, 31.860] | 13.610 [12.011, 16.142] | 2632.336 [2502.270, 2783.992] | 2595.255 [2488.133, 2732.405] |
| pressure_8192 | 476.344 [445.988, 496.951] | 57.302 [54.609, 65.580] | 3202.582 [3174.390, 3237.632] | 2810.774 [2705.389, 2893.006] |

| Case | Baseline peak RSS MiB | Candidate peak RSS MiB |
| --- | ---: | ---: |
| control | 187.547 [187.082, 188.926] | 188.547 [188.223, 189.062] |
| pressure_1024 | 188.918 [188.641, 189.172] | 188.746 [188.410, 188.961] |
| pressure_8192 | 206.375 [205.922, 206.449] | 208.691 [208.293, 208.957] |

The [CLI report](eviction-pressure-cli-comparison.json) records all 36 runs,
full configurations, canonical prediction JSON and hashes, totals, completion,
and sampled RSS. Every canonical prediction matched across both revisions
and all repetitions. The 8,192-block case reduces median native time by 88.0%
and full CLI time by 12.2%. Native replay is 2.1x faster at 1,024 blocks;
process startup and supervision dominate its full CLI interval, whose ranges
overlap. No full CLI improvement is established for that case or the control.

The control adds 0.066 ms median native time over 32 requests (about 2 us per
request); its timing ranges overlap, including a baseline outlier. Its full
CLI medians differ by 5.7%, also with overlapping ranges. Maintaining the
index has a small cost before cache pressure, while avoiding the repeated
resident scans once the cache fills. At 8,192 blocks the added median sampled
RSS is 2.32 MiB (1.1%); indexes are bounded by resident nodes and do not grow
with the number of evictions. These are synthetic host-cost observations,
not a claim about production traces, other machines, or serving accuracy.

## Correctness and check results

The production correction passed `cargo test --workspace --locked`: 1,808 unit
tests, 103 integration tests, and three doc tests. Three explicitly manual or
optional tests were ignored, including this pressure measurement, which was
then run separately in all 12 paired probe processes. The focused command
`cargo test -p aisimulate-core --locked --lib engine::cache::radix_cache`
passed 34 cases, including 16 new parameterized regressions; its manual
measurement is ignored in the ordinary suite.

The exact optimized candidate wheel was installed and its loaded extension
hash checked before running:

```bash
python -m pytest -q -rx -n 2 -c python/aisimulate/pytest.ini \
  -p no:cacheprovider \
  crates/core/parity_tests/perfmodel/test_engine_step_parity.py \
  crates/core/parity_tests/perfmodel/test_compile_engine_parity.py
```

All 386 cases passed in 100.83 seconds with the existing assertions and
120-second test timeout. Numerical goldens are unchanged. The CODEOWNERS,
workflow-contract, qualification, and release-policy suites passed 434 tests.
Rust formatting, requested full Python Ruff checks/formatting, Python syntax
compilation, copyright, packaged legal files, generated CODEOWNERS integrity,
and documentation checks also passed. These local checks do not establish
hosted Full CI or technical reapproval: the scoped workflow-log API returned
HTTP 403, and those gates remain with the controller.

Earlier unsuccessful attempts are retained in the repair scratch logs:

- Cargo was initially absent from `PATH`; the successful commands use
  `/opt/rust/bin` and the same frozen dependencies.
- The newly authored copy-bound test initially assumed 4N key atoms.
  Inspection counted three one-atom native child-map keys per split/removal,
  in addition to the copied suffix and geometric compactions. The derived
  bound is 5N; all sizes passed after correcting that new assertion. Existing
  assertions were unchanged.
- The first CODEOWNERS invocation used system Python without PyYAML. It
  passed in the scratch environment installed from the frozen Python lockfile.
- A CLI control preflight during compilation failed the unchanged memory
  headroom guard before execution. After all builds finished, task-owned clean
  file cache was released with `posix_fadvise(POSIX_FADV_DONTNEED)`. The same
  configuration then passed all 12 control runs without lowering the memory
  budget or host reserve.
- An initial report-aggregation assertion compared a workload specification
  after adding its baseline timing fields. The exporter was corrected to
  compare specifications before adding results; all recorded samples passed
  that check. Measurements were preserved without a timing rerun.

Superseded candidate artifacts from before deferred checkpoint-index
reconstruction were rebuilt before final parity or measurement. Only the
final production correction's artifacts appear in these reports. Dependency
registry publication and the existing release-contract decision are separate
from these three performance findings.
