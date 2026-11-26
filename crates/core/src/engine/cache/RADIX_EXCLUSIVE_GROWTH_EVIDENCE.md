<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Exclusive radix growth qualification

Addresses `PRRT_kwDOT0t9uc657kxU`. Baseline is pinned PR head
`57a902aad462ab13b7994956ecdeda905220dc8b`; source correction is
`d4e983e1bae6705dd14f6d64e16d4bf6224e71a8`. [Raw samples and artifact hashes](radix-exclusive-growth-comparison.json)
include every warmup and measured invocation.

## Transition and cost

The lease already binds its immutable cached hash prefix to a locked anchor.
Exclusive leaf growth now passes only new hashes/pages to `append_locked_leaf`.
The native leaf constructor preserves the existing topology and recency effects;
only the new child acquires a lock. Ancestor lock counts remain unchanged,
matching acquisition of the extended path followed by release of the old path.
Root, shared, and overlapping anchors retain validated canonical insertion and
physical-page reconciliation. No generated SGLang source or upstream patch changed.

For an exclusive lease growing by N pages, growth touches O(N) new key/value
state in total, followed by an O(path) finish/unlock. The previous retained-key
validation and two full-path lock walks accumulated O(N squared) work. The
resident node layout and optional Belady metadata are unchanged; no new persistent
cache or request state was added. Overlap reconciliation retains its path cost.

The new regressions bound borrowed key exposure at each one-page append across
32, 256, and 1,024 growth pages, check every retained lock, and invoke native tree
invariant checks. LRU and Belady both cover shared ancestors, canonical overlap,
retraction/readmission, exact logical Stored/Removed hashes, reuse, and full cleanup.

## Matched release measurements

Both arms use Rust 1.91.1, the identical Cargo lockfile, opt-level 3,
codegen-units 1, thin LTO, and CPU 0 on the same WSL2 container (two-CPU quota,
5 GiB memory cap). Compilation completed before measurement. Each case has one
warmup per arm and five alternating pairs; all samples are retained. These
are synthetic host-cost measurements with no measured serving-accuracy claim.

The identical `radix_growth_measurement` probe was transplanted into the baseline.
Both optimized test binaries enable only Cargo's empty `default` feature. The
candidate's test-only key counter adds measurement overhead; the production
wheel comparisons below omit it. The probe checks reuse, overlap canonical pages,
and cleanup on every lifecycle. It varies growth through 2/128/512/2,048 pages
at both 64- and 4,096-token starting prompts, with and without an interleaved
insert. The JSON contains all sixteen cases, ranges through their raw samples,
and separate whole-lifecycle timings.

Median growth-only wall milliseconds for 20 lifecycles, 64-token prompt,
32-token pages, no overlap (allocation, hashing, and cache extension included;
initial admission and finish/eviction excluded):

| Added pages | Baseline host ms | Candidate host ms |
| --- | --- | --- |
| 2 | 0.028 | 0.017 |
| 128 | 2.656 | 0.808 |
| 512 | 22.273 | 2.916 |
| 2048 | 275.493 | 11.742 |

The production consumer is `python -m aisimulate predict --stack engine --config
CONFIG --output-dir OUTPUT --format json`, with separate verified optimized wheels
(`extension-module,python`). Each case completes eight requests, 64 input tokens
per request, one SGLang worker, concurrency one, page size one, and 32,768 cache
blocks. The generated model uses the repository tiny-model metadata with context
131,072; no weights were downloaded. The 1.2 GB execution budget and 1 GB host
reserve remain unchanged.

Median native `wall_time_ms` follows. This timer covers `Replayer::run_inner`
through collector finalization and excludes model/CLI setup; full CLI timings
are also retained in the JSON. Neither timer is a replay-loop-only timer.

| Output tokens/request | Baseline host ms | Candidate host ms | Modeled duration ms, both arms |
| --- | --- | --- | --- |
| 64 | 1.252 | 1.019 | 512 |
| 512 | 14.568 | 5.494 | 4096 |
| 2048 | 134.088 | 21.285 | 16384 |

Every warmup and measured run completes the exact configured request/input/output
totals. All emitted metrics match after excluding only host `wall_time_ms` and
its two derived processing-rate metrics. Modeled serving timing remains fixed at
1 ms prefill/decode. The JSON records canonical metric digests, native binary and
`.text` sizes, exact configurations, and source/probe checksums. This qualifies
the scoped exclusive-growth optimization on these workloads.

## Verification and reproduction

- `cargo test --release --locked --workspace --jobs 1`: 1,965 passed, 3 existing
  manual benchmarks ignored.
- `cargo test --release --locked --lib exclusive_growth --jobs 1`: 4 passed.
- Optimized-wheel Python engine-step/compile-engine contracts: 102 passed.
- Radix source/comparison tool tests: 10 passed.
- Rustfmt, repository Python Ruff lint/format, and packaged legal-file checks pass.

The following recipe reproduces this table's probe, rather than the older
10-repetition probe documented in `RADIX_REPAIR_EVIDENCE.md`. It requires local
Git objects for both immutable revisions and keeps every artifact outside the
checkout. Build both arms before collecting timings; repeat the last invocation
with the paired schedule above for comparable measurements.

```bash
set -euo pipefail
growth_work=$(mktemp -d "${TMPDIR:?}/exclusive-growth.XXXXXXXX")
growth_targets="${CARGO_TARGET_DIR:-$growth_work/target}/$(basename "$growth_work")"
probe_revision=d4e983e1bae6705dd14f6d64e16d4bf6224e71a8
probe_path=crates/core/src/engine/kv_manager/radix_lifecycle_evidence.rs
git show "$probe_revision:$probe_path" > "$growth_work/probe.rs"
printf '%s  %s\n' \
  852732a3b202490edb6372ba5331a280b73dc129a16315889d42d266bc97b787 \
  "$growth_work/probe.rs" | sha256sum --check
for side in baseline candidate; do
  revision=57a902aad462ab13b7994956ecdeda905220dc8b
  if [ "$side" = candidate ]; then revision=$probe_revision; fi
  snapshot="$growth_work/$side"
  mkdir "$snapshot"
  git archive "$revision" Cargo.toml Cargo.lock crates/core python/aisimulate | tar -x -C "$snapshot"
  cp "$growth_work/probe.rs" "$snapshot/$probe_path"
  module="$snapshot/crates/core/src/engine/kv_manager/sglang_backend.rs"
  if ! rg -q '^mod radix_lifecycle_evidence;$' "$module"; then
    printf '\n#[cfg(test)]\n#[path = "radix_lifecycle_evidence.rs"]\nmod radix_lifecycle_evidence;\n' >> "$module"
  fi
  (cd "$snapshot"
    CARGO_TARGET_DIR="$growth_targets/$side" cargo test --release --locked --lib \
      radix_growth_measurement --no-run > "$growth_work/$side-build.log" 2>&1)
done
for side in baseline candidate; do
  (cd "$growth_work/$side"
    CARGO_TARGET_DIR="$growth_targets/$side" \
      AISIM_EVICTION_EVIDENCE_OUTPUT="$growth_work/$side.json" \
      cargo test --release --locked --lib radix_growth_measurement -- --ignored --exact \
      engine::kv_manager::sglang_backend::radix_lifecycle_evidence::radix_growth_measurement)
done
```

For the CLI rows, build and install each snapshot's optimized production wheel
before any timing. Use the same pinned Rust/Python versions and lockfiles when
comparing binary hashes; otherwise record newly built hashes rather than claiming
they equal the historical artifacts. `maturin` must be available on `PATH`.

```bash
for side in baseline candidate; do
  uv venv "$growth_work/$side-venv"
  (cd "$growth_work/$side/python/aisimulate"
    CARGO_TARGET_DIR="$growth_targets/$side-wheel" \
      maturin build --release --locked --out "$growth_work/$side-dist")
  uv pip install --python "$growth_work/$side-venv/bin/python" "$growth_work/$side-dist/"*.whl
  "$growth_work/$side-venv/bin/python" - "$growth_work/$side-dist/"*.whl <<'PYCODE'
import hashlib, importlib, sys, zipfile
runtime = importlib.import_module("aisimulate._runtime")
loaded = hashlib.sha256(open(runtime.__file__, "rb").read()).hexdigest()
with zipfile.ZipFile(sys.argv[1]) as wheel:
    entry = next(p for p in wheel.namelist() if p.endswith("/_runtime.abi3.so"))
    assert loaded == hashlib.sha256(wheel.read(entry)).hexdigest()
print(runtime.__file__, loaded)
PYCODE
done
```

Copy `tests/e2e/configs/unified_cli/fixtures/tiny-model/config.json` to a scratch
`tiny-model/config.json` and set `max_position_embeddings` to 131072, as recorded
in `model_provenance.modification`. Replace only the absolute model path in each
`cli_rows[].config` before invoking its recorded CLI command. Preserve the traffic, cache, timing and resource settings, and the
`cli_canonical_exclusions` list. No weights or numerical goldens are required.

Earlier setup/resource failures remain in the scratch logs: the initial wheel
command selected CFFI instead of the Python project; pytest needed its asyncio
plugin and explicit fixture scope; a comparison preflight needed the package's
hyphenated fingerprint directory. Successful reruns corrected those setups.
Concurrent release compilation exceeded the 5 GiB cap; serial compilation passed
with unchanged optimization, assertions, and timeouts. A CLI inspection during
compilation was rejected by the host-headroom guard; every final CLI run passed
after compilation with the same budget. No measured sample was discarded.

Hosted workflow-log access returned HTTP 403. The live PR still reports main
conflicts, which this comment-repair scope excludes merging. Fresh CI, bot review,
and technical reapproval remain required after publication.
