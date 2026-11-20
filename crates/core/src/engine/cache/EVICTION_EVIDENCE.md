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
