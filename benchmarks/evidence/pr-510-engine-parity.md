<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PR #510 local engine parity

Verified on 2026-11-24 at `99c2535a5acb48f103736bfdccb2181fdba72485`.
The runtime sources and frozen goldens are identical to pinned PR head
`5b1b8c9b659d3ec8a3454078b30af0c1b1761546`; the correction changes only
the forward-benchmark workflow and its regression tests. The follow-up
evidence commit adds only this file.

Both requested native parity suites passed. No goldens, assertions, or
timeouts changed. This is local regression evidence against frozen engine
answers, with no claim of measured serving accuracy or hosted CI results.

The environment used Linux x86_64, CPython 3.12.12, Rust 1.91.1,
maturin 1.16.0, pytest 9.0.3, and dependencies from the frozen project lock.
`REPAIR_SCRATCH` below denotes this repair's scratch directory; `TMPDIR`,
`CARGO_TARGET_DIR`, and `UV_CACHE_DIR` remained outside the checkout, with
`CARGO_BUILD_JOBS=2`. Python and pytest below use its `venv/bin` directory.

Build and installation commands, run before either parity suite:

```bash
# From python/aisimulate:
python -m maturin build --release --locked --auditwheel skip --out "$REPAIR_SCRATCH/wheel"
# From the repository root:
uv pip install --python "$REPAIR_SCRATCH/venv/bin/python" --no-deps "$REPAIR_SCRATCH/wheel"/aisimulate-*.whl
```

The build completed with Cargo's optimized release profile (`opt-level=3`,
thin LTO, one codegen unit). The tests imported the installed wheel's
`aisimulate/_runtime.abi3.so`, SHA-256
`046bf333493615f5a454ff4435c1c6ee2e70b68f8306365db20aa84949e0c607`.

Exact parity commands from the repository root:

```bash
python -m pytest -q -rx -n 2 -c python/aisimulate/pytest.ini crates/core/parity_tests/perfmodel/test_engine_step_parity.py -p no:cacheprovider --basetemp "$TMPDIR/pytest-engine-parity"
python -m pytest -q -rx -n 2 -c python/aisimulate/pytest.ini crates/core/parity_tests/perfmodel/test_compile_engine_parity.py -p no:cacheprovider --basetemp "$TMPDIR/pytest-compile-parity"
```

Results: **305 passed in 68.83s** and **85 passed in 12.45s**, respectively;
neither suite skipped cases. The configured 120-second test timeout remained
enabled.

The Cargo isolation correction also passed all **423** workflow/release
contract cases with:

```bash
python -m pytest -c /dev/null tests/test_ci_workflow_contracts.py tests/test_ci_qualification.py tests/test_release_fpe.py -q -p no:cacheprovider -o asyncio_default_fixture_loop_scope=function --override-ini="addopts=" --override-ini="filterwarnings=error" --basetemp "$TMPDIR/pytest-ci-contracts-final"
```

The new tests execute the workflow's setup before two real offline Cargo
builds. Poisoned global configurations fail the control builds; the isolated
base and head builds pass, retaining registry/git download caches with either
an explicit image `CARGO_HOME` or the default home. The image recipe inherits
its environment from `BASE_IMAGE`, so setup inspects the effective environment
before checkout. It adds a constant amount of job setup work (at most two
cache links) and changes no prediction or replay execution path.

Earlier setup attempts are preserved in the repair scratch logs:
`wheel-build.log` records a repository-root manifest invocation that selected
CFFI packaging and failed for missing `cffi`; `wheel-build-python.log` records
the successful mixed-project release build above. `ci-contracts.log` records
418 passes and five missing-prerequisite failures (`pip-licenses` and Node).
After installing the scripts' locked CI requirements and Node,
`ci-contracts-final.log` records the 423-pass rerun. The two parity logs are
`engine-step-parity.log` and `compile-engine-parity.log`.

Thread `PRRT_kwDOT0t9uc66hmrt` also requests these commands and results in the
PR description. That publication step remains open: this repair has read-only
GitHub access. A maintainer with PR-edit permission must copy the parity
evidence into the description before resolving that thread.

## Sequence-limit and attention-documentation follow-up

Reverified on 2026-11-24 with the accompanying source, documentation, and
regression-test changes on top of pinned head
`07587182c7ea6027d78232f5718439cd57c28963`. The same Python, Rust, maturin,
pytest, frozen dependencies, scratch-directory isolation, and two Cargo build
jobs described above were used. No goldens, assertions, or timeouts changed.

The installed mixed-project wheel was built from `python/aisimulate` with:

```bash
python -m maturin build --profile ci --locked --config 'profile.ci.inherits="release"' --config 'profile.ci.opt-level=2' --config 'profile.ci.lto=false' --config 'profile.ci.codegen-units=16' --auditwheel skip --out "$REPAIR_SCRATCH/wheel"
uv pip install --python "$REPAIR_SCRATCH/venv/bin/python" --no-deps "$REPAIR_SCRATCH/wheel"/aisimulate-*.whl
```

Cargo reported the optimized `ci` profile. The imported wheel's
`aisimulate/_runtime.abi3.so` SHA-256 was
`0a146d5e1cb20d98f28172babb7041c111ef5bd76d842c11cd583fe6abeb4718`.
The two parity commands above passed again: **305 passed in 72.09s** and
**85 passed in 13.29s**, with no skips. Follow-up logs are
`pr510-engine-parity.log` and `pr510-compile-parity.log` under `TMPDIR`.

The targeted caller regressions also passed:

```bash
python -m pytest -q tests/sweeper/test_afd_fastafd.py tests/sweeper/test_afd_perfmodel.py tests/sweeper/test_afd_engine.py -p no:cacheprovider --basetemp "$TMPDIR/pytest-afd"
```

Result: **267 passed in 12.70s** (`pr510-afd.log`). New cases exercise the
canonical model and FastAFD adapter, reject undersized sequence limits before
timing or memory estimation, and preserve default, exact-fit, and larger KV
allocations. Native SOL timing and real memory accounting verify the documented
uncomposed attention return and the adapter's router addition. These are
synthetic-profile regression checks, not measured serving-accuracy evidence.
The guard adds constant work before model construction, with no retained state
or change to valid-input numerical computation.

Repository-wide Ruff lint and format checks, seven documentation-link unit
tests, the 79-file documentation destination check, and `git diff --check`
passed. The initial formatting check requested formatting of the new test;
after formatting it, the full check passed (`pr510-initial-format.log` and
`pr510-ruff-format.log`). An initial repository-root wheel command selected
crate-only packaging and warned about its module name; it was never installed.
The application-directory build reused the optimized native artifacts and
produced the installed mixed-project wheel (`pr510-wheel-build.log` and
`pr510-wheel-mixed.log`).

The description-publication requirement for `PRRT_kwDOT0t9uc66hmrt` remains
open. This worker has read-only GitHub access and cannot add these commands and
results to the PR description. Threads `PRRT_kwDOT0t9uc66tXLC` and
`PRRT_kwDOT0t9uc66tXLI` are covered by the fixes and targeted tests above.

## Scoped capture and replay boundary repair

Verified on 2026-11-24 against source/test correction
`f8361f49f92b2c2e228e91be057642897ff80cf1`, descended from pinned head
`5ea4120ebd294125506178b5243bf0d149c0ae82`. The accompanying evidence-only
commit changes no executable code. Seven selected threads are covered:
`PRRT_kwDOT0t9uc66t3g2`, `PRRT_kwDOT0t9uc66t3g6`,
`PRRT_kwDOT0t9uc66t3g_`, `PRRT_kwDOT0t9uc66t3hD`,
`PRRT_kwDOT0t9uc66t3hF`, `PRRT_kwDOT0t9uc66t3hJ`, and
`PRRT_kwDOT0t9uc66t3hM`.

The environment was Linux x86_64 under WSL2, Python 3.12.12, Rust 1.91.1,
maturin 1.16.0, pytest 9.1.1, PyArrow 24.0.0, and NumPy 2.5.2. Dependencies
came from `uv sync --project python/aisimulate --extra dev --frozen
--no-install-project`, with `UV_PROJECT_ENVIRONMENT` pointing into this repair's
scratch directory. The optimized mixed-project wheel was built from
`python/aisimulate` with the `--profile ci` command in the preceding section,
before any native verification. `CARGO_BUILD_JOBS=2` and all build/cache/temp
directories remained in scratch. The imported `_runtime.abi3.so` SHA-256 was
`57aa719e6013b924a7d5b31824302fed863f6dda2abfbd041d19efb0e747d43e`.
All five edited Python modules were checked byte-for-byte against the installed
wheel. Rust source, goldens, assertions, and timeouts were unchanged.

With the scratch venv on PATH, the exact parity invocations were:

```bash
python -m pytest -q -rx -n 2 -c python/aisimulate/pytest.ini crates/core/parity_tests/perfmodel/test_engine_step_parity.py -p no:cacheprovider --basetemp "$TMPDIR/pytest-engine-parity"
python -m pytest -q -rx -n 2 -c python/aisimulate/pytest.ini crates/core/parity_tests/perfmodel/test_compile_engine_parity.py -p no:cacheprovider --basetemp "$TMPDIR/pytest-compile-parity"
```

Results: **305 passed in 77.93s** and **85 passed in 11.94s**, without skips.
Logs: `engine-parity.log` and `compile-parity.log` under `TMPDIR`.

Final caller and import verification:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q python/aisimulate/tests/unit/test_fastafd_profile.py python/aisimulate/tests/unit/test_fastafd_profile_build.py tests/sweeper/test_afd_fastafd.py tests/sweeper/test_afd_perfmodel.py tests/sweeper/test_afd_engine.py tests/test_afd_runner.py -p no:cacheprovider --timeout=120 --basetemp "$TMPDIR/pytest-targeted-final"
```

Result: **652 passed in 15.01s** (`targeted.log`). This includes both inspection
and native loading for binary/UTF-8 Parquet metadata, byte preservation of raw
captures, repeated symlink rejection, invalid routing, positive sequence lengths,
matching replay after source-profile deletion, changed replay geometry, and
conflicting typed/provenance traces. Existing raw-only and matching typed traces
remain valid. Restoring the pinned Python modules with the same native binary
made the new negative regressions fail: **39 failed, 4 passed** (the passes are
UTF-8 controls), recorded in `baseline-regressions.log`.

Earlier attempts are preserved: `baseline-collection.log` used the application's
pytest config for root sweeper tests and failed collection on its missing
`gpu_0` marker; the rerun used the root config. `targeted-initial.log` contains
645 passes and seven new test-setup failures: the fixtures serialized a provenance
dictionary instead of the deployment topology schema. Correcting that setup,
without weakening assertions, produced the passing final run. The timing-span
publication regression now starts with a valid generated profile, preserving its
original rejection and byte-preservation assertions under the new output guard.

The replay guard compares a fixed five-field stage key and supported geometry;
it neither reopens a profile nor scans decode samples per pass. Typed-trace
validation adds comparisons within the existing preparation walk, with no new
trace copy or retained state. A matched CLI comparison additionally measured
the complete supervised `aisimulate predict --stack engine` call, including
startup, preparation, replay, and report writing. It used identical installed
dependencies/native binary and import hooks selecting the five Python modules
from the exact baseline/candidate commits above. No build, test, or profiler ran
concurrently with these measurements.

Both fixtures used the test's synthetic Qwen3-235B stage, gb200/sglang, 94 layers,
two microbatches, 64 A GPUs, eight F GPUs, a fixed prefill companion, ISL 1024,
and explicit KV capacity 1152. The control completed one request with OSL 2;
the longer case completed 32 with OSL 128. The initial control's default KV
capacity was infeasible (`perf/preflight.log`); explicit sufficient capacity
passed (`perf/preflight-fit.log`). This is synthetic-stage regression and host
cost evidence, not measured serving accuracy.

One alternating warmup pair preceded five measured alternating pairs per fixture:

| Fixture | Baseline median seconds | Candidate median seconds | Five candidate/baseline ratios | Completed AFD passes / intervals |
| --- | --- | --- | --- | --- |
| Control | 3.4044 | 3.4052 | 0.9991, 1.0016, 1.0128, 0.9959, 1.0013 | 1 / 752 |
| Longer replay | 5.6906 | 5.6912 | 1.0092, 1.0180, 0.9880, 0.9902, 1.0001 | 4064 / 3056128 |

Every normalized `prediction.json` (all 24 warmup/measured invocations) matched
within each fixture, including lifecycle counts and prediction metrics. Canonical
JSON SHA-256 values were
`0feaeaeda0a1ecc2d1f34f595fcc490422f7c39230ae204e909c6d84c9275b3d`
(control) and
`22f63d65b5560942178aac1ddddb15fa99762f287e1ae23eb8ec6d098835ae16`
(longer replay). The small comparison detected no material integrated overhead;
it is not a statistical performance qualification. Commands, frozen YAMLs,
checksums, paired samples, and logs are retained under `TMPDIR/perf/`, with
the comparison command `python "$TMPDIR/perf/compare.py"` and aggregate log
`TMPDIR/perf-comparison.log`.

Repository-wide Ruff lint/format, Python syntax compilation, Rust formatting,
SPDX/packaged-legal checks, seven documentation-link tests, the 79-file
documentation-destination check, and changed-line whitespace checks passed.
The live PR still pointed at the pinned starting head and base `main` at final
inspection. The workflow-runs API returned HTTP 403, so hosted workflow logs
were unavailable; the readable current-head checks reported pre-existing DCO
failures on eight commits by another author. No history was rewritten or
attestation added for those commits.

`PRRT_kwDOT0t9uc66hmrt` remains open solely for the explicitly requested
PR-description publication. The live description still lacks the parity commands
and results. A maintainer with PR-edit permission must copy them there; this
read-only worker cannot complete that part of the request.
