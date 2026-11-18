<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PR #908 admission evidence

This repair addresses the Cross-Layer Contract and Review Evidence findings in
CodeRabbit review 5961759651 on `7c4b10d90640146c085d373843eb8608f6a72adb`.
It adds regression evidence and clarifies documentation; production scheduler
and numerical behavior remain at that revision.

## SimpleCPU queue invariant

The supported `simple_cpu_v0_32` configurations cannot construct the suggested
zero-G1 connector waiter through `StartLoad::Retry`:

- `G2Connector::rules` disables store and peer-load waiting and enables
  `pins_hits`. SimpleCPU supports neither G3 nor KVCR nor sliding-window states.
- `VllmHostOffloadAdapter::lookup` pins every resident key of a hit, including
  recurrent-state keys, before returning it. Pending stores are misses rather
  than deferred hits. `load_by_key` is populated only for connectors that wait
  for peer loads.
- `HostTier::pin_external` removes pinned entries from the free queue. A store
  cannot evict the hit between lookup and `start_load`. The hit is a nonempty,
  unique set of resident keys fitting in G2, so `schedule_load` cannot return
  `LoadOutcome::Miss`. The load takes over its pins before lookup pins drop.
- `start_load` reserves G1 before queuing. Capacity exhaustion leaves the
  request in its ordinary waiting position; it does not enqueue a connector
  waiter. Loading requests retain their reservation through activation.

`simple_cpu_pins_a_hit_until_the_request_loads_it_or_looks_up_again` now uses a
one-block G2 and attempts a competing store after lookup. The store must wait,
the hit remains resident, and the load queues with one reserved G1 block.
`simple_cpu_loads_a_block_into_two_requests_at_once` covers peer loads.
`a_landed_connector_load_cannot_block_a_materialized_handoff` also checks both
SimpleCPU waiters' reservations and that the old queue selection picks the same
owner before adding the handoffs.

## Before/after selection checks

The baseline is the exact `g1_owner` selection block from PR merge-base
`bf1b539fbac7825219384b1b7b38b5d19ebf324a`; the candidate uses the published
selection from `7c4b10d90640146c085d373843eb8608f6a72adb`. This focused ablation
holds the rest of the engine and every fixture constant. It is not a claim
that all of current main's predictions were compared. Temporary edits were
restored before final workspace tests.

| Test | Previous selection | Published selection |
| --- | --- | --- |
| `simple_cpu_duplicate_loads_keep_connector_waiter_order` | Pass | Pass |
| `a_decode_rank_with_g3_admits_an_activated_handoff_before_a_landed_load` | Admits load owner UUID 3; assertion fails | Admits activated handoff UUID 5 |
| `hybrid_v0_32_connector_waiters_complete_under_g1_contention` | Effect-free zero-duration invariant error, two requests unfinished | All seven requests complete; A, B and C reuse host tokens; C finishes before B |

The SimpleCPU replay has six G1 blocks, four tokens/block, 250,000 bytes/token,
32 G2 blocks, H2D bandwidth 0.1 GB/s and fixed 1 ms forward timings. Its trace is
`waiter_contention()` in `host_offload_scopes.rs`. Both selections produce these
same asserted outputs (ms, host reuse in tokens):

| Request | First admission before/after | Terminal before/after | Host reuse before/after |
| --- | --- | --- | --- |
| A | 120 / 120 | 121 / 121 | 8 / 8 |
| B | 130 / 130 | 131 / 131 | 4 / 4 |
| C | 151 / 151 | 152 / 152 | 8 / 8 |

The new G3 test uses the supported `offloading_v0_24` decode configuration,
an attached G3 tier (`retry_epoch().is_some()`), a landed G1-holding load and an
activated destination, with **no pending destination**. It isolates decode-role
precedence. One admission slot makes the selected order observable. The changed
order is intentional: received KV must run ahead of a fresh restore. The
existing aggregated handoff test still covers both v0.32 profiles with a pending
destination. The docs now explicitly describe both cases and the aggregated
case where a landed load can precede a handoff.

`v0_24_connector_waiters_keep_their_existing_admission_times` separately pins
the unchanged aggregated v0.24 behavior without G3 or KVCR.
`v0_32_admits_a_waiter_that_holds_g1_before_one_that_holds_none` covers plain G2.
The hybrid regression seeds both recurrent-state boundaries and fills ten G1
blocks before restoring, accounting for working and cached state copies.

These are synthetic fixed-timing engine predictions, not measured vLLM runs
or evidence of serving accuracy. The repair adds no production work or retained
state; its new replay fixture contains seven requests. Host performance was not
benchmarked.

## Timing identity

`HostOffloadConfig.connector` is independent of timing version. The compiler's
`_worker_engine_args` serializes the host-offload descriptor to `native_host_offload`
and the engine version to `aic_backend_version`; `G2Connector::rules` selects
cache behavior without consulting timing identity. Therefore the PR's 0.24.0
operation timings with `offloading_v0_32` intentionally model v0.32 cache rules
using those available measured operation timings. They do not represent a
measured v0.32 deployment.

## Local verification and revision identity

The retained local results below cover the source and regression tests published
in **`4e70f67a81c857d76af8f7d0b25e5f58a64f0d4b`**. The follow-up to CodeRabbit
review **5962318003** changes this evidence report only. The PR description
records the exact final documentation commit and its verification reruns;
results from an earlier revision are not hosted checks for that commit.

Local environment: Linux x86_64, Rust 1.91.1, Python 3.12.3, Cargo jobs=2.
Python dependencies came from the locked project dev environment plus the
`scripts/pyproject.toml` CI, FPM-accuracy and E2E-accuracy dependency groups.
The native wheel was built first with `maturin build --release --auditwheel skip`
and features `extension-module,jemalloc`, matching CI's optimized build profile.
The loaded extension was verified in the scratch virtual environment:
SHA-256 `4cef78b0ad90ab7fdbf207816cb8f2751b5fd9f01d25f27037a021a0ed052048`.

- `CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 cargo test --workspace`:
  **3571 passed, 3 ignored** at
  `4e70f67a81c857d76af8f7d0b25e5f58a64f0d4b`: **3263 library tests**, **305
  integration tests** and **3 doctests**. The library has two ignored tests and
  the integration binaries have one. The workspace contains only `aisim-core`,
  so `cargo test -p aisim-core` covers the same package. This includes the new
  G3 and hybrid tests and the strengthened pin test. The PR description's older
  3262 figure counted only library tests at
  `a511fbaaa4264ecf078e75e22952a4d4290829e7`, before the additional G3 test.
- `python -m pytest tests -k 'g2 or host_offload or state_cache or kvcr or g3 or connector or replay'`:
  **949 passed**, 6477 deselected, covering
  `4e70f67a81c857d76af8f7d0b25e5f58a64f0d4b`. Cache and test temporary
  directories were redirected to the repair scratch directory; the runtime
  came from the release wheel, not an editable/debug build. The previously
  reported 882 result predates the main merge and is historical only.
- The earlier Python collection attempt lacked repository-tooling dependencies;
  installing the same three dependency groups as CI cleared all 26 collection
  errors. Initial fixture development also exposed a wrong G3 constructor name
  and insufficient hybrid state boundaries/capacity; these were corrected before
  the passing final suite. Negative ablation failures above were expected.

## Hosted verification

As inspected on 2026-11-18, [Fast CI run 44364037173](https://github.com/ai-dynamo/aisim/actions/runs/44364037173)
passed on **`4e70f67a81c857d76af8f7d0b25e5f58a64f0d4b`**. Run 44357257066
covered **`7c4b10d90640146c085d373843eb8608f6a72adb`** and is historical only.
No Full CI run exists for `4e70f67a81c857d76af8f7d0b25e5f58a64f0d4b` as
inspected during repair. There can be no hosted run for the unpublished
evidence correction yet. Fresh reviews, technical reapproval and Full CI on
the final commit remain pending controller admission; local results and Fast
CI do not substitute for those requirements.
