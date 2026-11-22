# Bundled SGLang native radix core

Source: <https://github.com/sgl-project/sglang/tree/f0fda25fc95c357368b714f90a13a69c1b67f50a/rust/sglang-radix-tree>

Revision: `f0fda25fc95c357368b714f90a13a69c1b67f50a`.
Copyright SGLang contributors. Licensed under Apache-2.0; the upstream
repository's license is retained in [LICENSE](LICENSE). The upstream repository
has no root NOTICE file. NVIDIA modifications are identified in each source
header and listed below. The canonical repository notices also identify this
copy in `THIRD_PARTY_NOTICES.md`.

The upstream native core is bundled as a private module of `aisimulate-core`.
There is no separately published or buildable vendor crate. Cargo cannot package
the original Git-only dependency, even when that dependency is optional. Keeping
the sources inside the existing crate preserves its registry release path and
the two-artifact product contract while the upstream crate is unpublished.

## Original paths and modifications

All paths below are relative to upstream `rust/sglang-radix-tree/`:

| Bundled file | Original file |
| --- | --- |
| `mod.rs` | `src/lib.rs` |
| `node.rs` | `src/node.rs` |
| `unified_lru_list.rs` | `src/unified_lru_list.rs` |
| `unified_tree_core.rs` | `src/unified_tree_core.rs` |
| `value.rs` | `src/value.rs` |
| `components/*.rs` | `src/components/*.rs` |
| `page_value_core.rs` | `tests/page_value_core.rs` |

The mechanical modifications are:

- Add attribution headers and qualify upstream `crate::` paths with
  `crate::engine::cache::sglang_radix_tree::`.
- Alias upstream's `thiserror` 1.x as `thiserror1`, preserving the existing
  AISimulate dependency on `thiserror` 2.x. Keep upstream's `hashbrown` 0.16
  with only its borrowed-lookup and inlining features,
  and Linux/aarch64 SHA acceleration dependencies.
- Omit the disabled Torch/PyO3 modules, Tensor implementations, and Torch-only
  test module declarations. Compile upstream's native integration tests as a
  unit-test module instead; retain its inline native value tests.
- Change inspection's `cfg(any(test, feature = "inspection"))` to `cfg(test)`.
  It remains absent from production builds and available to adapter tests.
- Allow unused upstream exports within the private module and apply Rustfmt.

After those adaptations, `scripts/sglang_radix_core_fixes.patch` applies the
following local corrections to this pinned revision:

- Rebuild Full eviction candidates after accesses and final device-lock releases,
  including buried MRU priority improvements, and recheck popped priorities. Preserve skipped Full host and
  SWA early-release lock receipts. Revalidate saved Mamba/SWA device cursors
  against newly acquired component locks before taking values.
- Reject zero Mamba chunk sizes, copy donated Mamba slots for adoption, and
  compare SWA continuation boundaries against the complete insert length.
- Reject Mamba donations with other than one slot before insert mutations;
  leave unused donations with the caller on empty inserts and continuations.
- Preserve unbacked write-through leaves with live host receipts until the
  final release refreshes eviction eligibility. Backed leaves can still demote,
  and write-back leaves can still request a backup, without deleting the anchor.
- Validate continuation key content in release builds during the existing
  ancestor walk, return retained-prefix backup actions for empty suffixes, and
  report reached SWA branch boundaries for those empty continuations.
- Keep stable cursor and commit-target handles for suspended insert walks.
  Reject resumption before mutation if eviction deleted either node, including
  when another node has reused its arena slot.
- Defer demotion while device locks are held or device children have appeared,
  and reject SWA attachment over host-only Full spans before any mutation.
- Count only evicted Full rows as host hits. Finalize Full first and pass the
  logical matched key depth separately to Mamba/SWA branching calculations.
  Compute that depth during the existing recency walk, without another walk
  or path allocation. Skip paged SWA load-back for ring boundaries without
  cached SWA rows.
- Reject Full and auxiliary load-back plans whose per-node row counts changed
  after a split, or whose row totals do not match the device allocation, before
  any adoption.
  Reject auxiliary destinations filled since planning, and retain empty row
  counts on foreign-pin deferrals so they still commit as no-ops.
  Refresh each restored node's parent eligibility before another eviction.
  Publish CPU store events only for the matching write-through acknowledgement.
- Drop failed-backup subtrees in stable NodeId sibling order, descendants first,
  preserving the order of host-free chunks and CPU removal hashes.
- Strengthen the rejected-rotation test to check the retained anchor's hit count,
  access tick, and priority before and after each rejected continuation.
- Keep borrowed namespace fields private so their cached hash stays consistent.
- Describe the bundled module as native-only.
- Borrow child edge keys during removal and copy visible page slices directly
  into compact shared storage during adoption.

These are local deviations from upstream, covered by the AISimulate-authored
`../sglang_radix_regressions.rs` and adapter tests. The pinned eviction policies,
including split recency and whole-leaf LRU eviction, remain in use. Upstream's
native assertions are retained, with the rejected-rotation assertions strengthened.
Adoption removes an intermediate allocation;
removal avoids a key allocation. Continuation validation compares the retained key atoms without allocating a
path or adding another ancestor walk.

## Updating

Start from a verified immutable upstream checkout, preserve its license and
notices, and use `scripts/sync_sglang_radix_core.py` to reproduce the documented
adaptations and the local patch. The script reads committed source at the pinned
revision with replacement refs disabled, without changing the upstream checkout.
Verify an existing copy with:

```bash
python3 scripts/sync_sglang_radix_core.py --source-checkout /path/to/sglang --check
```

For a deliberate version update, update the script's revision, regenerate without
`--check`, and update this revision and both canonical notice copies together.
Reconcile the local patch against the new revision; remove corrections once
they are included upstream. After updating either notice copy, run:

```bash
python3 scripts/check_packaged_legal_files.py
```

Keep the native tests, AISimulate cache/manager/scheduler regressions, and
`cargo package --manifest-path crates/core/Cargo.toml --locked` passing.
Changes to upstream behavior require separate behavioral qualification.
