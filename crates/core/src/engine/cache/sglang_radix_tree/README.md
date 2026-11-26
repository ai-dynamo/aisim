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

- Apply the existing-target terminal backup predicate after empty continuations
  adopt Mamba checkpoints. Emit only one backup for a retained anchor and keep
  empty roots and already-backed checkpoints as no-backup cases.
- Serialize Full load-backs per anchor, rejecting planning or commit while its
  earlier Full source pin is pending. A rejected operation has no acknowledgment;
  retry only after the owning operation is acknowledged.
- Refresh an active Full eviction heap when a resumed insert, device restore,
  or Full backup with a live auxiliary host receipt exposes an eligible leaf.
  Refresh backup eligibility through both commit entrypoints, and use the
  production eligibility predicate for inspection.
- Report an unmatched split suffix's pre-access tick in match/insert results.
  Belady records that tick at the physical tail, including on the first partial
  access before any eviction; an actual suffix access clears the correction.
  This adds constant-size result metadata without another traversal, copied
  page buffer, or change to native LRU ticks or split behavior. Saved tail
  corrections remain bounded by resident physical pages and are removed on
  real accesses or physical frees.
- Refresh an active Full eviction walk after load-back acknowledgments and
  final auxiliary host-lock releases expose eligible leaves. Keep normal
  eviction's incremental parent insertion.
- Report freed unbacked Full rows when a failed write-back backup drops its
  subtree; successful host-backed demotion still reports zero.
- Rebuild Full eviction candidates after accesses and final device-lock releases,
  including buried MRU priority improvements, and recheck popped priorities. Preserve skipped Full host and
  SWA early-release lock receipts. Revalidate saved Mamba/SWA device cursors
  against newly acquired component locks before taking values. Reselect the
  Mamba cursor when a match moves its saved victim to MRU, preserving the
  incremental walk when other nodes are matched.
- Count resident, non-chunked insert hits for LFU in write-back mode while
  keeping write-through backup scheduling disabled for that mode.
- Reject zero Mamba chunk sizes, copy donated Mamba slots for adoption, and
  compare SWA continuation boundaries against the complete insert length.
- Reject missing Mamba donations and donations with other than one slot before
  insert mutations. For an empty continuation at a nonempty retained prefix,
  commit the Mamba checkpoint if missing without replaying Full overlap frees.
  Leave unused donations with the caller at the empty root or an anchor that
  already holds a checkpoint.
- Preserve unbacked write-through leaves with live host receipts until the
  final release refreshes eviction eligibility. Backed leaves can still demote,
  and write-back leaves can still request a backup, without deleting the anchor.
- Validate continuation key content in release builds during the existing
  ancestor walk, return retained-prefix backup actions for empty suffixes, and
  report reached SWA branch boundaries for those empty continuations only when
  SWA is registered.
- Keep stable cursor and commit-target handles for suspended insert walks.
  Reject resumption before mutation if eviction deleted either node, including
  when another node has reused its arena slot.
  Reject demoted retained paths as well; restore their device rows and resume
  the same walk, preserving duplicate frees already consumed at the barrier.
- Validate SWA eviction-boundary alignment before inserting Full KV or events.
- Reject host insertions with a page-hash count that differs from the key's
  page count before touching recency, splitting edges, or attaching host rows.
- Reject unaligned host keys, page hashes other than 64 ASCII hex characters,
  and host buffers too short for the key before any mutation. Preserve imported
  parent hashes as chain seeds.
- Preflight Full, SWA, and Mamba backup spans before any component adopts host
  rows. Reject occupied Full host slots, Mamba transfers with other than one
  checkpoint row, and plans invalidated by an edge split. Empty Full transfers
  still permit auxiliary-only backups. The caller releases rejected staged
  pool allocations and rebuilds the spec before retrying.
- Bind Mamba load-back's first tree transfer to the restore anchor, preserving
  subsequent request-owned copies without treating them as tree destinations.
- Preserve integer-first T-LRU arithmetic when it fits; convert positive
  history-plus-estimate overflow to finite floating-point arithmetic.
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
  Compare ordered Full and auxiliary host source rows and the Full anchor path
  before commit, rejecting plans invalidated by host reclaim and replacement
  with equal counts.
  Index comparisons use borrowed slices without another concatenated buffer.
  Reject auxiliary destinations filled since planning. Mark foreign-pin
  deferrals explicitly so they commit as no-ops even for host-only anchors,
  while ordinary empty plans still require device-resident Full KV.
  Before adopting any rows, recheck that the required SWA window is still
  device-resident or covered by the staged transfer, including device spans
  skipped at planning and plans that originally needed no SWA transfer.
  Reject newly missing spans so the caller can release staging and replan;
  preserve Full-only ring restores. This borrows handles during one
  window-bounded ancestor walk, without copying rows or scanning other nodes.
  Full-only caches skip this validation.
  Refresh each restored node's parent eligibility before another eviction.
  Publish CPU store events only for the matching write-through acknowledgement.
- Drop failed-backup subtrees in stable NodeId sibling order, descendants first,
  preserving the order of host-free chunks and CPU removal hashes.
- Document retained-key validation's O(P + depth) per-call cost, including
  chunked continuations, and O(N^2) work over N one-page fallback appends.
  AISimulate's usual exclusive growth uses its lease-bound suffix append;
  shared/overlapping fallback is bounded by request context and KV capacity.
- Assert the ordered SWA window values as well as presence and row counts.
- Strengthen the rejected-rotation test to check the retained anchor's hit count,
  access tick, and priority before and after each rejected continuation.
- Assert that the read-only prefix probe preserves the compressed edge and its
  access tick immediately, before the mutating match control.
- Assert that chunked continuation leaves the retained anchor's hit count
  unchanged, separately from finished-continuation backup actions.
- Assert rejected SWA attachment and locked suffix eviction leave the original
  four-atom edge unsplit before retrying or releasing the lock.
- Keep borrowed namespace fields private so their cached hash stays consistent.
- Reject duplicate SWA backup targets, occupied Mamba backup destinations, extra
  Full backup descriptors, multiple backup descriptors for a component, and
  disabled component keys before adopting any staged rows. Generic load-back
  commits are rejected; use `commit_load_back` for validation and source pins.
- Reject extra SWA load-back descriptors and any SWA descriptor on a ring
  Full-only restore. The ring exemption permits rebuilding request-owned SWA,
  not adopting rows for an unrelated anchor.
- Share immutable child-edge maps between admission snapshots, copying a map
  only on a structural edit. Recency, component values, locks, and node handles
  remain independently snapshotted. Preserve native node boundaries and eviction
  victims; snapshot-isolation tests cover splits, insertion, and removal.
- Reject both sharded-to-unsharded and unsharded-to-sharded extensions. Revalidate
  a suspended insert's remaining path before restoring a host graft; a rotation
  conflict preserves the walk and returns `InsertRotationMismatch` for cancellation.
- Backfill parent-first storage hashes when storage is enabled, before later
  children can be hashed. Reselect empty auxiliary eviction cursors after unlock,
  and refresh Mamba's saved victim after exact inserts as well as matches.
- Refresh Full eviction after the final device receipt release in either driver
  registration order. Defer failed-backup drops whose victim became ineligible.
  Validate SWA cascade locks before taking rows, both on selection and after a
  deferred backup; skip blocked candidates and continue the walk.
- Hold a Full-only lock receipt on consumed prefixes while SWA inserts suspend.
  This prevents demotion of tombstoned SWA prefixes that normal match load-back
  cannot restore. Release it on resume or cancellation; Full-only inserts keep
  their existing demotion/recovery behavior. No lock walk is added to ordinary
  Full-only simulation, and normal eviction keeps its incremental heap updates.
- Assert the original compressed edge immediately after unaligned suffix eviction
  is rejected, as well as after rejection of locked suffix eviction.
- Describe the bundled module as native-only.
- Borrow child edge keys during removal and copy visible page slices directly
  into compact shared storage during adoption and splitting, avoiding temporary
  vectors for both split halves.

These are local deviations from upstream, covered by the AISimulate-authored
`../sglang_radix_regressions.rs` and adapter tests. The pinned eviction policies,
including split recency and whole-leaf LRU eviction, remain in use. The
[two-leaf split comparison](../EVICTION_EVIDENCE.md#two-leaf-split-recency)
records the legacy and native physical victims for a partial match and its
full-match control. Upstream's
native assertions are retained, with rejected-rotation, SWA-attachment, and locked
suffix-eviction assertions strengthened.
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

For a deliberate version update, update the script's `REVISION` and the SGLang
checkout `ref` in `.github/workflows/fast-ci.yml` to the same immutable commit.
Regenerate without `--check`, and update this revision and both canonical notice
copies together.
Reconcile the local patch against the new revision; remove corrections once
they are included upstream. After updating either notice copy, run:

```bash
python3 scripts/check_packaged_legal_files.py
```

Keep the native tests, AISimulate cache/manager/scheduler regressions, and
`cargo package --manifest-path crates/core/Cargo.toml --locked` passing.
Changes to upstream behavior require separate behavioral qualification.
