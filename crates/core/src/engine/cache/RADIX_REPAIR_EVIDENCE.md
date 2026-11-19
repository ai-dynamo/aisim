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
  --candidate e2f62f787b54bc6440d8e3b290e5327b3ee6e8dc \
  --work-dir "$TMPDIR/lifecycle-evidence"
```
