<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4 Pro calibration qualification (PR #718)

The new SGLang 0.5.21 Pro calibration changes decode predictions by replacing
older shared calibration, even though its keys are disjoint from the existing
Flash rows. [summary.json](summary.json) records a physical-table audit for
B200, B300 and H200 and a Rust-backed before/after comparison. The perf tables,
selection rules and numerical implementation are unchanged by this repair.

Baseline: `9ff96cf9ee5c913160f00915d6bf1e89f6a19f6c`.
Data head: `abc4e92124ff63048ccfbc92e24d6ed6fd0bde63`.
The comparison uses the same optimized release-wheel native binary on both
trees; the Rust and SDK consumer sources are identical at these revisions.
The baseline systems tree is extracted from Git, including its actual module
and fallback tables. This compares data selection, without substituting a
Python latency estimator or synthetic measurements.

## Physical data and reachability

| System | Baseline rows | Added Pro rows | Final rows | SDK matches for added rows |
| --- | ---: | ---: | ---: | ---: |
| B200 | 5,182 | 1,526 | 6,708 | 1,526 |
| B300 | 5,182 | 1,526 | 6,708 | 1,526 |
| H200 | 4,502 | 1,526 | 6,028 | 1,526 |

Each table has 18 columns. Its original rows are preserved in order and value;
the new rows all have 128 native heads. The key is
`(num_heads, step, isl, batch_size, score_mode)`. There are no duplicate keys,
no overlap with the baseline primary keys, no missing selector pairs, and no
null keys, negative/nonfinite latencies or reversed flat/top-last pairs.
Each system adds 563 v1 context pairs and 200 v2 decode pairs over 699 distinct
`(step, isl, batch_size)` coordinates. The JSON lists every axis value and the
phase counts; these describe a ragged grid, not a Cartesian coverage claim.

The 896 zero-latency rows per system are expected: the collector emits a zero
pair when the compressed causal length is at most the selector size (Pro
`index_topk=1024`), so no selection or degenerate-score correction is needed.
See `_bench_topk_shape` in
[deepseekv4_sparse_modules.py](../../../python/aisimulate/collector/sglang/deepseekv4_sparse_modules.py).
Latency units are milliseconds; `isl` and `step` are query and past-KV tokens
per request. The report also records schemas, hashes and collection events
(`eb04c6888cf33ef4f48a995511a23429127f3baa`, collected 2026-11-04).

Every added row is compared with the same coordinate in
`fetch_table_view(db, "_dsv4_csa_topk_calib_data")`, including its latency.
This exercises the actual SDK binding, Rust resolver, parquet loader and
table-view FFI with strict provenance enabled. The regression tests repeat
this check against the installed wheel's three shipped tables.

## Why the older calibration is replaced

The shared-layer contract gives the requested version's own measurements
priority, then fills missing keys from eligible donors. Before this change,
0.5.21 had only Flash/64-head primary calibration, so Pro/128-head queries
inherited 0.5.14 rows. The new primary keys overlap 1,526 donor keys on B200
and B300 and 1,334 on H200. Disjointness from Flash therefore does not imply
unchanged Pro predictions.

The intended replacement uses Pro's 1024-entry selector measured with the
matching 0.5.21 runtime and collector, rather than carrying the older runtime's
degenerate-score penalty forward after same-version measurements exist.
The 0.5.21 sidecar pins its runtime image and collection hashes; the 0.5.14
sidecar is marked `provenance: legacy`. The calibration schema distinguishes
v1 context and v2 decode. The current collector measures flat versus
representative top-last scores under CUDA graphs. Its entry-point mapping is documented in
`_bench_topk_512`. A larger legacy flat-score penalty would subtract too much
from the unchanged 0.5.21 module measurement **relative to the newly measured
0.5.21 correction**. Keeping the donor in preference to these rows would
contradict the existing primary-first contract.

This is a version-matched calibration rationale. The archived measurements
alone do not identify the GPU/kernel cause of their timing difference or
establish end-to-end serving accuracy. This evidence qualifies the intended
consumer output and selection; it does not claim held-out serving validation.

## Before/after consumer result

The consumer is `EngineHandle.evaluate_ops_json` with `Dsv4Generation`, B200,
SGLang 0.5.21, CSA, native/local heads 128, TP1, batch 1, KV FP8 and
GEMM `fp8_block`. The JSON includes Pro HCA and Flash CSA controls.

| Total sequence tokens | Previous Pro CSA (ms) | New Pro CSA (ms) |
| ---: | ---: | ---: |
| 4,097 | 0.118000000 | 0.118000000 |
| 8,192 | 0.061421604 | 0.121351603 |
| 8,193 | 0.061422000 | 0.121352000 |
| 16,385 | 0.061060000 | 0.120940000 |

At total length 8,193, `past_kv=8192` is an exact calibration anchor:

- The unchanged module measurement is **0.1234 ms**.
- The old v2 flat/top-last pair is **0.065100 / 0.003122 ms**, giving a
  **0.061978 ms** correction and **0.061422 ms** corrected module latency.
- The new pair is **0.005172 / 0.003124 ms**, giving a **0.002048 ms**
  correction and **0.121352 ms** corrected module latency.

At total length 8,192, the consumer queries `past_kv=8191`, interpolating
between calibration points. The observed result is
**0.12135160253906249 ms**, up **97.5715%** from
**0.06142160400390624 ms**. This increase comes from subtracting a much
smaller flat-score penalty, while the representative top-last measurement
and the underlying module latency barely change or remain unchanged.
The report verifies that both B200 generation-module files are byte-identical
between baseline and head. All four Flash CSA and Pro HCA controls are
unchanged, including the zero-correction Pro point at total length 4,097.

The two new output-regression cases pin the reviewed interpolated result and
the independently explained exact anchor. The latter also checks the physical
module and calibration measurements used in the arithmetic above.

## Reproduction and checks

From the repository root, after installing a wheel built with
`maturin build --release --locked` from `python/aisimulate`:

```sh
python benchmarks/evidence/dsv4-pro-topk-calibration/generate.py
pytest -c python/aisimulate/pytest.ini \
  python/aisimulate/tests/unit/sdk/database/test_dsv4_pro_calibration_evidence.py
```

The generator requires the two Git revisions above, pyarrow, PyYAML and the
installed native wheel. Its temporary baseline tree respects `TMPDIR`.
It processes at most 6,708 rows per calibration table and reuses one engine
handle for each set of comparison queries. This repair adds evidence and
tests; it adds no production query work or cache. These values are predicted
per-module GPU latencies, separate from the host cost of running the audit.

The broader targeted run covers the SDK table views, sparse tables,
collection-data checker and provenance tests (100 passed). Repository Ruff
lint and formatting checks and the R1–R7 collection-data check pass.
