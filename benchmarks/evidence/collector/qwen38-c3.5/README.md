<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.8-Flash-Next C3.5: staged h200_sxm tables

**Status: staged, not published.** This bundle records a merge performed against
the pre-cutover packaged tables. Since `#857` moved operation data into the
pinned `ai-dynamo/aisim-data` dataset, the merged tables do not ship in-tree.
They are held as a candidate bundle (see
[Publication handoff](#publication-handoff) and
[`publish-handoff.json`](publish-handoff.json)). SILICON resolves for
`Qwen/Qwen3.8-Flash-Next` only after a dataset revision carrying them is merged
and `op_latency_pin.json` is raised to it — and only for a backend whose
collected version is a queryable slot (SGLang 0.5.21 is; vLLM 0.31.0 is not).

This non-packaged bundle records the C3.5 merge step for
`Qwen/Qwen3.8-Flash-Next`: merging the accepted C3.3 QSA and C3.4
GEMM/MoE/GDN campaign rows into the `h200_sxm` perf tables, and checking that
the canonical consumer can actually read them.

It is data-plumbing evidence, not serving accuracy. C3.6 (first FPM cell) and
A6 (8-GPU serve + replay comparison) are separate stages.

## Sources

| lane | artifact | accepted ref |
| --- | --- | --- |
| SGLang 0.5.21 QSA | `model-onboarding/c3/logs/c3.3-sglang-heal/` | context `e7492da4…` (staged-package digest), generation `3b98b3dae…` |
| vLLM 0.31.0 QSA | `model-onboarding/c3/logs/c3.3-vllm/out/` | `be0df6ed85…` |
| SGLang 0.5.21 GEMM/MoE/GDN | `model-onboarding/c3/logs/c3.4-full/parquet/` | tag `c3.4`, `2090ecdb90…` |
| vLLM 0.31.0 GEMM/MoE/GDN | `model-onboarding/c3/logs/c3.4-full/parquet/*.vllm.parquet` | tag `c34-fix2`, `5613bf7763…` |

Runtime identity: vLLM `v0.31.0`
(`sha256:c1c9f6fd…67c3`, source commit `db9527a4…`), SGLang `v0.5.21`
(`sha256:b1259f3e…f469`). Both were measured on H200/sm90. Precision is BF16,
which is also the FP8 checkpoint's serving dtype: it quantizes routed experts
only, so QSA and its KV pool stay BF16 and both checkpoints read these rows
(see [Known gaps](#known-gaps)).

The SGLang context `collector_ref` `e7492da4e09ea12056cdea5adab33bce9fec36fe7e0277835e76356da40bce36`
is a 64-hex digest of the staged heal package, not a git object.
`collector_hash` is the byte attestation that the sidecar validator checks.

## Merge result

`model-onboarding/c3/80-publish-c3.5.py --apply` appended the campaign rows,
key-sorted, after the packaged rows and preserved their row order.
`model-onboarding/c3/81-verify-c3.5.py` re-derives the pre-merge table from
`git HEAD`, so it stays valid after the copy. The packaged base compared here is
`f6c89241f`, byte- and row-identical (row-for-row after restamping) to the
`ai-dynamo/aisim-data@c231f32b` tables under the same version keys, so the merge
applies unchanged to the dataset tree.

| published table | pre-merge | campaign | merged | overlap | codec |
| --- | ---: | ---: | ---: | ---: | --- |
| `gemm/sglang/0.5.21` | 103970 | 1924 | 105894 | 0 | snappy |
| `moe/sglang/0.5.21` | 111942 | 3564 | 115506 | 0 | snappy |
| `linear_attention/sglang/0.5.21` | 9977 | 944 | 10921 | 0 | snappy |
| `gemm/vllm/0.31.0` | 109742 | 1924 | 111666 | 0 | zstd |
| `moe/vllm/0.31.0` | 891 | 2025 | 2916 | 0 | zstd |
| `linear_attention/vllm/0.31.0` | — (new) | 984 | 984 | 0 | snappy |
| `msa/sglang/0.5.21` context | 33112 | 11782 | 44894 | 0 | snappy |
| `msa/sglang/0.5.21` generation | 1841 | 740 | 2581 | 0 | snappy |
| `msa/vllm/0.31.0` context | — (new) | 976 | 976 | 0 | snappy |
| `msa/vllm/0.31.0` generation | — (new) | 736 | 736 | 0 | snappy |

`merge-check.json` is the machine-readable form. For every table it asserts:

- campaign rows equal the accepted count in the C3.3/C3.4 sidecars;
- the campaign key set is disjoint from the `git HEAD` key set (zero overlap —
  no existing cell was re-measured);
- merged rows = published + campaign and the merged key set is exactly the
  union, with no duplicate identity keys;
- no nulls in any column and every latency is finite and positive;
- the parquet codec and SHA-256, and that the sidecar's `# data_sha256:`
  comment for that table ends with the digest on disk.

The full digests are in `merge-check.json`; for example
`msa/sglang/0.5.21/msa_context_module_perf.parquet` is
`bc843d842b501b68af69d560242063501929e63331e3805c31af988225eb2f92` and
`msa/vllm/0.31.0/msa_context_module_perf.parquet` is
`b8ca9b641f3d99f5184451879c32844da8006bc855c756d6ae18f45fdc5ff28d`.

## Publication handoff

The merged tables are not shipped from this repository: `#857` moved operation
data into the pinned `ai-dynamo/aisim-data` dataset, so a default install reads
`op_latency_pin.json` (`ai-dynamo/aisim-data@c231f32bf3510fcd1addfe613d14619a79a4c24e`).
Everything below is a **candidate bundle, built and verified locally and never
uploaded**; `publish-handoff.json` is its machine-readable record.

Publication stops here for four reasons, none of which this change can resolve:

- `ai-dynamo/aisim-data` is NVIDIA-owned and the account preparing this bundle
  has no write access.
- A Hugging Face pull request cannot come from a fork: `hf upload
  --create-pr` pushes a `refs/pr/N` revision into the source repository, so
  holding a fork is not enough.
- The dataset license is still pending OSRB (AISIM-2311/AISIM-2312, "do not
  redistribute until it is set").
- The public-profile content check fails on the published base revision itself,
  so promotion has to carry a redaction step first (see below).

### Build and gates

Built with `tools/perf_database/publish_op_latency_tree.py` from a composed
systems root: the pinned dataset tree (`c231f32b`, 3197 files, no per-leaf
manifests) plus the pre-cutover `perf_data_reuse_manifest.yaml` extended with the
five Qwen reuse groups, plus the 18 accepted C3.3/C3.4 files (10 tables and their
sidecars) from `eb183db59`.
Feeding the dataset tree itself to `build` fails artifact integrity, because it
ships its own `manifest.json` files; the composed root is the input.

| gate | result |
| --- | --- |
| `build` | 3204 sources → 1188 manifests, 1998 converted tables, 1206 copied, 0 skipped; 239266718 B → 244545037 B; 56 declared-reuse groups resolved |
| `audit` | `ok: true`, 3204 checked, 0 problems, 0 unmapped sources, 0 unmapped destinations (1996 `arrow-rows`, 1206 `file-bytes`, 2 `embedded-source`) |
| `check_dataset` | pass — `aisim-dataset-index/v1`, op-latency index re-derived from the built tree |
| `check_catalog_size` | no findings (0 errors, 0 warnings) |
| mirror public profile | **fails, and fails on the base revision too** — 3551 errors: 3507 `image-name`, 37 `image-digest`, 4 `patch-reference`, 2 `orphan-file`, 1 `internal-host`; validator fingerprint `b0c8406cb360` |
| candidate, not promotion | nothing writes the pin, the package or the mirror; `status` in `publish-handoff.json` stays `candidate-built-not-uploaded` |

The public-profile result is dataset-wide, not a property of these rows: the
pinned `c231f32b` revision fails in every system and family, and this bundle
fails the same way. Each of the eight cells contributes four `image-name`
findings from the same `image` / `image_digest` sidecar fields that every other
cell carries; no new finding code appears. `docs/ci/accuracy.md` marks the
mirror validator and promotion tool as **planned**, so public disclosure is a
prerequisite the promotion step must satisfy (redact or `[REDACTED]` the
withheld fields), not a defect introduced here.

Row accounting, independent of the builder's own checks: all ten tables carry
the merged row counts in `merge-check.json`, and every `source_sha256` recorded
in the bundle's `source_map.parquet` equals the SHA-256 of the `eb183db59` blob
it was converted from. The QSA slices are 11782 context + 740 generation rows on
SGLang 0.5.21 and 976 + 736 on vLLM 0.31.0.

### Bundle identity

| object | value |
| --- | --- |
| bundle | 4411 files (4393 data, 18 catalog), 244545037 B, SHA-256 `e3af85b31631e0bfaeb7bc366ffa67576a3a1ca63d6b330b2b7f02ec37066367` |
| `catalog/` | SHA-256 `84f82db0571a9241d0a9c77300e6096482fa7467cbe8b1b8d70aa905e576a215` |
| `catalog/op-latency.json` | 1219223 B, SHA-256 `649bbe23723ebc038a07d74711d659a098085ed24e62b541a5a21549bab9bac9` |

Both tree digests are SHA-256 over sorted `<relative path>\0<sha256(file)>\n`
lines, so they are reproducible from a checkout. Per-cell digests:

| cell | rows | SHA-256 |
| --- | ---: | --- |
| `h200_sxm/gemm/sglang/0.5.21` | 105894 | `47ab482fe6f1bd1888070966d6db5c27d1805bbe9bb17b8e1dc518eded0e8100` |
| `h200_sxm/gemm/vllm/0.31.0` | 111666 | `27326afad611ed5b1a60d2742bff936ad0bb1a01d4f77f2602d141d719ce2367` |
| `h200_sxm/moe/sglang/0.5.21` | 115506 | `aa5cf99d61c52dbe7fb7b9b431472d31bad44440421d0dfa1e194bef6c405883` |
| `h200_sxm/moe/vllm/0.31.0` | 2916 | `771957e1c934cf8c37522462193bf68eb74c915bf96cf479b836b090f7c6ab14` |
| `h200_sxm/linear_attention/sglang/0.5.21` | 10921 | `f586c3a0ba568801de489252f4835832ff189b0b3a9736c854338c72a24c5b09` |
| `h200_sxm/linear_attention/vllm/0.31.0` | 984 | `7f4e3a74e17ee3cda0449ea27ccb5d9e5d9b59da0748d82f0f03cd27edbf27f2` |
| `h200_sxm/msa/sglang/0.5.21` | 2581 | `273f180b04287c8c78340f1538fb12c66bba0bc488592cd986012b08bb584bea` |
| `h200_sxm/msa/vllm/0.31.0` | 736 | `c5df4ce2e4e569b0d06c9f1a31443c53841cd84ee6e3c751f2e1637b3261744e` |

### Payload, and what the bundle is not

The bundle verifies the merge; it is not the upload payload. `build`
re-converts every parquet source, so its tables are not byte-identical to the
published dataset even where the content is unchanged (sampled:
`h200_sxm/gemm/vllm/0.32.0/gemm_perf.parquet` 515920 → 515774 B). The audit
proves the content is preserved on its `arrow-rows` basis, but the bytes are a
new encoding. The composed root is also not a git checkout, so every manifest
the build regenerates carries `source_commit: null`, where the published
manifests carry `dbff10b4c32bdcd808356f4110e4b0c5fae134d4`.

Uploading the bundle's category tree as-is would therefore rewrite every table:
1996 measurement tables (the 1993 carried over plus the 3 new ones) and the two
lookup tables, plus all 1188 manifests, and drop the commit lineage from every
manifest. That is not a merge of ten tables. The payload is:

| payload | files | location |
| --- | ---: | --- |
| 10 merged tables and 8 sidecars | 18 | `eb183db59:python/aisim/src/aisim_core/systems/data/h200_sxm/…` |
| five Qwen reuse groups | +25 lines | `perf_data_reuse_manifest.yaml`, reproduced in `publish-handoff.json.sources` |

Five of the 18 files are new (three tables and two sidecars under
`msa/vllm/0.31.0/` and `linear_attention/vllm/0.31.0/`); the other thirteen
replace existing cell files. The bundle's own tables are a re-encoding of these,
not a different measurement.

Publishing them needs the post-cutover flow, which this repository does not
have: `publish_op_latency_tree build` expects the pre-cutover
`python/aisim/src/aisim_core/systems/data/` layout and stamps
`source_commit` from that checkout, and no `main` commit carries the layout
since `#857`. Two ways out, both maintainer calls:

1. **Apply the payload in a checkout that has the layout**, run `build` so the
   manifests keep a real `source_commit`, then upload the resulting category
   tree. `catalog/op-latency.json` and `catalog/op-latency/h200_sxm.parquet`
   are re-derived by the same run; every other shard stays byte-identical.
2. **Upload the verified bundle category** (`data/op-latency/`,
   `catalog/op-latency.json`, `catalog/op-latency/`) as a one-off and accept a
   broad, row-identical diff with null manifest lineage.

Either way, do **not** upload the bundle's `catalog/dataset.json`: it is a
single-category index, while the dataset's copy names `e2e`, `fpm` and
`op-latency`. Set `categories.op-latency.index` in the existing file to the
uploaded `catalog/op-latency.json` bytes and SHA-256 instead, keeping the other
two references. `catalog/op-latency-reuse.parquet` and `catalog/systems.parquet`
are byte-identical to the base, so re-uploading them is a no-op.

`hf upload` only adds or overwrites, so the six system-level files under
`b200_sxm/` that sit outside the builder's
`<system>/<family>/<backend>/<version>/` layout (`README.md`, two
`power_upstream/*.parquet.source`, and the `vllm-0.25.0` report, notes and
failure bundle) stay in place regardless of channel. The file counts reconcile:
4389 base system files + 7 new-cell files − 6 unmanaged aux files = 4390, plus 3
root parquet files = 4393.

After that revision merges, raise `revision` in
`python/aisim/src/aisim_core/data/op_latency_pin.json` to it. The order cannot be
reversed: CI downloads the pinned tree, so a pin bump ahead of the merge has
nothing to fetch. The bump is a maintainer step, not part of this change.

The bump makes the QSA slice resolvable under SILICON only where the collected
version is a queryable one. The dataset's slots are `sglang 0.5.23/0.5.21` and
`vllm 0.32.0/0.30.0` (identical in the in-tree `systems/query_versions.yaml`),
so:

| lane | collected | slot | after publish + pin bump |
| --- | --- | --- | --- |
| SGLang | 0.5.21 | `previous` | queryable |
| vLLM | 0.31.0 | neither | `UnlistedQueryVersionError` unless `AIC_ALLOW_UNLISTED_VERSIONS=1` |

That was probed directly on this tree, not inferred:
`get_database_view("h200_sxm", "vllm", "0.31.0", database_mode="SILICON")` raises
while `vllm/0.32.0`, `sglang/0.5.21` and `sglang/0.5.23` return a view. The
escape hatch is an environment variable; there is no CLI flag. The bundle as
built leaves the slots untouched, and `decisions` in `publish-handoff.json`
records the maintainer choice: keep the slots (vLLM lane needs the escape
hatch), promote 0.31.0 to `previous` (a dataset-wide change affecting every
vLLM family), or recollect the lane on 0.32.0.

## Checks

| check | result |
| --- | --- |
| `check_collector_data.py` (7 rules: sidecar coverage, reuse validity, comm exclusion, family placement, identity v2 resolution, legacy markers, attested events) | all OK |
| Rust `perfmodel::operators::msa` tests (includes the two guard tests) | 9 passed |
| Rust `cargo test -p aisim-core --lib` (full crate) | 3576 passed, 0 failed, 2 ignored |
| Rust engine parity (`test_compile_engine_parity.py`, `test_engine_step_parity.py`) | 389 passed |
| Python `tests/unit/sdk/database` + `tests/unit/sdk/models/test_qwen4_exp.py` | 737 passed (1 warning) |
| ruff check / format | clean |

`cargo clippy -D warnings` is red on this checkout's Homebrew rustc 1.99 for
pre-existing lints elsewhere in the crate; the files touched here produce no
clippy diagnostics.

### Consumer gap fixed in this step

The Rust `MsaModuleOp` hardcoded `MiniMaxM3ForCausalLM` as the msa-table
architecture, so the accepted Qwen QSA rows were unreachable regardless of the
data. The op now carries an `architecture` field (serde-defaulted to M3 for
existing specs), the PyO3 wrapper exposes it as a keyword/getter and round-trips
it through `__getnewargs_ex__`, and the Qwen4-Exp model builder passes
`architecture="Qwen4ExpForConditionalGeneration"`. The new tests
`msa_silicon_lookup_keys_on_the_op_architecture` and
`msa_op_json_without_architecture_defaults_to_m3` pin both behaviours.

Because that field sits between `scale_factor` and `num_heads` on the positional
bincode wire, `ENGINE_SPEC_SCHEMA_VERSION` moved 29 → 30 in the original
campaign branch, which landed as 30 → 31 on this tree (AFD already owned 30);
the changelog is in both `crates/core/src/perfmodel/config.rs` and
`aisim_core/sdk/engine.py`. The serde default covers JSON only: a schema-30
bincode payload is rejected instead of being misparsed.

## Consumer queries

`model-onboarding/c3/82-consumer-queries.py` calls the canonical
`RustForwardPassPerfModel.best_available` with `database_mode=SILICON`,
`estimation_mode=op_level`, `fallback_policy=deny`, aggregated TP8 / MoE EP8,
one 8192-token request (prefill and decode). Results are in
`consumer-queries.json`.

| backend | BF16 prefill | BF16 decode |
| --- | ---: | ---: |
| vLLM 0.31.0 | 144.092 ms | 5.850 ms |
| SGLang 0.5.21 | 162.708 ms | 5.606 ms |

Both backends resolve from SILICON; no fallback is involved. FP8 fails closed on
the QSA slice (see below), which is why the table above is BF16 only. This run
also predates the data cutover and used the merged in-tree tables, where vLLM
0.31.0 was the `next` slot; on the current tree it is outside the slots, so the
vLLM lane needs the published revision plus a slot promotion or
`AIC_ALLOW_UNLISTED_VERSIONS=1`.

## Default `aisimulate predict` workflow

`configs/predict-{vllm,sglang}.yaml` run the ordinary unified CLI (synthetic
1024-in / 128-out, concurrency 8, 64 requests, `context_length` 4096, the C2
serving topology). Both complete with `source=aic` and the pinned data version.

| metric | vLLM 0.31.0 | SGLang 0.5.21 |
| --- | ---: | ---: |
| mean TTFT (ms) | 125.19 | 150.03 |
| mean ITL (ms) | 7.54 | 7.44 |
| mean request latency (ms) | 1082.32 | 1094.76 |
| output throughput (tok/s) | 946.1 | 935.4 |
| request throughput (req/s) | 7.39 | 7.31 |

Artifacts are under `predict-vllm/` and `predict-sglang/`. Power detail is
unavailable (insufficient energy coverage); this bundle makes no power claim.
These numbers are a plumbing smoke, not an accuracy result.

Both runs read the merged tables from the repository tree as it stood before
the data cutover. On the current tree neither config runs, for two independent
reasons: the model's rows are not in the pinned dataset (the SGLang config fails
first on the Qwen MoE shape), and vLLM 0.31.0 is no longer a listed query
version (that config fails during validation, before any data lookup).
Reproducing them needs the published revision **and**, for vLLM, either a slot
promotion or `AIC_ALLOW_UNLISTED_VERSIONS=1`; SGLang 0.5.21 needs only the
published revision.

Reproduce from the repository root:

```bash
python/aisimulate/.venv/bin/aisimulate predict \
  --config benchmarks/evidence/collector/qwen38-c3.5/configs/predict-vllm.yaml \
  --output-dir benchmarks/evidence/collector/qwen38-c3.5/predict-vllm \
  --overwrite
```

Use `predict-sglang.yaml` for the SGLang run.

## Known gaps

- **FP8 QSA keys on BF16 (resolved).** `msa` rows are BF16-only, which is
  correct: the FP8 release quantizes routed experts only. The SDK used to infer
  an FP8 KV pool (and, on SGLang, an FP8 FMHA) from the weight precision alone,
  so the FP8 checkpoint resolved to `DsaKey { kv_quant: "fp8" }` (vLLM) and
  `DsaKey { fmha_quant: "fp8", kv_quant: "fp8" }` (SGLang) and raised
  `PerfDataNotAvailableError`. Both keys describe a quantization neither
  runtime enters: the checkpoint has no `kv_cache_scheme` and lists every
  attention projection in `modules_to_not_convert`, so the QSA module runs at
  the projection dtype over a BF16 pool. `_infer_quant_modes_from_raw_config`
  now keeps KV and FMHA in BF16 for `Qwen4ExpForConditionalGeneration` when the
  checkpoint excludes `self_attn` and declares no KV algorithm; an explicit
  `kv_cache_quant_algo` (or an explicit `kvcache_quant_mode`) still wins and
  fails loudly. Whole-model prefill latency (TP8/EP8 aggregated, b=1,
  input 8192, `estimation_mode=op_level`), measured after the key fix:

  | backend | precision | SOL | SILICON | HYBRID | EMPIRICAL |
  | --- | --- | ---: | ---: | ---: | --- |
  | vLLM | BF16 | 71.3636 ms | 144.0921 ms | 144.0921 ms | `PerfDataNotAvailableError` |
  | vLLM | FP8 | 67.4970 ms | 144.1256 ms | 144.1256 ms | `PerfDataNotAvailableError` |
  | SGLang | BF16 | 70.8946 ms | 162.7078 ms | 162.7078 ms | `PerfDataNotAvailableError` |
  | SGLang | FP8 | 67.0280 ms | 160.4421 ms | 160.4421 ms | `PerfDataNotAvailableError` |

  These rows were measured before the fail-closed guard below landed; they
  describe that tree, where EMPIRICAL surfaced as `PerfDataNotAvailableError`
  through the DSA-transfer path. On the current tree the same query raises
  `EmpiricalNotImplementedError` from the guard (both are typed coverage misses
  and both are reported as `Error: <message>` by the CLI).

  Before the fix the FP8 SILICON row raised `PerfDataNotAvailableError` on both
  backends; the typed keys are recorded verbatim in
  `onboarding-checkpoint.json` (revision 49) and `consumer-queries.json` now
  carries the post-fix results. EMPIRICAL raises because there is no measured
  QSA util and the M3-calibrated DSA transfer is refused for this architecture;
  earlier it raised for the weaker reason that the donor key was absent. No
  collection was needed for this: the accepted BF16 rows already describe the
  FP8 serving lane.

- **The DSA transfer is fail-closed for other architectures.** Until this
  bundle, `MsaModuleOp` skipped the transfer only when no DSA donor slice
  existed. On SGLang 0.5.21 the donor table does exist (`GlmMoeDsaForCausalLM`,
  head grid `[8, 16, 32, 64, 128]`), so a Qwen op without its own rows would
  have silently borrowed a value measured 15-28x below the M3 module, at TP1/TP2
  but not TP4/TP8. `ensure_xop_calibrated` now makes any architecture other
  than `MiniMaxM3ForCausalLM` raise the typed terminal miss in HYBRID and
  EMPIRICAL instead of borrowing; SILICON was already strict. Pinned by
  `msa_xop_refuses_an_uncalibrated_architecture` and
  `msa_xop_guard_leaves_calibrated_and_collected_paths_intact`.
- SGLang QSA context has 11782 of the 12000 planned points (98.2%). The audit
  is closed: the plan grid is 3000 points per TP and the shortfall is exactly
  the counted device-memory filter, `[QSA] plan: dropped N/M shapes` — TP1
  drops 10/240, 18/216, 24/192 and 28/168 (batches 128/256/512/1024 at seq
  <=1024 = 80 points) and TP2/4/8 drop 9/216, 16/192 and 21/168 (batches
  256/512/1024 at seq <=512 = 46 points each). 80 + 3x46 = 218 = the shortfall,
  with no unexplained missing point and no unexpected key. Those cells exceed
  the 2200 B/token pool + 4096 B/token QSA workspace budget, so they are a
  feasibility drop, not lost data. The filter has since moved out of the op plan
  into the worker: the plan now queues every point and the worker refuses one it
  cannot hold, before allocating it, with a `FIXME(kernel-limit): does not fit`
  reason. The same cells are therefore recorded as refused instead of absent;
  no collected row changes.
- SGLang GDN has 472 of 492 planned points (the `causal_conv1d` int32 envelope
  drops the rest). Every missing SGLang GDN point has a vLLM counterpart.
- MoE topology coverage is kernel-limited, not missing work: SGLang fp8_block
  only `moe_tp_size=1`; BF16 up to `moe_tp_size=16`; vLLM fp8_block only
  `moe_tp_size<=4`; 9 off-topology vLLM fp8_block cases are recorded as
  `kernel_limit`. The serving topology (`moe_tp_size=1`, `moe_ep_size=8`) is
  present for both dtypes on both backends.
- NVFP4 is deferred: it needs Blackwell hardware, which was not available.
- This stage does not establish serving accuracy. C3.6/A6 still have to compare
  measured serve against these predictions.
