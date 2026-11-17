<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.8-Flash-Next C3.5: staged h200_sxm tables

**Status: staged, not published.** This bundle records a merge performed against
the pre-cutover packaged tables. Since `#857` moved operation data into the
pinned `ai-dynamo/aisim-data` dataset, the merged tables do not ship in-tree.
They are held as a candidate bundle (see
[Publication handoff](#publication-handoff)); SILICON resolves for
`Qwen/Qwen3.8-Flash-Next` only after a dataset revision carrying them is merged
and `op_latency_pin.json` is raised to it.

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

## Checks

| check | result |
| --- | --- |
| `check_collector_data.py` (7 rules: sidecar coverage, reuse validity, comm exclusion, family placement, identity v2 resolution, legacy markers, attested events) | all OK |
| Rust `perfmodel::operators::msa` tests | 7 passed |
| Rust engine parity (`test_compile_engine_parity.py`, `test_engine_step_parity.py`) | 389 passed |
| Python `tests/unit/sdk/database` + `tests/unit/sdk/models/test_qwen4_exp.py` | 701 passed, 15 deselected |
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
the QSA slice (see below), which is why the table above is BF16 only.

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
