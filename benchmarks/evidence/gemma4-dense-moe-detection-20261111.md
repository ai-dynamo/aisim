<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dense Gemma 4 MoE classification evidence

The shape and recommendation probes below were originally reported on
2026-11-13 on macOS arm64 (Python 3.13.18), comparing the then-branch sources
with `f85518b36`. Their reported pre-rebase identity, `29bb28580`, is not
available in this checkout; these tables remain historical author-reported
results, not verification of the current head. No GPU was run, and no
performance data was added, re-collected, or re-fitted.

The published review head is `8f3a62e0fde729cf74db0b6b3551608338b367dd`,
based on `0809faae9e543487dcf2142dc6afe2427a42f9bd`. The classifier fix
is in `7d97aac50`; `ad2bb0b68` adds the embedded bridge tests, and
`a584153b9` adds the request-wide fallback regression and the CI enforcement
flag. **`8f3a62e0` changes production behavior**: both expert fields must be
integers (excluding booleans), or absent, before sentinel and range checks.
Earlier results do not validate that change. The exact-head verification below
supersedes the PR description's `a584153b9` test totals. The branch does change
`.github/workflows/ci.yml` to enforce bridge imports in an admitted Full CI run.

## Scope

This evidence covers parallel-shape selection, the generator producer
enumeration, the CLI and SDK consumers of the classification, and the public
recommendation path for the two Gemma 4 checkpoint kinds. It does not measure
serving accuracy, latency, or throughput. The recommendation probe supplies
constant metrics through offline runner and performance-model doubles, so
candidate scores are placeholders; only the selected shape set and the
feasible/infeasible status are meaningful.

## Exact-head verification (2026-11-18)

Local verification below covers the unchanged published head
`8f3a62e0fde729cf74db0b6b3551608338b367dd`, on Linux x86_64 (WSL2),
Python 3.12.3, pytest 9.1.1 and Rust 1.91.1. A maturin 1.16.0 release wheel
was built first, using the CI wheel's optimized release profile and
`extension-module,jemalloc` features from `pyproject.toml`. Dependencies,
wheel, Cargo targets, caches and pytest data were kept outside the checkout.
No weights were downloaded. The installed classifier and KV-cache sources were
byte-compared with the checkout before testing.

With `REPAIR_SCRATCH` set to an external scratch directory, the build and test
commands were:

```bash
export TMPDIR="$REPAIR_SCRATCH/tmp"
export CARGO_TARGET_DIR="$REPAIR_SCRATCH/cargo-target"
export UV_CACHE_DIR="$REPAIR_SCRATCH/uv-cache"
export CARGO_BUILD_JOBS=2
export UV_PROJECT_ENVIRONMENT="$REPAIR_SCRATCH/venv"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
uv sync --project python/aisim --frozen --extra dev --no-install-project
(cd python/aisim && uv tool run --from 'maturin>=1.12,<2' maturin build \
  --release --interpreter /usr/bin/python3 --auditwheel skip \
  --out "$REPAIR_SCRATCH/wheels")
uv pip install --python "$REPAIR_SCRATCH/venv/bin/python" --no-deps \
  "$REPAIR_SCRATCH"/wheels/*.whl
source "$REPAIR_SCRATCH/venv/bin/activate"
export PYO3_PYTHON="$VIRTUAL_ENV/bin/python"
export PYTHONPATH="$VIRTUAL_ENV/lib/python3.12/site-packages"
export AIC_REQUIRE_DENSE_GEMMA4_BRIDGE=1

python -m pytest -c python/aisim/pytest.ini \
  -o cache_dir="$REPAIR_SCRATCH/pytest-cache" \
  --basetemp="$REPAIR_SCRATCH/pytest-pinned-rerun" \
  python/aisim/tests/unit/generator/test_gemma4_dense_enumeration.py \
  python/aisim/tests/unit/cli/test_disagg_moe_comm_resolution.py \
  python/aisim/tests/unit/sdk/models/test_model_config.py \
  python/aisim/tests/unit/sdk/task_v2/test_task_config.py \
  python/aisim/tests/unit/sdk/test_memory_estimation.py \
  python/aisim/tests/unit/sdk/test_moe_kernel_source_consumers.py \
  python/aisim/tests/unit/sdk/test_utils.py \
  python/aisim/tests/unit/test_modern_recommend_context.py \
  python/aisim/tests/unit/support_matrix
cargo test -p aisim-core --features embed-python --lib \
  --test perfmodel_dense_gemma4_bridge gemma4 -- --nocapture
```

| Verification | Tested commit | Result |
| --- | --- | --- |
| Targeted pytest command above | `8f3a62e0fde729cf74db0b6b3551608338b367dd` | 1,160 passed |
| Rust policy test `dense_gemma4_expert_parallelism_is_not_fallback_safe` | `8f3a62e0fde729cf74db0b6b3551608338b367dd` | 1 passed |
| Embedded `perfmodel_dense_gemma4_bridge`, with import enforcement enabled | `8f3a62e0fde729cf74db0b6b3551608338b367dd` | 4 passed; no skips |

The pytest command includes all 19 inconsistent-metadata cases and all 48
public fallback-path type cases named in the regression table below. In the
first full invocation, while the embedded test executable was compiling,
1,159 passed and the public recommendation test failed because all eight
candidates were `resource_limited` (`evaluated=0`). That test then passed in
isolation, and the complete unchanged selection passed after compilation
finished. No assertion, timeout or production source was changed for the rerun.
The initial failure and both subsequent results are retained in the repair
run's scratch logs (`pytest-pinned.log`, `pytest-resource-probe.log`, and
`pytest-pinned-rerun.log`); native results are in `native-pinned.log`.

The PR description's 1,098-pass result belongs to `a584153b9`. Its older full
unit result (14,014 passed, 75 failed, 45 errors) was reported at `29bb28580`,
whose object is unavailable here; that broad suite was not revalidated by this
targeted rerun. Neither historical total is current-head passing evidence.

### Hosted CI evidence, separate from local verification

The Actions API query for exact head `8f3a62e0fde729cf74db0b6b3551608338b367dd`
on 2026-11-18 returned Fast CI, codeowners and GitHub Pages runs, and **no
Full CI run**. Thus `rust-feature-modes` is configured to enforce the bridge
check, but has no hosted validation result for this head. Full CI admission,
fresh reviews and technical reapproval remain with the controller.

The Repository Policy failure was independently checked in both hosted logs:

| Source | Exact source SHA | Run and failing job |
| --- | --- | --- |
| PR head | `8f3a62e0fde729cf74db0b6b3551608338b367dd` | [Fast CI 44420298369](https://github.com/ai-dynamo/aisim/actions/runs/44420298369), [Repository Policy 135211174950](https://github.com/ai-dynamo/aisim/actions/runs/44420298369/job/135211174950) |
| main | `0809faae9e543487dcf2142dc6afe2427a42f9bd` | [Fast CI 44409204366](https://github.com/ai-dynamo/aisim/actions/runs/44409204366), [Repository Policy 135172502498](https://github.com/ai-dynamo/aisim/actions/runs/44409204366/job/135172502498) |

Both fail
`tests/test_artifactory_wheel_handoff.py::test_wheel_handoffs_require_internal_containerized_runners`
at line 371 with `AttributeError: 'str' object has no attribute 'get'`.
The test calls `job.get("container", {}).get("image")`, while the Dynamo job
in `.github/workflows/readme-checks.yml` uses the valid string container form
`${{ vars.CI_JOB_CONTAINER_IMAGE }}`. Both that workflow and the failing test
are byte-unchanged between base `0809faae9` and reviewed head `8f3a62e0`.
Fast CI's aggregate fails as a consequence. This is a verified pre-existing
failure, not a passing check or a transient-infrastructure claim; its repair
is outside the two selected findings.

## Inputs and provenance

`check_is_moe()` classifies `GEMMA4MIX` from the parsed `num_experts`/`topk`
pair. The bundled checkpoints pin that pair:

| Bundled checkpoint | Git blob (`git hash-object`) | `text_config.num_experts` | `text_config.top_k_experts` | Classification |
| --- | --- | ---: | ---: | --- |
| `nvidia--Gemma-4-31B-IT-NVFP4_config.json` | `b8a5581721139016753fb549f1d7083393202348` | `null` | `null` | dense |
| `nvidia--Gemma-4-26B-A4B-NVFP4_config.json` | `1247fb379f4ac91070ebdc083389c6d005b72ccb` | `128` | `8` | routed |
| `google--gemma-4-26B-A4B_config.json` | `29512a0d8108fc06ea4cec3cf2be250cfe0b960a` | `128` | `8` | routed |

All three declare `architectures=["Gemma4ForConditionalGeneration"]` and
`model_type="gemma4"`, so the architecture alone cannot separate dense from
routed. The configs also carry `text_config.enable_moe_block` (`false` dense,
`true` routed), but no Python or Rust source reads that key; it appears only in
a `gemma4.py` docstring. The parsed `num_experts`/`top_k_experts` fields that
`check_is_moe()` uses are therefore the only dense/routed signal the model build
consumes.

The blob hashes are identical on `origin/main` (`f85518b36` and the current base
`0809faae9`); this branch does not modify the bundled configs.

## Reproduction

Shape enumeration, from the repository root with the built native extension:

```python
from aisim.sweeper import model_hw

for budget in (1, 2, 4, 8):
    model_hw.parallel_configs_for(
        "nvidia/Gemma-4-31B-IT-NVFP4",
        "gb300",
        gpu_budget=budget,
        deployment_mode="agg",
        backend="vllm",
        max_seq_len=2048 + 128,
    )
```

The "before" column was produced by the same snippet in a clean
`git worktree add --detach /tmp/aisim-origin-main origin/main` checkout with
`PYTHONPATH=/tmp/aisim-origin-main/python/aisim/src`, not by overriding
`check_is_moe` in this tree.

The public recommendation path is covered by
`test_dense_gemma4_public_recommendation_is_expert_parallel_free` in
`python/aisim/tests/unit/test_modern_recommend_context.py`. That test fails
on `origin/main` with the `ValueError` below and passes on this branch.

## Shape enumeration

`gb300`, `vllm`, `deployment_mode="agg"`, `max_seq_len=2176`:

| GPU budget | `origin/main` `f85518b36` | This branch |
| ---: | --- | --- |
| 1 | 1 shape: `tp1/dp1/moe_tp1/moe_ep1` | 1 shape: `tp1/dp1/moe_tp1/moe_ep1` |
| 2 | `ValueError: ... dense Gemma 4 variants require moe_ep_size=1, got 2` | 3 shapes, all `moe_tp=1, moe_ep=1` |
| 4 | same `ValueError` | 7 shapes, all `moe_tp=1, moe_ep=1` |
| 8 | same `ValueError` | 15 shapes, all `moe_tp=1, moe_ep=1` |

The branch shapes are pure TP/DP/replica combinations (`tp{1,2,4,8}/dp1`); no
expert-parallel dimension is offered for the dense checkpoint.

Routed control, `nvidia/Gemma-4-26B-A4B-NVFP4`, is unchanged by the fix:

| GPU budget | `origin/main` | This branch |
| ---: | ---: | ---: |
| 2 | 6 shapes | 6 shapes (identical set) |
| 4 | 16 shapes | 16 shapes (identical set) |
| 8 | 36 shapes | 36 shapes (identical set) |

## Public recommendation path

`run_recommendation(..., stack="engine")` with offline doubles,
`max_candidate_gpus=8`, `max_trials=8`, `parallelism=1`, `concurrency=1`,
2048 input / 128 output tokens, `gb300`/`vllm`/aggregated:

| Checkpoint | `origin/main` `f85518b36` | This branch |
| --- | --- | --- |
| `nvidia/Gemma-4-31B-IT-NVFP4` | `ValueError: ... dense Gemma 4 variants require moe_ep_size=1, got 2` | 8 evaluated, 8 feasible, 8 selected, `moe` dims `{(1, 1)}` |
| `nvidia/Gemma-4-26B-A4B-NVFP4` | 8 evaluated, 8 feasible, 8 selected, `moe` dims `{(1, 1), (1, 2), (1, 4), (1, 8), (2, 1), (4, 1)}` | identical |

The dense checkpoint no longer aborts the whole recommendation, and the routed
checkpoint keeps every expert-parallel candidate it had before.

## Generator producer path

`enumerate_profiling_configs()` is the other consumer of `check_is_moe()`: the
result decides the wideEP gate, the multi-node memory floor, and
`build_disagg_parallel_lists(is_moe=...)`. The probe replaces the system-config,
weight, hardware-support and DGD-rendering lookups with fixed stubs (8 GPUs per
node, 180 GiB per GPU, 1.5x memory multiplier, 60 GiB dense / 800 GiB routed
weight so the routed case crosses the wideEP threshold) and calls it for
`gb300`/`vllm`:

| Checkpoint | `origin/main` `f85518b36` | This branch |
| --- | --- | --- |
| `nvidia/Gemma-4-31B-IT-NVFP4` | 6 prefill + 6 decode candidates, all 6 shapes expert-parallel (`moe_ep` 2/4/8 at `tp` 1/2/4/8), `is_moe=True`, wideEP off | 4 prefill + 4 decode candidates, `moe_tp=moe_ep=1`, `is_moe=False`, wideEP off |
| `nvidia/Gemma-4-26B-A4B-NVFP4` | 4 + 4 candidates, 4 expert-parallel shapes, `is_moe=True`, wideEP auto-enabled | identical |

The dense producer candidates were shapes a dense checkpoint cannot serve; the
routed ladder is unchanged.

## Cross-layer consumers of the classification

`check_is_moe()` also gates the search-space, CLI, and capacity paths. Each row
is pinned by a test rather than argued from reachability:

| Consumer | Dense Gemma 4 behavior | Test |
| --- | --- | --- |
| `generator/enumerate.py::enumerate_profiling_configs` | wideEP gate and multi-node floor stay off, and `build_disagg_parallel_lists(is_moe=False)` yields only pure TP/DP candidates | `test_dense_gemma4_producer_configs_are_expert_parallel_free`, `test_routed_gemma4_producer_configs_keep_expert_parallel_shapes` |
| `generator/request/sweeper.py::_resolve_model_facts` | `ModelFacts.is_moe` is `False`, so the rule engine never selects MoE fields | `test_generator_model_facts_follow_the_classification` |
| `sdk/task_v2.py::Task._is_moe` → `attention_dp_size` in `run_single_agg` / `run_single_disagg` | a requested `dp` never reaches the attention axis (`attention_dp_size=1`); routed keeps `dp>1` on both roles | `test_run_single_agg_attention_dp_follows_the_gemma4_classification`, `test_run_single_disagg_attention_dp_follows_the_gemma4_classification` |
| `legacy_cli/api.py::_resolve_moe_parallelism` | MoE widths pass through untouched instead of being resolved | `test_dense_gemma4_moe_parallelism_is_passed_through_untouched` |
| `legacy_cli/api.py` static and disagg `check_is_moe` gates | neither role calls `resolve_model_config_moe_comm`; routed static and disagg still resolve every required phase | `test_dense_gemma4_static_estimate_skips_moe_comm_resolution`, `test_dense_gemma4_disagg_estimate_skips_moe_comm_resolution`, `test_routed_gemma4_static_estimate_still_resolves_moe_comm`, `test_disagg_resolves_each_role_moe_comm_before_session_run` |
| `sdk/moe_comm_resolver.py::resolve_model_config_moe_comm` | returns `None` before any database access | `test_dense_gemma4_never_resolves_moe_communication` |
| `sdk/memory/kv_cache.py` preflight | classification and the dense `moe_ep_size>1` rule run before the perf-DB lookup, so neither corrupt metadata nor unsupported expert parallelism can reach the naive fallback | `test_estimate_kv_cache_rejects_corrupt_gemma4_metadata_before_the_database`, `test_estimate_kv_cache_rejects_dense_gemma4_moe_ep_before_the_database` |
| `sdk/models/blocks/moe.py::MoEBlockShape.from_model_info` | reached only behind `_is_moe` or the resolver's early return, and its exception is tolerated | covered by the `task_v2` and resolver tests above |

The `gemma4.py` dense guard now raises `InvalidEngineConfigurationError` instead
of asserting, so `python -O` cannot disable it and the Rust bridge treats an
explicit dense `moe_ep_size>1` as non-fallback-safe. The diagnostic is built by
the shared `dense_gemma4_moe_ep_error` helper, which the KV-cache preflight also
uses before it resolves the perf database. The support-matrix classifier matches
it by the same anchored diagnostic
(`tools/support_matrix/fpe_support_matrix.py`).

The `estimate_kv_cache` re-raise is deliberately request-wide rather than
Gemma-specific. It sits in the generic native-build `except` and re-raises every
`InvalidEngineConfigurationError`, because pricing an invalid request naively
would return a capacity for a configuration the native path just rejected.
Missing performance data still takes the documented fallback. The following
before/after was author-reported at `a584153b9` on a non-Gemma MoE model
(`deepseek-ai/DeepSeek-V3`,
`h200_sxm`/`vllm`, `tp_size=3`, `moe_tp_size=1`, `moe_ep_size=2`, 800 GiB
override, `allow_naive_fallback=True`): `origin/main` `0809faae9` returns
`source="naive_fallback"` with `total_kv_size_tokens=1840487`; this branch
raises `InvalidEngineConfigurationError: Parallelism width mismatch ...`.
Pinned by
`test_estimate_kv_cache_does_not_naive_fallback_on_a_non_gemma_configuration_error`.

## Regression tests added

| Test | Contract |
| --- | --- |
| `test_dense_gemma4_producer_configs_are_expert_parallel_free` | Generator enumeration: non-empty prefill/decode producer candidates, every candidate `moe_tp=moe_ep=1`, `is_moe=False`, wideEP never enabled |
| `test_routed_gemma4_producer_configs_keep_expert_parallel_shapes` | Generator enumeration keeps the MoE branch, the wideEP auto-enable, and `moe_ep>1` candidates for the routed checkpoint |
| `test_generator_model_facts_follow_the_classification` | `ModelFacts.is_moe` (generator rule-engine input) is `False` dense / `True` routed |
| `test_dense_gemma4_never_resolves_moe_communication` | `resolve_model_config_moe_comm` returns `None` for dense Gemma 4 before it touches the database |
| `test_dense_gemma4_public_recommendation_is_expert_parallel_free` | Final consumer: at least one feasible candidate, non-empty selection, every candidate `moe_tp=1`/`moe_ep=1`, no dense-rejection reason |
| `test_dense_gemma4_rejects_moe_controls_before_database_or_capacity` | `compile_engine`, `KVCacheEstimator.from_request`, and `estimate_num_gpu_blocks` reject `moe_backend`/`enable_eplb`/`wideep_num_slots`/`moe_kernel_source` before database or capacity fallback |
| `test_routed_gemma4_still_accepts_the_same_moe_controls` | The same four controls stay valid for the routed checkpoint |
| `test_estimate_kv_cache_never_falls_back_after_corrupt_checkpoint_metadata` | `allow_naive_fallback=True` cannot turn corrupt expert metadata into a naive capacity |
| `test_estimate_kv_cache_rejects_corrupt_gemma4_metadata_before_the_database` | Classification precedes the perf-DB lookup: corrupt metadata raises even when the database is unavailable and the fallback is allowed |
| `test_estimate_kv_cache_keeps_the_naive_fallback_for_a_valid_dense_gemma4_checkpoint` | A well-formed dense checkpoint with a missing database still takes the documented naive fallback |
| `test_estimate_kv_cache_rejects_dense_gemma4_moe_ep_before_the_database` | A dense checkpoint with `moe_ep_size=2` and `allow_naive_fallback=True` is rejected before the perf-DB lookup and never priced by the naive estimator |
| `test_estimate_kv_cache_does_not_naive_fallback_on_a_non_gemma_configuration_error` | Outside Gemma 4: a `deepseek-ai/DeepSeek-V3` width mismatch with `allow_naive_fallback=True` raises instead of returning a `source="naive_fallback"` capacity |
| `test_dense_gemma4_static_estimate_skips_moe_comm_resolution` | Static CLI gate: no `resolve_model_config_moe_comm` call for dense Gemma 4 |
| `test_dense_gemma4_disagg_estimate_skips_moe_comm_resolution` | Disagg CLI gate: neither role resolves MoE communication for dense Gemma 4 |
| `test_routed_gemma4_static_estimate_still_resolves_moe_comm` | Routed Gemma 4 keeps the static-path resolution for both required phases |
| `test_dense_gemma4_moe_parallelism_is_passed_through_untouched` | `_resolve_moe_parallelism` leaves dense MoE widths untouched and still resolves routed ones |
| `test_dense_variant_rejects_moe_ep_gt_1` | The dense guard raises `InvalidEngineConfigurationError` (not `AssertionError`) for an explicit `moe_ep_size>1`, so `python -O` cannot disable it and the Rust bridge keeps the rejection non-fallback-safe |
| `test_run_single_agg_attention_dp_follows_the_gemma4_classification` | Agg `run_single_agg` forces `attention_dp_size=1` for dense and preserves `dp` for routed |
| `test_run_single_disagg_attention_dp_follows_the_gemma4_classification` | Both disagg roles apply the same dense/routed attention-DP rule |
| `test_gemma4_rejects_inconsistent_expert_config` / `test_gemma4_moe_detection_follows_expert_config` in `tests/unit/sdk/models/test_model_config.py` | `(128, 0)`, `(0, 8)`, `(None, 8)`, `(128, None)`, `(128, 129)`, `(128, -8)` and `(-128, 8)` raise `InvalidEngineConfigurationError`; `(None, 0)`, `(0, 0)`, and `(128, 8)` classify as dense/dense/routed |
| Non-integer cases in `test_gemma4_rejects_inconsistent_expert_config` | Boolean pairs `(True, 1)`, `(128, True)`, `(False, 0)`, `(None, False)`; fractional pairs `(128.5, 8)`, `(128, 8.5)`; integer-valued floats `(128.0, 8)`, `(128, 8.0)`, `(0.0, 0)`, `(None, 0.0)`; and strings `("128", 8)`, `(128, "8")` all raise `InvalidEngineConfigurationError` at `8f3a62e0` |
| `test_malformed_local_gemma4_expert_types_cannot_fall_back` in `tests/unit/sdk/test_memory_estimation.py` | Boolean `True`/`False`, fractional `128.5`/`8.5`, integer-valued floats `128.0`/`8.0`/`0.0`, and strings `"128"`/`"8"` are tested for both fields through the real local-config parser, `estimate_kv_cache(allow_naive_fallback=True)` and `best_available(auto, allow)`, on both `gb300` and an unavailable system (48 cases at `8f3a62e0`) |
| `test_no_current_cli_matrix_row_records_the_dense_gemma4_rejection` | Generated-artifact guard: no CLI support-matrix row the generator still enumerates records the dense rejection for a checkpoint the classifier calls dense |
| `test_the_current_cli_matrix_still_covers_a_dense_gemma4_checkpoint` | The same guard cannot pass vacuously: a currently enumerated dense Gemma 4 row must exist |
| `test_dense_gemma4_coverage_check_ignores_other_dense_families` | Dropping every Gemma 4 row fails the coverage assertion even though other dense families remain in the matrix |
| `dense_gemma4_expert_parallelism_is_not_fallback_safe` (Rust) | The bridge's `InvalidEngineConfig` mapping is not fallback-safe, so `best_available` surfaces the dense rejection instead of using the regression estimator |

## Support-matrix artifacts

The bundled support matrices recorded the pre-fix rejection
(`dense Gemma 4 variants require moe_ep_size=1`) for
`nvidia/Gemma-4-31B-IT-NVFP4`. The CLI matrix is reproducible on any host, so
the affected rows were regenerated in this branch; the op-level FPE matrix is a
campaign artifact and is left as an explicit blocking follow-up.

### CLI matrix: regenerated here

Twelve rows moved from `FAIL` (a `sweep_agg` / `_get_disagg_worker_candidates`
traceback ending in the dense rejection) to `HYBRID_PASS`:

| File | Rows |
| --- | ---: |
| `systems/support_matrix/b200_sxm.csv` | 2 |
| `systems/support_matrix/b300_sxm.csv` | 2 |
| `systems/support_matrix/gb200.csv` | 2 |
| `systems/support_matrix/gb300.csv` | 2 |
| `systems/support_matrix/rtx_pro_6000_server.csv` | 4 |

They were produced with the repository's own generator, one invocation per
`(system, backend, version)`, and only the previously failing rows were
replaced with the generated rows:

```bash
python python/aisim/tools/support_matrix/generate_support_matrix.py \
  --model nvidia/Gemma-4-31B-IT-NVFP4 --system gb300 \
  --backend trtllm --backend-version 1.3.0rc20 --output <tmp-dir>
```

Each run reports 2/2 `Hybrid estimable` (`HYBRID_PASS`, source
`hybrid:xshape`): SILICON is still missing perf data for that backend version,
so the documented hybrid retry answers, and the dense rejection is gone.

Ten `FAIL` rows remain, all for `trtllm 1.3.0rc23` on the same five systems.
That backend version is retired: `SupportMatrix().generate_combinations()` now
enumerates `1.3.0rc20` and `1.3.0rc29` for `trtllm`, so the generator cannot
produce those rows. They are historical records in a cumulative matrix, not a
statement about the current classifier.

### Op-level FPE matrix: blocking follow-up

103 `FAIL` rows across ten `systems/fpe_support_matrix/*.csv` files still carry
the dense rejection text (counts are per file): `a100_sxm` 7, `b200_sxm` 9,
`b300_sxm` 8, `b60` 2, `gb200` 10, `gb300` 9, `h100_sxm` 16, `h200_sxm` 17,
`l40s` 13, `rtx_pro_6000_server` 12.

These rows are **stale and must not be read as current support**, but they
cannot be refreshed from a single row, system, or PR:
`build_fpe_support_matrix.py` rejects shards whose `source_sha` or workload
identity differ, so the split CSVs come from one campaign that probes every
model on every system with a single source identity
([`docs/ci/accuracy.md`](../../docs/ci/accuracy.md#fpe-support-qualification)).
A per-model shard would also change the per-shard probe and topology counts that
sibling rows in the same file carry, so patching the affected rows alone would
produce an internally inconsistent artifact.

Blocking follow-up: rerun the FPE campaign on the fixed sources, rebuild the
split CSVs, and confirm no Gemma 4 row keeps the dense rejection. Rows that
still fail for a real reason (`PERF_DATA_MISSING`, `SDK_UNREPRESENTABLE`,
`BUILD_FAILED`) stay failed with their own cause.

### Remaining artifact decision (2026-11-18)

Inspection at `8f3a62e0` confirms all 103 FPE rows and ten retired-version
CLI rows above still record `FAIL`. Neither set was regenerated by the local
verification above. The author's
[retirement-contract objection](https://github.com/ai-dynamo/aisim/pull/783#issuecomment-6993101186)
and [follow-up](https://github.com/ai-dynamo/aisim/pull/783#issuecomment-6994253205)
remain unresolved: there are 212 failed `trtllm 1.3.0rc23` rows across all
models, including ten for the routed Gemma 4 checkpoint too.
`VALID_STATUSES` in `tools/support_matrix/support_matrix.py`
has no retired/non-current status, and `scripts/pages/prepare_fpe_pages.py`
rejects unknown FPE statuses. Introducing one changes the artifact/consumer
contract; the author explicitly requests a separate, consistent lifecycle
change rather than inventing a Gemma-only status.

A maintainer must settle that disputed contract (including treatment of the
other retired-version rows), or supply a complete same-source FPE campaign
and an agreed retirement representation. The FPE guard must then land with
the regenerated or explicitly non-current rows. Adding it against these
unchanged artifacts would fail; an allowlist or skipped assertion would hide
the remaining requirement. The Cross-Layer Contract finding stays open.

### Guard

Three checks keep the classification and the rejection from regressing.

1. `tests/unit/support_matrix/test_support_matrix_dense_gemma4.py` fails when a
   row the CLI generator still enumerates records the dense rejection for a
   checkpoint the classifier calls dense, and also fails if the dense checkpoint
   drops out of the matrix entirely (so the first check cannot pass vacuously).
   It reports the ten rows above before the regeneration and passes after it.

2. `dense_gemma4_expert_parallelism_is_not_fallback_safe` in
   `crates/core/src/perfmodel/fpm/model.rs` pins that `best_available` does not
   treat the dense rejection as fallback-safe. It builds the `AicError`
   directly, so it covers the predicate but never crosses the Python boundary.

3. `crates/core/tests/perfmodel/dense_gemma4_bridge.rs` (integration test
   `perfmodel_dense_gemma4_bridge`) drives the rejection through the embedded
   interpreter, where `py.rs` maps the SDK's
   `InvalidEngineConfigurationError` to `AicError::InvalidEngineConfig`:
   - `dense_gemma4_rejection_survives_the_bridge_with_fallback_allowed` sets
     `ForwardPassFallbackPolicy::Allow` with `moe_ep_size=2` and requires
     `best_available` to return `AicError::InvalidEngineConfig` carrying both
     `compile_engine:` and the dense diagnostic. That pins the real mapping, not
     a Rust-side construction, and shows the policy does not degrade to the
     regression estimator.
   - `dense_gemma4_capacity_rejects_instead_of_estimating_naively` calls
     `estimate_kv_cache` with `allow_naive_fallback: true` and a capacity
     override, and requires the dense diagnostic instead of a naive estimate
     built from the same checkpoint metadata.
   - `dense_gemma4_num_gpu_blocks_rejects_instead_of_returning_a_count` calls
     `aisim_core.sdk.memory.estimate_num_gpu_blocks` with the same request. That
     is the Python entry point behind `python.rs::estimate_aic_num_gpu_blocks`,
     which `materialize_aic_capacity` installs as the replay path's capacity
     estimator from `resolve_role_timing`; it must raise the dense diagnostic
     instead of returning a scheduler block count.
   - `routed_gemma4_is_not_rejected_by_the_dense_guard` is the control: the
     routed checkpoint keeps `moe_ep_size=2` and must never produce the dense
     diagnostic.

   Each test prints a visible `SKIP` and returns when the SDK is not
   importable, and hard-fails instead when `AIC_REQUIRE_DENSE_GEMMA4_BRIDGE=1`
   is set, so a bare `cargo test` cannot read as coverage. The
   `rust-feature-modes` job is configured to set that variable in its embedded
   test step, so an SDK import failure would fail that step when the job is
   admitted and executed. This is a workflow contract, not hosted test evidence.
   A local enforced run
   needs the maturin-built extension, the SDK sources on `PYTHONPATH`, and
   `PYO3_PYTHON` pointing at the same interpreter the extension was built for;
   the module header records the exact command.

   The capacity cases assert the message and its bridge context, not an error
   variant: `memory.rs::estimate_kv_cache` maps every Python failure to
   `KvCacheEstimateError::Unsupported` and
   `python.rs::estimate_aic_num_gpu_blocks` wraps the `PyErr` in `anyhow`, so
   those variants cannot separate a dense rejection from an unrepresentable
   model. No caller branches on the difference today (`allow_naive_fallback` is
   decided in Python), so a typed mapping is left as a separate change.

## Limitations

- Only `gb300`/`vllm` `0.30.0` and the two bundled checkpoints above were
  exercised. Shape availability on other databases is not claimed.
- Generator producer numbers use fixed stubs for the system config, model weight
  and DGD rendering. They pin the branch taken and the shape sets, not absolute
  candidate counts for a real cluster.
- Recommendation scores come from constant offline metrics. This document makes
  no accuracy claim, and it does not compare against a GPU serving run.
- Historical local measurements used a release native extension built with
  `cargo build -p aisim-core --release --features extension-module`. The
  2026-11-18 verification uses the CI release wheel configuration, including
  `jemalloc`; it does not measure serving performance.
