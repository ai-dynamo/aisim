<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mamba mixed-step accounting

These are simulator predictions for the modeled `b200_sxm` platform, for
Nemotron-3-Nano-30B-A3B BF16 at TP1. They are accounting diagnostics. They are
not silicon measurements, calibration or accuracy claims.

## Setup

| Variant | Commit | Content |
| --- | --- | --- |
| Candidate | `686275e7350e4b4a39e88dec55c135312b671038` | The fix. |
| Baseline | `49d93f5a4aec1ec6033ae63228548348167b4e12` (not published, rebuilt by the recipe below) | The Candidate with `session.rs` and the mixed-step hunks of `runtime.rs` reverted. |
| Main | `bf373a132b6ce72e9fdd2a5d1b5124cefb84ed0c` | Unqualified control only. |

Baseline and Candidate share all qualification code. The accounting code is the
only difference. The Candidate is fetchable from the PR fork. To rebuild the
Baseline, take the Main versions of the two accounting files and add back the
`nextn + 1` width check:

```bash
git fetch https://github.com/milesial/aisimulate-public.git 686275e7350e4b4a39e88dec55c135312b671038
git fetch https://github.com/ai-dynamo/aisim.git bf373a132b6ce72e9fdd2a5d1b5124cefb84ed0c
git cat-file -e 686275e7350e4b4a39e88dec55c135312b671038^{commit}  # stop if either is missing
git cat-file -e bf373a132b6ce72e9fdd2a5d1b5124cefb84ed0c^{commit}
git checkout 686275e7350e4b4a39e88dec55c135312b671038
git checkout bf373a132b6ce72e9fdd2a5d1b5124cefb84ed0c -- \
  crates/core/src/perfmodel/session.rs crates/core/src/perfmodel/engine/runtime.rs
git apply baseline-width-check.patch   # the patch below
git hash-object crates/core/src/perfmodel/session.rs        # 58952aa6
git hash-object crates/core/src/perfmodel/engine/runtime.rs  # 28d8d759
```

`baseline-width-check.patch` against `bf373a13`'s `runtime.rs`:

```diff
--- a/crates/core/src/perfmodel/engine/runtime.rs
+++ b/crates/core/src/perfmodel/engine/runtime.rs
@@ -495,6 +495,21 @@ impl Engine {
             .as_ref()
             .and_then(|s| s.nextn)
             .unwrap_or(0);
+        // Generation Mamba queries carry `requests * (nextn + 1)` flattened
+        // tokens and the op divides by its own width; the two must agree.
+        for op in &spec.generation_ops {
+            if let Op::Mamba2(m) = op
+                && op.is_qualified_mamba()
+                && m.token_width != nextn.saturating_add(1)
+            {
+                return Err(AicError::InvalidEngineConfig(format!(
+                    "qualified Mamba op {} has token_width {} but speculative nextn+1 is {}",
+                    m.name,
+                    m.token_width,
+                    nextn.saturating_add(1)
+                )));
+            }
+        }
         super::readiness::validate_glm53_kv(
             &db,
             spec.context_ops.iter().chain(&spec.generation_ops),
@@ -6276,6 +6291,8 @@ pub(crate) mod tests {
             head_dim: 64,
             n_groups: 8,
             chunk_size: 256,
+            state_dtype: "unrecorded".into(),
+            token_width: 1,
         })];
         let ops_json = serde_json::to_string(&ops).unwrap();
         let err = engine
```

No built-in model sets `state_dtype` or `token_width` yet. The original
Baseline-to-Candidate runs (`agg.py`, `phases.py` and the historical telemetry
table) applied this `float16` wrapper (`qual_patch.py`) before they built the
model:

```python
def apply(nextn):
    """Same wrapper in every variant: record the recurrent-state dtype and the decode width."""
    import aisimulate_core.sdk.models.nemotron_h as nh
    real = nh.ops.Mamba2Kernel
    def wrapped(name, scale, kernel, phase, *args, **kw):
        kw.setdefault("state_dtype", "float16")
        kw.setdefault("token_width", (nextn + 1) if phase == "generation" else 1)
        return real(name, scale, kernel, phase, *args, **kw)
    class _Shim:
        def __init__(self): self._o = nh.ops
        def __getattr__(self, n): return wrapped if n == "Mamba2Kernel" else getattr(self._o, n)
    nh.ops = _Shim()
```

The later comparisons use their own scripts, `telem.py` and `agg_caller.py`,
which give each qualified op a `bfloat16` state.

## Operation sources

| Data | Context conv1d | Context scan | Generation |
| --- | --- | --- | --- |
| TensorRT-LLM `1.3.0rc20` | Measured table, exact shape key | Analytical bound | Analytical bound |
| vLLM `0.24.0` | Analytical bound | Analytical bound | Analytical bound |

`fallback_policy: deny` does not prove measured Mamba coverage, because the
analytical bound is an allowed result.

## Results

Qualified Baseline to qualified Candidate.

`InferenceSession.run_agg`, TensorRT-LLM. The script below (`agg.py`) ran
each case at each commit, from the commit's `python/aisimulate` package
environment, with `qual_patch.py` beside it. The commands were
`python agg.py OUT0.json --qual 1 --nextn 0` and
`python agg.py OUT3.json --qual 1 --nextn 3`. The
measured runs reused one session for all cases. This version builds a fresh
session per case, because `run_agg` caches results by a key that omits
`prefix`:

```python
import argparse, json, os, sys
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--qual", type=int, default=0)
ap.add_argument("--nextn", type=int, default=0); ap.add_argument("--backend", default="trtllm")
ap.add_argument("--version", default="1.3.0rc20")
ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
a = ap.parse_args()
if a.qual:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import qual_patch; qual_patch.apply(a.nextn)
from aisimulate_core.sdk import config
from aisimulate_core.sdk.models import get_model
from aisimulate_core.sdk.perf_database import get_database
from aisimulate_core.sdk.backends.factory import get_backend
from aisimulate.sdk.inference_session import InferenceSession
mc = config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, nextn=a.nextn,
                        nextn_accept_rates=([0.8, 0.7, 0.6][:a.nextn] if a.nextn else None))
out = {}
# (batch, isl, osl, prefix, ctx_tokens): ctx_tokens is the prefill budget per step.
for name, (b, isl, osl, px, ctx) in {
    "chunked_uncached": (16, 6000, 64, 0, 2048),
    "chunked_partial_cached": (16, 6000, 64, 3000, 2048),
    "split_budget_cached": (16, 4096, 64, 1024, 8128),
}.items():
    sess = InferenceSession(get_model(a.model, mc, a.backend), get_database("b200_sxm", a.backend, a.version),
                            get_backend(a.backend, "b200_sxm"))
    kw = {"ctx_tokens": ctx}
    if a.nextn: kw["decode_tokens_per_iteration"] = 1 + 0.8 + 0.8 * 0.7 + 0.8 * 0.7 * 0.6
    s = sess.run_agg(config.RuntimeConfig(batch_size=b, isl=isl, osl=osl, prefix=px), **kw)
    d = s.get_summary_df().iloc[0].to_dict()
    out[name] = {k: float(d[k]) for k in ("ttft", "tpot", "tokens/s/gpu")}
json.dump({"qual": a.qual, "nextn": a.nextn, "backend": a.backend, "runs": out}, open(a.out, "w"), indent=1)
```

`chunked` is `chunked_uncached` and `split budget` is `split_budget_cached`.
`--backend vllm --version 0.24.0` gives the vLLM rows.

| MTP | Case | TTFT ms | tok/s/GPU |
| --- | --- | --- | --- |
| 0 | chunked, 6000 input | 249.54 to 250.68 (+0.46%) | 930.2 to 926.4 (-0.41%) |
| 0 | split budget, 1024 cached | 100.69 to 101.24 (+0.54%) | 1370.2 to 1366.2 (-0.29%) |
| 3 | chunked, 6000 input | 251.95 to 253.57 (+0.64%) | 1021.5 to 1015.0 (-0.64%) |
| 3 | split budget, 1024 cached | 101.51 to 102.40 (+0.87%) | 1820.7 to 1809.4 (-0.62%) |

On vLLM, TTFT changes by +0.11% to +0.75% and tok/s/GPU by -0.12% to -1.28%.
The measured runs did not measure `chunked_partial_cached`. The shared session
returned the cached `chunked_uncached` summary for it, because the `run_agg`
cache key omits `prefix`. The per-phase result below covers a cached prefix.

Packed telemetry (`RustForwardPassPerfModel.estimate_forward_pass_time_ms`),
both backends:

| Shape | MTP0 | MTP3 |
| --- | --- | --- |
| Pure prefill | 0% | 0% |
| Pure decode, 8 requests | 0% | +2.03% to +2.14% |
| Mixed, 1x2048 + 8 decode | +0.25% to +0.27% | +0.68% to +0.75% |
| Mixed, 2 requests of 1500 tokens + 8 decode | +0.24% to +0.36% | +0.62% to +0.71% |

These percentages are transcribed observations from `686275e7` and its
Baseline. Their full `ForwardPassMetrics` payloads, decode KV totals included,
and their commands and raw outputs were not saved, so they cannot be
reproduced. `telem.py` below uses another revision pair, MTP3 and BF16 states,
so it does not reproduce them either. The reproducible checks of the MTP0
packed path are `qualified_mamba_mixed_steps_preserve_prefill_and_decode_shapes`,
which prices width-one packed mixed steps against exact bytes, and the
hand-derived oracle sections below.

The MTP3 column is historical and not valid for the current code.
`num_decode_requests` counts packed decode slots, not logical requests: 8
logical requests at width 4 fill 32 slots. Commit `686275e7` multiplied that
slot count by the width again before the Mamba query, so it priced 32
logical requests and 128 tokens. The current code divides the 32 slots by
the width once, which gives 8 requests and 32 tokens. The original qualified Baseline-to-Candidate comparison has no
corrected MTP3 delta. Exact-value tests check the current path instead, and the
width-only comparison below isolates the width correction. `qualified_mamba_mixed_steps_preserve_prefill_and_decode_shapes`
prices packed mixed steps, for example `num_prefill_requests: 1`,
`sum_prefill_tokens: 512`, `num_decode_requests: 8` at width four, against
their expected bytes. `op_level_replay_prices_qualified_mamba_once_per_request_and_by_phase`
compares replay's pure-decode and mixed iterations with the static decode
cost. The width correction does not apply to the MTP0 column, because its
decode width is one. Its mixed-shape deltas come from the per-phase accounting.
Both variants give every qualified op the same `float16` state. The operation
sources table lists every scan as an analytical bound for both data sets.
Therefore the state dtype does not change a source in either variant. The `run_agg`,
per-phase and replay results above and below do not use packed telemetry.

Width-only packed-telemetry comparison, TensorRT-LLM, MTP3, `HYBRID` mode.
This compares `35944ae4` (before: the width applied twice) with `c3ed1513`
(after). Both revisions already contain the per-phase accounting, so this is
not a rerun of the Baseline-to-Candidate comparison. It measures only the width
correction.
Both commits sit on main `bf1b539f`. Each ran
`python telem.py OUT.json` from its own `python/aisim` build, with the
packaged performance data of that build. The `systems` data trees of the
two commits are identical: `git diff 35944ae4 c3ed1513 --
python/aisim/src/aisim_core/systems` is empty. The script is
[`telem.py`](mamba-mixed-step-accounting/telem.py). It gives every Mamba2 op
a `bfloat16` state. Generation ops get `token_width` 4, and context ops get
`token_width` 1. Its inputs use the packed
decode convention: 8 requests at width 4 give `num_decode_requests: 32` and
`sum_decode_kv_tokens: 32800` (1025 tokens each). The mixed case adds
`num_prefill_requests: 1` and `sum_prefill_tokens: 2048`.

| Case | Before ms | After ms | Change |
| --- | --- | --- | --- |
| Pure decode, 8 requests | 9.459918 | 9.033769 | -0.426149 (-4.50%) |
| Mixed, 1x2048 + 8 requests | 21.171992 | 20.745843 | -0.426149 (-2.01%) |

The outputs record each Mamba row and one sum of all non-Mamba rows. Of the
recorded values, only the two generation Mamba rows change, both with source
`sol`. `generation_mamba_conv1d` falls from 0.029815 to 0.007454 ms, and
`generation_mamba_ssm` falls from 0.538383 to 0.134596 ms. Each is exactly
four times smaller. Their sum is the 0.426149 ms total change. The context
Mamba rows and the non-Mamba sum stay the same. The outputs do not show
whether each non-Mamba row stays the same. The raw outputs are
[`35944ae4.json`](mamba-mixed-step-accounting/35944ae4.json) and
[`c3ed1513.json`](mamba-mixed-step-accounting/c3ed1513.json). These are
simulated latencies. Host runtime was not measured.

### Current evidence against a hand-derived oracle

The values below come from the current code. Each changed value is checked
against bytes moved divided by memory bandwidth, computed by hand. The oracle
does not use the pricing code. These are simulated latencies, not prediction
accuracy or host runtime.

For Nemotron Nano, a qualified scan moves `(4096 + 2 * 8 * 128 + 64) * 2 +
4096 * 2 = 20,608` bytes per token. One BF16 state is `4096 * 128 * 2 = 1 MiB`.
The model has 23 Mamba layers, and `b200_sxm` has 7.7e12 B/s.

**Width-only telemetry rows.** At MTP3, 8 decode requests verify 4 tokens
each, so 32 tokens. Each request reads its state once and writes 4
checkpoints. One layer of `generation_mamba_ssm` therefore moves
`32 * 20,608 + 8 * 1 MiB * 5 = 42,602,496` bytes. AISim scales decode work by
`(nextn + L) / L = 55 / 52` for the MTP layer. The result is 0.134596 ms, the
`c3ed1513` value above. The convolution update moves
`32 * 6144 * 5 * 2 + 32 * 6144 * 2 = 2,359,296` bytes per layer, which gives
0.007454 ms. Before the fix, each term counted 32 requests instead of 8, so
both rows were exactly four times larger.

**`run_agg` caller.** `InferenceSession.run_agg` passes its per-step prefill
budget `ctx_tokens` to the mixed step. It does not pass a prefill request
count. In that contract, a step holds `ceil(ctx_tokens / isl)` requests, the
same count that context attention prices. Commit `668e98fd` split a budget
remainder longer than `isl - prefix` into extra requests. Commit `62293977`
prices one partial request, capped at `isl - prefix` new tokens.
These runs used the four-case version of `agg_caller.py`, published in commit
`6511a26d`. The checked-in [`agg_caller.py`](mamba-mixed-step-accounting/agg_caller.py)
adds later cases, so it does not reproduce these two outputs. Extract the
four-case driver, then run it from each commit's `python/aisim` build. It uses
`HYBRID` mode at MTP0 with BF16 states:

```bash
git show 6511a26d:benchmarks/evidence/mamba-mixed-step-accounting/agg_caller.py > agg_caller_4.py
python agg_caller_4.py OUT.json
```

The raw outputs are
[`agg-668e98fd.json`](mamba-mixed-step-accounting/agg-668e98fd.json) and
[`agg-62293977.json`](mamba-mixed-step-accounting/agg-62293977.json).

| Case (batch, isl, prefix, budget) | Requests and new tokens | `context_mamba_ssm` before ms | After ms | Oracle ms |
| --- | --- | --- | --- | --- |
| One request (1, 4096, 3000, 2048) | 2 and 2048 to 1 and 1096 | 0.138596 | 0.073730 | 0.073730 |
| Split budget (16, 4096, 1024, 8128) | 3 and 7104 to 2 and 6144 | 0.456089 | 0.390731 | 0.390731 |
| Two complete requests (16, 4096, 3000, 8192) | 2 and 2192, unchanged | 0.147460 | 0.147460 | 0.147460 |
| Chunked, uncached (16, 6000, 0, 2048) | 1 and 2048, unchanged | 0.132332 | 0.132332 | 0.132332 |

Each request reads and writes its state once, so a context layer moves
`tokens * 20,608 + requests * 2 MiB`. The context convolution reads a measured
table, so the oracle does not cover it. It changes with the same request
shapes, from 0.670842 to 0.348105 ms and from 1.493353 to 1.338048 ms. The
non-Mamba sum is identical in every case. TTFT changes from 79.355 to 77.844 ms
for one request and from 101.272 to 100.765 ms for the split budget. The other
cases do not change. The `run_agg` regression
`test_run_agg_prices_scheduled_prefill_requests_not_budget` checks the first and
third rows. It fails on `668e98fd` with 0.138596 ms.

**Native Replay.** The regression
`op_level_replay_prices_qualified_mamba_once_per_request_and_by_phase` runs the
op-level Replay timing path, with and without diagnostic capture. Its fixture
has one qualified scan per phase, with `n_groups` 4, so a token moves 18,560
bytes. A pure-decode iteration holds 2 requests at width 4, which Replay packs
as 8 decode requests. The decode moves `8 * 18,560 + 2 * 1 MiB * 5 =
10,634,240` bytes, 0.001381 ms at 7.7e12 B/s. A mixed iteration adds one
16-token prefill, `16 * 18,560 + 2 * 1 MiB = 2,394,112` bytes or 0.000311 ms,
for a total of 0.001692 ms. Replay must report the prefill bytes as prefill
evidence and the decode bytes as decode evidence. The test asserts each value
against these bytes. At `35944ae4`, the pure decode gave 0.005524 ms. That is
the oracle for 8 requests and 32 tokens, `42,536,960` bytes, so the width was
applied twice.

The embedded Replay test, the public API, engine-step parity (305), the golden
set (121) and the Python qualification tests (54 with the new regression) passed
at `62293977`. `telem.py` at `62293977` gave output identical to
`c3ed1513.json`. At `9a2bde6b`, `agg_caller.py` gave output identical to
`agg-62293977.json` for all four cases, the split budget included, and the same
suites passed (56 Python tests).

### `run_agg` caller corrections

These two fixes change the `run_agg` caller, not the Mamba formulas above. They
apply to every operation of every model.

- **One request under a large budget.** With one request, `run_agg` passed
  its whole prefill budget to the mixed step. A budget above one prompt then
  priced extra requests. Commit `19e8daf6` schedules at most one packed
  prompt, `min(ctx_tokens, isl)`.
- **Cache identity.** The `run_agg` result cache omitted the prefix. A second
  call on the same session with a different prefix returned the first result.
  Commit `19e8daf6` adds the prefix to the cache key.

`agg_caller.py` gained the large-budget case and a same-session prefix
sequence in commit `19e8daf6`, then the exact 4096-token control in commit
`df8a4277`. `caller-19e8daf6.json` comes from the `19e8daf6` driver. The
other `caller-*.json` outputs come from the `df8a4277` driver, which is the
checked-in version. To run an earlier driver, extract it with
`git show <commit>:benchmarks/evidence/mamba-mixed-step-accounting/agg_caller.py`.
The drivers ran from the `9a2bde6b`, `19e8daf6` and `70a8ea00` builds. The raw outputs are
[`caller-9a2bde6b.json`](mamba-mixed-step-accounting/caller-9a2bde6b.json),
[`caller-19e8daf6.json`](mamba-mixed-step-accounting/caller-19e8daf6.json) and
[`caller-70a8ea00.json`](mamba-mixed-step-accounting/caller-70a8ea00.json). The
`19e8daf6` run predates the control. `70a8ea00` gives the same values for
every case the two share.

| Case | Value | `9a2bde6b` | `19e8daf6` and `70a8ea00` | Oracle |
| --- | --- | --- | --- | --- |
| One request (1, 4096, 3000, 8192) | `context_mamba_ssm` ms | 0.147460 | 0.073730 | 0.073730 |
| One request (1, 4096, 3000, 8192) | Non-Mamba sum ms | 18.053611 | 13.180794 | 13.180794, the control |
| One request (1, 4096, 3000, 8192) | TTFT ms | 41.048 | 29.010 | 29.010, the control |
| Control (1, 4096, 3000, 4096) | `context_mamba_ssm`, non-Mamba sum, TTFT | 0.073730, 13.180794, 29.010 | Same | Unchanged by the fix |
| Same session, prefix 0, then 3000 | `context_mamba_ssm` ms | 0.132332, 0.132332 | 0.132332, 0.073730 | 0.132332, 0.073730 |
| Same session, prefix 0, then 3000 | TTFT ms | 79.852, 79.852 | 79.852, 77.844 | Second call equal to a fresh session |

Before the fix, the large budget priced two requests and 2192 new tokens, as
two complete prompts would. After it, the step prices one request and 1096
new tokens. The saved outputs then record the same Mamba rows, the same
non-Mamba sum and the same TTFT for the 8192- and 4096-token budgets. They do
not record each non-Mamba row. The regression
`test_run_agg_lone_request_ignores_unused_budget` compares the whole per-operation
mixed-step dictionary of the two budgets and finds every row equal. Among the Mamba rows, the context
convolution, both projection GEMMs and the norm also fall. Separately, the
recorded sum of the operations without `mamba` in their names falls by 27%,
from 18.053611 to 13.180794 ms. The saved outputs do not show which of those
operations change, or at which coordinates. In the
same-session sequence, the repeated third call returns the cached second
result. The four earlier cases give output identical to `agg-62293977.json`.

The regressions `test_run_agg_lone_request_ignores_unused_budget`,
`test_run_agg_cache_separates_prefixes` and the large-budget case of
`test_run_agg_prices_scheduled_prefill_requests_not_budget` fail on `9a2bde6b`
and pass on `19e8daf6`. At `19e8daf6`, 727 Python tests in the backend,
session, speculation, single-oracle and Mamba suites passed, with engine-step
parity (305) and the golden set (121).

Commit `70a8ea00` prices a qualified scan with a cached prefix with the SOL
estimate, because measured scan rows start from a cold state. The `1.3.0rc20`
data has no scan rows, so every scan above already used SOL. At `70a8ea00`,
`agg_caller.py` and `telem.py` reproduce the earlier outputs exactly, and 727
Python tests, the public API (28) and the embedded Replay tests (9) passed.

### Prefill step count and reuse scope

Two more caller fixes, in commits `55d2821b` and `35e632cb`, also apply to
every model:

- **Prefill steps in TTFT.** With one request, `run_agg` counted
  `ceil(isl / ctx_tokens)` prefill steps, the cached prefix included. If the
  uncached tokens fit the budget (`0 < isl - prefix <= ctx_tokens`), the step
  now schedules the whole request and TTFT counts one step. This changes
  every one-request result with a prefix, an uncached part that fits the
  budget and a budget below `isl`. Results without a prefix, with a budget of
  at least `isl`, or with more than one request do not change. Commit
  `59c4575a` also passes that one step to the TensorRT-LLM queuing factor
  (qf), so the result no longer depends on the budget. The qf formula and
  its coefficients do not change. This is not a general cached-prefix TTFT
  fix. If the uncached tokens exceed
  the budget, TTFT still counts `ceil(isl / ctx_tokens)` steps, the cached
  prefix included. The count is too high whenever
  `ceil(isl / ctx_tokens) > ceil((isl - prefix) / ctx_tokens)`, for one
  request or more. For one legacy Nano request with isl 8192, prefix 3000
  and a 2048-token budget, TTFT is `20.636 * 4 * 2.05 = 169.214` ms. With
  the step latency held fixed, the 3 chunks of uncached tokens and the
  three-step `qf = 2.00` give `20.636 * 3 * 2.00 = 123.815` ms. This remains
  unresolved.
- **Reuse scope.** The result cache key held neither the model graph nor the
  database mode, and the compiled-handle identity did not hold the Mamba
  qualification fields. A legacy and a qualified Nano graph on one backend,
  or over one handle cache, returned each other's TTFT. A database mode
  changed in place, from `HYBRID` to `SOL`, kept the `HYBRID` result. The
  result key now carries the compiled-engine identity. That identity lists
  the constructor arguments of each Mamba2 operation, legacy operations
  included. Its memo is reused only for the same database mode and transfer
  policy.

`agg_caller.py` ran from the `55d2821b` build and from `35e632cb`. Both runs
gave the same output,
[`caller-55d2821b.json`](mamba-mixed-step-accounting/caller-55d2821b.json).
At `59c4575a` it gave
[`caller-59c4575a.json`](mamba-mixed-step-accounting/caller-59c4575a.json).
Against `caller-70a8ea00.json`, only the cases below change:

| Case (batch, isl, prefix, budget) | Value | `70a8ea00` | `55d2821b` | `59c4575a` |
| --- | --- | --- | --- | --- |
| One request (1, 4096, 3000, 2048) | Non-Mamba sum ms | 16.855182 | 13.180794 | 13.180794 |
| One request (1, 4096, 3000, 2048) | TTFT ms | 77.844 | 29.773 | 29.010 |
| Same session, prefix 0, then 3000 | TTFT ms | 79.852, 77.844 | 79.852, 29.773 | 79.852, 29.010 |

The context-phase Mamba GEMMs and norm in that case also fall. The step now
holds the 4096-token request, not a 2048-token chunk with 2048 cached
tokens. `context_mamba_ssm` stays 0.073730 ms. TTFT is
`step * chunks * qf`, with `qf = 2 + (steps - 3) / 20`. The step is the
per-operation sum, 15.268 ms. At `55d2821b`, qf still read the budget steps,
`ceil(4096 / 2048) = 2`, so TTFT was `15.268 * 1 * 1.95 = 29.773` ms, one
chunk instead of two. At `59c4575a`, qf reads the one step the request
takes, so TTFT is `15.268 * 1 * 1.90 = 29.010` ms, the same as the
exact-prompt control (budget 4096). The uncached chunked case keeps two
chunks, `qf = 1.95` and 79.852 ms. A fully cached prompt
(prefix 4096) still fails with `isl must be greater than`.

The regressions ran against earlier builds:

| Regression | Earlier build | Failure |
| --- | --- | --- |
| `test_run_agg_ttft_counts_uncached_prefill_steps[3000-1]` | `9eaa66e8` | TTFT 77.844 ms, expected 38.922 ms from that build's step |
| `test_run_agg_reuses_results_only_for_the_same_graph_and_database_mode` | `9eaa66e8` | Second graph TTFT 78.669 ms, cold 77.868 ms |
| `test_run_agg_reuse_keeps_each_mamba_op_state_dtype` | `55d2821b` | Swapped dtypes TTFT 29.786 ms, cold 29.773 ms |
| `test_run_agg_reuse_keeps_the_partial_qualification_rejection` | `55d2821b` | Partially qualified graph did not raise |
| `test_run_agg_ttft_counts_uncached_prefill_steps[3000-1-1.9]` | `1ea0211c` | TTFT 29.773 ms, expected 29.010 ms |
| `test_run_agg_one_step_ttft_does_not_depend_on_the_budget` | `1ea0211c` | Prefix 3968: budget 128 TTFT 33.735 ms, budget 4096 18.579 ms |
| `test_run_agg_reuse_keeps_a_wrapped_mamba_op_distinct` | `1ea0211c` | Wrapped graph did not raise |
| `test_run_agg_reuse_keeps_a_nested_mamba_op_distinct` | `1db9b803` | Nested qualified scan did not raise after a warm cache |
| `test_run_agg_b1_fully_cached_fpm_prompt_reaches_the_prefix_check` | `1db9b803` | `MixedStepInput` rejected a zero budget |
| `test_compute_ttft_single_request_feeds_the_step_its_model_tokens[fpm-1096]` | `1db9b803` | XPU step got 4096 tokens, expected 1096 |

The uncached case `[0-2]` and `test_run_agg_rejects_a_fully_cached_prompt`
also passed on `9eaa66e8`. They pin behavior that did not change. At
`35e632cb`, the Python qualification, backend and session suites (734 tests)
passed.

### Review fixes after `35e632cb`

Commits `79a9e652` and `1ea0211c` add four fixes:

- **Whole-model FPM input.** An FPM reads the mixed-step budget as new
  tokens. With one request, `run_agg` passed the prompt with its prefix, so
  isl 4096 and prefix 3000 priced `ceil(4096 / 1096) = 4` requests. The FPM
  now receives at most `isl - prefix` new tokens, one request. Op-level
  pricing keeps the full prompt. No comparison used measured FPM data. The
  synthetic-anchor comparison below runs the compiled estimator. The
  regression `test_run_agg_b1_schedules_one_request_and_sizes_its_new_tokens`
  checks the scheduled input.
- **One-request activations.** The activation estimate now uses the new
  tokens of the step, `min(isl - prefix, ctx_tokens)`, as the memory
  helper's default does. TensorRT-LLM keeps its `max_num_tokens`
  reservation, so `agg_caller.py` and `telem.py` give the same output.
- **Kernel labels.** A qualified Mamba2 op must use a collected kernel
  label in its own phase. A misspelled label now fails at build time.
  Legacy ops keep any label.
- **Exact-total prefill.** `predict_prefill_latency_total` passes whether
  any token is cached, as the packed metrics path does. Two 16-token
  prefills with one cached token now skip the cold scan row.

Commit `108835ac` closes three gaps in the same paths. The XPU vLLM TTFT
hook now feeds a whole-model FPM at most `isl - prefix` new tokens per
request instead of the prompt with its prefix. Op-level pricing there is
unchanged. A fully cached one-request FPM prompt again reaches the engine's
prefix check instead of a zero step budget. The engine identity now reads
the compiled spec of each op that holds a Mamba2 op, so a qualified scan
nested in a composite fails cold and warm. At `108835ac`, the Python suites
(744 tests), engine-step parity (305), the golden set (121), the public API
(28) and the embedded Replay tests (9) passed. `agg_caller.py`, `telem.py`,
`mem.py` and `fpm_caller.py` give the same values as before.

Commit `59c4575a` adds two more checks. The compiled-engine identity
records a Mamba2 op's draft query width, which compiles the op inside a
`TokenScale` wrapper, so a wrapped qualified graph fails cold and warm
instead of reusing the plain graph's engine. A qualified prefill query with
no new token now fails instead of pricing state traffic.

Commits `71fcef7b` and `bf03c37d` add three more fixes:

- **Equal-length extends.** In the mixed step, a partial extend as long as
  one request's new tokens now joins the batch of complete requests. Two
  1096-token requests price as one `(2, 1096)` query, as packed telemetry
  prices them, not as two `(1, 1096)` queries. A shorter extend keeps its own
  query. SOL rows scale with the batch, so only measured rows change. In
  `agg_caller.py`, the split budget `(16, 4096, 1024, 8128)` holds two
  3072-token requests. Its measured TensorRT-LLM context convolution falls
  from 1.338048 to 1.169136 ms, and its TTFT from 100.765 to 100.376 ms. Its
  TPOT falls from 9.984 to 9.968 ms. No other case changes. The output is
  [`caller-71fcef7b.json`](mamba-mixed-step-accounting/caller-71fcef7b.json).
- **XPU FPM chunks.** With one vLLM XPU request, a whole-model FPM step
  smaller than the request prices the mean of its chunks. The TTFT hook now
  multiplies that mean by the chunk count, `ceil(new tokens / budget)`. For
  isl 4096, prefix 1096 and a 2048 budget, the step averages a 2048- and a
  952-token chunk. Commit `108835ac` scaled it by `3000 / 2048`. The
  regression `test_compute_ttft_single_request_counts_a_short_final_fpm_chunk`
  fails on `108835ac` with 61.59 ms instead of 83.0 ms.
- **Zero-work prefills.** With a qualified Mamba2 operation, a packed step
  with fewer new prefill tokens than prefill requests now fails in the mixed
  path too, as it already did in the pure-prefill path.

At `71fcef7b`, `mem.py`, `fpm_caller.py` and `telem.py` reproduce their
earlier values. At `bf03c37d`, `agg_caller.py` and `telem.py` give the same
output as at `71fcef7b`. These suites passed at both commits:

- Python suites: 955 tests.
- Engine-step parity: 305 tests. Golden set: 121 tests.
- Rust Mamba tests: 31. Public API: 28. Embedded Replay tests: 9.

[`mem.py`](mamba-mixed-step-accounting/mem.py) ran from the `35e632cb` and
`1ea0211c` builds. It prices one vLLM `0.24.0` Nano request with isl 4096
and osl 64. The script imports the SDK of the active environment, not of the
checkout. Run it with the interpreter of each build's own environment, for
example `BUILD_VENV/bin/python mem.py > mem-COMMIT.json`. Its `build` entry
records the SHA-256 of the imported `base_backend.py`, which holds the
activation estimate, and of the compiled `aisim._runtime` extension. The raw outputs are
[`mem-35e632cb.json`](mamba-mixed-step-accounting/mem-35e632cb.json) and
[`mem-1ea0211c.json`](mamba-mixed-step-accounting/mem-1ea0211c.json).
They come from the build-stamped script, rerun on each build. The values
match the first runs exactly, and the two `base_backend.py` hashes differ.
Values are in GiB:

| Prefix and budget | Activations before | After | Total before | After |
| --- | --- | --- | --- | --- |
| 3000, 2048 | 0.156250 | 0.083618 | 62.542492 | 62.469860 |
| 3000, 4096 | 0.312500 | 0.083618 | 62.698742 | 62.469860 |
| 3000, 8192 | 0.312500 | 0.083618 | 62.698742 | 62.469860 |
| 0, 2048 | 0.156250 | 0.156250 | 62.542492 | 62.542492 |

The activations scale with the tokens: `0.15625 * 1096 / 2048 = 0.083618`.
Weights, KV cache and the other terms do not change.

[`fpm_caller.py`](mamba-mixed-step-accounting/fpm_caller.py) checks the FPM
input through the compiled whole-model estimator. It builds a Qwen3-0.6B FPM
model and a synthetic vLLM table with the FPM unit-test helpers, with one
anchor at each coordinate a caller queries. `run_agg` compiles every
whole-model op with `interpolation: sol`, so this is the only FPM method
that it reaches. Direct interpolation needs an `fpm_profile`, which
`run_agg` does not pass. The script ran from the `35e632cb` and `1ea0211c`
builds as `python fpm_caller.py REPO_ROOT OUT.json`. Both runs used the
same table (SHA-256 prefix `2071e1799556`). The raw outputs are
[`fpm-35e632cb.json`](mamba-mixed-step-accounting/fpm-35e632cb.json) and
[`fpm-1ea0211c.json`](mamba-mixed-step-accounting/fpm-1ea0211c.json).
The FPM derives `ceil(context_tokens / (isl - prefix))` requests:

| Case (isl 512, budget 512) | Build | Caller input | Requests | Step ms | TTFT ms |
| --- | --- | --- | --- | --- | --- |
| Prefix 256 | `35e632cb` | 512 | 2 | 30.0, anchor (2, 512, 512) | 52.4 |
| Prefix 256 | `1ea0211c` | 256 | 1 | 12.0, anchor (1, 256, 256) | 34.4 |
| Prefix 0 | Both | 512 | 1 | 22.0, anchor (1, 512, 0) | 44.4 |

For one vLLM request, the queuing factor is 1 and the dispatch overhead is
`28 layers * 0.8 = 22.4` ms, so TTFT is the step plus 22.4 ms. The script
also prices both inputs directly in each build and gets the same anchors.
The values are synthetic. They show that the estimator receives one request
instead of two, not a latency on real hardware. The estimator comes from
the active environment, not from `REPO_ROOT`. So the script now also records
the SHA-256 of the compiled `aisim._runtime` extension that priced the run.
An earlier version hashed the `aisim_core._native` module instead, which is
a Python shim with the same bytes in every build. At
`108835ac` it reproduces the `1ea0211c` values,
[`fpm-108835ac.json`](mamba-mixed-step-accounting/fpm-108835ac.json). The deprecated hybrid FPM
path with draft ops takes the same input, and its draft ops use the same
request count, but no fixture with draft-op data exercises it here.

The input regression fails on
`35e632cb` for the 8-token budget, with 8 scheduled tokens instead of 6, and
passes on `1ea0211c`. At `1ea0211c`, `agg_caller.py` and `telem.py` give
output identical to `caller-55d2821b.json` and `c3ed1513.json`. The Python
suites (738 tests), the public API (28) and the embedded Replay tests (9)
passed, and `cargo test -p aisim-core --lib` passed locally (3269, 2 ignored).
The whole `-m unit` Python suite ran from a tarball of each build, which has
no Git metadata. At `1ea0211c`, its failures are a subset of the `9eaa66e8`
failures (286 against 287). The shared failures come from the tarball
environment, for example a collector source revision that needs Git.

The two parity suites ran from the repository root of each build, with that
build's `python/aisim` environment:

```bash
python -m pytest -q -rx -n 4 -c python/aisim/pytest.ini -p no:cacheprovider \
  crates/core/parity_tests/perfmodel/test_engine_step_parity.py
python -m pytest -q -rx -n 4 -c python/aisim/pytest.ini -p no:cacheprovider \
  crates/core/parity_tests/perfmodel/test_compile_engine_parity.py \
  crates/core/parity_tests/perfmodel/test_afd_parity.py \
  crates/core/parity_tests/perfmodel/test_afd_execution_plan_parity.py
```

| Code revision | Later heads with no product-code change | Engine-step parity | Golden set |
| --- | --- | --- | --- |
| `62293977` | `6511a26d` | 305 passed | 121 passed |
| `9a2bde6b` | `d321defd` | 305 passed | 121 passed |
| `19e8daf6` | `879aebde`, `35a323d6` | 305 passed | 121 passed |
| `70a8ea00` | `df8a4277` | 305 passed | 121 passed |
| `9eaa66e8` | `663344ae` | 305 passed | 121 passed |
| `35e632cb` | None | 305 passed | 121 passed |
| `1ea0211c` | `7cbe9c48`, `14ffdb4c` | 305 passed | 121 passed |
| `59c4575a` | None | 305 passed | 121 passed |
| `1db9b803` | None | 305 passed | 121 passed |
| `108835ac` | `5edfa0d9`, `d69b2bc7`, `7b93131d` | 305 passed | 121 passed |
| `71fcef7b` | None | 305 passed | 121 passed |
| `bf03c37d` | None | 305 passed | 121 passed |

Each result applies to the code revision in the first column. The later
heads add only tests, documentation or evidence, but the parity suites were
not rerun at those heads. Their exact-head results are the published head's
CI, not this table.

Per-phase `mixed_step_latency`, TensorRT-LLM, MTP3, with `ctx_tokens=6000`,
`gen_tokens=8`, `isl=4096`, `osl=64` and `prefix=3000`: the total rises 1.21%
(26.013 to 26.328 ms) from Baseline to Candidate. The decode component rises
from 0.20 to 0.35 ms because decode Mamba now runs at decode coordinates. Pure
prefill does not change. Each revision ran `python phases.py` from its
`python/aisimulate` environment, with `qual_patch.py` beside it:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qual_patch; qual_patch.apply(3)
from aisimulate_core import AicEngine
from aisimulate_core.sdk.engine import compile_engine
spec = compile_engine("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "b200_sxm", "trtllm",
                      "1.3.0rc20", tp_size=1, nextn=3, moe_tp_size=1, moe_ep_size=1)
eng = AicEngine.from_spec(spec)
args = (6000, 8, 4096, 64, 3000)  # ctx_tokens, gen_tokens, isl, osl, prefix
_, _, decode = eng.mixed_step_breakdown_per_op(*args)
print(eng.mixed_step_latency(*args), sum(row[1] for row in decode))
```

The same script at commit `50ded579`, with the `aisim_core` package names,
gives 26.294 ms and a 0.345 ms decode component. That value is not a
before/after delta, because main changed between the revisions. That commit
also splits this shape's 1904-token remainder into two request chunks.
Commit `62293977` prices that remainder as one request capped at
`isl - prefix = 1096` new tokens.

## Controls

- Replay (`predict --stack engine`) did not change between Baseline `49d93f5a`
  and Candidate `686275e7`. At those revisions, Replay priced prefill and
  decode as separate per-rank phase passes. Over 14 configurations, requests,
  output tokens, cache reuse and per-request status matched. Latency metrics
  differed only by rounding, about 1e-16. Current op-level Replay prices a
  mixed rank as one mix-step iteration. It then splits the evidence by phase,
  so this control does not describe the current code. The native Replay
  oracle above covers the current path.
- At the measured revisions, Main, Baseline and Candidate gave identical
  results for unqualified ops. The later `run_agg` caller corrections change
  unqualified results too. See [`run_agg` caller corrections](#run_agg-caller-corrections)
  and [Prefill step count and reuse scope](#prefill-step-count-and-reuse-scope).
- Qualified and unqualified runs differ without the fix, because the analytical
  bound adds recurrent-state bytes. Compare Baseline with Candidate only.
- Nemotron-3-Super-120B-A12B FP8 on TensorRT-LLM fails in both variants with a
  missing MoE row. No other estimate replaces it.

## Request-count fix

Commit `b2b489d999e97749c2932c3f91b1dddb544c81a8` spreads the telemetry prefill
token sum over the observed request count. Before it, an uneven sum added a
request. The shapes above divide evenly, so they do not change. A same-host check
of `686275e7` against `b2b489d9` changed only qualified uneven shapes. On vLLM,
one step had `num_prefill_requests: 2`, `sum_prefill_tokens: 17` and
`num_decode_requests: 1`. Its latency went from 5.860174 to 5.853910 ms. Before
the fix, the two prefills that total 17 tokens priced as two requests of 8
tokens and one request of 1 token. After it, they price as one request of 9
tokens and one of 8. The difference, 0.006264 ms, matches one FP16 state pair
(2 MiB) for each of 23 Mamba layers at 7.7e12 B/s (0.0062642 ms) to the six
decimals shown. All other shapes were identical.
These two latencies are transcribed observations. Their full vLLM configuration,
metrics payload and command were not saved, so they cannot be reproduced. The
reproducible check is `qualified_mamba_mixed_steps_preserve_prefill_and_decode_shapes`.
It prices `num_prefill_requests: 2` and `sum_prefill_tokens: 17` against
17 token rows plus 2 state pairs, never 3.

## Host cost

Six alternating Baseline and Candidate pairs ran. The retained replay results
record the in-process runner time (`wall_s`) and the process peak memory
(`maxrss_kb`):

| Field | Baseline | Candidate |
| --- | --- | --- |
| `wall_s` | median 1.627 s (1.622 to 1.660) | median 1.637 s (1.635 to 1.645) |
| `maxrss_kb` | median 171 MiB (162 to 198) | median 162 MiB (162 to 198) |

Six samples cannot separate the variants. The whole-process times (1.70 s and
2.37 s) and the `run_agg` memory (about 700 MB) are transcribed observations.
Their raw log was not saved. No speed, memory or significance claim is made.

### Forward-prediction host time

The advisory Forward Prediction Performance gate compared main `bf1b539f`
with `14ffdb4c` and found 81 regressions among 89 stable comparisons. For
example, Qwen3-32B SILICON context at batch 1 and isl 1024 went from 20.48
to 24.23 us warm. The gate times `InferenceSession.run_static_latency_only`.
Each query checks the compiled-engine identity memo once. Commit `55d2821b`
made that check rebuild the database mode key and sort the transfer policy
on every query. On the workstation, `engine_identity` took 0.30 us per call,
against 0.10 us for main's object-only check. Commit `59c4575a` compares
the mode and the policy by object identity, and `1db9b803` reads their
stored attributes directly, which takes 0.17 us. The setters replace both
values, so the check still detects a mode or policy change.

The gate's own `worker.py` ran each build on the 13 SILICON cases of
Qwen3-32B and Nemotron-3-Super-120B FP8. Each case ran 10 warm-up and 100
timed queries. This local comparison is separate from the CI gate. The
driver is [`host_perf.py`](mamba-mixed-step-accounting/host_perf.py). Put each
build's checkout in `ROOT/BUILD/src` and its environment in
`ROOT/BUILD/venv`, then run
`python host_perf.py ROOT 6 OUT.json rbf1b539f r1db9b803 rbf03c37d`. The case
list comes from the gate's `expand_cases`. Round `r` rotates the build order
by `r`, so in six rounds each build runs twice in each position. The output
records each round's order and the per-round cold and warm medians of every
case:
[`host-perf-bf03c37d.json`](mamba-mixed-step-accounting/host-perf-bf03c37d.json).
Warm medians over the rounds, in microseconds, with the range of the round
medians:

| Case | `bf1b539f` (main) | `bf03c37d` | Change |
| --- | --- | --- | --- |
| Qwen3-32B context, batch 1, isl 1024 | 14.11 (13.80 to 14.27) | 14.57 (14.24 to 16.12) | +3.3% |
| Qwen3-32B context, batch 1, isl 32768 | 15.59 (15.29 to 15.98) | 16.13 (15.80 to 18.09) | +3.5% |
| Qwen3-32B generation, batch 1, isl 1024 | 24.83 (24.52 to 25.26) | 25.20 (24.75 to 25.95) | +1.5% |
| Qwen3-32B generation, batch 128, isl 1024 | 24.83 (24.57 to 25.20) | 25.30 (24.87 to 27.77) | +1.9% |
| Nemotron Super context, batch 1, isl 1024 | 20.83 (20.61 to 21.01) | 21.68 (21.24 to 23.92) | +4.1% |
| Nemotron Super generation, batch 128, isl 1024 | 42.47 (42.06 to 43.17) | 43.24 (43.12 to 44.28) | +1.8% |

`bf03c37d` is slower than main in all 13 cases, by +0.3% to +4.1% (median
+1.9%), about 0.07 to 0.86 us per query. The round ranges overlap in 11 cases.
`1db9b803`, the code of the earlier comparison, gives +1.4% to +14.7%
(median +2.1%). The compiled-engine handle lookup accounts for about 0.1 us:
it takes 0.34 us on main and 0.45 us at `bf03c37d`. The rest of the
difference has no change in Python function calls under cProfile. Inlining
the lookup saved less than 0.04 us, so it was not kept.

An earlier five-round run, in
[`host-perf-1db9b803.json`](mamba-mixed-step-accounting/host-perf-1db9b803.json),
ran the builds in the same order every round, `1db9b803` last. It gave -7.4%
to +2.3%. Because the order was fixed, it does not control for build
position, and the rotated run replaces it. It also replaces an earlier three-round fixed-order run, in
[`host-perf-1ea0211c.json`](mamba-mixed-step-accounting/host-perf-1ea0211c.json).
That run measured `1ea0211c` at 17.08 us for the first case, against
14.08 us for main.

The CI gate on `c425c61a` found one confirmed regression among 101 stable
comparisons, DeepSeek-V3.2 TP8 generation from a cold query (+14.7%
geometric mean). The rotated driver also ran the six DeepSeek-V3.2 gate
cases. At `bf03c37d`, the two TP8 generation cold queries change by +0.2%
and +3.2%, and all six by -1.6% to +3.3%. So it does not reproduce that
result. That run used the `bc84a5fd` driver with its model filter replaced
by `model_id.startswith("deepseek-v3.2")`. The checked-in driver selects
the same six cases with
`HOST_PERF_MODELS=deepseek-v3.2 python host_perf.py ROOT 6 OUT.json rbf1b539f r1db9b803 rbf03c37d`.
The raw rounds are in
[`host-perf-deepseek-bf03c37d.json`](mamba-mixed-step-accounting/host-perf-deepseek-bf03c37d.json). These are simulator host timings, not predicted
serving latency. The paired CI gate on the published head remains the
exact-head result.

## Limits

- Summing uniform-batch queries can overcount launch overhead for
  variable-length batches.
- One model family, one modeled platform, synthetic shapes.
- With one request whose uncached tokens exceed the budget, `run_agg`
  counts one mix step in throughput and energy, as main does, although the
  request takes several prefill chunks. This is part of the unresolved
  multi-chunk TTFT limit above.
- A pure-prefill packed step prices every operation at the integer mean of
  its new tokens per request, as main does. Two prefills of 8 and 9 new
  tokens price as two of 8. The exact-total query
  `predict_prefill_latency_total` does the same for a graph without a
  whole-model FPM. For example, 2 requests and 17 new tokens price 16 tokens.
  The mixed packed step keeps the remainder for qualified Mamba operations.
- Replay prices a rank through `ForwardPassMetrics`, which carries prefill
  totals and request counts, not each chunk's length. A qualified Mamba scan
  therefore spreads the total evenly over the requests: chunks of 1000 and
  500 tokens price as two chunks of 750. Other operations also see only the
  totals. Per-chunk lengths would need a change to the shared telemetry
  contract.
- The compiled-engine identity is computed once per model object, database
  object, query mode and transfer policy, as main computes it per model and
  database object. Changing a model's operation lists after its first
  estimate is not detected. Build a new model object instead.
- In the mixed step, shared token-major operations take
  `ctx_tokens - prefix * floor(ctx_tokens / isl)` prefill tokens for every
  graph, as main does. A remainder longer than `isl - prefix` therefore
  prices more shared tokens than qualified Mamba and context attention
  price. For example, `ctx_tokens = 6000`, `isl = 4096` and `prefix = 3000`
  give 3000 shared prefill tokens, but 2 requests of 1096 tokens for Mamba
  and attention. The shared count stays the same for legacy and qualified
  graphs, so a Mamba qualification does not change non-Mamba prices.
