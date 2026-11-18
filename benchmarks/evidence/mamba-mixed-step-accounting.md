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

The MTP3 column is historical and not valid for the current code. Commit
`686275e7` multiplied the decode count by the verification width before the
Mamba query. Packed telemetry already carries `requests * (nextn + 1)` decode
requests, so that commit applied the width twice. The current code applies it
once. The original qualified Baseline-to-Candidate comparison has no
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
`python telem.py OUT.json` from its own `python/aisim` build. The script is
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
[`agg_caller.py`](mamba-mixed-step-accounting/agg_caller.py) ran
`python agg_caller.py OUT.json` from each commit's `python/aisim` build, in
`HYBRID` mode at MTP0 with BF16 states. The raw outputs are
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

`agg_caller.py` gained the large-budget case, its exact 4096-token control and
a same-session prefix sequence. It ran from the `9a2bde6b`, `19e8daf6` and
`70a8ea00` builds. The raw outputs are
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
new tokens. Every mixed-step operation then equals the result for an exact
4096-token budget, and so does TTFT. The context convolution, both projection
GEMMs and the norm also fall, so the non-Mamba sum drops by 27%. In the
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

| Code revision | Published heads with this code | Engine-step parity | Golden set |
| --- | --- | --- | --- |
| `62293977` | `6511a26d` | 305 passed | 121 passed |
| `9a2bde6b` | `d321defd` | 305 passed | 121 passed |
| `19e8daf6` | `879aebde`, `35a323d6` | 305 passed | 121 passed |
| `70a8ea00` | The head that adds this row | 305 passed | 121 passed |

Each published head adds no product code to its code revision, only tests,
documentation or evidence, so it carries that revision's parity results.

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
- Main, Baseline and Candidate give identical results for unqualified ops.
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

## Limits

- Summing uniform-batch queries can overcount launch overhead for
  variable-length batches.
- One model family, one modeled platform, synthetic shapes.
