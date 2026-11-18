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

No built-in model sets `state_dtype` or `token_width` yet, so every qualified run
applied this wrapper (`qual_patch.py`) before it built the model:

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

## Operation sources

| Data | Context conv1d | Context scan | Generation |
| --- | --- | --- | --- |
| TensorRT-LLM `1.3.0rc20` | Measured table, exact shape key | Analytical bound | Analytical bound |
| vLLM `0.24.0` | Analytical bound | Analytical bound | Analytical bound |

`fallback_policy: deny` does not prove measured Mamba coverage, because the
analytical bound is an allowed result.

## Results

Qualified Baseline to qualified Candidate.

`InferenceSession.run_agg`, TensorRT-LLM. The script below (`agg.py`) ran each case with
`python agg.py OUT.json --qual 1 --nextn {0,3}` at each commit, from the
commit's `python/aisimulate` package environment, with `qual_patch.py` beside it:

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
sess = InferenceSession(get_model(a.model, mc, a.backend), get_database("b200_sxm", a.backend, a.version),
                        get_backend(a.backend, "b200_sxm"))
out = {}
# (batch, isl, osl, prefix, ctx_tokens): ctx_tokens is the prefill budget per step.
for name, (b, isl, osl, px, ctx) in {
    "chunked_uncached": (16, 6000, 64, 0, 2048),
    "chunked_partial_cached": (16, 6000, 64, 3000, 2048),
    "split_budget_cached": (16, 4096, 64, 1024, 8128),
}.items():
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
With a 3000-token cached prefix, `run_agg` returned the same values as without
it in both variants, so it does not show partial-cache behavior.

Packed telemetry (`RustForwardPassPerfModel.estimate_forward_pass_time_ms`),
both backends:

| Shape | MTP0 | MTP3 |
| --- | --- | --- |
| Pure prefill | 0% | 0% |
| Pure decode, 8 requests | 0% | +2.03% to +2.14% |
| Mixed, 1x2048 + 8 decode | +0.25% to +0.27% | +0.68% to +0.75% |
| Mixed, 2 requests of 1500 tokens + 8 decode | +0.24% to +0.36% | +0.62% to +0.71% |

The MTP3 column is not valid for the current code. Commit `686275e7` multiplied
the decode count by the verification width before the Mamba query. Packed
telemetry already carries `requests * (nextn + 1)` decode requests, so that
commit applied the width twice. The current code applies it once. The MTP0
column has width one and does not change. The `run_agg`, per-phase and replay
results above and below do not use packed telemetry.

Per-phase `mixed_step_latency`, TensorRT-LLM, MTP3: a 6000-token chunk with a
3000-token prefix rises 1.21% (26.013 to 26.328 ms). The decode component rises
from 0.20 to 0.35 ms because decode Mamba now runs at decode coordinates. Pure
prefill does not change.

## Controls

- Replay (`predict --stack engine`) did not change between Baseline `49d93f5a`
  and Candidate `686275e7`. At those revisions, Replay priced prefill and
  decode as separate per-rank phase passes. Over 14 configurations, requests,
  output tokens, cache reuse and per-request status matched. Latency metrics
  differed only by rounding, about 1e-16. Current op-level Replay prices a
  mixed rank as one mix-step iteration and then splits its evidence by phase,
  so this control does not describe the current code. The replay regression
  `op_level_replay_prices_qualified_mamba_once_per_request_and_by_phase`
  covers the current path.
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
2 prefills of 17 tokens plus 1 decode went from 5.860174 to 5.853910 ms. That
difference is one FP16 state pair (2 MiB) for each of 23 Mamba layers at 7.7e12
B/s, within 1e-13. All other shapes were identical.

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
