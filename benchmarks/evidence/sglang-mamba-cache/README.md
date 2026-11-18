<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Historical Qwen hybrid-cache diagnostic

[historical-accuracy.json](historical-accuracy.json) restores the measurement
receipt originally committed in this PR at
`f2a460101cfb09bdb52c00ddf2680bc16b0dba84`, formerly under
`docs/replay/engine/mamba-cache-accuracy.json`. Only the prose typo `all95` is
corrected. These are historical author-reported measurements, not a rerun on
the repaired head. The input captures are not distributed in this repository
and were unavailable to the repair agent.

The capture `qwen-tp4-c1-source24-5563558` used Qwen3.5-397B-A17B FP8,
4 B200 GPUs, SGLang TP4 / attention DP 1, one lane, STP, page size 64 and
chunked prefill size 16,384. Its 95 requests came from 24 selected short trace
windows, with outputs capped at 128 and inter-request gaps capped at one
second. It is not an unmodified published benchmark or a submission result.

| Historical metric | Before | After |
| --- | ---: | ---: |
| Exact cached-prefix agreement, 95 requests | 77/95 | 95/95 |
| Cached-prefix MAE | 663.58 tokens | 0 tokens |
| Conditional first-prefill step MAPE | 7.3890% | 0.02392% |
| Client TTFT MAPE, 94 profiling requests | 50.8630% | 47.1290% |
| Client TPOT MAPE, 94 profiling requests | 2.6444% | 2.6444% |

The fixed FPM was trained on this same capture: **zero independent test rows**.
The first-step score evaluates that unchanged forest at the admission geometry;
it does not establish prediction accuracy on held-out data or aligned later
batches. Native generated output IDs and complete live checkpoint inventory
were not captured. The remaining client TTFT error is unresolved. The receipt
does not qualify memory sizing, state eviction, downstream deployment recipes,
or current-head accuracy. Current repair tests instead assert synthetic cache
and scheduler transitions and unsupported-scope rejection.

## Reproduce the historical comparison

Use the exact six inputs named by `inputs_sha256` in the receipt: execution
descriptor, observed agentic trace, frozen forest, canonical native FPM rows,
prompt map and client metrics. Verify their SHA-256 values first. The trace
header must bind the native prompt map, and the execution descriptor must
reference the captured trace and forest. No model weights or GPU rerun are
needed. Missing inputs mean the measurement cannot be independently reproduced.

Build separate checkouts of baseline
`4ff7bd2ab8748c4c843db8a1755eb0bc59c8a0c6` and behavior candidate
`93a9c551555331b0b9825b9f6c332bcc1819abfe`, with the same toolchain and flags:

```bash
cargo +1.99.0 build --release --locked -p aisim-core --features extension-module
```

Set `MAMBA_BASE_RUNTIME` and `MAMBA_FIXED_RUNTIME` to the resulting
`target/release/libaisim_core.so` files. Set `MAMBA_EXECUTION` to the captured
execution descriptor, `MAMBA_NATIVE` to its native postprocess directory and
`MAMBA_RESULTS` to a fresh scratch output directory. Extract the exact
historical scorer from this repository's immutable commit, keeping it outside
the checkout:

```bash
git show f2a460101cfb09bdb52c00ddf2680bc16b0dba84:scripts/prediction_regression/compare_mamba_cache_accuracy.py > "$MAMBA_RESULTS/scorer.py"
python3 "$MAMBA_RESULTS/scorer.py" run \
  --runtime "$MAMBA_BASE_RUNTIME" --revision 4ff7bd2ab8748c4c843db8a1755eb0bc59c8a0c6 \
  --execution "$MAMBA_EXECUTION" --output "$MAMBA_RESULTS/before"
python3 "$MAMBA_RESULTS/scorer.py" run \
  --runtime "$MAMBA_FIXED_RUNTIME" --revision 93a9c551555331b0b9825b9f6c332bcc1819abfe \
  --execution "$MAMBA_EXECUTION" --output "$MAMBA_RESULTS/after" \
  --mamba-config '{"prefill_checkpoint_interval":64,"decode_checkpoint_interval":256,"max_states":994}'
python3 "$MAMBA_RESULTS/scorer.py" compare \
  --before "$MAMBA_RESULTS/before" --after "$MAMBA_RESULTS/after" \
  --native-canonical "$MAMBA_NATIVE/canonical/canonical-fpm.jsonl" \
  --native-prompts "$MAMBA_NATIVE/prompt-map/native-request-token-map.jsonl" \
  --native-client-metrics "$MAMBA_NATIVE/request-metrics.json" \
  --output "$MAMBA_RESULTS/accuracy.json"
```

This frozen scorer is retained in Git history for reproducing this capture,
not restored as a supported general-purpose tool. It does not validate which
side enables Mamba and assumes the captured forest's `-1` leaf sentinel. Check
that `before/execution.json` omits `mamba_cache` and `after/execution.json`
enables exactly the receipt's 64/256/994 configuration before interpreting
its output. Do not use it with other leaf sentinels. Compare the executed
payload, runtime, runner and report hashes against the receipt. The receipt's
forest-interpreter control covers all 210 captured prefill rows only.

Repeat each run into fresh directories and use the scorer's `parity`
subcommand with `--first`, `--second` and `--output`. It excludes only
`wall_time_ms`, `processed_tokens_per_s` and `processed_output_tokens_per_s`.
Also compare the baseline with the candidate run without `--mamba-config`.
These historical parity controls are separate from numerical accuracy and
from host performance measurements.
