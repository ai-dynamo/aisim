<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Analytical AFD and EPD

Attention/FFN disaggregation (AFD) and encoder/prefill/decode disaggregation
(EPD) are estimated analytically. Neither runs request-by-request token replay
for the disaggregated part. Their inputs and reports are therefore narrower
than ordinary aggregated or P/D replay. AFD and EPD cannot be combined.

<a id="afd"></a>
<a id="analytical-afd"></a>

## AFD

AFD places attention (A) and FFN/MoE (F) work in separate pools. Set
`engine.mode: afd` and an `engine.afd` block:

```yaml
traffic:
  source: {type: synthetic, input_tokens: 1024, output_tokens: 128}
  load: {type: concurrency, concurrency: 16}
  stop: {requests: 32}
engine:
  mode: afd
  model: Qwen/Qwen3-30B-A3B
  hardware: h200_sxm
  backend: vllm
  afd:
    phase: both
    combined_with_pd: false
    n_a_nodes: 2
    n_f_nodes: 4
    tp_a: 4
    a_batch_size: 64
```

| Knob | Default | Recommend | Rules |
|---|---|---|---|
| `engine.afd.phase` | Required | fixed | `both` for pure AFD; `prefill` or `decode` with `combined_with_pd: true`. |
| `engine.afd.combined_with_pd` | Required | fixed | `false`: AFD serves both phases. `true`: AFD serves one phase and an ordinary worker serves the other. |
| `engine.afd.n_a_nodes`, `n_f_nodes`, `tp_a` | Required | Enumerated within the GPU budget | Positive. Recommend may constrain `tp_a`. |
| `engine.afd.a_batch_size` | Required | User-supplied domain | Positive attention-worker batch size. |
| `engine.afd.f_moe_ep_size` | `1` | Model-derived domain | Positive; recommend also accepts `n_f_nodes` or `ffn_tp`. Dense models require `1`. |
| `engine.afd.num_microbatches` | `3` | `{choices: [2, 3, 4]}` | Positive. |
| `engine.afd.pipeline_model` | `optimistic` | `{choices: [optimistic, conservative]}` | `optimistic`, `conservative` or `serial`. |
| `engine.afd.comm_overhead_factor` | `1.0` | fixed | Positive factor on A/F communication. Must be `1.0` with `fastafd_profile_path`. |
| `engine.afd.boundary_on_attn` | `true` | fixed | A/F boundary convention. |
| `engine.afd.fastafd_profile_path` | `null` | Prediction only | Measured FastAFD split MoE stage; see [below](#measured-fastafd-moe-stage). Requires `phase: decode`, `tp_a: 1`, `f_moe_ep_size` equal to the F GPU count, and `boundary_on_attn: true`. Backend version aliases are pinned before measurement. |
| `engine.afd.fastafd_moe_backend` | `null` | Prediction only | Required with `fastafd_profile_path`; `megamoe` or `deepep_deepgemm`. |
| `engine.moe_quant_mode` | `null` | FastAFD prediction only | Exact profile `moe_precision` selector; required when otherwise matching entries differ in precision. Also qualifies F memory at the selected precision. Does not override attention or companion-worker timing. Unsupported for AFD without a FastAFD profile. |
| `engine.afd.max_af_ratio`, `max_candidates` | `4.0`, `10000` | Recommend only | Bounds on the generated topology space. |

Workers:

- `combined_with_pd: false` requires `phase: both` and no other worker roles.
- `combined_with_pd: true` requires exactly the opposite-phase worker. For
  example, `phase: decode` requires `workers.prefill`.
- `engine.kv_transfer` is rejected; A↔F transfers come from `engine.afd`.

The pass time is built from measured per-layer A, F, A→F and F→A times:

| `pipeline_model` | Per-microbatch-layer cadence |
| --- | --- |
| `optimistic` | `max(A, F, A→F + F→A)` |
| `conservative` | `max(A + A→F, F + F→A)` |
| `serial` | `A + A→F + F + F→A` |

A pass adds pipeline fill to the cadence of every microbatch and layer. Queueing
contributes to TTFT and E2E latency. TPOT is
`(E2E − TTFT) / (OSL − 1)`, or zero for one-token outputs.

<a id="measured-fastafd-moe-stage"></a>

### Measured FastAFD MoE stage

`engine.afd.fastafd_profile_path` with `fastafd_moe_backend` takes the decode F
time through the canonical Rust `fastafd_stage` estimator from an AISimulate
[FastAFD MoE stage profile](../../perf-model/schemas/fastafd-moe-stage-profile.md)
instead of the operation model. The profile's `afd` entry must match the model,
hardware, `<A GPUs>A<F GPUs>F` topology, `a_batch_size`, MTP degree,
`num_microbatches`, the checkpoint's total layer count, and the model's routed
top-k exactly; nothing is interpolated or projected. The attention provider must
report the same layer count. The current pipeline models a MoE block at every
layer position, including dense-prefix positions: DeepSeek-V3 requires a capture
of 61 block repetitions, not its checkpoint's 58 routed layers. Captures with a
different repetition count are rejected; see the
[homogeneous layer contract](../../perf-model/schemas/fastafd-moe-stage-profile.md#consumers).
Top-k comes from normalized
model metadata (`topk`); an unknown or invalid width is infeasible. Its stage latency
covers dispatch, experts, return and the A-side combine for every MoE layer of one
decode step, so it is spread evenly over `moe_layers × num_microbatches` pipeline
units as F, and A→F and F→A are zero. With a FastAFD profile,
`comm_overhead_factor` must be `1.0`: communication is included in the measured
stage and cannot be calibrated separately. Non-unit factors are rejected by
both prediction configuration validation and the direct provider API.
Public `engine.backend_version` slot aliases (`current`, `previous`, `next`)
are resolved to a literal version before FastAFD stage construction. The A-side
model and companion worker use that same pinned version; unavailable slots fail.
A keeps the attention provider's
A-side compute without its own combine term and adds the router, which FastAFD
runs on the A side outside the measured stage. The attention provider must report
these per-layer components; the built-in estimator does. Its `a_router_ms` is
evaluated at the local A microbatch, widened by `nextn + 1` for MTP, using the A
model. It is distinct from `router_ms`, the F-side router contribution measured
at the combined F batch. Only `a_router_ms` is added to FastAFD attention time.
The built-in provider measures A only: it does not query the discarded F compute,
transfer, or combine models. Its combine component is zero because combine is
already inside the measured stage. A memory includes the local router and KV
cache; F memory uses the selected stage precision and excludes the router.
Both pools must fit, so an oversized default-precision F model cannot reject a
fitting low-bit stage. The resolved stage identity and both memory breakdowns
are retained in attention provenance. Explicit external providers retain their
own measurement and memory-qualification contracts.
Decode providers retain the KV samples and their repeat counts in
`provenance.decode_samples`, including each sample's attention boundary
components. FastAFD replaces the F stage and recomposes A for every sample.
When a provider also supplies typed `decode_attention_samples`, each value and
repeat count must match the provenance trace before boundary recomposition.
The serialized measurement's `decode_attention_samples` contains the resulting
`[attention_ms, repeat]` pairs; its headline A time is their weighted mean.
Loading a saved FastAFD measurement with `provenance.attention.decode_samples`
requires matching typed pairs for every recomposed sample and repeat count;
missing or inconsistent pairs are rejected before replay.
The trace's repeat total must equal `max(output_length - 1, 1)` at evaluation,
including serial replay. A saved trace cannot price a different decode length.
The saved FastAFD stage key must also match the replay's pool sizes, per-A-rank
batch, microbatch count, and layer count, with the supported TP, EP, and boundary
settings. Matching saved measurements remain usable without reopening the profile.
The pipeline maximum is evaluated before averaging the sampled cadence, so a
decode trace that crosses between F-bound and A-bound does not undercount TPOT.
Foreground intervals use the mean stage durations and sampled mean cadence.
Custom providers without a sample trace describe constant attention time.
The canonical model returned by `RustForwardPassPerfModel.best_available` also
provides `estimate_fastafd_attention(isl=..., osl=..., max_seq_len=...)` for a
selected `fastafd_stage` identity. It returns attention compute as
`attention_ms` and reports `a_router_ms` separately in `provenance.components`.
The FastAFD adapter adds the router later; direct callers needing total A time
must add both components. Both `isl` and `osl` must be positive integers, including
the supported one-token output case. A configured `max_seq_len` must be at least `isl + osl`
so the KV allocation covers the full evaluated decode sequence.
This widening describes the lower-level provider. Public AFD configuration
rejects nonzero `engine.nextn`, including `nextn: 1` with `nextn_accepted: 1`,
before profile lookup. MTP profiles therefore cannot be selected through
`engine.afd`; that path currently requires profile entries with `mtp_nextn: 0`.
Consumed components must be finite and non-negative, and attention compute plus
combine must reconcile
with the provider's attention total (relative tolerance `1e-9`, absolute tolerance
`1e-12` ms). A positive `a_router_ms` is required. When several entries
differ only in `moe_precision`, set `engine.moe_quant_mode` to the exact profile
precision, for example `fp8`; an unset selector is ambiguous, and an unmeasured
precision is infeasible. In FastAFD prediction this selects only the measured
stage, leaving attention and companion-worker timing unchanged. Other engine
model controls remain unsupported for AFD. The profile entry, its producer and
the attention composition are recorded in the measurement provenance. FastAFD
is not a runtime dependency.

Requirements: fixed synthetic input and output lengths and no trace.
`kv_capacity_fraction` load is rejected because the A/F pools have no
scheduler-visible KV capacity.

Each prediction writes `afd-replay-spec.json` and `afd-qualification.json`
with the model, topology and GPU accounting. They are marked
`native_deployment_supported: false`; no runnable backend deployment is
generated. See [search space](../../sweeper/search-space.md) for recommendation
rules.

<a id="epd"></a>
<a id="analytical-epd"></a>

## EPD

EPD adds an encoder pool for image inputs in front of ordinary language
replay. Add `engine.workers.encoder` and give the synthetic source fixed images,
as in [`examples/cli/epd-predict-aggregated.yaml`](../../../examples/cli/epd-predict-aggregated.yaml):

```yaml
traffic:
  source:
    type: synthetic
    input_tokens: 128
    output_tokens: 32
    images: {height: 448, width: 448, count: 1}
  load: {type: concurrency, concurrency: 8}
  stop: {requests: 16}
engine:
  model: Qwen/Qwen3-VL-8B-Instruct
  hardware: h200_sxm
  backend: sglang
  backend_version: 0.5.14
  mode: aggregated
  workers:
    encoder: {tensor: 1, replicas: 1, batch_size: 2}
    aggregated:
      parallelism: {tensor: 1, replicas: 1}
      scheduler: {max_batched_tokens: 8192, max_sequences: 8}
```

```bash
aisimulate predict --stack engine --config examples/cli/epd-predict-aggregated.yaml \
  --output-dir ./epd-prediction
```

A disaggregated variant is in
[`examples/cli/epd-predict-disaggregated.yaml`](../../../examples/cli/epd-predict-disaggregated.yaml).

| Knob | Default | Recommend | Rules |
|---|---|---|---|
| `engine.workers.encoder.hardware` | `engine.hardware` | fixed | Encoder system identifier. |
| `engine.workers.encoder.backend_version` | Resolved | fixed | Encoder performance data. The backend follows `engine.backend`. |
| `engine.workers.encoder.tensor` | `1` | Scalar or `choices` | Positive. |
| `engine.workers.encoder.replicas` | `1` | Scalar or `choices` | Positive. |
| `engine.workers.encoder.batch_size` | `1` | Scalar or `choices` | 1 to 8. |
| `engine.workers.encoder.latency_correction` | `1.0` | fixed | Positive multiplier on encoder latency. |
| `engine.workers.encoder.rate_degradation` | `0.9` | fixed | In `(0, 1]`. Fraction of ideal encoder throughput that is usable. |

Visual tokens are computed from the image size and added to each request's
text input once. Language replay then runs as usual. Afterwards, throughput is
capped at the degraded encoder capacity, and the raw encoder batch latency is
added to mean TTFT and E2E. Encoder queueing does not feed back into the
language schedule.

Results are marked `analytical_epd_overlay` and report means only, without
percentiles, per-request records or telemetry. GPU-hours include encoder GPUs.

Requirements: fixed synthetic text and image inputs, fixed `concurrency` load,
static workers, and default op-level language timing. Rejected: traces,
sessions, variable lengths, prefix sharing, rate or KV-relative load,
whole-forward (FPM) timing, adapters, and `--capture-per-request`. Encoder CPU
time and embedding transfer are not modeled, and no deployment artifacts are
generated. The encoder model follows
[AIC revision f8f2341](https://github.com/ai-dynamo/aiconfigurator/commit/f8f2341cb5761877bda694ab954cb6f5eff78fd4).
