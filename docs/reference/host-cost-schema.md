# Host cost data (`host-cost`)

Measured host-CPU time a serving framework spends turning a chat request into engine input, per request,
with the regression that turns it into replay coefficients. It feeds
[`engine.tokenization`](../replay/engine/README.md#tokenization) and
[`engine.extra_cpu_overhead`](../replay/engine/README.md#extra-cpu-overhead); see
[measured CPU overhead](../replay/engine/README.md#measured-cpu-overhead).

## What an entry is

One configuration is one measurement of (framework, framework version, system, tokenizer) on that
system's host CPU. The harness (`collector/host_cpu/`) runs the framework's own request code inside its pinned
image and times each step of a real chat request body: JSON parse, request validation, chat-template render,
encode, and (vLLM only) prefix-cache block hashing. It needs no GPU kernels; a GPU is held only because the
host CPU must be the serving node's. Collect with the
[`aisim-cpu-overhead-collect`](../../.agents/skills/aisim-cpu-overhead-collect/references/tokenizer.md) skill.

## Layout

`data/host-cost/<framework>/<version>/<system>/<tokenizer>/`

| File | Content |
|---|---|
| `requests.parquet` | One row per timed request, schema `aisim-host-cost/v1` |
| `aisim-manifest.json` | Envelope manifest (`category: host-cost`, `purpose: host-cost-profile`) |

## Row columns (`aisim-host-cost/v1`)

| Column | Meaning |
|---|---|
| `request`, `tokens`, `body_kb` | Request index, input tokens after the chat template, body size |
| `json_ms`, `validate_ms`, `template_ms`, `encode_ms` | Median of the repeats of each step, milliseconds |
| `hash_ms` | Prefix-cache block hashing; null where the framework does not hash before scheduling |

## Manifest provenance (`provenance.host_cost`)

- `environment`: framework and transformers versions, tokenizer class and whether it is fast, hash algorithm
  and block size, tokenizer worker count (null when unverified), workload hash, repeats and merged runs.
- `host`: CPU model, thread count, 1-minute load average of each run, Slurm job ids.
- `fit`: the regression `base_ms + per_input_token_us × tokens` over the queued steps (`template`, `encode`,
  `hash`): `base_ms`, `per_input_token_us`, `r`, `n`, dropped requests, token range, residual p50/p90 and a
  95% bootstrap interval. A request whose cost per token exceeds twice the median is dropped.
- `gpu_time`: the wall time the entry ran inside its Slurm allocation, in `aisim-gpu-time/v1`.

## Unqueued entries (`host-cost-unqueued`)

The rest of the serving path's host time before the engine (HTTP receive, input processing, first-chunk work),
measured at one request in flight as client TTFT minus the engine's prefill span. Collect with
[`references/unqueued.md`](../../.agents/skills/aisim-cpu-overhead-collect/references/unqueued.md).

`data/host-cost/<framework>/<version>/<system>/<tokenizer>/unqueued-<endpoint>/`

| File | Content |
|---|---|
| `runs.parquet` | One row per measured run, schema `aisim-host-cost-unqueued/v1`: `run`, `requests`, `base_ms`, `total_per_input_token_us`, `r2` (`requests` and `r2` may be null for a summarized run) |
| `aisim-manifest.json` | Envelope manifest (`purpose: host-cost-unqueued`); identity settings `cpu_model`, `tokenizer_key`, `endpoint` |

`<version>` is the version the runs were served on. `provenance.host_cost.fit` holds the mean over runs of
`base_ms` and `total_per_input_token_us`, and the tokenizer entry and slope the total was split with at write time.
`provenance.host_cost.method` describes the measurement; host names and job IDs stay in `aisim-data-internal`.

## Using an entry

`aisim_core.host_cost.lookup.lookup(...)` returns the entry for a deployment. The host CPU and tokenizer must
match exactly; nothing is extrapolated across CPUs or tokenizers, and no match means no default. Framework
versions are recollected only every one or two releases, so an uncollected version reuses the closest collected
one (at or below it, else the nearest above) and the result names the version that stood in.
`LookupResult.engine_tokenization()` returns the `engine.tokenization` block; a negative fitted `base_ms` is
clamped to 0. `choose(...)` picks the tokenizer and unqueued entries of one deployment from catalog identities, and
`overhead(...)` shapes them as replay blocks: the unqueued per-token term is the stored total less the tokenizer slope
it is paired with. `aisim_core.host_cost.replay.measured_overhead(...)` does both through `ensure_data` with a
`host-cost` pin.

## Limits

- Idle, single-thread cost. Contention under load, HTTP receive, streaming and detokenization are not measured.
- The fixed, unqueued overhead (`engine.extra_cpu_overhead`) is a separate collection (unqueued entries above);
  `json_ms` and `validate_ms` are stored but not fitted.
- One shared request set (60 real AgentX chat requests of 2.6k to 178k tokens) is re-tokenized by every tokenizer.
