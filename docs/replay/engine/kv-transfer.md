<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# P/D KV transfer

In `mode: disaggregated`, each request is prefilled by a `prefill` worker and
then decoded by a `decode` worker. `engine.kv_transfer` sets how long moving the
prompt's KV cache between them takes. Add it to a disaggregated configuration
such as the [P/D example](README.md#a-complete-pd-example):

```yaml
engine:
  mode: disaggregated
  kv_transfer:
    bytes_per_token: auto
    bandwidth_gb_per_second: 50
    timing_mode: destination_missing
```

Omitting `kv_transfer` is the same as `{}`: the transfer happens with no delay.
A native ReplaySpec or Sweeper search space whose prefill rank sets a positive
`kv_transfer_bandwidth` without `kv_transfer_bytes_per_token` is rejected; the
YAML above always resolves the bytes. A rank with `kv_cache_groups` is the
exception. It takes the bytes from its groups, and it refuses
`kv_transfer_bytes_per_token`. The YAML leaves that field out for such a rank.
A native ReplaySpec needs identical `kv_cache_groups` and the same
`vllm_version` on its prefill and decode roles.

## Transfer fields

| Knob | Default | Recommend | Rules |
|---|---|---|---|
| `engine.kv_transfer.bytes_per_token` | `auto` | fixed | Positive integer or `auto`. Bytes moved per prompt token. `auto` derives it from the model and the prefill role's TP/PP/MoE shape. For DeepSeek V4 it is the per-token share of vLLM's padded `fp8_ds_mla` pool row, which holds the compressed KV and indexer keys, not the dense per-layer formula; without `state_cache` the transfer leaves out V4's sliding-window KV and compressor state (about 26 MB per request for V4 Flash at TP8). It can differ from either worker's `kv_cache.bytes_per_token`. With `state_cache`, `auto` uses the prefill role's resolved `kv_cache.bytes_per_token`. Without it, an `fpm_profile` supplies its selected deployment's `kv_bytes_per_token`, which takes precedence over the model estimate, DeepSeek V4 included. A grouped FPM profile, which has no `kv_bytes_per_token`, does not use this field, and a value other than `auto` is refused. The transfer then takes its bytes from the `transfer_bytes_per_block` of each cache group, as described below. |
| `engine.kv_transfer.bandwidth_gb_per_second` | `null` | fixed | Positive, in decimal GB/s. Bandwidth of each prefill rank's sending link and each decode rank's receiving link; concurrent transfers [share them](#link-sharing). `null` means no transfer delay, not a zero-bandwidth link. |
| `engine.kv_transfer.timing_mode` | `destination_missing` | fixed | `destination_missing` charges prompt KV absent from the chosen decode worker's cache; SGLang bills whole uncached pages, including unused tail slots. `full_prompt` charges the logical prompt length without page rounding. |

All fields accept only concrete values; `recommend` does not search them.
`kv_transfer` is rejected in `aggregated` and `afd` modes.

A transfer alone on its links, with both roles at the same bandwidth, takes, in seconds,

```text
(transferred_tokens × bytes_per_token + state_bytes) / (bandwidth_gb_per_second × 10^9)
```

where `transferred_tokens` follows `timing_mode`. For SGLang,
`destination_missing` uses the number of uncached physical prompt pages multiplied
by the decode worker's `kv_cache.block_size`. This includes unused slots in the
last page. `full_prompt` uses the original logical prompt length, so
`destination_missing` can cost more than `full_prompt` even without a cache hit.

For example, a 10-token prompt with SGLang page size 4 occupies three pages.
With 16 bytes per token, bandwidth `0.000001` GB/s, no state bytes and no other
transfers, `full_prompt` bills 10 token slots (160 ms), while a cold
`destination_missing` transfer bills 12 slots (192 ms). If the decode cache
already holds one reusable four-token page, two pages remain: 8 slots (128 ms).
A shared six-token prefix still reuses only one complete page in this example.

Within `destination_missing`, a decode worker that already caches a shared prefix
receives the request sooner, so prefix reuse on the decode side can lower TTFT
even when compute time does not change. `state_bytes` is the prefill role's
`state_cache.bytes_per_request` for
[hybrid-KV models](kv-cache.md#state-cache), and 0 otherwise.

For a [grouped profile](kv-cache.md#other-cache-modes) the transferred bytes
are `full_blocks × bytes_per_full_block + fixed_bytes`. `full_blocks` is the
number of full-attention blocks of the transferred prompt, a partly filled last
block included. When the prefill stops one token short, the transferred prompt
leaves out the last prompt token, as the [request flow](#handoff) shows.
`fixed_bytes` holds one block of each state group and the window blocks of
each group with a `sliding_window`. Such a group moves only the blocks of the
transferred prompt inside the window of the next token. The decode worker
allocates only these blocks. With `T` transferred tokens, that is
`cdiv(T, block) - floor(max(0, T - window + 1) / block)` blocks. `block` is the
group's own `block_size_tokens`. The bytes of one block come from the profile's
`transfer_bytes_per_block`. `bytes_per_full_block` adds them up over the
full-attention groups. With `destination_missing`, a decode worker that already
caches a prefix of the prompt does not receive the full-attention blocks of
that prefix again. If a full-attention group is not prefix-cacheable, the
decode worker receives every full-attention block, as one transfer unit holds
a block of each full-attention group. No full-attention spec of vLLM 0.31.0,
0.32.0 or 0.33.0 opts out of prefix caching. Only a profile that describes no
deployment of these versions has such a group. The state moves with every
transfer that moves at least one token.
A request of zero input tokens owns no block on the prefill worker and moves
no state block; vLLM refuses an empty prompt, see vLLM v0.31.0
[`input_processor.py:516-517`](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/engine/input_processor.py#L516-L517),
vLLM v0.32.0
[`input_processor.py:565-566`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/engine/input_processor.py#L565-L566)
and vLLM v0.33.0
[`input_processor.py:565-566`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/engine/input_processor.py#L565-L566).
`transfer_bytes_per_block` is the bytes that vLLM's NIXL connector
reads for one block of the group on one TP rank, on vLLM 0.31.0, 0.32.0 or
0.33.0. So the result is the bytes that the connector reads for the request on
one rank. A live run of GLM-5.3-Flash measured the bytes of whole transfers on
vLLM 0.31.0 and 0.32.0. They give the value of the full-attention group and
the sum of the other groups. The values of the ring and the four KDA state
groups in the GLM-5.3-Flash test profiles of these two versions are a reading,
not a measurement of each group: they come from the regions that the connector
logged at start and a reading of the connector code, checked against that sum
and the descriptor counts. The evidence folder
[`nixl-pd-transfer-bytes-20261115`](../../../benchmarks/evidence/nixl-pd-transfer-bytes-20261115/README.md)
holds the numbers and the reading.
A later run on vLLM 0.33.0 measured the bytes of each group with a log of the
connector. They equal the values of vLLM 0.32.0. The evidence folder
[`nixl-pd-transfer-bytes-20261117`](../../../benchmarks/evidence/nixl-pd-transfer-bytes-20261117/README.md)
holds the numbers and the region table of the connector. The import of a vLLM
0.33.0 runtime probe record computes the field with the region rule of the
connector. It writes the field only from a probe phase with prefix caching and
only for the measured cache shape.
[Memory accounting](../../perf-model/memory.md#fpm-profile-cache-groups-and-byte-budgets)
gives that rule and its conditions. It also tells what a measurement gives for
a profile and when a group's value stays unset.

Replay differs from vLLM in these known ways for a grouped profile:

- When the decode worker already caches every full-attention block of the
  transferred prompt, vLLM moves nothing, but replay still charges
  `fixed_bytes`. Only a prefill that stops one token short can reach this case.
  It needs a prompt of `N` tokens whose first `N - 1` tokens fill whole
  full-attention blocks, all cached on the decode worker. If the decode worker
  lacks the state of those tokens, vLLM computes the prompt again from the hit
  that every group agrees on. Replay computes one token. With a hash block
  smaller than the full-attention block, vLLM also loads nothing when the first
  `N - 1` tokens end on a hash block boundary and are all cached, while replay
  rounds its hit down to the full-attention block and moves one attention block
  and `fixed_bytes`. Neither live run measured these cases.
- `fixed_bytes` charges every block of a sliding-window group inside the
  window, whatever the decode worker caches. vLLM moves only the window blocks
  that the decode worker does not cache. GLM-5.3-Flash has no sliding-window
  group.
- When one full-attention group is prefix-cacheable and another is not, replay
  moves every full-attention block of both groups. vLLM moves only the blocks
  that the decode worker does not hold, so it does not move the cached hit
  blocks of the prefix-cacheable group. No full-attention spec of vLLM 0.31.0,
  0.32.0 or 0.33.0 opts out of prefix caching. So no deployment of these
  versions has this shape. Only a profile that describes no such deployment
  reaches it. GLM-5.3-Flash has one full-attention group. Neither live run
  measured this case.

The decode-side differences under
[grouped caches](kv-cache.md#other-cache-modes) can also change the
transferred blocks.

<a id="link-sharing"></a>

### Link sharing

Each prefill rank sends over one link and each decode rank receives over one
link, both at `bandwidth_gb_per_second`. Transfers in flight at the same time
share these links max-min fairly: a link's bandwidth is split equally among
its transfers, and bandwidth a transfer cannot use because its other link is
the tighter one goes to that link's remaining transfers.

For example, two 1-second transfers from different prefill ranks into one
decode rank both finish after 2 seconds. Sent to two decode ranks, they both
finish after 1 second. Rates are recomputed whenever a transfer starts,
finishes or is canceled, so a transfer can speed up or slow down mid-flight.

A transfer keeps its share until its bytes arrive. Canceling a request releases
its share once the cancellation completes on both sides.

A rank is one attention-data-parallel rank of a worker. YAML gives both roles
the same bandwidth. A native ReplaySpec can set the roles'
`kv_transfer_bandwidth` separately; the slower link then limits even a transfer
alone on its links. A decode role that omits it leaves receiving links
unlimited, so only the prefill ranks' sending links are shared.

<a id="handoff"></a>

## Request flow

1. The prefill worker runs the prompt. A grouped profile with recurrent state
   layers stops the prefill one token short, as vLLM does. The prefill of a
   one-token prompt computes that token, and the decode worker computes it
   again.
2. The request is handed to a decode worker at the same virtual time. KV
   transfer starts, and the prefill worker's copy stays reserved until the
   transfer finishes.
3. After the transfer completes, the decode worker admits the request. Its
   first step produces the first token. On vLLM, that step computes the last
   prompt token. Later steps generate the remaining tokens.

The reported TTFT includes prefill queueing, prefill compute, the transfer, the
decode worker's queue and its first step.
Transfer events share one virtual clock with compute and arrivals.

Backends order the handoff differently:

| Backend | Order |
| --- | --- |
| vLLM, TensorRT-LLM | Source-first: the prefill side finishes and then the decode side reserves space. |
| SGLang | Destination-first: the decode side reserves space before the transfer. |

TensorRT-LLM decode workers use `GUARANTEED_NO_EVICT` scheduling, which reserves
room for the whole output while transferred KV is held.

The two roles may use different `attention_data` sizes. A request moves from one
concrete prefill rank to one concrete decode rank. A trace request's
`prefill_dp_rank` chooses the prefill rank; if omitted, its `dp_rank` is used
for both roles.

## Limitations

- The transfer is a single request-level event. Per-layer streaming and KV
  layout conversion between different TP shapes are not modeled.
- Links are per rank. Ranks that share a NIC or a switch uplink do not
  contend beyond their own links.
- vLLM P/D can be combined with G2 host offload; see
  [KV cache](kv-cache.md#host-offload-g2). G3 offload is aggregated-only.
