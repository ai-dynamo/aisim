<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NIXL P/D transfer bytes of GLM-5.3-Flash on vLLM 0.33.0

This run measured the bytes that vLLM's NIXL connector moves for one request
in a prefill/decode (P/D) pair on vLLM 0.33.0. It is the evidence for the
`transfer_bytes_per_block` values of the GLM-5.3-Flash test profiles on vLLM
0.33.0. It is also the evidence for the vLLM 0.33.0 rows of the oracle test
of `crates/core/src/engine/kv_manager/grouped_transfer.rs`. The run of
[`nixl-pd-transfer-bytes-20261115`](../nixl-pd-transfer-bytes-20261115/README.md)
measured vLLM 0.31.0 and 0.32.0 with the same method. `measurements.json`
holds every request of this run, and its `sources` object marks each number
as measured or derived. `region-table.json` holds the region table of the
connector, as the group log of arm C recorded it.

## Setup

- Model: `zai-org/GLM-5.3-Flash`, FP8 checkpoint, revision `04c4e9e`.
- GPUs: NVIDIA GB300. TP 4 on the prefill server and TP 4 on the decode
  server, on two nodes.
- Image: the public arm64 image `vllm/vllm-openai:v0.33.0` (vLLM 0.33.0,
  NIXL 1.5.0). No Dynamo.
- Both servers: `vllm serve <checkpoint> --tensor-parallel-size 4
  --enable-prefix-caching --kv-cache-dtype fp8 --language-model-only
  --cudagraph-metrics --max-num-batched-tokens 8192 --max-num-seqs 32
  --max-model-len 131072 --kv-transfer-config <see the arms>`, with
  `VLLM_SSM_CONV_STATE_LAYOUT=DS` and the default prefix match unit.
- The run had three arms. Each arm started new servers.

| Arm | `kv_role` of the prefill / decode server | Async scheduling | Group log |
| --- | --- | --- | --- |
| A | `kv_both` / `kv_both` | off (`--no-async-scheduling`) | no |
| B | `kv_producer` / `kv_consumer` | the vLLM 0.33.0 default, on | no |
| C | `kv_both` / `kv_both` | off (`--no-async-scheduling`) | yes |

Arm A repeats the method of the 2026-11-15 run. Arm B uses the role
`kv_producer` on the prefill server and `kv_consumer` on the decode server,
with the default async scheduling of vLLM 0.33.0. Arm C is arm A with a log-only hook
that splits the descriptors of each transfer by cache group.

## Method

A small proxy sends each request to the prefill server with `max_tokens` 1.
Then it sends the request to the decode server with the transfer parameters
that the prefill returns. It sends one request at a time. Each prompt is a list of new random
token IDs, except the repeated prompt of 5,000 tokens. The three arms used the
same token IDs for the same prompt. For each request the run reads three
counters of the decode server before and after:
`vllm:nixl_bytes_transferred_sum`, `vllm:nixl_bytes_transferred_count` and
`vllm:nixl_num_descriptors_sum`. The counters give the sum and the count of
the transfers. Every request made 4 transfers, one for each TP rank. The tables
give the mean of the 4, the bytes of one rank. NIXL counts the sum of the
lengths of the descriptors that the reader asks for, so regions that overlap
count in full.

In arm C, a `sitecustomize` module on the `PYTHONPATH` of `vllm serve` wrapped
the function that computes the descriptor IDs of a transfer
(`_compute_desc_ids`). For each transfer of each decode rank, it logged the
number of descriptors and the sum of their lengths for each cache group. It
changed no descriptor and no return value. It also wrapped
`register_kv_caches`, to log the region table after the registration, and
`_read_blocks`, to tag each record with its request.

## Result

The prefill computes N - 1 tokens of a prompt of N tokens. B is the number of
full-attention blocks of 2,176 tokens that moved: ceil((N - 1) / 2,176), minus
the blocks that the decode server already held. Arm A ran 14 requests, arm B
8 and arm C 6, 28 in all. In each arm the new prompt of 5,000 tokens ran
three times, the repeated prompt of 5,000 tokens ran once, and every other
row ran twice. The repeated runs of a row moved the same bytes. An empty cell
is a prompt that the arm did not send.

| Prompt tokens | B | Arm A, bytes | Arm B, bytes | Arm C, bytes | Descriptors (arm A) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1 | 61,146,624 | 61,146,624 | 61,146,624 | 195 |
| 2,176 | 1 | 61,146,624 | | | 195 |
| 2,177 | 1 | 61,146,624 | | | 195 |
| 5,000 | 3 | 105,103,616 | 105,103,616 | 105,103,616 | 275 |
| 5,000, the same prompt again | 1 | 61,146,624 | 61,146,624 | 61,146,624 | 195 |
| 30,000 | 14 | 346,867,072 | 346,867,072 | | 715 |
| 130,000 | 60 | 1,357,877,888 | | | 2,555 |

All 28 requests lie on one line: 21,978,496 * B + 39,168,128. This is the
line of vLLM 0.32.0, and arm A repeats every row of the vLLM 0.32.0 run, in
bytes and in descriptors.

Arm B against arm A: for each prompt, arm B moved the bytes and the
descriptors of arm A. Arm C against arm A: for each prompt, arm C moved the
bytes and the descriptors of arm A, so the group log changed no transfer.

The repeated prompt shows that the decode server reused 2 blocks of its own
prefix cache, so only 1 block moved, and that every state group still moved
its block. Only the bytes are an oracle. The descriptor count is recorded for
reference.

## The bytes of one block of each group

The connector logged its regions at start, on each of the 8 workers (the 4
TP ranks of the prefill server and the 4 TP ranks of the decode server), in
every arm: `ratio=1, num_regions=40`, 13,900 blocks, block lengths 71,808,
1,114,112 and 2,048 bytes. `region-table.json` holds the table of arm C: the
length, the cache group and the name of each region, the state regions and
the scratch regions.

The group log of arm C measured each group in each of its 24 transfers (bytes
of one rank):

| Group | Blocks that move | Bytes | Descriptors |
| --- | ---: | ---: | ---: |
| Attention (11 MLA and 11 indexer layers) | B | 21,978,496 * B | 40 * B |
| Ring (11 layers) | 1 | 92,288 | 11 |
| Each of the 4 KDA state groups (9, 9, 8 and 8 layers) | 1 | 9,768,960 | 36 |

An attention block is 21,978,496 bytes in 40 descriptors, one in each region.
The ring is 92,288 bytes in 11 descriptors, its scratch regions. Each of the
4 KDA groups is 9,768,960 bytes in 36 descriptors. In each transfer the six
groups add up to the counter of vLLM: 4 * 9,768,960 + 92,288 = 39,168,128
bytes and 4 * 36 + 11 = 155 descriptors for the fixed part. The 2026-11-15
run measured the same fixed part on vLLM 0.32.0, 39,168,128 bytes and 155
descriptors, but it derived the split among the groups, 92,288 bytes for the
ring and 9,768,960 bytes for each KDA group, from the connector code. On vLLM
0.33.0 every value of a group is measured.

## Pool

The connector registered 13,900 blocks on each worker, and each server logged
174.99 GiB of available KV cache memory. The group log of arm C recorded a
block stride of 13,516,800 bytes in every region. This is the block row that
the 2026-11-15 run read from the code of vLLM 0.32.0, and 13,900 such rows are
174.98 GiB. The transfer bytes above do not depend on the row.

## Limits

One model, TP 4, one GPU type, an fp8 KV cache, and vLLM alone. Both sides
had equal TP, the default prefix match unit and no speculative decoding. The
decode server never held every block of a prompt in this run. Arm B changes
the roles and async scheduling at the same time. It moved the bytes of arm A in
all 8 requests, and the run did not test each change alone. The values change with the TP size, because the block size and the
state sizes change with it.

One default changed. vLLM 0.33.0 selects an fp8 KV cache for GLM-5.3-Flash on
a GPU of the SM100 family, but only for a launch that names no KV cache data
type (v0.33.0 `vllm/model_executor/models/config.py:119-142` and
`:1228-1229`). vLLM 0.32.0 has no such rule for this model. Every live run of
both folders set fp8 by its `--kv-cache-dtype fp8` flag, so the default did
not change these bytes.
