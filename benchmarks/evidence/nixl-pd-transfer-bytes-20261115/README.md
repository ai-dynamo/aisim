<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NIXL P/D transfer bytes of GLM-5.3-Flash on vLLM 0.31.0 and 0.32.0

This run measured the bytes that vLLM's NIXL connector moves for one request
in a prefill/decode (P/D) pair. It is the evidence for the
`transfer_bytes_per_block` values of the GLM-5.3-Flash test profiles and for
the oracle test of `crates/core/src/engine/kv_manager/grouped_transfer.rs`.
`measurements.json` holds every request of the run.
Its `sources` objects mark each number as measured or derived.

## Setup

- Model: `zai-org/GLM-5.3-Flash`, FP8 checkpoint, revision `04c4e9e`.
- GPUs: NVIDIA GB300. TP 4 on the prefill server and TP 4 on the decode
  server, on two nodes.
- Images: the public arm64 images `vllm/vllm-openai:v0.31.0` (NIXL 1.4.1) and
  `vllm/vllm-openai:v0.32.0` (NIXL 1.5.0). No Dynamo.
- Both servers: `vllm serve <checkpoint> --tensor-parallel-size 4
  --enable-prefix-caching --kv-cache-dtype fp8 --no-async-scheduling
  --language-model-only --cudagraph-metrics --max-num-batched-tokens 8192
  --max-num-seqs 32
  --max-model-len 131072 --kv-transfer-config
  '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'`, with
  `VLLM_SSM_CONV_STATE_LAYOUT=DS` and the default prefix match unit.

## Method

A small proxy sends each request to the prefill server with `max_tokens` 1,
then to the decode server with the transfer parameters that the prefill
returns. It sends one request at a time. Each prompt is a list of new random
token IDs, except the repeated prompt of 5,000 tokens. For each request the
run reads three counters of the decode server before and after:
`vllm:nixl_bytes_transferred_sum`, `vllm:nixl_bytes_transferred_count` and
`vllm:nixl_num_descriptors_sum`. The counters give the sum and the count of
the transfers. Every request made 4 transfers, one for each TP rank. The table
gives the mean of the 4, the bytes of one rank. NIXL counts
the sum of the lengths of the descriptors that the reader asks for, so regions
that overlap count in full.

## Result

The prefill computes N - 1 tokens of a prompt of N tokens. B is the number of
full-attention blocks of 2,176 tokens that moved: ceil((N - 1) / 2,176), minus
the blocks that the decode server already held. The new prompt of 5,000
tokens ran three times, the repeated prompt of 5,000 tokens ran once, and
every other row ran twice. The repeated runs of a row moved the same bytes.

| Prompt tokens | B | vLLM 0.31.0, bytes | Descriptors | vLLM 0.32.0, bytes | Descriptors |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1 | 65,166,080 | 188 | 61,146,624 | 195 |
| 2,176 | 1 | 65,166,080 | 188 | 61,146,624 | 195 |
| 2,177 | 1 | 65,166,080 | 188 | 61,146,624 | 195 |
| 5,000 | 3 | 91,256,320 | 232 | 105,103,616 | 275 |
| 5,000, the same prompt again | 1 | 65,166,080 | 188 | 61,146,624 | 195 |
| 30,000 | 14 | 234,752,640 | 452 and 474 | 346,867,072 | 715 |
| 130,000 | 60 | 834,828,160 | 1,486 | 1,357,877,888 | 2,555 |

All rows fit one line for each version:

- vLLM 0.31.0: 13,045,120 * B + 52,120,960.
- vLLM 0.32.0: 21,978,496 * B + 39,168,128.

The repeated prompt shows that the decode server reused 2 blocks of its own
prefix cache, so only 1 block moved, and that every state group still moved
its block. NIXL joins neighbour descriptors, so the descriptor count is not
stable on vLLM 0.31.0 (452 and 474 for equal bytes). Only the bytes are an
oracle.

## The bytes of one block of each group

The connector logged its regions at start:

- vLLM 0.31.0: `ratio=34, num_regions=22`, block lengths 32,768 and 2,112
  bytes for one kernel block (1,114,112 and 71,808 bytes for one block of
  2,176 tokens).
- vLLM 0.32.0: `ratio=1, num_regions=40`, block lengths 71,808, 1,114,112 and
  2,048 bytes.

The connector reads a block of an attention group in every region, a block of
a `CircularBufferSpec` ring in its own regions only, and a block of a Mamba
group in every Mamba region, with the conv and SSM state of one layer
(36,864 + 1,048,576 = 1,085,440 bytes). The sums for one block of each group:

| Group | Blocks that move | vLLM 0.31.0 | vLLM 0.32.0 |
| --- | ---: | ---: | ---: |
| Attention (11 MLA and 11 indexer layers) | B | 11 * 1,114,112 + 11 * 71,808 = 13,045,120 | 11 * 71,808 + 19 * 1,114,112 + 10 * 2,048 = 21,978,496 |
| Ring (11 layers) | 1 | every region: 13,045,120 | 71,808 + 10 * 2,048 = 92,288 |
| Each of the 4 KDA state groups (9, 9, 8 and 8 layers) | 1 | 9 state regions: 9,768,960 | 9 state regions: 9,768,960 |

In vLLM 0.31.0 the ring is a `KpoolTailSpec`, an attention spec, so it reads
every region. The measured sums equal these constants. The split of the fixed
part among the groups is a reading of the connector code. The descriptor
counts support it: 22 * B + 166 on vLLM 0.31.0 and 40 * B + 155 on vLLM
0.32.0.

## Pool

The decode servers logged their KV cache pools. vLLM 0.31.0 held 14,351
blocks in 174.36 GiB, which fits a block row of 13,045,120 bytes, the sum of
the pages of the attention group. vLLM 0.32.0 held 13,888 blocks in 174.83
GiB, a block row between 13,515,510 and 13,517,256 bytes by the rounding of
the logged size. The code of vLLM 0.32.0 gives the row as 13,516,800 bytes:
it rounds the largest sum of the layer pages of one group, the 13,045,120
bytes of the attention group, up to the least common multiple of the
block-stride alignments of all layers. That multiple is 540,672 bytes, the
least common multiple of the 16,384-byte alignment of the MLA layers and the
4,224-byte alignment of the indexer layers (v0.32.0
`vllm/v1/core/kv_cache_utils.py:1533-1551`, `vllm/v1/worker/utils.py:476-486`
and `vllm/v1/kv_cache_interface.py:183-193`). This is a reading of the code
that agrees with the log: 13,888 rows of 13,516,800 bytes are 174.829 GiB.
The transfer bytes above do not depend on the row.

## Limits

One model, one TP size, one GPU type, equal TP on both sides, the default
prefix match unit, no speculative decoding, and vLLM alone. The decode server
never held every block of a prompt in this run. A later run measured vLLM 0.33.0: see
[`nixl-pd-transfer-bytes-20261117`](../nixl-pd-transfer-bytes-20261117/README.md).
The values change with the TP size, because the block size and the state sizes
change with it.
