<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.8-Flash-Next text modeling

`Qwen/Qwen3.8-Flash-Next` (`architectures: ["Qwen4ExpForConditionalGeneration"]`)
uses the pinned checkpoint configuration
[`de4b8e4d43b917e7706784d8bb445c9af86a3540`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json);
the FP8 release is the same geometry at
[`236dfdf285828023ca3bcd3f37366c58a3469b13`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/236dfdf285828023ca3bcd3f37366c58a3469b13/config.json).
Both are multimodal: the decoder fields live under `text_config`, and the
packaged [provenance note](../../../python/aisim/src/aisim_core/model_configs/QWEN38_FLASH_NEXT_PROVENANCE.md)
records the upstream revisions and the serving facts read from the pinned
runtimes.

The text decoder is the modeling and collection target. `vision_config` is
expanded into the same encoder op graph as Qwen3.5 and Qwen3-VL so multimodal
requests are not silently unpriced, but the tower is not measured here.

## Decoder structure

48 decoder layers: 36 GDN linear-attention layers and 12 QSA compressed
sparse-attention layers (`full_attention_interval: 4`, indices 3, 7, …, 47),
one n-gram PLE branch at layer 2, 512 routed experts with top-10 routing and a
640-wide shared expert, and a one-layer QSA MTP draft. Three structural families
are priced explicitly rather than folded into the Qwen3.5 template:

- **QSA replaces GQA on full-attention layers.** Qwen Sparse Attention is a
  block-sparse GQA: an indexer scores KV blocks and full attention runs over the
  top blocks. AISimulate models it through the MSA module op with
  `index_topk = block_topk * block_size`; it is not modeled as dense attention
  with a shortened context. Geometry is 24 query / 2 KV heads at 256 dims with
  a doubled `q_proj` (sigmoid output gate) and a 4-query / 1-KV indexer at 128
  dims.
- **Gated residual ("hyper connection").** Each attention and MLP block is
  preceded by a 4-way residual mixer over a `hc_count * hidden_size` = 10240
  manifold: an RMS norm, a low-rank down/up pair and a small injection GEMM.
  These replace `input_layernorm`/`post_attention_layernorm`, which the
  checkpoint does not contain.
- **N-gram PLE.** A 20M-row × 2560-wide embedding table (51.2e9 parameters) plus
  a merged `kv_proj`, a depthwise `conv1d` and group norms. See
  [memory](#memory-and-residency).

## Backends and precision

The analytical graph is built for SGLang and vLLM. TensorRT-LLM leaves the
model out of every plan in
[`collector/skips.py`](../../../python/aisim/collector/skips.py): neither the
pinned TRT-LLM nor the checkpoint has a QSA attention path, so there is no
silicon-true row to collect. NVFP4 exists upstream as a ModelOpt derivative but
requires Blackwell; no SM100+ system is qualified for this model, so the
checkpoint is not registered.

The FP8 release quantizes **routed experts only**. Its
`modules_to_not_convert` covers `self_attn.*`, `indexer`, `linear_attn.*`,
`mlp.shared_expert.*`, `mlp.gate*`, the attention/MLP hyper-connections, `ple.*`,
`visual.*`, `mtp.*`, `lm_head` and `embed_tokens`. QSA, its KV pool and the
indexer therefore run in BF16 under both artifacts. The SDK keeps KV and FMHA
quantization in BF16 for this architecture even though the checkpoint reports
the model-wide `fp8_block` mode; an explicit `kv_cache_quant_algo` or
`kvcache_quant_mode` still wins and fails loudly when the data are missing.
Consequently the QSA module rows carry exactly one precision triple
(`compute`, `kv_cache`, `gemm` = BF16) and serve both artifacts.

## QSA collection

QSA dispatches inside the existing MSA module op, so its rows live in
`msa_context_module_perf` / `msa_generation_module_perf` next to MiniMax-M3's.
The persisted `architecture` column keys them apart, and the Rust `MsaModuleOp`
selects the row by that field rather than assuming M3.

A module row is one phase and head shard for one checkpoint, with every
`(batch, seq_len, prefix_len)` point swept inside it:

- **Head axis.** The module is collected at one TP rank's shard — 24, 12, 6 and
  3 query heads at TP 1/2/4/8, from the row's declared
  `tensor_parallel_sizes`. The shared MSA/DSA head grid carries 24 and 12 but
  not 6 or 3, so the declared shard axis supplies the counts the grid lacks;
  the shared grid keeps its own order for the rest.
- **Compressed blocks.** The checkpoint declares `indexer_compress_ratio: 4`:
  every context prefix must be a multiple of the compressed-block width, because
  the extend write plan asserts `prefix % ratio == 0` on the device and a
  violation poisons the CUDA context for the rest of the sweep. The op plan
  raises each declared anchor to a group boundary (deduping collisions such as
  prefix 1 → 4) and drops a raised point that would cross the rotary-table
  bound rather than moving it. The plan never depends on the device.
- **Serving geometry.** QSA pins a 64-token page
  (`arg_groups/model_overrides/qwen4_exp.py@v0.5.21`), not M3's 128.
- **Capacity.** The prefixed extend path keeps the case's KV pool resident and
  allocates a per-shape gather workspace on top of it. Bounding the pool alone
  admitted 215 of 11888 shapes that OOM'd mid-sweep on H200, so the worker
  refuses a point whose pool plus workspace cannot fit, before allocating it,
  with a `FIXME(kernel-limit): does not fit` reason. The remaining shapes still
  measure; the refusal is recorded, not silently dropped.

Inspect the retained plan before a GPU run:

```bash
python3 collector/collect.py --backend sglang \
  --model-path Qwen/Qwen3.8-Flash-Next \
  --ops msa_context_module msa_generation_module --sm 90 --plan-only
```

On H200 that reports 2192 context cases in 36 rows (one per batch × head
shard) and 736 generation cases in 4 rows (one per head shard, batch 0), all
BF16. `--sm` is what makes the op plans print; without it the command only
resolves the case plan.

## Memory and residency

The n-gram table is priced as **HBM residency sharded over TP**
(`num_embeddings / tp` rows per rank), which is the only residency model
AISimulate implements for n-gram tables. Both runtimes default differently, so a
collection or serving run must pin the flag:

| Runtime | Flag |
| --- | --- |
| vLLM | `--engram-config '{"cpu_offload": false}'` |
| SGLang | `--no-ple-offload-embedding` |

Leaving the defaults keeps the whole 102.4 GB BF16 table in pinned host memory
and breaks the memory identity between model and process.

Sharding is deliberately uneven and was read off the pinned runtime source. The
n-gram table is a `VocabParallelEmbedding`, but the gated-residual projections,
the PLE merged `kv_proj`, its `conv1d` and its norms are replicated on every rank
(`ReplicatedLinear` / `MergedColumnParallelLinear(disable_tp=True)` in vLLM,
plain `nn.Linear` in SGLang). Treating them as TP-sharded would under-report
per-rank memory by ~1.2 GB and under-predict mixer compute by the TP factor.
Backend activation coefficients remain heuristic; runtime allocator
measurements are still required to qualify capacity.

## Result provenance

The campaign measured `h200_sxm` QSA-module, GEMM, MoE and GDN rows for BF16
and FP8 on both backends. They are staged under
[`benchmarks/evidence/collector/qwen38-c3.5/`](../../../benchmarks/evidence/collector/qwen38-c3.5/README.md)
and are not in this tree: operation data lives in the pinned
`ai-dynamo/aisim-data` dataset. Paths differ by backend because the rows are
keyed by the version they were collected on:

- SGLang 0.5.21 is a queryable version (`previous`), so once a dataset revision
  carries the rows and
  [`op_latency_pin.json`](../../../python/aisim/src/aisim_core/data/op_latency_pin.json)
  is raised to it, SILICON resolves for BF16 and FP8 and HYBRID combines those
  measurements with SOL contributions.
- vLLM 0.31.0 is outside the dataset's slots (`current` 0.32.0, `previous`
  0.30.0), so publishing the rows is not enough: a query at 0.31.0 still
  raises `UnlistedQueryVersionError` unless the caller sets
  `AIC_ALLOW_UNLISTED_VERSIONS=1`, until 0.31.0 is promoted into a slot or the
  lane is recollected on 0.32.0.

Until those conditions hold SILICON and HYBRID raise the typed data miss, and
EMPIRICAL fails closed for the same reason: there is no measured QSA utility,
and the M3-calibrated DSA transfer must not be borrowed for another
architecture (`operators/msa.rs` enforces it).

The QSA module prices the checkpoint's fused halves itself: the MSA op's
projection formula (Q, K+V, O, indexer-Q) carries two extra widths
(`output_gate_dim` / `indexer_k_dim`) for QSA's sigmoid output gate, which
shares `q_proj`'s weight, and the indexer K half of the fused `index_qk_proj`.
Both are charged inside the module the collector measures, so SILICON and
HYBRID — which return that measured latency — do not double-charge them. The
analytical indexer path still assumes an FP8 indexer; this checkpoint keeps it
in BF16, so the analytical graph alone under-predicts index time and the module
rows are the measured anchor.

The C3.5 evidence bundle
([`benchmarks/evidence/collector/qwen38-c3.5/`](../../../benchmarks/evidence/collector/qwen38-c3.5/README.md))
records the staged tables, the disjoint merge check, the architecture-keyed
consumer queries (TP8/EP8, one 8192-token request: vLLM 144.09 ms prefill /
5.85 ms decode, SGLang 162.71 ms / 5.61 ms) and a default `aisimulate predict`
smoke. Those are plumbing checks, not accuracy results. The tables reach a
default install only through an `ai-dynamo/aisim-data` revision and the pin bump
above.

## Collection boundary

Collection covered the SGLang 0.5.21 and vLLM 0.31.0 QSA module, GEMM, MoE and
GDN rows on H200/sm90 in BF16 — which is also the FP8 checkpoint's attention and
KV dtype — and merged them into a staged table set without re-measuring any
existing cell. The merge is a candidate bundle, not an in-tree edit: after
`#857` the shipped operation tables come from the pinned dataset. Keep the
checkpoint/config hash, image digests, resolved runtime settings (including the
residency flags above) and the accepted sidecars with the data.

Three things remain separate from that coverage:

- **Serving accuracy** is not established. The first FPM cell and the 8-GPU
  serve-versus-replay comparison are separate stages; loading the checkpoint,
  merging rows, or passing a consumer lookup does not qualify timing.
- **FP8 routed-expert precision** shares the BF16 QSA rows by construction; no
  FP8 attention row exists or is expected.
- **Power** is unavailable for this campaign (insufficient energy coverage), so
  no power claim is made.

The [self-service guide](../fpm-self-service/implementation.md) owns campaign,
transport, validation and publication steps.
