<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kimi K3.1 preview runtime observation contract

This original observer reads the initialized interfaces in the official
`vllm/vllm-openai:kimi-k31` image. The immutable AMD64 image manifest is
`sha256:88175ddf4f59944ccbb24c6b60c1bc535a160be3c4f09a3b08702c5e301a599e`;
the ARM64 child of the same published image index is
`sha256:52b793567651780dc529647a211a3d15cec11ebc872ac7492636d7d75340eab4`.
Both report `0.1.dev24216+g266e88c05`. The wheel provides only the short VCS ID
`g266e88c05`; no full Git revision is asserted. The 64-character manifest
`source_revision` is SHA-256 of the canonical JSON `source_files` hash map
(sorted keys, compact separators). Every listed installed file is hashed before
import can qualify. It is not a hash of the complete repository or native binaries.
An image identity alone does not qualify ARM kernels or a model/topology.

The source is [vLLM](https://github.com/vllm-project/vllm), Apache-2.0,
Copyright contributors to the vLLM project. The manifest also pins
[`dynamo/vllm/instrumented_scheduler.py`](https://github.com/ai-dynamo/dynamo/blob/17628144dc33c9d5916b08789f9d317924217dad/components/src/dynamo/vllm/instrumented_scheduler.py),
from Dynamo commit `17628144dc33c9d5916b08789f9d317924217dad` (Apache-2.0,
NVIDIA Corporation & Affiliates). Its hash is computed from the exact Git blob,
including the original trailing newline; temporary download copies are not the
canonical source. No upstream implementation or model checkpoint
is redistributed by this adapter.

The block-table width helper in `vllm/v1/worker/block_table.py` is pinned to
SHA-256 `4068aea7336627f809def92f45401659d7c7c592d4da5dace6a11c90e7850c4a`,
computed from the exact [upstream source at
`266e88c051ca46607fd9de20216bd58426f80052`](https://github.com/vllm-project/vllm/blob/266e88c051ca46607fd9de20216bd58426f80052/vllm/v1/worker/block_table.py).
At that revision, its caller in `vllm/v1/worker/utils.py` matches the existing
installed-file pin. The installed helper must also match before observation
can qualify. This source inspection does not requalify image execution or
refresh previously collected runtime observations.

## Source-to-observation mapping

* `v1/kv_cache_interface.py`: `KVCacheTensor.layers`, `layer_stride`,
  `block_stride`, `offset`, `host_resident`, and whole allocation `size` are read
  verbatim. `AttentionSpec.unpadded_page_size_bytes` records the complete native
  attention payload independently of the padded page footprint. Direct layer
  views must cover that payload contiguously; pool accounting retains padding.
  `MambaSpec` exposes shapes, dtypes, payload bytes, padded page bytes,
  cache mode, checkpoint counts and `max_num_blocks_per_req`.
* `v1/core/kv_cache_planning.py` and `v1/worker/utils.py`: all groups share one
  byte allocation. Block-outer layouts repeat a slot for each pool block ID.
  Groups may overlay that slot because ownership is exclusive by block ID;
  distinct layers within a group must occupy disjoint intervals. The importer
  verifies actual storage, strides, offsets, coverage and bounds, and charges
  the complete pool slot including padding. Layer-outer/subdivided layouts are
  not accepted by this initial adapter.
  Native `UniformTypeKVCacheSpecs` groups preserve every per-layer spec. A mixed
  MLA/indexer group uses one block table and remains one resource group; the
  observer never splits it into separately charged attention groups.
  `v1/core/kv_cache_utils.py::generate_scheduler_kv_cache_config` projects each
  wrapper to its first native child for scheduler use. The observer retains that
  child's layer name before JSON key sorting; import compares the scheduler's
  exact representative spec and group membership while independently checking
  every worker child, aggregate page size and actual tensor allocation.
* `model_executor/layers/mamba/abstract.py::bind_kv_cache` and
  `models/kimi_k3/nvidia/kda.py`: initialized `kv_cache` is a tuple of convolution
  and recurrent tensors. The observer records each actual shape, dtype, byte
  stride and offset; the importer independently checks both components against
  the native spec and the hash-bound checkpoint's `linear_attn_config` heads,
  head dimension, convolution size and TP size. Convolution orientation follows
  the reviewed `VLLM_SSM_CONV_STATE_LAYOUT` environment (default `SD`), as in
  `model_executor/layers/mamba/mamba_utils.py::MambaStateShapeCalculator`.
  BF16/FP32 state is never relabeled as FP8 KV.
* `config/vllm.py::validate_mamba_block_size`, `MambaSpec`, and
  `v1/core/single_type_kv_cache_manager.py::MambaManager`: with prefix caching
  disabled, no speculation and mode `none`, the state block covers maximum
  context and retains one state page per request. The importer requires all
  these observed facts. `align` and prefix-cache/checkpoint recycling remain
  unsupported; its maximum allocation is not substituted for actual occupancy.
* `models/kimi_k3/common/dsa_cache.py`, `nvidia/dsa.py`,
  `model_executor/layers/attention/mla_attention.py`, and the sparse MLA backends:
  MLA and indexer retain full-history token pages. Explicit `flashmla` or
  `trtllm_mla` policy must match the observed sparse backend and FP8 indexer.
  Native pool-group block sizes remain distinct from an attention backend's
  kernel block size. `worker/gpu/attn_utils.py::init_kv_cache` binds the views
  produced by `worker/utils.py::AttentionGroup.map_kv_cache_to_kernel_blocks`.
  The initialized group's size, stride, offset tensor and block-table geometry
  are observed directly. Import checks the table width against the reviewed
  maximum context: at CP1, round up to manager blocks, align their capacity to
  128 tokens (`v1/worker/block_table.py::get_block_table_width`), then multiply
  by the number of kernel blocks per manager block. For a packed indexer,
  a manager block larger than 64 tokens requires the observed subdivision map.
  The table's leading dimension equals the effective metadata-builder count
  used by `gpu/attn_utils.py::init_attn_backend`: one with microbatching disabled,
  otherwise the resolved `parallel_config.num_ubatches`. Both microbatching
  settings are recorded alongside the initialized cache evidence. For this map,
  kernel page `j` of manager block
  `b` is `b * kernel_block_stride + j`; import verifies that each addressed page
  lies in the layer's native allocation interval. The tensor spans intervening
  layers' pages but does not own them, so that span is never charged as another
  allocation. `KimiK3IndexerMetadataBuilder` uses
  `attention/backends/mla/compressor_utils.py::get_compressed_slot_mapping`
  to write from those kernel block tables. Unmapped MLA and recurrent state
  views retain their direct manager-block geometry checks.
* `quantization/mxfp4.py`, `fused_moe/{routed_experts,config,modular_kernel}.py`,
  `fused_moe/oracle/mxfp4.py`, and `models/kimi_k3/nvidia/model.py`: enumerate
  initialized native `KimiMoE` and `RoutedExperts`, require one routed expert
  module per MoE, and record post-load method/kernel classes, backend, both
  activation descriptors and both weight descriptors. The initial mapping is
  only `FLASHINFER_TRTLLM_MXFP4_MXFP8`, both activations `mxfp8` and both weights
  `mxfp4`, to `w4a8_mxfp4_mxfp8_trtllm`. Missing, mixed, BF16 or other backend
  evidence is rejected. Linear layers remain the native unquantized base type.
  A checkpoint's MXFP4 declaration does not establish expert activation dtype.
  Import independently checks the expert-layer indices against the hash-bound
  checkpoint's decoder configuration (`num_hidden_layers`, `num_experts`,
  `first_k_dense_replace`, and `moe_layer_freq`), using the pinned native dense
  prefix and frequency defaults of 0 and 1. The current PP1 restriction means
  each worker must cover that complete inventory; matching incomplete model-tree
  lists cannot establish precision. The referenced local model config must
  remain available for this check when observations are imported or revalidated.
* `gpu_worker.py`, both model runners, and the pinned Dynamo scheduler preserve
  the existing post-profile, post-cache-initialization and post-warmup hooks.
  Native exceptions propagate. Probe evidence records actual initialized state;
  it does not claim per-point kernel dispatch, timing accuracy or full coverage.

## Collection scope

Use this explicit example bundle with a fresh campaign and the normal
`aisimulate onboard probe-runtime` workflow. Default release bundles are unchanged.
The baseline supports independent decode and zero-history prefill workers, with
no speculation, prefix caching off and Mamba `none`. Each role's Dynamo
`native_exact_context` path must be qualified on the intended runtime. Historical-KV
prefill seeding uses prefix-cache lookup and is not qualified by this baseline;
aggregated serving also remains outside it. No timing data is produced by source
inspection or unit-test fakes. Rank-complete native probe evidence is required
for each role before import.
