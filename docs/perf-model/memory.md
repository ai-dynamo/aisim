<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GPU memory and cache capacity

## Engine context limits

`EngineConfig.max_model_len` is an optional positive prompt-plus-output token
limit for vLLM, TRT-LLM, and SGLang. Recipe adapters can map TRT-LLM
`max_seq_len` and SGLang `context_length` to this field.

- A prompt at or above the limit is rejected before prefill computation.
- A decode destination rejects such a prompt before queuing the handoff or
  reserving KV blocks, even when destination admission is deferred.
- Terminal rejection removes the request's outstanding Belady input demand
  before subsequent cache admission and eviction.
- Generation stops when prompt plus output reaches the limit, including
  speculative bursts and requests with explicit output token IDs. Reports
  retain the requested output length and count only tokens actually generated.
- TRT-LLM completion reservations and SGLang output reservations use the capped
  output budget. Physical KV capacity remains a separate constraint.
- The limit applies to aggregated workers and both roles in disaggregated replay.
  An unset limit preserves the existing backend behavior.

The prediction compiler and recommendation deployment builder pass explicit
`engine.context_length` values to every backend. Prediction with
`context_length: max` retains its existing behavior: vLLM resolves model
metadata, while TRT-LLM and SGLang leave the scheduler limit unset.

This is the simulator's normalized context-limit contract. It does not model
backend-version-specific frontend validation margins or automatic prompt
truncation. Successful replay alone does not establish silicon timing accuracy.
See the [SGLang GPU parity observations](https://github.com/ai-dynamo/aisim/pull/261#pullrequestreview-5250881045)
for stricter boundary behavior; they do not establish a fixed token offset
across versions or configurations.

## KV-cache capacity reservation

For native SDK estimates, `estimate_kv_cache`, `estimate_num_gpu_blocks`, and
`KVCacheEstimator.from_request` resolve an omitted `backend_version` (or `None`)
through the backend's `current` data alias for the requested system and data
roots. Explicit version selectors are preserved. FPM profile estimates still
require the profile's literal backend version.

For Mistral3's gated Pixtral encoder, the analytical activation estimate includes
the replicated residual alongside the gate and up intermediates. Encoder tensor
parallelism shards those intermediates; encoder data parallelism keeps a full
tower on each rank. This remains an analytical lower bound, not a measured peak.

SGLang's native estimator treats `mem_fraction_static` as a static weights/KV
pool. Peak activation/workspace estimates remain visible in
`memory_breakdown.activations_bytes`, but are not deducted from that pool;
transient execution headroom is already outside the static fraction. Increasing
the prefill token budget alone therefore does not reduce SGLang KV capacity.
Resident runtime/communication estimates reduce the pre-load free-memory pool
before applying the fraction; weights are deducted afterward. The budget is
`(capacity - resident_overhead) * mem_fraction_static - weights`, less any
explicit additional graph reservation. For ordinary SGLang DeepSeek-V3/R1
(non-CP, non-PP, non-speculative, non-large-EP), the estimator also respects
checkpoint dense/MoE layer counts and TP-sharded embeddings independently of
the unchanged timing graph. Other model layouts retain their prior weight
accounting.
vLLM and TRT-LLM continue to deduct activation memory under their own budget
semantics. No measured server capacity is required by this calculation.

`estimate_kv_cache` and `estimate_num_gpu_blocks` accept
`cuda_graph_reserved_bytes=<rank-local bytes>`. The value must be a
non-negative integer no greater than `2**53` and defaults to zero. It is treated
as fixed non-KV memory before the backend-specific KV fraction is applied. For
SGLang, it is an additional reservation beyond graph/runtime headroom already
encoded by `mem_fraction_static`. Native estimates return it as
`memory_breakdown.cuda_graph_reserved_bytes`; naive fallback estimates apply it
but keep `memory_breakdown=None` because the other components are unavailable.

The Rust `KvCacheEstimateRequest` exposes the same field. Engine replay accepts
the value in its engine arguments and carries it through native AIC capacity
rematerialization, so the Python estimator and native scheduler use the same
rank-local KV capacity. The `aisim predict` YAML exposes it at
`engine.workers.<role>.kv_cache.capacity.cuda_graph_reserved_bytes`. This API
does not estimate the reservation; callers must supply a value from a source
they trust.

Serialized Rust requests and estimates that omit the field remain compatible
because deserialization defaults it to zero. See [Rust literal migration](../aic-backward-compatibility/migration.md#rust-resource-and-memory-literals)
for required source changes to request and memory-breakdown structs.

### FPM profile cache groups and byte budgets

FPM resources support `cache_layout: linear` with a positive
`kv_bytes_per_token`, or `cache_layout: grouped` with a nonempty `cache_groups`
list and no scalar token rate. Existing linear profiles keep their serialized
shape. Python exports `FpmCacheGroup` from `aisim.fpm_profile` and
`aisim_core.fpm_profile`; Rust exports it from `aisim_core::perfmodel`.

The resource fields `max_num_tokens` and `max_batch_size` are positive
unsigned 32-bit integers, from 1 through 4,294,967,295. Each limits one
attention-DP rank; their bounds match the native scheduler resource type.

An explicit profile can use `cache_layout: unmodeled` when the runtime cache
allocator has no AIS model, such as Mamba checkpoint retention.
This layout requires observed `runtime_memory` and omits both
`kv_bytes_per_token` and `cache_groups`. Store the native cache descriptors in
the referenced provenance artifact. Canonical standalone timing queries can
consume this profile. Cache sizing and replay reject it. An observed allocation
does not establish how request lengths, cache reuse, or speculation consume it.

Memory can be pending, declared, or observed at runtime. The four legacy fields
`weights_bytes`, `activations_bytes`, `runtime_overhead_bytes`, and
`comm_overhead_bytes` are optional. A complete set retains the existing non-KV
budget calculation. A partial set is planning evidence only: missing values are
not zero, and cache sizing or replay fails with an instruction to finalize
runtime memory. Schema validation, collection planning, and direct timing queries
can use a pending profile. Omitted fields remain absent from serialized output.

Alternatively, `resources.runtime_memory` records the selected deployment's
initialized cache allocation:

```json
{
  "kv_cache_bytes": 1073741824,
  "gpu_memory_utilization": 0.9,
  "max_model_len": 32768,
  "provenance": "Verified collection worker initialization; see saved evidence."
}
```

The byte count is a positive integer no greater than `2**53`; utilization is
finite and in `(0, 1]`; context and provenance are required. Runtime memory
cannot coexist with any of the four legacy non-KV fields. Python exposes
`FpmRuntimeMemoryProfile` and resource properties `memory_source`
(`pending`, `declared`, or `runtime`) and `memory_ready`; `require_memory()`
rejects pending resources. Rust exposes `FpmRuntimeMemoryConfig` and
`FpmResourceConfig::require_memory()`. See [Rust literal migration](../aic-backward-compatibility/migration.md#rust-resource-and-memory-literals)
for resource construction changes.

A runtime memory record can also keep the prefix-caching settings of the
[runtime probe](fpm-self-service/implementation.md#resolve-cache-geometry-with-a-runtime-probe)
that observed it: `enable_prefix_caching` (a boolean), `mamba_cache_mode`
(`none` or `align`) and `hash_block_size_tokens` (the positive hash block size,
in tokens, that the runtime resolved). All three are optional, and an absent
field is left out of the serialized record. `align` requires
`enable_prefix_caching` to be `true`. vLLM's `all` mode is not modeled and is
rejected. Rust exports the mode type as `FpmMambaCacheMode`. These fields are
evidence of the probe's settings. Selection does not read them. Sizing reads
only `mamba_cache_mode`, and only for the peak bound of a `state` group
(`request_peak_cache_bytes`).
Grouped prefix caching needs `enable_prefix_caching: true` and
`hash_block_size_tokens`, and a prefix-cacheable `state` group also needs
`mamba_cache_mode: align`. Replay then hashes prompts in blocks of
`hash_block_size_tokens`. Grouped prefix caching is modeled for vLLM 0.31.0,
0.32.0 and 0.33.0, and the version of the evidence, `runtime_framework_version`
below, must equal the deployment's `backend_version`. So the plan of a profile
that records vLLM 0.33.0 evidence under the label `0.33.0` can replay with
prefix caching, and a configuration that turns prefix caching on for such evidence
under another label is refused. [Grouped caches](../replay/engine/kv-cache.md#other-cache-modes)
also gives the case of a dataset label that differs from the probed vLLM
version and of a profile imported before that field existed. For a linear profile, the simulated deployment decides prefix
caching. The vLLM 0.33.0 runtime
probe fills these fields from the prefill phase of an aggregated or prefill
role. A decode role takes them from its decode phase.

The import of a vLLM 0.33.0 runtime probe record also writes
`runtime_framework_version`: the version of the runtime that observed the cache
groups and the prefix-caching settings, such as `0.33.0`. It is optional,
nonempty text, and an absent field is left out of the serialized record. It
can differ from the deployment's `backend_version`, because
`onboard probe-runtime` keeps an explicit dataset label and probes the runtime
of the image. The import of a record without hybrid evidence, a vLLM 0.27.0,
0.28.0 or 0.32.0 record, does not write it,
and the linear layout of a measured cache leaves it out with the groups.
Sizing does not read it. Grouped prefix caching reads it: the onboarding plan
turns prefix caching on only when it equals the `backend_version`, and
prediction and recommendation refuse prefix caching when it differs. A profile
imported before this field existed has none; see the [migration guide](../aic-backward-compatibility/migration.md#rust-resource-and-memory-literals).

Each group has a unique `name`, `kind` (`attention`, `convolution` or `state`),
positive `num_layers`, `block_size_tokens`, and `page_size_bytes`, and an optional
positive `sliding_window`. An omitted or null window retains full history;
convolution groups require a window. In an attention or convolution group,
`page_size_bytes` is the **rank-local aggregate for all layers in the group**,
including runtime padding. Do not
multiply it by `num_layers` again. Runtime block sizes and padding are deployment
inputs; model geometry alone does not establish them. See the
[grouped-profile review workflow](fpm-self-service/implementation.md#review-grouped-cache-resources).

A `state` group holds fixed per-request state, such as the recurrent state of
linear-attention layers. It has no `sliding_window`, and no other kind accepts
`blocks_per_request`. Three fields have their own meaning in a `state` group:

- `page_size_bytes` is one pool block: every byte that one block ID reserves in
  the shared pool. This can be more than the bytes of the group's own layers.
  vLLM pads each Mamba page to the attention page, and the group with the most
  bytes per block sets the size of a pool block.
- `block_size_tokens` is the block size of vLLM's state spec, as the runtime
  probe records it. For a Mamba spec, that is the Mamba block: the scheduler
  block in `align` mode and `max_model_len` in mode `none`. For a ring spec,
  such as an indexer tail, it is the ring block. Sizing and replay with
  prefix caching off do not read this field. Grouped prefix caching reads it
  as the Mamba block of `align` mode, and it takes the least common multiple
  of the blocks of all groups as its scheduler block. The field is part of the group
  shape that onboarding compares with the runtime probe.
- `blocks_per_request` is the positive number of pool blocks that a running
  request holds between block boundaries. For a Mamba spec, that is `1 + k`,
  where `k` is the spec's `num_speculative_blocks`. Grouped execution forbids
  speculative decoding, so `k` is 0 there. A ring spec, such as
  `CircularBufferSpec` in vLLM 0.33.0, holds 1 block for the whole life of
  the request in every mode. The count does not change with
  the request length. When a request crosses a block boundary in `align` mode,
  a Mamba group also holds the previous block for one or two steps. vLLM's
  startup sizing in that mode counts `2 + k + c` blocks per request for a
  Mamba group, where `c` is the spec's `num_prefill_checkpoint_blocks`. It
  counts 1 block for a ring group. See vLLM v0.33.0
  [`kv_cache_interface.py:1135-1141`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/kv_cache_interface.py#L1135-L1141)
  and
  [`single_type_kv_cache_manager.py:1640-1657`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/single_type_kv_cache_manager.py#L1640-L1657).
  `request_peak_cache_bytes` counts these `2 + k + c` blocks for a
  prefix-cacheable `state` group of a profile that records `align` mode, and
  `request_occupancy_cache_bytes` counts `blocks_per_request`.

A `state` group can also hold two optional fields from the runtime probe
import. A profile imported before these fields existed has neither of them, and
an absent field is left out of the serialized group. Replay with prefix
caching off does not read them. Sizing reads only `prefill_checkpoint_blocks`,
and only for the peak bound of a prefix-cacheable `state` group in `align`
mode. Grouped prefix caching on vLLM 0.32.0
and 0.33.0 reads them for a prefix-cacheable `state` group, and on vLLM 0.31.0 it refuses
a positive `prefill_checkpoint_blocks`. On every modeled version it refuses a
positive `prefill_checkpoint_blocks` on a `state` group with
`prefix_cacheable: false`.

- `prefill_checkpoint_blocks` is the number of prefill checkpoint blocks of a
  Mamba spec: 0 or 1. Only a `state` group can have it. A vLLM 0.33.0 import
  writes it for every Mamba group. It refuses a record with another count.
  In `align` mode, vLLM's startup sizing counts these blocks as `c` in
  `2 + k + c`.
- `prefill_checkpoint_alignment` is the token alignment of that checkpoint. It
  must be positive, and it needs `prefill_checkpoint_blocks` of 1. A vLLM
  0.33.0 import writes it for a Mamba group with one checkpoint block. It
  refuses a record whose alignment does not match the count. In vLLM
  0.33.0, a GLM-5.3-Flash KDA layer that prefills with FlashKDA has one
  checkpoint block with an alignment of 16 tokens. With the Triton kernel it
  has no checkpoint block and no alignment. See vLLM v0.33.0
  [`kda.py:168-176`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/models/glm5next/common/kda.py#L168-L176)
  and
  [`kda_checkpoint.py:9-10`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/model_executor/layers/mamba/kda_checkpoint.py#L9-L10).

Every group can also hold two more optional fields. A profile imported before
these fields existed has neither of them, and an absent field is left out of
the serialized group. The [migration guide](../aic-backward-compatibility/migration.md#rust-resource-and-memory-literals)
says how a session that was imported before them keeps working. Sizing reads
only `prefix_cacheable`, and only for the peak bound of a `state` group.
Aggregated replay with prefix caching off reads neither.
Grouped prefix caching reads `prefix_cacheable`. Aggregated replay does not
read `transfer_bytes_per_block`. The grouped P/D transfer rules read both
fields, so a prefill or decode rank refuses a group that leaves either unset.
So a model with several state groups can run grouped P/D only with evidence
for the transfer bytes of each of these groups.

- `prefix_cacheable` tells whether vLLM's manager for the group takes part in
  prefix caching. A ring spec (`CircularBufferSpec`) does not, and a uniform
  group takes part only when every inner spec does. The import of a vLLM
  0.33.0 record writes it for every group except a group that holds a
  `SlidingWindowMLASpec` layer. Such a group
  takes no part if it replays a bounded window, and the probe does not record
  that setting. See vLLM v0.33.0
  [`kv_cache_interface.py:998-1000`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/kv_cache_interface.py#L998-L1000),
  [`:1027-1029`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/kv_cache_interface.py#L1027-L1029)
  and
  [`:1305-1307`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/kv_cache_interface.py#L1305-L1307).
- `transfer_bytes_per_block` is the bytes that vLLM's P/D connector (NIXL)
  reads for one block of the group in a transfer on one TP rank, for the vLLM
  version of the deployment. It is positive and at most 2^53, and it can be
  larger than `page_size_bytes`, because the connector reads a block in regions
  that layers of other groups can share. A hand-made profile may state it only
  for a group whose transfer size has been measured or identified by other
  evidence for that group. The import of a vLLM 0.33.0 probe record writes it
  only from a probe phase with prefix caching and only for the measured cache
  shape, which the paragraphs below describe.

  To measure the full-attention value and the combined transfer size of the
  other groups, send two prompts of different length through a P/D pair, one at
  a time, and read the change of
  `vllm:nixl_bytes_transferred_sum` on the decode server for each. Divide each
  change by the change of `vllm:nixl_bytes_transferred_count`, because each TP
  rank makes its own transfer and the field holds the bytes of one rank. The
  difference between the two results, divided by the difference in
  full-attention blocks that move, is the value of the full-attention group.
  The rest of one transfer is the sum of the values of the other groups when
  each of them moves one block, as a state group does. With multiple state
  groups, this determines only their combined transfer size.

  This procedure does not give the value of each group for a model with
  several state groups. For a cache outside the measured shape, for a record of
  that shape from a probe phase without prefix caching, and for a profile of
  vLLM 0.31.0 or 0.32.0 without evidence for each group, leave
  `transfer_bytes_per_block` unset for those groups until per-group transfer
  measurements, additional runtime-observer evidence or another source, such
  as the regions that the connector logs at start, read with the connector
  code, identifies the bytes read for each group on one TP rank. Do not split
  the combined size evenly or use the page sizes as substitutes.

  The probe import computes the field from the worker records of a vLLM 0.33.0
  probe, with the region rule of the connector. In the measured shape, the
  connector registers one region for each distinct page address of a layer, in
  the order of the allocation plan, and the length of a region is the page of
  its layer. Layers of different groups can share a region, and an MLA layer
  then sets its length.
  For one block, the MLA attention group reads every region. A ring group
  reads the regions of its own layers. A Mamba group reads its state bytes
  without padding in every region of a Mamba layer. See vLLM v0.33.0
  [`base_worker.py:1744-1766`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L1744-L1766),
  [`:1774-1786`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L1774-L1786),
  [`:237-278`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L237-L278)
  and
  [`ssm_conv_transfer_utils.py:155-156`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py#L155-L156).

  The import writes the field only for a record that meets all of these
  conditions. They describe the structure of the cache. They do not name the
  TP size, the GPU type or the KV cache data type.

  1. It is a vLLM 0.33.0 record with hybrid evidence.
  2. One storage holds every block row, and every allocation uses the same
     block stride.
  3. The groups hold one or more Mamba groups and one or more ring groups.
  4. The cache holds one or more MLA attention layers, every layer outside the
     Mamba groups and the ring groups is such a layer, and no group holds a
     sliding window.
  5. Every Mamba group holds two or more layers, and all Mamba layers hold the
     same two state tensors.
  6. The record comes from a probe phase with prefix caching on.

  A probe can hold two phases and several TP ranks. The import computes the
  value from each worker record of each phase with prefix caching on, and a
  record of a phase with prefix caching off does not change it. It writes the
  value only when every record of a phase with prefix caching meets the
  conditions and all of them give the same value. For every other record it
  writes no value. A record without a value never makes an import incomplete,
  and the result of a complete import names the reason in
  `transfer_bytes_reason`. A record that does not meet condition 1 gives that
  condition first. When the records meet condition 1 and every phase of the
  import records prefix caching off, the reason is condition 6. Otherwise it is
  the first condition that the record does not meet, a field of the rule that
  the record lacks or holds in a form that the import does not know, or worker
  records that give different values.

  The record does not hold four facts that the connector reads. A reading of
  the vLLM source shows that the value does not depend on them inside the
  conditions:

  - The ratio between the manager block and the kernel block of the
    connector. Storage of whole block rows starts only with a ratio of 1. vLLM
    never splits the block of a group whose layers share block rows.
  - The `tp_replicated` flag of a Mamba layer. A ring model accepts a
    `tp_replicated` state only in a group of one layer. Every Mamba group of
    the conditions holds two or more layers.
  - The layout that the connector asks for, which follows `use_mla`. The
    connector asks for another layout only for a model without MLA.
  - The `mamba_type` of a Mamba layer. The connector starts only for Mamba1,
    Mamba2 and GDN layers. For each of them it reads the two state tensors
    whole. For Mamba2 this holds because vLLM makes the TP size divide the
    group columns; see vLLM v0.33.0
    [`mamba_mixer2.py:320-341`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/model_executor/layers/mamba/mamba_mixer2.py#L320-L341)
    and
    [`mamba_utils.py:197-205`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/model_executor/layers/mamba/mamba_utils.py#L197-L205).

  The connector orders the regions of a ring model by the manager views. vLLM
  builds these views in the order of the allocation plan, which the record
  holds. See vLLM v0.33.0
  [`base_worker.py:1611-1615`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L1611-L1615),
  [`:1670-1672`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L1670-L1672),
  [`:1720-1730`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L1720-L1730),
  [`:680-697`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L680-L697),
  [`:1573-1580`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L1573-L1580),
  [`utils.py:598-627`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/utils.py#L598-L627),
  [`:663-675`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/utils.py#L663-L675),
  [`connector.py:148-166`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py#L148-L166),
  [`ssm_conv_transfer_utils.py:127-136`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py#L127-L136)
  and
  [`:155-208`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py#L155-L208).

  A live run measured the values of GLM-5.3-Flash at TP 4 with an fp8 KV cache
  on GB300: 21,978,496 bytes for the attention group, 92,288 for the ring and
  9,768,960 for each KDA state group. The import of a probe record of the same
  model, TP size, data type and GPU type gives these values. Its region table
  equals the table that the connector logged. See
  [`nixl-pd-transfer-bytes-20261117`](../../benchmarks/evidence/nixl-pd-transfer-bytes-20261117/README.md).
  For another TP size or KV cache data type of the same cache shape, the
  import computes the value from that record by the same rule. No run
  measured such a value. The value describes the probe launch with the
  connector added and no other change.

  A claim in `resource_overrides.cache_groups` that states this field must
  state the value of the import. If the probe evidence holds no value, the
  claim is refused, because it has no value to compare with.

Use `RustForwardPassPerfModel.estimate_cache_budget(config, budget)` with the
same canonical `ForwardPassPerfModelConfig` used for timing. Rust exposes
`ForwardPassPerfModelConfig::estimate_cache_budget(&request)` and
`ForwardPassPerfModel::estimate_cache_budget(&request)`, taking
`FpmCacheBudgetRequest` and returning `FpmCacheBudget`. The config method checks
the exact profile deployment and scheduler envelope without constructing a
timing model, loading timings, or building an operation graph. Grouped planning
requires the native extension; it does not require GPU access or measured data.

| Budget field | Meaning |
| --- | --- |
| `total_gpu_capacity_bytes` | Positive rank-local GPU capacity. |
| `memory_fraction_kind`, `memory_fraction_value` | `of_total` and the fraction of GPU memory available to the worker. |
| `max_num_tokens`, `max_batch_size` | Rank-local scheduler limits in `[1, 4294967295]` within a declared resource envelope with the same bounds; exact recorded values for runtime memory. |
| `context_length` | Optional positive request bound, at most the profile context and runtime `max_model_len` when present; defaults to the profile context. |
| `request_occupancy_tokens` | Optional positive logical request length within the selected context, for a separate steady decode footprint with one new token per forward pass. |
| `cuda_graph_reserved_bytes` | Separate nonnegative reservation for declared memory; must be zero for runtime memory. |
| `tolerance_fraction` | Optional fraction in `[0, 1)` deducted from the resulting cache byte budget. |

The result includes `total_kv_size_bytes`, `memory_breakdown`,
`resource_provenance`, `cache_layout`, `cache_groups`, and
`request_peak_cache_bytes`. The latter is a single-request bound, not a
reservation for every scheduler slot. For grouped caches it is what vLLM's
startup check reserves for one request of the context length: every block of
a full-attention group, the admission cap of a sliding-window or
`convolution` group with the in-flight tokens of two batches of the
configured prefill chunk, and the startup blocks of a `state` group (`2 + k +
c` for a prefix-cacheable group in a profile that records `align` mode). The
window term counts two batches for every deployment. Two is vLLM's count with
async scheduling at pipeline parallel size 1, and grouped execution supports
PP1 only. With `async_scheduling: false` vLLM counts one batch, so the bound
is then high by at most `ceil(max_num_tokens / block_size_tokens)` pages per
window group. The bound is vLLM's startup check, not a promise that every
prompt up to the context length completes: with async scheduling a request
can hold one block per prefix-cacheable state group more for a step, and the
item on a hit inside a block below names a case that then never completes.
For a linear layout it is the context length times the bytes per token. For
an observed Mamba align pool it counts the pages of the context length, the
old and the new state of each state group with its speculative states, and
one page for a partial attention source. A smaller context changes this peak bound,
not the declared non-cache resource bounds. `tolerance_adjusted`, when present,
provides the reduced byte budget. For grouped caches, `kv_size_per_token_bytes`,
`total_kv_size_tokens`, and the adjusted token capacity are null (`None` in Python).
There is no equivalent scalar token capacity.

For a `state` group the peak bound counts the `2 + k + c` blocks of vLLM's
`align` startup count only when the group states `prefix_cacheable: true` and
the runtime memory records `mamba_cache_mode: align`. A profile that was
imported before the import wrote `prefix_cacheable` leaves that field unset on
its cache groups. Its `state` groups then count `blocks_per_request` blocks
each, because the profile does not say which `state` group is a ring. For the
GLM-5.3-Flash profile of the vLLM 0.33.0 runtime probe on GB300 at TP 4, with
blocks of 2,176 tokens and a context of 1,048,576 tokens, the bound is 487
pages, where vLLM's startup check reserves 495: each of the four
KDA groups has one checkpoint block, so vLLM counts 3 blocks for it and the
bound counts 1. A new import of the same runtime probe writes the field, and
the bound then equals vLLM's count. The
[migration guide](../aic-backward-compatibility/migration.md#rust-resource-and-memory-literals)
gives the steps of that import. The import writes the field only from a vLLM
0.33.0 record, whatever the label of the profile.

`aisim onboard plan` refuses a pool below this bound, and the sweeper leaves
out a deployment shape whose pool is below it. Prediction does not test the
pool against the bound. With prefix caching on, the native rank then refuses
a pool below its startup count. With prefix caching off the replay runs, and
it rejects each request whose whole sequence is larger than the pool. vLLM
does not start with a pool that cannot hold one request of `max_model_len`
tokens.

When `request_occupancy_tokens` is supplied, the result also includes
`request_occupancy_cache_bytes`. KV-relative synthetic loads use this native
footprint while retaining the actual scheduler settings and context for resource
admission. Observed linear pools retain their recorded capacity even when timing
coverage is smaller; uncovered queries report timing errors.

Runtime memory uses `kv_cache_bytes` directly and returns
`memory_breakdown: null` (`Option::None` in Rust). It does not invent activation
or other non-KV components. The requested memory fraction must exactly match the
recorded utilization; the device budget must contain the observed cache pool.
A different device capacity argument never rescales the pool or qualifies a new
deployment. Changed scheduler settings require new runtime evidence. Explicit
scalar cache-capacity overrides are rejected for runtime profiles in prediction,
recommendation and replay; use the recorded allocation. A tolerance margin may
reduce it through the canonical budget API.

Python `estimate_kv_cache(..., fpm_profile=..., context_length=...)` delegates
grouped and runtime-memory profiles to this native budget method.
`estimate_num_gpu_blocks` also accepts `context_length` for runtime validation;
it and the
legacy Rust scalar `KvCacheEstimate` transport reject grouped profiles; use the
groups and shared byte budget instead. Existing complete linear declarations
keep their budget calculation and serialized shape.

Grouped native execution supports aggregated and P/D vLLM, PP1/CP1, HBM-only
cache and non-speculative decoding, cold or with prefix caching. Grouped prefix caching
follows the rules of vLLM 0.31.0, 0.32.0 and 0.33.0. It needs a profile deployment whose
`backend_version` is one of these versions, with prefix-caching evidence and
equal `page_size_bytes` across groups, whose recorded evidence version, when
present, is that version too. The probe import takes prefix-caching evidence
only from a vLLM 0.33.0 record and records that version, so the plan of a
profile from the runtime probe with the label `0.33.0` replays with prefix
caching when it meets the conditions. [Replay KV cache](../replay/engine/kv-cache.md#other-cache-modes)
gives the case of a profile imported before the version was recorded.
Grouped P/D follows the transfer rules of vLLM 0.31.0, 0.32.0 and 0.33.0. It
needs a profile deployment of one of these versions whose groups all state
`prefix_cacheable` and `transfer_bytes_per_block`, and a prefill and a decode
deployment with equal tensor parallelism and identical cache groups. The probe
import records vLLM 0.33.0 and writes `transfer_bytes_per_block` only from a
probe phase with prefix caching and only for the measured cache shape. A
profile from such a probe runs P/D under the label `0.33.0`. Every other
profile needs a hand-made value on each group.
[Replay KV cache](../replay/engine/kv-cache.md#other-cache-modes) lists the conditions. Host/G3
offload and fixed scalar cache capacity are rejected. With
prefix caching off, allocation is atomic across all groups against one shared
rank-local byte budget. A waiting request is admitted only when the bytes of
its whole sequence fit the free bytes, with the admission cap of a window
group, as vLLM's default `scheduler_reserve_full_isl` does. A request whose
whole sequence is larger than the pool is rejected. A window
of `W` keeps the blocks covering the last `W - 1` processed tokens and all
tokens up to the end of the scheduled forward. The processed tokens are the
computed tokens less the tokens of the request's forward in flight. They equal
the computed tokens with `async_scheduling: false`. A prefill or decode
worker uses the processed tokens with async scheduling too. Expired
completed pages are evicted before
subsequent forwards; temporary prefill pages remain charged through completion.
Full-attention groups retain full history. A `state` group holds its
`blocks_per_request` blocks from the request's first scheduled forward until the
request finishes, is cancelled or is preempted. A resumed request reserves them
again once. This physical retention never truncates logical request progress or
the context coordinates sent to FPM timing queries.

With prefix caching on, replay keeps the blocks of all groups in one block pool:

- Every group has the same `page_size_bytes`, so one block of any group takes
  one pool block. The pool holds `kv_cache_capacity_bytes / page_size_bytes`
  blocks, rounded down.
- A hand-made profile of vLLM 0.32.0 or 0.33.0 with a block-outermost KV cache layout
  states the block row of the shared pool as the page of every group: the
  largest sum of the layer pages of one group, rounded up to the least common
  multiple of the block-stride alignments of all layers. See vLLM v0.32.0
  [`kv_cache_utils.py:1520-1551`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_utils.py#L1520-L1551)
  and vLLM v0.33.0
  [`kv_cache_planning.py:574-605`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/kv_cache_planning.py#L574-L605).
  vLLM logs the layout at start, for example `Using BLHNC KV cache layout`. A
  layout name that starts with `B` is block-outermost.
- A native rank with `kv_cache_groups` and `enable_prefix_caching` also needs
  `prefix_match_unit` (the profile's `hash_block_size_tokens`), `vllm_version`
  (`0.31.0`, `0.32.0` or `0.33.0`) and `max_model_len`. Prediction and
  recommendation set all three. With `vllm_version` `0.33.0`, the rank's
  `prefill_schedule_interval` is 1. Every `state` group needs a `blocks_per_request` of 1.
- The pool must hold the blocks of one request of `max_model_len` tokens, as
  vLLM's startup check counts them. That is
  `ceil(max_model_len / block_size_tokens)` blocks for full attention. For a
  sliding window it is vLLM's admission cap,
  `ceil(min(sliding_window - 1 + in-flight tokens, max_model_len) / block_size_tokens) + 1`.
  With `async_scheduling` on, the default, the in-flight tokens are
  `max_num_batched_tokens` (`scheduler.max_batched_tokens` in the engine
  YAML) times 2. With it off they are `max_num_batched_tokens`. This is the
  same on every modeled version. vLLM v0.31.0
  [`kv_cache_interface.py:840-866`](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/kv_cache_interface.py#L840-L866),
  v0.32.0
  [`:859-885`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/kv_cache_interface.py#L859-L885)
  and v0.33.0
  [`:918-944`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/kv_cache_interface.py#L918-L944)
  use the same formula. They also add `extra_retained_tokens`. vLLM sets it on
  every sliding-window spec to `max(0, num_prefill_lookahead_tokens - 1)`,
  which is above 0 only with multi-module speculative decoding. A rank with
  `kv_cache_groups` refuses speculative decoding, so replay leaves the term
  out. The count is 2 blocks for a
  prefix-cacheable `state` group, 3 with a checkpoint block, and 1 block for a
  ring.
- Replay admits a waiting request only with room for the blocks of its whole
  prompt, as vLLM does by default. Cached blocks that no request holds count as
  room. A hit on such a block still costs one block of that room. The blocks of
  the request's first chunk must also fit, without the admission cap of a
  sliding window. A waiting request that fails either test stays in the queue
  and preempts no running request, as in vLLM. A hit inside a block can need
  more blocks than the pool holds, also in a pool that passes the startup
  check. The request then waits until running requests evict its hit blocks,
  as in vLLM. With no running request, vLLM waits forever, and replay stops
  with a livelock error. With async scheduling one request on an aggregated
  worker can reach this state alone. That can happen when the pool holds
  exactly vLLM's startup count, the hash block is smaller than the state
  block, the state groups have no checkpoint block, and the prompt's partial
  tail lies in the last attention block of the context length, and the
  layout is not one of the layouts in which an aggregated worker frees on
  the computed tokens (see the known differences in
  [Replay KV cache](../replay/engine/kv-cache.md#other-cache-modes)). The request
  preempts itself several times, then hits its own partial tail inside a
  block and is refused at every step. It is not preempted without end. vLLM's
  whole-sequence test refuses it in the same way; see vLLM v0.33.0
  [`kv_cache_manager.py:531-548`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/kv_cache_manager.py#L531-L548).
  With
  `async_scheduling: false` or with a checkpoint block the request completes.
- Full-attention groups keep their blocks while the request is running. A
  sliding-window group releases the blocks below its window before each
  allocation. A prefix-cacheable `state` group holds one running block. After
  a block boundary, it also holds the previous block until the first later
  allocation whose processed token count lies past that block: for one
  scheduler step with `async_scheduling: false`, and on an aggregated worker
  for two with a step in flight. In the layouts that the known
  differences in
  [Replay KV cache](../replay/engine/kv-cache.md#other-cache-modes) name,
  an aggregated worker holds it for one step, one step less than vLLM with
  a step in flight. A
  prefill or decode worker holds it for two steps with a step in flight, in
  every layout. After it writes past a state that it published inside a
  block, it holds the published block to the end of that scheduling pass,
  or as a previous block when the write also crosses a block boundary. A ring
  holds one block. [Replay KV cache](../replay/engine/kv-cache.md#other-cache-modes)
  describes the blocks of a hit and of a checkpoint.
- A request releases its blocks tail first. With async scheduling, a finished
  request releases its blocks after replay has formed the next pass, as vLLM
  does. A released block without a cache key is free at once. A released
  block with a key stays in one least-recently-used queue for all groups.
  When no free block is left, a new block of any group evicts the oldest
  block in that queue.
- [Replay KV cache](../replay/engine/kv-cache.md#other-cache-modes) lists
  where replay block counts, hits and admission differ from vLLM.

The Rust Replay observer's `ReplaySchedulerMetricsSnapshot` reports optional
`kv_cache_used_bytes` and `kv_cache_capacity_bytes` for grouped caches; its
`active_cache_usage` and `physical_cache_usage` use byte occupancy. `active_blocks`
counts resident group pages, while `total_blocks` is zero because heterogeneous
pages have no scalar block capacity. Use the byte fields for capacity comparisons.
With prefix caching on, the snapshot also reports `kv_cache_cached_bytes`, the
bytes of cached blocks that no request holds, and `inactive_blocks` counts these
blocks. `kv_cache_used_bytes`, `active_blocks` and `active_cache_usage` then
count only the blocks that requests hold. `physical_cache_usage` also counts the
cached blocks, and `kv_cache_capacity_bytes` counts whole pool blocks.
[Replay KV cache](../replay/engine/kv-cache.md#other-cache-modes) lists why the
cached blocks can differ from vLLM.
These fields are currently exposed through the Rust observer API, not the Python
JSON replay runtime. Linear snapshots retain their existing block metrics.

Cache reuse, eviction, and offload lifecycle are documented in [Replay KV cache](../replay/engine/kv-cache.md).

### Observed Mamba align pools

The `mamba_align` layout describes one full-attention group and padded
Mamba groups that share one physical pool. Supply observed runtime memory,
physical page bytes, block length, state-group count, prefix unit, and
speculative state count. Set `mamba_cache.state_storage: full_state` explicitly.
The count must match the measured draft depth. Internal prefill checkpoint
slots are unsupported.

| Field under `resources.mamba_cache` | Meaning |
| --- | --- |
| `state_storage` | Required `full_state`. Other policies are rejected. |
| `block_size_tokens` | Physical attention block length, at least two tokens. |
| `page_size_bytes` | Padded page size shared by attention and state groups. |
| `state_groups` | Number of state pages in one complete working state. |
| `prefix_match_unit` | Positive divisor of the block length. |
| `num_speculative_states` | Full-state scratch copies, equal to the measured draft depth, from zero through five. |

The native budget returns `replay_cache`. This descriptor excludes one reserved
null page and includes working-state and speculative scratch reservations.
Do not interpret its allocation byte rate as logical KV bytes per token.
The profile does not expose an aggregate token capacity.

Use aggregated vLLM replay with the measured block length and scheduler limits.
Keep prefix caching enabled. Fixed capacity overrides, offload, and disaggregated
FPM replay are unsupported for this layout. Original timing imports remain valid
as timing-only evidence.

ReplaySSM and RecoverSSM lay out and size their states differently, so
profile them as `unmodeled` for timing only; do not relabel them `full_state`.
Lookups follow vLLM revision `db9527a46873454610df6dbedf79a36d6bf1a7f6`, which
applies the logits cap before the MTP hash-unit drop, so some exactly aligned
prompts miss the cached tail.
