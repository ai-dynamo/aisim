<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.3-Flash FPM

Whole-forward (FPM) profiles for GLM-5.3-Flash on GB300, GB200, B300, H200
and H100, collected with engine-native timing on the default serving
configuration (prefix/radix cache on). Text only, pure TP, DP = PP = EP = 1.

| System | vLLM stock `0.31.0` (`db9527a4`) | SGLang `0.5.20` (`94602c9c`) | KV |
|---|---|---|---|
| GB300 | FP8 TP2/TP4, NVFP4 TP1/TP2/TP4 | FP8 TP2/TP4, NVFP4 TP2/TP4 | FP8 |
| GB200 | FP8 TP4, NVFP4 TP2 | FP8 TP4, NVFP4 TP2/TP4 | FP8 |
| B300 SXM | FP8 TP2, FP8 TP4 (development), NVFP4 TP1/TP2/TP4 | FP8 TP2/TP4, NVFP4 TP2/TP4 | FP8 |
| H200 | FP8 TP4/TP8 | FP8 TP4/TP8 | BF16 |
| H100 | FP8 TP8 | FP8 TP8 | BF16 |

Checkpoints: `zai-org/GLM-5.3-Flash@eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`
and `nvidia/GLM-5.3-Flash-NVFP4@09b04e5e74bca08ca8549fc736d4cdd8624bfde3`.
FP8 TP1 does not fit on one GPU. Hopper profiles use BF16 KV for both backends.
The 15 published vLLM profiles are data from stock vLLM 0.31.0. New vLLM
collections run the [producer](#vllm-producer) on vLLM 0.33.0 with the Dynamo
scheduler file of the `release/1.7.0` branch that `runtime-source-sha256.json`
pins.

The tables (`fpm_forward_perf.{parquet,metadata.json}`) are not stored in this
repository. They are published in `ai-dynamo/aisim-data` under `data/fpm/`
(moved byte-identically from the retired `nvidia/aisimulate-fpm-dataset`, where the review
discussions remain:
[GB300 discussion 15](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/discussions/15),
[experimental B300 discussion 19](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/discussions/19),
[GB200/B300/H200/H100 discussion 20](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/discussions/20)).
[`hf_dataset.json`](../python/aisim/src/aisim_core/systems/profiles/glm53flash_fpm/hf_dataset.json)
pins the 29 profiles to the immutable `ai-dynamo/aisim-data` revision it records
(the files are the byte-identical successors of `nvidia/aisimulate-fpm-dataset@dbfcdd9eb62ea733efbc3f0cbd78277f2960061a`)
and records each file's SHA-256.
Profile keys are `<system>-<backend>-<fp8|nvfp4>-tp<N>` with system `gb300`,
`gb200`, `b300`, `h200` or `h100`, for example `gb300-vllm-nvfp4-tp1` or
`h100-sglang-fp8-tp8`. Pass the profile identity's `kv_cache_dtype` and
`fmha_quant_mode` (bfloat16 on H200/H100). Materialize one with
`python -m aisim_core.sdk.fpm_dataset <manifest> <key>` (or
`materialize_fpm_profile`) and pass the printed root as `systems_paths`, with
`estimation_mode="fpm_interpolation"` and the profile identity's
`backend_version`.

The experimental B300 key is `b300-vllm-fp8-tp4`. Default materialization is blocked; use `--allow-unqualified` only for diagnostic reproduction. It uses system `b300_sxm`, FP8 KV, TP4 and vLLM `0.31.0`. Its expanded table has 427 prefill and 175 decode measurements. Its collection covers batch sizes up to 32 and per-request contexts up to 131,072 tokens. Validation covers align-4 forward-pass inputs only; end-to-end serving, cache reuse, 1M context and a supported DynoSim dependency pair remain unqualified.

## vLLM producer

`python/aisim/collector/fpm_forward/runtime/glm53flash/` contains the
producer. It is staged next to `fpm_exec.sh` and `preflight.py` and runs with
the pinned Dynamo `InstrumentedScheduler` overlay on `PYTHONPATH`. The first
part of the overlay below is the `ai-dynamo 1.7.0.dev20261109` build, whose
scheduler file equals `ai-dynamo/dynamo@9abd2ef0d9` on `release/1.7.0`, and
`runtime-source-sha256.json` pins that file. It also pins
`dynamo/vllm/worker_factory.py`, whose merge checks every measurement record
before the engine writes its merged artifact. The same file pins three more
Dynamo files and vLLM source files, among them `vllm/sampling_params.py`,
`vllm/tokenizers/registry.py`, `vllm/config/scheduler.py` and
`vllm/v1/engine/core.py`.

The producer runs on exactly stock vLLM 0.33.0 (`4ba787ec`) with the Dynamo
scheduler file of the `release/1.7.0` branch that `runtime-source-sha256.json`
pins. It refuses every other vLLM version, a patch release such as 0.33.1
included, before the scheduler loads.

- `sitecustomize.py` checks the installed vLLM/Dynamo sources against
  `runtime-source-sha256.json` before the scheduler loads. It then replaces
  `InstrumentedScheduler` with `Glm53FlashPrefixSeedScheduler` from
  `glm53flash_prefix_scheduler.py`.
- Timing is Dynamo's unchanged FPM `wall_time`. Prefill runs from the end of
  `schedule()` to the arrival of `update_from_output`. Steady decode is the
  interval between consecutive output arrivals.
- The adapter only adds requests to the native waiting queue. vLLM's own
  scheduler schedules the seed, warmup and measured requests, so prefix-cache
  lookup, Mamba `align` chunking, KDA/conv, MLA and IndexPool state all run
  their serving code paths.
- Stock v0.31.0 does not complete IndexPool entries at a cached or chunked
  prefill start that is not a multiple of 4, and the sources do not show that
  v0.33.0 removed the defect. Inputs are therefore built on the 4-token grid,
  and the producer fails the run on any unaligned prefill chunk start. A decode
  point keeps its true context `c`: the prompt has `c - 1` tokens and the
  cached seed `(c - 2) // hash_block * hash_block`, where `hash_block` is the
  hash block size of the engine.
- Each point runs 5 warmups and 10 measurements. The table stores the median
  of measurements 2 to 10, which is Dynamo's `adjacent_upper_median` estimate.
  Measurement 1 is a valid measured repetition that this estimate leaves out.
  The merge of Dynamo 1.7 accepts no other median of several samples. Each raw
  sample of the record is numbered by its position. The first repetition of a
  prefill point with a prefix is rejected. It runs before the warmups and
  counts as neither a warmup nor a measurement, and the later repetitions are
  accepted: 5 warmups, then 10 measurements. With the default
  `prefix_cache_retention_interval` of 0, the seed does not keep the Mamba
  state at the end of the prefix. vLLM 0.33.0 stops the first repetition there,
  at the shared-prefix junction, and caches the state. The later repetitions
  hit the whole prefix. One rejected repetition for each such prefill point is
  the expected result. The producer writes a repetition evidence file next to
  the artifact, with the name of the artifact and the ending
  `.repetitions.jsonl` in place of `.json`. The file has one line for each
  point. The line holds the accepted repetitions of the point and each
  rejected repetition with its reason. The producer fails the run when one
  point has more than 10 rejected repetitions. The collector does not read
  the rejected repetitions, so their count does not decide whether the
  collector accepts an artifact.
- Request tokens come from `collector/glm53flash_attention_tokens.py`
  (seed 53, ordinary BPE vocabulary of the pinned tokenizer).
- The repetition evidence file records each prompt as offsets into the seeded
  token stream. The measurement record of a point holds no prompt hashes: its
  `prompts.status` is `unavailable`. The collector reports that as a gap and
  does not refuse the artifact. With measurement evidence version 2, the
  default, the execution evidence of each cell names the gap
  `injected prompt hashes unavailable`, so its status is `incomplete` or
  `failed` and not `qualified`. Publication does not read that status.
  `aisim onboard validate-collection` does not finish on a GLM-5.3-Flash
  campaign. It writes a first report whose execution gate is `incomplete` or
  `failed`, and then its interpolation holdout stops with
  `No FPM cell matches`: a query that is built from an FPM profile carries no
  execution identity, and the rows of a producer cell carry one. For the same
  reason `aisim predict` and `aisim recommend` with the files of the
  onboarding plan do not read the collected table, and
  `finalize --collection-report` has no report of such a campaign to record.
  A query without a profile reads the published table.
- A pinned revision needs the configuration of its own checkpoint. The
  producer accepts any other revision whose configuration equals the
  configuration of a pinned checkpoint. `DYN_FPM_TOKENIZER_REVISION` is the
  40-hex revision that was loaded. A folder launch reads its folder, and a
  launch by Hub ID reads that revision's snapshot in the local Hugging Face
  cache. The artifact records the loaded revision and the pinned checkpoint
  that matched.
- The producer refuses a run whose vLLM model revision is set and differs from
  `DYN_FPM_TOKENIZER_REVISION`. A launch by Hub ID therefore passes
  `--revision` with the same value. Without it, vLLM loads the default branch.
- `--max-model-len` must be at least the measured context plus 7 internal
  positions, and a larger value is accepted. `DYN_FPM_GLM53FLASH_MEASURED_CONTEXT`
  sets the measured context (131,072 at most). The producer refuses a point
  whose per-request length exceeds the measured context.
- The producer refuses a point of more than 32 requests, the batch limit that
  its artifact states.
- Before a decode point decodes, each request prefills its prompt beyond its
  cached seed. The producer refuses a decode point of two or more requests whose
  suffixes together exceed the token budget of one step. That budget is the
  smaller of `--max-num-batched-tokens` and the `max_num_scheduled_tokens` of
  the vLLM scheduler. If they exceed it, vLLM prefills them over several
  steps. The first decode step then does not hold every request, and the
  producer rejects every repetition. One request can prefill its suffix over
  several steps. No other request decodes before it, and the producer records
  no step with a prefill request.
- With prefix caching, vLLM runs GLM-5.3-Flash in Mamba `align` mode. vLLM
  0.33.0 stops a prefill chunk at the last block boundary before the prompt
  end, `(end - 1) // block * block`. After a prefix that ends inside a block, it
  also stops the chunk at the next block boundary. A prefill point whose chunk
  holds such a stop runs in more than one step, and the producer accepts one
  step per repetition. So the producer refuses the point before it seeds
  anything. Without a prefill checkpoint block, new tokens of at most one block
  after a prefix of whole blocks run in one step.
- With the FlashKDA prefill backend, each KDA state group holds one prefill
  checkpoint block with an alignment of 16 tokens. vLLM 0.33.0 then saves the
  checkpoint inside the last chunk and takes neither stop when the chunk starts
  on a hash block boundary, ends more than one block after its start, and its
  last block boundary before the prompt end lies a multiple of 16 tokens after
  its start. If the block is a multiple of 16 tokens, every prefill after a
  prefix of whole blocks then runs in one step. The Triton backend gives no
  checkpoint block. `_glm_align_stop` in the producer holds the exact rule,
  and the artifact records the checkpoint blocks and their alignment in
  `serving_config.mamba_prefill_checkpoint`.

The collector reads the artifact of a GLM-5.3-Flash cell only as an artifact of
this producer. The cell selects these rules, not the artifact:

- The `producer` record must equal the record that this producer writes,
  field by field: the instrumentation revision `54960177`, the scheduler
  revision `9abd2ef0d9` and the `ai-dynamo 1.7.0.dev20261109` distribution,
  the sources of the real-seed and queued-request ports, vLLM `0.33.0`, the
  reviewed vLLM revision `4ba787ec`, the digests of the two producer files,
  `runtime-source-sha256.json` and `glm53flash_prefix_scheduler.py`, the
  context policy version and the repetition counts above. The two digests
  are those of the producer receipt of the cell, the file
  `producer-receipt.json` that the collector writes when the cell passes. When
  the cell folder holds no receipt, they are those of the files of the
  checkout that reads the artifact. A missing, changed or added field is
  refused.
- The vLLM version of the record must equal the runtime that the Collector
  provenance of the launch names.
- The seeded input of the record must be the input that this collector draws:
  the generator and its version, the seed, the token count, the vocabulary
  bounds and the digest of `tokenizer.json`. The reader draws the token
  stream again and compares its digest and its count of distinct tokens. A
  record of another generator version is refused with
  `input_provenance.generator_version`.
- The artifact must name the repetition evidence file and state its digest
  and its record count. The count must be an integer and equal the number of
  rows. The `repetition_evidence_sha256` of each row must be the SHA-256 of
  the line of that file with the `benchmark_id` of the row.
- `serving_config` must record prefix caching, synchronous scheduling, the
  Mamba cache mode `align`, an integer `block_size`, a `hash_block_size`
  equal to it, and the block-aligned split of prefill chunks.
- The measurement protocol must be present. It must hold the producer's record
  of 5 warmups, 10 measurements and their reduction, the `seeded_tokens`
  content and Dynamo's execution evidence contract. It must record a global
  warm-up count of 0 in `preparation.warmup_iterations`, because the producer
  runs its 5 warmups for each point.
- Every point must hold the 10 raw samples and the `adjacent_upper_median`
  estimate above. Each raw sample must hold the workload of its point in
  `scheduled_requests`: the five request and token counts, each as an
  integer. The warm-up history must be recorded, and each raw sample must
  have its timing start and an observed CUDA graph dispatch.
- A rank artifact of another model that carries the producer's protocol record
  is refused.

The first five rules apply wherever the collector reads the artifact,
publication included. Readiness and repeatability assessment apply the other
three. Publication (`aggregate_cell`) applies the protocol rule and the raw
sample rule to a GLM-5.3-Flash cell with the same reader, and refuses an
artifact that breaks one with the reader's text. A gap that the reader only
reports, such as the prompt hashes that the producer does not write, does not
refuse publication. Publication does not read the measurement records of the
cells of other models, so it does not apply the last rule. The checks of the
warm-up history, of the timing start and of the CUDA graph dispatch need
measurement evidence version 2, the default, which publication uses. The
other checks of the protocol and of the raw samples also run at version 1.

When a GLM-5.3-Flash cell passes, the collector writes a producer receipt,
`producer-receipt.json`, into the cell folder. The receipt holds the digests
of the collector's own `runtime-source-sha256.json` and
`glm53flash_prefix_scheduler.py`, which the artifact has matched, with the
plan hash, the cell ID and the attempt ID. Every later read of the cell takes
the two digests from the receipt: readiness, repeatability assessment,
publication, finalization and serving validation, and each later check of a
profile that was finalized from the campaign. So a campaign with receipts
stays readable after a change of either file. The receipt covers the two
digests only. The reader compares every other field of the record, the vLLM
version among them, with its own values.

The reader refuses a receipt that is not a regular file or that is malformed
with `invalid producer receipt: <rule>: <path>`. It refuses a receipt that
names another plan, cell or attempt with `producer receipt names another`
and the name and both values of the binding that differs. A new attempt of a
cell removes the receipt file of the attempt before it. A resume that
recovers a cell from complete artifacts keeps a receipt that the reader
accepts, and writes one when the cell folder holds none. It does not recover
a cell whose receipt the reader refuses. No command writes the receipt of a
passed cell again, so a passed cell with a refused receipt needs a new
collection. The plan hash covers the collector revision, which
`collection-plan.json` records as `aic_revision`. A direct
`python3 -m collector.fpm_forward --resume` by a collector of another revision
is therefore refused with
`FPM checkpoint does not match the current frozen plan`. A formal
`aisim onboard collect-fpm --resume --execute` stops before that, at its
readiness check, with `full collection requires usable prefill and decode
readiness for this exact runtime/launch; inspect --check-readiness and run
--smoke --execute first`.

A cell folder without a receipt is compared with the files of the checkout
that reads it. After any change of either file, a comment included, the
collector refuses such a campaign wherever it reads the raw artifacts again,
each later check of a profile that was finalized from the campaign included.
The refusal names the first field of the record that differs. When only the
bytes of the files changed, that field is
`producer.runtime_source_manifest_sha256` or `producer.overlay_sha256`.
For an accepted profile, `onboard checkpoint` and `onboard resume` run that
check. They report the refusal as an integrity issue, report the profile as not
accepted and exit with status 2. They also run it for a request or profile file
that the checkpoint registers for the configuration, and report the refusal as
an integrity issue of that file. An attempt to accept such a profile fails with
the same text and status 2. The saved-plan check of a finalized directory whose finalization
kept a collection report refuses the campaign too.
`onboard import-observations` reads only the observation files and is not
affected. Read such a campaign with the collector revision that collected it,
or collect a new smoke and a new campaign. The simulator keeps loading a
published table, because loading reads the table and not the raw artifacts.
Artifacts of other models are not affected.

Engine arguments, beyond the usual Dynamo benchmark flags:
`--enable-prefix-caching --kv-cache-dtype fp8
--no-async-scheduling --language-model-only --cudagraph-metrics
--max-num-batched-tokens 8192 --max-num-seqs 32 --benchmark-warmup-iterations 0`.
Without `--prefix-match-unit`, the hash block is the native block of the
engine. The published vLLM 0.31.0 profiles used `--prefix-match-unit 4`, and
the producer refuses that launch on vLLM 0.33.0, because vLLM 0.33.0 turns on
partial hash hits (`mamba_partial_cache_hit`) for a hash block below the Mamba
block, and the `align` rule of the producer does not model them.

Environment:

| Variable | Value |
|---|---|
| `DYN_FPM_GLM53FLASH_REAL_KV` | `1` |
| `DYN_FPM_GLM53FLASH_PREFIX_SEED` | `1` |
| `DYN_FPM_GLM53FLASH_MEASURED_CONTEXT` | `131072` |
| `DYN_FPM_TOKENIZER_REVISION` | checkpoint revision |
| `DYN_FPM_GLM53FLASH_DECODE_CONTEXTS` | optional `[[batch, total, [contexts...]], ...]` with the per-request decode split |

## Guided onboarding collection

`aisim onboard collect-fpm` runs the producer above as the runtime adapter
for GLM-5.3-Flash collection. The
[self-service guide](perf-model/fpm-self-service/implementation.md) describes
the workflow. The collector renders the producer launch for each prefill and
decode phase of a `Glm5NextForConditionalGeneration` collection, formal or
smoke. A runtime probe of the same deployment keeps Dynamo's generic grid and
does not run the producer.

Both phases get `--language-model-only`, `--cudagraph-metrics`,
`--no-async-scheduling`, `--benchmark-warmup-iterations 0`,
`--benchmark-points-file` for the staged point list and `--revision` with the
value of `DYN_FPM_TOKENIZER_REVISION`. The request sets
`--max-model-len`, `--max-num-batched-tokens` and `--max-num-seqs`. The
collector passes no `--prefix-match-unit`. It also sets these variables:

| Variable | Value |
|---|---|
| `DYN_FPM_GLM53FLASH_REAL_KV` | `1` |
| `DYN_FPM_GLM53FLASH_PREFIX_SEED` | `1` |
| `DYN_FPM_BENCH_COLLECT_IMBALANCED` | `1` |
| `DYN_FPM_TOKENIZER_REVISION` | launch revision of the accepted probe, else the model revision of the accepted profile |
| `DYN_FPM_GLM53FLASH_MEASURED_CONTEXT` | the smaller of `131072` and the runtime context minus 7 |
| `PYTHONPATH` | `/tmp/fpm-bench:/opt/glm53flash-dynamo` |

`DYN_FPM_GLM53FLASH_PREFIX_MATCH_UNIT` stays unset, so the native hash block of
the engine is the prefix unit. For a generated point list the planner refuses
a hash block that is not a multiple of 4. The planner also refuses an accepted profile in which the hash
block of a selected deployment differs from the block size of one of its Mamba
groups. For a generated point list, as in guided onboarding, it checks every
selected deployment. For a point file, it checks the deployments that keep
cells after memory admission. A Mamba group is a state group that the profile
records with `prefill_checkpoint_blocks`: the KDA groups, and not the ring
group, a state group that the profile records without that field. The refusal
names the group and both sizes. The producer still checks every prefill chunk
start. A decode point seeds `(c - 2) // hash_block * hash_block` tokens, as in
the launch above.

The container needs one overlay directory at `/opt/glm53flash-dynamo`. A Slurm
run can mount it with `--container-mount`. The collector uses that path unless
the deployment environment declares its own `PYTHONPATH`. The overlay is a
composite of two parts. The first part is the `dynamo` files of the
`ai-dynamo 1.7.0.dev20261109` build, the Dynamo 1.7 scheduler and merge whose
files `runtime-source-sha256.json` pins. The second part is the `collector`,
`aisim_core` and `aisim` packages of the AISim checkout that
plans the collection. The producer imports them. Run the runtime probe with the
deployment of the collection, the overlay mount included, because the collector
refuses a collection deployment that differs from the accepted probe.

The request must follow these rules:

- `collection.warmup_iterations` is 0. The producer runs its own warm-ups and
  refuses a global warm-up. `aisim onboard init --warmup-iterations 0` sets
  it. Set it before the runtime probe: a request whose count differs from the
  accepted probe needs a fresh probe, and the collector refuses a collection
  whose count differs from the accepted probe.
- The runtime context is at least the measured context plus 7. The runtime
  context is `search.context_length`, and a larger value is accepted. The
  collector sets the measured context to the smaller of 131,072 and the runtime
  context minus 7.
- The token budget `max_num_tokens` is a multiple of 4.
- `identity.model_revision` is the 40-hex revision of the loaded checkpoint. The
  revision rule above applies, with its checkpoint lookup for a folder launch
  and for a launch by Hub ID.
- The parallel layout is pure tensor parallelism: TP 1, 2, 4 or 8, with DP,
  PP, CP and expert parallelism 1. The planner checks the layouts that keep
  cells after memory admission and refuses another layout.
- The runtime is vLLM 0.33.0. If the plan holds the vLLM version of its
  runtime, the planner refuses another version. The plan takes that version
  from the instrumentation bundle that it stages or, without a bundle, from the
  observed runtime version (`--fpm-runtime-backend-version`, which guided
  onboarding sets from `identity.runtime_framework_version`). The planner does
  not read the data label `identity.framework_version` for this rule. A plan
  that holds no runtime version passes the rule, and the producer refuses
  another vLLM version in the container.
- The checkpoint `config.json` holds the `architectures` field. The planner
  refuses a GLM-5.3-Flash collection without it, and `--model-architecture`
  does not replace it.

Plan the collection from the accepted runtime probe. The planner generates the
point list from `max_num_tokens`, `max_batch_size` and `context_length` of the
resolved request. When the request leaves `max_num_tokens` or `max_batch_size`
unset, that limit takes the accepted profile's bound. The planner takes
`hash_block_size_tokens` and the cache capacity from the accepted profile, and
it refuses a profile without the hash block. The planner freezes the list into
the collection plan, so the plan hash covers it. The producer divides each
total evenly. Every request of a point therefore has the same new tokens,
prefix and decode context. These are the axes:

- Prefill new tokens per request: the powers of two from 4 to the token budget,
  every whole hash block up to the budget, and the budget itself. The list caps
  the budget at 8,192. The list keeps a length only if vLLM's `align` mode runs
  it in one step after each planned prefix, as the producer rules above say.
  Without a prefill checkpoint block, that leaves no length above one hash
  block. With the FlashKDA checkpoint block and a hash block that is a multiple
  of 16 tokens, it keeps every length. At default serving the hash block is the
  Mamba block.
- Prefill requests per batch: the powers of two up to the smallest of 32,
  `max_batch_size` and the number of requests whose new tokens fit the budget,
  and that smallest value itself.
- Prefill prefix per request, in whole hash blocks: 0, the powers of two, and
  the largest number that keeps prefix and new tokens within the measured
  context.
- Decode context per request: the powers of two from 4 to the measured context,
  and the measured context itself.
- Decode requests per batch: the powers of two up to the smallest of 32,
  `max_batch_size` and the number of requests whose suffixes fit the budget,
  and that smallest value itself. Here the budget is the whole token budget.
  The 8,192 cap applies to prefill only. The suffix of a request is the part
  of its prompt beyond its seed, `(c - 1) - (c - 2) // hash_block * hash_block`
  tokens. If the suffixes of a batch fit the budget together, vLLM prefills
  them in one step, and the first decode step holds every request. The axis
  always holds a batch of one request. If its suffix exceeds the budget, vLLM
  prefills it over several steps, and no other request decodes before it.

The planner also drops a point that does not fit its estimate of the accepted
cache capacity. It counts the points that each rule drops: the align rule for
prefill, the token budget for decode, and the cache capacity for both. Each
rule counts against the candidates that it sees. The decode candidates at a
context are the batches without the token budget and the largest batch whose
suffixes fit the budget. The planner logs the counts in one line. Guided
onboarding sets no log handler, so the collection plan also records them in
`point_generation.dropped_points`. For a request with an accepted profile,
`aisim onboard collect-fpm` also prints them to the error stream, in the
preview and with `--execute`, in the line `Point list: candidates and dropped
points by phase and rule: <counts as JSON>`. The planner refuses a plan with
no prefill point or no decode point. The estimate counts the blocks that the
requests of a point hold. Each request holds the attention blocks of its tokens
and one block in each state group. A prefill request with a prefix holds one
more block in each Mamba group, the cached block of its prefix beside its
running block. So does a decode request at a context of at least one hash
block. In a launch where each Mamba group holds a prefill checkpoint block, a
prefill of more than one hash block of new tokens also holds that block in
each Mamba group. With several deployments, a point takes the count of the
deployment that needs the most blocks. At TP4 with 2,176-token hash blocks,
8,192 tokens, 256 sequences and a 1,048,576-token context, a FlashKDA
launch (five state groups, four of them KDA groups with a checkpoint block) has
496 prefill points and 81 decode points. The align rule drops none of 496
prefill candidates, and the token budget drops 16 of 97 decode candidates.
Without a prefill checkpoint block, the align rule drops 40 prefill candidates,
and the list has 456 prefill points.
The largest decode batch is 32 at contexts up to 256 and at 32,768 and 65,536.
It is 16 at 512 and 131,072, 8 at 1,024, 4 at 2,048 to 8,192, and 7 at 16,384.
By the estimate, each of the two lists needs an accepted cache of at least
2,240 free pool blocks (the profile's `kv_cache_bytes` divided by its largest
`page_size_bytes`). The largest need is that of the prefill points of 32
requests after a prefix of 60 hash blocks. The decode points alone need 1,280
free pool blocks. The profile's `kv_cache_bytes` counts the blocks that are
free when the engine starts, so vLLM's null block is not part of these counts.

The smoke measures two points of this list for each phase that the plan
collects. If the list holds the third prefill point below, the smoke also
measures it. An aggregated request collects both phases, so its smoke measures
four or five points. A prefill or decode role collects one phase, so its smoke
measures the points of that phase. The smoke takes no point of the other phase.
`aisim onboard collect-fpm --smoke` runs every cell of the plan, or the first
cells up to `--limit`. A direct `python3 -m collector.fpm_forward --smoke`
call without `--limit` runs the first cell of each phase that the plan
collects.
The smoke runs under the plan hash of formal collection and keeps its own
campaign directory. A smoke campaign can be the source of a repeatability run
only in the `bounded` comparison mode
(`--repeatability-comparison-mode bounded`). The default mode, `full_grid`,
stages the full list, so the collector refuses a smoke source in that mode
before any launch. The prefill points of the smoke are one request with 64 new
tokens, once without a prefix and once after one hash block of prefix. The
third prefill point is one request with two hash blocks of new tokens after
one hash block of prefix. The list holds it only for a launch with a prefill
checkpoint block. Its chunk holds a block boundary before its end, and vLLM's
`align` mode saves the checkpoint inside the chunk instead of stopping there.
So the smoke shows
that such a point runs in one step. An aggregated smoke therefore measures
four points when the launch has no prefill checkpoint block, and five points
with the FlashKDA backend, as for the TP4 list above. The decode points are one
request and two requests at the smallest context above one hash block where
the list holds both. The token
budget can leave a smaller context with one request only. If no context above
one hash block holds two requests, the smoke takes one request at each of the
two smallest contexts above the block. A `max_batch_size` of 1 or a cache with
room for one request causes this. So does a budget too small for two suffixes
at each context above the block.
A direct `collector --fpm` call can pass its own manifest with
`--fpm-benchmark-points-file`. The collector refuses that file together with
`--smoke`. Guided onboarding passes a batch limit for both phases. A direct
call can leave a phase without one: it sets neither `--fpm-max-num-seqs` nor
the limit of that phase, `--fpm-max-prefill-batch-size` or
`--fpm-max-decode-batch-size`. The generated list of such a phase takes the
`max_batch_size` that the selected deployments share. If the plan collects
such a phase and the selected deployments hold different `max_batch_size`
values, the planner refuses the plan, because one list would leave out the
larger points of one deployment. Two changes build a plan: select
deployments that share one `max_batch_size` with `--fpm-gpu-counts` or
`--fpm-parallel-presets`, or probe the runtime again so that the deployments
record one `max_batch_size`.

Cached prefill rows and decode rows publish the regime `real_kv`, the only
regime that the consumer accepts for them. A prefill row without a prefix
publishes `n/a`. The native artifacts keep the `real_prefix` stamp for cached
prefill.

## SGLang client

`python/aisim/collector/fpm_forward/sglang_native_fpm.py` launches stock
`sglang.launch_server --enable-forward-pass-metrics` and reads the native
rank-0 `ForwardPassMetrics` ZMQ stream. `wall_time` is the native DeviceTimer
forward interval.

- Each geometry is built from real requests: a warm request per prefix, then B
  requests that hit the cached prefix. The radix cache is flushed per point.
- The client does no work while a round is in flight.
- The measured decode step is attributed by geometry, not by message order.
- Request tokens come from the same seeded generator as the vLLM producer.
  The client checks the model's `tokenizer.json` against the pinned hash
  first. The run metadata (`input_provenance`) records the generator spec and
  the token-stream hash.

Usage, with `python/aisim` on `PYTHONPATH`:
`python3 -m collector.fpm_forward.sglang_native_fpm --model <dir>
--model-revision <rev> --tp {1,2,4,8} --quant {fp8,nvfp4} --mem-fraction <f>
--phase {prefill,decode} --points <manifest.json> --out <dir>
[--kv-cache-dtype {fp8_e4m3,bfloat16}] [--port <p>] [--ipc <name>]`.

- `--kv-cache-dtype` defaults to `fp8_e4m3`. The client checks that the
  server resolved the requested dtype and records it in
  `serving_config.kv_cache_dtype` (`fp8` or `bfloat16`). BF16 KV is a distinct
  identity and is never mixed with FP8-KV rows.
- The resolved attention, MoE and other `*_backend` server arguments are
  recorded in `serving_config.resolved_server_args` and are not asserted. The
  resolved page size must equal the 64-token hit unit of the request plans;
  otherwise the run fails at startup (`failed.json` lists the failed checks).
- Servers packed on one node need distinct `--port` values and GPUs
  (`CUDA_VISIBLE_DEVICES`). The default IPC name includes the run id and port.

The published SGLang 0.5.20 tables were collected with an earlier client
revision that took its request tokens from a text corpus. That input
provenance is recorded in the HF dataset metadata. Future collections use the
seeded generator.

## Consumer

`crates/core/src/perfmodel/perf_database/fpm_forward.rs` admits two schema-7
median policies. Each policy is bound to its backend, state protocol and timing
boundary (see [whole-forward](perf-model/methods/whole-forward.md)):

| Backend | `measurement_policy` | `state_protocol` | `timing_boundary` |
|---|---|---|---|
| vLLM | `vllm_native_fpm_prefix_seed_median_v1` | `glm53flash_prefix_cache_real_seed_v1` | `dynamo_vllm_instrumented_scheduler_fpm_wall_time` |
| SGLang | `sglang_native_fpm_radix_seed_median_v1` | `glm53flash_sglang_radix_real_seed_v1` | `sglang_native_fpm_rank0_device_timer_wall_time` |

The collector publishes the rows of a GLM-5.3-Flash cell that the vLLM
producer measured, a producer cell, under the vLLM policy. Each row carries
`measurement_policy` `vllm_native_fpm_prefix_seed_median_v1`, `warmup_repeats`
5, `measurement_repeats` 10, and the state protocol and the timing boundary of
the table. The sidecar, the `.metadata.json` file beside the Parquet file,
carries the same policy and the two counts. The two files are one database
pair. Besides the reader rules of the section [vLLM producer](#vllm-producer),
two conditions of this policy refuse the publication of such a cell. In the
first, the global warm-up count of the plan (`--fpm-warmup-iterations`, which
guided onboarding sets from `collection.warmup_iterations`) is not the
integer 0. In the second, the cell wrote power-window artifacts. The collector
also refuses to put rows of this policy into one database pair with rows of
another policy or rows without a policy, in either order, and it leaves the
existing pair unchanged. A database root holds one pair for each system,
backend and backend version, so publish a GLM-5.3-Flash collection to a clean
destination, or to a root whose pair for them holds no rows of another policy
and no rows without a policy. Finalization and collection validation read the
policy from the sidecar of the publication that they check.

Align-4 constrains collection only. The consumer answers queries for any
prefix length.

The pinned checkpoints are listed in `DefaultHFModels`, so model configs
resolve offline from the bundled copies.

## Systems and parallelism

The producer and the SGLang client have no system gate and accept pure TP1, TP2,
TP4 and TP8 (DP = PP = EP = 1). Whether a deployment fits a system (for example
H100, H200, B300, GB200 or GB300) is decided when the runtime loads weights and
allocates KV; a deployment that does not fit fails observably. FP8 TP1 does not
fit on one GPU of any of these systems. Profile keys in `hf_dataset.json`
follow `<system>-<backend>-<fp8|nvfp4>-tp<N>`.
