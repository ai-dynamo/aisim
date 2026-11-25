<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Introduction
Data collection is a standalone process for collecting the database for aiconfigurator. By default, you don't have to collect the data by yourself.
Small versions of database will not introduce huge perf difference. Say, you can use 1.0.0rc3 data of trtllm on h200_sxm and deploy the generated
configs with Dynamo + trtllm 1.0.0rc4 worker.

If you want to go through the process, you can try belowing commands. However, you need to prepare the env by yourself such as installing a specific trtllm version.
This process is not well verified, you need to debug sometimes.

For a framework-version or GPU-platform upgrade, follow the
[Collector Upgrade Playbook](../../../docs/perf-model/collector/upgrade.md).
The repo-tracked [`aic-auto-collect`](../.claude/skills/aic-auto-collect/SKILL.md)
skill applies that workflow during long, resumable collection runs.
SGLang 0.5.14 Hopper/Blackwell follow-up work must also consult the
platform alignment ledger, a local campaign record kept outside the repo
(`docs/perf_database/sglang-0.5.14-hopper-blackwell-ledger.md` in the
campaign workspace).

# Optional model-owned long-context coverage

Normal collection retains the shared grid and model geometry coverage. Add
`--include-long-context` to collect the additional correlated points declared
on selected model rows; it defaults to off. A model selection or
`--model-cases-full` is required. Existing long-context points in the normal
shared module grid remain enabled.

```bash
python collect.py --backend sglang --model-path zai-org/GLM-5.2 --ops dsa_generation_module --include-long-context
python collect.py --backend sglang --model-cases-full --sm 103 --plan-only --include-long-context
```

The optional points reuse the native geometry, actual projection precision and
existing framework dispatch. Model context limits validate declarations; they
do not automatically request every length or batch. Workers receive explicit
points and retain grouped setup. No environment switch changes their rows.
`--plan-only` reports the selected mode and hashes its actual case list. Use
separate checkpoint/output roots for before/after cost experiments; changing a
group's point list changes its existing task ID and can repeat its older points
on resume, while preventing newly added points from being skipped.

See [the declaration fields](cases/README.md#optional-long-context-points) and
[coverage rationale](cases/README.md#choosing-coverage). Collection-cost
estimates count setup changes, point preparation, warmup/capture and timed
forwards; scheduler wait and GPU allocation overhead are excluded.

Kimi-K3 adds no optional MLA long-context points yet: pinned vLLM 0.30.0
[`MultiHeadLatentAttention`](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/kimi_k3/nvidia/mla.py#L136)
uses NoPE and fused cache operations that the existing generic DeepSeek module
does not reproduce. A native adapter needs cache registration/binding, absorbed
weight processing, forward/metadata and FP8 policy support, plus a consumer
review of its output-gate timing boundary and silicon qualification. See the
[deferred extension](cases/README.md#deferred-kimi-k3-mla-extension).
Normal coverage and independent KDA width additions remain enabled; the missing
native long-context route is an unmeasured gap, not zero collection cost.

Follow the [benchmark guidelines](../../../docs/perf-model/collector/README.md#benchmark-guidelines)
for settings, workload, warmup, boundaries, ranks, GPU time and provenance.

# Preparation

SGLang 0.5.14 standalone MLA rows distinguish kernel compute from KV storage.
On B200/B300, `trtllm_mla` quantizes Q/K/V internally when the KV cache is FP8,
so those rows use `mla_dtype=fp8, kv_cache_dtype=fp8`. On H200, the absorbed
576-dimensional FA3 path uses BF16 compute, including when KV storage is FP8.
BF16 inputs to the collector do not determine the kernel's compute precision.
Compute-precision labels are restricted to the audited `trtllm_mla`, `fa3`,
and `triton` backends; an unknown backend raises instead of receiving a BF16 label.
Legacy Blackwell rows labeled `bfloat16/fp8` need fresh measurements with the
correct label; do not copy their timings into a second precision key.

Before collecting the data, make sure you own the whole node and no interfierence happens.
Next, please enable persistent-mode and lock frequency of the node. Make sure the cooling system of the node is working well.
```bash
sudo nvidia-smi -pm 1
```
```bash
sudo nvidia-smi -ac yyy,xxx
```
xxx, yyy frequency can be queried by nvidia-smi -q -i 0, refer to the Max Clocks part, xxx is SM frequency, yyy is Memory frequency.
A script to set frequency:
```
#!/bin/bash

# Run nvidia-smi query and extract SM and Memory frequencies from Max Clocks
sm_freq=$(nvidia-smi -q -i 0 | grep -A 4 "Max Clocks" | grep "SM " | grep -o "[0-9]\+ MHz" | grep -o "[0-9]\+")
mem_freq=$(nvidia-smi -q -i 0 | grep -A 4 "Max Clocks" | grep "Memory " | grep -o "[0-9]\+ MHz" | grep -o "[0-9]\+")

# Check if frequencies were successfully extracted
if [ -z "$sm_freq" ] || [ -z "$mem_freq" ]; then
    echo "Error: Could not extract SM or Memory frequency from Max Clocks."
    exit 1
fi

# Generate the command
echo "sudo nvidia-smi -ac $mem_freq,$sm_freq"
```
Prepare a clean env with the target framework and nccl lib installed.

# Collect comm data
```bash
export PATH=$PATH:${NCCL_TEST_BIN_PATH}/
network/collect_comm.sh #all_reduce data will be collected using default trtllm backend
network/collect_comm.sh --all_reduce_backend vllm #all_reduce data will be collected using vllm backend
network/collect_comm.sh --all_reduce_backend vllm --device xpu #all_reduce data will be collected using vllm backend on XPU
```
Today we only collect intra-node comm. This script will collect custom allreduce data for trtllm within a node.
It will also collect nccl allreduce, all_gather, all2all, reduce_scatter using nccl.
The standalone scripts stage `nccl_perf.txt`, `oneccl_perf.txt`, and
`custom_allreduce_perf.txt`. Collector finalization converts accepted output
to parquet under
`python/aisim/src/aisim_core/systems/data/<system>/comm/<backend>/<version>/`.

# Model-centric cases and healing runs

TRT-LLM MoE workers refuse a case before module allocation when its local
resident expert weights alone exceed live device capacity. This lower bound
accounts for gated/non-gated matrices and TP/EP sharding; passing it does not
establish room for scales, inputs, outputs, graph buffers or native workspace.
Those later allocation failures remain recorded failures rather than hidden
shape exclusions. Plans remain independent of device memory.

The model-centric case layer (introduced as "collector v2") is how collection
coverage is declared, and it is unchanged under Collector V3 — V3 adds
op-centric *management* on top (per-family runtime pins, provenance sidecars,
declared reuse), it does not replace how cases are declared or healed. A healing run collects
cases for a specific model/GPU pair instead of running every op bucket and
hoping the support matrix improves. Use `--model-cases-full` when you want
the case YAML to define a full model-centric run. Omitting all model-case
flags runs the backend registry directly without a model-centric case plan.

```bash
# Heal one model on one GPU type.
python3 collect.py --backend sglang \
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \
  --gpu b200_sxm

# Inspect the resolved model plan without collecting.
python3 collect.py --backend sglang \
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \
  --gpu b200_sxm \
  --plan-only

# Select the architecture case file directly.
python3 collect.py --backend trtllm \
  --model-architecture Qwen3MoeForCausalLM \
  --gpu b200_sxm \
  --plan-only

# Full model-centric run: aggregate base ops plus every model case YAML file.
python3 collect.py --backend trtllm --model-cases-full

# Raw backend registry run: no model-centric case plan.
python3 collect.py --backend trtllm

# Ephemeral subset filter (healing): substring match, repeatable.
python3 collect.py --backend trtllm --ops moe --case-filter "tp=4"
```

Case files (see `cases/README.md` for the full authoring guide):

```text
cases/base_ops/<op>.yaml               — shared sweep recipes (interpolation grid)
cases/models/<architecture>_cases.yaml — op activation + model shape values
model_cases.py                         — resolves which ops a model plan collects
skips.py                               — the one list of cases a collection leaves out
```

The plan is one equation: cases = dedup(base grid ∪ model shapes), minus what
`skips.py` skips. For an op with an op plan (`OP_PLANS` in `op_plans.py`),
`--plan-only` with `--sm` or `--gpu` also prints case counts per dtype and what
was left out under which entry; `--plan-table DIR` writes one CSV row per case.

TRT-LLM MLA precision and GPT-OSS TP profile collection evidence is recorded
in [the September 17 campaign](trtllm/gym-profiles-20260917.md).

## Whole-forward FPM campaign

Use the dedicated `python3 -m collector.fpm_forward` entry point for
whole-forward campaigns. `--backend` selects the engine: `vllm`, the default of
this entry point, runs Dynamo's vLLM worker as described here, and `sglang` runs
SGLang's own engine as described under [SGLang cells](#sglang-cells). The
compatibility route
`python3 collector/collect.py --ops fpm_forward` remains supported and must be
selected alone. Both routes require a resolved model, `--gpu`, and
`--fpm-max-gpus`. The campaign derives a latency-blind prefill/decode
design from the model's AIC attention cases, asks Generator for three rendered
artifacts per cell — a keepalive Pod, LeaderWorkerSet, or Grove PodCliqueSet
manifest plus `fpm_env.sh` and a thin engine-launch `run.sh` — stages them with
its own in-pod runtime `fpm_exec.sh` (etcd lifecycle, result gate and checker,
completion barrier, exit-code policy), and publishes the formal table only
after every frozen cell passes.
For multinode runs, select the installed controller with
`--fpm-orchestrator lws|grove`. Start with `--plan-only` or a one-cell
`--smoke --limit 1` run.

```bash
python3 -m collector.fpm_forward \
  --model-path nvidia/GLM-5.2-NVFP4 \
  --gpu b200_sxm \
  --fpm-max-gpus 8 \
  --plan-only
```

For profile-based FPM collection, the profile's `backend_version` is the dataset
identity label. It may be a custom string such as `my-vllm-patch-3` and need not
match the installed vLLM package version. The collector retains the actual
container version separately in `collector-provenance.json` under
`runtime.backend_version`; runtime capability, benchmark-schema, topology and
precision checks still apply. An optional Dynamo version is provenance only and
does not select FPM templates or resolve the backend version. See the
[onboarding version contract](../../../docs/perf-model/fpm-self-service/implementation.md#plan-preview-and-explicitly-execute)
for guided setup, omitted-version detection inside the target image, and
publication identity.

The low-level entry point also accepts `--fpm-backend-version LABEL` for dataset
identity and `--fpm-runtime-backend-version VERSION` for an independently observed
package-version pin. Guided onboarding supplies that pin after its image
preflight. The low-level collector does not launch this separate CPU preflight:
without an observed pin it can collect timings and record actual package
provenance, but it cannot select audited version-specific memory/execution hooks.
Supply the observed pin or accepted runtime instrumentation when those hooks are
required. The op-level collector's runtime manifest and version-routing rules
below are unchanged.

Generation-time admission may omit a topology only when a concrete AIC
size-vs-capacity estimate proves its configured token envelope cannot fit.
Missing performance data, bootstrap architectures, and structural estimator
errors remain runnable so the target runtime supplies the observation.

Model identity is resolved from a local checkpoint/config, the packaged AIC
config cache, or Hugging Face. A real model architecture that is not registered
by AIC uses a config-derived bootstrap template; an unresolved or empty config
fails before planning. When the checkpoint exists only inside the runtime Pod,
pass its real local metadata explicitly with
`--fpm-model-config /path/to/config.json` (a directory containing `config.json`
is also accepted). The resolved config snapshot and SHA-256 are frozen into the
collection plan.

The collector detects a cache layout that `config.json` cannot describe, such as
linear-attention or recurrent state, with the same check as config intake. The
[onboarding guide](../../../docs/perf-model/fpm-self-service/implementation.md#cache-geometry-pending-the-runtime-probe)
lists the markers. No FPM attention template represents such a layout. Without
`--fpm-model-profile` or `--fpm-backend-version`, the collector refuses it, and
registered model cases do not change this. With one of these options, the
collector gives the layout the `native_runtime` capability. Precision then comes
from the checkpoint, and the runtime measures every cell.

Generator and Collector share the accepted Dynamo-native benchmark result
schema versions (2, and 3 with a power window) through the rendered
`fpm_env.sh`; the staged `fpm_exec.sh` checks each rank result's schema version
and completion in-pod without a Collector-side text patch. It does not inspect
the power block: the pod preflight rejects an image that does not record window
step timings, and the Collector's native artifact reader rejects a schema-3
power block without `step_wall_times_s`. `--fpm-measure-power` (default on; never run by
`--smoke`) requests the per-point power window and publishes `power_w`; see the
[onboarding guide](../../../docs/perf-model/fpm-self-service/implementation.md#how-the-collection-grid-is-determined)
for its requirements and publication rules. Formal
`fpm_forward_perf.parquet` output uses metadata schema 7, or schema 8 once any
row comes from a power-window (schema-3 artifact) run, even if every
`power_w` is null (see the
[schema-8 contract](../../../docs/perf-model/methods/whole-forward.md#measured-power-schema-8)), and records
`collector_attempt_id`, `runtime_run_id`, `runtime_grid_digest`, and
`kv_seed_regime` on every row. The seed regime is `real_kv`, `fake_fallback`,
an approved `skip:<reason>`, `legacy`, or `n/a`; non-benign warm-up skips
remain `fake_fallback` rather than acceptable evidence. For a cache layout that
`config.json` cannot describe, a cached prefill needs real prefix seeding.
Planning refuses `DYN_BENCH_PREFILL_REAL_SEED` set to `off`, `0` or `false`, and
publication refuses a cached prefill without the native `real_prefix` regime.
Decode rows keep the classification above. Republishing the same
attempt is idempotent. If a cell already exists under a different attempt or
runtime identity, first publisher wins: that entire cell is skipped while
non-overlapping cells in the same publication still land.

The default publication gate requires every frozen cell to pass.
`--fpm-publish-partial` explicitly publishes passed cells, records the exact
`missing_cells` in the checkpoint, and still returns a nonzero incomplete-run
result. The database checkpoint also records
`skipped_first_publisher_wins`. A validated completed schema-7 or 8 database
is terminal on resume even if raw cell artifacts have since been pruned.
A completed schema-6 commit record cannot be finalized by this collector.
A latency-only publication into a destination containing a schema-6 pair
merges its rows, assigns them legacy execution identity, and writes a
schema-7 commit record. Publish to a fresh output directory to leave the
schema-6 pair unchanged.

To resume a campaign created before power collection was introduced, pass
`--no-fpm-measure-power` along with its original arguments and `--resume`.
The default now enables power, which changes the frozen plan and produces
`FPM checkpoint does not match the current frozen plan` for that archived
campaign. Keep power enabled only in a new campaign; do not edit the old
plan or checkpoint to bypass the mismatch.

`run-manifest.json` schema v2 is invocation-scoped: its `attempts` include the
attempt ID, timestamps, final status, and partial phase timings for failed
attempts; a no-op resume writes no historical attempts.

### SGLang cells

`--backend sglang`, on either route, collects cells from SGLang itself: the
tensor-parallel cells of a dense model, and the one-GPU cell, the pure TP
cells, the TEP cells and the DEP cells of an MoE model
([MoE cells](#moe-cells)). A cell starts
`runtime/sglbench_launcher.py` in an SGLang image,
with no Dynamo worker and no etcd. The launcher builds SGLang's engine in its
own process, plans the points from the limits and CUDA graph sizes that engine
reports, and writes the rank and merged artifacts of the same result schema as
the vLLM producer. Staging, the in-pod gate, validation and publication are
therefore the ones described above, and the rows are published with
`backend: sglang` under the same measurement policy.

```bash
python3 -m collector.fpm_forward \
  --backend sglang \
  --model-path Qwen/Qwen3-0.6B \
  --gpu h200_sxm \
  --fpm-max-gpus 2 \
  --fpm-prefill-cudagraph-policy runtime \
  --fpm-runtime-backend-version 0.5.23 \
  --fpm-backend-version 0.5.23 \
  --generator-set K8sConfig.k8s_image=<SGLang 0.5.23 image> \
  --fpm-database-root <systems data root> \
  --plan-only
```

An SGLang plan must give:

| Option | Why |
|---|---|
| `--fpm-runtime-backend-version` | The SGLang release of the image. The launcher accepts `0.5.22` and `0.5.23` (`TESTED_VERSIONS` in `runtime/sglbench_compat.py`), and the pod refuses any other installed release before the engine starts. |
| `--fpm-backend-version` | The data label: the table directory and the `backend_version` and `aic_database_version` of every row. Without it the plan would take the newest curated SGLang version as its database version (`aic_database_version`) and check precision against that version's operator tables, while `backend_version` and the directory carried the pod's release. |
| `--fpm-prefill-cudagraph-policy runtime` | The default policy passes a vLLM capture list. SGLang captures its own graph sizes, and the launcher reads them from the engine. |
| An image | On Kubernetes `K8sConfig.k8s_image`, through `--generator-config` or `--generator-set`: the Generator's default is a Dynamo vLLM image. On Slurm `--fpm-slurm-container-image`. |

It may not give `--fpm-model-profile`, `--fpm-runtime-instrumentation`,
`--fpm-runtime-launch`, `--fpm-runtime-configuration`, `--fpm-cudagraph-mode`,
`--fpm-cudagraph-capture-sizes`, `--fpm-max-cudagraph-capture-size`,
`--fpm-enforce-eager`, `--fpm-decoder-replay`, or an explicit backend identity
(`--fpm-moe-backend`, `--fpm-attention-backend`, `--fpm-enable-wideep true`,
`--fpm-enable-eplb true`). Planning also refuses a model with an FPM execution
identity (DeepSeek V4.1, GLM-5.3-Flash) and, for an MoE model, every cell
other than the one-GPU cell, the pure TP cells, the TEP cells and the DEP
cells ([MoE cells](#moe-cells)).
Repeatability validation and the runtime probe remain vLLM only.

An SGLang cell runs on one node. Planning does not know how many GPUs a node
holds, so a plan whose `--fpm-max-gpus` exceeds one node is admitted and each
of its multi-node cells is refused when it is rendered, before a pod is
scheduled. Such a campaign stays incomplete and publishes only with
`--fpm-publish-partial`. A cell is refused in the same way when its KV-cache
dtype is one SGLang does not take (`int8`), when `K8sConfig.extra_env`
switches `DYN_BENCH_KV_WARMUP` or `DYN_BENCH_PREFILL_REAL_SEED` off (the
launcher stages real cached state only), or when it sets
`DYN_BENCH_PREFILL_CONTENT` to a word the launcher does not accept, or to
`random` in the plan of an MoE model ([MoE cells](#moe-cells)).

The resource Pod and the environment of an SGLang cell come from the
Generator's vLLM FPM render of the same cell. `sglang_fpm_builder.py` asks for
that render, translates the worker command flag by flag into the launcher's and
SGLang's flags, and switches the gate to its mode without etcd. A flag without
a translation refuses the cell. The cell stages every `runtime/sglbench_*.py`
file, which are the launcher's modules and `sglbench_preflight.py`, in the place
of `preflight.py`.

`collector-provenance.json` records `runtime.backend` as `sglang` and the
version of the installed `sglang` distribution as `runtime.backend_version`.
Validation requires that version to equal the `producer.sglang_version` the
launcher wrote into every rank artifact. Publication into the packaged data
tree refuses a data label that has no measured data there, and the tree holds
none for SGLang 0.5.22 or 0.5.23, so a formal SGLang table labelled with its
release needs `--fpm-database-root`.

The prefill table of an SGLang cell starts at one new token per request and
holds both sides of the first step of SGLang's prefill graphs. SGLang runs a
prefill batch with a captured graph only when the size it pads the batch to
holds at most twice the batch's own tokens. Its default sizes start at 4, so
a batch of one new token runs without a graph and a batch of two to four
with the graph of 4, which is another time: in the MoE cells collected on
H200 a batch of one new token took 36 to 48 ms and a batch of four took 5 to
7 ms. The launcher therefore plans 1 and 2 new tokens per request beside the
graph sizes (`prefill_new_token_axis` in `runtime/sglbench_grid.py`), and
the two values do not count against `--prefill-max-new-token-samples`.

The 2 was added on 2026-11-23. Every cell named in this file, except the DEP
cells and the pure TP cells collected that day, was collected before that, with an axis of 1, 4, 5,
8 and so on, and the point counts given for those cells are the counts of
that grid. From such a table AISim's
reader prices a prefill of 2 or 3 new tokens on the line between a row
without a graph and a row with one: for 8 such steps of a served engine the
prediction was 3.7 times the measured time. A cell collected now plans the
2 at each of its batch sizes: 3,347 points where the formal DEP prefill cell
had 3,314, and 8 where a smoke prefill cell had 6. Its grid digest differs,
so rank files of an earlier collection cannot be mixed with new ones, and
its rows reach a table only in a root that does not hold the cell yet
(first publisher wins). The rule is read from SGLang's source at both
releases, and the unit tests publish a grid with typed-in times and read it
back at 2 and 3 new tokens. On an engine the row has been collected with the
default grid in two prefill cells of `Qwen/Qwen1.5-MoE-A2.7B` at SGLang
0.5.23 on two H200 GPUs, both on 2026-11-23: the DEP cell
([DEP cells](#dep-cells)) and the pure TP cell ([MoE cells](#moe-cells)). In
the DEP cell one request of two new tokens on each rank took 8.2 to 8.8 ms,
as one of four new tokens did, and one of one new token 45 to 50 ms. SGLang's
own batch lines show every batch of one request with two new tokens with a
captured graph, and from that table the reader answers 8.3 ms for 2 and for 3
new tokens. In the pure TP cell one request of two new tokens took 5.1 to
5.8 ms, one of four 4.9 to 6.1 ms and one of one new token 36 to 39 ms. Smoke
grids held the row in the DEP, pure TP and TEP prefill cells at 0.5.23 and in
the pure TP prefill cell at 0.5.22.

An SGLang table carries these deviation notes. Each lives at the code that the
item names. The first two are differences from AISim's generated SGLang
serving deployment, and the last two apply to DEP cells only
([DEP cells](#dep-cells)):

- Mixed chunk and the decode graph list (`sglang_fpm_builder.py`). The
  generated serving deployment switches mixed chunk on and passes its own
  decode graph list. The launcher refuses mixed chunk and the cell passes no
  graph list, so the table measures SGLang's defaults for both.
- Context length (`sglang_fpm_builder.py`). `--fpm-max-model-len -1`, the
  default, means vLLM's auto-fit. For SGLang no context length is passed and the
  model's own limit applies.
- Environment (`sglang_fpm_builder.py`). The cell keeps the environment the
  Generator renders for the hardware, for example `NCCL_CUMEM_ENABLE=1`. The
  generated serving deployment gets the same variables, so this differs only
  from a plain SGLang start, where SGLang chooses that variable itself.
- Releases (`planner.py`, `_require_sglang_launcher_inputs`). The launcher
  accepts two SGLang releases, and neither is the collector's operator pin
  (`0.5.21` in `framework_manifest.yaml`). The formal FPM path does not consult
  that pin for any backend.
- KV cache of a `bfloat16` cell (`sglang_fpm_builder.py`). A cell whose
  KV-cache dtype is `bfloat16` passes `--kv-cache-dtype auto`, as the
  generated serving deployment does, where the vLLM render of the same cell
  passes `bfloat16`. Its rows carry `bfloat16`. For some checkpoints SGLang
  makes an FP8 KV cache of `auto`, and neither the launcher nor the host
  compares the type SGLang stored with the label of the rows.
  [KV cache of a bfloat16 cell](#kv-cache-of-a-bfloat16-cell) names the
  checkpoints and says what to do today.
- Static memory of a DEP prefill cell (`sglang_fpm_builder.py`). The cell
  writes the chunk size as the token budget times the number of ranks, so
  that each attention-DP rank gets the whole budget, as a vLLM rank does.
  SGLang works out the share of GPU memory for weights and KV pool from the
  written number. With the default budget the KV pool of a rank is then
  smaller than with the budget written unchanged, and smaller still with
  more ranks, unless the plan sets `--fpm-gpu-memory-utilization`. The cell
  passes a share only when the plan gives one. "DEP cells" below has the
  measured numbers, how a pool that is too small shows, and what to set
  then.
- Input blocker of a DEP cell (`runtime/sglbench_launcher.py`). The launcher
  switches on SGLang's input blocker, which serving does not use. Its poll
  over all ranks is inside every measured decode step; a prefill time also
  spans one receive step with the poll, and its effect there was not
  measured.

#### KV cache of a bfloat16 cell

This is a known gap, not closed. A plan gives a cell the KV-cache dtype
`bfloat16` when `--fpm-kv-cache-dtypes` names it, and when the option is
left out and AISim takes the checkpoint's own KV cache for bfloat16. The
cell passes `--kv-cache-dtype auto`, and every row of it carries `bfloat16`
as `kv_cache_dtype`. With `auto` SGLang stores the dtype it loads the model
in, unless one of its own rules puts an FP8 KV cache (`fp8_e4m3`) in its
place. The rules were read in SGLang's source at 0.5.22 and 0.5.23, not
run:

- A DeepSeek sparse-attention model on a GPU of compute capability 10 or
  above, the B200 class (`_dsa_kv_cache_dtype_default` in
  `srt/arg_groups/overrides.py`). SGLang takes a model for one by its
  architecture name and an `index_topk` in its config (`is_deepseek_dsa` in
  `srt/configs/model_config.py`); the packaged configs of the architectures
  `DeepseekV32ForCausalLM`, `GlmMoeDsaForCausalLM` and
  `Glm5NextForConditionalGeneration` are of this kind. The last keeps its
  `index_topk` in `text_config`, and SGLang's config class copies it to the
  top (`srt/configs/glm5_next.py`). A config with `learnable_sink` keeps
  bfloat16, and so does every such model on a GPU below that capability, an
  H200 for one.
- `DeepseekV4ForCausalLM`, on every NVIDIA GPU
  (`_deepseek_v4_kv_cache_dtype` in the same file).
- At 0.5.23 only: `MuseSparkForConditionalGeneration`
  (`_muse_spark_overrides` in the same file), and a MiniMax-M3 checkpoint
  that declares MXFP8 weights with NVFP4 packed experts. On a Blackwell GPU
  SGLang turns the argument into `fp8_e4m3`
  (`srt/arg_groups/model_overrides/minimax_m3.py`). On another GPU the
  argument stays `auto`, and a comment in that file says the KV dtype then
  stays the model's. The quantization config SGLang builds for such a
  checkpoint always names FP8 as its KV-cache algorithm, though
  (`srt/layers/quantization/minimax_m3_nvfp4.py`), so by the next rule the
  engine would store FP8 there as well. Which of the two holds on a GPU
  that is not Blackwell was not run.
- A checkpoint whose quantization config asks for an FP8 KV cache: a
  `kv_cache_quant_algo` of `FP8`, for a ModelOpt checkpoint a
  `kv_cache_scheme` of 8-bit floats
  (`srt/layers/quantization/modelopt_quant.py`), or for a
  compressed-tensors checkpoint a static, symmetric `kv_cache_scheme` of
  8-bit floats for the whole tensor
  (`srt/layers/quantization/compressed_tensors/compressed_tensors.py`). The
  first three rules act when SGLang resolves its arguments. This one acts
  later, in each scheduler process once it has loaded the model
  (`configure_kv_cache_dtype` in `srt/mem_cache/kv_cache_dtype.py`). It
  reads the quantization config that the loaded model keeps. Not every
  model class keeps one, and which of a checkpoint's config files SGLang
  builds it from depends on the checkpoint, so a key in `config.json` or
  `hf_quant_config.json` is a strong sign and not the proof.

Where no rule applies, the dtype of the model is not always bfloat16
either. For a config that names float16 it is float16, and for one that
names float32 or no dtype it is float16 unless the model is a Gemma
(`_get_and_verify_dtype` in `srt/configs/model_config.py`).

Neither the launcher nor the host checks any of this today. No result file
records the type SGLang stored and validation compares none, so a
`bfloat16` cell of such a checkpoint completes, and its rows, measured with
an FP8 KV cache, are published under the label `bfloat16`.

The 97 packaged model configs were held against the rules on 2026-11-24:
the architecture names and config keys above, beside the KV-cache dtype
AISim itself takes for each checkpoint. No engine ran for it, and whether
SGLang loads each of these checkpoints was not checked. The configs under
a rule are of two kinds:

- For most of them AISim takes the checkpoint's KV cache for FP8 as well. A
  plan without `--fpm-kv-cache-dtypes` then holds `fp8` cells, which pass
  `--kv-cache-dtype fp8_e4m3`, so the label is the type SGLang is told to
  store. These are the DeepSeek V3.2 and V4 configs, the GLM-5 and
  GLM-5.3-Flash configs that AISim takes for FP8, and the other checkpoints
  with FP8 or NVFP4 weights whose quantization config names an FP8 KV
  cache. With `--fpm-kv-cache-dtypes bfloat16` their plans hold `bfloat16`
  cells, and each such cell that renders and runs is measured with an FP8
  cache where its rule gives one.
- For four, AISim takes the KV cache for bfloat16 where SGLang's first rule
  does not: `zai-org/GLM-5`, `zai-org/GLM-5.1`, `zai-org/GLM-5.2` and
  `zai-org/GLM-5.3-BF16`. Their plans hold `bfloat16` cells without any
  option, and on a B200-class GPU SGLang would run them with an FP8 cache.
  Their architecture is `GlmMoeDsaForCausalLM`, whose cells are refused
  when they are rendered ([MoE cells](#moe-cells)), so none of them
  publishes such a row at these releases. No other packaged config that
  AISim takes for bfloat16 is under a rule.

No packaged config has the Muse Spark architecture, and the packaged
`nvidia/MiniMax-M3-NVFP4` config does not meet the MiniMax-M3 rule: it
names `modelopt` as its quantization method and no KV-cache algorithm. A
checkpoint that is not packaged falls under the same rules by its own
config.

The models collected so far, `Qwen/Qwen3-0.6B` and
`Qwen/Qwen1.5-MoE-A2.7B`, are not affected. Neither architecture is under a
rule, neither config has a quantization config, and both name bfloat16 as
their dtype. The resolved arguments kept from their runs at both releases
hold `auto` as the KV-cache dtype; those files are not in this repository.

What to do today:

- **Plan the cells as `fp8`.** For a checkpoint under one of the rules, on
  a GPU where the rule gives FP8, `--fpm-kv-cache-dtypes fp8` plans `fp8`
  cells. Such a cell passes `--kv-cache-dtype fp8_e4m3`, and its rows carry
  `fp8`, the type the cell asks SGLang for. No option of a plan makes an
  SGLang cell pass `--kv-cache-dtype bfloat16`: `bf16` and `bfloat16` both
  name the `bfloat16` cell, which passes `auto`.
- **Check a collected `bfloat16` cell by hand.** The launcher writes
  SGLang's resolved arguments to
  `cells/<cell_id>/raw/<pod>/resolved-config-node0.json` before the engine
  is built. Its `config.server_args.kv_cache_dtype` shows what the first
  three rules made of `auto`: read in SGLang's source, it is `fp8_e4m3`
  where one of them gave FP8, and no such file has been seen. The fourth
  rule acts after the file is written, so the file still shows `auto` where
  only that rule applies; the MiniMax-M3 checkpoint above on a GPU that is
  not Blackwell would be such a case. For it, read the `quantization_config`
  or `compression_config` of the checkpoint's `config.json` and its
  `hf_quant_config.json`.

#### MoE cells

For an MoE model an SGLang plan holds four kinds of cell: the one-GPU cell
(strategy `single`); the pure TP cell (strategy `pure_tp`), where every
tensor-parallel rank holds a slice of every expert; the TEP cell (strategy
`tep`), which has one data-parallel rank and one expert-parallel rank per
tensor-parallel rank; and the DEP cell (strategy `dep`), which has one GPU and
one expert-parallel rank per data-parallel rank and runs with SGLang's DP
attention ([DEP cells](#dep-cells)). The default preset `auto` plans a pure
TP, a TEP and a DEP cell for every GPU count above 1, as it does for vLLM.
`--fpm-parallel-presets pure_tp`, `tep` or `dep` plans one of the three, and
several presets can be given. A plan of one GPU only holds the one-GPU cell
under `auto`, `pure_tp`, `tep` and `dep` alike.

Pure TP was added to the SGLang plan on 2026-11-23, after TEP and DEP. `auto`
planned 3, 5 and 7 cells per phase for `--fpm-max-gpus` 2, 4 and 8 before,
and plans 4, 7 and 10 now when every cell fits the GPUs.
`--fpm-parallel-presets tep dep` plans the earlier set of cells. A width
filter binds only the presets that have its axis. Under `auto` an
expert-parallel or a data-parallel filter (`--fpm-moe-ep-sizes`,
`--fpm-dp-sizes`) therefore leaves the pure TP cells in the plan, as it does
for vLLM; name the presets to get the plan such a filter gave before. The
plan records whether pure TP is admitted, so the plan hash of every SGLang
plan of an MoE model changed with it, also for a plan whose cells are the
same: a campaign that was frozen before cannot be resumed. The ids of the
earlier cells did not change, so a table that holds their rows keeps them
(first publisher wins). Rows collected now carry the new plan hash in
`source_plan_sha256`.

The plan of one and two GPUs with the TEP preset:

```bash
python3 -m collector.fpm_forward \
  --backend sglang \
  --model-path Qwen/Qwen3-30B-A3B \
  --gpu h200_sxm \
  --fpm-max-gpus 2 \
  --fpm-parallel-presets tep \
  --fpm-prefill-cudagraph-policy runtime \
  --fpm-runtime-backend-version 0.5.23 \
  --fpm-backend-version 0.5.23 \
  --generator-set K8sConfig.k8s_image=<SGLang 0.5.23 image> \
  --fpm-database-root <systems data root> \
  --plan-only
```

The vLLM render of a TEP or a DEP cell carries the switch
`--enable-expert-parallel`. The cell builder writes it as `--ep-size` with the
value it wrote for `--tp-size`, right after that flag. For a TEP cell that is
the tensor-parallel size; for a DEP cell it is the number of data-parallel
ranks ([DEP cells](#dep-cells)). This is the size AISim's generated SGLang
serving deployment passes under SGLang's alias `--expert-parallel-size`. A
one-GPU cell and a pure TP cell get no `--ep-size`. SGLang's expert-parallel
size is then its default of 1, and SGLang gives every tensor-parallel rank a
slice of every expert: the command of a pure TP cell of N GPUs is `--tp-size
N` alone. AISim's generated SGLang serving deployment asks for the same
layout with `--tensor-parallel-size N --expert-parallel-size 1`. No other MoE
flag has a translation. The
launcher then checks what SGLang resolved. It requires an expert-parallel size
of 1 or of SGLang's `tp_size`, a `moe_dp_size` of 1, no all-to-all backend
(`moe_a2a_backend` is `none`), no separate TP size for the dense MLP layers
(`moe_dense_tp_size`), no EPLB and no redundant experts, and it refuses an
expert-parallel size above 1 for a model without expert layers. The launcher
is not told which layout its cell is: a size of 1 and a size of `tp_size` both
pass, and validation holds the recorded size against the cell (below). A
layout with both expert sizes above 1, DP attention in any layout other
than DEP, elastic EP, a model with hybrid sliding-window or state-space layers
and a model with an FPM execution identity stay refused, each at its own step:

- Planning refuses an execution identity and every topology that is none of
  the four kinds of cell, for example tensor parallelism inside a
  data-parallel rank, or an MoE tensor-parallel size and an expert-parallel
  size that are both above 1. The message names the topology.
- A cell whose vLLM command holds a flag without a translation is planned and
  refused when it is rendered, before a pod is scheduled. The cells of a
  `GlmMoeDsaForCausalLM` model are refused in this way (`--trust-remote-code`
  and `--reasoning-parser`), and so are those of some other packaged MoE
  models: some DeepSeek V4 checkpoints (`--trust-remote-code`) and MiniMax-M3
  (`--block-size`).
- The launcher refuses elastic EP, and DP attention in any layout other than
  the DEP layout, in the pod, when it checks what SGLang resolved, before the
  engine is built.
- A hybrid model is planned and rendered. The launcher refuses it in the pod
  only after the engine is built, because SGLang decides by its own lists of
  architectures, which planning does not read. By those lists about half of
  the packaged MoE models that plan and render are hybrid at both releases,
  among them Qwen3.5, Nemotron 3, Kimi K3, Llama 4, Gemma 4, DeepSeek V4,
  MiMo-V2 and gpt-oss. Try one cell with `--smoke --limit 1` before a
  campaign.
- A TEP or a DEP cell whose GPU count does not divide the model's number of
  experts is planned and rendered and passes the launcher's checks. SGLang
  then stops while it builds the engine, at an assertion of its MoE layer on
  the expert-parallel size. This is read in SGLang's source at both releases
  and was not run. No packaged MoE model has a number of experts that 2, 4 or
  8 does not divide.
- A pure TP cell of an MoE model whose cells are planned and rendered passes
  the launcher's checks, and SGLang can still refuse it while it builds the
  engine. Its rules are on the slice of an expert that one rank holds, which
  is the expert's intermediate size divided by the GPU count. Its MoE layer
  asserts that the GPU count divides the intermediate size; no packaged MoE
  model has an expert intermediate size that 2, 4 or 8 does not divide. For
  a checkpoint with block-wise FP8 weights the slice must also be a whole
  number of weight blocks, and the packaged configs of five checkpoints
  break that on 4 or on 8 GPUs. This is a known limitation of the default
  preset.
  [Pure TP cells that SGLang refuses to build](#pure-tp-cells-that-sglang-refuses-to-build)
  names the checkpoints, says what a refused cell looks like and gives the
  ways around it. The rules are read in SGLang's source at both releases, and
  the block rule ran on an engine once.

Which experts a token is sent to depends on the token, so the cells of an MoE
model send dataset text. In such a run every prompt is text: the cached
prefixes, the new tokens of a prefill point, the decode contexts, the warm-up
and the self-test. A prompt part is a chain of whole conversations of the
dataset, tokenized by the engine's tokenizer and put in an order that the
content seed fixes, each conversation at most once per chain. The first token
of each prompt part is not text: the launcher chooses it so that no two
prompts share a cached prefix that the point did not ask for
(`runtime/sglbench_content.py`).

`DYN_BENCH_PREFILL_CONTENT` chooses the content of a cell:

| Value | Content |
|---|---|
| unset or empty | Dataset text for a model with expert layers, seeded token ids for any other model. |
| `random` | Seeded token ids. The launcher refuses this value for a model with expert layers. |
| `sharegpt_chain` | Dataset text for any model. Set it to collect a dense model with text, for example to try the text path on a small model. |

Any other value is refused, `sharegpt` and another spelling of the two words
included. A cell whose `K8sConfig.extra_env` sets a refused value, or sets
`random` in the plan of an MoE model, is refused when it is rendered, before a
pod is scheduled. The launcher makes the same test in the pod and exits with
status 64. Only that test catches `random` for a model that the plan does not
take for an MoE model and the launcher does: the launcher counts a truthy
value of any of six expert keys of the model config (`EXPERT_KEYS` in
`runtime/sglbench_compat.py`), the plan only an expert count above 1. Only
that test also sees a value that reaches the pod in another way than the
cell's inputs, for example from the image. Three more variables name the
dataset. They have the meaning they have in Dynamo's vLLM self-benchmark, with
two differences: an empty value counts as unset, and the hash of a local path
is compared too.

| Variable | Meaning | When unset |
|---|---|---|
| `DYN_BENCH_KV_WARMUP_DATASET` | A URL (`http://` or `https://`), or the path of a file in the pod. | The file `ShareGPT_V3_unfiltered_cleaned_split.json` of the Hugging Face dataset `anon8231489123/ShareGPT_Vicuna_unfiltered`, the default of Dynamo's vLLM self-benchmark. |
| `DYN_BENCH_KV_WARMUP_CACHE_DIR` | The folder where a URL's file is looked for under its own name, and where it is downloaded when it is not there. | `fpm_datasets` beside the folder `HF_HOME` names, or `/tmp/fpm_datasets` when `HF_HOME` is unset too. |
| `DYN_BENCH_KV_WARMUP_SHA256` | The SHA-256 the file must have, whether it was downloaded, found in the cache folder or given as a path. | The pinned hash of the default file when the dataset is the default URL. For any other dataset nothing is compared. |

A pod without access to the internet cannot download the file. Put it into the
cache folder under its own name, or give its path as the dataset. Either goes
into `K8sConfig.extra_env` of the file that `--generator-config` names:

```yaml
K8sConfig:
  extra_env:
    - name: DYN_BENCH_KV_WARMUP_CACHE_DIR
      value: <folder in the pod that holds the file>
```

Name the cache folder in this way whenever the pod cannot write `fpm_datasets`
beside `HF_HOME`, for example on a read-only model volume: Generator's
`run.sh` then points the variable at a new, empty scratch folder unless the
variable is already set ([A shared model-cache volume](#a-shared-model-cache-volume)).
Give a dataset other than the default together with its hash. Without
`DYN_BENCH_KV_WARMUP_SHA256` the launcher records the hash of the file it read
and compares it with nothing.

A text run never falls back to seeded token ids. When it cannot get its text,
the launcher writes a failed artifact, exits with status 65, and the cell
fails with one of four reasons, which are the names of Dynamo's vLLM
self-benchmark:

| Reason | What happened |
|---|---|
| `dataset_unavailable` | The file is missing, cannot be downloaded, cannot be read or parsed, or does not have the expected hash. |
| `dataset_empty` | The dataset holds no conversation the launcher can use. |
| `tokenizer_unavailable` | The engine's tokenizer could not turn the text into token ids. |
| `content_too_shallow` | The text pool holds too few tokens for the longest prompt of the grid, or a prompt used up the pool. |

The launcher finds, hashes and parses the dataset before it builds the engine,
so a missing or damaged dataset costs seconds and no model load. Once the
engine is built it tokenizes conversations in dataset order until its text
pool holds 8,000,000 tokens or the conversations run out. The pool keeps the
token ids of a text as the tokenizer returns them, as Dynamo's vLLM
self-benchmark does. That includes the id of a special token, which a
tokenizer returns for text that holds the written form of that token. The
pool leaves out only an id the model does not count, one at or above the
vocabulary size of its config, and counts what it left out
(`dropped_tokens` in `producer.content.pool`). That count is 0 unless the
tokenizer holds ids the model does not have. `dropped_conversations` counts
the conversations of which no id was kept: those made only of such ids, and
one for which the tokenizer returned no id at all. The token ids the
launcher makes up itself, the first token of each prompt part and the seeded
ids of a run without text, stay ordinary ids (`producer.content.token_range`).

The result file says what was sent. Its `producer.content` block holds the
content source, the dataset record and the record of the text pool: its four
counts and `id_range`, the range of token ids the pool keeps, written as
`[low, high]` for the ids from `low` up to and not including `high`, as
`token_range` is. That range holds `token_range` and is wider when the model
counts ids outside `token_range`.
The `kvwarm` block of a decode artifact names the dataset with the four keys
Dynamo writes (`path`, `sha256`, `collection_pool`, `conversations`). The
checkpoint keeps that block as `native_kvwarm` in the record of a decode cell.
The published rows have no column for the content: `kv_seed_regime` says how
the cached state was produced, not which tokens it holds. The rows of a TEP
cell carry `parallel_strategy: tep` and the cell's `moe_ep`; those of a pure
TP cell carry `parallel_strategy: pure_tp`, the cell's `moe_tp` and a `moe_ep`
of 1; those of a DEP cell are described under [DEP cells](#dep-cells).

The rows take those labels from the plan, so for a cell of an MoE plan
validation holds what the pod recorded against the cell:
`producer.content.source` must be `dataset_text_chains`, and the three sizes
SGLang resolved must be the ones of the cell. SGLang's `tp_size` counts every
rank of the engine, so `producer.tp_size` must be the cell's `tp` times its
`dp`, `producer.moe.ep_size` its `moe_ep`, and `producer.attn_dp_size` its
`dp`. Each size must be there as an integer. A producer block without
`attn_dp_size` is refused in every cell: the launcher writes the key into
every result file of a run that completed, with 1 for a cell without DP
attention. It wrote no such key before it could run DP attention; a result
file of that time belongs to a plan of that source revision and is refused
under a plan made now for its plan hash, before this comparison. Where
`FPM_COLLECTOR_SOURCE_REVISION` holds the revision at one value, such a
file reaches this comparison and is refused by it.
For a pure TP cell of N GPUs the three sizes are N, 1 and 1, and for
its TEP cell N, N and 1: the expert-parallel size is all that tells a result
of the one from a result of the other. Two kinds of run
complete in the pod and are refused by this comparison. A
pod whose launcher finds no expert key in the model config (`EXPERT_KEYS`)
sends seeded token ids unless `sharegpt_chain` is set, and records no
`producer.moe` block. A
command whose `--ep-size` does not fit its cell runs the other layout, which
the launcher accepts without DP attention: the command of a TEP cell that
reaches the pod without the flag, or that of a pure TP cell with it. The cell
fails with `native SGLang rank artifact of a MoE plan did not send dataset
text` or `... did not run the layout of its cell`. The comparison is made
when a cell has run, when a resume looks at the files of a cell again, and
when rows are built. The plan of a dense model makes no such comparison.

A cell of a dense model has the same cell id and the same row labels whether
it sent dataset text or seeded token ids: apart from the measured times, only
`source_plan_sha256` and the two run ids (`collector_attempt_id`,
`runtime_run_id`) of its rows differ. A table root that already holds the cell
keeps the rows it has (first publisher wins, as described above), and the
`database` block of the checkpoint still says `passed`. Publish the text run
of a dense model into a root of its own.

Rows depend only on the pin and the plan, never on environment variables. The
default dataset and its hash are constants of the launcher's code, so they are
part of the pin, as is the scratch folder `run.sh` chooses by itself. Every
other dataset, cache folder, hash or content value must be given through
`K8sConfig.extra_env`: the plan hash covers that input, so a changed value is a
new campaign. A variable set in the image or in any other way is outside the
plan.

The unit tests cover this path without a GPU: the plan, the command of each
cell, and the validation and publication of result files whose times were
typed in. On H200, the cells of `Qwen/Qwen1.5-MoE-A2.7B` have been collected
with the default grid as it was before 2026-11-23, without the row of 2 new
tokens per request ([SGLang cells](#sglang-cells)): the one-GPU cells at
SGLang 0.5.22 and the TEP cells on two GPUs at 0.5.23. The texts of those cells gave no token id outside the
ordinary ids, so a prompt that holds the id of a special token has run in the
unit tests only. The DEP cells of that model on two GPUs have run with smoke
grids at both releases; [DEP cells](#dep-cells) says what ran and what did
not.

The pure TP cells of that model on two GPUs ran on 2026-11-23, with the row
of 2 new tokens: with the default grid at SGLang 0.5.23 (3,347 prefill and
890 decode points, none staged again, rows published as `pure_tp`) and with
smoke grids at both releases. Each cell ran `--tp-size 2` with a resolved
expert-parallel size of 1 and passed validation with the recorded sizes 2, 1
and 1. One smoke campaign at 0.5.23 used the default preset and held the DEP,
the pure TP and the TEP cells of two GPUs. Held against the cells of the
other two layouts, each of its results was refused by the comparison above.
Not run to a result on an engine: a pure TP cell of more than two GPUs or of
another model, and the default grid at 0.5.22. A cell that SGLang refuses to
build ran once, on 2026-11-24 (next section).

##### Pure TP cells that SGLang refuses to build

This is a known limitation of the default preset, left as it is. For an MoE
model that an SGLang plan admits, `auto` plans a pure TP cell on every GPU
count above 1 that passes memory admission, and for a checkpoint with
block-wise FP8 weights SGLang refuses to build some of those cells.
One refused cell ran on GPUs on 2026-11-24: the pure TP prefill cell of
`Qwen/Qwen3-30B-A3B-FP8` on four H200 GPUs at SGLang 0.5.23, through the
collector in a smoke run. SGLang's server then ran alone in a pod of the
same image and resources, with the cell's model path, `--tp-size 4` and
KV-cache dtype. Two more things ran in a pod of that image on that day and
are marked where they are used: a read of the published configs and a test
of the gate's stream filter. Everything else in this section was read in
the collector's code and in SGLang's source at 0.5.22 and 0.5.23, which
agree, or run on a development machine without a GPU. No other cell of the
checkpoints below has run in any layout.

A config with `quantization_config.weight_block_size` marks such a
checkpoint. It keeps one scale per block of a weight, 128 by 128 in the
packaged configs below. A pure TP rank holds a slice of every expert, the
expert's intermediate size divided by the GPU count, and SGLang requires the
slice to be a whole number of blocks. A TEP or a DEP rank holds whole
experts and passes this rule, and so does the one-GPU cell.

| Checkpoint | Expert size | Slice on 2 GPUs | Slice on 4 | Slice on 8 |
|---|---|---|---|---|
| `Qwen/Qwen3-30B-A3B-FP8` | 768 | 384 | 192, refused | 96, refused |
| `stepfun-ai/Step-3.7-Flash-FP8` | 1280 | 640 | 320, refused | 160, refused |
| `Qwen/Qwen3-235B-A22B-FP8` | 1536 | 768 | 384 | 192, refused |
| `MiniMaxAI/MiniMax-M2.5`, `MiniMaxAI/MiniMax-M2.7` | 1536 | 768 | 384 | 192, refused |

The sizes and key names are those of the packaged configs, and the published
configs held the same values on 2026-11-24. An entry without "refused" says
that the slice passes this rule, not that the cell builds. Every other packaged MoE config with a
weight block size passes the rule on 2, 4 and 8 GPUs.

Two rows rest on more than the others:

- `stepfun-ai/Step-3.7-Flash-FP8`. SGLang has no model class of the
  architecture name in the packaged config. The published config names
  `Step3p7ForConditionalGeneration`, which builds `Step3p5ForCausalLM`, and
  has the packaged sizes. Unlike the packaged one it asks for remote code;
  whether SGLang then needs it was not checked. SGLang lists that
  architecture as a hybrid sliding-window model, which the launcher refuses
  in every cell once the engine is built (above). No cell of this checkpoint
  is therefore expected to collect at these releases, with or without a
  filter. Its shared expert, a dense layer 1280 wide that SGLang cuts over
  all tensor-parallel ranks when no all-to-all backend is set, may also
  break SGLang's like rule for dense block-quantized layers in the TEP and
  DEP cells on 4 and 8 GPUs; that was not settled.
- The two MiniMax checkpoints. Their config asks for remote code, and a
  collector cell has no flag for it. Whether such a cell loads the
  checkpoint at all was not checked.

SGLang has the rule in two places:

- Its FP8 expert layer (`Fp8MoEMethod` in `layers/quantization/fp8.py`)
  tests the slice while the model object is built, before a weight file is
  read. For a slice of 192 its message is `The output_size of gate's and
  up's weight = 192 is not divisible by weight quantization block_n = 128.`
  This ran in the server run: on four H200 GPUs every rank raised it about
  a minute after the server command started, and that pod had fetched only
  the config and the tokenizer files by then.
  On an NVIDIA GPU no environment variable switches this test off; its one
  exemption needs an AMD build. Only the MoE runner backends
  `flashinfer_trtllm` and `flashinfer_trtllm_routed` get past it, because
  the MoE layer rounds the slice up to a multiple of 128 before the test.
- `check_quantized_moe_compatibility` tests the slice again after the
  weights are loaded. Its message begins `For quantized MoE models`. It
  reads `moe_intermediate_size`, which the packaged MiniMax configs do not
  have, so it returns early for those two checkpoints and only the first
  rule tests them. `SGLANG_SHARED_EXPERT_TP1` switches this test off and
  leaves the first. It also changes how SGLang places shared experts in
  other models, and is not a way around.

On a GPU of compute capability 10 with CUDA 12.8 or later, the B200 class,
SGLang selects `flashinfer_trtllm` by itself for a `Qwen3MoeForCausalLM`
checkpoint with FP8 weights when the runner backend is left at `auto` and no
all-to-all backend is set, as in a collector cell. The two Qwen checkpoints
then pass the first rule and are left to the second, which runs only after
every rank has loaded its weights. What happens in between was not traced.

What a refused cell looks like. Each line says what it rests on.

- The prefill cell that ran failed after its pod had started, and took
  243 s from its start to the deletion of its pod. The decode cell of a
  refused GPU count builds the same engine and was not run.
- In the server run SGLang raised in every scheduler process and logged the
  traceback, and its handler logged `Received sigquit from a child process`
  and killed the process tree. The server ended 63 s after it started. Read
  in the code, the same happens in a cell, where the signalled process is
  the launcher. The launcher therefore writes no reason line and no failed
  result file, unlike its own refusals above. The cell that ran kept the
  preflight file, the provenance file and the resolved configuration, and
  no result file.
- On Kubernetes the cell was recorded as `failed` with `remote_exit=137`.
  With the gate as it was on that day, the recorded error and
  `cells/<cell_id>/logs/<pod>.run.stderr.log` under the campaign's artifact
  folder held three lines of the gate's output, joined into one line of
  error text, and nothing of the engine: the shell's notice that `run.sh`
  was `Killed`, `Engine did not stop within 30s; sending SIGKILL` and
  `Engine exited before writing all FPM benchmark outputs`. The gate waits
  those 30 seconds for what is left of the engine's process group. The
  record of a Slurm cell was not read.
- SGLang's message was in none of the kept files of that run, for two
  reasons found in the gate as it was then. Its two `awk` filters, which
  copy the engine's output to the log files and to the exec stream, were
  started on the engine's command line and so were descendants of the
  engine's process, and SGLang kills every descendant of that process when
  it kills the process tree and again when the engine shuts down. And
  `awk` is `mawk` in the SGLang image, which reads a pipe in blocks (a
  test in that image showed it holding a line until its input ended), so
  a killed filter took with it every line it still held. So the cell that
  ran kept no engine line, the smoke cells of 2026-11-23 kept none, and
  the formal cells of that day kept their engine log up to a multiple of
  512 KiB, without the launcher's last lines.
- The gate has since been changed. It starts the two filters itself, on two
  named pipes in a temporary folder of its own, and waits up to 10 s for
  them before it ends. The same cell was then run again on 2026-11-24. Its
  `cells/<cell_id>/raw/<pod>/engine.stderr.log` held 202 lines: the
  launcher's lines, SGLang's traceback from two ranks with the message
  above, `Received sigquit from a child process` and SGLang's last line.
  The recorded error and `<pod>.run.stderr.log` held the same lines, about
  57 KB, most of it SGLang's two lines of server arguments of 19 KB each.
  The cell was again recorded as `failed` with `remote_exit=137`.
- `mawk` still reads in blocks, so in an image with `mawk` both copies run
  behind the engine and are complete once the engine has gone. A first
  form of the change gave `mawk` its `-W interactive`, which reads a line
  at a time. In a cell that cut the line of server arguments at 4096 bytes
  and kept nothing after it, so the option was removed. Not run with the
  changed gate: a cell that passes, a vLLM cell, and a Slurm cell.
- On that day the reason was read from SGLang's server alone: start it by
  hand in a pod of the same image and GPU count, `python3 -m
  sglang.launch_server` with the cell's `--model-path` and `--tp-size`, and
  read its stderr. That stays the way for a cell collected before the
  change. The launcher command of the cell's `run.sh` needs the staged
  `runtime/sglbench_*.py` files, which are gone with the cell's pod.
- The cell was not run again and its pod was deleted. The campaign that ran
  held one cell, so this was read and not run: the other cells of a
  campaign still run. Unless the campaign was started with
  `--fpm-publish-partial`, it then publishes no table. In both cases it
  returns a nonzero status, as the smoke campaign that ran did.

Ways around it:

- **Find it first.** A smoke run with `--limit` set to the number of cells
  per phase builds every layout once, in its prefill cell, and never
  publishes. A refused layout shows as a failed cell with
  `remote_exit=137`. With the changed gate its recorded error holds
  SGLang's message (above). The record of a cell
  collected before the change does not name the reason: a pure TP cell that
  fails with `remote_exit=137` on a GPU count that the table marks points
  to it, and any other exception while SGLang builds the engine leaves the
  same record, so confirm it with the server run above. Under the default
  preset that number is 4, 7 or 10 for `--fpm-max-gpus` 2, 4 or 8 when
  planning leaves no cell out for memory; `--plan-only` prints the cells.
- **Plan without the refused cells.** Under the default preset
  `--fpm-moe-tp-sizes` names the GPU counts whose pure TP cell stays in the
  plan. The one-GPU, TEP and DEP cells are not touched. With
  `--fpm-max-gpus 8`, `--fpm-moe-tp-sizes 2` leaves out the refused cells
  of the first two rows of the table and `--fpm-moe-tp-sizes 2 4` those of
  the other two. A value above `--fpm-max-gpus` is refused, and a value that
  is no GPU count of the plan is accepted and keeps no pure TP cell. No
  filter is needed when the maximum is below the first refused count.
  `--fpm-parallel-presets tep dep` leaves out every pure TP cell, also the
  ones SGLang builds. Naming the layout that is served, for example
  `--fpm-parallel-presets tep --fpm-gpu-counts 4`, plans the prefill and the
  decode cell of that layout alone.
- **Publish what passed.** `--fpm-publish-partial` on the first run
  publishes the passed cells at its end, with the refused ones under
  `missing_cells`. After a formal campaign without it in which only these
  cells failed, the same command with `--resume --fpm-publish-partial` runs
  no cell again and publishes the passed cells. The status stays nonzero in
  both cases. The resume needs what the first run used and left: the same
  options, the same source revision (uncommitted changes to tracked files
  count), the checkpoint folder and the artifact folder. Without the
  checkpoint file `--resume` starts a new campaign and runs every cell.

A filter makes another plan with another hash. A failed campaign cannot be
resumed with a filter added, and a new campaign measures every cell again.
A new campaign in the same `--checkpoint-dir` also replaces the checkpoint
file of the failed one, so publish what passed first or give the new
campaign its own folder.

`--fpm-moe-tp-sizes` leaves out the refused pure TP cells and nothing else.
Other cells that cannot run stay in the plan. Planning asks for a memory
estimate under the data label (`--fpm-backend-version`). With the label
0.5.22 or 0.5.23 there is none on `h200_sxm` and `b200_sxm`, the two systems
planned for here, so the plan of a checkpoint that does not fit one GPU
still holds the one-GPU cell; `--fpm-gpu-counts 2 4 8` leaves that out.

Planning leaves the refused cells in on purpose. The rule belongs to the
engine and to its release, and the collector does not turn a rule of AISim's
model layer into a skipped cell (`memory_admission.py`): the engine that
runs decides. AISim's model layer holds the same rule
(`_validate_fp8_block_quantized_moe_config`) and refuses to model such a
layout.

##### DEP cells

A DEP cell has one GPU and one expert-parallel rank per data-parallel rank:
topology `tp=1, dp=D, moe_ep=D` with D of 2 or more. For backend `sglang` it
runs with SGLang's DP attention. One engine starts D scheduler processes, one
per GPU. Each is an attention-DP rank: it schedules its own requests and runs
attention on them, while the expert layers stay spread over all ranks. The
default preset `auto` plans the cell for every GPU count above 1, and
`--fpm-parallel-presets dep` plans it without the pure TP and the TEP cells.
Planning refuses
every other topology with more than one data-parallel rank and names it in
the message; tensor parallelism inside a data-parallel rank is not built. A
dense model has no such cell.

The vLLM render of the cell carries `--tensor-parallel-size 1
--data-parallel-size D --enable-expert-parallel --data-parallel-backend mp`.
SGLang's `tp_size` counts every rank of the engine, so the cell builder
writes, for a tensor-parallel size T and a data-parallel size D:

| vLLM flag | Launcher command |
|---|---|
| `--tensor-parallel-size T` | `--tp-size T*D` |
| `--enable-expert-parallel` | `--ep-size T*D`, right after `--tp-size` |
| `--data-parallel-size D` | `--attn-dp-size D`, and nothing for D of 1 |
| `--data-parallel-backend mp` | Nothing. Any other value refuses the cell. |
| `--max-num-batched-tokens B` | `--chunked-prefill-size B*D --max-prefill-tokens B` |
| `--max-num-seqs N` | `--max-running-requests N*D` |

The DEP cell of two GPUs gets `--tp-size 2 --ep-size 2 --attn-dp-size 2`, and
its prefill command carries `--chunked-prefill-size 16384 --max-prefill-tokens
8192` for the default budget of 8,192. The last two rows give each rank the
budget and the cap a vLLM rank gets; "The token budget and the request cap of
one rank" below has the reason. A cell with one data-parallel rank gets the
command it got before.
`--data-parallel-size` above 1 without `--enable-expert-parallel` refuses the
cell. vLLM's `--data-parallel-size` is never passed on: in SGLang that name is
an alias of `--dp-size`, the number of whole replicas of the model. AISim's
generated SGLang serving deployment asks for the same layout with the pair
`--data-parallel-size D --enable-dp-attention`, which SGLang marks as
deprecated and resolves to the same sizes at both releases.

The launcher admits an attention-DP size above 1 for this layout only: a model
with expert layers, one replica, one scheduler process per attention-DP rank
(`tp_size` equals `attn_dp_size`) and one expert-parallel rank on each
(`ep_size` equals `tp_size`). It also requires SGLang's round-robin load
balancing, and it refuses `enable_dp_lm_head`,
`enable_tokenizer_batch_encode` and the environment variables
`SGLANG_SCHEDULER_SKIP_ALL_GATHER` and
`SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION`. Any other run with DP attention
ends with status 64 before the engine is built.

A point of a DEP cell is a point of one rank. For a point of batch size b the
launcher sends every stage (seed, warm shot, measured shot, decode run,
warm-up and self-test) as one batch call of D*b prompts, ordered so that item
i belongs to rank i mod D. SGLang deals the requests of such a call to the
ranks in turn. The launcher checks the rank that each response names. A
request on another rank fails the run with `dp_rank_placement_mismatch`. From
the scheduler logs it also checks that the kept batches of all ranks ran in
the same steps with the same token counts. Ranks out of step fail the point
with `dp_ranks_out_of_step`. Every other check of a point runs once per rank.
A wrong cached-token count on one rank stages the point once more on all
ranks, and any other failure on one rank fails the point.

The launcher writes one rank file per attention-DP rank and the merged file:
rank 0 at the output path (`benchmark.json`), rank r at `benchmark_dp<r>.json`.
The gate waits for all of them. Each rank's times come from its own scheduler
log, by the rules of a cell with one rank. A row of a DEP cell carries:

- `parallel_strategy: dep`, `tp: 1`, `dp: D` and `moe_ep: D`, from the plan;
- the batch size and the token counts of one rank. The engine ran D times as
  many requests and tokens in that step;
- the time of the slowest rank, as a vLLM DEP row does.

The `producer` block of every rank file holds `attn_dp_size` (1 for a cell
without DP attention). For a DEP cell it also holds `dp_attention`, with
`local_control_broadcast`, `input_blocker` and `placement`, and its `staging`
reads `balanced_batch_call`. No published column changes.

Four things differ from serving or from the vLLM table of the same cell:

- The input blocker. For a DEP cell the launcher sets
  `SGLANG_ENABLE_COLOCATED_BATCH_GEN=1` before it builds the engine. SGLang
  then holds the requests of one batch call back until every rank has them, so
  that all ranks start them in the same step. Serving does not set the
  variable. With it, every scheduler process makes one more poll over all
  ranks in each of its loop iterations for the whole life of the engine
  (`managers/scheduler_input_blocker.py` and `utils/poll_based_barrier.py`
  under `python/sglang/srt/`, each identical at both releases), so the poll is
  inside every measured decode step; a prefill time also spans one receive
  step with the poll, and its effect there was not measured. The cost in a
  decode step cannot be measured inside a benchmark run and is in no result
  file. It was measured apart, outside the launcher, with stock SGLang
  0.5.22 and 0.5.23 on two H200 GPUs for `Qwen/Qwen1.5-MoE-A2.7B`
  (`--tp-size 2 --attn-dp-size 2 --ep-size 2`): the steady decode step of
  free-running requests (48 tokens, context 1,024), with the variable on and
  off and local control broadcast off and on, eight launches in all. A launch
  gives, for each rank, the median over five runs of the median time between
  the decode outputs of steps 16 to 40. At 32 requests per rank the step was
  6.45 to 6.60 ms in all eight launches, with no difference between on and
  off. At 4 requests per rank it was 3.32 to 3.50 ms, with one launch at
  3.72 ms (0.5.22, variable on, broadcast off). A repeat of the pair with
  broadcast off, two launches each per release, gave at 4 requests per rank
  3.32 to 3.37 ms with the variable on and 3.33 to 3.34 ms with it off at
  0.5.22, and 3.45 to 3.49 ms on and 3.45 to 3.47 ms off at 0.5.23. At 32 per
  rank it gave 6.45 ms at 0.5.22 and 6.58 to 6.59 ms at 0.5.23, on and off
  alike. The 3.72 ms did not repeat. An earlier probe of the same variable at
  two ranks on dense models, with 4 requests per rank, found the step 8 to 21
  percent longer with the variable on for `Qwen/Qwen3-0.6B`, whose step is
  2.2 to 2.4 ms without it, and no difference for `Qwen/Qwen3-8B` (4.7 ms,
  0.5.23 only). So a cell with a shorter decode step than the 3.3 to 3.5 ms
  of this model may pay more. The cost was not measured for another MoE
  model, for more than two ranks or inside a launcher run.
- The token budget and the request cap of one rank
  (`sglang_fpm_builder.py`). vLLM gives every data-parallel rank the whole
  `--fpm-max-num-batched-tokens` B and the whole `--fpm-max-num-seqs` N.
  Under DP attention SGLang divides its chunk size and its running-request
  cap by D, so the cell writes `--chunked-prefill-size B*D` and
  `--max-running-requests N*D`, and each rank gets B and N. SGLang does not
  divide `--max-prefill-tokens`, which is written as B. With the default B of
  8,192 the prefill cell of two ranks carries `--chunked-prefill-size 16384
  --max-prefill-tokens 8192`. Both ranks then reported 8,192 for both, and
  the table reached 8,192 new tokens for a rank, summed over the requests of
  its batch. One thing follows: the KV pool of a rank is smaller than with B
  written, and smaller than in the decode cell, which carries no token
  budget. It is the next item. That SGLang
  divides the request cap was read in its source at both releases; no cell
  with `--fpm-max-num-seqs` has run. Before this
  rule the cell wrote B and N as they are. Both ranks then reported a chunk
  size of 4,096, with 8,192 on the command and with SGLang's own default of
  8,192, through the launcher and outside it. Outside the launcher, a call
  of four requests of 1,024 new tokens per rank ran as one batch per rank,
  and with 1,025 each both ranks split alike into 4,096 and 4 tokens.
- The memory share of a DEP prefill cell (`sglang_fpm_builder.py`,
  `runtime/sglbench_launcher.py`). The share is the part of a GPU's memory
  that the weights and the KV pool take together. The cell passes one only
  when the plan gives `--fpm-gpu-memory-utilization`; otherwise SGLang works
  it out. It does so from the chunk size as it is written, before it divides
  it, and keeps 1.5 MiB free for every token of it. With B*D written and the
  default B of 8,192, the share is smaller than with B written, and the KV
  pool of a rank with it. At two ranks SGLang chose 0.805 and a pool of
  529,112 tokens, where the command before this rule, with 8,192 written,
  gave 0.892 and 594,583 tokens. At four ranks it chose 0.574 and a pool of
  389,686 tokens. Part of that fall is SGLang's own reserve for decode
  graphs, which is larger at four ranks.

  Most of what SGLang keeps free was not used. Of the memory left after
  weights and pool, a GPU used 3.8 of 26.7 GB at its sampled peak at two
  ranks and 6.0 of 58.6 GB at four (GB of 2^30 bytes, as SGLang prints them;
  one sample every 0.1 s). The peak was while SGLang
  captured its graphs. While the benchmark ran, the used memory of a GPU
  rose by under 0.1 GB, and every prefill batch above one token ran a
  captured graph. The 1.5 MiB for each token are room for a prefill forward
  without a graph, and no run held one of more than one token. With a share
  given on the command the same grid ran: 0.95 at two ranks, and 0.90 at
  four ranks with a pool of 635,112 tokens and 7.5 GB of each GPU left at
  the peak. At four ranks 0.95 would have left about 0.6 GB and was not run.
  These numbers are of one small model. A large model has larger graphs, and
  none was measured.

  A pool that is too small shows in two ways:

  - The engine does not start. SGLang stops with "Loaded weights leave no
    GPU memory for the KV cache", which names `--mem-fraction-static` and a
    value, or with "Not enough memory. Please try to increase
    --mem-fraction-static." Read in SGLang's source at both releases; no
    cell has failed this way. SGLang raises it in a scheduler process, so
    such a cell ends like a refused pure TP cell, with `remote_exit=137`
    ([Pure TP cells that SGLang refuses to build](#pure-tp-cells-that-sglang-refuses-to-build)).
    With the gate as changed on 2026-11-24 the message should then be among
    the engine's lines in the recorded error and in `engine.stderr.log`,
    as SGLang's refusal was in the cell that ran with that gate. This is
    inferred from that cell and was not run.
  - The cell passes and its table reaches less far. The launcher plans only
    the points the pool holds, so the longest cached prefixes of the largest
    batches are missing, while the number of points can be the same. The
    launcher then prints two lines on stderr, once when it plans the grid
    and once more before its last lines. The first starts with
    `[sglbench] the KV pool of one rank`. It counts pairs, where a pair is
    one number of new tokens at one batch size: how many pairs the grid has
    without the pool limit, how many of them are left out, and how many end
    at a shorter cached prefix. It also gives the pool that would hold the
    whole grid. The second names `--mem-fraction-static`,
    `--fpm-gpu-memory-utilization` and `--max-total-tokens`. The last is
    SGLang's cap on the pool in tokens; a collector cell never carries it.

    The pool can shorten the grid in a second way, through the input limit,
    the longest input SGLang takes for one request. Where the pool is below
    the model's context, SGLang takes that limit from the pool. The limits
    an engine reports do not tell a pool below the context from one equal to
    it. For a pool that is not above the context the first line therefore
    says that a larger pool raises the input limit if the pool is below the
    context. Its counts and its pool figure are then for the grid of the
    present input limit, and the line is also printed, without counts, when
    the pool itself cuts nothing.

    The lines are for the generated prefill grid of any SGLang cell, not
    only of a DEP cell. A decode grid is not looked at: with SGLang's
    default request cap and no cap on the batch size, the pool ends the
    decode grid in an ordinary run, so there is nothing to report. A point
    of `--fpm-benchmark-points-file` that does not fit is refused by name.
    No cell of the runs below was cut: the largest batch of the default grid
    held 32,740 tokens, and a pool of 32,749 tokens holds that grid.

  Where to look. The collector keeps a copy of the cell's stderr as
  `engine.stderr.log` in the cell's `raw/` folder. In every formal cell of
  the runs below, made with the gate as it was before 2026-11-24, that copy
  held the launcher's first lines and ended before its last ones, and the
  smoke cells kept none of it. The gate has since been changed to keep the
  whole log
  ([Pure TP cells that SGLang refuses to build](#pure-tp-cells-that-sglang-refuses-to-build));
  one refused cell has run with it and no cell that passes. The two lines
  are printed when the
  grid is planned as well, early in the log. They have not run in a
  collected cell.

  In either case give the campaign `--fpm-gpu-memory-utilization`, which the
  cell writes as `--mem-fraction-static`. 0.95 ran at two ranks and 0.90 at
  four, and both left memory free for this small model. A larger model needs
  more room for its graphs, and no share was run for one. No SGLang cell has
  been collected with the flag: the memory runs passed
  `--mem-fraction-static` on the launcher's own command, outside the
  collector.
- Local control broadcast. SGLang's
  `--enable-dp-attention-local-control-broadcast` replaces a broadcast over
  all ranks in every scheduler iteration by one inside each rank (the help
  text of the flag, the same at both releases). No collector input switches
  it on: AISim's generated SGLang serving deployment does not pass the flag,
  so a collected DEP cell runs with it off. The launcher takes the flag on its
  own command line and records what SGLang resolved as
  `producer.dp_attention.local_control_broadcast`.

The pod of a DEP cell asks for the GPUs, CPUs and memory of the TEP cell of
the same GPU count. On Slurm a DEP cell of D ranks is refused when
`--fpm-slurm-cpus-per-task` is below D. D is the least the collector accepts,
not what the cell needs: it runs D scheduler processes, SGLang's DP controller
process and the launcher. No CPU count was measured for such a cell.

What has run on GPUs. Every run named here is one of the runs of 2026-11-18:
stock SGLang images, one pod with two H200 GPUs, `Qwen/Qwen1.5-MoE-A2.7B`
with two ranks (`--tp-size 2 --ep-size 2 --attn-dp-size 2`) and SGLang's page
size of 1; the two runs with four ranks are named as such, and so is the
one campaign of 2026-11-23. The runs of 2026-11-18 used the prefill grid as
it was then, without the row of 2 new tokens per request
([SGLang cells](#sglang-cells)), and their point counts are the counts of
that grid. Their files are
not kept in this repository, and neither are those of the earlier probe of
the blocker on dense models.

Through the collector, at SGLang 0.5.22 and 0.5.23: one campaign per release
with `--fpm-parallel-presets dep --smoke --limit 2`, each with one prefill
cell of 6 points and one decode cell of 4 points, dataset text in every
prompt and the launcher's default options. All four cells went through plan,
render, pod, launcher, gate and validation and passed, and no point was
staged again. In each of them:

- the preflight in the pod passed, the launcher admitted the layout and set
  the blocker variable, and the self-test passed with a pair of requests per
  rank;
- the two scheduler processes were the leaders of rank 0 and rank 1, and both
  reported the blocker, a chunk size of 4,096 and a running-request cap of
  4,096;
- every response of every call named the planned rank;
- every batch line of one rank had a line of the other rank with the same
  step number, mode, number of requests and token counts, and no rank logged
  an idle step. The launch times of the two ranks in a step differed by a
  median of about 0.01 ms and by at most 0.51 ms;
- the largest decode points ran 4,096 requests per rank, 8,192 in one call;
- the launcher wrote the two rank files and the merged file, the gate waited
  for the three, and validation accepted the cell with `producer.tp_size`,
  `producer.moe.ep_size` and `producer.attn_dp_size` of 2;
- the two ranks' times of a point differed by at most 0.9 percent, and the
  merged time is the larger one.

The formal cells, at 0.5.23 through the collector: two campaigns with
`--fpm-parallel-presets dep` and the default grid.

The first ran before the cell multiplied the token budget. Its prefill cell
carried `--chunked-prefill-size 8192 --max-prefill-tokens 8192`; both ranks
reported a chunk size of 4,096, and the table ended at 4,096 new tokens per
rank. The prefill cell gave 2,892 points and the decode cell 857, no point
was staged again, and the 3,749 rows were published with the strategy `dep`,
`tp` 1, `dp` 2 and `moe_ep` 2. Every batch line of the prefill cell and of
the decode cell's runs had a line of the other rank with the same step
number, mode, number of requests and token counts: all 8,433 steps of the
prefill cell, and the 857 prefill and 5,142 decode steps of the decode
cell's runs. The seeds of the decode cell
did not run alike on the two ranks, which the procedure allows, since a seed
only has to complete on its rank: 534 seed steps held different numbers of
requests on the two ranks, and in 281 steps one rank ran a seed batch while
the other logged an idle step. None of those steps lay inside a run. The two
ranks' times of a point differed by at most 7.5 percent in the prefill cell
and 3.1 percent in the decode cell.

The second ran with the command the cell writes now, and with the code after
the two later changes named below. Its prefill cell carried
`--chunked-prefill-size 16384 --max-prefill-tokens 8192`; both ranks
reported 8,192 for both, and a KV pool of 529,112 tokens. It gave 3,314
points, 455 of them above 4,096 new tokens per rank: the same points as the
TEP prefill cell of the same model and GPUs, run on 2026-11-17. The decode
cell carries no token budget and gave the same 857 points as in the first
campaign, with a pool of 594,583 tokens. No point was staged again, and the
4,171 rows were published with the same labels. Every batch line of a
measured call had its line on the other rank as above: all 9,654 steps of
the prefill cell, and the 857 prefill and 5,142 decode steps of the decode
cell's runs. The seeds of the decode cell ran as in the first campaign, with
the same counts. The two ranks' times of a point differed by at most 4.7
percent in the prefill cell and 2.7 percent in the decode cell.

A third formal campaign ran on 2026-11-23 with the same command and the grid
that plans 2 new tokens per request. The boot time that its schedulers'
clocks imply is not that of the first two campaigns, so it ran on another
machine. Its prefill cell
gave 3,347 points: the 3,314 points of the second campaign and 33 more, 2 new
tokens per request at batch 1, 2 and 4 with each of the eleven cached lengths.
Its decode cell gave 857 points, four of them one token of context shorter
than before, because the engine reported a KV pool of 594,573 tokens where it
had reported 594,583. No point was staged again, every batch line of a rank
had its line on the other rank in the prefill cell (9,750 steps), and the
4,204 rows were published to a root of their own. SGLang's batch lines show
the 44 batches of one request with two new tokens with a captured graph, and
no batch of more than one token without one; the kept copy of that log ends
about two seconds before the run did.

Two things in the table of that campaign are not explained:

- Its rows of at most about 10 new tokens in a rank's batch are about 1.1 ms
  above those of the second campaign (8.3 against 7.0 ms for one request of
  four new tokens), the 33 new rows among them. For those batches the forward
  call of rank 1 took 6.8 to 7.3 ms where that of rank 0 took 5.8 to 6.1 ms,
  and a step ends when both ranks are done. It began about 130 points before
  the first point of two new tokens, so those points did not start it. The
  first campaign shows the same on rank 0 in 12 of its 99 rows of that size,
  the second in none, this one in 118 of 132. The cause is not known: the
  machine and the launcher and driver, which hold the pool line since the
  second campaign, both differ from that campaign's, and neither is ruled
  out.
- Its first decode point, 4,096 requests per rank at 140 tokens each, holds
  589.5 ms where the second campaign's holds 116.9 ms. That point had a step
  of about 0.6 s among its three kept steps in all three campaigns. In this
  one two of the three were slow, and the median of three is then a slow
  step. Each decode cell has 6 to 10 such steps among its 4,285 decode steps
  after the first; the time lies on the host between one output and the
  launch of the next batch. The cause is not known.

With local control broadcast on, at 0.5.23 only and outside the collector:
the launcher commands of the two cells with the flag added, started in a pod
with the environment of the cell's run script and a smoke grid. Both ended
with status 0, the prefill run with 6 points and the decode run with 4. They
showed the same placement and the same steps on both ranks, with no idle
step, and their result files record `local_control_broadcast: true`. Those
files were not put through validation.

The memory of the prefill cell, at 0.5.23 and outside the collector: the
launcher command of the prefill cell with the default grid, run in a pod
without the collector. `nvidia-smi` sampled the used memory of each GPU
every 100 ms, and a copy of the recorder, made for these runs only, also
noted PyTorch's peak memory around the forwards. Four runs: two ranks with
SGLang's own share and with `--mem-fraction-static 0.95`, and four ranks
(one pod with four H200 GPUs, `--tp-size 4 --ep-size 4 --attn-dp-size 4
--chunked-prefill-size 32768 --max-prefill-tokens 8192`) with SGLang's own
share and with 0.90. All four ended with status 0 and the same 3,314 points,
no point was staged again, and the runs with four ranks wrote four rank
files. The numbers are in the item on the memory share above. In every run
the peak was during graph capture, the used memory of a GPU rose by under
100 MiB while the benchmark ran, and SGLang's own batch lines say that every
prefill batch of 2 to 8,192 new tokens ran a captured graph and every batch
of one new token ran without one. The files were not put through validation,
and the times are not formal: the recorder carried the probe, and two of
the runs shared a host. These are the only runs with more than two ranks.

The launcher runs above, except the second formal campaign and the four
memory runs, used the code before two later changes: what the driver does
when SGLang runs a decode
step of one rank as a prefill batch, and the launcher's refusal of
`enable_tokenizer_batch_encode` under DP attention. After them the smoke
campaign at 0.5.23 ran once more and passed with the same observations, 6
prefill points and 4 decode points, and then the second formal campaign ran.
No decode step ran as a prefill batch in either, so that part of the driver
has run in the unit tests only.

Outside the launcher: stock SGLang of both releases with the same layout, the
blocker variable set, SGLang's public batch call and random token ids. In 24
of 24 batch calls (b of 1, 4 and 32, streamed and not), with local control
broadcast off and on, and then at 60 of 60 points with it off, item i ran on
rank i mod 2 and each rank's log held one batch with exactly its b requests
in the same step. A seed sent that way sat on the rank of its slot: the
measured prompt found its cached tokens there, and none when it was sent to
the other rank. A seven-token run gave one prefill batch and six decode
batches of b requests in the same seven steps on both ranks, and the kept
steps 3 to 5 differed between the ranks by at most 0.64 ms (by more than
0.1 ms in 14 of 72 steps); the dropped steps differed more, step 1 by up to
108 ms and step 2 by up to 3.5 ms. A batch call of one prompt went to the
next rank in turn while the other rank logged idle steps, which is why every
call of the launcher carries the same number of prompts for every rank. Only
these runs stand behind the blocker's cost, the chunk check, the cache check
on the other rank, the call of one prompt and local control broadcast at
0.5.22.

Not run on an engine:

- the row of 2 new tokens per request, which the grid plans since
  2026-11-23, with the default grid in any cell but the DEP and the pure TP
  prefill cell of two GPUs at 0.5.23, and with any grid in a cell of a dense
  model or a one-GPU cell (smoke grids held it in the DEP, pure TP and TEP
  cells at 0.5.23 and in the pure TP cell at 0.5.22);
- a DEP cell of more than four ranks, four ranks in a decode cell or
  through the collector, or a model other than `Qwen/Qwen1.5-MoE-A2.7B`;
- a prefill grid that the KV pool cut, and with it the launcher's two lines
  about it, which have run in the unit tests only; an engine that SGLang
  refuses to start for its memory share; a prefill forward of more than one
  token without a captured graph; an SGLang cell collected with
  `--fpm-gpu-memory-utilization`;
- any failure path under DP attention: a request on another rank than
  planned, ranks out of step, a wrong cached-token count on one rank and the
  second staging on all ranks, an idle step inside a run, and SGLang's change
  of a decode batch into a prefill batch while another rank prefills;
- a page size other than 1;
- `--fpm-max-num-seqs` or `--fpm-max-model-len` on a DEP cell;
- local control broadcast through the launcher at 0.5.22, and in a collected
  cell at either release;
- a DEP cell on Slurm.

The unit tests cover the plan, the command, the gate with two rank files, and
the validation and publication of two rank files whose times were typed in.

### Prefill batch sizes

A prefill cell of either backend collects few batch sizes
([How individual points are selected](../../../docs/perf-model/fpm-self-service/implementation.md#how-individual-points-are-selected)).
For each new-token count the self-benchmark takes the powers of two up to the
largest batch that fits, adds that largest batch, and keeps the first N, three
by default. The largest collected prefill batch is then 4. Dynamo's vLLM
self-benchmark and the SGLang launcher share this rule, the flag
`--prefix-max-batch-size-samples` and its default.

You need this section when a simulation fails with
`FPM prefill query batch_size=N is outside the collected domain [1, 4]`.
With the `sol` method, the default for a registered architecture, FPM
interpolation answers a prefill batch above the collected range only when the
table passes the batch-flatness check (`prefill_batch_axis_is_flat` in
[`fpm_forward.rs`](../../../crates/core/src/perfmodel/perf_database/fpm_forward.rs)):
at equal token totals, the median difference between consecutive collected
batches must be at most 5 percent. With the `direct` method it answers
collected batch sizes only. For a table that misses the check, collect the
batches the simulation needs.

That number of kept batch sizes is the cap. The collector has no option for it.
Set it in `K8sConfig.extra_env` through `--generator-config`; the plan hash
covers that file:

```yaml
K8sConfig:
  extra_env:
    - name: DYN_PREFIX_MAX_BATCH_SIZE_SAMPLES
      value: "5"
```

The entry is written into the `run.sh` of every cell, and the benchmark of a
formal prefill cell takes it as the cap. A campaign with
`--fpm-benchmark-points-file` ignores it: its points come from the manifest.
The largest collected batch is `2^(cap - 1)`, up to the engine's sequence
limit:

| Cap | Largest collected batch | Reached for requests of at most |
|---|---|---|
| 3 (default) | 4 | 2,048 new tokens |
| 4 | 8 | 1,024 new tokens |
| 5 | 16 | 512 new tokens |
| 6 | 32 | 256 new tokens |

The last column holds for the default prefill token limit of 8,192
(`--fpm-max-prefill-isl`), which a batch must fit as a whole. For a longer
request the cell collects the largest batch that fits. For the SGLang cells
run so far, a cap of 5 collected about one and a half times the points of the
default.

What to know before using it:

- The entry changes the plan hash, so it starts a new campaign with its own
  artifact directory, and the decode cells are collected again. `--resume`
  against a checkpoint of the earlier plan is refused.
- The cell ids do not change. Publish with a `--fpm-database-root` that does
  not hold those cells yet. In a root that holds them the first publisher
  wins, as described above: the rows of the new run are skipped, and the
  `database` block of the checkpoint still says `passed`. Check
  `published_cells` and `skipped_first_publisher_wins` in that block, not only
  its `status`.
- Above the new largest batch nothing changes. A table that missed the
  batch-flatness check will very likely miss it again: almost all of the
  check's comparisons are between batches 1 and 2, which a larger cap does not
  change. Choose a cap that covers every batch that is needed.
- Inside the range, the `sol` method estimates a batch that was collected at
  few new-token counts, or not at all, from the neighbouring batches, with no
  flatness check. It refuses the few queries that have no collected point near
  enough. The `direct` method refuses a batch that was not collected.
- A `--smoke` run does not exercise the cap: the collector passes
  `--prefix-max-batch-size-samples 1` to a smoke prefill cell, and the flag
  wins over the variable.
- The cap is recorded in the benchmark result file, as
  `config.prefix_max_batch_size_samples`, and in the `run.sh` of each cell.
  The checkpoint and the published rows do not carry it.
- `--fpm-max-prefill-batch-size` is a different setting. It narrows the
  sequence limit of the prefill cell's engine, and no batch above that limit
  can be collected.
- A bad value (empty, not an integer, or below 1) is refused only when the
  SGLang launcher starts, after the Pod has been scheduled.

This route has been run once: one SGLang campaign on Kubernetes at a cap of 5,
with one model on one GPU. No other cap has been run; the other rows of the
table come from the launcher's grid code, and above 5 the cell may seed more
prefix tokens than the KV cache holds. Dynamo's vLLM self-benchmark declares
the same variable for its flag, so by the code the entry applies to a formal
vLLM prefill cell as well, but no vLLM cell and no Slurm campaign has been run
with it.

### Two campaigns in one namespace

The id of a cell, and with it the name of its Pod or workload, is derived from
the backend, the model, the system, the phase, the topology, the dtypes and
the backend identity. The backend version and the plan hash are not part of
it. Before a cell starts, the collector deletes any workload of that name in
the namespace, so that what an interrupted run left behind cannot write into
the new attempt.

Two campaigns that share a cell, for example one model collected at two
backend versions, must therefore not run in one namespace at the same time:
the second deletes the first one's Pod in the middle of its run. Run them one
after the other, in separate namespaces, or with different GPU counts. This
holds for both backends.

### A shared model-cache volume

Generator's `run.sh` sets `HF_HOME` to the model's path inside the model-cache
volume, unless `K8sConfig.k8s_hf_home` names another path. Where it can write,
it creates `flashinfer-cubins` under `HF_HOME` and `fpm_datasets` beside it, so
that vLLM pods reuse downloaded kernels and the KV warm-up dataset. On a
read-only volume it uses scratch directories in the pod.

On a writable volume that other users share, set `K8sConfig.k8s_hf_home` to a
path outside the volume, for example `/tmp/hf-home`. An SGLang cell loads the
model from its path inside the volume. A cell that sends dataset text reads
the dataset folder, which with this setting is an empty folder of the pod, so
name the folder or the file as [MoE cells](#moe-cells) describes. A cell that
sends seeded token ids does not read the dataset folder, so
it needs neither directory there. SGLang cells on H200 have been collected
with this setting.

## Failure philosophy: observe, don't predict

There is no declarative expected-failure layer. A case that cannot run on the
current framework version fails fast and is recorded in `errors_<module>.json`
and `collection_summary_<backend>.json` with a classification and a
(model, dtype) group label. The summary aggregates failures by group: a whole
group failing is a fix-me signal (collector bug, unverified combo, framework
gap), not something to tolerate or auto-skip. Missing points are tolerated
downstream by interpolation and fallback. Escalation rules — when a failure
deserves a `skips.py` entry or a registry `unverified` marker — live in
`.claude/rules/collector/failure_handling.md`.

`--gpu <type>` resolves the SM version from
`src/aisim_core/systems/<gpu>.yaml`; use `--sm <version>` on an
unregistered GPU. Without either, the local device capability is detected
automatically.

To add a new architecture, create one `cases/models/<architecture>_cases.yaml`
file. To add a new model in an existing architecture, add the model path to
that architecture's `model_paths` list. Add shared op sweeps to the matching
`cases/base_ops/<op>.yaml` file. Add a new op collector only when the existing
ops cannot generate the needed data points.

# Version Management

## Overview

Each backend (trtllm, vllm, sglang) has a **registry** (`registry.py`) that maps ops to collector modules, and a **version resolver** (`version_resolver.py`) that picks the right module at runtime. Individual collector files declare their compatibility via `__compat__`. The current collector framework versions and runtime images are declared in `framework_manifest.yaml`.

```text
framework_manifest.yaml — current collector framework versions and images (schema v2)
framework_manifest.py   — manifest loader/validator/resolver
op_catalog.py         — table→family identity map (op_backend_catalog.yaml)
model_cases.py       — model-centric model/SM case-plan resolver
registry.py          — declares which module handles which version range
version_resolver.py  — routes runtime version → module (packaging.version)
collect.py/collect_ops — validates __compat__ and fails incompatible ops
__compat__           — per-file metadata declaring supported framework versions
cases/               — model-centric case manifests and SM exceptions
wideep/              — WideEP collector namespace for special images/runtimes
wideep/*/registry.py — WideEP-only ops appended when the v2 plan requests them
network/             — collective communication collectors and Slurm network jobs
```

`framework_manifest.yaml` is `schema_version: 2`: each framework entry has a
`default` runtime (`version` + `images`) plus an optional `families:` map of
per-family overrides. Resolution for a given `(framework, op)` walks
`table → catalog family → families[family] or default` — exactly one runtime
per op, always. `families:` overrides require the op catalog
(`op_backend_catalog.yaml`); without it, family-scoped pins fail closed rather
than silently falling back to `default`.

WideEP runtimes (`wideep_sglang`, `wideep_trtllm`) are flattened into peer
framework entries rather than nested under a `wideep:` key, so the same
per-op resolution logic applies uniformly. Each WideEP entry adds
`base_framework` (the stock framework it forks from), `data_backend` (which
table namespace its output is written under), and `collector_dir`.
`collect.py` requires the exact public version for a run and rejects
mixed-pin selections across ops.

Image digests are policy, not decoration: the rule is syntactic — any image
reference containing `/` must be pinned with `@sha256:<64 hex>`; only bare
internal image names (no `/`, e.g. `deepseek-v4-blackwell`) are exempt. In
practice that means every public-registry reference is digest-pinned.
When bumping a version pin, re-fetch and update the digest for every image
variant touched — a stale digest silently keeps pulling the old image. Always
pin the manifest-list (index) digest, never a platform-child digest —
`docker buildx imagetools inspect <image>` prints the index digest; a child
digest pins one architecture and breaks the other platforms.

Use `collector.framework_manifest.resolve_op_runtime(framework, op)` to
resolve a single op's runtime, and
`collector.framework_manifest.validate_resolution()` to check the whole
manifest against every backend registry at once — it returns a list of error
strings (empty = valid) and is the fail-closed CI gate asserting every
registered op resolves to exactly one pinned runtime.

## File Naming Convention

- **No version fork**: `collect_{op}.py` (no suffix)
- **Version forks**: `collect_{op}_v1.py`, `collect_{op}_v2.py`, ... (sequential)
- The number is just an ordering key; the actual version range lives in `__compat__`

## `__compat__` Format

Every collector file must declare `__compat__` after the license header:

```python
# SPDX-FileCopyrightText: ...
# SPDX-License-Identifier: Apache-2.0

__compat__ = "trtllm>=1.1.0"
```

The format is `<framework><constraints>` using PEP 440-style operators:
- `"trtllm>=1.1.0"` — open-ended (latest version)
- `"trtllm>=0.21.0,<1.1.0"` — bounded range
- `"sglang>=0.5.5"` — sglang example

Pre-release ordering is respected: `1.1.0rc2 < 1.1.0 < 1.1.0.post1`.

## Registry Format

Each entry in `registry.py` is an `OpEntry` dataclass (defined in `collector/registry_types.py`).
Exactly one of `module` (unversioned) or `versions` (versioned) must be provided — this is
validated at construction time.

```python
from collector.registry_types import OpEntry, VersionRoute

# Unversioned (no fork):
OpEntry(op="gemm", module="collector.trtllm.collect_gemm", get_func="...", run_func="...")

# Versioned (has forks) — VersionRoutes in descending min_version order:
OpEntry(op="myop", get_func="...", run_func="...", versions=(
    VersionRoute("X.Y.Z", "collector.<backend>.collect_myop_v2"),
    VersionRoute("0.0.0", "collector.<backend>.collect_myop_v1"),
))
```

The resolver picks the first `VersionRoute` where `min_version <= runtime_version`.

## Adding a New Op

1. Create `collector/<backend>/collect_myop.py`
2. Add `__compat__ = "<backend>>=X.Y.Z"` after the license header
3. Export `get_myop_test_cases()` and `run_myop()` (or `run_myop_torch()`)
4. Add an entry to `collector/<backend>/registry.py`:
   ```python
   OpEntry(op="myop", module="collector.<backend>.collect_myop", get_func="get_myop_test_cases", run_func="run_myop")
   ```
5. Run `pytest tests/unit/collector/ -m unit` to verify registry integrity

## Handling a Framework API Change

When upstream framework `X.Y.Z` changes an API that a collector depends on:

1. Rename the original file to `collect_{op}_v1.py` (if not already versioned)
2. Create `collect_{op}_v2.py` with the new API calls
3. Add `__compat__ = "<backend>>=X.Y.Z"` to the new file
4. Add `__compat__` upper bound to the old file if it doesn't have one (e.g. change `">=1.1.0"` to `">=1.1.0,<X.Y.Z"`)
5. Convert the registry entry from unversioned to versioned (or prepend a new `VersionRoute`):
   ```python
   # Before (unversioned):
   OpEntry(op="gemm", module="collector.trtllm.collect_gemm", ...)

   # After (versioned — all forks carry explicit _vN suffix):
   OpEntry(op="gemm", versions=(
       VersionRoute("X.Y.Z", "collector.<backend>.collect_gemm_v2"),
       VersionRoute("0.0.0", "collector.<backend>.collect_gemm_v1"),
   ), ...)
   ```
6. Run tests to validate

## Runtime Behavior

At collection time:
1. `collect.py` reads the backend registry
2. `build_collections()` resolves each op to the correct module for the detected framework version
3. If the resolved module declares `__compat__`, it is validated against the runtime version — mismatches fail that op explicitly and are recorded in the error summary
4. Unsupported versions (no matching entry) are skipped with a warning

## Tests

```bash
pytest tests/unit/collector/ -m unit
```

Covers: version parsing (PEP 440 with rc/post), `__compat__` constraint evaluation, module routing, and structural validation of all three registries.

# Collect gemm/attention/moe data/etc.

## Smoke Test

Use `--smoke` to quickly verify the collector runs end-to-end. It randomly samples 4 test cases per op instead of running the full suite.

```bash
# Smoke test for sglang
python3 collect.py --backend sglang --smoke

# Smoke a specific op
python3 collect.py --backend trtllm --ops moe --smoke
```

## Power Monitoring (Optional)

The collector supports GPU power monitoring during kernel execution using NVML. This feature is optional and disabled by default.

### Enable Power Monitoring
```bash
# Basic power monitoring
python3 collect.py --backend trtllm --measure_power

# With custom minimum duration (default: 1.0s)
python3 collect.py --backend trtllm --measure_power --power_test_duration_sec 2.0
```

### Options
- `--measure_power`: Enable NVML-based power monitoring (samples at 100ms intervals)
- `--power_test_duration_sec`: Minimum test duration for accurate power readings (default: 1.0s)

### Output
When power monitoring is enabled, performance CSV files will include additional columns:
- `power`: Average power consumption during kernel execution (Watts)
- `power_limit`: GPU power management limit (Watts)

**Example output:**
```csv
framework,version,device,op_name,kernel_source,gemm_dtype,m,n,k,latency,power,power_limit
TRTLLM,1.2.0,NVIDIA H200 SXM,gemm,torch_flow,bfloat16,1024,4096,4096,0.234,523.4,700.0
```

### Requirements
Power monitoring requires:
- `pynvml` Python package: `pip install pynvml`
- NVML support (NVIDIA drivers)

If unavailable, a warning is logged and execution continues without power data.

### Notes
- Power monitoring adds minimal overhead (<1%)
- Kernel iterations are automatically adjusted to meet minimum duration for accurate measurements
- Backward compatible: without `--measure_power`, CSVs remain unchanged

## CUDA Graph Timing Policy

`benchmark_with_power` times kernels under CUDA-graph replay. A failed capture
raises (`allow_graph_fail=False`, the default) and the case fails classified.

- **No collector that writes a perf table may pass `allow_graph_fail=True`.**
  The eager fallback times the Python launch loop and ships that floor as the
  kernel latency, and no perf table records the helper's `used_cuda_graph`
  flag — `.claude/rules/collector/layer_permissions.md` allows a
  measurement-method degradation only when it is recorded in the output row.
  `tests/unit/collector/test_no_eager_graph_fallback_sites.py` pins this for
  every file under `collector/`. The last fallback sites (trtllm mamba2 and
  moe, wideep moe_compute, network trtllm alltoall) were closed;
  the sglang trtllm_mla prefill (B1) and the trtllm GDN lanes the day before.
- A collector that sweeps several cells inside one task counts failed cells
  and raises a strict-completeness error at the end (GDN, mamba2), so the
  other cells still ship rows and the failure lands in `errors_<module>.json`.
- Multi-rank collectors agree on the capture outcome before any rank replays
  (`collect_trtllm_alltoall._agree_graph_capturable`); a one-sided capture
  failure would otherwise hang the peers' first replay.
- `allow_graph_fail=True` remains available for ad-hoc diagnosis only; the
  result then carries `used_cuda_graph=False` and must not be persisted.

### Usage Example
```python
from helper import benchmark_with_power

def my_kernel():
    # Your kernel code here
    moe.forward(hidden_states, logits)

with benchmark_with_power(
    device=device,
    kernel_func=my_kernel,
    num_warmups=3,
    num_runs=6,
    repeat_n=1,
) as results:
    latency = results["latency_ms"]
    power_stats = results["power_stats"]
```

## For TensorRT-LLM
### Optional( Read if collecting for mxfp4 kernels)
To collect performance data for mxfp4 kernels used in GPTOSS models, depending on the version of TensorRT-LLM, you might need to manually install triton-kernels. If your version is **>= 1.3.0rc2**, nothing needs to be done and you can run the collection process directly. For any version before **1.3.0rc2**. Follow the below instructions to install triton-kernels and expose them to TensorRT-LLM
1. Build and install Triton (tested with the commit below):
```bash
git clone https://github.com/triton-lang/triton.git
cd triton
# Specific commit verified with TensorRT-LLM
git checkout f3067cd3bd0c29065fa4ecdb724b6f29cbabea5f
pip install -r python/requirements.txt # build-time dependencies
pip install wheel build
python3 setup.py bdist_wheel # if this step fails due cuda related error, you might need to set CUDA_HOME following the optional step
pip install ./dist/*.whl
```
2. (Optional) You may need to set CUDA_HOME env variable to successfully build triton
```bash
export CUDA_HOME=/usr/local/cuda
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
python3 setup.py bdist_wheel
```
3. Expose the Triton kernels to TensorRT-LLM The kernels are not packaged in the wheel, so set the environment variable TRITON_ROOT to your Triton clone:
```bash
export TRITON_ROOT=/local/user/triton
# TensorRT-LLM expects the kernels at:
#   $TRITON_ROOT/python/triton_kernels
```
### Run collection for all available operations
```bash
python3 collect.py --backend trtllm
```
For trtllm, the whole collecting process takes about 30 gpu-hours. On 8-gpu, it takes 3-4 hours.
Please note that the whole process will report a lot of missing datapoints with errors. But it's okay. Our system is kindof robust to fair amount of missing data.
Once everything is done, you might see mutliple xxx.txt files under the same folder. Refer to src/aisim_core/systems/ folder to prepare the database including
how many files are needed accordingly.

## Resume Collection (Checkpoint)

Checkpoint files are always written so an interrupted run can be resumed later:

```bash
# Normal run (writes checkpoint to .collector_checkpoint/)
python3 collect.py --backend trtllm

# Resume an interrupted run (skips already-attempted tasks)
python3 collect.py --backend trtllm --resume

# Custom checkpoint directory
python3 collect.py --backend trtllm --resume --checkpoint-dir /path/to/checkpoints
```

A task is marked **done** once it is attempted (success or failure).
Only tasks that never finished are re-queued on `--resume`;
`--resume-retry-failed` additionally re-runs previously failed tasks.
Running without `--resume` always starts fresh (overwrites old checkpoint).

## For SGLang

Use the pinned stock image, `lmsysorg/sglang:v0.5.14` (or its `-cu130`
variant), for normal SGLang collection.

```bash
python3 collect.py --backend sglang
```

This collects the stock SGLang ops, including:

- GEMM operations (FP8, BF16, INT8, INT4)
- MLA (Multi-head Latent Attention) for context and generation
- MLA BMM (Batch Matrix Multiplication) operations
- MoE (Mixture of Experts) operations
- Normal attention operations

Large-EP local expert compute is available as the separately routed `moe_ep`
operator under `wideep/sglang/collect_deepep_moe.py`; run it explicitly in its
pinned SGLang 0.5.10 image. WideEP MLA is not registered because its legacy
wrapper now reaches the stock 0.5.14-only module implementation.

### DeepEP multi-node collector
For models with DeepEP MoE, inter-node communication data requires a separate
multi-node setup. It is collected by the standalone
`wideep/sglang/collect_moe_a2a.py` collector, launched one Slurm job per
world size by `network/slurm/submit_moe_a2a.sh` — see section 4 of
`network/slurm/README.md`. The older manual log-scraping pipeline under
`wideep/sglang/deepep/` is deprecated (its README carries the retirement
story); do not add new data through it.

# Supporting a new large-EP (WideEP) model

Large-EP MoE uses stock `moe_perf` for modeled local expert compute and
`moe_a2a_perf` for measured dispatch/combine communication. To support model X:

1. **Declare the shapes.** Add the model to
   `cases/models/<Architecture>_cases.yaml` (new architecture: one new file;
   new model in an existing architecture: append to `model_paths`) and mark
   its `model_case_values.moe` rows with `wideep: true`. Correlated MoE
   dimensions (hidden/inter size, topk, expert count) stay together in one row;
   never cross another model's values. The `wideep` flag activates standalone
   A2A shape population.

2. **Ensure stock `moe_perf` coverage.** Local compute is modeled as a
   uniformly balanced stream over EP-local experts, using TP=1 stock MoE
   coordinates. Cross-version stock-table reuse is allowed.

3. **Collect `moe_a2a` across nodes.** From `network/slurm/`:
   ```bash
   bash submit_moe_a2a.sh                      # sglang DeepEP HT+LL, one job per world size
   bash submit_trtllm_alltoall.sh              # trtllm NVLink alltoall (single/multi-GPU NVL)
   ```
   The comm tables have no model column — a new model whose
   (hidden_size, topk, num_experts) tuple is already covered needs no new
   runs; a new physical shape lands in the sweep automatically once declared
   in step 1.

4. **Publish with sidecars.** Finalized communication parquet goes into
   `.../<system>/comm/<backend>/<version>/moe_a2a_perf.parquet` together
   with its `collection_meta.yaml` entry — never a parquet without its
   provenance. The per-world `moe_a2a` outputs need the cross-job merge
   procedure in `network/slurm/README.md` section 4.3.

## MoE table units and caveats

`moe_a2a_perf` records latency in microseconds; `load_moe_a2a_data` converts
leaves to milliseconds. Stock `moe_perf` retains its existing timing contract.

Stock `moe_perf` also supports optional Boolean `default_eligible` selection metadata. The [Core API contract](../../../docs/perf-model/configuration.md#choosing-a-forward-pass-api) defines automatic and exact-source selection. Finalization validates non-null Boolean flags and the named source required by `false`, excludes the flag from measurement identity, and preserves existing annotations when merging a legacy collection that omits the column. Collectors do not infer eligibility from kernel labels or row order.

**Legacy-overwrite caveats.** A new-schema row replaces a legacy-adapted
leaf only at the *same* key, and the legacy adapters derive their node/EP
geometry rather than reading it:

- Legacy sglang DeepEP comm tables carry no `ep_size`; the adapter assumes
  `ep_size = node_num * 8` (8-GPU HGX fleets). New `moe_a2a` worlds
  therefore overwrite those leaves only when collected at 8 GPUs/node — a
  4-GPU/node (GB200) world with the same node count keys differently and
  **coexists** with the legacy data instead of replacing it.
- The legacy trtllm alltoall table (GB200 NVL4) carries no `num_nodes`
  column; the adapter derives `node_num = max(1, moe_ep_size // 4)`. New
  rows overwrite only at 4 GPUs/node; a row with an explicit `num_nodes`
  column is honored as written.

**`log_perf` freezes the CSV header from the first row**, so optional
columns are all-or-nothing per file. In particular, power columns exist in a
staging CSV only if the very first logged row had them: flipping
`--measure_power` across a `--resume` of the same staging file appends rows
whose fields no longer match the frozen header — misaligned columns or empty
cells that fail at load time. Collect any one file with a single power
setting end to end. This property is also why the `moe_a2a` collector ships
**no power column at all** (owner ruling during PR review): its low-latency
timing covers one round-trip rather than a per-phase region, and meaningful
per-phase power would need winning-config re-runs — a measurement-method
design for hardware, not a column to fake. The loaders treat absent (or
null) power as 0.0.

# Test
Rebuild and install the new aiconfigurator. Please make sure you have your new system definition file prepared. It's src/aisim_core/systems/xxx.yaml

# Validate the correctness
Today, we have limited method to validate the database. You can try tools/sanity_check to validate the database a little bit. But it highly depends on your understanding
of the GPU system and kernel optimization.

# Known Issues

## NFS File Locking (Worker Deadlock)

**Symptom**: Collection stalls after a few test cases with no error messages.

**Cause**: `fcntl.flock()` doesn't work reliably on NFS. Workers deadlock when writing to shared output files.

**Solution**: Use `/tmp/` for output files, then copy results after collection.

# Support Matrix

Browse the [Legacy AIC Support Matrix](https://ai-dynamo.org/aisim/support-matrix/)
for CLI compatibility coverage, or inspect the
[per-system CSV files](../src/aisim_core/systems/support_matrix/).
For strict-native estimator coverage, use the
[FPE Support Matrix](https://ai-dynamo.org/aisim/fpe-support-matrix/).

## SGLang ordinary MLA context modules

For uniform-projection checkpoints, `run_attention_torch(..., ordinary_mla=True)`
collects the complete context attention module into
`mla_context_module_perf.txt`. It invokes SGLang's real `prepare_qkv_latent`
and creates fresh `AttentionInputs` on every warmup and timed iteration, so
Q/KV down-projection is included. Prefill is timed eagerly. The loaded projection
weights must match the requested single `gemm_type`; mixed BF16/NVFP4 modules
are rejected instead of mislabeled.

The CLI flag `--ordinary-mla` requires `--mode context --attn-type mla`.
Other combinations fail during argument parsing, before collection starts.
If any dispatch fails, the CLI reports its traceback, finishes the remaining
dispatches, and exits with status 1 so partial collection is not reported as success.

Pass `--target-tp-size 4` (or `8`) to derive the per-rank head count from the
model. It must be positive and divide the model's native head count. If
`--num-heads` is also supplied, it must equal native heads divided by TP.
Without an explicit TP size, ordinary MLA derives TP separately for each
per-rank head count. Python callers pass the actual per-rank head count and
`target_tp_size`. Set
`--chunked-prefill-size 16384` on the CLI, or `chunked_prefill_size=16384`
on `run_mla_module()` or `run_mla_module_worker()`, to reproduce a serving
chunk limit of 16,384 tokens. The worker forwards the limit into its subprocess.
Explicit limits must be positive for one-shot module collection.
Omitting the option keeps SGLang's default. The existing
wide-EP and DSA paths keep their separate behavior. The ordinary context table
uses the existing consumer schema; this does not add mixed-precision modeling.

## TRT-LLM MoE autotuning failures

TRT-LLM MoE collection requires either a loaded tactic cache or at least one
successful autotuning attempt before measuring a tuning-enabled case. Ordinary
failures may try smaller eligible token counts; if every attempt fails, the
case raises before timing or publishing rows. CUDA accelerator errors propagate
immediately. The successful collection path and historical profile values are
unchanged by this failure guard.

# Appendix: estimated extra collection time for long-context shapes

These are **provisional planning estimates**, in additional collection GPU-hours
above the pinned #676 baseline. They price the extra model-owned points enabled
by `--include-long-context`; the existing baseline anchors remain enabled when
the flag is off. They are not measured full-sweep totals or confidence bounds.
The table reflects the frozen pilot inventory; the later MiniMax MXFP8 alias
coverage correction adds projection-lane points that remain unpriced here.

## Estimated long-context increment

The estimates use each sampled SKU's own framework measurements and the frozen
added-case inventory. Rounded planning centers across the three frameworks are:

| GPU SKU | vLLM GPUh | SGLang GPUh | TRT-LLM GPUh | Long-context total GPUh |
|---|---:|---:|---:|---:|
| L40S | 2.0 | 2.6 | 3.4 | ~8 |
| H100 SXM | 4.4 | 2.7 | 1.9 | ~9 |
| H200 SXM | 3.9 | 10.2 | 1.7 | ~16 |
| B200 SXM | 2.6 | 18.3 | 5.5 | ~26 |
| B300 SXM | 2.2 | 18.7 | 1.3 | ~22 |
| GB200 | 3.5 | 9.5 | 4.5 | ~18 |
| GB300 | 5.6 | 13.9 | 10.6 | ~30 |
| A100 SXM80 | Unmeasured | Unmeasured | Unmeasured | Unmeasured |

The seven sampled SKUs total **approximately 130 GPU-hours** for the modeled
long-context work. Rows include explicit analogous-operation provisions where
native measurements are missing; 1,628 additional history cells still lack a
same-framework analogue. The table does not establish that every planned shape
fits memory or has a working native kernel.

Preparation scales with cache/input payload and execution with attention work,
including the sliding-window cap. Precision, geometry and dispatch extrapolations
remain assumptions. B300 uses the intended GLM NVFP4 collection recipe and its
own measurements; these timings do not establish canonical BF16 startup or
capacity equivalence. Other SKUs do not borrow B300 rates.

Inclusive point clocks contain input/history preparation, warmup, compilation,
graph capture and forwards. Forward latency is not added again. Existing worker
initialization is paid once when its group is extended; larger buffers and
shape-dependent setup can still add cost. Scheduler wait, GPU allocation launch
and idle reservation are excluded. Operator tests use one physical GPU even when
their geometry represents a TP rank; actual collective ranks are counted for NCCL.

## Broader expansion budget

The independent MoE token, KDA width and NCCL payload changes add a separate
**approximately 55 GPU-hours** of planning proxies across the seven sampled SKUs.
They are outside the history flag and are not part of the 130-hour history total.
Together, the modeled expansion subtotal is approximately **185 GPU-hours**.

A provisional **eight-SKU working budget of 300 GPU-hours** adds selected reserves
of 50 GPU-hours for unmeasured A100 work and 65 GPU-hours for remaining cases,
setup changes and retries. A larger engineering allowance is **750 GPU-hours
total**, not 750 additional hours. These reserves are budget choices, not measured
rates, statistical intervals or guarantees of successful coverage. No RTX PRO
estimate is included in this eight-SKU budget.

The supporting planning calculations, unrounded per-framework values, case
counts, native/proxy distinctions, source hashes and scratch calculations are
retained as local campaign artifacts outside this PR, together with the six
collection-cost audit scripts and their tests. These audit-only scripts are not
shipped under `tools/perf_database/`. The separate strict audit
keeps unqualified full totals unknown; these planning figures do not change its
validation results. Deferred native Kimi coverage is not priced as zero or counted
as equivalent-coverage savings.
