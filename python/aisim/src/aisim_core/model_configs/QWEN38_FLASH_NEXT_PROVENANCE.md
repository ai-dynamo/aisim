# Qwen3.8-Flash-Next configuration provenance

The Qwen3.8-Flash-Next configuration files in this directory are byte-identical
copies of their upstream `config.json` files at these immutable Hugging Face
revisions:

| Local file | Upstream source |
| --- | --- |
| `Qwen--Qwen3.8-Flash-Next_config.json` | [`Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json) |
| `Qwen--Qwen3.8-Flash-Next-FP8_config.json` | [`Qwen/Qwen3.8-Flash-Next-FP8@236dfdf285828023ca3bcd3f37366c58a3469b13`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/236dfdf285828023ca3bcd3f37366c58a3469b13/config.json) |

Both repositories use `architectures: ["Qwen4ExpForConditionalGeneration"]` and
`model_type: qwen4_exp`. The text decoder lives under `text_config`; the
`vision_config` ViT is not modeled (see "Scope" below).

The FP8 checkpoint is a quantized derivative of the BF16 checkpoint
(`base_model: Qwen/Qwen3.8-Flash-Next`); its configuration adds
`quantization_config` and is otherwise a copy of the BF16 geometry.

## Not registered: NVFP4

`nvidia/Qwen3.8-Flash-Next-NVFP4@fc694b54fb0174e0913e6adf86691ef85a4ead47`
exists upstream as a `MIXED_PRECISION` (modelopt) derivative, but it requires
Blackwell (`sm100`+) hardware. No Blackwell system is qualified for this model,
so the checkpoint is deliberately absent from `DefaultHFModels` and has no
bundled configuration. Registering it would advertise an option AISimulate
cannot currently serve or validate.

## License

Both files are licensed under the upstream **Qwen Community License 1.0**. The
full license text and copyright notice are preserved in the distribution's
`THIRD_PARTY_NOTICES.md`.

## Scope

The text decoder is the modeling target for the Qwen3.8-Flash-Next campaign.
The vision tower keeps the same treatment as Qwen3.5 and Qwen3-VL in this
repository: `vision_config` is parsed into a `VisionEncoderConfig` and expanded
by `build_encoder_ops` so multimodal requests are not silently unpriced. It is
`deepstack_visual_indexes: []`, i.e. exactly one PatchMerger, matching the
Qwen3.5 ViT shape.

## Serving-side facts (pinned runtimes)

Both facts below were read from the pinned runtime source on an H200 node, not
inferred from the configuration. The pinned images are:

| Runtime | Tag | Digest |
| --- | --- | --- |
| vLLM | `v0.31.0` | `sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3` |
| SGLang | `v0.5.21` | `sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469` |

### The n-gram table is offloaded by default; the model prices it in HBM

`vllm.config.engram.EngramConfig.cpu_offload` defaults to `True`, which keeps
the 51.2e9-parameter table in **pinned host memory** and gathers rows through
Unified Virtual Addressing (`models/qwen4_exp/common/ngram_embedding.py`). The
table is `PLEVocabParallelEmbedding` over the **ETP group**;
`embedding_across_dp` defaults to `False`, so the ETP group is TP alone and each
rank holds `num_embeddings / tp` rows. `EngramConfig.dp_shared_memory` defaults
on whenever `cpu_offload` is on and DP > 1.

SGLang has the equivalent switch under a different name: its
`arg_groups/model_overrides/qwen4_exp.py` sets
`ple_offload_embedding = get_platform().is_cuda and dtype == torch.bfloat16`
when the flag is unset, so the BF16 checkpoint offloads and the FP8 one does
not. Its offloaded table is `Qwen4ExpPinnedHostEmbedding`, a
`VocabParallelEmbedding` subclass, and `--ple-offload-dir` selects an
`mmap`-backed backend for unified-memory parts.

AISimulate prices the table as **HBM residency sharded over TP**, i.e. the
`cpu_offload=False` case, because that is the only residency model it
implements for n-gram tables. The DeepSeek-V4.1 FPM identity does the same thing
more strictly: `sdk/fpm_identity.py` raises unless `engram_cpu_offload is False`
is passed explicitly.

A Qwen3.8-Flash-Next collection run must therefore pin the flag off, and the two
CLIs spell it differently (both checked against the pinned images):

| Runtime | Flag |
| --- | --- |
| vLLM | `--engram-config '{"cpu_offload": false}'` (or `--engram-config.cpu_offload false`) |
| SGLang | `--no-ple-offload-embedding` |

vLLM exposes `EngramConfig` only as the grouped `--engram-config` option;
`--engram-cpu-offload` does not parse. SGLang's default is already off for FP8
and on for BF16, so the fluent flag matters most for the BF16 checkpoint.
Leaving the defaults would place 51.2e9 parameters in host memory and break the
memory identity between the model and the served process.

### Sharding is uneven: the manifold and PLE projections are replicated

The n-gram table shards over TP, but the modules that consume it do not:

| Module | vLLM | SGLang |
| --- | --- | --- |
| `ple.kv_proj` (checkpoint `key_proj` + `value_proj`) | `MergedColumnParallelLinear(..., disable_tp=True)` | two `ReplicatedLinear` |
| `ple.conv1d` | `nn.Conv1d` over `hc_hidden_size` | `nn.Conv1d` over `hc_hidden_size` |
| `ple.norm_key` / `norm_query` / `norm_conv` | group norm, weight `[hc_hidden_size]` | `GroupedGemmaRMSNorm` |
| gated residual (3 projections per block) | `ReplicatedLinear`, down+inject merged with `disable_tp=True` | `nn.Linear` (`layers/hyperconnection.py`) |

`hc_hidden_size` is `hc_count * hidden_size` = 10240. The checkpoint tensors
confirm the widths: `input_mix_weight_down` `[320, 10240]`,
`input_mix_weight_up` `[10240, 320]`, `block_inject_weight` `[4, 10240]`,
`key_proj` `[10240, 2560]`, `value_proj` `[2560, 2560]`, `conv1d`
`[10240, 1, 4]`.

Consequently the model keeps those widths whole rather than dividing by TP.
Treating them as sharded under-counted per-rank weights by ~1.2 GB and
under-predicted the per-layer mixer compute by the TP factor.

### The FP8 release quantizes the table with a per-tensor scale

The FP8 repository stores `ple_embedding.ngram_embedding` as `F8_E4M3` with a
single `weight_scale`, while the model otherwise reports `fp8_block` for GEMMs.
The embedding op is priced by bytes and both modes are one byte per element, so
the mismatch has no cost effect; it is recorded so the label is not read as a
claim about the storage layout.
