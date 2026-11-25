# FastAFD measured MoE stage profiles

A profile is a parquet table or legacy JSON import document owned by AISimulate
that carries exact measurements of
the complete decode MoE stage, produced by FastAFD's MegaMoE model benchmarks.
AISimulate reads the profile and never imports, installs, or runs FastAFD.
Lookup is exact-match only, and a
missing measurement is an error, not an estimate.

## Native parquet table

The Rust table loader reads the eleven `key` fields below as individual columns:
identity strings as UTF-8 and count columns (`logical_batch_per_source_rank`,
`mtp_nextn`, `microbatches`, `moe_layers`, `routed_topk`) as signed INT64, plus a floating-point
`latency_ms` (Parquet FLOAT or DOUBLE; FLOAT is widened to f64). The Python writer
always emits DOUBLE latency and signed INT64 count columns, including when a
directly constructed measurement has an integer-valued latency.
`source_json` contains the source object; `capture_json` contains
exactly `model_profile`, `measurement`, and `validation` from the JSON entry.
The source must be identical for all rows. These JSON columns preserve raw
capture metadata; selection and latency normalization run in Rust through the
[canonical estimator](../configuration.md#measured-fastafd-stage).
Both Python inspection and native loading reject other count encodings, including
INT32 and UINT32. Counts must fit in an unsigned 32-bit value; only `mtp_nextn`
may be zero.
Topology components in `ep<N>` and `<A>A<F>F` must also be positive u32 values.
Both loaders trim leading and trailing whitespace from key text using Python's
`str.strip()` rule: Unicode White_Space plus ASCII separators U+001C–U+001F.
Supported Parquet compression codecs are UNCOMPRESSED, SNAPPY, and ZSTD.
Python inspection rejects other codecs, including GZIP, before native estimation.
Both entrypoints require a flat schema, including unused columns; nested structs,
lists, and maps are rejected before row inspection.

`FastAFDMoEStageProfile.load(path).write_parquet(output)` converts a validated
legacy document without changing its measured values. Python's canonical model
constructor imports legacy JSON as an in-memory Parquet snapshot before native
estimation; direct Rust callers supply Parquet. Saved resolved configurations
retain the absolute source path and selected precision, but do not pin the
artifact's digest. To detect replacement of that path, save the stage provenance
returned by `estimate_fastafd_stage` alongside the configuration and compare its
`stage.profile_sha256` with the restored estimate's digest. The AFD adapter
includes both `config` and `stage` in its measurement provenance; persist both.
Reconstruction loads the current artifact, while an existing model retains its
original snapshot and digest. The Python profile objects expose raw data for
inspection and conversion; production AFD timing does not use their lookup API.
Their `entries` tuple is read-only so cached exact lookups retain their identity.

## Document

```json
{
  "schema": "aisimulate.fastafd-moe-stage.v1",
  "lookup_policy": "exact-only",
  "source": {
    "repository": "https://github.com/<owner>/FastAFD",
    "commit": "<40-character git commit>",
    "benchmark": "fastafd.megamoe-colocated-model-benchmark.v2",
    "boundary": "<the benchmark's own description of what the measured interval covers>",
    "boundary_components": ["dispatch", "experts", "return", "combine"],
    "timed_span": "all_moe_layers_and_microbatches",
    "microbatch_split": "equal_padded"
  },
  "entries": [
    {
      "key": {
        "model_path": "Qwen/Qwen3-235B-A22B-FP8",
        "system": "gb200",
        "stage": "agg",
        "topology": "ep8",
        "logical_batch_per_source_rank": 48,
        "mtp_nextn": 0,
        "microbatches": 1,
        "moe_layers": 94,
        "routed_topk": 8,
        "moe_precision": "w4a8_mxfp4_mxfp8",
        "moe_backend": "megamoe"
      },
      "model_profile": "qwen3_235b_fp4",
      "latency_ms": 12.37,
      "measurement": {
        "scope": "complete_moe_stage",
        "method": "cuda_event",
        "method_version": "torch 2.10.0+cu130; cuda 13.0",
        "statistic": "p50",
        "sample_count": 40,
        "cv_percent": 0.56,
        "routing": "balanced",
        "source_tree_sha256": "<SHA-256 of the source tree that ran>",
        "raw_artifact": "qwen3_235b_fp4_8ep_t48_n0_balanced_mega.json",
        "raw_sha256": "<SHA-256 of that raw result file>"
      },
      "validation": {"stable": true, "correctness": null}
    }
  ]
}
```

`source` names the code that produced every entry: the repository URL and commit,
the raw result schema that identifies the benchmark procedure, and the benchmark's
measurement boundary. The commit is not required to belong to the upstream FastAFD
repository; a fork or a modified tree is recorded as such, and each entry's
`source_tree_sha256` identifies the exact tree, so uncommitted modifications are
visible as a tree that differs from the commit.

| Source field | Required value | Behavior |
| --- | --- | --- |
| `boundary` | Nonempty description | Preserves the benchmark's own interval description as provenance. |
| `boundary_components` | `["dispatch", "experts", "return", "combine"]` | Explicit ordered interval contract. Includes A-side combine and excludes the router, which attention charges separately. Missing or incompatible metadata is rejected before prediction. |
| `timed_span` | `all_moe_layers_and_microbatches` | Each timed iteration covers every microbatch across every measured MoE layer of one decode step. Missing metadata or a partial span is rejected. Retained in stage provenance. |
| `microbatch_split` | `equal_padded` | Every microbatch on a source rank contains `ceil(logical_batch_per_source_rank / microbatches)` sequences, including padding. Missing metadata or a different split is rejected by inspection and native loading. Retained in stage provenance. |

The description alone cannot establish the interval contract. Existing import
documents without `boundary_components`, `timed_span`, or `microbatch_split` must be annotated from
verified capture evidence before conversion; do not infer coverage from `scope`,
workload counts, or benchmark names.

The `key` identifies the model, hardware, stage (`agg` for a colocated EP group,
`afd` for an attention/FFN split), topology (`ep<N>` or `<A>A<F>F`), per-source-rank
batch, MTP degree, microbatches, measured MoE layer count, top-k, MoE precision, and
MoE backend. `latency_ms` is the p50 over `sample_count` timed iterations of the
stage for all `moe_layers` layers and all `microbatches` of one decode step.
It must not be a per-microbatch or per-layer timer. Native estimation divides
this complete interval by `moe_layers × microbatches` to obtain the per-layer,
per-microbatch F-stage time. `routing` records the expert
routing distribution the benchmark used; `cv_percent` is the dispersion of those
iterations. `stable` is the benchmark's own stability gate and must be true;
`correctness` is the matched-output check against a reference backend, `null` when
the benchmark ran no reference.

The split applies before expanding sequences by `mtp_nextn + 1` tokens.
For batch 129 with two microbatches, both measured microbatches must contain
65 sequences, matching the attention provider's pricing of 130 total rows.
A 65/64 capture is incompatible and must not be labeled `equal_padded`.
Batch 128 uses two microbatches of 64 without padding. The loader accepts
only the declared equal padded split; it never infers a capture's split from
divisibility or silently reprices an incompatible measurement.

| Key field | Required value | Behavior |
| --- | --- | --- |
| `moe_layers` | Positive u32 | Number of MoE block repetitions actually timed, copied from the capture's `workload.layers`. It is not inferred from the checkpoint's number of routed layers. |

| Measurement field | Required value | Behavior |
| --- | --- | --- |
| `sample_count` | Integer from 1 to 18446744073709551615 (u64) | Counts timed iterations independently of the u32 topology keys. JSON and Parquet imports preserve the same range through native estimation and provenance. |

## Consumers

`afd` entries drive analytical AFD prediction through
`engine.afd.fastafd_profile_path`; see
[measured FastAFD MoE stage](../../replay/engine/analytical.md#measured-fastafd-moe-stage).
The consumer resolves `routed_topk` from the model's normalized `topk` metadata;
missing or invalid routing width is infeasible. The AFD adapter resolves the
pipeline layer count from normalized checkpoint metadata (`layers`) and requires
both the profile's `moe_layers` and the attention provider's `num_layers` to equal
it. The profile count and routing width are matched before precision selection.
Other routing widths or layer counts do not make an exact match ambiguous. The
selected key, including `routed_topk` and `moe_layers`, is retained in measurement
provenance.

The current AFD pipeline uses homogeneous layer accounting: DeepSeek-V3 models
61 attention and MoE block repetitions, although its checkpoint has only 58
routed layers after a three-layer dense prefix. A compatible capture must
therefore actually time 61 MoE block repetitions and record `moe_layers: 61`.
A capture of 58 repetitions remains a valid stage-only profile, but cannot be
composed with that 61-layer AFD model. It is rejected rather than relabeled,
rescaled, or treated as including the dense FFNs. Custom attention providers
must use the same checkpoint-derived count; they cannot select a different
pipeline by returning 58. Modeling the heterogeneous dense prefix separately
is outside this homogeneous pipeline's contract.

## Building a profile

`python -m aisimulate.fastafd_profile <results_dir> --repository <url> --output <profile.parquet>`
writes a native parquet profile (`.json` output retains the import document)
from one directory of raw benchmark results (one benchmark, one commit) into a
profile. Results with another routing distribution (`--routing`, default
`balanced`), an unstable timing, or a failed correctness check are skipped and
counted. Repeated measurements of one key are an error; keep one run per directory.
An existing designated output inside the results directory must validate as a
generated profile before it is excluded from input discovery and replaced. A raw
capture cannot be used as the output, and symlink outputs are rejected before
discovery. Other unsupported inputs fail. Malformed routing values fail before
filtering; only nonempty string routing labels can be counted as routing skips.
Raw results must include `measurement_boundary_components` with the exact ordered
components above, in addition to the free-form `measurement_boundary` description.
They must also include `measurement_timed_span: "all_moe_layers_and_microbatches"`;
the builder copies it to `source.timed_span` and rejects missing or incompatible
values. This metadata is a required capture assertion, not proof of measurement
accuracy. Synthetic fixtures establish validation and normalization only. A real
capture with multiple microbatches and its recorded benchmark tree are needed to
verify the timer's actual coverage.
Raw results must also record `measurement_microbatch_split: "equal_padded"`;
the builder preserves it as `source.microbatch_split` and rejects missing or
incompatible values. Verify the capture's actual splitting and padding before
adding this assertion to an older capture.
For older captures, verify the measured interval from the recorded benchmark tree
before annotating this metadata. The builder never assumes a complete boundary.
The raw results, the benchmark revision, and the procedure stay with the profile so
that every entry can be traced to its evidence.
