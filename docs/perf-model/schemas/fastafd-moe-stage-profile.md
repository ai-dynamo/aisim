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
`latency_ms` (Parquet FLOAT or DOUBLE; FLOAT is widened to f64).
`source_json` contains the source object; `capture_json` contains
exactly `model_profile`, `measurement`, and `validation` from the JSON entry.
The source must be identical for all rows. These JSON columns preserve raw
capture metadata; selection and latency normalization run in Rust through the
[canonical estimator](../configuration.md#measured-fastafd-stage).
Both Python inspection and native loading reject other count encodings, including
INT32 and UINT32. Counts must fit in an unsigned 32-bit value; only `mtp_nextn`
may be zero.

`FastAFDMoEStageProfile.load(path).write_parquet(output)` converts a validated
legacy document without changing its measured values. Python's canonical model
constructor imports legacy JSON as an in-memory Parquet snapshot before native
estimation; direct Rust callers supply Parquet. Provenance and saved
configurations retain the original source path and SHA-256 digest. The Python
profile objects expose raw data for
inspection and conversion; production AFD timing does not use their lookup API.

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
    "boundary_components": ["dispatch", "experts", "return", "combine"]
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

The description alone cannot establish the interval contract. Existing import
documents without `boundary_components` must be annotated from verified capture
evidence before conversion; do not infer coverage from `scope` or benchmark names.

The `key` identifies the model, hardware, stage (`agg` for a colocated EP group,
`afd` for an attention/FFN split), topology (`ep<N>` or `<A>A<F>F`), per-source-rank
batch, MTP degree, microbatches, measured MoE layer count, top-k, MoE precision, and
MoE backend. `latency_ms` is the p50 over `sample_count` timed iterations of the
stage for all `moe_layers` layers of one decode step. `routing` records the expert
routing distribution the benchmark used; `cv_percent` is the dispersion of those
iterations. `stable` is the benchmark's own stability gate and must be true;
`correctness` is the matched-output check against a reference backend, `null` when
the benchmark ran no reference.

| Measurement field | Required value | Behavior |
| --- | --- | --- |
| `sample_count` | Integer from 1 to 18446744073709551615 (u64) | Counts timed iterations independently of the u32 topology keys. JSON and Parquet imports preserve the same range through native estimation and provenance. |

## Consumers

`afd` entries drive analytical AFD prediction through
`engine.afd.fastafd_profile_path`; see
[measured FastAFD MoE stage](../../replay/engine/analytical.md#measured-fastafd-moe-stage).
The consumer resolves `routed_topk` from the model's normalized `topk` metadata;
missing or invalid routing width is infeasible. Both this width and the attention
provider's layer count must match the profile key before precision selection.
Other routing widths or layer counts do not make an exact match ambiguous. The
selected key, including `routed_topk` and `moe_layers`, is retained in measurement
provenance.

## Building a profile

`python -m aisimulate.fastafd_profile <results_dir> --repository <url> --output <profile.parquet>`
writes a native parquet profile (`.json` output retains the import document)
from one directory of raw benchmark results (one benchmark, one commit) into a
profile. Results with another routing distribution (`--routing`, default
`balanced`), an unstable timing, or a failed correctness check are skipped and
counted. Repeated measurements of one key are an error; keep one run per directory.
The designated output is excluded from input discovery, so the same command can
replace a JSON profile inside the results directory. Other unsupported inputs fail.
Raw results must include `measurement_boundary_components` with the exact ordered
components above, in addition to the free-form `measurement_boundary` description.
For older captures, verify the measured interval from the recorded benchmark tree
before annotating this metadata. The builder never assumes a complete boundary.
The raw results, the benchmark revision, and the procedure stay with the profile so
that every entry can be traced to its evidence.
