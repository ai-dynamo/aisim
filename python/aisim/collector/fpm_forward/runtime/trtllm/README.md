# TRT-LLM FPM observer

This opt-in bundle observes the pinned TensorRT-LLM v1.3.0rc29 non-overlap
executor on one node with TP2/DP1. It uses the stock external serving client.
The pinned source is
[`py_executor.py` at v1.3.0rc29](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc29/tensorrt_llm/_torch/pyexecutor/py_executor.py).
The import loader verifies the installed package version and exact executor
bytes before installing the observer.

For the pilot, run separate prefill and decode sessions. Place the bundle
directory first on `PYTHONPATH` for the engine and its MPI workers and set:

| Environment variable | Value |
| --- | --- |
| `AISIMULATE_TRTLLM_FPM_OBSERVER` | `1` |
| `AISIMULATE_RUNTIME_CONTEXT` | Path to the per-session JSON context |
| `AISIMULATE_RUNTIME_INSTRUMENTATION` | Absolute path to the bundle's `manifest.json` |
| `AISIMULATE_RUNTIME_OBSERVATION_DIR` | Fresh shared writable output directory on the node |

The context contains `attempt_id`, `collector_attempt_id`, `configuration`,
`cell_id`, `phase`, `plan_sha256`, `run_id`, `grid_digest` and
`bundle_sha256`. The bundle hash is the SHA-256 of canonical JSON with sorted
keys and compact separators for `{"manifest": manifest, "files": files}`,
where `files` maps each manifest filename to its content SHA-256. Set
`decode_total_kv_read_tokens` for decode. The stock-client pilot may specify
`decode_kv_tolerance` up to 128 tokens to capture an achieved coordinate near
the target. The artifact always records the actual coordinate. An exact
collection plan uses zero tolerance.

The context may also set `power_window_s`, the minimum NVML energy window in
seconds. It is optional and defaults to `2.0`. It must be finite and at least
`2`; otherwise the observer raises when the final executor schedules its first
batch. The leader writes the value to `benchmark.json` as
`config.power_window_s`.

Use the pinned eager TP2 Qwen3-32B-FP8 pilot settings, with block reuse, overlap
scheduling, chunked prefill and speculation disabled. Drive the batch-four
2,048-token prefill backlog or four concurrent decode requests near 8,192
prompt tokens each. Keep the import receipts, both rank records and the
GPU UUIDs, along with the rank artifact and provenance. Each engine process
writes its import receipt as
`runtime-instrumentation-imports-<hostname>-<pid>.json` with schema
`aisim-trtllm-fpm-observer-imports/v1`. The receipt records the hashes of this
loader, `observer.py` and the upstream `py_executor.py`. It is not the
collector's `aisimulate-runtime-instrumentation-imports/v1` binding, and the
vLLM runtime observation importer rejects it. Inspect the rank
records to confirm contiguous iteration IDs, per-rank timings, shape, counter
bounds and the out-of-span diagnostic. The leader is the only process that
publishes the native rank artifact. The integrated launch and publication
qualification are tracked separately in AISIM-2520 PR 3.
