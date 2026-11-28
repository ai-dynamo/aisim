<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang engine probe for the self-benchmark producer core

## What this is

This folder holds the results of a probe of running SGLang engines. By the operator's record the probe was run
before the producer core of the SGLang self-benchmark was written: `sglbench_grid.py`, `sglbench_content.py` and
`sglbench_artifact.py` in `python/aisim/collector/fpm_forward/runtime/`, with their tests in
`python/aisim/tests/unit/collector/`. The build notes of that work call the probe "phase 0", and the two scripts
carry that name.

Comments and tests of the producer core cite this folder, by check, for what running engines showed beside the
reading of the SGLang source. The folder holds the 17 result files those statements rest on, the two probe
scripts (the saved files with a header, the repository layout and lint marks; see below), and `verify.py`, which
checks the cited numbers against the result files.

The driver sends prompts of random token ids. Nothing here measures a served workload.

## What was run

- `phase0_spike.py`, the driver, builds SGLang's in-process engine (`sglang.srt.entrypoints.engine.Engine`)
  from SGLang server arguments. It sends batches of token-id prompts with request ids of its own, temperature
  0 and `ignore_eos`. In the checks kept here a request asks for one or for seven output tokens.
- `phase0_recorder.py`, the recorder, runs in every scheduler process. It registers a subclass of SGLang's
  `Scheduler` through SGLang's `HookRegistry`. Every scheduler header in the results says `"install": "hook"`
  and class `RecorderScheduler`. The subclass calls the stock `run_batch` and `process_batch_result` and
  writes one line per batch: forward mode, batch size, request ids, token counts, the launch time SGLang
  stamps on the batch, and the time taken at the start of `process_batch_result` after waiting for the copy
  event of the result. Both times are on the monotonic clock of the scheduler process.
- The driver reads those lines, works out each check and writes one `results.json` per run.

The scripts edit no SGLang file. The command line was
`python phase0_spike.py --out DIR [driver options] -- [SGLang server arguments]`, and each `results.json` records
both argument lists (`argv`, `sglang_argv`).

### The two scripts as committed

Both scripts are the saved files with three changes: the licence header and a few comment lines, the layout
that `ruff format` gives with the repository configuration, and lint marks (`# noqa`) on seven lines where the
lint fix would have changed a statement. Each mark has a comment with its reason. No statement was changed: the
syntax tree of each committed script equals that of the saved file (`ast.dump`, docstrings included).
`provenance.json` has the sha256 of the saved and of the committed files.

The scripts import SGLang and subclass two of its classes, `Engine` and `Scheduler`. Both scripts were compared
line by line with every Python file under `python/` of SGLang at v0.5.22 and v0.5.23 (4389 and 4810 files; the
checkouts that were used hold no other directory of the SGLang repository). The comparison looks for runs of three or
more identical lines, indentation aside, that hold at least one line of 25 characters or more which is not a
bracket, a decorator, `pass`, `return` or an import. It found one such run, three lines that read a JSON-lines
file (`if not line:`, `continue`, `record = json.loads(line)`), which also stand in two SGLang test utilities.

## On what, and when

Three kinds of fact are kept apart here and in `provenance.json`.

Recorded in the result files:

- The SGLang package version, in every scheduler header: `0.5.22` in 8 runs and `0.5.23` in 9.
- The models, by the value of `--model-path`, a path inside the pod in Hugging Face hub layout:
  Qwen/Qwen3-0.6B at snapshot `c1899de289a04d12100db370d81485cdf75e47ca` and Qwen/Qwen3-8B at snapshot
  `b968826d9c46dd6066d109eabc6255188de91218`.
- The SGLang server arguments: the model path alone in 7 runs (the schedulers then report page size 1), with
  `--page-size 64` in 8 runs, and with `--tp-size 2` in 2 runs.
- What every scheduler reported alike: overlap scheduling on, DP attention off, chunked prefill size 8192, max
  prefill tokens 16384, max running requests 4096, max request input length 40954. The KV pool size differs
  by run and is in the table below.
- The start of each run, as a stamp in `recorder_dir`: `20261114-035605` to `20261114-041305`.

Stated by the operator, from the operator's session record of 14 November 2026, 03:54 UTC. None of this is in
a result file:

- `nvidia-smi` in the pod printed two `NVIDIA H200, 143771 MiB`.
- The images were `lmsysorg/sglang:v0.5.22-cu130-runtime` and `lmsysorg/sglang:v0.5.23-cu130-runtime`.
- The v0.5.23 pod reported `torch 2.14.1+cu130`.

From the file times of the saved copies. They are not in the files and git does not keep them;
`provenance.json` holds them in UTC:

- The 17 result files were saved between 03:57:15 and 04:14:16 UTC on 14 November 2026, each 32 to 87 seconds
  after the stamp of its run. So the stamps read as UTC. The date in the folder name, 13 November, is the date
  of the runs in the operator's time zone (the operator's statement).
- The saved driver was last changed at 04:12:19 UTC and the saved recorder at 01:14:28 UTC.

The result files record a package version, not a commit. The producer core cites the SGLang source at the
commits of the tags v0.5.22 (`baa9befa381511cd4d56f83980951e5a45fffc6d`) and v0.5.23
(`297e54fd105e974cdf401e5b4d20253a22c16f90`). Whether the two images were built from those commits is not
recorded.

## The runs

One row per folder under `results/`. "Checks" lists the `check_N` keys of the file. `--repeats` is the driver
option of that name; "not given" means the driver's default, which is 5 in the saved driver. In the saved
driver, checks 4, 6 and 8 and the decode part of check 9 repeat their runs.

| Run | SGLang | Model | Page size | TP | KV pool tokens | Checks | `--repeats` |
|---|---|---|---:|---:|---:|---|---|
| `v0522-q06-p1` | 0.5.22 | Qwen3-0.6B | 1 | 1 | 1165358 | 1, 2, 3, 4, 5, 6 | not given |
| `v0522-q06-p64` | 0.5.22 | Qwen3-0.6B | 64 | 1 | 1165312 | 1, 2, 3, 4, 5, 6 | not given |
| `v0522-q06-p1-over` | 0.5.22 | Qwen3-0.6B | 1 | 1 | 1165358 | 1, 9 | not given |
| `v0522-q06-p64-over` | 0.5.22 | Qwen3-0.6B | 64 | 1 | 1165312 | 1, 9 | not given |
| `v0522-q06-p64-over2` | 0.5.22 | Qwen3-0.6B | 64 | 1 | 1165312 | 1, 9 | 5 |
| `v0522-q8-p1` | 0.5.22 | Qwen3-8B | 1 | 1 | 803762 | 1, 2, 3, 4, 5, 6, 8 | 7 |
| `v0522-q8-p64` | 0.5.22 | Qwen3-8B | 64 | 1 | 803712 | 1, 2, 3, 4, 5, 8 | 7 |
| `v0522-q8-tp2` | 0.5.22 | Qwen3-8B | 1 | 2 | 1704996 | 1, 2, 3, 4 | 5 |
| `v0523-q06-p1-a` | 0.5.23 | Qwen3-0.6B | 1 | 1 | 1160904 | 1, 2, 3 | not given |
| `v0523-q06-p1-b` | 0.5.23 | Qwen3-0.6B | 1 | 1 | 1160904 | 1, 4, 5, 6 | not given |
| `v0523-q06-p64` | 0.5.23 | Qwen3-0.6B | 64 | 1 | 1160896 | 1, 2, 3, 4, 5, 6 | not given |
| `v0523-q06-p1-over` | 0.5.23 | Qwen3-0.6B | 1 | 1 | 1160904 | 1, 9 | not given |
| `v0523-q06-p64-over` | 0.5.23 | Qwen3-0.6B | 64 | 1 | 1160896 | 1, 9 | not given |
| `v0523-q06-p64-over2` | 0.5.23 | Qwen3-0.6B | 64 | 1 | 1160896 | 1, 9 | 5 |
| `v0523-q8-p1` | 0.5.23 | Qwen3-8B | 1 | 1 | 800297 | 1, 2, 3, 4, 5, 6, 8 | 7 |
| `v0523-q8-p64` | 0.5.23 | Qwen3-8B | 64 | 1 | 800256 | 1, 2, 3, 4, 5, 8 | 7 |
| `v0523-q8-tp2` | 0.5.23 | Qwen3-8B | 1 | 2 | 1698074 | 1, 2, 3, 4 | 5 |

By the operator's record the same session also ran 13 probes of DP attention. The producer core does not cite them and they are not in
this folder.

## What each check shows

What a check found is taken from the result files. What a check did is described from the saved driver: which
prompts it sent, how many output tokens it asked for (one or seven) and which decode steps the sample keeps (3
to 5). A prompt or prefix length that only the saved driver gives is marked "(driver)"; see "Limits" for what
that means.

**check_1** (17 runs). Every scheduler process wrote a recorder header: one process at TP 1 and two at TP 2.

**check_2** (10 runs). One call with 2 prompts and one call with 128 prompts each ran as exactly one extend
batch that held all its requests, with no decode batch after it. Nine runs have the 128 case: both tags, both
models, both page sizes, and TP 2. `v0523-q06-p1-a` ran 32 in its place. The file records the size of the
extend batch, not its token count. Each request has 64 new tokens (driver), so 128 requests are 8192 tokens,
the chunked prefill size.

**check_3** (10 runs: 6 at page size 1, 4 at page size 64). Cached-token counts that the engine reported for two
requests in each case:

- Prefill, aligned prefix. A prefix of 128 tokens at page size 1, or 512 tokens at page size 64 (driver), is
  sent with one output token. The prefix followed by 64 other tokens is then sent twice, with different
  tails. Both calls report the whole prefix as cached: 128 and 512.
- Prefill, unaligned prefix. A prefix seven tokens longer, 135 or 519 tokens (driver), is sent with one output
  token, and the prefix followed by 64 other tokens is then sent once. The engine reports 135 cached at page
  size 1 and 512 at page size 64.
- Decode. Two prompts of 1021 tokens are sent with one output token and report 0 cached. They are sent again
  with seven output tokens and report 1020 cached at page size 1 and 960 at page size 64. The extend batch of
  that second call held 2 tokens at page size 1 and 122 at page size 64, which is 1 and 61 for each request.

**check_4** (10 runs, 20 cases: 12 on Qwen3-8B, of which 4 at TP 2, and 8 on Qwen3-0.6B). Decode runs of 2 and
of 8 requests at context 1024, seven output tokens each, repeated 5 or 7 times. Every repeat was one extend
batch and then six decode batches (`modes_seen`). The file holds no batch sizes for these runs: that every batch
held all requests is what `pass` means in the saved driver. For each case the file holds, per decode
step, the median over the repeats of the interval between two batch outputs. Step 1 is the interval from the
output of the extend batch to the output of the first decode batch. "Sample" is `median_kept_ms`: for each
repeat the middle value of steps 3 to 5, then the median over the repeats. All values are milliseconds.

| Run | Batch | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 | Step 6 | Sample | Sample against step 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v0522-q06-p1` | 2 | 2.160 | 1.407 | 1.335 | 1.522 | 1.554 | 1.555 | 1.522 | -2.06 % |
| `v0522-q06-p1` | 8 | 2.875 | 1.396 | 1.609 | 1.716 | 1.713 | 1.707 | 1.712 | -0.06 % |
| `v0522-q06-p64` | 2 | 2.015 | 1.401 | 1.419 | 1.442 | 1.549 | 1.551 | 1.442 | -6.91 % |
| `v0522-q06-p64` | 8 | 2.587 | 1.467 | 1.544 | 1.691 | 1.690 | 1.683 | 1.689 | -0.06 % |
| `v0522-q8-p1` | 2 | 4.891 | 5.226 | 5.221 | 5.235 | 5.233 | 5.232 | 5.235 | +0.04 % |
| `v0522-q8-p1` | 8 | 5.327 | 5.665 | 5.661 | 5.684 | 5.681 | 5.674 | 5.679 | -0.04 % |
| `v0522-q8-p64` | 2 | 4.930 | 5.252 | 5.252 | 5.257 | 5.255 | 5.248 | 5.257 | +0.04 % |
| `v0522-q8-p64` | 8 | 5.644 | 5.632 | 5.627 | 5.648 | 5.655 | 5.647 | 5.648 | -0.12 % |
| `v0522-q8-tp2` | 2 | 3.245 | 3.550 | 3.549 | 3.558 | 3.559 | 3.557 | 3.556 | -0.08 % |
| `v0522-q8-tp2` | 8 | 3.467 | 3.807 | 3.800 | 3.802 | 3.804 | 3.800 | 3.802 | -0.05 % |
| `v0523-q06-p1-b` | 2 | 2.098 | 1.416 | 1.397 | 1.562 | 1.563 | 1.560 | 1.562 | -0.06 % |
| `v0523-q06-p1-b` | 8 | 2.679 | 1.411 | 1.605 | 1.708 | 1.711 | 1.707 | 1.704 | -0.41 % |
| `v0523-q06-p64` | 2 | 2.207 | 1.593 | 1.611 | 1.569 | 1.575 | 1.186 | 1.575 | 0.00 % |
| `v0523-q06-p64` | 8 | 2.748 | 1.599 | 1.619 | 1.604 | 1.615 | 1.567 | 1.615 | 0.00 % |
| `v0523-q8-p1` | 2 | 4.898 | 5.266 | 5.263 | 5.254 | 5.259 | 5.263 | 5.259 | 0.00 % |
| `v0523-q8-p1` | 8 | 5.274 | 5.667 | 5.661 | 5.693 | 5.686 | 5.686 | 5.686 | 0.00 % |
| `v0523-q8-p64` | 2 | 4.886 | 5.257 | 5.250 | 5.260 | 5.256 | 5.257 | 5.256 | 0.00 % |
| `v0523-q8-p64` | 8 | 5.656 | 5.635 | 5.641 | 5.654 | 5.658 | 5.653 | 5.652 | -0.11 % |
| `v0523-q8-tp2` | 2 | 3.216 | 3.592 | 3.596 | 3.594 | 3.595 | 3.591 | 3.596 | +0.03 % |
| `v0523-q8-tp2` | 8 | 3.411 | 3.790 | 3.802 | 3.806 | 3.793 | 3.803 | 3.802 | +0.24 % |

The file also holds the median time from the launch of each decode batch to its output. For steps 3 to 6 it
was 1.8 to 2.0 times the step-5 interval on Qwen3-8B and 1.3 to 1.8 times on Qwen3-0.6B, so a batch was
launched before the batch ahead of it had delivered its output. Overlap scheduling was on in every run; no
run had it off.

**check_5** (8 runs: 4 at page size 1, 4 at page size 64). Twenty decode points in a row on the same prompts,
with no flush between them: batch 4 at contexts 4096, 2048, 1024, 512, 256, 128, 64 and 32, then batch 2 at
the same contexts, then batch 1 at 4096 to 512. A prompt is the context less three tokens (driver), and a
shorter prompt is a prefix of a longer one. Each point sends its prompts with one output token (the seed) and
again with seven. The file records, for each seed request, the prompt tokens less the cached tokens.

- The first point is cold: 4093 tokens computed for each request. In three of the four runs at page size 1
  one request of the four computed 4092, so one of its tokens was reported as cached. The file does not record
  why.
- Every later seed computed 1 token at page size 1. At page size 64 it computed 61 tokens at contexts 4096 to
  64 and 29 at context 32. At contexts 64 and 32 the prompt is shorter than one page, and all of it was
  computed.
- The rows for batch 2 and batch 1 send again a prompt that a seven-token run at batch 4 had used. At context
  1024 that prompt of 1021 tokens computed 1 token at page size 1 (1020 cached) and 61 at page size 64 (960
  cached). A seed request is the prompt alone, and in every row at least one of its tokens was computed, so
  these rows do not show whether the six generated tokens were stored.
- `run_sequence_ok` is true for all 160 seven-token runs. The file holds no batch modes or sizes for them: in
  the saved driver the field is true when a run is one extend batch and then six decode batches with all
  requests in every batch.

**check_6** (6 runs). Informational. The same kind of sample as in check_4, measured again at batch 8 and context
1024 with streaming off and with streaming on: the ratio of the two medians, on over off, was 0.9626 to 1.0006.

**check_8** (4 runs, Qwen3-8B at TP 1). Informational. The same sample after a cache flush, and again after
966656 tokens of other prompts had been sent into a pool of 800256 to 803762 tokens: the ratio of the two
medians, full over empty, was 0.9996 to 1.0005. The file records no eviction and no admission decision.

**check_9** (6 runs, Qwen3-0.6B, both tags). Informational.

- Prefill, all six runs, page size 1 and 64. One call of 136 requests of 64 new tokens ran as two extend
  batches in consecutive forward passes: 128 requests with 8192 tokens, then 8 requests with 512 tokens. The
  chunked prefill size was 8192.
- Decode, 136 requests, page size 64 (`v0522-q06-p64-over`, `v0523-q06-p64-over`). One decode run: an extend
  batch of 128, an extend batch of 8, then six decode batches of all 136. These files record no context.
- Decode, 384 requests, page size 64 (`v0522-q06-p64-over2`, `v0523-q06-p64-over2`; 384 is the 128 that fit
  plus `--over-budget-extra 256`). Five runs in each file. Every run was three extend batches of 128 and then
  six decode batches of all 384. Of the five intervals between the six decode outputs, the last was 1.895
  to 2.503 ms and the four before it were 17.138 to 20.110 ms, over the ten runs. The context was 256 (driver):
  a prompt of 253 tokens, of which 61 are new at page size 64, one page when rounded up. The files record no
  context and no token count for these batches.
- Decode, page size 1 (`v0522-q06-p1-over`, `v0523-q06-p1-over`). Skipped: it needs 8200 requests, above the
  running limit of 4096.

## How to read `results.json`

- `argv` and `sglang_argv`: the driver's arguments and the SGLang server arguments. `/models/...` and
  `/work/out/...` are paths inside the pod.
- `recorder_dir`: where the recorder wrote its lines; its last part is the start stamp of the run.
- `headers`: one entry per scheduler process, written when the scheduler was built. It holds the SGLang
  version, the class and its first base classes, the TP and DP ranks, and seven scheduler settings:
  `page_size`, `enable_overlap`, `chunked_prefill_size`, `max_prefill_tokens`, `max_running_requests`,
  `max_total_num_tokens` (the KV pool in tokens) and `max_req_input_len`.
- `check_N`: one entry per check that ran, with `name`, `pass` and `seconds`. `pass` is true or false for a
  check with a criterion and null for an informational one.
- `extend_batches` in check_2 holds pairs of forward pass number and batch size. In `check_9.prefill` it holds
  triples of forward pass number, batch size and new tokens of the batch.
- `batches` in `check_9.decode` holds pairs of forward mode and batch size, in order.
- Times ending in `_ms` are milliseconds on the scheduler's monotonic clock. In check_4 they are medians over
  the repeats; in `check_9.decode.runs` they are single runs. `seconds` and `_wall_s` are driver wall times.
- `faults`: recorder errors. It is empty in all 17 files.

## Which code statement rests on which file and field

File names are relative to `python/aisim/collector/fpm_forward/runtime/` for the modules and to
`python/aisim/tests/unit/collector/` for the tests.

- `sglbench_grid.py`, the note "Status of the evidence" above `prefill_unfit_reason`, and the chunk-rule rows
  of `test_fpm_sglbench_grid.py` (128 requests of 64 tokens fit, 136 do not):
  - `check_9.prefill` in the six runs with check_9: `batch`, `new_tokens_each` and `extend_batches`;
  - `check_2.cases[]` with batch 128, field `extend_batches`, in the nine runs that have that case;
  - `check_4.cases[]`, fields `modes_seen` and `pass`, for the decode runs of 2 and 8 requests;
  - `headers[]`: `chunked_prefill_size`, `max_prefill_tokens` and `max_running_requests`.
- `sglbench_grid.py`, the last bullet of the note "Status of the evidence" and the docstring of
  `decode_unfit_reason` (every decode batch holds all requests although the prompts took several extend
  batches), and the 384-request decode row of `test_fpm_sglbench_grid.py`:
  `check_9.decode.batches` in `v0522-q06-p64-over` and `v0523-q06-p64-over`; `check_9.decode.runs[].batches`
  and `decode_batches_hold_all` in `v0522-q06-p64-over2` and `v0523-q06-p64-over2`. The 384-request runs were
  at context 256 (driver); the test row is at context 1024.
- `sglbench_content.py`, the module docstring on the cache rules, and the cache-model test of
  `test_fpm_sglbench_content.py` that names the engine probe,
  `test_cache_model_reproduces_the_engine_probe_counts`: `check_3.cases[].cached_tokens`,
  `check_3.decode_prompt_tokens` and `check_3.decode_extend_tokens` in the ten runs with check_3;
  `check_5.rows[].seed_computed_tokens` in the eight runs with check_5, rows with batch 2 or 1.
- `test_fpm_sglbench_grid.py`, `test_expected_cached_and_extend_tokens_match_the_engine_probe` (960 and 61 at
  page size 64, 1020 and 1 at page size 1, 128 of 128, 512 of 512): the same fields of check_3.
- `sglbench_content.py`, the input limit of 40954 tokens: `headers[].max_req_input_len`, the same in every run.
- `sglbench_artifact.py`, the comment on `DECODE_RAW_STEPS` (seven output tokens give one extend batch and six
  decode batches): `check_4.cases[].modes_seen` in the ten runs with check_4.
- `sglbench_artifact.py`, the decode deviation note: `check_4.cases[].median_inter_output_ms_by_step` and
  `median_kept_ms`, the 20 cases of the table above; `headers[].enable_overlap`. Its figures for 384
  requests: `check_9.decode.runs[].decode_inter_output_ms` in the two `-over2` runs.
- `test_fpm_sglbench_grid.py`, the limits records `H200_PAGE_1` and `H200_PAGE_64`, also used by
  `test_fpm_sglbench_producer.py`: `headers[0]` of `v0522-q06-p1` and of `v0522-q06-p64`, fields `page_size`,
  `max_req_input_len`, `max_total_num_tokens`, `max_running_requests`, `max_prefill_tokens` and
  `chunked_prefill_size`. Both are SGLang 0.5.22 with Qwen3-0.6B at TP 1. The GPU model in the names is the
  operator's statement above.

Not from this folder: the graph-size rows `H200_DECODE_GRAPH_ROWS` and `H200_PREFILL_GRAPH_ROWS` of
`test_fpm_sglbench_grid.py`. No result file records a graph size.

## Limits

- The saved driver is its last revision. By file time, 15 of the 17 result files were written before the
  driver was last changed; only `v0522-q06-p64-over2` and `v0522-q8-tp2` are later. No result file holds a
  hash of the script that wrote it. The two files `v0522-q06-p64-over` and `v0523-q06-p64-over` hold
  `check_9.decode` in a shape that the saved driver no longer writes: one run under `batches` and
  `sequence_ok`, where the saved driver writes a list `runs`. The other two `-over` files hold only a
  `skipped` note there, which the saved driver also writes. So the saved scripts are not shown to be the
  exact ones that produced every file. Each number marked "(driver)" above, and every statement of what a
  check did, rests on the saved driver alone.
- `"sequence_ok": false` in the two 136-request files. In the saved driver the field of that name is true when
  a run is exactly one extend batch and then six decode batches with all requests in every batch. With 136
  requests the prompts took two extend batches, so false is what such a test gives; the six decode batches
  held all 136. The revision that wrote these two files was not kept.
- The GPU model, the images and the torch version are in no result file. They are the operator's statements.
- The raw recorder lines were not kept. Check_4 therefore has medians over the repeats only.
- Runs overlapped in time. `v0522-q8-p1` and `v0523-q8-tp2` have the same start stamp, `20261114-040757`. In
  three more pairs one run started before the result file of the other was saved: `v0523-q8-p64` and
  `v0522-q06-p1`, `v0522-q06-p64-over` and `v0523-q06-p64-over2`, `v0523-q06-p64-over2` and
  `v0522-q06-p1-over`. Each pair is one run of each SGLang version. No result file records which GPU a run
  used, so the files do not show whether a timed run had a GPU to itself. Five of these runs hold step times
  that are cited: the four with check_4 and `v0523-q06-p64-over2`.
- Not covered:
  - the input limit and the running-request limit of admission. No call reached either;
  - the KV pool boundary and the `max_prefill_tokens` boundary of admission. The chunked prefill size (8192)
    was below `max_prefill_tokens` (16384) in every run;
  - a request cut by the chunk budget. No file records one. The first seed call of check_5 sent four prompts
    of 4093 tokens in one call, above the chunked prefill size, but its batches were not recorded;
  - chunked prefill off, and overlap scheduling off;
  - page size 16. Only page sizes 1 and 64 were run;
  - which radix-tree core was active;
  - captured graph sizes. The recorder header has no such field;
  - what a finished seven-token run stores beyond its prompt.

## How to run `verify.py`

```sh
python benchmarks/evidence/collector/sglbench-engine-probe-20261113/verify.py
```

It needs only the Python standard library and reads the files next to it. It prints one line for each of its
16 groups of assertions and a last line with the totals, and exits with status 1 if any assertion fails. It
checks the limits records, the batch splits of check_2 and check_9, the cached counts of check_3 and check_5,
the step figures of the decode deviation note, the run conditions and ranges given in this file, the constants
of `phase0_spike.py` behind the numbers that this file marks "(driver)", and the sha256 of every file that `provenance.json` lists. After a
change to any file of this folder, its sha256 in `provenance.json` must be brought up to date, or the first
group fails.
