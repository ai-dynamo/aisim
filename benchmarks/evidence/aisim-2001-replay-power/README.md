<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISIM-2001 Replay host cost

November 18, 2026. The baseline is upstream `78370acd7c3f6b9547eea59b36dfabe0b64cf9eb`. The candidate is the AISIM-2001 worktree after folding active rank evidence at pricing time. Both libraries used the workspace Cargo release profile with `embed-python`, Rust 1.99.1, and Python 3.12.15 on the same Intel Core Ultra 9 285H WSL host. The runner was pinned to logical CPU 7.

The timer surrounds the complete native `run_replay_json` call, including model construction and final power aggregation. The serialized request is identical within each pair. Two warmups precede 20 alternating baseline/candidate pairs per case, with detailed capture disabled during timing. The earlier implementation used 16 pairs under the same protocol.

The powered cases use synthetic schema-v8 tables, 1,024 requests, 512 input and 32 output tokens, and a seeded Poisson load of 200 requests/s. Both exercised 1,084 mixed rank calls, verified in a separate diagnostic run. SOL uses Qwen3-0.6B and direct uses a verified profile for an unregistered model. The v6 SOL control uses 128 requests with the same token lengths and load. All pairs matched completions, simulated duration, mean TTFT, and mean TPOT. Candidate v8 power coverage was 100% in both cases.

| Case | Baseline median | Candidate median | Median paired change | Paired middle 50% | Before evidence folding |
| --- | ---: | ---: | ---: | ---: | ---: |
| v8 SOL, DP=2 | 34.67 ms | 36.08 ms | +4.6% | +1.4% to +7.3% | +9.8% |
| v8 direct, DP=2 | 33.10 ms | 34.49 ms | +5.4% | +0.2% to +9.6% | +5.9% |
| v6 SOL, DP=1 | 25.94 ms | 25.69 ms | -3.4% | -8.0% to +2.8% | -0.2% |

Folding the per-rank samples before caching limits a powered iteration to two Replay evidence components and one accumulator lock per execution. The residual median host cost was about 1.4 ms per 1,024-request powered run in this benchmark. These synthetic measurements characterize host overhead. Mixed-step watts still need measured qualification.

Release binary SHA-256 values

- Baseline `7c6f924321206b16b68aeb91c18fe8a4741ce62bf9b796a33ce26142f168ae0d`
- Candidate before folding `ae78e4deaf165a6839dade19a0c2fc0b809153709d6ac35f0e230dee72347aeb`
- Candidate after folding `c6205e7c6a34dd908c856c791f2681cee302117f84eeb0564a5975f0e5ec2465`

<details>
<summary>Final paired samples in ms (baseline / candidate)</summary>

| Pair | v8 SOL, DP=2 | v8 direct, DP=2 | v6 SOL, DP=1 |
| ---: | ---: | ---: | ---: |
| 1 | 34.18 / 36.60 | 31.83 / 34.54 | 25.63 / 27.25 |
| 2 | 31.82 / 65.99 | 32.21 / 35.76 | 25.52 / 24.30 |
| 3 | 33.06 / 36.80 | 33.83 / 31.08 | 25.56 / 24.89 |
| 4 | 35.61 / 41.79 | 61.74 / 32.16 | 27.44 / 24.98 |
| 5 | 36.29 / 39.22 | 32.01 / 34.71 | 25.20 / 26.08 |
| 6 | 33.34 / 33.98 | 31.52 / 36.10 | 27.02 / 25.79 |
| 7 | 34.18 / 36.08 | 32.09 / 34.36 | 25.74 / 27.59 |
| 8 | 36.55 / 35.31 | 31.57 / 34.43 | 25.73 / 25.93 |
| 9 | 34.52 / 36.78 | 32.82 / 32.56 | 28.31 / 27.68 |
| 10 | 33.06 / 35.19 | 31.18 / 36.39 | 27.23 / 26.12 |
| 11 | 35.61 / 35.97 | 30.69 / 35.91 | 26.39 / 25.18 |
| 12 | 34.83 / 36.08 | 34.07 / 46.49 | 24.23 / 24.46 |
| 13 | 35.72 / 35.03 | 34.54 / 35.67 | 33.88 / 24.81 |
| 14 | 34.93 / 35.46 | 33.99 / 33.82 | 24.58 / 25.59 |
| 15 | 34.27 / 34.87 | 35.93 / 37.23 | 31.39 / 27.38 |
| 16 | 35.43 / 35.79 | 33.69 / 33.83 | 25.20 / 23.25 |
| 17 | 36.51 / 36.25 | 33.57 / 35.45 | 24.77 / 31.54 |
| 18 | 38.19 / 38.84 | 33.38 / 34.06 | 26.15 / 26.81 |
| 19 | 32.77 / 38.05 | 31.73 / 33.39 | 26.34 / 22.98 |
| 20 | 33.96 / 36.02 | 34.55 / 34.23 | 27.20 / 23.77 |

</details>
