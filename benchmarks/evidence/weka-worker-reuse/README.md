<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kimi K3 AgentX report comparison

The benchmark compared main at `19d8a245a` and worker reuse at `6366f1896`.
Each CLI run evaluated six candidates on the same 32-play AgentX sample.
The SHA-256 of `agentx-32.jsonl` was `32f48684cd46688f082b4379d22b28b29cfdc2718536b680bcae70a72a0ac60b`.

| Simulated window | Reports compared | SHA-256 of each normalized report |
| --- | ---: | --- |
| 120 seconds | 14 | `b433e0b486431842e04fbbf53fdba21f5665fa5da7ea6444bfb04365ea12d3da` |
| 900 seconds | 18 | `caf5f0cfc61331ad110bb509f2fe65887f4cb08bf40b94816535ef9fdd1a4349` |

Compare reports from the same window with this command.

```sh
python compare_reports.py BASE/recommendation.json PR/recommendation.json
```

The script removes the run ID, creation time, wall time, and two rates derived
from wall time. It replaces the trace path with its filename and the packaged
systems directory with a fixed label. All simulated metrics, candidate settings,
and rankings contribute to the digest. Different digests cause the command to fail.
