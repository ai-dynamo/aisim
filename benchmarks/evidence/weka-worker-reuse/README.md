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
| 120 seconds | 14 | `2ccd7e279fb016e6baef8675c2ad8a980e1931ff114de9c245aa3ac4ac0ace45` |
| 900 seconds | 18 | `030083a2216197b529b471672420e2fcad9e50d45676b35f35a08d28389f6c01` |

Compare reports from the same window with this command.

```sh
python compare_reports.py BASE/recommendation.json PR/recommendation.json
```

The script removes the run ID, creation time, wall time, and two rates derived
from wall time. It omits the fingerprint of the original configuration because
that hash includes absolute paths. The full configuration stays in the comparison.
The script replaces the trace path with its filename and the packaged
systems directory with a fixed label. All simulated metrics, candidate settings,
and rankings contribute to the digest. Different digests cause the command to fail.
