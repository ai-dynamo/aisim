# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write the two B300 unqueued host-cost entries into a dataset-layout root and register them.

Usage: python publish.py ROOT
ROOT is a copy of ``ai-dynamo/aisim-data`` holding at least ``catalog/dataset.json`` and the ``host-cost``
category (the tokenizer entries the runs are split against).
"""

import json
import sys
from pathlib import Path

from aisim_core.data import Manifest
from aisim_core.data.host_cost import (
    MANIFEST_NAME,
    check_entry,
    entry_directory,
    register_entries,
    write_unqueued_entry,
)

HERE = Path(__file__).parent
root = Path(sys.argv[1])


def tokenizer(framework, version, key):
    path = root / entry_directory(framework, version, "b300_sxm", key) / MANIFEST_NAME
    return Manifest.model_validate_json(path.read_bytes())


seed42 = json.loads((HERE / "sglang-glm-seed42.json").read_text())
entries = [
    write_unqueued_entry(
        root,
        framework="sglang",
        version="0.5.20",
        system="b300_sxm",
        endpoint="chat",
        model="nvidia/GLM-5.3-NVFP4",
        runs=[
            {
                "run": "c1-exact-mtp-on-seed42",
                "requests": seed42["matched"],
                "base_ms": seed42["outside_engine_fit"]["base_ms"],
                "total_per_input_token_us": seed42["outside_engine_fit"]["us_per_token"],
                "r2": seed42["outside_engine_fit"]["r2"],
            },
            # Step file not retained; the fit as the measurement printed it (two decimals of base, three of slope).
            {"run": "mtp-on-seed7", "requests": 166, "base_ms": 5.7, "total_per_input_token_us": 1.587, "r2": 0.92},
        ],
        tokenizer_manifest=tokenizer("sglang", "0.5.21", "glm"),
        method={
            "measure": "client TTFT minus the engine prefill span per request at one lane (lane 1), fitted "
            "against total input tokens; benchmarks/evidence/host-cost-unqueued-20261125/measure.py",
            "load_generator": "aiperf 1.0.12, AgentX GLM-5.3 corpus, profiling phase only",
            "deployment": "SGLang 0.5.20, nvidia/GLM-5.3-NVFP4, B300 TP8, MTP on, chat endpoint",
            "engine_timestamps": "SGLang FPM step hooks (host_scheduled_at_s, host_emitted_at_s)",
            "in_sample": False,
        },
    ),
    write_unqueued_entry(
        root,
        framework="vllm",
        version="0.31.0",
        system="b300_sxm",
        endpoint="chat",
        model="Qwen/Qwen3.5-397B-A17B",
        runs=[
            # Midpoint of the l1 and l4 runs (25-34 ms fixed, 1.65-1.72 us/token in total); the per-run pairs were
            # not retained. 1.681 = 1.436 + 0.245, the total replay was validated with on 2026-11-18.
            {"run": "agentx-weka-l1-l4-midpoint", "requests": None, "base_ms": 29.5, "total_per_input_token_us": 1.681},
        ],
        tokenizer_manifest=tokenizer("vllm", "0.31.0", "qwen2"),
        method={
            "measure": "client TTFT minus engine time per request on the AgentX Weka l1 and l4 e2e runs; "
            "benchmarks/evidence/vllm-tokenization-host-cpu-20261116 (commit 0e87f7b35)",
            "deployment": "vLLM 0.31.0, Qwen/Qwen3.5-397B-A17B BF16, B300 TP8, chat endpoint",
            "in_sample": True,
        },
    ),
]
print(register_entries(root, entries))
for path in entries:
    check_entry(root, path)
print("\n".join(entries))
