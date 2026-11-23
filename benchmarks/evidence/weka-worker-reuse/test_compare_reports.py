# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

from compare_reports import report_digest


def test_relocated_trace_matches_and_changed_input_differs(tmp_path):
    def digest(directory, lanes):
        config = {"workload": {"trace_paths": [f"{directory}/agentx-32.jsonl"], "agentic_lanes": lanes}}
        provenance = {
            "config": config,
            "input_fingerprint": hashlib.sha256(json.dumps(config).encode()).hexdigest(),
            "created_at": "",
            "run_id": "",
        }
        timing = {"systems_paths": [f"{directory}/systems"]}
        engine = {"workers": {"aggregated": {"timing": timing}}}
        metrics = {"wall_time_ms": 1, "processed_tokens_per_s": 1, "processed_output_tokens_per_s": 1}
        candidate = {"metrics": metrics, "prediction_config": {"engine": engine}}
        path = tmp_path / "report.json"
        path.write_text(json.dumps({"provenance": provenance, "candidates": [candidate]}))
        return report_digest(path)

    original = digest("/base", 16)
    assert original == digest("/pr", 16)
    assert original != digest("/pr", 8)
