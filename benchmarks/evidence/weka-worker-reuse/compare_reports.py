# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare Kimi K3 AgentX recommendation reports from independent checkouts."""

import argparse
import hashlib
import json
from pathlib import Path


def report_digest(path: Path) -> str:
    report = json.loads(path.read_text())
    del report["provenance"]["created_at"], report["provenance"]["run_id"]
    for candidate in report["candidates"]:
        for field in ("wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s"):
            del candidate["metrics"][field]

    (trace,) = report["provenance"]["config"]["workload"]["trace_paths"]
    timing = report["candidates"][0]["prediction_config"]["engine"]["workers"]["aggregated"]["timing"]
    (systems,) = timing["systems_paths"]
    # Replace complete path values. Keep the trace name and all simulated metrics.
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"))
    encoded = encoded.replace(json.dumps(trace), json.dumps(Path(trace).name))
    encoded = encoded.replace(json.dumps(systems), json.dumps("<packaged systems>"))
    return hashlib.sha256(encoded.encode()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+", metavar="recommendation.json")
    args = parser.parse_args()
    digests = set()
    for path in args.reports:
        digest = report_digest(path)
        print(digest, path)
        digests.add(digest)
    if len(digests) != 1:
        parser.exit(1, "Recommendation reports differ.\n")
