# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the alternating six-round forward benchmark envelope.

Usage: canonical_mtp_forward_rounds.py BASE_SDK CANDIDATE_SDK OUT.json
Each SDK directory must contain its installed aisim and aisim_core packages.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def collect_rounds(baseline_sdk, candidate_sdk, driver=None):
    driver = Path(driver or Path(__file__).with_name("canonical_mtp_forward.py")).resolve()
    sdks = {"baseline": Path(baseline_sdk).resolve(), "candidate": Path(candidate_sdk).resolve()}
    rounds = []
    with tempfile.TemporaryDirectory() as temporary:
        for index in range(6):
            order = ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
            record = {"order": order, "results": {}}
            for label in order:
                output = Path(temporary) / f"round-{index + 1}-{label}.json"
                env = os.environ | {"PYTHONPATH": str(sdks[label]), "AIC_ALLOW_UNLISTED_VERSIONS": "1"}
                # A new interpreter owns each model/data cache. Never import
                # either SDK into this controller's process.
                subprocess.run([sys.executable, str(driver), str(output)], env=env, check=True)
                result = json.loads(output.read_text())
                for key in ("constructor_path", "native_path"):
                    loaded = Path(result["identity"][key]).resolve()
                    if not loaded.is_relative_to(sdks[label]):
                        raise ValueError(f"{label}: {key} imported outside selected SDK: {loaded}")
                record["results"][label] = result
            rounds.append(record)
    return {"rounds": rounds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_sdk", type=Path)
    parser.add_argument("candidate_sdk", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(collect_rounds(args.baseline_sdk, args.candidate_sdk), indent=1) + "\n")


if __name__ == "__main__":
    main()
