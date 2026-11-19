#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the radix-selection evidence using two immutable local revisions.

Only the test probe and its fixtures are overlaid. Production code and lockfiles
come from each recorded revision. No refs are fetched and no goldens are updated.
All snapshots, builds, logs, and observations stay in --work-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "8288046ac88fccaee124714cd60e62b2ad4e184d"
PROBE = Path("crates/core/src/engine/cache/eviction_evidence.rs")
SCENARIOS = PROBE.with_name("eviction-selection-scenarios.json")
COMPARISON = SCENARIOS.with_name("eviction-selection-comparison.json")
TEST = "eviction_selection_matches_recorded_evidence"
MODULE = '\n#[cfg(test)]\n#[path = "eviction_evidence.rs"]\nmod eviction_evidence;\n'


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--candidate", default="HEAD")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--probe", choices=("selection", "lifecycle"), default="selection")
    args = parser.parse_args()
    lifecycle = args.probe == "lifecycle"
    probe = Path("crates/core/src/engine/kv_manager/radix_lifecycle_evidence.rs") if lifecycle else PROBE
    module_file = probe.with_name("sglang_backend.rs") if lifecycle else PROBE.with_name("radix_cache.rs")
    module = (
        '\n#[cfg(test)]\n#[path = "radix_lifecycle_evidence.rs"]\nmod radix_lifecycle_evidence;\n'
        if lifecycle
        else MODULE
    )
    test = "radix_lifecycle_matches_recorded_behavior" if lifecycle else TEST
    work = args.work_dir.resolve()
    if work.is_relative_to(ROOT):
        parser.error("--work-dir must be outside the source checkout")
    # Refuse reuse: artifacts must belong to exactly the recorded revisions.
    work.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "probe": args.probe,
        "probe_sha256": sha256(ROOT / probe),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "profile": "test",
        "features": [],
    }
    if not lifecycle:
        report["scenarios_sha256"] = sha256(ROOT / SCENARIOS)
    for side in ("baseline", "candidate"):
        revision = git("rev-parse", "--verify", f"{getattr(args, side)}^{{commit}}")
        snapshot = work / side
        snapshot.mkdir()
        archive = work / f"{side}.tar"
        with archive.open("wb") as output:
            subprocess.run(
                [
                    "git",
                    "archive",
                    revision,
                    "Cargo.toml",
                    "Cargo.lock",
                    "crates/core",
                    "python/aisimulate/tests/fixtures",
                ],
                cwd=ROOT,
                stdout=output,
                check=True,
            )
        with tarfile.open(archive) as source:
            source.extractall(snapshot, filter="data")
        archive.unlink()
        for relative in (probe,) if lifecycle else (probe, SCENARIOS, COMPARISON):
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
        radix = snapshot / module_file
        if module.strip() not in radix.read_text():
            radix.write_text(radix.read_text() + module)
        observed = work / f"{side}.json"
        command = ["cargo", "test", "-p", "aisimulate-core", "--locked", "--lib", test]
        env = {
            **os.environ,
            "CARGO_BUILD_JOBS": "2",
            "CARGO_TARGET_DIR": os.environ.get("CARGO_TARGET_DIR", str(work / "target")),
            "AISIM_EVICTION_EVIDENCE_OUTPUT": str(observed),
            "AISIM_EVICTION_EVIDENCE_SIDE": side,
        }
        with (work / f"{side}.log").open("w") as log:
            result = subprocess.run(command, cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT)
        report[side] = {
            "commit": revision,
            "command": command,
            "exit_code": result.returncode,
            "scenarios": json.loads(observed.read_text()) if observed.exists() else None,
        }
        (work / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        if result.returncode:
            raise SystemExit(f"{side} failed: inspect {work / (side + '.log')}")
        print(f"{side} {revision}: recorded eviction outputs match", flush=True)


if __name__ == "__main__":
    main()
