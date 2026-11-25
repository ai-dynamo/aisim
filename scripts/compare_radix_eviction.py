#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the radix-selection evidence using two immutable local revisions.

Only the test probe and its fixtures are overlaid. Production code and lockfiles
come from each recorded revision. No refs are fetched and no goldens are updated.
Snapshots, logs, and observations stay in --work-dir. Builds use isolated
snapshot subdirectories under CARGO_TARGET_DIR, or --work-dir/target if unset.
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
LIFECYCLE_RECORD = SCENARIOS.with_name("radix-matched-prefix-comparison.json")
TEST = "eviction_selection_matches_recorded_evidence"
MODULE = '\n#[cfg(test)]\n#[path = "eviction_evidence.rs"]\nmod eviction_evidence;\n'


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


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
    # Hash and install the same captured bytes even if sources change mid-build.
    overlay_paths = (probe, LIFECYCLE_RECORD) if lifecycle else (probe, SCENARIOS, COMPARISON)
    overlays = {relative: (ROOT / relative).read_bytes() for relative in overlay_paths}
    report = {
        "schema_version": 1,
        "probe": args.probe,
        "probe_sha256": hashlib.sha256(overlays[probe]).hexdigest(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "profile": "test",
        "features": [],
    }
    if lifecycle:
        report["record_sha256"] = hashlib.sha256(overlays[LIFECYCLE_RECORD]).hexdigest()
    if not lifecycle:
        report["scenarios_sha256"] = hashlib.sha256(overlays[SCENARIOS]).hexdigest()
    revisions = {
        side: git("rev-parse", "--verify", f"{getattr(args, side)}^{{commit}}") for side in ("baseline", "candidate")
    }
    for side in ("baseline", "candidate"):
        revision = revisions[side]
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
        for relative, contents in overlays.items():
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
        radix = snapshot / module_file
        if module.strip() not in radix.read_text():
            radix.write_text(radix.read_text() + module)
        # Keep artifacts (including embedded manifest paths) unique to this
        # snapshot. Archive mtimes and incremental caches are insufficient to
        # distinguish workspaces sharing the same package/dependency identity.
        target_root = Path(os.environ.get("CARGO_TARGET_DIR", str(work / "target")))
        snapshot_id = hashlib.sha256(str(snapshot).encode()).hexdigest()[:16]
        observed = work / f"{side}.json"
        command = ["cargo", "test", "-p", "aisimulate-core", "--locked", "--lib", test]
        env = {
            **os.environ,
            "CARGO_BUILD_JOBS": "2",
            "CARGO_TARGET_DIR": str(target_root / "radix-evidence" / snapshot_id),
            "CARGO_INCREMENTAL": "0",
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
        if lifecycle:
            expected = json.loads(overlays[LIFECYCLE_RECORD])[side]["scenarios"]
            observed_scenarios = report[side]["scenarios"] or []
            observed_by_name = {item["scenario"]: item for item in observed_scenarios}
            matches = (
                bool(expected)
                and len(observed_by_name) == len(observed_scenarios)
                and len(observed_scenarios) == len(expected)
                and observed_by_name.keys() == {item["scenario"] for item in expected}
                and all(observed_by_name.get(item["scenario"]) == item for item in expected)
            )
            report[side]["recorded_scenarios_match"] = matches
            (work / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
            if not matches:
                raise SystemExit(f"{side} lifecycle observations differ from the committed record: {observed}")
            print(
                f"{side} {revision}: {len(expected)} recorded lifecycle scenarios match "
                f"({len(observed_scenarios)} observed)",
                flush=True,
            )
        else:
            print(f"{side} {revision}: recorded eviction outputs match", flush=True)


if __name__ == "__main__":
    main()
