# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify immutable comparison overlays while source files change during builds."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("compare_radix", Path(__file__).parents[1] / "compare_radix_eviction.py")
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


class CompareRadixTests(unittest.TestCase):
    def check_frozen_overlays(
        self,
        mode,
        *,
        corrupt_record=False,
        corrupt_observation=False,
        extra_observation=False,
        missing_observation=False,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "checkout"
            checkout.mkdir()
            work = root / "comparison"
            lifecycle = mode == "lifecycle"
            probe = (
                Path("crates/core/src/engine/kv_manager/radix_lifecycle_evidence.rs") if lifecycle else compare.PROBE
            )
            overlays = {probe: b"// original probe\n"}
            observations = (
                [
                    {"scenario": "two_page_leaf", "removed_block_hashes": [101]},
                    {"scenario": "complete_without_new_page", "removed_block_hashes": [102]},
                    {"scenario": "match_and_lock_compressed_prefix", "removed_block_hashes": [103]},
                ]
                if lifecycle
                else []
            )
            if lifecycle:
                recorded = [dict(item) for item in observations]
                if corrupt_record:
                    recorded[2] = {**recorded[2], "removed_block_hashes": [999]}
                overlays[compare.LIFECYCLE_RECORD] = json.dumps(
                    {side: {"scenarios": recorded} for side in ("baseline", "candidate")}
                ).encode()
                if corrupt_observation:
                    observations[2] = {**observations[2], "removed_block_hashes": [999]}
                if extra_observation:
                    observations.append({"scenario": "unrecorded", "removed_block_hashes": [104]})
                if missing_observation:
                    observations.pop()
            if not lifecycle:
                overlays.update({compare.SCENARIOS: b"[]\n", compare.COMPARISON: b"{}\n"})
            module = probe.with_name("sglang_backend.rs" if lifecycle else "radix_cache.rs")
            files = {
                **overlays,
                module: b"// original module\n",
                Path("Cargo.toml"): b"# fixture manifest\n",
                Path("Cargo.lock"): b"# fixture lockfile\n",
                Path("python/aisimulate/tests/fixtures/fixture.txt"): b"fixture\n",
            }
            for relative, contents in files.items():
                target = checkout / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(contents)

            def git(*args):
                return subprocess.check_output(["git", *args], cwd=checkout, text=True).strip()

            git("init", "--quiet")
            git("add", ".")
            git(
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "fixture",
            )
            revision = git("rev-parse", "HEAD")
            run = subprocess.run
            builds = []

            def run_with_source_edits(command, **kwargs):
                if command[0] == "rustc":
                    return subprocess.CompletedProcess(command, 0, stdout="rustc fixture\n")
                if command[0] != "cargo":
                    return run(command, **kwargs)
                side = kwargs["env"]["AISIM_EVICTION_EVIDENCE_SIDE"]
                builds.append(side)
                for relative, contents in overlays.items():
                    self.assertEqual((kwargs["cwd"] / relative).read_bytes(), contents, f"{side}: {relative}")
                    # A deterministic edit during each simulated Cargo build.
                    (checkout / relative).write_bytes(f"edited during {side}\n".encode())
                Path(kwargs["env"]["AISIM_EVICTION_EVIDENCE_OUTPUT"]).write_text(json.dumps(observations))
                return subprocess.CompletedProcess(command, 0)

            argv = [
                "compare_radix_eviction.py",
                "--baseline",
                revision,
                "--candidate",
                revision,
                "--work-dir",
                str(work),
                "--probe",
                mode,
            ]
            with (
                patch.object(compare, "ROOT", checkout),
                patch("sys.argv", argv),
                patch.object(compare.subprocess, "run", side_effect=run_with_source_edits),
            ):
                if corrupt_record or corrupt_observation or extra_observation or missing_observation:
                    with self.assertRaisesRegex(SystemExit, "lifecycle observations differ"):
                        compare.main()
                    report = json.loads((work / "comparison.json").read_text())
                    self.assertFalse(report["baseline"]["recorded_scenarios_match"])
                    self.assertEqual(builds, ["baseline"])
                    return
                compare.main()

            self.assertEqual(builds, ["baseline", "candidate"])
            report = json.loads((work / "comparison.json").read_text())
            self.assertEqual(report["probe_sha256"], hashlib.sha256(overlays[probe]).hexdigest())
            if not lifecycle:
                self.assertEqual(report["scenarios_sha256"], hashlib.sha256(overlays[compare.SCENARIOS]).hexdigest())
            for relative, contents in overlays.items():
                self.assertNotEqual((checkout / relative).read_bytes(), contents)
                for side in builds:
                    self.assertEqual((work / side / relative).read_bytes(), contents)

    def test_selection_overlays_stay_frozen_during_builds(self):
        self.check_frozen_overlays("selection")

    def test_lifecycle_rejects_a_changed_recorded_removal_hash(self):
        self.check_frozen_overlays("lifecycle", corrupt_record=True)

    def test_lifecycle_overlay_stays_frozen_during_builds(self):
        self.check_frozen_overlays("lifecycle")

    def test_lifecycle_rejects_a_changed_third_observation_removal_hash(self):
        self.check_frozen_overlays("lifecycle", corrupt_observation=True)

    def test_lifecycle_rejects_an_extra_observation(self):
        self.check_frozen_overlays("lifecycle", extra_observation=True)

    def test_lifecycle_rejects_a_missing_observation(self):
        self.check_frozen_overlays("lifecycle", missing_observation=True)


if __name__ == "__main__":
    unittest.main()
