# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check immutable source reads and safe version-update path handling with real Git."""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("sync_radix", Path(__file__).parents[1] / "sync_sglang_radix_core.py")
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


class SyncRadixTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.checkout = self.root / "upstream"
        self.checkout.mkdir()
        self.destination = self.root / "bundled"
        self.destination.mkdir()
        self.git("init", "--quiet")
        self.source = self.checkout / sync.UPSTREAM / "src"
        self.source.mkdir(parents=True)
        (self.checkout / "LICENSE").write_text("fixture license\n")
        (self.source / "lib.rs").write_text("// pinned source\n")
        (self.source / "new.rs").write_text("// newly mapped source\n")
        self.git("add", ".")
        self.git(
            "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "fixture"
        )
        self.revision = self.git("rev-parse", "HEAD").strip()
        for name, value in {
            "REVISION": self.revision,
            "DESTINATION": self.destination,
            "SOURCES": {"mod.rs": "src/lib.rs", "new.rs": "src/new.rs"},
            "native_source": lambda text, _source: text,
            "apply_local_fixes": lambda _generated: None,
        }.items():
            patcher = patch.object(sync, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        run = subprocess.run

        def run_without_rustfmt(command, **kwargs):
            if command[0] == "rustfmt":
                return subprocess.CompletedProcess(command, 0)
            return run(command, **kwargs)

        patcher = patch.object(sync.subprocess, "run", side_effect=run_without_rustfmt)
        patcher.start()
        self.addCleanup(patcher.stop)

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.checkout, text=True)

    def test_update_installs_newly_mapped_files_and_check_is_exact(self):
        (self.destination / "mod.rs").write_text("// old source\n")
        with self.assertRaisesRegex(SystemExit, "unexpected bundled source paths"):
            sync.synchronize(self.checkout, check=True)
        sync.synchronize(self.checkout, check=False)
        self.assertEqual((self.destination / "new.rs").read_text(), "// newly mapped source\n")
        sync.synchronize(self.checkout, check=True)
        (self.destination / "new.rs").write_text("// drift\n")
        with self.assertRaisesRegex(SystemExit, "bundled source differs"):
            sync.synchronize(self.checkout, check=True)

    def test_update_and_check_reject_unexpected_sources_without_writing(self):
        (self.destination / "unexpected.rs").write_text("// local source\n")
        for check in [False, True]:
            with self.assertRaisesRegex(SystemExit, "unexpected bundled source paths"):
                sync.synchronize(self.checkout, check=check)
            self.assertFalse((self.destination / "mod.rs").exists())

    def test_replacement_ref_cannot_change_pinned_source(self):
        (self.source / "lib.rs").write_text("// replacement source\n")
        self.git("add", ".")
        self.git(
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "replacement",
        )
        self.git("replace", self.revision, "HEAD")
        self.assertEqual(self.git("show", f"{self.revision}:{sync.UPSTREAM}/src/lib.rs"), "// replacement source\n")
        sync.synchronize(self.checkout, check=False)
        self.assertEqual((self.destination / "mod.rs").read_text(), "// pinned source\n")
        sync.synchronize(self.checkout, check=True)


if __name__ == "__main__":
    unittest.main()
