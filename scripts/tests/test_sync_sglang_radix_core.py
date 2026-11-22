# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check immutable source reads and safe version-update path handling with real Git."""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("sync_radix", Path(__file__).parents[1] / "sync_sglang_radix_core.py")
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


class NativeSourceTests(unittest.TestCase):
    def transform(self, source, text):
        return sync.native_source(dedent(text).lstrip(), source).removeprefix(sync.HEADER)

    def test_library_removes_python_and_torch_modules_but_preserves_native_modules(self):
        result = self.transform(
            "src/lib.rs",
            """
            #![allow(
                dead_code,
            )]
            #[cfg(feature = "python-extension")]
            mod python_bindings;
            #[cfg(all(test, feature = "torch"))]
            #[path = "tests/test_utils.rs"]
            pub(crate) mod test_utils;
            mod value;
            pub use crate::value::RadixValue;
            #[cfg(test)]
            mod native_tests { #[test] fn still_runs() { assert_eq!(2 + 2, 4); } }
            """,
        )
        self.assertNotIn("python_bindings", result)
        self.assertNotIn("test_utils", result)
        self.assertIn("    unused_imports,", result)
        self.assertIn("mod value;", result)
        self.assertIn("pub use crate::engine::cache::sglang_radix_tree::value::RadixValue;", result)
        self.assertIn("mod native_tests { #[test] fn still_runs() { assert_eq!(2 + 2, 4); } }", result)
        self.assertTrue(result.endswith("#[cfg(test)]\nmod page_value_core;\n"))

    def test_value_removes_tensor_implementation_and_preserves_following_native_assertions(self):
        result = self.transform(
            "src/value.rs",
            """
            pub struct NativeValue;
            #[cfg(feature = "torch")]
            impl RadixValue for Tensor { fn torch_only() {} }
            #[cfg(test)]
            mod tests {
                use crate::value::NativeValue;
                #[test] fn native_value() { assert_eq!(vec![1, 2].len(), 2); }
            }
            #[cfg(all(test, feature = "torch"))]
            mod tensor_tests { fn torch_only_test() {} }
            """,
        )
        self.assertEqual(
            result,
            dedent("""
                pub struct NativeValue;
                #[cfg(test)]
                mod tests {
                    use crate::engine::cache::sglang_radix_tree::value::NativeValue;
                    #[test] fn native_value() { assert_eq!(vec![1, 2].len(), 2); }
                }
                """).lstrip(),
        )

    def test_tree_removes_torch_wiring_and_preserves_following_native_code(self):
        result = self.transform(
            "src/unified_tree_core.rs",
            """
            use crate::value::RadixValue;
            #[cfg(feature = "torch")]
            use torch::Tensor;
            #[cfg(feature = "torch")]
            fn tensor_helper() {}
            #[derive(Clone, Copy, PartialEq, Eq, Debug)]
            pub enum NativePhase { Walk, Commit }
            #[cfg(any(test, feature = "inspection"))]
            fn inspect_native() { assert!(matches!(NativePhase::Walk, NativePhase::Walk)); }
            #[cfg(all(test, feature = "torch"))]
            mod torch_tests { fn tensor_test() {} }
            """,
        )
        self.assertEqual(
            result,
            dedent("""
                use crate::engine::cache::sglang_radix_tree::value::RadixValue;
                #[derive(Clone, Copy, PartialEq, Eq, Debug)]
                pub enum NativePhase { Walk, Commit }
                #[cfg(test)]
                fn inspect_native() { assert!(matches!(NativePhase::Walk, NativePhase::Walk)); }
                """).lstrip(),
        )

    def test_node_and_integration_test_imports_are_rewritten(self):
        node = self.transform("src/node.rs", "use std::fmt::Debug;\nuse crate::value::RadixValue;\n")
        self.assertEqual(
            node,
            "use thiserror1 as thiserror;\n\nuse std::fmt::Debug;\n"
            "use crate::engine::cache::sglang_radix_tree::value::RadixValue;\n",
        )
        integration = self.transform(
            "tests/page_value_core.rs",
            "use sglang_radix_tree::{PageValue};\n#[test] fn native() { assert_eq!(1, 1); }\n",
        )
        self.assertEqual(
            integration,
            "use crate::engine::cache::sglang_radix_tree::{self,PageValue};\n"
            "#[test] fn native() { assert_eq!(1, 1); }\n",
        )


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
        (self.source / "lib.rs").write_text("// pinned source\n", encoding="utf-8")
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

    def test_non_ascii_source_stays_utf8_in_update_and_check_modes(self):
        contents = "// size \u00d7 count; café; 缓存\n".encode()
        (self.source / "lib.rs").write_bytes(contents)
        self.git("add", ".")
        self.git(
            "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "unicode"
        )
        with patch.object(sync, "REVISION", self.git("rev-parse", "HEAD").strip()):
            write_text = Path.write_text

            def require_encoding(path, text, **kwargs):
                self.assertEqual(kwargs.get("encoding"), "utf-8")
                return write_text(path, text, **kwargs)

            with patch.object(Path, "write_text", require_encoding):
                sync.synchronize(self.checkout, check=False)
                self.assertEqual((self.destination / "mod.rs").read_bytes(), contents)
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
