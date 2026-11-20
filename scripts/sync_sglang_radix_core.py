#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the bundled native core from an immutable, local SGLang checkout.

No network access or upstream working-tree changes are required. The source
mapping, mechanical adaptations, and local corrections are documented alongside
the bundled files.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "crates/core/src/engine/cache/sglang_radix_tree"
REVISION = "f0fda25fc95c357368b714f90a13a69c1b67f50a"
UPSTREAM = "rust/sglang-radix-tree"
LOCAL_PATCH = Path(__file__).with_name("sglang_radix_core_fixes.patch")
SOURCES = {
    "mod.rs": "src/lib.rs",
    "node.rs": "src/node.rs",
    "unified_lru_list.rs": "src/unified_lru_list.rs",
    "unified_tree_core.rs": "src/unified_tree_core.rs",
    "value.rs": "src/value.rs",
    "components/mod.rs": "src/components/mod.rs",
    "components/full.rs": "src/components/full.rs",
    "components/mamba.rs": "src/components/mamba.rs",
    "components/swa.rs": "src/components/swa.rs",
    "page_value_core.rs": "tests/page_value_core.rs",
}
HEADER = f"""// SPDX-FileCopyrightText: SGLang contributors
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Derived from sgl-project/sglang at {REVISION}.
// See README.md for original paths, license, and documented modifications.

"""


def native_source(original: str, source: str) -> str:
    """Adapt build wiring and names; leave native code and assertions intact."""
    text = original
    torch_tests = '#[cfg(all(test, feature = "torch"))]'
    if source != "src/lib.rs" and torch_tests in text:
        assert text.count(torch_tests) == 1
        text = text[: text.index(torch_tests)].rstrip() + "\n"
    if source in {"src/unified_tree_core.rs", "src/value.rs"}:
        start = text.index('#[cfg(feature = "torch")]')
        if source == "src/value.rs":
            text = text[:start] + text[text.index("#[cfg(test)]", start) :]
        else:
            text = text[:start] + text[text.index("#[derive(Clone, Copy, PartialEq, Eq, Debug)]", start) :]
    if source == "src/lib.rs":
        text = text.replace('#[cfg(feature = "python-extension")]\nmod python_bindings;\n', "")
        text = text.replace(
            '#[cfg(all(test, feature = "torch"))]\n#[path = "tests/test_utils.rs"]\npub(crate) mod test_utils;\n',
            "",
        )
        text = text.replace("    dead_code,", "    dead_code,\n    unused_imports,")
        text += "\n#[cfg(test)]\nmod page_value_core;\n"
    module = "crate::engine::cache::sglang_radix_tree::"
    text = text.replace("crate::", module)
    if source == "src/node.rs":
        # thiserror 1.x's derives also emit paths through the `thiserror` name.
        text = text.replace("use std::", "use thiserror1 as thiserror;\n\nuse std::", 1)
    text = text.replace('#[cfg(any(test, feature = "inspection"))]', "#[cfg(test)]")
    if source == "tests/page_value_core.rs":
        text = text.replace("use sglang_radix_tree::{", f"use {module}{{self,")
    assert 'feature = "torch"' not in text and 'feature = "python-extension"' not in text
    return HEADER + text


def synchronize(checkout: Path, *, check: bool) -> None:
    def upstream(path: str) -> bytes:
        return subprocess.check_output(["git", "--no-replace-objects", "show", f"{REVISION}:{path}"], cwd=checkout)

    with tempfile.TemporaryDirectory(prefix="aisimulate-radix-source-") as directory:
        generated = Path(directory)
        for destination, source in SOURCES.items():
            path = generated / destination
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(native_source(upstream(f"{UPSTREAM}/{source}").decode(), source))
        (generated / "LICENSE").write_bytes(upstream("LICENSE"))
        subprocess.run(["rustfmt", "--edition", "2024", str(generated / "mod.rs")], check=True)
        apply_local_fixes(generated)
        expected = {*SOURCES, "LICENSE"}
        actual = {str(path.relative_to(DESTINATION)) for path in DESTINATION.rglob("*.rs")} | {"LICENSE"}
        if actual - expected or (check and expected - actual):
            raise SystemExit(f"unexpected bundled source paths: {actual ^ expected}")
        for name in sorted(expected):
            destination = DESTINATION / name
            if check:
                if destination.read_bytes() != (generated / name).read_bytes():
                    raise SystemExit(f"bundled source differs from mechanical upstream adaptation: {name}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(generated / name, destination)
    print(f"{'Verified' if check else 'Updated'} bundled native core from SGLang {REVISION}")


def apply_local_fixes(generated: Path) -> None:
    """Apply the reviewed corrections after the mechanical source adaptation."""
    subprocess.run(["git", "apply", str(LOCAL_PATCH)], cwd=generated, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    synchronize(args.source_checkout.resolve(), check=args.check)


if __name__ == "__main__":
    main()
