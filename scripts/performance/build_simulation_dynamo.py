# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the pinned Dynamo replay adapter against one exact AISim checkout."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.release.package_identity import core_package

DYNAMO_REVISION = "df666fbc6284e540ea0f5cac613da7f992b1fc7b"
_GROUPED_REQUEST_ADAPTER = "lib/mocker/src/grouped_scheduler/compatibility.rs"
# Adapted from the pinned Apache-2.0 source. See THIRD_PARTY_NOTICES.md.
# https://github.com/ai-dynamo/dynamo/blob/df666fbc6284e540ea0f5cac613da7f992b1fc7b/lib/mocker/src/grouped_scheduler/compatibility.rs
_GROUPED_REQUEST_ADAPTER_SHA256 = "08c34b2ee709f5b038fa0801f40b3d493b58bd5aec8ae77767fa0bbc8265d84c"


def forward_fixed_output_reservation(core: Path, dynamo: Path) -> None:
    """Bridge the pinned request conversion when building against the new core API."""
    protocol = (core / "src/engine/protocol.rs").read_text()
    request = re.search(r"^pub struct Request \{\n(.*?)^\}", protocol, re.MULTILINE | re.DOTALL)
    if request is None:
        raise ValueError("could not find the AISim engine Request definition")
    field = re.search(r"^[ \t]*((?:pub(?:\([^)]*\))?[ \t]+)?ignore_eos[ \t]*:[^\n]+)$", request[1], re.MULTILINE)
    if field is None:
        return  # The base side still uses the pre-ignore_eos Request API.
    if field[1].strip() != "pub ignore_eos: bool,":
        raise ValueError("expected a public boolean Request.ignore_eos")

    adapter = dynamo / _GROUPED_REQUEST_ADAPTER
    source = adapter.read_bytes()
    if hashlib.sha256(source).hexdigest() != _GROUPED_REQUEST_ADAPTER_SHA256:
        raise ValueError("unexpected pinned Dynamo grouped request adapter contents")
    anchor = b"            max_output_tokens: request.max_output_tokens,\n"
    if source.count(anchor) != 1:
        raise ValueError("could not locate the pinned Dynamo request conversion")
    adapter.write_bytes(
        source.replace(
            anchor,
            anchor
            + b"            // Modified for the local AISim Request API.\n"
            + b"            ignore_eos: request.ignore_eos,\n",
        )
    )


def locked_pyo3_version(lock: Path) -> str:
    """Return the pyo3 version that a source's Cargo.lock pins.

    Only one package in a build may link `python`, so Dynamo's bindings and
    the selected AISim core must use the same pyo3.
    """
    versions = re.findall(r'^name = "pyo3"\nversion = "([^"]+)"$', lock.read_text(), re.MULTILINE)
    if len(versions) != 1:
        raise ValueError(f"expected one locked pyo3 version in {lock}, found {versions}")
    return versions[0]


def build(source: Path, dynamo: Path, output: Path) -> None:
    source, dynamo, output = source.resolve(), dynamo.resolve(), output.resolve()
    revision = subprocess.check_output(["git", "-C", str(dynamo), "rev-parse", "HEAD"], text=True).strip()
    if revision != DYNAMO_REVISION:
        raise ValueError("unexpected Dynamo revision")
    subprocess.run(["git", "-C", str(dynamo), "diff", "--exit-code", "HEAD"], check=True)
    core = source / "crates/core"
    package = core_package(core)
    forward_fixed_output_reservation(core, dynamo)
    bindings = dynamo / "lib/bindings/python"
    for manifest, options in (
        (dynamo / "Cargo.toml", ""),
        (bindings / "Cargo.toml", ', optional = true, features = ["python"]'),
    ):
        text, count = re.subn(
            r"^aisim-core = .*",
            lambda _: (f"aisim-core = {{ package = {json.dumps(package)}, path = {json.dumps(str(core))}{options} }}"),
            manifest.read_text(),
            flags=re.MULTILINE,
        )
        if count != 1:
            raise ValueError(f"expected one AISim dependency in {manifest}")
        manifest.write_text(text)
    # Resolve the local core's dependency graph from Dynamo's existing lock.
    # Compilation stays locked to this resolved graph, which is saved below.
    lock = bindings / "Cargo.lock"
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--format-version", "1", "--features", "ais-forward-pass"],
            cwd=bindings,
        )
    )
    linked = [p["manifest_path"] for p in metadata["packages"] if p["name"] == package]
    if linked != [str(core / "Cargo.toml")]:
        raise ValueError(f"Dynamo resolved a different AISim core: {linked}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "maturin",
            "build",
            "--release",
            "--locked",
            "--features",
            "ais-forward-pass",
            "--out",
            str(output),
        ],
        cwd=bindings,
        check=True,
    )
    subprocess.run(
        ["uv", "build", "--wheel", "--no-build-isolation", "--out-dir", str(output), str(dynamo)],
        check=True,
    )
    (output / "dynamo-Cargo.lock").write_bytes(lock.read_bytes())
    (output / "dynamo-build.json").write_text(
        json.dumps(
            {
                "source_sha": revision,
                "aisimulate_sha": subprocess.check_output(
                    ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
                ).strip(),
                "features": ["ais-forward-pass"],
            },
            indent=2,
        )
        + "\n"
    )
    (output / "dynamo-build.patch").write_bytes(subprocess.check_output(["git", "-C", str(dynamo), "diff", "HEAD"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dynamo", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.dynamo, args.output)
