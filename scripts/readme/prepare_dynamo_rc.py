#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare exact release-candidate wheels while the README pair is unpublished."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = {"aisim": "0.14.0", "ai-dynamo": "1.7.0", "ai-dynamo-runtime": "1.7.0"}
SOURCES = {
    "aisim": {
        "repo": "https://github.com/ai-dynamo/aisim.git",
        "revision": "77d82652b2f83fd67afefc3141d69a5f06e630be",
    },
    "dynamo": {
        "repo": "https://github.com/ai-dynamo/dynamo.git",
        "revision": "9653d05441b37b6a004837651fd991e44158d755",
    },
}
INDEXES = ("https://pypi.org/simple", "https://pypi.nvidia.com")


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


class SimpleIndexParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.is_html = False
        self.filenames: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.is_html |= tag == "html"
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value:
                    self.filenames.append(unquote(Path(urlsplit(value).path).name))


def index_filenames(payload: bytes, content_type: str) -> list[str]:
    if "json" in content_type:
        data = json.loads(payload)
        if not isinstance(data, dict) or not isinstance(data.get("files"), list):
            raise ValueError("Malformed simple JSON index: expected files array")
        files = data["files"]
        if any(not isinstance(item, dict) or not isinstance(item.get("filename"), str) for item in files):
            raise ValueError("Malformed simple JSON index: expected filenames")
        return [item["filename"] for item in files]
    if "html" not in content_type:
        raise ValueError(f"Unexpected simple index content type: {content_type}")
    parser = SimpleIndexParser()
    parser.feed(payload.decode("utf-8"))
    parser.close()
    if not parser.is_html:
        raise ValueError("Malformed simple HTML index: expected HTML document")
    return parser.filenames


def distribution_identity(filename: str) -> tuple[str, str] | None:
    if filename.endswith(".whl"):
        parts = filename[:-4].split("-")
        if len(parts) in (5, 6):
            return normalized(parts[0]), parts[1]
    for suffix in (".tar.gz", ".zip"):
        if filename.endswith(suffix) and "-" in filename[: -len(suffix)]:
            name, version = filename[: -len(suffix)].rsplit("-", 1)
            return normalized(name), version
    return None


def available(index: str, name: str, version: str) -> bool:
    request = Request(
        f"{index}/{name}/",
        headers={"Accept": "application/vnd.pypi.simple.v1+json, text/html;q=0.9"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            filenames = index_filenames(response.read(), response.headers.get("Content-Type", ""))
    except HTTPError as error:
        if error.code == 404:
            return False
        raise
    return any(distribution_identity(filename) == (name, version) for filename in filenames)


def validate_versions(root: Path = ROOT) -> None:
    project = tomllib.loads((root / "python/aisim/pyproject.toml").read_text())["project"]
    if project["name"] != "aisim" or project["version"] != PACKAGES["aisim"]:
        raise ValueError("AISim source version does not match RC preparation pins")
    readme = (root / "README.md").read_text()
    block = re.search(r"<!-- readme-check: dynamo-install -->\n```bash\n(.*?)^```", readme, re.M | re.S)
    if block is None:
        raise ValueError("README dynamo-install block is missing")
    pins = dict(re.findall(r"\b([a-z][a-z0-9-]*)==([a-zA-Z0-9.+-]+)", block[1]))
    if pins != {name: PACKAGES[name] for name in ("aisim", "ai-dynamo")}:
        raise ValueError("README Dynamo versions do not match RC preparation pins")


def probe(root: Path = ROOT) -> dict:
    validate_versions(root)
    missing = []
    for name, version in PACKAGES.items():
        # An outage must fail rather than silently select a source build.
        found = [available(index, name, version) for index in INDEXES]
        if not any(found):
            missing.append(name)
    return {
        "source": "release-candidate" if missing else "public-index",
        "packages": dict(PACKAGES),
        "sources": SOURCES,
        "missing": missing,
        "wheels": [],
    }


def inspect_wheel(path: Path) -> dict:
    identity = distribution_identity(path.name)
    if identity is None or PACKAGES.get(identity[0]) != identity[1]:
        raise ValueError(f"Unexpected wheel filename/version: {path.name}")
    with ZipFile(path) as archive:
        metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError(f"Expected one wheel METADATA: {path.name}")
        message = BytesParser().parsebytes(archive.read(metadata[0]))
    if (normalized(message.get("Name", "")), message.get("Version")) != identity:
        raise ValueError(f"Wheel METADATA does not match filename/version: {path.name}")
    with path.open("rb") as wheel:
        digest = hashlib.file_digest(wheel, "sha256").hexdigest()
    return {"filename": path.name, "sha256": digest, "name": identity[0], "version": identity[1]}


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def checkout(source: dict, destination: Path) -> None:
    run("git", "init", str(destination))
    run("git", "-C", str(destination), "remote", "add", "origin", source["repo"])
    run("git", "-C", str(destination), "fetch", "--depth", "1", "origin", source["revision"])
    run("git", "-C", str(destination), "checkout", "--detach", source["revision"])


def build(wheelhouse: Path, root: Path = ROOT) -> dict:
    manifest = probe(root)
    if manifest["source"] == "public-index":
        return manifest
    wheelhouse = wheelhouse.resolve()
    wheelhouse.mkdir(parents=True, exist_ok=True)
    if any(wheelhouse.glob("*.whl")):
        raise ValueError("RC wheelhouse already contains wheels; use an empty directory")
    with tempfile.TemporaryDirectory(prefix="aisim-dynamo-rc-") as temporary:
        sources = Path(temporary)
        for name, source in SOURCES.items():
            checkout(source, sources / name)
        projects = (
            ("aisim", sources / "aisim/python/aisim", "--locked"),
            ("ai-dynamo", sources / "dynamo", None),
            ("ai-dynamo-runtime", sources / "dynamo/lib/bindings/python", "--locked --features ais-forward-pass"),
        )
        for name, directory, build_args in projects:
            project = tomllib.loads((directory / "pyproject.toml").read_text())["project"]
            if project["name"] != name or project["version"] != PACKAGES[name]:
                raise ValueError(f"Pinned source project version mismatch: {name}")
            command = ["uv", "build", "--wheel", "--python", sys.executable, "--out-dir", str(wheelhouse)]
            if build_args:
                command.append(f"--config-setting=build-args={build_args}")
            run(*command, str(directory))
    manifest["wheels"] = [inspect_wheel(path) for path in sorted(wheelhouse.glob("*.whl"))]
    if sorted(item["name"] for item in manifest["wheels"]) != sorted(PACKAGES):
        raise ValueError("RC build must produce exactly one wheel for each expected package")
    manifest["wheelhouse"] = str(wheelhouse)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "build"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name == "build":
            command.add_argument("--wheelhouse", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.wheelhouse) if args.command == "build" else probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{manifest['source']}: {', '.join(manifest['missing']) or 'all packages published'}")


if __name__ == "__main__":
    main()
