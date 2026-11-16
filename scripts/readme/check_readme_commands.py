#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute annotated README blocks; keep installation profiles isolated."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import venv
import zipfile
from email.parser import BytesParser
from pathlib import Path

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ("source", "macos", "development", "release", "dynamo")


def read_blocks(readme: str, manifest: dict) -> dict[str, str]:
    fences = list(re.finditer(r"^```(bash|yaml)\n(.*?)^```", readme, re.M | re.S))
    blocks = {}
    for fence in fences:
        marker = re.search(r"<!-- readme-check: ([a-z0-9-]+) -->\n$", readme[: fence.start()])
        if not marker:
            raise ValueError("Every bash/yaml block needs a readme-check ID")
        name = marker[1]
        if name in blocks or name not in manifest:
            raise ValueError(f"Duplicate or unregistered block: {name}")
        entry = manifest[name]
        if (fence[1] == "yaml") != ("file" in entry):
            raise ValueError(f"Wrong block type: {name}")
        blocks[name] = fence[2]
    if blocks.keys() != manifest.keys():
        raise ValueError(f"Manifest drift: {manifest.keys() - blocks.keys()}")
    for name, entry in manifest.items():
        if "file" in entry:
            if Path(entry["file"]).name != entry["file"]:
                raise ValueError(f"Unsafe output path: {name}")
            yaml.safe_load(blocks[name])
        else:
            if not entry["profiles"] or set(entry["profiles"]) - set(PROFILES):
                raise ValueError(f"Invalid profiles: {name}")
            for dependency in entry["needs"]:
                if dependency not in list(blocks)[: list(blocks).index(name)] and dependency != "source-install":
                    raise ValueError(f"Dependency must run first: {name}: {dependency}")
    return blocks


def run_shell(command: str, cwd: Path, env: dict, log: Path, timeout: float) -> tuple[int, bool]:
    """Track descendants too: sweeper workers can create their own sessions."""
    descendants: set[psutil.Process] = set()
    timed_out = False
    with log.open("w") as output:
        output.write(command + "\n\n")
        output.flush()
        process = subprocess.Popen(
            ["bash", "-euo", "pipefail", "-c", command],
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        parent = psutil.Process(process.pid)
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                try:
                    descendants.update(parent.children(recursive=True))
                except psutil.NoSuchProcess:
                    pass
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                time.sleep(0.1)
        finally:
            # Also remove leaked workers after an otherwise successful command.
            for child in descendants:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            psutil.wait_procs(list(descendants), timeout=5)
    return process.returncode, timed_out


def assert_outputs(entry: dict, cwd: Path, log: Path) -> None:
    if path := entry.get("prediction"):
        report = json.loads((cwd / path).read_text())
        summary = report.get("summary", report)
        if summary.get("completed_requests", 0) <= 0:
            raise ValueError("Prediction completed no requests")
        if "completed_requests" in entry and summary["completed_requests"] != entry["completed_requests"]:
            raise ValueError("Prediction completed an unexpected request count")
        for metric, expected in entry.get("metrics", {}).items():
            if summary.get(metric) != expected:
                raise ValueError(f"Unexpected {metric}: {summary.get(metric)}")
    if (path := entry.get("json_summary")) and (
        json.loads((cwd / path).read_text())["completed_requests"] != entry["completed_requests"]
    ):
        raise ValueError("JSON stdout disagrees with the requested count")
    if path := entry.get("requests"):
        requests = [json.loads(line) for line in (cwd / path).read_text().splitlines()]
        if len(requests) != entry["completed_requests"]:
            raise ValueError("Per-request capture is incomplete")
    if path := entry.get("recommendations"):
        candidates = sorted((cwd / path).glob("*.yaml"))
        if not candidates or candidates[0].name != "0001.yaml":
            raise ValueError("Recommendation produced no ranked candidate")
        for candidate in candidates:
            config = yaml.safe_load(candidate.read_text())
            # Check the saved shape; the round-trip command validates the public schema.
            if not isinstance(config, dict) or not isinstance(config.get("engine"), dict):
                raise ValueError(f"Invalid candidate: {candidate.name}")

            def concrete(value):
                if isinstance(value, dict):
                    if set(value) & {"choices", "range", "preset"}:
                        raise ValueError(f"Unresolved search domain: {candidate.name}")
                    for item in value.values():
                        concrete(item)
                elif isinstance(value, list):
                    for item in value:
                        concrete(item)

            concrete(config["engine"])
    for expected in entry.get("stdout_contains", []):
        # The first part of the log is the command; only inspect actual output.
        if expected not in log.read_text().split("\n\n", 1)[1]:
            raise ValueError(f"Missing expected output: {expected}")


def candidate_environment(path: Path, output: Path, env: dict) -> dict:
    """Constrain installs to verified RC artifacts without rewriting README commands."""
    candidate = json.loads(path.read_text())
    if candidate["source"] == "public-index":
        if candidate.get("wheels"):
            raise ValueError("Public-index validation must not substitute local wheels")
        return candidate
    if candidate["source"] != "release-candidate":
        raise ValueError("Unknown candidate source")
    wheelhouse = Path(candidate["wheelhouse"]).resolve(strict=True)
    expected = candidate["packages"]
    if set(expected) != {"aisim", "ai-dynamo", "ai-dynamo-runtime"}:
        raise ValueError("Candidate must include the complete Dynamo release pair and runtime")
    constraints = []
    seen = set()
    for wheel in candidate["wheels"]:
        filename = wheel["filename"]
        if Path(filename).name != filename or not filename.endswith(".whl"):
            raise ValueError("Unsafe candidate wheel filename")
        artifact = (wheelhouse / filename).resolve(strict=True)
        if artifact.parent != wheelhouse:
            raise ValueError("Candidate wheel escaped its wheelhouse")
        with artifact.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != wheel["sha256"]:
            raise ValueError(f"Candidate wheel hash mismatch: {filename}")
        with zipfile.ZipFile(artifact) as archive:
            metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError(f"Invalid wheel metadata: {filename}")
            package = BytesParser().parsebytes(archive.read(metadata[0]))
        name = package["Name"].lower().replace("_", "-")
        version = package["Version"]
        if name in seen or expected.get(name) != version or (wheel["name"], wheel["version"]) != (name, version):
            raise ValueError(f"Candidate wheel package/version mismatch: {filename}")
        seen.add(name)
        constraints.append(f"{name} @ {artifact.as_uri()}")
    if seen != set(expected):
        raise ValueError("Candidate wheel set is incomplete")
    constraint_path = output / "candidate-constraints.txt"
    constraint_path.write_text("\n".join(constraints) + "\n")
    env["PIP_CONSTRAINT"] = str(constraint_path)
    return candidate


def run_profile(
    profile: str,
    workspace: Path,
    output: Path,
    blocks: dict,
    manifest: dict,
    candidate_manifest: Path | None = None,
) -> dict:
    workspace.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    package_source = {"source": "public-index"}
    if candidate_manifest:
        if profile != "dynamo":
            raise ValueError("Candidate wheels are only supported for the Dynamo profile")
        package_source = candidate_environment(candidate_manifest, output, env)
    print(f"Package source: {package_source['source']}", flush=True)
    # All examples use public models. Isolate host credentials and caches without
    # changing the user's Hugging Face login or production authentication code.
    env.pop("HF_TOKEN", None)
    env.pop("HUGGING_FACE_HUB_TOKEN", None)
    env["HF_HOME"] = str(workspace / ".huggingface")
    env["HF_TOKEN_PATH"] = str(workspace / ".huggingface/token")
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    env["AISIM_REF"] = sha
    # Clone exactly the checked-out tree, including commits on a not-yet-merged PR.
    # The README command itself remains unchanged; no network fetch of mutable main.
    env.update(
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0=f"url.{ROOT}.insteadOf",
        GIT_CONFIG_VALUE_0="https://github.com/ai-dynamo/aisim.git",
    )
    source = profile in {"source", "macos", "development"}
    cwd = workspace / "aisim" if source else workspace
    environment = cwd / ".venv"
    if not source:
        venv.create(environment, with_pip=True)
    env["PATH"] = str(environment / "bin") + os.pathsep + env["PATH"]
    env["VIRTUAL_ENV"] = str(environment)
    results = []
    selected = [name for name, entry in manifest.items() if profile in entry.get("profiles", [])]
    if source:
        selected.remove("source-install")
        selected.insert(0, "source-install")
    for name in selected:
        entry = manifest[name]
        failed = [
            dependency
            for dependency in entry["needs"]
            if not any(row["id"] == dependency and row["status"] == "passed" for row in results)
        ]
        row = {
            "id": name,
            "status": "blocked",
            "seconds": 0,
            "reason": "prerequisite: " + ", ".join(failed),
        }
        results.append(row)
        if not failed:
            for config, config_entry in manifest.items():
                if "file" in config_entry and cwd.exists():
                    (cwd / config_entry["file"]).write_text(blocks[config])
            env["PIP_REPORT"] = str(output / f"{name}-install.json")
            started = time.monotonic()
            log = output / f"{name}.log"
            try:
                code, timed_out = run_shell(
                    blocks[name],
                    workspace if name == "source-install" else cwd,
                    env,
                    log,
                    entry["timeout"],
                )
                row.update(
                    status="timeout" if timed_out else "failed" if code else "passed",
                    returncode=code,
                    reason="",
                )
                if row["status"] == "passed":
                    assert_outputs(entry, cwd, log)
            except (OSError, ValueError, KeyError) as error:
                row.update(status="failed", reason=str(error))
            row["seconds"] = round(time.monotonic() - started, 2)
        print(f"{profile}/{name}: {row['status']} ({row['seconds']}s)", flush=True)
        report = {
            "profile": profile,
            "package_source": package_source,
            "sha": sha,
            "checks": results,
            "expected": selected,
            "complete": len(results) == len(selected),
            "platform": sys.platform,
            "run_attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")),
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if environment.exists():
        run_shell(
            "python --version\npython -m pip freeze --all\npython -m pip check",
            cwd,
            env,
            output / "environment.log",
            60,
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "scripts/readme/readme_commands.json").read_text())
    blocks = read_blocks((ROOT / "README.md").read_text(), manifest)
    if args.validate:
        print(f"Covered all {len(blocks)} README blocks")
        return 0
    if not args.profile or not args.workspace or not args.output:
        parser.error("--profile, --workspace and --output are required for execution")
    report = run_profile(
        args.profile, args.workspace.resolve(), args.output.resolve(), blocks, manifest, args.candidate_manifest
    )
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a") as stream:
            stream.write(f"### README / {args.profile}\n\n")
            stream.write(f"Package source: **{report['package_source']['source']}**.\n\n")
            if report["package_source"]["source"] == "release-candidate":
                stream.write("RC validation only; public-index installation is not checked.\n\n")
            for check in report["checks"]:
                stream.write(f"- `{check['id']}`: {check['status']}\n")
    return int(any(row["status"] != "passed" for row in report["checks"]))


if __name__ == "__main__":
    raise SystemExit(main())
