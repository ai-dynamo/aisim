# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every round must retain the samples underlying its warm-call median."""

import hashlib
import json
import os
import runpy
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_retains_warm_samples_for_every_rotated_round(tmp_path, monkeypatch):
    responses = []

    def worker(command, **kwargs):
        samples = [1.0, 3.0 + len(responses), 9.0]
        response = {
            "build": {"revision": command[-1], "native_sha256": f"fixture-{len(responses)}"},
            "results": {
                "fixture": {
                    "value": 12.0,
                    "cold_us": 42.0,
                    "warm_us": statistics.median(samples),
                    "warm_samples_us": samples,
                }
            },
        }
        responses.append(response)
        return SimpleNamespace(stdout=json.dumps(response))

    output = tmp_path / "evidence.json"
    driver = Path(__file__).with_name("affected_perf.py")
    monkeypatch.setattr(sys, "argv", [str(driver), str(tmp_path), "2", str(output), "base", "head"])
    monkeypatch.setattr("subprocess.run", worker)
    runpy.run_path(str(driver), run_name="__main__")
    result = json.loads(output.read_text())
    assert result["orders"] == [["base", "head"], ["head", "base"]]
    for build, indices in (("base", [0, 3]), ("head", [1, 2])):
        saved = result["results"][build]["fixture"]
        expected = [responses[i]["results"]["fixture"] for i in indices]
        assert result["build_identities"][build] == [responses[i]["build"] for i in indices]
        assert saved["warm_samples_us"] == [row["warm_samples_us"] for row in expected]
        assert saved["warm"] == [statistics.median(samples) for samples in saved["warm_samples_us"]]
        assert saved["cold"] == [row["cold_us"] for row in expected]
        assert saved["value"] == [row["value"] for row in expected]


def test_shim_calls_old_and_new_native_signatures():
    from affected_perf import qualifying_constructor

    def old(name, scale, kernel, phase):
        return name, scale, kernel, phase

    def new(name, scale, kernel, phase, *, state_dtype="unrecorded", token_width=1):
        return state_dtype, token_width

    qualify = {"on": True}
    arguments = ("scan", 1, "selective_state_update", "generation")
    assert qualifying_constructor(old, qualify)(*arguments) == arguments
    wrapped = qualifying_constructor(new, qualify)
    assert wrapped(*arguments) == ("bfloat16", 1)
    assert wrapped(*arguments, state_dtype="float16", token_width=4) == ("float16", 4)
    qualify["on"] = False
    assert wrapped(*arguments) == ("unrecorded", 1)


def _build_fixture(root, label):
    """A tiny committed checkout and installed SDK; no timing/model dependencies."""
    src = root / label / "src"
    packages = src / "python/aisim/src"
    for name in ("aisim", "aisim_core"):
        (packages / name).mkdir(parents=True)
        (packages / name / "__init__.py").write_text(f"BUILD = {label!r}\n")
    inputs = ("Cargo.toml", "Cargo.lock", "crates/core/Cargo.toml", "crates/core/build.rs", "crates/core/src/lib.rs")
    native = {}
    for name in inputs:
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture {label}: {name}\n")
        native[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    # A synthetic Python module stands in for the extension so this test can
    # exercise its source record without compiling a second native library.
    runtime = packages / "aisim/_runtime.py"
    runtime.write_text(f"def _build_source_files_json(): return {json.dumps(native)!r}\n")
    subprocess.run(["git", "init", "-q", str(src)], check=True)
    subprocess.run(["git", "-C", str(src), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(src),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-q",
            "-m",
            "test fixture",
        ],
        check=True,
    )
    return src, packages, runtime


def _verify_process(src, packages):
    driver = Path(__file__).parent
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from affected_perf import verified_worker_build; import json, sys; "
            "print(json.dumps(verified_worker_build(sys.argv[1])))",
            str(src),
        ],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, (driver, packages)))},
        capture_output=True,
        text=True,
    )


def test_worker_verifies_and_records_selected_packages_and_native_build(tmp_path):
    src, packages, runtime = _build_fixture(tmp_path, "base")
    proc = _verify_process(src, packages)
    assert proc.returncode == 0, proc.stderr
    identity = json.loads(proc.stdout)
    assert identity["package_paths"] == {name: str(packages / name) for name in ("aisim", "aisim_core")}
    assert identity["native_path"] == str(runtime)
    assert identity["native_sha256"] == hashlib.sha256(runtime.read_bytes()).hexdigest()
    assert identity["python_sources_sha256"] and identity["native_sources_sha256"]
    assert len(identity["revision"]) == 40


def test_candidate_pythonpath_cannot_be_recorded_as_baseline(tmp_path):
    baseline, _, _ = _build_fixture(tmp_path, "base")
    _, candidate, _ = _build_fixture(tmp_path, "candidate")
    # A real subprocess inherits only the candidate SDK. It must fail before
    # importing any model or timing helper, rather than label it as baseline.
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("affected_perf.py")), "--worker", str(baseline)],
        env={**os.environ, "PYTHONPATH": str(candidate)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "imported packages do not match" in proc.stderr


@pytest.mark.parametrize("mismatch", ["path", "sources"])
def test_worker_rejects_wrong_native_extension(tmp_path, mismatch):
    src, packages, runtime = _build_fixture(tmp_path, "base")
    if mismatch == "sources":
        # Different native build inputs despite an otherwise matching SDK.
        (src / "crates/core/src/lib.rs").write_text("different native build\n")
        subprocess.run(["git", "-C", str(src), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(src),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-q",
                "-m",
                "change native fixture",
            ],
            check=True,
        )
    else:
        outside = tmp_path / "foreign_runtime.py"
        outside.write_bytes(runtime.read_bytes())
        runtime.unlink()
        runtime.symlink_to(outside)
    proc = _verify_process(src, packages)
    assert proc.returncode != 0
    assert ("outside selected build" if mismatch == "path" else "imported native extension") in proc.stderr
