# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The evidence driver must retain every child process's qualification record."""

import json
import os
import runpy
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest


def import_paths(sdk):
    return {
        "package_path": str(sdk / "aisim" / "__init__.py"),
        "constructor_path": str(sdk / "aisim_core" / "sdk" / "models" / "nemotron_h.py"),
        "native_path": str(sdk / "aisim" / "_runtime.py"),
    }


def test_multiple_process_records_remain_linked_to_each_prediction(tmp_path, monkeypatch):
    records = [
        {"calls": 4, "native_sha256": "fixture-binary", "native_sources_sha256": "fixture-source"},
        {"calls": 0, "native_sha256": "fixture-binary", "native_sources_sha256": "fixture-source"},
    ]
    baseline = tmp_path / "baseline"
    sdks = {"baseline": baseline, "candidate": tmp_path / "candidate"}

    def cli(command, *, env, **kwargs):
        sdk = baseline if str(baseline) in env["PYTHONPATH"].split(os.pathsep) else sdks["candidate"]
        dest = Path(command[command.index("--output-dir") + 1])
        dest.mkdir()
        report = {
            "completed_requests": 1000,
            "total_input_tokens": 2_048_000,
            "total_output_tokens": 32_000,
            "mean_ttft_ms": 2,
            "mean_tpot_ms": 1,
            "duration_ms": 100,
            "wall_time_ms": 10,
        }
        (dest / "prediction.json").write_text(json.dumps(report))
        (dest / "resource-runtime.json").write_text(json.dumps({"peak_observed_rss_bytes": 1024}))
        Path(env["REPAIR_BUILD_RECORD"]).write_text(
            "".join(json.dumps(record | import_paths(sdk)) + "\n" for record in records)
        )
        return SimpleNamespace(returncode=0)

    output = tmp_path / "evidence.json"
    driver = Path(__file__).with_name("repair_replay_perf.py")
    monkeypatch.setattr(sys, "argv", [str(driver), sys.executable, str(baseline), "2", str(output)])
    monkeypatch.setattr("subprocess.run", cli)
    runpy.run_path(str(driver), run_name="__main__")
    result = json.loads(output.read_text())
    assert len(result["rounds"]) == 2
    for round_result in result["rounds"]:
        for build in ("baseline", "candidate"):
            for case in ("nano", "control"):
                assert round_result["results"][build][case]["build"] == [
                    record | import_paths(sdks[build]) for record in records
                ]


@pytest.fixture
def installed_sdks(tmp_path):
    """Install tiny SDK fixtures; real interpreters exercise Python's import fallback."""
    candidate = tmp_path / "candidate"
    venv.EnvBuilder(with_pip=False).create(candidate)
    python = candidate / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    sdk = Path(
        subprocess.check_output(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True
        ).strip()
    )
    baseline = tmp_path / "baseline"
    for label, packages in (("baseline", baseline), ("candidate", sdk)):
        for package in ("aisim", "aisim_core", "aisim_core/sdk", "aisim_core/sdk/models"):
            path = packages / package
            path.mkdir(parents=True, exist_ok=True)
            (path / "__init__.py").write_text("")
        (packages / "aisim" / "_runtime.py").write_text(f"def _build_source_files_json(): return {label!r}\n")
        (packages / "aisim_core" / "sdk" / "models" / "nemotron_h.py").write_text(
            "from types import SimpleNamespace\nops = SimpleNamespace(Mamba2Kernel=lambda **kwargs: None)\n"
        )
        # Synthetic CLI output isolates provenance validation from numerical estimation.
        (packages / "aisim" / "__main__.py").write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "from aisim_core.sdk.models import nemotron_h as nh\n"
            "nh.ops.Mamba2Kernel()\n"
            "dest = Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
            "dest.mkdir()\n"
            "report = dict(completed_requests=1000, total_input_tokens=2048000, total_output_tokens=32000,\n"
            "              mean_ttft_ms=2, mean_tpot_ms=1, duration_ms=100, wall_time_ms=10)\n"
            "(dest / 'prediction.json').write_text(json.dumps(report))\n"
            "(dest / 'resource-runtime.json').write_text(json.dumps(dict(peak_observed_rss_bytes=1024)))\n"
        )
    return python, baseline, sdk


def run_driver(tmp_path, python, baseline):
    output = tmp_path / "evidence.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("repair_replay_perf.py")),
            str(python),
            str(baseline),
            "2",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    return proc, output


def test_accepts_installed_baseline_and_records_child_import_paths(tmp_path, installed_sdks):
    python, baseline, candidate = installed_sdks
    proc, output = run_driver(tmp_path, python, baseline)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(output.read_text())
    assert [r["order"] for r in result["rounds"]] == [["baseline", "candidate"], ["candidate", "baseline"]]
    for record in result["rounds"]:
        for build, sdk in (("baseline", baseline), ("candidate", candidate)):
            for case in ("nano", "control"):
                identities = record["results"][build][case]["build"]
                assert identities
                for identity in identities:
                    assert {key: identity[key] for key in import_paths(sdk)} == import_paths(sdk)


@pytest.mark.parametrize(
    ("selection", "error"),
    [
        ("empty_directory", "package_path imported outside BASE_SDK/aisim"),
        ("empty_argument", "BASE_SDK must name"),
        ("venv_root", "package_path imported outside BASE_SDK/aisim"),
        ("mixed_packages", "constructor_path imported outside BASE_SDK/aisim_core"),
        ("candidate", "package_path resolves to the candidate installation"),
    ],
)
def test_rejects_invalid_baseline_even_when_candidate_can_run(tmp_path, installed_sdks, selection, error):
    python, baseline, candidate = installed_sdks
    if selection == "empty_directory":
        baseline = tmp_path / "empty"
        baseline.mkdir()
    elif selection == "empty_argument":
        baseline = ""
    elif selection == "venv_root":
        baseline = tmp_path / "candidate"
    elif selection == "mixed_packages":
        # The baseline CLI/native package exists but its model package falls back.
        baseline = tmp_path / "mixed"
        baseline.mkdir()
        shutil.copytree(installed_sdks[1] / "aisim", baseline / "aisim")
    else:
        baseline = candidate
    proc, output = run_driver(tmp_path, python, baseline)
    assert proc.returncode != 0, "accepted a mislabeled candidate as the baseline"
    assert error in proc.stderr
    assert not output.exists()
