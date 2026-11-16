# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time

import psutil
import pytest

from scripts.readme.check_readme_commands import ROOT, assert_outputs, read_blocks, run_shell


def test_every_readme_command_is_registered():
    manifest = json.loads((ROOT / "scripts/readme/readme_commands.json").read_text())
    readme = (ROOT / "README.md").read_text()
    assert len(read_blocks(readme, manifest)) == len(manifest)
    with pytest.raises(ValueError, match="needs a readme-check"):
        read_blocks(readme + "\n```bash\nunregistered\n```\n", manifest)
    with pytest.raises(ValueError, match="Manifest drift"):
        read_blocks(readme, {**manifest, "stale": {}})


def test_no_requests_is_failure_even_with_valid_json(tmp_path):
    (tmp_path / "prediction.json").write_text('{"completed_requests": 0}')
    with pytest.raises(ValueError, match="no requests"):
        assert_outputs({"prediction": "prediction.json"}, tmp_path, tmp_path / "log")


def test_unresolved_recommendation_is_failure(tmp_path):
    (tmp_path / "0001.yaml").write_text("engine:\n  backend:\n    choices: [vllm, sglang]\n")
    with pytest.raises(ValueError, match="Unresolved"):
        assert_outputs({"recommendations": "."}, tmp_path, tmp_path / "log")


def test_nonzero_and_timeout_are_distinct(tmp_path):
    assert run_shell("exit 7", tmp_path, dict(os.environ), tmp_path / "failed", 10) == (
        7,
        False,
    )
    code, timeout = run_shell("sleep 20", tmp_path, dict(os.environ), tmp_path / "timeout", 0.3)
    assert timeout and code != 0


def test_timeout_kills_workers_in_new_sessions(tmp_path):
    command = (
        "python3 -c 'import subprocess,time; "
        'p=subprocess.Popen(["sleep","60"], start_new_session=True); '
        'open("pid","w").write(str(p.pid)); time.sleep(60)' + "'"
    )
    _, timeout = run_shell(command, tmp_path, dict(os.environ), tmp_path / "log", 1)
    assert timeout
    pid = int((tmp_path / "pid").read_text())
    deadline = time.monotonic() + 3
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
            break
        time.sleep(0.1)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_request_capture_must_match_override(tmp_path):
    (tmp_path / "requests.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="incomplete"):
        assert_outputs({"requests": "requests.jsonl", "completed_requests": 4}, tmp_path, tmp_path / "log")


@pytest.mark.parametrize("actual", [2, 1, None])
def test_prediction_tensor_override_metric(tmp_path, actual):
    # ReplayReport serializes this metric in crates/core/src/replay/report.rs.
    summary = {"completed_requests": 100}
    if actual is not None:
        summary["decode_gpus_per_worker"] = actual
    (tmp_path / "prediction.json").write_text(json.dumps({"summary": summary}))
    entry = {"prediction": "prediction.json", "metrics": {"decode_gpus_per_worker": 2}}
    if actual == 2:
        assert_outputs(entry, tmp_path, tmp_path / "log")
    else:
        with pytest.raises(ValueError, match="Unexpected decode_gpus_per_worker"):
            assert_outputs(entry, tmp_path, tmp_path / "log")


def rc_manifest(tmp_path):
    import hashlib
    import zipfile

    packages = {"aisim": "0.14.0", "ai-dynamo": "1.7.0", "ai-dynamo-runtime": "1.7.0"}
    wheels = []
    for name, version in packages.items():
        normalized = name.replace("-", "_")
        path = tmp_path / f"{normalized}-{version}-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"{normalized}-{version}.dist-info/METADATA", f"Name: {name}\nVersion: {version}\n")
        wheels.append(
            {
                "filename": path.name,
                "name": name,
                "version": version,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    candidate = {"source": "release-candidate", "wheelhouse": str(tmp_path), "packages": packages, "wheels": wheels}
    manifest = tmp_path / "candidate.json"
    manifest.write_text(json.dumps(candidate))
    return manifest, candidate


def test_rc_wheels_constrain_exact_artifacts_and_retain_provenance(tmp_path):
    from scripts.readme.check_readme_commands import candidate_environment

    manifest, candidate = rc_manifest(tmp_path)
    env = {}
    assert candidate_environment(manifest, tmp_path, env) == candidate
    constraints = (tmp_path / "candidate-constraints.txt").read_text()
    for wheel in candidate["wheels"]:
        assert f"{wheel['name']} @ {(tmp_path / wheel['filename']).as_uri()}" in constraints
    assert env["PIP_CONSTRAINT"] == str(tmp_path / "candidate-constraints.txt")


@pytest.mark.parametrize("damage", ["hash", "version", "missing-runtime", "duplicate", "path"])
def test_invalid_rc_artifacts_fail_before_install(tmp_path, damage):
    from scripts.readme.check_readme_commands import candidate_environment

    manifest, candidate = rc_manifest(tmp_path)
    if damage == "hash":
        candidate["wheels"][0]["sha256"] = "0" * 64
    elif damage == "version":
        candidate["packages"]["aisim"] = "0.15.0"
    elif damage == "missing-runtime":
        candidate["wheels"].pop()
    elif damage == "duplicate":
        candidate["wheels"].append(candidate["wheels"][0])
    else:
        candidate["wheels"][0]["filename"] = "../outside.whl"
    manifest.write_text(json.dumps(candidate))
    with pytest.raises(ValueError):
        candidate_environment(manifest, tmp_path, {})


def test_published_release_does_not_substitute_rc_wheels(tmp_path):
    from scripts.readme.check_readme_commands import candidate_environment

    manifest = tmp_path / "candidate.json"
    manifest.write_text(json.dumps({"source": "public-index", "wheels": []}))
    env = {}
    assert candidate_environment(manifest, tmp_path, env)["source"] == "public-index"
    assert env == {}
    manifest.write_text(json.dumps({"source": "public-index", "wheels": [{}]}))
    with pytest.raises(ValueError, match="must not substitute"):
        candidate_environment(manifest, tmp_path, env)


def test_rc_install_failure_is_still_a_failure(tmp_path, monkeypatch):
    from scripts.readme import check_readme_commands as harness

    candidate_path, candidate = rc_manifest(tmp_path)
    monkeypatch.setattr(harness.venv, "create", lambda *args, **kwargs: None)
    observed = []

    def fail_install(command, cwd, env, log, timeout):
        observed.append(env.get("PIP_CONSTRAINT"))
        return 1, False

    monkeypatch.setattr(harness, "run_shell", fail_install)
    report = harness.run_profile(
        "dynamo",
        tmp_path / "workspace",
        tmp_path / "results",
        {"install": "failing install"},
        {"install": {"profiles": ["dynamo"], "needs": [], "timeout": 10}},
        candidate_path,
    )
    assert report["checks"][0]["status"] == "failed"
    assert report["package_source"] == candidate
    assert observed and observed[0]
