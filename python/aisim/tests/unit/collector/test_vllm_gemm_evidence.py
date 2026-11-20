# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check runner provenance using CPU-only campaign workers and recorded timings."""

import hashlib
import importlib.util
import io
import json
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[5]
CAPSULE = ROOT / "benchmarks/evidence/vllm-fp8-block-rotation-pr613-20261027"


def load_module(name):
    spec = importlib.util.spec_from_file_location(f"pr613_{name}", CAPSULE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fresh_campaign(tmp_path, monkeypatch):
    runner = load_module("run")
    protocol = json.loads((CAPSULE / "protocol.json").read_text())
    output = tmp_path / "fresh"
    executed_hashes = []

    def command(*args, **kwargs):
        if args[:2] == ("git", "rev-parse"):
            return args[2]
        if args[:2] == ("git", "diff"):
            return f"{runner.COLLECTOR}/vllm/collect_gemm.py"
        assert args[0] == "nvidia-smi"
        return ""

    def worker(child, **kwargs):
        # Stand in only for the GPU process. Require campaign provenance to
        # exist before any worker runs, without copying the bundled archive.
        executed_source = Path(child[1]).read_bytes()
        with tarfile.open(output / "runner-source.tar.gz") as archive:
            assert archive.getnames() == ["run.py"]
            assert archive.extractfile("run.py").read() == executed_source
        runner_hash = hashlib.sha256(executed_source).hexdigest()
        assert json.loads((output / "invocation.json").read_text())["runner_sha256"] == runner_hash
        executed_hashes.append(runner_hash)
        arm = child[child.index("--worker") + 1]
        replicate = child[child.index("--replicate") + 1]
        prefix = f"{arm}-{replicate}"
        environment = json.loads((CAPSULE / f"direct/{prefix}-environment.json").read_text())
        environment["runner_sha256"] = runner_hash
        (output / f"{prefix}-environment.json").write_text(json.dumps(environment))
        shutil.copyfile(CAPSULE / f"direct/{prefix}-measurements.jsonl", output / f"{prefix}-measurements.jsonl")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner, "command", command)
    monkeypatch.setattr(runner, "gpu_status", lambda: "CPU-only test worker")
    monkeypatch.setattr(runner.subprocess, "run", worker)
    runner.campaign(SimpleNamespace(output=output, repo=ROOT), protocol)
    assert len(executed_hashes) == 2 * protocol["replicates"]
    assert len(set(executed_hashes)) == 1
    return output


def test_fresh_campaign_can_be_analyzed(fresh_campaign):
    result = load_module("analyze").direct(fresh_campaign)
    assert result["measurement_count"] == 504
    assert result["excluded_measurements"] == 0
    assert result["flagged_outliers"] == []


@pytest.mark.parametrize("update_environment", [False, True], ids=["hash-mismatch", "behavior-mismatch"])
def test_fresh_campaign_rejects_modified_runner(fresh_campaign, update_environment):
    with tarfile.open(fresh_campaign / "runner-source.tar.gz") as archive:
        source = archive.extractfile("run.py").read()
    modified = source.replace(b'result["num_runs_executed"] != 6', b'result["num_runs_executed"] != 1')
    assert modified != source
    source = modified
    with tarfile.open(fresh_campaign / "runner-source.tar.gz", "w:gz") as archive:
        member = tarfile.TarInfo("run.py")
        member.size = len(source)
        archive.addfile(member, io.BytesIO(source))
    if update_environment:
        for path in fresh_campaign.glob("*-environment.json"):
            environment = json.loads(path.read_text())
            environment["runner_sha256"] = hashlib.sha256(source).hexdigest()
            path.write_text(json.dumps(environment))
    message = "Packaged runner behavior differs" if update_environment else "Runner snapshot mismatch"
    with pytest.raises(ValueError, match=message):
        load_module("analyze").direct(fresh_campaign)


def test_archived_campaign_remains_readable():
    result = load_module("analyze").direct(CAPSULE / "direct")
    assert result["measurement_count"] == 504
    assert result["excluded_measurements"] == 0
