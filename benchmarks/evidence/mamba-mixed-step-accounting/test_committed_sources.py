# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An extra source file, untracked or ignored, must fail the revision check."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from committed_sources import require_committed, verified_model_configs


def _repo(tmp_path):
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "a.py").write_text("A = 1\n")
    (tmp_path / ".gitignore").write_text("pkg/ignored.py\n")
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "init"], check=True)
    return package


@pytest.mark.parametrize("name", ["untracked.py", "ignored.py"])
@pytest.mark.parametrize("link", [False, True])
def test_extra_source_is_rejected(tmp_path, name, link):
    package = _repo(tmp_path)
    require_committed(tmp_path, "HEAD", list(package.rglob("*.py")))
    if link:
        # A link to a committed file resolves to a committed path but is not one.
        (package / name).symlink_to("a.py")
    else:
        (package / name).write_text("B = 2\n")
    with pytest.raises(ValueError, match=f"pkg/{name}"):
        require_committed(tmp_path, "HEAD", list(package.rglob("*.py")))


@pytest.mark.parametrize("override", [False, True])
def test_resolved_model_config_matches_revision(tmp_path, monkeypatch, override):
    from aisim_core.sdk import utils

    _repo(tmp_path)
    model = "Qwen/Qwen3-0.6B"
    name = "Qwen--Qwen3-0.6B_config.json"
    core = tmp_path / "python/aisim/src/aisim_core"
    configs = core / "model_configs"
    configs.mkdir(parents=True)
    content = json.dumps({"num_hidden_layers": 28}).encode()
    (configs / name).write_bytes(content)
    subprocess.run(["git", "-C", str(tmp_path), "add", str(configs)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "commit",
            "-qm",
            "model config",
        ],
        check=True,
    )
    installed = tmp_path / "installed/aisim_core"
    resolved = tmp_path / "override" if override else installed / "model_configs"
    resolved.mkdir(parents=True)
    (resolved / name).write_bytes(content)
    monkeypatch.setattr(utils.pkg_resources, "files", lambda package: installed)
    monkeypatch.delenv("AICONFIGURATOR_MODEL_CONFIGS_PATH", raising=False)
    if override:
        monkeypatch.setenv("AICONFIGURATOR_MODEL_CONFIGS_PATH", str(resolved))
    evidence = verified_model_configs(tmp_path, "HEAD", model)
    assert evidence["override"] == (str(resolved) if override else None)
    assert evidence["files"][name]["resolved_path"] == str(resolved / name)
    assert evidence["files"][name]["sha256"] == hashlib.sha256(content).hexdigest()
    # The Python and native sources are unchanged, but the dispatch layer
    # count would now differ. Reject both installed and override mismatches.
    (resolved / name).write_text(json.dumps({"num_hidden_layers": 29}))
    with pytest.raises(ValueError, match="resolved model configuration .* does not match"):
        verified_model_configs(tmp_path, "HEAD", model)
    (resolved / name).write_bytes(content)
    assert verified_model_configs(tmp_path, "HEAD", model) == evidence
    # An unrecorded optional input must not be merged into get_model either.
    (resolved / "Qwen--Qwen3-0.6B_hf_quant_config.json").write_text("{}")
    with pytest.raises(ValueError, match="model configuration .* is absent"):
        verified_model_configs(tmp_path, "HEAD", model)
