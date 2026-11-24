# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject extra sources and model inputs that differ from the recorded revision."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from committed_sources import require_committed, verified_model_configs


def _repo(tmp_path):
    git = [
        "git",
        "-C",
        str(tmp_path),
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.com",
    ]
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


@pytest.mark.parametrize("override", [False, True])
def test_local_model_directory_cannot_certify_bundled_inputs(tmp_path, monkeypatch, override):
    from aisim_core.sdk import utils

    repo = Path(__file__).resolve().parents[3]
    core = repo / "python/aisim/src/aisim_core"
    model = "Qwen/Qwen3-0.6B"
    monkeypatch.setattr(utils.pkg_resources, "files", lambda package: core)
    monkeypatch.delenv("AICONFIGURATOR_MODEL_CONFIGS_PATH", raising=False)
    if override:
        monkeypatch.setenv("AICONFIGURATOR_MODEL_CONFIGS_PATH", str(core / "model_configs"))
    monkeypatch.chdir(tmp_path)
    # The bundled inputs match HEAD until a local directory shadows the ID.
    evidence = verified_model_configs(repo, "HEAD", model)
    config_path = evidence["files"]["Qwen--Qwen3-0.6B_config.json"]["resolved_path"]
    local_config = json.loads(Path(config_path).read_text())
    local_config["num_hidden_layers"] += 1
    local = tmp_path / model
    local.mkdir(parents=True)
    (local / "config.json").write_text(json.dumps(local_config))
    # Bypass the loader's cache, as in a fresh benchmark process. This valid
    # config takes precedence over both installed and overridden bundled JSON.
    loaded = utils._load_model_config_from_model_path.__wrapped__(model)
    assert loaded["num_hidden_layers"] == local_config["num_hidden_layers"]
    assert utils._parse_hf_config_json(loaded)["layers"] == local_config["num_hidden_layers"]
    with pytest.raises(ValueError, match="local model directory"):
        verified_model_configs(repo, "HEAD", model)


@pytest.mark.parametrize("edit", ["changed", "extra", "missing"])
def test_package_inventory_rejects_different_loaded_sources(tmp_path, edit):
    from types import SimpleNamespace

    from committed_sources import package_inventory, verify_imported_packages

    git = [
        "git",
        "-C",
        str(tmp_path),
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.com",
    ]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    modules = []
    for package in ("aisim", "aisim_core"):
        root = tmp_path / "python/aisim/src" / package
        root.mkdir(parents=True)
        (root / "__init__.py").write_text("VERSION = 1\n")
        modules.append(SimpleNamespace(__name__=package, __file__=str(root / "__init__.py")))
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "fixture"], check=True)
    inventory = package_inventory(tmp_path, "HEAD")
    verified = verify_imported_packages(inventory, modules)
    assert len(verified["revision"]) == 40
    assert verified["revision"] == inventory["revision"]
    root = Path(modules[0].__file__).parent
    if edit == "changed":
        (root / "__init__.py").write_text("VERSION = 2\n")
    elif edit == "extra":
        (root / "extra.py").write_text("EXTRA = True\n")
    else:
        (root / "__init__.py").unlink()
    with pytest.raises(ValueError, match="imported packages do not match"):
        verify_imported_packages(inventory, modules)
