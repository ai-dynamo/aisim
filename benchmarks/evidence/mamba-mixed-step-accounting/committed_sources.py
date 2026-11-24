# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject source files that a recorded Git revision does not contain."""

import hashlib
import os
import subprocess
from pathlib import Path


def require_committed(repo, revision, paths):
    """Raise unless every path in `paths` is a file of `revision`'s tree.

    `git diff HEAD` ignores untracked and ignored files, so a hashed source
    inventory can hold files the revision does not. Paths are absolute or
    relative to `repo`. Each path's own name is checked, without following
    symlinks: an uncommitted link to a committed file is still uncommitted.
    """
    repo = os.path.abspath(repo)
    relative = sorted(
        {Path(os.path.relpath(os.path.abspath(os.path.join(repo, path)), repo)).as_posix() for path in paths}
    )
    tree = subprocess.run(
        [
            "git",
            "-C",
            repo,
            "ls-tree",
            "-r",
            "--name-only",
            "-z",
            revision,
            "--",
            *relative,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    missing = sorted(set(relative) - set(filter(None, tree.split("\0"))))
    if missing:
        raise ValueError(f"source files absent from {revision}: {', '.join(missing[:5])}")


def verified_model_configs(repo, revision, model_path):
    """Fingerprint the actual bundled/overridden JSON inputs used by get_model.

    Compare against Git blobs, including absence of the optional quant config,
    so neither an installed copy nor an external config directory can silently
    change the measured workload.
    """
    from aisim_core.sdk.utils import _get_model_config_path

    # Model loading prefers a local directory over bundled or overridden JSON.
    # This benchmark only certifies the latter against the recorded Git tree.
    if Path(model_path).is_dir():
        raise ValueError(f"local model directory {model_path} cannot be certified as bundled inputs")
    resolved_root = _get_model_config_path()
    prefix = "python/aisim/src/aisim_core/model_configs/"
    names = [f"{model_path.replace('/', '--')}_{suffix}.json" for suffix in ("config", "hf_quant_config")]
    tracked = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "ls-tree",
            "-r",
            "--name-only",
            revision,
            "--",
            *[prefix + n for n in names],
        ],
        text=True,
    ).splitlines()
    files = {}
    for name in names:
        relative = prefix + name
        resolved = resolved_root / name
        if relative not in tracked:
            if name == names[0] or resolved.exists():
                raise ValueError(f"model configuration {relative} is absent from {revision}")
            files[name] = None
            continue
        committed = subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{relative}"])
        if not resolved.is_file() or resolved.read_bytes() != committed:
            raise ValueError(f"resolved model configuration {resolved} does not match {revision}")
        files[name] = {
            "path": relative,
            "resolved_path": str(resolved),
            "sha256": hashlib.sha256(committed).hexdigest(),
        }
    return {
        "override": os.environ.get("AICONFIGURATOR_MODEL_CONFIGS_PATH"),
        "files": files,
    }


def package_inventory(repo, revision):
    """Hash committed Python package bytes, including the complete file set."""
    import io
    import tarfile

    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--verify", f"{revision}^{{commit}}"],
        text=True,
    ).strip()
    prefix = "python/aisim/src/"
    archive = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "archive",
            revision,
            prefix + "aisim",
            prefix + "aisim_core",
        ]
    )
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        for member in tree:
            if member.isfile() and member.name.endswith(".py"):
                files[member.name.removeprefix(prefix)] = hashlib.sha256(tree.extractfile(member).read()).hexdigest()
    if not files:
        raise ValueError(f"no package sources at {revision}")
    return {"revision": revision, "files": files}


def verify_imported_packages(inventory, modules):
    """Reject edited, extra or missing Python sources in either loaded package."""
    loaded = {}
    roots = {}
    for module in modules:
        root = Path(module.__file__).resolve().parent
        roots[module.__name__] = str(root)
        for path in root.rglob("*.py"):
            name = module.__name__ + "/" + path.relative_to(root).as_posix()
            loaded[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if loaded != inventory["files"]:
        raise ValueError(f"imported packages do not match {inventory['revision']}")
    return {
        "revision": inventory["revision"],
        "package_paths": roots,
        "python_sources_sha256": hashlib.sha256(json_bytes(loaded)).hexdigest(),
    }


def json_bytes(value):
    import json

    return json.dumps(value, sort_keys=True).encode()
