# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A checkout hash must never stand in for a different imported backend."""

import hashlib
import importlib.util
import subprocess
import sys

import pytest
from xpu_scoped_repair import verified_candidate_hash


def test_fingerprint_binds_imported_source(tmp_path, monkeypatch):
    installed = tmp_path / "installed_backend.py"
    checkout = tmp_path / "checkout_backend.py"
    installed.write_text("class Backend:\n    version = 'old'\n")
    checkout.write_text("class Backend:\n    version = 'new'\n")
    spec = importlib.util.spec_from_file_location("fixture_backend", installed)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    with pytest.raises(
        ValueError,
        match="imported candidate backend differs from checkout: .*check out .*reinstall its SDK",
    ):
        verified_candidate_hash(module.Backend, checkout)
    # An installed copy with identical content is valid; its imported bytes
    # are what the accepted fingerprint describes.
    checkout.write_bytes(installed.read_bytes())
    assert verified_candidate_hash(module.Backend, checkout) == hashlib.sha256(installed.read_bytes()).hexdigest()


def test_fingerprint_binds_pinned_revision(tmp_path, monkeypatch):
    source = tmp_path / "backend.py"
    source.write_text("class Backend:\n    pass\n")
    spec = importlib.util.spec_from_file_location("pinned_backend", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    calls = []

    def git_show(command):
        calls.append(command)
        return source.read_bytes()

    monkeypatch.setattr(subprocess, "check_output", git_show)
    assert (
        verified_candidate_hash(module.Backend, "repo/backend.py", revision="abc123")
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert calls == [["git", "show", "abc123:repo/backend.py"]]
    monkeypatch.setattr(subprocess, "check_output", lambda *_: b"different revision")
    with pytest.raises(ValueError, match="imported candidate backend differs"):
        verified_candidate_hash(module.Backend, "repo/backend.py", revision="abc123")
