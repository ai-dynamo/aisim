# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static trust-model checks for the public PR-validation workflow."""

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOW = Path(__file__).resolve().parents[5] / ".github/workflows/pr-validation.yml"


@pytest.fixture(scope="module")
def text():
    if not WORKFLOW.is_file():
        pytest.skip("workflow not present in this checkout")
    return WORKFLOW.read_text()


def test_dispatch_only_and_read_only(text):
    doc = yaml.safe_load(text)
    triggers = doc.get("on", doc.get(True))
    assert set(triggers) == {"workflow_dispatch"}
    assert doc["permissions"] == {"contents": "read"}
    assert doc["jobs"]["validate"]["permissions"] == {"contents": "read", "pull-requests": "read"}


def test_no_secrets_or_token_leaks(text):
    assert "secrets." not in text and "HF_TOKEN" not in text
    assert text.count("github.token") == 3  # API steps only, never the evaluation step


def test_actions_pinned_to_commit(text):
    uses = re.findall(r"uses:\s*(\S+)", text)
    assert uses and all(re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", u) for u in uses)


def test_pr_code_is_never_checked_out_or_interpolated(text):
    assert "pull_request_target" not in text and "head.ref" not in text
    assert "ref:" not in text.replace("?ref=", "")  # checkout stays on the default branch
    _check_run_inputs(yaml.safe_load(text))
    assert "persist-credentials: false" in text


def _check_run_inputs(doc):
    for job in doc["jobs"].values():
        for step in job.get("steps", []):
            assert not re.search(r"\$\{\{\s*inputs(?:\.|\[)", step.get("run", ""))


def test_echo_input_is_rejected_while_env_is_allowed(text):
    doc = yaml.safe_load(text)
    doc["jobs"]["validate"]["steps"].append({"run": 'echo "${{ inputs.request_path }}"'})
    with pytest.raises(AssertionError):
        _check_run_inputs(doc)
    doc["jobs"]["validate"]["steps"][-1] = {
        "run": 'echo "$REQUEST_PATH"',
        "env": {"REQUEST_PATH": "${{ inputs.request_path }}"},
    }
    _check_run_inputs(doc)


@pytest.mark.parametrize(
    "role,permission,admitted", [("admin", "admin", True), ("maintain", "write", True), ("write", "write", False)]
)
def test_maintainer_roles(text, tmp_path, role, permission, admitted):
    doc = yaml.safe_load(text)
    script = doc["jobs"]["validate"]["steps"][0]["run"]
    gh = tmp_path / "gh"
    gh.write_text(
        '#!/bin/sh\nfor arg; do field="$arg"; done\n'
        'case "$field" in\n.role_name) echo "$FIXTURE_ROLE";;\n'
        '.permission) echo "$FIXTURE_PERMISSION";;\n*) exit 2;;\nesac\n'
    )
    gh.chmod(0o755)
    result = subprocess.run(
        ["bash", "-e", "-c", script],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "GH_REPO": "ai-dynamo/aisim",
            "GITHUB_ACTOR": "fixture",
            "FIXTURE_ROLE": role,
            "FIXTURE_PERMISSION": permission,
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == admitted, result.stdout + result.stderr
