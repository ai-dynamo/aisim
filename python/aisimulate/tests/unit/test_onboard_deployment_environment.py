# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment startup inputs survive the public collection command."""

import json
import subprocess
import traceback

import pytest

from aisimulate.support.fpm import fpm_cli_args
from aisimulate.support.schema import FPMDeployment, SupportRequest

pytestmark = pytest.mark.unit


def test_environment_reaches_formal_generator_arguments(tmp_path):
    request = SupportRequest.model_validate(
        {
            "identity": {
                "model": "fixture/model",
                "model_revision": "a" * 40,
                "model_kind": "moe",
                "framework": "vllm",
                "framework_version": "0.28.0",
                "gpu": "gb300",
                "interconnect": "nvlink",
            },
        }
    )
    env = [{"name": "PYTHONPATH", "value": "/runtime/dynamo:/runtime/wheel"}]
    deployment = FPMDeployment(executor="slurm", image="fixture.sqsh", extra_env=env)
    command = fpm_cli_args(request, output_dir=tmp_path, plan_only=True, deployment=deployment)
    assert "K8sConfig.extra_env=" + json.dumps(env) in command
    assert "--extra-env" not in command
    assert "extra_env" not in FPMDeployment().model_dump(exclude_none=True)


@pytest.mark.parametrize(
    "env",
    [
        [{"name": "BAD-NAME", "value": "x"}],
        [{"name": "PYTHONPATH", "value": "x"}, {"name": "PYTHONPATH", "value": "y"}],
        [{"name": "PYTHONPATH", "value": "x", "unknown": "x"}],
        [{"name": "PYTHONPATH", "value": "x\x00y"}],
    ],
)
def test_deployment_rejects_ambiguous_environment(env):
    with pytest.raises(ValueError, match="extra_env"):
        FPMDeployment(extra_env=env)


def test_preview_redacts_environment_credentials_and_keeps_execution_arguments(tmp_path, capsys):
    from aisimulate.support.fpm import run_fpm

    request = SupportRequest.model_validate(
        {
            "identity": {
                "model": "fixture/model",
                "model_revision": "a" * 40,
                "model_kind": "moe",
                "framework": "vllm",
                "framework_version": "0.28.0",
                "gpu": "gb300",
                "interconnect": "nvlink",
            }
        }
    )
    token = "dummy-private-token-746"
    env = [{"name": "HF_TOKEN", "value": token}, {"name": "CUSTOM_AUTH", "value": token}]
    deployment = FPMDeployment(image="fixture:image", extra_env=env)
    assert run_fpm(request, output_dir=tmp_path / "preview", deployment=deployment) == 0
    output = capsys.readouterr().out
    assert token not in output
    assert "<redacted>" in output
    assert "K8sConfig.extra_env=" + json.dumps(env) in fpm_cli_args(
        request, output_dir=tmp_path, plan_only=False, deployment=deployment
    )


@pytest.mark.parametrize("executor", ["slurm", "kubernetes"])
@pytest.mark.parametrize("failure", ["timeout", "stderr", "bytes_stderr", "oserror"])
@pytest.mark.parametrize("token", ["dummy-private-token-746", 'private-"quoted"-\\path\n$secret'])
def test_runtime_probe_failures_redact_environment_and_exception_chain(monkeypatch, executor, failure, token):
    from aisimulate.support import versioning

    deployment = FPMDeployment(
        executor=executor,
        image="fixture:image",
        extra_env=[{"name": "HF_TOKEN", "value": token}, {"name": "EMPTY", "value": ""}],
    )
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-allocation")

    def fail(command, **kwargs):
        if executor == "slurm":
            assert "HF_TOKEN=" + token not in command
            diagnostic = str(command) + token
        else:
            manifest = json.loads(kwargs["input"])
            assert manifest["spec"]["containers"][0]["env"][0]["value"] == token.replace("$", "$$")
            diagnostic = kwargs["input"]
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 300, stderr=b"")
        if failure == "oserror":
            raise OSError("probe unavailable: " + token)
        if failure == "bytes_stderr":
            diagnostic = diagnostic.encode()
        raise subprocess.CalledProcessError(1, command, stderr=diagnostic)

    monkeypatch.setattr(versioning.subprocess, "run", fail)
    with pytest.raises(ValueError, match="runtime version detection failed") as error:
        versioning.detect_runtime_versions(deployment)
    rendered = "".join(traceback.format_exception(error.value))
    for value in (token, token.replace("$", "$$")):
        for encoded in (value, repr(value)[1:-1], json.dumps(value)[1:-1]):
            assert encoded not in rendered
    assert error.value.__cause__ is None and error.value.__context__ is None


def test_short_environment_values_preserve_diagnostic_numbers():
    from aisimulate.support.versioning import _redact_diagnostic

    diagnostic = "HF_HUB_OFFLINE=1 --nodes=1 exit status 1 node1 version 0.1.1"
    assert _redact_diagnostic(diagnostic, ("1",), ("HF_HUB_OFFLINE",)) == (
        "HF_HUB_OFFLINE=<redacted> --nodes=1 exit status 1 node1 version 0.1.1"
    )


@pytest.mark.parametrize("failure", ["timeout", "oserror", "returncode"])
def test_probe_fallback_does_not_format_exception_command(monkeypatch, failure):
    from aisimulate.support.versioning import _run

    def fail(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(["HF_TOKEN=short"], 300)
        if failure == "returncode":
            raise subprocess.CalledProcessError(1, ["HF_TOKEN=short"])
        raise OSError("HF_TOKEN=short")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(ValueError) as error:
        _run(["HF_TOKEN=short"])
    assert "short" not in str(error.value)
    if failure == "returncode":
        assert "CalledProcessError (returncode=1)" in str(error.value)
