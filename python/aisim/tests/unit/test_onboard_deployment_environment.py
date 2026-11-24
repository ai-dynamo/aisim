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
        [{"name": "PYTHONHOME", "value": "/alternate/python"}],
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


def test_short_private_value_is_redacted_outside_environment_assignments():
    from aisimulate.support.versioning import _redact_diagnostic

    assert _redact_diagnostic("HF_TOKEN: xyz; rejected xyz", ("xyz",), ("HF_TOKEN",)) == (
        "HF_TOKEN: <redacted>; rejected <redacted>"
    )


def test_shell_quoted_private_value_is_redacted():
    import shlex

    from aisimulate.support.versioning import _redact_diagnostic

    value = "synthetic'quote"
    assert _redact_diagnostic(shlex.quote(value), (value,), ("CUSTOM_AUTH",)) == "<redacted>"


@pytest.mark.parametrize("separator", [":", ": ", ":  "])
def test_short_credentials_redacted_in_json_diagnostics(separator):
    from aisimulate.support.versioning import _redact_diagnostic

    diagnostic = '{"name":"CUSTOM_AUTH","value"' + separator + '"s3cr3t","exit_code":1}'
    redacted = _redact_diagnostic(diagnostic, ("s3cr3t",), ("CUSTOM_AUTH",))
    assert json.loads(redacted) == {"name": "CUSTOM_AUTH", "value": "<redacted>", "exit_code": 1}


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


@pytest.mark.parametrize(
    "value", ["/cache/ascii", "/cache/😀", "/cache/𠮷", "/cache/\u0085separator", "/cache/\u2028separator\u2029"]
)
def test_public_collection_environment_roundtrips_yaml_unicode(tmp_path, value):
    from collector.fpm_forward.entry import _inline_overrides

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
    env = [{"name": "HF_HOME", "value": value}]
    command = fpm_cli_args(request, output_dir=tmp_path, plan_only=True, deployment=FPMDeployment(extra_env=env))
    overrides = [command[index + 1] for index, item in enumerate(command) if item == "--generator-set"]
    parsed = _inline_overrides(overrides)
    assert parsed["K8sConfig"]["extra_env"] == env
    assert parsed["K8sConfig"]["extra_env"][0]["value"].encode() == value.encode()


@pytest.mark.parametrize("credential", ["matching", "missing", "changed"])
def test_probe_backed_collection_delivers_caller_literals_privately(tmp_path, monkeypatch, capsys, credential):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from aisimulate.support import collection_readiness, fpm, plan, runtime
    from collector.fpm_forward import entry
    from collector.fpm_forward.cli import _parser
    from collector.fpm_forward.config import archive_generator_overrides

    token = "/synthetic/private-hf-cache"
    request = SupportRequest.model_validate(
        {
            "identity": {
                "model": "fixture/model",
                "model_kind": "moe",
                "model_revision": "a" * 40,
                "framework": "vllm",
                "framework_version": "0.28.0",
                "gpu": "gb300",
                "interconnect": "nvlink",
            }
        }
    )
    environment = [{"name": "HF_HOME", "value": token}]
    archived = archive_generator_overrides({"K8sConfig": {"extra_env": environment}})["K8sConfig"]["extra_env"]
    saved = {"executor": "kubernetes", "image": "synthetic:image", "extra_env": archived}
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(
        runtime,
        "runtime_probe_manifest",
        lambda _: {"observations_index": str(tmp_path / "probe/observations.json"), "configuration": "tp1"},
    )
    monkeypatch.setattr(
        runtime,
        "verify_runtime_profile",
        lambda _: {
            "provenance": {
                "launch": {"deployment": saved},
                "instrumentation": {"manifest": "instrumentation/manifest.json"},
            }
        },
    )
    monkeypatch.setattr(runtime, "verify_runtime_acceptance", lambda _: None)
    monkeypatch.setattr(plan, "check_plan", lambda *a: None)
    monkeypatch.setattr(plan, "plan_lock", lambda *a: nullcontext())
    monkeypatch.setattr(collection_readiness, "assess_readiness", lambda *a, **kw: {"ready_for_full_collection": True})
    commands = []

    def resolve(command):
        commands.append(command)
        args = _parser().parse_args(command[3:])
        return args, (SimpleNamespace(), entry._load_generator_overrides(args))

    monkeypatch.setattr(fpm, "_resolve_execution", resolve)
    deliveries = []

    def execute(args, resolved):
        deliveries.append(resolved[1])
        assert {"name": "HF_HOME", "value": token} in resolved[1]["K8sConfig"]["extra_env"]
        return []

    if credential == "matching":
        monkeypatch.setattr(entry, "run_resolved", execute)
    else:
        from collector.fpm_forward import runner

        def unexpected_launch(*args, **kwargs):
            pytest.fail("invalid deployment credentials reached the worker launcher")

        monkeypatch.setattr(runner, "_cell_runner", unexpected_launch)
    supplied = environment if credential == "matching" else None
    deployment = FPMDeployment(image="synthetic:image", extra_env=supplied) if supplied else None
    if credential == "changed":
        monkeypatch.setenv("HF_HOME", "changed-private-cache")
    status = fpm.run_fpm(request, output_dir=tmp_path / "collection", execute=True, smoke=True, deployment=deployment)
    assert status == (0 if credential == "matching" else 1)
    assert bool(deliveries) == (credential == "matching")
    output = capsys.readouterr()
    assert token not in output.out + output.err
    if credential != "matching":
        assert "requires the original value" in output.err
    assert all(token not in " ".join(command) for command in commands)
    for artifact in (tmp_path / "collection").rglob("*"):
        if artifact.is_file():
            assert token not in artifact.read_text()
