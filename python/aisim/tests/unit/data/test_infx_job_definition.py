# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The internal GitLab job definition keeps its safety properties (it is never executed here)."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

JOB = Path(__file__).resolve().parents[5] / "scripts/ci/gitlab/aisim-infx-ingest.gitlab-ci.yml"


class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in keys:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark, f"duplicate key: {key}", key_node.start_mark
                )
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def test_definition_parser_rejects_duplicate_keys():
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key: script"):
        yaml.load("job:\n  script: first\n  script: second\n", Loader=UniqueKeyLoader)


@pytest.fixture(scope="module")
def definition():
    spec, definition = yaml.load_all(JOB.read_text(), Loader=UniqueKeyLoader)
    inputs = spec["spec"]["inputs"]
    assert inputs["job"]["default"] == "ingest"
    assert inputs["job"]["options"] == ["ingest", "promote", "staleness"]
    assert inputs["candidate-snapshot-id"]["regex"] == "^([0-9a-f]{64})?$"
    return definition


def _text(job):
    return "\n".join(str(line) for line in job.get("script", []))


def test_runs_only_on_schedule_or_manual_and_is_pinned(definition):
    assert {r.get("if") for r in definition["workflow"]["rules"] if "if" in r} == {
        '$CI_PIPELINE_SOURCE == "schedule"',
        '$CI_PIPELINE_SOURCE == "web"',
    }
    pinned = "\n".join(definition[".pinned"]["before_script"])
    assert "AISIM_REVIEWED_COMMIT" in pinned and "-eq 40" in pinned and "rev-parse HEAD" in pinned


def test_ingestion_runs_without_the_cache_token_and_never_pushes_to_the_default_branch(definition):
    job = definition["publish_candidate"]
    text = _text(job)
    assert "env -u INFX_CACHE_WRITE_TOKEN" in text and 'push -q origin "candidate/' in text
    assert "main" not in text and "unchanged refresh: no commit" in text and "git status --porcelain" in text
    assert "qualified" not in text.replace("not qualified", "")


def test_promotion_is_manual_with_a_named_reviewer_and_staleness_is_alert_only(definition):
    assert [r["when"] for r in definition["promote"]["rules"]] == ["manual"]
    assert '--reviewer "$GITLAB_USER_NAME"' in _text(definition["promote"])
    assert definition["staleness"]["allow_failure"] == {"exit_codes": [10, 11]}


def test_failure_alert_job_exists(definition):
    alert = definition["alert_failure"]
    assert alert["when"] == "on_failure" and alert["stage"] == "notify"
    assert "needs" not in alert and "extends" not in alert and "before_script" not in alert
    assert "AISIM_REPO_URL" not in _text(alert) and "JOB_PY" not in _text(alert)
    rules = " ".join(r["if"] for r in alert["rules"])
    assert '"$[[ inputs.job ]]" == "staleness"' in rules and '"$[[ inputs.job ]]" == "ingest"' in rules


def test_ci_contract_clean_workspace_default_pipelines_and_staleness_status(definition):
    assert definition["variables"]["GIT_STRATEGY"] == "empty"
    rules = " ".join(r["if"] for r in definition["publish_candidate"]["rules"])
    assert (
        '"$[[ inputs.job ]]" == "ingest"' in rules and "== null" not in rules and "!=" not in rules
    )  # promote never acquires
    assert "ls-remote --exit-code --heads origin" in _text(definition["publish_candidate"])
    assert "||" not in _text(definition["staleness"])  # unexpected statuses must fail the job


def test_staleness_only_uses_the_read_credential(definition):
    assert "INFX_CACHE_READ_TOKEN" in _text(definition["staleness"])
    assert "INFX_CACHE_WRITE_TOKEN" not in _text(definition["staleness"])
    for name in ("publish_candidate", "promote"):
        assert definition[name]["environment"] == {"name": "infx-cache-writer", "action": "access"}
    assert "environment" not in definition["staleness"] and "environment" not in definition["alert_failure"]


def test_dependency_setup_cannot_read_the_cache_write_token(definition, tmp_path):
    setup = "\n".join(definition[".pinned"]["before_script"])
    sync = next(line.strip() for line in setup.splitlines() if "uv sync" in line)
    uv = tmp_path / "uv"
    uv.write_text('#!/bin/sh\n[ "${INFX_CACHE_WRITE_TOKEN+x}" != x ]\n')
    uv.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "INFX_CACHE_WRITE_TOKEN": "synthetic"}
    subprocess.run(["sh", "-eu", "-c", sync], env=env, cwd=tmp_path, check=True)


def test_job_commands_do_not_resolve_new_dependencies(definition):
    command = shlex.split(definition[".pinned"]["variables"]["JOB_PY"])
    assert command[:3] == ["uv", "run", "--frozen"]
    assert command[command.index("--project") + 1] == "aisim/python/aisim"


@pytest.mark.parametrize("slack_fails", [False, True])
@pytest.mark.parametrize("has_report", [False, True])
def test_failure_notification_works_without_repository_fetch(
    definition, monkeypatch, tmp_path, slack_fails, has_report
):
    import json
    import smtplib
    import urllib.request

    for key, value in {
        "CI_PIPELINE_URL": "https://ci.example.test/pipeline/1",
        "AISIM_REPO_URL": "/unavailable-repository",
        "SLACK_WEBHOOK_URL": "https://alerts.example.test/hook",
        "SMTP_HOST": "smtp.example.test",
        "SMTP_FROM": "bot@example.test",
        "AISIM_JOB_OWNER_EMAIL": "owner@example.test",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)
    if has_report:
        (tmp_path / "run-report.json").write_text(json.dumps({"status": "failed", "error": "structured fixture error"}))
    delivered = []

    class Response:
        def close(self):
            pass

    def slack(request, timeout):
        assert ("structured fixture error" in json.loads(request.data)["text"]) is has_report
        delivered.append("slack")
        if slack_fails:
            raise OSError("fixture Slack failure")
        return Response()

    class SMTP:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def send_message(self, message):
            assert message["To"] == "owner@example.test" and "pipeline/1" in message.get_content()
            assert ("structured fixture error" in message.get_content()) is has_report
            delivered.append("mail")

    monkeypatch.setattr(urllib.request, "urlopen", slack)
    monkeypatch.setattr(smtplib, "SMTP", SMTP)
    script = _text(definition["alert_failure"]).split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    with pytest.raises(SystemExit) as exc:
        exec(compile(script, "gitlab-notification", "exec"), {})
    assert exc.value.code == int(slack_fails) and delivered == ["slack", "mail"]


def test_ci_ingestion_defers_failure_alerts_to_the_fallback(definition):
    assert "run --no-notify" in _text(definition["publish_candidate"])
    assert "run-report.json" in _text(definition["alert_failure"])
    assert definition["publish_candidate"]["artifacts"]["when"] == "always"
