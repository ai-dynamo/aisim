# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The internal GitLab job definition keeps its safety properties (it is never executed here)."""

import json
import os
import re
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
    assert "env -u INFX_CACHE_WRITE_TOKEN" in text and 'push -q origin "$BRANCH"' in text
    assert 'BRANCH="candidate/${SID}-${CI_PIPELINE_ID}"' in text
    assert "main" not in text and "unchanged refresh: no commit" in text and "git status --porcelain" in text
    assert "$JOB_PY promote" not in text


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
    assert '$INFX_JOB == "staleness"' in rules and '$INFX_JOB == "ingest"' in rules


def test_ci_contract_clean_workspace_default_pipelines_and_staleness_status(definition):
    assert definition["variables"]["GIT_STRATEGY"] == "empty"
    rules = " ".join(r["if"] for r in definition["publish_candidate"]["rules"])
    assert '$INFX_JOB == "ingest"' in rules and "== null" not in rules and "!=" not in rules  # promote never acquires
    assert "ls-remote --heads origin" in _text(definition["publish_candidate"])
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


@pytest.mark.parametrize(
    "source,selector,expected",
    [
        ("schedule", "ingest", {"publish_candidate", "alert_failure"}),
        ("schedule", "staleness", {"staleness", "alert_failure"}),
        ("schedule", "promote", set()),
        ("web", "ingest", {"publish_candidate", "alert_failure"}),
        ("web", "promote", {"promote", "alert_failure"}),
        ("web", "staleness", {"alert_failure"}),
        ("push", "ingest", set()),
    ],
)
def test_selector_rules_use_gitlab_variable_comparisons_and_route_jobs(definition, source, selector, expected):
    assert definition["variables"]["INFX_JOB"] == "$[[ inputs.job ]]"
    variables = {"CI_PIPELINE_SOURCE": source, "INFX_JOB": selector}
    selected = set()
    # This definition uses GitLab's documented variable-left string equality
    # subset. Validate the complete expression, then exercise its routing table.
    comparison = r'\$([A-Z_]+)\s*==\s*"([^"]*)"'
    for name in ("publish_candidate", "promote", "staleness", "alert_failure"):
        for rule in definition[name]["rules"]:
            parts = rule["if"].split(" && ")
            matches = [re.fullmatch(comparison, part) for part in parts]
            assert all(matches), rule["if"]
            if all(variables[m[1]] == m[2] for m in matches):
                selected.add(name)
                break
    assert selected == expected


@pytest.mark.parametrize("kind", ["historical", "different-baseline", "pending", "generated-pending"])
def test_recurring_snapshot_keeps_historical_reviews_and_reuses_only_current_pending(definition, tmp_path, kind):
    sid = "a" * 64
    qualified_a = {
        "snapshot_id": sid,
        "snapshot_path": f"snapshots/{sid}/snapshot.json",
        "created_at": "2026-11-01T00:00:00Z",
        "qualified_at": "2026-11-01T00:00:00Z",
        "reviewer": "reviewer",
    }
    qualified_b = {**qualified_a, "snapshot_id": "b" * 64, "snapshot_path": f"snapshots/{'b' * 64}/snapshot.json"}
    pending = {
        "snapshot_id": sid,
        "snapshot_path": f"snapshots/{sid}/snapshot.json",
        "created_at": "2026-11-19T00:00:00Z",
    }
    schema = "aisim-infx-ingestion-state/v1"

    def git(path, *args):
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(path),
                "-c",
                "user.name=Synthetic Fixture",
                "-c",
                "user.email=fixture@example.test",
                *args,
            ],
            text=True,
        ).strip()

    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "init", "-qb", "main")

    def commit_state(qualified, candidate):
        (seed / "ingestion-state.json").write_text(
            json.dumps({"schema_version": schema, "qualified": qualified, "candidate": candidate})
        )
        git(seed, "add", "-A")
        git(seed, "commit", "-qm", "synthetic state")

    commit_state(qualified_b, None)
    git(seed, "remote", "add", "origin", str(remote))
    git(seed, "push", "-q", "origin", "main")
    branch = f"candidate/{sid}" + ("-17" if kind == "generated-pending" else "")
    git(seed, "checkout", "-qb", branch)
    commit_state(qualified_b if "pending" in kind else qualified_a, None if kind == "historical" else pending)
    git(seed, "push", "-q", "origin", branch)
    old = git(remote, "rev-parse", branch)

    workspace = tmp_path / "work"
    workspace.mkdir()
    subprocess.run(["git", "clone", "-q", "--branch", "main", str(remote), str(workspace / "cache-repo")], check=True)
    (workspace / "cache-repo/ingestion-state.json").write_text(
        json.dumps({"schema_version": schema, "qualified": qualified_b, "candidate": pending})
    )
    (workspace / "run-report.json").write_text(json.dumps({"snapshot_id": sid}))
    script = definition["publish_candidate"]["script"][-1]
    subprocess.run(
        ["sh", "-eu", "-c", script],
        cwd=workspace,
        env={**os.environ, "CI_PIPELINE_ID": "42", "INFX_CACHE_WRITE_TOKEN": ""},
        check=True,
    )
    assert git(remote, "rev-parse", branch) == old
    branches = git(remote, "for-each-ref", "--format=%(refname:short)", "refs/heads").splitlines()
    new_branch = f"candidate/{sid}-42"
    if kind in ("pending", "generated-pending"):
        assert new_branch not in branches
    else:
        assert new_branch in branches
        state = json.loads(git(remote, "show", f"{new_branch}:ingestion-state.json"))
        assert state["qualified"] == qualified_b and state["candidate"]["snapshot_id"] == sid


@pytest.mark.parametrize("generation", ["", "42"])
def test_promotion_selects_the_reviewed_candidate_generation(definition, tmp_path, generation):
    # Execute the clone command with a fake git to observe the exact branch;
    # promotion must push back to that same branch, preserving historical refs.
    script = next(line for line in definition["promote"]["script"] if "GENERATION=" in line)
    script = script.replace("$[[ inputs.candidate-snapshot-id ]]", "a" * 64).replace(
        "$[[ inputs.candidate-generation ]]", generation
    )
    git = tmp_path / "git"
    git.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > clone-args\n')
    git.chmod(0o755)
    subprocess.run(
        ["sh", "-eu", "-c", script],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "INFX_CACHE_WRITE_TOKEN": "",
            "INFX_CACHE_REPO_URL": "synthetic",
        },
        check=True,
    )
    args = (tmp_path / "clone-args").read_text().splitlines()
    assert args[args.index("--branch") + 1] == "candidate/" + "a" * 64 + (f"-{generation}" if generation else "")
    assert "HEAD:refs/heads/${BRANCH}" in _text(definition["promote"])
