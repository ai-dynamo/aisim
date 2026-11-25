# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trust boundary of the credentialed E2E accuracy job.

The private HF read token may only reach steps that run main's code (``trusted/``) in a virtualenv that
never receives the evaluated wheel. Preview and PR-controlled code never shares a step, interpreter or
environment variable with it. The nightly page keeps publishing from default-branch artifacts.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
NAMES = ("e2e-accuracy.yml", "e2e-accuracy-branch.yml")
TRUSTED_PYTHON = '"${RUNNER_TEMP}/trusted-venv/bin/python" trusted/'
PIN = re.compile(r"[^@]+@[0-9a-f]{40}( #.*)?$")
ENVIRONMENT = "${{ inputs.preview && 'e2e-accuracy-preview' || 'e2e-accuracy-private' }}"
# Jobs that also download the wheel keep the storage environment that holds its Artifactory credential.
WHEEL_ENVIRONMENT = "${{ inputs.preview && 'pr-wheel-staging' || 'automated-release' }}"
WHEEL_JOBS = {"shards": WHEEL_ENVIRONMENT, "agentx": WHEEL_ENVIRONMENT}


def load(name: str) -> dict:
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)


def steps(name: str):
    for job_id, job in load(name)["jobs"].items():
        for step in job.get("steps", []):
            yield job_id, job, step


# GitHub Actions reads a context property with dot or index notation: secrets.NAME, secrets['NAME'], secrets["NAME"].
_SECRET = r"""\bsecrets\s*(?:\.\s*|\[\s*['"])"""
_HF_SECRET = re.compile(_SECRET + "HF_", re.IGNORECASE)
_ANY_SECRET = re.compile(_SECRET, re.IGNORECASE)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def references_secret(value) -> bool:
    # The private HF read token; the wheel storage credential is a separate secret handled by its own action.
    return any(_HF_SECRET.search(text) for text in _strings(value))


def references_any_secret(value) -> bool:
    return any(_ANY_SECRET.search(text) for text in _strings(value))


def expression(source: str, **contexts):
    """Evaluate the GitHub Actions subset these gates use: contexts, literals, ==, !=, &&, || and startsWith."""
    text = source.strip().removeprefix("${{").removesuffix("}}")
    if not re.fullmatch(r"[\w\s.'/&|=!(),-]*", text) or "__" in text:
        raise ValueError(f"unsupported workflow expression: {source!r}")
    namespace = {name: SimpleNamespace(**values) for name, values in contexts.items()}
    namespace |= {"true": True, "false": False, "always": lambda: True}
    namespace["startsWith"] = lambda value, prefix: str(value).lower().startswith(str(prefix).lower())
    return eval("(" + text.replace("&&", " and ").replace("||", " or ") + ")", {"__builtins__": {}}, namespace)


def render(template: str, **contexts) -> str:
    return re.sub(r"\$\{\{(.*?)\}\}", lambda match: str(expression(match[1], **contexts)), template)


@pytest.mark.parametrize(
    "reference",
    [
        "${{ secrets.HF_READ_AISIM_INFX_CACHE }}",
        "${{ secrets['HF_READ_AISIM_INFX_CACHE'] }}",
        '${{ secrets["HF_READ_AISIM_INFX_CACHE"] }}',
    ],
)
def test_secret_references_are_recognized_in_every_context_notation(reference):
    step = {"env": {"HF_TOKEN": reference}}
    assert references_secret(step) and references_any_secret(step)
    other = {"env": {"TOKEN": reference.replace("HF_READ_AISIM_INFX_CACHE", "ARTIFACTORY_TOKEN")}}
    assert references_any_secret(other) and not references_secret(other)
    assert not references_any_secret({"env": {"TOKEN": "${{ github.token }}", "NAME": "secrets are private"}})


@pytest.mark.parametrize("name", NAMES)
def test_token_steps_run_only_main_code_in_the_trusted_environment(name):
    seen = 0
    for job_id, _, step in steps(name):
        if not references_secret(step):
            continue
        seen += 1
        assert "uses" not in step, f"{name}:{job_id}: a secret must not reach an action ({step.get('name')})"
        lines = [ln.strip() for ln in step["run"].splitlines() if ln.strip() and not ln.strip().startswith("#")]
        launches = [ln for ln in lines if "python" in ln]
        assert launches, f"{name}:{job_id}: token step {step.get('name')!r} runs nothing"
        for line in launches:
            assert line.startswith(TRUSTED_PYTHON), f"{name}:{job_id}: {line}"
        assert "${{" not in step["run"], "expressions must go through env, never into the shell text"
        assert not step["env"].keys() - {"HF_TOKEN", "HF_HOME", "AISIM_DATA_CACHE", "GITHUB_TOKEN"}
    assert seen


@pytest.mark.parametrize("name", NAMES)
def test_no_other_step_receives_a_secret_and_secrets_are_not_job_or_workflow_wide(name):
    document = load(name)
    assert not references_any_secret(document.get("env", {}))
    for job_id, job, step in steps(name):
        assert not references_any_secret(job.get("env", {})), job_id
        if references_secret(step):
            assert step["env"].get("HF_TOKEN", "").startswith("${{ secrets.HF_READ_AISIM_INFX_CACHE"), step
    # The wheel is built from the evaluated revision and so runs PR-controlled build code: no secret, ever.
    wheel = load("e2e-accuracy-branch.yml")["jobs"]["wheel"]
    assert not references_secret(wheel)


def test_every_job_that_seals_or_opens_artifacts_uses_the_same_key_secret():
    """The artifact key derives from HF_TOKEN, so producers and consumers must name the same secret."""
    expected = "${{ secrets.HF_READ_AISIM_INFX_CACHE }}"
    seen = {"seal": set(), "open": set()}
    for name in NAMES:
        for job_id, _, step in steps(name):
            for action in re.findall(r"private_artifacts\.py (seal|open)\b", step.get("run", "")):
                assert step["env"]["HF_TOKEN"] == expected, f"{name}:{job_id}: {step.get('name')}"
                seen[action].add(job_id)
    # Every producer has a consumer in another job (the environments differ between them).
    assert {"measurements", "shards", "agentx"} <= seen["seal"]
    assert {"shards", "campaign", "qualify-agentx"} <= seen["open"]


def test_credentialed_jobs_declare_the_environment_and_check_out_main():
    for name, job_ids in (
        ("e2e-accuracy.yml", ["resolve"]),
        ("e2e-accuracy-branch.yml", ["measurements", "shards", "campaign", "agentx", "qualify-agentx"]),
    ):
        document = load(name)
        for job_id in job_ids:
            job = document["jobs"][job_id]
            assert job["environment"] == WHEEL_JOBS.get(job_id, ENVIRONMENT), job_id
            trusted = [s for s in job["steps"] if s.get("with", {}).get("path") == "trusted"]
            assert len(trusted) == 1, job_id
            assert trusted[0]["with"] == {"ref": "main", "path": "trusted", "persist-credentials": "false"}
            # Every token step comes after the trusted checkout and the trusted venv exists.
            names = [s.get("name", s.get("uses")) for s in job["steps"]]
            first_token = next(i for i, s in enumerate(job["steps"]) if references_secret(s))
            assert job["steps"].index(trusted[0]) < first_token, job_id
            venv = next(
                i for i, s in enumerate(job["steps"]) if "trusted-venv" in s.get("run", "") and "venv" in s["run"]
            )
            assert venv < first_token, (job_id, names)


def _assert_no_evaluated_wheel_in_trusted_venv(run):
    if "trusted-venv/bin/pip" in run:
        assert "accuracy-wheel" not in run and "agentx-wheel" not in run and "*.whl" not in run
    installs = re.search(r"\bpip(?:\d+(?:\.\d+)?)?[\"']?\s+install\b", run)
    if installs and any(directory in run for directory in ("accuracy-wheel", "agentx-wheel")):
        assert "trusted-venv" not in run


@pytest.mark.parametrize("job_id", ["campaign", "qualify-agentx"])
def test_evaluated_requirements_install_only_after_the_last_token_step(job_id):
    """Installing the evaluated checkout's requirements runs its install-time code, which may poison later steps
    through $GITHUB_ENV or $GITHUB_PATH. Unlike shards and agentx, these two jobs never need that code before
    the token is gone, so the install must come after every step that holds a secret."""
    job_steps = load("e2e-accuracy-branch.yml")["jobs"][job_id]["steps"]
    token = [index for index, step in enumerate(job_steps) if references_secret(step)]
    installs = [
        index
        for index, step in enumerate(job_steps)
        if "pip install" in step.get("run", "") and "-r scripts/" in step.get("run", "")
    ]
    assert token and installs, job_id
    assert max(token) < min(installs), (job_id, token, installs)
    # Steps from the first token step to the install (such as the revision resolver) use the trusted interpreter only.
    for step in job_steps[min(token) : min(installs)]:
        for line in step.get("run", "").splitlines():
            assert not line.strip().startswith(("python ", "python3 ")), (job_id, step.get("name"), line)
            assert "$(python " not in line, (job_id, step.get("name"), line)


def trusted_bootstraps():
    """(workflow, job, command) for every step that creates the trusted venv with the stock interpreter."""
    for name in NAMES:
        for job_id, _, step in steps(name):
            for line in step.get("run", "").splitlines():
                if '-m venv "${RUNNER_TEMP}/trusted-venv"' in line:
                    yield name, job_id, line.strip()


def test_every_trusted_interpreter_invocation_that_reads_the_cwd_runs_isolated():
    """``python -m`` and ``python -`` put the working directory first on sys.path, and the evaluated checkout
    occupies it, so a checkout-local ``venv.py`` or ``json.py`` would run before a token step. ``-I`` drops it."""
    bootstraps = list(trusted_bootstraps())
    assert {job for _, job, _ in bootstraps} >= {
        "resolve",
        "measurements",
        "shards",
        "campaign",
        "agentx",
        "qualify-agentx",
    }
    for name, job_id, command in bootstraps:
        assert command.startswith("python -I -m venv "), (name, job_id, command)
    for name in NAMES:
        for job_id, _, step in steps(name):
            for line in step.get("run", "").splitlines():
                if re.search(r'trusted-venv/bin/python"? (-[a-zA-Z]+ )*(-|-c|-m)( |$)', line):
                    assert re.search(r'python"? -I ', line), (name, job_id, line.strip())


def _poisoned_workspace(tmp_path, module):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / f"{module}.py").write_text("open('POISONED', 'w').write('ran')\nraise SystemExit(0)\n")
    interpreter = tmp_path / "bin/python"
    interpreter.parent.mkdir()
    interpreter.symlink_to(sys.executable)
    environment = os.environ | {
        "PATH": f"{interpreter.parent}{os.pathsep}{os.environ['PATH']}",
        "RUNNER_TEMP": str(tmp_path / "runner"),
    }
    (tmp_path / "runner").mkdir()
    return workspace, environment


@pytest.mark.parametrize("command", sorted({command for _, _, command in trusted_bootstraps()}))
def test_a_checkout_local_venv_module_never_runs_in_the_trusted_bootstrap(tmp_path, command):
    workspace, environment = _poisoned_workspace(tmp_path, "venv")

    def run(text):
        return subprocess.run(
            ["bash", "-e", "-c", text + " --without-pip"],
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
        )

    # Control: without -I the poisoned module shadows the standard library and runs.
    control = run(command.replace(" -I", "", 1))
    assert control.returncode == 0 and (workspace / "POISONED").exists(), control.stderr
    (workspace / "POISONED").unlink()
    result = run(command)
    assert result.returncode == 0, result.stderr
    assert not (workspace / "POISONED").exists(), "the trusted bootstrap executed a checkout-local venv.py"
    assert (tmp_path / "runner/trusted-venv/bin/python").exists()


@pytest.mark.parametrize("job_id", ["measurements", "shards", "campaign", "agentx", "qualify-agentx"])
def test_trusted_pip_ignores_checkout_configuration(tmp_path, job_id):
    workspace = tmp_path / "workspace"
    config = workspace / ".pip/pip.conf"
    config.parent.mkdir(parents=True)
    interpreter = workspace / "config-python"
    interpreter.write_text('#!/bin/sh\n: > "$HOME/PIP_CONFIG_EXECUTED"\nexit 86\n')
    interpreter.chmod(0o755)
    config.write_text(f"[global]\npython = {interpreter}\n")
    requirements = workspace / "trusted/scripts/e2e_accuracy/requirements.txt"
    requirements.parent.mkdir(parents=True)
    requirements.write_text("")
    runner_temp = tmp_path / "runner"
    subprocess.run([sys.executable, "-I", "-m", "venv", str(runner_temp / "trusted-venv")], check=True)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PIP_") and key != "_PIP_RUNNING_IN_SUBPROCESS"
    } | {
        "HOME": str(workspace),
        "XDG_CONFIG_HOME": str(workspace / ".config"),
        "RUNNER_TEMP": str(runner_temp),
        "PIP_NO_INDEX": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }
    command = next(
        line.strip()
        for step in load("e2e-accuracy-branch.yml")["jobs"][job_id]["steps"]
        for line in step.get("run", "").splitlines()
        if 'trusted-venv/bin/pip" install' in line
    )

    def run(text):
        return subprocess.run(
            ["bash", "-e", "-c", text], cwd=workspace, env=environment, capture_output=True, text=True
        )

    # Control: even hash-required installs execute the configured interpreter before checking requirements.
    marker = workspace / "PIP_CONFIG_EXECUTED"
    control = run(command.replace("PIP_CONFIG_FILE=/dev/null ", "", 1))
    assert control.returncode == 86 and marker.exists(), control.stderr
    marker.unlink()
    result = run(command)
    assert result.returncode == 0, result.stderr
    assert not marker.exists(), "the trusted pip install executed the checkout-configured interpreter"


@pytest.mark.parametrize("module", ["json", "re"])
def test_a_checkout_local_module_never_runs_in_the_revision_resolver(tmp_path, module):
    workspace, environment = _poisoned_workspace(tmp_path, module)
    revision = "0123456789abcdef" * 2 + "01234567"
    shard = tmp_path / "runner/agentx-results/shard-0.json"
    shard.parent.mkdir(parents=True)
    shard.write_text(json.dumps({"snapshot": {"dataset_revision": revision}}))
    python = tmp_path / "runner/trusted-venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    resolve = next(
        step
        for step in load("e2e-accuracy-branch.yml")["jobs"]["qualify-agentx"]["steps"]
        if step.get("name") == "Resolve the AgentX revision and options (no secret)"
    )
    environment |= {"GITHUB_ENV": str(tmp_path / "github-env"), "PREVIEW": "false", "AGENTX_CAMPAIGN_CATALOG": ""}
    result = subprocess.run(
        ["bash", "-e", "-c", resolve["run"]], cwd=workspace, env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert not (workspace / "POISONED").exists(), f"the resolver executed a checkout-local {module}.py"
    assert f"AGENTX_RESOLVED_REVISION={revision}" in (tmp_path / "github-env").read_text()


def test_the_trusted_venv_never_sees_the_evaluated_wheel():
    for name in NAMES:
        for job_id, _, step in steps(name):
            run = step.get("run", "")
            _assert_no_evaluated_wheel_in_trusted_venv(run)


@pytest.mark.parametrize("directory", ["accuracy-wheel", "agentx-wheel"])
@pytest.mark.parametrize("installer", ["pip", "pip3", "python -m pip"])
@pytest.mark.parametrize("venv", ["trusted-venv", "evaluated-venv"])
def test_wheel_install_guard_recognizes_both_pip_forms_and_wheel_directories(directory, installer, venv):
    executable, *arguments = installer.split()
    run = (
        f'"${{RUNNER_TEMP}}/{venv}/bin/{executable}" {" ".join(arguments)} install "${{RUNNER_TEMP}}"/{directory}/*.whl'
    )
    if venv == "trusted-venv":
        with pytest.raises(AssertionError):
            _assert_no_evaluated_wheel_in_trusted_venv(run)
    else:
        _assert_no_evaluated_wheel_in_trusted_venv(run)


def test_agentx_qualification_uses_the_replay_checkout_without_the_private_token():
    jobs = load("e2e-accuracy-branch.yml")["jobs"]
    for job_id in ("agentx", "qualify-agentx"):
        checkout = next(s for s in jobs[job_id]["steps"] if s.get("uses", "").startswith("actions/checkout@"))
        assert checkout["with"] == {"ref": "${{ github.sha }}", "persist-credentials": "false"}
    steps = jobs["qualify-agentx"]["steps"]
    fetch = next(s for s in steps if "run_agentx_accuracy.py fetch" in s.get("run", ""))
    qualify = next(s for s in steps if "run_agentx_accuracy.py qualify" in s.get("run", ""))
    assert references_secret(fetch)
    assert "--metadata-only" in fetch["run"]
    assert steps.index(fetch) < steps.index(qualify)
    assert not references_secret(qualify)
    assert qualify["run"].lstrip().startswith("python scripts/e2e_accuracy/run_agentx_accuracy.py qualify")


@pytest.mark.parametrize(
    "revision", ["a" * 40 + "\nBASH_ENV=/tmp/startup.sh", "a" * 40 + "\n", "A" * 40, "a" * 39, "main", None, 123]
)
def test_agentx_rejects_invalid_revision_before_writing_environment(tmp_path, revision):
    shard = tmp_path / "agentx-results/shard-0.json"
    shard.parent.mkdir()
    shard.write_text(json.dumps({"snapshot": {"dataset_revision": revision}}))
    environment = tmp_path / "github-env"
    environment.write_text("EXISTING=value\n")
    python = tmp_path / "trusted-venv/bin/python"  # the revision step runs on the trusted interpreter
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    resolve = next(
        step
        for step in load("e2e-accuracy-branch.yml")["jobs"]["qualify-agentx"]["steps"]
        if step.get("name") == "Resolve the AgentX revision and options (no secret)"
    )
    result = subprocess.run(
        ["bash", "-e", "-c", resolve["run"]],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_ENV": str(environment),
            "PREVIEW": "false",
            "AGENTX_CAMPAIGN_CATALOG": "",
        },
    )
    assert result.returncode != 0
    assert "dataset_revision must be" in result.stderr
    assert environment.read_text() == "EXISTING=value\n"


def test_agentx_valid_revision_reaches_fetch_and_qualification_unchanged(tmp_path):
    revision = "0123456789abcdef" * 2 + "01234567"
    shard = tmp_path / "agentx-results/shard-0.json"
    shard.parent.mkdir()
    shard.write_text(json.dumps({"snapshot": {"dataset_revision": revision}}))
    environment = tmp_path / "github-env"
    env = os.environ | {
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_ENV": str(environment),
        "PREVIEW": "false",
        "AGENTX_CAMPAIGN_CATALOG": "",
    }
    # Stub only the external fetch/qualification commands; execute the workflow's shell verbatim.
    script = """import json, sys
from pathlib import Path
args = sys.argv[1:]
output = Path(args[args.index("--output") + 1])
output.mkdir()
if args[0] == "fetch":
    (output / "revision").write_text(args[args.index("--revision") + 1])
else:
    inputs = Path(args[args.index("--inputs") + 1])
    shards = Path(args[args.index("--shards") + 1])
    revision = json.loads((shards / "shard-0.json").read_text())["snapshot"]["dataset_revision"]
    assert (inputs / "revision").read_text() == revision
    (output / "revision").write_text(revision)
"""
    for prefix in ("trusted/", ""):
        target = tmp_path / f"{prefix}scripts/e2e_accuracy/run_agentx_accuracy.py"
        target.parent.mkdir(parents=True)
        target.write_text(script)
    python = tmp_path / "trusted-venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    job_steps = load("e2e-accuracy-branch.yml")["jobs"]["qualify-agentx"]["steps"]
    selected = [
        step
        for step in job_steps
        if step.get("name") == "Resolve the AgentX revision and options (no secret)"
        or "run_agentx_accuracy.py fetch" in step.get("run", "")
        or "run_agentx_accuracy.py qualify" in step.get("run", "")
    ]
    assert len(selected) == 3
    for step in selected:
        result = subprocess.run(
            ["bash", "-e", "-c", step["run"]], cwd=tmp_path, env=env, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        env.update(line.split("=", 1) for line in environment.read_text().splitlines())
    assert env["AGENTX_RESOLVED_REVISION"] == revision
    assert (tmp_path / "agentx-public/revision").read_text() == revision


@pytest.mark.parametrize(
    "event,ref,preview,runs",
    [
        ("schedule", "refs/heads/main", False, True),
        ("workflow_dispatch", "refs/heads/main", False, True),
        ("workflow_dispatch", "refs/heads/pull-request/7", True, True),
        ("workflow_dispatch", "refs/heads/pull-request/7", False, False),  # preview disabled on a preview branch
        ("schedule", "refs/heads/pull-request/7", False, False),
        ("workflow_dispatch", "refs/heads/feature", True, False),  # preview enabled on an unrelated branch
        ("workflow_dispatch", "refs/heads/feature", False, False),
        ("workflow_dispatch", "refs/heads/pull-request-other/7", True, False),
        ("push", "refs/heads/pull-request/7", True, False),
    ],
)
@pytest.mark.parametrize("repository,allowed", [("ai-dynamo/aisim", True), ("someone/aisim", False)])
def test_credentialed_job_runs_only_for_main_or_an_enabled_preview_branch(
    event, ref, preview, runs, repository, allowed
):
    gate = load("e2e-accuracy.yml")["jobs"]["resolve"]["if"]
    decision = expression(
        gate, github={"repository": repository, "event_name": event, "ref": ref}, inputs={"preview": preview}
    )
    assert bool(decision) is (runs and allowed)


def test_secrets_flow_only_through_the_default_branch_or_exact_preview_gate():
    document = load("e2e-accuracy.yml")
    assert set(document["on"]) == {"schedule", "workflow_dispatch"}  # never pull_request(_target) or workflow_run
    branches = document["jobs"]["branches"]
    assert branches["secrets"] == {
        "ARTIFACTORY_TOKEN": "${{ secrets.ARTIFACTORY_TOKEN }}",
        "HF_READ_AISIM_INFX_CACHE": "${{ secrets.HF_READ_AISIM_INFX_CACHE }}",
    }
    assert branches["uses"] == "./.github/workflows/e2e-accuracy-branch.yml"
    assert document["permissions"] == {"contents": "read"}


@pytest.mark.parametrize("name", NAMES)
def test_actions_are_sha_pinned_and_checkouts_drop_credentials(name):
    for job_id, _, step in steps(name):
        if "uses" in step and not step["uses"].startswith("./"):  # local actions only in the wheel jobs
            assert PIN.match(step["uses"]), (name, job_id, step["uses"])
        if step.get("uses", "").startswith("./"):
            assert job_id in {"wheel", *WHEEL_JOBS}, (name, job_id)
        if step.get("uses", "").startswith("actions/checkout@"):
            assert step["with"]["persist-credentials"] == "false"


def test_preview_never_publishes_and_the_nightly_page_publishes_from_main_without_secrets():
    branch = load("e2e-accuracy-branch.yml")
    pages = load("pages.yml")
    assert not references_any_secret(pages)
    assert pages["jobs"]["deploy"]["environment"]["name"] == "github-pages"
    assert "github.ref == 'refs/heads/main'" in pages["jobs"]["deploy"]["if"]
    assert "E2E Accuracy Matrix" in pages["on"]["workflow_run"]["workflows"]
    assert "campaign" in branch["jobs"]


# The nightly page selects these names (scripts/pages/prepare_e2e_accuracy_pages.py, agentx_pages.py).
PUBLIC_ARTIFACT = re.compile(r"e2e-(?:accuracy|agentx)-web(?:-[0-9a-f]{16})?")
ARTIFACT_KEY = "0123456789abcdef"


def upload_names(preview: bool) -> list[str]:
    """Every artifact name a run of the branch workflow can upload, rendered for one preview setting."""
    names = []
    for _, _, step in steps("e2e-accuracy-branch.yml"):
        # A conditional upload counts only when its own gate allows it.
        if step.get("uses", "").startswith("actions/upload-artifact@") and expression(
            step.get("if", "true"), inputs={"preview": preview}
        ):
            contexts = {"inputs": {"preview": preview, "artifact_key": ARTIFACT_KEY}, "matrix": {"shard": 0}}
            names.append(render(step["with"]["name"], **contexts))
    return names


def test_no_preview_upload_is_eligible_for_the_nightly_page():
    preview = upload_names(preview=True)
    assert preview
    assert [name for name in preview if PUBLIC_ARTIFACT.fullmatch(name)] == []
    # Positive control: the default-branch run publishes exactly one accuracy and one AgentX web artifact.
    default = upload_names(preview=False)
    assert sorted(name for name in default if PUBLIC_ARTIFACT.fullmatch(name)) == [
        f"e2e-accuracy-web-{ARTIFACT_KEY}",
        f"e2e-agentx-web-{ARTIFACT_KEY}",
    ]
    assert not any("preview" in name for name in default)


def test_upload_name_rendering_would_notice_a_preview_run_taking_the_public_name():
    leaked = render("${{ 'e2e-accuracy-web' }}-${{ inputs.artifact_key }}", inputs={"artifact_key": ARTIFACT_KEY})
    assert PUBLIC_ARTIFACT.fullmatch(leaked)
    gated = "${{ inputs.preview && 'e2e-accuracy-preview' || 'e2e-accuracy-web' }}-${{ inputs.artifact_key }}"
    assert not PUBLIC_ARTIFACT.fullmatch(render(gated, inputs={"preview": True, "artifact_key": ARTIFACT_KEY}))
    assert PUBLIC_ARTIFACT.fullmatch(render(gated, inputs={"preview": False, "artifact_key": ARTIFACT_KEY}))


def test_the_credentialed_dataset_pin_is_read_from_main_and_names_a_supported_layout():
    for _, _, step in steps("e2e-accuracy.yml"):
        if references_secret(step):
            assert "--manifest trusted/.github/e2e-accuracy-dataset.json" in step["run"]
    pin = json.loads((ROOT / ".github/e2e-accuracy-dataset.json").read_text())
    assert pin.get("layout", "legacy-inferencex") in {"legacy-inferencex", "aisim-e2e/v1"}
    assert re.fullmatch(r"[0-9a-f]{40}", pin["revision"])
