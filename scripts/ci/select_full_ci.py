# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the expensive Full CI components required by changed paths.

Pushes to main are batched: after a bounded wait, a commit that main has
already moved past builds, verifies, and stages its wheel only, and the newest
commit runs every component against the last full-scope main commit whose
comparison jobs completed successfully. Reruns always keep the full scope.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath
from urllib.parse import urlencode

COMPONENTS = (
    "readme_commands",
    "platform_wheels",
    "collector_data",
    "prediction_regression",
    "cargo_deny",
    "rust",
    "rust_feature_modes",
    "public_api_rust",
    "application_test_wheel",
    "application_wheel",
    "application_tests",
    "python_compatibility",
    "engine_golden_regression",
    "release_artifact_contract",
)

# Components whose jobs download the per-architecture application test wheel;
# selecting any of them also selects the wheel build.
APPLICATION_TEST_WHEEL_CONSUMERS = {
    "application_wheel",
    "application_tests",
    "release_artifact_contract",
}
# A main commit that a newer main commit has superseded still builds, verifies,
# and stages its own wheel. Its test suite runs with the newer commit.
SUPERSEDED_MAIN_COMPONENTS = {
    "application_test_wheel",
    "application_wheel",
}

# Main batching. Every name below must match .github/workflows/ci.yml: later
# main runs read the recorded scope of earlier runs through the Actions API.
MAIN_DEBOUNCE_SECONDS = 600
MAIN_LOOKBACK_RUNS = 30
WORKFLOW_PATH = ".github/workflows/ci.yml"
SCOPE_JOB = "Select Full CI Scope"
PLAN_STEP = "Plan required Full CI components"
SUPERSEDED_STEP = "Defer the test suite to the newer main commit"
# The reusable prediction workflow exposes its final comparison as a nested
# job name. License evidence also requires both baseline inventory producers.
COMPARISON_JOBS = {
    "Prediction Regression / Prediction Regression Gate",
    "Python Dependency Licenses",
    "Cargo Dependency Inventories",
    "License compliance evidence",
}
FULL_SHA = re.compile(r"[0-9a-f]{40}")

PYTHON_PACKAGE_COMPONENTS = {
    "readme_commands",
    "platform_wheels",
    "application_wheel",
    "application_tests",
    "python_compatibility",
    "release_artifact_contract",
}
# Rust is the implementation behind the Python package and prediction engine,
# so Rust changes exercise every compiled and consumer boundary, including the
# collector validator's maturin install. Cargo policy is added separately for manifest and
# dependency changes.
RUST_COMPONENTS = {
    "readme_commands",
    "platform_wheels",
    "collector_data",
    "prediction_regression",
    "rust",
    "rust_feature_modes",
    "public_api_rust",
    "application_wheel",
    "application_tests",
    "python_compatibility",
    "engine_golden_regression",
    "release_artifact_contract",
}
# Prediction inputs and implementations must exercise both before/after output
# comparison and the lower-level engine parity goldens.
PREDICTION_COMPONENTS = {
    "prediction_regression",
    "engine_golden_regression",
}
# Collector source and performance data affect the data validator as well as
# the prediction and parity consumers that read the resulting records.
COLLECTOR_COMPONENTS = {
    "collector_data",
    "prediction_regression",
    "engine_golden_regression",
}

ROOT_DOCUMENTATION = {
    "AGENTS.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "CONTRIBUTORS.md",
    "DEVELOPMENT.md",
    "README.md",
    "REVIEW.md",
}
ROOT_POLICY_ONLY = {
    ".coderabbit.yaml",
    ".gitignore",
    "CODEOWNERS",
}
PACKAGE_CONTENT_FILES = {
    ".dockerignore",
    ".gitattributes",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "python/aisim/.dockerignore",
    "python/aisim/.gitattributes",
    "python/aisim/THIRD_PARTY_NOTICES.md",
}
PYTHON_DEPENDENCY_FILES = {
    "python/aisim/pyproject.toml",
    "python/aisim/uv.lock",
}
RUST_DEPENDENCY_FILES = {
    "Cargo.lock",
    "Cargo.toml",
    "deny.toml",
}


def _normalize(paths: Iterable[str]) -> list[str]:
    normalized = []
    for raw in paths:
        path = raw
        if not path:
            continue
        pure = PurePosixPath(path)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or any(ord(character) < 32 or ord(character) == 127 for character in path)
        ):
            raise ValueError(f"changed path is not repository-relative: {raw!r}")
        normalized.append(str(pure))
    return sorted(set(normalized))


def _is_documentation(path: str) -> bool:
    return path in ROOT_DOCUMENTATION or path.startswith("docs/")


def _is_policy_only(path: str) -> bool:
    return (
        path in ROOT_POLICY_ONLY
        or path == ".github/pull_request_template.md"
        or path.startswith(".github/ISSUE_TEMPLATE/")
        or path.startswith(".github/codeowners/")
    )


def _is_prediction_path(path: str) -> bool:
    return path.startswith(
        (
            "crates/core/parity_tests/perfmodel/",
            "crates/core/perfmodel/",
            "crates/core/src/engine/",
            "crates/core/src/perfmodel/",
            "crates/core/tests/perfmodel/",
            "python/aisim/src/aisim/sdk/",
            "python/aisim/src/aisim/generator/",
            "python/aisim/src/aisim/legacy_cli/",
            "python/aisim/src/aisim/fpm_contract.py",
            "python/aisim/src/aisim/logging_utils.py",
            "python/aisim/src/aisim_core/",
            "python/aisim/src/aisim/capacity.py",
            "python/aisim/tools/accuracy_regression_testing/",
            "python/aisim/tools/accuracy_tracking/",
            "python/aisim/tools/prediction_regression_gate/",
            "tests/test_aic.py",
        )
    )


def _is_collector_path(path: str) -> bool:
    return path.startswith(
        (
            "python/aisim/collector/",
            "python/aisim/tools/perf_database/",
            "python/aisim/tests/unit/collector/",
            "python/aisim/docs/perf_database/",
        )
    ) or path.startswith("python/aisim/src/aisim_core/systems/data/")


def _all(reason: str, paths: list[str]) -> dict[str, object]:
    return {
        "run_all": True,
        "reason": reason,
        "paths": paths,
        "components": dict.fromkeys(COMPONENTS, True),
    }


def select_components(
    paths: Iterable[str],
    *,
    force_all: bool = False,
    superseded: bool = False,
) -> dict[str, object]:
    """Return a fail-closed Full CI plan for repository-relative paths.

    ``superseded`` is only set for a push to main that main has already moved
    past; it replaces the forced full scope with the wheel build and staging.
    """

    changed = _normalize(paths)
    if superseded:
        return {
            "run_all": False,
            "reason": "main moved past this commit; build, verify, and stage its wheel only",
            "paths": changed,
            "components": {component: component in SUPERSEDED_MAIN_COMPONENTS for component in COMPONENTS},
        }
    if force_all:
        return _all("full matrix required by workflow event", changed)
    if not changed:
        return _all("changed-path set is empty", changed)

    selected: set[str] = set()
    unknown: list[str] = []
    reasons: set[str] = set()

    for path in changed:
        if path.startswith((".github/workflows/", ".github/actions/")) or path in {
            ".github/actionlint.yaml",
            ".github/copy-pr-bot.yaml",
            "scripts/ci/select_full_ci.py",
            "tests/test_ci_workflow_contracts.py",
        }:
            return _all(f"CI execution contract changed: {path!r}", changed)

        if path == "README.md":
            selected.add("readme_commands")
            reasons.add("executable README")
            continue

        if _is_documentation(path) or _is_policy_only(path):
            reasons.add("documentation or review policy")
            continue

        if path in PYTHON_DEPENDENCY_FILES:
            return _all(f"Python dependency contract changed: {path!r}", changed)

        if path == "python/aisim/pytest.ini":
            return _all(f"shared test configuration changed: {path!r}", changed)

        if path in PACKAGE_CONTENT_FILES or path.startswith("python/aisim/docker/"):
            selected.update(PYTHON_PACKAGE_COMPONENTS)
            reasons.add("package or distribution metadata")
            continue

        if path in RUST_DEPENDENCY_FILES or (path.startswith("crates/") and path.endswith(("Cargo.toml", "deny.toml"))):
            selected.update(RUST_COMPONENTS)
            selected.add("cargo_deny")
            reasons.add("Rust dependency or package contract")
            continue

        if path.startswith("crates/"):
            selected.update(RUST_COMPONENTS)
            reasons.add("Rust implementation or tests")
            continue

        if path.startswith("python/aisim/"):
            selected.update(PYTHON_PACKAGE_COMPONENTS)
            reasons.add("Python package implementation or tests")
            if _is_prediction_path(path):
                selected.update(PREDICTION_COMPONENTS)
                reasons.add("prediction or engine behavior")
            if _is_collector_path(path):
                selected.update(COLLECTOR_COMPONENTS)
                reasons.add("collector or performance data")
            continue

        if path.startswith("tests/"):
            selected.add("application_tests")
            reasons.add("repository application tests")
            if _is_prediction_path(path):
                selected.update(PREDICTION_COMPONENTS)
            continue

        if path.startswith("examples/"):
            selected.add("application_tests")
            reasons.add("executable examples")
            continue

        if path.startswith("scripts/") or path in {"pytest.ini", "SECURITY.md"}:
            return _all(f"shared repository contract changed: {path!r}", changed)

        unknown.append(path)

    if unknown:
        return _all(f"unclassified path: {unknown[0]!r}", changed)

    if selected & APPLICATION_TEST_WHEEL_CONSUMERS:
        selected.add("application_test_wheel")
    reason = ", ".join(sorted(reasons)) if reasons else "documentation-only change"
    return {
        "run_all": False,
        "reason": reason,
        "paths": changed,
        "components": {component: component in selected for component in COMPONENTS},
    }


class GitHubError(RuntimeError):
    """GitHub metadata needed for main batching could not be read."""


def github_api(endpoint: str) -> dict:
    """Read one GitHub API response with the job's read-only token."""
    try:
        result = subprocess.run(
            ["gh", "api", "-X", "GET", endpoint],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        response = json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise GitHubError(f"GitHub API request failed for {endpoint}") from error
    if not isinstance(response, dict):
        raise GitHubError(f"GitHub API returned an unexpected response for {endpoint}")
    return response


def main_scope_decision(jobs_page: dict) -> str:
    """Classify the scope an earlier main push recorded: full, superseded, or undecided."""
    scope_jobs = [job for job in jobs_page.get("jobs") or [] if job.get("name") == SCOPE_JOB]
    if len(scope_jobs) != 1 or (scope_jobs[0].get("status"), scope_jobs[0].get("conclusion")) != (
        "completed",
        "success",
    ):
        # Still waiting, failed, or missing: it has not run the full suite.
        return "undecided"
    steps = scope_jobs[0].get("steps") or []
    plans = [step for step in steps if step.get("name") == PLAN_STEP]
    markers = [step for step in steps if step.get("name") == SUPERSEDED_STEP]
    if len(plans) != 1 or plans[0].get("conclusion") != "success":
        return "undecided"
    if not markers:
        # Runs created before main batching always selected the full scope.
        return "full"
    if len(markers) == 1 and markers[0].get("status") == "completed":
        return {"skipped": "full", "success": "superseded"}.get(markers[0].get("conclusion"), "undecided")
    return "undecided"


def _main_run_jobs(repository: str, run_id: int, *, api: Callable[[str], dict]) -> dict:
    """Read all latest-attempt job results, including nested comparison jobs."""
    endpoint = f"repos/{repository}/actions/runs/{run_id}/jobs?filter=latest&per_page=100"
    jobs = []
    page = 1
    while True:
        results = api(endpoint if page == 1 else f"{endpoint}&page={page}").get("jobs")
        if not isinstance(results, list):
            raise GitHubError(f"GitHub API returned no Full CI jobs for run {run_id}")
        jobs.extend(results)
        if len(results) < 100:
            return {"jobs": jobs}
        page += 1


def _main_comparisons_succeeded(jobs_page: dict) -> bool:
    for name in COMPARISON_JOBS:
        matches = [job for job in jobs_page["jobs"] if job.get("name") == name]
        if len(matches) != 1 or (matches[0].get("status"), matches[0].get("conclusion")) != (
            "completed",
            "success",
        ):
            return False
    return True


def last_full_scope_commit(
    repository: str,
    sha: str,
    run_id: int,
    parent: str,
    *,
    api: Callable[[str], dict] = github_api,
) -> tuple[str, list[tuple[str, str]]]:
    """Return the newest earlier full-scope main commit with successful comparisons.

    Superseded, still-undecided, and unvalidated comparison runs are skipped.
    Without a validated full-scope run in the bounded window, compare with the
    previous main commit instead.
    """
    query = urlencode({"branch": "main", "event": "push", "per_page": MAIN_LOOKBACK_RUNS})
    runs = api(f"repos/{repository}/actions/workflows/ci.yml/runs?{query}").get("workflow_runs")
    if not isinstance(runs, list):
        raise GitHubError("GitHub API returned no main Full CI run list")
    earlier = [
        run
        for run in runs
        if type(run.get("id")) is int
        and run["id"] < run_id
        and run.get("path") == WORKFLOW_PATH
        and run.get("event") == "push"
        and run.get("head_branch") == "main"
        and FULL_SHA.fullmatch(str(run.get("head_sha", "")))
        and run["head_sha"] != sha
    ]
    deferred = []
    for run in sorted(earlier, key=lambda run: run["id"], reverse=True):
        jobs_page = _main_run_jobs(repository, run["id"], api=api)
        decision = main_scope_decision(jobs_page)
        if decision == "full" and not _main_comparisons_succeeded(jobs_page):
            decision = "comparisons not successful"
        if decision != "full":
            deferred.append(f"`{run['head_sha']}` ({decision}, run {run['id']})")
            continue
        base = run["head_sha"]
        status = api(f"repos/{repository}/compare/{base}...{sha}?per_page=1").get("status")
        if status != "ahead":
            return parent, [
                (
                    "warning",
                    f"Full-scope main commit `{base}` is not an ancestor of this commit "
                    f"(compare status {status!r}); comparing with the previous main commit `{parent}`.",
                )
            ]
        notes = [
            (
                "notice",
                f"Comparison base `{base}`: the last full-scope main commit with successful comparisons "
                f"(run {run['id']}).",
            )
        ]
        if deferred:
            notes.append(
                ("notice", "This run also validates the main commits deferred to it: " + ", ".join(deferred) + ".")
            )
        return base, notes
    return parent, [
        (
            "warning",
            f"No full-scope Full CI run with successful comparisons among the last {MAIN_LOOKBACK_RUNS} main pushes; "
            f"comparing with the previous main commit `{parent}`.",
        )
    ]


def resolve_main_push(
    repository: str,
    sha: str,
    run_id: int,
    parent: str,
    *,
    run_attempt: int = 1,
    api: Callable[[str], dict] = github_api,
    sleep: Callable[[float], None] = time.sleep,
    debounce: float = MAIN_DEBOUNCE_SECONDS,
) -> dict[str, object]:
    """Wait for newer merges; only first attempts may defer their full scope."""
    if (
        not re.fullmatch(r"[\w.-]+/[\w.-]+", repository)
        or not FULL_SHA.fullmatch(sha)
        or not FULL_SHA.fullmatch(parent)
        or run_id <= 0
        or run_attempt <= 0
    ):
        raise ValueError("Main batching requires owner/name, a positive run ID/attempt, and full commit SHAs")
    sleep(debounce)
    notes: list[tuple[str, str]] = []
    main_sha = ""
    try:
        main_sha = api(f"repos/{repository}/commits/main").get("sha", "")
        if not isinstance(main_sha, str) or not FULL_SHA.fullmatch(main_sha):
            raise GitHubError("main did not resolve to a full commit SHA")
        status = (
            "identical"
            if main_sha == sha
            else api(f"repos/{repository}/compare/{sha}...{main_sha}?per_page=1").get("status")
        )
    except GitHubError as error:
        status = "unavailable"
        notes.append(("warning", f"Could not compare this commit with main ({error}); running the full scope."))
    if status == "ahead" and run_attempt == 1:
        return {
            "superseded": True,
            "main_sha": main_sha,
            "comparison_base": parent,
            "notes": [
                (
                    "notice",
                    f"main moved to `{main_sha}`, which contains this commit. This run builds, verifies, "
                    "and stages this commit's wheel; its full test suite runs with the newer main commit.",
                )
            ],
        }
    if run_attempt > 1:
        notes.append(("notice", f"Run attempt {run_attempt} keeps the full scope to retry every required component."))
    if status not in {"ahead", "identical", "unavailable"}:
        notes.append(
            (
                "warning",
                f"main is at `{main_sha}`, which does not contain this commit "
                f"(compare status {status!r}); running the full scope.",
            )
        )
    base, base_notes = last_full_scope_commit(repository, sha, run_id, parent, api=api)
    return {"superseded": False, "main_sha": main_sha, "comparison_base": base, "notes": notes + base_notes}


def _write_github_output(path: Path, plan: dict[str, object], *, superseded: bool, comparison_base: str) -> None:
    components = plan["components"]
    assert isinstance(components, dict)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"run_all={str(plan['run_all']).lower()}\n")
        for component in COMPONENTS:
            handle.write(f"{component}={str(components[component]).lower()}\n")
        handle.write(f"superseded={str(superseded).lower()}\n")
        handle.write(f"comparison_base={comparison_base}\n")


def _write_summary(path: Path, plan: dict[str, object], batch: dict[str, object] | None) -> None:
    components = plan["components"]
    assert isinstance(components, dict)
    selected = [name for name in COMPONENTS if components[name]]
    skipped = [name for name in COMPONENTS if not components[name]]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("### Full CI selection\n\n")
        handle.write(f"Reason: {plan['reason']}\n\n")
        handle.write(f"Selected: {', '.join(selected) if selected else 'none'}\n\n")
        handle.write(f"Explicitly N/A: {', '.join(skipped) if skipped else 'none'}\n")
        if batch is not None:
            handle.write(
                f"\n### Main batching\n\nWaited {MAIN_DEBOUNCE_SECONDS // 60} minutes for newer main merges.\n\n"
            )
            for _, note in batch["notes"]:
                handle.write(f"- {note}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base64-paths-file", type=Path)
    parser.add_argument("--force-all", action="store_true")
    parser.add_argument("--comparison-base", default="", help="Comparison commit from exact-target verification")
    parser.add_argument("--main-push", action="store_true", help="Batch post-merge test suites for a push to main")
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--sha", default=os.environ.get("GITHUB_SHA", ""))
    parser.add_argument("--run-id", type=int, default=int(os.environ.get("GITHUB_RUN_ID") or 0))
    parser.add_argument("--run-attempt", type=int, default=int(os.environ.get("GITHUB_RUN_ATTEMPT") or 1))
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    paths = []
    if args.base64_paths_file:
        encoded_paths = args.base64_paths_file.read_text(encoding="ascii").splitlines()
        paths = [base64.b64decode(encoded, validate=True).decode("utf-8") for encoded in encoded_paths]
    batch = None
    if args.main_push:
        try:
            batch = resolve_main_push(
                args.repository,
                args.sha,
                args.run_id,
                args.comparison_base,
                run_attempt=args.run_attempt,
                api=github_api,
                sleep=time.sleep,
            )
        except (GitHubError, ValueError) as error:
            raise SystemExit(f"::error::{error}") from error
        for level, note in batch["notes"]:
            print(f"::{level}::{note}")
    comparison_base = str(batch["comparison_base"]) if batch else args.comparison_base
    # Downstream checkouts treat an empty ref as the target itself, so the
    # workflow output must always name an exact comparison commit.
    if (args.github_output or comparison_base) and not FULL_SHA.fullmatch(comparison_base):
        raise SystemExit("::error::comparison base must be a full commit SHA")
    superseded = bool(batch and batch["superseded"])
    plan = select_components(paths, force_all=args.force_all, superseded=superseded)
    print(json.dumps({**plan, "superseded": superseded, "comparison_base": comparison_base}, indent=2, sort_keys=True))
    if args.github_output:
        _write_github_output(args.github_output, plan, superseded=superseded, comparison_base=comparison_base)
    if args.summary:
        _write_summary(args.summary, plan, batch)


if __name__ == "__main__":
    main()
