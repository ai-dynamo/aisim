# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Approval boundaries beyond the legacy config provenance/integrity check."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.ci import check_config_license_approvals as gate

pytestmark = pytest.mark.unit
PATH = "aisim_core/model_configs/example_config.json"
SHA = "a" * 64
TEXT = "b" * 64
REVISION = "c" * 40


def entry(license_id="LicenseRef-Example", origin="upstream"):
    return {
        "path": PATH,
        "sha256": SHA,
        "license": license_id,
        "origin": origin,
        "notice": {"license_text_sha256": TEXT},
    }


def record(ownership="third_party", underlying=()):
    return {
        "path": PATH,
        "sha256": SHA,
        "ownership": ownership,
        "underlying_licenses": list(underlying),
        "approval": {"status": "pending"},
        "ownership_evidence": {
            "url": f"https://huggingface.co/nvidia/example/blob/{REVISION}/README.md",
            "revision": REVISION,
            "status": "recorded_source",
        },
    }


def approved(item=None, required=("LicenseRef-Example",)):
    item = item or record()
    item["approval"] = {
        "status": "approved",
        "approved_by": "reviewer",
        "approved_on": "2026-01-01",
        "reference": "https://example.com/decisions/config-review-1",
        "scope": "oss-config-redistribution",
        "reported_license": "LicenseRef-Example",
        "required_licenses": sorted(required),
        "license_text_sha256": TEXT,
    }
    return item


def policy(tmp_path, records):
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "releasing_entity": "NVIDIA",
                "approved_standard_licenses": sorted(gate.STANDARD_LICENSES),
                "files": records,
            }
        )
    )
    return path


@pytest.mark.parametrize("license_id", sorted(gate.STANDARD_LICENSES))
def test_standard_license_requires_no_custom_decision(license_id):
    result = gate.assess([entry(license_id)], {PATH: record()})[0]
    assert result["passed"] and result["approval_basis"] == "standard_license_policy"


def test_custom_license_needs_explicit_approval_and_rejection_wins():
    assert gate.assess([entry()], {PATH: record()})[0]["problems"] == ["unapproved_license"]
    assert gate.assess([entry()], {PATH: approved()})[0]["passed"]
    rejected = record()
    rejected["approval"]["status"] = "rejected"
    assert "rejected" in gate.assess([entry("MIT")], {PATH: rejected})[0]["problems"]


@pytest.mark.parametrize("ownership,underlying", [("third_party", ()), ("nvidia_developed", ())])
def test_approval_or_nvidia_ownership_cannot_clear_unknown_provenance(ownership, underlying):
    item = approved(record(ownership, underlying), required=underlying)
    result = gate.assess([entry("UNKNOWN", "legacy_unresolved")], {PATH: item})[0]
    assert not result["passed"] and "unknown_license_or_provenance" in result["problems"]


def test_nvidia_licensor_has_no_outbound_licensee_permission_gate():
    result = gate.assess([entry("LicenseRef-NVIDIA-Internal-RD")], {PATH: record("nvidia_developed")})[0]
    assert result["passed"] and result["approval_basis"] == "nvidia_own_contribution"


def test_nvidia_derivative_retains_unapproved_base_rights():
    item = record("nvidia_derivative", ["LicenseRef-MiniMax-Noncommercial"])
    result = gate.assess([entry("LicenseRef-NVIDIA-Evaluation")], {PATH: item})[0]
    assert not result["passed"] and result["required_licenses"] == ["LicenseRef-MiniMax-Noncommercial"]
    assert "unapproved_license" in result["problems"]


def test_nvidia_derivative_of_standard_base_does_not_need_self_approval():
    item = record("nvidia_derivative", ["MIT"])
    assert gate.assess([entry("LicenseRef-NVIDIA-Open-Model")], {PATH: item})[0]["passed"]


def test_unknown_base_and_unverified_ownership_fail():
    item = record("nvidia_derivative", ["UNKNOWN"])
    item["ownership_evidence"]["status"] = "source_unverified"
    result = gate.assess([entry("MIT")], {PATH: item})[0]
    assert {"unknown_underlying_license", "ownership_source_unverified"} <= set(result["problems"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("sha256", "d" * 64),
        ("reported_license", "MIT"),
        ("required_licenses", ["MIT"]),
        ("license_text_sha256", "e" * 64),
    ],
)
def test_decision_cannot_be_reused_for_changed_bytes_terms_or_scope(field, value):
    item = approved()
    if field == "sha256":
        item[field] = value
    else:
        item["approval"][field] = value
    assert not gate.assess([entry()], {PATH: item})[0]["passed"]


@pytest.mark.parametrize("ownership,underlying", [("nvidia_derivative", []), ("nvidia_developed", ["MIT"])])
def test_invalid_ownership_cannot_discard_declared_base_rights(tmp_path, ownership, underlying):
    with pytest.raises(ValueError):
        gate.load_review(policy(tmp_path, [record(ownership, underlying)]))


@pytest.mark.parametrize("change", ["no_reference", "no_reviewer", "wrong_scope", "missing_text_hash"])
def test_approved_label_alone_is_not_a_decision(tmp_path, change):
    item = approved()
    field = {
        "no_reference": "reference",
        "no_reviewer": "approved_by",
        "wrong_scope": "scope",
        "missing_text_hash": "license_text_sha256",
    }[change]
    item["approval"].pop(field)
    with pytest.raises((ValueError, TypeError)):
        gate.load_review(policy(tmp_path, [item]))


def test_duplicate_reviews_and_non_nvidia_evidence_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        gate.load_review(policy(tmp_path, [record(), record()]))
    item = record("nvidia_developed")
    item["ownership_evidence"]["url"] = f"https://huggingface.co/third-party/example/blob/{REVISION}/README.md"
    with pytest.raises(ValueError, match="pinned evidence"):
        gate.load_review(policy(tmp_path, [item]))


def test_approval_record_loads_and_standard_policy_is_explicit(tmp_path):
    assert gate.load_review(policy(tmp_path, [approved()]))[PATH]["approval"]["status"] == "approved"
    path = policy(tmp_path, [record()])
    document = json.loads(path.read_text())
    document["approved_standard_licenses"].append("LicenseRef-Example")
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="six-license"):
        gate.load_review(path)


def test_cli_failure_keeps_findings_and_missing_inputs_are_not_green(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    review = policy(tmp_path, [record()])
    monkeypatch.setattr(gate.provenance, "validate", lambda *a, **kw: ([entry()], None))
    output = tmp_path / "report"
    argv = ["--root", str(source), "--review", str(review), "--output-dir", str(output)]
    assert gate.main(argv) == 1
    result = json.loads((output / "config-license-approval.json").read_text())
    assert not result["passed"] and result["files"][0]["problems"] == ["unapproved_license"]
    review.unlink()
    assert gate.main(argv) == 1
    assert "error" in json.loads((output / "config-license-approval.json").read_text())


def test_malformed_reviewer_is_a_reported_failure(tmp_path, monkeypatch):
    item = approved()
    item["approval"]["approved_by"] = 123
    review = policy(tmp_path, [item])
    monkeypatch.setattr(gate.provenance, "validate", lambda *a, **kw: ([entry()], None))
    output = tmp_path / "report"
    assert gate.main(["--review", str(review), "--output-dir", str(output)]) == 1
    assert "reviewer" in json.loads((output / "config-license-approval.json").read_text())["error"]


def test_cli_approved_source_succeeds_but_extra_review_paths_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(gate.provenance, "validate", lambda *a, **kw: ([entry()], "f" * 64))
    review = policy(tmp_path, [approved()])
    argv = ["--review", str(review), "--output-dir", str(tmp_path / "output")]
    assert gate.main(argv) == 0
    document = json.loads(review.read_text())
    extra = copy.deepcopy(document["files"][0])
    extra["path"] = "aisim_core/model_configs/stale.json"
    document["files"].append(extra)
    review.write_text(json.dumps(document))
    assert gate.main(argv) == 1


def test_workflows_run_strict_gate_before_dependency_execution_and_keep_failure_report():
    import yaml

    root = Path(__file__).resolve().parents[1]
    for name in ("ci.yml", "nightly-ci.yml"):
        job = yaml.safe_load((root / ".github/workflows" / name).read_text())["jobs"]["python-compliance"]
        steps = job["steps"]
        gate_index = next(i for i, s in enumerate(steps) if "check_config_license_approvals.py" in s.get("run", ""))
        install_index = next(i for i, s in enumerate(steps) if s.get("name") == "Check Python dependency licenses")
        assert gate_index < install_index
        upload = steps[gate_index + 1]
        assert upload["if"] == "always()" and upload["with"]["if-no-files-found"] == "error"
        if name == "nightly-ci.yml":
            assert "--root source" in steps[gate_index]["run"]
        else:
            assert "outputs.application_wheel" not in job["if"]
