# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Approval boundaries beyond the legacy config provenance/integrity check."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import zipfile
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
        "source": {
            "url": f"https://huggingface.co/nvidia/example/blob/{REVISION}/config.json",
            "revision": REVISION,
            "path": "config.json",
        },
    }


def record(ownership="third_party", underlying=(), contribution=()):
    return {
        "path": PATH,
        "sha256": SHA,
        "ownership": ownership,
        "underlying_licenses": list(underlying),
        "nvidia_contribution_licenses": list(contribution),
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
        "path": item["path"],
        "sha256": item["sha256"],
        "approved_by": "reviewer",
        "approved_on": "2026-01-01",
        "reference": "https://example.com/decisions/config-review-1",
        "scope": "oss-config-redistribution",
        "reported_license": "LicenseRef-Example",
        "required_licenses": sorted(required),
        "license_text_sha256": TEXT,
        "license_text_sha256_by_license": {
            license_id: TEXT for license_id in required if license_id not in gate.STANDARD_LICENSES
        },
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


@pytest.mark.parametrize("expression", ["MIT AND ISC", "MIT OR ISC"])
def test_third_party_expressions_still_need_explicit_decisions(expression):
    result = gate.assess([entry(expression)], {PATH: record()})[0]
    assert result["required_licenses"] == [expression]
    assert not result["passed"] and "unapproved_license" in result["problems"]


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
    item = record("nvidia_developed", contribution=["LicenseRef-NVIDIA-Internal-RD"])
    result = gate.assess([entry("LicenseRef-NVIDIA-Internal-RD")], {PATH: item})[0]
    assert result["passed"] and result["approval_basis"] == "nvidia_own_contribution"


def test_nvidia_derivative_retains_unapproved_base_rights():
    item = record("nvidia_derivative", ["LicenseRef-MiniMax-Noncommercial"], ["LicenseRef-NVIDIA-Evaluation"])
    result = gate.assess([entry("LicenseRef-NVIDIA-Evaluation")], {PATH: item})[0]
    assert not result["passed"] and result["required_licenses"] == ["LicenseRef-MiniMax-Noncommercial"]
    assert "unapproved_license" in result["problems"]


def test_nvidia_derivative_of_standard_base_does_not_need_self_approval():
    item = record("nvidia_derivative", ["MIT"], ["LicenseRef-NVIDIA-Open-Model"])
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


@pytest.mark.parametrize(
    "change", ["no_reference", "no_reviewer", "wrong_scope", "missing_text_hash", "no_path", "no_hash", "no_text_map"]
)
def test_approved_label_alone_is_not_a_decision(tmp_path, change):
    item = approved()
    field = {
        "no_reference": "reference",
        "no_reviewer": "approved_by",
        "wrong_scope": "scope",
        "missing_text_hash": "license_text_sha256",
        "no_path": "path",
        "no_hash": "sha256",
        "no_text_map": "license_text_sha256_by_license",
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


def test_cli_corrupt_wheel_retains_failure_report(tmp_path):
    wheel = tmp_path / "invalid.whl"
    wheel.write_bytes(b"not a ZIP archive")
    output = tmp_path / "report"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(gate.__file__)),
            "--root",
            str(gate.provenance.ROOT),
            "--wheel",
            str(wheel),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    report = json.loads((output / "config-license-approval.json").read_text())
    assert report["passed"] is False
    assert report["legal_approval_asserted"] is False
    assert "zip" in report["error"].lower()
    assert "Config license compliance failed" in result.stdout


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
    extra["approval"]["path"] = extra["path"]
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
        assert upload["with"]["path"] == "${{ runner.temp }}/config-license-approval/config-license-approval.json"
        if name == "nightly-ci.yml":
            assert "--root source" in steps[gate_index]["run"]
        else:
            assert "outputs.application_wheel" not in job["if"]


@pytest.mark.parametrize("change", ["repository", "revision", "missing_source"])
def test_unrelated_nvidia_card_cannot_exempt_config(tmp_path, change):
    item = record("nvidia_developed", contribution=["LicenseRef-Example"])
    current = entry()
    if change == "missing_source":
        current.pop("source")
    elif change == "repository":
        current["source"]["url"] = current["source"]["url"].replace("nvidia/example", "Qwen/Qwen3.8-Max")
    else:
        current["source"]["revision"] = "d" * 40
        current["source"]["url"] = current["source"]["url"].replace(REVISION, "d" * 40)
    records = gate.load_review(policy(tmp_path, [item]))
    result = gate.assess([current], records)[0]
    assert not result["passed"] and "ownership_source_mismatch" in result["problems"]


@pytest.mark.parametrize("change", [None, "path", "sha256", "source", "reference"])
def test_cross_source_ownership_requires_exact_per_file_mapping(tmp_path, change):
    current = entry()
    current["source"]["url"] = current["source"]["url"].replace("nvidia/example", "example/original")
    item = record("nvidia_developed", contribution=["LicenseRef-Example"])
    mapping = {
        "path": PATH,
        "sha256": SHA,
        "source": copy.deepcopy(current["source"]),
        "reference": "https://example.com/ownership/file-mapping",
        "explanation": "Synthetic mapping for the config published separately from its model card.",
    }
    if change:
        mapping[change] = "unrelated"
    item["ownership_evidence"]["config_mapping"] = mapping
    result = gate.assess([current], gate.load_review(policy(tmp_path, [item])))[0]
    assert result["passed"] is (change is None)
    if change:
        assert "ownership_source_mismatch" in result["problems"]


@pytest.mark.parametrize(
    "current,item",
    [
        (
            entry("LicenseRef-GLM AND LicenseRef-NVIDIA-Export"),
            record("nvidia_derivative", ["MIT"], ["LicenseRef-NVIDIA-Export"]),
        ),
        (entry("MIT"), record("third_party", ["LicenseRef-Base"])),
        (entry("MIT"), record("nvidia_derivative", ["MIT"], ["LicenseRef-NVIDIA-Export"])),
        (entry("LicenseRef-Example"), record("nvidia_developed")),
    ],
)
def test_inventory_and_declared_obligations_cannot_replace_each_other(tmp_path, current, item):
    result = gate.assess([current], gate.load_review(policy(tmp_path, [item])))[0]
    assert not result["passed"] and "inconsistent_license_obligations" in result["problems"]
    assert (gate.license_terms(current["license"]) - set(item["nvidia_contribution_licenses"])) <= set(
        result["required_licenses"]
    )
    assert set(item["underlying_licenses"]) <= set(result["required_licenses"])


@pytest.mark.parametrize("change", ["path", "bytes"])
def test_provenance_refresh_does_not_renew_old_approval(licensed_source, tmp_path, change):
    root, config, current, item, _, save = licensed_source
    if change == "path":
        current["path"] = item["path"] = "aisim_core/model_configs/another/example_config.json"
        moved = root / "python/aisim/src" / current["path"]
        moved.parent.mkdir()
        config.rename(moved)
    else:
        config.write_text('{"model_type": "updated-synthetic"}\n')
        current["sha256"] = item["sha256"] = gate.provenance.digest(config.read_bytes())
    save()
    gate.provenance.validate(root, root / gate.provenance.INVENTORY)
    output = tmp_path / "report"
    assert gate.main(["--root", str(root), "--output-dir", str(output)]) == 1
    report = json.loads((output / "config-license-approval.json").read_text())
    if change == "path":
        assert "bind path" in report["error"]
    else:
        assert "stale_approval_scope_or_license_text" in report["files"][0]["problems"]


@pytest.fixture
def licensed_source(tmp_path):
    """A complete synthetic source, ledger and legal payload for real CLI checks."""
    root = tmp_path / "source"
    config = root / "python/aisim/src" / PATH
    config.parent.mkdir(parents=True)
    config.write_text('{"model_type": "synthetic"}\n')
    (root / "LICENSE").write_text("Synthetic project license.\n")
    texts = {"LicenseRef-Example": "Synthetic export license. Copyright Example Owner."}
    current = entry()
    current.update(sha256=gate.provenance.digest(config.read_bytes()), copyright_owner="Example Owner", modified=False)
    current["notice"] = {
        "file": "THIRD_PARTY_NOTICES.md",
        "section": "Synthetic configuration",
        "license_text_sha256": gate.provenance.text_digest(texts["LicenseRef-Example"]),
    }
    item = approved()
    item["sha256"] = item["approval"]["sha256"] = current["sha256"]
    item["approval"]["license_text_sha256"] = current["notice"]["license_text_sha256"]
    item["approval"]["license_text_sha256_by_license"] = {
        "LicenseRef-Example": current["notice"]["license_text_sha256"]
    }
    inventory = {"schema_version": 1, "licenses": {}, "files": [current]}
    inventory_path = root / gate.provenance.INVENTORY
    inventory_path.parent.mkdir(parents=True)

    def save():
        (root / "THIRD_PARTY_NOTICES.md").write_text(
            f"## Synthetic configuration\n\nexample_config.json {REVISION} Example Owner\n\n"
            + "\n\n".join(f"```text\n{text}\n```" for text in texts.values())
            + "\n"
        )
        inventory["licenses"] = {
            license_id: {"license_text_sha256": [gate.provenance.text_digest(text)]}
            for license_id, text in texts.items()
        }
        inventory_path.write_text(json.dumps(inventory))
        (root / gate.REVIEW).write_text(policy(tmp_path, [item]).read_text())

    save()
    return root, config, current, item, texts, save


@pytest.mark.parametrize("change", [None, "missing_notice", "wrong_text", "missing_text", "changed_text"])
def test_each_nonstandard_base_text_is_validated_and_bound(licensed_source, tmp_path, change):
    root, _, current, item, texts, save = licensed_source
    bases = ["LicenseRef-Base-A", "LicenseRef-Base-B"]
    item.update(
        ownership="nvidia_derivative", underlying_licenses=bases, nvidia_contribution_licenses=["LicenseRef-Example"]
    )
    item["license_notices"] = {}
    item["approval"]["required_licenses"] = bases
    item["approval"]["license_text_sha256_by_license"] = {}
    for base in bases:
        texts[base] = f"Synthetic {base} license. Copyright Example Owner."
        text_hash = gate.provenance.text_digest(texts[base])
        item["license_notices"][base] = current["notice"] | {"license_text_sha256": text_hash}
        item["approval"]["license_text_sha256_by_license"][base] = text_hash
    if change == "missing_notice":
        del item["license_notices"][bases[-1]]
    elif change == "wrong_text":
        item["license_notices"][bases[-1]]["license_text_sha256"] = current["notice"]["license_text_sha256"]
    elif change == "changed_text":
        texts[bases[-1]] += " New condition."
        item["license_notices"][bases[-1]]["license_text_sha256"] = gate.provenance.text_digest(texts[bases[-1]])
    save()
    if change == "missing_text":
        notices = root / "THIRD_PARTY_NOTICES.md"
        notices.write_text(notices.read_text().replace(texts[bases[-1]], "Unrelated terms."))
    output = tmp_path / "report"
    result = gate.main(["--root", str(root), "--output-dir", str(output)])
    report = json.loads((output / "config-license-approval.json").read_text())
    assert result == (0 if change is None else 1)
    assert report["legal_approval_asserted"] is False
    if change:
        expected = (
            "stale_approval_scope_or_license_text"
            if change == "changed_text"
            else "missing_or_invalid_required_license_text"
        )
        assert expected in report["files"][0]["problems"]


@pytest.mark.parametrize("mismatch", ["missing", "extra", "both"])
def test_cli_names_missing_and_extra_review_paths(licensed_source, tmp_path, mismatch):
    root, _, _, item, _, _ = licensed_source
    extra = "aisim_core/model_configs/extra_config.json"
    records = [] if mismatch in {"missing", "both"} else [item]
    if mismatch in {"extra", "both"}:
        another = copy.deepcopy(item)
        another["path"] = another["approval"]["path"] = extra
        records.append(another)
    review = policy(tmp_path, records)
    output = tmp_path / "report"
    assert gate.main(["--root", str(root), "--review", str(review), "--output-dir", str(output)]) == 1
    report = json.loads((output / "config-license-approval.json").read_text())
    assert report["missing_review_paths"] == ([PATH] if mismatch in {"missing", "both"} else [])
    assert report["extra_review_paths"] == ([extra] if mismatch in {"extra", "both"} else [])
    for name in report["missing_review_paths"] + report["extra_review_paths"]:
        assert name in report["error"]


def test_minimax_unknown_base_cannot_be_approved_after_export_provenance_is_fixed(tmp_path):
    records = gate.load_review(gate.provenance.ROOT / gate.REVIEW)
    selected = [
        item
        for item in records.values()
        if item["underlying_licenses"] == ["UNKNOWN"] and "MiniMax-M2.5" in item["path"]
    ]
    assert len(selected) == 2
    for original in selected:
        item = approved(copy.deepcopy(original), required=["UNKNOWN"])
        item["nvidia_contribution_licenses"] = ["LicenseRef-Example"]
        current = entry()
        current.update(path=item["path"], sha256=item["sha256"])
        current["source"]["url"] = original["ownership_evidence"]["url"].replace("README.md", "config.json")
        current["source"]["revision"] = original["ownership_evidence"]["revision"]
        records = gate.load_review(policy(tmp_path, [item]))
        result = gate.assess([current], records)[0]
        assert not result["passed"] and "unknown_underlying_license" in result["problems"]


def test_cli_wheel_digest_survives_missing_review(licensed_source, tmp_path):
    root, config, _, _, _, _ = licensed_source
    wheel = tmp_path / "synthetic.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(PATH, config.read_bytes())
        for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
            archive.writestr(f"aisim-1.dist-info/licenses/{name}", (root / name).read_bytes())
    (root / gate.REVIEW).unlink()
    output = tmp_path / "report"
    assert gate.main(["--root", str(root), "--wheel", str(wheel), "--output-dir", str(output)]) == 1
    report = json.loads((output / "config-license-approval.json").read_text())
    assert report["wheel_sha256"] == gate.provenance.digest(wheel.read_bytes())
    assert "error" in report and report["passed"] is False
