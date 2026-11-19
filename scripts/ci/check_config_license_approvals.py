#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail the configuration compliance gate on unknown or unapproved licenses.

Provenance integrity, ownership, license policy, and an explicit approval are
separate inputs. Neither a frozen legacy exception nor NVIDIA publication
establishes third-party approval. This command never infers legal clearance.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

if __package__:
    from . import check_bundled_configs as provenance
else:
    import check_bundled_configs as provenance


REVIEW = Path("scripts/ci/config_license_review.json")
STANDARD_LICENSES = {"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Zlib"}
OWNERSHIP = {"third_party", "nvidia_developed", "nvidia_derivative"}


def https_reference(value: object) -> bool:
    if not isinstance(value, str):
        return False
    url = urlsplit(value)
    return bool(url.scheme == "https" and url.hostname and not url.username and not url.password)


def load_review(path: Path) -> dict:
    review = provenance.load_json(path.read_bytes())
    if review.get("schema_version") != 1 or review.get("releasing_entity") != "NVIDIA":
        raise ValueError("unsupported config-license review schema or releasing entity")
    if review.get("approved_standard_licenses") != sorted(STANDARD_LICENSES):
        raise ValueError("standard-license policy must match the reviewed six-license set")
    if not isinstance(review.get("files"), list):
        raise ValueError("missing per-file config-license review")
    files = {}
    for record in review["files"]:
        if not isinstance(record, dict):
            raise ValueError("review file entries must be objects")
        name = provenance.relative_path(record.get("path"))
        if name in files:
            raise ValueError(f"duplicate review record: {name}")
        if not re.fullmatch(r"[a-f0-9]{64}", str(record.get("sha256"))):
            raise ValueError(f"invalid review hash: {name}")
        if record.get("ownership") not in OWNERSHIP:
            raise ValueError(f"invalid ownership classification: {name}")
        licenses = record.get("underlying_licenses")
        if not isinstance(licenses, list) or any(not isinstance(x, str) or not x.strip() for x in licenses):
            raise ValueError(f"invalid underlying license list: {name}")
        if record["ownership"] == "nvidia_developed" and licenses:
            raise ValueError(f"NVIDIA-developed record has third-party base licenses: {name}")
        if record["ownership"] == "nvidia_derivative" and not licenses:
            raise ValueError(f"NVIDIA derivative must retain base licenses: {name}")
        if record["ownership"] != "third_party":
            evidence = record.get("ownership_evidence", {})
            if not isinstance(evidence, dict):
                raise ValueError(f"invalid ownership evidence: {name}")
            if evidence.get("status") not in {"recorded_source", "source_unverified"}:
                raise ValueError(f"invalid ownership evidence status: {name}")
            source = evidence.get("url", "")
            revision = evidence.get("revision", "")
            pinned = (
                https_reference(source)
                and urlsplit(source).hostname == "huggingface.co"
                and urlsplit(source).path.startswith("/nvidia/")
                and re.fullmatch(r"[a-f0-9]{40}", revision)
                and f"/blob/{revision}/README.md" in urlsplit(source).path
            )
            # A declaration of an unavailable source cannot pass the gate.
            if not pinned and evidence.get("status") != "source_unverified":
                raise ValueError(f"NVIDIA ownership needs pinned evidence: {name}")
        approval = record.get("approval", {})
        if not isinstance(approval, dict):
            raise ValueError(f"invalid approval record: {name}")
        if approval.get("status") not in {"pending", "approved", "rejected"}:
            raise ValueError(f"missing or invalid approval status: {name}")
        if approval["status"] == "approved":
            reviewer = approval.get("approved_by")
            if not https_reference(approval.get("reference")) or not isinstance(reviewer, str) or not reviewer.strip():
                raise ValueError(f"approval needs decision reference and reviewer: {name}")
            approved_on = date.fromisoformat(approval.get("approved_on", ""))
            if approved_on > date.today():
                raise ValueError(f"approval date is in the future: {name}")
            if approval.get("scope") != "oss-config-redistribution":
                raise ValueError(f"approval scope is not config redistribution: {name}")
            if (
                not isinstance(approval.get("reported_license"), str)
                or not approval["reported_license"].strip()
                or not isinstance(approval.get("required_licenses"), list)
                or any(not isinstance(x, str) or not x for x in approval["required_licenses"])
                or not re.fullmatch(r"[a-f0-9]{64}", str(approval.get("license_text_sha256")))
            ):
                raise ValueError(f"approval must bind the reported license, required licenses and text hash: {name}")
        files[name] = record
    return files


def assess(entries: list[dict], records: dict) -> list[dict]:
    """Evaluate declarations after the source/wheel provenance check succeeds."""
    results = []
    for entry in entries:
        name = entry["path"]
        record = records.get(name)
        problems = []
        required = [entry["license"]]
        approval_basis = "none"
        if not record:
            problems.append("missing_review")
        elif record["sha256"] != entry["sha256"]:
            problems.append("stale_review_hash")
        else:
            ownership = record["ownership"]
            if ownership != "third_party":
                required = record["underlying_licenses"]
                if record["ownership_evidence"].get("status") == "source_unverified":
                    problems.append("ownership_source_unverified")
            if record["approval"]["status"] == "rejected":
                problems.append("rejected")
            if entry["origin"] == "legacy_unresolved" or entry["license"] in {"UNKNOWN", "NOASSERTION", ""}:
                problems.append("unknown_license_or_provenance")
            if any(license_id in {"UNKNOWN", "NOASSERTION"} for license_id in required):
                problems.append("unknown_underlying_license")
            outside_policy = sorted(set(required) - STANDARD_LICENSES)
            if outside_policy and record["approval"]["status"] != "approved":
                problems.append("unapproved_license")
            elif record["approval"]["status"] == "approved":
                approval = record["approval"]
                if (
                    approval.get("reported_license") != entry["license"]
                    or approval.get("required_licenses") != sorted(required)
                    or approval.get("license_text_sha256") != entry.get("notice", {}).get("license_text_sha256")
                ):
                    problems.append("stale_approval_scope_or_license_text")
                else:
                    approval_basis = "recorded_decision"
            elif not outside_policy:
                approval_basis = "nvidia_own_contribution" if not required else "standard_license_policy"
        # Unknown provenance cannot be approved away, including for NVIDIA files.
        if entry["origin"] == "legacy_unresolved" and "unknown_license_or_provenance" not in problems:
            problems.append("unknown_license_or_provenance")
        results.append(
            {
                "path": name,
                "sha256": entry["sha256"],
                "reported_license": entry["license"],
                "ownership": record["ownership"] if record else "unreviewed",
                "required_licenses": required,
                "passed": not problems,
                "problems": problems,
                "approval_basis": approval_basis if not problems else "none",
            }
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=provenance.ROOT, help="Exact source being qualified")
    parser.add_argument("--review", type=Path, help="Trusted per-file review; defaults to the source's policy")
    parser.add_argument("--wheel", type=Path, help="Also check the exact repaired wheel")
    parser.add_argument("--source-sha", default="", help="Workflow-verified source identity for the report")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = {"schema_version": 1, "passed": False, "legal_approval_asserted": False, "files": []}
    try:
        if args.source_sha and not re.fullmatch(r"[a-f0-9]{40}", args.source_sha):
            raise ValueError("source identity must be a full commit SHA")
        report["source_sha"] = args.source_sha
        root = args.root.resolve()
        entries, wheel_sha = provenance.validate(root, root / provenance.INVENTORY, wheel_path=args.wheel)
        records = load_review(args.review or root / REVIEW)
        # Extra records are not silently reused for another branch/source.
        if set(records) != {entry["path"] for entry in entries}:
            raise ValueError("review paths differ from the exact source inventory")
        results = assess(entries, records)
        report.update(passed=all(r["passed"] for r in results), files=results, wheel_sha256=wheel_sha)
        report["problem_counts"] = dict(Counter(problem for r in results for problem in r["problems"]))
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["error"] = str(error)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config-license-approval.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        detail = report.get("error", report.get("problem_counts"))
        print(f"::error::Config license compliance failed; see config-license-approval.json: {detail}")
        return 1
    print(f"Configuration license policy passed for {len(report['files'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
