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
import zipfile
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


def license_terms(expression: str) -> set[str]:
    """Split plain conjunctions only; other SPDX expressions remain obligations."""
    if re.fullmatch(r"[A-Za-z0-9.+:-]+(?: AND [A-Za-z0-9.+:-]+)*", expression):
        return set(expression.split(" AND "))
    return {expression}


def https_reference(value: object) -> bool:
    if not isinstance(value, str):
        return False
    url = urlsplit(value)
    return bool(url.scheme == "https" and url.hostname and not url.username and not url.password)


def pinned_nvidia_card(evidence: dict) -> bool:
    source = evidence.get("url", "")
    revision = evidence.get("revision", "")
    if not https_reference(source) or not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        return False
    url = urlsplit(source)
    return bool(
        url.hostname == "huggingface.co"
        and not url.query
        and not url.fragment
        and re.fullmatch(r"/nvidia/[^/]+/blob/" + revision + r"/README\.md", url.path)
    )


def ownership_matches(entry: dict, record: dict) -> bool:
    evidence = record["ownership_evidence"]
    if evidence.get("status") != "recorded_source" or not pinned_nvidia_card(evidence):
        return False
    source = entry.get("source", {})
    if not isinstance(source, dict):
        return False
    card = evidence["url"].removesuffix("README.md")
    if source.get("revision") == evidence["revision"] and source.get("url") == card + source.get("path", ""):
        return True
    # Cross-source ownership needs a deliberate file/bytes/source-bound mapping.
    mapping = evidence.get("config_mapping", {})
    return bool(
        isinstance(mapping, dict)
        and mapping.get("path") == entry["path"]
        and mapping.get("sha256") == entry["sha256"]
        and mapping.get("source") == source
        and https_reference(mapping.get("reference"))
        and isinstance(mapping.get("explanation"), str)
        and mapping["explanation"].strip()
    )


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
        contribution = record.get("nvidia_contribution_licenses", [])
        if not isinstance(contribution, list) or any(not isinstance(x, str) or not x.strip() for x in contribution):
            raise ValueError(f"invalid NVIDIA contribution license list: {name}")
        if record["ownership"] == "third_party" and contribution:
            raise ValueError(f"third-party record cannot exempt NVIDIA contribution licenses: {name}")
        notices = record.get("license_notices", {})
        if not isinstance(notices, dict) or any(
            not isinstance(k, str) or not k.strip() or not isinstance(v, dict) for k, v in notices.items()
        ):
            raise ValueError(f"invalid per-license notice references: {name}")
        if record["ownership"] != "third_party":
            evidence = record.get("ownership_evidence", {})
            if not isinstance(evidence, dict):
                raise ValueError(f"invalid ownership evidence: {name}")
            if evidence.get("status") not in {"recorded_source", "source_unverified"}:
                raise ValueError(f"invalid ownership evidence status: {name}")
            # A declaration of an unavailable source cannot pass the gate.
            if not pinned_nvidia_card(evidence) and evidence.get("status") != "source_unverified":
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
                or approval.get("path") != name
                or not re.fullmatch(r"[a-f0-9]{64}", str(approval.get("sha256")))
                or not isinstance(approval.get("license_text_sha256_by_license"), dict)
                or any(
                    not isinstance(k, str) or not k or not re.fullmatch(r"[a-f0-9]{64}", str(v))
                    for k, v in approval.get("license_text_sha256_by_license", {}).items()
                )
            ):
                raise ValueError(
                    f"approval must bind path, hash, reported/required licenses and each text hash: {name}"
                )
        files[name] = record
    return files


def required_text_hashes(entry: dict, record: dict, licenses: list[str], root: Path | None, registry: dict) -> dict:
    hashes = {}
    for license_id in licenses:
        if license_id == entry["license"]:
            # The provenance check already validated this identifier and text.
            notice = entry["notice"]
        else:
            notice = record.get("license_notices", {}).get(license_id)
            if not isinstance(notice, dict) or root is None:
                raise ValueError(f"missing legal text for {license_id}")
            if not registry:
                registry.update(provenance.load_json((root / provenance.INVENTORY).read_bytes())["licenses"])
            registered_license = registry.get(license_id)
            if not isinstance(registered_license, dict):
                raise ValueError(f"unregistered legal text for {license_id}")
            registered = registered_license.get("license_text_sha256", [])
            if not isinstance(registered, list):
                raise ValueError(f"invalid legal text registry for {license_id}")
            if notice.get("license_text_sha256") not in registered:
                raise ValueError(f"unregistered legal text for {license_id}")
            provenance.check_notice(root, entry | {"notice": notice}, None)
        hashes[license_id] = notice["license_text_sha256"]
    return hashes


def assess(entries: list[dict], records: dict, *, root: Path | None = None) -> list[dict]:
    """Evaluate declarations after the source/wheel provenance check succeeds."""
    results = []
    registry = {}
    for entry in entries:
        name = entry["path"]
        record = records.get(name)
        problems = []
        reported = license_terms(entry["license"])
        required = [entry["license"]]
        approval_basis = "none"
        if not record:
            problems.append("missing_review")
        elif record["sha256"] != entry["sha256"]:
            problems.append("stale_review_hash")
        else:
            ownership = record["ownership"]
            underlying = set(record["underlying_licenses"])
            # Preserve the original exact-identifier standard-license policy.
            # In particular, third-party SPDX expressions still need a decision.
            required = sorted({entry["license"]} | underlying)
            if ownership != "third_party":
                if record["ownership_evidence"].get("status") == "source_unverified":
                    problems.append("ownership_source_unverified")
                if not ownership_matches(entry, record):
                    problems.append("ownership_source_mismatch")
                else:
                    contribution = set(record.get("nvidia_contribution_licenses", []))
                    if (
                        not contribution <= reported
                        or contribution & underlying
                        or not (reported - contribution) <= underlying
                    ):
                        problems.append("inconsistent_license_obligations")
                    required = sorted((reported - contribution) | underlying)
            elif underlying and underlying != reported:
                problems.append("inconsistent_license_obligations")
            if record["approval"]["status"] == "rejected":
                problems.append("rejected")
            if entry["origin"] == "legacy_unresolved" or entry["license"] in {"UNKNOWN", "NOASSERTION", ""}:
                problems.append("unknown_license_or_provenance")
            if any(license_id in {"UNKNOWN", "NOASSERTION"} for license_id in required):
                problems.append("unknown_underlying_license")
            outside_policy = sorted(set(required) - STANDARD_LICENSES)
            try:
                text_hashes = required_text_hashes(entry, record, outside_policy, root, registry)
            except (OSError, ValueError, KeyError, TypeError) as error:
                problems.append("missing_or_invalid_required_license_text")
                text_hashes = {}
                text_error = str(error)
            if outside_policy and record["approval"]["status"] != "approved":
                problems.append("unapproved_license")
            elif record["approval"]["status"] == "approved":
                approval = record["approval"]
                if (
                    approval.get("path") != name
                    or approval.get("sha256") != entry["sha256"]
                    or approval.get("reported_license") != entry["license"]
                    or approval.get("required_licenses") != sorted(required)
                    or approval.get("license_text_sha256") != entry.get("notice", {}).get("license_text_sha256")
                    or approval.get("license_text_sha256_by_license") != text_hashes
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
        if "missing_or_invalid_required_license_text" in problems:
            results[-1]["license_text_error"] = text_error
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
        report["wheel_sha256"] = wheel_sha
        records = load_review(args.review or root / REVIEW)
        # Extra records are not silently reused for another branch/source.
        paths = {entry["path"] for entry in entries}
        if set(records) != paths:
            report["missing_review_paths"] = sorted(paths - records.keys())
            report["extra_review_paths"] = sorted(records.keys() - paths)
            raise ValueError(
                "review paths differ from the exact source inventory: "
                f"missing={report['missing_review_paths']}, extra={report['extra_review_paths']}"
            )
        results = assess(entries, records, root=root)
        report.update(passed=all(r["passed"] for r in results), files=results, wheel_sha256=wheel_sha)
        report["problem_counts"] = dict(Counter(problem for r in results for problem in r["problems"]))
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
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
