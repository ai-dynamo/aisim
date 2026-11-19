# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure candidate-hygiene rules, tested with in-memory fixtures."""

import dataclasses

import pytest

from aisim_core.data.hygiene import (
    CandidateFile,
    CandidatePolicy,
    check_candidate,
    check_content,
    check_creator,
    check_listing,
)

pytestmark = pytest.mark.unit

POLICY = CandidatePolicy(allowed_creators=frozenset({"Alice"}), max_files=3, max_total_bytes=1000, max_file_bytes=600)
GOOD = CandidateFile("data/eval/x/record.json", 10)
PQ = CandidateFile("data/eval/x/truth.parquet", 100)


def codes(findings):
    return {f.code for f in findings}


def test_creator_allowlist_fails_closed():
    assert check_creator(POLICY, "alice") == []  # case-insensitive
    assert codes(check_creator(POLICY, "mallory")) == {"creator-not-allowed"}
    assert codes(check_creator(POLICY, None)) == {"creator-unknown"}
    assert codes(check_creator(CandidatePolicy(), "alice")) == {"creator-not-allowed"}


def test_listing_accepts_valid():
    assert check_listing(POLICY, [GOOD, PQ]) == []


@pytest.mark.parametrize(
    ("file", "code"),
    [
        (CandidateFile("data/../etc/x.json", 1), "path-traversal"),
        (CandidateFile("/data/x.json", 1), "path-unsafe"),
        (CandidateFile("data\\x.json", 1), "path-unsafe"),
        (CandidateFile("data//x.json", 1), "path-traversal"),
        (CandidateFile("other/x.json", 1), "path-root"),
        (CandidateFile("README.md", 1), "path-root"),
        (CandidateFile(".gitattributes", 1), "path-reserved"),
        (CandidateFile(".github/workflows/validate.yml", 1), "path-root"),
        (CandidateFile("data/x.pkl", 1), "suffix"),
        (CandidateFile("data/x.py", 1), "suffix"),
        (CandidateFile("data/x.txt", 1), "suffix"),
        (CandidateFile("data/.gitattributes", 1), "path-reserved"),
        (CandidateFile("data/big.parquet", 601), "file-too-large"),
        (CandidateFile("data/l.json", 1, is_symlink=True), "symlink"),
        (CandidateFile("data/l.parquet", 1, is_lfs=True), "lfs-pointer"),
    ],
)
def test_listing_rejects(file, code):
    assert code in codes(check_listing(POLICY, [file]))


def test_gz_suffix_follows_the_policy_allow_list():
    gz = CandidateFile("data/trace.jsonl.gz", 1)
    assert "suffix" in codes(check_listing(POLICY, [gz]))  # default candidates stay parquet and json
    allowed = dataclasses.replace(POLICY, allowed_suffixes=POLICY.allowed_suffixes | {".gz"})
    assert check_listing(allowed, [gz]) == []
    pkl = dataclasses.replace(POLICY, allowed_suffixes=POLICY.allowed_suffixes | {".pkl"})
    assert "suffix" in codes(check_listing(pkl, [CandidateFile("data/x.pkl", 1)]))  # code stays forbidden


def test_root_matching_uses_directory_boundaries():
    policy = dataclasses.replace(POLICY, allowed_roots=("data",))
    assert check_listing(policy, [CandidateFile("data/x.json", 2)]) == []
    assert "path-root" in codes(check_listing(policy, [CandidateFile("database/x.json", 2)]))


def test_deep_json_is_reported_as_invalid():
    import json

    depth = 10000  # also exceeds CPython 3.12's separate C JSON decoder recursion limit
    doc = b"[" * depth + b"0" + b"]" * depth
    policy = CandidatePolicy()
    file = CandidateFile("data/deep.json", len(doc))
    assert check_listing(policy, [file]) == []
    with pytest.raises(RecursionError):
        json.loads(doc)
    assert "json-invalid" in codes(check_content(policy, file, doc))
    assert check_content(POLICY, CandidateFile("data/shallow.json", 3), b"[0]") == []


def test_listing_limits_and_duplicates():
    many = [CandidateFile(f"data/{i}.json", 1) for i in range(4)]
    assert "too-many-files" in codes(check_listing(POLICY, many))
    assert "too-large" in codes(
        check_listing(POLICY, [CandidateFile("data/a.json", 500), CandidateFile("data/b.json", 600)])
    )
    assert "duplicate-path" in codes(
        check_listing(POLICY, [CandidateFile("data/A.json", 1), CandidateFile("data/a.json", 1)])
    )
    assert codes(check_listing(POLICY, [])) == {"empty"}


def test_content_parquet_and_json():
    assert check_content(POLICY, PQ, b"PAR1" + b"x" * 8, b"PAR1") == []
    assert "parquet-magic" in codes(check_content(POLICY, PQ, b"notparquet", b"PAR1"))
    assert "parquet-magic" in codes(check_content(POLICY, PQ, b"PAR1", b"xxxx"))
    ok = CandidateFile("data/a.json", 2)
    assert check_content(POLICY, ok, b"{}") == []
    for doc in (b'{"a":NaN}', b'{"a":Infinity}', b"[-Infinity]"):  # complete documents, invalid only for the constant
        found = check_content(POLICY, CandidateFile("data/a.json", len(doc)), doc)
        assert codes(found) == {"json-invalid"} and "non-finite" in found[0].message
    assert "json-invalid" in codes(check_content(POLICY, CandidateFile("data/a.json", 4), b"{NaN"))
    assert "json-incomplete" in codes(check_content(POLICY, CandidateFile("data/a.json", 50), b"{}"))
    assert "json-too-large" in codes(check_content(CandidatePolicy(max_json_bytes=1), ok, b"{}"))


@pytest.mark.parametrize(
    "doc,valid",
    [
        (b'{"setting":1,"setting":2}', False),
        (b'{"nested":{"setting":1,"setting":2}}', False),
        (b'{"nested":[{"setting":1,"setting":2}]}', False),
        (b'{"setting":1}', True),
        (b'{"first":{"setting":1},"second":{"setting":2}}', True),
    ],
)
def test_json_keys_are_unique_within_each_object(doc, valid):
    findings = check_content(POLICY, CandidateFile("data/settings.json", len(doc)), doc)
    assert ("json-invalid" not in codes(findings)) is valid
    if not valid:
        assert "duplicate JSON key" in findings[0].message


@pytest.mark.parametrize("head", [b"\x7fELF....", b"#!/bin/sh", b"PK\x03\x04zz", b"MZ\x90\x00"])
def test_content_rejects_executables_even_when_renamed(head):
    assert "forbidden-content" in codes(check_content(POLICY, CandidateFile("data/a.json", len(head)), head))


def test_lfs_pointer_content():
    head = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\n"
    assert "lfs-pointer" in codes(check_content(POLICY, CandidateFile("data/a.json", len(head)), head))


def test_candidate_aggregate_and_missing_sample():
    samples = {GOOD.path: (b"{" + b" " * 8 + b"}", b""), PQ.path: (b"PAR1" + b"x" * 8, b"PAR1")}
    assert check_candidate(POLICY, "alice", [GOOD, PQ], samples).ok
    report = check_candidate(POLICY, "alice", [GOOD, PQ], {GOOD.path: samples[GOOD.path]})
    assert not report.ok and codes(report.errors) == {"no-sample"}
    assert not check_candidate(POLICY, "mallory", [GOOD], {GOOD.path: samples[GOOD.path]}).ok


def test_allowed_gz_suffix_is_exempt_from_the_gzip_magic_rule_only():
    gz = b"\x1f\x8b\x08\x00" + b"x" * 12
    allowed = dataclasses.replace(POLICY, allowed_suffixes=POLICY.allowed_suffixes | {".gz"})
    f = CandidateFile("data/eval/x/log.gz", len(gz))
    assert check_candidate(allowed, "alice", [f], {f.path: (gz, b"")}).ok
    assert "forbidden-content" in codes(check_content(POLICY, f, gz))  # not allowed by default
    renamed = CandidateFile("data/a.json", len(gz))
    assert "forbidden-content" in codes(check_content(allowed, renamed, gz))
    assert "forbidden-content" in codes(check_content(allowed, f, b"\x7fELF" + b"x" * 12))


@pytest.mark.parametrize("number,ok", [("1e400", False), ("-1e400", False), ("1e30", True)])
def test_json_overflow_is_not_finite(number, ok):
    data = ('{"value":' + number + "}").encode()
    assert (not check_content(CandidatePolicy(), CandidateFile("data/test.json", len(data)), data)) is ok


@pytest.mark.parametrize(
    "text,ok",
    [
        ("setting: 1\nsetting: 2\n", False),
        ("nested:\n  setting: 1\n  setting: 2\n", False),
        ("defaults: &d {setting: 1}\nconfig: {<<: *d, setting: 2}\nalias: *d\n", True),
    ],
)
def test_yaml_explicit_duplicates_and_merge_overrides(text, ok):
    from aisim_core.data.hygiene import yaml_sidecar_value

    if ok:
        value = yaml_sidecar_value(text.encode())
        assert value["config"]["setting"] == 2
        assert value["defaults"] is value["alias"]
    else:
        with pytest.raises(ValueError, match="duplicate YAML key"):
            yaml_sidecar_value(text.encode())
