# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate hygiene for public dataset candidates (DEP #435, "CI and public evaluation").

A contributor publishes a candidate as a Hugging Face PR revision of the public
dataset repository. Before a maintainer starts the trusted validation workflow,
these pure checks decide whether the candidate may be fetched at all:

* who may create candidates (:func:`check_creator`),
* file-count and size limits plus allowed paths and suffixes
  (:func:`check_listing`),
* content checks on the leading and trailing bytes of each file
  (:func:`check_content`).

The candidate checks touch neither network nor filesystem. Callers pass the listing and
byte samples they obtained; ``checked_path`` additionally preflights local document references. A
finding is final: a candidate with any error finding must not reach a workflow.
"""

from __future__ import annotations

import json
import math
import posixpath
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import yaml

PARQUET_MAGIC = b"PAR1"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"
# Executable and archive magics that never belong in a data candidate.
_FORBIDDEN_MAGIC: dict[str, bytes] = {
    "ELF executable": b"\x7fELF",
    "script": b"#!",
    "zip archive": b"PK\x03\x04",
    "gzip archive": b"\x1f\x8b",
    "Windows executable": b"MZ",
}
_FORBIDDEN_SUFFIXES = frozenset({".pkl", ".pickle", ".pt", ".pth", ".bin", ".so", ".py", ".sh", ".exe", ".zip"})
# Never loadable code or archives; ``.gz`` is data and is allowed only when a policy lists it
# (the mirror policy does, for declared provenance artifacts).
_RESERVED_NAMES = frozenset({".gitattributes", ".gitignore", ".gitmodules"})


@dataclass(frozen=True)
class CandidatePolicy:
    """Limits and allow-lists. Defaults are conservative; the repository owner tunes them."""

    # HF user names allowed to open candidate PRs. Empty means nobody (fail closed).
    allowed_creators: frozenset[str] = frozenset()
    max_files: int = 200
    max_total_bytes: int = 512 * 1024 * 1024
    max_file_bytes: int = 256 * 1024 * 1024
    max_json_bytes: int = 4 * 1024 * 1024
    allowed_roots: tuple[str, ...] = ("data/", "catalog/")
    allowed_suffixes: frozenset[str] = frozenset({".parquet", ".json"})
    max_path_length: int = 240


@dataclass(frozen=True)
class CandidateFile:
    path: str
    size: int
    is_symlink: bool = False
    is_lfs: bool = False


@dataclass(frozen=True)
class Finding:
    code: str
    message: str
    path: str | None = None
    severity: Literal["error", "warning"] = "error"


def checked_path(root: Path, relative: str) -> Path:
    """Preflight a document reference before any loader may read it."""
    parts = relative.split("/")
    if not relative or relative.startswith("/") or "\\" in relative or any(p in {"", ".", ".."} for p in parts):
        raise ValueError(f"unsafe repository-relative path: {relative!r}")
    current = root
    for part in ("", *parts):
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink in document path: {relative}")
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"document path escapes root: {relative}")
    return current


@dataclass
class HygieneReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, items: Iterable[Finding]) -> HygieneReport:
        self.findings.extend(items)
        return self


def check_creator(policy: CandidatePolicy, creator: str | None) -> list[Finding]:
    """Only listed HF users may create candidates; an unknown author fails closed."""
    if not creator:
        return [Finding("creator-unknown", "candidate creator could not be determined")]
    if creator.casefold() not in {c.casefold() for c in policy.allowed_creators}:
        return [Finding("creator-not-allowed", f"{creator!r} may not create candidates")]
    return []


def _path_findings(policy: CandidatePolicy, path: str) -> list[Finding]:
    bad: list[Finding] = []

    def err(code: str, msg: str) -> None:
        bad.append(Finding(code, msg, path))

    if not path or len(path) > policy.max_path_length:
        err("path-length", "path is empty or too long")
        return bad
    if path.startswith("/") or "\\" in path or "\x00" in path or any(ord(c) < 32 for c in path):
        err("path-unsafe", "absolute path, backslash or control character")
    parts = path.split("/")
    if ".." in parts or "." in parts or "" in parts or posixpath.normpath(path) != path:
        err("path-traversal", "path is not normalized or escapes the repository")
    if any(p in _RESERVED_NAMES for p in parts):
        err("path-reserved", "repository control files may not be changed by a candidate")
    if not any(path.startswith(root.rstrip("/") + "/") for root in policy.allowed_roots):
        err("path-root", f"path is outside {list(policy.allowed_roots)}")
    suffix = posixpath.splitext(parts[-1])[1].lower()
    if suffix in _FORBIDDEN_SUFFIXES or suffix not in policy.allowed_suffixes:
        err("suffix", f"suffix {suffix or '(none)'!r} is not allowed")
    return bad


def check_listing(policy: CandidatePolicy, files: Iterable[CandidateFile]) -> list[Finding]:
    """Count, size, path and symlink rules over the whole candidate file list."""
    listing = list(files)
    out: list[Finding] = []
    if not listing:
        out.append(Finding("empty", "candidate has no files"))
    if len(listing) > policy.max_files:
        out.append(Finding("too-many-files", f"{len(listing)} files exceed the limit of {policy.max_files}"))
    total = sum(f.size for f in listing)
    if total > policy.max_total_bytes:
        out.append(Finding("too-large", f"{total} bytes exceed the limit of {policy.max_total_bytes}"))
    seen: dict[str, str] = {}
    for f in listing:
        out.extend(_path_findings(policy, f.path))
        folded = f.path.casefold()
        if folded in seen:
            out.append(Finding("duplicate-path", f"collides with {seen[folded]!r} ignoring case", f.path))
        seen[folded] = f.path
        if f.size < 0 or f.size > policy.max_file_bytes:
            out.append(Finding("file-too-large", f"{f.size} bytes exceed {policy.max_file_bytes}", f.path))
        if f.is_symlink:
            out.append(Finding("symlink", "symbolic links are not allowed", f.path))
        if f.is_lfs:
            out.append(Finding("lfs-pointer", "LFS pointers are not allowed; upload the content", f.path))
    return out


def check_content(policy: CandidatePolicy, file: CandidateFile, head: bytes, tail: bytes = b"") -> list[Finding]:
    """Check magic bytes and structured text from the first and last bytes of a file.

    ``head`` is the first bytes (at least 16 when available), ``tail`` the last
    four bytes. JSON/YAML must fit ``max_json_bytes`` and be complete in ``head``.
    """
    path = file.path
    suffix = posixpath.splitext(path)[1].lower()
    out: list[Finding] = []
    if head.startswith(LFS_POINTER_PREFIX):
        out.append(Finding("lfs-pointer", "file is a Git LFS pointer", path))
    for label, magic in _FORBIDDEN_MAGIC.items():
        if label == "gzip archive" and suffix == ".gz" and suffix in policy.allowed_suffixes:
            continue  # a gzip file under an allowed .gz suffix is the expected content
        if head.startswith(magic):
            out.append(Finding("forbidden-content", f"content looks like a {label}", path))
    if suffix == ".parquet":
        if head[:4] != PARQUET_MAGIC or tail[-4:] != PARQUET_MAGIC:
            out.append(Finding("parquet-magic", "not a Parquet file (missing PAR1 magic)", path))
    elif suffix in {".json", ".yaml"}:
        kind = suffix[1:]
        if file.size > policy.max_json_bytes:
            out.append(Finding(f"{kind}-too-large", f"{kind.upper()} over {policy.max_json_bytes} bytes", path))
        elif len(head) < file.size:
            out.append(Finding(f"{kind}-incomplete", f"full {kind.upper()} content is needed for the check", path))
        else:
            try:
                if suffix == ".yaml":
                    yaml_sidecar_value(head)
                else:
                    json_sidecar_value(head)
            except (UnicodeDecodeError, ValueError, RecursionError) as exc:
                out.append(Finding(f"{kind}-invalid", f"not strict UTF-8 {kind.upper()}: {exc}", path))
    return out


def bounded_json_bytes(path: Path) -> bytes:
    """Read structured JSON within the shared limit, including files exempt from blob limits."""
    limit = CandidatePolicy().max_json_bytes
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"JSON exceeds the {limit}-byte limit")
    return data


class DuplicateJsonKeyError(ValueError):
    """JSON is ambiguous because an object repeats a key."""


def _unique_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def json_sidecar_value(data: bytes):
    """Parse strict UTF-8 JSON, rejecting ambiguous keys and non-finite numbers."""
    return json.loads(
        data.decode("utf-8"),
        parse_constant=_reject_constant,
        parse_float=_finite_float,
        object_pairs_hook=_unique_json_keys,
    )


def _check_yaml_merge_budget(root, limit=100_000):
    """Bound the composed graph before SafeLoader materializes flattened merge pairs."""
    sizes = {}
    active = set()
    total = 0

    def visit(node):
        nonlocal total
        marker = id(node)
        if marker in active:
            raise ValueError("cyclic YAML aliases are not supported")
        if marker in sizes:
            return sizes[marker]
        active.add(marker)
        size = 0
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                visit(key)
                visit(value)
                if key.tag == "tag:yaml.org,2002:merge":
                    sources = value.value if isinstance(value, yaml.SequenceNode) else [value]
                    size += sum(visit(source) for source in sources)
                else:
                    size += 1
                if size > limit:
                    raise ValueError("YAML merge expansion exceeds the pair budget")
            total += size
            if total > limit:
                raise ValueError("YAML merge expansion exceeds the pair budget")
        elif isinstance(node, yaml.SequenceNode):
            for child in node.value:
                visit(child)
        active.remove(marker)
        sizes[marker] = size
        return size

    visit(root)


class _SidecarLoader(yaml.SafeLoader):
    preserve_merges = False

    def get_single_data(self):
        node = self.get_single_node()
        if node is None:
            return None
        _check_yaml_merge_budget(node)
        self._checked_mapping_nodes = set()
        return self.construct_document(node)

    def flatten_mapping(self, node):
        # SafeLoader calls this recursively for merge sources, including inline
        # mappings that never pass through construct_mapping. Aliases may revisit
        # an already flattened node, whose inherited keys are no longer explicit.
        marker = id(node)
        if marker in self._checked_mapping_nodes:
            return
        self._checked_mapping_nodes.add(marker)
        # Check explicit keys before flattening merges: an explicit override of a
        # merged default is valid YAML, while repeated explicit settings are ambiguous.
        seen = set()
        seen_merge = False
        for key, _ in node.value:
            if key.tag == "tag:yaml.org,2002:merge":
                if seen_merge:
                    raise ValueError("duplicate YAML key: '<<'")
                seen_merge = True
                continue
            if not isinstance(key, yaml.ScalarNode):
                raise ValueError("YAML mapping keys must be scalar")
            value = self.construct_object(key)
            if value is not None and not isinstance(value, (str, bool, int, float, date, datetime)):
                raise ValueError("unsupported YAML mapping key")
            try:
                if value in seen:
                    raise ValueError(f"duplicate YAML key: {value!r}")
                seen.add(value)
            except TypeError as exc:
                raise ValueError("YAML mapping keys must be scalar") from exc
        if self.preserve_merges:
            # Keep every source value for disclosure scanning, including defaults
            # that an explicit override would otherwise erase.
            for key, _ in node.value:
                if key.tag == "tag:yaml.org,2002:merge":
                    # A literal string key "<<" is legal alongside a merge key.
                    # Keep the source map under a collision-free scan-only name.
                    name = "<<"
                    while name in seen:
                        name += "<"
                    key.value = name
                    key.tag = "tag:yaml.org,2002:str"
        else:
            super().flatten_mapping(node)


class _SourceSidecarLoader(_SidecarLoader):
    preserve_merges = True


def yaml_sidecar_value(data: bytes, *, preserve_merges: bool = False):
    """Safely read operation metadata; retain shared YAML aliases without expanding them."""
    try:
        value = yaml.load(data.decode("utf-8"), Loader=_SourceSidecarLoader if preserve_merges else _SidecarLoader)
    except (yaml.YAMLError, RecursionError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid YAML sidecar: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("YAML sidecar must contain a mapping")

    active: set[int] = set()
    checked: set[int] = set()

    def check(item):
        if isinstance(item, (dict, list)):
            marker = id(item)
            if marker in active:
                raise ValueError("cyclic YAML aliases are not supported")
            if marker in checked:
                return
            active.add(marker)
            for child in item:
                check(child)
                if isinstance(item, dict):
                    check(item[child])
            active.remove(marker)
            checked.add(marker)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("non-finite YAML scalar")
        elif item is not None and not isinstance(item, (str, bool, int, float, date, datetime)):
            raise ValueError(f"unsupported YAML scalar type: {type(item).__name__}")

    # Binary, sets and cycles must fail rather than escape the text scanner.
    try:
        check(value)
    except RecursionError as exc:
        raise ValueError("YAML sidecar is too deeply nested") from exc
    return value


def _reject_constant(name: str):
    raise ValueError(f"non-finite constant {name}")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite number {text}")
    return value


def check_candidate(
    policy: CandidatePolicy,
    creator: str | None,
    files: Iterable[CandidateFile],
    samples: Mapping[str, tuple[bytes, bytes]],
) -> HygieneReport:
    """Run every check. ``samples`` maps path -> (head, tail); missing samples are errors."""
    listing = list(files)
    report = HygieneReport()
    report.extend(check_creator(policy, creator))
    report.extend(check_listing(policy, listing))
    for f in listing:
        sample = samples.get(f.path)
        if sample is None:
            report.findings.append(Finding("no-sample", "content was not inspected", f.path))
            continue
        if not sample[0] and f.size > 0:
            report.findings.append(Finding("no-sample", "head sample is empty; content was not inspected", f.path))
            continue
        report.extend(check_content(policy, f, *sample))
    return report
