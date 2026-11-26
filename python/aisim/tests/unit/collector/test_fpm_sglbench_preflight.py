# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of ``sglbench_preflight.py``, the preflight of the SGLang self-benchmark. No SGLang, no GPU.

Everything that stands in for SGLang is FAKE and is named so: ``fake_tree`` puts the FAKE ``sglang`` package
tree of test_fpm_sglbench_compat.py into ``sys.modules`` and fills in whatever the seam rows name and that tree
lacks, from the rows themselves. Like the module under test, this file types no module, class or field of
SGLang: every test takes the rows from ``sglbench_compat.SGLANG_SEAM_NAMES`` and
``sglbench_recorder.SGLANG_SEAM_NAMES``. The start of the shipped file by path, in an interpreter without
SGLang, is in test_fpm_sglbench_gate.py.

This file is also the home of three helpers about seam rows that the suites of sglbench_compat.py and
sglbench_recorder.py use: ``sglang_names_reached_by_import``, ``seam_rows_missing_for`` and
``seam_rows_not_in_source``, with ``sglang_source_roots``.

What the tests cannot show is that the names are there in a real image. ``seam_rows_not_in_source`` reads
the SGLang source of both tested releases when a developer has it; nothing here imports SGLang.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[3] / "collector/fpm_forward/runtime"
# The pod holds the runtime files flat in one directory and imports them by their plain names. The directory
# goes to the END of the path so that it shadows nothing.
if str(RUNTIME) not in sys.path:
    sys.path.append(str(RUNTIME))

import sglbench_compat as compat
import sglbench_preflight as preflight
import sglbench_recorder as recorder

pytestmark = pytest.mark.unit

MODULE_PATH = RUNTIME / "sglbench_preflight.py"

# A folder with one checkout of sgl-project/sglang per tested release, each in a folder named after its tag
# (``v`` and the version). CI has none, and a test that asks for it is skipped there.
SOURCES_ENV = "AISIM_SGLBENCH_SGLANG_SOURCES"

SOURCES = {"sglbench_compat": compat, "sglbench_recorder": recorder}
ALL_ROWS = [
    (source, *row) for source in preflight.SEAM_MODULES for row in getattr(SOURCES[source], preflight.SEAM_CONSTANT)
]
ROWS_OF_KIND = {kind: [row for row in ALL_ROWS if row[3] == kind] for kind in sorted({row[3] for row in ALL_ROWS})}
SEAM_MODULE_NAMES = sorted({row[1] for row in ALL_ROWS})
# The root package is left out of the tests that take one module away: the version is read from it.
ROOT_PACKAGE = min(SEAM_MODULE_NAMES, key=len)

FAKE_VALUE = "FAKE placeholder"


def _row_id(row) -> str:
    return f"{row[1]}:{row[2]}"


# ---------------------------------------------------------------------------------------------------------------
# Helpers about seam rows, shared with the suites of sglbench_compat.py and sglbench_recorder.py
# ---------------------------------------------------------------------------------------------------------------


def sglang_names_reached_by_import(tree: ast.Module) -> set:
    """``(module, name)`` for every SGLang import of a module and every attribute it reads off what it bound.

    ``import M`` gives ``(M, None)`` and, for a read of ``M.x``, ``(M, "x")``. ``from M import N`` gives
    ``(M, "N")`` and, for a read of ``N.x``, ``(M, "N.x")``. The launcher's modules import SGLang inside
    functions only, so each function is looked at on its own: its imports, then its attribute reads.
    """
    reached = set()
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        bound = {}
        for node in ast.walk(function):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] != ROOT_PACKAGE:
                        continue
                    reached.add((alias.name, None))
                    # ``import a.b`` without ``as`` binds ``a`` alone, and a read through it would be missed.
                    assert alias.asname or "." not in alias.name, f"import {alias.name} needs an alias"
                    bound[alias.asname or alias.name] = (alias.name, "")
            elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == ROOT_PACKAGE:
                for alias in node.names:
                    reached.add((node.module, alias.name))
                    bound[alias.asname or alias.name] = (node.module, alias.name + ".")
        for node in ast.walk(function):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in bound:
                module, prefix = bound[node.value.id]
                reached.add((module, prefix + node.attr))
    return reached


def seam_rows_missing_for(reached: set, rows) -> list:
    """The entries of ``reached`` that no row of ``rows`` names."""
    modules = {module for module, _name, _kind in rows}
    names = {(module, name) for module, name, _kind in rows}
    missing = []
    for module, name in sorted(reached, key=lambda pair: (pair[0], pair[1] or "")):
        if name is None:
            listed = module in modules
        else:
            listed = any(m == module and (n == name or n.startswith(name + ".")) for m, n in names)
        if not listed:
            missing.append((module, name))
    return missing


def sglang_source_roots(versions) -> dict:
    """The ``python`` folder of an SGLang checkout for each version; skips the calling test without them."""
    base = os.environ.get(SOURCES_ENV)
    if not base:
        pytest.skip(f"{SOURCES_ENV} is not set; it names a folder with one SGLang checkout per tested release")
    roots = {version: Path(base) / f"v{version}" / "python" for version in versions}
    absent = [version for version, root in roots.items() if not (root / ROOT_PACKAGE / "__init__.py").is_file()]
    # A variable that is set and wrong is a mistake to show, not a reason to skip.
    assert not absent, f"{SOURCES_ENV} holds no checkout for {absent}"
    return roots


def _bindings(statements) -> dict:
    """Every statement that binds a name, by name, for a module body or a class body.

    ``if``, ``try`` and ``with`` blocks are looked into; functions and classes are not.
    """
    found: dict = {}
    for node in statements:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            found.setdefault(node.name, []).append(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for leaf in ast.walk(target):
                    if isinstance(leaf, ast.Name):
                        found.setdefault(leaf.id, []).append(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.setdefault(node.target.id, []).append(node)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                found.setdefault((alias.asname or alias.name).split(".")[0], []).append(node)
        elif isinstance(node, ast.If | ast.Try | ast.With):
            blocks = [getattr(node, part, []) for part in ("body", "orelse", "finalbody")]
            blocks += [handler.body for handler in getattr(node, "handlers", [])]
            for block in blocks:
                for name, nodes in _bindings(block).items():
                    found.setdefault(name, []).extend(nodes)
    return found


def _is_dataclass_statement(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if ast.unparse(target) in ("dataclass", "dataclasses.dataclass"):
            return True
    return False


def _source_problem(python_root: Path, module: str, name: str, kind: str) -> str | None:
    base = python_root.joinpath(*module.split("."))
    path = next((path for path in (base.with_suffix(".py"), base / "__init__.py") if path.is_file()), None)
    if path is None:
        return "the module has no file"
    owner_name, _, member = name.partition(".")
    bound = _bindings(ast.parse(path.read_text(encoding="utf-8")).body).get(owner_name, [])
    if not bound:
        return f"the module binds no {owner_name}"
    if kind == "name":
        return None
    # A name may be bound more than once: SGLang writes one of its classes as a class statement and then
    # rebuilds it from the namespace of that statement. The class statement is what holds the members.
    classes = [node for node in bound if isinstance(node, ast.ClassDef)]
    if not classes:
        return f"{owner_name} is not a class statement"
    if kind == "class":
        return None
    members = [(owner, node) for owner in classes for node in _bindings(owner.body).get(member, [])]
    if kind == "attribute":
        valued = [node for _owner, node in members if not (isinstance(node, ast.AnnAssign) and node.value is None)]
        return None if valued else f"class {owner_name} binds no {member}"
    if kind == "field":
        fields = [node for owner, node in members if _is_dataclass_statement(owner) and isinstance(node, ast.AnnAssign)]
        return None if fields else f"class {owner_name} is no dataclass with a field {member}"
    return f"the kind {kind!r} is not known"


def seam_rows_not_in_source(rows, python_root: Path) -> list:
    """The rows whose name is not written in the SGLang source under ``python_root``, each with the reason.

    The source is parsed, never imported. A "name" is any module-level binding, an import included. A
    "class" is a class statement. An "attribute" is bound in the body of that class statement, with a value.
    A "field" is an annotated name in the body of a class statement that is decorated as a dataclass.
    """
    problems = []
    for module, name, kind in rows:
        problem = _source_problem(python_root, module, name, kind)
        if problem is not None:
            problems.append(f"{module}: {kind} {name}: {problem}")
    return problems


# ---------------------------------------------------------------------------------------------------------------
# The FAKE tree
# ---------------------------------------------------------------------------------------------------------------


def _fake_module(monkeypatch, name: str) -> types.ModuleType:
    """The module ``name`` of the FAKE tree, made with its parents when the tree has none."""
    parent = None
    parts = name.split(".")
    for end in range(1, len(parts) + 1):
        prefix = ".".join(parts[:end])
        module = sys.modules.get(prefix)
        if module is None:
            module = types.ModuleType(prefix)
            # A package without a search path: a submodule that is taken away is then simply not found.
            module.__path__ = []
            monkeypatch.setitem(sys.modules, prefix, module)
            if parent is not None:
                monkeypatch.setattr(parent, parts[end - 1], module, raising=False)
        parent = module
    return parent


def _complete_fake_tree(monkeypatch) -> None:
    """Add to the FAKE tree whatever the seam rows name and it lacks, as FAKE placeholders.

    The tree of test_fpm_sglbench_compat.py has the names sglbench_compat.py imports. The rows also name
    methods of its classes and everything sglbench_recorder.py takes, so the rest comes from the rows: a
    module, a placeholder for a name, an empty class, an attribute on a class, and for the rows of kind
    "field" a dataclass whose fields have no default, which makes none of them an attribute of the class.
    """
    fields: dict = {}
    for _source, module_name, name, kind in ALL_ROWS:
        if kind == "field":
            owner_name, _, member = name.partition(".")
            fields.setdefault((module_name, owner_name), []).append(member)
    for _source, module_name, name, kind in ALL_ROWS:
        module = _fake_module(monkeypatch, module_name)
        owner_name, _, member = name.partition(".")
        if kind == "name":
            if not hasattr(module, name):
                monkeypatch.setattr(module, name, FAKE_VALUE, raising=False)
            continue
        if not hasattr(module, owner_name):
            owner_fields = fields.get((module_name, owner_name))
            if owner_fields:
                owner = dataclasses.make_dataclass(owner_name, [(field, object) for field in owner_fields])
            else:
                owner = type(owner_name, (), {})
            monkeypatch.setattr(module, owner_name, owner, raising=False)
        if kind == "attribute" and not hasattr(getattr(module, owner_name), member):
            monkeypatch.setattr(getattr(module, owner_name), member, FAKE_VALUE, raising=False)


@pytest.fixture
def fake_tree(monkeypatch, tmp_path):
    """A complete FAKE ``sglang`` tree in ``sys.modules`` and the audit in a folder of the test."""
    # Imported here: that module imports the helpers above from this one.
    from .test_fpm_sglbench_compat import _install_fake_sglang

    state = _install_fake_sglang(monkeypatch)
    _complete_fake_tree(monkeypatch)
    monkeypatch.setattr(preflight, "_AUDIT_PATH", tmp_path / "runtime-preflight.json")
    return state


def _audit() -> dict:
    return json.loads(preflight._AUDIT_PATH.read_text(encoding="utf-8"))


def _expected_rows(not_found=()) -> list:
    return [
        {
            "source": source,
            "module": module,
            "name": name,
            "kind": kind,
            "found": (source, module, name, kind) not in not_found,
        }
        for source, module, name, kind in ALL_ROWS
    ]


def _failure_lines(captured) -> list:
    prefix = "[sglbench] preflight FAILED: "
    lines = captured.err.splitlines()
    assert all(line.startswith(prefix) for line in lines), lines
    return [line[len(prefix) :] for line in lines]


def _fails(capsys, not_found=()) -> list:
    """Run the preflight, require the failed outcome with exactly ``not_found`` missing, return the failures."""
    assert preflight.main() == 1
    captured = capsys.readouterr()
    audit = _audit()
    assert captured.out == ""
    assert audit["status"] == "failed"
    assert audit["failures"] == _failure_lines(captured) and audit["failures"]
    assert audit["seam_names"] == _expected_rows(not_found)
    return audit["failures"]


# ---------------------------------------------------------------------------------------------------------------
# A complete tree passes
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("version", compat.TESTED_VERSIONS)
def test_a_tested_release_with_every_name_passes_and_the_audit_says_what_was_looked_up(fake_tree, capsys, version):
    fake_tree.modules[ROOT_PACKAGE].__version__ = version
    assert preflight.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == f"[sglbench] preflight passed: SGLang {version}, {len(ALL_ROWS)} names found\n"
    assert _audit() == {
        "schema_version": 1,
        "runtime_contract": "sglbench_launcher_seam_v1",
        "launcher_modules": list(preflight.LAUNCHER_MODULES),
        "sglang_version": version,
        "tested_versions": list(compat.TESTED_VERSIONS),
        "seam_names": _expected_rows(),
        "failures": [],
        "status": "passed",
    }
    # Nothing of SGLang was called: the FAKE tree records every call.
    assert fake_tree.events == [] and fake_tree.parsed == [] and fake_tree.engine_kwargs == []


# ---------------------------------------------------------------------------------------------------------------
# The version
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["0.5.24", "0.5.23.post1"])
def test_an_untested_release_fails_and_the_names_are_still_looked_up(fake_tree, capsys, version):
    fake_tree.modules[ROOT_PACKAGE].__version__ = version
    assert _fails(capsys) == [f"SGLang {version} is not a tested version (tested: {', '.join(compat.TESTED_VERSIONS)})"]
    audit = _audit()
    assert audit["sglang_version"] == version and audit["tested_versions"] == list(compat.TESTED_VERSIONS)


# ---------------------------------------------------------------------------------------------------------------
# One thing is missing
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", [name for name in SEAM_MODULE_NAMES if name != ROOT_PACKAGE])
def test_a_module_that_does_not_import_fails_once_and_none_of_its_names_is_found(
    fake_tree, monkeypatch, capsys, module_name
):
    gone = [name for name in list(sys.modules) if name == module_name or name.startswith(module_name + ".")]
    for name in gone:
        monkeypatch.delitem(sys.modules, name)
    not_found = [row for row in ALL_ROWS if row[1] in gone]
    failures = _fails(capsys, not_found)
    # One line for each module that is gone and that a row names, in the order of the rows.
    expected = list(dict.fromkeys(row[1] for row in not_found))
    assert len(failures) == len(expected)
    for failure, name in zip(failures, expected, strict=True):
        assert failure.startswith(f"cannot import {name}: ModuleNotFoundError: "), failure


@pytest.mark.parametrize("row", ROWS_OF_KIND["name"], ids=_row_id)
def test_a_missing_module_level_name_fails(fake_tree, monkeypatch, capsys, row):
    _source, module_name, name, _kind = row
    monkeypatch.delattr(sys.modules[module_name], name)
    failures = _fails(capsys, [other for other in ALL_ROWS if other[1:3] == (module_name, name)])
    assert f"{module_name}: name {name} is missing" in failures
    # The version is itself a module-level name, so taking it away fails the version check as well.
    others = [failure for failure in failures if failure != f"{module_name}: name {name} is missing"]
    assert all(failure.startswith("cannot read the SGLang version: AttributeError: ") for failure in others)
    assert bool(others) == (module_name == ROOT_PACKAGE)


@pytest.mark.parametrize("row", ROWS_OF_KIND["class"], ids=_row_id)
def test_a_class_that_is_missing_or_is_no_class_fails(fake_tree, monkeypatch, capsys, row):
    _source, module_name, name, _kind = row
    module = sys.modules[module_name]
    members = [other for other in ALL_ROWS if other[1] == module_name and other[2].startswith(name + ".")]

    # An object in the place of the class, with every member the rows name: only the class row fails.
    monkeypatch.setattr(module, name, types.SimpleNamespace(**{other[2].partition(".")[2]: 1 for other in members}))
    assert _fails(capsys, [row]) == [f"{module_name}: class {name} is not a class"]

    monkeypatch.delattr(module, name)
    failures = _fails(capsys, [row, *members])
    assert failures[0] == f"{module_name}: class {name} is missing"


@pytest.mark.parametrize("row", ROWS_OF_KIND["attribute"], ids=_row_id)
def test_a_missing_attribute_of_a_class_fails(fake_tree, monkeypatch, capsys, row):
    _source, module_name, name, _kind = row
    owner_name, _, member = name.partition(".")
    monkeypatch.delattr(getattr(sys.modules[module_name], owner_name), member)
    assert _fails(capsys, [row]) == [f"{module_name}: attribute {name} is missing"]


@pytest.mark.parametrize("row", ROWS_OF_KIND["field"], ids=_row_id)
def test_a_missing_dataclass_field_fails_even_when_the_class_has_an_attribute_of_that_name(
    fake_tree, monkeypatch, capsys, row
):
    _source, module_name, name, _kind = row
    owner_name, _, member = name.partition(".")
    module = sys.modules[module_name]
    kept = [field.name for field in dataclasses.fields(getattr(module, owner_name)) if field.name != member]
    owner = dataclasses.make_dataclass(owner_name, [(field, object) for field in kept])
    # A class attribute of the same name is not a field: the object SGLang builds would not carry it.
    setattr(owner, member, FAKE_VALUE)
    monkeypatch.setattr(module, owner_name, owner)
    assert _fails(capsys, [row]) == [f"{module_name}: field {name} is missing"]


# ---------------------------------------------------------------------------------------------------------------
# The launcher's own files
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sibling", preflight.LAUNCHER_MODULES)
def test_a_launcher_module_that_does_not_import_fails(fake_tree, monkeypatch, capsys, sibling):
    # Every module is loaded first, so that only the one named here fails. None in sys.modules then makes its
    # import raise, as a file that is not there would.
    for name in preflight.LAUNCHER_MODULES:
        importlib.import_module(name)
    monkeypatch.setitem(sys.modules, sibling, None)
    unread = [row for row in ALL_ROWS if row[0] == sibling]
    assert preflight.main() == 1
    failures = _failure_lines(capsys.readouterr())
    audit = _audit()
    assert audit["status"] == "failed" and audit["failures"] == failures
    assert len(failures) == 1 and failures[0].startswith(f"cannot import {sibling}: ModuleNotFoundError: ")
    # The rows of a module that did not import cannot be read, so they are not in the audit at all.
    assert audit["seam_names"] == [record for record in _expected_rows() if record["source"] != sibling]
    assert bool(unread) == (sibling in preflight.SEAM_MODULES)
    version_read = sibling != preflight.VERSION_MODULE
    assert audit["sglang_version"] == (fake_tree.modules[ROOT_PACKAGE].__version__ if version_read else None)
    assert audit["tested_versions"] == (list(compat.TESTED_VERSIONS) if version_read else [])


# ---------------------------------------------------------------------------------------------------------------
# The module's own lists and the form of the rows
# ---------------------------------------------------------------------------------------------------------------


def test_the_launcher_modules_are_the_runtime_files_and_the_seam_modules_are_those_that_import_sglang():
    siblings = sorted(path.stem for path in RUNTIME.glob("sglbench_*.py") if path != MODULE_PATH)
    assert list(preflight.LAUNCHER_MODULES) == siblings
    importers = []
    for name in siblings:
        tree = ast.parse((RUNTIME / f"{name}.py").read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                roots.add((node.module or "").split(".")[0])
        if ROOT_PACKAGE in roots:
            importers.append(name)
    # A launcher module that starts to import SGLang needs a seam constant and a place in SEAM_MODULES.
    assert list(preflight.SEAM_MODULES) == importers


def test_every_kind_a_row_gives_is_one_the_preflight_knows_and_every_kind_is_used():
    assert set(ROWS_OF_KIND) == set(preflight._KIND_PROBLEMS) == {"name", "class", "attribute", "field"}
    for _source, module_name, name, kind in ALL_ROWS:
        assert module_name.split(".")[0] == ROOT_PACKAGE
        assert name.count(".") == (1 if kind in ("attribute", "field") else 0), name
