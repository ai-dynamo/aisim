# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight of AISim's SGLang self-benchmark: fail before the model is loaded unless the launcher can run.

Where it runs
    In the SGLang pod, as a process of its own, started by path by the collector's gate in the place of
    preflight.py (collector/fpm_forward/runtime/fpm_exec.sh): ``python3 <dir>/sglbench_preflight.py``. AISim is
    not installed there. At import the module needs the standard library only. The launcher's modules lie
    flat in the same directory and are found because Python puts the directory of a script started by path
    on ``sys.path``. With ``-I``, ``-P``, ``PYTHONSAFEPATH`` or ``-m`` it does not, and the launcher would not
    start either (sglbench_launcher.py).

What it checks
    1. Every module of ``LAUNCHER_MODULES`` imports. That proves the staging and the interpreter.
    2. The installed SGLang reports a version of ``sglbench_compat.TESTED_VERSIONS``. The launcher refuses
       every other release too, with a failed artifact; here the refusal comes before ``run.sh`` is started.
    3. Every row of ``SGLANG_SEAM_NAMES`` in sglbench_compat.py and in sglbench_recorder.py is there: its
       module imports and holds the name as the kind the row gives. The rows are kept beside the code that
       uses the names. This file walks them and holds no SGLang name of its own.

    All three run to their end, so one start reports everything that is wrong with an image.

What it does not check
    What only a built engine or a running scheduler holds; the two constants say what that is. It builds no
    engine and loads no model. Of SGLang it imports the modules of the rows and calls the version accessor of
    sglbench_compat.py, nothing else. In the SGLang source of both tested releases the engine module imports
    every other module of the rows, so nothing is imported here that the launcher's process does not import
    itself before it builds its engine.

Not run
    With SGLang installed. The tests use a FAKE package tree, and the rows were read in the SGLang source.

Result
    ``/results/runtime-preflight.json`` on every outcome: the launcher modules, the version, the tested
    versions, every row with whether it was found, and every failure. One line on stdout when nothing
    failed, one line per failure on stderr otherwise. The exit status is 0 or 1. The gate runs under
    ``set -e``, so a failure ends the cell before the engine is started.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import sys
from pathlib import Path

_AUDIT_PATH = Path("/results/runtime-preflight.json")

AUDIT_SCHEMA_VERSION = 1
RUNTIME_CONTRACT = "sglbench_launcher_seam_v1"

# The launcher's files. The collector stages them flat beside this one.
LAUNCHER_MODULES = (
    "sglbench_artifact",
    "sglbench_compat",
    "sglbench_content",
    "sglbench_driver",
    "sglbench_grid",
    "sglbench_launcher",
    "sglbench_recorder",
)

# The module that reads the installed version and names the tested ones, and the modules that hold a
# constant of seam rows.
VERSION_MODULE = "sglbench_compat"
SEAM_MODULES = ("sglbench_compat", "sglbench_recorder")
SEAM_CONSTANT = "SGLANG_SEAM_NAMES"

# What an import or the version accessor may raise and this file records as a failure. ``SystemExit`` is in
# it: a module that ends the interpreter while it is imported would otherwise end this process with ITS
# status, 0 included, before the audit is written, and the gate would take that for a preflight that passed.
_RECORDED_ERRORS = (Exception, SystemExit)


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _name_problem(module: object, name: str) -> str | None:
    return None if hasattr(module, name) else "is missing"


def _class_problem(module: object, name: str) -> str | None:
    if not hasattr(module, name):
        return "is missing"
    return None if isinstance(getattr(module, name), type) else "is not a class"


def _attribute_problem(module: object, name: str) -> str | None:
    owner_name, _, member = name.partition(".")
    if not member:
        return "does not have the form Owner.attribute"
    if not hasattr(module, owner_name):
        return f"is missing: the module has no {owner_name}"
    return None if hasattr(getattr(module, owner_name), member) else "is missing"


def _field_problem(module: object, name: str) -> str | None:
    owner_name, _, member = name.partition(".")
    if not member:
        return "does not have the form Owner.field"
    if not hasattr(module, owner_name):
        return f"is missing: the module has no {owner_name}"
    owner = getattr(module, owner_name)
    if not isinstance(owner, type) or not dataclasses.is_dataclass(owner):
        return f"cannot be looked up: {owner_name} is not a dataclass"
    # Not ``hasattr``: a field without a default is no attribute of its class.
    return None if member in {field.name for field in dataclasses.fields(owner)} else "is missing"


# The kinds a seam row may give, each with the test that returns what is wrong or None.
_KIND_PROBLEMS = {
    "name": _name_problem,
    "class": _class_problem,
    "attribute": _attribute_problem,
    "field": _field_problem,
}


class _SeamLookup:
    """Looks seam rows up. A module is imported once, and a module or a row that fails is reported once."""

    def __init__(self, failures: list[str]):
        self._failures = failures
        self._modules: dict[str, object] = {}
        self._found: dict[tuple[str, str, str], bool] = {}

    def found(self, module_name: str, name: str, kind: str) -> bool:
        row = (module_name, name, kind)
        if row not in self._found:
            self._found[row] = self._look_up(*row)
        return self._found[row]

    def _module(self, module_name: str) -> object:
        """The imported module, or None when its import failed."""
        if module_name not in self._modules:
            try:
                self._modules[module_name] = importlib.import_module(module_name)
            except _RECORDED_ERRORS as error:
                self._modules[module_name] = None
                self._failures.append(f"cannot import {module_name}: {_describe(error)}")
        return self._modules[module_name]

    def _look_up(self, module_name: str, name: str, kind: str) -> bool:
        module = self._module(module_name)
        if module is None:
            return False
        try:
            problem = _KIND_PROBLEMS[kind](module, name)
        except Exception as error:
            problem = f"cannot be looked up: {_describe(error)}"
        if problem is not None:
            self._failures.append(f"{module_name}: {kind} {name} {problem}")
        return problem is None


def _import_launcher_modules(failures: list[str]) -> dict[str, object]:
    modules = {}
    for name in LAUNCHER_MODULES:
        try:
            modules[name] = importlib.import_module(name)
        except _RECORDED_ERRORS as error:
            failures.append(f"cannot import {name}: {_describe(error)}")
    return modules


def _check_version(compat: object, failures: list[str]) -> tuple[str | None, list[str]]:
    """Return the installed version and the tested ones; an untested or unreadable version is a failure."""
    tested = list(compat.TESTED_VERSIONS)
    try:
        version = compat.sglang_version()
    except _RECORDED_ERRORS as error:
        failures.append(f"cannot read the SGLang version: {_describe(error)}")
        return None, tested
    if version not in tested:
        failures.append(f"SGLang {version} is not a tested version (tested: {', '.join(tested)})")
    return version, tested


def _check_seam_rows(source: str, rows: object, lookup: _SeamLookup, failures: list[str]) -> list[dict]:
    """Look up every row of one constant and return what was looked up, in the order of the rows."""
    # A constant that is gone or empty must not pass as "nothing to look up".
    if not isinstance(rows, tuple) or not rows:
        failures.append(f"{source} has no {SEAM_CONSTANT}")
        return []
    checked = []
    for row in rows:
        if not isinstance(row, tuple) or len(row) != 3 or not all(isinstance(part, str) and part for part in row):
            failures.append(f"{source} holds the row {row!r}, which is not (module, name, kind)")
            continue
        module_name, name, kind = row
        record = {"source": source, "module": module_name, "name": name, "kind": kind, "found": False}
        checked.append(record)
        if kind not in _KIND_PROBLEMS:
            failures.append(f"{source} gives {module_name} {name} the unknown kind {kind!r}")
            continue
        record["found"] = lookup.found(module_name, name, kind)
    return checked


def run_checks() -> dict:
    """Run the three checks and return the audit. Nothing is written here."""
    failures: list[str] = []
    modules = _import_launcher_modules(failures)

    version, tested = None, []
    if VERSION_MODULE in modules:
        version, tested = _check_version(modules[VERSION_MODULE], failures)

    seam_names: list[dict] = []
    lookup = _SeamLookup(failures)
    for source in SEAM_MODULES:
        # A module that did not import has its failure above; its rows cannot be read.
        if source in modules:
            rows = getattr(modules[source], SEAM_CONSTANT, None)
            seam_names.extend(_check_seam_rows(source, rows, lookup, failures))

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "runtime_contract": RUNTIME_CONTRACT,
        "launcher_modules": list(LAUNCHER_MODULES),
        "sglang_version": version,
        "tested_versions": tested,
        "seam_names": seam_names,
        "failures": failures,
        "status": "failed" if failures else "passed",
    }


def _write_audit(audit: dict) -> None:
    _AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    audit = run_checks()
    failures = list(audit["failures"])
    try:
        _write_audit(audit)
    except OSError as error:
        # The audit is the record of what this image was tested for, so a run without it does not pass.
        failures.append(f"cannot write {_AUDIT_PATH}: {_describe(error)}")
    if failures:
        for failure in failures:
            print(f"[sglbench] preflight FAILED: {failure}", file=sys.stderr, flush=True)
        return 1
    print(
        f"[sglbench] preflight passed: SGLang {audit['sglang_version']}, {len(audit['seam_names'])} names found",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
