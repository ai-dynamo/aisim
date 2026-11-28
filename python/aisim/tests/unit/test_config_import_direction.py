# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``aisim.config`` reaches into ``aisim.sweeper`` only from inside known functions.

``aisim.sweeper`` imports ``aisim.config`` at module scope, so the dependency runs
sweeper -> config. Importing ``aisim.config`` must not load the sweeper. A few
config functions need sweeper code, so they import it when they are called. This
module rejects any module-scope import of the sweeper from config. It also pins
the function-local imports, so that adding a new one is a deliberate choice.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal, NamedTuple

import pytest

pytestmark = pytest.mark.unit

CONFIG_ROOT = Path(__file__).parents[2] / "src" / "aisim" / "config"
SWEEPER = "aisim.sweeper"
THIS_FILE = "python/aisim/tests/unit/test_config_import_direction.py"

#: ``(file under aisim/config, enclosing function, imported sweeper module)`` for
#: every function-local import from ``aisim.config`` into ``aisim.sweeper``.
FUNCTION_LOCAL_SWEEPER_IMPORTS = frozenset(
    {
        ("ais_perf_model.py", "written_old_keys", "aisim.sweeper.ais_perf_model"),
        ("ais_perf_model_migrate.py", "_SearchLayout.model", "aisim.sweeper.config"),
        ("ais_perf_model_migrate.py", "_SearchLayout.uncovered_reason", "aisim.sweeper.ais_perf_model"),
        ("ais_perf_model_migrate.py", "_SearchLayout.worker_types", "aisim.sweeper.ais_perf_model"),
        ("epd.py", "_language_execution", "aisim.sweeper.deploy"),
        ("epd.py", "_language_execution", "aisim.sweeper.forward_pass_estimator"),
        ("vl.py", "_execution_traffic", "aisim.sweeper.epd"),
    }
)

Where = Literal["module", "function", "type_checking"]


class SweeperImport(NamedTuple):
    file: str
    line: int
    #: ``module`` runs when the config module is imported, ``function`` runs when
    #: the enclosing function is called, and ``type_checking`` never runs.
    where: Where
    #: Qualified name of the enclosing class or function; empty at module level.
    scope: str
    module: str


def _is_sweeper(module: str) -> bool:
    return module == SWEEPER or module.startswith(f"{SWEEPER}.")


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _sweeper_modules(node: ast.Import | ast.ImportFrom, package: tuple[str, ...]) -> list[str]:
    """The sweeper modules an import statement loads, with relative imports resolved."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names if _is_sweeper(alias.name)]
    if node.level > len(package):
        return []
    parent = package[: len(package) - node.level + 1] if node.level else ()
    base = ".".join((*parent, *([node.module] if node.module else [])))
    if _is_sweeper(base):
        return [base]
    # ``from aisim import sweeper`` and ``from .. import sweeper`` load the package itself.
    return [f"{base}.{alias.name}" for alias in node.names if _is_sweeper(f"{base}.{alias.name}")]


class _Scanner(ast.NodeVisitor):
    def __init__(self, rel_path: Path) -> None:
        self.file = rel_path.as_posix()
        self.package = ("aisim", "config", *rel_path.parent.parts)
        self.scope: list[str] = []
        self.function_depth = 0
        self.type_checking = False
        self.found: list[SweeperImport] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_If(self, node: ast.If) -> None:
        if not _is_type_checking(node.test):
            self.generic_visit(node)
            return
        outer, self.type_checking = self.type_checking, True
        for child in node.body:
            self.visit(child)
        self.type_checking = outer
        # The ``else`` branch of ``if TYPE_CHECKING:`` runs.
        for child in node.orelse:
            self.visit(child)

    def visit_Import(self, node: ast.Import | ast.ImportFrom) -> None:
        where: Where = "type_checking" if self.type_checking else "function" if self.function_depth else "module"
        for module in _sweeper_modules(node, self.package):
            self.found.append(SweeperImport(self.file, node.lineno, where, ".".join(self.scope), module))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.visit_Import(node)


def _scan(rel_path: Path, source: str) -> list[SweeperImport]:
    scanner = _Scanner(rel_path)
    scanner.visit(ast.parse(source, filename=str(rel_path)))
    return scanner.found


def _config_sweeper_imports() -> list[SweeperImport]:
    paths = sorted(CONFIG_ROOT.rglob("*.py"))
    assert paths, f"no aisim.config modules found under {CONFIG_ROOT}"
    return [found for path in paths for found in _scan(path.relative_to(CONFIG_ROOT), path.read_text(encoding="utf-8"))]


def test_config_does_not_import_sweeper_at_module_scope() -> None:
    offenders = [
        f"  aisim/config/{found.file}:{found.line} imports {found.module}"
        for found in _config_sweeper_imports()
        if found.where == "module"
    ]
    assert not offenders, "\n".join(
        [
            "aisim.config imports aisim.sweeper when it is imported:",
            *offenders,
            "aisim.sweeper imports aisim.config at module scope, so the dependency runs sweeper -> config.",
            "A module-scope import back makes a cycle, and every import of aisim.config then loads the sweeper.",
            "Prefer moving the code config needs into aisim.config, or into a module both packages import.",
            "If config really needs sweeper code, import it inside the function that uses it,",
            "or under `if TYPE_CHECKING:` when only annotations need it.",
            f"Add a new function-local import to FUNCTION_LOCAL_SWEEPER_IMPORTS in {THIS_FILE}.",
        ]
    )


def test_function_local_sweeper_imports_match_the_frozen_set() -> None:
    found = {(each.file, each.scope, each.module) for each in _config_sweeper_imports() if each.where == "function"}
    added = sorted(found - FUNCTION_LOCAL_SWEEPER_IMPORTS)
    removed = sorted(FUNCTION_LOCAL_SWEEPER_IMPORTS - found)
    assert not added and not removed, "\n".join(
        [
            "The function-local imports from aisim.config into aisim.sweeper changed (+ new, - gone):",
            *(f"  + {entry!r}" for entry in added),
            *(f"  - {entry!r}" for entry in removed),
            "aisim.sweeper imports aisim.config, so the dependency runs sweeper -> config.",
            "Each entry in FUNCTION_LOCAL_SWEEPER_IMPORTS is a deliberate exception to that direction.",
            "Before adding one, try moving the code config needs into aisim.config,",
            "or into a module both packages import.",
            f"If the import must stay, add its triple to FUNCTION_LOCAL_SWEEPER_IMPORTS in {THIS_FILE},",
            "and say why in the pull request.",
            "If an import was removed, or its function renamed, update the triples to match.",
        ]
    )


@pytest.mark.parametrize(
    ("path", "source", "expected"),
    [
        (Path("x.py"), "def f():\n    from aisim.sweeper.replay import ReplaySpec\n", "aisim.sweeper.replay"),
        (Path("x.py"), "def f():\n    from aisim.sweeper import replay\n", "aisim.sweeper"),
        (Path("x.py"), "def f():\n    from aisim import sweeper\n", "aisim.sweeper"),
        (Path("x.py"), "def f():\n    import aisim.sweeper.replay\n", "aisim.sweeper.replay"),
        (Path("x.py"), "def f():\n    import aisim.sweeper.replay as replay\n", "aisim.sweeper.replay"),
        (Path("x.py"), "def f():\n    from ..sweeper.replay import ReplaySpec\n", "aisim.sweeper.replay"),
        (Path("x.py"), "def f():\n    from .. import sweeper\n", "aisim.sweeper"),
        (Path("sub/x.py"), "def f():\n    from ...sweeper import epd\n", "aisim.sweeper"),
    ],
)
def test_scanner_resolves_every_spelling(path: Path, source: str, expected: str) -> None:
    assert _scan(path, source) == [SweeperImport(path.as_posix(), 2, "function", "f", expected)]


@pytest.mark.parametrize(
    "source",
    [
        "from .sweeper import x\n",
        "from aisim.config import sweeper\n",
        "import aisim.sweeperish\n",
        "from aisim import config\n",
        "from . import epd\n",
        "x = 'aisim.sweeper'\n",
    ],
)
def test_scanner_ignores_modules_outside_the_sweeper(source: str) -> None:
    assert _scan(Path("x.py"), source) == []


@pytest.mark.parametrize(
    ("source", "where", "scope"),
    [
        ("from ..sweeper import replay\n", "module", ""),
        ("try:\n    from ..sweeper import replay\nexcept ImportError:\n    pass\n", "module", ""),
        ("class C:\n    from ..sweeper import replay\n", "module", "C"),
        ("if TYPE_CHECKING:\n    from ..sweeper import replay\n", "type_checking", ""),
        ("if typing.TYPE_CHECKING:\n    from ..sweeper import replay\n", "type_checking", ""),
        ("if TYPE_CHECKING:\n    pass\nelse:\n    from ..sweeper import replay\n", "module", ""),
        ("def f():\n    if TYPE_CHECKING:\n        from ..sweeper import replay\n", "type_checking", "f"),
        ("async def f():\n    from ..sweeper import replay\n", "function", "f"),
        ("class C:\n    def m(self):\n        from ..sweeper import replay\n", "function", "C.m"),
        ("def f():\n    def g():\n        from ..sweeper import replay\n", "function", "f.g"),
    ],
)
def test_scanner_classifies_when_an_import_runs(source: str, where: Where, scope: str) -> None:
    [found] = _scan(Path("x.py"), source)
    assert (found.where, found.scope, found.module) == (where, scope, "aisim.sweeper")
