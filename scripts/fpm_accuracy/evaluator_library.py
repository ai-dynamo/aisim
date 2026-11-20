# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind ``aisim_core.data`` to this checkout, whatever AISim wheel is installed.

The accuracy campaign evaluates the AISim wheel of the *evaluated* commit, which may be an older release
branch that predates the shared dataset library. Dataset access (catalog, selection, fetch, verification) is an
evaluator concern, so it always comes from the evaluator's own checkout. Everything else in ``aisim_core``
(the native forward-pass SDK under test) keeps coming from the installed wheel.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "python/aisim/src"


# Installed wheels expose their SDK under ``aisim_core`` or, before the renames, ``aisimulate_core`` and then
# ``aiconfigurator_core`` (the order ``scripts.release.package_identity.runtime_module`` uses for the first two).
HISTORICAL_NAMESPACES = ("aisimulate_core", "aiconfigurator_core")


def _installed_wheel_namespace():
    """Alias the first historical wheel SDK namespace as ``aisim_core``; None when no wheel provides one."""
    for name in HISTORICAL_NAMESPACES:
        try:
            package = importlib.import_module(name)
            sdk = importlib.import_module(f"{name}.sdk")
        except ModuleNotFoundError as error:
            # A shim that re-exports ``aisim_core`` (this checkout's, or a current wheel's) cannot supply a missing
            # one; any other missing module means the wheel is present but broken: do not mask it with the checkout.
            if error.name not in {name, f"{name}.sdk", "aisim_core"}:
                raise
            continue
        # Alias the package, preserving its module paths and active database state.
        sys.modules["aisim_core"] = package
        sys.modules["aisim_core.sdk"] = sdk
        return package
    return None


def bind() -> None:
    try:
        import aisim_core
    except ImportError:
        aisim_core = _installed_wheel_namespace()
        if aisim_core is None:  # no installed wheel provides an SDK namespace: use the evaluator's source tree
            sys.path.insert(0, str(SOURCE))
            import aisim_core
    package = SOURCE / "aisim_core/data"
    loaded = sys.modules.get("aisim_core.data")
    if loaded is not None and Path(loaded.__file__).resolve() == package / "__init__.py":
        return
    if loaded is not None:
        for name in [n for n in sys.modules if n == "aisim_core.data" or n.startswith("aisim_core.data.")]:
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        "aisim_core.data", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["aisim_core.data"] = module
    spec.loader.exec_module(module)
    aisim_core.data = module


bind()
