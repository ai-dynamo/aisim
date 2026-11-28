# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logical-to-physical path rule for FPM files after the move to ``data/fpm/``.

The move was byte-identical, so 202 text files (every leaf ``manifest.json``, FPM
metadata, import records) still embed the legacy ``data/<model>/...`` spelling.
Those strings are *logical keys*: they keep configuration, case and observation
identities stable across the move. The physical repository path is
``data/fpm/<rest>`` (``data/history/...`` becomes ``data/fpm/history/...``).
"""

from __future__ import annotations

from collections.abc import Mapping

LEGACY_PREFIX = "data/"
PHYSICAL_PREFIX = "data/fpm/"
# Roots of the other dataset categories. A key under one is never a legacy FPM key, even when the pinned catalog
# does not declare that category (an execution-profile bundle can reference operation tables by their own path).
OTHER_CATEGORY_PREFIXES = ("data/op-latency/", "data/e2e/", "data/eval/", "data/systems/", "data/host-cost/")


def physical_path(logical: str) -> str:
    """``data/<rest>`` -> ``data/fpm/<rest>``; an already physical path is returned unchanged."""
    if logical.startswith(PHYSICAL_PREFIX):
        return logical
    if not logical.startswith(LEGACY_PREFIX):
        raise ValueError(f"not a dataset path: {logical!r}")
    return PHYSICAL_PREFIX + logical[len(LEGACY_PREFIX) :]


def logical_path(physical: str) -> str:
    """Inverse of :func:`physical_path` (the legacy spelling stored inside moved files)."""
    if not physical.startswith(PHYSICAL_PREFIX):
        raise ValueError(f"not a physical FPM path: {physical!r}")
    return LEGACY_PREFIX + physical[len(PHYSICAL_PREFIX) :]


def storage_path(path: str, category_roots: Mapping[str, str]) -> str:
    """Resolve moved FPM keys using the pinned catalog's layout, never file-existence fallback.

    This also works after a symbolic retired-repository reference has resolved to an immutable successor SHA.
    Historical catalogs and paths already under a declared category root remain verbatim.
    """
    if (
        category_roots.get("fpm") == PHYSICAL_PREFIX.rstrip("/")
        and path.startswith(LEGACY_PREFIX)
        and not path.startswith(OTHER_CATEGORY_PREFIXES)
        and not any(path == root or path.startswith(root + "/") for root in category_roots.values())
    ):
        return physical_path(path)
    return path
