# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where production operation data lives now that the wheel and source tree no longer package it.

The tools in this directory and the prediction-regression gate read ``<systems root>/data``. That root is the one
named by ``AICONFIGURATOR_SYSTEMS_PATH`` when set, otherwise the default systems root composed from the pinned
op-latency dataset (downloaded into the data cache on first use, ``AISIM_DATA_OFFLINE=1`` for a warm cache only).
"""

from __future__ import annotations

import os
from pathlib import Path


def systems_root() -> Path:
    """The systems root the SDK reads by default (never a path inside the source tree)."""
    configured = os.environ.get("AICONFIGURATOR_SYSTEMS_PATH")
    if configured:
        return Path(configured)
    from aisim_core.data.default_systems import default_systems_path

    return Path(default_systems_path())


def op_data_root(explicit: Path | str | None = None) -> Path:
    """``explicit`` when given (a ``--data-root`` flag), otherwise ``<systems root>/data``."""
    return Path(explicit) if explicit is not None else systems_root() / "data"
