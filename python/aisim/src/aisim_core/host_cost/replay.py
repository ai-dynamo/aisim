# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measured serving-path CPU overhead for replay, read from a pinned ``host-cost`` dataset revision.

:func:`measured_overhead` returns the ``engine.tokenization`` and ``engine.extra_cpu_overhead`` blocks of one
deployment, keyed by (backend, backend version, endpoint, host CPU, tokenizer), or ``None`` for a deployment that has
not been measured: there is no default. The host CPU is the caller's: a system name does not say which CPU its
serving nodes have, so it is matched exactly against the CPU each entry was measured on. The data comes through
:func:`aisim_core.data.ensure.ensure_data` with a ``host-cost`` pin:

* ``AISIM_HOST_COST_LOCAL=<dir>``: a local dataset-layout directory (no network),
* ``AISIM_HOST_COST_DATASET=<repo>@<revision>``: another revision (a full commit SHA reuses the cache offline),
* otherwise the revision in ``host_cost_pin.json``, which ships in the wheel.

``AISIM_DATA_OFFLINE``, ``AISIM_DATA_CACHE`` and ``AISIM_DATA_MAX_BYTES`` apply as for other pins. Missing or
unverifiable data for a pinned revision raises :class:`~aisim_core.data.ensure.MissingDataError`; it never falls back
to other data.
"""

from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

from aisim_core.data.ensure import DEFAULT_MAX_BYTES, DataPin, ensure_data
from aisim_core.host_cost.lookup import HostOverhead

PIN_FILE = Path(__file__).resolve().parents[1] / "data" / "host_cost_pin.json"
DATASET_ENV = "AISIM_HOST_COST_DATASET"
LOCAL_ENV = "AISIM_HOST_COST_LOCAL"
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def host_cost_pin() -> tuple[str, str]:
    """``(repo, full commit SHA)`` this release reads host-cost data from (``host_cost_pin.json``)."""
    pin = json.loads(PIN_FILE.read_text(encoding="utf-8"))
    repo, revision = pin["repo_id"], pin["revision"]
    if not _FULL_SHA.fullmatch(revision):
        raise ValueError(f"{PIN_FILE.name} must pin a full commit SHA, got {revision!r}")
    return repo, revision


def _pin(env: Mapping[str, str]) -> DataPin:
    common = {
        "categories": ("host-cost",),
        "offline": env.get("AISIM_DATA_OFFLINE", "").lower() in {"1", "true", "yes"},
        "cache_dir": Path(env["AISIM_DATA_CACHE"]) if env.get("AISIM_DATA_CACHE") else None,
        "max_bytes": int(env["AISIM_DATA_MAX_BYTES"]) if env.get("AISIM_DATA_MAX_BYTES") else DEFAULT_MAX_BYTES,
    }
    if env.get(LOCAL_ENV):
        return DataPin(local_root=Path(env[LOCAL_ENV]), **common)
    if env.get(DATASET_ENV):
        repo, sep, revision = env[DATASET_ENV].partition("@")
        if not sep or not repo or not revision:
            raise ValueError(f"{DATASET_ENV} must be <owner>/<name>@<revision>")
    else:
        repo, revision = host_cost_pin()
    return DataPin(repo=repo, revision=revision, **common)


def measured_overhead(
    *,
    backend: str,
    backend_version: str,
    endpoint: str,
    system: str,
    host_cpu: str,
    model: str,
    env: Mapping[str, str] | None = None,
) -> HostOverhead | None:
    """The measured overhead of this deployment on host CPU ``host_cpu``, or ``None`` when it has not been measured."""
    pin = _pin(os.environ if env is None else env)
    args = (pin, backend, backend_version, endpoint, system, host_cpu, model)
    # Only a full commit SHA is immutable; a local directory or a branch can be republished in place, so it is read
    # again. Callers get their own copy either way.
    found = _cached(*args) if pin.local_root is None and _FULL_SHA.fullmatch(pin.revision or "") else _read(*args)
    return copy.deepcopy(found)


@lru_cache(maxsize=64)
def _cached(
    pin: DataPin, backend: str, backend_version: str, endpoint: str, system: str, host_cpu: str, model: str
) -> HostOverhead | None:
    return _read(pin, backend, backend_version, endpoint, system, host_cpu, model)


def _read(
    pin: DataPin, backend: str, backend_version: str, endpoint: str, system: str, host_cpu: str, model: str
) -> HostOverhead | None:
    config = {
        "system": system,
        "backend": backend,
        "backend_version": backend_version,
        "model": model,
        "endpoint": endpoint,
        "host_cpu": host_cpu,
    }
    found = ensure_data(config, pin).config.get("host_cost")
    return None if found is None else HostOverhead.from_dict(found)
