# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Look up a measured serving-path host cost and shape it as ``engine.tokenization`` and ``engine.extra_cpu_overhead``.

A result is exact for (framework, endpoint, tokenizer, host CPU). Nothing is extrapolated across host CPUs,
tokenizers or endpoints. Framework versions are recollected only every one or two releases, so an uncollected
version reuses the closest collected one (at or below it, else the nearest above) and the result says so.

Two entry purposes pair up (see :mod:`aisim_core.data.host_cost`): the tokenizer entry gives the queued
``engine.tokenization``; the unqueued entry gives ``engine.extra_cpu_overhead``. The unqueued entry stores the
measured cost outside the engine, so its terms are the total less the tokenizer terms it is paired with here, and the
two together reproduce the measured total. A deployment with no tokenizer entry has no
default at all.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from packaging.version import InvalidVersion, Version

from aisim_core.data import Entry as CatalogEntry
from aisim_core.data import Manifest
from aisim_core.data.host_cost import (
    CATEGORY,
    MANIFEST_NAME,
    PURPOSE,
    UNQUEUED_PURPOSE,
    read_fit,
    read_unqueued_fit,
)
from aisim_core.host_cost.fit import Fit

# Served models whose tokenizer is a measured family but whose repository is not the one the family was measured
# from. A model that a tokenizer entry was measured from needs no row here.
TOKENIZER_BY_MODEL: dict[str, str] = {
    "nvidia/GLM-5.3-NVFP4": "glm",
}

_T = TypeVar("_T")


@dataclass(frozen=True)
class Entry:
    framework: str
    version: str
    tokenizer: str
    cpu_model: str
    system: str
    tokenizer_workers: int | None
    fit: Fit
    endpoint: str = "chat"


@dataclass(frozen=True)
class LookupResult:
    entry: Entry
    requested_version: str
    reused_version: str | None  # the collected version that stood in, or None for an exact match

    def engine_tokenization(self) -> dict[str, Any]:
        """The ``engine.tokenization`` block for this entry.

        ``base_ms`` is clamped at zero (a fit can have a small negative intercept, which replay rejects). ``workers``
        is the tokenizer worker count the entry was measured with, or 1 when the framework's count is unverified.
        """
        return {
            "workers": self.entry.tokenizer_workers or 1,
            "base_ms": max(self.entry.fit.base_ms, 0.0),
            "per_input_token_us": self.entry.fit.per_input_token_us,
        }


def _version(text: str) -> Version | None:
    try:
        return Version(text)
    except InvalidVersion:
        return None


def _closest(candidates: Sequence[_T], version: str, version_of: Callable[[_T], str], allow_reuse: bool):
    """``(candidate, reused version or None)`` for ``version``, or ``None``."""
    for candidate in candidates:
        if version_of(candidate) == version:
            return candidate, None
    wanted = _version(version)
    if not allow_reuse or wanted is None:
        return None
    parsed = [(v, c) for c in candidates if (v := _version(version_of(c))) is not None]
    if not parsed:
        return None
    below = [(v, c) for v, c in parsed if v <= wanted]
    # Closest at or below; if every collected version is newer, the nearest one above.
    chosen = max(below, key=lambda item: item[0]) if below else min(parsed, key=lambda item: item[0])
    return chosen[1], version_of(chosen[1])


def _entry(manifest: Manifest) -> Entry:
    settings = manifest.identity.effective_settings
    workers = settings.get("tokenizer_workers")
    return Entry(
        framework=manifest.identity.framework,
        version=manifest.identity.framework_version,
        tokenizer=str(settings["tokenizer_key"]),
        cpu_model=str(settings["cpu_model"]),
        system=manifest.identity.hardware,
        tokenizer_workers=int(workers) if isinstance(workers, int) else None,
        fit=read_fit(manifest),
        endpoint=str(settings.get("endpoint", "chat")),
    )


def load_entries(root: str | Path) -> list[Entry]:
    """Every tokenizer (``host-cost-profile``) entry under a dataset root."""
    base = Path(root) / "data" / CATEGORY
    entries = []
    for path in sorted(base.rglob(MANIFEST_NAME)) if base.is_dir() else []:
        manifest = Manifest.model_validate_json(path.read_bytes())
        if manifest.purpose == PURPOSE:
            entries.append(_entry(manifest))
    return entries


def lookup(
    entries: list[Entry],
    *,
    framework: str,
    version: str,
    tokenizer: str,
    cpu_model: str,
    endpoint: str | None = None,
    allow_version_reuse: bool = True,
) -> LookupResult | None:
    """The tokenizer entry for this deployment, or ``None``. Host CPU, tokenizer and endpoint must match exactly."""
    candidates = [
        e
        for e in entries
        if (e.framework, e.tokenizer, e.cpu_model) == (framework, tokenizer, cpu_model)
        and (endpoint is None or e.endpoint == endpoint)
    ]
    found = _closest(candidates, version, lambda e: e.version, allow_version_reuse)
    return None if found is None else LookupResult(found[0], version, found[1])


# --------------------------------------------------------------------------- catalog selection for replay


@dataclass(frozen=True)
class Choice:
    """One catalog entry chosen for a deployment, and the collected version that stood in for the requested one."""

    entry: CatalogEntry
    reused_version: str | None


def _setting(entry: CatalogEntry, name: str) -> str | None:
    value = entry.identity.effective_settings.get(name)
    return None if value is None else str(value)


def tokenizer_key(model: str, catalog: Sequence[CatalogEntry]) -> str | None:
    """The measured tokenizer family of ``model``: :data:`TOKENIZER_BY_MODEL`, else the entry measured from it."""
    if model in TOKENIZER_BY_MODEL:
        return TOKENIZER_BY_MODEL[model]
    keys = {_setting(e, "tokenizer_key") for e in catalog if e.purpose == PURPOSE and e.identity.model == model}
    keys.discard(None)
    if len(keys) > 1:
        raise ValueError(f"host-cost entries measured from {model} name several tokenizers: {sorted(keys)}")
    return next(iter(keys), None)


def choose(
    catalog: Sequence[CatalogEntry],
    *,
    framework: str,
    version: str,
    endpoint: str,
    system: str,
    tokenizer: str,
) -> tuple[Choice | None, Choice | None]:
    """The tokenizer entry and the unqueued entry for one deployment, from catalog identities alone.

    The host CPU is the one the tokenizer entries of ``system`` were measured on; the unqueued entry must name the
    same CPU. ``(None, None)`` when no tokenizer entry matches: an unmeasured deployment has no default, and an
    unqueued entry without a tokenizer slope to split against is not used.
    """

    def matches(entry: CatalogEntry, purpose: str) -> bool:
        return (
            entry.purpose == purpose
            and entry.identity.framework == framework
            and entry.identity.hardware == system
            and _setting(entry, "tokenizer_key") == tokenizer
            and _setting(entry, "endpoint") == endpoint
        )

    tokenizers = [e for e in catalog if matches(e, PURPOSE)]
    cpus = {_setting(e, "cpu_model") for e in tokenizers}
    if len(cpus) > 1:
        raise ValueError(f"{system} has host-cost entries from several host CPUs {sorted(map(str, cpus))}")
    found = _closest(tokenizers, version, lambda e: e.identity.framework_version, True)
    if found is None:
        return None, None
    cpu = _setting(found[0], "cpu_model")
    unqueued = [e for e in catalog if matches(e, UNQUEUED_PURPOSE) and _setting(e, "cpu_model") == cpu]
    extra = _closest(unqueued, version, lambda e: e.identity.framework_version, True)
    return Choice(*found), None if extra is None else Choice(*extra)


@dataclass(frozen=True)
class HostOverhead:
    """The ``engine.tokenization`` and ``engine.extra_cpu_overhead`` blocks for one deployment, with their sources."""

    tokenization: dict[str, Any]
    extra_cpu_overhead: dict[str, Any] | None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenization": dict(self.tokenization),
            "extra_cpu_overhead": None if self.extra_cpu_overhead is None else dict(self.extra_cpu_overhead),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HostOverhead:
        return cls(data["tokenization"], data.get("extra_cpu_overhead"), data.get("provenance") or {})


def overhead(
    requested_version: str,
    tokenizer: tuple[Choice, Manifest],
    unqueued: tuple[Choice, Manifest] | None,
) -> HostOverhead:
    """Shape a chosen tokenizer entry and optional unqueued entry as replay config blocks."""
    tokenizer_choice, tokenizer_manifest = tokenizer
    tokenization = LookupResult(
        _entry(tokenizer_manifest), requested_version, tokenizer_choice.reused_version
    ).engine_tokenization()
    provenance: dict[str, Any] = {
        "requested_version": requested_version,
        "tokenization": {
            "configuration_id": tokenizer_manifest.configuration_id,
            "version": tokenizer_manifest.identity.framework_version,
            "reused_version": tokenizer_choice.reused_version,
        },
    }
    extra = None
    if unqueued is not None:
        unqueued_choice, unqueued_manifest = unqueued
        fit = read_unqueued_fit(unqueued_manifest)
        # Split the measured total against the tokenizer stage replay actually queues (fixed and per-token terms), so
        # the two stages reproduce the measured total.
        extra = {
            "base_ms": max(fit.base_ms - tokenization["base_ms"], 0.0),
            "per_input_token_us": max(fit.total_per_input_token_us - tokenization["per_input_token_us"], 0.0),
        }
        provenance["extra_cpu_overhead"] = {
            "configuration_id": unqueued_manifest.configuration_id,
            "version": unqueued_manifest.identity.framework_version,
            "reused_version": unqueued_choice.reused_version,
            "total_per_input_token_us": fit.total_per_input_token_us,
        }
    return HostOverhead(tokenization, extra, provenance)
