# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset defaults and the ``aisim init`` configuration file.

The init choice lives in ``config.json`` in the cache root, the directory above the ``data/`` cache
(``~/.cache/aisim/config.json``). ``AISIM_DATA_CACHE`` names the data cache itself, so the file
sits inside that directory (``$AISIM_DATA_CACHE/config.json``). No file means on-demand.

Precedence for every setting: command-line flag, then environment variable, then this file, then the
shipped default.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aisim_core.data.repos import PUBLIC_REPO

DEFAULT_REPO = PUBLIC_REPO
#: Revision used when the shipped pin file is absent. Moves with the dataset's ``main`` branch.
FALLBACK_REVISION = "main"
PIN_FILE = Path(__file__).with_name("op_latency_pin.json")
CONFIG_NAME = "config.json"
MODES = ("all", "hardware", "on-demand")
DEFAULT_MODE = "on-demand"


class ConfigError(ValueError):
    """The saved init config is unreadable."""


def shipped_pin() -> tuple[str, str]:
    """``(repo, revision)`` shipped with the package; ``main`` only when the pin file is absent.

    An existing pin that is unreadable or lacks ``revision`` raises :class:`ConfigError`.
    """
    try:
        text = PIN_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return DEFAULT_REPO, FALLBACK_REVISION
    except OSError as exc:
        raise ConfigError(f"{PIN_FILE}: unreadable shipped pin: {exc}") from exc
    try:
        data = json.loads(text)
        return str(data.get("repo") or DEFAULT_REPO), str(data["revision"])
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ConfigError(f"{PIN_FILE}: malformed shipped pin: {type(exc).__name__}: {exc}") from exc


def config_path(env: Mapping[str, str] | None = None, cache_dir: str | Path | None = None) -> Path:
    """Where the init config lives; an explicit ``cache_dir`` keeps it inside that directory."""
    env = os.environ if env is None else env
    if cache_dir:
        return Path(cache_dir) / CONFIG_NAME
    if env.get("AISIM_DATA_CACHE"):
        return Path(env["AISIM_DATA_CACHE"]) / CONFIG_NAME
    base = Path(env["XDG_CACHE_HOME"]) if env.get("XDG_CACHE_HOME") else Path.home() / ".cache"
    return base / "aisim" / CONFIG_NAME


@dataclass(frozen=True)
class InitConfig:
    mode: str = DEFAULT_MODE
    hardware: str | None = None
    repo: str | None = None
    revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in {"version": 1, **self.__dict__}.items() if v is not None}


def load_config(path: Path | None = None) -> InitConfig | None:
    """The saved config, or ``None`` when there is none (on-demand)."""
    path = path or config_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{path}: unreadable init config: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("mode") not in MODES:
        raise ConfigError(f"{path}: expected an object with mode one of {', '.join(MODES)}")
    return InitConfig(
        mode=raw["mode"], hardware=raw.get("hardware"), repo=raw.get("repo"), revision=raw.get("revision")
    )


def save_config(config: InitConfig, path: Path | None = None) -> Path:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".config.", suffix=".part")
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        out.write(json.dumps(config.to_dict(), indent=2) + "\n")
    os.replace(tmp, path)
    return path


def resolve_source(
    repo: str | None = None,
    revision: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
    config: InitConfig | None = None,
) -> tuple[str, str]:
    """``(repo, revision)``: flag, then ``AISIM_DATASET=<repo>@<revision>``, then config, then the shipped pin."""
    env = os.environ if env is None else env
    shipped_repo, shipped_revision = shipped_pin()
    env_repo = env_revision = None
    spec = env.get("AISIM_DATASET")
    if spec:
        env_repo, sep, env_revision = spec.partition("@")
        if not sep or not env_repo or not env_revision:
            raise ValueError("AISIM_DATASET must be <owner>/<name>@<revision>")
    repo = repo or env_repo or (config.repo if config else None) or shipped_repo
    # A revision only means something for its repository: keep the pair together unless overridden.
    default_revision = shipped_revision if repo == shipped_repo else FALLBACK_REVISION
    revision = (
        revision
        or (env_revision if repo == env_repo else None)
        or (config.revision if config and (config.repo or shipped_repo) == repo else None)
        or None
    )
    return repo, revision or default_revision
