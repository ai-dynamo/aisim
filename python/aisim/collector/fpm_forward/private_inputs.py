# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private worker delivery and redaction at retained diagnostic boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from aisim.fpm_contract import FPM_RESOLVED_CONFIG_GLOB

from .config import archive_generator_overrides, resolve_archived_generator_overrides

RECEIPT = ".private-delivery.json"


def private_values(overrides):
    archived = archive_generator_overrides(overrides)
    private = {item["name"] for item in archived["K8sConfig"]["extra_env"] if "value_from_env" in item}
    literals = {
        item["name"]: item["value"] for item in overrides.get("K8sConfig", {}).get("extra_env") or [] if "value" in item
    }
    return tuple(value for name in private if (value := literals.get(name, os.environ.get(name))))


def redact(text, values):
    for value in sorted(set(values), key=len, reverse=True):
        for representation in (
            json.dumps(value)[1:-1],
            json.dumps(value, ensure_ascii=False)[1:-1],
            repr(value)[1:-1],
            value,
        ):
            text = text.replace(representation, "<redacted>")
    return text


def redact_diagnostics(directory, values, *, cell=None, update=None):
    """Sanitize retained text after exact transport verification, before archival."""
    if not values:
        return
    for name in ("raw", "logs"):
        root = Path(directory) / name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                # Native measurements and source-bound observation envelopes are
                # structured evidence, not diagnostic config dumps. Replacing a
                # short environment value in their numeric JSON would corrupt it.
                if name == "raw" and path.suffix == ".json" and "config" not in path.name:
                    continue
                try:
                    original = path.read_text()
                except UnicodeDecodeError:
                    if update is None:
                        path.unlink()
                    else:
                        update(path, None)
                    continue
                sanitized = redact(original, values)
                if path.suffix == ".json":
                    try:
                        payload = json.loads(original)
                    except ValueError:
                        pass  # Preserve malformed diagnostic text, with secrets removed.
                    else:
                        markers = {}
                        if name == "raw" and path.match(FPM_RESOLVED_CONFIG_GLOB) and cell is not None:
                            markers = {
                                tuple(key.split(".")): val for key, val in cell.backend_policy.expected_markers.items()
                            }

                        def scrub(value, location=()):
                            if isinstance(value, str):
                                return value if markers.get(location) == value else redact(value, values)
                            if isinstance(value, list):
                                return [scrub(item, (*location, index)) for index, item in enumerate(value)]
                            if isinstance(value, dict):
                                return {key: scrub(item, (*location, key)) for key, item in value.items()}
                            return value

                        redacted = scrub(payload)
                        if redacted == payload:
                            continue
                        sanitized = json.dumps(redacted, ensure_ascii=False) + "\n"
                if sanitized != original:
                    if update is None:
                        path.write_text(sanitized)
                    else:
                        update(path, sanitized.encode())


def reclaim_delivery(cell_dir, *, update=None):
    """Call only after verified teardown of this cell's owned Slurm steps."""
    cell_dir = Path(cell_dir).resolve()
    receipt = cell_dir / RECEIPT
    if not receipt.exists():
        return
    saved = json.loads(receipt.read_text())
    path = Path(saved["path"])
    prefix = "aisimulate-" + hashlib.sha256(str(cell_dir).encode()).hexdigest()[:20] + "-"
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not re.fullmatch(re.escape(prefix) + r"[a-z0-9_]+", path.name)
        or path.is_relative_to(Path(saved["campaign"]).resolve())
    ):
        raise ValueError("invalid private Slurm delivery receipt")
    if path.exists():
        if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
            raise ValueError("private Slurm delivery requires caller-owned mode 0700")
        shutil.rmtree(path)
    if update is None:
        receipt.unlink()
    else:
        update(receipt, None)


def discard_diagnostics(directory, *, keep=(), update=None):
    """Remove logs and config dumps that could not be sanitized."""
    for name in ("raw", "logs"):
        root = Path(directory) / name
        for path in root.rglob("*"):
            if path in keep or not (path.is_symlink() or path.is_file()):
                continue
            if name == "raw" and path.suffix == ".json" and "config" not in path.name:
                continue
            if update is None:
                path.unlink()
            else:
                update(path, None)


def sanitize_recovered_diagnostics(directory, overrides, *, cell=None, environment=None, update=None):
    """Retain diagnostics only when all redaction values match their saved digests."""
    try:
        resolved = resolve_archived_generator_overrides(overrides, environment=environment)
    except ValueError:
        # Recovery of native measurements does not require credentials. Without
        # verified values, retain only the verified, allowlisted backend markers.
        from .database import _observed_backend_markers
        from .runner import _atomic_json

        retained = set()
        try:
            projections = _observed_backend_markers(cell, Path(directory)) if cell is not None else {}
            for path, projection in projections.items():
                if update is None:
                    _atomic_json(path, projection)
                else:
                    update(path, (json.dumps(projection, indent=2, sort_keys=True) + "\n").encode())
                retained.add(path)
        finally:
            discard_diagnostics(directory, keep=retained, update=update)
    else:
        redact_diagnostics(directory, private_values(resolved), cell=cell, update=update)


class SlurmDelivery:
    """No GC cleanup: mounted files must survive until verified teardown."""

    def __init__(self, campaign, cell_dir):
        campaign, self.cell_dir = Path(campaign).resolve(), Path(cell_dir).resolve()
        configured_root = os.environ.get("AISIMULATE_PRIVATE_DELIVERY_ROOT")
        if not configured_root:
            raise ValueError(
                "Slurm requires AISIMULATE_PRIVATE_DELIVERY_ROOT on shared storage readable by compute nodes"
            )
        root = Path(configured_root).expanduser().resolve()
        if root.is_relative_to(campaign):
            raise ValueError("private Slurm delivery root must be outside the campaign tree")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
            raise ValueError("private Slurm delivery root requires caller-owned mode 0700")
        if (self.cell_dir / RECEIPT).exists():
            raise ValueError("reclaim abandoned private delivery after verified Slurm teardown before rendering")
        prefix = "aisimulate-" + hashlib.sha256(str(self.cell_dir).encode()).hexdigest()[:20] + "-"
        self.name = tempfile.mkdtemp(prefix=prefix, dir=root)
        from .runner import _atomic_json

        try:
            _atomic_json(self.cell_dir / RECEIPT, {"path": self.name, "campaign": str(campaign)})
        except Exception:
            Path(self.name).rmdir()
            raise

    def cleanup(self):
        reclaim_delivery(self.cell_dir)


def delivery_directory(campaign, cell_dir, executor):
    return (
        SlurmDelivery(campaign, cell_dir)
        if executor == "slurm"
        else tempfile.TemporaryDirectory(prefix="aisimulate-delivery-")
    )
