#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Set or check the unified FPM dataset pin in every consumer (AISIM-2338).

The single source is ``scripts/fpm_accuracy/dataset_pin.json``. ``--revision SHA`` rewrites it and the
``repo_id``/``revision`` of every pin file that already uses ``data/fpm/`` paths: the packaged FPM profiles
(``systems/profiles/*/*hf_dataset.json``) and the collector recipes. A pin file that still points at the retired
repository because its files do not exist at the migrated revision (``systems/dsv41_fpm_hf.json``) is left alone.
``--check`` fails when any consumer disagrees with the pin file; ``--require-set`` also fails while the pin is the
all-zero placeholder. A README that links its profile's pinned revision (``REVISION_READMES``) moves with the pin.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIN_FILE = ROOT / "scripts/fpm_accuracy/dataset_pin.json"
SYSTEMS = ROOT / "python/aisim/src/aisim_core/systems"
RECIPES = ROOT / "python/aisim/collector/fpm_forward/recipes"
PLACEHOLDER = "0" * 40
# README -> the profile manifest whose pinned revision it links (``/tree/<revision>/``).
REVISION_READMES = {"minimax_m27_fpm/README.md": "minimax_m27_fpm/hf_dataset.json"}


def consumers() -> list[Path]:
    return sorted([*SYSTEMS.glob("profiles/*/*hf_dataset.json"), *RECIPES.glob("*.json")])


def revision_readmes() -> dict[Path, Path]:
    return {
        SYSTEMS / "profiles" / readme: SYSTEMS / "profiles" / manifest for readme, manifest in REVISION_READMES.items()
    }


def _paths(document: dict) -> list[str]:
    if "profiles" in document:
        return [item["path"] for entry in document["profiles"].values() for item in entry["files"]]
    return [document["filename"]]


def is_unified(path: Path) -> bool:
    """True when every pinned path is already physical (``data/fpm/...``)."""
    return all(p.startswith("data/fpm/") for p in _paths(json.loads(path.read_text(encoding="utf-8"))))


def _with_pin(text: str, repo_id: str, revision: str) -> str:
    """``text`` with the first ``repo_id`` and ``revision`` values replaced, whatever JSON whitespace it uses."""
    new = re.sub(r'("repo_id"\s*:\s*")[^"]*(")', lambda m: m[1] + repo_id + m[2], text, count=1)
    return re.sub(r'("revision"\s*:\s*")[^"]*(")', lambda m: m[1] + revision + m[2], new, count=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="full 40-hex commit SHA of the unified dataset")
    parser.add_argument("--repo-id", help="dataset repository (default: the one in the pin file)")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--require-set", action="store_true", help="with --check: fail on the placeholder")
    args = parser.parse_args()
    pin = json.loads(PIN_FILE.read_text(encoding="utf-8"))
    if args.revision or args.repo_id:
        revision = args.revision or pin["revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            parser.error("--revision must be a full lowercase 40-hex commit SHA")
        pin["repo_id"] = args.repo_id or pin["repo_id"]
        pin["revision"] = revision
        if revision != PLACEHOLDER:
            pin.pop("note", None)
        # Validate every README link before the first write.
        moved = {}
        for readme, manifest in revision_readmes().items():
            current = json.loads(manifest.read_text(encoding="utf-8"))["revision"]
            text = readme.read_text(encoding="utf-8")
            moved[readme] = text.replace(f"/tree/{current}/", f"/tree/{revision}/")
            if f"/tree/{revision}/" not in moved[readme]:
                parser.error(f"{readme}: does not link the pinned revision {current}")
        rewritten = {}
        for path in consumers():
            if is_unified(path):
                text = path.read_text(encoding="utf-8")
                new = _with_pin(text, pin["repo_id"], pin["revision"])
                try:
                    document = json.loads(new)
                    updated = (document["repo_id"], document["revision"]) == (pin["repo_id"], pin["revision"])
                except (ValueError, KeyError):
                    updated = False
                if not updated:
                    parser.error(f"{path}: cannot update the pin fields in place")
                rewritten[path] = (text, new)
        PIN_FILE.write_text(json.dumps(pin, indent=2) + "\n", encoding="utf-8")
        changed = []
        for path, (text, new) in rewritten.items():
            if new != text:
                path.write_text(new, encoding="utf-8")
                changed.append(path)
        for readme, text in moved.items():
            if text != readme.read_text(encoding="utf-8"):
                readme.write_text(text, encoding="utf-8")
                changed.append(readme)
        print(f"pin {pin['repo_id']}@{pin['revision']}: updated {len(changed)} consumer file(s)")
    problems = []
    for path in consumers():
        document = json.loads(path.read_text(encoding="utf-8"))
        if not is_unified(path):
            problems.append(f"{path.relative_to(ROOT)}: still uses legacy data/ paths")
        elif (document["repo_id"], document["revision"]) != (pin["repo_id"], pin["revision"]):
            problems.append(f"{path.relative_to(ROOT)}: pin differs from {PIN_FILE.name}")
    for readme in revision_readmes():
        if f"/tree/{pin['revision']}/" not in readme.read_text(encoding="utf-8"):
            problems.append(f"{readme.relative_to(ROOT)}: does not link the pinned revision")
    if args.require_set and pin["revision"] == PLACEHOLDER:
        problems.append("the dataset pin is still the placeholder")
    for problem in problems:
        print(problem, file=sys.stderr)
    # An update that leaves any consumer inconsistent has failed, as a check does.
    return 1 if problems and (args.check or args.require_set or args.revision or args.repo_id) else 0


if __name__ == "__main__":
    sys.exit(main())
