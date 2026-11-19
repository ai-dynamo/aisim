# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check, and optionally verify and attest, a stored ``aisim-eval/v1`` record.

Without ``--predictor`` it loads the record (the file name must equal its content
hash) and reports ``gpu_time`` findings at archival and qualification level.

With ``--predictor FILE.py`` (a module exposing ``predict(inputs) -> float | None``
and ``fingerprint() -> str``, both built from the canonical ``best_available``
predictor) it recomputes the prediction from the recorded truth table with
``verify_record``. For a complete attempt, ``--attest-root DIR`` then writes the
superseding ``verified`` record with ``attest`` + ``write_record``. Partial and
failed attempts are reported and retained without attestation. Exit code 1 means
a failed check.
"""

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from aisim_core.data import _verify
from aisim_core.data.eval import attest, check_record_gpu_time, load_record, record_dir, verify_record, write_record
from aisim_core.data.gpu_time import check_gpu_time
from aisim_core.data.hygiene import checked_path


def _load_predictor(path: Path):
    spec = importlib.util.spec_from_file_location("benchmark_predictor", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load predictor module: {path}")
    module = importlib.util.module_from_spec(spec)
    missing = object()
    previous = sys.modules.get(spec.name, missing)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is missing:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
        raise
    return module.predict, module.fingerprint()


def _publish_artifact(source: Path, target: Path) -> None:
    """Publish complete bytes without replacing a competing writer's artifact."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, pending = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as incoming:
            shutil.copyfileobj(incoming, output)
        try:
            os.link(pending, target)
        except FileExistsError:
            pass  # the caller verifies the winner against the expected artifact hash
    finally:
        Path(pending).unlink(missing_ok=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("record", type=Path)
    p.add_argument(
        "--truth-root", type=Path, help="directory holding the truth artifacts (default: record's directory)"
    )
    p.add_argument("--predictor", type=Path)
    p.add_argument("--attest-root", type=Path, help="dataset root to write the verified record into")
    args = p.parse_args(argv)
    if args.attest_root and not args.predictor:
        p.error("--attest-root requires --predictor")

    record = load_record(args.record)
    if args.attest_root and record.tier == "qualified":
        p.error("--attest-root cannot downgrade a qualified record; use verification only")
    out = {
        "record_id": record.record_id,
        "tier": record.tier,
        "status": record.status,
        "n_scored": record.results.n_scored,
        "n_total": record.results.n_total,
        "mape_pct": record.results.mape_pct,
        "gpu_time_findings": {
            level: [f.__dict__ for f in check_gpu_time(record.gpu_time, level)]
            for level in ("archival", "qualification")
        },
        "tier_findings": [f.__dict__ for f in check_record_gpu_time(record)],
    }
    code = 0
    if args.predictor:
        predict, fingerprint = _load_predictor(args.predictor)
        report = verify_record(record, args.truth_root or args.record.parent, predict, fingerprint)
        out["verification"] = {"ok": report.ok, "problems": report.problems}
        if not report.ok:
            code = 1
        elif args.attest_root and record.status == "complete":
            verified = attest(record, report)
            history = [record]
            while history[-1].supersedes:
                history.append(load_record(args.record.parent / f"{history[-1].supersedes}.json"))
            # Preflight every input and collision before staging any record.
            destinations = {
                item.record_id: checked_path(args.attest_root, record_dir(Path(), item.truth.identity).as_posix())
                for item in [*history, verified]
            }
            copies = {}
            for item in [*history, verified]:
                destination = destinations[item.record_id]
                for artifact in [*item.truth.artifacts, *item.findings]:
                    source = _verify(args.truth_root or args.record.parent, artifact)
                    target = checked_path(destination, artifact.path)
                    if target in copies and copies[target][1] != artifact.sha256:
                        raise ValueError(f"conflicting artifact content: {target}")
                    if target.exists():
                        _verify(destination, artifact)
                    copies[target] = (source, artifact.sha256)
            for target, (source, _) in copies.items():
                if not target.exists():
                    _publish_artifact(source, target)
            for item in [*history, verified]:
                destination = destinations[item.record_id]
                for artifact in [*item.truth.artifacts, *item.findings]:
                    _verify(destination, artifact)
            for predecessor in reversed(history):
                write_record(args.attest_root, predecessor)
            out["verified_record_path"] = str(write_record(args.attest_root, verified))
            out["verified_record_id"] = verified.record_id
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
