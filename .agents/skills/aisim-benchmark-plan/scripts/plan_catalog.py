# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query the coverage catalog for a benchmark plan; print JSON, invent nothing.

Reads only ``<dataset-root>/catalog/`` through ``aisim_core.data.catalog.Catalog``.
The GPU budget is arithmetic on the numbers you pass in and is labelled
``estimated``; prior GPU time comes from recorded cells and stays null when none
is recorded.
"""

import argparse
import json
import math
import sys
from pathlib import Path

from aisim_core.data.catalog import Catalog


def _row(r):
    return {
        "layer": r.layer,
        "model": r.model,
        "hardware": r.hardware,
        "framework": r.framework,
        "status": r.status,
        "versions": r.versions,
        "multi_version": "," in r.versions,  # the tier below belongs to one record, not every listed version
        "n_configs": r.n_configs,
        "acc_mape_pct": r.acc_mape_pct,
        "acc_tier": r.acc_tier,
        "acc_prediction_fingerprint": r.acc_prediction_fingerprint,
        "gpu_seconds": r.gpu_seconds,
        "stale_reason": r.stale_reason,
    }


def build_plan(args) -> dict:
    catalog = Catalog(args.dataset_root)
    rows = catalog.rows(hardware=args.hardware, model=args.model, framework=args.framework, layer=args.layer)
    # The op layer is model-independent (model is null), so keep it when a model filter is set.
    if args.model is not None and args.layer in (None, "op"):
        rows += [r for r in catalog.rows(hardware=args.hardware, framework=args.framework, layer="op") if r not in rows]
    fingerprints = set(args.prediction_fingerprint)
    reusable = lambda r: r.acc_mape_pct is not None and r.acc_prediction_fingerprint in fingerprints
    groups = {
        "reuse": [r for r in rows if r.status == "have" and reusable(r)],
        "evaluate_only": [r for r in rows if r.status == "have" and not reusable(r)],
        "recollect_or_reevaluate": [r for r in rows if r.status == "stale"],
        "collect": [r for r in rows if r.status == "missing"],
    }
    recorded = [r.gpu_seconds for r in rows if r.gpu_seconds is not None]
    plan = {
        "hardware": args.hardware,
        "known_hardware": catalog.hardware(),
        "cells": {name: [_row(r) for r in group] for name, group in groups.items()},
        "prior_recorded_gpu_seconds": sum(recorded) if recorded else None,  # a recorded zero stays zero
    }
    if args.model is not None:
        wanted = [args.layer] if args.layer in ("fpm", "e2e") else ["fpm", "e2e"] if args.layer is None else []
        present = {r.layer for r in rows if r.model == args.model}
        # The catalog lists only models that have appeared; absence is "unknown", not "covered".
        plan["not_in_catalog"] = [
            {"layer": layer, "model": args.model, "hardware": args.hardware} for layer in wanted if layer not in present
        ]
    if args.gpus is not None and args.wall_hours is not None:
        base = args.gpus * args.wall_hours * 3600.0
        plan["budget"] = {
            "authority": "estimated",
            "gpu_seconds": base * (1 + args.retry_margin),
            "gpu_hours": base * (1 + args.retry_margin) / 3600.0,
            "inputs": {"gpus": args.gpus, "wall_hours": args.wall_hours, "retry_margin": args.retry_margin},
            "note": "allocated time, queue wait excluded; replace with a measured gpu_time after the run",
        }
    return plan


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True, type=Path, help="directory that contains catalog/")
    p.add_argument("--hardware", required=True)
    p.add_argument("--model")
    p.add_argument("--framework")
    p.add_argument("--layer", choices=["op", "fpm", "e2e"])
    p.add_argument(
        "--prediction-fingerprint",
        action="append",
        default=[],
        help="current PredictionIdentity fingerprint(s); without a match accuracy must be re-evaluated",
    )
    p.add_argument("--gpus", type=int, help="GPUs the planned run allocates")
    p.add_argument("--wall-hours", type=float, help="planned wall hours of the whole allocation")
    p.add_argument("--retry-margin", type=float, default=0.25, help="fraction added for failed/retried attempts")
    args = p.parse_args(argv)
    if (args.gpus is None) != (args.wall_hours is None):
        p.error("--gpus and --wall-hours must be given together")
    if args.gpus is not None and args.gpus <= 0:
        p.error("--gpus must be positive")
    if args.wall_hours is not None and not (math.isfinite(args.wall_hours) and args.wall_hours > 0):
        p.error("--wall-hours must be a positive finite number")
    if not (math.isfinite(args.retry_margin) and args.retry_margin >= 0):
        p.error("--retry-margin must be a non-negative finite number")
    try:
        plan = build_plan(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}; fetch the catalog first (aisim_core.data.fetch)", file=sys.stderr)
        return 2
    json.dump(plan, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
