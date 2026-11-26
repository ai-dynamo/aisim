# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from collector.fpm_forward.cli import _parser
from collector.fpm_forward.config import FPMCollectionOptions
from collector.fpm_forward.runtime_instrumentation import canonical_json, freeze_instrumentation, load_instrumentation
from collector.fpm_forward.trtllm_plan import BUNDLED_MANIFEST, IMAGE_SHA256, MODEL, POINTS, SYSTEM, build_plan
from collector.fpm_forward.trtllm_runner import (
    _cell_context,
    _validate_clock,
    publish_campaign,
    query_published_profile,
)

from . import test_fpm_trtllm_observer as simulated

pytestmark = pytest.mark.unit


def _plan():
    options = replace(
        FPMCollectionOptions.from_args(_parser().parse_args(["--fpm-max-gpus", "2"])),
        measure_power=True,
        benchmark_points_json=canonical_json({"schema_version": 1, **{key: [point] for key, point in POINTS.items()}}),
        benchmark_points_sha256="grid",
    )
    return build_plan(MODEL, SYSTEM, options, load_instrumentation(BUNDLED_MANIFEST), "revision", IMAGE_SHA256)


def test_frozen_trtllm_plan_binds_observer_and_points(tmp_path):
    plan = _plan()
    frozen = freeze_instrumentation(plan.runtime_instrumentation, tmp_path / "bundle")
    assert frozen.sha256 == plan.to_dict()["bundle_sha256"]
    assert [cell.workload_kind for cell in plan.cells] == ["prefill", "decode"]
    with pytest.raises(ValueError, match="explicit points"):
        build_plan(MODEL, SYSTEM, replace(plan.options, benchmark_points_json=None), frozen, "revision", IMAGE_SHA256)


@pytest.mark.parametrize("baseline_power", [200.5, None])
def test_observed_campaign_publishes_both_points(tmp_path, monkeypatch, baseline_power):
    plan = _plan()
    arms = []
    receipt = {"gpu_uuids": ["GPU-0", "GPU-1"], "mean_w_per_gpu": 200.5, "serving_frames_s": 100}
    for cell in plan.cells:
        with monkeypatch.context() as patch:
            out = tmp_path / cell.workload_kind
            raw = out / "raw" / "node-0"
            ctx = _cell_context(plan, cell, cell.cell_id, "1")
            patch.setattr(simulated, "context", lambda _: ctx)
            simulated.run(raw, patch, phase=cell.workload_kind)
            arms.append((cell, cell.cell_id, out, receipt | {"mean_w_per_gpu": baseline_power}, receipt))
            assert _validate_clock(raw, {"GPU-0", "GPU-1"})[0] is None

    parquet, rows, summary = publish_campaign(plan, arms, tmp_path / "systems")
    assert parquet.is_file()
    assert {row["workload_kind"] for row in rows} == {"prefill", "decode"}
    expected = pytest.approx(620 / 2.1) if baseline_power is not None else None
    assert all(row["power_w"] == expected for row in rows)
    assert all(entry["serving_rate_change_pct"] == 0 for entry in summary.values())
    # Use the compiled profile query on the same schema-8 file just published.
    query = query_published_profile(plan, parquet, rows)
    assert all(point["power_w"] == expected for point in query.values())
