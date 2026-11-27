# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
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


@pytest.mark.parametrize(
    ("baseline_power", "shutdown", "shifted_reads"),
    [
        (200.5, True, ()),
        (None, True, ()),
        (200.5, False, ()),
        (200.5, True, ("energy_start", "energy_end")),
        (200.5, True, ("energy_end",)),
    ],
)
def test_observed_campaign_publishes_both_points(tmp_path, monkeypatch, baseline_power, shutdown, shifted_reads):
    plan = _plan()
    arms = []
    receipt = {
        "gpu_uuids": ["GPU-0", "GPU-1"],
        "mean_w_per_gpu": 200.5,
        "serving_frames_s": 100,
        "arm_finished_s": 5.0,
    }
    for cell in plan.cells:
        with monkeypatch.context() as patch:
            out = tmp_path / cell.workload_kind
            raw = out / "raw" / "node-0"
            ctx = _cell_context(plan, cell, cell.cell_id, "1")
            patch.setattr(simulated, "context", lambda _: ctx)
            simulated.run(
                raw, patch, phase=cell.workload_kind, read_cost=0.025, count=16 if shutdown else 6, shutdown=shutdown
            )
            arms.append((cell, cell.cell_id, out, receipt | {"mean_w_per_gpu": baseline_power}, receipt))
            if shutdown:
                (leader_path,) = raw.glob("trtllm-*/rank-0.json")
                leader = json.loads(leader_path.read_text())
                for read in shifted_reads:
                    for bound in ("before_s", "after_s"):
                        leader[read][bound] += 30
                if shifted_reads:
                    leader_path.write_text(json.dumps(leader))
                reason, clock = _validate_clock(raw, {"GPU-0", "GPU-1"}, arm_finished_s=receipt["arm_finished_s"])
                if shifted_reads:
                    assert reason == "ambiguous_energy_boundaries"
                    assert clock["boundary_displacement_s"] == pytest.approx(30.025)
                else:
                    assert reason is None
                    assert clock["boundary_displacement_s"] == pytest.approx(0.025)
                    assert clock["common_energy_window_s"] == pytest.approx(2.075)
            else:
                assert not any(raw.glob("trtllm-*/published.json"))

    parquet, rows, summary = publish_campaign(plan, arms, tmp_path / "systems")
    assert parquet.is_file()
    assert {row["workload_kind"] for row in rows} == {"prefill", "decode"}
    assert all(row["latency_ms"] == pytest.approx(210) for row in rows)
    expected = pytest.approx(620 / 2.1) if shutdown and baseline_power is not None and not shifted_reads else None
    if shifted_reads:
        assert all(entry["power_unavailable_reason"] == "ambiguous_energy_boundaries" for entry in summary.values())
    assert all(row["power_w"] == expected for row in rows)
    assert all(entry["serving_rate_change_pct"] == 0 for entry in summary.values())
    # Use the compiled profile query on the same schema-8 file just published.
    query = query_published_profile(plan, parquet, rows)
    assert all(point["power_w"] == expected for point in query.values())


def test_shared_node_allocation_is_allowed(monkeypatch):
    from types import SimpleNamespace

    from collector.fpm_forward import trtllm_runner

    monkeypatch.setenv("SLURM_JOB_ID", "646")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "pilot-node")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")
    monkeypatch.setattr(trtllm_runner.socket, "gethostname", lambda: "pilot-node")
    monkeypatch.setattr(trtllm_runner.subprocess, "check_output", lambda *a, **kw: "pilot-node\n")
    monkeypatch.setattr(
        trtllm_runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="646\n647\n")
    )
    assert trtllm_runner._require_allocation() == {"job_id": "646", "node": "pilot-node", "other_jobs_at_start": 1}
