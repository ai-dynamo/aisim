# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0

"""Test of the three modules of the SGLang self-benchmark producer core together. No SGLang, no GPU.

One test compares what the three pod modules must agree on. One test publishes a planned grid with AISim's own
Python path (``aggregate_cell``, then ``write_formal_database``) and reads the table back through AISim's
compiled reader.

Every time value that goes into an artifact is FAKE. It comes from ``_fake_reader_seconds`` or a ``FAKE_*``
constant. These values were typed in to exercise the arithmetic and were never measured. The artifacts say so
in ``producer``.

The collected layout is written by ``_pod`` of ``test_fpm_sglbench_artifact.py``, with the provenance file of
an SGLang cell (``"backend": "sglang"`` and a tested release). Publication then wants the same release in the
producer block of every rank artifact, where the launcher writes it; here ``_publish_with_fake_times`` writes
it in the launcher's place.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import shutil
import statistics
from collections.abc import Callable
from pathlib import Path

import pytest
from collector.fpm_forward import planner as fpm_planner
from collector.fpm_forward.config import FPMCollectionOptions
from collector.fpm_forward.database import aggregate_cell, write_formal_database
from collector.fpm_forward.planner import build_collection_plan
from collector.fpm_forward.runtime import sglbench_artifact as artifact
from collector.fpm_forward.runtime import sglbench_content as content
from collector.fpm_forward.runtime import sglbench_grid as grid

import aisim_core
from aisim.sdk import common, models
from aisim.sdk import config as sdk_config
from aisim.sdk.perf_database import PerfDatabase
from aisim_core.sdk.engine import EngineHandle, _fpm_spec_dict
from aisim_core.sdk.operations.fpm_forward import _CELL_MATCH_COLUMNS

from .test_fpm_sglbench_artifact import POD_KV_CACHE, POD_SGLANG_VERSION, _pod
from .test_fpm_sglbench_grid import H200_DECODE_GRAPH_ROWS, H200_PAGE_1, H200_PAGE_64, H200_PREFILL_GRAPH_ROWS, _ladder

pytestmark = pytest.mark.unit

# FAKE run inputs. A caller of ``ContentPlan`` must supply the token range (inside the served model's
# vocabulary) and the seed (the value of DYN_BENCH_CONTENT_SEED), as the docstring of sglbench_content.py says.
TOKEN_LOW, TOKEN_HIGH = 1000, 51_000
FAKE_CONTENT_SEED = "0"
FAKE_RUN_ID = "fake-run-2026.11.14"
FAKE_POINT_OVERHEAD_SECONDS = 0.25

# FAKE shape of the six decode steps around one base time: a short first step, a short last step, unsorted
# kept steps. Decode steps 3 to 5 are the list items 2, 3 and 4, and their upper median is the base itself.
FAKE_DECODE_STEP_FACTORS = (0.6, 1.5, 1.2, 0.9, 1.0, 0.3)


def _fake_reader_seconds(point: dict) -> float:
    """FAKE time of a point in the reader test: one millisecond per benchmark id.

    No two points of one phase share a value, so a value the reader hands back can only be the point's own row.
    """
    return 0.001 * point["benchmark_id"]


# ---------------------------------------------------------------------------------------------------------------
# What the three modules must agree on
# ---------------------------------------------------------------------------------------------------------------


def _with_types(value):
    """``value`` with the type of each of its parts beside it: 7 and 7.0, or a list and a tuple, then differ."""
    if isinstance(value, dict):
        return type(value), {_with_types(key): _with_types(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return type(value), tuple(_with_types(item) for item in value)
    if isinstance(value, set | frozenset):
        return type(value), frozenset(_with_types(item) for item in value)
    return type(value), value


def test_the_three_modules_agree_on_their_shared_constants():
    """Each module is a leaf and repeats what it needs; this is where the copies are compared."""
    assert grid.STAMP_EXPLICIT not in artifact.REGIME_STAMPS  # "explicit" is a free reason, not a regime stamp
    assert set(artifact.MEASUREMENT_POLICY) == {grid.PREFILL, grid.DECODE}
    # Of seven output tokens the first comes from an extend batch and the other six from six decode batches.
    assert artifact.DECODE_RAW_STEPS == grid.DECODE_OUTPUT_TOKENS - 1
    # The middle kept decode step reads exactly the point's context: the prompt has k - 3 tokens and the decode
    # step with zero-based index i reads the prompt plus i generated tokens.
    middle = artifact.DECODE_KEPT_STEP_INDICES[len(artifact.DECODE_KEPT_STEP_INDICES) // 2]
    assert middle == grid.DECODE_PROMPT_LEAD
    plan = content.ContentPlan(FAKE_CONTENT_SEED, TOKEN_LOW, TOKEN_HIGH)
    for context in (grid.MIN_DECODE_CONTEXT, 5, 67, 1024):
        assert len(plan.decode_prompt(0, 0, context)) + middle == context
    # The stages the driver uses are stages the request ids know.
    assert {"seed", "warm", "meas", "run", "free"} == set(content.STAGES)
    # Every public upper-case name that two of the modules define has one value, of one type at every depth.
    # The integer 7 and the float 7.0 compare equal and are not one value here.
    shared_names = {}
    for first, second in itertools.combinations((grid, content, artifact), 2):
        names = {name for name in vars(first) if name.isupper() and not name.startswith("_")} & set(vars(second))
        for name in names:
            one, other = getattr(first, name), getattr(second, name)
            assert one == other and _with_types(one) == _with_types(other), (first.__name__, second.__name__, name)
        shared_names[first, second] = names
    assert shared_names[grid, content] >= {"DECODE_PROMPT_LEAD", "MIN_DECODE_CONTEXT", "DECODE_OUTPUT_TOKENS"}
    assert shared_names[grid, artifact] >= {"PREFILL", "DECODE", "AGG", "MODES", "STAMP_DECODE_REAL_KV"}


# ---------------------------------------------------------------------------------------------------------------
# Publication: the planned grid as a table in AISim's compiled reader
# ---------------------------------------------------------------------------------------------------------------

# Where the FAKE table is published and how the reader is asked for it. The system file and the model's packaged
# config are the reader's own inputs; nothing was measured on that system or model here. The version is the data
# label of the plan (--fpm-backend-version): it names the table directory and no SGLang release.
READER_SYSTEM = "h200_sxm"
READER_BACKEND = "sglang"
READER_MODEL = "Qwen/Qwen3-0.6B"
FAKE_READER_VERSION = "fake-sglbench-reader-test"


def _default_h200_limits(h200: grid.Limits) -> grid.Limits:
    """The limits of one of the two engine probe runs, with the graph sizes SGLang captures by default."""
    # H200_PAGE_1 and H200_PAGE_64 hold the six limits in the recorder headers of two runs of the engine probe,
    # benchmarks/evidence/collector/sglbench-engine-probe-20261113 (headers of v0522-q06-p1 and v0522-q06-p64):
    # SGLang v0.5.22, Qwen3-0.6B, TP 1, page size 1 and 64. No result file records the GPU model; the README of
    # that folder gives it from the operator's record. The graph sizes were not read off an engine. They are
    # what generate_decode_cuda_graph_batch_sizes (graph padding not disabled, no speculative algorithm) and
    # generate_prefill_cuda_graph_batch_sizes return for a maximum of 256 and of 8192 in SGLang v0.5.22
    # (baa9befa381511cd4d56f83980951e5a45fffc6d) and v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90),
    # python/sglang/srt/arg_groups/cuda_graph_hook.py.
    return dataclasses.replace(
        h200,
        decode_graph_sizes=_ladder(H200_DECODE_GRAPH_ROWS),
        prefill_graph_sizes=_ladder(H200_PREFILL_GRAPH_ROWS),
    )


@dataclasses.dataclass
class _FakeTable:
    """What ``_publish_with_fake_times`` hands back: the planned grids, their FAKE times and the reader."""

    grids: dict[str, grid.Grid]  # phase -> the planned grid
    fake_ms: dict[tuple[str, int], float]  # (phase, benchmark id) -> the FAKE time in the published unit
    op_specs: dict[str, dict[str, str]]  # phase -> interpolation mode -> the reader's operator list, as JSON
    handle: EngineHandle

    def reader_ms(self, phase: str, mode: str, batch: int, per_request: int, cached_per_request: int = 0) -> float:
        """Ask the reader for one forward pass; it raises when it cannot answer.

        ``per_request`` is the new tokens per request (prefill) or the context per request (decode); the
        reader multiplies by the batch itself (crates/core/src/perfmodel/operators/fpm_forward.rs
        ``FpmForwardOp::query``).
        """
        answer = self.handle.evaluate_ops_json(
            self.op_specs[phase][mode],
            is_context=phase == "prefill",
            batch_size=batch,
            s=per_request,
            prefix=cached_per_request,
        )
        return answer[0][1]


def _publish_with_fake_times(
    root: Path, monkeypatch, limits: grid.Limits, fake_seconds_of: Callable[[dict], float]
) -> _FakeTable:
    """Plan the default grid for ``limits``, write it with FAKE times, publish it under ``root``, open the reader.

    The path is ``aggregate_cell`` and ``write_formal_database`` (collector/fpm_forward/database.py), then
    ``PerfDatabase`` and ``EngineHandle.for_database``, loaded as tests/unit/sdk/test_fpm_forward.py loads them.
    ``fake_seconds_of`` gives the FAKE time of a point from its artifact form.

    The plan and its cells come from ``build_collection_plan`` with backend ``sglang``, so the rows carry the
    identity a formal SGLang collection of this model gets on the host. The reader's operators are built on
    their own from the same model, and the caller's test fails when the two identities differ.
    """
    systems_root = root / "systems"
    systems_root.mkdir(parents=True)
    shutil.copy(Path(aisim_core.__file__).parent / "systems" / f"{READER_SYSTEM}.yaml", systems_root)
    # The plan of a formal SGLang collection of the model on one GPU: one TP 1 cell per phase. The runtime
    # version is the release of the FAKE pod, which publication compares with the provenance file of _pod.
    monkeypatch.setattr(fpm_planner, "_git_revision", lambda: "test-revision")  # plan without a git checkout
    plan = build_collection_plan(
        backend=READER_BACKEND,
        model_path=READER_MODEL,
        system=READER_SYSTEM,
        selected_ops={"attention_context", "attention_generation"},
        has_model_cases=False,
        options=FPMCollectionOptions.from_args(
            argparse.Namespace(backend=READER_BACKEND, fpm_max_gpus=1, fpm_prefill_cudagraph_policy="runtime")
        ),
        collector_config={"runtime_backend_version": POD_SGLANG_VERSION, "aic_database_version": FAKE_READER_VERSION},
    )
    assert [(cell.workload_kind, cell.parallel_strategy, cell.topology.tp) for cell in plan.cells] == [
        ("prefill", "single", 1),
        ("decode", "single", 1),
    ]
    # "plan-sha" is the plan hash in the provenance file of _pod.
    plan = dataclasses.replace(plan, sha256="plan-sha")
    cells = {cell.workload_kind: cell for cell in plan.cells}
    # The reader's own operators for the same model, built without the plan.
    model_config = sdk_config.ModelConfig(
        tp_size=1,
        pp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        forward_model="fpm",
    )
    model = models.get_model(READER_MODEL, model_config, READER_BACKEND)
    reader_ops = {"prefill": model.context_ops[0], "decode": model.generation_ops[0]}
    assert model.model_path == plan.model_path

    # Plan each phase, write its rank artifact with FAKE times, and collect the rows AISim publishes for it.
    grids = {}
    fake_ms = {}  # (phase, benchmark id) -> the FAKE time in the published unit
    rows = []
    for phase, cell in cells.items():
        built = grid.build_grid(phase, limits)
        assert built.missing_phases == () and built.source == "generated"
        grids[phase] = built
        cell_dir = root / "cells" / cell.cell_id
        pod = _pod(cell_dir, cell)
        measurements = []
        fake_elapsed = 0.0
        for record in built.point_dicts():
            fake_seconds = fake_seconds_of(record)
            if phase == "prefill":
                fpm = artifact.build_fpm(record, 0, fake_seconds)
            else:
                fake_steps = [fake_seconds * factor for factor in FAKE_DECODE_STEP_FACTORS]
                assert statistics.median_high(fake_steps[2:5]) == fake_seconds
                fake_sample = artifact.decode_wall_time(fake_steps)
                fpm = artifact.build_fpm(record, 0, fake_sample, extra={artifact.DECODE_STEPS_KEY: fake_steps})
            measurements.append(artifact.build_measurement(record, {0: fpm}))
            fake_elapsed += FAKE_POINT_OVERHEAD_SECONDS + fake_seconds
            # AISim publishes milliseconds: the wall time times 1000.0 (aggregate_cell).
            fake_ms[phase, record["benchmark_id"]] = fake_seconds * 1000.0
        payload = artifact.build_rank_artifact(
            dp_rank=0,
            dp_size=1,
            mode=phase,
            run_id=FAKE_RUN_ID,
            grid_digest=built.digest,
            expected_points=len(built.points),
            measurements=measurements,
            benchmark_elapsed_seconds=fake_elapsed,
            config={"points_source": built.source},
            producer={
                "timing_source": "FAKE: typed into test_fpm_sglbench_producer.py, never measured",
                "sglang_version": POD_SGLANG_VERSION,
                # What the launcher records of the layout in a pod of this cell: SGLang's TP size, one
                # attention-DP rank and no block of expert layers. Publication holds the three against the
                # cell of a dense plan.
                "tp_size": cell.topology.tp,
                "attn_dp_size": 1,
                "moe": None,
                # What the pod of this cell records about its KV cache. Publication holds the stored dtype
                # against the KV-cache label of the cell.
                **POD_KV_CACHE[cell.kv_cache_dtype],
            },
        )
        artifact.write_artifacts(str(pod / "benchmark.json"), [payload])
        cell_rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
        # Every row carries the cell identity the reader's operator of this phase asks for, column by column.
        published_identity = {
            tuple("" if row[column] is None else str(row[column]) for column in _CELL_MATCH_COLUMNS)
            for row in cell_rows
        }
        assert published_identity == {reader_ops[phase]._match_identity}
        # One row per planned point, in order, each with its own FAKE time.
        published = [row["latency_ms"] for row in cell_rows]
        assert published == [fake_ms[phase, point.benchmark_id] for point in built.points]
        rows.extend(cell_rows)
    parquet_path, metadata_path, skipped_cells = write_formal_database(plan, rows, systems_root=systems_root / "data")
    # The table holds every planned point of both phases, in the directory the reader opens for this
    # system, backend and version.
    assert skipped_cells == ()
    assert parquet_path.parent == systems_root / "data" / READER_SYSTEM / READER_BACKEND / FAKE_READER_VERSION
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["row_count"] == len(rows)
    assert len(rows) == sum(len(built.points) for built in grids.values())

    database = PerfDatabase(READER_SYSTEM, READER_BACKEND, FAKE_READER_VERSION, systems_root=str(systems_root))
    handle = EngineHandle.for_database(database, systems_path=str(systems_root))
    # The reader's two interpolation modes: the one the op spec carries by default, and "direct".
    op_specs = {}
    for phase, reader_op in reader_ops.items():
        default_spec = _fpm_spec_dict(reader_op)
        default_mode = default_spec["FpmForward"]["interpolation"]
        assert default_mode != "direct"
        direct_spec = {"FpmForward": {**default_spec["FpmForward"], "interpolation": "direct"}}
        op_specs[phase] = {default_mode: json.dumps([default_spec]), "direct": json.dumps([direct_spec])}

    return _FakeTable(grids, fake_ms, op_specs, handle)


@pytest.mark.parametrize("h200", [H200_PAGE_1, H200_PAGE_64], ids=["page-1", "page-64"])
def test_default_h200_grid_published_with_fake_times_is_served_by_the_compiled_reader(tmp_path, monkeypatch, h200):
    """Plan the default grid, write it with FAKE times, publish it with AISim's own Python path, and read it back.

    The path is ``_publish_with_fake_times``. This is the one test that fails when an axis rule plans a table
    the reader cannot serve. It says nothing about how close an interpolated answer is to an engine: the FAKE
    times have no shape.
    """
    limits = _default_h200_limits(h200)
    table = _publish_with_fake_times(tmp_path, monkeypatch, limits, _fake_reader_seconds)
    grids, fake_ms, op_specs, reader_ms = table.grids, table.fake_ms, table.op_specs, table.reader_ms
    # No two points of a phase have the same FAKE time, so a value the reader hands back is the point's own row.
    for phase, built in grids.items():
        assert len({fake_ms[phase, point.benchmark_id] for point in built.points}) == len(built.points)

    # 1. Every planned point reads back its own FAKE time, in both modes. The reader hands a collected point
    #    back as it is stored (default mode: crates/core/src/perfmodel/perf_database/perf_interp.rs
    #    ``SiteIndex::eval_curve``; direct mode: crates/core/src/perfmodel/operators/fpm_forward.rs
    #    ``direct_curve_value``), so the values are compared for equality.
    wrong = []
    for phase, built in grids.items():
        for point in built.points:
            per_request = point.new_tokens if phase == "prefill" else point.cached_tokens
            cached = point.cached_tokens if phase == "prefill" else 0
            for mode in op_specs[phase]:
                read = reader_ms(phase, mode, point.batch_size, per_request, cached)
                if read != fake_ms[phase, point.benchmark_id]:
                    wrong.append((phase, mode, point, read))
    assert not wrong, (len(wrong), wrong[:5])

    # 2. Queries between collected rows are answered, in both modes. Only that an answer comes back is
    #    asserted: the reader raises when it has none.
    decode_sizes = limits.decode_graph_sizes
    decode_batches = {point.batch_size for point in grids["decode"].points}
    new_tokens = {point.new_tokens for point in grids["prefill"].points}
    answered = {}
    # A decode batch of 6 lies between two neighbouring decode graph sizes, 4 and 8, and was not collected.
    assert {4, 8} <= set(decode_sizes) and not any(4 < size < 8 for size in decode_sizes)
    # A decode batch of 300 lies above the largest decode graph size, 256, and was not collected.
    assert max(decode_sizes) == 256
    assert not {6, 300} & decode_batches
    for mode in op_specs["decode"]:
        for batch in (6, 300):
            answered["decode", mode, batch] = reader_ms("decode", mode, batch, 1024)
    # A prefill step of one request with 1, 2 or 3 new tokens and nothing cached lies below the smallest
    # prefill graph size, 4. Of the three, 1 and 2 were collected: the low end the grid puts on the axis. 3 was
    # not, and lies between the collected rows at 2 and 4.
    assert min(limits.prefill_graph_sizes) == 4 and new_tokens & {1, 2, 3, 4} == {1, 2, 4}
    for mode in op_specs["prefill"]:
        for new in (1, 2, 3):
            answered["prefill", mode, new] = reader_ms("prefill", mode, 1, new)
    assert len(answered) == 10 and all(math.isfinite(value) for value in answered.values()), answered
