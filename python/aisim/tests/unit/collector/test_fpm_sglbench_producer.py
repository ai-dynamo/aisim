# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# The fake engine follows the prefix cache of SGLang v0.5.22 and v0.5.23; see THIRD_PARTY_NOTICES.md.
# Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d
# Source: https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90
# The simulated driver follows the staging procedure and two failure reason strings of ai-dynamo/dynamo (Apache-2.0,
# https://github.com/ai-dynamo/dynamo), components/src/dynamo/vllm/instrumented_scheduler.py at
# b20fe6fe94021b854597965c7305931175e8f482. See THIRD_PARTY_NOTICES.md.

"""Test of the three modules of the SGLang self-benchmark producer core together. No SGLang, no GPU.

One simulated run goes through the three pod modules and then through AISim's two real checkers::

    sglbench_grid      plan the points (generated grid or points manifest)
    sglbench_content   prompts and request ids for every point
    fake engine        answers each call with cached-token counts (a model, see below)
    sglbench_artifact  rows, rank artifacts, merged file, files on disk
    validator          collector.fpm_forward.native_artifact.validate_native_collection
    gate               the Python text inside check_result_files() of runtime/fpm_exec.sh

One test goes past the two checkers: it publishes a planned grid with AISim's own Python path
(``aggregate_cell``, then ``write_formal_database``) and reads the table back through AISim's compiled reader.
One test of partial and failed runs does without them: both reject such an artifact on its envelope, so what
the file carries beyond the envelope is compared with the complete run of the same grid.

Everything that stands in for an engine is FAKE and is named so:

- ``_FakeEngine`` is one ``PageCacheModel`` per attention-DP rank (the model of SGLang's prefix cache written
  in ``test_fpm_sglbench_content.py``). Its cached-token counts are model results, not measurements.
- Every time value that goes into an artifact is FAKE. It comes from ``_fake_extend_seconds``,
  ``_fake_decode_step_seconds``, ``_fake_reader_seconds``, ``_FakeClock`` or a ``FAKE_*`` constant, or it is
  typed into the pod script, which cannot import this file. These values were typed in to exercise the
  arithmetic and were never measured. The artifacts say so in ``producer``.
- ``_simulate_run`` is the simulated driver: test code in the place of the code that drives an engine. For a
  prefill point it sends the fake engine a seed (when the point has cached tokens), a warm shot and the
  measured shot; for a decode point a seed and the run. A point that fails a validity check is staged once
  more and ends the run when it fails again. The soft deadline is looked at between two points. Three of
  these steps follow Dynamo's vLLM self-benchmark (``instrumented_scheduler.py`` at the revision in the header
  of this file): the order seed, warm shot, measured shot of a cached prefill point
  (``_bench_realseed_stage_point``, ``_bench_realseed_pending_step``), the one re-stage of such a point after
  its measured shot missed the cache, and the deadline check before the next point
  (``_bench_stop_at_timeout_boundary``).

The two checkers are used as shipped, through the helpers of ``test_fpm_sglbench_artifact.py``. Its ``_pod``
writes the collected layout with a provenance file that says ``"backend": "vllm"``: fixture scaffolding,
because the validator accepts no other value (native_artifact.py, ``_validate_collector_provenance``), and no
statement about which engine produced an artifact.
"""

from __future__ import annotations

import collections
import dataclasses
import itertools
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.fpm_forward.config import _freeze_benchmark_points
from collector.fpm_forward.database import aggregate_cell, write_formal_database
from collector.fpm_forward.runtime import sglbench_artifact as artifact
from collector.fpm_forward.runtime import sglbench_content as content
from collector.fpm_forward.runtime import sglbench_grid as grid

import aisim_core
from aisim.sdk import common, models
from aisim.sdk import config as sdk_config
from aisim.sdk.perf_database import PerfDatabase
from aisim_core.sdk.engine import EngineHandle, _fpm_spec_dict
from aisim_core.sdk.operations.fpm_forward import _CELL_MATCH_COLUMNS

from .test_fpm_runner import _cell, _plan
from .test_fpm_sglbench_artifact import GATE_ACCEPT, GATE_INVALID, _gate, _pod, _validate
from .test_fpm_sglbench_content import PageCacheModel
from .test_fpm_sglbench_grid import H200_DECODE_GRAPH_ROWS, H200_PAGE_1, H200_PAGE_64, H200_PREFILL_GRAPH_ROWS, _ladder

pytestmark = pytest.mark.unit

RUNTIME = Path(__file__).resolve().parents[3] / "collector/fpm_forward/runtime"

PAGE_SIZES = (1, 16, 64)
# For complete runs also 3 and 24, which do not divide 16: there the cached-token unit ceilP(16) is 18 and 24.
COMPLETE_RUN_PAGE_SIZES = (1, 3, 16, 24, 64)
PHASES = ("prefill", "decode")

# FAKE run inputs. A caller of ``ContentPlan`` must supply the token range (inside the served model's
# vocabulary) and the seed (the value of DYN_BENCH_CONTENT_SEED), as the docstring of sglbench_content.py says;
# the run id has hyphens on purpose (request ids must still parse).
TOKEN_LOW, TOKEN_HIGH = 1000, 51_000
FAKE_CONTENT_SEED = "0"
FAKE_RUN_ID = "fake-run-2026.11.14"
FAKE_TIMEOUT_SECONDS = 10_800.0
FAKE_POINT_OVERHEAD_SECONDS = 0.25
WARMUP_PROMPT_TOKENS = 256

# The reasons the simulated driver gives for a point that fails a validity check. Both are reason strings of
# Dynamo's vLLM self-benchmark (instrumented_scheduler.py at the revision in the header of this file). There
# "measured_prefill_tokens_mismatch" means that the new tokens of the measured batch differ from the point's
# total (_bench_fpm_validation_failure), and "real_seed_cache_validation_failed" that the measured shot of a
# seeded prefill point missed its cache hit twice (_bench_realseed_pending_step). Here the first is given when
# a prompt is not as long as the planned cached and new tokens together, the second when a request reports
# another cached count than planned. No other AISim code uses the two strings.
REASON_PROMPT_TOKENS = "measured_prefill_tokens_mismatch"
REASON_CACHED_TOKENS = "real_seed_cache_validation_failed"
# FAKE reason of a stop that belongs to no point: the failed form ``{"point": None, "reason": ...}``.
FAKE_FAULT_REASON = "fake_fault_outside_any_point"


def _small_limits(page_size: int, **overrides) -> grid.Limits:
    """Small limits that grow with the page size, so every page size has several cached prefixes and contexts.

    Page 1 and 16: I = 169, S = 1184, R = 9, M = 80, C = 96. Page 64: I = 553, S = 2304, R = 9, M = 320,
    C = 384. The input limit ends the axes of small batches and the pool ends those of large ones.
    """
    unit = grid.cached_unit(page_size)  # G, the step of the cached-token axis
    record = {
        "page_size": page_size,
        "max_req_input_len": 8 * unit + 41,
        "max_total_num_tokens": grid.floor_pages(24 * unit + 800, page_size),
        "max_running_requests": 9,
        "max_prefill_tokens": 5 * unit,
        "chunked_prefill_size": 6 * unit,
        "decode_graph_sizes": (1, 2, 4, 8),
        "prefill_graph_sizes": (4, 8, 16, 32, 64),
    }
    record.update(overrides)
    return grid.Limits(**record)


# ---------------------------------------------------------------------------------------------------------------
# Fake engine, fake clock, fake timings
# ---------------------------------------------------------------------------------------------------------------


class _FakeEngine:
    """Stands in for SGLang: one prefix-cache model per attention-DP rank, and nothing else.

    A call is one batch: every request of the call is matched against the cache before any of them is stored.
    """

    def __init__(self, page_size: int, dp_size: int) -> None:
        self.page_size = page_size
        self.dp_size = dp_size
        self.caches = [PageCacheModel(page_size) for _ in range(dp_size)]
        self.request_ids: list[str] = []  # every id ever submitted, in order
        # FAKE model output. Real output is vocabulary tokens, so the ids are drawn inside the plan's range.
        self._fake_output = random.Random(20261114)

    def generate(self, rank: int, request_ids: list[str], prompts: list[list[int]], output_tokens: int) -> list[int]:
        """Submit one batch call to one rank; return the cached-token count of each request."""
        assert len(request_ids) == len(prompts)
        self.request_ids.extend(request_ids)
        cache = self.caches[rank]
        cached = [cache.cached(prompt) for prompt in prompts]
        for prompt in prompts:
            if output_tokens == grid.PREFILL_OUTPUT_TOKENS:
                cache.finish_prefill(prompt)
            else:
                assert output_tokens == grid.DECODE_OUTPUT_TOKENS
                # All but the last generated token have KV and reach the cache.
                generated = [self._fake_output.randrange(TOKEN_LOW, TOKEN_HIGH) for _ in range(output_tokens - 1)]
                cache.finish_decode(prompt, generated)
        return cached

    def fake_eviction(self) -> None:
        """FAKE fault: the engine needed room and dropped every unlocked cache entry, on every rank."""
        self.caches = [PageCacheModel(self.page_size) for _ in range(self.dp_size)]


class _FakeClock:
    """A FAKE monotonic clock that only moves when the simulated driver says so."""

    def __init__(self) -> None:
        self.now = 5000.0

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _fake_extend_seconds(point: dict, rank: int) -> float:
    """FAKE duration of the measured extend batch of a prefill point. Each rank is the slower one in turn."""
    slower = 0.0005 if (point["benchmark_id"] + rank) % 2 else 0.0
    return 0.004 + 1e-6 * point["total_prefill_tokens"] + slower


# FAKE shape of the six decode steps around one base time: a short first step, a short last step, unsorted
# kept steps. Decode steps 3 to 5 are the list items 2, 3 and 4, and their upper median is the base itself.
FAKE_DECODE_STEP_FACTORS = (0.6, 1.5, 1.2, 0.9, 1.0, 0.3)


def _fake_decode_step_seconds(point: dict, rank: int) -> list[float]:
    """FAKE durations of the six decode steps: a short first step, a short last step, unsorted kept steps."""
    slower = 0.0005 if (point["benchmark_id"] + rank) % 2 else 0.0
    base = 0.003 + 1e-7 * point["total_kv_read_tokens"] + slower
    return [base * factor for factor in FAKE_DECODE_STEP_FACTORS]


def _expected_sample_seconds(point: dict, rank: int) -> float:
    """The FAKE sample a row must carry, worked out without the artifact module."""
    if point["point_type"] == "decode":
        # Decode steps 3 to 5 are the list items 2, 3 and 4; the upper median of three values is the middle one.
        return statistics.median_high(_fake_decode_step_seconds(point, rank)[2:5])
    return _fake_extend_seconds(point, rank)


def _fake_reader_seconds(point: dict) -> float:
    """FAKE time of a point in the reader test: one millisecond per benchmark id.

    No two points of one phase share a value, so a value the reader hands back can only be the point's own row.
    """
    return 0.001 * point["benchmark_id"]


# ---------------------------------------------------------------------------------------------------------------
# Simulated driver
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class _SimulatedRun:
    """What the simulated driver hands back."""

    built: grid.Grid
    payloads: list[dict]  # one rank artifact per attention-DP rank
    request_ids: list[str]  # every request id sent, in order
    # One entry per TIMED request: (benchmark_id, stage, attempt, rank, slot, cached tokens, extend tokens).
    observed: list[tuple[int, str, int, int, int, int, int]]
    restaged: list[int]  # benchmark ids whose first attempt failed a validity check
    plan: content.ContentPlan  # the content plan after the run
    content_stats: dict  # what the plan had issued when the run ended


def _request_ids(point: grid.Point, stage: str, attempt: int, rank: int) -> list[str]:
    return [
        content.request_id(FAKE_RUN_ID, point.benchmark_id, stage, attempt, rank, slot)
        for slot in range(point.batch_size)
    ]


def _stage_point(
    engine: _FakeEngine, plan: content.ContentPlan, point: grid.Point, attempt: int, evict: bool, observed: list
) -> str | None:
    """Stage and run one attempt of one point on every rank; return the failed check's reason, or ``None``.

    Prefill: seed (when k > 0), warm shot, measured shot. Decode: seed, measured run with the same prompts.
    ``evict`` makes the fake engine drop its cache just before the measured call of this attempt.
    """
    page = engine.page_size
    batch, new, cached_or_context = point.batch_size, point.new_tokens, point.cached_tokens
    expected_cached = grid.expected_cached_tokens(point.phase, cached_or_context, page)
    expected_extend = grid.expected_extend_tokens(point.phase, new, cached_or_context, page)
    reason = None
    for rank in range(engine.dp_size):
        timed = []
        if point.phase == grid.PREFILL:
            if cached_or_context > 0:
                seeds = [plan.prefill_seed_prompt(rank, slot, cached_or_context) for slot in range(batch)]
                engine.generate(rank, _request_ids(point, "seed", attempt, rank), seeds, grid.PREFILL_OUTPUT_TOKENS)
            for stage in ("warm", "meas"):
                if evict and stage == "meas":
                    engine.fake_eviction()
                prompts = [plan.prefill_prompt(rank, slot, cached_or_context, new) for slot in range(batch)]
                counts = engine.generate(
                    rank, _request_ids(point, stage, attempt, rank), prompts, grid.PREFILL_OUTPUT_TOKENS
                )
                timed.append((stage, prompts, counts))
        else:
            prompts = [plan.decode_prompt(rank, slot, cached_or_context) for slot in range(batch)]
            engine.generate(rank, _request_ids(point, "seed", attempt, rank), prompts, grid.PREFILL_OUTPUT_TOKENS)
            if evict:
                engine.fake_eviction()
            run_prompts = [plan.decode_prompt(rank, slot, cached_or_context) for slot in range(batch)]
            assert run_prompts == prompts  # the seed and the run use the same prompt
            counts = engine.generate(
                rank, _request_ids(point, "run", attempt, rank), prompts, grid.DECODE_OUTPUT_TOKENS
            )
            timed.append(("run", prompts, counts))
        # The driver's validity checks, on what the engine reported for each timed request.
        for stage, prompts, counts in timed:
            for slot, (prompt, count) in enumerate(zip(prompts, counts, strict=True)):
                observed.append((point.benchmark_id, stage, attempt, rank, slot, count, len(prompt) - count))
                if len(prompt) != expected_cached + expected_extend:
                    reason = reason or REASON_PROMPT_TOKENS
                elif count != expected_cached:
                    reason = reason or REASON_CACHED_TOKENS
    return reason


def _kept_batch_counters(point: grid.Point, observed: list, attempt: int, rank: int) -> dict:
    """The five scheduler counters of the kept batch of one rank, from what the fake engine reported for it.

    Prefill: the extend batch of the measured shot. Decode: the middle kept decode step. The decode step with
    zero-based index ``i`` reads the prompt plus the ``i`` output tokens that already have KV.
    """
    stage = "meas" if point.phase == grid.PREFILL else "run"
    entries = [entry for entry in observed if entry[:4] == (point.benchmark_id, stage, attempt, rank)]
    cached = sum(entry[5] for entry in entries)
    extend = sum(entry[6] for entry in entries)
    if point.phase == grid.PREFILL:
        return {
            "num_prefill_requests": len(entries),
            "sum_prefill_tokens": extend,
            "sum_prefill_kv_tokens": cached,
            "num_decode_requests": 0,
            "sum_decode_kv_tokens": 0,
        }
    middle = artifact.DECODE_KEPT_STEP_INDICES[len(artifact.DECODE_KEPT_STEP_INDICES) // 2]
    return {
        "num_prefill_requests": 0,
        "sum_prefill_tokens": 0,
        "sum_prefill_kv_tokens": 0,
        "num_decode_requests": len(entries),
        "sum_decode_kv_tokens": cached + extend + middle * len(entries),
    }


def _simulate_run(
    mode: str,
    limits: grid.Limits,
    *,
    dp_size: int,
    caps: grid.SampleCaps | None = None,
    manifest: dict | None = None,
    config: dict | None = None,
    fake_deadline_after: int | None = None,
    fake_evictions: dict[int, int] | None = None,
    fake_fault_after: int | None = None,
) -> _SimulatedRun:
    """Plan a run, stage every point against the fake engine and build the rank artifacts.

    fake_deadline_after
        After this many completed points the fake clock jumps past the soft deadline.
    fake_evictions
        ``{benchmark_id: attempts}``: the fake engine drops its cache before the measured call of that many
        attempts of the point. One is recovered by the re-stage; two fail the point.
    fake_fault_after
        After this many completed points a FAKE fault that belongs to no point stops the run. 0 stops it before
        the first point; the number of planned points stops it after the last one.
    """
    page = limits.page_size
    built = grid.build_grid(mode, limits, caps, manifest=manifest)
    plan = content.ContentPlan(FAKE_CONTENT_SEED, TOKEN_LOW, TOKEN_HIGH)
    engine = _FakeEngine(page, dp_size)
    clock = _FakeClock()
    started = clock.now
    evictions = fake_evictions or {}

    # Global warm-up: one prompt per rank that belongs to no point (benchmark id 0) and finds nothing cached.
    for rank in range(dp_size):
        warm_up = plan.free_prompt(rank, WARMUP_PROMPT_TOKENS)
        rid = content.request_id(FAKE_RUN_ID, 0, "free", 0, rank, 0)
        assert engine.generate(rank, [rid], [warm_up], grid.PREFILL_OUTPUT_TOKENS) == [0]
    clock.advance(FAKE_POINT_OVERHEAD_SECONDS)

    measurements: list[dict] = []
    observed: list = []
    restaged: list[int] = []
    stop_reason = None
    failure = None
    # A phase without any point fails the run before the sweep starts.
    for point in () if built.missing_phases else built.points:
        if len(measurements) == fake_fault_after:  # the FAKE fault ends the run at a point boundary
            break
        if clock.now - started >= FAKE_TIMEOUT_SECONDS:  # the soft deadline ends the run at a point boundary
            stop_reason = artifact.STOP_REASON_TIMEOUT
            break
        record = point.as_dict()
        reason = None
        attempt = 0
        for attempt in (0, 1):  # a point that fails its checks is re-staged once
            evict = evictions.get(point.benchmark_id, 0) > attempt
            reason = _stage_point(engine, plan, point, attempt, evict, observed)
            if reason is None:
                break
            if attempt == 0:
                restaged.append(point.benchmark_id)
        if reason is not None:  # still failing after the re-stage: the run stops here
            failure = {"point": record, "reason": reason}
            break

        rank_fpms = {}
        for rank in range(dp_size):
            if point.phase == grid.PREFILL:
                fpm = artifact.build_fpm(record, rank, _fake_extend_seconds(record, rank))
            else:
                steps = _fake_decode_step_seconds(record, rank)
                extra = {artifact.DECODE_STEPS_KEY: steps, "worker_id": f"fake-worker-{rank}"}
                fpm = artifact.build_fpm(record, rank, artifact.decode_wall_time(steps), extra=extra)
            # The counters the artifact derives from the planned totals are the ones the engine reported.
            assert fpm["scheduled_requests"] == _kept_batch_counters(point, observed, attempt, rank), record
            rank_fpms[rank] = fpm
        measurements.append(artifact.build_measurement(record, rank_fpms))
        clock.advance(FAKE_POINT_OVERHEAD_SECONDS + max(fpm["wall_time"] for fpm in rank_fpms.values()))
        if len(measurements) == fake_deadline_after:
            clock.advance(FAKE_TIMEOUT_SECONDS)
    if len(measurements) == fake_fault_after:
        # The fault names no point. It also ends a run whose last point has completed: every row is there and
        # the run has failed all the same.
        assert failure is None and stop_reason is None
        failure = {"point": None, "reason": FAKE_FAULT_REASON}

    producer = {
        "timing_source": "FAKE: typed into test_fpm_sglbench_producer.py, never measured",
        "engine": "FAKE: a page-cache model, no SGLang",
        "page_size": page,
        "limits": limits.as_dict(),
        "content": {
            "source": "seeded_token_ids",
            "seed": plan.seed,
            "token_range": [plan.token_low, plan.token_high],
            "issued": plan.stats(),
        },
        "restaged_points": restaged,
    }
    payloads = [
        artifact.build_rank_artifact(
            dp_rank=rank,
            dp_size=dp_size,
            mode=mode,
            run_id=FAKE_RUN_ID,
            grid_digest=built.digest,
            expected_points=len(built.points),
            measurements=measurements,
            benchmark_elapsed_seconds=clock.now - started,
            config={"points_source": built.source, "content_seed": FAKE_CONTENT_SEED, **(config or {})},
            producer=producer,
            missing_phases=built.missing_phases,
            stop_reason=stop_reason,
            failure=failure,
        )
        for rank in range(dp_size)
    ]
    return _SimulatedRun(built, payloads, engine.request_ids, observed, restaged, plan, plan.stats())


def _expected_regime(point: grid.Point) -> str:
    if point.phase == "decode":
        return "real_kv"
    return "real_prefix" if point.cached_tokens else "not_applicable"


def _coordinates(point_dicts: list[dict]) -> list[tuple[int, int, int]]:
    return [
        (point["batch_size"], point["total_prefill_tokens"], point["total_kv_read_tokens"]) for point in point_dicts
    ]


def _assert_accepted(tmp_path: Path, cell, run: _SimulatedRun, written: dict) -> None:
    """A complete run: the validator returns every planned point with its FAKE times and the gate exits 0."""
    points = run.built.point_dicts()
    dp_size = len(run.payloads)
    collection = _validate(cell, tmp_path)
    assert [measurement.point for measurement in collection.points] == points
    assert [measurement.kv_seed_regime for measurement in collection.points] == [
        _expected_regime(point) for point in run.built.points
    ]
    for point, measurement in zip(points, collection.points, strict=True):
        assert measurement.rank_wall_times == tuple(
            (rank, _expected_sample_seconds(point, rank)) for rank in range(dp_size)
        )
    assert (collection.runtime_run_id, collection.runtime_grid_digest) == (FAKE_RUN_ID, run.built.digest)
    gate = _gate(written["rank_files"], cell.workload_kind)
    assert gate.returncode == GATE_ACCEPT, gate.stderr


def _assert_rejected(tmp_path: Path, cell, written: dict, status: str) -> str:
    """A run that is not complete: the validator raises and the gate fails the pod. Returns the gate's message."""
    with pytest.raises(ValueError, match="invalid native terminal result envelope"):
        _validate(cell, tmp_path)
    gate = _gate(written["rank_files"], cell.workload_kind)
    assert gate.returncode == GATE_INVALID
    assert f"status={status!r} valid=False" in gate.stderr
    return gate.stderr


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
    assert (content.DECODE_PROMPT_LEAD, content.MIN_DECODE_CONTEXT, content.DECODE_OUTPUT_TOKENS) == (
        grid.DECODE_PROMPT_LEAD,
        grid.MIN_DECODE_CONTEXT,
        grid.DECODE_OUTPUT_TOKENS,
    )
    assert (artifact.PREFILL, artifact.DECODE, artifact.AGG, artifact.MODES) == (
        grid.PREFILL,
        grid.DECODE,
        grid.AGG,
        grid.MODES,
    )
    assert (artifact.STAMP_PREFILL_REAL_SEED, artifact.STAMP_DECODE_REAL_KV) == (
        grid.STAMP_PREFILL_REAL_SEED,
        grid.STAMP_DECODE_REAL_KV,
    )
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
    # Nothing is left out above: every public upper-case name that two of the modules define has one value, of
    # one type at every depth. The integer 7 and the float 7.0 compare equal and are not one value here.
    assert _with_types((7, [1, (2,)], {"a": {3}})) != _with_types((7.0, [1, (2,)], {"a": {3}}))
    assert _with_types((7, [1, (2,)], {"a": {3}})) != _with_types((7, [1, [2]], {"a": {3}}))
    assert _with_types((7, [1, (2,)], {"a": {3}})) != _with_types((7, [1, (2,)], {"a": {3.0}}))
    assert _with_types((7, [1, (2,)], {"a": {3}})) == _with_types((7, [1, (2,)], {"a": {3}}))
    shared_names = {}
    for first, second in itertools.combinations((grid, content, artifact), 2):
        names = {name for name in vars(first) if name.isupper() and not name.startswith("_")} & set(vars(second))
        for name in names:
            one, other = getattr(first, name), getattr(second, name)
            assert one == other and _with_types(one) == _with_types(other), (first.__name__, second.__name__, name)
        shared_names[first, second] = names
    assert shared_names[grid, content] >= {"DECODE_PROMPT_LEAD", "MIN_DECODE_CONTEXT", "DECODE_OUTPUT_TOKENS"}
    assert shared_names[grid, artifact] >= {"PREFILL", "DECODE", "AGG", "MODES", "STAMP_DECODE_REAL_KV"}
    # The fpm key of the raw decode step times has one home, the artifact module. The other two modules do not
    # name it, so there is no copy to compare.
    assert artifact.DECODE_STEPS_KEY == "sglbench_decode_steps"
    for module in (grid, content):
        assert not hasattr(module, "DECODE_STEPS_KEY")
        assert artifact.DECODE_STEPS_KEY not in Path(module.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize("page_size", PAGE_SIZES)
def test_every_planned_point_is_a_point_the_artifact_module_accepts(page_size):
    limits = _small_limits(page_size)
    built = grid.build_grid("agg", limits)
    assert built.missing_phases == ()
    for point, record in zip(built.points, built.point_dicts(), strict=True):
        artifact.validate_point(record)
        assert artifact.kv_seed_regime(record) == _expected_regime(point)
        counters = artifact.scheduled_requests(record)
        if point.phase == "prefill":
            assert (counters["num_prefill_requests"], counters["sum_prefill_tokens"]) == (
                point.batch_size,
                point.batch_size * point.new_tokens,
            )
            assert counters["sum_prefill_kv_tokens"] == point.batch_size * point.cached_tokens
        else:
            assert (counters["num_decode_requests"], counters["sum_decode_kv_tokens"]) == (
                point.batch_size,
                point.batch_size * point.cached_tokens,
            )
    assert json.loads(json.dumps(built.point_dicts())) == built.point_dicts()  # plain JSON data


# ---------------------------------------------------------------------------------------------------------------
# Complete runs: both checkers accept
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dp_size", "strategy", "chunked"),
    [(1, "single", True), (2, "dep", True), (2, "tp", False)],
    ids=["dp1-single", "dp2-dep", "dp2-tp-unchunked"],
)
@pytest.mark.parametrize("page_size", COMPLETE_RUN_PAGE_SIZES)
@pytest.mark.parametrize("phase", PHASES)
def test_complete_run_passes_the_validator_and_the_gate(tmp_path, phase, page_size, dp_size, strategy, chunked):
    # "single" and "tp" are what a dense SGLang cell gets; "dep" makes the kvwarm block mandatory for decode.
    limits = _small_limits(page_size) if chunked else _small_limits(page_size, chunked_prefill_size=None)
    cell = _cell(phase=phase, dp=dp_size, strategy=strategy)
    pod = _pod(tmp_path, cell)

    run = _simulate_run(phase, limits, dp_size=dp_size)

    points = run.built.points
    assert len(points) > 40 and run.built.missing_phases == () and run.built.source == "generated"
    assert run.restaged == []
    # Every timed request found exactly the planned tokens in the cache and computed exactly the planned rest.
    for benchmark_id, stage, attempt, _rank, _slot, cached, extend in run.observed:
        point = points[benchmark_id - 1]
        assert attempt == 0
        assert cached == grid.expected_cached_tokens(phase, point.cached_tokens, page_size), (benchmark_id, stage)
        assert extend == grid.expected_extend_tokens(phase, point.new_tokens, point.cached_tokens, page_size)
        # The same two numbers, worked out here from the staging facts.
        if phase == "prefill":
            assert (cached, extend) == (point.cached_tokens, point.new_tokens)
        else:
            whole_pages = (point.cached_tokens - 4) // page_size * page_size
            assert (cached, extend) == (whole_pages, point.cached_tokens - 3 - whole_pages)
    timed_calls = 2 if phase == "prefill" else 1  # warm and measured shot, or the decode run
    requests = sum(point.batch_size for point in points)
    assert len(run.observed) == dp_size * timed_calls * requests

    # Request ids: one per prompt, never repeated, none a prefix of another, all parse back.
    seeded = sum(point.batch_size for point in points if phase == "decode" or point.cached_tokens)
    stages = collections.Counter(content.parse_request_id(rid)["stage"] for rid in run.request_ids)
    expected_stages = {"free": dp_size, "seed": dp_size * seeded}
    expected_stages.update(
        {"warm": dp_size * requests, "meas": dp_size * requests} if phase == "prefill" else {"run": dp_size * requests}
    )
    assert stages == {stage: count for stage, count in expected_stages.items() if count}
    assert len(set(run.request_ids)) == len(run.request_ids)
    ordered = sorted(run.request_ids)
    assert not any(longer.startswith(shorter) for shorter, longer in itertools.pairwise(ordered))
    assert all(content.parse_request_id(rid)["run_id"] == FAKE_RUN_ID for rid in run.request_ids)

    # What the content plan issued is what the grid planned.
    zero_cached = sum(point.batch_size for point in points if phase == "prefill" and not point.cached_tokens)
    assert run.content_stats["free_prompts"] == dp_size
    assert run.content_stats["zero_cached_prompts"] == dp_size * 2 * zero_cached
    if phase == "prefill":
        assert run.content_stats["seed_prompts"] == dp_size * seeded
        assert run.content_stats["shots"] == dp_size * 2 * seeded
        assert run.content_stats["decode_prompts"] == 0
    else:
        assert run.content_stats["decode_prompts"] == dp_size * 2 * requests  # seed and run
        assert (run.content_stats["seed_prompts"], run.content_stats["shots"]) == (0, 0)

    for payload in run.payloads:
        assert (payload["status"], payload["valid"], payload["usable"]) == ("complete", True, True)
        assert payload["coverage"] == {
            "expected_points": len(points),
            "completed_points": len(points),
            "skipped_points": 0,
        }
        assert payload["grid_digest"] == run.built.digest == grid.grid_digest(run.built.point_dicts())
        assert payload["producer"]["limits"] == limits.as_dict()
        assert payload["producer"]["timing_source"].startswith("FAKE")
    written = artifact.write_artifacts(str(pod / "benchmark.json"), run.payloads)
    for path, payload in zip(written["rank_files"], run.payloads, strict=True):
        assert json.loads(Path(path).read_text(encoding="utf-8")) == payload
    # The merged file binds the run to its rank files and carries every rank's row.
    merged = json.loads(Path(written["merged_output_path"]).read_text(encoding="utf-8"))
    assert (merged["run_id"], merged["grid_digest"], merged["rank_files"]) == (
        FAKE_RUN_ID,
        run.built.digest,
        written["rank_files"],
    )
    assert merged["dp"]["source_ranks"] == list(range(dp_size))
    for index, row in enumerate(merged["results"]):
        assert row["fpms"] == [payload["results"][index]["fpms"][0] for payload in run.payloads]

    _assert_accepted(tmp_path, cell, run, written)

    # Publication: one row per planned point (a repeated coordinate would raise), at the planned coordinates,
    # with the slowest rank's FAKE time. Decode rows are published as real KV.
    rows = aggregate_cell(_plan(cell), cell, tmp_path, expected_attempt_id="attempt")
    assert [(row["batch_size"], row["total_prefill_tokens"], row["total_kv_read_tokens"]) for row in rows] == (
        _coordinates(run.built.point_dicts())
    )
    for record, row in zip(run.built.point_dicts(), rows, strict=True):
        slowest = max(_expected_sample_seconds(record, rank) for rank in range(dp_size))
        assert row["latency_ms"] == slowest * 1000.0
    assert {row["kv_seed_regime"] for row in rows} == ({"real_kv"} if phase == "decode" else {"n/a"})


@pytest.mark.parametrize("phase", PHASES)
def test_a_point_that_recovers_on_its_one_restage_leaves_the_run_complete(tmp_path, phase):
    limits = _small_limits(16)
    cell = _cell(phase=phase, dp=2, strategy="tp")
    pod = _pod(tmp_path, cell)
    planned = grid.build_grid(phase, limits).points
    # The third point that expects cached tokens: a fake eviction costs it its staged state.
    victim = [point for point in planned if grid.expected_cached_tokens(phase, point.cached_tokens, 16)][2]

    run = _simulate_run(phase, limits, dp_size=2, fake_evictions={victim.benchmark_id: 1})

    assert run.restaged == [victim.benchmark_id]
    first = [entry for entry in run.observed if entry[0] == victim.benchmark_id and entry[2] == 0]
    second = [entry for entry in run.observed if entry[0] == victim.benchmark_id and entry[2] == 1]
    measured_stage = "meas" if phase == "prefill" else "run"
    # The first attempt's measured call found nothing; the re-stage found exactly the planned tokens.
    assert {entry[5] for entry in first if entry[1] == measured_stage} == {0}
    assert {entry[5] for entry in second} == {grid.expected_cached_tokens(phase, victim.cached_tokens, 16)}
    assert len(second) == len(first) == 2 * victim.batch_size * (2 if phase == "prefill" else 1)
    attempts = collections.Counter(content.parse_request_id(rid)["attempt"] for rid in run.request_ids)
    seeds = 1 if phase == "decode" or victim.cached_tokens else 0
    assert attempts[1] == 2 * victim.batch_size * (seeds + (2 if phase == "prefill" else 1))
    assert len(set(run.request_ids)) == len(run.request_ids)
    assert [payload["status"] for payload in run.payloads] == ["complete", "complete"]
    assert run.payloads[0]["producer"]["restaged_points"] == [victim.benchmark_id]

    written = artifact.write_artifacts(str(pod / "benchmark.json"), run.payloads)
    _assert_accepted(tmp_path, cell, run, written)


@pytest.mark.parametrize("phase", PHASES)
def test_a_capped_run_measures_the_lower_end_that_is_added_after_the_cap(tmp_path, phase):
    # Both graph ladders start above 1 and both graph-driven axes are cut by their cap. The lower end 1 is put
    # in front after the cut, so each of the two axes holds one value more than its cap, and the points at 1
    # are planned points like any other: staged, timed, accepted and published.
    limits = _small_limits(16, decode_graph_sizes=(4, 8))
    caps = grid.SampleCaps(prefill_max_new_token_samples=4, decode_max_batch_size_samples=3)
    cell = _cell(phase=phase, dp=2, strategy="tp")
    pod = _pod(tmp_path, cell)

    run = _simulate_run(phase, limits, dp_size=2, caps=caps)

    points = run.built.points
    if phase == "prefill":
        # Graph axis up to the largest n, C = 96: 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 96. The cap of 4 keeps
        # the two values above the largest graph size (65, 96) and both ends of the rest (4, 64).
        axis = sorted({point.new_tokens for point in points})
        assert axis == [1, 4, 64, 65, 96] and len(axis) == caps.prefill_max_new_token_samples + 1
        lower_end = [point for point in points if point.new_tokens == 1]
    else:
        # Graph axis up to the largest batch, R = 9: 4, 5, 8, 9. The cap of 3 keeps 4, 8 and 9.
        axis = sorted({point.batch_size for point in points})
        assert axis == [1, 4, 8, 9] and len(axis) == caps.decode_max_batch_size_samples + 1
        lower_end = [point for point in points if point.batch_size == 1]
    assert len(lower_end) > 3 and run.restaged == []
    assert [payload["status"] for payload in run.payloads] == ["complete", "complete"]
    written = artifact.write_artifacts(str(pod / "benchmark.json"), run.payloads)

    _assert_accepted(tmp_path, cell, run, written)
    rows = aggregate_cell(_plan(cell), cell, tmp_path, expected_attempt_id="attempt")
    assert [(row["batch_size"], row["total_prefill_tokens"], row["total_kv_read_tokens"]) for row in rows] == (
        _coordinates(run.built.point_dicts())
    )


# ---------------------------------------------------------------------------------------------------------------
# The same run, cut short or stopped: both checkers reject
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("page_size", PAGE_SIZES)
@pytest.mark.parametrize("phase", PHASES)
def test_run_cut_short_by_a_fake_deadline_is_partial_and_rejected(tmp_path, phase, page_size):
    limits = _small_limits(page_size)
    cell = _cell(phase=phase, dp=2, strategy="tp")
    pod = _pod(tmp_path, cell)
    completed = 7

    run = _simulate_run(phase, limits, dp_size=2, fake_deadline_after=completed)

    expected_points = len(run.built.points)
    assert expected_points > completed
    for payload in run.payloads:
        assert (payload["status"], payload["valid"], payload["usable"]) == ("partial", False, True)
        assert (payload["stop_reason"], payload["error"]) == ("timeout", None)
        assert payload["coverage"] == {
            "expected_points": expected_points,
            "completed_points": completed,
            "skipped_points": 0,
        }
        # The evidence of the points that did complete is kept, in order.
        assert [row["point"] for row in payload["results"]] == run.built.point_dicts()[:completed]
        assert payload["timing"]["benchmark_elapsed_seconds"] >= FAKE_TIMEOUT_SECONDS
    # Nothing was sent for a point after the deadline.
    assert {content.parse_request_id(rid)["benchmark_id"] for rid in run.request_ids} == set(range(completed + 1))
    written = artifact.write_artifacts(str(pod / "benchmark.json"), run.payloads)

    merged = json.loads(Path(written["merged_output_path"]).read_text(encoding="utf-8"))
    assert (merged["artifact_type"], merged["status"], merged["valid"]) == ("merged", "partial", False)
    # The gate's message says why: it prints the artifact's "errors".
    assert f"stopped by timeout after {completed} of {expected_points} points" in _assert_rejected(
        tmp_path, cell, written, "partial"
    )


def test_a_deadline_that_passes_after_the_last_point_leaves_the_run_complete(tmp_path):
    limits = _small_limits(16)
    cell = _cell(phase="decode", dp=1, strategy="single")
    pod = _pod(tmp_path, cell)
    every_point = len(grid.build_grid("decode", limits).points)

    run = _simulate_run("decode", limits, dp_size=1, fake_deadline_after=every_point)

    (payload,) = run.payloads
    assert (payload["status"], payload["stop_reason"]) == ("complete", None)
    assert payload["timing"]["benchmark_elapsed_seconds"] >= FAKE_TIMEOUT_SECONDS
    _assert_accepted(tmp_path, cell, run, artifact.write_artifacts(str(pod / "benchmark.json"), run.payloads))


@pytest.mark.parametrize("page_size", PAGE_SIZES)
@pytest.mark.parametrize("phase", PHASES)
def test_run_stopped_by_a_fake_failed_point_is_failed_and_rejected(tmp_path, phase, page_size):
    limits = _small_limits(page_size)
    cell = _cell(phase=phase, dp=2, strategy="tp")
    pod = _pod(tmp_path, cell)
    planned = grid.build_grid(phase, limits).points
    # The sixth point that expects cached tokens loses its staged state on both attempts.
    victim = [point for point in planned if grid.expected_cached_tokens(phase, point.cached_tokens, page_size)][5]
    completed = victim.benchmark_id - 1
    assert completed >= 5

    run = _simulate_run(phase, limits, dp_size=2, fake_evictions={victim.benchmark_id: 2})

    assert run.restaged == [victim.benchmark_id]
    message = f"benchmark_id={victim.benchmark_id}: benchmark point failed: {REASON_CACHED_TOKENS}"
    for payload in run.payloads:
        assert (payload["status"], payload["valid"], payload["usable"]) == ("failed", False, False)
        assert (payload["error"], payload["errors"], payload["stop_reason"]) == (message, [message], None)
        assert payload["skipped_points"] == [
            {"point": victim.as_dict(), "kv_seed_regime": _expected_regime(victim), "reason": REASON_CACHED_TOKENS}
        ]
        assert payload["coverage"] == {
            "expected_points": len(planned),
            "completed_points": completed,
            "skipped_points": 1,
        }
        assert [row["point"] for row in payload["results"]] == run.built.point_dicts()[:completed]
    # The failed point was tried twice, and nothing after it was sent.
    sent = collections.Counter(
        (parts["benchmark_id"], parts["attempt"]) for parts in map(content.parse_request_id, run.request_ids)
    )
    assert max(benchmark_id for benchmark_id, _attempt in sent) == victim.benchmark_id
    assert sent[(victim.benchmark_id, 0)] == sent[(victim.benchmark_id, 1)] > 0
    written = artifact.write_artifacts(str(pod / "benchmark.json"), run.payloads)

    assert written["merged_output_path"] is None
    assert sorted(path.name for path in pod.glob("benchmark*")) == ["benchmark.json", "benchmark_dp1.json"]
    assert message in _assert_rejected(tmp_path, cell, written, "failed")


@pytest.mark.parametrize("completed", [0, 5, None], ids=["before-the-first", "after-five", "after-the-last"])
@pytest.mark.parametrize(("dp_size", "strategy"), [(1, "single"), (2, "tp")], ids=["dp1-single", "dp2-tp"])
@pytest.mark.parametrize("phase", PHASES)
def test_run_stopped_by_a_fake_fault_outside_any_point_is_failed_and_rejected(
    tmp_path, phase, dp_size, strategy, completed
):
    # The third failed form: the failure names no point. It is written before the first point, between two
    # points, and after the last one, where the rank files hold every planned row and are rejected all the same.
    limits = _small_limits(16)
    cell = _cell(phase=phase, dp=dp_size, strategy=strategy)
    pod = _pod(tmp_path, cell)
    planned = grid.build_grid(phase, limits)
    if completed is None:
        completed = len(planned.points)
    assert len(planned.points) > 5

    run = _simulate_run(phase, limits, dp_size=dp_size, fake_fault_after=completed)

    assert run.built == planned and run.restaged == []
    for payload in run.payloads:
        assert (payload["status"], payload["valid"], payload["usable"]) == ("failed", False, False)
        # The reason is the error as it stands; no point is listed as skipped and no phase as missing.
        assert (payload["error"], payload["errors"]) == (FAKE_FAULT_REASON, [FAKE_FAULT_REASON])
        assert (payload["stop_reason"], payload["skipped_points"], payload["missing_phases"]) == (None, [], [])
        assert payload["coverage"] == {
            "expected_points": len(planned.points),
            "completed_points": completed,
            "skipped_points": 0,
        }
    # Nothing was sent for a point after the fault (benchmark id 0 is the global warm-up).
    assert {content.parse_request_id(rid)["benchmark_id"] for rid in run.request_ids} == set(range(completed + 1))
    written = artifact.write_artifacts(str(pod / "benchmark.json"), run.payloads)

    # A failed run has no merged file, and its rank files keep the rows of the points that completed.
    assert written["merged_output_path"] is None
    rank_file_names = ["benchmark.json", "benchmark_dp1.json"][:dp_size]
    assert sorted(path.name for path in pod.glob("benchmark*")) == rank_file_names
    assert [Path(path).name for path in written["rank_files"]] == rank_file_names
    completed_points = planned.point_dicts()[:completed]
    for rank, path in enumerate(written["rank_files"]):
        on_disk = json.loads(Path(path).read_text(encoding="utf-8"))
        assert on_disk == run.payloads[rank]
        assert [row["point"] for row in on_disk["results"]] == completed_points
        assert [row["fpms"][0]["wall_time"] for row in on_disk["results"]] == [
            _expected_sample_seconds(point, rank) for point in completed_points
        ]
        assert [group["benchmark_id"] for group in on_disk["iteration_groups"]] == list(range(1, completed + 1))
    # Both checkers reject it, and the gate's message carries the reason.
    assert FAKE_FAULT_REASON in _assert_rejected(tmp_path, cell, written, "failed")


@pytest.mark.parametrize("phase", PHASES)
def test_a_phase_without_a_feasible_point_fails_the_run_and_is_rejected(tmp_path, phase):
    # An input limit of one token admits no prompt at all; nothing is staged and nothing is measured.
    limits = _small_limits(16, max_req_input_len=1)
    cell = _cell(phase=phase, dp=1, strategy="single")
    pod = _pod(tmp_path, cell)

    run = _simulate_run(phase, limits, dp_size=1)

    assert run.built.points == () and run.built.missing_phases == (phase,)
    (payload,) = run.payloads
    message = f"no feasible benchmark points for phase(s): {phase}"
    assert (payload["status"], payload["error"], payload["missing_phases"]) == ("failed", message, [phase])
    assert payload["coverage"] == {"expected_points": 0, "completed_points": 0, "skipped_points": 0}
    assert len(run.request_ids) == 1  # the global warm-up only
    written = artifact.write_artifacts(str(pod / "benchmark.json"), run.payloads)

    assert written["merged_output_path"] is None
    assert message in _assert_rejected(tmp_path, cell, written, "failed")


@pytest.mark.parametrize("ending", ["deadline", "fault", "failed-point", "fault-before-first", "fault-after-last"])
@pytest.mark.parametrize("dp_size", [1, 2, 3])
@pytest.mark.parametrize("phase", [*PHASES, "agg"])
def test_a_rejected_artifact_keeps_the_evidence_of_its_completed_points_whole(tmp_path, phase, dp_size, ending):
    # Both checkers stop at the envelope of a partial or failed artifact, so nothing they do says what else
    # such a file carries. Here each rank's rejected artifact is compared with the same rank's artifact of the
    # complete run of the same grid: the rows and groups of the points that completed, with their FAKE times,
    # its own ending, its own timing and content counts, and nothing else different.
    limits = _small_limits(16)
    whole_run = _simulate_run(phase, limits, dp_size=dp_size)
    planned = whole_run.built.points
    records = whole_run.built.point_dicts()
    if ending == "deadline":
        completed = 7
        run = _simulate_run(phase, limits, dp_size=dp_size, fake_deadline_after=completed)
        ending_block = {
            "status": "partial",
            "usable": True,
            "stop_reason": "timeout",
            "error": None,
            "errors": [f"benchmark stopped by timeout after {completed} of {len(planned)} points"],
            "skipped_points": [],
        }
    elif ending.startswith("fault"):
        completed = {"fault": 7, "fault-before-first": 0, "fault-after-last": len(planned)}[ending]
        run = _simulate_run(phase, limits, dp_size=dp_size, fake_fault_after=completed)
        ending_block = {
            "status": "failed",
            "usable": False,
            "stop_reason": None,
            "error": FAKE_FAULT_REASON,
            "errors": [FAKE_FAULT_REASON],
            "skipped_points": [],
        }
    else:
        # The sixth point that expects cached tokens loses its staged state on both attempts.
        victim = [point for point in planned if grid.expected_cached_tokens(point.phase, point.cached_tokens, 16)][5]
        completed = victim.benchmark_id - 1
        run = _simulate_run(phase, limits, dp_size=dp_size, fake_evictions={victim.benchmark_id: 2})
        message = f"benchmark_id={victim.benchmark_id}: benchmark point failed: {REASON_CACHED_TOKENS}"
        ending_block = {
            "status": "failed",
            "usable": False,
            "stop_reason": None,
            "error": message,
            "errors": [message],
            "skipped_points": [
                {"point": victim.as_dict(), "kv_seed_regime": _expected_regime(victim), "reason": REASON_CACHED_TOKENS}
            ],
        }
    assert run.built == whole_run.built

    # The FAKE time of the group of each completed point is the slower rank's sample. Their sum, added left
    # to right, is the measured time, and the fake clock of the simulated driver gives the elapsed time: the
    # global warm-up, each completed point, and the jump past the deadline when that ended the run.
    fake_slowest = [
        max(_expected_sample_seconds(record, rank) for rank in range(dp_size)) for record in records[:completed]
    ]
    fake_measured_seconds = 0.0
    fake_clock = _FakeClock()
    fake_started = fake_clock.now
    fake_clock.advance(FAKE_POINT_OVERHEAD_SECONDS)
    for fake_seconds in fake_slowest:
        fake_measured_seconds += fake_seconds
        fake_clock.advance(FAKE_POINT_OVERHEAD_SECONDS + fake_seconds)
    if ending == "deadline":
        fake_clock.advance(FAKE_TIMEOUT_SECONDS)
    fake_timing = {
        "benchmark_elapsed_seconds": fake_clock.now - fake_started,
        "measured_iteration_seconds": fake_measured_seconds,
    }
    assert 0 <= fake_measured_seconds < fake_timing["benchmark_elapsed_seconds"]

    for rank, (whole, cut) in enumerate(zip(whole_run.payloads, run.payloads, strict=True)):
        assert (whole["status"], whole["valid"], cut["valid"]) == ("complete", True, False)
        # The rows and groups of the completed points are those of the complete run.
        assert cut["results"] == whole["results"][:completed]
        assert cut["iteration_groups"] == whole["iteration_groups"][:completed]
        # The same rows and groups against the FAKE inputs: a row holds this rank's fpm with its FAKE sample
        # (a decode fpm also the six FAKE steps), and a group holds the fpm of each rank's own row and the
        # slower rank's time.
        for index, record in enumerate(records[:completed]):
            row, group = cut["results"][index], cut["iteration_groups"][index]
            (fpm,) = row["fpms"]
            assert (row["point"], row["kv_seed_regime"]) == (record, _expected_regime(planned[index]))
            assert (fpm["dp_rank"], fpm["wall_time"]) == (rank, _expected_sample_seconds(record, rank))
            if record["point_type"] == "decode":
                assert fpm[artifact.DECODE_STEPS_KEY] == _fake_decode_step_seconds(record, rank)
            assert group["rank_results"] == [
                {"dp_rank": other, "fpms": payload["results"][index]["fpms"]}
                for other, payload in enumerate(run.payloads)
            ]
            assert (group["wall_time"], group["complete"], group["expected_dp_ranks"]) == (
                fake_slowest[index],
                True,
                list(range(dp_size)),
            )
        assert cut["timing"] == fake_timing
        # Everything else: the complete artifact with this ending, and with the content counts and re-staged
        # points of this run in "producer". Identity, config, dp, the measurement policy and, for decode,
        # the kvwarm block are the complete run's.
        assert cut == {
            **whole,
            **ending_block,
            "valid": False,
            "coverage": {
                "expected_points": len(planned),
                "completed_points": completed,
                "skipped_points": len(ending_block["skipped_points"]),
            },
            "results": whole["results"][:completed],
            "iteration_groups": whole["iteration_groups"][:completed],
            "timing": fake_timing,
            "producer": {
                **whole["producer"],
                "content": {**whole["producer"]["content"], "issued": run.content_stats},
                "restaged_points": run.restaged,
            },
        }
        assert ("kvwarm" in cut) == (phase != "prefill")
        assert cut["producer"]["timing_source"].startswith("FAKE")
    assert (run.content_stats != whole_run.content_stats) == (ending != "fault-after-last")

    # On disk: each rank file is its payload. A failed run has no merged file; a partial run has one, built
    # like that of a complete run: rank 0's artifact, the binding to the rank files, the dp block of the
    # merge and both ranks' fpms in every row.
    written = artifact.write_artifacts(str(tmp_path / "benchmark.json"), run.payloads)
    names = ["benchmark.json", "benchmark_dp1.json", "benchmark_dp2.json"][:dp_size]
    assert written["rank_files"] == [str(tmp_path / name) for name in names]
    for path, payload in zip(written["rank_files"], run.payloads, strict=True):
        assert json.loads(Path(path).read_text(encoding="utf-8")) == payload
    if ending != "deadline":
        assert written["merged_output_path"] is None
        assert sorted(path.name for path in tmp_path.iterdir()) == sorted(names)
        return
    assert written["merged_output_path"] == str(tmp_path / "benchmark_merged.json")
    merged = json.loads(Path(written["merged_output_path"]).read_text(encoding="utf-8"))
    assert merged == {
        **run.payloads[0],
        "artifact_type": "merged",
        "rank_files": written["rank_files"],
        "merged_output_path": written["merged_output_path"],
        "dp": {
            "size": dp_size,
            "ranks": list(range(dp_size)),
            "source_ranks": list(range(dp_size)),
            "managed_size": dp_size,
            "global_size": dp_size,
        },
        "results": [
            {**row, "fpms": [payload["results"][index]["fpms"][0] for payload in run.payloads]}
            for index, row in enumerate(run.payloads[0]["results"])
        ],
    }
    assert len(merged["results"]) == completed and all(len(row["fpms"]) == dp_size for row in merged["results"])


# ---------------------------------------------------------------------------------------------------------------
# Manifest-driven runs
# ---------------------------------------------------------------------------------------------------------------

# Page size 16 with I = 169, S = 1184, R = 9, M = 80, C = 96. Not sorted: file order is execution order.
MANIFEST = {
    "schema_version": 3,
    "prefill": [
        # Per request: n = 32, k = 32.
        {"batch_size": 2, "total_prefill_tokens": 64, "total_kv_read_tokens": 64},
        # n = 5, k = 0, with the two optional keys present and null.
        {"batch_size": 1, "total_prefill_tokens": 5, "total_kv_read_tokens": 0, "partition": None, "rows": None},
        # n = 4, k = 96, with rows that restate the even split.
        {"batch_size": 4, "total_prefill_tokens": 16, "total_kv_read_tokens": 384, "rows": [[4, 96]] * 4},
        # n = 96 = C, k = 64: the largest single request.
        {"batch_size": 1, "total_prefill_tokens": 96, "total_kv_read_tokens": 64},
        # n = 1, k = 16.
        {"batch_size": 3, "total_prefill_tokens": 3, "total_kv_read_tokens": 48},
    ],
    "decode": [
        # Per request: k = 4, the lowest context.
        {"batch_size": 2, "total_kv_read_tokens": 8},
        # k = 100, with the prefill total AISim's freezer lets through.
        {"batch_size": 8, "total_kv_read_tokens": 800, "total_prefill_tokens": 0},
        # k = 169 = I, the deepest context.
        {"batch_size": 1, "total_kv_read_tokens": 169},
        # k = 20.
        {"batch_size": 3, "total_kv_read_tokens": 60},
    ],
}


@pytest.mark.parametrize("phase", PHASES)
def test_manifest_driven_run_measures_exactly_the_requested_coordinates(tmp_path, phase):
    limits = _small_limits(16)
    cell = _cell(phase=phase, dp=2, strategy="dep")
    pod = _pod(tmp_path, cell)
    # The pod gets the text AISim's own freezer produced from the user's file, not the file itself.
    manifest_path = tmp_path / "points.json"
    manifest_path.write_text(json.dumps(MANIFEST, indent=2), encoding="utf-8")
    frozen, frozen_sha256 = _freeze_benchmark_points(str(manifest_path))
    manifest = grid.parse_points_manifest(frozen)

    run = _simulate_run(phase, limits, dp_size=2, manifest=manifest, config={"points_manifest_sha256": frozen_sha256})

    requested = [
        (point["batch_size"], point.get("total_prefill_tokens", 0), point["total_kv_read_tokens"])
        for point in MANIFEST[phase]
    ]
    records = run.built.point_dicts()
    assert run.built.source == "manifest" and run.built.missing_phases == ()
    # File order is execution order, ids are 1..N, and every point is marked explicit before its regime stamp.
    assert _coordinates(records) == requested
    assert [record["benchmark_id"] for record in records] == list(range(1, len(requested) + 1))
    stamps = {"real_kv": ["kvwarm_real_kv"], "real_prefix": ["prefill_real_seed"], "not_applicable": []}
    assert [record["sample_reasons"] for record in records] == [
        ["explicit", *stamps[_expected_regime(point)]] for point in run.built.points
    ]
    assert run.restaged == []
    for benchmark_id, _stage, _attempt, _rank, _slot, cached, extend in run.observed:
        point = run.built.points[benchmark_id - 1]
        assert cached == grid.expected_cached_tokens(phase, point.cached_tokens, 16)
        assert extend == grid.expected_extend_tokens(phase, point.new_tokens, point.cached_tokens, 16)
    assert [payload["status"] for payload in run.payloads] == ["complete", "complete"]
    assert run.payloads[0]["config"] == {
        "mode": phase,
        "points_source": "manifest",
        "content_seed": FAKE_CONTENT_SEED,
        "points_manifest_sha256": frozen_sha256,
    }
    written = artifact.write_artifacts(str(pod / "benchmark.json"), run.payloads)

    _assert_accepted(tmp_path, cell, run, written)
    # Publication covers the requested coordinates exactly: nothing clamped, relabelled or dropped.
    rows = aggregate_cell(_plan(cell), cell, tmp_path, expected_attempt_id="attempt")
    assert [(row["batch_size"], row["total_prefill_tokens"], row["total_kv_read_tokens"]) for row in rows] == requested


# Page size 1, where every token count is a whole number of pages, with I = 169, S = 1184, R = 9, M = 80, C = 96.
# The first point of each phase is the smallest cached state there is: one cached token per request.
MANIFEST_PAGE_1 = {
    "schema_version": 1,
    "prefill": [
        # One request, n = 4, k = 1: the seed is a single token.
        {"batch_size": 1, "total_prefill_tokens": 4, "total_kv_read_tokens": 1},
        # The same request without a prefix.
        {"batch_size": 1, "total_prefill_tokens": 4, "total_kv_read_tokens": 0},
        # Per request: n = 2, k = 1.
        {"batch_size": 3, "total_prefill_tokens": 6, "total_kv_read_tokens": 3},
        # n = 1, k = 7: a prefix that is no multiple of 16.
        {"batch_size": 2, "total_prefill_tokens": 2, "total_kv_read_tokens": 14},
    ],
    "decode": [
        # One request, k = 5: the prompt has two tokens, one is found cached and one is computed.
        {"batch_size": 1, "total_kv_read_tokens": 5},
        # k = 4, the lowest context: a prompt of one token, nothing cached.
        {"batch_size": 1, "total_kv_read_tokens": 4},
        # k = 6 for each of three requests: two cached tokens, one computed.
        {"batch_size": 3, "total_kv_read_tokens": 18},
    ],
}


@pytest.mark.parametrize("phase", PHASES)
def test_manifest_driven_run_at_page_size_one_stages_a_single_cached_token(tmp_path, phase):
    limits = _small_limits(1)
    cell = _cell(phase=phase, dp=2, strategy="dep")
    pod = _pod(tmp_path, cell)
    manifest = grid.parse_points_manifest(json.dumps(MANIFEST_PAGE_1))

    run = _simulate_run(phase, limits, dp_size=2, manifest=manifest)

    points = run.built.points
    records = run.built.point_dicts()
    requested = [
        (point["batch_size"], point.get("total_prefill_tokens", 0), point["total_kv_read_tokens"])
        for point in MANIFEST_PAGE_1[phase]
    ]
    assert _coordinates(records) == requested and run.restaged == []
    # What the fake engine reported for the timed requests of the first point, on both ranks:
    # (stage, attempt, rank, slot, cached tokens, extend tokens).
    first_point = [entry[1:] for entry in run.observed if entry[0] == 1]
    if phase == "prefill":
        assert (points[0].batch_size, points[0].new_tokens, points[0].cached_tokens) == (1, 4, 1)
        # One cached token is a cached prefix: the point carries the real-seed stamp, and the same request
        # without a prefix carries none.
        assert records[0]["sample_reasons"] == ["explicit", "prefill_real_seed"]
        assert records[1]["sample_reasons"] == ["explicit"]
        assert [artifact.kv_seed_regime(record) for record in records[:2]] == ["real_prefix", "not_applicable"]
        # The warm shot and the measured shot each found exactly the one seeded token and computed four.
        assert first_point == [(stage, 0, rank, 0, 1, 4) for rank in (0, 1) for stage in ("warm", "meas")]
    else:
        assert (points[0].batch_size, points[0].cached_tokens) == (1, 5)
        assert records[0]["sample_reasons"] == ["explicit", "kvwarm_real_kv"]
        # The decode run found one of its two prompt tokens cached and computed the other.
        assert first_point == [("run", 0, rank, 0, 1, 1) for rank in (0, 1)]
    for benchmark_id, _stage, _attempt, _rank, _slot, cached, extend in run.observed:
        point = points[benchmark_id - 1]
        assert cached == grid.expected_cached_tokens(phase, point.cached_tokens, 1)
        assert extend == grid.expected_extend_tokens(phase, point.new_tokens, point.cached_tokens, 1)
    assert [payload["status"] for payload in run.payloads] == ["complete", "complete"]
    written = artifact.write_artifacts(str(pod / "benchmark.json"), run.payloads)

    # Both checkers accept. The validator returns each point with its regime: "real_prefix" for the one-token
    # prefix, "real_kv" for a decode point.
    _assert_accepted(tmp_path, cell, run, written)
    rows = aggregate_cell(_plan(cell), cell, tmp_path, expected_attempt_id="attempt")
    assert [(row["batch_size"], row["total_prefill_tokens"], row["total_kv_read_tokens"]) for row in rows] == requested


@pytest.mark.parametrize(
    ("phase", "unstageable", "reason"),
    [
        # 8 cached tokens per request are half a page of 16: SGLang caches whole pages only.
        ("prefill", {"batch_size": 1, "total_prefill_tokens": 8, "total_kv_read_tokens": 8}, "cached_not_page_aligned"),
        # 3 tokens of context per request: the prompt of k - 3 tokens would be empty.
        ("decode", {"batch_size": 2, "total_kv_read_tokens": 6}, "context_below_minimum"),
    ],
)
def test_a_manifest_point_that_cannot_be_staged_stops_the_run_at_grid_build(phase, unstageable, reason):
    limits = _small_limits(16)
    points = {"prefill": [], "decode": [], phase: [MANIFEST[phase][0], unstageable]}
    manifest = grid.parse_points_manifest(json.dumps({"schema_version": 1, **points}))  # the schema allows it

    # The point is named with its place in the file and nothing is planned, sent or relabelled.
    with pytest.raises(ValueError, match=rf"{phase}\[1\]: explicit benchmark point is infeasible: reason={reason} "):
        _simulate_run(phase, limits, dp_size=1, manifest=manifest)

    if phase == "prefill":
        # Why it is refused: the engine cannot hold that state. A seed of 8 tokens stores no whole page, so
        # the shot would find 0 cached tokens, never the 8 the point asks for.
        plan = content.ContentPlan(FAKE_CONTENT_SEED, TOKEN_LOW, TOKEN_HIGH)
        cache = PageCacheModel(16)
        cache.finish_prefill(plan.prefill_seed_prompt(0, 0, 8))
        assert cache.cached(plan.prefill_prompt(0, 0, 8, 8)) == 0


# ---------------------------------------------------------------------------------------------------------------
# agg: the prefill block, then the decode block
# ---------------------------------------------------------------------------------------------------------------


def test_agg_run_stages_prefill_before_decode_and_passes_the_gate_in_agg_mode(tmp_path):
    # Page size 1 is where a prefill shot after a decode run on the same slot could find too much cached; the
    # content plan refuses that order. The agg order never asks for it.
    limits = _small_limits(1)

    run = _simulate_run("agg", limits, dp_size=2)

    phases = [point.phase for point in run.built.points]
    prefill_points = phases.count("prefill")
    assert prefill_points > 40 and phases.count("decode") > 40
    assert phases == ["prefill"] * prefill_points + ["decode"] * (len(phases) - prefill_points)
    assert run.restaged == []
    for benchmark_id, _stage, _attempt, _rank, _slot, cached, _extend in run.observed:
        point = run.built.points[benchmark_id - 1]
        assert cached == grid.expected_cached_tokens(point.phase, point.cached_tokens, 1)
    assert [payload["status"] for payload in run.payloads] == ["complete", "complete"]
    assert "kvwarm" in run.payloads[0]
    # The other order is refused by the plan, not risked: a prefill shot whose prefix ends where a decode run
    # of the same slot left model-generated tokens in the cache.
    decode_point = run.built.points[prefill_points]
    decode_prompt_tokens = decode_point.cached_tokens - grid.DECODE_PROMPT_LEAD
    with pytest.raises(ValueError, match="follows a decode run"):
        run.plan.prefill_prompt(0, 0, decode_prompt_tokens, 4)
    written = artifact.write_artifacts(str(tmp_path / "benchmark.json"), run.payloads)

    gate = _gate(written["rank_files"], "agg")
    assert gate.returncode == GATE_ACCEPT, gate.stderr
    # A gate started for one phase refuses the file (AISim launches one phase per cell).
    assert _gate(written["rank_files"], "prefill").returncode == GATE_INVALID


# ---------------------------------------------------------------------------------------------------------------
# Publication: the planned grid as a table in AISim's compiled reader
# ---------------------------------------------------------------------------------------------------------------

# Where the FAKE table is published and how the reader is asked for it. The system file and the model's packaged
# config are the reader's own inputs; nothing was measured on that system or model here. The version is a label
# for the table directory and names no SGLang release.
READER_SYSTEM = "h200_sxm"
READER_BACKEND = "sglang"
READER_MODEL = "Qwen/Qwen3-0.6B"
FAKE_READER_VERSION = "fake-sglbench-reader-test"


@pytest.mark.parametrize("h200", [H200_PAGE_1, H200_PAGE_64], ids=["page-1", "page-64"])
def test_default_h200_grid_published_with_fake_times_is_served_by_the_compiled_reader(tmp_path, h200):
    """Plan the default grid, write it with FAKE times, publish it with AISim's own Python path, and read it back.

    The path is ``aggregate_cell`` and ``write_formal_database`` (collector/fpm_forward/database.py), then
    ``PerfDatabase`` and ``EngineHandle.for_database``, loaded as tests/unit/sdk/test_fpm_forward.py loads them.
    This is the one test that fails when an axis rule plans a table the reader cannot serve. It says nothing
    about how close an interpolated answer is to an engine: the FAKE times have no shape.
    """
    # H200_PAGE_1 and H200_PAGE_64 hold the six limits in the recorder headers of two runs of the engine probe,
    # benchmarks/evidence/collector/sglbench-engine-probe-20261113 (headers of v0522-q06-p1 and v0522-q06-p64):
    # SGLang v0.5.22, Qwen3-0.6B, TP 1, page size 1 and 64. No result file records the GPU model; the README of
    # that folder gives it from the operator's record. The graph sizes were not read off an engine. They are
    # what generate_decode_cuda_graph_batch_sizes (graph padding not disabled, no speculative algorithm) and
    # generate_prefill_cuda_graph_batch_sizes return for a maximum of 256 and of 8192 in SGLang v0.5.22
    # (baa9befa381511cd4d56f83980951e5a45fffc6d) and v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90),
    # python/sglang/srt/arg_groups/cuda_graph_hook.py.
    limits = dataclasses.replace(
        h200,
        decode_graph_sizes=_ladder(H200_DECODE_GRAPH_ROWS),
        prefill_graph_sizes=_ladder(H200_PREFILL_GRAPH_ROWS),
    )
    systems_root = tmp_path / "systems"
    systems_root.mkdir()
    shutil.copy(Path(aisim_core.__file__).parent / "systems" / f"{READER_SYSTEM}.yaml", systems_root)
    # The reader matches a table row on the identity of the model's own op, so the cells are given that identity.
    model_config = sdk_config.ModelConfig(
        tp_size=1,
        pp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        forward_model="fpm",
    )
    model = models.get_model(READER_MODEL, model_config, READER_BACKEND)
    reader_ops = {"prefill": model.context_ops[0], "decode": model.generation_ops[0]}
    identity = dict(zip(_CELL_MATCH_COLUMNS, reader_ops["prefill"]._match_identity, strict=True))
    cells = {
        phase: dataclasses.replace(
            _cell(phase=phase, dp=1, strategy="single"),
            weight_quantization=identity["gemm_quant_mode"],
            kv_cache_dtype=identity["kv_cache_dtype"],
            gemm_quant_mode=identity["gemm_quant_mode"],
            moe_quant_mode=identity["moe_quant_mode"],
            fmha_quant_mode=identity["fmha_quant_mode"],
            comm_quant_mode=identity["comm_quant_mode"],
        )
        for phase in PHASES
    }
    # Only what aggregate_cell and write_formal_database read from a collection plan. "plan-sha" is what the
    # provenance file of _pod says; the version labels the rows instead of that file's fixture value.
    plan = SimpleNamespace(
        sha256="plan-sha",
        model_path=model.model_path,
        system=READER_SYSTEM,
        backend=READER_BACKEND,
        backend_version=FAKE_READER_VERSION,
        cells=tuple(cells.values()),
        options=SimpleNamespace(warmup_iterations=0),
        capability=SimpleNamespace(
            support_level="native", template_id=None, template_version=None, aic_database_version="test"
        ),
        aic_revision="test-revision",
    )

    # Plan each phase, write its rank artifact with FAKE times, and collect the rows AISim publishes for it.
    grids = {}
    fake_ms = {}  # (phase, benchmark id) -> the FAKE time in the published unit
    rows = []
    for phase, cell in cells.items():
        built = grid.build_grid(phase, limits)
        assert built.missing_phases == () and built.source == "generated"
        grids[phase] = built
        cell_dir = tmp_path / "cells" / cell.cell_id
        pod = _pod(cell_dir, cell)
        measurements = []
        fake_elapsed = 0.0
        for record in built.point_dicts():
            fake_seconds = _fake_reader_seconds(record)
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
            producer={"timing_source": "FAKE: typed into test_fpm_sglbench_producer.py, never measured"},
        )
        artifact.write_artifacts(str(pod / "benchmark.json"), [payload])
        cell_rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
        # One row per planned point, in order, each with its own FAKE time: no two rows of a phase are equal.
        published = [row["latency_ms"] for row in cell_rows]
        assert published == [fake_ms[phase, point.benchmark_id] for point in built.points]
        assert len(set(published)) == len(built.points)
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

    def reader_ms(phase: str, mode: str, batch: int, per_request: int, cached_per_request: int = 0) -> float:
        """Ask the reader for one forward pass; it raises when it cannot answer.

        ``per_request`` is the new tokens per request (prefill) or the context per request (decode); the
        reader multiplies by the batch itself (crates/core/src/perfmodel/operators/fpm_forward.rs
        ``FpmForwardOp::query``).
        """
        answer = handle.evaluate_ops_json(
            op_specs[phase][mode],
            is_context=phase == "prefill",
            batch_size=batch,
            s=per_request,
            prefix=cached_per_request,
        )
        return answer[0][1]

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
    # prefill graph size, 4. Of the three only 1 was collected: the lower end the grid puts on the axis.
    assert min(limits.prefill_graph_sizes) == 4 and new_tokens & {1, 2, 3} == {1}
    for mode in op_specs["prefill"]:
        for new in (1, 2, 3):
            answered["prefill", mode, new] = reader_ms("prefill", mode, 1, new)
    assert len(answered) == 10 and all(math.isfinite(value) for value in answered.values()), answered


# ---------------------------------------------------------------------------------------------------------------
# The pod: all three modules by flat name, standard library only
# ---------------------------------------------------------------------------------------------------------------

# Runs with "-I -S": no environment, no site-packages, no AISim. argv: runtime directory, limits record as
# JSON, mode, output path. Its time values are FAKE.
_POD_SCRIPT = """
import json
import sys

sys.path.insert(0, sys.argv[1])
names = ("sglbench_grid", "sglbench_content", "sglbench_artifact")
modules = []
for name in names:
    # Each module is imported alone: it leaves sys.modules again before the next one comes in, so a sibling
    # that its import loads is seen whichever of the three it is.
    modules.append(__import__(name))
    siblings = sorted(m for m in sys.modules if m.startswith("sglbench_") and m != name)
    assert not siblings, (name, siblings)  # each module is a leaf
    del sys.modules[name]
# With the folder off sys.path as well, none of the three can be imported from here on. The loop cannot see a
# sibling import that runs only inside a call; in a call made below it now raises.
sys.path.remove(sys.argv[1])
forbidden = {"sglang", "dynamo", "aisim", "aisim_core", "collector", "torch", "numpy"}
loaded = sorted(m for m in sys.modules if m.split(".")[0] in forbidden)
assert not loaded, loaded
third_party = sorted(
    m for m, module in sys.modules.items() if "site-packages" in (getattr(module, "__file__", None) or "")
)
assert not third_party, third_party
grid, content, artifact = modules

limits = grid.Limits.from_mapping(json.loads(sys.argv[2]))
mode = sys.argv[3]
built = grid.build_grid(mode, limits)
plan = content.ContentPlan("0", 1000, 51000)
measurements = []
for point in built.points:
    record = point.as_dict()
    for slot in range(point.batch_size):
        if point.phase == grid.PREFILL:
            prompt = plan.prefill_prompt(0, slot, point.cached_tokens, point.new_tokens)
            assert len(prompt) == point.cached_tokens + point.new_tokens
        else:
            prompt = plan.decode_prompt(0, slot, point.cached_tokens)
            assert len(prompt) == point.cached_tokens - grid.DECODE_PROMPT_LEAD
        rid = content.request_id("fake-pod-run", point.benchmark_id, "meas", 0, 0, slot)
        assert content.parse_request_id(rid)["slot"] == slot
    fake_seconds = 0.001 * point.benchmark_id
    if point.phase == grid.PREFILL:
        fpm = artifact.build_fpm(record, 0, fake_seconds)
    else:
        # A decode fpm must carry its six raw step times, and its wall_time must be their sample.
        fake_steps = [fake_seconds] * artifact.DECODE_RAW_STEPS
        fake_sample = artifact.decode_wall_time(fake_steps)
        fpm = artifact.build_fpm(record, 0, fake_sample, extra={artifact.DECODE_STEPS_KEY: fake_steps})
    measurements.append(artifact.build_measurement(record, {0: fpm}))
fake_elapsed_seconds = 1.0e6  # above the sum of the FAKE point times, which an artifact requires
payload = artifact.build_rank_artifact(
    dp_rank=0,
    dp_size=1,
    mode=mode,
    run_id="fake-pod-run",
    grid_digest=built.digest,
    expected_points=len(built.points),
    measurements=measurements,
    benchmark_elapsed_seconds=fake_elapsed_seconds,
    producer={"timing_source": "FAKE", "limits": limits.as_dict(), "content": plan.stats()},
)
written = artifact.write_artifacts(sys.argv[4], [payload])
print(json.dumps({"digest": built.digest, "points": len(built.points), "written": written}))
"""


@pytest.mark.parametrize("phase", PHASES)
def test_pod_imports_all_three_modules_with_the_standard_library_only_and_its_files_pass(tmp_path, phase):
    """The pod has no AISim: the modules are loaded by flat name, and what they write there must be accepted here."""
    limits = _small_limits(64)
    cell = _cell(phase=phase, dp=1, strategy="single")
    pod = _pod(tmp_path, cell)
    output_path = str(pod / "benchmark.json")

    done = subprocess.run(
        [sys.executable, "-I", "-S", "-c", _POD_SCRIPT, str(RUNTIME), json.dumps(limits.as_dict()), phase, output_path],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert done.returncode == 0, done.stderr
    summary = json.loads(done.stdout)
    # The bare interpreter planned the same grid as this process.
    built = grid.build_grid(phase, limits)
    assert (summary["digest"], summary["points"]) == (built.digest, len(built.points))
    assert summary["written"] == {
        "rank_files": [output_path],
        "merged_output_path": str(pod / "benchmark_merged.json"),
    }
    collection = _validate(cell, tmp_path)
    assert [measurement.point for measurement in collection.points] == built.point_dicts()
    assert collection.runtime_grid_digest == built.digest
    gate = _gate([output_path], phase)
    assert gate.returncode == GATE_ACCEPT, gate.stderr
