# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# The result envelope, its status rule, the point checks, the decode sample and the merged file are re-implemented
# from ai-dynamo/dynamo (Apache-2.0, https://github.com/ai-dynamo/dynamo), components/src/dynamo/vllm/
# instrumented_scheduler.py, benchmark_points.py and worker_factory.py and components/src/dynamo/common/
# forward_pass_metrics.py at b20fe6fe94021b854597965c7305931175e8f482. See THIRD_PARTY_NOTICES.md.

"""Build and write the result files of AISim's SGLang self-benchmark.

This module runs inside the SGLang pod, where AISim is not installed: standard library only, and no import of
another ``sglbench_*`` module. Points, samples and artifacts are plain dicts, lists, ints, floats and strings.

Files
    One rank artifact per attention-DP rank and one merged file, in the schema-version-2 envelope that AISim
    already reads from Dynamo's vLLM self-benchmark. Two AISim checkers decide whether a run counts:

    - the gate: the Python text inside ``check_result_files()`` of ``collector/fpm_forward/runtime/fpm_exec.sh``.
      It runs in the pod, polls the files, and fails the pod at the first readable rank file that is not
      ``complete``. It is the only reader that uses the content of the merged file.
    - the validator: ``validate_native_collection`` in ``collector/fpm_forward/native_artifact.py``. It runs on
      the collector host on the copied files and validates rank files only.

    The host readers that sweep ``benchmark*.json`` (``_rank_artifacts`` in native_artifact.py, which the
    validator calls, and ``_runtime_timing_summary`` in ``collector/fpm_forward/runner.py``) parse the merged
    file as JSON, must find a JSON object, and then skip it.

    A decode or agg artifact carries a ``kvwarm`` block, which says how the decode contexts were produced:
    the four keys ``enabled``, ``warm_eligible``, ``skip_reason`` and ``method`` with fixed values. When the
    prompts of the run were dataset text, the caller hands over the record of that dataset and the block
    carries it as a fifth key, ``dataset`` (``path``, ``sha256``, ``collection_pool``, ``conversations``).
    Every rank file of a run and its merged file carry the same block. A prefill artifact has none.

Statuses
    ``complete``  every planned point was measured. Both checkers accept it for the modes ``prefill`` and
                  ``decode``.
    ``partial``   the soft deadline stopped the run between two points. Both checkers reject it.
    ``failed``    a point failed, a requested phase has no feasible point, or the run failed outside any point.
                  Both checkers reject it, and no merged file is written.

    The rejected forms are still written, because a failure is a recorded observation (which point, which
    reason) and never a silently dropped point.

    An ``agg`` artifact (prefill points, then decode points) is for standalone runs only. The gate accepts it
    when it was started with mode ``agg``. ``validate_native_collection`` rejects it for every cell, because
    ``config.mode`` must equal the cell's workload kind and AISim launches one phase per cell.

Numbers
    - A point carries batch TOTALS: ``total_prefill_tokens`` is the new tokens summed over the requests of the
      batch; ``total_kv_read_tokens`` is the cached tokens (prefill) or the past tokens read (decode) summed
      the same way.
    - ``wall_time`` is seconds on the monotonic clock of the scheduler process, one float per point and rank.
      The same float goes into the row, the group and the sum. Nothing here rounds it or does arithmetic on it
      apart from that sum.

Checked against
    - AISim: ``collector/fpm_forward/native_artifact.py`` (``validate_native_collection``,
      ``_expected_scheduled``, ``_validate_fpm``, ``_validate_seed_regime``, ``_validate_kvwarm_contract``,
      ``_rank_artifacts``, ``select_native_measurements``); ``collector/fpm_forward/runner.py``
      (``_runtime_timing_summary``); ``collector/fpm_forward/runtime/fpm_exec.sh`` (``check_result_files``,
      ``benchmark_path_for_dp_rank``); ``src/aisim/fpm_contract.py`` (``fpm_benchmark_result_name``,
      ``FPM_NATIVE_BENCHMARK_RESULT_SCHEMA_VERSION``, ``FPM_BENCHMARK_RESULT_GLOB``);
      ``collector/fpm_forward/database.py`` (``aggregate_cell``);
      ``collector/fpm_forward/measurement_evidence.py`` (``extract_measurement_evidence``).
    - SGLang v0.5.22 (baa9befa381511cd4d56f83980951e5a45fffc6d) and
      v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90) (``sgl-project/sglang``, Apache-2.0):
      ``python/sglang/srt/managers/scheduler.py``, ``Scheduler.run_batch`` (sets ``batch.launch_ts`` from
      ``time.monotonic()``, the clock and unit of ``wall_time``) and ``Scheduler.process_batch_result`` with
      ``GenerationBatchResult.copy_done`` in ``python/sglang/srt/managers/utils.py`` (the end of a timed step).
    - Dynamo's vLLM self-benchmark, in the files and at the revision the header of this file names. The key
      names of the envelope are its names, as the AISim readers above expect them, with five exceptions that
      it does not write: the top-level keys ``errors`` and ``producer``, ``method`` in the ``kvwarm`` block,
      ``size`` in the ``dp`` block of the merged file, and the fpm key ``sglbench_decode_steps``. The
      ``dataset`` record of the ``kvwarm`` block has the four keys Dynamo writes there
      (``_kvwarm_load_texts``); Dynamo fills them from the file it loaded, here the caller supplies them.

No upstream function or block of source is copied. What follows SGLang and Dynamo is named in the header of this
file and recorded in THIRD_PARTY_NOTICES.md. For this module that is Dynamo alone; SGLang is named in comments and
docstrings only.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import secrets
from pathlib import PurePosixPath
from typing import Any

# ---------------------------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------------------------

# The only envelope version both checkers accept. The test pins it to
# aisim.fpm_contract.FPM_NATIVE_BENCHMARK_RESULT_SCHEMA_VERSION, which the pod cannot import.
SCHEMA_VERSION = 2

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
STOP_REASON_TIMEOUT = "timeout"

PREFILL = "prefill"
DECODE = "decode"
AGG = "agg"
MODES = (PREFILL, DECODE, AGG)

# Point types a run of each mode may contain (the gate applies the same rule to every result row, and the
# validator requires point_type == the cell's phase).
_MODE_POINT_TYPES = {PREFILL: (PREFILL,), DECODE: (DECODE,), AGG: (PREFILL, DECODE)}

# A decode run asks for seven output tokens. SGLang then forms extend batches for the prompts (one, when the
# requests fit one extend batch), which yield the first token, and six decode batches (one token each). The
# engine probe recorded that sequence: benchmarks/evidence/collector/sglbench-engine-probe-20261113 (check_4),
# 20 cases of 2 and of 8 requests, each one extend batch and then six decode batches. The six decode step
# times are the raw sample; a step time is the interval from the previous batch's output to this batch's
# output on the scheduler's clock.
DECODE_RAW_STEPS = 6

# Deviation note (collector rule "measure what serving runs", layer_permissions.md "Deviations are notes").
# What differs: the decode sample is the upper median of decode steps 3 to 5 of the six (zero-based indices
#   2, 3, 4), with step 4 reading exactly the point's context. Dynamo's vLLM producer keeps the second step of
#   a decode point, the step after the one that admits the requests. With its KV warm-up on, on real-KV
#   points and on fake-injected points whose total KV is at or above a threshold, it runs up to three such
#   steps (its default count) and keeps their upper median (instrumented_scheduler.py at the revision in the
#   header of this file, _bench_step_decode and _bench_save_current_point). A serving decode step is any step.
# Why: in the engine probe decode steps 1, 2 and 6 were not steady in part of the cases; the figures follow
#   under "Direction of the error". Overlap scheduling was on in every probe run (enable_overlap in the
#   recorder headers) and no run had it off, so the probe does not show whether it is the cause. AISim's
#   existing SGLang producer measures decode step 4 and names a full overlap pipeline as its reason
#   (sglang_native_fpm.py, DECODE_LEAD).
# Direction of the error: the dropped steps are not steady in either direction, so keeping them would bias the
#   sample by an amount and a sign that depend on the model. Figures of the engine probe (check_4: SGLang
#   v0.5.22 and v0.5.23, page sizes 1 and 64, decode runs of 2 and of 8 requests at context 1024; Qwen3-8B,
#   12 cases, 4 of them at TP 2, and Qwen3-0.6B, 8 cases at TP 1; per step the median over the repeats,
#   compared with step 5, the last kept step; "the sample" is the median over the repeats of the upper
#   median of steps 3 to 5; the ends of every range are rounded outward):
#   - step 1 was 6.1 to 10.6 percent shorter in 10 of the 12 8B cases and within 0.2 percent in the other two;
#     it was 30 to 71 percent LONGER in all 8 0.6B cases;
#   - step 2 was within 0.5 percent on the 8B model and 9 to 19 percent shorter in 6 of the 8 0.6B cases;
#   - step 6 was within 0.3 percent on the 8B model and 3.0 and 24.7 percent shorter in 2 of the 8 0.6B cases.
#     In the decode runs of 384 requests on the 0.6B model (check_9: page size 64, 5 runs at each tag) the
#     last of the five recorded intervals between decode outputs was 1.8 to 2.6 ms against 17.1 to 20.2 ms
#     for the four before it.
#   The kept steps are not always steady either. On the 8B model steps 3 and 4 were within 0.5 percent of
#   step 5 and the sample within 0.3 percent. On the 0.6B model kept step 3 was 6 to 15 percent shorter in 6
#   of the 8 cases. The upper median of three absorbs one short step: in 4 of those 6 cases the sample was
#   within 0.5 percent of step 5, as it was in the 2 cases without a short step 3. In the other 2 of the 6
#   step 4 was short as well and the sample was low: 2.1 percent low in one and 6.9 percent low in the other
#   (v0.5.22, page size 64, batch 2: steps 2.015, 1.401, 1.419, 1.442, 1.549 and 1.551 ms, sample 1.442 ms).
#   In no case was the sample more than 0.3 percent high.
# Where it is recorded: MEASUREMENT_POLICY below (in every artifact) and the six raw step times of every
#   decode fpm under the key DECODE_STEPS_KEY. The key is required, and its values are checked against the
#   fpm's wall_time (_check_decode_steps), in three places: in build_fpm, for the fpm it builds; in
#   build_measurement and build_rank_artifact, for every fpm they are given; in build_merged_artifact and
#   write_artifacts, for every fpm in the groups and in the rows of a complete rank artifact
#   (_check_complete_payload). A partial or failed rank artifact is not checked again in that third place,
#   and atomic_write_json does not look at fpms.
# How to retire: if a later SGLang release makes all six steps steady, widen the kept indices here.
DECODE_KEPT_STEP_INDICES = (2, 3, 4)

# The fpm key under which a decode fpm carries the DECODE_RAW_STEPS raw step times its wall_time was taken from.
# No AISim reader parses it: both checkers ignore unknown fpm keys.
DECODE_STEPS_KEY = "sglbench_decode_steps"

# Free-text names of the per-phase sample rule, written into every artifact. No AISim reader parses them.
# "single_extend_batch": a prefill sample is the one extend batch of the measured shot, timed from
# Scheduler.run_batch (batch.launch_ts) to the batch result being on the host.
MEASUREMENT_POLICY = {"prefill": "single_extend_batch", "decode": "upper_median_of_decode_steps_3_to_5"}

# Markers a point carries in "sample_reasons" to say how its cached state was produced. The validator knows four
# (native_artifact.py, _validate_seed_regime) and ties each to one row-level "kv_seed_regime" value. This
# producer only ever uses real state, so it emits the two real ones and refuses the two fake ones.
STAMP_PREFILL_REAL_SEED = "prefill_real_seed"  # cached prefill: the prefix was computed by a real seed request
STAMP_DECODE_REAL_KV = "kvwarm_real_kv"  # decode: the context was computed by a real seed request
REGIME_STAMPS = frozenset(
    {STAMP_PREFILL_REAL_SEED, "prefill_fake_prefix", STAMP_DECODE_REAL_KV, "kvwarm_fake_fallback"}
)

REGIME_NOT_APPLICABLE = "not_applicable"  # prefill with no cached tokens: nothing was seeded
REGIME_REAL_PREFIX = "real_prefix"  # prefill with cached tokens
REGIME_REAL_KV = "real_kv"  # decode

PRODUCER_NAME = "aisim_sglbench"
ARTIFACT_FORMAT = 1

# The four fixed keys of the "kvwarm" block of every decode and agg artifact. Why the block is written is said
# where build_rank_artifact writes it; _check_complete_payload requires it before such an artifact is written
# as complete.
_KVWARM_BLOCK = {"enabled": True, "warm_eligible": True, "skip_reason": None, "method": "radix_cache_seed"}

# The one key a kvwarm block may carry beside the fixed four, and the keys of the record under it, in the
# order they are written. The record names the dataset whose text the prompts of the run were made of: the
# local path of the file, the SHA-256 of the file, the name of the part of its conversations that was used,
# and how many conversations that part holds. These are the keys of the record that Dynamo's vLLM producer
# writes under the same key (_kvwarm_load_texts). No AISim reader parses the record: the validator reads
# three of the fixed keys and carries the rest of the block along (_validate_kvwarm_contract).
_KVWARM_DATASET_KEY = "dataset"
_KVWARM_DATASET_FIELDS = ("path", "sha256", "collection_pool", "conversations")
_SHA256_HEX_DIGITS = 64
_LOWER_HEX_DIGITS = frozenset("0123456789abcdef")

# Top-level key names that already have an AISim reader with vLLM-shaped expectations
# (measurement_evidence.py, execution_evidence.py, database.py, the gate's worker-probe branch, and in
# native_artifact.py _validate_execution_provenance for "execution_mode" and "input_provenance" and
# _validate_token_streams for "input_provenance" and "warmup_results"). Writing any of them with SGLang content
# makes a reader raise or mislabel the run, so they are never written; SGLang evidence goes into "config" and
# "producer" instead.
FORBIDDEN_TOP_LEVEL_KEYS = (
    "measurement_protocol",
    "warmup_evidence",
    "engine",
    "cudagraph",
    "limits",
    "recurrent_state",
    "worker_probe_path",
    "input_provenance",
    "execution_mode",
    "warmup_results",
)

# The four fpm keys this module owns, and two evidence keys that measurement_evidence.py reads with the vLLM
# layout ("benchmark_measurement" raises when no measurement_protocol is present). Callers may not supply them.
_FPM_CORE_KEYS = ("counter_id", "dp_rank", "wall_time", "scheduled_requests")
_FPM_RESERVED_EVIDENCE_KEYS = ("benchmark_measurement", "benchmark_sample")

_POINT_INT_KEYS = ("benchmark_id", "total_prefill_tokens", "total_kv_read_tokens", "batch_size")
_POINT_KEYS = ("point_type", *_POINT_INT_KEYS, "sample_reasons")
_MEASUREMENT_KEYS = ("point", "kv_seed_regime", "rank_fpms")


# ---------------------------------------------------------------------------------------------------------------
# Input checks. Nothing here repairs or coerces: a wrong input is a producer bug and raises.
# ---------------------------------------------------------------------------------------------------------------


def _require_int(value: Any, name: str, *, minimum: int) -> int:
    if type(value) is not int:  # rejects bool, float and numeric strings
        raise TypeError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {value!r}")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _require_wall_time(value: Any, name: str) -> float:
    """A measured step time: a float (not int, not bool), finite and above zero.

    The validator rejects a wall_time that is not finite or not positive (native_artifact.py, _validate_fpm).
    """
    if type(value) is not float:
        raise TypeError(f"{name} must be a float in seconds, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return value


def _require_sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):  # a str or a dict would iterate, but is never what is meant
        raise TypeError(f"{name} must be a list or tuple, got {type(value).__name__}")
    return list(value)


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict, got {type(value).__name__}")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be str, got {key!r}")
    return value


def _require_str_keys(value: Any, name: str, ancestors: tuple[Any, ...] = ()) -> None:
    """Raise TypeError, naming the argument and the key, if a dict key at any depth of ``value`` is not a str.

    Dicts, lists and tuples are searched, which are the containers json writes. ``ancestors`` holds the
    containers the search is inside of: one that contains itself is not entered again, and the serialisation
    that follows this check in ``_require_json`` reports the reference cycle.
    """
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise TypeError(f"{name} is not JSON data: dict keys must be str at every depth, got {key!r}")
        children: Any = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    else:
        return
    if any(value is ancestor for ancestor in ancestors):
        return
    for child in children:
        _require_str_keys(child, name, (*ancestors, value))


def _require_json(value: Any, name: str) -> None:
    """Raise, naming the argument, unless ``value`` is strict JSON data in which every dict key is a str.

    Two rules:

    - Every dict key, at every depth, is a str (``_require_str_keys``; TypeError, naming the key). json would
      write an int, float, bool or None key as text, so the file would not read back equal to the payload,
      and ``{1: "a", "1": "b"}`` would be written as one JSON object with the key ``"1"`` twice.
    - The value can be written as strict JSON, with no NaN and no infinity (the TypeError or ValueError of
      the serialisation, with the argument's name in front).

    A tuple is accepted and is written as a JSON list, so it reads back as a list.

    The files are written only after the sweep (``write_artifacts``), where a value that cannot be written
    loses the whole run with nothing on disk. Evidence is therefore serialised once, and thrown away, by the
    call that receives it, so the error surfaces at the point or the setting that produced the value.

    The key rule is applied here only. The serialisation of the write does not repeat it: a key that is put
    into a payload after this check and is not a str is still written as text where json can do that.
    """
    _require_str_keys(value, name)
    try:
        _serialize(value)  # the same serialisation as the write, so the two can never disagree
    except TypeError as error:
        raise TypeError(f"{name} is not JSON data: {error}") from error
    except ValueError as error:
        raise ValueError(f"{name} is not JSON data: {error}") from error


def _check_kvwarm_dataset(dataset: Any, name: str) -> None:
    """Raise unless ``dataset`` is the dataset record a kvwarm block may carry; ``name`` says where it stands.

    The record is a dict with exactly the four keys of ``_KVWARM_DATASET_FIELDS``: ``path`` and
    ``collection_pool`` are non-empty strings, ``sha256`` is the 64 lower-case hexadecimal digits of a SHA-256,
    and ``conversations`` is an int above zero (a run cannot have taken its text from no conversation). A
    value of the wrong type raises TypeError, any other fault ValueError.
    """
    if not isinstance(dataset, dict):
        raise TypeError(f"{name} must be a dict, got {type(dataset).__name__}")
    if set(dataset) != set(_KVWARM_DATASET_FIELDS):
        raise ValueError(
            f"{name} must have exactly the keys {list(_KVWARM_DATASET_FIELDS)}, got {sorted(dataset, key=repr)}"
        )
    _require_text(dataset["path"], f"{name}['path']")
    digest = dataset["sha256"]
    if not isinstance(digest, str):
        raise TypeError(f"{name}['sha256'] must be a str, got {digest!r}")
    if len(digest) != _SHA256_HEX_DIGITS or not set(digest) <= _LOWER_HEX_DIGITS:
        raise ValueError(
            f"{name}['sha256'] must be the {_SHA256_HEX_DIGITS} lower-case hexadecimal digits of a SHA-256, "
            f"got {digest!r}"
        )
    _require_text(dataset["collection_pool"], f"{name}['collection_pool']")
    _require_int(dataset["conversations"], f"{name}['conversations']", minimum=1)


# ---------------------------------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------------------------------


def _pure_path(path: Any, name: str) -> PurePosixPath:
    """Parse a result path, refusing forms that have no file name.

    A trailing slash passes AISim's render-time check (fpm_contract.py, fpm_validate_benchmark_output_path
    normalises through PurePosixPath). The gate's shell naming, AISim's Python naming and the file write then
    disagree about the rank files, so such a path is refused here. A caller should call
    ``rank_output_path`` for every rank at startup, so that a bad path fails before the sweep and not after it.
    """
    _require_text(path, name)
    parsed = PurePosixPath(path)
    if path.endswith("/") or not parsed.name:
        raise ValueError(f"{name} must name a file, got {path!r}")
    return parsed


def rank_output_path(output_path: str, dp_rank: int) -> str:
    """Result path of one attention-DP rank.

    Rank 0 writes the configured output path unchanged; rank ``r`` inserts ``_dp<r>`` before the suffix. This is
    the expression of ``aisim.fpm_contract.fpm_benchmark_result_name`` (pathlib's stem and suffix), so on one
    Python version the two give the same path for every input.

    The gate waits for files at the paths of its own shell function ``benchmark_path_for_dp_rank``
    (fpm_exec.sh). That function gives the same path for every name with a non-empty stem and a suffix, and
    AISim renders no other name: the basename must match ``benchmark*.json`` (fpm_contract.py,
    ``fpm_validate_benchmark_output_path``). It gives ANOTHER path for a hidden name without a second dot
    (``.hidden``: the shell writes ``_dp1.hidden``, pathlib ``.hidden_dp1``). A name ending in a dot depends on
    the Python version: before 3.14 pathlib reports no suffix for it and the shell function differs; from 3.14
    on the suffix is ``.`` and the two agree.
    """
    parsed = _pure_path(output_path, "output_path")
    _require_int(dp_rank, "dp_rank", minimum=0)
    if dp_rank == 0:
        return output_path
    return str(parsed.with_name(f"{parsed.stem}_dp{dp_rank}{parsed.suffix}"))


def merged_output_path(first_rank_path: str) -> str:
    """Path of the merged file: the first local rank file's name with ``_merged`` before the suffix.

    Same rule as the gate's checker (fpm_exec.sh, ``first_path.with_name(f"{stem}_merged{suffix}")``).
    """
    parsed = _pure_path(first_rank_path, "first_rank_path")
    return str(parsed.with_name(f"{parsed.stem}_merged{parsed.suffix}"))


def _recorded_path(path: str) -> str:
    """The form of a path as it is written INSIDE the merged file.

    The gate compares these strings with ``str(pathlib.Path(argv))`` of the paths it was started with. pathlib
    only tidies the text (it collapses doubled slashes) and follows no symlink, so the path goes through
    PurePosixPath once, which tidies it the same way, and is never resolved: a symlinked results directory
    must keep its own name.
    """
    return str(_pure_path(path, "rank path"))


# ---------------------------------------------------------------------------------------------------------------
# Per-point pieces
# ---------------------------------------------------------------------------------------------------------------


def validate_point(point: Any) -> None:
    """Raise unless ``point`` is a point dict the validator will accept.

    These are the validator's own point rules (native_artifact.py, ``_require_int`` and ``_expected_scheduled``).
    Points from ``sglbench_grid`` satisfy them by construction; checking again here is the last chance to
    catch a driver bug inside the pod, before a ``complete`` artifact is written that AISim then rejects.
    """
    if not isinstance(point, dict):
        raise TypeError(f"benchmark point must be a dict, got {type(point).__name__}")
    missing = [key for key in _POINT_KEYS if key not in point]
    if missing:
        raise ValueError(f"benchmark point lacks {missing}: {point!r}")
    phase = point["point_type"]
    if phase not in (PREFILL, DECODE):
        raise ValueError(f"benchmark point has unknown point_type {phase!r}")
    for key in _POINT_INT_KEYS:
        if type(point[key]) is not int:  # the validator rejects bool and float coordinates
            raise TypeError(f"benchmark point field {key!r} must be an int, got {point[key]!r}")
    batch = point["batch_size"]
    total_new = point["total_prefill_tokens"]
    total_kv = point["total_kv_read_tokens"]
    if point["benchmark_id"] < 1:
        raise ValueError(f"benchmark_id must be >= 1: {point!r}")
    if batch < 1:
        raise ValueError(f"batch_size must be >= 1: {point!r}")
    if phase == PREFILL:
        # Every prefill request computes at least one new token; cached tokens may be zero.
        if total_new < batch:
            raise ValueError(f"prefill point schedules fewer new tokens than requests: {point!r}")
        if total_kv < 0:
            raise ValueError(f"prefill point has negative cached tokens: {point!r}")
    else:
        # A decode step computes no prompt tokens, and every request reads at least one past token.
        if total_new != 0:
            raise ValueError(f"decode point carries prefill tokens: {point!r}")
        if total_kv < batch:
            raise ValueError(f"decode point reads fewer past tokens than requests: {point!r}")
    reasons = point["sample_reasons"]
    if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
        raise TypeError(f"sample_reasons must be a list of str: {point!r}")


def scheduled_requests(point: dict[str, Any]) -> dict[str, int]:
    """The five counters of one forward pass, derived from the point's own totals.

    The validator derives the same five values from the artifact's point and compares them key by key
    (native_artifact.py, ``_expected_scheduled`` and the workload check in ``_validate_fpm``). For decode,
    ``sum_decode_kv_tokens`` counts the tokens that are in the cache BEFORE the step, which is the point's
    context.

    For this producer that host check is not independent evidence: ``build_fpm`` writes these derived counts,
    never counters reported by the engine, so the host compares one derivation from the point with another
    derivation from the same point, and the check cannot fail for an artifact built here. Whether the engine
    ran the planned shape is decided in the pod: the caller must compare the counters the engine observed for
    the measured batch with these values before it builds the fpm, and must fail the point when they differ.
    """
    validate_point(point)
    batch = point["batch_size"]
    if point["point_type"] == PREFILL:
        return {
            "num_prefill_requests": batch,
            "sum_prefill_tokens": point["total_prefill_tokens"],
            "sum_prefill_kv_tokens": point["total_kv_read_tokens"],
            "num_decode_requests": 0,
            "sum_decode_kv_tokens": 0,
        }
    return {
        "num_prefill_requests": 0,
        "sum_prefill_tokens": 0,
        "sum_prefill_kv_tokens": 0,
        "num_decode_requests": batch,
        "sum_decode_kv_tokens": point["total_kv_read_tokens"],
    }


def kv_seed_regime(point: dict[str, Any]) -> str:
    """How the cached state of a point was produced, as the row-level ``kv_seed_regime`` string.

    - decode: ``real_kv``, and the point must carry the stamp ``kvwarm_real_kv``;
    - prefill with cached tokens: ``real_prefix``, and the point must carry ``prefill_real_seed``;
    - prefill with no cached tokens: ``not_applicable``, and the point must carry no stamp. This is the literal
      AISim's publication step recognises for an ordinary uncached sample
      (native_artifact.py, ``_zero_kv_prefill_sample``).

    The validator requires the stamps found in ``sample_reasons`` to be exactly the one that belongs to the
    regime, out of all four it knows (``_validate_seed_regime``). A fake stamp beside the right one is therefore
    refused here too. Other reasons, such as ``explicit`` for a manifest point, are free.
    """
    validate_point(point)
    found = set(point["sample_reasons"]) & REGIME_STAMPS
    if point["point_type"] == DECODE:
        regime, expected = REGIME_REAL_KV, {STAMP_DECODE_REAL_KV}
    elif point["total_kv_read_tokens"] > 0:
        regime, expected = REGIME_REAL_PREFIX, {STAMP_PREFILL_REAL_SEED}
    else:
        regime, expected = REGIME_NOT_APPLICABLE, set()
    if found != expected:
        raise ValueError(
            f"benchmark_id={point['benchmark_id']}: regime {regime!r} needs the stamps {sorted(expected)} in "
            f"sample_reasons, found {sorted(found)}"
        )
    return regime


def decode_wall_time(raw_step_seconds: Any) -> float:
    """Reduce the six decode step times of one run to the point's sample.

    Returns the upper median of the kept steps (see the deviation note at DECODE_KEPT_STEP_INDICES): with
    three kept values that is the middle one. The result is one of the inputs, unchanged; no arithmetic is
    done on it. The caller stores the six raw values beside it in the fpm (DECODE_STEPS_KEY, which a decode fpm
    must carry), so a reader can find the sample among them.
    """
    steps = _require_sequence(raw_step_seconds, "raw_step_seconds")
    if len(steps) != DECODE_RAW_STEPS:
        raise ValueError(f"a decode run has exactly {DECODE_RAW_STEPS} decode steps, got {len(steps)}")
    for index, value in enumerate(steps):
        _require_wall_time(value, f"raw_step_seconds[{index}]")
    kept = sorted(steps[index] for index in DECODE_KEPT_STEP_INDICES)
    return kept[len(kept) // 2]


def build_fpm(
    point: dict[str, Any], dp_rank: int, wall_time: float, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """One forward-pass record: what one rank ran for one point, and how long it took.

    ``wall_time`` is seconds on the monotonic clock of that rank's scheduler process. Prefill: from
    ``batch.launch_ts`` (set in ``Scheduler.run_batch``, python/sglang/srt/managers/scheduler.py) to the batch
    result being on the host. Decode: the value returned by ``decode_wall_time``. AISim multiplies by 1000
    itself (database.py, ``aggregate_cell``), so the value is never scaled or rounded here.

    ``extra`` adds evidence keys, for example ``worker_id``. It may not replace the four keys built here, may
    not use a key that measurement_evidence.py reads with the vLLM layout, and must be JSON data without NaN
    or infinity, with a str for every dict key at every depth: a value that cannot be written, or a key that
    json would have to turn into text, raises here, at the point that produced it. A tuple is accepted and
    is written as a list.

    A decode fpm must carry its six raw step times in ``extra`` under ``sglbench_decode_steps``
    (DECODE_STEPS_KEY), and ``wall_time`` must be ``decode_wall_time`` of them. A prefill fpm may not carry
    that key. ``build_measurement`` and ``build_rank_artifact`` apply the same rule to every fpm they are
    given, and ``build_merged_artifact`` and ``write_artifacts`` to every fpm of a ``complete`` rank artifact.
    """
    validate_point(point)
    _require_int(dp_rank, "dp_rank", minimum=0)
    _require_wall_time(wall_time, "wall_time")
    fpm: dict[str, Any] = {
        "counter_id": point["benchmark_id"],  # the validator requires counter_id == benchmark_id
        "dp_rank": dp_rank,
        "wall_time": wall_time,
        "scheduled_requests": scheduled_requests(point),
    }
    if extra is not None:
        _require_mapping(extra, "extra")
        refused = [key for key in extra if key in _FPM_CORE_KEYS or key in _FPM_RESERVED_EVIDENCE_KEYS]
        if refused:
            raise ValueError(f"extra may not contain the fpm keys {refused}")
        fpm.update(copy.deepcopy(extra))
    # The steps first: a NaN among them is then reported as a bad step time, not as a value JSON cannot hold.
    _check_decode_steps(point, fpm, f"benchmark_id={point['benchmark_id']} rank={dp_rank}")
    if extra is not None:
        _require_json(extra, "extra")
    return fpm


def _check_decode_steps(point: dict[str, Any], fpm: dict[str, Any], label: str) -> None:
    """Raise unless a decode fpm carries its raw step times and its wall_time is their sample.

    The deviation note at DECODE_KEPT_STEP_INDICES says that the six raw step times are recorded beside the
    sample that was reduced from them. This check holds an fpm to that: a decode fpm without them, with
    another number of them, or with a wall_time that ``decode_wall_time`` does not return for them, is
    refused. A prefill fpm has no decode steps and may not carry the key.

    It runs in three places: in ``build_fpm``, on the fpm being built; in ``_check_fpm``, on every fpm handed
    to ``build_measurement`` or ``build_rank_artifact``; in ``_check_complete_payload``, on every fpm in the
    groups and in the rows of a ``complete`` rank artifact handed to ``build_merged_artifact`` or
    ``write_artifacts``. It does not run on a ``partial`` or ``failed`` rank artifact in that third place,
    and not in ``atomic_write_json``.

    Every refusal starts with ``label``, the place of the fpm: the point and the rank in the first two places;
    in the third the rank file with the group and the rank entry, or with the row.
    """
    if point["point_type"] != DECODE:
        if DECODE_STEPS_KEY in fpm:
            raise ValueError(f"{label}: a prefill fpm may not contain {DECODE_STEPS_KEY!r}")
        return
    if DECODE_STEPS_KEY not in fpm:
        raise ValueError(
            f"{label}: a decode fpm must carry its {DECODE_RAW_STEPS} raw step times under {DECODE_STEPS_KEY!r}"
        )
    steps = fpm[DECODE_STEPS_KEY]
    if not isinstance(steps, list):  # a list, as every sequence in an artifact: it is written as JSON
        raise TypeError(f"{label}: fpm {DECODE_STEPS_KEY} must be a list, got {type(steps).__name__}")
    # decode_wall_time checks the count and every value, but it knows neither the point nor the rank: its error
    # is raised again, as the same type, with the place and the fpm key in front.
    try:
        sample = decode_wall_time(steps)
    except TypeError as error:
        raise TypeError(f"{label}: fpm {DECODE_STEPS_KEY}: {error}") from error
    except ValueError as error:
        raise ValueError(f"{label}: fpm {DECODE_STEPS_KEY}: {error}") from error
    if fpm["wall_time"] != sample:
        raise ValueError(f"{label}: fpm wall_time {fpm['wall_time']!r} is not the sample {sample!r} of its raw steps")


def _check_fpm(point: dict[str, Any], dp_rank: int, fpm: Any) -> None:
    """Raise unless ``fpm`` is the record of ``point`` on ``dp_rank`` (the validator's ``_validate_fpm`` rules)."""
    label = f"benchmark_id={point['benchmark_id']} rank={dp_rank}"
    if not isinstance(fpm, dict):
        raise TypeError(f"{label}: fpm must be a dict, got {type(fpm).__name__}")
    missing = [key for key in _FPM_CORE_KEYS if key not in fpm]
    if missing:
        raise ValueError(f"{label}: fpm lacks {missing}")
    if type(fpm["counter_id"]) is not int or fpm["counter_id"] != point["benchmark_id"]:
        raise ValueError(f"{label}: fpm counter_id is {fpm['counter_id']!r}")
    if type(fpm["dp_rank"]) is not int or fpm["dp_rank"] != dp_rank:
        raise ValueError(f"{label}: fpm dp_rank is {fpm['dp_rank']!r}")
    _require_wall_time(fpm["wall_time"], f"{label}: fpm wall_time")
    if fpm["scheduled_requests"] != scheduled_requests(point):
        raise ValueError(f"{label}: fpm scheduled_requests do not match the point: {fpm['scheduled_requests']!r}")
    refused = [key for key in _FPM_RESERVED_EVIDENCE_KEYS if key in fpm]
    if refused:
        raise ValueError(f"{label}: fpm may not contain {refused}")
    _check_decode_steps(point, fpm, label)


def build_measurement(point: dict[str, Any], rank_fpms: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """One completed point: the point, its regime, and one fpm per attention-DP rank (``{rank: fpm}``)."""
    regime = kv_seed_regime(point)  # validates the point as well
    if not isinstance(rank_fpms, dict):
        raise TypeError(f"rank_fpms must be a dict of rank to fpm, got {type(rank_fpms).__name__}")
    if not rank_fpms:
        raise ValueError(f"benchmark_id={point['benchmark_id']}: rank_fpms is empty")
    for rank, fpm in rank_fpms.items():
        _require_int(rank, "rank_fpms key", minimum=0)
        _check_fpm(point, rank, fpm)
    return {
        "point": copy.deepcopy(point),
        "kv_seed_regime": regime,
        "rank_fpms": {rank: copy.deepcopy(rank_fpms[rank]) for rank in sorted(rank_fpms)},
    }


def _check_measurement(measurement: Any, *, position: int, dp_size: int, mode: str) -> None:
    """Re-check one measurement at artifact build: the dicts are plain data and may have been hand-made."""
    label = f"measurements[{position}]"
    if not isinstance(measurement, dict):
        raise TypeError(f"{label} must be a dict from build_measurement, got {type(measurement).__name__}")
    missing = [key for key in _MEASUREMENT_KEYS if key not in measurement]
    if missing:
        raise ValueError(f"{label} lacks {missing}")
    point = measurement["point"]
    regime = kv_seed_regime(point)
    if measurement["kv_seed_regime"] != regime:
        raise ValueError(f"{label}: kv_seed_regime is {measurement['kv_seed_regime']!r}, the point says {regime!r}")
    # Execution order is id order, and ids start at 1: the validator requires exactly the set 1..N.
    if point["benchmark_id"] != position + 1:
        raise ValueError(f"{label}: benchmark_id is {point['benchmark_id']}, expected {position + 1}")
    if point["point_type"] not in _MODE_POINT_TYPES[mode]:
        raise ValueError(f"{label}: a {point['point_type']!r} point is not valid in mode {mode!r}")
    rank_fpms = measurement["rank_fpms"]
    if not isinstance(rank_fpms, dict):
        raise TypeError(f"{label}: rank_fpms must be a dict, got {type(rank_fpms).__name__}")
    for rank in rank_fpms:
        _require_int(rank, f"{label}: rank_fpms key", minimum=0)
    # One fpm per rank of the run, no more and no fewer: every rank file embeds every rank's fpm in its groups.
    if set(rank_fpms) != set(range(dp_size)):
        raise ValueError(f"{label}: rank_fpms has ranks {sorted(rank_fpms)}, expected {list(range(dp_size))}")
    for rank, fpm in rank_fpms.items():
        _check_fpm(point, rank, fpm)


# ---------------------------------------------------------------------------------------------------------------
# Rank artifact
# ---------------------------------------------------------------------------------------------------------------


def _check_missing_phases(missing_phases: Any, mode: str) -> list[str]:
    phases = _require_sequence(missing_phases, "missing_phases")
    allowed = _MODE_POINT_TYPES[mode]
    for phase in phases:
        if phase not in allowed:
            raise ValueError(f"missing phase {phase!r} is not a phase of mode {mode!r}")
    # Unique, and prefill before decode (the order in which an agg run executes its phases).
    if phases != [phase for phase in allowed if phase in phases]:
        raise ValueError(f"missing_phases must be unique and ordered prefill, decode: {phases!r}")
    return phases


def _check_failure(failure: Any, *, completed: int, expected_points: int, mode: str) -> dict[str, Any] | None:
    """Validate ``failure`` and return a copy of the failed point, or None when the failure names no point."""
    if not isinstance(failure, dict):
        raise TypeError(f"failure must be a dict, got {type(failure).__name__}")
    if set(failure) != {"point", "reason"}:
        raise ValueError(f"failure must have exactly the keys 'point' and 'reason', got {sorted(failure, key=repr)}")
    _require_text(failure["reason"], "failure reason")
    point = failure["point"]
    if point is None:
        return None
    kv_seed_regime(point)  # validates the point and its stamps
    # Points run in id order and the run stops at the first failure, so the failed point is the next id.
    if point["benchmark_id"] != completed + 1 or point["benchmark_id"] > expected_points:
        raise ValueError(
            f"the failed point has benchmark_id={point['benchmark_id']}, but {completed} of {expected_points} "
            "points completed before it"
        )
    if point["point_type"] not in _MODE_POINT_TYPES[mode]:
        raise ValueError(f"the failed point is a {point['point_type']!r} point in mode {mode!r}")
    return copy.deepcopy(point)


def build_rank_artifact(
    *,
    dp_rank: int,
    dp_size: int,
    mode: str,
    run_id: str,
    grid_digest: str,
    expected_points: int,
    measurements: Any,
    benchmark_elapsed_seconds: float,
    config: dict[str, Any] | None = None,
    producer: dict[str, Any] | None = None,
    missing_phases: Any = (),
    stop_reason: str | None = None,
    failure: dict[str, Any] | None = None,
    kvwarm_dataset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the artifact of one attention-DP rank.

    Call it once per rank with the SAME measurements: every rank file must carry identical ``iteration_groups``
    and differs only in ``dp.rank``, its own fpm per row, and possibly its elapsed time. The other arguments
    are the same for every rank as well; ``build_merged_artifact`` and ``write_artifacts`` refuse ranks that
    differ in the run identity, the mode, the status, the coverage or the ``kvwarm`` block.

    measurements
        The completed points in execution order, from ``build_measurement``. Their ids must be exactly
        ``1..len(measurements)`` and each must have one fpm for every rank in ``range(dp_size)``.
    expected_points
        The number of points in the grid. Fewer measurements means the run stopped early.
    failure
        ``None``, or ``{"point": point dict or None, "reason": str}`` for what stopped the run. A failed point
        must be the next one in id order.
    missing_phases
        Phases of ``mode`` for which the grid has no feasible point (``"prefill"`` before ``"decode"``).
    stop_reason
        ``None`` or ``"timeout"`` (the soft deadline passed).
    config
        Run-INVARIANT benchmark settings (sample caps, manifest digest, the limits the grid was built from).
        It has a reader: ``extract_measurement_evidence`` copies it, minus ``output_path``, into
        ``benchmark_config`` and compares that between two launches of one cell. A run id, timestamp, pid or
        host name here would turn that comparison into a mismatch; such values belong in ``producer``.
        ``mode`` is added here and may not be supplied.
    producer
        Free-form evidence: timing metric and clock, staging method, page size, overlap and streaming settings,
        content source and seed, limits record, SGLang version, run-specific values. Four keys have a reader.
        For a cell of an SGLang plan, ``validate_native_collection`` (native_artifact.py) requires
        ``sglang_version`` in every rank artifact and compares it with the installed release in the pod's
        provenance file. For a cell of such a plan whose model has expert layers it also requires
        ``content.source`` to name dataset text, and ``tp_size`` and ``moe.ep_size`` to be the tensor-parallel
        and the expert-parallel size of the cell (``_validate_sglang_moe_producer``). No reader parses the
        other keys.
    kvwarm_dataset
        ``None`` for a run whose prompts were not dataset text: the ``kvwarm`` block is then the four fixed
        keys. Otherwise the record of the dataset the prompts were made of, a dict with exactly the keys
        ``path`` (the local path of the file, a non-empty str), ``sha256`` (the 64 lower-case hexadecimal
        digits of the file's SHA-256), ``collection_pool`` (the name of the part of its conversations that
        was used, a non-empty str) and ``conversations`` (how many conversations that part holds, an int
        above zero). It is written into the ``kvwarm`` block under ``dataset``, whatever the status of the
        run. It may only be given for the modes ``decode`` and ``agg``: a prefill artifact has no ``kvwarm``
        block, so there it raises ValueError. No reader parses the record.

    ``config`` and ``producer`` must be JSON data without NaN or infinity, with a str for every dict key at
    every depth; a tuple is accepted and is written as a list. They are serialised once here, so a value that
    cannot be written raises at this call and not in ``write_artifacts`` after the sweep.

    Status, in this order:
        1. ``failure`` or ``missing_phases`` given: ``failed``. Not valid, not usable.
        2. fewer measurements than ``expected_points``: ``partial``; ``stop_reason`` must be ``"timeout"``,
           anything else means points were lost without a recorded reason and raises. Usable when at least one
           point completed.
        3. otherwise ``complete`` (at least one point). A deadline that passes after the last point leaves the
           run complete, so ``stop_reason`` is written as ``None``.
    Every other inconsistency is a producer bug and raises.
    """
    _require_int(dp_size, "dp_size", minimum=1)
    _require_int(dp_rank, "dp_rank", minimum=0)
    if dp_rank >= dp_size:
        raise ValueError(f"dp_rank {dp_rank} is outside range({dp_size})")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    _require_text(run_id, "run_id")
    _require_text(grid_digest, "grid_digest")
    _require_int(expected_points, "expected_points", minimum=0)
    if stop_reason is not None and stop_reason != STOP_REASON_TIMEOUT:
        raise ValueError(f"stop_reason must be None or {STOP_REASON_TIMEOUT!r}, got {stop_reason!r}")
    if type(benchmark_elapsed_seconds) not in (float, int):
        raise TypeError(f"benchmark_elapsed_seconds must be a number, got {benchmark_elapsed_seconds!r}")
    if not math.isfinite(benchmark_elapsed_seconds) or benchmark_elapsed_seconds < 0:
        raise ValueError(f"benchmark_elapsed_seconds must be finite and >= 0, got {benchmark_elapsed_seconds!r}")
    if kvwarm_dataset is not None:
        if mode == PREFILL:
            raise ValueError("kvwarm_dataset was given for a prefill artifact, which has no kvwarm block")
        _check_kvwarm_dataset(kvwarm_dataset, "kvwarm_dataset")

    config_block: dict[str, Any] = {"mode": mode}
    if config is not None:
        if "mode" in _require_mapping(config, "config"):
            raise ValueError("config may not contain 'mode'; it is set from the mode argument")
        _require_json(config, "config")
        config_block.update(copy.deepcopy(config))
    producer_block: dict[str, Any] = {"name": PRODUCER_NAME, "artifact_format": ARTIFACT_FORMAT}
    if producer is not None:
        refused = [key for key in _require_mapping(producer, "producer") if key in producer_block]
        if refused:
            raise ValueError(f"producer may not contain {refused}")
        _require_json(producer, "producer")
        producer_block.update(copy.deepcopy(producer))

    measured = _require_sequence(measurements, "measurements")
    completed = len(measured)
    if completed > expected_points:
        raise ValueError(f"{completed} measurements for a grid of {expected_points} points")
    coordinates: set[tuple[str, int, int, int]] = set()
    for position, measurement in enumerate(measured):
        _check_measurement(measurement, position=position, dp_size=dp_size, mode=mode)
        point = measurement["point"]
        # AISim publishes one row per physical coordinate and raises on a second sample of the same one
        # (native_artifact.py, select_native_measurements), after the whole run has been collected.
        coordinate = (
            point["point_type"],
            point["batch_size"],
            point["total_prefill_tokens"],
            point["total_kv_read_tokens"],
        )
        if coordinate in coordinates:
            raise ValueError(f"benchmark_id={point['benchmark_id']}: second measurement of coordinate {coordinate}")
        coordinates.add(coordinate)

    missing = _check_missing_phases(missing_phases, mode)
    skipped_points: list[dict[str, Any]] = []
    error: str | None = None
    errors: list[str] = []
    written_stop_reason: str | None = None
    if failure is not None or missing:
        status = STATUS_FAILED
        usable = False
        written_stop_reason = stop_reason  # a failure after the deadline keeps both facts
        if failure is not None:
            failed_point = _check_failure(failure, completed=completed, expected_points=expected_points, mode=mode)
            if failed_point is None:
                error = failure["reason"]
            else:
                skipped_points.append(
                    {"point": failed_point, "kv_seed_regime": kv_seed_regime(failed_point), "reason": failure["reason"]}
                )
                error = f"benchmark_id={failed_point['benchmark_id']}: benchmark point failed: {failure['reason']}"
        else:
            error = "no feasible benchmark points for phase(s): " + ", ".join(missing)
        errors = [error]
    elif completed < expected_points:
        if stop_reason != STOP_REASON_TIMEOUT:
            raise ValueError(
                f"only {completed} of {expected_points} points completed and there is no failure: the only "
                f"accepted reason is stop_reason={STOP_REASON_TIMEOUT!r}, got {stop_reason!r}"
            )
        status = STATUS_PARTIAL
        usable = completed >= 1
        written_stop_reason = STOP_REASON_TIMEOUT
        # "error" stays None (the run did not fail). The gate prints "errors" when it rejects the file, so
        # the reason for the rejection is put there.
        errors = [f"benchmark stopped by timeout after {completed} of {expected_points} points"]
    else:
        if expected_points < 1:
            raise ValueError("a run with no points cannot be complete; report the phase in missing_phases")
        status = STATUS_COMPLETE
        usable = True

    results = []
    iteration_groups = []
    measured_iteration_seconds = 0.0
    for measurement in measured:
        point = copy.deepcopy(measurement["point"])
        fpms = {rank: copy.deepcopy(measurement["rank_fpms"][rank]) for rank in range(dp_size)}
        # A row holds this rank's own fpm only: the validator and the gate both require exactly that.
        results.append({"point": point, "kv_seed_regime": measurement["kv_seed_regime"], "fpms": [fpms[dp_rank]]})
        # A group holds every rank's fpm, equal to the fpm in that rank's own file, and the group's time is
        # the largest rank wall_time: the validator requires that value and AISim publishes it (database.py,
        # aggregate_cell). For prefill each rank's wall_time is its one extend batch of the point, so the
        # group time is the slowest rank's batch. For decode with more than one rank it is the largest
        # per-rank SAMPLE: each rank's wall_time is its own upper median of steps 3 to 5 (decode_wall_time)
        # and the ranks need not pick the same step, so the group time is not the time of any one step. It
        # is never above, and can be below, the upper median of the per-step slowest-rank times: this module
        # does not reduce across ranks before it takes the median. max() returns one of the floats unchanged.
        group_wall_time = max(fpms[rank]["wall_time"] for rank in range(dp_size))
        iteration_groups.append(
            {
                "benchmark_id": point["benchmark_id"],
                "point": point,
                "expected_dp_ranks": list(range(dp_size)),
                "complete": True,
                "rank_results": [{"dp_rank": rank, "fpms": [fpms[rank]]} for rank in range(dp_size)],
                "wall_time": group_wall_time,
            }
        )
        # Plain left-to-right addition in id order: the validator recomputes the sum the same way.
        measured_iteration_seconds += group_wall_time
    if measured_iteration_seconds > benchmark_elapsed_seconds:
        raise ValueError(
            f"the measured steps sum to {measured_iteration_seconds!r} s, more than the elapsed "
            f"{benchmark_elapsed_seconds!r} s of the run"
        )

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "rank",
        "status": status,
        "valid": status == STATUS_COMPLETE,
        "usable": usable,
        # Inconsistent timing raises above instead of being written with timing_valid false.
        "timing_valid": True,
        "stop_reason": written_stop_reason,
        "error": error,
        "errors": errors,
        "run_id": run_id,
        "grid_digest": grid_digest,
        "config": config_block,
        "coverage": {
            "expected_points": expected_points,
            "completed_points": completed,
            "skipped_points": len(skipped_points),
        },
        "dp": {"size": dp_size, "rank": dp_rank},
        "results": results,
        "iteration_groups": iteration_groups,
        "skipped_points": skipped_points,
        "missing_phases": missing,
        "timing": {
            "benchmark_elapsed_seconds": benchmark_elapsed_seconds,
            "measured_iteration_seconds": measured_iteration_seconds,
        },
        "measurement_policy": dict(MEASUREMENT_POLICY),
        "producer": producer_block,
    }
    if mode in (DECODE, AGG):
        # The decode context of every point was computed by a real seed request through SGLang's radix cache.
        # The validator needs enabled/warm_eligible/skip_reason for the strategies tep, dep and pure_tp
        # (_validate_kvwarm_contract); for the others the block is what makes AISim publish "real_kv" instead
        # of "legacy" (database.py, aggregate_cell). "method" is extra evidence. Prefill artifacts carry no
        # such block. Dynamo's vLLM producer writes none either in a prefill run with its default prompt
        # content: _bench_write_results writes the key only when the block exists. _kvwarm_meta_init creates
        # it, and _bench_make_local_capacity and _kvwarm_prepare reach that only for the modes decode and agg.
        kvwarm_block: dict[str, Any] = dict(_KVWARM_BLOCK)
        if kvwarm_dataset is not None:
            # The prompts were dataset text: the block names the dataset, under the key and with the keys of
            # Dynamo's record (_kvwarm_load_texts). The four values are strings and an int, copied in the
            # fixed order, so the block shares nothing with the caller's dict.
            kvwarm_block[_KVWARM_DATASET_KEY] = {key: kvwarm_dataset[key] for key in _KVWARM_DATASET_FIELDS}
        artifact["kvwarm"] = kvwarm_block
    return artifact


# ---------------------------------------------------------------------------------------------------------------
# Merged file
# ---------------------------------------------------------------------------------------------------------------


def _check_complete_payload(payload: dict[str, Any]) -> None:
    """Re-check a rank artifact whose status is ``complete``, in one pass over its rows.

    A payload that comes unchanged from ``build_rank_artifact`` satisfies all of it by construction. The check
    is for a payload that was edited after it was built. The rules are the builder's own invariants.

    The envelope. Each envelope rule stands for a rule the validator applies to a complete rank artifact
    (native_artifact.py, ``validate_native_collection``, ``_validate_seed_regime``,
    ``_validate_kvwarm_contract``). Where the validator is looser (float sums within a relative tolerance, a
    kvwarm block that only some strategies need) the check still asks for exactly what the builder writes. Of
    these rules the gate applies only ``valid`` and the coverage counts, which it wants as exact ints that
    agree with the number of results (fpm_exec.sh, ``check_result_files``). Without this check most edited
    payloads would therefore be written as ``complete``, pass the gate in the pod, and be rejected on the
    collector host after the run. Each violation raises ValueError naming the rank and the field.

    The ``kvwarm`` block of a decode or agg artifact is one of the two forms the builder writes: the four
    fixed keys with their values and types, alone or with a ``dataset`` record that ``build_rank_artifact``
    accepts as ``kvwarm_dataset`` (``_check_kvwarm_dataset``). Any other added key is refused, and so is a
    record with a missing, an added or a malformed value. Neither checker has a rule for the record: the
    validator reads three of the fixed keys and carries the rest of the block along.

    The decode steps. For each row, once its envelope rules hold, every fpm of every rank entry of its group
    goes through ``_check_decode_steps`` with the point type of the row's point: a decode fpm must still carry
    its six raw step times with ``wall_time`` as their sample, and a prefill fpm may not carry the key.
    Neither checker has such a rule (both ignore the key), so without this one an fpm whose steps were
    removed or changed would be written as ``complete`` and accepted. A violation raises what ``build_fpm``
    raises for the same fault (ValueError, or TypeError for a step list or a step time of the wrong type). Its
    text starts with the rank of the file being checked, the group and the rank entry, as in
    ``rank 0: iteration_groups[1].rank_results[1]: ...``. The row's own fpm is then checked in the same way,
    with a text that starts as ``rank 0: results[1].fpms[0]``: it equals its group entry, but an int or a
    bool equals a float of the same value, so a row that is not the same object as its entry could
    otherwise hold a whole-number time or step of another type.

    This is not a second validator: an edit outside these rules (a point's coordinates, ``config.mode``, the
    counters or an evidence key of an fpm changed alike in its row and its group) is not looked for.

    ``_check_rank_set`` has already checked the shapes this function indexes: ``dp``, ``config`` and
    ``coverage`` are dicts, ``results`` and ``iteration_groups`` are lists, and every row holds one fpm dict.
    """
    rank, size = payload["dp"]["rank"], payload["dp"]["size"]
    label = f"rank {rank}"
    for key in ("valid", "usable", "timing_valid"):
        if payload.get(key) is not True:
            raise ValueError(f"{label}: {key} must be True in a complete artifact, got {payload.get(key)!r}")
    for key in ("stop_reason", "error"):
        if payload.get(key) is not None:
            raise ValueError(f"{label}: {key} must be None in a complete artifact, got {payload.get(key)!r}")
    for key in ("skipped_points", "missing_phases"):
        if payload.get(key) != []:
            raise ValueError(f"{label}: {key} must be [] in a complete artifact, got {payload.get(key)!r}")

    rows, groups, coverage = payload["results"], payload["iteration_groups"], payload["coverage"]
    if not rows:
        raise ValueError(f"{label}: results is empty in a complete artifact")
    for key, count in (("expected_points", len(rows)), ("completed_points", len(rows)), ("skipped_points", 0)):
        if type(coverage.get(key)) is not int or coverage[key] != count:  # the gate refuses a bool count
            raise ValueError(
                f"{label}: coverage.{key} is {coverage.get(key)!r}, but the complete artifact holds {len(rows)} "
                "results and no skipped point"
            )
    if len(groups) != len(rows):
        raise ValueError(f"{label}: iteration_groups holds {len(groups)} groups for {len(rows)} results")

    measured = 0.0
    # Row i and group i belong to the same point: the builder writes both in id order.
    for index, (row, group) in enumerate(zip(rows, groups, strict=True)):
        entries = group.get("rank_results") if isinstance(group, dict) else None
        if not isinstance(entries, list) or len(entries) != size:
            raise ValueError(f"{label}: iteration_groups[{index}].rank_results must hold one entry per rank")
        wall_times = []
        for entry_rank, entry in enumerate(entries):
            fpms = entry.get("fpms") if isinstance(entry, dict) and entry.get("dp_rank") == entry_rank else None
            if not isinstance(fpms, list) or len(fpms) != 1 or not isinstance(fpms[0], dict):
                raise ValueError(
                    f"{label}: iteration_groups[{index}].rank_results[{entry_rank}] must hold dp_rank "
                    f"{entry_rank} and exactly one fpm"
                )
            if type(fpms[0].get("wall_time")) is not float:
                raise ValueError(
                    f"{label}: iteration_groups[{index}].rank_results[{entry_rank}] has no float fpm wall_time"
                )
            wall_times.append(fpms[0]["wall_time"])
        # The validator compares the two fpm objects whole, not only their times.
        if row["fpms"][0] != entries[rank]["fpms"][0]:
            raise ValueError(
                f"{label}: results[{index}].fpms[0] differs from the fpm of rank {rank} in iteration_groups[{index}]"
            )
        if group.get("wall_time") != max(wall_times):
            raise ValueError(
                f"{label}: iteration_groups[{index}].wall_time is {group.get('wall_time')!r}, the largest rank "
                f"wall_time is {max(wall_times)!r}"
            )
        measured += group["wall_time"]
        try:
            regime = kv_seed_regime(row.get("point"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}: results[{index}].point is not a point of a known regime: {error}") from error
        if row.get("kv_seed_regime") != regime:
            raise ValueError(
                f"{label}: results[{index}].kv_seed_regime is {row.get('kv_seed_regime')!r}, the point says {regime!r}"
            )
        # The raw decode steps of every rank's fpm, by the rule of build_fpm. The point type is the row's:
        # kv_seed_regime has just validated that point.
        for entry_rank, entry in enumerate(entries):
            place = f"{label}: iteration_groups[{index}].rank_results[{entry_rank}]"
            _check_decode_steps(row["point"], entry["fpms"][0], place)
        # The row's own fpm equals its group entry, but 1 == 1.0 and True == 1.0: when the two are not one
        # object (after a deep copy or a JSON round trip), the row's copy is held to the same types.
        if type(row["fpms"][0]["wall_time"]) is not float:
            raise ValueError(f"{label}: results[{index}].fpms[0] has no float fpm wall_time")
        _check_decode_steps(row["point"], row["fpms"][0], f"{label}: results[{index}].fpms[0]")

    timing = payload.get("timing")
    recorded = timing.get("measured_iteration_seconds") if isinstance(timing, dict) else None
    # The same left-to-right sum the builder wrote, so the comparison is exact.
    if recorded != measured:
        raise ValueError(
            f"{label}: timing.measured_iteration_seconds is {recorded!r}, the group times sum to {measured!r}"
        )
    mode = payload["config"].get("mode")
    if mode in (DECODE, AGG):
        kvwarm = payload.get("kvwarm")
        # The block is the four fixed keys, alone or with a dataset record: the two forms the builder writes.
        # The record is set aside and what is left must be the fixed keys, so any other added key is refused.
        fixed = kvwarm
        if isinstance(kvwarm, dict) and _KVWARM_DATASET_KEY in kvwarm:
            fixed = {key: value for key, value in kvwarm.items() if key != _KVWARM_DATASET_KEY}
        # Equal, and of the same types: 1 == True, but the validator wants real booleans.
        if fixed != _KVWARM_BLOCK or any(type(fixed[key]) is not type(value) for key, value in _KVWARM_BLOCK.items()):
            raise ValueError(
                f"{label}: kvwarm must be {_KVWARM_BLOCK!r}, alone or with a {_KVWARM_DATASET_KEY!r} record, "
                f"in a complete {mode!r} artifact, got {kvwarm!r}"
            )
        if fixed is not kvwarm:
            try:
                _check_kvwarm_dataset(kvwarm[_KVWARM_DATASET_KEY], f"kvwarm.{_KVWARM_DATASET_KEY}")
            except (TypeError, ValueError) as error:
                # An envelope rule like the others of this check: one exception type, the rank in front.
                raise ValueError(f"{label}: {error}") from error


def _check_rank_set(rank_payloads: list[Any]) -> tuple[int, list[int]]:
    """Check that the payloads are the rank artifacts of ONE run on consecutive ranks.

    Returns ``(dp_size, ranks)``. The cross-rank rules are the validator's (run identity, grid and
    ``iteration_groups`` equal in every rank file) and the gate's (consecutive ranks, one fpm per row with the
    file's own rank), and one that goes beyond both: the ``kvwarm`` block is the same in every rank file,
    dataset record included, where the validator compares three of its keys. A ``complete`` payload is then
    re-checked against the validator's envelope rules and the decode step rule (``_check_complete_payload``);
    ``partial`` and ``failed`` payloads are rejected by both checkers whatever they hold, so nothing more is
    checked for them.
    """
    if not rank_payloads:
        raise ValueError("there is no rank artifact")
    for position, payload in enumerate(rank_payloads):
        label = f"rank_payloads[{position}]"
        if not isinstance(payload, dict):
            raise TypeError(f"{label} must be a dict, got {type(payload).__name__}")
        if payload.get("artifact_type") != "rank" or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{label} is not a schema-{SCHEMA_VERSION} rank artifact")
        if payload.get("status") not in (STATUS_COMPLETE, STATUS_PARTIAL, STATUS_FAILED):
            raise ValueError(f"{label} has unknown status {payload.get('status')!r}")
        refused = [key for key in FORBIDDEN_TOP_LEVEL_KEYS if key in payload]
        if refused:
            raise ValueError(f"{label} contains the keys {refused}, which have vLLM-shaped readers")
        dp = payload.get("dp")
        if not isinstance(dp, dict) or type(dp.get("size")) is not int or type(dp.get("rank")) is not int:
            raise ValueError(f"{label} has no integer dp.size and dp.rank")
        if not 0 <= dp["rank"] < dp["size"]:
            raise ValueError(f"{label}: dp.rank {dp['rank']} is outside range({dp['size']})")
        if not isinstance(payload.get("config"), dict) or not isinstance(payload.get("coverage"), dict):
            raise ValueError(f"{label} has no config or coverage object")
        rows = payload.get("results")
        if not isinstance(rows, list) or not isinstance(payload.get("iteration_groups"), list):
            raise ValueError(f"{label} has no results or iteration_groups list")
        for row in rows:
            fpms = row.get("fpms") if isinstance(row, dict) else None
            if not isinstance(fpms, list) or len(fpms) != 1 or not isinstance(fpms[0], dict):
                raise ValueError(f"{label}: every result row must hold exactly one fpm")
            if fpms[0].get("dp_rank") != dp["rank"]:
                raise ValueError(f"{label}: a result row holds the fpm of rank {fpms[0].get('dp_rank')!r}")

    first = rank_payloads[0]
    size = first["dp"]["size"]
    ranks = [payload["dp"]["rank"] for payload in rank_payloads]
    # The gate requires dp.source_ranks == list(range(start_rank, start_rank + number of local rank files)).
    if ranks != list(range(ranks[0], ranks[0] + len(ranks))):
        raise ValueError(f"rank artifacts must be passed in consecutive ascending rank order, got ranks {ranks}")
    shared = (
        ("run_id", lambda payload: payload.get("run_id")),
        ("grid_digest", lambda payload: payload.get("grid_digest")),
        ("config.mode", lambda payload: payload["config"].get("mode")),
        ("status", lambda payload: payload["status"]),
        ("coverage", lambda payload: payload["coverage"]),
        ("dp.size", lambda payload: payload["dp"]["size"]),
        ("result points", lambda payload: [row.get("point") for row in payload["results"]]),
        ("iteration_groups", lambda payload: payload["iteration_groups"]),
        # One run has one way of producing its decode contexts and one dataset, and the merged file takes
        # the block of the first rank. The validator compares three keys of the block across ranks
        # (validate_native_collection); here the whole block, absent in every rank of a prefill run.
        ("kvwarm", lambda payload: payload.get("kvwarm")),
    )
    for name, read in shared:
        expected = read(first)
        for rank, payload in zip(ranks, rank_payloads, strict=True):
            if read(payload) != expected:
                raise ValueError(f"ranks disagree on {name}: rank {rank} differs from rank {ranks[0]}")
    if first["status"] == STATUS_COMPLETE:  # the ranks agree on the status
        for payload in rank_payloads:
            _check_complete_payload(payload)
    return size, ranks


def build_merged_artifact(rank_payloads: Any, rank_paths: Any) -> dict[str, Any]:
    """Build the merged file of one node: the first rank's artifact plus every rank's fpm per row.

    ``rank_payloads`` are the rank artifacts in ascending rank order and ``rank_paths`` their file paths, one
    per payload. Only the gate uses the content of the merged file; it requires ``artifact_type``,
    ``schema_version``, ``status``, ``valid``, ``run_id``, ``grid_digest``, ``config.mode``, ``rank_files``,
    ``merged_output_path`` and ``dp.source_ranks``. The host readers (native_artifact.py ``_rank_artifacts``,
    runner.py ``_runtime_timing_summary``) parse the file as JSON, must find a JSON object, and then skip it.
    The key ``worker_probe_path`` is never written: with it the gate would wait for a sidecar this producer
    does not write. The ``kvwarm`` block of the merged file is the first rank's, with its dataset record
    when it has one; the ranks are required to carry the same block.

    A ``failed`` run has no merged file (raises). A ``partial`` run has one; the gate fails on the rank file
    before it reads the merged file.
    """
    payloads = _require_sequence(rank_payloads, "rank_payloads")
    paths = _require_sequence(rank_paths, "rank_paths")
    if len(paths) != len(payloads):
        raise ValueError(f"{len(payloads)} rank artifacts but {len(paths)} rank paths")
    size, ranks = _check_rank_set(payloads)
    if payloads[0]["status"] == STATUS_FAILED:
        raise ValueError("a failed run has no merged file")
    rank_files = [_recorded_path(path) for path in paths]
    if len(set(rank_files)) != len(rank_files):
        raise ValueError(f"rank paths are not distinct: {rank_files}")

    merged = copy.deepcopy(payloads[0])
    merged["artifact_type"] = "merged"
    merged["rank_files"] = rank_files
    merged["merged_output_path"] = merged_output_path(paths[0])
    merged["dp"] = {
        "size": size,
        "ranks": list(range(size)),  # every rank of the run
        "source_ranks": ranks,  # the ranks whose files were merged here (this node's)
        "managed_size": len(ranks),
        "global_size": size,
    }
    for index, row in enumerate(merged["results"]):
        row["fpms"] = [copy.deepcopy(payload["results"][index]["fpms"][0]) for payload in payloads]
    return merged


# ---------------------------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------------------------


def _serialize(payload: Any) -> str:
    """JSON text of a payload. NaN and infinity raise instead of producing a file no strict reader accepts.

    The default ``ensure_ascii`` keeps the text pure ASCII, so it reads the same under UTF-8 (the gate) and
    under the locale's default encoding (the validator).
    """
    return json.dumps(payload, allow_nan=False)


def _atomic_write_text(path: str, text: str) -> None:
    directory, name = os.path.split(path)
    if not name:
        raise ValueError(f"path must name a file, got {path!r}")
    # The leading dot and the ".tmp" ending keep the temporary out of every "benchmark*.json" sweep: AISim
    # parses each match as a rank artifact (native_artifact.py, _rank_artifacts; runner.py,
    # _runtime_timing_summary). The random token keeps a temporary left behind by a killed earlier process
    # with the same pid (pids repeat across container restarts) from colliding with this one.
    temporary = os.path.join(directory, f".{name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    created = False
    replaced = False
    try:
        with open(temporary, "x", encoding="utf-8") as handle:  # "x": never write into an existing file
            created = True
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        # The rename is atomic: a reader sees the old file or the complete new one. The gate treats a
        # missing file as "still running" and any readable one as final, so a half-written file must not exist.
        os.replace(temporary, path)
        replaced = True
    finally:
        if created and not replaced:  # remove only a temporary this call created
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> None:
    """Write ``payload`` as JSON to ``path`` so that the file appears complete or not at all.

    The payload is serialised first, so a value that is not JSON (or NaN, infinity) raises before anything
    touches the disk. The parent directory must exist; it is not created here.
    """
    target = os.fspath(path)
    if not isinstance(target, str):
        raise TypeError(f"path must be a str path, got {path!r}")
    _atomic_write_text(target, _serialize(payload))


def write_artifacts(output_path: str, rank_payloads: Any) -> dict[str, Any]:
    """Write the rank files of one run, then the merged file.

    ``rank_payloads`` is the list for ranks ``0..dp_size-1`` (every rank of the run is written here, so all
    ranks are on one node), with the artifact of rank ``i`` at index ``i``; it goes to
    ``rank_output_path(output_path, i)``.

    Everything is checked and serialised before the first file is written, so an error leaves the disk
    untouched. "Checked" means the rank-set rules of ``_check_rank_set`` and, for ``complete`` payloads, the
    rules of ``_check_complete_payload``. A payload whose envelope was edited after ``build_rank_artifact``
    so that the validator would reject it raises ValueError here, naming the rank and the field. One whose
    decode steps were edited raises what ``build_fpm`` raises for the same fault (ValueError or TypeError),
    naming the rank, the group and the rank entry. The payloads should still come unchanged from
    ``build_rank_artifact``; an edit outside those rules is not looked for.

    Order on disk: rank files in DESCENDING rank order (rank 0 last), then the merged file. The gate reports
    "pending" while rank 0's file is missing and fails the pod at the first rank file that is not complete; with
    rank 0 last, every rank file of a partial or failed run is on disk before the gate can react. For a complete
    run the gate treats "rank files valid, merged missing" as still finalizing. A ``failed`` run writes no
    merged file.

    Returns ``{"rank_files": [...], "merged_output_path": str | None}``, with the paths in the form recorded
    inside the merged file.
    """
    payloads = _require_sequence(rank_payloads, "rank_payloads")
    size, ranks = _check_rank_set(payloads)
    if ranks != list(range(size)):
        raise ValueError(f"write_artifacts needs the artifacts of ranks {list(range(size))} in order, got {ranks}")
    rank_paths = [rank_output_path(output_path, rank) for rank in ranks]
    documents = [(path, _serialize(payload)) for path, payload in zip(rank_paths, payloads, strict=True)]
    merged_path: str | None = None
    if payloads[0]["status"] != STATUS_FAILED:
        merged = build_merged_artifact(payloads, rank_paths)
        merged_path = merged["merged_output_path"]
        documents.insert(0, (merged_path, _serialize(merged)))

    # "documents" is [merged, rank 0, rank 1, ...]; written back to front, the merged file lands last.
    for path, text in reversed(documents):
        _atomic_write_text(path, text)
    return {"rank_files": [_recorded_path(path) for path in rank_paths], "merged_output_path": merged_path}
