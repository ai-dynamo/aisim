# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# The feasibility rules and the first prefill graph step follow SGLang v0.5.22 and v0.5.23; see THIRD_PARTY_NOTICES.md.
# Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d
# Source: https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90
# The sample caps, the axis rules, the points manifest and the order and digest of the grid are re-implemented from
# ai-dynamo/dynamo (Apache-2.0, https://github.com/ai-dynamo/dynamo), components/src/dynamo/vllm/
# instrumented_scheduler.py, backend_args.py and benchmark_points.py at b20fe6fe94021b854597965c7305931175e8f482.
# See THIRD_PARTY_NOTICES.md.

"""Benchmark grid of the SGLang self-benchmark: limits, feasibility, axes, points, manifest and digest.

This module runs inside the SGLang pod, where AISim is not installed. It uses the standard library only, imports
no other ``sglbench_*`` module and reads no environment variable. It decides which points a run measures and in
which order. It measures nothing.

Symbols (every limit is per attention-DP rank)::

    P  page size                         S  KV pool tokens (max_total_num_tokens)
    C  chunk budget, None = chunking off M  max_prefill_tokens
    R  running-request cap               I  max_req_input_len
    b  requests in a point               n  new tokens per request (prefill only)
    k  cached tokens per request (prefill), or context per request (decode)
    ceilP(x), floorP(x)   x rounded up, down to a whole number of pages
    G  ceilP(16), the unit of the cached-token axis

Points. Every request of a point is the same. A prefill point is ``b`` requests of ``k`` cached tokens followed by
``n`` new tokens, with one output token. A decode point is ``b`` requests whose prompt has ``k - 3`` tokens, with
seven output tokens: six decode batches follow the extend batch, decode steps 3 to 5 are kept, and step 4 reads
``k`` past tokens per request. The artifact and the points manifest carry batch totals, ``total_prefill_tokens =
b * n`` and ``total_kv_read_tokens = b * k``.

Failures. A missing, non-integer, boolean or non-positive limit raises; nothing is coerced and nothing has a
fallback. Generated points fit by construction: no value of a generated axis is above the largest value that
fits the limits record (by size against the limits only, the collector's capacity rule), that largest value is
proven with the reason function, and the values below it fit because every rule is monotone in the searched
variable (see the comment above ``_largest_fitting``). Grid build checks every generated candidate once more.
One that does not fit is a bug in this module and raises ``RuntimeError``; it is never left out. A manifest
point that cannot fit raises ``ValueError`` here. After grid build no planned point is ever dropped.

Checked against SGLang v0.5.22 (baa9befa381511cd4d56f83980951e5a45fffc6d) and
v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90) of ``sgl-project/sglang`` (Apache-2.0), by reading these files
under ``python/sglang`` to check the arithmetic::

    srt/managers/tp_worker.py                TpModelWorker.get_worker_info
    srt/managers/utils.py                    validate_input_length
    srt/managers/scheduler.py                Scheduler.get_next_batch_to_run, Scheduler._get_new_batch_prefill_raw,
                                             Scheduler.get_num_allocatable_reqs, Scheduler.init_req_max_new_tokens
    srt/managers/schedule_policy.py          PrefillAdder.add_one_req, PrefillAdder._select_prefill_admission,
                                             PrefillAdder._update_prefill_budget, PrefillAdder.budget_state
    srt/mem_cache/prefill_budget.py          PrefillBudget.check_prefill, PrefillBudget.reserve
    srt/mem_cache/allocator/base.py          BaseTokenToKVPoolAllocator.max_new_tokens_for_memory
    srt/mem_cache/radix_cache.py             RadixKey.page_aligned, RadixKey.match_at
    srt/managers/schedule_batch.py           Req._compute_max_prefix_len
    srt/model_executor/pool_configurator.py  DefaultPoolConfigurator.calculate_pool_sizes
    srt/mem_cache/kv_cache_configurator.py   KVCacheConfigurator.resolve_max_num_reqs
    srt/arg_groups/validation_hook.py        check_server_args
    srt/arg_groups/cuda_graph_hook.py        generate_decode_cuda_graph_batch_sizes,
                                             generate_prefill_cuda_graph_batch_sizes
    srt/model_executor/runner/prefill_cuda_graph_runner.py
                                             _MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR,
                                             get_prefill_num_tokens_to_capture,
                                             PrefillCudaGraphRunner.can_replay_locally
    srt/model_executor/runner/base_cuda_graph_runner.py
                                             BaseCudaGraphRunner._pad_to_bucket

The candidate rule of the graph-driven axes is mirrored on the collector host in
``collector/fpm_forward/config.py`` (``_dynamo_cudagraph_axis_points``). The sample-cap flag names are the ones
``collector/fpm_forward/runner.py`` renders for the vLLM self-benchmark; it renders five of the six, not
``--benchmark-max-batch-size``. The six names and their defaults are those of the flags of Dynamo's vLLM
self-benchmark (``backend_args.py``). The three cap rules follow its ``instrumented_scheduler.py``: the uniform
cut is ``_uniformly_limit_axis``, the tail-protecting cut is ``_limit_cudagraph_axis`` and "the first N" is the
slice at the end of ``_bench_prefill_batch_sizes``. The header of this file gives the Dynamo revision. AISim has
no mirror of the defaults or of the cap rules.

No upstream function or block of source is copied. What follows SGLang and Dynamo is named in the header of this
file and recorded in THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

PREFILL = "prefill"
DECODE = "decode"
AGG = "agg"
MODES = (PREFILL, DECODE, AGG)

# Staging constants. The feasibility rules, the expected cache counts and the decode context all follow from them.
#
# A decode request is a prompt of ``k - DECODE_PROMPT_LEAD`` tokens with ``DECODE_OUTPUT_TOKENS`` output tokens.
# The extend batch samples the first output token and six decode batches follow. Decode step j reads
# ``k - 4 + j`` past tokens, so step 4, the middle one of the kept steps 3 to 5, reads exactly ``k``.
DECODE_PROMPT_LEAD = 3
DECODE_OUTPUT_TOKENS = 7
PREFILL_OUTPUT_TOKENS = 1

# Deviation note (lowest decode context).
# What differs: the lowest decode context is DECODE_PROMPT_LEAD + 1 = 4 tokens per request, because the prompt
# needs one token and three decode steps run before the middle kept step. Dynamo's vLLM producer goes down to 2
# tokens per request: its generated ladder starts at a batch total of ``2 * batch_size``
# (``_bench_decode_kv_read_points``), and a manifest point that holds a one-token context is recorded at the
# total its measured step read, with the stamp ``context_clamped`` (``_bench_step_decode``). Serving decodes at
# any context.
# Here a generated context below 4 does not exist and a manifest point below 4 raises (no clamping, no relabel).
# Effect: the decode table starts at ``total_kv_read_tokens = 4 * b``; below 4 tokens per request it has no row.
# AISim's reader (crates/core/src/perfmodel/operators/fpm_forward.rs) refuses such a query in direct
# interpolation (``FpmForwardOp::direct_decode_rows``). In its default interpolation it answers the query from
# the rows of other batches when one of them covers the query's total, and refuses it otherwise: a batch that
# was collected is answered by cross-batch transfer (crates/core/src/perfmodel/perf_database/perf_interp.rs
# ``SiteIndex::resolve_pair``), a batch between two collected rows from one bracket row
# (``FpmForwardOp::decode_bracket``). Such an answer is an estimate, not a measurement.
# How to retire it: keep an earlier decode step when the context is below 4.
MIN_DECODE_CONTEXT = 4

# Deviation note (page-aligned cached prefixes).
# What differs: a prefill point's cached prefix is a multiple of the page size P, and the cached-token axis steps
# in units of G = ceilP(16). Dynamo's vLLM producer steps in the engine's hash block: its ladder counts hash
# blocks for the whole batch, and the block size is taken from the engine and is not a constant
# (``_bench_prefill_kv_read_points``, ``_bench_hash_block_size``). SGLang's prefix cache stores and matches
# whole pages only (srt/mem_cache/radix_cache.py ``RadixKey.page_aligned`` and ``RadixKey.match_at``), so a
# cached count that is not a multiple of P cannot occur in serving and cannot be staged; the prefill rule
# refuses it. G is the smallest page multiple of at least 16 tokens: 16 itself when P divides 16, and a page
# multiple for every other P ("the larger of P and 16" would not be page aligned for a P such as 3 or 12). One
# block per request is the first step of Dynamo's ladder, so G equals that step when the engine's hash block
# has 16 tokens and P divides 16.
# Effect: no measured value changes. For P above 16 the axis steps in whole pages, which is coarser than 16
# tokens.
# How to retire it: nothing to retire while SGLang caches whole pages.
CACHED_UNIT_MIN_TOKENS = 16

MANIFEST_SCHEMA_VERSIONS = (1, 2, 3)

# ``sample_reasons`` entries. AISim's validator maps the two regime stamps to ``real_prefix`` and ``real_kv``.
STAMP_EXPLICIT = "explicit"
STAMP_PREFILL_REAL_SEED = "prefill_real_seed"
STAMP_DECODE_REAL_KV = "kvwarm_real_kv"

SOURCE_GENERATED = "generated"
SOURCE_MANIFEST = "manifest"

# Why a point does not fit: the name of the first failing rule, in the order the reason functions test them.
REASON_INVALID_SHAPE = "invalid_shape"
REASON_CACHED_NOT_PAGE_ALIGNED = "cached_not_page_aligned"
REASON_CONTEXT_BELOW_MINIMUM = "context_below_minimum"
REASON_INPUT_TOO_LONG = "input_too_long"
REASON_RUNNING_LIMIT = "running_limit"
REASON_CHUNK_BUDGET = "chunk_budget"
REASON_PREFILL_TOKEN_BUDGET = "prefill_token_budget"
REASON_KV_POOL = "kv_pool"

_LIMIT_SCALARS = (
    "page_size",
    "max_req_input_len",
    "max_total_num_tokens",
    "max_running_requests",
    "max_prefill_tokens",
)
_LIMIT_GRAPH_FIELDS = ("decode_graph_sizes", "prefill_graph_sizes")
_LIMIT_FIELDS = (*_LIMIT_SCALARS, "chunked_prefill_size", *_LIMIT_GRAPH_FIELDS)

# An error message does not carry whole a rejected value that can be of any type or size: such a value comes
# from a file or from a caller, and the message is what a caller logs or stores. It is shown through ``_shown``,
# and so is every number of a points manifest. Any other integer that a Python caller handed over, that passed
# its type check and that fails a range check or a page-multiple check (in ``Limits``, for example) is shown in
# full. All of this holds up to the interpreter's limit for converting an integer to text
# (``sys.get_int_max_str_digits()``, 4300 digits by default). An integer with more digits that a message would
# show, alone or inside another value, makes that conversion raise ``ValueError`` ("Exceeds the limit ...") in
# place of the module's own error: the message then names neither the argument nor the value, and it is a
# ``ValueError`` also where the module would raise ``TypeError``. Only a Python caller can hand over such an
# integer; in manifest text it is refused as invalid JSON.
_SHOWN_VALUE_CHARACTERS = 200
_SHOWN_KEYS = 8


def _shown(value: object) -> str:
    """``repr(value)`` for an error message: its first 200 characters and then its total length when longer."""
    text = repr(value)
    if len(text) <= _SHOWN_VALUE_CHARACTERS:
        return text
    return f"{text[:_SHOWN_VALUE_CHARACTERS]}... ({len(text)} characters)"


def _shown_keys(keys: Iterable[object]) -> str:
    """Rejected keys for an error message: the first 8 in the order of their ``repr``, then how many more."""
    shown = sorted(_shown(key) for key in keys)
    listed = ", ".join(shown[:_SHOWN_KEYS])
    if len(shown) <= _SHOWN_KEYS:
        return listed
    return f"{listed} and {len(shown) - _SHOWN_KEYS} more"


def _require_int(name: str, value: object) -> int:
    """Return ``value`` when it is exactly an ``int``. ``bool`` is an ``int`` subclass and is refused."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}: {_shown(value)}")
    return value


def _require_at_least(name: str, value: object, minimum: int) -> int:
    number = _require_int(name, value)
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {number}")
    return number


def _graph_sizes(name: str, values: object) -> tuple[int, ...]:
    """Validate a graph-size list of a limits record and return it sorted and de-duplicated."""
    if not isinstance(values, list | tuple):
        raise TypeError(f"{name} must be a list or tuple of ints, got {type(values).__name__}: {_shown(values)}")
    return tuple(sorted({_require_at_least(f"{name} entry", value, 1) for value in values}))


# ---------------------------------------------------------------------------------------------------------------
# Limits record
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Limits:
    """What one SGLang scheduler (one attention-DP rank) admits. Read off the running engine, never guessed.

    The grid is built once from one record; the caller must check that every rank reports the same one.
    Under DP attention the grid is therefore a grid of per-rank points: a point with batch ``b`` is ``b``
    requests on every attention-DP rank, and each limit is the one a rank's scheduler reports for itself.
    Construction validates and raises; the only conversion is that the two graph-size lists are stored as sorted,
    de-duplicated tuples.
    """

    # P. Unit of KV allocation and of prefix-cache matching.
    # SGLang: srt/managers/scheduler.py ``Scheduler.page_size`` (set from ``get_schedule().page_size``).
    page_size: int
    # I. A prompt of I tokens or more is refused; it also bounds the decode context (see the decode rule).
    # SGLang: srt/managers/tp_worker.py ``TpModelWorker.get_worker_info``: ``max_req_len = min(context_len - 1,
    # kv_capacity - 1)`` and ``max_req_input_len = max_req_len - 5``. Reported in the scheduler's init info.
    max_req_input_len: int
    # S. Tokens in the KV pool, always a whole number of pages.
    # SGLang: srt/model_executor/pool_configurator.py ``DefaultPoolConfigurator.calculate_pool_sizes`` floors it
    # to pages. Reported in the scheduler's init info as ``max_total_num_tokens``.
    max_total_num_tokens: int
    # R. Requests one scheduler runs at once. The cap SGLang applies is ``pp_max_micro_batch_size`` less the
    # requests already running, and never more than the free request rows (srt/managers/scheduler.py
    # ``Scheduler.get_num_allocatable_reqs``). That value equals the scheduler's ``max_running_requests`` only
    # when ``pp_max_micro_batch_size`` is unset and ``pp_size`` is 1: ``Scheduler.init_model_worker`` then sets
    # it to ``max(max_running_requests // pp_size, 1)``. Both are among the settings the feasibility rules
    # require (see the comment above ``prefill_unfit_reason``). ``max_running_requests`` itself is the per-rank
    # value after the DP division and the capacity clamp (srt/mem_cache/kv_cache_configurator.py
    # ``KVCacheConfigurator.resolve_max_num_reqs``).
    max_running_requests: int
    # M. Page-rounded new tokens one prefill pass takes before it stops adding requests.
    # SGLang: field ``max_prefill_tokens`` (srt/arg_groups/fields/schedule.py); it is the ``rem_input_tokens``
    # budget of srt/managers/schedule_policy.py ``PrefillAdder``.
    max_prefill_tokens: int
    # C. Page-rounded new tokens one extend batch holds before a request is split in two ("chunked").
    # ``None`` = chunking off. A multiple of P when set (srt/arg_groups/validation_hook.py ``check_server_args``).
    # SGLang: srt/managers/scheduler.py ``Scheduler.chunked_prefill_size``, where a resolved value <= 0 becomes
    # ``None``; it is the ``rem_chunk_tokens`` budget of ``PrefillAdder``. The caller must hand over ``None`` for
    # "off": a value <= 0 is refused here.
    chunked_prefill_size: int | None
    # Batch sizes that have a captured decode graph; empty = no decode graphs.
    # SGLang: srt/arg_groups/cuda_graph_hook.py ``generate_decode_cuda_graph_batch_sizes``; the captured list is
    # srt/model_executor/model_runner.py ``ModelRunner.decode_cuda_graph_capture_bs``.
    decode_graph_sizes: tuple[int, ...] = ()
    # Token counts that have a captured prefill graph; empty = no prefill graphs.
    # SGLang: srt/arg_groups/cuda_graph_hook.py ``generate_prefill_cuda_graph_batch_sizes``
    # (``cuda_graph_config.prefill.bs``); the captured list, which SGLang pads a batch to, is
    # srt/model_executor/runner/prefill_cuda_graph_runner.py ``PrefillCudaGraphRunner.capture_num_tokens``.
    prefill_graph_sizes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in _LIMIT_SCALARS:
            _require_at_least(name, getattr(self, name), 1)
        if self.chunked_prefill_size is not None:
            _require_at_least("chunked_prefill_size", self.chunked_prefill_size, 1)
            if self.chunked_prefill_size % self.page_size:
                raise ValueError(
                    f"chunked_prefill_size {self.chunked_prefill_size} is not a multiple of page_size {self.page_size}"
                )
        if self.max_total_num_tokens % self.page_size:
            raise ValueError(
                f"max_total_num_tokens {self.max_total_num_tokens} is not a multiple of page_size {self.page_size}"
            )
        for name in _LIMIT_GRAPH_FIELDS:
            # The dataclass is frozen; this is the one sanctioned way to store the normalized tuple.
            object.__setattr__(self, name, _graph_sizes(name, getattr(self, name)))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Limits:
        """Build the record from a mapping that has exactly the eight field names as keys.

        Every key must be present, the two graph-size keys and ``chunked_prefill_size`` (which may be ``None``)
        included. Values are validated as they are, never converted, except that a graph-size list or tuple is
        stored as a sorted, de-duplicated tuple. ``Limits.from_mapping(x.as_dict()) == x``.
        """
        if not isinstance(mapping, Mapping):
            raise TypeError(f"limits record must be a mapping, got {type(mapping).__name__}")
        missing = [name for name in _LIMIT_FIELDS if name not in mapping]
        unknown = [key for key in mapping if key not in _LIMIT_FIELDS]
        if missing or unknown:
            raise ValueError(
                f"limits record must have exactly the keys {list(_LIMIT_FIELDS)}: "
                f"missing {missing}, unknown [{_shown_keys(unknown)}]"
            )
        return cls(**{name: mapping[name] for name in _LIMIT_FIELDS})

    def as_dict(self) -> dict:
        """Plain JSON-ready form: ints, ``None`` and lists."""
        record: dict = {name: getattr(self, name) for name in _LIMIT_SCALARS}
        record["chunked_prefill_size"] = self.chunked_prefill_size
        for name in _LIMIT_GRAPH_FIELDS:
            record[name] = list(getattr(self, name))
        return record


def _require_limits(limits: object) -> Limits:
    if not isinstance(limits, Limits):
        raise TypeError(f"limits must be a Limits record, got {type(limits).__name__}")
    return limits


# ---------------------------------------------------------------------------------------------------------------
# Page arithmetic
# ---------------------------------------------------------------------------------------------------------------


def _ceil_pages(tokens: int, page_size: int) -> int:
    return -(-tokens // page_size) * page_size


def _floor_pages(tokens: int, page_size: int) -> int:
    return tokens // page_size * page_size


def ceil_pages(tokens: int, page_size: int) -> int:
    """``ceilP``: ``tokens`` rounded up to a whole number of pages."""
    return _ceil_pages(_require_int("tokens", tokens), _require_at_least("page_size", page_size, 1))


def floor_pages(tokens: int, page_size: int) -> int:
    """``floorP``: ``tokens`` rounded down to a whole number of pages."""
    return _floor_pages(_require_int("tokens", tokens), _require_at_least("page_size", page_size, 1))


def cached_unit(page_size: int) -> int:
    """``G``: the step of the prefill cached-token axis, the smallest page multiple of at least 16 tokens."""
    return ceil_pages(CACHED_UNIT_MIN_TOKENS, page_size)


# ---------------------------------------------------------------------------------------------------------------
# Feasibility: does a point run as ONE unchunked extend batch (prefill), or as six full decode batches (decode)?
#
# The rules describe an idle scheduler (nothing running, no half-prefilled request) of a plain attention model on
# an NVIDIA GPU. They rest on the engine settings and request properties listed here, and the caller must verify
# every one of them before it builds a grid. The SGLang symbols named were read at both tags; their paths are
# under ``python/sglang``.
#
# Engine settings:
# - radix cache on, ``schedule_policy`` "fcfs", ``prefill_decode_interval`` 0, ``prefill_max_requests`` unset
#   (or at least the largest batch), ``enable_mixed_chunk`` off, ``scheduler_recv_interval`` 1, no KV sharding;
# - ``pp_size`` 1 and ``pp_max_micro_batch_size`` unset: the running cap SGLang applies is
#   ``pp_max_micro_batch_size``, and it is ``max_running_requests`` only then (see the comment at
#   ``Limits.max_running_requests``);
# - ``max_queued_requests`` unset, or at least the largest batch: a request that would make the waiting queue
#   longer than that is aborted (srt/managers/scheduler.py ``Scheduler._abort_on_queued_limit``);
# - the min-free-slots delay off: ``min_free_slots_delay`` unset or 1, and no DFlash speculative decoding,
#   which switches the delay on when the flag is unset (srt/managers/min_free_slots_delayer.py
#   ``resolve_min_free_slots``). While requests run and fewer slots than its threshold are free, no prefill pass
#   starts (``MinFreeSlotsDelayer.should_delay``, called from ``Scheduler._get_new_batch_prefill_raw``), so a
#   decode point that needs several extend batches would start to decode without all ``b``;
# - the prefill delayer off (``enable_prefill_delayer``; the environment variable
#   ``SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE`` switches it on too, srt/arg_groups/serving_hook.py
#   ``handle_prefill_delayer_env_compat``): it can refuse a candidate that every rule here admits
#   (srt/managers/schedule_policy.py ``PrefillAdder.add_one_req``);
# - the environment variable ``SGLANG_MAX_NEW_TOKENS_LIMIT`` unset, or at least 7: it caps the output tokens
#   (``Scheduler.init_req_max_new_tokens``), and below 7 a decode run has fewer than six decode batches.
#
# Request properties: every request is sent with ``ignore_eos``, and without ``return_logprob``,
# ``logprob_start_len``, ``priority``, ``extra_key``, ``cache_salt``, ``input_embeds`` or
# ``positional_embed_overrides``. ``return_logprob`` with a ``logprob_start_len`` of 0 or more caps the matched
# prefix (srt/managers/schedule_batch.py ``Req._compute_max_prefix_len``), ``positional_embed_overrides``
# switches prefix matching off (``Req.init_next_round_input``) and ``input_embeds`` replaces the token ids by
# placeholders (``Scheduler._handle_generate_request``), so ``expected_cached_tokens`` would be wrong.
# ``extra_key`` and ``cache_salt`` are part of the cache key (srt/mem_cache/radix_cache.py ``RadixKey``). A
# priority changes the order of the waiting queue when priority scheduling is on (srt/managers/schedule_policy.py
# ``SchedulePolicy.calc_priority``) and can abort the request when it is off
# (``Scheduler._set_or_validate_priority``).
#
# Status of the evidence: every rule was read from the admission code at both tags. Some of them were also seen
# on running engines. The result files are in benchmarks/evidence/collector/sglbench-engine-probe-20261113
# (check_2, check_4 and check_9); every engine of these three checks reported SGLang version 0.5.22 or 0.5.23
# ("both tags" below; the files record the package version, not a commit), C = 8192, M = 16384 and R = 4096.
# - Chunk rule, only at its boundary with whole requests (check_9; both tags, Qwen3-0.6B, TP 1, page sizes 1
#   and 64): one call of 136 requests of 64 new tokens ran as an extend batch of 128 requests and 8192 tokens,
#   then one of 8 requests and 512 tokens. In check_2 (both tags, Qwen3-0.6B and Qwen3-8B) one call of 128
#   requests ran as one extend batch; those files hold the batch sizes and no token counts, and the 64 new
#   tokens of each request (128 * 64 = 8192 = C) are in the probe driver saved in that folder only.
# - Decode (check_4; both tags, Qwen3-0.6B and Qwen3-8B at page sizes 1 and 64 with TP 1, Qwen3-8B also with
#   TP 2 at page size 1): 2 and 8 requests at context 1024 ran as one extend batch and six decode batches. For
#   these runs the files hold the batch modes and no batch sizes; that every batch held every request is what
#   their field ``pass`` means in the probe driver saved in that folder.
# - Decode above the chunk budget (check_9; both tags, Qwen3-0.6B, TP 1, page size 64): 136 requests were
#   prefilled in extend batches of 128 and 8 and 384 requests in three of 128, and each of the six decode
#   batches that followed held all of them.
# Not covered by the result files: the input-length rule, the running-cap rule, a request cut by the chunk
# budget, chunking off, and a point at the boundary of the pool rule or of the ``max_prefill_tokens`` rule.
# ---------------------------------------------------------------------------------------------------------------


def prefill_unfit_reason(limits: Limits, batch_size: int, new_tokens: int, cached_tokens: int) -> str | None:
    """Name of the first rule a prefill point ``(b, n, k)`` breaks, or ``None`` when the point fits.

    "Fits" means: on an idle engine, SGLang puts all ``b`` requests in one extend batch and splits none of them.
    Arguments must be ``int`` and not ``bool`` (``TypeError`` otherwise). Integers outside the shape do not
    raise: they return ``invalid_shape``.
    """
    limits = _require_limits(limits)
    batch = _require_int("batch_size", batch_size)
    new = _require_int("new_tokens", new_tokens)
    cached = _require_int("cached_tokens", cached_tokens)
    page = limits.page_size

    # 0. A request computes at least one token; the prefix may be empty.
    if batch < 1 or new < 1 or cached < 0:
        return REASON_INVALID_SHAPE

    # 1. k % P == 0. The cache matches whole pages only (see the deviation note at CACHED_UNIT_MIN_TOKENS), so
    #    only a page-aligned prefix is reported as exactly k cached tokens.
    if cached % page:
        return REASON_CACHED_NOT_PAGE_ALIGNED

    # 2. k + n < I, strict. The prompt has k + n tokens and SGLang aborts a request whose prompt length is
    #    >= max_req_input_len (srt/managers/utils.py ``validate_input_length``).
    if cached + new >= limits.max_req_input_len:
        return REASON_INPUT_TOO_LONG

    # 3. b <= R. A prefill pass stops adding requests at the running cap (srt/managers/scheduler.py
    #    ``Scheduler._get_new_batch_prefill_raw`` with ``Scheduler.get_num_allocatable_reqs``).
    if batch > limits.max_running_requests:
        return REASON_RUNNING_LIMIT

    # SGLang charges the new tokens of a request to both step budgets rounded up to whole pages
    # (srt/managers/schedule_policy.py ``PrefillAdder._update_prefill_budget``).
    new_pages = _ceil_pages(new, page)

    # 4. b * ceilP(n) <= C, equality allowed. A request whose page-rounded new tokens exceed what is left of
    #    the chunk budget is cut in two and finishes in the next batch
    #    (``PrefillAdder._select_prefill_admission``). No rule when chunking is off.
    chunk = limits.chunked_prefill_size
    if chunk is not None and batch * new_pages > chunk:
        return REASON_CHUNK_BUDGET

    # 5. max_prefill_tokens. The budget M is checked AFTER a request is added, so it is a different rule from 4.
    #    Chunking on:  the pass ends once the admitted requests have used M or more (``PrefillAdder.budget_state``),
    #                  so request b gets in only if the b - 1 before it used less: (b - 1) * ceilP(n) < M.
    #    Chunking off: the first request always gets in; each later one only while the total, itself included,
    #                  stays below M (``PrefillAdder._select_prefill_admission``): b == 1 or b * ceilP(n) < M.
    budget = limits.max_prefill_tokens
    if chunk is not None:
        within_budget = (batch - 1) * new_pages < budget
    else:
        within_budget = batch == 1 or batch * new_pages < budget
    if not within_budget:
        return REASON_PREFILL_TOKEN_BUDGET

    # 6. b * (k + ceilP(n) + P + 1) < S, strict. Each admitted request holds its cached prefix (locked, so it can
    #    no longer be evicted) and reserves ceilP(n) + P + max_new_tokens pool tokens
    #    (srt/mem_cache/prefill_budget.py ``PrefillBudget.reserve``); a candidate is admitted only while its
    #    need is strictly below what remains (``PrefillBudget.check_prefill``, called from
    #    ``PrefillAdder.add_one_req``). SGLang's exact line for the last request is
    #    b * (k + P + 1) + (b - 1) * ceilP(n) + n < S. The form coded here charges the last request a whole
    #    page too: it is at most ceilP(n) - n tokens stricter and never admits a point SGLang would split.
    #    SGLang also clips ``max_new_tokens`` of every arriving request to S - ceilP(prompt) - P - 1
    #    (srt/mem_cache/allocator/base.py ``BaseTokenToKVPoolAllocator.max_new_tokens_for_memory``, called from
    #    ``Scheduler.init_req_max_new_tokens``). The prompt has k + n tokens and k is a page multiple, so the one
    #    output token survives the clip exactly when k + ceilP(n) + P + 1 < S: for one request the clip IS the
    #    rule coded here. The coded form must therefore not be relaxed to the admission line, which would admit
    #    a single request whose output SGLang has clipped away.
    per_request = cached + new_pages + page + PREFILL_OUTPUT_TOKENS
    if batch * per_request >= limits.max_total_num_tokens:
        return REASON_KV_POOL
    return None


def decode_unfit_reason(limits: Limits, batch_size: int, context_tokens: int) -> str | None:
    """Name of the first rule a decode point ``(b, k)`` breaks, or ``None`` when the point fits.

    "Fits" means: all ``b`` requests of ``k - 3`` prompt tokens are admitted without waiting for one another and
    every one of the six decode batches holds all ``b``. Arguments must be ``int`` and not ``bool``
    (``TypeError`` otherwise). Integers outside the shape return ``invalid_shape`` or ``context_below_minimum``.

    There is no chunk rule and no ``max_prefill_tokens`` rule: requests beyond those budgets are prefilled in
    further extend batches, SGLang prefers prefill over decode (srt/managers/scheduler.py
    ``Scheduler.get_next_batch_to_run``), and every decode batch still holds all ``b``. The engine probe
    (check_9) recorded this for the chunk budget at v0.5.22 and v0.5.23 with Qwen3-0.6B, TP 1, page size 64 and
    C = 8192: 384 requests were prefilled in three extend batches of 128 and 136 requests in batches of 128 and
    8, and each of the six decode batches that followed held all of them.
    """
    limits = _require_limits(limits)
    batch = _require_int("batch_size", batch_size)
    context = _require_int("context_tokens", context_tokens)
    page = limits.page_size

    # 0. At least one request.
    if batch < 1:
        return REASON_INVALID_SHAPE

    # 1. k >= 4: the prompt of k - 3 tokens needs one token (deviation note at MIN_DECODE_CONTEXT).
    if context < MIN_DECODE_CONTEXT:
        return REASON_CONTEXT_BELOW_MINIMUM

    # 2. k <= I. SGLang silently clips the output to max_req_len - prompt - 1 tokens (srt/managers/scheduler.py
    #    ``Scheduler.init_req_max_new_tokens``). All seven survive when 7 <= max_req_len - (k - 3) - 1, which is
    #    k <= max_req_len - 5 = max_req_input_len. Above that the run has fewer than six decode batches.
    if context > limits.max_req_input_len:
        return REASON_INPUT_TOO_LONG

    # 3. b <= R: the cap counts the requests that are already running (``Scheduler.get_num_allocatable_reqs``).
    if batch > limits.max_running_requests:
        return REASON_RUNNING_LIMIT

    # 4. b * (ceilP(k - 3) + P + 7) < S, strict. The same admission arithmetic as prefill rule 6 with seven
    #    output tokens: the prompt's cached part floorP(k - 4) plus its page-rounded remainder is ceilP(k - 3).
    #    It also covers a run prefilled over several extend batches, and it keeps the six decode steps from
    #    running out of pool (the peak is b * ceilP(k + 3) tokens, which is smaller).
    #    SGLang also clips ``max_new_tokens`` of every arriving request to S - ceilP(k - 3) - P - 1
    #    (srt/mem_cache/allocator/base.py ``BaseTokenToKVPoolAllocator.max_new_tokens_for_memory``, called from
    #    ``Scheduler.init_req_max_new_tokens``). All seven output tokens survive the clip exactly when
    #    ceilP(k - 3) + P + 7 < S: for one request the clip IS the rule coded here. The coded form must therefore
    #    not be relaxed to the admission line, which would admit a single request that SGLang has left with
    #    fewer than seven output tokens, so with fewer than six decode batches.
    per_request = _ceil_pages(context - DECODE_PROMPT_LEAD, page) + page + DECODE_OUTPUT_TOKENS
    if batch * per_request >= limits.max_total_num_tokens:
        return REASON_KV_POOL
    return None


def _require_phase(phase: object) -> str:
    if phase not in (PREFILL, DECODE):
        raise ValueError(f"phase must be {PREFILL!r} or {DECODE!r}, got {_shown(phase)}")
    return str(phase)


def expected_cached_tokens(phase: str, cached_tokens: int, page_size: int) -> int:
    """Cached tokens SGLang reports for ONE request of a staged point.

    Prefill: ``k``, which must be page-aligned. Decode: ``floorP(k - 4)``. The decode prompt has ``k - 3``
    tokens, SGLang always recomputes at least one token (srt/managers/schedule_batch.py
    ``Req._compute_max_prefix_len``: at most ``input_len - 1`` are cached), and the match is floored to whole
    pages.
    """
    page = _require_at_least("page_size", page_size, 1)
    cached = _require_int("cached_tokens", cached_tokens)
    if _require_phase(phase) == PREFILL:
        if cached < 0 or cached % page:
            raise ValueError(f"prefill cached_tokens must be a non-negative multiple of page_size {page}: {cached}")
        return cached
    if cached < MIN_DECODE_CONTEXT:
        raise ValueError(f"decode context must be >= {MIN_DECODE_CONTEXT}, got {cached}")
    return _floor_pages(cached - DECODE_PROMPT_LEAD - 1, page)


def expected_extend_tokens(phase: str, new_tokens: int, cached_tokens: int, page_size: int) -> int:
    """Tokens ONE request of a staged point computes in the extend batch.

    Prefill: ``n``. Decode: ``(k - 3) - floorP(k - 4)``, the part of the prompt that is not cached, between 1
    and P tokens; ``new_tokens`` must be 0 there, as it is on a decode ``Point``.
    """
    new = _require_int("new_tokens", new_tokens)
    cached = expected_cached_tokens(phase, cached_tokens, page_size)
    if phase == PREFILL:
        if new < 1:
            raise ValueError(f"prefill new_tokens must be >= 1, got {new}")
        return new
    if new != 0:
        raise ValueError(f"a decode point has no new-token coordinate: new_tokens must be 0, got {new}")
    return cached_tokens - DECODE_PROMPT_LEAD - cached


# ---------------------------------------------------------------------------------------------------------------
# Axis primitives
# ---------------------------------------------------------------------------------------------------------------


def _doubling(first: int, limit: int) -> list[int]:
    """``first, 2 * first, 4 * first, ...`` up to and including ``limit``."""
    values = []
    value = first
    while value <= limit:
        values.append(value)
        value *= 2
    return values


def _require_max_samples(max_samples: object) -> int:
    if _require_int("max_samples", max_samples) < 2:
        raise ValueError(f"an axis sample cap must be at least 2, got {max_samples}")
    return max_samples


def graph_axis(graph_sizes: Iterable[int], limit: int) -> list[int]:
    """Candidate values, ascending, of an axis driven by captured graph sizes, up to ``limit``.

    A graph of size ``c`` serves every value up to ``c`` by padding, so latency jumps between ``c`` and
    ``c + 1``. Sampling both sides of each size shows the reader where the steps are. Above the largest size
    nothing is padded and the rule adds a doubling tail without the ``+ 1`` values. That is the rule of Dynamo's
    vLLM self-benchmark (``_cudagraph_axis_points``) and it is kept, but for the decode batch axis it is not
    enough for AISim's reader in its default interpolation. The reader treats a collected batch as a rung only
    when ``batch + 1`` was collected too (crates/core/src/perfmodel/perf_database/fpm_forward.rs,
    ``decode_rungs`` of ``FpmForwardCell``), and it prices a batch above the last rung from the row after the
    last rung and the LAST collected row (crates/core/src/perfmodel/operators/fpm_forward.rs
    ``FpmForwardOp::decode_bracket_rows``). In that mode a tail row in between therefore answers a query at
    exactly its own batch size and no other; direct interpolation (``FpmForwardOp::direct_decode_rows``) does
    bracket with the tail rows. Collecting ``batch + 1`` beside each tail value, or a change in the reader,
    would remove this; neither is made here.

    - ``limit < 1``: no values.
    - No graph size (entries below 1 are ignored): the powers of two up to ``limit``, and ``limit``.
    - Otherwise: every size ``c <= limit``; ``c + 1`` for every size ``c < limit``; the doubling tail
      ``2 * largest, 4 * largest, ...`` below ``limit`` (there is none when the largest size is above the
      limit); and always ``limit``.

    AISim's collector host has the same rule as ``collector.fpm_forward.config._dynamo_cudagraph_axis_points``
    (it cannot be imported in the pod); a contract test compares the two on fixed and random cases with limits
    from 1 to 200.
    """
    limit = _require_int("limit", limit)
    sizes = sorted({size for size in graph_sizes if _require_int("graph size", size) >= 1})
    if limit < 1:
        return []
    if not sizes:
        return sorted({*_doubling(1, limit), limit})
    values = {limit}
    for size in sizes:
        if size <= limit:
            values.add(size)
        if size < limit:
            values.add(size + 1)
    # The doubling tail above the largest size. It is empty when that size is already above the limit.
    values.update(_doubling(2 * sizes[-1], limit))
    return sorted(values)


def limit_uniform(values: Sequence[int], max_samples: int) -> list[int]:
    """Cut an ascending axis to ``max_samples`` values, evenly spaced by POSITION, both ends always kept.

    With ``L`` values and ``m = max_samples < L``, sample ``s`` (``0 .. m - 1``) is the value at index
    ``(s * (L - 1) + (m - 1) // 2) // (m - 1)``: the position ``s * (L - 1) / (m - 1)`` rounded to the nearest
    index, halves up. The result has exactly ``m`` values and is strictly increasing when the input is.
    """
    last_sample = _require_max_samples(max_samples) - 1
    axis = list(values)
    if len(axis) <= max_samples:
        return axis
    last_index = len(axis) - 1
    return [axis[(sample * last_index + last_sample // 2) // last_sample] for sample in range(max_samples)]


def limit_graph_axis(values: Sequence[int], graph_sizes: Iterable[int], max_samples: int) -> list[int]:
    """Cut a graph-driven axis to ``max_samples`` values without losing its tail above the largest graph size.

    The values above the largest graph size are the only samples of the unpadded region. When they are few (at
    most a fifth of the axis) they are all kept and the cap is spent on the graph part; otherwise the whole axis
    is cut with ``limit_uniform``. Exactly:

    - at most ``max_samples`` values: unchanged;
    - no graph size ``>= 1``: ``limit_uniform``;
    - split at ``top = max(graph_sizes)`` into ``graph_part`` (``<= top``) and ``tail`` (``> top``). When the
      tail is non-empty, ``len(tail) * 5 <= len(values)``, the graph part is non-empty and
      ``budget = max_samples - len(tail) >= 1``: the graph part cut to ``budget`` values (its first value alone
      when ``budget == 1``), then the whole tail;
    - in every other case ``limit_uniform`` over the whole axis.
    """
    _require_max_samples(max_samples)
    axis = list(values)
    sizes = [size for size in graph_sizes if _require_int("graph size", size) >= 1]
    if len(axis) <= max_samples:
        return axis
    if not sizes:
        return limit_uniform(axis, max_samples)
    top = max(sizes)
    graph_part = [value for value in axis if value <= top]
    tail = [value for value in axis if value > top]
    budget = max_samples - len(tail)
    if tail and len(tail) * 5 <= len(axis) and graph_part and budget >= 1:
        limited = [graph_part[0]] if budget == 1 else limit_uniform(graph_part, budget)
        return limited + tail
    return limit_uniform(axis, max_samples)


# ---------------------------------------------------------------------------------------------------------------
# Sample caps
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleCaps:
    """How many values each generated axis keeps.

    Names and defaults are those of the flags of Dynamo's vLLM self-benchmark (``backend_args.py`` at the revision
    the header of this file names).

    The two graph-driven axes (prefill new tokens and decode batch) always start at 1. When the cut axis does
    not hold 1, it is put in front AFTER the cut and does not count against the cap, so the decode batch axis
    can hold one value more than its cap. The prefill new-token axis also gets the two sides of its first
    step in that way (``prefill_new_token_axis``), so it can hold up to three values more than its cap.

    A points manifest replaces the generated grid, and then every cap here is ignored.
    """

    # --prefill-max-new-token-samples: prefill new-token axis (``limit_graph_axis``); up to cap + 3 values with
    # the low end added. It bounds the values of ``n`` PER REQUEST, not the batch totals: every ``n`` is planned
    # at up to ``prefix_max_batch_size_samples`` batch sizes, so a grid can hold more distinct
    # ``total_prefill_tokens`` values than this cap. In Dynamo the flag of this name caps an axis of batch totals
    # (``_bench_generate_prefill_grid``).
    prefill_max_new_token_samples: int = 64
    # --prefill-max-kv-read-token-samples: prefill cached-token axis, per (n, b) (``limit_uniform``).
    prefill_max_kv_read_token_samples: int = 16
    # --decode-max-kv-read-token-samples: decode context axis, per batch size (``limit_uniform``).
    decode_max_kv_read_token_samples: int = 128
    # --decode-max-batch-size-samples: decode batch axis (``limit_graph_axis``); cap + 1 values when 1 is added.
    decode_max_batch_size_samples: int = 128
    # --prefix-max-batch-size-samples: prefill batch axis, per n. Despite the name it is the PREFILL batch axis,
    # and it is not a uniform cut: the first N values are kept.
    prefix_max_batch_size_samples: int = 3
    # --benchmark-max-batch-size: upper end of the decode batch axis. It does not apply to prefill.
    benchmark_max_batch_size: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "prefill_max_new_token_samples",
            "prefill_max_kv_read_token_samples",
            "decode_max_kv_read_token_samples",
            "decode_max_batch_size_samples",
        ):
            _require_at_least(name, getattr(self, name), 2)
        _require_at_least("prefix_max_batch_size_samples", self.prefix_max_batch_size_samples, 1)
        if self.benchmark_max_batch_size is not None:
            _require_at_least("benchmark_max_batch_size", self.benchmark_max_batch_size, 1)


def _require_caps(caps: object) -> SampleCaps:
    if not isinstance(caps, SampleCaps):
        raise TypeError(f"caps must be a SampleCaps record, got {type(caps).__name__}")
    return caps


# ---------------------------------------------------------------------------------------------------------------
# Largest feasible values
#
# Each search below is exact. The value comes from a closed form, obtained by solving the rules of the reason
# function for the searched variable, and is then PROVEN with the reason function itself. The proof is sound
# because every rule is monotone in the searched variable: if a value does not fit, no larger one does.
# ---------------------------------------------------------------------------------------------------------------


def _largest_fitting(
    name: str, minimum: int, step: int, closed_form: Callable[[], int], fits: Callable[[int], bool]
) -> int | None:
    """Largest candidate ``minimum, minimum + step, ...`` that fits, or ``None`` when none does.

    When the smallest candidate does not fit, nothing does. Otherwise the closed form must name a candidate that
    fits and whose successor does not. A closed form that fails this is a bug in this module, so it raises:
    the result is never repaired and never silently dropped.
    """
    if not fits(minimum):
        return None
    value = closed_form()
    if value < minimum or (value - minimum) % step or not fits(value) or fits(value + step):
        raise RuntimeError(
            f"{name}: closed form gave {value}, which is not the largest fitting value "
            f"(minimum {minimum}, step {step}); this is a bug in the grid arithmetic"
        )
    return value


def largest_prefill_new_tokens(limits: Limits) -> int | None:
    """``n_limit``: the largest ``n`` for which one request of ``n`` new tokens and no cached prefix fits."""
    limits = _require_limits(limits)
    page = limits.page_size

    def closed_form() -> int:
        # Rules for (b=1, n, k=0): n < I; ceilP(n) <= C, and C is a page multiple, so n <= C;
        # ceilP(n) + P + 1 < S, and ceilP(n) is a page multiple, so n <= floorP(S - P - 2).
        bounds = [
            limits.max_req_input_len - 1,
            _floor_pages(limits.max_total_num_tokens - page - PREFILL_OUTPUT_TOKENS - 1, page),
        ]
        if limits.chunked_prefill_size is not None:
            bounds.append(limits.chunked_prefill_size)
        return min(bounds)

    return _largest_fitting("n_limit", 1, 1, closed_form, lambda new: prefill_unfit_reason(limits, 1, new, 0) is None)


def largest_prefill_batch(limits: Limits, new_tokens: int) -> int | None:
    """``b_max(n)``: the largest ``b`` for which ``b`` requests of ``n`` new tokens and no cached prefix fit."""
    limits = _require_limits(limits)
    new = _require_int("new_tokens", new_tokens)
    page = limits.page_size

    def closed_form() -> int:
        # With w = ceilP(n), the rules for (b, n, k=0) give: b <= R; b * (w + P + 1) < S;
        # chunking on: b * w <= C and (b - 1) * w < M; chunking off: b == 1 or b * w < M.
        new_pages = _ceil_pages(new, page)
        bounds = [
            limits.max_running_requests,
            (limits.max_total_num_tokens - 1) // (new_pages + page + PREFILL_OUTPUT_TOKENS),
        ]
        if limits.chunked_prefill_size is not None:
            bounds.append(limits.chunked_prefill_size // new_pages)
            bounds.append((limits.max_prefill_tokens - 1) // new_pages + 1)
        else:
            bounds.append(max(1, (limits.max_prefill_tokens - 1) // new_pages))
        return min(bounds)

    return _largest_fitting(
        "prefill b_max", 1, 1, closed_form, lambda batch: prefill_unfit_reason(limits, batch, new, 0) is None
    )


def largest_prefill_cached(limits: Limits, batch_size: int, new_tokens: int) -> int | None:
    """``k_max(n, b)``: the largest multiple of ``G`` that fits as the cached prefix of ``(b, n)``.

    ``None`` when ``(b, n)`` does not fit even without a prefix; otherwise at least 0.
    """
    limits = _require_limits(limits)
    batch = _require_int("batch_size", batch_size)
    new = _require_int("new_tokens", new_tokens)
    page = limits.page_size
    unit = _ceil_pages(CACHED_UNIT_MIN_TOKENS, page)

    def closed_form() -> int:
        # Only two rules involve k: k + n < I, and b * (k + ceilP(n) + P + 1) < S. G is a page multiple, so a
        # multiple of G is page-aligned.
        bound = min(
            limits.max_req_input_len - new - 1,
            (limits.max_total_num_tokens - 1) // batch - _ceil_pages(new, page) - page - PREFILL_OUTPUT_TOKENS,
        )
        return bound // unit * unit

    return _largest_fitting(
        "prefill k_max", 0, unit, closed_form, lambda cached: prefill_unfit_reason(limits, batch, new, cached) is None
    )


def largest_decode_batch(limits: Limits) -> int | None:
    """``b_limit``: the largest ``b`` for which ``b`` requests at the lowest context (4 tokens) fit.

    ``None`` when not even one fits, for example when ``max_req_input_len`` is below 4. The
    ``benchmark_max_batch_size`` cap is not applied here; ``decode_batch_axis`` applies it afterwards.
    """
    limits = _require_limits(limits)
    page = limits.page_size

    def closed_form() -> int:
        # Rules for (b, k=4), given that one request fits (so 4 <= I): b <= R; b * (ceilP(1) + P + 7) < S.
        lowest_prompt_pages = _ceil_pages(MIN_DECODE_CONTEXT - DECODE_PROMPT_LEAD, page)
        per_request = lowest_prompt_pages + page + DECODE_OUTPUT_TOKENS
        return min(limits.max_running_requests, (limits.max_total_num_tokens - 1) // per_request)

    return _largest_fitting(
        "decode b_limit",
        1,
        1,
        closed_form,
        lambda batch: decode_unfit_reason(limits, batch, MIN_DECODE_CONTEXT) is None,
    )


def largest_decode_context(limits: Limits, batch_size: int) -> int | None:
    """``k_max(b)``: the largest context ``k`` for which ``b`` decode requests fit; ``None`` when none does."""
    limits = _require_limits(limits)
    batch = _require_int("batch_size", batch_size)
    page = limits.page_size

    def closed_form() -> int:
        # k <= I; b * (ceilP(k - 3) + P + 7) < S, and ceilP(k - 3) is a page multiple, so
        # k - 3 <= floorP((S - 1) // b - P - 7).
        pool_bound = _floor_pages((limits.max_total_num_tokens - 1) // batch - page - DECODE_OUTPUT_TOKENS, page)
        return min(limits.max_req_input_len, pool_bound + DECODE_PROMPT_LEAD)

    return _largest_fitting(
        "decode k_max",
        MIN_DECODE_CONTEXT,
        1,
        closed_form,
        lambda context: decode_unfit_reason(limits, batch, context) is None,
    )


# ---------------------------------------------------------------------------------------------------------------
# Generated axes. All axes are per request and ascending; the grid multiplies by b to get the totals.
# ---------------------------------------------------------------------------------------------------------------


def _starting_at_one(axis: list[int]) -> list[int]:
    """A non-empty ascending axis of positive values, with 1 put in front unless 1 is already its first value."""
    return axis if axis[0] == 1 else [1, *axis]


# SGLang replays a captured prefill graph for a batch only when the size it pads the batch to holds at most
# this many times the batch's own tokens; a batch below that runs without a graph
# (``_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR`` and the last statements of ``can_replay_locally`` in
# python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py, which are the same at v0.5.22 and v0.5.23).
PREFILL_GRAPH_PADDING_FACTOR = 2


def first_prefill_graph_tokens(graph_sizes: Iterable[int]) -> int | None:
    """The smallest token count of a batch that SGLang runs with its smallest prefill graph; None without one.

    With the smallest captured size ``c`` a batch of ``t`` tokens is padded to ``c`` and replays that graph when
    ``c <= PREFILL_GRAPH_PADDING_FACTOR * t``, so the count is ``ceil(c / PREFILL_GRAPH_PADDING_FACTOR)``. A
    smaller batch runs without a graph. For SGLang's default sizes, which start at 4, the count is 2: a batch
    of one new token runs without a graph and a batch of two with the graph of 4.
    """
    sizes = [size for size in graph_sizes if _require_int("graph size", size) >= 1]
    if not sizes:
        return None
    return -(-min(sizes) // PREFILL_GRAPH_PADDING_FACTOR)


def prefill_new_token_axis(limits: Limits, caps: SampleCaps) -> list[int]:
    """New tokens per request: the low end, each prefill graph size ``c`` and ``c + 1``, a doubling tail, ``n_limit``.

    The graph axis is cut to ``prefill_max_new_token_samples`` values first. The values of the low end are put on
    the axis afterwards when they are not among them, so they do not count against the cap and the axis can hold
    up to three values more than the cap (two with SGLang's default graph sizes). AISim's collector host sizes
    this cap as the number of candidates of its own mirror of ``graph_axis``
    (collector/fpm_forward/config.py ``PrefillSamplingProfile.build``), which has no low end; a cap sized that
    way must not cut a graph value.

    The low end is 1 and, when prefill graphs are captured, the two sides of the first step of the table:
    ``first - 1`` and ``first``, where ``first`` is ``first_prefill_graph_tokens``. Below ``first`` a batch runs
    without a graph, and from ``first`` on with the smallest one, which is another time: on H200 a batch of one
    new token took 36 to 48 ms and a batch of 4 took 5 to 7 ms in the collected MoE cells, and 20 ms against 6 ms
    for a small dense model. ``graph_axis`` samples both sides of the steps between graph sizes, and this is the
    one step below the smallest size. With SGLang's default sizes the low end is 1 and 2. Without the 2 the
    table held 1 and 4, and AISim's reader priced a batch of 2 or 3 new tokens on the line between a row
    without a graph and a row with one: in serving of 2026-11-23 such steps were predicted 3.7 times too slow.
    SGLang keys a prefill graph on the batch's TOTAL new tokens (``b * n``), so the graph sizes and the low end
    line up with this per-request axis exactly at ``b = 1``.

    Not sampled, and neither can happen with SGLang's default sizes:

    - A step inside the graph sizes, where one size is more than ``PREFILL_GRAPH_PADDING_FACTOR`` times the
      size before it and the batches just above the smaller one run without a graph. The default sizes have
      no such pair.
    - The first step of the rows of two or more requests, when the smallest size is 7 or more. With a
      smallest size of 8 the low end is 1, 3 and 4: two requests of one new token run without a graph and
      two requests of two with it, and the table holds two requests of one and of three. With the default
      smallest size of 4 every batch of two or more requests runs with a graph.
    """
    caps = _require_caps(caps)
    n_limit = largest_prefill_new_tokens(limits)
    if n_limit is None:
        return []
    sizes = limits.prefill_graph_sizes
    axis = limit_graph_axis(graph_axis(sizes, n_limit), sizes, caps.prefill_max_new_token_samples)
    # 1 is always on the axis: it is the lower end of the table. AISim's reader refuses a query outside the
    # collected range of an axis and never extrapolates, SGLang's default prefill graph sizes start at 4, and a
    # prefill step of 1 to 3 new tokens is ordinary serving (at page size 1 a fully cached prompt recomputes
    # exactly one token).
    low_end = {1}
    first = first_prefill_graph_tokens(sizes)
    if first is not None:
        low_end.update(value for value in (first - 1, first) if 1 <= value <= n_limit)
    return sorted({*axis, *low_end})


def prefill_batch_axis(limits: Limits, caps: SampleCaps, new_tokens: int) -> list[int]:
    """Batch sizes for one ``n``: the powers of two up to ``b_max(n)`` and ``b_max(n)``, then the FIRST N kept.

    ``b_max`` is added BEFORE the first ``prefix_max_batch_size_samples`` values are kept, so it is on the axis
    only when it is among them. With the default of three the axis is 1, 2, 4 when ``b_max >= 4``; 1, 2, 3 when
    ``b_max == 3``; 1, 2 when ``b_max == 2``; and 1 when ``b_max == 1``. ``benchmark_max_batch_size`` does not
    apply to prefill.
    """
    caps = _require_caps(caps)
    b_max = largest_prefill_batch(limits, new_tokens)
    if b_max is None:
        return []
    candidates = sorted({*_doubling(1, b_max), b_max})
    return candidates[: caps.prefix_max_batch_size_samples]


def prefill_cached_axis(limits: Limits, caps: SampleCaps, batch_size: int, new_tokens: int) -> list[int]:
    """Cached tokens per request for one ``(b, n)``: 0, ``G * 2**j`` up to ``k_max``, and ``k_max``.

    The cap keeps both ends, so 0 and ``k_max`` are always on the axis. When not even ``G`` fits, ``k_max`` is 0
    and the axis is ``[0]``.
    """
    caps = _require_caps(caps)
    k_max = largest_prefill_cached(limits, batch_size, new_tokens)
    if k_max is None:
        return []
    unit = _ceil_pages(CACHED_UNIT_MIN_TOKENS, limits.page_size)
    candidates = sorted({0, *_doubling(unit, k_max), k_max})
    return limit_uniform(candidates, caps.prefill_max_kv_read_token_samples)


def decode_batch_axis(limits: Limits, caps: SampleCaps) -> list[int]:
    """Decode batch sizes: 1, each decode graph size ``c`` and ``c + 1``, a doubling tail, and the largest batch.

    The graph axis is cut to ``decode_max_batch_size_samples`` values first. 1 is put in front afterwards when it
    is not among them, so it does not count against the cap and the axis can hold one value more than the cap.
    """
    caps = _require_caps(caps)
    b_limit = largest_decode_batch(limits)
    if b_limit is None:
        return []
    if caps.benchmark_max_batch_size is not None:
        b_limit = min(b_limit, caps.benchmark_max_batch_size)
    # A graph size above the running cap can never be the size of a batch.
    sizes = [size for size in limits.decode_graph_sizes if size <= limits.max_running_requests]
    axis = limit_graph_axis(graph_axis(sizes, b_limit), sizes, caps.decode_max_batch_size_samples)
    # 1 is always on the axis. A decode graph ladder that starts above 1 would otherwise give an axis that
    # starts above 1, and AISim's reader refuses a batch below the collected range. Beside a graph size 2 the
    # row at 1 makes the reader take 1 for a rung (crates/core/src/perfmodel/perf_database/fpm_forward.rs,
    # ``decode_rungs`` of ``FpmForwardCell``: a batch whose ``batch + 1`` was collected), but no whole batch
    # lies between 1 and 2, so no query gets a different bracket.
    return _starting_at_one(axis)


def decode_context_axis(limits: Limits, caps: SampleCaps, batch_size: int) -> list[int]:
    """Contexts per request for one ``b``: the powers of two from 4 up to ``k_max``, and ``k_max``.

    The cap keeps both ends, so the 4-token context and ``k_max`` are always on the axis.
    """
    caps = _require_caps(caps)
    k_max = largest_decode_context(limits, batch_size)
    if k_max is None:
        return []
    candidates = sorted({*_doubling(MIN_DECODE_CONTEXT, k_max), k_max})
    return limit_uniform(candidates, caps.decode_max_kv_read_token_samples)


# ---------------------------------------------------------------------------------------------------------------
# Points, digest, grid
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Point:
    """One planned benchmark point, in per-request terms.

    ``new_tokens`` is ``n`` for prefill and 0 for decode. ``cached_tokens`` is the cached prefix ``k`` for
    prefill and the context ``k`` for decode. ``explicit`` marks a point that came from a points manifest.
    """

    benchmark_id: int
    phase: str
    batch_size: int
    new_tokens: int
    cached_tokens: int
    explicit: bool = False

    def __post_init__(self) -> None:
        _require_phase(self.phase)
        _require_at_least("benchmark_id", self.benchmark_id, 1)
        _require_at_least("batch_size", self.batch_size, 1)
        if self.phase == PREFILL:
            _require_at_least("prefill new_tokens", self.new_tokens, 1)
            _require_at_least("prefill cached_tokens", self.cached_tokens, 0)
        else:
            if _require_int("decode new_tokens", self.new_tokens) != 0:
                raise ValueError(f"decode new_tokens must be 0, got {self.new_tokens}")
            _require_at_least("decode cached_tokens", self.cached_tokens, 1)
        if type(self.explicit) is not bool:
            raise TypeError(f"explicit must be a bool, got {type(self.explicit).__name__}: {_shown(self.explicit)}")

    @property
    def total_prefill_tokens(self) -> int:
        """New tokens of the whole batch, ``b * n``; 0 for decode."""
        return self.batch_size * self.new_tokens

    @property
    def total_kv_read_tokens(self) -> int:
        """Cached tokens (prefill) or context tokens (decode) of the whole batch, ``b * k``."""
        return self.batch_size * self.cached_tokens

    def sample_reasons(self) -> list[str]:
        """``"explicit"`` first for a manifest point, then the one regime stamp that applies, if any.

        A prefill point with a cached prefix is staged from a real seeded prefix; every decode point reads real
        KV. A prefill point without a prefix has no seed regime and gets no stamp.
        """
        reasons = [STAMP_EXPLICIT] if self.explicit else []
        if self.phase == DECODE:
            reasons.append(STAMP_DECODE_REAL_KV)
        elif self.cached_tokens > 0:
            reasons.append(STAMP_PREFILL_REAL_SEED)
        return reasons

    def as_dict(self) -> dict:
        """The point as the artifact carries it: batch totals and exactly these six keys.

        No graph fields (``expected_cudagraph_mode``, ``expected_capture_size``, ``padding_tokens``), no
        ``partition`` and no ``rows``: for prefill they would be claims about a graph keyed by ``b * n``.
        """
        return {
            "point_type": self.phase,
            "benchmark_id": self.benchmark_id,
            "total_prefill_tokens": self.total_prefill_tokens,
            "total_kv_read_tokens": self.total_kv_read_tokens,
            "batch_size": self.batch_size,
            "sample_reasons": self.sample_reasons(),
        }


def grid_digest(point_dicts: Sequence[Mapping[str, object]]) -> str:
    """SHA-256 over the numbered point dicts in execution order, as compact JSON with sorted keys.

    Deterministic, and it changes when any point, id or the order changes. AISim only requires a non-empty
    string that is equal across rank files and in the merged file.
    """
    return hashlib.sha256(json.dumps(list(point_dicts), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Grid:
    """The planned points of a run, in execution order, with ids ``1..N``."""

    points: tuple[Point, ...]
    digest: str
    # Phases the mode asked for that have no point, "prefill" before "decode". Any entry fails the run.
    missing_phases: tuple[str, ...]
    # SOURCE_GENERATED or SOURCE_MANIFEST.
    source: str

    def point_dicts(self) -> list[dict]:
        """The artifact form of every point, in execution order."""
        return [point.as_dict() for point in self.points]


def _generated_prefill(limits: Limits, caps: SampleCaps) -> list[tuple[int, int, int]]:
    """``(b, n, k)`` of the generated prefill block: new tokens, then batch, then cached tokens, all descending.

    Inside one ``(n, b)`` group the cached prefix descends: the group's longest prefix ``k_max(n, b)`` comes
    first, and a seed of that length holds every shorter prefix of the group. Across groups the prefix does not
    descend. ``k_max(n, b)`` is never smaller for a smaller ``n`` or a smaller ``b`` and is usually larger, so
    the first point of the block need not be its deepest one and the prefix a slot needs can grow through the
    block. The caller must therefore extend the seed of every slot at the first point of every group; it may
    never assume that an earlier point left the prefix a point needs.

    Every candidate fits by construction: no axis value is above a largest value proven with the reason
    function, and every rule is monotone in the searched variable. A candidate that does not fit is a bug in
    this module and raises ``RuntimeError``; it is never left out.
    """
    shapes = []
    for new in reversed(prefill_new_token_axis(limits, caps)):
        for batch in reversed(prefill_batch_axis(limits, caps, new)):
            for cached in reversed(prefill_cached_axis(limits, caps, batch, new)):
                reason = prefill_unfit_reason(limits, batch, new, cached)
                if reason is not None:
                    raise RuntimeError(
                        f"generated prefill point (batch_size={batch}, new_tokens={new}, cached_tokens={cached}) "
                        f"does not fit: reason={reason}; this is a bug in the grid axes"
                    )
                shapes.append((batch, new, cached))
    return shapes


def _generated_decode(limits: Limits, caps: SampleCaps) -> list[tuple[int, int, int]]:
    """``(b, 0, k)`` of the generated decode block: batch, then context, descending.

    Inside one batch size the context descends. Across batch sizes it does not: the longest context ``k_max(b)``
    is never smaller for a smaller ``b`` and is usually larger, so the first point of the block need not be its
    deepest one. The caller must therefore extend the seed of every slot at the first point of every batch size.

    Every candidate fits by construction, as in ``_generated_prefill``; one that does not fit raises
    ``RuntimeError``.
    """
    shapes = []
    for batch in reversed(decode_batch_axis(limits, caps)):
        for context in reversed(decode_context_axis(limits, caps, batch)):
            reason = decode_unfit_reason(limits, batch, context)
            if reason is not None:
                raise RuntimeError(
                    f"generated decode point (batch_size={batch}, context_tokens={context}) "
                    f"does not fit: reason={reason}; this is a bug in the grid axes"
                )
            shapes.append((batch, 0, context))
    return shapes


def _explicit_shapes(phase: str, entries: Sequence[tuple[int, ...]], limits: Limits) -> list[tuple[int, int, int]]:
    """``(b, n, k)`` of one manifest phase, in file order. A point that cannot be staged raises."""
    shapes = []
    for index, entry in enumerate(entries):
        label = f"{phase}[{index}]"
        if phase == PREFILL:
            batch, total_new, total_kv = entry
            point = {"batch_size": batch, "total_prefill_tokens": total_new, "total_kv_read_tokens": total_kv}
        else:
            batch, total_kv = entry
            total_new = 0
            point = {"batch_size": batch, "total_kv_read_tokens": total_kv}
        # Deviation from Dynamo's vLLM producer: it splits a total as evenly as it can (``_balanced_partition``),
        # so that two requests differ by at most one token in new tokens and in decode context, and by at most
        # one hash block in cached prefill tokens. Here every request of a point is the same, so a total must
        # divide by the batch size.
        if total_new % batch or total_kv % batch:
            raise ValueError(
                f"{label}: uniform requests only: totals are not divisible by batch_size: point={_shown(point)}"
            )
        new, cached = total_new // batch, total_kv // batch
        if phase == PREFILL:
            reason = prefill_unfit_reason(limits, batch, new, cached)
        else:
            reason = decode_unfit_reason(limits, batch, cached)
        if reason is not None:
            raise ValueError(
                f"{label}: explicit benchmark point is infeasible: reason={reason} point={_shown(point)} "
                f"limits={limits.as_dict()}"
            )
        shapes.append((batch, new, cached))
    return shapes


def build_grid(mode: str, limits: Limits, caps: SampleCaps | None = None, manifest: Mapping | None = None) -> Grid:
    """Plan the points of a run.

    ``mode`` is ``"prefill"``, ``"decode"`` or ``"agg"`` (the prefill block, then the decode block).
    ``manifest`` is the result of ``parse_points_manifest``, or ``None`` for the generated grid.

    Generated grid: the axes above, cut by ``caps`` (default ``SampleCaps()``), every axis descending (the order
    is described at ``_generated_prefill`` and ``_generated_decode``). Every generated point fits; a candidate
    that does not is a bug in this module and raises ``RuntimeError``.
    Manifest: only the lists of the mode's phases are used, in FILE order; ``caps`` is ignored; a point whose
    totals do not divide by its batch size, or that does not fit, raises ``ValueError`` with its ``phase[index]``.

    Either way ``benchmark_id`` runs ``1..N`` in execution order, continuous across the two blocks of ``agg``,
    and a phase of the mode without any point is listed in ``missing_phases``.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {_shown(mode)}")
    limits = _require_limits(limits)
    caps = SampleCaps() if caps is None else _require_caps(caps)
    explicit = manifest is not None
    if explicit:
        manifest = _checked_manifest(manifest)

    points: list[Point] = []
    missing_phases = []
    for phase in (PREFILL, DECODE):
        if mode not in (phase, AGG):
            continue
        if explicit:
            shapes = _explicit_shapes(phase, manifest[phase], limits)
        elif phase == PREFILL:
            shapes = _generated_prefill(limits, caps)
        else:
            shapes = _generated_decode(limits, caps)
        if not shapes:
            missing_phases.append(phase)
        for batch, new, cached in shapes:
            points.append(Point(len(points) + 1, phase, batch, new, cached, explicit))

    # AISim raises at publication on two samples at one (phase, batch, total new, total kv). The axes are
    # strictly increasing and the manifest parser refuses duplicates, so a duplicate here is a bug.
    seen: dict[tuple, int] = {}
    for point in points:
        coordinate = (point.phase, point.batch_size, point.total_prefill_tokens, point.total_kv_read_tokens)
        if coordinate in seen:
            raise RuntimeError(
                f"benchmark points {seen[coordinate]} and {point.benchmark_id} share the coordinate {coordinate}"
            )
        seen[coordinate] = point.benchmark_id

    return Grid(
        points=tuple(points),
        digest=grid_digest([point.as_dict() for point in points]),
        missing_phases=tuple(missing_phases),
        source=SOURCE_MANIFEST if explicit else SOURCE_GENERATED,
    )


# ---------------------------------------------------------------------------------------------------------------
# What the KV pool takes from the generated prefill block
#
# The generated axes end where the first rule ends them, and nothing in a result file says which rule that was.
# For the prefill block it is usually the input limit and the two step budgets. When it is the KV pool, the
# table reaches less far than the other limits allow and the run still completes: the longest cached prefixes
# of the largest batches are not planned, and the number of points can stay the same. A pool can be small
# because SGLang sized its memory share for more than the engine uses (the deviation note on the static memory
# of a DEP cell, collector/fpm_forward/sglang_fpm_builder.py), so the launcher says when this happens.
#
# Only the prefill block is looked at. With SGLang's default request cap and no cap on the batch size the decode
# block reaches the pool in an ordinary run: its largest batch at the input limit is more than the pool holds,
# so the pool is what ends its context axis, and a line about it would say nothing. With a request cap set on
# the command, or with --benchmark-max-batch-size, it may or may not, and nothing says which.
# ---------------------------------------------------------------------------------------------------------------

# How far SGLang keeps the input limit below a pool that is not above the model's context:
# ``max_req_input_len = min(context_len - 1, kv_capacity - 1) - 5`` (``Limits.max_req_input_len``). Read at
# both tags. ``kv_capacity`` is the reported pool except in three engines, none of which reaches this code: a
# hybrid SWA model (the driver refuses it from the header), logical-page sharding of the KV pool and a diffusion
# model, which also takes its block size from the input limit (the launcher refuses both).
_INPUT_LIMIT_BELOW_POOL = 6


@dataclass(frozen=True)
class PoolCut:
    """What the KV pool takes from the generated prefill block. A "pair" is one (new tokens, batch size).

    The pairs counted are those of the block that the other limits alone give. The planned block can hold
    pairs that are not among them, for example batch 3 where batch 4 no longer fits, so ``pairs`` less
    ``dropped_pairs`` is not the number of planned pairs.
    """

    # S of the limits the block was planned for: the pool of one scheduler, so of one attention-DP rank.
    pool_tokens: int
    # Pairs of the block that the other limits alone give.
    pairs: int
    # Of those, pairs without any point in the planned block.
    dropped_pairs: int
    # Of those, pairs in the planned block whose longest cached prefix there is shorter.
    shortened_pairs: int
    # The smallest pool, in tokens, that holds every point of the block that the other limits alone give.
    whole_grid_pool_tokens: int
    # The input limit is the pool less 6, which SGLang gives exactly when the pool is not above the model's
    # context. The counts and ``whole_grid_pool_tokens`` are for the grid of this input limit. If the pool is
    # below the context, SGLang took the input limit from the pool, and a larger pool raises the input limit
    # too, which can lengthen the grid. If the pool equals the context it does not. The limits cannot tell
    # the two apart.
    pool_sets_input_limit: bool


def prefill_pool_cut(limits: Limits, caps: SampleCaps | None = None) -> PoolCut | None:
    """What the KV pool takes from the generated prefill block, or ``None`` when it takes nothing.

    The block is generated for ``limits`` and for the same record with a pool that no point can reach. The
    answer is ``None`` when the two blocks are the same list of points, so that the pool ends no planned axis,
    and the pool is above the model's context.

    The other limits are taken as reported. SGLang lowers ``max_req_input_len`` itself when the pool is below
    the model's context. The block "without the pool" can then already be shorter than a larger pool would
    give, also when the two blocks are the same, so a pool that is not above the context always gives a record,
    which says so in ``pool_sets_input_limit``. Its two blocks can both be empty.
    """
    limits = _require_limits(limits)
    caps = SampleCaps() if caps is None else _require_caps(caps)
    page = limits.page_size
    # Rule 6 of ``prefill_unfit_reason`` holds b * (k + ceilP(n) + P + 1) below S. Rules 2 and 3 keep k + n
    # below I and b at or below R, and ceilP(n) is below n + P, so the left side stays below
    # R * (I + 2 * P). A pool of R * (I + 2 * P + 1) tokens, rounded up to whole pages, is therefore out of
    # reach of every point, and of every closed form of the largest-value searches.
    out_of_reach = _ceil_pages(limits.max_running_requests * (limits.max_req_input_len + 2 * page + 1), page)
    without_pool = _generated_prefill(replace(limits, max_total_num_tokens=out_of_reach), caps)
    planned = _generated_prefill(limits, caps)
    sets_input_limit = limits.max_req_input_len + _INPUT_LIMIT_BELOW_POOL == limits.max_total_num_tokens
    if planned == without_pool and not sets_input_limit:
        return None

    # The cached axis of a pair always holds its longest prefix (``prefill_cached_axis``).
    def reach(block: list[tuple[int, int, int]]) -> dict[tuple[int, int], int]:
        longest: dict[tuple[int, int], int] = {}
        for batch, new, cached in block:
            longest[(new, batch)] = max(longest.get((new, batch), 0), cached)
        return longest

    wanted, kept = reach(without_pool), reach(planned)
    # Rule 6 for every point of the block without the pool: the largest left side, plus one, in whole pages.
    # An input limit that admits no request leaves both blocks empty: nothing is held.
    extra = page + PREFILL_OUTPUT_TOKENS
    held = max((batch * (cached + _ceil_pages(new, page) + extra) for batch, new, cached in without_pool), default=0)
    return PoolCut(
        pool_tokens=limits.max_total_num_tokens,
        pairs=len(wanted),
        dropped_pairs=sum(1 for pair in wanted if pair not in kept),
        shortened_pairs=sum(1 for pair, longest in wanted.items() if pair in kept and kept[pair] < longest),
        whole_grid_pool_tokens=_ceil_pages(held + 1, page),
        pool_sets_input_limit=sets_input_limit,
    )


# ---------------------------------------------------------------------------------------------------------------
# Points manifest (--benchmark-points-file), schema versions 1 to 3
# ---------------------------------------------------------------------------------------------------------------

_MANIFEST_KEYS = ("schema_version", PREFILL, DECODE)
_PREFILL_REQUIRED = ("batch_size", "total_prefill_tokens", "total_kv_read_tokens")
_PREFILL_OPTIONAL = ("partition", "rows")
_DECODE_REQUIRED = ("batch_size", "total_kv_read_tokens")
# AISim's manifest freezer lets a decode point carry ``total_prefill_tokens: 0`` on purpose; nothing else.
_DECODE_OPTIONAL = ("total_prefill_tokens",)


class _ManifestRefusal(ValueError):
    """What the two ``json.loads`` hooks below raise. It never leaves this module.

    ``parse_points_manifest`` passes its message on as a plain ``ValueError`` and reports every other
    ``ValueError`` of ``json.loads`` as invalid JSON. The hooks see one object or one constant and not its
    place, so the message names the manifest and neither the phase nor the index.
    """


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    """``object_pairs_hook``: a JSON object as a dict, refusing a key that appears twice."""
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise _ManifestRefusal(f"benchmark points manifest: duplicate key {_shown(key)}")
        result[key] = value
    return result


def _refuse_constant(name: str) -> None:
    """``parse_constant`` hook: ``NaN``, ``Infinity`` and ``-Infinity`` are not JSON."""
    raise _ManifestRefusal(f"benchmark points manifest: the constant {name} is not allowed")


def _manifest_int(label: str, name: str, value: object) -> int:
    """A manifest number. The manifest is data, so a wrong type is a ``ValueError`` that names the point."""
    if type(value) is not int:
        raise ValueError(f"{label}: {name} must be an integer, got {_shown(value)}")
    return value


def _prefill_coordinate(label: str, batch: object, total_new: object, total_kv: object) -> tuple[int, int, int]:
    batch = _manifest_int(label, "batch_size", batch)
    total_new = _manifest_int(label, "total_prefill_tokens", total_new)
    total_kv = _manifest_int(label, "total_kv_read_tokens", total_kv)
    if batch < 1:
        raise ValueError(f"{label}: batch_size must be >= 1, got {_shown(batch)}")
    if total_new < batch:
        raise ValueError(
            f"{label}: total_prefill_tokens must be >= batch_size, got {_shown(total_new)} for {_shown(batch)}"
        )
    if total_kv < 0:
        raise ValueError(f"{label}: total_kv_read_tokens must be >= 0, got {_shown(total_kv)}")
    return batch, total_new, total_kv


def _decode_coordinate(label: str, batch: object, total_kv: object) -> tuple[int, int]:
    batch = _manifest_int(label, "batch_size", batch)
    total_kv = _manifest_int(label, "total_kv_read_tokens", total_kv)
    if batch < 1:
        raise ValueError(f"{label}: batch_size must be >= 1, got {_shown(batch)}")
    if total_kv < batch:
        raise ValueError(
            f"{label}: total_kv_read_tokens must be >= batch_size, got {_shown(total_kv)} for {_shown(batch)}"
        )
    return batch, total_kv


def _refuse_duplicate(phase: str, index: int, coordinate: tuple[int, ...], seen: dict[tuple[int, ...], int]) -> None:
    """AISim cannot publish two samples at one coordinate, so a repeated point is refused at its later index."""
    if coordinate in seen:
        raise ValueError(f"{phase}[{index}]: duplicate of {phase}[{seen[coordinate]}]: {_shown(coordinate)}")
    seen[coordinate] = index


def _point_keys(label: str, point: Mapping, required: tuple[str, ...], optional: tuple[str, ...]) -> None:
    unknown = [key for key in point if key not in required and key not in optional]
    if unknown:
        raise ValueError(f"{label}: unknown key(s) {_shown_keys(unknown)}")
    missing = [key for key in required if key not in point]
    if missing:
        raise ValueError(f"{label}: missing key(s) {', '.join(missing)}")


def _refuse_uneven_rows(label: str, version: int, rows: object, coordinate: tuple[int, int, int]) -> None:
    """``rows`` lists ``[new, cached]`` per request. Only rows that restate the even split are accepted."""
    batch, total_new, total_kv = coordinate
    if version != 3:
        raise ValueError(f"{label}: rows needs schema_version 3, got {version}")
    even_split = [total_new // batch, total_kv // batch]
    uniform = (
        isinstance(rows, list)
        and len(rows) == batch
        and total_new % batch == 0
        and total_kv % batch == 0
        and all(
            isinstance(row, list) and len(row) == 2 and all(type(value) is int for value in row) and row == even_split
            for row in rows
        )
    )
    if not uniform:
        raise ValueError(
            f"{label}: rows must be {_shown(batch)} rows that all equal [total_prefill_tokens // batch_size, "
            f"total_kv_read_tokens // batch_size] = {_shown(even_split)}; uneven requests are not supported"
        )


def parse_points_manifest(raw: str | bytes) -> dict:
    """Check a points manifest against its schema and return its coordinates. No limits are involved.

    Returns ``{"schema_version": v, "prefill": [(b, total_new, total_kv), ...], "decode": [(b, total_kv), ...]}``
    in file order. ``bytes`` must be UTF-8 without a byte-order mark, as AISim's manifest freezer reads the file
    (collector/fpm_forward/config.py ``_freeze_benchmark_points``).

    Raises ``ValueError``. These errors are about the document as a whole; their message starts with
    ``benchmark points manifest`` and names no phase and no index:

    - invalid JSON, which includes bytes in another encoding and an integer literal longer than the
      interpreter's digit limit;
    - a key that appears twice in any object, and a ``NaN`` or ``Infinity`` constant. Both are refused while the
      JSON is read, when the place is not known yet: the message names the key or the constant, not the point
      it is in;
    - a top level that is not an object with exactly ``schema_version``, ``prefill`` and ``decode``;
    - a ``schema_version`` that is not the integer 1, 2 or 3.

    Every other error names the phase and, for a point, its zero-based index (``prefill[3]: ...``):

    - a phase that is not a list of objects;
    - a prefill point with an unknown key, without one of ``batch_size``, ``total_prefill_tokens``,
      ``total_kv_read_tokens``, or with a value that is not an ``int`` (a ``bool`` or a float is not), or with
      ``batch_size < 1``, ``total_prefill_tokens < batch_size`` or ``total_kv_read_tokens < 0``;
    - a ``partition`` that is not ``null`` (an uneven split; not supported);
    - ``rows`` that are not ``null``, unless the schema version is 3 and every row equals the even split;
    - a decode point with an unknown key, without ``batch_size`` or ``total_kv_read_tokens``, with a non-``int``
      value, with ``batch_size < 1`` or ``total_kv_read_tokens < batch_size``, or with a
      ``total_prefill_tokens`` that is not the integer 0;
    - a point that repeats an earlier point of its phase (the error names the later index).

    A rejected value is shown in the message cut to about 200 characters, and unknown keys as the first eight
    and a count. Both lists may be empty. Whether a point fits, and whether its totals divide by its batch size,
    is decided by ``build_grid``, which knows the limits and which phases the mode uses.
    """
    if not isinstance(raw, str | bytes):
        raise TypeError(f"benchmark points manifest must be str or bytes, got {type(raw).__name__}")
    try:
        # Bytes are decoded strictly as UTF-8 first. Given bytes, ``json.loads`` would detect UTF-16, UTF-32 and
        # a byte-order mark by itself and accept files that the freezer refuses.
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        document = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_refuse_constant)
    except _ManifestRefusal as refusal:
        raise ValueError(str(refusal)) from None
    except (ValueError, RecursionError) as error:
        # ``ValueError`` is the base of ``json.JSONDecodeError`` and ``UnicodeDecodeError``, and it is what
        # ``json.loads`` itself raises for an integer literal longer than the interpreter's digit limit.
        raise ValueError(f"benchmark points manifest is not valid JSON: {error}") from error

    if not isinstance(document, dict) or set(document) != set(_MANIFEST_KEYS):
        raise ValueError(
            "benchmark points manifest must be an object with exactly the keys schema_version, prefill and decode"
        )
    version = document["schema_version"]
    if type(version) is not int or version not in MANIFEST_SCHEMA_VERSIONS:
        raise ValueError(f"benchmark points manifest schema_version must be 1, 2 or 3, got {_shown(version)}")

    parsed: dict = {"schema_version": version, PREFILL: [], DECODE: []}
    for phase in (PREFILL, DECODE):
        if not isinstance(document[phase], list):
            raise ValueError(f"{phase}: must be a list of point objects")
        seen: dict[tuple[int, ...], int] = {}
        for index, point in enumerate(document[phase]):
            label = f"{phase}[{index}]"
            if not isinstance(point, dict):
                raise ValueError(f"{label}: must be an object, got {_shown(point)}")
            if phase == PREFILL:
                _point_keys(label, point, _PREFILL_REQUIRED, _PREFILL_OPTIONAL)
                coordinate = _prefill_coordinate(label, *(point[key] for key in _PREFILL_REQUIRED))
                # A key that is present with the value null is the same as an absent key.
                if point.get("partition") is not None:
                    raise ValueError(f"{label}: partition is not supported (it asks for uneven requests)")
                if point.get("rows") is not None:
                    _refuse_uneven_rows(label, version, point["rows"], coordinate)
                _refuse_duplicate(phase, index, coordinate, seen)
            else:
                _point_keys(label, point, _DECODE_REQUIRED, _DECODE_OPTIONAL)
                coordinate = _decode_coordinate(label, *(point[key] for key in _DECODE_REQUIRED))
                if "total_prefill_tokens" in point:
                    value = point["total_prefill_tokens"]
                    if type(value) is not int or value != 0:
                        raise ValueError(
                            f"{label}: a decode point's total_prefill_tokens must be 0, got {_shown(value)}"
                        )
                # With or without ``total_prefill_tokens: 0`` it is the same point.
                _refuse_duplicate(phase, index, coordinate, seen)
            parsed[phase].append(coordinate)
    return parsed


def _checked_manifest(manifest: object) -> dict:
    """Re-check the parsed form handed to ``build_grid``; it may have crossed a module or JSON boundary.

    Accepts lists as well as tuples for the coordinates and returns them as tuples.
    """
    if not isinstance(manifest, Mapping) or set(manifest) != set(_MANIFEST_KEYS):
        raise ValueError("manifest must be the result of parse_points_manifest: schema_version, prefill and decode")
    version = manifest["schema_version"]
    if type(version) is not int or version not in MANIFEST_SCHEMA_VERSIONS:
        raise ValueError(f"manifest schema_version must be 1, 2 or 3, got {_shown(version)}")
    checked: dict = {"schema_version": version, PREFILL: [], DECODE: []}
    for phase, width in ((PREFILL, 3), (DECODE, 2)):
        if not isinstance(manifest[phase], list | tuple):
            raise ValueError(f"{phase}: must be a list of coordinates")
        seen: dict[tuple[int, ...], int] = {}
        for index, entry in enumerate(manifest[phase]):
            label = f"{phase}[{index}]"
            if not isinstance(entry, list | tuple) or len(entry) != width:
                raise ValueError(f"{label}: must be a coordinate of {width} integers, got {_shown(entry)}")
            coordinate = _prefill_coordinate(label, *entry) if phase == PREFILL else _decode_coordinate(label, *entry)
            _refuse_duplicate(phase, index, coordinate, seen)
            checked[phase].append(coordinate)
    return checked
