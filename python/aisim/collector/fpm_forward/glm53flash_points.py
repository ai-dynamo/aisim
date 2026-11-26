# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The GLM-5.3-Flash FPM point list, generated from the onboarding limits and the accepted profile.

The GLM-5.3-Flash producer (``runtime/glm53flash/glm53flash_prefix_scheduler.py``) measures an explicit point list
only, and onboarding passes grid limits only. The planner freezes this list into the collection plan, so the plan
hash covers it. The list keeps the shape of Dynamo's generic grid where the producer's rules allow it.

``glm_align_stop`` and ``glm_point_blocks`` are modified adaptations of rules of vLLM v0.33.0
(https://github.com/vllm-project/vllm/tree/4ba787ec9959e566d3b587a796dff07673324da5), Copyright contributors to the
vLLM project, Apache-2.0. The comments in the two functions name the source files and lines, and
THIRD_PARTY_NOTICES.md holds the entry.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from collector.glm53flash_runtime_identity import BASELINE_VERSIONS

from .config import _powers_of_two_up_to

if TYPE_CHECKING:
    from aisim_core.sdk.fpm_profile import FpmDeploymentProfile

    from .config import FPMCollectionOptions
    from .types import ParallelTopology

logger = logging.getLogger(__name__)

GLM_ARCHITECTURE = "Glm5NextForConditionalGeneration"
GLM_POINTS_ORIGIN = "glm53flash_grid_v1"
GLM_SMOKE_MANIFEST = "glm53flash_grid_v1_smoke"
# The producer's CHUNK_ALIGN, MAX_BATCH, MAX_NEW, MAX_CONTEXT and TP_SIZES. The producer module imports vLLM and Dynamo,
# so the planner keeps copies.
CHUNK_ALIGN = 4
TP_SIZES = (1, 2, 4, 8)
MAX_BATCH = 32
MAX_NEW = 8192
MAX_CONTEXT = 131072
# The smallest power of two above the producer's minimum decode context of 3.
MIN_CONTEXT = 4


def _manifest(payload: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    # The canonical form of a user point file, so the runner stages and checks this list like one.
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return payload, canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def glm_measured_context(options: FPMCollectionOptions) -> int:
    """The deepest per-request context that the producer measures.

    vLLM needs seven positions beyond it, the producer's ``CONTEXT_HEADROOM``.
    """
    return min(MAX_CONTEXT, options.vllm_max_model_len - 7)


def glm_align_stop(
    start: int, end: int, block: int, *, hash_block: int, checkpoint_alignment: int | None
) -> int | None:
    """Where vLLM 0.33.0 in Mamba align mode stops the last prefill chunk from ``start`` to the prompt end ``end``.

    It holds without EAGLE, without hash lookahead tokens and without partial hash hits. None means one step.
    """
    # The last cacheable position is (end - 1) // block * block (scheduler.py:579-584, kv_cache_interface.py:1169-1186).
    # With a prefill checkpoint block of ``checkpoint_alignment``, a chunk that passes is_mamba_prefill_checkpoint_valid
    # saves the checkpoint inside it (kv_cache_interface.py:1189-1210) and takes neither that stop nor the stop at the
    # next boundary after a start inside a block (scheduler.py:593-606, :644-651). The chunk stops at the first
    # remaining stop strictly inside it (:661). The rule leaves out the shared-prefix junction stop (:632-658). vLLM
    # sets a junction only when the prefix hit of one cache group is longer than the hit that the request uses
    # (kv_cache_manager.py:319-325). The chunk then starts before the planned prefix, and the producer rejects a
    # repetition whose prefix differs from the planned one.
    last_cache_position = (end - 1) // block * block
    internal = (
        checkpoint_alignment is not None
        and start % hash_block == 0
        and -(-end // block) - 2 > (start - 1) // block
        and start + hash_block <= last_cache_position
        and start < last_cache_position < end
        and (last_cache_position - start) % checkpoint_alignment == 0
    )
    if internal:
        return None
    stops = (last_cache_position, (start // block + 1) * block if start % block else 0)
    return min((stop for stop in stops if start < stop < end), default=None)


def glm_point_blocks(
    workload_kind: str,
    point: dict[str, int],
    *,
    hash_block_size: int,
    state_groups: Sequence[tuple[int, int]],
    checkpoint_alignment: int | None,
) -> int:
    """Return the cache blocks that the requests of one point hold, for the deployment that needs the most.

    ``state_groups`` holds, for each deployment, the number of its ring groups and of its Mamba groups. The count is
    the one of vLLM 0.33.0's allocator when it admits a request, for the launch that the collector renders: prefix
    caching, the Mamba mode align, no speculative block and no async scheduling. It holds for a prefill that runs in
    one chunk and a decode whose suffix is at most one hash block, as every point of the list does. A request holds the
    attention blocks of its tokens and one block in each ring group and each Mamba group. A request with a prefix holds
    one more block in each Mamba group, the cached block of its prefix beside its running block, and so does a decode
    request at a context of at least one hash block. A prefill of more than one hash block of new tokens in a launch
    with a prefill checkpoint block (``checkpoint_alignment``) also holds that block in each Mamba group.
    """
    # Mamba groups: single_type_kv_cache_manager.py:1740-1795 caps the new blocks of a request at its running block,
    # the block of a partial prefix hit and the checkpoint block, and adds the cached block of the prefix. The
    # checkpoint block counts only when the chunk saves a checkpoint inside it (kv_cache_interface.py:1189-1210),
    # which needs more than one hash block of new tokens. A decode step takes no checkpoint block. The ring takes one
    # block (kv_cache_interface.py:982-989).
    unit = hash_block_size
    batch = point["batch_size"]
    context = point["total_kv_read_tokens"] // batch
    if workload_kind == "prefill":
        new = point["total_prefill_tokens"] // batch
        attention = -(-(context + new) // unit)
        mamba = 1 + (context > 0) + (checkpoint_alignment is not None and new > unit)
    else:
        # A decode step at a context of x tokens holds x + 1 tokens: the context and the token that it writes.
        attention = -(-(context + 1) // unit)
        mamba = 1 + (context >= unit)
    return batch * (attention + max(ring + mamba_groups * mamba for ring, mamba_groups in state_groups))


def glm_points(
    *,
    hash_block_size: int,
    token_budget: int,
    prefill_batch_limit: int,
    decode_batch_limit: int,
    measured_context: int,
    kv_cache_bytes: int,
    page_size_bytes: int,
    state_groups: Sequence[tuple[int, int]],
    checkpoint_alignment: int | None,
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    """Return the point list and, for each phase, its candidates and the number of them that each rule dropped.

    Every point meets the producer's rules. Prefixes are whole hash blocks. New-token and prefix lengths are multiples
    of 4. Prefix plus new tokens stay within the measured context. The new tokens of a batch fit the token budget, so
    each request's suffix runs in one step. vLLM's Mamba align mode does not stop the prefill of a request inside its
    prompt either (``glm_align_stop``): without a prefill checkpoint block, the new tokens of a request are at most one
    hash block. With one, ``checkpoint_alignment`` decides which longer prefills run in one step. At default serving the
    hash block of GLM-5.3-Flash is its Mamba block (vLLM 0.33.0 interface.py:1114-1115, kv_cache_utils.py:885-891).
    Before a decode batch decodes, each request prefills its prompt beyond its cached seed. These suffixes of a batch
    of two or more requests fit the token budget too, so every request decodes in the same step. The producer divides
    a total evenly, so every request of a point gets the same prefix, the same new tokens and the same decode context.
    The capacity rule keeps a point whose blocks by ``glm_point_blocks`` fit the usable blocks of the cache.
    """
    if token_budget % CHUNK_ALIGN:
        raise ValueError("GLM-5.3-Flash collection requires a token budget that is a multiple of 4")
    if hash_block_size % CHUNK_ALIGN:
        raise ValueError("GLM-5.3-Flash collection requires a hash block size that is a multiple of 4")
    # The producer measures at most MAX_NEW new tokens in a prefill point. It checks decode suffixes against the whole
    # token budget.
    prefill_budget = min(token_budget, MAX_NEW)
    measured = min(measured_context, MAX_CONTEXT)
    unit = hash_block_size
    usable = kv_cache_bytes // page_size_bytes

    def fits(workload_kind: str, point: dict[str, int]) -> bool:
        blocks = glm_point_blocks(
            workload_kind,
            point,
            hash_block_size=unit,
            state_groups=state_groups,
            checkpoint_alignment=checkpoint_alignment,
        )
        return blocks <= usable

    def batch_sizes(limit: int) -> list[int]:
        return sorted({*_powers_of_two_up_to(limit), limit})

    def decode_suffix(context: int) -> int:
        # The producer's decode prompt (_glm_decode_prompt) has context - 1 tokens. Its seed (_glm_decode_seed) is the
        # largest whole hash block prefix strictly shorter than the prompt, so a repetition prefills the rest.
        prompt = context - 1
        return prompt - (prompt - 1) // unit * unit

    new_tokens = {prefill_budget, *range(unit, prefill_budget + 1, unit)}
    new_tokens.update(value for value in _powers_of_two_up_to(prefill_budget) if value >= CHUNK_ALIGN)
    prefill = []
    prefill_candidates = split = 0
    for new in sorted(new_tokens):
        blocks = (measured - new) // unit
        rungs = sorted(rung for rung in {0, 1, blocks, *_powers_of_two_up_to(blocks)} if 0 <= rung <= blocks)
        # A prefill that vLLM's Mamba align mode stops inside the prompt runs in two steps, and the producer measures
        # one.
        whole = [
            rung
            for rung in rungs
            if glm_align_stop(
                rung * unit, rung * unit + new, unit, hash_block=unit, checkpoint_alignment=checkpoint_alignment
            )
            is None
        ]
        for batch in batch_sizes(min(MAX_BATCH, prefill_batch_limit, prefill_budget // new)):
            prefill_candidates += len(rungs)
            split += len(rungs) - len(whole)
            for rung in whole:
                point = {
                    "batch_size": batch,
                    "total_prefill_tokens": batch * new,
                    "total_kv_read_tokens": batch * rung * unit,
                }
                if fits("prefill", point):
                    prefill.append(point)
    contexts = {measured, *(value for value in _powers_of_two_up_to(measured) if value >= MIN_CONTEXT)}
    # The producer accepts a decode repetition only if its first decode step holds every request. vLLM prefills the
    # suffixes of two or more requests in one step only if they fit the token budget together, so the budget bounds the
    # batch at each context. One request may prefill its suffix over several steps: no other request can decode before
    # it, and the producer records no step with a prefill request.
    decode_limit = min(MAX_BATCH, decode_batch_limit)
    decode = []
    decode_candidates = over_budget = 0
    for context in sorted(contexts):
        limit = min(decode_limit, max(1, token_budget // decode_suffix(context)))
        # The candidates are the batches without the budget and the largest batch that the budget holds.
        candidates = sorted({*batch_sizes(decode_limit), limit})
        decode_candidates += len(candidates)
        over_budget += sum(batch > limit for batch in candidates)
        for batch in candidates:
            point = {"batch_size": batch, "total_kv_read_tokens": batch * context}
            if batch <= limit and fits("decode", point):
                decode.append(point)
    dropped = {
        "prefill": {
            "candidates": prefill_candidates,
            "mamba_align_split": split,
            "cache_capacity": prefill_candidates - split - len(prefill),
        },
        "decode": {
            "candidates": decode_candidates,
            "token_budget": over_budget,
            "cache_capacity": decode_candidates - over_budget - len(decode),
        },
    }
    # The rules remove points before any is queued, so a removal is never silent: one line counts the points that each
    # rule drops against the candidates that it sees, also when it drops none.
    logger.info(
        "GLM-5.3-Flash points: dropped %d/%d prefill cases (Mamba align split, hash_block_size=%d), %d/%d decode cases "
        "(token budget, token_budget=%d), %d/%d prefill and %d/%d decode cases (cache capacity, usable_blocks=%d)",
        split,
        prefill_candidates,
        unit,
        over_budget,
        decode_candidates,
        token_budget,
        dropped["prefill"]["cache_capacity"],
        prefill_candidates - split,
        dropped["decode"]["cache_capacity"],
        decode_candidates - over_budget,
        usable,
    )
    # A phase without a point leaves its cells nothing to measure, so the plan is refused before any GPU work.
    if not prefill or not decode:
        raise ValueError("GLM-5.3-Flash collection has no feasible point for the accepted cache capacity")
    return {"schema_version": 3, "prefill": prefill, "decode": decode}, dropped


def glm_point_manifest(**limits: int) -> tuple[dict[str, Any], str, str]:
    """Return the point list of ``glm_points`` for these limits, its canonical JSON and the SHA-256 of that JSON."""
    return _manifest(glm_points(**limits)[0])


def glm_plan_manifest(
    options: FPMCollectionOptions, deployments: Sequence[FpmDeploymentProfile]
) -> tuple[str, str, str]:
    """Return the canonical JSON and the SHA-256 of the point list for a plan's options and profile deployments.

    The third value is the canonical JSON of the dropped counts of ``glm_points``. The planner resolves
    ``options.max_prefill_isl`` first. Both phases take one token budget, ``options.max_prefill_isl``; the function does
    not read ``options.max_num_batched_tokens``. A phase whose batch limit neither its own option nor
    ``options.max_num_seqs`` sets takes the ``max_batch_size`` that the deployments share, which the runner renders for
    its cells. The cells of a plan share one list, so the deployments must share one hash block and one prefill
    checkpoint alignment. For the same reason a phase that the plan collects is refused when it has no batch limit and
    the deployments hold different ``max_batch_size`` values: one list would leave out the larger points of one of them.
    A phase that the plan does not collect is never measured, and its points take the smallest value. The capacity rule
    takes the smallest capacity and the largest page among the deployments, and each point takes the most blocks that
    the groups of one deployment give it.
    """
    memories = [deployment.resources.runtime_memory for deployment in deployments]
    hash_blocks = {None if memory is None else memory.hash_block_size_tokens for memory in memories}
    if len(hash_blocks) != 1 or None in hash_blocks:
        raise ValueError("GLM-5.3-Flash collection requires the accepted profile's hash_block_size_tokens")
    # vLLM 0.33.0 saves a prefill checkpoint inside a chunk only if every Mamba group holds a checkpoint block, and it
    # uses the alignment of the first one (scheduler.py:437-449). The import writes the checkpoint fields for each Mamba
    # state group, so a deployment whose groups do not all hold a block takes no internal checkpoint.
    # The plan's one point list serves every cell, so a deployment without an internal checkpoint cannot share it with
    # one that has it.
    alignments, state_groups = set(), []
    for deployment in deployments:
        groups = deployment.resources.cache_groups
        mamba = [group for group in groups if group.prefill_checkpoint_blocks is not None]
        holding = [group for group in mamba if group.prefill_checkpoint_blocks == 1]
        # The tail ring is a state group without the prefill checkpoint field.
        rings = sum(group.kind == "state" and group.prefill_checkpoint_blocks is None for group in groups)
        state_groups.append((rings, len(mamba)))
        if mamba and len(holding) == len(mamba):
            alignments.update(group.prefill_checkpoint_alignment for group in holding)
        else:
            alignments.add(None)
    if None in alignments and len(alignments) > 1:
        raise ValueError(
            "GLM-5.3-Flash collection requires the same prefill checkpoint setting across the accepted deployments"
        )
    if len(alignments) != 1:
        raise ValueError(
            "GLM-5.3-Flash collection requires one prefill checkpoint alignment across the accepted deployments"
        )
    batch_limits = {}
    for phase, own in (("prefill", options.max_prefill_batch_size), ("decode", options.max_decode_batch_size)):
        limit = own or options.max_num_seqs
        if limit is None:
            # The runner renders the max_batch_size of each cell's deployment for a phase without a limit. The list is
            # one list for every deployment, so for deployments of different batch sizes it would leave out the larger
            # points of one of them.
            sizes = {deployment.resources.max_batch_size for deployment in deployments}
            if len(sizes) > 1 and phase in options.workload_kinds:
                shared = [deployment.tp for deployment in deployments]
                held = ", ".join(
                    f"the accepted deployment at TP {deployment.tp}"
                    + (
                        f" (DP {deployment.dp}, MoE TP {deployment.moe_tp}, MoE EP {deployment.moe_ep})"
                        if shared.count(deployment.tp) > 1
                        else ""
                    )
                    + f" holds {deployment.resources.max_batch_size}"
                    for deployment in deployments
                )
                raise ValueError(
                    "GLM-5.3-Flash collection generates one point list for every deployment, and the request sets no "
                    f"batch limit for the {phase} cells while the deployments that it selects hold different "
                    f"max_batch_size values: {held}; select deployments that share one max_batch_size with "
                    "--fpm-gpu-counts or --fpm-parallel-presets, or probe the runtime again so that the deployments "
                    "record one max_batch_size"
                )
            limit = min(sizes)
        batch_limits[phase] = limit
    _require_mamba_hash_block(deployments)
    payload, dropped = glm_points(
        hash_block_size=hash_blocks.pop(),
        token_budget=options.max_prefill_isl,
        prefill_batch_limit=batch_limits["prefill"],
        decode_batch_limit=batch_limits["decode"],
        measured_context=glm_measured_context(options),
        kv_cache_bytes=min(memory.kv_cache_bytes for memory in memories),
        page_size_bytes=max(
            group.page_size_bytes for deployment in deployments for group in deployment.resources.cache_groups
        ),
        state_groups=state_groups,
        checkpoint_alignment=alignments.pop(),
    )
    _, canonical, digest = _manifest(payload)
    return canonical, digest, json.dumps(dropped, sort_keys=True, separators=(",", ":"))


def glm_smoke_subset(
    payload: dict[str, Any], hash_block_size: int, phases: Sequence[str] = ("prefill", "decode")
) -> tuple[dict[str, Any], str, str]:
    """Return the smoke points of a generated list, with their canonical JSON and its SHA-256.

    Each phase that the plan collects takes two points, so an aggregated plan takes four and a prefill or decode role
    takes two. The prefill phase takes a third point when the list holds it. The other phase of a role keeps an empty
    list, because the cells of the plan never measure it.

    The prefill points are the 64-token prefill of one request, once without a prefix and once after one hash block of
    prefix. The third point is the prefill of two hash blocks for one request after one hash block of prefix. The list
    holds it only if the launch has a prefill checkpoint block whose alignment lets vLLM's Mamba align mode save the
    checkpoint inside the chunk. Its chunk then holds a block boundary that the align mode does not stop at, so the
    smoke shows that such a point runs in one step. The context of a decode point is its total KV read tokens divided by
    its batch size. The decode points are the points of one request and of two requests at the smallest context above
    one hash block where the list holds both. A small token budget can leave a smaller context with one request only. If
    no context above one hash block holds a point of two requests, as in the list of a request limited to one sequence,
    the decode points are the points of one request at the two smallest contexts above one hash block. A point is taken
    at a context only if its total KV read tokens equal its batch size times that context. A list that does not give the
    two 64-token prefill points or the two decode points of a collected phase is refused. So is a list whose points of
    two requests above one hash block have no point of one request at the same context, because the generator puts one
    request on the batch axis of every context.
    """
    taken = ((64, 0), (64, hash_block_size), (2 * hash_block_size, hash_block_size))
    prefill = [
        point
        for point in payload["prefill"]
        if point["batch_size"] == 1 and (point["total_prefill_tokens"], point["total_kv_read_tokens"]) in taken
    ]

    def context(point: dict[str, Any]) -> int:
        return point["total_kv_read_tokens"] // point["batch_size"]

    def contexts_of(batch_size: int) -> set[int]:
        return {
            context(point)
            for point in payload["decode"]
            if point["batch_size"] == batch_size
            and point["total_kv_read_tokens"] == batch_size * context(point)
            and context(point) > hash_block_size
        }

    above = sorted({context(point) for point in payload["decode"] if context(point) > hash_block_size})
    pairs = sorted(contexts_of(1) & contexts_of(2))
    # The token budget bounds the batch of each context separately, so the pair can sit above the smallest context.
    # Points of two requests without a pair leave no context to take and are refused below. Only a list without such
    # a point, where the batch limit, the cache capacity or the token budget leaves one request, takes one request at
    # each of its two smallest contexts instead.
    if contexts_of(2):
        batch_sizes, contexts = (1, 2), pairs[:1]
    else:
        batch_sizes, contexts = (1,), above[:2]
    decode = [
        point
        for point in payload["decode"]
        if point["batch_size"] in batch_sizes
        and any(point["total_kv_read_tokens"] == point["batch_size"] * value for value in contexts)
    ]
    # The cells of the plan never measure a phase that it does not collect, so that phase keeps no point and needs none.
    smoke = {"prefill": prefill if "prefill" in phases else [], "decode": decode if "decode" in phases else []}
    # A small token budget or context, or a launch without an internal prefill checkpoint, leaves no prefill of two hash
    # blocks, so only the two 64-token points are needed.
    required = {"prefill": sum(point["total_prefill_tokens"] == 64 for point in prefill), "decode": len(decode)}
    if any(required[phase] != 2 for phase in phases):
        held = "four points" if len(phases) == 2 else f"two {phases[0]} points"
        raise ValueError(f"GLM-5.3-Flash smoke subset requires the generated list to hold its {held}")
    return _manifest({"schema_version": payload["schema_version"], **smoke})


def _require_mamba_hash_block(deployments: Sequence[FpmDeploymentProfile]) -> None:
    """Refuse a deployment whose hash block differs from the block of one of its Mamba groups.

    The producer refuses such a launch when the engine starts. A Mamba group is a cache group with the prefill
    checkpoint field, which the import writes for a MambaSpec group only. The tail ring is a state group of its own
    4-token block without that field, so the rule does not read it. A deployment without a hash block is skipped. For a
    generated list, ``glm_plan_manifest`` refuses it before this rule.
    """
    for deployment in deployments:
        memory = deployment.resources.runtime_memory
        hash_block = None if memory is None else memory.hash_block_size_tokens
        if hash_block is None:
            continue
        for group in deployment.resources.cache_groups:
            if group.prefill_checkpoint_blocks is not None and group.block_size_tokens != hash_block:
                raise ValueError(
                    "GLM-5.3-Flash collection requires a hash block of the Mamba block size: the accepted profile "
                    f"holds hash_block_size_tokens={hash_block} and cache group {group.name} holds "
                    f"block_size_tokens={group.block_size_tokens}; use a profile from a runtime probe at the default "
                    "prefix-caching settings, which give the hash block the Mamba block size"
                )


def validate_glm_limits(options: FPMCollectionOptions, deployments: Sequence[FpmDeploymentProfile]) -> None:
    """Refuse a token budget and a hash block of a plan that the producer refuses when the engine starts.

    The rules hold for a generated and for a user list. ``deployments`` are those that get cells. Each phase that the
    plan collects is checked with the budget that the runner renders for its cells:
    ``options.prefill_sampling.max_total_prefill_tokens`` for a prefill cell, and for a decode cell
    ``options.max_num_batched_tokens`` or, with that unset, the ``max_num_tokens`` of its deployment. Each must be a
    multiple of the producer's chunk alignment. Each deployment's hash block must then be the block of its Mamba groups.

    The text names the source of the value and what to change. A value that equals the token limit that the runtime
    probe of a deployment with runtime memory recorded needs a new probe, since runtime memory takes only that limit.
    Otherwise, for an aggregated plan, for a prefill budget that is the shared option and for a decode budget without
    runtime memory, the text names both options at one value; else it names --fpm-max-prefill-isl. The other limits of
    the plan, such as the batch limit, still apply to the new value, and a plan that sets neither option can meet the
    decode text after the prefill text.
    """
    rule = "GLM-5.3-Flash collection requires a token budget that is a multiple of 4"
    recorded = {
        deployment.resources.max_num_tokens
        for deployment in deployments
        if deployment.resources.runtime_memory is not None
    }
    probe = (
        ", the token limit that a runtime probe recorded in the accepted profile; probe the runtime again with a token "
        "limit that is a multiple of 4"
    )

    # The value must also stay within the token limit of each deployment that the request selects, whose envelope the
    # planner checks before memory admission.
    bound = " within the max_num_tokens of each deployment that the request selects" if deployments else ""
    shared = f"; set --fpm-max-prefill-isl and --fpm-max-num-batched-tokens to one multiple of 4{bound}"

    phases = options.workload_kinds
    if "prefill" in phases:
        budget = options.prefill_sampling.max_total_prefill_tokens
        if budget % CHUNK_ALIGN:
            if budget in recorded:
                remedy = probe
            elif options.worker_type == "aggregated" or budget == options.max_num_batched_tokens:
                remedy = shared
            else:
                remedy = "; set --fpm-max-prefill-isl to a multiple of 4"
            raise ValueError(f"{rule}: the prefill cells get a budget of {budget} tokens{remedy}")
    if "decode" in phases:
        if options.max_num_batched_tokens is not None:
            budget = options.max_num_batched_tokens
            if budget % CHUNK_ALIGN:
                # Runtime memory requires the option to equal the recorded limit.
                remedy = probe if recorded else shared
                raise ValueError(f"{rule}: the decode cells get --fpm-max-num-batched-tokens={budget}{remedy}")
        else:
            for deployment in deployments:
                budget = deployment.resources.max_num_tokens
                if budget % CHUNK_ALIGN:
                    remedy = probe if deployment.resources.runtime_memory is not None else shared
                    raise ValueError(
                        f"{rule}: the decode cells of the accepted deployment at TP {deployment.tp} get its "
                        f"resources.max_num_tokens={budget}{remedy}"
                    )
    _require_mamba_hash_block(deployments)


def validate_glm_layout(topologies: Sequence[ParallelTopology], runtime_version: str | None) -> None:
    """Refuse a runtime version and a parallel layout that the producer refuses when the engine starts.

    ``runtime_version`` is the vLLM version of the runtime, never a data label. The producer takes the exact version
    string only, so a local version suffix is another runtime. A plan without a runtime version leaves the version to
    the producer's gate. The layout rule reads the axes of each topology, not its strategy name: a one-GPU cell has the
    strategy ``single``.
    """
    baseline = BASELINE_VERSIONS["vllm"]
    if runtime_version is not None and runtime_version != baseline:
        raise ValueError(
            f"GLM-5.3-Flash collection supports vLLM {baseline} only, and the runtime of this plan is vLLM "
            f"{runtime_version!r}"
        )
    for topology in topologies:
        if topology.tp not in TP_SIZES or (topology.pp, topology.dp, topology.moe_ep, topology.cp) != (1, 1, 1, 1):
            raise ValueError(
                "GLM-5.3-Flash collection supports pure tensor parallelism only (TP 1, 2, 4 or 8, with DP, PP, CP and "
                f"expert parallelism 1): request the preset pure_tp. Got {topology.to_dict()}"
            )
