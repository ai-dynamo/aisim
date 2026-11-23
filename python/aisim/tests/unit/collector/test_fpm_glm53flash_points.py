# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import Counter
from types import SimpleNamespace

import pytest
from collector.fpm_forward.config import _freeze_benchmark_points
from collector.fpm_forward.glm53flash_points import (
    CHUNK_ALIGN,
    MAX_BATCH,
    MAX_CONTEXT,
    MAX_NEW,
    TP_SIZES,
    _require_mamba_hash_block,
    glm_align_stop,
    glm_measured_context,
    glm_plan_manifest,
    glm_point_blocks,
    glm_point_manifest,
    glm_points,
    glm_smoke_subset,
    validate_glm_layout,
)
from collector.fpm_forward.types import ParallelTopology

from .test_fpm_glm53flash_prefix_seed import load_producer

pytestmark = pytest.mark.unit

# GLM-5.3-Flash at TP 4 as the runtime probe measured it on vLLM 0.33.0: 2,176-token hash blocks, a
# 185,126,092,800-byte cache of 13,516,800-byte pool blocks and five state groups, the tail ring and four Mamba groups,
# with the onboarding limits of 8,192 tokens and 256 sequences.
_TP4_LIMITS = {
    "hash_block_size": 2176,
    "token_budget": 8192,
    "prefill_batch_limit": 256,
    "decode_batch_limit": 256,
    "measured_context": 131072,
    "kv_cache_bytes": 185_126_092_800,
    "page_size_bytes": 13_516_800,
    "state_groups": ((1, 4),),
}
# vLLM 0.33.0 prefills KDA with FlashKDA by default on SM90, SM10x and SM12x with bf16, head_dim 128 and a bounded
# gate (models/glm5next/common/kda.py:137-160). Then each of the four KDA state groups holds one prefill checkpoint
# block with an alignment of 16 tokens (kda.py:168-176, model_executor/layers/mamba/kda_checkpoint.py:9-10).
_TP4 = {**_TP4_LIMITS, "checkpoint_alignment": 16}


def _limits(**changes: int) -> dict[str, int]:
    """The TP4 limits without the checkpoint arguments, with ``changes`` applied."""
    return {**_TP4_LIMITS, **changes}


_GENERATOR_LOGGER = "collector.fpm_forward.glm53flash_points"


def _logged_lines(caplog) -> list[tuple[int, str]]:
    """The level and the text of each line that the point generator logged."""
    return [(record.levelno, record.getMessage()) for record in caplog.records if record.name == _GENERATOR_LOGGER]


def test_tp4_point_list_follows_the_producer_rules(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=_GENERATOR_LOGGER)
    payload, canonical, digest = glm_point_manifest(**_TP4)
    prefill, decode = payload["prefill"], payload["decode"]

    assert payload["schema_version"] == 3
    assert (len(prefill), len(decode)) == (496, 81)
    # One line counts the points that each rule drops, against the candidates that the rule sees. With a checkpoint
    # alignment of 16 tokens, the align rule drops none. The token budget drops 16 decode batches and adds its own limit
    # of 7 requests at context 16,384 to the candidates. The cache holds every point, and its count is logged too.
    assert _logged_lines(caplog) == [
        (
            logging.INFO,
            "GLM-5.3-Flash points: dropped 0/496 prefill cases (Mamba align split, hash_block_size=2176), 16/97 "
            "decode cases (token budget, token_budget=8192), 0/496 prefill and 0/81 decode cases (cache capacity, "
            "usable_blocks=13696)",
        )
    ]
    # vLLM saves the prefill checkpoint inside the chunk of every length above one hash block, because after a
    # whole-block prefix each of them puts its last cache position a multiple of 16 tokens after its start. So 4,096
    # and 8,192 are on the axis too.
    assert sorted({point["total_prefill_tokens"] // point["batch_size"] for point in prefill}) == [
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        2176,
        4096,
        4352,
        6528,
        8192,
    ]
    assert sorted({point["total_kv_read_tokens"] // point["batch_size"] for point in decode}) == [
        2**exponent for exponent in range(2, 18)
    ]
    assert prefill[0] == {"batch_size": 1, "total_prefill_tokens": 4, "total_kv_read_tokens": 0}
    assert prefill[-1] == {"batch_size": 1, "total_prefill_tokens": 8192, "total_kv_read_tokens": 121856}
    # The formal cell of an aggregated TP4 request stages these bytes.
    assert digest == "2c81f2f524251c18a5a0c33607544695042c48f8151f2b9c3f067d983fa4cbc6"
    assert decode[0] == {"batch_size": 1, "total_kv_read_tokens": 4}
    assert decode[-1] == {"batch_size": 16, "total_kv_read_tokens": 2097152}
    # Prefill points ascend by per-request new tokens, batch size and prefix; decode points by context and batch size.
    prefill_keys = [
        (point["total_prefill_tokens"] // point["batch_size"], point["batch_size"], point["total_kv_read_tokens"])
        for point in prefill
    ]
    decode_keys = [(point["total_kv_read_tokens"] // point["batch_size"], point["batch_size"]) for point in decode]
    assert prefill_keys == sorted(set(prefill_keys))
    assert decode_keys == sorted(set(decode_keys))
    for point in prefill:
        batch, new, prefix = point["batch_size"], point["total_prefill_tokens"], point["total_kv_read_tokens"]
        assert new % batch == 0 and new // batch % 4 == 0
        assert new <= 8192 and batch <= 32
        assert prefix % (batch * 2176) == 0
        assert (prefix + new) // batch <= 131072
    for point in decode:
        batch, kv = point["batch_size"], point["total_kv_read_tokens"]
        assert kv % batch == 0 and 4 <= kv // batch <= 131072
        assert batch <= 32
    path = tmp_path / "points.json"
    path.write_text(json.dumps(payload))
    assert _freeze_benchmark_points(str(path)) == (canonical, digest)


def test_tp4_prefill_points_run_in_one_align_mode_step():
    payload, _, _ = glm_point_manifest(**_TP4)
    # With prefix caching, vLLM 0.33.0 runs GLM-5.3-Flash in Mamba align mode, on 2,176-token blocks at TP 4. Without
    # an internal checkpoint, it stops a prefill chunk at the last cacheable block boundary before the end of its
    # prompt. A point whose chunk holds that stop runs in two steps, and the producer accepts one step per repetition.
    split = []
    for point in payload["prefill"]:
        batch = point["batch_size"]
        start = point["total_kv_read_tokens"] // batch
        end = start + point["total_prefill_tokens"] // batch
        if glm_align_stop(start, end, 2176, hash_block=2176, checkpoint_alignment=16) is not None:
            split.append(point)
    assert split == []
    # The closed form of the vLLM 0.33.0 cut for a start of k whole blocks B and n new tokens, without the rule. The
    # last cache position is kB + (n - 1) // B * B, strictly inside the chunk exactly when n > B. With an alignment that
    # divides B, the checkpoint at that position is valid whenever n > B, so no stop remains. The cache holds every
    # point of both lists, so the list with the 16-token alignment holds every candidate, and the list without a
    # checkpoint holds exactly its points of at most one block of new tokens.
    with_checkpoint, dropped = glm_points(**_TP4_LIMITS, checkpoint_alignment=16)
    without, _ = glm_points(**_TP4_LIMITS, checkpoint_alignment=None)
    assert dropped["prefill"]["mamba_align_split"] == dropped["prefill"]["cache_capacity"] == 0
    assert len(with_checkpoint["prefill"]) == dropped["prefill"]["candidates"]
    assert without["prefill"] == [
        point for point in with_checkpoint["prefill"] if point["total_prefill_tokens"] // point["batch_size"] <= 2176
    ]


def test_point_list_is_deterministic():
    _, first, digest = glm_point_manifest(**_TP4)
    _, second, _ = glm_point_manifest(**_TP4)

    assert first == second
    assert digest == hashlib.sha256(first.encode("utf-8")).hexdigest()


# The decode batches of the TP4 list at each context. Before a batch decodes, each request prefills its prompt beyond
# its cached seed, and the 8,192-token budget must hold these suffixes of the whole batch in one step.
_TP4_DECODE_BATCHES = {
    **{2**exponent: [1, 2, 4, 8, 16, 32] for exponent in range(2, 9)},
    512: [1, 2, 4, 8, 16],
    1024: [1, 2, 4, 8],
    2048: [1, 2, 4],
    4096: [1, 2, 4],
    8192: [1, 2, 4],
    16384: [1, 2, 4, 7],
    32768: [1, 2, 4, 8, 16, 32],
    65536: [1, 2, 4, 8, 16, 32],
    131072: [1, 2, 4, 8, 16],
}


def _producer_suffix(producer, context: int, hash_block_size: int) -> int:
    """The tokens that one request of a decode point prefills after its seed, from the producer's own functions."""
    prompt = producer._glm_decode_prompt(context)
    return prompt - producer._glm_decode_seed(prompt, hash_block_size)


def _decode_batches(payload: dict) -> dict[int, list[int]]:
    """The batch sizes of the decode points at each per-request context."""
    batches: dict[int, list[int]] = {}
    for point in payload["decode"]:
        batches.setdefault(point["total_kv_read_tokens"] // point["batch_size"], []).append(point["batch_size"])
    return batches


def test_tp4_decode_batches_fit_the_one_step_suffix_prefill(monkeypatch):
    producer = load_producer(monkeypatch).Glm53FlashPrefixSeedScheduler
    payload, _, _ = glm_point_manifest(**_TP4)
    batches = _decode_batches(payload)

    assert batches == _TP4_DECODE_BATCHES
    assert sum(len(sizes) for sizes in batches.values()) == 81
    # vLLM prefills the suffixes of two or more requests in one step only if they fit the token budget together.
    # Otherwise the first decode step never holds every request, and the producer rejects every repetition of the point.
    for point in payload["decode"]:
        context = point["total_kv_read_tokens"] // point["batch_size"]
        assert point["batch_size"] * _producer_suffix(producer, context, 2176) <= 8192
    # Each context takes the largest batch that the budget holds, up to the producer's limit of 32.
    assert {context: max(sizes) for context, sizes in batches.items()} == {
        context: min(32, 8192 // _producer_suffix(producer, context, 2176)) for context in batches
    }


@pytest.mark.parametrize(
    ("hash_block_size", "token_budget", "measured_context"),
    [
        # The prompt at the measured context fills one hash block, so the seed is empty and the suffix is the block.
        (2176, 8192, 2177),
        # One position more: the seed is the whole block and the suffix is one token.
        (2176, 8192, 2178),
        # A hash block that does not divide the powers of two.
        (96, 1024, 1000),
        # One request's suffix at the measured context exceeds the budget. It prefills over several steps, and no other
        # request can decode before it, so that context keeps the point of one request.
        (2176, 2048, 2100),
    ],
)
def test_decode_batches_follow_the_producer_suffix_rule(monkeypatch, hash_block_size, token_budget, measured_context):
    module = load_producer(monkeypatch)
    producer = module.Glm53FlashPrefixSeedScheduler
    # The cache holds every candidate, so only the token budget bounds the batches.
    limits = {
        "hash_block_size": hash_block_size,
        "token_budget": token_budget,
        "measured_context": measured_context,
        "kv_cache_bytes": 10**8 * 13_516_800,
    }
    payload, _, _ = glm_point_manifest(**{**_TP4, **limits})

    contexts = {measured_context, *(2**exponent for exponent in range(2, 18) if 2**exponent <= measured_context)}
    expected = {}
    for context in sorted(contexts):
        suffix = _producer_suffix(producer, context, hash_block_size)
        limit = min(module.MAX_BATCH, _TP4["decode_batch_limit"], max(1, token_budget // suffix))
        expected[context] = sorted({*(2**exponent for exponent in range(6) if 2**exponent <= limit), limit})
    assert _decode_batches(payload) == expected
    # A point of two or more requests prefills all its suffixes in one step.
    for point in payload["decode"]:
        context = point["total_kv_read_tokens"] // point["batch_size"]
        suffix = _producer_suffix(producer, context, hash_block_size)
        assert point["batch_size"] == 1 or point["batch_size"] * suffix <= token_budget


def test_decode_batches_use_the_raw_token_budget(monkeypatch):
    producer = load_producer(monkeypatch).Glm53FlashPrefixSeedScheduler
    # The producer checks the decode suffixes against vLLM's whole token budget, not the prefill cap of 8,192. The
    # suffixes of 1,919 tokens at context 4,096 fit 8,192 tokens four times and 16,384 tokens eight times.
    payload, _, _ = glm_point_manifest(**{**_TP4, "token_budget": 16384})

    assert _decode_batches(payload)[4096] == [1, 2, 4, 8]
    for point in payload["decode"]:
        context = point["total_kv_read_tokens"] // point["batch_size"]
        assert point["batch_size"] == 1 or point["batch_size"] * _producer_suffix(producer, context, 2176) <= 16384
    # Prefill keeps the cap, because the producer measures at most 8,192 new tokens in a point.
    assert max(point["total_prefill_tokens"] for point in payload["prefill"]) == 8192


def test_smoke_subset_is_five_points_of_the_list(tmp_path):
    payload, _, _ = glm_point_manifest(**_TP4)
    subset, canonical, digest = glm_smoke_subset(payload, 2176)

    # The third prefill point runs two hash blocks after one block of prefix. Its chunk holds the block boundary at
    # 4,352, which vLLM's Mamba align mode runs in one step.
    assert subset == {
        "schema_version": 3,
        "prefill": [
            {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 0},
            {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 2176},
            {"batch_size": 1, "total_prefill_tokens": 4352, "total_kv_read_tokens": 2176},
        ],
        "decode": [
            {"batch_size": 1, "total_kv_read_tokens": 4096},
            {"batch_size": 2, "total_kv_read_tokens": 8192},
        ],
    }
    for phase in ("prefill", "decode"):
        assert all(point in payload[phase] for point in subset[phase])
    path = tmp_path / "smoke-points.json"
    path.write_text(json.dumps(subset))
    assert _freeze_benchmark_points(str(path)) == (canonical, digest)
    # The live smoke of an aggregated TP4 request stages these bytes, so their digest stays pinned.
    assert digest == "35060cd4787dc7cb14fb00480359fa5c63a907bd514692e0397f839315c018fd"
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    # Without the point of two requests at 4,096, the pair comes from 8,192, the next context that holds both.
    without = {**payload, "decode": [point for point in payload["decode"] if point != subset["decode"][1]]}
    assert glm_smoke_subset(without, 2176)[0]["decode"] == [
        {"batch_size": 1, "total_kv_read_tokens": 8192},
        {"batch_size": 2, "total_kv_read_tokens": 16384},
    ]
    # The generator puts one request on the batch axis of every context, so a list whose points of two requests above
    # one hash block have no point of one request at their context is incomplete.
    unmatched = {
        **payload,
        "decode": [
            point for point in payload["decode"] if point["batch_size"] != 1 or point["total_kv_read_tokens"] <= 2176
        ],
    }
    with pytest.raises(ValueError) as error:
        glm_smoke_subset(unmatched, 2176)
    assert str(error.value) == "GLM-5.3-Flash smoke subset requires the generated list to hold its four points"


@pytest.mark.parametrize(
    ("token_budget", "prefill", "decode", "digest"),
    [
        # Two suffixes of 1,919 tokens at 4,096 fit the default budget in one step, so the pair stays at 4,096.
        (
            8192,
            [
                {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 0},
                {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 2176},
                {"batch_size": 1, "total_prefill_tokens": 4352, "total_kv_read_tokens": 2176},
            ],
            [{"batch_size": 1, "total_kv_read_tokens": 4096}, {"batch_size": 2, "total_kv_read_tokens": 8192}],
            "35060cd4787dc7cb14fb00480359fa5c63a907bd514692e0397f839315c018fd",
        ),
        # 3,836 tokens hold one suffix of 1,919 tokens at 4,096 but not two, so 4,096 holds one request only. Two
        # suffixes of 1,663 tokens at 8,192 fit, so the pair comes from 8,192. The budget holds no prefill of two hash
        # blocks.
        (
            3836,
            [
                {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 0},
                {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 2176},
            ],
            [{"batch_size": 1, "total_kv_read_tokens": 8192}, {"batch_size": 2, "total_kv_read_tokens": 16384}],
            "b817a7f76c7aea73992af154bee98fed24a0bd1e417e714e8654b62a8d075a70",
        ),
    ],
    ids=["default-budget", "budget-3836"],
)
def test_smoke_decode_pair_is_at_the_smallest_context_that_holds_both(token_budget, prefill, decode, digest):
    payload, _, _ = glm_point_manifest(**{**_TP4, "token_budget": token_budget})
    subset, canonical, subset_digest = glm_smoke_subset(payload, 2176)

    assert subset == {"schema_version": 3, "prefill": prefill, "decode": decode}
    assert all(point in payload[phase] for phase in ("prefill", "decode") for point in subset[phase])
    assert subset_digest == digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_smoke_subset_serves_every_tp4_token_budget():
    # 256 sequences need a budget of at least 256 tokens. Every budget up to 8,192 gives four smoke points of its list.
    # A budget of two hash blocks or more also gives the prefill of two blocks.
    for token_budget in range(256, 8193, 4):
        payload, _, _ = glm_point_manifest(**{**_TP4, "token_budget": token_budget})
        subset, _, _ = glm_smoke_subset(payload, 2176)
        prefill = 3 if token_budget >= 2 * 2176 else 2
        assert [len(subset[phase]) for phase in ("prefill", "decode")] == [prefill, 2], token_budget
        assert all(point in payload[phase] for phase in ("prefill", "decode") for point in subset[phase])


_PREFILL_PAIR = [
    {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 0},
    {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 2176},
]
# The smoke prefill points of a list that holds two hash blocks of new tokens after one block of prefix.
_PREFILL_POINTS = [*_PREFILL_PAIR, {"batch_size": 1, "total_prefill_tokens": 4352, "total_kv_read_tokens": 2176}]
_TWO_REQUEST_DECODE_PAIR = [
    {"batch_size": 1, "total_kv_read_tokens": 4096},
    {"batch_size": 2, "total_kv_read_tokens": 8192},
]
_FOUR_POINTS = "GLM-5.3-Flash smoke subset requires the generated list to hold its four points"


def test_smoke_subset_keeps_the_points_of_the_collected_phases_only():
    payload, _, _ = glm_point_manifest(**_TP4)

    for phases, expected in (
        (("prefill", "decode"), {"prefill": _PREFILL_POINTS, "decode": _TWO_REQUEST_DECODE_PAIR}),
        (("prefill",), {"prefill": _PREFILL_POINTS, "decode": []}),
        (("decode",), {"prefill": [], "decode": _TWO_REQUEST_DECODE_PAIR}),
    ):
        subset, canonical, digest = glm_smoke_subset(payload, 2176, phases)
        assert subset == {"schema_version": 3, **expected}
        assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    # Both phases are the default, and they give the same bytes as before the argument existed.
    assert glm_smoke_subset(payload, 2176) == glm_smoke_subset(payload, 2176, ("prefill", "decode"))


def test_smoke_subset_of_a_prefill_role_needs_no_second_decode_context(tmp_path):
    # A prefill role limited to one sequence, 4,096 tokens and a 4,096-token context measures contexts up to 4,089.
    # Its list holds both smoke prefill points, but only one decode context above the hash block of 2,176 tokens.
    measured = glm_measured_context(SimpleNamespace(vllm_max_model_len=4096))
    limits = {"token_budget": 4096, "prefill_batch_limit": 1, "decode_batch_limit": 1, "measured_context": measured}
    payload, _, _ = glm_point_manifest(**{**_TP4, **limits})
    above = {point["total_kv_read_tokens"] for point in payload["decode"] if point["total_kv_read_tokens"] > 2176}
    assert above == {4089}
    assert all(point in payload["prefill"] for point in _PREFILL_PAIR)

    # The role never measures a decode point, so the list has no decode point to lack.
    subset, canonical, digest = glm_smoke_subset(payload, 2176, ("prefill",))
    assert subset == {"schema_version": 3, "prefill": _PREFILL_PAIR, "decode": []}
    path = tmp_path / "smoke-points.json"
    path.write_text(json.dumps(subset))
    assert _freeze_benchmark_points(str(path)) == (canonical, digest)
    # An aggregated plan measures both phases, so the same list is refused for it, with or without the argument.
    for arguments in ((), (("prefill", "decode"),)):
        with pytest.raises(ValueError) as error:
            glm_smoke_subset(payload, 2176, *arguments)
        assert str(error.value) == _FOUR_POINTS


def test_smoke_subset_of_a_decode_role_needs_no_prefill_points():
    payload, _, _ = glm_point_manifest(**_TP4)
    without = {**payload, "prefill": [point for point in payload["prefill"] if point["total_prefill_tokens"] != 64]}

    # A decode role never measures a prefill point, so a list without the 64-token prefill points still serves it.
    subset, _, _ = glm_smoke_subset(without, 2176, ("decode",))
    assert subset == {"schema_version": 3, "prefill": [], "decode": _TWO_REQUEST_DECODE_PAIR}
    with pytest.raises(ValueError) as error:
        glm_smoke_subset(without, 2176)
    assert str(error.value) == _FOUR_POINTS


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_smoke_subset_of_one_phase_refuses_a_list_without_its_points(phase):
    payload, _, _ = glm_point_manifest(**_TP4)
    smoke = glm_smoke_subset(payload, 2176)[0]["prefill"]
    # The prefill smoke takes these three points only. The decode smoke can take any context above one hash block, so
    # only a list without a decode point above the block lacks its points.
    keep = {
        "prefill": lambda point: point not in smoke,
        "decode": lambda point: point["total_kv_read_tokens"] // point["batch_size"] <= 2176,
    }[phase]
    without = {**payload, phase: [point for point in payload[phase] if keep(point)]}

    with pytest.raises(ValueError) as error:
        glm_smoke_subset(without, 2176, (phase,))
    # A role that collects one phase needs the two points of that phase, not the four of an aggregated plan.
    assert str(error.value) == f"GLM-5.3-Flash smoke subset requires the generated list to hold its two {phase} points"


@pytest.mark.parametrize(
    "limits",
    [{"decode_batch_limit": 1}, {"prefill_batch_limit": 1, "decode_batch_limit": 1}],
    ids=["decode-limit", "one-sequence-request"],
)
def test_smoke_subset_of_a_list_without_two_request_decode_points_takes_two_contexts(tmp_path, limits):
    payload, _, _ = glm_point_manifest(**{**_TP4, **limits})
    assert {point["batch_size"] for point in payload["decode"]} == {1}
    subset, canonical, digest = glm_smoke_subset(payload, 2176)

    # The prefill points are those of the TP4 smoke. The decode points are one request at each of the two smallest
    # contexts of the list above one hash block.
    assert subset == {
        "schema_version": 3,
        "prefill": [
            {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 0},
            {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 2176},
            {"batch_size": 1, "total_prefill_tokens": 4352, "total_kv_read_tokens": 2176},
        ],
        "decode": [
            {"batch_size": 1, "total_kv_read_tokens": 4096},
            {"batch_size": 1, "total_kv_read_tokens": 8192},
        ],
    }
    for phase in ("prefill", "decode"):
        assert all(point in payload[phase] for point in subset[phase])
    path = tmp_path / "smoke-points.json"
    path.write_text(json.dumps(subset))
    assert _freeze_benchmark_points(str(path)) == (canonical, digest)


def test_smoke_subset_follows_a_cache_capacity_that_excludes_two_requests():
    # 21 usable blocks: two requests fit only below one hash block. A decode request at a context of 4,096 tokens holds
    # 2 attention blocks, the ring block and 2 blocks in each of the 4 KDA groups, the cached block of its seed beside
    # its running block: 11 blocks, and two such requests need 22. Below one hash block it holds 1 + 1 + 4 = 6.
    payload, _, _ = glm_point_manifest(**{**_TP4, "kv_cache_bytes": 21 * 13_516_800})
    two_request_contexts = {
        point["total_kv_read_tokens"] // 2 for point in payload["decode"] if point["batch_size"] == 2
    }
    assert two_request_contexts and max(two_request_contexts) < 2176

    subset, _, _ = glm_smoke_subset(payload, 2176)
    assert subset["decode"] == [
        {"batch_size": 1, "total_kv_read_tokens": 4096},
        {"batch_size": 1, "total_kv_read_tokens": 8192},
    ]


def test_smoke_subset_of_a_one_request_list_with_one_context_above_the_hash_block_is_refused():
    # The measured context of 4,096 tokens leaves one decode context above one hash block, so there is no second one.
    payload, _, _ = glm_point_manifest(**{**_TP4, "decode_batch_limit": 1, "measured_context": 4096})
    above = {point["total_kv_read_tokens"] for point in payload["decode"] if point["total_kv_read_tokens"] > 2176}
    assert above == {4096}
    with pytest.raises(ValueError) as error:
        glm_smoke_subset(payload, 2176)
    assert str(error.value) == "GLM-5.3-Flash smoke subset requires the generated list to hold its four points"


def test_capacity_rule_drops_points_beyond_the_accepted_capacity(caplog):
    caplog.set_level(logging.INFO, logger=_GENERATOR_LOGGER)
    # 99 usable blocks. A request holds the attention blocks of its tokens, the ring block and one block in each of
    # the 4 KDA groups, one more in each KDA group with a prefix (a decode context of at least one hash block) and its
    # checkpoint block in each KDA group for a prefill of more than one hash block of new tokens.
    payload, _, _ = glm_point_manifest(**{**_TP4, "kv_cache_bytes": 99 * 13_516_800})
    decode = payload["decode"]

    # A decode request at context x needs ceil((x + 1) / 2,176) + 1 + 4 blocks below one hash block, and 4 more above,
    # so a batch of b requests fits if b times that is at most 99. The 7 contexts from 4 to 256 (need 6, b <= 16) keep
    # the five batches up to 16: 35 points. 512, 1,024 and 2,048 keep their batches within the budget limits of 16, 8
    # and 4: 5 + 4 + 3. 4,096 (need 11), 8,192 (13) and 16,384 (17) keep 1, 2 and 4: 9. 32,768 (25) and 65,536 (40)
    # keep 1 and 2: 4. 131,072 (70) keeps 1. 35 + 12 + 9 + 4 + 1 = 61.
    assert len(decode) == 61
    assert {"batch_size": 1, "total_kv_read_tokens": 131072} in decode
    assert {"batch_size": 2, "total_kv_read_tokens": 262144} not in decode
    # A prefill of n new tokens after r hash blocks needs r + ceil(n / 2,176) + 1 + 4 blocks, 4 more with a prefix and 4
    # more above one hash block of new tokens. The 7 lengths from 4 to 256 tokens keep their 8 rungs for one request,
    # 7 for two (r <= 39), 5 for four (r <= 14), 3 for eight (r <= 2) and the rung 0 for sixteen: 24 each, and so does
    # 512 (batches up to 16). 1,024 (up to 8) keeps 23, 2,048 (up to 4) 20 and 2,176 (up to 3, r <= 23 for three) 21.
    # 4,096 (r + 15, batches 1 and 2) keeps 8 + 7 = 15, and 4,352, 6,528 and 8,192 keep their 8 rungs of one request:
    # 192 + 23 + 20 + 21 + 15 + 24 = 295.
    assert len(payload["prefill"]) == 295
    assert _logged_lines(caplog) == [
        (
            logging.INFO,
            "GLM-5.3-Flash points: dropped 0/496 prefill cases (Mamba align split, hash_block_size=2176), 16/97 "
            "decode cases (token budget, token_budget=8192), 201/496 prefill and 20/81 decode cases (cache capacity, "
            "usable_blocks=99)",
        )
    ]


def test_capacity_without_a_feasible_point_is_refused(caplog):
    caplog.set_level(logging.INFO, logger=_GENERATOR_LOGGER)
    # Five usable blocks: every request needs at least 1 + 1 + 4 = 6 blocks, one attention block, the ring block and one
    # block in each of the four KDA groups.
    with pytest.raises(ValueError) as error:
        glm_point_manifest(**{**_TP4, "kv_cache_bytes": 5 * 13_516_800})
    assert str(error.value) == "GLM-5.3-Flash collection has no feasible point for the accepted cache capacity"
    # The logged count explains the refusal.
    assert _logged_lines(caplog) == [
        (
            logging.INFO,
            "GLM-5.3-Flash points: dropped 0/496 prefill cases (Mamba align split, hash_block_size=2176), 16/97 "
            "decode cases (token budget, token_budget=8192), 496/496 prefill and 81/81 decode cases (cache capacity, "
            "usable_blocks=5)",
        )
    ]


def test_copied_limits_match_the_producer(monkeypatch):
    producer = load_producer(monkeypatch)

    assert (CHUNK_ALIGN, MAX_BATCH, MAX_NEW, MAX_CONTEXT, TP_SIZES) == (
        producer.CHUNK_ALIGN,
        producer.MAX_BATCH,
        producer.MAX_NEW,
        producer.MAX_CONTEXT,
        producer.TP_SIZES,
    )
    # The measured context keeps the producer's headroom below the runtime context.
    assert glm_measured_context(SimpleNamespace(vllm_max_model_len=4096)) == 4096 - producer.CONTEXT_HEADROOM


@pytest.mark.parametrize(
    "field,value,text",
    [
        ("token_budget", 8190, "GLM-5.3-Flash collection requires a token budget that is a multiple of 4"),
        ("hash_block_size", 2178, "GLM-5.3-Flash collection requires a hash block size that is a multiple of 4"),
    ],
)
def test_budget_and_hash_block_must_be_multiples_of_four(field, value, text):
    with pytest.raises(ValueError) as error:
        glm_point_manifest(**{**_TP4, field: value})
    assert str(error.value) == text


@pytest.mark.parametrize(
    ("start", "end", "alignment", "stop"),
    [
        (0, 64, None, None),  # one block or less: no boundary strictly inside
        (64, 128, None, None),
        (64, 192, None, 128),  # two blocks after one block: cut one block before the end
        (64, 193, None, 192),
        (0, 65, None, 64),
        (64, 192, 16, None),  # the internal checkpoint removes the stop
        (0, 65, 16, None),
        (64, 192, 48, 128),  # 128 - 64 is no multiple of the alignment
        (32, 192, 16, 64),  # a start inside a block stops at the next boundary
    ],
)
def test_align_stop_follows_vllm_0330(start, end, alignment, stop):
    assert glm_align_stop(start, end, 64, hash_block=64, checkpoint_alignment=alignment) == stop


def test_points_without_a_checkpoint_keep_at_most_one_block_of_new_tokens():
    payload, dropped = glm_points(**_limits(hash_block_size=64, token_budget=256), checkpoint_alignment=None)
    assert max(point["total_prefill_tokens"] // point["batch_size"] for point in payload["prefill"]) == 64
    assert dropped["prefill"]["mamba_align_split"] > 0


def test_points_with_a_checkpoint_keep_every_length():
    payload, dropped = glm_points(**_limits(hash_block_size=64, token_budget=256), checkpoint_alignment=16)
    assert dropped["prefill"]["mamba_align_split"] == 0
    assert {point["total_prefill_tokens"] for point in payload["prefill"] if point["batch_size"] == 1} >= {
        128,
        192,
        256,
    }


@pytest.mark.parametrize("alignment", [None, 16, 48])
def test_planned_prefill_points_run_in_one_step(alignment):
    # Each request of a prefill point starts at its whole-block prefix and runs to the end of its prompt. With the
    # checkpoint alignment of the launch, vLLM 0.33.0 does not stop that chunk inside the prompt.
    payload, _ = glm_points(**_limits(hash_block_size=96, token_budget=1024), checkpoint_alignment=alignment)
    for point in payload["prefill"]:
        start = point["total_kv_read_tokens"] // point["batch_size"]
        end = start + point["total_prefill_tokens"] // point["batch_size"]
        assert glm_align_stop(start, end, 96, hash_block=96, checkpoint_alignment=alignment) is None, point


PAGE = _TP4_LIMITS["page_size_bytes"]


def test_capacity_estimate_counts_the_checkpoint_block_for_a_prefill_of_more_than_one_hash_block():
    # 64-token hash blocks and four KDA groups without a ring. A prefill saves its checkpoint inside the chunk only with
    # more than one hash block of new tokens, so only then does it hold the checkpoint block of each KDA group.
    def blocks(new, prefix, alignment):
        point = {"batch_size": 1, "total_prefill_tokens": new, "total_kv_read_tokens": prefix}
        return glm_point_blocks(
            "prefill", point, hash_block_size=64, state_groups=((0, 4),), checkpoint_alignment=alignment
        )

    # 64 new tokens: 1 attention block and 1 block in each KDA group, 5. 128 new tokens: 2 attention blocks and 2 in
    # each KDA group with the checkpoint block, 10, and 6 without it. After a prefix of 64 tokens: 3 attention blocks
    # and 3 in each KDA group, 15.
    assert [blocks(64, 0, 16), blocks(128, 0, 16), blocks(128, 0, None), blocks(128, 64, 16)] == [5, 10, 6, 15]
    payload, _ = glm_points(
        **_limits(
            hash_block_size=64,
            token_budget=128,
            prefill_batch_limit=1,
            decode_batch_limit=1,
            measured_context=256,
            kv_cache_bytes=PAGE * 9,
            state_groups=((0, 4),),
        ),
        checkpoint_alignment=16,
    )
    # 9 usable blocks hold the cold prefill of 64 new tokens and not the one of 128.
    assert {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 0} in payload["prefill"]
    assert {"batch_size": 1, "total_prefill_tokens": 128, "total_kv_read_tokens": 0} not in payload["prefill"]


@pytest.mark.parametrize(
    ("pages", "prefill_dropped", "decode_dropped"), [(13, 0, 0), (12, 0, 1), (11, 5, 1)], ids=["13", "12", "11"]
)
def test_capacity_rule_counts_every_free_block_of_the_cache(caplog, pages, prefill_dropped, decode_dropped):
    caplog.set_level(logging.INFO, logger=_GENERATOR_LOGGER)
    # The cache bytes are the free blocks at engine start, which already exclude vLLM's null block, so every page of
    # them is usable. 64-token hash blocks, four KDA groups and no ring. The decode point at context 256 needs
    # ceil(257 / 64) = 5 attention blocks and 2 blocks in each KDA group: 13. A prefill after 192 cached tokens needs
    # ceil((192 + n) / 64) = 4 attention blocks and 2 in each KDA group: 12, for each of its 5 lengths from 4 to 64
    # tokens. Every other candidate needs at most 11.
    payload, dropped = glm_points(
        **_limits(
            hash_block_size=64,
            token_budget=64,
            prefill_batch_limit=1,
            decode_batch_limit=1,
            measured_context=256,
            kv_cache_bytes=PAGE * pages,
            page_size_bytes=PAGE,
            state_groups=((0, 4),),
        ),
        checkpoint_alignment=16,
    )
    decode_point = {"batch_size": 1, "total_kv_read_tokens": 256}
    prefill_point = {"batch_size": 1, "total_prefill_tokens": 64, "total_kv_read_tokens": 192}
    assert (decode_point in payload["decode"], prefill_point in payload["prefill"]) == (pages >= 13, pages >= 12)
    assert (dropped["prefill"]["cache_capacity"], dropped["decode"]["cache_capacity"]) == (
        prefill_dropped,
        decode_dropped,
    )
    assert _logged_lines(caplog) == [
        (
            logging.INFO,
            "GLM-5.3-Flash points: dropped 0/20 prefill cases (Mamba align split, hash_block_size=64), 0/7 decode "
            f"cases (token budget, token_budget=64), {prefill_dropped}/20 prefill and {decode_dropped}/7 decode cases "
            f"(cache capacity, usable_blocks={pages})",
        )
    ]


def _vllm_blocks(workload_kind, point, checkpoint):
    """The blocks of one TP 4 point and the blocks of one request in one KDA group, stated apart from the generator.

    vLLM 0.33.0's scheduler gave every point of the two TP 4 lists exactly these blocks, in a run of its own code with
    the settings of the collector's launch (prefix caching, Mamba mode align, 2,176-token blocks, no async scheduling,
    four KDA groups, one attention group and the ring), and one block less failed every point that an engine starts for
    with that pool (269 of the 577 points of the list with a checkpoint block and 277 of the 537 points of the list
    without one).
    """
    batch = point["batch_size"]
    if workload_kind == "prefill":
        prefix = point["total_kv_read_tokens"] / batch
        new = point["total_prefill_tokens"] / batch
        mamba = 1 + int(prefix > 0) + int(checkpoint and new > 2176)
        return batch * (math.ceil((prefix + new) / 2176) + 1 + 4 * mamba), mamba
    context = point["total_kv_read_tokens"] / batch
    mamba = 2 if context >= 2176 else 1
    return batch * (math.ceil((context + 1) / 2176) + 1 + 4 * mamba), mamba


@pytest.mark.parametrize(
    ("alignment", "per_request"),
    [(16, {1: 111, 2: 431, 3: 35}), (None, {1: 111, 2: 426})],
    ids=["flashkda", "no-checkpoint"],
)
def test_capacity_estimate_of_every_tp4_point_is_the_count_of_vllm(alignment, per_request):
    # Blocks of one request in one KDA group: 1 for the 54 decode points below one hash block and the 57 prefills of at
    # most one hash block without a prefix, 2 for the 27 decode points above it, the 5 longer prefills without a prefix
    # and the 399 prefills of at most one hash block with a prefix, and 3 for the 35 longer prefills with a prefix. The
    # list without a checkpoint block holds no prefill of more than one hash block. vLLM 0.33.0's scheduler gave every
    # point these blocks in a run of its own code.
    payload, _ = glm_points(**{**_TP4_LIMITS, "checkpoint_alignment": alignment})
    counts = Counter()
    for kind in ("prefill", "decode"):
        for point in payload[kind]:
            expected, mamba = _vllm_blocks(kind, point, alignment is not None)
            estimate = glm_point_blocks(
                kind, point, hash_block_size=2176, state_groups=((1, 4),), checkpoint_alignment=alignment
            )
            assert estimate == expected, (kind, point)
            counts[mamba] += 1
    assert counts == per_request


@pytest.mark.parametrize(("context", "blocks"), [(2048, 6), (2175, 6), (2176, 11), (2177, 11)])
def test_decode_request_at_one_hash_block_holds_two_blocks_of_each_mamba_group(context, blocks):
    # The two TP 4 lists hold a context of 2,048 tokens and none of the three contexts at the edge of the first block.
    # vLLM 0.33.0's scheduler, run with the settings of the collector's launch, gave one decode request at a context
    # of 2,048 tokens 1 attention block, the ring block and 1 block in each KDA group, 6 in all, and the same 6 at
    # 2,175 tokens, where the decode step fills the first block. At 2,176 and 2,177 tokens, without a seed, its decode
    # steps enter the second block: 2 attention blocks, the ring block and 2 blocks in each KDA group, 11.
    point = {"batch_size": 1, "total_kv_read_tokens": context}
    assert glm_point_blocks("decode", point, hash_block_size=2176, state_groups=((1, 4),), checkpoint_alignment=16) == (
        blocks
    )


_TP4_LARGEST = [
    {"batch_size": 32, "total_prefill_tokens": 32 * new, "total_kv_read_tokens": 32 * 60 * 2176}
    for new in (4, 8, 16, 32, 64, 128, 256)
]


@pytest.mark.parametrize(
    ("alignment", "counts"), [(16, (496, 81)), (None, (456, 81))], ids=["flashkda", "no-checkpoint"]
)
def test_smallest_cache_that_keeps_the_whole_tp4_list(alignment, counts):
    # The largest need of both lists is a prefill of 32 requests, each with up to 256 new tokens after a prefix of 60
    # hash blocks: ceil((60 * 2,176 + 256) / 2,176) = 61 attention blocks, the ring block and 2 blocks in each of the 4
    # KDA groups, the cached block of the prefix beside the running block. 32 * (61 + 1 + 8) = 2,240 blocks. These
    # prefills hold no checkpoint block, so the list without one needs the same. vLLM 0.33.0's scheduler, run with the
    # settings of the collector's launch, completed every prefill point of either list in one engine with 2,240 free
    # blocks, and with 2,239 every point but these 7.
    def lists(pages):
        return glm_points(**_limits(kv_cache_bytes=PAGE * pages), checkpoint_alignment=alignment)

    payload, dropped = lists(2240)
    assert (len(payload["prefill"]), len(payload["decode"])) == counts
    assert dropped["prefill"]["cache_capacity"] == dropped["decode"]["cache_capacity"] == 0
    smaller, dropped = lists(2239)
    assert [point for point in payload["prefill"] if point not in smaller["prefill"]] == _TP4_LARGEST
    assert dropped["decode"]["cache_capacity"] == 0


@pytest.mark.parametrize("alignment", [16, None], ids=["flashkda", "no-checkpoint"])
def test_smallest_cache_that_keeps_every_tp4_decode_point(alignment):
    # The largest decode need is 32 requests at a context of 65,536 tokens: ceil(65,537 / 2,176) = 31 attention blocks,
    # the ring block and 2 blocks in each of the 4 KDA groups, 32 * 40 = 1,280. 16 requests at 131,072 need
    # 16 * (61 + 1 + 8) = 1,120. A decode request holds no checkpoint block. vLLM 0.33.0's scheduler, run with the
    # settings of the collector's launch, completed the 81 decode points in one engine with 1,280 free blocks, and with
    # 1,279 every point but this one.
    def decode(pages):
        payload, dropped = glm_points(**_limits(kv_cache_bytes=PAGE * pages), checkpoint_alignment=alignment)
        return payload["decode"], dropped["decode"]["cache_capacity"]

    points, dropped = decode(1280)
    assert (len(points), dropped) == (81, 0)
    smaller, dropped = decode(1279)
    assert ([point for point in points if point not in smaller], dropped) == (
        [{"batch_size": 32, "total_kv_read_tokens": 32 * 65536}],
        1,
    )


def test_capacity_estimate_counts_two_blocks_for_each_mamba_group():
    # 2,239 usable blocks. The 32-request prefills after 60 hash blocks need 32 * (61 + 1 + 4 * 2) = 2,240 blocks with
    # the cached prefix block of each KDA group beside its running block, and drop. Counted as ring groups, the KDA
    # groups would give 32 * (61 + 5) = 2,112, and every point would stay.
    limits = {**_TP4, "kv_cache_bytes": PAGE * 2239}
    payload, dropped = glm_points(**limits)
    assert (len(payload["prefill"]), len(payload["decode"])) == (489, 81)
    assert (dropped["prefill"]["cache_capacity"], dropped["decode"]["cache_capacity"]) == (7, 0)
    _, without = glm_points(**{**limits, "state_groups": ((5, 0),)})
    assert without["prefill"]["cache_capacity"] == 0


def _deployment(
    *state_groups: tuple[int | None, int | None],
    kv_cache_bytes: int = _TP4_LIMITS["kv_cache_bytes"],
    page_size_bytes: int = PAGE,
) -> SimpleNamespace:
    """A TP4 deployment whose recurrent-state groups hold these (prefill checkpoint blocks, alignment) pairs.

    Its runtime memory holds ``kv_cache_bytes``, and every group a page of ``page_size_bytes``.
    """
    memory = SimpleNamespace(hash_block_size_tokens=2176, kv_cache_bytes=kv_cache_bytes)
    groups = [
        SimpleNamespace(kind="attention", page_size_bytes=page_size_bytes, prefill_checkpoint_blocks=None),
        # The state group of the 4-token blocks is no Mamba group. The import writes the checkpoint fields only for a
        # MambaSpec group (collector/fpm_forward/runtime_memory.py:643, :662-676), so this group has none.
        SimpleNamespace(kind="state", page_size_bytes=page_size_bytes, prefill_checkpoint_blocks=None),
        *(
            SimpleNamespace(
                name=f"mamba_{index}",
                kind="state",
                block_size_tokens=2176,
                page_size_bytes=page_size_bytes,
                prefill_checkpoint_blocks=blocks,
                prefill_checkpoint_alignment=align,
            )
            for index, (blocks, align) in enumerate(state_groups)
        ),
    ]
    return SimpleNamespace(resources=SimpleNamespace(runtime_memory=memory, cache_groups=groups))


_OPTIONS = SimpleNamespace(
    max_prefill_isl=8192,
    max_prefill_batch_size=256,
    max_decode_batch_size=256,
    max_num_seqs=256,
    vllm_max_model_len=131072 + 7,
)


@pytest.mark.parametrize(
    ("groups", "alignment", "checkpoint_groups"),
    [
        ([(0, None)] * 4, None, 0),
        ([(1, 16)] * 4, 16, 4),
        # Each Mamba layer sets its own checkpoint block in vLLM 0.33.0 (kda.py:171-176 of glm5next), so a model with
        # two kinds of Mamba layers can hold groups with and without one. The scheduler then saves no checkpoint inside
        # the last chunk (scheduler.py:445-449), and the point list is the one without a checkpoint.
        ([(1, 16), (1, 16), (1, 16), (0, None)], None, 3),
    ],
    ids=["no-checkpoint", "checkpoint-16", "checkpoint-in-some-groups"],
)
def test_plan_manifest_reads_the_checkpoint_of_the_accepted_profile(groups, alignment, checkpoint_groups):
    canonical, digest, _ = glm_plan_manifest(_OPTIONS, [_deployment(*groups)])
    # Every group of the four holds the checkpoint field, so the deployment has four Mamba groups and the ring.
    _, expected, expected_digest = glm_point_manifest(**_TP4_LIMITS, checkpoint_alignment=alignment)
    assert (canonical, digest) == (expected, expected_digest)


@pytest.mark.parametrize(
    ("deployments", "text"),
    [
        (
            [[(1, 16)] * 4, [(1, 32)] * 4],
            "GLM-5.3-Flash collection requires one prefill checkpoint alignment across the accepted deployments",
        ),
        # The plan's one point list serves every cell, and a deployment without a checkpoint block would cut the
        # prefill that the other one runs in one step.
        (
            [[(1, 16)] * 4, [(0, None)] * 4],
            "GLM-5.3-Flash collection requires the same prefill checkpoint setting across the accepted deployments",
        ),
    ],
    ids=["two-alignments", "checkpoint-and-none"],
)
def test_plan_manifest_refuses_deployments_with_different_checkpoint_alignments(deployments, text):
    with pytest.raises(ValueError) as error:
        glm_plan_manifest(_OPTIONS, [_deployment(*groups) for groups in deployments])
    assert str(error.value) == text


def test_plan_manifest_refuses_a_mamba_block_that_is_not_the_hash_block():
    deployment = _deployment((1, 16), (1, 16))
    deployment.resources.cache_groups[2].block_size_tokens = 4352
    with pytest.raises(ValueError) as error:
        glm_plan_manifest(_OPTIONS, [deployment])
    assert str(error.value) == _hash_block_refusal(2176, "mamba_0", 4352)


def test_plan_manifest_takes_the_smallest_capacity_the_largest_page_and_the_most_groups():
    # One deployment has the smaller cache and three KDA groups. The other has the smaller page and four. Together they
    # give the TP4 limits with 2,239 usable blocks: the TP4 FlashKDA list without the 7 prefill points of 32 requests
    # after 60 hash blocks (one for each length from 4 to 256 new tokens), whose SHA-256 is pinned below.
    small = _deployment(*[(1, 16)] * 3, kv_cache_bytes=PAGE * 2239)
    large = _deployment(*[(1, 16)] * 4, kv_cache_bytes=PAGE * 4000, page_size_bytes=PAGE // 2)
    canonical, digest, dropped = glm_plan_manifest(_OPTIONS, [small, large])
    _, expected, expected_digest = glm_point_manifest(**{**_TP4, "kv_cache_bytes": PAGE * 2239})
    assert (canonical, digest) == (expected, expected_digest)
    assert digest == "9b1e63994616f901fe748f621f8482122090605c8f62c02f9757120f9fc81370"
    assert json.loads(dropped)["prefill"]["cache_capacity"] == 7
    # The larger capacity, the smaller page or the smaller number of groups each give the list without a drop.
    assert digest != glm_point_manifest(**_TP4)[2]


@pytest.mark.parametrize("checkpoint", [(1, 16), (0, None)], ids=["flashkda", "no-checkpoint"])
@pytest.mark.parametrize(("pages", "dropped_count"), [(2240, 0), (2239, 7)])
def test_plan_manifest_counts_every_mamba_group_of_the_deployment(checkpoint, pages, dropped_count):
    # Four KDA groups with or without a checkpoint block, and the ring. The 7 prefills of 32 requests after 60 hash
    # blocks need 32 * (61 + 1 + 4 * 2) = 2,240 blocks either way, and every other point fewer.
    _, _, dropped = glm_plan_manifest(_OPTIONS, [_deployment(*[checkpoint] * 4, kv_cache_bytes=PAGE * pages)])
    assert json.loads(dropped)["prefill"]["cache_capacity"] == dropped_count


@pytest.mark.parametrize("order", [1, -1], ids=["four-first", "four-last"])
def test_plan_manifest_counts_the_most_mamba_groups_of_the_deployments(order):
    # 2,239 usable blocks. With the four KDA groups of one deployment a request after 60 hash blocks needs
    # 61 + 1 + 4 * 2 = 70 blocks, so the 7 prefills of 32 requests at that rung drop. With the three of the other one it
    # would need 61 + 1 + 3 * 2 = 68 and fit.
    four = _deployment(*[(1, 16)] * 4, kv_cache_bytes=PAGE * 4000)
    three = _deployment(*[(1, 16)] * 3, kv_cache_bytes=PAGE * 2239)
    _, _, dropped = glm_plan_manifest(_OPTIONS, [four, three][::order])
    assert json.loads(dropped)["prefill"]["cache_capacity"] == 7
    _, alone = glm_points(**{**_TP4, "kv_cache_bytes": PAGE * 2239, "state_groups": ((1, 3),)})
    assert alone["prefill"]["cache_capacity"] == 0


@pytest.mark.parametrize(("pages", "dropped_count"), [(2240, 0), (2239, 7)])
def test_plan_manifest_counts_the_groups_of_each_deployment_together(pages, dropped_count):
    # One deployment holds two ring groups and three KDA groups, the other one ring group and four. A request after 60
    # hash blocks holds 2 + 3 * 2 = 8 blocks of state in the first and 1 + 4 * 2 = 9 in the second, so the 7 prefills
    # of 32 requests at that rung need 32 * (61 + 9) = 2,240 blocks. The two rings with the four KDA groups of the other
    # deployment would give 32 * (61 + 10) = 2,272.
    two_rings = _deployment(*[(1, 16)] * 3, kv_cache_bytes=PAGE * 4000)
    two_rings.resources.cache_groups.append(
        SimpleNamespace(kind="state", page_size_bytes=PAGE, prefill_checkpoint_blocks=None)
    )
    one_ring = _deployment(*[(1, 16)] * 4, kv_cache_bytes=PAGE * pages)
    _, _, dropped = glm_plan_manifest(_OPTIONS, [two_rings, one_ring])
    assert json.loads(dropped)["prefill"]["cache_capacity"] == dropped_count


@pytest.mark.parametrize("sizes", [(16, 8), (8, 16)], ids=["smaller-last", "smaller-first"])
def test_plan_manifest_gives_a_phase_that_the_plan_does_not_collect_the_smallest_batch_size(sizes):
    # A decode role with its own decode limit of 4 never measures the prefill points, which take the smaller of the two
    # batch sizes of the deployments: 8.
    deployments = [_deployment(*[(1, 16)] * 4), _deployment(*[(1, 16)] * 4)]
    for deployment, batch_size in zip(deployments, sizes, strict=True):
        deployment.resources.max_batch_size = batch_size
    options = SimpleNamespace(
        **{**vars(_OPTIONS), "max_num_seqs": None, "max_prefill_batch_size": None, "max_decode_batch_size": 4},
        workload_kinds=("decode",),
    )
    canonical, digest, _ = glm_plan_manifest(options, deployments)
    _, expected, expected_digest = glm_point_manifest(**{**_TP4, "prefill_batch_limit": 8, "decode_batch_limit": 4})
    assert (canonical, digest) == (expected, expected_digest)


def test_a_phase_without_a_point_is_refused(caplog):
    caplog.set_level(logging.INFO, logger=_GENERATOR_LOGGER)
    # A measured context of 3 tokens holds one decode context and no prefill point: the shortest prefill is 4 tokens.
    # The one context has the 6 batches from 1 to 32, its suffix of 2 tokens fits the budget for each, and each needs
    # 32 * (1 + 13) = 448 blocks at most.
    with pytest.raises(ValueError) as error:
        glm_point_manifest(**{**_TP4, "measured_context": 3})
    assert str(error.value) == "GLM-5.3-Flash collection has no feasible point for the accepted cache capacity"
    assert _logged_lines(caplog) == [
        (
            logging.INFO,
            "GLM-5.3-Flash points: dropped 0/0 prefill cases (Mamba align split, hash_block_size=2176), 0/6 decode "
            "cases (token budget, token_budget=8192), 0/0 prefill and 0/6 decode cases (cache capacity, "
            "usable_blocks=13696)",
        )
    ]


def test_both_phases_take_the_prefill_isl_as_their_token_budget():
    # The list reads one token budget for both phases, options.max_prefill_isl, and not options.max_num_batched_tokens:
    # with 8,192 tokens, 16 of 97 decode candidates exceed the budget. With 16,384 tokens for both, 9 of 98 would: the
    # suffixes of 1,023, 2,047, 1,919, 1,663 and 1,151 tokens at the contexts 1,024, 2,048, 4,096, 8,192 and 16,384 then
    # hold 16, 8, 8, 9 and 14 requests, which drops 1 + 2 + 2 + 2 + 2 of the batches up to 32, and the limits of 9 and
    # 14 add 2 candidates to the 6 batches of each of the 16 contexts.
    options = SimpleNamespace(**{**vars(_OPTIONS), "max_num_batched_tokens": 16384})
    _, _, dropped = glm_plan_manifest(options, [_deployment(*[(1, 16)] * 4)])
    assert json.loads(dropped)["decode"] == {"candidates": 97, "token_budget": 16, "cache_capacity": 0}
    _, both = glm_points(**{**_TP4, "token_budget": 16384})
    assert both["decode"] == {"candidates": 98, "token_budget": 9, "cache_capacity": 0}


def _hash_block_refusal(hash_block, group, block):
    return (
        "GLM-5.3-Flash collection requires a hash block of the Mamba block size: the accepted profile holds "
        f"hash_block_size_tokens={hash_block} and cache group {group} holds block_size_tokens={block}; use a profile "
        "from a runtime probe at the default prefix-caching settings, which give the hash block the Mamba block size"
    )


def test_hash_block_rule_reads_every_deployment_that_holds_a_hash_block():
    # The first deployment holds no hash block and is skipped; the rule still reads the second one.
    first, second = _deployment((1, 16), (1, 16)), _deployment((1, 16), (1, 16))
    first.resources.runtime_memory.hash_block_size_tokens = None
    second.resources.cache_groups[3].block_size_tokens = 4352
    with pytest.raises(ValueError) as error:
        _require_mamba_hash_block([first, second])
    assert str(error.value) == _hash_block_refusal(2176, "mamba_1", 4352)
    # A deployment without runtime memory is skipped too.
    first.resources.runtime_memory = None
    second.resources.cache_groups[3].block_size_tokens = 2176
    assert _require_mamba_hash_block([first, second]) is None


def test_hash_block_rule_refuses_a_mamba_block_smaller_than_the_hash_block():
    deployment = _deployment((1, 16), (1, 16))
    deployment.resources.cache_groups[2].block_size_tokens = 1088
    with pytest.raises(ValueError) as error:
        _require_mamba_hash_block([deployment])
    assert str(error.value) == _hash_block_refusal(2176, "mamba_0", 1088)


_LAYOUT_REFUSAL = (
    "GLM-5.3-Flash collection supports pure tensor parallelism only (TP 1, 2, 4 or 8, with DP, PP, CP and expert "
    "parallelism 1): request the preset pure_tp. Got "
)


def test_layout_check_reads_the_axes_and_the_runtime_version():
    # A one-GPU cell has the strategy "single", so the rule reads the axes of each topology.
    single = ParallelTopology(tp=1, pp=1, dp=1, moe_tp=1, moe_ep=1, cp=1)
    assert validate_glm_layout([single], "0.33.0") is None
    # A plan without a version at plan time leaves the version to the gate in the container.
    assert validate_glm_layout([single], None) is None
    # The producer's gate takes the exact version string, so a local version suffix is another runtime.
    for version in ("0.33.1", "0.33.0+local"):
        with pytest.raises(ValueError) as error:
            validate_glm_layout([single], version)
        assert str(error.value) == (
            f"GLM-5.3-Flash collection supports vLLM 0.33.0 only, and the runtime of this plan is vLLM {version!r}"
        )
    eight = ParallelTopology(**{**single.to_dict(), "tp": 8, "moe_tp": 8})
    assert validate_glm_layout([single, eight], "0.33.0") is None
    for changes in ({"tp": 3, "moe_tp": 3}, {"tp": 16, "moe_tp": 16}, {"pp": 2}, {"dp": 2}, {"cp": 2}):
        topology = ParallelTopology(**{**single.to_dict(), **changes})
        # The wrong topology comes second, so the rule reads every topology.
        with pytest.raises(ValueError) as error:
            validate_glm_layout([single, topology], "0.33.0")
        assert str(error.value) == _LAYOUT_REFUSAL + str(topology.to_dict())


def test_layout_check_refuses_the_runtime_version_before_the_layout():
    expert = ParallelTopology(tp=4, pp=1, dp=1, moe_tp=1, moe_ep=4, cp=1)
    with pytest.raises(ValueError) as error:
        validate_glm_layout([expert], "0.33.1")
    assert str(error.value) == (
        "GLM-5.3-Flash collection supports vLLM 0.33.0 only, and the runtime of this plan is vLLM '0.33.1'"
    )
