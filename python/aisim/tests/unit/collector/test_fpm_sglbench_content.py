# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# The prefix-cache model PageCacheModel follows SGLang v0.5.22 and v0.5.23; see THIRD_PARTY_NOTICES.md.
# Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d
# Source: https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90
# The expected texts and chains restate the text recipe and the chain of ai-dynamo/dynamo (Apache-2.0,
# https://github.com/ai-dynamo/dynamo), components/src/dynamo/vllm/instrumented_scheduler.py at
# b20fe6fe94021b854597965c7305931175e8f482. See THIRD_PARTY_NOTICES.md.

"""CPU checks of the SGLang self-benchmark content plan and request ids. No SGLang, no GPU.

The cached-token counts asserted here come from ``PageCacheModel``, a model of SGLang's prefix cache written in
this file. They are model results, not measurements of an engine. The model is first checked against cached
counts that running engines reported, kept in ``benchmarks/evidence/collector/sglbench-engine-probe-20261113``
(check_3 and check_5) and called the engine probe below. It is then used to drive the content plan through
simulated runs, with seeded token ids and with text chains.

Every text, dataset record and conversation of token ids in this file is FAKE: it was typed in or generated
here. No dataset file is read and no tokenizer runs.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import random
import subprocess
import sys
from array import array
from pathlib import Path

import pytest
from collector.fpm_forward.runtime import sglbench_content as content

pytestmark = pytest.mark.unit

RUNTIME = Path(__file__).resolve().parents[3] / "collector/fpm_forward/runtime"

LOW = 1000
HIGH = LOW + 50_000  # a mid-size token range for most tests
SMALL_HIGH = LOW + 4096  # the smallest range the plan accepts
# FAKE: how far the range of a FAKE text pool reaches above the token range of its plan. A pool keeps more ids
# than its plan allocates from, as the launcher's pool does.
FAKE_POOL_IDS_ABOVE = 1000
RANKS = 2


class PageCacheModel:
    """One rank's SGLang prefix cache, reduced to the three facts the content plan must respect.

    Read in the SGLang source at v0.5.22 (baa9befa381511cd4d56f83980951e5a45fffc6d) and
    v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90), under ``python/sglang/srt``:

    * a prefix is matched in whole pages from the start of the prompt (``mem_cache/radix_cache.py``,
      ``RadixKey.page_aligned`` and ``RadixKey.match_at``); at page size 1 that is token by token;
    * at most ``input_len - 1`` tokens are reported cached (``managers/schedule_batch.py``,
      ``Req._compute_max_prefix_len``);
    * a finished request stores, floored to whole pages, its prompt and the generated tokens that have KV
      (``mem_cache/common.py``, ``release_kv_cache``): none for a one-token prefill request, six for a
      seven-token decode run.

    The stored paths are kept as a trie of pages. ``test_fpm_sglbench_producer.py`` imports this class.
    """

    def __init__(self, page_size: int) -> None:
        if type(page_size) is not int or page_size < 1:
            raise ValueError(f"page_size must be a positive int, got {page_size!r}")
        self.page_size = page_size
        self._root: dict[tuple[int, ...], dict] = {}

    def _whole_pages(self, tokens: list[int], limit: int):
        """The pages of ``tokens`` that lie completely inside its first ``limit`` tokens."""
        page_size = self.page_size
        for start in range(0, limit - page_size + 1, page_size):
            yield tuple(tokens[start : start + page_size])

    def cached(self, prompt: list[int]) -> int:
        """Tokens of ``prompt`` an arriving request finds in the cache."""
        node = self._root
        matched = 0
        for page in self._whole_pages(prompt, len(prompt) - 1):
            node = node.get(page)
            if node is None:
                break
            matched += self.page_size
        return matched

    def _store(self, tokens: list[int]) -> None:
        node = self._root
        for page in self._whole_pages(tokens, len(tokens)):
            node = node.setdefault(page, {})

    def finish_prefill(self, prompt: list[int]) -> None:
        """A request with one output token finished: its prompt is stored."""
        self._store(list(prompt))

    def finish_decode(self, prompt: list[int], generated6: list[int]) -> None:
        """A decode run finished: its prompt and the six generated tokens that have KV are stored."""
        if len(generated6) != 6:
            raise ValueError(f"a decode run stores six generated tokens, got {len(generated6)}")
        self._store([*prompt, *generated6])


def _floor(tokens: int, page_size: int) -> int:
    return tokens // page_size * page_size


def _submit(cache: PageCacheModel, prompts: list[list[int]], one_by_one: bool) -> list[int]:
    """Cached counts of one batch call of one-token requests, which are then stored.

    ``one_by_one`` models a call that SGLang splits over several extend batches: later requests then see what
    earlier requests of the same call stored.
    """
    if one_by_one:
        counts = []
        for prompt in prompts:
            counts.append(cache.cached(prompt))
            cache.finish_prefill(prompt)
        return counts
    counts = [cache.cached(prompt) for prompt in prompts]
    for prompt in prompts:
        cache.finish_prefill(prompt)
    return counts


def _reference_stream(seed: str, token_low: int, token_high: int, kind: str, parts: tuple[int, ...], count: int):
    """The token stream that the docstring of ``sglbench_content`` documents ("Content layout"), restated
    without the module: returns (tokens, skipped words)."""
    vocab_size = token_high - token_low
    accept_below = 2**32 - 2**32 % vocab_size
    key = ":".join([kind, *map(str, parts)])
    tokens: list[int] = []
    skipped = 0
    block = 0
    while len(tokens) < count:
        text = f"sglbench-content-v1:{len(seed.encode('utf-8'))}:{seed}:{key}:{block}"
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        for start in range(0, 32, 4):
            word = int.from_bytes(digest[start : start + 4], "big")
            if word >= accept_below:
                skipped += 1
                continue
            tokens.append(token_low + word % vocab_size)
        block += 1
    return tokens[:count], skipped


def _reference_rotation(seed: str, token_low: int, token_high: int, rank: int) -> int:
    return _reference_stream(seed, token_low, token_high, "rot", (rank,), 1)[0][0] - token_low


def _reference_chain(seed: str, conversations: list[list[int]], kind: str, parts: tuple[int, ...]) -> list[int]:
    """The whole text chain that the docstring of ``sglbench_content`` documents ("Content layout"), restated
    without the module: every conversation once, in the order in which its index is first drawn."""
    total = len(conversations)
    accept_below = 2**32 - 2**32 % total
    key = ":".join([kind, *map(str, parts)])
    order: list[int] = []
    block = 0
    while len(order) < total:
        text = f"sglbench-content-v1:{len(seed.encode('utf-8'))}:{seed}:chain:{key}:{block}"
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        for start in range(0, 32, 4):
            word = int.from_bytes(digest[start : start + 4], "big")
            if word < accept_below and word % total not in order:
                order.append(word % total)
        block += 1
    return [token for index in order for token in conversations[index]]


# ---------------------------------------------------------------------------------------------------------
# The cache model against the engine probe
# ---------------------------------------------------------------------------------------------------------

# The engines of the probe report SGLang 0.5.22 and 0.5.23 and ran Qwen3-0.6B and Qwen3-8B at page sizes 1 and
# 64 with TP 1; check_3 also ran Qwen3-8B at page size 1 with TP 2. Which requests a case sends is read from the
# driver script saved in the evidence folder; the result files hold the counts.


# The engine probe (check_3): a 1021-token prompt (context 1024) reported 0 cached tokens when it was sent cold
# and, sent a second time, 1020 at page size 1 and 960 at page size 64 (cases "decode seed (cold)" and "decode
# measured run"). The extend batch of the two requests of that second call held 2 tokens at page size 1 and 122
# at page size 64 (decode_extend_tokens), which is 1 and 61 per request. The engine probe (check_5): sent once
# more after its seven-token run had finished, a prompt of that length again had 1 uncached token at page size
# 1 and 61 at page size 64 (rows with context 1024 and batch 2 or 1, seed_computed_tokens). The last assertion
# below models that.
@pytest.mark.parametrize(
    ("page_size", "context", "cached", "extend"),
    [(64, 1024, 960, 61), (1, 1024, 1020, 1)],
)
def test_cache_model_reproduces_the_engine_probe_decode_counts(page_size, context, cached, extend):
    cache = PageCacheModel(page_size)
    prompt = list(range(5000, 5000 + context - 3))
    assert cache.cached(prompt) == 0  # cold seed
    cache.finish_prefill(prompt)
    assert cache.cached(prompt) == cached
    assert len(prompt) - cache.cached(prompt) == extend
    cache.finish_decode(prompt, [-1, -2, -3, -4, -5, -6])
    assert cache.cached(prompt) == cached


# The engine probe (check_3): a prefix of whole pages that an earlier request had stored was reported cached in
# full, 128 tokens at page size 1 and 512 at page size 64. A stored prefix seven tokens longer (135 and 519
# tokens) was reported as 135 tokens at page size 1 and as 512 at page size 64. The prefix lengths are those of
# the driver script: 8 * max(page size, 16) tokens, and 7 more.
@pytest.mark.parametrize(
    ("page_size", "prefix", "cached"),
    [(1, 128, 128), (64, 512, 512), (1, 135, 135), (64, 519, 512)],
)
def test_cache_model_reproduces_the_engine_probe_prefill_counts(page_size, prefix, cached):
    cache = PageCacheModel(page_size)
    base = list(range(7000, 7000 + prefix))
    cache.finish_prefill(base)
    assert cache.cached([*base, -1, -2, -3]) == cached


def test_cache_model_exposes_each_hazard_the_allocation_prevents():
    # Page size 1: every equal token counts. Numbers below are worked out by hand from the three cache facts.
    plan = content.ContentPlan("0", LOW, HIGH)
    cache = PageCacheModel(1)
    sequence = plan.slot_prefix(0, 0, 40)
    cache.finish_prefill(sequence[:32])  # a deeper seed of the same slot
    # A suffix that starts with the slot's own next token runs on into the deeper seed.
    assert cache.cached([*sequence[:16], sequence[16], HIGH + 1]) == 17
    shot = plan.prefill_prompt(0, 0, 16, 3)
    assert cache.cached(shot) == 16
    cache.finish_prefill(shot)
    # A later shot that repeats this first suffix token runs on into the earlier shot.
    assert cache.cached([*shot[:17], HIGH + 2, HIGH + 3]) == 17
    # The same shot submitted twice finds itself (19 tokens, capped at 18).
    assert cache.cached(shot) == 18
    # A zero-cached prompt that starts with a stored first token is not zero-cached.
    assert cache.cached([sequence[0], HIGH + 4]) == 1
    # A slot whose first token equals another slot's first token would continue on that slot's path.
    assert cache.cached([sequence[0], sequence[1], HIGH + 5]) == 2


# ---------------------------------------------------------------------------------------------------------
# Token streams and first-token allocation against an independent restatement
# ---------------------------------------------------------------------------------------------------------


def test_pinned_vectors_identify_the_content_of_a_seed():
    # A change here changes what a recorded seed means, so it needs a new STREAM_DOMAIN version. These literals
    # are the only content check that does not go through _reference_stream, so they are what catches a key
    # grammar that is changed in the module and in that helper alike. The first three were computed with
    # _reference_stream (not with the module). The others were computed twice, with the module and with a
    # second restatement of the module docstring that shares no code with the module or with
    # _reference_stream, and are pinned because both gave the same values.
    assert content.STREAM_DOMAIN == "sglbench-content-v1"
    plan = content.ContentPlan("0", 0, 4096)
    assert plan.slot_prefix(0, 0, 6) == [466, 1165, 4085, 1625, 2493, 3690]
    assert plan.free_prompt(1, 4) == [1987, 2417, 3663, 1985]
    # A range that does not divide 2**32: the second word of this stream is rejected.
    wide = content.ContentPlan("0", 0, 3 * 2**30)
    assert wide.slot_prefix(0, 1, 5) == [811569619, 2340674644, 875598992, 2566159672, 2446927993]
    # The "off" and "shot" streams: the suffixes of the first two shots of one (rank, slot, k), whose three
    # numbers differ so that their order in the key shows. The first suffix token is the offset of
    # (1, 2, 16) for shot 0 and the next id upwards for shot 1 (the slot's own next token, 2588, is not near);
    # the three tokens after it come from the shot stream of (1, 2, 16, 0) and of (1, 2, 16, 1).
    assert plan.prefill_prompt(1, 2, 16, 4)[16:] == [1654, 1629, 2307, 1276]
    assert plan.prefill_prompt(1, 2, 16, 4)[16:] == [1655, 1040, 3893, 2203]
    # Block counters above 0. With 4,096 ids no word is rejected, so a block gives eight tokens and token i
    # of a slot is word i - 1 of its stream: tokens 8 to 10 span blocks 0 and 1, tokens 88 to 91 span blocks
    # 10 and 11 (where a counter not written in decimal would show).
    sequence = plan.slot_prefix(0, 0, 92)
    assert sequence[8:11] == [2368, 2836, 712]
    assert sequence[88:92] == [3778, 2655, 1359, 2640]
    # A seed of two characters and three UTF-8 bytes: the length in front of the seed counts bytes.
    assert content.ContentPlan("s\u00e9", 0, 4096).slot_prefix(0, 0, 3) == [3506, 440, 2731]


@pytest.mark.parametrize(
    ("seed", "token_low", "token_high"),
    [
        ("0", 0, 4096),
        ("run-7", LOW, HIGH),
        ("s\u00e9ed:with:colons", 5, 5 + 151_643),  # non-ASCII: the length prefix counts UTF-8 bytes
        ("0", 2**32 - 4096, 2**32),  # the top of the 32-bit id space
        ("0", 0, 3 * 2**30),  # one word in four is rejected
    ],
)
def test_content_equals_the_documented_streams(seed, token_low, token_high):
    plan = content.ContentPlan(seed, token_low, token_high)
    vocab_size = token_high - token_low
    zero_issued = dict.fromkeys(range(3), 0)  # the zero-cached counter runs per rank, across slots
    for rank, slot in ((0, 0), (0, 3), (1, 0), (2, 41)):
        rotation = _reference_rotation(seed, token_low, token_high, rank)
        assert 0 <= rotation < vocab_size
        tail, _ = _reference_stream(seed, token_low, token_high, "slot", (rank, slot), 299)
        sequence = [token_low + (rotation + slot) % vocab_size, *tail]
        assert plan.slot_prefix(rank, slot, 300) == sequence

        for cached in (1, 16, 64):
            start = _reference_stream(seed, token_low, token_high, "off", (rank, slot, cached), 1)[0][0] - token_low
            for shot in range(4):
                # The rule in words: walk the range as a circle from the offset, skip the slot's own next token.
                step = 0
                remaining = shot
                while True:
                    candidate = token_low + (start + step) % vocab_size
                    step += 1
                    if candidate == sequence[cached]:
                        continue
                    if remaining == 0:
                        break
                    remaining -= 1
                suffix, _ = _reference_stream(seed, token_low, token_high, "shot", (rank, slot, cached, shot), 6)
                assert plan.prefill_prompt(rank, slot, cached, 7) == [*sequence[:cached], candidate, *suffix]

        for _ in range(3):  # prefill and free prompts take turns on one counter
            index = zero_issued[rank]
            zero_tail, _ = _reference_stream(seed, token_low, token_high, "zero", (rank, index), 4)
            first = token_low + (rotation - 1 - index) % vocab_size
            assert plan.prefill_prompt(rank, slot, 0, 5) == [first, *zero_tail]
            zero_tail, _ = _reference_stream(seed, token_low, token_high, "zero", (rank, index + 1), 2)
            first = token_low + (rotation - 2 - index) % vocab_size
            assert plan.free_prompt(rank, 3) == [first, *zero_tail]
            zero_issued[rank] = index + 2


def test_rejection_sampling_skips_words_and_removes_the_modulo_bias():
    # With 3 * 2**30 ids, "word % vocab" alone would send the top quarter of the words to the lowest third of
    # the ids and make them twice as likely as the rest (one half of all tokens instead of one third).
    token_high = 3 * 2**30
    plan = content.ContentPlan("bias", 0, token_high)
    tokens = plan.slot_prefix(0, 0, 30_001)[1:]
    reference, skipped = _reference_stream("bias", 0, token_high, "slot", (0, 0), 30_000)
    assert tokens == reference
    assert 9_000 < skipped < 11_000  # about a quarter of 40,000 words
    assert all(0 <= token < token_high for token in tokens)
    lowest_third = sum(1 for token in tokens if token < 2**30) / len(tokens)
    assert 0.31 < lowest_third < 0.36  # 1/3 with a standard deviation of 0.003; a biased stream gives 0.5


def test_a_stream_does_not_depend_on_how_it_is_consumed():
    whole = content.ContentPlan("0", LOW, HIGH).slot_prefix(0, 5, 1000)
    piecewise = content.ContentPlan("0", LOW, HIGH)
    for length in (1, 2, 3, 9, 10, 17, 64, 65, 999, 1000):
        assert piecewise.slot_prefix(0, 5, length) == whole[:length]
    backwards = content.ContentPlan("0", LOW, HIGH)
    for length in (1000, 64, 1):
        assert backwards.slot_prefix(0, 5, length) == whole[:length]
    # A shot needs seq[cached]; asking for it first must not change the sequence either.
    lookahead = content.ContentPlan("0", LOW, HIGH)
    assert lookahead.prefill_prompt(0, 5, 500, 2)[:500] == whole[:500]
    assert lookahead.slot_prefix(0, 5, 1000) == whole


# ---------------------------------------------------------------------------------------------------------
# Invariants I0 to I6
# ---------------------------------------------------------------------------------------------------------


def test_i0_prompt_shapes():
    plan = content.ContentPlan("0", LOW, HIGH)
    for rank, slot, cached, new in ((0, 0, 16, 1), (0, 0, 16, 9), (1, 4, 64, 128), (0, 2, 1, 1)):
        prompt = plan.prefill_prompt(rank, slot, cached, new)
        assert len(prompt) == cached + new
        assert prompt[:cached] == plan.slot_prefix(rank, slot, cached)
        assert plan.prefill_seed_prompt(rank, slot, cached) == plan.slot_prefix(rank, slot, cached)
    for new in (1, 2, 77):
        assert len(plan.prefill_prompt(0, 3, 0, new)) == new
        assert len(plan.free_prompt(0, new)) == new
    for context in (4, 5, 67, 1024):
        prompt = plan.decode_prompt(1, 2, context)
        assert len(prompt) == context - 3
        assert prompt == plan.slot_prefix(1, 2, context - 3)
        assert plan.decode_prompt(1, 2, context) == prompt  # the seed and the run use the same prompt


def test_i1_i4_prefix_property_on_a_long_sequence():
    # One slot of 40,000 tokens, near the input limit of the engines of the probe (max_req_input_len 40954 in
    # the recorder headers). Sized to stay fast; no wall-clock assertion.
    plan = content.ContentPlan("0", LOW, HIGH)
    longest = plan.slot_prefix(0, 0, 40_000)
    assert len(longest) == 40_000
    assert all(LOW <= token < HIGH for token in longest)
    for length in (1, 2, 15, 16, 17, 4096, 39_999, 40_000):
        assert plan.slot_prefix(0, 0, length) == longest[:length]
    ascending = content.ContentPlan("0", LOW, HIGH)
    for length in (1, 64, 4097, 40_000):
        assert ascending.slot_prefix(0, 0, length) == longest[:length]
    assert plan.stats()["slot_tokens"] == 40_000


def test_i1_i4_i6_many_slots_fill_the_smallest_range():
    # 4,096 slots of 300 tokens on the smallest range: the first tokens must be all 4,096 ids, each once.
    plan = content.ContentPlan("0", LOW, SMALL_HIGH)
    firsts = []
    for slot in range(4096):
        prefix = plan.slot_prefix(0, slot, 300)
        assert min(prefix) >= LOW and max(prefix) < SMALL_HIGH
        assert plan.slot_prefix(0, slot, 7) == prefix[:7]
        firsts.append(prefix[0])
    assert sorted(firsts) == list(range(LOW, SMALL_HIGH))
    assert plan.stats() == {
        "slots": 4096,
        "slot_tokens": 4096 * 300,
        "seed_prompts": 0,
        "shots": 0,
        "zero_cached_prompts": 0,
        "free_prompts": 0,
        "decode_prompts": 0,
    }


def test_i2_first_suffix_tokens_are_distinct_and_never_the_slots_next_token():
    plan = content.ContentPlan("0", LOW, HIGH)
    keys = [(0, 0, 1), (0, 0, 16), (0, 0, 17), (0, 1, 16), (1, 0, 16), (1, 7, 256)]
    firsts: dict[tuple[int, int, int], list[int]] = {key: [] for key in keys}
    for _ in range(200):  # interleaved: warm shots, measured shots and re-stages of several points
        for rank, slot, cached in keys:
            firsts[(rank, slot, cached)].append(plan.prefill_prompt(rank, slot, cached, 2)[cached])
    for (rank, slot, cached), tokens in firsts.items():
        own_next = plan.slot_prefix(rank, slot, cached + 1)[cached]
        assert len(set(tokens)) == 200
        assert own_next not in tokens


def test_i3_zero_cached_first_tokens_are_distinct_from_each_other_and_from_slots():
    plan = content.ContentPlan("0", LOW, HIGH)
    zero_firsts = []
    slot_firsts = []
    for index in range(300):  # slots appear before, between and after the zero-cached prompts
        zero_firsts.append(plan.prefill_prompt(0, index % 4, 0, 3)[0])
        slot_firsts.append(plan.slot_prefix(0, index, 2)[0])
        zero_firsts.append(plan.free_prompt(0, 1)[0])
    assert len(set(zero_firsts)) == 600
    assert len(set(slot_firsts)) == 300
    assert not set(zero_firsts) & set(slot_firsts)


def test_i4_every_token_is_in_range_at_both_ends_of_the_id_space():
    for token_low, token_high in ((0, 4096), (2**32 - 4096, 2**32), (LOW, HIGH)):
        plan = content.ContentPlan("edge", token_low, token_high)
        prompts = [
            plan.slot_prefix(1, 4095, 500),  # the last slot the smallest range allows
            plan.prefill_seed_prompt(1, 0, 64),
            plan.decode_prompt(0, 9, 300),
            plan.free_prompt(0, 200),
            *(plan.prefill_prompt(0, 2, 0, 50) for _ in range(20)),
            *(plan.prefill_prompt(0, 2, 16, 50) for _ in range(20)),
        ]
        for prompt in prompts:
            assert min(prompt) >= token_low
            assert max(prompt) < token_high


def _recorded_calls(plan):
    return [
        plan.slot_prefix(0, 0, 40),
        plan.prefill_seed_prompt(0, 1, 16),
        plan.prefill_prompt(0, 1, 16, 5),
        plan.prefill_prompt(0, 1, 16, 5),
        plan.prefill_prompt(0, 0, 0, 5),
        plan.free_prompt(1, 9),
        plan.decode_prompt(1, 3, 20),
        plan.prefill_prompt(1, 0, 0, 2),
    ]


def test_i5_same_seed_and_calls_give_identical_output():
    first = _recorded_calls(content.ContentPlan("seed-a", LOW, HIGH))
    second = _recorded_calls(content.ContentPlan("seed-a", LOW, HIGH))
    assert first == second


def test_i5_content_is_a_pure_function_of_its_key_not_of_the_call_order():
    quiet = content.ContentPlan("0", LOW, HIGH)
    busy = content.ContentPlan("0", LOW, HIGH)
    # The busy plan does unrelated work between the calls that are compared.
    busy.slot_prefix(0, 9, 100)
    busy.prefill_prompt(0, 2, 32, 4)
    busy.decode_prompt(1, 1, 50)
    shots_quiet = [quiet.prefill_prompt(0, 1, 16, 6) for _ in range(3)]
    shots_busy = []
    for index in range(3):
        shots_busy.append(busy.prefill_prompt(0, 1, 16, 6))
        busy.prefill_prompt(0, 1, 32, 6)  # another k of the same slot has its own counter
        busy.slot_prefix(0, 20 + index, 3)
    assert shots_busy == shots_quiet
    zero_quiet = [quiet.prefill_prompt(0, 0, 0, 4), quiet.free_prompt(0, 4)]
    busy.slot_prefix(0, 300, 1)
    zero_busy = [busy.prefill_prompt(0, 7, 0, 4), busy.free_prompt(0, 4)]  # the slot argument is not used
    assert zero_busy == zero_quiet
    assert busy.slot_prefix(0, 1, 64) == quiet.slot_prefix(0, 1, 64)


def _differing_positions(first: list[int], second: list[int]) -> int:
    return sum(1 for left, right in zip(first, second, strict=True) if left != right)


def test_i5_another_seed_rank_or_slot_changes_the_content():
    base = content.ContentPlan("0", LOW, HIGH)
    reference = base.slot_prefix(0, 0, 64)
    # The first token alone can coincide (1 in V), so compare the positions after it.
    for seed in ("1", "00", "0:0", "seed"):
        other = content.ContentPlan(seed, LOW, HIGH).slot_prefix(0, 0, 64)
        assert _differing_positions(reference[1:], other[1:]) >= 32
    assert _differing_positions(reference[1:], base.slot_prefix(0, 1, 64)[1:]) >= 32
    assert _differing_positions(reference[1:], base.slot_prefix(1, 0, 64)[1:]) >= 32
    other = content.ContentPlan("1", LOW, HIGH)
    assert _differing_positions(base.prefill_prompt(0, 0, 16, 64)[17:], other.prefill_prompt(0, 0, 16, 64)[17:]) >= 32
    assert _differing_positions(base.free_prompt(0, 64)[1:], other.free_prompt(0, 64)[1:]) >= 32


# ---------------------------------------------------------------------------------------------------------
# Simulated runs through the cache model
# ---------------------------------------------------------------------------------------------------------


def _prefill_points(page_size: int) -> list[tuple[int, int, int]]:
    """(batch, new, cached) per request. Cached prefixes are page-aligned, as the grid plans them."""
    unit = -(-16 // page_size) * page_size
    cached_values = [0, unit, 2 * unit, 4 * unit]
    if page_size == 1:
        cached_values += [1, 2]  # legal at page size 1; a points manifest can ask for them
    new_values = sorted({1, 2, page_size, page_size + 1, 2 * page_size + 3})
    return [(batch, new, cached) for cached in cached_values for new in new_values for batch in (1, 2, 5)]


def _decode_points(page_size: int) -> list[tuple[int, int]]:
    """(batch, context) per request."""
    contexts = {4, 5, 6, 7, page_size + 3, page_size + 4, 2 * page_size + 3, 100, 131, 200}
    # Two contexts around a page boundary that lies above every prefill seed (those reach 4 * unit tokens): a
    # prompt that ends two tokens short of it and a prompt that reaches one token beyond it. Shortest first,
    # the cache holds the page that ends at the boundary for the second prompt only if a decode run of the
    # first one stored generated tokens that continue the slot's sequence.
    boundary = 5 * (-(-16 // page_size) * page_size)
    contexts |= {boundary - 2 + 3, boundary + 1 + 3}
    return [(batch, context) for context in sorted(contexts) for batch in (1, 3, 6)]


def _ordered(points: list, order: str) -> list:
    if order == "shuffled":
        shuffled = list(points)
        random.Random(20261114).shuffle(shuffled)
        return shuffled
    return sorted(points, key=lambda point: point[::-1], reverse=order == "descending")


def _rounds(index: int) -> int:
    """Every third point is re-staged: a second full round of seed, warm shot and measured shot."""
    return 2 if index % 3 == 0 else 1


@pytest.mark.parametrize("generated_in_range", [False, True], ids=["generated-outside", "generated-inside"])
@pytest.mark.parametrize("one_by_one", [False, True], ids=["one-batch", "split-batches"])
@pytest.mark.parametrize("order", ["ascending", "descending", "shuffled"])
@pytest.mark.parametrize("page_size", [1, 16, 64])
def test_simulated_run_gets_the_planned_cached_tokens(page_size, order, one_by_one, generated_in_range):
    _simulate_and_check(content.ContentPlan("0", LOW, HIGH), page_size, order, one_by_one, generated_in_range)


def _simulate_and_check(plan, page_size: int, order: str, one_by_one: bool, generated_in_range: bool) -> None:
    """Drive ``plan`` (over ``[LOW, HIGH)``) through a prefill sweep and a decode sweep on the cache model."""
    # "generated-outside": the six stored generated tokens of a decode run are ids above the token range and
    # above the range of every FAKE text pool, whose text can hold ids outside the token range. So they match
    # nothing. "generated-inside": they repeat the slot's own next tokens for 0, 1, ... 6 tokens (in
    # turn, so six runs in seven continue the slot's sequence) and go on with random ids of the range. The test
    # builds those tokens, so it knows how deep every slot is stored and every count is asserted exactly.
    caches = [PageCacheModel(page_size) for _ in range(RANKS)]
    # Whole-page tokens of each slot's own sequence that the test knows to be stored: (rank, slot) -> depth.
    stored_depth: dict[tuple[int, int], int] = {}
    # The same depth counted from the prompts alone, as if no generated token had continued a sequence.
    prompt_depth: dict[tuple[int, int], int] = {}
    seeds_beyond_prompts = 0  # seeds that found more cached tokens than the prompts alone had stored
    outside = itertools.count(HIGH + FAKE_POOL_IDS_ABOVE)
    inside = random.Random(7)
    continued = itertools.cycle(range(7))

    def check_seeds(rank: int, prompts: list[list[int]]) -> None:
        nonlocal seeds_beyond_prompts
        counts = _submit(caches[rank], prompts, one_by_one)
        for slot, (prompt, count) in enumerate(zip(prompts, counts, strict=True)):
            limit = _floor(len(prompt) - 1, page_size)
            assert count == min(limit, stored_depth.get((rank, slot), 0)), (rank, slot, len(prompt))
            seeds_beyond_prompts += count > min(limit, prompt_depth.get((rank, slot), 0))
            for depths in (stored_depth, prompt_depth):
                depths[(rank, slot)] = max(depths.get((rank, slot), 0), _floor(len(prompt), page_size))

    def own_next_tokens(rank: int, slot: int, length: int) -> list[int]:
        return plan.slot_prefix(rank, slot, length + 6)[length:]

    def generated_tokens(rank: int, slot: int, length: int) -> list[int]:
        """The six generated tokens that a decode run with a prompt of ``length`` tokens leaves in the cache."""
        if not generated_in_range:
            return [next(outside) for _ in range(6)]
        repeated = next(continued)
        return [
            *own_next_tokens(rank, slot, length)[:repeated],
            *(inside.randrange(LOW, HIGH) for _ in range(6 - repeated)),
        ]

    def record_decode_run(rank: int, slot: int, length: int, generated: list[int]) -> None:
        """A decode run stored its prompt and ``generated``: note how deep the slot's own sequence now is."""
        if generated_in_range:  # compared token by token, so a random id that happens to continue counts too
            own = own_next_tokens(rank, slot, length)
            matching = next((index for index in range(6) if generated[index] != own[index]), 6)
            stored_depth[(rank, slot)] = max(stored_depth[(rank, slot)], _floor(length + matching, page_size))

    prefill = _ordered(_prefill_points(page_size), order)
    for index, (batch, new, cached) in enumerate(prefill):
        for _ in range(_rounds(index)):
            for rank in range(RANKS):
                if cached > 0:
                    seeds = [plan.prefill_seed_prompt(rank, slot, cached) for slot in range(batch)]
                    check_seeds(rank, seeds)
                for stage in ("warm", "meas"):
                    shots = [plan.prefill_prompt(rank, slot, cached, new) for slot in range(batch)]
                    counts = _submit(caches[rank], shots, one_by_one)
                    assert counts == [cached] * batch, (stage, rank, batch, new, cached)
                    assert [len(shot) - count for shot, count in zip(shots, counts, strict=True)] == [new] * batch

    decode = _ordered(_decode_points(page_size), order)
    for index, (batch, context) in enumerate(decode):
        for _ in range(_rounds(index)):
            for rank in range(RANKS):
                prompts = [plan.decode_prompt(rank, slot, context) for slot in range(batch)]
                check_seeds(rank, prompts)
                run = [plan.decode_prompt(rank, slot, context) for slot in range(batch)]
                assert run == prompts
                counts = []
                tails = [generated_tokens(rank, slot, context - 3) for slot in range(batch)]
                if one_by_one:
                    for prompt, tail in zip(run, tails, strict=True):
                        counts.append(caches[rank].cached(prompt))
                        caches[rank].finish_decode(prompt, tail)
                else:
                    counts = [caches[rank].cached(prompt) for prompt in run]
                    for prompt, tail in zip(run, tails, strict=True):
                        caches[rank].finish_decode(prompt, tail)
                assert counts == [_floor(context - 4, page_size)] * batch, (rank, batch, context)
                extend = [len(prompt) - count for prompt, count in zip(run, counts, strict=True)]
                assert all(1 <= tokens <= page_size for tokens in extend)
                for slot, tail in enumerate(tails):
                    record_decode_run(rank, slot, context - 3, tail)

    # The generated-inside variant must have reached the case it exists for: a seed that found pages which only
    # generated tokens had completed, while every decode run above still got its planned count. Deepest first
    # that cannot happen: every later prompt of a slot is a prefix of a stored prompt and is cached up to its
    # limit by the prompts alone.
    assert (seeds_beyond_prompts > 0) == (generated_in_range and order != "descending")

    # The plan's own counts, against counts taken from the point lists.
    cached_points = [(batch, _rounds(index)) for index, (batch, _, cached) in enumerate(prefill) if cached > 0]
    zero_points = [(batch, _rounds(index)) for index, (batch, _, cached) in enumerate(prefill) if cached == 0]
    decode_rounds = [(batch, _rounds(index)) for index, (batch, _) in enumerate(decode)]
    stats = plan.stats()
    assert stats["seed_prompts"] == RANKS * sum(batch * rounds for batch, rounds in cached_points)
    assert stats["shots"] == RANKS * 2 * sum(batch * rounds for batch, rounds in cached_points)
    assert stats["zero_cached_prompts"] == RANKS * 2 * sum(batch * rounds for batch, rounds in zero_points)
    assert stats["decode_prompts"] == RANKS * 2 * sum(batch * rounds for batch, rounds in decode_rounds)
    assert stats["free_prompts"] == 0
    assert stats["slots"] == RANKS * 6  # the largest batch that uses slot sequences
    for rank in range(RANKS):
        assert plan.remaining_zero_cached(rank) == (HIGH - LOW) - 6 - stats["zero_cached_prompts"] // RANKS


def test_simulated_run_with_a_wide_batch_and_free_prompts():
    _simulate_a_wide_batch(content.ContentPlan("wide", LOW, SMALL_HIGH))


def _simulate_a_wide_batch(plan) -> None:
    # 300 slots at page size 1, where a single shared token would show: warm-up, a k == 0 point, a cached
    # point at the smallest k, and a decode point, each split over several extend batches.
    cache = PageCacheModel(1)
    warm_up = plan.free_prompt(0, 256)
    assert cache.cached(warm_up) == 0
    cache.finish_prefill(warm_up)
    for cached, new in ((0, 3), (1, 2), (16, 2)):  # two or more new tokens, so a longer match could show
        if cached:
            _submit(cache, [plan.prefill_seed_prompt(0, slot, cached) for slot in range(300)], True)
        for _ in range(2):
            shots = [plan.prefill_prompt(0, slot, cached, new) for slot in range(300)]
            assert _submit(cache, shots, True) == [cached] * 300
    prompts = [plan.decode_prompt(0, slot, 40) for slot in range(300)]
    _submit(cache, prompts, True)
    assert [cache.cached(prompt) for prompt in prompts] == [36] * 300
    another = plan.free_prompt(0, 8)
    assert cache.cached(another) == 0


# ---------------------------------------------------------------------------------------------------------
# Exhaustion: the plan raises instead of repeating a first token
# ---------------------------------------------------------------------------------------------------------


def _assert_refused(plan, calls) -> None:
    """Each call raises ValueError and leaves the plan's counts as they were."""
    before = (plan.stats(), [plan.remaining_zero_cached(rank) for rank in range(RANKS)])
    for call in calls:
        with pytest.raises(ValueError, match="exhausted"):
            call()
    assert (plan.stats(), [plan.remaining_zero_cached(rank) for rank in range(RANKS)]) == before


def test_zero_cached_range_is_exhausted_without_repeating_a_first_token():
    plan = content.ContentPlan("s", LOW, SMALL_HIGH)
    cache = PageCacheModel(1)  # page size 1: one repeated first token would be one cached token
    slot_firsts = set()
    for slot in range(100):
        prefix = plan.slot_prefix(0, slot, 2)
        cache.finish_prefill(prefix)
        slot_firsts.add(prefix[0])
    assert plan.remaining_zero_cached(0) == 4096 - 100
    zero_firsts = []
    for index in range(4096 - 100):  # prefill and free prompts share one counter
        prompt = plan.free_prompt(0, 2) if index % 3 == 0 else plan.prefill_prompt(0, index % 7, 0, 2)
        assert cache.cached(prompt) == 0
        cache.finish_prefill(prompt)
        zero_firsts.append(prompt[0])
    assert len(set(zero_firsts)) == 4096 - 100
    assert not set(zero_firsts) & slot_firsts
    assert set(zero_firsts) | slot_firsts == set(range(LOW, SMALL_HIGH))  # every id is taken exactly once
    assert plan.remaining_zero_cached(0) == 0

    _assert_refused(
        plan,
        [
            lambda: plan.prefill_prompt(0, 0, 0, 1),
            lambda: plan.free_prompt(0, 3),
            # A higher slot would take the first token of a zero-cached prompt: refused through every method.
            lambda: plan.slot_prefix(0, 100, 1),
            lambda: plan.prefill_seed_prompt(0, 100, 16),
            lambda: plan.prefill_prompt(0, 100, 16, 1),
            lambda: plan.decode_prompt(0, 100, 4),
        ],
    )
    # Slots that already exist keep working, at any depth.
    assert plan.slot_prefix(0, 99, 50)[0] in slot_firsts
    assert len(plan.prefill_prompt(0, 99, 16, 2)) == 18
    assert len(plan.decode_prompt(0, 0, 9)) == 6

    # Another rank has its own range.
    assert plan.remaining_zero_cached(1) == 4096
    plan.slot_prefix(1, 4095, 1)
    assert plan.remaining_zero_cached(1) == 0
    _assert_refused(plan, [lambda: plan.free_prompt(1, 1), lambda: plan.prefill_prompt(1, 0, 0, 1)])


def test_slot_indices_are_bounded_by_the_range_and_by_issued_zero_cached_prompts():
    plan = content.ContentPlan("s", LOW, SMALL_HIGH)
    _assert_refused(plan, [lambda: plan.slot_prefix(0, 4096, 1), lambda: plan.decode_prompt(0, 5000, 4)])
    for _ in range(10):
        plan.free_prompt(0, 1)
    _assert_refused(plan, [lambda: plan.slot_prefix(0, 4086, 1)])  # 4087 slots + 10 prompts > 4096
    assert len(plan.slot_prefix(0, 4085, 3)) == 3  # 4086 slots + 10 prompts == 4096
    assert plan.remaining_zero_cached(0) == 0
    _assert_refused(plan, [lambda: plan.free_prompt(0, 1)])


def _zero_cached_firsts(plan, count: int) -> list[int]:
    """First tokens of the next ``count`` zero-cached prompts of rank 0, prefill and free prompts in turn."""
    return [(plan.free_prompt(0, 1) if index % 2 else plan.prefill_prompt(0, 0, 0, 1))[0] for index in range(count)]


def test_a_rank_without_slots_issues_exactly_one_zero_cached_prompt_per_token_id():
    # A corner of the exhaustion rule: no slot sequence on the rank (a run of k == 0 points only). The range
    # holds 4,096 ids, so exactly 4,096 zero-cached prompts fit and the next one would repeat a first token.
    plan = content.ContentPlan("s", LOW, SMALL_HIGH)
    assert plan.remaining_zero_cached(0) == 4096
    firsts = _zero_cached_firsts(plan, 4096)
    assert sorted(firsts) == list(range(LOW, SMALL_HIGH))  # every id exactly once
    assert plan.stats()["slots"] == 0
    assert plan.remaining_zero_cached(0) == 0
    _assert_refused(
        plan,
        [
            lambda: plan.free_prompt(0, 1),
            lambda: plan.prefill_prompt(0, 0, 0, 1),
            lambda: plan.slot_prefix(0, 0, 1),  # slot 0 would need a first token that a prompt already has
            lambda: plan.decode_prompt(0, 0, 4),
        ],
    )


def test_a_rank_with_only_slot_0_keeps_its_first_token_out_of_the_zero_cached_prompts():
    # The other corner: slot 0 is the only slot in use (a run of batch size 1). It takes one of the 4,096 ids,
    # so exactly 4,095 zero-cached prompts fit and none of them starts with the slot's first token.
    plan = content.ContentPlan("s", LOW, SMALL_HIGH)
    slot_first = plan.slot_prefix(0, 0, 2)[0]
    assert plan.remaining_zero_cached(0) == 4095
    firsts = _zero_cached_firsts(plan, 4095)
    assert slot_first not in firsts
    assert sorted([*firsts, slot_first]) == list(range(LOW, SMALL_HIGH))  # every id exactly once
    assert plan.remaining_zero_cached(0) == 0
    _assert_refused(
        plan,
        [
            lambda: plan.free_prompt(0, 1),
            lambda: plan.prefill_prompt(0, 0, 0, 1),
            lambda: plan.slot_prefix(0, 1, 1),  # a second slot would take the first token of a prompt
        ],
    )
    assert plan.slot_prefix(0, 0, 9)[0] == slot_first  # the slot that exists keeps working


def test_zero_cached_prefill_prompt_does_not_use_its_slot():
    plan = content.ContentPlan("s", LOW, SMALL_HIGH)
    plan.prefill_prompt(0, 4000, 0, 3)
    assert plan.remaining_zero_cached(0) == 4095
    assert plan.stats()["slots"] == 0
    assert plan.stats()["zero_cached_prompts"] == 1


def test_shot_range_is_exhausted_without_repeating_a_first_suffix_token():
    plan = content.ContentPlan("s", LOW, SMALL_HIGH)
    cache = PageCacheModel(1)
    deeper_seed = plan.slot_prefix(0, 0, 32)  # the cache holds the slot beyond the 16 tokens of this point
    cache.finish_prefill(deeper_seed)
    own_next = deeper_seed[16]
    firsts = []
    for _ in range(4095):  # every shot the range allows, each one stored before the next arrives
        shot = plan.prefill_prompt(0, 0, 16, 2)
        assert cache.cached(shot) == 16
        cache.finish_prefill(shot)
        firsts.append(shot[16])
    assert len(set(firsts)) == 4095
    assert own_next not in firsts
    assert set(firsts) | {own_next} == set(range(LOW, SMALL_HIGH))  # every id but the slot's next token
    _assert_refused(plan, [lambda: plan.prefill_prompt(0, 0, 16, 1), lambda: plan.prefill_prompt(0, 0, 16, 9)])
    # Other prefix lengths, slots and ranks have their own ranges.
    assert len(plan.prefill_prompt(0, 0, 32, 1)) == 33
    assert len(plan.prefill_prompt(0, 1, 16, 1)) == 17
    assert len(plan.prefill_prompt(1, 0, 16, 1)) == 17


# ---------------------------------------------------------------------------------------------------------
# The known limit: a prefill shot after a decode run on the same slot
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("context", [4, 40])
def test_prefill_shot_inside_a_decode_tail_is_refused(context):
    plan = content.ContentPlan("0", LOW, HIGH)
    length = context - 3
    plan.decode_prompt(0, 2, context)
    before = plan.stats()
    for cached in range(length, length + 6):
        with pytest.raises(ValueError, match="decode run"):
            plan.prefill_prompt(0, 2, cached, 4)
    assert plan.stats() == before
    # Outside the six stored generated tokens the shot is safe: below the decode prompt the cache holds the
    # slot's own next token, and from length + 6 on the decode run stored nothing.
    assert len(plan.prefill_prompt(0, 2, length + 6, 4)) == length + 10
    if length > 1:
        assert len(plan.prefill_prompt(0, 2, length - 1, 4)) == length + 3
    # Other slots and ranks, zero-cached prompts, seeds and further decode runs are not affected.
    assert len(plan.prefill_prompt(0, 3, length, 4)) == length + 4
    assert len(plan.prefill_prompt(1, 2, length, 4)) == length + 4
    assert len(plan.prefill_prompt(0, 2, 0, 4)) == 4
    assert len(plan.prefill_seed_prompt(0, 2, length)) == length
    assert len(plan.decode_prompt(0, 2, context + 2)) == length + 2
    with pytest.raises(ValueError, match="decode run"):
        plan.prefill_prompt(0, 2, length + 6, 4)  # now inside the tail of the second decode run


@pytest.mark.parametrize(
    ("page_size", "length", "gap"),
    [(1, 37, 0), (1, 37, 5), (2, 36, 4), (4, 38, 2), (6, 36, 0), (7, 35, 0), (16, 32, 0), (64, 64, 0)],
)
def test_generated_tokens_of_a_decode_run_are_why_the_guard_exists(page_size, length, gap):
    # A twin plan with no decode run shows the prompt the guarded plan would issue (same seed, same key).
    cached = length + gap
    assert cached % page_size == 0
    twin = content.ContentPlan("0", LOW, HIGH)
    shot = twin.prefill_prompt(0, 0, cached, 8)
    decode = twin.slot_prefix(0, 0, length)
    # Worst case: the model continues the slot's own sequence and then emits the shot's suffix.
    adversarial = shot[length : length + 6]
    cache = PageCacheModel(page_size)
    cache.finish_prefill(decode)
    assert cache.cached(decode) == _floor(length - 1, page_size)
    cache.finish_decode(decode, adversarial)
    cache.finish_prefill(twin.prefill_seed_prompt(0, 0, cached))
    # The shot matches every whole page the decode run stored: more than planned at page sizes up to 6.
    assert cache.cached(shot) == max(cached, _floor(length + 6, page_size))
    assert (cache.cached(shot) > cached) == (page_size <= 6)

    guarded = content.ContentPlan("0", LOW, HIGH)
    assert guarded.decode_prompt(0, 0, length + 3) == decode
    with pytest.raises(ValueError, match="decode run"):
        guarded.prefill_prompt(0, 0, cached, 8)


@pytest.mark.parametrize("page_size", [1, 2])
def test_shots_just_outside_a_decode_tail_stay_exact(page_size):
    # The shots the guard lets through, against the worst generated tokens for them: the model continues the
    # slot's own sequence, so the cache holds the slot six tokens deeper than the decode prompt.
    plan = content.ContentPlan("0", LOW, HIGH)
    cache = PageCacheModel(page_size)
    length = 36
    decode = plan.decode_prompt(0, 0, length + 3)
    cache.finish_prefill(decode)
    cache.finish_decode(decode, plan.slot_prefix(0, 0, length + 6)[length:])
    for cached in (length - page_size, length + 6):
        cache.finish_prefill(plan.prefill_seed_prompt(0, 0, cached))
        for _ in range(3):
            shot = plan.prefill_prompt(0, 0, cached, page_size + 2)
            assert cache.cached(shot) == cached
            cache.finish_prefill(shot)


# ---------------------------------------------------------------------------------------------------------
# Text: the recipe that turns dataset records into texts
# ---------------------------------------------------------------------------------------------------------

FAKE_QUESTION = "How does a prefix cache decide which tokens it can reuse?"
FAKE_ANSWER = "It compares the new prompt with stored prompts, page by page."
FAKE_TWO_TURNS = f"{FAKE_QUESTION}\n{FAKE_ANSWER}"
FAKE_KEPT_TURN = "A single turn that is long enough to be kept by the length rule on its own, with room."
FAKE_ODD_TURN = "Another single turn of fake text, also longer than sixty-four characters in total."
FAKE_NUMBER_TURN = "A turn whose value is a number is written as text, like every other value."
# The first byte of the SHA-256 of each FAKE text, worked out with hashlib outside the module. An even byte
# puts a text into the kept half.
FAKE_FIRST_BYTES = {
    FAKE_TWO_TURNS: 114,
    FAKE_KEPT_TURN: 128,
    FAKE_ODD_TURN: 11,
    f"{FAKE_NUMBER_TURN}\n7": 48,
    "x" * 64: 124,
    "y" * 64: 255,
    "v" * 63: 86,
}


def test_the_text_recipe_joins_the_turn_values_and_keeps_the_even_half_of_the_long_texts():
    for text, first_byte in FAKE_FIRST_BYTES.items():
        assert hashlib.sha256(text.encode("utf-8")).digest()[0] == first_byte
    records = [
        # Kept: the values of the turns, joined by a newline. No other key is read.
        {"id": "fake-0", "conversations": [{"from": "human", "value": FAKE_QUESTION}, {"value": FAKE_ANSWER}]},
        # Dropped: long enough, but in the odd half.
        {"conversations": [{"from": "human", "value": FAKE_ODD_TURN}]},
        {"conversations": [{"value": FAKE_KEPT_TURN}]},
        # Dropped: 63 characters, although its hash is even.
        {"conversations": [{"value": "v" * 63}]},
        # Kept: 64 characters are enough. The next one has 64 too and is in the odd half.
        {"conversations": [{"value": "x" * 64}]},
        {"conversations": [{"value": "y" * 64}]},
        # Kept, and the same text as the first record: a turn without a value adds nothing, not even a newline.
        # That is a turn without the key and a turn whose value is an empty string, None or zero.
        {
            "conversations": [
                {"from": "system"},
                {"value": FAKE_QUESTION},
                {"value": ""},
                {"value": None},
                {"value": 0},
                {"value": FAKE_ANSWER},
            ]
        },
        # Kept: a value that is not a string is written as one.
        {"conversations": [{"value": FAKE_NUMBER_TURN}, {"value": 7}]},
        # Dropped: records without a turn have no text.
        {"conversations": []},
        {"conversations": None},
        {"id": "fake-no-turns"},
        {},
        # Kept: 64 characters. The last one cannot be encoded as UTF-8 and is left out of the hashed bytes,
        # which are those of the 63-character text above; the text itself keeps it.
        {"conversations": [{"value": "v" * 63 + "\ud800"}]},
    ]
    before = repr(records)

    assert content.conversation_texts(records) == [
        FAKE_TWO_TURNS,
        FAKE_KEPT_TURN,
        "x" * 64,
        FAKE_TWO_TURNS,
        f"{FAKE_NUMBER_TURN}\n7",
        "v" * 63 + "\ud800",
    ]
    assert repr(records) == before  # the records are read, not changed
    # A dataset without a kept text gives an empty list; what that means is the caller's decision.
    assert content.conversation_texts([]) == []
    assert content.conversation_texts([records[1], records[3], records[5], {}]) == []
    assert (content.MIN_TEXT_CHARACTERS, content.COLLECTION_POOL) == (64, "conversation_sha256_even")


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ({"conversations": []}, r"^the dataset must be a list of records, got dict$"),
        ("[]", r"^the dataset must be a list of records, got str$"),
        (None, r"^the dataset must be a list of records, got NoneType$"),
        (({"conversations": []},), r"^the dataset must be a list of records, got tuple$"),
        # The first bad record is the one that is named.
        ([{"conversations": []}, "text", 7], r"^record 1 of the dataset must be a mapping, got str$"),
        ([{}, {}, None, 7], r"^record 2 of the dataset must be a mapping, got NoneType$"),
        ([[{"value": "text"}]], r"^record 0 of the dataset must be a mapping, got list$"),
        ([{"conversations": "text"}], r"^record 0 of the dataset: 'conversations' must be a list of turns, got str$"),
        (
            [{}, {"conversations": {"value": "text"}}],
            r"^record 1 of the dataset: 'conversations' must be a list of turns, got dict$",
        ),
        ([{"conversations": 5}], r"^record 0 of the dataset: 'conversations' must be a list of turns, got int$"),
        (
            [{"conversations": [{"value": "text"}, "text"]}],
            r"^record 0 of the dataset: turn 1 must be a mapping, got str$",
        ),
        ([{}, {"conversations": [None]}], r"^record 1 of the dataset: turn 0 must be a mapping, got NoneType$"),
        ([{"conversations": [["value", "text"]]}], r"^record 0 of the dataset: turn 0 must be a mapping, got list$"),
    ],
)
def test_the_text_recipe_refuses_records_it_cannot_read(records, message):
    with pytest.raises(ValueError, match=message):
        content.conversation_texts(records)


# ---------------------------------------------------------------------------------------------------------
# Text: the pool of tokenized conversations
# ---------------------------------------------------------------------------------------------------------


def _fake_conversations(seed: int, count: int, token_low: int, token_high: int, shortest: int = 1) -> list[list[int]]:
    """``count`` FAKE conversations of ``shortest`` to 40 random ids of the range each."""
    rng = random.Random(seed)
    return [[rng.randrange(token_low, token_high) for _ in range(rng.randint(shortest, 40))] for _ in range(count)]


def _pool_facts(pool) -> tuple:
    """Everything a pool shows of itself."""
    return (
        (pool.token_low, pool.token_high, pool.frozen),
        (pool.conversations, pool.tokens, pool.dropped_tokens, pool.dropped_conversations),
        [pool.conversation(index).tolist() for index in range(pool.conversations)],
    )


def test_a_pool_keeps_the_ids_inside_the_range_and_counts_what_it_drops():
    conversations = [
        [LOW, LOW + 1, HIGH - 1],  # both ends of the range are kept
        [LOW - 1, HIGH, 2**40, -5],  # no id inside the range: the conversation is dropped
        [],  # no id at all: dropped as well
        (LOW + 5, HIGH + 3, LOW + 6, 0),  # a tuple; the ids outside are left out, the others keep their order
        array("I", [LOW + 7, LOW + 7]),
    ]
    before = repr(conversations)
    pool = content.TextPool(conversations, LOW, HIGH)

    assert _pool_facts(pool) == (
        (LOW, HIGH, False),
        (3, 7, 6, 2),
        [[LOW, LOW + 1, HIGH - 1], [LOW + 5, LOW + 6], [LOW + 7, LOW + 7]],
    )
    assert repr(conversations) == before
    for index in range(3):
        stored = pool.conversation(index)
        assert type(stored) is array and stored.typecode == "I"
        assert pool.conversation(index) is stored  # the stored object, not a copy
    assert pool.conversation(2) is not conversations[4]

    # Further conversations follow the same rules, from any iterable.
    pool.add(ids for ids in ([HIGH, LOW + 9], [HIGH + 1]))
    assert _pool_facts(pool)[1:] == (
        (4, 8, 8, 3),
        [[LOW, LOW + 1, HIGH - 1], [LOW + 5, LOW + 6], [LOW + 7, LOW + 7], [LOW + 9]],
    )
    pool.add([])
    assert _pool_facts(pool)[1][0] == 4


def test_a_pool_may_be_empty_and_uses_the_whole_range_of_32_bit_ids():
    empty = content.TextPool([], LOW, HIGH)
    assert _pool_facts(empty) == ((LOW, HIGH, False), (0, 0, 0, 0), [])
    for token_low, token_high in ((0, 4096), (2**32 - 4096, 2**32), (0, 2**32)):
        pool = content.TextPool([[token_low, token_high - 1, token_high, -1]], token_low, token_high)
        assert _pool_facts(pool) == ((token_low, token_high, False), (1, 2, 2, 0), [[token_low, token_high - 1]])


@pytest.mark.parametrize(
    ("token_range", "error"),
    [
        ((True, 4096), TypeError),
        ((0.0, 4096), TypeError),
        (("0", 4096), TypeError),
        ((0, 4096.0), TypeError),
        ((0, None), TypeError),
        ((-1, 4096), ValueError),
        ((0, 4095), ValueError),  # one id short of the minimum range
        ((5000, 5000), ValueError),
        ((5000, 100), ValueError),
        ((0, 2**32 + 1), ValueError),  # ids are stored in 32 bits
    ],
)
def test_a_pool_validates_its_range_as_a_plan_does(token_range, error):
    with pytest.raises(error) as of_the_pool:
        content.TextPool([], *token_range)
    with pytest.raises(error) as of_the_plan:
        content.ContentPlan("0", *token_range)
    assert str(of_the_pool.value) == str(of_the_plan.value)


@pytest.mark.parametrize(
    ("conversations", "message"),
    [
        (None, r"^conversations must be an iterable of lists of token ids, got NoneType$"),
        (5, r"^conversations must be an iterable of lists of token ids, got int$"),
        ([LOW, LOW + 1], r"^conversations\[0\] must be a list of token ids, got int$"),  # one flat list of ids
        ("text", r"^conversations\[0\] must be a list of token ids, got str$"),
        ([[LOW], "text"], r"^conversations\[1\] must be a list of token ids, got str$"),
        ([[LOW], [LOW], None], r"^conversations\[2\] must be a list of token ids, got NoneType$"),
        ([[LOW], {LOW + 1}], r"^conversations\[1\] must be a list of token ids, got set$"),
        ([[LOW], {LOW: 1}], r"^conversations\[1\] must be a list of token ids, got dict$"),
        ([[LOW, True]], r"^conversations\[0\] holds a token id that is not an int: bool: True$"),
        ([[LOW], [LOW, 1.0, "2"]], r"^conversations\[1\] holds a token id that is not an int: float: 1\.0$"),
        ([[str(LOW)]], r"^conversations\[0\] holds a token id that is not an int: str: '1000'$"),
        ([[LOW, None]], r"^conversations\[0\] holds a token id that is not an int: NoneType: None$"),
        ([array("d", [1000.0])], r"^conversations\[0\] holds a token id that is not an int: float: 1000\.0$"),
    ],
)
def test_a_pool_refuses_what_is_not_lists_of_plain_ints_and_changes_nothing(conversations, message):
    with pytest.raises(TypeError, match=message):
        content.TextPool(conversations, LOW, HIGH)
    pool = content.TextPool([[LOW, HIGH]], LOW, HIGH)
    before = _pool_facts(pool)
    with pytest.raises(TypeError, match=message):
        pool.add(conversations)
    # The good conversations ahead of the bad one were not added either.
    assert _pool_facts(pool) == before


def test_a_pool_gives_a_conversation_by_its_index_only():
    pool = content.TextPool([[LOW], [LOW + 1, LOW + 2]], LOW, HIGH)
    for bad in (True, 0.0, "0", None):
        with pytest.raises(TypeError, match="index must be an int"):
            pool.conversation(bad)
    with pytest.raises(ValueError, match=r"^index must be >= 0, got -1$"):
        pool.conversation(-1)
    with pytest.raises(ValueError, match=r"^index must be < 2, the number of conversations, got 2$"):
        pool.conversation(2)


def test_a_chain_raises_when_a_stored_conversation_was_shortened_behind_the_pool():
    # The pool hands out its stored arrays and asks callers not to change them. A caller that shortens one
    # must get an error from the next chain that runs out, not a search for an unused conversation that never
    # ends: the pool still counts four tokens and holds three.
    pool = content.TextPool([[LOW + 1, LOW + 2, LOW + 3], [LOW + 4]], LOW, HIGH)
    plan = content.ContentPlan("0", LOW, HIGH, pool)
    pool.conversation(0).pop()
    assert (pool.tokens, sum(len(pool.conversation(index)) for index in range(2))) == (4, 3)
    message = (
        r"^the text pool no longer holds the 4 tokens it counted: the chain slot:0:0 used every conversation "
        r"and is 1 token\(s\) short$"
    )
    with pytest.raises(RuntimeError, match=message):
        plan.slot_prefix(0, 0, 5)


def test_the_first_plan_built_on_a_pool_freezes_it():
    pool = content.TextPool(_fake_conversations(1, 20, LOW, HIGH), LOW, HIGH)
    pool.add([[LOW + 1, LOW + 2]])  # a pool grows in batches before a plan exists
    # A plan that cannot be built leaves the pool as it was.
    for arguments, error in (
        ((0, LOW, HIGH), TypeError),
        (("", LOW, HIGH), ValueError),
        (("0", LOW, HIGH + 1), ValueError),
    ):
        with pytest.raises(error):
            content.ContentPlan(*arguments, pool)
    assert pool.frozen is False

    plan = content.ContentPlan("0", LOW, HIGH, pool)
    assert pool.frozen is True
    before = _pool_facts(pool)
    for conversations in ([[LOW + 3]], [], None):
        with pytest.raises(RuntimeError, match="^the text pool is frozen: a ContentPlan was built on it"):
            pool.add(conversations)
    pool.freeze()  # freezing a frozen pool changes nothing
    assert _pool_facts(pool) == before
    # A frozen pool serves further plans, and the same seed then gives the same content.
    again = content.ContentPlan("0", LOW, HIGH, pool)
    assert again.slot_prefix(0, 3, 40) == plan.slot_prefix(0, 3, 40)
    assert again.free_prompt(1, 40) == plan.free_prompt(1, 40)


def test_a_plan_refuses_a_pool_it_cannot_use():
    # A pool without a conversation can be built, also one that dropped every conversation it was given.
    for conversations in ([], [[HIGH], [LOW - 1, HIGH + 1], []]):
        empty = content.TextPool(conversations, LOW, HIGH)
        assert (empty.conversations, empty.tokens) == (0, 0)
        with pytest.raises(ValueError, match="^the text pool holds no conversation$"):
            content.ContentPlan("0", LOW, HIGH, empty)
        # The refused plan did not freeze it: it can still grow, and a plan then takes it.
        assert empty.frozen is False
        empty.add([[LOW + 4]])
        assert content.ContentPlan("0", LOW, HIGH, empty).slot_prefix(0, 0, 2)[1:] == [LOW + 4]

    # The range of a pool must hold the token range of the plan: a narrower pool would leave ordinary ids out
    # of the text.
    pool = content.TextPool([[LOW]], LOW, HIGH)
    for token_low, token_high in ((LOW, HIGH + 1), (LOW - 1, HIGH), (LOW + 1, HIGH + 1)):
        message = (
            rf"^the text pool keeps the token ids \[{LOW}, {HIGH}\), which does not hold the plan's token range "
            rf"\[{token_low}, {token_high}\)$"
        )
        with pytest.raises(ValueError, match=message):
            content.ContentPlan("0", token_low, token_high, pool)
    # A plan whose range lies inside the pool's is taken, whichever end is narrower.
    for token_low, token_high in ((LOW, HIGH), (LOW + 1, HIGH), (LOW, HIGH - 1), (LOW + 7, HIGH - 7)):
        ids = [LOW, HIGH - 1, LOW + 3]
        plan = content.ContentPlan("0", token_low, token_high, content.TextPool([ids], LOW, HIGH))
        assert plan.slot_prefix(0, 0, 4)[1:] == ids
    for not_a_pool in ([[LOW]], (), "pool", 0, False):
        with pytest.raises(TypeError, match="^pool must be a TextPool or None, got "):
            content.ContentPlan("0", LOW, HIGH, not_a_pool)
    assert pool.frozen is False


def test_a_plan_names_its_source_and_its_pool():
    assert (content.CONTENT_SEEDED_IDS, content.CONTENT_TEXT_CHAINS) == ("seeded_token_ids", "dataset_text_chains")
    seeded = content.ContentPlan("0", LOW, HIGH)
    assert (seeded.source, seeded.pool) == ("seeded_token_ids", None)
    # Passing None for the pool is the plan without a pool.
    explicit = content.ContentPlan("0", LOW, HIGH, None)
    assert (explicit.source, explicit.pool) == ("seeded_token_ids", None)
    assert _recorded_calls(explicit) == _recorded_calls(seeded)

    pool = content.TextPool(_fake_conversations(2, 30, LOW, HIGH), LOW, HIGH)
    text = content.ContentPlan("0", LOW, HIGH, pool)
    assert (text.source, text.pool is pool) == ("dataset_text_chains", True)
    assert (text.seed, text.token_low, text.token_high, text.vocab_size) == ("0", LOW, HIGH, HIGH - LOW)
    # The counts of a plan have the same keys for both sources; the pool has its own counts.
    _recorded_calls(text)
    assert text.stats() == seeded.stats()
    for name in ("source", "pool"):
        with pytest.raises(AttributeError):
            setattr(text, name, None)


# ---------------------------------------------------------------------------------------------------------
# Text: chains against an independent restatement
# ---------------------------------------------------------------------------------------------------------

# A FAKE pool whose ids say which conversation they come from.
PINNED_POOL = [[11, 12, 13], [21, 22], [31, 32, 33, 34], [41], [51, 52, 53, 54, 55], [61, 62], [71, 72, 73]]


def test_pinned_vectors_identify_the_text_content_of_a_seed_and_a_pool():
    # As for the seeded ids above: a change here changes what a recorded seed means for a recorded dataset, so
    # it needs a new STREAM_DOMAIN version. The literals were computed with _reference_chain and
    # _reference_stream (not with the module) and are pinned because the module gave the same values.
    plan = content.ContentPlan("0", 0, 4096, content.TextPool(PINNED_POOL, 0, 4096))
    # The first token is allocated: it is the first token of the same slot with seeded ids (466, pinned
    # above). The 20 ids of the pool follow as whole conversations, each once.
    assert plan.slot_prefix(0, 0, 21) == [
        *(466, 71, 72, 73, 11, 12, 13, 41, 21, 22),
        *(51, 52, 53, 54, 55, 61, 62, 31, 32, 33, 34),
    ]
    # A zero-cached prompt: the allocated first token of the seeded prompt above (1987), then its own chain.
    assert plan.free_prompt(1, 12) == [1987, 21, 22, 11, 12, 13, 31, 32, 33, 34, 51, 52]
    # The first two shots of one (rank, slot, k): the slot's first four tokens, an allocated first suffix
    # token (the offset of (1, 2, 4) for shot 0 and the next id upwards for shot 1; the slot's own next token,
    # 12, is not near), then the chains of (1, 2, 4, 0) and of (1, 2, 4, 1).
    assert plan.prefill_prompt(1, 2, 4, 6) == [1990, 61, 62, 11, 4084, 31, 32, 33, 34, 51]
    assert plan.prefill_prompt(1, 2, 4, 6) == [1990, 61, 62, 11, 4085, 71, 72, 73, 51, 52]
    # A seed of two characters and three UTF-8 bytes: the length in front of the seed counts bytes.
    other = content.ContentPlan("s\u00e9", 0, 4096, content.TextPool(PINNED_POOL, 0, 4096))
    assert other.slot_prefix(0, 0, 10) == [3506, 21, 22, 71, 72, 73, 31, 32, 33, 34]


def _reference_shot_firsts(token_low: int, token_high: int, start: int, next_token: int) -> list[int]:
    """Every first suffix token of one ``(rank, slot, k)`` in shot order, restated from "Content layout".

    The token range as a circle from the offset, without one element: the slot's own next token, or, when that
    is a text id outside the range, the id of the range it falls on modulo the size of the range.
    """
    vocab_size = token_high - token_low
    circle = [token_low + (start + step) % vocab_size for step in range(vocab_size)]
    circle.remove(token_low + (next_token - token_low) % vocab_size)
    return circle


def _reference_first_suffix(token_low: int, vocab_size: int, start: int, own_next: int, shot: int) -> int:
    """The rule in words: walk the range as a circle from the offset, skip the slot's own next token.

    For a next token of the token range. ``_reference_shot_firsts`` also covers one outside it.
    """
    step = 0
    remaining = shot
    while True:
        candidate = token_low + (start + step) % vocab_size
        step += 1
        if candidate == own_next:
            continue
        if remaining == 0:
            return candidate
        remaining -= 1


@pytest.mark.parametrize(
    ("seed", "token_low", "token_high", "conversations"),
    [
        ("0", 0, 4096, 1),  # one conversation: every chain is that conversation
        ("0", 0, 4096, 2),
        ("run-7", LOW, HIGH, 37),
        ("s\u00e9ed:with:colons", 5, 5 + 151_643, 64),  # non-ASCII: the length prefix counts UTF-8 bytes
        ("0", 2**32 - 4096, 2**32, 100),  # the top of the 32-bit id space
    ],
)
def test_text_content_equals_the_documented_chains(seed, token_low, token_high, conversations):
    fake = _fake_conversations(20261117, conversations, token_low, token_high, shortest=30)
    total = sum(len(ids) for ids in fake)
    plan = content.ContentPlan(seed, token_low, token_high, content.TextPool(fake, token_low, token_high))
    vocab_size = token_high - token_low
    zero_issued = dict.fromkeys(range(3), 0)  # the zero-cached counter runs per rank, across slots
    for rank, slot in ((0, 0), (0, 3), (1, 0), (2, 41)):
        # The allocation is that of the seeded ids: the rotation and the offsets are streams of token ids.
        rotation = _reference_rotation(seed, token_low, token_high, rank)
        chain = _reference_chain(seed, fake, "slot", (rank, slot))
        assert len(chain) == total and sorted(chain) == sorted(token for ids in fake for token in ids)
        sequence = [token_low + (rotation + slot) % vocab_size, *chain]
        assert plan.slot_prefix(rank, slot, total + 1) == sequence

        for cached in (1, 16, 29):
            start = _reference_stream(seed, token_low, token_high, "off", (rank, slot, cached), 1)[0][0] - token_low
            for shot in range(4):
                first = _reference_first_suffix(token_low, vocab_size, start, sequence[cached], shot)
                suffix = _reference_chain(seed, fake, "shot", (rank, slot, cached, shot))[:6]
                assert plan.prefill_prompt(rank, slot, cached, 7) == [*sequence[:cached], first, *suffix]

        for _ in range(3):  # prefill and free prompts take turns on one counter
            index = zero_issued[rank]
            zero_tail = _reference_chain(seed, fake, "zero", (rank, index))[:4]
            first = token_low + (rotation - 1 - index) % vocab_size
            assert plan.prefill_prompt(rank, slot, 0, 5) == [first, *zero_tail]
            zero_tail = _reference_chain(seed, fake, "zero", (rank, index + 1))[:2]
            first = token_low + (rotation - 2 - index) % vocab_size
            assert plan.free_prompt(rank, 3) == [first, *zero_tail]
            zero_issued[rank] = index + 2


def test_a_chain_does_not_depend_on_how_it_is_consumed():
    fake = _fake_conversations(7, 50, LOW, HIGH)
    total = sum(len(ids) for ids in fake)
    assert total > 600

    def new_plan():
        return content.ContentPlan("0", LOW, HIGH, content.TextPool(fake, LOW, HIGH))

    whole = new_plan().slot_prefix(0, 5, total + 1)
    assert whole[1:] == _reference_chain("0", fake, "slot", (0, 5))
    piecewise = new_plan()
    for length in (1, 2, 3, 9, 10, 17, 64, 65, total, total + 1):
        assert piecewise.slot_prefix(0, 5, length) == whole[:length]
    backwards = new_plan()
    for length in (total + 1, 64, 1):
        assert backwards.slot_prefix(0, 5, length) == whole[:length]
    token_by_token = new_plan()
    assert [token_by_token.slot_prefix(0, 5, length)[-1] for length in range(1, 301)] == whole[:300]
    # A shot needs seq[cached]; asking for it first must not change the sequence either.
    lookahead = new_plan()
    assert lookahead.prefill_prompt(0, 5, 500, 2)[:500] == whole[:500]
    assert lookahead.slot_prefix(0, 5, total + 1) == whole


def test_a_chain_is_every_conversation_of_the_pool_once_in_an_order_of_its_own():
    # Conversation i holds ids from LOW + 100 * i on, so a chain shows which conversations it is made of.
    fake = [[LOW + 100 * index + position for position in range(1 + index % 7)] for index in range(40)]
    pool = content.TextPool(fake, LOW, HIGH)
    plan = content.ContentPlan("0", LOW, HIGH, pool)
    chains = [
        plan.slot_prefix(0, 0, pool.tokens + 1)[1:],
        plan.slot_prefix(0, 1, pool.tokens + 1)[1:],
        plan.slot_prefix(1, 0, pool.tokens + 1)[1:],
        plan.free_prompt(0, pool.tokens + 1)[1:],
        plan.prefill_prompt(0, 0, 0, pool.tokens + 1)[1:],
        plan.prefill_prompt(0, 0, 3, pool.tokens + 1)[4:],
        plan.prefill_prompt(0, 0, 3, pool.tokens + 1)[4:],
    ]
    orders = []
    for chain in chains:
        # Cut where the conversation changes: the pieces are whole conversations with nothing between them.
        pieces = [list(piece) for _, piece in itertools.groupby(chain, key=lambda token: (token - LOW) // 100)]
        assert sorted(pieces) == sorted(fake)  # every conversation, each exactly once
        orders.append(tuple((piece[0] - LOW) // 100 for piece in pieces))
    assert len(set(orders)) == len(orders)  # each key has its own order
    assert tuple(range(40)) not in orders  # and none is the order of the pool


def _plan_facts(plan) -> tuple:
    return (plan.stats(), [plan.remaining_zero_cached(rank) for rank in range(RANKS)])


def test_a_prompt_the_pool_is_too_short_for_is_refused_and_changes_nothing():
    fake = _fake_conversations(11, 12, LOW, HIGH)
    total = sum(len(ids) for ids in fake)

    def new_plan():
        plan = content.ContentPlan("0", LOW, HIGH, content.TextPool(fake, LOW, HIGH))
        plan.slot_prefix(0, 0, 5)
        plan.prefill_prompt(0, 1, 4, 3)
        plan.free_prompt(0, 3)
        plan.decode_prompt(1, 0, 9)
        return plan

    plan, twin = new_plan(), new_plan()
    before = _plan_facts(plan)
    # Each call needs one token more than a chain holds, and names the chain that is too short.
    refused = [
        (lambda: plan.slot_prefix(0, 0, total + 2), "slot:0:0"),  # a slot that exists
        (lambda: plan.slot_prefix(0, 7, total + 2), "slot:0:7"),  # a slot that would be new
        (lambda: plan.prefill_seed_prompt(0, 8, total + 2), "slot:0:8"),
        (lambda: plan.decode_prompt(1, 3, total + 5), "slot:1:3"),
        (lambda: plan.free_prompt(0, total + 2), "zero:0:1"),  # the second zero-cached prompt of rank 0
        (lambda: plan.prefill_prompt(0, 0, 0, total + 2), "zero:0:1"),
        (lambda: plan.prefill_prompt(0, 1, 4, total + 2), "shot:0:1:4:1"),  # the suffix of the second shot
        (lambda: plan.prefill_prompt(0, 9, 4, total + 2), "shot:0:9:4:0"),  # slot 9 is not made for it either
        (lambda: plan.prefill_prompt(0, 9, total + 1, 1), "slot:0:9"),  # the prefix and the slot's next token
    ]
    for call, key in refused:
        message = (
            rf"^text pool exhausted: the chain {key} needs {total + 1} tokens, but a chain uses each conversation "
            rf"once and the pool holds {total} tokens in 12 conversations$"
        )
        with pytest.raises(ValueError, match=message):
            call()
        assert _plan_facts(plan) == before

    # The refused calls took no first token, no shot number and no token of a chain: the plan goes on exactly
    # as its twin, which never made them. The last three prompts use a chain up to its last token.
    def later_calls(each):
        return [
            each.free_prompt(0, 4),
            each.prefill_prompt(0, 1, 4, 3),
            each.slot_prefix(0, 7, 6),
            each.prefill_prompt(0, 9, 4, 2),
            each.slot_prefix(0, 0, total + 1),
            each.free_prompt(0, total + 1),
            each.prefill_prompt(0, 2, total, total + 1),
        ]

    assert later_calls(plan) == later_calls(twin)
    assert _plan_facts(plan) == _plan_facts(twin)


# ---------------------------------------------------------------------------------------------------------
# Text: the invariants and the simulated runs with very repetitive text
# ---------------------------------------------------------------------------------------------------------

REPETITIVE_POOLS = ("shared-openings", "one-token", "one-conversation", "outside-the-range")
FAKE_REPEATED_TOKEN_OFFSET = 123


def _repetitive_conversations(kind: str, token_low: int, token_high: int) -> list[list[int]]:
    """FAKE conversations of very repetitive text, with ids of ``[token_low, token_high)`` or, in the last kind,
    outside it.

    ``shared-openings``: 64 conversations that all begin with the same 150 tokens, more than two pages of 64,
    and end with one to nine tokens of their own. The opening repeats five ids in a cycle.
    ``one-token``: 48 conversations of 20 to 67 tokens in which every token is the same id.
    ``one-conversation``: a single conversation of 3,000 tokens, so that every chain is the same text.
    ``outside-the-range``: 48 conversations of 40 to 87 tokens, made of three ids that lie outside the token
    range in a cycle: one below it and two at or above it, as the ids of special and added tokens do.
    """
    rng = random.Random(64)
    if kind == "shared-openings":
        opening = [token_low + 7 + position % 5 for position in range(150)]
        return [
            [*opening, *(rng.randrange(token_low, token_high) for _ in range(1 + index % 9))] for index in range(64)
        ]
    if kind == "one-token":
        return [[token_low + FAKE_REPEATED_TOKEN_OFFSET] * (20 + index) for index in range(48)]
    if kind == "outside-the-range":
        outside = (token_low - 1, token_high, token_high + 7)
        return [[outside[(index + position) % 3] for position in range(40 + index)] for index in range(48)]
    assert kind == "one-conversation"
    return [[rng.randrange(token_low, token_high) for _ in range(3000)]]


def _repetitive_plan(kind: str, seed: str = "0", token_low: int = LOW, token_high: int = HIGH):
    # The pool keeps more ids than the plan allocates from, as the launcher's pool does.
    conversations = _repetitive_conversations(kind, token_low, token_high)
    pool = content.TextPool(conversations, 0, token_high + FAKE_POOL_IDS_ABOVE)
    assert pool.dropped_tokens == 0
    return content.ContentPlan(seed, token_low, token_high, pool)


@pytest.mark.parametrize("page_size", [1, 64])
def test_text_without_allocated_tokens_would_not_keep_the_counts_exact(page_size):
    # Why the first tokens stay allocated with a text pool. The pool holds one id only, so prompts made of
    # text alone are the same tokens up to their length. Numbers worked out by hand from the cache facts.
    token = LOW + FAKE_REPEATED_TOKEN_OFFSET
    cache = PageCacheModel(page_size)
    cache.finish_prefill([token] * 128)  # a seed of text alone
    assert cache.cached([token] * 65) == 64  # a second prompt of text alone is not zero-cached
    # A shot of 64 cached and 67 new tokens of text alone finds both stored pages, or 128 single tokens.
    assert cache.cached([token] * 64 + [token] * 67) == 128

    plan = _repetitive_plan("one-token")
    sequence = plan.slot_prefix(0, 0, 129)
    assert sequence[0] != token and sequence[1:] == [token] * 128  # one allocated token, then the text
    cache.finish_prefill(sequence[:128])
    zero_cached = plan.free_prompt(0, 65)
    assert zero_cached[0] not in (token, sequence[0]) and zero_cached[1:] == [token] * 64
    assert cache.cached(zero_cached) == 0
    shot = plan.prefill_prompt(0, 0, 64, 67)
    assert shot[:64] == sequence[:64] and shot[64] != token and shot[65:] == [token] * 66
    assert cache.cached(shot) == 64  # the allocated first suffix token leaves the stored path
    cache.finish_prefill(shot)
    again = plan.prefill_prompt(0, 0, 64, 67)
    assert again[64] not in (token, shot[64]) and again[65:] == shot[65:]
    assert cache.cached(again) == 64  # and the path of the earlier shot


@pytest.mark.parametrize("kind", REPETITIVE_POOLS)
def test_i0_i1_i4_shapes_prefixes_and_range_with_repetitive_text(kind):
    plan = _repetitive_plan(kind)
    prompts = []
    for rank, slot, cached, new in ((0, 0, 16, 1), (0, 0, 16, 9), (1, 4, 64, 128), (0, 2, 1, 1)):
        prompt = plan.prefill_prompt(rank, slot, cached, new)
        assert len(prompt) == cached + new
        assert prompt[:cached] == plan.slot_prefix(rank, slot, cached)
        assert plan.prefill_seed_prompt(rank, slot, cached) == plan.slot_prefix(rank, slot, cached)
        prompts.append(prompt)
    for new in (1, 2, 77):
        prompts += [plan.prefill_prompt(0, 3, 0, new), plan.free_prompt(0, new)]
        assert [len(prompt) for prompt in prompts[-2:]] == [new, new]
    for context in (4, 5, 67, 1024):
        prompt = plan.decode_prompt(1, 2, context)
        assert len(prompt) == context - 3
        assert prompt == plan.slot_prefix(1, 2, context - 3)
        assert plan.decode_prompt(1, 2, context) == prompt  # the seed and the run use the same prompt
        prompts.append(prompt)
    longest = plan.slot_prefix(0, 0, 1500)
    for length in (1, 2, 15, 16, 17, 63, 64, 65, 1499, 1500):
        assert plan.slot_prefix(0, 0, length) == longest[:length]
    ascending = _repetitive_plan(kind)
    for length in (1, 64, 1023, 1500):
        assert ascending.slot_prefix(0, 0, length) == longest[:length]
    # A token is an allocated one, inside the token range, or an id of the text, wherever that id lies.
    text_ids = {token for ids in _repetitive_conversations(kind, LOW, HIGH) for token in ids}
    assert all(LOW <= token < HIGH or token in text_ids for prompt in [*prompts, longest] for token in prompt)
    assert all(LOW <= prompt[0] < HIGH for prompt in prompts)


def test_i4_a_text_keeps_the_ids_its_pool_keeps_and_only_those():
    # Every third id of the FAKE text is outside the token range of the plan, as the id of a special or an added
    # token is. The pool keeps the two of them that lie inside its own range and leaves out the third.
    rng = random.Random(4)
    fake = [
        [
            rng.choice((LOW - 1, HIGH, HIGH + 7)) if position % 3 == 0 else rng.randrange(LOW, HIGH)
            for position in range(30)
        ]
        for _ in range(40)
    ]
    pool_high = HIGH + 7
    kept = [[token for token in ids if token < pool_high] for ids in fake]
    dropped = sum(ids.count(HIGH + 7) for ids in fake)
    assert 100 < dropped < 300
    pool = content.TextPool(fake, 0, pool_high)
    assert (pool.conversations, pool.tokens, pool.dropped_tokens, pool.dropped_conversations) == (
        40,
        1200 - dropped,
        dropped,
        0,
    )
    plan = content.ContentPlan("0", LOW, HIGH, pool)
    # The chain is made of what the pool kept, the ids outside the token range among them and in their places;
    # the dropped ids leave no gap and no mark.
    chain = plan.slot_prefix(0, 6, pool.tokens + 1)[1:]
    assert chain == _reference_chain("0", kept, "slot", (0, 6))
    assert {LOW - 1, HIGH} <= set(chain) and HIGH + 7 not in chain
    prompts = [
        plan.slot_prefix(1, 4095, 500),
        plan.prefill_seed_prompt(1, 0, 64),
        plan.decode_prompt(0, 9, 300),
        plan.free_prompt(0, 200),
        *(plan.prefill_prompt(0, 2, 0, 50) for _ in range(20)),
        *(plan.prefill_prompt(0, 2, 16, 50) for _ in range(20)),
    ]
    for prompt in prompts:
        # No id the pool left out reaches a prompt, and the allocated first token is inside the token range.
        assert max(prompt) < pool_high
        assert LOW <= prompt[0] < HIGH
    # So is the allocated first new token of a shot with cached tokens.
    assert all(LOW <= prompt[16] < HIGH for prompt in prompts[-20:])
    assert any(token in (LOW - 1, HIGH) for prompt in prompts for token in prompt)


@pytest.mark.parametrize("outside", [0, LOW - 1, SMALL_HIGH, SMALL_HIGH + 99_999])
def test_shots_are_distinct_when_the_slots_next_token_lies_outside_the_token_range(outside):
    # FAKE text of one id that the plan does not allocate, so the slot's own next token is outside the token
    # range. The walk over the range then meets no element it has to leave out. It still gives every shot a
    # first new token of its own inside the range, vocab_size - 1 of them, and then refuses.
    pool = content.TextPool([[outside] * 40], 0, SMALL_HIGH + 100_000)
    plan = content.ContentPlan("0", LOW, SMALL_HIGH, pool)
    cached = 8
    assert plan.slot_prefix(0, 0, cached + 1)[1:] == [outside] * cached
    firsts = [plan.prefill_prompt(0, 0, cached, 1)[cached] for _ in range(4095)]
    assert len(set(firsts)) == 4095 and all(LOW <= token < SMALL_HIGH for token in firsts)
    # Which id of the range no shot gets is part of the content of a seed: the one the next token falls on
    # modulo the size of the range.
    start = _reference_stream("0", LOW, SMALL_HIGH, "off", (0, 0, cached), 1)[0][0] - LOW
    assert firsts == _reference_shot_firsts(LOW, SMALL_HIGH, start, outside)
    with pytest.raises(ValueError, match="^shots exhausted for rank 0 slot 0 cached 8: all 4095 "):
        plan.prefill_prompt(0, 0, cached, 1)


@pytest.mark.parametrize("before_the_offset", [True, False])
def test_no_shot_gets_a_next_token_that_lies_just_before_or_at_the_start_offset(before_the_offset):
    # FAKE text of one id of the token range: the element the walk over the range reaches last (it lies just
    # before the start offset) or first (it is the start offset). All 4,095 shots are issued, so the walk
    # passes it either way.
    cached = 8
    start = _reference_stream("0", LOW, SMALL_HIGH, "off", (0, 0, cached), 1)[0][0] - LOW
    assert 0 < start < 4095  # so the first case lies below the offset and neither case wraps
    next_token = LOW + start - 1 if before_the_offset else LOW + start
    plan = content.ContentPlan("0", LOW, SMALL_HIGH, content.TextPool([[next_token] * 40], LOW, SMALL_HIGH))
    assert plan.slot_prefix(0, 0, cached + 1)[cached] == next_token
    firsts = [plan.prefill_prompt(0, 0, cached, 1)[cached] for _ in range(4095)]
    assert next_token not in firsts
    assert firsts == _reference_shot_firsts(LOW, SMALL_HIGH, start, next_token)
    assert sorted([*firsts, next_token]) == list(range(LOW, SMALL_HIGH))


def test_an_outside_next_token_leaves_out_the_id_it_falls_on_in_a_range_that_does_not_divide_the_word_space():
    # 50,000 ids do not divide 2**32. Three FAKE next tokens, one below the token range and two above it, all
    # fall on the third element of the walk: shots 0 and 1 get the first two elements and shot 2 the fourth.
    token_low, token_high, cached = 60_000, 110_000, 8
    size = token_high - token_low
    start = _reference_stream("0", token_low, token_high, "off", (0, 0, cached), 1)[0][0] - token_low
    third = token_low + (start + 2) % size
    for next_token in (third - size, third + size, third + 3 * size):
        assert not token_low <= next_token < token_high
        pool = content.TextPool([[next_token] * 40], 0, token_high + 4 * size)
        plan = content.ContentPlan("0", token_low, token_high, pool)
        firsts = [plan.prefill_prompt(0, 0, cached, 1)[cached] for _ in range(6)]
        assert firsts == [token_low + (start + step) % size for step in (0, 1, 3, 4, 5, 6)]
        assert firsts == _reference_shot_firsts(token_low, token_high, start, next_token)[:6]


@pytest.mark.parametrize("kind", REPETITIVE_POOLS)
def test_i2_first_suffix_tokens_are_distinct_and_never_the_slots_next_token_with_repetitive_text(kind):
    plan = _repetitive_plan(kind)
    keys = [(0, 0, 1), (0, 0, 16), (0, 0, 17), (0, 1, 16), (1, 0, 16), (1, 7, 256)]
    firsts: dict[tuple[int, int, int], list[int]] = {key: [] for key in keys}
    for _ in range(200):  # interleaved: warm shots, measured shots and re-stages of several points
        for rank, slot, cached in keys:
            firsts[(rank, slot, cached)].append(plan.prefill_prompt(rank, slot, cached, 2)[cached])
    for (rank, slot, cached), tokens in firsts.items():
        own_next = plan.slot_prefix(rank, slot, cached + 1)[cached]
        assert len(set(tokens)) == 200
        assert own_next not in tokens
    if kind == "one-token":
        # Every slot goes on with the one id of the pool, so that id is the token every shot had to avoid.
        assert {plan.slot_prefix(rank, slot, cached + 1)[cached] for rank, slot, cached in keys} == {
            LOW + FAKE_REPEATED_TOKEN_OFFSET
        }


@pytest.mark.parametrize("kind", REPETITIVE_POOLS)
def test_i3_i6_first_tokens_are_distinct_with_repetitive_text(kind):
    # The smallest range: 3,796 slots and 300 zero-cached prompts take every one of its 4,096 ids once.
    plan = _repetitive_plan(kind, token_high=SMALL_HIGH)
    seeded = content.ContentPlan("0", LOW, SMALL_HIGH)
    zero_firsts = []
    slot_firsts = []
    for index in range(150):  # slots appear before, between and after the zero-cached prompts
        zero_firsts.append(plan.prefill_prompt(0, index % 4, 0, 3)[0])
        slot_firsts.append(plan.slot_prefix(0, index, 2)[0])
        zero_firsts.append(plan.free_prompt(0, 1)[0])
    slot_firsts += [plan.slot_prefix(0, slot, 20)[0] for slot in range(150, 3796)]
    assert len(set(zero_firsts)) == 300
    assert len(set(slot_firsts)) == 3796
    assert sorted([*zero_firsts, *slot_firsts]) == list(range(LOW, SMALL_HIGH))
    assert plan.remaining_zero_cached(0) == 0
    # The first tokens are those of the plan without a pool: the allocation does not look at the source.
    assert slot_firsts[:50] == [seeded.slot_prefix(0, slot, 1)[0] for slot in range(50)]
    assert zero_firsts[:50] == [seeded.free_prompt(0, 1)[0] for _ in range(50)]
    if kind == "one-token":
        # After its first token every slot is the same text; only the allocated token tells two slots apart.
        assert {tuple(plan.slot_prefix(0, slot, 20)[1:]) for slot in range(3796)} == {
            (LOW + FAKE_REPEATED_TOKEN_OFFSET,) * 19
        }


def test_i5_text_content_is_a_function_of_the_seed_the_pool_and_the_indices():
    fake = _fake_conversations(5, 60, LOW, HIGH)

    def new_plan(seed: str = "seed-a", conversations=fake):
        return content.ContentPlan(seed, LOW, HIGH, content.TextPool(conversations, LOW, HIGH))

    first = _recorded_calls(new_plan())
    assert _recorded_calls(new_plan()) == first  # two pools that hold the same conversations in the same order
    assert _recorded_calls(new_plan(conversations=[tuple(ids) for ids in fake])) == first

    # Content is a function of its key, not of the call order: the busy plan does unrelated work in between.
    quiet, busy = new_plan(), new_plan()
    busy.slot_prefix(0, 9, 100)
    busy.prefill_prompt(0, 2, 32, 4)
    busy.decode_prompt(1, 1, 50)
    shots_quiet = [quiet.prefill_prompt(0, 1, 16, 6) for _ in range(3)]
    shots_busy = []
    for index in range(3):
        shots_busy.append(busy.prefill_prompt(0, 1, 16, 6))
        busy.prefill_prompt(0, 1, 32, 6)  # another k of the same slot has its own counter
        busy.slot_prefix(0, 20 + index, 3)
    assert shots_busy == shots_quiet
    zero_quiet = [quiet.prefill_prompt(0, 0, 0, 4), quiet.free_prompt(0, 4)]
    busy.slot_prefix(0, 300, 1)
    zero_busy = [busy.prefill_prompt(0, 7, 0, 4), busy.free_prompt(0, 4)]  # the slot argument is not used
    assert zero_busy == zero_quiet
    assert busy.slot_prefix(0, 1, 64) == quiet.slot_prefix(0, 1, 64)

    # Another seed, another slot or rank, and the same conversations in another order all change the content.
    reference = new_plan().slot_prefix(0, 0, 64)
    for other in (
        new_plan("seed-b").slot_prefix(0, 0, 64),
        new_plan().slot_prefix(0, 1, 64),
        new_plan().slot_prefix(1, 0, 64),
        new_plan(conversations=fake[::-1]).slot_prefix(0, 0, 64),
        new_plan(conversations=fake[:-1]).slot_prefix(0, 0, 64),
    ):
        assert _differing_positions(reference[1:], other[1:]) >= 32


@pytest.mark.parametrize("generated_in_range", [False, True], ids=["generated-outside", "generated-inside"])
@pytest.mark.parametrize("one_by_one", [False, True], ids=["one-batch", "split-batches"])
@pytest.mark.parametrize("order", ["ascending", "descending", "shuffled"])
@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize("kind", REPETITIVE_POOLS)
def test_simulated_text_run_gets_the_planned_cached_tokens(kind, page_size, order, one_by_one, generated_in_range):
    # The sweeps of the seeded run above, with every prompt made of very repetitive text: at page size 1 one
    # token that two prompts share where none was planned changes a count, and at page size 64 a whole page.
    _simulate_and_check(_repetitive_plan(kind), page_size, order, one_by_one, generated_in_range)


@pytest.mark.parametrize("kind", REPETITIVE_POOLS)
def test_simulated_text_run_with_a_wide_batch_and_free_prompts(kind):
    _simulate_a_wide_batch(_repetitive_plan(kind, seed="wide", token_high=SMALL_HIGH))


# ---------------------------------------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ((0, 0, 4096), TypeError),  # the seed is a string
        ((b"0", 0, 4096), TypeError),
        ((None, 0, 4096), TypeError),
        (("", 0, 4096), ValueError),
        (("\ud800", 0, 4096), ValueError),  # not encodable
        (("0", True, 4096), TypeError),
        (("0", 0.0, 4096), TypeError),
        (("0", "0", 4096), TypeError),
        (("0", 0, 4096.0), TypeError),
        (("0", 0, None), TypeError),
        (("0", -1, 4096), ValueError),
        (("0", 0, 4095), ValueError),  # one id short of the minimum range
        (("0", 5000, 5000), ValueError),
        (("0", 5000, 100), ValueError),
        (("0", 0, 2**32 + 1), ValueError),  # ids are stored in 32 bits
    ],
)
def test_plan_rejects_bad_construction_arguments(arguments, error):
    with pytest.raises(error):
        content.ContentPlan(*arguments)


def test_plan_accepts_the_bounds_and_reports_its_range():
    for token_low, token_high in ((0, 4096), (2**32 - 4096, 2**32), (0, 2**32)):
        plan = content.ContentPlan("0", token_low, token_high)
        assert (plan.seed, plan.token_low, plan.token_high) == ("0", token_low, token_high)
        assert plan.vocab_size == token_high - token_low
        assert plan.remaining_zero_cached(0) == token_high - token_low
    assert content.MIN_VOCAB_SIZE == 4096


_METHOD_ARGUMENTS = {
    # method -> (valid arguments, lower bound of each argument)
    "slot_prefix": ((0, 0, 1), (0, 0, 1)),
    "prefill_seed_prompt": ((0, 0, 1), (0, 0, 1)),
    "prefill_prompt": ((0, 0, 0, 1), (0, 0, 0, 1)),
    "decode_prompt": ((0, 0, 4), (0, 0, 4)),
    "free_prompt": ((0, 1), (0, 1)),
    "remaining_zero_cached": ((0,), (0,)),
}


@pytest.mark.parametrize("method", sorted(_METHOD_ARGUMENTS))
def test_methods_accept_their_lower_bounds_and_reject_everything_else(method):
    valid, bounds = _METHOD_ARGUMENTS[method]
    plan = content.ContentPlan("0", LOW, HIGH)
    getattr(plan, method)(*valid)  # zero is a normal rank, slot and cached count
    before = plan.stats()
    for position, bound in enumerate(bounds):
        for bad in (True, False, 1.0, "1", None, [1]):
            arguments = [*valid]
            arguments[position] = bad
            with pytest.raises(TypeError):
                getattr(plan, method)(*arguments)
        for bad in (bound - 1, -5):
            arguments = [*valid]
            arguments[position] = bad
            with pytest.raises(ValueError, match=">="):
                getattr(plan, method)(*arguments)
    assert plan.stats() == before


def test_decode_staging_constants():
    # test_fpm_sglbench_producer.py pins these to the grid module's values.
    assert content.DECODE_PROMPT_LEAD == 3
    assert content.MIN_DECODE_CONTEXT == 4
    assert content.DECODE_OUTPUT_TOKENS == 7
    assert content.STAGES == ("seed", "warm", "meas", "run", "free")


# ---------------------------------------------------------------------------------------------------------
# Request ids
# ---------------------------------------------------------------------------------------------------------

_RUN_IDS = [
    "run",
    "run-1",
    "2026.11.14_cell-07",
    "-",
    "p1",
    "a-p1-seed0-r0-s0-e",  # looks like the tail of an id
    "sglb-x-p1-meas0-r0-s0-e",  # is itself an id
    "x-p2-run",
]


def test_request_id_format():
    assert content.request_id("run-1", 5, "meas", 0, 1, 10) == "sglb-run-1-p5-meas0-r1-s10-e"
    assert content.request_id("r", 0, "free", 12, 0, 0) == "sglb-r-p0-free12-r0-s0-e"


def test_request_id_round_trips_including_hyphenated_run_ids():
    for run_id, benchmark_id, stage, attempt, rank, slot in itertools.product(
        _RUN_IDS, (0, 1, 10, 123456), content.STAGES, (0, 1, 12), (0, 3), (0, 1, 10, 4095)
    ):
        rid = content.request_id(run_id, benchmark_id, stage, attempt, rank, slot)
        parts = content.parse_request_id(rid)
        assert parts == {
            "run_id": run_id,
            "benchmark_id": benchmark_id,
            "stage": stage,
            "attempt": attempt,
            "rank": rank,
            "slot": slot,
        }
        assert all(type(parts[key]) is int for key in ("benchmark_id", "attempt", "rank", "slot"))
        assert content.request_id(**parts) == rid


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        (("run", 1, "measure", 0, 0, 0), ValueError),  # bad stage
        (("run", 1, "", 0, 0, 0), ValueError),
        (("run", 1, "MEAS", 0, 0, 0), ValueError),
        (("run", 1, None, 0, 0, 0), TypeError),
        (("run", 1, 1, 0, 0, 0), TypeError),
        (("", 1, "meas", 0, 0, 0), ValueError),  # bad run id
        (("a b", 1, "meas", 0, 0, 0), ValueError),
        (("a/b", 1, "meas", 0, 0, 0), ValueError),
        (("a:b", 1, "meas", 0, 0, 0), ValueError),
        (("run\n", 1, "meas", 0, 0, 0), ValueError),
        (("r\u00e9sum\u00e9", 1, "meas", 0, 0, 0), ValueError),
        ((None, 1, "meas", 0, 0, 0), TypeError),
        ((7, 1, "meas", 0, 0, 0), TypeError),
        (("run", -1, "meas", 0, 0, 0), ValueError),  # bad numbers
        (("run", 1, "meas", -1, 0, 0), ValueError),
        (("run", 1, "meas", 0, -1, 0), ValueError),
        (("run", 1, "meas", 0, 0, -1), ValueError),
        (("run", True, "meas", 0, 0, 0), TypeError),
        (("run", 1, "meas", 0.0, 0, 0), TypeError),
        (("run", 1, "meas", 0, "0", 0), TypeError),
        (("run", 1, "meas", 0, 0, None), TypeError),
    ],
)
def test_request_id_rejects_bad_parts(arguments, error):
    with pytest.raises(error):
        content.request_id(*arguments)


@pytest.mark.parametrize(
    "foreign",
    [
        None,
        5,
        b"sglb-x-p1-meas0-r0-s0-e",
        "",
        "HEALTH_CHECK_1",
        "0f8fad5bd9cb469fa16570867728950e",
        "sglb-x-p1-meas0-r0-s0",  # no terminator
        "sglb-x-p1-meas0-r0-s0-e-e",
        "sglb-x-p1-meas0-r0-s0-e\n",  # request_id never emits a newline
        "sglb-x-p01-meas0-r0-s0-e",  # leading zero
        "sglb-x-p1-meas00-r0-s0-e",
        "sglb-x-p+1-meas0-r0-s0-e",  # sign
        "sglb-x-p1-meas0-r-0-s0-e",
        "sglb-x-p\u0661-meas0-r0-s0-e",  # Arabic-Indic digit one
        "sglb-x-p1-meas0-r0-s\uff11-e",  # full-width digit one
        "sglb-x y-p1-meas0-r0-s0-e",  # a character request_id rejects
        "sglb--p1-meas0-r0-s0-e",  # empty run id
        "sglb-x-p1-test0-r0-s0-e",  # unknown stage
        "sglb-x-p1-meas-r0-s0-e",  # no attempt
        "sglb-x-p1-meas0-s0-r0-e",  # fields swapped
        "SGLB-x-p1-meas0-r0-s0-e",
        "xsglb-x-p1-meas0-r0-s0-e",
    ],
)
def test_parse_request_id_returns_none_for_foreign_ids(foreign):
    assert content.parse_request_id(foreign) is None


def test_parse_request_id_does_not_raise_on_a_number_beyond_the_int_conversion_limit():
    rid = "sglb-x-p" + "9" * 5000 + "-meas0-r0-s0-e"
    parts = content.parse_request_id(rid)
    if 0 < sys.get_int_max_str_digits() < 5000:
        assert parts is None  # request_id cannot format such a number either
    else:  # the interpreter's limit was lifted
        assert content.request_id(**parts) == rid


def test_request_ids_of_a_full_run_are_unique_and_prefix_free():
    ids = [content.request_id("run-1", 0, "free", 0, rank, 0) for rank in range(RANKS)]
    for benchmark_id in range(1, 41):
        stages = ("seed", "warm", "meas") if benchmark_id <= 20 else ("seed", "run")
        for stage, attempt, rank, slot in itertools.product(stages, (0, 1), range(RANKS), range(130)):
            ids.append(content.request_id("run-1", benchmark_id, stage, attempt, rank, slot))
    assert len(set(ids)) == len(ids) == RANKS + 20 * 3 * 2 * RANKS * 130 + 20 * 2 * 2 * RANKS * 130
    assert not any(rid.startswith("HEALTH_CHECK") for rid in ids)

    def prefix_pairs(strings: list[str]) -> int:
        # In sorted order a string and the strings it is a prefix of are neighbours, so adjacent pairs suffice
        # to tell whether any pair exists.
        ordered = sorted(strings)
        return sum(1 for shorter, longer in itertools.pairwise(ordered) if longer.startswith(shorter))

    # SGLang aborts every request whose id starts with the aborted id, so no id may be a prefix of another.
    assert prefix_pairs(ids) == 0
    # Without the terminator slot 1 is a prefix of slots 10 to 19 and 100 to 129.
    assert prefix_pairs([rid.removesuffix("-e") for rid in ids]) > 0
    assert "sglb-run-1-p5-meas0-r0-s10-e".startswith("sglb-run-1-p5-meas0-r0-s1")


# ---------------------------------------------------------------------------------------------------------
# The module is a standard-library-only leaf
# ---------------------------------------------------------------------------------------------------------


def test_module_imports_in_the_pod_with_the_standard_library_only():
    # The pod has no AISim package: import by the flat name from the runtime directory, isolated from the
    # environment and from site-packages.
    code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import sglbench_content\n"
        "forbidden = set(sys.argv[2].split(','))\n"
        "loaded = sorted(name for name in sys.modules if name.split('.')[0] in forbidden)\n"
        "siblings = sorted(name for name in sys.modules if name.startswith('sglbench_'))\n"
        "assert not loaded, loaded\n"
        "assert siblings == ['sglbench_content'], siblings\n"
        "plan = sglbench_content.ContentPlan('0', 0, 4096)\n"
        "assert len(plan.prefill_prompt(0, 0, 16, 4)) == 20\n"
        "assert sglbench_content.parse_request_id(sglbench_content.request_id('r', 1, 'meas', 0, 0, 0))\n"
        # The text path, with FAKE records and FAKE ids: the recipe, a pool and a plan built on it.
        "records = [{'conversations': [{'value': 'x' * 64}]}]\n"
        "assert sglbench_content.conversation_texts(records) == ['x' * 64]\n"
        "pool = sglbench_content.TextPool([[5, 6, 7]] * 8, 0, 4096)\n"
        "text = sglbench_content.ContentPlan('0', 0, 4096, pool)\n"
        "assert text.prefill_prompt(0, 0, 16, 4)[1:16] == [5, 6, 7] * 5\n"
    )
    forbidden = "sglang,dynamo,aisim,aisim_core,collector,torch,numpy"
    done = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(RUNTIME), forbidden],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr


def test_module_source_uses_only_the_standard_library_and_no_environment():
    source = (RUNTIME / "sglbench_content.py").read_text()
    tree = ast.parse(source, feature_version=(3, 11))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import in a pod module"
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Attribute):
            assert node.attr not in {"environ", "getenv"}, "the module must not read the environment"
        elif isinstance(node, ast.Name):
            # A dynamic import is seen by no import statement; the pod would meet it only when the call runs.
            assert node.id != "__import__", "dynamic import in a pod module"
            # The dataset is read and tokenized by the caller: records and token ids come in as arguments.
            assert node.id != "open", "the module must not read a file"
    assert imported <= set(sys.stdlib_module_names)
    assert imported == {"__future__", "array", "collections", "hashlib", "re", "struct"}
    assert "importlib" not in imported, "dynamic import in a pod module"
    assert "os" not in imported
    assert not any(name.startswith("sglbench_") for name in imported)
    assert source.splitlines()[:6] == [
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        (
            "# The text recipe (conversation_texts) and the text chains follow _kvwarm_load_texts and "
            "_kvwarm_chain_token_ids of"
        ),
        "# ai-dynamo/dynamo (Apache-2.0, https://github.com/ai-dynamo/dynamo), components/src/dynamo/vllm/",
        (
            "# instrumented_scheduler.py at b20fe6fe94021b854597965c7305931175e8f482. The docstring also names its "
            "content-seed"
        ),
        "# variable; the caller passes its value as the seed. See THIRD_PARTY_NOTICES.md.",
    ]
    # The docstring says what follows Dynamo, in the words the other launcher modules use.
    assert (
        "No upstream function or block of source is copied. What follows SGLang and Dynamo is named in the header "
        "of this\nfile and recorded in THIRD_PARTY_NOTICES.md. "
    ) in source
