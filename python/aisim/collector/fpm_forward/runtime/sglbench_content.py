# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# The docstring names the content-seed variable of ai-dynamo/dynamo (Apache-2.0, https://github.com/ai-dynamo/dynamo),
# components/src/dynamo/vllm/instrumented_scheduler.py at b20fe6fe94021b854597965c7305931175e8f482; the caller passes
# its value as the seed. See THIRD_PARTY_NOTICES.md.

"""Token content and request ids for the SGLang self-benchmark producer (dense cells).

This module is for timing prefill and decode batches inside a running SGLang engine, where every timed
request must find an exact, planned number of its prompt tokens in SGLang's prefix cache. It supplies the
prompts (lists of token ids) that make that number exact by construction, and request ids that name the
run, point, stage, attempt, rank and slot of a request, for joining responses to scheduler batches. It
submits nothing and times nothing; that is the caller's part. It runs inside the SGLang pod: standard
library only, no import of another ``sglbench_*`` module, no environment variable (the seed is an argument).

Checked against
---------------
SGLang v0.5.22 (baa9befa381511cd4d56f83980951e5a45fffc6d) and
v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90) of ``sgl-project/sglang`` (Apache-2.0), files under
``python/sglang/srt``:

* ``mem_cache/radix_cache.py``, ``RadixKey.page_aligned`` and ``RadixKey.match_at``: a prefix is looked up
  and matched in whole pages from the start of the prompt, and at page size 1 one equal token is one
  cached token;
* ``managers/schedule_batch.py``, ``Req._compute_max_prefix_len``: at most ``input_len - 1`` tokens of a
  prompt are ever reported cached;
* ``mem_cache/common.py``, ``release_kv_cache`` with ``Req.owned_kv_len`` (``managers/schedule_batch.py``)
  and ``UnifiedRadixCache.checkpoint_into_tree`` (``mem_cache/unified_radix_cache.py``): a finished request
  stores, floored to whole pages, its prompt followed by the generated tokens that already have KV. That is
  none for a one-token prefill request and six for a seven-token decode run
  (``mem_cache/allocation.py``, ``alloc_for_decode_default`` commits one token per decode step);
* ``managers/io_struct.py``, ``GenerateReqInput._validate_rid_uniqueness`` and
  ``managers/tokenizer_manager.py``, ``TokenizerManager._init_req_state``: one request id per prompt, and
  no id may repeat inside a call or equal the id of a live request;
* ``managers/scheduler.py``, ``Scheduler.abort_request``: an abort removes every request whose id STARTS
  WITH the aborted id;
* ``constants.py``, ``HEALTH_CHECK_RID_PREFIX``: SGLang gives its own health-check requests ids that start
  with ``HEALTH_CHECK`` and takes a request whose id starts with it for one (``managers/utils.py``,
  ``is_health_check_generate_req``). ``observability/request_metrics_exporter.py``,
  ``FileRequestMetricsExporter.write_record`` writes no record for a request whose id contains it anywhere.
  It is not the only id prefix with a meaning: v0.5.22 builds internal ids that start with ``draft:``
  (``speculative/decoupled_spec_io.py``).

The cache rules were read in the Python tree classes. The default value of
``SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND`` (``environ.py``) selects a compiled Rust tree core, whose source
was not read; the comment beside the variable lists the cases in which SGLang falls back to the Python core.
Cached counts that running engines reported are kept in
``benchmarks/evidence/collector/sglbench-engine-probe-20261113`` (check_3 and check_5), called the engine
probe below. Its engines report SGLang 0.5.22 and 0.5.23 and ran Qwen3-0.6B and Qwen3-8B at page sizes 1 and
64 with TP 1; check_3 also ran Qwen3-8B at page size 1 with TP 2. The counts agree with the rules above in
the cases that were run:

* check_3: a prefix of whole pages that an earlier request had stored was reported cached in full, 128
  tokens at page size 1 and 512 at page size 64. A stored prefix seven tokens longer was reported as 135
  tokens at page size 1 and as 512, whole pages only, at page size 64.
* check_3: a 1021-token prompt sent a second time reported 1020 cached tokens at page size 1
  (``input_len - 1``) and 960 at page size 64.
* check_5: a 4093-token prompt sent again after its seven-token run had finished had 1 uncached token at
  page size 1 and 61 at page size 64, as the cap of ``input_len - 1`` and the floor to whole pages give.

Which requests a case sends is read from the driver script saved in that folder; the result files hold the
counts. They do not record which tree core was active. They also do not show what a seven-token run stores
after its prompt, because every prompt that was sent again had already been stored by its own seed request.
So that rule, and the guard under "Known limit" with it, rests on the source alone.

The seed is an argument: the caller must pass, as a string, the value of ``DYN_BENCH_CONTENT_SEED``. That is
the content-seed variable of Dynamo's vLLM self-benchmark (``ENV_BENCH_CONTENT_SEED``, which
``InstrumentedScheduler._bench_init`` reads with the default ``"0"``; the header of this file gives the file
and the revision).

No upstream function or block of source is copied. What follows SGLang and Dynamo is named in the header of this
file and recorded in THIRD_PARTY_NOTICES.md. For this module that is the name of Dynamo's content-seed variable,
given here for the caller; no code follows Dynamo. SGLang is named in comments and docstrings only.

Content layout
--------------
A slot is ``(rank, slot)``: the attention-DP rank and the position of a request inside a batch. A slot
has ONE token sequence ``seq`` for the whole run, so a shorter prompt of a slot is always a prefix of a
longer one and a seed computed for a deep point is reused by every shallower point.

Prompts, with ``low = token_low`` and ``V = token_high - token_low``:

* slot prefix (prefill seed, decode seed and decode run): ``seq[:length]``;
* shot with ``k > 0`` cached tokens and ``n`` new ones: ``seq[:k]`` then a suffix of ``n`` tokens;
* zero-cached prompt (a ``k == 0`` prefill request, the global warm-up, self-tests): its own tokens.

First tokens are ALLOCATED, not drawn, because at page size 1 a single repeated token changes a cached
count. Per rank, with a rotation ``a`` in ``[0, V)`` derived from the seed:

* slot first token: ``seq[0] = low + (a + slot) % V``;
* first token of the ``j``-th zero-cached prompt of the rank: ``low + (a - 1 - j) % V``;
* first suffix token of the ``j``-th shot ever issued for one ``(rank, slot, k)``: the ``j``-th element of
  the cyclic enumeration of the token range that starts at an offset derived from the seed and
  ``(rank, slot, k)`` and skips the slot's own next token ``seq[k]``.

Every other token comes from a stream, so all content depends on the seed and on nothing else.
``stream(kind, *parts)`` is SHA-256 in counter mode. Block ``c = 0, 1, ...`` is the digest of the UTF-8 text
``f"sglbench-content-v1:{len(seed.encode())}:{seed}:{kind}:{':'.join(str(p) for p in parts)}:{c}"``, read as
eight big-endian unsigned 32-bit words in order. A word ``w`` is used when ``w < (2**32 // V) * V`` and
gives the token ``low + w % V``; other words are skipped (rejection sampling, no modulo bias). Words left
over from a block stay with the stream, so its ``i``-th token does not depend on how it is consumed.

* ``("slot", rank, slot)``: its ``i``-th token is ``seq[i + 1]``;
* ``("shot", rank, slot, k, j)``: the suffix tokens after the first one of shot ``j`` of ``(rank, slot, k)``;
* ``("zero", rank, j)``: the tokens after the first one of the ``j``-th zero-cached prompt of the rank;
* ``("rot", rank)``: its first token minus ``low`` is the rotation ``a``;
* ``("off", rank, slot, k)``: its first token minus ``low`` is the start offset of the shot enumeration.

Invariants (the correctness argument; each one is tested)
---------------------------------------------------------
* I0 shape: ``prefill_prompt(r, s, k, n)`` has ``k + n`` tokens and, for ``k > 0``, starts with
  ``slot_prefix(r, s, k)``; ``decode_prompt(r, s, c) == slot_prefix(r, s, c - 3)``.
* I1 prefix property: ``slot_prefix(r, s, x)`` is a prefix of ``slot_prefix(r, s, y)`` for ``x <= y``.
* I2 for ``k > 0``: within one ``(rank, slot, k)`` all first suffix tokens are pairwise distinct and none
  equals ``seq[k]``. So a shot matches the slot's stored prefix up to ``k`` and then leaves every stored
  path: deeper seeds continue with ``seq[k]``, earlier shots with another first suffix token.
* I3 zero-cached prompts: on one rank their first tokens are pairwise distinct and distinct from the first
  token of every slot sequence of that rank. Nothing stored can match their first page.
* I4 every token is in ``[token_low, token_high)``.
* I5 determinism: content is a pure function of ``(seed, rank, slot, index)`` for slot sequences and of
  ``(seed, rank, slot, k, j)`` / ``(seed, rank, j)`` for shots and zero-cached prompts, where ``j`` only
  counts earlier prompts of the same kind. ``slot_prefix(r, s, x)`` is the same list whatever was requested
  before it. Two plans with the same seed and the same call sequence produce identical output; another
  seed changes the content.
* I6 slot first tokens: on one rank the first tokens of all slot sequences are pairwise distinct. With I3,
  every prompt of a rank starts with a token owned by exactly one slot or one zero-cached prompt, so the
  stored paths of one slot never lie on the path of another prompt.

At page size 1, I2, I3 and I6 are exactly what keeps the counts exact. Above it a page is a miss as soon as
one of its tokens differs, which random tokens alone make very likely; the allocated first tokens make it
certain. The module does not know the page size, so the invariants always hold.

Conditions on the caller
------------------------
* A prompt returned by ``prefill_prompt`` or ``free_prompt`` is submitted at most once (a second
  submission would find itself in the cache).
* Every prompt is submitted to the rank it was issued for and to no other rank.
* A seed has finished before the shots that rely on it are submitted.
* The cached length of a prefill shot is a multiple of the engine's page size (the grid module plans only
  such points); otherwise SGLang reports the prefix floored to whole pages.
* One plan belongs to one driver; it is not thread-safe.
* Output tokens. A request with ``n`` output tokens stores its prompt followed by its first ``n - 1``
  generated tokens (``Req.owned_kv_len``, see "Checked against"). Seeds (``prefill_seed_prompt``, and the
  seed submission of ``decode_prompt``) and prefill shots are submitted with exactly one output token, so
  that each stores its prompt and nothing after it; the invariants are argued for that case only. A prefill
  seed that stored one generated token could already hold the first suffix token of the next shot. A decode
  run is submitted with exactly ``DECODE_OUTPUT_TOKENS`` output tokens and its prompt comes from
  ``decode_prompt``: that call is what arms the guard described under "Known limit", and the guard covers
  six stored generated tokens, not more. ``slot_prefix`` returns the same tokens and arms nothing; it is for
  inspection and for one-token requests only. A prompt from ``free_prompt`` may be submitted with any number
  of output tokens: what it stores follows a first token that no other prompt of the rank starts with (I3).
* One plan per engine. Every prompt that reaches one engine during a run, self-test and warm-up included,
  comes from ONE plan. A second plan with the same seed issues the same zero-cached prompts again, and they
  find themselves in the cache; a plan with another seed can reuse first tokens of this plan, which at page
  size 1 is one cached token where none was planned.
* Token range. ``[token_low, token_high)`` lies inside the served model's ordinary vocabulary and holds no
  special or added token id. The module cannot check this: it sees two integers and no tokenizer. SGLang
  does not check input ids against the vocabulary either (``managers/tokenizer_manager.py``,
  ``TokenizerManager._validate_one_request`` checks how many input ids a request has, not their values).
  AISim's existing SGLang producer (``collector/fpm_forward/sglang_native_fpm.py``) takes its range from
  ``collector/glm53flash_attention_tokens.py``, ``ordinary_vocab_range``, which raises when a special or
  added id lies inside the tokenizer's base vocabulary.
* First-token capacity is a whole-run quantity. Per rank, the highest slot index used plus one, plus every
  zero-cached prompt issued (``prefill_prompt`` with ``cached == 0`` and ``free_prompt``, warm shots and
  re-stages included), must fit in ``vocab_size``; a first token is never given back. The caller should
  compare the planned total of the whole run with ``vocab_size`` before the sweep, because the plan
  otherwise raises ``ValueError`` in the middle of the run, at the first call that does not fit.
  ``MIN_VOCAB_SIZE`` is the smallest range the constructor accepts, not a range that holds every run.

Known limit, enforced
---------------------
A decode run stores its prompt followed by six model-generated tokens. A prefill shot issued LATER on the
same ``(rank, slot)`` with ``L <= k <= L + 5``, where ``L = context - 3`` is the prompt length of an earlier
``decode_prompt``, is not covered by I2: at page sizes up to 6 the generated tokens can reproduce
``seq[L:k]`` and then the shot's first suffix token, and the shot would report more than ``k`` cached. At
page sizes of 7 and above, and for zero-cached prompts and decode runs at any page size, generated tokens
cannot change a cached count. The plan records ``L`` for every ``decode_prompt`` and ``prefill_prompt``
raises ``ValueError`` for such a ``k``. The caller must run every prefill point before any decode point;
then the guard never fires.
"""

from __future__ import annotations

import hashlib
import re
import struct
from array import array

# Versioned domain string of the token streams. It is part of the content identity: together with the seed
# and the token range it determines every prompt, so a change here needs a new version suffix.
STREAM_DOMAIN = "sglbench-content-v1"

# The first-token allocation needs room for the slots of a batch plus the zero-cached prompts of a run.
MIN_VOCAB_SIZE = 4096

# Staging constants of a decode point, repeated here because this module imports no other sglbench module
# (tests/unit/collector/test_fpm_sglbench_producer.py pins them to the grid module's values). A decode
# request has ``context - 3`` prompt tokens and seven output tokens: six decode batches follow the extend
# batch, and the middle kept one (decode step 4) reads ``context`` past tokens. So the smallest context, a
# one-token prompt, is 4.
DECODE_PROMPT_LEAD = 3
MIN_DECODE_CONTEXT = 4
DECODE_OUTPUT_TOKENS = 7

# Request stages: seed (untimed prefix), warm and meas (the two prefill shots), run (decode), free
# (prompts outside any slot: global warm-up, self-tests).
STAGES = ("seed", "warm", "meas", "run", "free")

# Generated tokens of a decode run that reach the prefix cache: all but the last sampled one, which has no
# KV. SGLang: mem_cache/allocation.py alloc_for_decode_default advances Req.kv.kv_committed_len by one per
# decode step, and mem_cache/common.py release_kv_cache checkpoints up to Req.owned_kv_len().
_DECODE_STORED_OUTPUT_TOKENS = DECODE_OUTPUT_TOKENS - 1

_WORD_SPACE = 2**32
_BLOCK_WORDS = struct.Struct(">8I")  # one SHA-256 digest as eight big-endian unsigned 32-bit words

_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
_NUMBER = r"(0|[1-9][0-9]*)"  # ASCII digits, no sign, no leading zero: exactly what str(int) prints
_REQUEST_ID_PATTERN = re.compile(
    rf"sglb-([A-Za-z0-9_.-]+)-p{_NUMBER}-({'|'.join(STAGES)}){_NUMBER}-r{_NUMBER}-s{_NUMBER}-e"
)

# Slot sequences are kept as array("I"). Token ids go up to 2**32 - 1, so the item must hold 32 bits.
if array("I").itemsize < 4:
    raise RuntimeError('sglbench_content needs array("I") items of at least 32 bits on this platform')


def _check_int(name: str, value: object, minimum: int) -> None:
    """Raise unless ``value`` is a plain int (not a bool, not a subclass) and at least ``minimum``.

    Zero is a normal value for a rank, a slot, a cached count, a benchmark id and an attempt, so each bound
    is stated by its caller; nothing is coerced or repaired.
    """
    if type(value) is not int:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}: {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


class _TokenStream:
    """One SHA-256 counter-mode stream of token ids, uniform over the token range.

    The ``i``-th token of a stream is fixed by its key: words left over from a block stay with the stream,
    so it does not matter in how many calls the tokens are taken.
    """

    __slots__ = ("_accept_below", "_block", "_key_prefix", "_pending", "_token_low", "_vocab_size")

    def __init__(self, key_prefix: str, token_low: int, vocab_size: int) -> None:
        self._key_prefix = key_prefix
        self._token_low = token_low
        self._vocab_size = vocab_size
        # Rejection sampling: 2**32 is not a multiple of most vocabulary sizes, so "word % vocab_size" alone
        # would favour low ids. Words at or above the largest multiple of vocab_size are skipped.
        self._accept_below = _WORD_SPACE // vocab_size * vocab_size
        self._block = 0
        self._pending: list[int] = []

    def take(self, count: int) -> list[int]:
        """Return the next ``count`` tokens of the stream."""
        tokens = self._pending
        key_prefix = self._key_prefix
        token_low = self._token_low
        vocab_size = self._vocab_size
        accept_below = self._accept_below
        block = self._block
        while len(tokens) < count:
            digest = hashlib.sha256(f"{key_prefix}{block}".encode()).digest()
            block += 1
            tokens.extend(
                [token_low + word % vocab_size for word in _BLOCK_WORDS.unpack(digest) if word < accept_below]
            )
        self._block = block
        self._pending = tokens[count:]
        return tokens[:count]


class _SlotSequence:
    """The lazily grown token sequence of one ``(rank, slot)``."""

    __slots__ = ("stream", "tokens")

    def __init__(self, first_token: int, stream: _TokenStream) -> None:
        # array("I") holds 4 bytes per token. One slot can grow up to the engine's input limit
        # (max_req_input_len, 40954 tokens in every recorder header of the engine probe) and a run can use
        # thousands of slots, which is too much for lists of Python ints.
        self.tokens = array("I", [first_token])
        self.stream = stream


# Deviation note (collector rule 2, "measure what serving runs"): serving prompts are tokenized text; these
# prompts are seeded pseudo-random token ids drawn uniformly from [token_low, token_high).
# - What differs: the token values only. Request shape, batching, prefix-cache matching and the timed forward
#   pass are the serving path. It applies to every dense (non-MoE) cell on every system.
# - Direction of the error: none expected for dense models, whose forward time does not depend on token
#   values. For MoE models expert routing depends on content, so this module must not be used for them. It
#   cannot tell which model is served: the caller must reject a MoE model.
# - Sources: AISim's existing SGLang producer (collector/fpm_forward/sglang_native_fpm.py) also sends seeded
#   ids, drawn by collector/glm53flash_attention_tokens.py with the same method (a SHA-256 counter stream
#   with rejection sampling).
# - Retire by: replacing this id source with a text pool tokenized by the served tokenizer.
class ContentPlan:
    """Deterministic prompts for one benchmark run; see the module docstring for the layout and the proof.

    ``seed`` is a non-empty string. Token ids are drawn from ``[token_low, token_high)``, with
    ``0 <= token_low < token_high <= 2**32`` and at least ``MIN_VOCAB_SIZE`` ids in the range.

    Every method validates its arguments: a value that is not a plain ``int`` (a ``bool`` included) raises
    ``TypeError``, a value below its bound raises ``ValueError``. A call that raises changes no state.
    """

    def __init__(self, seed: str, token_low: int, token_high: int) -> None:
        if type(seed) is not str:
            raise TypeError(f"seed must be a str, got {type(seed).__name__}: {seed!r}")
        if not seed:
            raise ValueError("seed must be a non-empty str")
        try:
            seed_bytes = len(seed.encode())
        except UnicodeEncodeError as error:
            raise ValueError(f"seed must be encodable as UTF-8: {seed!r}") from error
        _check_int("token_low", token_low, 0)
        _check_int("token_high", token_high, 0)
        if token_high > _WORD_SPACE:
            raise ValueError(f"token_high must be <= 2**32 (ids are stored in 32 bits), got {token_high}")
        if token_high - token_low < MIN_VOCAB_SIZE:
            raise ValueError(
                f"the token range [{token_low}, {token_high}) must hold at least {MIN_VOCAB_SIZE} ids, "
                f"got {token_high - token_low}"
            )
        self._seed = seed
        self._token_low = token_low
        self._token_high = token_high
        self._vocab_size = token_high - token_low
        # The seed is length-prefixed so that no seed text can imitate the fields that follow it.
        self._key_prefix = f"{STREAM_DOMAIN}:{seed_bytes}:{seed}:"

        self._slots: dict[tuple[int, int], _SlotSequence] = {}
        self._rotations: dict[int, int] = {}  # rank -> a (a memo: the value is a pure function of the seed)
        self._highest_slot: dict[int, int] = {}  # rank -> highest slot index with a sequence
        self._zero_cached_issued: dict[int, int] = {}  # rank -> zero-cached prompts issued (prefill and free)
        self._shots_issued: dict[tuple[int, int, int], int] = {}  # (rank, slot, k) -> shots issued
        self._decode_prompt_lengths: dict[tuple[int, int], set[int]] = {}  # (rank, slot) -> decode prompt lengths
        self._counts = dict.fromkeys(
            ("seed_prompts", "shots", "zero_cached_prompts", "free_prompts", "decode_prompts"), 0
        )

    # ------------------------------------------------------------------ read-only facts

    @property
    def seed(self) -> str:
        return self._seed

    @property
    def token_low(self) -> int:
        return self._token_low

    @property
    def token_high(self) -> int:
        return self._token_high

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    # ------------------------------------------------------------------ streams and allocation

    def _stream(self, kind: str, *parts: int) -> _TokenStream:
        key = ":".join([kind, *(str(part) for part in parts)])
        return _TokenStream(f"{self._key_prefix}{key}:", self._token_low, self._vocab_size)

    def _rotation(self, rank: int) -> int:
        """The rank's rotation ``a`` in ``[0, vocab_size)``: where its first-token allocation starts."""
        rotation = self._rotations.get(rank)
        if rotation is None:
            rotation = self._stream("rot", rank).take(1)[0] - self._token_low
            self._rotations[rank] = rotation
        return rotation

    def _slot_tokens(self, rank: int, slot: int, length: int) -> array:
        """Return the slot's sequence, grown to at least ``length`` tokens.

        First tokens of a rank are laid out on a circle of ``vocab_size`` ids: slots take ``a, a + 1, ...``
        upwards and zero-cached prompts take ``a - 1, a - 2, ...`` downwards. The two runs must not meet
        (invariants I3 and I6), so the highest slot index in use plus one, plus the zero-cached prompts
        issued, may not exceed ``vocab_size``. The check runs here before the highest slot index is raised
        and in ``_zero_cached_prompt`` before a prompt is issued. It never wraps silently.
        """
        highest = self._highest_slot.get(rank, -1)
        if slot > highest:
            issued = self._zero_cached_issued.get(rank, 0)
            if slot + 1 + issued > self._vocab_size:
                raise ValueError(
                    f"first tokens exhausted on rank {rank}: slot index {slot} plus {issued} zero-cached "
                    f"prompt(s) already issued do not fit in a range of {self._vocab_size} token ids"
                )
            self._highest_slot[rank] = slot
        sequence = self._slots.get((rank, slot))
        if sequence is None:
            first_token = self._token_low + (self._rotation(rank) + slot) % self._vocab_size
            sequence = _SlotSequence(first_token, self._stream("slot", rank, slot))
            self._slots[(rank, slot)] = sequence
        missing = length - len(sequence.tokens)
        if missing > 0:
            sequence.tokens.extend(sequence.stream.take(missing))
        return sequence.tokens

    def _zero_cached_prompt(self, rank: int, length: int) -> list[int]:
        """Issue the next zero-cached prompt of the rank (one counter for prefill and free prompts)."""
        issued = self._zero_cached_issued.get(rank, 0)
        slots_in_use = self._highest_slot.get(rank, -1) + 1
        if slots_in_use + issued + 1 > self._vocab_size:
            raise ValueError(
                f"first tokens exhausted on rank {rank}: {slots_in_use} slot(s) in use and {issued} zero-cached "
                f"prompt(s) already issued fill the range of {self._vocab_size} token ids"
            )
        first_token = self._token_low + (self._rotation(rank) - 1 - issued) % self._vocab_size
        tail = self._stream("zero", rank, issued).take(length - 1)
        self._zero_cached_issued[rank] = issued + 1
        return [first_token, *tail]

    def _check_not_after_decode(self, rank: int, slot: int, cached: int) -> None:
        """The known limit (module docstring): refuse a shot whose prefix ends inside a decode run's tail.

        An earlier decode run with a prompt of ``L`` tokens left ``seq[:L]`` plus six generated tokens in the
        cache. Those tokens are not under this plan's control, so for ``L <= cached <= L + 5`` they could
        continue the slot's prefix up to ``cached`` and then equal the shot's first suffix token.
        """
        lengths = self._decode_prompt_lengths.get((rank, slot))
        if not lengths:
            return
        for gap in range(_DECODE_STORED_OUTPUT_TOKENS):
            if cached - gap in lengths:
                raise ValueError(
                    f"prefill shot with cached={cached} on rank {rank} slot {slot} follows a decode run with a "
                    f"prompt of {cached - gap} token(s) on the same slot: its {_DECODE_STORED_OUTPUT_TOKENS} "
                    "stored generated tokens could extend the cached prefix. Run prefill points before decode "
                    "points."
                )

    # ------------------------------------------------------------------ prompts

    def slot_prefix(self, rank: int, slot: int, length: int) -> list[int]:
        """The first ``length`` tokens of the slot's sequence (``rank >= 0``, ``slot >= 0``, ``length >= 1``)."""
        _check_int("rank", rank, 0)
        _check_int("slot", slot, 0)
        _check_int("length", length, 1)
        return self._slot_tokens(rank, slot, length)[:length].tolist()

    def prefill_seed_prompt(self, rank: int, slot: int, cached: int) -> list[int]:
        """The untimed seed of a prefill point: the slot's first ``cached`` tokens (``cached >= 1``)."""
        _check_int("rank", rank, 0)
        _check_int("slot", slot, 0)
        _check_int("cached", cached, 1)
        tokens = self._slot_tokens(rank, slot, cached)[:cached].tolist()
        self._counts["seed_prompts"] += 1
        return tokens

    def prefill_prompt(self, rank: int, slot: int, cached: int, new: int) -> list[int]:
        """One prefill shot: ``cached + new`` tokens of which exactly ``cached`` are expected in the cache.

        ``cached == 0``: a zero-cached prompt of ``new`` tokens. ``slot`` is validated but not used: the prompt
        does not belong to the slot's sequence and does not count the slot as in use.

        ``cached > 0``: ``slot_prefix(rank, slot, cached)`` followed by a suffix of ``new`` tokens.

        Each call is one shot and advances a counter, so the warm shot, the measured shot and a re-stage are
        separate calls that return different prompts.
        """
        _check_int("rank", rank, 0)
        _check_int("slot", slot, 0)
        _check_int("cached", cached, 0)
        _check_int("new", new, 1)
        if cached == 0:
            tokens = self._zero_cached_prompt(rank, new)
            self._counts["zero_cached_prompts"] += 1
            return tokens

        # Both checks below read state only; _slot_tokens then checks the first-token range before it
        # changes anything. So a refused call leaves the plan as it was.
        self._check_not_after_decode(rank, slot, cached)
        shot_key = (rank, slot, cached)
        shot = self._shots_issued.get(shot_key, 0)
        if shot >= self._vocab_size - 1:
            raise ValueError(
                f"shots exhausted for rank {rank} slot {slot} cached {cached}: all {self._vocab_size - 1} "
                "first suffix tokens that differ from the slot's own next token were issued"
            )

        # One token more than the prefix: seq[cached] is the token a deeper seed of this slot continues with.
        sequence = self._slot_tokens(rank, slot, cached + 1)
        # Invariant I2. Walk the token range as a circle from a start offset, leave out the slot's own next
        # token, and give shot number ``shot`` the element at that position. The vocab_size - 1 elements are
        # pairwise distinct and none is seq[cached], with no draw-and-retry and no memory of earlier shots.
        start = self._stream("off", rank, slot, cached).take(1)[0] - self._token_low
        own_next = sequence[cached] - self._token_low
        skipped_position = (own_next - start) % self._vocab_size
        position = shot if shot < skipped_position else shot + 1
        first_suffix_token = self._token_low + (start + position) % self._vocab_size
        tail = self._stream("shot", rank, slot, cached, shot).take(new - 1)

        self._shots_issued[shot_key] = shot + 1
        self._counts["shots"] += 1
        return [*sequence[:cached].tolist(), first_suffix_token, *tail]

    def decode_prompt(self, rank: int, slot: int, context: int) -> list[int]:
        """The prompt of a decode request whose middle kept decode step reads ``context`` past tokens.

        It is ``slot_prefix(rank, slot, context - 3)`` with ``context >= 4``. The same prompt serves the seed
        and the run (no counter): the run then finds every whole page of its first ``context - 4`` tokens.
        """
        _check_int("rank", rank, 0)
        _check_int("slot", slot, 0)
        _check_int("context", context, MIN_DECODE_CONTEXT)
        length = context - DECODE_PROMPT_LEAD
        tokens = self._slot_tokens(rank, slot, length)[:length].tolist()
        self._decode_prompt_lengths.setdefault((rank, slot), set()).add(length)
        self._counts["decode_prompts"] += 1
        return tokens

    def free_prompt(self, rank: int, length: int) -> list[int]:
        """A zero-cached prompt of ``length >= 1`` tokens that is not tied to a slot (warm-up, self-tests)."""
        _check_int("rank", rank, 0)
        _check_int("length", length, 1)
        tokens = self._zero_cached_prompt(rank, length)
        self._counts["free_prompts"] += 1
        return tokens

    # ------------------------------------------------------------------ bookkeeping

    def remaining_zero_cached(self, rank: int) -> int:
        """How many more zero-cached prompts the rank can issue with the slots in use now.

        Each new higher slot index lowers it by one per index. It is there so that a caller can compare it with
        a planned run (largest batch size, number of zero-cached prompts) before the run starts.
        """
        _check_int("rank", rank, 0)
        slots_in_use = self._highest_slot.get(rank, -1) + 1
        return self._vocab_size - slots_in_use - self._zero_cached_issued.get(rank, 0)

    def stats(self) -> dict[str, int]:
        """Counts issued per kind, summed over ranks (for the artifact's producer block).

        ``slots`` is the number of slot sequences, ``slot_tokens`` the tokens generated and kept for them;
        the other keys count calls: ``prefill_seed_prompt``, ``prefill_prompt`` with ``cached > 0`` (shots)
        and with ``cached == 0`` (zero-cached), ``free_prompt`` and ``decode_prompt``.
        """
        return {
            "slots": len(self._slots),
            "slot_tokens": sum(len(sequence.tokens) for sequence in self._slots.values()),
            "seed_prompts": self._counts["seed_prompts"],
            "shots": self._counts["shots"],
            "zero_cached_prompts": self._counts["zero_cached_prompts"],
            "free_prompts": self._counts["free_prompts"],
            "decode_prompts": self._counts["decode_prompts"],
        }


def request_id(run_id: str, benchmark_id: int, stage: str, attempt: int, rank: int, slot: int) -> str:
    """Build the request id of one prompt: ``sglb-{run_id}-p{benchmark_id}-{stage}{attempt}-r{rank}-s{slot}-e``.

    ``run_id`` is non-empty and uses ``[A-Za-z0-9_.-]`` only; ``stage`` is one of ``STAGES``; the numbers are
    non-negative ints (``benchmark_id`` 0 is used for prompts outside any point).

    SGLang needs one id per prompt and rejects a repeated id (managers/io_struct.py
    GenerateReqInput._validate_rid_uniqueness, managers/tokenizer_manager.py
    TokenizerManager._init_req_state); ids are unique per (run, point, stage, attempt, rank, slot).

    The fixed ``-e`` terminator makes the ids of one run prefix-free. SGLang's abort matches
    ``rid.startswith(...)`` (managers/scheduler.py Scheduler.abort_request), so without it an abort of slot 1
    would also remove slots 10 to 19 of the same batch. Ids start with ``sglb-`` and so never with
    ``HEALTH_CHECK``, the prefix of SGLang's own health-check ids (constants.py HEALTH_CHECK_RID_PREFIX).
    """
    if type(run_id) is not str:
        raise TypeError(f"run_id must be a str, got {type(run_id).__name__}: {run_id!r}")
    if _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(f"run_id must be non-empty and use only [A-Za-z0-9_.-], got {run_id!r}")
    if type(stage) is not str:
        raise TypeError(f"stage must be a str, got {type(stage).__name__}: {stage!r}")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    _check_int("benchmark_id", benchmark_id, 0)
    _check_int("attempt", attempt, 0)
    _check_int("rank", rank, 0)
    _check_int("slot", slot, 0)
    return f"sglb-{run_id}-p{benchmark_id}-{stage}{attempt}-r{rank}-s{slot}-e"


def parse_request_id(rid: object) -> dict | None:
    """Split an id built by ``request_id`` into its six parts, or return ``None`` for a foreign id.

    The result has the keys ``run_id``, ``benchmark_id``, ``stage``, ``attempt``, ``rank`` and ``slot`` (numbers
    as ints) and ``request_id(**parts) == rid`` always holds. Anything ``request_id`` could not have produced
    gives ``None``: a value that is not a ``str``, leading zeros, signs, non-ASCII digits, a trailing newline,
    an unknown stage, a missing terminator.

    The split is unique for run ids that contain hyphens: the five fields after the run id hold no hyphen,
    so they are always the last five hyphen-separated fields.
    """
    if type(rid) is not str:
        return None
    match = _REQUEST_ID_PATTERN.fullmatch(rid)
    if match is None:
        return None
    run_id, benchmark_id, stage, attempt, rank, slot = match.groups()
    try:
        return {
            "run_id": run_id,
            "benchmark_id": int(benchmark_id),
            "stage": stage,
            "attempt": int(attempt),
            "rank": int(rank),
            "slot": int(slot),
        }
    except ValueError:
        # A digit string beyond the interpreter's int conversion limit; request_id cannot format such a
        # number either, so the id is foreign.
        return None
