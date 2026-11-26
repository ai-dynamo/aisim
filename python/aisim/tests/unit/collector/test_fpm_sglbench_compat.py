# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# The fake package tree and the fake batch stream follow SGLang's interfaces at v0.5.22 and v0.5.23.
# See THIRD_PARTY_NOTICES.md.
# Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d
# Source: https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90

"""Tests of ``sglbench_compat.py``, the SGLang accessors of the self-benchmark launcher. No SGLang, no GPU.

Everything that stands in for SGLang is FAKE and is named so:

- ``fake_sglang`` puts a FAKE ``sglang`` package tree into ``sys.modules`` and takes it out again. It has the
  modules and names the accessors import, and its functions and its engine note that they were called. The names
  and their modules are those of SGLang v0.5.22 and v0.5.23 (``__init__.py``, ``srt/plugins/__init__.py``,
  ``srt/server_args.py``, ``srt/arg_groups/overrides.py``, ``srt/runtime_context.py``,
  ``srt/configs/model_config.py``, ``srt/entrypoints/engine.py``).
- ``_FakeStreamEngine`` answers ``async_generate`` with a scripted batch stream on a real asyncio loop. The
  chunks have the form SGLang gives them (``srt/managers/tokenizer_manager.py``,
  ``TokenizerManager._handle_batch_output``, ``_handle_abort_req`` and ``_stream_batch_responses``): a dict
  with ``text``, ``output_ids``, ``meta_info`` and ``index``. Their counts are typed in; no engine produced
  them.
- Every time value is FAKE. A time limit is reached by moving a fake clock that the test puts on the loop, so
  no test waits on a real clock. ``FAKE_E2E_LATENCY`` is a typed-in number in the place of a latency.

What the tests cannot show is that a real engine behaves as the fakes do. The module docstring of
``sglbench_compat.py`` says which parts were seen on running engines and which rest on the source alone.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import sys
import types
from pathlib import Path

import pytest

from .test_fpm_sglbench_preflight import (
    seam_rows_missing_for,
    seam_rows_not_in_source,
    sglang_names_reached_by_import,
    sglang_source_roots,
)

RUNTIME = Path(__file__).resolve().parents[3] / "collector/fpm_forward/runtime"
# The pod holds the runtime files flat in one directory and imports them by their plain names. The directory
# goes to the END of the path so that it shadows nothing.
if str(RUNTIME) not in sys.path:
    sys.path.append(str(RUNTIME))

import sglbench_compat as compat

pytestmark = pytest.mark.unit

MODULE_PATH = RUNTIME / "sglbench_compat.py"

# FAKE values. None of them was measured.
FAKE_E2E_LATENCY = 0.25
FAKE_CLOCK_START = 1000.0
CALL_TIMEOUT = 900.0

RESULT_KEYS = [
    "rid",
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "finish_type",
    "num_retractions",
    "dp_rank",
]
# In the place of a ``dp_rank`` when a FAKE ``meta_info`` is to have no such key.
NO_RANK_KEY = object()


# ---------------------------------------------------------------------------------------------------------------
# The FAKE sglang package tree
# ---------------------------------------------------------------------------------------------------------------


def _fake_stock_entry(*args, **kwargs):
    """FAKE of SGLang's stock scheduler entry (srt/managers/scheduler.py ``run_scheduler_process``)."""
    raise AssertionError("the fake stock entry is never run")


def _fake_recorder_entry(*args, **kwargs):
    """FAKE of the entry the launcher passes to ``build_engine``; a module-level function, as spawn needs."""
    raise AssertionError("the fake recorder entry is never run")


class _FakeServerArgs:
    """FAKE of ``ServerArgs``: the record the FAKE parser returns. No test reads a field of it."""

    def __init__(self, argv):
        self.argv = argv


class _FakeLazyImport:
    """FAKE of ``sglang.utils.LazyImport``: ``sglang.Engine`` is an instance of it, not a class."""

    def __init__(self, events):
        self.events = events

    def __call__(self, *args, **kwargs):
        self.events.append("sglang.Engine proxy called")
        raise AssertionError("the proxy must not be used to build the engine")


def _install_fake_sglang(monkeypatch):
    """Put the FAKE ``sglang`` package tree into ``sys.modules`` through ``monkeypatch``, which undoes it."""
    state = types.SimpleNamespace(
        events=[],
        parsed=[],
        engine_kwargs=[],
        hf_config=types.SimpleNamespace(architectures=["FakeDenseForCausalLM"], vocab_size=1000),
        hf_text_config=types.SimpleNamespace(vocab_size=1000),
        modules={},
    )

    def install(name, package=False, **attributes):
        module = types.ModuleType(name)
        if package:
            module.__path__ = []
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(state.modules[parent], child, module)
        state.modules[name] = module
        return module

    def load_plugins():
        state.events.append("load_plugins")

    def prepare_server_args(argv):
        state.events.append("prepare_server_args")
        state.parsed.append(argv)
        return _FakeServerArgs(argv)

    def resolving_view(server_args):
        state.events.append("resolving_view")

    def get_platform():
        state.events.append("get_platform")

    class FakeModelConfig:
        """FAKE of ``ModelConfig``: only the two configs the accessor reads."""

        def __init__(self):
            self.hf_config = state.hf_config
            self.hf_text_config = state.hf_text_config

        @staticmethod
        def from_server_args(*args, **kwargs):
            return FakeModelConfig()

    class FakeRealEngine:
        """FAKE of the real ``Engine`` class: the seam, and an ``__init__`` that reads it as SGLang does."""

        run_scheduler_process_func = staticmethod(_fake_stock_entry)

        def __init__(self, **kwargs):
            state.engine_kwargs.append(kwargs)
            self.server_args = kwargs["server_args"]
            # Engine.__init__ reads the seam from the instance and hands it to the process launcher.
            self.launched_with = self.run_scheduler_process_func

    install("sglang", package=True, __version__="0.5.23", Engine=_FakeLazyImport(state.events))
    install("sglang.srt", package=True)
    install("sglang.srt.plugins", package=True, load_plugins=load_plugins)
    install("sglang.srt.server_args", prepare_server_args=prepare_server_args)
    install("sglang.srt.arg_groups", package=True)
    install("sglang.srt.arg_groups.overrides", resolving_view=resolving_view)
    install("sglang.srt.runtime_context", get_platform=get_platform)
    install("sglang.srt.configs", package=True)
    install("sglang.srt.configs.model_config", ModelConfig=FakeModelConfig)
    install("sglang.srt.entrypoints", package=True)
    install("sglang.srt.entrypoints.engine", Engine=FakeRealEngine)
    state.real_engine_class = FakeRealEngine
    return state


@pytest.fixture
def fake_sglang(monkeypatch):
    """Install the FAKE ``sglang`` package tree; pytest removes it again after the test."""
    return _install_fake_sglang(monkeypatch)


FAKE_ARGV = ["--model-path", "/fake/model", "--incremental-streaming-output"]


# ---------------------------------------------------------------------------------------------------------------
# model_expert_keys
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "num_local_experts",
        "num_experts",
        "n_routed_experts",
        "moe_num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
    ],
)
@pytest.mark.parametrize("part", ["hf_config", "hf_text_config"])
def test_each_expert_key_counts_on_the_top_config_and_on_the_text_config(fake_sglang, key, part):
    setattr(getattr(fake_sglang, part), key, 8)
    assert compat.model_expert_keys(object()) == {key: 8}


@pytest.mark.parametrize("value", [0, None])
def test_a_falsy_expert_value_is_a_dense_model(fake_sglang, value):
    for key in compat.EXPERT_KEYS:
        setattr(fake_sglang.hf_config, key, value)
        setattr(fake_sglang.hf_text_config, key, value)
    assert compat.model_expert_keys(object()) == {}


@pytest.mark.parametrize("part", ["hf_config", "hf_text_config"])
def test_a_model_config_without_one_of_its_two_configs_is_an_error_not_a_dense_model(fake_sglang, part):
    fake_sglang.hf_text_config.num_experts = 8
    fake_sglang.hf_config.num_experts = 8
    setattr(fake_sglang, part, None)
    with pytest.raises(ValueError, match=part):
        compat.model_expert_keys(object())


# ---------------------------------------------------------------------------------------------------------------
# build_engine
# ---------------------------------------------------------------------------------------------------------------


def test_build_engine_subclasses_the_real_class_with_a_staticmethod_entry_and_server_args(fake_sglang):
    record = compat.parse_server_args(list(FAKE_ARGV))
    engine = compat.build_engine(record, _fake_recorder_entry)
    real = fake_sglang.real_engine_class

    assert isinstance(engine, real)
    assert type(engine) is not real and type(engine).__mro__[1] is real
    # The seam is a staticmethod in the subclass's own namespace.
    seam = type(engine).__dict__["run_scheduler_process_func"]
    assert isinstance(seam, staticmethod) and seam.__func__ is _fake_recorder_entry
    # So the engine reads the plain function back, not a method bound to itself, and SGLang launches with it.
    assert engine.run_scheduler_process_func is _fake_recorder_entry
    assert engine.launched_with is _fake_recorder_entry
    assert not inspect.ismethod(engine.launched_with)
    # The real class keeps its stock entry: nothing was assigned on it.
    assert real.run_scheduler_process_func is _fake_stock_entry
    # Built with server_args= and nothing else.
    assert fake_sglang.engine_kwargs == [{"server_args": record}]
    assert engine.server_args is record
    # sglang.Engine, the proxy, was not used.
    assert "sglang.Engine proxy called" not in fake_sglang.events


# ---------------------------------------------------------------------------------------------------------------
# generate_batch: the FAKE engine and its stream
# ---------------------------------------------------------------------------------------------------------------


def _length(count):
    """The finish reason of a request that produced all its tokens (schedule_batch.py ``FINISH_LENGTH``)."""
    return {"type": "length", "length": count}


def _meta(rid, *, finish=None, prompt=8, cached=0, completion=1, retractions=0, rank=None):
    """A ``meta_info`` as ``TokenizerManager._handle_batch_output`` builds it for a generation request.

    ``dp_rank`` is its last key and is ``None`` unless ``rank`` says otherwise: SGLang sets the key whenever
    the output record lists a rank for every request, and without a DP controller every one of them is
    ``None``. ``NO_RANK_KEY`` leaves the key out.
    """
    meta_info = {
        "id": rid,
        "finish_reason": finish,
        "prompt_tokens": prompt,
        "weight_version": "default",
        "num_retractions": retractions,
        "reasoning_tokens": 0,
        "completion_tokens": completion,
        "cached_tokens": cached,
    }
    if rank is not NO_RANK_KEY:
        meta_info["dp_rank"] = rank
    return meta_info


def _chunk(index, rid, **meta):
    """One chunk of a batch stream; ``_stream_batch_responses`` adds ``index`` to what a request yields."""
    return {"text": "", "output_ids": [1], "meta_info": _meta(rid, **meta), "index": index}


def _abort_echo(index, rid, message="Abort in waiting queue"):
    """The echo of an abort as ``TokenizerManager._handle_abort_req`` builds it: no prompt or cached count."""
    meta_info = {
        "id": rid,
        "finish_reason": {"type": "abort", "message": message},
        "weight_version": "default",
        "e2e_latency": FAKE_E2E_LATENCY,
        "completion_tokens": 0,
    }
    return {"text": "", "output_ids": [], "meta_info": meta_info, "index": index}


class _FakeClock:
    """FAKE clock for the loop: it moves only when a test moves it."""

    def __init__(self):
        self.now = FAKE_CLOCK_START

    def time(self):
        return self.now


class _FakeStreamEngine:
    """FAKE engine: ``async_generate`` returns a scripted batch stream that runs on ``self.loop``.

    A script step is a chunk (yielded), an exception instance (raised) or an async function (awaited). The
    stream notes how it ended: read to its end, closed before its end, or cancelled.
    """

    def __init__(self, loop, script):
        self.loop = loop
        self.script = list(script)
        self.calls = []
        self.loops_seen = []
        self.stream_events = []

    async def async_generate(self, **kwargs):
        self.calls.append(kwargs)
        return self._stream()

    async def _stream(self):
        self.loops_seen.append(asyncio.get_running_loop())
        try:
            for step in self.script:
                await asyncio.sleep(0)  # other tasks of the loop run between two chunks, as with SGLang
                if isinstance(step, BaseException):
                    raise step
                if callable(step):
                    await step()
                else:
                    yield step
        except GeneratorExit:
            self.stream_events.append("closed before its end")
            raise
        except asyncio.CancelledError:
            self.stream_events.append("cancelled")
            raise
        else:
            self.stream_events.append("read to its end")


@pytest.fixture
def loop():
    """A real asyncio loop with a FAKE clock, closed after the test."""
    event_loop = asyncio.new_event_loop()
    event_loop.fake_clock = _FakeClock()
    event_loop.time = event_loop.fake_clock.time
    yield event_loop
    try:
        pending = asyncio.all_tasks(event_loop)
        for task in pending:
            task.cancel()
        if pending:
            event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        event_loop.run_until_complete(event_loop.shutdown_asyncgens())
    except (SystemExit, KeyboardInterrupt):
        # A stop that a test sent through the loop can come out once more when the task of that call is
        # cancelled here (asyncio.wait_for of Python 3.11 runs the call in a task of its own).
        pass
    finally:
        event_loop.close()


def _prompts(count, length=8):
    return [[100 + request] * length for request in range(count)]


def _rids(count):
    return [f"fakerun-b0-free-a0-r0-s{request}" for request in range(count)]


def _generate(engine, count, max_new_tokens=2, timeout_seconds=CALL_TIMEOUT):
    return compat.generate_batch(engine, _prompts(count), _rids(count), max_new_tokens, timeout_seconds)


def _fails(engine, count, match):
    """The call raises EngineCallError with ``match``, was sent exactly once and returns the error."""
    with pytest.raises(compat.EngineCallError, match=match) as caught:
        _generate(engine, count)
    assert len(engine.calls) == 1, "a failed call is never sent again"
    return caught.value


# ---------------------------------------------------------------------------------------------------------------
# generate_batch: a complete stream
# ---------------------------------------------------------------------------------------------------------------


def test_interleaved_chunks_are_joined_by_index_and_the_last_chunk_of_each_request_is_kept(loop):
    r0, r1, r2 = _rids(3)
    script = [
        _chunk(0, r0, prompt=8, cached=0, completion=1),
        _chunk(2, r2, prompt=10, cached=4, completion=1),
        _chunk(1, r1, prompt=9, cached=2, completion=1),
        _chunk(1, r1, prompt=9, cached=2, completion=2, finish=_length(2)),
        _chunk(0, r0, prompt=8, cached=0, completion=2, finish=_length(2), retractions=1),
        _chunk(2, r2, prompt=10, cached=4, completion=2, finish={"type": "stop", "matched": 2}),
    ]
    engine = _FakeStreamEngine(loop, script)
    results = _generate(engine, 3)

    assert results == [
        {
            "rid": r0,
            "prompt_tokens": 8,
            "cached_tokens": 0,
            "completion_tokens": 2,
            "finish_type": "length",
            "num_retractions": 1,
            "dp_rank": None,
        },
        {
            "rid": r1,
            "prompt_tokens": 9,
            "cached_tokens": 2,
            "completion_tokens": 2,
            "finish_type": "length",
            "num_retractions": 0,
            "dp_rank": None,
        },
        {
            "rid": r2,
            "prompt_tokens": 10,
            "cached_tokens": 4,
            "completion_tokens": 2,
            "finish_type": "stop",
            "num_retractions": 0,
            "dp_rank": None,
        },
    ]
    assert [list(result) for result in results] == [RESULT_KEYS] * 3
    assert engine.stream_events == ["read to its end"]


def test_the_request_has_exactly_the_fields_of_a_benchmark_call(loop):
    prompts, rids = _prompts(2), _rids(2)
    script = [_chunk(index, rid, completion=7, finish=_length(7)) for index, rid in enumerate(rids)]
    engine = _FakeStreamEngine(loop, script)
    compat.generate_batch(engine, prompts, rids, 7, CALL_TIMEOUT)

    assert len(engine.calls) == 1
    call = engine.calls[0]
    assert set(call) == {"input_ids", "sampling_params", "rid", "stream"}
    assert call["input_ids"] == _prompts(2) and call["rid"] == _rids(2)
    assert call["sampling_params"] == {"max_new_tokens": 7, "temperature": 0.0, "ignore_eos": True}
    assert type(call["sampling_params"]["temperature"]) is float
    assert call["sampling_params"]["ignore_eos"] is True
    # Engine.async_generate tests ``stream is True``; any other truthy value returns the first chunk only.
    assert call["stream"] is True
    # The caller's lists are not changed.
    assert prompts == _prompts(2) and rids == _rids(2)


def test_the_rank_of_each_request_is_passed_on_as_sglang_reports_it(loop):
    rids = _rids(4)
    # FAKE of an engine with two attention-DP ranks: the requests of a call ran on the ranks in turn. The
    # unfinished chunks of a request carry the rank too; only the final one is read.
    script = [_chunk(index, rid, completion=1, rank=index % 2) for index, rid in enumerate(rids)]
    script += [_chunk(index, rid, completion=2, finish=_length(2), rank=index % 2) for index, rid in enumerate(rids)]
    results = _generate(_FakeStreamEngine(loop, script), 4)

    assert [result["dp_rank"] for result in results] == [0, 1, 0, 1]
    assert all(type(result["dp_rank"]) is int for result in results)
    assert [list(result) for result in results] == [RESULT_KEYS] * 4
    # Nothing is concluded from the rank here: a call whose requests all ran on one rank is returned as it is.
    one_rank = [_chunk(index, rid, completion=2, finish=_length(2), rank=1) for index, rid in enumerate(rids)]
    assert [result["dp_rank"] for result in _generate(_FakeStreamEngine(loop, one_rank), 4)] == [1, 1, 1, 1]


@pytest.mark.parametrize("rank", [None, NO_RANK_KEY], ids=["none", "no key"])
def test_a_request_without_a_rank_has_none_as_its_rank(loop, rank):
    # ``None`` under the key is what SGLang reports without a DP controller; no key at all is an output record
    # without ranks.
    rids = _rids(2)
    script = [_chunk(index, rid, completion=2, finish=_length(2), rank=rank) for index, rid in enumerate(rids)]
    results = _generate(_FakeStreamEngine(loop, script), 2)

    assert [result["dp_rank"] for result in results] == [None, None]
    assert [list(result) for result in results] == [RESULT_KEYS] * 2


@pytest.mark.parametrize("rank", ["1", True])
def test_a_rank_that_is_neither_a_plain_int_nor_none_fails_the_call(loop, rank):
    r0, r1 = _rids(2)
    script = [_chunk(0, r0, completion=2, finish=_length(2), rank=0), _chunk(1, r1, completion=2, finish=_length(2))]
    script[1]["meta_info"]["dp_rank"] = rank
    _fails(_FakeStreamEngine(loop, script), 2, "not a rank")


def test_every_call_runs_on_the_engines_own_loop_and_leaves_it_open(loop):
    rids = _rids(2)
    script = [_chunk(index, rid, completion=1, finish=_length(1)) for index, rid in enumerate(rids)]
    engine = _FakeStreamEngine(loop, script)
    for _ in range(3):
        assert len(_generate(engine, 2, max_new_tokens=1)) == 2
    assert len(engine.loops_seen) == 3 and all(seen is loop for seen in engine.loops_seen)
    assert not loop.is_closed() and not loop.is_running()


# ---------------------------------------------------------------------------------------------------------------
# generate_batch: the time limit
# ---------------------------------------------------------------------------------------------------------------


def _move_clock(event_loop, seconds):
    """A script step that moves the FAKE clock and lets the loop look at its timers. No real waiting."""

    async def step():
        event_loop.fake_clock.now += seconds
        for _ in range(3):
            await asyncio.sleep(0)

    return step


def _wait_to_be_cancelled(event_loop, seconds):
    """A script step that moves the FAKE clock past the limit and then waits for the cancellation.

    It gives up after a fixed number of loop turns, so a call without a time limit fails the test and does not
    hang it.
    """

    async def step():
        event_loop.fake_clock.now += seconds
        for _ in range(200):
            await asyncio.sleep(0)
        raise RuntimeError("FAKE stream: the call was not cancelled at its time limit")

    return step


def test_the_time_limit_cancels_a_call_that_is_still_streaming(loop):
    r0, r1 = _rids(2)
    script = [
        _chunk(0, r0),
        _chunk(1, r1, completion=2, finish=_length(2)),
        _wait_to_be_cancelled(loop, CALL_TIMEOUT + 0.5),
        _chunk(0, r0, completion=2, finish=_length(2)),
    ]
    engine = _FakeStreamEngine(loop, script)
    error = _fails(engine, 2, r"timed out after 900\.0 s")
    assert isinstance(error.__cause__, TimeoutError)
    # The cancellation reached the stream. That is where SGLang aborts the requests it had dispatched.
    assert engine.stream_events == ["cancelled"]
    assert not loop.is_closed()


def test_a_call_that_ends_just_inside_its_limit_is_complete(loop):
    r0, r1 = _rids(2)
    script = [
        _chunk(0, r0),
        _move_clock(loop, CALL_TIMEOUT - 1.0),
        _chunk(1, r1, completion=2, finish=_length(2)),
        _chunk(0, r0, completion=2, finish=_length(2)),
    ]
    engine = _FakeStreamEngine(loop, script)
    assert [result["completion_tokens"] for result in _generate(engine, 2)] == [2, 2]
    assert engine.stream_events == ["read to its end"]


# ---------------------------------------------------------------------------------------------------------------
# generate_batch: aborts
# ---------------------------------------------------------------------------------------------------------------


def test_the_echo_of_an_abort_fails_the_call_and_closes_the_stream(loop):
    r0, r1 = _rids(2)
    script = [
        _chunk(0, r0),
        _abort_echo(1, r1, message="The request queue is full."),
        _chunk(0, r0, completion=2, finish=_length(2)),
    ]
    engine = _FakeStreamEngine(loop, script)
    error = _fails(engine, 2, "was aborted")
    assert r1 in str(error) and "The request queue is full." in str(error)
    # The stream was closed inside the call, with a chunk of the other request still to come.
    assert engine.stream_events == ["closed before its end"]


# ---------------------------------------------------------------------------------------------------------------
# generate_batch: chunks that cannot be placed, a stream that ends early, a count that is no count
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("index", "match"),
    [(2, "has index"), (True, "has index"), (None, "has index"), ("absent", "has no index")],
)
def test_a_chunk_without_an_index_or_with_one_outside_the_call_fails_it(loop, index, match):
    r0, r1 = _rids(2)
    chunk = _chunk(1, r1)
    chunk["index"] = index
    if index == "absent":
        del chunk["index"]
    engine = _FakeStreamEngine(loop, [_chunk(0, r0), chunk, _chunk(0, r0, completion=2, finish=_length(2))])
    _fails(engine, 2, match)
    assert engine.stream_events == ["closed before its end"]


def test_a_chunk_whose_request_id_is_not_the_one_at_its_index_fails_the_call(loop):
    r0, r1 = _rids(2)
    script = [_chunk(0, r1, completion=2, finish=_length(2)), _chunk(1, r0, completion=2, finish=_length(2))]
    _fails(_FakeStreamEngine(loop, script), 2, "carries the id")


def test_a_stream_that_ends_without_a_final_chunk_fails_the_call(loop):
    r0, r1 = _rids(2)
    script = [_chunk(0, r0), _chunk(1, r1), _chunk(0, r0, completion=2, finish=_length(2)), _chunk(1, r1)]
    engine = _FakeStreamEngine(loop, script)
    error = _fails(engine, 2, "without the final chunk of 1 of 2 requests")
    assert r1 in str(error) and r0 not in str(error)
    assert engine.stream_events == ["read to its end"]


@pytest.mark.parametrize(("key", "value"), [("cached_tokens", True), ("completion_tokens", "8")])
def test_a_count_that_is_not_a_plain_int_fails_the_call(loop, key, value):
    (rid,) = _rids(1)
    final = _chunk(0, rid, completion=2, finish=_length(2))
    final["meta_info"][key] = value
    _fails(_FakeStreamEngine(loop, [final]), 1, f"{key} is")


# ---------------------------------------------------------------------------------------------------------------
# generate_batch: exceptions of the call, and what must pass through
# ---------------------------------------------------------------------------------------------------------------


def test_an_exception_of_the_stream_becomes_an_engine_call_error(loop):
    (rid,) = _rids(1)
    failure = ValueError("FAKE: the input is longer than the model's context length")
    engine = _FakeStreamEngine(loop, [_chunk(0, rid), failure])
    error = _fails(engine, 1, "the engine call raised ValueError: FAKE: the input is longer")
    assert error.__cause__ is failure


def test_an_exception_of_async_generate_itself_becomes_an_engine_call_error(loop):
    class _FakeRefusingEngine(_FakeStreamEngine):
        async def async_generate(self, **kwargs):
            self.calls.append(kwargs)
            raise ValueError("FAKE: routed_dp_rank out of range")

    _fails(_FakeRefusingEngine(loop, []), 1, "the engine call raised ValueError: FAKE: routed_dp_rank")


def test_a_cancellation_that_is_not_the_time_limit_becomes_an_engine_call_error(loop):
    (rid,) = _rids(1)
    engine = _FakeStreamEngine(loop, [_chunk(0, rid), asyncio.CancelledError()])
    error = _fails(engine, 1, "was cancelled")
    assert "timed out" not in str(error)


@pytest.mark.parametrize("stop", [SystemExit(0), KeyboardInterrupt()])
def test_system_exit_and_keyboard_interrupt_pass_through(loop, stop):
    (rid,) = _rids(1)
    engine = _FakeStreamEngine(loop, [_chunk(0, rid), stop])
    with pytest.raises(type(stop)) as caught:
        _generate(engine, 1)
    assert caught.value is stop
    assert not isinstance(caught.value, compat.EngineCallError)
    assert len(engine.calls) == 1


# ---------------------------------------------------------------------------------------------------------------
# generate_batch: arguments that SGLang would read as another kind of call
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompts", "rids", "match"),
    [
        ([1, 2, 3], ["a"], "prompt 0"),  # a flat list is ONE request for SGLang, with no index in its stream
        ([[1, 2], [3]], ["a"], "one per prompt"),
        ([[1, 2], [3]], ["a", "a"], "repeat"),
    ],
)
def test_a_call_that_is_not_a_batch_of_prompts_with_one_id_each_is_refused_before_it_is_sent(
    loop, prompts, rids, match
):
    engine = _FakeStreamEngine(loop, [])
    with pytest.raises(compat.EngineCallError, match=match):
        compat.generate_batch(engine, prompts, rids, 2, CALL_TIMEOUT)
    assert engine.calls == [] and engine.loops_seen == []


@pytest.mark.parametrize("timeout_seconds", [None, float("inf"), 0])
def test_a_call_without_a_finite_time_limit_is_refused(loop, timeout_seconds):
    engine = _FakeStreamEngine(loop, [])
    with pytest.raises(compat.EngineCallError, match="timeout_seconds"):
        _generate(engine, 1, timeout_seconds=timeout_seconds)
    assert engine.calls == []


# ---------------------------------------------------------------------------------------------------------------
# vocabulary, model_vocabulary_size, encode_texts
# ---------------------------------------------------------------------------------------------------------------


class _FakeTokenizer:
    """FAKE tokenizer with the four names the accessor reads; SGLang attaches the last one itself."""

    def __init__(self, vocab_size=1000, special=(0, 1, 2), added=None, stop_ids=None):
        self.vocab_size = vocab_size
        self.all_special_ids = None if special is None else list(special)
        self.added = {"<fake_a>": 1000, "<fake_b>": 1001} if added is None else added
        self.additional_stop_token_ids = stop_ids

    def get_added_vocab(self):
        return self.added


def _fake_engine_with(tokenizer, model_vocab_size=1000):
    manager = types.SimpleNamespace(
        tokenizer=tokenizer, model_config=types.SimpleNamespace(vocab_size=model_vocab_size)
    )
    return types.SimpleNamespace(tokenizer_manager=manager)


def test_vocabulary_excludes_special_added_and_stop_ids():
    tokenizer = _FakeTokenizer(
        special=(2, 0, 1, 2), added={"<fake_a>": 1001, "<fake_b>": 1000, "<fake_c>": 5}, stop_ids={7, 1001}
    )
    vocabulary = compat.vocabulary(_fake_engine_with(tokenizer))
    assert vocabulary == {"size": 1000, "excluded": [0, 1, 2, 5, 7, 1000, 1001]}
    assert list(vocabulary) == ["size", "excluded"] and isinstance(vocabulary["excluded"], list)

    def takes_the_two_keys(size, excluded):
        return size, excluded

    assert takes_the_two_keys(**vocabulary) == (1000, [0, 1, 2, 5, 7, 1000, 1001])


@pytest.mark.parametrize(("tokenizer_size", "model_size", "size"), [(1000, 900, 900), (900, 1000, 900), (7, 7, 7)])
def test_vocabulary_size_is_the_smaller_of_the_tokenizers_and_the_models(tokenizer_size, model_size, size):
    engine = _fake_engine_with(_FakeTokenizer(vocab_size=tokenizer_size), model_vocab_size=model_size)
    assert compat.vocabulary(engine)["size"] == size


@pytest.mark.parametrize("attribute", ["all_special_ids", "additional_stop_token_ids"])
def test_vocabulary_refuses_a_tokenizer_without_one_of_its_attributes(attribute):
    tokenizer = _FakeTokenizer()
    delattr(tokenizer, attribute)
    with pytest.raises(AttributeError, match=attribute) as caught:
        compat.vocabulary(_fake_engine_with(tokenizer))
    assert type(tokenizer).__name__ in str(caught.value)


@pytest.mark.parametrize(
    ("change", "model_size", "error", "match"),
    [
        ({"vocab_size": 0}, 1000, ValueError, "tokenizer.vocab_size"),
        ({"special": (0, "1")}, 1000, TypeError, "all_special_ids"),
        ({"special": None}, 1000, TypeError, "all_special_ids"),
        ({"added": ["<fake_a>"]}, 1000, TypeError, "get_added_vocab"),
        ({"stop_ids": 7}, 1000, TypeError, "additional_stop_token_ids"),
        ({}, None, TypeError, "model_config.vocab_size"),
        ({}, -1, ValueError, "model_config.vocab_size"),
    ],
)
def test_vocabulary_refuses_values_that_are_not_what_the_names_promise(change, model_size, error, match):
    with pytest.raises(error, match=match):
        compat.vocabulary(_fake_engine_with(_FakeTokenizer(**change), model_vocab_size=model_size))


@pytest.mark.parametrize(("tokenizer_size", "model_size"), [(1000, 900)])
def test_model_vocabulary_size_is_the_models_bound_whatever_the_tokenizers_is(tokenizer_size, model_size):
    engine = _fake_engine_with(_FakeTokenizer(vocab_size=tokenizer_size), model_vocab_size=model_size)
    assert compat.model_vocabulary_size(engine) == model_size
    # It reads the model config only: an engine without a tokenizer still has the bound.
    assert compat.model_vocabulary_size(_fake_engine_with(None, model_vocab_size=model_size)) == model_size


@pytest.mark.parametrize(("model_size", "error"), [(None, TypeError), (0, ValueError)])
def test_model_vocabulary_size_refuses_a_value_that_is_not_a_positive_int(model_size, error):
    with pytest.raises(error, match=r"^tokenizer_manager\.model_config\.vocab_size is "):
        compat.model_vocabulary_size(_fake_engine_with(_FakeTokenizer(), model_vocab_size=model_size))


class _FakeBatchEncoding:
    """FAKE of the mapping a Hugging Face tokenizer returns: it answers ``["input_ids"]`` and is no dict."""

    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        return {"input_ids": self.rows}[key]


class _FakeBatchTokenizer:
    """FAKE tokenizer with the batch call ``encode_texts`` makes: ``tokenizer(texts, add_special_tokens=False)``.

    Each text becomes one id per character, the character's code point plus ``FAKE_ID_BASE``. The real call
    belongs to the transformers library and returns a mapping with ``input_ids`` that is no dict, and so does
    this one; ``answer`` puts another object in its place.
    """

    FAKE_ID_BASE = 100

    def __init__(self, answer=None):
        self.calls = []
        self.answer = answer

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        if self.answer is not None:
            return self.answer(texts)
        return _FakeBatchEncoding([[self.FAKE_ID_BASE + ord(character) for character in text] for text in texts])


def test_encode_texts_makes_one_batch_call_without_special_tokens_and_returns_one_id_list_per_text():
    tokenizer = _FakeBatchTokenizer()
    texts = ["ab", "", "cab"]
    ids = compat.encode_texts(_fake_engine_with(tokenizer), texts)
    assert ids == [[197, 198], [], [199, 197, 198]]
    assert all(type(token) is int for row in ids for token in row)
    # One call for all texts, with the one keyword and the caller's own list, which is not changed.
    assert len(tokenizer.calls) == 1
    assert tokenizer.calls[0][0] is texts and tokenizer.calls[0][1] == {"add_special_tokens": False}
    assert texts == ["ab", "", "cab"]


@pytest.mark.parametrize(
    ("texts", "error", "match"),
    [("one text", TypeError, "texts must be a list"), ([], ValueError, "must not be empty")],
)
def test_encode_texts_refuses_what_is_not_a_non_empty_list_of_strings_before_the_tokenizer_is_called(
    texts, error, match
):
    tokenizer = _FakeBatchTokenizer()
    with pytest.raises(error, match=match):
        compat.encode_texts(_fake_engine_with(tokenizer), texts)
    assert tokenizer.calls == []


@pytest.mark.parametrize(
    ("answer", "error", "match"),
    [
        (lambda texts: {"ids": [[1] for _text in texts]}, TypeError, "returned dict without input_ids"),
        (lambda texts: {"input_ids": [[1]]}, ValueError, "returned 1 id lists for 2 texts"),
        (lambda texts: {"input_ids": [[1], (2, 3)]}, TypeError, "returned tuple for text 1"),
        (lambda texts: {"input_ids": [[1], [3, True]]}, TypeError, "returned True among the ids of text 1"),
    ],
)
def test_encode_texts_refuses_a_result_that_is_not_one_list_of_plain_ints_per_text(answer, error, match):
    tokenizer = _FakeBatchTokenizer(answer=answer)
    with pytest.raises(error, match=match) as caught:
        compat.encode_texts(_fake_engine_with(tokenizer), ["a", "b"])
    assert "_FakeBatchTokenizer" in str(caught.value), "the message names the kind of tokenizer"
    assert len(tokenizer.calls) == 1


# ---------------------------------------------------------------------------------------------------------------
# The seam rows
# ---------------------------------------------------------------------------------------------------------------


def _module_tree():
    return ast.parse(MODULE_PATH.read_text(encoding="utf-8"))


# The seam rows of the module itself, the rows the preflight looks up in the image. No copy is typed here.
SEAM_NAMES = compat.SGLANG_SEAM_NAMES


def test_seam_rows_name_everything_the_module_imports_from_sglang():
    rows = compat.SGLANG_SEAM_NAMES
    reached = sglang_names_reached_by_import(_module_tree())
    # An import or a read that gets no row is noticed here, so the list cannot fall behind the code.
    assert seam_rows_missing_for(reached, rows) == []
    # The last two rows name what the module never imports: the field behind the ``dp_rank`` of a chunk, in
    # both output records of SGLang. Every other row is of a module the accessors import.
    imported_modules = {module for module, _name in reached}
    assert [row[0] in imported_modules for row in rows] == [True] * (len(rows) - 2) + [False, False]
    assert [row[1].partition(".")[2] for row in rows[-2:]] == ["dp_ranks", "dp_ranks"]


def test_every_seam_name_is_in_the_sglang_source_at_both_tags():
    # Skipped where no SGLang source is at hand. With it, each row is looked up in the parsed files of both
    # releases; nothing is imported.
    for version, python_root in sglang_source_roots(compat.TESTED_VERSIONS).items():
        assert seam_rows_not_in_source(SEAM_NAMES, python_root) == [], version
        # The look-up notices a name the source does not have.
        assert seam_rows_not_in_source(
            (("sglang.srt.entrypoints.engine", "Engine.no_such_name", "attribute"),), python_root
        )
