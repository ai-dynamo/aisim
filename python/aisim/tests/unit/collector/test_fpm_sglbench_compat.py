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
  modules and names the accessors import, and each fake records how it was called. The names and their
  modules are those of SGLang v0.5.22 and v0.5.23 (``__init__.py``, ``srt/plugins/__init__.py``,
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
import gc
import inspect
import re
import subprocess
import sys
import types
import warnings
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
TEST_PATH = Path(__file__).resolve()

SGLANG_22 = "baa9befa381511cd4d56f83980951e5a45fffc6d"
SGLANG_23 = "297e54fd105e974cdf401e5b4d20253a22c16f90"

# FAKE values. None of them was measured.
FAKE_E2E_LATENCY = 0.25
FAKE_CLOCK_START = 1000.0
CALL_TIMEOUT = 900.0

RESULT_KEYS = ["rid", "prompt_tokens", "cached_tokens", "completion_tokens", "finish_type", "num_retractions"]


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
    """FAKE of ``ServerArgs``: raw fields stay the input, resolution adds declarations beside them."""

    def __init__(self, argv):
        self.argv = argv
        self.model_path = argv[argv.index("--model-path") + 1]
        self.page_size = None
        self.declared = {}
        self.events = []

    def resolve_once(self):
        self.events.append("resolve_once")
        self.declared = {"page_size": 64, "device": "cuda"}

    def resolved_dict(self):
        self.events.append("resolved_dict")
        return {"model_path": self.model_path, "page_size": self.declared.get("page_size", self.page_size)}


class _FakeResolvingView:
    """FAKE of ``ResolvingConfig``: a live view that reads a declaration first and the raw field after it."""

    def __init__(self, server_args):
        object.__setattr__(self, "record", server_args)

    def __getattr__(self, name):
        record = object.__getattribute__(self, "record")
        if name in record.declared:
            return record.declared[name]
        return getattr(record, name)


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
        model_config_calls=[],
        platform=types.SimpleNamespace(is_cuda=True, is_hip=False),
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
        if "--no-such-flag" in argv:
            # argparse's answer to an unknown flag, which SGLang's parser does not catch.
            raise SystemExit(2)
        return _FakeServerArgs(argv)

    def resolving_view(server_args):
        state.events.append("resolving_view")
        return _FakeResolvingView(server_args)

    def get_platform():
        return state.platform

    class FakeModelConfig:
        """FAKE of ``ModelConfig``: only the two configs the accessor reads."""

        def __init__(self):
            self.hf_config = state.hf_config
            self.hf_text_config = state.hf_text_config

        @staticmethod
        def from_server_args(*args, **kwargs):
            state.model_config_calls.append((args, kwargs))
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


def _sglang_modules():
    return {name: module for name, module in sys.modules.items() if name.split(".")[0] == "sglang"}


def test_the_fake_package_tree_is_removed_again():
    before = _sglang_modules()
    with pytest.MonkeyPatch.context() as patch:
        state = _install_fake_sglang(patch)
        assert sys.modules["sglang"] is state.modules["sglang"]
        assert len(_sglang_modules()) >= 11
        assert compat.sglang_version() == "0.5.23"
    assert _sglang_modules() == before


FAKE_ARGV = ["--model-path", "/fake/model", "--incremental-streaming-output"]


# ---------------------------------------------------------------------------------------------------------------
# Constants and the plain accessors
# ---------------------------------------------------------------------------------------------------------------


def test_tested_versions_and_the_error_class():
    assert compat.TESTED_VERSIONS == ("0.5.22", "0.5.23")
    assert issubclass(compat.EngineCallError, RuntimeError)
    assert compat.EngineCallError is not RuntimeError


def test_public_functions_keep_the_signatures_the_launcher_is_written_against():
    expected = {
        "sglang_version": [],
        "load_plugins": [],
        "parse_server_args": ["argv"],
        "resolve": ["server_args"],
        "resolved_dict": ["server_args"],
        "platform_facts": [],
        "model_expert_keys": ["server_args"],
        "build_engine": ["server_args", "scheduler_entry"],
        "generate_batch": ["engine", "prompts", "rids", "max_new_tokens", "timeout_seconds"],
        "vocabulary": ["engine"],
        "child_pids": ["engine"],
        "shutdown": ["engine"],
    }
    for name, parameters in expected.items():
        function = getattr(compat, name)
        assert inspect.isfunction(function) and not inspect.iscoroutinefunction(function), name
        signature = inspect.signature(function)
        assert list(signature.parameters) == parameters, name
        for parameter in signature.parameters.values():
            assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, name
            assert parameter.default is inspect.Parameter.empty, name


@pytest.mark.parametrize("version", ["0.5.22", "0.5.23", "0.0.0.dev0"])
def test_sglang_version_is_the_package_version_string(fake_sglang, version):
    fake_sglang.modules["sglang"].__version__ = version
    assert compat.sglang_version() == version


def test_sglang_version_refuses_a_value_that_is_not_a_string(fake_sglang):
    fake_sglang.modules["sglang"].__version__ = (0, 5, 23)
    with pytest.raises(TypeError, match="__version__"):
        compat.sglang_version()


def test_load_plugins_calls_sglangs_loader_once_and_returns_nothing(fake_sglang):
    assert compat.load_plugins() is None
    assert fake_sglang.events == ["load_plugins"]


def test_parse_server_args_goes_through_sglangs_own_parser(fake_sglang):
    argv = list(FAKE_ARGV)
    record = compat.parse_server_args(argv)
    assert isinstance(record, _FakeServerArgs)
    assert fake_sglang.events == ["prepare_server_args"]
    assert fake_sglang.parsed == [FAKE_ARGV] and isinstance(fake_sglang.parsed[0], list)
    assert argv == FAKE_ARGV, "the caller's list is not changed"
    # Parsing resolves nothing: the raw field is still the operator's input.
    assert record.page_size is None and record.events == []


def test_parse_server_args_lets_the_exit_of_an_unknown_flag_pass(fake_sglang):
    with pytest.raises(SystemExit) as caught:
        compat.parse_server_args(["--model-path", "/fake/model", "--no-such-flag"])
    assert caught.value.code == 2


def test_resolve_resolves_once_and_returns_the_live_view_of_the_same_record(fake_sglang):
    record = compat.parse_server_args(list(FAKE_ARGV))
    view = compat.resolve(record)
    assert record.events == ["resolve_once"]
    assert fake_sglang.events == ["prepare_server_args", "resolving_view"]
    assert isinstance(view, _FakeResolvingView)
    assert object.__getattribute__(view, "record") is record
    # The view has the resolved value; the record keeps the raw input.
    assert view.page_size == 64 and record.page_size is None
    assert view.model_path == "/fake/model"
    with pytest.raises(AttributeError):
        _ = view.no_such_field


def test_resolve_lets_sglangs_refusal_pass(fake_sglang):
    record = compat.parse_server_args(list(FAKE_ARGV))

    def refuse():
        raise ValueError("FAKE: SGLang refuses this setup")

    record.resolve_once = refuse
    with pytest.raises(ValueError, match="refuses this setup"):
        compat.resolve(record)
    assert "resolving_view" not in fake_sglang.events


def test_resolved_dict_is_the_records_own_dump(fake_sglang):
    record = compat.parse_server_args(list(FAKE_ARGV))
    compat.resolve(record)
    assert compat.resolved_dict(record) == {"model_path": "/fake/model", "page_size": 64}
    assert record.events == ["resolve_once", "resolved_dict"]


def test_resolved_dict_refuses_a_value_that_is_not_a_dict():
    record = types.SimpleNamespace(resolved_dict=lambda: [("page_size", 1)])
    with pytest.raises(TypeError, match="resolved_dict"):
        compat.resolved_dict(record)


@pytest.mark.parametrize(("is_cuda", "is_hip"), [(True, False), (False, True), (False, False), (True, True)])
def test_platform_facts_reports_the_two_probes_under_their_own_names(fake_sglang, is_cuda, is_hip):
    fake_sglang.platform = types.SimpleNamespace(is_cuda=is_cuda, is_hip=is_hip)
    assert compat.platform_facts() == {"is_cuda": is_cuda, "is_hip": is_hip}


@pytest.mark.parametrize("name", ["is_cuda", "is_hip"])
def test_platform_facts_refuses_a_fact_that_is_not_a_bool(fake_sglang, name):
    setattr(fake_sglang.platform, name, None)
    with pytest.raises(TypeError, match=name):
        compat.platform_facts()


def test_platform_facts_needs_both_probes(fake_sglang):
    fake_sglang.platform = types.SimpleNamespace(is_cuda=True)
    with pytest.raises(AttributeError, match="is_hip"):
        compat.platform_facts()


# ---------------------------------------------------------------------------------------------------------------
# model_expert_keys
# ---------------------------------------------------------------------------------------------------------------


def test_expert_keys_are_the_six_named_ones():
    assert compat.EXPERT_KEYS == (
        "num_local_experts",
        "num_experts",
        "n_routed_experts",
        "moe_num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
    )


def test_a_dense_model_has_no_expert_keys_and_the_config_is_built_from_the_record(fake_sglang):
    record = compat.parse_server_args(list(FAKE_ARGV))
    assert compat.model_expert_keys(record) == {}
    # Built once, from the record alone: no model path, revision or draft flag of our own.
    assert fake_sglang.model_config_calls == [((record,), {})]


@pytest.mark.parametrize("key", list(compat.EXPERT_KEYS))
@pytest.mark.parametrize("part", ["hf_config", "hf_text_config"])
def test_each_expert_key_counts_on_the_top_config_and_on_the_text_config(fake_sglang, key, part):
    setattr(getattr(fake_sglang, part), key, 8)
    assert compat.model_expert_keys(object()) == {key: 8}


@pytest.mark.parametrize("value", [0, None, False, 0.0, "", []])
def test_a_falsy_expert_value_is_a_dense_model(fake_sglang, value):
    for key in compat.EXPERT_KEYS:
        setattr(fake_sglang.hf_config, key, value)
        setattr(fake_sglang.hf_text_config, key, value)
    assert compat.model_expert_keys(object()) == {}


def test_expert_keys_come_in_their_fixed_order_with_the_top_configs_value_first(fake_sglang):
    fake_sglang.hf_text_config.moe_intermediate_size = 1408
    fake_sglang.hf_text_config.num_experts = 128
    fake_sglang.hf_config.num_experts = 64
    fake_sglang.hf_config.num_experts_per_tok = 1  # truthy: stricter than SGLang's own test, which wants above 1
    fake_sglang.hf_config.n_routed_experts = 0
    found = compat.model_expert_keys(object())
    assert found == {"num_experts": 64, "num_experts_per_tok": 1, "moe_intermediate_size": 1408}
    assert list(found) == ["num_experts", "num_experts_per_tok", "moe_intermediate_size"]


@pytest.mark.parametrize("part", ["hf_config", "hf_text_config"])
def test_a_model_config_without_one_of_its_two_configs_is_an_error_not_a_dense_model(fake_sglang, part):
    fake_sglang.hf_text_config.num_experts = 8
    fake_sglang.hf_config.num_experts = 8
    setattr(fake_sglang, part, None)
    with pytest.raises(ValueError, match=part):
        compat.model_expert_keys(object())
    module = fake_sglang.modules["sglang.srt.configs.model_config"]
    module.ModelConfig = types.SimpleNamespace(
        from_server_args=lambda record: types.SimpleNamespace(
            **{name: types.SimpleNamespace() for name in ("hf_config", "hf_text_config") if name != part}
        )
    )
    with pytest.raises(AttributeError, match=part):
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


def test_build_engine_makes_a_new_subclass_for_each_entry(fake_sglang):
    def other_entry(*args, **kwargs):
        raise AssertionError("never run")

    first = compat.build_engine(object(), _fake_recorder_entry)
    second = compat.build_engine(object(), other_entry)
    assert first.launched_with is _fake_recorder_entry and second.launched_with is other_entry
    assert type(first) is not type(second)


def test_build_engine_refuses_a_running_event_loop(fake_sglang):
    async def inside():
        return compat.build_engine(object(), _fake_recorder_entry)

    event_loop = asyncio.new_event_loop()
    try:
        with pytest.raises(RuntimeError, match="running event loop"):
            event_loop.run_until_complete(inside())
    finally:
        event_loop.close()
    assert fake_sglang.engine_kwargs == []


def test_build_engine_refuses_an_entry_that_is_not_callable(fake_sglang):
    with pytest.raises(TypeError, match="scheduler_entry"):
        compat.build_engine(object(), "sglbench_recorder.recorder_entry")
    assert fake_sglang.engine_kwargs == []


def test_build_engine_refuses_an_engine_class_without_the_seam(fake_sglang):
    class FakeEngineWithoutSeam:
        """FAKE of an Engine class that has lost the scheduler entry attribute."""

        def __init__(self, **kwargs):
            fake_sglang.engine_kwargs.append(kwargs)

    fake_sglang.modules["sglang.srt.entrypoints.engine"].Engine = FakeEngineWithoutSeam
    with pytest.raises(AttributeError, match="run_scheduler_process_func"):
        compat.build_engine(object(), _fake_recorder_entry)
    assert fake_sglang.engine_kwargs == []


def test_build_engine_refuses_a_proxy_in_the_place_of_the_class(fake_sglang):
    fake_sglang.modules["sglang.srt.entrypoints.engine"].Engine = _FakeLazyImport(fake_sglang.events)
    with pytest.raises(TypeError, match="not a class"):
        compat.build_engine(object(), _fake_recorder_entry)
    assert "sglang.Engine proxy called" not in fake_sglang.events


def test_build_engine_lets_a_build_error_of_sglang_pass(fake_sglang):
    def refuse(self, **kwargs):
        raise AssertionError("FAKE: check_server_args refuses this setup")

    fake_sglang.real_engine_class.__init__ = refuse
    with pytest.raises(AssertionError, match="refuses this setup"):
        compat.build_engine(object(), _fake_recorder_entry)


# ---------------------------------------------------------------------------------------------------------------
# generate_batch: the FAKE engine and its stream
# ---------------------------------------------------------------------------------------------------------------


def _length(count):
    """The finish reason of a request that produced all its tokens (schedule_batch.py ``FINISH_LENGTH``)."""
    return {"type": "length", "length": count}


def _meta(rid, *, finish=None, prompt=8, cached=0, completion=1, retractions=0):
    """A ``meta_info`` as ``TokenizerManager._handle_batch_output`` builds it for a generation request."""
    return {
        "id": rid,
        "finish_reason": finish,
        "prompt_tokens": prompt,
        "weight_version": "default",
        "num_retractions": retractions,
        "reasoning_tokens": 0,
        "completion_tokens": completion,
        "cached_tokens": cached,
    }


def _chunk(index, rid, output_ids=(1,), **meta):
    """One chunk of a batch stream; ``_stream_batch_responses`` adds ``index`` to what a request yields."""
    return {"text": "", "output_ids": list(output_ids), "meta_info": _meta(rid, **meta), "index": index}


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
    if event_loop.is_closed():
        return
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


def _fails(engine, count, match, **kwargs):
    """The call raises EngineCallError with ``match``, was sent exactly once and returns the error."""
    with pytest.raises(compat.EngineCallError, match=match) as caught:
        _generate(engine, count, **kwargs)
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
        },
        {
            "rid": r1,
            "prompt_tokens": 9,
            "cached_tokens": 2,
            "completion_tokens": 2,
            "finish_type": "length",
            "num_retractions": 0,
        },
        {
            "rid": r2,
            "prompt_tokens": 10,
            "cached_tokens": 4,
            "completion_tokens": 2,
            "finish_type": "stop",
            "num_retractions": 0,
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


def test_a_batch_of_one_with_an_index_and_a_single_stream_without_one(loop):
    (rid,) = _rids(1)
    with_index = _FakeStreamEngine(loop, [_chunk(0, rid), _chunk(0, rid, cached=7, completion=2, finish=_length(2))])
    assert _generate(with_index, 1) == [
        {
            "rid": rid,
            "prompt_tokens": 8,
            "cached_tokens": 7,
            "completion_tokens": 2,
            "finish_type": "length",
            "num_retractions": 0,
        }
    ]
    assert with_index.calls[0]["input_ids"] == [[100] * 8], "a batch of one is still a list of lists"

    # The stream of a single request carries no index; with one request there is one place to put it.
    first, last = _chunk(0, rid), _chunk(0, rid, completion=2, finish=_length(2))
    del first["index"], last["index"]
    without_index = _FakeStreamEngine(loop, [first, last])
    assert _generate(without_index, 1)[0]["completion_tokens"] == 2


def test_a_merged_final_chunk_counts_like_separate_chunks(loop):
    r0, r1 = _rids(2)
    # Request 0 is seen in ONE chunk: SGLang merged its seven queued chunks, so the token ids are joined and
    # the meta_info is the last one. Request 1 is seen step by step.
    script = [_chunk(1, r1, [11], completion=1)]
    script += [_chunk(0, r0, [1, 2, 3, 4, 5, 6, 7], completion=7, finish=_length(7))]
    script += [_chunk(1, r1, [12 + step], completion=2 + step) for step in range(5)]
    script += [_chunk(1, r1, [17], completion=7, finish=_length(7))]
    results = _generate(_FakeStreamEngine(loop, script), 2, max_new_tokens=7)
    assert [result["completion_tokens"] for result in results] == [7, 7]
    assert [result["finish_type"] for result in results] == ["length", "length"]


def test_a_finish_reason_without_a_type_gives_none_as_finish_type(loop):
    (rid,) = _rids(1)
    engine = _FakeStreamEngine(loop, [_chunk(0, rid, completion=2, finish={"length": 2})])
    assert _generate(engine, 1)[0]["finish_type"] is None


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


class _FakeLoopAtItsLimit:
    """FAKE loop that does not run the call and reports the time limit, as ``asyncio.wait_for`` does."""

    def __init__(self):
        self.given = []

    def run_until_complete(self, awaitable):
        self.given.append(awaitable)
        raise TimeoutError


def test_a_timeout_of_the_loop_becomes_an_engine_call_error(loop):
    fake_loop = _FakeLoopAtItsLimit()
    engine = _FakeStreamEngine(fake_loop, [])
    with pytest.raises(compat.EngineCallError, match=r"timed out after 900\.0 s") as caught:
        _generate(engine, 2)
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert _rids(2)[0] in str(caught.value)
    # The loop was given the call wrapped in asyncio.wait_for, once.
    assert len(fake_loop.given) == 1 and fake_loop.given[0].__qualname__ == "wait_for"
    assert engine.calls == []


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


def test_the_limit_is_the_one_the_caller_gives(loop):
    r0, r1 = _rids(2)
    script = [
        _chunk(0, r0),
        _wait_to_be_cancelled(loop, 30.5),
        _chunk(0, r0, completion=2, finish=_length(2)),
        _chunk(1, r1, completion=2, finish=_length(2)),
    ]
    engine = _FakeStreamEngine(loop, script)
    _fails(engine, 2, r"timed out after 30 s", timeout_seconds=30)
    assert engine.stream_events == ["cancelled"]


def test_a_timeout_error_raised_inside_the_call_is_not_reported_as_the_time_limit(loop):
    (rid,) = _rids(1)
    engine = _FakeStreamEngine(loop, [_chunk(0, rid), TimeoutError("FAKE: a socket timed out")])
    error = _fails(engine, 1, "the engine call raised TimeoutError: FAKE: a socket timed out")
    assert "timed out after" not in str(error)


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


def test_an_abort_with_all_counts_fails_the_call(loop):
    r0, r1 = _rids(2)
    abort = {"type": "abort", "message": "FAKE: input is too long", "status_code": 400, "err_type": None}
    script = [
        _chunk(0, r0, completion=2, finish=_length(2)),
        _chunk(1, r1, completion=0, finish=abort),
    ]
    error = _fails(_FakeStreamEngine(loop, script), 2, "was aborted")
    assert r1 in str(error) and "input is too long" in str(error) and "400" in str(error)


@pytest.mark.parametrize("absent", ["prompt_tokens", "cached_tokens"])
def test_a_final_chunk_without_prompt_or_cached_tokens_is_an_abort_whatever_its_finish_type(loop, absent):
    (rid,) = _rids(1)
    final = _chunk(0, rid, completion=2, finish=_length(2))
    del final["meta_info"][absent]
    error = _fails(_FakeStreamEngine(loop, [final]), 1, "was aborted")
    assert absent in str(error)


def test_an_unfinished_chunk_without_counts_is_not_an_abort(loop):
    (rid,) = _rids(1)
    early = _chunk(0, rid)
    del early["meta_info"]["prompt_tokens"], early["meta_info"]["cached_tokens"]
    engine = _FakeStreamEngine(loop, [early, _chunk(0, rid, completion=2, finish=_length(2))])
    assert _generate(engine, 1)[0]["completion_tokens"] == 2


# ---------------------------------------------------------------------------------------------------------------
# generate_batch: chunks that cannot be placed, missing keys, a stream that ends early
# ---------------------------------------------------------------------------------------------------------------


def test_a_chunk_without_index_fails_a_call_with_more_than_one_request(loop):
    r0, r1 = _rids(2)
    lost = _chunk(1, r1)
    del lost["index"]
    script = [_chunk(0, r0), lost, _chunk(0, r0, completion=2, finish=_length(2))]
    engine = _FakeStreamEngine(loop, script)
    _fails(engine, 2, "has no index")
    assert engine.stream_events == ["closed before its end"]


@pytest.mark.parametrize("index", [2, -1, True, "1", None, 1.0])
def test_an_index_outside_the_call_fails_it(loop, index):
    r0, r1 = _rids(2)
    chunk = _chunk(1, r1)
    chunk["index"] = index
    _fails(_FakeStreamEngine(loop, [_chunk(0, r0), chunk]), 2, "has index")


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


def test_an_empty_stream_fails_the_call(loop):
    _fails(_FakeStreamEngine(loop, []), 3, "without the final chunk of 3 of 3 requests")


def test_a_chunk_after_the_final_chunk_of_its_request_fails_the_call(loop):
    (rid,) = _rids(1)
    script = [_chunk(0, rid, completion=2, finish=_length(2)), _chunk(0, rid, completion=3, finish=_length(3))]
    _fails(_FakeStreamEngine(loop, script), 1, "after its final chunk")


@pytest.mark.parametrize("key", ["completion_tokens", "num_retractions"])
def test_a_final_chunk_without_a_count_fails_the_call(loop, key):
    (rid,) = _rids(1)
    final = _chunk(0, rid, completion=2, finish=_length(2))
    del final["meta_info"][key]
    _fails(_FakeStreamEngine(loop, [final]), 1, f"has no {key}")


@pytest.mark.parametrize("key", ["prompt_tokens", "cached_tokens", "completion_tokens", "num_retractions"])
@pytest.mark.parametrize("value", ["8", None, True, -1, 8.0])
def test_a_count_that_is_not_a_plain_int_fails_the_call(loop, key, value):
    (rid,) = _rids(1)
    final = _chunk(0, rid, completion=2, finish=_length(2))
    final["meta_info"][key] = value
    _fails(_FakeStreamEngine(loop, [final]), 1, f"{key} is")


def test_chunks_without_meta_info_or_finish_reason_fail_the_call(loop):
    (rid,) = _rids(1)
    no_meta = _chunk(0, rid)
    del no_meta["meta_info"]
    _fails(_FakeStreamEngine(loop, [no_meta]), 1, "no meta_info")

    no_finish = _chunk(0, rid)
    del no_finish["meta_info"]["finish_reason"]
    _fails(_FakeStreamEngine(loop, [no_finish]), 1, "no finish_reason")

    _fails(_FakeStreamEngine(loop, ["not a dict"]), 1, "not a dict")
    _fails(_FakeStreamEngine(loop, [_chunk(0, rid, finish="length")]), 1, "finish_reason is 'length'")
    _fails(_FakeStreamEngine(loop, [_chunk(0, rid, finish={"type": 7})]), 1, "finish_reason type")


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


def test_an_error_while_the_stream_is_closed_becomes_an_engine_call_error(loop):
    class _FakeStreamThatFailsToClose:
        """FAKE stream whose cleanup raises, as a failing abort inside SGLang would."""

        def __init__(self, chunks):
            self.chunks = iter(chunks)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.chunks)
            except StopIteration:
                raise StopAsyncIteration from None

        async def aclose(self):
            raise TimeoutError("FAKE: the abort could not be sent")

    class _FakeEngine(_FakeStreamEngine):
        async def async_generate(self, **kwargs):
            self.calls.append(kwargs)
            return _FakeStreamThatFailsToClose(self.script)

    (rid,) = _rids(1)
    engine = _FakeEngine(loop, [_chunk(0, rid, completion=2, finish=_length(2))])
    error = _fails(engine, 1, "closing the stream raised TimeoutError")
    assert "timed out after" not in str(error)


@pytest.mark.parametrize("stop", [SystemExit(0), KeyboardInterrupt()])
def test_system_exit_and_keyboard_interrupt_pass_through(loop, stop):
    (rid,) = _rids(1)
    engine = _FakeStreamEngine(loop, [_chunk(0, rid), stop])
    with pytest.raises(type(stop)) as caught:
        _generate(engine, 1)
    assert caught.value is stop
    assert not isinstance(caught.value, compat.EngineCallError)
    assert len(engine.calls) == 1


def test_another_base_exception_of_the_call_passes_through(loop):
    class _FakeStop(BaseException):
        """FAKE of a launcher's stop exception, raised by a signal handler while the call runs."""

    (rid,) = _rids(1)
    stop = _FakeStop("FAKE: SIGINT")
    with pytest.raises(_FakeStop) as caught:
        _generate(_FakeStreamEngine(loop, [_chunk(0, rid), stop]), 1)
    assert caught.value is stop


def _no_coroutine_is_left_unawaited(call):
    """Run ``call``, which must raise EngineCallError, and return its text; fail on a never-awaited warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            call()
        except compat.EngineCallError as error:
            message = str(error)
        else:
            pytest.fail("the call did not fail")
        gc.collect()
    unawaited = [str(warning.message) for warning in caught if issubclass(warning.category, RuntimeWarning)]
    assert unawaited == []
    return message


def test_a_closed_loop_fails_the_call_and_leaves_no_coroutine_behind(loop):
    engine = _FakeStreamEngine(loop, [])
    loop.close()
    message = _no_coroutine_is_left_unawaited(lambda: _generate(engine, 1))
    assert "the engine call failed: RuntimeError" in message and "closed" in message
    assert engine.calls == []


def test_a_loop_that_refuses_the_call_leaves_no_coroutine_behind():
    engine = _FakeStreamEngine(_FakeLoopAtItsLimit(), [])
    message = _no_coroutine_is_left_unawaited(lambda: _generate(engine, 1))
    assert "timed out after" in message
    assert inspect.getcoroutinestate(engine.loop.given[0]) == inspect.CORO_CLOSED


def test_an_engine_without_a_loop_fails_the_call():
    fake_engine = types.SimpleNamespace(calls=[])
    message = _no_coroutine_is_left_unawaited(lambda: _generate(fake_engine, 1))
    assert "AttributeError" in message and "loop" in message


def test_a_call_from_inside_the_running_engine_loop_fails(loop):
    (rid,) = _rids(1)
    engine = _FakeStreamEngine(loop, [_chunk(0, rid, completion=2, finish=_length(2))])

    async def inside():
        return _generate(engine, 1)

    with pytest.raises(compat.EngineCallError, match="already running"):
        loop.run_until_complete(inside())
    assert engine.calls == []


# ---------------------------------------------------------------------------------------------------------------
# generate_batch: arguments that SGLang would read as another kind of call
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompts", "rids", "match"),
    [
        ([1, 2, 3], ["a"], "prompt 0"),  # a flat list is ONE request for SGLang, with no index in its stream
        ([], [], "non-empty list"),
        ([[1, 2], []], ["a", "b"], "prompt 1"),
        (([1, 2],), ["a"], "non-empty list"),
        ([(1, 2)], ["a"], "prompt 0"),
        ([[1, 2], [3]], ["a"], "one per prompt"),
        ([[1, 2]], "a", "one per prompt"),  # SGLang would expand a string id to "a_0"
        ([[1, 2], [3]], ["a", "a"], "repeat"),
        ([[1, 2]], [""], "request id 0"),
        ([[1, 2]], [7], "request id 0"),
    ],
)
def test_a_call_that_is_not_a_batch_of_prompts_with_one_id_each_is_refused_before_it_is_sent(
    loop, prompts, rids, match
):
    engine = _FakeStreamEngine(loop, [])
    with pytest.raises(compat.EngineCallError, match=match):
        compat.generate_batch(engine, prompts, rids, 2, CALL_TIMEOUT)
    assert engine.calls == [] and engine.loops_seen == []


@pytest.mark.parametrize("max_new_tokens", [0, -1, True, "7", 7.0, None])
def test_an_output_length_that_is_not_a_positive_int_is_refused(loop, max_new_tokens):
    engine = _FakeStreamEngine(loop, [])
    with pytest.raises(compat.EngineCallError, match="max_new_tokens"):
        _generate(engine, 1, max_new_tokens=max_new_tokens)
    assert engine.calls == []


@pytest.mark.parametrize("timeout_seconds", [None, 0, -5.0, True, "900", float("inf"), float("nan")])
def test_a_call_without_a_finite_time_limit_is_refused(loop, timeout_seconds):
    engine = _FakeStreamEngine(loop, [])
    with pytest.raises(compat.EngineCallError, match="timeout_seconds"):
        _generate(engine, 1, timeout_seconds=timeout_seconds)
    assert engine.calls == []


# ---------------------------------------------------------------------------------------------------------------
# vocabulary, child_pids, shutdown
# ---------------------------------------------------------------------------------------------------------------


class _FakeTokenizer:
    """FAKE tokenizer with the four names the accessor reads; SGLang attaches the last one itself."""

    def __init__(self, vocab_size=1000, special=(0, 1, 2), added=None, stop_ids=None):
        self.vocab_size = vocab_size
        self.all_special_ids = list(special)
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


def test_vocabulary_without_stop_ids_takes_special_and_added_ids_only():
    vocabulary = compat.vocabulary(_fake_engine_with(_FakeTokenizer(stop_ids=None)))
    assert vocabulary == {"size": 1000, "excluded": [0, 1, 2, 1000, 1001]}
    only_special = compat.vocabulary(_fake_engine_with(_FakeTokenizer(added={})))
    assert only_special == {"size": 1000, "excluded": [0, 1, 2]}
    only_added = compat.vocabulary(_fake_engine_with(_FakeTokenizer(special=())))
    assert only_added == {"size": 1000, "excluded": [1000, 1001]}
    only_stop = compat.vocabulary(_fake_engine_with(_FakeTokenizer(special=(), added={}, stop_ids={9})))
    assert only_stop == {"size": 1000, "excluded": [9]}


@pytest.mark.parametrize(("tokenizer_size", "model_size", "size"), [(1000, 900, 900), (900, 1000, 900), (7, 7, 7)])
def test_vocabulary_size_is_the_smaller_of_the_tokenizers_and_the_models(tokenizer_size, model_size, size):
    engine = _fake_engine_with(_FakeTokenizer(vocab_size=tokenizer_size), model_vocab_size=model_size)
    assert compat.vocabulary(engine)["size"] == size


@pytest.mark.parametrize("attribute", ["vocab_size", "all_special_ids", "get_added_vocab", "additional_stop_token_ids"])
def test_vocabulary_refuses_a_tokenizer_without_one_of_its_attributes(attribute):
    tokenizer = _FakeTokenizer()
    if attribute == "get_added_vocab":
        tokenizer = types.SimpleNamespace(vocab_size=1000, all_special_ids=[0], additional_stop_token_ids=None)
    else:
        delattr(tokenizer, attribute)
    with pytest.raises(AttributeError, match=attribute) as caught:
        compat.vocabulary(_fake_engine_with(tokenizer))
    assert type(tokenizer).__name__ in str(caught.value)


def test_vocabulary_refuses_an_engine_without_a_tokenizer():
    with pytest.raises(ValueError, match="skip-tokenizer-init"):
        compat.vocabulary(_fake_engine_with(None))


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        ({"vocab_size": None}, TypeError, "tokenizer.vocab_size"),
        ({"vocab_size": True}, TypeError, "tokenizer.vocab_size"),
        ({"vocab_size": 0}, ValueError, "tokenizer.vocab_size"),
        ({"special": (0, "1")}, TypeError, "all_special_ids"),
        ({"special": (0, None)}, TypeError, "all_special_ids"),
        ({"special": (0, -1)}, TypeError, "all_special_ids"),
        ({"special": (0, True)}, TypeError, "all_special_ids"),
        ({"added": ["<fake_a>"]}, TypeError, "get_added_vocab"),
        ({"added": {"<fake_a>": "1000"}}, TypeError, "get_added_vocab"),
        ({"stop_ids": {"7"}}, TypeError, "additional_stop_token_ids"),
        ({"stop_ids": 7}, TypeError, "additional_stop_token_ids"),
    ],
)
def test_vocabulary_refuses_values_that_are_not_what_the_names_promise(change, error, match):
    with pytest.raises(error, match=match):
        compat.vocabulary(_fake_engine_with(_FakeTokenizer(**change)))


def test_vocabulary_refuses_a_special_id_list_that_is_none_and_a_bad_model_size():
    tokenizer = _FakeTokenizer()
    tokenizer.all_special_ids = None
    with pytest.raises(TypeError, match="all_special_ids"):
        compat.vocabulary(_fake_engine_with(tokenizer))
    with pytest.raises(TypeError, match="model_config.vocab_size"):
        compat.vocabulary(_fake_engine_with(_FakeTokenizer(), model_vocab_size=None))
    with pytest.raises(ValueError, match="model_config.vocab_size"):
        compat.vocabulary(_fake_engine_with(_FakeTokenizer(), model_vocab_size=-1))


def test_child_pids_is_a_copy_of_the_engines_list():
    held = [4101, 4102, 4103]
    fake_engine = types.SimpleNamespace(get_all_child_pids=lambda: held)
    pids = compat.child_pids(fake_engine)
    assert pids == [4101, 4102, 4103] and pids is not held
    assert compat.child_pids(types.SimpleNamespace(get_all_child_pids=lambda: ())) == []


@pytest.mark.parametrize("pid", ["4101", None, True, 41.0])
def test_child_pids_refuses_what_is_not_a_process_id(pid):
    fake_engine = types.SimpleNamespace(get_all_child_pids=lambda: [4101, pid])
    with pytest.raises(TypeError, match="process id"):
        compat.child_pids(fake_engine)


def test_shutdown_calls_the_engines_shutdown_once_and_lets_its_error_pass():
    calls = []
    fake_engine = types.SimpleNamespace(shutdown=lambda: calls.append("shutdown"))
    assert compat.shutdown(fake_engine) is None
    assert calls == ["shutdown"]

    failure = RuntimeError("FAKE: children did not exit within 60 s")

    def fail():
        calls.append("shutdown")
        raise failure

    with pytest.raises(RuntimeError) as caught:
        compat.shutdown(types.SimpleNamespace(shutdown=fail))
    assert caught.value is failure and calls == ["shutdown", "shutdown"]


# ---------------------------------------------------------------------------------------------------------------
# Module hygiene: what the pod and the repository gates require of the file itself
# ---------------------------------------------------------------------------------------------------------------

SGLANG_CALLERS = (
    "sglang_version",
    "load_plugins",
    "parse_server_args",
    "resolve",
    "platform_facts",
    "model_expert_keys",
    "build_engine",
)


def test_module_imports_without_sglang_torch_or_numpy_and_needs_sglang_only_inside_its_functions():
    code = (
        "import sys\n"
        "class Refuse:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('sglang', 'torch', 'numpy'):\n"
        "            raise ModuleNotFoundError('refused: ' + name, name=name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Refuse())\n"
        "sys.path.append(sys.argv[1])\n"
        "import sglbench_compat as c\n"
        "taboo = ('sglang', 'torch', 'numpy', 'dynamo', 'aisim', 'aisim_core', 'collector')\n"
        "bad = [n for n in sys.modules if n.split('.')[0] in taboo]\n"
        "bad += [n for n in sys.modules if n.startswith('sglbench_') and n != c.__name__]\n"
        "assert not bad, bad\n"
        "class Record:\n"
        "    def resolve_once(self):\n"
        "        raise AssertionError('resolved before SGLang was imported')\n"
        "calls = {\n"
        "    'sglang_version': lambda: c.sglang_version(),\n"
        "    'load_plugins': lambda: c.load_plugins(),\n"
        "    'parse_server_args': lambda: c.parse_server_args(['--model-path', 'm']),\n"
        "    'resolve': lambda: c.resolve(Record()),\n"
        "    'platform_facts': lambda: c.platform_facts(),\n"
        "    'model_expert_keys': lambda: c.model_expert_keys(Record()),\n"
        "    'build_engine': lambda: c.build_engine(Record(), print),\n"
        "}\n"
        "assert sorted(calls) == sorted(sys.argv[2].split(',')), sorted(calls)\n"
        "for name, call in calls.items():\n"
        "    try:\n"
        "        call()\n"
        "    except ModuleNotFoundError as error:\n"
        "        assert error.name.split('.')[0] == 'sglang', (name, error.name)\n"
        "    else:\n"
        "        raise AssertionError(name + ' did not import sglang')\n"
        "print(c.TESTED_VERSIONS)\n"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(RUNTIME), ",".join(SGLANG_CALLERS)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "('0.5.22', '0.5.23')"


def _module_tree():
    return ast.parse(MODULE_PATH.read_text(encoding="utf-8"))


def test_module_level_imports_are_standard_library_and_sglang_is_imported_inside_functions_only():
    tree = _module_tree()
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports do not resolve in the pod"
            top_level.add(node.module.split(".")[0])
    assert top_level <= set(sys.stdlib_module_names) | {"__future__"}
    assert top_level == {"__future__", "asyncio", "inspect"}

    everywhere = set()
    sglang_importers = set()
    for function in [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)]:
        for node in ast.walk(function):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0
                names = [node.module]
            for name in names:
                everywhere.add(name.split(".")[0])
                if name.split(".")[0] == "sglang":
                    sglang_importers.add(function.name)
    assert everywhere == {"sglang"}, "inside functions only SGLang is imported"
    # resolved_dict, generate_batch, vocabulary, child_pids and shutdown work on objects they are given.
    assert sglang_importers == set(SGLANG_CALLERS)

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"environ", "getenv", "putenv"}, "the module reads no environment variable"
        elif isinstance(node, ast.Name):
            assert node.id not in {"__import__", "importlib", "os"}


def test_module_imports_the_sglang_names_from_the_modules_that_hold_them_at_both_tags():
    imported = set()
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.ImportFrom):
            imported.update((node.module, alias.name) for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update((alias.name, None) for alias in node.names)
    assert {pair for pair in imported if pair[0].split(".")[0] == "sglang"} == {
        ("sglang", None),
        ("sglang.srt.plugins", "load_plugins"),
        ("sglang.srt.server_args", "prepare_server_args"),
        ("sglang.srt.arg_groups.overrides", "resolving_view"),
        ("sglang.srt.runtime_context", "get_platform"),
        ("sglang.srt.configs.model_config", "ModelConfig"),
        ("sglang.srt.entrypoints.engine", "Engine"),
    }


# The seam rows, typed here on their own, so that a change of the list has to be made in two places. The
# rows of kind "name" and "class" and ``ModelConfig.from_server_args`` come from the imports of the test
# above; the others are methods the accessors call on a record or an engine that SGLang built.
SEAM_NAMES = (
    ("sglang", "__version__", "name"),
    ("sglang.srt.plugins", "load_plugins", "name"),
    ("sglang.srt.server_args", "prepare_server_args", "name"),
    ("sglang.srt.server_args", "ServerArgs.resolve_once", "attribute"),
    ("sglang.srt.server_args", "ServerArgs.resolved_dict", "attribute"),
    ("sglang.srt.arg_groups.overrides", "resolving_view", "name"),
    ("sglang.srt.runtime_context", "get_platform", "name"),
    ("sglang.srt.configs.model_config", "ModelConfig.from_server_args", "attribute"),
    ("sglang.srt.entrypoints.engine", "Engine", "class"),
    ("sglang.srt.entrypoints.engine", "Engine.run_scheduler_process_func", "attribute"),
    ("sglang.srt.entrypoints.engine", "Engine.async_generate", "attribute"),
    ("sglang.srt.entrypoints.engine", "Engine.get_all_child_pids", "attribute"),
    ("sglang.srt.entrypoints.engine", "Engine.shutdown", "attribute"),
)


def test_seam_names_are_the_typed_rows_and_name_everything_the_module_imports_from_sglang():
    assert compat.SGLANG_SEAM_NAMES == SEAM_NAMES
    assert len(set(SEAM_NAMES)) == len(SEAM_NAMES)
    reached = sglang_names_reached_by_import(_module_tree())
    assert reached == {
        ("sglang", None),
        ("sglang", "__version__"),
        ("sglang.srt.plugins", "load_plugins"),
        ("sglang.srt.server_args", "prepare_server_args"),
        ("sglang.srt.arg_groups.overrides", "resolving_view"),
        ("sglang.srt.runtime_context", "get_platform"),
        ("sglang.srt.configs.model_config", "ModelConfig"),
        ("sglang.srt.configs.model_config", "ModelConfig.from_server_args"),
        ("sglang.srt.entrypoints.engine", "Engine"),
    }
    # An import or a read that gets no row is noticed here, so the list cannot fall behind the code.
    assert seam_rows_missing_for(reached, SEAM_NAMES) == []
    assert seam_rows_missing_for(reached, SEAM_NAMES[1:]) == [("sglang", None), ("sglang", "__version__")]
    assert seam_rows_missing_for(reached, SEAM_NAMES[:7] + SEAM_NAMES[8:]) == [
        ("sglang.srt.configs.model_config", "ModelConfig"),
        ("sglang.srt.configs.model_config", "ModelConfig.from_server_args"),
    ]
    # The two tests ``build_engine`` makes of the class before it subclasses it are rows as well.
    assert ("sglang.srt.entrypoints.engine", "Engine", "class") in SEAM_NAMES
    assert ("sglang.srt.entrypoints.engine", "Engine.run_scheduler_process_func", "attribute") in SEAM_NAMES


def test_every_seam_name_is_in_the_sglang_source_at_both_tags():
    # Skipped where no SGLang source is at hand. With it, each row is looked up in the parsed files of both
    # releases; nothing is imported.
    for version, python_root in sglang_source_roots(compat.TESTED_VERSIONS).items():
        assert seam_rows_not_in_source(SEAM_NAMES, python_root) == [], version
        # The look-up notices a name the source does not have.
        assert seam_rows_not_in_source(
            (("sglang.srt.entrypoints.engine", "Engine.no_such_name", "attribute"),), python_root
        )


def test_module_has_no_version_branch():
    tree = _module_tree()
    version_strings = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in compat.TESTED_VERSIONS
    ]
    assert sorted(version_strings) == ["0.5.22", "0.5.23"], "the two strings stand in TESTED_VERSIONS only"
    uses = [node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == "TESTED_VERSIONS"]
    assert len(uses) == 1 and isinstance(uses[0].ctx, ast.Store), "no function reads the tested versions"
    version_reads = [
        function.name
        for function in ast.walk(tree)
        if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef)
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute) and "version" in node.attr
    ]
    assert version_reads == ["sglang_version"]


def test_module_header_docstring_and_first_statement():
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert text.splitlines()[:8] == [
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team",
        "# SPDX-License-Identifier: Apache-2.0",
        "# The accessors follow SGLang's argument, engine and tokenizer interfaces at v0.5.22 and v0.5.23.",
        "# See THIRD_PARTY_NOTICES.md.",
        f"# Source: https://github.com/sgl-project/sglang/tree/{SGLANG_22}",
        f"# Source: https://github.com/sgl-project/sglang/tree/{SGLANG_23}",
        "",
    ]
    tree = ast.parse(text)
    docstring = ast.get_docstring(tree)
    assert isinstance(tree.body[1], ast.ImportFrom) and tree.body[1].module == "__future__"
    for required in ("v0.5.22", "v0.5.23", SGLANG_22, SGLANG_23, "sgl-project/sglang", "Apache-2.0"):
        assert required in docstring, required
    flat = " ".join(docstring.split())
    assert (
        "No upstream function or block of source is copied. What follows SGLang and Dynamo is named in the "
        "header of this file and recorded in THIRD_PARTY_NOTICES.md."
    ) in flat


def test_the_test_file_has_the_same_source_lines_as_the_module():
    module_lines = MODULE_PATH.read_text(encoding="utf-8").splitlines()[:8]
    test_lines = TEST_PATH.read_text(encoding="utf-8").splitlines()[:8]
    assert test_lines[:3] == module_lines[:3]
    assert test_lines[4:] == module_lines[4:]


@pytest.mark.parametrize("path", [MODULE_PATH, TEST_PATH], ids=["module", "test"])
def test_files_are_ascii_within_120_columns_and_cite_only_what_the_repository_holds(path):
    text = path.read_text(encoding="utf-8")
    assert text.isascii()
    assert max(len(line) for line in text.splitlines()) <= 120
    # Working notes that the repository does not hold must not be cited.
    patterns = (
        r"phase\s*[0-9]",
        r"the\s+design\b",
        r"the\s+contract\b",
        r"\b(engine|scheduler|rejections|gate|vllm|spec)\.md\b",
        r"design_extrac[t]",
    )
    for pattern in patterns:
        assert not re.search(pattern, text, flags=re.IGNORECASE), pattern
    # Every evidence folder that is cited exists in the repository.
    repository = Path(__file__).resolve().parents[5]
    cited = set(re.findall(r"benchmarks/evidence/collector/[A-Za-z0-9_.-]+", text))
    for folder in cited:
        assert (repository / folder.rstrip(".")).is_dir(), folder
    if path == MODULE_PATH:
        assert cited == {"benchmarks/evidence/collector/sglbench-engine-probe-20261113"}


def test_a_chunk_without_id_in_its_meta_info_fails_the_call(loop):
    (rid,) = _rids(1)
    final = _chunk(0, rid, completion=2, finish=_length(2))
    del final["meta_info"]["id"]
    _fails(_FakeStreamEngine(loop, [final]), 1, "carries the id None")


@pytest.mark.parametrize("index", [1, -1, True, None])
def test_a_wrong_index_fails_a_batch_of_one_too(loop, index):
    (rid,) = _rids(1)
    chunk = _chunk(0, rid, completion=2, finish=_length(2))
    chunk["index"] = index
    _fails(_FakeStreamEngine(loop, [chunk]), 1, "has index")
