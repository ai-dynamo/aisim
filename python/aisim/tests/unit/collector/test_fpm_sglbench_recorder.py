# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# The fake scheduler, batch, result and hook registry follow what SGLang's Scheduler sets and calls at v0.5.22
# and v0.5.23. See THIRD_PARTY_NOTICES.md.
# Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d
# Source: https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90

"""CPU-only checks of ``runtime/sglbench_recorder.py``. No SGLang, no GPU and no engine are involved.

Everything that stands in for SGLang is FAKE and is named so:

- ``FakeSglang`` builds a FAKE ``sglang`` package tree with only the names the recorder imports. The fixture
  ``fake_sglang`` places it in ``sys.modules`` and pytest's ``monkeypatch`` takes it out again.
- ``FakeSchedulerBase`` holds the attributes the recorder reads from SGLang's ``Scheduler`` and the three
  methods it wraps. ``FakeBatch``, ``FakeRequest``, ``FakeResult`` and ``FakeCopyDone`` carry the fields of
  SGLang's batch, request and result objects that the recorder reads. The names and the places that set them
  were read in the SGLang source at both tags; the module under test names the files and symbols.
- The dtype of the FAKE model runner's KV pool is FAKE (``_fake_torch_dtype``): an object of a class named
  ``dtype`` in a module named ``torch`` whose text is ``torch.<name>``. No torch is involved, and what a real
  torch dtype looks like to the recorder was not run here.
- The loops in the tests call ``run_batch`` and ``process_batch_result`` in the order of SGLang's
  ``Scheduler.event_loop_normal`` and ``Scheduler.event_loop_overlap``. They are test code, not SGLang.
- Every time value is FAKE. ``FakeClock`` hands out numbers that grow by a fixed step; none was measured.
- The six limits of the FAKE scheduler are the numbers that the schedulers of one probe run reported
  (benchmarks/evidence/collector/sglbench-engine-probe-20261113, the headers of ``v0522-q06-p64``). They are
  typed in to give the header realistic values. The graph sizes are typed in as well; that folder records none.

The names of the header and of the batch line are typed here on their own, not taken from the module, because a
reader of the log is written against the same text and must find exactly these names.
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import enum
import json
import os
import pickle
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
MODULE_PATH = RUNTIME / "sglbench_recorder.py"

# In the pod the runtime files sit flat in one directory and import each other by their flat names. The
# directory goes to the END of sys.path, so that none of its files can shadow an installed module.
if str(RUNTIME) not in sys.path:
    sys.path.append(str(RUNTIME))

import sglbench_recorder as recorder

pytestmark = pytest.mark.unit

ENV = "AISIM_SGLBENCH_RECORDER_DIR"
CLASS_NAME = "SglbenchRecorderScheduler"

HEADER_KEYS = [
    "type",
    "format",
    "pid",
    "class",
    "mro",
    "sglang_version",
    "leader",
    "ranks",
    "limits",
    "decode_graph_sizes",
    "prefill_graph_sizes",
    "enable_overlap",
    "kv_cache_stored_dtype",
    "checks",
]
BATCH_KEYS = [
    "type",
    "seq",
    "forward_iter",
    "mode",
    "launch_ts",
    "run_return_ts",
    "out_ts",
    "copy_done",
    "rids",
    "extend_num_tokens",
    "prefix_lens",
    "extend_lens",
    "global_num_tokens",
]

FAKE_SGLANG_VERSION = "0.5.23"
FAKE_PORT_ARGS = "FAKE port args"
FAKE_CLOCK_START = 1000.0
FAKE_CLOCK_STEP = 0.125
FAKE_LIMITS = {
    "page_size": 64,
    "max_req_input_len": 40954,
    "max_total_num_tokens": 1165312,
    "max_running_requests": 4096,
    "max_prefill_tokens": 16384,
    "chunked_prefill_size": 8192,
}
FAKE_DECODE_GRAPH_SIZES = [1, 2, 4, 8]
FAKE_PREFILL_GRAPH_SIZES = [16, 32, 64]
FAKE_RANKS = {
    "tp_size": 1,
    "pp_size": 1,
    "attn_dp_size": 1,
    "attn_dp_enabled": False,
    "attn_tp_rank": 0,
    "attn_cp_rank": 0,
    "attn_dp_rank": 0,
    "pp_rank": 0,
}
# The keys of the ranks and of the limits of a header, in the order they are typed in above.
RANK_KEYS = list(FAKE_RANKS)
LIMIT_KEYS = list(FAKE_LIMITS)

# Request ids in the shape the benchmark's content module gives them. The text is typed in.
RID_A = "sglb-fakerun1-p3-meas0-r0-s0-e"
RID_B = "sglb-fakerun1-p3-meas0-r0-s1-e"
RID_C = "sglb-fakerun1-p3-meas0-r0-s2-e"


# ---------------------------------------------------------------------------------------------------------------
# FAKE SGLang objects
# ---------------------------------------------------------------------------------------------------------------


class FakeForwardMode(enum.IntEnum):
    """FAKE of SGLang's ForwardMode: an IntEnum whose member names the recorder logs."""

    EXTEND = 1
    DECODE = 2
    IDLE = 4


class FakeRequest:
    """FAKE of SGLang's Req: only ``rid``."""

    def __init__(self, rid):
        self.rid = rid


@dataclasses.dataclass
class FakeBatch:
    """FAKE of SGLang's ScheduleBatch with the fields the recorder reads, and with its ``copy``."""

    reqs: list
    forward_mode: FakeForwardMode
    forward_iter: int | None = None
    launch_ts: float | None = None
    extend_num_tokens: int | None = None
    prefix_lens: list | None = None
    extend_lens: list | None = None
    # None, as the dataclass default of SGLang's field, until a sync step has bound a list to it.
    global_num_tokens: list | None = None

    def copy(self):
        # As ScheduleBatch.copy: a new object, a copy of the list reqs, the other fields as they are now.
        return FakeBatch(
            reqs=self.reqs[:],
            forward_mode=self.forward_mode,
            forward_iter=self.forward_iter,
            launch_ts=self.launch_ts,
            extend_num_tokens=self.extend_num_tokens,
            prefix_lens=self.prefix_lens,
            extend_lens=self.extend_lens,
            global_num_tokens=self.global_num_tokens,
        )


class FakeCopyDone:
    """FAKE of the copy event of a result: ``synchronize`` notes that it was called and may raise."""

    def __init__(self, events, error=None):
        self._events = events
        self.error = error

    def synchronize(self):
        self._events.append("synchronize")
        if self.error is not None:
            raise self.error


class FakeResult:
    """FAKE of SGLang's GenerationBatchResult: only ``copy_done``."""

    def __init__(self, copy_done):
        self.copy_done = copy_done


class FakeClock:
    """FAKE monotonic clock: every reading is one fixed step after the one before. Nothing here was measured.

    The FAKE stock ``run_batch`` reads it for ``launch_ts`` (label ``stock_launch_ts``). The module under test
    reads it through ``monotonic``, because the fixture puts this object in the place of its ``time`` module.
    """

    def __init__(self, events):
        self._events = events
        self._now = FAKE_CLOCK_START
        self.recorder_error = None

    def read(self, label):
        self._now += FAKE_CLOCK_STEP
        self._events.append((label, self._now))
        return self._now

    def monotonic(self):
        if self.recorder_error is not None:
            error, self.recorder_error = self.recorder_error, None
            self._events.append("recorder_clock_raises")
            raise error
        return self.read("recorder_clock")


class FakeGraphRunner:
    """FAKE of a captured-graph runner: it has the attributes it is given and no others."""

    def __init__(self, **attributes):
        for name, value in attributes.items():
            setattr(self, name, value)


def _fake_torch_dtype(name):
    """A FAKE of a torch dtype: an object of a class ``dtype`` of a module ``torch`` with the text ``torch.<name>``.

    The recorder imports no torch and tells a dtype by the name of its class. A real ``torch.bfloat16`` is an
    object of ``torch.dtype`` and its text is ``"torch.bfloat16"``; SGLang's own code relies on that text
    (``str(dtype).removeprefix("torch.")``). Neither was run here: no torch is installed for these tests.
    """
    fake_class = type("dtype", (), {"__module__": "torch", "__str__": lambda self: f"torch.{name}"})
    return fake_class()


# The dtype of the FAKE KV pool: bfloat16, what SGLang stores for ``auto`` and a bfloat16 model.
FAKE_KV_CACHE_DTYPE = _fake_torch_dtype("bfloat16")
FAKE_KV_CACHE_DTYPE_TEXT = "torch.bfloat16"


class FakeModelRunner:
    """FAKE of SGLang's ModelRunner: the two runner attributes and the two attributes of the KV-cache dtype.

    ``kv_cache_dtype_str`` is there because SGLang's runner has it beside ``kv_cache_dtype``; the recorder
    must not read it, and its FAKE value would show in the header if it did.
    """

    def __init__(self, decode_runner, prefill_runner, kv_cache_dtype=FAKE_KV_CACHE_DTYPE):
        self.decode_cuda_graph_runner = decode_runner
        self.prefill_cuda_graph_runner = prefill_runner
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_cache_dtype_str = "FAKE: the string SGLang keeps beside the dtype"


class FakeTpWorker:
    """FAKE of SGLang's TpModelWorker: ``model_runner``."""

    def __init__(self, model_runner):
        self.model_runner = model_runner


def _fake_tp_worker(
    decode_runner="default",
    prefill_runner="default",
    kv_cache_dtype=FAKE_KV_CACHE_DTYPE,
):
    if decode_runner == "default":
        decode_runner = FakeGraphRunner(capture_bs=list(FAKE_DECODE_GRAPH_SIZES))
    if prefill_runner == "default":
        prefill_runner = FakeGraphRunner(capture_num_tokens=list(FAKE_PREFILL_GRAPH_SIZES))
    return FakeTpWorker(FakeModelRunner(decode_runner, prefill_runner, kv_cache_dtype))


def _fake_scheduler_attributes():
    return {
        **FAKE_LIMITS,
        "max_queued_requests": None,
        "enable_overlap": True,
        "recv_from_tokenizer": object(),
        "tp_worker": _fake_tp_worker(),
        "disable_radix_cache": False,
        "is_hybrid_swa": False,
        "is_hybrid_ssm": False,
        "max_new_tokens_limit": None,
        "schedule_policy": "fcfs",
        "enable_priority_scheduling": False,
        "is_mixed_chunk": False,
        "prefill_decode_interval": 0,
        "prefill_delayer": None,
        "min_free_slots_delayer": None,
        "recv_skipper": None,
        "input_blocker": None,
    }


class FakeServerArgs:
    """What a test hands to the FAKE scheduler constructor in the place of SGLang's server arguments."""

    def __init__(self, world, overrides=None, missing=(), init_error=None):
        self.world = world
        self.overrides = dict(overrides or {})
        self.missing = tuple(missing)
        self.init_error = init_error


class FakeSchedulerBase:
    """FAKE of SGLang's Scheduler: the attributes the recorder reads and the three methods it wraps.

    ``run_batch`` stamps ``forward_iter`` and ``launch_ts`` on the batch as the stock method does.
    ``process_batch_result`` returns a marker object, so that a test can see that the value is passed on; the
    stock method of SGLang returns None.
    """

    def __init__(self, server_args, port_args):
        world = server_args.world
        self.world = world
        world.events.append(("stock_init", sorted(path.name for path in world.directory.glob("*"))))
        if server_args.init_error is not None:
            raise server_args.init_error
        self.server_args = server_args
        self.port_args = port_args
        self.forward_ct = 0
        for name, value in _fake_scheduler_attributes().items():
            setattr(self, name, value)
        for name, value in server_args.overrides.items():
            setattr(self, name, value)
        for name in server_args.missing:
            delattr(self, name)

    def run_batch(self, batch, pp_proxy_tensors=None):
        world = self.world
        if world.stock_run_error is not None:
            world.events.append("stock_run_batch_raises")
            raise world.stock_run_error
        if world.stock_stamps_batch:
            self.forward_ct += 1
            batch.forward_iter = self.forward_ct
            batch.launch_ts = world.clock.read("stock_launch_ts")
        result = world.results.popleft() if world.results else FakeResult(FakeCopyDone(world.events))
        world.stock_calls.append(("run_batch", batch, pp_proxy_tensors, result))
        world.events.append(("stock_run_batch", batch.forward_iter))
        return result

    def process_batch_result(self, batch, result):
        world = self.world
        world.stock_calls.append(("process_batch_result", batch, result))
        kinds = [record["type"] for record in world.records()]
        world.events.append(("stock_process_batch_result", kinds.count("batch"), kinds.count("fault")))
        if world.stock_process_error is not None:
            raise world.stock_process_error
        return world.stock_process_return


class FakeParallel:
    """FAKE of the object ``get_parallel()`` returns. A name in ``failures`` raises the error stored for it."""

    def __init__(self):
        self.values = {**FAKE_RANKS, "pp_max_micro_batch_size": FAKE_LIMITS["max_running_requests"]}
        self.failures = {}

    def __getattr__(self, name):
        if name in ("values", "failures"):
            raise AttributeError(name)
        if name in self.failures:
            raise self.failures[name]
        if name in self.values:
            return self.values[name]
        raise AttributeError(f"FAKE parallel context has no {name!r}")


class FakeHookType(enum.Enum):
    """FAKE of SGLang's HookType."""

    BEFORE = "before"
    AFTER = "after"
    AROUND = "around"
    REPLACE = "replace"


class FakeSglang:
    """A FAKE ``sglang`` package tree that holds only the names ``sglbench_recorder`` imports.

    The stock class is made new for every tree and is held by the FAKE scheduler module and by this object
    only. ``apply_hooks`` rebinds the name in that module and then every module-level name in ``sys.modules``
    whose value is the stock class, which is what SGLang's registry does for a class hook.
    """

    def __init__(self):
        self.events = []
        self.hooks = []
        self.entry_calls = []
        self.built = []
        self.entry_return = object()
        self.parallel = FakeParallel()
        self.stock = type("Scheduler", (FakeSchedulerBase,), {})

        self.root = self._module("sglang", package=True)
        self.root.__version__ = FAKE_SGLANG_VERSION
        self.srt = self._module("sglang.srt", package=True)
        self.managers = self._module("sglang.srt.managers", package=True)
        self.plugins = self._module("sglang.srt.plugins", package=True)
        self.scheduler_module = self._module("sglang.srt.managers.scheduler")
        self.policy_module = self._module("sglang.srt.managers.schedule_policy")
        self.runtime_context = self._module("sglang.srt.runtime_context")
        self.hook_registry = self._module("sglang.srt.plugins.hook_registry")
        self.root.srt = self.srt
        self.srt.managers = self.managers
        self.srt.plugins = self.plugins
        self.srt.runtime_context = self.runtime_context
        self.managers.scheduler = self.scheduler_module
        self.managers.schedule_policy = self.policy_module
        self.plugins.hook_registry = self.hook_registry

        self.scheduler_module.Scheduler = self.stock
        self.scheduler_module.run_scheduler_process = self._stock_entry()
        self.policy_module._IS_HIP = False
        self.runtime_context.get_parallel = self._get_parallel
        self.hook_registry.HookType = FakeHookType
        self.hook_registry.HookRegistry = self._registry()

    @property
    def modules(self):
        return {
            module.__name__: module
            for module in (
                self.root,
                self.srt,
                self.managers,
                self.plugins,
                self.scheduler_module,
                self.policy_module,
                self.runtime_context,
                self.hook_registry,
            )
        }

    @staticmethod
    def _module(name, package=False):
        module = types.ModuleType(name)
        if package:
            module.__path__ = []
        return module

    def _get_parallel(self):
        return self.parallel

    def _registry(self):
        tree = self

        class FakeHookRegistry:
            """FAKE of SGLang's HookRegistry: ``register`` stores a hook and refuses a class without REPLACE."""

            @classmethod
            def register(cls, target, hook, hook_type=FakeHookType.AFTER, *, source=None):
                if isinstance(hook, type) and hook_type != FakeHookType.REPLACE:
                    raise TypeError("FAKE registry: a class needs REPLACE")
                tree.hooks.append((target, hook, hook_type))
                tree.events.append("register")

        return FakeHookRegistry

    def apply_hooks(self):
        for target, hook, _hook_type in self.hooks:
            module_name, _, attribute = target.rpartition(".")
            owner = sys.modules[module_name]
            original = getattr(owner, attribute)
            setattr(owner, attribute, hook)
            for module in list(sys.modules.values()):
                if not isinstance(module, types.ModuleType) or module is owner:
                    continue
                for name, value in list(vars(module).items()):
                    if value is original:
                        setattr(module, name, hook)

    def _stock_entry(self):
        tree = self

        def run_scheduler_process(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            pp_rank,
            dp_rank,
            pipe_writer,
            display_tp_rank=None,
            display_dp_rank=None,
            display_moe_ep_rank=None,
        ):
            """FAKE stock entry: apply the hooks first, then build the class the module name holds now."""
            tree.events.append("stock_entry")
            tree.entry_calls.append(
                (
                    server_args,
                    port_args,
                    gpu_id,
                    tp_rank,
                    pp_rank,
                    dp_rank,
                    pipe_writer,
                    display_tp_rank,
                    display_dp_rank,
                    display_moe_ep_rank,
                )
            )
            tree.apply_hooks()
            tree.built.append(tree.scheduler_module.Scheduler(server_args, port_args))
            return tree.entry_return

        return run_scheduler_process


class FakeWorld:
    """What one test shares with its FAKE scheduler: the events, the FAKE clock and the recorder directory."""

    def __init__(self, directory, sglang):
        self.directory = directory
        self.sglang = sglang
        self.events = sglang.events
        self.clock = FakeClock(self.events)
        self.results = collections.deque()
        self.stock_calls = []
        self.stock_run_error = None
        self.stock_process_error = None
        self.stock_process_return = object()
        self.stock_stamps_batch = True

    @property
    def log_path(self):
        return self.directory / f"sched-{os.getpid()}.jsonl"

    def raw_lines(self):
        if not self.log_path.exists():
            return []
        text = self.log_path.read_text(encoding="utf-8")
        assert text == "" or text.endswith("\n"), "the log ends in the middle of a line"
        return text.split("\n")[:-1]

    def records(self):
        return [json.loads(line) for line in self.raw_lines()]

    def lines_of(self, kind):
        return [record for record in self.records() if record["type"] == kind]


@pytest.fixture
def fake_sglang(monkeypatch):
    """A FAKE ``sglang`` package in ``sys.modules`` for one test. monkeypatch removes it again afterwards."""
    tree = FakeSglang()
    for name, module in tree.modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return tree


@pytest.fixture
def world(fake_sglang, tmp_path, monkeypatch):
    """The recorder directory of one FAKE run, the variable that names it, and the FAKE clock in the module."""
    directory = tmp_path / "sglbench-recorder" / "fake-run-id"
    directory.mkdir(parents=True)
    monkeypatch.setenv(ENV, str(directory))
    fake_world = FakeWorld(directory, fake_sglang)
    monkeypatch.setattr(recorder, "time", fake_world.clock)
    return fake_world


def _scheduler(world, **options):
    """Build the recorder class on the FAKE stock class and make one scheduler of it."""
    cls = recorder.make_recorder_class(world.sglang.stock)
    return cls(FakeServerArgs(world, **options), FAKE_PORT_ARGS)


def _header_of(world, **options):
    _scheduler(world, **options)
    records = world.records()
    assert [record["type"] for record in records] == ["header"]
    return records[0]


def _extend_batch():
    rids = (RID_A, RID_B)
    return FakeBatch(
        reqs=[FakeRequest(rid) for rid in rids],
        forward_mode=FakeForwardMode.EXTEND,
        extend_num_tokens=64 * len(rids),
        prefix_lens=[0] * len(rids),
        extend_lens=[64] * len(rids),
    )


def _expected_header():
    return {
        "type": "header",
        "format": 3,
        "pid": os.getpid(),
        "class": CLASS_NAME,
        "mro": [CLASS_NAME, "Scheduler", "FakeSchedulerBase"],
        "sglang_version": FAKE_SGLANG_VERSION,
        "leader": True,
        "ranks": dict(FAKE_RANKS),
        "limits": dict(FAKE_LIMITS),
        "decode_graph_sizes": list(FAKE_DECODE_GRAPH_SIZES),
        "prefill_graph_sizes": list(FAKE_PREFILL_GRAPH_SIZES),
        "enable_overlap": True,
        "kv_cache_stored_dtype": FAKE_KV_CACHE_DTYPE_TEXT,
        "checks": {
            "disable_radix_cache": False,
            "is_hybrid_swa": False,
            "is_hybrid_ssm": False,
            "max_queued_requests": None,
            "max_new_tokens_limit": None,
            "schedule_policy": "fcfs",
            "enable_priority_scheduling": False,
            "is_mixed_chunk": False,
            "prefill_decode_interval": 0,
            "prefill_delayer": False,
            "min_free_slots_delayer": False,
            "recv_skipper": False,
            "input_blocker": False,
            "pp_max_micro_batch_size": 4096,
            "is_hip": False,
        },
    }


# The keys of the checks of a header, in the order they are typed in above.
CHECK_KEYS = list(_expected_header()["checks"])


def _assert_unreadable(value, error_type, *needles):
    assert isinstance(value, dict), value
    assert list(value) == ["unreadable"], value
    text = value["unreadable"]
    assert isinstance(text, str), value
    assert text.startswith(error_type + "("), text
    for needle in needles:
        assert needle in text, text


def _fake_time(reading):
    """The FAKE clock value of its n-th reading, counted from 1."""
    return FAKE_CLOCK_START + reading * FAKE_CLOCK_STEP


# ---------------------------------------------------------------------------------------------------------------
# The seam rows
# ---------------------------------------------------------------------------------------------------------------


# The two releases the module was read against; its header names their commits.
SEAM_VERSIONS = ("0.5.22", "0.5.23")

# The seam rows of the module itself, the rows the preflight looks up in the image. No copy is typed here.
SEAM_NAMES = recorder.SGLANG_SEAM_NAMES


def _seam_members(owner):
    return [name.partition(".")[2] for _module, name, _kind in SEAM_NAMES if name.startswith(owner + ".")]


def test_seam_rows_name_everything_the_module_takes_from_sglang():
    import sglbench_driver as driver  # the reader of the log, for its three mode names

    reached = sglang_names_reached_by_import(ast.parse(MODULE_PATH.read_text(encoding="utf-8")))
    # An import or a read that gets no row is noticed here, so the list cannot fall behind the code.
    assert seam_rows_missing_for(reached, recorder.SGLANG_SEAM_NAMES) == []
    # What no import shows: the fields read from the batch, and the mode names that are logged. The FAKE batch
    # of this file has exactly the fields the recorder reads, and a recorder that read one more would fail the
    # batch tests until the FAKE had it.
    assert sorted(_seam_members("ScheduleBatch")) == sorted(field.name for field in dataclasses.fields(FakeBatch))
    assert _seam_members("ForwardMode") == [driver.MODE_EXTEND, driver.MODE_DECODE, driver.MODE_IDLE]


def test_every_seam_name_is_in_the_sglang_source_at_both_tags():
    # Skipped where no SGLang source is at hand. With it, each row is looked up in the parsed files of both
    # releases; nothing is imported.
    for version, python_root in sglang_source_roots(SEAM_VERSIONS).items():
        assert seam_rows_not_in_source(SEAM_NAMES, python_root) == [], version
        # The look-up notices a field the source does not have, and a field is not found as an attribute.
        module = "sglang.srt.managers.schedule_batch"
        assert seam_rows_not_in_source(((module, "ScheduleBatch.no_such_field", "field"),), python_root)
        assert seam_rows_not_in_source(((module, "ScheduleBatch.reqs", "attribute"),), python_root)


# ---------------------------------------------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------------------------------------------


def test_header_is_the_first_line_and_holds_every_value(world):
    header = _header_of(world)
    assert header == _expected_header()
    assert list(header) == HEADER_KEYS
    assert list(header["ranks"]) == RANK_KEYS
    assert list(header["limits"]) == LIMIT_KEYS
    assert list(header["checks"]) == CHECK_KEYS
    assert header["class"] == recorder.RECORDER_CLASS_NAME
    assert header["format"] == recorder.RECORD_FORMAT
    assert len(header["mro"]) == 3


def test_header_names_the_class_of_the_object_and_the_version_sglang_reports(world):
    # A class built on top of the recorder class is reported under its own name, so the header shows what was
    # built and not what was meant.
    cls = recorder.make_recorder_class(world.sglang.stock)
    further = type("FakeFurtherScheduler", (cls,), {})
    world.sglang.root.__version__ = "0.5.22"
    further(FakeServerArgs(world), FAKE_PORT_ARGS)
    (header,) = world.records()
    assert header["class"] == "FakeFurtherScheduler"
    assert header["mro"] == ["FakeFurtherScheduler", CLASS_NAME, "Scheduler"]
    assert header["sglang_version"] == "0.5.22"


def test_stock_constructor_gets_its_arguments_unchanged_and_its_error_is_not_caught(world):
    cls = recorder.make_recorder_class(world.sglang.stock)
    server_args = FakeServerArgs(world)
    scheduler = cls(server_args, port_args=FAKE_PORT_ARGS)
    assert scheduler.server_args is server_args
    assert scheduler.port_args == FAKE_PORT_ARGS

    error = RuntimeError("FAKE failure of the stock constructor")
    with pytest.raises(RuntimeError) as caught:
        cls(FakeServerArgs(world, init_error=error), FAKE_PORT_ARGS)
    assert caught.value is error


@pytest.mark.parametrize(
    ("socket", "leader"),
    [
        # The answer is "is not None" on the socket attribute, not a truth test of it.
        pytest.param(None, False, id="rank-zero-without-socket"),
        pytest.param(0, True, id="socket-that-is-false-in-a-truth-test"),
    ],
)
def test_leader_says_whether_the_process_holds_the_request_socket(world, socket, leader):
    header = _header_of(world, overrides={"recv_from_tokenizer": socket})
    assert header["leader"] is leader
    assert header["ranks"] == FAKE_RANKS


def test_checks_report_objects_as_present_and_values_as_they_are(world):
    # Two of the four objects are false in a truth test: the header asks "is not None", not "is true".
    objects = {"prefill_delayer": object(), "min_free_slots_delayer": 0, "recv_skipper": object(), "input_blocker": []}
    values = {
        "max_queued_requests": 512,
        "max_new_tokens_limit": 4,
        "schedule_policy": "lpm",
        "prefill_decode_interval": 22,
        "disable_radix_cache": True,
        "is_hybrid_swa": True,
        "is_hybrid_ssm": True,
        "enable_priority_scheduling": True,
        "is_mixed_chunk": True,
    }
    world.sglang.policy_module._IS_HIP = True
    world.sglang.parallel.values["pp_max_micro_batch_size"] = 7
    header = _header_of(world, overrides={**objects, **values})
    assert header["checks"] == {
        **_expected_header()["checks"],
        **values,
        **dict.fromkeys(objects, True),
        "pp_max_micro_batch_size": 7,
        "is_hip": True,
    }


@pytest.mark.parametrize(
    ("decode_runner", "prefill_runner", "decode_sizes", "prefill_sizes"),
    [
        pytest.param(None, "default", [], FAKE_PREFILL_GRAPH_SIZES, id="decode-runner-none"),
        pytest.param("default", None, FAKE_DECODE_GRAPH_SIZES, [], id="prefill-runner-none"),
        pytest.param("default", FakeGraphRunner(), FAKE_DECODE_GRAPH_SIZES, [], id="eager-prefill-runner"),
    ],
)
def test_graph_sizes_of_both_runners(world, decode_runner, prefill_runner, decode_sizes, prefill_sizes):
    header = _header_of(world, overrides={"tp_worker": _fake_tp_worker(decode_runner, prefill_runner)})
    assert header["decode_graph_sizes"] == decode_sizes
    assert header["prefill_graph_sizes"] == prefill_sizes
    rest = _expected_header()
    for key in ("decode_graph_sizes", "prefill_graph_sizes"):
        del header[key], rest[key]
    assert header == rest


@pytest.mark.parametrize(
    ("tp_worker", "decode_error", "prefill_error", "dtype_error"),
    [
        # Without a worker nothing behind it can be read: both graph lists and the dtype.
        pytest.param(None, "AttributeError", "AttributeError", "AttributeError", id="tp-worker-none"),
        pytest.param(_fake_tp_worker(FakeGraphRunner()), "AttributeError", None, None, id="decode-runner-without-list"),
        pytest.param(_fake_tp_worker(FakeGraphRunner(capture_bs=[1, True])), "TypeError", None, None, id="decode-bool"),
    ],
)
def test_graph_sizes_that_cannot_be_read_are_marked_and_never_guessed(
    world, tp_worker, decode_error, prefill_error, dtype_error
):
    header = _header_of(world, overrides={"tp_worker": tp_worker})
    rest = _expected_header()
    errors = (
        ("decode_graph_sizes", decode_error),
        ("prefill_graph_sizes", prefill_error),
        ("kv_cache_stored_dtype", dtype_error),
    )
    for key, error in errors:
        if error is not None:
            _assert_unreadable(header[key], error)
            del header[key], rest[key]
    assert header == rest


@pytest.mark.parametrize("name", ["float8_e4m3fn"])
def test_header_holds_the_text_of_the_dtype_the_model_runner_built_its_kv_pool_with(world, name):
    header = _header_of(world, overrides={"tp_worker": _fake_tp_worker(kv_cache_dtype=_fake_torch_dtype(name))})
    assert header["kv_cache_stored_dtype"] == f"torch.{name}"
    # Only that key follows the dtype.
    rest = _expected_header()
    del header["kv_cache_stored_dtype"], rest["kv_cache_stored_dtype"]
    assert header == rest


@pytest.mark.parametrize(
    ("value", "needle"),
    [
        # The string SGLang keeps beside the dtype is no dtype.
        pytest.param("auto", "is a builtins.str, not a torch.dtype", id="the-argument"),
        pytest.param(None, "is a builtins.NoneType, not a torch.dtype", id="none"),
    ],
)
def test_a_kv_cache_dtype_that_is_no_torch_dtype_is_marked_and_never_written_as_text(world, value, needle):
    header = _header_of(world, overrides={"tp_worker": _fake_tp_worker(kv_cache_dtype=value)})
    _assert_unreadable(header["kv_cache_stored_dtype"], "TypeError", "model_runner.kv_cache_dtype", needle)
    rest = _expected_header()
    del header["kv_cache_stored_dtype"], rest["kv_cache_stored_dtype"]
    assert header == rest


@pytest.mark.parametrize("name", ["attn_cp_rank"])
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("FAKE: the rank has not been written in this process"),
        ValueError("FAKE: config namespace 'parallel' not published"),
        AttributeError("FAKE: no such name"),
    ],
    ids=["runtime-error", "value-error", "attribute-error"],
)
def test_one_rank_that_cannot_be_read_marks_that_name_only(world, name, error):
    world.sglang.parallel.failures[name] = error
    header = _header_of(world)
    assert header["ranks"][name] == {"unreadable": repr(error)}
    expected = _expected_header()
    expected["ranks"][name] = {"unreadable": repr(error)}
    assert header == expected


@pytest.mark.parametrize("name", ["chunked_prefill_size", "page_size"])
def test_one_limit_that_cannot_be_read_marks_that_name_only(world, name):
    header = _header_of(world, missing=(name,))
    _assert_unreadable(header["limits"][name], "AttributeError", name)
    expected = _expected_header()
    del header["limits"][name], expected["limits"][name]
    assert header == expected


@pytest.mark.parametrize("name", ["max_queued_requests", "prefill_delayer"])
def test_one_check_that_cannot_be_read_marks_that_name_only(world, name):
    header = _header_of(world, missing=(name,))
    _assert_unreadable(header["checks"][name], "AttributeError", name)
    assert list(header["checks"]) == CHECK_KEYS
    expected = _expected_header()
    del header["checks"][name], expected["checks"][name]
    assert header == expected


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        pytest.param(object(), "TypeError", id="object"),
        pytest.param(float("nan"), "ValueError", id="nan"),
        pytest.param([1, object()], "TypeError", id="list-with-object"),
    ],
)
def test_value_that_json_cannot_hold_is_marked(world, value, error_type):
    header = _header_of(world, overrides={"page_size": value, "schedule_policy": value})
    _assert_unreadable(header["limits"]["page_size"], error_type, "page_size")
    _assert_unreadable(header["checks"]["schedule_policy"], error_type, "schedule_policy")


@pytest.mark.parametrize(
    ("options", "error_type"),
    [
        pytest.param({"missing": ("enable_overlap",)}, "AttributeError", id="missing"),
        pytest.param({"overrides": {"enable_overlap": 1}}, "TypeError", id="int"),
    ],
)
def test_overlap_flag_that_is_not_a_readable_bool_is_marked(world, options, error_type):
    header = _header_of(world, **options)
    _assert_unreadable(header["enable_overlap"], error_type, "enable_overlap")
    expected = _expected_header()
    del header["enable_overlap"], expected["enable_overlap"]
    assert header == expected


# ---------------------------------------------------------------------------------------------------------------
# Batch lines
# ---------------------------------------------------------------------------------------------------------------


def test_extend_batch_line(world):
    scheduler = _scheduler(world)
    batch = _extend_batch()
    result = scheduler.run_batch(batch)
    returned = scheduler.process_batch_result(batch, result)

    assert returned is world.stock_process_return
    (line,) = world.lines_of("batch")
    assert list(line) == BATCH_KEYS
    assert line == {
        "type": "batch",
        "seq": 1,
        "forward_iter": 1,
        "mode": "EXTEND",
        "launch_ts": _fake_time(1),
        "run_return_ts": _fake_time(2),
        "out_ts": _fake_time(3),
        "copy_done": True,
        "rids": [RID_A, RID_B],
        "extend_num_tokens": 128,
        "prefix_lens": [0, 0],
        "extend_lens": [64, 64],
        "global_num_tokens": None,
    }
    assert world.lines_of("fault") == []


def test_an_idle_batch_of_a_rank_without_work_gets_a_line_without_request_ids_and_no_fault(world):
    # FAKE of a scheduler process under DP attention: rank 1 of two, with an input blocker. SGLang gives a rank
    # that has nothing to run an idle batch in every step in which another rank runs one.
    world.sglang.parallel.values.update({"tp_size": 2, "attn_dp_size": 2, "attn_dp_enabled": True, "attn_dp_rank": 1})
    scheduler = _scheduler(world, overrides={"input_blocker": object()})

    # Step 1: the other rank extends alone. Step 2: both ranks extend. Step 3: the other rank decodes alone.
    steps = [
        FakeBatch(reqs=[], forward_mode=FakeForwardMode.IDLE, global_num_tokens=[128, 0]),
        dataclasses.replace(_extend_batch(), global_num_tokens=[128, 128]),
        FakeBatch(reqs=[], forward_mode=FakeForwardMode.IDLE, global_num_tokens=[2, 0]),
    ]
    for batch in steps:
        assert scheduler.process_batch_result(batch, scheduler.run_batch(batch)) is world.stock_process_return

    lines = world.lines_of("batch")
    assert [list(line) for line in lines] == [BATCH_KEYS] * 3
    assert [(line["seq"], line["forward_iter"], line["mode"], line["rids"]) for line in lines] == [
        (1, 1, "IDLE", []),
        (2, 2, "EXTEND", [RID_A, RID_B]),
        (3, 3, "IDLE", []),
    ]
    # An idle step counts as a step, and its line holds what every rank ran in it.
    assert [line["global_num_tokens"] for line in lines] == [[128, 0], [128, 128], [2, 0]]
    idle = lines[0]
    assert (idle["extend_num_tokens"], idle["prefix_lens"], idle["extend_lens"]) == (None, None, None)
    assert type(idle["launch_ts"]) is type(idle["out_ts"]) is float
    assert [line["copy_done"] for line in lines] == [True] * 3
    assert world.lines_of("fault") == []
    # The stock methods got every batch, the idle ones included.
    assert [call[0] for call in world.stock_calls] == ["run_batch", "process_batch_result"] * 3


def test_overlap_order_joins_by_forward_iter_and_reads_the_copy(world):
    scheduler = _scheduler(world)
    queue = collections.deque()
    live = _extend_batch()
    # FAKE: the sync step of a two-rank engine bound the counts of both ranks before step 1 ran.
    live.global_num_tokens = [128, 128]

    # As Scheduler.event_loop_overlap. Pass 1: run step 1 and queue a copy of the batch with its result.
    result_1 = scheduler.run_batch(live)
    queue.append((live.copy(), result_1))
    # The live object becomes the batch of the next step. Its request list is rebound with a third request,
    # as when SGLang merges a batch into the running one, and its extend fields are overwritten here, so that
    # the line of step 1 can only be right when it is built from the queued copy. The sync step of step 2
    # binds a new list of counts, as SGLang does: it never edits the list the copy still holds.
    live.forward_mode = FakeForwardMode.DECODE
    live.reqs = [*live.reqs, FakeRequest(RID_C)]
    live.extend_num_tokens = 3
    live.prefix_lens = [63, 63, 63]
    live.extend_lens = [1, 1, 1]
    live.global_num_tokens = [3, 3]
    # Pass 2: run step 2 first, then process the result of step 1.
    result_2 = scheduler.run_batch(live)
    queue.append((live.copy(), result_2))
    copy_1, queued_1 = queue.popleft()
    assert copy_1 is not live
    assert queued_1 is result_1
    returned_1 = scheduler.process_batch_result(copy_1, queued_1)
    # Pass 3: no new batch; process the result of step 2.
    copy_2, queued_2 = queue.popleft()
    returned_2 = scheduler.process_batch_result(copy_2, queued_2)

    assert returned_1 is world.stock_process_return
    assert returned_2 is world.stock_process_return
    assert world.events == [
        ("stock_init", []),
        ("stock_launch_ts", _fake_time(1)),
        ("stock_run_batch", 1),
        ("recorder_clock", _fake_time(2)),
        ("stock_launch_ts", _fake_time(3)),
        ("stock_run_batch", 2),
        ("recorder_clock", _fake_time(4)),
        "synchronize",
        ("recorder_clock", _fake_time(5)),
        ("stock_process_batch_result", 1, 0),
        "synchronize",
        ("recorder_clock", _fake_time(6)),
        ("stock_process_batch_result", 2, 0),
    ]
    assert world.lines_of("batch") == [
        {
            "type": "batch",
            "seq": 1,
            "forward_iter": 1,
            "mode": "EXTEND",
            "launch_ts": _fake_time(1),
            "run_return_ts": _fake_time(2),
            "out_ts": _fake_time(5),
            "copy_done": True,
            "rids": [RID_A, RID_B],
            "extend_num_tokens": 128,
            "prefix_lens": [0, 0],
            "extend_lens": [64, 64],
            "global_num_tokens": [128, 128],
        },
        {
            "type": "batch",
            "seq": 2,
            "forward_iter": 2,
            "mode": "DECODE",
            "launch_ts": _fake_time(3),
            "run_return_ts": _fake_time(4),
            "out_ts": _fake_time(6),
            "copy_done": True,
            "rids": [RID_A, RID_B, RID_C],
            "extend_num_tokens": None,
            "prefix_lens": None,
            "extend_lens": None,
            "global_num_tokens": [3, 3],
        },
    ]
    # Step 2 was launched before the output of step 1 was taken, and the live object says nothing about step 1.
    first, second = world.lines_of("batch")
    assert second["launch_ts"] < first["out_ts"]
    assert (live.forward_iter, live.forward_mode) == (2, FakeForwardMode.DECODE)
    assert world.stock_calls[2][1] is copy_1
    assert world.stock_calls[3][1] is copy_2


def test_copy_done_none_takes_the_time_without_a_wait(world):
    scheduler = _scheduler(world)
    world.results.append(FakeResult(None))
    batch = _extend_batch()
    result = scheduler.run_batch(batch)
    returned = scheduler.process_batch_result(batch, result)

    assert returned is world.stock_process_return
    assert "synchronize" not in world.events
    (line,) = world.lines_of("batch")
    assert line["copy_done"] is False
    assert line["out_ts"] == _fake_time(3)
    assert world.lines_of("fault") == []


# ---------------------------------------------------------------------------------------------------------------
# Faults
# ---------------------------------------------------------------------------------------------------------------


def _no_stamp(world, batch):
    world.stock_stamps_batch = False


@pytest.mark.parametrize(("spoil", "error_type"), [pytest.param(_no_stamp, "TypeError", id="forward-iter-none")])
def test_fault_in_the_run_batch_step_leaves_the_stock_result_untouched(world, spoil, error_type):
    scheduler = _scheduler(world)
    batch = _extend_batch()
    spoil(world, batch)
    marker = FakeResult(None)
    world.results.append(marker)

    returned = scheduler.run_batch(batch)

    assert returned is marker
    assert len(world.stock_calls) == 1
    assert world.stock_calls[0][:2] == ("run_batch", batch)
    assert world.stock_calls[0][1] is batch
    (fault,) = world.lines_of("fault")
    assert list(fault) == ["type", "where", "error"]
    assert fault["where"] == "run_batch"
    assert fault["error"].startswith(error_type + "(")
    assert world.lines_of("batch") == []


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        pytest.param("forward_iter", True, "TypeError", id="forward-iter-bool"),
        pytest.param("launch_ts", 1000, "TypeError", id="launch-ts-int"),
        pytest.param("launch_ts", float("nan"), "ValueError", id="launch-ts-nan"),
        pytest.param("reqs", [FakeRequest(RID_A), FakeRequest(None)], "TypeError", id="rid-none"),
        pytest.param("extend_num_tokens", True, "TypeError", id="extend-num-tokens-bool"),
        pytest.param("prefix_lens", None, "TypeError", id="prefix-lens-none"),
        pytest.param("global_num_tokens", [128.0, 128.0], "TypeError", id="global-num-tokens-float"),
        pytest.param("copy_done", "absent", "AttributeError", id="result-without-copy-done"),
        # The one fault with a real trigger: the wait on the copy event raises, as HostCopyDone.synchronize
        # does at v0.5.23 for a copy that failed.
        pytest.param("copy_done", "raises", "RuntimeError", id="synchronize-raises"),
        pytest.param("batch", "empty", "AttributeError", id="batch-without-fields"),
    ],
)
def test_fault_in_the_result_step_leaves_the_stock_call_and_its_value_untouched(world, field, value, error_type):
    scheduler = _scheduler(world)
    batch = _extend_batch()
    result = scheduler.run_batch(batch)
    if field == "copy_done" and value == "raises":
        result = FakeResult(FakeCopyDone(world.events, RuntimeError("FAKE: async device->host copy failed")))
    elif field == "copy_done":
        result = types.SimpleNamespace()
    elif field == "batch":
        batch = types.SimpleNamespace()
    else:
        setattr(batch, field, value)

    returned = scheduler.process_batch_result(batch, result)

    assert returned is world.stock_process_return
    assert world.stock_calls[-1][0] == "process_batch_result"
    assert world.stock_calls[-1][1] is batch
    assert world.stock_calls[-1][2] is result
    (fault,) = world.lines_of("fault")
    assert list(fault) == ["type", "where", "error"]
    assert fault["where"] == "process_batch_result"
    assert fault["error"].startswith(error_type + "("), fault
    assert world.lines_of("batch") == []
    # The stock method ran after the fault line was in the file.
    assert world.events[-1] == ("stock_process_batch_result", 0, 1)


def test_stock_run_batch_error_passes_through_unchanged(world):
    scheduler = _scheduler(world)
    error = RuntimeError("FAKE failure of the stock run_batch")
    world.stock_run_error = error
    with pytest.raises(RuntimeError) as caught:
        scheduler.run_batch(_extend_batch())
    assert caught.value is error
    # The recorder did nothing after the failed stock call: no clock reading and no line.
    assert world.events == [("stock_init", []), "stock_run_batch_raises"]
    assert [record["type"] for record in world.records()] == ["header"]


def test_stock_process_batch_result_error_passes_through_after_the_line_is_written(world):
    scheduler = _scheduler(world)
    batch = _extend_batch()
    result = scheduler.run_batch(batch)
    error = RuntimeError("FAKE failure of the stock process_batch_result")
    world.stock_process_error = error
    with pytest.raises(RuntimeError) as caught:
        scheduler.process_batch_result(batch, result)
    assert caught.value is error
    assert [record["type"] for record in world.records()] == ["header", "batch"]


def test_broken_log_never_stops_a_batch(world):
    scheduler = _scheduler(world)
    # The file is closed under the recorder: every write now raises, the fault line included.
    scheduler._sglbench_recorder._file.close()
    batch = _extend_batch()
    marker = FakeResult(FakeCopyDone(world.events))
    world.results.append(marker)
    assert scheduler.run_batch(batch) is marker
    assert scheduler.process_batch_result(batch, marker) is world.stock_process_return
    assert [record["type"] for record in world.records()] == ["header"]


# ---------------------------------------------------------------------------------------------------------------
# A log that cannot be opened
# ---------------------------------------------------------------------------------------------------------------


def _assert_scheduler_works_without_a_log(world, scheduler):
    assert scheduler._sglbench_recorder is None
    assert scheduler.page_size == FAKE_LIMITS["page_size"]
    batch = _extend_batch()
    marker = FakeResult(FakeCopyDone(world.events))
    world.results.append(marker)
    assert scheduler.run_batch(batch) is marker
    assert scheduler.process_batch_result(batch, marker) is world.stock_process_return
    assert [call[0] for call in world.stock_calls] == ["run_batch", "process_batch_result"]
    # Without a log the recorder does not wait and reads no clock.
    assert "synchronize" not in world.events
    assert [event for event in world.events if isinstance(event, tuple) and event[0] == "recorder_clock"] == []


def test_missing_directory_leaves_a_working_scheduler_and_no_header(world, monkeypatch, tmp_path, capsys):
    missing = tmp_path / "not-created-by-the-launcher"
    monkeypatch.setenv(ENV, str(missing))
    scheduler = _scheduler(world)
    _assert_scheduler_works_without_a_log(world, scheduler)
    assert not missing.exists(), "the recorder must not create the directory"
    assert list(world.directory.iterdir()) == []
    stderr = capsys.readouterr().err
    assert stderr.startswith(f"[sglbench] scheduler process {os.getpid()} runs without the recorder: ")
    assert "FileNotFoundError" in stderr


def test_existing_log_file_is_never_appended_to(world, capsys):
    world.log_path.write_text('{"type":"header","note":"FAKE line of an earlier process"}\n', encoding="utf-8")
    before = world.log_path.read_bytes()
    scheduler = _scheduler(world)
    assert scheduler._sglbench_recorder is None
    batch = _extend_batch()
    scheduler.process_batch_result(batch, scheduler.run_batch(batch))
    assert world.log_path.read_bytes() == before
    assert "FileExistsError" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------------------------
# The entry
# ---------------------------------------------------------------------------------------------------------------


def test_entry_registers_the_class_before_the_stock_entry_runs(world):
    tree = world.sglang
    server_args = FakeServerArgs(world)
    returned = recorder.recorder_entry(server_args, FAKE_PORT_ARGS, 0, 0, 0, None, "FAKE pipe")

    assert returned is tree.entry_return
    assert tree.events[:2] == ["register", "stock_entry"]
    ((target, cls, hook_type),) = tree.hooks
    assert target == recorder.TARGET == "sglang.srt.managers.scheduler.Scheduler"
    assert hook_type is FakeHookType.REPLACE
    assert cls.__name__ == CLASS_NAME
    assert cls.__mro__[1] is tree.stock

    # The FAKE stock entry applied the hook and built the class that the module name then held.
    (scheduler,) = tree.built
    assert type(scheduler) is cls
    assert tree.scheduler_module.Scheduler is cls
    (header,) = world.records()
    assert header["class"] == CLASS_NAME
    assert header["mro"] == [CLASS_NAME, "Scheduler", "FakeSchedulerBase"]


def test_entry_class_still_calls_the_stock_methods_after_sglang_rebinds_the_stock_name(world):
    tree = world.sglang
    recorder.recorder_entry(FakeServerArgs(world), FAKE_PORT_ARGS, 0, 0, 0, None, "FAKE pipe")
    (scheduler,) = tree.built
    # SGLang rebinds every module-level name that held the stock class. No name of the recorder module may
    # have been one of them, and the class must still reach the stock methods.
    assert [name for name, value in vars(recorder).items() if value is tree.stock or value is type(scheduler)] == []
    batch = _extend_batch()
    result = scheduler.run_batch(batch)
    assert scheduler.process_batch_result(batch, result) is world.stock_process_return
    assert [call[0] for call in world.stock_calls] == ["run_batch", "process_batch_result"]
    assert [record["type"] for record in world.records()] == ["header", "batch"]


@pytest.mark.parametrize(
    "positional",
    [
        pytest.param(("FAKE server args", FAKE_PORT_ARGS, 3, 1, 0, None, "FAKE pipe"), id="seven"),
        pytest.param(("FAKE server args", FAKE_PORT_ARGS, 3, 1, 0, 2, "FAKE pipe", 1, 2, 0), id="ten"),
    ],
)
def test_entry_passes_seven_and_ten_positional_arguments_through(fake_sglang, monkeypatch, tmp_path, positional):
    monkeypatch.setenv(ENV, str(tmp_path))
    world = FakeWorld(tmp_path, fake_sglang)
    arguments = (FakeServerArgs(world), *positional[1:])

    returned = recorder.recorder_entry(*arguments)

    assert returned is fake_sglang.entry_return
    (call,) = fake_sglang.entry_calls
    assert call == arguments + (None,) * (10 - len(arguments))
    assert call[0] is arguments[0]
    assert len(fake_sglang.hooks) == 1


def test_entry_is_a_module_level_function_that_spawn_can_name():
    # multiprocessing "spawn" sends the target of a process by module and name.
    assert recorder.recorder_entry.__module__ == "sglbench_recorder"
    assert recorder.recorder_entry.__qualname__ == "recorder_entry"
    assert pickle.loads(pickle.dumps(recorder.recorder_entry)) is recorder.recorder_entry


# ---------------------------------------------------------------------------------------------------------------
# The timed span, and an interrupt
# ---------------------------------------------------------------------------------------------------------------


class FakeBatchThatNotesItsReads:
    """FAKE batch that notes every attribute the recorder reads from it, in the events of the test."""

    def __init__(self, batch, events):
        object.__setattr__(self, "_batch", batch)
        object.__setattr__(self, "_events", events)

    def __getattr__(self, name):
        object.__getattribute__(self, "_events").append(("batch_read", name))
        return getattr(object.__getattribute__(self, "_batch"), name)


def test_out_time_is_taken_right_after_the_wait_and_before_any_batch_field_is_read(world):
    scheduler = _scheduler(world)
    batch = _extend_batch()
    result = scheduler.run_batch(batch)
    watched = FakeBatchThatNotesItsReads(batch.copy(), world.events)
    del world.events[:]

    scheduler.process_batch_result(watched, result)

    # The wait, then the clock, and only then the fields of the batch: reading and checking them costs time
    # that grows with the batch and must not be inside out_ts.
    assert world.events[:2] == ["synchronize", ("recorder_clock", _fake_time(3))]
    reads = [event[1] for event in world.events if isinstance(event, tuple) and event[0] == "batch_read"]
    assert reads[0] == "forward_iter" and "reqs" in reads and "launch_ts" in reads
    (line,) = world.lines_of("batch")
    assert line["out_ts"] == _fake_time(3)


def test_interrupt_in_a_recorder_step_or_in_the_log_open_is_not_swallowed(world, monkeypatch):
    scheduler = _scheduler(world)
    world.clock.recorder_error = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        scheduler.run_batch(_extend_batch())
    assert world.lines_of("fault") == []

    # The same in the result step.
    batch = _extend_batch()
    result = scheduler.run_batch(batch)
    world.clock.recorder_error = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        scheduler.process_batch_result(batch, result)
    assert world.lines_of("fault") == []

    def fake_header_that_is_interrupted(scheduler):
        raise KeyboardInterrupt

    monkeypatch.setattr(recorder, "_header", fake_header_that_is_interrupted)
    world.log_path.unlink()
    with pytest.raises(KeyboardInterrupt):
        _scheduler(world)
