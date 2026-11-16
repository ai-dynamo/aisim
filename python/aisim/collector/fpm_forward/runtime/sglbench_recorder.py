# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# The recorder subclasses SGLang's Scheduler and reads its batch fields at v0.5.22 and v0.5.23.
# See THIRD_PARTY_NOTICES.md.
# Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d
# Source: https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90

"""Scheduler-side recorder of AISim's SGLang self-benchmark: one JSON line per batch and scheduler process.

Where it runs
    In every SGLang scheduler process. AISim is not installed there, so the module uses the standard library
    only and imports SGLang inside the functions that need it. SGLang starts its scheduler processes with
    ``spawn`` and hands them the entry function by name. ``recorder_entry`` is therefore a module-level
    function, and this file has to be importable in the child.

What it does
    ``recorder_entry`` registers a subclass of SGLang's ``Scheduler`` with SGLang's own hook registry and then
    calls SGLang's stock scheduler entry. The subclass changes no scheduling decision. It calls the stock
    ``__init__``, ``run_batch`` and ``process_batch_result`` with the arguments it was given and passes their
    result or exception on unchanged. Around these calls it writes a log:

    - one header line, after the stock ``__init__`` has returned: which process this is, the admission limits
      the scheduler holds, the captured graph sizes, and the settings a reader tests before it trusts a run;
    - one batch line per call of ``process_batch_result``: the forward mode, the request ids, the launch time
      SGLang stamped on the batch, and the time at which the output of the batch was on the host;
    - one fault line when a recorder step raised. A fault never stops a batch.

    A recorder that cannot open its log leaves a working scheduler and no header. A reader must refuse a run
    in which a scheduler process has no header, and must fail a run on any fault line.

Log format 1
    Directory: the value of ``AISIM_SGLBENCH_RECORDER_DIR``, which the launcher creates for each run. One file
    per scheduler process, ``sched-<pid>.jsonl``, created new (an existing file is never appended to), UTF-8,
    one compact JSON object per line, flushed after every line. The header is always the first line.

    Header::

        {"type": "header", "format": 1, "pid": int, "class": str, "mro": [str, str, str], "sglang_version": str,
         "leader": bool, "ranks": {...}, "limits": {...}, "decode_graph_sizes": [int, ...],
         "prefill_graph_sizes": [int, ...], "enable_overlap": bool, "checks": {...}}

    ``ranks``, ``limits`` and ``checks`` hold the names listed in ``_RANK_NAMES``, ``_LIMIT_NAMES`` and
    ``_CHECK_NAMES``. A value that could not be read is written as ``{"unreadable": "<repr of the error>"}``.
    It is never left out and never replaced by a default.

    Batch line::

        {"type": "batch", "seq": int, "forward_iter": int, "mode": str, "launch_ts": float,
         "run_return_ts": float | null, "out_ts": float, "copy_done": bool, "rids": [str, ...],
         "extend_num_tokens": int | null, "prefix_lens": [int, ...] | null, "extend_lens": [int, ...] | null}

    ``seq`` counts the batch lines of the file from 1. The three extend fields are written for mode ``EXTEND``
    only; a decode batch still carries the values of the extend batch its object came from. A batch whose
    values cannot be read as these types gets a fault line and no batch line.

    Fault line: ``{"type": "fault", "where": "run_batch" | "process_batch_result", "error": str}``.

Times
    All three times are ``time.monotonic()`` seconds of the scheduler process. ``launch_ts`` is taken by SGLang
    at the start of ``run_batch``. ``run_return_ts`` is taken here when the stock ``run_batch`` has returned;
    under overlap scheduling the GPU work is then still in flight. ``out_ts`` is taken here in
    ``process_batch_result``, right after the wait on ``result.copy_done`` and before the stock method runs.
    Clocks of different scheduler processes are not comparable.

Checked against SGLang v0.5.22 (baa9befa381511cd4d56f83980951e5a45fffc6d) and
v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90) of ``sgl-project/sglang`` (Apache-2.0), by reading these files
under ``python/sglang`` at both tags::

    srt/managers/scheduler.py            run_scheduler_process, Scheduler.__init__, Scheduler.init_ipc_channels,
                                         Scheduler.init_model_worker, Scheduler.init_chunked_prefill,
                                         Scheduler.init_schedule_policy,
                                         Scheduler.init_watch_dog_memory_saver_input_blocker,
                                         Scheduler.run_batch, Scheduler.process_batch_result,
                                         Scheduler.event_loop_normal, Scheduler.event_loop_overlap
    srt/managers/schedule_batch.py       ScheduleBatch (fields), ScheduleBatch.prepare_for_extend,
                                         ScheduleBatch.prepare_for_decode, ScheduleBatch.copy, Req.rid
    srt/managers/utils.py                GenerationBatchResult.copy_done, GenerationBatchResult.copy_to_cpu,
                                         EmbeddingBatchResult.copy_done
    srt/managers/async_d2h_copy_worker.py  HostCopyDone.synchronize (v0.5.23 only)
    srt/managers/scheduler_components/batch_result_processor.py
                                         process_batch_result_prefill, process_batch_result_decode,
                                         process_batch_result_idle
    srt/managers/scheduler_components/ipc_channels.py  SchedulerIpcChannels.create
    srt/managers/schedule_policy.py      _IS_HIP
    srt/managers/tp_worker.py            TpModelWorker.model_runner, TpModelWorker.get_worker_info
    srt/model_executor/model_runner.py   ModelRunner.init_cuda_graphs
    srt/model_executor/runner/decode_cuda_graph_runner.py   DecodeCudaGraphRunner.capture_bs
    srt/model_executor/runner/prefill_cuda_graph_runner.py  PrefillCudaGraphRunner.capture_num_tokens
    srt/model_executor/runner/eager_runner.py                EagerRunner
    srt/runtime_context.py               get_parallel, ParallelContext._read
    srt/arg_groups/fields/parallel.py    Parallel (field names)
    srt/plugins/__init__.py              load_plugins
    srt/plugins/hook_registry.py         HookRegistry.register, HookRegistry._apply_target, _propagate_patch,
                                         HookType
    srt/entrypoints/engine.py            Engine.run_scheduler_process_func, Engine._launch_scheduler_processes
    srt/managers/data_parallel_controller.py  run_scheduler_process_with_init_pipe

Seen on running engines: benchmarks/evidence/collector/sglbench-engine-probe-20261113. The recorder script kept
in that folder is an earlier form of this module with another class name and other line fields. It was
installed through the same hook at SGLang 0.5.22 and 0.5.23: every scheduler process wrote a header, one
process at TP 1 and two at TP 2 (check_1), and no result file of the folder holds a recorder fault.

This module itself ran on engines on 2026-11-16, in eight launches of sglbench_launcher.py with SGLang 0.5.22
and 0.5.23 on H200 GPUs, at TP 1 and TP 2: every scheduler process wrote its header with every value readable,
and no log holds a fault line. The files of those launches are not kept in this repository.

No upstream function or block of source is copied. What follows SGLang and Dynamo is named in the header of this
file and recorded in THIRD_PARTY_NOTICES.md. For this module that is SGLang alone.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

# The one environment variable this module reads. The launcher sets it before it builds the engine, and the
# scheduler processes inherit it.
ENV_RECORDER_DIR = "AISIM_SGLBENCH_RECORDER_DIR"

RECORD_FORMAT = 1

# The name of the subclass. A reader compares the header's "class" with it. It is the only sign that the hook
# was applied: SGLang logs a hook that failed to apply and builds the stock class instead
# (sglang/srt/plugins/hook_registry.py, HookRegistry.apply_hooks), and the last REPLACE hook registered for a
# target wins, so a plugin installed in the image can displace this class.
RECORDER_CLASS_NAME = "SglbenchRecorderScheduler"

LOG_FILE_PREFIX = "sched-"
LOG_FILE_SUFFIX = ".jsonl"

# The dotted path SGLang's hook registry resolves: the module sglang.srt.managers.scheduler and its name
# Scheduler (sglang/srt/plugins/hook_registry.py, HookRegistry._apply_target).
TARGET = "sglang.srt.managers.scheduler.Scheduler"

# Times of run_batch returns that are kept for the join with their batch lines. In Scheduler.event_loop_normal
# one is open at a time, in Scheduler.event_loop_overlap at most two. A batch whose result is never processed
# leaves its note behind (the overlap loop leaves on shutdown with a queued result), so the oldest note is
# dropped above this count.
_MAX_RUN_RETURN_NOTES = 64

# Fields of sglang.srt.runtime_context.get_parallel(), a ParallelContext. The names are declared in
# sglang/srt/arg_groups/fields/parallel.py (class Parallel): tp_size, pp_size and attn_dp_size are configured
# fields, attn_dp_enabled is derived at publish, and the four ranks are written by publish(), which
# run_scheduler_process calls before it builds the Scheduler.
_RANK_NAMES = (
    "tp_size",
    "pp_size",
    "attn_dp_size",
    "attn_dp_enabled",
    "attn_tp_rank",
    "attn_cp_rank",
    "attn_dp_rank",
    "pp_rank",
)

# Attributes of the Scheduler object that bound what it admits (sglang/srt/managers/scheduler.py):
#   page_size                Scheduler.__init__, from get_schedule().page_size
#   max_total_num_tokens, max_prefill_tokens, max_running_requests, max_req_input_len
#                            Scheduler.init_model_worker, one tuple from self.tp_worker.get_worker_info(),
#                            after graph capture
#   chunked_prefill_size     Scheduler.init_chunked_prefill; None when chunking is off, never 0 or -1
_LIMIT_NAMES = (
    "page_size",
    "max_req_input_len",
    "max_total_num_tokens",
    "max_running_requests",
    "max_prefill_tokens",
    "chunked_prefill_size",
)

# Attributes of the Scheduler object that are written as they are (sglang/srt/managers/scheduler.py):
#   disable_radix_cache, is_hybrid_swa, is_hybrid_ssm   Scheduler.__init__, from the result of build_kv_cache
#   max_queued_requests                                 Scheduler.init_model_worker (get_worker_info tuple)
#   max_new_tokens_limit                                Scheduler.__init__, envs.SGLANG_MAX_NEW_TOKENS_LIMIT
#   schedule_policy, enable_priority_scheduling         Scheduler.__init__, from get_schedule()
#   is_mixed_chunk, prefill_decode_interval             Scheduler.init_chunked_prefill
_CHECK_VALUE_NAMES = (
    "disable_radix_cache",
    "is_hybrid_swa",
    "is_hybrid_ssm",
    "max_queued_requests",
    "max_new_tokens_limit",
    "schedule_policy",
    "enable_priority_scheduling",
    "is_mixed_chunk",
    "prefill_decode_interval",
)

# Attributes of the Scheduler object that hold an object or None. The header says whether one is there
# (sglang/srt/managers/scheduler.py):
#   prefill_delayer                 Scheduler.init_schedule_policy
#   min_free_slots_delayer          Scheduler.init_model_worker
#   recv_skipper, input_blocker     Scheduler.init_watch_dog_memory_saver_input_blocker
_CHECK_PRESENCE_NAMES = (
    "prefill_delayer",
    "min_free_slots_delayer",
    "recv_skipper",
    "input_blocker",
)

# pp_max_micro_batch_size is not an attribute of the Scheduler at either tag. It is a field of get_parallel();
# Scheduler.init_model_worker sets its default there (max_running_requests // pp_size, at least 1) when the
# operator gave none. is_hip is the module constant sglang.srt.managers.schedule_policy._IS_HIP.
_CHECK_NAMES = (*_CHECK_VALUE_NAMES, *_CHECK_PRESENCE_NAMES, "pp_max_micro_batch_size", "is_hip")


# ---------------------------------------------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------------------------------------------


def _describe(error: BaseException) -> str:
    """Return ``repr(error)``, or the type name when the repr itself raises, so that a line is still written."""
    try:
        return repr(error)
    except Exception:
        return type(error).__name__


def _unreadable(error: BaseException) -> dict:
    """The header form of a value that could not be read."""
    return {"unreadable": _describe(error)}


def _plain(value, name: str):
    """Return ``value`` when JSON can hold it: None, a bool, an int, a str, a finite float, or a list of these."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} is {value!r}, which JSON cannot hold")
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item, name) for item in value]
    raise TypeError(f"{name} is a {type(value).__name__}, not a JSON value")


def _whole_number(value, name: str) -> int:
    """Return ``value`` when it is an int and not a bool."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} is {value!r}, not an int")
    return value


def _whole_numbers(values, name: str) -> list:
    """Return ``values`` as a list when it is a list or tuple of ints without a bool.

    Only the distinct element types are looked at, so the cost stays low for a batch of thousands of requests.
    """
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} is a {type(values).__name__}, not a list of ints")
    for kind in set(map(type, values)):
        if kind is bool or not issubclass(kind, int):
            raise TypeError(f"{name} holds a {kind.__name__}, not only ints")
    return list(values)


def _guarded(read):
    """Return ``read()``, or the unreadable form when it raises."""
    try:
        return read()
    except Exception as error:
        return _unreadable(error)


def _read_each(source, names) -> dict:
    """Read every name from ``source`` on its own. A failed read marks that name only."""
    values = {}
    for name in names:
        try:
            values[name] = _plain(getattr(source, name), name)
        except Exception as error:
            values[name] = _unreadable(error)
    return values


# ---------------------------------------------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------------------------------------------


def _parallel_values(names) -> dict:
    """Read ``names`` from SGLang's parallel context, each on its own.

    sglang/srt/runtime_context.py, ParallelContext._read: a name that is not declared raises AttributeError, a
    rank that was not written in this process raises RuntimeError, and a configured field that was not
    published raises ValueError. ``getattr`` with a default would catch the first only, so every read has its
    own guard here.
    """
    try:
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
    except Exception as error:
        return {name: _unreadable(error) for name in names}
    return _read_each(parallel, names)


def _sglang_version() -> str:
    # sglang/__init__.py imports __version__ from sglang/version.py.
    import sglang

    version = sglang.__version__
    if not isinstance(version, str):
        raise TypeError(f"sglang.__version__ is a {type(version).__name__}, not a str")
    return version


def _is_hip():
    # sglang/srt/managers/schedule_policy.py: _IS_HIP = is_hip(), read once at import. The prefill admission in
    # that file has AMD-only rules behind it.
    from sglang.srt.managers.schedule_policy import _IS_HIP

    return _plain(_IS_HIP, "_IS_HIP")


def _is_leader(scheduler) -> bool:
    # sglang/srt/managers/scheduler.py, Scheduler.init_ipc_channels: recv_from_tokenizer is the request socket
    # in the process with pp_rank, attn_tp_rank and attn_cp_rank all 0, and None in every other scheduler
    # process (sglang/srt/managers/scheduler_components/ipc_channels.py, SchedulerIpcChannels.create). Only
    # that process receives requests and sends outputs; the others run the same batches.
    return scheduler.recv_from_tokenizer is not None


def _enable_overlap(scheduler) -> bool:
    # sglang/srt/managers/scheduler.py, Scheduler.__init__:
    # enable_overlap = not get_schedule().disable_overlap_schedule and not use_mlx(), always a bool.
    value = scheduler.enable_overlap
    if not isinstance(value, bool):
        raise TypeError(f"enable_overlap is {value!r}, not a bool")
    return value


def _decode_graph_sizes(scheduler) -> list:
    # sglang/srt/model_executor/model_runner.py, ModelRunner.init_cuda_graphs sets decode_cuda_graph_runner. It
    # is None when decode graphs are off. capture_bs is the sorted list of captured batch sizes
    # (sglang/srt/model_executor/runner/decode_cuda_graph_runner.py, DecodeCudaGraphRunner.__init__).
    # Not read: tp_worker.get_decode_cuda_graph_bs() and model_runner.decode_cuda_graph_capture_bs. At v0.5.22
    # ModelRunner.init_cuda_graphs does not fill that list, so it is empty after a normal start.
    runner = scheduler.tp_worker.model_runner.decode_cuda_graph_runner
    if runner is None:
        return []
    return _whole_numbers(runner.capture_bs, "decode_cuda_graph_runner.capture_bs")


def _prefill_graph_sizes(scheduler) -> list:
    # sglang/srt/model_executor/model_runner.py, ModelRunner.init_cuda_graphs sets prefill_cuda_graph_runner.
    # It is a PrefillCudaGraphRunner when prefill graphs were captured; capture_num_tokens is then the sorted
    # list of captured token counts (sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py,
    # PrefillCudaGraphRunner.__init__). Without captured prefill graphs it is None or an EagerRunner
    # (sglang/srt/model_executor/runner/eager_runner.py), which has no capture_num_tokens. Both mean that no
    # prefill graph exists, so the test is for the attribute and not for truth.
    runner = scheduler.tp_worker.model_runner.prefill_cuda_graph_runner
    if runner is None or not hasattr(runner, "capture_num_tokens"):
        return []
    return _whole_numbers(runner.capture_num_tokens, "prefill_cuda_graph_runner.capture_num_tokens")


def _checks(scheduler) -> dict:
    values = _read_each(scheduler, _CHECK_VALUE_NAMES)
    for name in _CHECK_PRESENCE_NAMES:
        try:
            values[name] = getattr(scheduler, name) is not None
        except Exception as error:
            values[name] = _unreadable(error)
    values.update(_parallel_values(("pp_max_micro_batch_size",)))
    values["is_hip"] = _guarded(_is_hip)
    return {name: values[name] for name in _CHECK_NAMES}


def _header(scheduler) -> dict:
    """Build the header of one scheduler process. Every value is read on its own, so this does not raise."""
    return {
        "type": "header",
        "format": RECORD_FORMAT,
        "pid": os.getpid(),
        "class": type(scheduler).__name__,
        "mro": [cls.__name__ for cls in type(scheduler).__mro__[:3]],
        "sglang_version": _guarded(_sglang_version),
        "leader": _guarded(lambda: _is_leader(scheduler)),
        "ranks": _parallel_values(_RANK_NAMES),
        "limits": _read_each(scheduler, _LIMIT_NAMES),
        "decode_graph_sizes": _guarded(lambda: _decode_graph_sizes(scheduler)),
        "prefill_graph_sizes": _guarded(lambda: _prefill_graph_sizes(scheduler)),
        "enable_overlap": _guarded(lambda: _enable_overlap(scheduler)),
        "checks": _checks(scheduler),
    }


# ---------------------------------------------------------------------------------------------------------------
# Log of one scheduler process
# ---------------------------------------------------------------------------------------------------------------


class _RecorderLog:
    """The log file of one scheduler process, and the run_batch return times that wait for their batch line.

    Both scheduler methods run on the scheduler's own thread (sglang/srt/managers/scheduler.py: the entry
    builds the Scheduler and runs its event loop in one thread), so nothing here takes a lock. SGLang starts
    the process with ``spawn`` and no file under sglang/srt calls ``os.fork``, so the file opened here stays
    valid for the life of the process.
    """

    def __init__(self, scheduler):
        directory = os.environ.get(ENV_RECORDER_DIR)
        if not directory:
            raise RuntimeError(f"{ENV_RECORDER_DIR} is not set")
        self._seq = 0
        self._run_return: dict[int, float] = {}
        header = _header(scheduler)
        path = os.path.join(directory, f"{LOG_FILE_PREFIX}{os.getpid()}{LOG_FILE_SUFFIX}")
        # Mode "x": the file must be new. Process ids repeat, and a second header after the lines of another
        # process would break the rule that the header is the first line. The directory is not created here:
        # the launcher makes it, and a missing directory means the run is not set up.
        self._file = open(path, "x", encoding="utf-8", newline="\n")  # noqa: SIM115 - open for the process life
        try:
            self._write(json.dumps(header, separators=(",", ":"), allow_nan=False))
        except BaseException:
            self._file.close()
            raise

    def _write(self, text: str) -> None:
        # One write and one flush per line, so a reader that polls the file sees whole lines apart from the
        # one being written. No fsync: the reader is on the same host.
        self._file.write(text + "\n")
        self._file.flush()

    def fault(self, where: str, error: BaseException) -> None:
        """Write a fault line. Never raises: a fault must not stop a batch."""
        try:
            record = {"type": "fault", "where": where, "error": _describe(error)}
            self._write(json.dumps(record, separators=(",", ":"), allow_nan=False))
        except Exception:
            # The log itself is broken (for example a full disk). Nothing more can be written. The batch line
            # of this batch is missing, so the reader fails the stage when it waits for that line.
            pass

    def note_run_return(self, batch) -> None:
        """Keep the time at which the stock ``run_batch`` returned, under the batch's ``forward_iter``."""
        returned = time.monotonic()
        # sglang/srt/managers/scheduler.py, Scheduler.run_batch: self.forward_ct += 1, then
        # batch.forward_iter = self.forward_ct. A count of run_batch calls in this process, from 1.
        forward_iter = _whole_number(batch.forward_iter, "batch.forward_iter")
        notes = self._run_return
        notes[forward_iter] = returned
        if len(notes) > _MAX_RUN_RETURN_NOTES:
            del notes[next(iter(notes))]

    def record_batch(self, batch, result) -> None:
        """Wait for the output of the batch, stamp it and write the batch line. Runs before the stock method."""
        # Deviation from serving: one more wait on the copy event, ahead of the stock one.
        #   What differs: serving waits on result.copy_done inside the result processors
        #   (sglang/srt/managers/scheduler_components/batch_result_processor.py, process_batch_result_prefill,
        #   process_batch_result_decode, process_batch_result_idle). Before them Scheduler.process_batch_result
        #   runs flush_trace_batch, publish_load_snapshot and publish_load_stat. Here the wait comes first, so
        #   these three calls no longer run while the GPU works on this batch. The event is the one the stock
        #   code waits on next in the same call, so no blocking case is added that serving does not have. On
        #   every engine with overlap scheduling on.
        #   Direction: a timed step can only get longer than in serving, by at most the host time of the three
        #   calls, and only where the host side of the scheduler loop sets the step time and not the GPU.
        #   Sources: Scheduler.run_batch creates the event (self.device_module.Event()) and
        #   GenerationBatchResult.copy_to_cpu records it after it has queued the host copies
        #   (sglang/srt/managers/utils.py). Seen on running engines:
        #   benchmarks/evidence/collector/sglbench-engine-probe-20261113 (check_1). The recorder script kept there
        #   calls synchronize() on the event in the same place, overlap scheduling was on in every run and
        #   "faults" is empty in all 17 result files. The files do not record whether the event was there.
        #   Retire: when SGLang stamps the end of the host copy on the result object itself at every tested
        #   version; read that stamp then and drop the wait.
        # The attribute is read plainly. Both result classes have it at both tags (GenerationBatchResult and
        # EmbeddingBatchResult), so a result without it is a changed seam and becomes a fault line, never a
        # line that says "no event". With overlap scheduling off the event is None on the plain path and
        # out_ts can lie before the end of the GPU work; the line says so in "copy_done".
        copy_done = result.copy_done
        if copy_done is not None:
            # v0.5.23 under confidential computing: copy_done is a HostCopyDone, whose synchronize() raises
            # RuntimeError when the copy failed (sglang/srt/managers/async_d2h_copy_worker.py). The caller
            # turns that into a fault line and the stock method then raises the same error itself.
            copy_done.synchronize()
        out_ts = time.monotonic()

        # Everything below is read from the batch argument. Under overlap scheduling it is a copy
        # (Scheduler.event_loop_overlap queues batch.copy() with the result), and the live object has by now
        # been turned into the next step, so object identity and the live batch tell nothing about this
        # result. ScheduleBatch.copy carries forward_iter, launch_ts, forward_mode, a copy of the list reqs
        # and extend_num_tokens (sglang/srt/managers/schedule_batch.py). It passes prefix_lens and
        # extend_lens on as the SAME list objects the live batch holds. They still describe this batch
        # because SGLang only rebinds these two attributes to new lists (ScheduleBatch.prepare_for_extend,
        # mix_with_running, convert_decode_to_extend) and edits neither in place; read in the source at
        # both tags.
        forward_iter = _whole_number(batch.forward_iter, "batch.forward_iter")
        run_return_ts = self._run_return.pop(forward_iter, None)
        # ScheduleBatch.forward_mode is a ForwardMode, an IntEnum (sglang/srt/model_executor/
        # forward_batch_info.py). Its member name is logged: EXTEND, DECODE, IDLE and so on.
        mode = batch.forward_mode.name
        if not isinstance(mode, str):
            raise TypeError(f"batch.forward_mode.name is {mode!r}, not a str")
        # Scheduler.run_batch: batch.launch_ts = time.monotonic().
        launch_ts = batch.launch_ts
        if not isinstance(launch_ts, float):
            raise TypeError(f"batch.launch_ts is {launch_ts!r}, not a float")
        # Req.rid is fixed when the request is built. ScheduleBatch.reqs is rebound and never edited in place.
        rids = [req.rid for req in batch.reqs]
        for kind in set(map(type, rids)):
            if not issubclass(kind, str):
                raise TypeError(f"a request id of the batch is a {kind.__name__}, not a str")
        if mode == "EXTEND":
            # ScheduleBatch.prepare_for_extend sets all three: the tokens this extend batch computes, and per
            # request the cached and the computed tokens. ScheduleBatch.prepare_for_decode leaves them as they
            # are, so on a decode batch they are those of an earlier extend batch and are not written.
            extend_num_tokens = _whole_number(batch.extend_num_tokens, "batch.extend_num_tokens")
            prefix_lens = _whole_numbers(batch.prefix_lens, "batch.prefix_lens")
            extend_lens = _whole_numbers(batch.extend_lens, "batch.extend_lens")
        else:
            extend_num_tokens = prefix_lens = extend_lens = None

        # Deviation from serving: a line is written and flushed between two scheduler steps.
        #   What differs: serving writes no file per batch. Here the scheduler thread reads the fields above,
        #   builds one JSON line and makes one write and one flush, on every engine and for every batch.
        #   Direction: a timed step can only get longer than in serving. The cost grows with the number of
        #   requests in the batch. Under overlap scheduling the next batch is already on the GPU while the line
        #   is written, so the write adds to a timed step only where the host side of the loop sets the step
        #   time.
        #   Sources: Scheduler.event_loop_overlap (run_batch of the next step comes before process_batch_result
        #   of this one). The size of the cost was not measured.
        #   Retire: measure it on an engine; if it shows, keep the lines in memory and write them when no
        #   batch is in flight.
        # Nothing else goes into the line. NaN and infinity are refused, so every line is valid JSON.
        seq = self._seq + 1
        text = json.dumps(
            {
                "type": "batch",
                "seq": seq,
                "forward_iter": forward_iter,
                "mode": mode,
                "launch_ts": launch_ts,
                "run_return_ts": run_return_ts,
                "out_ts": out_ts,
                "copy_done": copy_done is not None,
                "rids": rids,
                "extend_num_tokens": extend_num_tokens,
                "prefix_lens": prefix_lens,
                "extend_lens": extend_lens,
            },
            separators=(",", ":"),
            allow_nan=False,
        )
        # The number is used up once the line is built, so a write that fails half way never gives two lines
        # the same number.
        self._seq = seq
        self._write(text)


def _report_no_log(error: BaseException) -> None:
    """Say on stderr why this scheduler process records nothing. Never raises."""
    try:
        print(
            f"[sglbench] scheduler process {os.getpid()} runs without the recorder: {_describe(error)}",
            file=sys.stderr,
            flush=True,
        )
    except Exception:
        # stderr is gone. The missing header still tells the reader.
        pass


# ---------------------------------------------------------------------------------------------------------------
# The subclass and the entry
# ---------------------------------------------------------------------------------------------------------------


def make_recorder_class(base):
    """Return a subclass of ``base``, SGLang's ``Scheduler``, that records every batch.

    The class is built here, with ``base`` in a closure, and no module-level name of this file ever holds
    ``base``. When SGLang applies a class hook it rebinds every module-level name in ``sys.modules`` whose value
    is the stock class (sglang/srt/plugins/hook_registry.py, _propagate_patch); a module-level name here would
    then point at the subclass itself.
    """

    class SglbenchRecorderScheduler(base):
        # Present on the class, so the two methods below also work when the stock __init__ calls them before
        # the log exists, and when the log could not be opened.
        _sglbench_recorder = None

        def __init__(self, *args, **kwargs):
            # sglang/srt/managers/scheduler.py, Scheduler.__init__(self, server_args, port_args). The limits,
            # the graph runners and the ranks exist only after it has returned. run_scheduler_process sends
            # the scheduler's init info to the parent right after the constructor, so every header exists
            # once the engine constructor has returned in the launcher.
            super().__init__(*args, **kwargs)
            try:
                self._sglbench_recorder = _RecorderLog(self)
            except Exception as error:
                # An exception that leaves this constructor stops the engine (run_scheduler_process signals
                # the parent). The scheduler therefore starts without a log, and the reader refuses the run
                # because this process has no header.
                _report_no_log(error)

        def run_batch(self, batch, *args, **kwargs):
            # Scheduler.run_batch(self, batch, pp_proxy_tensors=None). It stamps batch.forward_iter and
            # batch.launch_ts and returns the result object. The time of its return is on no object, so it is
            # noted here under forward_iter.
            result = super().run_batch(batch, *args, **kwargs)
            log = self._sglbench_recorder
            if log is not None:
                try:
                    log.note_run_return(batch)
                except Exception as error:
                    log.fault("run_batch", error)
            return result

        def process_batch_result(self, batch, result, *args, **kwargs):
            # Scheduler.process_batch_result(self, batch, result). The recorder step comes first; the stock
            # method then runs with the same arguments whatever the step did.
            log = self._sglbench_recorder
            if log is not None:
                try:
                    log.record_batch(batch, result)
                except Exception as error:
                    log.fault("process_batch_result", error)
            return super().process_batch_result(batch, result, *args, **kwargs)

    SglbenchRecorderScheduler.__name__ = RECORDER_CLASS_NAME
    SglbenchRecorderScheduler.__qualname__ = RECORDER_CLASS_NAME
    return SglbenchRecorderScheduler


def recorder_entry(*args, **kwargs):
    """Entry of a scheduler process: register the recorder class, then run SGLang's stock entry.

    The launcher gives this function to the engine in the place of ``run_scheduler_process``
    (sglang/srt/entrypoints/engine.py, Engine.run_scheduler_process_func). SGLang calls it with positional
    arguments only: seven from Engine._launch_scheduler_processes, ten from
    run_scheduler_process_with_init_pipe and from the data-parallel controller
    (sglang/srt/managers/data_parallel_controller.py). Everything is passed through unchanged.

    The class is registered before the stock entry runs, because the first statement of
    ``run_scheduler_process`` is ``load_plugins()``, which applies the registered hooks, and it applies them
    once per process: a later call returns at once (sglang/srt/plugins/__init__.py). After that the stock
    entry builds ``Scheduler(server_args, port_args)`` by a look-up of the module name at call time, which is
    then this class. Nothing is registered at import, because the launcher process and other SGLang processes
    import this file as well and call ``load_plugins()`` themselves.

    Without ``AISIM_SGLBENCH_RECORDER_DIR`` the stock entry runs with no change.
    """
    import sglang.srt.managers.scheduler as scheduler_module

    if os.environ.get(ENV_RECORDER_DIR):
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        # HookRegistry.register(target, hook, hook_type): a class is accepted with HookType.REPLACE only.
        HookRegistry.register(TARGET, make_recorder_class(scheduler_module.Scheduler), HookType.REPLACE)
    return scheduler_module.run_scheduler_process(*args, **kwargs)
