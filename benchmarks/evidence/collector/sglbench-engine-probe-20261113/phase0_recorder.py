# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# The recorder of the engine probe, as saved. Added for the repository: these comment lines, the layout that
# "ruff format" gives, and one lint mark with its reason. No statement was changed: the syntax tree equals that
# of the saved file, whose sha256 is in provenance.json. See README.md in this folder.
"""Phase 0 spike: scheduler-side recorder for the AISim SGLang self-benchmark.

This module runs inside every SGLang scheduler process. SGLang starts those
processes with ``spawn`` and looks the entry function up by name, so
``recorder_entry`` must stay a module-level function in a file that sits next
to ``phase0_spike.py``.

It changes no scheduling decision. It subclasses ``Scheduler``, calls the stock
method in both overrides, and appends one JSON line per batch to a file.
"""

from __future__ import annotations

import json
import os
import time

ENV_DIR = "AISIM_SGLBENCH_RECORDER_DIR"
ENV_INSTALL = "AISIM_SGLBENCH_INSTALL"  # "hook" (default) or "setattr"
TARGET = "sglang.srt.managers.scheduler.Scheduler"


def _plain(value):
    """Make a value safe for json.dumps."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    try:
        return [_plain(item) for item in value.tolist()]  # tensors
    except Exception:
        return repr(value)


def _ranks() -> dict:
    try:
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        names = (
            "attn_tp_rank",
            "attn_dp_rank",
            "attn_cp_rank",
            "pp_rank",
            "tp_size",
            "attn_dp_enabled",
        )
        return {name: _plain(getattr(parallel, name, None)) for name in names}
    except Exception as error:  # older layouts
        return {"error": repr(error)}


class _Log:
    def __init__(self, scheduler, install: str):
        self._run_return = {}
        directory = os.environ[ENV_DIR]
        os.makedirs(directory, exist_ok=True)
        # Lint mark SIM115, kept as saved: the log file stays open for the life of the scheduler process.
        self._file = open(os.path.join(directory, f"sched-{os.getpid()}.jsonl"), "a", buffering=1)  # noqa: SIM115
        try:
            import sglang

            version = getattr(sglang, "__version__", "unknown")
        except Exception:
            version = "unknown"
        header = {
            "type": "header",
            "pid": os.getpid(),
            "sglang_version": version,
            "class": type(scheduler).__name__,
            "mro": [cls.__name__ for cls in type(scheduler).__mro__[:3]],
            "install": install,
            "ranks": _ranks(),
        }
        for name in (
            "page_size",
            "enable_overlap",
            "chunked_prefill_size",
            "max_prefill_tokens",
            "max_running_requests",
            "max_total_num_tokens",
            "max_req_input_len",
        ):
            header[name] = _plain(getattr(scheduler, name, None))
        header["input_blocker"] = getattr(scheduler, "input_blocker", None) is not None
        self._write(header)

    def _write(self, record: dict) -> None:
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._file.flush()

    def fault(self, where: str, error: BaseException) -> None:
        try:
            self._write({"type": "fault", "where": where, "error": repr(error)})
        except Exception:
            pass

    def after_run(self, batch) -> None:
        self._run_return[getattr(batch, "forward_iter", None)] = time.monotonic()
        if len(self._run_return) > 64:
            self._run_return.pop(next(iter(self._run_return)))

    def before_result(self, batch, result) -> None:
        copy_done = getattr(result, "copy_done", None)
        if copy_done is not None:
            synchronize = getattr(copy_done, "synchronize", None)
            if synchronize is not None:
                synchronize()
        out_ts = time.monotonic()
        mode = getattr(batch, "forward_mode", None)
        reqs = list(getattr(batch, "reqs", None) or [])
        forward_iter = getattr(batch, "forward_iter", None)
        self._write(
            {
                "type": "batch",
                "forward_iter": forward_iter,
                "mode": getattr(mode, "name", str(mode)),
                "launch_ts": getattr(batch, "launch_ts", None),
                "run_return_ts": self._run_return.pop(forward_iter, None),
                "out_ts": out_ts,
                "copy_done": copy_done is not None,
                "bs": len(reqs),
                "extend_num_tokens": _plain(getattr(batch, "extend_num_tokens", None)),
                "global_num_tokens": _plain(getattr(batch, "global_num_tokens", None)),
                "rids": [getattr(req, "rid", None) for req in reqs],
                "cached": [_plain(getattr(req, "cached_tokens", None)) for req in reqs],
                "seq": [
                    len(getattr(req, "origin_input_ids", ()) or ()) + len(getattr(req, "output_ids", ()) or ())
                    for req in reqs
                ],
            }
        )


def _make_recorder_class(base, install: str):
    class RecorderScheduler(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._sglbench = None
            try:
                self._sglbench = _Log(self, install)
            except Exception:
                # Without a header the driver fails the run; never stop the engine here.
                pass

        def run_batch(self, batch, *args, **kwargs):
            result = super().run_batch(batch, *args, **kwargs)
            if self._sglbench is not None:
                try:
                    self._sglbench.after_run(batch)
                except Exception as error:
                    self._sglbench.fault("run_batch", error)
            return result

        def process_batch_result(self, batch, result, *args, **kwargs):
            if self._sglbench is not None:
                try:
                    self._sglbench.before_result(batch, result)
                except Exception as error:
                    self._sglbench.fault("process_batch_result", error)
            return super().process_batch_result(batch, result, *args, **kwargs)

    return RecorderScheduler


def recorder_entry(*args, **kwargs):
    """Replacement for SGLang's scheduler-process entry; runs in the scheduler process."""
    import sglang.srt.managers.scheduler as scheduler_module

    stock_entry = scheduler_module.run_scheduler_process
    if os.environ.get(ENV_DIR):
        install = os.environ.get(ENV_INSTALL, "hook")
        recorder_class = _make_recorder_class(scheduler_module.Scheduler, install)
        if install == "hook":
            from sglang.srt.plugins.hook_registry import HookRegistry, HookType

            # Applied by load_plugins() at the top of run_scheduler_process.
            HookRegistry.register(TARGET, recorder_class, HookType.REPLACE)
        else:
            scheduler_module.Scheduler = recorder_class
    return stock_entry(*args, **kwargs)
