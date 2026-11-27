# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install a lazy observer loader. The explicit executor import fails on a mismatch."""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import json
import os
import socket
import sys
from pathlib import Path
from types import ModuleType

_TARGET = "tensorrt_llm._torch.pyexecutor.py_executor"
_SOURCE = "tensorrt_llm/_torch/pyexecutor/py_executor.py"
_ROOT = Path(__file__).resolve().parent
# Distinct from the collector's aisimulate-runtime-instrumentation-imports/v1
# binding. This receipt hashes this loader and the upstream executor, which
# that contract's reader (runtime_observations._validate_import_binding) does
# not accept.
RECEIPT_SCHEMA = "aisim-trtllm-fpm-observer-imports/v1"


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _verify(context, manifest_path, source_path):
    manifest = json.loads(manifest_path.read_text())
    if manifest_path.parent != _ROOT or manifest.get("files") != [
        "sitecustomize.py",
        "observer.py",
        "runtime-source-sha256.json",
    ]:
        raise RuntimeError("invalid TRT-LLM observer bundle manifest")
    files = {name: _digest((_ROOT / name).read_bytes()) for name in manifest["files"]}
    payload = json.dumps({"manifest": manifest, "files": files}, sort_keys=True, separators=(",", ":"))
    if _digest(payload.encode()) != context["bundle_sha256"]:
        raise RuntimeError("TRT-LLM observer bundle hash mismatch")
    if importlib.metadata.version("tensorrt_llm") != "1.3.0rc29":
        raise RuntimeError("TRT-LLM observer requires 1.3.0rc29")
    expected = json.loads((_ROOT / "runtime-source-sha256.json").read_text())[_SOURCE]
    raw = source_path.read_bytes()
    actual = _digest(raw)
    if actual != expected:
        raise RuntimeError(f"TRT-LLM pinned executor source mismatch: expected {expected}, got {actual}")
    return raw, files, actual


def _write_receipt(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


class _ExecutorLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        directory = Path(os.environ["AISIMULATE_RUNTIME_OBSERVATION_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        receipt = directory / f"runtime-instrumentation-imports-{socket.gethostname()}-{os.getpid()}.json"
        record = {
            "schema_version": RECEIPT_SCHEMA,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "root": str(_ROOT),
            "loader_sha256": _digest(Path(__file__).read_bytes()),
            "imports": {},
            "classes": {},
        }
        try:
            context = json.loads(Path(os.environ["AISIMULATE_RUNTIME_CONTEXT"]).read_text())
            record.update({key: context[key] for key in ("attempt_id", "configuration", "phase")})
            record["expected_bundle_sha256"] = context["bundle_sha256"]
            manifest = Path(os.environ["AISIMULATE_RUNTIME_INSTRUMENTATION"]).resolve()
            source = Path(module.__spec__.origin)
            raw, files, source_digest = _verify(context, manifest, source)
            observer = ModuleType("_aisim_trtllm_fpm_observer")
            observer.__file__ = str(_ROOT / "observer.py")
            observer_source = Path(observer.__file__).read_bytes()
            if _digest(observer_source) != files["observer.py"]:
                raise RuntimeError("TRT-LLM observer source changed after verification")
            sys.modules[observer.__name__] = observer
            exec(compile(observer_source, observer.__file__, "exec"), observer.__dict__)
            exec(compile(raw, str(source), "exec"), module.__dict__)
            observer.install_hooks(module.PyExecutor, context, directory)
            record.update(
                status="verified",
                files=files,
                bundle_sha256=context["bundle_sha256"],
                imports={
                    "observer": {"file": "observer.py", "path": observer.__file__, "sha256": files["observer.py"]},
                    _TARGET: {"file": _SOURCE, "path": str(source), "sha256": source_digest},
                },
            )
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            _write_receipt(receipt, record)
            raise
        _write_receipt(receipt, record)


class _ExecutorFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or not spec.origin:
            raise ImportError("TRT-LLM executor source is missing")
        spec.loader = _ExecutorLoader()
        return spec


if os.environ.get("AISIMULATE_TRTLLM_FPM_OBSERVER") == "1":
    sys.meta_path.insert(0, _ExecutorFinder())
