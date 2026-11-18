# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-request run_agg through the compiled whole-model FPM path.

Usage: fpm_caller.py REPO_ROOT OUT.json

The table is synthetic, built with the FPM unit-test helpers: one anchor at
each coordinate an old or a corrected caller queries. The values test the
caller input, not prediction accuracy.
"""

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(sys.argv[1], "python/aisim/tests/unit/sdk"))
import test_fpm_forward as t

from aisim.sdk.inference_session import InferenceSession
from aisim_core import _native
from aisim_core.sdk import models
from aisim_core.sdk.backends.factory import get_backend
from aisim_core.sdk.config import RuntimeConfig
from aisim_core.sdk.perf_database import PerfDatabase
from aisim_core.sdk.step_estimate import MixedStepInput

ISL, OSL = 512, 2
PREFILL = {(1, 256, 256): 12.0, (2, 512, 512): 30.0, (1, 512, 0): 22.0}
DECODE = {(1, 0, ISL + OSL // 2 + 1): 3.0, (1, 0, ISL + 1): 2.5}

model = models.get_model("Qwen/Qwen3-0.6B", t._model_config(forward_model="fpm"), t.BACKEND)
identity = dict(zip(t._CELL_MATCH_COLUMNS, model.context_ops[0]._match_identity, strict=True))
rows = [t._row("prefill", *key, ms, model_path=model.model_path, identity=identity) for key, ms in PREFILL.items()]
rows += [t._row("decode", *key, ms, model_path=model.model_path, identity=identity) for key, ms in DECODE.items()]
root = tempfile.mkdtemp()
try:
    shutil.copy(os.path.join(t._CORE_SYSTEMS, f"{t.SYSTEM}.yaml"), os.path.join(root, f"{t.SYSTEM}.yaml"))
    data_dir = os.path.join(root, "data", t.SYSTEM, t.BACKEND, t.VERSION)
    t._write_pair(data_dir, rows)
    with open(os.path.join(data_dir, "fpm_forward_perf.parquet"), "rb") as handle:
        table_sha = hashlib.sha256(handle.read()).hexdigest()
    database = PerfDatabase(t.SYSTEM, t.BACKEND, t.VERSION, systems_root=root)
    backend = get_backend(t.BACKEND)
    dispatch_ms = backend._prefill_dispatch_overhead_ms(model)
    out = {
        "model": model.model_path,
        "forward_model": model.forward_model,
        "interpolation": "sol (run_agg compiles every FpmForward op with it)",
        "system": t.SYSTEM,
        "backend": t.BACKEND,
        "version": t.VERSION,
        "table_sha256": table_sha,
        # The imported build: the helpers come from REPO_ROOT, the estimator from
        # the active environment, so record which native extension priced the run.
        "native_extension_sha256": hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
        "dispatch_ms": dispatch_ms,
        "cases": {},
    }
    for name, prefix, ctx in (("cached", 256, ISL), ("uncached", 0, ISL)):
        runtime = RuntimeConfig(batch_size=1, beam_width=1, isl=ISL, osl=OSL, prefix=prefix)
        new_tokens = ISL - prefix
        # Inputs each caller version passes to the mixed step, priced in this build.
        inputs = {"old_caller": ISL, "corrected_caller": min(ctx, new_tokens)}
        steps = {
            key: {
                "context_tokens": tokens,
                "expected_requests": math.ceil(tokens / new_tokens),
                "step_ms": backend.run_mixed(model, database, runtime, MixedStepInput(tokens, 0)).latency_ms,
            }
            for key, tokens in inputs.items()
        }
        summary = InferenceSession(model, database, backend).run_agg(runtime, ctx_tokens=ctx)
        out["cases"][name] = {
            "inputs": {"batch_size": 1, "isl": ISL, "osl": OSL, "prefix": prefix, "ctx_tokens": ctx},
            "steps": steps,
            "run_agg_mix_step_ms": float(sum(summary.get_per_ops_data()["mix_step"].values())),
            "run_agg_ttft_ms": float(summary.get_summary_df().iloc[0]["ttft"]),
        }
finally:
    shutil.rmtree(root)
with open(sys.argv[2], "w") as f:
    json.dump(out, f, indent=1)
