# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XPU caller/fit comparison on synthetic FPM prices, not silicon accuracy.

Usage from the repository root:
    python benchmarks/evidence/mamba-mixed-step-accounting/xpu_scoped_repair.py OUT.json
The baseline class is read from the pinned commit; the candidate is imported.
The base mixed-step seam records the shapes handed to native pricing and
returns 1 ms/launch + .01 ms/new prefill token + .1 ms/decode request.
Decode growth is held at one and capacity admits the full batch. The real
XPU budget conversion, own-prefill scaling and burst factor execute unchanged.
"""

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

BASE = "0cfbdb5535583493e4ab073241e1f74d707acdde"
SOURCE = "python/aisim/src/aisim_core/sdk/backends/vllm_backend_xpu.py"


def verified_candidate_hash(backend_class, expected_source):
    """Bind the fingerprint to the Python class actually used by the comparison."""
    imported_source = inspect.getsourcefile(backend_class)
    if imported_source is None:
        raise ValueError("candidate backend has no inspectable source")
    imported_bytes = Path(imported_source).read_bytes()
    if imported_bytes != Path(expected_source).read_bytes():
        raise ValueError(f"imported candidate backend differs from checkout: {imported_source}")
    return hashlib.sha256(imported_bytes).hexdigest()


def main():
    from aisim_core.sdk.backends.vllm_backend import VLLMBackend
    from aisim_core.sdk.backends.vllm_backend_xpu import VLLMXPUBackend
    from aisim_core.sdk.step_estimate import StepEstimate

    expected_source = Path(__file__).resolve().parents[3] / SOURCE
    candidate_hash = verified_candidate_hash(VLLMXPUBackend, expected_source)
    baseline_source = subprocess.check_output(["git", "show", f"{BASE}:{SOURCE}"], text=True)
    baseline = ModuleType("baseline_xpu")
    exec(compile(baseline_source, f"{BASE}:{SOURCE}", "exec"), baseline.__dict__)
    result = {
        "candidate_source_sha256": candidate_hash,
        "baseline": BASE,
        "price_kind": "synthetic FPM caller fixture",
        "cases": [],
    }
    for prefix, budget in [(3000, 2048), (4088, 4096), (0, 2048)]:
        case = {"batch": 16, "isl": 4096, "prefix": prefix, "osl": 128, "budget": budget, "decode": 15}
        for name, cls in [("before", baseline.VLLMXPUBackend), ("after", VLLMXPUBackend)]:
            backend = cls()
            backend._in_agg = True
            backend._visual_context_tokens = lambda *_: 0
            backend._max_kv_slots = lambda *_, **__: 1000
            backend._mix_step_gen_tokens = lambda *_: 15
            backend._effective_decode_bs = lambda *_: 16
            backend._get_genonly_step_latency = lambda *_: (1, 0, {}, {}, ())
            shapes = []

            def native_step(self, model, database, runtime, step):
                shapes.append([step.context_tokens, step.num_decode_requests])
                return StepEstimate(
                    latency_ms=1 + 0.01 * step.context_tokens + 0.1 * step.num_decode_requests,
                    energy_wms=0,
                    context_tokens=step.context_tokens,
                )

            with patch.object(VLLMBackend, "run_mixed", native_step):
                ttft = backend._compute_ttft(
                    SimpleNamespace(forward_model="fpm", _num_layers=10),
                    None,
                    SimpleNamespace(isl=4096, osl=128, prefix=prefix, batch_size=16),
                    16,
                    4096,
                    128,
                    budget,
                    prefix,
                    prefill_step_ms=0,
                    genonly_step_latency_ms=1,
                    encoder_latency_ms=0,
                    steps_to_finish_ctx=2,
                    decode_iterations=127,
                )
            case[name] = {"pricing_shapes": shapes, "ttft_ms": ttft}
        result["cases"].append(case)
    result["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["cases"], indent=2))


if __name__ == "__main__":
    main()
