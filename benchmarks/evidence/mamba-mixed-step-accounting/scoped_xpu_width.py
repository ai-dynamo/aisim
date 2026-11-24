# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare XPU reservations using one native SOL engine and an isolated scan graph.

Run from the repository root with an optimized wheel from --candidate-revision:
    python benchmarks/evidence/mamba-mixed-step-accounting/scoped_xpu_width.py OUT.json

Only the baseline Python XPU class is loaded from the pinned revision. Both
classes use the candidate's native engine; this isolates caller scheduling,
rather than claiming a complete historical replay. Occupancy is fixed at two
decoders/four admitted requests and decode growth at one. No silicon accuracy,
host execution speed, or production throughput is measured.
"""

import argparse
import hashlib
import inspect
import json
import subprocess
from pathlib import Path
from types import ModuleType

import aisim._runtime as native_runtime
from aisim_core.sdk.backends.base_backend import BaseBackend
from aisim_core.sdk.backends.vllm_backend_xpu import VLLMXPUBackend
from aisim_core.sdk.config import ModelConfig, RuntimeConfig
from aisim_core.sdk.models import get_model
from aisim_core.sdk.operations import Mamba2Kernel
from aisim_core.sdk.perf_database import get_database_view
from aisim_core.sdk.step_estimate import MixedStepInput

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--candidate-revision", default="HEAD")
args = parser.parse_args()
candidate = subprocess.check_output(["git", "rev-parse", args.candidate_revision], text=True).strip()

source_hashes = {}
for cls in (BaseBackend, VLLMXPUBackend):
    imported = Path(inspect.getsourcefile(cls))
    source_path = "python/aisim/src/aisim_core/sdk/backends/" + imported.name
    published = subprocess.check_output(["git", "show", f"{candidate}:{source_path}"])
    if imported.read_bytes() != published:
        raise ValueError(f"loaded {cls.__name__} differs from candidate {candidate}")
    source_hashes[source_path] = hashlib.sha256(published).hexdigest()
baseline = "25c424119adaa47ad6c748d982719384e8a16b04"
source = "python/aisim/src/aisim_core/sdk/backends/vllm_backend_xpu.py"
old = ModuleType("baseline_xpu")
exec(
    compile(
        subprocess.check_output(["git", "show", f"{baseline}:{source}"], text=True),
        f"{baseline}:{source}",
        "exec",
    ),
    old.__dict__,
)
database = get_database_view("b200_sxm", "vllm", "0.24.0", database_mode="SOL", allow_unlisted_version=True)
model = get_model(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, nextn=3),
    "vllm",
)
model.context_ops = [
    Mamba2Kernel(
        "context_scan",
        1,
        "mamba_chunk_scan_combined",
        "context",
        4096,
        64,
        64,
        128,
        4,
        8,
        128,
        state_dtype="float16",
    )
]
model.generation_ops = [
    Mamba2Kernel(
        "generation_scan",
        1,
        "selective_state_update",
        "generation",
        4096,
        64,
        64,
        128,
        4,
        8,
        128,
        state_dtype="float16",
        token_width=4,
    )
]
model._num_layers = 1
result = {
    "candidate": candidate,
    "candidate_sources_sha256": source_hashes,
    "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "baseline": baseline,
    "native_extension_sha256": hashlib.sha256(Path(native_runtime.__file__).read_bytes()).hexdigest(),
    "workload": {
        "system": "b200_sxm",
        "backend": "vllm",
        "backend_version": "0.24.0",
        "database_mode": "SOL",
        "gpu_mem_bw_bytes_per_second": database.system_spec["gpu"]["mem_bw"],
        "isl": 4096,
        "osl": 64,
        "batch": 4,
        "layers": 1,
    },
    "kind": "native SOL accounting predictions, not silicon accuracy or host performance",
    "comparison_boundary": "baseline XPU Python class versus candidate class, both using the candidate native engine",
    "reproduction_command": (
        "python benchmarks/evidence/mamba-mixed-step-accounting/scoped_xpu_width.py "
        f"--candidate-revision {candidate} OUT.json"
    ),
    "fixture": {
        "stub_returns": {
            "_max_kv_slots": 100,
            "_mix_step_gen_tokens": 2,
            "_effective_decode_bs": 4,
            "_get_genonly_step_latency": [1.0, 0.0, {}, {}, []],
            "_decode_tpot_ms": 1.5,
        },
        "compute_ttft_inputs": {
            "prefill_step_ms": 0,
            "genonly_step_latency_ms": 1,
            "encoder_latency_ms": 0,
            "steps_to_finish_ctx": 4,
            "decode_iterations": 63,
        },
        "native_fields": ["prefill_tokens", "decode_query_tokens", "mixed_latency_ms"],
        "derived_with_stubs": ["ttft_ms"],
    },
    "cases": [],
}
for prefix in (0, 3000):
    case = {"prefix": prefix, "budget": 514, "decoders": 2, "width": 4}
    for label, cls in [("before", old.VLLMXPUBackend), ("after", VLLMXPUBackend)]:
        backend = cls()
        backend._in_agg = True
        backend._max_kv_slots = lambda *a, **kw: 100
        backend._mix_step_gen_tokens = lambda *a: 2
        backend._effective_decode_bs = lambda *a: 4
        backend._get_genonly_step_latency = lambda *a: (1.0, 0.0, {}, {}, ())
        backend._decode_tpot_ms = lambda *a, **kw: 1.5
        runtime = RuntimeConfig(batch_size=4, isl=4096, osl=64, prefix=prefix)
        step = backend.run_mixed(model, database, runtime, MixedStepInput(514, 2))
        ttft = backend._compute_ttft(
            model,
            database,
            runtime,
            4,
            4096,
            64,
            514,
            prefix,
            prefill_step_ms=0,
            genonly_step_latency_ms=1,
            encoder_latency_ms=0,
            steps_to_finish_ctx=4,
            decode_iterations=63,
        )
        case[label] = {
            "prefill_tokens": step.context_tokens,
            "decode_query_tokens": step.num_decode_query_tokens,
            "mixed_latency_ms": step.latency_ms,
            "ttft_ms": ttft,
        }
    result["cases"].append(case)
for case in result["cases"]:
    assert case["before"]["prefill_tokens"] + case["before"]["decode_query_tokens"] == 520
    assert case["after"]["prefill_tokens"] + case["after"]["decode_query_tokens"] == case["budget"]
args.output.write_text(json.dumps(result, indent=2) + "\n")
