# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard-library launcher for the pinned engine container."""

import argparse
import json
import os
import sys
from pathlib import Path

TRTLLM_RUNTIME_VERSION = "1.3.0rc29"


def _engine(args):
    # Executed by the allocation's Pyxis step. CUDA enumeration verifies the
    # rank-visible GPUs before MPI starts, including when the job owns more.
    import importlib.metadata

    import torch

    if importlib.metadata.version("tensorrt_llm") != TRTLLM_RUNTIME_VERSION:
        raise RuntimeError(f"pilot requires TensorRT-LLM {TRTLLM_RUNTIME_VERSION}")
    devices = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    if len(devices) != 2 or any("H200" not in device.name for device in devices):
        raise RuntimeError("TRT-LLM pilot requires exactly two visible H200 GPUs")
    uuids = [str(device.uuid) for device in devices]
    uuids = [uuid if uuid.startswith("GPU-") else f"GPU-{uuid}" for uuid in uuids]
    Path(args.gpus).write_text(json.dumps({"uuids": uuids, "model": [device.name for device in devices]}))
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "tensorrt_llm.commands.serve",
            args.snapshot,
            "--grpc",
            "--grpc-protocol",
            "openengine",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--tp_size",
            "2",
            "--max_batch_size",
            "4",
            "--max_num_tokens",
            "8192",
            "--max_seq_len",
            "40960",
            "--free_gpu_memory_fraction",
            "0.85",
            "--extra_llm_api_options",
            args.options,
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, action="store_true")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--options", required=True)
    _engine(parser.parse_args())
