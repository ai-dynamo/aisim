# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The independent shared-token check must reject the old token count."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("substitute_old_count", [False, True])
def test_shared_token_check_exit_status(substitute_old_count):
    script = Path(__file__).with_name("shared_tokens.py")
    # Exercise main in a child process so a failed evidence check really has
    # a nonzero exit. Synthetic prices distinguish only token-major geometry.
    code = f"""
import runpy
module = runpy.run_path({str(script)!r})
def step(qualified, ctx):
    tokens = 7104 if ctx == 8128 and (not qualified or {substitute_old_count!r}) else 6144
    return {{'context_mamba_in_proj_gemm': tokens * 0.001, 'context_moe': tokens * 0.002,
             'context_attention (scaled)': ctx * 0.003}}
module['main'].__globals__['step'] = step
module['main']()
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert (result.returncode != 0) == substitute_old_count, result.stderr
    if substitute_old_count:
        assert "expected the 6144-token query" in result.stderr
