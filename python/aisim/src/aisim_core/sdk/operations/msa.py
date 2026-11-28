# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax Sparse Attention (MSA) module ops for MiniMax-M3.

MSA (github.com/MiniMax-AI/MSA) is structurally a GQA version of DSA: an indexer
does a cheap per-block "dense proxy" pass to score KV blocks, the top-k blocks
are selected, and full attention runs over only the selected tokens. Versus DSA
the main attention is standard GQA (not MLA-compressed), and the indexer scores
per *block* (block_size tokens) rather than per token.

MSA has its own module-level silicon tables (``msa_context_module_perf`` /
``msa_generation_module_perf``, DSA-module row schema; collected on the
serving path for trtllm/vllm/sglang across the mainstream systems). Rows are
keyed by the checkpoint's architecture (``MiniMaxM3ForCausalLM`` for M3,
``Qwen4ExpForConditionalGeneration`` for Qwen3.8-Flash-Next QSA); pass
``architecture=...`` when the op does not read the M3 slice. SILICON queries
resolve on those tables through the engine's interpolation with the
analytic MSA SOL (same three-group split as DSA/DSV4 — GEMM projections,
indexer, sparse attention) as the anchor. HYBRID / EMPIRICAL try the silicon
path first; on a typed data miss they fall back to the legacy CROSS-OP
TRANSFER from DSA's measured utilisation at the same workload, scaled by a
manual ``dsa_scale_k`` (util_scale hook): ``latency = SOL_msa /
(util_dsa * k)`` — the pre-silicon behaviour, kept for (backend, version)
tuples with no MSA tables (measured 15-28x below silicon on SM90; prefer
versions with collected data).

That transfer is calibrated on ``MiniMaxM3ForCausalLM`` only. An op built with
another ``architecture`` resolves that architecture's own rows and, when they
are absent, raises the terminal empirical miss instead of borrowing: the
borrowed util prices a different kernel, so the substitution is not a level
shift. Qwen3.8-Flash-Next (``Qwen4ExpForConditionalGeneration``) is the first
such consumer; SOL mode still answers analytically.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import aisim_core._native as _core
from aisim_core.sdk.operations.base import OpShellKit

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ContextMSAModule(_core.ContextMSAModule, OpShellKit):
    """Context (prefill) MSA. SILICON raises (no data); HYBRID/EMPIRICAL transfer from DSA.

    The transfer is M3-only: another ``architecture`` without its own rows
    raises the empirical miss instead of borrowing (see the module docstring).
    """


class GenerationMSAModule(_core.GenerationMSAModule, OpShellKit):
    """Generation (decode) MSA. s = total kv length.

    Same transfer rule as :class:`ContextMSAModule`: M3-only.
    """
