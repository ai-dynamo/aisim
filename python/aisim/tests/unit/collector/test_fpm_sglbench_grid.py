# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# The admission and graph-size models in these tests follow SGLang v0.5.22 and v0.5.23; see THIRD_PARTY_NOTICES.md.
# Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d
# Source: https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90
# The expected cuts, order and digest follow ai-dynamo/dynamo (Apache-2.0, https://github.com/ai-dynamo/dynamo),
# components/src/dynamo/vllm/instrumented_scheduler.py and backend_args.py at
# b20fe6fe94021b854597965c7305931175e8f482. See THIRD_PARTY_NOTICES.md.

"""CPU-only checks of ``sglbench_grid``: limits, feasibility, axes, grid, digest and points manifest.

No SGLang and no GPU: the tests run no engine and measure nothing. Expected values are worked out by hand in
the comments, enumerated by brute force, or taken from AISim's own code (the host-side axis mirror, the manifest
freezer and the artifact validator). The expected sample-cap defaults are the flag defaults of Dynamo's vLLM
self-benchmark. Where a comment names the engine probe, it says what a result file of
benchmarks/evidence/collector/sglbench-engine-probe-20261113 holds and what the case adds to that.

The last test reads SGLang's source and runs none of it. It needs a local checkout of SGLang for each release
the module names. ``AISIM_SGLBENCH_SGLANG_SOURCES`` names a folder that holds them as ``v<version>``, the variable
and the layout of ``test_fpm_sglbench_preflight.py``; without it that test is skipped.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import itertools
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest
from collector.fpm_forward.config import _dynamo_cudagraph_axis_points, _freeze_benchmark_points
from collector.fpm_forward.native_artifact import _expected_scheduled, _validate_seed_regime
from collector.fpm_forward.runtime import sglbench_grid as grid

pytestmark = pytest.mark.unit

COLLECTOR = Path(__file__).resolve().parents[3] / "collector"
RUNTIME = COLLECTOR / "fpm_forward/runtime"
MODULE_PATH = RUNTIME / "sglbench_grid.py"

# 1 is a page of one token, 16 equals the 16-token minimum of the cached unit and 64 is above it. 128 is the page
# size SGLang sets on an NPU when none is given (srt/hardware_backend/npu/utils.py ``set_default_server_args``)
# and 256 is the page size its DeepSeek V4 attention requires (srt/layers/attention/dsv4/metadata.py
# ``PagedIndexerMetadata``), both at v0.5.22 and v0.5.23. 24 does not divide 16; 3 is below 16 and does not
# divide 16, the only kind of page size where ceilP(16) and max(P, 16) differ.
PAGE_SIZES = (1, 3, 16, 24, 64, 128, 256)

NO_CAPS = grid.SampleCaps(
    prefill_max_new_token_samples=10_000,
    prefill_max_kv_read_token_samples=10_000,
    decode_max_kv_read_token_samples=10_000,
    decode_max_batch_size_samples=10_000,
    prefix_max_batch_size_samples=10_000,
)


def _limits(page_size: int = 16, *, chunked: bool = True, graphs: bool = True, **overrides) -> grid.Limits:
    """A small limits record at any page size; every limit binds somewhere in the grid it produces."""
    record = {
        "page_size": page_size,
        "max_req_input_len": 1500,
        "max_total_num_tokens": grid.floor_pages(40_000, page_size),
        "max_running_requests": 40,
        "max_prefill_tokens": 1024,
        "chunked_prefill_size": grid.floor_pages(600, page_size) if chunked else None,
        "decode_graph_sizes": (1, 2, 4, 8, 12, 16, 24) if graphs else (),
        "prefill_graph_sizes": (4, 8, 16, 32, 64, 128, 256, 512) if graphs else (),
    }
    record.update(overrides)
    return grid.Limits(**record)


def _prefill_fits(limits, batch, new, cached) -> bool:
    return grid.prefill_unfit_reason(limits, batch, new, cached) is None


def _decode_fits(limits, batch, context) -> bool:
    return grid.decode_unfit_reason(limits, batch, context) is None


def _brute_largest(candidates, fits):
    """Largest fitting candidate by trying every one, and whether the fitting ones are a prefix of the list."""
    candidates = list(candidates)
    fitting = [value for value in candidates if fits(value)]
    return (fitting[-1] if fitting else None), fitting == candidates[: len(fitting)]


def _coordinate(point: grid.Point) -> tuple:
    return (point.phase, point.batch_size, point.batch_size * point.new_tokens, point.batch_size * point.cached_tokens)


# ---------------------------------------------------------------------------------------------------------------
# Module hygiene: what the pod and the repository gates require of the file itself
# ---------------------------------------------------------------------------------------------------------------


def test_module_imports_in_a_bare_interpreter_and_plans_the_same_grid_there():
    """The pod imports the file by its flat name with no site-packages; the plan must not depend on the process."""
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import sglbench_grid as g; "
        "taboo = ('sglang', 'dynamo', 'aisim', 'aisim_core', 'collector', 'torch', 'numpy'); "
        "bad = [n for n in sys.modules if n.split('.')[0] in taboo]; "
        "bad += [n for n in sys.modules if n.startswith('sglbench_') and n != g.__name__]; "
        "assert not bad, bad; "
        "print(g.build_grid('agg', g.Limits(16, 1500, 39984, 40, 1024, 592, (1, 2, 4), (4, 8, 16))).digest)"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code, str(RUNTIME)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    here = grid.build_grid("agg", grid.Limits(16, 1500, 39984, 40, 1024, 592, (1, 2, 4), (4, 8, 16)))
    assert done.stdout.strip() == here.digest


def test_module_source_is_a_standard_library_leaf_that_reads_no_environment():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports do not resolve in the pod"
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Attribute):
            assert node.attr not in {"environ", "getenv", "putenv"}
        elif isinstance(node, ast.Name):
            # A dynamic import is seen by no import statement; the pod would meet it only when the call runs.
            assert node.id != "__import__", "dynamic import in a pod module"
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}
    assert "importlib" not in imported, "dynamic import in a pod module"
    assert imported.isdisjoint({"os", "sglang", "dynamo", "aisim", "aisim_core", "collector", "torch", "numpy"})
    assert not any(name.startswith("sglbench_") for name in imported)


def test_module_header_docstring_and_first_statement():
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert text.splitlines()[:11] == [
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team",
        "# SPDX-License-Identifier: Apache-2.0",
        (
            "# The feasibility rules and the first prefill graph step follow SGLang v0.5.22 and v0.5.23; "
            "see THIRD_PARTY_NOTICES.md."
        ),
        "# Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d",
        "# Source: https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90",
        (
            "# The sample caps, the axis rules, the points manifest and the order and digest of the grid are "
            "re-implemented from"
        ),
        "# ai-dynamo/dynamo (Apache-2.0, https://github.com/ai-dynamo/dynamo), components/src/dynamo/vllm/",
        (
            "# instrumented_scheduler.py, backend_args.py and benchmark_points.py at "
            "b20fe6fe94021b854597965c7305931175e8f482."
        ),
        "# See THIRD_PARTY_NOTICES.md.",
        "",
    ]
    tree = ast.parse(text)
    docstring = ast.get_docstring(tree)
    assert isinstance(tree.body[1], ast.ImportFrom) and tree.body[1].module == "__future__"
    for required in (
        "v0.5.22",
        "v0.5.23",
        "sgl-project/sglang",
        "Apache-2.0",
        "collector/fpm_forward/config.py",
        "_dynamo_cudagraph_axis_points",
        "collector/fpm_forward/runner.py",
        (
            "No upstream function or block of source is copied. What follows SGLang and Dynamo is named in the "
            "header of this\nfile and recorded in THIRD_PARTY_NOTICES.md."
        ),
    ):
        assert required in docstring


def test_constants():
    assert (grid.PREFILL, grid.DECODE, grid.AGG) == ("prefill", "decode", "agg")
    assert grid.MODES == ("prefill", "decode", "agg")
    assert grid.MIN_DECODE_CONTEXT == 4
    assert grid.DECODE_PROMPT_LEAD == 3
    assert grid.DECODE_OUTPUT_TOKENS == 7
    assert grid.PREFILL_OUTPUT_TOKENS == 1
    assert grid.CACHED_UNIT_MIN_TOKENS == 16
    assert grid.MANIFEST_SCHEMA_VERSIONS == (1, 2, 3)
    assert grid.STAMP_EXPLICIT == "explicit"
    assert grid.STAMP_PREFILL_REAL_SEED == "prefill_real_seed"
    assert grid.STAMP_DECODE_REAL_KV == "kvwarm_real_kv"
    # The lowest context is the shortest prompt (one token) plus the lead.
    assert grid.MIN_DECODE_CONTEXT == grid.DECODE_PROMPT_LEAD + 1


# ---------------------------------------------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------------------------------------------

VALID_LIMITS = {
    "page_size": 16,
    "max_req_input_len": 200,
    "max_total_num_tokens": 1600,
    "max_running_requests": 8,
    "max_prefill_tokens": 96,
    "chunked_prefill_size": 64,
    "decode_graph_sizes": [4, 1, 2, 4],
    "prefill_graph_sizes": (32, 16),
}


def test_limits_accepts_a_valid_record_and_normalizes_only_the_graph_sizes():
    limits = grid.Limits(**VALID_LIMITS)
    assert limits.decode_graph_sizes == (1, 2, 4)
    assert limits.prefill_graph_sizes == (16, 32)
    assert limits.chunked_prefill_size == 64
    assert grid.Limits(16, 200, 1600, 8, 96, None).chunked_prefill_size is None
    assert grid.Limits(16, 200, 1600, 8, 96, None).decode_graph_sizes == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        limits.page_size = 1
    with pytest.raises(TypeError):
        grid.Limits(16, 200, 1600, 8, 96)  # chunked_prefill_size has no default: "off" must be said, not assumed


@pytest.mark.parametrize("field", ["page_size", "max_req_input_len", "max_total_num_tokens", "max_running_requests"])
@pytest.mark.parametrize(
    ("value", "error"),
    [(0, ValueError), (-16, ValueError), (True, TypeError), (16.0, TypeError), ("16", TypeError), (None, TypeError)],
)
def test_limits_rejects_a_bad_scalar(field, value, error):
    with pytest.raises(error, match=field):
        grid.Limits(**{**VALID_LIMITS, field: value})


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"max_prefill_tokens": 0}, ValueError, "max_prefill_tokens"),
        ({"max_prefill_tokens": False}, TypeError, "max_prefill_tokens"),
        # SGLang's own "off" values are refused: the caller must pass None for "off".
        ({"chunked_prefill_size": 0}, ValueError, "chunked_prefill_size"),
        ({"chunked_prefill_size": -1}, ValueError, "chunked_prefill_size"),
        ({"chunked_prefill_size": True}, TypeError, "chunked_prefill_size"),
        ({"chunked_prefill_size": 64.0}, TypeError, "chunked_prefill_size"),
        # 72 is not a multiple of the page size 16; neither is 1608.
        ({"chunked_prefill_size": 72}, ValueError, "not a multiple of page_size"),
        ({"max_total_num_tokens": 1608}, ValueError, "not a multiple of page_size"),
        ({"decode_graph_sizes": [1, 0]}, ValueError, "decode_graph_sizes"),
        ({"decode_graph_sizes": [1, True]}, TypeError, "decode_graph_sizes"),
        ({"decode_graph_sizes": [1, 2.0]}, TypeError, "decode_graph_sizes"),
        ({"decode_graph_sizes": {1, 2}}, TypeError, "decode_graph_sizes"),
        ({"decode_graph_sizes": None}, TypeError, "decode_graph_sizes"),
        ({"prefill_graph_sizes": [-4]}, ValueError, "prefill_graph_sizes"),
        ({"prefill_graph_sizes": "48"}, TypeError, "prefill_graph_sizes"),
        ({"prefill_graph_sizes": 4}, TypeError, "prefill_graph_sizes"),
    ],
)
def test_limits_rejects_each_bad_input(overrides, error, match):
    with pytest.raises(error, match=match):
        grid.Limits(**{**VALID_LIMITS, **overrides})


def test_limits_mapping_round_trip_is_exact_and_json_ready():
    limits = grid.Limits(**VALID_LIMITS)
    record = limits.as_dict()
    assert record == {
        "page_size": 16,
        "max_req_input_len": 200,
        "max_total_num_tokens": 1600,
        "max_running_requests": 8,
        "max_prefill_tokens": 96,
        "chunked_prefill_size": 64,
        "decode_graph_sizes": [1, 2, 4],
        "prefill_graph_sizes": [16, 32],
    }
    assert grid.Limits.from_mapping(record) == limits
    assert grid.Limits.from_mapping(json.loads(json.dumps(record))) == limits
    unchunked = grid.Limits(1, 64, 4096, 8, 512, None)
    assert unchunked.as_dict()["chunked_prefill_size"] is None
    assert grid.Limits.from_mapping(json.loads(json.dumps(unchunked.as_dict()))) == unchunked


@pytest.mark.parametrize("missing", sorted(VALID_LIMITS))
def test_limits_from_mapping_requires_every_key(missing):
    record = {key: value for key, value in VALID_LIMITS.items() if key != missing}
    with pytest.raises(ValueError, match=f"missing.*{missing}"):
        grid.Limits.from_mapping(record)


def test_limits_from_mapping_rejects_unknown_keys_other_types_and_never_coerces():
    with pytest.raises(ValueError, match="unknown.*context_length"):
        grid.Limits.from_mapping({**VALID_LIMITS, "context_length": 4096})
    with pytest.raises(TypeError, match="mapping"):
        grid.Limits.from_mapping(list(VALID_LIMITS.items()))
    for value in ("16", 16.0, True, None):
        with pytest.raises(TypeError, match="page_size"):
            grid.Limits.from_mapping({**VALID_LIMITS, "page_size": value})


# ---------------------------------------------------------------------------------------------------------------
# Feasibility: hand-computed cases
# ---------------------------------------------------------------------------------------------------------------

# Record A: P = 16, I = 200, S = 1600, R = 8, M = 96, C = 64.
LIMITS_A = grid.Limits(16, 200, 1600, 8, 96, 64)
# Record B: as A with a small pool, S = 800.
LIMITS_B = grid.Limits(16, 200, 800, 8, 96, 64)
# Record C: page size 1, where the pool rule can be met with equality: P = 1, I = 50, S = 100, R = 10, M = 64.
LIMITS_C = grid.Limits(1, 50, 100, 10, 64, 32)
# Record D: as A with M = 48 (chunking on). Record E: the same with chunking off.
LIMITS_D = grid.Limits(16, 200, 1600, 8, 48, 64)
LIMITS_E = grid.Limits(16, 200, 1600, 8, 48, None)
# The six limits in the recorder headers of two runs of the engine probe, at page size 1 and at page size 64:
# ``headers[0]`` of results/v0522-q06-p1/results.json and of results/v0522-q06-p64/results.json (SGLang v0.5.22,
# Qwen3-0.6B, TP 1). The pool is different in the runs of the other tag and of the other model. No result file
# holds the GPU model; the README of the folder states it from the operator's record, and "H200" in the names
# below is that statement.
H200_PAGE_1 = grid.Limits(1, 40954, 1165358, 4096, 16384, 8192)
H200_PAGE_64 = grid.Limits(64, 40954, 1165312, 4096, 16384, 8192)


@pytest.mark.parametrize(
    ("limits", "point", "expected"),
    [
        # ceilP(1) = 16 <= 64; pool 1 * (0 + 16 + 16 + 1) = 33 < 1600.
        (LIMITS_A, (1, 1, 0), None),
        # 4 * 16 = 64 <= C with equality; (4 - 1) * 16 = 48 < M; pool 4 * 33 = 132.
        (LIMITS_A, (4, 16, 0), None),
        # 17 tokens round up to two pages: 4 * 32 = 128 > 64.
        (LIMITS_A, (4, 17, 0), "chunk_budget"),
        (LIMITS_A, (5, 16, 0), "chunk_budget"),
        # k = 160 = 10 pages; 160 + 32 = 192 < 200; 2 * 32 = 64 <= 64; pool 2 * (160 + 32 + 16 + 1) = 418.
        (LIMITS_A, (2, 32, 160), None),
        # 160 + 39 = 199 < 200 fits; 160 + 40 = 200 is not below I.
        (LIMITS_A, (1, 39, 160), None),
        (LIMITS_A, (1, 40, 160), "input_too_long"),
        (LIMITS_A, (1, 8, 8), "cached_not_page_aligned"),
        (LIMITS_A, (8, 1, 0), "chunk_budget"),  # 8 <= R, then 8 * 16 = 128 > 64
        (LIMITS_A, (9, 1, 0), "running_limit"),
        (LIMITS_A, (0, 1, 0), "invalid_shape"),
        (LIMITS_A, (1, 0, 0), "invalid_shape"),
        (LIMITS_A, (1, 1, -16), "invalid_shape"),
        # At page size 1 every integer is page-aligned, so only the shape rule can refuse k = -1.
        (LIMITS_C, (1, 1, -1), "invalid_shape"),
        # The first failing rule wins: each of these breaks its own rule and every later one that it can.
        (LIMITS_A, (0, 300, 7), "invalid_shape"),
        (LIMITS_A, (9, 300, 7), "cached_not_page_aligned"),
        (LIMITS_A, (9, 300, 16), "input_too_long"),
        (LIMITS_A, (9, 100, 16), "running_limit"),
        # Pool: 4 * (160 + 16 + 16 + 1) = 772 < 800 fits; 4 * (176 + 33) = 836 >= 800 does not.
        (LIMITS_B, (4, 16, 160), None),
        (LIMITS_B, (4, 16, 176), "kv_pool"),
        # Pool, strict: 4 * (17 + 5 + 1 + 1) = 96 < 100 fits; 4 * (18 + 7) = 100 is not below S.
        (LIMITS_C, (4, 5, 17), None),
        (LIMITS_C, (4, 5, 18), "kv_pool"),
        # max_prefill_tokens with chunking on: (b - 1) * 16 < 48 holds for b = 3 (32), not for b = 4 (48).
        (LIMITS_D, (3, 16, 0), None),
        (LIMITS_D, (4, 16, 0), "prefill_token_budget"),
        # With chunking on, a single request above the chunk budget is cut: ceilP(100) = 112 > 64.
        (LIMITS_D, (1, 100, 0), "chunk_budget"),
        # max_prefill_tokens with chunking off: b * 16 < 48 holds for b = 2 (32), not for b = 3 (48) ...
        (LIMITS_E, (2, 16, 0), None),
        (LIMITS_E, (3, 16, 0), "prefill_token_budget"),
        # ... and the first request is always admitted, even above M: ceilP(100) = 112 >= 48.
        (LIMITS_E, (1, 100, 0), None),
        (LIMITS_E, (2, 100, 0), "prefill_token_budget"),
        # The engine probe (check_9; SGLang v0.5.22 and v0.5.23, Qwen3-0.6B, TP 1, page sizes 1 and 64, C = 8192):
        # one call of 136 requests of 64 new tokens ran as an extend batch of 128 requests and 8192 tokens
        # (128 * 64 = 8192), then one of 8 requests and 512 tokens. 129 * 64 = 8256 > 8192 follows from the same
        # rule; no result file holds a call of 129 requests.
        (H200_PAGE_1, (128, 64, 0), None),
        (H200_PAGE_1, (129, 64, 0), "chunk_budget"),
        (H200_PAGE_1, (136, 64, 0), "chunk_budget"),
        (H200_PAGE_64, (128, 64, 0), None),
        (H200_PAGE_64, (129, 64, 0), "chunk_budget"),
    ],
)
def test_prefill_unfit_reason_hand_computed(limits, point, expected):
    assert grid.prefill_unfit_reason(limits, *point) == expected


@pytest.mark.parametrize(
    ("limits", "point", "expected"),
    [
        # Prompt of 1 token: ceilP(1) = 16; pool 1 * (16 + 16 + 7) = 39 < 1600.
        (LIMITS_A, (1, 4), None),
        (LIMITS_A, (1, 3), "context_below_minimum"),
        (LIMITS_A, (1, -5), "context_below_minimum"),
        (LIMITS_A, (0, 4), "invalid_shape"),
        (LIMITS_A, (0, 3), "invalid_shape"),
        # k = I = 200 is allowed: ceilP(197) = 208; pool 6 * (208 + 16 + 7) = 1386 < 1600, 7 * 231 = 1617 is not.
        (LIMITS_A, (6, 200), None),
        (LIMITS_A, (7, 200), "kv_pool"),
        (LIMITS_A, (1, 201), "input_too_long"),
        (LIMITS_A, (8, 4), None),
        (LIMITS_A, (9, 4), "running_limit"),
        # The first failing rule wins.
        (LIMITS_A, (9, 3), "context_below_minimum"),
        (LIMITS_A, (9, 201), "input_too_long"),
        # No chunk rule for decode: 8 prompts of 16 tokens need 8 * 16 = 128 > C = 64 page-rounded tokens, and
        # 8 * 16 = 128 >= M = 96, and still fit (pool 8 * 39 = 312).
        (LIMITS_A, (8, 19), None),
        # Pool, strict, at page size 1: 4 * ((24 - 3) + 1 + 7) = 116 < 120 fits; 4 * ((25 - 3) + 8) = 120 does not.
        (grid.Limits(1, 50, 120, 10, 64, 32), (4, 24), None),
        (grid.Limits(1, 50, 120, 10, 64, 32), (4, 25), "kv_pool"),
        # One request, page 64, a pool of ten pages. k = 515: the prompt of 512 tokens is eight whole pages and
        # 512 + 64 + 7 = 583 < 640. k = 516: the prompt of 513 tokens rounds up to 576 and 576 + 64 + 7 = 647.
        # SGLang's admission line alone would take k = 516 (one new token: 1 + 64 + 7 = 72 < 640 - 512), but
        # SGLang has already clipped its output to max(0, 640 - 576 - 64 - 1) = 0 tokens, so the run cannot have
        # its six decode batches.
        (grid.Limits(64, 100_000, 640, 8, 16384, 8192), (1, 515), None),
        (grid.Limits(64, 100_000, 640, 8, 16384, 8192), (1, 516), "kv_pool"),
        # The engine probe (check_9; SGLang v0.5.22 and v0.5.23, Qwen3-0.6B, TP 1, page size 64, C = 8192) recorded
        # a decode run of 384 requests as three extend batches of 128 and then six decode batches of 384. The
        # result files hold the mode and size of each batch and no context. The probe driver in that folder seeds
        # the prompts first and sets context 256 for this check, not the 1024 of this row. By the cache rule of
        # the module (``expected_extend_tokens``) a seeded request has 61 uncached tokens at either context
        # (253 - 192 and 1021 - 960), one page when rounded up, so the call charges 384 * 64 = 24576 tokens
        # against C, three times 8192.
        (H200_PAGE_64, (384, 1024), None),
    ],
)
def test_decode_unfit_reason_hand_computed(limits, point, expected):
    assert grid.decode_unfit_reason(limits, *point) == expected


def test_decode_input_rule_is_the_point_where_sglang_stops_clipping_the_output():
    """``max_req_input_len = max_req_len - 5``; SGLang clips the output to ``max_req_len - prompt - 1`` tokens."""
    limits = grid.Limits(1, 300, 100_000, 8, 512, None)
    max_req_len = limits.max_req_input_len + 5
    for context in range(4, 320):
        output_tokens = max(0, min(grid.DECODE_OUTPUT_TOKENS, max_req_len - (context - grid.DECODE_PROMPT_LEAD) - 1))
        all_seven_survive = output_tokens == grid.DECODE_OUTPUT_TOKENS
        assert _decode_fits(limits, 1, context) == all_seven_survive, context
    assert _decode_fits(limits, 1, 300) and not _decode_fits(limits, 1, 301)


@pytest.mark.parametrize("bad", [True, False, 2.0, "2", None, [2]])
def test_reason_functions_refuse_non_integers_and_do_not_raise_on_integers(bad):
    for arguments in ((bad, 1, 0), (1, bad, 0), (1, 1, bad)):
        with pytest.raises(TypeError):
            grid.prefill_unfit_reason(LIMITS_A, *arguments)
    for arguments in ((bad, 4), (1, bad)):
        with pytest.raises(TypeError):
            grid.decode_unfit_reason(LIMITS_A, *arguments)
    with pytest.raises(TypeError, match="Limits"):
        grid.prefill_unfit_reason(LIMITS_A.as_dict(), 1, 1, 0)
    with pytest.raises(TypeError, match="Limits"):
        grid.decode_unfit_reason(None, 1, 4)
    # Integers outside the shape are answered, not raised.
    assert grid.prefill_unfit_reason(LIMITS_A, -1, -1, -1) == "invalid_shape"
    assert grid.decode_unfit_reason(LIMITS_A, -1, -1) == "invalid_shape"


# ---------------------------------------------------------------------------------------------------------------
# Feasibility: an independent step-by-step model of one prefill pass
# ---------------------------------------------------------------------------------------------------------------


def _one_pass_takes_all(limits: grid.Limits, batch: int, new: int, cached: int) -> bool:
    """Walk ``batch`` identical requests through one prefill pass on an idle engine, one budget per limit.

    A test-side model of the pass itself (how each SGLang budget is charged and when a pass stops; the header of
    this file names what it follows). It does not use the closed-form inequalities under test. True when every
    request is taken into the one batch and none is cut in two.
    """
    page = limits.page_size
    paged = -(-new // page) * page  # SGLang charges new tokens in whole pages
    if cached + new >= limits.max_req_input_len:
        return False  # the request itself is refused
    # When a request arrives SGLang clips its output to what an empty pool leaves after the page-rounded prompt,
    # one page and one token. A request clipped below its one output token is not the request of the point.
    paged_prompt = -(-(cached + new) // page) * page
    if limits.max_total_num_tokens - paged_prompt - page - 1 < grid.PREFILL_OUTPUT_TOKENS:
        return False
    pool_left = limits.max_total_num_tokens  # free plus evictable tokens of an idle engine
    step_left = limits.max_prefill_tokens
    chunk_left = limits.chunked_prefill_size
    for taken in range(batch):  # ``taken`` requests are already in the batch
        if taken >= limits.max_running_requests:
            return False  # the batch is full
        if step_left <= 0 or (chunk_left is not None and chunk_left <= 0):
            return False  # a budget ran out with the previous request: the pass has ended
        pool_left -= cached  # the request's cached prefix is locked and can no longer be evicted
        if not new + limits.page_size + grid.PREFILL_OUTPUT_TOKENS < pool_left:
            return False  # the candidate needs its new tokens, one page and its output, strictly below the rest
        if chunk_left is None:
            if taken and paged >= step_left:
                return False  # chunking off: only the first request may exceed the step budget
        elif paged > chunk_left:
            return False  # chunking on: the request would be cut at the chunk budget
        pool_left -= paged + page + grid.PREFILL_OUTPUT_TOKENS  # reserved by the admitted request
        step_left -= paged
        if chunk_left is not None:
            chunk_left -= paged
    return True


def _random_limits(rng: random.Random, page: int) -> grid.Limits:
    """Small limits, so that every rule is the binding one for some point and brute force stays cheap."""
    chunk = None if rng.random() < 0.5 else page * rng.randint(1, 200 // page + 3)
    return grid.Limits(
        page_size=page,
        max_req_input_len=rng.randint(1, 90),
        max_total_num_tokens=page * rng.randint(1, 400 // page + 6),
        max_running_requests=rng.randint(1, 12),
        max_prefill_tokens=rng.randint(1, 220),
        chunked_prefill_size=chunk,
    )


@pytest.mark.parametrize("page", [1, 3, 16, 64])
def test_prefill_rules_agree_with_the_step_by_step_pass_model(page):
    rng = random.Random(1000 + page)
    exact = stricter = 0
    for _ in range(400):
        limits = _random_limits(rng, page)
        top = limits.max_req_input_len + 2
        for _ in range(6):
            # Half of the new-token counts are whole pages, where the coded pool rule is SGLang's exact line.
            new = page * rng.randint(1, top // page + 1) if rng.random() < 0.5 else rng.randint(1, top)
            cached = page * rng.randint(0, top // page)
            for batch in range(1, limits.max_running_requests + 2):
                reason = grid.prefill_unfit_reason(limits, batch, new, cached)
                model = _one_pass_takes_all(limits, batch, new, cached)
                if reason is None:
                    assert model, (limits, batch, new, cached)  # never plan a point SGLang would split
                elif model:
                    # The only place the coded rules are stricter is the pool rule for two or more requests, by
                    # less than a page. For one request the pool rule is SGLang's output clip, exactly.
                    assert reason == "kv_pool" and new % page and batch > 1, (limits, batch, new, cached, reason)
                    stricter += 1
                exact += reason is None
    assert exact > 1000  # the sample does contain fitting points
    # At page size 1 the two always agree; with larger pages the sample contains points that only the coded
    # rule refuses, each of them within one page of SGLang's exact pool line.
    assert (stricter > 0) == (page > 1)


def _decode_run_holds_all(limits: grid.Limits, batch: int, context: int) -> bool:
    """Whether ``batch`` decode requests of context ``context`` run as six decode batches that hold them all.

    A test-side model of the three things described below (the header of this file names what it follows). It
    does not use the closed-form inequality under test.

    Output clips: when a request arrives SGLang clips its output to ``max_req_len - prompt - 1`` tokens, with
    ``max_req_len = I + 5``, and to what an empty pool leaves after the page-rounded prompt, one page and one
    token. All seven output tokens must survive both.

    Admission, whatever the split into extend batches: ``r`` requests already running each hold ``ceilP(k - 3)``
    tokens and reserve up to 7 more; the ``j`` requests of the last extend batch each lock ``q = floorP(k - 4)``
    cached tokens and reserve ``ceilP(e) + P + 7`` with ``e = (k - 3) - q``; the last candidate needs
    ``e + P + 7`` strictly below what is left.

    Decode peak: at the last decode step every request holds ``ceilP(k + 3)`` tokens.
    """
    page = limits.page_size
    pool = limits.max_total_num_tokens
    prompt = context - 3
    if batch < 1 or prompt < 1 or batch > limits.max_running_requests:
        return False
    prompt_pages = -(-prompt // page) * page
    if min(limits.max_req_input_len + 5 - prompt - 1, pool - prompt_pages - page - 1) < 7:
        return False  # SGLang would clip the output: fewer than six decode batches
    cached = (context - 4) // page * page
    extend = prompt - cached
    extend_pages = -(-extend // page) * page
    assert cached + extend_pages == prompt_pages
    for last_batch in range(1, batch + 1):
        running = batch - last_batch
        held = running * (prompt_pages + 7) + last_batch * cached
        reserved = (last_batch - 1) * (extend_pages + page + 7)
        if not extend + page + 7 < pool - held - reserved:
            return False  # the last request of this split is not admitted
    return batch * (-(-(context + 3) // page) * page) < pool


@pytest.mark.parametrize("page", [1, 3, 16, 64])
def test_decode_pool_rule_covers_every_admission_line_the_output_clip_and_the_decode_peak(page):
    """A fitting decode point is admitted whatever the split, keeps its seven tokens and never runs out of pool."""
    rng = random.Random(2000 + page)
    checked = stricter = 0
    for _ in range(300):
        limits = _random_limits(rng, page)
        for batch in range(1, limits.max_running_requests + 1):
            for context in {4, 5, rng.randint(4, 95), rng.randint(4, 95), limits.max_req_input_len}:
                reason = grid.decode_unfit_reason(limits, batch, context)
                model = _decode_run_holds_all(limits, batch, context)
                if reason is None:
                    assert model, (limits, batch, context)  # never plan a run SGLang would cut short or split
                    checked += 1
                elif model:
                    # The only place the coded rules are stricter is the pool rule for two or more requests whose
                    # prompt is not whole pages, by less than a page. For one request the pool rule is SGLang's
                    # output clip, exactly.
                    assert reason == "kv_pool" and (context - 3) % page and batch > 1, (limits, batch, context)
                    stricter += 1
    assert checked > 100
    assert (stricter > 0) == (page > 1)


# ---------------------------------------------------------------------------------------------------------------
# Feasibility: every inequality at its boundary, and the closed forms against brute force
# ---------------------------------------------------------------------------------------------------------------


def test_each_prefill_inequality_at_its_boundary():
    # Roomy limits, so that only the rule under test can bind. P = 8.
    roomy = {
        "page_size": 8,
        "max_req_input_len": 10_000,
        "max_total_num_tokens": 8_000_000,
        "max_running_requests": 1000,
        "max_prefill_tokens": 1_000_000,
        "chunked_prefill_size": None,
    }

    def reason(point, **overrides):
        return grid.prefill_unfit_reason(grid.Limits(**{**roomy, **overrides}), *point)

    # k + n < I: 64 + 35 = 99 fits I = 100, 64 + 36 = 100 does not.
    assert reason((1, 35, 64), max_req_input_len=100) is None
    assert reason((1, 36, 64), max_req_input_len=100) == "input_too_long"
    # b <= R.
    assert reason((5, 1, 0), max_running_requests=5) is None
    assert reason((6, 1, 0), max_running_requests=5) == "running_limit"
    # b * ceilP(n) <= C with equality: 5 * 16 = 80. One more token rounds up to 24 and 5 * 24 = 120.
    assert reason((5, 16, 0), chunked_prefill_size=80) is None
    assert reason((5, 17, 0), chunked_prefill_size=80) == "chunk_budget"
    assert reason((6, 16, 0), chunked_prefill_size=80) == "chunk_budget"
    # Chunking on, (b - 1) * ceilP(n) < M: with M = 33, b = 3 gives 32 and b = 4 gives 48; with M = 32, b = 3
    # gives 32, which is not below 32.
    assert reason((3, 16, 0), chunked_prefill_size=800, max_prefill_tokens=33) is None
    assert reason((4, 16, 0), chunked_prefill_size=800, max_prefill_tokens=33) == "prefill_token_budget"
    assert reason((3, 16, 0), chunked_prefill_size=800, max_prefill_tokens=32) == "prefill_token_budget"
    assert reason((2, 16, 0), chunked_prefill_size=800, max_prefill_tokens=32) is None
    # Chunking off, b == 1 or b * ceilP(n) < M: with M = 49, b = 3 gives 48; with M = 48 it does not fit.
    assert reason((3, 16, 0), max_prefill_tokens=49) is None
    assert reason((3, 16, 0), max_prefill_tokens=48) == "prefill_token_budget"
    assert reason((1, 16, 0), max_prefill_tokens=1) is None
    # b * (k + ceilP(n) + P + 1) < S: 3 * (24 + 16 + 8 + 1) = 147. S is a page multiple: 152 fits, 144 does not.
    assert reason((3, 16, 24), max_total_num_tokens=152) is None
    assert reason((3, 16, 24), max_total_num_tokens=144) == "kv_pool"
    # The same rule at page size 1, where S can sit exactly on the product: 3 * (24 + 16 + 1 + 1) = 126.
    assert reason((3, 16, 24), page_size=1, max_total_num_tokens=127) is None
    assert reason((3, 16, 24), page_size=1, max_total_num_tokens=126) == "kv_pool"


def test_each_decode_inequality_at_its_boundary():
    roomy = {
        "page_size": 8,
        "max_req_input_len": 10_000,
        "max_total_num_tokens": 8_000_000,
        "max_running_requests": 1000,
        "max_prefill_tokens": 1,
        "chunked_prefill_size": 8,
    }

    def reason(point, **overrides):
        return grid.decode_unfit_reason(grid.Limits(**{**roomy, **overrides}), *point)

    assert reason((1, 4)) is None
    assert reason((1, 3)) == "context_below_minimum"
    assert reason((1, 100), max_req_input_len=100) is None
    assert reason((1, 101), max_req_input_len=100) == "input_too_long"
    assert reason((5, 4), max_running_requests=5) is None
    assert reason((6, 4), max_running_requests=5) == "running_limit"
    # b * (ceilP(k - 3) + P + 7) < S: k = 20 gives ceilP(17) = 24, so 3 * (24 + 8 + 7) = 117; k = 19 gives
    # ceilP(16) = 16, so 3 * 31 = 93. S = 96 holds k = 19 and not k = 20; S = 120 holds both.
    assert reason((3, 19), max_total_num_tokens=96) is None
    assert reason((3, 20), max_total_num_tokens=96) == "kv_pool"
    assert reason((3, 20), max_total_num_tokens=120) is None
    # Page size 1: 3 * ((20 - 3) + 1 + 7) = 75.
    assert reason((3, 20), page_size=1, max_total_num_tokens=76) is None
    assert reason((3, 20), page_size=1, max_total_num_tokens=75) == "kv_pool"


@pytest.mark.parametrize("page", [1, 3, 16, 24, 64])
def test_largest_values_equal_brute_force_on_small_limits(page):
    """Each closed form names the largest fitting value: the value fits and no larger candidate does.

    Nothing with ``n >= I``, ``k > I`` or ``b > R`` can fit, so trying every candidate a little beyond those
    bounds is a complete enumeration.
    """
    rng = random.Random(3000 + page)
    unit = grid.cached_unit(page)
    seen = {"n": 0, "b": 0, "k": 0, "decode_b": 0, "decode_k": 0, "none": 0}
    for _ in range(400):
        limits = _random_limits(rng, page)
        new_range = range(1, limits.max_req_input_len + 4)
        batch_range = range(1, limits.max_running_requests + 4)
        cached_range = range(0, limits.max_req_input_len + 2 * unit, unit)
        context_range = range(4, limits.max_req_input_len + 4)

        n_limit, prefix = _brute_largest(new_range, lambda new: _prefill_fits(limits, 1, new, 0))
        assert prefix and grid.largest_prefill_new_tokens(limits) == n_limit, limits
        seen["n" if n_limit else "none"] += 1

        for new in {1, n_limit or 1, (n_limit or 0) + 1, rng.choice(new_range), rng.choice(new_range)}:
            b_max, prefix = _brute_largest(batch_range, lambda batch: _prefill_fits(limits, batch, new, 0))
            assert prefix and grid.largest_prefill_batch(limits, new) == b_max, (limits, new)
            seen["b" if b_max else "none"] += 1
            for batch in {1, b_max or 1, (b_max or 0) + 1, rng.choice(batch_range)}:
                k_max, prefix = _brute_largest(cached_range, lambda cached: _prefill_fits(limits, batch, new, cached))
                assert prefix and grid.largest_prefill_cached(limits, batch, new) == k_max, (limits, batch, new)
                seen["k" if k_max is not None else "none"] += 1

        b_limit, prefix = _brute_largest(batch_range, lambda batch: _decode_fits(limits, batch, 4))
        assert prefix and grid.largest_decode_batch(limits) == b_limit, limits
        seen["decode_b" if b_limit else "none"] += 1
        for batch in {1, b_limit or 1, (b_limit or 0) + 1, rng.choice(batch_range)}:
            k_max, prefix = _brute_largest(context_range, lambda context: _decode_fits(limits, batch, context))
            assert prefix and grid.largest_decode_context(limits, batch) == k_max, (limits, batch)
            seen["decode_k" if k_max else "none"] += 1
    # The sample exercises every search, and the "nothing fits" answer too.
    assert all(count > 20 for count in seen.values()), seen


def test_no_decode_point_fits_below_a_four_token_input_limit():
    """With I < 4 the lowest context is already too long, whatever the pool allows."""
    for input_limit in (1, 2, 3):
        limits = grid.Limits(1, input_limit, 70, 3, 62, None)
        assert grid.largest_decode_batch(limits) is None
        assert grid.decode_batch_axis(limits, grid.SampleCaps()) == []
        assert grid.build_grid("decode", limits).missing_phases == ("decode",)
    assert grid.largest_decode_batch(grid.Limits(1, 4, 70, 3, 62, None)) == 3


def test_largest_value_search_raises_on_a_wrong_closed_form_and_never_repairs_it():
    def fits(value):
        return value <= 7

    assert grid._largest_fitting("exact", 1, 1, lambda: 7, fits) == 7
    assert grid._largest_fitting("exact", 0, 7, lambda: 7, fits) == 7
    assert grid._largest_fitting("none", 8, 1, lambda: 99, fits) is None  # the smallest candidate does not fit
    for wrong in (6, 8, 0, -3):  # too small, too large, below the minimum
        with pytest.raises(RuntimeError, match="broken closed form"):
            grid._largest_fitting("broken closed form", 1, 1, lambda wrong=wrong: wrong, fits)
    with pytest.raises(RuntimeError, match="off the lattice"):
        grid._largest_fitting("off the lattice", 0, 4, lambda: 6, fits)  # 6 is not a multiple of the step


@pytest.mark.parametrize("page", range(1, 41))
def test_cached_unit_is_the_smallest_page_multiple_that_holds_sixteen_tokens(page):
    unit = grid.cached_unit(page)
    assert unit == grid.ceil_pages(16, page)
    assert unit % page == 0 and unit >= 16 and unit - page < 16
    # ceilP(16) differs from max(P, 16) exactly for a page size below 16 that does not divide 16.
    assert (unit != max(page, 16)) == (page < 16 and 16 % page != 0)


def test_page_helpers():
    assert [grid.ceil_pages(tokens, 16) for tokens in (0, 1, 16, 17, 32)] == [0, 16, 16, 32, 32]
    assert [grid.floor_pages(tokens, 16) for tokens in (0, 1, 15, 16, 17, 33)] == [0, 0, 0, 16, 16, 32]
    assert grid.ceil_pages(1021, 1) == 1021 and grid.floor_pages(1021, 1) == 1021
    assert (grid.cached_unit(1), grid.cached_unit(3), grid.cached_unit(12), grid.cached_unit(24)) == (16, 18, 24, 24)
    assert (grid.cached_unit(16), grid.cached_unit(64), grid.cached_unit(256)) == (16, 64, 256)
    for bad_page in (0, -1):
        with pytest.raises(ValueError, match="page_size"):
            grid.ceil_pages(5, bad_page)
        with pytest.raises(ValueError, match="page_size"):
            grid.floor_pages(5, bad_page)
    for arguments in ((True, 4), (5, True), (5.0, 4), (5, "4")):
        with pytest.raises(TypeError):
            grid.ceil_pages(*arguments)
        with pytest.raises(TypeError):
            grid.floor_pages(*arguments)


def test_expected_cached_and_extend_tokens_match_the_engine_probe():
    # The engine probe (check_3; SGLang v0.5.22 and v0.5.23, Qwen3-0.6B and Qwen3-8B): two requests with a prompt
    # of 1021 tokens (context 1024), seeded first, each reported 960 cached tokens at page size 64 and 1020 at
    # page size 1. Their extend batch computed 122 tokens at page size 64 and 2 at page size 1: 61 and 1 for each.
    assert grid.expected_cached_tokens("decode", 1024, 64) == 960
    assert grid.expected_extend_tokens("decode", 0, 1024, 64) == 61
    assert grid.expected_cached_tokens("decode", 1024, 1) == 1020
    assert grid.expected_extend_tokens("decode", 0, 1024, 1) == 1
    # The same check: a prefill prefix of whole pages was reported in full (128 of 128 tokens at page size 1, 512 of
    # 512 at page size 64).
    assert grid.expected_cached_tokens("prefill", 128, 1) == 128
    assert grid.expected_cached_tokens("prefill", 512, 64) == 512
    # Not from the probe: a prefill request without a cached prefix, and the new tokens of a prefill request.
    assert grid.expected_cached_tokens("prefill", 0, 64) == 0
    assert grid.expected_extend_tokens("prefill", 37, 512, 64) == 37


@pytest.mark.parametrize("page", PAGE_SIZES)
def test_decode_expected_tokens_split_the_prompt_into_whole_cached_pages_and_a_remainder(page):
    for context in range(4, 4 + 3 * page + 40):
        prompt = context - 3
        cached = grid.expected_cached_tokens("decode", context, page)
        extend = grid.expected_extend_tokens("decode", 0, context, page)
        assert cached + extend == prompt
        assert cached % page == 0 and cached <= prompt - 1  # whole pages; one token is always recomputed
        assert 1 <= extend <= page


def test_expected_tokens_refuse_inputs_that_are_not_a_staged_point():
    with pytest.raises(ValueError, match="phase"):
        grid.expected_cached_tokens("agg", 16, 16)
    with pytest.raises(ValueError, match="phase"):
        grid.expected_extend_tokens("agg", 1, 16, 16)
    with pytest.raises(ValueError, match="multiple of page_size"):
        grid.expected_cached_tokens("prefill", 24, 16)
    with pytest.raises(ValueError, match="multiple of page_size"):
        grid.expected_cached_tokens("prefill", -16, 16)
    with pytest.raises(ValueError, match="non-negative multiple of page_size 1: -1"):
        grid.expected_cached_tokens("prefill", -1, 1)  # every integer is a multiple of 1; this one is negative
    with pytest.raises(ValueError, match="decode context"):
        grid.expected_cached_tokens("decode", 3, 16)
    with pytest.raises(ValueError, match="new_tokens"):
        grid.expected_extend_tokens("prefill", 0, 16, 16)
    with pytest.raises(ValueError, match="new_tokens must be 0"):
        grid.expected_extend_tokens("decode", 5, 1024, 16)
    with pytest.raises(ValueError, match="page_size"):
        grid.expected_cached_tokens("prefill", 0, 0)
    for arguments in (("prefill", True, 16), ("decode", 8.0, 16), ("prefill", 16, True)):
        with pytest.raises(TypeError):
            grid.expected_cached_tokens(*arguments)
    with pytest.raises(TypeError):
        grid.expected_extend_tokens("prefill", True, 16, 16)


# ---------------------------------------------------------------------------------------------------------------
# Axis primitives
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sizes", "limit", "expected"),
    [
        # Each size and its successor, the doubling tail 20 (40 is past the limit), the limit.
        ((3, 10), 25, [3, 4, 10, 11, 20, 25]),
        # The limit equals a size: no successor for it; the tail would start at 20.
        ((3, 10), 10, [3, 4, 10]),
        # The limit equals a size plus one.
        ((3, 10), 11, [3, 4, 10, 11]),
        # The limit is below the largest size: no tail.
        ((3, 10), 7, [3, 4, 7]),
        # The limit is below every size.
        ((3, 10), 2, [2]),
        ((1, 2), 1, [1]),
        ((), 1, [1]),
        # No sizes: powers of two and the limit.
        ((), 12, [1, 2, 4, 8, 12]),
        ((), 8, [1, 2, 4, 8]),
        # Entries below 1 are not sizes; repeats count once.
        ((0, -2, 5), 6, [5, 6]),
        ((0,), 5, [1, 2, 4, 5]),
        ((5, 5), 21, [5, 6, 10, 20, 21]),
        # The tail stops strictly below the limit, which is then added once.
        ((5,), 20, [5, 6, 10, 20]),
        # No values at all below 1.
        ((3, 10), 0, []),
        ((), 0, []),
        ((4,), -5, []),
    ],
)
def test_graph_axis_hand_worked(sizes, limit, expected):
    assert grid.graph_axis(sizes, limit) == expected


def test_graph_axis_equals_the_collector_host_mirror_for_every_limit_from_one():
    """``_dynamo_cudagraph_axis_points`` is what AISim's planner uses on the host; the pod must agree with it."""
    rng = random.Random(4000)
    vectors = [
        ((), 1),
        ((), 7),
        ((), 8),
        ((5, 9), 3),  # limit below every size
        ((5, 9), 5),  # limit equal to a size
        ((5, 9), 6),  # limit equal to a size + 1
        ((5, 9), 9),
        ((5, 9), 10),
        ((5, 9), 18),  # limit equal to twice the largest size
        ((5, 9), 19),
        ((1,), 1),
        ((1, 2, 4), 1),
        ((0, -3), 5),
        ((3, 3, 3), 100),
    ]
    for _ in range(4000):
        sizes = tuple(rng.randint(-2, 60) for _ in range(rng.randint(0, 8)))
        vectors.append((sizes, rng.randint(1, 200)))
    for sizes, limit in vectors:
        assert grid.graph_axis(sizes, limit) == list(_dynamo_cudagraph_axis_points(tuple(sizes), limit)), (sizes, limit)
    # Below 1 the pod returns no values. The host mirror returns the limit itself there and is never called so.
    for sizes, limit in (((), 0), ((4,), 0), ((4,), -5)):
        assert grid.graph_axis(sizes, limit) == []


def test_graph_axis_refuses_non_integers():
    for arguments in (((1, 2), True), ((1, 2), 4.0), ((1, True), 4), ((1, 2.0), 4), (("1",), 4)):
        with pytest.raises(TypeError):
            grid.graph_axis(*arguments)


@pytest.mark.parametrize(
    ("values", "max_samples", "expected"),
    [
        # 11 values, 4 samples: positions 0, 10/3, 20/3, 10 round to indices 0, 3, 7, 10.
        (list(range(10, 21)), 4, [10, 13, 17, 20]),
        # Two samples are the two ends.
        ([1, 2, 3, 4, 5, 6, 7], 2, [1, 7]),
        # 7 values, 3 samples: positions 0, 3, 6.
        ([0, 16, 32, 64, 128, 256, 300], 3, [0, 64, 300]),
        # 8 values, 3 samples: position 3.5 rounds half up to index 4.
        (list(range(8)), 3, [0, 4, 7]),
        # 6 values, 5 samples: positions 0, 1.25, 2.5, 3.75, 5 round to 0, 1, 3, 4, 5.
        (list(range(6)), 5, [0, 1, 3, 4, 5]),
        # Not more values than samples: unchanged.
        ([4, 8, 9], 3, [4, 8, 9]),
        ([4, 8, 9], 64, [4, 8, 9]),
        ([], 2, []),
    ],
)
def test_limit_uniform_hand_worked(values, max_samples, expected):
    assert grid.limit_uniform(values, max_samples) == expected


def test_limit_uniform_keeps_both_ends_and_exactly_max_samples_increasing_values():
    for length in range(3, 140):
        values = [3 * index + 1 for index in range(length)]
        for max_samples in range(2, length):
            kept = grid.limit_uniform(values, max_samples)
            assert len(kept) == max_samples
            assert kept[0] == values[0] and kept[-1] == values[-1]
            assert all(low < high for low, high in itertools.pairwise(kept))
            assert set(kept) <= set(values)


@pytest.mark.parametrize("max_samples", [1, 0, -2])
def test_axis_caps_below_two_raise(max_samples):
    with pytest.raises(ValueError, match="at least 2"):
        grid.limit_uniform([1, 2, 3], max_samples)
    with pytest.raises(ValueError, match="at least 2"):
        grid.limit_graph_axis([1, 2, 3], [2], max_samples)
    with pytest.raises(ValueError, match="at least 2"):
        grid.limit_uniform([], max_samples)  # the cap is checked even when nothing would be cut
    with pytest.raises(ValueError, match="at least 2"):
        grid.limit_graph_axis([7], [2], max_samples)  # the same for the graph cut, which returns early here


def test_axis_caps_must_be_integers():
    for bad in (True, 2.0, "2", None):
        with pytest.raises(TypeError):
            grid.limit_uniform([1, 2, 3], bad)
        with pytest.raises(TypeError):
            grid.limit_graph_axis([1, 2, 3], [2], bad)
    with pytest.raises(TypeError):
        grid.limit_graph_axis([1, 2, 3], [True], 2)


@pytest.mark.parametrize(
    ("values", "sizes", "max_samples", "expected"),
    [
        # Tail protection: 12 values, tail [20, 40] is 2 * 5 = 10 <= 12. Budget 5 - 2 = 3 for the graph part
        # 1..10 gives indices 0, 4.5 -> 5, 9, so 1, 6, 10; then the whole tail.
        ([*range(1, 11), 20, 40], [10], 5, [1, 6, 10, 20, 40]),
        # Budget exactly 1: only the first graph value, then the tail (10 values, tail 2 * 5 = 10 <= 10).
        ([*range(1, 9), 16, 32], [8], 3, [1, 16, 32]),
        # Budget 0: no protection; a uniform cut of the whole axis to its two ends.
        ([*range(1, 9), 16, 32], [8], 2, [1, 32]),
        # The tail is more than a fifth of the axis (5 * 5 = 25 > 8): a uniform cut. 8 values, 4 samples:
        # positions 0, 7/3, 14/3, 7 round to indices 0, 2, 5, 7.
        ([2, 3, 6, 7, 12, 24, 48, 50], [2, 6], 4, [2, 6, 24, 50]),
        # No value at or below the largest size: a uniform cut. 6 values, 3 samples: indices 0, 2.5 -> 3, 5.
        ([5, 6, 7, 8, 9, 10], [2], 3, [5, 8, 10]),
        # No tail (every value is at or below the largest size): a uniform cut.
        ([5, 6, 7, 8, 9, 10], [64], 3, [5, 8, 10]),
        # No graph size at all, or none that is at least 1: a uniform cut.
        ([5, 6, 7, 8, 9, 10], [], 3, [5, 8, 10]),
        ([5, 6, 7, 8, 9, 10], [0, -1], 3, [5, 8, 10]),
        # Not more values than samples: unchanged, whatever the sizes.
        ([1, 2, 3, 20, 40], [3], 5, [1, 2, 3, 20, 40]),
    ],
)
def test_limit_graph_axis_hand_worked(values, sizes, max_samples, expected):
    assert grid.limit_graph_axis(values, sizes, max_samples) == expected


def test_limit_graph_axis_without_protection_would_lose_the_tail():
    """The first hand-worked case, cut uniformly instead: 12 values, 5 samples keep indices 0, 3, 6, 8, 11."""
    values = [*range(1, 11), 20, 40]
    assert grid.limit_uniform(values, 5) == [1, 4, 7, 9, 40]
    assert 20 in grid.limit_graph_axis(values, [10], 5)


def test_limit_graph_axis_properties_on_generated_axes():
    rng = random.Random(5000)
    protected = 0
    for _ in range(3000):
        sizes = sorted({rng.randint(1, 60) for _ in range(rng.randint(1, 25))})
        # Limits just above the largest size give the short tails that the rule protects.
        limit = rng.choice([sizes[-1] + rng.randint(1, 3), 2 * sizes[-1] + 1, rng.randint(sizes[0], 400)])
        values = grid.graph_axis(sizes, limit)
        max_samples = rng.randint(2, 12)
        kept = grid.limit_graph_axis(values, sizes, max_samples)
        assert len(kept) == min(len(values), max_samples)
        assert kept[0] == values[0] and kept[-1] == values[-1]
        assert all(low < high for low, high in itertools.pairwise(kept))
        assert set(kept) <= set(values)
        tail = [value for value in values if value > sizes[-1]]
        if len(values) > max_samples and tail and len(tail) * 5 <= len(values) and max_samples > len(tail):
            # The whole unpadded tail survives whenever it is at most a fifth of the axis and leaves room for
            # one graph value (every axis here has a value at or below its largest size).
            assert kept[-len(tail) :] == tail
            protected += 1
    assert protected > 100


# ---------------------------------------------------------------------------------------------------------------
# Sample caps
# ---------------------------------------------------------------------------------------------------------------


def test_sample_cap_defaults_are_the_dynamo_flag_defaults():
    # The defaults of the six flags of Dynamo's vLLM self-benchmark (components/src/dynamo/vllm/backend_args.py at
    # the revision the header of this file names): 64, 16, 128, 128, 3 and no batch cap.
    caps = grid.SampleCaps()
    assert dataclasses.asdict(caps) == {
        "prefill_max_new_token_samples": 64,
        "prefill_max_kv_read_token_samples": 16,
        "decode_max_kv_read_token_samples": 128,
        "decode_max_batch_size_samples": 128,
        "prefix_max_batch_size_samples": 3,
        "benchmark_max_batch_size": None,
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        caps.prefix_max_batch_size_samples = 1


def test_sample_cap_names_are_the_flags_aisim_renders_for_the_vllm_self_benchmark():
    runner_source = (COLLECTOR / "fpm_forward/runner.py").read_text(encoding="utf-8")
    for field in dataclasses.fields(grid.SampleCaps):
        if field.name != "benchmark_max_batch_size":  # AISim does not pass this flag today
            assert f'"--{field.name.replace("_", "-")}"' in runner_source, field.name


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"prefill_max_new_token_samples": 1}, ValueError),
        ({"prefill_max_kv_read_token_samples": 1}, ValueError),
        ({"decode_max_kv_read_token_samples": 0}, ValueError),
        ({"decode_max_batch_size_samples": 1}, ValueError),
        ({"prefix_max_batch_size_samples": 0}, ValueError),
        ({"benchmark_max_batch_size": 0}, ValueError),
        ({"prefill_max_new_token_samples": True}, TypeError),
        ({"prefill_max_kv_read_token_samples": 2.0}, TypeError),
        ({"decode_max_kv_read_token_samples": "128"}, TypeError),
        ({"decode_max_batch_size_samples": None}, TypeError),
        ({"prefix_max_batch_size_samples": True}, TypeError),
        ({"benchmark_max_batch_size": True}, TypeError),
        ({"benchmark_max_batch_size": 4.0}, TypeError),
    ],
)
def test_sample_caps_reject_each_bad_input(overrides, error):
    with pytest.raises(error, match=next(iter(overrides))):
        grid.SampleCaps(**overrides)
    assert grid.SampleCaps(prefix_max_batch_size_samples=1, benchmark_max_batch_size=1).benchmark_max_batch_size == 1


def _axes(points) -> dict:
    """The axes a list of points was built from, recovered from the points alone."""
    axes = {"new": set(), "batches": {}, "cached": {}, "decode_batches": set(), "contexts": {}}
    for point in points:
        if point.phase == "prefill":
            axes["new"].add(point.new_tokens)
            axes["batches"].setdefault(point.new_tokens, set()).add(point.batch_size)
            axes["cached"].setdefault((point.new_tokens, point.batch_size), []).append(point.cached_tokens)
        else:
            axes["decode_batches"].add(point.batch_size)
            axes["contexts"].setdefault(point.batch_size, []).append(point.cached_tokens)
    return axes


def _decode_shapes(points) -> list[tuple]:
    return [(point.batch_size, point.cached_tokens) for point in points if point.phase == "decode"]


def _prefill_shapes(points) -> list[tuple]:
    return [(point.batch_size, point.new_tokens, point.cached_tokens) for point in points if point.phase == "prefill"]


def test_each_cap_changes_only_its_own_axis():
    limits = _limits(16)
    base = grid.build_grid("agg", limits, NO_CAPS).points
    full = _axes(base)
    assert len(full["new"]) > 8 and len(full["decode_batches"]) > 8  # there is something to cut

    def capped(**overrides):
        points = grid.build_grid("agg", limits, dataclasses.replace(NO_CAPS, **overrides)).points
        return points, _axes(points)

    # New-token cap: five values of the graph axis, both ends kept, and in front of them the low end, 1 and 2,
    # which does not count against the cap (the prefill ladder starts at 4, so a batch of two tokens is the
    # first that runs with a graph). The graph axis has 17 values: both sides of the eight sizes 4 to 512, and
    # the limit 592. The tail above 512 is [513, 592], 2 * 5 = 10 <= 17 values, so it is kept and three samples
    # are left for the 15 values up to 512: indices 0, 7 and 14, the values 4, 33 and 512. Batches and cached
    # prefixes of the surviving n are unchanged.
    points, axes = capped(prefill_max_new_token_samples=5)
    assert sorted(axes["new"]) == [1, 2, 4, 33, 512, 513, 592] and axes["new"] <= full["new"]
    assert {min(axes["new"]), max(axes["new"])} == {min(full["new"]), max(full["new"])}
    assert axes["batches"] == {new: full["batches"][new] for new in axes["new"]}
    assert axes["cached"] == {key: value for key, value in full["cached"].items() if key[0] in axes["new"]}
    assert _decode_shapes(points) == _decode_shapes(base)

    # Prefill batch cap ("prefix" in the flag name): the FIRST values, here the smallest one only.
    points, axes = capped(prefix_max_batch_size_samples=1)
    assert axes["new"] == full["new"]
    assert axes["batches"] == {new: {min(batches)} for new, batches in full["batches"].items()}
    assert all(batches == {1} for batches in axes["batches"].values())
    assert axes["cached"] == {key: value for key, value in full["cached"].items() if key[1] == 1}
    assert _decode_shapes(points) == _decode_shapes(base)

    # Cached-token cap: at most three per (n, b), 0 and the maximum always kept.
    points, axes = capped(prefill_max_kv_read_token_samples=3)
    assert axes["new"] == full["new"] and axes["batches"] == full["batches"]
    assert set(axes["cached"]) == set(full["cached"])
    assert any(len(values) > 3 for values in full["cached"].values())
    for key, values in axes["cached"].items():
        assert len(values) == min(3, len(full["cached"][key]))
        assert values[0] == full["cached"][key][0] and values[-1] == 0  # descending: the maximum, then 0 last
        assert set(values) <= set(full["cached"][key])
    assert _decode_shapes(points) == _decode_shapes(base)

    # Decode batch cap: four values, both ends kept; contexts of the surviving batches unchanged.
    points, axes = capped(decode_max_batch_size_samples=4)
    assert _prefill_shapes(points) == _prefill_shapes(base)
    assert len(axes["decode_batches"]) == 4 and axes["decode_batches"] <= full["decode_batches"]
    assert {1, max(full["decode_batches"])} <= axes["decode_batches"]
    assert axes["contexts"] == {batch: full["contexts"][batch] for batch in axes["decode_batches"]}

    # Decode context cap: at most three per batch, the 4-token context and the maximum always kept.
    points, axes = capped(decode_max_kv_read_token_samples=3)
    assert _prefill_shapes(points) == _prefill_shapes(base)
    assert axes["decode_batches"] == full["decode_batches"]
    for batch, contexts in axes["contexts"].items():
        assert len(contexts) == min(3, len(full["contexts"][batch]))
        assert contexts[0] == full["contexts"][batch][0] and contexts[-1] == 4
        assert set(contexts) <= set(full["contexts"][batch])


def test_benchmark_max_batch_size_caps_the_decode_batch_axis_only():
    limits = _limits(16)  # decode graph sizes 1, 2, 4, 8, 12, 16, 24; largest decode batch 40 (the running cap)
    base = grid.build_grid("agg", limits, NO_CAPS).points
    assert sorted(_axes(base)["decode_batches"]) == [1, 2, 3, 4, 5, 8, 9, 12, 13, 16, 17, 24, 25, 40]
    # Cap 6: sizes 1, 2, 4 with their successors, and the cap itself as the new upper end.
    points = grid.build_grid("agg", limits, dataclasses.replace(NO_CAPS, benchmark_max_batch_size=6)).points
    assert sorted(_axes(points)["decode_batches"]) == [1, 2, 3, 4, 5, 6]
    assert _prefill_shapes(points) == _prefill_shapes(base)
    assert max(point.batch_size for point in points if point.phase == "prefill") > 6
    # A cap above the largest feasible batch changes nothing; a cap of 1 leaves one batch size.
    roomy = grid.build_grid("agg", limits, dataclasses.replace(NO_CAPS, benchmark_max_batch_size=4096)).points
    assert roomy == base
    single = grid.build_grid("decode", limits, dataclasses.replace(NO_CAPS, benchmark_max_batch_size=1)).points
    assert {point.batch_size for point in single} == {1}


# ---------------------------------------------------------------------------------------------------------------
# Generated grid
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("graphs", [True, False], ids=["graphs", "no-graphs"])
@pytest.mark.parametrize("chunked", [True, False], ids=["chunked", "unchunked"])
@pytest.mark.parametrize("page", PAGE_SIZES)
def test_generated_grid_invariants(page, chunked, graphs):
    limits = _limits(page, chunked=chunked, graphs=graphs)
    unit = grid.cached_unit(page)
    assert unit % page == 0 and unit == grid.ceil_pages(16, page)
    built = grid.build_grid("agg", limits)
    points = built.points
    prefill = [point for point in points if point.phase == "prefill"]
    decode = [point for point in points if point.phase == "decode"]
    assert prefill and decode
    assert built.missing_phases == () and built.source == "generated"

    # Ids are 1..N in execution order; the prefill block comes first.
    assert [point.benchmark_id for point in points] == list(range(1, len(points) + 1))
    assert list(points) == prefill + decode
    assert not any(point.explicit for point in points)

    # Every point fits, and no coordinate is emitted twice. The oracle is the two test-side models, not the
    # module's reason functions: the generator has consulted those already.
    assert all(_one_pass_takes_all(limits, p.batch_size, p.new_tokens, p.cached_tokens) for p in prefill)
    assert all(p.cached_tokens % page == 0 for p in prefill)
    assert all(_decode_run_holds_all(limits, p.batch_size, p.cached_tokens) for p in decode)
    assert len({_coordinate(point) for point in points}) == len(points)

    # Order: every axis descending.
    prefill_keys = [(p.new_tokens, p.batch_size, p.cached_tokens) for p in prefill]
    decode_keys = [(p.batch_size, p.cached_tokens) for p in decode]
    assert prefill_keys == sorted(prefill_keys, reverse=True)
    assert decode_keys == sorted(decode_keys, reverse=True)

    # Prefill axes: new tokens from 1 to the largest single request; cached prefixes are multiples of G and
    # run from 0 to the largest multiple of G that fits.
    axes = _axes(points)
    n_limit, _ = _brute_largest(range(1, 1504), lambda new: _prefill_fits(limits, 1, new, 0))
    assert min(axes["new"]) == 1 and max(axes["new"]) == n_limit
    assert all(point.cached_tokens % unit == 0 for point in prefill)
    for (new, batch), cached in axes["cached"].items():
        k_max, _ = _brute_largest(range(0, 1504 + unit, unit), lambda k: _prefill_fits(limits, batch, new, k))
        assert cached[0] == k_max and cached[-1] == 0, (new, batch)
    if page == 3:
        # Pins ceilP(16) = 18 against max(P, 16) = 16, which is not a page multiple.
        assert min(point.cached_tokens for point in prefill if point.cached_tokens) == 18

    # Decode axes: every batch has the 4-token context and the largest context that fits.
    for batch, contexts in axes["contexts"].items():
        k_max, _ = _brute_largest(range(4, 1504), lambda k: _decode_fits(limits, batch, k))
        assert contexts[0] == k_max and contexts[-1] == 4, batch
    b_limit, _ = _brute_largest(range(1, 44), lambda batch: _decode_fits(limits, batch, 4))
    assert min(axes["decode_batches"]) == 1 and max(axes["decode_batches"]) == b_limit

    # The artifact form: batch totals, six keys, stamps, and AISim's validator accepts each point.
    dicts = built.point_dicts()
    assert built.digest == grid.grid_digest(dicts)
    for point, record in zip(points, dicts, strict=True):
        assert list(record) == [
            "point_type",
            "benchmark_id",
            "total_prefill_tokens",
            "total_kv_read_tokens",
            "batch_size",
            "sample_reasons",
        ]
        assert record["point_type"] == point.phase and record["benchmark_id"] == point.benchmark_id
        assert record["batch_size"] == point.batch_size
        assert record["total_kv_read_tokens"] == point.batch_size * point.cached_tokens
        scheduled = _expected_scheduled(record)
        if point.phase == "prefill":
            assert record["total_prefill_tokens"] == point.batch_size * point.new_tokens
            assert record["sample_reasons"] == (["prefill_real_seed"] if point.cached_tokens else [])
            regime = "real_prefix" if point.cached_tokens else "not_applicable"
            assert scheduled["num_prefill_requests"] == point.batch_size
            assert scheduled["sum_prefill_tokens"] == point.batch_size * point.new_tokens
            assert scheduled["sum_prefill_kv_tokens"] == point.batch_size * point.cached_tokens
        else:
            assert point.new_tokens == 0 and record["total_prefill_tokens"] == 0
            assert record["sample_reasons"] == ["kvwarm_real_kv"]
            regime = "real_kv"
            assert scheduled["num_decode_requests"] == point.batch_size
            assert scheduled["sum_decode_kv_tokens"] == point.batch_size * point.cached_tokens
        # AISim's validator: the stamp on the point is the one its seed regime requires.
        assert _validate_seed_regime({"kv_seed_regime": regime, "point": record}, Path("benchmark.json")) == regime


@pytest.mark.parametrize("page", [1, 3, 64])
def test_generated_axes_are_the_documented_candidates(page):
    """Without caps, each axis is exactly the rule its docstring in the module states; maxima by brute force."""
    limits = _limits(page)
    unit = grid.cached_unit(page)
    axes = _axes(grid.build_grid("agg", limits, NO_CAPS).points)

    def doubling(first, last):
        values = []
        while first <= last:
            values.append(first)
            first *= 2
        return values

    n_limit, _ = _brute_largest(range(1, 1504), lambda new: _prefill_fits(limits, 1, new, 0))
    # The low end: 1 and both sides of the first step. The smallest prefill graph has 4 tokens, and a batch
    # replays it when the 4 tokens are at most twice its own: the first such batch has 2 tokens.
    smallest = min(limits.prefill_graph_sizes)
    first = next(tokens for tokens in range(1, smallest + 1) if smallest <= 2 * tokens)
    assert (smallest, first) == (4, 2)
    graph_values = _dynamo_cudagraph_axis_points(limits.prefill_graph_sizes, n_limit)
    assert sorted(axes["new"]) == sorted({1, first - 1, first, *graph_values})
    assert len(axes["new"]) == len(graph_values) + 2
    for new in axes["new"]:
        b_max, _ = _brute_largest(range(1, 44), lambda batch: _prefill_fits(limits, batch, new, 0))
        assert sorted(axes["batches"][new]) == sorted({*doubling(1, b_max), b_max})
        for batch in axes["batches"][new]:
            k_max, _ = _brute_largest(range(0, 1504 + unit, unit), lambda k: _prefill_fits(limits, batch, new, k))
            expected = [0] if k_max < unit else sorted({0, *doubling(unit, k_max), k_max})
            assert sorted(axes["cached"][(new, batch)]) == expected
    b_limit, _ = _brute_largest(range(1, 44), lambda batch: _decode_fits(limits, batch, 4))
    assert sorted(axes["decode_batches"]) == list(_dynamo_cudagraph_axis_points(limits.decode_graph_sizes, b_limit))
    for batch in axes["decode_batches"]:
        k_max, _ = _brute_largest(range(4, 1504), lambda k: _decode_fits(limits, batch, k))
        assert sorted(axes["contexts"][batch]) == sorted({*doubling(4, k_max), k_max})


def test_decode_graph_sizes_above_the_running_cap_are_not_on_the_batch_axis():
    # R = 40, so a graph of 64 can never be the size of a batch. The largest size that counts is 4, and the
    # doubling tail 8, 16, 32 starts from it. (Counting 64 would leave no tail: 1, 2, 3, 4, 5, 40.)
    limits = _limits(16, decode_graph_sizes=(1, 2, 4, 64))
    assert grid.decode_batch_axis(limits, NO_CAPS) == [1, 2, 3, 4, 5, 8, 16, 32, 40]
    # Sizes 12, 16 and 24 are above R = 10; nothing above 10 is on the axis.
    limits = _limits(16, max_running_requests=10)
    assert grid.decode_batch_axis(limits, NO_CAPS) == [1, 2, 3, 4, 5, 8, 9, 10]
    # Cut to three: the tail above the largest counted size 8 is [9, 10], 2 of 8 values and too many to
    # protect, so the cut is uniform: indices 0, 3.5 -> 4, 7.
    assert grid.decode_batch_axis(limits, dataclasses.replace(NO_CAPS, decode_max_batch_size_samples=3)) == [1, 5, 10]


def test_a_decode_graph_size_equal_to_the_running_cap_stays_on_the_batch_axis():
    # R = 40 and a graph of exactly 40: a batch of 40 can run, so the size counts. Sizes 1, 2, 4 with their
    # successors, then 40, which is also the largest batch. The doubling tail would start at 80, so there is
    # none. (Dropping the size 40 would leave 4 as the largest and bring the tail 8, 16, 32 back.)
    limits = _limits(16, decode_graph_sizes=(1, 2, 4, 40))
    assert grid.largest_decode_batch(limits) == 40
    assert grid.decode_batch_axis(limits, NO_CAPS) == [1, 2, 3, 4, 5, 40]


def test_the_decode_batch_cap_splits_the_tail_at_the_largest_size_that_fits_the_running_cap():
    # R = 12, decode graph sizes 1..10 and 64. The size 64 is above R and does not count, so the largest size is
    # 10. Candidates: the sizes 1..10, their successors 2..11, no doubling tail (20 > 12), and 12: the values
    # 1..12.
    limits = _limits(16, max_running_requests=12, decode_graph_sizes=(*range(1, 11), 64))
    assert grid.decode_batch_axis(limits, NO_CAPS) == list(range(1, 13))
    # Cap 6, split at the largest counted size 10: the graph part is 1..10 and the tail is [11, 12]. The tail is
    # non-empty and 2 * 5 = 10 <= 12 values, so it is kept whole and the budget for the graph part is 6 - 2 = 4.
    # Four samples of the ten values 1..10: positions 0, 3, 6, 9 are the indices themselves: 1, 4, 7, 10.
    caps = dataclasses.replace(NO_CAPS, decode_max_batch_size_samples=6)
    assert grid.decode_batch_axis(limits, caps) == [1, 4, 7, 10, 11, 12]
    # Split at 64 instead (the unfiltered sizes) there would be no tail, and the uniform cut of 1..12 to six
    # samples takes positions 0, 2.2, 4.4, 6.6, 8.8, 11, the indices 0, 2, 4, 7, 9, 11: the row 11 is lost.
    assert grid.limit_uniform(list(range(1, 13)), 6) == [1, 3, 5, 8, 10, 12]


def test_graph_driven_axes_keep_their_unpadded_tail_under_a_cap():
    sizes = tuple(range(2, 41, 2))  # 2, 4, ..., 40
    limits = grid.Limits(1, 101, 100_000, 100, 512, None, sizes, sizes)
    caps = dataclasses.replace(NO_CAPS, prefill_max_new_token_samples=8, decode_max_batch_size_samples=8)
    # New tokens: n_limit = I - 1 = 100. Graph axis: each size and its successor (2..41), 80, 100: 42 values.
    # The tail above the largest size is [41, 80, 100]; 3 * 5 = 15 <= 42, so it is kept whole and the other
    # five samples come from 2..40 (39 values): positions 0, 9.5, 19, 28.5, 38 round to indices 0, 10, 19, 29,
    # 38. The ladder has no 1, so 1 is put in front after the cut: nine values under a cap of eight. A batch of
    # one token already replays the smallest graph, of 2 tokens, so the low end of the axis is that 1 alone.
    assert grid.prefill_new_token_axis(limits, caps) == [1, 2, 12, 21, 31, 40, 41, 80, 100]
    # Decode batches: b_limit = R = 100 and the same ladder, so the same axis.
    assert grid.decode_batch_axis(limits, caps) == [1, 2, 12, 21, 31, 40, 41, 80, 100]
    # A uniform cut of the whole graph axis would have dropped 41 and 80: positions 0, 5.86, 11.7, 17.6, 23.4,
    # 29.3, 35.1, 41 round to indices 0, 6, 12, 18, 23, 29, 35, 41.
    assert grid.limit_uniform(grid.graph_axis(sizes, 100), 8) == [2, 8, 14, 20, 25, 31, 37, 100]


def test_a_duplicate_coordinate_is_a_bug_and_raises(monkeypatch):
    monkeypatch.setattr(grid, "decode_batch_axis", lambda limits, caps: [2, 2])
    with pytest.raises(RuntimeError, match="share the coordinate"):
        grid.build_grid("decode", _limits(16))


def test_a_generated_candidate_that_does_not_fit_is_a_bug_and_raises(monkeypatch):
    """Generated points fit by construction. An axis that proposes one that does not must not lose it silently."""
    # The unpatched grid of record A builds, and each block has points.
    prefill_points = grid.build_grid("prefill", LIMITS_A).points
    decode_points = grid.build_grid("decode", LIMITS_A).points
    assert prefill_points and decode_points
    assert len(grid.build_grid("agg", LIMITS_A).points) == len(prefill_points) + len(decode_points)

    # Prefill. Record A plans n = 64 first, one request (C = 64). A cached prefix of 192 tokens is page-aligned
    # and makes the prompt 256 tokens, above I = 200.
    with monkeypatch.context() as patch:
        patch.setattr(grid, "prefill_cached_axis", lambda limits, caps, batch, new: [0, 192])
        for mode in ("prefill", "agg"):
            with pytest.raises(RuntimeError) as caught:
                grid.build_grid(mode, LIMITS_A)
            assert str(caught.value) == (
                "generated prefill point (batch_size=1, new_tokens=64, cached_tokens=192) does not fit: "
                "reason=input_too_long; this is a bug in the grid axes"
            )
        assert grid.build_grid("decode", LIMITS_A).points == decode_points  # the decode block does not use that axis

    # Decode. Record A plans b = R = 8 first. Eight requests at the context 200 need 8 * (208 + 16 + 7) = 1848
    # pool tokens, above S = 1600.
    with monkeypatch.context() as patch:
        patch.setattr(grid, "decode_context_axis", lambda limits, caps, batch: [4, 200])
        for mode in ("decode", "agg"):
            with pytest.raises(RuntimeError) as caught:
                grid.build_grid(mode, LIMITS_A)
            assert str(caught.value) == (
                "generated decode point (batch_size=8, context_tokens=200) does not fit: "
                "reason=kv_pool; this is a bug in the grid axes"
            )
        assert grid.build_grid("prefill", LIMITS_A).points == prefill_points

    # One value above each real maximum, the kind of error an axis bug would make.
    real_cached_axis, real_context_axis = grid.prefill_cached_axis, grid.decode_context_axis
    unit = grid.cached_unit(LIMITS_A.page_size)
    with monkeypatch.context() as patch:
        patch.setattr(
            grid,
            "prefill_cached_axis",
            lambda limits, caps, batch, new: [*(axis := real_cached_axis(limits, caps, batch, new)), axis[-1] + unit],
        )
        with pytest.raises(RuntimeError, match=r"generated prefill point .* does not fit: reason="):
            grid.build_grid("prefill", LIMITS_A)
    with monkeypatch.context() as patch:
        patch.setattr(
            grid,
            "decode_context_axis",
            lambda limits, caps, batch: [*(axis := real_context_axis(limits, caps, batch)), axis[-1] + 1],
        )
        with pytest.raises(RuntimeError, match=r"generated decode point .* does not fit: reason="):
            grid.build_grid("decode", LIMITS_A)


def test_cached_and_context_axes_are_empty_when_nothing_fits():
    caps = grid.SampleCaps()
    # Nine requests are above the running cap R = 8 of record A, whatever their size.
    assert grid.largest_prefill_cached(LIMITS_A, 9, 16) is None
    assert grid.prefill_cached_axis(LIMITS_A, caps, 9, 16) == []
    assert grid.largest_decode_context(LIMITS_A, 9) is None
    assert grid.decode_context_axis(LIMITS_A, caps, 9) == []
    # For a batch that fits, the same calls return an axis.
    assert grid.prefill_cached_axis(LIMITS_A, caps, 4, 16)[0] == 0
    assert grid.decode_context_axis(LIMITS_A, caps, 8)[0] == 4


def test_prefill_new_token_axis_gets_its_low_end_after_the_cut_and_outside_the_cap():
    """AISim's reader refuses a query below the collected range, so 1 new token must be on the axis.

    With prefill graphs the low end also holds both sides of the first step of the table: the largest batch
    that runs without a graph and the smallest that runs with one. The low end is put on the axis AFTER the
    cut. The collector host sizes the cap as the number of candidates of its own mirror, which has no low end;
    with that cap nothing may be cut.
    """
    sizes = (4, 8, 16)
    limits = _limits(1, prefill_graph_sizes=sizes)
    n_limit = grid.largest_prefill_new_tokens(limits)
    assert n_limit == 600  # the chunk budget
    mirror = list(_dynamo_cudagraph_axis_points(tuple(sizes), n_limit))
    assert mirror == [4, 5, 8, 9, 16, 17, 32, 64, 128, 256, 512, 600]
    # The smallest graph has 4 tokens. A batch of one token would be padded to four times its size and runs
    # without a graph; a batch of two is padded to twice its size and runs with it. The low end is 1 and 2.
    assert grid.prefill_new_token_axis(limits, NO_CAPS) == [1, 2, *mirror]

    def capped(cap, record=limits):
        return grid.prefill_new_token_axis(record, dataclasses.replace(NO_CAPS, prefill_max_new_token_samples=cap))

    # The host-sized cap, 12: nothing is cut, and the axis holds cap + 2 values.
    assert capped(len(mirror)) == [1, 2, *mirror]
    # Cap 2: the tail above 16 is 7 of the 12 values, too many to protect, so the cut is uniform: both ends.
    assert capped(2) == [1, 2, 4, 600]
    # Cap 3: uniform again; positions 0, 5.5, 11 round to indices 0, 6, 11.
    assert capped(3) == [1, 2, 4, 32, 600]
    for cap in (2, 3, 5, 11):
        assert capped(cap) == [1, 2, *grid.limit_graph_axis(mirror, sizes, cap)]
        assert len(capped(cap)) == cap + 2

    # The default cap of 64 on a ladder long enough to be cut: the even sizes 2..200 give 2..201, 400, 800 and
    # n_limit = I - 1 = 1000, 203 values. A batch of one token already runs with the smallest graph, of 2
    # tokens, so the low end is 1 alone: the first value is 1 and the rest is the cut graph axis.
    even = tuple(range(2, 201, 2))
    long_ladder = grid.Limits(1, 1001, 1_000_000, 100, 4096, None, (), even)
    graph_values = grid.graph_axis(even, 1000)
    assert len(graph_values) == 203
    axis = grid.prefill_new_token_axis(long_ladder, grid.SampleCaps())
    assert axis[0] == 1 and axis[1:] == grid.limit_graph_axis(graph_values, even, 64) and len(axis) == 65

    # A ladder that contains 1, or no ladder (powers of two): 1 is on the graph axis already and there is no
    # step below it, so nothing is added and a cut axis holds exactly cap values. Ladder (1, 2): 1, 2, 3, the
    # doubling tail 4..512, and 600: 12 values. Cap 3 is a uniform cut (the tail above 2 is 10 of the 12
    # values): indices 0, 6, 11.
    with_one = _limits(1, prefill_graph_sizes=(1, 2))
    assert grid.prefill_new_token_axis(with_one, NO_CAPS) == [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 600]
    assert capped(2, with_one) == [1, 600]
    assert capped(3, with_one) == [1, 32, 600]
    assert grid.prefill_new_token_axis(_limits(1, graphs=False), NO_CAPS)[:3] == [1, 2, 4]
    assert capped(2, _limits(1, graphs=False)) == [1, 600]


# The low end of the prefill new-token axis: 1 and, when prefill graphs are captured, both sides of the first
# step. The tests below take the step from a test-side model of SGLang's replay rule, not from the module.


def _runs_with_a_graph(tokens: int, sizes) -> bool:
    """Whether SGLang runs a prefill batch of ``tokens`` new tokens with one of the captured ``sizes``.

    A test-side model. The batch is padded to the smallest captured size that holds it, and it replays that
    graph unless the padded size is more than twice its own tokens; a batch above the largest size has no
    graph. ``sizes`` are ascending. The last test of this file reads the rule in SGLang's source
    (``PrefillCudaGraphRunner.can_replay_locally`` and ``BaseCudaGraphRunner._pad_to_bucket``).
    """
    if tokens > sizes[-1]:
        return False
    padded = next(size for size in sizes if size >= tokens)
    return padded <= 2 * tokens


def _expected_low_end(sizes, n_limit: int) -> set[int]:
    """The low end for ``n_limit``: 1 and both sides of the first step, the step found by trying every batch."""
    values = {1}
    captured = sorted({size for size in sizes if size >= 1})
    if captured:
        first = next(tokens for tokens in range(1, captured[0] + 1) if _runs_with_a_graph(tokens, captured))
        values |= {first - 1, first}
    return {value for value in values if 1 <= value <= n_limit}


def _low_end_limits(page: int, n_limit: int, sizes) -> grid.Limits:
    """Limits whose largest new-token count is ``n_limit`` at any page size: the input limit ends the axis."""
    limits = grid.Limits(page, n_limit + 1, 100_000 * page, 8, 100_000, None, (), sizes)
    assert grid.largest_prefill_new_tokens(limits) == n_limit
    return limits


@pytest.mark.parametrize(
    ("sizes", "expected"),
    [
        # One token runs with a graph of 1 (nothing is padded) and with a graph of 2 (twice its size).
        ((1, 2, 4), 1),
        ((2, 4, 8), 1),
        # A graph of 3 or of 4 is more than twice one token and at most twice two tokens.
        ((3, 6), 2),
        ((4, 8, 16), 2),
        # A graph of 5: 5 > 2 * 2 and 5 <= 2 * 3. A graph of 7 or of 8 is more than 2 * 3 and at most 2 * 4.
        ((5, 10), 3),
        ((7,), 4),
        ((8, 16), 4),
        # Only the smallest size counts, in whatever order and however often it is given.
        ((16, 4, 8, 4), 2),
        # No graph size, or none that is at least 1: there is no first step.
        ((), None),
        ((0, -3), None),
        # Entries below 1 are not sizes.
        ((0, 4, -8), 2),
        ((-1, 0, 1), 1),
    ],
)
def test_first_prefill_graph_tokens_hand_worked(sizes, expected):
    assert grid.first_prefill_graph_tokens(sizes) == expected
    assert grid.first_prefill_graph_tokens(list(sizes)) == expected
    assert grid.first_prefill_graph_tokens(iter(sizes)) == expected


def test_first_prefill_graph_tokens_is_the_smallest_batch_that_runs_with_the_smallest_graph():
    assert grid.PREFILL_GRAPH_PADDING_FACTOR == 2
    for smallest in range(1, 300):
        sizes = (smallest, 5 * smallest + 3)
        first = grid.first_prefill_graph_tokens(sizes)
        with_graph = [tokens for tokens in range(1, smallest + 1) if _runs_with_a_graph(tokens, sizes)]
        # Every batch from ``first`` up to the size runs with the graph, and no smaller one does.
        assert with_graph == list(range(first, smallest + 1)), smallest
        assert 1 <= first <= smallest
        # The size is at most twice ``first`` and more than twice ``first - 1``.
        assert 2 * (first - 1) < smallest <= 2 * first
    for bad in ((True,), (4.0,), ("4",), (None,), (4, 0.5)):
        with pytest.raises(TypeError, match="graph size"):
            grid.first_prefill_graph_tokens(bad)


LOW_END_CASES = [
    # (prefill graph sizes, n_limit, the axis without a cut, the axis under a cap of 2).
    # Sizes from 1 or from 2: a batch of one token runs with the smallest graph, so the low end is 1 alone.
    ((1, 2, 4), 1, [1], [1]),
    ((1, 2, 4), 2, [1, 2], [1, 2]),
    ((1, 2, 4), 3, [1, 2, 3], [1, 3]),  # the cut keeps the two ends of 1, 2, 3
    ((1, 2, 4), 100, [1, 2, 3, 4, 5, 8, 16, 32, 64, 100], [1, 100]),
    ((2, 4), 1, [1], [1]),
    ((2, 4), 2, [1, 2], [1, 2]),
    ((2, 4), 3, [1, 2, 3], [1, 2, 3]),  # the graph axis is 2, 3: nothing to cut, and 1 in front
    ((2, 4), 100, [1, 2, 3, 4, 5, 8, 16, 32, 64, 100], [1, 2, 100]),
    # Sizes from 3 or from 4: the first batch with a graph has 2 tokens, so the low end is 1 and 2.
    ((3, 6), 1, [1], [1]),
    ((3, 6), 2, [1, 2], [1, 2]),
    ((3, 6), 3, [1, 2, 3], [1, 2, 3]),
    ((3, 6), 100, [1, 2, 3, 4, 6, 7, 12, 24, 48, 96, 100], [1, 2, 3, 100]),
    ((4, 8), 1, [1], [1]),
    ((4, 8), 2, [1, 2], [1, 2]),
    ((4, 8), 3, [1, 2, 3], [1, 2, 3]),  # the graph axis is the limit 3 alone
    ((4, 8), 100, [1, 2, 4, 5, 8, 9, 16, 32, 64, 100], [1, 2, 4, 100]),
    # Sizes from 5: the first batch with a graph has 3 tokens, so the low end is 1, 2 and 3.
    ((5, 10), 1, [1], [1]),
    ((5, 10), 2, [1, 2], [1, 2]),
    ((5, 10), 3, [1, 2, 3], [1, 2, 3]),
    ((5, 10), 100, [1, 2, 3, 5, 6, 10, 11, 20, 40, 80, 100], [1, 2, 3, 5, 100]),
    # Sizes from 8: the first batch with a graph has 4 tokens, so the low end is 1, 3 and 4. With n_limit = 2
    # the graph axis is that limit alone and neither side of the step is in range; with n_limit = 3 only 3 is.
    ((8, 16), 1, [1], [1]),
    ((8, 16), 2, [1, 2], [1, 2]),
    ((8, 16), 3, [1, 3], [1, 3]),
    ((8, 16), 100, [1, 3, 4, 8, 9, 16, 17, 32, 64, 100], [1, 3, 4, 8, 100]),
    # No prefill graphs: the powers of two and the limit, which start at 1. There is no step and no low end.
    ((), 1, [1], [1]),
    ((), 2, [1, 2], [1, 2]),
    ((), 3, [1, 2, 3], [1, 3]),
    ((), 100, [1, 2, 4, 8, 16, 32, 64, 100], [1, 100]),
]


@pytest.mark.parametrize("page", [1, 64])
@pytest.mark.parametrize(("sizes", "n_limit", "uncut", "cut_to_two"), LOW_END_CASES)
def test_the_low_end_of_the_prefill_new_token_axis_hand_worked(sizes, n_limit, uncut, cut_to_two, page):
    """The new-token axis counts tokens, not pages: the same values at page size 1 and at page size 64.

    Under a cap of 2, the smallest one, every graph axis that is cut here is cut to its two ends, and the
    low end is added to them: up to five values under a cap of two.
    """
    limits = _low_end_limits(page, n_limit, sizes)
    assert grid.prefill_new_token_axis(limits, NO_CAPS) == uncut
    smallest_cap = dataclasses.replace(NO_CAPS, prefill_max_new_token_samples=2)
    assert grid.prefill_new_token_axis(limits, smallest_cap) == cut_to_two
    # The same low end from the test-side model of SGLang's rule.
    graph_values = grid.graph_axis(sizes, n_limit)
    low_end = _expected_low_end(sizes, n_limit)
    assert uncut == sorted({*graph_values, *low_end})
    ends = graph_values if len(graph_values) <= 2 else [graph_values[0], graph_values[-1]]
    assert cut_to_two == sorted({*ends, *low_end})
    # Every value of the low end is planned: one request of that many new tokens, without a cached prefix.
    for caps in (NO_CAPS, smallest_cap):
        planned = set(_prefill_shapes(grid.build_grid("prefill", limits, caps).points))
        assert {(1, new, 0) for new in low_end} <= planned


@pytest.mark.parametrize("page", [1, 64])
def test_the_low_end_never_counts_against_the_cap_and_the_axis_stays_ascending_and_in_range(page):
    rng = random.Random(8100 + page)
    added_counts = dict.fromkeys(range(4), 0)
    for _ in range(3000):
        sizes = tuple(sorted({rng.randint(1, 40) for _ in range(rng.randint(0, 6))}))
        n_limit = rng.choice([1, 2, 3, rng.randint(1, 12), rng.randint(1, 300)])
        cap = rng.choice([2, 3, rng.randint(2, 12), 10_000])
        limits = _low_end_limits(page, n_limit, sizes)
        axis = grid.prefill_new_token_axis(limits, dataclasses.replace(NO_CAPS, prefill_max_new_token_samples=cap))
        graph_values = grid.graph_axis(sizes, n_limit)
        cut = grid.limit_graph_axis(graph_values, sizes, cap)
        low_end = _expected_low_end(sizes, n_limit)

        # The axis is the cut graph axis and the low end, and nothing else.
        assert set(axis) == {*cut, *low_end}, (sizes, n_limit, cap)
        assert all(low < high for low, high in itertools.pairwise(axis)), (sizes, n_limit, cap)
        assert axis[0] == 1 and axis[-1] == n_limit
        # The cap is spent on the graph axis alone: the cut holds as many values as the cap allows, every one
        # of them is on the axis, and the low end comes on top, with at most three values.
        assert len(cut) == min(cap, len(graph_values))
        added = len(axis) - len(cut)
        assert added == len(low_end - set(cut)) <= 3
        added_counts[added] += 1

        # What the low end is for: the smallest batch that runs with a graph is a row, and so is the batch of
        # one token less, the largest that runs without one.
        if sizes and n_limit >= sizes[0]:
            with_graph = [tokens for tokens in range(1, sizes[0] + 1) if _runs_with_a_graph(tokens, sizes)]
            assert with_graph[0] in axis
            assert with_graph[0] == 1 or with_graph[0] - 1 in axis
    # The sample holds every number of added values, from none (the low end is on the graph axis) to three.
    assert all(count > 20 for count in added_counts.values()), added_counts


def test_the_low_end_switched_off_gives_the_earlier_axis_and_the_same_other_points(monkeypatch):
    """The only difference from the axis without the two sides of the first step is whole groups of new tokens.

    ``first_prefill_graph_tokens`` is the one switch: with it answering ``None`` the axis is the cut graph axis
    with 1 in front, as it was before the two sides were added. Every point of that grid is a point of the
    grid of today, in the same order, and the decode block is the same block.
    """
    records = [
        (_limits(1), NO_CAPS),
        (_limits(16), grid.SampleCaps()),
        (_limits(64, chunked=False), grid.SampleCaps(prefill_max_new_token_samples=3)),
        (_limits(16, prefill_graph_sizes=(8, 16, 32)), NO_CAPS),
        (_limits(1, prefill_graph_sizes=(5, 10, 20)), grid.SampleCaps(prefill_max_new_token_samples=2)),
    ]
    for limits, caps in records:
        now = grid.build_grid("agg", limits, caps)
        axis = grid.prefill_new_token_axis(limits, caps)
        with monkeypatch.context() as patch:
            patch.setattr(grid, "first_prefill_graph_tokens", lambda sizes: None)
            earlier = grid.build_grid("agg", limits, caps)
            earlier_axis = grid.prefill_new_token_axis(limits, caps)
            decode_only = grid.build_grid("decode", limits, caps)
        cut = grid.limit_graph_axis(
            grid.graph_axis(limits.prefill_graph_sizes, grid.largest_prefill_new_tokens(limits)),
            limits.prefill_graph_sizes,
            caps.prefill_max_new_token_samples,
        )
        assert earlier_axis == [1, *cut]
        added = set(axis) - set(earlier_axis)
        assert added == _expected_low_end(limits.prefill_graph_sizes, axis[-1]) - {1} and added
        kept = [shape for shape in _prefill_shapes(now.points) if shape[1] not in added]
        assert kept == _prefill_shapes(earlier.points)
        assert {shape[1] for shape in _prefill_shapes(now.points)} - {shape[1] for shape in kept} == added
        assert _decode_shapes(now.points) == _decode_shapes(earlier.points)
        # The ids of the added points move every later id, so the digest of the grid changes with them. A
        # decode-only grid holds no prefill point and is the same grid.
        assert now.digest != earlier.digest
        assert grid.build_grid("decode", limits, caps) == decode_only


@pytest.mark.parametrize("page", [1, 3, 16, 64])
def test_every_generated_prefill_point_fits_with_the_low_end_on_the_axis(page):
    """Random small limits with a prefill ladder that starts between 1 and 9 tokens.

    The oracle is the test-side model of one prefill pass. A low-end value is at most ``n_limit``, so one
    request of that size without a prefix always fits; its larger batches and its prefixes go through the
    same searches as those of every other value.
    """
    rng = random.Random(8200 + page)
    cut_caps = grid.SampleCaps(prefill_max_new_token_samples=2, prefill_max_kv_read_token_samples=2)
    seen = {"points": 0, "two_sided": 0, "empty": 0}
    for _ in range(300):
        smallest = rng.randint(1, 9)
        sizes = tuple(smallest * factor for factor in (1, 2, 4, 8))
        limits = dataclasses.replace(_random_limits(rng, page), prefill_graph_sizes=sizes)
        n_limit, _ = _brute_largest(
            range(1, limits.max_req_input_len + 4), lambda new: _prefill_fits(limits, 1, new, 0)
        )
        for caps in (NO_CAPS, grid.SampleCaps(), cut_caps):
            built = grid.build_grid("prefill", limits, caps)  # a candidate that does not fit would raise here
            shapes = _prefill_shapes(built.points)
            if n_limit is None:
                assert shapes == [] and built.missing_phases == ("prefill",)
                seen["empty"] += 1
                continue
            low_end = _expected_low_end(sizes, n_limit)
            new_values = {new for _batch, new, _cached in shapes}
            assert low_end <= new_values and min(new_values) == 1 and max(new_values) == n_limit
            assert {(1, new, 0) for new in low_end} <= set(shapes)
            assert all(_one_pass_takes_all(limits, *shape) for shape in shapes), limits
            assert len(set(shapes)) == len(shapes)
            seen["points"] += len(shapes)
            seen["two_sided"] += len(low_end) > 1
    assert seen["points"] > 1000 and seen["two_sided"] > 50 and seen["empty"] > 5, seen


def test_a_step_inside_the_graph_sizes_is_not_sampled():
    """What the docstring of ``prefill_new_token_axis`` calls not sampled, pinned so that a change shows.

    A batch just above a size is padded to the next size and runs without a graph when that is more than
    twice its own tokens. After a size of 4 the first such size is 11: five tokens would be padded to 11.
    The sizes 9 and 10 are more than twice 4 as well and leave no batch without a graph, so "more than the
    factor times the size before it" is needed for such a step and is not enough for one.
    """
    assert all(_runs_with_a_graph(5, (4, after)) for after in range(5, 11))
    sizes = (4, 11)
    assert [tokens for tokens in range(1, 12) if not _runs_with_a_graph(tokens, sizes)] == [1, 5]
    # Both sides of 4 and of 11, the doubling tail and the limit, with the low end 1 and 2 in front. The row
    # at 5 runs without a graph and the next row, at 11, with one: the step between 5 and 6 has no row at 6.
    axis = grid.prefill_new_token_axis(_low_end_limits(1, 100, sizes), NO_CAPS)
    assert axis == [1, 2, 4, 5, 11, 12, 22, 44, 88, 100]


def test_decode_batch_axis_starts_at_one_and_the_one_does_not_count_against_the_cap():
    """A decode graph ladder that starts above 1 still gives an axis that starts at 1."""
    limits = _limits(16, decode_graph_sizes=(8, 16))  # the largest batch is the running cap, 40

    def capped(cap, record=limits):
        return grid.decode_batch_axis(record, dataclasses.replace(NO_CAPS, decode_max_batch_size_samples=cap))

    # Both sides of 8 and 16, the doubling tail 32, and 40; then 1 in front.
    assert grid.decode_batch_axis(limits, NO_CAPS) == [1, 8, 9, 16, 17, 32, 40]
    assert grid.decode_batch_axis(limits, grid.SampleCaps()) == [1, 8, 9, 16, 17, 32, 40]
    # Cap 2: the tail above 16 is 3 of the 6 graph values, too many to protect, so the cut is uniform.
    assert capped(2) == [1, 8, 40]
    # Cap 3: uniform; positions 0, 2.5, 5 round to indices 0, 3, 5.
    assert capped(3) == [1, 8, 17, 40]
    # A cap of 6 cuts nothing: seven values under a cap of six.
    assert capped(6) == [1, 8, 9, 16, 17, 32, 40]
    # benchmark_max_batch_size below the first graph size: the axis is 1 and the cap.
    assert grid.decode_batch_axis(limits, dataclasses.replace(NO_CAPS, benchmark_max_batch_size=5)) == [1, 5]

    # A ladder that contains 1 is unchanged by the rule: the axis is the host mirror's, and a cut axis holds
    # exactly cap values.
    with_one = _limits(16)
    mirror = list(_dynamo_cudagraph_axis_points(with_one.decode_graph_sizes, 40))
    assert mirror == [1, 2, 3, 4, 5, 8, 9, 12, 13, 16, 17, 24, 25, 40]
    assert grid.decode_batch_axis(with_one, NO_CAPS) == mirror
    assert capped(2, with_one) == [1, 40]
    # Cap 3: the tail above 24 is [25, 40], 2 * 5 = 10 <= 14 values, so it is kept and one sample is left for
    # the graph part: its first value.
    assert capped(3, with_one) == [1, 25, 40]
    for cap in (2, 3, 4, 13):
        assert capped(cap, with_one) == grid.limit_graph_axis(mirror, with_one.decode_graph_sizes, cap)
        assert len(capped(cap, with_one)) == cap

    # Every decode batch of the grid, the added 1 included, gets its points.
    built = grid.build_grid("decode", limits)
    assert sorted({point.batch_size for point in built.points}) == [1, 8, 9, 16, 17, 32, 40]


def test_the_decode_block_does_not_depend_on_the_prefill_graph_sizes(monkeypatch):
    """The two sides of the first step are a rule of the prefill new-token axis alone.

    The decode batch axis gets 1 in front and nothing else, whatever the decode ladder starts at: the module
    has the padding rule for prefill graphs only. And no prefill graph size, with or without a low end,
    changes a decode point.
    """
    caps_list = (NO_CAPS, grid.SampleCaps(), dataclasses.replace(NO_CAPS, decode_max_batch_size_samples=3))
    for decode_sizes, axis in (
        # A decode ladder from 8: 1 and both sides of 8, with no row at 4 and none at 3.
        ((8, 16), [1, 8, 9, 16, 17, 32, 40]),
        ((5, 10), [1, 5, 6, 10, 11, 20, 40]),
        ((1, 2, 4, 8, 12, 16, 24), [1, 2, 3, 4, 5, 8, 9, 12, 13, 16, 17, 24, 25, 40]),
    ):
        without_prefill_graphs = _limits(16, decode_graph_sizes=decode_sizes, prefill_graph_sizes=())
        assert grid.decode_batch_axis(without_prefill_graphs, NO_CAPS) == axis
        reference = {caps: grid.build_grid("decode", without_prefill_graphs, caps) for caps in caps_list}
        for prefill_sizes in ((1, 2, 4), (2, 4), (3, 6), (4, 8, 16), (5, 10), (8, 16), (4, 8, 16, 32, 64, 128)):
            limits = dataclasses.replace(without_prefill_graphs, prefill_graph_sizes=prefill_sizes)
            for caps in caps_list:
                assert grid.decode_batch_axis(limits, caps) == grid.decode_batch_axis(without_prefill_graphs, caps)
                # A decode-only grid is the same grid, ids and digest included.
                assert grid.build_grid("decode", limits, caps) == reference[caps]
                # In agg mode the decode points are the same and only their ids follow the prefill block.
                agg = grid.build_grid("agg", limits, caps)
                assert _decode_shapes(agg.points) == _decode_shapes(reference[caps].points)
                with monkeypatch.context() as patch:
                    patch.setattr(grid, "first_prefill_graph_tokens", lambda sizes: None)
                    assert grid.build_grid("decode", limits, caps) == reference[caps]
                    assert _decode_shapes(grid.build_grid("agg", limits, caps).points) == _decode_shapes(agg.points)


def test_prefill_batch_axis_keeps_the_first_values_not_the_largest():
    limits = _limits(16)  # C = 592: 37 pages
    # n = 16 is one page: b_max = min(R = 40, C // 16 = 37, (M - 1) // 16 + 1 = 64, pool) = 37.
    assert grid.largest_prefill_batch(limits, 16) == 37
    assert grid.prefill_batch_axis(limits, NO_CAPS, 16) == [1, 2, 4, 8, 16, 32, 37]
    assert grid.prefill_batch_axis(limits, grid.SampleCaps(), 16) == [1, 2, 4]
    # n = 300 is 19 pages = 304 tokens: only one request fits the chunk budget.
    assert grid.prefill_batch_axis(limits, grid.SampleCaps(), 300) == [1]
    # n = 200 is 13 pages = 208 tokens: 592 // 208 = 2.
    assert grid.prefill_batch_axis(limits, grid.SampleCaps(), 200) == [1, 2]
    # n = 160 is 10 pages: 592 // 160 = 3. b_max joins the powers of two BEFORE the first three are kept, so the
    # default axis is 1, 2, 3 here and not 1, 2, 4.
    assert grid.largest_prefill_batch(limits, 160) == 3
    assert grid.prefill_batch_axis(limits, grid.SampleCaps(), 160) == [1, 2, 3]
    # n = 112 is 7 pages: 592 // 112 = 5. The candidates are 1, 2, 4, 5 and the first three leave b_max out.
    assert grid.largest_prefill_batch(limits, 112) == 5
    assert grid.prefill_batch_axis(limits, NO_CAPS, 112) == [1, 2, 4, 5]
    assert grid.prefill_batch_axis(limits, grid.SampleCaps(), 112) == [1, 2, 4]
    assert grid.prefill_batch_axis(limits, grid.SampleCaps(), 601) == []


def test_modes_select_blocks_and_number_them_continuously():
    limits = _limits(16)
    prefill = grid.build_grid("prefill", limits)
    decode = grid.build_grid("decode", limits)
    agg = grid.build_grid("agg", limits)
    assert {point.phase for point in prefill.points} == {"prefill"}
    assert {point.phase for point in decode.points} == {"decode"}
    assert [point.benchmark_id for point in decode.points] == list(range(1, len(decode.points) + 1))
    assert _prefill_shapes(agg.points) == _prefill_shapes(prefill.points)
    assert _decode_shapes(agg.points) == _decode_shapes(decode.points)
    assert agg.points[: len(prefill.points)] == prefill.points
    assert agg.points[len(prefill.points)].benchmark_id == len(prefill.points) + 1
    assert len({prefill.digest, decode.digest, agg.digest}) == 3
    assert grid.build_grid("agg", limits, None) == grid.build_grid("agg", limits, grid.SampleCaps()) == agg


def test_a_phase_without_points_is_listed_as_missing_and_the_digest_is_still_a_string():
    empty_digest = hashlib.sha256(b"[]").hexdigest()
    # I = 1: no prompt of even one token is below the input limit, and no decode context fits.
    nothing = grid.Limits(1, 1, 4096, 8, 512, 256)
    for mode, missing in (("prefill", ("prefill",)), ("decode", ("decode",)), ("agg", ("prefill", "decode"))):
        built = grid.build_grid(mode, nothing)
        assert built.points == () and built.missing_phases == missing
        assert built.digest == empty_digest and built.point_dicts() == []
    # I = 3: prefill has points (n = 1 and 2), decode has none (the lowest context is 4).
    built = grid.build_grid("agg", grid.Limits(1, 3, 4096, 8, 512, 256))
    assert built.missing_phases == ("decode",)
    assert sorted({point.new_tokens for point in built.points}) == [1, 2]
    # The digest is that of the points the grid still has, not that of no points.
    compact = json.dumps(built.point_dicts(), sort_keys=True, separators=(",", ":"))
    assert built.digest == hashlib.sha256(compact.encode()).hexdigest() != empty_digest
    # A pool of two pages cannot hold one request (it needs ceilP(n) + P + 1 < S).
    assert grid.build_grid("prefill", grid.Limits(16, 100, 32, 8, 512, 256)).missing_phases == ("prefill",)


def test_build_grid_rejects_bad_arguments():
    limits = _limits(16)
    for mode in ("AGG", "both", "", None, 1):
        with pytest.raises(ValueError, match="mode"):
            grid.build_grid(mode, limits)
    with pytest.raises(TypeError, match="Limits"):
        grid.build_grid("agg", limits.as_dict())
    with pytest.raises(TypeError, match="SampleCaps"):
        grid.build_grid("agg", limits, {"prefill_max_new_token_samples": 4})
    for axis_function in (grid.prefill_new_token_axis, grid.decode_batch_axis):
        with pytest.raises(TypeError, match="SampleCaps"):
            axis_function(limits, None)


# ---------------------------------------------------------------------------------------------------------------
# What the KV pool takes from the generated prefill block
# ---------------------------------------------------------------------------------------------------------------


def test_pool_cut_hand_worked():
    """Page 1, I = 20, R = 2, M = 1000, chunking off, no graphs, no caps; the cached unit is 16 tokens.

    Without the pool: n is 1, 2, 4, 8, 16 and 19 (n < 20), each at batch 1 and 2, twelve pairs. A prefix of 16
    tokens fits beside n = 1 and n = 2 only (k + n < 20).

    With S = 36 the pool rule is b * (k + n + 2) < 36:
      (n=1, b=2) and (n=2, b=2): the prefix 16 needs 38 and 40, the empty prefix fits: two pairs end shorter.
      (n=16, b=2) and (n=19, b=2): 36 and 42 without a prefix: two pairs are left out.
      Every pair at batch 1 keeps its prefix (at most 19 + 2 = 21, and 16 + 2 + 2 = 20).
    The whole block needs 2 * (19 + 2) = 42 < S, so 43 tokens are the smallest pool that takes nothing.
    """
    without_pool = grid.Limits(1, 20, 1_000_000, 2, 1000, None)
    assert grid.prefill_new_token_axis(without_pool, NO_CAPS) == [1, 2, 4, 8, 16, 19]
    assert grid.prefill_pool_cut(without_pool, NO_CAPS) is None

    cut = grid.prefill_pool_cut(grid.Limits(1, 20, 36, 2, 1000, None), NO_CAPS)
    assert cut == grid.PoolCut(
        pool_tokens=36,
        pairs=12,
        dropped_pairs=2,
        shortened_pairs=2,
        whole_grid_pool_tokens=43,
        pool_sets_input_limit=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        cut.pairs = 0

    # One token more and the pair of two requests of 16 tokens fits again: 2 * 18 = 36 < 37.
    assert grid.prefill_pool_cut(grid.Limits(1, 20, 37, 2, 1000, None), NO_CAPS).dropped_pairs == 1
    # At 42 only the pair of two requests of 19 tokens is left out, and at 43 nothing is taken.
    assert grid.prefill_pool_cut(grid.Limits(1, 20, 42, 2, 1000, None), NO_CAPS) == grid.PoolCut(
        42, 12, 1, 0, 43, False
    )
    assert grid.prefill_pool_cut(grid.Limits(1, 20, 43, 2, 1000, None), NO_CAPS) is None


def test_pool_cut_when_the_pool_leaves_no_point():
    # One request of one token needs 1 * (0 + 1 + 1 + 1) = 3 < S.
    limits = grid.Limits(1, 20, 3, 2, 1000, None)
    assert grid.build_grid("prefill", limits, NO_CAPS).missing_phases == ("prefill",)
    assert grid.prefill_pool_cut(limits, NO_CAPS) == grid.PoolCut(
        pool_tokens=3,
        pairs=12,
        dropped_pairs=12,
        shortened_pairs=0,
        whole_grid_pool_tokens=43,
        pool_sets_input_limit=False,
    )


def test_pool_cut_counts_pairs_against_the_planned_block():
    """A pair that fits the pool but has no point in the planned block is left out, not shortened.

    Page 7, I = 33, S = 28, R = 1, M = 1, chunking off, and caps that keep four new-token values. Without the
    pool the new-token axis is 1, 4, 8 and 32. The pool lowers the largest n to 14, and the cap then keeps
    1, 2, 8 and 14: n = 4 fits the pool and is not planned, n = 32 does not fit.
    """
    limits = grid.Limits(7, 33, 28, 1, 1, None)
    caps = grid.SampleCaps(prefill_max_new_token_samples=4, prefill_max_kv_read_token_samples=2)
    far = dataclasses.replace(limits, max_total_num_tokens=7 * 10**6)
    assert grid.prefill_new_token_axis(far, caps) == [1, 4, 8, 32]
    assert grid.prefill_new_token_axis(limits, caps) == [1, 2, 8, 14]
    assert grid.prefill_unfit_reason(limits, 1, 4, 0) is None

    cut = grid.prefill_pool_cut(limits, caps)
    assert (cut.pairs, cut.dropped_pairs) == (4, 2)
    # The planned block holds two pairs that are not among the four: n = 2 and n = 14.
    planned = {(point.new_tokens, point.batch_size) for point in grid.build_grid("prefill", limits, caps).points}
    assert planned == {(1, 1), (2, 1), (8, 1), (14, 1)}


# Caps that cut the new-token axis, so that a pool which lowers the largest n makes the cut pick other values.
CUTTING_CAPS = grid.SampleCaps(
    prefill_max_new_token_samples=3, prefill_max_kv_read_token_samples=2, prefix_max_batch_size_samples=3
)


@pytest.mark.parametrize("page", [1, 3, 16, 24, 64])
@pytest.mark.parametrize(
    "caps", [NO_CAPS, grid.SampleCaps(), CUTTING_CAPS], ids=["no_caps", "default_caps", "cutting_caps"]
)
def test_pool_cut_agrees_with_the_two_planned_blocks_on_small_limits(page, caps):
    """The answer is checked against the planned blocks alone, with a pool far beyond the one the module uses.

    The pairs are those of the block for a pool of a million pages. A pair without a point in the block of the
    real pool is left out; a pair whose longest prefix there is shorter ends shorter. ``None`` is the answer
    exactly when the two blocks are the same list of points and the input limit is not the pool less 6. The
    pool the record names for the whole grid is tried: with it nothing is taken, and with one page less
    something is. Half of the limits have prefill graph sizes, so that a cap cuts the new-token axis, and a
    fifth have the input limit SGLang reports for a pool that is not above the context, down to a pool of 7
    tokens at page 1, whose input limit of 1 admits no request.
    """
    rng = random.Random(7100 + page)
    seen = {"none": 0, "dropped": 0, "shortened": 0, "input_limit": 0}
    for _ in range(300):
        limits = _random_limits(rng, page)
        if rng.random() < 0.5:
            limits = dataclasses.replace(limits, prefill_graph_sizes=(4, 8, 16, 32, 64))
        if rng.random() < 0.2:
            # A pool that is not above the context: SGLang then reports the pool less 6 as the input limit.
            pool = page * rng.randint(-(-7 // page), 96 // page + 2)
            limits = dataclasses.replace(limits, max_req_input_len=pool - 6, max_total_num_tokens=pool)
        far = dataclasses.replace(limits, max_total_num_tokens=page * 1_000_000)
        planned = grid.build_grid("prefill", limits, caps)
        without_pool = grid.build_grid("prefill", far, caps)
        cut = grid.prefill_pool_cut(limits, caps)
        sets_input_limit = limits.max_req_input_len + 6 == limits.max_total_num_tokens

        if planned.point_dicts() == without_pool.point_dicts() and not sets_input_limit:
            assert cut is None, limits
            seen["none"] += 1
            continue

        def longest(built):
            reach = {}
            for point in built.points:
                pair = (point.new_tokens, point.batch_size)
                reach[pair] = max(reach.get(pair, 0), point.cached_tokens)
            return reach

        wanted, kept = longest(without_pool), longest(planned)
        dropped = [pair for pair in wanted if pair not in kept]
        shortened = [pair for pair in wanted if pair in kept and kept[pair] < wanted[pair]]
        assert (cut.pool_tokens, cut.pairs) == (limits.max_total_num_tokens, len(wanted)), limits
        assert (cut.dropped_pairs, cut.shortened_pairs) == (len(dropped), len(shortened)), limits
        assert cut.pool_sets_input_limit == sets_input_limit, limits
        # The blocks differ exactly when a pair is left out or ends shorter.
        assert bool(dropped or shortened) == (planned.point_dicts() != without_pool.point_dicts()), limits

        # The pool the record names holds the whole block; one page less does not.
        need = cut.whole_grid_pool_tokens
        assert need % page == 0
        enough = dataclasses.replace(limits, max_total_num_tokens=need)
        assert grid.build_grid("prefill", enough, caps).point_dicts() == without_pool.point_dicts(), limits
        if need > page:
            short = dataclasses.replace(limits, max_total_num_tokens=need - page)
            assert grid.build_grid("prefill", short, caps).point_dicts() != without_pool.point_dicts(), limits
        if dropped or shortened:
            assert need > limits.max_total_num_tokens, limits

        seen["dropped"] += bool(dropped)
        seen["shortened"] += bool(shortened)
        seen["input_limit"] += sets_input_limit
    # The sample holds every answer: nothing taken, pairs left out, pairs that end shorter, and a pool that
    # also sets the input limit. A pool that leaves no point has its own test above.
    assert all(count > 5 for count in seen.values()), seen


@pytest.mark.parametrize("page", [1, 3, 16, 24, 64])
def test_a_shortened_pair_is_one_whose_longest_fitting_prefix_is_shorter(page):
    """Where the pool leaves the pairs of the block as they are, the reason function alone gives the count.

    For each pair the longest cached prefix is found by trying every multiple of the cached unit, once with the
    real pool and once with a pool of a million pages.
    """
    rng = random.Random(7300 + page)
    unit = grid.cached_unit(page)
    checked = 0
    for _ in range(300):
        limits = _random_limits(rng, page)
        far = dataclasses.replace(limits, max_total_num_tokens=page * 1_000_000)
        pairs = {(point.new_tokens, point.batch_size) for point in grid.build_grid("prefill", far, NO_CAPS).points}
        planned_pairs = {
            (point.new_tokens, point.batch_size) for point in grid.build_grid("prefill", limits, NO_CAPS).points
        }
        cut = grid.prefill_pool_cut(limits, NO_CAPS)
        if cut is None or pairs != planned_pairs:
            continue

        def longest(record, batch, new):
            prefixes = range(0, record.max_req_input_len + 2 * unit, unit)
            return _brute_largest(prefixes, lambda cached: _prefill_fits(record, batch, new, cached))[0]

        shorter = [pair for pair in pairs if longest(limits, pair[1], pair[0]) < longest(far, pair[1], pair[0])]
        assert (cut.dropped_pairs, cut.shortened_pairs) == (0, len(shorter)), limits
        checked += 1
    assert checked > 5


# The limits one rank reported in the DEP prefill cells of 2026-11-18 (Qwen1.5-MoE-A2.7B on H200 GPUs at SGLang
# 0.5.23, 8,192 tokens for each rank), typed here from the rank files of those runs, and the two caps the cell's
# command carried. The pool is the argument. The decode graph sizes are shortened to four; no number below
# depends on them.
MEASURED_PREFILL_GRAPH_SIZES = (
    *range(4, 33, 4),
    *range(48, 257, 16),
    *range(288, 513, 32),
    *range(576, 1025, 64),
    *range(1280, 4097, 256),
    *range(4608, 8193, 512),
)
MEASURED_CAPS = grid.SampleCaps(prefill_max_new_token_samples=8192, prefill_max_kv_read_token_samples=29)


def _measured_limits(pool: int, **overrides) -> grid.Limits:
    record = {
        "page_size": 1,
        "max_req_input_len": 8186,
        "max_total_num_tokens": pool,
        "max_running_requests": 4096,
        "max_prefill_tokens": 8192,
        "chunked_prefill_size": 8192,
        "decode_graph_sizes": (1, 2, 4, 8),
        "prefill_graph_sizes": MEASURED_PREFILL_GRAPH_SIZES,
    }
    record.update(overrides)
    return grid.Limits(**record)


def _measured_grid(pool: int) -> grid.Grid:
    return grid.build_grid("prefill", _measured_limits(pool), MEASURED_CAPS)


# The measured cells were collected with the earlier new-token axis, which had 1 in front of the graph sizes
# and no 2: 116 values, 305 pairs and 3,314 points. The axis of today also holds 2, the first batch that runs
# with a graph: 117 values, 308 pairs and 3,347 points. The tests below pin the grid of today and show, with
# the low end switched off, that the grid of the measured cells is the same grid less the group of n = 2.
MEASURED_EARLIER = {"new_values": 116, "pairs": 305, "points": 3314}
MEASURED_TODAY = {"new_values": 117, "pairs": 308, "points": 3347}


def _measured_counts(low_end) -> dict:
    """New-token values, pairs and points of the measured prefill block, counted without the module's axes.

    For a pool that ends no axis. The new-token axis is the low end, both sides of every graph size below
    ``n_limit`` and ``n_limit`` itself, which is 8,185: the prompt must stay below the input limit of 8,186.
    The chunk budget of 8,192 tokens holds ``8192 // n`` requests of ``n`` new tokens, and the batch axis is
    the first three of the powers of two up to that number and the number itself: ``min(3, 8192 // n)`` batch
    sizes. The cached prefixes of a pair are 0, the doublings of 16 and the longest one, the largest multiple
    of 16 with ``k + n <= 8185``; there are at most 11 of them, below the cap of 29.
    """
    below = [size for size in MEASURED_PREFILL_GRAPH_SIZES if size < 8185]
    new_values = sorted({*low_end, *below, *(size + 1 for size in below), 8185})
    pairs = points = 0
    for new in new_values:
        longest = (8185 - new) // 16 * 16
        prefixes = {0, longest, *(16 * 2**step for step in range(10) if 16 * 2**step <= longest)}
        pairs += min(3, 8192 // new)
        points += min(3, 8192 // new) * len(prefixes)
    return {"new_values": len(new_values), "pairs": pairs, "points": points}


def test_measured_grid_counts_follow_from_the_measured_limits():
    assert len(MEASURED_PREFILL_GRAPH_SIZES) == 58
    assert _measured_counts({1}) == MEASURED_EARLIER
    assert _measured_counts({1, 2}) == MEASURED_TODAY
    # n = 2 runs at the batch sizes 1, 2 and 4, each with 11 cached prefixes: 0, the nine doublings from 16
    # to 4,096, and 8,176, the longest that keeps the prompt below the input limit.
    assert MEASURED_TODAY["pairs"] == MEASURED_EARLIER["pairs"] + 3
    assert MEASURED_TODAY["points"] == MEASURED_EARLIER["points"] + 3 * 11
    assert len(grid.prefill_new_token_axis(_measured_limits(638_230), MEASURED_CAPS)) == 117
    prefixes = [0, *(16 * 2**step for step in range(9)), 8176]
    assert prefixes[-2:] == [4096, 8176] and len(prefixes) == 11
    assert sorted(shape for shape in _prefill_shapes(_measured_grid(638_230).points) if shape[1] == 2) == [
        (batch, 2, cached) for batch in (1, 2, 4) for cached in prefixes
    ]


@pytest.mark.parametrize("pool", [389_686, 529_112, 635_112, 638_230])
def test_no_pool_of_the_measured_dep_cells_cut_the_prefill_grid(pool, monkeypatch):
    """The pools of the four runs: two and four ranks, each with SGLang's own share and with a share given.

    None of them cut the grid the cells were collected with, and none cuts the grid of today, which has the
    33 points of n = 2 on top. The largest of those, four requests with a prefix of 8,176 tokens, needs
    4 * (8,176 + 2 + 2) = 32,720 pool tokens.
    """
    today = _measured_grid(pool)
    assert len(today.points) == 3347
    assert grid.prefill_pool_cut(_measured_limits(pool), MEASURED_CAPS) is None

    # As collected: without the two sides of the first step.
    monkeypatch.setattr(grid, "first_prefill_graph_tokens", lambda sizes: None)
    collected = _measured_grid(pool)
    assert len(collected.points) == 3314
    assert _prefill_shapes(collected.points) == [shape for shape in _prefill_shapes(today.points) if shape[1] != 2]
    assert grid.prefill_pool_cut(_measured_limits(pool), MEASURED_CAPS) is None


def test_the_smallest_pool_that_leaves_the_measured_grid_whole(monkeypatch):
    """Four requests of 8,185 tokens need 4 * (8,185 + 2) = 32,748 pool tokens, strictly below the pool.

    The 8,185 tokens are a cached prefix of 8,176 and 9 new tokens, or one of 8,160 and 25. One token less
    and the grid still has its 3,347 points, so a point count does not show the cut: two pairs end at a
    shorter cached prefix. The pool is the same for the grid the cells were collected with: the points of
    n = 2 need less, 32,720 tokens at most.
    """
    assert grid.prefill_pool_cut(_measured_limits(32_749), MEASURED_CAPS) is None
    cut = grid.prefill_pool_cut(_measured_limits(32_748), MEASURED_CAPS)
    assert cut == grid.PoolCut(
        pool_tokens=32_748,
        pairs=308,
        dropped_pairs=0,
        shortened_pairs=2,
        whole_grid_pool_tokens=32_749,
        pool_sets_input_limit=False,
    )
    assert len(_measured_grid(32_748).points) == 3347
    # The two pairs: four requests of 9 and of 25 new tokens lose their longest prefix.
    lost = set(_prefill_shapes(_measured_grid(32_749).points)) - set(_prefill_shapes(_measured_grid(32_748).points))
    assert sorted(lost) == [(4, 9, 8176), (4, 25, 8160)]

    # As collected, without the two sides of the first step: the same pool, and the three pairs of n = 2 fewer.
    monkeypatch.setattr(grid, "first_prefill_graph_tokens", lambda sizes: None)
    assert grid.prefill_pool_cut(_measured_limits(32_749), MEASURED_CAPS) is None
    assert grid.prefill_pool_cut(_measured_limits(32_748), MEASURED_CAPS) == dataclasses.replace(cut, pairs=305)
    assert len(_measured_grid(32_748).points) == 3314


def test_pool_cut_when_the_input_limit_admits_no_request():
    """An input limit of 1 leaves both blocks empty. With a pool of 7 tokens that is still a record."""
    for page, whole in ((1, 1), (7, 7)):
        limits = grid.Limits(page, 1, 7, 2, 100, None)
        assert grid.build_grid("prefill", limits, grid.SampleCaps()).missing_phases == ("prefill",)
        assert grid.prefill_pool_cut(limits) == grid.PoolCut(
            pool_tokens=7,
            pairs=0,
            dropped_pairs=0,
            shortened_pairs=0,
            whole_grid_pool_tokens=whole,
            pool_sets_input_limit=True,
        )
    # The same input limit under a larger pool: the pool is above the context, and nothing is said.
    assert grid.prefill_pool_cut(grid.Limits(1, 1, 8, 2, 100, None)) is None


def test_a_pool_equal_to_the_context_gives_the_record_of_a_pool_below_it(monkeypatch):
    """The measured engine has a context of 8,192, so I = 8,186 whatever the pool from 8,192 up.

    A pool of exactly 8,192 tokens also gives I = S - 6, and the limits do not tell it from a pool below a
    longer context. The record is the same kind, and here a pool of the 32,749 tokens it names does hold the
    whole grid: the input limit stays where it is.

    The counts. One request reaches the input limit before this pool, so the 117 pairs at batch 1 keep their
    prefix. Of the 191 pairs at a larger batch two are left out, two requests of 4,096 new tokens
    (2 * 4,098 = 8,196) and four of 2,048 (4 * 2,050 = 8,200), and the other 189 end shorter. The pairs of
    n = 2 at the batch sizes 2 and 4 are two of them: without them the counts are 305 and 187.
    """
    assert MEASURED_TODAY["pairs"] - MEASURED_TODAY["new_values"] - 2 == 189
    cut = grid.prefill_pool_cut(_measured_limits(8_192), MEASURED_CAPS)
    assert cut == grid.PoolCut(
        pool_tokens=8_192,
        pairs=308,
        dropped_pairs=2,
        shortened_pairs=189,
        whole_grid_pool_tokens=32_749,
        pool_sets_input_limit=True,
    )
    planned = {(batch, new) for batch, new, _cached in _prefill_shapes(_measured_grid(8_192).points)}
    wanted = {(batch, new) for batch, new, _cached in _prefill_shapes(_measured_grid(32_749).points)}
    assert sorted(wanted - planned) == [(2, 4096), (4, 2048)]
    assert grid.prefill_pool_cut(_measured_limits(32_749), MEASURED_CAPS) is None

    monkeypatch.setattr(grid, "first_prefill_graph_tokens", lambda sizes: None)
    assert MEASURED_EARLIER["pairs"] - MEASURED_EARLIER["new_values"] - 2 == 187
    assert grid.prefill_pool_cut(_measured_limits(8_192), MEASURED_CAPS) == dataclasses.replace(
        cut, pairs=305, shortened_pairs=187
    )


def test_a_pool_that_is_not_above_the_context_also_sets_the_input_limit(monkeypatch):
    """SGLang reports an input limit of the pool less 6 when the pool is not above the model's context.

    With a context of 40,960 and a pool of 30,000 the engine reports I = 29,994. The record says that the
    input limit is the pool less 6: the block "without the pool" is planned for that I, so the 191 of 308 pairs
    and the 119,981 tokens are for the grid of that input limit. A pool of 119,981 tokens would be above this
    context and would raise the input limit to 40,954, which gives another, longer grid.

    The counts. The new-token axis ends at the chunk budget of 8,192 here and has as many values and pairs as
    the measured one. One request reaches the input limit before the pool, and every pair at a larger batch
    ends shorter: 308 - 117 = 191. The largest point is four requests with a prompt of 29,993 tokens, which
    hold 4 * (29,993 + 2) = 119,980 tokens. Without the pairs of n = 2 the counts are 305 and 189.
    """
    limits = _measured_limits(30_000, max_req_input_len=29_994)
    assert grid.largest_prefill_new_tokens(limits) == 8192
    assert len(grid.prefill_new_token_axis(limits, MEASURED_CAPS)) == MEASURED_TODAY["new_values"]
    assert MEASURED_TODAY["pairs"] - MEASURED_TODAY["new_values"] == 191
    cut = grid.prefill_pool_cut(limits, MEASURED_CAPS)
    assert cut == grid.PoolCut(
        pool_tokens=30_000,
        pairs=308,
        dropped_pairs=0,
        shortened_pairs=191,
        whole_grid_pool_tokens=119_981,
        pool_sets_input_limit=True,
    )
    with monkeypatch.context() as patch:
        patch.setattr(grid, "first_prefill_graph_tokens", lambda sizes: None)
        assert MEASURED_EARLIER["pairs"] - MEASURED_EARLIER["new_values"] == 189
        assert grid.prefill_pool_cut(limits, MEASURED_CAPS) == dataclasses.replace(cut, pairs=305, shortened_pairs=189)
    # With batch 1 alone the pool rule never binds before the input limit, and the two blocks are the same
    # list of points. The record is still given, because the input limit itself can come from the pool.
    one_batch = dataclasses.replace(MEASURED_CAPS, prefix_max_batch_size_samples=1)
    alone = grid.prefill_pool_cut(limits, one_batch)
    assert (alone.dropped_pairs, alone.shortened_pairs, alone.pool_sets_input_limit) == (0, 0, True)
    # One token of pool more, with the same reported input limit, and the pool is above the context.
    assert grid.prefill_pool_cut(_measured_limits(30_001, max_req_input_len=29_994), one_batch) is None
    assert grid._INPUT_LIMIT_BELOW_POOL == 6


def test_pool_cut_follows_the_caps_and_takes_the_default_caps_for_none():
    limits = grid.Limits(1, 20, 36, 2, 1000, None)
    assert grid.prefill_pool_cut(limits) == grid.prefill_pool_cut(limits, grid.SampleCaps())
    # With one batch size for every n the block holds batch 1 only, and no pair at batch 1 reaches the pool.
    one_batch = dataclasses.replace(NO_CAPS, prefix_max_batch_size_samples=1)
    assert grid.prefill_pool_cut(limits, one_batch) is None
    assert grid.prefill_pool_cut(limits, NO_CAPS) is not None


def test_pool_cut_takes_the_other_limits_as_they_are_and_leaves_the_record_alone():
    limits = _limits(16, max_total_num_tokens=grid.floor_pages(2_000, 16))
    before = limits.as_dict()
    cut = grid.prefill_pool_cut(limits, NO_CAPS)
    assert cut is not None and cut.pool_tokens == 2_000 // 16 * 16 and not cut.pool_sets_input_limit
    assert limits.as_dict() == before
    # The block without the pool is the block of the same record with any pool that no point reaches.
    far_block = grid.build_grid("prefill", dataclasses.replace(limits, max_total_num_tokens=16 * 10**9), NO_CAPS)
    assert cut.pairs == len({(point.new_tokens, point.batch_size) for point in far_block.points})
    enough = dataclasses.replace(limits, max_total_num_tokens=cut.whole_grid_pool_tokens)
    assert grid.build_grid("prefill", enough, NO_CAPS).point_dicts() == far_block.point_dicts()
    assert grid.prefill_pool_cut(enough, NO_CAPS) is None


def test_pool_cut_rejects_bad_arguments():
    limits = _limits(16)
    with pytest.raises(TypeError, match="Limits"):
        grid.prefill_pool_cut(limits.as_dict())
    with pytest.raises(TypeError, match="SampleCaps"):
        grid.prefill_pool_cut(limits, {"prefix_max_batch_size_samples": 1})


def test_the_decode_block_reaches_the_pool_with_the_default_request_cap_and_not_with_a_small_one():
    """Why only the prefill block is reported.

    With the largest pool of the measured cells and SGLang's request cap of 4,096, 256 decode requests can
    hold a context of 2,488 tokens each, far below the input limit of 8,186: the pool ends the decode context
    axis, while the prefill block of the same limits is untouched. With a request cap of 64 the largest decode
    batch reaches the input limit, and the pool ends nothing.
    """
    limits = _measured_limits(638_230)
    assert grid.largest_decode_context(limits, 256) == (638_230 - 1) // 256 - 1 - 7 + 3 == 2488
    assert grid.largest_decode_context(limits, 256) < limits.max_req_input_len
    assert grid.prefill_pool_cut(limits, MEASURED_CAPS) is None
    capped = _measured_limits(638_230, max_running_requests=64)
    assert grid.largest_decode_batch(capped) == 64
    assert grid.largest_decode_context(capped, 64) == capped.max_req_input_len == 8186


# ---------------------------------------------------------------------------------------------------------------
# Point and digest
# ---------------------------------------------------------------------------------------------------------------


def test_point_dict_and_stamps():
    cached = grid.Point(7, "prefill", 4, 32, 64)
    assert cached.as_dict() == {
        "point_type": "prefill",
        "benchmark_id": 7,
        "total_prefill_tokens": 128,
        "total_kv_read_tokens": 256,
        "batch_size": 4,
        "sample_reasons": ["prefill_real_seed"],
    }
    assert grid.Point(1, "prefill", 4, 32, 0).as_dict()["sample_reasons"] == []
    # One cached token per request is the smallest prefix (legal at page size 1) and already a seeded one.
    assert grid.Point(1, "prefill", 1, 4, 1).as_dict()["sample_reasons"] == ["prefill_real_seed"]
    assert grid.Point(1, "prefill", 1, 4, 1, explicit=True).as_dict()["sample_reasons"] == [
        "explicit",
        "prefill_real_seed",
    ]
    assert grid.Point(1, "prefill", 4, 32, 0, explicit=True).as_dict()["sample_reasons"] == ["explicit"]
    assert grid.Point(1, "prefill", 4, 32, 64, explicit=True).as_dict()["sample_reasons"] == [
        "explicit",
        "prefill_real_seed",
    ]
    decode = grid.Point(2, "decode", 3, 0, 100, explicit=True)
    assert decode.as_dict() == {
        "point_type": "decode",
        "benchmark_id": 2,
        "total_prefill_tokens": 0,
        "total_kv_read_tokens": 300,
        "batch_size": 3,
        "sample_reasons": ["explicit", "kvwarm_real_kv"],
    }
    assert grid.Point(2, "decode", 3, 0, 100).as_dict()["sample_reasons"] == ["kvwarm_real_kv"]
    # A fresh list every time: a caller that edits one dict cannot change the next.
    cached.as_dict()["sample_reasons"].append("edited")
    assert cached.as_dict()["sample_reasons"] == ["prefill_real_seed"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cached.batch_size = 5


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ((0, "prefill", 1, 1, 0), ValueError),
        ((True, "prefill", 1, 1, 0), TypeError),
        ((1, "agg", 1, 1, 0), ValueError),
        ((1, "prefill", 0, 1, 0), ValueError),
        ((1, "prefill", 1, 0, 0), ValueError),
        ((1, "prefill", 1, 1, -1), ValueError),
        ((1, "prefill", 1, 1.0, 0), TypeError),
        ((1, "decode", 1, 1, 8), ValueError),  # a decode point has no new tokens
        ((1, "decode", 1, 0, 0), ValueError),
        ((1, "decode", 1, 0, True), TypeError),
        ((1, "decode", 1, 0, 8, 1), TypeError),  # explicit must be a bool
    ],
)
def test_point_rejects_a_shape_that_is_not_a_point(arguments, error):
    with pytest.raises(error):
        grid.Point(*arguments)


def test_point_checks_its_phase_and_a_decode_point_may_hold_one_token():
    # In every other field this is a valid decode point, so only the phase check can refuse it.
    with pytest.raises(ValueError, match="phase must be 'prefill' or 'decode', got 'agg'"):
        grid.Point(1, "agg", 1, 0, 8)
    # The lower bound of a decode Point is one token per request, as in the manifest schema. Whether such a
    # context can be staged is the reason function's question (it cannot: context_below_minimum).
    assert grid.Point(1, "decode", 2, 0, 1).as_dict()["total_kv_read_tokens"] == 2
    with pytest.raises(ValueError, match="decode cached_tokens must be >= 1, got 0"):
        grid.Point(1, "decode", 2, 0, 0)


def test_grid_is_frozen():
    built = grid.build_grid("decode", _limits(16))
    with pytest.raises(dataclasses.FrozenInstanceError):
        built.digest = "edited"
    with pytest.raises(dataclasses.FrozenInstanceError):
        built.points = ()


def test_digest_is_the_documented_hash_and_tracks_points_order_and_ids():
    # "Documented" is the docstring of ``grid_digest``: SHA-256 over the point dicts as compact JSON, sorted keys.
    limits = _limits(16)
    first = grid.build_grid("agg", limits)
    second = grid.build_grid("agg", limits)
    dicts = first.point_dicts()
    assert first.digest == second.digest and first == second
    assert first.digest == hashlib.sha256(json.dumps(dicts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert len(first.digest) == 64 and grid.grid_digest(tuple(dicts)) == first.digest

    def changed(mutate) -> str:
        copy = json.loads(json.dumps(dicts))
        mutate(copy)
        return grid.grid_digest(copy)

    def bump_total(copy):
        copy[3]["total_kv_read_tokens"] += 16

    def swap_two(copy):
        copy[0], copy[1] = copy[1], copy[0]

    def swap_two_ids(copy):
        copy[0]["benchmark_id"], copy[1]["benchmark_id"] = copy[1]["benchmark_id"], copy[0]["benchmark_id"]

    def renumber_from_zero(copy):
        for record in copy:
            record["benchmark_id"] -= 1

    def drop_last(copy):
        copy.pop()

    def add_a_stamp(copy):
        copy[0]["sample_reasons"].append("explicit")

    digests = {changed(m) for m in (bump_total, swap_two, swap_two_ids, renumber_from_zero, drop_last, add_a_stamp)}
    assert len(digests) == 6 and first.digest not in digests
    # Different caps or limits plan different points, hence a different digest.
    assert grid.build_grid("agg", limits, grid.SampleCaps(prefix_max_batch_size_samples=1)).digest != first.digest
    assert grid.build_grid("agg", _limits(16, max_running_requests=39)).digest != first.digest


# ---------------------------------------------------------------------------------------------------------------
# Points manifest: schema
# ---------------------------------------------------------------------------------------------------------------


def _manifest(prefill=(), decode=(), version=1) -> str:
    return json.dumps({"schema_version": version, "prefill": list(prefill), "decode": list(decode)})


def _prefill_point(batch=2, new=64, cached=64, **extra) -> dict:
    return {"batch_size": batch, "total_prefill_tokens": new, "total_kv_read_tokens": cached, **extra}


def _decode_point(batch=2, context=200, **extra) -> dict:
    return {"batch_size": batch, "total_kv_read_tokens": context, **extra}


PARTITION = {"axis": "new", "high_count": 1, "fraction": 0.5}

MANIFEST_ACCEPTS = {
    "v1": (_manifest([_prefill_point()], [_decode_point()], 1), [(2, 64, 64)], [(2, 200)]),
    "v2": (_manifest([_prefill_point()], [_decode_point()], 2), [(2, 64, 64)], [(2, 200)]),
    "v3": (_manifest([_prefill_point()], [_decode_point()], 3), [(2, 64, 64)], [(2, 200)]),
    "v1 null partition and rows": (
        _manifest([_prefill_point(partition=None, rows=None)], [], 1),
        [(2, 64, 64)],
        [],
    ),
    "v2 null partition": (_manifest([_prefill_point(partition=None)], [], 2), [(2, 64, 64)], []),
    "v3 null rows": (_manifest([_prefill_point(rows=None)], [], 3), [(2, 64, 64)], []),
    "v3 uniform rows": (_manifest([_prefill_point(rows=[[32, 32], [32, 32]])], [], 3), [(2, 64, 64)], []),
    "v3 uniform rows without a prefix": (_manifest([_prefill_point(1, 5, 0, rows=[[5, 0]])], [], 3), [(1, 5, 0)], []),
    "decode with total_prefill_tokens 0": (
        _manifest([], [_decode_point(total_prefill_tokens=0)], 1),
        [],
        [(2, 200)],
    ),
    # The schema does not ask for even totals or a minimum context; build_grid does.
    "uneven totals": (
        _manifest([_prefill_point(4, 257, 512)], [_decode_point(4, 513)], 1),
        [(4, 257, 512)],
        [(4, 513)],
    ),
    "one token per decode request": (_manifest([], [_decode_point(2, 2)], 1), [], [(2, 2)]),
    "prefill prefix smaller than the batch": (_manifest([_prefill_point(4, 8, 2)], [], 1), [(4, 8, 2)], []),
    # The same numbers in both phases, and points that differ in one coordinate, are not duplicates.
    "near duplicates": (
        _manifest(
            [_prefill_point(2, 64, 64), _prefill_point(2, 64, 0), _prefill_point(2, 66, 64), _prefill_point(4, 64, 64)],
            [_decode_point(2, 64), _decode_point(2, 66), _decode_point(4, 64)],
            1,
        ),
        [(2, 64, 64), (2, 64, 0), (2, 66, 64), (4, 64, 64)],
        [(2, 64), (2, 66), (4, 64)],
    ),
}


@pytest.mark.parametrize("name", sorted(MANIFEST_ACCEPTS))
def test_manifest_accepts(name, tmp_path):
    raw, prefill, decode = MANIFEST_ACCEPTS[name]
    version = json.loads(raw)["schema_version"]
    expected = {"schema_version": version, "prefill": prefill, "decode": decode}
    assert grid.parse_points_manifest(raw) == expected
    assert grid.parse_points_manifest(raw.encode("utf-8")) == expected
    parsed = grid.parse_points_manifest(raw)
    assert all(type(entry) is tuple for phase in ("prefill", "decode") for entry in parsed[phase])
    # AISim's freezer accepts each of these manifests too. The parser is looser than the freezer in one case, a
    # manifest whose two lists are both empty: the parser accepts it (see the next test) and the freezer refuses it.
    path = tmp_path / "points.json"
    path.write_text(raw, encoding="utf-8")
    _freeze_benchmark_points(str(path))


def test_manifest_with_both_lists_empty_is_accepted_by_the_parser():
    assert grid.parse_points_manifest(_manifest()) == {"schema_version": 1, "prefill": [], "decode": []}


def test_manifest_keeps_file_order():
    points = [_prefill_point(1, 8, 0), _prefill_point(4, 64, 64), _prefill_point(2, 16, 32)]
    parsed = grid.parse_points_manifest(_manifest(points, [_decode_point(1, 4), _decode_point(8, 800)], 2))
    assert parsed["prefill"] == [(1, 8, 0), (4, 64, 64), (2, 16, 32)]
    assert parsed["decode"] == [(1, 4), (8, 800)]


UNEVEN_ROWS = [[30, 32], [34, 32]]

MANIFEST_REJECTS = {
    # JSON level.
    "invalid json": ('{"schema_version": 1,', "not valid JSON"),
    "empty": ("", "not valid JSON"),
    "duplicate top-level key": ('{"schema_version": 1, "prefill": [], "decode": [], "decode": []}', "duplicate key"),
    "duplicate point key": (
        '{"schema_version": 1, "prefill": [], "decode": [{"batch_size": 1, "batch_size": 2, "total_kv_read_tokens": 8}]'
        "}",
        "duplicate key 'batch_size'",
    ),
    "duplicate nested key": (
        '{"schema_version": 2, "prefill": [{"batch_size": 2, "total_prefill_tokens": 8, "total_kv_read_tokens": 0,'
        ' "partition": {"axis": "new", "axis": "kv"}}], "decode": []}',
        "duplicate key 'axis'",
    ),
    "NaN": ('{"schema_version": 1, "prefill": [], "decode": [{"batch_size": 1, "total_kv_read_tokens": NaN}]}', "NaN"),
    "Infinity": (
        '{"schema_version": 1, "prefill": [], "decode": [{"batch_size": 1, "total_kv_read_tokens": Infinity}]}',
        "Infinity",
    ),
    "-Infinity": (
        '{"schema_version": 1, "prefill": [], "decode": [{"batch_size": 1, "total_kv_read_tokens": -Infinity}]}',
        "-Infinity",
    ),
    # Envelope.
    "top-level list": ("[]", "exactly the keys"),
    "top-level number": ("1", "exactly the keys"),
    "missing decode": ('{"schema_version": 1, "prefill": []}', "exactly the keys"),
    "missing schema_version": ('{"prefill": [], "decode": []}', "exactly the keys"),
    "extra top-level key": ('{"schema_version": 1, "prefill": [], "decode": [], "points": []}', "exactly the keys"),
    "schema_version true": ('{"schema_version": true, "prefill": [], "decode": []}', "schema_version"),
    "schema_version float": ('{"schema_version": 1.0, "prefill": [], "decode": []}', "schema_version"),
    "schema_version string": ('{"schema_version": "1", "prefill": [], "decode": []}', "schema_version"),
    "schema_version null": ('{"schema_version": null, "prefill": [], "decode": []}', "schema_version"),
    "schema_version 0": (_manifest(version=0), "schema_version"),
    "schema_version 4": (_manifest(version=4), "schema_version"),
    "prefill is an object": ('{"schema_version": 1, "prefill": {}, "decode": []}', "prefill: must be a list"),
    "decode is null": ('{"schema_version": 1, "prefill": [], "decode": null}', "decode: must be a list"),
    "decode entry is a number": (
        '{"schema_version": 1, "prefill": [], "decode": [8]}',
        r"decode\[0\]: must be an object",
    ),
    "prefill entry is a list": (
        '{"schema_version": 1, "prefill": [{"batch_size": 1, "total_prefill_tokens": 1, "total_kv_read_tokens": 0},'
        ' [1, 1, 0]], "decode": []}',
        r"prefill\[1\]: must be an object",
    ),
    # Prefill points.
    "prefill unknown key": (_manifest([_prefill_point(note="x")]), r"prefill\[0\]: unknown key\(s\) 'note'"),
    "prefill missing batch_size": (
        _manifest([{"total_prefill_tokens": 8, "total_kv_read_tokens": 0}]),
        r"prefill\[0\]: missing key\(s\) batch_size",
    ),
    "prefill missing total_prefill_tokens": (
        _manifest([{"batch_size": 1, "total_kv_read_tokens": 0}]),
        r"prefill\[0\]: missing key\(s\) total_prefill_tokens",
    ),
    "prefill missing total_kv_read_tokens": (
        _manifest([{"batch_size": 1, "total_prefill_tokens": 8}]),
        r"prefill\[0\]: missing key\(s\) total_kv_read_tokens",
    ),
    "prefill bool batch": (_manifest([_prefill_point(batch=True)]), r"prefill\[0\]: batch_size must be an integer"),
    "prefill float new": (_manifest([_prefill_point(new=64.0)]), r"prefill\[0\]: total_prefill_tokens must be an int"),
    "prefill string cached": (_manifest([_prefill_point(cached="64")]), r"prefill\[0\]: total_kv_read_tokens must be"),
    "prefill null batch": (_manifest([_prefill_point(batch=None)]), r"prefill\[0\]: batch_size must be an integer"),
    "prefill bool cached": (_manifest([_prefill_point(cached=False)]), r"prefill\[0\]: total_kv_read_tokens must be"),
    "prefill batch 0": (_manifest([_prefill_point(batch=0)]), r"prefill\[0\]: batch_size must be >= 1"),
    "prefill new below batch": (
        _manifest([_prefill_point(), _prefill_point(4, 3, 0)]),
        r"prefill\[1\]: total_prefill_tokens must be >= batch_size",
    ),
    "prefill negative cached": (
        _manifest([_prefill_point(cached=-16)]),
        r"prefill\[0\]: total_kv_read_tokens must be >=",
    ),
    "prefill cached -1": (
        _manifest([_prefill_point(cached=-1)]),
        r"prefill\[0\]: total_kv_read_tokens must be >= 0, got -1",
    ),
    # Uneven requests.
    "partition at v1": (_manifest([_prefill_point(partition=PARTITION)], version=1), r"prefill\[0\]: partition"),
    "partition at v2": (_manifest([_prefill_point(partition=PARTITION)], version=2), r"prefill\[0\]: partition"),
    "partition at v3": (
        _manifest([_prefill_point(), _prefill_point(4, 64, 0, partition=PARTITION)], version=3),
        r"prefill\[1\]: partition",
    ),
    "empty partition object": (_manifest([_prefill_point(partition={})], version=2), r"prefill\[0\]: partition"),
    "uniform rows at v1": (
        _manifest([_prefill_point(rows=[[32, 32], [32, 32]])], version=1),
        "rows needs schema_version 3",
    ),
    "uniform rows at v2": (
        _manifest([_prefill_point(rows=[[32, 32], [32, 32]])], version=2),
        "rows needs schema_version 3",
    ),
    "uneven rows": (_manifest([_prefill_point(rows=UNEVEN_ROWS)], version=3), r"prefill\[0\]: rows must be 2 rows"),
    "rows of the wrong count": (_manifest([_prefill_point(rows=[[32, 32]])], version=3), "rows must be 2 rows"),
    "rows with an extra row": (_manifest([_prefill_point(rows=[[32, 32]] * 3)], version=3), "rows must be 2 rows"),
    "rows is an object": (_manifest([_prefill_point(rows={"0": [32, 32]})], version=3), "rows must be 2 rows"),
    "rows is empty": (_manifest([_prefill_point(rows=[])], version=3), "rows must be 2 rows"),
    "row is not a list": (_manifest([_prefill_point(rows=[[32, 32], "32,32"])], version=3), "rows must be 2 rows"),
    "row of three": (_manifest([_prefill_point(rows=[[32, 32, 0], [32, 32, 0]])], version=3), "rows must be 2 rows"),
    "row with a float": (_manifest([_prefill_point(rows=[[32.0, 32], [32.0, 32]])], version=3), "rows must be 2 rows"),
    "row with a bool": (_manifest([_prefill_point(1, 1, 0, rows=[[True, 0]])], version=3), "rows must be 1 rows"),
    # Rows cannot restate an even split that does not exist: 65 new tokens over 2 requests.
    "rows on uneven totals": (
        _manifest([_prefill_point(2, 65, 64, rows=[[32, 32], [32, 32]])], version=3),
        "rows must",
    ),
    # Decode points.
    "decode unknown key": (_manifest([], [_decode_point(context_tokens=4)]), r"decode\[0\]: unknown key\(s\)"),
    "decode rows": (
        _manifest([], [_decode_point(rows=[[0, 100], [0, 100]])], 3),
        r"decode\[0\]: unknown key\(s\) 'rows'",
    ),
    "decode null rows": (_manifest([], [_decode_point(rows=None)], 3), r"decode\[0\]: unknown key\(s\) 'rows'"),
    "decode null partition": (_manifest([], [_decode_point(partition=None)], 2), r"decode\[0\]: unknown key\(s\)"),
    "decode missing batch_size": (
        _manifest([], [{"total_kv_read_tokens": 8}]),
        r"decode\[0\]: missing key\(s\) batch_size",
    ),
    "decode missing total_kv_read_tokens": (
        _manifest([], [_decode_point(), {"batch_size": 1}]),
        r"decode\[1\]: missing key\(s\) total_kv_read_tokens",
    ),
    "decode bool batch": (
        _manifest([], [_decode_point(batch=True, context=8)]),
        r"decode\[0\]: batch_size must be an int",
    ),
    "decode float context": (_manifest([], [_decode_point(context=8.0)]), r"decode\[0\]: total_kv_read_tokens must be"),
    "decode batch 0": (_manifest([], [_decode_point(batch=0)]), r"decode\[0\]: batch_size must be >= 1"),
    "decode context below batch": (
        _manifest([], [_decode_point(2, 1)]),
        r"decode\[0\]: total_kv_read_tokens must be >=",
    ),
    "decode total_prefill_tokens 1": (_manifest([], [_decode_point(total_prefill_tokens=1)]), "must be 0"),
    "decode total_prefill_tokens false": (_manifest([], [_decode_point(total_prefill_tokens=False)]), "must be 0"),
    "decode total_prefill_tokens 0.0": (_manifest([], [_decode_point(total_prefill_tokens=0.0)]), "must be 0"),
    "decode total_prefill_tokens null": (_manifest([], [_decode_point(total_prefill_tokens=None)]), "must be 0"),
    # Duplicates: the error names the later index and the earlier one.
    "duplicate prefill point": (
        _manifest([_prefill_point(), _prefill_point(4, 64, 64), _prefill_point()]),
        r"prefill\[2\]: duplicate of prefill\[0\]",
    ),
    "duplicate decode point": (
        _manifest([], [_decode_point(), _decode_point()]),
        r"decode\[1\]: duplicate of decode\[0\]",
    ),
    "duplicate decode point, one with total_prefill_tokens 0": (
        _manifest([], [_decode_point(4, 64), _decode_point(), _decode_point(total_prefill_tokens=0)]),
        r"decode\[2\]: duplicate of decode\[1\]",
    ),
    "duplicate prefill point, one with uniform rows": (
        _manifest([_prefill_point(rows=[[32, 32], [32, 32]]), _prefill_point(rows=None)], version=3),
        r"prefill\[1\]: duplicate of prefill\[0\]",
    ),
}


@pytest.mark.parametrize("name", sorted(MANIFEST_REJECTS))
def test_manifest_rejects(name):
    raw, match = MANIFEST_REJECTS[name]
    with pytest.raises(ValueError, match=match):
        grid.parse_points_manifest(raw)
    with pytest.raises(ValueError, match=match):
        grid.parse_points_manifest(raw.encode("utf-8"))


def test_manifest_rejects_undecodable_bytes_and_other_types():
    for raw in (b'{"schema_version": \xff}', b"\xff\xfe\xfa{}"):
        with pytest.raises(ValueError, match="not valid JSON"):
            grid.parse_points_manifest(raw)
    for raw in (None, {"schema_version": 1, "prefill": [], "decode": []}, 1, ["{}"], Path("points.json")):
        with pytest.raises(TypeError, match="str or bytes"):
            grid.parse_points_manifest(raw)


@pytest.mark.parametrize(
    "encoding", ["utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "utf-32", "utf-32-le", "utf-32-be"]
)
def test_manifest_bytes_must_be_utf8_without_a_byte_order_mark(encoding, tmp_path):
    """AISim's freezer reads the file as UTF-8 text. The parser must refuse every encoding the freezer refuses."""
    text = _manifest([_prefill_point()], [_decode_point()])
    assert grid.parse_points_manifest(text.encode("utf-8"))["prefill"] == [(2, 64, 64)]
    raw = text.encode(encoding)
    assert raw != text.encode("utf-8")
    with pytest.raises(ValueError, match="benchmark points manifest is not valid JSON"):
        grid.parse_points_manifest(raw)
    path = tmp_path / "points.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):  # JSONDecodeError or UnicodeDecodeError, depending on the encoding
        _freeze_benchmark_points(str(path))


def test_manifest_text_and_bytes_get_the_same_verdict():
    text = _manifest([_prefill_point()], [_decode_point()])
    # A byte-order mark is refused in a str as it is in bytes.
    with pytest.raises(ValueError, match="benchmark points manifest is not valid JSON"):
        grid.parse_points_manifest("\ufeff" + text)
    # Bytes are read as UTF-8 and as nothing else. The byte E9 is "e acute" in Latin-1 and not UTF-8, so this
    # is invalid JSON and not a manifest with an unknown top-level key.
    with pytest.raises(ValueError, match="benchmark points manifest is not valid JSON"):
        grid.parse_points_manifest(b'{"schema_version": 1, "prefill": [], "decode": [], "caf\xe9": []}')
    # A key outside ASCII arrives as the same key whether the manifest is a str or its UTF-8 bytes, and whether
    # the two characters are written out (two bytes each in UTF-8) or as JSON escapes.
    document = {"schema_version": 1, "prefill": [_prefill_point(**{"gr\u00f6\u00dfe": 1})], "decode": []}
    written_out = json.dumps(document, ensure_ascii=False)
    escaped = json.dumps(document)
    assert len(written_out.encode("utf-8")) == len(written_out) + 2 and escaped.isascii()
    for raw in (written_out, written_out.encode("utf-8"), escaped, escaped.encode("utf-8")):
        with pytest.raises(ValueError) as caught:
            grid.parse_points_manifest(raw)
        assert str(caught.value) == "prefill[0]: unknown key(s) 'gr\u00f6\u00dfe'"


def test_manifest_integer_longer_than_the_interpreter_digit_limit_is_reported_as_invalid_json():
    """``json.loads`` raises a plain ``ValueError`` for such a literal; the caller must still learn what failed."""
    previous = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4300)  # the interpreter's default, set here so the test does not depend on it
    try:
        head = '{"schema_version": 1, "prefill": [], "decode": [{"batch_size": 1, "total_kv_read_tokens": '
        raw = head + "9" * 5000 + "}]}"
        for manifest in (raw, raw.encode("utf-8")):
            with pytest.raises(ValueError, match="benchmark points manifest is not valid JSON: Exceeds the limit"):
                grid.parse_points_manifest(manifest)
        # A literal inside the limit is an ordinary integer and passes the schema.
        assert grid.parse_points_manifest(head + "9" * 40 + "}]}")["decode"] == [(1, int("9" * 40))]
    finally:
        sys.set_int_max_str_digits(previous)


def test_manifest_nested_deeper_than_the_interpreter_allows_is_reported_as_invalid_json():
    """``json.loads`` raises ``RecursionError``, which is not a ``ValueError``; the caller must still get one."""
    depth = 200_000
    raw = '{"schema_version": 1, "prefill": ' + "[" * depth + "]" * depth + ', "decode": []}'
    for manifest in (raw, raw.encode("utf-8")):
        with pytest.raises(ValueError, match="^benchmark points manifest is not valid JSON: ") as caught:
            grid.parse_points_manifest(manifest)
        assert type(caught.value) is ValueError
        assert isinstance(caught.value.__cause__, RecursionError)
        assert len(str(caught.value)) < 1000


def test_manifest_duplicate_key_and_constant_errors_name_the_manifest_and_not_the_point():
    """Both are refused while the JSON is read, where the phase and index are not known yet."""
    point = '{"batch_size": 1, "total_kv_read_tokens": 8}'
    duplicate = '{"batch_size": 1, "batch_size": 2, "total_kv_read_tokens": 8}'
    with pytest.raises(ValueError) as caught:
        grid.parse_points_manifest(f'{{"schema_version": 1, "prefill": [], "decode": [{point}, {duplicate}]}}')
    assert str(caught.value) == "benchmark points manifest: duplicate key 'batch_size'"
    assert type(caught.value) is ValueError
    for constant in ("NaN", "Infinity", "-Infinity"):
        broken = point.replace("8", constant)
        with pytest.raises(ValueError) as caught:
            grid.parse_points_manifest(f'{{"schema_version": 1, "prefill": [], "decode": [{point}, {broken}]}}')
        assert str(caught.value) == f"benchmark points manifest: the constant {constant} is not allowed"
        assert type(caught.value) is ValueError


HUGE_VALUE = list(range(10_000))  # its repr has 58,890 characters
HUGE_SHOWN = f"{repr(HUGE_VALUE)[:200]}... (58890 characters)"
SMALL_POINT = {"batch_size": 1, "total_prefill_tokens": 1, "total_kv_read_tokens": 0}
# An integer of 300 digits. It is legal JSON and an ``int``, so it passes every type check and is refused by a
# range check, a duplicate check or the rows check. Its negative has 301 characters with the sign.
LONG_INT = 10**299
LONG_SHOWN = "1" + "0" * 199 + "... (300 characters)"
NEGATIVE_LONG_SHOWN = "-1" + "0" * 198 + "... (301 characters)"
LONG_POINT = {"batch_size": LONG_INT, "total_prefill_tokens": LONG_INT, "total_kv_read_tokens": LONG_INT}

OVERSIZED_MANIFESTS = {
    # Numbers that are integers and out of range.
    "prefill batch_size below 1": (
        {"schema_version": 1, "prefill": [{**SMALL_POINT, "batch_size": -LONG_INT}], "decode": []},
        f"prefill[0]: batch_size must be >= 1, got {NEGATIVE_LONG_SHOWN}",
    ),
    "prefill total_prefill_tokens below batch_size": (
        {"schema_version": 1, "prefill": [{**LONG_POINT, "total_prefill_tokens": -LONG_INT}], "decode": []},
        f"prefill[0]: total_prefill_tokens must be >= batch_size, got {NEGATIVE_LONG_SHOWN} for {LONG_SHOWN}",
    ),
    "prefill total_kv_read_tokens below 0": (
        {"schema_version": 1, "prefill": [{**SMALL_POINT, "total_kv_read_tokens": -LONG_INT}], "decode": []},
        f"prefill[0]: total_kv_read_tokens must be >= 0, got {NEGATIVE_LONG_SHOWN}",
    ),
    "decode batch_size below 1": (
        {"schema_version": 1, "prefill": [], "decode": [{"batch_size": -LONG_INT, "total_kv_read_tokens": 8}]},
        f"decode[0]: batch_size must be >= 1, got {NEGATIVE_LONG_SHOWN}",
    ),
    "decode total_kv_read_tokens below batch_size": (
        {"schema_version": 1, "prefill": [], "decode": [{"batch_size": LONG_INT, "total_kv_read_tokens": -LONG_INT}]},
        f"decode[0]: total_kv_read_tokens must be >= batch_size, got {NEGATIVE_LONG_SHOWN} for {LONG_SHOWN}",
    ),
    # A repeated point: the coordinate is three numbers of 300 digits, 906 characters with the punctuation.
    "duplicate point": (
        {"schema_version": 1, "prefill": [LONG_POINT, LONG_POINT], "decode": []},
        f"prefill[1]: duplicate of prefill[0]: ({'1' + '0' * 198}... (906 characters)",
    ),
    # Rows that do not restate the even split: first the number of rows is long, then the split itself
    # ("[", 300 digits, ", 0]": 305 characters).
    "rows for a long batch_size": (
        {
            "schema_version": 3,
            "prefill": [{**LONG_POINT, "total_kv_read_tokens": 0, "rows": [[1, 0]]}],
            "decode": [],
        },
        f"prefill[0]: rows must be {LONG_SHOWN} rows that all equal [total_prefill_tokens // batch_size, "
        "total_kv_read_tokens // batch_size] = [1, 0]; uneven requests are not supported",
    ),
    "rows for a long even split": (
        {
            "schema_version": 3,
            "prefill": [{**SMALL_POINT, "total_prefill_tokens": LONG_INT, "rows": [[1, 0]]}],
            "decode": [],
        },
        "prefill[0]: rows must be 1 rows that all equal [total_prefill_tokens // batch_size, "
        f"total_kv_read_tokens // batch_size] = [{'1' + '0' * 198}... (305 characters); "
        "uneven requests are not supported",
    ),
    "schema_version": (
        {"schema_version": HUGE_VALUE, "prefill": [], "decode": []},
        f"benchmark points manifest schema_version must be 1, 2 or 3, got {HUGE_SHOWN}",
    ),
    "prefill batch_size": (
        {"schema_version": 1, "prefill": [{**SMALL_POINT, "batch_size": HUGE_VALUE}], "decode": []},
        f"prefill[0]: batch_size must be an integer, got {HUGE_SHOWN}",
    ),
    "prefill total_kv_read_tokens": (
        {
            "schema_version": 1,
            "prefill": [SMALL_POINT, {**SMALL_POINT, "total_kv_read_tokens": HUGE_VALUE}],
            "decode": [],
        },
        f"prefill[1]: total_kv_read_tokens must be an integer, got {HUGE_SHOWN}",
    ),
    "prefill point is a list": (
        {"schema_version": 1, "prefill": [HUGE_VALUE], "decode": []},
        f"prefill[0]: must be an object, got {HUGE_SHOWN}",
    ),
    "decode total_kv_read_tokens": (
        {"schema_version": 1, "prefill": [], "decode": [{"batch_size": 1, "total_kv_read_tokens": HUGE_VALUE}]},
        f"decode[0]: total_kv_read_tokens must be an integer, got {HUGE_SHOWN}",
    ),
    "decode total_prefill_tokens": (
        {
            "schema_version": 1,
            "prefill": [],
            "decode": [{"batch_size": 1, "total_kv_read_tokens": 8, "total_prefill_tokens": HUGE_VALUE}],
        },
        f"decode[0]: a decode point's total_prefill_tokens must be 0, got {HUGE_SHOWN}",
    ),
    # Unknown keys: the first eight in the order of their repr, then how many more.
    "1000 unknown keys": (
        {
            "schema_version": 1,
            "prefill": [{**SMALL_POINT, **{f"key{index}": 0 for index in range(1000)}}],
            "decode": [],
        },
        "prefill[0]: unknown key(s) 'key0', 'key1', 'key10', 'key100', 'key101', 'key102', 'key103', 'key104' "
        "and 992 more",
    ),
    "9 unknown keys": (
        {"schema_version": 1, "prefill": [], "decode": [{f"k{index}": 0 for index in range(9)}]},
        "decode[0]: unknown key(s) 'k0', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7' and 1 more",
    ),
    "8 unknown keys": (
        {"schema_version": 1, "prefill": [], "decode": [{f"k{index}": 0 for index in range(8)}]},
        "decode[0]: unknown key(s) 'k0', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7'",
    ),
    "a 5000-character unknown key": (
        {"schema_version": 1, "prefill": [{**SMALL_POINT, "x" * 5000: 0}], "decode": []},
        f"prefill[0]: unknown key(s) '{'x' * 199}... (5002 characters)",
    ),
}


@pytest.mark.parametrize("name", sorted(OVERSIZED_MANIFESTS))
def test_manifest_error_messages_do_not_carry_a_rejected_value_whole(name):
    """The message is what a caller logs or stores; it stays small whatever the size of the rejected value."""
    document, expected = OVERSIZED_MANIFESTS[name]
    with pytest.raises(ValueError) as caught:
        grid.parse_points_manifest(json.dumps(document))
    assert str(caught.value) == expected
    assert len(str(caught.value)) < 1000


def test_other_error_messages_do_not_carry_a_rejected_value_whole():
    def message(error, call) -> str:
        with pytest.raises(error) as caught:
            call()
        assert len(str(caught.value)) < 1000, len(str(caught.value))
        return str(caught.value)

    # A key that appears twice, 5,000 characters long (its repr adds the two quotes).
    long_key = "k" * 5000
    duplicate = f'{{"schema_version": 1, "prefill": [], "decode": [], "{long_key}": 1, "{long_key}": 2}}'
    assert message(ValueError, lambda: grid.parse_points_manifest(duplicate)) == (
        f"benchmark points manifest: duplicate key '{'k' * 199}... (5002 characters)"
    )
    # The parsed form handed to build_grid.
    for manifest, start in (
        ({"schema_version": HUGE_VALUE, "prefill": [], "decode": []}, "manifest schema_version must be 1, 2 or 3, got"),
        (
            {"schema_version": 1, "prefill": [HUGE_VALUE], "decode": []},
            "prefill[0]: must be a coordinate of 3 integers",
        ),
        ({"schema_version": 1, "prefill": [], "decode": [(1, HUGE_VALUE)]}, "decode[0]: total_kv_read_tokens must be"),
    ):
        text = message(ValueError, lambda manifest=manifest: grid.build_grid("agg", MANIFEST_LIMITS, manifest=manifest))
        assert text.startswith(start) and text.endswith(HUGE_SHOWN)
    # A manifest point that passes the schema and is refused at grid build: its totals do not divide by the
    # batch size (10**299 leaves 1 when divided by 3), or it does not fit. The message shows the point as a dict
    # of its totals: 42 characters up to the 300-digit number, the number, and 28 characters after it.
    shown_point = "point={'batch_size': %d, 'total_prefill_tokens': " + "1" + "0" * 157 + "... (370 characters)"
    uneven = {"schema_version": 1, "prefill": [(3, LONG_INT, 0)], "decode": []}
    assert message(ValueError, lambda: grid.build_grid("prefill", MANIFEST_LIMITS, manifest=uneven)) == (
        "prefill[0]: uniform requests only: totals are not divisible by batch_size: " + shown_point % 3
    )
    too_long = {"schema_version": 1, "prefill": [(1, LONG_INT, 0)], "decode": []}
    assert message(ValueError, lambda: grid.build_grid("prefill", MANIFEST_LIMITS, manifest=too_long)) == (
        "prefill[0]: explicit benchmark point is infeasible: reason=input_too_long "
        + shown_point % 1
        + f" limits={MANIFEST_LIMITS.as_dict()}"
    )
    # Limits, the mode and a point: values handed over by a Python caller.
    assert message(TypeError, lambda: grid.Limits(HUGE_VALUE, 200, 1600, 8, 96, 64)).endswith(HUGE_SHOWN)
    text = message(TypeError, lambda: grid.Limits(16, 200, 1600, 8, 96, 64, decode_graph_sizes="8" * 5000))
    assert text == f"decode_graph_sizes must be a list or tuple of ints, got str: '{'8' * 199}... (5002 characters)"
    unknown = {**VALID_LIMITS, **{f"key{index}": 0 for index in range(1000)}}
    assert message(ValueError, lambda: grid.Limits.from_mapping(unknown)).endswith(
        "unknown ['key0', 'key1', 'key10', 'key100', 'key101', 'key102', 'key103', 'key104' and 992 more]"
    )
    assert message(ValueError, lambda: grid.build_grid(HUGE_VALUE, MANIFEST_LIMITS)).endswith(HUGE_SHOWN)
    assert message(ValueError, lambda: grid.expected_cached_tokens(HUGE_VALUE, 16, 16)).endswith(HUGE_SHOWN)
    assert message(TypeError, lambda: grid.Point(1, "decode", 1, 0, 8, HUGE_VALUE)).endswith(HUGE_SHOWN)
    # A value of up to 200 characters is shown whole, a longer one is cut and its length is given.
    assert grid._shown("a" * 198) == repr("a" * 198) and len(repr("a" * 198)) == 200
    assert grid._shown("a" * 199) == "'" + "a" * 199 + "... (201 characters)"
    assert grid._shown(7) == "7" and grid._shown(None) == "None" and grid._shown([1, "x"]) == "[1, 'x']"


# ---------------------------------------------------------------------------------------------------------------
# Points manifest: grid build
# ---------------------------------------------------------------------------------------------------------------

# P = 16, I = 2000, S = 64000, R = 16, M = 1024, C = 512.
MANIFEST_LIMITS = grid.Limits(16, 2000, 64_000, 16, 1024, 512, (1, 2, 4, 8), (16, 32, 64))


def _parsed(prefill=(), decode=(), version=1) -> dict:
    return grid.parse_points_manifest(_manifest(prefill, decode, version))


def test_manifest_grid_keeps_file_order_numbers_across_phases_and_marks_points_explicit():
    manifest = _parsed(
        # Per request: (n = 1, k = 0), (n = 32, k = 32), (n = 64, k = 0), (n = 8, k = 1024); not sorted.
        [_prefill_point(1, 1, 0), _prefill_point(2, 64, 64), _prefill_point(4, 256, 0), _prefill_point(1, 8, 1024)],
        # Per request: k = 4, k = 500, k = 4.
        [_decode_point(1, 4), _decode_point(2, 1000), _decode_point(4, 16)],
    )
    built = grid.build_grid("agg", MANIFEST_LIMITS, manifest=manifest)
    assert built.source == "manifest" and built.missing_phases == ()
    assert [(p.benchmark_id, p.phase, p.batch_size, p.new_tokens, p.cached_tokens) for p in built.points] == [
        (1, "prefill", 1, 1, 0),
        (2, "prefill", 2, 32, 32),
        (3, "prefill", 4, 64, 0),
        (4, "prefill", 1, 8, 1024),
        (5, "decode", 1, 0, 4),
        (6, "decode", 2, 0, 500),
        (7, "decode", 4, 0, 4),
    ]
    assert all(point.explicit for point in built.points)
    assert [record["sample_reasons"] for record in built.point_dicts()] == [
        ["explicit"],
        ["explicit", "prefill_real_seed"],
        ["explicit"],
        ["explicit", "prefill_real_seed"],
        ["explicit", "kvwarm_real_kv"],
        ["explicit", "kvwarm_real_kv"],
        ["explicit", "kvwarm_real_kv"],
    ]
    # The totals in the artifact are the totals of the file.
    assert [(r["batch_size"], r["total_prefill_tokens"], r["total_kv_read_tokens"]) for r in built.point_dicts()] == [
        (1, 1, 0),
        (2, 64, 64),
        (4, 256, 0),
        (1, 8, 1024),
        (1, 0, 4),
        (2, 0, 1000),
        (4, 0, 16),
    ]
    assert built.digest == grid.grid_digest(built.point_dicts())

    # One phase only: its own list, numbered from 1.
    decode_only = grid.build_grid("decode", MANIFEST_LIMITS, manifest=manifest)
    assert [(p.benchmark_id, p.batch_size, p.cached_tokens) for p in decode_only.points] == [
        (1, 1, 4),
        (2, 2, 500),
        (3, 4, 4),
    ]
    prefill_only = grid.build_grid("prefill", MANIFEST_LIMITS, manifest=manifest)
    assert prefill_only.points == built.points[:4]

    # File order is part of the plan: the same points in another order are another grid.
    reordered = _parsed(
        [_prefill_point(2, 64, 64), _prefill_point(1, 1, 0), _prefill_point(4, 256, 0), _prefill_point(1, 8, 1024)],
        [_decode_point(1, 4), _decode_point(2, 1000), _decode_point(4, 16)],
    )
    assert grid.build_grid("agg", MANIFEST_LIMITS, manifest=reordered).digest != built.digest


def test_manifest_grid_ignores_the_caps_and_the_batch_cap():
    manifest = _parsed(
        [_prefill_point(batch, 16 * batch, 0) for batch in (1, 2, 4, 8, 16)],
        [_decode_point(batch, 100 * batch) for batch in (16, 3, 9)]
        + [_decode_point(1, context) for context in range(4, 40)],
    )
    tiny = grid.SampleCaps(2, 2, 2, 2, 1, benchmark_max_batch_size=1)
    built = grid.build_grid("agg", MANIFEST_LIMITS, tiny, manifest)
    assert built == grid.build_grid("agg", MANIFEST_LIMITS, None, manifest)
    assert len(built.points) == 5 + 3 + 36
    assert max(point.batch_size for point in built.points if point.phase == "decode") == 16


def test_manifest_grid_ignores_the_list_of_the_other_phase():
    # The decode point is below the 4-token minimum and the prefill point is not page-aligned: each is fatal
    # only when its phase runs.
    manifest = _parsed([_prefill_point(1, 8, 8)], [_decode_point(2, 6)])
    with pytest.raises(ValueError, match=r"prefill\[0\].*cached_not_page_aligned"):
        grid.build_grid("prefill", MANIFEST_LIMITS, manifest=manifest)
    with pytest.raises(ValueError, match=r"decode\[0\].*context_below_minimum"):
        grid.build_grid("decode", MANIFEST_LIMITS, manifest=manifest)
    good_prefill = _parsed([_prefill_point(1, 8, 16)], [_decode_point(2, 6)])
    built = grid.build_grid("prefill", MANIFEST_LIMITS, manifest=good_prefill)
    assert [(p.batch_size, p.new_tokens, p.cached_tokens) for p in built.points] == [(1, 8, 16)]
    assert built.missing_phases == ()
    good_decode = _parsed([_prefill_point(1, 8, 8)], [_decode_point(2, 8)])
    built = grid.build_grid("decode", MANIFEST_LIMITS, manifest=good_decode)
    assert [(p.batch_size, p.cached_tokens) for p in built.points] == [(2, 4)]


def test_manifest_grid_lists_an_empty_phase_of_the_mode_as_missing():
    only_decode = _parsed([], [_decode_point(1, 4)])
    assert grid.build_grid("agg", MANIFEST_LIMITS, manifest=only_decode).missing_phases == ("prefill",)
    assert grid.build_grid("prefill", MANIFEST_LIMITS, manifest=only_decode).missing_phases == ("prefill",)
    assert grid.build_grid("prefill", MANIFEST_LIMITS, manifest=only_decode).points == ()
    assert grid.build_grid("decode", MANIFEST_LIMITS, manifest=only_decode).missing_phases == ()
    only_prefill = _parsed([_prefill_point(1, 1, 0)], [])
    built = grid.build_grid("agg", MANIFEST_LIMITS, manifest=only_prefill)
    assert built.missing_phases == ("decode",) and len(built.points) == 1 and built.source == "manifest"
    compact = json.dumps(built.point_dicts(), sort_keys=True, separators=(",", ":"))
    assert built.digest == hashlib.sha256(compact.encode()).hexdigest() != hashlib.sha256(b"[]").hexdigest()
    empty = grid.build_grid("agg", MANIFEST_LIMITS, manifest=_parsed())
    assert empty.missing_phases == ("prefill", "decode") and empty.digest == hashlib.sha256(b"[]").hexdigest()


@pytest.mark.parametrize(
    ("prefill", "decode", "match"),
    [
        # 257 new tokens over 4 requests is AISim's own vLLM fixture; here requests must be identical.
        ([_prefill_point(4, 257, 512)], [], r"prefill\[0\]: .*totals are not divisible by batch_size"),
        ([_prefill_point(1, 1, 0), _prefill_point(4, 256, 2)], [], r"prefill\[1\]: .*not divisible by batch_size"),
        ([], [_decode_point(1, 4), _decode_point(4, 513)], r"decode\[1\]: .*not divisible by batch_size"),
        # Divisibility is checked before feasibility: this point is also far too long.
        ([_prefill_point(3, 9000, 16)], [], r"prefill\[0\]: .*not divisible by batch_size"),
    ],
)
def test_manifest_grid_rejects_totals_that_do_not_divide(prefill, decode, match):
    with pytest.raises(ValueError, match=match):
        grid.build_grid("agg", MANIFEST_LIMITS, manifest=_parsed(prefill, decode))


@pytest.mark.parametrize(
    ("prefill", "decode", "label", "reason"),
    [
        ([_prefill_point(1, 8, 8)], [], "prefill[0]", "cached_not_page_aligned"),
        ([_prefill_point(1, 1, 0), _prefill_point(1, 1000, 1008)], [], "prefill[1]", "input_too_long"),
        ([_prefill_point(17, 17, 0)], [], "prefill[0]", "running_limit"),
        ([_prefill_point(4, 4 * 144, 0)], [], "prefill[0]", "chunk_budget"),  # 4 * 144 = 576 > 512
        ([], [_decode_point(2, 6)], "decode[0]", "context_below_minimum"),
        ([], [_decode_point(1, 4), _decode_point(1, 2001)], "decode[1]", "input_too_long"),
        ([], [_decode_point(17, 68)], "decode[0]", "running_limit"),
    ],
)
def test_manifest_grid_rejects_an_infeasible_point_with_its_label_reason_point_and_limits(
    prefill, decode, label, reason
):
    with pytest.raises(ValueError) as caught:
        grid.build_grid("agg", MANIFEST_LIMITS, manifest=_parsed(prefill, decode))
    message = str(caught.value)
    assert message.startswith(f"{label}: explicit benchmark point is infeasible: reason={reason} point=")
    point = (prefill if label.startswith("prefill") else decode)[int(label[-2])]
    assert f"point={point} limits=" in message
    assert f"limits={MANIFEST_LIMITS.as_dict()}" in message


def test_manifest_grid_rejects_points_that_break_the_budget_and_pool_rules():
    small_budget = grid.Limits(16, 2000, 64_000, 16, 48, 512)
    with pytest.raises(ValueError, match=r"prefill\[0\]: .*reason=prefill_token_budget"):
        grid.build_grid("prefill", small_budget, manifest=_parsed([_prefill_point(4, 64, 0)]))  # 3 * 16 = 48
    small_pool = grid.Limits(16, 2000, 800, 16, 1024, 512)
    with pytest.raises(ValueError, match=r"prefill\[0\]: .*reason=kv_pool"):
        grid.build_grid("prefill", small_pool, manifest=_parsed([_prefill_point(4, 64, 704)]))  # 4 * (176 + 33) = 836
    with pytest.raises(ValueError, match=r"decode\[0\]: .*reason=kv_pool"):
        grid.build_grid("decode", small_pool, manifest=_parsed([], [_decode_point(4, 800)]))  # 4 * (208 + 23) = 924
    assert grid.build_grid("decode", small_pool, manifest=_parsed([], [_decode_point(3, 600)])).missing_phases == ()


def test_manifest_grid_accepts_the_parsed_form_after_a_json_round_trip_and_rechecks_it():
    manifest = _parsed([_prefill_point(2, 64, 64)], [_decode_point(2, 1000)])
    round_tripped = json.loads(json.dumps(manifest))  # tuples become lists
    assert grid.build_grid("agg", MANIFEST_LIMITS, manifest=round_tripped) == grid.build_grid(
        "agg", MANIFEST_LIMITS, manifest=manifest
    )
    bad_forms = [
        ("not a mapping", "result of parse_points_manifest"),
        ({"schema_version": 1, "prefill": []}, "result of parse_points_manifest"),
        ({"schema_version": 1, "prefill": [], "decode": [], "extra": 1}, "result of parse_points_manifest"),
        ({"schema_version": 5, "prefill": [], "decode": []}, "schema_version"),
        ({"schema_version": True, "prefill": [], "decode": []}, "schema_version"),
        ({"schema_version": 1, "prefill": None, "decode": []}, "prefill: must be a list"),
        ({"schema_version": 1, "prefill": [(2, 64)], "decode": []}, r"prefill\[0\]: must be a coordinate of 3"),
        ({"schema_version": 1, "prefill": [], "decode": [(2, 0, 1000)]}, r"decode\[0\]: must be a coordinate of 2"),
        ({"schema_version": 1, "prefill": [{"batch_size": 2}], "decode": []}, r"prefill\[0\]: must be a coordinate"),
        ({"schema_version": 1, "prefill": [(2, 64.0, 64)], "decode": []}, r"prefill\[0\]: total_prefill_tokens"),
        ({"schema_version": 1, "prefill": [], "decode": [(True, 8)]}, r"decode\[0\]: batch_size"),
        ({"schema_version": 1, "prefill": [(0, 64, 64)], "decode": []}, r"prefill\[0\]: batch_size must be >= 1"),
        ({"schema_version": 1, "prefill": [], "decode": [(2, 1)]}, r"decode\[0\]: total_kv_read_tokens"),
        ({"schema_version": 1, "prefill": [(2, 64, 64), (2, 64, 64)], "decode": []}, r"prefill\[1\]: duplicate"),
    ]
    for bad, match in bad_forms:
        with pytest.raises(ValueError, match=match):
            grid.build_grid("agg", MANIFEST_LIMITS, manifest=bad)


# ---------------------------------------------------------------------------------------------------------------
# The default-caps grid for the limits of the two engine probe runs (see H200_PAGE_1 and H200_PAGE_64)
# ---------------------------------------------------------------------------------------------------------------

# Graph sizes as (first, last, step) rows. They were not read off an engine: no probe result file holds graph
# sizes. They are what ``generate_decode_cuda_graph_batch_sizes`` (without speculative decoding, graph padding
# on) and ``generate_prefill_cuda_graph_batch_sizes`` return at SGLang v0.5.22 and v0.5.23 for a maximum of 256
# and of 8192 (srt/arg_groups/cuda_graph_hook.py). "H200" in the two names says only that the rows are used
# with the limits records above.
H200_DECODE_GRAPH_ROWS = ((1, 2, 1), (4, 12, 4), (16, 256, 8))
H200_PREFILL_GRAPH_ROWS = (
    (4, 32, 4),
    (48, 256, 16),
    (288, 512, 32),
    (576, 1024, 64),
    (1280, 4096, 256),
    (4608, 8192, 512),
)


def _ladder(rows) -> tuple[int, ...]:
    return tuple(size for first, last, step in rows for size in range(first, last + 1, step))


@pytest.mark.parametrize(
    ("page", "pool", "first_prefill", "first_decode", "points_per_batch_without_two", "prefixes_of_two"),
    [
        # Page 1. First prefill point: n = C = 8192, one request, k = floor16(40954 - 8192 - 1) = 32752.
        # First decode point: b = R = 4096, k = (1165357 // 4096) - 1 - 7 + 3 = 279.
        # The cached prefixes of n = 2: 0, the twelve doublings from 16 to 32768, and
        # floor16(40954 - 2 - 1) = 40944, 14 values under the default cap of 16.
        (1, 1_165_358, (1, 8192, 32752), (4096, 279), {1: 909, 2: 784, 3: 42, 4: 658}, 14),
        # Page 64. k = floor64(32761) = 32704. Decode: floor64(284 - 64 - 7) + 3 = 192 + 3 = 195.
        # The cached prefixes of n = 2: 0, the ten doublings from 64 to 32768, and floor64(40951) = 40896.
        (64, 1_165_312, (1, 8192, 32704), (4096, 195), {1: 779, 2: 672, 3: 36, 4: 564}, 12),
    ],
)
def test_default_grid_for_the_engine_probe_limits(
    page, pool, first_prefill, first_decode, points_per_batch_without_two, prefixes_of_two, monkeypatch
):
    decode_sizes = _ladder(H200_DECODE_GRAPH_ROWS)
    prefill_sizes = _ladder(H200_PREFILL_GRAPH_ROWS)
    assert decode_sizes[:6] == (1, 2, 4, 8, 12, 16) and len(decode_sizes) == 36
    assert prefill_sizes[0] == 4 and prefill_sizes[-1] == 8192 and len(prefill_sizes) == 58
    limits = grid.Limits(
        page_size=page,
        max_req_input_len=40954,
        max_total_num_tokens=pool,
        max_running_requests=4096,
        max_prefill_tokens=16384,
        chunked_prefill_size=8192,
        decode_graph_sizes=decode_sizes,
        prefill_graph_sizes=prefill_sizes,
    )
    caps = grid.SampleCaps()
    built = grid.build_grid("agg", limits)
    points = built.points
    prefill = [point for point in points if point.phase == "prefill"]
    decode = [point for point in points if point.phase == "decode"]

    assert built.missing_phases == () and built.source == "generated"
    assert [point.benchmark_id for point in points] == list(range(1, len(points) + 1))
    assert list(points) == prefill + decode
    assert len({_coordinate(point) for point in points}) == len(points)
    # Every point fits, by the test-side models and not by the module's own reason functions.
    assert all(_one_pass_takes_all(limits, p.batch_size, p.new_tokens, p.cached_tokens) for p in prefill)
    assert all(p.cached_tokens % grid.cached_unit(page) == 0 for p in prefill)
    assert all(_decode_run_holds_all(limits, p.batch_size, p.cached_tokens) for p in decode)
    assert built.digest == grid.build_grid("agg", limits, caps).digest

    # Hand-computed ends of the sweep. The first point of a block is the first of its first group, not the
    # deepest point of the block: the longest prefix and the longest context come later, in smaller groups.
    first = prefill[0]
    assert (first.batch_size, first.new_tokens, first.cached_tokens) == first_prefill
    assert (prefill[-1].batch_size, prefill[-1].new_tokens, prefill[-1].cached_tokens) == (1, 1, 0)
    assert (decode[0].batch_size, decode[0].cached_tokens) == first_decode
    assert (decode[-1].batch_size, decode[-1].cached_tokens) == (1, 4)
    assert max(point.cached_tokens for point in prefill) > first.cached_tokens
    assert max(point.cached_tokens for point in decode) == 40954 > decode[0].cached_tokens

    # New-token axis. The graph axis is both sides of the 58 graph sizes, of which the largest is the limit
    # itself and has no successor: 115 candidates, cut to the default 64. The ladder starts at 4, so a batch of
    # one token runs without a graph and a batch of two with the graph of 4: the low end, 1 and 2, is put in
    # front after the cut: 66 values.
    new_axis = grid.prefill_new_token_axis(limits, caps)
    uncut = grid.prefill_new_token_axis(limits, NO_CAPS)
    graph_values = grid.graph_axis(prefill_sizes, 8192)
    assert len(graph_values) == 115 and uncut == [1, 2, *graph_values]
    assert uncut[:5] == [1, 2, 4, 5, 8] and uncut[-2:] == [7681, 8192]
    assert new_axis[:2] == [1, 2] and new_axis[2:] == grid.limit_graph_axis(graph_values, prefill_sizes, 64)
    assert len(new_axis) == 66 and new_axis[-1] == 8192
    assert sorted({point.new_tokens for point in prefill}) == new_axis

    # Prefill batch axis: the first three of the powers of two and b_max. It is 1, 2, 4 wherever four requests
    # fit; 1, 2, 3 for the three n between C / 4 and C / 3, where b_max = C // ceilP(n) = 3; 1, 2 where two fit;
    # and 1 where one fits.
    assert grid.prefill_batch_axis(limits, caps, 64) == [1, 2, 4]
    assert grid.largest_prefill_batch(limits, 64) == 128  # 128 * 64 = 8192 = C
    batches_of = {}
    for point in prefill:
        batches_of.setdefault(point.new_tokens, set()).add(point.batch_size)
    shapes = {}
    for new, batches in batches_of.items():
        shapes.setdefault(tuple(sorted(batches)), []).append(new)
    # The 48 values with the batch sizes 1, 2 and 4 are n = 2 and the 47 that the axis without it has.
    assert {shape: len(values) for shape, values in shapes.items()} == {(1,): 9, (1, 2): 6, (1, 2, 3): 3, (1, 2, 4): 48}
    assert sorted(shapes[(1, 2, 3)]) == [2049, 2305, 2561]
    assert [grid.largest_prefill_batch(limits, new) for new in (2048, 2049, 2561, 2817)] == [4, 3, 3, 2]
    # The points of n = 2: the batch sizes 1, 2 and 4, each with every cached prefix up to the input limit.
    unit = grid.cached_unit(page)
    longest = (40954 - 2 - 1) // unit * unit
    prefixes = [0, *(unit * 2**step for step in range(12) if unit * 2**step <= longest), longest]
    assert len(prefixes) == prefixes_of_two <= caps.prefill_max_kv_read_token_samples
    assert sorted(shape for shape in _prefill_shapes(prefill) if shape[1] == 2) == [
        (batch, 2, cached) for batch in (1, 2, 4) for cached in prefixes
    ]
    # The prefill batch sizes that occur, and the number of points at each: those of the axis without n = 2,
    # and the points of n = 2 on top at the batch sizes 1, 2 and 4.
    points_per_batch = {}
    for point in prefill:
        points_per_batch[point.batch_size] = points_per_batch.get(point.batch_size, 0) + 1
    assert points_per_batch == {
        batch: count + (prefixes_of_two if batch in (1, 2, 4) else 0)
        for batch, count in points_per_batch_without_two.items()
    }
    # Decode batch axis: 9 small values, both sides of the 31 sizes from 16 to 256, the tail 512, 1024, 2048
    # and the running cap: 75 values, under the default cap of 128.
    batch_axis = grid.decode_batch_axis(limits, caps)
    assert len(batch_axis) == 75 and batch_axis[:9] == [1, 2, 3, 4, 5, 8, 9, 12, 13]
    assert batch_axis[-6:] == [256, 257, 512, 1024, 2048, 4096]
    assert len(prefill) <= 66 * 3 * 16 and len(decode) <= 128 * 128  # what the default caps allow

    # Without the two sides of the first step the grid is the same grid less the group of n = 2: the axis of
    # 65 values and the point counts that the probe limits had before the group was added.
    with monkeypatch.context() as patch:
        patch.setattr(grid, "first_prefill_graph_tokens", lambda sizes: None)
        earlier = grid.build_grid("agg", limits)
        assert len(grid.prefill_new_token_axis(limits, caps)) == 65
    earlier_per_batch = {}
    for batch, _new, _cached in _prefill_shapes(earlier.points):
        earlier_per_batch[batch] = earlier_per_batch.get(batch, 0) + 1
    assert earlier_per_batch == points_per_batch_without_two
    assert _prefill_shapes(earlier.points) == [shape for shape in _prefill_shapes(prefill) if shape[1] != 2]
    assert _decode_shapes(earlier.points) == _decode_shapes(decode)

    # Printed for reference (``pytest -s`` shows it): how many points the default caps plan for these limits.
    print(
        f"sglbench default grid, H200 limits, page size {page}: {len(prefill)} prefill and {len(decode)} decode points"
    )


# ---------------------------------------------------------------------------------------------------------------
# The first step of the prefill table, read in SGLang's source at each tested release
# ---------------------------------------------------------------------------------------------------------------

SOURCES_ENV = "AISIM_SGLBENCH_SGLANG_SOURCES"
# The two releases the header of the module names. They are the ones the launcher accepts
# (``TESTED_VERSIONS`` of sglbench_compat.py), which the test below reads in that file.
SGLANG_RELEASES = ("0.5.22", "0.5.23")


def _sglang_source(version: str, relative: str) -> ast.Module:
    """One file of a local SGLang checkout, parsed and not run. Skips the test when the variable is unset."""
    base = os.environ.get(SOURCES_ENV)
    if not base:
        pytest.skip(f"{SOURCES_ENV} is not set; it names a folder with one SGLang checkout per tested release")
    path = Path(base) / f"v{version}" / "python/sglang" / relative
    # A variable that is set and wrong is a mistake to show, not a reason to skip.
    assert path.is_file(), f"{SOURCES_ENV} holds no checkout of SGLang {version} with {relative}"
    return ast.parse(path.read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    """The one function or method of that name in a parsed file."""
    (found,) = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    return found


def _evaluated(node: ast.expr, names: dict):
    """The value of an expression that is made of integers, the given names, ``+``, ``range`` and ``list``."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _evaluated(node.left, names) + _evaluated(node.right, names)
    if isinstance(node, ast.Call) and not node.keywords and ast.unparse(node.func) in ("range", "list"):
        arguments = [_evaluated(argument, names) for argument in node.args]
        return range(*arguments) if ast.unparse(node.func) == "range" else list(*arguments)
    raise AssertionError(f"SGLang's size list has an expression this test does not read: {ast.unparse(node)}")


@pytest.mark.parametrize("version", SGLANG_RELEASES)
def test_the_first_step_of_the_prefill_table_is_sglangs_rule_at_each_tested_release(version):
    """What ``first_prefill_graph_tokens`` and the docstring of ``prefill_new_token_axis`` say about SGLang.

    The padding factor is 2. ``can_replay_locally`` pads the batch to the smallest captured size that holds it
    and refuses the graph when that size is above the batch's tokens times the factor. SGLang's default sizes
    start at 4, so the first batch with a graph has 2 tokens, and from there to the largest size every batch
    runs with a graph: the default sizes have no second step.
    """
    compat = ast.parse((RUNTIME / "sglbench_compat.py").read_text(encoding="utf-8"))
    (tested,) = [
        node.value
        for node in compat.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "TESTED_VERSIONS"
    ]
    assert ast.literal_eval(tested) == SGLANG_RELEASES

    # The factor.
    runner = _sglang_source(version, "srt/model_executor/runner/prefill_cuda_graph_runner.py")
    (factor,) = [
        node.value
        for node in runner.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR"
    ]
    assert ast.literal_eval(factor) == 2 == grid.PREFILL_GRAPH_PADDING_FACTOR

    # The refusal. These are the last statements of ``can_replay_locally``: a batch that passed every check
    # before them runs with a graph exactly when it is not above the largest size and the size it is padded
    # to is at most its own tokens times the factor.
    replay = _function(runner, "can_replay_locally")
    assert [ast.unparse(statement) for statement in replay.body[-5:]] == [
        "if num_tokens is None:\n    return True",
        "if num_tokens > self.max_num_tokens:\n    return False",
        "padded_num_tokens = self._pad_to_bucket(num_tokens, self.capture_num_tokens)",
        "if padded_num_tokens > num_tokens * _MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR:\n    return False",
        "return True",
    ]
    # The padded size is the smallest captured size that holds the batch. The prefill runner has no padding
    # function of its own: it takes the one of its base class, on the sizes it sorted when it read them.
    (runner_class,) = [
        node for node in runner.body if isinstance(node, ast.ClassDef) and node.name == "PrefillCudaGraphRunner"
    ]
    assert [ast.unparse(base) for base in runner_class.bases] == ["BaseCudaGraphRunner"]
    assert not any(isinstance(node, ast.FunctionDef) and node.name == "_pad_to_bucket" for node in ast.walk(runner))
    base_runner = _sglang_source(version, "srt/model_executor/runner/base_cuda_graph_runner.py")
    pad = _function(base_runner, "_pad_to_bucket")
    assert [argument.arg for argument in pad.args.args] == ["raw_size", "buckets"]
    assert [ast.unparse(statement) for statement in pad.body[-2:]] == [
        "index = bisect.bisect_left(buckets, raw_size)",
        "return buckets[index]",
    ]
    runner_source = ast.unparse(runner)
    assert "self.capture_num_tokens = get_prefill_num_tokens_to_capture(capture_tokens)" in runner_source
    assert "self.max_num_tokens = max(self.capture_num_tokens)" in runner_source
    captured = _function(runner, "get_prefill_num_tokens_to_capture")
    assert ast.unparse(captured.body[-1]).startswith("return sorted(")

    # The default sizes. The function is a sum of ranges, cut at its argument; it is evaluated here, not run.
    hook = _sglang_source(version, "srt/arg_groups/cuda_graph_hook.py")
    generator = _function(hook, "generate_prefill_cuda_graph_batch_sizes")
    assert [argument.arg for argument in generator.args.args] == ["max_bs"]
    assert ast.get_docstring(generator) and len(generator.body) == 4
    ladder, keep, result = generator.body[1:]
    assert isinstance(ladder, ast.Assign) and ast.unparse(ladder.targets[0]) == "capture_sizes"
    assert ast.unparse(keep) == "capture_sizes = [s for s in capture_sizes if s <= max_bs]"
    assert ast.unparse(result) == "return capture_sizes"

    def default_sizes(largest: int) -> list[int]:
        return [size for size in _evaluated(ladder.value, {"max_bs": largest}) if size <= largest]

    # The ladders this file types in by hand are the default sizes for a largest size of 8,192.
    assert tuple(default_sizes(8192)) == MEASURED_PREFILL_GRAPH_SIZES == _ladder(H200_PREFILL_GRAPH_ROWS)
    assert default_sizes(3) == []
    # The largest sizes tried: 8,192 is that of the cells in this file, and the others lie on and between
    # the rows of the ladder, from the first size up to 16,384.
    for largest in (4, 7, 32, 100, 2048, 3000, 4096, 8192, 16384):
        sizes = default_sizes(largest)
        assert sizes == sorted(set(sizes)) and sizes[0] == 4 and sizes[-1] <= largest
        # The first step: one token runs without a graph, two tokens run with the graph of 4.
        assert grid.first_prefill_graph_tokens(sizes) == 2
        assert not _runs_with_a_graph(1, sizes) and _runs_with_a_graph(2, sizes)
        assert _expected_low_end(sizes, largest) == {1, 2}
        # The claim of the docstring of ``prefill_new_token_axis``: no default size is more than the factor
        # times the size before it ...
        assert all(high <= grid.PREFILL_GRAPH_PADDING_FACTOR * low for low, high in itertools.pairwise(sizes))
        # ... and, batch by batch, what the claim is for: from two tokens up to the largest size no batch
        # runs without a graph, so the first step is the only one.
        assert all(_runs_with_a_graph(tokens, sizes) for tokens in range(2, sizes[-1] + 1))
