# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-table regression evidence for the Pro calibration added by PR #718."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import aisimulate_core
from aisimulate_core.sdk.engine import EngineHandle
from aisimulate_core.sdk.engine_table_view import fetch_table_view
from aisimulate_core.sdk.perf_database import PerfDatabase

pytestmark = pytest.mark.unit

SYSTEMS = Path(aisimulate_core.__file__).parent / "systems"
REL = Path("sparse_attention/sglang/0.5.21/dsv4_csa_topk_calib_perf.parquet")
KEYS = ("num_heads", "step", "isl", "batch_size", "score_mode")
EVIDENCE = Path(__file__).resolve().parents[6] / "benchmarks/evidence/dsv4-pro-topk-calibration/summary.json"


@pytest.mark.parametrize("system,total", [("b200_sxm", 6708), ("b300_sxm", 6708), ("h200_sxm", 6028)])
def test_shipped_pro_calibration_is_unique_paired_and_sdk_reachable(system, total):
    """Every new physical row survives the real Rust-backed SDK view."""
    path = SYSTEMS / "data" / system / REL
    record = json.loads(EVIDENCE.read_text())["systems"][system]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    rows = pq.read_table(path).to_pylist()
    assert len(rows) == total == record["shape"][0]
    assert len({tuple(row[k] for k in KEYS) for row in rows}) == total
    added = [r for r in rows if r["num_heads"] == 128]
    assert len(added) == 1526 == record["sdk_added_rows_matched"]
    assert Counter(r["score_mode"] for r in added) == {
        "v1_flat": 563,
        "v1_top_last": 563,
        "v2_flat": 200,
        "v2_top_last": 200,
    }
    pairs = defaultdict(set)
    db = PerfDatabase(system, "sglang", "0.5.21", systems_root=str(SYSTEMS), strict_provenance=True)
    view = fetch_table_view(db, "_dsv4_csa_topk_calib_data")
    for row in added:
        variant, mode = row["score_mode"].split("_", 1)
        pairs[(row["step"], row["isl"], row["batch_size"], variant)].add(mode)
        leaf = view
        for column in KEYS:
            leaf = leaf[row[column]]
        assert leaf["latency"] == row["latency"]
    assert all(modes == {"flat", "top_last"} for modes in pairs.values())


@pytest.mark.parametrize("sequence_tokens,expected_ms", [(8192, 0.121351603), (8193, 0.121352)])
def test_b200_pro_decode_uses_0521_primary_calibration(sequence_tokens, expected_ms):
    """Expected output derives from measured module and selector rows, not a refreshed golden.

    At s=8193 the module anchor is 0.1234 ms and the new v2 correction
    is 0.005172 - 0.003124 = 0.002048 ms, hence 0.121352 ms. At s=8192
    Rust interpolates the module and correction at past_kv=8191, yielding
    the independently archived review value 0.121351603 ms (formerly
    0.061421604 ms with the 0.5.14 donor). See the evidence README.
    """
    db = PerfDatabase("b200_sxm", "sglang", "0.5.21", systems_root=str(SYSTEMS), strict_provenance=True)
    calib = fetch_table_view(db, "_dsv4_csa_topk_calib_data")[128][8192][1][1]
    assert calib["v2_flat"]["latency"] == 0.005172
    assert calib["v2_top_last"]["latency"] == 0.003124
    module = pq.read_table(SYSTEMS / "data/b200_sxm" / REL.parent / "dsv4_csa_generation_module_perf.parquet")
    anchors = [
        r
        for r in module.to_pylist()
        if r["num_heads"] == 128
        and r["batch_size"] == 1
        and r["isl"] == 1
        and r["step"] == 8192
        and r["kv_cache_dtype"] == "fp8_e4m3"
        and r["gemm_type"] == "fp8_block"
    ]
    assert len(anchors) == 1
    assert anchors[0]["latency"] == 0.1234
    op = {
        "Dsv4Generation": {
            "name": "pro-csa",
            "scale_factor": 1.0,
            "attn_kind": "Csa",
            "num_heads": 128,
            "native_heads": 128,
            "tp_size": 1,
            "kv_cache_dtype": "fp8",
            "fmha_quant_mode": "fp8",
            "gemm_quant_mode": "fp8_block",
            "architecture": "DeepseekV4ForCausalLM",
        }
    }
    handle = EngineHandle.for_database(db, systems_path=str(SYSTEMS))
    result = handle.evaluate_ops_json(json.dumps([op]), is_context=False, batch_size=1, s=sequence_tokens)
    assert result[0][1] == pytest.approx(expected_ms, rel=0, abs=1e-9)
