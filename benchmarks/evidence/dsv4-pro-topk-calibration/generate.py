# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce PR #718's calibration audit and Rust-backed consumer comparison.

Run from the repository root with an installed release wheel. Temporary baseline
data uses TMPDIR; this script neither collects GPU timings nor changes perf data.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import subprocess
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from aisimulate_core.sdk.engine import EngineHandle
from aisimulate_core.sdk.engine_table_view import fetch_table_view
from aisimulate_core.sdk.perf_database import PerfDatabase

BASE = "9ff96cf9ee5c913160f00915d6bf1e89f6a19f6c"
HEAD = "abc4e92124ff63048ccfbc92e24d6ed6fd0bde63"
SYSTEMS = Path("python/aisimulate/src/aisimulate_core/systems")
REL = Path("sparse_attention/sglang/0.5.21/dsv4_csa_topk_calib_perf.parquet")
KEYS = ["num_heads", "step", "isl", "batch_size", "score_mode"]


def git_bytes(ref: str, path: Path) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"])


def database(root: Path, system: str) -> PerfDatabase:
    return PerfDatabase(system, "sglang", "0.5.21", systems_root=str(root.resolve()), strict_provenance=True)


def generation_op(heads: int, kind: str) -> dict:
    return {
        "Dsv4Generation": {
            "name": f"{kind}-{heads}",
            "scale_factor": 1.0,
            "attn_kind": kind,
            "num_heads": heads,
            "native_heads": heads,
            "tp_size": 1,
            "kv_cache_dtype": "fp8",
            "fmha_quant_mode": "fp8",
            "gemm_quant_mode": "fp8_block",
            "architecture": "DeepseekV4ForCausalLM",
            # Flash's dimensions differ from the Pro serde defaults.
            "hidden_size": 7168 if heads == 128 else 4096,
            "q_lora_rank": 1536 if heads == 128 else 1024,
            "index_topk": 1024 if heads == 128 else 512,
            "o_groups": 16 if heads == 128 else 8,
        }
    }


def compare_consumers(root: Path) -> dict:
    db = database(root, "b200_sxm")
    view = fetch_table_view(db, "_dsv4_csa_topk_calib_data")
    handle = EngineHandle.for_database(db, systems_path=str(root.resolve()))
    queries = [generation_op(128, "Csa"), generation_op(128, "Hca"), generation_op(64, "Csa")]
    outputs = {}
    for sequence_tokens in (4097, 8192, 8193, 16385):
        values = handle.evaluate_ops_json(json.dumps(queries), is_context=False, batch_size=1, s=sequence_tokens)
        outputs[str(sequence_tokens)] = {v[0]: v[1] for v in values}
    return {
        "latency_ms_by_sequence_tokens": outputs,
        "pro_v2_at_step_8192_ms": {mode: view[128][8192][1][1][mode]["latency"] for mode in ("v2_flat", "v2_top_last")},
    }


def audit(system: str) -> dict:
    path = SYSTEMS / "data" / system / REL
    raw = path.read_bytes()
    assert raw == git_bytes(HEAD, path), "evidence head's table differs from the checkout"
    table = pq.read_table(io.BytesIO(raw))
    before = pq.read_table(io.BytesIO(git_bytes(BASE, path)))
    rows, old = table.to_pylist(), before.to_pylist()
    added = rows[len(old) :]
    key = lambda row: tuple(row[k] for k in KEYS)
    old_keys, new_keys = {key(r) for r in old}, {key(r) for r in added}
    donor_path = path.parent.parent / "0.5.14" / path.name
    donor_rows = pq.read_table(donor_path).to_pylist()
    pairs = defaultdict(dict)
    for r in added:
        variant, mode = r["score_mode"].split("_", 1)
        pairs[(*key(r)[:-1], variant)][mode] = r["latency"]
    view = fetch_table_view(database(SYSTEMS, system), "_dsv4_csa_topk_calib_data")
    matched = 0
    for r in added:
        leaf = view
        for part in key(r):
            leaf = leaf[part]
        assert leaf["latency"] == r["latency"]
        matched += 1
    meta = yaml.safe_load(path.with_name("collection_meta.yaml").read_text())
    assert rows[: len(old)] == old
    assert len(added) == 1526 and {r["num_heads"] for r in added} == {128}
    assert len({key(r) for r in rows}) == len(rows)
    assert not old_keys & new_keys
    assert meta["tables"]["dsv4_csa_topk_calib_perf"]["rows"] == len(rows)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "columns": {f.name: str(f.type) for f in table.schema},
        "shape": [table.num_rows, table.num_columns],
        "base_rows": len(old),
        "added_rows": len(added),
        "removed_or_modified_rows": 0,
        "original_rows_preserved_in_order": True,
        "unique_keys": len({key(r) for r in rows}),
        "duplicate_keys": len(rows) - len({key(r) for r in rows}),
        "added_base_key_overlap": len(old_keys & new_keys),
        "donor_0514": {
            "path": str(donor_path),
            "sha256": hashlib.sha256(donor_path.read_bytes()).hexdigest(),
            "added_key_overlap": len(new_keys & {key(r) for r in donor_rows}),
        },
        "added_mode_counts": dict(sorted(Counter(r["score_mode"] for r in added).items())),
        "added_axis_values": {k: sorted({r[k] for r in added}) for k in KEYS[:-1]},
        "added_shape_count": len({key(r)[:-1] for r in added}),
        "added_paired_shapes_by_variant": dict(sorted(Counter(k[-1] for k in pairs).items())),
        "anomalies": {
            "null_key_rows": sum(any(r[k] is None for k in KEYS) for r in rows),
            "nonfinite_or_negative_latency_rows": sum(
                not math.isfinite(r["latency"]) or r["latency"] < 0 for r in rows
            ),
            "added_incomplete_pairs": sum(set(p) != {"flat", "top_last"} for p in pairs.values()),
            "added_flat_below_top_last_pairs": sum(p["flat"] < p["top_last"] for p in pairs.values()),
            "added_zero_latency_rows": sum(r["latency"] == 0 for r in added),
        },
        "sdk_added_rows_matched": matched,
        "collection_event": meta["tables"]["dsv4_csa_topk_calib_perf"]["collections"][-1],
    }


def main() -> None:
    modules = {}
    for kind in ("csa", "hca"):
        path = SYSTEMS / "data/b200_sxm" / REL.parent / f"dsv4_{kind}_generation_module_perf.parquet"
        raw = path.read_bytes()
        assert raw == git_bytes(BASE, path) == git_bytes(HEAD, path)
        modules[str(path)] = hashlib.sha256(raw).hexdigest()
    report = {
        "base_commit": BASE,
        "data_head_commit": HEAD,
        "key_columns": KEYS,
        "units": {
            "latency": "ms",
            "step": "past KV tokens per request",
            "isl": "query tokens per request",
            "batch_size": "requests",
            "num_heads": "native model attention heads",
            "tp_size": "tensor parallel ranks (calibration is TP-independent)",
            "compress_ratio": "input tokens per compressed KV token",
        },
        "coverage_note": "Ragged measured points, not the Cartesian product of axis values; v1=context, v2=decode.",
        "systems": {s: audit(s) for s in ("b200_sxm", "b300_sxm", "h200_sxm")},
        "comparison": {"head": compare_consumers(SYSTEMS)},
        "unchanged_b200_generation_module_sha256": modules,
    }
    # Extract the actual baseline, including its module and donor tables. Both
    # sides use the same native binary; this PR changes no consumer source.
    with tempfile.TemporaryDirectory(prefix="pr718-baseline-") as tmp:
        archive = subprocess.Popen(["git", "archive", BASE, str(SYSTEMS)], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=archive.stdout, mode="r|") as tar:
            tar.extractall(tmp, filter="data")
        assert archive.wait() == 0
        report["comparison"]["base"] = compare_consumers(Path(tmp) / SYSTEMS)
    output = Path(__file__).with_name("summary.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}; 4,578 added rows matched through the SDK's Rust table view")


if __name__ == "__main__":
    main()
