# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Key native K3.1 rows by their coordinates in the DSA query loader."""


def validate_union(parts, table):
    if table not in {"dsa_context_module_perf", "dsa_generation_module_perf"}:
        raise ValueError(f"unexpected Kimi DSA table: {table}")
    seen = set()
    for part in parts:
        for row in part.to_pylist():
            if row["architecture"] != "KimiK31ForConditionalGeneration" or row["kv_cache_dtype"] != "fp8":
                raise ValueError("expected native Kimi K3.1 FP8 DSA rows")
            source = row["kernel_source"]
            markers = [marker for marker in ("_kv_flashmla_packed", "_kv_raw_fp8") if marker in source]
            if len(markers) != 1 or not source.endswith(markers[0] + "_eager"):
                raise ValueError(f"ambiguous native cache layout: {source}")
            step, isl = row["step"], row["isl"]
            if table == "dsa_generation_module_perf":
                step, isl = 0, isl + step
            key = (
                row["architecture"],
                row["mla_dtype"],
                row["kv_cache_dtype"],
                row["gemm_type"],
                "skip_indexer" in row["op_name"],
                markers[0],
                row["num_heads"],
                step,
                isl,
                row["batch_size"],
            )
            if key in seen:
                raise ValueError(f"{table} contains overlapping DSA consumer key: {key}")
            seen.add(key)
