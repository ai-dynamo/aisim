# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisim_core import _native
from aisim_core.datasets import (
    Artifact,
    Identity,
    check_dataset,
    configuration_id,
)
from aisim_core.sdk import engine as engine_module
from aisim_core.sdk import utils

pytestmark = pytest.mark.unit


def write(root, path, payload):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload))


def artifact(root, path, role, **kwargs):
    data = (root / path).read_bytes()
    return dict(
        artifact_id=path,
        path=path,
        role=role,
        format="json",
        schema="test/v1",
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        **kwargs,
    )


@pytest.fixture
def dataset(tmp_path, monkeypatch, request):
    def no_download(*args, **kwargs):
        pytest.fail("local profile preparation must not download model metadata")

    monkeypatch.setattr(utils, "_download_hf_json", no_download)
    root = tmp_path / "dataset"
    systems = tmp_path / "producer-systems"
    systems.mkdir()
    hardware = systems / "h200_sxm.yaml"
    hardware.write_bytes((Path(engine_module.__file__).parents[1] / "systems/h200_sxm.yaml").read_bytes())
    runtime_path = "data/h200_sxm/gemm/vllm/1.0/table.parquet"
    (systems / runtime_path).parent.mkdir(parents=True)
    # A real Parquet artifact also retains the runtime version directory. The
    # replacement measured graph consumes no baseline GEMM/attention tables.
    pq.write_table(pa.table({"latency": [2.0, 5.0]}), systems / runtime_path)
    model = "meta-llama/Meta-Llama-3.1-8B"
    if getattr(request, "param", "bundled") == "local":
        local = tmp_path / "local-model"
        local.mkdir()
        config = Path(engine_module.__file__).parents[1] / "model_configs" / f"{model.replace('/', '--')}_config.json"
        shutil.copyfile(config, local / "config.json")
        model = str(local)
    basis = json.loads(
        _native.engine_spec_json_from_bincode(
            engine_module.compile_engine(
                model,
                "h200_sxm",
                "vllm",
                "1.0",
                systems_path=str(systems),
                database_mode="SILICON",
                shared_layer=False,
                forward_model="op_level",
            )
        )
    )
    table_path = "data/op-latency/h200_sxm/gemm/vllm/1.0/table.parquet"
    (root / table_path).parent.mkdir(parents=True)
    shutil.copyfile(systems / runtime_path, root / table_path)
    table = artifact(root, table_path, "operation-measurements", rows=2, runtime_path=runtime_path)
    table["format"] = "parquet"
    (root / "metadata").mkdir()
    shutil.copyfile(hardware, root / "metadata/h200_sxm.yaml")
    hardware_ref = artifact(root, "metadata/h200_sxm.yaml", "hardware", runtime_path=hardware.name)
    profile = {
        "schema_version": 1,
        "basis": basis,
        "worker_type": "prefill",
        "ops": [
            {
                "MeasuredComposite": {
                    "name": "context_measured_scope",
                    "system": "h200_sxm",
                    "backend": "vllm",
                    "version": "1.0",
                    "is_context": True,
                    "weight_bytes": sum(_native.weights_ops_json(json.dumps(basis["context_ops"]))),
                    "scale_factor": 3.0,
                    "cells": [{"shape": [1, 4, 0], "latency_ms": 2.0}, {"shape": [1, 4, 4], "latency_ms": 5.0}],
                }
            }
        ],
        "retained_files": [
            {"path": a["runtime_path"], "sha256": a["sha256"], "size_bytes": a["bytes"]} for a in [table, hardware_ref]
        ],
    }
    write(root, "metadata/profile.json", profile)
    profile_ref = artifact(root, "metadata/profile.json", "execution-profile")
    config = {
        "model": model,
        "system": "h200_sxm",
        "backend": "vllm",
        "backend_version": "1.0",
        "worker_type": "prefill",
        "estimation_mode": "op_level",
        "fallback_policy": "deny",
        "database_mode": "SILICON",
        "enable_shared_layer": False,
        "systems_paths": [str(systems)],
        "estimator_config": {
            "correction": {"enabled": False},
            "op_level": {"execution_profile": {"path": "profile.json", "sha256": profile_ref["sha256"]}},
        },
    }
    write(root, "metadata/config.json", config)
    identity = Identity(
        hardware="h200_sxm",
        framework="vllm",
        framework_version="1.0",
        operation_family=None,
        model="meta-llama/Meta-Llama-3.1-8B",
        effective_settings={"tp": 1},
        unresolved_identity=None,
    )
    manifest = dict(
        schema_version="aisim-artifact-manifest/v1",
        category="op-latency",
        purpose="execution-profile",
        configuration_id=configuration_id("op-latency", "execution-profile", identity),
        identity=identity.model_dump(),
        requested_settings=None,
        artifacts=[table, profile_ref, artifact(root, "metadata/config.json", "predictor-config"), hardware_ref],
        provenance={"producer_revision": None},
        evidence_gaps=[],
    )
    path = "metadata/manifest.json"
    write(root, path, manifest)
    return root, path, manifest


@pytest.mark.parametrize(
    "legacy,valid",
    [
        ({"catalog_version": 5}, True),
        ({"catalog_version": 6}, True),
        ({"catalog_version": 7}, True),
        ([], False),
        ({}, False),
        ({"catalog_version": []}, False),
        ({"catalog_version": {}}, False),
        ({"catalog_version": 5.0}, False),
        ({"catalog_version": "5"}, False),
        ({"catalog_version": True}, False),
    ],
)
def test_legacy_fpm_discovery_requires_supported_integer_version(tmp_path, legacy, valid):
    path = "catalog/fpm/index.json"
    write(tmp_path, path, legacy)
    write(
        tmp_path,
        "catalog/dataset.json",
        {
            "schema_version": "aisim-dataset-index/v1",
            "dataset_id": "public/example",
            "categories": {
                "fpm": {"root": "data/fpm", "index": artifact(tmp_path, path, "category-index"), "layout": "legacy-fpm"}
            },
        },
    )
    if valid:
        assert check_dataset(tmp_path).categories["fpm"].layout == "legacy-fpm"
    else:
        with pytest.raises(ValueError, match="unsupported legacy FPM index"):
            check_dataset(tmp_path)


def test_paths_are_canonical():
    for value in ("/absolute", "a//b", "a/./b", "../b", "a\\b"):
        with pytest.raises(ValueError, match="repository-relative"):
            Artifact(
                artifact_id="a", path=value, role="test", format="json", schema="test/v1", sha256="a" * 64, bytes=0
            )
