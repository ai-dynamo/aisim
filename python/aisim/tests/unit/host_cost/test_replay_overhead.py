# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay reads the measured CPU overhead of a deployment from a pinned ``host-cost`` dataset."""

import json
import random
import re
import shutil

import pytest

from aisim.compiler import prediction_to_replay_spec
from aisim.config.cli import CorePredictionConfig
from aisim_core.data import DatasetIndex, Manifest
from aisim_core.data.ensure import DataConflictError, DataPin, MissingDataError, ensure_data
from aisim_core.data.host_cost import (
    check_entry,
    read_fit,
    read_unqueued_fit,
    register_entries,
    write_entry,
    write_unqueued_entry,
)
from aisim_core.host_cost import replay

pytestmark = pytest.mark.unit

CPU = "Intel(R) Xeon(R) Platinum 8559C"
QWEN = "Qwen/Qwen3.5-397B-A17B"
GLM = "nvidia/GLM-5.3-NVFP4"


def _tokenizer_run(tmp_path, name, *, framework, version, tokenizer, per_token_us, base_ms=0.0):
    rng = random.Random(name)
    rows = []
    for index in range(60):
        tokens = rng.randrange(2_000, 180_000)
        total = base_ms + per_token_us * tokens / 1000.0 * (1 + rng.uniform(-0.005, 0.005))
        rows.append(
            {
                "request": index,
                "tokens": tokens,
                "body_kb": 1.0,
                "json_ms": 1.0,
                "validate_ms": 0.5,
                "template_ms": 0.0,
                "encode_ms": total,
            }
        )
    run = tmp_path / name
    run.mkdir()
    (run / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    environment = {
        "framework": framework,
        "framework_version": version,
        "tokenizer_key": tokenizer,
        "model": f"/tokenizers/{tokenizer}",
        "tokenizer_class": "Tokenizer",
        "tokenizer_is_fast": True,
        "tokenizer_workers": 1,
        "host": {"cpu_model": CPU, "cpu_threads": 192},
    }
    (run / "environment.json").write_text(json.dumps(environment))
    return run


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Two measured B300 deployments, as published: vLLM 0.31.0 Qwen2 and SGLang GLM (tokenizer 0.5.21, runs 0.5.20)."""
    tmp_path = tmp_path_factory.mktemp("host-cost")
    root = tmp_path / "dataset"
    (root / "catalog").mkdir(parents=True)
    index = DatasetIndex(schema_version="aisim-dataset-index/v1", dataset_id="test/aisim-data", categories={})
    (root / "catalog/dataset.json").write_text(index.model_dump_json(by_alias=True))
    tokenizers = {}
    for framework, version, tokenizer, model, per_token, base in (
        ("vllm", "0.31.0", "qwen2", QWEN, 1.5, 0.0),
        ("sglang", "0.5.21", "glm", "zai-org/GLM-5", 1.2, 0.3),
    ):
        run = _tokenizer_run(
            tmp_path,
            framework,
            framework=framework,
            version=version,
            tokenizer=tokenizer,
            per_token_us=per_token,
            base_ms=base,
        )
        tokenizers[framework] = write_entry(root, run, system="b300_sxm", model=model, gpu_time=None)
    register_entries(root, list(tokenizers.values()))

    def manifest(framework):
        return Manifest.model_validate_json((root / tokenizers[framework]).read_bytes())

    unqueued = [
        write_unqueued_entry(
            root,
            framework="vllm",
            version="0.31.0",
            system="b300_sxm",
            endpoint="chat",
            model=QWEN,
            runs=[{"run": "l1-l4", "base_ms": 29.5, "total_per_input_token_us": 1.7}],
            tokenizer_manifest=manifest("vllm"),
            method={"measure": "test"},
        ),
        write_unqueued_entry(
            root,
            framework="sglang",
            version="0.5.20",
            system="b300_sxm",
            endpoint="chat",
            model=GLM,
            runs=[
                {"run": "seed42", "requests": 163, "base_ms": 19.5, "total_per_input_token_us": 1.53, "r2": 0.81},
                {"run": "seed7", "requests": 166, "base_ms": 5.7, "total_per_input_token_us": 1.59, "r2": 0.92},
            ],
            tokenizer_manifest=manifest("sglang"),
            method={"measure": "test"},
        ),
    ]
    register_entries(root, unqueued)
    return root


@pytest.fixture
def local(dataset, monkeypatch):
    monkeypatch.setenv(replay.LOCAL_ENV, str(dataset))
    replay._cached.cache_clear()
    yield dataset
    replay._cached.cache_clear()


def _measured(**kwargs):
    defaults = {
        "backend": "vllm",
        "backend_version": "0.31.0",
        "endpoint": "chat",
        "system": "b300_sxm",
        "host_cpu": CPU,
        "model": QWEN,
    }
    return replay.measured_overhead(**{**defaults, **kwargs})


def _tokenizer_us(root, framework):
    for manifest_path in (root / "data/host-cost" / framework).rglob("aisim-manifest.json"):
        manifest = Manifest.model_validate_json(manifest_path.read_bytes())
        if manifest.purpose == "host-cost-profile":
            return read_fit(manifest).per_input_token_us
    raise AssertionError(framework)


def test_unqueued_entry_round_trips_and_stores_the_run_mean(dataset):
    path = "data/host-cost/sglang/0.5.20/b300_sxm/glm/unqueued-chat/aisim-manifest.json"
    fit = read_unqueued_fit(check_entry(dataset, path))
    assert (fit.runs, fit.base_ms, fit.total_per_input_token_us) == (2, pytest.approx(12.6), pytest.approx(1.56))
    assert fit.per_input_token_us == pytest.approx(1.56 - _tokenizer_us(dataset, "sglang"))


def test_unqueued_entry_rejects_a_negative_term(dataset, tmp_path):
    tokenizer = Manifest.model_validate_json(
        (dataset / "data/host-cost/vllm/0.31.0/b300_sxm/qwen2/aisim-manifest.json").read_bytes()
    )
    with pytest.raises(ValueError, match="non-negative"):
        write_unqueued_entry(
            tmp_path,
            framework="vllm",
            version="0.31.0",
            system="b300_sxm",
            endpoint="chat",
            model=QWEN,
            runs=[{"run": "r", "base_ms": 1.0, "total_per_input_token_us": 0.5}],
            tokenizer_manifest=tokenizer,
            method={},
        )


def test_both_measured_deployments_read_tokenization_and_unqueued_overhead(local):
    vllm = _measured()
    assert vllm.tokenization["workers"] == 1
    assert vllm.tokenization["per_input_token_us"] == pytest.approx(_tokenizer_us(local, "vllm"))
    assert vllm.tokenization["base_ms"] + vllm.extra_cpu_overhead["base_ms"] == pytest.approx(29.5)
    # The two stages reproduce the measured total outside the engine.
    total = vllm.tokenization["per_input_token_us"] + vllm.extra_cpu_overhead["per_input_token_us"]
    assert total == pytest.approx(1.7)

    sglang = _measured(backend="sglang", backend_version="0.5.20", model=GLM)
    assert sglang.tokenization["per_input_token_us"] == pytest.approx(_tokenizer_us(local, "sglang"))
    # Neither term is counted twice: the stages split the measured fixed cost too.
    assert sglang.tokenization["base_ms"] + sglang.extra_cpu_overhead["base_ms"] == pytest.approx(12.6)
    assert sglang.tokenization["per_input_token_us"] + sglang.extra_cpu_overhead["per_input_token_us"] == (
        pytest.approx(1.56)
    )
    # The tokenizer was collected at 0.5.21 only: 0.5.20 reuses it and says so; the runs were served on 0.5.20.
    assert sglang.provenance["tokenization"]["reused_version"] == "0.5.21"
    assert sglang.provenance["extra_cpu_overhead"]["reused_version"] is None
    newer = _measured(backend="sglang", backend_version="0.5.21", model=GLM)
    assert newer.provenance["extra_cpu_overhead"]["reused_version"] == "0.5.20"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"system": "h200_sxm"},  # no measured host
        {"model": "example/model"},  # no measured tokenizer
        {"endpoint": "completions"},  # no measured endpoint
        {"backend": "trtllm", "backend_version": "1.3.0rc29"},  # no measured framework
        {"host_cpu": "AMD EPYC 9654 96-Core Processor"},  # a B300 host with another CPU
    ],
)
def test_unmeasured_deployments_have_no_default(local, kwargs):
    assert _measured(**kwargs) is None


CONFIG = {
    "system": "b300_sxm",
    "backend": "vllm",
    "backend_version": "0.31.0",
    "model": QWEN,
    "endpoint": "chat",
    "host_cpu": CPU,
}


def test_ensure_data_wires_host_cost_and_names_the_entries(local):
    config = dict(CONFIG)
    ensured = ensure_data(config, DataPin(local_root=local, categories=("host-cost",)))
    host_cost = ensured.config["host_cost"]
    assert host_cost["tokenization"]["base_ms"] + host_cost["extra_cpu_overhead"]["base_ms"] == pytest.approx(29.5)
    assert len(ensured.datasets["host-cost"]["configuration_ids"]) == 2

    unmeasured = ensure_data({**config, "system": "h200_sxm"}, DataPin(local_root=local, categories=("host-cost",)))
    assert unmeasured.config["host_cost"] is None
    assert unmeasured.datasets["host-cost"]["configuration_ids"] == []

    for name in ("endpoint", "host_cpu"):
        with pytest.raises(MissingDataError, match=name):
            ensure_data(
                {k: v for k, v in config.items() if k != name}, DataPin(local_root=local, categories=("host-cost",))
            )


def _copy(dataset, tmp_path):
    root = tmp_path / "copy"
    shutil.copytree(dataset, root)
    return root


def _rewrite_unqueued(root, framework, version, runs):
    """Replace one unqueued entry in place (same configuration ID), as a partly refreshed dataset would."""
    directory = next((root / "data/host-cost" / framework / version).rglob("unqueued-chat"))
    old = Manifest.model_validate_json((directory / "aisim-manifest.json").read_bytes())
    tokenizer_path = next(
        path
        for path in (root / "data/host-cost" / framework).rglob("aisim-manifest.json")
        if Manifest.model_validate_json(path.read_bytes()).purpose == "host-cost-profile"
    )
    shutil.rmtree(directory)
    write_unqueued_entry(
        root,
        framework=framework,
        version=version,
        system="b300_sxm",
        endpoint="chat",
        model=old.identity.model,
        runs=runs,
        tokenizer_manifest=Manifest.model_validate_json(tokenizer_path.read_bytes()),
        method={"measure": "test"},
    )
    return old


def test_a_manifest_that_differs_from_its_catalog_entry_is_refused(dataset, tmp_path):
    root = _copy(dataset, tmp_path)
    old = _rewrite_unqueued(root, "vllm", "0.31.0", [{"run": "new", "base_ms": 100.0, "total_per_input_token_us": 1.7}])
    new = Manifest.model_validate_json(
        (root / "data/host-cost/vllm/0.31.0/b300_sxm/qwen2/unqueued-chat/aisim-manifest.json").read_bytes()
    )
    # Same configuration ID; only the catalog still names the old manifest bytes.
    assert new.configuration_id == old.configuration_id
    with pytest.raises(MissingDataError, match="failed verification"):
        ensure_data(dict(CONFIG), DataPin(local_root=root, categories=("host-cost",)))


def test_a_total_smaller_than_its_tokenizer_stage_is_refused(dataset, tmp_path):
    root = _copy(dataset, tmp_path)
    # A fixed cost outside the engine (0.05 ms) below the selected tokenizer's fixed cost (about 0.3 ms) passes the
    # writer, but cannot be split into the two stages.
    _rewrite_unqueued(root, "sglang", "0.5.20", [{"run": "r", "base_ms": 0.05, "total_per_input_token_us": 1.56}])
    register_entries(root, ["data/host-cost/sglang/0.5.20/b300_sxm/glm/unqueued-chat/aisim-manifest.json"])
    config = {**CONFIG, "backend": "sglang", "backend_version": "0.5.20", "model": GLM}
    with pytest.raises(DataConflictError, match="cannot be split"):
        ensure_data(config, DataPin(local_root=root, categories=("host-cost",)))


def test_a_configuration_id_cannot_pick_host_cost_entries(local):
    config = dict(CONFIG)
    pin = DataPin(local_root=local, categories=("host-cost",), configuration_id="cfg-" + "0" * 64)
    with pytest.raises(DataConflictError, match="cannot select host-cost data"):
        ensure_data(config, pin)


def test_the_shipped_pin_names_a_full_commit():
    repo, revision = replay.host_cost_pin()
    assert repo == "ai-dynamo/aisim-data"
    assert re.fullmatch(r"[0-9a-f]{40}", revision)


def _prediction(*, backend="vllm", version="0.31.0", model=QWEN, hardware="b300_sxm", **engine):
    config = {
        "model": model,
        "hardware": hardware,
        "backend": backend,
        "backend_version": version,
        "context_length": 1024,
        "workers": {"aggregated": {}},
        **engine,
    }
    return CorePredictionConfig.model_validate({"engine": config})


def test_compiler_applies_the_measured_overhead_only_when_asked(local):
    off = prediction_to_replay_spec(_prediction()).backend_deployment
    assert off.tokenization is None and off.extra_cpu_overhead is None

    on = prediction_to_replay_spec(
        _prediction(), cpu_overhead_endpoint="chat", cpu_overhead_host_cpu=CPU
    ).backend_deployment
    measured = _measured()
    assert on.tokenization == pytest.approx(
        {
            "workers": 1,
            "base_ms": measured.tokenization["base_ms"],
            "per_input_token_ms": measured.tokenization["per_input_token_us"] / 1000,
        }
    )
    assert on.extra_cpu_overhead == pytest.approx(
        {
            "base_ms": measured.extra_cpu_overhead["base_ms"],
            "per_input_token_ms": measured.extra_cpu_overhead["per_input_token_us"] / 1000,
        }
    )

    glm = prediction_to_replay_spec(
        _prediction(backend="sglang", version="0.5.20", model=GLM),
        cpu_overhead_endpoint="chat",
        cpu_overhead_host_cpu=CPU,
    ).backend_deployment
    assert glm.tokenization["base_ms"] + glm.extra_cpu_overhead["base_ms"] == pytest.approx(12.6)

    unmeasured = prediction_to_replay_spec(
        _prediction(hardware="h200_sxm"), cpu_overhead_endpoint="chat", cpu_overhead_host_cpu=CPU
    ).backend_deployment
    assert unmeasured.tokenization is None and unmeasured.extra_cpu_overhead is None

    online = prediction_to_replay_spec(
        _prediction(), execution_mode="online", cpu_overhead_endpoint="chat", cpu_overhead_host_cpu=CPU
    ).backend_deployment
    assert online.tokenization is None and online.extra_cpu_overhead is None


def test_caller_set_fields_win_over_the_measured_overhead(local):
    both = prediction_to_replay_spec(
        _prediction(tokenization={"per_input_token_us": 2}, extra_cpu_overhead={"base_ms": 1}),
        cpu_overhead_endpoint="chat",
        cpu_overhead_host_cpu=CPU,
    ).backend_deployment
    assert both.tokenization == pytest.approx({"workers": 1, "base_ms": 0.0, "per_input_token_ms": 0.002})
    assert both.extra_cpu_overhead == pytest.approx({"base_ms": 1.0, "per_input_token_ms": 0.0})

    # An explicit zero turns one stage off and leaves the other measured.
    partial = prediction_to_replay_spec(
        _prediction(tokenization={"per_input_token_us": 0}), cpu_overhead_endpoint="chat", cpu_overhead_host_cpu=CPU
    ).backend_deployment
    assert partial.tokenization is None
    assert partial.extra_cpu_overhead["base_ms"] == pytest.approx(_measured().extra_cpu_overhead["base_ms"])


def test_compiler_resolves_an_omitted_backend_version_before_the_lookup(local, monkeypatch):
    import aisim_core.sdk.perf_database as perf_database

    # The documented default ``current`` resolves through the performance data's version slots.
    monkeypatch.setattr(perf_database, "resolve_query_version", lambda system, backend, version, **_: "0.31.0")
    config = _prediction()
    config = config.model_copy(update={"engine": config.engine.model_copy(update={"backend_version": None})})
    spec = prediction_to_replay_spec(config, cpu_overhead_endpoint="chat", cpu_overhead_host_cpu=CPU)
    deployment = spec.backend_deployment
    assert deployment.tokenization is not None and deployment.extra_cpu_overhead is not None
    assert deployment.extra_cpu_overhead["base_ms"] == pytest.approx(_measured().extra_cpu_overhead["base_ms"])


def test_the_compiler_needs_the_host_cpu_with_the_endpoint():
    with pytest.raises(ValueError, match="given together"):
        prediction_to_replay_spec(_prediction(), cpu_overhead_endpoint="chat")
