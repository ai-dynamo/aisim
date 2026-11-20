# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The op-latency pin composes every operation family with the system specs into one verified root."""

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisim_core.data.ensure import DataConflictError, DataPin, MissingDataError, ensure_data
from aisim_core.data.systems import PACKAGED_SYSTEMS_DIR
from aisim_core.sdk import ForwardPassPerfModelConfig
from aisim_core.sdk import rust_engine_step as res
from aisim_core.sdk.errors import PerfDataNotAvailableError

from .test_op_publish import _build, _table, tree  # noqa: F401  (fixture reuse)

pytestmark = pytest.mark.unit

CONFIG = {"system": "h200_sxm", "backend": "vllm", "backend_version": "1.0", "model": "any/model"}


@pytest.fixture
def two_families(tree, tmp_path):  # noqa: F811
    """gemm and attention at one hardware/backend/version, as the publisher emits them."""
    v1 = tree / "data/h200_sxm/gemm/vllm/1.0"
    attention = tree / "data/h200_sxm/attention/vllm/1.0"
    attention.mkdir(parents=True)
    pq.write_table(_table(3), attention / "attention_perf.parquet")
    shutil.copyfile(v1 / "collection_meta.yaml", attention / "collection_meta.yaml")
    out = tmp_path / "published"
    _build(tree, out)
    return out


def _pin(root, tmp_path, **kw):
    return DataPin(local_root=root, categories=("op-latency",), cache_dir=tmp_path / "cache", **kw)


def test_all_families_and_the_system_spec_arrive_in_one_root(two_families, tmp_path):
    out = ensure_data(dict(CONFIG), _pin(two_families, tmp_path))
    root = out.config["systems_paths"][0]
    families = {p.parent.parent.parent.name for p in (tmp_path / "cache").rglob("*_perf.parquet")}
    assert {"gemm", "attention"} <= families
    assert (Path(root) / "h200_sxm.yaml").is_file(), "the native database needs the system spec beside its tables"
    assert (Path(root) / "data/h200_sxm/attention/vllm/1.0").is_dir()
    assert len(out.datasets["op-latency"]["configuration_ids"]) == 2


@pytest.fixture
def native_dataset(tmp_path, monkeypatch):
    """Small synthetic tables, published through the real producer; no model downloads."""
    from aisim_core.sdk import utils

    def no_download(*args, **kwargs):
        pytest.fail("the synthetic local model must not download metadata")

    monkeypatch.setattr(utils, "_download_hf_json", no_download)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "max_position_embeddings": 128,
                "vocab_size": 16,
                "torch_dtype": "bfloat16",
            }
        )
    )
    systems = tmp_path / "producer-systems"
    gemm = systems / "data/h200_sxm/gemm/vllm/1.0"
    attention = systems / "data/h200_sxm/attention/vllm/1.0"
    gemm.mkdir(parents=True)
    attention.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"gemm_dtype": "bfloat16", "m": 4, "n": n, "k": k, "latency": 1.0}
                for n in (8, 16, 24, 32)
                for k in (8, 16)
            ]
        ),
        gemm / "gemm_perf.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "batch_size": 1,
                    "isl": 4,
                    "num_heads": 2,
                    "num_key_value_heads": 2,
                    "head_dim": 4,
                    "attn_dtype": "bfloat16",
                    "kv_cache_dtype": "bfloat16",
                    "latency": 7.0,
                }
            ]
        ),
        attention / "context_attention_perf.parquet",
    )
    shutil.copyfile(PACKAGED_SYSTEMS_DIR / "h200_sxm.yaml", systems / "h200_sxm.yaml")
    (systems / "perf_data_reuse_manifest.yaml").write_text("groups: []\n")
    (systems / "query_versions.yaml").write_text(
        'schema_version: 1\ndefaults: {vllm: {current: "1.0", previous: null}}\noverrides: {}\n'
    )
    published = tmp_path / "native-published"
    _build(systems, published)
    config = ForwardPassPerfModelConfig(
        model=str(model),
        system="h200_sxm",
        backend="vllm",
        backend_version="1.0",
        worker_type="prefill",
        estimation_mode="op_level",
        enable_shared_layer=False,
        systems_paths=(str(systems),),
        estimator_config={"correction": {"enabled": False}},
    )
    return published, systems, config


@pytest.mark.parametrize("as_mapping", [False, True])
def test_the_best_available_consumer_receives_the_complete_root(native_dataset, tmp_path, as_mapping):
    published, systems, config = native_dataset
    # Remove the producer tables: prediction must consume both published families.
    shutil.rmtree(systems / "data")
    payload = config.to_dict() if as_mapping else config
    if as_mapping:
        payload.pop("systems_paths")
    model = res.RustForwardPassPerfModel.best_available(payload, dataset=_pin(published, tmp_path))
    provenance = model.diagnostics()["provenance"]
    selected = Path(provenance["selected_systems_root"])
    assert selected != systems
    assert (selected / "h200_sxm.yaml").is_file()
    assert (selected / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet").is_file()
    assert (selected / "data/h200_sxm/attention/vllm/1.0/context_attention_perf.parquet").is_file()
    assert model.dataset_revisions == {"op-latency": f"local:{published}"}
    # The single synthetic layer alone has a 7 ms attention measurement;
    # GEMMs and other operations add positive time. This is no accuracy claim.
    assert model.predict_prefill_latency(1, 4) >= 7.0


@pytest.mark.parametrize("as_mapping", [False, True])
@pytest.mark.parametrize("family", ["gemm", "attention"])
def test_incomplete_pin_cannot_fall_back_to_other_operation_tables(native_dataset, tmp_path, as_mapping, family):
    from aisim_core.data.ensure import _matching_entries
    from aisim_core.data.loaders import OpLatencyLoader

    published, systems, config = native_dataset
    payload = config.to_dict() if as_mapping else config
    # The supplied fallback root is valid and complete, so it would mask a
    # missing pinned family if best_available tried it after the ensured root.
    assert res.RustForwardPassPerfModel.best_available(payload).predict_prefill_latency(1, 4) >= 7.0
    entry = next(
        e
        for e in _matching_entries(published, OpLatencyLoader().select(config.to_dict()))
        if e.identity.operation_family == family
    )
    missing_table = "context_attention_perf" if family == "gemm" else "gemm_perf"
    with pytest.raises(PerfDataNotAvailableError, match=missing_table):
        res.RustForwardPassPerfModel.best_available(
            payload, dataset=_pin(published, tmp_path, configuration_id=entry.configuration_id)
        )


def test_second_call_reuses_the_materialized_root(two_families, tmp_path):
    first = ensure_data(dict(CONFIG), _pin(two_families, tmp_path)).config["systems_paths"][0]
    again = ensure_data(dict(CONFIG), _pin(two_families, tmp_path)).config["systems_paths"][0]
    assert first == again


def test_two_entries_for_one_family_are_a_real_conflict(two_families, tmp_path, monkeypatch):
    from aisim_core.data import ensure as mod

    real = mod._matching_entries
    monkeypatch.setattr(mod, "_matching_entries", lambda root, sel: [*real(root, sel), real(root, sel)[0]])
    with pytest.raises(DataConflictError, match="one operation family"):
        ensure_data(dict(CONFIG), _pin(two_families, tmp_path))


def test_configuration_id_still_selects_one_entry(two_families, tmp_path):
    from aisim_core.data.ensure import _matching_entries
    from aisim_core.data.loaders import OpLatencyLoader

    entries = _matching_entries(two_families, OpLatencyLoader().select(CONFIG))
    out = ensure_data(dict(CONFIG), _pin(two_families, tmp_path, configuration_id=entries[0].configuration_id))
    assert out.datasets["op-latency"]["configuration_ids"] == [entries[0].configuration_id]


def test_unpublished_version_has_no_fallback(two_families, tmp_path):
    with pytest.raises(MissingDataError):
        ensure_data({**CONFIG, "backend_version": "9.9"}, _pin(two_families, tmp_path))
