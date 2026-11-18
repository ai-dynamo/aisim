# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FPM contract: loader over library plus sidecar, staging, resource precedence and entry checks."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisim_core.data import Identity, Manifest, configuration_id
from aisim_core.data.fetch import Cache, plan
from aisim_core.data.fpm_checks import check_fpm_dataset, check_fpm_entry, check_fpm_identities
from aisim_core.data.fpm_library import (
    EVALUATION_ROLES,
    NATIVE_REQUIRED_COLUMNS,
    FpmContractError,
    FpmLibraryMetadata,
    FpmResourcesPendingError,
    complete_fpm_profile,
    missing_resource_fields,
    read_library_metadata,
    recover_reconstructed_metadata,
    resolve_resources,
    stage_fpm_library,
    with_estimator_path,
)
from aisim_core.data.loaders import FpmLoader, LoaderError
from aisim_core.data.schemas import SCHEMA_MODELS, check_snapshots, render_snapshot
from aisim_core.sdk import RustForwardPassPerfModel, engine
from aisim_core.sdk.fpm_profile import _require_forward_pass_profile_memory, load_fpm_profile
from tests.unit.sdk.test_fpm_profile import (  # noqa: F401
    _fail_graph,
    _request,
    _write_external_profile_pair,
    checkpoint_configs,
    profile_dict,
)

pytestmark = pytest.mark.unit

REPO, SHA = "org/dataset", "c" * 40
ENTRY = "data/fpm/MiniMaxAI--MiniMax-M2.7/h200-sxm/vllm/0.25.1/tp4"
MANIFEST = f"{ENTRY}/aisim-manifest.json"
RESOURCES = {
    "cache_layout": "linear",
    "kv_bytes_per_token": 65536,
    "max_num_tokens": 8192,
    "max_batch_size": 256,
    "provenance": "Fixture resources; not a measurement.",
}
RUNTIME_MEMORY = {
    "kv_cache_bytes": 10**9,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 4096,
    "provenance": "Fixture runtime memory; not a measurement.",
}
CONFIG = {
    "model": "MiniMaxAI/MiniMax-M2.7",
    "system": "h200_sxm",
    "backend": "vllm",
    "backend_version": "0.25.1",
    "worker_type": "aggregated",
    "tp": 4,
    "attention_dp": 1,
    "moe_tp_size": 4,
    "moe_ep_size": 1,
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact(root: Path, relative: str, role: str, fmt: str, schema: str, **extra) -> dict:
    data = (root / relative).read_bytes()
    return {
        "artifact_id": relative.removeprefix(f"{ENTRY}/"),
        "path": relative,
        "role": role,
        "format": fmt,
        "schema": schema,
        "sha256": _sha(data),
        "bytes": len(data),
        "rows": extra.get("rows"),
        "runtime_path": None,
    }


@pytest.fixture
def profile(profile_dict, monkeypatch):  # noqa: F811
    """A MiniMax pure-TP4 profile whose deployment omits ``resources`` (the user's draft)."""
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    for name in ("get_model", "build_model_config", "_maybe_load_database"):
        monkeypatch.setattr(engine, name, _fail_graph)
    profile_dict.update(model="MiniMaxAI/MiniMax-M2.7", architecture="MiniMaxM2ForCausalLM")
    profile_dict["deployments"][0].update(
        system="h200_sxm", tp=4, dp=1, moe_tp=4, moe_ep=1, gemm_quant_mode="fp8_block", moe_quant_mode="fp8_block"
    )
    profile_dict["deployments"][0]["fmha_quant_mode"] = "bfloat16"
    return profile_dict


def make_entry(root: Path, profile: dict, *, resources=None, with_measurements=False, sidecar_extra=None):
    """Write one published-shaped entry. Measurement files exist only on request, to prove they are never needed."""
    directory = root / ENTRY / "fpm"
    directory.mkdir(parents=True, exist_ok=True)
    drafted = copy.deepcopy(profile)
    drafted["deployments"][0].setdefault("resources", RESOURCES)
    source = _write_external_profile_pair(drafted, directory)
    source.rename(directory / "fpm_forward_perf.parquet")
    sidecar = json.loads(source.with_suffix(".metadata.json").read_text())
    source.with_suffix(".metadata.json").unlink()
    parquet = directory / "fpm_forward_perf.parquet"
    sidecar["parquet_sha256"] = _sha(parquet.read_bytes())
    sidecar["configuration_selector"] = {"tp": 4, "pp": 1, "dp": 1, "moe_tp": 4, "moe_ep": 1, "cp": 1}
    if resources is not None:
        sidecar["resources"] = resources
    sidecar.update(sidecar_extra or {})
    (directory / "fpm_forward_perf.metadata.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    artifacts = [
        _artifact(
            root,
            f"{ENTRY}/fpm/fpm_forward_perf.parquet",
            "fpm-library",
            "parquet",
            "aic_fpm_forward_perf/v6",
            rows=sidecar["row_count"],
        ),
        _artifact(root, f"{ENTRY}/fpm/fpm_forward_perf.metadata.json", "fpm-library-metadata", "json", "unversioned"),
    ]
    (root / ENTRY / "measurements").mkdir(parents=True, exist_ok=True)
    manifest_bytes = b'{"manifest_version": 5}'
    iterations = b"not parquet: never read"
    if with_measurements:
        (root / ENTRY / "measurements/manifest.json").write_bytes(manifest_bytes)
        (root / ENTRY / "measurements/iterations-01.parquet").write_bytes(iterations)
    for relative, data, role, fmt, schema in (
        ("measurements/manifest.json", manifest_bytes, "measurement-manifest", "json", "measurement-manifest/v5"),
        ("measurements/iterations-01.parquet", iterations, "measurement-data", "parquet", "unversioned"),
        ("manifest.json", b"{}", "legacy-configuration-manifest", "json", "configuration-manifest/v4"),
    ):
        artifacts.append(
            {
                "artifact_id": relative,
                "path": f"{ENTRY}/{relative}",
                "role": role,
                "format": fmt,
                "schema": schema,
                "sha256": _sha(data),
                "bytes": len(data),
                "rows": 1 if fmt == "parquet" else None,
                "runtime_path": None,
            }
        )
    identity = Identity(
        hardware="h200_sxm",
        framework="vllm",
        framework_version="0.25.1",
        operation_family=None,
        model="MiniMaxAI/MiniMax-M2.7",
        effective_settings={"parallelism_label": "tp4"},
        unresolved_identity=None,
        parallelism={"aggregated": {"tp": 4, "moe_tp": 4}},
        parallelism_complete=True,
        precision={"kv_cache": "fp8", "weights": "fp8_block"},
    )
    manifest = {
        "schema_version": "aisim-artifact-manifest/v1",
        "category": "fpm",
        "purpose": "execution-profile",
        "configuration_id": configuration_id("fpm", "execution-profile", identity),
        "identity": identity.model_dump(),
        "requested_settings": None,
        "artifacts": artifacts,
        "provenance": {},
        "evidence_gaps": [],
    }
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return MANIFEST


@pytest.fixture
def root(tmp_path):
    return tmp_path / "dataset"


def user_config(profile, **resources):
    drafted = copy.deepcopy(profile)
    drafted["deployments"][0].pop("resources", None)
    if resources:
        drafted["deployments"][0]["resources"] = resources
    return {**_request(profile, "direct"), "fpm_profile": drafted, "attention_dp": 1}


def load(root, config, tmp_path, **kwargs):
    return FpmLoader().load(root, MANIFEST, output_dir=tmp_path / "staged", config=config, **kwargs)


# ------------------------------------------------------------------ loader on library + metadata only


def test_loader_builds_a_model_from_library_and_metadata_only(root, profile, tmp_path):
    make_entry(root, profile, resources=RESOURCES)  # no measurement or legacy file exists on disk
    assert not (root / ENTRY / "measurements/manifest.json").exists()
    data = load(root, user_config(profile), tmp_path)
    assert data.kind == "library" and data.resource_sources == ("metadata",)
    model = RustForwardPassPerfModel.best_available(data.config)
    prefill = {"scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 1}}
    decode = {"scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": 1}}
    assert model.estimate_forward_pass_time_ms(prefill) == pytest.approx(2.0)
    assert model.estimate_forward_pass_time_ms(decode) == pytest.approx(3.0)
    # The staged root is what the native model reads, and the original files are only linked.
    staged = list(data.systems_root.glob("**/fpm_forward_perf.parquet"))
    assert len(staged) == 1 and staged[0].is_symlink()
    assert data.config["systems_paths"][0] == str(data.systems_root)
    assert data.config["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"] == str(staged[0])


def test_user_choices_are_kept(root, profile, tmp_path):
    make_entry(root, profile, resources=RESOURCES)
    config = {**user_config(profile), "systems_paths": [str(tmp_path / "mine")]}
    config["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"] = "/user/chosen.parquet"
    data = load(root, config, tmp_path)
    assert data.config["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"] == "/user/chosen.parquet"
    assert data.config["systems_paths"][1:] == [str(tmp_path / "mine")]


def test_loader_rejects_tampered_library_and_wrong_identity(root, profile, tmp_path):
    make_entry(root, profile, resources=RESOURCES)
    with pytest.raises(LoaderError, match="differs from configuration"):
        load(root, {**user_config(profile), "system": "b200_sxm"}, tmp_path)
    with pytest.raises(LoaderError, match="parallelism"):
        load(root, {**user_config(profile), "tp": 2, "moe_tp_size": 2}, tmp_path)
    library = root / ENTRY / "fpm/fpm_forward_perf.parquet"
    library.write_bytes(library.read_bytes() + b"x")
    with pytest.raises(ValueError, match="integrity"):
        load(root, user_config(profile), tmp_path)


def test_default_selection_and_plan_skip_measurements(root, profile, tmp_path):
    make_entry(root, profile, resources=RESOURCES, with_measurements=True)
    cache_root = tmp_path / "cache"
    commit = Cache(cache_root).commit_root(REPO, SHA)
    commit.parent.mkdir(parents=True)
    root.rename(commit)
    (commit / "catalog/fpm").mkdir(parents=True)
    manifest = Manifest.model_validate_json((commit / MANIFEST).read_bytes())
    index = {
        "schema_version": "aisim-category-index/v1",
        "category": "fpm",
        "entries": [
            {
                "configuration_id": manifest.configuration_id,
                "purpose": "execution-profile",
                "identity": manifest.identity.model_dump(),
                "manifest": _artifact(commit, MANIFEST, "manifest", "json", "aisim-artifact-manifest/v1"),
            }
        ],
    }
    (commit / "catalog/fpm/index.json").write_text(json.dumps(index))
    ref = _artifact(commit, "catalog/fpm/index.json", "index", "json", "aisim-category-index/v1")
    (commit / "catalog/dataset.json").write_text(
        json.dumps(
            {
                "schema_version": "aisim-dataset-index/v1",
                "dataset_id": REPO,
                "categories": {"fpm": {"root": "data/fpm", "index": ref, "layout": "category"}},
            }
        )
    )
    selection = FpmLoader().select(CONFIG)
    assert set(EVALUATION_ROLES) <= set(selection.exclude_roles) and selection.parallelism == (("moe_tp", 4), ("tp", 4))
    default = plan(selection, repo=REPO, revision=SHA, cache=Cache(cache_root), offline=True)
    everything = plan(
        selection.__class__(**{**selection.__dict__, "exclude_roles": ()}),
        repo=REPO,
        revision=SHA,
        cache=Cache(cache_root),
        offline=True,
    )
    planned = {f.path.removeprefix(f"{ENTRY}/") for f in default.files if f.role == "artifact"}
    assert planned == {"fpm/fpm_forward_perf.parquet", "fpm/fpm_forward_perf.metadata.json"}
    assert not default.missing and default.total_bytes < everything.total_bytes
    assert {f.path for f in everything.files if f.role == "artifact"} > {
        f.path for f in default.files if f.role == "artifact"
    }


def test_parallelism_is_part_of_selection(root, profile):
    from aisim_core.data.fetch import Selection

    make_entry(root, profile)
    manifest = Manifest.model_validate_json((root / MANIFEST).read_bytes())
    entry = type("E", (), {"configuration_id": "x", "purpose": "execution-profile", "identity": manifest.identity})
    base = dict(category="fpm", model="MiniMaxAI/MiniMax-M2.7", hardware="h200_sxm")
    assert Selection(**base, parallelism=(("moe_tp", 4), ("tp", 4))).matches(entry)
    assert not Selection(**base, parallelism=(("dp", 4), ("ep", 4))).matches(entry)  # dep4 is another entry
    assert FpmLoader().select({**CONFIG, "tp": 1, "moe_tp_size": 1}).parallelism == ()  # explicit all-ones
    assert FpmLoader().select({"model": "m", "system": "h200_sxm"}).parallelism is None


# ------------------------------------------------------------------ resources: precedence and pending


def test_user_resources_override_sidecar_resources(root, profile, tmp_path):
    make_entry(root, profile, resources=RESOURCES)
    user = {**RESOURCES, "max_num_tokens": 4096, "provenance": "User override fixture."}
    data = load(root, user_config(profile, **user), tmp_path)
    assert data.resource_sources == ("user",)
    assert data.profile.deployments[0].resources.max_num_tokens == 4096
    assert data.config["fpm_profile"]["deployments"][0]["resources"]["provenance"] == "User override fixture."


def test_sidecar_resources_fill_an_omitted_block_and_round_trip(root, profile, tmp_path):
    make_entry(root, profile, resources={**RESOURCES, "runtime_memory": RUNTIME_MEMORY})
    data = load(root, user_config(profile), tmp_path)
    assert data.resource_sources == ("metadata",) and data.memory_pending == ()
    model = RustForwardPassPerfModel.best_available(data.config)
    reloaded = RustForwardPassPerfModel.best_available(
        json.loads(json.dumps(model.diagnostics()["provenance"]["config"]))
    )
    assert reloaded.diagnostics()["provenance"]["config"] == model.diagnostics()["provenance"]["config"]
    _require_forward_pass_profile_memory(RustForwardPassPerfModel.normalize_config(data.config))  # ready for replay


def test_pending_names_exactly_the_missing_fields(root, profile, tmp_path):
    make_entry(root, profile)  # an older entry: no resources in the sidecar
    with pytest.raises(FpmResourcesPendingError) as error:
        load(root, user_config(profile), tmp_path)
    assert error.value.fields == (
        "cache_layout",
        "max_num_tokens",
        "max_batch_size",
        "provenance",
        "kv_bytes_per_token (linear) or cache_groups (grouped)",
    )
    for name in ("cache_layout", "max_num_tokens", "max_batch_size", "provenance"):
        assert name in str(error.value)
    assert "runtime_memory" in str(error.value) and "fpm_forward_perf.metadata.json" in str(error.value)
    assert missing_resource_fields({"cache_layout": "linear", "max_num_tokens": 1}) == [
        "max_batch_size",
        "provenance",
        "kv_bytes_per_token",
    ]
    assert missing_resource_fields({"cache_layout": "grouped", **{k: 1 for k in RESOURCES if k != "cache_layout"}}) == [
        "cache_groups"
    ]


def test_old_entry_without_resources_still_loads_for_timing(root, profile, tmp_path):
    make_entry(root, profile)
    config = {**_request(profile, "sol", estimation_mode="fpm_interpolation"), "attention_dp": 1}
    config.pop("fpm_profile")
    config["estimator_config"]["fpm_interpolation"].update(text_only=True)
    data = load(root, config, tmp_path)
    assert data.resources.source == "pending" and data.resources.resources is None and data.profile is None
    with pytest.raises(FpmResourcesPendingError, match="cache_layout"):
        load(root, config, tmp_path / "again", require_resources=True)
    assert "cache_layout" in data.memory_pending and "max_batch_size" in data.memory_pending


def test_pending_memory_serves_timing_and_blocks_replay(root, profile, tmp_path):
    make_entry(root, profile, resources=RESOURCES)  # geometry known, memory not yet profiled
    data = load(root, user_config(profile), tmp_path)
    assert data.memory_pending == (
        "runtime_memory, or all of weights_bytes, activations_bytes, runtime_overhead_bytes, comm_overhead_bytes",
    )
    model = RustForwardPassPerfModel.best_available(data.config)
    assert model.estimate_forward_pass_time_ms(
        {"scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 1}}
    ) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="FPM memory is pending runtime profiling"):
        _require_forward_pass_profile_memory(RustForwardPassPerfModel.normalize_config(data.config))


def test_invalid_sidecar_resources_are_rejected_with_the_field(root, profile, tmp_path):
    make_entry(root, profile, resources={**RESOURCES, "surprise": 1})
    with pytest.raises(LoaderError, match="surprise"):
        load(root, user_config(profile), tmp_path)
    assert resolve_resources(None, None).source == "pending"
    with pytest.raises(FpmContractError, match="max_batch_size"):
        resolve_resources({k: v for k, v in RESOURCES.items() if k != "max_batch_size"}, None)


# ------------------------------------------------------------------ staging and recovery


def test_staging_links_files_and_reuses_a_verified_root(root, profile, tmp_path):
    make_entry(root, profile, resources=RESOURCES)
    parquet, sidecar = (
        root / ENTRY / "fpm" / name for name in ("fpm_forward_perf.parquet", "fpm_forward_perf.metadata.json")
    )
    kwargs = dict(system="h200_sxm", backend="vllm", backend_version="0.25.1", output_dir=tmp_path / "out")
    first = stage_fpm_library(parquet, sidecar, **kwargs)
    assert first.parquet_path.resolve() == parquet.resolve() and first.metadata_path.resolve() == sidecar.resolve()
    assert first.resources == RESOURCES and first.diagnostics["fpm_row_count"] == 3
    marker = first.systems_root / ".staged.json"
    marker.write_text(marker.read_text())  # an untouched marker is what makes a root reusable
    inode = first.parquet_path.lstat().st_ino
    again = stage_fpm_library(parquet, sidecar, **kwargs)
    assert again.parquet_path.lstat().st_ino == inode
    sidecar.write_text(sidecar.read_text().replace('"row_count": 3', '"row_count": 3 '))  # new bytes: restaged
    rebuilt = stage_fpm_library(parquet, sidecar, **kwargs)
    assert rebuilt.parquet_path.lstat().st_ino != inode
    temp = stage_fpm_library(parquet, sidecar, system="h200_sxm", backend="vllm", backend_version="0.25.1")
    root_dir = temp.systems_root
    assert root_dir.is_dir()
    temp.close()
    assert not root_dir.exists()


def test_staging_reuse_follows_the_source_paths(root, profile, tmp_path):
    make_entry(root, profile, resources=RESOURCES)
    source = root / ENTRY / "fpm"
    kwargs = dict(system="h200_sxm", backend="vllm", backend_version="0.25.1", output_dir=tmp_path / "out")
    stage_fpm_library(source / "fpm_forward_perf.parquet", source / "fpm_forward_perf.metadata.json", **kwargs)
    moved = tmp_path / "moved"
    shutil.copytree(source, moved)
    shutil.rmtree(source)
    staged = stage_fpm_library(moved / "fpm_forward_perf.parquet", moved / "fpm_forward_perf.metadata.json", **kwargs)
    assert staged.parquet_path.resolve() == (moved / "fpm_forward_perf.parquet").resolve()


def test_restaging_a_returned_configuration_never_links_through_its_own_root(root, profile, tmp_path):
    make_entry(root, profile, resources=RESOURCES)
    source = root / ENTRY / "fpm"
    kwargs = dict(system="h200_sxm", backend="vllm", backend_version="0.25.1", output_dir=tmp_path / "out")
    args = (source / "fpm_forward_perf.parquet", source / "fpm_forward_perf.metadata.json")
    first = stage_fpm_library(*args, **kwargs)
    again = stage_fpm_library(*args, systems_paths=[first.systems_root], **kwargs)
    links = [p for p in again.systems_root.rglob("*") if p.is_symlink()]
    assert links and all(p.exists() for p in links)


def test_systems_paths_use_the_canonical_default_and_home_forms(monkeypatch, tmp_path):
    from aisim_core.data.fpm_library import _stageable_roots

    monkeypatch.setenv("HOME", str(tmp_path))
    roots = _stageable_roots(["default", "~/custom"], None)
    assert roots[0].name == "systems" and roots[0].is_absolute() and roots[1] == tmp_path / "custom"
    assert _stageable_roots(["~/custom", "/x"], tmp_path / "custom") == [Path("/x")]


def test_sidecar_resources_need_the_library_identity(root, profile):
    selector = {"tp": 4, "pp": 1, "dp": 1, "moe_tp": 4, "moe_ep": 1, "cp": 1, "backend_version": "0.25.1"}
    metadata = {"configuration_selector": selector, "resources": RESOURCES}
    draft = copy.deepcopy(profile)
    draft["deployments"][0].pop("resources", None)
    draft["deployments"][0].update(backend_version="0.25.1")
    _, sources = complete_fpm_profile(draft, metadata, identity={"system": "h200_sxm", "backend_version": "0.25.1"})
    assert sources == ["metadata"]
    other_version = copy.deepcopy(draft)
    other_version["deployments"][0]["backend_version"] = "0.26.0"
    with pytest.raises(FpmResourcesPendingError):
        complete_fpm_profile(other_version, metadata)
    other_hardware = copy.deepcopy(draft)
    other_hardware["deployments"][0]["system"] = "b200_sxm"
    with pytest.raises(FpmResourcesPendingError):
        complete_fpm_profile(other_hardware, metadata, identity={"system": "h200_sxm"})


def test_estimator_path_insertion_leaves_the_callers_mapping_alone():
    given = {"estimator_config": {"fpm_interpolation": {"k": 1}}}
    out = dict(given)
    with_estimator_path(out, Path("/staged/a.parquet"))
    assert given == {"estimator_config": {"fpm_interpolation": {"k": 1}}}
    assert out["estimator_config"]["fpm_interpolation"] == {"k": 1, "fpm_parquet_path": "/staged/a.parquet"}


def test_temporary_staging_lives_as_long_as_the_loaded_data(root, profile):
    make_entry(root, profile, resources=RESOURCES)
    data = FpmLoader().load(root, MANIFEST, config=user_config(profile))
    gc.collect()
    assert (
        data.systems_root.is_dir()
        and Path(data.config["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"]).exists()
    )
    data.close()
    assert not data.systems_root.exists()


def test_staging_rejects_unknown_system_and_unsafe_names(root, profile, tmp_path):
    make_entry(root, profile)
    files = (root / ENTRY / "fpm/fpm_forward_perf.parquet", root / ENTRY / "fpm/fpm_forward_perf.metadata.json")
    with pytest.raises(FpmContractError, match="no system definition"):
        stage_fpm_library(*files, system="no_such_gpu", backend="vllm", backend_version="1")
    with pytest.raises(FpmContractError, match="safe path component"):
        stage_fpm_library(*files, system="h200_sxm", backend="vllm", backend_version="../x")


def test_library_schema_versions(tmp_path):
    def sidecar(version):
        path = tmp_path / f"v{version}.json"
        path.write_text(json.dumps({"schema_name": "aic_fpm_forward_perf", "schema_version": version, "row_count": 1}))
        return path

    assert read_library_metadata(sidecar(6))["schema_version"] == 6
    for version in (7, 8):
        with pytest.raises(FpmContractError, match=f"v{version} is not supported yet.*follow-up"):
            read_library_metadata(sidecar(version))
    with pytest.raises(FpmContractError, match="unsupported FPM library schema version 5"):
        read_library_metadata(sidecar(5))


RECONSTRUCTED_ROW = {
    "model_path": "m",
    "system": "h200_sxm",
    "backend": "vllm",
    "backend_version": "0.25.1",
    "partition_policy": "balanced_v1",
    "batch_size": 1,
    "total_prefill_tokens": 1,
    "total_kv_read_tokens": 0,
    "measurement_policy": "dynamo_native_single_sample_v1",
    "measurement_repeats": 1,
    "warmup_repeats": 0,
    "latency_ms": 1.0,
}


def reconstructed(tmp_path, *, row_changes=None):
    row = {**RECONSTRUCTED_ROW, **(row_changes or {})}
    parquet = tmp_path / "lib.parquet"
    pq.write_table(pa.Table.from_pylist([row]), parquet)
    digest = _sha(parquet.read_bytes())
    selector = {k: RECONSTRUCTED_ROW[k] for k in ("model_path", "system", "backend", "backend_version")}
    metadata = {
        "schema_name": "aic_fpm_forward_perf",
        "schema_version": 6,
        "metadata_provenance": "reconstructed_from_parquet",
        "configuration_selector": selector,
        "row_count": 1,
        "parquet_sha256": digest,
    }
    return metadata, parquet, digest


def test_reconstructed_schema6_metadata_is_recovered_only_from_verified_rows(tmp_path):
    metadata, parquet, digest = reconstructed(tmp_path)
    recovered, facts = recover_reconstructed_metadata(
        metadata, parquet, parquet_sha256=digest, metadata_sha256="0" * 64
    )
    assert recovered["coordinate_system"] == "iteration_totals_balanced_v1" and recovered["warmup_repeats"] == 0
    assert set(facts) >= {"coordinate_system", "measurement_policy", "system", "backend", "backend_version"}
    with pytest.raises(FpmContractError, match="pinned hashes"):
        recover_reconstructed_metadata(metadata, parquet, parquet_sha256="1" * 64, metadata_sha256="0" * 64)
    metadata, parquet, digest = reconstructed(tmp_path, row_changes={"partition_policy": "other"})
    with pytest.raises(FpmContractError, match="balanced_v1"):
        recover_reconstructed_metadata(metadata, parquet, parquet_sha256=digest, metadata_sha256="0" * 64)
    complete = {
        **metadata,
        "coordinate_system": "x",
        "measurement_policy": "y",
        "system": "s",
        "backend": "b",
        "backend_version": "v",
    }
    assert recover_reconstructed_metadata(complete, parquet, parquet_sha256="9" * 64, metadata_sha256="0" * 64) == (
        complete,
        {},
    )


def test_staging_writes_recovered_sidecar_copy_and_leaves_the_original(tmp_path):
    metadata, parquet, _ = reconstructed(tmp_path)
    sidecar = tmp_path / "lib.metadata.json"
    sidecar.write_text(json.dumps(metadata))
    staged = stage_fpm_library(
        parquet, sidecar, system="h200_sxm", backend="vllm", backend_version="0.25.1", output_dir=tmp_path / "out"
    )
    assert not staged.metadata_path.is_symlink()
    assert json.loads(staged.metadata_path.read_text())["coordinate_system"] == "iteration_totals_balanced_v1"
    assert json.loads(sidecar.read_text()) == metadata
    recovery = staged.diagnostics["fpm_metadata_recovery"]
    assert recovery["policy"] == "schema6_reconstructed_row_contract_v1"


# ------------------------------------------------------------------ entry checks


def test_clean_entry_has_no_findings_without_measurement_files(root, profile):
    make_entry(root, profile, resources=RESOURCES)
    assert check_fpm_entry(root, MANIFEST) == []
    codes = {f.code for f in check_fpm_entry(root, MANIFEST, include_measurements=True)}
    assert codes == {"hash-mismatch"}  # the evaluation tier is only checked on request, and is absent here


def test_entry_checks_report_roles_hashes_and_sidecar_agreement(root, profile):
    make_entry(root, profile)
    path = root / MANIFEST
    document = json.loads(path.read_text())

    def check(mutate):
        changed = copy.deepcopy(document)
        mutate(changed)
        path.write_text(json.dumps(changed))
        return {f.code for f in check_fpm_entry(root, MANIFEST)}

    assert check(lambda d: d.update(artifacts=[a for a in d["artifacts"] if a["role"] != "fpm-library"])) == {
        "missing-role"
    }
    assert check(
        lambda d: d["artifacts"].append({**d["artifacts"][0], "artifact_id": "dup", "path": d["artifacts"][1]["path"]})
    ) == {"duplicate-role"}
    assert check(lambda d: d.update(artifacts=[a for a in d["artifacts"] if a["role"] != "measurement-manifest"])) == {
        "missing-role"
    }
    assert check(lambda d: d["artifacts"][0].update(sha256="0" * 64)) == {"hash-mismatch"}
    assert check(lambda d: d["artifacts"][0].update(schema="aic_fpm_forward_perf/v7")) == {"library-schema"}
    path.write_text(json.dumps(document))
    sidecar = root / ENTRY / "fpm/fpm_forward_perf.metadata.json"
    changed = json.loads(sidecar.read_text())
    changed["parquet_sha256"] = "2" * 64
    changed["row_count"] = 99
    sidecar.write_text(json.dumps(changed))
    for artifact in document["artifacts"]:
        if artifact["role"] == "fpm-library-metadata":
            artifact.update(sha256=_sha(sidecar.read_bytes()), bytes=sidecar.stat().st_size)
    path.write_text(json.dumps(document))
    assert {f.code for f in check_fpm_entry(root, MANIFEST)} == {"sidecar-hash", "sidecar-rows"}


def test_one_identity_maps_to_at_most_one_entry(root, profile):
    make_entry(root, profile)
    manifest = Manifest.model_validate_json((root / MANIFEST).read_bytes())
    assert check_fpm_identities([("a/aisim-manifest.json", manifest)]) == []
    other = manifest.model_copy(
        update={"identity": manifest.identity.model_copy(update={"parallelism": {"aggregated": {"tp": 8}}})}
    )
    assert check_fpm_identities([("a", manifest), ("b", other)]) == []
    twin = manifest.model_copy(
        update={"identity": manifest.identity.model_copy(update={"effective_settings": {"x": 1}})}
    )
    findings = check_fpm_identities([("a", manifest), ("b", twin)])
    assert [(f.code, f.path) for f in findings] == [("duplicate-identity", "b")] and "same identity as a" in findings[
        0
    ].message


def test_dataset_check_runs_every_entry_check(root, profile):
    make_entry(root, profile)
    manifest = Manifest.model_validate_json((root / MANIFEST).read_bytes())
    (root / "catalog/fpm").mkdir(parents=True)
    reference = _artifact(root, MANIFEST, "manifest", "json", "aisim-artifact-manifest/v1")
    entry = {
        "configuration_id": manifest.configuration_id,
        "purpose": "execution-profile",
        "identity": manifest.identity.model_dump(),
        "manifest": reference,
    }
    index = {"schema_version": "aisim-category-index/v1", "category": "fpm", "entries": [entry, entry]}
    (root / "catalog/fpm/index.json").write_text(json.dumps(index))
    (root / "catalog/dataset.json").write_text(
        json.dumps(
            {
                "schema_version": "aisim-dataset-index/v1",
                "dataset_id": REPO,
                "categories": {
                    "fpm": {
                        "root": "data/fpm",
                        "index": _artifact(root, "catalog/fpm/index.json", "index", "json", "x"),
                        "layout": "category",
                    }
                },
            }
        )
    )
    assert [f.code for f in check_fpm_dataset(root)] == ["duplicate-identity"]
    assert [f.code for f in check_fpm_dataset(root / "nowhere")] == ["invalid-catalog"]


# ------------------------------------------------------------------ schema snapshots


def test_fpm_schema_snapshots_are_registered_and_current():
    assert {"aic_fpm_forward_perf/v6", "aic_fpm_forward_perf.metadata/v6", "measurement-manifest/v5"} <= set(
        SCHEMA_MODELS
    )
    assert check_snapshots() == []
    schema = json.loads(render_snapshot("aic_fpm_forward_perf.metadata/v6"))
    assert "resources" in schema["properties"] and "resources" not in schema.get("required", [])
    assert schema["properties"]["schema_name"]["const"] == "aic_fpm_forward_perf"


def test_sidecar_model_accepts_published_shapes_and_validates_resources():
    base = {"schema_name": "aic_fpm_forward_perf", "schema_version": 6, "row_count": 2, "future_key": {"a": 1}}
    assert FpmLibraryMetadata.model_validate(base).resources is None  # entries published before `resources`
    assert FpmLibraryMetadata.model_validate({**base, "resources": RESOURCES}).resources.max_num_tokens == 8192
    with pytest.raises(ValueError, match="max_num_tokens"):
        FpmLibraryMetadata.model_validate({**base, "resources": {**RESOURCES, "max_num_tokens": 0}})
    assert load_fpm_profile  # the profile class stays the single source for resources


def test_entry_checks_report_unreadable_files_and_missing_native_columns(root, profile):
    make_entry(root, profile)
    library = root / ENTRY / "fpm/fpm_forward_perf.parquet"
    path = root / MANIFEST
    document = json.loads(path.read_text())
    table = pq.read_table(library)
    assert set(NATIVE_REQUIRED_COLUMNS) <= set(table.column_names)
    pq.write_table(table.drop(["cell_id"]), library)
    sidecar = root / ENTRY / "fpm/fpm_forward_perf.metadata.json"
    changed = json.loads(sidecar.read_text())
    changed["parquet_sha256"] = _sha(library.read_bytes())
    sidecar.write_text(json.dumps(changed))
    for artifact in document["artifacts"]:
        for target in (library, sidecar):
            if artifact["path"].endswith(target.name):
                artifact.update(sha256=_sha(target.read_bytes()), bytes=target.stat().st_size)
        if artifact["path"].endswith(library.name):
            artifact["rows"] = table.num_rows
    path.write_text(json.dumps(document))
    findings = check_fpm_entry(root, MANIFEST)
    assert [f.code for f in findings] == ["library-columns"] and "cell_id" in findings[0].message


def test_an_unreadable_artifact_is_a_finding_not_a_crash(root, profile, monkeypatch):
    make_entry(root, profile)

    def deny(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr("aisim_core.data.fpm_checks._verify", deny)
    assert {f.code for f in check_fpm_entry(root, MANIFEST)} == {"hash-mismatch"}


def test_entries_differing_only_in_dp_attention_are_distinct(root, profile):
    make_entry(root, profile)
    manifest = Manifest.model_validate_json((root / MANIFEST).read_bytes())
    on = manifest.model_copy(
        update={"identity": manifest.identity.model_copy(update={"dp_attention": {"aggregated": True}})}
    )
    off = manifest.model_copy(
        update={"identity": manifest.identity.model_copy(update={"dp_attention": {"aggregated": False}})}
    )
    assert check_fpm_identities([("a", on), ("b", off)]) == []
