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
from jsonschema import Draft202012Validator, ValidationError

from aisim_core.data import Entry, Identity, Manifest, configuration_id
from aisim_core.data.fetch import Cache, fetch, plan
from aisim_core.data.fpm_checks import check_fpm_dataset, check_fpm_entry, check_fpm_identities
from aisim_core.data.fpm_library import (
    EVALUATION_ROLES,
    NATIVE_REQUIRED_COLUMNS,
    FpmContractError,
    FpmLibraryMetadata,
    FpmMeasurementManifest,
    FpmResourcesPendingError,
    check_effective_measurement_policy,
    complete_fpm_profile,
    missing_resource_fields,
    read_library_metadata,
    recover_reconstructed_metadata,
    resolve_resources,
    resource_metadata_for_deployment,
    stage_fpm_library,
    with_estimator_path,
)
from aisim_core.data.loaders import FpmLoader, LoaderError, _resolve_library_resources
from aisim_core.data.schemas import SCHEMA_MODELS, check_snapshots, render_snapshot
from aisim_core.sdk import RustForwardPassPerfModel, engine, memory
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


def _measurement_document(size=1):
    return {
        "manifest_version": 5,
        "measurement_artifact_id": "synthetic",
        "measurement_protocol_id": "synthetic",
        "model_id": "synthetic/model",
        "framework": "vllm",
        "framework_version": "0.25.1",
        "files": [
            {
                "measurement_file_id": "truth",
                "role": "truth",
                "path": "measurements/truth.parquet",
                "format": "parquet",
                "sha256": "0" * 64,
                "bytes": size,
            }
        ],
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
    note = f"{ENTRY}/publisher-note.txt"
    (root / note).write_text("An optional publisher note, outside the prediction tier.")
    document = json.loads((root / MANIFEST).read_text())
    document["artifacts"].append(_artifact(root, note, "publisher-note", "text", "unversioned"))
    (root / MANIFEST).write_text(json.dumps(document))
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
    assert set(EVALUATION_ROLES) <= set(selection.library_exclude_roles)
    assert selection.parallelism == (("moe_tp", 4), ("tp", 4))
    default = plan(selection, repo=REPO, revision=SHA, cache=Cache(cache_root), offline=True)
    everything = plan(
        selection.__class__(**{**selection.__dict__, "exclude_roles": (), "library_exclude_roles": ()}),
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
    assert note in {f.path for f in everything.files}
    (commit / note).unlink()
    fetched = fetch(selection, repo=REPO, revision=SHA, cache=Cache(cache_root), offline=True)
    assert note not in fetched.paths


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


@pytest.mark.parametrize("cp", [1, 2])
def test_context_parallelism_manifest_selection_only(root, profile, tmp_path, cp):
    # This checks catalog identity, not prediction support: the fixture's table and profile are CP1.
    make_entry(root, profile)
    document = json.loads((root / MANIFEST).read_text())
    dims = {"tp": 4, "moe_tp": 4, **({"cp": cp} if cp != 1 else {})}
    document["identity"]["parallelism"] = {"aggregated": dims}
    document["configuration_id"] = configuration_id("fpm", "execution-profile", Identity(**document["identity"]))
    (root / MANIFEST).write_text(json.dumps(document))
    entry = Entry(
        configuration_id=document["configuration_id"],
        purpose=document["purpose"],
        identity=document["identity"],
        manifest=_artifact(root, MANIFEST, "manifest", "json", "aisim-artifact-manifest/v1"),
    )
    config = {**CONFIG, "cp_size": cp}
    loader = FpmLoader()
    assert loader.select(config).parallelism == tuple(sorted(dims.items()))
    assert loader.select(config).matches(entry)
    different = {**config, "cp_size": 3}
    assert not loader.select(different).matches(entry)
    data = load(root, config, tmp_path)
    assert data.config["cp_size"] == cp
    with pytest.raises(LoaderError, match="parallelism"):
        load(root, different, tmp_path)


def test_context_parallelism_profile_rejects_cp2_prediction(root, profile, tmp_path):
    # The direct FPM profile contract supports CP1 only. Catalog selection can still describe CP2.
    make_entry(root, profile)
    config = load(root, user_config(profile), tmp_path).config
    config["fpm_profile"]["deployments"][0]["cp"] = 2
    config["cp_size"] = 2
    # Native admission must reject the unsupported profile rather than select a CP1 prediction cell.
    with pytest.raises(ValueError, match="cp"):
        RustForwardPassPerfModel.best_available(config)


@pytest.mark.parametrize("as_model", [False, True])
def test_library_loader_names_a_non_mapping_profile(root, profile, tmp_path, as_model):
    make_entry(root, profile)
    invalid = load_fpm_profile(profile) if as_model else json.dumps(profile)
    config = {**CONFIG, "fpm_profile": invalid}
    with pytest.raises(LoaderError, match="fpm_profile must be a mapping"):
        load(root, config, tmp_path)
    assert not (tmp_path / "staged").exists()


@pytest.mark.parametrize("selected_ready", [False, True])
def test_readiness_is_available_for_each_role_deployment(root, profile, tmp_path, selected_ready):
    make_entry(root, profile)  # no metadata resources to fill the pending deployment
    config = user_config(profile)
    first = config["fpm_profile"]["deployments"][0]
    first["worker_type"] = "prefill"
    second = copy.deepcopy(first)
    second["worker_type"] = "decode"
    config["fpm_profile"]["deployments"].append(second)
    for index, deployment in enumerate((first, second)):
        deployment.pop("resources", None)
        if (index == 1) == selected_ready:
            deployment["resources"] = {**RESOURCES, "runtime_memory": RUNTIME_MEMORY}
    config["worker_type"] = "decode"  # the consumer selects deployment one, not zero
    data = load(root, config, tmp_path)
    pending = tuple(missing_resource_fields(None))
    assert data.memory_pending_by_deployment == ((pending, ()) if selected_ready else ((), pending))
    assert data.memory_pending == data.memory_pending_by_deployment[0]
    assert data.resources == data.resources_by_deployment[0]
    assert data.resource_sources == (("pending", "user") if selected_ready else ("user", "pending"))
    selected = data.resources_by_deployment[1]
    assert (selected.resources is not None) == selected_ready
    normalized = RustForwardPassPerfModel.normalize_config(data.config)
    if selected_ready:
        _require_forward_pass_profile_memory(normalized)
    else:
        with pytest.raises(ValueError, match="resources are pending"):
            _require_forward_pass_profile_memory(normalized)


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
        load(root, user_config(profile), tmp_path, require_resources=True)
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
    assert "cache sizing and replay" in str(error.value)
    assert "Plain autoregressive timing queries do not require resources" in str(error.value)
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


def test_draft_profile_keeps_pending_resources_until_required(root, profile, tmp_path):
    make_entry(root, profile)
    config = user_config(profile)
    data = load(root, config, tmp_path)
    assert data.profile is None
    assert data.resource_sources == ("pending",)
    assert data.resources.source == "pending" and data.resources.resources is None
    assert data.config["fpm_profile"] == config["fpm_profile"]
    assert "cache_layout" in data.memory_pending
    assert Path(data.config["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"]).is_file()
    # The synthetic table has a measured-shaped prefill cell at one request/one token with latency 2 ms.
    model = RustForwardPassPerfModel.best_available(data.config)
    try:
        assert model.estimate_forward_pass_time_ms(
            [{"scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 1}}]
        ) == pytest.approx(2.0)
        saved = model.diagnostics()["provenance"]["config"]
    finally:
        model.close()
    assert "resources" not in saved["fpm_profile"]["deployments"][0]
    reloaded = RustForwardPassPerfModel.best_available(json.loads(json.dumps(saved)))
    try:
        assert reloaded.estimate_forward_pass_time_ms(
            [{"scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 1}}]
        ) == pytest.approx(2.0)
    finally:
        reloaded.close()
    with pytest.raises(ValueError, match="cache_layout.*max_num_tokens.*max_batch_size.*provenance"):
        _require_forward_pass_profile_memory(saved)
    with pytest.raises(ValueError, match="resources"):
        load_fpm_profile(saved["fpm_profile"])  # Resource-dependent public profiles remain strict.
    with pytest.raises(ValueError, match="cache_layout.*max_num_tokens.*max_batch_size.*provenance"):
        memory.estimate_kv_cache(
            CONFIG["model"],
            CONFIG["system"],
            CONFIG["backend"],
            CONFIG["backend_version"],
            tp_size=4,
            moe_tp_size=4,
            fpm_profile=saved["fpm_profile"],
            memory_fraction_kind="of_total",
            memory_fraction_value=0.9,
            max_num_tokens=8192,
            max_batch_size=256,
        )
    with pytest.raises(ValueError, match="cache_layout.*max_num_tokens.*max_batch_size.*provenance"):
        RustForwardPassPerfModel.estimate_cache_budget(
            saved,
            {
                "total_gpu_capacity_bytes": 10**9,
                "memory_fraction_kind": "of_total",
                "memory_fraction_value": 0.9,
                "max_num_tokens": 8192,
                "max_batch_size": 256,
            },
        )
    with pytest.raises(FpmResourcesPendingError, match="cache_layout"):
        load(root, config, tmp_path, require_resources=True)


@pytest.mark.parametrize("tp", [None, 4, 8])
def test_resources_without_profile_require_the_selected_topology(root, profile, tmp_path, tp):
    make_entry(root, profile, resources=RESOURCES)
    manifest_path = root / MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"]["parallelism"] = None
    identity = Identity.model_validate(manifest["identity"])
    manifest["configuration_id"] = configuration_id("fpm", "execution-profile", identity)
    manifest_path.write_text(json.dumps(manifest))
    config = {**CONFIG, "tp": tp, "moe_tp_size": tp} if tp is not None else None
    data = load(root, config, tmp_path)
    if tp == 4:
        assert data.resources.source == "metadata"
        assert load(root, config, tmp_path, require_resources=True).resources.source == "metadata"
    else:
        assert data.resources.source == "pending" and data.resources.resources is None
        with pytest.raises(FpmResourcesPendingError, match="cache_layout"):
            load(root, config, tmp_path, require_resources=True)


@pytest.mark.parametrize(
    ("field", "matching", "different"),
    [
        ("kv_cache_dtype", "fp8", "bfloat16"),
        ("gemm_quant_mode", "fp8_block", "bfloat16"),
        ("moe_quant_mode", "fp8_block", "bfloat16"),
        ("fmha_quant_mode", "bfloat16", "fp8"),
        ("comm_quant_mode", "half", "bfloat16"),
        ("attention_backend", "auto", "flashinfer"),
        ("moe_backend", "auto", "other"),
        ("enable_wideep", False, True),
        ("enable_eplb", False, True),
        ("model_path", "MiniMaxAI/MiniMax-M2.7", "other/model"),
        ("model_revision", "profile-test-fixture-v1", "other-revision"),
    ],
)
def test_resources_only_fill_profiles_matching_selector(root, profile, tmp_path, field, matching, different):
    # Change the sidecar selector, keeping the requested profile valid throughout.
    selector = {"tp": 4, "dp": 1, "moe_tp": 4, field: matching}
    make_entry(root, profile, resources=RESOURCES, sidecar_extra={"configuration_selector": selector})
    assert load(root, user_config(profile), tmp_path).resource_sources == ("metadata",)
    selector[field] = different
    make_entry(root, profile, resources=RESOURCES, sidecar_extra={"configuration_selector": selector})
    pending = load(root, user_config(profile), tmp_path)
    assert pending.resource_sources == ("pending",)
    with pytest.raises(FpmResourcesPendingError):
        load(root, user_config(profile), tmp_path, require_resources=True)
    user = {**RESOURCES, "provenance": "Explicit user resource override."}
    assert load(root, user_config(profile, **user), tmp_path, require_resources=True).resource_sources == ("user",)


@pytest.mark.parametrize(
    ("field", "matching", "different"),
    [("system", "h200_sxm", "b200_sxm"), ("backend", "vllm", "sglang"), ("backend_version", "0.25.1", "0.26.0")],
)
def test_selector_runtime_identity_must_agree_with_the_library(root, profile, tmp_path, field, matching, different):
    selector = {"tp": 4, "dp": 1, "moe_tp": 4, field: matching}
    make_entry(root, profile, resources=RESOURCES, sidecar_extra={"configuration_selector": selector})
    assert load(root, user_config(profile), tmp_path).resource_sources == ("metadata",)
    selector[field] = different
    make_entry(root, profile, resources=RESOURCES, sidecar_extra={"configuration_selector": selector})
    for config in (user_config(profile), dict(CONFIG)):  # with and without a requested profile
        data = load(root, config, tmp_path / "again")
        assert set(data.resource_sources or ("pending",)) == {"pending"}
        assert data.resources.resources is None and data.memory_pending
        with pytest.raises(FpmResourcesPendingError):
            load(root, config, tmp_path / "required", require_resources=True)


@pytest.mark.parametrize("dcp", [1, 2])
def test_dcp_sidecar_resources_do_not_fill_an_unrecorded_profile(root, profile, tmp_path, dcp):
    make_entry(
        root,
        profile,
        resources=RESOURCES,
        sidecar_extra={"configuration_selector": {"tp": 4, "moe_tp": 4, "dcp": dcp}},
    )
    config = user_config(profile)
    assert "dcp" not in config["fpm_profile"]["deployments"][0]
    data = load(root, config, tmp_path)
    assert data.resource_sources == ("pending",)
    assert data.resources.resources is None
    assert "resources" not in data.config["fpm_profile"]["deployments"][0]
    with pytest.raises(FpmResourcesPendingError):
        load(root, config, tmp_path, require_resources=True)
    override = {**RESOURCES, "provenance": "Explicit user resource override."}
    assert load(root, user_config(profile, **override), tmp_path, require_resources=True).resource_sources == ("user",)


@pytest.mark.parametrize("requested_dcp", [None, 1, 2])
def test_dcp_sidecar_resources_require_the_recorded_dcp(requested_dcp):
    # Exercise the resource boundary with canonical config normalization, without pairing a DCP request
    # with the unrecorded-DCP catalog/table used by the TP-only integration fixture above.
    config = RustForwardPassPerfModel.normalize_config(
        {**CONFIG, **({"dcp": requested_dcp} if requested_dcp is not None else {})}
    )
    metadata = {"resources": RESOURCES, "configuration_selector": {"dcp": 2}}
    selected = resource_metadata_for_deployment(config, config, metadata)
    assert selected == (metadata if requested_dcp == 2 else None)


@pytest.mark.parametrize("selector_role", ["prefill", "decode", "aggregated"])
@pytest.mark.parametrize("deployment_role", ["omitted", None, "prefill", "decode", "aggregated"])
def test_role_specific_sidecar_resources_require_an_explicit_deployment_role(
    root, profile, tmp_path, selector_role, deployment_role
):
    make_entry(
        root,
        profile,
        resources={**RESOURCES, "runtime_memory": RUNTIME_MEMORY},
        sidecar_extra={"configuration_selector": {"tp": 4, "moe_tp": 4, "worker_type": selector_role}},
    )
    config = user_config(profile)
    if deployment_role != "omitted":
        config["fpm_profile"]["deployments"][0]["worker_type"] = deployment_role
    # Shared deployments can be selected for another role; that must not promote role-specific evidence.
    shared_role = "decode" if selector_role == "prefill" else "prefill"
    config["worker_type"] = deployment_role if deployment_role in ("prefill", "decode", "aggregated") else shared_role
    original = copy.deepcopy(config)
    data = load(root, config, tmp_path)
    matches = deployment_role == selector_role
    assert data.resource_sources == (("metadata",) if matches else ("pending",))
    assert config == original
    saved = RustForwardPassPerfModel.normalize_config(json.loads(json.dumps(data.config)))
    memory_args = dict(
        tp_size=4,
        moe_tp_size=4,
        worker_type=config["worker_type"],
        fpm_profile=saved["fpm_profile"],
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        max_num_tokens=8192,
        max_batch_size=256,
        gpu_memory_capacity_bytes_override=10**10,
    )
    if matches:
        assert data.memory_pending == ()
        assert load(root, config, tmp_path, require_resources=True).resource_sources == ("metadata",)
        _require_forward_pass_profile_memory(saved)
        estimate = memory.estimate_kv_cache(
            CONFIG["model"], CONFIG["system"], CONFIG["backend"], CONFIG["backend_version"], **memory_args
        )
        assert estimate["total_kv_size_bytes"] == RUNTIME_MEMORY["kv_cache_bytes"]
    else:
        assert "resources" not in saved["fpm_profile"]["deployments"][0]
        with pytest.raises(FpmResourcesPendingError, match="cache_layout"):
            load(root, config, tmp_path, require_resources=True)
        with pytest.raises(ValueError, match="FPM resources are pending.*cache_layout"):
            _require_forward_pass_profile_memory(saved)
        with pytest.raises(ValueError, match="FPM resources are pending.*cache_layout"):
            memory.estimate_kv_cache(
                CONFIG["model"], CONFIG["system"], CONFIG["backend"], CONFIG["backend_version"], **memory_args
            )


@pytest.mark.parametrize("selector_role", ["prefill", "decode", "aggregated"])
def test_explicit_shared_resources_override_role_specific_sidecar(root, profile, tmp_path, selector_role):
    make_entry(
        root,
        profile,
        resources=RESOURCES,
        sidecar_extra={"configuration_selector": {"worker_type": selector_role}},
    )
    user = {**RESOURCES, "max_num_tokens": 4096, "provenance": "Explicit shared-resource fixture."}
    data = load(root, user_config(profile, **user), tmp_path, require_resources=True)
    assert data.resource_sources == ("user",)
    assert data.profile.deployments[0].worker_type is None
    assert data.profile.deployments[0].resources.max_num_tokens == user["max_num_tokens"]
    assert data.profile.deployments[0].resources.provenance == user["provenance"]


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


@pytest.mark.parametrize("invalid", [[1], 1, "invalid"])
def test_non_object_user_resources_report_the_contract_and_deployment(profile, invalid):
    profile["deployments"][0]["resources"] = invalid
    with pytest.raises(FpmContractError, match=r"invalid resources from configuration for deployment 0 .*tp=4"):
        complete_fpm_profile(profile, None)


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


@pytest.mark.parametrize("changed", ["parquet", "metadata"])
def test_staging_verifies_supplied_hashes_before_build_and_reuse(root, profile, tmp_path, changed):
    make_entry(root, profile, resources=RESOURCES)
    source = root / ENTRY / "fpm"
    parquet, sidecar = source / "fpm_forward_perf.parquet", source / "fpm_forward_perf.metadata.json"
    kwargs = dict(system="h200_sxm", backend="vllm", backend_version="0.25.1", output_dir=tmp_path / "out")
    pins = {"parquet_sha256": _sha(parquet.read_bytes()), "metadata_sha256": _sha(sidecar.read_bytes())}
    with pytest.raises(FpmContractError, match="supplied hash"):
        stage_fpm_library(parquet, sidecar, **kwargs, **{**pins, f"{changed}_sha256": "0" * 64})
    assert not kwargs["output_dir"].exists()
    first = stage_fpm_library(parquet, sidecar, **kwargs, **pins)
    marker = (first.systems_root / ".staged.json").read_bytes()
    altered = parquet if changed == "parquet" else sidecar
    altered.write_bytes(altered.read_bytes() + b" ")
    with pytest.raises(FpmContractError, match="supplied hash"):
        stage_fpm_library(parquet, sidecar, **kwargs, **pins)
    assert (first.systems_root / ".staged.json").read_bytes() == marker


def test_staging_reuse_tracks_the_copied_system_definition(root, profile, tmp_path):
    from aisim_core.data.fpm_library import _stageable_roots

    make_entry(root, profile)
    source = root / ENTRY / "fpm"
    systems = tmp_path / "systems"
    systems.mkdir()
    definition = systems / "h200_sxm.yaml"
    definition.write_bytes((_stageable_roots(None, None)[0] / definition.name).read_bytes())
    args = (source / "fpm_forward_perf.parquet", source / "fpm_forward_perf.metadata.json")
    kwargs = dict(
        system="h200_sxm",
        backend="vllm",
        backend_version="0.25.1",
        output_dir=tmp_path / "out",
        systems_paths=[systems],
    )
    staged = stage_fpm_library(*args, **kwargs)
    marker = staged.systems_root / ".staged.json"
    inode = marker.stat().st_ino
    stage_fpm_library(*args, **kwargs)
    assert marker.stat().st_ino == inode
    definition.write_text(definition.read_text() + "\n# Updated system definition\n")
    stage_fpm_library(*args, **kwargs)
    assert marker.stat().st_ino != inode
    assert (staged.systems_root / definition.name).read_bytes() == definition.read_bytes()


def test_staging_digest_describes_the_yaml_copied_during_a_source_edit(root, profile, tmp_path, monkeypatch):
    make_entry(root, profile)
    systems = tmp_path / "systems"
    systems.mkdir()
    definition = systems / "h200_sxm.yaml"
    original = "data_dir: data/custom\n# A\n"
    changed = "data_dir: data/custom\n# B\n"
    definition.write_text(original)
    copy2 = shutil.copy2

    def edit_then_copy(source, target):
        if Path(source) == definition:
            definition.write_text(changed)
        return copy2(source, target)

    kwargs = {**KWARGS, "systems_paths": [systems], "output_dir": tmp_path / "out"}
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copy2", edit_then_copy)
        staged = stage_fpm_library(*_stage_args(root), **kwargs)
    assert (staged.systems_root / definition.name).read_text() == changed
    marker = json.loads((staged.systems_root / ".staged.json").read_text())
    assert marker["system_definition"]["sha256"] == _sha(changed.encode())
    definition.write_text(original)
    again = stage_fpm_library(*_stage_args(root), **kwargs)
    assert (again.systems_root / definition.name).read_text() == original


def test_overlay_rejects_targets_removed_by_replacement_and_preserves_old_staging(root, profile, tmp_path):
    make_entry(root, profile)
    systems = tmp_path / "systems"
    version = systems / "data/custom/vllm/0.25.1"
    version.mkdir(parents=True)
    (systems / "h200_sxm.yaml").write_text("data_dir: data/custom\n")
    kwargs = {**KWARGS, "systems_paths": [systems], "output_dir": tmp_path / "out"}
    staged = stage_fpm_library(*_stage_args(root), **kwargs)
    marker = (staged.systems_root / ".staged.json").read_bytes()
    owned = staged.parquet_path.with_name("gemm.parquet")
    owned.write_bytes(b"previous staging op data")
    link = version / "gemm.parquet"
    link.symlink_to(owned)
    with pytest.raises(FpmContractError, match="overlay source .* resolves inside output_dir"):
        stage_fpm_library(*_stage_args(root), **kwargs)
    assert (staged.systems_root / ".staged.json").read_bytes() == marker
    assert owned.read_bytes() == b"previous staging op data"
    assert staged.parquet_path.is_file() and staged.metadata_path.is_file()
    assert not list(tmp_path.glob(".out.staging-*"))
    external = tmp_path / "external.parquet"
    external.write_bytes(b"external op data")
    link.unlink()
    link.symlink_to(external)
    again = stage_fpm_library(*_stage_args(root), **kwargs)
    assert again.parquet_path.with_name("gemm.parquet").read_bytes() == b"external op data"


def test_failed_replacement_restores_the_previous_staged_root(root, profile, tmp_path, monkeypatch):
    import os

    make_entry(root, profile)
    source = root / ENTRY / "fpm"
    args = (source / "fpm_forward_perf.parquet", source / "fpm_forward_perf.metadata.json")
    kwargs = dict(system="h200_sxm", backend="vllm", backend_version="0.25.1", output_dir=tmp_path / "out")
    first = stage_fpm_library(*args, **kwargs)
    before = {str(p.relative_to(first.systems_root)): p.read_bytes() for p in (first.parquet_path, first.metadata_path)}
    marker = (first.systems_root / ".staged.json").read_bytes()
    # Force a rebuild without changing the linked source files used by the earlier configuration.
    systems = tmp_path / "other-systems"
    systems.mkdir()
    replace = os.replace

    def fail_install(src, dst):
        if ".staging-" in str(src):
            raise PermissionError("injected final rename failure")
        return replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_install)
    with pytest.raises(FpmContractError, match="final rename failure"):
        stage_fpm_library(*args, **kwargs, systems_paths=[systems])
    assert (first.systems_root / ".staged.json").read_bytes() == marker
    assert all((first.systems_root / relative).read_bytes() == raw for relative, raw in before.items())
    assert not list(tmp_path.glob(".out.staging-*"))


def test_system_overlay_fills_missing_operation_files_from_fallback_roots(root, profile, tmp_path, monkeypatch):
    from aisim_core.sdk import perf_database

    make_entry(root, profile)
    source = root / ENTRY / "fpm"
    first, fallback = tmp_path / "first", tmp_path / "fallback"
    first.mkdir()
    (first / "h200_sxm.yaml").write_text("data_dir: data/custom\n")
    for systems, files in (
        (first, {"gemm.parquet": b"first"}),
        (fallback, {"gemm.parquet": b"later", "moe.parquet": b"fallback"}),
    ):
        data = systems / "data/custom/vllm/0.25.1"
        data.mkdir(parents=True)
        for name, raw in files.items():
            (data / name).write_bytes(raw)
    monkeypatch.setattr(perf_database, "get_systems_paths", lambda: [str(fallback)])
    staged = stage_fpm_library(
        source / "fpm_forward_perf.parquet",
        source / "fpm_forward_perf.metadata.json",
        system="h200_sxm",
        backend="vllm",
        backend_version="0.25.1",
        systems_paths=[first],
    )
    try:
        target = staged.systems_root / "data/custom/vllm/0.25.1"
        assert (target / "gemm.parquet").read_bytes() == b"first"
        assert (target / "moe.parquet").read_bytes() == b"fallback"
        assert (staged.systems_root / "h200_sxm.yaml").read_bytes() == (first / "h200_sxm.yaml").read_bytes()
    finally:
        staged.close()


def test_system_overlay_rejects_a_dangling_source_before_it_hides_a_fallback(root, profile, tmp_path, monkeypatch):
    from aisim_core.sdk import perf_database

    make_entry(root, profile)
    source = root / ENTRY / "fpm"
    first, fallback = tmp_path / "first", tmp_path / "fallback"
    first.mkdir()
    (first / "h200_sxm.yaml").write_text("data_dir: data/custom\n")
    for systems in (first, fallback):
        (systems / "data/custom/vllm/0.25.1").mkdir(parents=True)
    broken = first / "data/custom/vllm/0.25.1/gemm.parquet"
    broken.symlink_to(tmp_path / "missing.parquet")
    (fallback / "data/custom/vllm/0.25.1/gemm.parquet").write_bytes(b"fallback")
    monkeypatch.setattr(perf_database, "get_systems_paths", lambda: [str(fallback)])
    destination = tmp_path / "staged"
    with pytest.raises(FpmContractError, match="gemm.parquet"):
        stage_fpm_library(
            source / "fpm_forward_perf.parquet",
            source / "fpm_forward_perf.metadata.json",
            system="h200_sxm",
            backend="vllm",
            backend_version="0.25.1",
            systems_paths=[first],
            output_dir=destination,
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".staged.staging-*"))


@pytest.mark.parametrize("marker", [None, "not json", "{}", "[]", '{"system": "h200_sxm"}'])
def test_staging_preserves_an_unrecognized_nonempty_destination(root, profile, tmp_path, marker):
    make_entry(root, profile)
    destination = tmp_path / "out"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("unrelated data")
    if marker is not None:
        (destination / ".staged.json").write_text(marker)
    source = root / ENTRY / "fpm"
    with pytest.raises(FpmContractError, match="staging marker"):
        stage_fpm_library(
            source / "fpm_forward_perf.parquet",
            source / "fpm_forward_perf.metadata.json",
            system="h200_sxm",
            backend="vllm",
            backend_version="0.25.1",
            output_dir=destination,
        )
    assert sentinel.read_text() == "unrelated data"


@pytest.mark.parametrize("explicit_roots", [True, False])
def test_ensure_library_isolates_effective_systems_roots(root, profile, tmp_path, monkeypatch, explicit_roots):
    from aisim_core.data.ensure import DataPin, ensure_data
    from aisim_core.data.fpm_library import _stageable_roots
    from aisim_core.sdk import perf_database

    make_entry(root, profile, resources=RESOURCES)
    manifest = Manifest.model_validate_json((root / MANIFEST).read_bytes())
    (root / "catalog/fpm").mkdir(parents=True)
    (root / "catalog/fpm/index.json").write_text(
        json.dumps(
            {
                "schema_version": "aisim-category-index/v1",
                "category": "fpm",
                "entries": [
                    {
                        "configuration_id": manifest.configuration_id,
                        "purpose": manifest.purpose,
                        "identity": manifest.identity.model_dump(),
                        "manifest": _artifact(root, MANIFEST, "manifest", "json", "aisim-artifact-manifest/v1"),
                    }
                ],
            }
        )
    )
    (root / "catalog/dataset.json").write_text(
        json.dumps(
            {
                "schema_version": "aisim-dataset-index/v1",
                "dataset_id": REPO,
                "categories": {
                    "fpm": {
                        "root": "data/fpm",
                        "layout": "category",
                        "index": _artifact(root, "catalog/fpm/index.json", "index", "json", "aisim-category-index/v1"),
                    }
                },
            }
        )
    )
    roots = [tmp_path / name for name in ("systems-a", "systems-b")]
    system_yaml = (_stageable_roots(["default"], None)[0] / "h200_sxm.yaml").read_text()
    for index, systems in enumerate(roots):
        systems.mkdir()
        (systems / "h200_sxm.yaml").write_text(system_yaml + f"\n# fixture systems root: {index}\n")
        data = systems / "data/h200_sxm/vllm/0.25.1"
        data.mkdir(parents=True)
        (data / "operation.txt").write_text(str(index))
    pin = DataPin(local_root=root, categories=("fpm",), cache_dir=tmp_path / "cache")
    config = user_config(profile)
    selected = []
    if not explicit_roots:
        empty = tmp_path / "empty-user-root"
        empty.mkdir()
        config["systems_paths"] = [str(empty)]
    for systems in roots:
        if explicit_roots:
            config["systems_paths"] = [str(systems)]
        else:
            monkeypatch.setattr(perf_database, "get_systems_paths", lambda: [str(systems)])
        result = ensure_data(config, pin)
        staged = Path(result.config["systems_paths"][0])
        inode = (staged / ".staged.json").stat().st_ino
        again = ensure_data(config, pin)
        assert again.config == result.config
        assert (staged / ".staged.json").stat().st_ino == inode
        # A saved configuration returned by an earlier call lists its own staging root; it must not key a new one.
        reloaded = ensure_data(result.config, pin)
        assert Path(reloaded.config["systems_paths"][0]) == staged
        assert (staged / ".staged.json").stat().st_ino == inode
        if explicit_roots:
            aliased = ensure_data({**config, "systems_paths": [str(systems) + "/.", str(systems)]}, pin)
            assert Path(aliased.config["systems_paths"][0]) == staged
            assert (staged / ".staged.json").stat().st_ino == inode
        selected.append(result.config)
    assert selected[0]["systems_paths"][0] != selected[1]["systems_paths"][0]
    for index, returned_config in enumerate(selected):
        staged = Path(returned_config["systems_paths"][0])
        assert (staged / "h200_sxm.yaml").read_text().endswith(f"# fixture systems root: {index}\n")
        assert (staged / "data/h200_sxm/vllm/0.25.1/operation.txt").read_text() == str(index)
        model = RustForwardPassPerfModel.best_available(returned_config)
        try:
            assert model.estimate_forward_pass_time_ms(
                {"scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 1}}
            ) == pytest.approx(2.0)
        finally:
            model.close()


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
    assert _stageable_roots(["~/custom", "/x"], tmp_path / "custom") == [Path("/x"), *_stageable_roots(None, None)]


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


@pytest.mark.parametrize("invalid", [[], "", 0, False, ["bad"], "bad"])
def test_estimator_path_rejects_non_mapping_interpolation(invalid):
    config = {"estimator_config": {"fpm_interpolation": invalid}}
    original = copy.deepcopy(config)
    with pytest.raises(FpmContractError, match=r"estimator_config.fpm_interpolation must be a mapping"):
        with_estimator_path(config, Path("/staged/a.parquet"))
    assert config == original


@pytest.mark.parametrize("interpolation", [None, {}, {"fpm_parquet_path": "/chosen.parquet"}])
def test_estimator_path_defaults_only_absent_or_null_and_keeps_explicit_paths(interpolation):
    config = {"estimator_config": {"fpm_interpolation": interpolation}}
    with_estimator_path(config, Path("/staged/a.parquet"))
    assert config["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"] == (
        (interpolation or {}).get("fpm_parquet_path", "/staged/a.parquet")
    )
    absent = {}
    with_estimator_path(absent, Path("/staged/a.parquet"))
    assert absent["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"] == "/staged/a.parquet"


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
    outside = tmp_path / "outside.yaml"
    outside.write_text("data_dir: data/outside\n")
    for system in (str(outside.with_suffix("")), "../outside"):
        with pytest.raises(FpmContractError, match="system must be one safe path component"):
            stage_fpm_library(*files, system=system, backend="vllm", backend_version="1")


def test_library_schema_versions(tmp_path):
    def sidecar(version):
        path = tmp_path / f"v{version}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_name": "aic_fpm_forward_perf",
                    "schema_version": version,
                    "row_count": 1,
                    "parquet_sha256": "0" * 64,
                }
            )
        )
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


@pytest.mark.parametrize("source_override", [False, True])
def test_recovered_staged_pair_answers_through_native_best_available(root, profile, tmp_path, source_override):
    make_entry(root, profile)
    parquet, sidecar = _stage_args(root)
    rows = pq.read_table(parquet).to_pylist()
    for row in rows:
        row.update(measurement_policy="dynamo_native_single_sample_v1", measurement_repeats=1, warmup_repeats=0)
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    metadata = json.loads(sidecar.read_text())
    metadata.update(
        metadata_provenance="reconstructed_from_parquet",
        parquet_sha256=_sha(parquet.read_bytes()),
        configuration_selector={name: rows[0][name] for name in ("model_path", "system", "backend", "backend_version")},
    )
    for name in ("coordinate_system", "measurement_policy", "system", "backend", "backend_version"):
        metadata.pop(name)
    sidecar.write_text(json.dumps(metadata))
    original = sidecar.read_bytes()
    _refresh_prediction_manifest(root)
    config = user_config(profile)
    if source_override:
        config["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"] = str(parquet)
    data = load(root, config, tmp_path)
    staged = data._staged
    assert not staged.metadata_path.is_symlink()
    assert staged.diagnostics["fpm_metadata_recovery"]["recovered_fields"]["coordinate_system"] == (
        "iteration_totals_balanced_v1"
    )
    assert data.config["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"] == str(staged.parquet_path)
    model = RustForwardPassPerfModel.best_available(data.config)
    try:
        for query, expected in (
            ({"num_prefill_requests": 1, "sum_prefill_tokens": 1}, 2.0),
            ({"num_decode_requests": 1, "sum_decode_kv_tokens": 1}, 3.0),
        ):
            assert model.estimate_forward_pass_time_ms([{"scheduled_requests": query}]) == pytest.approx(expected)
    finally:
        model.close()
    assert sidecar.read_bytes() == original


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


def test_restaging_recovered_sidecar_keeps_it_readable(tmp_path):
    metadata, parquet, _ = reconstructed(tmp_path)
    sidecar = tmp_path / "lib.metadata.json"
    sidecar.write_text(json.dumps(metadata))
    kwargs = dict(system="h200_sxm", backend="vllm", backend_version="0.25.1", output_dir=tmp_path / "out")
    staged = stage_fpm_library(parquet, sidecar, **kwargs)
    expected = staged.metadata_path.read_bytes()
    restaged = stage_fpm_library(staged.parquet_path, staged.metadata_path, **kwargs)
    assert not restaged.metadata_path.is_symlink()
    assert restaged.metadata_path.read_bytes() == expected
    assert restaged.parquet_path.resolve() == parquet.resolve()
    assert read_library_metadata(restaged.metadata_path) == staged.metadata
    assert json.loads(sidecar.read_text()) == metadata


def test_restaging_rejects_an_owned_parquet_before_replacement(tmp_path):
    metadata, parquet, _ = reconstructed(tmp_path)
    sidecar = tmp_path / "lib.metadata.json"
    sidecar.write_text(json.dumps(metadata))
    kwargs = dict(system="h200_sxm", backend="vllm", backend_version="0.25.1", output_dir=tmp_path / "out")
    staged = stage_fpm_library(parquet, sidecar, **kwargs)
    staged.parquet_path.unlink()
    shutil.copyfile(parquet, staged.parquet_path)
    expected = staged.metadata_path.read_bytes()
    with pytest.raises(FpmContractError, match="parquet input must resolve outside output_dir"):
        stage_fpm_library(staged.parquet_path, staged.metadata_path, **kwargs)
    assert staged.parquet_path.read_bytes() == parquet.read_bytes()
    assert staged.metadata_path.read_bytes() == expected


@pytest.mark.parametrize("invalid", [True, "1", 1.0])
def test_staging_requires_an_integer_row_count(tmp_path, invalid):
    parquet = tmp_path / "one.parquet"
    pq.write_table(pa.Table.from_pylist([RECONSTRUCTED_ROW]), parquet)
    sidecar = tmp_path / "one.metadata.json"
    metadata = {
        "schema_name": "aic_fpm_forward_perf",
        "schema_version": 6,
        "row_count": invalid,
        "parquet_sha256": _sha(parquet.read_bytes()),
        "coordinate_system": "iteration_totals_balanced_v1",
        **{key: RECONSTRUCTED_ROW[key] for key in ("system", "backend", "backend_version", "measurement_policy")},
    }
    sidecar.write_text(json.dumps(metadata))
    kwargs = dict(system="h200_sxm", backend="vllm", backend_version="0.25.1", output_dir=tmp_path / "out")
    with pytest.raises(FpmContractError, match="row_count"):
        stage_fpm_library(parquet, sidecar, **kwargs)
    sidecar.write_text(json.dumps({**metadata, "row_count": 1}))
    staged = stage_fpm_library(parquet, sidecar, **kwargs)
    assert type(staged.metadata["row_count"]) is int
    assert staged.metadata["row_count"] == 1


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
    ) == {"duplicate-role", "duplicate-artifact"}
    assert check(lambda d: d["artifacts"][1].update(artifact_id=d["artifacts"][0]["artifact_id"])) == {
        "duplicate-artifact"
    }
    assert check(lambda d: d.update(artifacts=[a for a in d["artifacts"] if a["role"] != "measurement-manifest"])) == {
        "missing-role"
    }
    assert check(lambda d: d["artifacts"][0].update(sha256="0" * 64)) == {"hash-mismatch"}
    assert check(lambda d: d["artifacts"][0].update(schema="aic_fpm_forward_perf/v7")) == {
        "library-schema",
        "hash-mismatch",  # an unstamped table is accepted only under the supported schema id
    }
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


def test_entry_checks_reject_mismatched_configuration_id(root, profile):
    make_entry(root, profile)
    path = root / MANIFEST
    document = json.loads(path.read_text())
    document["configuration_id"] = "cfg-" + "0" * 64
    path.write_text(json.dumps(document))
    assert [f.code for f in check_fpm_entry(root, MANIFEST)] == ["configuration-identity"]
    with pytest.raises(LoaderError, match="configuration identity hash mismatch"):
        FpmLoader().read_manifest(root, MANIFEST, roles=())


@pytest.mark.parametrize(("fmt", "rows"), [("json", None), ("json", 3), ("parquet", None)])
def test_entry_checks_require_parquet_and_rows(root, profile, fmt, rows):
    make_entry(root, profile)
    path = root / MANIFEST
    document = json.loads(path.read_text())
    document["artifacts"][0].update(format=fmt, rows=rows)
    sidecar = root / ENTRY / "fpm/fpm_forward_perf.metadata.json"
    metadata = json.loads(sidecar.read_text())
    metadata["row_count"] += 1
    sidecar.write_text(json.dumps(metadata))
    document["artifacts"][1].update(sha256=_sha(sidecar.read_bytes()), bytes=sidecar.stat().st_size)
    path.write_text(json.dumps(document))
    assert "library-declaration" in {f.code for f in check_fpm_entry(root, MANIFEST)}


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
    # A well-formed but tampered index must not hide all of its entries from validation.
    index["entries"] = []
    (root / "catalog/fpm/index.json").write_text(json.dumps(index))
    assert [f.code for f in check_fpm_dataset(root)] == ["invalid-catalog"]


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
    base = {
        "schema_name": "aic_fpm_forward_perf",
        "schema_version": 6,
        "row_count": 2,
        "parquet_sha256": "0" * 64,
        "future_key": {"a": 1},
    }
    assert FpmLibraryMetadata.model_validate(base).resources is None  # entries published before `resources`
    assert FpmLibraryMetadata.model_validate({**base, "resources": RESOURCES}).resources.max_num_tokens == 8192
    with pytest.raises(ValueError, match="max_num_tokens"):
        FpmLibraryMetadata.model_validate({**base, "resources": {**RESOURCES, "max_num_tokens": 0}})
    assert load_fpm_profile  # the profile class stays the single source for resources


def test_sidecar_version_is_constrained_in_model_and_packaged_snapshot():
    from aisim_core.data.schemas import PACKAGED_SNAPSHOT_DIR, snapshot_filename

    base = {"schema_name": "aic_fpm_forward_perf", "schema_version": 6, "row_count": 2, "parquet_sha256": "0" * 64}
    schema = json.loads((PACKAGED_SNAPSHOT_DIR / snapshot_filename("aic_fpm_forward_perf.metadata/v6")).read_text())
    validator = Draft202012Validator(schema)
    validator.validate(base)
    assert FpmLibraryMetadata.model_validate(base).resources is None
    for version in (5, 7, 8):
        unsupported = {**base, "schema_version": version}
        with pytest.raises(ValueError, match="schema_version"):
            FpmLibraryMetadata.model_validate(unsupported)
        with pytest.raises(ValidationError):
            validator.validate(unsupported)


_MEMORY_NAMES = ("weights_bytes", "activations_bytes", "runtime_overhead_bytes", "comm_overhead_bytes")
_NO_KV = {k: v for k, v in RESOURCES.items() if k != "kv_bytes_per_token"}
_STATE_GROUP_WITHOUT_BLOCKS = {
    "name": "state",
    "kind": "state",
    "num_layers": 1,
    "block_size_tokens": 16,
    "page_size_bytes": 1024,
}
_STATE_GROUP = {**_STATE_GROUP_WITHOUT_BLOCKS, "blocks_per_request": 1}


@pytest.mark.parametrize(
    "resources",
    [
        {k: v for k, v in RESOURCES.items() if k != "kv_bytes_per_token"},
        {**RESOURCES, "kv_bytes_per_token": None},
        {**_NO_KV, "cache_layout": "grouped", "cache_groups": []},
        {**_NO_KV, "cache_layout": "unmodeled"},
        {**_NO_KV, "cache_layout": "mamba_align", "runtime_memory": RUNTIME_MEMORY},
        {**RESOURCES, "runtime_memory": RUNTIME_MEMORY, "weights_bytes": 0},
        {
            **_NO_KV,
            "cache_layout": "grouped",
            "cache_groups": [{**_STATE_GROUP, "blocks_per_request": 0}],
        },
        {**_NO_KV, "cache_layout": "grouped", "cache_groups": [_STATE_GROUP_WITHOUT_BLOCKS]},
    ],
)
def test_resource_layout_requirements_agree_in_runtime_and_exported_schema(resources):
    from aisim_core.data.schemas import PACKAGED_SNAPSHOT_DIR, snapshot_filename

    schema = json.loads((PACKAGED_SNAPSHOT_DIR / snapshot_filename("aic_fpm_forward_perf.metadata/v6")).read_text())
    document = {
        "schema_name": "aic_fpm_forward_perf",
        "schema_version": 6,
        "row_count": 1,
        "parquet_sha256": "0" * 64,
        "resources": resources,
    }
    with pytest.raises(ValueError):
        FpmLibraryMetadata.model_validate(document)
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(document)


def test_resource_layout_schema_keeps_valid_and_absent_resources():
    schema = json.loads(render_snapshot("aic_fpm_forward_perf.metadata/v6"))
    base = {"schema_name": "aic_fpm_forward_perf", "schema_version": 6, "row_count": 1, "parquet_sha256": "0" * 64}
    group = {
        "name": "attention",
        "kind": "attention",
        "num_layers": 1,
        "block_size_tokens": 16,
        "page_size_bytes": 1024,
    }
    grouped = {k: v for k, v in RESOURCES.items() if k != "kv_bytes_per_token"}
    grouped.update(cache_layout="grouped", cache_groups=[group])
    unmodeled = {**grouped, "cache_layout": "unmodeled", "cache_groups": [], "runtime_memory": RUNTIME_MEMORY}
    mamba = {
        **unmodeled,
        "cache_layout": "mamba_align",
        "runtime_memory": {**RUNTIME_MEMORY, "kv_cache_bytes": 1024},
        "mamba_cache": {
            "state_storage": "full_state",
            "block_size_tokens": 16,
            "page_size_bytes": 256,
            "state_groups": 1,
            "prefix_match_unit": 16,
            "num_speculative_states": 0,
        },
    }
    state = {**grouped, "cache_groups": [_STATE_GROUP]}
    runtime_only = {**RESOURCES, "runtime_memory": RUNTIME_MEMORY}
    declared = {**RESOURCES, **dict.fromkeys(_MEMORY_NAMES, 0)}
    values = (RESOURCES, grouped, unmodeled, mamba, state, runtime_only, declared)
    for document in (base, *({**base, "resources": value} for value in values)):
        Draft202012Validator(schema).validate(document)
        FpmLibraryMetadata.model_validate(document)


@pytest.mark.parametrize("field", ["max_num_tokens", "max_batch_size"])
@pytest.mark.parametrize("limit", [2**32 - 1, 2**32])
def test_scheduler_limit_bounds_match_schema_python_and_native(profile, field, limit):
    resources = {**RESOURCES, field: limit}
    document = {
        "schema_name": "aic_fpm_forward_perf",
        "schema_version": 6,
        "row_count": 1,
        "parquet_sha256": "0" * 64,
        "resources": resources,
    }
    validator = Draft202012Validator(json.loads(render_snapshot("aic_fpm_forward_perf.metadata/v6")))
    profile["deployments"][0]["resources"] = resources
    config = _request(profile, "direct")
    if limit == 2**32 - 1:
        validator.validate(document)
        assert FpmLibraryMetadata.model_validate(document).resources.model_dump()[field] == limit
        normalized = RustForwardPassPerfModel.normalize_config(config)
        assert normalized["fpm_profile"]["deployments"][0]["resources"][field] == limit
    else:
        with pytest.raises(ValidationError):
            validator.validate(document)
        with pytest.raises(ValueError, match=field):
            FpmLibraryMetadata.model_validate(document)
        with pytest.raises(ValueError, match=field):
            RustForwardPassPerfModel.normalize_config(config)


@pytest.mark.parametrize("degree", [0, 1])
@pytest.mark.parametrize("dimension", ["tp", "pp", "dp", "moe_tp", "moe_ep", "cp"])
def test_selector_topology_matches_exported_schema(tmp_path, dimension, degree):
    document = {
        "schema_name": "aic_fpm_forward_perf",
        "schema_version": 6,
        "row_count": 1,
        "parquet_sha256": "0" * 64,
        "configuration_selector": {dimension: degree, "future_selector": "kept open"},
    }
    sidecar = tmp_path / "metadata.json"
    sidecar.write_text(json.dumps(document))
    validator = Draft202012Validator(json.loads(render_snapshot("aic_fpm_forward_perf.metadata/v6")))
    if degree == 1:
        validator.validate(document)
        assert read_library_metadata(sidecar) == document
    else:
        with pytest.raises(ValidationError):
            validator.validate(document)
        with pytest.raises(FpmContractError, match=f"configuration_selector.{dimension}"):
            read_library_metadata(sidecar)


@pytest.mark.parametrize(
    "group,valid",
    [
        ({"kind": "convolution"}, False),
        ({"kind": "convolution", "sliding_window": None}, False),
        ({"kind": "convolution", "sliding_window": 0}, False),
        ({"kind": "convolution", "sliding_window": 16}, True),
        ({"kind": "attention", "blocks_per_request": 1}, False),
        ({"kind": "convolution", "sliding_window": 16, "blocks_per_request": 1}, False),
        ({"kind": "attention", "blocks_per_request": None}, True),
        ({"kind": "attention"}, True),
        ({"kind": "attention", "prefill_checkpoint_blocks": None}, True),
        ({"kind": "attention", "prefill_checkpoint_blocks": 0}, False),
        ({"kind": "attention", "prefill_checkpoint_blocks": 1}, False),
        ({"kind": "convolution", "sliding_window": 16, "prefill_checkpoint_blocks": 1}, False),
        ({"kind": "state", "blocks_per_request": 1}, True),
        ({"kind": "state", "blocks_per_request": 1, "prefill_checkpoint_blocks": 0}, True),
        ({"kind": "state", "blocks_per_request": 1, "prefill_checkpoint_blocks": 1}, True),
        ({"kind": "state", "blocks_per_request": 1, "sliding_window": None}, True),
        ({"kind": "state", "blocks_per_request": 1, "sliding_window": 16}, False),
        ({"kind": "state", "blocks_per_request": 1, "prefill_checkpoint_alignment": 16}, False),
        (
            {
                "kind": "state",
                "blocks_per_request": 1,
                "prefill_checkpoint_alignment": 16,
                "prefill_checkpoint_blocks": 0,
            },
            False,
        ),
        (
            {
                "kind": "state",
                "blocks_per_request": 1,
                "prefill_checkpoint_alignment": 16,
                "prefill_checkpoint_blocks": 1,
            },
            True,
        ),
    ],
)
def test_cache_group_conditions_match_exported_schema(group, valid):
    group = {**_STATE_GROUP_WITHOUT_BLOCKS, **group}
    resources = {**_NO_KV, "cache_layout": "grouped", "cache_groups": [group]}
    document = {
        "schema_name": "aic_fpm_forward_perf",
        "schema_version": 6,
        "row_count": 1,
        "parquet_sha256": "0" * 64,
        "resources": resources,
    }
    validator = Draft202012Validator(json.loads(render_snapshot("aic_fpm_forward_perf.metadata/v6")))
    if valid:
        validator.validate(document)
        FpmLibraryMetadata.model_validate(document)
    else:
        with pytest.raises(ValidationError):
            validator.validate(document)
        with pytest.raises(ValueError):
            FpmLibraryMetadata.model_validate(document)


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


def _republish_library(root, table=None, **sidecar):
    """Rewrite the entry's library and/or sidecar, then re-pin both digests in the entry manifest."""
    library = root / ENTRY / "fpm/fpm_forward_perf.parquet"
    sidecar_path = root / ENTRY / "fpm/fpm_forward_perf.metadata.json"
    if table is not None:
        pq.write_table(table, library)
    declared = json.loads(sidecar_path.read_text())
    declared.update(sidecar, parquet_sha256=_sha(library.read_bytes()))
    sidecar_path.write_text(json.dumps(declared))
    path = root / MANIFEST
    document = json.loads(path.read_text())
    for artifact in document["artifacts"]:
        for target in (library, sidecar_path):
            if artifact["path"].endswith(target.name):
                artifact.update(sha256=_sha(target.read_bytes()), bytes=target.stat().st_size)
        if artifact["path"].endswith(library.name):
            artifact["rows"] = pq.ParquetFile(library).metadata.num_rows
    path.write_text(json.dumps(document))
    return pq.read_table(library)


def _with_column(table, name, values):
    if name in table.column_names:
        table = table.drop([name])
    return table.append_column(name, pa.array(values))


def test_entry_checks_bind_row_runtime_identity(root, profile):
    make_entry(root, profile)
    assert check_fpm_entry(root, MANIFEST) == []
    table = pq.read_table(root / ENTRY / "fpm/fpm_forward_perf.parquet")
    versions = [table.column("backend_version")[0].as_py()] * (table.num_rows - 1) + ["0.26.0"]
    _republish_library(root, _with_column(table, "backend_version", versions))
    findings = check_fpm_entry(root, MANIFEST)
    assert [f.code for f in findings] == ["library-runtime"] and "backend_version" in findings[0].message


@pytest.mark.parametrize(
    ("recorded", "selector", "codes"),
    [
        (None, {"dcp": 2}, ["library-dcp"]),  # declared DCP2 against rows that record none
        (2, {"dcp": 2}, []),  # TP4/DCP2 agrees
        (2, {}, ["library-dcp"]),  # recorded DCP2 against a selector that declares none
        (None, {}, []),  # unrecorded DCP without a selector value is not DCP1
        (3, {"dcp": 3}, ["library-dcp"]),  # DCP must divide tp
        (0, {"dcp": 1}, ["library-dcp", "library-dcp"]),  # DCP must be positive
    ],
)
def test_entry_checks_bind_declared_dcp_to_rows(root, profile, recorded, selector, codes):
    make_entry(root, profile)
    table = pq.read_table(root / ENTRY / "fpm/fpm_forward_perf.parquet")
    assert set(table.column("tp").to_pylist()) == {4}
    if recorded is not None:
        table = _with_column(table, "dcp", [recorded] * table.num_rows)
    base = {"tp": 4, "pp": 1, "dp": 1, "moe_tp": 4, "moe_ep": 1, "cp": 1}
    _republish_library(root, table, configuration_selector={**base, **selector})
    assert [f.code for f in check_fpm_entry(root, MANIFEST)] == codes


@pytest.mark.parametrize(
    "policy", ["per_row_single_sample_or_median_of_3", "per_row_single_sample_or_full_grid_median_v1"]
)
def test_entry_checks_require_row_evidence_for_a_per_row_policy(root, profile, policy):
    median = {
        "per_row_single_sample_or_median_of_3": "kvwarm_median_of_3",
        "per_row_single_sample_or_full_grid_median_v1": "independent_full_grid_median_of_3",
    }[policy]
    make_entry(root, profile)
    table = pq.read_table(root / ENTRY / "fpm/fpm_forward_perf.parquet")
    assert "measurement_policy" not in table.column_names  # the single-sample control needs no row columns
    assert check_fpm_entry(root, MANIFEST) == []
    _republish_library(root, table, measurement_policy=policy)
    assert [f.code for f in check_fpm_entry(root, MANIFEST)] == ["library-measurement-policy"]

    def rows(pairs):
        evidence = _with_column(table, "measurement_policy", [p for p, _ in pairs])
        return _with_column(evidence, "measurement_repeats", [r for _, r in pairs])

    size = table.num_rows
    single, repeated = ("dynamo_native_single_sample_v1", 1), (median, 3)
    mixed = [single] * (size - 1) + [repeated]
    _republish_library(root, rows(mixed), measurement_policy=policy)
    assert check_fpm_entry(root, MANIFEST) == []
    for inconsistent in [(median, 1), ("dynamo_native_single_sample_v1", 3), ("other_policy", 3)]:
        _republish_library(root, rows([*mixed[:-1], inconsistent]), measurement_policy=policy)
        findings = check_fpm_entry(root, MANIFEST)
        assert [f.code for f in findings] == ["library-measurement-policy"], inconsistent


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


# ------------------------------------------------------------------ review follow-ups


@pytest.mark.parametrize("schema", [None, "aic_fpm_forward_perf/v6", "aic_fpm_forward_perf/v7"])
def test_loader_checks_library_schema_even_with_matching_stamp(root, profile, tmp_path, schema):
    make_entry(root, profile, resources=RESOURCES)
    parquet = root / ENTRY / "fpm/fpm_forward_perf.parquet"
    table = pq.read_table(parquet)
    pq.write_table(table.replace_schema_metadata({b"aisim.schema": schema.encode()} if schema else None), parquet)
    sidecar = parquet.with_suffix(".metadata.json")
    metadata = json.loads(sidecar.read_text())
    metadata["parquet_sha256"] = _sha(parquet.read_bytes())
    sidecar.write_text(json.dumps(metadata))
    path = root / MANIFEST
    document = json.loads(path.read_text())
    document["artifacts"][:2] = [
        _artifact(
            root,
            parquet.relative_to(root).as_posix(),
            "fpm-library",
            "parquet",
            schema or "aic_fpm_forward_perf/v6",
            rows=table.num_rows,
        ),
        _artifact(root, sidecar.relative_to(root).as_posix(), "fpm-library-metadata", "json", "unversioned"),
    ]
    path.write_text(json.dumps(document))
    if schema == "aic_fpm_forward_perf/v7":
        with pytest.raises(ValueError, match="unsupported FPM library schema ID"):
            load(root, user_config(profile), tmp_path)
        assert not (tmp_path / "staged").exists()
    else:
        data = load(root, user_config(profile), tmp_path)
        assert data.kind == "library" and data.systems_root.is_dir()


@pytest.mark.parametrize("purpose", ["operation-tables", "execution-profile"])
def test_entry_checker_requires_loadable_purpose(root, profile, purpose):
    make_entry(root, profile)
    path = root / MANIFEST
    document = json.loads(path.read_text())
    document["purpose"] = purpose
    document["configuration_id"] = configuration_id("fpm", purpose, Identity.model_validate(document["identity"]))
    path.write_text(json.dumps(document))
    findings = check_fpm_entry(root, MANIFEST)
    assert [f.code for f in findings] == ([] if purpose == "execution-profile" else ["wrong-purpose"])


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("measurement", [{}, _measurement_document(), _measurement_document(1.0)])
def test_entry_checks_measurement_manifest_contents_only_when_enabled(root, profile, legacy, measurement):
    make_entry(root, profile)
    path = root / MANIFEST
    document = json.loads(path.read_text())
    if legacy:
        profile_path = f"{ENTRY}/profile.json"
        (root / profile_path).write_text(json.dumps(profile))
        document["artifacts"] = [_artifact(root, profile_path, "fpm-profile", "json", "unversioned")]
    else:
        document["artifacts"] = document["artifacts"][:2]
    measurement_path = f"{ENTRY}/measurements/manifest.json"
    if measurement:
        Draft202012Validator(json.loads(render_snapshot("measurement-manifest/v5"))).validate(measurement)
    (root / measurement_path).write_text(json.dumps(measurement))
    document["artifacts"].append(
        _artifact(root, measurement_path, "measurement-manifest", "json", "measurement-manifest/v5")
    )
    path.write_text(json.dumps(document))
    assert check_fpm_entry(root, MANIFEST) == []
    findings = check_fpm_entry(root, MANIFEST, include_measurements=True)
    valid = bool(measurement) and type(measurement["files"][0]["bytes"]) is int
    assert [f.code for f in findings] == ([] if valid else ["invalid-measurement-manifest"])
    if not valid:
        expected_field = "manifest_version" if not measurement else "bytes"
        assert expected_field in findings[0].message
    # Prediction-tier checks must still work without any evaluation files on disk.
    (root / measurement_path).unlink()
    assert check_fpm_entry(root, MANIFEST) == []


@pytest.mark.parametrize("field", ["row_count", "configuration_selector.tp", "measurement_repeats", "warmup_repeats"])
def test_publication_checks_strict_integer_spellings_beyond_json_schema(root, profile, field):
    make_entry(root, profile)
    sidecar = root / ENTRY / "fpm/fpm_forward_perf.metadata.json"
    metadata = json.loads(sidecar.read_text())
    parent = metadata["configuration_selector"] if field.startswith("configuration_selector.") else metadata
    key = field.rsplit(".", 1)[-1]
    parent[key] = float(parent.get(key, 1))
    # JSON Schema allows integral floats; publication must additionally apply the strict reader.
    Draft202012Validator(json.loads(render_snapshot("aic_fpm_forward_perf.metadata/v6"))).validate(metadata)
    sidecar.write_text(json.dumps(metadata))
    path = root / MANIFEST
    document = json.loads(path.read_text())
    document["artifacts"][1] = _artifact(
        root, sidecar.relative_to(root).as_posix(), "fpm-library-metadata", "json", "unversioned"
    )
    path.write_text(json.dumps(document))
    findings = check_fpm_entry(root, MANIFEST)
    assert [f.code for f in findings] == ["invalid-sidecar"]
    assert field in findings[0].message


@pytest.mark.parametrize("pp", [1, 2])
@pytest.mark.parametrize("key", ["pp", "pp_size"])
def test_loader_selects_and_loads_pipeline_parallel_entry(root, profile, tmp_path, pp, key):
    make_entry(root, profile)
    # The typed resource profile supports PP1 only. Rows and loader selection accept PP2; no PP2 request is priced.
    parquet = root / ENTRY / "fpm/fpm_forward_perf.parquet"
    table = pq.read_table(parquet)
    table = table.set_column(table.schema.get_field_index("pp"), "pp", pa.array([pp] * table.num_rows))
    pq.write_table(table, parquet)
    sidecar = parquet.with_suffix(".metadata.json")
    metadata = json.loads(sidecar.read_text())
    metadata["configuration_selector"]["pp"] = pp
    metadata["parquet_sha256"] = _sha(parquet.read_bytes())
    sidecar.write_text(json.dumps(metadata))
    path = root / MANIFEST
    document = json.loads(path.read_text())
    for artifact in document["artifacts"][:2]:
        data = (root / artifact["path"]).read_bytes()
        artifact.update(sha256=_sha(data), bytes=len(data))
    identity = Identity.model_validate(
        {**document["identity"], "parallelism": {"aggregated": {"tp": 4, "moe_tp": 4, "pp": pp}}}
    )
    document.update(
        identity=identity.model_dump(), configuration_id=configuration_id("fpm", "execution-profile", identity)
    )
    path.write_text(json.dumps(document))
    entry = Entry.model_validate(
        {
            "configuration_id": document["configuration_id"],
            "identity": document["identity"],
            "purpose": document["purpose"],
            "manifest": _artifact(root, MANIFEST, "manifest", "json", "aisim-artifact-manifest/v1"),
        }
    )
    config = {**CONFIG, key: pp}
    selection = FpmLoader().select(config)
    assert selection.matches(entry)
    data = load(root, config, tmp_path)
    assert data.config[key] == pp
    staged = data.config["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"]
    assert pq.read_table(staged, columns=["pp"]).column("pp").to_pylist() == [pp] * table.num_rows


@pytest.mark.parametrize(
    "setting", [[], "", False, 0, [["fpm_interpolation", {}]], None, {}, {"fpm_interpolation": {"k": 1}}]
)
def test_loader_validates_estimator_config_parent(root, profile, tmp_path, setting):
    make_entry(root, profile, resources=RESOURCES)
    config = {**user_config(profile), "estimator_config": setting}
    before = copy.deepcopy(config)
    if setting is not None and not isinstance(setting, dict):
        with pytest.raises(FpmContractError, match="estimator_config must be a mapping"):
            load(root, config, tmp_path)
    else:
        data = load(root, config, tmp_path)
        interpolation = data.config["estimator_config"]["fpm_interpolation"]
        assert Path(interpolation["fpm_parquet_path"]).is_file()
        if setting:
            assert interpolation["k"] == 1
    assert config == before


def test_stamp_exception_covers_only_the_supported_library_schema(tmp_path):
    from aisim_core.data import Artifact, _verify

    path = tmp_path / "lib.parquet"
    pq.write_table(pa.table({"x": [1]}), path)
    raw = path.read_bytes()

    def artifact(schema):
        return Artifact(
            artifact_id="lib",
            path="lib.parquet",
            role="fpm-library",
            format="parquet",
            schema=schema,
            sha256=_sha(raw),
            bytes=len(raw),
            rows=1,
            runtime_path=None,
        )

    assert _verify(tmp_path, artifact("aic_fpm_forward_perf/v6")) == path
    with pytest.raises(ValueError, match="schema ID"):
        _verify(tmp_path, artifact("aic_fpm_forward_perf/v7"))


def test_topology_aliases_select_like_their_canonical_names():
    from aisim_core.data.loaders import _parallelism

    canonical = {"tp": 4, "moe_tp_size": 4, "moe_ep_size": 1, "attention_dp": 2}
    aliased = {"tp_size": 4, "moe_tp_size": 4, "moe_ep_size": 1, "attention_dp_size": 2}
    assert _parallelism(aliased) == _parallelism(canonical) == (("dp", 2), ("moe_tp", 4), ("tp", 4))


def test_sidecar_rejects_duplicate_keys_and_boolean_topology(tmp_path):
    path = tmp_path / "meta.json"
    base = '"schema_name": "aic_fpm_forward_perf", "row_count": 1, "parquet_sha256": "' + "0" * 64 + '"'
    path.write_text("{" + base + ', "schema_version": 7, "schema_version": 6}')
    with pytest.raises(FpmContractError, match="duplicate JSON key"):
        read_library_metadata(path)
    path.write_text("{" + base + ', "schema_version": 6, "configuration_selector": {"tp": true}}')
    with pytest.raises(FpmContractError, match="tp"):
        read_library_metadata(path)
    path.write_text("{" + base + ', "schema_version": 6, "configuration_selector": {"tp": 1}}')
    assert read_library_metadata(path)["configuration_selector"] == {"tp": 1}


def test_staging_rejects_a_sidecar_that_describes_another_table(root, profile, tmp_path):
    make_entry(root, profile)
    source = root / ENTRY / "fpm"
    sidecar = source / "fpm_forward_perf.metadata.json"
    document = json.loads(sidecar.read_text())
    for change in ({"parquet_sha256": "0" * 64}, {"row_count": document["row_count"] + 1}):
        sidecar.write_text(json.dumps({**document, **change}))
        with pytest.raises(FpmContractError, match="differs"):
            stage_fpm_library(
                source / "fpm_forward_perf.parquet",
                sidecar,
                system="h200_sxm",
                backend="vllm",
                backend_version="0.25.1",
                output_dir=tmp_path / "out",
            )
    document.pop("parquet_sha256")  # native v6 admission requires the digest
    sidecar.write_text(json.dumps(document))
    with pytest.raises(FpmContractError, match="parquet_sha256"):
        stage_fpm_library(
            source / "fpm_forward_perf.parquet",
            sidecar,
            system="h200_sxm",
            backend="vllm",
            backend_version="0.25.1",
            output_dir=tmp_path / "out",
        )


def test_concurrent_staging_of_one_destination_does_not_collide(root, profile, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    make_entry(root, profile)
    source = root / ENTRY / "fpm"

    def stage(_):
        return stage_fpm_library(
            source / "fpm_forward_perf.parquet",
            source / "fpm_forward_perf.metadata.json",
            system="h200_sxm",
            backend="vllm",
            backend_version="0.25.1",
            output_dir=tmp_path / "out",
        ).systems_root

    with ThreadPoolExecutor(8) as pool:
        assert len(set(pool.map(stage, range(16)))) == 1
    assert not list(tmp_path.glob(".out.*"))


def test_staging_reuse_follows_overlay_precedence(root, profile, tmp_path):
    make_entry(root, profile)
    source = root / ENTRY / "fpm"
    first, fallback = tmp_path / "first", tmp_path / "fallback"
    for systems in (first, fallback):
        data = systems / "data/custom/vllm/0.25.1"
        data.mkdir(parents=True)
        (systems / "h200_sxm.yaml").write_text("data_dir: data/custom\n")
    (fallback / "data/custom/vllm/0.25.1/gemm.parquet").write_bytes(b"fallback")
    kwargs = dict(
        system="h200_sxm",
        backend="vllm",
        backend_version="0.25.1",
        output_dir=tmp_path / "out",
        systems_paths=[first, fallback],
    )
    args = (source / "fpm_forward_perf.parquet", source / "fpm_forward_perf.metadata.json")
    warm = stage_fpm_library(*args, **kwargs)
    gemm = warm.systems_root / "data/custom/vllm/0.25.1/gemm.parquet"
    assert gemm.read_bytes() == b"fallback"
    (first / "data/custom/vllm/0.25.1/gemm.parquet").write_bytes(b"first")
    stage_fpm_library(*args, **kwargs)
    assert gemm.read_bytes() == b"first"  # the same result as a cold staging


def test_sidecar_resources_wait_for_an_identified_deployment(profile):
    metadata = {"resources": RESOURCES}
    two = copy.deepcopy(profile)
    two["deployments"] = [{**two["deployments"][0], "tp": tp, "moe_tp": tp} for tp in (4, 8)]
    for deployment in two["deployments"]:
        deployment.pop("resources", None)
    _, sources = complete_fpm_profile(two, metadata, required=False)
    assert sources == ["pending", "pending"]  # no selector: the sidecar does not say which one it describes
    one = copy.deepcopy(two)
    one["deployments"] = one["deployments"][:1]
    assert complete_fpm_profile(one, metadata, required=False)[1] == ["metadata"]


def test_selectorless_sidecar_resources_describe_the_one_requested_deployment(root, profile):
    make_entry(root, profile)
    identity = Manifest.model_validate_json((root / MANIFEST).read_bytes()).identity
    metadata = {"resources": RESOURCES}  # published without a configuration_selector
    for has_config in (True, False):
        resolved = _resolve_library_resources(dict(CONFIG), None, identity, metadata, True, has_config, False)[0]
        assert resolved.source == "metadata"
    # A selector with no requested deployment to match against still leaves the resources pending.
    selected = {**metadata, "configuration_selector": {"tp": 4}}
    assert _resolve_library_resources({}, None, identity, selected, False, False, False)[0].source == "pending"


def test_partial_selector_must_distinguish_dep4_from_dep8(profile):
    draft = copy.deepcopy(profile)
    draft["deployments"] = [
        {**draft["deployments"][0], "tp": 1, "moe_tp": 1, "dp": degree, "moe_ep": degree} for degree in (4, 8)
    ]
    # These are supported DEP configurations, rather than identities rejected by profile validation.
    valid = copy.deepcopy(draft)
    for deployment in valid["deployments"]:
        deployment["resources"] = RESOURCES
    load_fpm_profile(valid)
    for deployment in draft["deployments"]:
        deployment.pop("resources", None)
    identity = {"system": "h200_sxm", "backend": "vllm", "backend_version": "0.25.1"}
    metadata = {"resources": RESOURCES, "configuration_selector": {"tp": 1}}
    assert complete_fpm_profile(draft, metadata, identity=identity, required=False)[1] == ["pending", "pending"]
    with pytest.raises(FpmResourcesPendingError):
        complete_fpm_profile(draft, metadata, identity=identity)
    # An explicit block does not establish which deployment the sidecar evidence describes.
    user = {**RESOURCES, "max_num_tokens": 4096, "provenance": "Explicit DEP8 fixture resources."}
    draft["deployments"][1]["resources"] = user
    assert complete_fpm_profile(draft, metadata, identity=identity, required=False)[1] == ["pending", "user"]
    metadata["configuration_selector"]["moe_ep"] = 4
    completed, sources = complete_fpm_profile(draft, metadata, identity=identity)
    assert sources == ["metadata", "user"]
    assert completed["deployments"][0]["resources"]["max_num_tokens"] == RESOURCES["max_num_tokens"]
    assert completed["deployments"][1]["resources"]["max_num_tokens"] == user["max_num_tokens"]


def test_selector_precision_needs_a_deployment_that_establishes_it(profile):
    draft = copy.deepcopy(profile)
    draft["deployments"][0].pop("resources", None)
    draft["deployments"][0].pop("weight_quantization", None)
    draft.pop("weight_quantization", None)
    metadata = {"resources": RESOURCES, "configuration_selector": {"tp": 4, "weight_quantization": "fp8_block"}}
    assert complete_fpm_profile(draft, metadata, required=False)[1] == ["pending"]


def _entry_catalog(root, manifest_path=MANIFEST, **entry_changes):
    manifest = Manifest.model_validate_json((root / manifest_path).read_bytes())
    (root / "catalog/fpm").mkdir(parents=True, exist_ok=True)
    reference = _artifact(root, manifest_path, "manifest", "json", "aisim-artifact-manifest/v1")
    entry = {
        "configuration_id": manifest.configuration_id,
        "purpose": "execution-profile",
        "identity": manifest.identity.model_dump(),
        "manifest": reference,
        **entry_changes,
    }
    index = {"schema_version": "aisim-category-index/v1", "category": "fpm", "entries": [entry]}
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
    return reference


def test_dataset_check_reports_index_digest_and_entry_mismatches(root, profile):
    make_entry(root, profile)
    reference = _entry_catalog(root)
    assert check_fpm_dataset(root) == []
    other = Manifest.model_validate_json((root / MANIFEST).read_bytes()).identity.model_dump()
    other["model"] = "other/model"
    _entry_catalog(root, identity=other)
    assert [f.code for f in check_fpm_dataset(root)] == ["entry-mismatch"]
    # A manifest edited without updating its indexed digest is reported, and the other entries continue.
    _entry_catalog(root)
    document = json.loads((root / MANIFEST).read_text())
    document["provenance"] = {"edited": True}
    (root / MANIFEST).write_text(json.dumps(document))
    assert [f.code for f in check_fpm_dataset(root)] == ["hash-mismatch"]
    assert reference["sha256"] != _sha((root / MANIFEST).read_bytes())


@pytest.mark.parametrize("valid", [True, False])
def test_entry_check_parses_a_legacy_profile_entry(root, profile, valid):
    make_entry(root, profile)
    document = json.loads((root / MANIFEST).read_text())
    keep = [a for a in document["artifacts"] if a["role"] == "legacy-configuration-manifest"]
    keep[0] = {**keep[0], "role": "fpm-profile"}
    document["artifacts"] = keep
    (root / keep[0]["path"]).write_text(json.dumps(profile if valid else {}))
    keep[0] = _artifact(root, keep[0]["path"], "fpm-profile", "json", "unversioned")
    (root / MANIFEST).write_text(json.dumps(document))
    if valid:
        assert check_fpm_entry(root, MANIFEST) == []
        assert FpmLoader().load(root, MANIFEST).profile == load_fpm_profile(profile)
    else:
        assert [f.code for f in check_fpm_entry(root, MANIFEST)] == ["invalid-profile"]
        with pytest.raises(ValueError):
            FpmLoader().load(root, MANIFEST)
    (root / keep[0]["path"]).write_text("edited without updating its hash")
    assert [f.code for f in check_fpm_entry(root, MANIFEST)] == ["hash-mismatch"]


def test_entry_check_applies_the_reconstruction_contract(root, profile):
    make_entry(root, profile)
    source = root / ENTRY / "fpm"
    sidecar = source / "fpm_forward_perf.metadata.json"
    document = json.loads(sidecar.read_text())
    for name in ("coordinate_system", "measurement_policy", "system", "backend", "backend_version"):
        document.pop(name, None)
    document["metadata_provenance"] = "reconstructed_from_parquet"
    document["configuration_selector"] = {
        "model_path": profile["model"],
        "system": "h200_sxm",
        "backend": "vllm",
        "backend_version": "0.25.1",
    }
    sidecar.write_text(json.dumps(document))
    document_manifest = json.loads((root / MANIFEST).read_text())
    for artifact in document_manifest["artifacts"]:
        if artifact["role"] == "fpm-library-metadata":
            data = sidecar.read_bytes()
            artifact.update(sha256=_sha(data), bytes=len(data))
    (root / MANIFEST).write_text(json.dumps(document_manifest))
    codes = {f.code for f in check_fpm_entry(root, MANIFEST)}
    assert codes == {"invalid-sidecar"}


def test_dataset_check_reports_nonfinite_identity_and_continues(root, profile):
    make_entry(root, profile)
    document = json.loads((root / MANIFEST).read_text())
    document["identity"]["effective_settings"]["invalid"] = float("nan")
    (root / MANIFEST).write_text(json.dumps(document))
    _entry_catalog(root)
    second = copy.deepcopy(document)
    second["identity"]["effective_settings"] = {}
    second["identity"]["model"] = "other/model"
    second["configuration_id"] = "cfg-" + "0" * 64
    second_path = f"{ENTRY}/second-manifest.json"
    (root / second_path).write_text(json.dumps(second))
    index_path = root / "catalog/fpm/index.json"
    index = json.loads(index_path.read_text())
    index["entries"].append(
        {
            "configuration_id": second["configuration_id"],
            "purpose": second["purpose"],
            "identity": second["identity"],
            "manifest": _artifact(root, second_path, "manifest", "json", "aisim-artifact-manifest/v1"),
        }
    )
    index_path.write_text(json.dumps(index))
    catalog_path = root / "catalog/dataset.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["categories"]["fpm"]["index"] = _artifact(root, "catalog/fpm/index.json", "index", "json", "x")
    catalog_path.write_text(json.dumps(catalog))
    findings = check_fpm_dataset(root)
    assert [(f.path, f.code) for f in findings] == [
        (MANIFEST, "invalid-manifest"),
        (second_path, "configuration-identity"),
    ]


def test_legacy_entry_checks_measurements_only_on_request(root, profile):
    make_entry(root, profile)
    document = json.loads((root / MANIFEST).read_text())
    profile_path = f"{ENTRY}/profile.json"
    (root / profile_path).write_text(json.dumps(profile))
    document["artifacts"] = [
        _artifact(root, profile_path, "fpm-profile", "json", "unversioned"),
        *(a for a in document["artifacts"] if a["role"] in EVALUATION_ROLES),
    ]
    (root / MANIFEST).write_text(json.dumps(document))
    assert check_fpm_entry(root, MANIFEST) == []
    findings = check_fpm_entry(root, MANIFEST, include_measurements=True)
    assert len(findings) == 2 and all(f.code == "hash-mismatch" for f in findings)
    assert "measurement-data" in findings[1].message


@pytest.mark.parametrize("field", ["measurement_repeats", "warmup_repeats"])
@pytest.mark.parametrize("value", [True, False, 1.5, "1", 1, None, "omitted"])
def test_sidecar_repeat_counts_agree_with_packaged_schema(tmp_path, field, value):
    from aisim_core.data.schemas import PACKAGED_SNAPSHOT_DIR, snapshot_filename

    schema = json.loads((PACKAGED_SNAPSHOT_DIR / snapshot_filename("aic_fpm_forward_perf.metadata/v6")).read_text())
    document = {"schema_name": "aic_fpm_forward_perf", "schema_version": 6, "row_count": 1, "parquet_sha256": "0" * 64}
    if value != "omitted":
        document[field] = value
    sidecar = tmp_path / "metadata.json"
    sidecar.write_text(json.dumps(document))
    if type(value) is int or value is None or value == "omitted":
        Draft202012Validator(schema).validate(document)
        assert read_library_metadata(sidecar) == document
    else:
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate(document)
        with pytest.raises(FpmContractError, match=field):
            read_library_metadata(sidecar)


@pytest.mark.parametrize("value", [True, False, 1.5, "1", 1, 0])
def test_measurement_bytes_agree_with_packaged_schema(value):
    from aisim_core.data.schemas import PACKAGED_SNAPSHOT_DIR, snapshot_filename

    schema = json.loads((PACKAGED_SNAPSHOT_DIR / snapshot_filename("measurement-manifest/v5")).read_text())
    document = _measurement_document(value)
    if type(value) is int:
        Draft202012Validator(schema).validate(document)
        assert FpmMeasurementManifest.model_validate(document).files[0].bytes == value
    else:
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate(document)
        with pytest.raises(ValueError, match="bytes"):
            FpmMeasurementManifest.model_validate(document)


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 0),
        ("batch_size", -1),
        ("total_prefill_tokens", -1),
        ("total_kv_read_tokens", -1),
    ],
)
def test_entry_checks_coordinates_with_a_complete_sidecar(root, profile, field, value):
    make_entry(root, profile)
    directory = root / ENTRY / "fpm"
    parquet = directory / "fpm_forward_perf.parquet"
    table = pq.read_table(parquet)
    # Keep enough valid rows to require scanning beyond the first Arrow batch.
    rows = [dict(row) for _ in range(1400) for row in table.to_pylist()]
    rows[-1][field] = value
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    sidecar = directory / "fpm_forward_perf.metadata.json"
    metadata = json.loads(sidecar.read_text())
    metadata.update(parquet_sha256=_sha(parquet.read_bytes()), row_count=len(rows))
    sidecar.write_text(json.dumps(metadata))
    manifest = json.loads((root / MANIFEST).read_text())
    manifest["artifacts"][:2] = [
        _artifact(
            root,
            f"{ENTRY}/fpm/fpm_forward_perf.parquet",
            "fpm-library",
            "parquet",
            "aic_fpm_forward_perf/v6",
            rows=len(rows),
        ),
        _artifact(root, f"{ENTRY}/fpm/fpm_forward_perf.metadata.json", "fpm-library-metadata", "json", "unversioned"),
    ]
    (root / MANIFEST).write_text(json.dumps(manifest))
    findings = check_fpm_entry(root, MANIFEST)
    assert [f.code for f in findings] == ["library-coordinates"]
    assert field in findings[0].message


@pytest.mark.parametrize("latency", [-1.0, 0.0, float("nan"), float("inf"), -float("inf"), None, 1.0])
@pytest.mark.parametrize("row_count", [1, 4097])
def test_entry_checks_latency_with_valid_declarations(root, profile, latency, row_count):
    make_entry(root, profile)
    directory = root / ENTRY / "fpm"
    parquet = directory / "fpm_forward_perf.parquet"
    table = pq.read_table(parquet)
    row = table.slice(0, 1).to_pylist()[0]
    # Distinct physical coordinates: Rust rejects duplicate rows, so this fixture must be natively loadable.
    rows = [{**row, "latency_ms": 1.0, "batch_size": index + 1} for index in range(row_count)]
    rows[-1]["latency_ms"] = latency
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), parquet)
    sidecar = directory / "fpm_forward_perf.metadata.json"
    metadata = json.loads(sidecar.read_text())
    metadata.update(parquet_sha256=_sha(parquet.read_bytes()), row_count=row_count)
    sidecar.write_text(json.dumps(metadata))
    manifest = json.loads((root / MANIFEST).read_text())
    manifest["artifacts"][:2] = [
        _artifact(
            root,
            f"{ENTRY}/fpm/fpm_forward_perf.parquet",
            "fpm-library",
            "parquet",
            "aic_fpm_forward_perf/v6",
            rows=row_count,
        ),
        _artifact(root, f"{ENTRY}/fpm/fpm_forward_perf.metadata.json", "fpm-library-metadata", "json", "unversioned"),
    ]
    (root / MANIFEST).write_text(json.dumps(manifest))
    findings = check_fpm_entry(root, MANIFEST)
    if latency == 1.0:
        assert findings == []
    else:
        assert [finding.code for finding in findings] == ["library-latency"]
        assert "latency_ms" in findings[0].message


# ------------------------------------------------------------------ review hardening


def test_library_declaration_is_validated_before_the_format_branch(root, profile):
    from aisim_core.data import _verify

    make_entry(root, profile, resources=RESOURCES)
    manifest = Manifest.model_validate_json((root / MANIFEST).read_bytes())
    library = next(a for a in manifest.artifacts if a.role == "fpm-library")
    assert _verify(root, library).is_file()
    for update in ({"format": "json"}, {"rows": None}, {"artifact_schema": "aic_fpm_forward_perf/v7"}):
        with pytest.raises(ValueError, match="fpm-library|FPM library schema"):
            _verify(root, library.model_copy(update=update))


def test_library_row_integers_reject_booleans():
    from pydantic import ValidationError as ModelValidationError

    from aisim_core.data.fpm_library import FpmLibraryRow

    samples = {str: "x", int: 1, float: 1.0, bool: False}
    row = {name: samples[info.annotation] for name, info in FpmLibraryRow.model_fields.items() if info.is_required()}
    assert FpmLibraryRow.model_validate(row).tp == 1
    with pytest.raises(ModelValidationError):
        FpmLibraryRow.model_validate({**row, "tp": True})


@pytest.mark.parametrize("inventory", [None, ["other/model"], ["MiniMaxAI/MiniMax-M2.7"]])
@pytest.mark.parametrize("temporary", [False, True])
def test_manifest_model_must_be_covered_by_library_rows(root, profile, tmp_path, inventory, temporary):
    make_entry(root, profile)
    parquet, sidecar = _stage_args(root)
    table = pq.read_table(parquet)
    table = table.set_column(
        table.schema.get_field_index("model_path"), "model_path", pa.array(["other/model"] * table.num_rows)
    )
    pq.write_table(table, parquet)
    metadata = json.loads(sidecar.read_text())
    metadata["parquet_sha256"] = _sha(parquet.read_bytes())
    if inventory is not None:
        metadata["model_paths"] = inventory
    sidecar.write_text(json.dumps(metadata))
    document = json.loads((root / MANIFEST).read_text())
    for artifact in document["artifacts"][:2]:
        data = (root / artifact["path"]).read_bytes()
        artifact.update(sha256=_sha(data), bytes=len(data))
    (root / MANIFEST).write_text(json.dumps(document))
    findings = check_fpm_entry(root, MANIFEST)
    assert [f.code for f in findings] == ["invalid-sidecar"]
    expected = "model_paths differs" if inventory == [profile["model"]] else "does not cover manifest model"
    assert expected in findings[0].message
    with pytest.raises(LoaderError, match=expected):
        FpmLoader().load(
            root, MANIFEST, output_dir=None if temporary else tmp_path / "out", config=user_config(profile)
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("inventory", [False, True])
def test_shared_library_covers_multiple_manifest_models(root, profile, tmp_path, inventory):
    make_entry(root, profile)
    parquet, sidecar = _stage_args(root)
    rows = pq.read_table(parquet).to_pylist()
    rows += [
        {**row, "cell_id": "other-" + row["cell_id"], "model_path": "other/model", "latency_ms": 17.0} for row in rows
    ]
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    metadata = json.loads(sidecar.read_text())
    metadata.update(row_count=len(rows), parquet_sha256=_sha(parquet.read_bytes()))
    if inventory:
        metadata["model_paths"] = [profile["model"], "other/model"]
    sidecar.write_text(json.dumps(metadata))
    document = json.loads((root / MANIFEST).read_text())
    for artifact in document["artifacts"][:2]:
        data = (root / artifact["path"]).read_bytes()
        artifact.update(sha256=_sha(data), bytes=len(data))
        if artifact["role"] == "fpm-library":
            artifact["rows"] = len(rows)
    (root / MANIFEST).write_text(json.dumps(document))
    assert check_fpm_entry(root, MANIFEST) == []
    data = load(root, user_config(profile), tmp_path)
    model = RustForwardPassPerfModel.best_available(data.config)
    try:
        assert model.estimate_forward_pass_time_ms(
            [{"scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 1}}]
        ) == pytest.approx(2.0)
    finally:
        model.close()


@pytest.mark.parametrize("value", [None, "unsupported", "iteration_totals_balanced_v1"])
@pytest.mark.parametrize("provenance", [None, "reconstructed_from_parquet"])
def test_coordinate_system_agrees_in_schema_publication_and_staging(root, profile, tmp_path, value, provenance):
    make_entry(root, profile, sidecar_extra={"coordinate_system": value, "metadata_provenance": provenance})
    _, sidecar = _stage_args(root)
    metadata = json.loads(sidecar.read_text())
    schema = json.loads(render_snapshot("aic_fpm_forward_perf.metadata/v6"))
    if value == "iteration_totals_balanced_v1":
        Draft202012Validator(schema).validate(metadata)
        assert check_fpm_entry(root, MANIFEST) == []
        stage_fpm_library(*_stage_args(root), **KWARGS, output_dir=tmp_path / "out")
    else:
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate(metadata)
        findings = check_fpm_entry(root, MANIFEST)
        assert [f.code for f in findings] == ["invalid-sidecar"]
        assert "coordinate_system" in findings[0].message
        with pytest.raises(FpmContractError, match="coordinate_system"):
            stage_fpm_library(*_stage_args(root), **KWARGS, output_dir=tmp_path / "out")


@pytest.mark.parametrize("version", [" ", "\t\n", "", None, "0.25.1"])
def test_runtime_version_nonblank_constraint_matches_sidecar_schema_and_reader(tmp_path, version):
    metadata = {
        "schema_name": "aic_fpm_forward_perf",
        "schema_version": 6,
        "row_count": 1,
        "parquet_sha256": "0" * 64,
        "resources": {**RESOURCES, "runtime_memory": {**RUNTIME_MEMORY, "runtime_framework_version": version}},
    }
    schema = json.loads(render_snapshot("aic_fpm_forward_perf.metadata/v6"))
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(json.dumps(metadata))
    if version is None or version.strip():
        Draft202012Validator(schema).validate(metadata)
        assert read_library_metadata(sidecar) == metadata
    else:
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate(metadata)
        with pytest.raises(FpmContractError, match="runtime_framework_version"):
            read_library_metadata(sidecar)


def test_sidecar_reader_rejects_duplicate_keys_and_float_overflow(tmp_path):
    sidecar = tmp_path / "sidecar.json"
    base = (
        '{"schema_name": "aic_fpm_forward_perf", "schema_version": 6, "row_count": 1, "parquet_sha256": "'
        + "0" * 64
        + '", %s}'
    )
    sidecar.write_text(base % '"b": 1, "a": 1, "b": 2, "a": 3')
    with pytest.raises(FpmContractError, match="duplicate JSON key 'a'"):
        read_library_metadata(sidecar)
    sidecar.write_text(base % '"scale": 1e400')
    with pytest.raises(FpmContractError, match="non-finite"):
        read_library_metadata(sidecar)
    sidecar.write_text(base % '"scale": 1.5e3')
    assert read_library_metadata(sidecar)["scale"] == 1500.0


def _stage_args(root):
    source = root / ENTRY / "fpm"
    return (source / "fpm_forward_perf.parquet", source / "fpm_forward_perf.metadata.json")


KWARGS = dict(system="h200_sxm", backend="vllm", backend_version="0.25.1")


def _systems_root_with_tables(tmp_path, name, tables):
    """A systems root with the packaged h200 definition and stub op tables at the given data-relative paths."""
    from aisim_core.data.fpm_library import _stageable_roots

    systems = tmp_path / name
    systems.mkdir()
    shutil.copyfile(_stageable_roots(["default"], None)[0] / "h200_sxm.yaml", systems / "h200_sxm.yaml")
    for relative, content in tables.items():
        table = systems / "data/h200_sxm" / relative
        table.parent.mkdir(parents=True, exist_ok=True)
        table.write_bytes(content)
    return systems


def test_staging_merges_family_directories_of_every_root(root, profile, tmp_path, monkeypatch):
    import aisim_core
    from aisim_core.sdk import perf_database

    make_entry(root, profile, resources=RESOURCES)
    monkeypatch.setattr(perf_database, "get_systems_paths", lambda: [])
    first = _systems_root_with_tables(
        tmp_path, "first", {"gemm/vllm/0.24/gemm_perf.parquet": b"first", "gemm/vllm/0.24/shared.txt": b"first"}
    )
    second = _systems_root_with_tables(
        tmp_path,
        "second",
        {"gemm/vllm/0.25.1/gemm_perf.parquet": b"second", "gemm/vllm/0.24/shared.txt": b"second"},
    )
    parquet, sidecar = _stage_args(root)
    staged = stage_fpm_library(parquet, sidecar, output_dir=tmp_path / "out", systems_paths=[first, second], **KWARGS)
    records = json.loads(
        aisim_core.resolve_op_sources_report_json(
            str(staged.systems_root),
            str(staged.systems_root / "data/h200_sxm"),
            "vllm",
            "0.25.1",
            "gemm_perf.parquet",
        )
    )["records"]
    # The table only the later root has is reachable through the staged root alone; the earlier root wins ties.
    assert records[0]["channel"] == "primary" and records[0]["exists"]
    assert Path(records[0]["path"]).read_bytes() == b"second"
    shared = staged.systems_root / "data/h200_sxm/gemm/vllm/0.24/shared.txt"
    assert shared.read_bytes() == b"first"
    assert (staged.systems_root / "data/h200_sxm/gemm/vllm/0.24/gemm_perf.parquet").read_bytes() == b"first"


@pytest.mark.parametrize("lost", ["sidecar", "library", "system", "data"])
def test_restaging_repairs_an_installed_root_that_lost_a_file(root, profile, tmp_path, monkeypatch, lost):
    from aisim_core.sdk import perf_database

    make_entry(root, profile, resources=RESOURCES)
    monkeypatch.setattr(perf_database, "get_systems_paths", lambda: [])
    source = _systems_root_with_tables(tmp_path, "source", {"gemm/vllm/0.25.1/gemm_perf.parquet": b"table"})
    parquet, sidecar = _stage_args(root)
    kwargs = dict(output_dir=tmp_path / "out", systems_paths=[source], **KWARGS)
    staged = stage_fpm_library(parquet, sidecar, **kwargs)
    marker = staged.systems_root / ".staged.json"
    marker_bytes = marker.read_bytes()
    damaged = {
        "sidecar": staged.metadata_path,
        "library": staged.parquet_path,
        "system": staged.systems_root / "h200_sxm.yaml",
        "data": staged.systems_root / "data/h200_sxm/gemm",
    }[lost]
    damaged.unlink()  # the marker still says this root was staged completely
    assert marker.read_bytes() == marker_bytes
    repaired = stage_fpm_library(parquet, sidecar, **kwargs)
    assert repaired.parquet_path.exists() and repaired.metadata_path.exists()
    assert (repaired.systems_root / "h200_sxm.yaml").is_file()
    assert (repaired.systems_root / "data/h200_sxm/gemm/vllm/0.25.1/gemm_perf.parquet").read_bytes() == b"table"


def test_repin_to_another_cache_does_not_keep_the_old_library(root, profile, tmp_path):
    from aisim_core.data.ensure import DataPin, ensure_data

    other = tmp_path / "other"
    make_entry(root, profile, resources=RESOURCES)
    make_entry(other, profile, resources=RESOURCES)
    parquet, sidecar = _stage_args(other)
    rows = pq.read_table(parquet).to_pylist()
    for row in rows:
        if row["workload_kind"] == "decode":
            row["latency_ms"] = 7.0  # the repinned library answers differently from the first
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    metadata = json.loads(sidecar.read_text())
    metadata["parquet_sha256"] = _sha(parquet.read_bytes())
    sidecar.write_text(json.dumps(metadata))
    _refresh_prediction_manifest(other)
    for source in (root, other):
        _write_entry_catalog(source)
    first = ensure_data(
        user_config(profile), DataPin(local_root=root, categories=("fpm",), cache_dir=tmp_path / "cache-a")
    )
    second = ensure_data(first.config, DataPin(local_root=other, categories=("fpm",), cache_dir=tmp_path / "cache-b"))
    assert str(tmp_path / "cache-a") not in json.dumps(second.config)  # a generated path is not a user override
    query = [{"scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": 1}}]
    model_a = RustForwardPassPerfModel.best_available(first.config)
    model_b = RustForwardPassPerfModel.best_available(second.config)
    try:
        assert model_a.estimate_forward_pass_time_ms(query) == pytest.approx(3.0)
        assert model_b.estimate_forward_pass_time_ms(query) == pytest.approx(7.0)
    finally:
        model_a.close()
        model_b.close()


def test_staged_root_keeps_the_reuse_policy_of_its_source_root(root, profile, tmp_path, monkeypatch):
    import aisim_core
    from aisim_core.data.fpm_library import ROOT_POLICY_FILES, _stageable_roots
    from aisim_core.sdk import perf_database

    make_entry(root, profile, resources=RESOURCES)
    source = tmp_path / "policy-root"
    source.mkdir()
    shutil.copyfile(_stageable_roots(["default"], None)[0] / "h200_sxm.yaml", source / "h200_sxm.yaml")
    monkeypatch.setattr(perf_database, "get_systems_paths", lambda: [])  # the staging sees the source root alone
    data = source / "data/h200_sxm"
    for backend, version in (("vllm", "0.25.1"), ("sglang", "0.5.0")):
        table = data / "gemm" / backend / version / "gemm_perf.parquet"
        table.parent.mkdir(parents=True)
        table.write_bytes(b"stub")
    policy = {
        "perf_data_reuse_manifest.yaml": (
            "groups:\n  - op_file: gemm_perf.parquet\n    kernel_source: shared_kernel\n    tier: shared\n"
            "    frameworks: [sglang, vllm]\n"
        ),
        "query_versions.yaml": "schema_version: 1\ndefaults: {vllm: {current: '0.25.1'}}\noverrides: {}\n",
        "attention_lane_defaults.yaml": "lanes: {}\n",
    }
    for name, text in policy.items():
        (source / name).write_text(text)
    parquet, sidecar = _stage_args(root)

    def channels(systems_root):
        report = json.loads(
            aisim_core.resolve_op_sources_report_json(
                str(systems_root), str(systems_root / "data/h200_sxm"), "vllm", "0.25.1", "gemm_perf.parquet"
            )
        )
        return [record["channel"] for record in report["records"]]

    expected = channels(source)
    assert "cross_backend" in expected  # the donor is admitted only through the root's reuse manifest
    output = tmp_path / "staged"
    staged = stage_fpm_library(parquet, sidecar, output_dir=output, systems_paths=[source], **KWARGS)
    assert channels(staged.systems_root) == expected
    for name, text in policy.items():
        assert (staged.systems_root / name).read_text() == text
    assert set(ROOT_POLICY_FILES) == set(policy)
    # An unchanged policy reuses the staging; a changed one restages it with the new policy.
    marker = staged.systems_root / ".staged.json"
    inode = marker.stat().st_ino
    stage_fpm_library(parquet, sidecar, output_dir=output, systems_paths=[source], **KWARGS)
    assert marker.stat().st_ino == inode
    (source / "perf_data_reuse_manifest.yaml").write_text("groups: []\n")
    restaged = stage_fpm_library(parquet, sidecar, output_dir=output, systems_paths=[source], **KWARGS)
    assert (restaged.systems_root / "perf_data_reuse_manifest.yaml").read_text() == "groups: []\n"
    assert "cross_backend" not in channels(restaged.systems_root)


def test_invalid_system_yaml_is_a_contract_error(root, profile, tmp_path):
    make_entry(root, profile, resources=RESOURCES)
    systems = tmp_path / "systems"
    systems.mkdir()
    (systems / "h200_sxm.yaml").write_text("data_dir: [")
    with pytest.raises(FpmContractError, match="not valid YAML"):
        stage_fpm_library(*_stage_args(root), **KWARGS, systems_paths=[systems], output_dir=tmp_path / "out")


def test_staging_inside_a_system_data_tree_is_rejected(root, profile, tmp_path):
    make_entry(root, profile, resources=RESOURCES)
    systems = tmp_path / "systems"
    (systems / "data/h200_sxm/vllm").mkdir(parents=True)
    (systems / "h200_sxm.yaml").write_text("data_dir: data/h200_sxm\n")
    with pytest.raises(FpmContractError, match="must not be inside the system data tree"):
        stage_fpm_library(
            *_stage_args(root), **KWARGS, systems_paths=[systems], output_dir=systems / "data/h200_sxm/vllm/out"
        )
    assert not any(p.is_symlink() and not p.exists() for p in systems.rglob("*"))


def test_failed_resource_contract_leaves_the_previous_staging_in_place(root, profile, tmp_path):
    make_entry(root, profile, resources=RESOURCES)
    first = load(root, user_config(profile), tmp_path)
    before = first.config["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"]
    other = tmp_path / "other"
    make_entry(other, profile)  # no sidecar resources
    with pytest.raises(FpmResourcesPendingError):
        load(other, user_config(profile), tmp_path, require_resources=True)
    staged = Path(before)
    assert staged.resolve() == (root / ENTRY / "fpm/fpm_forward_perf.parquet").resolve()


def test_sidecar_resources_do_not_attach_to_a_different_explicit_library(root, profile, tmp_path):
    make_entry(root, profile, resources=RESOURCES)
    config = {**user_config(profile)}
    config["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"] = "/user/other.parquet"
    data = load(root, config, tmp_path)
    assert data.resource_sources == ("pending",)
    with pytest.raises(FpmResourcesPendingError):
        load(root, config, tmp_path / "again", require_resources=True)
    own = user_config(profile)
    own["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"] = str(_stage_args(root)[0])
    assert load(root, own, tmp_path / "own").resource_sources == ("metadata",)
    explicit = user_config(profile, **RESOURCES)
    explicit["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"] = "/user/other.parquet"
    assert load(root, explicit, tmp_path / "user").resource_sources == ("user",)


def _refresh_prediction_manifest(root):
    document = json.loads((root / MANIFEST).read_text())
    for artifact in document["artifacts"][:2]:
        path = root / artifact["path"]
        artifact.update(sha256=_sha(path.read_bytes()), bytes=path.stat().st_size)
        if artifact["role"] == "fpm-library":
            artifact["rows"] = pq.ParquetFile(path).metadata.num_rows
    (root / MANIFEST).write_text(json.dumps(document))


def _write_entry_catalog(root):
    manifest = Manifest.model_validate_json((root / MANIFEST).read_bytes())
    (root / "catalog/fpm").mkdir(parents=True, exist_ok=True)
    index = {
        "schema_version": "aisim-category-index/v1",
        "category": "fpm",
        "entries": [
            {
                "configuration_id": manifest.configuration_id,
                "purpose": manifest.purpose,
                "identity": manifest.identity.model_dump(),
                "manifest": _artifact(root, MANIFEST, "manifest", "json", "aisim-artifact-manifest/v1"),
            }
        ],
    }
    (root / "catalog/fpm/index.json").write_text(json.dumps(index))
    (root / "catalog/dataset.json").write_text(
        json.dumps(
            {
                "schema_version": "aisim-dataset-index/v1",
                "dataset_id": REPO,
                "categories": {
                    "fpm": {
                        "root": "data/fpm",
                        "layout": "category",
                        "index": _artifact(root, "catalog/fpm/index.json", "index", "json", "aisim-category-index/v1"),
                    }
                },
            }
        )
    )


@pytest.mark.parametrize("setting", [["bad"], "bad", 1])
def test_ensure_rejects_a_non_mapping_estimator_with_a_field_error(root, profile, tmp_path, setting):
    from aisim_core.data.ensure import DataPin, MissingDataError, ensure_data

    make_entry(root, profile)
    _write_entry_catalog(root)
    config = {**user_config(profile), "estimator_config": setting}
    with pytest.raises(MissingDataError, match="estimator_config must be a mapping"):
        ensure_data(config, DataPin(local_root=root, categories=("fpm",), cache_dir=tmp_path / "cache"))


def _active_systems_root(tmp_path, monkeypatch, marker):
    """An active systems root with a system definition tagged ``marker`` and an operation table that only it holds."""
    from aisim_core.data.fpm_library import _stageable_roots
    from aisim_core.sdk import perf_database

    active = tmp_path / "active"
    (active / "data/h200_sxm/vllm/0.25.1").mkdir(parents=True)
    (active / "data/h200_sxm/vllm/0.25.1/packaged_only.txt").write_text("packaged")
    yaml = (_stageable_roots(["default"], None)[0] / "h200_sxm.yaml").read_text()
    monkeypatch.setattr(perf_database, "get_systems_paths", lambda: [str(active)])

    def write(tag):
        (active / "h200_sxm.yaml").write_text(yaml + f"\n# fixture active root: {tag}\n")

    write(marker)
    return active, write


def test_a_source_definition_edited_in_place_gets_its_own_staging(root, profile, tmp_path, monkeypatch):
    from aisim_core.data.ensure import DataPin, ensure_data

    make_entry(root, profile, resources=RESOURCES)
    _write_entry_catalog(root)
    _, write = _active_systems_root(tmp_path, monkeypatch, "old")
    pin = DataPin(local_root=root, categories=("fpm",), cache_dir=tmp_path / "cache")
    first = ensure_data(user_config(profile), pin)
    write("new")  # the same path now holds another system definition
    second = ensure_data(user_config(profile), pin)
    staged_first, staged_second = (Path(c.config["systems_paths"][0]) for c in (first, second))
    assert staged_first != staged_second
    # The configuration returned earlier still reads the definition it was built from.
    assert (staged_first / "h200_sxm.yaml").read_text().endswith("# fixture active root: old\n")
    assert (staged_second / "h200_sxm.yaml").read_text().endswith("# fixture active root: new\n")
    again = ensure_data(user_config(profile), pin)  # unchanged inputs reuse the staging
    assert Path(again.config["systems_paths"][0]) == staged_second


def test_a_corrected_pair_under_new_file_names_gets_its_own_staging(root, profile, tmp_path):
    from aisim_core.data.ensure import DataPin, ensure_data

    make_entry(root, profile, resources=RESOURCES)
    _write_entry_catalog(root)
    pin = DataPin(local_root=root, categories=("fpm",), cache_dir=tmp_path / "cache")
    first = ensure_data(user_config(profile), pin)
    # Publish a corrected pair beside the old files and re-point the manifest and index; the ID does not change.
    parquet, sidecar = _stage_args(root)
    rows = pq.read_table(parquet).to_pylist()
    for row in rows:
        if row["workload_kind"] == "decode":
            row["latency_ms"] = 7.0
    corrected = parquet.with_name("fpm_forward_perf_v2.parquet")
    pq.write_table(pa.Table.from_pylist(rows), corrected)
    metadata = json.loads(sidecar.read_text())
    metadata["parquet_sha256"] = _sha(corrected.read_bytes())
    sidecar.with_name("fpm_forward_perf_v2.metadata.json").write_text(json.dumps(metadata))
    document = json.loads((root / MANIFEST).read_text())
    for artifact in document["artifacts"][:2]:
        artifact["path"] = artifact["path"].replace("fpm_forward_perf.", "fpm_forward_perf_v2.")
    (root / MANIFEST).write_text(json.dumps(document))
    _refresh_prediction_manifest(root)
    _write_entry_catalog(root)
    second = ensure_data(user_config(profile), pin)
    assert Path(first.config["systems_paths"][0]) != Path(second.config["systems_paths"][0])
    query = [{"scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": 1}}]
    model_a = RustForwardPassPerfModel.best_available(first.config)
    model_b = RustForwardPassPerfModel.best_available(second.config)
    try:
        assert model_a.estimate_forward_pass_time_ms(query) == pytest.approx(3.0)  # still the pair it was built from
        assert model_b.estimate_forward_pass_time_ms(query) == pytest.approx(7.0)
    finally:
        model_a.close()
        model_b.close()


def test_the_key_and_the_staging_read_the_same_active_roots(root, profile, tmp_path, monkeypatch):
    from aisim_core.data.ensure import DataPin, ensure_data
    from aisim_core.sdk import perf_database

    make_entry(root, profile, resources=RESOURCES)
    _write_entry_catalog(root)
    first, second = (_systems_root_with_tables(tmp_path, name, {f"gemm/vllm/0.25.1/{name}.txt": b"x"}) for name in "ab")
    reads = []

    def changing_roots():  # the active roots change after the first read
        reads.append(1)
        return [str(first if len(reads) == 1 else second)]

    monkeypatch.setattr(perf_database, "get_systems_paths", changing_roots)
    out = ensure_data(user_config(profile), DataPin(local_root=root, categories=("fpm",), cache_dir=tmp_path / "c"))
    staged = Path(out.config["systems_paths"][0])
    assert (staged / "data/h200_sxm/gemm/vllm/0.25.1/a.txt").exists()  # the roots the key describes
    assert not (staged / "data/h200_sxm/gemm/vllm/0.25.1/b.txt").exists()  # not a root read after it


def test_a_malformed_timing_draft_leaves_the_previous_staging_intact(root, profile, tmp_path):
    make_entry(root, profile)
    first = load(root, user_config(profile), tmp_path)
    marker = first.systems_root / ".staged.json"
    inode = marker.stat().st_ino
    draft = {**user_config(profile), "fpm_profile": {"deployments": [{}]}}  # no schema, model or resources
    with pytest.raises(FpmContractError, match="invalid fpm_profile"):
        load(root, draft, tmp_path)
    assert marker.stat().st_ino == inode and first._staged.parquet_path.exists()
    assert load(root, user_config(profile), tmp_path).kind == "library"  # a valid draft still loads for timing


def test_repin_drops_the_generated_roots_of_an_earlier_pin(root, profile, tmp_path, monkeypatch):
    from aisim_core.data.ensure import DataPin, ensure_data

    other = tmp_path / "other"
    for source in (root, other):
        make_entry(source, profile, resources=RESOURCES)
        _write_entry_catalog(source)
    _, write = _active_systems_root(tmp_path, monkeypatch, "old")
    config = user_config(profile)
    first = ensure_data(config, DataPin(local_root=root, categories=("fpm",), cache_dir=tmp_path / "cache"))
    write("new")  # the active system definition changes before the saved configuration is repinned
    pin = DataPin(local_root=other, categories=("fpm",), cache_dir=tmp_path / "cache")
    repinned, fresh = ensure_data(first.config, pin), ensure_data(config, pin)
    staged = Path(fresh.config["systems_paths"][0])
    assert (staged / "h200_sxm.yaml").read_text().endswith("# fixture active root: new\n")
    assert repinned.config["systems_paths"] == fresh.config["systems_paths"]
    assert Path(first.config["systems_paths"][0]) not in map(Path, repinned.config["systems_paths"])


def test_op_latency_pin_keeps_the_fpm_staging_off_other_roots(root, profile, tmp_path, monkeypatch):
    from aisim_core.data import ensure as ensure_mod
    from aisim_core.data.ensure import DataPin, ensure_data
    from aisim_core.data.fpm_library import _stageable_roots
    from aisim_core.data.op_tree import OpTreeLoad

    make_entry(root, profile, resources=RESOURCES)
    _write_entry_catalog(root)
    manifest = Manifest.model_validate_json((root / MANIFEST).read_bytes())
    identity = manifest.identity.model_copy(update={"model": None, "operation_family": "gemm"})
    cfg_id = configuration_id("op-latency", "operation-tables", identity)
    (root / "catalog/op-latency").mkdir(parents=True)
    (root / "catalog/op-latency/index.json").write_text(
        json.dumps(
            {
                "schema_version": "aisim-category-index/v1",
                "category": "op-latency",
                "entries": [
                    {
                        "configuration_id": cfg_id,
                        "purpose": "operation-tables",
                        "identity": identity.model_dump(),
                        "manifest": _artifact(root, MANIFEST, "manifest", "json", "aisim-artifact-manifest/v1"),
                    }
                ],
            }
        )
    )
    dataset = json.loads((root / "catalog/dataset.json").read_text())
    dataset["categories"]["op-latency"] = {
        "root": "data/op-latency",
        "layout": "category",
        "index": _artifact(root, "catalog/op-latency/index.json", "index", "json", "aisim-category-index/v1"),
    }
    (root / "catalog/dataset.json").write_text(json.dumps(dataset))
    # The pinned root holds the system definition but not the table that only the active root has.
    pinned = tmp_path / "pinned/systems"
    (pinned / "data/h200_sxm/vllm/0.25.1").mkdir(parents=True)
    shutil.copyfile(_stageable_roots(["default"], None)[0] / "h200_sxm.yaml", pinned / "h200_sxm.yaml")
    _active_systems_root(tmp_path, monkeypatch, "active")
    monkeypatch.setattr(
        ensure_mod,
        "_load_op_latency",
        lambda *args, **kwargs: (OpTreeLoad(pinned, 1, 0, 0, (), frozenset(), 0.0), [cfg_id]),
    )
    for categories in (("op-latency", "fpm"), ("fpm", "op-latency")):
        pin = DataPin(local_root=root, categories=categories, cache_dir=tmp_path / "cache")
        result = ensure_data(user_config(profile), pin)
        staged, *rest = map(Path, result.config["systems_paths"])
        assert rest == [pinned.resolve()] or rest == [pinned]
        assert not (staged / "data/h200_sxm/vllm/0.25.1/packaged_only.txt").exists()
        assert (staged / "h200_sxm.yaml").read_text() == (pinned / "h200_sxm.yaml").read_text()


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("new_resources", [None, {**RESOURCES, "max_num_tokens": 1024}])
def test_repin_resolves_metadata_resources_and_preserves_user_blocks(root, profile, tmp_path, explicit, new_resources):
    from aisim_core.data.ensure import DataPin, ensure_data

    other = tmp_path / "other"
    for source, resources in ((root, RESOURCES), (other, new_resources)):
        make_entry(source, profile, resources=resources)
        _write_entry_catalog(source)
    config = user_config(profile, **(RESOURCES if explicit else {}))
    config["fpm_profile"]["deployments"].append(
        {
            **copy.deepcopy(config["fpm_profile"]["deployments"][0]),
            "system": "b200_sxm",
            "resources": {**RESOURCES, "max_num_tokens": 2048, "provenance": "Explicit other-deployment fixture."},
        }
    )
    first = ensure_data(config, DataPin(local_root=root, categories=("fpm",), cache_dir=tmp_path / "cache"))
    # Include native normalization and JSON persistence: derived resources must retain their origin.
    saved = json.loads(json.dumps(RustForwardPassPerfModel.normalize_config(first.config)))
    assert saved["fpm_profile"]["deployments"][0].get("resources_source", "user") == (
        "user" if explicit else "metadata"
    )
    second = ensure_data(saved, DataPin(local_root=other, categories=("fpm",), cache_dir=tmp_path / "cache"))
    deployment = second.config["fpm_profile"]["deployments"][0]
    assert second.config["fpm_profile"]["deployments"][1]["resources"]["max_num_tokens"] == 2048
    if explicit or new_resources is not None:
        assert deployment["resources"]["max_num_tokens"] == (8192 if explicit else 1024)
    else:
        assert "resources" not in deployment
        assert "resources_source" not in deployment


def test_restaging_changed_library_does_not_reuse_a_live_native_table(root, profile, tmp_path):
    make_entry(root, profile)
    config = user_config(profile)
    first = load(root, config, tmp_path)
    model_a = RustForwardPassPerfModel.best_available(first.config)
    query = [{"scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": 1}}]
    try:
        # Synthetic fixture cell: 3 ms before replacement, 7 ms afterwards.
        assert model_a.estimate_forward_pass_time_ms(query) == pytest.approx(3.0)
        other = tmp_path / "other"
        make_entry(other, profile)
        parquet, sidecar = _stage_args(other)
        rows = pq.read_table(parquet).to_pylist()
        for row in rows:
            if row["workload_kind"] == "decode":
                row["latency_ms"] = 7.0
        pq.write_table(pa.Table.from_pylist(rows), parquet)
        metadata = json.loads(sidecar.read_text())
        metadata["parquet_sha256"] = _sha(parquet.read_bytes())
        sidecar.write_text(json.dumps(metadata))
        _refresh_prediction_manifest(other)
        second = load(other, config, tmp_path)
        assert second.systems_root == first.systems_root
        model_b = RustForwardPassPerfModel.best_available(second.config)
        try:
            assert model_b.estimate_forward_pass_time_ms(query) == pytest.approx(7.0)
            assert model_a.estimate_forward_pass_time_ms(query) == pytest.approx(3.0)
            inode = second._staged.parquet_path.lstat().st_ino
            again = load(other, config, tmp_path)
            assert again._staged.parquet_path.lstat().st_ino == inode
        finally:
            model_b.close()
    finally:
        model_a.close()


@pytest.mark.parametrize("policy", ["unknown_v1", None, "omitted"])
def test_effective_sidecar_policy_must_be_native_v6(root, profile, tmp_path, policy):
    make_entry(root, profile)
    parquet, sidecar = _stage_args(root)
    metadata = json.loads(sidecar.read_text())
    if policy == "omitted":
        metadata.pop("measurement_policy")
    else:
        metadata["measurement_policy"] = policy
    sidecar.write_text(json.dumps(metadata))
    _refresh_prediction_manifest(root)
    findings = check_fpm_entry(root, MANIFEST)
    assert [f.code for f in findings] == ["invalid-sidecar"]
    assert "measurement_policy" in findings[0].message
    with pytest.raises(FpmContractError, match="measurement_policy"):
        stage_fpm_library(parquet, sidecar, **KWARGS, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_sidecar_digest_fails_model_publication_and_staging(root, profile, tmp_path):
    make_entry(root, profile)
    parquet, sidecar = _stage_args(root)
    metadata = json.loads(sidecar.read_text())
    metadata.pop("parquet_sha256")
    schema = json.loads(render_snapshot("aic_fpm_forward_perf.metadata/v6"))
    with pytest.raises(ValidationError, match="parquet_sha256"):
        Draft202012Validator(schema).validate(metadata)
    with pytest.raises(ValueError, match="parquet_sha256"):
        FpmLibraryMetadata.model_validate(metadata)
    sidecar.write_text(json.dumps(metadata))
    _refresh_prediction_manifest(root)
    findings = check_fpm_entry(root, MANIFEST)
    assert [f.code for f in findings] == ["invalid-sidecar"]
    assert "parquet_sha256" in findings[0].message
    with pytest.raises(FpmContractError, match="parquet_sha256"):
        stage_fpm_library(parquet, sidecar, **KWARGS, output_dir=tmp_path / "out")


@pytest.mark.parametrize("selector", [None, "omitted"])
def test_sidecar_selector_omission_and_null_match_native_admission(root, profile, tmp_path, selector):
    make_entry(root, profile)
    parquet, sidecar = _stage_args(root)
    metadata = json.loads(sidecar.read_text())
    metadata.pop("configuration_selector")
    if selector is None:
        metadata["configuration_selector"] = None
    schema = json.loads(render_snapshot("aic_fpm_forward_perf.metadata/v6"))
    sidecar.write_text(json.dumps(metadata))
    _refresh_prediction_manifest(root)
    if selector is None:
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate(metadata)
        with pytest.raises(FpmContractError, match="configuration_selector"):
            read_library_metadata(sidecar)
        assert [f.code for f in check_fpm_entry(root, MANIFEST)] == ["invalid-sidecar"]
    else:
        Draft202012Validator(schema).validate(metadata)
        assert read_library_metadata(sidecar) == metadata
        assert check_fpm_entry(root, MANIFEST) == []
        data = load(root, user_config(profile), tmp_path)
        model = RustForwardPassPerfModel.best_available(data.config)
        try:
            assert model.estimate_forward_pass_time_ms(
                [{"scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 1}}]
            ) == pytest.approx(2.0)
        finally:
            model.close()


@pytest.mark.parametrize("field", ["batch_size", "total_prefill_tokens", "total_kv_read_tokens"])
@pytest.mark.parametrize("value", [2**32 - 1, 2**32])
def test_workload_coordinate_u32_bounds_match_model_schema_and_publication(root, profile, field, value):
    from aisim_core.data.fpm_library import FpmLibraryRow

    make_entry(root, profile)
    parquet, sidecar = _stage_args(root)
    rows = pq.read_table(parquet).to_pylist()
    row = next(row for row in rows if row["workload_kind"] == "prefill")
    row[field] = value
    schema = json.loads(render_snapshot("aic_fpm_forward_perf/v6"))
    if value == 2**32 - 1:
        Draft202012Validator(schema).validate(row)
        assert getattr(FpmLibraryRow.model_validate(row), field) == value
    else:
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate(row)
        with pytest.raises(ValueError, match=field):
            FpmLibraryRow.model_validate(row)
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    metadata = json.loads(sidecar.read_text())
    metadata["parquet_sha256"] = _sha(parquet.read_bytes())
    sidecar.write_text(json.dumps(metadata))
    _refresh_prediction_manifest(root)
    findings = check_fpm_entry(root, MANIFEST)
    assert [f.code for f in findings] == ([] if value == 2**32 - 1 else ["library-coordinates"])


@pytest.mark.parametrize("dcp", [True, 0, 2**32, 1.0, "1"])
def test_sidecar_rejects_invalid_declared_dcp_selector(dcp):
    metadata = {
        "schema_name": "aic_fpm_forward_perf",
        "schema_version": 6,
        "row_count": 1,
        "parquet_sha256": "0" * 64,
        "coordinate_system": "iteration_totals_balanced_v1",
        "configuration_selector": {"dcp": dcp},
    }
    with pytest.raises(ValueError, match="dcp"):
        FpmLibraryMetadata.model_validate(metadata)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"coordinate_system": None}, "coordinate_system"),
        ({"system": None}, "system is required"),
        ({"backend": "other"}, "backend 'other' differs"),
        ({"backend_version": "9"}, "backend_version '9' differs"),
    ],
)
def test_effective_sidecar_must_match_native_admission(mutation, message):
    metadata = {
        "measurement_policy": "dynamo_native_single_sample_v1",
        "coordinate_system": "iteration_totals_balanced_v1",
        "system": "s",
        "backend": "vllm",
        "backend_version": "1",
        **mutation,
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}
    with pytest.raises(FpmContractError, match=message):
        check_effective_measurement_policy(metadata, system="s", backend="vllm", backend_version="1")


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("with_inventory", [False, True])
def test_shared_library_resources_need_a_selector_naming_the_model(root, profile, tmp_path, named, with_inventory):
    selector = {"tp": 4, "dp": 1, "moe_tp": 4, **({"model_path": "MiniMaxAI/MiniMax-M2.7"} if named else {})}
    make_entry(root, profile, resources=RESOURCES, sidecar_extra={"configuration_selector": selector})
    table = pq.read_table(root / ENTRY / "fpm/fpm_forward_perf.parquet")
    models = ["MiniMaxAI/MiniMax-M2.7"] * (table.num_rows - 1) + ["other/model"]
    inventory = {"model_paths": sorted(set(models))} if with_inventory else {}
    _republish_library(root, _with_column(table, "model_path", models), configuration_selector=selector, **inventory)
    for config in (user_config(profile), dict(CONFIG)):  # with and without a requested profile
        data = load(root, config, tmp_path / "shared")
        sources = set(data.resource_sources or (data.resources.source,))
        assert sources == ({"metadata"} if named else {"pending"})
        if not named:
            with pytest.raises(FpmResourcesPendingError):
                load(root, config, tmp_path / "required", require_resources=True)
    # Explicit user resources keep working for a shared library.
    assert load(root, user_config(profile, **RESOURCES), tmp_path / "user").resource_sources == ("user",)


def test_library_without_profile_builds_through_the_default_estimator_search(root, profile):
    make_entry(root, profile, resources=RESOURCES)
    data = load(root, dict(CONFIG), root.parent / "out")
    assert data.config["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"].endswith(".parquet")
    model = RustForwardPassPerfModel.best_available(data.config)
    saved = model.diagnostics()["provenance"]["config"]
    # The op-level engine is compiled without the library path; the saved canonical configuration keeps the pin.
    assert saved["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"] == str(
        data.config["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"]
    )
    # An explicitly selected estimator that never reads a library is not handed one.
    for mode in ("op_level", "fpm_regression"):
        explicit = load(root, {**CONFIG, "estimation_mode": mode}, root.parent / f"out-{mode}")
        assert "fpm_parquet_path" not in explicit.config.get("estimator_config", {}).get("fpm_interpolation", {})
    interpolation = load(root, {**CONFIG, "estimation_mode": "fpm_interpolation"}, root.parent / "out-fpm")
    assert interpolation.config["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"].endswith(".parquet")
