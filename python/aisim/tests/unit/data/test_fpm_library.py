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
        if str(src).endswith(".staging"):
            raise PermissionError("injected final rename failure")
        return replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_install)
    with pytest.raises(PermissionError, match="final rename failure"):
        stage_fpm_library(*args, **kwargs, systems_paths=[systems])
    assert (first.systems_root / ".staged.json").read_bytes() == marker
    assert all((first.systems_root / relative).read_bytes() == raw for relative, raw in before.items())
    assert not list(tmp_path.glob(".out.*.staging"))


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
    base = {"schema_name": "aic_fpm_forward_perf", "schema_version": 6, "row_count": 2, "future_key": {"a": 1}}
    assert FpmLibraryMetadata.model_validate(base).resources is None  # entries published before `resources`
    assert FpmLibraryMetadata.model_validate({**base, "resources": RESOURCES}).resources.max_num_tokens == 8192
    with pytest.raises(ValueError, match="max_num_tokens"):
        FpmLibraryMetadata.model_validate({**base, "resources": {**RESOURCES, "max_num_tokens": 0}})
    assert load_fpm_profile  # the profile class stays the single source for resources


def test_sidecar_version_is_constrained_in_model_and_packaged_snapshot():
    from aisim_core.data.schemas import PACKAGED_SNAPSHOT_DIR, snapshot_filename

    base = {"schema_name": "aic_fpm_forward_perf", "schema_version": 6, "row_count": 2}
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


@pytest.mark.parametrize(
    "resources",
    [
        {k: v for k, v in RESOURCES.items() if k != "kv_bytes_per_token"},
        {**RESOURCES, "kv_bytes_per_token": None},
        {**RESOURCES, "cache_layout": "grouped", "cache_groups": []},
        {**RESOURCES, "cache_layout": "unmodeled"},
        {**RESOURCES, "cache_layout": "mamba_align", "runtime_memory": RUNTIME_MEMORY},
    ],
)
def test_resource_layout_requirements_agree_in_runtime_and_exported_schema(resources):
    from aisim_core.data.schemas import PACKAGED_SNAPSHOT_DIR, snapshot_filename

    schema = json.loads((PACKAGED_SNAPSHOT_DIR / snapshot_filename("aic_fpm_forward_perf.metadata/v6")).read_text())
    document = {"schema_name": "aic_fpm_forward_perf", "schema_version": 6, "row_count": 1, "resources": resources}
    with pytest.raises(ValueError):
        FpmLibraryMetadata.model_validate(document)
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(document)


def test_resource_layout_schema_keeps_valid_and_absent_resources():
    schema = json.loads(render_snapshot("aic_fpm_forward_perf.metadata/v6"))
    base = {"schema_name": "aic_fpm_forward_perf", "schema_version": 6, "row_count": 1}
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
    for document in (base, *({**base, "resources": value} for value in (RESOURCES, grouped, unmodeled, mamba))):
        Draft202012Validator(schema).validate(document)
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
