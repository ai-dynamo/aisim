# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FPM profile compatibility exports and runtime-facing validation bridges."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from aisim_core.fpm_profile import _MUTABLE_REFERENCES, _load_fpm_timing_profile
from aisim_core.fpm_profile import FpmCacheGroup as FpmCacheGroup
from aisim_core.fpm_profile import FpmDeploymentProfile as FpmDeploymentProfile
from aisim_core.fpm_profile import FpmMambaCache as FpmMambaCache
from aisim_core.fpm_profile import FpmModelProfile as FpmModelProfile
from aisim_core.fpm_profile import FpmResourceProfile as FpmResourceProfile
from aisim_core.fpm_profile import FpmRuntimeMemoryProfile as FpmRuntimeMemoryProfile
from aisim_core.fpm_profile import FpmSpeculativeExecutionProfile as FpmSpeculativeExecutionProfile
from aisim_core.fpm_profile import load_fpm_profile as load_fpm_profile


def _fpm_kda_cache_dtype(deployment: FpmDeploymentProfile, field: str) -> str | None:
    """Read unanimous effective cache precision from probe, collection or finalized evidence.

    Finalization retains each cell's provenance, including when raw artifacts
    are replaced by hash-bound source references. Missing or conflicting cells
    must not establish an analytical element width.
    """

    def from_provenance(provenance):
        try:
            evidence = json.loads(provenance)
        except (ValueError, TypeError):
            return None
        return from_evidence(evidence)

    def from_evidence(evidence):
        if not isinstance(evidence, dict):
            return None
        if "cell_observations" in evidence:
            cells = evidence["cell_observations"]
            if not isinstance(cells, list):
                return None
            recorded = [
                from_provenance(cell.get("runtime_memory", {}).get("provenance"))
                if isinstance(cell, dict) and isinstance(cell.get("runtime_memory"), dict)
                else None
                for cell in cells
            ]
        elif evidence.get("source") == "bound_collection_phases":
            phases = evidence.get("phases")
            recorded = [from_evidence(phase) for phase in phases.values()] if isinstance(phases, dict) else []
        else:
            settings = evidence.get("runtime_settings")
            cache = settings.get("cache_config") if isinstance(settings, dict) else None
            dtype = cache.get(field) if isinstance(cache, dict) else None
            if dtype == "auto":
                if field == "mamba_ssm_cache_dtype":
                    from collector.fpm_forward.runtime_k31 import is_audited_runtime

                    return "float32" if is_audited_runtime(evidence.get("runtime")) else "auto"
                model = settings.get("model_config")
                dtype = model.get("dtype") if isinstance(model, dict) else None
            return dtype.removeprefix("torch.") if isinstance(dtype, str) else None
        return recorded[0] if recorded and recorded[0] is not None and all(d == recorded[0] for d in recorded) else None

    runtime = deployment.resources.runtime_memory
    return from_provenance(runtime.provenance) if runtime is not None else None


def _fpm_kda_state_dtype(deployment: FpmDeploymentProfile) -> str | None:
    return _fpm_kda_cache_dtype(deployment, "mamba_ssm_cache_dtype")


def _require_forward_pass_profile_memory(config: dict) -> None:
    """Check simulation readiness after canonical identity normalization.

    Standalone timing queries can use pending resources; simulation consumers
    call this before loading timings so a new model gets a memory finalization
    diagnostic even when collection has not produced any timing data yet.
    """
    if config.get("fpm_profile") is None:
        return
    deployment = _load_fpm_timing_profile(config["fpm_profile"]).select(
        model=config["model"],
        system=config["system"],
        backend=config["backend"],
        backend_version=config["backend_version"],
        worker_type=config.get("worker_type", "aggregated"),
        tp_size=config["tp"],
        pp_size=config["pp"],
        attention_dp_size=config["attention_dp"],
        moe_tp_size=config.get("moe_tp_size"),
        moe_ep_size=config.get("moe_ep_size"),
    )
    deployment.require_resources().require_memory()


def _quantization_from_engine_config(config_json: str) -> dict[str, str]:
    """Restore exact profile modes after checking the native wire dtypes.

    EngineConfig stores dtypes rather than quantization modes: ``sq`` and
    ``int8_wo`` both serialize as ``int8``. Both native timing and memory
    bridges use the profile to recover the mode, but only when every supplied
    wire dtype agrees with the selected deployment's canonical encoding.
    """
    from aisim_core.sdk.rust_engine_step import _moe_quant_to_dtype, _quant_to_dtype

    config = json.loads(config_json)
    profile = _load_fpm_timing_profile(config["extra"]["fpm_profile"])
    deployment = profile.select(
        model=config["model_name"],
        system=config["system_name"],
        backend=config["backend"],
        backend_version=config["backend_version"],
        worker_type=config["extra"].get("worker_type", "aggregated"),
        tp_size=config["tp_size"],
        pp_size=config["pp_size"],
        attention_dp_size=1 if config.get("attention_dp_size") is None else config["attention_dp_size"],
        moe_tp_size=config.get("moe_tp_size"),
        moe_ep_size=config.get("moe_ep_size"),
        cp_size=1 if config.get("cp_size") is None else config["cp_size"],
    )
    modes = deployment.quantization_kwargs()
    for field, mode, encode in (
        ("weight_dtype", "gemm_quant_mode", _quant_to_dtype),
        ("moe_dtype", "moe_quant_mode", _moe_quant_to_dtype),
        ("activation_dtype", "fmha_quant_mode", _quant_to_dtype),
        ("kv_cache_dtype", "kvcache_quant_mode", _quant_to_dtype),
    ):
        supplied = config.get(field)
        expected = encode(modes[mode])
        if supplied is not None and supplied != expected:
            raise ValueError(
                f"FPM profile identity conflict: {field}={supplied!r}, "
                f"profile {mode}={modes[mode]!r} requires wire dtype {expected!r}"
            )
    # Native EngineConfig may repeat the recorded backend identity. Validate
    # it here before the memory bridge omits it as a user override.
    supplied_backend = config["extra"].get("moe_backend")
    if supplied_backend is not None and supplied_backend != deployment.moe_backend:
        raise ValueError(
            f"FPM profile identity conflict: moe_backend={supplied_backend!r}, "
            f"profile moe_backend={deployment.moe_backend!r}"
        )
    supplied_comm = config["extra"].get("comm_quant_mode")
    if supplied_comm is not None and supplied_comm != modes["comm_quant_mode"]:
        raise ValueError(
            f"FPM profile identity conflict: comm_quant_mode={supplied_comm!r}, "
            f"profile comm_quant_mode={modes['comm_quant_mode']!r}"
        )
    supplied_attention = config["extra"].get("attention_backend")
    if supplied_attention is not None and supplied_attention != modes["attention_backend"]:
        raise ValueError(
            f"FPM profile identity conflict: attention_backend={supplied_attention!r}, "
            f"profile attention_backend={modes['attention_backend']!r}"
        )
    modes.pop("attention_backend")
    return modes


@cache
def _load_pinned_checkpoint_config(model: str, revision: str) -> dict | None:
    """Reuse immutable remote metadata across candidate normalization and replay."""
    from aisim_core.sdk.utils import _download_hf_json

    return _download_hf_json(model, "config.json", revision=revision)


def _validate_checkpoint_architecture(profile: FpmModelProfile) -> str:
    """Verify raw checkpoint metadata without loading weights or remote code."""
    from aisim_core.decoder_config import decoder_config_key, decoder_config_problem
    from aisim_core.sdk.utils import HuggingFaceDownloadError, _load_local_config

    checkpoint = Path(profile.model).expanduser()
    identity = f"{profile.model!r} at model_revision={profile.model_revision!r}"
    try:
        if checkpoint.exists() or checkpoint.is_absolute() or profile.model.startswith(("./", "../", "~")):
            metadata = _load_local_config(str(checkpoint))
        else:
            # Bundled configs do not record their revision and cannot establish
            # the architecture of this pinned checkpoint.
            metadata = _load_pinned_checkpoint_config(profile.model, profile.model_revision)
    except (HuggingFaceDownloadError, OSError, ValueError) as error:
        raise ValueError(
            f"cannot verify FPM profile architecture: config.json metadata for {identity} "
            f"could not be resolved; make the pinned checkpoint configuration available: {error}"
        ) from error
    if not isinstance(metadata, dict):
        raise ValueError(f"FPM profile checkpoint config.json must be a mapping for {identity}")
    wrapper_architectures = metadata.get("architectures")
    text_config = metadata.get("text_config")
    # Match the collector's text-model view of checkpoint metadata. A decoder that declares no architectures,
    # absent or null, keeps the wrapper's, as the collector and config intake read it. A decoder key that does not
    # hold one text decoder is refused, as the collector and config intake refuse it.
    key = decoder_config_key(metadata)
    decoder = metadata.get(key)
    if key is not None and (problem := decoder_config_problem(key, decoder)) is not None:
        raise ValueError(f"FPM profile checkpoint config.json for {identity} has no single text decoder: {problem}")
    if isinstance(decoder, dict):
        merged = {**metadata, **decoder}
        if decoder.get("architectures") is None:
            merged["architectures"] = metadata.get("architectures")
        metadata = merged
    architectures = metadata.get("architectures")
    if (
        not isinstance(architectures, list)
        or not architectures
        or not isinstance(architectures[0], str)
        or not architectures[0].strip()
    ):
        raise ValueError(
            f"FPM profile checkpoint config.json must declare a nonempty architectures list for {identity}"
        )
    if profile.architecture != architectures[0]:
        raise ValueError(
            f"FPM profile architecture mismatch for {identity}: "
            f"checkpoint={architectures[0]!r}, profile={profile.architecture!r}"
        )
    # Profile identity describes the decoder. The analytical SDK retains a
    # registered multimodal wrapper when it unwraps text_config; check that
    # same class for SOL availability without changing the profile identity.
    from aisim_core.sdk.common import MULTIMODAL_TEXT_CONFIG_KEY

    if (
        isinstance(wrapper_architectures, list)
        and wrapper_architectures
        and isinstance(wrapper_architectures[0], str)
        and MULTIMODAL_TEXT_CONFIG_KEY.get(wrapper_architectures[0]) == "text_config"
        and isinstance(text_config, dict)
    ):
        return wrapper_architectures[0]
    return profile.architecture


def _validate_forward_pass_profile(config_json: str) -> str:
    """Validate identity against checkpoint metadata and return registration facts.

    This path reads no timing data and does not construct an analytical model.
    Estimator defaults and interpolation selection belong to Rust.
    """
    config = json.loads(config_json)
    profile = _load_fpm_timing_profile(config["fpm_profile"])
    from aisim_core.sdk.utils import _resolve_hf_revision

    # Rust retains this normalized profile in the model and serialized config,
    # so eligibility, graph sidecars and replay all use the same Hub commit.
    profile = profile.model_copy(update={"model_revision": _resolve_hf_revision(profile.model, profile.model_revision)})
    version = config.get("backend_version")
    if not isinstance(version, str) or version.lower() in _MUTABLE_REFERENCES | {"previous", "next"}:
        raise ValueError("fpm_profile requires a literal backend_version")
    deployment = profile.select(
        model=config["model"],
        system=config["system"],
        backend=config["backend"],
        backend_version=version,
        worker_type=config["worker_type"],
        tp_size=config["tp"],
        pp_size=config["pp"],
        attention_dp_size=config["attention_dp"],
        moe_tp_size=config.get("moe_tp_size"),
        moe_ep_size=config.get("moe_ep_size"),
    )
    deployment.validate_overrides(**{key: config.get(key) for key in deployment.quantization_kwargs()})
    analytical_architecture = _validate_checkpoint_architecture(profile)
    text_only = config.get("estimator_config", {}).get("fpm_interpolation", {}).get("text_only", False)
    if not text_only:
        # Preserve existing decoder-profile auto selection until the caller
        # explicitly requests the wrapper's language-only analytical graph.
        analytical_architecture = profile.architecture

    from aisim_core.sdk import common
    from aisim_core.sdk.models.base import _MODEL_REGISTRY

    family = common.ARCHITECTURE_TO_MODEL_FAMILY.get(analytical_architecture, analytical_architecture)
    # Qwen3.5 emits GDN and NemotronH emits Mamba2, neither of which has an
    # FPM SOL roofline. This applies to wrapper and decoder profiles alike.
    # Auto can use direct interpolation; explicit SOL fails during validation.
    unsupported_sol_graph = family in {"QWEN35", "NEMOTRONH"}
    unsupported_sol_reason = None
    if analytical_architecture == "KimiK31ForConditionalGeneration":
        # The fused decode graph models BF16 convolution state. FP32 uses a
        # separate native convolution update whose traffic is not modeled.
        convolution_dtype = _fpm_kda_cache_dtype(deployment, "mamba_cache_dtype")
        unsupported_sol_graph |= convolution_dtype != "bfloat16"
        if convolution_dtype == "float32":
            unsupported_sol_reason = (
                "K3.1 SOL does not support FP32 convolution state; "
                "use estimator_config.fpm_interpolation.method='direct' or 'auto'"
            )
    state_dtype = (
        _fpm_kda_state_dtype(deployment) if analytical_architecture == "KimiK31ForConditionalGeneration" else None
    )
    unsupported_kda_state = analytical_architecture == "KimiK31ForConditionalGeneration" and state_dtype != "float32"
    unsupported_sol_graph |= analytical_architecture == "KimiK31ForConditionalGeneration" and state_dtype == "auto"
    return json.dumps(
        {
            "profile": profile.model_dump(mode="json"),
            # K2.5/K3 graphs retain MlaModule* fallback primaries whose SOL
            # rooflines are unsupported. Auto must use direct interpolation;
            # K3.1 has its own supported DSA/KDA graph.
            "registered": family in _MODEL_REGISTRY,
            "supports_sol": family in _MODEL_REGISTRY
            and analytical_architecture not in {"KimiK25ForConditionalGeneration", "KimiK3ForConditionalGeneration"}
            and not unsupported_sol_graph,
            "supports_sol_state": not unsupported_kda_state,
            "unsupported_sol_reason": unsupported_sol_reason,
            **deployment.quantization_kwargs(),
        }
    )
