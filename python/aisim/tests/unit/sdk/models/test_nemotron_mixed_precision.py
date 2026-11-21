# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
from importlib.resources import files

import pytest

import aisim_core.sdk.operations as ops
from aisim_core.sdk import common
from aisim_core.sdk.config import ModelConfig
from aisim_core.sdk.models import get_model
from aisim_core.sdk.models.checkpoint_precision import checkpoint_precision
from aisim_core.sdk.models.nemotron_h import _moe_modes, _projection_modes
from aisim_core.sdk.utils import _load_json_with_infinity

pytestmark = pytest.mark.unit

FP8 = common.GEMMQuantMode.fp8_static
NVFP4 = common.GEMMQuantMode.nvfp4
NVFP4_WO = common.GEMMQuantMode.nvfp4_wo
BF16 = common.GEMMQuantMode.bfloat16
MOE = common.MoEQuantMode
SUPER = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
ULTRA = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
PROJECTIONS = (
    "mamba_in_proj_gemm",
    "mamba_out_proj_gemm",
    "qkv_gemm",
    "proj_gemm",
    "shared_up_gemm",
    "shared_down_gemm",
    "fc1_latent_proj_gemm",
    "fc2_latent_proj_gemm",
)
SHARED_UP = "model.layers.0.mixer.shared_experts.up_proj"


def _buckets(pipeline, names=PROJECTIONS, scale=1):
    """``{op: {mode: layers}}`` over the ops that price each projection (or ``moe``)."""
    buckets = {}
    for op in pipeline:
        name = op._name.split("_", 1)[1]
        if name in names:
            # One op per (name, precision): a duplicate would add its cost twice.
            assert op._quant_mode not in buckets.get(name, {}), (name, op._quant_mode)
            buckets.setdefault(name, {})[op._quant_mode] = op._scale_factor / scale
    return buckets


def _experts(model):
    return _buckets(model.context_ops, ("moe",)).get("moe", {})


def _groups(**groups):
    """A compressed-tensors ``config_groups`` section: ``name=(num_bits, targets)``."""
    return {
        name: {"weights": {"num_bits": bits, "type": "float"}, "targets": targets}
        for name, (bits, targets) in groups.items()
    }


def _listing(layers, **extra):
    return {"quantization_config": {"quant_algo": "MIXED_PRECISION", "quantized_layers": layers, **extra}}


def _expert_listing(algos_by_layer):
    return _listing(
        {
            f"backbone.layers.{layer}.mixer.experts.0.{proj}": {"quant_algo": algo}
            for layer, algo in algos_by_layer.items()
            if algo
            for proj in ("up_proj", "down_proj")
        }
    )


def _write_super(tmp_path, edit):
    """The published Super NVFP4 export, with ``edit`` applied to both quantization sections."""
    root = files("aisim_core") / "model_configs"
    data = json.loads((root / f"{SUPER.replace('/', '--')}_config.json").read_text())
    hf_quant = json.loads((root / f"{SUPER.replace('/', '--')}_hf_quant_config.json").read_text())
    e_layers = [i for i, kind in enumerate(data["hybrid_override_pattern"]) if kind == "E"]
    for section in (data["quantization_config"], hf_quant["quantization"]):
        edit(section, e_layers)
    (tmp_path / "config.json").write_text(json.dumps(data))
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(hf_quant))
    return str(tmp_path)


def _super_with_fp8_experts(tmp_path, fp8_layers=10):
    """Published Super with the routed experts of its first ``fp8_layers`` MoE layers listed FP8."""

    def edit(section, e_layers):
        for name, entry in section["quantized_layers"].items():
            if any(f".layers.{layer}.mixer.experts." in name for layer in e_layers[:fp8_layers]):
                entry["quant_algo"] = "FP8"

    return _write_super(tmp_path, edit)


def _task(model_path, system="gb200", version="0.31.0", **kwargs):
    from aisim.sdk.task_v2 import Task

    return Task(
        serving_mode="agg",
        model_path=model_path,
        system_name=system,
        backend_name="vllm",
        backend_version=version,
        database_mode="SOL",
        **kwargs,
    )


def _task_model(task, model_path, role="agg"):
    return get_model(model_path, task.build_model_config(role=role, parallel=(1, 1, 1, 1, 1, 1)), "vllm")


def _latencies(model, system="gb200", *, tp=1, moe_tp=1, moe_ep=1, **modes):
    """``{prefill: {op: latency_ms}}`` from the canonical API at decode batch 1 and context 1024."""
    from aisim_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    perf_model = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model=model,
            system=system,
            backend="vllm",
            backend_version="0.30.0",
            worker_type="decode",
            tp=tp,
            moe_tp_size=moe_tp,
            moe_ep_size=moe_ep,
            database_mode="SOL",
            estimation_mode="op_level",
            **{key: mode.name for key, mode in modes.items()},
        )
    )
    try:
        return {
            prefill: {
                row["name"].split("_", 1)[1]: row["latency_ms"]
                for row in perf_model.static_phase_diagnostics(batch_size=1, context_length=1024, prefill=prefill)
            }
            for prefill in (True, False)
        }
    finally:
        perf_model.close()


@pytest.mark.parametrize(
    ("raw", "pattern", "expected"),
    [
        pytest.param(
            {
                "quantization_config": {
                    "config_groups": _groups(
                        dense=(8, ["*.in_proj", "*.out_proj"]),
                        experts=(4, ["*.shared_experts.up_proj", "*.shared_experts.down_proj"]),
                    ),
                    "ignore": ["*.fc1_latent_proj", "*.fc2_latent_proj"],
                }
            },
            "ME*",
            # Group targets are not a listing, so untargeted attention keeps the default.
            {"mamba_in_proj_gemm": FP8, "shared_up_gemm": NVFP4, "fc1_latent_proj_gemm": BF16, "qkv_gemm": FP8},
            id="group-targets-and-exclusions",
        ),
        pytest.param(
            {"quantization_config": {"config_groups": _groups(all=(4, ["Linear"])), "ignore": ["*.fc1_latent_proj"]}},
            "E",
            {"shared_up_gemm": NVFP4, "fc1_latent_proj_gemm": BF16},
            id="linear-target-respects-exclusions",
        ),
        pytest.param(
            {"quantization_config": {"config_groups": _groups(one=(4, [SHARED_UP]))}},
            "E",
            {"shared_up_gemm": NVFP4, "shared_down_gemm": FP8},
            id="exact-group-target-is-not-a-listing",
        ),
        pytest.param(
            _listing({SHARED_UP: {"quant_algo": "NVFP4"}}, config_groups=_groups(all=(8, ["Linear"]))),
            "E",
            {"shared_up_gemm": NVFP4, "shared_down_gemm": BF16},
            id="listing-beats-groups-and-unlisted-is-bf16",
        ),
        pytest.param(
            {"quantization_config": {"quantized_layers": {SHARED_UP: "NVFP4"}}},
            "E",
            # Without a MIXED_PRECISION marker the listing is not known to be complete: do not guess BF16.
            {"shared_up_gemm": NVFP4, "shared_down_gemm": FP8, "fc1_latent_proj_gemm": FP8},
            id="unlisted-keeps-default-outside-the-contract",
        ),
        pytest.param(
            {
                "hf_quant_config": {
                    "quantization": {
                        "quantized_layers": {"backbone.layers.0.mixer.fc1_latent_proj": "FP8"},
                        "exclude_modules": ["backbone.layers.0.mixer.fc1_latent_proj"],
                    }
                }
            },
            "E",
            {"fc1_latent_proj_gemm": BF16},
            id="exclusion-beats-listing",
        ),
        pytest.param(
            {
                "quantization_config": {
                    "config_groups": _groups(all=(8, ["Linear"])),
                    "ignore": ["backbone.layers.0.mixer"],
                }
            },
            "MM",
            {"mamba_in_proj_gemm": {BF16: 1, FP8: 1}},
            id="parent-exclusion-covers-its-projections",
        ),
        pytest.param(
            _listing(
                {
                    **{f"model.layers.{i}.mixer.shared_experts.up_proj": {"quant_algo": "FP8"} for i in range(3)},
                    "model.layers.3.mixer.shared_experts.up_proj": {"quant_algo": "NVFP4"},
                }
            ),
            "EEEE",
            {"shared_up_gemm": {FP8: 3, NVFP4: 1}},
            id="layers-are-counted-per-precision",
        ),
        pytest.param(
            {
                "quantization_config": {
                    "config_groups": _groups(all=(8, ["Linear"])),
                    "ignore": [r"re:(?i)^BACKBONE\.layers\.0\.mixer\.in_proj"],
                }
            },
            "MM",
            {"mamba_in_proj_gemm": {BF16: 1, FP8: 1}},
            id="re-target-with-a-leading-flag",
        ),
        pytest.param(
            _listing(
                {
                    # A listed descendant declares its module; an exact listing wins over descendants.
                    "backbone.layers.0.mixer.in_proj.weight": {"quant_algo": "FP8"},
                    "backbone.layers.0.mixer.out_proj": {"quant_algo": "FP8"},
                    "backbone.layers.0.mixer.out_proj.weight": {"quant_algo": "NVFP4"},
                }
            ),
            "M",
            {"mamba_in_proj_gemm": FP8, "mamba_out_proj_gemm": FP8},
            id="listed-descendants",
        ),
    ],
)
def test_projection_modes(raw, pattern, expected):
    buckets = _projection_modes(raw, pattern, FP8)
    got = {key: buckets[key] if isinstance(want, dict) else set(buckets[key]) for key, want in expected.items()}
    assert got == {key: want if isinstance(want, dict) else {want} for key, want in expected.items()}


@pytest.mark.parametrize("dynamic,fp8", [(True, common.GEMMQuantMode.fp8), (None, FP8)])
def test_fp8_projection_lane_follows_quant_dynamic_not_the_global_default(dynamic, fp8):
    raw = _listing(
        {
            "backbone.layers.0.mixer.in_proj": {"quant_algo": "MXFP8"},
            "backbone.layers.1.mixer.in_proj": {"quant_algo": "FP8"},
        }
    )
    if dynamic is not None:
        raw["quant_dynamic"] = dynamic
    # The MXFP8 layer makes the global default fp8_block; the FP8 layer keeps its own lane.
    block = common.GEMMQuantMode.fp8_block
    assert _projection_modes(raw, "MM", block)["mamba_in_proj_gemm"] == {block: 1, fp8: 1}


@pytest.mark.parametrize("quant_method", ["compressed-tensors", "modelopt"])
@pytest.mark.parametrize(
    "scaling,gemm,moe",
    [
        ({"strategy": "block"}, common.GEMMQuantMode.fp8_block, MOE.fp8_block),
        ({"strategy": "tensor", "block_structure": [128, 128]}, common.GEMMQuantMode.fp8_block, MOE.fp8_block),
        ({"strategy": "tensor"}, common.GEMMQuantMode.fp8, MOE.fp8),
    ],
    ids=["block-strategy", "block-structure", "per-tensor-control"],
)
def test_fp8_config_group_scaling_reaches_native_pricing(monkeypatch, tmp_path, scaling, gemm, moe, quant_method):
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")

    def compressed_fp8(section, _e_layers):
        section.clear()
        section.update(
            quant_method=quant_method,
            config_groups={
                "all": {
                    "weights": {"num_bits": 8, "type": "float", **scaling},
                    "input_activations": {"num_bits": 8, "type": "float", "dynamic": True},
                    "targets": ["Linear", "*.experts.*"],
                }
            },
        )
        if quant_method == "modelopt":
            section["quant_algo"] = "MIXED_PRECISION"

    path = _write_super(tmp_path, compressed_fp8)
    (tmp_path / "hf_quant_config.json").unlink()  # config.json is self-contained
    for model in (
        get_model(path, ModelConfig(moe_tp_size=1, system="gb200"), "vllm"),
        _task_model(_task(path), path),
    ):
        assert model.config.gemm_quant_mode == gemm
        assert model.config.moe_quant_mode == moe
        for pipeline in (model.context_ops, model.generation_ops):
            assert all(set(modes) == {gemm} for modes in _buckets(pipeline).values())
            assert set(_buckets(pipeline, ("moe",))["moe"]) == {moe}

    inferred = _latencies(path)
    explicit = _latencies(path, gemm_quant_mode=gemm, moe_quant_mode=moe)
    # Every Linear target declares the same weight scaling, so inferred native
    # pricing must equal the explicit kernel family in both serving phases.
    for prefill in (True, False):
        assert inferred[prefill] == pytest.approx(explicit[prefill], rel=1e-9)


def test_config_loader_keeps_quoted_infinity_markers(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"bare": [0.0, Infinity, -Infinity, NaN], "marker": {"__float__": "Infinity"}}')
    assert _load_json_with_infinity(path) == {"bare": [0.0, None, None, None], "marker": {"__float__": "Infinity"}}


def test_untargeted_projection_takes_the_system_nvfp4_lane():
    raw = {"quantization_config": {"config_groups": _groups(a=(4, ["backbone.layers.0.mixer.in_proj"]))}}
    assert _projection_modes(raw, "MM", NVFP4, "h200_sxm")["mamba_in_proj_gemm"] == {NVFP4_WO: 2}
    # Without a system, the supplied default is kept.
    assert _projection_modes(raw, "MM", NVFP4_WO)["mamba_in_proj_gemm"] == {NVFP4: 1, NVFP4_WO: 1}


def test_regex_exclusions_keep_their_own_capture_groups():
    raw = {
        "quantization_config": {
            "config_groups": _groups(all=(8, ["Linear"])),
            "ignore": [r"re:^backbone\.(layers)\.0", r"re:^model\.(layers)\.1\.mixer\.\1"],
        }
    }
    # The second pattern's \1 is its own group: model.layers.1.mixer.layers matches it alone.
    checkpoint = checkpoint_precision(raw)
    assert checkpoint._excluded("model.layers.1.mixer.layers") and checkpoint._excluded("backbone.layers.0.x")


@pytest.mark.parametrize("prefix", ["model", "backbone", "language_model.backbone", "language_model.model"])
@pytest.mark.parametrize("key", ["quant_algo", "quantization_algo", "quant_method"])
def test_checkpoint_path_aliases_and_algo_keys(prefix, key):
    raw = _listing({f"{prefix}.layers.0.mixer.shared_experts.up_proj": {key: "NVFP4"}})
    assert _projection_modes(raw, "E", FP8)["shared_up_gemm"] == {NVFP4: 1}


@pytest.mark.parametrize(
    "modules", [("qkv_proj",), ("q_proj", "k_proj", "v_proj"), ("q_proj",), ("k_proj",), ("v_proj",)]
)
def test_qkv_precision_resolves_packed_and_agreeing_split_entries(modules):
    raw = _listing({f"model.layers.3.mixer.{name}": {"quant_algo": "FP8"} for name in modules})
    assert _projection_modes(raw, "MMM*", NVFP4)["qkv_gemm"] == {FP8: 1}


def test_packed_qkv_declaration_precedes_split_entries():
    raw = _listing(
        {
            "model.layers.0.mixer.qkv_proj": {"quant_algo": "FP8"},
            "model.layers.0.mixer.q_proj": {"quant_algo": "NVFP4"},
        }
    )
    assert _projection_modes(raw, "*", NVFP4)["qkv_gemm"] == {FP8: 1}


@pytest.mark.parametrize(
    "target",
    ["*.qkv_proj", r"re:.*\.qkv_proj$", r"re:.*\.(qkv_proj|q_proj)$", r"re:.*\.(qkv_proj|q_proj|k_proj|v_proj)$"],
)
def test_packed_qkv_pattern_target_precedes_split_names(target):
    raw = {"quantization_config": {"config_groups": _groups(qkv=(4, [target]), mamba=(8, ["*.in_proj"]))}}
    # The global mode inferred from in_proj is FP8; QKV keeps its packed NVFP4 target.
    modes = _projection_modes(raw, "M*", FP8)
    assert modes["qkv_gemm"] == {NVFP4: 1} and modes["mamba_in_proj_gemm"] == {FP8: 1}


def test_generic_target_does_not_bypass_split_qkv_exclusions():
    def raw(*excluded):
        ignore = [f"backbone.layers.0.mixer.{name}" for name in excluded]
        return {"quantization_config": {"config_groups": _groups(all=(8, ["Linear"])), "ignore": ignore}}

    # Linear also matches the unlisted packed qkv_proj; the split exclusions still win.
    assert _projection_modes(raw("q_proj", "k_proj", "v_proj"), "*", FP8)["qkv_gemm"] == {BF16: 1}
    with pytest.raises(ValueError, match="disagree on precision"):
        _projection_modes(raw("q_proj"), "*", FP8)


@pytest.mark.parametrize("projection", ["qkv_proj", "q_proj", "k_proj", "v_proj"])
def test_packed_qkv_precision_reaches_native_pricing(monkeypatch, tmp_path, projection):
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")

    def packed_attention(section, _e_layers):
        # Super's first attention layer is 7; its other seven attention layers
        # remain unlisted (BF16). This synthetic edit prices one packed FP8 QKV.
        section["quantized_layers"][f"model.layers.7.mixer.{projection}"] = {"quant_algo": "FP8"}

    path = _write_super(tmp_path, packed_attention)
    model = get_model(path, ModelConfig(moe_tp_size=1, system="gb200"), backend_name="vllm")
    assert _buckets(model.context_ops)["qkv_gemm"] == {BF16: 7, FP8: 1}
    mixed = _latencies(path)
    uniform = {mode: _latencies(path, gemm_quant_mode=mode) for mode in (BF16, FP8)}
    for prefill in (True, False):
        expected = (7 * uniform[BF16][prefill]["qkv_gemm"] + uniform[FP8][prefill]["qkv_gemm"]) / 8
        assert mixed[prefill]["qkv_gemm"] == pytest.approx(expected, rel=1e-9)


def test_overlapping_packed_qkv_target_reaches_native_pricing(monkeypatch, tmp_path):
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")

    def packed_attention(section, _e_layers):
        section.clear()
        section.update(
            quant_method="modelopt_mixed",
            quant_algo="MIXED_PRECISION",
            config_groups=_groups(
                qkv=(4, [r"re:.*\.(qkv_proj|q_proj)$"]), mamba=(8, ["*.in_proj"]), experts=(4, ["*.experts.*"])
            ),
        )

    path = _write_super(tmp_path, packed_attention)
    model = get_model(path, ModelConfig(moe_tp_size=1, system="gb200"), backend_name="vllm")
    assert model.config.gemm_quant_mode == FP8
    assert _buckets(model.context_ops)["qkv_gemm"] == {NVFP4: 8}
    mixed = _latencies(path)
    uniform = _latencies(path, gemm_quant_mode=NVFP4)
    # All eight packed QKV projections target NVFP4, despite the inferred FP8 default.
    for prefill in (True, False):
        assert mixed[prefill]["qkv_gemm"] == pytest.approx(uniform[prefill]["qkv_gemm"], rel=1e-9)


@pytest.mark.parametrize(
    ("raw", "pattern", "match"),
    [
        (
            {
                "hf_quant_config": {"quantization": {"quantized_layers": {SHARED_UP: "NVFP4"}}},
                "quantization_config": {"quantized_layers": {SHARED_UP: "FP8"}},
            },
            "E",
            "Conflicting checkpoint precisions",
        ),
        (
            _listing(
                {
                    "backbone.layers.0.mixer.in_proj.weight": {"quant_algo": "NVFP4"},
                    "backbone.layers.0.mixer.in_proj.bias": {"quant_algo": "FP8"},
                }
            ),
            "M",
            "Conflicting checkpoint precisions",
        ),
        ({"quantization_config": {"quantized_layers": {SHARED_UP: "int4_awq"}}}, "E", "Unsupported mixed precision"),
        (
            _listing(
                {
                    "backbone.layers.0.mixer.q_proj": {"quant_algo": "FP8"},
                    "backbone.layers.0.mixer.k_proj": {"quant_algo": "NVFP4"},
                }
            ),
            "*",
            "disagree on precision",
        ),
    ],
    ids=["two-sources", "descendants", "unsupported", "fused-qkv"],
)
def test_rejects_inconsistent_metadata(raw, pattern, match):
    with pytest.raises(ValueError, match=match):
        _projection_modes(raw, pattern, FP8)


@pytest.mark.parametrize("excluded", [True, False])
def test_targets_only_scope_does_not_validate_excluded_conflicts(monkeypatch, tmp_path, excluded):
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    key = "backbone.layers.0.mixer.in_proj"

    def edit(section, _e_layers):
        section["quantized_layers"][key] = {"quant_algo": "FP8"}
        section["exclude_modules"] = [key] if excluded else []

    path = _write_super(tmp_path, edit)
    hf_path = tmp_path / "hf_quant_config.json"
    hf = json.loads(hf_path.read_text())
    hf["quantization"]["quantized_layers"][key] = {"quant_algo": "NVFP4"}
    hf_path.write_text(json.dumps(hf))
    raw = json.loads((tmp_path / "config.json").read_text()) | {"hf_quant_config": hf}
    if excluded:
        assert _projection_modes(raw, "M", FP8)["mamba_in_proj_gemm"] == {BF16: 1}
        model = get_model(path, ModelConfig(tp_size=1, moe_tp_size=1, system="gb200"), "vllm")
        # One formerly FP8 layer is now excluded, irrespective of conflicting metadata.
        assert _buckets(model.context_ops)["mamba_in_proj_gemm"] == {FP8: 27, BF16: 13}
    else:
        with pytest.raises(ValueError, match="Conflicting checkpoint precisions"):
            get_model(path, ModelConfig(tp_size=1, moe_tp_size=1, system="gb200"), "vllm")


@pytest.mark.parametrize(
    ("raw", "default", "system", "expected"),
    [
        (
            _expert_listing({1: "NVFP4", 3: "FP8", 5: "NVFP4", 7: None}),
            MOE.nvfp4,
            "gb200",
            {MOE.nvfp4: 2, MOE.fp8: 1, MOE.bfloat16: 1},
        ),
        # A uniform listing keeps the global, system-resolved mode.
        (_expert_listing({1: "NVFP4", 3: "NVFP4", 5: "NVFP4", 7: "NVFP4"}), MOE.nvfp4, "gb200", {}),
        # Non-Blackwell NVFP4 runs weight-only, as the global mode does.
        (
            _expert_listing({1: "NVFP4", 3: "FP8", 5: "FP8", 7: "FP8"}),
            MOE.nvfp4_wo,
            "h200_sxm",
            {MOE.nvfp4_wo: 1, MOE.fp8: 3},
        ),
        # An untargeted layer keeps the global mode outside a targets-only export.
        (
            {"quantization_config": {"config_groups": _groups(a=(4, ["backbone.layers.1.mixer.experts.*"]))}},
            MOE.fp8,
            "gb200",
            {MOE.nvfp4: 1, MOE.fp8: 3},
        ),
        # An exclusion is explicit BF16, unlike an untargeted layer.
        (
            {
                "quantization_config": {
                    "config_groups": _groups(a=(8, ["Linear"])),
                    "ignore": ["backbone.layers.1.mixer.experts.*"],
                }
            },
            MOE.fp8,
            "gb200",
            {MOE.fp8: 3, MOE.bfloat16: 1},
        ),
        # Unlisted expert 0 reads the expert block, as vLLM does: a block listing or another expert.
        (
            _listing(
                {
                    "backbone.layers.1.mixer.experts": {"quant_algo": "FP8"},
                    "backbone.layers.3.mixer.experts.1.up_proj": {"quant_algo": "FP8"},
                    **_expert_listing({5: "NVFP4", 7: "NVFP4"})["quantization_config"]["quantized_layers"],
                }
            ),
            MOE.nvfp4,
            "gb200",
            {MOE.fp8: 2, MOE.nvfp4: 2},
        ),
    ],
)
def test_moe_modes(raw, default, system, expected):
    assert _moe_modes(raw, "MEMEMEME", default, system) == expected


@pytest.mark.parametrize("proj", ["up", "down"])
def test_a_sparse_expert_listing_resolves_the_fused_block(proj):
    # Only one of expert 0's projections listed: vLLM resolves the fused experts prefix from it.
    raw = _listing({f"backbone.layers.1.mixer.experts.0.{proj}_proj": {"quant_algo": "FP8"}})
    assert _moe_modes(raw, "MEME", MOE.nvfp4, "gb200") == {MOE.fp8: 1, MOE.bfloat16: 1}


def test_an_explicit_expert_block_listing_precedes_its_experts():
    def raw(block, expert):
        listing = {"backbone.layers.1.mixer.experts": {"quant_algo": block}}
        listing.update(_expert_listing({1: expert})["quantization_config"]["quantized_layers"])
        return _listing(listing)

    # vLLM looks the fused experts prefix up before its children.
    assert _moe_modes(raw("NVFP4", "FP8"), "ME", MOE.fp8, "gb200") == {MOE.nvfp4: 1}
    assert _moe_modes(raw("FP8", "FP8"), "ME", MOE.nvfp4, "gb200") == {MOE.fp8: 1}
    # The block entry wins before its heterogeneous experts are compared.
    mixed = raw("FP8", "NVFP4")
    mixed["quantization_config"]["quantized_layers"].update(
        {f"backbone.layers.1.mixer.experts.1.{proj}_proj": {"quant_algo": "FP8"} for proj in ("up", "down")}
    )
    assert _moe_modes(mixed, "ME", MOE.nvfp4, "gb200") == {MOE.fp8: 1}


def test_routed_experts_of_one_layer_must_share_a_precision():
    raw = _listing(
        {
            f"backbone.layers.1.mixer.experts.{expert}.{proj}_proj": {"quant_algo": algo}
            for expert, algo in ((0, "NVFP4"), (1, "FP8"))
            for proj in ("up", "down")
        }
    )
    with pytest.raises(ValueError, match="Routed experts of layer 1 disagree"):
        _moe_modes(raw, "ME", MOE.nvfp4, "gb200")
    # An excluded layer is BF16 whatever its experts list (a fresh dict: parses are cached per config).
    excluded = {"quantization_config": {**raw["quantization_config"], "ignore": ["backbone.layers.1.mixer.experts.*"]}}
    assert _moe_modes(excluded, "ME", MOE.nvfp4, "gb200") == {MOE.bfloat16: 1}
    # An exclusion that matches only the fused expert block also makes the layer BF16, as in vLLM.
    block = {"quantization_config": {**raw["quantization_config"], "ignore": ["backbone.layers.1.mixer.expert[s]"]}}
    assert _moe_modes(block, "ME", MOE.nvfp4, "gb200") == {MOE.bfloat16: 1}
    # The block exclusion wins before expert 0's partial listing is validated.
    partial_listing = _listing(
        {"backbone.layers.1.mixer.experts.0.up_proj": {"quant_algo": "FP8"}},
        ignore=["backbone.layers.1.mixer.expert[s]"],
    )
    assert _moe_modes(partial_listing, "ME", MOE.nvfp4, "gb200") == {MOE.bfloat16: 1}
    # Excluding expert 0 alone leaves expert 1 at FP8: still a mixed layer.
    partial = {"quantization_config": {**raw["quantization_config"], "ignore": ["backbone.layers.1.mixer.experts.0.*"]}}
    with pytest.raises(ValueError, match="Routed experts of layer 1 disagree"):
        _moe_modes(partial, "ME", MOE.nvfp4, "gb200")


def test_checkpoint_parse_is_indexed_and_cached(monkeypatch):
    import aisim_core.sdk.models.checkpoint_precision as precision

    layers = {f"model.layers.0.mixer.experts.{i}.up_proj": {"quant_algo": "NVFP4"} for i in range(4096)}
    raw = {"quantization_config": {"quantized_layers": {**layers, SHARED_UP: {"quant_algo": "NVFP4"}}}}
    # Literal names are dict lookups, not pattern matches.
    monkeypatch.setattr(precision, "_target_regex", lambda t: pytest.fail("pattern match on literals") if t else None)
    parsed = checkpoint_precision(raw)
    assert _projection_modes(raw, "E", FP8)["shared_up_gemm"] == {NVFP4: 1}
    assert checkpoint_precision(raw) is parsed
    # The cache keys on identity, not content, and keeps the latest 8 configs.
    assert checkpoint_precision(json.loads(json.dumps(raw))) is not parsed
    for _ in range(8):
        checkpoint_precision(_expert_listing({1: "FP8"}))
    assert checkpoint_precision(raw) is not parsed


_PUBLISHED = {
    SUPER: lambda nvfp4: {
        "mamba_in_proj_gemm": {FP8: 28, BF16: 12},
        "mamba_out_proj_gemm": {FP8: 29, BF16: 11},
        "qkv_gemm": {BF16: 8},
        "proj_gemm": {FP8: 2, BF16: 6},
        "shared_up_gemm": {FP8: 39, BF16: 1},
        "shared_down_gemm": {FP8: 37, nvfp4: 1, BF16: 2},
        "fc1_latent_proj_gemm": {FP8: 3, BF16: 37},
        "fc2_latent_proj_gemm": {FP8: 1, BF16: 39},
    },
    ULTRA: lambda nvfp4: {
        "mamba_in_proj_gemm": {FP8: 48},
        "mamba_out_proj_gemm": {FP8: 48},
        "qkv_gemm": {BF16: 12},
        "proj_gemm": {BF16: 12},
        "shared_up_gemm": {FP8: 48},
        "shared_down_gemm": {FP8: 48},
        "fc1_latent_proj_gemm": {BF16: 48},
        "fc2_latent_proj_gemm": {BF16: 48},
    },
}


@pytest.mark.parametrize("model_path", sorted(_PUBLISHED))
@pytest.mark.parametrize(("system", "nvfp4"), [("gb200", NVFP4), ("h200_sxm", NVFP4_WO)])
def test_published_exports_price_each_layer_at_its_listed_precision(model_path, system, nvfp4):
    model = get_model(model_path, ModelConfig(tp_size=2, moe_tp_size=2, system=system), backend_name="vllm")
    expected = _PUBLISHED[model_path](nvfp4)
    for pipeline, scale in ((model.context_ops, 1), (model.generation_ops, model._mtp_scale_factor)):
        assert _buckets(pipeline, scale=scale) == expected
        # Weight bytes equal a per-layer reference with one single-layer GEMM per layer.
        for name, layers in expected.items():
            gemms = [op for op in pipeline if op._name.split("_", 1)[1] == name]
            n, k = gemms[0]._n, gemms[0]._k
            reference = sum(count * ops.GEMM(name, scale, n, k, mode).get_weights() for mode, count in layers.items())
            assert sum(op.get_weights() for op in gemms) == pytest.approx(reference)
        # Every routed-expert layer is NVFP4: one MoE op on the system's lane, weight-only on Hopper.
        lane = MOE.nvfp4 if system == "gb200" else MOE.nvfp4_wo
        assert _buckets(pipeline, ("moe",), scale)["moe"] == {lane: 40 if model_path == SUPER else 48}
    # Explicit modes price every projection and expert with one mode.
    config = ModelConfig(tp_size=2, moe_tp_size=2, system=system, gemm_quant_mode=BF16, moe_quant_mode=MOE.fp8)
    explicit = get_model(model_path, config, backend_name="vllm")
    assert all(set(modes) == {BF16} for modes in _buckets(explicit.context_ops).values())
    assert set(_experts(explicit)) == {MOE.fp8}


@pytest.mark.parametrize(
    "model_path",
    [
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8",
        "nvidia/Nemotron-H-56B-Base-8K",
    ],
)
def test_models_without_usable_per_layer_metadata_keep_the_global_modes(model_path):
    model = get_model(model_path, ModelConfig(moe_tp_size=1, system="gb200"), backend_name="vllm")
    assert not model._moe_quant_modes
    for name, modes in _buckets(model.context_ops).items():
        assert set(modes) == {model.config.gemm_quant_mode}, name


@pytest.mark.parametrize("system", ["gb200", "h200_sxm"])
def test_lightning_moe_fallback_preserves_checkpoint_projection_precision(system):
    path = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    with pytest.warns(UserWarning, match="routed experts are priced at the global MoE mode"):
        model = get_model(path, ModelConfig(moe_tp_size=1, system=system), backend_name="vllm")
    for pipeline, scale in ((model.context_ops, 1), (model.generation_ops, model._mtp_scale_factor)):
        buckets = _buckets(pipeline, scale=scale)
        # The published Lightning listing declares all 23 shared-expert pairs W4A16_NVFP4.
        for name in ("shared_up_gemm", "shared_down_gemm"):
            assert buckets[name] == {common.GEMMQuantMode.w4a16_nvfp4: 23}
    with pytest.warns(UserWarning, match="W4A16_NVFP4"):
        override = get_model(path, ModelConfig(moe_tp_size=1, system=system, gemm_quant_mode=BF16), backend_name="vllm")
    assert all(set(modes) == {BF16} for modes in _buckets(override.context_ops).values())


def test_mixed_precision_latency_sums_the_per_precision_layers(monkeypatch, tmp_path):
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    path = _super_with_fp8_experts(tmp_path)
    mixed = _latencies(path)
    projections = _PUBLISHED[SUPER](NVFP4)
    experts = {MOE.fp8: 10, MOE.nvfp4: 30}
    uniform = {mode: _latencies(path, gemm_quant_mode=mode) for mode in (FP8, NVFP4, BF16)}
    uniform |= {mode: _latencies(path, moe_quant_mode=mode) for mode in experts}
    for prefill in (True, False):
        for names, counts in [((name,), layers) for name, layers in projections.items()] + [
            (("moe", "moe_pre_dispatch", "moe_post_dispatch"), experts)
        ]:
            total = sum(counts.values())
            for name in names:
                reference = sum(count / total * uniform[mode][prefill][name] for mode, count in counts.items())
                assert mixed[prefill][name] == pytest.approx(reference, rel=1e-9), (prefill, name)


@pytest.mark.parametrize("system", ["h200_sxm", "gb200"])
@pytest.mark.parametrize("mixed", [False, True])
def test_entry_points_agree_on_per_layer_experts(monkeypatch, tmp_path, system, mixed):
    from aisim_core.sdk.models.helpers import resolve_nvfp4_for_system

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    path = _super_with_fp8_experts(tmp_path) if mixed else SUPER
    # The canonical API sets the system but leaves the inferred global MoE mode native.
    canonical = get_model(path, ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, system=system), "vllm")
    # The legacy CLI estimate path infers the global mode (nvfp4_wo on Hopper) before get_model.
    legacy_config = ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1)
    resolve_nvfp4_for_system(legacy_config, system, path, backend_name="vllm")
    legacy = get_model(path, legacy_config, "vllm")
    task = _task_model(_task(path, system=system, version="0.30.0"), path)
    nvfp4 = MOE.nvfp4_wo if system == "h200_sxm" else MOE.nvfp4
    expected = {MOE.fp8: 10, nvfp4: 30} if mixed else {nvfp4: 40}
    for model in (canonical, legacy, task):
        assert _experts(model) == pytest.approx(expected)
        assert {op._quant_mode for op in model.context_ops if op._name == "context_moe_pre_dispatch"} == set(expected)
        assert _buckets(model.context_ops) == _buckets(canonical.context_ops)
    weights = [sum(op.get_weights() for op in m.context_ops if op._name == "context_moe") for m in (canonical, legacy)]
    assert weights[0] == weights[1]
    # The canonical API prices the per-layer lanes; native FP4 on Hopper would raise MissingSystemFlopsError.
    latencies = _latencies(path, system=system)
    assert all(latencies[prefill]["moe"] > 0 for prefill in (True, False))


@pytest.mark.parametrize("system", ["h200_sxm", "gb200"])
def test_task_and_legacy_estimation_price_the_mixed_experts(monkeypatch, tmp_path, system):
    from aisim.legacy_cli.api import cli_estimate
    from aisim.sdk import predict

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    path = _super_with_fp8_experts(tmp_path)
    # EP=2 exercises real dispatch/combine communication as well as expert compute.
    canonical = _latencies(path, system=system, tp=2, moe_ep=2)
    task_diagnostics = []
    original_predict = predict.predict_agg_worker

    def capture_prediction(**kwargs):
        summary = original_predict(**kwargs)
        task_diagnostics.append(summary.get_per_ops_data())
        return summary

    monkeypatch.setattr(predict, "predict_agg_worker", capture_prediction)
    task = _task(path, system=system, version="0.30.0", isl=1024, osl=2)
    task.run_single_agg(tp=2, moe_tp=1, moe_ep=2, batch_size=1, ctx_tokens=1024)
    legacy = cli_estimate(
        path,
        system,
        backend_name="vllm",
        backend_version="0.30.0",
        database_mode="SOL",
        isl=1024,
        osl=2,
        batch_size=1,
        ctx_tokens=1024,
        tp_size=2,
        moe_tp_size=1,
        moe_ep_size=2,
    )
    assert len(task_diagnostics) == 1
    for diagnostics in (task_diagnostics[0], legacy.per_ops_data):
        for prefill, phase, step in ((True, "context", "mix_step"), (False, "generation", "genonly_step")):
            for name in ("moe", "moe_pre_dispatch", "moe_post_dispatch"):
                # Require the real named rows; empty phase diagnostics cannot pass.
                expected = canonical[prefill][name]
                assert expected > 0
                assert diagnostics[step][f"{phase}_{name}"] == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("serving_mode", ["agg", "disagg"])
@pytest.mark.parametrize("explicit", [False, True])
def test_task_yaml_round_trip_and_recommend_rebuild_keep_inferred_precision(
    monkeypatch, tmp_path, serving_mode, explicit
):
    import yaml

    from aisim.legacy_cli.api import _build_recommend_tasks
    from aisim.sdk.task_v2 import Task

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    path = _super_with_fp8_experts(tmp_path)
    overrides = {"gemm_quant_mode": BF16, "moe_quant_mode": MOE.fp8} if explicit else {}
    if serving_mode == "agg":
        roles = explicit_roles = ["agg"]
        task = _task(path, **overrides)
    else:
        # Only decode takes the overrides, so swapping the roles' fields fails the per-role checks.
        roles, explicit_roles = ["prefill", "decode"], ["decode"]
        kwargs = {"total_gpus": 2}
        for role in roles:
            kwargs |= {f"{role}_model_path": path, f"{role}_system_name": "gb200", f"{role}_backend_name": "vllm"}
        kwargs |= {f"decode_{key}": value for key, value in overrides.items()}
        task = Task(serving_mode="disagg", backend_version="0.31.0", database_mode="SOL", **kwargs)
    reloaded = Task.from_yaml(yaml.safe_load(task.to_yaml()))
    rebuilt = _build_recommend_tasks({serving_mode: task}, 8)[serving_mode]
    assert rebuilt.total_gpus == 8
    for role in roles:
        model = _task_model(task, path, role)
        before = _buckets(model.context_ops), _experts(model)
        for other in (reloaded, rebuilt):
            other_model = _task_model(other, path, role)
            assert (_buckets(other_model.context_ops), _experts(other_model)) == before
        if explicit and role in explicit_roles:
            assert all(set(modes) == {BF16} for modes in before[0].values())
            assert set(before[1]) == {MOE.fp8}
        else:
            assert any(len(modes) > 1 for modes in before[0].values())
            assert set(before[1]) == {MOE.fp8, MOE.nvfp4}


def test_config_only_modelopt_mixed_listing_prices_unlisted_layers_bf16(monkeypatch, tmp_path):
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")

    def edit(section, e_layers):
        # One Mamba in_proj FP8 and one MoE layer's experts NVFP4; the rest unlisted.
        section.clear()
        section["quant_method"] = "modelopt_mixed"
        section["quant_algo"] = "MIXED_PRECISION"
        section["quantized_layers"] = {
            "backbone.layers.0.mixer.in_proj": {"quant_algo": "FP8"},
            **{
                f"backbone.layers.{e_layers[0]}.mixer.experts.0.{p}_proj": {"quant_algo": "NVFP4"}
                for p in ("up", "down")
            },
        }

    path = _write_super(tmp_path, edit)
    (tmp_path / "hf_quant_config.json").unlink()  # self-contained config.json
    pattern = json.loads((tmp_path / "config.json").read_text())["hybrid_override_pattern"]
    m_layers, e_layers = pattern.count("M"), pattern.count("E")
    for model in (
        get_model(path, ModelConfig(tp_size=2, moe_tp_size=2, system="gb200"), "vllm"),
        _task_model(_task(path), path),
    ):
        buckets = _buckets(model.context_ops)
        assert buckets["mamba_in_proj_gemm"] == {FP8: 1, BF16: m_layers - 1}
        assert set(buckets["mamba_out_proj_gemm"]) == set(buckets["qkv_gemm"]) == {BF16}
        assert _experts(model) == pytest.approx({MOE.nvfp4: 1, MOE.bfloat16: e_layers - 1})


def test_canonical_api_resolves_nvfp4_lanes_under_the_selected_systems_root(monkeypatch, tmp_path):
    from aisim_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, engine

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    # A Blackwell alias that only the explicit systems root defines.
    packaged = files("aisim_core") / "systems"
    (tmp_path / "custom_bw.yaml").write_text((packaged / "gb200.yaml").read_text())
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "custom_bw").symlink_to(packaged / "data" / "gb200")
    built = []
    monkeypatch.setattr(engine, "get_model", lambda *args: built.append(get_model(*args)) or built[-1])
    RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model=SUPER,
            system="custom_bw",
            systems_paths=(str(tmp_path),),
            backend="vllm",
            backend_version="0.30.0",
            worker_type="decode",
            tp=1,
            moe_tp_size=1,
            moe_ep_size=1,
            database_mode="SOL",
            estimation_mode="op_level",
        )
    ).close()
    (model,) = built
    assert _buckets(model.context_ops)["shared_down_gemm"] == _PUBLISHED[SUPER](NVFP4)["shared_down_gemm"]
    from aisim_core.sdk.perf_database import is_blackwell_system

    # A pathlib root resolves like its string form.
    assert is_blackwell_system("custom_bw", tmp_path) is is_blackwell_system("custom_bw", str(tmp_path)) is True
    assert set(_experts(model)) == {MOE.nvfp4}


def test_invalid_checkpoint_precision_stops_estimator_selection(monkeypatch, tmp_path):
    from aisim_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    root = files("aisim_core") / "model_configs"
    data = json.loads((root / f"{SUPER.replace('/', '--')}_config.json").read_text())
    hf_quant = json.loads((root / f"{SUPER.replace('/', '--')}_hf_quant_config.json").read_text())

    def best_available(model):
        return RustForwardPassPerfModel.best_available(
            ForwardPassPerfModelConfig(
                model=model,
                system="gb200",
                backend="vllm",
                backend_version="0.30.0",
                worker_type="decode",
                tp=1,
                moe_tp_size=1,
                moe_ep_size=1,
                database_mode="SOL",
                estimation_mode="auto",
                fallback_policy="deny",
            )
        )

    best_available(SUPER).close()  # a valid mixed checkpoint resolves
    # The two quantization files disagree on one module: auto selection must reject, not fall back.
    key = "backbone.layers.0.mixer.in_proj"
    data["quantization_config"]["quantized_layers"][key] = {"quant_algo": "FP8"}
    hf_quant["quantization"]["quantized_layers"][key] = {"quant_algo": "NVFP4", "group_size": 16}
    (tmp_path / "config.json").write_text(json.dumps(data))
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(hf_quant))
    with pytest.raises(ValueError, match="conflicting algorithms"):
        best_available(str(tmp_path))


def test_rebuilding_from_a_config_tracks_its_current_quant_modes(monkeypatch):
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")

    import copy

    def build(config):
        model = get_model(SUPER, config, backend_name="vllm")
        return _buckets(model.context_ops), _experts(model), sum(op.get_weights() for op in model.context_ops)

    def assert_uniform(built, gemm, moe):
        assert all(set(modes) == {gemm} for modes in built[0].values()) and set(built[1]) == {moe}

    direct = ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, system="gb200")
    # A Task-built config carries explicit=False flags for its inferred modes.
    task = _task(SUPER).build_model_config(role="agg", parallel=(1, 1, 1, 1, 1, 1))
    for config in (direct, task):
        inferred = build(config)
        # An unchanged rebuild, from a copy or a topology-only replace, keeps the checkpoint split
        # (Mamba in_proj: FP8 and BF16).
        assert build(config) == build(copy.copy(config)) == build(dataclasses.replace(config, tp_size=1)) == inferred
        assert len(inferred[0]["mamba_in_proj_gemm"]) == 2
        # Overrides on a copy, or assigned to the config itself, are explicit on every later build.
        assert_uniform(build(dataclasses.replace(config, gemm_quant_mode=BF16, moe_quant_mode=MOE.fp8)), BF16, MOE.fp8)
        config.gemm_quant_mode, config.moe_quant_mode = BF16, MOE.fp8
        for _ in range(3):
            assert_uniform(build(config), BF16, MOE.fp8)
        # Clearing the overrides restores inference.
        config.gemm_quant_mode = config.moe_quant_mode = None
        assert build(config) == build(config) == inferred
    # An explicit mode equal to the one the checkpoint fills is still explicit when assigned, also
    # through a later topology-only replace (as sweeps do).
    for config in (direct, task):
        config.gemm_quant_mode, config.moe_quant_mode = FP8, MOE.nvfp4
        for _ in range(2):
            assert_uniform(build(config), FP8, MOE.nvfp4)
        assert_uniform(build(dataclasses.replace(config, tp_size=1)), FP8, MOE.nvfp4)
    # Also when assigned before the first build of a Task-built config.
    fresh = _task(SUPER).build_model_config(role="agg", parallel=(1, 1, 1, 1, 1, 1))
    fresh.gemm_quant_mode = fresh.gemm_quant_mode
    assert set(
        _buckets(get_model(SUPER, dataclasses.replace(fresh, tp_size=1), "vllm").context_ops)["mamba_in_proj_gemm"]
    ) == {FP8}


def test_task_mode_overrides_survive_model_building_yaml_and_recommend(monkeypatch):
    import yaml

    from aisim.legacy_cli.api import _build_recommend_tasks
    from aisim.sdk.task_v2 import Task

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    task = _task(SUPER)

    def experts(task_or_config):
        if isinstance(task_or_config, Task):
            return set(_experts(_task_model(task_or_config, SUPER)))
        return set(_experts(get_model(SUPER, task_or_config, "vllm")))

    assert experts(task) == {MOE.nvfp4}
    # Overriding the config Task built, before its first model build, is explicit.
    config = task.build_model_config(role="agg", parallel=(1, 1, 1, 1, 1, 1))
    assert experts(dataclasses.replace(config, moe_quant_mode=MOE.fp8)) == {MOE.fp8}
    # A mode assigned to the Task after construction is explicit for building, YAML and recommend,
    # and refreshes that role's cached large-EP coverage, which depends on the MoE mode.
    task._large_ep_coverage_cache.update(agg={"stale": {}}, prefill={"kept": {}})
    task.moe_quant_mode = MOE.fp8
    assert task._large_ep_coverage_cache == {"prefill": {"kept": {}}, "agg": {}}
    reloaded = Task.from_yaml(yaml.safe_load(task.to_yaml()))
    rebuilt = _build_recommend_tasks({"agg": task}, 8)["agg"]
    for other in (task, reloaded, rebuilt):
        assert other.moe_quant_mode == MOE.fp8 and experts(other) == {MOE.fp8}


def test_resolvers_keep_overrides_and_inferred_modes_apart(monkeypatch, tmp_path):
    import copy

    from aisim_core.sdk.models.helpers import resolve_nvfp4_for_system

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")

    def in_proj(config):
        return set(_buckets(get_model(SUPER, config, "vllm").context_ops)["mamba_in_proj_gemm"])

    # A replace() override stays explicit through the legacy resolver.
    config = ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, system="gb200")
    assert in_proj(config) == {FP8, BF16}
    override = dataclasses.replace(config, gemm_quant_mode=BF16)
    resolve_nvfp4_for_system(override, "h200_sxm", SUPER, backend_name="vllm")
    assert in_proj(override) == in_proj(override) == {BF16}
    # A cleared Task override stays inferred through MoE communication resolution, at any EP.
    task = _task(SUPER, gemm_quant_mode=BF16)
    for gemm, expected in ((None, {FP8, BF16}), (BF16, {BF16})):
        task.gemm_quant_mode = gemm
        for ep in (1, 2):
            assert in_proj(task.build_model_config(role="agg", parallel=(ep, 1, 1, 1, ep, 1))) == expected
    # Clearing an override hands the mode back to inference, also through the legacy resolver.
    config = ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, system="gb200", moe_quant_mode=MOE.fp8)
    config.moe_quant_mode = None
    path = _super_with_fp8_experts(tmp_path)
    resolve_nvfp4_for_system(config, "h200_sxm", path, backend_name="vllm")
    assert _experts(get_model(path, config, "vllm")) == pytest.approx({MOE.fp8: 10, MOE.nvfp4_wo: 30})
    # Editing a shallow copy of a Task leaves the original's provenance alone.
    original = _task(SUPER)
    copy.copy(original).gemm_quant_mode = BF16
    assert set(_buckets(_task_model(original, SUPER).context_ops)["mamba_in_proj_gemm"]) == {FP8, BF16}


def test_engine_cache_identity_tells_the_precision_split_from_a_uniform_override():
    from types import SimpleNamespace

    from aisim_core.sdk.rust_engine_step import _engine_config_json

    database = SimpleNamespace(system="gb200", backend="vllm", version="0.31.0")
    inferred = get_model(SUPER, ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, system="gb200"), "vllm")
    explicit_config = ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, system="gb200", gemm_quant_mode=FP8)
    explicit = get_model(SUPER, explicit_config, "vllm")
    # Same global modes, different graphs: the cached engines must not be shared, in either order.
    assert inferred.config.gemm_quant_mode == explicit.config.gemm_quant_mode == FP8
    assert _engine_config_json(inferred, database) != _engine_config_json(explicit, database)


def test_clearing_an_explicit_task_mode_restores_inference_on_every_build(monkeypatch):
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    config = _task(SUPER, gemm_quant_mode=BF16).build_model_config(role="agg", parallel=(1, 1, 1, 1, 1, 1))
    assert set(_buckets(get_model(SUPER, config, "vllm").context_ops)["mamba_in_proj_gemm"]) == {BF16}
    config.gemm_quant_mode = None
    builds = [_buckets(get_model(SUPER, config, "vllm").context_ops)["mamba_in_proj_gemm"] for _ in range(2)]
    assert builds[0] == builds[1] == {FP8: 28, BF16: 12}


@pytest.mark.parametrize("explicit_first", [False, True])
def test_precision_split_engine_cache_is_independent_of_evaluation_order(monkeypatch, explicit_first):
    from collections import OrderedDict

    from aisim_core.sdk import perf_database, rust_engine_step

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    monkeypatch.setattr(rust_engine_step, "_ENGINE_HANDLE_CACHE", OrderedDict())
    database = perf_database.get_database_view("gb200", "vllm", "0.30.0", database_mode="SOL")
    models = {
        explicit: _task_model(_task(SUPER, **({"gemm_quant_mode": FP8} if explicit else {})), SUPER)
        for explicit in (False, True)
    }

    def in_proj(handle):
        context, _ = handle.run_static_per_op(batch_size=1, isl=1024, osl=2)
        return next(latency for name, latency, *_ in context if name == "context_mamba_in_proj_gemm")

    # Isolated compiles establish each graph's answer before either cache order.
    references = {}
    for explicit, model in models.items():
        rust_engine_step._engine_handle_cache_clear()
        references[explicit] = in_proj(rust_engine_step._cached_engine_handle(model, database))
    assert references[False] != pytest.approx(references[True], rel=1e-6)
    rust_engine_step._engine_handle_cache_clear()
    handles = {}
    for explicit in (explicit_first, not explicit_first):
        handles[explicit] = rust_engine_step._cached_engine_handle(models[explicit], database)
        assert in_proj(handles[explicit]) == pytest.approx(references[explicit], rel=1e-12)
    assert handles[False] is not handles[True]


def test_native_kv_sizing_uses_the_database_root_after_skipping_an_overlay(monkeypatch, tmp_path):
    from aisim_core.sdk.memory.kv_cache import estimate_kv_cache

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    packaged = str(files("aisim_core") / "systems")
    (tmp_path / "gb200.yaml").write_text("gpu: [")  # invalid YAML, skipped by get_database
    kwargs = dict(
        backend_version="0.30.0",
        tp_size=2,
        moe_tp_size=2,
        max_num_tokens=1024,
        max_batch_size=8,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
    )
    expected = estimate_kv_cache(SUPER, "gb200", "vllm", systems_path=packaged, **kwargs)
    actual = estimate_kv_cache(SUPER, "gb200", "vllm", systems_path=[str(tmp_path), packaged], **kwargs)
    assert actual["source"] == "native"
    assert actual == expected


@pytest.mark.parametrize("system,mode", [("h100_sxm", NVFP4_WO), ("gb200", NVFP4)])
@pytest.mark.parametrize("serving_mode", ["agg", "disagg"])
@pytest.mark.parametrize("kind", ["gemm", "moe"])
def test_live_task_nvfp4_override_uses_the_same_hardware_lane_as_reload(monkeypatch, system, mode, serving_mode, kind):
    import yaml

    from aisim.legacy_cli.api import _build_recommend_tasks
    from aisim.sdk.task_v2 import Task

    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
    if serving_mode == "agg":
        task = _task(SUPER, system=system, **({"gemm_quant_mode": BF16} if kind == "moe" else {}))
        role = "agg"
    else:
        role = "prefill"
        task = Task(
            serving_mode="disagg",
            backend_version="0.31.0",
            database_mode="SOL",
            **({f"{r}_gemm_quant_mode": BF16 for r in ("prefill", "decode")} if kind == "moe" else {}),
            **{
                f"{r}_{key}": value
                for r in ("prefill", "decode")
                for key, value in {"model_path": SUPER, "system_name": system, "backend_name": "vllm"}.items()
            },
        )
    enum = common.GEMMQuantMode if kind == "gemm" else common.MoEQuantMode
    mode = enum[mode.name]
    field = f"{kind}_quant_mode" if role == "agg" else f"{role}_{kind}_quant_mode"
    setattr(task, field, enum.nvfp4)
    assert getattr(task, field) == mode
    for other in (
        task,
        Task.from_yaml(yaml.safe_load(task.to_yaml())),
        _build_recommend_tasks({serving_mode: task}, 8)[serving_mode],
    ):
        other.validate()
        assert getattr(other, f"_{kind}_quant_mode_explicit_by_role")[role] is True
        model = _task_model(other, SUPER, role)
        assert getattr(model.config, f"{kind}_quant_mode") == mode
        if kind == "gemm":
            assert all(set(modes) == {mode} for modes in _buckets(model.context_ops).values())
        else:
            assert set(_experts(model)) == {mode}
    if serving_mode == "agg":
        # Execute the real Task engine path; Hopper native FP4 would fail here.
        task.run_single_agg(tp=2, moe_tp=2, batch_size=1, ctx_tokens=128)
