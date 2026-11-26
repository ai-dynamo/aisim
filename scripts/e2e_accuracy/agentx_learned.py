# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Learned-FPM inputs for AgentX: pinned step captures, fixed-seed training, and held-out step error.

No trained model is stored. Every evaluation downloads the pinned public step captures and retrains:

- the replay artifact on seeds 42 and 7 pooled;
- the held-out artifact on seed 42 only, which prices the seed-7 steps for the step-error columns.

Both performance models score the same held-out seed-7 steps at the evaluated lane count, through
``RustForwardPassPerfModel.best_available`` with the replay's compiled timing configuration. SGLang 0.5.20
captures drop steps whose device time landed on an adjacent record (``unattributed`` and
``shifted_pairs``); training drops only the model-free ``unattributed`` steps.
"""

import gzip
import hashlib
import json
import os
import re
from pathlib import Path

from scripts.e2e_accuracy.agentx_metrics import step_accumulator
from scripts.release.package_identity import runtime_module

TRAIN_SEED = 42
TEST_SEED = 7
RANDOM_STATE = 0
LEARNED = "fpm_learned"
OP_LEVEL = "op_level"
PERFORMANCE_MODELS = (OP_LEVEL, LEARNED)

# Public step captures per evaluated model. Revisions are immutable dataset commits.
STEP_CAPTURES = {
    "Qwen/Qwen3.5-397B-A17B": {
        "repository": "ai-dynamo/aisim-data",
        "revision": "b871eca91188c028f01f1f2206ff27db85f5186e",
        "root": "data/fpm/Qwen--Qwen3.5-397B-A17B/b300-sxm/vllm/0.31.0/native-tp8",
        "layout": "dynamo_native",
        "features": "fpm5",
        "drop_unattributed": False,
    },
    "nvidia/GLM-5.3-NVFP4": {
        "repository": "ai-dynamo/aisim-data",
        "revision": "1faa7906a72a111bcc663592300026a4d09f9396",
        "root": "data/fpm/nvidia--GLM-5.3-NVFP4/b300-sxm/sglang/0.5.20/native-tp8-mtp3",
        "layout": "sglang_hook",
        "features": "sglang18",
        "drop_unattributed": True,
    },
}


def _download(spec, name, cache):
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            spec["repository"],
            f"{spec['root']}/{name}",
            repo_type="dataset",
            revision=spec["revision"],
            cache_dir=cache,
        )
    )


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unwrap(line):
    raw = json.loads(line)
    fpm = raw.get("fpm") or (raw.get("event") or {}).get("fpm")
    return fpm if isinstance(fpm, dict) and "scheduled_requests" in fpm else None


def windows(spec, seed, cache, *, scoring=False):
    """Yield ``(lanes, records)`` per capture window in record order.

    ``dynamo_native``: the published training selection names each trace and its 1-based inclusive
    line range. ``sglang_hook``: the published manifest lists one warm-up and one profiling capture per
    seed and lane count (``step-capture/seed<S>/c<lanes>/<window>.jsonl.gz``); scoring uses profiling only.
    """
    if spec["layout"] == "dynamo_native":
        selection = json.loads(_download(spec, f"training/seed{seed}.selection.json", cache).read_text())
        for item in selection:
            lanes = int(re.search(r"/l(\d+)/", item["path"]).group(1))
            path = _download(spec, f"native-step/seed{seed}/l{lanes}/traces/{Path(item['path']).name}", cache)
            if _sha256(path) != item["sha256"]:
                raise ValueError(f"Step capture checksum mismatch: {item['path']}")
            with gzip.open(path, "rt") as stream:
                records = [
                    _unwrap(line)
                    for number, line in enumerate(stream, start=1)
                    if item["first_line"] <= number <= item["last_line"]
                ]
            yield lanes, [r for r in records if r is not None]
    elif spec["layout"] == "sglang_hook":
        manifest = json.loads(_download(spec, "aisim-manifest.json", cache).read_text())
        for artifact in sorted(manifest["artifacts"], key=lambda a: a["artifact_id"]):
            match = re.fullmatch(
                r"step-capture/seed(\d+)/c(\d+)/(warmup|profiling)\.jsonl\.gz", artifact["artifact_id"]
            )
            if not match or int(match.group(1)) != seed or (scoring and match.group(3) != "profiling"):
                continue
            path = _download(spec, artifact["artifact_id"], cache)
            if _sha256(path) != artifact["sha256"]:
                raise ValueError(f"Step capture checksum mismatch: {artifact['artifact_id']}")
            with gzip.open(path, "rt") as stream:
                records = [_unwrap(line) for line in stream]
            yield int(match.group(2)), [r for r in records if r is not None]
    else:
        raise ValueError(f"Unknown step-capture layout: {spec['layout']}")


def _busy(fpm):
    s = fpm["scheduled_requests"]
    return s["num_prefill_requests"] > 0 or s["num_decode_requests"] > 0


def unattributed(records):
    """Flags for steps whose device-time categories lack their target forward, and the record before.

    With overlap scheduling and speculative decoding v2, SGLang 0.5.20 can attribute a step's target
    forward (``extend`` for prefill, ``target_verify`` for MTP decode) to the adjacent record, so neither
    record's wall time is that step's time.
    """
    flags = [False] * len(records)
    for index, fpm in enumerate(records):
        if not _busy(fpm):
            continue
        s = fpm["scheduled_requests"]
        categories = s.get("device_time_s_by_category") or {}
        target = "extend" if s["num_prefill_requests"] > 0 else "target_verify"
        if categories and target not in categories:
            flags[index] = True
            if index:
                flags[index - 1] = True
    return flags


def kept(spec, records):
    flags = unattributed(records) if spec["drop_unattributed"] else [False] * len(records)
    return [fpm for fpm, flag in zip(records, flags, strict=True) if not flag], sum(flags)


def train_artifact(spec, seeds, cache, output):
    """Retrain deterministically; an existing artifact from the same inputs is reused."""
    if output.exists():
        return output
    learned = runtime_module("aisim_core.sdk.fpm_learned")

    records = []
    for seed in seeds:
        for _, window in windows(spec, seed, cache):
            records.extend(kept(spec, window)[0])
    artifact = learned.train(
        learned.group_iterations(records),
        "aggregated",
        features=list(learned.FEATURE_PRESETS[spec["features"]]),
        random_state=RANDOM_STATE,
        metadata={
            "step_captures": {key: spec[key] for key in ("repository", "revision", "root")},
            "seeds": list(seeds),
            "features_preset": spec["features"],
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    temporary.write_text(json.dumps(artifact, separators=(",", ":")))
    temporary.replace(output)
    return output


def artifacts(model, workdir):
    """``{"replay": path, "held_out": path}`` for a model with step captures, else None."""
    spec = STEP_CAPTURES.get(model)
    if spec is None:
        return None
    cache = workdir / "hf"
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    return {
        "replay": train_artifact(spec, (TRAIN_SEED, TEST_SEED), cache, workdir / f"{key}-replay.json"),
        "held_out": train_artifact(spec, (TRAIN_SEED,), cache, workdir / f"{key}-held-out.json"),
    }


def step_kind(fpm):
    s = fpm["scheduled_requests"]
    prefill, decode = s["num_prefill_requests"] > 0, s["num_decode_requests"] > 0
    if prefill and decode:
        return None  # Mixed steps are not shown.
    return "prefill" if prefill else "decode" if decode else None


def _estimator(timing_config):
    engine_step = runtime_module("aisim_core.sdk.rust_engine_step")
    # The compiled config names the collection the replay was pinned to; like agentx_predict, accept it
    # for this one construction only, so the opt-in never reaches later replays.
    previous = os.environ.get("AIC_ALLOW_UNLISTED_VERSIONS")
    os.environ["AIC_ALLOW_UNLISTED_VERSIONS"] = "1"
    try:
        return engine_step.RustForwardPassPerfModel.best_available(timing_config)
    finally:
        if previous is None:
            del os.environ["AIC_ALLOW_UNLISTED_VERSIONS"]
        else:
            os.environ["AIC_ALLOW_UNLISTED_VERSIONS"] = previous


def shifted_pairs(records, predict):
    """Flags adjacent records whose measured times split one step: one under 0.5x the reference
    prediction next to one over 1.5x. Category checks miss chunk-to-chunk shifts between prefills."""
    ratio = [records[i]["wall_time"] * 1e3 / predict(r) if _busy(r) else None for i, r in enumerate(records)]
    flags = [False] * len(records)
    for index, value in enumerate(ratio):
        if value is None or value >= 0.5:
            continue
        for neighbour in (index - 1, index + 1):
            if 0 <= neighbour < len(records) and ratio[neighbour] is not None and ratio[neighbour] > 1.5:
                flags[index] = flags[neighbour] = True
    return flags


def score_steps(model, lanes, timing_config, workdir, reference_config=None):
    """Held-out seed-7 step error at one lane count with the replay's compiled timing config.

    Where the captures carry SGLang 0.5.20's device-time shift, ``reference_config`` (the held-out learned
    model) also flags shifted pairs. Both performance models drop the same steps.
    """
    spec = STEP_CAPTURES.get(model)
    if spec is None:
        return None
    estimator = _estimator(timing_config)
    reference = _estimator(reference_config) if spec["drop_unattributed"] else None
    errors = {"decode": [], "prefill": []}
    dropped = 0
    for window_lanes, records in windows(spec, TEST_SEED, workdir / "hf", scoring=True):
        if window_lanes != lanes:
            continue
        flags = [False] * len(records)
        if reference is not None:
            pairs = shifted_pairs(records, lambda fpm: reference.estimate_forward_pass_time_ms([fpm]))
            flags = [a or b for a, b in zip(unattributed(records), pairs, strict=True)]
        dropped += sum(flags)
        for fpm, flag in zip(records, flags, strict=True):
            kind, measured = step_kind(fpm), fpm["wall_time"] * 1e3
            if flag or kind is None or measured <= 0:
                continue
            errors[kind].append(abs(estimator.estimate_forward_pass_time_ms([fpm]) / measured - 1) * 100)
    if not any(errors.values()):
        return None
    return {
        "seed": TEST_SEED,
        "dropped_shifted": dropped,
        **{kind: step_accumulator(values) for kind, values in errors.items()},
    }
