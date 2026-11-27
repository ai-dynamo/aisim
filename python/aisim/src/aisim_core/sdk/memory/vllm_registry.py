# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""Resolve stock vLLM generation classes without installing a serving runtime.

The class names and normalization order come from the tagged registries and
model configuration in vLLM 0.31.0, 0.32.0 and 0.33.0. See THIRD_PARTY_NOTICES.md.
"""

from functools import cache
from importlib.resources import files
from typing import NamedTuple

from packaging.version import Version

from aisim_core.sdk.common import parse_support_matrix_version

_RELEASES = ("0.31.0", "0.32.0", "0.33.0")
# These release candidates use the same native registry and resolver as the final releases.
_REGISTRY_RELEASES = {Version(tag): tag for tag in _RELEASES} | {
    Version("0.31.0rc5"): "0.31.0",
    Version("0.33.0rc3"): "0.33.0",
}


class _Registry(NamedTuple):
    names: dict[str, set[str]]
    causal_lms: dict[str, str]


@cache
def _registry(version: str) -> _Registry:
    names: dict[str, set[str]] = {"vllm": set(), "hf": set(), "hf_incompatible": set(), "config": set()}
    causal_lms: dict[str, str] = {}
    minor = version.split(".")[1]
    with files(__package__).joinpath("vllm_text_architectures.txt").open(encoding="utf-8") as source:
        for line in source:
            if line.startswith("#"):
                continue
            namespace, versions, suffix, *prefixes = line.split()
            if minor not in versions.split("/"):
                continue
            ending = "" if suffix == "-" else suffix
            if namespace == "causal":
                for pair in prefixes:
                    model_type, prefix = pair.split("=")
                    causal_lms[model_type] = prefix + ending
            else:
                names[namespace].update(prefix + ending for prefix in prefixes)
    return _Registry(names, causal_lms)


def _recorded_release(version: str | None) -> str | None:
    parsed = parse_support_matrix_version(version)
    if parsed is not None and parsed.local is not None:
        parsed = Version(parsed.public)
    return _REGISTRY_RELEASES.get(parsed)


def default_causal_lm(model_type: str, version: str | None) -> str | None:
    """The class with_hf_config supplies from Transformers when a child has no architectures."""
    release = _recorded_release(version)
    if release is not None:
        return _registry(release).causal_lms.get(model_type)
    choices = {_registry(v).causal_lms.get(model_type) for v in _RELEASES}
    if len(choices) != 1:
        raise ValueError(f"vLLM default decoder for {model_type!r} depends on the backend release")
    return choices.pop()


def is_known_transformers_config(model_type: str, version: str | None) -> bool:
    """The parent indexes AutoConfig by model type before its text architecture can be selected."""
    release = _recorded_release(version)
    if release is not None:
        return model_type in _registry(release).names["config"]
    if version is None:
        return all(model_type in _registry(v).names["config"] for v in _RELEASES)
    return any(model_type in _registry(v).names["config"] for v in _RELEASES)


def is_registered_conditional_generator(name: str | None, version: str | None) -> bool:
    """A registered conditional generation wrapper can construct auxiliary modules from defaults."""
    if not isinstance(name, str) or not name.endswith("ForConditionalGeneration"):
        return False
    release = _recorded_release(version)
    return any(name in _registry(v).names["vllm"] for v in ((release,) if release is not None else _RELEASES))


# These registrations load the same decoder implementation under another name.
_IMPLEMENTATIONS = {
    "DeepseekV3BidirectionalModel": "DeepseekV3ForCausalLM",
    "IQuestCoderForCausalLM": "LlamaForCausalLM",
    "InternLM3ForCausalLM": "LlamaForCausalLM",
    "InternVLForConditionalGeneration": "InternS1ForConditionalGeneration",
    "LLaMAForCausalLM": "LlamaForCausalLM",
    "LlamaModel": "LlamaForCausalLM",
    "Ministral3ForCausalLM": "MistralForCausalLM",
    "MistralModel": "LlamaForCausalLM",
    "NemotronHPuzzleForCausalLM": "NemotronHForCausalLM",
    "Qwen2Model": "Qwen2ForCausalLM",
    "TeleChat3ForCausalLM": "LlamaForCausalLM",
}
_GENERATION_SUFFIXES = ("ForCausalLM", "ForConditionalGeneration", "ChatModel", "LMHeadModel")
_SUFFIXES = (
    *_GENERATION_SUFFIXES,
    "ForTextEncoding",
    "EmbeddingModel",
    "ForSequenceClassification",
    "ForTokenClassification",
    "ForAudioClassification",
    "ForImageClassification",
    "ForVideoClassification",
    "ClassificationModel",
    "ForRewardModeling",
    "RewardModel",
    "Model",
)


def _normalize(name: str, stock: set[str]) -> str | None:
    if name in stock:
        return name
    for suffix in _GENERATION_SUFFIXES:
        if name.endswith(suffix):
            return next((candidate for s in _SUFFIXES if (candidate := name.replace(suffix, s)) in stock), None)
    return None


def _select(names: list[str], version: str, remote_model: bool) -> str | None:
    registrations = _registry(version).names
    stock = registrations["vllm"]
    if not any(name in stock for name in names):
        # An installed export takes precedence over AutoModel. An incompatible
        # stock class returns control to native dispatch even when AutoModel exists.
        first = names[0]
        if first in registrations["hf"] or (remote_model and first not in registrations["hf_incompatible"]):
            return None
    for name in names:
        registered = _normalize(name, stock)
        if registered is not None:
            if version == "0.31.0" and registered == "CwmForCausalLM":
                return "LlamaForCausalLM"
            return _IMPLEMENTATIONS.get(registered, registered)
    return None


def select_architecture(
    names: list[str], version: str | None, auto_map: object, *, modeled: set[str] | None = None
) -> str | None:
    """Select the decoder implementation for a nonempty list on a generation worker."""
    remote_model = isinstance(auto_map, dict) and any(
        isinstance(key, str) and key.startswith("AutoModel") for key in auto_map
    )
    if version is not None:
        release = _recorded_release(version)
        # Keep the original class and its compatibility checks on unrecorded releases.
        choices = {_select(names, release, remote_model) if release is not None else names[0]}
    else:
        choices = {_select(names, release, remote_model) for release in _RELEASES}
    if modeled is not None:
        choices = {name if name in modeled else None for name in choices}
    if len(choices) != 1:
        raise ValueError(f"vLLM decoder selection for {names!r} depends on the backend release")
    return choices.pop()


def select_wrapper_architecture(
    names: list[str], version: str | None, auto_map: object, wrappers: set[str]
) -> str | None:
    """Resolve a list containing a modeled wrapper while preserving generic nested configurations."""
    if not names or not any(_normalize(name, wrappers) for name in names):
        return names[0] if names else None
    if version is not None and _recorded_release(version) is None:
        return names[0]
    selected = select_architecture(names, version, auto_map)
    if selected not in wrappers:
        raise ValueError(f"vLLM cannot derive the text decoder from the selected wrapper in {names!r}")
    return selected
