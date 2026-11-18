# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collector metadata intake through the real SDK downloader and an isolated cache."""

import io
import json

import pytest
from collector.fpm_forward.model_capability import load_model_config

from aisim.sdk import utils

pytestmark = pytest.mark.unit

_MODEL = "test-org/collector-cache-model"
_COMMIT = "a" * 40
_CONFIG = {"architectures": ["LlamaForCausalLM"], "hidden_size": 1024}
_QUANT = {"quantization": {"quant_algo": "FP8"}}


@pytest.fixture
def hub_cache(monkeypatch, tmp_path):
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(utils, "_get_hf_auth_headers", lambda: {})

    def no_network(request, *, timeout):
        pytest.fail(f"unexpected network request: {request.full_url}")

    monkeypatch.setattr(utils.urllib.request, "urlopen", no_network)

    def write(filename, payload, *, commit=_COMMIT):
        repo = tmp_path / "models--test-org--collector-cache-model"
        snapshot = repo / "snapshots" / commit
        snapshot.mkdir(parents=True, exist_ok=True)
        blobs = repo / "blobs"
        blobs.mkdir(exist_ok=True)
        blob = blobs / f"{commit}-{filename}"
        blob.write_text(json.dumps(payload))
        (snapshot / filename).symlink_to(f"../../blobs/{blob.name}")
        refs = repo / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "main").write_text(commit)

    return write


@pytest.mark.parametrize("with_quant", [False, True])
def test_collector_offline_config_snapshot(monkeypatch, hub_cache, with_quant):
    hub_cache("config.json", _CONFIG)
    if with_quant:
        hub_cache("hf_quant_config.json", _QUANT)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    resolved = load_model_config(_MODEL)
    assert resolved.source_kind == "huggingface"
    assert resolved.payload["hidden_size"] == 1024
    assert resolved.payload.get("hf_quant_config") == (_QUANT if with_quant else None)


@pytest.mark.parametrize("quant_cached", [False, True])
def test_collector_config_fallback_pins_quant_request(monkeypatch, hub_cache, quant_cached):
    hub_cache("config.json", _CONFIG)
    if quant_cached:
        hub_cache("hf_quant_config.json", _QUANT)
    requested = []

    def fetch(request, *, timeout):
        url = request.full_url
        requested.append(url)
        if url.endswith("/main/config.json"):
            raise utils.urllib.error.URLError("transient")
        # Main deliberately disagrees with the cached snapshot. Only the pinned
        # request may contribute metadata to the frozen Collector payload.
        quant = _QUANT if f"/{_COMMIT}/" in url else {"quantization": {"quant_algo": "NVFP4"}}
        return io.BytesIO(json.dumps(quant).encode())

    monkeypatch.setattr(utils.urllib.request, "urlopen", fetch)
    resolved = load_model_config(_MODEL)
    assert resolved.payload["hidden_size"] == 1024
    assert resolved.payload["hf_quant_config"] == _QUANT
    assert requested == [f"https://huggingface.co/{_MODEL}/raw/main/config.json"] + (
        [] if quant_cached else [f"https://huggingface.co/{_MODEL}/raw/{_COMMIT}/hf_quant_config.json"]
    )


@pytest.mark.parametrize("cached_config", [False, True])
@pytest.mark.parametrize("failure", ["network", "server"])
def test_collector_remote_config_rejects_cached_quant_fallback(monkeypatch, hub_cache, cached_config, failure):
    if cached_config:
        hub_cache("config.json", _CONFIG)
    hub_cache("hf_quant_config.json", _QUANT, commit="b" * 40)

    def fetch(request, *, timeout):
        url = request.full_url
        if url.endswith("/main/config.json"):
            return io.BytesIO(json.dumps({**_CONFIG, "hidden_size": 2048}).encode())
        if failure == "network":
            raise utils.urllib.error.URLError("transient")
        raise utils.urllib.error.HTTPError(url, 503, "unavailable", {}, None)

    monkeypatch.setattr(utils.urllib.request, "urlopen", fetch)
    with pytest.raises(ValueError, match="Cannot mix cached hf_quant_config.json") as error:
        load_model_config(_MODEL)
    assert isinstance(error.value.__cause__, utils.HuggingFaceDownloadError)
    assert utils._SNAPSHOT_CAPTURE.get() is None


def test_collector_does_not_read_quant_from_another_cached_snapshot(monkeypatch, hub_cache):
    hub_cache("hf_quant_config.json", _QUANT, commit="b" * 40)
    hub_cache("config.json", _CONFIG)  # main -> A, only B has a quant file
    requested = []

    def fetch(request, *, timeout):
        url = request.full_url
        requested.append(url)
        if url.endswith("/main/config.json"):
            raise utils.urllib.error.URLError("transient")
        raise utils.urllib.error.HTTPError(url, 404, "absent at selected commit", {}, None)

    monkeypatch.setattr(utils.urllib.request, "urlopen", fetch)
    resolved = load_model_config(_MODEL)
    assert resolved.payload == _CONFIG
    assert requested[-1] == f"https://huggingface.co/{_MODEL}/raw/{_COMMIT}/hf_quant_config.json"
