# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collector metadata intake through the real SDK downloader and an isolated cache."""

import hashlib
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
    assert resolved.source_reference == f"{_MODEL}@{_COMMIT}"
    assert resolved.payload["hidden_size"] == 1024
    assert resolved.payload.get("hf_quant_config") == (_QUANT if with_quant else None)


@pytest.mark.parametrize("confirmed_absent", [False, True])
def test_collector_offline_optional_absence_is_observable(monkeypatch, hub_cache, tmp_path, caplog, confirmed_absent):
    hub_cache("config.json", _CONFIG)
    if confirmed_absent:
        marker = tmp_path / "models--test-org--collector-cache-model" / ".no_exist" / _COMMIT / "hf_quant_config.json"
        marker.parent.mkdir(parents=True)
        marker.touch()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with caplog.at_level("WARNING"):
        resolved = load_model_config(_MODEL)
    assert resolved.payload == _CONFIG
    if confirmed_absent:
        assert not caplog.records
    else:
        assert "absence is unconfirmed" in caplog.text
        assert f"hf download {_MODEL} hf_quant_config.json --revision {_COMMIT}" in caplog.text


@pytest.mark.parametrize("payload", [None, {"quantization": {"quant_algo": "FP8"}}])
def test_collector_network_quant_json_must_be_an_object(monkeypatch, hub_cache, payload):
    def fetch(request, *, timeout):
        data = _CONFIG if request.full_url.endswith("/config.json") else payload
        response = io.BytesIO(json.dumps(data).encode())
        response.headers = {"X-Repo-Commit": _COMMIT}
        return response

    monkeypatch.setattr(utils.urllib.request, "urlopen", fetch)
    if payload is None:
        with pytest.raises(ValueError, match="hf_quant_config.json.*expected a JSON object"):
            load_model_config(_MODEL)
    else:
        resolved = load_model_config(_MODEL)
        assert resolved.payload["hf_quant_config"] == payload


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
    assert resolved.source_reference == f"{_MODEL}@{_COMMIT}"
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
    with pytest.raises(ValueError, match="config.json has no verified snapshot commit") as error:
        load_model_config(_MODEL)
    assert isinstance(error.value.__cause__, utils._HuggingFaceProvenanceError)
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


def test_collector_retains_snapshot_identity_without_changing_payload_hash(monkeypatch, hub_cache):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    hub_cache("config.json", _CONFIG)
    first = load_model_config(_MODEL)
    hub_cache("config.json", _CONFIG, commit="b" * 40)
    second = load_model_config(_MODEL)

    assert first.source_reference == f"{_MODEL}@{_COMMIT}"
    assert second.source_reference == f"{_MODEL}@{'b' * 40}"
    assert first.payload == second.payload == _CONFIG
    expected = hashlib.sha256(json.dumps(_CONFIG, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert first.sha256 == second.sha256 == expected


def test_collector_network_config_does_not_invent_a_snapshot(monkeypatch, hub_cache):
    hub_cache("config.json", _CONFIG)

    def fetch(request, *, timeout):
        if request.full_url.endswith("/main/config.json"):
            return io.BytesIO(json.dumps(_CONFIG).encode())
        raise utils.urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)

    monkeypatch.setattr(utils.urllib.request, "urlopen", fetch)
    resolved = load_model_config(_MODEL)
    assert resolved.source_reference == _MODEL
    assert resolved.payload == _CONFIG
