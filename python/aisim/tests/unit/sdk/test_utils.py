# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for SDK utility functions.

Tests HuggingFace config parsing and model config retrieval.
"""

import io
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aisim.sdk import common, config, utils
from aisim.sdk.backends.base_backend import BaseBackend
from aisim.sdk.models import Gemma4MixModel, HybridMoEModel
from aisim.sdk.utils import (
    _parse_hf_config_json,
    enumerate_parallel_config,
    enumerate_ttft_tpot_constraints,
    get_model_config_from_model_path,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "revision,path_revision",
    [(None, "main"), ("a" * 40, "a" * 40), ("release/v1", "release%2Fv1")],
)
def test_download_hf_config_json_revision(monkeypatch, tmp_path, revision, path_revision):
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-cache"))
    requests = []
    metadata = {"architectures": ["UnregisteredDecoderForCausalLM"]}

    def fetch(request, *, timeout):
        requests.append(request.full_url)
        assert timeout == 30
        return io.BytesIO(json.dumps(metadata).encode())

    monkeypatch.setattr(utils, "_get_hf_auth_headers", lambda: {})
    monkeypatch.setattr(utils.urllib.request, "urlopen", fetch)
    kwargs = {} if revision is None else {"revision": revision}
    assert utils._download_hf_json("test/model", "config.json", **kwargs) == metadata
    assert requests == [f"https://huggingface.co/test/model/raw/{path_revision}/config.json"]


@pytest.fixture
def hf_cache(monkeypatch, tmp_path):
    """An isolated Hugging Face hub cache, network access that fails the test, and offline mode cleared."""
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(name, raising=False)
    cache = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(utils, "_get_hf_auth_headers", lambda: {})

    def no_network(request, *, timeout):
        pytest.fail(f"unexpected network access: {request.full_url}")

    monkeypatch.setattr(utils.urllib.request, "urlopen", no_network)
    return cache


def _cache_file(cache, hf_id, filename, payload, *, ref="main", commit="c" * 40, symlink=True):
    """Lay out a file the way huggingface_hub does: a blob, a snapshot symlink to it, and an optional ref."""
    repo = cache / f"models--{hf_id.replace('/', '--')}"
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir(parents=True, exist_ok=True)
    if symlink:
        blobs = repo / "blobs"
        blobs.mkdir(exist_ok=True)
        blob = blobs / f"{abs(hash((hf_id, filename, commit))):x}"
        blob.write_text(json.dumps(payload))
        (snapshot / filename).symlink_to(os.path.relpath(blob, snapshot))
    else:
        (snapshot / filename).write_text(json.dumps(payload))
    if ref is not None:
        ref_path = repo / "refs" / ref
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(commit)


@pytest.mark.parametrize("variable", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_offline_mode_reads_the_local_cache_without_network(monkeypatch, hf_cache, variable):
    _cache_file(hf_cache, "org/model", "config.json", {"architectures": ["X"]})
    monkeypatch.setenv(variable, "1")

    assert utils._download_hf_json("org/model", "config.json") == {"architectures": ["X"]}


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_offline_mode_fails_fast_when_not_cached(monkeypatch, hf_cache, value):
    monkeypatch.setenv("HF_HUB_OFFLINE", value)

    with pytest.raises(utils.HuggingFaceDownloadError) as error:
        utils._download_hf_json("org/missing", "config.json")

    message = str(error.value)
    assert "offline mode is on (HF_HUB_OFFLINE is set)" in message
    assert "not in the local Hugging Face cache" in message
    assert "hf download org/missing config.json" in message


def test_offline_mode_skips_optional_files(monkeypatch, hf_cache):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    assert utils._download_hf_json("org/model", "hf_quant_config.json", raise_on_404=False) is None


@pytest.mark.parametrize("confirmed_absent", [False, True])
def test_offline_optional_absence_is_observable_through_public_loader(monkeypatch, hf_cache, caplog, confirmed_absent):
    model = "org/optional-absence"
    config = utils._load_pre_downloaded_hf_config("Qwen/Qwen3-32B")
    _cache_file(hf_cache, model, "config.json", config)
    if confirmed_absent:
        marker = hf_cache / "models--org--optional-absence" / ".no_exist" / ("c" * 40) / "hf_quant_config.json"
        marker.parent.mkdir(parents=True)
        marker.touch()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    try:
        with caplog.at_level(logging.WARNING):
            loaded = get_model_config_from_model_path(model)
        assert loaded["architecture"] == "Qwen3ForCausalLM"
        assert "hf_quant_config" not in loaded["raw_config"]
        if confirmed_absent:
            assert not caplog.records
        else:
            assert "absence is unconfirmed" in caplog.text
            assert f"hf download {model} hf_quant_config.json --revision {'c' * 40}" in caplog.text
    finally:
        utils._load_model_config_from_model_path.cache_clear()
        utils.get_model_config_from_model_path.cache_clear()


@pytest.mark.parametrize("confirmed_absent", [False, True])
@pytest.mark.parametrize("consumer", ["sdk", "collector"])
def test_pinned_optional_absence_survives_network_outage(monkeypatch, hf_cache, confirmed_absent, consumer):
    from collector.fpm_forward.model_capability import _download_model_config

    model = f"org/confirmed-absence-{consumer}-{confirmed_absent}"
    config = utils._load_pre_downloaded_hf_config("Qwen/Qwen3-32B")
    _cache_file(hf_cache, model, "config.json", config)
    if confirmed_absent:
        marker = hf_cache / f"models--{model.replace('/', '--')}" / ".no_exist" / ("c" * 40) / "hf_quant_config.json"
        marker.parent.mkdir(parents=True)
        marker.touch()
    requested = []

    def unavailable(request, *, timeout):
        requested.append(request.full_url)
        raise _http_error(503, "unavailable")

    monkeypatch.setattr(utils.urllib.request, "urlopen", unavailable)
    load = get_model_config_from_model_path if consumer == "sdk" else _download_model_config
    try:
        if confirmed_absent:
            loaded = load(model)
            payload = loaded["raw_config"] if consumer == "sdk" else loaded.payload
            assert "hf_quant_config" not in payload
            assert len(requested) == 1
        else:
            expected_error = utils.HuggingFaceDownloadError if consumer == "sdk" else ValueError
            with pytest.raises(expected_error, match="unavailable"):
                load(model)
            assert len(requested) == 2
    finally:
        utils._load_model_config_from_model_path.cache_clear()
        utils.get_model_config_from_model_path.cache_clear()


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("section", [[], None, "FP8"])
def test_invalid_quantization_section_fails_public_loader(monkeypatch, hf_cache, offline, section):
    model = "org/invalid-quant-section"
    config = utils._load_pre_downloaded_hf_config("Qwen/Qwen3-32B")
    quant = {"quantization": section}
    if offline:
        _cache_file(hf_cache, model, "config.json", config)
        _cache_file(hf_cache, model, "hf_quant_config.json", quant)
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    else:

        def remote(request, *, timeout):
            payload = config if request.full_url.endswith("/config.json") else quant
            return io.BytesIO(json.dumps(payload).encode())

        monkeypatch.setattr(utils.urllib.request, "urlopen", remote)
    try:
        with pytest.raises(utils._HuggingFaceMetadataError, match="quantization must be an object"):
            get_model_config_from_model_path(model)
    finally:
        utils._load_model_config_from_model_path.cache_clear()
        utils.get_model_config_from_model_path.cache_clear()


def test_kimi_partial_video_sidecar_retains_inline_patch_limit(tmp_path):
    config = utils._load_pre_downloaded_hf_config("moonshotai/Kimi-K2.5")
    config["preprocessor_config"] = {"video_processor": {"max_patches": 1024}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "video_preprocessor_config.json").write_text('{"temporal_patch_size": 4}')
    try:
        loaded = get_model_config_from_model_path(str(tmp_path))
        assert loaded["encoder_config"].video_max_patches == 1024
        assert loaded["raw_config"]["preprocessor_config"]["video_processor"] == {
            "max_patches": 1024,
            "temporal_patch_size": 4,
        }
    finally:
        utils._load_model_config_from_model_path.cache_clear()
        utils.get_model_config_from_model_path.cache_clear()


@pytest.mark.parametrize("payload", [None, [], 7, {"quantization": {"quant_algo": "FP8"}}])
def test_network_quant_json_must_be_an_object_through_public_loader(monkeypatch, hf_cache, payload):
    model = "org/network-quant-object"
    config = utils._load_pre_downloaded_hf_config("Qwen/Qwen3-32B")

    def remote(request, *, timeout):
        data = config if request.full_url.endswith("/config.json") else payload
        return io.BytesIO(json.dumps(data).encode())

    monkeypatch.setattr(utils.urllib.request, "urlopen", remote)
    try:
        if isinstance(payload, dict):
            loaded = get_model_config_from_model_path(model)
            assert loaded["raw_config"]["hf_quant_config"] == payload
            assert loaded["raw_config"]["quant_algo"] == "fp8"
        else:
            with pytest.raises(utils.HuggingFaceDownloadError, match="hf_quant_config.json.*expected a JSON object"):
                get_model_config_from_model_path(model)
    finally:
        utils._load_model_config_from_model_path.cache_clear()
        utils.get_model_config_from_model_path.cache_clear()


def test_llama4_missing_offline_processor_reports_revision_and_can_retry(monkeypatch, hf_cache):
    model = "org/llama4-offline-processor"
    config = utils._load_pre_downloaded_hf_config("meta-llama/Llama-4-Scout-17B-16E-Instruct")
    processor = config.pop("image_processor_config")
    _cache_file(hf_cache, model, "config.json", config)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    try:
        with pytest.raises(utils.HuggingFaceDownloadError, match="preprocessor_config.json.*offline mode") as error:
            get_model_config_from_model_path(model)
        assert f"hf download {model} preprocessor_config.json --revision {'c' * 40}" in str(error.value)
        _cache_file(hf_cache, model, "preprocessor_config.json", processor)
        loaded = get_model_config_from_model_path(model)
        assert loaded["extra_params"].vision_config.max_num_tiles == 16
        assert utils._config_snapshot_commit(loaded["raw_config"]) == "c" * 40
    finally:
        utils._load_model_config_from_model_path.cache_clear()
        utils.get_model_config_from_model_path.cache_clear()


@pytest.mark.parametrize(
    "sidecar",
    [None, {"min_pixels": 262144, "max_pixels": 2097152}, {"size": {"shortest_edge": 262144, "longest_edge": 2097152}}],
)
def test_local_qwen_inline_processor_bounds_and_sidecar_precedence(tmp_path, sidecar):
    config = utils._load_pre_downloaded_hf_config("Qwen/Qwen3-VL-8B-Instruct")
    config["preprocessor_config"] = {"min_pixels": 131072, "max_pixels": 1048576}
    (tmp_path / "config.json").write_text(json.dumps(config))
    if sidecar is not None:
        (tmp_path / "preprocessor_config.json").write_text(json.dumps(sidecar))
    try:
        loaded = get_model_config_from_model_path(str(tmp_path))
        expected = {"min_pixels": 262144, "max_pixels": 2097152} if sidecar else config["preprocessor_config"]
        encoder = loaded["extra_params"]
        assert (encoder.min_pixels, encoder.max_pixels) == (expected["min_pixels"], expected["max_pixels"])
        assert loaded["raw_config"]["preprocessor_config"]["min_pixels"] == expected["min_pixels"]
        assert loaded["raw_config"]["preprocessor_config"]["max_pixels"] == expected["max_pixels"]
    finally:
        utils._load_model_config_from_model_path.cache_clear()
        utils.get_model_config_from_model_path.cache_clear()


def test_offline_value_zero_still_downloads(monkeypatch, hf_cache):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(
        utils.urllib.request, "urlopen", lambda request, *, timeout: io.BytesIO(b'{"architectures": ["Y"]}')
    )

    assert utils._download_hf_json("org/model", "config.json") == {"architectures": ["Y"]}


def test_pinned_commit_is_read_from_cache_before_network(hf_cache):
    commit = "d" * 40
    _cache_file(hf_cache, "org/model", "config.json", {"pinned": True}, commit=commit, ref=None)
    assert not (hf_cache / "models--org--model" / "refs").exists()

    assert utils._download_hf_json("org/model", "config.json", revision=commit) == {"pinned": True}


def test_network_failure_falls_back_to_cache(monkeypatch, hf_cache, caplog):
    _cache_file(hf_cache, "org/model", "config.json", {"cached": True})

    def unreachable(request, *, timeout):
        raise utils.urllib.error.URLError("nodename nor servname provided")

    monkeypatch.setattr(utils.urllib.request, "urlopen", unreachable)
    with caplog.at_level(logging.WARNING):
        assert utils._download_hf_json("org/model", "config.json") == {"cached": True}
    assert "using the cached copy" in caplog.text


@pytest.mark.parametrize(
    "failure",
    [utils.urllib.error.URLError("nodename nor servname provided"), TimeoutError("timed out")],
)
def test_network_failure_without_cache_explains_the_options(monkeypatch, hf_cache, failure):
    def unreachable(request, *, timeout):
        raise failure

    monkeypatch.setattr(utils.urllib.request, "urlopen", unreachable)
    with pytest.raises(utils.HuggingFaceDownloadError) as error:
        utils._download_hf_json("org/model", "config.json")

    message = str(error.value)
    assert "the download from Hugging Face failed" in message
    assert "AISim bundles configs only for its supported models" in message
    assert "HF_HUB_OFFLINE=1" in message


def test_cache_symlinks_resolve_to_blobs(monkeypatch, hf_cache):
    _cache_file(hf_cache, "org/model", "config.json", {"linked": True}, symlink=True)
    snapshot_file = hf_cache / "models--org--model" / "snapshots" / ("c" * 40) / "config.json"
    assert snapshot_file.is_symlink()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    assert utils._download_hf_json("org/model", "config.json") == {"linked": True}


def test_offline_mode_reads_a_ref_containing_a_slash(monkeypatch, hf_cache):
    _cache_file(hf_cache, "org/model", "config.json", {"branch": "release/v1"}, ref="release/v1", commit="f" * 40)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    assert utils._download_hf_json("org/model", "config.json", revision="release/v1") == {"branch": "release/v1"}
    with pytest.raises(utils.HuggingFaceDownloadError):
        utils._download_hf_json("org/model", "config.json", revision="release/v2")


@pytest.mark.parametrize("revision", ["../../escape", "main/../../escape"])
def test_cache_lookup_rejects_path_traversal(monkeypatch, hf_cache, revision):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(utils.HuggingFaceDownloadError):
        utils._download_hf_json("org/model", "config.json", revision=revision)


def test_absolute_revision_is_rejected_before_any_cache_lookup(monkeypatch, hf_cache, tmp_path):
    commit = "e" * 40
    _cache_file(hf_cache, "org/model", "config.json", {"absolute": True}, commit=commit)
    revision_file = tmp_path / "revision"
    revision_file.write_text(commit)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    with pytest.raises(utils.HuggingFaceDownloadError, match="unsupported Hugging Face revision"):
        utils._download_hf_json("org/model", "config.json", revision=str(revision_file))
    # A slash inside a branch name is still a revision, not a path.
    _cache_file(hf_cache, "org/model", "config.json", {"branch": True}, ref="release/v1", commit="f" * 40)
    assert utils._download_hf_json("org/model", "config.json", revision="release/v1") == {"branch": True}


def test_interrupted_response_body_falls_back_to_cache(monkeypatch, hf_cache, caplog):
    _cache_file(hf_cache, "org/model", "config.json", {"cached": True})

    class Truncated(io.BytesIO):
        def read(self, *args):
            raise utils.http.client.IncompleteRead(b'{"partial', 100)

    monkeypatch.setattr(utils.urllib.request, "urlopen", lambda request, *, timeout: Truncated())
    with caplog.at_level(logging.WARNING):
        assert utils._download_hf_json("org/model", "config.json") == {"cached": True}
    assert "using the cached copy" in caplog.text


def test_cached_config_pins_quant_metadata_to_the_same_commit(monkeypatch, hf_cache):
    cached_commit = "a" * 40
    _cache_file(hf_cache, "org/pinned", "config.json", {"cached_dims": True}, commit=cached_commit)
    _cache_file(hf_cache, "org/pinned", "hf_quant_config.json", {"quant": "cached"}, commit=cached_commit)
    requested = []

    def flaky(request, *, timeout):
        requested.append(request.full_url)
        if request.full_url.endswith("/main/config.json"):
            raise utils.urllib.error.URLError("transient")
        # The remote main has moved on; its metadata must not be mixed in.
        return io.BytesIO(json.dumps({"quant": "remote-main"}).encode())

    monkeypatch.setattr(utils.urllib.request, "urlopen", flaky)
    raw, commit = utils._download_hf_json_at("org/pinned", "config.json", raise_on_404=True)
    assert raw == {"cached_dims": True} and commit == cached_commit
    # The loader reads the companion file at the commit the config came from.
    assert utils._download_hf_json("org/pinned", "hf_quant_config.json", revision=commit) == {"quant": "cached"}
    assert all(cached_commit in url or url.endswith("/main/config.json") for url in requested)


def test_model_loader_reads_quant_config_at_the_cached_config_commit(monkeypatch, hf_cache):
    cached_commit = "b" * 40
    config = {
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 2,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "intermediate_size": 14336,
        "vocab_size": 128256,
        "max_position_embeddings": 4096,
    }
    _cache_file(hf_cache, "org/not-bundled", "config.json", config, commit=cached_commit)
    requested = []

    def flaky(request, *, timeout):
        requested.append(request.full_url)
        if request.full_url.endswith("/main/config.json"):
            raise utils.urllib.error.URLError("transient")
        raise utils.urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)

    monkeypatch.setattr(utils.urllib.request, "urlopen", flaky)
    utils._load_model_config_from_model_path.__wrapped__("org/not-bundled")
    # The optional quant file is requested at the cached config's commit, not main.
    assert [url for url in requested if "hf_quant_config.json" in url] == [
        f"https://huggingface.co/org/not-bundled/raw/{cached_commit}/hf_quant_config.json"
    ]


def test_concurrent_loads_keep_their_own_snapshot_pins(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Lock

    barrier = Barrier(2)
    quant_revisions: dict[str, str] = {}
    lock = Lock()

    def fake_at(hf_id, filename, *, raise_on_404=True, revision="main"):
        if filename == "config.json":
            barrier.wait(timeout=10)  # both loads are mid-flight together
            commit = "c" * 40 if hf_id == "org/cached" else None
            return {"architectures": ["LlamaForCausalLM"]}, commit
        with lock:
            quant_revisions[hf_id] = revision
        return None, None

    monkeypatch.setattr(utils, "_download_hf_json_at", fake_at)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(utils._load_model_config_from_model_path.__wrapped__, model)
            for model in ("org/cached", "org/online")
        ]
        loaded = [future.result(timeout=30) for future in futures]
    assert all(raw["architectures"] == ["LlamaForCausalLM"] for raw in loaded)
    assert [utils._config_snapshot_commit(raw) for raw in loaded] == ["c" * 40, None]
    assert quant_revisions == {"org/cached": "c" * 40, "org/online": "main"}


@pytest.mark.parametrize("architecture", ["Qwen3VLForConditionalGeneration", "KimiK25ForConditionalGeneration"])
def test_processor_files_follow_the_cached_config_commit(monkeypatch, hf_cache, architecture):
    commit = "d" * 40
    model = f"org/pinned-processor-{architecture}"
    _cache_file(hf_cache, model, "config.json", {"architectures": [architecture]}, commit=commit)
    requested = []
    filenames = ["preprocessor_config.json"]
    if architecture == "KimiK25ForConditionalGeneration":
        filenames.append("video_preprocessor_config.json")

    def remote(request, *, timeout):
        url = request.full_url
        requested.append(url)
        if url.endswith("/main/config.json"):
            raise utils.urllib.error.URLError("transient")
        filename = url.rsplit("/", 1)[1]
        if filename in filenames:
            # Remote main deliberately contains different metadata.
            source = "pinned" if f"/{commit}/" in url else "remote-main"
            return io.BytesIO(json.dumps({"source": source}).encode())
        raise utils.urllib.error.HTTPError(url, 404, "missing", {}, None)

    monkeypatch.setattr(utils.urllib.request, "urlopen", remote)
    monkeypatch.setattr(utils, "_parse_hf_config_json", lambda raw: {"architecture": architecture})
    loaded = utils.get_model_config_from_model_path.__wrapped__(model)
    processor = loaded["raw_config"]["preprocessor_config"]
    assert processor["source"] == "pinned"
    if len(filenames) == 2:
        assert processor["video_processor"]["source"] == "pinned"
    for filename in filenames:
        assert f"https://huggingface.co/{model}/raw/{commit}/{filename}" in requested


@pytest.mark.parametrize("ref_contents", [None, b"not-a-commit", b"\xff"])
def test_loader_distinguishes_missing_and_corrupt_cache_refs(monkeypatch, hf_cache, ref_contents):
    model = "org/ref-validation"
    _cache_file(hf_cache, model, "config.json", {"architectures": ["LlamaForCausalLM"]})
    ref = hf_cache / "models--org--ref-validation" / "refs" / "main"
    if ref_contents is None:
        ref.unlink()
    else:
        ref.write_bytes(ref_contents)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    message = "offline mode" if ref_contents is None else "Invalid cached ref"
    with pytest.raises(utils.HuggingFaceDownloadError, match=message):
        utils._load_model_config_from_model_path.__wrapped__(model)


@pytest.mark.parametrize("entry", ["missing", "directory", "dangling", "permission", "valid"])
def test_cached_quant_snapshot_entries_are_validated(monkeypatch, hf_cache, entry):
    model = f"org/snapshot-{entry}"
    metadata = {"architectures": ["LlamaForCausalLM"]}
    _cache_file(hf_cache, model, "config.json", metadata)
    path = hf_cache / f"models--org--snapshot-{entry}" / "snapshots" / ("c" * 40) / "hf_quant_config.json"
    denied = PermissionError("cannot stat snapshot")
    if entry == "directory":
        path.mkdir()
    elif entry == "dangling":
        path.symlink_to("missing-blob")
    elif entry == "permission":
        real_stat = Path.stat

        def probe(candidate, *args, **kwargs):
            if candidate == path:
                raise denied
            return real_stat(candidate, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", probe)
    elif entry == "valid":
        _cache_file(hf_cache, model, "hf_quant_config.json", {"quantization": {"quant_algo": "FP8"}})
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    if entry in ("missing", "valid"):
        loaded = utils._load_model_config_from_model_path.__wrapped__(model)
        assert ("hf_quant_config" in loaded) == (entry == "valid")
    else:
        with pytest.raises(utils.HuggingFaceDownloadError, match="Invalid cached hf_quant_config.json") as error:
            utils._load_model_config_from_model_path.__wrapped__(model)
        if entry == "permission":
            assert error.value.__cause__ is denied


def test_pinned_profile_consumer_reports_offline_miss_and_loads_cached_config(monkeypatch, hf_cache):
    from aisim_core.sdk import fpm_profile

    model = "org/profile-offline-config"
    commit = "d" * 40
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    fpm_profile._load_pinned_checkpoint_config.cache_clear()
    profile = SimpleNamespace(model=model, model_revision=commit)
    with pytest.raises(ValueError, match="cannot verify FPM profile architecture.*could not be resolved") as error:
        fpm_profile._validate_checkpoint_architecture(profile)
    assert isinstance(error.value.__cause__, utils.HuggingFaceDownloadError)
    metadata = {"architectures": ["LlamaForCausalLM"], "hidden_size": 1024}
    _cache_file(hf_cache, model, "config.json", metadata, commit=commit)
    try:
        profile.architecture = "LlamaForCausalLM"
        fpm_profile._validate_checkpoint_architecture(profile)
        assert fpm_profile._load_pinned_checkpoint_config(model, commit) == metadata
    finally:
        fpm_profile._load_pinned_checkpoint_config.cache_clear()


@pytest.mark.parametrize("contents", ["{broken", "[]", "null"])
def test_corrupt_cached_quant_config_is_not_ignored(monkeypatch, hf_cache, contents):
    model = "org/corrupt-quant"
    _cache_file(hf_cache, model, "config.json", {"architectures": ["LlamaForCausalLM"]})
    _cache_file(hf_cache, model, "hf_quant_config.json", {})
    quant_file = hf_cache / "models--org--corrupt-quant" / "snapshots" / ("c" * 40) / "hf_quant_config.json"
    quant_file.write_text(contents)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    with pytest.raises(utils.HuggingFaceDownloadError, match="Invalid cached hf_quant_config.json"):
        utils._load_model_config_from_model_path.__wrapped__(model)
    assert quant_file.read_text() == contents


def test_corrupt_config_after_network_failure_keeps_download_error_cause(monkeypatch, hf_cache):
    model = "org/corrupt-config"
    _cache_file(hf_cache, model, "config.json", {})
    _cache_file(hf_cache, model, "hf_quant_config.json", {"quantization": {"quant_algo": "FP8"}})
    config_file = hf_cache / "models--org--corrupt-config" / "snapshots" / ("c" * 40) / "config.json"
    config_file.write_text("{broken")

    def remote(request, *, timeout):
        raise utils.urllib.error.URLError("transient")

    monkeypatch.setattr(utils.urllib.request, "urlopen", remote)
    with pytest.raises(utils.HuggingFaceDownloadError, match="Invalid cached config.json") as error:
        utils._load_model_config_from_model_path.__wrapped__(model)
    assert isinstance(error.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize(
    "architecture,filename",
    [
        ("Qwen3VLForConditionalGeneration", "preprocessor_config.json"),
        ("KimiK25ForConditionalGeneration", "video_preprocessor_config.json"),
        ("Llama4ForConditionalGeneration", "preprocessor_config.json"),
    ],
)
def test_processor_only_cache_fallback_cannot_mix_with_remote_config(monkeypatch, hf_cache, architecture, filename):
    model = f"org/processor-fallback-{architecture}"
    _cache_file(hf_cache, model, filename, {"snapshot": "cached"})

    def remote(request, *, timeout):
        if request.full_url.endswith("/main/config.json"):
            return io.BytesIO(json.dumps({"architectures": [architecture]}).encode())
        if request.full_url.endswith(f"/{filename}"):
            raise utils.urllib.error.URLError("transient")
        raise utils.urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)

    def unexpected_parse(raw):
        pytest.fail("mixed-snapshot processor metadata reached the model parser")

    monkeypatch.setattr(utils.urllib.request, "urlopen", remote)
    monkeypatch.setattr(utils, "_parse_hf_config_json", unexpected_parse)
    with pytest.raises(utils.HuggingFaceDownloadError, match="Cannot mix cached"):
        utils.get_model_config_from_model_path.__wrapped__(model)


def test_offline_processor_and_config_from_one_snapshot_still_load(monkeypatch, hf_cache):
    model = "org/processor-same-snapshot"
    architecture = "Qwen3VLForConditionalGeneration"
    _cache_file(hf_cache, model, "config.json", {"architectures": [architecture]})
    _cache_file(hf_cache, model, "preprocessor_config.json", {"snapshot": "cached"})
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(utils, "_parse_hf_config_json", lambda raw: {"architecture": architecture})

    loaded = utils.get_model_config_from_model_path.__wrapped__(model)
    assert loaded["raw_config"]["preprocessor_config"] == {"snapshot": "cached"}


@pytest.mark.parametrize("cached_config", [True, False])
def test_remote_config_with_cached_quant_fallback_rejects_unverified_snapshot(monkeypatch, hf_cache, cached_config):
    cached = "a" * 40
    if cached_config:
        _cache_file(
            hf_cache,
            "org/mixed",
            "config.json",
            {"architectures": ["LlamaForCausalLM"], "hidden_size": 1024},
            commit=cached,
        )
    _cache_file(hf_cache, "org/mixed", "hf_quant_config.json", {"quantization": {"quant_algo": "FP8"}}, commit=cached)

    def remote(request, *, timeout):
        if request.full_url.endswith("/main/config.json"):
            return io.BytesIO(json.dumps({"architectures": ["LlamaForCausalLM"], "hidden_size": 2048}).encode())
        raise utils.urllib.error.URLError("transient")

    monkeypatch.setattr(utils.urllib.request, "urlopen", remote)
    # The cached quant file cannot replace successful remote dimensions or be
    # attached to them. An explicit error leaves the model selection unchanged.
    with pytest.raises(utils.HuggingFaceDownloadError, match="Cannot mix cached hf_quant_config.json"):
        utils._load_model_config_from_model_path.__wrapped__("org/mixed")


@pytest.mark.parametrize(
    "settings,expected",
    [
        ({"HF_HUB_CACHE": "a", "HUGGINGFACE_HUB_CACHE": "b", "HF_HOME": "c", "XDG_CACHE_HOME": "d"}, "a"),
        ({"HUGGINGFACE_HUB_CACHE": "b", "HF_HOME": "c", "XDG_CACHE_HOME": "d"}, "b"),
        ({"HF_HOME": "c", "XDG_CACHE_HOME": "d"}, "c/hub"),
        ({"XDG_CACHE_HOME": "d"}, "d/huggingface/hub"),
    ],
)
def test_hub_cache_location_follows_huggingface_hub_precedence(monkeypatch, tmp_path, settings, expected):
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    for name, value in settings.items():
        monkeypatch.setenv(name, str(tmp_path / value))

    assert utils._hf_hub_cache() == tmp_path / expected


@pytest.mark.parametrize("setting", ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"])
def test_empty_hub_cache_override_uses_working_directory(monkeypatch, hf_cache, setting):
    _cache_file(hf_cache, "org/model", "config.json", {"cached": True})
    monkeypatch.chdir(hf_cache)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hf_cache / "legacy"))
    monkeypatch.setenv("HF_HOME", str(hf_cache / "home"))
    monkeypatch.setenv(setting, "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    assert utils._hf_hub_cache() == Path(".")
    assert utils._download_hf_json("org/model", "config.json") == {"cached": True}


@pytest.mark.parametrize("setting", ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"])
def test_offline_cache_expands_environment_variables(monkeypatch, hf_cache, setting):
    _cache_file(hf_cache, "org/model", "config.json", {"cached": True})
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    cache_root = {
        "HF_HUB_CACHE": hf_cache,
        "HUGGINGFACE_HUB_CACHE": hf_cache,
        "HF_HOME": hf_cache.parent,
        "XDG_CACHE_HOME": hf_cache.parent,
    }[setting]
    if setting == "XDG_CACHE_HOME":
        hub = hf_cache.parent / "huggingface" / "hub"
        hub.parent.mkdir()
        hf_cache.rename(hub)
    monkeypatch.setenv("AISIM_TEST_SCRATCH", str(cache_root))
    monkeypatch.setenv(setting, "$AISIM_TEST_SCRATCH")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    assert utils._download_hf_json("org/model", "config.json") == {"cached": True}


@pytest.mark.parametrize("model", ["org--model", "org/model--variant", "org/../model", "org/model.git", "/org/model"])
def test_invalid_model_ids_cannot_alias_a_cached_model(monkeypatch, hf_cache, model):
    _cache_file(hf_cache, "org/model", "config.json", {"cached": True})
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    with pytest.raises(utils.HuggingFaceDownloadError, match="unsupported Hugging Face model ID"):
        utils._download_hf_json(model, "config.json")
    assert utils._download_hf_json("org/model", "config.json") == {"cached": True}


def test_local_directory_with_reserved_hub_spelling_still_loads(tmp_path):
    model = tmp_path / "org--model"
    model.mkdir()
    (model / "config.json").write_text('{"architectures": ["LlamaForCausalLM"]}')
    assert utils._load_model_config_from_model_path.__wrapped__(str(model))["architectures"] == ["LlamaForCausalLM"]


@pytest.mark.parametrize("filename", ["config.json", "hf_quant_config.json"])
def test_dangling_cache_ref_is_corruption_even_for_optional_files(monkeypatch, hf_cache, filename):
    _cache_file(hf_cache, "org/model", filename, {})
    ref = hf_cache / "models--org--model" / "refs" / "main"
    ref.unlink()
    ref.symlink_to("missing-target")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    with pytest.raises(utils.HuggingFaceDownloadError, match="Invalid cached ref.*dangling ref link"):
        utils._download_hf_json("org/model", filename, raise_on_404=filename == "config.json")
    ref.unlink()
    assert utils._download_hf_json("org/model", filename, raise_on_404=False) is None


def test_cached_json_preserves_quoted_nan_and_infinity_like_network(monkeypatch, hf_cache):
    payload = {"_name_or_path": "org/NaN", "note": 'Infinity -Infinity NaN with "quotes"'}
    _cache_file(hf_cache, "org/model", "config.json", payload)
    monkeypatch.setattr(
        utils.urllib.request, "urlopen", lambda request, *, timeout: io.BytesIO(json.dumps(payload).encode())
    )
    online = utils._download_hf_json("org/model", "config.json")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert utils._download_hf_json("org/model", "config.json") == online == payload


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-32"])
def test_cached_json_decodes_the_same_bytes_as_network(monkeypatch, hf_cache, encoding):
    payload = {"architectures": ["LlamaForCausalLM"], "note": "\u00e9"}
    _cache_file(hf_cache, "org/model", "config.json", payload)
    contents = json.dumps(payload, ensure_ascii=False).encode(encoding)
    path = hf_cache / "models--org--model" / "snapshots" / ("c" * 40) / "config.json"
    path.write_bytes(contents)
    monkeypatch.setattr(utils.urllib.request, "urlopen", lambda request, *, timeout: io.BytesIO(contents))

    assert utils._download_hf_json("org/model", "config.json") == payload
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert utils._download_hf_json("org/model", "config.json") == payload


def test_local_nonfinite_json_normalization_preserves_strings(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"values": [Infinity, -Infinity, NaN], "name": "org/NaN-Infinity"}')
    assert utils._load_json_with_infinity(path) == {"values": [None, None, None], "name": "org/NaN-Infinity"}


@pytest.mark.parametrize("architecture", ["Qwen3VLForConditionalGeneration", "KimiK25ForConditionalGeneration"])
def test_failed_processor_load_can_retry_offline_in_same_process(monkeypatch, hf_cache, architecture):
    model = f"org/retry-{architecture}"
    config = {"architectures": [architecture], "hidden_size": 1024}
    _cache_file(hf_cache, model, "config.json", config)
    _cache_file(hf_cache, model, "preprocessor_config.json", {"source": "cached"})
    requests = []

    def remote(request, *, timeout):
        requests.append(request.full_url)
        if request.full_url.endswith("/config.json"):
            return io.BytesIO(json.dumps({**config, "hidden_size": 2048}).encode())
        if request.full_url.endswith("/preprocessor_config.json"):
            raise utils.urllib.error.URLError("transient")
        raise _http_error(404, "missing")

    monkeypatch.setattr(utils.urllib.request, "urlopen", remote)
    monkeypatch.setattr(utils, "_parse_hf_config_json", lambda raw: {"architecture": architecture})
    utils._load_model_config_from_model_path.cache_clear()
    utils.get_model_config_from_model_path.cache_clear()
    try:
        with pytest.raises(utils.HuggingFaceDownloadError, match="Cannot mix cached") as error:
            utils.get_model_config_from_model_path(model)
        assert "config.json and the companion files from the same revision before setting HF_HUB_OFFLINE=1" in str(
            error.value
        )
        assert "local model directory" in str(error.value)
        assert utils._load_model_config_from_model_path.cache_info().currsize == 0
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        request_count = len(requests)
        loaded = utils.get_model_config_from_model_path(model)
        assert loaded["raw_config"]["hidden_size"] == 1024
        assert loaded["raw_config"]["preprocessor_config"] == {"source": "cached"}
        assert utils._config_snapshot_commit(loaded["raw_config"]) == "c" * 40
        assert len(requests) == request_count
        assert utils.get_model_config_from_model_path(model) is loaded
    finally:
        utils._load_model_config_from_model_path.cache_clear()
        utils.get_model_config_from_model_path.cache_clear()


def test_failed_same_model_reader_preserves_active_readers_snapshot(monkeypatch, hf_cache):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, current_thread

    model = "org/concurrent-processor-readers"
    commit = "c" * 40
    architecture = "Qwen3VLForConditionalGeneration"
    _cache_file(hf_cache, model, "config.json", {"architectures": [architecture], "hidden_size": 1024})
    paused = Event()
    resume = Event()
    real_lookup = utils._config_snapshot_commit
    requests = []

    def lookup(raw):
        if current_thread().name.startswith("paused-reader"):
            paused.set()
            assert resume.wait(10), "failed reader did not release the paused reader"
        return real_lookup(raw)

    def remote(request, *, timeout):
        url = request.full_url
        requests.append(url)
        if url.endswith("/main/config.json"):
            raise utils.urllib.error.URLError("config transport failed")
        if url.endswith("/hf_quant_config.json"):
            raise _http_error(404, "absent")
        if url.endswith("/preprocessor_config.json"):
            if not current_thread().name.startswith("paused-reader"):
                raise _http_error(503, "processor unavailable")
            source = "pinned" if f"/{commit}/" in url else "remote-main"
            return io.BytesIO(json.dumps({"source": source}).encode())
        pytest.fail(f"unexpected request: {url}")

    monkeypatch.setattr(utils, "_config_snapshot_commit", lookup)
    monkeypatch.setattr(utils.urllib.request, "urlopen", remote)
    monkeypatch.setattr(utils, "_parse_hf_config_json", lambda raw: {"architecture": architecture})
    utils._load_model_config_from_model_path.cache_clear()
    utils.get_model_config_from_model_path.cache_clear()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="paused-reader") as pool:
        try:
            # Both readers use the identical base object from the raw cache.
            utils._load_model_config_from_model_path(model)
            pending = pool.submit(utils.get_model_config_from_model_path, model)
            assert paused.wait(10), "reader did not reach provenance lookup"
            with pytest.raises(utils.HuggingFaceDownloadError, match="processor unavailable"):
                utils.get_model_config_from_model_path(model)
            resume.set()
            loaded = pending.result(timeout=10)
            assert loaded["raw_config"]["hidden_size"] == 1024
            assert loaded["raw_config"]["preprocessor_config"] == {"source": "pinned"}
            assert not any(url.endswith("/main/preprocessor_config.json") for url in requests)
            # A later raw reader must reuse A even after refs/main advances to B.
            _cache_file(
                hf_cache, model, "config.json", {"architectures": [architecture], "hidden_size": 2048}, commit="b" * 40
            )
            raw = utils._load_model_config_from_model_path(model)
            parsed = utils.get_model_config_from_model_path(model)
            assert utils._config_snapshot_commit(raw) == utils._config_snapshot_commit(parsed["raw_config"]) == commit
            assert raw["hidden_size"] == parsed["raw_config"]["hidden_size"] == 1024
        finally:
            resume.set()
            utils._load_model_config_from_model_path.cache_clear()
            utils.get_model_config_from_model_path.cache_clear()


@pytest.mark.parametrize("consumer", ["sdk", "collector"])
def test_negative_cache_probe_error_keeps_metadata_error_chain(monkeypatch, hf_cache, consumer):
    from collector.fpm_forward.model_capability import _download_model_config

    model = f"org/negative-probe-{consumer}"
    _cache_file(hf_cache, model, "config.json", utils._load_pre_downloaded_hf_config("Qwen/Qwen3-32B"))
    marker = hf_cache / f"models--{model.replace('/', '--')}" / ".no_exist" / ("c" * 40) / "hf_quant_config.json"
    real_stat = Path.stat
    denied = PermissionError("cannot traverse absence markers")

    def probe(path, *args, **kwargs):
        if path == marker:
            raise denied
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", probe)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    load = get_model_config_from_model_path if consumer == "sdk" else _download_model_config
    expected_error = utils.HuggingFaceDownloadError if consumer == "sdk" else ValueError
    try:
        with pytest.raises(expected_error, match="Invalid cached absence marker") as error:
            load(model)
        metadata_error = error.value if consumer == "sdk" else error.value.__cause__
        assert isinstance(metadata_error, utils._HuggingFaceCacheError)
        assert metadata_error.__cause__ is denied
    finally:
        utils._load_model_config_from_model_path.cache_clear()
        utils.get_model_config_from_model_path.cache_clear()


def test_reader_reloads_instead_of_restoring_evicted_base(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from functools import wraps
    from threading import Event, current_thread

    generation = 1
    cached = utils._cache_raw_model_config(lambda path: {"generation": generation})
    paused = Event()
    resume = Event()

    @wraps(cached)
    def lookup(path):
        raw = cached(path)
        if current_thread().name.startswith("paused-lease") and not paused.is_set():
            paused.set()
            assert resume.wait(10)
        return raw

    monkeypatch.setattr(utils, "_load_model_config_from_model_path", lookup)

    def reader():
        with cached.cache_reader("model") as entry:
            return entry.raw["generation"]

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="paused-lease") as pool:
        pending = pool.submit(reader)
        try:
            assert paused.wait(10)
            with pytest.raises(ValueError, match="failed parse"), cached.cache_reader("model"):
                raise ValueError("failed parse")
            assert cached.cache_info().currsize == 0
            generation = 2
            resume.set()
            assert pending.result(timeout=10) == 2
            assert cached("model")["generation"] == 2
        finally:
            resume.set()


def test_concurrent_llama4_readers_publish_one_validated_processor(monkeypatch, hf_cache):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Lock
    from time import sleep

    model = "org/concurrent-llama4-publication"
    config = utils._load_pre_downloaded_hf_config("meta-llama/Llama-4-Scout-17B-16E-Instruct")
    processor = config.pop("image_processor_config")
    requests = []
    lock = Lock()
    barrier = Barrier(4)

    def remote(request, *, timeout):
        if request.full_url.endswith("/config.json"):
            return io.BytesIO(json.dumps(config).encode())
        if request.full_url.endswith("/hf_quant_config.json"):
            raise _http_error(404, "absent")
        with lock:
            requests.append(request.full_url)
            max_patches = 16 if len(requests) == 1 else 4
        sleep(0.02)
        return io.BytesIO(json.dumps({**processor, "max_patches": max_patches}).encode())

    def load():
        barrier.wait(timeout=10)
        return get_model_config_from_model_path(model)

    monkeypatch.setattr(utils.urllib.request, "urlopen", remote)
    try:
        utils._load_model_config_from_model_path(model)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(load) for _ in range(4)]
            loaded = [future.result(timeout=10) for future in futures]
        assert len(requests) == 1
        assert all(item["extra_params"].vision_config.max_num_tiles == 16 for item in loaded)
        assert utils._load_model_config_from_model_path(model)["image_processor_config"]["max_patches"] == 16
    finally:
        utils._load_model_config_from_model_path.cache_clear()
        utils.get_model_config_from_model_path.cache_clear()


def test_failed_cached_processor_retries_release_base_configs(monkeypatch, hf_cache):
    import gc
    import weakref

    model = "org/reclaimed-processor-retries"
    config = utils._load_pre_downloaded_hf_config("Qwen/Qwen3-VL-8B-Instruct")
    _cache_file(hf_cache, model, "config.json", config)
    _cache_file(hf_cache, model, "preprocessor_config.json", {"min_pixels": True, "max_pixels": 16777216})
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    bases = []
    real_lookup = utils._config_snapshot_commit

    def observe(raw):
        bases.append(weakref.ref(raw))
        return real_lookup(raw)

    monkeypatch.setattr(utils, "_config_snapshot_commit", observe)
    utils._load_model_config_from_model_path.cache_clear()
    utils.get_model_config_from_model_path.cache_clear()
    try:
        for _ in range(5):
            with pytest.raises(ValueError, match="pixel budget must be positive integers"):
                utils.get_model_config_from_model_path(model)
            assert utils._load_model_config_from_model_path.cache_info().currsize == 0
        gc.collect()
        assert len(bases) == 5
        assert all(reference() is None for reference in bases)
    finally:
        utils._load_model_config_from_model_path.cache_clear()
        utils.get_model_config_from_model_path.cache_clear()


def test_kimi_video_limits_follow_pinned_processor_through_real_parser(monkeypatch, hf_cache):
    model = "org/kimi-video-limits"
    commit = "d" * 40
    # Reuse the checked-in checkpoint; the processor budgets below are synthetic.
    config = utils._load_pre_downloaded_hf_config("moonshotai/Kimi-K2.5")
    _cache_file(hf_cache, model, "config.json", config, commit=commit)
    requested = []

    def remote(request, *, timeout):
        url = request.full_url
        requested.append(url)
        if url.endswith("/main/config.json"):
            raise utils.urllib.error.URLError("transient")
        if url.endswith("/preprocessor_config.json"):
            return io.BytesIO(b'{"max_patches": 8192}')
        if url.endswith("/video_preprocessor_config.json"):
            return io.BytesIO(b'{"max_patches": 1024}')
        raise _http_error(404, "missing")

    monkeypatch.setattr(utils.urllib.request, "urlopen", remote)
    loaded = utils.get_model_config_from_model_path.__wrapped__(model)
    encoder = loaded["encoder_config"]
    assert encoder.video_max_patches == 1024
    assert encoder.image_max_patches == 8192
    assert encoder.max_video_frames == 4
    assert f"https://huggingface.co/{model}/raw/{commit}/video_preprocessor_config.json" in requested


def _http_error(code, reason):
    return utils.urllib.error.HTTPError("https://huggingface.co/x", code, reason, {}, None)


@pytest.mark.parametrize("failure", ["server", "network", "auth"])
def test_failed_quant_sidecar_is_not_absence_and_can_retry(monkeypatch, hf_cache, failure):
    model = f"org/quant-sidecar-failure-{failure}"
    config = utils._load_pre_downloaded_hf_config("Qwen/Qwen3-32B")
    quant = {"quantization": {"quant_algo": "FP8"}}
    recovered = False
    requests = []

    def remote(request, *, timeout):
        requests.append(request.full_url)
        if request.full_url.endswith("/config.json"):
            return io.BytesIO(json.dumps(config).encode())
        if not recovered:
            if failure == "network":
                raise utils.urllib.error.URLError("transient")
            raise _http_error(503 if failure == "server" else 403, "unavailable")
        return io.BytesIO(json.dumps(quant).encode())

    monkeypatch.setattr(utils.urllib.request, "urlopen", remote)
    try:
        with pytest.raises(utils.HuggingFaceDownloadError):
            utils.get_model_config_from_model_path(model)
        recovered = True
        loaded = utils.get_model_config_from_model_path(model)
        assert loaded["raw_config"]["hf_quant_config"] == quant
        assert loaded["raw_config"]["quant_algo"] == "fp8"
        assert len(requests) == 4
    finally:
        utils._load_model_config_from_model_path.cache_clear()
        utils.get_model_config_from_model_path.cache_clear()


@pytest.mark.parametrize(
    "architecture",
    ["Qwen3VLForConditionalGeneration", "KimiK25ForConditionalGeneration", "Llama4ForConditionalGeneration"],
)
def test_raw_config_and_quant_consumer_do_not_fetch_processors(monkeypatch, hf_cache, architecture):
    from aisim.generator.utils import _bundled_quantization

    model = f"org/base-only-{architecture}"
    config = {"architectures": [architecture], "hidden_size": 1024}
    quant = {"quantization": {"quant_algo": "FP8"}}
    requests = []

    def remote(request, *, timeout):
        requests.append(request.full_url)
        if request.full_url.endswith("/config.json"):
            return io.BytesIO(json.dumps(config).encode())
        if request.full_url.endswith("/hf_quant_config.json"):
            return io.BytesIO(json.dumps(quant).encode())
        raise _http_error(503, "processor unavailable")

    monkeypatch.setattr(utils.urllib.request, "urlopen", remote)
    try:
        raw = utils._load_model_config_from_model_path(model)
        assert raw["hidden_size"] == 1024
        assert raw["hf_quant_config"] == quant
        assert raw["quant_algo"] == "fp8"
        assert _bundled_quantization(model)["quant_algo"] == "FP8"
        assert len(requests) == 2
        with pytest.raises(utils.HuggingFaceDownloadError, match="processor unavailable"):
            utils.get_model_config_from_model_path(model)
        assert requests[-1].endswith("/preprocessor_config.json")
        assert "preprocessor_config" not in raw
        assert "image_processor_config" not in raw
    finally:
        utils._load_model_config_from_model_path.cache_clear()
        utils.get_model_config_from_model_path.cache_clear()


@pytest.mark.parametrize("local", [False, True])
def test_repaired_processor_retries_through_real_parser(monkeypatch, hf_cache, tmp_path, local):
    model = str(tmp_path / "qwen") if local else "org/repaired-processor"
    config = utils._load_pre_downloaded_hf_config("Qwen/Qwen3-VL-8B-Instruct")
    processor = {"min_pixels": True, "max_pixels": 16777216}
    if local:
        Path(model).mkdir()
        (Path(model) / "config.json").write_text(json.dumps(config))
        path = Path(model) / "preprocessor_config.json"
    else:
        _cache_file(hf_cache, model, "config.json", config)
        _cache_file(hf_cache, model, "preprocessor_config.json", processor)
        path = hf_cache / "models--org--repaired-processor" / "snapshots" / ("c" * 40) / "preprocessor_config.json"
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    path.write_text(json.dumps(processor))
    stable_model = tmp_path / "stable-model"
    stable_model.mkdir()
    (stable_model / "config.json").write_text('{"architectures": ["LlamaForCausalLM"]}')
    stable_config = utils._load_model_config_from_model_path(str(stable_model))
    try:
        with pytest.raises(ValueError, match="pixel budget must be positive integers"):
            utils.get_model_config_from_model_path(model)
        assert utils._load_model_config_from_model_path(str(stable_model)) is stable_config
        path.write_text(json.dumps({**processor, "min_pixels": 65536}))
        loaded = utils.get_model_config_from_model_path(model)
        assert loaded["raw_config"]["preprocessor_config"]["min_pixels"] == 65536
        assert (loaded["extra_params"].min_pixels, loaded["extra_params"].max_pixels) == (65536, 16777216)
        assert utils.get_model_config_from_model_path(model) is loaded
        assert "preprocessor_config" not in utils._load_model_config_from_model_path(model)
    finally:
        utils._load_model_config_from_model_path.cache_clear()
        utils.get_model_config_from_model_path.cache_clear()


@pytest.mark.parametrize("code,reason", [(429, "Too Many Requests"), (500, "Server Error"), (503, "Unavailable")])
def test_transient_http_errors_fall_back_to_cache(monkeypatch, hf_cache, code, reason):
    _cache_file(hf_cache, "org/model", "config.json", {"cached": True})

    def failing(request, *, timeout):
        raise _http_error(code, reason)

    monkeypatch.setattr(utils.urllib.request, "urlopen", failing)
    assert utils._download_hf_json("org/model", "config.json") == {"cached": True}


@pytest.mark.parametrize("code", [429, 502])
def test_transient_http_errors_without_cache_do_not_suggest_auth(monkeypatch, hf_cache, code):
    def failing(request, *, timeout):
        raise _http_error(code, "busy")

    monkeypatch.setattr(utils.urllib.request, "urlopen", failing)
    with pytest.raises(utils.HuggingFaceDownloadError) as error:
        utils._download_hf_json("org/model", "config.json")

    message = str(error.value)
    assert f"HTTP error {code}" in message
    assert "not in the local Hugging Face cache" in message
    assert "HF_TOKEN" not in message


@pytest.mark.parametrize("code", [401, 403])
def test_auth_errors_keep_the_authentication_advice(monkeypatch, hf_cache, code):
    def failing(request, *, timeout):
        raise _http_error(code, "denied")

    monkeypatch.setattr(utils.urllib.request, "urlopen", failing)
    with pytest.raises(utils.HuggingFaceDownloadError, match="HF_TOKEN"):
        utils._download_hf_json("org/model", "config.json")


def test_missing_optional_file_still_returns_none(monkeypatch, hf_cache):
    def failing(request, *, timeout):
        raise _http_error(404, "Not Found")

    monkeypatch.setattr(utils.urllib.request, "urlopen", failing)
    assert utils._download_hf_json("org/model", "hf_quant_config.json", raise_on_404=False) is None


@pytest.mark.parametrize("offline", [True, False])
def test_recovery_command_names_the_pinned_revision(monkeypatch, hf_cache, offline):
    commit = "b" * 40
    if offline:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    else:

        def unreachable(request, *, timeout):
            raise utils.urllib.error.URLError("down")

        monkeypatch.setattr(utils.urllib.request, "urlopen", unreachable)

    with pytest.raises(utils.HuggingFaceDownloadError) as error:
        utils._download_hf_json("org/model", "config.json", revision=commit)

    assert f"hf download org/model config.json --revision {commit}" in str(error.value)
    assert f"at revision {commit}" in str(error.value)


def test_recovery_command_omits_main_revision(monkeypatch, hf_cache):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(utils.HuggingFaceDownloadError) as error:
        utils._download_hf_json("org/model", "config.json")
    assert "`hf download org/model config.json`" in str(error.value)


def test_offline_mode_reaches_the_model_config_loader(monkeypatch, hf_cache):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    utils._load_model_config_from_model_path.cache_clear()

    with pytest.raises(utils.HuggingFaceDownloadError, match="offline mode is on"):
        utils._load_model_config_from_model_path("org/not-bundled-model")


@pytest.mark.parametrize("use_tilde", [False, True])
@pytest.mark.parametrize("setting", ["HF_HOME", "HF_TOKEN_PATH", "XDG_CACHE_HOME"])
def test_hf_metadata_uses_configured_token_location(monkeypatch, tmp_path, setting, use_tilde):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HOME", "HF_TOKEN_PATH", "HF_HUB_DISABLE_IMPLICIT_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    token = tmp_path / "huggingface" / "token"
    token.parent.mkdir()
    token.write_text("test-token\n")
    location = {"HF_HOME": token.parent, "HF_TOKEN_PATH": token, "XDG_CACHE_HOME": tmp_path}[setting]
    if use_tilde:
        monkeypatch.setenv("HOME", str(tmp_path))
        location = "~/" + str(location.relative_to(tmp_path))
    monkeypatch.setenv(setting, str(location))
    assert utils._get_hf_auth_headers() == {"Authorization": "Bearer test-token"}


def test_hf_metadata_can_disable_implicit_credentials(monkeypatch, tmp_path):
    token = tmp_path / "token"
    token.write_text("cached-token")
    monkeypatch.setenv("HF_TOKEN_PATH", str(token))
    monkeypatch.setenv("HF_TOKEN", "environment-token")
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "true")
    assert utils._get_hf_auth_headers() == {}
    assert token.read_text() == "cached-token"
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN")
    assert utils._get_hf_auth_headers() == {"Authorization": "Bearer environment-token"}


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "failure", [PermissionError("token file denied"), UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")]
)
def test_token_read_error_is_not_a_network_failure(monkeypatch, tmp_path, caplog, cached, failure):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_DISABLE_IMPLICIT_TOKEN", *utils._OFFLINE_VARIABLES):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    token_path = tmp_path / "unreadable-token"
    monkeypatch.setenv("HF_TOKEN_PATH", str(token_path))
    if cached:
        _cache_file(tmp_path, "org/model", "config.json", {"cached": True})
    real_read = Path.read_text

    def read(path, *args, **kwargs):
        if path == token_path:
            raise failure
        return real_read(path, *args, **kwargs)

    def no_network(request, *, timeout):
        pytest.fail(f"token read failure must precede network access: {request.full_url}")

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(utils.urllib.request, "urlopen", no_network)
    with pytest.raises(utils.HuggingFaceDownloadError, match="Cannot read the Hugging Face token") as error:
        utils._download_hf_json("org/model", "config.json")
    assert error.value.__cause__ is failure
    assert "using the cached copy" not in caplog.text


class TestParseHFConfig:
    """Test HuggingFace config parsing."""

    def test_parse_llama_config(self):
        """Test parsing a Llama model config."""
        config = {
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "intermediate_size": 14336,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "num_experts_per_tok": 0,
        }

        result = _parse_hf_config_json(config)

        assert result["architecture"] == "LlamaForCausalLM"  # architecture
        assert result["layers"] == 32  # num_layers
        assert result["n"] == 32  # num_heads
        assert result["n_kv"] == 8  # num_kv_heads
        assert result["hidden_size"] == 4096  # hidden_size
        assert result["inter_size"] == 14336  # inter_size
        assert result["vocab"] == 128256  # vocab_size

    def test_parse_moe_config(self):
        """Test parsing a MoE model config."""
        config = {
            "architectures": ["MixtralForCausalLM"],
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "intermediate_size": 14336,
            "vocab_size": 32000,
            "max_position_embeddings": 32768,
            "num_experts_per_tok": 2,
            "num_local_experts": 8,
            "moe_intermediate_size": 14336,
        }

        result = _parse_hf_config_json(config)

        assert result["architecture"] == "MixtralForCausalLM"  # architecture
        assert result["topk"] == 2  # topk
        assert result["num_experts"] == 8  # num_experts
        assert result["moe_inter_size"] == 14336  # moe_inter_size

    def test_parse_deepseek_config(self):
        """Test parsing a DeepSeek model config."""
        config = {
            "architectures": ["DeepseekV3ForCausalLM"],
            "num_hidden_layers": 61,
            "num_key_value_heads": 128,
            "hidden_size": 7168,
            "num_attention_heads": 128,
            "intermediate_size": 18432,
            "vocab_size": 129280,
            "max_position_embeddings": 4096,
            "num_experts_per_tok": 8,
            "n_routed_experts": 256,
            "moe_intermediate_size": 2048,
        }

        result = _parse_hf_config_json(config)

        assert result["architecture"] == "DeepseekV3ForCausalLM"  # architecture
        assert result["num_experts"] == 256  # num_experts from n_routed_experts

    def test_parse_config_with_head_dim(self):
        """Test parsing config that explicitly provides head_dim."""
        config = {
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 64,
            "num_key_value_heads": 8,
            "hidden_size": 5120,
            "num_attention_heads": 64,
            "intermediate_size": 25600,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "num_experts_per_tok": 0,
            "head_dim": 80,  # Explicit head_dim
        }

        result = _parse_hf_config_json(config)

        assert result["d"] == 80  # head_dim

    def test_parse_nemotronh_config(self):
        """Test parsing a NemotronH hybrid model config (Mamba + MoE + Transformer)."""
        config = {
            "architectures": ["NemotronHForCausalLM"],
            "num_hidden_layers": 52,
            "num_key_value_heads": 2,
            "hidden_size": 2688,
            "num_attention_heads": 32,
            "intermediate_size": 1856,
            "vocab_size": 131072,
            "max_position_embeddings": 262144,
            "num_experts_per_tok": 6,
            "n_routed_experts": 128,
            "moe_intermediate_size": 1856,
            "head_dim": 128,
            # NemotronH-specific fields
            "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
            "mamba_num_heads": 64,
            "mamba_head_dim": 64,
            "ssm_state_size": 128,
            "conv_kernel": 4,
            "n_groups": 8,
            "chunk_size": 128,
            "moe_shared_expert_intermediate_size": 3712,
            "mamba_ssm_cache_dtype": "bfloat16",
        }

        result = _parse_hf_config_json(config)

        assert result["architecture"] == "NemotronHForCausalLM"  # architecture
        assert result["layers"] == 52  # num_layers
        assert result["hidden_size"] == 2688  # hidden_size
        assert result["topk"] == 6  # topk (num_experts_per_tok)
        assert result["num_experts"] == 128  # num_experts (n_routed_experts)
        # extra_params should be NemotronHConfig
        extra_params = result["extra_params"]
        assert extra_params is not None
        assert hasattr(extra_params, "hybrid_override_pattern")
        assert extra_params.hybrid_override_pattern == "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
        assert extra_params.mamba_num_heads == 64
        assert extra_params.moe_shared_expert_intermediate_size == 3712
        assert extra_params.mamba_ssm_cache_dtype == "bfloat16"

    def test_parse_nemotronh_without_moe(self):
        """Test parsing a NemotronH config without MoE layers (no 'E' in pattern)."""
        config = {
            "architectures": ["NemotronHForCausalLM"],
            "num_hidden_layers": 118,
            "num_key_value_heads": 8,
            "hidden_size": 8192,
            "num_attention_heads": 64,
            "intermediate_size": 32768,
            "vocab_size": 131072,
            "max_position_embeddings": 8192,
            "attention_head_dim": 128,  # Uses attention_head_dim instead of head_dim
            # NemotronH-specific fields (no MoE)
            "hybrid_override_pattern": "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-",
            "mamba_num_heads": 256,
            "mamba_head_dim": 64,
            "ssm_state_size": 256,
            "conv_kernel": 4,
            "n_groups": 8,
            "chunk_size": 128,
        }

        result = _parse_hf_config_json(config)

        assert result["architecture"] == "NemotronHForCausalLM"  # architecture
        assert result["layers"] == 118  # num_layers
        assert result["hidden_size"] == 8192  # hidden_size
        assert result["d"] == 128  # head_dim from attention_head_dim
        # extra_params should be NemotronHConfig with moe_shared_expert_intermediate_size=0
        extra_params = result["extra_params"]
        assert extra_params is not None
        assert "E" not in extra_params.hybrid_override_pattern  # No MoE layers
        assert extra_params.moe_shared_expert_intermediate_size == 0
        assert extra_params.mamba_ssm_cache_dtype == "float32"  # absent field

    @pytest.mark.parametrize(("declared", "parsed"), [(None, "float32"), ("", ""), (0, 0)])
    def test_parse_nemotronh_ssm_cache_dtype_defaults_only_when_null(self, declared, parsed):
        # Only an absent or null field selects float32; an explicit value, valid or
        # not, reaches NemotronHConfig unchanged so that set_hybrid_config can reject it.
        config = {
            "architectures": ["NemotronHForCausalLM"],
            "num_hidden_layers": 118,
            "num_key_value_heads": 8,
            "hidden_size": 8192,
            "num_attention_heads": 64,
            "intermediate_size": 32768,
            "vocab_size": 131072,
            "max_position_embeddings": 8192,
            "attention_head_dim": 128,
            "hybrid_override_pattern": "M-M*-",
            "mamba_num_heads": 256,
            "mamba_head_dim": 64,
            "ssm_state_size": 256,
            "conv_kernel": 4,
            "n_groups": 8,
            "chunk_size": 128,
            "mamba_ssm_cache_dtype": declared,
        }

        assert _parse_hf_config_json(config)["extra_params"].mamba_ssm_cache_dtype == parsed

    @pytest.mark.parametrize("architecture", ["NemotronHForCausalLM", "NemotronH_Omni_Reasoning_V3"])
    @pytest.mark.parametrize("attention_names", [("mamba", "attention"), ("linear_attention", "full_attention")])
    def test_parse_nemotronh_layers_block_type_config(self, architecture, attention_names, tmp_path):
        """Test parsing NemotronH configs that use explicit layer block names."""
        config = {
            "architectures": ["NemotronHForCausalLM"],
            "num_key_value_heads": 2,
            "hidden_size": 8192,
            "num_attention_heads": 64,
            "intermediate_size": 5120,
            "vocab_size": 131072,
            "max_position_embeddings": 262144,
            "num_experts_per_tok": 22,
            "n_routed_experts": 512,
            "moe_intermediate_size": 5120,
            "head_dim": 128,
            "layers_block_type": [attention_names[0], "moe", attention_names[0], attention_names[1], "mlp"],
            "mamba_num_heads": 256,
            "mamba_head_dim": 64,
            "ssm_state_size": 128,
            "conv_kernel": 4,
            "n_groups": 8,
            "chunk_size": 128,
            "moe_shared_expert_intermediate_size": 10240,
        }

        if architecture == "NemotronH_Omni_Reasoning_V3":
            config = {"architectures": [architecture], "llm_config": config}
        result = _parse_hf_config_json(config)

        assert result["architecture"] == architecture
        assert result["layers"] == 5
        assert result["hidden_size"] == 8192
        assert result["topk"] == 22
        assert result["num_experts"] == 512
        extra_params = result["extra_params"]
        assert isinstance(extra_params, common.NemotronHConfig)
        assert extra_params.hybrid_override_pattern == "MEM*-"
        assert extra_params.mamba_num_heads == 256
        assert extra_params.moe_shared_expert_intermediate_size == 10240

        from aisim.sdk import models
        from aisim.sdk.config import ModelConfig

        (tmp_path / "config.json").write_text(json.dumps(config))
        model = models.get_model(str(tmp_path), ModelConfig(tp_size=1, moe_tp_size=1), "vllm")
        assert isinstance(model, models.NemotronHModel)

    def test_parse_qwen35_dense_config(self):
        """Test parsing Qwen3.5-27B (dense hybrid) config → Qwen35Config."""
        # Mimics Qwen/Qwen3.5-27B HF config structure (params nested under text_config).
        # 64 layers: 48 linear_attention + 16 full_attention (3:1 ratio).
        layer_types = ["linear_attention"] * 3 + ["full_attention"]
        layer_types = layer_types * 16  # 64 layers total (48 GDN + 16 GQA)
        config = {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "image_token_id": 248056,
            "video_token_id": 248057,
            "vision_config": {
                "depth": 27,
                "hidden_size": 1152,
                "num_heads": 16,
                "intermediate_size": 4304,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "out_hidden_size": 5120,
                # Qwen3.5 ignores inherited deepstack metadata and keeps one merger.
                "deepstack_visual_indexes": [8, 16, 24],
                "in_channels": 4,
            },
            "text_config": {
                "num_hidden_layers": 64,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "vocab_size": 151936,
                "max_position_embeddings": 32768,
                "head_dim": 256,
                "layer_types": layer_types,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_num_value_heads": 48,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
            },
        }

        result = _parse_hf_config_json(config)

        assert result["architecture"] == "Qwen3_5ForConditionalGeneration"
        assert result["layers"] == 64
        assert result["hidden_size"] == 5120
        assert result["inter_size"] == 17408
        assert result["n"] == 24
        assert result["n_kv"] == 4
        assert result["d"] == 256

        extra_params = result["extra_params"]
        assert isinstance(extra_params, common.Qwen35Config)
        assert len(extra_params.layer_types) == 64
        assert extra_params.layer_types.count("linear_attention") == 48
        assert extra_params.layer_types.count("full_attention") == 16
        assert extra_params.linear_num_key_heads == 16
        assert extra_params.linear_key_head_dim == 128
        assert extra_params.linear_num_value_heads == 48
        assert extra_params.linear_value_head_dim == 128
        assert extra_params.linear_conv_kernel_dim == 4
        # Dense model: no MoE routing
        assert extra_params.topk == 0
        assert extra_params.num_experts == 0
        # For dense models moe_inter_size falls back to intermediate_size
        assert extra_params.moe_inter_size == 17408
        assert extra_params.shared_expert_inter_size == 0
        assert extra_params.image_token_id == 248056
        assert extra_params.video_token_id == 248057
        assert extra_params.vision_config == common.VisionEncoderConfig(
            depth=27,
            hidden_size=1152,
            num_heads=16,
            intermediate_size=4304,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=5120,
            projector_dims=((4608, 4608), (4608, 5120)),
            projector_n_instances=1,
            partial_rotary_factor=1.0,
            in_channels=4,
        )

    def test_parse_qwen35_moe_config(self):
        """Test parsing Qwen3.5-35B-A3B (MoE hybrid) config → Qwen35Config with MoE fields."""
        # 40 layers: 30 linear_attention + 10 full_attention (3:1 ratio).
        layer_types = ["linear_attention"] * 3 + ["full_attention"]
        layer_types = layer_types * 10  # 40 layers total (30 GDN + 10 GQA)
        config = {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "image_token_id": 248056,
            "video_token_id": 248057,
            "vision_config": {
                "depth": 27,
                "hidden_size": 1152,
                "num_heads": 16,
                "intermediate_size": 4304,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "out_hidden_size": 2048,
                "deepstack_visual_indexes": [],
            },
            "text_config": {
                "num_hidden_layers": 40,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "hidden_size": 2048,
                "vocab_size": 151936,
                "max_position_embeddings": 32768,
                "head_dim": 256,
                "layer_types": layer_types,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_num_value_heads": 32,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                # MoE fields
                "num_experts_per_tok": 8,
                "num_experts": 256,
                "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512,
            },
        }

        result = _parse_hf_config_json(config)

        assert result["architecture"] == "Qwen3_5MoeForConditionalGeneration"
        assert result["layers"] == 40
        assert result["hidden_size"] == 2048
        assert result["topk"] == 8
        assert result["num_experts"] == 256

        extra_params = result["extra_params"]
        assert isinstance(extra_params, common.Qwen35Config)
        assert len(extra_params.layer_types) == 40
        assert extra_params.layer_types.count("linear_attention") == 30
        assert extra_params.layer_types.count("full_attention") == 10
        assert extra_params.linear_num_value_heads == 32
        assert extra_params.topk == 8
        assert extra_params.num_experts == 256
        assert extra_params.moe_inter_size == 512
        assert extra_params.shared_expert_inter_size == 512
        assert extra_params.vision_config is not None
        assert extra_params.vision_config.out_hidden_size == 2048
        assert extra_params.vision_config.projector_dims == ((4608, 4608), (4608, 2048))
        assert extra_params.vision_config.projector_n_instances == 1

    def test_parse_qwen35_rejects_vision_language_projection_mismatch(self):
        config = {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "vision_config": {
                "depth": 27,
                "hidden_size": 1152,
                "num_heads": 16,
                "intermediate_size": 4304,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "out_hidden_size": 4096,
            },
            "text_config": {
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "hidden_size": 5120,
                "intermediate_size": 1024,
                "vocab_size": 1024,
                "max_position_embeddings": 4096,
                "head_dim": 128,
                "layer_types": ["linear_attention"] * 3 + ["full_attention"],
                "linear_num_key_heads": 4,
                "linear_key_head_dim": 128,
                "linear_num_value_heads": 4,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
            },
        }

        with pytest.raises(ValueError, match="out_hidden_size must match"):
            _parse_hf_config_json(config)

    def test_parse_qwen35_layer_types_length_mismatch_raises(self):
        """Test that mismatched layer_types length raises ValueError."""
        layer_types = ["linear_attention"] * 3 + ["full_attention"]  # 4 entries, not 64
        config = {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {
                "num_hidden_layers": 64,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "vocab_size": 151936,
                "max_position_embeddings": 32768,
                "head_dim": 256,
                "layer_types": layer_types,  # length 4, not 64 → should raise
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_num_value_heads": 48,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
            },
        }
        with pytest.raises(ValueError, match="layer_types length"):
            _parse_hf_config_json(config)

    def test_parse_qwen38_max_flat_config(self):
        """Test parsing Qwen3.8-Max (Qwen/Qwen3.8-2.4T-A95B) → Qwen35Config from a FLAT config.

        Unlike the two Qwen3_5*ForConditionalGeneration VLM classes above, the
        released Qwen3.8-Max checkpoint ships Qwen3_5MoeForCausalLM with every
        LLM field top-level (no text_config nesting).
        """
        # 92 layers: interval-4 pattern (3 GDN then 1 full) x 23.
        layer_types = ["linear_attention"] * 3 + ["full_attention"]
        layer_types = layer_types * 23  # 92 layers total (69 GDN + 23 GQA)
        config = {
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "hidden_size": 8192,
            "num_hidden_layers": 92,
            "layer_types": layer_types,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "attn_output_gate": True,
            "partial_rotary_factor": 0.25,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_num_value_heads": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_dtype": "float32",
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 2048,
            "shared_expert_intermediate_size": 2048,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "mtp_num_hidden_layers": 1,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "dtype": "bfloat16",
            "rope_parameters": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
            "tie_word_embeddings": False,
        }

        result = _parse_hf_config_json(config)

        assert result["architecture"] == "Qwen3_5MoeForCausalLM"
        assert result["layers"] == 92
        assert result["hidden_size"] == 8192
        assert result["n"] == 64
        assert result["n_kv"] == 4
        # config head_dim (256) wins over hidden_size // n (8192 // 64 = 128).
        assert result["d"] == 256
        assert result["vocab"] == 248320
        assert result["topk"] == 10
        assert result["num_experts"] == 512
        assert result["moe_inter_size"] == 2048

        extra_params = result["extra_params"]
        assert isinstance(extra_params, common.Qwen35Config)
        assert len(extra_params.layer_types) == 92
        assert extra_params.layer_types.count("linear_attention") == 69
        assert extra_params.layer_types.count("full_attention") == 23
        assert extra_params.linear_num_key_heads == 16
        assert extra_params.linear_key_head_dim == 128
        assert extra_params.linear_num_value_heads == 128
        assert extra_params.linear_value_head_dim == 128
        assert extra_params.linear_conv_kernel_dim == 4
        assert extra_params.shared_expert_inter_size == 2048

    def test_parse_llama4_scout_config(self):
        """Test Llama 4 Scout (VLM, step=1: all-MoE) → HybridMoEConfig with alternating attn pattern."""
        config = {
            "architectures": ["Llama4ForConditionalGeneration"],
            "model_type": "llama4",
            "image_processor_config": {
                "add_global_tile": True,
                "max_patches": 16,
                "resize_to_max_canvas": False,
            },
            "vision_config": {
                "hidden_size": 1408,
                "num_hidden_layers": 34,
                "num_attention_heads": 16,
                "num_channels": 3,
                "intermediate_size": 5632,
                "image_size": 336,
                "patch_size": 14,
                "pixel_shuffle_ratio": 0.5,
                "projector_input_dim": 4096,
                "projector_output_dim": 4096,
                "vision_output_dim": 4096,
            },
            "text_config": {
                "num_hidden_layers": 48,
                "hidden_size": 5120,
                "num_attention_heads": 40,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 8192,
                "intermediate_size_mlp": 16384,
                "vocab_size": 202048,
                "max_position_embeddings": 10485760,
                "num_experts_per_tok": 1,
                "num_local_experts": 16,
                "interleave_moe_layer_step": 1,
                "attention_chunk_size": 8192,
            },
        }

        result = _parse_hf_config_json(config)

        assert result["architecture"] == "Llama4ForConditionalGeneration"
        assert result["layers"] == 48
        assert result["num_experts"] == 16
        assert result["moe_inter_size"] == 8192
        cfg = result["extra_params"]
        assert cfg is not None
        # step=1: all layers MoE
        assert all(m == 1 for m in cfg.moe_layer_freq)
        # empty no_rope_layers: every 4th layer has no RoPE and attends globally (1), the rest are chunked (0)
        assert cfg.attn_layer_pattern == tuple(int((i + 1) % 4 == 0) for i in range(48))
        assert sum(cfg.attn_layer_pattern) == 12
        assert cfg.sliding_window_size == 8192
        assert cfg.dense_inter_size == 16384
        assert cfg.vision_config is not None
        assert cfg.vision_config.image_size == 336
        # Llama 4 uses same dims for all layers → all four dim fields are 0
        assert cfg.swa_num_kv_heads == 0
        assert cfg.swa_head_dim == 0

    def test_parse_llama4_maverick_config(self):
        """Test Llama 4 Maverick (VLM, step=2: alternating MoE/dense) → HybridMoEConfig."""
        config = {
            "architectures": ["Llama4ForConditionalGeneration"],
            "model_type": "llama4",
            "image_processor_config": {
                "add_global_tile": True,
                "max_patches": 16,
                "resize_to_max_canvas": False,
            },
            "vision_config": {
                "hidden_size": 1408,
                "num_hidden_layers": 34,
                "num_attention_heads": 16,
                "num_channels": 3,
                "intermediate_size": 5632,
                "image_size": 336,
                "patch_size": 14,
                "pixel_shuffle_ratio": 0.5,
                "projector_input_dim": 4096,
                "projector_output_dim": 4096,
                "vision_output_dim": 4096,
            },
            "text_config": {
                "num_hidden_layers": 48,
                "hidden_size": 5120,
                "num_attention_heads": 40,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 8192,
                "intermediate_size_mlp": 16384,
                "vocab_size": 202048,
                "max_position_embeddings": 1048576,
                "num_experts_per_tok": 1,
                "num_local_experts": 128,
                "interleave_moe_layer_step": 2,
                "attention_chunk_size": 8192,
            },
        }

        result = _parse_hf_config_json(config)

        assert result["architecture"] == "Llama4ForConditionalGeneration"
        assert result["num_experts"] == 128
        cfg = result["extra_params"]
        # step=2: odd layers are MoE (1), even layers are dense (0)
        assert sum(cfg.moe_layer_freq) == 24  # 24 MoE layers
        assert cfg.moe_layer_freq.count(0) == 24  # 24 dense layers
        assert cfg.dense_inter_size == 16384
        assert cfg.vision_config is not None
        assert cfg.vision_config.image_size == 336

    def test_parse_mimov2flash_config(self):
        """Test MiMo-V2-Flash (explicit per-layer patterns, different SWA/global dims) → HybridMoEConfig."""
        hybrid_pattern = [0, 1, 1, 1, 1, 0, 1, 1, 1, 1]  # 10-layer test
        moe_freq = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        hf_config = {
            "architectures": ["MiMoV2FlashForCausalLM"],
            "num_hidden_layers": 10,
            "hidden_size": 4096,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 192,
            "v_head_dim": 128,
            "intermediate_size": 16384,
            "vocab_size": 152576,
            "max_position_embeddings": 262144,
            "num_experts_per_tok": 8,
            "num_local_experts": 64,
            "moe_intermediate_size": 2048,
            "hybrid_layer_pattern": hybrid_pattern,
            "moe_layer_freq": moe_freq,
            "swa_num_key_value_heads": 8,
            "swa_head_dim": 192,
            "swa_v_head_dim": 128,
            "sliding_window_size": 128,
        }

        result = _parse_hf_config_json(hf_config)

        assert result["architecture"] == "MiMoV2FlashForCausalLM"
        assert result["layers"] == 10
        cfg = result["extra_params"]
        assert cfg is not None
        # hybrid_layer_pattern marks sliding-window layers with 1; attn_layer_pattern marks global layers
        assert cfg.attn_layer_pattern == tuple(1 - v for v in hybrid_pattern)
        assert cfg.moe_layer_freq == tuple(moe_freq)
        assert cfg.swa_num_kv_heads == 8
        assert cfg.swa_head_dim == 192
        assert cfg.swa_v_head_dim == 128
        assert cfg.global_v_head_dim == 128  # from v_head_dim
        assert cfg.sliding_window_size == 128
        assert cfg.dense_inter_size == 0  # dense layers use model-level inter_size

    def test_mimo_pattern_length_mismatch_raises(self):
        """Test that mismatched hybrid pattern length raises ValueError."""
        hf_config = {
            "architectures": ["MiMoV2FlashForCausalLM"],
            "num_hidden_layers": 10,
            "hidden_size": 4096,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 192,
            "intermediate_size": 16384,
            "vocab_size": 152576,
            "max_position_embeddings": 262144,
            "num_experts_per_tok": 8,
            "num_local_experts": 64,
            "moe_intermediate_size": 2048,
            "hybrid_layer_pattern": [0, 1, 1],  # wrong length
            "moe_layer_freq": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
        with pytest.raises(ValueError, match="pattern length mismatch"):
            _parse_hf_config_json(hf_config)

    def test_llama4_invalid_step_raises(self):
        """Test that interleave_moe_layer_step <= 0 raises ValueError."""
        hf_config = {
            "architectures": ["Llama4ForConditionalGeneration"],
            "text_config": {
                "num_hidden_layers": 8,
                "hidden_size": 1024,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "intermediate_size": 2048,
                "vocab_size": 10000,
                "max_position_embeddings": 4096,
                "num_experts_per_tok": 1,
                "num_local_experts": 4,
                "interleave_moe_layer_step": 0,
                "attention_chunk_size": 512,
            },
        }
        with pytest.raises(ValueError, match="positive integer"):
            _parse_hf_config_json(hf_config)

    @staticmethod
    def _llama4_text_only_config(**text_overrides):
        text_config = {
            "num_hidden_layers": 8,
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "intermediate_size": 2048,
            "vocab_size": 10000,
            "max_position_embeddings": 4096,
            "num_experts_per_tok": 1,
            "num_local_experts": 4,
            "interleave_moe_layer_step": 1,
            "attention_chunk_size": 512,
        }
        text_config.update(text_overrides)
        return {
            "architectures": ["Llama4ForConditionalGeneration"],
            "model_type": "llama4",
            "image_processor_config": {
                "add_global_tile": True,
                "max_patches": 16,
                "resize_to_max_canvas": False,
            },
            "vision_config": {
                "hidden_size": 1408,
                "num_hidden_layers": 34,
                "num_attention_heads": 16,
                "num_channels": 3,
                "intermediate_size": 5632,
                "image_size": 336,
                "patch_size": 14,
                "pixel_shuffle_ratio": 0.5,
                "projector_input_dim": 4096,
                "projector_output_dim": 4096,
                "vision_output_dim": 4096,
            },
            "text_config": text_config,
        }

    def test_llama4_global_layers_follow_no_rope_layers(self):
        """Layers without RoPE (no_rope_layers 0) attend globally; an explicit list and an interval both count."""
        explicit = [1, 0, 1, 1, 1, 1, 1, 0]
        result = _parse_hf_config_json(self._llama4_text_only_config(no_rope_layers=explicit))
        assert result["extra_params"].attn_layer_pattern == (0, 1, 0, 0, 0, 0, 0, 1)

        result = _parse_hf_config_json(self._llama4_text_only_config(no_rope_layer_interval=2))
        assert result["extra_params"].attn_layer_pattern == (0, 1, 0, 1, 0, 1, 0, 1)

    def test_llama4_pattern_inputs_are_validated(self):
        """A non-positive interval or a non-list no_rope_layers raises instead of producing a wrong pattern."""
        for interval in (0, -1, None, "4"):
            with pytest.raises(ValueError, match="no_rope_layer_interval"):
                _parse_hf_config_json(self._llama4_text_only_config(no_rope_layer_interval=interval))
        with pytest.raises(TypeError, match="no_rope_layers"):
            _parse_hf_config_json(self._llama4_text_only_config(no_rope_layers=0))
        # An empty list means the interval rule, as in transformers.
        result = _parse_hf_config_json(self._llama4_text_only_config(no_rope_layers=[]))
        assert result["extra_params"].attn_layer_pattern == (0, 0, 0, 1, 0, 0, 0, 1)

    def test_llama4_no_rope_layers_length_mismatch_raises(self):
        """A no_rope_layers list that does not cover every layer raises ValueError."""
        with pytest.raises(ValueError, match="no_rope_layers"):
            _parse_hf_config_json(self._llama4_text_only_config(no_rope_layers=[1, 0, 1]))

    @staticmethod
    def _gemma4_text_config(layer_types):
        """Minimal valid Gemma 4 HF config wrapping the text_config branch."""
        return {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "text_config": {
                "num_hidden_layers": len(layer_types),
                "hidden_size": 2816,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "num_global_key_value_heads": 2,
                "head_dim": 256,
                "global_head_dim": 512,
                "intermediate_size": 2112,
                "moe_intermediate_size": 704,
                "vocab_size": 262144,
                "max_position_embeddings": 262144,
                "top_k_experts": 8,
                "num_experts": 128,
                "layer_types": layer_types,
                "sliding_window": 1024,
                "attention_k_eq_v": True,
            },
        }

    def test_parse_gemma4_config(self):
        """Real google/gemma-4-26B-A4B shape: 5:1 SWA:global, k_eq_v on global, top_k_experts."""
        # Exact pattern from config.json: [SWA, SWA, SWA, SWA, SWA, global] x 5 = 30 layers.
        layer_types = ["sliding_attention"] * 5 + ["full_attention"]
        layer_types = layer_types * 5
        hf_config = self._gemma4_text_config(layer_types)

        result = _parse_hf_config_json(hf_config)

        assert result["architecture"] == "Gemma4ForConditionalGeneration"
        assert result["layers"] == 30
        assert result["n"] == 16
        assert result["n_kv"] == 8  # SWA default at top level
        assert result["d"] == 256  # SWA default at top level
        assert result["hidden_size"] == 2816
        assert result["inter_size"] == 2112  # shared dense MLP intermediate
        assert result["moe_inter_size"] == 704  # per routed expert
        assert result["topk"] == 8  # picked up from top_k_experts fallback
        assert result["num_experts"] == 128
        assert result["vocab"] == 262144

        cfg = result["extra_params"]
        assert isinstance(cfg, common.Gemma4MixConfig)
        assert cfg.layer_types == tuple(layer_types)
        assert cfg.layer_types.count("sliding_attention") == 25
        assert cfg.layer_types.count("full_attention") == 5
        assert cfg.layer_types[-1] == "full_attention"
        assert cfg.swa_num_kv_heads == 8
        assert cfg.swa_head_dim == 256
        assert cfg.global_num_kv_heads == 2
        assert cfg.global_head_dim == 512
        assert cfg.sliding_window_size == 1024
        assert cfg.attention_k_eq_v is True
        assert cfg.vision_config is None

    def test_parse_gemma4_vision_config_alongside_text_config(self):
        """Gemma 4 keeps its fixed-budget pooled ViT contract beside the text/MoE config."""
        layer_types = (["sliding_attention"] * 5 + ["full_attention"]) * 5
        hf_config = self._gemma4_text_config(layer_types)
        hf_config["text_config"]["use_bidirectional_attention"] = "vision"
        hf_config["vision_soft_tokens_per_image"] = 280
        hf_config["vision_config"] = {
            "model_type": "gemma4_vision",
            "num_hidden_layers": 27,
            "hidden_size": 1152,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "head_dim": 72,
            "intermediate_size": 4304,
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "position_embedding_size": 10240,
            "default_output_length": 280,
            "standardize": True,
        }

        result = _parse_hf_config_json(hf_config)

        cfg = result["extra_params"]
        assert isinstance(cfg, common.Gemma4MixConfig)
        assert cfg.layer_types == tuple(layer_types)
        assert cfg.use_bidirectional_vision_attention is True
        vision = cfg.vision_config
        assert isinstance(vision, common.Gemma4VisionEncoderConfig)
        assert vision.depth == 27
        assert vision.hidden_size == 1152
        assert vision.num_heads == vision.num_key_value_heads == 16
        assert vision.head_dim == 72
        assert vision.intermediate_size == 4304
        assert vision.patch_size == 16
        assert vision.pooling_kernel_size == vision.spatial_merge_size == 3
        assert vision.position_embedding_size == 10240
        assert vision.soft_tokens_per_image == 280
        assert vision.supported_soft_token_budgets == (70, 140, 280, 560, 1120)
        assert vision.out_hidden_size == 2816
        assert vision.projector_dims == ((1152, 2816),)
        assert vision.projector_n_instances == 1
        assert vision.partial_rotary_factor == 1.0
        assert vision.standardize is True

    def test_gemma4_layer_types_length_mismatch_raises(self):
        """layer_types length must equal num_hidden_layers."""
        hf_config = self._gemma4_text_config(["sliding_attention"] * 5)
        hf_config["text_config"]["num_hidden_layers"] = 30  # model claims 30, config has 5
        with pytest.raises(ValueError, match="layer_types length"):
            _parse_hf_config_json(hf_config)

    def test_gemma4_invalid_layer_type_raises(self):
        """layer_types must contain only sliding_attention / full_attention."""
        hf_config = self._gemma4_text_config(["sliding_attention", "linear_attention", "full_attention"])
        with pytest.raises(ValueError, match="must contain only"):
            _parse_hf_config_json(hf_config)


class TestGemma4MixModelBuilder:
    """Builder-level tests that verify Gemma4MixModel wiring through set_gemma4_config."""

    @staticmethod
    def _make_model_config(tp_size=1, moe_tp_size=1, moe_ep_size=1):
        return config.ModelConfig(
            tp_size=tp_size,
            pp_size=1,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            attention_dp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            moe_quant_mode=common.MoEQuantMode.bfloat16,
        )

    @staticmethod
    def _make_model(model_config, layer_types=None, attention_k_eq_v=True):
        """Build a Gemma4MixModel at the gemma-4-26B-A4B shape (or a custom layer pattern)."""
        if layer_types is None:
            layer_types = (["sliding_attention"] * 5 + ["full_attention"]) * 5
        cfg = common.Gemma4MixConfig(
            layer_types=tuple(layer_types),
            swa_num_kv_heads=8,
            swa_head_dim=256,
            global_num_kv_heads=2,
            global_head_dim=512,
            sliding_window_size=1024,
            attention_k_eq_v=attention_k_eq_v,
        )
        model = Gemma4MixModel(
            8,  # topk
            128,  # num_experts
            704,  # moe_inter_size
            "google/gemma-4-26B-A4B",  # model_path
            "GEMMA4MIX",  # model_family
            "Gemma4ForConditionalGeneration",  # architecture
            len(layer_types),  # num_layers
            16,  # num_heads
            8,  # num_kv_heads (SWA default)
            256,  # head_size (SWA default)
            2816,  # hidden_size
            2112,  # inter_size (shared MLP)
            262144,  # vocab_size
            262144,  # context_length
            model_config,
            None,  # extra_params (we use set_gemma4_config instead)
            backend_name="trtllm",
        )
        model.set_gemma4_config(cfg)
        return model

    @staticmethod
    def _make_dense_model():
        """Build a dense Gemma 4 variant (e.g. gemma-4-31B-it: topk=0, num_experts=None):
        every layer is just shared dense MLP + attention, no routed-MoE block."""
        layer_types = (["sliding_attention"] * 5 + ["full_attention"]) * 2  # 12 layers
        cfg = common.Gemma4MixConfig(
            layer_types=tuple(layer_types),
            swa_num_kv_heads=16,
            swa_head_dim=256,
            global_num_kv_heads=4,
            global_head_dim=512,
            sliding_window_size=1024,
            attention_k_eq_v=True,
        )
        model = Gemma4MixModel(
            0,  # topk = 0 (dense)
            None,  # num_experts = None (dense)
            21504,  # moe_inter_size (unused for dense, but kept aligned with HF config)
            "google/gemma-4-31B-it",
            "GEMMA4MIX",
            "Gemma4ForConditionalGeneration",
            len(layer_types),
            32,
            16,
            256,
            5376,
            21504,
            262144,
            262144,
            TestGemma4MixModelBuilder._make_model_config(),  # tp=1, moe_tp=1, moe_ep=1
            None,
            backend_name="trtllm",
        )
        model.set_gemma4_config(cfg)
        return model

    def test_builds_both_layer_recipes(self):
        """30-layer 5:1 SWA:global pattern emits both recipes with shared-MLP + MoE ops."""
        model = self._make_model(self._make_model_config())
        op_names = {op._name for op in model.context_ops}
        # SWA recipe
        assert "context_swa_qkv_gemm" in op_names
        assert "context_swa_shared_mlp_gate_up_gemm" in op_names
        assert "context_swa_shared_mlp_down_gemm" in op_names
        assert "context_swa_router_gemm" in op_names
        # Global recipe
        assert "context_global_qkv_gemm" in op_names
        assert "context_global_shared_mlp_gate_up_gemm" in op_names
        assert "context_global_router_gemm" in op_names

    def test_global_qkv_omits_v_when_k_eq_v(self):
        """attention_k_eq_v=True drops V from the global-layer QKV-GEMM output width."""
        model = self._make_model(self._make_model_config(), attention_k_eq_v=True)
        global_qkv = next(op for op in model.context_ops if op._name == "context_global_qkv_gemm")
        swa_qkv = next(op for op in model.context_ops if op._name == "context_swa_qkv_gemm")
        # Global: Q (16*512) + K (2*512) = 9216, no V buffer.
        assert global_qkv._n == 16 * 512 + 2 * 512
        # SWA always has separate K and V: Q (16*256) + K (8*256) + V (8*256) = 8192.
        assert swa_qkv._n == 16 * 256 + 8 * 256 * 2

    def test_global_qkv_includes_v_when_k_eq_v_false(self):
        """attention_k_eq_v=False (defensive default) keeps V in the global QKV-out width."""
        model = self._make_model(self._make_model_config(), layer_types=["full_attention"], attention_k_eq_v=False)
        qkv = next(op for op in model.context_ops if op._name == "context_global_qkv_gemm")
        # With V re-enabled: Q + K + V = 16*512 + 2*512*2 = 10240.
        assert qkv._n == 16 * 512 + 2 * 512 * 2

    def test_kvcache_bytes_window_caps_swa(self):
        """SWA contribution caps at sliding_window_size; global grows linearly with seq_len."""
        model = self._make_model(self._make_model_config())

        # bf16, TP=1 hand derivation:
        #   per-token SWA layer = 8 KV * 256 dim * 2 (K+V) * 2 bytes = 8192
        #   per-token global layer = 2 KV * 512 dim * 2 (K+V) * 2 bytes = 4096
        def expected(seq_len):
            swa_seq = min(seq_len, 1024)
            return 25 * 8192 * swa_seq + 5 * 4096 * seq_len

        for seq_len in (512, 1024, 4096, 65536, 262144):
            assert model.get_kvcache_bytes_per_sequence(seq_len) == expected(seq_len)

    def test_kvcache_bytes_at_256k_in_expected_range(self):
        """Architectural payoff: full 256K context KV fits in ~5 GiB on a single GPU at bf16."""
        model = self._make_model(self._make_model_config())
        gib = model.get_kvcache_bytes_per_sequence(262144) / (1024**3)
        assert 4.5 <= gib <= 5.5, f"expected ~5.2 GiB, got {gib:.3f}"

    def test_kvcache_bytes_tp_sharding(self):
        """KV heads round up per GPU; TP=8 with KV-heads (8,2) → (1,1) per GPU."""
        model = self._make_model(self._make_model_config(tp_size=8, moe_ep_size=8))
        seq_len = 262144
        # SWA: 8/8 = 1 KV per GPU.  Global: ceil(2/8) = 1 KV per GPU.
        # SWA per-token per-layer per-GPU = 1*256*2*2 = 1024.  Global = 1*512*2*2 = 2048.
        expected = 25 * 1024 * 1024 + 5 * 2048 * seq_len
        assert model.get_kvcache_bytes_per_sequence(seq_len) == expected

    def test_get_kvcache_max_tokens_follows_window_capped_curve(self):
        """The capacity inverse caps SWA layers at the window instead of using the
        (larger) seq_len=1 slope, so it fits far more tokens past the window."""
        model = self._make_model(self._make_model_config())
        budget = model.get_kvcache_bytes_per_sequence(65536)  # >> 1024 window
        tokens = model.get_kvcache_max_tokens(budget)
        # Exact monotonic inverse: `tokens` fits, one more token does not.
        assert tokens == 65536
        assert model.get_kvcache_bytes_per_sequence(tokens) <= budget
        assert model.get_kvcache_bytes_per_sequence(tokens + 1) > budget
        # The seq_len=1 extrapolation under-counts: SWA layers are charged forever.
        per_token = model.get_kvcache_bytes_per_sequence(1)
        assert tokens > int(budget // per_token)

    def test_get_kvcache_max_tokens_linear_below_window(self):
        """Below the window every layer still grows, so the inverse is plain
        floor-division by the per-token size."""
        model = self._make_model(self._make_model_config())
        per_token = model.get_kvcache_bytes_per_sequence(1)
        budget = model.get_kvcache_bytes_per_sequence(512)  # < 1024 window
        assert model.get_kvcache_max_tokens(budget) == int(budget // per_token) == 512

    def test_get_kvcache_max_tokens_saturated_caps_at_context_length(self):
        """A fully window-capped cache (all SWA, no global layers) saturates: KV
        stops growing past the window, so memory never binds and capacity is
        bounded by the model context length -- not an arbitrary doubling step."""
        # No full_attention layer -> every layer caps at the 1024 window.
        model = self._make_model(self._make_model_config(), layer_types=["sliding_attention"] * 6)
        # KV is flat past the window: confirm we are actually in the saturated regime.
        assert model.get_kvcache_bytes_per_sequence(2048) == model.get_kvcache_bytes_per_sequence(1024)
        saturated = model.get_kvcache_bytes_per_sequence(model._context_length)
        # Any budget at/above the saturated size returns the context length, and is
        # stable across wildly different (large) budgets rather than tracking 2^k.
        assert model.get_kvcache_max_tokens(saturated) == model._context_length == 262144
        assert model.get_kvcache_max_tokens(saturated * 1000) == model._context_length

    def test_set_gemma4_config_rejects_wrong_type(self):
        """Passing a HybridMoEConfig (or any non-Gemma4MixConfig) raises."""
        model = Gemma4MixModel(
            8,
            128,
            704,
            "test",
            "GEMMA4MIX",
            "Gemma4ForConditionalGeneration",
            2,
            16,
            8,
            256,
            2816,
            2112,
            262144,
            262144,
            self._make_model_config(),
            None,
            backend_name="trtllm",
        )
        with pytest.raises(ValueError, match="requires a Gemma4MixConfig"):
            model.set_gemma4_config(common.HybridMoEConfig(attn_layer_pattern=(0, 1), moe_layer_freq=(1, 1)))

    def test_set_gemma4_config_rejects_wrong_layer_count(self):
        """layer_types length must match num_layers passed at construction."""
        model = Gemma4MixModel(
            8,
            128,
            704,
            "test",
            "GEMMA4MIX",
            "Gemma4ForConditionalGeneration",
            30,
            16,
            8,
            256,
            2816,
            2112,
            262144,
            262144,
            self._make_model_config(),
            None,
            backend_name="trtllm",
        )
        bad_cfg = common.Gemma4MixConfig(
            layer_types=("sliding_attention",) * 5,  # only 5, but num_layers=30
            swa_num_kv_heads=8,
            swa_head_dim=256,
            global_num_kv_heads=2,
            global_head_dim=512,
            sliding_window_size=1024,
        )
        with pytest.raises(ValueError, match="layer_types length"):
            model.set_gemma4_config(bad_cfg)

    def test_dense_variant_builds_without_moe_ops(self):
        """Dense Gemma 4 variants (e.g. gemma-4-31B-it: topk=0, num_experts=None) have
        no routed-MoE block: every layer is just shared dense MLP + attention.

        Regression test for the assertion crash when num_experts is None.
        """
        model = self._make_dense_model()

        op_names = {op._name for op in model.context_ops}
        # Shared dense MLP ops MUST be present.
        assert "context_swa_shared_mlp_gate_up_gemm" in op_names
        assert "context_global_shared_mlp_gate_up_gemm" in op_names
        # Routed-MoE ops MUST NOT be present.
        assert not any("moe" in n.lower() or "router" in n.lower() for n in op_names), (
            f"dense variant should not emit MoE/router ops, found: "
            f"{[n for n in op_names if 'moe' in n.lower() or 'router' in n.lower()]}"
        )
        gen_names = {op._name for op in model.generation_ops}
        assert not any("moe" in n.lower() or "router" in n.lower() for n in gen_names)

    def test_dense_variant_memory_usage_no_crash(self):
        """Dense Gemma 4 (num_experts=None) must flow through _get_memory_usage without
        crashing on the missing MoE workspace; activations fall back to MIN_ACTIVATION_BYTES."""
        model = self._make_dense_model()

        database = SimpleNamespace(
            system_spec={
                "misc": {
                    "nccl_mem": {1: 0},
                    "other_mem": 0,
                }
            }
        )
        memory = BaseBackend()._get_memory_usage(
            model,
            database,
            batch_size=1,
            beam_width=1,
            isl=1,
            osl=1,
        )

        assert memory["activations"] == pytest.approx(BaseBackend.MIN_ACTIVATION_BYTES / (1 << 30))
        assert memory["total"] > 0

    def test_dense_variant_rejects_moe_ep_gt_1(self):
        """Dense Gemma 4 has no experts, so any moe_ep_size > 1 must be rejected
        (otherwise pareto search would enumerate equivalent dense configurations)."""
        bad_config = config.ModelConfig(
            tp_size=2,
            pp_size=1,
            moe_tp_size=1,
            moe_ep_size=2,
            attention_dp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            moe_quant_mode=common.MoEQuantMode.bfloat16,
        )
        with pytest.raises(AssertionError, match="moe_ep_size=1"):
            Gemma4MixModel(
                0,
                None,
                21504,
                "google/gemma-4-31B-it",
                "GEMMA4MIX",
                "Gemma4ForConditionalGeneration",
                2,
                32,
                16,
                256,
                5376,
                21504,
                262144,
                262144,
                bad_config,
                None,
                backend_name="trtllm",
            )


class TestHybridMoEModelBuilder:
    """Builder-level tests that verify HybridMoEModel wiring through set_hybrid_config."""

    @staticmethod
    def _make_model_config():
        return config.ModelConfig(
            tp_size=1,
            pp_size=1,
            moe_tp_size=1,
            moe_ep_size=1,
            attention_dp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            moe_quant_mode=common.MoEQuantMode.bfloat16,
        )

    def test_mimov2flash_model_builds_all_three_layer_types(self):
        """MiMo-V2-Flash config produces context/generation ops for global_dense, global_moe, swa_moe."""
        hybrid_cfg = common.HybridMoEConfig(
            attn_layer_pattern=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0),
            moe_layer_freq=(0, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            swa_num_kv_heads=8,
            swa_head_dim=192,
            swa_v_head_dim=128,
            global_v_head_dim=128,
            sliding_window_size=128,
        )
        model = HybridMoEModel(
            8,
            64,
            2048,  # topk, num_experts, moe_inter_size
            "test-model",
            "HYBRIDMOE",
            "MiMoV2FlashForCausalLM",
            10,
            64,
            4,
            192,
            4096,
            16384,
            152576,
            262144,
            self._make_model_config(),
            backend_name="trtllm",
        )
        model.set_hybrid_config(hybrid_cfg)

        assert len(model.context_ops) > 0
        assert len(model.generation_ops) > 0
        op_names = [op._name for op in model.context_ops]
        assert any("global" in n and "moe" in n for n in op_names), "Missing global_moe ops"
        assert any("swa" in n and "moe" in n for n in op_names), "Missing swa_moe ops"
        assert any("global" in n and "dense" in n for n in op_names), "Missing global_dense ops"
        assert not any("swa" in n and "dense" in n for n in op_names), "MiMo-V2-Flash has no swa_dense layer"

    def test_llama4_scout_model_builds_global_and_swa_moe(self):
        """Llama 4 Scout (step=1, all MoE) produces global_moe + swa_moe ops."""
        layers = 8
        hybrid_cfg = common.HybridMoEConfig(
            attn_layer_pattern=tuple(int((i + 1) % 4 == 0) for i in range(layers)),
            moe_layer_freq=tuple(1 for _ in range(layers)),
            sliding_window_size=8192,
            dense_inter_size=16384,
        )
        model = HybridMoEModel(
            1,
            16,
            8192,  # topk, num_experts, moe_inter_size
            "test-model",
            "HYBRIDMOE",
            "Llama4ForConditionalGeneration",
            layers,
            40,
            8,
            128,
            5120,
            8192,
            202048,
            10485760,
            self._make_model_config(),
            backend_name="trtllm",
        )
        model.set_hybrid_config(hybrid_cfg)

        assert len(model.context_ops) > 0
        assert len(model.generation_ops) > 0
        op_names = [op._name for op in model.context_ops]
        assert any("global" in n for n in op_names)
        assert any("swa" in n for n in op_names)
        assert not any("dense" in n for n in op_names), "Scout has no dense layers"

    def test_llama4_maverick_model_builds_global_moe_and_swa_dense(self):
        """Llama 4 Maverick (step=2) produces global_moe + swa_dense ops."""
        layers = 8
        step = 2
        hybrid_cfg = common.HybridMoEConfig(
            attn_layer_pattern=tuple(int((i + 1) % 4 == 0) for i in range(layers)),
            moe_layer_freq=tuple(1 if (i + 1) % step == 0 else 0 for i in range(layers)),
            sliding_window_size=8192,
            dense_inter_size=16384,
        )
        model = HybridMoEModel(
            1,
            128,
            8192,  # topk, num_experts, moe_inter_size
            "test-model",
            "HYBRIDMOE",
            "Llama4ForConditionalGeneration",
            layers,
            40,
            8,
            128,
            5120,
            8192,
            202048,
            1048576,
            self._make_model_config(),
            backend_name="trtllm",
        )
        model.set_hybrid_config(hybrid_cfg)

        assert len(model.context_ops) > 0
        op_names = [op._name for op in model.context_ops]
        assert any("global" in n and "moe" in n for n in op_names)
        assert any("dense" in n for n in op_names), "Maverick needs dense FFN ops"


class TestGetModelConfigFromHFID:
    """Test getting model config from HuggingFace ID."""

    @patch("aisim.sdk.utils._download_hf_json")
    @patch("aisim.sdk.utils._download_hf_config")
    def test_successful_download(self, mock_download, mock_download_quant):
        """Test successful download from HuggingFace."""
        mock_config = {
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "intermediate_size": 14336,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "num_experts_per_tok": 0,
        }
        mock_download.return_value = mock_config

        mock_download_quant.return_value = None

        model_id = "acme/Fake-Model-32B"
        result = get_model_config_from_model_path(model_id)

        assert result["architecture"] == "LlamaForCausalLM"  # architecture
        mock_download.assert_called_once_with(model_id)


class TestSafeMkdir:
    """Test safe_mkdir utility (existing tests can be expanded if needed)."""

    def test_safe_mkdir_exists(self):
        """Test that safe_mkdir function exists and is importable."""
        from aisim.sdk.utils import safe_mkdir

        assert callable(safe_mkdir)


class TestEnumerateTTFTTPOTConstraints:
    """Tests for request-latency driven TTFT/TPOT enumeration."""

    def test_constraints_respect_request_latency_and_include_explicit_ttft(self):
        """Passing request_latency + ttft yields tuples below the latency budget."""
        constraints = enumerate_ttft_tpot_constraints(osl=500, request_latency=12000, ttft=4000)

        expected_tpot = (12000 - 4000) / (500 - 1)
        assert any(ttft == 4000 and tpot == pytest.approx(expected_tpot) for ttft, tpot in constraints)
        assert all(ttft < 12000 for ttft, _ in constraints)
        assert all(tpot > 0 for _, tpot in constraints)

    def test_constraints_default_to_95_percent_ttft_when_not_provided(self):
        """When ttft is omitted, we fall back to 95% of request latency."""
        constraints = enumerate_ttft_tpot_constraints(osl=50, request_latency=1000)

        expected_ttft = 0.95 * 1000
        derived_pair = next((pair for pair in constraints if pair[0] == pytest.approx(expected_ttft)), None)
        assert derived_pair is not None
        assert derived_pair[1] == pytest.approx((1000 - 950) / (50 - 1))


class TestKVCacheQuantInference:
    @pytest.mark.parametrize("source", ["quantization_config", "hf_quant_config"])
    @pytest.mark.parametrize("kv_algo", ["none", " NONE "])
    @pytest.mark.parametrize(
        "weight_algo,gemm,moe",
        [("NVFP4", "nvfp4", "nvfp4"), ("FP8", "fp8_static", "fp8"), (None, None, None)],
    )
    def test_explicit_none_is_distinct_from_missing_metadata(self, source, kv_algo, weight_algo, gemm, moe):
        from aisim.sdk.models import _infer_quant_modes_from_raw_config
        from aisim.sdk.utils import _attach_inferred_quant_fields

        quant = {"quant_algo": weight_algo, "kv_cache_quant_algo": kv_algo}
        raw = {source: {"quantization": quant} if source == "hf_quant_config" else quant}
        _attach_inferred_quant_fields(raw)
        modes = _infer_quant_modes_from_raw_config(raw)

        assert raw["kv_cache_quant_algo"] == "none"
        assert quant["kv_cache_quant_algo"] == kv_algo
        assert modes["kvcache_quant_mode"] == common.KVCacheQuantMode.bfloat16
        assert modes["fmha_quant_mode"] == common.FMHAQuantMode.bfloat16
        assert modes.get("gemm_quant_mode") == (common.GEMMQuantMode[gemm] if gemm else None)
        assert modes.get("moe_quant_mode") == (common.MoEQuantMode[moe] if moe else None)

    @pytest.mark.parametrize("weight_algo", [None, "NVFP4"])
    @pytest.mark.parametrize(
        "kv_fields,normalized_kv",
        [
            ({}, None),
            ({"kv_cache_quant_algo": None}, None),
            ({"kv_cache_quant_algo": "FP8"}, "fp8"),
            ({"kv_cache_quant_algo": "E4M3"}, "fp8"),
            ({"kv_cache_quant_algo": "bfloat16"}, "bfloat16"),
            ({"kv_cache_quant_algo": "FP16"}, "bfloat16"),
        ],
    )
    def test_legacy_defaults_and_supported_algorithms_are_unchanged(self, weight_algo, kv_fields, normalized_kv):
        from aisim.sdk.models import _infer_quant_modes_from_raw_config
        from aisim.sdk.utils import _attach_inferred_quant_fields

        raw = _attach_inferred_quant_fields({"quantization_config": {"quant_algo": weight_algo, **kv_fields}})
        modes = _infer_quant_modes_from_raw_config(raw)

        assert raw.get("kv_cache_quant_algo") == normalized_kv
        if weight_algo:
            assert modes["kvcache_quant_mode"] == common.KVCacheQuantMode.fp8
            assert modes["fmha_quant_mode"] == common.FMHAQuantMode.fp8
        else:
            assert modes.get("kvcache_quant_mode") == (
                common.KVCacheQuantMode[normalized_kv] if normalized_kv else None
            )
            assert "fmha_quant_mode" not in modes

    # NVFP4 weights must not reset an NVFP4 KV declaration to FP8.
    @pytest.mark.parametrize(
        "kv_declaration",
        [
            {"kv_cache_quant_algo": "NVFP4"},
            {"kv_cache_scheme": "NVFP4"},
            {"kv_cache_scheme": {"num_bits": 4, "type": "float"}},
        ],
    )
    def test_modelopt_nvfp4_kv_declarations(self, kv_declaration):
        from aisim.sdk.models import _infer_quant_modes_from_raw_config
        from aisim.sdk.utils import _attach_inferred_quant_fields

        quant = {"quant_method": "modelopt", "quant_algo": "NVFP4", **kv_declaration}
        raw = _attach_inferred_quant_fields({"quantization_config": quant})
        assert _infer_quant_modes_from_raw_config(raw)["kvcache_quant_mode"] is common.KVCacheQuantMode.nvfp4

    def test_modelopt_unknown_kv_scheme_string_fails(self):
        from aisim.sdk.models import _infer_quant_modes_from_raw_config
        from aisim.sdk.utils import _attach_inferred_quant_fields

        quant = {"quant_method": "modelopt", "quant_algo": "NVFP4", "kv_cache_scheme": "NVFP4_AFFINE"}
        raw = _attach_inferred_quant_fields({"quantization_config": quant})
        with pytest.raises(ValueError, match="Unsupported kv cache algorithm"):
            _infer_quant_modes_from_raw_config(raw)

    @pytest.mark.parametrize("kv_algo", ["not-an-algorithm", "null", ["none"], {"algorithm": "none"}, True])
    @pytest.mark.parametrize("source", ["quantization_config", "hf_quant_config"])
    def test_unknown_or_malformed_kv_algorithms_still_fail(self, kv_algo, source):
        from aisim.sdk.models import _infer_quant_modes_from_raw_config
        from aisim.sdk.utils import _attach_inferred_quant_fields

        quant = {"quant_algo": "NVFP4", "kv_cache_quant_algo": kv_algo}
        raw = _attach_inferred_quant_fields({source: {"quantization": quant} if source == "hf_quant_config" else quant})
        with pytest.raises(ValueError, match="Unsupported kv cache algorithm"):
            _infer_quant_modes_from_raw_config(raw)

    @pytest.mark.parametrize("architecture", ["DeepseekV4ForCausalLM", "DeepseekV41ForCausalLM"])
    @pytest.mark.parametrize("weight_algo", [None, "NVFP4"])
    @pytest.mark.parametrize("kv_algo", [None, "none", "FP8", "bfloat16"])
    def test_deepseek_native_cache_requirement_is_preserved(self, architecture, weight_algo, kv_algo):
        from aisim.sdk.models import _infer_quant_modes_from_raw_config
        from aisim.sdk.utils import _attach_inferred_quant_fields

        raw = _attach_inferred_quant_fields(
            {"quantization_config": {"quant_algo": weight_algo, "kv_cache_quant_algo": kv_algo}}
        )
        modes = _infer_quant_modes_from_raw_config(raw, architecture)

        assert modes["kvcache_quant_mode"] == common.KVCacheQuantMode.fp8
        if weight_algo:
            assert modes["fmha_quant_mode"] == common.FMHAQuantMode.fp8
        else:
            assert "fmha_quant_mode" not in modes


class TestParseCompressedTensorsQuant:
    """Tests for parse_compressed_tensors_quant and its _categorize_ignore_pattern helper."""

    @staticmethod
    def _make_quant_config(num_bits: int, w_type: str, ignore: list | None = None) -> dict:
        return {
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": num_bits, "type": w_type},
                    "input_activations": None,
                    "output_activations": None,
                    "targets": ["Linear"],
                }
            },
            "ignore": ignore or [],
            "quant_method": "compressed-tensors",
        }

    # --- input guards ---

    def test_none_input(self):
        from aisim.sdk.utils import parse_compressed_tensors_quant

        assert parse_compressed_tensors_quant(None) == (None, frozenset())

    def test_non_dict_input(self):
        from aisim.sdk.utils import parse_compressed_tensors_quant

        assert parse_compressed_tensors_quant("not-a-dict") == (None, frozenset())

    def test_empty_config_groups(self):
        from aisim.sdk.utils import parse_compressed_tensors_quant

        assert parse_compressed_tensors_quant({"config_groups": {}}) == (None, frozenset())

    # --- base_algo detection ---

    def test_int4_base_algo(self):
        from aisim.sdk.utils import parse_compressed_tensors_quant

        algo, _ = parse_compressed_tensors_quant(self._make_quant_config(4, "int"))
        assert algo == "int4_wo"

    def test_int8_base_algo(self):
        from aisim.sdk.utils import parse_compressed_tensors_quant

        algo, _ = parse_compressed_tensors_quant(self._make_quant_config(8, "int"))
        assert algo == "int8_wo"

    def test_fp8_base_algo(self):
        from aisim.sdk.utils import parse_compressed_tensors_quant

        algo, _ = parse_compressed_tensors_quant(self._make_quant_config(8, "float"))
        assert algo == "fp8"

    @pytest.mark.parametrize(
        ("strategy", "block_structure"),
        [
            pytest.param("block", None, id="block-strategy"),
            pytest.param("group", [128, 128], id="block-structure"),
        ],
    )
    def test_block_fp8_base_algo(self, strategy, block_structure):
        """Either compressed-tensors block signal selects block-scaled FP8."""
        from aisim.sdk.utils import parse_compressed_tensors_quant

        cfg = self._make_quant_config(8, "float")
        cfg["config_groups"]["group_0"]["weights"].update(
            {
                "strategy": strategy,
                "block_structure": block_structure,
            }
        )
        algo, _ = parse_compressed_tensors_quant(cfg)
        assert algo == "fp8_block"

    # --- ignored_categories detection ---

    def test_no_ignore_empty_set(self):
        from aisim.sdk.utils import parse_compressed_tensors_quant

        _, ignored = parse_compressed_tensors_quant(self._make_quant_config(4, "int", ignore=[]))
        assert ignored == frozenset()

    def test_attention_pattern_detected(self):
        from aisim.sdk.utils import parse_compressed_tensors_quant

        _, ignored = parse_compressed_tensors_quant(self._make_quant_config(4, "int", ignore=["re:.*self_attn.*"]))
        assert "attention" in ignored

    def test_routing_experts_pattern_detected(self):
        from aisim.sdk.utils import parse_compressed_tensors_quant

        _, ignored = parse_compressed_tensors_quant(
            self._make_quant_config(4, "int", ignore=["re:.*mlp\\.experts\\..*"])
        )
        assert "routing_experts" in ignored

    def test_shared_experts_pattern_detected_not_routing(self):
        """re:.*shared_experts.* must categorize as shared_experts, NOT routing_experts."""
        from aisim.sdk.utils import parse_compressed_tensors_quant

        _, ignored = parse_compressed_tensors_quant(self._make_quant_config(4, "int", ignore=["re:.*shared_experts.*"]))
        assert "shared_experts" in ignored
        assert "routing_experts" not in ignored

    def test_dense_mlp_pattern_detected(self):
        from aisim.sdk.utils import parse_compressed_tensors_quant

        _, ignored = parse_compressed_tensors_quant(
            self._make_quant_config(4, "int", ignore=["re:.*mlp\\.(gate|up|gate_up|down)_proj.*"])
        )
        assert "dense_mlp" in ignored

    def test_lm_head_pattern_detected(self):
        from aisim.sdk.utils import parse_compressed_tensors_quant

        _, ignored = parse_compressed_tensors_quant(self._make_quant_config(4, "int", ignore=["lm_head"]))
        assert "lm_head" in ignored

    # --- Kimi K2.5 end-to-end ---

    def test_kimi_k25_actual_config(self):
        """Kimi K2.5: attention + shared_experts + dense_mlp + lm_head all ignored.
        Only routing experts are quantized → those four categories in ignored set."""
        from aisim.sdk.utils import parse_compressed_tensors_quant

        cfg = self._make_quant_config(
            4,
            "int",
            ignore=[
                "lm_head",
                "re:.*self_attn.*",
                "re:.*shared_experts.*",
                "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
            ],
        )
        base_algo, ignored = parse_compressed_tensors_quant(cfg)
        assert base_algo == "int4_wo"
        assert ignored == frozenset({"attention", "shared_experts", "dense_mlp", "lm_head"})
        assert "routing_experts" not in ignored

    # --- models.py mapping (end-to-end via _infer_quant_modes_from_raw_config) ---

    def test_kimi_k25_maps_to_correct_sdk_modes(self):
        """Kimi K2.5 compressed-tensors config → gemm=bfloat16, moe=int4_wo."""
        from aisim.sdk import common
        from aisim.sdk.models import _infer_quant_modes_from_raw_config

        raw_config = {
            "quant_algo": "compressed-tensors",
            "quantization_config": self._make_quant_config(
                4,
                "int",
                ignore=[
                    "lm_head",
                    "re:.*self_attn.*",
                    "re:.*shared_experts.*",
                    "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
                ],
            ),
        }
        overrides = _infer_quant_modes_from_raw_config(raw_config)
        assert overrides.get("gemm_quant_mode") is None  # not set → falls back to bfloat16
        assert overrides.get("moe_quant_mode") == common.MoEQuantMode.int4_wo

    def test_all_layers_quantized_maps_both_modes(self):
        """No ignore list → both gemm and moe overrides are set."""
        from aisim.sdk import common
        from aisim.sdk.models import _infer_quant_modes_from_raw_config

        raw_config = {
            "quant_algo": "compressed-tensors",
            "quantization_config": self._make_quant_config(4, "int", ignore=[]),
        }
        overrides = _infer_quant_modes_from_raw_config(raw_config)
        assert overrides.get("gemm_quant_mode") == common.GEMMQuantMode.int4_wo
        assert overrides.get("moe_quant_mode") == common.MoEQuantMode.int4_wo

    def test_redhat_dsv4_maps_block_fp8_attention_and_fp4_experts(self):
        """The real RedHatAI DSV4 quant layout maps attention and experts independently."""
        from aisim.sdk import common
        from aisim.sdk.models import _infer_quant_modes_from_raw_config

        fp8_weights = {
            "num_bits": 8,
            "type": "float",
            "strategy": "block",
            "block_structure": [128, 128],
        }
        raw_config = {
            "architecture": "DeepseekV4ForCausalLM",
            "expert_dtype": "fp4",
            "quant_algo": "compressed-tensors",
            "quantization_config": {
                "config_groups": {
                    "group_0": {
                        "targets": ["re:.*attn.*(fused_wqa_wkv|wq_b|wo_a|wo_b)$"],
                        "weights": fp8_weights,
                    },
                    "group_1": {
                        "targets": ["re:.*ffn.*(gate|up|down)_proj$"],
                        "weights": {
                            "num_bits": 4,
                            "type": "float",
                            "strategy": "tensor_group",
                        },
                    },
                },
                "ignore": [],
            },
        }

        overrides = _infer_quant_modes_from_raw_config(raw_config)
        assert overrides["gemm_quant_mode"] == common.GEMMQuantMode.fp8_block
        assert overrides["moe_quant_mode"] == common.MoEQuantMode.w4a8_mxfp4_mxfp8
        assert overrides["kvcache_quant_mode"] == common.KVCacheQuantMode.fp8

    def test_modelopt_mixed_precision_config_groups_map_sdk_modes(self):
        """ModelOpt MIXED_PRECISION config groups map FP8 dense layers and NVFP4 routed experts."""
        from aisim.sdk import common
        from aisim.sdk.models import _infer_quant_modes_from_raw_config
        from aisim.sdk.utils import _attach_inferred_quant_fields

        raw_config = _attach_inferred_quant_fields(
            {
                "quantization_config": {
                    "quant_algo": "MIXED_PRECISION",
                    "kv_cache_scheme": {"type": "float", "num_bits": 8},
                    "config_groups": {
                        "group_0": {
                            "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
                            "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
                            "targets": [
                                "backbone.layers.0.mixer.in_proj",
                                "backbone.layers.1.mixer.shared_experts.up_proj",
                            ],
                        },
                        "group_1": {
                            "weights": {
                                "dynamic": False,
                                "group_size": 16,
                                "num_bits": 4,
                                "type": "float",
                            },
                            "input_activations": {
                                "dynamic": False,
                                "group_size": 16,
                                "num_bits": 4,
                                "type": "float",
                            },
                            "targets": ["backbone.layers.1.mixer.experts.0.up_proj"],
                        },
                    },
                },
            }
        )

        overrides = _infer_quant_modes_from_raw_config(raw_config)

        assert raw_config["quant_algo"] == "mixed_precision"
        assert overrides["gemm_quant_mode"] == common.GEMMQuantMode.fp8_static
        assert overrides["moe_quant_mode"] == common.MoEQuantMode.nvfp4
        assert overrides["kvcache_quant_mode"] == common.KVCacheQuantMode.fp8
        assert overrides["fmha_quant_mode"] == common.FMHAQuantMode.fp8

    def test_modelopt_mixed_precision_hf_quant_layers_map_sdk_modes(self):
        """Standalone hf_quant_config.json MIXED_PRECISION metadata no longer raises unsupported quant_algo."""
        from aisim.sdk import common
        from aisim.sdk.models import _infer_quant_modes_from_raw_config
        from aisim.sdk.utils import _attach_inferred_quant_fields

        raw_config = _attach_inferred_quant_fields(
            {
                "hf_quant_config": {
                    "quantization": {
                        "quant_algo": "MIXED_PRECISION",
                        "kv_cache_quant_algo": "FP8",
                        "quantized_layers": {
                            "backbone.layers.0.mixer.in_proj": {"quant_algo": "FP8"},
                            "backbone.layers.1.mixer.shared_experts.up_proj": {"quant_algo": "FP8"},
                            "backbone.layers.1.mixer.experts.0.up_proj": {"quant_algo": "NVFP4"},
                        },
                    }
                },
            }
        )

        overrides = _infer_quant_modes_from_raw_config(raw_config)

        assert raw_config["quant_algo"] == "mixed_precision"
        assert overrides["gemm_quant_mode"] == common.GEMMQuantMode.fp8_static
        assert overrides["moe_quant_mode"] == common.MoEQuantMode.nvfp4
        assert overrides["kvcache_quant_mode"] == common.KVCacheQuantMode.fp8
        assert overrides["fmha_quant_mode"] == common.FMHAQuantMode.fp8

    def test_modelopt_mixed_precision_mxfp8_dense_nvfp4_experts_map_sdk_modes(self):
        """MXFP8 dense/attention layers ride the fp8_block approximation; NVFP4 routed experts stay nvfp4.

        Mirrors nvidia/MiniMax-M3-NVFP4 (ModelOpt MIXED_PRECISION): attention
        q/k/v/o + MSA index projections + dense MLP + shared expert = MXFP8,
        routed experts w1/w2/w3 = NVFP4 group_size 16, kv_cache_quant_algo null.
        """
        from aisim.sdk import common
        from aisim.sdk.models import _infer_quant_modes_from_raw_config
        from aisim.sdk.utils import _attach_inferred_quant_fields

        prefix = "language_model.model.layers"
        raw_config = _attach_inferred_quant_fields(
            {
                "hf_quant_config": {
                    "producer": {"name": "modelopt"},
                    "quantization": {
                        "quant_algo": "MIXED_PRECISION",
                        "kv_cache_quant_algo": None,
                        "exclude_modules": ["lm_head", "model.embed_tokens", f"{prefix}.3.block_sparse_moe.gate"],
                        "quantized_layers": {
                            f"{prefix}.3.self_attn.q_proj": {"quant_algo": "MXFP8"},
                            f"{prefix}.3.self_attn.o_proj": {"quant_algo": "MXFP8"},
                            f"{prefix}.3.self_attn.index_q_proj": {"quant_algo": "MXFP8"},
                            f"{prefix}.0.mlp.gate_proj": {"quant_algo": "MXFP8"},
                            f"{prefix}.3.block_sparse_moe.shared_experts.up_proj": {"quant_algo": "MXFP8"},
                            f"{prefix}.3.block_sparse_moe.experts.0.w1": {"quant_algo": "NVFP4", "group_size": 16},
                            f"{prefix}.3.block_sparse_moe.experts.0.w2": {"quant_algo": "NVFP4", "group_size": 16},
                        },
                    },
                },
            }
        )

        overrides = _infer_quant_modes_from_raw_config(raw_config)

        assert raw_config["quant_algo"] == "mixed_precision"
        assert overrides["gemm_quant_mode"] == common.GEMMQuantMode.fp8_block
        assert overrides["moe_quant_mode"] == common.MoEQuantMode.nvfp4
        # kv_cache_quant_algo=null and a non-fp8-family quant_algo: no kv/fmha
        # override, both fall through to the bfloat16 defaults in get_model.
        assert "kvcache_quant_mode" not in overrides
        assert "fmha_quant_mode" not in overrides

    def test_modelopt_mixed_precision_mxfp8_config_group_not_misread_as_per_tensor_fp8(self):
        """A config_groups MXFP8 entry (8-bit float, group_size 32) maps to fp8_block, not fp8_static."""
        from aisim.sdk import common
        from aisim.sdk.models import _infer_quant_modes_from_raw_config
        from aisim.sdk.utils import _attach_inferred_quant_fields

        raw_config = _attach_inferred_quant_fields(
            {
                "quantization_config": {
                    "quant_algo": "MIXED_PRECISION",
                    "config_groups": {
                        "group_0": {
                            "weights": {"dynamic": False, "num_bits": 8, "type": "float", "group_size": 32},
                            "targets": ["model.layers.0.self_attn.q_proj"],
                        },
                    },
                },
            }
        )

        overrides = _infer_quant_modes_from_raw_config(raw_config)

        assert overrides["gemm_quant_mode"] == common.GEMMQuantMode.fp8_block

    def test_modelopt_mixed_precision_mxfp8_wins_over_per_tensor_fp8_for_gemms(self):
        """A checkpoint mixing MXFP8 and per-tensor FP8 dense layers keeps the block-scaled lane."""
        from aisim.sdk import common
        from aisim.sdk.models import _infer_quant_modes_from_raw_config
        from aisim.sdk.utils import _attach_inferred_quant_fields

        raw_config = _attach_inferred_quant_fields(
            {
                "hf_quant_config": {
                    "quantization": {
                        "quant_algo": "MIXED_PRECISION",
                        "quantized_layers": {
                            "model.layers.0.self_attn.q_proj": {"quant_algo": "MXFP8"},
                            "model.layers.0.mlp.up_proj": {"quant_algo": "FP8"},
                            "model.layers.1.mixer.experts.0.up_proj": {"quant_algo": "MXFP8"},
                        },
                    }
                },
            }
        )

        overrides = _infer_quant_modes_from_raw_config(raw_config)

        assert overrides["gemm_quant_mode"] == common.GEMMQuantMode.fp8_block
        assert overrides["moe_quant_mode"] == common.MoEQuantMode.fp8_block

    def test_expert_dtype_fp4_is_deepseek_v4_specific(self):
        from aisim.sdk import common
        from aisim.sdk.models import _infer_quant_modes_from_raw_config

        raw_config = {"expert_dtype": "fp4"}

        llama_overrides = _infer_quant_modes_from_raw_config(raw_config, "LlamaForCausalLM")
        deepseek_v4_overrides = _infer_quant_modes_from_raw_config(raw_config, "DeepseekV4ForCausalLM")

        assert "moe_quant_mode" not in llama_overrides
        assert deepseek_v4_overrides["moe_quant_mode"] == common.MoEQuantMode.w4a8_mxfp4_mxfp8


class TestEnumerateParallelConfigSGLangMoE:
    """Test enumerate_parallel_config for SGLang MoE scenarios."""

    def test_sglang_non_wideep_moe_includes_moe_ep_gt_1(self):
        """Test that SGLang + enable_wideep=False includes configs with moe_ep > 1."""
        configs = enumerate_parallel_config(
            num_gpu_list=[1, 2, 4, 8],
            tp_list=[1, 2, 4, 8],
            pp_list=[1],
            dp_list=[1, 2, 4, 8],
            moe_tp_list=[1, 2, 4, 8],
            moe_ep_list=[1, 2, 4, 8],
            is_moe=True,
            backend=common.BackendName.sglang,
            enable_wideep=False,
        )
        assert len(configs) > 0, "Should generate at least one config"
        moe_ep_values = [c[4] for c in configs]
        assert any(ep > 1 for ep in moe_ep_values), (
            f"Should include at least one config with moe_ep > 1, got moe_ep values: {set(moe_ep_values)}"
        )

    def test_sglang_megamoe_excludes_moe_tp_gt_1(self):
        """MegaMoE is the only SGLang MoE backend that is still EP-only in the
        enumerator: large EP is decided per config from data coverage now, so
        the deprecated enable_wideep flag no longer narrows the search space."""
        configs = enumerate_parallel_config(
            num_gpu_list=[8, 16, 32],
            tp_list=[1, 2, 4, 8],
            pp_list=[1],
            dp_list=[1, 2, 4, 8, 16, 32],
            moe_tp_list=[1, 2, 4, 8],
            moe_ep_list=[8, 16, 32],
            is_moe=True,
            backend=common.BackendName.sglang,
            moe_backend="megamoe",
        )
        assert len(configs) > 0, "Should generate at least one config"
        # All configs should have moe_tp == 1 (EP-only for megamoe)
        for c in configs:
            assert c[3] == 1, f"MegaMoE config should have moe_tp=1, got {c}"

    def test_sglang_enable_wideep_is_inert(self):
        """The deprecated flag is accepted and ignored: the same lists enumerate
        the same configs with and without it (coverage decides large EP)."""
        lists = dict(
            num_gpu_list=[8, 16, 32],
            tp_list=[1, 2, 4, 8],
            pp_list=[1],
            dp_list=[1, 2, 4, 8, 16, 32],
            moe_tp_list=[1, 2, 4, 8],
            moe_ep_list=[8, 16, 32],
            is_moe=True,
            backend=common.BackendName.sglang,
        )
        assert enumerate_parallel_config(**lists, enable_wideep=True) == enumerate_parallel_config(
            **lists, enable_wideep=False
        )
        assert any(c[3] > 1 for c in enumerate_parallel_config(**lists, enable_wideep=True))

    def test_sglang_non_wideep_moe_allows_mixed_tp_ep(self):
        """Test that SGLang + enable_wideep=False allows configs with both moe_tp > 1 and moe_ep > 1."""
        configs = enumerate_parallel_config(
            num_gpu_list=[1, 2, 4, 8],
            tp_list=[1, 2, 4, 8],
            pp_list=[1],
            dp_list=[1, 2, 4, 8],
            moe_tp_list=[1, 2, 4, 8],
            moe_ep_list=[1, 2, 4, 8],
            is_moe=True,
            backend=common.BackendName.sglang,
            enable_wideep=False,
        )
        # Should include configs with moe_ep == 1 (pure TP)
        has_pure_tp = any(c[4] == 1 and c[3] > 1 for c in configs)
        # Should include configs with moe_ep > 1
        has_ep_gt_1 = any(c[4] > 1 for c in configs)
        # Should include truly mixed configs (both moe_tp > 1 and moe_ep > 1)
        has_mixed = any(c[3] > 1 and c[4] > 1 for c in configs)
        assert has_pure_tp, "Should include pure TP configs (moe_ep=1, moe_tp>1)"
        assert has_ep_gt_1, "Should include configs with moe_ep > 1"
        assert has_mixed, "Should include mixed configs with both moe_tp > 1 and moe_ep > 1"

    def test_sglang_deepep_moe_no_longer_narrows_the_enumeration(self):
        """``moe_backend="deepep_moe"`` was a wideEP selector; it is deprecated
        and inert in the enumerator now (Task narrows the deepep_moe LADDER to
        moe_tp=[1], and per-config large EP comes from coverage)."""
        configs = enumerate_parallel_config(
            num_gpu_list=[1, 2, 4, 8],
            tp_list=[1, 2, 4, 8],
            pp_list=[1],
            dp_list=[1, 2, 4, 8],
            moe_tp_list=[1, 2, 4, 8],
            moe_ep_list=[1, 2, 4, 8],
            is_moe=True,
            backend=common.BackendName.sglang,
            moe_backend="deepep_moe",
        )
        assert len(configs) > 0, "Should generate at least one config"
        assert any(c[3] > 1 for c in configs), "deepep_moe must not exclude moe_tp > 1 any more"
        moe_ep_values = [c[4] for c in configs]
        assert any(ep > 1 for ep in moe_ep_values), "Should include configs with moe_ep > 1"


class TestEnumerateParallelConfigVLLMMoE:
    """Test enumerate_parallel_config for vLLM MoE scenarios."""

    def test_vllm_excludes_simultaneous_moe_tp_and_ep(self):
        """vLLM should never have both moe_tp > 1 and moe_ep > 1."""
        configs = enumerate_parallel_config(
            num_gpu_list=[1, 2, 4, 8],
            tp_list=[1, 2, 4, 8],
            pp_list=[1],
            dp_list=[1, 2, 4, 8],
            moe_tp_list=[1, 2, 4, 8],
            moe_ep_list=[1, 2, 4, 8],
            is_moe=True,
            backend=common.BackendName.vllm,
        )
        assert len(configs) > 0, "Should generate at least one config"
        for c in configs:
            tp, pp, dp, moe_tp, moe_ep, cp = c
            assert not (moe_tp > 1 and moe_ep > 1), f"vLLM should not have both moe_tp > 1 and moe_ep > 1, got {c}"

    def test_vllm_allows_pure_moe_tp(self):
        """vLLM should allow configs with moe_tp > 1 and moe_ep == 1."""
        configs = enumerate_parallel_config(
            num_gpu_list=[1, 2, 4, 8],
            tp_list=[1, 2, 4, 8],
            pp_list=[1],
            dp_list=[1, 2, 4, 8],
            moe_tp_list=[1, 2, 4, 8],
            moe_ep_list=[1, 2, 4, 8],
            is_moe=True,
            backend=common.BackendName.vllm,
        )
        has_pure_tp = any(c[3] > 1 and c[4] == 1 for c in configs)
        assert has_pure_tp, "Should include pure MoE TP configs (moe_tp>1, moe_ep=1)"

    def test_vllm_allows_pure_moe_ep(self):
        """vLLM should allow configs with moe_tp == 1 and moe_ep > 1."""
        configs = enumerate_parallel_config(
            num_gpu_list=[1, 2, 4, 8],
            tp_list=[1, 2, 4, 8],
            pp_list=[1],
            dp_list=[1, 2, 4, 8],
            moe_tp_list=[1, 2, 4, 8],
            moe_ep_list=[1, 2, 4, 8],
            is_moe=True,
            backend=common.BackendName.vllm,
        )
        has_pure_ep = any(c[3] == 1 and c[4] > 1 for c in configs)
        assert has_pure_ep, "Should include pure MoE EP configs (moe_tp=1, moe_ep>1)"


# ── Qwen3VL config parsing constants ──────────────────────────────────────────

_QWEN3VL_ARCH = "Qwen3VLForConditionalGeneration"

# Minimal Qwen3VL config matching the actual downloaded structure
_QWEN3VL_HF_CONFIG = {
    "architectures": [_QWEN3VL_ARCH],
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "initializer_range": 0.02,
        "intermediate_size": 25600,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 64,
        "num_hidden_layers": 64,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_theta": 5000000,
        "use_cache": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": False,
    "vision_config": {
        "depth": 27,
        "hidden_size": 1152,
        "num_heads": 16,
        "intermediate_size": 4304,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": 5120,
    },
}


class TestQwen3VLConfigParsing:
    """Test that _parse_hf_config_json correctly unwraps text_config for Qwen3VL."""

    def test_architecture_preserved(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["architecture"] == _QWEN3VL_ARCH

    def test_llm_layers_from_text_config(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["layers"] == 64

    def test_llm_hidden_size_from_text_config(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["hidden_size"] == 5120

    def test_llm_attention_heads_from_text_config(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["n"] == 64

    def test_llm_kv_heads_from_text_config(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["n_kv"] == 8

    def test_llm_head_dim_from_text_config(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["d"] == 128

    def test_llm_inter_size_from_text_config(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["inter_size"] == 25600

    def test_llm_vocab_from_text_config(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["vocab"] == 151936

    def test_not_moe(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["topk"] == 0
        assert result["num_experts"] == 0


class TestQwen3VLVisionEncoderParsing:
    """Test that vision_config is captured before text_config unwrap and parsed correctly."""

    def test_extra_params_is_vision_encoder_config(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert isinstance(result["extra_params"], common.VisionEncoderConfig)

    def test_vision_encoder_depth(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["extra_params"].depth == 27

    def test_vision_encoder_hidden_size(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["extra_params"].hidden_size == 1152

    def test_vision_encoder_num_heads(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["extra_params"].num_heads == 16

    def test_vision_encoder_intermediate_size(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["extra_params"].intermediate_size == 4304

    def test_vision_encoder_patch_size(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["extra_params"].patch_size == 16

    def test_vision_encoder_temporal_patch_size(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["extra_params"].temporal_patch_size == 2

    def test_vision_encoder_spatial_merge_size(self):
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["extra_params"].spatial_merge_size == 2

    def test_vision_encoder_out_hidden_size_matches_llm(self):
        """out_hidden_size must equal LLM hidden_size for the projection to be valid."""
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["extra_params"].out_hidden_size == result["hidden_size"]

    def test_vision_config_not_lost_after_text_config_unwrap(self):
        """Regression: vision_config must be captured before the text_config overwrite."""
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["extra_params"] is not None

    def test_missing_vision_config_does_not_raise(self):
        """If vision_config is absent, extra_params should be None, not an error."""
        cfg = {**_QWEN3VL_HF_CONFIG}
        cfg.pop("vision_config")
        result = _parse_hf_config_json(cfg)
        assert result["extra_params"] is None

    def test_vision_encoder_deepstack_visual_indexes_default(self):
        """deepstack_visual_indexes defaults to empty tuple when absent from vision_config."""
        result = _parse_hf_config_json(_QWEN3VL_HF_CONFIG)
        assert result["extra_params"].deepstack_visual_indexes == ()

    def test_vision_encoder_deepstack_visual_indexes_populated(self):
        """deepstack_visual_indexes is parsed as a tuple when present in vision_config."""
        vision_cfg = {**_QWEN3VL_HF_CONFIG["vision_config"], "deepstack_visual_indexes": [8, 17, 26]}
        cfg = {**_QWEN3VL_HF_CONFIG, "vision_config": vision_cfg}
        result = _parse_hf_config_json(cfg)
        assert result["extra_params"].deepstack_visual_indexes == (8, 17, 26)


def test_qwen3_vl_pixel_budget_follows_the_checkpoint_processor(tmp_path, caplog):
    checkpoint = tmp_path / "with-processor"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(_QWEN3VL_HF_CONFIG))
    (checkpoint / "preprocessor_config.json").write_text(
        json.dumps({"size": {"shortest_edge": 3136, "longest_edge": 1003520}})
    )
    enc_cfg = get_model_config_from_model_path(str(checkpoint))["extra_params"]
    assert (enc_cfg.min_pixels, enc_cfg.max_pixels) == (3136, 1003520)

    invalid = tmp_path / "invalid-processor"
    invalid.mkdir()
    (invalid / "config.json").write_text(json.dumps(_QWEN3VL_HF_CONFIG))
    (invalid / "preprocessor_config.json").write_text("[]")
    with pytest.raises(ValueError, match="Qwen3-VL preprocessor_config.json must be an object"):
        get_model_config_from_model_path(str(invalid))

    # A checkpoint without the file keeps the pinned Qwen3-VL budget instead of failing.
    bare = tmp_path / "without-processor"
    bare.mkdir()
    (bare / "config.json").write_text(json.dumps(_QWEN3VL_HF_CONFIG))
    with caplog.at_level(logging.WARNING):
        enc_cfg = get_model_config_from_model_path(str(bare))["extra_params"]
    assert (enc_cfg.min_pixels, enc_cfg.max_pixels) == utils.QWEN3_VL_PIXEL_BOUNDS
    assert "no pixel budget" in caplog.text
