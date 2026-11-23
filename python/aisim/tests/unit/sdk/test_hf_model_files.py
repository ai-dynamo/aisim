# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face model metadata: on-disk cache, commit pinning, offline mode, retry and prefetch (no network)."""

from __future__ import annotations

import copy
import email.utils
import io
import json
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

from aisim_core.sdk import hf_model_files as hf
from aisim_core.sdk import utils
from aisim_core.sdk.common import DefaultHFModels
from aisim_core.sdk.hf_model_files import default_cache_root as real_default_cache_root

pytestmark = pytest.mark.unit

MODEL = "acme/Fake-Model"
SHA_A, SHA_B = "a" * 40, "b" * 40
CONFIG = {
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


def blob(data) -> bytes:
    return json.dumps(data).encode()


class FakeTransport:
    """A Hugging Face that serves ``files[(repo, sha, name)]`` and resolves ``refs[(repo, revision)]``."""

    def __init__(self, files=None, refs=None, errors=None):
        self.files = {(MODEL, SHA_A, "config.json"): blob(CONFIG), **(files or {})}
        self.refs = {(MODEL, "main"): SHA_A, **(refs or {})}
        self.errors = errors or {}  # (repo, name or "@revision") -> exception to raise
        self.calls: list[tuple[str, ...]] = []

    def resolve_revision(self, repo, revision):
        self.calls.append(("resolve", repo, revision))
        if (repo, f"@{revision}") in self.errors:
            raise self.errors[(repo, f"@{revision}")]
        if (repo, revision) not in self.refs:
            raise hf.HuggingFaceHTTPError(f"{repo}@{revision} not found", 404)
        return self.refs[(repo, revision)]

    def read_file(self, repo, sha, filename):
        self.calls.append(("read", repo, sha, filename))
        if (repo, filename) in self.errors:
            raise self.errors[(repo, filename)]
        return self.files.get((repo, sha, filename))

    @property
    def reads(self):
        return [c[3] for c in self.calls if c[0] == "read"]


@pytest.fixture
def cache(tmp_path):
    return hf.ModelFileCache(tmp_path / "models")


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """Nothing in this module reaches Hugging Face: a test injects a transport or none may be built."""

    def refuse():
        raise AssertionError("the test built a real Hugging Face transport")

    monkeypatch.setattr(hf, "make_transport", refuse)


@pytest.fixture
def transport(monkeypatch):
    fake = FakeTransport()
    monkeypatch.setattr(hf, "make_transport", lambda: fake)
    return fake


# ------------------------------------------------------------------ cache and pin


def test_a_download_is_cached_and_read_before_any_network_call(cache):
    first = FakeTransport()
    assert hf.load_model_json(MODEL, "config.json", transport=first, cache=cache) == CONFIG
    assert first.calls == [("resolve", MODEL, "main"), ("read", MODEL, SHA_A, "config.json")]
    assert (cache.root / "acme" / "Fake-Model" / SHA_A / "config.json").is_file()

    second = FakeTransport()  # a new process: nothing in memory, only the disk
    assert hf.load_model_json(MODEL, "config.json", transport=second, cache=cache) == CONFIG
    assert second.calls == []


def test_a_branch_is_pinned_to_the_commit_it_resolved_to(cache):
    first = FakeTransport()
    hf.load_model_json(MODEL, "config.json", transport=first, cache=cache)
    moved = FakeTransport(
        files={(MODEL, SHA_B, "config.json"): blob({**CONFIG, "hidden_size": 8192})}, refs={(MODEL, "main"): SHA_B}
    )
    # main moved upstream; the pin keeps every later run on the commit it was first resolved to.
    assert hf.load_model_json(MODEL, "config.json", transport=moved, cache=cache) == CONFIG
    assert moved.calls == []
    assert cache.pinned(MODEL, "main") == SHA_A


def test_refresh_resolves_the_branch_again_and_keeps_the_old_commit_cached(cache):
    hf.load_model_json(MODEL, "config.json", transport=FakeTransport(), cache=cache)
    newer = {**CONFIG, "hidden_size": 8192}
    moved = FakeTransport(files={(MODEL, SHA_B, "config.json"): blob(newer)}, refs={(MODEL, "main"): SHA_B})
    assert hf.load_model_json(MODEL, "config.json", transport=moved, cache=cache, refresh=True) == newer
    assert cache.pinned(MODEL, "main") == SHA_B
    assert (cache.root / "acme" / "Fake-Model" / SHA_A / "config.json").is_file()
    assert hf.load_model_json(MODEL, "config.json", transport=FakeTransport(), cache=cache) == newer


def test_a_full_sha_needs_no_resolution_and_is_not_recorded_as_a_ref(cache):
    fake = FakeTransport()
    assert hf.load_model_json(MODEL, "config.json", revision=SHA_A, transport=fake, cache=cache) == CONFIG
    assert fake.calls == [("read", MODEL, SHA_A, "config.json")]
    assert not (cache.root / "acme" / "Fake-Model" / "refs").exists()


def test_other_revisions_resolve_and_pin_independently(cache):
    fake = FakeTransport(
        files={(MODEL, SHA_B, "config.json"): blob({"architectures": ["Other"]})}, refs={(MODEL, "release/v1"): SHA_B}
    )
    assert hf.load_model_json(MODEL, "config.json", revision="release/v1", transport=fake, cache=cache) == {
        "architectures": ["Other"]
    }
    assert cache.pinned(MODEL, "release/v1") == SHA_B and cache.pinned(MODEL, "main") is None
    assert (cache.root / "acme" / "Fake-Model" / "refs" / "release%2Fv1").is_file()


def test_the_cache_is_keyed_by_repository_and_commit(cache):
    other = "acme/Other-Model"
    fake = FakeTransport(
        files={(other, SHA_A, "config.json"): blob({"architectures": ["Other"]})}, refs={(other, "main"): SHA_A}
    )
    assert hf.load_model_json(MODEL, "config.json", transport=fake, cache=cache) == CONFIG
    assert hf.load_model_json(other, "config.json", transport=fake, cache=cache) == {"architectures": ["Other"]}
    assert cache.repos() == [MODEL, other]


def test_a_repository_without_an_owner_has_its_own_directory(cache):
    fake = FakeTransport(files={("gpt2", SHA_A, "config.json"): blob(CONFIG)}, refs={("gpt2", "main"): SHA_A})
    hf.load_model_json("gpt2", "config.json", transport=fake, cache=cache)
    assert (cache.root / "_" / "gpt2" / SHA_A / "config.json").is_file() and cache.repos() == ["gpt2"]


def test_a_file_the_repository_lacks_is_recorded_and_not_asked_for_again(cache):
    first = FakeTransport()
    assert hf.load_model_json(MODEL, "hf_quant_config.json", required=False, transport=first, cache=cache) is None
    assert (cache.root / "acme" / "Fake-Model" / SHA_A / "hf_quant_config.json.absent").is_file()
    second = FakeTransport()
    assert hf.load_model_json(MODEL, "hf_quant_config.json", required=False, transport=second, cache=cache) is None
    assert second.calls == []


def test_a_required_file_the_repository_lacks_raises_a_404(cache):
    fake = FakeTransport()
    del fake.files[(MODEL, SHA_A, "config.json")]
    with pytest.raises(hf.HuggingFaceHTTPError, match="no config.json") as caught:
        hf.load_model_json(MODEL, "config.json", transport=fake, cache=cache)
    assert caught.value.status == 404
    with pytest.raises(hf.HuggingFaceHTTPError):  # the absence is cached: still an error, now without the network
        hf.load_model_json(MODEL, "config.json", transport=FakeTransport(), cache=cache)


def test_a_corrupt_cache_entry_is_downloaded_again(cache, caplog):
    hf.load_model_json(MODEL, "config.json", transport=FakeTransport(), cache=cache)
    (cache.root / "acme" / "Fake-Model" / SHA_A / "config.json").write_text("{not json")
    fake = FakeTransport()
    assert hf.load_model_json(MODEL, "config.json", transport=fake, cache=cache) == CONFIG
    assert fake.reads == ["config.json"] and "unreadable cached" in caplog.text


def test_an_answer_that_is_not_json_is_an_error_and_is_not_cached(cache):
    fake = FakeTransport(files={(MODEL, SHA_A, "config.json"): b"<html>rate limited</html>"})
    with pytest.raises(hf.HuggingFaceDownloadError, match="not valid JSON"):
        hf.load_model_json(MODEL, "config.json", transport=fake, cache=cache)
    assert not (cache.root / "acme" / "Fake-Model" / SHA_A / "config.json").exists()


@pytest.mark.parametrize("repo", ["../evil", "a/b/c", "/abs/path", "", "a/..", "has space/x"])
def test_only_hugging_face_model_ids_reach_the_cache(cache, repo):
    with pytest.raises(hf.HuggingFaceDownloadError, match="Hugging Face model ID"):
        hf.load_model_json(repo, "config.json", transport=FakeTransport(), cache=cache)


@pytest.mark.parametrize("name", ["../config.json", "config.txt", "a/config.json", ".hidden.json"])
def test_only_json_file_names_reach_the_cache(cache, name):
    with pytest.raises(hf.HuggingFaceDownloadError, match="JSON file name"):
        hf.load_model_json(MODEL, name, transport=FakeTransport(), cache=cache)


def test_default_cache_root_is_inside_the_data_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("AISIM_DATA_CACHE", str(tmp_path / "data"))
    assert real_default_cache_root() == tmp_path / "data" / "model-configs"
    monkeypatch.delenv("AISIM_DATA_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert real_default_cache_root() == tmp_path / "xdg" / "aisim" / "data" / "model-configs"


# ------------------------------------------------------------------ offline


@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_offline_reads_the_cache_and_never_builds_a_transport(cache, monkeypatch, value):
    hf.load_model_json(MODEL, "config.json", transport=FakeTransport(), cache=cache)
    monkeypatch.setenv("AISIM_DATA_OFFLINE", value)
    assert hf.offline_requested()
    assert hf.load_model_json(MODEL, "config.json", cache=cache) == CONFIG  # a built transport would fail the test


def test_offline_with_a_cold_cache_names_the_model_and_the_command_that_fetches_it(cache, monkeypatch):
    monkeypatch.setenv("AISIM_DATA_OFFLINE", "1")
    with pytest.raises(hf.ModelFileMissingError) as caught:
        hf.load_model_json(MODEL, "config.json", cache=cache)
    message = str(caught.value)
    assert MODEL in message and "AISIM_DATA_OFFLINE" in message and str(cache.root) in message
    assert f"aisim data fetch --category model-config --model {MODEL}" in message
    assert isinstance(caught.value, hf.HuggingFaceDownloadError) and caught.value.code == "missing-data"


def test_offline_names_the_file_that_is_not_cached_under_a_known_pin(cache, monkeypatch):
    hf.load_model_json(MODEL, "config.json", transport=FakeTransport(), cache=cache)
    monkeypatch.setenv("AISIM_DATA_OFFLINE", "1")
    assert hf.load_model_json(MODEL, "config.json", cache=cache) == CONFIG
    with pytest.raises(hf.ModelFileMissingError, match="hf_quant_config.json at revision 'main' is not cached"):
        hf.load_model_json(MODEL, "hf_quant_config.json", required=False, cache=cache)


def test_offline_answers_an_optional_file_the_cache_knows_is_absent(cache, monkeypatch):
    hf.load_model_json(MODEL, "hf_quant_config.json", required=False, transport=FakeTransport(), cache=cache)
    monkeypatch.setenv("AISIM_DATA_OFFLINE", "1")
    assert hf.load_model_json(MODEL, "hf_quant_config.json", required=False, cache=cache) is None


def test_offline_does_not_resolve_a_branch_it_has_no_pin_for(cache, monkeypatch):
    monkeypatch.setenv("AISIM_DATA_OFFLINE", "1")
    with pytest.raises(hf.ModelFileMissingError, match="no pinned commit"):
        hf.load_model_json(MODEL, "config.json", revision="release/v1", cache=cache)


def test_offline_with_a_full_sha_reads_the_cache(cache, monkeypatch):
    hf.load_model_json(MODEL, "config.json", revision=SHA_A, transport=FakeTransport(), cache=cache)
    monkeypatch.setenv("AISIM_DATA_OFFLINE", "1")
    assert hf.load_model_json(MODEL, "config.json", revision=SHA_A, cache=cache) == CONFIG


def test_an_explicit_offline_flag_beats_the_environment(cache, monkeypatch):
    monkeypatch.setenv("AISIM_DATA_OFFLINE", "1")
    fake = FakeTransport()
    assert hf.load_model_json(MODEL, "config.json", transport=fake, cache=cache, offline=False) == CONFIG


# ------------------------------------------------------------------ the loader in utils


@pytest.fixture(autouse=True)
def _fresh_loader_caches():
    utils._load_model_config_from_model_path.cache_clear()
    utils.get_model_config_from_model_path.cache_clear()
    yield
    utils._load_model_config_from_model_path.cache_clear()
    utils.get_model_config_from_model_path.cache_clear()


def test_download_hf_json_goes_through_the_cache(transport):
    assert utils._download_hf_json(MODEL, "config.json") == CONFIG
    assert utils._download_hf_json(MODEL, "config.json") == CONFIG
    assert transport.reads == ["config.json"]
    assert utils._download_hf_json(MODEL, "hf_quant_config.json", raise_on_404=False) is None
    with pytest.raises(utils.HuggingFaceDownloadError):
        utils._download_hf_json(MODEL, "hf_quant_config.json")  # a required file the repository lacks
    assert transport.reads == ["config.json", "hf_quant_config.json"]


def test_download_hf_json_honors_an_explicit_revision(transport):
    transport.files[(MODEL, SHA_B, "config.json")] = blob({"architectures": ["Pinned"]})
    assert utils._download_hf_json(MODEL, "config.json", revision=SHA_B) == {"architectures": ["Pinned"]}
    assert ("resolve", MODEL, SHA_B) not in transport.calls


def test_a_non_bundled_model_loads_offline_after_one_online_load(transport, monkeypatch):
    transport.files[(MODEL, SHA_A, "hf_quant_config.json")] = blob(
        {"quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8"}}
    )
    online = utils.get_model_config_from_model_path(MODEL)
    assert online["architecture"] == "LlamaForCausalLM"
    assert online["raw_config"]["quant_algo"] == "nvfp4"

    utils._load_model_config_from_model_path.cache_clear()
    utils.get_model_config_from_model_path.cache_clear()
    transport.calls.clear()
    monkeypatch.setenv("AISIM_DATA_OFFLINE", "1")
    offline = utils.get_model_config_from_model_path(MODEL)
    assert offline["raw_config"] == online["raw_config"] and transport.calls == []


def test_offline_without_a_cache_is_a_clear_missing_data_error(monkeypatch):
    monkeypatch.setenv("AISIM_DATA_OFFLINE", "1")
    with pytest.raises(hf.ModelFileMissingError, match="aisim data fetch --category model-config"):
        utils.get_model_config_from_model_path("acme/Uncached-Model")


@pytest.mark.parametrize("status", [429, 500, 503, 401, 403])
def test_a_failed_quant_file_download_stops_the_run_instead_of_dropping_the_metadata(transport, status):
    transport.errors[(MODEL, "hf_quant_config.json")] = hf.HuggingFaceHTTPError(f"HTTP {status}", status)
    with pytest.raises(hf.HuggingFaceHTTPError) as caught:
        utils.get_model_config_from_model_path(MODEL)
    assert caught.value.status == status


def test_a_timeout_on_the_quant_file_also_stops_the_run(transport):
    transport.errors[(MODEL, "hf_quant_config.json")] = hf.HuggingFaceDownloadError("timed out")
    with pytest.raises(utils.HuggingFaceDownloadError, match="timed out"):
        utils.get_model_config_from_model_path(MODEL)


def test_a_missing_quant_file_is_tolerated(transport):
    info = utils.get_model_config_from_model_path(MODEL)
    assert (
        "hf_quant_config" not in info["raw_config"]
        and ("read", MODEL, SHA_A, "hf_quant_config.json") in transport.calls
    )


@pytest.mark.parametrize(
    ("bundled_model", "extra"),
    [
        ("meta-llama/Llama-4-Scout-17B-16E-Instruct", {"preprocessor_config.json"}),
        ("moonshotai/Kimi-K3", {"preprocessor_config.json", "video_preprocessor_config.json"}),
        ("Qwen/Qwen3-VL-8B-Instruct", {"preprocessor_config.json"}),
        ("nvidia/Llama-3.1-70B-Instruct-FP8", set()),
    ],
)
def test_prefetch_caches_exactly_the_files_the_loader_reads(monkeypatch, bundled_model, extra):
    """``model_files_to_cache`` and the loader must name the same files, or an offline run finds a gap."""
    raw = copy.deepcopy(utils._load_pre_downloaded_hf_config(bundled_model))
    processor = raw.pop("image_processor_config", {})  # the bundled Llama 4 copy inlines it; a remote repo ships a file
    model = "acme/Clone-Of-" + bundled_model.split("/")[1]
    fake = FakeTransport(
        files={(model, SHA_A, "config.json"): blob(raw), (model, SHA_A, "preprocessor_config.json"): blob(processor)},
        refs={(model, "main"): SHA_A},
    )
    monkeypatch.setattr(hf, "make_transport", lambda: fake)
    utils.get_model_config_from_model_path(model)
    loaded = {name for name in fake.reads if name != "config.json"}
    assert loaded == extra | {"hf_quant_config.json"}
    assert set(utils.model_files_to_cache(raw)) == loaded


def test_a_model_with_a_complete_cache_runs_offline_after_prefetch(cache, monkeypatch):
    fake = FakeTransport()
    entry = hf.prefetch_model(MODEL, transport=fake, cache=cache, extra_files=utils.model_files_to_cache)
    assert entry.status == "fetched" and entry.files == {"config.json": "fetched", "hf_quant_config.json": "absent"}
    monkeypatch.setenv("AISIM_DATA_OFFLINE", "1")
    monkeypatch.setattr(hf, "default_cache_root", lambda: cache.root)
    assert utils.get_model_config_from_model_path(MODEL)["architecture"] == "LlamaForCausalLM"


# ------------------------------------------------------------------ prefetch


def test_prefetch_reports_each_outcome(cache):
    gated, gone, limited = "acme/Gated", "acme/Gone", "acme/Limited"
    fake = FakeTransport(
        refs={(gated, "main"): SHA_A, (limited, "main"): SHA_A},
        errors={
            (gated, "config.json"): hf.HuggingFaceHTTPError("HTTP 403", 403),
            (limited, "config.json"): hf.HuggingFaceHTTPError("HTTP 429 after retries", 429),
        },
    )
    entries = {
        m: hf.prefetch_model(m, bundled={"acme/Bundled"}, transport=fake, cache=cache)
        for m in (MODEL, "acme/Bundled", gated, gone, limited)
    }
    assert entries[MODEL].status == "fetched" and entries[MODEL].revision == SHA_A
    assert entries["acme/Bundled"].status == "bundled"
    assert entries[gated].status == "skipped" and "gated or private" in entries[gated].reason
    assert "aisim init" in entries[gated].reason
    assert entries[gone].status == "skipped" and "not found" in entries[gone].reason
    assert entries[limited].status == "failed" and "429" in entries[limited].reason
    assert not any(c[1] == "acme/Bundled" for c in fake.calls)
    assert hf.prefetch_model(MODEL, transport=FakeTransport(), cache=cache).status == "cached"


def test_prefetch_of_a_bundled_model_with_an_explicit_revision_still_caches_it(cache):
    bundled = next(iter(DefaultHFModels))
    fake = FakeTransport(files={(bundled, SHA_A, "config.json"): blob(CONFIG)})
    entry = hf.prefetch_model(bundled, revision=SHA_A, bundled=DefaultHFModels, transport=fake, cache=cache)
    assert entry.status == "fetched" and entry.revision == SHA_A


def test_prefetch_offline_reports_missing_and_makes_no_calls(cache):
    fake = FakeTransport()
    entry = hf.prefetch_model(MODEL, transport=fake, cache=cache, offline=True)
    assert entry.status == "missing" and "aisim data fetch" in entry.reason and fake.calls == []


def test_prefetch_with_a_failing_optional_file_is_failed_not_cached_as_complete(cache):
    fake = FakeTransport(errors={(MODEL, "hf_quant_config.json"): hf.HuggingFaceHTTPError("HTTP 503", 503)})
    entry = hf.prefetch_model(MODEL, transport=fake, cache=cache)
    assert entry.status == "failed" and "503" in entry.reason
    retry = hf.prefetch_model(MODEL, transport=FakeTransport(), cache=cache)
    assert retry.status == "fetched" and retry.files == {"config.json": "cached", "hf_quant_config.json": "absent"}


def test_prefetch_never_raises_for_one_model(cache):
    class Broken(FakeTransport):
        def read_file(self, repo, sha, filename):
            raise RuntimeError("boom")

    entry = hf.prefetch_model(MODEL, transport=Broken(), cache=cache)
    assert entry.status == "failed" and entry.reason == "RuntimeError: boom"


# ------------------------------------------------------------------ manifest


def test_the_shipped_manifest_lists_only_model_ids():
    document = json.loads(hf.MANIFEST_PATH.read_text())
    assert set(document) == {"version", "description", "models"}
    models = hf.load_manifest()
    assert len(models) == len(set(models)) and list(models) == sorted(models, key=str.lower)
    assert all("/" in m and not m.endswith(".json") for m in models)
    assert set(DefaultHFModels) <= set(models)  # every model AISim bundles is in the manifest


def test_the_manifest_covers_every_bundled_config_file():
    configs = Path(utils.__file__).resolve().parents[1] / "model_configs"
    bundled = {
        p.name[: -len("_config.json")].replace("--", "/", 1)
        for p in configs.glob("*_config.json")
        if not p.name.endswith(("_hf_quant_config.json", "_preprocessor_config.json"))
    }
    assert bundled <= set(hf.load_manifest())


@pytest.mark.parametrize("body", ["{", '{"models": "x"}', '{"models": ["../x"]}', '{"models": [1]}', "[]"])
def test_a_malformed_manifest_is_refused(tmp_path, body):
    path = tmp_path / "manifest.json"
    path.write_text(body)
    with pytest.raises(hf.HuggingFaceDownloadError, match="manifest|model IDs"):
        hf.load_manifest(path)


# ------------------------------------------------------------------ retry and the HTTP transport


class Sleeps(list):
    def __call__(self, seconds):
        self.append(seconds)


def http_error(status, retry_after=None, url="https://huggingface.co/x"):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError(url, status, "Too Many Requests" if status == 429 else "Error", headers, io.BytesIO())


class FakeNetwork:
    """``urlopen`` that plays back ``script``: an exception is raised, bytes are returned as a response."""

    def __init__(self, *script):
        self.script, self.requests = list(script), []

    def __call__(self, request, *, timeout):
        self.requests.append((request.full_url, dict(request.header_items()), timeout))
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return io.BytesIO(step)


def http(network, sleeps=None, **kwargs):
    return hf.UrllibTransport(
        urlopen=network, sleep=sleeps if sleeps is not None else Sleeps(), headers=lambda: {}, **kwargs
    )


def test_a_rate_limit_is_retried_after_the_time_the_server_asks_for():
    sleeps = Sleeps()
    network = FakeNetwork(http_error(429, retry_after=3), b'{"a": 1}')
    assert json.loads(http(network, sleeps).read_file(MODEL, SHA_A, "config.json")) == {"a": 1}
    assert sleeps == [3.0] and len(network.requests) == 2


def test_a_rate_limit_without_retry_after_backs_off_exponentially_up_to_the_cap(caplog):
    sleeps = Sleeps()
    network = FakeNetwork(*[http_error(429) for _ in range(4)], b"{}")
    transport = http(network, sleeps, retry=hf.RetryPolicy(attempts=5, base_delay_s=1, max_delay_s=3))
    assert transport.read_file(MODEL, SHA_A, "config.json") == b"{}"
    assert sleeps == [1.0, 2.0, 3.0, 3.0]
    assert "retrying in 1 s" in caplog.text and "aisim init" in caplog.text


def test_a_rate_limit_that_persists_fails_naming_the_ways_to_authenticate():
    network = FakeNetwork(*[http_error(429, retry_after=1) for _ in range(5)])
    sleeps = Sleeps()
    with pytest.raises(hf.HuggingFaceHTTPError) as caught:
        http(network, sleeps).read_file(MODEL, SHA_A, "config.json")
    message = str(caught.value)
    assert caught.value.status == 429 and len(network.requests) == 5 and len(sleeps) == 4
    assert "rate limiting" in message and "HF_TOKEN" in message and "`aisim init`" in message
    assert "`hf auth login`" in message and "huggingface-cli" not in message


def test_a_retry_after_longer_than_the_limit_is_reported_instead_of_slept():
    network = FakeNetwork(http_error(429, retry_after=3600), b"{}")
    sleeps = Sleeps()
    with pytest.raises(hf.HuggingFaceHTTPError, match="asked to wait 3600 s"):
        http(network, sleeps).read_file(MODEL, SHA_A, "config.json")
    assert sleeps == [] and len(network.requests) == 1


def test_retry_after_may_be_an_http_date():
    when = email.utils.formatdate(1_000_000_030, usegmt=True)
    assert hf.parse_retry_after(when, now=lambda: 1_000_000_000.0) == pytest.approx(30.0)
    assert hf.parse_retry_after("7") == 7.0 and hf.parse_retry_after(" 2.5 ") == 2.5


@pytest.mark.parametrize("value", [None, "", "soon", "inf", "nan", "Wed, nonsense"])
def test_an_unusable_retry_after_is_ignored(value):
    assert hf.parse_retry_after(value) is None


def test_a_past_retry_after_date_means_no_wait():
    assert hf.parse_retry_after(email.utils.formatdate(1_000, usegmt=True), now=lambda: 2_000.0) == 0.0
    assert hf.parse_retry_after("-5") == 0.0


def test_transient_server_errors_are_retried():
    sleeps = Sleeps()
    network = FakeNetwork(http_error(503), http_error(502), b"{}")
    assert http(network, sleeps).read_file(MODEL, SHA_A, "config.json") == b"{}" and sleeps == [1.0, 2.0]


@pytest.mark.parametrize("status", [400, 401, 403, 410])
def test_client_errors_are_not_retried(status):
    network = FakeNetwork(http_error(status), b"{}")
    with pytest.raises(hf.HuggingFaceHTTPError) as caught:
        http(network).read_file(MODEL, SHA_A, "config.json")
    assert caught.value.status == status and len(network.requests) == 1


def test_a_gated_model_error_explains_how_to_get_access():
    with pytest.raises(hf.HuggingFaceHTTPError, match="gated or private model") as caught:
        http(FakeNetwork(http_error(403))).read_file(MODEL, SHA_A, "config.json")
    assert "accept its terms" in str(caught.value) and "`hf auth login`" in str(caught.value)


def test_a_missing_file_is_none_but_a_missing_repository_is_an_error():
    network = FakeNetwork(http_error(404), http_error(404))
    transport = http(network)
    assert transport.read_file(MODEL, SHA_A, "hf_quant_config.json") is None
    with pytest.raises(hf.HuggingFaceHTTPError) as caught:
        transport.resolve_revision(MODEL, "main")
    assert caught.value.status == 404 and len(network.requests) == 2


def test_a_connection_failure_is_a_download_error_without_retry():
    network = FakeNetwork(urllib.error.URLError("no route to host"), b"{}")
    with pytest.raises(hf.HuggingFaceDownloadError, match="no route to host"):
        http(network).read_file(MODEL, SHA_A, "config.json")
    assert len(network.requests) == 1


def test_urls_revisions_and_the_token_header():
    network = FakeNetwork(blob({"sha": SHA_A}), b"{}")
    transport = hf.UrllibTransport(
        urlopen=network, sleep=Sleeps(), headers=lambda: {"Authorization": "Bearer hf_secret"}
    )
    assert transport.resolve_revision(MODEL, "release/v1") == SHA_A
    transport.read_file(MODEL, SHA_A, "config.json")
    (resolve, headers, timeout), (read, _, _) = network.requests
    assert resolve == f"https://huggingface.co/api/models/{MODEL}/revision/release%2Fv1" and timeout == 30
    assert read == f"https://huggingface.co/{MODEL}/raw/{SHA_A}/config.json"
    assert headers["Authorization"] == "Bearer hf_secret"


@pytest.mark.parametrize("body", [b"not json", b"[]", blob({"sha": "main"}), blob({})])
def test_an_unusable_resolution_answer_is_an_error(body):
    with pytest.raises(hf.HuggingFaceDownloadError):
        http(FakeNetwork(body)).resolve_revision(MODEL, "main")


def test_the_token_never_appears_in_an_error(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    network = FakeNetwork(http_error(401, url="https://huggingface.co/acme/Fake-Model/raw/x/config.json"))
    transport = hf.UrllibTransport(urlopen=network, sleep=Sleeps())
    with pytest.raises(hf.HuggingFaceHTTPError) as caught:
        transport.read_file(MODEL, SHA_A, "config.json")
    assert "hf_secret" not in str(caught.value)
    assert network.requests[0][1]["Authorization"] == "Bearer hf_secret"


def test_the_token_location_in_errors_follows_hf_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    with pytest.raises(hf.HuggingFaceHTTPError, match=str(tmp_path / "hf" / "token")):
        http(FakeNetwork(http_error(401))).read_file(MODEL, SHA_A, "config.json")


def test_a_repository_that_cannot_be_read_leaves_no_pin_behind(cache):
    gated = "acme/Gated"
    fake = FakeTransport(
        refs={(gated, "main"): SHA_A}, errors={(gated, "config.json"): hf.HuggingFaceHTTPError("HTTP 403", 403)}
    )
    with pytest.raises(hf.HuggingFaceHTTPError):
        hf.load_model_json(gated, "config.json", transport=fake, cache=cache)
    assert cache.pinned(gated, "main") is None and cache.repos() == []
    # Once access is granted the branch resolves afresh and is pinned with its first file.
    del fake.errors[(gated, "config.json")]
    fake.files[(gated, SHA_B, "config.json")] = blob(CONFIG)
    fake.refs[(gated, "main")] = SHA_B
    assert hf.load_model_json(gated, "config.json", transport=fake, cache=cache) == CONFIG
    assert cache.pinned(gated, "main") == SHA_B
