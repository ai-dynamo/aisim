# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``aisim data`` fetch defaults, init, status, clear and validate, with a fake transport (no network)."""

import io
import json
import sys

import pyarrow as pa
import pytest

from aisim import data_cli
from aisim.cli_args import build_parser
from aisim.main import main
from aisim_core.data import Identity, configuration_id
from aisim_core.data.config import InitConfig, config_path, load_config, resolve_source, save_config, shipped_pin
from aisim_core.data.ensure import DataPin, _command, pin_from_environment
from aisim_core.data.fetch import Selection
from aisim_core.data.parquet import write_parquet
from aisim_core.sdk import hf_model_files

from .test_fetch import REPO, SHA, FakeTransport, NoNetwork, _artifact_json

pytestmark = pytest.mark.unit


def build_ops(tmp_path, hardwares=("h200_sxm", "b200_sxm")):
    """A small op-latency dataset: one operation-tables entry per hardware."""
    files, entries = {}, []
    for hardware in hardwares:
        table = pa.table({"x": [1, 2, 3]})
        local = tmp_path / f"{hardware}.parquet"
        write_parquet(table, local, "aisim.test/v1")
        data = local.read_bytes()
        art = f"data/op-latency/{hardware}/ops.parquet"
        files[art] = data
        identity = Identity(
            hardware=hardware,
            framework="vllm",
            framework_version="0.1",
            operation_family=None,
            model=None,
            effective_settings={},
            unresolved_identity=None,
        )
        cfg = configuration_id("op-latency", "operation-tables", identity)
        manifest = {
            "schema_version": "aisim-artifact-manifest/v1",
            "category": "op-latency",
            "purpose": "operation-tables",
            "configuration_id": cfg,
            "identity": identity.model_dump(),
            "requested_settings": None,
            "artifacts": [_artifact_json("a", art, data, rows=3)],
            "provenance": {},
            "evidence_gaps": [],
        }
        mpath = f"data/op-latency/{hardware}/manifest.json"
        files[mpath] = json.dumps(manifest).encode()
        entries.append(
            {
                "configuration_id": cfg,
                "purpose": "operation-tables",
                "identity": identity.model_dump(),
                "manifest": _artifact_json(
                    "m", mpath, files[mpath], format="json", schema="aisim-artifact-manifest/v1"
                ),
            }
        )
    index = {"schema_version": "aisim-category-index/v1", "category": "op-latency", "entries": entries}
    files["catalog/op-latency.json"] = json.dumps(index).encode()
    dataset = {
        "schema_version": "aisim-dataset-index/v1",
        "dataset_id": REPO,
        "categories": {
            "op-latency": {
                "root": "data/op-latency",
                "layout": "category",
                "index": _artifact_json(
                    "i",
                    "catalog/op-latency.json",
                    files["catalog/op-latency.json"],
                    format="json",
                    schema="aisim-category-index/v1",
                ),
            }
        },
    }
    files["catalog/dataset.json"] = json.dumps(dataset).encode()
    return files


_NO_CHECKER_PURPOSE = {"eval": "evaluation-record", "systems": "system-spec"}


def build_unchecked(root, category, *, catalog_hardware=None):
    """A one-entry dataset of a category with no offline checker, written under ``root``."""
    identity = Identity(
        hardware="h200_sxm",
        framework="vllm",
        framework_version="0.1",
        operation_family=None,
        model="m",
        effective_settings={},
        unresolved_identity=None,
    )
    purpose = _NO_CHECKER_PURPOSE[category]
    artifact = f"data/{category}/h200_sxm/record.json"
    data = b'{"record": 1}'
    manifest_path = f"data/{category}/h200_sxm/manifest.json"
    manifest = {
        "schema_version": "aisim-artifact-manifest/v1",
        "category": category,
        "purpose": purpose,
        "configuration_id": configuration_id(category, purpose, identity),
        "identity": identity.model_dump(),
        "requested_settings": None,
        "artifacts": [_artifact_json("a", artifact, data, format="json", schema="aisim.test/v1")],
        "provenance": {},
        "evidence_gaps": [],
    }
    manifest_bytes = json.dumps(manifest).encode()
    entry_identity = identity.model_dump()
    if catalog_hardware:  # a canonical but different identity than the manifest's
        entry_identity["hardware"] = catalog_hardware
    index = {
        "schema_version": "aisim-category-index/v1",
        "category": category,
        "entries": [
            {
                "configuration_id": manifest["configuration_id"],
                "purpose": purpose,
                "identity": entry_identity,
                "manifest": _artifact_json(
                    "m", manifest_path, manifest_bytes, format="json", schema="aisim-artifact-manifest/v1"
                ),
            }
        ],
    }
    index_bytes = json.dumps(index).encode()
    dataset = {
        "schema_version": "aisim-dataset-index/v1",
        "dataset_id": REPO,
        "categories": {
            category: {
                "root": f"data/{category}",
                "layout": "category",
                "index": _artifact_json(
                    "i", f"catalog/{category}.json", index_bytes, format="json", schema="aisim-category-index/v1"
                ),
            }
        },
    }
    for path, content in {
        artifact: data,
        manifest_path: manifest_bytes,
        f"catalog/{category}.json": index_bytes,
        "catalog/dataset.json": json.dumps(dataset).encode(),
    }.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(content)
    return root / artifact, root / manifest_path


@pytest.fixture
def env(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    for name in ("AISIM_DATASET", "AISIM_DATA_MAX_BYTES", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AISIM_DATA_CACHE", str(cache))
    monkeypatch.setattr("aisim_core.data.config.PIN_FILE", tmp_path / "absent-pin.json")
    # No Hugging Face credential, and no real prompt or login: a test that expects one overrides these.
    monkeypatch.setattr(data_cli, "_hf_token", lambda: None)
    monkeypatch.setattr(data_cli, "_read_token", lambda: pytest.fail("unexpected token prompt"))
    monkeypatch.setattr(data_cli, "_save_token", lambda token: pytest.fail("unexpected token save"))
    return cache


ALPHA, BETA, GATED, LIMITED = "acme/Alpha", "acme/Beta", "acme/Gated", "acme/Limited"
BUNDLED = "meta-llama/Meta-Llama-3.1-8B"  # a model AISim bundles: nothing to fetch
MODEL_SHA = "d" * 40


class FakeModelHub:
    """Hugging Face model repositories: every model has a ``config.json`` at ``MODEL_SHA``; two are not readable."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.denied = {GATED: 403, LIMITED: 429}

    def resolve_revision(self, repo, revision):
        self.calls.append(("resolve", repo, revision))
        return MODEL_SHA

    def read_file(self, repo, sha, filename):
        self.calls.append(("read", repo, filename))
        if repo in self.denied:
            raise hf_model_files.HuggingFaceHTTPError(f"HTTP {self.denied[repo]} for {repo}", self.denied[repo])
        return b'{"architectures": ["LlamaForCausalLM"]}' if filename == "config.json" else None


@pytest.fixture
def run(monkeypatch, tmp_path, capsys, env):
    files = build_ops(tmp_path)
    transport = FakeTransport(files)
    monkeypatch.setattr(data_cli, "make_transport", lambda: transport)
    hub = FakeModelHub()
    manifest = tmp_path / "hf_model_manifest.json"
    manifest.write_text(json.dumps({"version": 1, "models": [ALPHA, BETA, GATED, LIMITED, BUNDLED]}))
    monkeypatch.setattr(hf_model_files, "make_transport", lambda: hub)
    monkeypatch.setattr(hf_model_files, "MANIFEST_PATH", manifest)

    def _run(*argv):
        status = main(list(argv))
        out = capsys.readouterr()
        return status, out.out, out.err

    _run.transport, _run.files, _run.cache, _run.hub = transport, files, env, hub
    return _run


def _tty(monkeypatch, answer="", token=""):
    """A terminal: ``answer`` is what the mode question reads; ``token`` what the hidden token prompt returns."""

    class Stdin:
        def isatty(self):
            return True

        def readline(self):
            return answer

    def read_token():
        if isinstance(token, type) and issubclass(token, BaseException):
            raise token()
        if callable(token):  # pytest.fail: this test expects no prompt
            token("unexpected token prompt")
        return token

    monkeypatch.setattr(sys, "stdin", Stdin())
    monkeypatch.setattr(data_cli, "_read_token", read_token)


# ------------------------------------------------------------------ parser and fetch defaults


def test_aliases_share_the_fetch_parser(run):
    for argv in (["data", "fetch", "--dry-run"], ["fetch", "--dry-run"]):
        status, out, _ = run(*argv)
        payload = json.loads(out)
        assert status == 0 and payload["repo"] == "ai-dynamo/aisim-data" and payload["requested_revision"] == "main"


def test_no_arguments_downloads_whole_op_database(run):
    status, out, _ = run("fetch", "--revision", SHA)
    paths = json.loads(out)["paths"]
    assert status == 0
    assert {"data/op-latency/h200_sxm/ops.parquet", "data/op-latency/b200_sxm/ops.parquet"} <= set(paths)
    assert run.transport.resolves == 0


@pytest.mark.parametrize("value", [",", " , ,", ""])
def test_empty_configuration_selector_is_a_usage_error_not_the_whole_database(run, value):
    status, out, err = run("fetch", "--configuration", value)
    assert status == 2 and out == "" and "--configuration" in err
    assert run.transport.opened == [] and run.transport.resolves == 0  # rejected before any transport access


def test_all_flag_equals_default_and_selectors_narrow(run):
    default = json.loads(run("fetch", "--dry-run", "--revision", SHA)[1])
    explicit = json.loads(run("fetch", "--all", "--dry-run", "--revision", SHA)[1])
    narrow = json.loads(run("fetch", "--hardware", "h200_sxm", "--dry-run", "--revision", SHA)[1])
    assert default["files"] == explicit["files"]
    assert len(narrow["files"]) < len(default["files"])


@pytest.mark.parametrize("category", ["fpm", "e2e"])
def test_other_categories_stay_selective(run, category):
    status, out, err = run("fetch", "--category", category, "--revision", SHA)
    assert status == 2 and out == ""
    status, out, err = run("fetch", "--category", category, "--all", "--revision", SHA)
    assert status == 2 and "only available for op-latency" in err
    assert run.transport.opened == []


def test_no_default_byte_cap_and_env_limit_is_opt_in(run, monkeypatch):
    assert run("fetch", "--revision", SHA, "--dry-run")[0] == 0
    monkeypatch.setenv("AISIM_DATA_MAX_BYTES", "1")
    status, _, err = run("fetch", "--revision", SHA)
    assert status == 1 and json.loads(err.strip().splitlines()[-1])["error"] == "over-budget"
    assert run("fetch", "--revision", SHA, "--no-limit")[0] == 0


def test_the_byte_cap_counts_the_metadata_that_planning_downloads(run, monkeypatch, tmp_path):
    probe = json.loads(run("fetch", "--revision", SHA, "--dry-run", "--cache-dir", str(tmp_path / "probe"))[1])
    metadata, artifacts = probe["metadata_bytes"], probe["uncached_bytes"]
    assert metadata > 0 and artifacts > 0
    # The artifacts alone fit the cap; with the metadata that planning downloads first they do not.
    monkeypatch.setenv("AISIM_DATA_MAX_BYTES", str(metadata + artifacts - 1))
    transferred = len(run.transport.opened)
    status, _, err = run("fetch", "--revision", SHA, "--cache-dir", str(tmp_path / "fresh"))
    assert status == 1 and json.loads(err.strip().splitlines()[-1])["error"] == "over-budget"
    assert not any(path.endswith(".parquet") for path in run.transport.opened[transferred:])
    # At exactly the whole cost the command succeeds, reports the metadata it downloaded, and stays inside the cap.
    monkeypatch.setenv("AISIM_DATA_MAX_BYTES", str(metadata + artifacts))
    status, out, _ = run("fetch", "--revision", SHA, "--cache-dir", str(tmp_path / "exact"))
    payload = json.loads(out)
    assert status == 0 and payload["plan"]["metadata_bytes"] == metadata
    assert payload["downloaded_bytes"] + payload["plan"]["metadata_bytes"] <= metadata + artifacts


def test_large_download_announces_planned_size(run, monkeypatch):
    monkeypatch.setattr(data_cli, "LARGE_DOWNLOAD_BYTES", 1)
    status, _, err = run("fetch", "--revision", SHA)
    assert status == 0 and "downloading" in err


def test_default_revision_is_main_without_shipped_pin(env):
    assert shipped_pin() == ("ai-dynamo/aisim-data", "main")


def test_shipped_pin_file_is_used(env, tmp_path, monkeypatch):
    pin = tmp_path / "pin.json"
    pin.write_text(json.dumps({"repo": "ai-dynamo/aisim-data", "revision": SHA}))
    monkeypatch.setattr("aisim_core.data.config.PIN_FILE", pin)
    assert resolve_source() == ("ai-dynamo/aisim-data", SHA)


def test_malformed_shipped_pin_fails_but_absent_pin_falls_back(env, tmp_path, monkeypatch):
    from aisim_core.data.config import ConfigError

    pin = tmp_path / "pin.json"
    monkeypatch.setattr("aisim_core.data.config.PIN_FILE", pin)
    assert shipped_pin() == ("ai-dynamo/aisim-data", "main")  # absent
    for text in ("{truncated", json.dumps({"repo": "x/y"}), "[]"):
        pin.write_text(text)
        with pytest.raises(ConfigError, match="shipped pin"):
            shipped_pin()


def test_ensure_hints_are_valid_commands(run):
    pin = DataPin(repo=REPO, revision=SHA)
    for selection in (
        Selection(category="op-latency", purpose="operation-tables"),
        Selection(category="op-latency", hardware="h200_sxm", framework="vllm"),
        Selection(category="fpm", model="m", hardware="h200_sxm"),
    ):
        argv = _command(pin, selection).split()[1:]
        assert argv[:2] == ["data", "fetch"]
        args = build_parser().parse_args(argv)
        data_cli.build_selection(args)  # raises on an invalid selection


# ------------------------------------------------------------------ init and config


def test_init_without_terminal_or_flags_is_a_usage_error(run, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    status, out, err = run("init")
    assert status == 2 and out == ""
    assert "--all" in err and "--on-demand" in err and "--hardware" in err
    assert not config_path().exists() and run.transport.opened == []


def test_init_on_demand_saves_config_without_download(run):
    status, out, _ = run("init", "--on-demand")
    assert status == 0 and json.loads(out)["mode"] == "on-demand"
    assert load_config() == InitConfig(mode="on-demand")
    assert config_path() == run.cache / "config.json" and run.transport.opened == []


def test_init_all_downloads_everything_then_saves(run):
    status, out, _ = run("init", "--all", "--revision", SHA)
    payload = json.loads(out)
    assert status == 0 and len(payload["fetch"]["paths"]) == 6
    assert load_config().mode == "all"


def test_init_download_saves_the_source_it_used(run):
    status, _, _ = run("init", "--all")  # no flag, no saved source: the shipped default (a branch here) is downloaded
    assert status == 0
    # The branch resolved to a commit; the saved source names that commit, not the moving branch.
    assert load_config() == InitConfig(mode="all", repo=shipped_pin()[0], revision=SHA)
    pin = pin_from_environment({"AISIM_DATA_CACHE": str(run.cache)})
    assert (pin.repo, pin.revision) == (shipped_pin()[0], SHA)  # the downloaded data is what predictors select


def test_init_from_a_branch_saves_its_commit_and_fetches_offline(run, monkeypatch):
    status, _, _ = run("init", "--all", "--revision", "main")
    assert status == 0 and load_config().revision == SHA
    run.transport.refs["main"] = "b" * 40  # the branch moves; the saved source must not follow it
    monkeypatch.setattr(data_cli, "make_transport", lambda: NoNetwork())
    status, out, _ = run("fetch", "--offline")  # no flag: the saved commit is cached and needs no lookup
    payload = json.loads(out)
    assert status == 0 and payload["downloaded_bytes"] == 0 and payload["plan"]["revision"] == SHA


def test_init_hardware_downloads_one_platform(run):
    status, out, _ = run("init", "--hardware", "h200_sxm", "--revision", SHA)
    paths = json.loads(out)["fetch"]["paths"]
    assert status == 0 and "data/op-latency/b200_sxm/ops.parquet" not in paths
    assert load_config() == InitConfig(mode="hardware", hardware="h200_sxm", repo=shipped_pin()[0], revision=SHA)


def test_init_mode_change_keeps_saved_source(run):
    run("init", "--on-demand", "--repo", "x/y", "--revision", SHA)
    run("init", "--on-demand")
    assert load_config() == InitConfig(mode="on-demand", repo="x/y", revision=SHA)
    run("init", "--on-demand", "--repo", "z/w")  # a revision does not follow a different repository
    assert load_config() == InitConfig(mode="on-demand", repo="z/w")
    run("init", "--on-demand", "--revision", "main")  # revision only: kept with the saved repository
    assert load_config() == InitConfig(mode="on-demand", repo="z/w", revision="main")


def test_init_failure_saves_nothing(run, monkeypatch):
    monkeypatch.setattr(data_cli, "make_transport", lambda: NoNetwork())
    status, _, _ = run("init", "--all", "--offline", "--revision", SHA)
    assert status == 1 and load_config() is None


@pytest.mark.parametrize(("answer", "mode"), [("1\n", "all"), ("2\n", "on-demand")])
def test_init_prompt(run, monkeypatch, answer, mode):
    _tty(monkeypatch, answer)
    status, out, err = run("init", "--revision", SHA)
    assert status == 0 and "1) op database now" in err and "2) op data on demand" in err
    assert load_config().mode == mode


def test_init_prompt_eof_does_not_loop(run, monkeypatch):
    _tty(monkeypatch, "")
    assert run("init")[0] == 2


def _record_saved_tokens(monkeypatch):
    saved = []
    monkeypatch.setattr(data_cli, "_save_token", saved.append)
    return saved


def test_init_with_a_token_uses_it_without_asking(run, monkeypatch):
    monkeypatch.setattr(data_cli, "_hf_token", lambda: "hf_secret")
    _tty(monkeypatch, "2\n", token=pytest.fail)  # a prompt would call this and fail the test
    status, out, err = run("init", "--revision", SHA)
    assert status == 0 and "Using your Hugging Face token" in err and "No Hugging Face token" not in err
    assert "hf_secret" not in out + err and load_config().mode == "on-demand"


def test_init_without_a_token_or_terminal_explains_the_rate_limit(run):
    status, out, err = run("init", "--on-demand")
    assert status == 0 and json.loads(out)["mode"] == "on-demand"  # stdout stays one JSON object
    assert "rate limited" in err and "HF_TOKEN" in err and "hf auth login" in err


def test_init_offers_to_save_a_token_and_keeps_it_out_of_output_and_config(run, monkeypatch):
    saved = _record_saved_tokens(monkeypatch)
    _tty(monkeypatch, "2\n", token="  hf_secret \n")
    status, out, err = run("init", "--revision", SHA)
    assert status == 0 and saved == ["hf_secret"]
    assert "rate limited" in err and "Token saved" in err
    assert "hf_secret" not in out + err
    assert "hf_secret" not in config_path().read_text()


@pytest.mark.parametrize("token", ["", "   ", EOFError])
def test_init_enter_or_closed_input_skips_the_token(run, monkeypatch, token):
    saved = _record_saved_tokens(monkeypatch)
    _tty(monkeypatch, "2\n", token=token)
    status, _, err = run("init", "--revision", SHA)
    assert status == 0 and saved == [] and "Token saved" not in err and load_config().mode == "on-demand"


def test_init_rejected_token_continues_without_echoing_it(run, monkeypatch):
    def reject(token):
        raise RuntimeError(f"401 Unauthorized for {token}")  # a library message may carry the token

    monkeypatch.setattr(data_cli, "_save_token", reject)
    _tty(monkeypatch, token="hf_secret")
    status, out, err = run("init", "--on-demand")
    assert status == 0 and "not saved (RuntimeError)" in err and "hf_secret" not in out + err
    assert load_config().mode == "on-demand"


def test_init_offline_skips_the_token_step(run, monkeypatch):
    _tty(monkeypatch, token=pytest.fail)  # a prompt would call this and fail the test
    status, _, err = run("init", "--on-demand", "--offline")
    assert status == 0 and "Hugging Face token" not in err


def test_saving_a_token_goes_through_hugging_face_login_and_never_writes_stdout(monkeypatch, capsys):
    import huggingface_hub

    calls = []

    def noisy_login(token=None, **kwargs):
        calls.append((token, kwargs))
        print("Login successful")

    monkeypatch.setattr(huggingface_hub, "login", noisy_login)
    data_cli._save_token("hf_secret")
    captured = capsys.readouterr()
    assert calls == [("hf_secret", {"add_to_git_credential": False})]
    assert captured.out == "" and "Login successful" in captured.err


def test_init_flags_are_mutually_exclusive(run):
    with pytest.raises(SystemExit):
        run("init", "--all", "--on-demand")


def test_config_location_honors_cache_env(monkeypatch, tmp_path):
    monkeypatch.delenv("AISIM_DATA_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert config_path() == tmp_path / "xdg" / "aisim" / "config.json"
    monkeypatch.setenv("AISIM_DATA_CACHE", str(tmp_path / "dc"))
    assert config_path() == tmp_path / "dc" / "config.json"


def test_config_revision_applies_only_to_its_own_repo(env):
    config = InitConfig(mode="all", repo="cfg/repo", revision="cfgrev")
    assert resolve_source("other/data", config=config) == ("other/data", "main")
    assert resolve_source("cfg/repo", config=config) == ("cfg/repo", "cfgrev")


def test_fetch_reads_config_from_cache_dir(run, tmp_path):
    cache = tmp_path / "explicit"
    save_config(InitConfig(mode="all", repo=REPO, revision=SHA), config_path(cache_dir=cache))
    status, out, _ = run("data", "fetch", "--category", "op-latency", "--all", "--cache-dir", str(cache))
    assert status == 0 and json.loads(out)["plan"]["revision"] == SHA


def test_precedence_flag_env_config_shipped(env, monkeypatch):
    config = InitConfig(mode="all", repo="cfg/repo", revision="cfgrev")
    assert resolve_source(config=config) == ("cfg/repo", "cfgrev")
    monkeypatch.setenv("AISIM_DATASET", "env/repo@envrev")
    assert resolve_source(config=config) == ("env/repo", "envrev")  # env beats config
    assert resolve_source("flag/repo", "flagrev", config=config) == ("flag/repo", "flagrev")  # flag beats env
    monkeypatch.delenv("AISIM_DATASET")
    assert resolve_source() == ("ai-dynamo/aisim-data", "main")  # shipped default
    save_config(InitConfig(mode="on-demand"))
    assert resolve_source(config=load_config()) == ("ai-dynamo/aisim-data", "main")


def test_saved_source_applies_to_fetch(run):
    save_config(InitConfig(mode="on-demand", revision=SHA))
    payload = json.loads(run("fetch", "--dry-run")[1])
    assert payload["revision"] == SHA and payload["requested_revision"] == SHA


def test_env_beats_config_for_fetch(run, monkeypatch):
    save_config(InitConfig(mode="on-demand", revision="b" * 40))
    monkeypatch.setenv("AISIM_DATASET", f"{REPO}@{SHA}")
    payload = json.loads(run("fetch", "--dry-run")[1])
    assert payload["repo"] == REPO and payload["requested_revision"] == SHA


# ------------------------------------------------------------------ status and clear


def test_status_empty_cache_is_on_demand(run):
    status, out, _ = run("data", "status")
    payload = json.loads(out)
    assert status == 0 and payload["init"] == {
        "mode": "on-demand",
        "hardware": None,
        "saved": False,
        "config": str(run.cache / "config.json"),
    }
    assert payload["cache"]["path"] == str(run.cache) and payload["revisions"] == []
    assert payload["pin"]["repo"] == "ai-dynamo/aisim-data"


def test_status_reports_cached_and_missing(run):
    run("fetch", "--hardware", "h200_sxm", "--repo", REPO, "--revision", SHA)
    run("init", "--on-demand")
    payload = json.loads(run("data", "status", "--repo", REPO, "--revision", SHA)[1])
    (rev,) = payload["revisions"]
    assert rev["revision"] == SHA and rev["pinned"] is True and rev["size_bytes"] > 0
    assert "data/op-latency/b200_sxm/manifest.json" in rev["missing"]
    assert rev["files_cached"] >= 4 and payload["cache"]["size_bytes"] >= rev["size_bytes"]
    assert payload["init"]["saved"] is True


def test_clear_removes_revisions_and_config(run, monkeypatch):
    run("init", "--all", "--revision", SHA)
    assert any(run.cache.iterdir())
    status, out, _ = run("data", "clear", "--yes")
    payload = json.loads(out)
    assert status == 0 and payload["removed_revisions"] == 1 and payload["config_removed"] is True
    assert load_config() is None and list(run.cache.iterdir()) == []
    assert json.loads(run("data", "status")[1])["revisions"] == []


def test_clear_reports_removal_failure(run, monkeypatch):
    run("init", "--all", "--revision", SHA)
    real = data_cli.shutil.rmtree

    def deny(path, *a, **kw):
        raise PermissionError(13, "denied", str(path))

    monkeypatch.setattr(data_cli.shutil, "rmtree", deny)
    status, out, err = run("data", "clear", "--yes")
    payload = json.loads(out)
    assert status == 1 and payload["removed_revisions"] == 0 and payload["freed_bytes"] == 0
    assert payload["failed"] and "could not remove" in err and load_config() is not None
    monkeypatch.setattr(data_cli.shutil, "rmtree", real)
    assert run("data", "clear", "--yes")[0] == 0


def test_clear_asks_unless_yes(run, monkeypatch):
    run("init", "--all", "--revision", SHA)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    status, _, err = run("data", "clear")
    assert status == 2 and "--yes" in err and load_config() is not None
    _tty(monkeypatch, "n\n")
    assert run("data", "clear")[0] == 1 and load_config() is not None
    _tty(monkeypatch, "y\n")
    assert run("data", "clear")[0] == 0 and load_config() is None


# ------------------------------------------------------------------ validate


def _write_dataset(root, files):
    for path, data in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def test_validate_clean_local_dataset(run, tmp_path):
    root = tmp_path / "ds"
    _write_dataset(root, run.files)
    status, out, _ = run("data", "validate", str(root))
    report = json.loads(out)
    assert status == 0 and report["ok"] is True and report["findings"] == []
    assert report["dataset_catalog_checked"] is True and report["manifests"] == 2
    assert run.transport.opened == []  # local: no network


def test_validate_reports_corruption_and_names(run, tmp_path):
    root = tmp_path / "ds"
    _write_dataset(root, run.files)
    (root / "data/op-latency/h200_sxm/ops.parquet").write_bytes(b"corrupt")
    manifest = root / "data/op-latency/b200_sxm/manifest.json"
    doc = json.loads(manifest.read_text())
    doc["identity"]["hardware"] = "B200"
    manifest.write_text(json.dumps(doc))
    status, out, _ = run("data", "validate", str(root))
    report = json.loads(out)
    checks = {f["check"] for f in report["findings"]}
    assert status == 1 and report["ok"] is False and {"names", "dataset"} <= checks


def test_validate_checks_uncataloged_manifest_when_catalog_present(run, tmp_path):
    root = tmp_path / "ds"
    _write_dataset(root, run.files)
    doc = json.loads((root / "data/op-latency/h200_sxm/manifest.json").read_text())
    doc["artifacts"][0]["sha256"] = "0" * 64
    extra = root / "data/op-latency/extra/manifest.json"
    extra.parent.mkdir(parents=True)
    extra.write_text(json.dumps(doc))
    status, out, _ = run("data", "validate", str(root))
    findings = json.loads(out)["findings"]
    assert status == 1 and any(f["check"] == "manifest" and f["path"].endswith("extra/manifest.json") for f in findings)


def test_validate_directory_reports_malformed_manifest(run, tmp_path):
    root = tmp_path / "ds"
    _write_dataset(root, run.files)
    bad = root / "data/op-latency/broken/manifest.json"
    bad.parent.mkdir(parents=True)
    bad.write_text('{"schema_version": "aisim-artifact')
    status, out, _ = run("data", "validate", str(root))
    report = json.loads(out)
    assert status == 1 and report["ok"] is False
    assert any(f["check"] == "manifest" and f["path"].endswith("broken/manifest.json") for f in report["findings"])


@pytest.mark.parametrize("scope", ["root", "category-directory"])
def test_validate_directory_reports_an_unsupported_envelope_version(run, tmp_path, scope):
    root = tmp_path / "ds"
    _write_dataset(root, run.files)
    doc = json.loads((root / "data/op-latency/h200_sxm/manifest.json").read_text())
    doc["schema_version"] = "aisim-artifact-manifest/v2"
    future = root / "data/op-latency/future/manifest.json"
    future.parent.mkdir(parents=True)
    future.write_text(json.dumps(doc))
    other = root / "data/op-latency/provenance/manifest.json"  # an unrelated format is still ignored
    other.parent.mkdir(parents=True)
    other.write_text(json.dumps({"schema_version": "someone-elses-manifest/v1", "x": 1}))
    status, out, _ = run("data", "validate", str(root if scope == "root" else root / "data/op-latency"))
    report = json.loads(out)
    paths = [f["path"] for f in report["findings"] if f["check"] == "manifest"]
    assert status == 1 and paths == ["data/op-latency/future/manifest.json"]
    assert run("data", "validate", str(future))[0] == 1  # the same manifest checked on its own


@pytest.mark.parametrize("version", ["missing", 7, None, ["aisim-artifact-manifest/v1"]])
def test_validate_directory_reports_an_envelope_without_a_usable_version(run, tmp_path, version):
    doc = json.loads(run.files["data/op-latency/h200_sxm/manifest.json"])
    if version == "missing":
        del doc["schema_version"]
    else:
        doc["schema_version"] = version
    root = tmp_path / "contribution"  # no catalog: discovery alone has to find the manifest
    (root / "h200_sxm").mkdir(parents=True)
    (root / "h200_sxm" / "manifest.json").write_text(json.dumps(doc))
    # Manifest-named files of unrelated formats stay ignored.
    (root / "bundle").mkdir()
    (root / "bundle" / "manifest.json").write_text(json.dumps({"name": "bundle", "files": ["a"], "schema_version": 3}))
    (root / "provenance").mkdir()
    (root / "provenance" / "manifest.json").write_text(json.dumps({"schema_version": "provenance/v2", "x": 1}))
    status, out, _ = run("data", "validate", str(root))
    report = json.loads(out)
    paths = [f["path"] for f in report["findings"] if f["check"] == "manifest"]
    assert status == 1 and paths == ["h200_sxm/manifest.json"]
    assert run("data", "validate", str(root / "h200_sxm" / "manifest.json"))[0] == 1  # the same file on its own


@pytest.mark.parametrize("body", ["[]", "null"])
def test_validate_non_object_manifest_is_a_finding(run, tmp_path, body):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(body)
    status, out, _ = run("data", "validate", str(manifest))
    report = json.loads(out)
    assert status == 1 and any(f["check"] == "manifest" for f in report["findings"])


@pytest.mark.parametrize("artifact_path", [7, None, ["a"]])
def test_validate_non_string_artifact_path_is_a_finding(run, tmp_path, artifact_path):
    doc = json.loads(run.files["data/op-latency/h200_sxm/manifest.json"])
    doc["artifacts"][0]["path"] = artifact_path
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(doc))
    status, out, _ = run("data", "validate", str(manifest))  # not a TypeError while locating the dataset root
    report = json.loads(out)
    assert status == 1 and any(f["check"] == "manifest" for f in report["findings"])


@pytest.mark.parametrize("category", ["eval", "systems"])
@pytest.mark.parametrize("damage", ["altered", "missing"])
@pytest.mark.parametrize("target", ["root", "directory", "manifest"])
def test_validate_verifies_artifacts_of_categories_without_a_checker(run, tmp_path, category, damage, target):
    root = tmp_path / "ds"
    artifact, manifest = build_unchecked(root, category)
    inputs = {"root": root, "directory": root / "data" / category, "manifest": manifest}
    status, out, _ = run("data", "validate", str(inputs[target]))
    report = json.loads(out)
    assert status == 0 and report["ok"] is True and report["categories_without_offline_checker"] == [category]
    if damage == "altered":
        artifact.write_bytes(b'{"record": 2}')
    else:
        artifact.unlink()
    status, out, _ = run("data", "validate", str(inputs[target]))
    report = json.loads(out)
    assert status == 1 and report["ok"] is False
    assert any(f["check"] in ("dataset", "manifest") for f in report["findings"])


def test_validate_catalog_identity_is_compared_for_categories_without_a_checker(run, tmp_path):
    root = tmp_path / "ds"
    build_unchecked(
        root, "eval", catalog_hardware="b200_sxm"
    )  # every digest is right; the catalog says another identity
    status, out, _ = run("data", "validate", str(root))
    report = json.loads(out)
    assert status == 1
    assert any(f["check"] == "dataset" and "identity mismatch" in f["message"] for f in report["findings"])


def test_validate_repo_uses_the_saved_source_like_fetch(run):
    saved = "b" * 40
    run("init", "--on-demand", "--repo", REPO, "--revision", saved)
    status, out, _ = run("data", "validate", "--repo", REPO)  # no --revision: the saved one for this repository
    report = json.loads(out)
    assert status == 0 and report["source"]["revision"] == saved and run.transport.resolves == 0


def test_validate_single_manifest(run, tmp_path):
    root = tmp_path / "ds"
    _write_dataset(root, run.files)
    status, out, _ = run("data", "validate", str(root / "data/op-latency/h200_sxm/manifest.json"))
    report = json.loads(out)
    assert status == 0 and report["manifests"] == 1 and report["dataset_catalog_checked"] is False
    (root / "catalog" / "dataset.json").unlink()  # no catalog: the artifact path still locates the root
    assert run("data", "validate", str(root / "data/op-latency/h200_sxm/manifest.json"))[0] == 0


def test_validate_repo_with_fake_transport(run):
    status, out, _ = run("data", "validate", "--repo", REPO, "--revision", "main")
    report = json.loads(out)
    assert status == 0 and report["ok"] is True
    assert report["source"]["repo"] == REPO and report["source"]["revision"] == SHA
    assert run.transport.resolves == 1


def test_validate_repo_downloads_the_support_files_of_a_category_without_entries(run, tmp_path):
    support = b"schema_version: 1\n"
    index = {
        "schema_version": "aisim-category-index/v1",
        "category": "op-latency",
        "entries": [],
        "support": [
            _artifact_json("s", "data/op-latency/support.yaml", support, format="yaml", schema="aisim.test/v1")
        ],
    }
    index_bytes = json.dumps(index).encode()
    dataset = json.loads(run.files["catalog/dataset.json"])
    dataset["categories"]["op-latency"]["index"] = _artifact_json(
        "i", "catalog/op-latency.json", index_bytes, format="json", schema="aisim-category-index/v1"
    )
    files = {
        "catalog/dataset.json": json.dumps(dataset).encode(),
        "catalog/op-latency.json": index_bytes,
        "data/op-latency/support.yaml": support,
    }
    run.files.clear()
    run.files.update(files)
    root = tmp_path / "ds"
    _write_dataset(root, files)
    local = json.loads(run("data", "validate", str(root))[1])
    assert local["ok"] is True, local["findings"]
    status, out, err = run("data", "validate", "--repo", REPO, "--revision", SHA)
    remote = json.loads(out)
    assert status == 0 and remote["ok"] is True, err  # the same verdict as the local check of the complete dataset
    assert "data/op-latency/support.yaml" in run.transport.opened


def test_validate_repo_corrupt_revision_fails(run):
    run.files["data/op-latency/h200_sxm/ops.parquet"] += b"x"
    status, _, err = run("data", "validate", "--repo", REPO, "--revision", SHA)
    assert status == 1 and "integrity" in err


def test_validate_usage_errors(run, tmp_path):
    assert run("data", "validate")[0] == 2
    assert run("data", "validate", str(tmp_path), "--repo", REPO)[0] == 2
    assert run("data", "validate", str(tmp_path / "missing"))[0] == 2


def test_validate_entry_check_extension_point(run, tmp_path):
    from aisim import data_validate

    root = tmp_path / "ds"
    _write_dataset(root, run.files)
    seen = []

    def check(root_, path, manifest):
        seen.append(path)
        return [{"check": "extra", "path": path, "message": "nope"}]

    data_validate.register_entry_check("op-latency", check)
    try:
        status, out, _ = run("data", "validate", str(root))
    finally:
        data_validate.ENTRY_CHECKS.clear()
    assert status == 1 and len(seen) == 2 and {f["check"] for f in json.loads(out)["findings"]} == {"extra"}


def test_check_names_is_not_a_user_command(run):
    with pytest.raises(SystemExit):
        run("data", "check-names", ".")


@pytest.mark.parametrize(
    ("retired", "expected"),
    [(True, {"revision": "main", "pinned_sha": SHA}), (False, {"revision": SHA, "pinned_sha": None})],
)
def test_fetch_hands_the_sha_only_to_the_repository_that_resolved_it(monkeypatch, tmp_path, retired, expected):
    """A retired repository's alias is followed by plan(); its successor's SHA must not go back under the old name."""
    from types import SimpleNamespace

    from aisim_core.data.repos import PUBLIC_REPO, RETIRED_FPM_REPO

    repo = RETIRED_FPM_REPO if retired else REPO
    seen = {}

    def fake_plan(selection, revision, **kwargs):
        return SimpleNamespace(
            repo=PUBLIC_REPO if retired else kwargs["repo"],
            revision=SHA,
            uncached_bytes=0,
            uncached=[],
            metadata_bytes=0,
            to_dict=lambda: {},
        )

    def fake_fetch(selection, revision, **kwargs):
        seen.update(revision=revision, repo=kwargs["repo"], pinned_sha=kwargs.get("pinned_sha"))
        plan = SimpleNamespace(requested_revision=None, metadata_bytes=0)
        return SimpleNamespace(plan=plan, to_dict=lambda: {})

    monkeypatch.setattr(data_cli, "plan", fake_plan)
    monkeypatch.setattr(data_cli, "fetch", fake_fetch)
    args = SimpleNamespace(cache_dir=str(tmp_path), offline=True, dry_run=False, no_limit=True)
    data_cli._download(Selection(category="fpm", model="m1", hardware="h200"), args, repo, "main")
    # Under the retired name fetch() follows the alias itself, for the commit the plan resolved and budgeted.
    assert seen == {**expected, "repo": repo}


# ------------------------------------------------------------------ Hugging Face model configs


def _cached_models(cache_root):
    return sorted(p.parent.name for p in (cache_root / "model-configs").glob("*/*/refs"))


def test_init_all_caches_the_manifest_models_and_warns_about_the_ones_it_cannot_read(run):
    status, out, err = run("init", "--all", "--revision", SHA)
    result = json.loads(out)["model_configs"]
    assert status == 0 and load_config().mode == "all"  # a gated or rate-limited model never fails init
    assert result["counts"] == {"fetched": 2, "skipped": 1, "failed": 1, "bundled": 1}
    by_model = {e["model"]: e for e in result["models"]}
    assert by_model[ALPHA]["status"] == "fetched" and by_model[ALPHA]["revision"] == MODEL_SHA
    assert by_model[GATED]["status"] == "skipped" and "gated or private" in by_model[GATED]["reason"]
    assert by_model[LIMITED]["status"] == "failed" and by_model[BUNDLED]["status"] == "bundled"
    assert _cached_models(run.cache) == ["Alpha", "Beta"]
    assert f"warning: model config {GATED} skipped" in err and f"warning: model config {LIMITED} failed" in err
    assert "1 model config(s) are not cached" in err and "aisim data fetch --category model-config" in err
    assert not any(call[1] == BUNDLED for call in run.hub.calls)  # AISim ships it: no request


def test_init_hardware_also_caches_the_model_configs(run):
    status, out, _ = run("init", "--hardware", "h200_sxm", "--revision", SHA)
    assert status == 0 and json.loads(out)["model_configs"]["counts"]["fetched"] == 2


def test_init_on_demand_and_no_models_touch_neither_the_hub_nor_the_cache(run):
    for argv in (["init", "--on-demand"], ["init", "--all", "--no-models", "--revision", SHA]):
        status, out, _ = run(*argv)
        assert status == 0 and "model_configs" not in json.loads(out)
    assert run.hub.calls == [] and _cached_models(run.cache) == []


def test_init_all_offline_reads_the_model_cache_and_makes_no_request(run):
    run("init", "--all", "--revision", SHA)
    run.hub.calls.clear()
    status, out, err = run("init", "--all", "--offline", "--revision", SHA)
    counts = json.loads(out)["model_configs"]["counts"]
    assert status == 0 and counts == {"cached": 2, "missing": 2, "bundled": 1}  # nothing was cached for the other two
    assert run.hub.calls == [] and f"model config {GATED} missing" in err


def test_init_without_a_terminal_prints_warnings_on_stderr_and_one_json_object_on_stdout(run):
    status, out, err = run("init", "--all", "--revision", SHA)
    assert status == 0 and json.loads(out)["mode"] == "all" and "warning" in err and "warning" not in out


def test_fetch_model_config_defaults_to_the_manifest_and_fails_only_for_unreachable_models(run):
    status, out, err = run("fetch", "--category", "model-config")
    payload = json.loads(out)
    assert (
        status == 1 and payload["category"] == "model-config" and payload["cache"] == str(run.cache / "model-configs")
    )
    assert payload["counts"] == {"fetched": 2, "skipped": 1, "failed": 1, "bundled": 1}
    assert "fetching 5 model config(s)" in err
    run.hub.denied.pop(LIMITED)  # the rate limit passed
    status, out, _ = run("data", "fetch", "--category", "model-config")
    assert status == 0 and json.loads(out)["counts"] == {"cached": 2, "fetched": 1, "skipped": 1, "bundled": 1}


def test_fetch_model_config_selects_models_and_a_revision(run):
    status, out, _ = run("fetch", "--category", "model-config", "--model", f"{ALPHA}, {BETA}")
    assert status == 0 and [e["model"] for e in json.loads(out)["models"]] == [ALPHA, BETA]
    status, out, _ = run("fetch", "--category", "model-config", "--model", ALPHA, "--revision", "e" * 40)
    entry = json.loads(out)["models"][0]
    assert status == 0 and entry["revision"] == "e" * 40 and ("resolve", ALPHA, "e" * 40) not in run.hub.calls
    # A bundled model with an explicit revision is cached: the FPM profile reads its pinned config that way.
    status, out, _ = run("fetch", "--category", "model-config", "--model", BUNDLED, "--revision", "f" * 40)
    assert json.loads(out)["models"][0]["status"] == "fetched"


def test_fetch_model_config_refresh_resolves_the_branch_again(run):
    run("fetch", "--category", "model-config", "--model", ALPHA)
    run.hub.calls.clear()
    run("fetch", "--category", "model-config", "--model", ALPHA)
    assert run.hub.calls == []  # the pinned commit is reused
    run("fetch", "--category", "model-config", "--model", ALPHA, "--refresh")
    assert ("resolve", ALPHA, "main") in run.hub.calls


def test_fetch_model_config_dry_run_opens_no_connection(run):
    status, out, _ = run("fetch", "--category", "model-config", "--dry-run")
    payload = json.loads(out)
    assert status == 0 and payload["dry_run"] is True and run.hub.calls == []
    assert payload["counts"] == {"to-fetch": 4, "bundled": 1}
    run("fetch", "--category", "model-config", "--model", ALPHA)
    status, out, _ = run("fetch", "--category", "model-config", "--model", ALPHA, "--dry-run")
    assert json.loads(out)["models"][0]["status"] == "cached"


def test_fetch_model_config_offline_reports_what_is_not_cached(run):
    status, out, err = run("fetch", "--category", "model-config", "--model", ALPHA, "--offline")
    entry = json.loads(out)["models"][0]
    assert status == 1 and entry["status"] == "missing" and run.hub.calls == []
    assert "aisim data fetch --category model-config" in err


@pytest.mark.parametrize("command", ["fetch", "init"])
def test_model_prefetch_honors_the_offline_environment_without_constructing_a_transport(run, monkeypatch, command):
    if command == "init":
        # Populate only the operation data, leaving the model cache cold.
        assert run("init", "--all", "--no-models", "--revision", SHA)[0] == 0
    monkeypatch.setenv("AISIM_DATA_OFFLINE", "1")
    monkeypatch.setattr(hf_model_files, "make_transport", lambda: pytest.fail("offline built a model transport"))
    if command == "init":
        status, out, _ = run("init", "--all", "--revision", SHA)
        result = json.loads(out)["model_configs"]
        assert status == 0  # init warns for missing models
    else:
        status, out, _ = run("fetch", "--category", "model-config", "--model", ALPHA)
        result = json.loads(out)
        assert status == 1
    assert result["offline"] and result["counts"]["missing"] > 0 and run.hub.calls == []


def test_fetch_model_config_fails_when_a_required_llama4_processor_is_absent(run, monkeypatch):
    monkeypatch.setattr(
        run.hub,
        "read_file",
        lambda repo, sha, filename: (
            b'{"architectures": ["Llama4ForConditionalGeneration"]}' if filename == "config.json" else None
        ),
    )
    status, out, err = run("fetch", "--category", "model-config", "--model", ALPHA)
    result = json.loads(out)
    assert status == 1 and result["counts"] == {"failed": 1}
    assert result["models"][0]["files"]["preprocessor_config.json"] == "absent"
    assert "Incomplete" in err and "preprocessor_config.json" in err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--hardware", "h200_sxm"], "--hardware does not apply"),
        (["--repo", REPO], "--repo does not apply"),
        (["--all"], "--all and --configuration do not apply"),
        (["--configuration", "x"], "--all and --configuration do not apply"),
        (["--revision", SHA], "exactly one --model"),
        (["--model", "a/b,c/d", "--revision", SHA], "exactly one --model"),
        (["--model", " , "], "--model needs at least one"),
    ],
)
def test_fetch_model_config_rejects_selectors_it_cannot_honor(run, argv, message):
    status, out, err = run("fetch", "--category", "model-config", *argv)
    assert status == 2 and out == "" and message in err and run.hub.calls == []


def test_refresh_is_only_for_model_configs(run):
    status, _, err = run("fetch", "--refresh")
    assert status == 2 and "--refresh only applies to --category model-config" in err and run.transport.opened == []


def test_status_reports_the_model_config_cache_and_clear_removes_it(run):
    run("fetch", "--category", "model-config", "--model", f"{ALPHA},{BETA}")
    models = json.loads(run("data", "status")[1])["model_configs"]
    assert models["path"] == str(run.cache / "model-configs") and models["models"] == 2 and models["size_bytes"] > 0
    status, out, _ = run("data", "clear", "--yes")
    assert status == 0 and json.loads(out)["freed_bytes"] > 0 and not (run.cache / "model-configs").exists()
    assert json.loads(run("data", "status")[1])["model_configs"]["models"] == 0


def test_the_model_cache_follows_cache_dir(run, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    status, out, _ = run("fetch", "--category", "model-config", "--model", ALPHA, "--cache-dir", str(elsewhere))
    assert status == 0 and json.loads(out)["cache"] == str(elsewhere / "model-configs")
    assert _cached_models(elsewhere) == ["Alpha"] and _cached_models(run.cache) == []
