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


@pytest.fixture
def env(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    for name in ("AISIM_DATASET", "AISIM_DATA_MAX_BYTES", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AISIM_DATA_CACHE", str(cache))
    monkeypatch.setattr("aisim_core.data.config.PIN_FILE", tmp_path / "absent-pin.json")
    return cache


@pytest.fixture
def run(monkeypatch, tmp_path, capsys, env):
    files = build_ops(tmp_path)
    transport = FakeTransport(files)
    monkeypatch.setattr(data_cli, "make_transport", lambda: transport)

    def _run(*argv):
        status = main(list(argv))
        out = capsys.readouterr()
        return status, out.out, out.err

    _run.transport, _run.files, _run.cache = transport, files, env
    return _run


def _tty(monkeypatch, answer=""):
    class Stdin:
        def isatty(self):
            return True

        def readline(self):
            return answer

    monkeypatch.setattr(sys, "stdin", Stdin())


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
    status, _, _ = run("init", "--all")  # no flag, no saved source: the shipped default is downloaded
    assert status == 0
    assert load_config() == InitConfig(mode="all", repo=shipped_pin()[0], revision=shipped_pin()[1])
    pin = pin_from_environment({"AISIM_DATA_CACHE": str(run.cache)})
    assert (pin.repo, pin.revision) == shipped_pin()  # the downloaded data is what predictors select


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


@pytest.mark.parametrize("body", ["[]", "null"])
def test_validate_non_object_manifest_is_a_finding(run, tmp_path, body):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(body)
    status, out, _ = run("data", "validate", str(manifest))
    report = json.loads(out)
    assert status == 1 and any(f["check"] == "manifest" for f in report["findings"])


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
