# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One pin for every FPM consumer of the unified dataset (AISIM-2338)."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("set_fpm_dataset_pin", ROOT / "scripts/ci/set_fpm_dataset_pin.py")
pin_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pin_tool)


def test_every_unified_consumer_carries_the_single_pin():
    pin = json.loads(pin_tool.PIN_FILE.read_text())
    assert re.fullmatch(r"[0-9a-f]{40}", pin["revision"])
    consumers = pin_tool.consumers()
    assert len(consumers) >= 15
    for path in consumers:
        document = json.loads(path.read_text())
        assert pin_tool.is_unified(path), path
        assert (document["repo_id"], document["revision"]) == (pin["repo_id"], pin["revision"]), path


def test_only_the_history_profiles_stay_on_the_retired_layout():
    """dsv41_fpm_hf.json pins sidecars that were rewritten after b35883ee: it keeps its old repo and commit."""
    document = json.loads((pin_tool.SYSTEMS / "dsv41_fpm_hf.json").read_text())
    assert document["repo_id"] == "nvidia/aisimulate-fpm-dataset"
    assert document["revision"] == "b35883ee5f8b4a82e24844a7e872056aff64287f"
    assert all(item["path"].startswith("data/history/") for e in document["profiles"].values() for item in e["files"])


def _pin_copy(tmp_path, monkeypatch):
    """A copy of every pin consumer, so the checked-in pins stay untouched."""
    copy = tmp_path / "repo"
    for source in [pin_tool.PIN_FILE, *pin_tool.consumers(), *pin_tool.revision_readmes()]:
        target = copy / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    monkeypatch.setattr(pin_tool, "ROOT", copy)
    monkeypatch.setattr(pin_tool, "PIN_FILE", copy / "scripts/fpm_accuracy/dataset_pin.json")
    monkeypatch.setattr(pin_tool, "SYSTEMS", copy / "python/aisim/src/aisim_core/systems")
    monkeypatch.setattr(pin_tool, "RECIPES", copy / "python/aisim/collector/fpm_forward/recipes")


@pytest.mark.parametrize("revision", ["c" * 40])
def test_set_pin_rewrites_every_consumer_and_check_agrees(tmp_path, monkeypatch, revision):
    _pin_copy(tmp_path, monkeypatch)
    readmes = pin_tool.revision_readmes()
    assert readmes  # the Minimax README links its profile's pinned revision
    previous = {manifest: json.loads(manifest.read_text())["revision"] for manifest in readmes.values()}
    assert all(f"/tree/{previous[manifest]}/" in readme.read_text() for readme, manifest in readmes.items())
    monkeypatch.setattr("sys.argv", ["set_fpm_dataset_pin.py", "--revision", revision])
    assert pin_tool.main() == 0
    monkeypatch.setattr("sys.argv", ["set_fpm_dataset_pin.py", "--check", "--require-set"])
    assert pin_tool.main() == 0
    assert {json.loads(p.read_text())["revision"] for p in pin_tool.consumers()} == {revision}
    for readme, manifest in readmes.items():  # the README follows the manifest it describes
        text = readme.read_text()
        assert json.loads(manifest.read_text())["revision"] == revision
        assert f"/tree/{revision}/" in text and f"/tree/{previous[manifest]}/" not in text


@pytest.mark.parametrize("separators", [(",", ":"), (", ", " : ")])
def test_update_reaches_consumers_with_any_json_whitespace(tmp_path, monkeypatch, separators):
    _pin_copy(tmp_path, monkeypatch)
    consumers = pin_tool.consumers()
    recipe = next(path for path in consumers if path.parent == pin_tool.RECIPES)
    profile = next(path for path in consumers if path.name == "hf_dataset.json")
    for path in (recipe, profile):
        path.write_text(json.dumps(json.loads(path.read_text()), separators=separators))
    monkeypatch.setattr("sys.argv", ["set_fpm_dataset_pin.py", "--revision", "c" * 40])
    assert pin_tool.main() == 0
    assert {json.loads(path.read_text())["revision"] for path in consumers} == {"c" * 40}
    monkeypatch.setattr("sys.argv", ["set_fpm_dataset_pin.py", "--check", "--require-set"])
    assert pin_tool.main() == 0


def test_an_update_that_cannot_reach_a_consumer_fails_before_any_write(tmp_path, monkeypatch, capsys):
    _pin_copy(tmp_path, monkeypatch)
    recipe = next(path for path in pin_tool.consumers() if path.parent == pin_tool.RECIPES)
    document = json.loads(recipe.read_text())
    document["revision"] = 7  # not a string, so the in-place rewrite cannot reach it
    recipe.write_text(json.dumps(document))
    files = [pin_tool.PIN_FILE, *pin_tool.consumers(), *pin_tool.revision_readmes()]
    before = {path: path.read_bytes() for path in files}
    monkeypatch.setattr("sys.argv", ["set_fpm_dataset_pin.py", "--revision", "c" * 40])
    with pytest.raises(SystemExit) as raised:
        pin_tool.main()
    assert raised.value.code != 0
    assert "cannot update the pin fields" in capsys.readouterr().err
    assert before == {path: path.read_bytes() for path in files}


def test_check_rejects_a_readme_that_lags_the_pin(tmp_path, monkeypatch, capsys):
    _pin_copy(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.argv", ["set_fpm_dataset_pin.py", "--check", "--require-set"])
    assert pin_tool.main() == 0
    readme = next(iter(pin_tool.revision_readmes()))
    revision = json.loads(pin_tool.PIN_FILE.read_text())["revision"]
    readme.write_text(readme.read_text().replace(f"/tree/{revision}/", f"/tree/{'d' * 40}/"))
    assert pin_tool.main() == 1
    assert "does not link the pinned revision" in capsys.readouterr().err


def test_an_unlinked_readme_stops_an_update_before_any_write(tmp_path, monkeypatch):
    _pin_copy(tmp_path, monkeypatch)
    readme = next(iter(pin_tool.revision_readmes()))
    readme.write_text("no revision link here\n")
    files = [pin_tool.PIN_FILE, *pin_tool.consumers(), readme]
    before = {path: path.read_bytes() for path in files}
    monkeypatch.setattr("sys.argv", ["set_fpm_dataset_pin.py", "--revision", "c" * 40])
    with pytest.raises(SystemExit) as raised:
        pin_tool.main()
    assert raised.value.code != 0
    assert before == {path: path.read_bytes() for path in files}
