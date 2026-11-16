# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
from urllib.error import HTTPError, URLError
from zipfile import ZipFile

import pytest

from scripts.readme import prepare_dynamo_rc as rc


def test_simple_html_and_json_identify_exact_versions(monkeypatch):
    responses = iter(
        (
            (
                b'<html><body><a href="../ai_dynamo-1.7.0-py3-none-any.whl#sha256=123">wheel</a></body></html>',
                "text/html",
            ),
            (
                json.dumps({"files": [{"filename": "ai_dynamo-1.7.0.dev20261116-py3-none-any.whl"}]}).encode(),
                "application/vnd.pypi.simple.v1+json",
            ),
        )
    )

    def open_index(*args, **kwargs):
        payload, content_type = next(responses)
        response = io.BytesIO(payload)
        response.headers = {"Content-Type": content_type}
        return response

    monkeypatch.setattr(rc, "urlopen", open_index)
    assert rc.available("https://example.test", "ai-dynamo", "1.7.0")
    assert not rc.available("https://example.test", "ai-dynamo", "1.7.0")


def test_probe_source_build_only_when_an_exact_package_is_absent(monkeypatch):
    monkeypatch.setattr(rc, "validate_versions", lambda root: None)
    monkeypatch.setattr(rc, "available", lambda index, name, version: name != "aisim")
    manifest = rc.probe()
    assert manifest["source"] == "release-candidate"
    assert manifest["missing"] == ["aisim"]
    monkeypatch.setattr(rc, "available", lambda *args: True)
    assert rc.probe()["source"] == "public-index"


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_http_error_does_not_select_rc(monkeypatch, status):
    def fail(*args, **kwargs):
        raise HTTPError("https://example.test", status, "unavailable", {}, None)

    monkeypatch.setattr(rc, "urlopen", fail)
    with pytest.raises(HTTPError):
        rc.available("https://example.test", "aisim", "0.14.0")


def test_missing_project_404_is_absence_but_network_errors_fail(monkeypatch):
    def missing(*args, **kwargs):
        raise HTTPError("https://example.test", 404, "not found", {}, None)

    monkeypatch.setattr(rc, "urlopen", missing)
    assert not rc.available("https://example.test", "aisim", "0.14.0")

    def offline(*args, **kwargs):
        raise URLError("DNS lookup failed")

    monkeypatch.setattr(rc, "urlopen", offline)
    with pytest.raises(URLError):
        rc.available("https://example.test", "aisim", "0.14.0")


@pytest.mark.parametrize(
    ("payload", "content_type"),
    [
        (b"not an index", "text/html"),
        (b'{"error":"unavailable"}', "application/json"),
        (b'{"files":[{}]}', "application/json"),
        (b'{"files":"bad"}', "application/json"),
        (b"unavailable", "text/plain"),
    ],
)
def test_malformed_index_is_not_package_absence(payload, content_type):
    with pytest.raises(ValueError):
        rc.index_filenames(payload, content_type)


@pytest.mark.parametrize("changed", ["readme", "project"])
def test_pins_must_match_readme_and_aisim_source(tmp_path, changed):
    project = tmp_path / "python/aisim/pyproject.toml"
    project.parent.mkdir(parents=True)
    version = "0.15.0" if changed == "project" else "0.14.0"
    project.write_text(f'[project]\nname="aisim"\nversion="{version}"\n')
    dynamo = "1.8.0" if changed == "readme" else "1.7.0"
    (tmp_path / "README.md").write_text(
        f'<!-- readme-check: dynamo-install -->\n```bash\npip install "aisim==0.14.0" "ai-dynamo=={dynamo}"\n```\n'
    )
    with pytest.raises(ValueError, match="version"):
        rc.validate_versions(tmp_path)


def wheel(tmp_path, filename="aisim-0.14.0-py3-none-any.whl", metadata="Name: aisim\nVersion: 0.14.0\n"):
    path = tmp_path / filename
    with ZipFile(path, "w") as archive:
        archive.writestr("aisim-0.14.0.dist-info/METADATA", metadata)
    return path


def test_wheel_manifest_has_identity_and_content_hash(tmp_path):
    entry = rc.inspect_wheel(wheel(tmp_path))
    assert (entry["name"], entry["version"]) == ("aisim", "0.14.0")
    assert len(entry["sha256"]) == 64


@pytest.mark.parametrize(
    ("filename", "metadata"),
    [
        ("aisim-0.15.0-py3-none-any.whl", "Name: aisim\nVersion: 0.15.0\n"),
        ("aisim-0.14.0-py3-none-any.whl", "Name: aisimulate\nVersion: 0.14.0\n"),
        ("aisim-0.14.0-py3-none-any.whl", "Name: aisim\nVersion: 0.13.0\n"),
    ],
)
def test_mismatched_wheel_name_or_metadata_is_rejected(tmp_path, filename, metadata):
    with pytest.raises(ValueError, match="wheel|Wheel"):
        rc.inspect_wheel(wheel(tmp_path, filename, metadata))


def test_public_build_does_not_build_or_create_wheelhouse(monkeypatch, tmp_path):
    manifest = {"source": "public-index", "wheels": []}
    monkeypatch.setattr(rc, "probe", lambda root: manifest)
    monkeypatch.setattr(rc, "run", lambda *args: pytest.fail("public artifacts must not be rebuilt"))
    assert rc.build(tmp_path / "wheels") == manifest
    assert not (tmp_path / "wheels").exists()
