# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic checkpoint transport for model consumer tests."""

import io
import json
import urllib.error
from email.message import Message

import pytest

from aisim_core.sdk import utils


@pytest.fixture
def hf_checkpoint_response(monkeypatch, tmp_path):
    """Serve one synthetic immutable Hub snapshot through the real JSON loader."""
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hub-cache"))
    monkeypatch.setattr(utils, "_get_hf_auth_headers", lambda: {})

    def serve(files):
        def fetch(request, *, timeout):
            filename = request.full_url.rsplit("/", 1)[-1]
            headers = Message()
            headers["X-Repo-Commit"] = "c" * 40
            if filename not in files:
                headers["X-Error-Code"] = "EntryNotFound"
                raise urllib.error.HTTPError(request.full_url, 404, "missing synthetic metadata", headers, None)
            response = io.BytesIO(json.dumps(files[filename]).encode())
            response.headers = headers
            return response

        monkeypatch.setattr(utils.urllib.request, "urlopen", fetch)

    return serve
