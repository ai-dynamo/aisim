# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject ambient model overrides before provenance collection or timing."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("override", ["", "/unverified/model-configs"])
def test_canonical_driver_rejects_model_config_override(override, tmp_path):
    driver = Path(__file__).with_name("canonical_mtp_perf.py")
    proc = subprocess.run(
        [
            sys.executable,
            str(driver),
            sys.executable,
            "missing-sdk",
            "2",
            str(tmp_path / "out.json"),
            "missing-repo",
            "base",
            "head",
        ],
        env=os.environ | {"AICONFIGURATOR_MODEL_CONFIGS_PATH": override},
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "AICONFIGURATOR_MODEL_CONFIGS_PATH must be unset" in proc.stderr
    assert not (tmp_path / "out.json").exists()
