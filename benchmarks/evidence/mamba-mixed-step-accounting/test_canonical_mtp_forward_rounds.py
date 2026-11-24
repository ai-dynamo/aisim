# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check interpreter isolation and SDK selection without performance data."""

import json

import pytest
from canonical_mtp_forward_rounds import collect_rounds


def test_rounds_use_fresh_interpreters_and_selected_sdk(tmp_path, monkeypatch):
    sdks = [tmp_path / label for label in ("baseline", "candidate")]
    for sdk in sdks:
        sdk.mkdir()
        (sdk / "fixture_sdk.py").write_text(f"LABEL = {sdk.name!r}\n")
    driver = tmp_path / "worker.py"
    driver.write_text(
        "import json, os, sys, fixture_sdk\n"
        "from pathlib import Path\n"
        "counter = globals().get('counter', 0) + 1\n"
        "Path(sys.argv[1]).write_text(json.dumps({\n"
        " 'sdk': fixture_sdk.LABEL, 'pid': os.getpid(), 'counter': counter,\n"
        " 'identity': {'constructor_path': fixture_sdk.__file__, 'native_path': fixture_sdk.__file__}}))\n"
    )
    # A stale ambient SDK selection must not leak into either child.
    monkeypatch.setenv("PYTHONPATH", "/unused/installed/sdk")
    result = collect_rounds(*sdks, driver=driver)
    assert result["env_overrides"] == {"AIC_ALLOW_UNLISTED_VERSIONS": "1"}
    rounds = result["rounds"]
    assert [r["order"] for r in rounds] == [
        ["baseline", "candidate"],
        ["candidate", "baseline"],
    ] * 3
    records = [r["results"][label] for r in rounds for label in r["order"]]
    assert len({r["pid"] for r in records}) == 12
    assert all(r["counter"] == 1 for r in records)
    for record in rounds:
        for label, value in record["results"].items():
            assert value["sdk"] == label
    # The persisted envelope retains each child's unmodified output.
    assert json.loads(json.dumps(result)) == result


def test_rejects_results_loaded_outside_selected_sdk(tmp_path):
    driver = tmp_path / "wrong_sdk.py"
    driver.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "Path(sys.argv[1]).write_text(json.dumps({'identity': {\n"
        f" 'constructor_path': {str(tmp_path / 'installed.py')!r}, 'native_path': '/wrong/native.so'"
        "}}))\n"
    )
    with pytest.raises(ValueError, match="imported outside selected SDK"):
        collect_rounds(tmp_path / "baseline", tmp_path / "candidate", driver=driver)
