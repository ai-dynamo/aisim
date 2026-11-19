# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static checks of the mirror CI template: same validator, no secrets, never runs here."""

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

TEMPLATE = Path(__file__).resolve().parents[5] / ".github/mirror/aisim-data-internal-validate.yml"


class _StrictLoader(yaml.SafeLoader):
    """Reject duplicate mapping keys, which safe_load would silently resolve to the last one."""

    def construct_mapping(self, node, deep=False):
        keys = [self.construct_object(k, deep=deep) for k, _ in node.value]
        dupes = {k for k in keys if keys.count(k) > 1}
        if dupes:
            raise yaml.constructor.ConstructorError(None, None, f"duplicate keys {sorted(dupes)}", node.start_mark)
        return super().construct_mapping(node, deep)


def _load(text):
    return yaml.load(text, Loader=_StrictLoader)


@pytest.fixture(scope="module")
def text():
    if not TEMPLATE.is_file():
        if (TEMPLATE.parents[2] / "pyproject.toml").exists() or (TEMPLATE.parents[2] / ".git").exists():
            pytest.fail(f"required mirror workflow template is missing: {TEMPLATE}")
        pytest.skip("distribution omits repository workflow templates")
    return TEMPLATE.read_text()


def test_template_is_not_an_active_workflow_here(text):
    assert TEMPLATE.parent.name == "mirror"
    assert "github.repository == 'ai-dynamo/aisim-data-internal'" in text


def test_runs_the_shared_validator_from_a_pinned_checkout(text):
    doc = _load(text)
    run = "\n".join(s.get("run", "") for s in doc["jobs"]["validate"]["steps"])
    assert "aisim_core.data.mirror validate" in run and "--profile mirror" in run
    assert re.fullmatch(r"[0-9a-f]{40}", doc["env"]["AISIM_VALIDATOR_REF"])
    assert 'AISIM_VALIDATOR_REF}" != 0000*' in run  # the placeholder SHA fails closed


def test_least_privilege_and_pinned_actions(text):
    doc = _load(text)
    assert doc["permissions"] == {"contents": "read"}
    assert "secrets." not in text and "pull_request_target" not in text
    checkouts = [st for st in doc["jobs"]["validate"]["steps"] if st.get("uses", "").startswith("actions/checkout@")]
    assert checkouts and all(st["with"]["persist-credentials"] is False for st in checkouts)
    assert all(re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", u) for u in re.findall(r"uses:\s*(\S+)", text))


def test_missing_template_fails_in_a_source_checkout(tmp_path, monkeypatch):
    root = tmp_path / "source"
    (root / ".git").mkdir(parents=True)
    monkeypatch.setitem(globals(), "TEMPLATE", root / ".github/mirror/missing.yml")
    with pytest.raises(pytest.fail.Exception, match="required mirror workflow"):
        text.__wrapped__()
