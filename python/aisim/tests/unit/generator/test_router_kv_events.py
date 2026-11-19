# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preserve explicit Router event observation choices in frontend arguments."""

import pytest

from aisim.generator.dynamo_features import frontend_cli_args_from_dyn_config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "router_config,expected",
    [
        ({}, []),
        ({"use_kv_events": False}, ["--no-router-kv-events"]),
        ({"use_kv_events": True}, ["--router-kv-events"]),
    ],
)
def test_frontend_router_kv_events_choice(router_config, expected):
    args = frontend_cli_args_from_dyn_config({"enable_router": True, "router_config": router_config})
    assert args == ["--router-mode", "kv", *expected]
