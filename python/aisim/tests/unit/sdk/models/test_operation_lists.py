# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural graph edits must invalidate identities without rescanning ops."""

import copy
import json
import operator
import pickle

import pytest

from aisim_core.sdk.models.base import _OperationList

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda ops: ops.append(3), [2, 1, 3]),
        (lambda ops: ops.extend([3, 4]), [2, 1, 3, 4]),
        (lambda ops: ops.insert(1, 3), [2, 3, 1]),
        (lambda ops: ops.pop(), [2]),
        (lambda ops: ops.remove(2), [1]),
        (lambda ops: ops.clear(), []),
        (lambda ops: ops.reverse(), [1, 2]),
        (lambda ops: ops.sort(), [1, 2]),
        (lambda ops: operator.setitem(ops, 0, 3), [3, 1]),
        (lambda ops: operator.setitem(ops, slice(None), [3]), [3]),
        (lambda ops: operator.delitem(ops, 0), [1]),
        (lambda ops: operator.delitem(ops, slice(None)), []),
        (lambda ops: operator.iadd(ops, [3]), [2, 1, 3]),
        (lambda ops: operator.imul(ops, 2), [2, 1, 2, 1]),
    ],
)
def test_mutations_advance_the_shared_revision(mutate, expected):
    revision = [0]
    ops = _OperationList([2, 1], revision)
    mutate(ops)
    assert ops == expected
    assert revision[0] > 0


def test_partial_extend_invalidates_even_when_iterator_raises():
    def values():
        yield 3
        raise ValueError("incomplete edit")

    revision = [0]
    ops = _OperationList([2, 1], revision)
    with pytest.raises(ValueError, match="incomplete edit"):
        ops.extend(values())
    assert ops == [2, 1, 3]
    assert revision[0] > 0


@pytest.mark.parametrize("clone", [copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))])
def test_restored_phase_lists_share_only_their_own_revision(clone):
    revision = [0]
    graph = {
        "revision": revision,
        "context": _OperationList([1], revision),
        "generation": _OperationList([2], revision),
    }
    restored = clone(graph)
    version = restored["revision"][0]
    restored["context"].append(3)
    assert restored["revision"][0] == version + 1
    version = restored["revision"][0]
    restored["generation"].clear()
    assert restored["revision"][0] == version + 2
    assert revision == [0]
    assert graph["context"] == [1]
    assert graph["generation"] == [2]


@pytest.fixture
def graph_estimator():
    from aisim_core.sdk import config, rust_engine_step
    from aisim_core.sdk.models import get_model
    from aisim_core.sdk.perf_database import get_database_view

    model = get_model(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1, nextn=1),
        "trtllm",
    )
    # Retain one qualified scan per phase. Its serialized graph participates
    # in the production identity, and every added scan contributes latency.
    for phase in ("context", "generation"):
        name = f"{phase}_ops"
        setattr(model, name, [op for op in getattr(model, name) if op._name == f"{phase}_mamba_ssm"])
        assert len(getattr(model, name)) == 1
    database = get_database_view("b200_sxm", "trtllm", "1.3.0rc20", database_mode="SOL", allow_unlisted_version=True)

    def estimate(graph):
        return rust_engine_step.estimate_mixed_step_latency_with_rust(
            graph, database, ctx_tokens=16, gen_tokens=1, isl=16, osl=2, prefix=0
        )

    rust_engine_step._engine_handle_cache_clear()
    yield model, estimate
    rust_engine_step._engine_handle_cache_clear()


def _extra_scan(model, phase, name):
    import aisim_core

    spec = json.loads(getattr(model, f"{phase}_ops")[0]._spec_json())
    spec["Mamba2"]["name"] = name
    return aisim_core.op_from_spec_json(json.dumps(spec))


@pytest.mark.parametrize("phase", ["context", "generation"])
@pytest.mark.parametrize("method", ["extend", "__iadd__"])
@pytest.mark.parametrize("fail", [False, True])
def test_estimate_during_iterable_edit_is_invalidated_on_exit(graph_estimator, phase, method, fail):
    model, estimate = graph_estimator
    cold = copy.deepcopy(model)
    first = _extra_scan(model, phase, "extra_first_scan")
    second = _extra_scan(model, phase, "extra_second_scan")
    getattr(cold, f"{phase}_ops").extend([first, second])
    intermediate = []
    initial = estimate(model)

    def values():
        yield first
        intermediate.append(estimate(model))
        yield second
        if fail:
            raise ValueError("incomplete edit")

    edit = getattr(getattr(model, f"{phase}_ops"), method)
    if fail:
        with pytest.raises(ValueError, match="incomplete edit"):
            edit(values())
    else:
        edit(values())
    # Both yielded scans remain after a failed extend. An estimate taken
    # between yields must not hide the second scan from subsequent queries.
    actual = estimate(model)
    assert initial < intermediate[0] < actual
    assert actual == estimate(cold)


@pytest.mark.parametrize("phase", ["context", "generation"])
@pytest.mark.parametrize("fail", [False, True])
def test_estimate_during_sort_is_invalidated_on_exit(graph_estimator, phase, fail):
    model, estimate = graph_estimator
    ops = getattr(model, f"{phase}_ops")
    ops.append(_extra_scan(model, phase, "extra_scan"))
    cold = copy.deepcopy(model)
    intermediate = []
    estimate(model)

    def key(op):
        if not intermediate:
            # CPython temporarily empties the list while sorting it.
            intermediate.append(estimate(model))
        if fail:
            raise ValueError("incomplete sort")
        return op._name

    if fail:
        with pytest.raises(ValueError, match="incomplete sort"):
            ops.sort(key=key)
    else:
        ops.sort(key=key)
        getattr(cold, f"{phase}_ops").sort(key=lambda op: op._name)
    assert estimate(model) > intermediate[0]
    assert estimate(model) == estimate(cold)


@pytest.mark.parametrize("phase", ["context", "generation"])
@pytest.mark.parametrize("fail", [False, True])
def test_replacement_iterable_cannot_leave_an_old_graph_identity(graph_estimator, phase, fail):
    model, estimate = graph_estimator
    name = f"{phase}_ops"
    original = getattr(model, name)
    replacement = [*original, _extra_scan(model, phase, "extra_scan")]
    cold = copy.deepcopy(model)
    if not fail:
        setattr(cold, name, replacement)
    initial = estimate(model)

    def values():
        yield from replacement
        assert estimate(model) == initial
        if fail:
            raise ValueError("incomplete replacement")

    if fail:
        with pytest.raises(ValueError, match="incomplete replacement"):
            setattr(model, name, values())
        assert getattr(model, name) is original
        assert estimate(model) == initial
    else:
        setattr(model, name, values())
        assert estimate(model) > initial
    assert estimate(model) == estimate(cold)
