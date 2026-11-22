# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural graph edits must invalidate identities without rescanning ops."""

import copy
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
    restored["generation"].clear()
    assert restored["revision"][0] == version + 2
    assert revision == [0]
    assert graph["context"] == [1]
    assert graph["generation"] == [2]
