# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from spdl.dataloader import MergeIterator


def test_mergeiterator_ordered():
    """MergeIterator iterates multiple iterators"""

    iterables = [
        [0, 1, 2],
        [10, 11, 12],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables))
    assert result == [0, 10, 20, 1, 11, 21, 2, 12, 22]


def test_mergeiterator_ordered_stop_after_first_exhaustion():
    """MergeIterator stops after the first exhaustion"""

    iterables = [
        [0],
        [10, 11, 12],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables, stop_after=-1))
    assert result == [0, 10, 20]

    iterables = [
        [0, 1, 2],
        [10],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables, stop_after=-1))
    assert result == [0, 10, 20, 1]

    iterables = [
        [0, 1, 2],
        [10, 11],
        [20],
    ]

    result = list(MergeIterator(iterables, stop_after=-1))
    assert result == [0, 10, 20, 1, 11]


def test_mergeiterator_ordered_stop_after_N():
    """MergeIterator stops after N items are yielded"""

    iterables = [
        [0, 1, 2],
        [10, 11, 12],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables, stop_after=1))
    assert result == [0]

    result = list(MergeIterator(iterables, stop_after=5))
    assert result == [0, 10, 20, 1, 11]

    result = list(MergeIterator(iterables, stop_after=7))
    assert result == [0, 10, 20, 1, 11, 21, 2]


def test_mergeiterator_ordered_stop_after_minus1():
    """MergeIterator stops after all the iterables are exhausted"""

    iterables = [
        [0, 1, 2],
        [10, 11, 12],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables))
    assert result == [0, 10, 20, 1, 11, 21, 2, 12, 22]

    iterables = [
        [0, 1, 2],
        [10],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables))
    assert result == [0, 10, 20, 1, 21, 2, 22]

    iterables = [
        [0, 1, 2],
        [10, 11, 12],
        [20],
    ]

    result = list(MergeIterator(iterables))
    assert result == [0, 10, 20, 1, 11, 2, 12]


def test_mergeiterator_ordered_n():
    """with stop_after=N, MergeIterator continues iterating after encountering an exhaustion."""
    iterables = [
        [0, 1, 2],
        [10],
        [20, 21, 22],
    ]

    result = list(MergeIterator(iterables, stop_after=5))
    assert result == [0, 10, 20, 1, 21]

    result = list(MergeIterator(iterables, stop_after=7))
    assert result == [0, 10, 20, 1, 21, 2, 22]

    result = list(MergeIterator(iterables, stop_after=8))
    assert result == [0, 10, 20, 1, 21, 2, 22]


def test_mergeiterator_stochastic_smoke_test():
    """MergeIterator with probabilitiies do not get stuck."""

    iterables = [
        [0, 1, 2],
        [10, 11, 12],
        [20, 21, 22],
    ]

    weights = [1, 1, 1]

    result = list(MergeIterator(iterables, weights=weights))
    assert set(result) == {0, 1, 2, 10, 11, 12, 20, 21, 22}


def test_mergeiterator_stochastic_rejects_zero():
    """weight=0 is rejected."""
    weights = [1, 0]

    with pytest.raises(ValueError):
        MergeIterator([[1]], weights=weights)

    weights = [1, 0.0]

    with pytest.raises(ValueError):
        MergeIterator([[1]], weights=weights)


def test_mergeiterator_stochastic_stop_after_N():
    """Values are taken from iterables with higher weights"""
    weights = [1000000, 1]

    iterables = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    ]

    result = list(MergeIterator(iterables, weights=weights, stop_after=3))
    assert result == [0, 1, 2]


def test_mergeiterator_stochastic_stop_after_first_exhaustion():
    """Values are taken from iterables with higher weights"""
    weights = [1000000, 1]

    iterables = [
        [0, 1, 2, 3],
        [10, 11, 12, 13],
    ]

    result = list(MergeIterator(iterables, weights=weights, stop_after=-1))
    assert result == [0, 1, 2, 3]
