# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from unittest.mock import patch

import pytest
from spdl.source.utils import MergeIterator, repeat_source


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


def test_repeat_source_iterable_with_shuffle():
    """repeat_source repeats souce while calling shuffle"""

    class _IteWithShuffle:
        def __init__(self) -> None:
            self.vals = list(range(3))

        def shuffle(self, seed: int) -> None:
            assert isinstance(seed, int)
            self.vals = self.vals[1:] + self.vals[:1]

        def __iter__(self) -> Iterator[int]:
            yield from self.vals

    src = _IteWithShuffle()
    gen = iter(repeat_source(src, epoch=2))

    with patch.object(src, "shuffle", side_effect=src.shuffle) as mock_method:
        assert next(gen) == 1
        mock_method.assert_called_with(seed=2)
        assert next(gen) == 2
        assert next(gen) == 0

        assert next(gen) == 2
        mock_method.assert_called_with(seed=3)
        assert next(gen) == 0
        assert next(gen) == 1

        assert next(gen) == 0
        mock_method.assert_called_with(seed=4)
        assert next(gen) == 1
        assert next(gen) == 2

        assert next(gen) == 1
        mock_method.assert_called_with(seed=5)
        assert next(gen) == 2
        assert next(gen) == 0


def test_repeat_source_iterable():
    """repeat_source works Iterable without shuffle method"""

    class _IteWithoutShuffle:
        def __init__(self) -> None:
            self.vals = list(range(3))

        def __iter__(self) -> Iterator[int]:
            yield from self.vals

    src = _IteWithoutShuffle()
    gen = iter(repeat_source(src, epoch=2))

    for _ in range(100):
        assert next(gen) == 0
        assert next(gen) == 1
        assert next(gen) == 2
