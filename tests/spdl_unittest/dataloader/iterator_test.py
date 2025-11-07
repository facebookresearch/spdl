# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import pickle
import random
import unittest
from collections.abc import Iterator
from functools import partial
from unittest.mock import patch

from spdl.pipeline import iterate_in_subprocess as _iterate_in_subprocess
from spdl.source.utils import (
    embed_shuffle,
    IterableWithShuffle,
    MergeIterator,
    repeat_source,
)


def iterate_in_subprocess(fn, *, timeout=10, **kwargs):
    return _iterate_in_subprocess(fn, timeout=timeout, **kwargs)


class TestMergeIterator(unittest.TestCase):
    def test_mergeiterator_ordered(self) -> None:
        """MergeIterator iterates multiple iterators"""

        iterables = [
            [0, 1, 2],
            [10, 11, 12],
            [20, 21, 22],
        ]

        result = list(MergeIterator(iterables))
        self.assertEqual(result, [0, 10, 20, 1, 11, 21, 2, 12, 22])

    def test_mergeiterator_ordered_stop_after_first_exhaustion(self) -> None:
        """MergeIterator stops after the first exhaustion"""

        iterables = [
            [0],
            [10, 11, 12],
            [20, 21, 22],
        ]

        result = list(MergeIterator(iterables, stop_after=-1))
        self.assertEqual(result, [0, 10, 20])

        iterables = [
            [0, 1, 2],
            [10],
            [20, 21, 22],
        ]

        result = list(MergeIterator(iterables, stop_after=-1))
        self.assertEqual(result, [0, 10, 20, 1])

        iterables = [
            [0, 1, 2],
            [10, 11],
            [20],
        ]

        result = list(MergeIterator(iterables, stop_after=-1))
        self.assertEqual(result, [0, 10, 20, 1, 11])

    def test_mergeiterator_ordered_stop_after_N(self) -> None:
        """MergeIterator stops after N items are yielded"""

        iterables = [
            [0, 1, 2],
            [10, 11, 12],
            [20, 21, 22],
        ]

        result = list(MergeIterator(iterables, stop_after=1))
        self.assertEqual(result, [0])

        result = list(MergeIterator(iterables, stop_after=5))
        self.assertEqual(result, [0, 10, 20, 1, 11])

        result = list(MergeIterator(iterables, stop_after=7))
        self.assertEqual(result, [0, 10, 20, 1, 11, 21, 2])

    def test_mergeiterator_ordered_stop_after_minus1(self) -> None:
        """MergeIterator stops after all the iterables are exhausted"""

        iterables = [
            [0, 1, 2],
            [10, 11, 12],
            [20, 21, 22],
        ]

        result = list(MergeIterator(iterables))
        self.assertEqual(result, [0, 10, 20, 1, 11, 21, 2, 12, 22])

        iterables = [
            [0, 1, 2],
            [10],
            [20, 21, 22],
        ]

        result = list(MergeIterator(iterables))
        self.assertEqual(result, [0, 10, 20, 1, 21, 2, 22])

        iterables = [
            [0, 1, 2],
            [10, 11, 12],
            [20],
        ]

        result = list(MergeIterator(iterables))
        self.assertEqual(result, [0, 10, 20, 1, 11, 2, 12])

    def test_mergeiterator_ordered_n(self) -> None:
        """with stop_after=N, MergeIterator continues iterating after exhaustion."""
        iterables = [
            [0, 1, 2],
            [10],
            [20, 21, 22],
        ]

        result = list(MergeIterator(iterables, stop_after=5))
        self.assertEqual(result, [0, 10, 20, 1, 21])

        result = list(MergeIterator(iterables, stop_after=7))
        self.assertEqual(result, [0, 10, 20, 1, 21, 2, 22])

        result = list(MergeIterator(iterables, stop_after=8))
        self.assertEqual(result, [0, 10, 20, 1, 21, 2, 22])

    def test_mergeiterator_stochastic_smoke_test(self) -> None:
        """MergeIterator with probabilitiies do not get stuck."""

        iterables = [
            [0, 1, 2],
            [10, 11, 12],
            [20, 21, 22],
        ]

        weights = [1, 1, 1]

        result = list(MergeIterator(iterables, weights=weights))
        self.assertEqual(set(result), {0, 1, 2, 10, 11, 12, 20, 21, 22})

    def test_mergeiterator_stochastic_rejects_zero(self) -> None:
        """weight=0 is rejected."""
        weights = [1, 0]

        with self.assertRaises(ValueError):
            MergeIterator([[1]], weights=weights)

        weights = [1, 0.0]

        with self.assertRaises(ValueError):
            MergeIterator([[1]], weights=weights)

    def test_mergeiterator_skip_zero_weight(self) -> None:
        """Iterables with zero weight are skipped."""
        iterables = [
            [0, 1, 2],
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
        ]

        weights = [1, 0, 2, 0]

        merge_iter = MergeIterator(iterables, weights=weights)

        self.assertEqual(len(merge_iter.iterables), 2)
        self.assertEqual(merge_iter.iterables[0], [0, 1, 2])
        self.assertEqual(merge_iter.iterables[1], [20, 21, 22])

        self.assertIsNotNone(merge_iter.weights)
        # pyre-ignore[16]: weights is not None after assertion
        self.assertEqual(len(merge_iter.weights), 2)
        # pyre-ignore[16]: weights is not None after assertion
        self.assertEqual(merge_iter.weights[0], 1)
        # pyre-ignore[16]: weights is not None after assertion
        self.assertEqual(merge_iter.weights[1], 2)

        result = list(merge_iter)
        self.assertEqual(set(result), {0, 1, 2, 20, 21, 22})

    def test_mergeiterator_stochastic_stop_after_N(self) -> None:
        """Values are taken from iterables with higher weights"""
        weights = [1000000, 1]

        iterables = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        ]

        result = list(MergeIterator(iterables, weights=weights, stop_after=3))
        self.assertEqual(result, [0, 1, 2])

    def test_mergeiterator_stochastic_stop_after_first_exhaustion(self) -> None:
        """Values are taken from iterables with higher weights"""
        weights = [1000000, 1]

        iterables = [
            [0, 1, 2, 3],
            [10, 11, 12, 13],
        ]

        result = list(MergeIterator(iterables, weights=weights, stop_after=-1))
        self.assertEqual(result, [0, 1, 2, 3])


class TestRepeatSource(unittest.TestCase):
    def test_repeat_source_iterable_with_shuffle(self) -> None:
        """repeat_source repeats source while calling shuffle"""

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
            self.assertEqual(next(gen), 1)
            mock_method.assert_called_with(seed=0)
            self.assertEqual(next(gen), 2)
            self.assertEqual(next(gen), 0)

            self.assertEqual(next(gen), 2)
            mock_method.assert_called_with(seed=1)
            self.assertEqual(next(gen), 0)
            self.assertEqual(next(gen), 1)

            self.assertEqual(next(gen), 0)
            mock_method.assert_called_with(seed=2)
            self.assertEqual(next(gen), 1)
            self.assertEqual(next(gen), 2)

            self.assertEqual(next(gen), 1)
            mock_method.assert_called_with(seed=3)
            self.assertEqual(next(gen), 2)
            self.assertEqual(next(gen), 0)

            self.assertEqual(next(gen), 2)
            mock_method.assert_called_with(seed=4)
            self.assertEqual(next(gen), 0)
            self.assertEqual(next(gen), 1)

    def test_repeat_source_iterable(self) -> None:
        """repeat_source works Iterable without shuffle method"""

        class _IteWithoutShuffle:
            def __init__(self) -> None:
                self.vals = list(range(3))

            def __iter__(self) -> Iterator[int]:
                yield from self.vals

        src = _IteWithoutShuffle()
        gen = iter(repeat_source(src, epoch=2))

        for _ in range(100):
            self.assertEqual(next(gen), 0)
            self.assertEqual(next(gen), 1)
            self.assertEqual(next(gen), 2)

    def test_repeat_source_picklable(self) -> None:
        """repeat_source is picklable."""

        src = list(range(10))
        src = repeat_source(src)

        serialized = pickle.dumps(src)
        src2 = pickle.loads(serialized)

        for _ in range(3):
            for i in range(10):
                self.assertEqual(next(src), i)
                self.assertEqual(next(src2), i)


class IterableWithShuffleSource:
    def __init__(self, n: int) -> None:
        self.vals = list(range(n))

    def __iter__(self) -> Iterator[int]:
        yield from self.vals

    def shuffle(self, seed: int) -> None:
        random.seed(seed)
        random.shuffle(self.vals)


class SourceIterableWithShuffle(IterableWithShuffle[int]):
    def __init__(self, n: int) -> None:
        self.i = 0
        self.vals = list(range(n))

    def shuffle(self, seed: int) -> None:
        assert isinstance(seed, int)
        self.vals = self.vals[1:] + self.vals[:1]

    def __iter__(self) -> Iterator[int]:
        yield from self.vals


class TestShuffleAndIterate(unittest.TestCase):
    def test_shuffle_and_iterate_picklable(self) -> None:
        """The result of embed_shuffle must be pickable (for multiprocessing)"""

        src = embed_shuffle(IterableWithShuffleSource(10))
        state = pickle.dumps(src)
        src2 = pickle.loads(state)

        # pyre-ignore[16]: embed_shuffle returns an object with src attribute
        self.assertEqual(src.src.vals, src2.src.vals)

    def test_shuffle_and_iterate(self) -> None:
        N = 10

        src = embed_shuffle(IterableWithShuffleSource(N))

        ref = list(range(N))
        for i in range(3):
            random.seed(i)
            random.shuffle(ref)

            hyp = list(src)
            self.assertEqual(hyp, ref)

    def test_move_iterable_to_subprocess_success_iterable_with_shuffle(self) -> None:
        """IterableWithShuffle can be executed in the subprocess."""
        iterator = iterate_in_subprocess(
            partial(embed_shuffle, SourceIterableWithShuffle(3))
        )

        self.assertEqual(list(iterator), [1, 2, 0])
        self.assertEqual(list(iterator), [2, 0, 1])
        self.assertEqual(list(iterator), [0, 1, 2])
