# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import functools
import itertools
import pickle
import random
import unittest
import warnings
from collections.abc import Iterator
from functools import partial
from unittest.mock import patch

from spdl.pipeline import iterate_in_subprocess as _iterate_in_subprocess
from spdl.source import utils as source_utils
from spdl.source.utils import (
    embed_shuffle,
    IterableWithShuffle,
    MergeIterator,
    repeat_source,
)


def _ignore_fork_warning(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"This process \(pid=\d+\) is multi-threaded, use of "
                    r"fork\(\) may lead to deadlocks in the child"
                ),
                category=DeprecationWarning,
            )
            return fn(*args, **kwargs)

    return wrapper


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

    def test_mergeiterator_stochastic_removes_exhausted_choices(self) -> None:
        """Weighted merging stops selecting an iterator after it is exhausted."""

        class _CountingIterator(Iterator[int]):
            def __init__(self, value: int) -> None:
                self._values = iter([value])
                self.num_next_calls = 0

            def __next__(self) -> int:
                self.num_next_calls += 1
                return next(self._values)

        first = _CountingIterator(0)
        second = _CountingIterator(10)

        result = list(MergeIterator([first, second], weights=[10, 1], seed=0))

        self.assertCountEqual(result, [0, 10])
        self.assertEqual(first.num_next_calls, 2)
        self.assertEqual(second.num_next_calls, 2)

    def test_mergeiterator_stochastic_draws_lazily_through_exhaustion(self) -> None:
        """Exhaustion does not discard a batch of precomputed RNG draws."""
        with patch.object(random.Random, "randrange", return_value=0) as draw:
            result = list(MergeIterator([[0], [10]], weights=[1, 1], seed=0))

        self.assertEqual(result, [0, 10])
        self.assertEqual(draw.call_count, 2)

    def test_mergeiterator_stochastic_rejects_unrepresentable_integer_weight(
        self,
    ) -> None:
        """An integer outside float range fails validation with a clear error."""
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            MergeIterator([[1], [2]], weights=[10**400, 1])

    def test_mergeiterator_stochastic_rejects_invalid_weight_sum(self) -> None:
        """Weighted merging requires a positive finite total weight."""
        for weights in ([0.0, 0.0], [1e308, 1e308]):
            with self.subTest(weights=weights):
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    MergeIterator([[1], [2]], weights=weights)

    def test_mergeiterator_stochastic_normalizes_subnormal_weights(self) -> None:
        """Equal subnormal weights produce an approximately even distribution."""
        result = list(
            MergeIterator(
                [itertools.repeat(0), itertools.repeat(1)],
                weights=[5e-324, 5e-324],
                stop_after=10_000,
                seed=0,
            )
        )

        count = result.count(0)
        self.assertGreater(count, 4_500)
        self.assertLess(count, 5_500)

    def test_mergeiterator_stochastic_preserves_ratio_after_exhaustion(self) -> None:
        """Surviving tiny weights preserve their ratio after a source ends."""
        result = list(
            MergeIterator(
                [[0], itertools.repeat(1), itertools.repeat(2)],
                weights=[1e308, 5e-324, 5e-324],
                stop_after=10_001,
                seed=0,
            )
        )

        self.assertEqual(result.count(0), 1)
        count = result.count(1)
        self.assertGreater(count, 4_500)
        self.assertLess(count, 5_500)

    def test_mergeiterator_stochastic_converts_weights_once(self) -> None:
        """Exhausting sources does not repeat exact bigint conversion."""
        with patch.object(
            source_utils,
            "_exact_integer_weights",
            wraps=source_utils._exact_integer_weights,
        ) as convert:
            result = list(
                MergeIterator(
                    [[0], [1], [2]],
                    weights=[1e308, 5e-324, 5e-324],
                    seed=0,
                )
            )

        self.assertCountEqual(result, [0, 1, 2])
        convert.assert_called_once()

    def test_mergeiterator_stochastic_preserves_tiny_positive_weight(self) -> None:
        """A tiny positive weight retains an exact selectable bucket."""
        with patch.object(
            random.Random,
            "getrandbits",
            return_value=1,
        ):
            result = list(
                MergeIterator(
                    [itertools.repeat(0), [1]],
                    weights=[1e308, 5e-324],
                    stop_after=1,
                )
            )

        self.assertEqual(result, [1])

    def test_mergeiterator_stochastic_avoids_wide_random_draws(self) -> None:
        """Extreme finite ratios use a lazy exact binary comparison."""
        with (
            patch.object(random.Random, "randrange") as wide_draw,
            patch.object(random.Random, "getrandbits", return_value=0) as bit_draw,
        ):
            result = list(
                MergeIterator(
                    [itertools.repeat(0), itertools.repeat(1)],
                    weights=[1e308, 5e-324],
                    stop_after=1,
                )
            )

        self.assertEqual(result, [0])
        wide_draw.assert_not_called()
        bit_draw.assert_called_once_with(1)

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
        """repeat_source forwards its starting epoch when shuffling."""

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
            mock_method.assert_called_with(seed=2)
            self.assertEqual(next(gen), 2)
            self.assertEqual(next(gen), 0)

            self.assertEqual(next(gen), 2)
            mock_method.assert_called_with(seed=3)
            self.assertEqual(next(gen), 0)
            self.assertEqual(next(gen), 1)

            self.assertEqual(next(gen), 0)
            mock_method.assert_called_with(seed=4)
            self.assertEqual(next(gen), 1)
            self.assertEqual(next(gen), 2)

            self.assertEqual(next(gen), 1)
            mock_method.assert_called_with(seed=5)
            self.assertEqual(next(gen), 2)
            self.assertEqual(next(gen), 0)

            self.assertEqual(next(gen), 2)
            mock_method.assert_called_with(seed=6)
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

    @_ignore_fork_warning
    def test_move_iterable_to_subprocess_success_iterable_with_shuffle(self) -> None:
        """IterableWithShuffle can be executed in the subprocess."""
        iterator = iterate_in_subprocess(
            partial(embed_shuffle, SourceIterableWithShuffle(3))
        )

        self.assertEqual(list(iterator), [1, 2, 0])
        self.assertEqual(list(iterator), [2, 0, 1])
        self.assertEqual(list(iterator), [0, 1, 2])
