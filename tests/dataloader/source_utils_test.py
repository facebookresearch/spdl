# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from collections.abc import Iterator

from spdl.source.utils import embed_shuffle


class IterableWithShuffle_:
    def __init__(self, n: int) -> None:
        self.vals = list(range(n))
        self._seed: int | None = None

    def __iter__(self) -> Iterator[int]:
        yield from self.vals

    def shuffle(self, seed: int) -> None:
        # rotate
        self._seed = seed
        self.vals = self.vals[1:] + self.vals[:1]


class SourceUtilsTest(unittest.TestCase):
    def test_embed_shuffle(self):
        """Iterable created by embed_shuffle calls shuffle automatically"""

        foo = IterableWithShuffle_(3)
        self.assertIsNone(foo._seed)
        iterable = embed_shuffle(foo)
        self.assertEqual(list(iterable), [1, 2, 0])
        self.assertEqual(foo._seed, 0)
        self.assertEqual(list(iterable), [2, 0, 1])
        self.assertEqual(foo._seed, 1)
        self.assertEqual(list(iterable), [0, 1, 2])
        self.assertEqual(foo._seed, 2)

    def test_embed_shuffle_halt(self):
        """The value is shuffled with different seed even after an iteration is halted."""

        foo = IterableWithShuffle_(5)
        iterable = embed_shuffle(foo)

        iterator = iter(iterable)
        self.assertIsNone(foo._seed)
        self.assertEqual(next(iterator), 1)
        self.assertEqual(foo._seed, 0)
        self.assertEqual(next(iterator), 2)
        del iterator

        iterator = iter(iterable)
        self.assertEqual(next(iterator), 2)
        self.assertEqual(foo._seed, 1)
        self.assertEqual(next(iterator), 3)
        del iterator

    def test_embed_shuffle_shuffle_after(self):
        """Iterable created by embed_shuffle calls shuffle automatically after iteration"""

        foo = IterableWithShuffle_(3)
        iterable = embed_shuffle(foo, shuffle_last=True)
        self.assertIsNone(foo._seed)
        self.assertEqual(list(iterable), [0, 1, 2])
        self.assertEqual(foo._seed, 0)
        self.assertEqual(list(iterable), [1, 2, 0])
        self.assertEqual(foo._seed, 1)
        self.assertEqual(list(iterable), [2, 0, 1])
        self.assertEqual(foo._seed, 2)

    def test_embed_shuffle_shuffle_after_halt(self):
        """The value is shuffled with different seed even after an iteration is halted."""

        foo = IterableWithShuffle_(5)
        iterable = embed_shuffle(foo, shuffle_last=True)

        iterator = iter(iterable)
        self.assertEqual(next(iterator), 0)
        self.assertEqual(next(iterator), 1)
        self.assertIsNone(foo._seed)
        del iterator
        self.assertEqual(foo._seed, 0)

        iterator = iter(iterable)
        self.assertEqual(next(iterator), 1)
        self.assertEqual(next(iterator), 2)
        del iterator
        self.assertEqual(foo._seed, 1)
