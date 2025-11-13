# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections.abc import Iterator

from spdl.dataloader import CacheDataLoader
from spdl.pipeline import cache_iterator


class TestCacheIterator(unittest.TestCase):
    def test_cache_iterator(self) -> None:
        """cache_iterator returns the cached values"""

        ite = iter(cache_iterator(range(5), 3))

        self.assertEqual(next(ite), 0)
        self.assertEqual(next(ite), 1)
        self.assertEqual(next(ite), 2)

        self.assertEqual(next(ite), 0)
        self.assertEqual(next(ite), 1)
        self.assertEqual(next(ite), 2)

        self.assertEqual(next(ite), 0)
        self.assertEqual(next(ite), 1)
        self.assertEqual(next(ite), 2)

    def test_cache_iterator_cache_return_after(self) -> None:
        """cache_iterator returns the cached values"""

        ite = iter(cache_iterator(range(7), 3, return_caches_after=5))

        self.assertEqual(next(ite), 0)
        self.assertEqual(next(ite), 1)
        self.assertEqual(next(ite), 2)
        self.assertEqual(next(ite), 3)
        self.assertEqual(next(ite), 4)

        self.assertEqual(next(ite), 0)
        self.assertEqual(next(ite), 1)
        self.assertEqual(next(ite), 2)

        self.assertEqual(next(ite), 0)
        self.assertEqual(next(ite), 1)
        self.assertEqual(next(ite), 2)

    def test_cache_iterator_cache_return_after_len(self) -> None:
        """cache_iterator returns the cached values"""

        ite = iter(cache_iterator(range(7), 3, return_caches_after=5, stop_after=10))

        self.assertEqual(next(ite), 0)
        self.assertEqual(next(ite), 1)
        self.assertEqual(next(ite), 2)
        self.assertEqual(next(ite), 3)
        self.assertEqual(next(ite), 4)

        self.assertEqual(next(ite), 0)
        self.assertEqual(next(ite), 1)
        self.assertEqual(next(ite), 2)

        self.assertEqual(next(ite), 0)
        self.assertEqual(next(ite), 1)

        with self.assertRaises(StopIteration):
            next(ite)


class TestCacheDataLoader(unittest.TestCase):
    def test_CacheDataLoader(self) -> None:
        """Smoke test"""

        class DL:
            def __init__(self, n: int) -> None:
                self.n = n

            def __iter__(self) -> Iterator[int]:
                yield from range(self.n)

            def __len__(self) -> int:
                return self.n

        N = 8
        dl = CacheDataLoader(DL(N), num_caches=2, return_caches_after=3, stop_after=N)

        self.assertEqual(dl.n, N)
        self.assertEqual(len(dl), N)

        ite = iter(dl)

        self.assertEqual(next(ite), 0)
        self.assertEqual(next(ite), 1)
        self.assertEqual(next(ite), 2)

        self.assertEqual(next(ite), 0)
        self.assertEqual(next(ite), 1)

        self.assertEqual(next(ite), 0)
        self.assertEqual(next(ite), 1)

        self.assertEqual(next(ite), 0)

        with self.assertRaises(StopIteration):
            next(ite)
