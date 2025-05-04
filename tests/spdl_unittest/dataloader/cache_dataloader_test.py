# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator

import pytest
from spdl.dataloader import CacheDataLoader
from spdl.pipeline import cache_iterator


def test_cache_iterator():
    """cache_iterator returns the cached values"""

    ite = iter(cache_iterator(range(5), 3))

    assert next(ite) == 0
    assert next(ite) == 1
    assert next(ite) == 2

    assert next(ite) == 0
    assert next(ite) == 1
    assert next(ite) == 2

    assert next(ite) == 0
    assert next(ite) == 1
    assert next(ite) == 2


def test_cache_iterator_cache_return_after():
    """cache_iterator returns the cached values"""

    ite = iter(cache_iterator(range(7), 3, return_caches_after=5))

    assert next(ite) == 0
    assert next(ite) == 1
    assert next(ite) == 2
    assert next(ite) == 3
    assert next(ite) == 4

    assert next(ite) == 0
    assert next(ite) == 1
    assert next(ite) == 2

    assert next(ite) == 0
    assert next(ite) == 1
    assert next(ite) == 2


def test_cache_iterator_cache_return_after_len():
    """cache_iterator returns the cached values"""

    ite = iter(cache_iterator(range(7), 3, return_caches_after=5, stop_after=10))

    assert next(ite) == 0
    assert next(ite) == 1
    assert next(ite) == 2
    assert next(ite) == 3
    assert next(ite) == 4

    assert next(ite) == 0
    assert next(ite) == 1
    assert next(ite) == 2

    assert next(ite) == 0
    assert next(ite) == 1

    with pytest.raises(StopIteration):
        next(ite)


def test_CacheDataLoader():
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

    assert dl.n == len(dl) == N

    ite = iter(dl)

    assert next(ite) == 0
    assert next(ite) == 1
    assert next(ite) == 2

    assert next(ite) == 0
    assert next(ite) == 1

    assert next(ite) == 0
    assert next(ite) == 1

    assert next(ite) == 0

    with pytest.raises(StopIteration):
        next(ite)
