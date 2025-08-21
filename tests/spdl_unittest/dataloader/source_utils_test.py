# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

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


def test_embed_shuffle():
    """Iterable created by embed_shuffle calls shuffle automatically"""

    foo = IterableWithShuffle_(3)
    assert foo._seed is None
    iterable = embed_shuffle(foo)
    assert list(iterable) == [1, 2, 0]
    assert foo._seed == 0
    assert list(iterable) == [2, 0, 1]
    assert foo._seed == 1
    assert list(iterable) == [0, 1, 2]
    assert foo._seed == 2


def test_embed_shuffle_halt():
    """The value is shuffled with different seed even after an iteration is halted."""

    foo = IterableWithShuffle_(5)
    iterable = embed_shuffle(foo)

    iterator = iter(iterable)
    assert foo._seed is None
    assert next(iterator) == 1
    assert foo._seed == 0
    assert next(iterator) == 2
    del iterator

    iterator = iter(iterable)
    assert next(iterator) == 2
    assert foo._seed == 1
    assert next(iterator) == 3
    del iterator


def test_embed_shuffle_shuffle_after():
    """Iterable created by embed_shuffle calls shuffle automatically after iteration"""

    foo = IterableWithShuffle_(3)
    iterable = embed_shuffle(foo, shuffle_last=True)
    assert foo._seed is None
    assert list(iterable) == [0, 1, 2]
    assert foo._seed == 0
    assert list(iterable) == [1, 2, 0]
    assert foo._seed == 1
    assert list(iterable) == [2, 0, 1]
    assert foo._seed == 2


def test_embed_shuffle_shuffle_after_halt():
    """The value is shuffled with different seed even after an iteration is halted."""

    foo = IterableWithShuffle_(5)
    iterable = embed_shuffle(foo, shuffle_last=True)

    iterator = iter(iterable)
    assert next(iterator) == 0
    assert next(iterator) == 1
    assert foo._seed is None
    del iterator
    assert foo._seed == 0

    iterator = iter(iterable)
    assert next(iterator) == 1
    assert next(iterator) == 2
    del iterator
    assert foo._seed == 1
