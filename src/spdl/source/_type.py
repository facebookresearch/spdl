# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

__all__ = [
    "IterableWithShuffle",
    "SizedIterable",
    "SizedIterableWithShuffle",
]

from collections.abc import Iterable, Sized
from typing import Protocol, runtime_checkable, TypeVar

T = TypeVar("T")


@runtime_checkable
class IterableWithShuffle(Iterable[T], Protocol):
    """IterableWithShuffle()

    A protocol that is often used to represent data source."""

    def shuffle(self, seed: int) -> None:
        """Apply in-place shuffling"""
        ...


@runtime_checkable
class SizedIterable(Sized, Iterable[T], Protocol):
    """SizedIterable()
    A protocol that is often used to represent data source."""

    pass


@runtime_checkable
class SizedIterableWithShuffle(SizedIterable[T], Protocol):
    """SizedIterableWithShuffle()
    A protocol that is often used to represent data source."""

    def shuffle(self, seed: int) -> None:
        """Apply in-place shuffling.

        .. note::

           The result of shuffle may not be observable until the iteration starts.
        """
        ...
