# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["IterableWithShuffle"]

from collections.abc import Iterator
from typing import Protocol, TypeVar

T = TypeVar("T")


class IterableWithShuffle(Protocol[T]):
    """IterableWithShuffle()
    A protocol that is often used to represent data source."""

    def shuffle(self, seed: int) -> None:
        """Apply in-place shuffling"""
        ...

    def __iter__(self) -> Iterator[T]:
        """Iterate over the source and yields the source data."""
        ...
