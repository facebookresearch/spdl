# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

__all__ = ["IterableWithShuffle"]

from collections.abc import Iterator
from typing import Protocol, runtime_checkable, TypeVar

T = TypeVar("T")


@runtime_checkable
class IterableWithShuffle(Protocol[T]):
    """IterableWithShuffle()
    A protocol that is often used to represent data source."""

    def shuffle(self, seed: int) -> None:
        """Apply in-place shuffling"""
        ...

    def __iter__(self) -> Iterator[T]:
        """Iterate over the source and yields the source data."""
        ...
