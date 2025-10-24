# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Module to stash things common to all the modules internal to SPDL"""

from asyncio import Queue
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable, Sequence
from typing import TypeAlias, TypeVar

__all__ = [
    "_TAsyncCallables",
    "_TCallables",
    "_TMergeOp",
]


T = TypeVar("T")
U = TypeVar("U")


_TCallables: TypeAlias = (
    Callable[[T], U]
    | Callable[[T], Iterable[U]]
    | Callable[[T], Awaitable[U]]
    | Callable[[T], AsyncIterable[U]]
)


_TAsyncCallables: TypeAlias = (
    Callable[[T], Awaitable[U]] | Callable[[T], AsyncIterable[U]]
)


_TMergeOp: TypeAlias = Callable[[str, Sequence[Queue], Queue], Awaitable[None]]
