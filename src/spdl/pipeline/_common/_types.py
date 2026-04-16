# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Module to stash things common to all the modules internal to SPDL"""

from asyncio import Queue
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeAlias, TypeVar

__all__ = [
    "_TAsyncCallables",
    "_TCallables",
    "_TMergeOp",
    "StageInfo",
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


@dataclass(frozen=True)
class StageInfo:
    """Structured identity for a pipeline stage.

    Carries the metadata that identifies a stage within a pipeline.
    Used as the primary identifier for queues, hooks, and nodes.
    """

    pipeline_id: int
    stage_id: str
    stage_name: str
    concurrency: int | None = None

    def __str__(self) -> str:
        base = self.stage_name
        if self.concurrency is not None:
            base = f"{base}[{self.concurrency}]"
        return f"{self.pipeline_id}:{self.stage_id}:{base}"


_TMergeOp: TypeAlias = Callable[[StageInfo, Sequence[Queue], Queue], Awaitable[None]]
