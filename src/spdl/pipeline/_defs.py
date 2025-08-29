# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from collections.abc import AsyncIterable, Iterable
from concurrent.futures import Executor
from dataclasses import dataclass
from enum import IntEnum
from typing import Generic, TypeVar

from ._convert import Callables

# pyre-strict


T = TypeVar("T")
U = TypeVar("U")

__all__ = [
    "_SourceConfig",
    "_PipeArgs",
    "_ProcessConfig",
    "_SinkConfig",
    "_PType",
]


@dataclass
class _PipeArgs(Generic[T, U]):
    op: Callables[T, U]
    executor: Executor | None = None
    concurrency: int = 1
    op_requires_eof: bool = False
    # Used to pass EOF to op.
    # Usually pipe does not pas EOF to op. This is because op is expected to be
    #  stateless, and requiring users to handle EOF is cumbersome, and there is
    # no real benefit.
    # However, some ops are exception. The aggregation (with drop_last=False)
    # requires to benotified when the pipeline reached the EOF, so that it can
    # flush the buffered items.

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError(
                f"`concurrency` value must be >= 1. Found: {self.concurrency}"
            )


@dataclass
class _SourceConfig(Generic[T]):
    source: Iterable | AsyncIterable

    def __post_init__(self) -> None:
        if not (hasattr(self.source, "__aiter__") or hasattr(self.source, "__iter__")):
            raise ValueError("Source must be either generator or async generator.")


class _PType(IntEnum):
    Pipe = 1
    OrderedPipe = 2
    Aggregate = 3
    Disaggregate = 4


@dataclass
class _ProcessConfig(Generic[T, U]):
    type_: _PType
    name: str
    args: _PipeArgs[T, U]

    def __post_init__(self) -> None:
        op = self.args.op
        if inspect.iscoroutinefunction(op) or inspect.isasyncgenfunction(op):
            if self.args.executor is not None:
                raise ValueError("`executor` cannot be specified when op is async.")
        if inspect.isasyncgenfunction(op):
            if self.type_ == _PType.OrderedPipe:
                raise ValueError(
                    "pipe does not support async generator function "
                    "when `output_order` is 'input'."
                )


@dataclass
class _SinkConfig(Generic[T]):
    buffer_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.buffer_size, int):
            raise ValueError(
                f"`buffer_size` must be int. Found: {type(self.buffer_size)}."
            )
        if self.buffer_size < 1:
            raise ValueError(
                f"`buffer_size` must be greater than 0. Found: {self.buffer_size}"
            )
