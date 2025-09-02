# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from collections.abc import AsyncIterable, Callable, Iterable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from typing import Any, Generic, Protocol, runtime_checkable, TypeAlias, TypeVar

from ._convert import Callables

# pyre-strict

T = TypeVar("T")
U = TypeVar("U")

__all__ = [
    "_SourceConfig",
    "_PipeType",
    "_PipeConfig",
    "_SinkConfig",
    "PipelineConfig",
    "Aggregate",
    "Disaggregate",
    "Pipe",
]


################################################################################
# Source
################################################################################
@dataclass
class _SourceConfig(Generic[T]):
    source: Iterable | AsyncIterable

    def __post_init__(self) -> None:
        if not (hasattr(self.source, "__aiter__") or hasattr(self.source, "__iter__")):
            raise ValueError("Source must be either generator or async generator.")

    def __repr__(self) -> str:
        # Overwrite source repr because it might print a huge string.
        source = f"<{self.source.__class__.__name__} object at 0x{id(self.source):0x}>"
        return f"SourceConfig({source=})"


################################################################################
# Pipe
################################################################################
class _PipeType(IntEnum):
    Pipe = 1
    OrderedPipe = 2
    Aggregate = 3
    Disaggregate = 4


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
class _PipeConfig(Generic[T, U]):
    type_: _PipeType
    name: str
    args: _PipeArgs[T, U]

    def __post_init__(self) -> None:
        op = self.args.op
        if inspect.iscoroutinefunction(op) or inspect.isasyncgenfunction(op):
            if self.args.executor is not None:
                raise ValueError("`executor` cannot be specified when op is async.")
        if inspect.isasyncgenfunction(op):
            if self.type_ == _PipeType.OrderedPipe:
                raise ValueError(
                    "pipe does not support async generator function "
                    "when `output_order` is 'input'."
                )

    def __repr__(self) -> str:
        match self.type_:
            case _PipeType.Pipe | _PipeType.OrderedPipe:
                args = [
                    f"concurrency={self.args.concurrency}",
                ]
                if self.args.executor is not None:
                    args.append(f"executor={self.args.executor!r}")
                if self.type_ == _PipeType.OrderedPipe:
                    args.append("output_order='input'")
                return f"{self.name}({', '.join(args)})"
            case _PipeType.Aggregate | _PipeType.Disaggregate:
                return self.name
            case _:
                return str(self.type_)


################################################################################
# Sink
################################################################################
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


##############################################################################
# Top-level Config
##############################################################################
@dataclass
class PipelineConfig(Generic[T, U]):
    src: _SourceConfig[T]
    pipes: Sequence[_PipeConfig[Any, Any]]
    sink: _SinkConfig[U]

    def __repr__(self) -> str:
        args = ["PipelineConfig"]
        args.append(f" - Source: {self.src!r}")
        if self.pipes:
            args.append(" - Pipes:")
        for p in self.pipes:
            args.append(f"   - {p!r}")
        args.append(f" - Sink: {self.sink!r}")
        return "\n".join(args)


##############################################################################
# Specialization for ease of use for users.
##############################################################################
def _get_op_name(op: Callable) -> str:
    if isinstance(op, partial):
        return _get_op_name(op.func)
    return getattr(op, "__name__", op.__class__.__name__)


@runtime_checkable
class SupportsGetItem(Protocol[T, U]):
    def __getitem__(self, key: T) -> U: ...


_TPipeInputs: TypeAlias = Callables[T, U] | SupportsGetItem[T, U]


def Pipe(
    op: _TPipeInputs[T, U],
    /,
    *,
    concurrency: int = 1,
    executor: Executor | None = None,
    name: str | None = None,
    output_order: str = "completion",
) -> _PipeConfig[T, U]:
    if output_order not in ["completion", "input"]:
        raise ValueError(
            '`output_order` must be either "completion" or "input". '
            f"Found: {output_order}"
        )

    type_ = _PipeType.Pipe if output_order == "completion" else _PipeType.OrderedPipe

    if isinstance(op, SupportsGetItem):
        # Note, if op is list/dict/tuple with a lot of elements, then
        # debug print on `_ProcessConfig` might produce extremely long string.
        # So it is important to extract the __getitem__ before it is passed to
        # `_ProcessConfig`.
        op = op.__getitem__

        # We could do the same for callable (__call__)
        # but usually callable class name contains readable information, so
        # we don't do that here. (it happens in to_async helper function)

    return _PipeConfig(
        type_=type_,
        name=name or _get_op_name(op),
        args=_PipeArgs(
            op=op,
            executor=executor,
            concurrency=concurrency,
        ),
    )


def Aggregate(num_items: int, /, *, drop_last: bool = False) -> _PipeConfig[Any, Any]:
    # To avoid circular deps
    from ._components._pipe import _Aggregate

    name = (
        f"aggregate({num_items}, {drop_last=})"
        if drop_last
        else f"aggregate({num_items})"
    )
    return _PipeConfig(
        _PipeType.Aggregate,
        name=name,
        args=_PipeArgs(
            op=_Aggregate(num_items, drop_last),
            op_requires_eof=True,
        ),
    )


def Disaggregate() -> _PipeConfig[Any, Any]:
    # To avoid circular deps
    from ._components._pipe import _disaggregate

    return _PipeConfig(
        _PipeType.Disaggregate,
        name="disaggregate",
        args=_PipeArgs(
            op=_disaggregate,  # pyre-ignore: [6]
        ),
    )
