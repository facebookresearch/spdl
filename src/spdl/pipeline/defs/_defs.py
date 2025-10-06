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

from .._convert import Callables

# pyre-strict

T = TypeVar("T")
U = TypeVar("U")

__all__ = [
    "_PipeArgs",
    "_PipeType",
    "_TPipeInputs",
    "Aggregate",
    "Disaggregate",
    "Pipe",
    "_ConfigBase",
    "PipeConfig",
    "PipelineConfig",
    "SinkConfig",
    "SourceConfig",
]


class _ConfigBase:
    pass


################################################################################
# Source
################################################################################
@dataclass
class SourceConfig(Generic[T], _ConfigBase):
    """A source configuration.

    A source in Pipeline yields a series of input data, that is going to be
    processed by downstream pipes.

    .. seealso::

       - :py:config:`PipelineConfig`
       - :py:func:`spdl.pipeline.build_pipeline`
    """

    source: Iterable | AsyncIterable
    """Generates the series of source data."""

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
class PipeConfig(Generic[T, U], _ConfigBase):
    """PipeConfig()

    A pipe configuration.

    The pipe is an operation applied to an incoming data.

    Use factory functions :py:func:`Pipe`, :py:func:`Aggregate`
    or :py:func:`Disaggregate` to create a config.

    .. seealso::

       - :py:config:`PipelineConfig`
       - :py:func:`spdl.pipeline.build_pipeline`
    """

    name: str
    """Name of the pipe."""

    _type: _PipeType

    _args: _PipeArgs[T, U]

    _max_failures: int | None = None

    def __post_init__(self) -> None:
        op = self._args.op
        if inspect.iscoroutinefunction(op) or inspect.isasyncgenfunction(op):
            if self._args.executor is not None:
                raise ValueError("`executor` cannot be specified when op is async.")
        if inspect.isasyncgenfunction(op):
            if self._type == _PipeType.OrderedPipe:
                raise ValueError(
                    "pipe does not support async generator function "
                    "when `output_order` is 'input'."
                )

    def __repr__(self) -> str:
        match self._type:
            case _PipeType.Pipe | _PipeType.OrderedPipe:
                args = [
                    f"concurrency={self._args.concurrency}",
                ]
                if self._args.executor is not None:
                    args.append(f"executor={self._args.executor!r}")
                if self._type == _PipeType.OrderedPipe:
                    args.append("output_order='input'")
                return f"{self.name}({', '.join(args)})"
            case _PipeType.Aggregate | _PipeType.Disaggregate:
                return self.name
            case _:
                return str(self._type)


################################################################################
# Sink
################################################################################
@dataclass
class SinkConfig(Generic[T], _ConfigBase):
    """A sink configuration.

    The sink is where the final result of pipeline is buffered.

    .. seealso::

       - :py:config:`PipelineConfig`
       - :py:func:`spdl.pipeline.build_pipeline`
    """

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
class PipelineConfig(Generic[T, U], _ConfigBase):
    """A pipeline configuration.

    A pipeline consists of source, a series of pipes and sink.

    You can use :py:func:`spdl.pipeline.build_pipeline` to build a
    :py:class:`spdl.pipeline.Pipeline` object.
    """

    src: SourceConfig[T]
    """Source configuration."""

    pipes: Sequence[PipeConfig[Any, Any]]
    """Pipe configurations."""

    sink: SinkConfig[U]
    """Sink configuration."""

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
    """Protocol for classes with ``__getitem__`` method, such as ``dict`` and ``list``."""

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
    max_failures: int | None = None,
) -> PipeConfig[T, U]:
    """Create a :py:class:`PipeConfig`.

    A pipe applys a function or mapping to the inocming item.

    Args:
        op: A function, callable or container with ``__getitem__`` method
            (such as dict, list and tuple).
            If it's function or callable, it is inovked with the input from the input queue.
            If it's container type, the input is passed to ``__getitem__`` method.

            The function or callable must take exactly one argument, which is the output
            from the upstream. If passing around multiple objects, take
            them as a tuple or use :py:class:`~dataclasses.dataclass` and
            define a custom protocol.

            If the result of applying ``op`` to an input item is ``None``,
            the pipeline skips absorb the result and it won't be propagated to
            the downstream stages.

            Optionally, the op can be a generator function, async function or
            async generator function.

            If ``op`` is (async) generator, the items yielded are put in the output
            queue separately.

            .. warning::

                If ``op`` is synchronous geneartor, and ``executor`` is an instance of
                :py:class:`concurrent.futures.ProcessPoolExecutor`, the output items
                are not put in the output queue until the generator is exhausted.

                Async generator, or synchronous generator without ``ProcessPoolExecutor``
                does not have this issue, and the yielded items are put in the output
                queue immediately.

            .. tip::

                When passing an async op, make sure that the op does not call sync
                function inside.
                If calling a sync function, use :py:meth:`asyncio.loop.run_in_executor`
                or :py:func:`asyncio.to_thread` to delegate the execution to the thread pool.

        concurrency: The maximum number of async tasks executed concurrently.
        executor: A custom executor object to be used to convert the synchronous operation
            into asynchronous one. If ``None``, the default executor is used.

            It is invalid to provide this argument when the given op is already async.
        name: The name (prefix) to give to the task.
        output_order: If ``"completion"`` (default), the items are put to output queue
            in the order their process is completed.
            If ``"input"``, then the items are put to output queue in the order given
            in the input queue.
        max_failures: The maximnum number of failures allowed berfore the pipe operation
            is considered failure and the whole Pipeline is shutdown.
            This overrides the value provided to the :py:meth:`~PipelineBuilder.build`
            method.

    Returns:
        The config object.

    .. seealso::

       - :py:config:`PipelineConfig`
       - :py:func:`spdl.pipeline.build_pipeline`
    """

    if output_order not in ["completion", "input"]:
        raise ValueError(
            '`output_order` must be either "completion" or "input". '
            f"Found: {output_order}"
        )

    _type = _PipeType.Pipe if output_order == "completion" else _PipeType.OrderedPipe

    if isinstance(op, SupportsGetItem):
        # Note, if op is list/dict/tuple with a lot of elements, then
        # debug print on `_ProcessConfig` might produce extremely long string.
        # So it is important to extract the __getitem__ before it is passed to
        # `_ProcessConfig`.
        op = op.__getitem__

        # We could do the same for callable (__call__)
        # but usually callable class name contains readable information, so
        # we don't do that here. (it happens in to_async helper function)

    return PipeConfig(
        name=name or _get_op_name(op),
        _type=_type,
        _args=_PipeArgs(
            op=op,
            executor=executor,
            concurrency=concurrency,
        ),
        _max_failures=max_failures,
    )


def Aggregate(num_items: int, /, *, drop_last: bool = False) -> PipeConfig[Any, Any]:
    """Create a :py:class:`PipeConfig` object for aggregation.

    The aggregation buffers the incoming items and emits once enough items are buffered.

    Args:
        num_items: The number of items to buffer.
        drop_last: Drop the last aggregation if it has less than ``num_aggregate`` items.

    Returns:
        The config object.

    .. seealso::

       - :py:config:`PipelineConfig`
       - :py:func:`spdl.pipeline.build_pipeline`
    """

    # To avoid circular deps
    from .._components._pipe import _Aggregate

    name = (
        f"aggregate({num_items}, {drop_last=})"
        if drop_last
        else f"aggregate({num_items})"
    )
    return PipeConfig(
        name=name,
        _type=_PipeType.Aggregate,
        _args=_PipeArgs(
            op=_Aggregate(num_items, drop_last),
            op_requires_eof=True,
        ),
    )


def Disaggregate() -> PipeConfig[Any, Any]:
    """Create a :py:class:`PipeConfig` object for disaggregation.

    The disaggregate slices the incoming list of items and yield them
    one by one.

    Returns:
        The config object.

    .. seealso::

       - :py:config:`PipelineConfig`
       - :py:func:`spdl.pipeline.build_pipeline`
    """

    # To avoid circular deps
    from .._components._pipe import _disaggregate

    return PipeConfig(
        name="disaggregate",
        _type=_PipeType.Disaggregate,
        _args=_PipeArgs(
            op=_disaggregate,  # pyre-ignore: [6]
        ),
    )
