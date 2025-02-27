# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import inspect
import logging
from collections.abc import (
    AsyncIterable,
    Callable,
    Iterable,
    Iterator,
)
from concurrent.futures import Executor, ThreadPoolExecutor
from functools import partial
from typing import Any, Generic, TypeVar

from ._components._build import (
    _build_pipeline,
    _ProcessConfig,
    _PType,
    _SinkConfig,
    _SourceConfig,
    PipelineFailure,
)
from ._components._pipe import (
    _Aggregate,
    _disaggregate,
    _PipeArgs,
)
from ._convert import Callables
from ._hook import PipelineHook
from ._pipeline import Pipeline
from ._queue import AsyncQueue
from ._utils import iterate_in_subprocess

__all__ = [
    "PipelineFailure",
    "PipelineBuilder",
    "_get_op_name",
    "run_pipeline_in_subprocess",
]

_LG: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")
T_ = TypeVar("T_")
U_ = TypeVar("U_")


################################################################################
# Build
################################################################################


def _get_op_name(op: Callable) -> str:
    if isinstance(op, partial):
        return _get_op_name(op.func)
    return getattr(op, "__name__", op.__class__.__name__)


class PipelineBuilder(Generic[T, U]):
    """Build :py:class:`~spdl.pipeline.Pipeline` object.

    See :py:class:`~spdl.pipeline.Pipeline` for details.
    """

    def __init__(self) -> None:
        self._src: _SourceConfig[T] | None = None
        self._process_args: list[_ProcessConfig] = []  # pyre-ignore: [24]
        self._sink: _SinkConfig[U] | None = None

        self._num_aggregate = 0
        self._num_disaggregate = 0

    def add_source(
        self,
        source: Iterable[T] | AsyncIterable[T],
        *,
        queue_class: type[AsyncQueue[T]] | None = None,
    ) -> "PipelineBuilder[T, U]":
        """Attach an iterator to the source buffer.

        .. code-block::

           ┌─────────────────┐
           │ (Async)Iterator │
           └───────┬─────────┘
                   ▼

        Args:
            source: A lightweight iterator that generates data.

                .. warning::

                   The source iterator must be lightweight as it is executed in async
                   event loop. If the iterator performs a blocking operation,
                   the entire pipeline will be blocked.

            queue_class: A queue class, used to connect this stage and the next stage.
                Must be a subclassing type (not an instance) of :py:class:`AsyncQueue`.
                Default: :py:class:`StatsQueue`.
        """
        if self._src is not None:
            raise ValueError("Source already set.")

        if not (hasattr(source, "__aiter__") or hasattr(source, "__iter__")):
            raise ValueError("Source must be either generator or async generator.")

        self._src = _SourceConfig(source, queue_class)
        return self

    def pipe(
        self,
        op: Callables[T_, U_],
        /,
        *,
        concurrency: int = 1,
        executor: Executor | None = None,
        name: str | None = None,
        hooks: list[PipelineHook] | None = None,
        output_order: str = "completion",
        queue_class: type[AsyncQueue[U_]] | None = None,
    ) -> "PipelineBuilder[T, U]":
        """Apply an operation to items in the pipeline.

        .. code-block::

                 │
           ┌─────▼─────┐
           │ Operation │
           └─────┬─────┘
                 ▼

        Args:
            op: A function applied to items in the queue.
                The function must take exactly one argument, which is the output
                from the upstream. If passing around multiple objects, take
                them as a tuple or use :py:class:`~dataclasses.dataclass` and
                define a custom protocol.

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
                   If calling a sync function, use :py:func:`asyncio.loop.run_in_executor`
                   or :py:func:`asyncio.to_thread` to delegate the execution to the thread pool.

            concurrency: The maximum number of async tasks executed concurrently.
            executor: A custom executor object to be used to convert the synchronous operation
                into asynchronous one. If ``None``, the default executor is used.

                It is invalid to provide this argument when the given op is already async.
            name: The name (prefix) to give to the task.
            hooks: Hook objects to be attached to the stage. Hooks are intended for
                collecting stats of the stage.
                If ``None``, a default hook,
                :py:class:`~spdl.pipeline.TaskStatsHook` is used.
            output_order: If ``"completion"`` (default), the items are put to output queue
                in the order their process is completed.
                If ``"input"``, then the items are put to output queue in the order given
                in the input queue.
            queue_class: A queue class, used to connect this stage and the next stage.
                Must be a subclassing type (not an instance) of :py:class:`AsyncQueue`.
                Default: :py:class:`StatsQueue`.
        """
        if output_order not in ["completion", "input"]:
            raise ValueError(
                '`output_order` must be either "completion" or "input". '
                f"Found: {output_order}"
            )

        if inspect.iscoroutinefunction(op) or inspect.isasyncgenfunction(op):
            if executor is not None:
                raise ValueError("`executor` cannot be specified when op is async.")
        if inspect.isasyncgenfunction(op):
            if output_order == "input":
                raise ValueError(
                    "pipe does not support async generator function "
                    "when `output_order` is 'input'."
                )
        name_ = name or _get_op_name(op)

        type_ = _PType.Pipe if output_order == "completion" else _PType.OrderedPipe

        self._process_args.append(
            _ProcessConfig(
                type_=type_,
                args=_PipeArgs(
                    name=name_,
                    op=op,
                    executor=executor,
                    concurrency=concurrency,
                    hooks=hooks,
                ),
                queue_class=queue_class,  # pyre-ignore: [6]
            )
        )
        return self

    def aggregate(
        self,
        num_items: int,
        /,
        *,
        drop_last: bool = False,
        hooks: list[PipelineHook] | None = None,
        queue_class: type[AsyncQueue[T]] | None = None,
    ) -> "PipelineBuilder[T, U]":
        """Buffer the items in the pipeline.

        Args:
            num_items: The number of items to buffer.
            drop_last: Drop the last aggregation if it has less than ``num_aggregate`` items.
            hooks: See :py:meth:`pipe`.
            queue_class: A queue class, used to connect this stage and the next stage.
                Must be a subclassing type (not an instance) of :py:class:`AsyncQueue`.
                Default: :py:class:`StatsQueue`.
        """

        if drop_last:
            name = f"aggregate_{self._num_aggregate}({num_items}, {drop_last=})"
        else:
            name = f"aggregate_{self._num_aggregate}({num_items})"
        self._num_aggregate += 1

        self._process_args.append(
            _ProcessConfig(
                _PType.Aggregate,
                _PipeArgs(
                    name=name,
                    op=_Aggregate(num_items, drop_last),
                    hooks=hooks,
                    op_requires_eof=True,
                ),
                queue_class=queue_class,
            )
        )
        return self

    def disaggregate(
        self,
        /,
        *,
        hooks: list[PipelineHook] | None = None,
        queue_class: type[AsyncQueue[T_]] | None = None,
    ) -> "PipelineBuilder[T, U]":
        """Disaggregate the items in the pipeline.

        Args:
            hooks: See :py:meth:`pipe`.

            queue_class: A queue class, used to connect this stage and the next stage.
                Must be a subclassing type (not an instance) of :py:class:`AsyncQueue`.
                Default: :py:class:`StatsQueue`.
        """
        name = f"disaggregate_{self._num_disaggregate}()"
        self._num_disaggregate += 1

        self._process_args.append(
            _ProcessConfig(
                _PType.Disaggregate,
                _PipeArgs(
                    name=name,
                    op=_disaggregate,  # pyre-ignore: [6]
                    hooks=hooks,
                ),
                queue_class=queue_class,
            ),
        )
        return self

    def add_sink(
        self,
        buffer_size: int = 3,
        queue_class: type[AsyncQueue[U]] | None = None,
    ) -> "PipelineBuilder[T, U]":
        """Attach a buffer to the end of the pipeline.

        .. code-block::

            │
           ┌▼┐
           │ │ buffer
           └─┘

        Args:
            buffer_size: The size of the buffer. Pass ``0`` for unlimited buffering.

            queue_class: A queue class, used to connect this stage and the next stage.
                Must be a subclassing type (not an instance) of :py:class:`AsyncQueue`.
                Default: :py:class:`StatsQueue`.
        """
        if self._sink is not None:
            raise ValueError("Sink is already set.")
        if not isinstance(buffer_size, int):
            raise ValueError(f"`buffer_size` must be int. Found: {type(buffer_size)}.")
        if buffer_size < 1:
            raise ValueError(
                f"`buffer_size` must be greater than 0. Found: {buffer_size}"
            )

        self._sink = _SinkConfig(buffer_size, queue_class)
        return self

    def _get_desc(self) -> list[str]:
        parts = []
        if self._src is not None:
            src = self._src
            src_repr = getattr(src.source, "__name__", type(src.source).__name__)
            parts.append(f"  - src: {src_repr}")
        else:
            parts.append("  - src: n/a")

        for cfg in self._process_args:
            args = cfg.args
            match cfg.type_:
                case _PType.Pipe:
                    part = f"{args.name}(concurrency={args.concurrency})"
                case _PType.OrderedPipe:
                    part = (
                        f"{args.name}(concurrency={args.concurrency}, "
                        'output_order="input")'
                    )
                case _PType.Aggregate | _PType.Disaggregate:
                    part = args.name
                case _:
                    part = str(cfg.type_)
            parts.append(f"  - {part}")

        if (sink := self._sink) is not None:
            parts.append(f"  - sink: buffer_size={sink.buffer_size}")

        return parts

    def __str__(self) -> str:
        return "\n".join([repr(self), *self._get_desc()])

    def build(
        self,
        *,
        num_threads: int,
        report_stats_interval: float = -1,
    ) -> Pipeline[U]:
        """Build the pipeline.

        Args:
            num_threads: The number of threads in the thread pool attached to
                async event loop.
            report_stats_interval: When provided, report the pipline performance stats
                every given interval. Unit: [sec]

                This is only effective if there is no custom hook or custom AsyncQueue
                provided for stages. The argument is passed to
                :py:class:`~TaskStatsHook` and :py:class:`~StatusQueue`.

                If a custom stage hook is provided and stats report is needed,
                you can instantiate :py:class:`~spdl.pipeline.TaskStatsHook` and include
                it in the hooks provided to :py:meth:`PipelineBuilder.pipe`.

                Similarly if you are providing a custom :py:class:`~AsyncQueue` class,
                you need to implement the same logic by your self.
        """
        if (src := self._src) is None:
            raise RuntimeError("Source is not set.")

        if (sink := self._sink) is None:
            raise RuntimeError("Sink is not set.")

        coro, queues = _build_pipeline(
            src, self._process_args, sink, report_stats_interval
        )

        executor = ThreadPoolExecutor(
            max_workers=num_threads,
            thread_name_prefix="spdl_",
        )
        return Pipeline(coro, queues, executor, desc=self._get_desc())


def _run_pipeline(builder: PipelineBuilder[T, U], num_threads: int) -> Iterator[U]:
    pipeline = builder.build(num_threads=num_threads)
    with pipeline.auto_stop():
        yield from pipeline


def run_pipeline_in_subprocess(
    builder: PipelineBuilder[T, U],
    *,
    num_threads: int,
    **kwargs: dict[str, Any],
) -> Iterator[T]:
    """Run the given Pipeline in a subprocess, and iterate on the result.

    Args:
        builder: The definition of :py:class:`Pipeline`.
        num_threads: Passed to :py:meth:`PipelineBuilder.build`.
        kwargs: Passed to :py:func:`iterate_in_subprocess`.

    Yields:
        The results yielded from the pipeline.
    """
    yield from iterate_in_subprocess(
        fn=partial(_run_pipeline, builder=builder, num_threads=num_threads),
        **kwargs,  # pyre-ignore: [6]
    )
