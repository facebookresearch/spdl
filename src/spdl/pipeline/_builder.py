# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from collections.abc import AsyncIterable, Callable, Iterable, Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from functools import partial
from typing import Any, Generic, TypeVar

from spdl._internal import log_api_usage_once

from ._components._build import (
    _build_pipeline_coro,
    _ProcessConfig,
    _PType,
    _SinkConfig,
    _SourceConfig,
    PipelineFailure,
)
from ._components._pipe import _Aggregate, _disaggregate, _PipeArgs
from ._convert import Callables
from ._hook import TaskHook, TaskStatsHook as DefaultHook
from ._pipeline import Pipeline
from ._queue import AsyncQueue, StatsQueue as DefaultQueue
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
# Build function
################################################################################
def _get_desc(
    src: _SourceConfig[T] | None,
    process_args: list[_ProcessConfig],  # pyre-ignore: [24]
    sink: _SinkConfig[U] | None,
) -> str:
    parts = []
    if (src_ := src) is not None:
        src_repr = getattr(src_.source, "__name__", type(src_.source).__name__)
        parts.append(f"  - src: {src_repr}")
    else:
        parts.append("  - src: n/a")

    for cfg in process_args:
        args = cfg.args
        match cfg.type_:
            case _PType.Pipe:
                part = f"{cfg.name}(concurrency={args.concurrency})"
            case _PType.OrderedPipe:
                part = (
                    f"{cfg.name}(concurrency={args.concurrency}, "
                    'output_order="input")'
                )
            case _PType.Aggregate | _PType.Disaggregate:
                part = cfg.name
            case _:
                part = str(cfg.type_)
        parts.append(f"  - {part}")

    if (sink_ := sink) is not None:
        parts.append(f"  - sink: buffer_size={sink_.buffer_size}")

    return "\n".join(parts)


def _build_pipeline(
    src: _SourceConfig[T],
    process_args: list[_ProcessConfig],  # pyre-ignore: [24]
    sink: _SinkConfig[U],
    *,
    num_threads: int,
    max_failures: int,
    queue_class: type[AsyncQueue[...]],
    task_hook_factory: Callable[[str], list[TaskHook]],
) -> Pipeline[U]:
    coro, queues = _build_pipeline_coro(
        src,
        process_args,
        sink,
        max_failures,
        queue_class,
        task_hook_factory,
    )

    executor = ThreadPoolExecutor(
        max_workers=num_threads,
        thread_name_prefix="spdl_worker_thread_",
    )
    return Pipeline(coro, queues, executor, desc=_get_desc(src, process_args, sink))


################################################################################
# Build
################################################################################


def _get_op_name(op: Callable) -> str:
    if isinstance(op, partial):
        return _get_op_name(op.func)
    return getattr(op, "__name__", op.__class__.__name__)


class PipelineBuilder(Generic[T, U]):
    """Build :py:class:`Pipeline` object.

    .. seealso::

       - :ref:`intro`
         explains the basic usage of ``PipelineBuilder`` and ``Pipeline``.
       - :ref:`pipeline-caveats`
         lists known anti-patterns that can cause a deadlock.
       - :ref:`pipeline-parallelism`
         covers how to switch (or combine)
         multi-threading and multi-processing in detail.
    """

    def __init__(self) -> None:
        log_api_usage_once("spdl.pipeline.PipelineBuilder")

        self._src: _SourceConfig[T] | None = None
        self._process_args: list[_ProcessConfig] = []  # pyre-ignore: [24]
        self._sink: _SinkConfig[U] | None = None

    def add_source(
        self, source: Iterable[T] | AsyncIterable[T]
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
        """
        if self._src is not None:
            raise ValueError("Source already set.")

        self._src = _SourceConfig(source)
        return self

    def pipe(
        self,
        op: Callables[T_, U_],
        /,
        *,
        concurrency: int = 1,
        executor: Executor | None = None,
        name: str | None = None,
        output_order: str = "completion",
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
        """
        if output_order not in ["completion", "input"]:
            raise ValueError(
                '`output_order` must be either "completion" or "input". '
                f"Found: {output_order}"
            )

        type_ = _PType.Pipe if output_order == "completion" else _PType.OrderedPipe

        self._process_args.append(
            _ProcessConfig(
                type_=type_,
                name=name or _get_op_name(op),
                args=_PipeArgs(
                    op=op,
                    executor=executor,
                    concurrency=concurrency,
                ),
            )
        )
        return self

    def aggregate(
        self,
        num_items: int,
        /,
        *,
        drop_last: bool = False,
    ) -> "PipelineBuilder[T, U]":
        """Buffer the items in the pipeline.

        Args:
            num_items: The number of items to buffer.
            drop_last: Drop the last aggregation if it has less than ``num_aggregate`` items.
            hooks: See :py:meth:`pipe`.
        """
        name = (
            f"aggregate({num_items}, {drop_last=})"
            if drop_last
            else f"aggregate({num_items})"
        )
        self._process_args.append(
            _ProcessConfig(
                _PType.Aggregate,
                name=name,
                args=_PipeArgs(
                    op=_Aggregate(num_items, drop_last),
                    op_requires_eof=True,
                ),
            )
        )
        return self

    def disaggregate(self) -> "PipelineBuilder[T, U]":
        """Disaggregate the items in the pipeline.

        Args:
            hooks: See :py:meth:`pipe`.

            queue_class: A queue class, used to connect this stage and the next stage.
                Must be a subclassing type (not an instance) of :py:class:`AsyncQueue`.
                Default: :py:class:`StatsQueue`.
        """
        self._process_args.append(
            _ProcessConfig(
                _PType.Disaggregate,
                name="disaggregate",
                args=_PipeArgs(
                    op=_disaggregate,  # pyre-ignore: [6]
                ),
            ),
        )
        return self

    def add_sink(
        self,
        buffer_size: int = 3,
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

        self._sink = _SinkConfig(buffer_size)
        return self

    def __str__(self) -> str:
        return "\n".join(
            [repr(self), _get_desc(self._src, self._process_args, self._sink)]
        )

    def build(
        self,
        *,
        num_threads: int,
        max_failures: int = -1,
        report_stats_interval: float = -1,
        queue_class: type[AsyncQueue[...]] | None = None,
        task_hook_factory: Callable[[str], list[TaskHook]] | None = None,
    ) -> Pipeline[U]:
        """Build the pipeline.

        Args:
            num_threads: The number of threads in the thread pool attached to
                async event loop.

            max_failures: The maximum number of failures each pipe stage can have before
                the pipeline is halted. Setting ``-1`` (default) disables it.

            report_stats_interval: When provided, report the pipeline performance stats
                every given interval. Unit: [sec]

                This is only effective if there is no custom hook or custom AsyncQueue
                provided for stages. The argument is passed to
                :py:class:`TaskStatsHook` and :py:class:`StatsQueue`.

                If a custom stage hook is provided and stats report is needed,
                you can instantiate :py:class:`TaskStatsHook` and include
                it in the hooks provided to :py:meth:`PipelineBuilder.pipe`.

                Similarly if you are providing a custom :py:class:`AsyncQueue` class,
                you need to implement the same logic by your self.

            queue_class: If provided, override the queue class used to connect stages.
                Must be a class (not an instance) inherits :py:class:`AsyncQueue`.

            task_hook_factory: If provided, used to create task hook objects, given a
                name of the stage. If ``None``, a default hook,
                :py:class:`TaskStatsHook` is used.
                To disable hooks, provide a function that returns an empty list.
        """
        if (src := self._src) is None:
            raise RuntimeError("Source is not set.")

        if (sink := self._sink) is None:
            raise RuntimeError("Sink is not set.")

        def _hook_factory(name: str) -> list[TaskHook]:
            return [DefaultHook(name=name, interval=report_stats_interval)]

        return _build_pipeline(
            src,
            self._process_args,
            sink,
            num_threads=num_threads,
            max_failures=max_failures,
            queue_class=(
                partial(DefaultQueue, interval=report_stats_interval)
                if queue_class is None
                else queue_class
            ),
            task_hook_factory=(
                _hook_factory if task_hook_factory is None else task_hook_factory
            ),
        )


################################################################################
# run in subprocess
################################################################################


class _Wrapper(Generic[U]):
    def __init__(
        self,
        builder: PipelineBuilder[T, U],
        num_threads: int,
        max_failures: int,
        report_stats_interval: float,
        queue_class: type[AsyncQueue[T]] | None,
        task_hook_factory: Callable[[str], list[TaskHook]] | None = None,
    ) -> None:
        self.builder = builder
        self.num_threads = num_threads
        self.max_failures = max_failures
        self.report_stats_interval = report_stats_interval
        self.queue_class = queue_class
        self.task_hook_factory = task_hook_factory

    def __iter__(self) -> Iterator[U]:
        pipeline = self.builder.build(
            num_threads=self.num_threads,
            max_failures=self.max_failures,
            report_stats_interval=self.report_stats_interval,
            queue_class=self.queue_class,
            task_hook_factory=self.task_hook_factory,
        )
        with pipeline.auto_stop():
            yield from pipeline


def run_pipeline_in_subprocess(
    builder: PipelineBuilder[T, U],
    *,
    num_threads: int,
    max_failures: int = -1,
    report_stats_interval: float = -1,
    queue_class: type[AsyncQueue[T]] | None = None,
    task_hook_factory: Callable[[str], list[TaskHook]] | None = None,
    **kwargs: dict[str, Any],
) -> Iterable[T]:
    """Run the given Pipeline in a subprocess, and iterate on the result.

    Args:
        builder: The definition of :py:class:`Pipeline`.
        num_threads,max_failures,report_stats_interval,queue_class,task_hook_factory:
            Passed to :py:meth:`PipelineBuilder.build`.
        kwargs: Passed to :py:func:`iterate_in_subprocess`.

    Yields:
        The results yielded from the pipeline.

    .. seealso::

       - :py:func:`iterate_in_subprocess` implements the logic for manipulating an iterable
         in a subprocess.
       - :ref:`parallelism-performance` for the context in which this function was created.
    """
    return iterate_in_subprocess(
        fn=partial(
            _Wrapper,
            builder=builder,
            num_threads=num_threads,
            max_failures=max_failures,
            report_stats_interval=report_stats_interval,
            queue_class=queue_class,
            task_hook_factory=task_hook_factory,
        ),
        **kwargs,  # pyre-ignore: [6]
    )
