# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from collections.abc import AsyncIterable, Callable, Iterable
from concurrent.futures import Executor
from typing import Generic, TypeVar

from spdl._internal import log_api_usage_once
from spdl.pipeline._components import AsyncQueue, TaskHook
from spdl.pipeline.defs import (
    _TPipeInputs,
    Aggregate,
    AggregateConfig,
    Disaggregate,
    DisaggregateConfig,
    Pipe,
    PipeConfig,
    PipelineConfig,
    SinkConfig,
    SourceConfig,
)

from ._build import build_pipeline
from ._pipeline import Pipeline

__all__ = [
    "PipelineBuilder",
]

_LG: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")
T_ = TypeVar("T_")
U_ = TypeVar("U_")


################################################################################
# Builder
################################################################################


class PipelineBuilder(Generic[T, U]):
    """Build :py:class:`Pipeline` object.

    .. note::

       ``PipelineBuilder`` supports only chain of operations.
       If you need to build a pipeline composed of multiple sub-pipelines,
       use :py:class:`~spdl.pipeline.defs.PipelineConfig`.

    .. seealso::

       :ref:`intro`
          Explains the basic usage of ``PipelineBuilder`` and ``Pipeline``.

       :ref:`pipeline-caveats`
          Lists known anti-patterns that can cause a deadlock.

       :ref:`pipeline-parallelism`
          Covers how to switch (or combine) multi-threading and
          multi-processing in detail.

       :ref:`Example: Pipeline definitions <example-pipeline-definitions>`
          Illustrates how to build a complex pipeline that
          ``PipelineBuilder`` does not support.
    """

    def __init__(self) -> None:
        log_api_usage_once("spdl.pipeline.PipelineBuilder")

        self._src: SourceConfig[T] | None = None
        self._process_args: list[PipeConfig | AggregateConfig | DisaggregateConfig] = []
        self._sink: SinkConfig[U] | None = None

    def add_source(
        self, source: Iterable[T] | AsyncIterable[T]
    ) -> "PipelineBuilder[T, U]":
        """Attach an iterator to the source buffer.

        Args:
            source: A lightweight iterator that generates data.

                .. warning::

                   The source iterator must be lightweight as it is executed in async
                   event loop. If the iterator performs a blocking operation,
                   the entire pipeline will be blocked.
        """
        if self._src is not None:
            raise ValueError("Source already set.")

        self._src = SourceConfig(source)
        return self

    def pipe(
        self,
        op: _TPipeInputs[T_, U_],
        /,
        *,
        concurrency: int = 1,
        executor: Executor | None = None,
        name: str | None = None,
        output_order: str = "completion",
        max_failures: int | None = None,
    ) -> "PipelineBuilder[T, U]":
        """Apply an operation to items in the pipeline.

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
        """
        self._process_args.append(
            Pipe(
                op,
                concurrency=concurrency,
                executor=executor,
                name=name,
                output_order=output_order,
                max_failures=max_failures,
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
        """
        self._process_args.append(Aggregate(num_items, drop_last=drop_last))
        return self

    def disaggregate(self) -> "PipelineBuilder[T, U]":
        """Disaggregate the items in the pipeline."""
        self._process_args.append(Disaggregate())
        return self

    def add_sink(
        self,
        buffer_size: int = 3,
    ) -> "PipelineBuilder[T, U]":
        """Attach a buffer to the end of the pipeline.

        Args:
            buffer_size: The size of the buffer. Pass ``0`` for unlimited buffering.
        """
        if self._sink is not None:
            raise ValueError("Sink is already set.")

        self._sink = SinkConfig(buffer_size)
        return self

    def get_config(self) -> PipelineConfig[U]:
        """Get the pipeline configuration.

        Returns:
            A PipelineConfig object representing the current pipeline configuration.

        Raises:
            RuntimeError: If source or sink is not set.
        """
        if (src := self._src) is None:
            raise RuntimeError("Source is not set. Did you call `add_source`?")

        if (sink := self._sink) is None:
            raise RuntimeError("Sink is not set. Did you call `add_sink`?")

        return PipelineConfig(src, self._process_args, sink)

    def build(
        self,
        *,
        num_threads: int,
        max_failures: int = -1,
        report_stats_interval: float = -1,
        queue_class: type[AsyncQueue] | None = None,
        task_hook_factory: Callable[[str], list[TaskHook]] | None = None,
        stage_id: int = 0,
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

            stage_id: The index of the initial stage  used for logging.
        """
        return build_pipeline(
            self.get_config(),
            num_threads=num_threads,
            max_failures=max_failures,
            queue_class=queue_class,
            report_stats_interval=report_stats_interval,
            task_hook_factory=task_hook_factory,
            stage_id=stage_id,
        )
