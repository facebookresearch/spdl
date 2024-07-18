# pyre-unsafe

import logging
import warnings
from asyncio import Queue

from collections.abc import Awaitable, Callable, Iterator, Sequence
from typing import TypeVar

from . import _pipeline
from ._hook import PipelineHook


T = TypeVar("T")
U = TypeVar("U")

_LG = logging.getLogger(__name__)


class AsyncPipeline:
    """**[Deprecated]** Use :py:class:`spdl.dataloader.Pipeline`.

    Construct data processing pipeline.

    ``AsyncPipeline`` facilitates building data processing pipeline consists of multiple
    stages of async operations. It allows to configure the concurrency of each stage
    independently.

    Typically, the source is a lightweight (synchronous) iterator that generates the
    source location of data, such as file paths and URLs. The first stage op
    retrieves data from corresponding (network) storage. The subsequent stages
    process the data, such as decoding and resizing images, or decoding and resampling
    audio.
    After the preprocessings are done, the data are passed to the sink, which is
    (synchronous) queue.

    The following diagram illustrates this.

    .. mermaid::

       flowchart TD
           Source["Source (Iterator)"]
           Queue
           subgraph Op1["Op1 (Concurrency = 4)"]
               op1_1(Task 1-1)
               op1_2(Task 1-2)
               op1_3(Task 1-3)
               op1_4(Task 1-4)
           end
           subgraph Op2["Op2 (Concurrency=2)"]
               op2_1(Task 2-1)
               op2_2(Task 2-2)
           end
           Queue["Sink (Queue)"]

           Source --> Op1
           Op1 --> Op2
           Op2 --> Queue

    .. admonition:: Example: Bulk loading images

        .. code-block::

           import asyncio
           from queue import Queue

           import spdl.io

           def source():
               with open("images.txt") as f:
                   for path in f:
                       yield path

           async def decode(path):
               return await spdl.io.async_decode_image(path)

           queue = Queue()

           pipeline = (
               AsyncPipeline()
               .add_source(source())
               .pipe(decode, concurrency=10)
               .add_sink(queue)
           )

           loop = asyncio.new_event_loop()
           loop.set_default_executor(
               ThreadPoolExecutor(
                   max_workers=10,
               )
           )
           loop.run_until_complete(pipeline.run())
    """

    def __init__(self):
        warnings.warn(
            "`AsyncPipeline` has been deprecated. Please use `PipelineBuilder`.",
            category=FutureWarning,
            stacklevel=2,
        )

        self._builder = _pipeline.PipelineBuilder()

        self._queues: list[Queue] = []

        try:
            from spdl.lib import _libspdl

            _libspdl.log_api_usage("spdl.dataloader.AsyncPipeline")
        except Exception:
            pass  # ignore if not supported.

    @property
    def output_queue(self) -> Queue:
        """The output queue of the pipeline."""
        if not self._queues:
            raise ValueError("No output queue is set.")
        return self._queues[-1]

    def add_source(self, source: Iterator[T], **kwargs) -> "AsyncPipeline":
        """Attach an iterator to the source buffer.

        .. code-block::

           ┌─────────────────┐
           │ Iterator (sync) │
           └───────┬─────────┘
                   │
                  ┌▼┐
                  │ │
                  │ │ Queue
                  │ │
                  └─┘

        Args:
            source: A lightweight iterator that generates data.

                .. warning::

                   The source iterator must be lightweight as it is executed in async
                   event loop. If the iterator performs a an operation that blocks,
                   the entire pipeline will be blocked.
        """
        self._builder.add_source(iter(source), **kwargs)
        return self

    def pipe(
        self,
        afunc: Callable[[T], Awaitable[U]],
        *,
        concurrency: int = 1,
        name: str | None = None,
        hooks: Sequence[PipelineHook] | None = None,
        report_stats_interval: float | None = None,
        output_order: str = "completion",
        **kwargs,
    ) -> "AsyncPipeline":
        """Apply an async function to items in the pipeline.

        .. code-block::

                  ┌─┐
                  │ │
                  │ │ Queue
                  │ │
                  └┬┘
                   │
           ┌───────▼────────┐
           │ Async Function │
           └───────┬────────┘
                   │
                  ┌▼┐
                  │ │
                  │ │ Queue
                  │ │
                  └─┘

        Args:
            afunc: Async function applied to the items in the queue.
            concurrency: The maximum number of async tasks executed concurrently.
            name: The name (prefix) to give to the task.
            hooks: Hook objects to be attached to the stage. Hooks are intended for
                collecting stats of the stage.
                If ``None``, a default hook,
                :py:class:`~spdl.dataloader.TaskStatsHook` is used.
            report_stats_interval:
                The interval (in seconds) to log the stats of this stage when no
                ``hooks`` is provided. This argument is passed to
                :py:class:`~spdl.dataloader.TaskStatsHook`.
                This argument is effective only when ``hooks`` are not provided.
                If ``hooks`` is provided and stats report is needed,
                the ``hooks`` argument should
                include one of :py:class:`~spdl.dataloader.TaskStatsHook`
                instance with the desired interval.
            output_order: If ``"completion"`` (default), the items are put to output queue
                in the order their process is completed.
                If ``"input"``, then the items are put to output queue in the order given
                in the input queue.
        """
        self._builder.pipe(
            afunc,
            concurrency=concurrency,
            name=name,
            hooks=hooks,
            report_stats_interval=report_stats_interval,
            output_order=output_order,
            **kwargs,
        )
        return self

    def aggregate(
        self,
        num_aggregate: int,
        /,
        *,
        drop_last: bool = False,
        hooks: Sequence[PipelineHook] | None = None,
        report_stats_interval: float | None = None,
    ) -> "AsyncPipeline":
        """Buffer the items in the pipeline.

        Args:
            num_aggregate: The number of items to buffer.
            drop_last: Drop the last aggregation if it has less than ``n`` items.
            hooks: See :py:meth:`pipe`.
            report_stats_interval: See :py:meth:`pipe`.
        """
        self._builder.aggregate(
            num_aggregate,
            drop_last=drop_last,
            hooks=hooks,
            report_stats_interval=report_stats_interval,
        )
        return self

    def add_sink(self, buffer_size: int) -> "AsyncPipeline":
        """Attach a queue to the end of the pipeline.

        .. code-block::

           ┌─┐
           │ │
           │ │ Queue
           │ │
           └┬┘
            │
           ┌▼┐
           │ │
           │ │ Queue
           │ │
           └─┘

        Args:
            buffer_size: The size of the last queue.
        """
        self._builder.add_sink(buffer_size)
        return self

    def __str__(self) -> str:
        return str(self._builder)

    def _build(self, num_items):
        return self._builder._build(num_items, queues=self._queues)

    # TODO [Python 3.11]: Try TaskGroup
    async def run(self, *, num_items: int | None = None) -> None:
        """Run the pipeline until its completion. All stages are executed concurrently.

        The pipeline completes when one of the following conditions is met.

        1. Source is exhauseted and all data went through all the stages.
        2. One or more stages encounter an internal error*. In this case, the remaining
           stages are cancelled in attempt at graceful shutdown.
        3. The pipeline is cancelled. All the stages are cancelled in attempt at graceful
           shutdown

        .. admonition:: *Internal Error

           The internal error here refers to the failure happens in the execution
           path of ``AsyncPipeline``, but outside of user provided functions.
           Therefore, it does not include the errors occur in source iterators
           and async ops. For example, data acquisition failure due to network issue
           or decoding failures.
           Errors happen inside of user-provided functions are simply logged and ignored.

        Args:
            num_items: *Optional:* The maximum number of items to process.
                If ``None``, the pipeline runs until the source is exhausted.

        Raises:

            PipelineFailure: Raised when a part of the pipeline has an error.
        """
        await self._build(num_items)
