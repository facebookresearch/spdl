# pyre-unsafe

import asyncio
import logging
from asyncio import Queue as AsyncQueue
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import asynccontextmanager
from queue import Queue
from typing import TypeVar

from ._hook import _stage_hooks, _task_hooks, PipelineHook, TaskStatsHook
from ._utils import create_task

__all__ = [
    "AsyncPipeline",
    "PipelineFailure",
]

_LG = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


# Sentinel objects used to instruct AsyncPipeline to take special actions.
_EOF = object()  # Indicate the end of stream.
_SKIP = object()  # Indicate that there is no data to process.


@asynccontextmanager
async def _put_eof_when_done(queue, put_on_error=False):
    try:
        yield
    except Exception:
        if put_on_error:
            await queue.put(_EOF)
        else:
            raise
    else:
        await queue.put(_EOF)


################################################################################
# _pipe
################################################################################


async def _pipe(
    input_queue: AsyncQueue[T],
    afunc: Callable[[T], Awaitable[U]],
    output_queue: AsyncQueue[U],
    concurrency: int = 1,
    name: str = "pipe",
    hooks: Sequence[PipelineHook] | None = None,
    report_stats_interval: float | None = None,
    _pipe_eof: bool = False,
) -> None:
    if input_queue is output_queue:
        raise ValueError("input queue and output queue must be different")

    if concurrency < 1:
        raise ValueError("`concurrency` value must be >= 1")

    hooks = (
        [TaskStatsHook(name, concurrency, interval=report_stats_interval)]
        if hooks is None
        else hooks
    )

    async def _wrap(coro: Awaitable[U]) -> None:
        async with _task_hooks(hooks):  # pyre-ignore: [16]
            result = await coro

        await output_queue.put(result)

    async with _stage_hooks(hooks), _put_eof_when_done(output_queue):
        i, tasks = 0, set()
        while True:
            item = await input_queue.get()
            if item is _SKIP:
                continue
            if item is _EOF and not _pipe_eof:
                break
            # note: Make sure that `afunc` is called directly in this function,
            # so as to detect user error. (incompatible `afunc` and `iterator` combo)
            task = create_task(_wrap(afunc(item)), name=f"{name}_{(i := i + 1)}")
            tasks.add(task)

            if len(tasks) >= concurrency:
                done, tasks = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

            if item is _EOF:
                break

        if tasks:
            await asyncio.wait(tasks)


################################################################################
# _enqueue
################################################################################


async def _enqueue(
    iterator: Iterator[T],
    queue: AsyncQueue[T],
    max_items: int | None = None,
) -> None:
    async with _put_eof_when_done(queue, put_on_error=True):
        num_items = 0
        for item in iterator:
            if item is not _SKIP:
                await queue.put(item)
                num_items += 1
                if max_items is not None and num_items >= max_items:
                    return


################################################################################
# _enqueue
################################################################################


async def _dequeue(
    input_queue: AsyncQueue[T],
    output_queue: Queue[T],
    timeout: int | float | None,
):
    loop = asyncio.get_running_loop()
    while (item := await input_queue.get()) is not _EOF:
        if item is not _SKIP:
            await loop.run_in_executor(
                None, lambda: output_queue.put(item, timeout=timeout)
            )  # pyre-ignore: [6]


################################################################################
# AsyncPipeline
################################################################################


# TODO [Python 3.11]: Migrate to ExceptionGroup
class PipelineFailure(Exception):
    """PipelineFailure()
    Thrown by :py:meth:`spdl.dataloader.AsyncPipeline.run`
    when pipeline encounters an error."""

    def __init__(self, errs):
        msg = []
        for k, v in errs.items():
            e = str(v)
            msg.append(f"{k}:{e if e else type(v).__name__}")
        msg.sort()

        super().__init__(self, ", ".join(msg))

        # This is for unittesting.
        self._errs = errs


class AsyncPipeline:
    """**[Experimental]** Construct data processing pipeline.

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


    Args:
        buffer_size: The size of the first queue to which data are
            pushed from the source iterator.
    """

    def __init__(self, *, buffer_size: int = 1):
        self.queues: list[AsyncQueue] = [AsyncQueue(buffer_size)]

        self._source = None
        self._process_args: list[tuple[str, dict]] = []

        self._sink = None
        self._sink_timeout = None

        try:
            from spdl.lib import _libspdl

            _libspdl.log_api_usage("spdl.dataloader.AsyncPipeline")
        except Exception:
            pass  # ignore if not supported.

    @property
    def output_queue(self) -> AsyncQueue:
        """The output queue of the pipeline."""
        return self.queues[-1]

    def add_source(self, source: Iterator[T]) -> "AsyncPipeline":
        """Attach an iterator to the source buffer.

        .. code-block::

           ┌─────────────────┐
           │ Iterator (sync) │
           └───────┬─────────┘
                   │
                  ┌▼┐
                  │ │
                  │ │ AsyncQueue
                  │ │
                  └─┘

        Args:
            source: A lightweight iterator that generates data.

                .. warning::

                   The source iterator must be lightweight as it is executed in async
                   event loop. If the iterator performs a an operation that blocks,
                   the entire pipeline will be blocked.
        """
        if self._source is not None:
            raise ValueError("Source already set.")
        self._source = iter(source)
        return self

    def pipe(
        self,
        afunc: Callable[[T], Awaitable[U]],
        *,
        concurrency: int = 1,
        buffer_size: int = 10,
        name: str | None = None,
        hooks: Sequence[PipelineHook] | None = None,
        report_stats_interval: float | None = None,
    ) -> "AsyncPipeline":
        """Apply an async function to items in the pipeline.

        .. code-block::

                  ┌─┐
                  │ │
                  │ │ AsyncQueue
                  │ │
                  └┬┘
                   │
           ┌───────▼────────┐
           │ Async Function │
           └───────┬────────┘
                   │
                  ┌▼┐
                  │ │
                  │ │ AsyncQueue
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
        """
        if name is None:
            if hasattr(afunc, "__name__"):
                name = afunc.__name__
            else:
                name = afunc.__class__.__name__

        self.queues.append(AsyncQueue(buffer_size))

        self._process_args.append(
            (
                "pipe",
                {
                    "input_queue": self.queues[-2],
                    "afunc": afunc,
                    "output_queue": self.queues[-1],
                    "concurrency": concurrency,
                    "name": name,
                    "hooks": hooks,
                    "report_stats_interval": report_stats_interval,
                },
            )
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
            n: The number of items to buffer.
            drop_last: Drop the last aggregation if it has less than ``n`` items.
            hooks: See :py:meth:`pipe`.
            report_stats_interval: See :py:meth:`pipe`.
        """

        vals = [[]]

        async def aggregate(i):
            if i is not _EOF:
                vals[0].append(i)

            if (i is _EOF and vals[0]) or (len(vals[0]) >= num_aggregate):
                ret = vals.pop(0)
                vals.append([])
                return ret
            return _SKIP

        name = f"aggregate({num_aggregate}, {drop_last=})"

        self.queues.append(AsyncQueue(1))

        self._process_args.append(
            (
                "aggregate",
                {
                    "input_queue": self.queues[-2],
                    "afunc": aggregate,
                    "output_queue": self.queues[-1],
                    "concurrency": 1,
                    "name": name,
                    "hooks": hooks,
                    "report_stats_interval": report_stats_interval,
                    "_pipe_eof": not drop_last,
                },
            )
        )
        return self

    def add_sink(
        self, sink: Queue[T], timeout: int | float | None = None
    ) -> "AsyncPipeline":
        """Attach a (synchronous) queue to the end of the pipeline.

        .. code-block::

           ┌─┐
           │ │
           │ │ AsyncQueue
           │ │
           └┬┘
            │
           ┌▼┐
           │ │
           │ │ Synchronous Queue
           │ │
           └─┘

        Args:
            Queue: Synchronous queue to pass the items.
            timeout: Timeout for the ``queue.put()`` operation in seconds.
                In case the output queue (synchronous) is full and the
                foreground thread is not consuming the queue, the pipeline will
                wait for this amount of time before giving up.
        """
        if self._sink is not None:
            raise ValueError("Sink already set.")

        self._sink = sink
        self._sink_timeout = timeout
        return self

    def __str__(self) -> str:
        parts = [repr(self)]
        parts.append(f"  - src: {self._source}")
        parts.append(f"    Buffer: buffer_size={self.queues[0].maxsize}")

        for queue, (type_, args) in zip(self.queues[1:], self._process_args):
            match type_:
                case "pipe":
                    part = f"{args['name']}(concurrency={args['concurrency']})"
                case "aggregate":
                    part = args["name"]
                case _:
                    part = type_
            parts.append(f"  - {part}")

            if type_ not in ["aggregate"]:
                parts.append(f"    Buffer: buffer_size={queue.maxsize}")

        if self._sink is not None:
            parts.append(f"  - sink(timeout={self._sink_timeout})")

        return "\n".join(parts)

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
        if self._source is None:
            raise ValueError("Source is not set.")

        # Build
        tasks = set()
        # Source
        tasks.add(
            create_task(
                _enqueue(self._source, self.queues[0], max_items=num_items),
                name="AsyncPipeline::0_source",
            )
        )

        # Rest
        for i, (type_, args) in enumerate(self._process_args, start=1):
            match type_:
                case "pipe":
                    coro = _pipe(**args)
                case "aggregate":
                    coro = _pipe(**args)
                case _:  # pragma: no cover
                    raise ValueError(f"Unexpected process type: {type_}")

            tasks.add(create_task(coro, name=f"AsyncPipeline::{i}_{args['name']}"))

        # sink
        if self._sink is not None:
            tasks.add(
                create_task(
                    _dequeue(self.queues[-1], self._sink, timeout=self._sink_timeout),
                    name=f"AsyncPipeline::{len(self._process_args) + 1}_sink",
                )
            )

        while tasks:
            # Note:
            # `asyncio.wait` does not automatically propagate the cancellation to its children.
            # For graceful shutdown, we manually cancel the child tasks.
            #
            # Also, it seems asyncio loop throws Cancellation on most outer task.
            # I am not sure where this behavior is documented, but here is an example script to
            # demonstrate the behavior.
            # https://gist.github.com/mthrok/3a1c11c2d8012e29f4835679ac0baaee
            try:
                done, tasks = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_EXCEPTION
                )
            except asyncio.CancelledError:
                for task in tasks:
                    task.cancel()
                await asyncio.wait(tasks)
                raise

            # 1. check what kind of errors occured
            # Note:
            # Byt the above assumption about the cancellation, the following code assumes that
            # CancelledError does not happen for done tasks.
            errs = {}
            for task in done:
                # Note:
                # `exception()` method can throw `CancelledError` or `InvalidStateError`,
                # both of which are assumed to not happen here.
                if (err := task.exception()) is not None:
                    errs[task.get_name()] = err

            # 2. if a failure presents, cancel the remaining tasks
            if errs:
                if tasks:
                    for task in tasks:
                        task.cancel()

                    done, _ = await asyncio.wait(tasks)

                    for task in done:
                        try:
                            err = task.exception()
                        except asyncio.CancelledError:
                            errs[task.get_name()] = "Cancelled"
                        else:
                            errs[task.get_name()] = err

                raise PipelineFailure(errs)
