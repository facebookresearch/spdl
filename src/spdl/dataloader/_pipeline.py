# pyre-unsafe

import asyncio
import logging
from asyncio import Queue as AsyncQueue
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from contextlib import asynccontextmanager
from queue import Queue
from typing import TypeVar


__all__ = [
    "AsyncPipeline",
    "PipelineFailure",
    "EOF_SENTINEL",
]

_LG = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


# Custom class just for the sake of Sphinx doc
# Functionally, just `object()` is sufficient.
class Sentinel:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


EOF_SENTINEL = Sentinel("EOF")
EOF_SENTINEL.__doc__ = (
    "The default sentinel object used to "
    "mark/detect the end of stream in async pipelines."
)


# Note:
# This function is intentionally made in a way it cannot be directly attached to
# task callback. Instead it should be wrapped in lambda as follow, even though
# there are some opposition on assigning lambda:
#
# `task.add_done_callback(lambda t: _check_exception(t, stacklevel=2))`
#
# This way, the filename and line number of the log points to the location where
# task was created.
# Otherwise the log will point to the location somewhere deep in `asyncio` module
# which is not very helpful.
def _check_exception(task, stacklevel):
    try:
        task.result()
    except asyncio.exceptions.CancelledError:
        _LG.warning("Task [%s] was cancelled.", task.get_name(), stacklevel=stacklevel)
    except (TimeoutError, asyncio.exceptions.TimeoutError):
        # Timeout does not contain any message
        _LG.error("Task [%s] timeout.", task.get_name(), stacklevel=stacklevel)
    except Exception as err:
        _LG.error(
            "Task [%s] failed: %s %s",
            task.get_name(),
            type(err).__name__,
            err,
            stacklevel=stacklevel,
        )


@asynccontextmanager
async def _put_sentinel_when_done(queue, put_on_error=False):
    try:
        yield
    except Exception:
        if put_on_error:
            await queue.put(EOF_SENTINEL)
        else:
            raise
    else:
        await queue.put(EOF_SENTINEL)


################################################################################
# _pipe
################################################################################


async def _pipe(
    input_queue: AsyncQueue[T],
    afunc: Callable[[T], Awaitable[U]],
    output_queue: AsyncQueue[U],
    concurrency: int = 1,
    name: str | None = None,
) -> None:
    if input_queue is output_queue:
        raise ValueError("input queue and output queue must be different")

    if concurrency < 1:
        raise ValueError("`concurrency` value must be >= 1")

    name_: str = name or f"pipe_{afunc.__name__}"

    num_task_failures = 0

    async def _wrap(coro: Awaitable[U]) -> None:
        try:
            result = await coro
            await output_queue.put(result)
        except Exception:
            nonlocal num_task_failures
            num_task_failures += 1
            raise

    tasks = set()

    # Note: the order of the contextmanager matters.
    # `_put_sentinel_when_done` must be placed first (its finally block is executed last)
    async with _put_sentinel_when_done(output_queue):
        i = 0
        while (item := await input_queue.get()) is not EOF_SENTINEL:
            i += 1
            # note: Make sure that `afunc` is called directly in this function,
            # so as to detect user error. (incompatible `afunc` and `iterator` combo)
            coro = afunc(item)
            task = asyncio.create_task(_wrap(coro), name=f"{name_}_{i}")
            task.add_done_callback(lambda t: _check_exception(t, stacklevel=2))
            tasks.add(task)

            if len(tasks) >= concurrency:
                done, tasks = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

        if tasks:
            await asyncio.wait(tasks)

        if num_task_failures > 0:
            _LG.warning(
                "[%s] %s task(s) did not succeed.",
                name_,
                num_task_failures,
            )


################################################################################
# _enqueue
################################################################################


async def _enqueue(
    iterator: Iterator[T],
    queue: AsyncQueue[T | Sentinel],
) -> None:
    async with _put_sentinel_when_done(queue, put_on_error=True):
        for item in iterator:
            await queue.put(item)


################################################################################
# _enqueue
################################################################################


async def _dequeue(
    input_queue: AsyncQueue[T | Sentinel],
    output_queue: Queue[T],
):
    while (item := await input_queue.get()) is not EOF_SENTINEL:
        output_queue.put(item)  # pyre-ignore: [6]


################################################################################
# AsyncPipeline
################################################################################


# TODO [Python 3.11]: Migrate to ExceptionGroup
class PipelineFailure(Exception):
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
               AsyncPipeline(buffer_size=10)
               .add_source(source())
               .pipe(decode, concurrency=10)
               .add_sink(queue)

           asyncio.run(pipeline.run())


    Args:
        buffer_size: The size of the first queue to which data are
            pushed from the source iterator.
    """

    def __init__(self, buffer_size: int = 10):
        self.queues = [AsyncQueue(buffer_size)]
        self.coros: list[tuple[Coroutine, str | None]] = []

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
        coro = _enqueue(source, self.queues[0])
        self.coros.append((coro, "source"))
        return self

    def pipe(
        self,
        afunc: Callable[[T], Awaitable[U]],
        concurrency: int = 1,
        name: str | None = None,
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
        """
        self.queues.append(AsyncQueue(1))
        in_queue, out_queue = self.queues[-2:]
        coro = _pipe(in_queue, afunc, out_queue, concurrency, name)
        self.coros.append((coro, afunc.__name__))
        return self

    def add_sink(self, sink: Queue[T]) -> "AsyncPipeline":
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
        """
        coro = _dequeue(self.queues[-1], sink)
        self.coros.append((coro, "sink"))
        return self

    # TODO [Python 3.11]: Try TaskGroup
    async def run(self) -> None:
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

        Raises:

            PipelineFailure: Rised when a part of the pipeline has an error.
        """
        tasks = set()
        for i, (coro, name) in enumerate(self.coros):
            task = asyncio.create_task(coro, name=f"AsyncPipeline::{i}_{name}")
            task.add_done_callback(lambda t: _check_exception(t, stacklevel=2))
            tasks.add(task)

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
