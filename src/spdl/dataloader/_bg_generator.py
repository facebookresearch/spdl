import asyncio
import logging
import warnings
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
)
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Event, Thread
from typing import Generic, TypeVar

_LG = logging.getLogger(__name__)

__all__ = [
    "BackgroundGenerator",
    "apply_async",
]

T = TypeVar("T")
U = TypeVar("U")


################################################################################
# Impl for AsyncTaskRunner
################################################################################
def _run_agen(loop, aiterable, sentinel, queue, stopped):
    async def _generator_loop():
        try:
            async for item in aiterable:
                queue.put(item)

                if stopped.is_set():
                    _LG.debug("Stop requested.")
                    break
        finally:
            queue.put(sentinel)
            stopped.set()
        _LG.debug("exiting _generator_loop.")

    loop.run_until_complete(_generator_loop())


class _thread_manager:
    """Context manager to stop and join the background thread."""

    def __init__(self, thread: Thread, stopped: Event, queue: Queue, sentinel: object):
        self.thread = thread
        self.stopped = stopped
        self.queue = queue
        self.sentinel = sentinel

    def __enter__(self):
        self.thread.start()

    def __exit__(self, exc_type, exc_value, traceback):
        # If stopped is not set, the background thread is still running,
        # but queue is no longer consumed.
        # The queue might get clogged which can block the background thread and
        # prevent it from joining. Therefore, we need to flush the queue.
        #
        # If stopped is set, the background thread is completed.
        # No more items are generated, and the sentinel value might have been consumed.
        # So we cannot/don't flush the queue.
        if not self.stopped.is_set():
            _LG.info("Stopping the background thread.")
            self.stopped.set()
            self._flush()

        _LG.info("Waiting for the background thread to join.")
        self.thread.join()

    def _flush(self):
        _LG.debug("Flushing the queue.")
        while (_ := self.queue.get()) is not self.sentinel:
            pass


def _get_loop(num_workers: int | None):
    loop = asyncio.new_event_loop()
    loop.set_default_executor(
        ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="SPDL_BackgroundGenerator",
        )
    )
    return loop


class BackgroundGenerator(Generic[T]):
    """Run generator in background and iterate the items.

    Args:
        iterable: Generator to run in the background. It can be an
            asynchronous generator or a regular generator.

        num_workers: The number of worker threads in the default thread executor.
            If `loop` is provided, this argument is ignored.

        queue_size: The size of the queue that is used to pass the
            generated items from the background thread to the main thread.
            If the queue is full, the background thread will be blocked.

        loop: If provided, use this event loop to execute the generator.
            Otherwise, a new event loop will be created. When providing a loop,
            `num_workers` is ignored, so the executor must be configured by
            client code.

    ??? note "Example"

        ```python
        async def generator():
            for i in range(10):
                yield i

        processor = BackgroundGenerator(generator())
        for item in processor:
            # Do something with the item.
        ```
    """

    def __init__(
        self,
        iterable: AsyncIterable[T],
        *,
        num_workers: int | None = 3,
        queue_size: int = 10,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.iterable = iterable
        self.queue_size = queue_size
        self.loop = _get_loop(num_workers) if loop is None else loop

    def __iter__(self) -> Iterator[T]:
        queue = Queue(maxsize=self.queue_size)
        # Used to indicate the end of the queue.
        sentinel = object()
        # Used to flag cancellation from outside and to
        # flag the completion from the inside.
        stopped = Event()

        thread = Thread(
            target=_run_agen,
            args=(self.loop, self.iterable, sentinel, queue, stopped),
        )

        with _thread_manager(thread, stopped, queue, sentinel):
            while (item := queue.get()) is not sentinel:
                yield item


class BackgroundTaskProcessor(BackgroundGenerator):
    """Deprecated. Use BackgroundGenerator instead."""

    def __init__(self, *args, **kwargs):
        message = "BackgroundTaskProcessor has been renamed to BackgroundGenerator."
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)

    def __enter__(self):
        message = (
            "Background thread management has been moved to __iter__. "
            f"You can now directly iterate on {type(self).__name__}, and "
            "get rid of the context manager. "
        )
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


################################################################################
# Impl for apply_async
################################################################################
def _check_exception(task, stacklevel=1):
    try:
        task.result()
    except asyncio.exceptions.CancelledError:
        _LG.warning("Task [%s] was cancelled.", task.get_name(), stacklevel=stacklevel)
    except Exception as err:
        _LG.error("Task [%s] failed: %s", task.get_name(), err, stacklevel=stacklevel)


async def _apply_async(async_func, generator, queue, sentinel, max_concurrency):
    async def _f(item):
        result = await async_func(item)
        await queue.put(result)

    sem = asyncio.BoundedSemaphore(max_concurrency)
    tasks = set()
    for i, item in enumerate(generator):
        await sem.acquire()

        task = asyncio.create_task(_f(item), name=f"item_{i}")
        tasks.add(task)
        task.add_done_callback(lambda t: _check_exception(t, stacklevel=2))
        task.add_done_callback(lambda _: sem.release())
        task.add_done_callback(tasks.discard)

    while tasks:
        await asyncio.sleep(0.1)

    _LG.debug("_apply_async - done")
    await queue.put(sentinel)


async def apply_async(
    func: Callable[[T], Awaitable[U]],
    generator: Iterable[T],
    *,
    buffer_size: int = 10,
    max_concurrency: int = 3,
    timeout: float = 300,
) -> AsyncIterator[U]:
    """Apply async function to the non-async generator.

    This function iterates the items in the generator, and apply async function,
    buffer the coroutines so that at any time, there are `max_concurrency`
    coroutines running. Each coroutines put the resulting items to the internal
    queue as soon as it's ready.

    !!! note

        The order of the output may not be the same as generator.

    Args:
        func: The async function to apply.
        generator: The generator to apply the async function to.
        buffer_size: The size of the internal queue.
            If it's full, the generator will be blocked.
        max_concurrency: The maximum number of async tasks scheduled concurrently.
        timeout: The maximum time to wait for the async function. (Unit: second)

    Yields:
        The output of the `func`.
    """
    # Implementation Note:
    #
    # The simplest form to apply the async function to the generator is
    #
    # ```
    # for item in generator:
    #     yield await func(item)
    # ```
    #
    # But this applies the function sequentially, so it is not efficient.
    #
    # We need to run multiple coroutines in parallel, and fetch the results.
    # But since the size of the generator is not know, and it can be as large
    # as millions, we need to set the max buffer size.
    #
    # A common approach is to put tasks in `set` and use `asyncio.wait`.
    # But this approach leaves some ambiguity as "when to wait"?
    #
    # ```
    # tasks = set()
    # for item in generator:
    #     task = asyncio.create_task(func(item))
    #     tasks.add(task)
    #
    #     if <SOME_OCCASION>:  # When is optimal?
    #         done, tasks = await asyncio.wait(
    #             tasks, return_when=asyncio.FIRST_COMPLETED)
    #         for task in done:
    #             yield task.result()
    # ```
    #
    # This kind of parameter has non-negligible impact on the performance, and
    # usually the best performance is achieved when there is no such parameter,
    # so that the async loop does the job for us.
    #
    # To eliminate such parameter, we use asyncio.Queue object and pass the results
    # into this queue, as soon as it's ready. We only await the `Queue.get()`. Then,
    # yield the results fetched from the queue.
    queue = asyncio.Queue(buffer_size)

    # We use sentinel to detect the end of the background job
    sentinel = object()

    coro = _apply_async(func, generator, queue, sentinel, max_concurrency)
    task = asyncio.create_task(coro, name="_apply_async")
    task.add_done_callback(_check_exception)

    while (item := await asyncio.wait_for(queue.get(), timeout)) is not sentinel:
        yield item

    await task
