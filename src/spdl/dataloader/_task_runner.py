import asyncio
import concurrent.futures
import logging
import warnings
from concurrent.futures import Future
from queue import Queue
from threading import BoundedSemaphore, Event, Thread
from typing import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Generic,
    Iterable,
    Iterator,
    TypeVar,
)

import spdl.utils

_LG = logging.getLogger(__name__)

__all__ = [
    "BackgroundGenerator",
    "apply_async",
    "apply_concurrent",
]

T = TypeVar("T")
U = TypeVar("U")


################################################################################
# Impl for AsyncTaskRunner
################################################################################
def _run_agen(aiterable, sentinel, queue, stopped):
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

    asyncio.set_event_loop(asyncio.new_event_loop())
    asyncio.run(_generator_loop())


def _run_gen(generator, sentinel, queue, stopped):
    try:
        for item in generator:
            queue.put(item)

            if stopped.is_set():
                _LG.debug("Stop requested.")
                generator.close()
                break
            _LG.debug("exiting _generator_loop.")
    finally:
        queue.put(sentinel)
        stopped.set()


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


class BackgroundGenerator(Generic[T]):
    """Run generator in background and iterate the items.

    Args:
        iterable: Generator to run in the background. It can be an
            asynchronous generator or a regular generator.

        queue_size: The size of the queue that is used to pass the
            generated items from the background thread to the main thread.
            If the queue is full, the background thread will be blocked.

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
        iterable: AsyncIterable[T] | Iterable[T],
        queue_size: int = 10,
    ):
        self.iterable = iterable
        self.queue_size = queue_size

    def __iter__(self) -> Iterator[T]:
        queue = Queue(maxsize=self.queue_size)
        # Used to indicate the end of the queue.
        sentinel = object()
        # Used to flag cancellation from outside and to
        # flag the completion from the inside.
        stopped = Event()
        thread = Thread(
            target=(_run_agen if hasattr(self.iterable, "__aiter__") else _run_gen),
            args=(self.iterable, sentinel, queue, stopped),
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
# Impl for apply_concurrent
################################################################################
def _apply_concurrent(generator, max_concurrency=10):
    semaphore = BoundedSemaphore(max_concurrency)

    def _cb(_):
        semaphore.release()

    futs = []
    try:
        for future in generator:
            semaphore.acquire()
            future.add_done_callback(_cb)
            futs.append(future)

            while len(futs) > 0 and futs[0].done():
                yield futs.pop(0)

        yield from futs
    except GeneratorExit:
        _LG.info("Generator exited - waiting for the ongoing futures to complete.")
        spdl.utils.wait_futures(futs).result()
        raise


def apply_concurrent(
    func: Callable[[T], Future[U]],
    generator: Iterable[T],
    max_concurrency: int = 10,
    timeout: float = 300,
) -> Iterator[U]:
    """Apply concurrent function to generator sequence.

    Args:
        func: A (non-async) function that takes some input and returns a
            `Future`.

        generator: An object generates series of inputs to the given function.

        max_concurrency: Controls how many futures should be in running state
            concurrently. The function will first initialize this number of
            Futures, then initialize more as the previous Futures complete.

        timeout: The maximum time to wait for each Future object to complete.
            (Unit: second)

    Yields:
        The output of the `func`.
    """

    def gen():
        for item in generator:
            yield func(item)

    for fut in _apply_concurrent(gen(), max_concurrency):
        try:
            yield fut.result(timeout)
        except concurrent.futures.CancelledError:
            _LG.warning("Future [%s] was cancelled.", fut)
        except Exception as err:
            _LG.error("Future [%s] failed: %s", fut, err)


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
    semaphore = asyncio.BoundedSemaphore(max_concurrency)

    async def _f(item):
        async with semaphore:
            result = await async_func(item)
            await queue.put(result)

    tasks = set()
    for i, item in enumerate(generator):
        task = asyncio.create_task(_f(item), name=f"item_{i}")
        task.add_done_callback(lambda t: _check_exception(t, stacklevel=2))
        tasks.add(task)

        # Occasionally remove the done tasks.
        if len(tasks) >= 3 * max_concurrency:
            _, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    if tasks:
        await asyncio.wait(tasks)

    _LG.debug("_apply_async - done")
    await queue.put(sentinel)


async def apply_async(
    func: Callable[[T], Awaitable[U]],
    generator: Iterable[T],
    max_concurrency: int = 10,
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
        max_concurrency: The maximum number of concurrent async tasks.
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
    queue = asyncio.Queue()

    # We use sentinel to detect the end of the background job
    sentinel = object()

    coro = _apply_async(func, generator, queue, sentinel, max_concurrency)
    task = asyncio.create_task(coro, name="_apply_async")
    task.add_done_callback(_check_exception)

    while (item := await asyncio.wait_for(queue.get(), timeout)) is not sentinel:
        yield item

    await task
