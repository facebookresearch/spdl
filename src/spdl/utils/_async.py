import asyncio
import functools
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import TypeVar

__all__ = [
    "apply_async",
    "pipe",
    "run_async",
]

_LG = logging.getLogger(__name__)

S = TypeVar("S")
T = TypeVar("T")
U = TypeVar("U")


def run_async(
    func: Callable[..., T], *args, executor=None, **kwargs
) -> asyncio.Future[T]:
    """Run the given function in the thread pool executor of the current event loop."""
    loop = asyncio.get_running_loop()
    _func = functools.partial(func, *args, **kwargs)
    return loop.run_in_executor(executor, _func)  # pyre-ignore: [6]


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


async def _apply_async(func, generator, queue, concurrency) -> None:
    sem = asyncio.BoundedSemaphore(concurrency)

    async def _f(item):
        async with sem:
            result = await func(item)
            await queue.put(result)

    tasks = set()

    for i, item in enumerate(generator):
        async with sem:
            task = asyncio.create_task(_f(item), name=f"item_{i}")
            tasks.add(task)
            task.add_done_callback(lambda t: _check_exception(t, stacklevel=2))
            task.add_done_callback(tasks.discard)

    while tasks:
        await asyncio.sleep(0.1)

    _LG.debug("_apply_async - done")


async def apply_async(
    func: Callable[[T], Awaitable[U]],
    generator: Iterable[T],
    *,
    buffer_size: int = 10,
    concurrency: int = 3,
) -> AsyncIterator[U]:
    """Apply async function to the non-async generator.

    This function iterates the items in the generator, and apply async function,
    buffer the coroutines so that at any time, there are `concurrency`
    coroutines running. Each coroutines put the resulting items to the internal
    queue as soon as it's ready.

    !!! note

        The order of the output may not be the same as generator.

    Args:
        func: The async function to apply.
        generator: The generator to apply the async function to.
        buffer_size: The size of the internal queue.
            If it's full, the generator will be blocked.
        concurrency: The maximum number of async tasks scheduled concurrently.

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

    async def _apply():
        try:
            await _apply_async(func, generator, queue, concurrency)
        finally:
            await queue.put(sentinel)  # Signal the end of the generator

    task = asyncio.create_task(_apply(), name="_apply_async")
    task.add_done_callback(_check_exception)

    while (item := await queue.get()) is not sentinel:
        yield item

    await task


################################################################################
# Impl for pipe
################################################################################
async def _pipe(
    func: Callable[[T], Awaitable[U]],
    input_queue: asyncio.Queue[T | S],
    output_queue: asyncio.Queue[U | S],
    sentinel: S,
    concurrency: int,
    name: str,
) -> None:
    sem = asyncio.BoundedSemaphore(concurrency)

    async def _f(item: T):
        async with sem:
            result = await func(item)
            await output_queue.put(result)

    tasks = set()

    i = 0
    while (item := await input_queue.get()) is not sentinel:
        async with sem:
            task = asyncio.create_task(_f(item), name=f"{name}_{i}")  # pyre-ignore: [6]
            tasks.add(task)
            task.add_done_callback(lambda t: _check_exception(t, stacklevel=2))
            task.add_done_callback(lambda t: tasks.discard)
        i += 1

    while tasks:
        await asyncio.sleep(0.1)


async def pipe(
    func: Callable[[T], Awaitable[U]],
    input_queue: asyncio.Queue[T | S],
    output_queue: asyncio.Queue[U | S],
    sentinel: S,
    *,
    concurrency: int = 1,
    name: str | None = None,
) -> None:
    name = name or func.__name__
    task = asyncio.create_task(
        _pipe(func, input_queue, output_queue, sentinel, concurrency, name),
        name=f"pipe_{name}",
    )
    task.add_done_callback(_check_exception)

    try:
        await task
    finally:
        await output_queue.put(sentinel)
