import asyncio
import functools
import logging
from asyncio import BoundedSemaphore, Future as AsyncFuture, Queue as AsyncQueue
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

__all__ = [
    "apply_async",
    "async_generate",
    "async_iterate",
    "async_pipe",
    "run_async",
]

_LG = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")

_Sentinel = object()


def run_async(
    func: Callable[..., T],
    *args,
    _executor: ThreadPoolExecutor | None = None,
    **kwargs,
) -> AsyncFuture[T]:
    """Run the given synchronous function asynchronously (in a thread).

    !!! note

        To achieve the true concurrency, the function must be thread-safe and must
        release the GIL.

    Args:
        func: The function to run.
        args: Positional arguments to the `func`.
        _executor: Custom executor.
            If `None` the default executor of the current event loop is used.
        kwargs: Keyword arguments to the `func`.
    """
    loop = asyncio.get_running_loop()
    _func = functools.partial(func, *args, **kwargs)
    return loop.run_in_executor(_executor, _func)  # pyre-ignore: [6]


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


async def _generate(
    func: Callable[[T], Awaitable[U]],
    generator: Iterable[T],
    queue: AsyncQueue[U],
    concurrency: int,
    name: str,
) -> None:
    sem = BoundedSemaphore(concurrency)

    async def _f(item):
        async with sem:
            await queue.put(await func(item))

    tasks = set()

    for i, item in enumerate(generator):
        async with sem:
            task = asyncio.create_task(_f(item), name=f"{name}_{i}")
            tasks.add(task)
            task.add_done_callback(lambda t: _check_exception(t, stacklevel=2))
            task.add_done_callback(tasks.discard)

    while tasks:
        await asyncio.sleep(0.1)

    _LG.debug("Done: %s", name)


async def async_generate(
    func: Callable[[T], Awaitable[U]],
    generator: Iterable[T],
    queue: AsyncQueue[U],
    *,
    concurrency: int = 3,
    name: str | None = None,
    _sentinel=_Sentinel,
) -> None:
    """Apply async function to the non-async generator.

    This function iterates on a (synchronous) generator, applies an async function, and
    put the results into an async queue.

    ```
    ┌───────────┐
    │ Generator │
    └─────┬─────┘
          │
         ┌▼┐
         │ │
         │ │ AsyncQueue
         │ │
         └─┘
    ```

    `concurrency` controls the number of coroutines that are scheduled
    concurrently. At most `concurrency` number of coroutines are scheduled
    at any time. Each coroutine will put the result into the queue independently.
    As a result, the order of the output may not be the same as generator.

    Args:
        func: The async function to apply.
        generator: The generator to apply the async function to.
        queue: Output queue.
        concurrency: The maximum number of async tasks scheduled concurrently.
        name: The name to give to the task.
    """
    name_: str = name or f"apply_async_{func.__name__}"

    task = asyncio.create_task(
        _generate(func, generator, queue, concurrency, name_),
        name=name_,
    )
    task.add_done_callback(_check_exception)

    try:
        await task
    finally:
        await queue.put(_sentinel)  # pyre-ignore: [6]


################################################################################
# apply_sync
################################################################################


async def apply_async(
    func: Callable[[T], Awaitable[U]],
    generator: Iterable[T],
    buffer_size: int = 10,
    concurrency: int = 3,
):
    queue = AsyncQueue(buffer_size)
    task = asyncio.create_task(
        async_generate(func, generator, queue, concurrency=concurrency)
    )
    task.add_done_callback(_check_exception)

    async for item in async_iterate(queue):
        yield item
    await task


################################################################################
# async_iterate
################################################################################
async def async_iterate(
    queue: AsyncQueue[T],
    *,
    _sentinel=_Sentinel,
) -> AsyncIterator[T]:
    """Iterate over the given queue.

    ```
           ┌─┐
           │ │
           │ │ AsyncQueue
           │ │
           └┬┘
            │
    ┌───────▼────────┐
    │ Async Iterator │
    └────────────────┘
    ```

    Args:
        queue: Asynchronous queue where the result of tasks are placed.

    Returns:
        Async iterator over the result objects.
    """
    while (item := await queue.get()) is not _sentinel:
        yield item


################################################################################
# Impl for pipe
################################################################################
async def _pipe(
    func: Callable[[T], Awaitable[U]],
    input_queue: AsyncQueue[T],
    output_queue: AsyncQueue[U],
    _sentinel,
    concurrency: int,
    name: str,
) -> None:
    sem = BoundedSemaphore(concurrency)

    async def _f(item: T):
        async with sem:
            await output_queue.put(await func(item))

    tasks = set()

    i = -1
    while (item := await input_queue.get()) is not _sentinel:
        i += 1
        async with sem:
            task = asyncio.create_task(_f(item), name=f"{name}_{i}")  # pyre-ignore: [6]
            tasks.add(task)
            task.add_done_callback(lambda t: _check_exception(t, stacklevel=2))
            task.add_done_callback(lambda t: tasks.discard)

    while tasks:
        await asyncio.sleep(0.1)

    _LG.debug("Done: %s", name)


async def async_pipe(
    func: Callable[[T], Awaitable[U]],
    input_queue: AsyncQueue[T],
    output_queue: AsyncQueue[U],
    *,
    concurrency: int = 1,
    name: str | None = None,
    _sentinel=_Sentinel,
) -> None:
    """Apply an async function to the outputs of the input queue and put the results to the output queue.

    ```
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
    ```

    Args:
        func: Async function that to be applied to the items in the input queue.
        input_queue: Input queue.
        output_queue: Output queue.
        concurrency: The maximum number of async tasks scheduled concurrently.
        name: The name to give to the task
    """
    name_: str = name or f"pipe_{func.__name__}"

    task = asyncio.create_task(
        _pipe(func, input_queue, output_queue, _sentinel, concurrency, name_),
        name=name_,
    )
    task.add_done_callback(_check_exception)

    try:
        await task
    finally:
        await output_queue.put(_sentinel)
