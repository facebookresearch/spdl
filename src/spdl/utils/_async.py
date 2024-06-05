import asyncio
import functools
import logging
from asyncio import BoundedSemaphore, Queue as AsyncQueue, Task
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
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

_SENTINEL = object()
_TIMEOUT = 30


async def run_async(
    func: Callable[..., T],
    *args,
    _executor: ThreadPoolExecutor | None = None,
    **kwargs,
) -> Awaitable[T]:
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
    return await loop.run_in_executor(_executor, _func)  # pyre-ignore: [6]


################################################################################
# async_generate
################################################################################


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
    except TimeoutError:
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


async def _with_sem(coro, sem):
    async with sem:
        return await coro


# Wrapper around asyncio.create_task with semaphore
def _create_task(coro, name: str, sem: BoundedSemaphore | None = None):
    if sem is not None:
        coro = _with_sem(coro, sem)
    return asyncio.create_task(coro, name=name)


async def _queue_tasks(
    iterator: Iterator[T],
    afunc: Callable[[T], Awaitable[U]],
    queue: AsyncQueue[Task[U]],
    name: str,
    timeout: float | None,
) -> None:
    sem = None if queue.maxsize == 0 else BoundedSemaphore(queue.maxsize)

    # Assumption: `iterator` does not get stuck.
    for i, item in enumerate(iterator):
        task = _create_task(afunc(item), f"{name}_{i}", sem)
        await asyncio.wait_for(queue.put(task), timeout)

    _LG.debug("Done: %s", name)


async def async_generate(
    iterator: Iterator[T],
    afunc: Callable[[T], Awaitable[U]],
    queue: AsyncQueue[U],
    *,
    concurrency: int = 3,
    name: str | None = None,
    timeout: float | None = _TIMEOUT,
    sentinel=_SENTINEL,
) -> None:
    """Apply async function to synchronous iterator.

    This function iterates on a synchronous iterator, applies an async function, and
    puts the results into an async queue.

    ```
    ┌───────────┐
    │ Iterator  │
    └─────┬─────┘
          │
         ┌▼┐
         │ │
         │ │ AsyncQueue
         │ │
         └─┘
    ```

    Optionally, it applies concurrency when applying the async function.
    `concurrency` argument controls the number of coroutines that are scheduled
    concurrently. At most `concurrency` number of coroutines are scheduled
    at a time.

    !!! warning:

        This function assumes that user-provided `iterator` returns
        within a reasonable time. If the `iterator` gets stuck,
        this function also gets stuck.

    !!! note:

        `afunc` will be applied to multiple items concurrently, so it should
        refrain from modifying global states.

    Args:
        iterator: Iterator to apply the async function to.
        afunc: The async function to apply.
        queue: Output queue.
        concurrency: The maximum number of async tasks scheduled concurrently.
        name: The name to give to the task.
        timeout: The maximum time to wait for async operations to complete.
            Operations include one execution of the given `afunc`, and putting
            a result object or `sentinel` object into the queue.
            If `None`, then it blocks indefinetly.

            !!! note:

                `timeout` is intended to only escape from unexpected deadlock
                condition, and it does not garantee a graceful exit.
                When timeout happens, the sentinel might not be propagated
                properly, which leads to the situation where the downstream
                tasks will not receive sentinel.
                So make sure that the downstream tasks also have timeout.
    """
    if concurrency < 0:
        raise ValueError("`concurrency` value must be >= 0")

    name_: str = name or f"apply_async_{afunc.__name__}"

    q_intern = AsyncQueue(concurrency)

    queue_task = asyncio.create_task(
        _queue_tasks(iterator, afunc, q_intern, name_, timeout),
        name=name_,
    )
    queue_task.add_done_callback(lambda t: _check_exception(t, stacklevel=2))

    num_failures = 0

    async def _handle_task(task):
        task.add_done_callback(lambda t: _check_exception(t, stacklevel=2))
        await asyncio.wait([task])
        # Exception is checked and logged by the above callback, so we don't
        # do anything about it.
        if task.exception() is None:
            await asyncio.wait_for(queue.put(task.result()), timeout)
        else:
            nonlocal num_failures
            num_failures += 1

    try:
        while not (queue_task.done() and q_intern.empty()):
            # note: if timeout occurs here, then queue_task is not producing
            # a task. That suggests that user-provided iterator is stuck.
            task = await asyncio.wait_for(q_intern.get(), timeout)
            await _handle_task(task)
    finally:
        if num_failures > 0:
            _LG.warning("[%s] %s task(s) did not succeed.", name_, num_failures)
        # note: if timeout occurs here, then downstream is not consuming the
        # queue fast enough.
        await asyncio.wait_for(queue.put(sentinel), timeout)  # pyre-ignore: [6]

    # Let the error in `_queue_task`, (such as iterator failure) bubble up.
    # Otherwise, the error will be swallowed.
    await queue_task


################################################################################
# apply_sync
################################################################################


async def apply_async(
    afunc: Callable[[T], Awaitable[U]],
    iterator: Iterator[T],
    buffer_size: int = 10,
    concurrency: int = 3,
):
    queue = AsyncQueue(buffer_size)
    task = asyncio.create_task(
        async_generate(iterator, afunc, queue, concurrency=concurrency)
    )
    task.add_done_callback(lambda t: _check_exception(t, stacklevel=2))

    async for item in async_iterate(queue):
        yield item
    await task


################################################################################
# async_iterate
################################################################################
async def async_iterate(
    queue: AsyncQueue[T],
    *,
    timeout: float | None = _TIMEOUT,
    sentinel=_SENTINEL,
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
        queue: Asynchronous queue where the results of some task are put.
        timeout: The maximum time to wait for the queue. If `None`, then it
            blocks indefinetly.

    Yields:
        The result objects in the queue.
    """
    while (item := await asyncio.wait_for(queue.get(), timeout)) is not sentinel:
        yield item


################################################################################
# async_pipe
################################################################################
async def _pipe_queue_tasks(
    input_queue: AsyncQueue[T],
    afunc: Callable[[T], Awaitable[U]],
    q_intern: AsyncQueue[U],
    name: str,
    timeout: float | None,
    sentinel,
) -> None:
    sem = None if q_intern.maxsize == 0 else BoundedSemaphore(q_intern.maxsize)

    i = -1
    while (item := await asyncio.wait_for(input_queue.get(), timeout)) is not sentinel:
        i += 1
        task = _create_task(afunc(item), f"{name}_{i}", sem)
        await asyncio.wait_for(q_intern.put(task), timeout)

    _LG.debug("Done: %s", name)


async def async_pipe(
    input_queue: AsyncQueue[T],
    func: Callable[[T], Awaitable[U]],
    output_queue: AsyncQueue[U],
    *,
    concurrency: int = 1,
    name: str | None = None,
    timeout: float | None = _TIMEOUT,
    sentinel=_SENTINEL,
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
    if input_queue is output_queue:
        raise ValueError("input queue and output queue must be different")

    if concurrency < 0:
        raise ValueError("`concurrency` value must be >= 0")

    name_: str = name or f"pipe_{func.__name__}"

    q_intern = AsyncQueue(concurrency)

    queue_task = asyncio.create_task(
        _pipe_queue_tasks(input_queue, func, q_intern, name_, timeout, sentinel),
        name=name_,
    )
    queue_task.add_done_callback(lambda t: _check_exception(t, stacklevel=2))

    num_failures = 0

    async def _handle_task(task):
        task.add_done_callback(lambda t: _check_exception(t, stacklevel=2))
        await asyncio.wait([task])
        # Exception is checked and logged by the above callback, so we don't
        # do anything about it.
        if task.exception() is None:
            await asyncio.wait_for(output_queue.put(task.result()), timeout)
        else:
            nonlocal num_failures
            num_failures += 1

    try:
        while not (queue_task.done() and q_intern.empty()):
            # note: if timeout occurs here, then queue_task is not producing
            # a task. That suggests that user-provided iterator is stuck.
            task = await asyncio.wait_for(q_intern.get(), timeout)
            await _handle_task(task)
    finally:
        if num_failures > 0:
            _LG.warning("[%s] %s task(s) did not succeed.", name_, num_failures)
        await asyncio.wait_for(output_queue.put(sentinel), timeout)

    # Let the error in `_queue_task`, (such as iterator failure) bubble up.
    # Otherwise, the error will be swallowed.
    await queue_task
