import asyncio
import functools
import logging
from asyncio import BoundedSemaphore, Queue as AsyncQueue, wait_for
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
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


async def _buffered_queue(
    queue: AsyncQueue[T],
    concurrency: int,
    name: str,
    timeout: float | None,
) -> AsyncGenerator[None, Awaitable[T]]:
    num_task_failures = num_timeout = 0

    async def _count_failure(coro: Awaitable[T]) -> T:
        try:
            return await asyncio.wait_for(coro, timeout)
        except (TimeoutError, asyncio.exceptions.TimeoutError):
            nonlocal num_timeout
            num_timeout += 1
            raise
        except Exception:
            nonlocal num_task_failures
            num_task_failures += 1
            raise

    sem = BoundedSemaphore(concurrency)

    async def _wrap(coro: Awaitable[T]):
        async with sem:
            result = await _count_failure(coro)
            await asyncio.wait_for(queue.put(result), timeout)

    tasks = set()
    i = -1
    # In case you are not familiar with Generator expression with `send` pattern, see
    # https://docs.python.org/3.12/reference/expressions.html#generator-iterator-methods
    # The `while` loop will be `break`-ed when `close` method on the resulting generator
    # object is called. (by Python throwing `GeneratorExit` exception at `yield` point.)
    try:
        while True:
            i += 1
            if num_timeout > 0:
                break
            coro = yield
            async with sem:
                task = asyncio.create_task(_wrap(coro), name=f"{name}_{i}")
                task.add_done_callback(lambda t: _check_exception(t, stacklevel=2))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
    finally:
        _LG.debug("[%s] Waiting for the tasks to complete.", name)
        while tasks:
            await asyncio.sleep(0.1)
        if (total_failure := num_task_failures + num_timeout) > 0:
            _LG.warning(
                "[%s] %s task(s) did not succeed. (Failure: %s, Timeout: %s)",
                name,
                total_failure,
                num_task_failures,
                num_timeout,
            )
        if num_timeout > 0:
            raise asyncio.exceptions.TimeoutError()


@asynccontextmanager
async def _put_sentinel_when_done(queue, sentinel, timeout):
    try:
        yield
    finally:
        # note: if timeout occurs here, then downstream is not consuming the
        # queue fast enough.
        await wait_for(queue.put(sentinel), timeout=timeout)


@asynccontextmanager
async def _agen(agen):
    await anext(agen)  # Init

    try:
        yield agen
    finally:
        await agen.aclose()


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
    if concurrency < 1:
        raise ValueError("`concurrency` value must be >= 1")

    name_: str = name or f"apply_async_{afunc.__name__}"

    # Note: the order of the contextmanager matters.
    # `_put_sentinel_when_done` must be placed first (its finally block is executed last)
    async with (
        _put_sentinel_when_done(queue, sentinel, timeout),
        _agen(_buffered_queue(queue, concurrency, name_, timeout)) as queuing_task,
    ):
        for item in iterator:
            # note: Make sure that `afunc` is called directly in this function,
            # so as to detect user error. (incompatible `afunc` and `iterator` combo)
            await queuing_task.asend(afunc(item))


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
    while (item := await wait_for(queue.get(), timeout)) is not sentinel:
        yield item


################################################################################
# async_pipe
################################################################################


async def async_pipe(
    input_queue: AsyncQueue[T],
    afunc: Callable[[T], Awaitable[U]],
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

    if concurrency < 1:
        raise ValueError("`concurrency` value must be >= 1")

    name_: str = name or f"pipe_{afunc.__name__}"

    # Note: the order of the contextmanager matters.
    # `_put_sentinel_when_done` must be placed first (its finally block is executed last)
    async with (
        _put_sentinel_when_done(output_queue, sentinel, timeout),
        _agen(
            _buffered_queue(output_queue, concurrency, name_, timeout)
        ) as queuing_task,
    ):
        while (item := await wait_for(input_queue.get(), timeout)) is not sentinel:
            # note: Make sure that `afunc` is called directly in this function,
            # so as to detect user error. (incompatible `afunc` and `iterator` combo)
            await queuing_task.asend(afunc(item))
