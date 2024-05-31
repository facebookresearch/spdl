import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

__all__ = [
    "apply_async",
]

_LG = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


def _iter_file(path, prefix):
    with open(path, "r") as f:
        for line in f:
            if path := line.strip():
                if prefix:
                    path = prefix + path
                yield path


def _iter_sample_every_n(gen, offset=0, every_n=1, max=None):
    offset = offset % every_n

    num = 0
    for i, item in enumerate(gen):
        if i % every_n == offset:
            yield item
            num += 1

            if max is not None and num >= max:
                return


def _iter_batch(gen, batch_size, drop_last=False):
    batch = []
    for item in gen:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch and not drop_last:
        yield batch


def _iter_flist(
    path: str | Path,
    *,
    prefix: str | None = None,
    batch_size: int = 1,
    n: int = 0,
    N: int = 1,
    max: int | None = None,
    drop_last: bool = False,
):
    gen = _iter_batch(
        _iter_sample_every_n(_iter_file(path, prefix), n, N, max),
        batch_size,
        drop_last=drop_last,
    )
    try:
        yield from gen
    except Exception:
        # Because this utility is intended to be used in background thread,
        # we supress the error and exit
        _LG.exception("Error while iterating over flist %s", path)
        return


def _get_loop(num_workers: int | None):
    loop = asyncio.new_event_loop()
    loop.set_default_executor(
        ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="SPDL_BackgroundGenerator",
        )
    )
    return loop


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
