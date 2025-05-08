# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "_pipe",
    "_ordered_pipe",
    "_PipeArgs",
    "_Aggregate",
    "_disaggregate",
]

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from concurrent.futures import Executor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Generic, TypeVar

from .._convert import Callables, convert_to_async
from .._hook import _stage_hooks, _task_hooks, TaskHook
from .._queue import AsyncQueue
from .._utils import create_task
from ._common import _EOF, _queue_stage_hook

# pyre-strict


T = TypeVar("T")
U = TypeVar("U")


_SKIP: None = None


@dataclass
class _PipeArgs(Generic[T, U]):
    op: Callables[T, U]
    executor: Executor | None = None
    concurrency: int = 1
    op_requires_eof: bool = False
    # Used to pass EOF to op.
    # Usually pipe does not pas EOF to op. This is because op is expected to be
    #  stateless, and requiring users to handle EOF is cumbersome, and there is
    # no real benefit.
    # However, some ops are exception. The aggregation (with drop_last=False)
    # requires to benotified when the pipeline reached the EOF, so that it can
    # flush the buffered items.

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError(
                f"`concurrency` value must be >= 1. Found: {self.concurrency}"
            )


class _FailCounter(TaskHook):
    num_failures: int = 0

    @classmethod
    def _increment(cls) -> None:
        cls.num_failures += 1

    @asynccontextmanager
    async def task_hook(self) -> AsyncIterator[None]:
        try:
            yield
        except Exception:
            self._increment()
            raise


async def _wrap_afunc(
    coro: Awaitable[U], hooks: list[TaskHook], queue: AsyncQueue[U]
) -> None:
    async with _task_hooks(hooks):
        result = await coro

    if result is _SKIP:
        return

    await queue.put(result)


async def _wrap_agen(
    coro: AsyncIterator[U], hooks: list[TaskHook], queue: AsyncQueue[U]
) -> None:
    exhausted = False
    while not exhausted:
        # NOTE:
        # Nested `except StopAsyncIteration` would look strange.
        # The following explains why.
        #
        # We want to give hooks opportunity to react to StopAsyncIteration,
        # for example, so that StatsHook will note record the task stats
        # for StopAsyncIteration case.
        #
        # When users implement hook, they might mistakenly absorb the
        # StopAsyncIteration exception by blanket `except Exception`,
        # and in this case, the StopAsyncIteration won't propagate to
        # the outside of `_task_hooks`.
        # When that happens, the control flow cannot exit the while loop.
        #
        # So when `StopAsyncIteration` is raised, we catch it once to set
        # the exhausted flag to True, then re-raise the exception so as
        # to give hooks chance to react to it.
        # If the hooks do not absorb the StopAsyncIteration, and
        # it propagates them, then we catch it and exit.
        try:
            async with _task_hooks(hooks):
                try:
                    result = await anext(coro)
                except StopAsyncIteration:
                    exhausted = True
                    raise

            # If task_hooks absorb the `StopAsyncIteration`, we need to exit here.
            if exhausted:
                return

            if result is _SKIP:
                continue

            await queue.put(result)
        except StopAsyncIteration:
            return


def _pipe(
    name: str,
    input_queue: AsyncQueue[T],
    output_queue: AsyncQueue[U],
    args: _PipeArgs[T, U],
    fail_counter: _FailCounter,
    task_hooks: list[TaskHook],
    max_failures: int = -1,
) -> Coroutine:
    if input_queue is output_queue:
        raise ValueError("input queue and output queue must be different")

    afunc: Callable[[T], Awaitable[U]] = (  # pyre-ignore: [9]
        convert_to_async(args.op, args.executor)
    )

    hooks: list[TaskHook] = [*task_hooks, fail_counter]

    if inspect.iscoroutinefunction(afunc):
        _wrap: Callable[[Awaitable[U]], Coroutine] = partial(
            _wrap_afunc, hooks=hooks, queue=output_queue
        )

    elif inspect.isasyncgenfunction(afunc):
        _wrap: Callable[[AsyncIterator[U]], Coroutine] = partial(
            _wrap_agen, hooks=hooks, queue=output_queue
        )

    else:
        raise ValueError(f"{afunc=} must be either async function or async generator.")

    def _too_many_failures() -> bool:
        if max_failures < 0:
            return False
        return fail_counter.num_failures >= max_failures

    @_queue_stage_hook(output_queue)
    @_stage_hooks(hooks)
    async def pipe() -> None:
        i, tasks = 0, set()
        while not _too_many_failures():
            item = await input_queue.get()
            if item is _EOF and not args.op_requires_eof:
                break
            # note: Make sure that `afunc` is called directly in this function,
            # so as to detect user error. (incompatible `afunc` and `iterator` combo)
            task = create_task(_wrap(afunc(item)), name=f"{name}:{(i := i + 1)}")
            tasks.add(task)

            if len(tasks) >= args.concurrency:
                _, tasks = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

            if item is _EOF:
                break

        if tasks:
            await asyncio.wait(tasks)

        if _too_many_failures():
            raise RuntimeError(
                f"The pipeline stage ({name}) failed more than "
                f"the given threshold. ({max_failures})."
            )

    return pipe()


def _ordered_pipe(
    name: str,
    input_queue: AsyncQueue[T],
    output_queue: AsyncQueue[U],
    args: _PipeArgs[T, U],
    fail_counter: _FailCounter,
    task_hooks: list[TaskHook],
    max_failures: int = -1,
) -> Coroutine:
    """

    Implementation Note:

    The core idea of ordered pipe implementation is to use queue as buffer for active tasks.

                  ┌─┐
                  │ │
                  │ │ AsyncQueue: Input
                  │ │
                  └┬┘
                   │
           ┌───────▼────────┐
           │ Async Function │
           └───────┬────────┘
                  ┌▼┐
                  │ │
                  │ │ AsyncQueue: Intermediate queue:
                  │ │ contains tasks. queue size == concurrency
                  └┬┘
           ┌───────▼────────┐
           │     enqueue    │
           └───────┬────────┘
                  ┌▼┐
                  │ │
                  │ │ AsyncQueue: Output
                  │ │
                  └─┘

    """
    if input_queue is output_queue:
        raise ValueError("input queue and output queue must be different")

    hooks: list[TaskHook] = [*task_hooks, fail_counter]

    # This has been checked in `PipelineBuilder.pipe()`
    assert not inspect.isasyncgenfunction(args.op)

    afunc: Callable[[T], Awaitable[U]] = (  # pyre-ignore: [9]
        convert_to_async(args.op, args.executor)
    )

    inter_queue: AsyncQueue[asyncio.Task[U]] = AsyncQueue(
        f"{name}_interqueue", buffer_size=args.concurrency
    )

    def _too_many_failures() -> bool:
        if max_failures < 0:
            return False
        return fail_counter.num_failures >= max_failures

    async def _run(item: T) -> U:
        async with _task_hooks(hooks):
            return await afunc(item)

    async def get_run_put() -> None:
        i = 0
        while not _too_many_failures():
            item = await input_queue.get()

            if item is _EOF:
                break

            task = create_task(_run(item), name=f"{name}:{(i := i + 1)}")
            await inter_queue.put(task)

        await inter_queue.put(_EOF)  # pyre-ignore: [6]

    async def get_check_put() -> None:
        while True:
            task = await inter_queue.get()

            if task is _EOF:
                # Propagating EOF is done by `_queue_stage_hook`
                return

            await asyncio.wait([task])

            try:
                result = task.result()
            except Exception:
                pass
            else:
                await output_queue.put(result)

    @_queue_stage_hook(output_queue)
    @_stage_hooks(hooks)
    async def ordered_pipe() -> None:
        await asyncio.wait({create_task(get_run_put()), create_task(get_check_put())})

        if _too_many_failures():
            raise RuntimeError(
                f"The pipeline stage ({name}) failed more than "
                f"the given threshold. ({max_failures})."
            )

    return ordered_pipe()


class _Aggregate(Generic[T]):
    def __init__(self, n: int, drop_last: bool) -> None:
        self.n = n
        self.drop_last = drop_last
        self._vals: list[T] = []

    def __call__(self, item: T) -> list[T]:
        if item is not _EOF:
            self._vals.append(item)

        if (len(self._vals) >= self.n) or (
            item is _EOF and not self.drop_last and self._vals
        ):
            ret, self._vals = self._vals, []
            return ret
        return _SKIP  # pyre-ignore: [7]

    def __repr__(self) -> str:
        return (
            f"aggregate({self.n}, drop_last={self.drop_last})"
            if self.drop_last
            else f"aggregate({self.n})"
        )


async def _disaggregate(items: list[T]) -> AsyncIterator[T]:
    for item in items:
        yield item
