# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "_pipe",
    "_ordered_pipe",
    "_Aggregate",
    "_disaggregate",
    "_merge",
    "_get_fail_counter",
]

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Sequence
from contextlib import asynccontextmanager
from typing import Any, Generic, TypeVar

from spdl.pipeline._common._convert import convert_to_async
from spdl.pipeline._common._misc import create_task
from spdl.pipeline._common._types import _TMergeOp
from spdl.pipeline.defs import _PipeArgs

from ._common import _EOF, is_eof
from ._hook import _stage_hooks, _task_hooks, TaskHook
from ._queue import _queue_stage_hook, AsyncQueue

# pyre-strict


T = TypeVar("T")
U = TypeVar("U")


_SKIP: None = None


class _FailCounter(TaskHook):
    _num_global_failures: int = 0

    def __init__(
        self, max_global_failures: int = -1, max_stage_failures: int | None = None
    ) -> None:
        """Task hook used to watch task failures.

        Args:
            max_global_failures: The maximum number of failures permitted across pipe
                stages.
                A negative value means any number of failure is permitted.
            max_stage_failures: The maximum number of failures permitted for one pipe
                stage.
        """
        super().__init__()
        self._max_global_failures = max_global_failures
        self._max_stage_failures = max_stage_failures

        self._num_stage_failures: int = 0
        self._exeeded: bool = False

    def _increment(self) -> None:
        self.__class__._num_global_failures += 1
        self._num_stage_failures += 1

        if (threshold := self.max_failures) >= 0:
            if self.num_failures >= threshold:
                self._exeeded = True

    @property
    def max_failures(self) -> int:
        return (
            self._max_global_failures
            if self._max_stage_failures is None
            else self._max_stage_failures
        )

    @property
    def num_failures(self) -> int:
        return (
            self._num_global_failures
            if self._max_stage_failures is None
            else self._num_stage_failures
        )

    def too_many_failures(self) -> bool:
        return self._exeeded

    @asynccontextmanager
    async def task_hook(self, input_item: Any = None) -> AsyncIterator[None]:
        try:
            yield
        except StopAsyncIteration:
            raise
        except Exception:
            self._increment()
            raise


# Create a different Counter class variables for each pipeline,
# so that we can also track the Pipeline-global failures.
def _get_fail_counter() -> type[_FailCounter]:
    class _FC(_FailCounter):
        pass

    return _FC


async def _wrap_afunc(
    coro: Awaitable[U],
    hooks: list[TaskHook],
    queue: AsyncQueue,
    item: Any = None,
) -> None:
    async with _task_hooks(hooks, item):
        result = await coro

    if result is _SKIP:
        return

    await queue.put(result)


async def _wrap_agen(
    coro: AsyncIterator[U],
    hooks: list[TaskHook],
    queue: AsyncQueue,
    item: Any = None,
) -> None:
    exhausted = False
    while not exhausted:
        # NOTE:
        # Nested `except StopAsyncIteration` would look strange.
        # The following explains why.
        #
        # We want to give hooks opportunity to react to StopAsyncIteration,
        # for example, so that StatsHook will not record the task stats
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
            async with _task_hooks(hooks, item):
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
    input_queue: AsyncQueue,
    output_queue: AsyncQueue,
    args: _PipeArgs[T, U],
    fail_counter: _FailCounter,
    task_hooks: list[TaskHook],
    op_requires_eof: bool,
) -> Coroutine:
    """Create a coroutine for processing data from input queue to output queue.

    This function creates a processing stage coroutine that consumes items from the
    input queue, applies a transformation operation (sync or async function/generator),
    and puts the results into the output queue. It manages concurrent execution of
    tasks up to the specified concurrency level.

    Args:
        name: The name of the pipeline stage for logging and task naming.
        input_queue: The queue to consume input items from.
        output_queue: The queue to put processed items into.
        args: Pipeline arguments containing the operation, executor, and concurrency.
        fail_counter: Hook for tracking and limiting task failures.
        task_hooks: List of hooks for monitoring task execution.
        op_requires_eof: If True, pass EOF token to the operation; otherwise stop
            processing before EOF.

    Returns:
        A coroutine that executes the pipeline stage.

    Raises:
        ValueError: If input_queue and output_queue are the same object.
        RuntimeError: If the number of failures exceeds the threshold.
    """
    if input_queue is output_queue:
        raise ValueError("input queue and output queue must be different.")

    afunc: Callable[[T], Awaitable[U]] = (  # pyre-ignore: [9]
        convert_to_async(args.op, args.executor)
    )

    hooks: list[TaskHook] = [*task_hooks, fail_counter]

    if inspect.iscoroutinefunction(afunc):

        def _wrap(coro: Awaitable[U], item: Any = None) -> Coroutine:
            return _wrap_afunc(coro, hooks, output_queue, item)

    elif inspect.isasyncgenfunction(afunc):

        def _wrap(coro: AsyncIterator[U], item: Any = None) -> Coroutine:
            return _wrap_agen(coro, hooks, output_queue, item)

    else:
        raise ValueError(f"{afunc=} must be either async function or async generator.")

    @_queue_stage_hook(output_queue)
    @_stage_hooks(hooks)
    async def pipe() -> None:
        i, tasks = 0, set()
        while not fail_counter.too_many_failures():
            item = await input_queue.get()
            if is_eof(item) and not op_requires_eof:
                break
            # note: Make sure that `afunc` is called directly in this function,
            # so as to detect user error. (incompatible `afunc` and `iterator` combo)
            task = create_task(_wrap(afunc(item), item), name=f"{name}:{(i := i + 1)}")
            tasks.add(task)

            if len(tasks) >= args.concurrency:
                _, tasks = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

            if is_eof(item):
                break

        if tasks:
            await asyncio.wait(tasks)

        if fail_counter.too_many_failures():
            raise RuntimeError(
                f"The pipeline stage ({name}) failed {fail_counter.num_failures} times, "
                f"which exceeds the threshold ({fail_counter.max_failures})."
            )

    return pipe()


def _ordered_pipe(
    name: str,
    input_queue: AsyncQueue,
    output_queue: AsyncQueue,
    args: _PipeArgs[T, U],
    fail_counter: _FailCounter,
    task_hooks: list[TaskHook],
) -> Coroutine:
    """Create a coroutine for processing data while preserving input order.

    This function creates a processing stage coroutine similar to ``_pipe``, but guarantees
    that output items maintain the same order as input items, regardless of task
    completion order. This is achieved by using an intermediate queue to buffer tasks
    and waiting for them to complete in sequence.

    Args:
        name: The name of the pipeline stage for logging and task naming.
        input_queue: The queue to consume input items from.
        output_queue: The queue to put processed items into (in order).
        args: Pipeline arguments containing the operation, executor, and concurrency.
        fail_counter: Hook for tracking and limiting task failures.
        task_hooks: List of hooks for monitoring task execution.

    Returns:
        A coroutine that executes the ordered pipeline stage.

    Raises:
        ValueError: If input_queue and output_queue are the same object.
        RuntimeError: If the number of failures exceeds the threshold.

    Note:
        The operation must be an async function (not an async generator) for ordered pipe.

    **Implementation Note**

    The core idea of ordered pipe implementation is to use queue as buffer for active tasks.

    .. code-block::

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

    # This has been checked in `PipelineBuilder.pipe()`
    assert not inspect.isasyncgenfunction(args.op)

    afunc: Callable[[T], Awaitable[U]] = (  # pyre-ignore: [9]
        convert_to_async(args.op, args.executor)
    )

    inter_queue: AsyncQueue = AsyncQueue(
        f"{name}_interqueue", buffer_size=args.concurrency
    )

    async def _run(item: T) -> U:
        async with _task_hooks(task_hooks, item):
            return await afunc(item)

    async def get_run_put() -> None:
        i = 0
        while not fail_counter.too_many_failures():
            item = await input_queue.get()

            if is_eof(item):
                break

            task = create_task(_run(item), name=f"{name}:{(i := i + 1)}")
            await inter_queue.put((task, item))

        await inter_queue.put(_EOF)

    async def get_check_put() -> None:
        while not fail_counter.too_many_failures():
            entry = await inter_queue.get()

            if is_eof(entry):
                # Propagating EOF is done by `_queue_stage_hook`
                return

            task, item = entry

            await asyncio.wait([task])

            try:
                async with fail_counter.task_hook(item):
                    result = task.result()
            except Exception:
                pass
            else:
                if result is _SKIP:
                    continue

                await output_queue.put(result)

        # Drain until EOF
        while not is_eof(await inter_queue.get()):
            pass

    @_queue_stage_hook(output_queue)
    @_stage_hooks(task_hooks)
    async def ordered_pipe() -> None:
        await asyncio.wait({create_task(get_run_put()), create_task(get_check_put())})

        if fail_counter.too_many_failures():
            raise RuntimeError(
                f"The pipeline stage ({name}) failed {fail_counter.num_failures} times, "
                f"which exceeds the threshold ({fail_counter.max_failures})."
            )

    return ordered_pipe()


class _Aggregate(Generic[T]):
    def __init__(self, n: int, drop_last: bool) -> None:
        self.n = n
        self.drop_last = drop_last
        self._vals: list[T] = []

    def __call__(self, item: T) -> list[T]:
        if not is_eof(item):
            self._vals.append(item)

        if (len(self._vals) >= self.n) or (
            is_eof(item) and not self.drop_last and self._vals
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


async def _default_merge(
    name: str, input_queues: Sequence[asyncio.Queue], output_queue: asyncio.Queue
) -> None:
    async def _pass(in_q: asyncio.Queue) -> None:
        while True:
            item = await in_q.get()
            if is_eof(item):
                return
            await output_queue.put(item)

    tasks = [
        create_task(_pass(in_q), name=f"{name}:{i}")
        for i, in_q in enumerate(input_queues)
    ]
    await asyncio.wait(tasks)


def _merge(
    name: str,
    input_queues: Sequence[AsyncQueue],
    output_queue: AsyncQueue,
    fail_counter: _FailCounter,
    task_hooks: list[TaskHook],
    merge_op: _TMergeOp | None,
) -> Coroutine:
    """Create a coroutine for merging data from multiple input queues.

    This function creates a merge stage coroutine that consumes items from multiple
    input queues and puts them into a single output queue. By default, items from
    all input queues are passed through in the order they become available. A custom
    merge operation can be provided for more complex merging logic.

    Args:
        name: The name of the pipeline stage for logging and task naming.
        input_queues: Sequence of queues to consume input items from.
        output_queue: The queue to put merged items into.
        fail_counter: Hook for tracking and limiting task failures.
        task_hooks: List of hooks for monitoring task execution.
        merge_op: Optional custom merge operation. If None, uses default pass-through
            merge that forwards items from all input queues to output queue.

    Returns:
        A coroutine that executes the merge stage.

    Raises:
        ValueError: If input_queues is empty or if any input queue is the same as
            the output queue.
    """
    if not input_queues:
        raise ValueError("There must be at least one input queue.")

    if any(q is output_queue for q in input_queues):
        raise ValueError("The input queue and output queue must be different.")

    hooks: list[TaskHook] = [*task_hooks, fail_counter]

    op: _TMergeOp = merge_op or _default_merge

    @_queue_stage_hook(output_queue)
    @_stage_hooks(hooks)
    async def merge() -> None:
        await op(name, input_queues, output_queue)

    return merge()
