# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "_pipe",
    "_ordered_pipe",
    "_disaggregate",
    "_merge",
    "_get_fail_counter",
]

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Sequence
from contextlib import asynccontextmanager
from fractions import Fraction
from typing import Any, TypeVar

from spdl.pipeline._common._convert import convert_to_async
from spdl.pipeline._common._misc import create_task
from spdl.pipeline._common._types import _TMergeOp
from spdl.pipeline.defs import _PipeArgs

from ._common import _EOF, _EPOCH_END, _SKIP, is_eof, is_epoch_end, StageInfo
from ._hook import _stage_hooks, _task_hooks, TaskHook
from ._queue import _queue_stage_hook, AsyncQueue
from ._semaphore import ResizableSemaphore

# pyre-strict


T = TypeVar("T")
U = TypeVar("U")


class _FailCounter(TaskHook):
    _num_global_failures: int = 0
    _num_global_invocations: int = 0

    def __init__(
        self,
        max_global_failures: int | Fraction = -1,
        max_stage_failures: int | Fraction | None = None,
    ) -> None:
        """Task hook used to watch task failures.

        Args:
            max_global_failures: The maximum number (int) or rate (Fraction) of failures
                permitted across pipe stages.
                A negative int means any number of failure is permitted.
                When using Fraction, it must be > 0 and <= 1.
            max_stage_failures: The maximum number (int) or rate (Fraction) of failures
                permitted for one pipe stage.
                When using Fraction, it must be > 0 and <= 1.

        Raises:
            ValueError: If a Fraction is not in the valid range (0, 1].
        """
        super().__init__()

        # Validate Fraction values
        for name, value in [
            ("max_global_failures", max_global_failures),
            ("max_stage_failures", max_stage_failures),
        ]:
            if isinstance(value, Fraction) and not (0 < value <= 1):
                raise ValueError(
                    f"`{name}` Fraction must be in range (0, 1]. Got: {value}"
                )

        self._max_global_failures = max_global_failures
        self._max_stage_failures = max_stage_failures

        self._num_stage_failures: int = 0
        self._num_stage_invocations: int = 0
        self._exceeded: bool = False

    # Fixed probation period for rate-based thresholds.
    # Rate checking only starts after this many invocations.
    _PROBATION_PERIOD: int = 100

    def _check_threshold(self) -> None:
        """Check if failure threshold exceeded.

        For rate-based thresholds (Fraction), we use a fixed probation period
        of 100 invocations before checking. This prevents early false positives
        when sample size is too small to be statistically meaningful.

        Note: Python's Fraction auto-reduces (e.g., Fraction(20, 100) becomes
        Fraction(1, 5)), so we cannot use the denominator as probation period.

        For example, Fraction(3, 10) means "30% failure rate, but only start
        checking after at least 100 invocations".
        """
        match threshold := self.max_failures:
            case Fraction():
                num, den = threshold.numerator, threshold.denominator
                if (invocations := self.num_invocations) >= self._PROBATION_PERIOD:
                    failures = self.num_failures
                    if failures * den > num * invocations:
                        self._exceeded = True
            case int():
                if threshold >= 0 and self.num_failures > threshold:
                    self._exceeded = True

    @property
    def max_failures(self) -> int | Fraction:
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

    @property
    def num_invocations(self) -> int:
        return (
            self._num_global_invocations
            if self._max_stage_failures is None
            else self._num_stage_invocations
        )

    def too_many_failures(self) -> bool:
        return self._exceeded

    @asynccontextmanager
    async def task_hook(self, input_item: Any = None) -> AsyncIterator[None]:
        _failed = False
        try:
            yield
        except StopAsyncIteration:
            raise
        except Exception:
            _failed = True
            raise
        finally:
            self.__class__._num_global_invocations += 1
            self._num_stage_invocations += 1
            if _failed:
                self.__class__._num_global_failures += 1
                self._num_stage_failures += 1
            self._check_threshold()

    def raise_for_failures(self, info: StageInfo) -> None:
        threshold = self.max_failures
        display = info
        if isinstance(threshold, Fraction):
            rate = Fraction(self.num_failures, self.num_invocations)
            raise RuntimeError(
                f"The pipeline stage ({display}) failed {self.num_failures} times "
                f"out of {self.num_invocations} invocations "
                f"({float(rate):.1%}), which exceeds the threshold "
                f"({float(threshold):.1%})."
            )
        else:
            raise RuntimeError(
                f"The pipeline stage ({display}) failed {self.num_failures} times, "
                f"which exceeds the threshold ({threshold})."
            )


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
    info: StageInfo,
    input_queue: AsyncQueue,
    output_queue: AsyncQueue,
    args: _PipeArgs[T, U],
    fail_counter: _FailCounter,
    task_hooks: list[TaskHook],
    op_requires_eof: bool,
    semaphore: ResizableSemaphore | None = None,
) -> Coroutine:
    """Create a coroutine for processing data from input queue to output queue.

    This function creates a processing stage coroutine that consumes items from the
    input queue, applies a transformation operation (sync or async function/generator),
    and puts the results into the output queue. It manages concurrent execution of
    tasks up to the specified concurrency level.

    Args:
        info: The stage identity for logging and task naming.
        input_queue: The queue to consume input items from.
        output_queue: The queue to put processed items into.
        args: Pipeline arguments containing the operation, executor, and concurrency.
        fail_counter: Hook for tracking and limiting task failures.
        task_hooks: List of hooks for monitoring task execution.
        op_requires_eof: If True, pass EOF token to the operation; otherwise stop
            processing before EOF.
        semaphore: Optional :py:class:`ResizableSemaphore` admission gate
            (V5.6). When provided, the static ``len(tasks) >= concurrency``
            gate is **REPLACED** by ``await semaphore.acquire()`` —
            ``args.concurrency`` is ignored for admission control.
            When ``None`` (default), behaviour is unchanged from
            pre-V5: the static admission gate applies.

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

    # V5.6: branch ONCE outside the hot loop. When `semaphore` is provided,
    # the admission gate is REPLACED — `await semaphore.acquire()` becomes
    # the gate, and the static `len(tasks) >= concurrency` check is
    # skipped entirely. When `semaphore` is None (the default), the
    # original gate applies unchanged so there is ZERO per-task overhead
    # added to the existing fast path.
    if semaphore is not None:
        return _pipe_with_semaphore(
            info,
            input_queue,
            output_queue,
            afunc,
            _wrap,
            args,
            fail_counter,
            hooks,
            op_requires_eof,
            semaphore,
        )

    @_queue_stage_hook(output_queue)
    @_stage_hooks(hooks)
    async def pipe() -> None:
        i, tasks = 0, set()
        while not fail_counter.too_many_failures():
            item = await input_queue.get()

            if is_epoch_end(item):
                # Epoch boundary: wait for all in-flight tasks, propagate, continue
                if tasks:
                    await asyncio.wait(tasks)
                    tasks = set()
                await output_queue.put(_EPOCH_END)
                continue

            if is_eof(item) and not op_requires_eof:
                break
            # note: Make sure that `afunc` is called directly in this function,
            # so as to detect user error. (incompatible `afunc` and `iterator` combo)
            task = create_task(
                _wrap(afunc(item), item),
                name=f"{info}:{(i := i + 1)}",
            )
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
            fail_counter.raise_for_failures(info)

    return pipe()


def _pipe_with_semaphore(
    info: StageInfo,
    input_queue: AsyncQueue,
    output_queue: AsyncQueue,
    # pyre-ignore[2]: afunc has a polymorphic shape (afunc / agen)
    afunc: Callable[[T], Any],
    # pyre-ignore[2]: _wrap closure type matches the branch in _pipe
    _wrap: Callable[..., Coroutine],
    args: _PipeArgs[T, U],
    fail_counter: _FailCounter,
    hooks: list[TaskHook],
    op_requires_eof: bool,
    semaphore: ResizableSemaphore,
) -> Coroutine:
    """V5.6 REPLACE branch: ``semaphore.acquire()`` IS the admission gate.

    ``args.concurrency`` is ignored — the registered semaphore (whose value
    is mutable via :py:meth:`Pipeline.resize_concurrency`) governs the
    in-flight cap. Task completion calls ``semaphore.release()`` via a
    done-callback so the admit cycle is symmetric with the gate.
    """

    # Define _on_done OUTSIDE the loop so the closure captures the
    # single ``tasks`` set / ``semaphore`` once, and so flake8 B023
    # ("loop variable") doesn't fire. ``tasks`` is mutated in-place via
    # ``add()`` / ``discard()`` / ``clear()`` (never rebound) so the
    # closure always sees the current contents.
    tasks: set[asyncio.Task[Any]] = set()

    def _on_done(t: asyncio.Task[Any]) -> None:
        tasks.discard(t)
        semaphore.release()

    @_queue_stage_hook(output_queue)
    @_stage_hooks(hooks)
    async def pipe() -> None:
        i = 0
        while not fail_counter.too_many_failures():
            item = await input_queue.get()

            if is_epoch_end(item):
                # Epoch boundary: wait for all in-flight tasks, propagate, continue
                if tasks:
                    await asyncio.wait(tasks)
                    # Done callbacks have already discarded each task as it
                    # completed, so ``tasks`` is empty here. Clear in-place
                    # (no rebind) for symmetry with the static-gate branch.
                    tasks.clear()
                await output_queue.put(_EPOCH_END)
                continue

            if is_eof(item) and not op_requires_eof:
                break

            # V5.6 admission gate: REPLACES `len(tasks) >= args.concurrency`.
            # `acquire()` blocks here when the in-flight count reaches the
            # semaphore's current value (which may have been resized via
            # Pipeline.resize_concurrency).
            await semaphore.acquire()

            task = create_task(
                _wrap(afunc(item), item),
                name=f"{info}:{(i := i + 1)}",
            )
            tasks.add(task)
            task.add_done_callback(_on_done)

            if is_eof(item):
                break

        if tasks:
            await asyncio.wait(tasks)

        if fail_counter.too_many_failures():
            fail_counter.raise_for_failures(info)

    return pipe()


def _ordered_pipe(
    info: StageInfo,
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
        info: The stage identity for logging and task naming.
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
        StageInfo(
            pipeline_id=info.pipeline_id,
            stage_id=info.stage_id,
            stage_name=f"{info.stage_name}_interqueue",
            concurrency=info.concurrency,
        ),
        buffer_size=args.concurrency,
    )

    async def _run(item: T) -> U:
        async with _task_hooks(task_hooks, item):
            return await afunc(item)

    async def get_run_put() -> None:
        i = 0
        while not fail_counter.too_many_failures():
            item = await input_queue.get()

            if is_epoch_end(item):
                await inter_queue.put(_EPOCH_END)
                continue

            if is_eof(item):
                break

            task = create_task(_run(item), name=f"{info}:{(i := i + 1)}")
            await inter_queue.put((task, item))

        await inter_queue.put(_EOF)

    async def get_check_put() -> None:
        while not fail_counter.too_many_failures():
            entry = await inter_queue.get()

            if is_epoch_end(entry):
                await output_queue.put(_EPOCH_END)
                continue

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
            fail_counter.raise_for_failures(info)

    return ordered_pipe()


async def _disaggregate(items: list[T]) -> AsyncIterator[T]:
    for item in items:
        yield item


async def _default_merge(
    info: StageInfo, input_queues: Sequence[asyncio.Queue], output_queue: asyncio.Queue
) -> None:
    epoch_end_count: int = 0
    epoch_end_barrier: asyncio.Event = asyncio.Event()

    async def _pass(in_q: asyncio.Queue) -> None:
        nonlocal epoch_end_count
        while True:
            item = await in_q.get()
            if is_eof(item):
                return
            if is_epoch_end(item):
                epoch_end_count += 1
                if epoch_end_count >= len(input_queues):
                    # All sub-pipelines reached epoch end
                    epoch_end_count = 0
                    await output_queue.put(_EPOCH_END)
                    epoch_end_barrier.set()
                    epoch_end_barrier.clear()
                else:
                    # Wait for other sub-pipelines to reach epoch end
                    await epoch_end_barrier.wait()
                continue
            await output_queue.put(item)

    tasks = [
        create_task(_pass(in_q), name=f"{info}:{i}")
        for i, in_q in enumerate(input_queues)
    ]
    await asyncio.wait(tasks)


def _merge(
    info: StageInfo,
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
        info: The stage identity for logging and task naming.
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
        await op(info, input_queues, output_queue)

    return merge()
