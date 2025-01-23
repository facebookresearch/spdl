# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import inspect
import logging
import time
from asyncio import Queue as AsyncQueue
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    Iterator,
    Sequence,
)
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from functools import partial
from typing import Any, AsyncGenerator, Generic, TypeVar

from ._convert import _to_async_gen, Callables, convert_to_async
from ._hook import (
    _stage_hooks,
    _task_hooks,
    _time_str,
    PipelineHook,
    StatsCounter,
    TaskStatsHook,
)
from ._pipeline import Pipeline
from ._utils import create_task

__all__ = ["PipelineFailure", "PipelineBuilder", "_get_op_name"]

_LG: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


# Sentinel objects used to instruct AsyncPipeline to take special actions.
class _Sentinel:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name


_EOF = _Sentinel("EOF")  # Indicate the end of stream.
_SKIP = _Sentinel("SKIP")  # Indicate that there is no data to process.


# The following is how we intend pipeline to behave. If the implementation
# is inconsistent with the following, it is a bug.
#
# 1. Successful execution.
#
#    Assumption: The consumer keeps consuming the data from the output queue.
#
#    Each stage completes without failure.
#    The source will pass EOF, and each stage receiving EOF will shut down,
#    then pass EOF.
#    EOF is not propagated to the output queue.
#
# 2. Failures at some stage. One or more stages fail.
#
#    Assumption: The consumer keeps consuming the data from the output queue.
#
#    Resolution:
#      We want to keep the tasks downstream to failed ones alive, so that they
#      can finish the ongoing works.
#
#    Actions:
#      - The stage must pass EOF to the next queue, so that downstream stages
#        can exit cleanly.
#      - Passing EOF to the next queue can be blocked if the queue is full.
#      - For a graceful exit, the assumption must be met.
#      - The stages upstream to the failure must be cancelled.
#
# 3. Cancelled: The pipeline execution is cancelled.
#
#    Assumption: The consumer stopped consuming the data from the output queue.
#
#    Actions:
#      - All the stages must be cancelled.
#        (If tasks are blocked on async await, cancelling should do).
#
# Following the above intention, each stage must pass EOF to the output queue
# unless the task was cancelled.
#
@asynccontextmanager
async def _put_eof_when_done(queue: AsyncQueue) -> AsyncGenerator[None, None]:
    # Note:
    # `asyncio.CancelledError` is a subclass of BaseException, so it won't be
    # caught in the following, and EOF won't be passed to the output queue.
    try:
        yield
    except Exception:
        await queue.put(_EOF)
        raise
    else:
        await queue.put(_EOF)


################################################################################
# _pipe
################################################################################


def _get_op_name(op: Callable) -> str:
    if isinstance(op, partial):
        return _get_op_name(op.func)
    return getattr(op, "__name__", op.__class__.__name__)


def _pipe(
    input_queue: AsyncQueue[T],
    op: Callables[T, U],
    output_queue: AsyncQueue[U],
    executor: type[Executor] | None = None,
    concurrency: int = 1,
    name: str = "pipe",
    hooks: Sequence[PipelineHook] | None = None,
    report_stats_interval: float | None = None,
    _pipe_eof: bool = False,
) -> Coroutine:
    if input_queue is output_queue:
        raise ValueError("input queue and output queue must be different")

    if concurrency < 1:
        raise ValueError("`concurrency` value must be >= 1")

    hooks = (
        [TaskStatsHook(name, concurrency, interval=report_stats_interval)]
        if hooks is None
        else hooks
    )

    afunc: Callable[[T], Awaitable[U]] = (  # pyre-ignore: [9]
        convert_to_async(op, executor)
    )

    if inspect.iscoroutinefunction(afunc):

        async def _wrap(coro: Awaitable[U]) -> None:
            async with _task_hooks(hooks):
                result = await coro

            await output_queue.put(result)

    elif inspect.isasyncgenfunction(afunc):

        async def _wrap(coro: AsyncIterator[U]) -> None:
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
                # the exhausted flag to True, then re-raise the execption so as
                # to give hooks chance to react to it.
                # If the hooks do not absorb the StopAsyncIteration, and
                # it propagates them, then we catch it and exit.
                try:
                    async with _task_hooks(hooks):  # pyre-ignore: [16]
                        try:
                            result = await anext(coro)
                        except StopAsyncIteration:
                            exhausted = True
                            raise

                    # If task_hooks absorb the `StopAsyncIteration`, we need to exit here.
                    if exhausted:
                        return

                    await output_queue.put(result)
                except StopAsyncIteration:
                    return

    else:
        raise ValueError(f"{afunc=} must be either async function or async generator.")

    @_put_eof_when_done(output_queue)
    @_stage_hooks(hooks)
    async def pipe() -> None:
        i, tasks = 0, set()
        while True:
            item = await input_queue.get()
            if item is _SKIP:
                continue
            if item is _EOF and not _pipe_eof:
                break
            # note: Make sure that `afunc` is called directly in this function,
            # so as to detect user error. (incompatible `afunc` and `iterator` combo)
            task = create_task(_wrap(afunc(item)), name=f"{name}:{(i := i + 1)}")
            tasks.add(task)

            if len(tasks) >= concurrency:
                done, tasks = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

            if item is _EOF:
                break

        if tasks:
            await asyncio.wait(tasks)

    return pipe()


def _ordered_pipe(
    input_queue: AsyncQueue[T],
    op: Callables[T, U],
    output_queue: AsyncQueue[U],
    executor: type[Executor] | None = None,
    concurrency: int = 1,
    name: str = "pipe",
    hooks: Sequence[PipelineHook] | None = None,
    report_stats_interval: float | None = None,
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

    if concurrency < 1:
        raise ValueError("`concurrency` value must be >= 1")

    hooks_: Sequence[PipelineHook] = (
        [TaskStatsHook(name, concurrency, interval=report_stats_interval)]
        if hooks is None
        else hooks
    )

    # This has been checked in `PipelineBuilder.pipe()`
    assert not inspect.isasyncgenfunction(op)

    afunc: Callable[[T], Awaitable[U]] = (  # pyre-ignore: [9]
        convert_to_async(op, executor)
    )

    async def _wrap(item: T) -> asyncio.Task[U]:
        async def _with_hooks() -> U:
            async with _task_hooks(hooks_):
                return await afunc(item)

        return create_task(_with_hooks())

    async def _unwrap(task: asyncio.Task[U]) -> U:
        return await task

    inter_queue = AsyncQueue(concurrency)

    coro1: Awaitable[None] = _pipe(  # pyre-ignore: [1001]
        input_queue,
        _wrap,
        inter_queue,
        executor,
        1,
        name,
        hooks=[],
    )

    coro2: Awaitable[None] = _pipe(  # pyre-ignore: [1001]
        inter_queue,
        _unwrap,
        output_queue,
        executor,
        1,
        name,
        hooks=[],
    )

    @_put_eof_when_done(output_queue)
    @_stage_hooks(hooks_)
    async def ordered_pipe() -> None:
        await asyncio.wait({create_task(coro1), create_task(coro2)})

    return ordered_pipe()


################################################################################
# _enqueue
################################################################################


def _enqueue(
    src: Iterable[T] | AsyncIterable[T],
    queue: AsyncQueue[T],
    max_items: int | None = None,
) -> Coroutine:
    src_: AsyncIterable[T] = (  # pyre-ignore: [9]
        src if hasattr(src, "__aiter__") else _to_async_gen(iter, None)(src)
    )

    @_put_eof_when_done(queue)
    async def enqueue() -> None:
        num_items = 0
        async for item in src_:
            if item is not _SKIP:
                await queue.put(item)
                num_items += 1
                if max_items is not None and num_items >= max_items:
                    return

    return enqueue()


################################################################################
# _sink
################################################################################


@contextmanager
def _sink_stats() -> Iterator[tuple[StatsCounter, StatsCounter]]:
    get_counter = StatsCounter()
    put_counter = StatsCounter()
    t0 = time.monotonic()
    try:
        yield get_counter, put_counter
    finally:
        elapsed = time.monotonic() - t0
        _LG.info(
            "[sink]\tProcessed %5d items in %s. "
            "QPS: %.2f. "
            "Average wait time: Upstream: %s, Downstream: %s.",
            put_counter.num_items,
            _time_str(elapsed),
            put_counter.num_items / elapsed if elapsed > 0.001 else float("nan"),
            get_counter,
            put_counter,
        )


async def _sink(input_queue: AsyncQueue[T], output_queue: AsyncQueue[T]) -> None:
    with _sink_stats() as (get_counter, put_counter):
        while True:
            with get_counter.count():
                item = await input_queue.get()

            if item is _EOF:
                break

            if item is _SKIP:
                continue

            with put_counter.count():
                await output_queue.put(item)


################################################################################
# Coroutine execution logics
################################################################################


# TODO [Python 3.11]: Migrate to ExceptionGroup
class PipelineFailure(RuntimeError):
    """PipelineFailure()
    Thrown by :py:class:`spdl.pipeline.Pipeline` when pipeline encounters an error.
    """

    def __init__(self, errs: dict[str, Exception]) -> None:
        msg = []
        for k, v in errs.items():
            e = str(v)
            msg.append(f"{k}:{e if e else type(v).__name__}")
        msg.sort()

        super().__init__(", ".join(msg))

        # This is for unittesting.
        self._errs = errs


async def _run_pipeline_coroutines(
    coros: list[tuple[str, Coroutine[None, None, None]]],
) -> None:
    """Run the pipeline coroutines and handle errors.

    Args:
        coros: The croutines each corresponds to a stage in pipelin.
            IMPORTANT: The coroutinues must be in the order of src to sink.
    """
    tasks = [create_task(coro, name=name) for name, coro in coros]
    pending = set(tasks)

    while pending:
        # Note:
        # `asyncio.wait` does not automatically propagate the cancellation to its children.
        # For graceful shutdown, we manually cancel the child tasks.
        #
        # Also, it seems asyncio loop throws Cancellation on most outer task.
        # I am not sure where this behavior is documented, but here is an example script to
        # demonstrate the behavior.
        # https://gist.github.com/mthrok/3a1c11c2d8012e29f4835679ac0baaee
        try:
            _, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_EXCEPTION
            )
        except asyncio.CancelledError:
            for task in pending:
                task.cancel()
            await asyncio.wait(pending)
            raise

        if not pending:
            break

        # Check if any of the task caused an error.
        # If an error occured, we cancel the stages upstream to the failed one,
        # then continue waiting the downstream ones.
        for i in range(len(tasks) - 1, -1, -1):
            task = tasks[i]
            if task.done() and not task.cancelled() and task.exception() is not None:
                for task in tasks[:i]:
                    task.cancel()
                break

    errs = {}
    for task in tasks:
        if not task.cancelled() and (err := task.exception()) is not None:
            errs[task.get_name()] = err

    if errs:
        raise PipelineFailure(errs)


################################################################################
# Build
################################################################################


def disaggregate(items: Sequence[T]) -> Iterator[T]:
    for item in items:
        yield item


class PipelineBuilder(Generic[T]):
    """Build :py:class:`~spdl.pipeline.Pipeline` object.

    See :py:class:`~spdl.pipeline.Pipeline` for details.
    """

    def __init__(self) -> None:
        self._source: Iterable | AsyncIterable | None = None
        self._source_buffer_size = 1

        self._process_args: list[tuple[str, dict[str, Any], int]] = []

        self._sink_buffer_size: int | None = None
        self._num_aggregate = 0
        self._num_disaggregate = 0

    def add_source(
        self,
        source: Iterable[T] | AsyncIterable[T],
        **_kwargs,  # pyre-ignore: [2]
    ) -> "PipelineBuilder[T]":
        """Attach an iterator to the source buffer.

        .. code-block::

           ┌─────────────────┐
           │ (Async)Iterator │
           └───────┬─────────┘
                   ▼

        Args:
            source: A lightweight iterator that generates data.

                .. warning::

                   The source iterator must be lightweight as it is executed in async
                   event loop. If the iterator performs a blocking operation,
                   the entire pipeline will be blocked.
        """
        if self._source is not None:
            raise ValueError("Source already set.")

        if not (hasattr(source, "__aiter__") or hasattr(source, "__iter__")):
            raise ValueError("Source must be either generator or async generator.")

        self._source = source

        # Note: Do not document this option.
        # See `pipe` method for detail.
        if "_buffer_size" in _kwargs:
            self._source_buffer_size = int(_kwargs["_buffer_size"])
        return self

    def pipe(
        self,
        op: Callables[T, U],
        /,
        *,
        concurrency: int = 1,
        executor: Executor | None = None,
        name: str | None = None,
        hooks: Sequence[PipelineHook] | None = None,
        report_stats_interval: float | None = None,
        output_order: str = "completion",
        kwargs: dict[str, ...] | None = None,
        **_kwargs,  # pyre-ignore: [2]
    ) -> "PipelineBuilder[T]":
        """Apply an operation to items in the pipeline.

        .. code-block::

                 │
           ┌─────▼─────┐
           │ Operation │
           └─────┬─────┘
                 ▼

        Args:
            op: A function applied to items in the queue.
                The function must take exactly one argument, which is the output
                from the upstream. If passing around multiple objects, take
                them as a tuple or use :py:class:`~dataclasses.dataclass` and
                define a custom protocol.

                Optionally, the op can be a generator function, async function or
                async generator function.

                If ``op`` is (async) generator, the items yielded are put in the output
                queue separately.

                .. warning::

                   If ``op`` is synchronous geneartor, and ``executor`` is an instance of
                   :py:class:`concurrent.futures.ProcessPoolExecutor`, the output items
                   are not put in the output queue until the generator is exhausted.

                   Async generator, or synchronous generator without ``ProcessPoolExecutor``
                   does not have this issue, and the yielded items are put in the output
                   queue immediately.

                .. tip::

                   When passing an async op, make sure that the op does not call sync
                   function inside.
                   If calling a sync function, use :py:func:`asyncio.loop.run_in_executor`
                   or :py:func:`asyncio.to_thread` to delegate the execution to the thread pool.

            concurrency: The maximum number of async tasks executed concurrently.
            executor: A custom executor object to be used to convert the synchronous operation
                into asynchronous one. If ``None``, the default executor is used.

                It is invalid to provide this argument when the given op is already async.
            name: The name (prefix) to give to the task.
            hooks: Hook objects to be attached to the stage. Hooks are intended for
                collecting stats of the stage.
                If ``None``, a default hook,
                :py:class:`~spdl.pipeline.TaskStatsHook` is used.
            report_stats_interval:
                The interval (in seconds) to log the stats of this stage when no
                ``hooks`` is provided. This argument is passed to
                :py:class:`~spdl.pipeline.TaskStatsHook`.
                This argument is effective only when ``hooks`` are not provided.
                If ``hooks`` is provided and stats report is needed,
                the ``hooks`` argument should
                include one of :py:class:`~spdl.pipeline.TaskStatsHook`
                instance with the desired interval.
            output_order: If ``"completion"`` (default), the items are put to output queue
                in the order their process is completed.
                If ``"input"``, then the items are put to output queue in the order given
                in the input queue.
            kwargs: Keyword arguments to be passed to the ``op``.
        """
        if output_order not in ["completion", "input"]:
            raise ValueError(
                '`output_order` must be either "completion" or "input". '
                f"Found: {output_order}"
            )

        if inspect.iscoroutinefunction(op) or inspect.isasyncgenfunction(op):
            if executor is not None:
                raise ValueError("`executor` cannot be specified when op is async.")
        if inspect.isasyncgenfunction(op):
            if output_order == "input":
                raise ValueError(
                    "pipe does not support async generator function "
                    "when `output_order` is 'input'."
                )

        name = name or _get_op_name(op)

        if kwargs:
            # pyre-ignore
            op = partial(op, **kwargs)

        self._process_args.append(
            (
                "pipe" if output_order == "completion" else "ordered_pipe",
                {
                    "op": op,
                    "executor": executor,
                    "concurrency": concurrency,
                    "name": name,
                    "hooks": hooks,
                    "report_stats_interval": report_stats_interval,
                },
                _kwargs.get("_buffer_size", 1),
                # Note:
                # `_buffer_size` option is intentionally not documented.
                #
                # The pipeline practically buffers `concurrency + _buffer_size`
                # items, which leads to confusing behavior when writing tests.
                #
                # And it does not affect throughput, however, showing it as
                # `_buffer_size=1` often make users think that this needs to be
                # increased to improve the performance.
                #
                # We hide the argument under `kwargs` to keep it undocumented,
                # while allowing users to adjust if it's absolutely necessary.
                #
                # So please keep it undocumented. Thanks.
                #
                # Oh, but if you ever find the case that this does affect the
                # performance, let us know.
            )
        )
        return self

    def aggregate(
        self,
        num_items: int,
        /,
        *,
        drop_last: bool = False,
        hooks: Sequence[PipelineHook] | None = None,
        report_stats_interval: float | None = None,
    ) -> "PipelineBuilder[T]":
        """Buffer the items in the pipeline.

        Args:
            num_items: The number of items to buffer.
            drop_last: Drop the last aggregation if it has less than ``num_aggregate`` items.
            hooks: See :py:meth:`pipe`.
            report_stats_interval: See :py:meth:`pipe`.
        """
        vals: list[list[T]] = [[]]

        def aggregate(i: T) -> list[T]:
            if i is not _EOF:
                vals[0].append(i)

            if (i is _EOF and vals[0]) or (len(vals[0]) >= num_items):
                ret = vals.pop(0)
                vals.append([])
                return ret
            return _SKIP  # pyre-ignore: [7]

        name = f"aggregate_{self._num_aggregate}({num_items}, {drop_last=})"
        self._num_aggregate += 1

        self._process_args.append(
            (
                "aggregate",
                {
                    "op": aggregate,
                    "executor": None,
                    "concurrency": 1,
                    "name": name,
                    "hooks": hooks,
                    "report_stats_interval": report_stats_interval,
                    "_pipe_eof": not drop_last,
                },
                1,
            )
        )
        return self

    def disaggregate(
        self,
        /,
        *,
        hooks: Sequence[PipelineHook] | None = None,
        report_stats_interval: float | None = None,
    ) -> "PipelineBuilder[T]":
        """Disaggregate the items in the pipeline.

        Args:
            hooks: See :py:meth:`pipe`.
            report_stats_interval: See :py:meth:`pipe`.
        """
        name = f"disaggregate_{self._num_disaggregate}()"
        self._num_disaggregate += 1

        self._process_args.append(
            (
                "disaggregate",
                {
                    "op": disaggregate,
                    "executor": None,
                    "concurrency": 1,
                    "name": name,
                    "hooks": hooks,
                    "report_stats_interval": report_stats_interval,
                },
                1,
            )
        )
        return self

    def add_sink(self, buffer_size: int) -> "PipelineBuilder[T]":
        """Attach a buffer to the end of the pipeline.

        .. code-block::

            │
           ┌▼┐
           │ │ buffer
           └─┘

        Args:
            buffer_size: The size of the buffer. Pass ``0`` for unlimited buffering.
        """
        if self._sink_buffer_size is not None:
            raise ValueError("Sink is already set.")
        if not isinstance(buffer_size, int):
            raise ValueError(f"`buffer_size` must be int. Found: {type(buffer_size)}.")
        self._sink_buffer_size = buffer_size
        return self

    def _build(self) -> tuple[Coroutine[None, None, None], list[AsyncQueue]]:
        if self._source is None:
            raise ValueError("Source is not set.")

        # Note:
        # Make sure that coroutines are ordered from source to sink.
        # `_run_pipeline_coroutines` expects and rely on this ordering.
        coros = []
        queues: list[AsyncQueue] = []

        # source
        queues.append(AsyncQueue(self._source_buffer_size))
        assert self._source is not None
        coros.append(
            (
                "AsyncPipeline::0_source",
                _enqueue(self._source, queues[0]),
            )
        )

        # pipes
        for i, (type_, args, buffer_size) in enumerate(self._process_args, start=1):
            queues.append(AsyncQueue(buffer_size))
            in_queue, out_queue = queues[i - 1 : i + 1]

            match type_:
                case "pipe" | "aggregate" | "disaggregate":
                    coro = _pipe(**args, input_queue=in_queue, output_queue=out_queue)
                case "ordered_pipe":
                    coro = _ordered_pipe(
                        **args, input_queue=in_queue, output_queue=out_queue
                    )
                case _:  # pragma: no cover
                    raise ValueError(f"Unexpected process type: {type_}")

            coros.append((f"AsyncPipeline::{i}_{args['name']}", coro))

        # sink
        if self._sink_buffer_size is not None:
            queues.append(AsyncQueue(self._sink_buffer_size))
            coros.append(
                (
                    f"AsyncPipeline::{len(self._process_args) + 1}_sink",
                    _sink(*queues[-2:]),
                )
            )

        return _run_pipeline_coroutines(coros), queues

    def _get_desc(self) -> list[str]:
        parts = []
        if self._source is not None:
            src_repr = getattr(self._source, "__name__", type(self._source).__name__)
            parts.append(f"  - src: {src_repr}")
        else:
            parts.append("  - src: n/a")

        if self._source_buffer_size != 1:
            parts.append(f"    Buffer: buffer_size={self._source_buffer_size}")

        for type_, args, buffer_size in self._process_args:
            match type_:
                case "pipe":
                    part = f"{args['name']}(concurrency={args['concurrency']})"
                case "ordered_pipe":
                    part = (
                        f"{args['name']}(concurrency={args['concurrency']}, "
                        'output_order="input")'
                    )
                case "aggregate":
                    part = args["name"]
                case _:
                    part = type_
            parts.append(f"  - {part}")

            if type_ not in ["aggregate"] and buffer_size > 1:
                parts.append(f"    Buffer: buffer_size={buffer_size}")

        if self._sink_buffer_size is not None:
            parts.append(f"  - sink: buffer_size={self._sink_buffer_size}")

        return parts

    def __str__(self) -> str:
        return "\n".join([repr(self), *self._get_desc()])

    def build(self, *, num_threads: int | None = None) -> Pipeline[T]:
        """Build the pipeline.

        Args:
            num_threads: The number of threads in the thread pool attached to
                async event loop.
                If not specified, the maximum concurrency value is used.
        """
        coro, queues = self._build()

        if num_threads is None:
            concurrencies = [
                args["concurrency"]
                for _, args, _ in self._process_args
                if "concurrency" in args
            ]
            num_threads = max(concurrencies) if concurrencies else 4
        assert num_threads is not None
        executor = ThreadPoolExecutor(
            max_workers=num_threads,
            thread_name_prefix="spdl_",
        )
        return Pipeline(coro, queues, executor, desc=self._get_desc())
