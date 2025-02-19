# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import enum
import inspect
import logging
import time
import warnings
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
from dataclasses import dataclass
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
from ._queue import AsyncQueue, StatsQueue as DefaultQueue
from ._utils import create_task, iterate_in_subprocess

__all__ = [
    "PipelineFailure",
    "PipelineBuilder",
    "_get_op_name",
    "run_pipeline_in_subprocess",
]

_LG: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")
T_ = TypeVar("T_")
U_ = TypeVar("U_")


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
async def _queue_stage_hook(queue: AsyncQueue[T]) -> AsyncGenerator[None, None]:
    # Responsibility
    #   1. Call the `stage_hook`` context manager
    #   2. Put _EOF when the stage is done for reasons other than cancell.

    # Note:
    # `asyncio.CancelledError` is a subclass of BaseException, so it won't be
    # caught in the following, and EOF won't be passed to the output queue.
    async with queue.stage_hook():
        try:
            yield
        except Exception:
            await queue.put(_EOF)  # pyre-ignore: [6]
            raise
        else:
            await queue.put(_EOF)  # pyre-ignore: [6]


################################################################################
# _pipe
################################################################################


@dataclass
class _PipeArgs(Generic[T, U]):
    name: str
    op: Callables[T, U]
    executor: Executor | None = None
    concurrency: int = 1
    hooks: list[PipelineHook] | None = None
    report_stats_interval: float | None = None
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


def _get_default_hook(args: _PipeArgs[T, U]) -> list[PipelineHook]:
    if args.hooks is not None:
        return args.hooks
    return [TaskStatsHook(args.name, args.concurrency, args.report_stats_interval)]


def _pipe(
    input_queue: AsyncQueue[T],
    output_queue: AsyncQueue[U],
    args: _PipeArgs[T, U],
) -> Coroutine:
    if input_queue is output_queue:
        raise ValueError("input queue and output queue must be different")

    hooks: list[PipelineHook] = _get_default_hook(args)

    afunc: Callable[[T], Awaitable[U]] = (  # pyre-ignore: [9]
        convert_to_async(args.op, args.executor)
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
                    async with _task_hooks(hooks):
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

    @_queue_stage_hook(output_queue)
    @_stage_hooks(hooks)
    async def pipe() -> None:
        i, tasks = 0, set()
        while True:
            item = await input_queue.get()
            if item is _SKIP:
                continue
            if item is _EOF and not args.op_requires_eof:
                break
            # note: Make sure that `afunc` is called directly in this function,
            # so as to detect user error. (incompatible `afunc` and `iterator` combo)
            task = create_task(_wrap(afunc(item)), name=f"{args.name}:{(i := i + 1)}")
            tasks.add(task)

            if len(tasks) >= args.concurrency:
                _, tasks = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

            if item is _EOF:
                break

        if tasks:
            await asyncio.wait(tasks)

    return pipe()


def _ordered_pipe(
    input_queue: AsyncQueue[T],
    output_queue: AsyncQueue[U],
    args: _PipeArgs[T, U],
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

    hooks: list[PipelineHook] = _get_default_hook(args)

    # This has been checked in `PipelineBuilder.pipe()`
    assert not inspect.isasyncgenfunction(args.op)

    afunc: Callable[[T], Awaitable[U]] = (  # pyre-ignore: [9]
        convert_to_async(args.op, args.executor)
    )

    async def _wrap(item: T) -> asyncio.Task[U]:
        async def _with_hooks() -> U:
            async with _task_hooks(hooks):
                return await afunc(item)

        return create_task(_with_hooks())

    async def _unwrap(task: asyncio.Task[U]) -> U:
        return await task

    inter_queue = AsyncQueue(f"{args.name}_interqueue", args.concurrency)

    coro1: Coroutine[None, None, None] = _pipe(  # pyre-ignore: [1001]
        input_queue,
        inter_queue,
        _PipeArgs(args.name, op=_wrap, executor=args.executor, hooks=[]),
    )

    coro2: Coroutine[None, None, None] = _pipe(  # pyre-ignore: [1001]
        inter_queue,
        output_queue,
        _PipeArgs(args.name, op=_unwrap, executor=args.executor, hooks=[]),
    )

    @_queue_stage_hook(output_queue)
    @_stage_hooks(hooks)
    async def ordered_pipe() -> None:
        await asyncio.wait({create_task(coro1), create_task(coro2)})

    return ordered_pipe()


################################################################################
# _enqueue
################################################################################


async def _enqueue(
    src: Iterable[T] | AsyncIterable[T],
    queue: AsyncQueue[T],
    max_items: int | None = None,
) -> None:
    src_: AsyncIterable[T] = (  # pyre-ignore: [9]
        src if hasattr(src, "__aiter__") else _to_async_gen(iter, None)(src)
    )

    async with _queue_stage_hook(queue):
        num_items = 0
        async for item in src_:
            if item is not _SKIP:
                await queue.put(item)
                num_items += 1
                if max_items is not None and num_items >= max_items:
                    return


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


def _disaggregate(items: Sequence[T_]) -> Iterator[T_]:
    for item in items:
        yield item


@dataclass
class _SourceConfig(Generic[T]):
    source: Iterable | AsyncIterable
    buffer_size: int
    queue_class: type[AsyncQueue[T]]


class _PType(enum.IntEnum):
    Pipe = 1
    OrderedPipe = 2
    Aggregate = 3
    Disaggregate = 4


@dataclass
class _ProcessConfig(Generic[T, U]):
    type_: _PType
    args: _PipeArgs[T, U]
    queue_class: type[AsyncQueue[U]]
    buffer_size: int = 1


@dataclass
class _SinkConfig(Generic[T]):
    buffer_size: int
    queue_class: type[AsyncQueue[T]]


def _get_op_name(op: Callable) -> str:
    if isinstance(op, partial):
        return _get_op_name(op.func)
    return getattr(op, "__name__", op.__class__.__name__)


class PipelineBuilder(Generic[T, U]):
    """Build :py:class:`~spdl.pipeline.Pipeline` object.

    See :py:class:`~spdl.pipeline.Pipeline` for details.
    """

    def __init__(self) -> None:
        self._src: _SourceConfig[T] | None = None
        self._process_args: list[_ProcessConfig] = []  # pyre-ignore: [24]
        self._sink: _SinkConfig[U] | None = None

        self._num_aggregate = 0
        self._num_disaggregate = 0

    def add_source(
        self,
        source: Iterable[T] | AsyncIterable[T],
        *,
        queue_class: type[AsyncQueue[T]] | None = None,
        **_kwargs,  # pyre-ignore: [2]
    ) -> "PipelineBuilder[T, U]":
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

            queue_class: A queue class, used to connect this stage and the next stage.
                Must be a subclassing type (not an instance) of :py:class:`AsyncQueue`.
                Default: :py:class:`StatsQueue`.
        """
        if self._src is not None:
            raise ValueError("Source already set.")

        if not (hasattr(source, "__aiter__") or hasattr(source, "__iter__")):
            raise ValueError("Source must be either generator or async generator.")

        # Note: Do not document this option.
        # See `pipe` method for detail.
        buffer_size = int(_kwargs.get("_buffer_size", 1))
        if buffer_size < 1:
            raise ValueError(
                f"buffer_size must be greater than 0. Found: {buffer_size}"
            )

        self._src = _SourceConfig(source, buffer_size, queue_class or DefaultQueue)
        return self

    def pipe(
        self,
        op: Callables[T_, U_],
        /,
        *,
        concurrency: int = 1,
        executor: Executor | None = None,
        name: str | None = None,
        hooks: list[PipelineHook] | None = None,
        report_stats_interval: float | None = None,
        output_order: str = "completion",
        queue_class: type[AsyncQueue[U_]] | None = None,
        **_kwargs,  # pyre-ignore: [2]
    ) -> "PipelineBuilder[T, U]":
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
            queue_class: A queue class, used to connect this stage and the next stage.
                Must be a subclassing type (not an instance) of :py:class:`AsyncQueue`.
                Default: :py:class:`StatsQueue`.
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
        name_ = name or _get_op_name(op)

        if (op_kwargs := _kwargs.get("kwargs")) is not None:
            warnings.warn(
                "`kwargs` argument is deprecated. "
                "Please use `functools.partial` to bind function arguments.",
                stacklevel=2,
            )
            op = partial(op, **op_kwargs)  # pyre-ignore: [9]

        type_ = _PType.Pipe if output_order == "completion" else _PType.OrderedPipe

        self._process_args.append(
            _ProcessConfig(
                type_=type_,
                args=_PipeArgs(
                    name=name_,
                    op=op,
                    executor=executor,
                    concurrency=concurrency,
                    hooks=hooks,
                    report_stats_interval=report_stats_interval,
                ),
                queue_class=queue_class or DefaultQueue,  # pyre-ignore: [6]
                buffer_size=_kwargs.get("_buffer_size", 1),
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
        hooks: list[PipelineHook] | None = None,
        report_stats_interval: float | None = None,
        queue_class: type[AsyncQueue[T]] | None = None,
    ) -> "PipelineBuilder[T, U]":
        """Buffer the items in the pipeline.

        Args:
            num_items: The number of items to buffer.
            drop_last: Drop the last aggregation if it has less than ``num_aggregate`` items.
            hooks: See :py:meth:`pipe`.
            report_stats_interval: See :py:meth:`pipe`.
            queue_class: A queue class, used to connect this stage and the next stage.
                Must be a subclassing type (not an instance) of :py:class:`AsyncQueue`.
                Default: :py:class:`StatsQueue`.
        """

        name = f"aggregate_{self._num_aggregate}({num_items}, {drop_last=})"
        self._num_aggregate += 1

        self._process_args.append(
            _ProcessConfig(
                _PType.Aggregate,
                _PipeArgs(
                    name=name,
                    op=_Aggregate(num_items, drop_last),
                    hooks=hooks,
                    report_stats_interval=report_stats_interval,
                    op_requires_eof=True,
                ),
                queue_class=queue_class or DefaultQueue,
            )
        )
        return self

    def disaggregate(
        self,
        /,
        *,
        hooks: list[PipelineHook] | None = None,
        report_stats_interval: float | None = None,
        queue_class: type[AsyncQueue[T_]] | None = None,
    ) -> "PipelineBuilder[T, U]":
        """Disaggregate the items in the pipeline.

        Args:
            hooks: See :py:meth:`pipe`.
            report_stats_interval: See :py:meth:`pipe`.

            queue_class: A queue class, used to connect this stage and the next stage.
                Must be a subclassing type (not an instance) of :py:class:`AsyncQueue`.
                Default: :py:class:`StatsQueue`.
        """
        name = f"disaggregate_{self._num_disaggregate}()"
        self._num_disaggregate += 1

        self._process_args.append(
            _ProcessConfig(
                _PType.Disaggregate,
                _PipeArgs(
                    name=name,
                    op=_disaggregate,  # pyre-ignore: [6]
                    hooks=hooks,
                    report_stats_interval=report_stats_interval,
                ),
                queue_class=queue_class or DefaultQueue,
            ),
        )
        return self

    def add_sink(
        self,
        buffer_size: int,
        queue_class: type[AsyncQueue[U]] | None = None,
    ) -> "PipelineBuilder[T, U]":
        """Attach a buffer to the end of the pipeline.

        .. code-block::

            │
           ┌▼┐
           │ │ buffer
           └─┘

        Args:
            buffer_size: The size of the buffer. Pass ``0`` for unlimited buffering.

            queue_class: A queue class, used to connect this stage and the next stage.
                Must be a subclassing type (not an instance) of :py:class:`AsyncQueue`.
                Default: :py:class:`StatsQueue`.
        """
        if self._sink is not None:
            raise ValueError("Sink is already set.")
        if not isinstance(buffer_size, int):
            raise ValueError(f"`buffer_size` must be int. Found: {type(buffer_size)}.")
        if buffer_size < 1:
            raise ValueError(
                f"`buffer_size` must be greater than 0. Found: {buffer_size}"
            )

        self._sink = _SinkConfig(buffer_size, queue_class or DefaultQueue)
        return self

    def _build(  # pyre-ignore: [3]
        self,
    ) -> tuple[Coroutine[None, None, None], list[AsyncQueue[Any]]]:
        # Note:
        # Make sure that coroutines are ordered from source to sink.
        # `_run_pipeline_coroutines` expects and rely on this ordering.
        coros = []
        queues = []

        # source
        if (src := self._src) is None:
            raise ValueError("Source is not set.")
        else:
            queues.append(src.queue_class("src_output", src.buffer_size))
            coros.append(("AsyncPipeline::0_source", _enqueue(src.source, queues[0])))

        # pipes
        for i, cfg in enumerate(self._process_args, start=1):
            queues.append(cfg.queue_class(f"{cfg.args.name}_output", cfg.buffer_size))
            in_queue, out_queue = queues[i - 1 : i + 1]

            match cfg.type_:
                case _PType.Pipe | _PType.Aggregate | _PType.Disaggregate:
                    coro = _pipe(in_queue, out_queue, cfg.args)
                case _PType.OrderedPipe:
                    coro = _ordered_pipe(in_queue, out_queue, cfg.args)
                case _:  # pragma: no cover
                    raise ValueError(f"Unexpected process type: {cfg.type_}")

            coros.append((f"AsyncPipeline::{i}_{cfg.args.name}", coro))

        # sink
        if (sink := self._sink) is not None:
            queues.append(sink.queue_class("src_output", sink.buffer_size))
            coros.append(
                (
                    f"AsyncPipeline::{len(self._process_args) + 1}_sink",
                    _sink(*queues[-2:]),
                )
            )

        return _run_pipeline_coroutines(coros), queues

    def _get_desc(self) -> list[str]:
        parts = []
        if self._src is not None:
            src = self._src
            src_repr = getattr(src.source, "__name__", type(src.source).__name__)
            parts.append(f"  - src: {src_repr}")
            if src.buffer_size != 1:
                parts.append(f"    Buffer: buffer_size={src.buffer_size}")
        else:
            parts.append("  - src: n/a")

        for cfg in self._process_args:
            args = cfg.args
            match cfg.type_:
                case _PType.Pipe:
                    part = f"{args.name}(concurrency={args.concurrency})"
                case _PType.OrderedPipe:
                    part = (
                        f"{args.name}(concurrency={args.concurrency}, "
                        'output_order="input")'
                    )
                case _PType.Aggregate | _PType.Disaggregate:
                    part = args.name
                case _:
                    part = str(cfg.type_)
            parts.append(f"  - {part}")

            if cfg.buffer_size > 1:
                parts.append(f"    Buffer: buffer_size={cfg.buffer_size}")

        if (sink := self._sink) is not None:
            parts.append(f"  - sink: buffer_size={sink.buffer_size}")

        return parts

    def __str__(self) -> str:
        return "\n".join([repr(self), *self._get_desc()])

    def build(self, *, num_threads: int | None = None) -> Pipeline[U]:
        """Build the pipeline.

        Args:
            num_threads: The number of threads in the thread pool attached to
                async event loop.
                If not specified, the maximum concurrency value is used.
        """
        coro, queues = self._build()

        if num_threads is None:
            concurrencies = [cfg.args.concurrency for cfg in self._process_args]
            num_threads = max(concurrencies) if concurrencies else 4
        assert num_threads is not None
        executor = ThreadPoolExecutor(
            max_workers=num_threads,
            thread_name_prefix="spdl_",
        )
        return Pipeline(coro, queues, executor, desc=self._get_desc())


def _run_pipeline(builder: PipelineBuilder[T, U], num_threads: int) -> Iterator[U]:
    pipeline = builder.build(num_threads=num_threads)
    with pipeline.auto_stop():
        yield from pipeline


def run_pipeline_in_subprocess(
    builder: PipelineBuilder[T, U],
    *,
    num_threads: int,
    **kwargs: dict[str, Any],
) -> Iterator[T]:
    """Run the given Pipeline in a subprocess, and iterate on the result.

    Args:
        builder: The definition of :py:class:`Pipeline`.
        num_threads: Passed to :py:meth:`PipelineBuilder.build`.
        kwargs: Passed to :py:func:`iterate_in_subprocess`.

    Yields:
        The results yielded from the pipeline.
    """
    yield from iterate_in_subprocess(
        fn=partial(_run_pipeline, builder=builder, num_threads=num_threads),
        **kwargs,  # pyre-ignore: [6]
    )
