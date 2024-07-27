# pyre-unsafe

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
from contextlib import asynccontextmanager, contextmanager
from typing import TypeVar

from ._hook import _stage_hooks, _task_hooks, PipelineHook, TaskStatsHook
from ._pipeline import Pipeline
from ._utils import create_task

__all__ = ["PipelineFailure", "PipelineBuilder"]

_LG = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


# Sentinel objects used to instruct AsyncPipeline to take special actions.
class _Sentinel:
    def __init__(self, name):
        self.name = name

    def __str__(self):
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
async def _put_eof_when_done(queue):
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


def _pipe(
    input_queue: AsyncQueue[T],
    afunc: Callable[[T], Awaitable[U]],
    output_queue: AsyncQueue[U],
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

    if inspect.iscoroutinefunction(afunc):

        async def _wrap(coro: Awaitable[U]) -> None:
            async with _task_hooks(hooks):  # pyre-ignore: [16]
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
    async def pipe():
        i, tasks = 0, set()
        while True:
            item = await input_queue.get()
            if item is _SKIP:
                continue
            if item is _EOF and not _pipe_eof:
                break
            # note: Make sure that `afunc` is called directly in this function,
            # so as to detect user error. (incompatible `afunc` and `iterator` combo)
            task = create_task(_wrap(afunc(item)), name=f"{name}_{(i := i + 1)}")
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
    afunc: Callable[[T], Awaitable[U]],
    output_queue: AsyncQueue[U],
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

    hooks = (
        [TaskStatsHook(name, concurrency, interval=report_stats_interval)]
        if hooks is None
        else hooks
    )

    async def _wrap(item: T) -> asyncio.Task[U]:
        async def _with_hooks():
            async with _task_hooks(hooks):  # pyre-ignore: [16]
                return await afunc(item)

        return create_task(_with_hooks())

    async def _unwrap(task: asyncio.Task[U]) -> U:
        return await task

    inter_queue = AsyncQueue(concurrency)

    coro1 = _pipe(  # pyre-ignore: [1001]
        input_queue,
        _wrap,
        inter_queue,
        1,
        name,
        hooks=[],
    )

    coro2 = _pipe(  # pyre-ignore: [1001]
        inter_queue,
        _unwrap,
        output_queue,
        1,
        name,
        hooks=[],
    )

    @_put_eof_when_done(output_queue)
    @_stage_hooks(hooks)
    async def ordered_pipe():
        tasks = {create_task(coro1), create_task(coro2)}
        await asyncio.wait(tasks)

    return ordered_pipe()


################################################################################
# _enqueue
################################################################################


def _enqueue(
    src: Iterable[T] | AsyncIterable[T],
    queue: AsyncQueue[T],
    max_items: int | None = None,
) -> Coroutine:
    if hasattr(src, "__aiter__"):

        @_put_eof_when_done(queue)
        async def enqueue():
            num_items = 0
            async for item in src:
                if item is not _SKIP:
                    await queue.put(item)
                    num_items += 1
                    if max_items is not None and num_items >= max_items:
                        return

    elif hasattr(src, "__iter__"):

        @_put_eof_when_done(queue)
        async def enqueue():
            num_items = 0
            for item in src:
                if item is not _SKIP:
                    await queue.put(item)
                    num_items += 1
                    if max_items is not None and num_items >= max_items:
                        return

    else:
        raise ValueError(f"{src=} must be either generator or async generator.")

    return enqueue()


################################################################################
# _sink
################################################################################


def _time_str(val: float) -> str:
    return "{:.4f} [{:>3s}]".format(
        val * 1000 if val < 1 else val,
        "ms" if val < 1 else "sec",
    )


class _Counter:
    def __init__(self):
        self.num_items = 0
        self.ave_time = 0

    @contextmanager
    def count(self):
        t0 = time.monotonic()
        yield
        elapsed = time.monotonic() - t0
        self.num_items += 1
        self.ave_time += (elapsed - self.ave_time) / self.num_items

    def __str__(self):
        return _time_str(self.ave_time)


@contextmanager
def _sink_stats():
    get_counter = _Counter()
    put_counter = _Counter()
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


async def _sink(input_queue: AsyncQueue[T], output_queue: AsyncQueue[T]):
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
    Thrown by :py:class:`spdl.dataloader.Pipeline` when pipeline encounters an error.
    """

    def __init__(self, errs):
        msg = []
        for k, v in errs.items():
            e = str(v)
            msg.append(f"{k}:{e if e else type(v).__name__}")
        msg.sort()

        super().__init__(", ".join(msg))

        # This is for unittesting.
        self._errs = errs


async def _run_pipeline_coroutines(
    coros: list[tuple[str, Coroutine[None, None, None]]]
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
            done, pending = await asyncio.wait(
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


def _to_async(func: Callable[[T], U]) -> Callable[[T], Awaitable[U]]:
    async def afunc(item: T) -> U:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, item)

    return afunc


class PipelineBuilder:
    """**[Experimental]** Build :py:class:`~spdl.dataloader.Pipeline` object.

    See :py:class:`~spdl.dataloader.Pipeline` for details.
    """

    def __init__(self):
        self._source = None
        self._source_buffer_size = 1

        self._process_args: list[tuple[str, dict, int]] = []

        self._sink_buffer_size = None

    def add_source(
        self, source: Iterable[T] | AsyncIterable[T], **kwargs
    ) -> "PipelineBuilder":
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
        if "buffer_size" in kwargs:
            self._source_buffer_size = int(kwargs["buffer_size"])
        return self

    def pipe(
        self,
        op: (
            Callable[[T], U]
            | Callable[[T], Awaitable[U]]
            | Callable[[T], AsyncIterator[U]]
        ),
        /,
        *,
        concurrency: int = 1,
        name: str | None = None,
        hooks: Sequence[PipelineHook] | None = None,
        report_stats_interval: float | None = None,
        output_order: str = "completion",
        **kwargs,
    ) -> "PipelineBuilder":
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

                Optionally, the op can be an async function or async generator function.
                If async generator, the items are put in the output queue separately.

                .. tip::

                   When passing an async op, make sure that the op does not call sync
                   function inside.
                   If calling a sync function, use :py:func:`asyncio.loop.run_in_executor`
                   or :py:func:`asyncio.to_thread` to delegate the execution to the thread pool.
            concurrency: The maximum number of async tasks executed concurrently.
            name: The name (prefix) to give to the task.
            hooks: Hook objects to be attached to the stage. Hooks are intended for
                collecting stats of the stage.
                If ``None``, a default hook,
                :py:class:`~spdl.dataloader.TaskStatsHook` is used.
            report_stats_interval:
                The interval (in seconds) to log the stats of this stage when no
                ``hooks`` is provided. This argument is passed to
                :py:class:`~spdl.dataloader.TaskStatsHook`.
                This argument is effective only when ``hooks`` are not provided.
                If ``hooks`` is provided and stats report is needed,
                the ``hooks`` argument should
                include one of :py:class:`~spdl.dataloader.TaskStatsHook`
                instance with the desired interval.
            output_order: If ``"completion"`` (default), the items are put to output queue
                in the order their process is completed.
                If ``"input"``, then the items are put to output queue in the order given
                in the input queue.
        """
        if output_order not in ["completion", "input"]:
            raise ValueError(
                '`output_order` must be either "completion" or "input". '
                f"Found: {output_order}"
            )

        if name is None:
            if hasattr(op, "__name__"):
                name = op.__name__  # type: ignore[attr-defined]
            else:
                name = op.__class__.__name__

        if inspect.iscoroutinefunction(op):
            pass
        elif inspect.isgeneratorfunction(op):
            raise ValueError("pipe does not support generator function.")
        elif inspect.isasyncgenfunction(op):
            if output_order == "input":
                raise ValueError(
                    "pipe does not support async generator function "
                    "when output_order is 'input'."
                )
        else:
            op = _to_async(op)  # pyre-ignore: [6]

        self._process_args.append(
            (
                "pipe" if output_order == "completion" else "ordered_pipe",
                {
                    "afunc": op,
                    "concurrency": concurrency,
                    "name": name,
                    "hooks": hooks,
                    "report_stats_interval": report_stats_interval,
                },
                kwargs.get("buffer_size", 1),
                # Note:
                # `buffer_size` option is intentionally not documented.
                #
                # The pipeline practically buffers `concurrency + buffer_size`
                # items, which leads to confusing behavior when writing tests.
                #
                # And it does not affect throughput, however, showing it as
                # `buffer_size=1` often make users think that this needs to be
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
        num_aggregate: int,
        /,
        *,
        drop_last: bool = False,
        hooks: Sequence[PipelineHook] | None = None,
        report_stats_interval: float | None = None,
    ) -> "PipelineBuilder":
        """Buffer the items in the pipeline.

        Args:
            num_aggregate: The number of items to buffer.
            drop_last: Drop the last aggregation if it has less than ``num_aggregate`` items.
            hooks: See :py:meth:`pipe`.
            report_stats_interval: See :py:meth:`pipe`.
        """
        vals = [[]]

        async def aggregate(i):
            if i is not _EOF:
                vals[0].append(i)

            if (i is _EOF and vals[0]) or (len(vals[0]) >= num_aggregate):
                ret = vals.pop(0)
                vals.append([])
                return ret
            return _SKIP

        name = f"aggregate({num_aggregate}, {drop_last=})"

        self._process_args.append(
            (
                "aggregate",
                {
                    "afunc": aggregate,
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

    def add_sink(self, buffer_size: int) -> "PipelineBuilder":
        """Attach a buffer to the end of the pipeline.

        .. code-block::

            │
           ┌▼┐
           │ │ buffer
           └─┘

        Args:
            buffer_size: The size of the buffer
        """
        if self._sink_buffer_size is not None:
            raise ValueError("Sink is already set.")
        self._sink_buffer_size = buffer_size
        return self

    def _build(
        self,
        num_items: int | None,
        # TODO: Once we remove AsyncPipeline, construct queues internally.
        queues: list[AsyncQueue],
    ) -> Coroutine[None, None, None]:
        if self._source is None:
            raise ValueError("Source is not set.")
        if num_items is not None and num_items < 1:
            raise ValueError("num_items must be >= 0")

        construct_queues = len(queues) == 0

        # Note:
        # Make sure that coroutines are ordered from source to sink.
        # `_run_pipeline_coroutines` expects and rely on this ordering.
        coros = []

        # source
        if construct_queues:
            queues.append(AsyncQueue(self._source_buffer_size))

        coros.append(
            (
                "AsyncPipeline::0_source",
                _enqueue(self._source, queues[0], max_items=num_items),
            )
        )

        # pipes
        for i, (type_, args, buffer_size) in enumerate(self._process_args, start=1):
            if construct_queues:
                queues.append(AsyncQueue(buffer_size))
            in_queue, out_queue = queues[i - 1 : i + 1]

            match type_:
                case "pipe" | "aggregate":
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
            if construct_queues:
                queues.append(AsyncQueue(self._sink_buffer_size))

            coros.append(
                (
                    f"AsyncPipeline::{len(self._process_args) + 1}_sink",
                    _sink(*queues[-2:]),
                )
            )

        return _run_pipeline_coroutines(coros)

    def _get_desc(self) -> list[str]:
        parts = []
        parts.append(f"  - src: {self._source}")
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
        return "\n".join([repr(self)] + self._get_desc())

    def build(self, *, num_threads: int | None = None) -> Pipeline:
        """Build the pipeline.

        Args:
            num_threads: The number of threads in the thread pool attached to
                async event loop.
                If not specified, the maximum concurrency value is used.
        """
        queues = []
        coro = self._build(None, queues)

        if num_threads is None:
            concurrencies = [
                args["concurrency"]
                for _, args, _ in self._process_args
                if "concurrency" in args
            ]
            num_threads = max(concurrencies) if concurrencies else 4
        assert num_threads is not None
        return Pipeline(coro, queues, num_threads, desc=self._get_desc())
