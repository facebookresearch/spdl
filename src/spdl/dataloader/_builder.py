# pyre-unsafe

import asyncio
import inspect
import logging
import warnings
from asyncio import Event as AsyncEvent, Queue
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import TypeVar

from . import _utils
from ._hook import _stage_hooks, _task_hooks, PipelineHook, TaskStatsHook
from ._pipeline import _EOF, _SKIP, AsyncPipelineImpl
from ._utils import create_task

__all__ = ["PipelineFailure", "PipelineBuilder"]

_LG = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


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
    input_queue: Queue[T],
    afunc: Callable[[T], Awaitable[U]],
    output_queue: Queue[U],
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

    if not inspect.iscoroutinefunction(afunc):
        warnings.warn(f"`pipe` expects async function, but {afunc=} is not coroutine.")

    hooks = (
        [TaskStatsHook(name, concurrency, interval=report_stats_interval)]
        if hooks is None
        else hooks
    )

    async def _wrap(coro: Awaitable[U]) -> None:
        async with _task_hooks(hooks):  # pyre-ignore: [16]
            result = await coro

        await output_queue.put(result)

    @_stage_hooks(hooks)
    @_put_eof_when_done(output_queue)
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
    input_queue: Queue[T],
    afunc: Callable[[T], Awaitable[U]],
    output_queue: Queue[U],
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
                  │ │ Queue: Input
                  │ │
                  └┬┘
                   │
           ┌───────▼────────┐
           │ Async Function │
           └───────┬────────┘
                  ┌▼┐
                  │ │
                  │ │ Queue: Intermediate queue:
                  │ │ contains tasks. queue size == concurrency
                  └┬┘
           ┌───────▼────────┐
           │     enqueue    │
           └───────┬────────┘
                  ┌▼┐
                  │ │
                  │ │ Queue: Output
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

    inter_queue = Queue(concurrency)

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

    @_stage_hooks(hooks)
    @_put_eof_when_done(output_queue)
    async def ordered_pipe():
        tasks = {create_task(coro1), create_task(coro2)}
        await asyncio.wait(tasks)

    return ordered_pipe()


################################################################################
# _enqueue
################################################################################


def _enqueue(
    iterator: Iterator[T],
    queue: Queue[T],
    max_items: int | None = None,
) -> Coroutine:
    @_put_eof_when_done(queue)
    async def enqueue():
        num_items = 0
        for item in iterator:
            if item is not _SKIP:
                await queue.put(item)
                num_items += 1
                if max_items is not None and num_items >= max_items:
                    return

    return enqueue()


################################################################################
# _dequeue
################################################################################


async def _dequeue(
    input_queue: Queue[T],
    output_queue: Queue[T],
):
    while (item := await input_queue.get()) is not _EOF:
        if item is not _SKIP:
            await output_queue.put(item)


# TODO [Python 3.11]: Migrate to ExceptionGroup
class PipelineFailure(Exception):
    """PipelineFailure()
    Thrown by :py:meth:`spdl.dataloader.AsyncPipeline.run`
    when pipeline encounters an error."""

    def __init__(self, errs):
        msg = []
        for k, v in errs.items():
            e = str(v)
            msg.append(f"{k}:{e if e else type(v).__name__}")
        msg.sort()

        super().__init__(self, ", ".join(msg))

        # This is for unittesting.
        self._errs = errs


async def _run_coroutines(coros) -> None:
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


def _run_coro_with_cancel(loop, coro, queue, stop_requested: AsyncEvent, name: str):

    @_put_eof_when_done(queue)
    async def run_coro_with_cancel():
        task = create_task(coro, name=name)

        while True:
            # Note:
            # There is nothing need to be done when pipeline execution
            # changes its state (successful completion / failure).
            # However, if a cancellation is requested, we want to react
            # and cancel the pipeline a.s.a.p.
            # For this reason, we wait on cancellation event, rather than
            # waiting on the pipeline execution.
            try:
                await asyncio.wait_for(stop_requested.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass
            else:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                return

            if task.done():
                return

        # Wait for the foreground to acknowledge that EOF was received.
        await stop_requested.wait()

    return run_coro_with_cancel()


class PipelineBuilder:
    """**[Experimental]** Build pipeline."""

    def __init__(self):
        self._source = None
        self._source_buffer_size = 1

        self._process_args: list[tuple[str, dict, int]] = []

        self._sink_buffer_size = None

    def add_source(self, source: Iterable[T], **kwargs) -> "PipelineBuilder":
        """See :py:meth:`spdl.dataloader.AsyncPipeline.add_source`."""
        if self._source is not None:
            raise ValueError("Source already set.")
        self._source = source

        # Note: Do not document this option.
        # See `pipe` method for detail.
        if "buffer_size" in kwargs:
            self._source_buffer_size = int(kwargs["buffer_size"])
        return self

    def pipe(
        self,
        afunc: Callable[[T], Awaitable[U]],
        *,
        concurrency: int = 1,
        name: str | None = None,
        hooks: Sequence[PipelineHook] | None = None,
        report_stats_interval: float | None = None,
        output_order: str = "completion",
        **kwargs,
    ) -> "PipelineBuilder":
        """See :py:meth:`spdl.dataloader.AsyncPipeline.pipe`."""
        if output_order not in ["completion", "input"]:
            raise ValueError(
                '`output_order` must be either "completion" or "input". '
                f"Found: {output_order}"
            )

        if name is None:
            if hasattr(afunc, "__name__"):
                name = afunc.__name__
            else:
                name = afunc.__class__.__name__

        self._process_args.append(
            (
                "pipe" if output_order == "completion" else "ordered_pipe",
                {
                    "afunc": afunc,
                    "concurrency": concurrency,
                    "name": name,
                    "hooks": hooks,
                    "report_stats_interval": report_stats_interval,
                },
                # Note:
                # `buffer_size` option is an intentionally not documented.
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
                kwargs.get("buffer_size", 1),
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
        """See :py:meth:`spdl.dataloader.AsyncPipeline.aggregate`."""
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
        """See :py:meth:`spdl.dataloader.AsyncPipeline.add_sink`."""
        self._sink_buffer_size = buffer_size
        return self

    def _build(self, num_items: int | None, queues: list[Queue]) -> Awaitable[None]:
        if self._source is None:
            raise ValueError("Source is not set.")
        if num_items is not None and num_items < 1:
            raise ValueError("num_items must be >= 0")

        construct_queues = len(queues) == 0

        coros = []

        # source
        if construct_queues:
            queues.append(Queue(self._source_buffer_size))

        coros.append(
            (
                "AsyncPipeline::0_source",
                _enqueue(self._source, queues[0], max_items=num_items),
            )
        )

        # pipes
        for i, (type_, args, buffer_size) in enumerate(self._process_args, start=1):
            if construct_queues:
                queues.append(Queue(buffer_size))
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

        if self._sink_buffer_size is not None:
            if construct_queues:
                queues.append(Queue(self._sink_buffer_size))

            coros.append(
                (
                    f"AsyncPipeline::{len(self._process_args) + 1}_sink",
                    _dequeue(*queues[-2:]),
                )
            )

        return _run_coroutines(coros)

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

    def build(self, *, num_threads: int = 4) -> AsyncPipelineImpl:
        """Build the pipeline."""
        queues = []
        coro = self._build(None, queues)
        loop = _utils._get_loop(num_threads)
        stop_requested = AsyncEvent()
        coro = _run_coro_with_cancel(
            loop, coro, queues[-1], stop_requested, name="AsyncPipeline::main"
        )

        return AsyncPipelineImpl(
            loop, coro, queues, stop_requested, desc=self._get_desc()
        )
