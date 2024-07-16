# pyre-unsafe

import asyncio
import logging
from asyncio import AbstractEventLoop as EventLoop, Event as AsyncEvent, Queue
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from threading import Thread
from typing import TypeVar

from ._hook import _stage_hooks, _task_hooks, PipelineHook, TaskStatsHook
from ._utils import _get_loop, create_task

__all__ = ["AsyncPipeline", "AsyncPipeline2", "PipelineFailure", "PipelineBuilder"]

_LG = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


# Sentinel objects used to instruct AsyncPipeline to take special actions.
_EOF = object()  # Indicate the end of stream.
_SKIP = object()  # Indicate that there is no data to process.


@asynccontextmanager
async def _put_eof_when_done(queue, put_on_error=False):
    try:
        yield
    except Exception:
        if put_on_error:
            await queue.put(_EOF)
        else:
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
    @_put_eof_when_done(queue, put_on_error=True)
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


################################################################################
# AsyncPipeline
################################################################################


def _run_coro_with_cancel(loop, coro, cancelled: AsyncEvent, name: str):
    async def _run():
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
                await asyncio.wait_for(cancelled.wait(), timeout=0.1)
            except TimeoutError:
                if task.done():
                    return
            else:
                task.cancel()
                await task
                return

    loop.run_until_complete(_run())


class AsyncPipeline2:
    """AsyncPipeline2()

    [Experimental][WIP] An alternative implementation of :py:class:`~spdl.dataloader.AsyncPipeline`,
    which will replace ``AsyncPipeline`` soon.

    Use :py:class:`PipelineBuilder` to instantiate one.
    """

    def __init__(
        self,
        queues: list[Queue],
        coro: Awaitable,
        loop: EventLoop,
        _str: str,
    ):
        self._queues = queues
        self._coro = coro
        self._str = _str
        self._loop = loop

        self._output_queue = queues[-1]

        self._cancelled = AsyncEvent()
        self._thread = Thread(
            target=_run_coro_with_cancel,
            args=[loop, coro, self._cancelled, "AsyncPipeline::run"],
        )

    def __str__(self) -> str:
        return self._str

    def _run_coro(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _run_func(self, func, /, *args, **kwargs):
        return self._run_coro(asyncio.to_thread(func, *args, **kwargs))

    def start(self):
        """Start the pipeline in background thread."""
        if not self._thread.is_alive():
            _LG.info("Starting the pipeline thread.")
            self._thread.start()

    def stop(self, timeout=None):
        """Stop the pipeline."""
        if self._thread.is_alive():
            _LG.info("Stopping the pipeline thread.")
            self._run_func(self._cancelled.set).result(timeout=timeout)
            self._thread.join(timeout=timeout)
            _LG.info("The pipeline thread is stopped.")

    @contextmanager
    def auto_stop(self, timeout=None):
        """Context manager to start/stop the background thread automatically."""
        self.start()
        try:
            yield
        finally:
            self.stop(timeout=timeout)

    def get(self, timeout=None):
        """Get the next item"""
        return self._run_coro(self._output_queue.get()).result(timeout=timeout)


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

        # 1. check what kind of errors occured
        # Note:
        # By the above assumption about the cancellation, the following code assumes that
        # CancelledError does not happen for done tasks.
        errs = {}
        for task in done:
            # Note:
            # `exception()` method can throw `CancelledError` or `InvalidStateError`,
            # both of which are assumed to not happen here.
            if (err := task.exception()) is not None:
                errs[task.get_name()] = err

        # 2. if a failure presents, cancel the remaining tasks
        if errs:
            if pending:
                for task in pending:
                    task.cancel()

                done, _ = await asyncio.wait(pending)

                for task in done:
                    try:
                        err = task.exception()
                    except asyncio.CancelledError:
                        errs[task.get_name()] = "Cancelled"
                    else:
                        errs[task.get_name()] = err

            raise PipelineFailure(errs)


class PipelineBuilder:
    """[Experimental] Build pipeline."""

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

    def __str__(self) -> str:
        parts = [repr(self)]
        parts.append(f"  - src: {self._source}")
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

        return "\n".join(parts)

    def build(self, num_threads: int = 4) -> AsyncPipeline2:
        """Build the pipeline."""
        queues = []
        coros = self._build(None, queues)
        loop = _get_loop(num_threads)

        return AsyncPipeline2(queues, coros, loop, str(self))


class AsyncPipeline:
    """**[Experimental]** Construct data processing pipeline.

    ``AsyncPipeline`` facilitates building data processing pipeline consists of multiple
    stages of async operations. It allows to configure the concurrency of each stage
    independently.

    Typically, the source is a lightweight (synchronous) iterator that generates the
    source location of data, such as file paths and URLs. The first stage op
    retrieves data from corresponding (network) storage. The subsequent stages
    process the data, such as decoding and resizing images, or decoding and resampling
    audio.
    After the preprocessings are done, the data are passed to the sink, which is
    (synchronous) queue.

    The following diagram illustrates this.

    .. mermaid::

       flowchart TD
           Source["Source (Iterator)"]
           Queue
           subgraph Op1["Op1 (Concurrency = 4)"]
               op1_1(Task 1-1)
               op1_2(Task 1-2)
               op1_3(Task 1-3)
               op1_4(Task 1-4)
           end
           subgraph Op2["Op2 (Concurrency=2)"]
               op2_1(Task 2-1)
               op2_2(Task 2-2)
           end
           Queue["Sink (Queue)"]

           Source --> Op1
           Op1 --> Op2
           Op2 --> Queue

    .. admonition:: Example: Bulk loading images

        .. code-block::

           import asyncio
           from queue import Queue

           import spdl.io

           def source():
               with open("images.txt") as f:
                   for path in f:
                       yield path

           async def decode(path):
               return await spdl.io.async_decode_image(path)

           queue = Queue()

           pipeline = (
               AsyncPipeline()
               .add_source(source())
               .pipe(decode, concurrency=10)
               .add_sink(queue)
           )

           loop = asyncio.new_event_loop()
           loop.set_default_executor(
               ThreadPoolExecutor(
                   max_workers=10,
               )
           )
           loop.run_until_complete(pipeline.run())
    """

    def __init__(self):
        self._builder = PipelineBuilder()

        self._queues: list[Queue] = []

        try:
            from spdl.lib import _libspdl

            _libspdl.log_api_usage("spdl.dataloader.AsyncPipeline")
        except Exception:
            pass  # ignore if not supported.

    @property
    def output_queue(self) -> Queue:
        """The output queue of the pipeline."""
        if not self._queues:
            raise ValueError("No output queue is set.")
        return self._queues[-1]

    def add_source(self, source: Iterator[T], **kwargs) -> "AsyncPipeline":
        """Attach an iterator to the source buffer.

        .. code-block::

           ┌─────────────────┐
           │ Iterator (sync) │
           └───────┬─────────┘
                   │
                  ┌▼┐
                  │ │
                  │ │ Queue
                  │ │
                  └─┘

        Args:
            source: A lightweight iterator that generates data.

                .. warning::

                   The source iterator must be lightweight as it is executed in async
                   event loop. If the iterator performs a an operation that blocks,
                   the entire pipeline will be blocked.
        """
        self._builder.add_source(iter(source), **kwargs)
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
    ) -> "AsyncPipeline":
        """Apply an async function to items in the pipeline.

        .. code-block::

                  ┌─┐
                  │ │
                  │ │ Queue
                  │ │
                  └┬┘
                   │
           ┌───────▼────────┐
           │ Async Function │
           └───────┬────────┘
                   │
                  ┌▼┐
                  │ │
                  │ │ Queue
                  │ │
                  └─┘

        Args:
            afunc: Async function applied to the items in the queue.
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
        self._builder.pipe(
            afunc,
            concurrency=concurrency,
            name=name,
            hooks=hooks,
            report_stats_interval=report_stats_interval,
            output_order=output_order,
            **kwargs,
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
    ) -> "AsyncPipeline":
        """Buffer the items in the pipeline.

        Args:
            num_aggregate: The number of items to buffer.
            drop_last: Drop the last aggregation if it has less than ``n`` items.
            hooks: See :py:meth:`pipe`.
            report_stats_interval: See :py:meth:`pipe`.
        """
        self._builder.aggregate(
            num_aggregate,
            drop_last=drop_last,
            hooks=hooks,
            report_stats_interval=report_stats_interval,
        )
        return self

    def add_sink(self, buffer_size: int) -> "AsyncPipeline":
        """Attach a queue to the end of the pipeline.

        .. code-block::

           ┌─┐
           │ │
           │ │ Queue
           │ │
           └┬┘
            │
           ┌▼┐
           │ │
           │ │ Queue
           │ │
           └─┘

        Args:
            buffer_size: The size of the last queue.
        """
        self._builder.add_sink(buffer_size)
        return self

    def __str__(self) -> str:
        return str(self._builder)

    def _build(self, num_items):
        return self._builder._build(num_items, queues=self._queues)

    # TODO [Python 3.11]: Try TaskGroup
    async def run(self, *, num_items: int | None = None) -> None:
        """Run the pipeline until its completion. All stages are executed concurrently.

        The pipeline completes when one of the following conditions is met.

        1. Source is exhauseted and all data went through all the stages.
        2. One or more stages encounter an internal error*. In this case, the remaining
           stages are cancelled in attempt at graceful shutdown.
        3. The pipeline is cancelled. All the stages are cancelled in attempt at graceful
           shutdown

        .. admonition:: *Internal Error

           The internal error here refers to the failure happens in the execution
           path of ``AsyncPipeline``, but outside of user provided functions.
           Therefore, it does not include the errors occur in source iterators
           and async ops. For example, data acquisition failure due to network issue
           or decoding failures.
           Errors happen inside of user-provided functions are simply logged and ignored.

        Args:
            num_items: *Optional:* The maximum number of items to process.
                If ``None``, the pipeline runs until the source is exhausted.

        Raises:

            PipelineFailure: Raised when a part of the pipeline has an error.
        """
        await self._build(num_items)
