# pyre-unsafe

import asyncio
import logging
import traceback
import warnings
from collections.abc import Iterator
from queue import Empty, Queue
from threading import Event, Thread
from typing import Generic, TypeVar

from ._pipeline import AsyncPipeline
from ._utils import _get_loop

_LG = logging.getLogger(__name__)

__all__ = [
    "BackgroundGenerator",
]

T = TypeVar("T")


################################################################################
# Impl for BackgroundGenerator
################################################################################
class _Sentinel:
    """Sentinel object to indicate the end of iteration."""

    def __init__(self):
        self.err_msg: str | None = None


def _run_pipeline(
    loop,
    coro,
    sentinel: _Sentinel,
    queue: Queue,
    stopped: Event,
):
    async def _run():
        task = asyncio.create_task(coro)
        tasks = [task]

        while tasks:
            done, tasks = await asyncio.wait(tasks, timeout=0.1)

            if done:
                if (err := task.exception()) is not None:
                    _LG.error(f"The background generator failed with {err}.")
                    sentinel.err_msg = traceback.print_exception(err)
                return

            if stopped.is_set():
                _LG.debug("Stop requested.")
                task.cancel()
                return

    try:
        loop.run_until_complete(_run())
    finally:
        queue.put(sentinel)
        stopped.set()


class _thread_manager:
    """Context manager to stop and join the background thread."""

    def __init__(
        self,
        thread: Thread,
        stopped: Event,
        queue: Queue,
        sentinel: object,
        timeout: int | float | None,
    ):
        self.thread = thread
        self.stopped = stopped
        self.queue = queue
        self.sentinel = sentinel
        self.timeout = timeout

    def __enter__(self):
        self.thread.start()

    def __exit__(self, exc_type, exc_value, traceback):
        # The foreground thread exited, and the queue is no longer consumerd.
        #
        # If stopped is not set, the background thread is still running, putting
        # items in queue.
        # The queue might get clogged, and it can block the background thread.
        # In turn, it prevent the bg thread from joining.
        # Therefore, we need to flush the queue.
        #
        # If stopped is set, the background thread is completed.
        # No more items are generated, and the sentinel value might have been consumed.
        # So we cannot/don't flush the queue.
        if not self.stopped.is_set():
            _LG.info("Stopping the background thread.")
            self.stopped.set()
            self._flush()

        _LG.info("Waiting for the background thread to join.")
        self.thread.join(self.timeout)
        if self.thread.is_alive():
            _LG.warning(
                "The background thread did not join after %f seconds.", self.timeout
            )
        else:
            _LG.info("The background thread joined.")

    def _flush(self):
        _LG.debug("Flushing the queue.")
        while (_ := self.queue.get()) is not self.sentinel:
            pass


class BackgroundGenerator(Generic[T]):
    """**[Experimental]** Run generator in background and iterate the items.

    Args:
        pipeline: Pipeline to run in the background.

        num_workers: The number of worker threads to be attached to the event loop.
            If ``loop`` is provided, this argument is ignored.

        queue_size: The size of the queue that is used to pass the
            generated items from the background thread to the main thread.
            If the queue is full, the background thread will be blocked.

        timeout: The maximum time to wait for the generator to yield an item.
            If the generator does not yield an item within this time, ``TimeoutError``
            is raised.

            This parameter is intended for breaking unforseen situations where
            the background generator is stuck for some reasons.

            It is also used for timeout when waiting for the background thread to join.

        loop: If provided, use this event loop to execute the generator.
            Otherwise, a new event loop will be created. When providing a loop,
            ``num_workers`` is ignored, so the executor must be configured by
            client code.

    .. admonition:: Example

       >>> apl = (
       >>>     spdl.dataloader.AsyncPipeline()
       >>>     .add_source(iter(range(10)))
       >>> )
       >>>
       >>> processor = BackgroundGenerator(apl)
       >>> for item in processor.run(3):
       >>>     # Do something with the item.
    """

    def __init__(
        self,
        pipeline: AsyncPipeline,
        *,
        num_workers: int | None = 3,
        queue_size: int = 10,
        timeout: int | float | None = 300,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.pipeline = pipeline
        self.queue_size = queue_size
        self.loop = _get_loop(num_workers) if loop is None else loop
        self.timeout = timeout

        self.queue = Queue(maxsize=self.queue_size)
        self.pipeline.add_sink(self.queue)

        try:
            from spdl.lib import _libspdl

            _libspdl.log_api_usage("spdl.dataloader.BackgroundGenerator")
        except Exception:
            pass  # ignore if not supported.

    def __iter__(self) -> Iterator[T]:
        """Run the generator in background thread and iterate the result.

        Yields:
            Items generated by the provided generator.
        """
        warnings.warn(
            "`BackgroundGenerator.__iter__` has been deprecated. "
            "Please use `BackgroundGenerator.run()`.",
            category=FutureWarning,
            stacklevel=2,
        )
        return self.run()

    def run(self, num_items: int | None = None) -> Iterator[T]:
        """Run the generator in background thread and iterate the result.

        Args:
            num_items: The number of items to yield. If omitted, the generator
                will be iterated until the end.

        Yields:
            Items generated by the provided generator.
        """
        coro = self.pipeline.run(num_items=num_items)

        # Used to indicate the end of the queue.
        sentinel = _Sentinel()
        # Used to flag cancellation from outside and to
        # flag the completion from the inside.
        stopped = Event()

        thread = Thread(
            target=_run_pipeline,
            args=(self.loop, coro, sentinel, self.queue, stopped),
        )

        with _thread_manager(thread, stopped, self.queue, sentinel, self.timeout):
            while True:
                try:
                    item = self.queue.get(timeout=self.timeout)
                except Empty:
                    # Foreground thread timeout, meaning the background pipeline is
                    # too slow or stuck.
                    raise TimeoutError(
                        "The background pipeline did not yield an item "
                        f"within {self.timeout} seconds."
                    ) from None
                else:
                    # Note:
                    # In the original implementation, the sentinel value was compared using
                    # identity checking (`is` operator). But strangely, this stopped working in
                    # https://github.com/facebookresearch/spdl/pull/32 .
                    #
                    # The sentinel object passed to the background thread and came back through
                    # a queue is no longer the same object. So now we use `isinstance`.
                    if not isinstance(item, _Sentinel):
                        yield item
                    elif item.err_msg is None:
                        # Sane termination
                        return
                    else:
                        # Something wrong happened in the background thread, including timeout
                        raise RuntimeError(
                            f"Error occured in background thread: {item.err_msg}"
                        )
