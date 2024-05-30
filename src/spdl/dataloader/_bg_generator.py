import asyncio
import logging
import warnings
from collections.abc import AsyncIterable, Iterator
from queue import Queue
from threading import Event, Thread
from typing import Generic, TypeVar

from ._utils import _get_loop

_LG = logging.getLogger(__name__)

__all__ = [
    "BackgroundGenerator",
]

T = TypeVar("T")


################################################################################
# Impl for AsyncTaskRunner
################################################################################
def _run_agen(loop, aiterable, sentinel, queue, stopped):
    async def _generator_loop():
        try:
            async for item in aiterable:
                queue.put(item)

                if stopped.is_set():
                    _LG.debug("Stop requested.")
                    break
        finally:
            queue.put(sentinel)
            stopped.set()
        _LG.debug("exiting _generator_loop.")

    loop.run_until_complete(_generator_loop())


class _thread_manager:
    """Context manager to stop and join the background thread."""

    def __init__(self, thread: Thread, stopped: Event, queue: Queue, sentinel: object):
        self.thread = thread
        self.stopped = stopped
        self.queue = queue
        self.sentinel = sentinel

    def __enter__(self):
        self.thread.start()

    def __exit__(self, exc_type, exc_value, traceback):
        # If stopped is not set, the background thread is still running,
        # but queue is no longer consumed.
        # The queue might get clogged which can block the background thread and
        # prevent it from joining. Therefore, we need to flush the queue.
        #
        # If stopped is set, the background thread is completed.
        # No more items are generated, and the sentinel value might have been consumed.
        # So we cannot/don't flush the queue.
        if not self.stopped.is_set():
            _LG.info("Stopping the background thread.")
            self.stopped.set()
            self._flush()

        _LG.info("Waiting for the background thread to join.")
        self.thread.join()

    def _flush(self):
        _LG.debug("Flushing the queue.")
        while (_ := self.queue.get()) is not self.sentinel:
            pass


class BackgroundGenerator(Generic[T]):
    """Run generator in background and iterate the items.

    Args:
        iterable: Generator to run in the background. It can be an
            asynchronous generator or a regular generator.

        num_workers: The number of worker threads in the default thread executor.
            If `loop` is provided, this argument is ignored.

        queue_size: The size of the queue that is used to pass the
            generated items from the background thread to the main thread.
            If the queue is full, the background thread will be blocked.

        loop: If provided, use this event loop to execute the generator.
            Otherwise, a new event loop will be created. When providing a loop,
            `num_workers` is ignored, so the executor must be configured by
            client code.

    ??? note "Example"

        ```python
        async def generator():
            for i in range(10):
                yield i

        processor = BackgroundGenerator(generator())
        for item in processor:
            # Do something with the item.
        ```
    """

    def __init__(
        self,
        iterable: AsyncIterable[T],
        *,
        num_workers: int | None = 3,
        queue_size: int = 10,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.iterable = iterable
        self.queue_size = queue_size
        self.loop = _get_loop(num_workers) if loop is None else loop

    def __iter__(self) -> Iterator[T]:
        queue = Queue(maxsize=self.queue_size)
        # Used to indicate the end of the queue.
        sentinel = object()
        # Used to flag cancellation from outside and to
        # flag the completion from the inside.
        stopped = Event()

        thread = Thread(
            target=_run_agen,
            args=(self.loop, self.iterable, sentinel, queue, stopped),
        )

        with _thread_manager(thread, stopped, queue, sentinel):
            while (item := queue.get()) is not sentinel:
                yield item


class BackgroundTaskProcessor(BackgroundGenerator):
    """Deprecated. Use BackgroundGenerator instead."""

    def __init__(self, *args, **kwargs):
        message = "BackgroundTaskProcessor has been renamed to BackgroundGenerator."
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)

    def __enter__(self):
        message = (
            "Background thread management has been moved to __iter__. "
            f"You can now directly iterate on {type(self).__name__}, and "
            "get rid of the context manager. "
        )
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
