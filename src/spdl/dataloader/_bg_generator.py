import asyncio
import logging
import traceback
import warnings
from collections.abc import AsyncIterator, Iterator
from queue import Empty, Queue
from threading import Event, Thread
from typing import Generic, TypeVar

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


def _run_agen(
    loop,
    aiterable: AsyncIterator,
    sentinel: _Sentinel,
    queue: Queue,
    stopped: Event,
    timeout: int | float | None,
):
    async def _generator_loop():
        while True:
            try:
                item = await asyncio.wait_for(anext(aiterable), timeout)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                _LG.warning("The background generator timed out.")
                sentinel.err_msg = f"Timed out after {timeout} seconds."
                return
            except Exception as err:
                _LG.error(f"The backgroud generator failed with {err}.")
                sentinel.err_msg = traceback.print_exception(err)
                return
            else:
                queue.put(item)

            if stopped.is_set():
                _LG.debug("Stop requested.")
                break

    try:
        loop.run_until_complete(_generator_loop())
    finally:
        queue.put(sentinel)
        stopped.set()


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

        timeout: The maximum time to wait for the generator to yield an item.
            If the generator does not yield an item within this time, `TimeoutError`
            is raised.

            This parameter is intended for breaking unforseen situations where
            the background generator is stuck for some reasons.

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
        iterable: AsyncIterator[T],
        *,
        num_workers: int | None = 3,
        queue_size: int = 10,
        timeout: int | float | None = 300,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.iterable = iterable
        self.queue_size = queue_size
        self.loop = _get_loop(num_workers) if loop is None else loop
        self.timeout = timeout

    def __iter__(self) -> Iterator[T]:
        queue = Queue(maxsize=self.queue_size)
        # Used to indicate the end of the queue.
        sentinel = _Sentinel()
        # Used to flag cancellation from outside and to
        # flag the completion from the inside.
        stopped = Event()

        thread = Thread(
            target=_run_agen,
            args=(self.loop, self.iterable, sentinel, queue, stopped, self.timeout),
        )

        with _thread_manager(thread, stopped, queue, sentinel):
            while True:
                try:
                    item = queue.get(timeout=self.timeout)
                except Empty:
                    # Foreground thread timeout, meaning the background generator is
                    # too slow or stuck.
                    raise TimeoutError(
                        "The background generator did not yield an item "
                        f"within {self.timeout} seconds."
                    ) from None
                else:
                    if item is not sentinel:
                        yield item
                    elif item.err_msg is not None:
                        # Something wrong happened in the background thread, including timeout
                        raise RuntimeError(
                            f"Error occured in background thread: {item.err_msg}"
                        )
                    else:
                        return


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
