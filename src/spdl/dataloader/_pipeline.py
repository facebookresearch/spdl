# pyre-unsafe

import asyncio
import logging
from asyncio import AbstractEventLoop as EventLoop, Event as AsyncEvent, Queue
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from typing import Generic, TypeVar

from . import _utils

__all__ = []

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


################################################################################
# AsyncPipeline
################################################################################
class AsyncPipelineImpl(Generic[T]):
    """class AsyncPipelineImpl()

    Use :py:class:`PipelineBuilder` to instantiate one.
    """

    def __init__(
        self,
        loop: EventLoop,
        coro: Awaitable,
        queues: list[Queue],
        stop_requested: AsyncEvent,
        *,
        desc: list[str],
    ):
        self._loop = loop
        self._coro = coro
        self._queues = queues
        self._stop_requested = AsyncEvent()
        self._str = "\n".join([repr(self)] + desc)

        self._output_queue = queues[-1]
        self._thread = _utils._EventLoopThread(loop)

        try:
            from spdl.lib import _libspdl

            _libspdl.log_api_usage("spdl.dataloader.AsyncPipelineImpl")
        except Exception:
            pass  # ignore if not supported.

    def __str__(self) -> str:
        return self._str

    def start(self) -> None:
        """Start the pipeline in background thread."""
        if not self._thread.is_alive():
            _LG.info("Starting the pipeline thread.")
            self._thread.start()

            asyncio.run_coroutine_threadsafe(self._coro, loop=self._loop)

    def stop(self, *, timeout: float | None = None) -> None:
        """Stop the pipeline.

        Args:
            timeout: Timeout value used when stopping the pipeline and
                waiting for the thread to join.
        """
        if not self._thread.is_alive():
            return

        _LG.info("Stopping the pipeline thread.")
        self._stop_requested.set()
        _utils._stop_loop(self._loop)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError(f"Thread did not join after {timeout} seconds.")
        self._loop.close()
        _LG.info("The pipeline thread is stopped.")

    @contextmanager
    def auto_stop(self, *, timeout: float | None = None):
        """Context manager to start/stop the background thread automatically.

        Args:
            timeout: Timeout value used when stopping the thread.
        """
        self.start()
        try:
            yield
        finally:
            self.stop(timeout=timeout)

    def get_item(self, *, timeout: float | None = None) -> T:
        """Get the next item.

        If pipeline is not producing the next item within the given timeout,
        then ``TimeoutError`` is raised.
        If the background thread is not running and the queue is empty, then
        ``EOFError`` is raised.

        Args:
            timeout: Timeout for each iteration.

        Raises:
            - If pipeline is not producing the next item within the given timeout,
              then ``TimeoutError`` is raised.
            - If the background thread is not running and the queue is empty, then
              ``EOFError`` is raised.
        """
        if self._stop_requested.is_set():
            # The background thread has been stopped. Either cancellation or EOF acked.

            # If the background thread has been stopped by user, then the queue might contain
            # some items.
            if not self._output_queue.empty():
                if (item := self._output_queue.get_nowait()) is not _EOF:
                    return item

            raise EOFError("Reached the end of the pipeline.")
        elif not self._thread.is_alive():
            # The background thread is not started.
            raise RuntimeError("Pipeline is not started.")

        item = _utils._run_coro_threadsafe(
            self._loop, self._output_queue.get(), "output_queue.get()", timeout=timeout
        )
        if item is _EOF:
            self._stop_requested.set()
            raise EOFError("Reached the end of the pipeline.")
        return item

    def get_iterator(self, *, timeout: float | None = None) -> Iterator[T]:
        """Get an iterator, which iterates over the pipeline outputs.

        Args:
            timeout: Timeout for each iteration.
        """
        return AsyncPipelineIterator(self, timeout)


class AsyncPipelineIterator(Generic[T]):
    """AsyncPipelineIterator()"""

    def __init__(self, pipeline: AsyncPipelineImpl[T], timeout):
        self._pipeline = pipeline
        self._timeout = timeout

    def __iter__(self):
        return self

    def __next__(self) -> T:
        try:
            return self._pipeline.get_item(timeout=self._timeout)
        except EOFError:
            raise StopIteration from None
