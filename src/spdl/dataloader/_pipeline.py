# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import asyncio
import concurrent.futures
import logging
import time
import warnings
from asyncio import Queue as AsyncQueue
from collections.abc import Coroutine, Iterator
from contextlib import contextmanager
from threading import Event as SyncEvent, Thread
from typing import Generic, TypeVar

from ._utils import create_task

__all__ = ["Pipeline"]

_LG = logging.getLogger(__name__)

T = TypeVar("T")


##############################################################################
# _EventLoop (Thread)
##############################################################################


# Note:
# This class has a bit exessive debug logs, because it is tricky to debug
# it from the outside.
class _EventLoop:
    def __init__(self, coro: Coroutine[None, None, None], num_threads: int):
        self._coro = coro
        self._num_threads = num_threads

        self._loop = None

        self._task_started = SyncEvent()
        self._task_completed = SyncEvent()
        self._stop_requested = SyncEvent()

        self._thread = None

    def __str__(self):
        return str(
            {
                "thread_alive": self._thread.is_alive(),
                "task_started": self._task_started.is_set(),
                "task_completed": self._task_completed.is_set(),
                "stop_requested": self._stop_requested.is_set(),
            }
        )

    async def _execute_task(self) -> None:
        _LG.debug("The event loop thread coroutine is started.")
        _LG.debug("Initializing the thread pool of size=%d.", self._num_threads)
        self._loop = asyncio.get_running_loop()
        self._loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(
                max_workers=self._num_threads,
                thread_name_prefix="spdl_",
            )
        )

        _LG.debug("Starting the task.")

        task = create_task(self._coro, name="Pipeline::main")
        task.add_done_callback(lambda t: self._task_completed.set())

        self._task_started.set()
        while not task.done():
            await asyncio.wait([task], timeout=0.1)

            if not task.done() and self._stop_requested.is_set():
                _LG.debug(
                    "Stop request is received, but the task is not complete. "
                    "Cancelling the task."
                )
                task.cancel()
                await asyncio.wait([task])

        _LG.debug("The task is completed.")

        _LG.debug("%s", self)
        if not self._stop_requested.is_set():
            _LG.debug("Keep the event loop alive until the stop request is made.")
            while not self._stop_requested.is_set():
                await asyncio.sleep(0.1)
        _LG.debug("The background task is completed.")
        _LG.debug("The event loop is now shutdown.")

    def start(self, *, timeout: float | None = None, daemon: bool = False) -> None:
        """Start the thread and block until the loop is initialized."""
        if self._thread is not None:
            raise RuntimeError("The thread can start only once.")
        _LG.debug("Starting the event loop thread.")
        if daemon:
            warnings.warn(
                "The event loop thread is started with daemon=True. "
                "This will let Python interpreter terminate before "
                "the event loop thread is shutdown. "
                "The event loop and the thread will be abruptly stopped "
                "while there might be running coroutines. "
                "This can cause various unexpected/unwanted side effects "
                "including abnormal exit. "
                "This option is provided only as a last resort to just "
                "let Python interpreter terminate, and "
                "it does not guarantee clean exit. "
                "You should not rely on this and should implement "
                "a graceful shutdown.",
                stacklevel=3,
            )

        self._thread = Thread(
            target=lambda: asyncio.run(self._execute_task()), daemon=daemon
        )
        self._thread.start()
        _LG.debug("Waiting for the loop to be initialized.")
        self._task_started.wait(timeout=timeout)
        _LG.debug("The event loop thread is initialized.")

    def is_started(self) -> bool:
        """Check if the event loop thread is started."""
        return self._task_started.is_set()

    def is_task_completed(self) -> bool:
        """Check if the task is completed."""
        return self._task_completed.is_set()

    def stop(self):
        """Issue loop stop request."""
        if not self._stop_requested.is_set():
            _LG.debug("Requesting the event loop thread to stop.")
            self._stop_requested.set()

    def join(self, *, timeout: float | None = None) -> None:
        """Let the thread join. ``stop`` must be called before calling ``join``."""
        if not self._stop_requested.is_set():
            raise RuntimeError(
                "The event loop thread is not stopped. Call stop() first."
            )

        _LG.debug("Waiting for the event loop thread to join.")
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError(f"Thread did not join after {timeout} seconds.")
        _LG.debug("The event loop thread joined.")

    def run_coroutine_threadsafe(self, coro) -> concurrent.futures.Future:
        """Call coroutine in the loop thread."""
        if not self._task_started.is_set():
            raise RuntimeError("Event loop is not started.")
        if not self._loop.is_running():
            raise RuntimeError("Event loop is not running.")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)


################################################################################
# Pipeline
################################################################################


class Pipeline(Generic[T]):
    """Pipeline()

    **[Experimental]** Data processing pipeline. Use :py:class:`PipelineBuilder` to
    instantiate.

    ``Pipeline`` and ``PipelineBuilder`` facilitate building data processing pipeline
    consists of multiple stages of async operations.
    It allows to configure the concurrency of each stage independently.

    Typically, the source is a lightweight (synchronous) iterable that generates the
    source location of data, such as file paths and URLs.
    The first stage retrieves  data from the (network) storage.

    The subsequent stages process the data, such as decoding images and resizing them,
    or decoding audio and resampling them.

    After the preprocessings are done, the data are buffered in a sink, which is a queue.

    The pipeline is executed in a background thread, so that the main thread can perform
    other tasks while the data are being processed.

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

           import spdl.io

           def source():
               with open("images.txt") as f:
                   for path in f:
                       yield path

           async def decode(path):
               return await spdl.io.async_decode_image(path)


           pipeline: Pipeline = (
               PipelineBuilder()
               .add_source(source())
               .pipe(decode, concurrency=10)
               .add_sink(3)
               .build(num_threads=10)
           )

           pipeline.start()
           try:
               for item in pipeline.get_iterator():
                   # do something with the decoded image
                   ...
           finally:
               pipeline.stop()
    """

    def __init__(
        self,
        coro: Coroutine[None, None, None],
        queues: list[AsyncQueue],
        num_threads: int,
        *,
        desc: list[str],
    ):
        self._queues = queues

        self._str = "\n".join([repr(self)] + desc)

        self._output_queue = queues[-1]
        self._event_loop = _EventLoop(coro, num_threads)

        try:
            from spdl.lib import _libspdl

            _libspdl.log_api_usage("spdl.dataloader.Pipeline")
        except Exception:
            pass  # ignore if not supported.

    def __str__(self) -> str:
        return self._str

    def start(self, *, timeout: float | None = None, **kwargs) -> None:
        """Start the pipeline in background thread.

        Args:
            timeout: Timeout value used when starting the thread and
                waiting for the pipeline to be initialized. [Unit: second]
        """
        self._event_loop.start(timeout=timeout, **kwargs)

    def stop(self, *, timeout: float | None = None) -> None:
        """Stop the pipeline.

        Args:
            timeout: Timeout value used when stopping the pipeline and
                waiting for the thread to join. [Unit: second]
        """
        self._event_loop.stop()
        self._event_loop.join(timeout=timeout)

    @contextmanager
    def auto_stop(self, *, timeout: float | None = None):
        """Context manager to start/stop the background thread automatically.

        Args:
            timeout: The duration to wait for the thread initialization / shutdown. [Unit: second]
                If ``None`` (default), it waits indefinitely.
        """
        self.start(timeout=timeout)
        try:
            yield
        finally:
            self.stop(timeout=timeout)

    def get_item(self, *, timeout: float | None = None) -> T:
        """Get the next item.

        Args:
            timeout: The duration to wait for the next item to become available. [Unit: second]
                If ``None`` (default), it waits indefinitely.

        Raises:
            RuntimeError: The pipeline is not started.

            TimeoutError: When pipeline is not producing the next item within the given time.

            EOFError: When the pipeline is exhausted or cancelled and there are no more items
                in the sink.
        """
        eof_message = "Reached the end of the pipeline."

        if not self._event_loop.is_started():
            raise RuntimeError("Pipeline is not started.")

        # The event loop (thread) was started, but it might be stopped by now.
        # However, what matters for `get_item` method is whether the task is running or not.
        # Because if the task is running, then accessing the sink queue must be done through
        # async method, invoked via event loop's `run_coroutine_threadsafe` method.
        # If the task is not running, then, sync method can be used to access sink queue,
        # even if the loop is not running.

        if self._event_loop.is_task_completed():
            # The pipeline is stopped.
            # The sink queue is not accessed by background event loop anymore, so we can use
            # sync access without being worried about thread safety.

            # There are remaining items in the queue if the pipeline was stopped by client code
            # before it processes all the items.
            if not self._output_queue.empty():
                return self._output_queue.get_nowait()

            # Now, all the items from the queue are fetched. We can stop the pipeline loop/thread.
            self._event_loop.stop()
            raise EOFError(eof_message)

        # The task is not completed. To access the sink queue, the async method must be used.
        # The loop keeps running unless we explicitly request stop, so the use of async method
        # itself is fine.

        # However, the background task can complete at any ponit.
        # It turned out to be very easy to hit the race condition where the foreground execution
        # control reaches here, in the short time window between the background thread puts the
        # last item and issues task completion.
        # In this case, without timeout, the foreground gets stuck.
        # Therefore, we split the timeout into small window and periodically check the state of
        # the background task.
        max_elapsed = float("inf") if timeout is None else timeout

        future = self._event_loop.run_coroutine_threadsafe(self._output_queue.get())
        t0 = time.monotonic()
        while (elapsed := time.monotonic() - t0) < max_elapsed:
            try:
                return future.result(timeout=min(0.1, max_elapsed))
            except concurrent.futures.TimeoutError:
                # The sink queue is empty.
                # In this condition, we cannot really tell if it is due to EOF or
                # pipeline being too slow.

                # One exception is that the task is now complete and queue is still empty.
                # This case we can switch to EOFError.
                if self._event_loop.is_task_completed() and self._output_queue.empty():
                    self._event_loop.stop()
                    raise EOFError(eof_message) from None

        _LG.debug("EventLoop: %s", str(self._event_loop))

        raise TimeoutError(f"The next item is not available after {elapsed:.0f} sec.")

    def get_iterator(self, *, timeout: float | None = None) -> Iterator[T]:
        """Get an iterator, which iterates over the pipeline outputs.

        Args:
            timeout: Timeout value used for each `get_item` call.
        """
        return PipelineIterator(self, timeout)

    def __iter__(self) -> Iterator[T]:
        """Call :py:meth:`~spdl.dataloader.Pipeline.get_iterator` without arguments."""
        return self.get_iterator()


class PipelineIterator(Generic[T]):
    """PipelineIterator()"""

    def __init__(self, pipeline: Pipeline[T], timeout):
        self._pipeline = pipeline
        self._timeout = timeout

    def __iter__(self) -> "PipelineIterator":
        return self

    def __next__(self) -> T:
        try:
            return self._pipeline.get_item(timeout=self._timeout)
        except EOFError:
            raise StopIteration from None
