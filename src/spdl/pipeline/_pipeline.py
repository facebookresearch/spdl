# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import concurrent.futures
import logging
import queue
import time
import warnings
import weakref
from asyncio import AbstractEventLoop, Queue as AsyncQueue
from collections.abc import Coroutine, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from enum import IntEnum
from threading import Event as SyncEvent, Thread
from typing import Any, Generic, TypeVar

from spdl.pipeline._common._misc import create_task
from spdl.pipeline._components import _ThreadBasedAsyncQueue, is_epoch_end

__all__ = ["Pipeline"]

_LG: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


##############################################################################
# _EventLoop (Thread)
##############################################################################


# Note:
# This class has a bit excessive debug logs, because it is tricky to debug
# it from the outside.
class _EventLoop:
    def __init__(
        self,
        coro: Coroutine[None, None, None],
        executor: ThreadPoolExecutor,
    ) -> None:
        self._coro = coro
        self._executor = executor

        self._loop: AbstractEventLoop | None = None

        self._task_started = SyncEvent()
        self._task_completed = SyncEvent()
        self._task_exception: BaseException | None = None
        self._stop_requested = SyncEvent()

        self._thread: Thread | None = None

    def __str__(self) -> str:
        return str(
            {
                "thread_alive": False
                if self._thread is None
                else self._thread.is_alive(),
                "task_started": self._task_started.is_set(),
                "task_completed": self._task_completed.is_set(),
                "stop_requested": self._stop_requested.is_set(),
            }
        )

    async def _execute_task(self) -> None:
        _LG.debug("The event loop thread coroutine is started.")
        self._loop = asyncio.get_running_loop()
        self._loop.set_default_executor(self._executor)

        _LG.debug("Starting the task.")

        task = create_task(self._coro, name="Pipeline::main")
        task.add_done_callback(lambda _: self._task_completed.set())

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

        try:
            self._task_exception = task.exception()
        except asyncio.CancelledError:
            pass

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
            # Using lambda to delay the creation of coroutine object.
            target=lambda: asyncio.run(self._execute_task()),
            name="spdl_event_loop_thread",
            daemon=daemon,
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

    def stop(self) -> None:
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
        assert self._thread is not None
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():  # pyre-ignore[undefined-attribute]
            raise TimeoutError(f"Thread did not join after {timeout} seconds.")
        _LG.debug("The event loop thread joined.")

    def run_coroutine_threadsafe(
        self, coro: Coroutine[None, None, T]
    ) -> concurrent.futures.Future[T]:
        """Call coroutine in the loop thread."""
        if not self._task_started.is_set():
            raise RuntimeError("Event loop is not started.")
        assert self._loop is not None
        if not self._loop.is_running():
            raise RuntimeError("Event loop is not running.")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)  # pyre-ignore[6]


################################################################################
# Pipeline
################################################################################


class _EventLoopState(IntEnum):
    NOT_STARTED = 0
    STARTED = 1
    STOPPED = 2


_EOF_MSG: str = "Reached the end of the pipeline."


class _PipelineImpl(Generic[T]):
    """Internal implementation of the data processing pipeline.

    Use :py:class:`Pipeline` (the public facade) instead.
    """

    def __init__(
        self,
        coro: Coroutine[None, None, None],
        output_queue: AsyncQueue,
        executor: ThreadPoolExecutor,
        *,
        desc: str,
        pools: Sequence[Any] = (),
    ) -> None:
        self._str: str = "\n".join([repr(self), desc])

        self._output_queue: AsyncQueue = output_queue
        self._event_loop = _EventLoop(coro, executor)
        self._event_loop_state: _EventLoopState = _EventLoopState.NOT_STARTED
        # Worker pools owned by this pipeline (from subprocess-stage fusion). They are reaped in
        # ``stop`` (and via the Pipeline finalizer), exactly once.
        self._pools: list[Any] = list(pools)

    def __str__(self) -> str:
        return self._str

    def start(self, *, timeout: float | None = None, **kwargs: Any) -> None:
        """Start the pipeline in background thread.

        Args:
            timeout: Timeout value used when starting the thread and
                waiting for the pipeline to be initialized. [Unit: second]

        .. note::

           Calling ``start`` multiple times raises ``RuntimeError``.
        """
        if self._event_loop_state >= _EventLoopState.STARTED:
            raise RuntimeError("The pipeline was already started.")

        self._event_loop.start(timeout=timeout, **kwargs)
        self._event_loop_state = _EventLoopState.STARTED

    def stop(self, *, timeout: float | None = None) -> None:
        """Stop the pipeline.

        Args:
            timeout: Timeout value used when stopping the pipeline and
                waiting for the thread to join. [Unit: second]

        .. note::

           It is safe to call ``stop`` multiple times.
        """
        if _EventLoopState.STARTED <= self._event_loop_state < _EventLoopState.STOPPED:
            self._event_loop.stop()

            # Try to join first. If it doesn't join, drain the output queue
            # to resolve the congestion, then retry.
            # (e.g. the frontend does not consume any data, thus the upstream tasks
            # are not able to complete),
            to1: float = 3 if timeout is None else min(3, timeout)
            to2: float | None = None if timeout is None else timeout - to1
            try:
                self._event_loop.join(timeout=to1)
            except TimeoutError:
                # Empty queue, release backpressure
                while not self._output_queue.empty():
                    try:
                        self._output_queue.get_nowait()
                    except Exception:
                        break
                self._event_loop.join(timeout=to2)
            self._event_loop_state = _EventLoopState.STOPPED

        self._shutdown_pools()

        if self._event_loop._task_exception is not None:
            raise self._event_loop._task_exception

    def _shutdown_pools(self) -> None:
        """Reap any owned worker pools exactly once (safe to call repeatedly)."""
        pools, self._pools = self._pools, []
        for pool in pools:
            try:
                pool.shutdown()
            except Exception:
                _LG.debug("Exception during worker pool shutdown.", exc_info=True)

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
        item = self._get_item(timeout=timeout)
        if is_epoch_end(item):
            raise EOFError(_EOF_MSG)
        return item

    def _get_item(self, *, timeout: float | None) -> T:
        if not self._event_loop.is_started():
            raise RuntimeError("Pipeline is not started.")

        if isinstance(self._output_queue, _ThreadBasedAsyncQueue):
            return self._get_item_thread_queue(timeout=timeout)
        return self._get_item_async_queue(timeout=timeout)

    def _get_item_async_queue(self, *, timeout: float | None) -> T:
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
            raise EOFError(_EOF_MSG)

        # The task is not completed. To access the sink queue, the async method must be used.
        # The loop keeps running unless we explicitly request stop, so the use of async method
        # itself is fine.

        # However, the background task can complete at any point.
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
                    raise EOFError(_EOF_MSG) from None

        _LG.debug("EventLoop: %s", str(self._event_loop))

        raise TimeoutError(f"The next item is not available after {elapsed:.1f} sec.")

    def _get_item_thread_queue(self, *, timeout: float | None) -> T:
        q = self._output_queue._queue  # pyre-ignore[16]

        if self._event_loop.is_task_completed():
            if not q.empty():
                return q.get_nowait()
            self._event_loop.stop()
            raise EOFError(_EOF_MSG)

        max_elapsed = float("inf") if timeout is None else timeout
        t0 = time.monotonic()
        while (elapsed := time.monotonic() - t0) < max_elapsed:
            remaining = max_elapsed - elapsed
            try:
                return q.get(timeout=min(0.1, remaining))
            except queue.Empty:
                if self._event_loop.is_task_completed() and q.empty():
                    self._event_loop.stop()
                    raise EOFError(_EOF_MSG) from None

        raise TimeoutError(
            f"The next item is not available after {time.monotonic() - t0:.1f} sec."
        )


################################################################################
# Pipeline (Public Facade)
################################################################################

_STOP_TIMEOUT: float = 10.0


def _stop_impl(impl: _PipelineImpl[Any]) -> None:
    try:
        impl.stop(timeout=_STOP_TIMEOUT)
    except Exception:
        _LG.debug("Exception during automatic pipeline shutdown.", exc_info=True)


class Pipeline(Generic[T]):
    """Pipeline()

    Data processing pipeline. Use :py:class:`PipelineBuilder` to instantiate.

    .. seealso::

       - :ref:`intro`
         explains the basic usage of ``PipelineBuilder`` and  ``Pipeline``.
       - :ref:`pipeline-caveats`
         lists known anti-patterns that can cause a deadlock.
       - :ref:`pipeline-parallelism`
         covers how to switch (or combine)
         multi-threading and multi-processing in detail.

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

           def load(path):
               return await spdl.io.load_image(path)


           pipeline: Pipeline = (
               PipelineBuilder()
               .add_source(source())
               .pipe(decode, concurrency=10)
               .add_sink(3)
               .build(num_threads=10)
           )

           with pipeline.auto_stop():
               for item in pipeline.get_iterator(timeout=30):
                   # do something with the decoded image
                   ...

    When the ``Pipeline`` object is garbage collected, the background thread is
    automatically stopped. You can still use :py:meth:`auto_stop` or
    :py:meth:`stop` for deterministic, scoped lifecycle management.

    .. versionchanged:: 0.4.0

       **[Experimental]** Calling :py:meth:`start` and :py:meth:`stop` is now
       optional. When iterating a pipeline that has not been explicitly started,
       the background thread is started automatically on the first item request.
       When the ``Pipeline`` object is garbage collected, the background thread
       is stopped automatically via :py:func:`weakref.finalize`.
       Explicit :py:meth:`start` / :py:meth:`stop` and the :py:meth:`auto_stop`
       context manager continue to work as before.
    """

    def __init__(
        self,
        coro: Coroutine[None, None, None],
        output_queue: AsyncQueue,
        executor: ThreadPoolExecutor,
        *,
        desc: str,
        pools: Sequence[Any] = (),
    ) -> None:
        self._impl: _PipelineImpl[T] = _PipelineImpl(
            coro, output_queue, executor, desc=desc, pools=pools
        )
        self._finalizer = weakref.finalize(self, _stop_impl, self._impl)

    def __str__(self) -> str:
        return str(self._impl)

    def start(self, *, timeout: float | None = None, **kwargs: Any) -> None:
        """Start the pipeline in background thread.

        Args:
            timeout: Timeout value used when starting the thread and
                waiting for the pipeline to be initialized. [Unit: second]

        .. note::

           Calling ``start`` multiple times raises ``RuntimeError``.
        """
        self._impl.start(timeout=timeout, **kwargs)

    def stop(self, *, timeout: float | None = None) -> None:
        """Stop the pipeline.

        Args:
            timeout: Timeout value used when stopping the pipeline and
                waiting for the thread to join. [Unit: second]

        .. note::

           It is safe to call ``stop`` multiple times.
        """
        self._impl.stop(timeout=timeout)
        self._finalizer.detach()

    @contextmanager
    def auto_stop(self, *, timeout: float | None = None) -> Iterator[None]:
        """Context manager to start/stop the background thread automatically.

        Args:
            timeout: The duration to wait for the thread
                initialization / shutdown. [Unit: second]
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
        # Ensure that the pipeline is started before accessing the sink queue.
        # Note: This check-then-start pattern is not thread-safe, but `get_item` is not
        # it is not supposed to be called from multiple threads.
        if self._impl._event_loop_state == _EventLoopState.NOT_STARTED:
            self._impl.start()
        return self._impl.get_item(timeout=timeout)

    def get_iterator(self, *, timeout: float | None = None) -> Iterator[T]:
        """Get an iterator, which iterates over the pipeline outputs.

        The returned iterator covers a single epoch (one pass over the source),
        regardless of whether the source is continuous (see the ``continuous``
        argument of :py:meth:`PipelineBuilder.add_source
        <spdl.pipeline.PipelineBuilder.add_source>`). Call this method again to
        iterate each subsequent epoch:

        .. code-block:: python

            for epoch in range(num_epochs):
                for item in pipeline.get_iterator(timeout=...):
                    ...

        Args:
            timeout: Timeout value used for each `get_item` call.

        .. versionchanged:: 0.6.0
           Fixed reuse with a continuous source: an iterator that reached its
           epoch boundary used to resume into the next epoch when reused, but
           now stays exhausted, consistent with non-continuous sources. Use one
           iterator per epoch.
        """
        return PipelineIterator(self, timeout)

    def __iter__(self) -> Iterator[T]:
        """Call :py:meth:`~spdl.pipeline.Pipeline.get_iterator` without arguments."""
        return self.get_iterator()


class PipelineIterator(Generic[T]):
    """PipelineIterator()"""

    def __init__(self, pipeline: Pipeline[T], timeout: float | None) -> None:
        self._pipeline = pipeline
        self._timeout = timeout
        self._epoch_ended: bool = False

    def __iter__(self) -> "PipelineIterator[T]":
        return self

    def __next__(self) -> T:
        # Each iterator covers a single epoch and is single-use: once it reaches
        # the epoch boundary it stays exhausted, so its behavior is the same
        # whether or not the source is continuous. Iterate the next epoch by
        # obtaining a fresh iterator via `Pipeline.get_iterator()`.
        if self._epoch_ended:
            raise StopIteration
        try:
            return self._pipeline.get_item(timeout=self._timeout)
        except EOFError:
            self._epoch_ended = True
            raise StopIteration from None
