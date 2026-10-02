# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import asyncio
import concurrent.futures
import logging
import queue
import threading
import time
import warnings
import weakref
from asyncio import AbstractEventLoop, Queue as AsyncQueue
from collections.abc import Coroutine, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from enum import IntEnum
from threading import Event as SyncEvent, Thread
from typing import Any, cast, Generic, TypeVar

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
        self._user_task_started = SyncEvent()
        self._task_completed = SyncEvent()
        self._task_exception: BaseException | None = None
        self._stop_requested = SyncEvent()

        self._thread: Thread | None = None
        self._joined = False
        self._start_gate: asyncio.Event | None = None

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

    async def _run_task_after_start(self) -> None:
        """Run user code only after ``start`` has observed loop initialization."""
        assert self._start_gate is not None
        await self._start_gate.wait()
        if self._stop_requested.is_set():
            return
        self._user_task_started.set()
        await self._coro

    async def _execute_task(self) -> None:
        _LG.debug("The event loop thread coroutine is started.")
        self._loop = asyncio.get_running_loop()
        self._loop.set_default_executor(self._executor)
        self._start_gate = asyncio.Event()

        _LG.debug("Starting the task.")

        task = create_task(self._run_task_after_start(), name="Pipeline::main")
        task.add_done_callback(lambda _: self._task_completed.set())

        self._task_started.set()
        while not task.done():
            if self._stop_requested.is_set():
                _LG.debug(
                    "Stop request is received, but the task is not complete. "
                    "Cancelling the task."
                )
                task.cancel()
                await asyncio.wait([task])
                continue
            await asyncio.wait([task], timeout=0.1)

        # Cancelling an asyncio Task before its first step does not enter the
        # wrapper coroutine, so its ``finally`` cannot close the still-unawaited
        # user coroutine. Close it here as a final cleanup backstop.
        if not self._user_task_started.is_set():
            self._coro.close()

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
        if not self._task_started.wait(timeout=timeout):
            # A timed-out start is terminal. Request shutdown, but do not join here:
            # the thread may not have reached its entry point yet, and an unbounded
            # join would make ``timeout`` meaningless. ``_PipelineImpl.stop`` (including
            # its finalizer path) can join this still-starting thread later.
            self.stop()
            raise TimeoutError(f"Event loop did not start after {timeout} seconds.")
        assert self._loop is not None
        assert self._start_gate is not None
        self._loop.call_soon_threadsafe(self._start_gate.set)
        _LG.debug("The event loop thread is initialized.")

    def is_started(self) -> bool:
        """Check if the event loop thread is started."""
        return self._task_started.is_set()

    def is_task_completed(self) -> bool:
        """Check if the task is completed."""
        return self._task_completed.is_set()

    def has_user_task_started(self) -> bool:
        """Check whether the caller-confirmed pipeline task began execution."""
        return self._user_task_started.is_set()

    def is_running(self) -> bool:
        """Check if the event loop can still service submitted work."""
        return self._loop is not None and self._loop.is_running()

    def is_alive(self) -> bool:
        """Check whether the event-loop thread still needs to be joined."""
        return self._thread is not None and self._thread.is_alive()

    def needs_join(self) -> bool:
        """Check whether a started thread still needs its first successful join."""
        return self._thread is not None and not self._joined

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
        self._joined = True
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
_NO_PENDING_OUTPUT_ITEM = object()


class _QueueReadTimedOut(Exception):
    """Signal that one loop-side queue-read slice expired."""


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
        self._pending_output_read: concurrent.futures.Future[T] | None = None
        self._pending_output_item: T | object = _NO_PENDING_OUTPUT_ITEM
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

        try:
            self._event_loop.start(timeout=timeout, **kwargs)
        except TimeoutError:
            # The event loop has requested stop, but its thread may still be starting.
            # Mark this pipeline terminal so a later auto-start cannot pretend it is
            # reusable. Resource cleanup remains with ``stop`` / the finalizer.
            self._event_loop_state = _EventLoopState.STOPPED
            raise
        self._event_loop_state = _EventLoopState.STARTED

    def stop(self, *, timeout: float | None = None) -> None:
        """Stop the pipeline.

        Args:
            timeout: Timeout value used when stopping the pipeline and
                waiting for the thread to join. [Unit: second]

        .. note::

           It is safe to call ``stop`` multiple times.
        """
        if (
            _EventLoopState.STARTED <= self._event_loop_state < _EventLoopState.STOPPED
            or self._event_loop.needs_join()
        ):
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
                # A timed-out start never owned the caller's queue. Once user code
                # has started, however, drain its output even if the loop has since
                # stopped reporting itself as running: this may be the only way to
                # release producer backpressure before the second join.
                if self._event_loop.has_user_task_started():
                    # Empty queue, release backpressure.
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

    async def _get_with_timeout_on_loop(self, timeout: float) -> T:
        """Wait for one item on the queue's owning loop without orphaning the get."""
        task = asyncio.create_task(self._output_queue.get())
        try:
            done, _ = await asyncio.wait([task], timeout=timeout)
        except BaseException:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if not task.cancelled() and task.exception() is None:
                # A producer may refill a bounded queue as soon as this read
                # frees a slot. Keep the recovered item outside the queue so
                # restoring it cannot fail with QueueFull or reorder it.
                assert self._pending_output_item is _NO_PENDING_OUTPUT_ITEM
                self._pending_output_item = task.result()
            raise

        if done or task.done():
            return task.result()

        task.cancel()
        try:
            return await task
        except asyncio.CancelledError:
            raise _QueueReadTimedOut from None

    def _consume_settled_output_read(self, future: concurrent.futures.Future[T]) -> T:
        """Consume a settled read, normalizing only internal empty results."""
        self._pending_output_read = None
        try:
            return future.result()
        except (_QueueReadTimedOut, asyncio.QueueEmpty):
            raise queue.Empty from None
        except concurrent.futures.CancelledError:
            raise queue.Empty from None

    def _get_pending_output_read(self, *, timeout: float) -> T:
        """Resolve the pending loop-side read without abandoning a consumed item."""
        future = self._pending_output_read
        if future is None:
            raise queue.Empty

        try:
            item = future.result(timeout=timeout)
        except _QueueReadTimedOut:
            self._pending_output_read = None
            raise queue.Empty from None
        except asyncio.QueueEmpty:
            self._pending_output_read = None
            raise queue.Empty from None
        except concurrent.futures.TimeoutError:
            if not future.done():
                raise
            # The Future completed as the foreground wait expired. Resolve its
            # settled result so a real sink TimeoutError is not mistaken for the
            # foreground deadline.
            return self._consume_settled_output_read(future)
        except concurrent.futures.CancelledError:
            self._pending_output_read = None
            raise queue.Empty from None
        except BaseException:
            if future.done():
                self._pending_output_read = None
            raise

        self._pending_output_read = None
        return item

    def _take_pending_output_item(self) -> T:
        """Return an item recovered from an interrupted loop-side read."""
        item = self._pending_output_item
        if item is _NO_PENDING_OUTPUT_ITEM:
            raise queue.Empty
        self._pending_output_item = _NO_PENDING_OUTPUT_ITEM
        return cast(T, item)

    def _resolve_output_read_after_loop_stop(self) -> T:
        """Cancel a stopped-loop read, harvesting a concurrently completed item."""
        future = self._pending_output_read
        assert future is not None
        if future.cancel():
            self._pending_output_read = None
        else:
            try:
                return self._get_pending_output_read(timeout=0.0)
            except concurrent.futures.TimeoutError:
                if self._pending_output_read is None:
                    raise
            except queue.Empty:
                pass
        self._pending_output_read = None
        try:
            return self._take_pending_output_item()
        except queue.Empty:
            pass
        raise TimeoutError(
            "The event loop stopped before publishing the queue read."
        ) from None

    def _poll_output_queue_on_loop(self) -> T:
        """Submit or resume one nonblocking owner-loop queue poll."""
        try:
            return self._take_pending_output_item()
        except queue.Empty:
            pass
        if self._pending_output_read is None:
            read = self._get_nowait_on_loop()
            try:
                self._pending_output_read = self._event_loop.run_coroutine_threadsafe(
                    read
                )
            except RuntimeError as error:
                read.close()
                return self._recover_output_after_read_submission_failure(error)
        try:
            return self._get_pending_output_read(timeout=0.0)
        except concurrent.futures.TimeoutError:
            if self._pending_output_read is None:
                raise
            raise queue.Empty from None

    def _get_completed_output_item(self) -> T:
        """Return buffered output or EOF after the pipeline task completes."""
        try:
            return self._take_pending_output_item()
        except queue.Empty:
            pass
        if not self._output_queue.empty():
            return self._output_queue.get_nowait()
        self._event_loop.stop()
        raise EOFError(_EOF_MSG) from None

    def _recover_output_after_read_submission_failure(self, error: RuntimeError) -> T:
        """Drain completed output when its owner loop stops before submission."""
        if not self._event_loop.is_task_completed() or self._event_loop.is_running():
            raise error
        return self._get_completed_output_item()

    def _poll_output_at_deadline(self, t0: float) -> T:
        """Make one final nonblocking poll before reporting a caller timeout."""
        try:
            return self._get_item_nowait()
        except queue.Empty:
            _LG.debug("EventLoop: %s", str(self._event_loop))
            raise TimeoutError(
                f"The next item is not available after {time.monotonic() - t0:.1f} sec."
            ) from None

    def _get_item_async_queue(self, *, timeout: float | None) -> T:
        try:
            return self._take_pending_output_item()
        except queue.Empty:
            pass

        # The event loop (thread) was started, but it might be stopped by now.
        # However, what matters for `get_item` method is whether the task is running or not.
        # Because if the task is running, then accessing the sink queue must be done through
        # async method, invoked via event loop's `run_coroutine_threadsafe` method.
        # If the task is not running, then, sync method can be used to access sink queue,
        # even if the loop is not running.

        if self._pending_output_read is None and self._event_loop.is_task_completed():
            return self._get_completed_output_item()

        # The task is not completed. To access the sink queue, the async method must be used.
        # The loop keeps running unless we explicitly request stop, so the use of async method
        # itself is fine.

        # Handle a zero timeout as a true one-shot poll. Submitting ``queue.get()`` and
        # timing out its cross-thread future would leave that coroutine pending,
        # allowing it to consume (and lose) the next item produced after this call
        # returns.
        if timeout == 0:
            try:
                return self._get_item_nowait()
            except queue.Empty:
                raise TimeoutError(
                    "The next item is not available after 0.0 sec."
                ) from None

        # The background task can complete at any point. Wait in bounded queue-get
        # slices so completion is observed even when no final item is written.
        # A single loop-side read remains registered until it settles. If the
        # foreground caller times out first, the next call resumes that read so
        # no consumed item is abandoned.
        deadline = None if timeout is None else time.monotonic() + timeout
        t0 = time.monotonic()
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                # A tiny positive timeout can round the deadline to ``t0``. Always
                # poll once so an item that was already buffered still wins.
                return self._poll_output_at_deadline(t0)
            if self._pending_output_read is None:
                wait_timeout = 0.1 if remaining is None else min(0.1, remaining)
                read = self._get_with_timeout_on_loop(wait_timeout)
                try:
                    self._pending_output_read = (
                        self._event_loop.run_coroutine_threadsafe(read)
                    )
                except RuntimeError as error:
                    read.close()
                    return self._recover_output_after_read_submission_failure(error)

            publication_timeout = 0.2
            if deadline is not None:
                publication_timeout = min(
                    publication_timeout, max(0.0, deadline - time.monotonic())
                )
            try:
                return self._get_pending_output_read(timeout=publication_timeout)
            except concurrent.futures.TimeoutError:
                if self._pending_output_read is None:
                    raise
                if not self._event_loop.is_running():
                    return self._resolve_output_read_after_loop_stop()
                # Preserve this Future across caller timeouts: it may already
                # own an item that has not yet been published cross-thread.
                continue
            except queue.Empty:
                pass

            # The sink queue is empty.
            # In this condition, we cannot really tell if it is due to EOF or
            # pipeline being too slow.

            # One exception is that the task is now complete and queue is still empty.
            # This case we can switch to EOFError.
            if self._event_loop.is_task_completed():
                return self._get_completed_output_item()

    def get_item_nowait(self) -> T:
        """Get the next item if one is already buffered in the sink, without blocking.

        Raises:
            RuntimeError: The pipeline is not started.

            queue.Empty: No item is currently available. The pipeline is still running, so an
                item may become available later.

            EOFError: The pipeline is exhausted (or reached an epoch boundary) and the sink is
                drained.
        """
        item = self._get_item_nowait()
        if is_epoch_end(item):
            raise EOFError(_EOF_MSG)
        return item

    async def _get_nowait_on_loop(self) -> T:
        # Runs on the event loop thread. Deliberately has no ``await``: it completes within a
        # single loop tick. ``get_item(timeout=0)`` routes through this same primitive so both
        # APIs perform a one-shot poll and never leave a pending consumer behind.
        return self._output_queue.get_nowait()

    def _get_item_nowait(self) -> T:
        """Non-blocking counterpart of :py:meth:`_get_item`.

        Normalizes the two queue backends onto one exception contract: ``queue.Empty`` for "not
        yet", ``EOFError`` for "never". Callers can therefore drain opportunistically without
        caring which backend the sink uses.
        """
        if not self._event_loop.is_started():
            raise RuntimeError("Pipeline is not started.")

        if isinstance(self._output_queue, _ThreadBasedAsyncQueue):
            q = self._output_queue._queue  # pyre-ignore[16]
            try:
                return q.get_nowait()
            except queue.Empty:
                # Empty *and* the producing task is done means no item is ever coming.
                if self._event_loop.is_task_completed() and q.empty():
                    self._event_loop.stop()
                    raise EOFError(_EOF_MSG) from None
                raise

        if self._pending_output_read is None and self._event_loop.is_task_completed():
            # The background loop no longer touches the sink, so direct access is thread-safe.
            if not self._output_queue.empty():
                return self._output_queue.get_nowait()
            self._event_loop.stop()
            raise EOFError(_EOF_MSG)

        try:
            return self._poll_output_queue_on_loop()
        except queue.Empty:
            if (
                self._pending_output_read is None
                and self._event_loop.is_task_completed()
                and self._output_queue.empty()
            ):
                self._event_loop.stop()
                raise EOFError(_EOF_MSG) from None
            raise queue.Empty from None

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


def _register_stop_at_exit(impl: _PipelineImpl[Any]) -> None:
    """Run :py:func:`_stop_impl` at the start of interpreter finalization.

    Uses ``threading._register_atexit`` -- **not** ``atexit.register``:
    threading-atexit callbacks run inside ``threading._shutdown()``, before
    non-daemon threads and child processes are joined. An ``atexit`` hook runs
    later, after the interpreter may have hung joining the pipeline's event-loop
    thread. This is the same private API, for the same reason, that
    ``concurrent.futures`` uses. Called from :py:meth:`Pipeline.start` (after the
    pipeline started its own threads/processes, so this registers last and,
    being LIFO, runs first). The hook holds only a weak reference, so it never
    keeps the pipeline alive; once the pipeline is stopped or collected it is a
    safe no-op (``stop`` is idempotent).
    """
    ref = weakref.ref(impl)

    def _hook() -> None:
        if (impl := ref()) is not None:
            _stop_impl(impl)

    try:
        threading._register_atexit(_hook)  # pyre-ignore[16]
    except (AttributeError, RuntimeError):
        # Defensive around a private stdlib API: ``RuntimeError`` if the
        # interpreter is already shutting down, ``AttributeError`` if a future
        # Python drops the hook. The exit hook is only a safety net (explicit
        # ``stop()`` and the GC finalizer still work), so failing to register
        # must never break pipeline start; a unit test asserts the API exists so
        # a genuine regression surfaces loudly in CI.
        _LG.debug("Could not register interpreter-exit stop hook.", exc_info=True)


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

           for item in pipeline.get_iterator(timeout=30):
               # do something with the decoded image
               ...

    A ``Pipeline`` cleans up its background thread and worker processes
    automatically, so a forgotten pipeline will not hang the process at exit: it
    is stopped when the object is garbage collected, and — as a safety net for a
    reference held until the program ends — by a hook that :py:meth:`start`
    registers to run at interpreter shutdown. Even so, it is **recommended to
    release the resources explicitly** once you are done, either by calling
    :py:meth:`stop` (or using the :py:meth:`auto_stop` context manager) or by
    dropping all strong references to the ``Pipeline`` so it is garbage collected.
    This frees the background thread, worker processes, and memory promptly,
    rather than leaving them alive until exit.

    .. versionchanged:: 0.4.0

       Calling :py:meth:`start` and :py:meth:`stop` is now optional.
       When iterating a pipeline that has not been explicitly started,
       the background thread is started automatically on the first item request.
       When the ``Pipeline`` object is garbage collected, the background thread
       is stopped automatically via :py:class:`weakref.finalize`.
       Explicit :py:meth:`start` / :py:meth:`stop` and the :py:meth:`auto_stop`
       context manager continue to work as before.

    .. versionchanged:: 0.6.0

       Cleanup is now more robust: :py:meth:`start` registers a hook (via
       ``threading._register_atexit``) that stops a still-running pipeline at
       interpreter shutdown, so a reference held until the program ends no longer
       risks a hang at exit. Explicitly releasing resources is still recommended
       — call :py:meth:`stop` (or use :py:meth:`auto_stop`), or drop all strong
       references so the pipeline is garbage collected — to free them promptly.
       See :py:meth:`start` for details.
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

        .. note::

           **Cleanup at interpreter exit.** The pipeline runs a background, *non-daemon*
           event-loop thread (and may own worker subprocesses). They are released
           when you :py:meth:`stop` the pipeline, or when the object is garbage
           collected -- a :py:func:`weakref.finalize` stops it at GC. But if a strong
           reference is held until the program ends (e.g. a training loop that keeps
           the dataloader for the whole run), GC does not run before interpreter
           shutdown, which would otherwise **hang** joining the still-running
           event-loop thread.

           To prevent that, :py:meth:`start` registers a process-wide hook that
           stops the pipeline at the very start of interpreter finalization. It holds
           only a weak reference, so it never keeps the pipeline alive; explicit
           :py:meth:`stop` and the GC path are unaffected.

           The hook uses ``threading._register_atexit`` and **not**
           :py:func:`atexit.register`, because of *when* CPython runs each. A plain
           ``atexit`` hook runs **after** non-daemon threads are already joined --
           too late. ``threading._register_atexit`` callbacks run earlier, inside
           ``threading._shutdown()``, *before* that join (the same mechanism, and the
           same reason, that :py:mod:`concurrent.futures` uses):

           .. mermaid::

              flowchart TD

                  subgraph C["run threading._register_atexit hooks (LIFO)"]
                      S1["Pipeline's stop hook runs here: pipeline stopped"]
                  end

                  subgraph E["atexit hooks (LIFO)"]
                      M1["multiprocessing joins children;"]
                      M2["weakref.finalize"]
                  end

                  A["Python reaches the end of program"]
                  --> B["threading._shutdown()"]
                  --> C
                  --> D["join non-daemon threads"]
                  --> E
                  --> F["GC and module teardown"]

           The stop hook runs before the non-daemon-thread join and before
           the ``atexit`` phase, so the pipeline is torn down before anything
           blocks on it.
        """
        self._impl.start(timeout=timeout, **kwargs)
        # Register only after a successful start (raises if already started -> no
        # double-register) and after the pipeline's threads/processes are up, so this
        # hook lands after the stdlib atexit hooks and -- being LIFO -- runs first.
        _register_stop_at_exit(self._impl)

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
        # Ensure that the pipeline is started before accessing the sink queue. Route through the
        # facade `start` (not `_impl.start`) so an auto-started pipeline also registers the
        # interpreter-exit stop hook.
        # Note: This check-then-start pattern is not thread-safe, but `get_item` is not
        # supposed to be called from multiple threads.
        if self._impl._event_loop_state == _EventLoopState.NOT_STARTED:
            self.start()
        return self._impl.get_item(timeout=timeout)

    def _get_item_nowait(self) -> T:
        """Get the next item if one is already buffered, without blocking.

        Internal: used to drain a burst of already-produced results in one go (see the
        fused-region worker in :py:mod:`spdl.pipeline._subprocess_pipeline_pool`). Unlike
        ``get_item(timeout=0)``, this cannot strand an item.

        Raises:
            RuntimeError: The pipeline is not started.

            queue.Empty: No item is currently available.

            EOFError: The pipeline is exhausted (or reached an epoch boundary) and drained.
        """
        # Unlike `get_item`, do not auto-start: a caller polling a not-yet-started pipeline
        # wants "nothing available", and silently starting it here would hide a usage error.
        return self._impl.get_item_nowait()

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
