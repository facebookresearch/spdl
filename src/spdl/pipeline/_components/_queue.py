# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import asyncio
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast, Protocol

from spdl.pipeline._common._misc import create_task

from ._common import (
    _EOF,
    _periodic_dispatch,
    _ShieldedHook,
    _StatsCounter,
    _time_str,
    StageInfo,
)

__all__ = [
    "_queue_stage_hook",
    "_AsyncQueueWithSyncMirror",
    "AsyncQueue",
    "_ThreadBasedAsyncQueue",
    "StatsQueue",
    "QueuePerfStats",
    "set_default_queue_class",
    "get_default_queue_class",
]

_LG: logging.Logger = logging.getLogger(__name__)


class _DequeBackedQueueProtocol(Protocol):
    _queue: deque[object]


class AsyncQueue(asyncio.Queue):
    """Extends :py:class:`asyncio.Queue` with init/finalize logic.

    When a pipeline stage starts, :py:class:`PipelineBuilder` calls
    :py:meth:`~AsyncQueue.stage_hook` method, and the initialization logic
    is executed. During the execution of the pipeline, ``get``/``put`` methods
    are used to pass data. At the end of pipeline execution, the finalization
    logic in :py:meth:`~AsyncQueue.stage_hook` is executed.


    One intended usage is to overload the ``get``/``put`` methods to capture the
    time each pipeline stage waits, then publish the stats. This helps identifying
    the bottleneck in the pipeline. See :py:class:`StatsQueue`.

    Args:
        info: The stage identity. Assigned by :py:class:`PipelineBuilder`.
        buffer_size: The buffer size. Assigned by :py:class:`PipelineBuilder`.
    """

    def __init__(
        self,
        info: StageInfo,
        *,
        buffer_size: int = 1,
    ) -> None:
        super().__init__(buffer_size)
        self.info = info
        self._put_commit_callback: Callable[[object], None] | None = None

    def _put(self, item: object) -> None:
        """Store an item and notify an optional same-turn commit observer."""
        super()._put(item)
        if (callback := self._put_commit_callback) is None:
            return
        try:
            callback(item)
        except BaseException:
            # asyncio.Queue has not updated task accounting or woken a getter yet,
            # so the just-appended item can still be rolled back transactionally.
            cast(_DequeBackedQueueProtocol, self)._queue.pop()
            raise

    @asynccontextmanager
    async def stage_hook(self) -> AsyncIterator[None]:
        """Context manager, which handles init/final logic for the stage."""
        yield


@asynccontextmanager
async def _queue_stage_hook(queue: AsyncQueue) -> AsyncGenerator[None, None]:
    # Responsibility
    #   1. Call the `stage_hook`` context manager
    #   2. Put _EOF when the stage is done for reasons other than cancel.

    # Note:
    # `asyncio.CancelledError` is a subclass of BaseException, so it won't be
    # caught in the following, and EOF won't be passed to the output queue.
    #
    # The stage hook is shielded so its finalization (e.g. flushing the final
    # `StatsQueue` stats) runs to completion even when the cancellation that
    # ends the stage propagates through this `async with`.
    async with _ShieldedHook(queue.stage_hook()):
        try:
            yield
        except Exception:
            await queue.put(_EOF)
            raise
        else:
            await queue.put(_EOF)


@dataclass
class QueuePerfStats:
    """Performance statistics collected by :py:class:`StatsQueue`.

    .. seealso::

       - :ref:`logging` explains how to export the runtime performance statistics.
       - :ref:`analysis` explains how to use the exported stats.
    """

    elapsed: float
    """The duration of measurement in second."""

    num_items: int
    """The number of items went through the queue."""

    ave_put_time: float
    """The average time (in second) ``put`` operation was blocked.

    It is the average time that the upstream task had to wait before
    the queue has a space to put the result.

    Note that when there are multiple tasks attempting to put a
    result (i.e. the upstream task has concurrency larger than 1),
    then the average time becomes longer.
    """

    ave_get_time: float
    """The average time (in second) ``get`` operation was blocked.

    It is the average time that the downstream task had to wait before
    the queue has the next item to fetch.

    Note that when there are multiple tasks attempting to get the
    next item (i.e. the downstream task has concurrency larger than 1),
    then the average time becomes longer.
    """

    p90_put_time: float
    """The 90th percentile time (in second) ``put`` operation was blocked."""

    p99_put_time: float
    """The 99th percentile time (in second) ``put`` operation was blocked."""

    p90_get_time: float
    """The 90th percentile time (in second) ``get`` operation was blocked."""

    p99_get_time: float
    """The 99th percentile time (in second) ``get`` operation was blocked."""

    occupancy_rate: float
    """The relative time where the queue was not empty.

    The value close to 1 means that the queue has always the
    next data available. The upstream stage is producing data faster
    than the speed of the downstream stage consuming them.

    The value close to 0 means that the queue is always empty.
    The downstream stage is waiting for the next data and fetches one
    as soon as the upstream stage puts it. This suggests that the pipeline
    is suffering from data starvation.
    """

    @property
    def qps(self) -> float:
        """Query per second. i.e. ``num_items / elapsed``."""
        if self.elapsed <= 0:
            return 0
        return self.num_items / self.elapsed


class StatsQueue(AsyncQueue):
    """Measures the time stages are blocked on upstream/downstream stage.
    Extends :py:class:`AsyncQueue`.

    .. seealso::

       - :ref:`logging` explains how to export the runtime performance statistics.
       - :ref:`analysis` explains how to use the exported stats.

    Args:
        info: The stage identity. Assigned by :py:class:`PipelineBuilder`.
        buffer_size: The buffer size. Assigned by :py:class:`PipelineBuilder`.
        interval: The interval (in second) between reporting performance numbers
            to console.
    """

    def __init__(
        self,
        info: StageInfo,
        *,
        buffer_size: int = 1,
        interval: float = -1,
    ) -> None:
        super().__init__(info, buffer_size=buffer_size)

        self.interval = interval

        # For measuring the pressure
        self._getc = _StatsCounter()
        self._putc = _StatsCounter()

        # For measuring starvation rate
        self._empty_t0 = 0.0
        self._dur_empty = 0.0

        # For interval
        self._lap_t0 = 0.0
        self._lap_num_get = 0
        self._lap_num_put = 0
        self._lap_ave_get_time = 0.0
        self._lap_ave_put_time = 0.0
        self._lap_dur_empty = 0.0

    async def get(self) -> object:
        """Remove and return an item from the queue, track the time."""
        with self._getc.count():
            item = await super().get()

        if self.qsize() == 0:
            self._empty_t0 = time.monotonic()

        return item

    async def put(self, item: object) -> None:
        """Remove and return an item from the queue, track the time."""
        if self.qsize() == 0:
            self._dur_empty += time.monotonic() - self._empty_t0

        with self._putc.count():
            return await super().put(item)

    @asynccontextmanager
    async def stage_hook(self) -> AsyncIterator[None]:
        t0 = self._lap_t0 = self._empty_t0 = time.monotonic()
        if self.interval > 0:
            done = asyncio.Event()
            report = create_task(
                _periodic_dispatch(self._log_interval_stats, done, self.interval),
                name=f"{self.info}_queue_periodic_report",
                log_cancelled=False,
            )

        try:
            yield
        finally:
            if self.interval > 0:
                # pyrefly: ignore [unbound-name]
                done.set()
                await report  # pyre-ignore: [61]

            elapsed = time.monotonic() - t0
            occupancy_rate = 0.0 if elapsed <= 0 else 1 - self._dur_empty / elapsed
            self._log_stats(
                QueuePerfStats(
                    elapsed=elapsed,
                    num_items=self._getc.num_items,
                    ave_put_time=self._putc.ave_time,
                    ave_get_time=self._getc.ave_time,
                    p90_put_time=self._putc.p90_time,
                    p99_put_time=self._putc.p99_time,
                    p90_get_time=self._getc.p90_time,
                    p99_get_time=self._getc.p99_time,
                    occupancy_rate=max(0.0, occupancy_rate),
                )
            )

    def _get_lap_stats(self) -> QueuePerfStats:
        # Get the current values
        now = time.monotonic()
        num_put, ave_put_time = self._putc.num_items, self._putc.ave_time
        num_get, ave_get_time = self._getc.num_items, self._getc.ave_time
        dur_empty = self._dur_empty

        # Compute delta
        elapsed = max(0.0, now - self._lap_t0)
        delta_num_put = max(0, num_put - self._lap_num_put)
        delta_num_get = max(0, num_get - self._lap_num_get)

        if delta_num_put <= 0:
            delta_ave_put_time = 0.0
        else:
            put_total_time = ave_put_time * num_put
            lap_put_total_time = self._lap_ave_put_time * self._lap_num_put
            delta_ave_put_time = (put_total_time - lap_put_total_time) / delta_num_put

        if delta_num_get <= 0:
            delta_ave_get_time = 0.0
        else:
            get_total_time = ave_get_time * num_get
            lap_get_total_time = self._lap_ave_get_time * self._lap_num_get
            delta_ave_get_time = (get_total_time - lap_get_total_time) / delta_num_get

        delta_dur_empty = dur_empty - self._lap_dur_empty
        occupancy_rate = 0 if elapsed <= 0 else 1 - delta_dur_empty / elapsed

        lap_put_p90, lap_put_p99 = self._putc.consume_lap_percentiles()
        lap_get_p90, lap_get_p99 = self._getc.consume_lap_percentiles()

        # Update the lap
        self._lap_t0 = now
        self._lap_num_put = num_put
        self._lap_num_get = num_get
        self._lap_ave_put_time = ave_put_time
        self._lap_ave_get_time = ave_get_time
        self._lap_dur_empty = dur_empty

        return QueuePerfStats(
            num_items=delta_num_get,
            elapsed=elapsed,
            ave_put_time=delta_ave_put_time,
            ave_get_time=delta_ave_get_time,
            p90_put_time=lap_put_p90,
            p99_put_time=lap_put_p99,
            p90_get_time=lap_get_p90,
            p99_get_time=lap_get_p99,
            occupancy_rate=max(0.0, occupancy_rate),
        )

    async def _log_interval_stats(self) -> None:
        await self.interval_stats_callback(self._get_lap_stats())

    async def interval_stats_callback(self, stats: QueuePerfStats) -> None:
        """Callback for processing interval performance statistics.

        When interval reporting is enabled, this method is periodically called
        with the delta metrics.

        The default behavior is to log the metrics to console.

        You can override this method and modify the destination of the log.
        """
        self._log_stats(stats)

    def _log_stats(self, stats: QueuePerfStats) -> None:
        _LG.info(
            "[%s]\tProcessed %5d items in %s (QPS: %6.1f) "
            "Ave wait time: put: %s, get (by next stage): %s. "
            "P90: put: %s, get: %s. P99: put: %s, get: %s. "
            "Occupancy rate: %d%%",
            f"{self.info}_queue",
            stats.num_items,
            _time_str(stats.elapsed),
            stats.qps,
            _time_str(stats.ave_put_time),
            _time_str(stats.ave_get_time),
            _time_str(stats.p90_put_time),
            _time_str(stats.p90_get_time),
            _time_str(stats.p99_put_time),
            _time_str(stats.p99_get_time),
            100 * stats.occupancy_rate,
            stacklevel=2,
        )


class _ThreadBasedAsyncQueue(AsyncQueue):
    """AsyncQueue backed by :py:class:`queue.Queue` instead of
    :py:class:`asyncio.Queue`.

    The default :py:class:`AsyncQueue` is an :py:class:`asyncio.Queue`.
    When the foreground (non-async) consumer thread needs the next item,
    it must go through ``asyncio.run_coroutine_threadsafe`` to schedule a
    ``get()`` coroutine on the background event loop, then poll the
    resulting ``Future``.  This cross-thread coroutine scheduling adds
    ~200-280us of CPU-side overhead per call — a fixed tax that cannot be
    hidden by GPU compute overlap.

    This class replaces the underlying storage with a standard
    :py:class:`queue.Queue`, whose blocking ``get``/``put`` release the
    GIL while waiting.  The async ``put``/``get`` methods delegate to the
    queue via ``run_in_executor`` so the event loop is never blocked.
    The foreground consumer can also call ``_queue.get(timeout=...)``
    directly, bypassing the asyncio event loop entirely and reducing
    handoff latency to ~10us when an item is already available.

    Benchmark (devserver, 500 int items, simulated foreground work)::

        FG work   default (p50)   default (p99)   thread_q (p50)   thread_q (p99)
        -------   -------------   -------------   --------------   --------------
          0ms         199us           625us           116us            313us
          3ms         246us           642us           223us            646us
          6ms         232us           532us           190us            555us
          9ms         287us           485us           153us            549us
         12ms         280us           822us            14us            464us
         15ms         221us           396us             8us            385us
         18ms         227us           431us             9us             70us
         21ms         224us           450us            11us             41us
         24ms         240us           533us            12us             34us
         27ms         227us           470us            12us             27us
         30ms         242us           512us            12us             26us

    See ``spdl/examples/benchmark_thread_output_queue.py`` for the full
    benchmark.
    """

    def __init__(
        self,
        info: StageInfo,
        *,
        buffer_size: int = 1,
    ) -> None:
        # Skip asyncio.Queue.__init__ — we don't use the underlying queue.
        self.info = info
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=buffer_size)

    async def put(self, item: object) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._queue.put, item)  # pyre-ignore[32]

    async def get(self) -> object:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._queue.get)  # pyre-ignore[32]

    def get_nowait(self) -> object:
        return self._queue.get_nowait()

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()


class _AsyncQueueWithSyncMirror(AsyncQueue):
    """Adapt an asyncio sink queue for non-blocking cross-thread reads.

    This adapter is used only with SPDL's identity-preserving built-in queue
    classes. Custom queues stay on their owning event loop so their ``put`` and
    ``get`` transformations and exceptions remain authoritative.

    The wrapped queue remains the source of backpressure and owns its stage hook. A
    standard-library ``queue.Queue`` mirrors items that have completed ``put`` so the
    foreground thread can inspect the sink without waiting for the event loop. After
    the foreground removes a mirrored item, it schedules ``_release_one`` on the
    event-loop thread to release the corresponding slot in the wrapped queue.

    This is intentionally only used for the terminal output queue. Internal pipeline
    queues have async consumers and do not need a cross-thread mirror.
    """

    def __init__(self, async_queue: AsyncQueue) -> None:
        # Initialize the base object so inherited diagnostics remain valid. Its
        # asyncio storage is intentionally unused; ``async_queue`` owns that state.
        super().__init__(async_queue.info, buffer_size=async_queue.maxsize)
        self._async_queue = async_queue
        self._sync_queue: queue.Queue[Any] = queue.Queue()
        self._mirror_available = asyncio.Event()
        self._pending_release_lock = threading.Lock()
        self._pending_releases = 0
        async_queue._put_commit_callback = self._publish_mirrored_item

    @property
    def maxsize(self) -> int:
        """The maximum number of items accepted by the wrapped queue."""
        return self._async_queue.maxsize

    async def put(self, item: object) -> None:
        """Put an item through the wrapped queue's mirrored commit boundary."""
        await self._async_queue.put(item)

    def put_nowait(self, item: object) -> None:
        """Put an item through the wrapped queue's mirrored commit boundary."""
        self._async_queue.put_nowait(item)

    def _publish_mirrored_item(self, item: object) -> None:
        """Publish an accepted wrapped item to the foreground mirror."""
        # The event is advisory and cannot expose an item by itself. Set it first
        # so a mirror-queue failure leaves no foreground-visible partial commit.
        self._mirror_available.set()
        self._sync_queue.put_nowait(item)

    async def get(self) -> object:
        """Get an item asynchronously while keeping the mirror synchronized."""
        while True:
            try:
                return self.get_nowait()
            except queue.Empty:
                # Foreground readers do not own this loop and cannot safely clear
                # its Event. Clear and recheck here so an item published between
                # the failed poll and ``wait`` cannot be missed.
                self._mirror_available.clear()
                try:
                    return self.get_nowait()
                except queue.Empty:
                    await self._mirror_available.wait()

    def get_nowait(self) -> object:
        """Get an item on the event-loop thread while keeping both queues aligned."""
        # Claim the externally visible item first. If a foreground reader won the
        # race, this raises without consuming the wrapped queue's capacity token.
        item = self._sync_queue.get_nowait()
        try:
            self._async_queue.get_nowait()
        except asyncio.QueueEmpty:
            # A defensive recovery for skew during shutdown: the visible mirror
            # item is already claimed, so do not discard it merely because its
            # wrapped capacity token was consumed elsewhere.
            pass
        return item

    def empty(self) -> bool:
        """Return whether the foreground-visible mirror is empty."""
        return self._sync_queue.empty()

    def full(self) -> bool:
        """Return whether the wrapped async queue is full."""
        return self._async_queue.full()

    def qsize(self) -> int:
        """Return the number of foreground-visible items."""
        return self._sync_queue.qsize()

    def task_done(self) -> None:
        """Delegate task accounting to the wrapped queue."""
        self._async_queue.task_done()

    async def join(self) -> None:
        """Wait for the wrapped queue's task accounting to reach zero."""
        await self._async_queue.join()

    def _get_lap_stats(self) -> QueuePerfStats:
        """Return interval statistics from a wrapped :class:`StatsQueue`."""
        if not isinstance(self._async_queue, StatsQueue):
            raise TypeError("Queue performance stats require a StatsQueue.")
        return self._async_queue._get_lap_stats()

    @asynccontextmanager
    async def stage_hook(self) -> AsyncIterator[None]:
        """Run the wrapped queue's initialization and finalization hook."""
        async with self._async_queue.stage_hook():
            yield

    def _register_release(self) -> None:
        """Record one wrapped slot claimed through the foreground mirror."""
        with self._pending_release_lock:
            self._pending_releases += 1

    def _claim_pending_release(self) -> bool:
        """Claim one registered foreground consumption for release."""
        with self._pending_release_lock:
            if self._pending_releases == 0:
                return False
            self._pending_releases -= 1
            return True

    def _release_one(self, retries_remaining: int = 1) -> None:
        """Release one mirrored item from the wrapped queue on its owning loop."""
        # Use the wrapped queue's public async path so custom queue behavior (in
        # particular StatsQueue's get-side accounting) still observes every
        # foreground consumption. SPDL's ``create_task`` observes every task
        # result and logs failures at error level, so this release cannot fail
        # as an unobserved fire-and-forget task.
        create_task(
            self._release_one_if_available(retries_remaining),
            name=f"{self.info}_release_mirrored_output",
        )

    def _release_one_nowait(self) -> None:
        """Release one slot directly after the wrapped queue's loop has closed."""
        if not self._claim_pending_release():
            return
        try:
            self._async_queue.get_nowait()
        except asyncio.QueueEmpty:
            # Shutdown may already have drained the wrapped side after the
            # foreground claimed its mirror entry.
            pass

    async def _release_one_if_available(self, retries_remaining: int = 1) -> None:
        """Release one item unless shutdown already drained it."""
        if not self._claim_pending_release():
            return
        # This adapter wraps exact AsyncQueue/StatsQueue instances. For both,
        # get() on a nonempty queue completes without yielding, so this check and
        # removal are atomic on the queue's event loop while retaining get stats.
        if self._async_queue.empty():
            # The wrapped slot corresponding to this claimed token was already
            # consumed by shutdown or defensive skew recovery. Restoring the
            # token here would let a stale callback consume a later refill.
            return
        size_before = self._async_queue.qsize()
        try:
            await self._async_queue.get()
        except asyncio.QueueEmpty:
            # Shutdown may already have drained the corresponding wrapped item.
            return
        except Exception:
            # Restore and retry only when the failed get left its slot in place.
            # After the one retry is exhausted, release that slot directly so a
            # bounded sink cannot remain permanently backpressured.
            if self._async_queue.qsize() >= size_before:
                if retries_remaining > 0:
                    self._register_release()
                    self._release_one(retries_remaining - 1)
                    return
                _LG.exception(
                    "Wrapped queue get failed after retries; releasing its slot directly."
                )
                try:
                    self._async_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                else:
                    return
            raise

    def _drain(self) -> None:
        """Drain both sides on the wrapped queue's owning event loop."""
        # Any wrapped entries corresponding to foreground reads are consumed by
        # this drain, so their already-scheduled callbacks must become no-ops.
        with self._pending_release_lock:
            self._pending_releases = 0
        while True:
            try:
                self._async_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        while True:
            try:
                self._sync_queue.get_nowait()
            except queue.Empty:
                break
        self._mirror_available.clear()


_DEFAULT_QUEUE_CLASS: type[AsyncQueue] = StatsQueue


def set_default_queue_class(queue_class: type[AsyncQueue] = StatsQueue) -> None:
    """Set the default queue class to be used for connecting pipeline stages.

    Args:
        queue_class: The queue class to use as default.
            Default: :py:class:`~spdl.pipeline.StatsQueue`.
    """
    global _DEFAULT_QUEUE_CLASS
    _DEFAULT_QUEUE_CLASS = StatsQueue if queue_class is None else queue_class


def get_default_queue_class() -> type[AsyncQueue]:
    """Get the currently configured default queue class.

    Returns:
        The default queue class, or ``None`` if not configured.
    """
    return _DEFAULT_QUEUE_CLASS
