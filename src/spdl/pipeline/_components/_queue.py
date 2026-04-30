# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from spdl.pipeline._common._misc import create_task

from ._common import _EOF, _periodic_dispatch, _StatsCounter, _time_str, StageInfo

__all__ = [
    "_queue_stage_hook",
    "AsyncQueue",
    "StatsQueue",
    "QueuePerfStats",
    "set_default_queue_class",
    "get_default_queue_class",
]

_LG: logging.Logger = logging.getLogger(__name__)


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
    async with queue.stage_hook():
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

        # Cached snapshot of the latest lap stats. ``_get_lap_stats()`` is
        # destructive — it resets the lap counters — so non-callback readers
        # (e.g., the LCA ``DomeVideoConcurrencyController``) cannot call it
        # safely. ``_log_interval_stats()`` now writes the freshly computed
        # stats here so an external reader can observe the latest interval
        # without disturbing the periodic logging path. ``None`` until the
        # first interval has elapsed.
        self._last_lap_stats: QueuePerfStats | None = None

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
        stats = self._get_lap_stats()
        # Cache for non-callback readers (e.g., adaptive concurrency
        # controller). ``_get_lap_stats()`` resets the lap counters, so
        # this is the only safe way for an external reader to observe
        # the most recent interval's stats without racing the periodic
        # logging path.
        self._last_lap_stats = stats
        await self.interval_stats_callback(stats)

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
