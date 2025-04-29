# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TypeVar

from ._hook import _periodic_dispatch, _StatsCounter, _time_str
from ._utils import create_task

__all__ = [
    "AsyncQueue",
    "StatsQueue",
]

T = TypeVar("T")
_LG: logging.Logger = logging.getLogger(__name__)


class AsyncQueue(asyncio.Queue[T]):
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
        name: The name of the queue. Assigned by :py:class:`PipelineBuilder`.
        buffer_size: The buffer size. Assigned by :py:class:`PipelineBuilder`.
    """

    def __init__(self, name: str, *, buffer_size: int = 1) -> None:
        super().__init__(buffer_size)
        self.name = name

    @asynccontextmanager
    async def stage_hook(self) -> AsyncIterator[None]:
        """Context manager, which handles init/final logic for the stage."""
        yield


@dataclass
class _QueuePerfStats:
    elapsed: float
    num_items: int
    ave_put_time: float
    ave_get_time: float
    occupancy_rate: float

    @property
    def qps(self) -> float:
        if self.elapsed <= 0:
            return 0
        return self.num_items / self.elapsed


class StatsQueue(AsyncQueue[T]):
    """Measures the time stages are blocked on upstream/downstream stage.
    Extends :py:class:`AsyncQueue`.

    Args:
        name: The name of the queue. Assigned by :py:class:`PipelineBuilder`.
        buffer_size: The buffer size. Assigned by :py:class:`PipelineBuilder`.
        interval: The interval (in second) between repoting performance numbers
            to console.
    """

    def __init__(
        self,
        name: str,
        *,
        buffer_size: int = 1,
        interval: float = -1,
    ) -> None:
        super().__init__(name, buffer_size=buffer_size)

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

    async def get(self) -> T:
        """Remove and return an item from the queue, track the time."""
        with self._getc.count():
            item = await super().get()

        if self.qsize() == 0:
            self._empty_t0 = time.monotonic()

        return item

    async def put(self, item: T) -> None:
        """Remove and return an item from the queue, track the time."""
        if self.qsize() == 0:
            self._dur_empty += time.monotonic() - self._empty_t0

        with self._putc.count():
            return await super().put(item)

    @asynccontextmanager
    async def stage_hook(self) -> AsyncIterator[None]:
        t0 = self._lap_t0 = self._empty_t0 = time.monotonic()
        if self.interval > 0:
            report = create_task(
                _periodic_dispatch(self._log_interval_stats, self.interval),
                name=f"{self.name}_periodic_report",
                log_cancelled=False,
            )

        try:
            yield
        finally:
            if self.interval > 0:
                report.cancel()

            elapsed = time.monotonic() - t0
            occupancy_rate = 0.0 if elapsed <= 0 else 1 - self._dur_empty / elapsed
            self._log_stats(
                _QueuePerfStats(
                    elapsed=elapsed,
                    num_items=self._getc.num_items,
                    ave_put_time=self._putc.ave_time,
                    ave_get_time=self._getc.ave_time,
                    occupancy_rate=max(0.0, occupancy_rate),
                )
            )

    def _get_lap_stats(self) -> _QueuePerfStats:
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

        # Update the lap
        self._lap_t0 = now
        self._lap_num_put = num_put
        self._lap_num_get = num_get
        self._lap_ave_put_time = ave_put_time
        self._lap_ave_get_time = ave_get_time
        self._lap_dur_empty = dur_empty

        return _QueuePerfStats(
            num_items=delta_num_get,
            elapsed=elapsed,
            ave_put_time=delta_ave_put_time,
            ave_get_time=delta_ave_get_time,
            occupancy_rate=max(0.0, occupancy_rate),
        )

    async def _log_interval_stats(self) -> None:
        stats = self._get_lap_stats()
        self._log_stats(stats)

    def _log_stats(self, stats: _QueuePerfStats) -> None:
        _LG.info(
            "[%s]\tProcessed %5d items in %s (QPS: %6.1f) "
            "Ave wait time: put: %s, get (by next stage): %s. "
            "Occupancy rate: %d%%",
            self.name,
            stats.num_items,
            _time_str(stats.elapsed),
            stats.qps,
            _time_str(stats.ave_put_time),
            _time_str(stats.ave_get_time),
            100 * stats.occupancy_rate,
            stacklevel=2,
        )
