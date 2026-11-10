# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Capture thread-sink observations; aggregate them on the owning event loop."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import replace
from typing import Any, TYPE_CHECKING

from ._common import _StatsCounter

if TYPE_CHECKING:
    from ._queue import QueuePerfStats

__all__ = ["_ThreadQueueWithStats"]

_LG: logging.Logger = logging.getLogger(__name__)
_COLLECT_INTERVAL = 0.02
_MAX_PENDING_SAMPLES = 16384
_AGGREGATION_BATCH_SIZE = 256


class _ThreadQueueWithStats(queue.Queue[Any]):
    """Capture numeric samples without aggregating in a foreground data call."""

    def __init__(self, maxsize: int) -> None:
        super().__init__(maxsize)
        self._capture_lock = threading.Lock()
        self._pending: deque[tuple[bool, float]] = deque()
        self._capture_error: str | None = None
        self._capture_closed = False
        self._aggregation_lock = asyncio.Lock()
        self._getc = _StatsCounter()
        self._putc = _StatsCounter()
        self._started = self._lap_t0 = self._empty_t0 = time.monotonic()
        self._snapshot_now = self._started
        self._snapshot_empty = 0.0
        self._dur_empty = self._lap_dur_empty = 0.0
        self._lap_num_get = self._lap_num_put = 0
        self._lap_get_total = self._lap_put_total = 0.0
        self._available = True
        self._aggregation_error: Exception | None = None
        self._snapshot: QueuePerfStats | None = None
        self._aggregation_thread: int | None = None

    @property
    def available(self) -> bool:
        return self._available

    @property
    def get_counter(self) -> _StatsCounter:
        return self._getc

    @property
    def put_counter(self) -> _StatsCounter:
        return self._putc

    def _capture(self, is_get: bool, elapsed: float) -> None:
        try:
            sample = (is_get, elapsed)
        except MemoryError:
            with self._capture_lock:
                self._capture_error = "numeric sample allocation failed"
            return
        with self._capture_lock:
            if self._capture_closed or self._capture_error is not None:
                return
            if len(self._pending) >= _MAX_PENDING_SAMPLES:
                self._capture_error = "numeric sample buffer overflow"
                return
            try:
                self._pending.append(sample)
            except MemoryError:
                self._capture_error = "numeric sample buffer allocation failed"

    def _disable(self, detail: str) -> None:
        with self._capture_lock:
            self._capture_error = detail
        if self._available:
            self._available = False
            _LG.error(
                "Thread sink statistics unavailable: %s; reporting disabled",
                detail,
            )

    def _check_stats(self) -> None:
        if not self._available:
            raise RuntimeError("Thread sink statistics are unavailable")

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        t0 = time.monotonic()
        super().put(item, block=block, timeout=timeout)
        self._capture(False, time.monotonic() - t0)

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        t0 = time.monotonic()
        item = super().get(block=block, timeout=timeout)
        self._capture(True, time.monotonic() - t0)
        return item

    def _put(self, item: Any) -> None:
        empty = not self._qsize()
        super()._put(item)
        if empty:
            self._dur_empty += time.monotonic() - self._empty_t0

    def _get(self) -> Any:
        item = super()._get()
        if not self._qsize():
            self._empty_t0 = time.monotonic()
        return item

    def _occupancy_snapshot(self) -> tuple[float, float]:
        with self.mutex:
            now = time.monotonic()
            empty = self._dur_empty
            if not self._qsize():
                empty += now - self._empty_t0
        return now, empty

    async def flush_stats(self) -> None:
        """Aggregate one detached capture boundary, yielding between fixed batches."""
        thread = threading.get_ident()
        if self._aggregation_thread not in (None, thread):
            raise RuntimeError("Thread sink aggregation belongs to another event loop")
        self._aggregation_thread = thread
        async with self._aggregation_lock:
            if self._aggregation_error is not None:
                raise self._aggregation_error
            pending: deque[tuple[bool, float]] = deque()
            now, empty = self._occupancy_snapshot()
            with self._capture_lock:
                pending, self._pending = self._pending, pending
                error = self._capture_error
            if error is not None:
                self._disable(error)
                return
            try:
                processed = 0
                while pending:
                    is_get, elapsed = pending.popleft()
                    (self._getc if is_get else self._putc).update(elapsed)
                    processed += 1
                    if processed == _AGGREGATION_BATCH_SIZE and pending:
                        processed = 0
                        # Cooperative aggregation, not a data wakeup/retry mechanism.
                        await asyncio.sleep(0)
            except MemoryError:
                self._disable("counter aggregation allocation failed")
                return
            except Exception as error:
                self._aggregation_error = error
                self._disable(f"aggregation failed: {error!r}")
                raise
            with self._capture_lock:
                error = self._capture_error
            if error is not None:
                self._disable(error)
                return
            self._snapshot_now, self._snapshot_empty = now, empty
            self._snapshot = self._make_stats()

    async def collect_stats(self, done: asyncio.Event) -> None:
        """Own the finite sampling pump through stage completion, even without reports."""
        try:
            while not done.is_set() and self._available:
                try:
                    await asyncio.wait_for(done.wait(), timeout=_COLLECT_INTERVAL)
                except TimeoutError:
                    pass
                await self.flush_stats()
        finally:
            with self._capture_lock:
                self._capture_closed = True
        await self.flush_stats()

    async def start_stats(self) -> None:
        with self._capture_lock:
            self._capture_closed = False
        await self.flush_stats()
        with self.mutex:
            self._empty_t0 = time.monotonic()
            self._dur_empty = 0.0
            started = self._empty_t0
        self._started = self._lap_t0 = self._snapshot_now = started
        self._snapshot_empty = self._lap_dur_empty = 0.0
        self._lap_num_get = self._getc.num_items
        self._lap_num_put = self._putc.num_items
        self._lap_get_total = self._getc.ave_time * self._lap_num_get
        self._lap_put_total = self._putc.ave_time * self._lap_num_put
        self._snapshot = self._make_stats()

    def get_stats(self) -> QueuePerfStats:
        """Copy the last completed snapshot without processing pending observations."""
        self._check_stats()
        if self._snapshot is None:
            raise RuntimeError("Thread sink statistics have not been collected")
        return replace(self._snapshot)

    def _make_stats(self) -> QueuePerfStats:
        from ._queue import QueuePerfStats

        elapsed = self._snapshot_now - self._started
        return QueuePerfStats(
            elapsed=elapsed,
            num_items=self._getc.num_items,
            ave_put_time=self._putc.ave_time,
            ave_get_time=self._getc.ave_time,
            p90_put_time=self._putc.p90_time,
            p99_put_time=self._putc.p99_time,
            p90_get_time=self._getc.p90_time,
            p99_get_time=self._getc.p99_time,
            occupancy_rate=0.0
            if elapsed <= 0
            else max(0.0, 1 - self._snapshot_empty / elapsed),
        )

    def get_lap_stats(self) -> QueuePerfStats:
        from ._queue import QueuePerfStats

        self._check_stats()
        if (
            threading.get_ident() != self._aggregation_thread
            or self._aggregation_lock.locked()
        ):
            raise RuntimeError(
                "Interval statistics require a settled owning-loop aggregation"
            )
        now, empty = self._snapshot_now, self._snapshot_empty
        elapsed = now - self._lap_t0
        gets, puts = self._getc.num_items, self._putc.num_items
        get_total = self._getc.ave_time * gets
        put_total = self._putc.ave_time * puts
        num_get = gets - self._lap_num_get
        num_put = puts - self._lap_num_put
        put90, put99 = self._putc.consume_lap_percentiles()
        get90, get99 = self._getc.consume_lap_percentiles()
        stats = QueuePerfStats(
            elapsed=elapsed,
            num_items=num_get,
            ave_put_time=0.0
            if not num_put
            else (put_total - self._lap_put_total) / num_put,
            ave_get_time=0.0
            if not num_get
            else (get_total - self._lap_get_total) / num_get,
            p90_put_time=put90,
            p99_put_time=put99,
            p90_get_time=get90,
            p99_get_time=get99,
            occupancy_rate=0.0
            if elapsed <= 0
            else max(0.0, 1 - (empty - self._lap_dur_empty) / elapsed),
        )
        self._lap_t0, self._lap_dur_empty = now, empty
        self._lap_num_get, self._lap_num_put = gets, puts
        self._lap_get_total, self._lap_put_total = get_total, put_total
        return stats
