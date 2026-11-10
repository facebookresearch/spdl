# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Thread-backed pipeline handoff and optional background statistics."""

from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from spdl.pipeline._common._misc import create_task

from ._common import _periodic_dispatch, _ShieldedHook, StageInfo
from ._queue import AsyncQueue, QueuePerfStats, StatsQueue
from ._thread_stats import _ThreadQueueWithStats

__all__ = [
    "ThreadOutputQueue",
    "StatsThreadOutputQueue",
    "_make_thread_output_queue",
    "_resolve_thread_output_queue_selection",
    "ThreadOutputQueueConfig",
    "get_default_thread_output_queue_config",
    "set_default_thread_output_queue_config",
]


class ThreadOutputQueue(AsyncQueue):
    """A thread-backed sink with direct foreground reads and no statistics.

    Pass this class, or a subclass, as ``use_thread_output_queue`` to
    :py:meth:`~spdl.pipeline.PipelineBuilder.build` or
    :py:func:`~spdl.pipeline.build_pipeline`. An omitted or ``None`` selector
    uses :py:class:`~spdl.pipeline.config.ThreadOutputQueueConfig`; its initial OSS
    configuration keeps the async sink. ``False`` explicitly keeps the async
    sink, and ``True`` selects the configured thread-output class. An explicit
    class takes priority over the configuration.
    Pass a class, not a prebuilt instance; the builder constructs the sink.

    Internal stage queues still use ``queue_class``. This plain sink does not
    collect statistics, require logging, or persist data. For optional
    background measurements and custom reporting, use
    :py:class:`~spdl.pipeline.StatsThreadOutputQueue`.

    Subclasses must accept the constructor keywords below and preserve bounded
    queue and lifecycle semantics. Override ``stage_hook`` to customize
    initialization/finalization. Classes passed through subprocess configuration
    must be importable and picklable, like other pipeline configuration classes.

    Args:
        info: Stage identity assigned by the builder.
        buffer_size: Maximum queued items; zero permits unbounded buffering.
        interval: Reporting interval supplied by the builder. This plain base
            ignores it; statistics-enabled subclasses may use it.

    Example::

        from spdl.pipeline import PipelineBuilder, ThreadOutputQueue

        pipeline = (
            PipelineBuilder()
            .add_source(range(100))
            .add_sink(8)
            .build(num_threads=2, use_thread_output_queue=ThreadOutputQueue)
        )
        with pipeline.auto_stop():
            for item in pipeline:
                print(item)

    Benchmark comparison
    --------------------

    The complete three-way comparison below measures foreground iterator
    handoff latency on a Linux CPU host with CPython 3.12.16. Lower is better.
    It compares ``use_thread_output_queue=False``, ``ThreadOutputQueue`` and
    ``StatsThreadOutputQueue`` using the same source-to-sink workload: 500
    integer items, buffer capacity 8, two configured executor workers and 20
    warmup items. Timed and untimed reads alternate, giving 240 observations
    per cell. Each cell lists **mean / p50 / p95 / p99**, all in microseconds.

    .. list-table:: Foreground handoff latency (mean / p50 / p95 / p99, us)
       :header-rows: 1
       :widths: 8 30 30 32

       * - Foreground work (ms)
         - ``False``
         - ``ThreadOutputQueue``
         - ``StatsThreadOutputQueue``
       * - 0
         - 142.6 / 142.2 / 290.7 / 340.8
         - 87.5 / 95.7 / 190.1 / 251.2
         - 106.3 / 102.7 / 220.2 / 286.5
       * - 3
         - 152.2 / 170.3 / 352.5 / 472.1
         - 119.9 / 128.5 / 303.1 / 368.3
         - 133.9 / 128.1 / 328.9 / 478.0
       * - 6
         - 53.4 / 11.9 / 242.2 / 357.9
         - 91.3 / 102.4 / 251.6 / 375.8
         - 89.7 / 13.8 / 314.6 / 380.3
       * - 9
         - 51.5 / 19.1 / 288.1 / 340.5
         - 62.8 / 10.2 / 208.5 / 308.6
         - 42.1 / 11.3 / 261.8 / 402.8
       * - 12
         - 23.4 / 21.2 / 42.6 / 87.9
         - 14.8 / 9.6 / 22.3 / 175.3
         - 22.7 / 9.9 / 177.2 / 248.5
       * - 15
         - 22.6 / 20.6 / 34.6 / 46.4
         - 12.7 / 11.8 / 21.0 / 28.7
         - 16.5 / 14.9 / 33.8 / 39.9
       * - 18
         - 23.4 / 20.5 / 39.1 / 87.6
         - 14.5 / 12.9 / 22.6 / 58.9
         - 19.6 / 17.6 / 34.3 / 59.9
       * - 21
         - 24.3 / 21.9 / 40.8 / 57.0
         - 14.5 / 14.0 / 22.1 / 33.1
         - 19.1 / 17.5 / 34.4 / 49.6
       * - 24
         - 25.7 / 22.9 / 49.5 / 63.1
         - 17.2 / 15.4 / 32.7 / 41.6
         - 17.7 / 17.1 / 28.2 / 40.0
       * - 27
         - 25.7 / 22.2 / 55.2 / 73.1
         - 15.2 / 14.3 / 26.5 / 39.9
         - 19.7 / 17.6 / 39.1 / 64.6
       * - 30
         - 36.9 / 31.3 / 64.7 / 91.1
         - 32.3 / 29.2 / 64.4 / 86.6
         - 33.0 / 30.5 / 58.3 / 69.7

    Results are one full sweep, in the listed variant order at each work
    setting, not interleaved repeated trials. The foreground work is a CPU
    busy loop that contends for the GIL, not GPU work. The timed operation is
    advancing a newly created pipeline iterator, not a bare queue call.
    The ``False`` control uses the built-in ``StatsQueue``/sync-mirror sink;
    both thread classes are selected explicitly. Internal queues/task hooks
    use ordinary defaults in all variants. The statistics class retains
    collection and its default final reporting callback; periodic reporting
    is disabled. P99 has few tail samples, and these lightweight results do
    not establish sustained-load, GPU-overlap or training performance.

    Reproduce the sweep with
    :py:mod:`spdl.examples.benchmark_thread_output_queue`::

        python -m spdl.examples.benchmark_thread_output_queue

    .. versionadded:: 0.9.0
    """

    def __init__(
        self, info: StageInfo, *, buffer_size: int = 1, interval: float = -1
    ) -> None:
        # The thread buffer owns storage/accounting instead of asyncio.Queue.
        self.info = info
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=buffer_size)

    async def put(self, item: object) -> None:
        """Enqueue an item, delegating full or unbounded queues to an executor.

        Args:
            item: The value to enqueue. A bounded queue with available capacity
                accepts it directly without an executor round trip.
        """
        if self._queue.maxsize > 0:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                pass
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._queue.put, item)

    async def get(self) -> object:
        """Wait for an item through an executor without blocking the event loop.

        Returns:
            The next queued value.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._queue.get)

    def get_nowait(self) -> object:
        """Return a queued value without waiting, raising queue.Empty if unavailable.

        Returns:
            The next queued value.
        """
        return self._queue.get_nowait()

    def empty(self) -> bool:
        """Return whether the underlying thread buffer is empty."""
        return self._queue.empty()

    def qsize(self) -> int:
        """Return the number of queued items."""
        return self._queue.qsize()

    @property
    def maxsize(self) -> int:
        """The configured item-capacity limit; zero means unbounded."""
        return self._queue.maxsize

    def full(self) -> bool:
        """Return whether the underlying thread buffer is full."""
        return self._queue.full()

    def put_nowait(self, item: object) -> None:
        """Enqueue without waiting, raising queue.Full when capacity is exhausted.

        Args:
            item: The value to enqueue.
        """
        self._queue.put_nowait(item)

    def task_done(self) -> None:
        """Acknowledge one previously enqueued item for join accounting."""
        self._queue.task_done()

    async def join(self) -> None:
        """Wait for every queued item to be acknowledged with task_done."""
        await asyncio.get_running_loop().run_in_executor(None, self._queue.join)


@dataclass(frozen=True, kw_only=True)
class ThreadOutputQueueConfig:
    """Process-local defaults for selecting the final pipeline sink.

    Configure the sink explicitly with
    :py:func:`~spdl.pipeline.config.set_default_thread_output_queue_config` rather than
    changing build-function keyword defaults. An omitted or ``None``
    ``use_thread_output_queue`` option uses ``enabled``. Explicit ``False``
    always keeps the async sink; ``True`` selects ``queue_class``; a caller's
    explicit queue class takes priority over this configuration.

    Configure defaults before building pipelines. Existing pipelines retain
    their queue instances. Internal stage queues and task hooks are independent.

    Example::

        from spdl.pipeline import StatsThreadOutputQueue
        from spdl.pipeline.config import (
            ThreadOutputQueueConfig,
            set_default_thread_output_queue_config,
        )

        set_default_thread_output_queue_config(
            ThreadOutputQueueConfig(
                enabled=True, queue_class=StatsThreadOutputQueue
            )
        )

    .. versionadded:: 0.9.0
    """

    enabled: bool = False
    """Whether omitted/None build options select the thread-backed sink."""

    queue_class: type[ThreadOutputQueue] = ThreadOutputQueue
    """The class used when the thread-backed sink is enabled without an explicit class."""


_DEFAULT_THREAD_OUTPUT_QUEUE_CONFIG: ThreadOutputQueueConfig | None = None


def get_default_thread_output_queue_config() -> ThreadOutputQueueConfig:
    """Return the immutable process-local thread-sink configuration.

    Returns:
        The installed configuration, or the OSS initial configuration with
        ``enabled=False`` and plain :py:class:`~spdl.pipeline.ThreadOutputQueue`.

    .. versionadded:: 0.9.0
    """
    if _DEFAULT_THREAD_OUTPUT_QUEUE_CONFIG is None:
        return ThreadOutputQueueConfig()
    return _DEFAULT_THREAD_OUTPUT_QUEUE_CONFIG


def set_default_thread_output_queue_config(config: ThreadOutputQueueConfig) -> None:
    """Set explicit thread-sink defaults for subsequently built pipelines.

    Args:
        config: Immutable defaults. Configure them before building pipelines;
            already-built pipelines and explicit caller selections are unchanged.

    Raises:
        TypeError: If the configuration, enabled flag or queue class is invalid.

    .. seealso::

       :py:class:`~spdl.pipeline.config.ThreadOutputQueueConfig`
          Selection rules and a configuration example.

    .. versionadded:: 0.9.0
    """
    if not isinstance(config, ThreadOutputQueueConfig):
        raise TypeError("config must be ThreadOutputQueueConfig")
    if not isinstance(config.enabled, bool):
        raise TypeError("enabled must be bool")
    if not isinstance(config.queue_class, type) or not issubclass(
        config.queue_class, ThreadOutputQueue
    ):
        raise TypeError("queue_class must inherit ThreadOutputQueue")
    global _DEFAULT_THREAD_OUTPUT_QUEUE_CONFIG
    _DEFAULT_THREAD_OUTPUT_QUEUE_CONFIG = config


def _resolve_thread_output_queue_selection(
    selection: bool | type[ThreadOutputQueue] | None,
) -> bool | type[ThreadOutputQueue]:
    """Resolve an omitted selector in the calling build/worker environment."""
    if selection is None:
        return get_default_thread_output_queue_config().enabled
    return selection


def _make_thread_output_queue(
    info: StageInfo,
    buffer_size: int,
    selection: bool | type[ThreadOutputQueue],
    interval: float,
) -> ThreadOutputQueue:
    """Instantiate an explicit class or the configured True default."""
    cls = (
        get_default_thread_output_queue_config().queue_class
        if selection is True
        else selection
    )
    if not isinstance(cls, type) or not issubclass(cls, ThreadOutputQueue):
        raise TypeError(
            "use_thread_output_queue must be bool or a ThreadOutputQueue class"
        )
    return cls(info, buffer_size=buffer_size, interval=interval)


class StatsThreadOutputQueue(ThreadOutputQueue):
    """A thread sink with background statistics and overridable reporting hooks.

    Select this class explicitly with ``use_thread_output_queue`` in
    :py:meth:`~spdl.pipeline.PipelineBuilder.build` or
    :py:func:`~spdl.pipeline.build_pipeline`. Foreground operations capture
    numeric observations; background batches update statistics. No storage
    backend is required. The default callbacks use Python logging; subclasses
    can instead keep packets in memory, display them, or send them to an
    application-owned destination. See
    :py:class:`~spdl.pipeline.ThreadOutputQueue` for class selection and the
    subclass constructor contract.

    Args:
        info: Stage identity assigned by the builder.
        buffer_size: Maximum queued items; zero permits unbounded buffering.
        interval: Positive seconds between interval callbacks. Nonpositive values
            disable interval reports, not collection or the final callback.

    Example without logging or a storage backend::

        from collections import deque
        from spdl.pipeline import (
            AsyncQueue,
            PipelineBuilder,
            QueuePerfStats,
            StatsThreadOutputQueue,
        )

        packets = deque(maxlen=100)

        class MyStatsSink(StatsThreadOutputQueue):
            async def interval_stats_callback(self, stats: QueuePerfStats) -> None:
                packets.append(stats)

            async def final_stats_callback(self, stats: QueuePerfStats) -> None:
                packets.append(stats)

        pipeline = (
            PipelineBuilder()
            .add_source(range(100))
            .add_sink(8)
            .build(
                num_threads=2,
                queue_class=AsyncQueue,
                task_hook_factory=lambda _: [],
                use_thread_output_queue=MyStatsSink,
                report_stats_interval=5,
            )
        )
        with pipeline.auto_stop():
            for item in pipeline:
                print(item)

    The example disables statistics on internal queues and task hooks; those
    choices are independent of the final sink's callbacks. A nonpositive
    ``report_stats_interval`` disables interval reports, not collection or the
    final callback. Changing logger levels does not disable measurement. Select
    :py:class:`~spdl.pipeline.ThreadOutputQueue` to avoid sink tracking entirely.

    Callbacks run on the background event loop. Keep their work bounded and avoid
    blocking that loop. They must terminate; finalization waits for the final
    callback. Classes passed through subprocess configuration must be importable
    and picklable, like other pipeline configuration classes.

    Reporting covers the producer-stage window. Items buffered when that stage
    finishes remain readable, but those later reads do not extend the final
    measurement window. ``num_items`` counts queue removals during that window,
    including cleanup drains, not necessarily batches consumed by a training loop.
    Overflow of bounded numeric capture marks statistics unavailable and
    suppresses invalid reports with a diagnostic instead of blocking data or
    silently biasing percentiles. Counter-allocation exhaustion also disables
    reporting. Unexpected aggregation errors propagate through the collector
    and stage finalization rather than silently discarding stage metrics.

    This customization does not change stage execution or serialization and does
    not make the thread-output option the default. Measure the intended workload
    before choosing an implementation.

    Benchmark comparison
    --------------------

    The complete three-way comparison below measures foreground iterator
    handoff latency on a Linux CPU host with CPython 3.12.16. Lower is better.
    It compares ``use_thread_output_queue=False``, ``ThreadOutputQueue`` and
    ``StatsThreadOutputQueue`` using the same source-to-sink workload: 500
    integer items, buffer capacity 8, two configured executor workers and 20
    warmup items. Timed and untimed reads alternate, giving 240 observations
    per cell. Each cell lists **mean / p50 / p95 / p99**, all in microseconds.

    .. list-table:: Foreground handoff latency (mean / p50 / p95 / p99, us)
       :header-rows: 1
       :widths: 8 30 30 32

       * - Foreground work (ms)
         - ``False``
         - ``ThreadOutputQueue``
         - ``StatsThreadOutputQueue``
       * - 0
         - 142.6 / 142.2 / 290.7 / 340.8
         - 87.5 / 95.7 / 190.1 / 251.2
         - 106.3 / 102.7 / 220.2 / 286.5
       * - 3
         - 152.2 / 170.3 / 352.5 / 472.1
         - 119.9 / 128.5 / 303.1 / 368.3
         - 133.9 / 128.1 / 328.9 / 478.0
       * - 6
         - 53.4 / 11.9 / 242.2 / 357.9
         - 91.3 / 102.4 / 251.6 / 375.8
         - 89.7 / 13.8 / 314.6 / 380.3
       * - 9
         - 51.5 / 19.1 / 288.1 / 340.5
         - 62.8 / 10.2 / 208.5 / 308.6
         - 42.1 / 11.3 / 261.8 / 402.8
       * - 12
         - 23.4 / 21.2 / 42.6 / 87.9
         - 14.8 / 9.6 / 22.3 / 175.3
         - 22.7 / 9.9 / 177.2 / 248.5
       * - 15
         - 22.6 / 20.6 / 34.6 / 46.4
         - 12.7 / 11.8 / 21.0 / 28.7
         - 16.5 / 14.9 / 33.8 / 39.9
       * - 18
         - 23.4 / 20.5 / 39.1 / 87.6
         - 14.5 / 12.9 / 22.6 / 58.9
         - 19.6 / 17.6 / 34.3 / 59.9
       * - 21
         - 24.3 / 21.9 / 40.8 / 57.0
         - 14.5 / 14.0 / 22.1 / 33.1
         - 19.1 / 17.5 / 34.4 / 49.6
       * - 24
         - 25.7 / 22.9 / 49.5 / 63.1
         - 17.2 / 15.4 / 32.7 / 41.6
         - 17.7 / 17.1 / 28.2 / 40.0
       * - 27
         - 25.7 / 22.2 / 55.2 / 73.1
         - 15.2 / 14.3 / 26.5 / 39.9
         - 19.7 / 17.6 / 39.1 / 64.6
       * - 30
         - 36.9 / 31.3 / 64.7 / 91.1
         - 32.3 / 29.2 / 64.4 / 86.6
         - 33.0 / 30.5 / 58.3 / 69.7

    Results are one full sweep, in the listed variant order at each work
    setting, not interleaved repeated trials. The foreground work is a CPU
    busy loop that contends for the GIL, not GPU work. The timed operation is
    advancing a newly created pipeline iterator, not a bare queue call.
    The ``False`` control uses the built-in ``StatsQueue``/sync-mirror sink;
    both thread classes are selected explicitly. Internal queues/task hooks
    use ordinary defaults in all variants. The statistics class retains
    collection and its default final reporting callback; periodic reporting
    is disabled. P99 has few tail samples, and these lightweight results do
    not establish sustained-load, GPU-overlap or training performance.

    Reproduce the sweep with
    :py:mod:`spdl.examples.benchmark_thread_output_queue`::

        python -m spdl.examples.benchmark_thread_output_queue

    .. versionadded:: 0.9.0
    """

    def __init__(
        self, info: StageInfo, *, buffer_size: int = 1, interval: float = -1
    ) -> None:
        super().__init__(info, buffer_size=buffer_size, interval=interval)
        provider = _ThreadQueueWithStats(buffer_size)
        self._queue = provider
        self._stats_provider = provider
        self.interval = interval
        self._stats_logger = StatsQueue(info)

    @asynccontextmanager
    async def stage_hook(self) -> AsyncIterator[None]:
        """Own collection and reporting through stage completion and cancellation."""
        async with _ShieldedHook(self._collect_and_report()):
            yield

    @asynccontextmanager
    async def _collect_and_report(self) -> AsyncIterator[None]:
        provider = self._stats_provider
        await provider.start_stats()
        collector_done = asyncio.Event()
        collector = create_task(
            provider.collect_stats(collector_done),
            name=f"{self.info}_queue_collect_stats",
        )
        report_done = asyncio.Event()
        report = None
        if self.interval > 0:
            report = create_task(
                _periodic_dispatch(self._report_interval, report_done, self.interval),
                name=f"{self.info}_queue_periodic_report",
                log_cancelled=False,
            )
        try:
            yield
        finally:
            try:
                report_done.set()
                if report is not None:
                    await report
            finally:
                collector_done.set()
                await collector
            if provider.available:
                await self.final_stats_callback(provider.get_stats())

    async def _report_interval(self) -> None:
        if not self._stats_provider.available:
            return
        await self._stats_provider.flush_stats()
        if self._stats_provider.available:
            await self.interval_stats_callback(self._stats_provider.get_lap_stats())

    async def interval_stats_callback(self, stats: QueuePerfStats) -> None:
        """Handle one interval packet on the background loop.

        Args:
            stats: Completed interval statistics. The default logs them; override
                this method to choose a different destination without storage
                dependencies in the queue implementation.
        """
        self._stats_logger._log_stats(stats)

    async def final_stats_callback(self, stats: QueuePerfStats) -> None:
        """Handle the final producer-stage packet after pending samples are flushed.

        Args:
            stats: Completed cumulative stage-window statistics. The default
                logs them; override this method to retain or consume them.
        """
        self._stats_logger._log_stats(stats)
