# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import unittest
from collections.abc import Callable

from spdl.pipeline._components._common import _percentile, _StatsCounter
from spdl.pipeline._components._hook import TaskPerfStats, TaskStatsHook
from spdl.pipeline._components._queue import QueuePerfStats, StatsQueue


class PercentileTest(unittest.TestCase):
    def test_empty_list(self) -> None:
        self.assertEqual(_percentile([], 90), 0.0)
        self.assertEqual(_percentile([], 99), 0.0)

    def test_single_element(self) -> None:
        self.assertEqual(_percentile([5.0], 90), 5.0)
        self.assertEqual(_percentile([5.0], 99), 5.0)

    def test_two_elements(self) -> None:
        self.assertEqual(_percentile([1.0, 2.0], 90), 2.0)
        self.assertEqual(_percentile([1.0, 2.0], 50), 2.0)

    def test_ten_elements(self) -> None:
        times = [float(i) for i in range(10)]
        self.assertEqual(_percentile(times, 90), 9.0)
        self.assertEqual(_percentile(times, 50), 5.0)

    def test_hundred_elements(self) -> None:
        times = [float(i) for i in range(100)]
        self.assertEqual(_percentile(times, 90), 90.0)
        self.assertEqual(_percentile(times, 99), 99.0)
        self.assertEqual(_percentile(times, 50), 50.0)

    def test_unsorted_input(self) -> None:
        times = [9.0, 1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 0.0]
        self.assertEqual(_percentile(times, 90), 9.0)
        self.assertEqual(_percentile(times, 50), 5.0)

    def test_p0(self) -> None:
        times = [1.0, 2.0, 3.0]
        self.assertEqual(_percentile(times, 0), 1.0)

    def test_p100(self) -> None:
        times = [1.0, 2.0, 3.0]
        self.assertEqual(_percentile(times, 100), 3.0)


class StatsCounterPercentileTest(unittest.TestCase):
    def test_initial_state(self) -> None:
        counter = _StatsCounter()
        self.assertEqual(counter.p90_time, 0.0)
        self.assertEqual(counter.p99_time, 0.0)
        self.assertEqual(counter.num_items, 0)
        self.assertEqual(counter.ave_time, 0.0)

    def test_update_tracks_times(self) -> None:
        counter = _StatsCounter()
        for i in range(10):
            counter.update(float(i))
        self.assertEqual(counter.num_items, 10)
        self.assertEqual(counter.p90_time, 9.0)

    def test_times_since(self) -> None:
        counter = _StatsCounter()
        for i in range(5):
            counter.update(float(i))
        since = counter.times_since(3)
        self.assertEqual(since, [3.0, 4.0])

    def test_times_since_zero(self) -> None:
        counter = _StatsCounter()
        for i in range(3):
            counter.update(float(i))
        self.assertEqual(counter.times_since(0), [0.0, 1.0, 2.0])

    def test_count_context_manager(self) -> None:
        counter = _StatsCounter()
        with counter.count():
            pass
        self.assertEqual(counter.num_items, 1)
        self.assertGreaterEqual(counter.p90_time, 0.0)
        self.assertGreaterEqual(counter.p99_time, 0.0)

    def test_percentiles_with_100_samples(self) -> None:
        counter = _StatsCounter()
        for i in range(100):
            counter.update(float(i))
        self.assertEqual(counter.p90_time, 90.0)
        self.assertEqual(counter.p99_time, 99.0)


class TaskPerfStatsTest(unittest.TestCase):
    def test_fields_present(self) -> None:
        stats = TaskPerfStats(
            num_tasks=10,
            num_failures=1,
            ave_time=0.5,
            p90_time=0.8,
            p99_time=1.2,
        )
        self.assertEqual(stats.num_tasks, 10)
        self.assertEqual(stats.num_failures, 1)
        self.assertEqual(stats.ave_time, 0.5)
        self.assertEqual(stats.p90_time, 0.8)
        self.assertEqual(stats.p99_time, 1.2)


class QueuePerfStatsTest(unittest.TestCase):
    def test_fields_present(self) -> None:
        stats = QueuePerfStats(
            elapsed=60.0,
            num_items=100,
            ave_put_time=0.01,
            ave_get_time=0.02,
            p90_put_time=0.015,
            p99_put_time=0.025,
            p90_get_time=0.03,
            p99_get_time=0.04,
            occupancy_rate=0.75,
        )
        self.assertEqual(stats.p90_put_time, 0.015)
        self.assertEqual(stats.p99_put_time, 0.025)
        self.assertEqual(stats.p90_get_time, 0.03)
        self.assertEqual(stats.p99_get_time, 0.04)

    def test_qps(self) -> None:
        stats = QueuePerfStats(
            elapsed=10.0,
            num_items=100,
            ave_put_time=0.0,
            ave_get_time=0.0,
            p90_put_time=0.0,
            p99_put_time=0.0,
            p90_get_time=0.0,
            p99_get_time=0.0,
            occupancy_rate=0.0,
        )
        self.assertAlmostEqual(stats.qps, 10.0)

    def test_qps_zero_elapsed(self) -> None:
        stats = QueuePerfStats(
            elapsed=0.0,
            num_items=100,
            ave_put_time=0.0,
            ave_get_time=0.0,
            p90_put_time=0.0,
            p99_put_time=0.0,
            p90_get_time=0.0,
            p99_get_time=0.0,
            occupancy_rate=0.0,
        )
        self.assertEqual(stats.qps, 0)


class TaskStatsHookPercentileTest(unittest.IsolatedAsyncioTestCase):
    async def test_task_hook_records_times(self) -> None:
        hook = TaskStatsHook(name="test", interval=-1)
        for _ in range(10):
            async with hook.task_hook():  # pyre-ignore[16]
                pass
        self.assertEqual(hook.num_tasks, 10)
        self.assertEqual(hook.num_success, 10)
        self.assertEqual(len(hook._times), 10)
        for t in hook._times:
            self.assertGreaterEqual(t, 0.0)

    async def test_failed_task_not_tracked_in_times(self) -> None:
        hook = TaskStatsHook(name="test", interval=-1)

        async with hook.task_hook():  # pyre-ignore[16]
            pass

        try:
            async with hook.task_hook():  # pyre-ignore[16]
                raise ValueError("fail")
        except ValueError:
            pass

        async with hook.task_hook():  # pyre-ignore[16]
            pass

        self.assertEqual(hook.num_tasks, 3)
        self.assertEqual(hook.num_success, 2)
        self.assertEqual(len(hook._times), 2)

    async def test_stage_hook_produces_stats_with_percentiles(self) -> None:
        hook = TaskStatsHook(name="test", interval=-1)
        logged_stats: list[TaskPerfStats] = []
        original_log: Callable[[TaskPerfStats], None] = hook._log_stats

        def capture_stats(stats: TaskPerfStats) -> None:
            logged_stats.append(stats)
            original_log(stats)

        hook._log_stats = capture_stats  # pyre-ignore[8]

        async with hook.stage_hook():  # pyre-ignore[16]
            for _ in range(5):
                async with hook.task_hook():  # pyre-ignore[16]
                    pass

        self.assertEqual(len(logged_stats), 1)
        stats = logged_stats[0]
        self.assertEqual(stats.num_tasks, 5)
        self.assertEqual(stats.num_failures, 0)
        self.assertGreater(stats.ave_time, 0.0)
        self.assertGreaterEqual(stats.p90_time, stats.ave_time)
        self.assertGreaterEqual(stats.p99_time, stats.p90_time)

    async def test_lap_stats_with_percentiles(self) -> None:
        hook = TaskStatsHook(name="test", interval=-1)

        for _ in range(10):
            async with hook.task_hook():  # pyre-ignore[16]
                pass

        lap1 = hook._get_lap_stats()
        self.assertEqual(lap1.num_tasks, 10)
        self.assertGreater(lap1.p90_time, 0.0)
        self.assertGreater(lap1.p99_time, 0.0)

        for _ in range(5):
            async with hook.task_hook():  # pyre-ignore[16]
                pass

        lap2 = hook._get_lap_stats()
        self.assertEqual(lap2.num_tasks, 5)
        self.assertGreater(lap2.p90_time, 0.0)
        self.assertGreater(lap2.p99_time, 0.0)

    async def test_empty_lap_stats(self) -> None:
        hook = TaskStatsHook(name="test", interval=-1)
        lap = hook._get_lap_stats()
        self.assertEqual(lap.num_tasks, 0)
        self.assertEqual(lap.p90_time, 0.0)
        self.assertEqual(lap.p99_time, 0.0)


class StatsQueuePercentileTest(unittest.IsolatedAsyncioTestCase):
    async def test_put_get_records_times(self) -> None:
        queue = StatsQueue(name="test", buffer_size=10, interval=-1)
        async with queue.stage_hook():  # pyre-ignore[16]
            for i in range(5):
                await queue.put(i)
            for _ in range(5):
                await queue.get()

        self.assertEqual(len(queue._putc._times), 5)
        self.assertEqual(len(queue._getc._times), 5)
        self.assertGreaterEqual(queue._putc.p90_time, 0.0)
        self.assertGreaterEqual(queue._getc.p90_time, 0.0)

    async def test_stage_hook_produces_stats_with_percentiles(self) -> None:
        queue = StatsQueue(name="test", buffer_size=10, interval=-1)
        logged_stats: list[QueuePerfStats] = []
        original_log: Callable[[QueuePerfStats], None] = queue._log_stats

        def capture_stats(stats: QueuePerfStats) -> None:
            logged_stats.append(stats)
            original_log(stats)

        queue._log_stats = capture_stats  # pyre-ignore[8]

        async with queue.stage_hook():  # pyre-ignore[16]
            for i in range(5):
                await queue.put(i)
            for _ in range(5):
                await queue.get()

        self.assertEqual(len(logged_stats), 1)
        stats = logged_stats[0]
        self.assertEqual(stats.num_items, 5)
        self.assertGreaterEqual(stats.p90_put_time, 0.0)
        self.assertGreaterEqual(stats.p99_put_time, 0.0)
        self.assertGreaterEqual(stats.p90_get_time, 0.0)
        self.assertGreaterEqual(stats.p99_get_time, 0.0)

    async def test_lap_stats_with_percentiles(self) -> None:
        queue = StatsQueue(name="test", buffer_size=10, interval=-1)
        queue._lap_t0 = asyncio.get_running_loop().time()
        queue._empty_t0 = queue._lap_t0

        for i in range(8):
            await queue.put(i)
        for _ in range(8):
            await queue.get()

        lap1 = queue._get_lap_stats()
        self.assertEqual(lap1.num_items, 8)
        self.assertGreaterEqual(lap1.p90_put_time, 0.0)
        self.assertGreaterEqual(lap1.p90_get_time, 0.0)
        self.assertGreaterEqual(lap1.p99_put_time, 0.0)
        self.assertGreaterEqual(lap1.p99_get_time, 0.0)

        for i in range(3):
            await queue.put(i)
        for _ in range(3):
            await queue.get()

        lap2 = queue._get_lap_stats()
        self.assertEqual(lap2.num_items, 3)
        self.assertGreaterEqual(lap2.p90_put_time, 0.0)
        self.assertGreaterEqual(lap2.p90_get_time, 0.0)
