# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import unittest
from collections.abc import Callable

from spdl.pipeline._components._common import _P2Percentile, _StatsCounter, StageInfo
from spdl.pipeline._components._hook import TaskPerfStats, TaskStatsHook
from spdl.pipeline._components._queue import QueuePerfStats, StatsQueue


class P2PercentileTest(unittest.TestCase):
    def test_empty(self) -> None:
        """A fresh _P2Percentile with no observations reports 0.0."""
        p = _P2Percentile(90)
        self.assertEqual(p.value, 0.0)

    def test_single_element(self) -> None:
        """With only one observation, the percentile value equals that observation."""
        p = _P2Percentile(90)
        p.update(5.0)
        self.assertEqual(p.value, 5.0)

    def test_two_elements(self) -> None:
        """With two observations, p90 returns the larger value."""
        p = _P2Percentile(90)
        p.update(1.0)
        p.update(2.0)
        self.assertEqual(p.value, 2.0)

    def test_four_elements(self) -> None:
        """With fewer than 5 observations, the fallback sorted-lookup is used.

        Checks that p50 of [1, 2, 3, 4] returns the median (3.0).
        """
        p = _P2Percentile(50)
        for v in [1.0, 2.0, 3.0, 4.0]:
            p.update(v)
        self.assertEqual(p.value, 3.0)

    def test_five_elements_p90(self) -> None:
        """With exactly 5 observations the P² algorithm initializes.

        Checks that the p90 estimate falls within a reasonable range.
        """
        p = _P2Percentile(90)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            p.update(v)
        self.assertGreaterEqual(p.value, 2.0)
        self.assertLessEqual(p.value, 5.0)

    def test_hundred_elements_p90(self) -> None:
        """P² p90 estimate over 100 sequential values is close to the true p90 (90.0)."""
        p = _P2Percentile(90)
        for i in range(100):
            p.update(float(i))
        self.assertAlmostEqual(p.value, 90.0, delta=3.0)

    def test_hundred_elements_p99(self) -> None:
        """P² p99 estimate over 100 sequential values is close to the true p99 (99.0)."""
        p = _P2Percentile(99)
        for i in range(100):
            p.update(float(i))
        self.assertAlmostEqual(p.value, 99.0, delta=3.0)

    def test_hundred_elements_p50(self) -> None:
        """P² p50 (median) estimate over 100 sequential values is close to 50.0."""
        p = _P2Percentile(50)
        for i in range(100):
            p.update(float(i))
        self.assertAlmostEqual(p.value, 50.0, delta=3.0)

    def test_thousand_elements_accuracy(self) -> None:
        """With 1000 observations, p90 and p99 estimates stay within tight bounds.

        Checks p90 ≈ 900 (±20) and p99 ≈ 990 (±20).
        """
        p90 = _P2Percentile(90)
        p99 = _P2Percentile(99)
        for i in range(1000):
            v = float(i)
            p90.update(v)
            p99.update(v)
        self.assertAlmostEqual(p90.value, 900.0, delta=20.0)
        self.assertAlmostEqual(p99.value, 990.0, delta=20.0)

    def test_reset(self) -> None:
        """After reset(), the estimator returns to its initial state (value 0.0)."""
        p = _P2Percentile(90)
        for i in range(20):
            p.update(float(i))
        self.assertGreater(p.value, 0.0)
        p.reset()
        self.assertEqual(p.value, 0.0)

    def test_reset_and_reuse(self) -> None:
        """After reset(), feeding new data produces estimates based only on the new data.

        First feeds [0..99], resets, then feeds [100..199] and checks p50 ≈ 150.
        """
        p = _P2Percentile(50)
        for i in range(100):
            p.update(float(i))
        p.reset()
        for i in range(100, 200):
            p.update(float(i))
        self.assertAlmostEqual(p.value, 150.0, delta=5.0)

    def test_unsorted_input(self) -> None:
        """P² produces accurate estimates regardless of input order.

        Feeds 10 values in shuffled order and checks p90 ≈ 8.0 (±2).
        """
        p = _P2Percentile(90)
        values = [9.0, 1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 0.0]
        for v in values:
            p.update(v)
        self.assertAlmostEqual(p.value, 8.0, delta=2.0)

    def test_constant_values(self) -> None:
        """When all observations are identical, the percentile equals that constant."""
        p = _P2Percentile(90)
        for _ in range(20):
            p.update(42.0)
        self.assertAlmostEqual(p.value, 42.0, delta=0.01)


class StatsCounterPercentileTest(unittest.TestCase):
    def test_initial_state(self) -> None:
        """A fresh _StatsCounter has zero items, zero average, and zero percentiles."""
        counter = _StatsCounter()
        self.assertEqual(counter.p90_time, 0.0)
        self.assertEqual(counter.p99_time, 0.0)
        self.assertEqual(counter.num_items, 0)
        self.assertEqual(counter.ave_time, 0.0)

    def test_update_tracks_percentiles(self) -> None:
        """After 100 updates, the counter's p90 and p99 properties reflect accurate estimates."""
        counter = _StatsCounter()
        for i in range(100):
            counter.update(float(i))
        self.assertEqual(counter.num_items, 100)
        self.assertAlmostEqual(counter.p90_time, 90.0, delta=3.0)
        self.assertAlmostEqual(counter.p99_time, 99.0, delta=3.0)

    def test_count_context_manager(self) -> None:
        """The count() context manager records one item with non-negative percentiles."""
        counter = _StatsCounter()
        with counter.count():
            pass
        self.assertEqual(counter.num_items, 1)
        self.assertGreaterEqual(counter.p90_time, 0.0)
        self.assertGreaterEqual(counter.p99_time, 0.0)

    def test_consume_lap_percentiles(self) -> None:
        """consume_lap_percentiles() returns current lap p90/p99, then resets lap trackers.

        After consuming, a second call returns (0.0, 0.0).
        """
        counter = _StatsCounter()
        for i in range(100):
            counter.update(float(i))

        p90, p99 = counter.consume_lap_percentiles()
        self.assertAlmostEqual(p90, 90.0, delta=3.0)
        self.assertAlmostEqual(p99, 99.0, delta=3.0)

        p90_after, p99_after = counter.consume_lap_percentiles()
        self.assertEqual(p90_after, 0.0)
        self.assertEqual(p99_after, 0.0)

    def test_consume_lap_does_not_affect_overall(self) -> None:
        """Consuming lap percentiles does not change the overall (lifetime) p90 value."""
        counter = _StatsCounter()
        for i in range(100):
            counter.update(float(i))

        overall_p90_before = counter.p90_time
        counter.consume_lap_percentiles()
        self.assertEqual(counter.p90_time, overall_p90_before)


class TaskPerfStatsTest(unittest.TestCase):
    def test_fields_present(self) -> None:
        """TaskPerfStats dataclass stores all fields including p90_time and p99_time."""
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
        """QueuePerfStats dataclass stores p90/p99 fields for both put and get operations."""
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
        """The qps property computes num_items / elapsed correctly."""
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
        """When elapsed is zero, qps returns 0 to avoid division by zero."""
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
    async def test_task_hook_records_percentiles(self) -> None:
        """Successful tasks are tracked by the P² percentile estimators.

        Checks that after 10 successful tasks, num_tasks/num_success are correct
        and both p90/p99 trackers have non-negative values.
        """
        hook = TaskStatsHook(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="test"), interval=-1
        )
        for _ in range(10):
            async with hook.task_hook():  # pyre-ignore[16]
                pass
        self.assertEqual(hook.num_tasks, 10)
        self.assertEqual(hook.num_success, 10)
        self.assertGreaterEqual(hook._p90.value, 0.0)
        self.assertGreaterEqual(hook._p99.value, 0.0)

    async def test_failed_task_not_tracked_in_percentiles(self) -> None:
        """Failed tasks increment num_tasks but are excluded from percentile tracking.

        Runs 3 tasks (2 succeed, 1 fails). Checks that the P² tracker count is 2.
        """
        hook = TaskStatsHook(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="test"), interval=-1
        )

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
        self.assertEqual(hook._p90._count, 2)

    async def test_stage_hook_produces_stats_with_percentiles(self) -> None:
        """When stage_hook exits, _log_stats is called with a TaskPerfStats that
        includes non-negative p90_time and p99_time values.
        """
        hook = TaskStatsHook(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="test"), interval=-1
        )
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
        self.assertGreaterEqual(stats.p90_time, 0.0)
        self.assertGreaterEqual(stats.p99_time, 0.0)

    async def test_lap_stats_with_percentiles(self) -> None:
        """Lap stats report percentiles only for the current interval, then reset.

        First lap covers 10 tasks; second lap covers 5 new tasks. Each lap's
        p90/p99 should be positive and independent of the other.
        """
        hook = TaskStatsHook(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="test"), interval=-1
        )

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
        """When no tasks have run, lap stats report zero for all percentile fields."""
        hook = TaskStatsHook(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="test"), interval=-1
        )
        lap = hook._get_lap_stats()
        self.assertEqual(lap.num_tasks, 0)
        self.assertEqual(lap.p90_time, 0.0)
        self.assertEqual(lap.p99_time, 0.0)


class StatsQueuePercentileTest(unittest.IsolatedAsyncioTestCase):
    async def test_put_get_records_percentiles(self) -> None:
        """After put/get operations, the queue's internal counters have
        non-negative p90 percentile values for both put and get.
        """
        queue = StatsQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="test"),
            buffer_size=10,
            interval=-1,
        )
        async with queue.stage_hook():  # pyre-ignore[16]
            for i in range(5):
                await queue.put(i)
            for _ in range(5):
                await queue.get()

        self.assertGreaterEqual(queue._putc.p90_time, 0.0)
        self.assertGreaterEqual(queue._getc.p90_time, 0.0)

    async def test_stage_hook_produces_stats_with_percentiles(self) -> None:
        """When stage_hook exits, _log_stats receives a QueuePerfStats with
        non-negative p90/p99 values for both put and get operations.
        """
        queue = StatsQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="test"),
            buffer_size=10,
            interval=-1,
        )
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
        """Lap stats for the queue report per-interval p90/p99 for put and get,
        resetting between laps.

        First lap covers 8 items; second lap covers 3 new items. Each lap's
        percentile fields should be non-negative and independent.
        """
        queue = StatsQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="test"),
            buffer_size=10,
            interval=-1,
        )
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
