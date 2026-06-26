# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import time
import unittest

from spdl.pipeline import BackgroundTask, build_pipeline
from spdl.pipeline.config import (
    get_default_background_tasks,
    set_default_background_tasks,
)
from spdl.pipeline.defs import Pipe, PipelineConfig, SinkConfig, SourceConfig


def _simple_cfg() -> PipelineConfig[int]:
    return PipelineConfig(
        src=SourceConfig(range(5)),
        pipes=[Pipe(lambda x: x)],
        sink=SinkConfig(3),
    )


def _slow_cfg(delay: float = 0.5) -> PipelineConfig[int]:
    """Pipeline that takes a while to complete, giving background tasks time to run."""

    def slow_op(x: int) -> int:
        time.sleep(delay)
        return x

    return PipelineConfig(
        src=SourceConfig(range(3)),
        pipes=[Pipe(slow_op)],
        sink=SinkConfig(3),
    )


class _TrackingTask(BackgroundTask):
    """Background task that tracks whether it started and was cancelled."""

    def __init__(self) -> None:
        self.started = False
        self.cancelled = False

    async def run(self) -> None:
        self.started = True
        try:
            while True:
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class _CountingTask(BackgroundTask):
    """Background task that counts iterations."""

    def __init__(self) -> None:
        self.count = 0

    async def run(self) -> None:
        try:
            while True:
                self.count += 1
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass


class _FailingTask(BackgroundTask):
    async def run(self) -> None:
        raise RuntimeError("bg task error")


class _ShortTask(BackgroundTask):
    """Background task that completes quickly."""

    def __init__(self) -> None:
        self.completed = False

    async def run(self) -> None:
        await asyncio.sleep(0.01)
        self.completed = True


class BackgroundTaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = get_default_background_tasks()
        set_default_background_tasks(None)

    def tearDown(self) -> None:
        set_default_background_tasks(self._saved)

    def test_background_task_runs_and_is_cancelled(self) -> None:
        """Background tasks run alongside pipeline and get cancelled on completion."""
        task_instance: _TrackingTask = _TrackingTask()

        def factory() -> BackgroundTask:
            return task_instance

        pipeline = build_pipeline(
            _simple_cfg(), num_threads=1, background_tasks=[factory]
        )

        with pipeline.auto_stop():
            items = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(sorted(items), [0, 1, 2, 3, 4])
        self.assertTrue(task_instance.started, "Background task should have started")
        self.assertTrue(
            task_instance.cancelled, "Background task should have been cancelled"
        )

    def test_background_task_error_does_not_crash_pipeline(self) -> None:
        """Background task errors are logged but don't fail the pipeline."""
        pipeline = build_pipeline(
            _simple_cfg(), num_threads=1, background_tasks=[_FailingTask]
        )

        with pipeline.auto_stop():
            items = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(sorted(items), [0, 1, 2, 3, 4])

    def test_multiple_background_tasks(self) -> None:
        """Multiple background tasks can run concurrently."""
        task_0 = _CountingTask()
        task_1 = _CountingTask()

        pipeline = build_pipeline(
            _simple_cfg(),
            num_threads=1,
            background_tasks=[lambda: task_0, lambda: task_1],
        )

        with pipeline.auto_stop():
            items = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(sorted(items), [0, 1, 2, 3, 4])
        self.assertGreater(task_0.count, 0, "First background task should have run")
        self.assertGreater(task_1.count, 0, "Second background task should have run")

    def test_no_background_tasks(self) -> None:
        """Pipeline works normally when no background tasks are provided."""
        pipeline = build_pipeline(_simple_cfg(), num_threads=1)

        with pipeline.auto_stop():
            items = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(sorted(items), [0, 1, 2, 3, 4])

    def test_empty_background_tasks_list(self) -> None:
        """Pipeline works normally with an empty background tasks list."""
        pipeline = build_pipeline(_simple_cfg(), num_threads=1, background_tasks=[])

        with pipeline.auto_stop():
            items = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(sorted(items), [0, 1, 2, 3, 4])

    def test_background_task_completes_before_pipeline(self) -> None:
        """A background task that finishes early doesn't affect the pipeline."""
        task_instance = _ShortTask()

        pipeline = build_pipeline(
            _slow_cfg(0.1),
            num_threads=1,
            background_tasks=[lambda: task_instance],
        )

        with pipeline.auto_stop():
            items = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(sorted(items), [0, 1, 2])
        self.assertTrue(
            task_instance.completed, "Background task should have completed"
        )

    def test_background_task_mixed_success_and_failure(self) -> None:
        """One failing and one succeeding background task — pipeline still works."""
        good_task = _TrackingTask()

        pipeline = build_pipeline(
            _simple_cfg(),
            num_threads=1,
            background_tasks=[lambda: good_task, _FailingTask],
        )

        with pipeline.auto_stop():
            items = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(sorted(items), [0, 1, 2, 3, 4])
        self.assertTrue(good_task.started, "Good background task should have run")

    def test_class_as_factory(self) -> None:
        """A BackgroundTask class itself can be used as a factory."""

        class MyTask(BackgroundTask):
            ran = False

            async def run(self) -> None:
                MyTask.ran = True
                try:
                    while True:
                        await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    pass

        MyTask.ran = False
        pipeline = build_pipeline(
            _simple_cfg(), num_threads=1, background_tasks=[MyTask]
        )

        with pipeline.auto_stop():
            items = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(sorted(items), [0, 1, 2, 3, 4])
        self.assertTrue(MyTask.ran, "Task class used as factory should have run")


class DefaultBackgroundTasksTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = get_default_background_tasks()
        set_default_background_tasks(None)

    def tearDown(self) -> None:
        set_default_background_tasks(self._saved)

    def test_get_set_default_background_tasks(self) -> None:
        """set/get_default_background_tasks round-trips correctly."""
        self.assertIsNone(get_default_background_tasks())

        set_default_background_tasks([_TrackingTask])
        result = get_default_background_tasks()
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)  # pyre-ignore[6]
        self.assertIs(result[0], _TrackingTask)  # pyre-ignore[16]

        set_default_background_tasks(None)
        self.assertIsNone(get_default_background_tasks())

    def test_default_background_tasks_run_automatically(self) -> None:
        """Default background tasks are started without explicit parameter."""
        task_instance = _TrackingTask()
        set_default_background_tasks([lambda: task_instance])

        pipeline = build_pipeline(_simple_cfg(), num_threads=1)

        with pipeline.auto_stop():
            items = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(sorted(items), [0, 1, 2, 3, 4])
        self.assertTrue(
            task_instance.started, "Default background task should have run"
        )

    def test_default_and_per_pipeline_tasks_merged(self) -> None:
        """Both default and per-pipeline background tasks run."""
        default_task = _TrackingTask()
        custom_task = _TrackingTask()

        set_default_background_tasks([lambda: default_task])

        pipeline = build_pipeline(
            _simple_cfg(), num_threads=1, background_tasks=[lambda: custom_task]
        )

        with pipeline.auto_stop():
            items = list(pipeline.get_iterator(timeout=30))

        self.assertEqual(sorted(items), [0, 1, 2, 3, 4])
        self.assertTrue(default_task.started, "Default background task should have run")
        self.assertTrue(custom_task.started, "Custom background task should have run")
