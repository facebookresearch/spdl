# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import os
import queue
import subprocess
import sys
import unittest
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError
from typing import Any, cast
from unittest.mock import patch

from spdl.pipeline import (
    AsyncQueue,
    build_pipeline,
    PipelineBuilder,
    QueuePerfStats,
    StatsThreadOutputQueue,
    ThreadOutputQueue,
)
from spdl.pipeline._components._common import StageInfo
from spdl.pipeline._components._thread_queue import _make_thread_output_queue
from spdl.pipeline.config import (
    get_default_thread_output_queue_config,
    set_default_thread_output_queue_config,
    ThreadOutputQueueConfig,
)


class _CollectedSink(StatsThreadOutputQueue):
    def __init__(self, info: StageInfo, *, buffer_size: int = 1, interval: float = -1):
        super().__init__(info, buffer_size=buffer_size, interval=interval)
        self.packets: list[QueuePerfStats] = []
        self.final: list[QueuePerfStats] = []

    async def interval_stats_callback(self, stats: QueuePerfStats) -> None:
        self.packets.append(stats)

    async def final_stats_callback(self, stats: QueuePerfStats) -> None:
        self.final.append(stats)


class ThreadOutputOverrideTest(unittest.TestCase):
    def setUp(self) -> None:
        defaults = get_default_thread_output_queue_config()
        self.addCleanup(set_default_thread_output_queue_config, defaults)
        set_default_thread_output_queue_config(
            ThreadOutputQueueConfig(enabled=False, queue_class=ThreadOutputQueue)
        )
        for path, value in [
            ("spdl.pipeline._build._DEFAULT_BUILD_CALLBACK", None),
            ("spdl.pipeline._bg_task._DEFAULT_BACKGROUND_TASKS", []),
        ]:
            guard = patch(path, value)
            guard.start()
            self.addCleanup(guard.stop)

    def test_initial_config_preserves_sink_statistics(self) -> None:
        """The new OSS initial defaults enable the public statistics thread sink."""
        defaults = ThreadOutputQueueConfig()
        self.assertTrue(defaults.enabled)
        self.assertIs(defaults.queue_class, StatsThreadOutputQueue)

    def test_true_default_is_plain_thread_queue(self) -> None:
        """Explicit True uses the configured class even when the enabled default is False."""
        info = StageInfo(pipeline_id=0, stage_id="0", stage_name="sink")
        output = _make_thread_output_queue(info, 2, True, -1)
        self.assertIs(type(output), ThreadOutputQueue)
        self.assertIs(type(output._queue), queue.Queue)
        self.assertFalse(hasattr(output, "_stats_provider"))

    def test_explicit_class_overrides_configured_default(self) -> None:
        """An explicit class wins over the immutable configured default."""
        info = StageInfo(pipeline_id=0, stage_id="0", stage_name="sink")
        defaults = ThreadOutputQueueConfig(enabled=True, queue_class=_CollectedSink)
        set_default_thread_output_queue_config(defaults)
        self.assertIs(get_default_thread_output_queue_config(), defaults)
        selected = _make_thread_output_queue(info, 3, True, 5)
        explicit = _make_thread_output_queue(info, 4, ThreadOutputQueue, 5)
        self.assertIsInstance(selected, _CollectedSink)
        self.assertEqual(selected.interval, 5)
        self.assertEqual(selected._queue.maxsize, 3)
        self.assertIs(type(explicit), ThreadOutputQueue)
        self.assertEqual(explicit._queue.maxsize, 4)

    def test_config_is_immutable_and_validated(self) -> None:
        """Invalid configurations cannot change defaults, and valid records are immutable."""
        defaults = get_default_thread_output_queue_config()
        with self.assertRaises(FrozenInstanceError):
            cast(Any, defaults).enabled = True
        for invalid in [
            object(),
            ThreadOutputQueueConfig(enabled=cast(Any, 1)),
            ThreadOutputQueueConfig(queue_class=cast(Any, AsyncQueue)),
        ]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    set_default_thread_output_queue_config(cast(Any, invalid))
                self.assertEqual(get_default_thread_output_queue_config(), defaults)

    def test_invalid_class_is_rejected(self) -> None:
        """Only thread-output subclasses are accepted as a class selection."""
        for value in [AsyncQueue, object, 1, "thread"]:
            with self.assertRaises(TypeError):
                PipelineBuilder().add_source([1]).add_sink(1).build(
                    num_threads=2,
                    task_hook_factory=lambda _: [],
                    use_thread_output_queue=cast(Any, value),
                )

    def test_false_keeps_async_sink(self) -> None:
        """Explicit False selects the asyncio sink even with Meta defaults enabled."""
        with (
            patch("spdl.pipeline._build._DEFAULT_BUILD_CALLBACK", None),
            patch("spdl.pipeline._bg_task._DEFAULT_BACKGROUND_TASKS", []),
        ):
            pipeline = (
                PipelineBuilder()
                .add_source([1])
                .add_sink(1)
                .build(
                    num_threads=2,
                    queue_class=AsyncQueue,
                    task_hook_factory=lambda _: [],
                    use_thread_output_queue=False,
                )
            )
            self.assertNotIsInstance(pipeline._impl._output_queue, ThreadOutputQueue)
            with pipeline.auto_stop():
                self.assertEqual(list(pipeline.get_iterator(timeout=3)), [1])

    def test_pipeline_without_meta_package(self) -> None:
        """OSS users can select plain/custom stats queues without any Meta extension."""
        script = """
import importlib.abc, sys, threading
class NoMeta(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "spdl.pipeline.fb":
            raise ModuleNotFoundError("Meta extension excluded", name=fullname)
sys.meta_path.insert(0, NoMeta())
from spdl.pipeline import PipelineBuilder, ThreadOutputQueue, StatsThreadOutputQueue, AsyncQueue
from spdl.pipeline.config import get_default_thread_output_queue_config
assert get_default_thread_output_queue_config().enabled is True
assert get_default_thread_output_queue_config().queue_class is StatsThreadOutputQueue
class MyStats(StatsThreadOutputQueue):
    final = []
    async def interval_stats_callback(self, stats):
        pass
    async def final_stats_callback(self, stats):
        self.final.append(stats)
for selection in [None, True, ThreadOutputQueue, MyStats, False]:
    consumed = threading.Event()
    async def source():
        for i in range(20): yield i
        assert await __import__("asyncio").to_thread(consumed.wait, 3)
    pipeline = (PipelineBuilder().add_source(source()).add_sink(8)
        .build(num_threads=2, queue_class=AsyncQueue, task_hook_factory=lambda _: [],
               use_thread_output_queue=selection))
    with pipeline.auto_stop():
        it = pipeline.get_iterator(timeout=3)
        try: assert [next(it) for _ in range(20)] == list(range(20))
        finally: consumed.set()
        assert list(it) == []
    if selection is MyStats: assert pipeline._impl._output_queue.final[0].num_items == 20
    if selection is None or selection is True:
        sink = pipeline._impl._output_queue
        assert type(sink) is StatsThreadOutputQueue
        assert sink._stats_provider.get_stats().num_items == 20
        assert sink._stats_provider.put_counter.num_items == 20
assert not any(name.startswith("spdl.pipeline.fb") for name in sys.modules)
print("OSS_CUSTOM_THREAD_QUEUE_PASSED")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OSS_CUSTOM_THREAD_QUEUE_PASSED", result.stdout)


class CustomSinkHooksTest(unittest.IsolatedAsyncioTestCase):
    async def test_plain_queue_capacity_and_task_accounting(self) -> None:
        """The public base forwards standard queue operations and join accounting."""
        output = ThreadOutputQueue(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="sink"), buffer_size=1
        )
        self.assertEqual(output.maxsize, 1)
        output.put_nowait(1)
        self.assertTrue(output.full())
        with self.assertRaises(queue.Full):
            output.put_nowait(2)
        self.assertEqual(output.get_nowait(), 1)
        self.assertFalse(output.full())
        output.task_done()
        await asyncio.wait_for(output.join(), 2)
        self.assertTrue(output.empty())

    async def test_custom_callbacks_need_no_logging(self) -> None:
        """Application hooks receive packets in memory without calling Python logging."""
        output = _CollectedSink(
            StageInfo(pipeline_id=0, stage_id="0", stage_name="sink"), interval=-1
        )
        with patch.object(
            output._stats_logger, "_log_stats", side_effect=AssertionError("logging")
        ):
            async with output.stage_hook():
                await output.put(1)
                self.assertEqual(output.get_nowait(), 1)
        self.assertEqual(len(output.final), 1)
        self.assertEqual(output.final[0].num_items, 1)

    async def test_constructor_receives_buffer_and_interval(self) -> None:
        """Direct config building forwards sink capacity and reporting interval."""
        config = PipelineBuilder().add_source([1]).add_sink(4).get_config()
        with (
            patch("spdl.pipeline._build._DEFAULT_BUILD_CALLBACK", None),
            patch("spdl.pipeline._bg_task._DEFAULT_BACKGROUND_TASKS", []),
        ):
            pipeline = build_pipeline(
                config,
                num_threads=2,
                queue_class=AsyncQueue,
                task_hook_factory=lambda _: [],
                report_stats_interval=3,
                use_thread_output_queue=_CollectedSink,
            )
            output = pipeline._impl._output_queue
            self.assertIsInstance(output, _CollectedSink)
            self.assertEqual(output.interval, 3)
            self.assertEqual(output._queue.maxsize, 4)
            with pipeline.auto_stop():
                self.assertEqual(list(pipeline.get_iterator(timeout=3)), [1])

    async def test_plain_custom_stage_hook_is_honored(self) -> None:
        """A caller-provided ordinary thread queue controls its stage hook."""
        events = []

        class MySink(ThreadOutputQueue):
            @asynccontextmanager
            async def stage_hook(self):
                events.append("start")
                try:
                    yield
                finally:
                    events.append("end")

        with (
            patch("spdl.pipeline._build._DEFAULT_BUILD_CALLBACK", None),
            patch("spdl.pipeline._bg_task._DEFAULT_BACKGROUND_TASKS", []),
        ):
            pipeline = (
                PipelineBuilder()
                .add_source([1, 2])
                .add_sink(2)
                .build(
                    num_threads=2,
                    queue_class=AsyncQueue,
                    task_hook_factory=lambda _: [],
                    use_thread_output_queue=MySink,
                )
            )
            with pipeline.auto_stop():
                self.assertEqual(list(pipeline.get_iterator(timeout=3)), [1, 2])
        self.assertEqual(events, ["start", "end"])
