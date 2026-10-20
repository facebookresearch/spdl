# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import threading
import unittest
from collections.abc import AsyncIterator

from spdl.pipeline import AsyncQueue, PipelineBuilder, PipelineFailure
from spdl.pipeline._components._common import StageInfo
from spdl.pipeline._components._pipe import _get_fail_counter, _stop_on_failure


class _FailingSourceQueue(AsyncQueue):
    async def get(self) -> object:
        item = await super().get()
        if self.info.stage_name == "src" and item == 1:
            raise RuntimeError("ordered input queue failure")
        return item


class FailureWakeupCancellationTest(unittest.IsolatedAsyncioTestCase):
    async def test_external_cancellation_survives_failure_during_cleanup(self) -> None:
        """A failure during teardown cannot replace the external stop request."""
        counter = _get_fail_counter()(max_stage_failures=0)
        info = StageInfo(pipeline_id=0, stage_id="0", stage_name="stage")
        ready = asyncio.Event()

        async def stage() -> None:
            try:
                ready.set()
                await asyncio.Event().wait()
            finally:
                with self.assertRaises(ValueError):
                    async with counter.task_hook():
                        raise ValueError("failure during teardown")

        task = asyncio.create_task(_stop_on_failure(stage(), counter, info))
        await ready.wait()
        task.cancel("external stop")
        with self.assertRaises(asyncio.CancelledError) as raised:
            await task
        self.assertEqual(raised.exception.args, ("external stop",))
        self.assertTrue(counter.too_many_failures())
        self.assertIsNone(counter._failure_handler)

    async def test_failure_wakeup_reports_stage_error(self) -> None:
        """The failure counter's own cancellation still reports its threshold."""
        counter = _get_fail_counter()(max_stage_failures=0)
        info = StageInfo(pipeline_id=0, stage_id="0", stage_name="stage")
        ready = asyncio.Event()

        async def stage() -> None:
            ready.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(_stop_on_failure(stage(), counter, info))
        await ready.wait()
        with self.assertRaises(ValueError):
            async with counter.task_hook():
                raise ValueError("operation failed")
        with self.assertRaisesRegex(RuntimeError, "exceeds the threshold"):
            await task
        self.assertFalse(task.cancelled())
        self.assertIsNone(counter._failure_handler)

    async def test_completed_stage_error_precedes_failure_wakeup(self) -> None:
        """A completed stage preserves its error when it also trips the threshold."""
        counter = _get_fail_counter()(max_stage_failures=0)
        info = StageInfo(pipeline_id=0, stage_id="0", stage_name="stage")

        async def stage() -> None:
            async with counter.task_hook():
                raise ValueError("stage failed")

        with self.assertRaisesRegex(ValueError, "stage failed"):
            await _stop_on_failure(stage(), counter, info)
        self.assertTrue(counter.too_many_failures())
        self.assertIsNone(counter._failure_handler)

    async def test_external_cancellation_supersedes_pending_failure_wakeup(
        self,
    ) -> None:
        """External cancellation remains authoritative over a pending wakeup."""
        counter = _get_fail_counter()(max_stage_failures=0)
        info = StageInfo(pipeline_id=0, stage_id="0", stage_name="stage")
        ready = asyncio.Event()
        wake = asyncio.Event()

        async def stage() -> None:
            ready.set()
            await wake.wait()

        task = asyncio.create_task(_stop_on_failure(stage(), counter, info))
        await ready.wait()
        wake.set()
        with self.assertRaises(ValueError):
            async with counter.task_hook():
                raise ValueError("operation failed")
        task.cancel("external stop")
        with self.assertRaises(asyncio.CancelledError) as raised:
            await task
        self.assertEqual(raised.exception.args, ("external stop",))
        self.assertTrue(task.cancelled())
        self.assertIsNone(counter._failure_handler)

    async def test_external_cancellation_precedes_failure_wakeup(self) -> None:
        """A pending external cancellation is not converted into a stage failure."""
        counter = _get_fail_counter()(max_stage_failures=0)
        info = StageInfo(pipeline_id=0, stage_id="0", stage_name="stage")
        ready = asyncio.Event()

        async def stage() -> None:
            ready.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(_stop_on_failure(stage(), counter, info))
        await ready.wait()
        task.cancel("external stop")
        with self.assertRaises(ValueError):
            async with counter.task_hook():
                raise ValueError("operation failed")
        with self.assertRaises(asyncio.CancelledError) as raised:
            await task
        self.assertEqual(raised.exception.args, ("external stop",))
        self.assertTrue(task.cancelled())
        self.assertIsNone(counter._failure_handler)


class IdleInputFailureTest(unittest.TestCase):
    def test_ordered_pipe_propagates_input_queue_failure(self) -> None:
        """An ordered stage reports its input queue failure through PipelineFailure."""
        pipeline = (
            PipelineBuilder()
            .add_source(range(2))
            .pipe(lambda item: item, concurrency=2, output_order="input")
            .add_sink(2)
            .build(num_threads=1, queue_class=_FailingSourceQueue)
        )

        with self.assertRaises(PipelineFailure) as raised:
            with pipeline.auto_stop():
                list(pipeline.get_iterator(timeout=10))

        self.assertTrue(
            any(
                isinstance(error, RuntimeError)
                and str(error) == "ordered input queue failure"
                for error in raised.exception.exceptions
            )
        )

    def _check_pending_cleanup(self, order: str) -> None:
        all_started = asyncio.Event()
        started: set[int] = set()
        cleaned = {item: threading.Event() for item in (1, 2, 3)}

        async def source() -> AsyncIterator[int]:
            for item in range(4):
                yield item
            await asyncio.Event().wait()

        async def process(item: int) -> int:
            if item == 0:
                await all_started.wait()
                raise ValueError("failure with other operations pending")
            started.add(item)
            if len(started) == 3:
                all_started.set()
            try:
                await asyncio.Event().wait()
                return item
            finally:
                cleaned[item].set()

        pipeline = (
            PipelineBuilder()
            .add_source(source())
            .pipe(process, concurrency=4, output_order=order, max_failures=0)
            .add_sink(1)
            .build(num_threads=1)
        )
        pipeline.start()
        try:
            with self.assertRaises(PipelineFailure):
                pipeline.get_item(timeout=10)
            self.assertTrue(all(event.is_set() for event in cleaned.values()))
        finally:
            try:
                pipeline.stop(timeout=10)
            except PipelineFailure:
                pass

    def _check_idle_failure(self, order: str) -> None:
        source_waiting = asyncio.Event()

        async def source() -> AsyncIterator[int]:
            yield 0
            source_waiting.set()
            await asyncio.Event().wait()

        async def fail(item: int) -> int:
            await source_waiting.wait()
            raise ValueError("failure while source waits")

        pipeline = (
            PipelineBuilder()
            .add_source(source())
            .pipe(fail, concurrency=4, output_order=order, max_failures=0)
            .add_sink(1)
            .build(num_threads=1)
        )
        pipeline.start()
        try:
            with self.assertRaises(PipelineFailure):
                pipeline.get_item(timeout=10)
        finally:
            try:
                pipeline.stop(timeout=10)
            except PipelineFailure:
                pass  # Teardown must not hide a reader timeout.

    def test_failure_wakes_stage_waiting_for_input(self) -> None:
        """An in-flight failure stops a stage even while its source cannot yield more data."""
        for order in ("completion", "input"):
            with self.subTest(output_order=order):
                self._check_idle_failure(order)

    def test_failure_cancels_other_inflight_operations(self) -> None:
        """Fatal failures release pending operations for completion and input order."""
        for order in ("completion", "input"):
            with self.subTest(output_order=order):
                self._check_pending_cleanup(order)
