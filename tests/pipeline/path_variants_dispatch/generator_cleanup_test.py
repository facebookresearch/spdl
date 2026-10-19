# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import threading
import unittest
from collections.abc import AsyncGenerator, AsyncIterator, Iterator

from spdl.pipeline._common._convert import _await_executor_result, _to_async_gen
from spdl.pipeline._components._common import _EOF, StageInfo
from spdl.pipeline._components._pipe import _get_fail_counter, _pipe, _wrap_agen
from spdl.pipeline._components._queue import AsyncQueue
from spdl.pipeline.defs import Pipe


class GeneratorCleanupTest(unittest.IsolatedAsyncioTestCase):
    async def test_completed_operation_failure_remains_tolerated(self) -> None:
        """Cleanup preserves the failure policy for already completed operations."""

        async def fail(_: None) -> None:
            raise ValueError("tolerated operation failure")

        info = StageInfo(0, "tolerated", "tolerated")
        inputs, output = AsyncQueue(info, buffer_size=0), AsyncQueue(info)
        inputs.put_nowait(None)
        inputs.put_nowait(_EOF)
        with self.assertLogs("spdl.pipeline._common._misc", level="ERROR"):
            await asyncio.wait_for(
                _pipe(
                    info,
                    inputs,
                    output,
                    Pipe(fail, concurrency=4)._args,
                    _get_fail_counter()(),
                    [],
                    False,
                ),
                2,
            )
        self.assertIs(output.get_nowait(), _EOF)

    async def test_cancelled_stage_preserves_cancellation_over_cleanup_failure(
        self,
    ) -> None:
        """Generator cleanup failures do not replace the stage cancellation."""
        blocked = asyncio.Event()
        closed = asyncio.Event()

        async def source(_: None) -> AsyncIterator[int]:
            try:
                yield 0
                blocked.set()
                await asyncio.Event().wait()
            finally:
                closed.set()
                raise ValueError("generator teardown failed")

        info = StageInfo(0, "generator", "generator")
        inputs, output = AsyncQueue(info), AsyncQueue(info)
        inputs.put_nowait(None)
        task = asyncio.create_task(
            _pipe(
                info,
                inputs,
                output,
                Pipe(source)._args,
                _get_fail_counter()(),
                [],
                False,
            )
        )
        self.assertEqual(await asyncio.wait_for(output.get(), 2), 0)
        await asyncio.wait_for(blocked.wait(), 2)
        with self.assertLogs("spdl.pipeline._common._misc", level="ERROR"):
            task.cancel("external stop")
            with self.assertRaises(asyncio.CancelledError) as caught:
                await asyncio.wait_for(task, 2)
        self.assertEqual(caught.exception.args, ("external stop",))
        self.assertTrue(closed.is_set())

    async def test_cancelled_thread_failure_retains_cancellation_cause(self) -> None:
        """A thread failure records cancellation delivered before it completed."""
        entered, release = threading.Event(), threading.Event()

        def source(_: None) -> Iterator[int]:
            yield 0
            entered.set()
            if not release.wait(2):
                raise TimeoutError("Thread failure fixture was never released.")
            raise ValueError("next failed during cancellation")

        gen = _to_async_gen(source, None)(None).__aiter__()
        self.assertEqual(await anext(gen), 0)
        task = asyncio.ensure_future(anext(gen))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            task.cancel("external stop")
            await asyncio.sleep(0)
        finally:
            release.set()
        with self.assertRaisesRegex(
            ValueError, "next failed during cancellation"
        ) as caught:
            await asyncio.wait_for(task, 2)
        cause = caught.exception.__cause__
        self.assertIsInstance(cause, asyncio.CancelledError)
        assert isinstance(cause, asyncio.CancelledError)
        self.assertEqual(cause.args, ("external stop",))

    async def test_cancelled_queued_operation_retains_cancellation_cause(self) -> None:
        """Executor cancellation retains cancellation previously sent to the task."""
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        async def get_cancellation_cause() -> BaseException | None:
            try:
                await _await_executor_result(future)
            except asyncio.CancelledError as error:
                return error.__cause__
            self.fail("Executor operation completed without cancellation.")

        task = asyncio.create_task(get_cancellation_cause())
        await asyncio.sleep(0)
        task.cancel("external stop")
        await asyncio.sleep(0)
        self.assertTrue(future.cancel())
        cause = await asyncio.wait_for(task, 2)
        self.assertIsInstance(cause, asyncio.CancelledError)
        assert isinstance(cause, asyncio.CancelledError)
        self.assertEqual(cause.args, ("external stop",))

    async def test_cancelled_stage_closes_active_generator(self) -> None:
        """Stopping a stage cancels its blocked per-item generator and runs cleanup."""
        loop = asyncio.get_running_loop()
        blocked, closed = asyncio.Event(), threading.Event()

        def source(_: None) -> Iterator[int]:
            try:
                for i in range(10):
                    if i == 2:
                        loop.call_soon_threadsafe(blocked.set)
                    yield i
            finally:
                closed.set()

        info = StageInfo(0, "generator", "generator")
        inputs, output = AsyncQueue(info), AsyncQueue(info)
        inputs.put_nowait(None)
        task = asyncio.create_task(
            _pipe(
                info,
                inputs,
                output,
                Pipe(source)._args,
                _get_fail_counter()(),
                [],
                False,
            )
        )
        self.assertEqual(await asyncio.wait_for(output.get(), 2), 0)
        await asyncio.wait_for(blocked.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        self.assertTrue(closed.is_set())

    async def test_cancelled_output_runs_generator_cleanup(self) -> None:
        """Cancellation while output is blocked closes the underlying generator."""
        loop = asyncio.get_running_loop()
        blocked = asyncio.Event()
        closed = threading.Event()

        def source(_: None) -> Iterator[int]:
            try:
                for i in range(10):
                    if i == 2:
                        loop.call_soon_threadsafe(blocked.set)
                    yield i
            finally:
                closed.set()

        output = AsyncQueue(StageInfo(0, "sink", "sink"))
        gen = _to_async_gen(source, None)(None).__aiter__()
        task = asyncio.create_task(_wrap_agen(gen, [], output))
        self.assertEqual(await asyncio.wait_for(output.get(), 2), 0)
        await asyncio.wait_for(blocked.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        self.assertTrue(closed.is_set())

    async def test_cancelled_next_finishes_before_close(self) -> None:
        """Cancellation cannot race generator cleanup with an executing next call."""
        entered, release, closed = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def source(_: None) -> Iterator[int]:
            try:
                yield 0
                entered.set()
                if not release.wait(2):
                    raise TimeoutError(
                        "Generator fixture next call was never released."
                    )
                yield 1
            finally:
                closed.set()

        gen = _to_async_gen(source, None)(None).__aiter__()
        self.assertEqual(await anext(gen), 0)
        task = asyncio.ensure_future(anext(gen))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        self.assertTrue(closed.is_set())

    async def test_cancelled_next_preserves_cancellation_over_close_failure(
        self,
    ) -> None:
        """Generator close failures do not replace an in-flight cancellation."""
        entered, release, closed = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def source(_: None) -> Iterator[int]:
            try:
                yield 0
                entered.set()
                if not release.wait(2):
                    raise TimeoutError(
                        "Generator fixture next call was never released."
                    )
                yield 1
            finally:
                closed.set()
                raise ValueError("generator teardown failed")

        gen = _to_async_gen(source, None)(None).__aiter__()
        self.assertEqual(await anext(gen), 0)
        task = asyncio.ensure_future(anext(gen))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            task.cancel("external stop")
            await asyncio.sleep(0)
            self.assertFalse(task.done())
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(task, 2)
        self.assertEqual(caught.exception.args, ("external stop",))
        self.assertTrue(closed.is_set())

    async def test_explicit_close_propagates_generator_cleanup_failure(self) -> None:
        """Explicit async close surfaces an underlying generator cleanup failure."""

        def source(_: None) -> Iterator[int]:
            try:
                yield 0
            finally:
                raise ValueError("generator teardown failed")

        gen = _to_async_gen(source, None)(None).__aiter__()
        self.assertIsInstance(gen, AsyncGenerator)
        assert isinstance(gen, AsyncGenerator)
        self.assertEqual(await anext(gen), 0)
        with self.assertRaisesRegex(ValueError, "generator teardown failed"):
            await gen.aclose()

    async def test_natural_exhaustion_preserves_outputs_and_closes(self) -> None:
        """Normal generator exhaustion preserves all outputs and runs cleanup."""
        closed = threading.Event()

        def source(_: None) -> Iterator[int]:
            try:
                yield from range(20)
            finally:
                closed.set()

        outputs = [item async for item in _to_async_gen(source, None)(None)]
        self.assertEqual(outputs, list(range(20)))
        self.assertTrue(closed.is_set())
