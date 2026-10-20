# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import threading
import unittest
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import asynccontextmanager

from spdl.pipeline import AsyncQueue, PipelineBuilder, PipelineFailure


class _AsyncSource:
    def __init__(self, size: int, *, fail: bool = False) -> None:
        self.size = size
        self.fail = fail
        self.started = threading.Event()
        self.iterators: list[AsyncIterator[int]] = []
        self.iterators_closed = 0

    def __aiter__(self) -> AsyncIterator[int]:
        iterator = self._iterate()
        self.iterators.append(iterator)
        return iterator

    async def _iterate(self) -> AsyncGenerator[int, None]:
        try:
            for value in range(self.size):
                self.started.set()
                yield value
            if self.fail:
                raise RuntimeError("source failure")
        finally:
            self.iterators_closed += 1


class _OwnedAsyncSource(_AsyncSource):
    def __init__(
        self, size: int, *, fail: bool = False, close_fail: bool = False
    ) -> None:
        super().__init__(size, fail=fail)
        self.close_fail = close_fail
        self.owners_closed = 0

    async def aclose(self) -> None:
        self.owners_closed += 1
        if self.close_fail:
            raise RuntimeError("close failure")


class _AsyncIterator:
    def __init__(self, *, fail: bool = False, close_fail: bool = False) -> None:
        self.value = self.closed = 0
        self.fail = fail
        self.close_fail = close_fail

    def __aiter__(self) -> "_AsyncIterator":
        return self

    async def __anext__(self) -> int:
        if self.value == 1 and self.fail:
            raise RuntimeError("source failure")
        if self.value == 2:
            raise StopAsyncIteration
        self.value += 1
        return self.value

    async def aclose(self) -> None:
        self.closed += 1
        if self.close_fail:
            raise RuntimeError("close failure")


class _SuspendingAsyncIterator(_AsyncIterator):
    def __init__(
        self,
        *,
        wait_for_cancel: bool = False,
        fail: bool = False,
        close_fail: bool = False,
    ) -> None:
        super().__init__(fail=fail, close_fail=close_fail)
        self.wait_for_cancel = wait_for_cancel
        self.stage_task: asyncio.Task[object] | None = None
        self.iteration_error: BaseException | None = None
        self.stage_error: BaseException | None = None
        self.close_started = threading.Event()
        self.close_finished = threading.Event()
        self.release_close = asyncio.Event()

    async def __anext__(self) -> int:
        if self.stage_task is None:
            self.stage_task = asyncio.current_task()
            if self.stage_task is None:
                raise RuntimeError("source iteration requires an asyncio task")
        try:
            if self.value == 1 and self.wait_for_cancel:
                await asyncio.Event().wait()
            return await super().__anext__()
        except BaseException as error:
            self.iteration_error = error
            raise

    async def aclose(self) -> None:
        self.close_started.set()
        await self.release_close.wait()
        try:
            await super().aclose()
        finally:
            self.close_finished.set()


def _source_error_queue(source: _SuspendingAsyncIterator) -> type[AsyncQueue]:
    class SourceErrorQueue(AsyncQueue):
        @asynccontextmanager
        async def stage_hook(self) -> AsyncIterator[None]:
            try:
                yield
            except BaseException as error:
                if self.info.stage_name == "src":
                    source.stage_error = error
                raise

    return SourceErrorQueue


class _ReusableAsyncIterator(_AsyncIterator):
    def __aiter__(self) -> "_ReusableAsyncIterator":
        if self.closed:
            raise RuntimeError("iterator is closed")
        self.value = 0
        return self


class _OwnedBufferSource:
    def __init__(self) -> None:
        self.buffer = bytearray([7])
        self.owners_closed = 0
        self.stage_done = threading.Event()

    def __aiter__(self) -> AsyncIterator[memoryview]:
        return self._iterate()

    async def _iterate(self) -> AsyncGenerator[memoryview, None]:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("source iteration requires an asyncio task")
        task.add_done_callback(lambda _: self.stage_done.set())
        yield memoryview(self.buffer)

    async def aclose(self) -> None:
        self.owners_closed += 1
        self.buffer[0] = 0


class _OwnedSyncSource:
    def __init__(self) -> None:
        self.owners_closed = 0

    def __iter__(self) -> Iterator[int]:
        return iter(range(2))

    def close(self) -> None:
        self.owners_closed += 1


class _BrokenAsyncSource(_OwnedAsyncSource):
    def __aiter__(self) -> AsyncIterator[int]:
        raise RuntimeError("iterator creation failure")


class AsyncSourceLifecycleTest(unittest.TestCase):
    def _cancel_stage(
        self, source: _SuspendingAsyncIterator, message: str
    ) -> asyncio.Task[object]:
        task = source.stage_task
        if task is None:
            self.fail("Source iteration did not start")

        async def cancel(target: asyncio.Task[object]) -> None:
            target.cancel(message)
            await asyncio.sleep(0)

        asyncio.run_coroutine_threadsafe(cancel(task), task.get_loop()).result(
            timeout=5
        )
        return task

    def test_stop_closes_active_iterator(self) -> None:
        """Cancellation releases the iterator without closing its source owner."""
        for thread_output in (False, True):
            with self.subTest(thread_output=thread_output):
                source = _OwnedAsyncSource(10000)
                pipeline = (
                    PipelineBuilder()
                    .add_source(source)
                    .add_sink(buffer_size=1)
                    .build(num_threads=1, use_thread_output_queue=thread_output)
                )
                pipeline.start()
                try:
                    self.assertTrue(source.started.wait(timeout=5))
                    self.assertEqual(0, pipeline.get_item(timeout=5))
                finally:
                    pipeline.stop(timeout=5)

                self.assertEqual(1, source.iterators_closed)
                self.assertEqual(0, source.owners_closed)

    def test_finite_source_keeps_owner_open(self) -> None:
        """Finite exhaustion closes the active iterator and leaves its owner open."""
        source = _OwnedAsyncSource(2)
        pipeline = (
            PipelineBuilder()
            .add_source(source)
            .add_sink(buffer_size=1)
            .build(num_threads=1)
        )
        with pipeline.auto_stop():
            self.assertEqual([0, 1], list(pipeline.get_iterator(timeout=5)))
            self.assertEqual(1, source.iterators_closed)
            self.assertEqual(0, source.owners_closed)

        self.assertEqual(0, source.owners_closed)

    def test_continuous_epochs_keep_owner_open(self) -> None:
        """Epoch boundaries release iterators while preserving the reusable owner."""
        source = _OwnedAsyncSource(2)
        pipeline = (
            PipelineBuilder()
            .add_source(source, continuous=True)
            .add_sink(buffer_size=1)
            .build(num_threads=1)
        )
        with pipeline.auto_stop():
            for _ in range(3):
                self.assertEqual([0, 1], list(pipeline.get_iterator(timeout=5)))
                self.assertEqual(0, source.owners_closed)

        self.assertGreaterEqual(len(source.iterators), 3)
        self.assertEqual(len(source.iterators), source.iterators_closed)
        self.assertEqual(0, source.owners_closed)

    def test_continuous_self_iterator_is_closed_only_at_shutdown(self) -> None:
        """A reusable self-iterator stays open between epochs and closes once."""
        source = _ReusableAsyncIterator()
        pipeline = (
            PipelineBuilder()
            .add_source(source, continuous=True)
            .add_sink(buffer_size=1)
            .build(num_threads=1)
        )
        with pipeline.auto_stop():
            for _ in range(3):
                self.assertEqual([1, 2], list(pipeline.get_iterator(timeout=5)))
                self.assertEqual(0, source.closed)

        self.assertEqual(1, source.closed)

    def test_iterator_that_owns_itself_is_closed_once(self) -> None:
        """An iterator that is also the source receives one close call."""
        source = _AsyncIterator()
        pipeline = (
            PipelineBuilder()
            .add_source(source)
            .add_sink(buffer_size=1)
            .build(num_threads=1)
        )
        try:
            self.assertEqual([1, 2], list(pipeline.get_iterator(timeout=5)))
        finally:
            pipeline.stop(timeout=5)
        pipeline.stop(timeout=5)

        self.assertEqual(1, source.closed)

    def test_source_without_close_releases_its_iterator(self) -> None:
        """Stopping releases an iterator even when its source has no close method."""
        source = _AsyncSource(10000)
        pipeline = (
            PipelineBuilder()
            .add_source(source)
            .add_sink(buffer_size=1)
            .build(num_threads=1)
        )
        with pipeline.auto_stop():
            self.assertEqual(0, pipeline.get_item(timeout=5))

        self.assertEqual(1, source.iterators_closed)

    def test_source_failure_is_not_masked_by_close_failure(self) -> None:
        """Cleanup preserves the original source failure when closing also fails."""
        source = _AsyncIterator(fail=True, close_fail=True)
        pipeline = (
            PipelineBuilder()
            .add_source(source)
            .add_sink(buffer_size=1)
            .build(num_threads=1)
        )
        with self.assertRaises(PipelineFailure) as ctx:
            with pipeline.auto_stop():
                list(pipeline.get_iterator(timeout=5))

        self.assertEqual(1, source.closed)
        self.assertEqual(1, len(ctx.exception.exceptions))
        self.assertEqual("source failure", str(ctx.exception.exceptions[0]))

    def test_close_failure_is_reported(self) -> None:
        """A cleanup failure is observable when the source itself succeeds."""
        source = _AsyncIterator(close_fail=True)
        pipeline = (
            PipelineBuilder()
            .add_source(source)
            .add_sink(buffer_size=1)
            .build(num_threads=1)
        )
        with self.assertRaises(PipelineFailure) as ctx:
            with pipeline.auto_stop():
                list(pipeline.get_iterator(timeout=5))

        self.assertEqual(1, source.closed)
        self.assertEqual(1, len(ctx.exception.exceptions))
        self.assertEqual("close failure", str(ctx.exception.exceptions[0]))

    def test_iterator_creation_failure_keeps_owner_open(self) -> None:
        """An iterator creation failure leaves the source owned by its caller."""
        source = _BrokenAsyncSource(1)
        pipeline = (
            PipelineBuilder()
            .add_source(source)
            .add_sink(buffer_size=1)
            .build(num_threads=1)
        )
        with self.assertRaises(PipelineFailure) as ctx:
            with pipeline.auto_stop():
                list(pipeline.get_iterator(timeout=5))

        self.assertEqual(0, source.owners_closed)
        self.assertEqual("iterator creation failure", str(ctx.exception.exceptions[0]))

    def test_repeated_cancellation_finishes_close(self) -> None:
        """Repeated cancellation finishes cleanup and preserves the first cancellation."""
        for close_fail in (False, True):
            with self.subTest(close_fail=close_fail):
                source = _SuspendingAsyncIterator(
                    wait_for_cancel=True, close_fail=close_fail
                )
                pipeline = (
                    PipelineBuilder()
                    .add_source(source)
                    .add_sink(buffer_size=3)
                    .build(num_threads=1, queue_class=_source_error_queue(source))
                )
                pipeline.start()
                try:
                    self.assertEqual(1, pipeline.get_item(timeout=5))
                    task = self._cancel_stage(source, "original cancellation")
                    self.assertTrue(source.close_started.wait(timeout=5))
                    self._cancel_stage(source, "repeated cancellation")
                finally:
                    if source.stage_task is not None:
                        source.stage_task.get_loop().call_soon_threadsafe(
                            source.release_close.set
                        )
                    pipeline.stop(timeout=5)

                self.assertTrue(source.close_finished.is_set())
                self.assertEqual(1, source.closed)
                self.assertTrue(task.cancelled())
                self.assertIsInstance(source.stage_error, asyncio.CancelledError)
                self.assertIs(source.iteration_error, source.stage_error)
                self.assertEqual("original cancellation", str(source.stage_error))

    def test_cancellation_during_successful_close_is_preserved(self) -> None:
        """Cancellation during normal cleanup is deferred until cleanup finishes."""
        source = _SuspendingAsyncIterator()
        pipeline = (
            PipelineBuilder()
            .add_source(source)
            .add_sink(buffer_size=3)
            .build(num_threads=1, queue_class=_source_error_queue(source))
        )
        pipeline.start()
        try:
            self.assertTrue(source.close_started.wait(timeout=5))
            task = self._cancel_stage(source, "cancellation during close")
            self._cancel_stage(source, "repeated cancellation")
        finally:
            if source.stage_task is not None:
                source.stage_task.get_loop().call_soon_threadsafe(
                    source.release_close.set
                )
            pipeline.stop(timeout=5)

        self.assertTrue(source.close_finished.is_set())
        self.assertEqual(1, source.closed)
        self.assertTrue(task.cancelled())
        self.assertIsInstance(source.stage_error, asyncio.CancelledError)
        self.assertEqual("cancellation during close", str(source.stage_error))

    def test_repeated_cancellation_preserves_source_failure(self) -> None:
        """Repeated cancellation during cleanup cannot replace a source failure."""
        for close_fail in (False, True):
            with self.subTest(close_fail=close_fail):
                source = _SuspendingAsyncIterator(fail=True, close_fail=close_fail)
                pipeline = (
                    PipelineBuilder()
                    .add_source(source)
                    .add_sink(buffer_size=3)
                    .build(num_threads=1)
                )
                pipeline.start()
                try:
                    self.assertTrue(source.close_started.wait(timeout=5))
                    self._cancel_stage(source, "cancellation during close")
                    self._cancel_stage(source, "repeated cancellation")
                    task = source.stage_task
                    if task is None:
                        self.fail("Source iteration did not start")
                    task.get_loop().call_soon_threadsafe(source.release_close.set)
                    with self.assertRaises(PipelineFailure) as ctx:
                        list(pipeline.get_iterator(timeout=5))
                finally:
                    if source.stage_task is not None:
                        source.stage_task.get_loop().call_soon_threadsafe(
                            source.release_close.set
                        )
                    pipeline.stop(timeout=5)

                self.assertTrue(source.close_finished.is_set())
                self.assertEqual(1, source.closed)
                self.assertEqual(1, len(ctx.exception.exceptions))
                self.assertEqual("source failure", str(ctx.exception.exceptions[0]))

    def test_queued_items_can_use_their_source_owner(self) -> None:
        """Queued items retain access to a separate owner after source exhaustion."""
        source = _OwnedBufferSource()
        downstream_started = threading.Event()
        release_downstream = threading.Event()

        def consume(item: memoryview) -> int:
            downstream_started.set()
            if not release_downstream.wait(timeout=5):
                raise TimeoutError("downstream was not released")
            return item[0]

        pipeline = (
            PipelineBuilder()
            .add_source(source)
            .pipe(consume)
            .add_sink(buffer_size=1)
            .build(num_threads=1)
        )
        pipeline.start()
        try:
            self.assertTrue(downstream_started.wait(timeout=5))
            self.assertTrue(source.stage_done.wait(timeout=5))
            self.assertEqual(0, source.owners_closed)
            release_downstream.set()
            self.assertEqual([7], list(pipeline.get_iterator(timeout=5)))
        finally:
            release_downstream.set()
            pipeline.stop(timeout=5)

        self.assertEqual(0, source.owners_closed)

    def test_sync_source_keeps_owner_open(self) -> None:
        """Synchronous source owners also remain owned by their callers."""
        source = _OwnedSyncSource()
        pipeline = (
            PipelineBuilder()
            .add_source(source)
            .add_sink(buffer_size=1)
            .build(num_threads=1)
        )
        with pipeline.auto_stop():
            self.assertEqual([0, 1], list(pipeline.get_iterator(timeout=5)))

        self.assertEqual(0, source.owners_closed)
