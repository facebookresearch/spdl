# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import queue
import sys
import threading
import time
import unittest
from collections.abc import Coroutine, Iterable
from concurrent.futures import (
    CancelledError as FutureCancelledError,
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from typing import Any
from unittest.mock import patch

from spdl.pipeline import Pipeline, PipelineBuilder
from spdl.pipeline._pipeline import _QueueReadTimedOut

_TIMEOUT: float = 30.0
_RELEASE_OP: threading.Event = threading.Event()


def _blocks_until_released(value: int) -> int:
    _RELEASE_OP.wait()
    return value


def _make_pipeline(
    coro: Coroutine[Any, Any, None], output_queue: asyncio.Queue[int]
) -> Pipeline[int]:
    return Pipeline(
        coro,
        output_queue,
        ThreadPoolExecutor(max_workers=1),
        desc="pipeline core lifecycle test",
    )


class PipelineGetItemTimeoutTest(unittest.TestCase):
    def test_recovered_item_is_rechecked_before_next_wait(self) -> None:
        """A cancelled wrapper cannot delay an item recovered by its read."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)

        async def wait_until_cancelled() -> None:
            await asyncio.Event().wait()

        pipeline = _make_pipeline(wait_until_cancelled(), output_queue)

        class _CancelledReadWithRecoveredItem(Future[int]):
            def result(self, timeout: float | None = None) -> int:
                pipeline._impl._pending_output_items.append(1)
                raise FutureCancelledError

        try:
            pipeline.start(timeout=_TIMEOUT)
            pipeline._impl._pending_output_read = _CancelledReadWithRecoveredItem()

            self.assertEqual(pipeline.get_item(timeout=_TIMEOUT), 1)
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_nowait_returns_item_recovered_by_cancelled_wrapper(self) -> None:
        """A nonblocking poll returns an item recovered during settlement."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)

        async def wait_until_cancelled() -> None:
            await asyncio.Event().wait()

        pipeline = _make_pipeline(wait_until_cancelled(), output_queue)

        class _CancelledReadWithRecoveredItem(Future[int]):
            def result(self, timeout: float | None = None) -> int:
                pipeline._impl._pending_output_items.append(1)
                raise FutureCancelledError

        try:
            pipeline.start(timeout=_TIMEOUT)
            pipeline._impl._pending_output_read = _CancelledReadWithRecoveredItem()

            self.assertEqual(pipeline._get_item_nowait(), 1)
            self.assertIsNone(pipeline._impl._pending_output_read)
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_nowait_returns_item_buffered_before_publication_timeout(self) -> None:
        """A nonblocking poll rechecks ownership after publication stalls."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)

        async def wait_until_cancelled() -> None:
            await asyncio.Event().wait()

        pipeline = _make_pipeline(wait_until_cancelled(), output_queue)

        class _UnpublishedBufferedRead(Future[int]):
            def result(self, timeout: float | None = None) -> int:
                pipeline._impl._pending_output_items.append(1)
                raise FutureTimeoutError

        try:
            pipeline.start(timeout=_TIMEOUT)
            pipeline._impl._pending_output_read = _UnpublishedBufferedRead()

            self.assertEqual(pipeline._get_item_nowait(), 1)
            self.assertIsNone(pipeline._impl._pending_output_read)
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_completed_nowait_returns_previously_recovered_item_before_eof(
        self,
    ) -> None:
        """Task completion cannot strand an item already in the transfer buffer."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)

        async def wait_until_cancelled() -> None:
            await asyncio.Event().wait()

        pipeline = _make_pipeline(wait_until_cancelled(), output_queue)
        try:
            pipeline.start(timeout=_TIMEOUT)
            pipeline._impl._pending_output_items.append(1)
            with patch.object(
                pipeline._impl._event_loop,
                "is_task_completed",
                return_value=True,
            ):
                self.assertEqual(pipeline._get_item_nowait(), 1)
                with self.assertRaises(EOFError):
                    pipeline._get_item_nowait()
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_interrupted_loop_wait_preserves_completed_queue_result(self) -> None:
        """An interrupted read stays ahead of a producer's bounded refill."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)
        output_queue.put_nowait(1)

        async def wait_until_cancelled() -> None:
            await asyncio.Event().wait()

        class _InterruptedWait(BaseException):
            pass

        original_wait = asyncio.wait
        interrupted = False

        async def interrupt_completed_read(
            futures: Iterable[asyncio.Task[Any]],
            *,
            timeout: float | None = None,
        ) -> tuple[set[asyncio.Task[Any]], set[asyncio.Task[Any]]]:
            nonlocal interrupted
            tasks = set(futures)
            if not interrupted and all(
                task.get_name() != "Pipeline::main" for task in tasks
            ):
                interrupted = True
                while not all(task.done() for task in tasks):
                    await asyncio.sleep(0)
                output_queue.put_nowait(2)
                raise _InterruptedWait
            return await original_wait(tasks, timeout=timeout)

        pipeline = _make_pipeline(wait_until_cancelled(), output_queue)
        try:
            pipeline.start(timeout=_TIMEOUT)
            with patch.object(asyncio, "wait", new=interrupt_completed_read):
                with self.assertRaises(_InterruptedWait):
                    pipeline.get_item(timeout=_TIMEOUT)
            self.assertEqual(pipeline.get_item(timeout=_TIMEOUT), 1)
            self.assertEqual(pipeline.get_item(timeout=_TIMEOUT), 2)
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_completed_task_waits_for_delayed_queue_publication(self) -> None:
        """Task completion does not abandon a queue result awaiting publication."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)

        async def wait_until_cancelled() -> None:
            await asyncio.Event().wait()

        delayed_future: Future[int] = Future()
        task_completed = threading.Event()

        def delay_publication(coro: Coroutine[None, None, int]) -> Future[int]:
            coro.close()
            return delayed_future

        pipeline = _make_pipeline(wait_until_cancelled(), output_queue)
        try:
            pipeline.start(timeout=_TIMEOUT)
            with (
                patch.object(
                    pipeline._impl._event_loop,
                    "run_coroutine_threadsafe",
                    side_effect=delay_publication,
                ) as submit,
                patch.object(
                    pipeline._impl._event_loop,
                    "is_task_completed",
                    side_effect=task_completed.is_set,
                ),
            ):
                with self.assertRaises(TimeoutError):
                    pipeline.get_item(timeout=0.01)
                task_completed.set()
                delayed_future.set_result(1)
                self.assertEqual(pipeline.get_item(timeout=_TIMEOUT), 1)
            self.assertEqual(submit.call_count, 1)
            self.assertFalse(delayed_future.cancelled())
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_caller_timeout_preserves_delayed_queue_result(self) -> None:
        """A caller deadline does not discard a queue result awaiting publication."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)

        async def wait_until_cancelled() -> None:
            await asyncio.Event().wait()

        class _RecordingFuture(Future[int]):
            def __init__(self) -> None:
                super().__init__()
                self.result_timeouts: list[float | None] = []

            def result(self, timeout: float | None = None) -> int:
                self.result_timeouts.append(timeout)
                return super().result(timeout)

        delayed_future = _RecordingFuture()

        def delay_publication(coro: Coroutine[None, None, int]) -> Future[int]:
            coro.close()
            return delayed_future

        pipeline = _make_pipeline(wait_until_cancelled(), output_queue)
        try:
            pipeline.start(timeout=_TIMEOUT)
            with patch.object(
                pipeline._impl._event_loop,
                "run_coroutine_threadsafe",
                side_effect=delay_publication,
            ) as submit:
                with self.assertRaises(TimeoutError):
                    pipeline.get_item(timeout=0.01)
                delayed_future.set_result(1)
                self.assertEqual(pipeline.get_item(timeout=_TIMEOUT), 1)

            self.assertEqual(submit.call_count, 1)
            first_timeout = delayed_future.result_timeouts[0]
            self.assertIsNotNone(first_timeout)
            assert first_timeout is not None
            self.assertLessEqual(first_timeout, 0.01)
            self.assertFalse(delayed_future.cancelled())
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_settled_slice_timeout_racing_foreground_timeout_is_internal(
        self,
    ) -> None:
        """A settled internal timeout never leaks across a foreground race."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)

        async def wait_until_cancelled() -> None:
            await asyncio.Event().wait()

        class _RacingTimeoutFuture(Future[int]):
            def result(self, timeout: float | None = None) -> int:
                if not self.done():
                    self.set_exception(_QueueReadTimedOut())
                    raise FutureTimeoutError
                return super().result(timeout)

        pipeline = _make_pipeline(wait_until_cancelled(), output_queue)
        try:
            pipeline.start(timeout=_TIMEOUT)
            pipeline._impl._pending_output_read = _RacingTimeoutFuture()
            with self.assertRaises(queue.Empty):
                pipeline._impl._get_pending_output_read(timeout=0.01)
            self.assertIsNone(pipeline._impl._pending_output_read)
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_interrupted_wait_preserves_pending_queue_result(self) -> None:
        """An interrupted foreground wait does not abandon its loop-side read."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)

        async def wait_until_cancelled() -> None:
            await asyncio.Event().wait()

        class _InterruptedFuture(Future[int]):
            def __init__(self) -> None:
                super().__init__()
                self._interrupt = True

            def result(self, timeout: float | None = None) -> int:
                if self._interrupt:
                    self._interrupt = False
                    raise KeyboardInterrupt
                return super().result(timeout)

        interrupted_future = _InterruptedFuture()

        def interrupt_wait(coro: Coroutine[None, None, int]) -> Future[int]:
            coro.close()
            return interrupted_future

        pipeline = _make_pipeline(wait_until_cancelled(), output_queue)
        try:
            pipeline.start(timeout=_TIMEOUT)
            with patch.object(
                pipeline._impl._event_loop,
                "run_coroutine_threadsafe",
                side_effect=interrupt_wait,
            ) as submit:
                with self.assertRaises(KeyboardInterrupt):
                    pipeline.get_item(timeout=_TIMEOUT)
                interrupted_future.set_result(1)
                self.assertEqual(pipeline.get_item(timeout=_TIMEOUT), 1)
            self.assertEqual(submit.call_count, 1)
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_settled_base_exception_clears_pending_read(self) -> None:
        """A settled read is cleared even when its error is re-raised distinctly."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)

        async def wait_until_cancelled() -> None:
            await asyncio.Event().wait()

        class _DistinctReraiseFuture(Future[int]):
            def result(self, timeout: float | None = None) -> int:
                try:
                    return super().result(timeout)
                except KeyboardInterrupt as error:
                    raise KeyboardInterrupt(*error.args) from None

        settled_future = _DistinctReraiseFuture()
        settled_future.set_exception(KeyboardInterrupt("settled read failed"))
        pipeline = _make_pipeline(wait_until_cancelled(), output_queue)
        pipeline._impl._pending_output_read = settled_future
        try:
            with self.assertRaisesRegex(KeyboardInterrupt, "settled read failed"):
                pipeline._impl._get_pending_output_read(timeout=0.0)
            self.assertIsNone(pipeline._impl._pending_output_read)
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_stopped_loop_does_not_block_queue_result_publication(self) -> None:
        """A stopped loop cannot leave the foreground waiting indefinitely."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)

        async def wait_until_cancelled() -> None:
            await asyncio.Event().wait()

        pending_future: Future[int] = Future()
        pending_future.set_running_or_notify_cancel()

        def delay_forever(coro: Coroutine[None, None, int]) -> Future[int]:
            coro.close()
            return pending_future

        pipeline = _make_pipeline(wait_until_cancelled(), output_queue)
        try:
            pipeline.start(timeout=_TIMEOUT)
            with (
                patch.object(
                    pipeline._impl._event_loop,
                    "run_coroutine_threadsafe",
                    side_effect=delay_forever,
                ),
                patch.object(
                    pipeline._impl._event_loop,
                    "is_task_completed",
                    return_value=False,
                ),
                patch.object(
                    pipeline._impl._event_loop,
                    "is_running",
                    return_value=False,
                ),
            ):
                with self.assertRaisesRegex(TimeoutError, "event loop stopped"):
                    pipeline.get_item(timeout=0.01)
                self.assertIs(pipeline._impl._pending_output_read, pending_future)
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_completed_stopped_loop_retires_unpublished_read_at_eof(self) -> None:
        """An unpublishable read cannot turn terminal EOF into repeated timeouts."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)

        async def wait_until_cancelled() -> None:
            await asyncio.Event().wait()

        pending_future: Future[int] = Future()
        pending_future.set_running_or_notify_cancel()
        pipeline = _make_pipeline(wait_until_cancelled(), output_queue)
        try:
            pipeline.start(timeout=_TIMEOUT)
            pipeline._impl._pending_output_read = pending_future
            with (
                patch.object(
                    pipeline._impl._event_loop,
                    "is_task_completed",
                    return_value=True,
                ),
                patch.object(
                    pipeline._impl._event_loop,
                    "is_running",
                    return_value=False,
                ),
            ):
                with self.assertRaises(EOFError):
                    pipeline.get_item(timeout=0.01)
                with self.assertRaises(EOFError):
                    pipeline.get_item(timeout=0.01)
            self.assertIsNone(pipeline._impl._pending_output_read)
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_stalled_publication_returns_buffered_final_item_before_eof(self) -> None:
        """A loop-consumed final item wins before its unpublishable Future."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)

        async def wait_until_cancelled() -> None:
            await asyncio.Event().wait()

        pending_future: Future[int] = Future()
        pending_future.set_running_or_notify_cancel()
        pipeline = _make_pipeline(wait_until_cancelled(), output_queue)
        try:
            pipeline.start(timeout=_TIMEOUT)
            pipeline._impl._pending_output_read = pending_future
            pipeline._impl._pending_output_items.append(1)
            with (
                patch.object(
                    pipeline._impl._event_loop,
                    "is_task_completed",
                    return_value=True,
                ),
                patch.object(
                    pipeline._impl._event_loop,
                    "is_running",
                    return_value=False,
                ),
            ):
                self.assertEqual(pipeline.get_item(timeout=0.01), 1)
                with self.assertRaises(EOFError):
                    pipeline.get_item(timeout=0.01)
            self.assertIsNone(pipeline._impl._pending_output_read)
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_stopped_loop_harvests_result_that_wins_cancellation_race(self) -> None:
        """A result completing during cancellation is returned instead of lost."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)

        async def wait_until_cancelled() -> None:
            await asyncio.Event().wait()

        class _CompletingOnCancelFuture(Future[int]):
            def cancel(self) -> bool:
                if not self.done():
                    self.set_result(1)
                return super().cancel()

        completing_future = _CompletingOnCancelFuture()

        def complete_on_cancel(coro: Coroutine[None, None, int]) -> Future[int]:
            coro.close()
            return completing_future

        pipeline = _make_pipeline(wait_until_cancelled(), output_queue)
        try:
            pipeline.start(timeout=_TIMEOUT)
            with (
                patch.object(
                    pipeline._impl._event_loop,
                    "run_coroutine_threadsafe",
                    side_effect=complete_on_cancel,
                ),
                patch.object(
                    pipeline._impl._event_loop,
                    "is_task_completed",
                    return_value=False,
                ),
                patch.object(
                    pipeline._impl._event_loop,
                    "is_running",
                    return_value=False,
                ),
            ):
                self.assertEqual(pipeline.get_item(timeout=0.01), 1)
            self.assertFalse(completing_future.cancelled())
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_stopped_loop_waits_for_final_result_publication(self) -> None:
        """A stopped loop gives an already-consumed item time to publish."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)

        async def wait_until_cancelled() -> None:
            await asyncio.Event().wait()

        class _PublishesDuringGraceFuture(Future[int]):
            def __init__(self) -> None:
                super().__init__()
                self._result_calls = 0
                self.set_running_or_notify_cancel()

            def cancel(self) -> bool:
                return False

            def result(self, timeout: float | None = None) -> int:
                self._result_calls += 1
                if self._result_calls == 1 or not timeout:
                    raise FutureTimeoutError
                self.set_result(1)
                return super().result(timeout)

        pending_future = _PublishesDuringGraceFuture()
        pipeline = _make_pipeline(wait_until_cancelled(), output_queue)
        try:
            pipeline.start(timeout=_TIMEOUT)
            pipeline._impl._pending_output_read = pending_future
            with (
                patch.object(
                    pipeline._impl._event_loop,
                    "is_task_completed",
                    return_value=False,
                ),
                patch.object(
                    pipeline._impl._event_loop, "is_running", return_value=False
                ),
            ):
                self.assertEqual(pipeline.get_item(timeout=0.01), 1)
            self.assertIsNone(pipeline._impl._pending_output_read)
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_stopped_loop_returns_item_recovered_from_cancelled_read(self) -> None:
        """A cancelled wrapper cannot hide an item recovered by its coroutine."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)

        async def wait_until_cancelled() -> None:
            await asyncio.Event().wait()

        class _UncancellableCancelledFuture(Future[int]):
            def cancel(self) -> bool:
                return False

            def result(self, timeout: float | None = None) -> int:
                raise FutureCancelledError

        pipeline = _make_pipeline(wait_until_cancelled(), output_queue)
        try:
            pipeline.start(timeout=_TIMEOUT)
            pipeline._impl._pending_output_items.append(1)
            pipeline._impl._pending_output_read = _UncancellableCancelledFuture()

            self.assertEqual(pipeline._impl._resolve_output_read_after_loop_stop(), 1)
            self.assertIsNone(pipeline._impl._pending_output_read)
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_nowait_submission_failure_recovers_completed_buffered_output(self) -> None:
        """A nonblocking read recovers output after its owner loop stops."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)
        output_queue.put_nowait(1)

        async def complete_immediately() -> None:
            return None

        pipeline = _make_pipeline(complete_immediately(), output_queue)
        try:
            pipeline.start(timeout=_TIMEOUT)
            event_loop = pipeline._impl._event_loop
            deadline = time.monotonic() + _TIMEOUT
            while not event_loop.is_task_completed():
                if time.monotonic() >= deadline:
                    self.fail("Pipeline task did not complete before the timeout.")
                time.sleep(0.001)

            with (
                patch.object(
                    event_loop,
                    "run_coroutine_threadsafe",
                    side_effect=RuntimeError("Event loop is closed"),
                ) as submit,
                patch.object(
                    event_loop,
                    "is_task_completed",
                    side_effect=(False, True),
                ),
                patch.object(event_loop, "is_running", return_value=False),
            ):
                self.assertEqual(pipeline._get_item_nowait(), 1)
            submit.assert_called_once()
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_wait_submission_failure_recovers_completed_buffered_output(self) -> None:
        """A blocking read recovers output after its owner loop stops."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)
        output_queue.put_nowait(1)

        async def complete_immediately() -> None:
            return None

        pipeline = _make_pipeline(complete_immediately(), output_queue)
        try:
            pipeline.start(timeout=_TIMEOUT)
            event_loop = pipeline._impl._event_loop
            deadline = time.monotonic() + _TIMEOUT
            while not event_loop.is_task_completed():
                if time.monotonic() >= deadline:
                    self.fail("Pipeline task did not complete before the timeout.")
                time.sleep(0.001)

            with (
                patch.object(
                    event_loop,
                    "run_coroutine_threadsafe",
                    side_effect=RuntimeError("Event loop is closed"),
                ) as submit,
                patch.object(
                    event_loop,
                    "is_task_completed",
                    side_effect=(False, True),
                ),
                patch.object(event_loop, "is_running", return_value=False),
            ):
                self.assertEqual(pipeline.get_item(timeout=_TIMEOUT), 1)
            submit.assert_called_once()
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_tiny_timeout_preserves_already_buffered_item(self) -> None:
        """A rounded-to-zero timeout preserves an already buffered item."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)
        output_queue.put_nowait(1)

        async def wait_until_cancelled() -> None:
            await asyncio.Event().wait()

        pipeline = _make_pipeline(wait_until_cancelled(), output_queue)
        try:
            pipeline.start(timeout=_TIMEOUT)
            try:
                item = pipeline.get_item(timeout=sys.float_info.min)
            except TimeoutError:
                item = pipeline.get_item(timeout=_TIMEOUT)
            self.assertEqual(item, 1)
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_internal_read_slices_do_not_escape_as_timeouts(self) -> None:
        """An empty queue waits until the caller deadline, not one internal slice."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)

        async def wait_until_cancelled() -> None:
            await asyncio.Event().wait()

        pipeline = _make_pipeline(wait_until_cancelled(), output_queue)
        try:
            pipeline.start(timeout=_TIMEOUT)
            with self.assertRaisesRegex(TimeoutError, "The next item is not available"):
                pipeline.get_item(timeout=0.25)
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_outer_cancellation_is_not_reported_as_slice_timeout(self) -> None:
        """Cancelling the caller propagates instead of looking like a read timeout."""

        async def exercise_cancellation() -> None:
            read_cancelled = asyncio.Event()
            release_cancellation = asyncio.Event()

            class _SlowCancellationQueue(asyncio.Queue[int]):
                async def get(self) -> int:
                    try:
                        return await super().get()
                    except asyncio.CancelledError:
                        read_cancelled.set()
                        await release_cancellation.wait()
                        raise

            async def never_run() -> None:
                await asyncio.Event().wait()

            executor = ThreadPoolExecutor(max_workers=1)
            pipeline = Pipeline(
                never_run(),
                _SlowCancellationQueue(1),
                executor,
                desc="pipeline cancellation test",
            )
            try:
                read = asyncio.create_task(
                    pipeline._impl._get_with_timeout_on_loop(0.01)
                )
                await asyncio.wait_for(read_cancelled.wait(), timeout=_TIMEOUT)
                read.cancel()
                release_cancellation.set()
                with self.assertRaises(asyncio.CancelledError):
                    await read
            finally:
                pipeline._impl._event_loop._coro.close()
                executor.shutdown()

        asyncio.run(exercise_cancellation())

    def test_outer_cancellation_preserves_read_that_finishes_cleanup(self) -> None:
        """A child result that wins cancellation cleanup remains available."""

        async def exercise_cancellation() -> None:
            first_cancellation = asyncio.Event()

            class _CancellationReturningQueue(asyncio.Queue[int]):
                async def get(self) -> int:
                    try:
                        return await super().get()
                    except asyncio.CancelledError:
                        first_cancellation.set()
                        try:
                            await asyncio.Event().wait()
                        except asyncio.CancelledError:
                            return 1
                        raise AssertionError("cleanup wait returned unexpectedly")

            async def never_run() -> None:
                await asyncio.Event().wait()

            executor = ThreadPoolExecutor(max_workers=1)
            pipeline = Pipeline(
                never_run(),
                _CancellationReturningQueue(1),
                executor,
                desc="pipeline cancellation recovery test",
            )
            try:
                read = asyncio.create_task(
                    pipeline._impl._get_with_timeout_on_loop(0.01)
                )
                await asyncio.wait_for(first_cancellation.wait(), timeout=_TIMEOUT)
                read.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await read
                self.assertEqual(pipeline._impl._take_pending_output_item(), 1)
            finally:
                pipeline._impl._event_loop._coro.close()
                executor.shutdown()

        asyncio.run(exercise_cancellation())

    def test_sink_timeout_error_is_not_an_internal_read_timeout(self) -> None:
        """A custom sink's TimeoutError propagates instead of being retried."""

        class _TimeoutQueue(asyncio.Queue[int]):
            async def get(self) -> int:
                await asyncio.sleep(0)
                raise TimeoutError("custom sink timeout")

        output_queue = _TimeoutQueue(1)

        async def wait_until_cancelled() -> None:
            await asyncio.Event().wait()

        pipeline = _make_pipeline(wait_until_cancelled(), output_queue)
        try:
            pipeline.start(timeout=_TIMEOUT)
            with self.assertRaisesRegex(TimeoutError, "custom sink timeout"):
                pipeline.get_item(timeout=_TIMEOUT)
        finally:
            pipeline.stop(timeout=_TIMEOUT)

    def test_zero_and_finite_timeouts_do_not_wait_for_blocked_loop(self) -> None:
        """A busy owner loop cannot make bounded foreground polls hang."""
        output_queue: asyncio.Queue[int] = asyncio.Queue(1)
        output_queue.put_nowait(1)
        loop_blocked = threading.Event()
        release_loop = threading.Event()

        async def wait_until_cancelled() -> None:
            await asyncio.Event().wait()

        def block_loop() -> None:
            loop_blocked.set()
            if not release_loop.wait(_TIMEOUT):
                raise RuntimeError("Test did not release the event loop.")

        pipeline = _make_pipeline(wait_until_cancelled(), output_queue)
        try:
            pipeline.start(timeout=_TIMEOUT)
            loop = pipeline._impl._event_loop._loop
            assert loop is not None
            loop.call_soon_threadsafe(block_loop)
            self.assertTrue(loop_blocked.wait(_TIMEOUT))

            for timeout in (0.0, 0.01):
                with self.subTest(timeout=timeout):
                    start = time.monotonic()
                    with self.assertRaises(TimeoutError):
                        pipeline.get_item(timeout=timeout)
                    self.assertLess(time.monotonic() - start, 1.0)

            release_loop.set()
            self.assertEqual(pipeline.get_item(timeout=_TIMEOUT), 1)
        finally:
            release_loop.set()
            pipeline.stop(timeout=_TIMEOUT)

    def _assert_timed_out_get_does_not_consume_next_item(self, timeout: float) -> None:
        """A timed-out async-queue read must not remain registered as a consumer."""
        _RELEASE_OP.clear()
        pipeline = (
            PipelineBuilder()
            .add_source([1])
            .pipe(_blocks_until_released)
            .add_sink(1)
            .build(num_threads=1, use_thread_output_queue=False)
        )
        try:
            pipeline.start(timeout=_TIMEOUT)
            with self.assertRaises(TimeoutError):
                pipeline.get_item(timeout=timeout)

            # This one-shot poll is scheduled behind the timed-out read. Once it returns,
            # an incorrectly abandoned ``Queue.get`` is registered and would steal the only
            # item produced below.
            with self.assertRaises(queue.Empty):
                pipeline._get_item_nowait()
            _RELEASE_OP.set()
            self.assertEqual(pipeline.get_item(timeout=_TIMEOUT), 1)
            with self.assertRaises(EOFError):
                pipeline.get_item(timeout=_TIMEOUT)
        finally:
            _RELEASE_OP.set()
            pipeline.stop(timeout=_TIMEOUT)

    def test_timed_out_get_does_not_consume_next_item(self) -> None:
        """Zero and finite timeouts both leave the next produced item intact."""
        for name, timeout in [("zero", 0.0), ("finite", 0.01)]:
            with self.subTest(name=name):
                self._assert_timed_out_get_does_not_consume_next_item(timeout)

    def test_last_item_wins_completion_race(self) -> None:
        """An item delivered as the task completes is returned before EOF."""
        reader_started = threading.Event()

        class _SignalingQueue(asyncio.Queue[int]):
            async def get(self) -> int:
                reader_started.set()
                return await super().get()

        output_queue = _SignalingQueue(1)

        async def put_after_reader_waits() -> None:
            while not reader_started.is_set():
                await asyncio.sleep(0)
            await output_queue.put(1)

        pipeline = _make_pipeline(put_after_reader_waits(), output_queue)
        try:
            self.assertEqual(pipeline.get_item(timeout=_TIMEOUT), 1)
            with self.assertRaises(EOFError):
                pipeline.get_item(timeout=_TIMEOUT)
        finally:
            pipeline.stop(timeout=_TIMEOUT)
