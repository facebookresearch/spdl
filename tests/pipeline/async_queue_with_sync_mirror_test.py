# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import unittest
from collections.abc import Coroutine
from typing import Any
from unittest.mock import patch

from spdl.pipeline._components._common import StageInfo
from spdl.pipeline._components._queue import _AsyncQueueWithSyncMirror, StatsQueue


class AsyncQueueWithSyncMirrorTest(unittest.IsolatedAsyncioTestCase):
    def _make_queue(
        self, buffer_size: int = 1
    ) -> tuple[_AsyncQueueWithSyncMirror, StatsQueue]:
        inner = StatsQueue(
            StageInfo(pipeline_id=0, stage_id="sink", stage_name="sink"),
            buffer_size=buffer_size,
        )
        return _AsyncQueueWithSyncMirror(inner), inner

    async def test_same_turn_cancellation_cannot_split_put(self) -> None:
        """A cancellation queued after put starts cannot split mirror publication."""
        queue, inner = self._make_queue()
        put = asyncio.create_task(queue.put(1))
        asyncio.get_running_loop().call_soon(put.cancel)

        await put

        self.assertEqual(inner.qsize(), 1)
        self.assertEqual(queue.qsize(), 1)
        self.assertEqual(await queue.get(), 1)

    async def test_cancelled_blocked_put_keeps_queues_aligned(self) -> None:
        """Cancelling a capacity-blocked put publishes to neither queue."""
        queue, inner = self._make_queue()
        await queue.put(1)
        put = asyncio.create_task(queue.put(2))
        await asyncio.sleep(0)
        self.assertFalse(put.done())

        put.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await put

        self.assertEqual(inner.qsize(), 1)
        self.assertEqual(queue.qsize(), 1)
        self.assertEqual(await queue.get(), 1)

    async def test_post_commit_stats_failure_keeps_item_mirrored(self) -> None:
        """Post-commit accounting failures leave the accepted item visible."""
        queue, inner = self._make_queue()

        with (
            patch.object(
                inner._putc,
                "update",
                side_effect=RuntimeError("stats failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "stats failure"),
        ):
            await queue.put(1)

        self.assertEqual(inner.qsize(), 1)
        self.assertEqual(queue.qsize(), 1)
        self.assertEqual(queue.get_nowait(), 1)

    async def test_mirror_failure_rolls_back_wrapped_put(self) -> None:
        """A mirror publication failure rolls back the wrapped queue commit."""
        queue, inner = self._make_queue()

        with (
            patch.object(
                queue._sync_queue,
                "put_nowait",
                side_effect=RuntimeError("mirror failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "mirror failure"),
        ):
            await queue.put(1)

        self.assertTrue(inner.empty())
        self.assertTrue(queue.empty())

        await queue.put(2)
        self.assertEqual(await queue.get(), 2)

    async def test_get_nowait_preserves_mirror_when_wrapped_slot_is_missing(
        self,
    ) -> None:
        """A defensive skew recovery does not discard the visible item."""
        queue, inner = self._make_queue()
        await queue.put(1)
        self.assertEqual(inner.get_nowait(), 1)

        self.assertEqual(queue.get_nowait(), 1)
        self.assertTrue(queue.empty())

    async def test_shutdown_drain_prevents_stale_release_getter(self) -> None:
        """A drained release callback cannot consume a later queue item."""
        queue, _ = self._make_queue()
        await queue.put(1)
        self.assertEqual(queue._sync_queue.get_nowait(), 1)
        queue._register_release()

        queue._drain()
        queue._release_one()
        await asyncio.sleep(0)

        await queue.put(2)
        await asyncio.sleep(0)

        self.assertTrue(queue.full())
        self.assertEqual(await queue.get(), 2)

    async def test_missing_wrapped_slot_drops_release_token(self) -> None:
        """A release for an absent wrapped slot cannot consume a later refill."""
        queue, inner = self._make_queue()
        await queue.put(1)
        self.assertEqual(queue._sync_queue.get_nowait(), 1)
        queue._register_release()
        self.assertEqual(inner.get_nowait(), 1)

        await queue._release_one_if_available()

        self.assertEqual(queue._pending_releases, 0)
        await queue.put(2)
        queue._release_one()
        await asyncio.sleep(0)

        self.assertEqual(inner.qsize(), 1)
        self.assertEqual(await queue.get(), 2)

    async def test_release_preserves_wrapped_get_accounting(self) -> None:
        """A mirrored foreground read records one wrapped StatsQueue get."""
        queue, inner = self._make_queue()
        await queue.put(1)
        self.assertEqual(queue._sync_queue.get_nowait(), 1)
        queue._register_release()

        queue._release_one()
        await asyncio.sleep(0)

        self.assertEqual(inner._getc.num_items, 1)

    async def test_failed_release_can_be_retried(self) -> None:
        """A wrapped get failure does not lose its pending release token."""

        class _FailOnceStatsQueue(StatsQueue):
            def __init__(self) -> None:
                super().__init__(
                    StageInfo(
                        pipeline_id=0,
                        stage_id="sink",
                        stage_name="sink",
                    ),
                    buffer_size=1,
                )
                self._fail_next_get = True
                self.get_calls = 0

            async def get(self) -> object:
                self.get_calls += 1
                if self._fail_next_get:
                    self._fail_next_get = False
                    raise RuntimeError("transient get failure")
                return await super().get()

        inner = _FailOnceStatsQueue()
        queue = _AsyncQueueWithSyncMirror(inner)
        await queue.put(1)
        self.assertEqual(queue._sync_queue.get_nowait(), 1)
        queue._register_release()

        with self.assertNoLogs("spdl.pipeline._components._queue", level="ERROR"):
            await queue._release_one_if_available()
            await asyncio.sleep(0)

        self.assertTrue(inner.empty())
        self.assertEqual(inner.get_calls, 2)
        self.assertEqual(inner._getc.num_items, 1)
        self.assertEqual(queue._pending_releases, 0)

    async def test_persistent_release_failure_falls_back_to_direct_release(
        self,
    ) -> None:
        """Exhausted wrapped-get retries still restore bounded capacity."""

        class _AlwaysFailStatsQueue(StatsQueue):
            def __init__(self) -> None:
                super().__init__(
                    StageInfo(
                        pipeline_id=0,
                        stage_id="sink",
                        stage_name="sink",
                    ),
                    buffer_size=1,
                )
                self.get_calls = 0

            async def get(self) -> object:
                self.get_calls += 1
                raise RuntimeError("persistent get failure")

        inner = _AlwaysFailStatsQueue()
        queue = _AsyncQueueWithSyncMirror(inner)
        await queue.put(1)
        self.assertEqual(queue._sync_queue.get_nowait(), 1)
        queue._register_release()

        scheduled: list[Coroutine[Any, Any, None]] = []

        def capture(coro: Coroutine[Any, Any, None], **_: object) -> None:
            scheduled.append(coro)

        with patch(
            "spdl.pipeline._components._queue.create_task",
            side_effect=capture,
        ):
            await queue._release_one_if_available()
            self.assertEqual(len(scheduled), 1)
            with self.assertLogs(
                "spdl.pipeline._components._queue", level="ERROR"
            ) as logs:
                await scheduled[0]

        self.assertEqual(inner.get_calls, 2)
        self.assertEqual(len(logs.records), 1)
        error = logs.records[0].exc_info
        self.assertIsNotNone(error)
        if error is not None:
            self.assertIs(error[0], RuntimeError)
            self.assertEqual(str(error[1]), "persistent get failure")
        self.assertEqual(len(scheduled), 1)
        self.assertTrue(inner.empty())
        self.assertEqual(queue._pending_releases, 0)

        await queue.put(2)
        self.assertTrue(inner.full())
        self.assertEqual(await queue.get(), 2)

    async def test_concurrent_fallback_drain_satisfies_release(self) -> None:
        """A slot drained during fallback does not revive the wrapped get failure."""

        class _DrainOnFallbackStatsQueue(StatsQueue):
            async def get(self) -> object:
                raise RuntimeError("persistent get failure")

            def get_nowait(self) -> object:
                super().get_nowait()
                raise asyncio.QueueEmpty

        inner = _DrainOnFallbackStatsQueue(
            StageInfo(pipeline_id=0, stage_id="sink", stage_name="sink"),
            buffer_size=1,
        )
        queue = _AsyncQueueWithSyncMirror(inner)
        await queue.put(1)
        self.assertEqual(queue._sync_queue.get_nowait(), 1)
        queue._register_release()

        with self.assertLogs("spdl.pipeline._components._queue", level="ERROR") as logs:
            await queue._release_one_if_available(retries_remaining=0)

        self.assertEqual(len(logs.records), 1)
        error = logs.records[0].exc_info
        self.assertIsNotNone(error)
        if error is not None:
            self.assertIs(error[0], RuntimeError)
            self.assertEqual(str(error[1]), "persistent get failure")
        self.assertTrue(inner.empty())
        self.assertEqual(queue._pending_releases, 0)
        await queue.put(2)
        self.assertTrue(inner.full())
        self.assertEqual(await queue.get(), 2)

    async def test_scheduled_releases_match_foreground_consumptions(self) -> None:
        """Concurrent callbacks release only their registered mirror slots."""
        queue, inner = self._make_queue(buffer_size=2)
        await queue.put(1)
        await queue.put(2)

        for expected in (1, 2):
            self.assertEqual(queue._sync_queue.get_nowait(), expected)
            queue._register_release()

        # Both callbacks are queued before either task can run.
        queue._release_one()
        queue._release_one()
        await asyncio.sleep(0)

        self.assertTrue(inner.empty())
        self.assertEqual(inner._getc.num_items, 2)

        # A duplicate/stale callback must not consume a later refill.
        await queue.put(3)
        queue._release_one()
        await asyncio.sleep(0)

        self.assertEqual(inner.qsize(), 1)
        self.assertEqual(inner._getc.num_items, 2)
