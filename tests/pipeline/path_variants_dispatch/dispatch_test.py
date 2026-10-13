# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import unittest
from collections.abc import Callable

from spdl.pipeline._components._common import _EOF, _EPOCH_END, StageInfo
from spdl.pipeline._components._queue import AsyncQueue
from spdl.pipeline._components._variants import _path_variants_router, _PendingPathPuts


class PathVariantsDispatchTest(unittest.IsolatedAsyncioTestCase):
    async def test_drain_waits_for_siblings_before_reraising_failure(self) -> None:
        """Drain completes valid sibling puts before reporting a route failure."""
        puts = _PendingPathPuts()
        failed = asyncio.Event()
        release = asyncio.Event()
        delivered = asyncio.Event()

        async def fail() -> None:
            failed.set()
            raise ValueError("route failed")

        async def deliver() -> None:
            await release.wait()
            delivered.set()

        puts.pending.update(
            [asyncio.create_task(fail()), asyncio.create_task(deliver())]
        )
        drain = asyncio.create_task(puts.drain())
        try:
            await failed.wait()
            await asyncio.sleep(0)
            self.assertFalse(drain.done())
            release.set()
            with self.assertLogs("spdl.pipeline._components._variants", level="ERROR"):
                with self.assertRaisesRegex(ValueError, "route failed"):
                    await drain
            self.assertTrue(delivered.is_set())
            self.assertFalse(puts.pending)
            self.assertFalse(puts.tails)
        finally:
            release.set()
            await asyncio.gather(drain, return_exceptions=True)
            await puts.cancel()

    async def test_overflow_preserves_real_failures_among_cancellations(self) -> None:
        """Cancellation cannot mask routing errors, and every failure is logged."""
        puts = _PendingPathPuts()

        async def fail(message: str) -> None:
            raise ValueError(message)

        tasks = [asyncio.create_task(fail(message)) for message in ("first", "second")]
        cancelled = asyncio.create_task(asyncio.sleep(10))
        cancelled.cancel()
        await asyncio.wait([*tasks, cancelled])
        puts.pending.update([*tasks, cancelled])
        for _ in range(28):
            task = asyncio.create_task(asyncio.sleep(0))
            await task
            puts.pending.add(task)
        queue = AsyncQueue(StageInfo(0, "blocked", "blocked"))
        queue.put_nowait(0)
        try:
            with self.assertLogs(
                "spdl.pipeline._components._variants", level="ERROR"
            ) as logs:
                with self.assertRaisesRegex(ValueError, "first|second"):
                    await puts.put(queue, 1)
            self.assertEqual(len(logs.records), 2)
            self.assertEqual(
                {
                    str(record.exc_info[1])
                    for record in logs.records
                    if record.exc_info is not None
                },
                {"first", "second"},
            )
        finally:
            await puts.cancel()

    def queues(self) -> tuple[AsyncQueue, list[AsyncQueue]]:
        source = AsyncQueue(StageInfo(0, "source", "source"), buffer_size=0)
        paths = [AsyncQueue(StageInfo(0, str(i), str(i))) for i in range(2)]
        return source, paths

    def start(
        self,
        source: AsyncQueue,
        paths: list[AsyncQueue],
        router: Callable[[tuple[int, int]], int] | None = None,
    ) -> asyncio.Task[None]:
        async def route(item: tuple[int, int]) -> int:
            return item[0] if router is None else router(item)

        task = asyncio.create_task(_path_variants_router(source, paths, route, []))
        self.addAsyncCleanup(self.stop, task)
        return task

    async def stop(self, task: asyncio.Task[None]) -> None:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def consume(self, queue: AsyncQueue) -> list[object]:
        rows: list[object] = []
        while (item := await queue.get()) is not _EOF:
            rows.append(item)
        return rows

    async def test_ready_path_progresses_with_other_path_full(self) -> None:
        """A full slow path does not prevent a later ready path from receiving data."""
        source, paths = self.queues()
        for i in range(12):
            source.put_nowait((0, i))
        source.put_nowait((1, 99))
        source.put_nowait(_EOF)
        task = self.start(source, paths)
        self.assertEqual(await asyncio.wait_for(paths[1].get(), 2), (1, 99))
        outputs = await asyncio.wait_for(
            asyncio.gather(*(self.consume(path) for path in paths)), 2
        )
        await task
        self.assertEqual(outputs, [[(0, i) for i in range(12)], []])

    async def test_per_path_fifo_under_backpressure(self) -> None:
        """Concurrent pending puts preserve each path's original input order."""
        source, paths = self.queues()
        for i in range(96):
            source.put_nowait((i % 2, i))
        source.put_nowait(_EOF)
        task = self.start(source, paths)
        outputs = await asyncio.wait_for(
            asyncio.gather(*(self.consume(path) for path in paths)), 2
        )
        await task
        self.assertEqual(outputs, [[(p, i) for i in range(p, 96, 2)] for p in range(2)])

    async def test_epoch_follows_all_preceding_items(self) -> None:
        """Every path observes its epoch boundary after all preceding routed items."""
        source, paths = self.queues()
        for epoch in range(2):
            for i in range(12):
                source.put_nowait((i % 2, epoch * 12 + i))
            source.put_nowait(_EPOCH_END)
        source.put_nowait(_EOF)
        task = self.start(source, paths)
        outputs = await asyncio.wait_for(
            asyncio.gather(*(self.consume(path) for path in paths)), 2
        )
        await task
        for p in range(2):
            self.assertEqual(
                outputs[p],
                [(p, i) for i in range(p, 12, 2)]
                + [_EPOCH_END]
                + [(p, i) for i in range(12 + p, 24, 2)]
                + [_EPOCH_END],
            )

    async def test_lookahead_is_bounded_and_cancellation_stops_paths(self) -> None:
        """A blocked path bounds source consumption and cancellation closes all paths."""
        source, paths = self.queues()
        routed: list[tuple[int, int]] = []
        reached = asyncio.Event()

        def route(item: tuple[int, int]) -> int:
            routed.append(item)
            if len(routed) == 40:
                reached.set()
            return item[0]

        for i in range(1000):
            source.put_nowait((0, i))
        task = self.start(source, paths, route)
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(reached.wait(), 0.05)
        self.assertLess(len(routed), 40)
        self.assertGreater(len(routed), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        self.assertEqual([path.get_nowait() for path in paths], [_EOF, _EOF])
        self.assertTrue(all(path.empty() for path in paths))
