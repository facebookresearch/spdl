# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import queue
import unittest
from unittest.mock import patch

from spdl.pipeline._components._common import StageInfo
from spdl.pipeline._components._thread_queue import ThreadOutputQueue


def _info() -> StageInfo:
    return StageInfo(pipeline_id=0, stage_id="0", stage_name="sink")


class ThreadOutputQueueTest(unittest.IsolatedAsyncioTestCase):
    async def test_buffered_put_avoids_executor(self) -> None:
        """A buffered thread sink accepts an item without scheduling executor work."""
        output = ThreadOutputQueue(_info(), buffer_size=1)
        loop = asyncio.get_running_loop()
        with patch.object(
            loop, "run_in_executor", side_effect=AssertionError("executor used")
        ):
            await output.put("item")
        self.assertEqual(output._queue.get_nowait(), "item")

    async def test_unbounded_queue_preserves_executor_yield(self) -> None:
        """An unbounded sink keeps the existing yielding producer path."""
        output = ThreadOutputQueue(_info(), buffer_size=0)
        loop = asyncio.get_running_loop()
        with patch.object(
            loop, "run_in_executor", wraps=loop.run_in_executor
        ) as submit:
            await output.put(1)
        self.assertEqual(submit.call_count, 1)
        self.assertEqual(output.get_nowait(), 1)

    async def test_full_queue_keeps_bounded_fallback(self) -> None:
        """A full sink keeps the existing executor-backed blocking put behavior."""
        output = ThreadOutputQueue(_info(), buffer_size=1)
        await output.put(1)
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        original = loop.run_in_executor

        def submit(executor, fn, *args):
            entered.set()
            return original(executor, fn, *args)

        with patch.object(loop, "run_in_executor", side_effect=submit):
            task = asyncio.create_task(output.put(2))
            await asyncio.wait_for(entered.wait(), 2)
            self.assertFalse(task.done())
            self.assertEqual(output._queue.get_nowait(), 1)
            await asyncio.wait_for(task, 2)
        self.assertEqual(output._queue.get_nowait(), 2)

    async def test_no_stats_keeps_plain_queue(self) -> None:
        """Disabling stats retains the original standard-library queue storage."""
        output = ThreadOutputQueue(_info(), buffer_size=1)
        self.assertIs(type(output._queue), queue.Queue)
        async with output.stage_hook():
            await output.put(1)
            self.assertEqual(output.get_nowait(), 1)
