# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for the subprocess-pipeline bridge internals (``_subprocess_pipe``).

These exercise the feeder/collector/stall-guard machinery that streams items to and from the
worker pool backing a ``.to()`` region. They are independent of how a region is expressed --
they drive ``_subprocess_pipe`` helpers directly -- so they live apart from the region tests.
Moved (unchanged) from the removed ``subprocess_pipeline_fuse_test.py``.
"""

import asyncio
import queue as _queue
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from spdl.pipeline import AsyncQueue
from spdl.pipeline._components import _subprocess_pipe
from spdl.pipeline._components._common import StageInfo


class FeedAbortTest(unittest.TestCase):
    """The bridge feeder must wind down promptly when the collector signals abort."""

    def test_feed_ends_session_when_aborted_while_idle(self) -> None:
        """A feeder parked on an empty input queue still emits the per-worker _SESSION_END.

        On a worker error the collector sets ``abort`` while the feeder is typically
        blocked waiting on a slow/idle upstream. The feeder must wake and send exactly one
        ``_SESSION_END`` onto each worker's own queue so the collector can drain every
        ``_DONE`` instead of hanging until its stall timeout. Driving ``_feed`` directly
        keeps the abort-while-idle race deterministic.
        """
        num_workers = 3

        async def _scenario() -> list[list[Any]]:
            in_qs: list[_queue.Queue[Any]] = [
                _queue.Queue() for _ in range(num_workers)
            ]
            input_queue = AsyncQueue(
                StageInfo(pipeline_id=0, stage_id="0", stage_name="input")
            )  # stays empty -> get() blocks
            abort = asyncio.Event()
            feeder_idle = asyncio.Event()
            put_stop = threading.Event()
            with ThreadPoolExecutor(max_workers=num_workers + 1) as ex:
                task = asyncio.ensure_future(
                    _subprocess_pipe._feed(
                        input_queue, in_qs, ex, abort, feeder_idle, put_stop
                    )
                )
                await asyncio.sleep(0.1)  # let the feeder park on input_queue.get()
                self.assertFalse(task.done(), "feeder should be parked on empty queue")
                abort.set()
                await asyncio.wait_for(task, timeout=5.0)
            return [[q.get_nowait() for _ in range(q.qsize())] for q in in_qs]

        msgs = asyncio.run(_scenario())
        # Every worker's own queue receives exactly one _SESSION_END.
        self.assertEqual(msgs, [[(_subprocess_pipe._SESSION_END, None)]] * num_workers)


class StallGuardTest(unittest.TestCase):
    """The collector's stall guard against an abruptly-dead worker."""

    def test_check_stall_raises_past_timeout(self) -> None:
        """``_check_stall`` raises once no message has arrived for longer than the bound."""
        orig = _subprocess_pipe._WORKER_STALL_TIMEOUT
        _subprocess_pipe._WORKER_STALL_TIMEOUT = 0.0
        try:
            with self.assertRaises(TimeoutError):
                _subprocess_pipe._check_stall(time.monotonic() - 1.0)
        finally:
            _subprocess_pipe._WORKER_STALL_TIMEOUT = orig

    def test_check_stall_quiet_within_timeout(self) -> None:
        """``_check_stall`` does not raise while progress is within the bound."""
        orig = _subprocess_pipe._WORKER_STALL_TIMEOUT
        _subprocess_pipe._WORKER_STALL_TIMEOUT = 60.0
        try:
            _subprocess_pipe._check_stall(time.monotonic())  # should not raise
        finally:
            _subprocess_pipe._WORKER_STALL_TIMEOUT = orig

    def test_collect_suppresses_stall_while_feeder_idle(self) -> None:
        """An idle feeder suppresses the collector's stall guard during input starvation.

        With the timeout pinned to zero, any stall check on an empty queue would trip instantly;
        the collector must instead keep draining while ``feeder_idle`` is set (nothing dispatched,
        no worker message due) and still finish once the worker reports ``_DONE``.
        """
        orig = _subprocess_pipe._WORKER_STALL_TIMEOUT
        _subprocess_pipe._WORKER_STALL_TIMEOUT = 0.0

        async def _scenario() -> None:
            out_q: _queue.Queue[Any] = _queue.Queue()
            output_queue = AsyncQueue(
                StageInfo(pipeline_id=0, stage_id="0", stage_name="output")
            )
            abort = asyncio.Event()
            feeder_idle = asyncio.Event()
            feeder_idle.set()  # feeder parked on an idle upstream -> no message expected
            with ThreadPoolExecutor(max_workers=2) as ex:
                task = asyncio.ensure_future(
                    _subprocess_pipe._collect(
                        out_q, 1, output_queue, ex, abort, feeder_idle
                    )
                )
                await asyncio.sleep(
                    0.6
                )  # several empty poll cycles; must not trip the guard
                self.assertFalse(
                    task.done(), "idle feeder must suppress the stall guard"
                )
                out_q.put((_subprocess_pipe._DONE, None))
                await asyncio.wait_for(task, timeout=5.0)

        try:
            asyncio.run(_scenario())
        finally:
            _subprocess_pipe._WORKER_STALL_TIMEOUT = orig


if __name__ == "__main__":
    unittest.main()
