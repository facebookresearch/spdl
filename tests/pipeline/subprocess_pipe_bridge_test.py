# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


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
from spdl.pipeline._components._common import _EOF, _EPOCH_END, StageInfo


def _unbounded_queue(name: str) -> AsyncQueue:
    """An AsyncQueue with no depth limit (the default is 1, which would block these tests)."""
    return AsyncQueue(
        StageInfo(pipeline_id=0, stage_id="0", stage_name=name), buffer_size=0
    )


def _make_input_queue(items: list[Any]) -> AsyncQueue:
    """A pre-filled stage input queue (unbounded, so the fill never blocks)."""
    q = _unbounded_queue("input")
    for item in items:
        q.put_nowait(item)
    return q


class _AlwaysOpenBarrier(asyncio.Event):
    """Epoch barrier stand-in that never blocks the feeder.

    ``_feed_continuous`` clears the barrier, broadcasts ``_EPOCH``, then waits for the collector
    to re-set it. These tests drive the feeder without a collector, so ``clear()`` is a no-op
    and ``wait()`` returns immediately -- letting the feeder run straight through the boundary.
    """

    def clear(self) -> None:
        pass


def _drain(q: "_queue.Queue[Any]") -> list[Any]:
    return [q.get_nowait() for _ in range(q.qsize())]


def _items_of(msgs: list[Any]) -> list[Any]:
    """Flatten the payloads of the ``_ITEM`` messages in ``msgs``, in order."""
    return [
        item
        for kind, payload in msgs
        if kind == _subprocess_pipe._ITEM
        for item in payload
    ]


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


class FeedBufferingTest(unittest.TestCase):
    """The feeder packs items into transfers of at most ``buffer_size``."""

    @staticmethod
    def _run_feed(
        items: list[Any], num_workers: int, buffer_size: int
    ) -> list[list[Any]]:
        async def _scenario() -> list[list[Any]]:
            in_qs: list[_queue.Queue[Any]] = [
                _queue.Queue() for _ in range(num_workers)
            ]
            input_queue = _make_input_queue([*items, _EOF])
            with ThreadPoolExecutor(max_workers=num_workers + 1) as ex:
                await asyncio.wait_for(
                    _subprocess_pipe._feed(
                        input_queue,
                        in_qs,
                        ex,
                        asyncio.Event(),
                        asyncio.Event(),
                        threading.Event(),
                        buffer_size,
                    ),
                    timeout=10.0,
                )
            return [_drain(q) for q in in_qs]

        return asyncio.run(_scenario())

    def test_packs_full_chunks_and_tail(self) -> None:
        """10 items at buffer_size=4 become transfers of 4, 4, 2 -- nothing dropped."""
        msgs = self._run_feed(list(range(10)), num_workers=1, buffer_size=4)
        payloads = [p for kind, p in msgs[0] if kind == _subprocess_pipe._ITEM]
        self.assertEqual(payloads, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])
        self.assertEqual(msgs[0][-1], (_subprocess_pipe._SESSION_END, None))

    def test_unbuffered_sends_singleton_lists(self) -> None:
        """buffer_size=1 (the default) still wraps each item in a one-element list."""
        msgs = self._run_feed([7, 8], num_workers=1, buffer_size=1)
        payloads = [p for kind, p in msgs[0] if kind == _subprocess_pipe._ITEM]
        self.assertEqual(payloads, [[7], [8]])

    def test_chunks_round_robin_across_workers(self) -> None:
        """Whole chunks, not individual items, are distributed across the workers."""
        msgs = self._run_feed(list(range(12)), num_workers=3, buffer_size=2)
        payloads = [
            [p for kind, p in worker if kind == _subprocess_pipe._ITEM]
            for worker in msgs
        ]
        self.assertEqual(payloads[0], [[0, 1], [6, 7]])
        self.assertEqual(payloads[1], [[2, 3], [8, 9]])
        self.assertEqual(payloads[2], [[4, 5], [10, 11]])

    def test_no_item_lost_when_count_not_divisible(self) -> None:
        """Across every worker, the items fed reproduce the input exactly."""
        items = list(range(23))
        msgs = self._run_feed(items, num_workers=4, buffer_size=5)
        fed = [item for worker in msgs for item in _items_of(worker)]
        self.assertCountEqual(fed, items)
        for worker in msgs:
            self.assertEqual(worker[-1], (_subprocess_pipe._SESSION_END, None))

    def test_empty_stream_sends_only_session_end(self) -> None:
        """No items means no transfer -- just the per-worker end markers."""
        msgs = self._run_feed([], num_workers=2, buffer_size=4)
        self.assertEqual(msgs, [[(_subprocess_pipe._SESSION_END, None)]] * 2)

    def test_abort_drops_partial_chunk(self) -> None:
        """An abort discards the partly-filled chunk but still ends every session.

        The feeder is given fewer items than ``buffer_size`` and no EOF, so it parks
        holding a partial chunk. Aborting must not flush it -- the pipeline is already failing
        -- but the ``_SESSION_END`` markers still have to go out, or the collector never sees
        every ``_DONE``.
        """

        async def _scenario() -> list[list[Any]]:
            in_qs: list[_queue.Queue[Any]] = [_queue.Queue() for _ in range(2)]
            input_queue = _make_input_queue([1, 2])  # no EOF; fewer than buffer_size
            abort = asyncio.Event()
            with ThreadPoolExecutor(max_workers=3) as ex:
                task = asyncio.ensure_future(
                    _subprocess_pipe._feed(
                        input_queue,
                        in_qs,
                        ex,
                        abort,
                        asyncio.Event(),
                        threading.Event(),
                        8,
                    )
                )
                await asyncio.sleep(0.1)  # let it consume both and park
                self.assertFalse(task.done())
                abort.set()
                await asyncio.wait_for(task, timeout=5.0)
            return [_drain(q) for q in in_qs]

        msgs = asyncio.run(_scenario())
        self.assertEqual(msgs, [[(_subprocess_pipe._SESSION_END, None)]] * 2)


class FeedContinuousBufferingTest(unittest.TestCase):
    """Chunking in continuous mode, where a chunk must not straddle an epoch boundary."""

    @staticmethod
    def _run_feed_continuous(
        items: list[Any], num_workers: int, buffer_size: int
    ) -> list[list[Any]]:
        async def _scenario() -> list[list[Any]]:
            in_qs: list[_queue.Queue[Any]] = [
                _queue.Queue() for _ in range(num_workers)
            ]
            input_queue = _make_input_queue([*items, _EOF])
            epoch_barrier = _AlwaysOpenBarrier()
            epoch_barrier.set()
            with ThreadPoolExecutor(max_workers=num_workers + 1) as ex:
                await asyncio.wait_for(
                    _subprocess_pipe._feed_continuous(
                        input_queue,
                        in_qs,
                        ex,
                        epoch_barrier,
                        asyncio.Event(),
                        threading.Event(),
                        buffer_size,
                    ),
                    timeout=10.0,
                )
            return [_drain(q) for q in in_qs]

        return asyncio.run(_scenario())

    def test_partial_chunk_flushed_before_epoch_marker(self) -> None:
        """The tail of an epoch is queued ahead of ``_EPOCH``, not merged into the next epoch.

        Five items at buffer_size=4 leave one buffered when the boundary arrives. If
        that item were sent after the ``_EPOCH`` marker the worker would drain it as part of
        epoch 1, silently moving data between epochs.
        """
        msgs = self._run_feed_continuous(
            [0, 1, 2, 3, 4, _EPOCH_END, 5, 6], num_workers=1, buffer_size=4
        )
        kinds = [kind for kind, _ in msgs[0]]
        epoch_at = kinds.index(_subprocess_pipe._EPOCH)
        before = _items_of(msgs[0][:epoch_at])
        after = _items_of(msgs[0][epoch_at:])
        self.assertEqual(before, [0, 1, 2, 3, 4])
        self.assertEqual(after, [5, 6])

    def test_partial_chunk_flushed_before_shutdown(self) -> None:
        """A partial chunk at end of stream is sent before ``_POOL_SHUTDOWN``."""
        msgs = self._run_feed_continuous([0, 1], num_workers=1, buffer_size=8)
        kinds = [kind for kind, _ in msgs[0]]
        shutdown_at = kinds.index(_subprocess_pipe._POOL_SHUTDOWN)
        self.assertEqual(_items_of(msgs[0][:shutdown_at]), [0, 1])

    def test_epoch_marker_broadcast_to_every_worker(self) -> None:
        """Every worker still receives the boundary, and no epoch's items leak across it."""
        msgs = self._run_feed_continuous(
            [*range(6), _EPOCH_END, *range(100, 106)],
            num_workers=3,
            buffer_size=2,
        )
        for worker in msgs:
            kinds = [kind for kind, _ in worker]
            self.assertIn(_subprocess_pipe._EPOCH, kinds)
            epoch_at = kinds.index(_subprocess_pipe._EPOCH)
            self.assertTrue(all(i < 100 for i in _items_of(worker[:epoch_at])))
            self.assertTrue(all(i >= 100 for i in _items_of(worker[epoch_at:])))


class CollectUnpackTest(unittest.TestCase):
    """The collector unpacks a transfer back into individual downstream items."""

    def test_collect_unpacks_in_order(self) -> None:
        """Items within a ``_RESULT`` payload reach the output queue individually, in order."""

        async def _scenario() -> list[Any]:
            out_q: _queue.Queue[Any] = _queue.Queue()
            out_q.put((_subprocess_pipe._RESULT, [1, 2, 3]))
            out_q.put((_subprocess_pipe._RESULT, [4]))
            out_q.put((_subprocess_pipe._DONE, None))
            output_queue = _unbounded_queue("output")
            with ThreadPoolExecutor(max_workers=2) as ex:
                await asyncio.wait_for(
                    _subprocess_pipe._collect(
                        out_q, 1, output_queue, ex, asyncio.Event(), asyncio.Event()
                    ),
                    timeout=10.0,
                )
            return [output_queue.get_nowait() for _ in range(output_queue.qsize())]

        self.assertEqual(asyncio.run(_scenario()), [1, 2, 3, 4])

    def test_collect_continuous_unpacks_in_order(self) -> None:
        """Same for the continuous collector, which also emits the epoch boundary."""

        async def _scenario() -> list[Any]:
            out_q: _queue.Queue[Any] = _queue.Queue()
            out_q.put((_subprocess_pipe._RESULT, [1, 2]))
            out_q.put((_subprocess_pipe._EPOCH_DONE, None))
            out_q.put((_subprocess_pipe._DONE, None))
            output_queue = _unbounded_queue("output")
            epoch_barrier = asyncio.Event()
            with ThreadPoolExecutor(max_workers=2) as ex:
                await asyncio.wait_for(
                    _subprocess_pipe._collect_continuous(
                        out_q, 1, output_queue, ex, epoch_barrier, asyncio.Event()
                    ),
                    timeout=10.0,
                )
            return [output_queue.get_nowait() for _ in range(output_queue.qsize())]

        self.assertEqual(asyncio.run(_scenario()), [1, 2, _EPOCH_END])

    def test_results_after_error_are_discarded(self) -> None:
        """A buffered transfer arriving after the first error is dropped wholesale."""

        async def _scenario() -> list[Any]:
            out_q: _queue.Queue[Any] = _queue.Queue()
            out_q.put((_subprocess_pipe._RESULT, [1, 2]))
            out_q.put((_subprocess_pipe._ERROR, RuntimeError("boom")))
            out_q.put((_subprocess_pipe._RESULT, [3, 4]))
            out_q.put((_subprocess_pipe._DONE, None))
            output_queue = _unbounded_queue("output")
            with ThreadPoolExecutor(max_workers=2) as ex:
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    await asyncio.wait_for(
                        _subprocess_pipe._collect(
                            out_q, 1, output_queue, ex, asyncio.Event(), asyncio.Event()
                        ),
                        timeout=10.0,
                    )
            return [output_queue.get_nowait() for _ in range(output_queue.qsize())]

        self.assertEqual(asyncio.run(_scenario()), [1, 2])


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
